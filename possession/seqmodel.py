"""The sequence model: a small temporal convolutional network over the whole match.

The boosted-tree model sees each frame with hand-made summaries of its surroundings. This
network instead reads the per-frame measurements as a sequence and learns its own summaries,
looking roughly 40 seconds either side of every frame. It follows the multi-stage design used
for action segmentation (MS-TCN): a first stage labels every frame, and later stages re-read
those labels and smooth them, which cuts the over-segmentation the ranking metric punishes.

Like the tree model it is trained only on other matches, and predicts twice -- as given and
with the teams swapped -- so it cannot favour one side.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import model as tree_model
from .match import H, A, D

CONFIG = {"channels": 64, "layers": 10, "stages": 3, "dropout": 0.2}


def frame_tensor(match, f, names=None):
    """([n_frames, n_features] float32, names) of per-frame measurements."""
    base = tree_model.base_features(match, f, shape=True)
    names = names or sorted(base)
    return np.column_stack([base[name] for name in names]).astype(np.float32), names


class _Block(nn.Module):
    def __init__(self, channels, dilation, dropout):
        super().__init__()
        self.dilated = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.mix = nn.Conv1d(channels, channels, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.drop(self.mix(F.relu(self.dilated(x))))


class _Stage(nn.Module):
    def __init__(self, n_in, channels, layers, dropout, n_out=3):
        super().__init__()
        self.enter = nn.Conv1d(n_in, channels, 1)
        self.blocks = nn.ModuleList(_Block(channels, 2 ** i, dropout) for i in range(layers))
        self.leave = nn.Conv1d(channels, n_out, 1)

    def forward(self, x):
        x = self.enter(x)
        for block in self.blocks:
            x = block(x)
        return self.leave(x)


class Net(nn.Module):
    def __init__(self, n_features, channels=64, layers=10, stages=3, dropout=0.2):
        super().__init__()
        self.first = _Stage(n_features, channels, layers, dropout)
        self.refine = nn.ModuleList(_Stage(3, channels, layers, dropout)
                                    for _ in range(stages - 1))

    def forward(self, x):
        """x: [batch, n_features, time]. Returns the logits of every stage, last one final."""
        out = [self.first(x)]
        for stage in self.refine:
            out.append(stage(F.softmax(out[-1], dim=1)))
        return out


def boundary_weights(target, boost, radius=25):
    """Per-frame weight: `boost` within `radius` frames of a change of label, 1 elsewhere."""
    graded = target != 3
    change = torch.zeros_like(target, dtype=torch.float32)
    change[:, 1:] = ((target[:, 1:] != target[:, :-1]) & graded[:, 1:] & graded[:, :-1]).float()
    near = F.max_pool1d(change[:, None], 2 * radius + 1, stride=1, padding=radius)[:, 0]
    return (1.0 + (boost - 1.0) * near) * graded.float()


def loss_fn(stage_logits, target, smooth=0.15, boost=1.0):
    """Cross-entropy on graded frames plus a penalty on jumpy frame-to-frame predictions.

    With boost > 1 the frames around every change of possession or of dead/alive count more:
    those are the frames that decide whether a possession is found, and they are few.
    """
    weight = boundary_weights(target, boost) if boost != 1.0 else None
    total = 0.0
    for logits in stage_logits:
        if weight is None:
            total = total + F.cross_entropy(logits, target, ignore_index=3)
        else:
            per = F.cross_entropy(logits, target.clamp(max=2), reduction="none")
            total = total + (per * weight).sum() / weight.sum().clamp(min=1.0)
        log_prob = F.log_softmax(logits, dim=1)
        jump = (log_prob[:, :, 1:] - log_prob[:, :, :-1].detach()) ** 2
        total = total + smooth * torch.clamp(jump, max=16.0).mean()
    return total


def save(path, net, names, mean, std, config=CONFIG):
    torch.save({"state": net.state_dict(), "names": names, "mean": mean, "std": std,
                "config": config}, path)


def load_all(path, tags=None):
    """Every model in a checkpoint as [(net, ckpt)]. A checkpoint holds one model or several
    ("members") whose predictions are averaged; `tags` keeps only the members so tagged."""
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    out = []
    for ckpt in loaded.get("members", [loaded]):
        if tags is not None and ckpt.get("tag") not in tags:
            continue
        net = Net(len(ckpt["names"]), **ckpt["config"])
        net.load_state_dict(ckpt["state"])
        net.eval()
        out.append((net, ckpt))
    return out


def load(path):
    return load_all(path)[0]


def predict_all(members, match, f, device="cpu"):
    """Mean of the members' probabilities."""
    total = None
    for net, ckpt in members:
        prob = predict(net, ckpt, match, f, device)
        total = prob if total is None else total + prob
    return total / len(members)


def _run(net, x, mean, std, device):
    scaled = (x - mean) / std
    with torch.no_grad():
        tensor = torch.from_numpy(scaled.T[None]).to(device)
        return F.softmax(net(tensor)[-1], dim=1)[0].T.cpu().numpy()


def predict(net, ckpt, match, f, device="cpu"):
    """[n_frames, 3] probabilities over H, A, D, averaged over both team orderings."""
    names, mean, std = ckpt["names"], ckpt["mean"], ckpt["std"]
    x, _ = frame_tensor(match, f, names)
    x_swapped, _ = frame_tensor(match, tree_model.swap_teams(f), names)
    prob = np.full((match.n_frames, 3), 1.0 / 3.0, dtype=np.float32)
    net = net.to(device)
    for start, stop in match.runs:
        p = _run(net, x[start:stop], mean, std, device)
        q = _run(net, x_swapped[start:stop], mean, std, device)
        prob[start:stop] = (p + q[:, [A, H, D]]) / 2.0
    return prob
