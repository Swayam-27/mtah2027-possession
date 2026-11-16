"""Does self-training on the test match help? Tested on external matches only.

The competition match has no answers, but the rules allow pseudo-labels made by an algorithm.
The idea: label the match with the model's own decoded output, keep the frames it is sure of,
and fine-tune on them together with the training matches, so the network adapts to the test
match's own tracking. Whether that helps or only reinforces the model's mistakes is an empirical
question, so it is answered here first, on held-out external matches with real answers:

  for each held-out DFL / Metrica match k:
    base   = network trained on every other match                 -> score on k
    pseudo = base's Viterbi labels on k (frames below `--conf` ignored)
    tuned  = base fine-tuned on the other matches + k with pseudo   -> score on k

Only `tuned` vs `base` on the same held-out match matters. Prints scores only.

    python scripts/selftrain_check.py                       # GPU recommended
    python scripts/selftrain_check.py --dfl 1 --metrica 1 --steps 30 --tune-steps 20 --copies 1   # smoke test
"""

import argparse
import glob
import os
import sys
import time
import warnings
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
warnings.filterwarnings("ignore")
import torch  # noqa: E402

from possession import decode, seqmodel  # noqa: E402
from possession.scoring import score  # noqa: E402
from train_seq import BATCH, CROP, SWAP, prepare, train  # noqa: E402

TEAM, DEAD = 12.0, 32.0


def pseudo_item(test_item, prob, conf):
    """The held-out copy with the model's own decoded labels; unsure frames ignored (3)."""
    y = decode.decode_match(prob, test_item["runs"], TEAM, DEAD).astype(np.int64)
    sure = prob.max(axis=1) >= conf
    y[~sure] = 3
    inside = np.zeros(len(y), dtype=bool)
    for a, b in test_item["runs"]:
        inside[a:b] = True
    y[~inside] = 3
    item = {"x": test_item["x"], "x_swapped": test_item["x_swapped"], "y": y,
            "runs": test_item["runs"]}
    return item, float(sure[inside].mean())


def mixed_batches(copies, pseudo, share, mean, std, rng, device):
    """Like train_seq.batches, but a `share` of the crops come from the pseudo-labelled match."""
    while True:
        xs, ys = [], []
        for _ in range(BATCH):
            item = pseudo if rng.random() < share else copies[rng.integers(len(copies))]
            start, stop = item["runs"][rng.integers(len(item["runs"]))]
            a = int(rng.integers(start, max(start + 1, stop - CROP)))
            b = min(stop, a + CROP)
            swap = rng.random() < 0.5
            x = (item["x_swapped" if swap else "x"][a:b].astype(np.float32) - mean) / std
            y = item["y"][a:b]
            if swap:
                y = SWAP[y]
            if b - a < CROP:
                x = np.vstack([x, np.zeros((CROP - (b - a), x.shape[1]), dtype=np.float32)])
                y = np.concatenate([y, np.full(CROP - (b - a), 3, dtype=np.int64)])
            xs.append(x.T)
            ys.append(y)
        yield (torch.from_numpy(np.stack(xs)).to(device),
               torch.from_numpy(np.stack(ys)).to(device))


def finetune(net, mean, std, copies, pseudo, share, steps, lr, seed, device):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net.train()
    optimiser = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=steps)
    for step, (x, y) in enumerate(mixed_batches(copies, pseudo, share, mean, std, rng, device),
                                  start=1):
        loss = seqmodel.loss_fn(net(x), y)
        optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimiser.step()
        schedule.step()
        if step >= steps:
            break
    net.eval()
    return net


def predict(net, mean, std, names, item, device):
    return seqmodel.predict(net, {"names": names, "mean": mean, "std": std}, item["match"],
                            item["features"], device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--copies", type=int, default=4)
    parser.add_argument("--steps", type=int, default=3000, help="base training steps")
    parser.add_argument("--tune-steps", type=int, default=800)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--share", type=float, default=0.25, help="crops from the pseudo match")
    parser.add_argument("--conf", default="0.0,0.9", help="confidence thresholds to compare")
    parser.add_argument("--dfl", type=int, default=0)
    parser.add_argument("--metrica", type=int, default=3)
    parser.add_argument("--pff", type=int, default=0)
    parser.add_argument("--pff-copies", type=int, default=2)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dfl = sorted(glob.glob(os.path.join(ROOT, "cache", "DFL-*.npz")))[:args.dfl or None]
    met = sorted(glob.glob(os.path.join(ROOT, "cache", "MET-*.npz")))[:args.metrica]
    pff = sorted(glob.glob(os.path.join(ROOT, "cache", "PFF-*.npz")))[:args.pff]
    paths = dfl + met + pff
    source = [os.path.basename(p)[:3] for p in paths]
    tested = [i for i, s in enumerate(source) if s != "PFF"]
    count = [args.pff_copies if s == "PFF" else args.copies for s in source]
    confs = [float(c) for c in args.conf.split(",")]
    print("device {}; {} DFL, {} Metrica, {} PFF; base {} steps, tune {} steps, lr {}, share {}"
          .format(device, len(dfl), len(met), len(pff), args.steps, args.tune_steps, args.lr,
                  args.share), flush=True)

    started = time.time()
    train_jobs = [(p, 100 * c + i, None, 0.0) for i, p in enumerate(paths) for c in range(count[i])]
    test_jobs = [(paths[i], 500 + i, "calibrated", 0.0) for i in tested]
    with Pool(args.workers) as pool:
        prepared = pool.map(prepare, train_jobs + test_jobs, chunksize=1)
    train_data, test_data = prepared[:len(train_jobs)], prepared[len(train_jobs):]
    offsets = np.concatenate(([0], np.cumsum(count)))
    by_match = [train_data[offsets[i]:offsets[i + 1]] for i in range(len(paths))]
    names = train_data[0]["names"]
    print("prepared {} copies ({:.0f} s)".format(len(train_data), time.time() - started), flush=True)

    rows = []
    for t, k in enumerate(tested):
        name = os.path.basename(paths[k])[:-4]
        item = test_data[t]
        copies = [c for i, group in enumerate(by_match) if i != k for c in group]
        net, mean, std = train(copies, args.steps, device, seed=k, label="fold {}".format(k))
        base_state = {key: v.detach().clone() for key, v in net.state_dict().items()}
        prob = predict(net, mean, std, names, item, device)
        base = score(item["y"].astype(np.int8), decode.decode_match(prob, item["runs"], TEAM, DEAD))
        row = [name, base["spell"]]
        print("{}  base spell {:.4f}".format(name, base["spell"]), flush=True)
        for conf in confs:
            net.load_state_dict(base_state)
            pseudo, kept = pseudo_item(item, prob, conf)
            net = finetune(net, mean, std, copies, pseudo, args.share, args.tune_steps, args.lr,
                           seed=1000 + k, device=device)
            tuned_prob = predict(net, mean, std, names, item, device)
            tuned = score(item["y"].astype(np.int8),
                          decode.decode_match(tuned_prob, item["runs"], TEAM, DEAD))
            row.append(tuned["spell"])
            print("   conf {:.2f} (keeps {:.0%} of frames): tuned spell {:.4f}  ({:+.4f})".format(
                conf, kept, tuned["spell"], tuned["spell"] - base["spell"]), flush=True)
        # Control: the same fine-tuning without the pseudo match, so a gain from simply
        # training longer is not mistaken for a gain from self-training.
        net.load_state_dict(base_state)
        net = finetune(net, mean, std, copies, None, 0.0, args.tune_steps, args.lr,
                       seed=1000 + k, device=device)
        control_prob = predict(net, mean, std, names, item, device)
        control = score(item["y"].astype(np.int8),
                        decode.decode_match(control_prob, item["runs"], TEAM, DEAD))
        row.append(control["spell"])
        print("   control (same fine-tune, no pseudo match): {:.4f}  ({:+.4f})".format(
            control["spell"], control["spell"] - base["spell"]), flush=True)
        rows.append(row)
        del net
        torch.cuda.empty_cache() if device == "cuda" else None

    table = np.array([r[1:] for r in rows], dtype=float)
    print("\nSUMMARY over {} held-out matches ({:.0f} min)".format(len(rows),
                                                                   (time.time() - started) / 60))
    print("  base                     mean {:.4f}".format(table[:, 0].mean()))
    labels = ["self-train conf {:.2f}".format(c) for c in confs] + ["control (no pseudo)"]
    for j, label in enumerate(labels, start=1):
        diff = table[:, j] - table[:, 0]
        print("  {:<24} mean {:.4f}  diff {:+.4f}  better on {} of {}".format(
            label, table[:, j].mean(), diff.mean(), int((diff > 0).sum()), len(diff)))
    for source_name in ("DFL", "MET"):
        idx = [i for i, r in enumerate(rows) if r[0].startswith(source_name)]
        if idx:
            sub = table[idx]
            print("  {}: base {:.4f} | ".format(source_name, sub[:, 0].mean()) + " | ".join(
                "{} {:.4f}".format(l, v) for l, v in zip(labels, sub[:, 1:].mean(axis=0))))
