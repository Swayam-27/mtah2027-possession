"""Train the sequence model on the external matches. Meant for a GPU (e.g. a Kaggle notebook).

Leave-one-match-out first, so the score printed is from matches the model never saw and can be
compared with the tree model's; then one final model trained on every match.

    python scripts/train_seq.py                                  # full run, GPU recommended
    python scripts/train_seq.py --copies 1 --cv-steps 20 --steps 20 --folds 1 --matches 2   # smoke test
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
warnings.filterwarnings("ignore")
import torch  # noqa: E402

from possession import decode, features, model as tree_model, relabel, seqmodel  # noqa: E402
from possession.degrade import Noise, degrade, random_noise  # noqa: E402
from possession.match import Match, H, A, D  # noqa: E402
from possession.scoring import graded_truth, line, score  # noqa: E402

MIN_CONTROL = 50
CROP = 4096
BATCH = 8
SWAP = np.array([A, H, D, 3], dtype=np.int64)


def prepare(job):
    """One degraded copy of one match: frame tensors for both team orderings, labels, runs."""
    path, seed, default_noise = job
    match = Match.load(path)
    match.labels = relabel.controlled(match.labels, match.runs, MIN_CONTROL)
    truth = graded_truth(match)
    noise = Noise() if default_noise else random_noise(np.random.default_rng(1000 + seed))
    noisy = degrade(match, noise, seed=seed)
    f = features.compute(noisy)
    x, names = seqmodel.frame_tensor(noisy, f)
    x_swapped, _ = seqmodel.frame_tensor(noisy, tree_model.swap_teams(f), names)
    return {"x": x, "x_swapped": x_swapped, "y": truth.astype(np.int64), "runs": noisy.runs,
            "names": names, "match": noisy if default_noise else None,
            "features": f if default_noise else None}


def batches(copies, mean, std, rng, device):
    """Endless random crops, half of them with the teams swapped."""
    while True:
        xs, ys = [], []
        for _ in range(BATCH):
            item = copies[rng.integers(len(copies))]
            start, stop = item["runs"][rng.integers(len(item["runs"]))]
            a = int(rng.integers(start, max(start + 1, stop - CROP)))
            b = min(stop, a + CROP)
            swap = rng.random() < 0.5
            x = (item["x_swapped" if swap else "x"][a:b] - mean) / std
            y = item["y"][a:b]
            if swap:
                y = SWAP[y]
            if b - a < CROP:  # a short half: pad, with padded frames ungraded
                x = np.vstack([x, np.zeros((CROP - (b - a), x.shape[1]), dtype=np.float32)])
                y = np.concatenate([y, np.full(CROP - (b - a), 3, dtype=np.int64)])
            xs.append(x.T)
            ys.append(y)
        yield (torch.from_numpy(np.stack(xs)).to(device),
               torch.from_numpy(np.stack(ys)).to(device))


def train(copies, steps, device, seed, label):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    sample = np.vstack([c["x"][::50] for c in copies])
    mean = sample.mean(axis=0).astype(np.float32)
    std = (sample.std(axis=0) + 1e-3).astype(np.float32)

    net = seqmodel.Net(len(mean), **seqmodel.CONFIG).to(device)
    optimiser = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=1e-3, total_steps=steps)
    net.train()
    started, running = time.time(), 0.0
    for step, (x, y) in enumerate(batches(copies, mean, std, rng, device), start=1):
        loss = seqmodel.loss_fn(net(x), y)
        optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimiser.step()
        schedule.step()
        running += float(loss)
        if step % max(1, steps // 10) == 0 or step == steps:
            every = max(1, steps // 10)
            print("  {} step {}/{}  loss {:.4f}  ({:.0f} s)".format(
                label, step, steps, running / every, time.time() - started), flush=True)
            running = 0.0
        if step >= steps:
            break
    net.eval()
    return net, mean, std


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--copies", type=int, default=6, help="degraded copies per match")
    parser.add_argument("--steps", type=int, default=6000, help="steps for the final model")
    parser.add_argument("--cv-steps", type=int, default=3000, help="steps per held-out test")
    parser.add_argument("--folds", type=int, default=7)
    parser.add_argument("--matches", type=int, default=0, help="use only the first N (testing)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", default=os.path.join(ROOT, "models", "sequence.pt"))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    paths = sorted(glob.glob(os.path.join(ROOT, "cache", "DFL-*.npz")))[:args.matches or None]
    if not paths:
        raise SystemExit("no cached external matches; run scripts/prepare_external.py first")
    print("device {}, {} matches, {} copies each".format(device, len(paths), args.copies),
          flush=True)

    started = time.time()
    train_jobs = [(p, 100 * c + i, False) for i, p in enumerate(paths) for c in range(args.copies)]
    test_jobs = [(p, 500 + i, True) for i, p in enumerate(paths)][:args.folds]
    with Pool(args.workers) as pool:
        prepared = pool.map(prepare, train_jobs + test_jobs, chunksize=1)
    train_data, test_data = prepared[:len(train_jobs)], prepared[len(train_jobs):]
    by_match = [train_data[i * args.copies:(i + 1) * args.copies] for i in range(len(paths))]
    names = train_data[0]["names"]
    print("prepared {} training copies, {} features ({:.0f} s)".format(
        len(train_data), len(names), time.time() - started), flush=True)

    totals = {}
    for k in range(min(args.folds, len(paths))):
        copies = [c for i, group in enumerate(by_match) if i != k for c in group]
        net, mean, std = train(copies, args.cv_steps, device, seed=k, label="fold {}".format(k))
        test = test_data[k]
        ckpt = {"names": names, "mean": mean, "std": std}
        prob = seqmodel.predict(net, ckpt, test["match"], test["features"], device)
        print("held out {}".format(os.path.basename(paths[k])), flush=True)
        argmax = np.full(len(prob), 3, dtype=np.int8)
        for a, b in test["runs"]:
            argmax[a:b] = prob[a:b].argmax(axis=1)
        results = {"sequence argmax": argmax}
        for team, dead in ((2.0, 2.0), (4.0, 8.0), (8.0, 16.0), (12.0, 32.0)):
            results["sequence viterbi {:g}/{:g}".format(team, dead)] = decode.decode_match(
                prob, test["runs"], team, dead)
        for name, pred in results.items():
            s = score(test["y"].astype(np.int8), pred)
            print("  " + line(name, s), flush=True)
            totals.setdefault(name, []).append(s["spell"])

    if totals:
        print("\nmean spellF1 over {} held-out matches (tree model: 0.68)".format(
            len(next(iter(totals.values())))))
        for name, values in totals.items():
            print("  {:<32} {:.4f}  (min {:.4f}, max {:.4f})".format(
                name, float(np.mean(values)), float(np.min(values)), float(np.max(values))))

    net, mean, std = train(train_data, args.steps, device, seed=99, label="final")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    seqmodel.save(args.out, net.cpu(), names, mean, std)
    print("\nsaved {} ({:.0f} s total)".format(args.out, time.time() - started))
