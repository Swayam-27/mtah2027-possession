"""Train the sequence model on the external matches. Meant for a GPU (e.g. a Kaggle notebook).

Leave-one-match-out first, so the score printed is from matches the model never saw and can be
compared with the tree model's; then one final model trained on every match.

    python scripts/train_seq.py                                  # full run, GPU recommended
    python scripts/train_seq.py --copies 1 --cv-steps 20 --steps 20 --dfl 1 --metrica 1 --no-stress   # smoke test
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
WIDE = {"channels": 96, "layers": 11, "stages": 3, "dropout": 0.2}
DECODERS = ((2.0, 2.0), (4.0, 8.0), (8.0, 16.0), (12.0, 32.0), (16.0, 48.0))
# Held-out matches are scored under the calibrated noise and under a few others, to see how
# much the score depends on the noise guess being right.
STRESS = {
    "calibrated": Noise(),
    "ball unseen 55%": Noise(ball_missing=0.55),
    "short gaps, jitter": Noise(ball_mean_gap=40.0, ball_fill=0, ball_smooth=1, ball_jitter=0.15,
                                ball_drift=0.0, player_mean_gap=30.0, player_fill=0,
                                player_smooth=1, player_jitter=0.10, dead_hidden=0.25,
                                cam_height=0.0, air_lost=99.0, out_seen=1.0),
}


def prepare(job):
    """One degraded copy of one match: frame tensors for both team orderings, labels, runs."""
    path, seed, variant = job
    default_noise = variant is not None
    match = Match.load(path)
    match.labels = relabel.controlled(match.labels, match.runs, MIN_CONTROL)
    truth = graded_truth(match)
    noise = STRESS[variant] if default_noise else random_noise(np.random.default_rng(1000 + seed))
    noisy = degrade(match, noise, seed=seed)
    f = features.compute(noisy)
    x, names = seqmodel.frame_tensor(noisy, f)
    x_swapped, _ = seqmodel.frame_tensor(noisy, tree_model.swap_teams(f), names)
    return {"x": x, "x_swapped": x_swapped, "y": truth.astype(np.int64), "runs": noisy.runs,
            "names": names, "match": noisy if default_noise else None,
            "features": f if default_noise else None}


def batches(copies, mean, std, rng, device, cols):
    """Endless random crops, half of them with the teams swapped."""
    while True:
        xs, ys = [], []
        for _ in range(BATCH):
            item = copies[rng.integers(len(copies))]
            start, stop = item["runs"][rng.integers(len(item["runs"]))]
            a = int(rng.integers(start, max(start + 1, stop - CROP)))
            b = min(stop, a + CROP)
            swap = rng.random() < 0.5
            x = (item["x_swapped" if swap else "x"][a:b][:, cols] - mean) / std
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


def train(copies, steps, device, seed, label, config=None, cols=None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    cols = np.arange(copies[0]["x"].shape[1]) if cols is None else np.asarray(cols)
    sample = np.vstack([c["x"][::50][:, cols] for c in copies])
    mean = sample.mean(axis=0).astype(np.float32)
    std = (sample.std(axis=0) + 1e-3).astype(np.float32)

    net = seqmodel.Net(len(mean), **(config or seqmodel.CONFIG)).to(device)
    optimiser = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=1e-3, total_steps=steps)
    net.train()
    started, running = time.time(), 0.0
    for step, (x, y) in enumerate(batches(copies, mean, std, rng, device, cols), start=1):
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
    parser.add_argument("--dfl", type=int, default=0, help="use only the first N DFL matches (testing)")
    parser.add_argument("--metrica", type=int, default=3, help="Metrica matches to use")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-stress", action="store_true", help="calibrated noise only")
    parser.add_argument("--seeds", type=int, default=3, help="models in the final file")
    parser.add_argument("--out", default=os.path.join(ROOT, "models", "sequence.pt"))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dfl = sorted(glob.glob(os.path.join(ROOT, "cache", "DFL-*.npz")))
    metrica = sorted(glob.glob(os.path.join(ROOT, "cache", "MET-*.npz")))
    paths = dfl[:args.dfl or None] + metrica[:args.metrica]
    if not dfl:
        raise SystemExit("no cached external matches; run scripts/prepare_external.py first")
    source = ["DFL" if os.path.basename(p).startswith("DFL") else "Metrica" for p in paths]
    print("device {}, {} DFL + {} Metrica matches, {} copies each".format(
        device, source.count("DFL"), source.count("Metrica"), args.copies), flush=True)

    started = time.time()
    variants = list(STRESS)[:1 if args.no_stress else None]
    train_jobs = [(p, 100 * c + i, None) for i, p in enumerate(paths) for c in range(args.copies)]
    test_jobs = [(p, 500 + i, v) for i, p in enumerate(paths) for v in variants]
    with Pool(args.workers) as pool:
        prepared = pool.map(prepare, train_jobs + test_jobs, chunksize=1)
    train_data, test_data = prepared[:len(train_jobs)], prepared[len(train_jobs):]
    by_match = [train_data[i * args.copies:(i + 1) * args.copies] for i in range(len(paths))]
    names = train_data[0]["names"]
    print("prepared {} training copies, {} features ({:.0f} s)".format(
        len(train_data), len(names), time.time() - started), flush=True)

    # The comparison: the same network trained on DFL matches only, and on DFL plus Metrica.
    # Every match is held out in turn; the held-out match is never in either training set.
    plan = [("DFL only", {"DFL"}), ("DFL + Metrica", {"DFL", "Metrica"})]

    def spell(prob, item, team=12.0, dead=32.0):
        pred = decode.decode_match(prob, item["runs"], team, dead)
        return score(item["y"].astype(np.int8), pred)

    totals, stress = {}, {}
    for k in range(len(paths)):
        print("held out {} ({})".format(os.path.basename(paths[k]), source[k]), flush=True)
        for tag, allowed in plan:
            copies = [c for i, group in enumerate(by_match)
                      if i != k and source[i] in allowed for c in group]
            if not copies:
                continue
            net, mean, std = train(copies, args.cv_steps, device, seed=k,
                                   label="fold {} {}".format(k, tag))
            ckpt = {"names": names, "mean": mean, "std": std}
            for j, variant in enumerate(variants):
                item = test_data[k * len(variants) + j]
                prob = seqmodel.predict(net, ckpt, item["match"], item["features"], device)
                s = spell(prob, item)
                key = "{:<8} test, {:<14} {}".format(source[k], tag, variant)
                if j == 0:
                    print("  " + line(tag, s), flush=True)
                    totals.setdefault(key, []).append(s["spell"])
                else:
                    stress.setdefault(key, []).append(s["spell"])
            del net

    def table(title, rows):
        print()
        print(title)
        for name, values in sorted(rows.items()):
            print("  {:<52} {:.4f}  (min {:.4f}, max {:.4f}, n {})".format(
                name, float(np.mean(values)), float(np.min(values)), float(np.max(values)),
                len(values)))

    if totals:
        table("mean spellF1 over held-out matches, calibrated noise (viterbi 12/32)", totals)
    if stress:
        table("same models under other noise (viterbi 12/32)", stress)

    # Final models on every match, DFL and Metrica together. If the comparison above says
    # Metrica does not help, keep the previous model file instead of this one.
    members = []
    for seed in range(args.seeds):
        net, mean, std = train(train_data, args.steps, device, seed=99 + seed,
                               label="final {}".format(seed))
        members.append({"state": net.cpu().state_dict(), "names": names, "mean": mean,
                        "std": std, "config": seqmodel.CONFIG, "tag": "shape"})
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({"members": members}, args.out)
    print()
    print("saved {} with {} models trained on DFL + Metrica ({:.0f} s total)".format(
        args.out, len(members), time.time() - started))
