"""Compare decoders on the sequence model's held-out probabilities (oof.npz from Kaggle).

No GPU needed: the probabilities were produced on Kaggle by models that never saw the match
they predict. Scores are means over held-out matches, DFL and Metrica reported separately.
Also reports how restarts are handled, split into goal-kick-like restarts (ball restarted
inside a goal area) and all others, using the clean ball position of the cached match.

    python scripts/tune_seq_decode.py --oof outputs/oof.npz
"""

import argparse
import os
import sys
import warnings
from itertools import product

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")
from possession import decode  # noqa: E402
from possession.match import Match, H, A, D  # noqa: E402
from possession.scoring import score  # noqa: E402


def load(path):
    z = np.load(path)
    data = {}
    for key in z.files:
        parts = key.split("|")
        data.setdefault(parts[0], {})["|".join(parts[1:])] = z[key]
    return data


def restarts(y, runs):
    out = []
    for a, b in runs:
        lab = y[a:b]
        idx = np.flatnonzero((lab[1:] != D) & (lab[1:] <= A) & (lab[:-1] == D)) + 1
        out += [a + i for i in idx]
    return np.array(out, dtype=np.int64)


def goal_area(match, frames):
    length, width = match.pitch
    x, y = match.bx[frames], match.by[frames]
    near_goal = (x < 6.5) | (x > length - 6.5)
    central = np.abs(y - width / 2.0) < 10.5
    return near_goal & central


def restart_report(pred, y, frames):
    """Share of restarts given to the right team 1 s later, and the predicted restart's offset."""
    right, offsets = [], []
    alive = (pred == H) | (pred == A)
    for f in frames:
        g = f + 25
        if g < len(pred):
            right.append(pred[g] == y[g])
        window = np.arange(max(0, f - 250), min(len(pred) - 1, f + 250))
        starts = window[(pred[window] == D) & alive[window + 1]] + 1
        if len(starts):
            offsets.append((starts[np.argmin(np.abs(starts - f))] - f) / 25.0)
    offsets = np.array(offsets)
    return (float(np.mean(right)) if right else float("nan"),
            float(np.median(np.abs(offsets))) if len(offsets) else float("nan"),
            float(np.mean(np.abs(offsets) < 1.0)) if len(offsets) else float("nan"))


def evaluate(data, model, variant, decoder):
    by_source = {}
    for name, d in data.items():
        key = "{}|{}".format(model, variant)
        if key not in d:
            continue
        pred = decoder(d[key].astype(np.float32), [tuple(r) for r in d["runs"]])
        s = score(d["y"], pred)
        by_source.setdefault(name.split("-")[0], []).append((s["spell"], s["n_pred"] if "n_pred" in s else 0))
    return {k: float(np.mean([v[0] for v in vals])) for k, vals in by_source.items()}


def viterbi(team, dead, dead_bias=1.0, shortest=0):
    def run(prob, runs):
        p = prob.copy()
        p[:, D] *= dead_bias
        p /= p.sum(axis=1, keepdims=True)
        out = decode.decode_match(p, runs, team, dead)
        if shortest:
            out = decode.drop_short(out, runs, shortest) if hasattr(decode, "drop_short") else out
        return out
    return run


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof", default=os.path.join(ROOT, "outputs", "oof.npz"))
    args = parser.parse_args()
    data = load(args.oof)
    # The average of all trained versions, per held-out match and noise variant.
    for d in data.values():
        groups = {}
        for key, value in list(d.items()):
            if key.count("|") == 1:
                groups.setdefault(key.split("|")[1], []).append(value.astype(np.float32))
        for variant, values in groups.items():
            d["average of all|" + variant] = sum(values) / len(values)
    models = sorted({k.split("|")[0] for d in data.values() for k in d if k.count("|") == 1})
    variants = sorted({k.split("|")[1] for d in data.values() for k in d if k.count("|") == 1})
    print("matches {}  models {}  noise variants {}".format(len(data), models, variants))

    print("\nDECODER GRID, calibrated noise (mean spellF1: DFL / Metrica)")
    for model in models:
        results = []
        for team, dead, bias in product((6, 9, 12, 16, 22), (16, 24, 32, 48, 64), (0.8, 1.0, 1.25)):
            r = evaluate(data, model, "calibrated", viterbi(team, dead, bias))
            results.append((r.get("DFL", 0.0), r.get("MET", 0.0), team, dead, bias))
        results.sort(reverse=True)
        base = [r for r in results if r[2:] == (12, 32, 1.0)][0]
        print("  {:<16} current 12/32: {:.4f} / {:.4f}".format(model, base[0], base[1]))
        for r in results[:5]:
            print("  {:<16} team {:>3} dead {:>3} dead-bias {:.2f}:  {:.4f} / {:.4f}".format(
                model, r[2], r[3], r[4], r[0], r[1]))

    print("\nRESTARTS (DFL, calibrated noise, viterbi 12/32): right team 1 s after, "
          "|offset| median s, within 1 s")
    for model in models:
        rows = {"goal-kick-like": [], "other": []}
        for name, d in data.items():
            key = "{}|calibrated".format(model)
            if key not in d or not name.startswith("DFL"):
                continue
            match = Match.load(os.path.join(ROOT, "cache", name + ".npz"))
            runs = [tuple(r) for r in d["runs"]]
            pred = decode.decode_match(d[key].astype(np.float32), runs, 12.0, 32.0)
            frames = restarts(d["y"], runs)
            ga = goal_area(match, frames)
            rows["goal-kick-like"].append(restart_report(pred, d["y"], frames[ga]) + (int(ga.sum()),))
            rows["other"].append(restart_report(pred, d["y"], frames[~ga]) + (int((~ga).sum()),))
        for kind, vals in rows.items():
            vals = np.array(vals)
            print("  {:<16} {:<15} n {:4d}   right team {:.3f}   offset {:.2f} s   within 1 s {:.3f}".format(
                model, kind, int(vals[:, 3].sum()), *np.nanmean(vals[:, :3], axis=0)))
