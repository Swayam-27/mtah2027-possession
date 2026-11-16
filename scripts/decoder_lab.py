"""Compare decoders on held-out probabilities (oof.npz). External matches only; CPU; scores only.

Each held-out match was predicted by networks that never saw it. Every decoder is scored on the
average of the training recipes (as the submission averages them), per match, under the
calibrated noise and under the stress noises. A decoder is worth adopting only if it beats the
current one on most matches and is not worse under the stress noises.

    python scripts/decoder_lab.py --oof outputs/oof.npz
"""

import argparse
import os
import sys
import time
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")
from possession import decode  # noqa: E402
from possession.match import H, A, D, X  # noqa: E402
from possession.scoring import score  # noqa: E402


def load(path):
    z = np.load(path)
    data = {}
    for key in z.files:
        parts = key.split("|")
        data.setdefault(parts[0], {})["|".join(parts[1:])] = z[key]
    return data


def viterbi_matrix(log_prob, team, dead_in, dead_out):
    """3-state Viterbi with separate prices: H<->A `team`, alive->D `dead_in`, D->alive `dead_out`."""
    n = len(log_prob)
    emit = log_prob.tolist()
    h, a, d = emit[0]
    back = []
    for t in range(1, n):
        e = emit[t]
        bh, ph = h, 0
        if a - team > bh:
            bh, ph = a - team, 1
        if d - dead_out > bh:
            bh, ph = d - dead_out, 2
        ba, pa = a, 1
        if h - team > ba:
            ba, pa = h - team, 0
        if d - dead_out > ba:
            ba, pa = d - dead_out, 2
        bd, pd = d, 2
        if h - dead_in > bd:
            bd, pd = h - dead_in, 0
        if a - dead_in > bd:
            bd, pd = a - dead_in, 1
        h, a, d = bh + e[0], ba + e[1], bd + e[2]
        back.append((ph, pa, pd))
    path = np.empty(n, dtype=np.int8)
    path[-1] = int(np.argmax([h, a, d]))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t - 1][path[t]]
    return path


def viterbi_min_team(log_prob, team, dead, k):
    """Viterbi where a team change is only allowed after the current team has held the ball for
    at least `k` frames (a change into or out of a dead ball is always allowed). States: H_1..H_k,
    A_1..A_k (frames held, capped at k), D."""
    n = len(log_prob)
    neg = -1e18
    hs = np.full(k, neg)
    as_ = np.full(k, neg)
    hs[k - 1], as_[k - 1] = log_prob[0, H], log_prob[0, A]
    d = log_prob[0, D]
    # Backpointers: for H_1 / A_1 / D, which state they came from; H_j (j>1) comes from H_{j-1}
    # except H_k, which may also come from H_k.
    bp_h1 = np.zeros(n, dtype=np.int8)   # 0: from A_k, 1: from D
    bp_a1 = np.zeros(n, dtype=np.int8)   # 0: from H_k, 1: from D
    bp_hk = np.zeros(n, dtype=bool)      # True: H_k stayed in H_k
    bp_ak = np.zeros(n, dtype=bool)
    bp_d = np.zeros((n, 2), dtype=np.int16)  # (kind, index): kind 0 D, 1 H_j, 2 A_j
    for t in range(1, n):
        eh, ea, ed = log_prob[t, H], log_prob[t, A], log_prob[t, D]
        new_h = np.empty(k)
        new_a = np.empty(k)
        # into H_1: from A_k (team change) or from D
        from_a, from_d = as_[k - 1] - team, d - dead
        new_h[0] = max(from_a, from_d)
        bp_h1[t] = 0 if from_a >= from_d else 1
        from_h, from_d2 = hs[k - 1] - team, d - dead
        new_a[0] = max(from_h, from_d2)
        bp_a1[t] = 0 if from_h >= from_d2 else 1
        if k > 1:
            new_h[1:] = hs[:-1]
            new_a[1:] = as_[:-1]
            stay_h, stay_a = hs[k - 1], as_[k - 1]
            bp_hk[t] = stay_h >= hs[k - 2]
            bp_ak[t] = stay_a >= as_[k - 2]
            new_h[k - 1] = max(stay_h, hs[k - 2])
            new_a[k - 1] = max(stay_a, as_[k - 2])
        else:
            # k == 1 reduces to the plain decoder: H_1 may also stay
            if hs[0] >= new_h[0]:
                new_h[0], bp_hk[t] = hs[0], True
            if as_[0] >= new_a[0]:
                new_a[0], bp_ak[t] = as_[0], True
        ih, ia = int(np.argmax(hs)), int(np.argmax(as_))
        cands = (d, hs[ih] - dead, as_[ia] - dead)
        c = int(np.argmax(cands))
        bp_d[t] = (c, (0, ih, ia)[c])
        hs, as_, d = new_h + eh, new_a + ea, cands[c] + ed
    # backtrack
    path = np.empty(n, dtype=np.int8)
    ih, ia = int(np.argmax(hs)), int(np.argmax(as_))
    best = int(np.argmax((d, hs[ih], as_[ia])))
    state = ("D", 0) if best == 0 else (("H", ih) if best == 1 else ("A", ia))
    for t in range(n - 1, -1, -1):
        kind, j = state
        path[t] = {"H": H, "A": A, "D": D}[kind]
        if t == 0:
            break
        if kind == "D":
            c, idx = bp_d[t]
            state = ("D", 0) if c == 0 else (("H", int(idx)) if c == 1 else ("A", int(idx)))
        elif kind in ("H", "A"):
            other = "A" if kind == "H" else "H"
            if j == 0 and not (k == 1 and (bp_hk[t] if kind == "H" else bp_ak[t])):
                came = bp_h1[t] if kind == "H" else bp_a1[t]
                state = (other, k - 1) if came == 0 else ("D", 0)
            elif j == k - 1 and k > 1 and (bp_hk[t] if kind == "H" else bp_ak[t]):
                state = (kind, k - 1)
            elif k == 1:
                state = (kind, 0)
            else:
                state = (kind, j - 1)
    return path


def run_decoder(fn, prob, runs):
    out = np.full(len(prob), X, dtype=np.int8)
    for a, b in runs:
        lp = np.log(np.clip(prob[a:b].astype(np.float64), 1e-6, 1.0))
        out[a:b] = fn(lp)
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof", default=os.path.join(ROOT, "outputs", "oof.npz"))
    parser.add_argument("--quick", action="store_true", help="calibrated noise only")
    args = parser.parse_args()
    data = load(args.oof)
    names = sorted(data)
    recipes = sorted({k.split("|")[0] for k in data[names[0]] if k.count("|") == 1})
    variants = sorted({k.split("|")[1] for k in data[names[0]] if k.count("|") == 1})
    if args.quick:
        variants = ["calibrated"]
    print("{} matches, recipes {}, noise {}".format(len(names), recipes, variants), flush=True)

    def plain(team, dead, round_log=None):
        def fn(lp):
            if round_log is not None:
                lp = np.round(lp, round_log)
            return decode.viterbi(lp, team, dead)
        return fn

    decoders = {
        "current 12/32": ("arith", plain(12.0, 32.0)),
        "current 12/32, rounded (submitted)": ("arith", plain(12.0, 32.0, 3)),
        "geometric mean of recipes, 12/32": ("geo", plain(12.0, 32.0)),
    }
    for din in (24.0, 32.0, 40.0):
        for dout in (16.0, 24.0, 32.0, 48.0):
            if (din, dout) != (32.0, 32.0):
                decoders["split dead prices in {:.0f} / out {:.0f}".format(din, dout)] = (
                    "arith", (lambda i, o: (lambda lp: viterbi_matrix(lp, 12.0, i, o)))(din, dout))
    for k, team in ((12, 8.0), (25, 8.0), (25, 6.0), (50, 6.0), (50, 4.0)):
        decoders["min {} frames before a team change, team {:.0f}".format(k, team)] = (
            "arith", (lambda kk, tt: (lambda lp: viterbi_min_team(lp, tt, 32.0, kk)))(k, team))

    results = {}
    started = time.time()
    for variant in variants:
        for name in names:
            d = data[name]
            probs = [d["{}|{}".format(r, variant)].astype(np.float32) for r in recipes]
            arith = sum(probs) / len(probs)
            geo = np.exp(sum(np.log(np.clip(p, 1e-6, 1)) for p in probs) / len(probs))
            geo = geo / geo.sum(axis=1, keepdims=True)
            runs = [tuple(r) for r in d["runs"]]
            for label, (kind, fn) in decoders.items():
                pred = run_decoder(fn, geo if kind == "geo" else arith, runs)
                s = score(d["y"], pred)
                results.setdefault((variant, label), []).append(s["spell"])
        print("  done {} ({:.0f} s)".format(variant, time.time() - started), flush=True)

    for variant in variants:
        base = np.array(results[(variant, "current 12/32")])
        print("\n{} noise: mean spellF1 over {} held-out matches (DFL 7 / Metrica 3)".format(
            variant, len(names)))
        rows = []
        for label in decoders:
            v = np.array(results[(variant, label)])
            diff = v - base
            rows.append((v.mean(), label, v[:7].mean(), v[7:].mean(), diff.mean(),
                         int((diff > 1e-9).sum()), int((diff < -1e-9).sum())))
        rows.sort(reverse=True)
        for mean, label, dfl, met, diff, up, down in rows:
            print("  {:<52} {:.4f}  (DFL {:.4f} / MET {:.4f})  {:+.4f}  better {} worse {}".format(
                label, mean, dfl, met, diff, up, down))
