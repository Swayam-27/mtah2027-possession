"""Leave-one-match-out test of the learned model on the degraded external matches.

For each held-out match: train on the others, predict the held-out one, decode, score.
The rule engine and the baseline are scored on the same match for comparison.

    python scripts/cv.py --folds 1          # quick: hold out the first match only
    python scripts/cv.py                    # all folds
"""

import argparse
import glob
import os
import sys
import time
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")
import lightgbm as lgb  # noqa: E402

from possession import baseline, decode, engine, features, model, relabel  # noqa: E402
from possession.degrade import Noise, degrade, random_noise  # noqa: E402
from possession.match import Match, H, A, D  # noqa: E402
from possession.scoring import graded_truth, line, score  # noqa: E402

CACHE = os.path.join(ROOT, "cache")
MIN_CONTROL = 50
STRIDE = 4
PARAMS = {
    "objective": "multiclass", "num_class": 3, "learning_rate": 0.05, "num_leaves": 63,
    "min_data_in_leaf": 200, "feature_fraction": 0.5, "bagging_fraction": 0.8, "bagging_freq": 1,
    "lambda_l2": 1.0, "seed": 0, "deterministic": True, "force_row_wise": True,
    "num_threads": 8, "verbose": -1,
}
ROUNDS = 300


def prepare(path, seed=0, min_control=MIN_CONTROL, noise=None):
    """(degraded match, truth, features). noise=None draws a random setting from the seed."""
    match = Match.load(path)
    match.labels = relabel.controlled(match.labels, match.runs, min_control)
    truth = graded_truth(match)
    noise = noise or random_noise(np.random.default_rng(1000 + seed))
    noisy = degrade(match, noise, seed=seed)
    return noisy, truth, features.compute(noisy)


# Held-out matches are scored under several noise settings, including ones training never saw
# the like of, to see how much the score depends on the noise guess being right.
TEST_NOISE = {
    "default": Noise(),
    "ball 55% unseen": Noise(ball_missing=0.55, ball_mean_gap=80.0),
    "dead ball never seen": Noise(dead_hidden=1.0),
    "dead ball always shown": Noise(dead_hidden=0.0, dead_factor=1.0),
}


def training_rows(noisy, truth, f, stride=STRIDE):
    """Rows for one match, both team orderings, every `stride`-th graded frame."""
    rows = np.concatenate([np.arange(a, b) for a, b in noisy.runs])[::stride]
    x, names = model.matrix(noisy, f)
    x_swapped, _ = model.matrix(noisy, model.swap_teams(f))
    y = truth[rows].astype(np.int64)
    y_swapped = np.where(y == H, A, np.where(y == A, H, D))
    change = model.boundary_target(truth, noisy.runs)[rows]
    return (np.vstack([x[rows], x_swapped[rows]]), np.concatenate([y, y_swapped]), names,
            np.concatenate([change, change]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=0)
    parser.add_argument("--quick", action="store_true", help="score the default noise only")
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(CACHE, "DFL-*.npz")))
    started = time.time()
    rows = [training_rows(*prepare(p, seed=i)) for i, p in enumerate(paths)]
    names = rows[0][2]
    print("prepared {} matches, {} features, {:.0f} s".format(len(paths), len(names),
                                                            time.time() - started))

    folds = range(len(paths))[:args.folds or None]
    totals = {}
    for k in folds:
        x = np.vstack([rows[i][0] for i in range(len(paths)) if i != k])
        y = np.concatenate([rows[i][1] for i in range(len(paths)) if i != k])
        change = np.concatenate([rows[i][3] for i in range(len(paths)) if i != k])
        booster = lgb.train(PARAMS, lgb.Dataset(x, y, feature_name=names), ROUNDS)
        boundary_booster = lgb.train(PARAMS, lgb.Dataset(x, change, feature_name=names), ROUNDS)
        del x, y, change
        print("\nheld out {}   ({:.0f} s)".format(os.path.basename(paths[k]), time.time() - started))

        for tag, noise in list(TEST_NOISE.items())[:1 if args.quick else None]:
            noisy, truth, f = prepare(paths[k], seed=500 + k, noise=noise)
            prob, change_prob = model.predict(booster, noisy, f, boundary_booster=boundary_booster)
            if tag == "default":
                # Held-out predictions, kept so decoding ideas can be tried without retraining.
                np.savez_compressed(
                    os.path.join(CACHE, "oof-{}".format(os.path.basename(paths[k]))),
                    prob=prob.astype(np.float32), change=change_prob.astype(np.float32),
                    truth=truth, runs=np.array(noisy.runs))
            results = {"rules": engine.label_match(noisy, f)}
            if tag == "default":
                results["baseline_smooth50"] = baseline.label_match(noisy, 50)
                costs = (4.0, 8.0, 16.0, 32.0)
            else:
                costs = (16.0,)
            for cost in costs:
                results["model viterbi {:g}".format(cost)] = decode.decode_match(
                    prob, noisy.runs, cost, cost)
            print("  noise: {}  (ball unseen {:.0%})".format(tag, 1.0 - f.seen[truth != 3].mean()))
            for name, pred in results.items():
                s = score(truth, pred)
                print("    " + line(name, s))
                totals.setdefault("{} | {}".format(tag, name), []).append(s["spell"])

        if k == folds[0]:
            gain = booster.feature_importance("gain")
            top = np.argsort(-gain)[:20]
            print("  top features: " + ", ".join(names[i] for i in top))

    print("\nmean spellF1 over {} folds".format(len(list(folds))))
    for name, values in totals.items():
        print("  {:<48} {:.4f}  (min {:.4f}, max {:.4f})".format(
            name, float(np.mean(values)), float(np.min(values)), float(np.max(values))))
