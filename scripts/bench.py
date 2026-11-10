"""The practice exam: score methods on external matches degraded to look like the competition ATD.

    python scripts/bench.py                 # every cached external match
    python scripts/bench.py --matches 2     # only the first two (quick)
"""

import argparse
import glob
import os
import sys
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")
from possession import baseline, engine, features, relabel  # noqa: E402
from possession.degrade import degrade  # noqa: E402
from possession.match import Match  # noqa: E402
from possession.scoring import graded_truth, line, score  # noqa: E402

CACHE = os.path.join(ROOT, "cache")
MIN_CONTROL = 50  # frames an opposing spell must last to count as a change of possession


def prepared(path, seed=0):
    """(degraded match, controlled-possession truth) for one cached external match."""
    match = Match.load(path)
    match.labels = relabel.controlled(match.labels, match.runs, MIN_CONTROL)
    return degrade(match, seed=seed), graded_truth(match)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--matches", type=int, default=0)
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(CACHE, "DFL-*.npz")))
    if args.matches:
        paths = paths[:args.matches]

    totals = {}
    for path in paths:
        noisy, truth = prepared(path)
        f = features.compute(noisy)
        graded = np.concatenate([np.arange(a, b) for a, b in noisy.runs])
        print("\n{}   ball unseen {:.1%}   players seen {:.1f}".format(
            noisy.name, 1.0 - f.seen[graded].mean(), f.n_players[graded].mean()))
        methods = {
            "baseline_smooth50": baseline.label_match(noisy, 50),
            "rules": engine.label_match(noisy, f),
        }
        for name, pred in methods.items():
            s = score(truth, pred)
            print("  " + line(name, s))
            totals.setdefault(name, []).append(s["spell"])

    print("\nmean spellF1 over {} matches".format(len(paths)))
    for name, values in totals.items():
        print("  {:<28} {:.4f}".format(name, float(np.mean(values))))
