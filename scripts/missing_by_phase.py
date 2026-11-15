"""Is the ball missing in live play or in dead time? Real tracking against our simulation.

The real match has no public labels beyond 5 minutes, so live/dead comes from our own submission
(a good guide: dead-ball stretches matched 53 of 58 tagged restarts). Prints aggregates only.

    python scripts/missing_by_phase.py --submission "outputs/Chandal Chowkdi.csv"
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from atd_stats import data_dir, load_competition  # noqa: E402
from possession import relabel  # noqa: E402
from possession.degrade import Noise, degrade  # noqa: E402
from possession.match import Match, H, A, D  # noqa: E402

CODES = {"H": H, "A": A, "D": D, "X": 3}


def gaps(missing):
    """Lengths (frames) of the unseen stretches."""
    edges = np.diff(np.concatenate(([0], missing.astype(np.int8), [0])))
    return np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)


def describe(title, bx, labels, runs):
    inside = np.zeros(len(bx), dtype=bool)
    for a, b in runs:
        inside[a:b] = True
    missing = np.isnan(bx)
    live = inside & ((labels == H) | (labels == A))
    dead = inside & (labels == D)
    print(title)
    print("  frames: live {:.3f}, dead {:.3f} of the halves".format(live.sum() / inside.sum(),
                                                                    dead.sum() / inside.sum()))
    print("  ball unseen: all {:.3f} | live play {:.3f} | dead ball {:.3f}".format(
        missing[inside].mean(), missing[live].mean(), missing[dead].mean()))
    print("  share of the unseen frames that are live play: {:.3f}".format(
        (missing & live).sum() / max((missing & inside).sum(), 1)))
    g = gaps(missing & live)
    if len(g):
        print("  unseen stretches in live play: {} per 90 min, length s (p50 p90) {:.1f} {:.1f}, "
              "longer than 3 s: {:.3f} of live play".format(
                  int(round(len(g) * 90 * 60 * 25 / inside.sum())), np.percentile(g, 50) / 25,
                  np.percentile(g, 90) / 25, g[g > 75].sum() / live.sum()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None)
    parser.add_argument("--submission", default=os.path.join(ROOT, "outputs", "Chandal Chowkdi.csv"))
    args = parser.parse_args()

    match = load_competition(data_dir(args.data))
    table = pd.read_csv(args.submission)
    labels = table["label"].map(CODES).to_numpy().astype(np.int8)
    describe("REAL TRACKING (live/dead from our submission)", match.bx, labels, match.runs)

    for path in sorted(glob.glob(os.path.join(ROOT, "cache", "DFL-*.npz")))[:3]:
        ext = Match.load(path)
        ext.labels = relabel.controlled(ext.labels, ext.runs, 50)
        noisy = degrade(ext, Noise(), seed=1)
        describe("\nSIMULATED, calibrated noise ({})".format(os.path.basename(path)[:-4]),
                 noisy.bx, ext.labels, ext.runs)
