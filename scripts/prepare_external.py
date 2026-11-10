"""Convert the downloaded open DFL matches into cached Match files and describe them.

    python scripts/prepare_external.py            # all matches in external/dfl
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from possession import io_dfl  # noqa: E402
from possession.match import Match, H, A, D  # noqa: E402
from possession.vendor import possession_metrics as pm  # noqa: E402
from possession.scoring import graded_truth  # noqa: E402

DFL_DIR = os.path.join(ROOT, "external", "dfl")
CACHE = os.path.join(ROOT, "cache")


def describe(match):
    truth = graded_truth(match)
    runs = pm.valid_runs(truth)
    stats = pm.possession_stats(truth, runs)
    graded = truth[truth != 3]
    minutes = len(graded) / match.fps / 60.0
    shares = [float(np.mean(graded == k)) for k in (H, A, D)]

    # Does BallPossession==1 really mean the home side? Nearest player to a live ball should
    # agree with the label far more often than not.
    alive = np.flatnonzero((truth == H) | (truth == A))[::10]
    dist = np.hypot(match.px[alive] - match.bx[alive, None], match.py[alive] - match.by[alive, None])
    dist = np.where(np.isnan(dist), np.inf, dist)
    nearest_team = match.pteam[np.argmin(dist, axis=1)]
    agree = float(np.mean(nearest_team == truth[alive]))

    print("{:<16} {:6.1f} min  H {:.3f} A {:.3f} D {:.3f}  possessions {:>4} ({:.2f}/min) "
          "mean {:5.1f} s  nearest-player agreement {:.3f}  slots {}".format(
              match.name, minutes, shares[0], shares[1], shares[2], stats["n_spells"],
              stats["n_spells"] / minutes, stats["mean_duration_s"], agree, match.px.shape[1]))


if __name__ == '__main__':
    os.makedirs(CACHE, exist_ok=True)
    for match_id in io_dfl.match_ids(DFL_DIR):
        path = os.path.join(CACHE, "DFL-{}.npz".format(match_id))
        if os.path.isfile(path):
            match = Match.load(path)
        else:
            match = io_dfl.load(DFL_DIR, match_id)
            match.save(path)
        describe(match)
