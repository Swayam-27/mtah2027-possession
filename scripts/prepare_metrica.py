"""Convert the Metrica sample games in external/metrica to cache/MET-*.npz and describe them.

The description is printed next to the same numbers for a DFL match, as a check that labels
built from events look like labels built from tracking: number and length of possessions,
share of dead ball, and how often the team in possession is the one nearest the ball.

    python scripts/prepare_metrica.py
"""

import glob
import os
import sys
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")
from possession import io_metrica, relabel  # noqa: E402
from possession.match import Match, H, A, D  # noqa: E402
from possession.scoring import graded_truth  # noqa: E402
from possession.vendor import possession_metrics as pm  # noqa: E402

SOURCE = os.path.join(ROOT, "external", "metrica")
CACHE = os.path.join(ROOT, "cache")
MIN_CONTROL = 50


def describe(match):
    labels = relabel.controlled(match.labels, match.runs, MIN_CONTROL)
    from dataclasses import replace
    truth = graded_truth(replace(match, labels=labels))
    raw = pm.possession_stats(match.labels, match.runs)
    stats = pm.possession_stats(truth, match.runs)
    rows = np.concatenate([np.arange(a, b) for a, b in match.runs])
    lab = truth[rows]
    alive = (lab == H) | (lab == A)
    # Who is nearest the ball, where someone is within 1 m of it.
    home, away = match.pteam == 0, match.pteam == 1
    d = np.hypot(match.px[rows] - match.bx[rows, None], match.py[rows] - match.by[rows, None])
    d = np.where(np.isnan(d), np.inf, d)
    dh, da = d[:, home].min(axis=1), d[:, away].min(axis=1)
    close = alive & (np.minimum(dh, da) < 1.0)
    nearest = np.where(dh < da, H, A)
    agree = float((nearest[close] == lab[close]).mean()) if close.any() else float("nan")
    seen = ~np.isnan(match.bx[rows])
    moving = np.hypot(np.diff(match.bx[rows]), np.diff(match.by[rows])) * match.fps
    dead = lab[1:] == D
    return ("{:<12} minutes {:5.1f}  possessions raw {:4d} -> controlled {:4d}  mean {:5.1f} s  "
            "dead share {:.3f}  ball seen {:.3f} (alive {:.3f}, dead {:.3f})  "
            "nearest team agrees {:.3f}  ball still while dead {:.3f}".format(
                match.name, len(rows) / match.fps / 60.0, raw["n_spells"], stats["n_spells"],
                stats["mean_duration_s"], float((lab == D).mean()), float(seen.mean()),
                float(seen[alive].mean()), float(seen[lab == D].mean()) if (lab == D).any() else 0.0,
                agree, float(np.nanmean(moving[dead] < 0.1)) if dead.any() else 0.0))


if __name__ == '__main__':
    os.makedirs(CACHE, exist_ok=True)
    for number in (1, 2, 3):
        path = os.path.join(CACHE, "MET-{}.npz".format(number))
        match = Match.load(path) if os.path.isfile(path) else io_metrica.load(SOURCE, number)
        if not os.path.isfile(path):
            match.save(path)
        print(describe(match), flush=True)
    dfl = sorted(glob.glob(os.path.join(CACHE, "DFL-*.npz")))
    for path in dfl[:2]:
        print(describe(Match.load(path)), flush=True)
