"""Compare decoders on the saved held-out predictions (cache/oof-*.npz from scripts/cv.py).

Every match's predictions came from a model that never saw that match, so the comparison is
fair, and it takes seconds because nothing is retrained.

    python scripts/tune_decode.py
"""

import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from possession import decode  # noqa: E402
from possession.scoring import score  # noqa: E402


def load():
    out = []
    for path in sorted(glob.glob(os.path.join(ROOT, "cache", "oof-*.npz"))):
        z = np.load(path)
        out.append((z["prob"], z["change"], z["truth"], [(int(a), int(b)) for a, b in z["runs"]]))
    return out


def report(name, matches, fn):
    scores = [score(truth, fn(prob, change, runs)) for prob, change, truth, runs in matches]
    spell = [s["spell"] for s in scores]
    print("{:<44} spell {:.4f} (min {:.4f})  macro {:.4f}  poss {:>4.0f}/{:<4.0f}".format(
        name, np.mean(spell), np.min(spell), np.mean([s["macro"] for s in scores]),
        np.mean([s["n_pred"] for s in scores]), np.mean([s["n_gt"] for s in scores])))
    return float(np.mean(spell))


if __name__ == '__main__':
    matches = load()
    print("{} held-out matches\n".format(len(matches)))

    print("constant switching price")
    for team in (8.0, 12.0, 16.0, 24.0):
        for dead in (8.0, 16.0, 32.0):
            report("  team {:g}  dead {:g}".format(team, dead), matches,
                   lambda p, c, r, t=team, d=dead: decode.decode_match(p, r, t, d))

    print("\nprice from the boundary model")
    for scale in (1.0, 1.5, 2.0, 3.0, 4.0, 6.0):
        report("  scale {:g}".format(scale), matches,
               lambda p, c, r, s=scale: decode.decode_match(p, r, change=c, scale=s))

    print("\nboundary model + dropping very short possessions")
    for scale in (1.5, 2.0, 3.0):
        for shortest in (12, 25, 38):
            report("  scale {:g}  shortest {} frames".format(scale, shortest), matches,
                   lambda p, c, r, s=scale, m=shortest: decode.drop_short(
                       decode.decode_match(p, r, change=c, scale=s), r, m))
