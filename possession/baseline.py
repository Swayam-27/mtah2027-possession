"""The organisers' baseline on a Match: nearest player to a seen ball, then a majority vote.

Kept only as the reference line on our benchmark, so every gain is measured against the same
thing the published floor (spellF1 0.2433) measures.
"""

import numpy as np

from .match import H, A, D, X, HOME


def label_match(match, window=50):
    n = match.n_frames
    codes = np.full(n, -1, dtype=np.int8)
    seen = np.flatnonzero(~np.isnan(match.bx))
    for a in range(0, len(seen), 20000):
        rows = seen[a:a + 20000]
        dist = np.hypot(match.px[rows] - match.bx[rows, None], match.py[rows] - match.by[rows, None])
        dist = np.where(np.isnan(dist), np.inf, dist)
        codes[rows] = np.where(match.pteam[dist.argmin(axis=1)] == HOME, H, A)

    home = np.concatenate(([0], np.cumsum(codes == H)))
    away = np.concatenate(([0], np.cumsum(codes == A)))
    index = np.arange(n)
    low, high = np.maximum(0, index - window), np.minimum(n, index + window + 1)
    n_home, n_away = home[high] - home[low], away[high] - away[low]
    labels = np.full(n, D, dtype=np.int8)
    some = (n_home + n_away) > 0
    labels[some] = np.where(n_home[some] >= n_away[some], H, A)

    out = np.full(n, X, dtype=np.int8)
    for start, stop in match.runs:
        out[start:stop] = labels[start:stop]
    return out
