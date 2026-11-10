"""Turn touch-level possession labels into controlled-possession labels.

The open DFL matches flip possession on every touch: a blocked pass or a lost header becomes a
half-second possession for the other side. The competition's definition (CHALLENGE.md) does not:
possession changes only when the other team gains *controlled* possession, and a loose or
contested ball stays with the last team that had it.

So before an external match is used for development, an opposing spell that lasts less than
`min_control` frames is handed back to the team that had the ball. A spell that opens a passage
of play (straight after a dead ball) is always kept: a restart has a clear possessor.
"""

import numpy as np

from .match import H, A, D, X


def controlled(labels, runs, min_control):
    out = labels.copy()
    for start, stop in runs:
        window = labels[start:stop]
        breaks = np.flatnonzero(window[1:] != window[:-1]) + 1
        bounds = np.concatenate(([0], breaks, [len(window)]))
        owner = None
        for i in range(len(bounds) - 1):
            a, b = int(bounds[i]), int(bounds[i + 1])
            code = int(window[a])
            if code not in (H, A):
                owner = None  # dead ball: the next spell is a restart
                continue
            if owner is None or code == owner or (b - a) >= min_control:
                owner = code
            else:
                out[start + a:start + b] = owner
    return out
