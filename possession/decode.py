"""Turn per-frame class probabilities into clean possession segments.

A per-frame classifier flickers, and the ranking metric punishes every flicker as an invented
possession. Viterbi decoding finds the single most likely label sequence when every change of
label has a price, so a change only happens where the evidence pays for it.

The price can vary from frame to frame. With a boundary model, a change is cheap where that
model says a change is happening and expensive everywhere else, which lets short possessions
through without letting flicker through.
"""

import numpy as np

from .match import H, A, D, X


def viterbi(log_prob, cost_team, cost_dead):
    """Most likely H/A/D path.

    log_prob is [n, 3]. cost_team and cost_dead are the prices, in log-probability units, of a
    team change and of a dead/alive change on entering each frame: scalars or length-n arrays.
    """
    n = len(log_prob)
    emit = log_prob.tolist()
    team = np.broadcast_to(np.asarray(cost_team, dtype=np.float64), (n,)).tolist()
    dead = np.broadcast_to(np.asarray(cost_dead, dtype=np.float64), (n,)).tolist()

    h, a, d = emit[0]
    back = np.zeros((n, 3), dtype=np.int8)
    back_rows = []
    for t in range(1, n):
        ct, cd = team[t], dead[t]
        e = emit[t]
        # Best predecessor for each state; ties keep the current state.
        bh, ph = h, 0
        if a - ct > bh:
            bh, ph = a - ct, 1
        if d - cd > bh:
            bh, ph = d - cd, 2
        ba, pa = a, 1
        if h - ct > ba:
            ba, pa = h - ct, 0
        if d - cd > ba:
            ba, pa = d - cd, 2
        bd, pd = d, 2
        if h - cd > bd:
            bd, pd = h - cd, 0
        if a - cd > bd:
            bd, pd = a - cd, 1
        h, a, d = bh + e[0], ba + e[1], bd + e[2]
        back_rows.append((ph, pa, pd))
    if back_rows:
        back[1:] = np.array(back_rows, dtype=np.int8)

    path = np.empty(n, dtype=np.int8)
    path[-1] = int(np.argmax([h, a, d]))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def change_costs(change_prob, scale, radius=8, floor=1e-4):
    """Per-frame switching prices from the boundary model's probabilities.

    The boundary model says a change lies within `radius` frames; spread over that window, the
    chance it falls on this exact frame is p / (2 * radius). The price is its negative log.
    """
    per_frame = np.clip(change_prob / (2.0 * radius), floor, 1.0)
    return -scale * np.log(per_frame)


def decode_match(prob, runs, switch_team=16.0, switch_dead=16.0, change=None, scale=None):
    """Decode each run on its own. prob is [n_frames, 3] over H, A, D.

    With `change` ([n_frames, 3] over no change / team change / dead change) and `scale`, the
    switching prices follow the boundary model instead of being constant.
    """
    labels = np.full(len(prob), X, dtype=np.int8)
    for start, stop in runs:
        log_prob = np.log(np.clip(prob[start:stop], 1e-6, 1.0))
        if change is None:
            cost_team, cost_dead = switch_team, switch_dead
        else:
            cost_team = change_costs(change[start:stop, 1], scale)
            cost_dead = change_costs(change[start:stop, 2], scale)
        labels[start:stop] = viterbi(log_prob, cost_team, cost_dead)
    return labels


def drop_short(labels, runs, min_frames):
    """Hand possessions shorter than `min_frames` to the label before them."""
    out = labels.copy()
    for start, stop in runs:
        window = out[start:stop]
        breaks = np.flatnonzero(window[1:] != window[:-1]) + 1
        bounds = np.concatenate(([0], breaks, [len(window)]))
        for i in range(1, len(bounds) - 1):
            a, b = int(bounds[i]), int(bounds[i + 1])
            if window[a] in (H, A) and b - a < min_frames:
                window[a:b] = window[a - 1]
    return out
