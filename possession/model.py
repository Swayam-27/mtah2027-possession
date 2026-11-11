"""The learned model: per-frame H/A/D probabilities from tracking geometry.

Trained only on OTHER matches (open DFL data, degraded to look like the competition tracking).
Each frame is described by what is happening at that instant and by short summaries of the
seconds before and after it -- possession is a property of an interval, so a frame on its own
does not hold the answer. The rule engine's own verdict is one of the inputs: the model learns
where the rules can be trusted and where they cannot.

The two teams are interchangeable, so every prediction is made twice -- once as given and once
with the teams swapped -- and averaged. The model therefore cannot favour "home".
"""

from dataclasses import replace

import numpy as np

from . import engine, shape as team_shape
from .match import H, A, D

OFFSETS = (5, 12, 25, 50)
WINDOWS = (12, 25, 50, 100, 200)
TRUST = 30


def _shift(x, k):
    """x at t+k, with the ends held."""
    out = np.empty_like(x)
    if k > 0:
        out[:-k], out[-k:] = x[k:], x[-1]
    else:
        out[-k:], out[:-k] = x[:k], x[0]
    return out


def _past_future_mean(x, w):
    """Mean of x over the w frames before t and over the w frames after t."""
    c = np.concatenate(([0.0], np.cumsum(x, dtype=np.float64)))
    n = len(x)
    i = np.arange(n)
    lo = np.maximum(0, i - w)
    past = (c[i] - c[lo]) / np.maximum(1, i - lo)
    hi = np.minimum(n, i + 1 + w)
    future = (c[hi] - c[i + 1]) / np.maximum(1, hi - (i + 1))
    return past.astype(np.float32), future.astype(np.float32)


def _since_until(flag, cap=250):
    """Frames since the flag was last true and until it is next true, capped, in seconds."""
    n = len(flag)
    index = np.arange(n)
    last = np.maximum.accumulate(np.where(flag, index, -cap))
    following = np.minimum.accumulate(np.where(flag, index, n + cap)[::-1])[::-1]
    since = np.minimum(index - last, cap) / 25.0
    until = np.minimum(following - index, cap) / 25.0
    return since.astype(np.float32), until.astype(np.float32)


def swap_teams(f):
    """The same features seen from the other side."""
    return replace(f, d_home=f.d_away, d_away=f.d_home, n_home=f.n_away, n_away=f.n_home,
                   shape=team_shape.swap(f.shape))


def base_features(match, f, rules=None, shape=False):
    """{name: [n_frames] float32}: what is happening on each frame, with no time context."""
    length, width = match.pitch
    rule_labels = engine.label_match(match, f, rules)

    dh = np.minimum(f.d_home, 40.0)
    da = np.minimum(f.d_away, 40.0)
    trusted = f.edge <= TRUST
    bx, by = np.nan_to_num(f.bx, nan=length / 2), np.nan_to_num(f.by, nan=width / 2)

    base = {
        "seen": f.seen.astype(np.float32),
        "gap_s": np.minimum(f.gap, 500) / 25.0,
        "edge_s": np.minimum(f.edge, 250) / 25.0,
        "speed": np.minimum(f.speed, 40.0),
        "accel": np.minimum(f.accel, 40.0),
        "d_home": dh,
        "d_away": da,
        "d_min": np.minimum(dh, da),
        "d_diff": np.clip(da - dh, -20.0, 20.0),
        "on_home": (trusted & (dh < 2.0) & (da > dh + 0.4)).astype(np.float32),
        "on_away": (trusted & (da < 2.0) & (dh > da + 0.4)).astype(np.float32),
        "n_home": f.n_home.astype(np.float32),
        "n_away": f.n_away.astype(np.float32),
        "out": np.minimum(f.out, 10.0),
        "to_touchline": np.minimum(by, width - by),
        "to_goal_line": np.minimum(bx, length - bx),
        "walk": f.walk,
        "n_players": f.n_players.astype(np.float32),
        "fast": f.fast,
        "speed_p90": f.speed_p90,
        "speed_mean": f.speed_mean,
        "spread_x": f.spread_x,
        "spread_y": f.spread_y,
        "drift": f.drift,
        "rule_home": (rule_labels == H).astype(np.float32),
        "rule_away": (rule_labels == A).astype(np.float32),
        "rule_dead": (rule_labels == D).astype(np.float32),
    }
    base = {k: np.nan_to_num(np.asarray(v, dtype=np.float32)) for k, v in base.items()}

    # Who was last on the ball, and who is on it next: the plainest evidence of possession
    # for all the frames in between, where nobody is near the ball.
    for side in ("home", "away"):
        since, until = _since_until(base["on_" + side] > 0)
        base["since_on_" + side] = since
        base["until_on_" + side] = until
    base["since_diff"] = base["since_on_away"] - base["since_on_home"]
    base["until_diff"] = base["until_on_away"] - base["until_on_home"]
    since, until = _since_until(base["seen"] > 0)
    base["since_seen"], base["until_seen"] = since, until
    if shape:  # the sequence model's extra inputs; the tree model was trained without them
        for key, values in f.shape.items():
            base["shape_" + key] = values
    return base


def matrix(match, f, rules=None):
    """(features [n_frames, n_features], names): each frame plus summaries of its surroundings."""
    base = base_features(match, f, rules)
    columns, names = [], []
    for name, values in base.items():
        columns.append(values)
        names.append(name)
    for name in ("d_diff", "d_min", "speed", "seen", "on_home", "on_away", "walk"):
        for k in OFFSETS:
            for sign, tag in ((-1, "m"), (1, "p")):
                columns.append(_shift(base[name], sign * k))
                names.append("{}_{}{}".format(name, tag, k))
    for name in ("on_home", "on_away", "seen", "speed", "walk", "out", "d_min",
                 "rule_home", "rule_away", "rule_dead", "fast", "speed_p90", "speed_mean",
                 "spread_x", "spread_y", "drift"):
        for w in WINDOWS:
            past, future = _past_future_mean(base[name], w)
            columns += [past, future]
            names += ["{}_past{}".format(name, w), "{}_next{}".format(name, w)]
    return np.column_stack(columns), names


NO_CHANGE, TEAM_CHANGE, DEAD_CHANGE = 0, 1, 2
BOUNDARY_RADIUS = 8


def boundary_target(truth, runs, radius=BOUNDARY_RADIUS):
    """Per frame: is a change of label close by, and of which kind?

    TEAM_CHANGE where possession passes between the teams within `radius` frames, DEAD_CHANGE
    where the ball goes dead or comes back into play. Trained as a second model so the decoder
    knows *where* a change is plausible, not only which label each frame looks like.
    """
    target = np.zeros(len(truth), dtype=np.int64)
    for start, stop in runs:
        window = truth[start:stop]
        for i in np.flatnonzero(window[1:] != window[:-1]) + 1:
            kind = DEAD_CHANGE if D in (window[i - 1], window[i]) else TEAM_CHANGE
            a, b = max(0, i - radius), min(len(window), i + radius)
            span = target[start + a:start + b]
            span[span == NO_CHANGE] = kind
    return target


def predict(booster, match, f, rules=None, boundary_booster=None):
    """[n_frames, 3] probabilities over H, A, D, averaged over both team orderings.

    With a boundary model, also returns its [n_frames, 3] probabilities over
    no change / team change / dead change.
    """
    x, _ = matrix(match, f, rules)
    x_swapped, _ = matrix(match, swap_teams(f), rules)
    prob = (booster.predict(x) + booster.predict(x_swapped)[:, [A, H, D]]) / 2.0
    if boundary_booster is None:
        return prob
    return prob, (boundary_booster.predict(x) + boundary_booster.predict(x_swapped)) / 2.0
