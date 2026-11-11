"""Make clean external tracking look like the competition's automatic tracking.

The external matches are corrected tracking: ball on every frame, every player on every frame.
The competition ATD is not: the ball is unseen on ~41% of graded frames and players drop in and
out. A method tuned on clean data would be tuned for a different problem, so every external
match is degraded first.

The defaults were calibrated on 11 Nov 2026 against summary statistics of the competition feed
(scripts/atd_stats.py). That feed is smoothed, not jittery: short losses of the ball or of a
player are filled in by the tracker, so what remains missing is long stretches (median about
10 s for the ball), and the seen ball often moves in straight filled-in lines.

Ball loss is not uniform. It is lost when it is in the air, when it is in a crowd and when the
ball is out of play, so the loss process is a two-state chain whose rates depend on those.
All randomness comes from one seeded generator, so a degraded match is reproducible.
"""

from dataclasses import dataclass, replace

import numpy as np

from .match import D, X


@dataclass
class Noise:
    ball_missing: float = 0.41      # target share of unseen ball frames inside the runs
    ball_mean_gap: float = 150.0    # mean length of a raw loss, frames, in open play
    ball_fill: int = 75             # losses up to this long are filled with a straight line
    ball_smooth: int = 9            # moving-average window on the seen ball, frames
    air_factor: float = 4.0         # how much likelier the ball is lost above head height
    crowd_factor: float = 2.5       # ... with 3+ players within 3 m
    feet_factor: float = 2.5        # ... with a player within 1 m (ball hidden by the body)
    dead_factor: float = 3.0        # ... while the ball is out of play
    ball_jitter: float = 0.01       # metres, per-frame gaussian
    ball_drift: float = 0.6         # metres, slowly wandering position error (per axis)
    ball_false: float = 0.0015      # share of seen frames replaced by a false detection
    player_missing: float = 0.07    # share of player observations dropped
    player_mean_gap: float = 150.0  # mean length of a raw player dropout, frames
    player_fill: int = 60           # dropouts up to this long are filled (player stays tracked)
    player_jitter: float = 0.003    # metres
    player_smooth: int = 7          # moving-average window on player positions, frames
    dead_hidden: float = 0.45       # share of stoppages where the ball is never seen at all


def random_noise(rng):
    """A noise setting drawn from a wide range.

    The real tracking's exact noise is not known in advance, so training matches are degraded
    with many different settings. A model that only works for one of them is not one we want.
    """
    return Noise(
        ball_missing=float(rng.uniform(0.30, 0.50)),
        ball_mean_gap=float(rng.uniform(60.0, 350.0)),
        ball_fill=int(rng.integers(25, 126)),
        ball_smooth=int(rng.integers(1, 16)),
        air_factor=float(rng.uniform(2.0, 6.0)),
        crowd_factor=float(rng.uniform(1.5, 4.0)),
        feet_factor=float(rng.uniform(1.0, 4.0)),
        dead_factor=float(rng.uniform(1.5, 5.0)),
        ball_jitter=float(rng.uniform(0.0, 0.05)),
        ball_drift=float(rng.uniform(0.0, 1.2)),
        ball_false=float(rng.uniform(0.0, 0.004)),
        player_missing=float(rng.uniform(0.03, 0.12)),
        player_mean_gap=float(rng.uniform(60.0, 400.0)),
        player_fill=int(rng.integers(20, 101)),
        player_jitter=float(rng.uniform(0.0, 0.03)),
        player_smooth=int(rng.integers(1, 12)),
        dead_hidden=float(rng.uniform(0.15, 0.75)))


def _chain(rng, lose, regain):
    """Two-state chain with per-frame probabilities of losing and regaining. True = unseen."""
    n = len(lose)
    draws = rng.random(n)
    missing = np.zeros(n, dtype=bool)
    state = False
    for i in range(n):
        if state:
            if draws[i] < regain[i]:
                state = False
        elif draws[i] < lose[i]:
            state = True
        missing[i] = state
    return missing


def _ball_missing(match, noise, rng, forced):
    n = match.n_frames
    risk = np.ones(n, dtype=np.float64)
    if match.bz is not None:
        risk[np.nan_to_num(match.bz) > 2.0] *= noise.air_factor
    dist = np.hypot(match.px - match.bx[:, None], match.py - match.by[:, None])
    risk[(np.nan_to_num(dist, nan=np.inf) < 3.0).sum(axis=1) >= 3] *= noise.crowd_factor
    risk[(np.nan_to_num(dist, nan=np.inf) < 1.0).any(axis=1)] *= noise.feet_factor
    if match.labels is not None:
        risk[match.labels == D] *= noise.dead_factor

    in_runs = np.zeros(n, dtype=bool)
    for start, stop in match.runs:
        in_runs[start:stop] = True

    # Riskier frames are both easier to lose and slower to recover. The base loss rate is found
    # by bisection so the unseen share lands on the target whatever the match looks like.
    regain = np.clip((1.0 / noise.ball_mean_gap) / np.sqrt(risk), 0.0, 1.0)
    low, high = 1e-5, 0.5
    seed = int(rng.integers(0, 2 ** 31 - 1))
    for _ in range(18):
        mid = np.sqrt(low * high)
        lose = np.clip(mid * np.sqrt(risk), 0.0, 1.0)
        missing = _long_only(_chain(np.random.default_rng(seed), lose, regain) | forced,
                             noise.ball_fill)
        if missing[in_runs].mean() > noise.ball_missing:
            high = mid
        else:
            low = mid
    return missing


def _segments(mask):
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist())


def _long_only(mask, shortest):
    """Drop the True stretches of at most `shortest` frames: the tracker fills those in."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    short = (stops - starts) <= shortest
    delta = np.zeros(len(mask) + 1, dtype=np.int32)
    np.add.at(delta, starts[short], 1)
    np.add.at(delta, stops[short], -1)
    return mask & (np.cumsum(delta[:-1]) == 0)


def _fill_line(values, keep_missing):
    """Straight line through every missing frame that is not in `keep_missing`."""
    known = ~np.isnan(values)
    if known.sum() < 2:
        return values
    index = np.arange(len(values))
    filled = np.interp(index, index[known], values[known]).astype(values.dtype)
    filled[keep_missing] = np.nan
    return filled


def _drift(rng, n, size, fps):
    """Smooth wandering error with standard deviation `size`, changing over a few seconds."""
    if size <= 0:
        return np.zeros(n, dtype=np.float32)
    window = int(3 * fps)
    walk = np.convolve(rng.normal(0.0, 1.0, n + window), np.hanning(window), mode="same")[:n]
    return (walk / walk.std() * size).astype(np.float32)


def _smooth(values, window):
    """Moving average that ignores missing frames and keeps them missing."""
    if window <= 1:
        return values
    known = ~np.isnan(values)
    kernel = np.ones(window)
    total = np.convolve(np.where(known, values, 0.0), kernel, mode="same")
    count = np.convolve(known.astype(np.float64), kernel, mode="same")
    out = (total / np.maximum(count, 1.0)).astype(values.dtype)
    out[~known] = np.nan
    return out


def simulate_dead_ball(match, noise, rng):
    """Replace the ball during stoppages with what a camera would plausibly see.

    The external tracking freezes the ball at one point for the whole of every stoppage. Real
    automatic tracking does not: the ball rolls on after going dead, disappears, then comes
    back being carried to the restart spot and is put down before the restart. A model trained
    on the frozen ball would learn "motionless ball = dead" as a perfect rule that does not
    exist in the competition data, so the frozen stretch is rebuilt here from the real player
    movement: roll-out, unseen, carried by the player who takes the restart, placed.
    """
    bx, by = match.bx.copy(), match.by.copy()
    fps = match.fps
    for start, stop in match.runs:
        dead = match.labels[start:stop] == D
        for a, b in _segments(dead):
            a, b = a + start, b + start
            bx[a:b], by[a:b] = np.nan, np.nan
            if rng.random() < noise.dead_hidden:
                continue
            # Roll-out: keep the last live velocity, slowing down.
            if a - 6 >= start and match.labels[a - 6] != D:
                vx = (match.bx[a - 1] - match.bx[a - 6]) / 5.0
                vy = (match.by[a - 1] - match.by[a - 6]) / 5.0
                k = min(b - a, int(rng.integers(10, 50)))
                decay = np.cumsum(np.exp(-np.arange(1, k + 1) / float(fps)))
                bx[a:a + k] = match.bx[a - 1] + vx * decay
                by[a:a + k] = match.by[a - 1] + vy * decay
            else:
                k = 0
            if b >= stop:
                continue  # the half ends dead: no restart to carry the ball to
            # Carried to the restart spot by whoever is nearest to it when play resumes.
            spot_x, spot_y = match.bx[b], match.by[b]
            dist = np.hypot(match.px[b] - spot_x, match.py[b] - spot_y)
            if np.all(np.isnan(dist)):
                continue
            taker = int(np.nanargmin(dist))
            still = int(rng.uniform(0.5, 4.0) * fps)
            carry = int(rng.uniform(3.0, 12.0) * fps)
            place = max(a + k, b - still)
            pick = max(a + k, place - carry)
            bx[pick:place] = match.px[pick:place, taker] + 0.3
            by[pick:place] = match.py[pick:place, taker]
            bx[place:b], by[place:b] = spot_x, spot_y
    return bx, by


def degrade(match, noise=None, seed=0):
    noise = noise or Noise()
    rng = np.random.default_rng(seed)
    n, slots = match.px.shape
    length, width = match.pitch

    if match.labels is not None:
        dead_x, dead_y = simulate_dead_ball(match, noise, rng)
    else:
        dead_x, dead_y = match.bx, match.by
    missing = _ball_missing(match, noise, rng, np.isnan(dead_x))
    bx = dead_x + rng.normal(0.0, noise.ball_jitter, n).astype(np.float32)
    by = dead_y + rng.normal(0.0, noise.ball_jitter, n).astype(np.float32)
    bx += _drift(rng, n, noise.ball_drift, match.fps)
    by += _drift(rng, n, noise.ball_drift, match.fps)
    false = rng.random(n) < noise.ball_false
    false &= ~np.isnan(bx)
    bx[false] = rng.uniform(0.0, length, int(false.sum()))
    by[false] = rng.uniform(0.0, width, int(false.sum()))
    # `missing` holds only the long losses. Everything shorter is drawn as a straight line
    # between the sightings on either side, which is what the competition tracker does.
    bx[np.isnan(dead_x)] = np.nan
    by[np.isnan(dead_x)] = np.nan
    lost = _chain(rng, np.full(n, 0.01), np.full(n, 0.10)) & ~missing
    bx[lost], by[lost] = np.nan, np.nan
    bx = _smooth(_fill_line(bx, missing), noise.ball_smooth)
    by = _smooth(_fill_line(by, missing), noise.ball_smooth)

    px = match.px + rng.normal(0.0, noise.player_jitter, (n, slots)).astype(np.float32)
    py = match.py + rng.normal(0.0, noise.player_jitter, (n, slots)).astype(np.float32)
    regain = np.full(n, 1.0 / noise.player_mean_gap)
    lose = np.full(n, noise.player_missing / (1.0 - noise.player_missing) / noise.player_mean_gap)
    for slot in range(slots):
        gone = _long_only(_chain(rng, lose, regain), noise.player_fill)
        px[:, slot] = _smooth(px[:, slot], noise.player_smooth)
        py[:, slot] = _smooth(py[:, slot], noise.player_smooth)
        px[gone, slot] = np.nan
        py[gone, slot] = np.nan

    return replace(match, name=match.name + "-noisy", px=px, py=py, bx=bx, by=by, bz=None)
