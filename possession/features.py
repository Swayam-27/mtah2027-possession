"""Per-frame measurements shared by the rule engine and the learned model.

Nothing here assumes a player column is a person: every player quantity is reduced over the
columns of a team (nearest, count within a radius, median speed), so identity switches and
fragmented tracks in the automatic tracking cannot break it.
"""

from dataclasses import dataclass

import numpy as np

from .match import HOME, AWAY

MAX_BALL_SPEED = 45.0     # m/s; anything faster between two sightings is a false detection
MAX_PLAYER_SPEED = 12.0   # m/s; faster means the slot jumped to another person
SPEED_LAG = 5             # frames used for finite-difference speeds
CHUNK = 20000


@dataclass
class Features:
    seen: np.ndarray        # ball observed on this frame (after removing false detections)
    bx: np.ndarray          # ball position with gaps filled by straight lines inside each run
    by: np.ndarray
    gap: np.ndarray         # length in frames of the unseen stretch this frame is in, 0 if seen
    edge: np.ndarray        # frames to the nearest sighting, 0 if seen
    speed: np.ndarray       # ball speed, m/s
    accel: np.ndarray       # magnitude of ball velocity change, m/s per frame-pair
    d_home: np.ndarray      # distance from ball to nearest home / away player, metres
    d_away: np.ndarray
    n_home: np.ndarray      # players of each team within 3 m of the ball
    n_away: np.ndarray
    out: np.ndarray         # metres the ball is beyond the nearest pitch line, 0 inside
    walk: np.ndarray        # median speed of all tracked players, m/s
    n_players: np.ndarray   # tracked players on the frame
    fast: np.ndarray        # share of tracked players moving faster than 3 m/s
    speed_p90: np.ndarray   # 90th percentile of player speed, m/s
    speed_mean: np.ndarray  # mean player speed, m/s
    spread_x: np.ndarray    # standard deviation of player positions along / across the pitch
    spread_y: np.ndarray
    drift: np.ndarray       # speed of the centroid of all players, m/s


def remove_ball_teleports(bx, by, fps, max_speed=MAX_BALL_SPEED):
    """Drop sightings that cannot be reached from both neighbouring sightings."""
    bx, by = bx.copy(), by.copy()
    for _ in range(3):
        idx = np.flatnonzero(~np.isnan(bx))
        if len(idx) < 3:
            break
        step = np.hypot(np.diff(bx[idx]), np.diff(by[idx])) / (np.diff(idx) / float(fps))
        too_fast = step > max_speed
        bad = np.zeros(len(idx), dtype=bool)
        bad[1:-1] = too_fast[:-1] & too_fast[1:]
        if not bad.any():
            break
        bx[idx[bad]] = np.nan
        by[idx[bad]] = np.nan
    return bx, by


def fill_ball(bx, by, runs):
    """Straight-line fill of unseen stretches, inside each run only. Returns x, y, gap, edge."""
    fx, fy = bx.copy(), by.copy()
    gap = np.zeros(len(bx), dtype=np.int32)
    edge = np.zeros(len(bx), dtype=np.int32)
    for start, stop in runs:
        window = np.arange(start, stop)
        known = window[~np.isnan(bx[start:stop])]
        if not len(known):
            gap[start:stop] = stop - start
            edge[start:stop] = stop - start
            continue
        fx[start:stop] = np.interp(window, known, bx[known])
        fy[start:stop] = np.interp(window, known, by[known])
        # Position of each frame relative to the sightings either side of it.
        after = np.searchsorted(known, window, side="left")
        before = after - 1
        prev_seen = np.where(before >= 0, known[np.clip(before, 0, len(known) - 1)], start - 1)
        next_seen = np.where(after < len(known), known[np.clip(after, 0, len(known) - 1)], stop)
        missing = np.isnan(bx[start:stop])
        gap[start:stop] = np.where(missing, next_seen - prev_seen - 1, 0)
        edge[start:stop] = np.where(missing, np.minimum(window - prev_seen, next_seen - window), 0)
    return fx, fy, gap, edge


def player_speeds(px, py, fps, lag=SPEED_LAG):
    """Per-slot speed by finite difference, NaN where the slot is unseen or jumped."""
    speed = np.full(px.shape, np.nan, dtype=np.float32)
    step = np.hypot(px[lag:] - px[:-lag], py[lag:] - py[:-lag]) * (fps / float(lag))
    step[step > MAX_PLAYER_SPEED] = np.nan
    speed[lag:] = step
    return speed


def compute(match):
    fps = match.fps
    length, width = match.pitch
    n = match.n_frames

    raw_x, raw_y = remove_ball_teleports(match.bx, match.by, fps)
    seen = ~np.isnan(raw_x)
    bx, by, gap, edge = fill_ball(raw_x, raw_y, match.runs)

    # Ball velocity on the filled track, central difference over +/-2 frames.
    vx = np.zeros(n, dtype=np.float32)
    vy = np.zeros(n, dtype=np.float32)
    vx[2:-2] = (bx[4:] - bx[:-4]) * (fps / 4.0)
    vy[2:-2] = (by[4:] - by[:-4]) * (fps / 4.0)
    vx, vy = np.nan_to_num(vx), np.nan_to_num(vy)
    speed = np.hypot(vx, vy)
    accel = np.zeros(n, dtype=np.float32)
    accel[3:-3] = np.hypot(vx[6:] - vx[:-6], vy[6:] - vy[:-6])

    d_home = np.full(n, np.inf, dtype=np.float32)
    d_away = np.full(n, np.inf, dtype=np.float32)
    n_home = np.zeros(n, dtype=np.int8)
    n_away = np.zeros(n, dtype=np.int8)
    home_cols = np.flatnonzero(match.pteam == HOME)
    away_cols = np.flatnonzero(match.pteam == AWAY)
    for a in range(0, n, CHUNK):
        b = min(n, a + CHUNK)
        dist = np.hypot(match.px[a:b] - bx[a:b, None], match.py[a:b] - by[a:b, None])
        dist = np.where(np.isnan(dist), np.inf, dist)
        d_home[a:b] = dist[:, home_cols].min(axis=1)
        d_away[a:b] = dist[:, away_cols].min(axis=1)
        n_home[a:b] = (dist[:, home_cols] < 3.0).sum(axis=1)
        n_away[a:b] = (dist[:, away_cols] < 3.0).sum(axis=1)

    out = np.maximum.reduce([-bx, bx - length, -by, by - width, np.zeros(n, dtype=np.float32)])
    out = np.nan_to_num(out)

    speeds = player_speeds(match.px, match.py, fps)
    n_players = (~np.isnan(match.px)).sum(axis=1)
    with np.errstate(all="ignore"):
        walk = np.nanmedian(speeds, axis=1)
        speed_p90 = np.nanpercentile(speeds, 90, axis=1)
        speed_mean = np.nanmean(speeds, axis=1)
        fast = np.nanmean(np.where(np.isnan(speeds), np.nan, speeds > 3.0), axis=1)
        spread_x = np.nanstd(match.px, axis=1)
        spread_y = np.nanstd(match.py, axis=1)
        cx, cy = np.nanmean(match.px, axis=1), np.nanmean(match.py, axis=1)
    drift = np.zeros(n, dtype=np.float32)
    drift[fps:] = np.hypot(cx[fps:] - cx[:-fps], cy[fps:] - cy[:-fps])
    clean = lambda v: np.nan_to_num(np.asarray(v, dtype=np.float32))  # noqa: E731

    return Features(seen=seen, bx=bx, by=by, gap=gap, edge=edge, speed=speed, accel=accel,
                    d_home=d_home, d_away=d_away, n_home=n_home, n_away=n_away, out=out,
                    walk=clean(walk), n_players=n_players, fast=clean(fast),
                    speed_p90=clean(speed_p90), speed_mean=clean(speed_mean),
                    spread_x=clean(spread_x), spread_y=clean(spread_y), drift=clean(drift))
