"""Summary statistics of a tracking feed. Aggregates only -- never rows, frames or positions.

Used to check that the degraded external matches look like the competition tracking: the same
table is printed for both and compared by eye.
"""

import numpy as np

from .features import remove_ball_teleports


def _stretches(mask):
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)


def _pct(values, points=(10, 25, 50, 75, 90, 99)):
    if not len(values):
        return "n/a"
    return " ".join("{:.4g}".format(v) for v in np.percentile(values, points))


def tracking_stats(match):
    """Dict of printable lines describing the feed inside the match's play runs."""
    fps = match.fps
    length, width = match.pitch
    rows = np.concatenate([np.arange(a, b) for a, b in match.runs])
    out = {}

    seen = ~np.isnan(match.bx)
    clean_x, _ = remove_ball_teleports(match.bx, match.by, fps)
    out["frames in play runs"] = "{}".format(len(rows))
    out["ball unseen share"] = "{:.3f}".format(1.0 - seen[rows].mean())
    out["ball false detections removed (share of seen)"] = "{:.4f}".format(
        float((seen & np.isnan(clean_x))[rows].sum()) / max(1, int(seen[rows].sum())))

    gaps, seen_runs = [], []
    for a, b in match.runs:
        gaps.append(_stretches(~seen[a:b]))
        seen_runs.append(_stretches(seen[a:b]))
    gaps, seen_runs = np.concatenate(gaps), np.concatenate(seen_runs)
    out["ball unseen stretches: count"] = "{}".format(len(gaps))
    out["ball unseen stretch length, frames (p10 p25 p50 p75 p90 p99)"] = _pct(gaps)
    out["ball seen stretch length, frames (p10 p25 p50 p75 p90 p99)"] = _pct(seen_runs)
    for lo, hi in ((1, 5), (5, 25), (25, 75), (75, 250), (250, 10 ** 9)):
        inside = gaps[(gaps >= lo) & (gaps < hi)]
        out["unseen frames in stretches of {}-{} frames (share of unseen)".format(
            lo, hi if hi < 10 ** 9 else "inf")] = "{:.3f}".format(
                float(inside.sum()) / max(1, int(gaps.sum())))

    bx, by = match.bx[rows], match.by[rows]
    off = (bx < 0) | (bx > length) | (by < 0) | (by > width)
    out["seen ball outside the pitch lines (share of seen)"] = "{:.4f}".format(
        float(np.nansum(off)) / max(1, int(seen[rows].sum())))

    # Jitter: spread of the second difference, which a smooth path keeps near zero.
    second = match.bx[2:] - 2 * match.bx[1:-1] + match.bx[:-2]
    out["ball jitter proxy, metres (median |2nd difference|)"] = "{:.4f}".format(
        float(np.nanmedian(np.abs(second))))
    step = np.hypot(np.diff(match.bx), np.diff(match.by)) * fps
    out["ball speed, m/s (p10 p25 p50 p75 p90 p99)"] = _pct(step[~np.isnan(step)])

    # A tracker that fills short losses draws straight lines at constant speed. How much of the
    # seen ball is exactly that tells us how much of it is filled in rather than detected.
    bend = np.hypot(second, match.by[2:] - 2 * match.by[1:-1] + match.by[:-2])
    pace = step[1:]
    ok = ~np.isnan(bend) & ~np.isnan(pace)
    out["ball bend per frame, metres (p10 p25 p50 p75 p90 p99)"] = _pct(bend[ok])
    out["ball exactly still (share of seen)"] = "{:.3f}".format(
        float((pace[ok] < 0.01).mean()) if ok.any() else 0.0)
    for limit in (0.001, 0.01):
        straight = _stretches(ok & (bend < limit) & (pace >= 0.5))
        out["ball moving dead straight, bend < {} m (share of seen)".format(limit)] =             "{:.3f}".format(float(straight.sum()) / max(1, int(ok.sum())))
        out["  ... stretch length, frames, bend < {} m (p10 p25 p50 p75 p90 p99)".format(limit)] =             _pct(straight)

    # How the ball behaves when it is seen: a ball that is often motionless or often right at a
    # player's feet says a lot about what the feed shows during stoppages.
    step_rows = np.hypot(match.bx[rows] - match.bx[rows - 1], match.by[rows] - match.by[rows - 1]) * fps
    moving = ~np.isnan(step_rows)
    out["seen ball slower than 0.5 m/s (share of consecutive sightings)"] = "{:.3f}".format(
        float((step_rows[moving] < 0.5).mean()) if moving.any() else 0.0)
    out["seen ball slower than 2 m/s (share of consecutive sightings)"] = "{:.3f}".format(
        float((step_rows[moving] < 2.0).mean()) if moving.any() else 0.0)

    present = ~np.isnan(match.px[rows])
    count = present.sum(axis=1)
    out["players tracked per frame (p10 p25 p50 p75 p90 p99)"] = _pct(count)
    out["players tracked per frame: mean"] = "{:.2f}".format(count.mean())
    out["frames with more than 22 tracked"] = "{:.4f}".format(float((count > 22).mean()))
    for team in (0, 1):
        cols = match.pteam == team
        out["team {} tracked per frame: mean".format(team)] = "{:.2f}".format(
            present[:, cols].sum(axis=1).mean())
    out["track slots"] = "{} (team0 {}, team1 {})".format(
        len(match.pteam), int((match.pteam == 0).sum()), int((match.pteam == 1).sum()))

    lengths, holes = [], []
    for a, b in match.runs:
        there = ~np.isnan(match.px[a:b])
        for slot in range(there.shape[1]):
            lengths.append(_stretches(there[:, slot]))
            inner = _stretches(~there[:, slot])
            holes.append(inner)
    lengths, holes = np.concatenate(lengths), np.concatenate(holes)
    out["player track pieces: count"] = "{}".format(len(lengths))
    out["player track piece length, frames (p10 p25 p50 p75 p90 p99)"] = _pct(lengths)
    out["player hole length, frames (p10 p25 p50 p75 p90 p99)"] = _pct(holes)

    second = match.px[2:] - 2 * match.px[1:-1] + match.px[:-2]
    out["player jitter proxy, metres (median |2nd difference|)"] = "{:.4f}".format(
        float(np.nanmedian(np.abs(second[::10]))))
    bend = np.hypot(second[::10], (match.py[2:] - 2 * match.py[1:-1] + match.py[:-2])[::10])
    out["player bend per frame, metres (p10 p25 p50 p75 p90 p99)"] = _pct(bend[~np.isnan(bend)])

    near = np.hypot(match.px[rows] - match.bx[rows, None], match.py[rows] - match.by[rows, None])
    nearest = np.nanmin(np.where(np.isnan(near), np.inf, near), axis=1)
    nearest = nearest[np.isfinite(nearest)]
    out["distance ball to nearest player, metres (p10 p25 p50 p75 p90 p99)"] = _pct(nearest)
    for limit in (0.5, 1.0, 2.0):
        out["ball within {} m of a player (share of seen)".format(limit)] = "{:.3f}".format(
            float((nearest < limit).mean()) if len(nearest) else 0.0)

    # Player movement, which is how stoppages show up without any ball at all.
    from .features import player_speeds
    speeds = player_speeds(match.px, match.py, fps)[rows]
    median = np.nanmedian(speeds, axis=1)
    median = median[~np.isnan(median)]
    out["median player speed per frame, m/s (p10 p25 p50 p75 p90 p99)"] = _pct(median)
    out["frames where players are walking, median < 1.2 m/s (share)"] = "{:.3f}".format(
        float((median < 1.2).mean()) if len(median) else 0.0)
    seen_rows = seen[rows]
    walking = np.nan_to_num(np.nanmedian(speeds, axis=1), nan=9.0) < 1.2
    out["ball unseen share while players are walking"] = "{:.3f}".format(
        float(1.0 - seen_rows[walking].mean()) if walking.any() else 0.0)
    out["ball unseen share while players are not walking"] = "{:.3f}".format(
        float(1.0 - seen_rows[~walking].mean()) if (~walking).any() else 0.0)
    return out


def print_stats(title, stats):
    print("\n" + title)
    for key, value in stats.items():
        print("  {:<70} {}".format(key, value))
