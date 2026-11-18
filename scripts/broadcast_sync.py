"""Line the broadcast tracking up with the tactical tracking, and say what it could add.

The broadcast recording is on its own timeline, with cuts and replays. Both feeds give the ball
in pitch coordinates, so where both see the ball in live play the two paths must coincide. For
every stretch of the broadcast feed this finds the shift onto the tactical timeline at which
the two ball paths agree best.

Prints aggregates only: counts, shares, percentiles. No rows, frames or positions.

    python scripts/broadcast_sync.py
"""

import argparse
import os
import sys
import warnings

import numpy as np
from scipy.signal import fftconvolve

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")
from possession.io_atd import data_dir, load_competition  # noqa: E402
from possession import io_atd  # noqa: E402
from possession.features import player_speeds, remove_ball_teleports  # noqa: E402
from possession.match import Match  # noqa: E402

CACHE = os.path.join(ROOT, "cache")
WINDOW = 1500        # broadcast frames per stretch (60 s)
MIN_PAIRS = 150      # frames with the ball in both feeds needed to trust a stretch
CLOSE = 3.0          # metres: the two feeds agree on the ball
LIVE = 1.5           # metres: broadcast players sit on tactical players


def cached(name, build):
    path = os.path.join(CACHE, name)
    if os.path.isfile(path):
        return Match.load(path)
    match = build()
    os.makedirs(CACHE, exist_ok=True)
    match.save(path)
    return match


def load_broadcast(folder):
    stem = os.path.join(folder, "broadcast", "BAR-ATH_broadcast")
    return io_atd.load_feed(stem + "_atd.xml", stem + "_atd.txt")


def lag_scores(tx, ty, bx, by):
    """For every shift L (tactical frame = broadcast frame + L): the number of frames with the
    ball in both feeds, and the mean squared distance between the two balls on them."""
    vt = (~np.isnan(tx)).astype(np.float64)
    vb = (~np.isnan(bx)).astype(np.float64)
    tx0, ty0 = np.nan_to_num(tx).astype(np.float64), np.nan_to_num(ty).astype(np.float64)
    bx0, by0 = np.nan_to_num(bx).astype(np.float64), np.nan_to_num(by).astype(np.float64)

    def corr(t, b):  # sum over broadcast frames i of b[i] * t[i + L]
        return fftconvolve(t, b[::-1], mode="full")

    pairs = corr(vt, vb)
    total = (corr(vt, vb * (bx0 ** 2 + by0 ** 2)) + corr(vt * (tx0 ** 2 + ty0 ** 2), vb)
             - 2.0 * corr(tx0, bx0) - 2.0 * corr(ty0, by0))
    lags = np.arange(-(len(bx) - 1), len(tx))
    return lags, pairs, total / np.maximum(pairs, 1.0)


def player_agreement(tactical, broadcast, shift, chunk=4000):
    """Per broadcast frame: median distance from its players to the nearest tactical player of
    the same team at the frame it maps to. inf where it cannot be checked."""
    n = broadcast.n_frames
    out = np.full(n, np.inf, dtype=np.float32)
    first, last = max(0, -shift), min(n, tactical.n_frames - shift)
    for a in range(first, last, chunk):
        b = min(last, a + chunk)
        worst = np.full((b - a, len(broadcast.pteam)), np.nan, dtype=np.float32)
        for team in (0, 1):
            mine = np.flatnonzero(broadcast.pteam == team)
            theirs = np.flatnonzero(tactical.pteam == team)
            dx = broadcast.px[a:b][:, mine, None] - tactical.px[a + shift:b + shift][:, None, theirs]
            dy = broadcast.py[a:b][:, mine, None] - tactical.py[a + shift:b + shift][:, None, theirs]
            dist = np.hypot(dx, dy)
            nearest = np.nanmin(np.where(np.isnan(dist), np.inf, dist), axis=2)
            nearest[np.isnan(broadcast.px[a:b][:, mine])] = np.nan
            worst[:, mine] = nearest
        enough = (~np.isnan(worst)).sum(axis=1) >= 3
        med = np.nanmedian(np.where(np.isfinite(worst), worst, np.nan), axis=1)
        out[a:b] = np.where(enough & np.isfinite(med), med, np.inf)
    return out


def pct(values, points=(10, 25, 50, 75, 90)):
    return " ".join("{:.4g}".format(v) for v in np.percentile(values, points)) if len(values) else "n/a"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None)
    args = parser.parse_args()
    folder = data_dir(args.data)

    tactical = cached("atd-tactical.npz", lambda: load_competition(folder))
    broadcast = cached("atd-broadcast.npz", lambda: load_broadcast(folder))
    tx, ty = remove_ball_teleports(tactical.bx, tactical.by, tactical.fps)
    bx, by = remove_ball_teleports(broadcast.bx, broadcast.by, broadcast.fps)

    print("BROADCAST FEED")
    print("  frames {}  ({:.1f} min at {} fps)   tactical frames {}".format(
        broadcast.n_frames, broadcast.n_frames / broadcast.fps / 60.0, broadcast.fps,
        tactical.n_frames))
    print("  ball seen share {:.3f}".format(float((~np.isnan(bx)).mean())))
    count = (~np.isnan(broadcast.px)).sum(axis=1)
    print("  players tracked per frame (p10 p25 p50 p75 p90): {}".format(pct(count)))
    print("  frames with no player at all: {:.3f}".format(float((count == 0).mean())))
    print("  track slots {}".format(len(broadcast.pteam)))

    # One shift for the whole feed first, as a sanity check that the idea works at all.
    lags, pairs, mse = lag_scores(tx, ty, bx, by)
    ok = pairs >= 5000
    best = int(np.argmin(np.where(ok, mse, np.inf)))
    print("\nBEST SINGLE SHIFT FOR THE WHOLE FEED")
    print("  tactical frame = broadcast frame + {}".format(int(lags[best])))
    print("  frames with the ball in both feeds at that shift: {}".format(int(pairs[best])))
    print("  root-mean-square distance between the two balls: {:.2f} m".format(
        float(np.sqrt(max(mse[best], 0.0)))))
    ranked = np.sort(np.sqrt(np.maximum(mse[ok], 0.0)))
    print("  for comparison, the same at other shifts (p1 p10 p50): {}".format(
        pct(ranked, (1, 10, 50))))

    # Then stretch by stretch, because the broadcast has cuts and replays.
    shifts, quality, used = [], [], []
    mapping = np.full(broadcast.n_frames, -1, dtype=np.int64)
    for start in range(0, broadcast.n_frames, WINDOW):
        stop = min(broadcast.n_frames, start + WINDOW)
        if int((~np.isnan(bx[start:stop])).sum()) < MIN_PAIRS:
            continue
        lags_w, pairs_w, mse_w = lag_scores(tx, ty, bx[start:stop], by[start:stop])
        ok = pairs_w >= MIN_PAIRS
        if not ok.any():
            continue
        k = int(np.argmin(np.where(ok, mse_w, np.inf)))
        shift = int(lags_w[k]) - start        # tactical = broadcast + shift
        rms = float(np.sqrt(max(mse_w[k], 0.0)))
        shifts.append(shift)
        quality.append(rms)
        used.append((start, stop))
    shifts, quality = np.asarray(shifts), np.asarray(quality)
    print("\nSTRETCH BY STRETCH ({} s stretches)".format(WINDOW // broadcast.fps))
    print("  stretches with enough ball to try: {} of {}".format(
        len(shifts), int(np.ceil(broadcast.n_frames / float(WINDOW)))))
    print("  ball distance at each stretch's best shift, m (p10 p25 p50 p75 p90): {}".format(
        pct(quality)))
    good = quality < 2.0
    print("  stretches that line up well (under 2 m): {}".format(int(good.sum())))
    values, counts = np.unique(shifts[good], return_counts=True)
    order = np.argsort(-counts)
    # Shifts within a few frames of each other are the same alignment.
    groups = []
    for v, c in zip(values[order], counts[order]):
        for g in groups:
            if abs(g[0] - v) <= 5:
                g[1] += c
                break
        else:
            groups.append([int(v), int(c)])
    print("  distinct shifts among them: {}".format(len(groups)))
    for shift, c in groups[:6]:
        print("    shift {:>8}  in {} stretches".format(shift, c))
    # Each shift found serves a whole part of the match. A broadcast frame is taken as live
    # play, lined up by a given shift, when its players stand where the tactical feed has
    # players of the same team at the frame it maps to -- that also throws out replays.
    candidates = [g[0] for g in groups if g[1] >= 3]
    best_error = np.full(broadcast.n_frames, np.inf, dtype=np.float32)
    for shift in candidates:
        error = player_agreement(tactical, broadcast, shift)
        better = error < best_error
        best_error[better] = error[better]
        mapping[better] = np.flatnonzero(better) + shift
    live_frame = best_error < LIVE
    mapping[~live_frame] = -1
    finite = best_error[np.isfinite(best_error)]
    print("\nFRAME BY FRAME, CHECKED AGAINST THE PLAYERS")
    print("  broadcast frames that could be checked (3+ players, maps inside the tactical "
          "video): {:.3f}".format(len(finite) / float(broadcast.n_frames)))
    print("  median distance from a broadcast player to the nearest tactical player of the "
          "same team, m (p10 p25 p50 p75 p90): {}".format(pct(finite)))
    print("  broadcast frames accepted as live and lined up (under {} m): {:.3f}".format(
        LIVE, float(live_frame.mean())))

    # What the broadcast would add on the graded tactical frames.
    extra_x = np.full(tactical.n_frames, np.nan, dtype=np.float32)
    extra_y = np.full(tactical.n_frames, np.nan, dtype=np.float32)
    source = np.flatnonzero(live_frame & ~np.isnan(bx))
    extra_x[mapping[source]] = bx[source]
    extra_y[mapping[source]] = by[source]
    covered = np.zeros(tactical.n_frames, dtype=bool)
    covered[mapping[live_frame]] = True

    rows = np.concatenate([np.arange(a, b) for a, b in tactical.runs])
    seen_t = ~np.isnan(tx[rows])
    seen_b = ~np.isnan(extra_x[rows])
    speeds = player_speeds(tactical.px, tactical.py, tactical.fps)[rows]
    walking = np.nan_to_num(np.nanmedian(speeds, axis=1), nan=9.0) < 1.2
    print("\nON THE GRADED TACTICAL FRAMES")
    print("  covered by a broadcast stretch that lines up: {:.3f}".format(
        float(covered[rows].mean())))
    print("  ball seen: tactical {:.3f}   broadcast (lined up) {:.3f}   either {:.3f}".format(
        float(seen_t.mean()), float(seen_b.mean()), float((seen_t | seen_b).mean())))
    print("  tactical ball unseen, share where the broadcast has it: {:.3f}".format(
        float(seen_b[~seen_t].mean())))
    for name, mask in (("players walking", walking), ("players not walking", ~walking)):
        gap = mask & ~seen_t
        print("  {:<20} tactical unseen {:.3f}  -> still unseen with broadcast {:.3f}".format(
            name, float((~seen_t)[mask].mean()), float((~seen_t & ~seen_b)[mask].mean())))
        del gap
    both = seen_t & seen_b
    dist = np.hypot(tx[rows][both] - extra_x[rows][both], ty[rows][both] - extra_y[rows][both])
    print("  both see the ball on {} frames; distance between them, m "
          "(p10 p25 p50 p75 p90): {}".format(int(both.sum()), pct(dist)))
    print("  share of those within {:.0f} m: {:.3f}".format(CLOSE, float((dist < CLOSE).mean())))

    # Unseen stretches of the tactical ball before and after adding the broadcast ball.
    def stretches(mask):
        edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
        return np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)

    before = np.concatenate([stretches(np.isnan(tx[a:b])) for a, b in tactical.runs])
    after = np.concatenate([stretches(np.isnan(tx[a:b]) & np.isnan(extra_x[a:b]))
                            for a, b in tactical.runs])
    print("  unseen stretches before: {}  lengths (p10 p25 p50 p75 p90): {}".format(
        len(before), pct(before)))
    print("  unseen stretches after:  {}  lengths (p10 p25 p50 p75 p90): {}".format(
        len(after), pct(after)))

    # Where the two feeds disagree about the ball. A ball in the air is drawn on the ground
    # where the camera's line of sight lands, further from the camera than it really is, and by
    # a different amount for each camera -- so disagreement mostly across the pitch, always in
    # the same direction, is the signature of airborne balls rather than of random error.
    dx = (extra_x[rows] - tx[rows])[both]
    dy = (extra_y[rows] - ty[rows])[both]
    print("\nDISAGREEMENT BETWEEN THE TWO BALLS (broadcast minus tactical)")
    for name, mask in (("under 1 m apart", dist < 1.0), ("1 to 3 m apart", (dist >= 1.0) & (dist < 3.0)),
                       ("over 3 m apart", dist >= 3.0)):
        if mask.any():
            print("  {:<16} share {:.3f}   median |along| {:.2f} m  |across| {:.2f} m   "
                  "across is positive on {:.2f}".format(
                      name, float(mask.mean()), float(np.median(np.abs(dx[mask]))),
                      float(np.median(np.abs(dy[mask]))), float((dy[mask] > 0).mean())))
    width = tactical.pitch[1]
    seen_y = ty[rows][seen_t]
    seen_x = tx[rows][seen_t]
    length = tactical.pitch[0]
    print("  tactical ball beyond the touchlines: y<0 {:.4f}   y>width {:.4f}   "
          "beyond the goal lines: {:.4f}".format(
              float((seen_y < 0).mean()), float((seen_y > width).mean()),
              float(((seen_x < 0) | (seen_x > length)).mean())))
    np.save(os.path.join(CACHE, "broadcast_mapping.npy"), mapping)
