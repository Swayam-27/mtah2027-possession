"""Find the ball in the tactical video where the tracking lost it, with a pretrained detector.

Runs entirely on this computer: the video never leaves it. Uses a public pretrained detector
(Ultralytics YOLOv8, COCO class "sports ball") and the organisers' camera-to-pitch homography to
turn each detection into a pitch position. Prints aggregate numbers only.

    python scripts/video_ball.py --sample 2000          # quick test: speed and quality
    python scripts/video_ball.py --analyse              # re-analyse the saved quick test, no video
    python scripts/video_ball.py                        # every graded frame with no tracked ball
    python scripts/video_ball.py --clip 60 --start 20000  # 60 s overlay video for a person to check

Output: cache/video_ball.npz (full run) or cache/video_ball_sample.npz (quick test): every
candidate detection with its frame, pitch position in metres, confidence and box size. Derived
from the match video, so it stays local like the data itself (cache/ is never committed).
"""

import argparse
import os
import sys
import time
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")
from possession.io_atd import data_dir, load_competition  # noqa: E402
from possession.features import remove_ball_teleports  # noqa: E402

BALL_CLASS = 32          # "sports ball" in COCO
OUT = os.path.join(ROOT, "cache", "video_ball.npz")
SAMPLE_OUT = os.path.join(ROOT, "cache", "video_ball_sample.npz")


def homographies(folder):
    import pandas as pd
    table = pd.read_csv(os.path.join(folder, "tactical", "BAR-ATH_tactical_homography.csv"))
    frames = table["frame"].to_numpy()
    mats = table[["h{}{}".format(i, j) for i in range(3) for j in range(3)]].to_numpy()
    out = np.full((int(frames.max()) + 1, 3, 3), np.nan)
    out[frames] = mats.reshape(-1, 3, 3)
    return out


def to_pitch(matrix, px, py, pitch):
    """Pixel -> metres on the pitch (the inverse of the organisers' pitch -> pixel matrix)."""
    inv = np.linalg.inv(matrix)
    w = inv[2, 0] * px + inv[2, 1] * py + inv[2, 2]
    u = (inv[0, 0] * px + inv[0, 1] * py + inv[0, 2]) / w
    v = (inv[1, 0] * px + inv[1, 1] * py + inv[1, 2]) / w
    return u * pitch[0], v * pitch[1]


def pct(values, points=(10, 25, 50, 75, 90)):
    return " ".join("{:.3g}".format(v) for v in np.percentile(values, points)) if len(values) else "n/a"


def best_per_frame(cand, score):
    """Index of the highest-scoring candidate on each frame."""
    order = np.lexsort((-score, cand["frame"]))
    first = np.ones(len(order), dtype=bool)
    first[1:] = cand["frame"][order][1:] != cand["frame"][order][:-1]
    return order[first]


def report(match, tx, ty, seen, graded, wanted, cand):
    n = match.n_frames
    want = np.zeros(n, dtype=bool)
    want[wanted] = True
    f, x, y, conf = cand["frame"], cand["x"], cand["y"], cand["conf"]
    w, h = cand["w"], cand["h"]
    looked_seen, looked_missing = want & seen & graded, want & ~seen & graded
    has = np.zeros(n, dtype=bool)
    has[f] = True
    print("\nVIDEO BALL DETECTION ({} frames looked at, {} candidates, {:.2f} per frame)".format(
        len(wanted), len(f), len(f) / max(len(wanted), 1)))
    print("  found a ball on {:.3f} of frames where the tracking has one".format(
        float(has[looked_seen].mean()) if looked_seen.any() else 0.0))
    print("  found a ball on {:.3f} of frames where the tracking has none".format(
        float(has[looked_missing].mean()) if looked_missing.any() else 0.0))
    if not len(f):
        return

    on_seen = seen[f]
    dist_all = np.full(len(f), np.nan)
    dist_all[on_seen] = np.hypot(x[on_seen] - tx[f[on_seen]], y[on_seen] - ty[f[on_seen]])

    best = best_per_frame(cand, conf)
    b = best[seen[f[best]]]
    dist = dist_all[b]
    print("\n  MOST CONFIDENT CANDIDATE, where both see the ball ({} frames)".format(len(b)))
    print("  distance video vs tracking, m (p10 p25 p50 p75 p90): {}".format(pct(dist)))
    print("  ... within 1 m {:.3f}, within 2 m {:.3f}, within 5 m {:.3f}".format(
        float((dist < 1).mean()), float((dist < 2).mean()), float((dist < 5).mean())))

    # Is the real ball among the candidates at all, just outranked?
    close = on_seen & (dist_all < 2)
    any_close = np.zeros(n, dtype=bool)
    any_close[f[close]] = True
    print("  ANY candidate within 2 m of the tracking ball: {:.3f} of these frames".format(
        float(any_close[f[b]].mean())))

    # Box size: are the right detections a different size from the wrong ones?
    right, wrong = on_seen & (dist_all < 2), on_seen & (dist_all > 5)
    print("\n  box width px (p10 p25 p50 p75 p90): right {} | wrong {}".format(
        pct(w[right]), pct(w[wrong])))
    print("  box height px                       : right {} | wrong {}".format(
        pct(h[right]), pct(h[wrong])))
    print("  aspect w/h                          : right {} | wrong {}".format(
        pct(w[right] / h[right]), pct(w[wrong] / h[wrong])))
    print("  confidence                          : right {} | wrong {}".format(
        pct(conf[right]), pct(conf[wrong])))
    # Near a tracked player (head/boot) or far from everyone (line marks, logos)?
    near_player = np.zeros(len(f), dtype=bool)
    for i in np.flatnonzero(right | wrong):
        d = np.hypot(match.px[f[i]] - x[i], match.py[f[i]] - y[i])
        near_player[i] = bool(np.nanmin(d) < 1.5) if np.isfinite(d).any() else False
    print("  within 1.5 m of a tracked player    : right {:.3f} | wrong {:.3f}".format(
        float(near_player[right].mean()) if right.any() else 0.0,
        float(near_player[wrong].mean()) if wrong.any() else 0.0))
    for name, lo, hi in (("in the pitch", 0, 0), ("beyond a line", 0.01, 99)):
        out = np.maximum.reduce([-x, x - match.pitch[0], -y, y - match.pitch[1]]) > 0
        sel = out if lo else ~out
        print("  share {:<14}                : right {:.3f} | wrong {:.3f}".format(
            name, float(sel[right].mean()) if right.any() else 0.0,
            float(sel[wrong].mean()) if wrong.any() else 0.0))

    # Convention checks: a time shift or a mirrored axis would show as a much better agreement.
    fb = f[b]
    print("\n  CHECKS (share of most-confident candidates within 2 m)")
    for label, xx, yy in (("as is", x[b], y[b]), ("x mirrored", match.pitch[0] - x[b], y[b]),
                          ("y mirrored", x[b], match.pitch[1] - y[b])):
        print("    {:<11} {:.3f}".format(label, float((np.hypot(xx - tx[fb], yy - ty[fb]) < 2).mean())))
    shifts = []
    for s in range(-75, 76):
        g = fb + s
        ok = (g >= 0) & (g < n)
        ok[ok] &= seen[g[ok]]
        if ok.sum() > 50:
            shifts.append((float((np.hypot(x[b][ok] - tx[g[ok]], y[b][ok] - ty[g[ok]]) < 2).mean()), s))
    shifts.sort(reverse=True)
    print("    best frame shifts: " + ", ".join("{:+d}: {:.3f}".format(s, r) for r, s in shifts[:4]))


def keep(match, cand, margin, min_conf, max_box, max_aspect):
    """Which candidates survive a filter: inside the pitch (+margin m), confident, ball-shaped."""
    x, y = cand["x"], cand["y"]
    beyond = np.maximum.reduce([-x, x - match.pitch[0], -y, y - match.pitch[1]])
    aspect = cand["w"] / np.maximum(cand["h"], 1e-6)
    return ((beyond <= margin) & (cand["conf"] >= min_conf)
            & (np.maximum(cand["w"], cand["h"]) <= max_box)
            & (aspect <= max_aspect) & (aspect >= 1.0 / max_aspect))


def filter_table(match, tx, ty, seen, graded, wanted, cand):
    """Accuracy (where both see the ball) against how much of the missing ball each filter fills."""
    want = np.zeros(match.n_frames, dtype=bool)
    want[wanted] = True
    n_seen, n_missing = int((want & seen & graded).sum()), int((want & ~seen & graded).sum())
    print("\n  FILTERS (most confident surviving candidate per frame)")
    print("  {:>6} {:>5} {:>4} {:>6} | {:>14} {:>9} {:>9} | {:>15}".format(
        "margin", "conf", "box", "aspect", "found (seen)", "within 2m", "within 5m",
        "found (missing)"))
    rows = []
    for margin in (5.0, 2.0, 1.0, 0.0, -1.0):
        for min_conf in (0.10, 0.15, 0.20, 0.30, 0.40):
            for max_box, max_aspect in ((99.0, 99.0), (16.0, 1.4)):
                sel = keep(match, cand, margin, min_conf, max_box, max_aspect)
                if not sel.any():
                    continue
                sub = {k: v[sel] for k, v in cand.items()}
                b = best_per_frame(sub, sub["conf"])
                f = sub["frame"][b]
                on_seen = seen[f] & graded[f]
                d = np.hypot(sub["x"][b][on_seen] - tx[f[on_seen]], sub["y"][b][on_seen] - ty[f[on_seen]])
                rows.append((margin, min_conf, max_box, max_aspect, on_seen.sum() / max(n_seen, 1),
                             float((d < 2).mean()) if len(d) else 0.0,
                             float((d < 5).mean()) if len(d) else 0.0,
                             int((~seen[f] & graded[f]).sum()) / max(n_missing, 1)))
    for r in rows:
        print("  {:>6.1f} {:>5.2f} {:>4.0f} {:>6.1f} | {:>14.3f} {:>9.3f} {:>9.3f} | {:>15.3f}".format(*r))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None)
    parser.add_argument("--weights", default=os.path.join(ROOT, "models", "yolov8n.pt"))
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--sample", type=int, default=0,
                        help="test on N frames: half with a tracked ball, half without")
    parser.add_argument("--analyse", action="store_true",
                        help="re-analyse the saved quick test without reading the video")
    parser.add_argument("--clip", type=float, default=0.0, help="write an overlay clip of N s")
    parser.add_argument("--start", type=int, default=20000, help="first frame of the clip")
    args = parser.parse_args()

    folder = data_dir(args.data)
    match = load_competition(folder)
    tx, ty = remove_ball_teleports(match.bx, match.by, match.fps)
    graded = np.zeros(match.n_frames, dtype=bool)
    for a, b in match.runs:
        graded[a:b] = True
    seen = ~np.isnan(tx)

    if args.analyse:
        saved = np.load(SAMPLE_OUT)
        cand = {k: saved[k] for k in ("frame", "x", "y", "conf", "w", "h")}
        report(match, tx, ty, seen, graded, saved["wanted"], cand)
        filter_table(match, tx, ty, seen, graded, saved["wanted"], cand)
        sys.exit(0)

    import cv2
    from ultralytics import YOLO

    H = homographies(folder)
    if args.clip:
        wanted = np.arange(args.start, min(match.n_frames, args.start + int(args.clip * 25)))
    elif args.sample:
        rng = np.random.default_rng(0)
        half = args.sample // 2
        wanted = np.sort(np.concatenate([
            rng.choice(np.flatnonzero(graded & seen), half, replace=False),
            rng.choice(np.flatnonzero(graded & ~seen), half, replace=False)]))
    else:
        # Every graded frame without a tracked ball, plus every 25th frame with one, to measure
        # how well the detector agrees with the tracking where both exist.
        wanted = np.flatnonzero(graded & (~seen | (np.arange(match.n_frames) % 25 == 0)))
    want = np.zeros(match.n_frames, dtype=bool)
    want[wanted] = True

    model = YOLO(args.weights)
    video = cv2.VideoCapture(os.path.join(folder, "tactical", "BAR-ATH_tactical.mp4"))
    print("video: {:.0f} frames at {:.2f} fps (tracking has {})".format(
        video.get(cv2.CAP_PROP_FRAME_COUNT), video.get(cv2.CAP_PROP_FPS), match.n_frames))
    writer = None
    if args.clip:
        os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)
        clip_path = os.path.join(ROOT, "outputs", "video_ball_check.mp4")
        writer = cv2.VideoWriter(clip_path, cv2.VideoWriter_fourcc(*"mp4v"), 25, (1920, 1080))

    cand = {k: [] for k in ("frame", "x", "y", "conf", "w", "h")}
    started, done, last = time.time(), 0, int(wanted.max())
    frame = -1
    while frame < last:
        frame += 1
        if not want[frame]:
            if not video.grab():
                break
            continue
        ok, image = video.read()
        if not ok:
            break
        result = model.predict(image, imgsz=args.imgsz, conf=args.conf, classes=[BALL_CLASS],
                               verbose=False)[0]
        best = None
        if len(result.boxes) and not np.isnan(H[frame]).any():
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            for (x0, y0, x1, y1), score in zip(boxes, scores):
                # The ball touches the ground at the bottom of its box, not its centre.
                x, y = to_pitch(H[frame], (x0 + x1) / 2.0, y1, match.pitch)
                if -5 < x < match.pitch[0] + 5 and -5 < y < match.pitch[1] + 5:
                    for key, value in zip(("frame", "x", "y", "conf", "w", "h"),
                                          (frame, x, y, float(score), x1 - x0, y1 - y0)):
                        cand[key].append(value)
                    if best is None or score > best[0]:
                        best = (score, (x0, y0, x1, y1))
        if writer is not None:
            if best is not None:
                x0, y0, x1, y1 = (int(v) for v in best[1])
                cv2.rectangle(image, (x0 - 6, y0 - 6), (x1 + 6, y1 + 6), (0, 0, 255), 2)
            cv2.putText(image, "frame {}  tracking ball: {}".format(
                frame, "yes" if seen[frame] else "NO"), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                (0, 255, 255), 3)
            writer.write(image)
        done += 1
        if done % 500 == 0:
            rate = done / (time.time() - started)
            print("  {} of {} frames ({:.1f} per s, about {:.0f} min left)".format(
                done, len(wanted), rate, (len(wanted) - done) / rate / 60.0), flush=True)
    if writer is not None:
        writer.release()
        print("wrote {} -- for a person to watch; never upload it".format(clip_path))

    cand = {k: np.array(v, dtype=np.int64 if k == "frame" else np.float64) for k, v in cand.items()}
    print("{:.1f} frames per s".format(done / (time.time() - started)))
    report(match, tx, ty, seen, graded, wanted, cand)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    path = OUT if not (args.clip or args.sample) else SAMPLE_OUT
    np.savez_compressed(path, wanted=wanted, **cand)
    print("saved {} (local only)".format(path))
