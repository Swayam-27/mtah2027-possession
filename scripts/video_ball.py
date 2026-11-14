"""Find the ball in the tactical video where the tracking lost it, with a pretrained detector.

Runs entirely on this computer: the video never leaves it. Uses a public pretrained detector
(Ultralytics YOLOv8, COCO class "sports ball") and the organisers' camera-to-pitch homography to
turn each detection into a pitch position. Prints aggregate numbers only.

    python scripts/video_ball.py --sample 2000          # quick test: speed and quality
    python scripts/video_ball.py                        # every graded frame with no tracked ball
    python scripts/video_ball.py --clip 60 --start 20000  # 60 s overlay video for a person to check

Output: cache/video_ball.npz (frame, x, y in metres, confidence). Derived from the match video,
so it stays local like the data itself (cache/ is never committed).
"""

import argparse
import os
import sys
import time
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
warnings.filterwarnings("ignore")
from atd_stats import data_dir, load_competition  # noqa: E402
from possession.features import remove_ball_teleports  # noqa: E402

BALL_CLASS = 32          # "sports ball" in COCO
OUT = os.path.join(ROOT, "cache", "video_ball.npz")


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None)
    parser.add_argument("--weights", default=os.path.join(ROOT, "models", "yolov8n.pt"))
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--sample", type=int, default=0,
                        help="test on N frames: half with a tracked ball, half without")
    parser.add_argument("--clip", type=float, default=0.0, help="write an overlay clip of N s")
    parser.add_argument("--start", type=int, default=20000, help="first frame of the clip")
    args = parser.parse_args()

    import cv2
    from ultralytics import YOLO

    folder = data_dir(args.data)
    match = load_competition(folder)
    H = homographies(folder)
    tx, ty = remove_ball_teleports(match.bx, match.by, match.fps)
    graded = np.zeros(match.n_frames, dtype=bool)
    for a, b in match.runs:
        graded[a:b] = True
    seen = ~np.isnan(tx)

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
    writer = None
    if args.clip:
        os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)
        clip_path = os.path.join(ROOT, "outputs", "video_ball_check.mp4")
        writer = cv2.VideoWriter(clip_path, cv2.VideoWriter_fourcc(*"mp4v"), 25, (1920, 1080))

    frames, xs, ys, confs = [], [], [], []
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
                x, y = to_pitch(H[frame], (x0 + x1) / 2.0, (y0 + y1) / 2.0, match.pitch)
                if -5 < x < match.pitch[0] + 5 and -5 < y < match.pitch[1] + 5:
                    if best is None or score > best[2]:
                        best = (x, y, float(score), (x0, y0, x1, y1))
        if best is not None:
            frames.append(frame)
            xs.append(best[0])
            ys.append(best[1])
            confs.append(best[2])
        if writer is not None:
            if best is not None:
                x0, y0, x1, y1 = (int(v) for v in best[3])
                cv2.rectangle(image, (x0 - 6, y0 - 6), (x1 + 6, y1 + 6), (0, 0, 255), 2)
            cv2.putText(image, "frame {}  tracking ball: {}".format(
                frame, "yes" if seen[frame] else "NO"), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                (0, 255, 255), 3)
            writer.write(image)
        done += 1
        if done % 2000 == 0:
            rate = done / (time.time() - started)
            print("  {} of {} frames ({:.1f} per s, about {:.0f} min left)".format(
                done, len(wanted), rate, (len(wanted) - done) / rate / 60.0), flush=True)
    if writer is not None:
        writer.release()
        print("wrote {} -- for a person to watch; never upload it".format(clip_path))

    frames = np.array(frames, dtype=np.int64)
    xs, ys, confs = np.array(xs), np.array(ys), np.array(confs)
    elapsed = time.time() - started
    print("\nVIDEO BALL DETECTION ({} frames looked at, {:.1f} per s)".format(done, done / elapsed))
    looked_seen = want & seen & graded
    looked_missing = want & ~seen & graded
    found = np.zeros(match.n_frames, dtype=bool)
    found[frames] = True
    print("  found a ball on {:.3f} of frames where the tracking has one".format(
        float(found[looked_seen].mean()) if looked_seen.any() else 0.0))
    print("  found a ball on {:.3f} of frames where the tracking has none".format(
        float(found[looked_missing].mean()) if looked_missing.any() else 0.0))
    both = seen[frames]
    if both.any():
        dist = np.hypot(xs[both] - tx[frames[both]], ys[both] - ty[frames[both]])
        print("  where both see it: distance video vs tracking, m (p10 p25 p50 p75 p90): {}".format(
            pct(dist)))
        print("  ... within 1 m {:.3f}, within 2 m {:.3f}, within 5 m {:.3f}".format(
            float((dist < 1).mean()), float((dist < 2).mean()), float((dist < 5).mean())))
        for low, high in ((0.15, 0.3), (0.3, 0.5), (0.5, 1.01)):
            sel = (confs[both] >= low) & (confs[both] < high)
            if sel.any():
                print("  ... confidence {:.2f}-{:.2f}: n {}, within 2 m {:.3f}".format(
                    low, high, int(sel.sum()), float((dist[sel] < 2).mean())))
    if not args.clip and not args.sample:
        np.savez_compressed(OUT, frame=frames, x=xs, y=ys, conf=confs)
        print("saved {} (local only)".format(OUT))
