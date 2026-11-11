"""Compare a submission with the Smart Tagging, in summary numbers.

The Smart Tagging is Metrica's automatic annotation of the same match: set pieces (restarts) and
tactical phases with a team. It is not ground truth, but it is an independent view of the real
match, which the practice matches cannot give. Two questions:

  - Do our dead-ball stretches end where the tagged restarts are?
  - While a phase is tagged with a team, do we give the ball to that team?

Prints aggregates only (counts, shares, percentiles).

    python scripts/tagging_check.py --csv "outputs/Chandal Chowkdi.csv"
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from atd_stats import data_dir  # noqa: E402
from possession.io_atd import GAMES  # noqa: E402
from possession.match import H, A, D, X  # noqa: E402

CODES = {"H": H, "A": A, "D": D, "X": X}


def load_tags(folder, fps=25):
    path = os.path.join(folder, "tactical", "BAR-ATH_tactical_smart_tagging.xml")
    out = []
    for instance in ET.parse(path).iter("instance"):
        tags = {label.findtext("group"): label.findtext("text")
                for label in instance.findall("label")}
        out.append({"code": (instance.findtext("code") or "").strip().upper(),
                    "start": int(round(float(instance.findtext("start")) * fps)),
                    "stop": int(round(float(instance.findtext("end")) * fps)),
                    "tags": tags})
    return out


def team_code(text):
    text = (text or "").strip().lower()
    if text.startswith("barcelona"):
        return H
    if text in ("n/a", ""):
        return None
    return A


def load_labels(path):
    with open(path) as handle:
        next(handle)
        return np.array([CODES[line.rstrip().split(",")[1]] for line in handle], dtype=np.int8)


def pct(values, points=(10, 25, 50, 75, 90)):
    return " ".join("{:.3g}".format(v) for v in np.percentile(values, points)) if len(values) else "n/a"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None)
    parser.add_argument("--csv", default=os.path.join(ROOT, "outputs", "Chandal Chowkdi.csv"))
    args = parser.parse_args()

    labels = load_labels(args.csv)
    tags = load_tags(data_dir(args.data))
    runs = GAMES["BAR-ATH"]["runs"]
    graded = np.zeros(len(labels), dtype=bool)
    for a, b in runs:
        graded[a:b] = True
    fps = 25

    # Predicted restarts: frames where a dead-ball stretch ends and a team has the ball.
    alive = (labels == H) | (labels == A)
    restarts = np.flatnonzero(alive[1:] & (labels[:-1] == D)) + 1
    print("submission: {} dead-ball stretches end in a restart; {} tagged instances".format(
        len(restarts), len(tags)))

    print("\nSET PIECES (tagged restarts)")
    pieces = [t for t in tags if t["code"] == "SET PIECES" and graded[min(t["start"], len(labels) - 1)]]
    kinds = {}
    for t in pieces:
        kinds.setdefault(t["tags"].get("Type", "?"), []).append(t)
    print("  {:<24} {:>3}  {:>9} {:>10} {:>10}   {}".format(
        "type", "n", "length s", "dead at", "restart", "our restart minus tag start, s"))
    print("  {:<24} {:>3}  {:>9} {:>10} {:>10}   {}".format(
        "", "", "(median)", "start", "within 3s", "(p10 p25 p50 p75 p90)"))
    all_offsets, team_hits, team_total = [], 0, 0
    for kind, items in sorted(kinds.items(), key=lambda kv: -len(kv[1])):
        dead_at_start, near, offsets = [], [], []
        for t in items:
            s = t["start"]
            dead_at_start.append(labels[s] == D)
            k = int(np.argmin(np.abs(restarts - s))) if len(restarts) else 0
            offset = (restarts[k] - s) / float(fps) if len(restarts) else 99.0
            # A restart counts if it falls inside the tagged window or within 3 s of it.
            near.append(-3.0 <= offset <= (t["stop"] - s) / float(fps) + 3.0)
            offsets.append(offset)
            side = team_code(t["tags"].get("Team"))
            if side is not None and near[-1]:
                team_total += 1
                after = labels[restarts[k]:restarts[k] + 2 * fps]
                team_hits += int((after == side).mean() > 0.5)
        all_offsets += [o for o, ok in zip(offsets, near) if ok]
        print("  {:<24} {:>3}  {:>9.1f} {:>10.2f} {:>10.2f}   {}".format(
            kind, len(items), float(np.median([(t["stop"] - t["start"]) / fps for t in items])),
            float(np.mean(dead_at_start)), float(np.mean(near)),
            pct([o for o, ok in zip(offsets, near) if ok])))
    print("  all matched restarts: our restart minus tag start, s (p10 p25 p50 p75 p90): {}".format(
        pct(all_offsets)))
    if team_total:
        print("  team taking the restart agrees with the tag: {} of {}".format(team_hits, team_total))

    print("\nPHASES WITH A TEAM")
    print("  {:<30} {:>4} {:>9} {:>9} {:>12}".format(
        "code", "n", "length s", "we say D", "team agrees"))
    by_code = {}
    for t in tags:
        if t["code"] == "SET PIECES" or team_code(t["tags"].get("Team")) is None:
            continue
        by_code.setdefault(t["code"], []).append(t)
    total_agree, total_alive = 0, 0
    for code, items in sorted(by_code.items(), key=lambda kv: -len(kv[1])):
        dead, agree, live = 0, 0, 0
        frames = 0
        for t in items:
            a, b = max(0, t["start"]), min(len(labels), t["stop"])
            part = labels[a:b][graded[a:b]]
            side = team_code(t["tags"].get("Team"))
            frames += len(part)
            dead += int((part == D).sum())
            live += int(((part == H) | (part == A)).sum())
            agree += int((part == side).sum())
        total_agree += agree
        total_alive += live
        print("  {:<30} {:>4} {:>9.1f} {:>9.2f} {:>12.2f}".format(
            code, len(items), float(np.median([(t["stop"] - t["start"]) / fps for t in items])),
            dead / float(max(1, frames)), agree / float(max(1, live))))
    print("  all phases together: team agrees on {:.3f} of the frames we call alive".format(
        total_agree / float(max(1, total_alive))))

    covered = np.zeros(len(labels), dtype=bool)
    for items in by_code.values():
        for t in items:
            covered[max(0, t["start"]):t["stop"]] = True
    print("  graded frames inside at least one team phase: {:.3f}".format(
        float(covered[graded].mean())))
