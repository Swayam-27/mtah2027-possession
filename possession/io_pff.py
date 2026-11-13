"""Load PFF FC's 2022 World Cup data into a Match at 25 fps, with labels built from its events.

Source: PFF FC's free 2022 World Cup dataset (registration required): broadcast tracking at
29.97 fps with every player's position (estimated when off screen) and the ball with its height,
plus detailed event data. Coordinates are centred metres; they are shifted to a corner origin.

Labels, from the event data (all times are seconds of video, so frame = time x fps):

- **Team**: every on-the-ball event (OTB) names the team on the ball; the team is carried
  forward until the next one. A loose ball stays with the last team.
- **Dead ball**: every restart is an on-the-ball event with a set-piece type (throw-in, free
  kick, corner, goal kick, kick-off, penalty). The ball is dead from the stoppage before it --
  the ball-out, goal or foul event if there is one, otherwise the end of the last on-the-ball
  event -- until the restart.
- One-touch spells are handed back by `relabel.controlled`, as for the other external matches.
"""

import bz2
import json
import os

import numpy as np

from .match import Match, H, A, D, X, HOME, AWAY

FPS_OUT = 25
LENGTH, WIDTH = 105.0, 68.0
RESTARTS = {"T", "F", "C", "G", "K", "P"}
STOPS = {"OUT", "G", "FOUL"}
MAX_SLOTS = 20


def _xy(item):
    if not item or item.get("x") is None or item.get("y") is None:
        return np.nan, np.nan
    return float(item["x"]) + LENGTH / 2.0, float(item["y"]) + WIDTH / 2.0


def _tracking(path, fps_in):
    """Arrays at the source frame rate, indexed by source frame number."""
    frames, home, away, ball, height = [], [], [], [], []
    slots = {HOME: {}, AWAY: {}}
    with bz2.open(path, "rt") as handle:
        for line in handle:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if d.get("frameNum") is None:
                continue
            frames.append(int(d["frameNum"]))
            for side, key, out in ((HOME, "homePlayers", home), (AWAY, "awayPlayers", away)):
                row = {}
                for p in d.get(key) or []:
                    number = p.get("jerseyNum")
                    if number is None:
                        continue
                    index = slots[side].setdefault(str(number), len(slots[side]))
                    if index < MAX_SLOTS:
                        row[index] = _xy(p)
                out.append(row)
            b = d.get("ballsSmoothed")
            if isinstance(b, list):
                b = b[0] if b else None
            ball.append(_xy(b))
            height.append(float(b["z"]) if b and b.get("z") is not None else np.nan)
    frames = np.asarray(frames)
    n = int(frames.max()) + 1
    px = np.full((n, 2 * MAX_SLOTS), np.nan, dtype=np.float32)
    py = np.full((n, 2 * MAX_SLOTS), np.nan, dtype=np.float32)
    bx = np.full(n, np.nan, dtype=np.float32)
    by = np.full(n, np.nan, dtype=np.float32)
    bz = np.full(n, np.nan, dtype=np.float32)
    for i, f in enumerate(frames):
        for index, (x, y) in home[i].items():
            px[f, index], py[f, index] = x, y
        for index, (x, y) in away[i].items():
            px[f, MAX_SLOTS + index], py[f, MAX_SLOTS + index] = x, y
        bx[f], by[f] = ball[i]
        bz[f] = height[i]
    used = [i for i in range(MAX_SLOTS) if i < len(slots[HOME])] + \
           [MAX_SLOTS + i for i in range(MAX_SLOTS) if i < len(slots[AWAY])]
    pteam = np.array([HOME] * min(len(slots[HOME]), MAX_SLOTS) +
                     [AWAY] * min(len(slots[AWAY]), MAX_SLOTS), dtype=np.int8)
    return px[:, used], py[:, used], pteam, bx, by, bz


def _events(path):
    out = []
    for e in json.load(open(path, encoding="utf-8")):
        g = e.get("gameEvents") or {}
        start = e.get("startTime")
        if start is None:
            continue
        end = e.get("endTime") or start
        out.append({"type": g.get("gameEventType"), "setpiece": g.get("setpieceType"),
                    "home": g.get("homeTeam"), "period": g.get("period"),
                    "start": float(start), "end": max(float(end), float(start))})
    return sorted(out, key=lambda e: e["start"])


def labels_from_events(events, meta, n):
    """[n] labels at 25 fps and the runs (one per regular half)."""
    labels = np.full(n, X, dtype=np.int8)
    runs = []
    for period in (1, 2):
        first = meta.get("startPeriod{}".format(period))
        last = meta.get("endPeriod{}".format(period))
        if first is None or last is None:
            continue
        start, stop = int(round(first * FPS_OUT)), min(n, int(round(last * FPS_OUT)))
        if stop <= start:
            continue
        runs.append((start, stop))
        mine = [e for e in events if first - 1.0 <= e["start"] <= last + 1.0]

        # Team: forward fill of the team named by each on-the-ball event.
        team = np.full(stop - start, -1, dtype=np.int8)
        for e in mine:
            if e["type"] == "OTB" and e["home"] is not None:
                f = int(round(e["start"] * FPS_OUT)) - start
                if 0 <= f < len(team):
                    team[f] = H if e["home"] else A
        current = next((t for t in team if t >= 0), H)
        for i in range(len(team)):
            if team[i] >= 0:
                current = team[i]
            team[i] = current
        part = team.copy()

        # Dead ball: from the stoppage before every restart to the restart.
        previous_end, stops_since = first, []
        for e in mine:
            if e["type"] in STOPS:
                stops_since.append(e["start"])
                continue
            if e["type"] != "OTB":
                continue
            if e["setpiece"] in RESTARTS and e["start"] > first + 0.5:
                dead_from = min(stops_since) if stops_since else previous_end
                a = max(0, int(round(dead_from * FPS_OUT)) - start)
                b = min(len(part), int(round(e["start"] * FPS_OUT)) - start)
                if b > a:
                    part[a:b] = D
            previous_end, stops_since = e["end"], []
        labels[start:stop] = part
    return labels, runs


def load(folder, match_id):
    meta = json.load(open(os.path.join(folder, "metadata", "{}.json".format(match_id))))[0]
    fps_in = float(meta.get("fps") or 29.97)
    px, py, pteam, bx, by, bz = _tracking(
        os.path.join(folder, "tracking", "{}.jsonl.bz2".format(match_id)), fps_in)
    # Resample to 25 fps: the nearest source frame to every output time.
    n = int(len(bx) / fps_in * FPS_OUT)
    source = np.clip(np.round(np.arange(n) / FPS_OUT * fps_in).astype(np.int64), 0, len(bx) - 1)
    px, py, bx, by, bz = px[source], py[source], bx[source], by[source], bz[source]
    events = _events(os.path.join(folder, "events", "{}.json".format(match_id)))
    labels, runs = labels_from_events(events, meta, n)
    return Match(name="PFF-{}".format(match_id), fps=FPS_OUT, pitch=(LENGTH, WIDTH), px=px, py=py,
                 pteam=pteam, bx=bx, by=by, runs=runs, labels=labels, bz=bz)
