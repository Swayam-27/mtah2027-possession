"""Load Metrica Sports' public sample games into a Match, with labels built from their events.

Source: https://github.com/metrica-sports/sample-data (three anonymised games, 25 fps tracking
of players and ball, plus synchronised event data). Games 1 and 2 are in Metrica's CSV format,
game 3 in the FIFA EPTS format (the same format as the competition tracking) with JSON events.

There is no per-frame possession in this data, so the labels are derived from the events:

- **Dead ball** starts at a stoppage event -- the ball going out (at the moment it leaves the
  pitch), a shot that goes out or in, a foul, an offside or a hand ball -- and ends at the
  restart that follows (a set piece, or a goal kick). If play carries on instead (advantage),
  there is no stoppage.
- **Team** is whoever made the last on-the-ball event: a pass, carry, set piece, recovery, shot,
  or the touch that lost the ball. A loose ball stays with the last team, as in the
  competition's definition. One-touch spells are then handed back by `relabel.controlled`,
  exactly as for the other external matches.

Stretches of 2 s or more with no ball in the tracking are also dead: in this data the ball is
removed exactly when play stops, so those are stoppages the events do not mention.

Event times are marked by people and are good to a few frames, so these labels are a little
less precise in time than ones derived from tracking.
"""

import csv
import json
import os
import xml.etree.ElementTree as ET

import numpy as np

from .match import Match, H, A, D, X, HOME, AWAY

PITCH = (105.0, 68.0)
FPS = 25
CONTROL = {"PASS", "CARRY", "SET PIECE", "RECOVERY", "SHOT", "BALL LOST"}
IGNORED = {"CARD", "CHALLENGE", "FAULT RECEIVED", "BALL OUT"}
ADVANTAGE_GAP = 5 * FPS   # a live event within 5 s of a foul means play went on
TAIL = 2 * FPS            # frames of play kept after the last event of a half


def _is_stop(event):
    kind, sub = event["type"], event["sub"]
    if kind == "BALL OUT":
        return True
    if kind == "SHOT" and ("OUT" in sub or "GOAL" in sub):
        return True
    if kind == "FAULT RECEIVED":
        return True
    if kind == "CHALLENGE" and "FAULT" in sub and "ADVANTAGE" not in sub:
        return True
    if kind == "BALL LOST" and ("OFFSIDE" in sub or "HAND BALL" in sub):
        return True
    return False


def _is_restart(event):
    return event["type"] == "SET PIECE" or "GOAL KICK" in event["sub"]


def _stop_frame(event):
    """The ball leaves play at the end of a ball-out or a shot, at the start of a foul."""
    if event["type"] in ("BALL OUT", "SHOT") and event["end"] > event["start"]:
        return event["end"]
    return event["start"]


def labels_from_events(events, n_frames, periods):
    """[n_frames] int8 labels and the runs (one per half) they cover."""
    labels = np.full(n_frames, X, dtype=np.int8)
    runs = []
    for period, (first, last) in sorted(periods.items()):
        mine = sorted((e for e in events if e["period"] == period), key=lambda e: e["start"])
        kick = next((e for e in mine if e["type"] == "SET PIECE" and "KICK OFF" in e["sub"]), None)
        if kick is None or not mine:
            continue
        start = max(first, kick["start"])
        stop = min(last + 1, max(e["end"] for e in mine) + TAIL, n_frames)
        if stop <= start:
            continue
        runs.append((start, stop))

        # Team: forward fill of the last on-the-ball event.
        team = np.full(stop - start, -1, dtype=np.int8)
        for e in mine:
            if e["type"] in CONTROL and start <= e["start"] < stop:
                team[e["start"] - start] = e["team"]
        current = kick["team"]
        for i in range(len(team)):
            if team[i] >= 0:
                current = team[i]
            team[i] = current
        part = np.where(team == HOME, H, A).astype(np.int8)

        # Dead ball: from a stoppage to the restart that follows it.
        plays = [e for e in mine if e["type"] not in IGNORED]
        for e in mine:
            if not _is_stop(e):
                continue
            a = _stop_frame(e)
            following = next((p for p in plays if p is not e and p["start"] >= a), None)
            if following is None:
                b = stop
            elif _is_restart(following) or following["start"] - a > ADVANTAGE_GAP:
                b = following["start"]
            else:
                continue  # play went on
            a, b = max(a, start), min(b, stop)
            if b > a:
                part[a - start:b - start] = D
        labels[start:stop] = part
    return labels, runs


# --- games 1 and 2: CSV ----------------------------------------------------------------------

def _csv_events(path):
    out = []
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            out.append({"team": HOME if row["Team"].strip().lower() == "home" else AWAY,
                        "type": row["Type"].strip().upper(),
                        "sub": row["Subtype"].strip().upper(),
                        "period": int(row["Period"]),
                        "start": int(row["Start Frame"]) - 1,
                        "end": int(row["End Frame"]) - 1})
    return out


def _csv_tracking(path):
    """(period [n], x [n, players], y [n, players], ball x [n], ball y [n]); frames from 1."""
    with open(path, newline="") as handle:
        reader = csv.reader(handle)
        next(reader), next(reader), next(reader)
        rows = [r for r in reader if r]
    frame = np.array([int(r[1]) for r in rows]) - 1
    values = np.array([[float(v) if v not in ("", "NaN") else np.nan for v in r[3:]] for r in rows],
                      dtype=np.float32)
    n = int(frame.max()) + 1
    period = np.zeros(n, dtype=np.int8)
    period[frame] = [int(r[0]) for r in rows]
    full = np.full((n, values.shape[1]), np.nan, dtype=np.float32)
    full[frame] = values
    players = full[:, :-2]
    return period, players[:, 0::2], players[:, 1::2], full[:, -2], full[:, -1]


def load_csv_game(folder, number):
    stem = os.path.join(folder, "Sample_Game_{}_".format(number))
    period, hx, hy, bx, by = _csv_tracking(stem + "RawTrackingData_Home_Team.csv")
    _, ax, ay, _, _ = _csv_tracking(stem + "RawTrackingData_Away_Team.csv")
    n = min(len(period), len(ax))
    px = np.concatenate([hx[:n], ax[:n]], axis=1)
    py = np.concatenate([hy[:n], ay[:n]], axis=1)
    pteam = np.array([HOME] * hx.shape[1] + [AWAY] * ax.shape[1], dtype=np.int8)
    periods = {}
    for p in (1, 2):
        idx = np.flatnonzero(period[:n] == p)
        if len(idx):
            periods[p] = (int(idx[0]), int(idx[-1]))
    labels, runs = labels_from_events(_csv_events(stem + "RawEventsData.csv"), n, periods)
    return _match("MET-{}".format(number), px, py, pteam, bx[:n], by[:n], runs, labels)


# --- game 3: EPTS + JSON ---------------------------------------------------------------------

def _epts_meta(path):
    root = ET.parse(path).getroot()
    team_of = {p.get("id"): p.get("teamId") for p in root.iter("Player")}
    home_id = next(root.iter("Team")).get("id")
    channels = [(c.get("id"), c.get("playerId")) for c in root.iter("PlayerChannel")
                if c.get("channelId") == "x"]
    pteam = np.array([HOME if team_of[player] == home_id else AWAY for _, player in channels],
                     dtype=np.int8)
    # Which player each group of a line belongs to changes with substitutions: every block of
    # frames lists its own order of player channels.
    slot_of = {channel: i for i, (channel, _) in enumerate(channels)}
    layouts = []
    for spec in root.iter("DataFormatSpecification"):
        order = [slot_of[r.get("playerChannelId")] for r in spec.iter("PlayerChannelRef")
                 if r.get("playerChannelId").endswith("_x")]
        layouts.append((int(spec.get("startFrame")), int(spec.get("endFrame")), order))
    params = {p.findtext("Name"): p.findtext("Value") for p in root.iter("ProviderParameter")}
    periods = {}
    for p, key in ((1, "first_half"), (2, "second_half")):
        a, b = params.get(key + "_start"), params.get(key + "_end")
        if a and b:
            periods[p] = (int(a) - 1, int(b) - 1)
    return pteam, home_id, periods, layouts


def _xy(group):
    if not group or group[0] in "Nn":
        return np.nan, np.nan
    parts = group.split(",")
    try:
        return float(parts[0]), float(parts[1])
    except (ValueError, IndexError):
        return np.nan, np.nan


def load_epts_game(folder, number=3):
    stem = os.path.join(folder, "Sample_Game_{}_".format(number))
    pteam, home_id, periods, layouts = _epts_meta(stem + "metadata.xml")
    slots = len(pteam)
    with open(stem + "tracking.txt") as handle:
        lines = [line.rstrip().split(":") for line in handle if ":" in line]
    n = max(int(parts[0]) for parts in lines)
    px = np.full((n, slots), np.nan, dtype=np.float32)
    py = np.full((n, slots), np.nan, dtype=np.float32)
    bx = np.full(n, np.nan, dtype=np.float32)
    by = np.full(n, np.nan, dtype=np.float32)
    for parts in lines:
        if len(parts) != 3:
            continue
        frame = int(parts[0])
        row = frame - 1
        order = next((o for a, b, o in layouts if a <= frame <= b), None)
        if order is None:
            continue
        for slot, group in zip(order, parts[1].split(";")):
            px[row, slot], py[row, slot] = _xy(group)
        bx[row], by[row] = _xy(parts[2])

    with open(stem + "events.json") as handle:
        raw = json.load(handle)["data"]
    events = []
    for e in raw:
        subs = e.get("subtypes") or []
        subs = [subs] if isinstance(subs, dict) else subs
        events.append({"team": HOME if (e.get("team") or {}).get("id") == home_id else AWAY,
                       "type": e["type"]["name"].strip().upper(),
                       "sub": "-".join(s["name"] for s in subs).upper(),
                       "period": int(e["period"]),
                       "start": int(e["start"]["frame"]) - 1,
                       "end": int(e["end"]["frame"]) - 1})
    labels, runs = labels_from_events(events, n, periods)
    return _match("MET-{}".format(number), px, py, pteam, bx, by, runs, labels)


def _dead_while_unseen(labels, bx, runs, shortest=2 * FPS):
    """Mark as dead every stretch of 2 s or more in which the tracking has no ball.

    In these sample games the ball is removed from the tracking exactly when play stops (it
    disappears within a frame of every stoppage found in the events), so a long stretch without
    a ball is a stoppage the events do not mention, such as play halted for an injury.
    """
    labels = labels.copy()
    for start, stop in runs:
        missing = np.isnan(bx[start:stop])
        edges = np.diff(np.concatenate(([0], missing.astype(np.int8), [0])))
        for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            if b - a >= shortest:
                labels[start + a:start + b] = D
    return labels


def _match(name, px, py, pteam, bx, by, runs, labels):
    labels = _dead_while_unseen(labels, bx, runs)
    length, width = PITCH
    return Match(name=name, fps=FPS, pitch=PITCH, px=px * length, py=py * width, pteam=pteam,
                 bx=bx * length, by=by * width, runs=runs, labels=labels)


def load(folder, number):
    return load_epts_game(folder, number) if number == 3 else load_csv_game(folder, number)
