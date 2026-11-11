"""Load the competition tracking (Metrica ATD, FIFA/EPTS text export) into a Match.

Raw line:  "<frame_id>:<x,y,speed>;<x,y,speed>;...:<ball x,y,...>"  with NaN for a missing
observation. Coordinates are normalised 0-1. frame_id counts from 1; video frames from 0.
"""

import xml.etree.ElementTree as ET

import numpy as np

from .match import Match, HOME, AWAY

# Per-match constants published by the organisers (official games.py / DATA.md).
GAMES = {
    "BAR-ATH": {
        "n_frames": 199319,
        "fps": 25,
        "pitch": (105.3, 68.0),
        # Graded video-frame spans, half-open: the two halves.
        "runs": [(11185, 90652), (110808, 190198)],
    },
}


def slot_teams(meta_path):
    """HOME/AWAY for every track slot. The first team in the metadata is the home side."""
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError("{}: expected 2 teams, found {}".format(meta_path, len(team_ids)))
    slots = [player.get("teamId") for player in root.iter("Player")]
    if not slots:
        raise ValueError("{}: no Player entries".format(meta_path))
    return np.array([HOME if t == team_ids[0] else AWAY for t in slots], dtype=np.int8)


def _xy(group):
    """(x, y) of an 'x,y,...' group, NaN if the observation is missing or malformed."""
    if not group or group[0] in "Nn":
        return np.nan, np.nan
    parts = group.split(",")
    try:
        return float(parts[0]), float(parts[1])
    except (ValueError, IndexError):
        return np.nan, np.nan


def frame_rate(meta_path, default=25.0):
    """Frames per second stated in the metadata, or the default if it does not say."""
    for node in ET.parse(meta_path).getroot().iter("FrameRate"):
        try:
            return float(node.text)
        except (TypeError, ValueError):
            pass
    return default


def load_feed(meta_path, raw_path, pitch=(105.3, 68.0)):
    """Any ATD feed on its own timeline (e.g. the broadcast recording), as a Match without runs.

    Row i is the feed's own video frame i; nothing here relates it to the tactical video.
    """
    pteam = slot_teams(meta_path)
    n_slots = len(pteam)
    with open(raw_path) as handle:
        n_frames = max((int(line.split(":", 1)[0]) for line in handle if ":" in line), default=0)
    px = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    py = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    bx = np.full(n_frames, np.nan, dtype=np.float32)
    by = np.full(n_frames, np.nan, dtype=np.float32)
    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip().split(":")
            if len(parts) != 3:
                continue
            row = int(parts[0]) - 1
            if row < 0:
                continue
            for slot, group in enumerate(parts[1].split(";")[:n_slots]):
                px[row, slot], py[row, slot] = _xy(group)
            bx[row], by[row] = _xy(parts[2])
    length, width = pitch
    return Match(name="feed", fps=int(round(frame_rate(meta_path))), pitch=pitch,
                 px=px * length, py=py * width, pteam=pteam,
                 bx=bx * length, by=by * width, runs=[(0, n_frames)])


def load(meta_path, raw_path, game="BAR-ATH"):
    info = GAMES[game]
    pteam = slot_teams(meta_path)
    n_frames, n_slots = info["n_frames"], len(pteam)
    length, width = info["pitch"]

    px = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    py = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    bx = np.full(n_frames, np.nan, dtype=np.float32)
    by = np.full(n_frames, np.nan, dtype=np.float32)

    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            row = int(parts[0]) - 1
            if not 0 <= row < n_frames:
                continue
            for slot, group in enumerate(parts[1].split(";")[:n_slots]):
                px[row, slot], py[row, slot] = _xy(group)
            bx[row], by[row] = _xy(parts[2])

    return Match(name=game, fps=info["fps"], pitch=info["pitch"],
                 px=px * length, py=py * width, pteam=pteam,
                 bx=bx * length, by=by * width, runs=list(info["runs"]))
