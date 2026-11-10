"""Load the open DFL / Sportec matches (Bassek et al., Scientific Data 2025, CC-BY 4.0).

These are OTHER games, used to develop and train on. The raw tracking carries a per-frame
BallStatus (in play or not) and BallPossession (home or away), which is the same kind of signal
the competition ground truth is derived from -- so each match gives a full H/A/D label track.

The position file is ~400 MB of one-Frame-per-line XML, so it is streamed with regexes rather
than parsed as a tree.
"""

import glob
import os
import re
import xml.etree.ElementTree as ET

import numpy as np

from .match import Match, H, A, D, X, HOME, AWAY

FPS = 25
SECTIONS = ("firstHalf", "secondHalf")
GAP_FRAMES = 1000  # ungraded filler between the halves, so no window ever spans both

_FRAMESET = re.compile(r'<FrameSet ([^>]*)>')
_ATTR = re.compile(r'(\w+)="([^"]*)"')
_N = re.compile(r' N="(\d+)"')
_X = re.compile(r' X="([^"]*)"')
_Y = re.compile(r' Y="([^"]*)"')
_Z = re.compile(r' Z="([^"]*)"')
_POSS = re.compile(r' BallPossession="(\d)"')
_STATUS = re.compile(r' BallStatus="(\d)"')


def match_ids(folder):
    paths = sorted(glob.glob(os.path.join(folder, "DFL_04_03_positions_raw_observed_*.xml")))
    return [os.path.basename(p)[:-4].rsplit("-", 1)[1] for p in paths]


def _one(folder, prefix, match_id):
    hits = glob.glob(os.path.join(folder, "{}_*{}.xml".format(prefix, match_id)))
    if len(hits) != 1:
        raise IOError("expected one {} file for {}, found {}".format(prefix, match_id, len(hits)))
    return hits[0]


def _teams(folder, match_id):
    root = ET.parse(_one(folder, "DFL_02_01_matchinformation", match_id)).getroot()
    general = next(root.iter("General"))
    env = next(root.iter("Environment"))
    pitch = (float(env.get("PitchX")), float(env.get("PitchY")))
    return general.get("HomeTeamId"), general.get("GuestTeamId"), pitch


def load(folder, match_id):
    home_id, away_id, pitch = _teams(folder, match_id)

    # First pass collects per (section, object) arrays keyed by frame number.
    tracks = []  # (section, team or "BALL", frame numbers, x, y, [z, poss, status])
    current = None
    with open(_one(folder, "DFL_04_03_positions_raw_observed", match_id), encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("<Frame "):
                if current is None:
                    continue
                current["n"].append(int(_N.search(line).group(1)))
                current["x"].append(float(_X.search(line).group(1)))
                current["y"].append(float(_Y.search(line).group(1)))
                if current["team"] == "BALL":
                    current["z"].append(float(_Z.search(line).group(1)))
                    current["poss"].append(int(_POSS.search(line).group(1)))
                    current["status"].append(int(_STATUS.search(line).group(1)))
                continue
            found = _FRAMESET.search(line)
            if found:
                attrs = dict(_ATTR.findall(found.group(1)))
                team = attrs.get("TeamId")
                if attrs.get("GameSection") in SECTIONS and team in (home_id, away_id, "BALL"):
                    current = {"section": attrs["GameSection"], "team": team,
                               "person": attrs.get("PersonId"),
                               "n": [], "x": [], "y": [], "z": [], "poss": [], "status": []}
                    tracks.append(current)
                else:
                    current = None  # referees and anything outside the two halves

    balls = {t["section"]: t for t in tracks if t["team"] == "BALL"}
    spans, offset, runs = {}, 0, []
    for section in SECTIONS:
        numbers = np.array(balls[section]["n"])
        first, last = int(numbers.min()), int(numbers.max())
        spans[section] = (first, offset)
        runs.append((offset, offset + last - first + 1))
        offset += last - first + 1 + GAP_FRAMES
    n_frames = offset - GAP_FRAMES

    # One column per person: a player keeps a column across both halves.
    people = []
    for t in tracks:
        if t["team"] != "BALL" and (t["team"], t["person"]) not in people:
            people.append((t["team"], t["person"]))
    column = {key: i for i, key in enumerate(people)}

    px = np.full((n_frames, len(people)), np.nan, dtype=np.float32)
    py = np.full((n_frames, len(people)), np.nan, dtype=np.float32)
    bx = np.full(n_frames, np.nan, dtype=np.float32)
    by = np.full(n_frames, np.nan, dtype=np.float32)
    bz = np.full(n_frames, np.nan, dtype=np.float32)
    labels = np.full(n_frames, X, dtype=np.int8)

    half_length, half_width = pitch[0] / 2.0, pitch[1] / 2.0
    for t in tracks:
        first, base = spans[t["section"]]
        rows = np.array(t["n"]) - first + base
        keep = (rows >= 0) & (rows < n_frames)
        rows = rows[keep]
        xs = np.array(t["x"], dtype=np.float32)[keep] + half_length
        ys = np.array(t["y"], dtype=np.float32)[keep] + half_width
        if t["team"] == "BALL":
            bx[rows], by[rows] = xs, ys
            bz[rows] = np.array(t["z"], dtype=np.float32)[keep]
            status = np.array(t["status"])[keep]
            poss = np.array(t["poss"])[keep]
            labels[rows] = np.where(status == 0, D, np.where(poss == 1, H, A))
        else:
            col = column[(t["team"], t["person"])]
            px[rows, col], py[rows, col] = xs, ys

    pteam = np.array([HOME if team == home_id else AWAY for team, _ in people], dtype=np.int8)
    return Match(name="DFL-" + match_id, fps=FPS, pitch=pitch, px=px, py=py, pteam=pteam,
                 bx=bx, by=by, bz=bz, runs=runs, labels=labels)
