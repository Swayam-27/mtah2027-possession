"""One in-memory format for every match, whatever provider it came from.

Everything downstream (features, engine, model) reads a Match and nothing else, so the same code
runs on the competition ATD and on the external matches we benchmark on.

Coordinates are metres with the origin at a pitch corner: x in [0, length], y in [0, width].
Player columns are *slots*, not people: nothing downstream may assume a column follows one
player, because the competition tracking has no identities.
"""

from dataclasses import dataclass, field

import numpy as np

H, A, D, X = 0, 1, 2, 3
LABELS = np.array(["H", "A", "D", "X"])
HOME, AWAY = 0, 1


@dataclass
class Match:
    name: str
    fps: int
    pitch: tuple                  # (length, width) in metres
    px: np.ndarray                # [n_frames, n_slots] float32, NaN where unseen
    py: np.ndarray                # [n_frames, n_slots] float32
    pteam: np.ndarray             # [n_slots] int8, HOME or AWAY
    bx: np.ndarray                # [n_frames] float32, NaN where the ball is unseen
    by: np.ndarray                # [n_frames] float32
    runs: list                    # [(start, stop)] half-open frame spans that are match play
    labels: np.ndarray = None     # [n_frames] int8 H/A/D/X, only for matches with ground truth
    bz: np.ndarray = None         # [n_frames] ball height, external data only (never an input)
    extra: dict = field(default_factory=dict)

    @property
    def n_frames(self):
        return len(self.bx)

    def save(self, path):
        np.savez_compressed(
            path, name=self.name, fps=self.fps, pitch=np.array(self.pitch),
            px=self.px, py=self.py, pteam=self.pteam, bx=self.bx, by=self.by,
            runs=np.array(self.runs, dtype=np.int64),
            labels=self.labels if self.labels is not None else np.array([]),
            bz=self.bz if self.bz is not None else np.array([]))

    @staticmethod
    def load(path):
        z = np.load(path, allow_pickle=False)
        return Match(
            name=str(z["name"]), fps=int(z["fps"]), pitch=tuple(float(v) for v in z["pitch"]),
            px=z["px"], py=z["py"], pteam=z["pteam"], bx=z["bx"], by=z["by"],
            runs=[(int(a), int(b)) for a, b in z["runs"]],
            labels=z["labels"] if z["labels"].size else None,
            bz=z["bz"] if z["bz"].size else None)
