"""The rule-based possession engine: football rules written as code, no training.

Three decisions, made in this order:

  1. DEAD OR ALIVE. The ball is dead only on positive evidence -- it is seen off the pitch, it
     sits still, the players have all slowed to a walk, or it has been unseen for a long time.
     An unseen ball on its own is never evidence: that is the baseline's main mistake.
  2. WHO IS ON THE BALL. Within reach of exactly one team's nearest player.
  3. WHO HAS POSSESSION. A team takes possession only by showing control: staying on the ball
     for `control` frames, or being on it twice in a row (a completed pass). A lone touch --
     deflection, block, lost header -- changes nothing. The change is dated to the first touch.
     Between touches the ball stays with the last team, as the competition rules say.
"""

from dataclasses import dataclass

import numpy as np

from .match import H, A, D, X


@dataclass
class Rules:
    reach: float = 2.0          # metres: a player this close to the ball is "on" it
    margin: float = 0.4         # metres the nearest opponent must be further away than that
    trust: int = 30             # frames from a sighting within which a filled position is trusted
    bridge: int = 8             # join two spells on the ball by the same team this close together
    min_touch: int = 3          # ignore spells on the ball shorter than this
    control: int = 20           # frames on the ball that prove control on their own
    pass_window: int = 100      # second touch by the same team within this many frames = control
    out_margin: float = 0.5     # metres beyond a line before a seen ball counts as out
    still_speed: float = 1.0    # m/s below which the ball counts as stationary
    still_frames: int = 50      # ... for this long
    walk_speed: float = 1.3     # m/s median player speed that looks like a stoppage
    walk_frames: int = 75       # ... for this long
    lost_frames: int = 150      # ball unseen this long means it left play
    dead_merge: int = 50        # join dead stretches closer than this
    dead_min: int = 60          # drop dead stretches shorter than this
    restart_speed: float = 4.0  # m/s: a seen ball this fast means play is on again


def _runs_of(mask):
    """[(start, stop)] of True stretches."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def _sustained(mask, length):
    """True where mask has held for at least `length` consecutive frames (whole stretch marked)."""
    out = np.zeros(len(mask), dtype=bool)
    for a, b in _runs_of(mask):
        if b - a >= length:
            out[a:b] = True
    return out


def _close_open(mask, merge, minimum):
    """Join True stretches separated by less than `merge`, then drop those under `minimum`."""
    mask = ~_drop_short(~mask, merge)
    return _drop_short(mask, minimum)


def _drop_short(mask, minimum):
    out = mask.copy()
    for a, b in _runs_of(mask):
        if b - a < minimum and a > 0 and b < len(mask):
            out[a:b] = False
    return out


def dead_mask(f, rules, start, stop):
    s = slice(start, stop)
    seen = f.seen[s]
    out = seen & (f.out[s] > rules.out_margin)
    still = _sustained((f.speed[s] < rules.still_speed) & (f.edge[s] <= 2), rules.still_frames)
    walk = _sustained(f.walk[s] < rules.walk_speed, rules.walk_frames)
    lost = f.gap[s] >= rules.lost_frames
    dead = out | still | walk | lost
    # Play is certainly on wherever a seen ball is moving fast inside the pitch.
    dead &= ~(seen & (f.speed[s] > rules.restart_speed) & (f.out[s] == 0) & ~walk)
    return _close_open(dead, rules.dead_merge, rules.dead_min)


def on_ball(f, rules, start, stop):
    """Per frame: 0 home on the ball, 1 away, -1 nobody (or position not trustworthy)."""
    s = slice(start, stop)
    dh, da = f.d_home[s], f.d_away[s]
    who = np.full(stop - start, -1, dtype=np.int8)
    trusted = f.edge[s] <= rules.trust
    who[trusted & (dh < rules.reach) & (da > dh + rules.margin)] = H
    who[trusted & (da < rules.reach) & (dh > da + rules.margin)] = A
    return who


def touches(who, rules):
    """[(start, stop, team)] spells on the ball, near-adjacent same-team spells joined."""
    spells = []
    for team in (H, A):
        mask = who == team
        mask = ~_drop_short(~mask, rules.bridge)
        # Bridging must not swallow the other team's time on the ball.
        mask &= who != (A if team == H else H)
        spells += [(a, b, team) for a, b in _runs_of(mask) if b - a >= rules.min_touch]
    spells.sort()
    return spells


def possession(f, rules, start, stop, dead):
    """Label one run. Returns int8 codes for frames start..stop."""
    n = stop - start
    labels = np.full(n, X, dtype=np.int8)
    labels[dead] = D
    who = on_ball(f, rules, start, stop)
    nearer = np.where(f.d_home[start:stop] <= f.d_away[start:stop], H, A)

    previous_owner = None
    for a, b in _runs_of(~dead):
        spells = [(s + a, e + a, team) for s, e, team in touches(who[a:b], rules)]
        if not spells:
            # Nobody clearly on the ball in this whole passage: majority of "nearer team".
            owner = int(np.bincount(nearer[a:b], minlength=2).argmax())
            labels[a:b] = owner
            previous_owner = owner
            continue

        owner = spells[0][2]
        cut = a
        for i, (s, e, team) in enumerate(spells):
            if team == owner:
                continue
            controlled = (e - s) >= rules.control
            if not controlled and i + 1 < len(spells):
                nxt = spells[i + 1]
                controlled = nxt[2] == team and nxt[0] - e <= rules.pass_window
            if controlled:
                labels[cut:s] = owner
                owner, cut = team, s
        labels[cut:b] = owner
        previous_owner = owner
    return labels


def label_match(match, f, rules=None):
    rules = rules or Rules()
    labels = np.full(match.n_frames, X, dtype=np.int8)
    for start, stop in match.runs:
        dead = dead_mask(f, rules, start, stop)
        labels[start:stop] = possession(f, rules, start, stop, dead)
    return labels
