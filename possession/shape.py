"""Team-shape measurements: what each team as a whole is doing on every frame.

The ball is unseen on about 41% of frames, in stretches of ten seconds and more. In those
stretches the only evidence of who has the ball is how the two teams are arranged and moving:
the team in possession spreads out and moves up the pitch, the other one closes up and drops.

Everything is measured in each team's own frame of reference -- distance from its own goal line
along its direction of attack -- so the two teams are described identically and swapping them
is just swapping the names. As elsewhere, nothing depends on which slot a player is in.
"""

import numpy as np

from .match import HOME, AWAY

LAG = 25          # frames over which movement is measured (1 s)
CHUNK = 10 ** 9   # one pass: movement is measured across frames, so no seams
SIDES = (("home", HOME), ("away", AWAY))


def attack_signs(match):
    """Per run, +1 if the home team attacks towards larger x, else -1.

    The goalkeeper gives it away: over a half, the team defending the goal at x = 0 has the
    player nearest to that goal line.
    """
    home = match.px[:, match.pteam == HOME]
    away = match.px[:, match.pteam == AWAY]
    signs = []
    for start, stop in match.runs:
        with np.errstate(all="ignore"):
            deepest_home = np.nanmedian(np.nanmin(home[start:stop], axis=1))
            deepest_away = np.nanmedian(np.nanmin(away[start:stop], axis=1))
        signs.append(1.0 if deepest_home <= deepest_away else -1.0)
    return signs


def _team(u, v, ball_u, ball_v, fps):
    """Shape of one team from its players' depth u and width v, in its own frame."""
    n = len(u)
    out = {}
    present = ~np.isnan(u)
    count = present.sum(axis=1)
    some = np.maximum(count, 1)
    with np.errstate(all="ignore"):
        cu = np.nansum(u, axis=1) / some
        cv = np.nansum(v, axis=1) / some
        su = np.sqrt(np.nansum((u - cu[:, None]) ** 2, axis=1) / some)
        sv = np.sqrt(np.nansum((v - cv[:, None]) ** 2, axis=1) / some)
    ordered = np.sort(u, axis=1)                      # NaN sorts last
    back = ordered[:, 1]                              # second deepest: the defensive line
    front = np.nanmax(np.where(present, u, -np.inf), axis=1)

    # Movement over the last second, using only the players tracked at both ends of it.
    du = np.full(u.shape, np.nan, dtype=np.float32)
    dv = np.full(u.shape, np.nan, dtype=np.float32)
    du[LAG:] = (u[LAG:] - u[:-LAG]) * (fps / float(LAG))
    dv[LAG:] = (v[LAG:] - v[:-LAG]) * (fps / float(LAG))
    jumped = np.hypot(du, dv) > 12.0
    du[jumped], dv[jumped] = np.nan, np.nan
    moving = np.maximum((~np.isnan(du)).sum(axis=1), 1)
    with np.errstate(all="ignore"):
        advance = np.nansum(du, axis=1) / moving
        pace = np.nansum(np.hypot(du, dv), axis=1) / moving
        fast = np.nansum(np.hypot(du, dv) > 4.0, axis=1) / moving
        back_move = np.zeros(n, dtype=np.float32)
        back_move[LAG:] = (back[LAG:] - back[:-LAG]) * (fps / float(LAG))

    dist = np.hypot(u - ball_u[:, None], v - ball_v[:, None])
    dist = np.sort(np.where(np.isnan(dist), np.inf, dist), axis=1)
    ahead = np.nansum(u > ball_u[:, None], axis=1) / some

    out["count"] = count
    out["depth"] = cu
    out["side"] = cv
    out["length"] = su
    out["width"] = sv
    out["line"] = back
    out["front"] = front
    out["advance"] = np.clip(advance, -8.0, 8.0)
    out["line_move"] = np.clip(back_move, -8.0, 8.0)
    out["pace"] = pace
    out["fast"] = fast
    out["ball_depth"] = ball_u
    out["ball_past_line"] = np.clip(ball_u - back, -40.0, 40.0)
    out["ahead_of_ball"] = ahead
    out["second_near"] = np.minimum(dist[:, 1], 40.0)
    out["third_near"] = np.minimum(dist[:, 2], 40.0)
    out["near_10m"] = (dist < 10.0).sum(axis=1)
    return out


def compute(match, bx, by):
    """{name_home / name_away: [n_frames] float32}. bx, by: the gap-filled ball."""
    length, width = match.pitch
    n = match.n_frames
    sign = np.ones(n, dtype=np.float32)
    for (start, stop), s in zip(match.runs, attack_signs(match)):
        sign[start:stop] = s
    ball_x = np.nan_to_num(bx, nan=length / 2.0)
    ball_y = np.nan_to_num(by, nan=width / 2.0)

    result = {}
    for name, team in SIDES:
        # The home team's frame follows `sign`; the away team attacks the other way.
        forward = sign if team == HOME else -sign
        flip = forward < 0
        cols = np.flatnonzero(match.pteam == team)
        parts = []
        for a in range(0, n, CHUNK):
            b = min(n, a + CHUNK)
            f = flip[a:b]
            u = np.where(f[:, None], length - match.px[a:b][:, cols], match.px[a:b][:, cols])
            v = np.where(f[:, None], width - match.py[a:b][:, cols], match.py[a:b][:, cols])
            ball_u = np.where(f, length - ball_x[a:b], ball_x[a:b])
            ball_v = np.where(f, width - ball_y[a:b], ball_y[a:b])
            parts.append(_team(u.astype(np.float32), v.astype(np.float32),
                               ball_u.astype(np.float32), ball_v.astype(np.float32), match.fps))
        for key in parts[0]:
            values = np.concatenate([p[key] for p in parts]).astype(np.float32)
            values[~np.isfinite(values)] = 0.0
            result["{}_{}".format(key, name)] = values
    return result


def swap(shape):
    """The same measurements with the two teams' names exchanged."""
    out = {}
    for key, values in shape.items():
        stem, side = key.rsplit("_", 1)
        out["{}_{}".format(stem, "away" if side == "home" else "home")] = values
    return out
