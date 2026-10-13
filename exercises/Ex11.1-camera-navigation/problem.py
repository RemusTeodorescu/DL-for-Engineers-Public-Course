r"""Ex_11.1 — the problem: a steering target from a camera frame.

*Deep Learning for Engineering* — MSc, Aalborg University.
Remus Teodorescu (ret@et.aau.dk), with support from Research Assistant
Noman Khan (nomank@energy.aau.dk).

**Reference texts.** Prince, *Understanding Deep Learning* (2023), ch. 10
(convolutional networks); Goodfellow, Bengio & Courville, *Deep Learning*
(2016), ch. 9. These are the works to read for the theory. The code, the
problem and the exposition here are original to this course.

## The problem

A small car follows a line of bright tape on a floor. Its camera gives a
frame; the policy has to say where the line is a little way ahead, as one
number between -1 (the left edge of the frame) and +1 (the right edge). The
steering follows from that number. This is L11.1's *Road Following as
Regression*.

## The frames are synthetic

**The frames here are drawn by this file, not taken by the car's camera.**
48 x 64 pixels, grey: a floor with a brightness slope across it, a tape line
whose position is set by the car's offset from the line, its heading and the
curve of the road, pixel noise, and on some frames a bright patch of glare, as
from a window. Because the frame is drawn, the true target is known exactly,
which is what makes the comparison in the notebook possible. On the car the
label is a click, and it is the on-car notebook that collects those.

## What is in here

The frame generator, three ways of collecting a dataset, the classical line
detector, a closed-loop simulation of the car on a track with a given loop
rate, and the rounding of a network's weights to a number of bits. The network
itself is written in the notebook.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "H", "W", "LOOK_AHEAD", "ROW", "frames", "shrink", "detect",
    "WHEELBASE", "STEER_MAX", "STRAIGHT", "RADIUS", "LAP", "HALF_CLEARANCE",
    "centreline", "drive", "round_weights", "describe_problem",
]

# ------------------------------------------------------------------ the frame
H, W = 48, 64            #: pixels: rows (far at the top, near at the bottom) and columns
LOOK_AHEAD = 0.6         #: where the target is read: 0 is the bottom row, 1 the top row
ROW = int(round((1.0 - LOOK_AHEAD) * (H - 1)))     #: the row of the target, 19
FLOOR, TAPE, NOISE = 0.35, 0.55, 0.04              #: floor brightness, the tape above it, pixel noise

#: how a dataset is collected: the spread of the car's offset and heading, and the share of frames with glare
COLLECTION = {
    "line":       dict(offset=0.10, heading=0.10, glare=0.0),    # driven along the line only
    "off-line":   dict(offset=0.60, heading=0.45, glare=0.0),    # also from where the car should not be
    "glare":      dict(offset=0.60, heading=0.45, glare=1.0),    # the same, every frame with a patch of glare
    "deliberate": dict(offset=0.60, heading=0.45, glare=0.3),    # collected on purpose: everything the car will meet
}


def frames(n, kind="deliberate", seed=0):
    """``n`` frames collected the way ``kind`` says (see :data:`COLLECTION`),
    and the true target of each. Returns ``(images, targets)``: images of
    shape ``(n, 1, H, W)`` with values 0 to 1, targets of shape ``(n,)``
    between -1 and 1."""
    c = COLLECTION[kind]
    rng = np.random.default_rng(seed)
    off = rng.uniform(-c["offset"], c["offset"], n)             # the car beside the line
    head = rng.uniform(-c["heading"], c["heading"], n)          # the car not pointing along it
    curv = rng.uniform(-0.5, 0.5, n)                            # the road curving
    depth = ((H - 1 - np.arange(H)) / (H - 1))[None, :]         # 0 at the bottom row, 1 at the top
    line = off[:, None] * (1 - 0.6 * depth) + head[:, None] * depth + curv[:, None] * depth ** 2
    x_line = (W - 1) / 2 * (1 + line)                           # the tape's column in every row, (n, H)
    half = (3.6 - 2.2 * depth) / 2                              # the tape narrows with distance
    cols = np.arange(W)[None, None, :]
    tape = np.clip(half[:, :, None] + 0.5 - np.abs(cols - x_line[:, :, None]), 0.0, 1.0)
    slope = rng.uniform(-0.15, 0.15, n)[:, None, None] * (cols / (W - 1) - 0.5)   # light from one side
    img = FLOOR + slope + TAPE * tape
    glare = rng.random(n) < c["glare"]
    if glare.any():
        rows = np.arange(H)[None, :, None]
        gy = rng.uniform(ROW - 6, ROW + 6, n)[:, None, None]
        side = rng.choice([-1.0, 1.0], n)
        gx = np.clip(x_line[:, ROW] + side * rng.uniform(12, 26, n), 3, W - 4)[:, None, None]
        patch = 0.6 * np.exp(-((cols - gx) / 5.0) ** 2 - ((rows - gy) / 3.0) ** 2)
        img = img + patch * glare[:, None, None]
    img = np.clip(img + rng.normal(0.0, NOISE, img.shape), 0.0, 1.0)
    target = np.clip(line[:, ROW], -1.0, 1.0)
    return img[:, None].astype(np.float32), target.astype(np.float32)


def shrink(images, factor):
    """The frames at a lower resolution: every ``factor`` x ``factor`` block of
    pixels replaced by its mean."""
    if factor == 1:
        return images
    n, c, h, w = images.shape
    return images.reshape(n, c, h // factor, factor, w // factor, factor).mean(axis=(3, 5))


def detect(images, threshold=0.15):
    """The classical detector. In a band of rows round the look-ahead row,
    take what is brighter than the floor by ``threshold``, and return the
    brightness-weighted mean column, between -1 and 1. A frame with nothing
    that bright returns 0, straight ahead. Works at any resolution
    :func:`shrink` produces."""
    n, _, h, w = images.shape
    row = int(round((1.0 - LOOK_AHEAD) * (h - 1)))
    r = max(1, round(2 * h / H))
    band = images[:, 0, max(row - r, 0): row + r + 1, :]        # a few rows round the look-ahead row
    floor = np.median(band, axis=(1, 2), keepdims=True)         # the floor's brightness in this frame
    weight = np.clip(band - floor - threshold, 0.0, None)       # what is clearly brighter than the floor
    total = weight.sum(axis=(1, 2))
    col = (weight * np.arange(w)[None, None, :]).sum(axis=(1, 2)) / np.maximum(total, 1e-9)
    return np.where(total > 0, col / ((w - 1) / 2) - 1.0, 0.0)


# ------------------------------------------------- the car and the track: ASSUMED, measure your own
WHEELBASE = 0.20          #: m, assumed until measured on the car
STEER_MAX = np.radians(30.0)   #: steering limit, assumed until measured
STRAIGHT, RADIUS = 1.4, 0.7    #: m: the vendor's 3 x 2 m map as a stadium
LAP = 2 * STRAIGHT + 2 * np.pi * RADIUS            #: 7.20 m
HALF_CLEARANCE = 0.175    #: m, half the narrowest gap of the on-car notebook's placeholder course (0.35 m)
AIM = 0.40                #: m, how far ahead on the line the controller aims


def centreline(s):
    """The point of the track's centreline at arc length ``s`` (any real
    number; the lap closes), as ``(x, y)``."""
    s = np.mod(s, LAP)
    a, b = STRAIGHT, STRAIGHT + np.pi * RADIUS
    x = np.where(s < a, s, np.where(s < b, a + RADIUS * np.sin((s - a) / RADIUS),
        np.where(s < b + a, a - (s - b), -RADIUS * np.sin((s - b - a) / RADIUS))))
    y = np.where(s < a, 0.0, np.where(s < b, RADIUS * (1 - np.cos((s - a) / RADIUS)),
        np.where(s < b + a, 2 * RADIUS, RADIUS * (1 + np.cos((s - b - a) / RADIUS)))))
    return x, y


_S_GRID = np.linspace(0.0, LAP, 2881)
_XY_GRID = np.column_stack(centreline(_S_GRID))


def drive(rate_hz, speed, target_error=0.0, laps=2.0, seed=0, dt=1e-3):
    """Drive the bicycle model round the track with a new steering decision
    ``rate_hz`` times a second. Computing a decision takes one period, so the
    command the wheels follow was computed from the frame before, and is held
    until the next: the car is blind for ``speed / rate_hz`` metres and acts
    on a view that old again. Each decision aims at the line ``AIM`` ahead,
    seen with a random error of standard deviation ``target_error`` (in the
    frame's units, where 1 is half the frame). Returns the largest distance
    from the centreline in metres, after the first quarter lap."""
    rng = np.random.default_rng(seed)
    x, y, th = 0.0, 0.0, 0.0
    steer, pending, next_decision, worst = 0.0, 0.0, 0.0, 0.0
    period = 1.0 / rate_hz
    n = int(laps * LAP / speed / dt)
    for k in range(n):
        t = k * dt
        d2 = (_XY_GRID[:, 0] - x) ** 2 + (_XY_GRID[:, 1] - y) ** 2
        i = int(np.argmin(d2))
        if t >= next_decision - 1e-12:                           # a new frame, a new decision
            tx, ty = centreline(_S_GRID[i] + AIM)
            alpha = np.arctan2(ty - y, tx - x) - th              # the angle to the aim point
            alpha = np.arctan2(np.sin(alpha), np.cos(alpha)) + target_error * 0.5 * rng.normal()
            steer = pending                                      # the command computed from the frame before
            pending = float(np.clip(np.arctan(2 * WHEELBASE * np.sin(alpha) / AIM), -STEER_MAX, STEER_MAX))
            next_decision += period
        x += speed * np.cos(th) * dt                             # the kinematic bicycle model
        y += speed * np.sin(th) * dt
        th += speed / WHEELBASE * np.tan(steer) * dt
        if t * speed > 0.25 * LAP:
            worst = max(worst, float(np.sqrt(d2[i])))
    return worst


def round_weights(model, bits):
    """A copy of ``model`` with every weight rounded onto ``2**bits`` levels,
    one scale per layer: what storing the network in ``bits`` bits does to
    it. The biases are kept."""
    import copy
    import torch
    out = copy.deepcopy(model)
    with torch.no_grad():
        for p in out.parameters():
            if p.dim() > 1:                                      # a weight, not a bias
                scale = p.abs().max() / (2 ** (bits - 1) - 1)
                p.copy_(torch.round(p / scale) * scale)
    return out


def describe_problem() -> None:
    """Print the frame, the car and the numbers they imply."""
    print("  the frames are SYNTHETIC, drawn by problem.py - not the car's camera")
    print(f"  frame            : {H} x {W} pixels, grey; the target is read in row {ROW}, {LOOK_AHEAD:.1f} of the way up")
    print(f"  tape             : {TAPE:.2f} brighter than a floor of {FLOOR:.2f}; pixel noise {NOISE:.2f}")
    print(f"  one pixel        : {2 / (W - 1):.3f} of the target's range of -1 to 1")
    print("  the car's wheelbase, steering limit and track are ASSUMED until measured")
    print(f"  car              : wheelbase {WHEELBASE:.2f} m, steering limit {np.degrees(STEER_MAX):.0f} degrees")
    print(f"  track            : a stadium, straights {STRAIGHT:.1f} m, turns of radius {RADIUS:.1f} m, lap {LAP:.2f} m")
    for v, f in ((1.0, 30.0), (2.0, 20.0), (2.0, 5.0)):
        print(f"  blind distance   : {100 * v / f:.0f} cm at {v:.0f} m/s and {f:.0f} decisions a second")


# ------------------------------------------------------- obstacles (Ex_11 sections 5 - 6)
#: an obstacle blocks the path when it stands in the near part of the frame
#: (below this depth, 0 the bottom row, 1 the top row) and overlaps the lane
BLOCK_DEPTH = 0.45
LANE_HALF = 0.30          #: half the lane the car needs, in the frame's units (1 is half the frame)


def obstacle_frames(n, seed=0, p_obstacle=0.8, p_shadow=0.4):
    """``n`` frames of the line with, on some of them, an obstacle - a box on
    the floor, dark with a lighter front face and sharp edges, smaller the
    further away it stands - and on some a shadow, a soft dark patch with no
    edges, which is NOT an obstacle. Returns ``(images, blocked, where)``:
    images ``(n, 1, H, W)``, ``blocked`` 1 when an obstacle stands in the lane
    in the near part of the frame (:data:`BLOCK_DEPTH`, :data:`LANE_HALF`),
    and ``where`` the obstacle's ``(depth, lateral)`` or ``nan``."""
    rng = np.random.default_rng(seed)
    img, _ = frames(n, "off-line", seed=seed + 1000)
    img = img[:, 0].copy()
    rows = np.arange(H)[:, None]
    cols = np.arange(W)[None, :]
    blocked = np.zeros(n, np.int64)
    where = np.full((n, 2), np.nan)
    has_obs = rng.random(n) < p_obstacle
    has_shadow = rng.random(n) < p_shadow
    for k in range(n):
        if has_shadow[k]:                                  # a soft dark patch: not an obstacle
            sx, sy = rng.uniform(8, W - 8), rng.uniform(H * 0.4, H - 4)
            img[k] -= 0.22 * np.exp(-((cols - sx) / rng.uniform(5, 9)) ** 2 - ((rows - sy) / rng.uniform(3, 6)) ** 2)
        if has_obs[k]:
            depth = rng.uniform(0.05, 0.95)                # 0 near, 1 far
            lat = rng.uniform(-0.7, 0.7)                   # the obstacle's centre, -1 left edge, +1 right edge
            scale = 1.0 - 0.7 * depth
            bw, bh = 14 * scale, 11 * scale                # its size in pixels
            base = (H - 1) * (1 - depth)                   # the row where it stands on the floor
            cx = (W - 1) / 2 * (1 + lat)
            inside = (np.abs(cols - cx) < bw / 2) & (rows <= base) & (rows > base - bh)
            face = inside & (rows > base - 0.35 * bh)
            img[k] = np.where(inside, 0.12, img[k])        # the dark box
            img[k] = np.where(face, 0.30, img[k])          # its lighter front face
            where[k] = depth, lat
            blocked[k] = int(depth < BLOCK_DEPTH and abs(lat) < LANE_HALF + bw / W)
    img = np.clip(img + rng.normal(0, NOISE / 2, img.shape), 0, 1)
    return img[:, None].astype(np.float32), blocked, where


def dark_rule(images, threshold=0.12):
    """The classical obstacle rule: in the lower half of the frame, in the
    lane, is anything darker than the floor by ``threshold``? Returns 1 for
    blocked. A shadow is dark too, which is where it fails."""
    n, _, h, w = images.shape
    near = images[:, 0, h // 2:, int(w * (0.5 - LANE_HALF / 2)): int(w * (0.5 + LANE_HALF / 2)) + 1]
    floor = np.median(images[:, 0, h // 2:, :], axis=(1, 2))
    dark = (near < (floor[:, None, None] - threshold)).mean(axis=(1, 2))
    return (dark > 0.04).astype(np.int64)


# ---------------------------------------------- the A-to-B course and the avoidance strategies
#: obstacles on a straight course from A (x = 0) to B (x = COURSE): (x, y, radius) in metres, ASSUMED
COURSE = 6.0
OBSTACLES = ((1.5, 0.00, 0.08), (3.0, 0.06, 0.08), (4.5, -0.05, 0.08))
CAR_HALF = 0.10           #: m, half the car's width, assumed
RANGE = 0.70              #: m, how far ahead the camera's "blocked" region reaches, assumed
K_FIELD = 0.03            #: the potential field's strength, m


def avoid(strategy, rate_hz=10.0, speed=1.0, p_miss=0.0, p_false=0.0, seed=0, dt=2e-3, t_max=30.0):
    """Drive from A to B with one avoidance strategy and a decision ``rate_hz``
    times a second. Every decision sees "blocked" when an obstacle is within
    :data:`RANGE` ahead and inside the lane, wrongly with the classifier's
    miss and false-alarm rates. Strategies:

    ``"stop"``   blocked: stand still; free: follow the line;
    ``"steer"``  blocked: aim at a lane 0.30 m to the side away from the
                 obstacle, slowed to 60 %, held 0.5 m past it, then back;
    ``"field"``  one aim distance ahead, aim at the lowest point across the
                 lane of ``U(y) = y**2 + sum K/d_i(y)``: the pull back to the
                 line plus a push from every obstacle within range - it needs
                 where the obstacle is, not only that it is there.

    Returns ``dict(time, clearance, path, reached)``: the time to B (``inf``
    when B is not reached), the smallest gap between car and obstacle in
    metres (negative is a collision) and the path."""
    rng = np.random.default_rng(seed)
    x, y, th = 0.0, 0.0, 0.0
    steer, v, y_ref, hold_until = 0.0, speed, 0.0, -1.0
    next_dec, t, clear, path = 0.0, 0.0, np.inf, [(0.0, 0.0)]
    obs = np.array(OBSTACLES)
    while t < t_max and x < COURSE:
        if t >= next_dec - 1e-12:
            next_dec += 1.0 / rate_hz
            rel = obs[:, :2] - [x, y]
            fwd = rel @ [np.cos(th), np.sin(th)]
            side = rel @ [-np.sin(th), np.cos(th)]
            ahead = (fwd > 0) & (fwd < RANGE) & (np.abs(side) < 0.15 + obs[:, 2] + CAR_HALF)
            seen = bool(ahead.any())
            if seen and rng.random() < p_miss:
                seen = False
            elif not seen and rng.random() < p_false:
                seen = True
            if strategy == "stop":
                v = 0.0 if seen else speed
                target = (x + AIM, 0.0)
            elif strategy == "steer":
                if seen and ahead.any():
                    i = int(np.argmin(np.where(ahead, fwd, np.inf)))
                    y_ref = -0.30 if obs[i, 1] >= y else 0.30
                    hold_until = obs[i, 0] + 0.5
                v = 0.6 * speed if x < hold_until else speed
                if x >= hold_until:
                    y_ref = 0.0
                target = (x + AIM, y_ref)
            else:                                          # potential field
                # U one aim distance ahead, across the lane: the pull back to the
                # line, (y - 0)^2, plus K/d for every obstacle within range; the
                # car aims at the lowest point of U.
                ax = x + AIM
                ys = np.linspace(-0.6, 0.6, 241)
                U = ys ** 2
                for (ox, oy, r) in obs:
                    if abs(ox - x) < RANGE and ox > x - r:
                        d = np.maximum(np.hypot(ax - ox, ys - oy) - r - CAR_HALF, 0.01)
                        U = U + K_FIELD / d
                v = speed
                target = (ax, float(ys[np.argmin(U)]))
            alpha = np.arctan2(target[1] - y, target[0] - x) - th
            steer = float(np.clip(np.arctan(2 * WHEELBASE * np.sin(alpha) / AIM), -STEER_MAX, STEER_MAX))
        x += v * np.cos(th) * dt
        y += v * np.sin(th) * dt
        th += v / WHEELBASE * np.tan(steer) * dt
        t += dt
        gaps = np.hypot(obs[:, 0] - x, obs[:, 1] - y) - obs[:, 2] - CAR_HALF
        clear = min(clear, float(gaps.min()))
        if len(path) == 0 or abs(x - path[-1][0]) > 0.02 or abs(y - path[-1][1]) > 0.01:
            path.append((round(x, 3), round(y, 3)))
    return dict(time=t if x >= COURSE else np.inf, clearance=clear, path=path, reached=x >= COURSE)
