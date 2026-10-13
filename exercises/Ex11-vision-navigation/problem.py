r"""Ex_11 — the problem: a car that drives from A to B on what its camera sees.

*Deep Learning for Engineering* — MSc, Aalborg University.
Remus Teodorescu (ret@et.aau.dk), with support from Research Assistant
Noman Khan (nomank@energy.aau.dk).

**Reference texts.** Prince, *Understanding Deep Learning* (2023), ch. 5
(loss functions), ch. 10 (convolutional networks); Goodfellow, Bengio &
Courville, *Deep Learning* (2016), ch. 9. These are the works to read for the
theory. The code, the problem and the exposition here are original to this
course.

## The problem

A small car drives from A to B along a line of bright tape, past obstacles
standing on it. From its camera frame it answers two questions of L11.1 -
where is the path, and how do I steer onto it - and two of L11.2 - is the path
blocked, and what do I do about it.

## Everything is drawn or simulated

**The frames here are drawn by this file, not taken by a camera**: 48 x 64
grey pixels, a floor with a brightness slope, the tape, pixel noise, on some
frames a patch of glare, a box, or a shadow. Because the frame is drawn, the
true answer is known exactly. **The car is simulated too**: a kinematic
bicycle model whose wheelbase, steering limit, width and friction are assumed.
Nothing here needs the real JetRacer.

## What is in here

The frame generators (the line; obstacles and shadows), the classical line
detector and dark-pixel rule, the lap on a stadium track at a given loop rate,
the course from A to B with three avoidance strategies, the rounding of a
network's weights to a number of bits, and, for the mini projects, the camera,
a LiDAR and an ultrasonic sensor simulated from one geometry on random
courses. The networks are written in the notebook.
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
HALF_CLEARANCE = 0.175    #: m, half the narrowest gap of an assumed course, 0.35 m
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
    """Print the frame, the car, the course and the numbers they imply."""
    print("  the frames are DRAWN by problem.py and the car is SIMULATED - no real car is needed")
    print(f"  frame            : {H} x {W} pixels, grey; the line is read in row {ROW}, {LOOK_AHEAD:.1f} of the way up")
    print(f"  tape             : {TAPE:.2f} brighter than a floor of {FLOOR:.2f}; pixel noise {NOISE:.2f}")
    print(f"  one pixel        : {2 / (W - 1):.3f} of the target's range of -1 to 1")
    print("  the car's size, its steering, the course and the friction are ASSUMED")
    print(f"  car              : wheelbase {WHEELBASE:.2f} m, steering limit {np.degrees(STEER_MAX):.0f} degrees, "
          f"half-width {100 * CAR_HALF:.0f} cm, aims {100 * AIM:.0f} cm ahead")
    print(f"  course A to B    : {COURSE:.1f} m, {len(OBSTACLES)} obstacles of radius {100 * OBSTACLES[0][2]:.0f} cm; "
          f"the camera judges free or blocked up to {RANGE:.1f} m ahead")
    print(f"  lap (section 7)  : a stadium, straights {STRAIGHT:.1f} m, turns of radius {RADIUS:.1f} m, lap {LAP:.2f} m")
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


def field_aim(x, near, k=None):
    """The potential field's choice. One aim distance ahead of the car at
    ``x``, across the lane, ``U(y) = y**2 + sum K/d_i(y)``: the pull back to
    the line plus a push from every obstacle in ``near`` (rows of ``(x, y,
    radius)``), ``d_i`` the gap between the car and obstacle ``i``. Returns
    the sideways position where ``U`` is lowest."""
    k = K_FIELD if k is None else k
    ax = x + AIM
    ys = np.linspace(-0.6, 0.6, 241)
    U = ys ** 2
    for (ox, oy, r) in near:
        d = np.maximum(np.hypot(ax - ox, ys - oy) - r - CAR_HALF, 0.01)
        U = U + k / d
    return float(ys[np.argmin(U)])


def avoid(strategy, rate_hz=10.0, speed=1.0, p_miss=0.0, p_false=0.0, seed=0, dt=2e-3, t_max=30.0):
    """Drive from A to B with one avoidance strategy and a decision ``rate_hz``
    times a second. Every decision sees "blocked" when an obstacle is within
    :data:`RANGE` ahead and inside the lane, wrongly with the classifier's
    miss and false-alarm rates. Strategies:

    ``"stop"``   blocked: stand still; free: follow the line;
    ``"steer"``  blocked: aim at a lane 0.30 m to the side away from the
                 obstacle, slowed to 60 %, held 0.5 m past it, then back;
    ``"field"``  one aim distance ahead, aim at the lowest point across the
                 lane of ``U(y) = y**2 + sum K/d_i(y)`` (:func:`field_aim`):
                 the pull back to the line plus a push from every obstacle
                 within range - it needs where the obstacle is, not only that
                 it is there;
    a function   ``f(x, near)`` returning the sideways position to aim at,
                 written like :func:`field_aim`.

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
            else:                                          # potential field, or a function of your own
                near = np.array([o for o in obs if abs(o[0] - x) < RANGE and o[0] > x - o[2]]).reshape(-1, 3)
                aim = field_aim if strategy == "field" else strategy
                v = speed
                target = (x + AIM, float(aim(x, near)))
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


def draw_course(ax=None, paths=None, dark=False):
    """The A-to-B course from above: the line, the three obstacles with the
    car's half-width around them, and, if given, ``paths`` as
    ``{label: [(x, y), ...]}``. Returns the axes."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 2.6))
    fg = "white" if dark else "black"
    ax.axhline(0, color="0.6", lw=3, zorder=0)                         # the taped line
    for (ox, oy, r) in OBSTACLES:
        ax.add_patch(plt.Circle((ox, oy), r + CAR_HALF, color="tab:red", alpha=0.15, lw=0))
        ax.add_patch(plt.Circle((ox, oy), r, color="tab:red"))
    for x, name in ((0.0, "A"), (COURSE, "B")):
        ax.plot([x, x], [-0.5, 0.5], color=fg, lw=1.5)
        ax.text(x, 0.52, name, ha="center", va="bottom", color=fg, fontsize=11, weight="bold")
    for label, p in (paths or {}).items():
        p = np.asarray(p)
        ax.plot(p[:, 0], p[:, 1], lw=1.6, label=label)
    if paths:
        ax.legend(loc="lower right", fontsize=8, ncol=len(paths))
    ax.set_xlim(-0.2, COURSE + 0.2); ax.set_ylim(-0.6, 0.7); ax.set_aspect("equal")
    ax.set_xlabel("along the course [m]", color=fg); ax.set_ylabel("sideways [m]", color=fg)
    ax.tick_params(colors=fg)
    return ax


# ====================================================================== the mini projects
# MP11A (the camera with a LiDAR) and MP11B (the camera with an ultrasonic
# sensor). One car on random A-to-B courses; obstacles of three kinds; the
# camera frame, the LiDAR scan and the ultrasonic range all computed from the
# same geometry, so the true answer is known. The ground truth is built from
# these functions by tools/miniprojects/ex11_truth.py.
#
# Sensor values marked VENDOR are the manufacturers' published figures,
# confirmed by the lecturer for the mini projects on 13 October 2026 (policy
# C10 rule 5). Values marked ASSUMED are this course's choices.

Z_FAR = RANGE / BLOCK_DEPTH   #: m, the distance drawn at the top row: the frame maps 0 to 1.56 m ahead linearly
HALF_VIEW = 0.60          #: m, the sideways distance drawn at the frame's edge, at every depth (ASSUMED; no perspective)

#: obstacle kinds: grey level of the box and of its front face, and its height [m]
KINDS = {
    "box":  dict(shade=0.12, face=0.30, height=0.20),   # dark, tall: every sensor sees it
    "grey": dict(shade=0.40, face=0.43, height=0.20),   # as light as the floor: the camera barely sees it
    "low":  dict(shade=0.12, face=0.30, height=0.05),   # dark and low: under the LiDAR's plane
}

LIDAR_BEAMS = 360         #: one range a degree (VENDOR: RPLIDAR A1, angular resolution 1 degree or finer)
LIDAR_MIN, LIDAR_MAX = 0.15, 12.0   #: m (VENDOR: RPLIDAR A1M8, 0.15 - 12 m)
LIDAR_HZ = 5.5            #: scans a second (VENDOR: typical 5.5 Hz)
LIDAR_HEIGHT = 0.12       #: m, the scan plane above the floor on the car (ASSUMED: the mount)
LIDAR_NOISE = 0.01        #: of the distance, standard deviation (ASSUMED)
SONAR_MIN, SONAR_MAX = 0.02, 5.00   #: m (VENDOR: DFRobot URM09, 2 - 500 cm)
SONAR_STEP = 0.01         #: m (VENDOR: 1 cm resolution)
SONAR_NOISE = 0.01        #: of the distance (VENDOR: accuracy 1 %)
SONAR_CONE = np.radians(30.0)   #: half-angle of the beam, 60 degrees in all (set by the lecturer, 13 October 2026)
SONAR_HZ = 20.0           #: readings a second (ASSUMED)
SONAR_HEIGHT = 0.04       #: m, the sensor's height on the bumper (ASSUMED): it sees low obstacles


def make_course(seed, n=4):
    """A random A-to-B course: ``n`` obstacles at least 0.9 m apart along it,
    each ``(x, y, radius, kind)``, and two shadows on the floor ``(x, y)``,
    which are not obstacles."""
    rng = np.random.default_rng(seed)
    xs = np.sort(rng.uniform(1.0, COURSE - 0.8, 40))
    pick = [xs[0]]
    for x in xs[1:]:
        if x - pick[-1] >= 0.9 and len(pick) < n:
            pick.append(x)
    obstacles = [(float(x), float(rng.uniform(-0.15, 0.15)), float(rng.uniform(0.06, 0.10)),
                  str(rng.choice(list(KINDS)))) for x in pick]
    shadows = [(float(rng.uniform(0.8, COURSE - 0.5)), float(rng.uniform(-0.3, 0.3))) for _ in range(2)]
    return dict(obstacles=obstacles, shadows=shadows, seed=seed)


def _relative(x, y, th, px, py):
    """Forward and sideways distance of the point (px, py) seen from the car."""
    dx, dy = px - x, py - y
    return dx * np.cos(th) + dy * np.sin(th), -dx * np.sin(th) + dy * np.cos(th)


def render(x, y, th, course, rng):
    """The camera frame at the car's pose: the floor, the taped line along
    y = 0, the shadows and the obstacles ahead, drawn as
    :func:`obstacle_frames` draws them. Shape ``(1, H, W)``."""
    depth = ((H - 1 - np.arange(H)) / (H - 1))[:, None]               # 0 bottom row, 1 top row
    cols = np.arange(W)[None, :]
    rows = np.arange(H)[:, None]
    z = depth * Z_FAR                                                 # metres ahead in each row
    line = (-y - np.tan(th) * z) / HALF_VIEW                          # the line y = 0, in frame units
    xl = (W - 1) / 2 * (1 + line)
    half = (3.6 - 2.2 * depth) / 2
    img = FLOOR + rng.uniform(-0.15, 0.15) * (cols / (W - 1) - 0.5)
    img = img + TAPE * np.clip(half + 0.5 - np.abs(cols - xl), 0, 1)
    for (sx, sy) in course["shadows"]:
        f, s = _relative(x, y, th, sx, sy)
        if 0 < f < Z_FAR:
            r0, c0 = (H - 1) * (1 - f / Z_FAR), (W - 1) / 2 * (1 + s / HALF_VIEW)
            img = img - 0.22 * np.exp(-((cols - c0) / 7) ** 2 - ((rows - r0) / 4) ** 2)
    for (ox, oy, r, kind) in sorted(course["obstacles"], key=lambda o: -_relative(x, y, th, o[0], o[1])[0]):
        f, s = _relative(x, y, th, ox, oy)
        d = f / Z_FAR
        if not 0.02 < d < 0.98:
            continue
        k = KINDS[kind]
        bw = 2 * r / HALF_VIEW * (W - 1) / 2 * (1 - 0.4 * d)
        bh = 11 * (1 - 0.7 * d) * k["height"] / 0.20 + 1
        base = (H - 1) * (1 - d)
        cx = (W - 1) / 2 * (1 + s / HALF_VIEW)
        inside = (np.abs(cols - cx) < bw / 2) & (rows <= base) & (rows > base - bh)
        img = np.where(inside, k["shade"], img)
        img = np.where(inside & (rows > base - 0.35 * bh), k["face"], img)
    img = np.clip(img + rng.normal(0, NOISE, img.shape), 0, 1)
    return img[None].astype(np.float32)


def _ray_circle(x, y, ang, cx, cy, r):
    """Distance along the ray from (x, y) at angle ``ang`` to the circle, or inf."""
    dx, dy = np.cos(ang), np.sin(ang)
    fx, fy = x - cx, y - cy
    b = fx * dx + fy * dy
    c = fx * fx + fy * fy - r * r
    disc = b * b - c
    t = -b - np.sqrt(np.maximum(disc, 0))
    return np.where((disc >= 0) & (t > 0), t, np.inf)


def lidar(x, y, th, course, rng):
    """One LiDAR scan at the car's pose: ``LIDAR_BEAMS`` ranges in metres,
    beam ``i`` at ``i`` degrees anticlockwise from straight ahead. Only
    obstacles taller than the scan plane return; no return reads 0, as the
    RPLIDAR does."""
    ang = th + np.radians(np.arange(LIDAR_BEAMS) * 360.0 / LIDAR_BEAMS)
    d = np.full(LIDAR_BEAMS, np.inf)
    for (ox, oy, r, kind) in course["obstacles"]:
        if KINDS[kind]["height"] > LIDAR_HEIGHT:
            d = np.minimum(d, _ray_circle(x, y, ang, ox, oy, r))
    d = d * (1 + LIDAR_NOISE * rng.normal(size=d.shape))
    return np.where((d >= LIDAR_MIN) & (d <= LIDAR_MAX), d, 0.0).astype(np.float32)


def sonar(x, y, th, course, rng):
    """One ultrasonic reading at the car's pose: the distance in metres to the
    nearest surface inside the beam's cone, rounded to the sensor's step;
    ``SONAR_MAX`` when nothing answers. It says how far, not where."""
    best = np.inf
    for (ox, oy, r, kind) in course["obstacles"]:
        f, s = _relative(x, y, th, ox, oy)
        dist = np.hypot(f, s)
        if f <= 0 or dist <= r:
            continue
        if abs(np.arctan2(s, f)) - np.arcsin(r / dist) < SONAR_CONE:
            best = min(best, dist - r)
    if not np.isfinite(best):
        return SONAR_MAX
    best = best * (1 + SONAR_NOISE * rng.normal())
    return float(np.clip(np.round(best / SONAR_STEP) * SONAR_STEP, SONAR_MIN, SONAR_MAX))


def truth_at(x, y, th, course):
    """The true answer at a pose: ``blocked`` (an obstacle within RANGE ahead
    and in the car's lane, as :func:`avoid` decides), and the nearest such
    obstacle's forward and sideways distance from the car and its kind."""
    hits = []
    for (ox, oy, r, kind) in course["obstacles"]:
        f, s = _relative(x, y, th, ox, oy)
        if 0 < f < RANGE and abs(s) < 0.15 + r + CAR_HALF:
            hits.append((f, s, kind))
    if not hits:
        return dict(blocked=0, forward=np.nan, sideways=np.nan, kind="")
    f, s, kind = min(hits)
    return dict(blocked=1, forward=f, sideways=s, kind=kind)


class SteerAround:
    """The steer-around rule as a decision function for :func:`run_course`:
    ``decide(reading) -> (speed_factor, y_ref)``. The reading carries
    ``blocked`` and where the obstacle stands, ``obstacle_x`` and
    ``obstacle_y`` on the course; the car aims at a lane 0.30 m beside the
    obstacle, on the side it is already on, slowed to 60 %, until 0.5 m past
    it. :func:`oracle` supplies the true reading; a fused network its own."""

    OFFSET = 0.30          #: m, from the obstacle's centre to the lane aimed at

    def __init__(self):
        self.y_ref, self.hold = 0.0, -1.0

    def __call__(self, r):
        if r["blocked"]:
            side = 1.0 if r["y"] >= r["obstacle_y"] else -1.0   # pass on the side the car is already on
            self.y_ref = r["obstacle_y"] + side * self.OFFSET
            self.hold = r["obstacle_x"] + 0.5
        if r["x"] >= self.hold:
            self.y_ref = 0.0
        return (0.6 if r["x"] < self.hold else 1.0), self.y_ref


def oracle(course):
    """The reading a perfect perception would give: ``perceive(x, y, th,
    sensors)`` for :func:`run_course`, from the true geometry."""
    def perceive(x, y, th, sensors=None):
        t = truth_at(x, y, th, course)
        if not t["blocked"]:
            return dict(blocked=0, obstacle_x=np.nan, obstacle_y=np.nan)
        return dict(blocked=1, obstacle_x=x + t["forward"] * np.cos(th) - t["sideways"] * np.sin(th),
                    obstacle_y=y + t["forward"] * np.sin(th) + t["sideways"] * np.cos(th))
    return perceive


def run_course(course, perceive, rate_hz=10.0, speed=1.0, dt=2e-3, t_max=30.0, seed=0):
    """Drive from A to B on ``course``. At every decision ``perceive(x, y,
    th, sensors)`` returns a reading for the steer-around rule; ``sensors``
    holds the latest camera frame, LiDAR scan and ultrasonic range, each
    refreshed at its own rate. Returns ``dict(time, clearance, reached,
    path)``: ``clearance`` the smallest gap in metres, negative a collision."""
    rng = np.random.default_rng(seed)
    decide = SteerAround()
    x = y = th = t = 0.0
    steer, v, y_ref = 0.0, speed, 0.0
    next_dec = next_lidar = next_sonar = 0.0
    sensors, clear, path = {}, np.inf, [(0.0, 0.0)]
    obs = np.array([o[:3] for o in course["obstacles"]])
    while t < t_max and x < COURSE:
        if t >= next_lidar - 1e-12:
            sensors["scan"], next_lidar = lidar(x, y, th, course, rng), next_lidar + 1 / LIDAR_HZ
        if t >= next_sonar - 1e-12:
            sensors["sonar"], next_sonar = sonar(x, y, th, course, rng), next_sonar + 1 / SONAR_HZ
        if t >= next_dec - 1e-12:
            sensors["frame"] = render(x, y, th, course, rng)
            reading = dict(perceive(x, y, th, sensors), x=x, y=y)
            fac, y_ref = decide(reading)
            v = speed * fac
            alpha = np.arctan2(y_ref - y, AIM) - th
            steer = float(np.clip(np.arctan(2 * WHEELBASE * np.sin(alpha) / AIM), -STEER_MAX, STEER_MAX))
            next_dec += 1 / rate_hz
        x += v * np.cos(th) * dt
        y += v * np.sin(th) * dt
        th += v / WHEELBASE * np.tan(steer) * dt
        t += dt
        clear = min(clear, float((np.hypot(obs[:, 0] - x, obs[:, 1] - y) - obs[:, 2] - CAR_HALF).min()))
        if abs(x - path[-1][0]) > 0.02 or abs(y - path[-1][1]) > 0.01:
            path.append((round(x, 3), round(y, 3)))
    return dict(time=t if x >= COURSE else np.inf, clearance=clear, reached=x >= COURSE, path=path)
