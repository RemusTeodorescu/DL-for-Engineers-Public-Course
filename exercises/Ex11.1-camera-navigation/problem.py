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
