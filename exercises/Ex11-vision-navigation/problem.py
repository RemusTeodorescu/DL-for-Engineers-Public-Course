r"""Ex_11 — the problem: a virtual JetRacer that drives from A to B on what its
camera sees.

*Deep Learning for Engineering* — MSc, Aalborg University.
Remus Teodorescu (ret@et.aau.dk), with support from Research Assistant
Noman Khan (nomank@energy.aau.dk).

**Reference texts.** Prince, *Understanding Deep Learning* (2023), ch. 5
(loss functions), ch. 10 (convolutional networks); Goodfellow, Bengio &
Courville, *Deep Learning* (2016), ch. 9. These are the works to read for the
theory. The code, the problem and the exposition here are original to this
course.

## The virtual JetRacer

A digital twin of the Waveshare JetRacer Pro, with the interface of its own
software, so that code written here runs on the car with the import changed:

* ``car.steering`` and ``car.throttle``, each from -1 to 1, with
  ``steering_gain`` (-0.65), ``steering_offset`` (0) and ``throttle_gain``
  (0.8) as in the JetRacer library's ``NvidiaRacecar``: the steering servo
  receives ``steering * steering_gain + steering_offset``, the motor a duty of
  ``throttle * throttle_gain``;
* ``camera.read()``, a 224 x 224 colour frame (height, width, red-green-blue,
  0 - 255), the size ``CSICamera(width=224, height=224)`` gives on the car.

The frame is **rendered** from the car's position and heading on the floor
through a pinhole camera: the taped line, obstacles, shadows and glare. The
true answer to every question is therefore known.

**What the vendor states** (docs/hardware/jetracer-pro.md): Jetson Nano 4 GB,
IMX219-160 camera (8 MP, 160 degrees), Ackermann steering with a 6 kg cm
servo, an RC380 brushed motor, a 2S2P 18650 pack at 8.4 V, **no encoder**.
**What is assumed here, to be measured on the car**: the camera's height,
tilt and field of view (a pinhole stands in for the wide lens), the wheelbase,
the wheel angle at full servo command, the speed at a given duty and how fast
it is reached, the tyres' friction.

## What else is in here

The courses (A to B, and a closed lap), the classical line detector and
dark-pixel rule, the datasets the notebook trains on, the avoidance
strategies, the rounding of a network's weights to a number of bits, and, for
developing the mini projects before the real car, a LiDAR and an ultrasonic
sensor simulated from the same geometry. The networks are written in the
notebook.
"""

from __future__ import annotations

import numpy as np

# ====================================================================== the car
WHEELBASE = 0.20          #: m (ASSUMED until measured on the car)
STEER_MAX = np.radians(30.0)    #: largest wheel angle the linkage allows (ASSUMED)
SERVO_ANGLE = np.radians(35.0)  #: wheel angle at a servo command of 1 (ASSUMED)
V_PER_DUTY = 5.0          #: m/s at a motor duty of 1, on the floor (ASSUMED: no encoder; time it over a known distance)
SPEED_LAG = 0.30          #: s, time constant of the speed after a throttle change (ASSUMED)
MU = 0.60                 #: tyre-floor friction coefficient (ASSUMED)
G = 9.81                  #: m/s^2
CAR_HALF = 0.10           #: m, half the car's width (ASSUMED)

# ====================================================================== the camera
H = W = 224               #: pixels: the frame the car's network sees, CSICamera(width=224, height=224)
CAM_HEIGHT = 0.10         #: m above the floor (ASSUMED)
CAM_PITCH = np.radians(20.0)    #: tilted down from the horizontal (ASSUMED)
CAM_AHEAD = 0.12          #: m, the camera ahead of the rear axle (ASSUMED)
HFOV = np.radians(120.0)  #: horizontal field of view of the pinhole that stands in for the 160 degree lens (ASSUMED)
F_PX = (W / 2) / np.tan(HFOV / 2)     #: focal length in pixels, 64.7
CU = CV = (W - 1) / 2     #: the image centre
AIM = 0.40                #: m ahead of the camera: where the line is read, the aim point
TAPE_HALF = 0.0125        #: m, half the width of 25 mm tape (ASSUMED)

# per-pixel rays, in the car's frame (forward, left, up), scaled so that a
# ray reaches the floor at t = 1 where it reaches it at all
_u, _v = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
_RF = -(_v - CV) * np.sin(CAM_PITCH) + F_PX * np.cos(CAM_PITCH)
_RL = -(_u - CU)
_RU = -(_v - CV) * np.cos(CAM_PITCH) - F_PX * np.sin(CAM_PITCH)
FLOOR = _RU < 0                                       #: the pixels that see the floor
_T = np.where(FLOOR, CAM_HEIGHT / np.where(FLOOR, -_RU, 1.0), np.inf)
FLOOR_F = np.where(FLOOR, _RF * _T + CAM_AHEAD, np.nan)   #: m ahead of the rear axle, per pixel
FLOOR_L = np.where(FLOOR, _RL * _T, np.nan)               #: m to the left, per pixel
_PIX = np.where(FLOOR, np.abs(np.gradient(np.nan_to_num(FLOOR_L), axis=1)), 1.0)   #: a pixel's width on the floor, m
#: the row in which the floor lies AIM ahead of the camera: where the target is read
ROW = int(round(CV + F_PX * np.tan(np.arctan(CAM_HEIGHT / AIM) - CAM_PITCH)))
HORIZON = CV - F_PX * np.tan(CAM_PITCH)               #: the row of the horizon, about 88


# ====================================================================== the courses
COURSE = 6.0              #: m from A to B (ASSUMED: the size of the lab's floor)
STRAIGHT, RADIUS = 1.4, 0.7     #: m: the vendor's 3 x 2 m track map as a stadium
LAP = 2 * STRAIGHT + 2 * np.pi * RADIUS     #: 7.20 m


def tape_offset(X, Y, course="ab"):
    """Signed distance in metres from the floor point (X, Y) to the taped
    line, positive to the left of the direction of travel. ``course`` is
    ``"ab"`` (the straight line y = 0 from A to B) or ``"lap"`` (the stadium,
    driven anticlockwise)."""
    if course == "ab":
        return np.asarray(Y, dtype=float)
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    a, R = STRAIGHT, RADIUS
    right = R - np.hypot(X - a, Y - R)
    left = R - np.hypot(X, Y - R)
    mid = np.where(Y < R, Y, 2 * R - Y)
    return np.where(X > a, right, np.where(X < 0, left, mid))


def centreline(s, course="ab"):
    """The point of the line at arc length ``s`` and the direction of travel there."""
    if course == "ab":
        return s, 0.0 * s, 0.0 * s
    s = np.mod(s, LAP)
    a, b, R = STRAIGHT, STRAIGHT + np.pi * RADIUS, RADIUS
    x = np.where(s < a, s, np.where(s < b, a + R * np.sin((s - a) / R),
        np.where(s < b + a, a - (s - b), -R * np.sin((s - b - a) / R))))
    y = np.where(s < a, 0.0, np.where(s < b, R * (1 - np.cos((s - a) / R)),
        np.where(s < b + a, 2 * R, R * (1 + np.cos((s - b - a) / R)))))
    th = np.where(s < a, 0.0, np.where(s < b, (s - a) / R,
         np.where(s < b + a, np.pi, np.pi + (s - b - a) / R)))
    return x, y, th


# ====================================================================== the world in a frame
#: obstacle kinds: colour of the side (red, green, blue, 0 - 1), height in m
KINDS = {
    "box":  dict(colour=(0.12, 0.12, 0.14), height=0.20),   # dark and tall: every sensor sees it
    "grey": dict(colour=(0.43, 0.41, 0.38), height=0.20),   # as light as the floor: the camera barely sees it
    "low":  dict(colour=(0.14, 0.12, 0.10), height=0.05),   # dark and low: under the LiDAR's plane
}
FLOOR_RGB = np.array([0.42, 0.40, 0.37])
TAPE_RGB = np.array([0.93, 0.93, 0.90])
WALL_RGB = np.array([0.62, 0.64, 0.68])
NOISE = 0.03              #: pixel noise, of full scale
_LIGHT = np.array([-0.5, 0.8]) / np.hypot(0.5, 0.8)   #: the room's light, from the left and ahead


def render(x, y, th, course="ab", obstacles=(), shadows=(), glare=(), rng=None, light=1.0, slope=0.0):
    """The camera frame at the car's pose: 224 x 224 x 3, red-green-blue,
    uint8 - what ``camera.read()`` gives. ``obstacles`` are ``(x, y, radius,
    kind)``, ``shadows`` and ``glare`` ``(x, y, size)`` on the floor, in
    metres; ``light`` scales the brightness and ``slope`` tilts it across the
    frame."""
    rng = np.random.default_rng() if rng is None else rng
    c, s = np.cos(th), np.sin(th)
    X = x + c * FLOOR_F - s * FLOOR_L                           # each floor pixel, on the floor
    Y = y + s * FLOOR_F + c * FLOOR_L
    d = tape_offset(X, Y, course)
    cover = np.clip((TAPE_HALF - np.abs(d)) / _PIX + 0.5, 0, 1)  # how much of the pixel the tape covers
    shade = np.ones((H, W))
    for (sx, sy, sz) in shadows:
        shade = shade * (1 - 0.45 * np.exp(-((X - sx) ** 2 + (Y - sy) ** 2) / sz ** 2))
    img = (FLOOR_RGB[:, None, None] * (1 - cover) + TAPE_RGB[:, None, None] * cover) * shade
    for (gx, gy, gz) in glare:                                  # a window reflected in the floor
        img = img + 0.55 * np.exp(-((X - gx) ** 2 + (Y - gy) ** 2) / gz ** 2)
    img = np.where(FLOOR, img, WALL_RGB[:, None, None] * (0.9 + 0.1 * (_v / HORIZON)))
    depth = np.where(FLOOR, _T, np.inf)
    cx, cy = x + c * CAM_AHEAD, y + s * CAM_AHEAD               # the camera, on the floor plan
    DX, DY = c * _RF - s * _RL, s * _RF + c * _RL               # each ray, on the floor plan
    for (ox, oy, r, kind) in obstacles:
        k = KINDS[kind]
        px, py = cx - ox, cy - oy
        a = DX * DX + DY * DY
        b = 2 * (px * DX + py * DY)
        disc = b * b - 4 * a * (px * px + py * py - r * r)
        t = (-b - np.sqrt(np.maximum(disc, 0))) / (2 * a)
        z = CAM_HEIGHT + t * _RU
        side = (disc > 0) & (t > 0) & (z >= 0) & (z <= k["height"]) & (t < depth)
        nx, ny = (px + t * DX) / r, (py + t * DY) / r           # the side's normal where the ray meets it
        lit = 0.65 + 0.45 * np.clip(nx * _LIGHT[0] + ny * _LIGHT[1], 0, 1)
        col = np.array(k["colour"])[:, None, None] * lit
        img = np.where(side, col, img)
        depth = np.where(side, t, depth)
        if k["height"] < CAM_HEIGHT:                            # a low block: its top is seen too
            tt = (k["height"] - CAM_HEIGHT) / np.where(_RU < 0, _RU, -1e-9)
            top = (_RU < 0) & ((px + tt * DX) ** 2 + (py + tt * DY) ** 2 <= r * r) & (tt < depth)
            img = np.where(top, np.array(k["colour"])[:, None, None] * 1.25, img)
            depth = np.where(top, tt, depth)
    img = img * light * (1 + slope * (_u / (W - 1) - 0.5))
    img = img + rng.normal(0, NOISE, img.shape)
    return np.round(255 * np.clip(img, 0, 1)).astype(np.uint8).transpose(1, 2, 0)


_S_LAP = np.linspace(0.0, LAP, 2881, endpoint=False)
_XY_LAP = np.column_stack(centreline(_S_LAP, "lap")[:2])


def project(x, y, th, px, py):
    """The image column of the floor point (px, py) seen from the pose, as a
    position from -1 (left edge) to +1 (right edge); behind the camera or out
    of the frame it reads -1 or +1."""
    cx, cy = x + np.cos(th) * CAM_AHEAD, y + np.sin(th) * CAM_AHEAD
    dx, dy = px - cx, py - cy
    f, l = dx * np.cos(th) + dy * np.sin(th), -dx * np.sin(th) + dy * np.cos(th)
    zc = f * np.cos(CAM_PITCH) + CAM_HEIGHT * np.sin(CAM_PITCH)      # depth along the camera's axis
    if zc <= 1e-6:
        return float(-np.sign(l)) if l != 0 else 0.0
    return float(np.clip(-F_PX * l / zc / (W / 2), -1.0, 1.0))


def target(x, y, th, course="ab"):
    """The true steering target at the pose: where the line crosses the row
    :data:`ROW` of the frame - the floor :data:`AIM` ahead of the camera - as
    an image position from -1 (the left edge) to +1 (the right edge). Where
    the row crosses the line twice (in a turn), the crossing nearer the point
    of the line AIM further along the course; where it does not cross it,
    that point's own position, at most -1 or +1. It is the label the
    JetRacer's road-following notebook takes by a click."""
    cx, cy = x + np.cos(th) * CAM_AHEAD, y + np.sin(th) * CAM_AHEAD
    if course == "ab":
        s0 = cx
    else:
        s0 = _S_LAP[int(np.argmin((_XY_LAP[:, 0] - cx) ** 2 + (_XY_LAP[:, 1] - cy) ** 2))]
    px, py, _ = centreline(s0 + AIM, course)
    ahead = project(x, y, th, float(px), float(py))
    c, s = np.cos(th), np.sin(th)
    d = tape_offset(x + c * FLOOR_F[ROW] - s * FLOOR_L[ROW], y + s * FLOOR_F[ROW] + c * FLOOR_L[ROW], course)
    cross = np.where(np.sign(d[:-1]) != np.sign(d[1:]))[0]
    if len(cross) == 0:
        return ahead
    u = cross + d[cross] / (d[cross] - d[cross + 1])          # each crossing, to a fraction of a pixel
    xs = (u - CU) / (W / 2)
    return float(xs[np.argmin(np.abs(xs - ahead))])


# ====================================================================== the virtual JetRacer
class VirtualJetRacer:
    """The JetRacer's ``NvidiaRacecar`` interface on a simulated car.

    Set ``car.steering`` and ``car.throttle`` (each clipped to -1 .. 1), then
    ``car.step(dt)`` moves it: the servo command ``steering * steering_gain +
    steering_offset`` turns the front wheels by that fraction of
    :data:`SERVO_ANGLE` (a positive steering turns right, as on the car), the
    duty ``throttle * throttle_gain`` sets the speed it approaches with time
    constant :data:`SPEED_LAG`, and the kinematic bicycle model carries the
    pose. ``car.camera.read()`` is the frame at the current pose."""

    def __init__(self, x=0.0, y=0.0, th=0.0, course="ab", obstacles=(), shadows=(), glare=(), seed=0):
        self.x, self.y, self.th, self.v = float(x), float(y), float(th), 0.0
        self.course, self.obstacles, self.shadows, self.glare = course, list(obstacles), list(shadows), list(glare)
        self.steering_gain, self.steering_offset, self.throttle_gain = -0.65, 0.0, 0.8
        self._steering = self._throttle = 0.0
        self.rng = np.random.default_rng(seed)
        self.camera = _Camera(self)

    steering = property(lambda self: self._steering,
                        lambda self, v: setattr(self, "_steering", float(np.clip(v, -1, 1))))
    throttle = property(lambda self: self._throttle,
                        lambda self, v: setattr(self, "_throttle", float(np.clip(v, -1, 1))))

    def wheel_angle(self):
        servo = np.clip(self._steering * self.steering_gain + self.steering_offset, -1, 1)
        return float(np.clip(servo * SERVO_ANGLE, -STEER_MAX, STEER_MAX))

    def step(self, dt):
        v_set = V_PER_DUTY * self._throttle * self.throttle_gain
        dv = np.clip((v_set - self.v) / SPEED_LAG * dt, -MU * G * dt, MU * G * dt)   # no faster than the tyres allow
        self.v += dv
        self.x += self.v * np.cos(self.th) * dt
        self.y += self.v * np.sin(self.th) * dt
        self.th += self.v / WHEELBASE * np.tan(self.wheel_angle()) * dt

    def gap(self):
        """The smallest gap between the car's edge and any obstacle, m."""
        if not self.obstacles:
            return np.inf
        o = np.array([ob[:3] for ob in self.obstacles])
        return float((np.hypot(o[:, 0] - self.x, o[:, 1] - self.y) - o[:, 2] - CAR_HALF).min())


class _Camera:
    def __init__(self, car):
        self.car = car
        self.width = self.height = W

    def read(self):
        c = self.car
        return render(c.x, c.y, c.th, c.course, c.obstacles, c.shadows, c.glare, c.rng)


def throttle_for(speed, gain=0.8):
    """The throttle that settles at ``speed`` m/s."""
    return speed / (V_PER_DUTY * gain)


def to_net(frames):
    """Frames from the camera (N, 224, 224, 3) uint8 -> (N, 3, 224, 224) uint8, the network's order."""
    return np.ascontiguousarray(np.asarray(frames).transpose(0, 3, 1, 2))


# ====================================================================== the data the notebook trains on
#: how a steering dataset is collected, on the lap: the spread of the car's
#: offset from the line (m) and its heading (rad), and the share of frames with glare
COLLECTION = {
    "line":       dict(offset=0.02, heading=0.05, glare=0.0),   # driven along the line only
    "off-line":   dict(offset=0.12, heading=0.40, glare=0.0),   # also from where the car should not be
    "glare":      dict(offset=0.12, heading=0.40, glare=1.0),   # the same, every frame with glare
    "deliberate": dict(offset=0.12, heading=0.40, glare=0.3),   # collected on purpose: everything the car will meet
}


def steering_frames(n, kind="deliberate", seed=0):
    """``n`` frames on the lap, collected the way ``kind`` says, and the true
    target of each. Returns ``(frames, targets)``: frames ``(n, 3, 224, 224)``
    uint8, targets ``(n,)`` from -1 to 1."""
    c = COLLECTION[kind]
    rng = np.random.default_rng(seed)
    out = np.empty((n, 3, H, W), np.uint8)
    tgt = np.empty(n, np.float32)
    for k in range(n):
        x0, y0, th0 = centreline(rng.uniform(0, LAP), "lap")
        off = rng.uniform(-c["offset"], c["offset"])
        x, y = x0 - np.sin(th0) * off, y0 + np.cos(th0) * off
        th = th0 + rng.uniform(-c["heading"], c["heading"])
        glare = []
        if rng.random() < c["glare"]:                          # a window reflected beside the line ahead
            a = rng.uniform(0.25, 0.7); b = rng.choice([-1, 1]) * rng.uniform(0.08, 0.22)
            glare = [(x + np.cos(th) * a - np.sin(th) * b, y + np.sin(th) * a + np.cos(th) * b, rng.uniform(0.04, 0.08))]
        img = render(x, y, th, "lap", glare=glare, rng=rng, light=rng.uniform(0.85, 1.15), slope=rng.uniform(-0.3, 0.3))
        out[k] = img.transpose(2, 0, 1)
        tgt[k] = target(x, y, th, "lap")
    return out, tgt


RANGE = 0.70              #: m: an obstacle closer than this, in the lane, blocks the path (ASSUMED)
LANE = 0.15               #: m: half the lane, from the car's centre line to the obstacle's edge, beyond the car's half-width


def is_blocked(x, y, th, obstacles):
    """1 when an obstacle stands within :data:`RANGE` ahead of the car and
    within the lane, and the nearest such obstacle's forward and sideways
    distance and kind."""
    hits = []
    for (ox, oy, r, kind) in obstacles:
        dx, dy = ox - x, oy - y
        f, s = dx * np.cos(th) + dy * np.sin(th), -dx * np.sin(th) + dy * np.cos(th)
        if 0 < f < RANGE and abs(s) < LANE + r + CAR_HALF:
            hits.append((f, s, kind))
    if not hits:
        return 0, (np.nan, np.nan), ""
    f, s, kind = min(hits)
    return 1, (f, s), kind


def obstacle_frames(n, seed=0, p_obstacle=0.8, p_shadow=0.4, kinds=("box", "box", "grey", "low")):
    """``n`` frames on the A-to-B course: the car near the line, and on most
    frames an obstacle somewhere ahead - dark, grey or low - and on some a
    shadow, which is NOT an obstacle. Returns ``(frames, blocked, where,
    kind)``: frames ``(n, 3, 224, 224)`` uint8, ``blocked`` 1 when an obstacle
    stands in the lane within :data:`RANGE`, ``where`` its forward and sideways
    distance (m, nan when free), ``kind`` its kind."""
    rng = np.random.default_rng(seed)
    out = np.empty((n, 3, H, W), np.uint8)
    blocked = np.zeros(n, np.int64)
    where = np.full((n, 2), np.nan, np.float32)
    kind = np.empty(n, dtype=object); kind[:] = ""
    for k in range(n):
        x, y, th = rng.uniform(0, 4), rng.uniform(-0.12, 0.12), rng.uniform(-0.3, 0.3)
        obs, sh = [], []
        if rng.random() < p_obstacle:
            f, s = rng.uniform(0.25, 1.4), rng.uniform(-0.45, 0.45)
            obs = [(x + np.cos(th) * f - np.sin(th) * s, y + np.sin(th) * f + np.cos(th) * s,
                    rng.uniform(0.05, 0.09), str(rng.choice(kinds)))]
        if rng.random() < p_shadow:
            f, s = rng.uniform(0.25, 1.0), rng.uniform(-0.3, 0.3)
            sh = [(x + np.cos(th) * f - np.sin(th) * s, y + np.sin(th) * f + np.cos(th) * s, rng.uniform(0.06, 0.12))]
        out[k] = render(x, y, th, "ab", obs, sh, rng=rng, light=rng.uniform(0.85, 1.15),
                        slope=rng.uniform(-0.3, 0.3)).transpose(2, 0, 1)
        b, w, kd = is_blocked(x, y, th, obs)
        blocked[k], where[k], kind[k] = b, w, kd
    return out, blocked, where, kind


# ====================================================================== the classical rules
def shrink(frames, factor):
    """The frames at a lower resolution: every ``factor`` x ``factor`` block of
    pixels replaced by its mean. Frames ``(n, 3, H, W)``; returns grey ``(n, 1, H/f, W/f)``, 0 - 1."""
    g = np.asarray(frames, np.float32).mean(axis=1, keepdims=True) / 255
    if factor == 1:
        return g
    n, c, h, w = g.shape
    return g.reshape(n, c, h // factor, factor, w // factor, factor).mean(axis=(3, 5))


def detect(grey, threshold=0.15):
    """The classical line detector, on grey frames from :func:`shrink`. In a
    band of rows round :data:`ROW`, keep what is brighter than the floor by
    ``threshold``, take the largest bright run of columns (the tape, not a
    stray far piece of it), and return its brightness-weighted mean column,
    from -1 to 1. Nothing that bright: 0, straight ahead."""
    n, _, h, w = grey.shape
    row = int(round(ROW * h / H))
    r = max(1, round(3 * h / H))
    band = grey[:, 0, max(row - r, 0): row + r + 1, :]
    floor = np.median(band, axis=(1, 2), keepdims=True)
    prof = np.clip(band - floor - threshold, 0.0, None).sum(axis=1)    # brightness above the floor, per column
    out = np.zeros(n)
    cols = np.arange(w) + 0.5
    for k in range(n):
        on = np.concatenate([[0], (prof[k] > 0).astype(int), [0]])
        starts, ends = np.where(np.diff(on) == 1)[0], np.where(np.diff(on) == -1)[0]
        if len(starts) == 0:
            continue
        best = max(zip(starts, ends), key=lambda se: prof[k, se[0]:se[1]].sum())
        p = prof[k, best[0]:best[1]]
        col = (p * cols[best[0]:best[1]]).sum() / p.sum()
        out[k] = (col * W / w - 0.5 - CU) / (W / 2)
    return out


#: where the dark-pixel rule looks: the floor of the car's lane up to 0.3 m beyond RANGE,
#: so that the near face of an obstacle standing at RANGE is in it
LANE_MASK = FLOOR & (FLOOR_F > CAM_AHEAD) & (FLOOR_F < RANGE + 0.3) & (np.abs(FLOOR_L) < LANE + CAR_HALF + 0.05)


def dark_rule(frames, threshold=0.15, share=0.002):
    """The classical obstacle rule: in :data:`LANE_MASK`, is more than
    ``share`` of the pixels darker than the frame's floor by ``threshold``?
    Returns 1 for blocked. A shadow is dark too, which is where it fails; an
    obstacle as light as the floor is not, which is where it fails the other
    way. The two numbers were tuned once, on frames like the notebook's."""
    g = np.asarray(frames, np.float32).mean(axis=1) / 255
    floor = np.median(g[:, FLOOR], axis=1)
    dark = (g[:, LANE_MASK] < floor[:, None] - threshold).mean(axis=1)
    return (dark > share).astype(np.int64)


# ====================================================================== driving the twin
STEERING_GAIN = 2.5       #: car.steering = target * STEERING_GAIN, as the road-following notebook writes it (tuned on the twin)


def drive_line(policy, rate_hz, speed, course="lap", laps=2.0, seed=0, dt=2e-3, start=None):
    """The virtual car follows the line with ``policy(frame) -> target``,
    deciding ``rate_hz`` times a second at ``speed`` m/s. Computing a decision
    takes one period, so the wheels follow the command from the frame before
    and hold it until the next. Returns the largest distance from the line
    after the first quarter lap, m, and the path."""
    x0, y0, th0 = centreline(0.0, course) if start is None else start
    car = VirtualJetRacer(float(x0), float(y0), float(th0), course=course, seed=seed)
    car.v = speed
    car.throttle = throttle_for(speed)
    length = laps * (LAP if course == "lap" else COURSE)
    pending, next_dec, worst, t, travelled = 0.0, 0.0, 0.0, 0.0, 0.0
    path = [(car.x, car.y)]
    while travelled < length:
        if t >= next_dec - 1e-12:
            car.steering = pending                               # the command computed from the frame before
            pending = float(policy(car.camera.read())) * STEERING_GAIN
            next_dec += 1.0 / rate_hz
        car.step(dt)
        t += dt
        travelled += car.v * dt
        if travelled > 0.25 * LAP:
            worst = max(worst, abs(float(tape_offset(car.x, car.y, course))))
        if len(path) == 0 or np.hypot(car.x - path[-1][0], car.y - path[-1][1]) > 0.02:
            path.append((car.x, car.y))
        if abs(float(tape_offset(car.x, car.y, course))) > 0.5:  # left the floor
            return 0.5, path
    return worst, path


HALF_CLEARANCE = 0.175    #: m, half the narrowest gap of an assumed course, 0.35 m

# ---------------------------------------------------- the A-to-B course with obstacles
OBSTACLES = ((1.5, 0.00, 0.08, "box"), (3.0, 0.06, 0.08, "box"), (4.5, -0.05, 0.08, "box"))   #: (x, y, radius, kind), m (ASSUMED)
K_FIELD = 0.03            #: the potential field's strength, m


def field_aim(x, near, k=None):
    """The potential field's choice. One aim distance ahead of the car at
    ``x``, across the lane, ``U(y) = y**2 + sum K/d_i(y)``: the pull back to
    the line plus a push from every obstacle in ``near`` (rows of ``(x, y,
    radius)``), ``d_i`` the gap between the car's edge and obstacle ``i``.
    Returns the sideways position where ``U`` is lowest."""
    k = K_FIELD if k is None else k
    ax = x + AIM
    ys = np.linspace(-0.6, 0.6, 241)
    U = ys ** 2
    for (ox, oy, r) in near:
        d = np.maximum(np.hypot(ax - ox, ys - oy) - r - CAR_HALF, 0.01)
        U = U + k / d
    return float(ys[np.argmin(U)])


def _steer_to(car, tx, ty):
    """The car.steering that points the wheels at (tx, ty) by pure pursuit."""
    alpha = np.arctan2(ty - car.y, tx - car.x) - car.th
    delta = np.arctan(2 * WHEELBASE * np.sin(alpha) / max(np.hypot(tx - car.x, ty - car.y), 1e-3))
    return float(delta / SERVO_ANGLE / car.steering_gain)       # wheel angle -> servo -> steering


def avoid(strategy, rate_hz=10.0, speed=1.0, p_miss=0.0, p_false=0.0, seed=0, dt=2e-3, t_max=30.0):
    """Drive the virtual car from A to B with one avoidance strategy and a
    decision ``rate_hz`` times a second, through ``car.steering`` and
    ``car.throttle``. Every decision sees "blocked" when :func:`is_blocked`
    says so, wrongly with the classifier's miss and false-alarm rates.
    Strategies:

    ``"stop"``   blocked: throttle 0; free: follow the line;
    ``"steer"``  blocked: aim at a lane 0.30 m to the side away from the
                 obstacle, slowed to 60 %, held 0.5 m past it, then back;
    ``"field"``  one aim distance ahead, aim at the lowest point across the
                 lane of ``U(y) = y**2 + sum K/d_i(y)`` (:func:`field_aim`);
    a function   ``f(x, near)`` returning the sideways position to aim at,
                 written like :func:`field_aim`.

    Returns ``dict(time, clearance, path, reached, poses)``: the time to B
    (``inf`` when B is not reached), the smallest gap between car and obstacle
    in metres (negative is a collision), the path, and every 1/25 s the time,
    pose, wheel angle, speed and whether the path was seen blocked, for
    :func:`film`."""
    rng = np.random.default_rng(seed)
    car = VirtualJetRacer(0.0, 0.0, 0.0, "ab", OBSTACLES, seed=seed)
    car.v = speed
    obs = np.array([o[:3] for o in OBSTACLES])
    y_ref, hold, next_dec, t, clear, path = 0.0, -1.0, 0.0, 0.0, np.inf, [(0.0, 0.0)]
    poses, next_rec, seen = [], 0.0, False
    while t < t_max and car.x < COURSE:
        if t >= next_rec - 1e-12:                                # for the film: every 1/25 s
            poses.append((t, car.x, car.y, car.th, car.wheel_angle(), car.v, int(seen)))
            next_rec += 0.04
        if t >= next_dec - 1e-12:
            next_dec += 1.0 / rate_hz
            seen, (f, s), _ = is_blocked(car.x, car.y, car.th, OBSTACLES)
            seen = bool(seen)
            if seen and rng.random() < p_miss:
                seen = False
            elif not seen and rng.random() < p_false:
                seen = True
            v = speed
            if strategy == "stop":
                v = 0.0 if seen else speed
                ty = 0.0
            elif strategy == "steer":
                if seen and np.isfinite(f):
                    oy = car.y + np.sin(car.th) * f + np.cos(car.th) * s
                    y_ref = -0.30 if oy >= car.y else 0.30
                    hold = car.x + np.cos(car.th) * f + 0.5
                v = 0.6 * speed if car.x < hold else speed
                if car.x >= hold:
                    y_ref = 0.0
                ty = y_ref
            else:
                near = np.array([o for o in obs if abs(o[0] - car.x) < RANGE and o[0] > car.x - o[2]]).reshape(-1, 3)
                aim = field_aim if strategy == "field" else strategy
                ty = float(aim(car.x, near))
            car.steering = _steer_to(car, car.x + AIM, ty)
            car.throttle = throttle_for(v)
        car.step(dt)
        t += dt
        clear = min(clear, car.gap())
        if abs(car.x - path[-1][0]) > 0.02 or abs(car.y - path[-1][1]) > 0.01:
            path.append((round(car.x, 3), round(car.y, 3)))
    poses.append((t, car.x, car.y, car.th, car.wheel_angle(), car.v, int(seen)))
    return dict(time=t if car.x >= COURSE else np.inf, clearance=clear, path=path, reached=car.x >= COURSE,
                poses=np.array(poses))


def stopping_distance(speed, rate_hz):
    """From the frame that shows the obstacle to standing still: one blind
    distance, then braking at the tyres' limit."""
    return speed / rate_hz + speed ** 2 / (2 * MU * G)


# ====================================================================== fewer bits
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


# ====================================================================== pictures and a description
def show_glare_failure(n=300, seed=102):
    """The failure of the classical rule, seen whole on one glare frame: the
    camera's frame with the detector's band of rows, the true column of the
    tape (green) and the aim the weighted mean actually picks (the orange
    cross); under it, column for column, the band row's brightness with the
    floor's threshold. The frame is the 80th percentile of the error over
    ``n`` held-out glare frames - the same collection section 3 measures
    (L11.1, *Where Conventional Vision Fails*). Returns the frame's numbers."""
    import matplotlib.pyplot as plt
    x, y = steering_frames(n, "glare", seed=seed)
    cases = []
    for k in range(n):
        r = shrink(x[k:k + 1], 1)[0, 0, ROW]                  # the detector's row of the full frame
        floor = float(np.median(r))                           # the floor's brightness: the median
        w = np.clip(r - floor - 0.15, 0.0, None)              # the weight: brightness above the floor
        on = np.concatenate([[0], (w > 0).astype(int), [0]])
        runs = list(zip(np.where(np.diff(on) == 1)[0], np.where(np.diff(on) == -1)[0]))
        if len(runs) < 2:                                     # keep frames where tape AND glare are bright
            continue
        cols = np.arange(W)
        x_true = CU + W / 2 * float(y[k])                     # where the tape really crosses the row
        x_est = float((w * cols).sum() / w.sum())             # the weighted mean over every bright pixel
        cases.append((abs(x_est - x_true), k, r, floor, x_true, x_est, w))
    cases.sort(key=lambda c: c[0])
    e, k, r, floor, x_true, x_est, w = cases[int(0.8 * len(cases))]
    share = float(w[np.abs(np.arange(W) - x_true) > 6].sum() / w.sum())
    det = CU + W / 2 * float(detect(shrink(x[k:k + 1], 2))[0])    # the notebook's detector (largest run) on the same frame

    fig, (a0, a1) = plt.subplots(2, 1, figsize=(9.0, 6.4), sharex=True,
                                 gridspec_kw=dict(height_ratios=[2.1, 1.0]))
    a0.imshow(np.moveaxis(x[k], 0, -1), extent=(0, W, H, 0))   # the frame is channels-first
    a0.axhspan(ROW - 3, ROW + 3, color="white", alpha=0.18, lw=0)
    a0.axvline(x_true, color="tab:green", lw=2)
    a0.axvline(x_est, color="tab:orange", ls="--", lw=2)
    a0.plot([x_est], [ROW], marker="+", ms=22, mew=3, color="tab:orange")
    a0.plot([x_est], [ROW], marker="o", ms=11, mfc="none", mec="tab:orange", mew=2)
    a0.text(x_true + 4, 14, "the tape", color="tab:green", fontsize=11)
    a0.text(x_est + 4, ROW - 8, "the aim the rule picks", color="tab:orange", fontsize=11)
    a0.set_ylabel("row of the frame")
    a0.set_title("one glare frame, and the band of rows the detector reads")
    a1.plot(np.arange(W), r, color="0.25", lw=1.2)
    a1.axhline(floor + 0.15, color="0.55", ls="--", lw=1)
    a1.axvline(x_true, color="tab:green", lw=2)
    a1.axvline(x_est, color="tab:orange", ls="--", lw=2)
    a1.text(2, floor + 0.17, "threshold: the floor + 0.15", color="0.45", fontsize=10)
    a1.set_xlabel("column of the frame, 0 to 223")
    a1.set_ylabel("brightness, 0 to 1")
    fig.tight_layout()
    print(f"the tape crosses the row at column {x_true:.0f}; the weighted mean of every bright pixel"
          f" is column {x_est:.0f}, pulled by the glare's {100 * share:.0f} % of the weight;"
          f" the largest-run detector picks column {det:.0f}")
    return dict(x_true=x_true, x_est=x_est, share=share, detector=det)


def draw_course(ax=None, paths=None, dark=False, poses=(), obstacles=OBSTACLES):
    """The A-to-B course from above: the line, the obstacles with the car's
    half-width round each, and, if given, ``paths`` as ``{label: [(x, y),
    ...]}`` and car ``poses`` ``(x, y, th)`` with the floor their camera sees
    within :data:`RANGE`. Returns the axes."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 2.6))
    fg = "white" if dark else "black"
    ax.axhline(0, color="0.6", lw=3, zorder=0)                         # the taped line
    for (ox, oy, r, kind) in obstacles:
        ax.add_patch(plt.Circle((ox, oy), r + CAR_HALF, color="tab:red", alpha=0.15, lw=0))
        ax.add_patch(plt.Circle((ox, oy), r, color="tab:red" if kind == "box" else "0.55" if kind == "grey" else "saddlebrown"))
    for x, name in ((0.0, "A"), (COURSE, "B")):
        ax.plot([x, x], [-0.5, 0.5], color=fg, lw=1.5)
        ax.text(x, 0.52, name, ha="center", va="bottom", color=fg, fontsize=11, weight="bold")
    for i, (x, y, th) in enumerate(poses):
        c, s = np.cos(th), np.sin(th)
        Fq, Lq = FLOOR_F[FLOOR & (FLOOR_F < 1.5)], FLOOR_L[FLOOR & (FLOOR_F < 1.5)]
        pick = np.random.default_rng(0).choice(len(Fq), 600, replace=False)
        ax.plot(x + c * Fq[pick] - s * Lq[pick], y + s * Fq[pick] + c * Lq[pick], ".", ms=1, color="tab:blue", alpha=0.3)
        ax.annotate("", (x + 0.25 * c, y + 0.25 * s), (x, y), arrowprops=dict(arrowstyle="->", color=fg, lw=1.5))
        ax.text(x, y - 0.12, str(i + 1), ha="center", va="top", color=fg, fontsize=10, weight="bold")
    for label, p in (paths or {}).items():
        p = np.asarray(p)
        ax.plot(p[:, 0], p[:, 1], lw=1.6, label=label)
    if paths:
        ax.legend(loc="lower right", fontsize=8, ncol=len(paths))
    ax.set_xlim(-0.2, COURSE + 0.2); ax.set_ylim(-0.6, 0.7); ax.set_aspect("equal")
    ax.set_xlabel("along the course [m]", color=fg); ax.set_ylabel("sideways [m]", color=fg)
    ax.tick_params(colors=fg)
    return ax


def describe_problem() -> None:
    """Print the car, the camera, the course and the numbers they imply."""
    print("  the frames are RENDERED from the car's pose and the car is SIMULATED - no real car is needed")
    print("  vendor          : Jetson Nano 4 GB, IMX219-160 camera, Ackermann steering, RC380 motor, 2S2P 8.4 V, no encoder")
    print(f"  frame           : {H} x {W} x 3 (red, green, blue), as CSICamera(width=224, height=224) on the car")
    print(f"  camera, ASSUMED : {100 * CAM_HEIGHT:.0f} cm high, tilted {np.degrees(CAM_PITCH):.0f} degrees down, "
          f"{np.degrees(HFOV):.0f} degree pinhole, {100 * CAM_AHEAD:.0f} cm ahead of the rear axle")
    print(f"  the line is read in row {ROW}, the floor {100 * AIM:.0f} cm ahead of the camera; horizon at row {HORIZON:.0f}")
    print(f"  car, ASSUMED    : wheelbase {WHEELBASE:.2f} m, wheel angle {np.degrees(SERVO_ANGLE):.0f} degrees at full servo, "
          f"limit {np.degrees(STEER_MAX):.0f}; {V_PER_DUTY:.1f} m/s at full duty, reached in {SPEED_LAG:.1f} s; friction {MU:.1f}")
    print(f"  interface       : car.steering, car.throttle in -1..1; steering_gain -0.65, throttle_gain 0.8 (JetRacer library)")
    print(f"  course A to B   : {COURSE:.1f} m, {len(OBSTACLES)} obstacles of radius {100 * OBSTACLES[0][2]:.0f} cm; "
          f"blocked means an obstacle in the lane within {RANGE:.1f} m")
    print(f"  lap (section 7) : a stadium, straights {STRAIGHT:.1f} m, turns of radius {RADIUS:.1f} m, lap {LAP:.2f} m")
    for v, f in ((1.0, 30.0), (2.0, 20.0), (2.0, 5.0)):
        print(f"  blind distance  : {100 * v / f:.0f} cm at {v:.0f} m/s and {f:.0f} decisions a second")


# ====================================================================== developing the mini projects
# MP11A (the camera with a LiDAR) and MP11B (the camera with an ultrasonic
# sensor) are done on the REAL JetRacer with the real sensor. These functions
# simulate both sensors from the twin's geometry, so that the fusion can be
# written and tested before the car is booked. tools/miniprojects/ex11_truth.py
# uses them for the worked example. Values marked VENDOR are the
# manufacturers' figures, confirmed by the lecturer on 13 October 2026; ASSUMED
# are this course's choices, to be measured on the car.
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
    each ``(x, y, radius, kind)``, and two shadows on the floor ``(x, y,
    size)``, which are not obstacles."""
    rng = np.random.default_rng(seed)
    xs = np.sort(rng.uniform(1.0, COURSE - 0.8, 40))
    pick = [xs[0]]
    for x in xs[1:]:
        if x - pick[-1] >= 0.9 and len(pick) < n:
            pick.append(x)
    obstacles = [(float(x), float(rng.uniform(-0.15, 0.15)), float(rng.uniform(0.06, 0.10)),
                  str(rng.choice(list(KINDS)))) for x in pick]
    shadows = [(float(rng.uniform(0.8, COURSE - 0.5)), float(rng.uniform(-0.3, 0.3)), 0.10) for _ in range(2)]
    return dict(obstacles=obstacles, shadows=shadows, seed=seed)


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
    """One simulated LiDAR scan at the car's pose: ``LIDAR_BEAMS`` ranges in
    metres, beam ``i`` at ``i`` degrees anticlockwise from straight ahead.
    Only obstacles taller than the scan plane return; no return reads 0, as
    the RPLIDAR does."""
    ang = th + np.radians(np.arange(LIDAR_BEAMS) * 360.0 / LIDAR_BEAMS)
    d = np.full(LIDAR_BEAMS, np.inf)
    for (ox, oy, r, kind) in course["obstacles"]:
        if KINDS[kind]["height"] > LIDAR_HEIGHT:
            d = np.minimum(d, _ray_circle(x, y, ang, ox, oy, r))
    d = d * (1 + LIDAR_NOISE * rng.normal(size=d.shape))
    return np.where((d >= LIDAR_MIN) & (d <= LIDAR_MAX), d, 0.0).astype(np.float32)


def sonar(x, y, th, course, rng):
    """One simulated ultrasonic reading at the car's pose: the distance in
    metres to the nearest surface inside the beam's cone, rounded to the
    sensor's step; ``SONAR_MAX`` when nothing answers. It says how far, not
    where."""
    best = np.inf
    for (ox, oy, r, kind) in course["obstacles"]:
        dx, dy = ox - x, oy - y
        f, s = dx * np.cos(th) + dy * np.sin(th), -dx * np.sin(th) + dy * np.cos(th)
        dist = np.hypot(f, s)
        if f <= 0 or dist <= r:
            continue
        if abs(np.arctan2(s, f)) - np.arcsin(r / dist) < SONAR_CONE:
            best = min(best, dist - r)
    if not np.isfinite(best):
        return SONAR_MAX
    best = best * (1 + SONAR_NOISE * rng.normal())
    return float(np.clip(np.round(best / SONAR_STEP) * SONAR_STEP, SONAR_MIN, SONAR_MAX))


class SteerAround:
    """The steer-around rule as a decision function for :func:`run_course`:
    ``decide(reading) -> (speed_factor, y_ref)``. The reading carries
    ``blocked`` and where the obstacle stands, ``obstacle_x`` and
    ``obstacle_y`` on the course; the car aims at a lane 0.30 m beside the
    obstacle, on the side it is already on, slowed to 60 %, until 0.5 m past
    it."""

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
        b, (f, s), _ = is_blocked(x, y, th, course["obstacles"])
        if not b:
            return dict(blocked=0, obstacle_x=np.nan, obstacle_y=np.nan)
        return dict(blocked=1, obstacle_x=x + f * np.cos(th) - s * np.sin(th),
                    obstacle_y=y + f * np.sin(th) + s * np.cos(th))
    return perceive


def run_course(course, perceive, rate_hz=10.0, speed=1.0, dt=2e-3, t_max=30.0, seed=0):
    """Drive the virtual car from A to B on ``course``. At every decision
    ``perceive(x, y, th, sensors)`` returns a reading for the steer-around
    rule; ``sensors`` holds the latest camera frame, LiDAR scan and
    ultrasonic range, each refreshed at its own rate. Returns ``dict(time,
    clearance, reached, path)``: ``clearance`` the smallest gap in metres,
    negative a collision."""
    rng = np.random.default_rng(seed)
    decide = SteerAround()
    car = VirtualJetRacer(0.0, 0.0, 0.0, "ab", course["obstacles"], course["shadows"], seed=seed)
    car.v = speed
    t = next_dec = next_lidar = next_sonar = 0.0
    sensors, clear, path = {}, np.inf, [(0.0, 0.0)]
    while t < t_max and car.x < COURSE:
        if t >= next_lidar - 1e-12:
            sensors["scan"], next_lidar = lidar(car.x, car.y, car.th, course, rng), next_lidar + 1 / LIDAR_HZ
        if t >= next_sonar - 1e-12:
            sensors["sonar"], next_sonar = sonar(car.x, car.y, car.th, course, rng), next_sonar + 1 / SONAR_HZ
        if t >= next_dec - 1e-12:
            sensors["frame"] = car.camera.read()
            reading = dict(perceive(car.x, car.y, car.th, sensors), x=car.x, y=car.y)
            fac, y_ref = decide(reading)
            car.steering = _steer_to(car, car.x + AIM, y_ref)
            car.throttle = throttle_for(speed * fac)
            next_dec += 1 / rate_hz
        car.step(dt)
        t += dt
        clear = min(clear, car.gap())
        if abs(car.x - path[-1][0]) > 0.02 or abs(car.y - path[-1][1]) > 0.01:
            path.append((round(car.x, 3), round(car.y, 3)))
    return dict(time=t if car.x >= COURSE else np.inf, clearance=clear, reached=car.x >= COURSE, path=path)


# ====================================================================== the film of a run
# The JetRacer Pro in 3-D, after the vendor's photograph (lectures/L11.1-...
# /assets/jetracer-pro-ai-kit-1.jpg): black chassis plate and foam bumper,
# four black tyres on white ten-spoke rims, blue-capped shock towers, the
# battery holder, the expansion board on white standoffs with its blue USB
# ports, the Jetson's finned heat sink and fan, the green camera bracket with
# the camera board and lens, the ribbon cable and two antennas. The shapes
# and sizes are a DRAWING to the twin's own numbers (wheelbase, half-width),
# not a measurement of the car.
WHEEL_R, WHEEL_W, TRACK = 0.033, 0.026, 0.085      #: m: wheel radius and width, centre-to-centre half-track (drawn)


def _rot(axis, a):
    c, s = np.cos(a), np.sin(a)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _box(x0, x1, y0, y1, z0, z1):
    """The six faces of a box, wound outwards."""
    c = np.array([(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
                  (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)], float)
    return [c[list(f)] for f in ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (2, 3, 7, 6), (1, 2, 6, 5), (0, 4, 7, 3))]


def _prism(outline, z0, z1):
    """A flat shape, ``outline`` anticlockwise from above, from height z0 to z1."""
    o = np.asarray(outline, float)
    lo = np.column_stack([o, np.full(len(o), z0)])
    hi = np.column_stack([o, np.full(len(o), z1)])
    faces = [hi, lo[::-1]]
    for i in range(len(o)):
        k = (i + 1) % len(o)
        faces.append(np.array([lo[i], lo[k], hi[k], hi[i]]))
    return faces


def _tube(p0, p1, r, n=12, caps=True):
    """A round rod from p0 to p1."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = (p1 - p0) / np.linalg.norm(p1 - p0)
    u = np.cross(d, [0, 0, 1.0] if abs(d[2]) < 0.9 else [1.0, 0, 0]); u /= np.linalg.norm(u)
    v = np.cross(d, u)
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.cos(a)[:, None] * u + np.sin(a)[:, None] * v
    b, t = p0 + r * ring, p1 + r * ring
    faces = [np.array([b[i], b[(i + 1) % n], t[(i + 1) % n], t[i]]) for i in range(n)]
    if caps:
        faces += [t, b[::-1]]
    return faces


def _spring(p0, p1, r, turns=6, n=10):
    """A coil spring as a thin band round the axis p0 -> p1."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    dh = d / np.linalg.norm(d)
    u = np.cross(dh, [1.0, 0, 0]); u /= np.linalg.norm(u)
    v = np.cross(dh, u)
    k = np.linspace(0, 1, turns * n + 1)
    pts = p0 + k[:, None] * d + r * (np.cos(2 * np.pi * turns * k)[:, None] * u + np.sin(2 * np.pi * turns * k)[:, None] * v)
    w = 0.0025 * dh
    return [np.array([pts[i], pts[i + 1], pts[i + 1] + w, pts[i] + w]) for i in range(len(pts) - 1)]


def _wheel_local():
    """One wheel at the origin, its axle along y, its outer face at +y: (faces, colour) pairs."""
    R, W, n = WHEEL_R, WHEEL_W, 28
    out = [(_tube((0, -W / 2, 0), (0, W / 2, 0), R, n, caps=False), "#1A1A1A")]          # the tread
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = lambda r, y: np.column_stack([r * np.sin(a), np.full(n, y), r * np.cos(a)])
    for y, sgn in ((W / 2, 1), (-W / 2, -1)):                                            # the side walls
        o, i_ = ring(R, y), ring(0.024, y)
        q = [np.array([o[k], o[(k + 1) % n], i_[(k + 1) % n], i_[k]])[:: sgn] for k in range(n)]
        out.append((q, "#222222"))
    y = W / 2 + 0.0008                                                                   # the white rim
    o, i_ = ring(0.024, y), ring(0.019, y)
    out.append(([np.array([o[k], o[(k + 1) % n], i_[(k + 1) % n], i_[k]]) for k in range(n)], "#F2F2F2"))
    spokes = []
    for k in range(10):
        b = 2 * np.pi * k / 10
        for (r0, r1, w0, w1) in ((0.006, 0.0195, 0.0028, 0.0016),):
            c0, c1 = np.array([np.sin(b), 0, np.cos(b)]), np.array([np.cos(b), 0, -np.sin(b)])
            spokes.append(np.array([r0 * c0 - w0 * c1, r1 * c0 - w1 * c1, r1 * c0 + w1 * c1, r0 * c0 + w0 * c1]) + [0, y, 0])
    out.append((spokes, "#EDEDED"))
    out.append(([ring(0.0065, y + 0.0004)], "#2B2B2B"))                                  # the hub nut
    out.append(([ring(0.019, y - 0.001)], "#3A3A3A"))                                    # behind the spokes
    return out


def _body_local():
    """Everything but the wheels, in the car's frame (x forward from the rear axle, y left, z up)."""
    P = []
    plate = [(-0.07, -0.030), (-0.03, -0.048), (0.24, -0.048), (0.275, -0.030),
             (0.275, 0.030), (0.24, 0.048), (-0.03, 0.048), (-0.07, 0.030)]
    P.append((_prism(plate, 0.026, 0.031), "#161616"))                                   # the chassis plate
    bump = [(0.262 + 0.048 * np.cos(t), 0.088 * np.sin(t)) for t in np.linspace(-1.35, 1.35, 17)]
    P.append((_prism(bump, 0.014, 0.040), "#1C1C1C"))                                    # the foam bumper, rounded at the front
    for ax_ in (0.0, WHEELBASE):                                                         # the suspension
        for sgn in (-1, 1):
            P.append((_box(ax_ - 0.010, ax_ + 0.010, sgn * 0.075 - 0.008, sgn * 0.075 + 0.008, 0.024, 0.044)
                      if sgn > 0 else _box(ax_ - 0.010, ax_ + 0.010, -0.083, -0.067, 0.024, 0.044), "#1E5BB8"))   # blue hub carrier
            P.append((_box(ax_ - 0.004, ax_ + 0.004, min(sgn * 0.035, sgn * 0.070), max(sgn * 0.035, sgn * 0.070), 0.028, 0.034), "#262626"))
            top = (ax_ - 0.012 * (1 if ax_ else -1), sgn * 0.052, 0.088)
            bot = (ax_, sgn * 0.066, 0.040)
            P.append((_tube(bot, top, 0.0045, 10), "#202020"))                           # the damper
            P.append((_spring(bot, top, 0.008, turns=7), "#303030"))                     # its spring
            P.append((_tube(top, np.add(top, (0, -sgn * 0.0, 0.008)), 0.0065, 12), "#1565C0"))   # the blue cap
        P.append((_box(ax_ - 0.02, ax_ + 0.02, -0.05, 0.05, 0.031, 0.040), "#202020"))   # the gearbox and tower
    P.append((_box(0.020, 0.170, -0.042, 0.042, 0.031, 0.056), "#151515"))               # the battery holder
    for k in range(4):                                                                   # its four cells
        y0 = -0.040 + k * 0.020
        P.append((_tube((0.024, y0 + 0.009, 0.050), (0.166, y0 + 0.009, 0.050), 0.0085, 10, caps=False), "#2A2A2A"))
    for (sx, sy) in ((0.025, -0.040), (0.025, 0.040), (0.165, -0.040), (0.165, 0.040)):  # the white standoffs
        P.append((_tube((sx, sy, 0.056), (sx, sy, 0.074), 0.0035, 8), "#EDEDE8"))
    P.append((_box(0.010, 0.180, -0.048, 0.048, 0.074, 0.077), "#101010"))               # the expansion board
    P.append((_box(0.012, 0.178, -0.046, 0.046, 0.0771, 0.0773), "#1F2A1F"))             # its copper-free face
    for k, yy in enumerate((-0.044, -0.026)):                                            # USB 3 ports, blue
        P.append((_box(0.030, 0.050, yy, yy + 0.015, 0.077, 0.093), "#B9BEC4"))
        P.append((_box(0.0295, 0.032, yy + 0.002, yy + 0.013, 0.079, 0.091), "#1E5BB8"))
    P.append((_box(0.012, 0.030, -0.044, -0.026, 0.077, 0.090), "#C8CDD2"))              # the Ethernet socket
    P.append((_box(0.060, 0.150, -0.035, 0.040, 0.077, 0.082), "#0E3B1E"))               # the Jetson module
    for k in range(9):                                                                   # the heat sink's fins
        xx = 0.072 + k * 0.008
        P.append((_box(xx, xx + 0.003, -0.030, 0.035, 0.082, 0.104), "#1B1B1B"))
    P.append((_box(0.070, 0.140, -0.030, 0.035, 0.082, 0.086), "#1B1B1B"))
    P.append((_box(0.075, 0.135, -0.027, 0.032, 0.104, 0.112), "#111111"))               # the fan
    fa = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    P.append(([np.column_stack([0.105 + 0.024 * np.cos(fa), 0.0025 + 0.024 * np.sin(fa), np.full(20, 0.1122)])], "#2C2C2C"))
    P.append(([np.column_stack([0.105 + 0.008 * np.cos(fa), 0.0025 + 0.008 * np.sin(fa), np.full(20, 0.1124)])], "#4A4A4A"))
    # the green camera bracket: a plate standing on the board, leaning forward
    br = [(0.150, -0.028), (0.163, -0.028), (0.163, 0.028), (0.150, 0.028)]
    P.append((_prism(br, 0.077, 0.082), "#5BBF2A"))
    lean = np.array([[0.150, -0.022, 0.080], [0.160, -0.022, 0.080], [0.190, -0.022, 0.150], [0.180, -0.022, 0.150]])
    for f in _prism([(0, 0), (1, 0), (1, 1), (0, 1)], 0, 1):                             # a slab between two y planes
        pts = []
        for (u, w_, h) in f:
            base = lean[0] + u * (lean[1] - lean[0]) + w_ * (lean[3] - lean[0])
            pts.append(base + (0, 0.044 * h, 0))
        P.append(([np.array(pts)], "#62C730"))
    cam_c = np.array([0.188, 0.0, 0.147])                                                # the camera board and lens
    tilt = _rot("y", CAM_PITCH)
    board = [tilt @ v + cam_c for v in np.array([[0.004, -0.013, -0.013], [0.004, 0.013, -0.013], [0.004, 0.013, 0.013], [0.004, -0.013, 0.013]])]
    P.append((_prism([(0, 0)], 0, 0)[:0] + [np.array(board)], "#1D63C9"))
    P.append((_tube(cam_c + tilt @ np.array([0.004, 0, 0]), cam_c + tilt @ np.array([0.016, 0, 0]), 0.0065, 14), "#0D0D0D"))
    P.append((_tube(cam_c + tilt @ np.array([0.016, 0, 0]), cam_c + tilt @ np.array([0.018, 0, 0]), 0.0045, 14), "#2B4A7A"))
    rib = [(0.176, 0.141), (0.168, 0.125), (0.160, 0.105), (0.153, 0.090), (0.146, 0.079)]   # the ribbon cable, down the back of the bracket
    P.append(([np.array([[rib[i][0], -0.008, rib[i][1]], [rib[i + 1][0], -0.008, rib[i + 1][1]],
                         [rib[i + 1][0], 0.008, rib[i + 1][1]], [rib[i][0], 0.008, rib[i][1]]]) for i in range(len(rib) - 1)], "#D9D9D4"))
    for (ax_, ay) in ((0.030, -0.046), (-0.030, -0.040)):                                # two antennas
        P.append((_tube((ax_, ay, 0.077), (ax_, ay, 0.092), 0.004, 10), "#262626"))
        P.append((_tube((ax_, ay, 0.092), (ax_, ay, 0.205), 0.0028, 8), "#141414"))
    for (px, py) in ((0.255, -0.022), (0.255, 0.022), (-0.055, -0.020), (-0.055, 0.020)):  # the body posts
        P.append((_tube((px, py, 0.031), (px, py, 0.095), 0.002, 6), "#121212"))
    wires = [((0.060, -0.030, 0.083), (0.040, 0.000, 0.100), (0.020, 0.030, 0.090), "#D32F2F"),
             ((0.062, -0.026, 0.083), (0.045, 0.012, 0.102), (0.030, 0.040, 0.088), "#FBC02D"),
             ((0.064, -0.022, 0.083), (0.050, 0.020, 0.098), (0.040, 0.042, 0.086), "#1976D2")]
    for (a0, a1, a2, col) in wires:                                                      # a few coloured wires
        P.append((_tube(a0, a1, 0.0012, 5, caps=False) + _tube(a1, a2, 0.0012, 5, caps=False), col))
    return P


_BODY = None
_WHEEL = None


def jetracer_model(x, y, th, delta=0.0, spin=0.0):
    """The JetRacer Pro drawn in 3-D at the pose: ``(faces, colour)`` pairs.
    The rear axle is at (x, y); ``delta`` turns the front wheels, ``spin``
    (radians) turns all four as they roll. A drawing, not a measurement."""
    global _BODY, _WHEEL
    if _BODY is None:
        _BODY, _WHEEL = _body_local(), _wheel_local()
    Rz, t = _rot("z", th), np.array([x, y, 0.0])
    out = [([f @ Rz.T + t for f in faces], col) for faces, col in _BODY]
    for ax_, steer in ((0.0, 0.0), (WHEELBASE, delta)):
        for sgn in (1, -1):
            M = _rot("z", steer) @ (np.eye(3) if sgn > 0 else _rot("z", np.pi)) @ _rot("y", spin * sgn)
            c = np.array([ax_, sgn * TRACK, WHEEL_R])
            for faces, col in _WHEEL:
                out.append(([(f @ M.T + c) @ Rz.T + t for f in faces], col))
    return out


def film(run, title="", fps=25, course=None):
    """A film of a run of :func:`avoid`: the JetRacer in 3-D, seen from
    behind and above as it drives from A to B, beside the frame its camera
    sees at each moment and the course from above. Returns a matplotlib
    animation; ``anim.save("run.mp4")`` writes it, and in a notebook
    ``HTML(anim.to_html5_video())`` plays it."""
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    obstacles = OBSTACLES if course is None else course
    P = run["poses"]
    fig = plt.figure(figsize=(12.8, 6.0), facecolor="white")
    ax3 = fig.add_axes([0.0, 0.0, 6.0 / 12.8, 1.0], projection="3d", computed_zorder=False)   # square: the floor below, by hand
    axc = fig.add_axes([0.52, 0.38, 0.43, 0.58])
    axt = fig.add_axes([0.51, 0.06, 0.47, 0.26])
    obst = [(_tube((ox, oy, 0), (ox, oy, KINDS[k]["height"]), r, 24), {"box": "#262626", "grey": "#8A8478", "low": "#3E2723"}[k])
            for (ox, oy, r, k) in obstacles]
    rng = np.random.default_rng(0)
    pb_path = np.array([(p[1], p[2]) for p in P])

    from matplotlib.colors import LightSource
    light = LightSource(azdeg=225, altdeg=50)
    spin = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(P[:, 1]), np.diff(P[:, 2])))]) / WHEEL_R   # the wheels' turn, rad

    def ellipse(cx_, cy_, a, b, ang, n=24):
        u = np.linspace(0, 2 * np.pi, n, endpoint=False)
        ex, ey = a * np.cos(u), b * np.sin(u)
        return [np.column_stack([cx_ + ex * np.cos(ang) - ey * np.sin(ang), cy_ + ex * np.sin(ang) + ey * np.cos(ang), np.full(n, 0.0008)])]

    def draw(i):
        t, x, y, th, delta, v, seen = P[i]
        cx, cy = x + 0.30 * np.cos(th), y + 0.30 * np.sin(th)     # the middle of the view, ahead of the car
        R_ = 0.55                                                  # half the box of the view, m
        x0, x1, y0, y1 = x - 0.4, x + 5.0, cy - 2.0, cy + 2.0      # what is drawn on the floor: to the horizon ahead
        ax3.cla()
        ax3.set_facecolor("#DADDE2")                               # the far wall
        win = lambda a, b, c_, d: [[(a, c_, 0), (b, c_, 0), (b, d, 0), (a, d, 0)]]
        hx, hy = np.cos(th), np.sin(th)                            # the floor: a rectangle turned with the car, all of it
        corners = [(-0.45, -2.2), (5.0, -2.2), (5.0, 2.2), (-0.45, 2.2)]   # in front of the viewer, so the perspective holds
        fl = [[(x + a_ * hx - b_ * hy, y + a_ * hy + b_ * hx, 0) for a_, b_ in corners]]
        ax3.add_collection3d(Poly3DCollection(fl, facecolor="#BDB8AE", edgecolor="none", zorder=0), autolim=False)
        if x1 > 0 and x0 < COURSE:
            tp = [[(max(x0, 0), -TAPE_HALF, 0.0005), (min(x1, COURSE), -TAPE_HALF, 0.0005),
                   (min(x1, COURSE), TAPE_HALF, 0.0005), (max(x0, 0), TAPE_HALF, 0.0005)]]
            ax3.add_collection3d(Poly3DCollection(tp, facecolor="#F7F7F2", edgecolor="none", zorder=1), autolim=False)
        for xm, col in ((0.0, "#1565C0"), (COURSE, "#2E7D32")):
            if x0 < xm < x1:
                mk = [[(xm - 0.02, -0.5, 0.0006), (xm + 0.02, -0.5, 0.0006), (xm + 0.02, 0.5, 0.0006), (xm - 0.02, 0.5, 0.0006)]]
                ax3.add_collection3d(Poly3DCollection(mk, facecolor=col, edgecolor="none", zorder=1), autolim=False)
        shadows = ellipse(x + 0.10 * np.cos(th) + 0.02, y + 0.10 * np.sin(th) - 0.02, 0.20, 0.11, th)   # soft shadows, light from above
        shadows += sum([ellipse(ox + 0.03, oy - 0.03, r * 1.5, r * 1.3, 0.6) for (ox, oy, r, k) in obstacles if x0 < ox < x1], [])
        ax3.add_collection3d(Poly3DCollection(shadows, facecolor=(0, 0, 0, 0.22), edgecolor="none", zorder=1), autolim=False)
        trail = pb_path[: i + 1]
        trail = trail[(trail[:, 0] > x0) & (trail[:, 0] < x1)]
        if len(trail):
            ax3.plot(trail[:, 0], trail[:, 1], 0.001, color="#FF8F00", lw=1.5, zorder=2)
        faces, cols = [], []                                       # car and obstacles in one collection: sorted by depth together
        seen_obs = [(f, col) for (f, col), (ox, oy, r, k) in zip(obst, obstacles) if x0 < ox < x1]
        for f, col in seen_obs + jetracer_model(x, y, th, delta, spin[i]):
            faces += list(f); cols += [col] * len(f)
        ax3.add_collection3d(Poly3DCollection(faces, facecolors=cols, edgecolor="none", shade=True,
                                              lightsource=light, zorder=3), autolim=False)           # lit from above and to one side
        ax3.set_xlim(cx - R_, cx + R_); ax3.set_ylim(cy - R_, cy + R_); ax3.set_zlim(0, 0.35)
        ax3.set_box_aspect((2 * R_, 2 * R_, 0.35), zoom=1.65)
        ax3.set_proj_type("persp", focal_length=0.35)
        ax3.view_init(elev=16, azim=np.degrees(th) + 180 + 25)                        # behind the car and a little to its left
        ax3.set_axis_off()
        ax3.text2D(0.03, 0.95, title, transform=ax3.transAxes, fontsize=12, weight="bold")
        ax3.text2D(0.03, 0.89, f"t = {t:4.2f} s    speed {v:4.2f} m/s    wheels {np.degrees(delta):+5.1f} deg"
                   + ("    path blocked" if seen else ""), transform=ax3.transAxes, fontsize=10)
        axc.cla(); axc.imshow(render(x, y, th, "ab", obstacles, rng=rng)); axc.axis("off")
        axc.set_title("camera.read(): what the car sees", fontsize=9)
        axt.cla(); draw_course(axt, obstacles=obstacles)
        axt.plot(pb_path[: i + 1, 0], pb_path[: i + 1, 1], color="#FF8F00", lw=1.2)
        axt.plot(x, y, "o", color="#B22222", ms=4)
        axt.set_xlabel(""); axt.set_ylabel(""); axt.tick_params(labelsize=7)
        return []

    anim = animation.FuncAnimation(fig, draw, frames=len(P), interval=1000 / fps, blit=False)
    plt.close(fig)
    return anim
