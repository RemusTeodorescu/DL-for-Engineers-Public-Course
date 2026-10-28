r"""A digital twin of the Yahboom ROSMASTER R2 with its LiDAR and depth camera,
on a closed racing track with obstacles, seeded noise at rising levels, the
conventional perception rules, two planners and a 3-D film. Shared by the mini
projects MP11A and MP11B of Ex11 (course policy C14: twin only, in Colab).

*Deep Learning for Engineering* — MSc, Aalborg University.
Remus Teodorescu (ret@et.aau.dk), with support from Research Assistant
Noman Khan (nomank@energy.aau.dk).

This module extends Ex11's ``problem.py`` (imported as ``pb``; course policy
C10 rule 1: one set of physics for the exercise and its mini projects): the
obstacle colours and the floor, the kinematic bicycle and its speed lag, the
potential field's form and the 3-D drawing helpers are Ex11's. What is new is
the car, its sensors, the track and the noise.

## The car (``spec()`` prints the table)

What the vendor states (docs/hardware/rosmaster-r2.md; the sensors' own
makers for the sensors): Jetson Orin Nano Super 8 GB, Ackermann steering,
65 mm wheels, at most 1.8 m/s, a SLAM A1 LiDAR (Slamtec RPLIDAR A1: 360
degrees, 0.15 - 12 m, 5.5 scans a second typical), an Astra Pro Plus depth
camera (Orbbec: depth 0.6 - 8 m, 640 x 480, 58.4 x 45.5 degrees, about 3 mm
at 1 m; colour 66.1 x 40.2 degrees), odometry and an IMU. **Everything else -
the size, the wheelbase, the steering limit, the sensors' heights and tilt,
the speed lag, the friction, the noise of each sensor beyond the vendor's
accuracy - is ASSUMED**, marked so below, and to be measured on the car.

## What the twin gives

* ``Track(seed)``: a closed track, 1.1 m wide between two taped edges, with a
  chicane, an S-bend and two turns, 4 to 8 obstacles of four kinds and, by
  noise level, shadows, sun on the floor and a dim room. Start and finish (A
  and B) are the same line: one lap.
* ``R2(track, level)``: the car - ``car.cmd(speed, wheel_angle)``, ``car.step(dt)``
  - and its three sensors: ``car.rgb()`` (64 x 112 colour frame), ``car.depth()``
  (48 x 64 depth image in metres, 0 where there is no reading), ``car.scan()``
  (360 ranges, 0 where there is no return).
* the conventional rules on each sensor (``camera_rule``, ``depth_rule``,
  ``lidar_rule``), perfect perception (``oracle``), two planners
  (``SteerAround``, ``FieldAround``) and ``drive(...)``, one lap in closed loop.
* ``samples(...)``: labelled sensor data for training and testing.
* ``r2_model``, ``view3d``, ``film``: the R2 in 3-D, as Ex11's JetRacer.

The car localises itself on the known track by its odometry and IMU (vendor:
"Odometer + IMU data"), assumed exact in the twin: what it must perceive are
the obstacles, which are not on the map.
"""
from __future__ import annotations

import numpy as np

import problem as pb                                   # Ex11's virtual JetRacer: colours, kinematics, 3-D helpers

# ====================================================================== the car (VENDOR / ASSUMED)
V_MAX = 1.8               #: m/s, the largest speed (VENDOR)
WHEEL_D = 0.065           #: m, wheel diameter (VENDOR: 65 mm anti-skid wheels)
WHEELBASE = 0.23          #: m (ASSUMED: not stated by the vendor)
TRACK_W = 0.20            #: m, centre-to-centre between the left and right wheels (ASSUMED)
HALF_W = 0.12             #: m, half the car's width (ASSUMED)
REAR = 0.07               #: m, the body behind the rear axle (ASSUMED)
FRONT = 0.09              #: m, the body ahead of the front axle (ASSUMED)
STEER_MAX = np.radians(25.0)   #: largest wheel angle (ASSUMED; servo resolution 0.1 degree is VENDOR)
SPEED_LAG = 0.30          #: s, time constant of the speed after a command (ASSUMED, as Ex11's)
MU = 0.60                 #: tyre-floor friction (ASSUMED, as Ex11's)
G = 9.81

# ---------------------------------------------------------------- the colour camera of the Astra Pro Plus
CAM_W, CAM_H = 112, 64    #: pixels of the frame the network sees: the 1920 x 1080 stream (VENDOR) reduced 17 times
CAM_HFOV = np.radians(66.1)    #: colour field of view, horizontal (VENDOR: 66.1 x 40.2 degrees)
CAM_HEIGHT = 0.17         #: m above the floor (ASSUMED: the bracket at the front)
CAM_AHEAD = 0.32          #: m ahead of the rear axle (ASSUMED)
CAM_PITCH = np.radians(10.0)   #: tilted down (ASSUMED)
CAM_HZ = 30.0             #: frames a second (VENDOR, both streams)
# ---------------------------------------------------------------- the depth camera of the Astra Pro Plus
DEP_W, DEP_H = 64, 48     #: pixels: the 640 x 480 depth stream (VENDOR) reduced 10 times
DEP_HFOV = np.radians(58.4)    #: depth field of view, horizontal (VENDOR: 58.4 x 45.5 degrees)
DEP_MIN, DEP_MAX = 0.6, 8.0    #: m, the depth range (VENDOR); outside it the camera returns 0
DEP_ACC = 0.003           #: m, the depth error at 1 m (VENDOR: +-3 mm at 1 m); growing with distance squared (ASSUMED)
# ---------------------------------------------------------------- the LiDAR
LIDAR_BEAMS = 360         #: one range a degree (VENDOR, RPLIDAR A1: 1 degree or finer)
LIDAR_MIN, LIDAR_MAX = 0.15, 12.0   #: m (VENDOR, RPLIDAR A1)
LIDAR_HZ = 5.5            #: scans a second (VENDOR, typical)
LIDAR_HEIGHT = 0.20       #: m, the scan plane above the floor: the top deck (ASSUMED)
LIDAR_AHEAD = 0.12        #: m ahead of the rear axle (ASSUMED)
# ---------------------------------------------------------------- the drive
RATE_HZ = 10.0            #: decisions a second, as Ex11's
RANGE = 1.50              #: m: an obstacle closer than this ahead of the front axle, in the lane, blocks the path (ASSUMED)
LANE = 0.55               #: m: the zone watched, to either side of the car's axis to an obstacle's edge: the track's half-width,
                          #: since in a bend an obstacle on the track ahead stands well to the side of the car's axis (ASSUMED)


def spec():
    """The twin's numbers, each marked VENDOR or ASSUMED, as rows (what, value, source)."""
    deg = np.degrees
    return [
        ("controller", "Jetson Orin Nano Super 8 GB", "VENDOR (Yahboom)"),
        ("steering", "Ackermann; servo resolution 0.1 degree", "VENDOR (Yahboom)"),
        ("largest speed", f"{V_MAX:.1f} m/s", "VENDOR (Yahboom)"),
        ("wheels", f"{1e3 * WHEEL_D:.0f} mm diameter", "VENDOR (Yahboom)"),
        ("localisation", "odometry and IMU (exact in the twin)", "VENDOR states the sensors; exactness ASSUMED"),
        ("wheelbase, track", f"{WHEELBASE:.2f} m, {TRACK_W:.2f} m", "ASSUMED"),
        ("body", f"{2 * HALF_W:.2f} m wide, {REAR + WHEELBASE + FRONT:.2f} m long", "ASSUMED"),
        ("steering limit", f"{deg(STEER_MAX):.0f} degrees (turning radius {WHEELBASE / np.tan(STEER_MAX):.2f} m)", "ASSUMED"),
        ("speed lag, friction", f"{SPEED_LAG:.2f} s, {MU:.1f}", "ASSUMED (Ex11's values)"),
        ("colour camera", f"field of view {deg(CAM_HFOV):.1f} x 40.2 degrees, 1920 x 1080, 30 frames/s", "VENDOR (Orbbec Astra Pro Plus)"),
        ("colour frame used", f"{CAM_W} x {CAM_H} pixels", "this course's reduction"),
        ("depth camera", f"{DEP_MIN:.1f} - {DEP_MAX:.0f} m, 640 x 480, {deg(DEP_HFOV):.1f} x 45.5 degrees, +-3 mm at 1 m", "VENDOR (Orbbec Astra Pro Plus)"),
        ("depth image used", f"{DEP_W} x {DEP_H} pixels", "this course's reduction"),
        ("depth camera size", "165 x 48 x 40 mm", "VENDOR (Orbbec)"),
        ("camera mount", f"{100 * CAM_HEIGHT:.0f} cm high, {deg(CAM_PITCH):.0f} degrees down, {100 * CAM_AHEAD:.0f} cm ahead of the rear axle", "ASSUMED"),
        ("LiDAR", f"{LIDAR_BEAMS} ranges a turn, {LIDAR_MIN:.2f} - {LIDAR_MAX:.0f} m, {LIDAR_HZ:.1f} scans/s", "VENDOR (Slamtec RPLIDAR A1)"),
        ("LiDAR mount", f"scan plane {100 * LIDAR_HEIGHT:.0f} cm above the floor, {100 * LIDAR_AHEAD:.0f} cm ahead of the rear axle", "ASSUMED"),
        ("sensor noise", "per noise level, table NOISE", "ASSUMED"),
        ("ultrasonic sensor", "none (an optional add-on, not modelled)", "VENDOR states none"),
    ]


# ====================================================================== the track
W_HALF = 0.55             #: m, half the track's width, centre line to a taped edge (ASSUMED: a lab track)
EDGE_HALF = 0.025         #: m, half the width of the 50 mm edge tape (ASSUMED)
#: the track's shape: control points of a closed curve, anticlockwise, in metres (ASSUMED)
_BASE = np.array([(-0.6, 0.0), (0.6, 0.0), (2.0, 0.0), (3.0, 0.6), (3.6, 0.6), (4.6, 0.0), (6.0, 0.15), (6.9, 0.9),
                  (7.1, 2.0), (6.6, 3.0), (5.6, 3.45), (4.5, 3.15), (3.4, 3.65), (2.2, 3.85), (0.9, 3.7), (-0.3, 3.2),
                  (-1.1, 2.3), (-1.35, 1.25), (-1.1, 0.45)])
_START = (0.3, 0.0)       #: m: the start line, on the start straight
SMOOTH = 0.50             #: m, the width of the Gaussian that rounds the spline's tightest bends (the tightest radius stays above 0.75 m)
KINDS = dict(pb.KINDS)    #: Ex11's three kinds, made the R2's size, and a fourth
KINDS["box"] = dict(colour=(0.12, 0.12, 0.14), height=0.30, gloss=0.0)    # dark and tall: every sensor sees it
KINDS["grey"] = dict(colour=(0.43, 0.41, 0.38), height=0.30, gloss=0.0)   # the floor's colour: the camera barely sees it
KINDS["low"] = dict(colour=(0.14, 0.12, 0.10), height=0.10, gloss=0.0)    # under the LiDAR's plane
KINDS["black"] = dict(colour=(0.04, 0.04, 0.05), height=0.30, gloss=1.0)  # glossy black: little light returns to the LiDAR and depth camera
KIND_NAMES = ("box", "grey", "low", "black")
ROOM_MARGIN = 1.5         #: m between the track's extent and the room's walls (ASSUMED)
ROOM_HEIGHT = 2.6         #: m (ASSUMED)

#: the noise levels (ASSUMED): 0 a clean lab, 1 a normal lab, 2 a difficult day, 3 a dim room with sun, 4 extreme
NOISE = {
    #       pixel   light range     slope  shadows sun  blur rows   lidar     lidar    depth     depth     sun
    #       noise                          (n)     (n)  per m/s     range     dropout  noise x   holes     holes
    0: dict(px=0.010, light=(1.00, 1.00), slope=0.0, shadows=0, sun=0, blur=0.0, l_noise=0.005, l_drop=0.00, d_noise=1.0, d_hole=0.00, d_sun=0.0),
    1: dict(px=0.020, light=(0.85, 1.00), slope=0.2, shadows=2, sun=1, blur=0.6, l_noise=0.010, l_drop=0.02, d_noise=1.5, d_hole=0.02, d_sun=0.5),
    2: dict(px=0.035, light=(0.70, 0.85), slope=0.3, shadows=3, sun=2, blur=1.2, l_noise=0.015, l_drop=0.05, d_noise=2.0, d_hole=0.05, d_sun=0.7),
    3: dict(px=0.050, light=(0.50, 0.65), slope=0.4, shadows=4, sun=3, blur=1.8, l_noise=0.020, l_drop=0.10, d_noise=3.0, d_hole=0.10, d_sun=0.9),
    4: dict(px=0.070, light=(0.35, 0.50), slope=0.5, shadows=5, sun=4, blur=2.4, l_noise=0.030, l_drop=0.15, d_noise=4.0, d_hole=0.15, d_sun=1.0),
}
GLOSS_DROP_LIDAR = 0.85   #: share of LiDAR beams lost on a glossy black surface (ASSUMED)
GLOSS_HOLE_DEPTH = 0.85   #: share of depth pixels lost on a glossy black surface (ASSUMED)


class Track:
    """A closed track from a seed: the centre line (a periodic spline through
    :data:`_BASE`, each control point moved by up to 0.12 m and the chicane's
    depth drawn from 0.30 to 0.50 m), sampled every centimetre; 4 to 8
    obstacles at least 1.6 m apart along it, each ``(x, y, radius, kind)``;
    and, for the noise ``level``, shadows and sun patches ``(x, y, size)`` and
    the room's light. The obstacles depend on the seed only, the conditions
    on the seed and the level. ``kinds`` weights the four obstacle kinds."""

    def __init__(self, seed=0, level=0, kinds=(0.35, 0.25, 0.20, 0.20), n_obstacles=None):
        from scipy.interpolate import CubicSpline
        from scipy.ndimage import gaussian_filter1d
        from scipy.spatial import cKDTree
        self.seed, self.level = int(seed), int(level)
        rng = np.random.default_rng([self.seed, 11])
        P = _BASE.copy()
        P[3:] += rng.uniform(-0.12, 0.12, P[3:].shape)               # the start straight stays straight
        P[3:5, 1] = rng.uniform(0.55, 0.75)                           # the chicane's sideways jog
        Q = np.vstack([P, P[:1]])
        u = np.r_[0, np.cumsum(np.hypot(*np.diff(Q, axis=0).T))]
        sp = CubicSpline(u, Q, bc_type="periodic")
        xy = sp(np.linspace(0, u[-1], 20001)[:-1])
        sl = np.r_[0, np.cumsum(np.hypot(*np.diff(np.vstack([xy, xy[:1]]), axis=0).T))]
        self.length = float(sl[-1])
        self.s = np.arange(0.0, self.length, 0.01)
        x = gaussian_filter1d(np.interp(self.s, sl[:-1], xy[:, 0]), SMOOTH / 0.01, mode="wrap")   # rounded: no bend tighter than the car can follow
        y = gaussian_filter1d(np.interp(self.s, sl[:-1], xy[:, 1]), SMOOTH / 0.01, mode="wrap")
        k0 = int(np.argmin((x - _START[0]) ** 2 + (y - _START[1]) ** 2))   # the start line A = B, on the start straight
        x, y = np.roll(x, -k0), np.roll(y, -k0)
        wrap = lambda a: np.gradient(np.r_[a[-2:], a, a[:2]])[2:-2]
        dx, dy = wrap(x), wrap(y)
        seg = np.hypot(dx, dy)
        self.s = np.r_[0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]          # arc length after the rounding
        self.length = float(self.s[-1] + np.hypot(x[0] - x[-1], y[0] - y[-1]))
        self.x, self.y = x, y
        self.th = np.arctan2(dy, dx)
        self._thu = np.unwrap(self.th)
        self.kappa = (dx * wrap(dy) - dy * wrap(dx)) / seg ** 3
        self._tree = cKDTree(np.column_stack([self.x, self.y]))
        # the obstacles: from the seed only
        n = int(rng.integers(4, 9)) if n_obstacles is None else int(n_obstacles)
        w = np.asarray(kinds, float) / np.sum(kinds)
        cand = np.sort(rng.uniform(1.6, self.length - 1.2, 60))
        pick = [cand[0]]
        for c in cand[1:]:
            if c - pick[-1] >= 1.6 and len(pick) < n:
                pick.append(c)
        self.obstacles, self.obstacle_s = [], []
        for so in pick:
            d = rng.choice([-1, 1]) * rng.uniform(0.0, 0.25)
            X, Y = self.point(so, d)
            self.obstacles.append((float(X), float(Y), float(rng.uniform(0.07, 0.10)), str(rng.choice(KIND_NAMES, p=w))))
            self.obstacle_s.append((float(so), float(d)))
        # the room
        self.room = (self.x.min() - ROOM_MARGIN, self.x.max() + ROOM_MARGIN, self.y.min() - ROOM_MARGIN, self.y.max() + ROOM_MARGIN)
        # the conditions: from the seed and the level
        c = NOISE[self.level]
        rc = np.random.default_rng([self.seed, 23, self.level])
        self.light = float(rc.uniform(*c["light"]))
        self.slope = float(rc.uniform(-c["slope"], c["slope"]))
        self.shadows = [(*self.point(rc.uniform(0, self.length), rc.uniform(-0.4, 0.4)), float(rc.uniform(0.12, 0.22)))
                        for _ in range(c["shadows"])]
        self.sun = [(*self.point(rc.uniform(0, self.length), rc.uniform(-0.45, 0.45)), float(rc.uniform(0.08, 0.16)))
                    for _ in range(c["sun"])]
        self.noise = c

    # ------------------------------------------------------------------ geometry
    def point(self, s, d=0.0):
        """The floor point at arc length ``s`` and ``d`` to the left of the centre line."""
        s = np.mod(s, self.length)
        x = np.interp(s, self.s, self.x); y = np.interp(s, self.s, self.y)
        th = np.interp(s, self.s, self._thu)
        return x - np.sin(th) * d, y + np.cos(th) * d

    def heading(self, s):
        return float(np.interp(np.mod(s, self.length), self.s, self._thu))

    def frenet(self, X, Y):
        """Arc length ``s`` and signed offset ``d`` (left positive) of floor points."""
        X, Y = np.asarray(X, float), np.asarray(Y, float)
        _, i = self._tree.query(np.column_stack([X.ravel(), Y.ravel()]))
        dx, dy = X.ravel() - self.x[i], Y.ravel() - self.y[i]
        c, s_ = np.cos(self.th[i]), np.sin(self.th[i])
        return (self.s[i] + dx * c + dy * s_).reshape(X.shape), (-dx * s_ + dy * c).reshape(X.shape)

    def describe(self):
        k = [o[3] for o in self.obstacles]
        print(f"track seed {self.seed}, noise level {self.level}: lap {self.length:.2f} m, {W_HALF * 2:.1f} m wide, "
              f"tightest turn radius {1 / np.abs(self.kappa).max():.2f} m; {len(k)} obstacles "
              f"({', '.join(f'{k.count(n)} {n}' for n in KIND_NAMES if k.count(n))}); "
              f"light {self.light:.2f}, {len(self.shadows)} shadows, {len(self.sun)} sun patches")


# ====================================================================== the sensors
def _rays(w, h, hfov, pitch):
    """Per-pixel rays of a pinhole camera tilted down by ``pitch``, in the
    car's frame (forward, left, up), scaled so that the camera-axis component
    is the focal length: a hit at ray parameter t lies t * f along the axis."""
    f = (w / 2) / np.tan(hfov / 2)
    u, v = np.meshgrid(np.arange(w, dtype=float), np.arange(h, dtype=float))
    cu, cv = (w - 1) / 2, (h - 1) / 2
    RF = -(v - cv) * np.sin(pitch) + f * np.cos(pitch)
    RL = -(u - cu)
    RU = -(v - cv) * np.cos(pitch) - f * np.sin(pitch)
    return f, RF, RL, RU, v


F_CAM, _CF, _CL, _CU, _CV = _rays(CAM_W, CAM_H, CAM_HFOV, CAM_PITCH)
F_DEP, _DF, _DL, _DU, _DV = _rays(DEP_W, DEP_H, DEP_HFOV, CAM_PITCH)
CAM_VFOV = 2 * np.arctan((CAM_H / 2) / F_CAM)
HORIZON = (CAM_H - 1) / 2 - F_CAM * np.tan(CAM_PITCH)        #: the horizon's row in the colour frame
# the flat floor seen by the colour camera, in the car's frame: m ahead of the rear axle and to the left
_FLOOR = _CU < 0
_TF = np.where(_FLOOR, CAM_HEIGHT / np.where(_FLOOR, -_CU, 1.0), np.inf)
FLOOR_F = np.where(_FLOOR, _CF * _TF + CAM_AHEAD, np.nan)
FLOOR_L = np.where(_FLOOR, _CL * _TF, np.nan)
NEAREST_FLOOR = float(np.nanmin(FLOOR_F) - CAM_AHEAD)          #: m: the camera sees the floor from this far ahead of it


def _trace(track, x, y, th, RF, RL, RU, height, ahead, far=9.0):
    """Cast every ray of a camera at the car's pose into the scene. Returns
    the ray parameter of the first hit ``t``, what was hit ``hit`` (0 floor, 1
    wall or ceiling, 2 + i obstacle i), the floor point of floor hits, the
    obstacle's surface normal (for its shading) and whether the hit is an
    obstacle's top."""
    c, s = np.cos(th), np.sin(th)
    ox, oy, oz = x + c * ahead, y + s * ahead, height
    DX, DY, DZ = c * RF - s * RL, s * RF + c * RL, RU
    with np.errstate(divide="ignore", invalid="ignore"):
        t_floor = np.where(DZ < 0, oz / -DZ, np.inf)
        x0, x1, y0, y1 = track.room
        tx = np.where(DX > 0, (x1 - ox) / DX, np.where(DX < 0, (x0 - ox) / DX, np.inf))
        ty = np.where(DY > 0, (y1 - oy) / DY, np.where(DY < 0, (y0 - oy) / DY, np.inf))
        tc = np.where(DZ > 0, (ROOM_HEIGHT - oz) / DZ, np.inf)
        tw = np.minimum(np.minimum(tx, ty), tc)
    t = np.minimum(t_floor, tw)
    hit = np.where(t_floor <= tw, 0, 1)
    nx = np.zeros(DX.shape); ny = np.zeros(DX.shape); top = np.zeros(DX.shape, bool)
    for i, (px_, py_, r, kind) in enumerate(track.obstacles):
        if np.hypot(px_ - ox, py_ - oy) > far:
            continue
        h = KINDS[kind]["height"]
        px, py = ox - px_, oy - py_
        a = DX * DX + DY * DY
        b = 2 * (px * DX + py * DY)
        disc = b * b - 4 * a * (px * px + py * py - r * r)
        with np.errstate(invalid="ignore", divide="ignore"):
            ts = (-b - np.sqrt(np.maximum(disc, 0))) / (2 * a)
        z = oz + ts * DZ
        side = (disc > 0) & (ts > 0) & (z >= 0) & (z <= h) & (ts < t)
        t = np.where(side, ts, t); hit = np.where(side, 2 + i, hit)
        nx = np.where(side, (px + ts * DX) / r, nx); ny = np.where(side, (py + ts * DY) / r, ny)
        top = top & ~side
        if h < oz:                                             # a low obstacle: its top is seen too
            with np.errstate(divide="ignore", invalid="ignore"):
                tt = np.where(DZ < 0, (h - oz) / DZ, np.inf)
            on = (DZ < 0) & ((px + tt * DX) ** 2 + (py + tt * DY) ** 2 <= r * r) & (tt < t)
            t = np.where(on, tt, t); hit = np.where(on, 2 + i, hit); top = top | on
    fx, fy = ox + t * DX, oy + t * DY
    return t, hit, fx, fy, nx, ny, top


_LIGHT = np.array([-0.5, 0.8]) / np.hypot(0.5, 0.8)            #: the room's light, as Ex11's


def _floor_colour(track, fx, fy, on_floor):
    """Floor colour with the taped edges, the start line, shadows and sun."""
    s, d = track.frenet(np.where(on_floor, fx, track.x[0]), np.where(on_floor, fy, track.y[0]))
    tape = (np.abs(np.abs(d) - W_HALF) < EDGE_HALF) | ((np.mod(s + 0.03, track.length) < 0.06) & (np.abs(d) < W_HALF))
    img = np.where(tape[None], pb.TAPE_RGB[:, None, None], pb.FLOOR_RGB[:, None, None])
    shade = np.ones(fx.shape)
    for (sx, sy, sz) in track.shadows:
        shade = shade * (1 - 0.45 * np.exp(-((fx - sx) ** 2 + (fy - sy) ** 2) / sz ** 2))
    img = img * shade
    sun = np.zeros(fx.shape)
    for (gx, gy, gz) in track.sun:
        sun = sun + np.exp(-((fx - gx) ** 2 + (fy - gy) ** 2) / gz ** 2)
    return img + 0.55 * sun, sun


def render_rgb(track, x, y, th, rng=None, speed=0.0, level=None):
    """The colour frame at the car's pose: ``(CAM_H, CAM_W, 3)`` uint8, red-
    green-blue. The noise level (the track's own by default) sets the pixel
    noise, the room's light, its slope across the frame and the motion blur
    at ``speed`` m/s."""
    rng = np.random.default_rng() if rng is None else rng
    c = NOISE[track.level if level is None else level]
    t, hit, fx, fy, nx, ny, top = _trace(track, x, y, th, _CF, _CL, _CU, CAM_HEIGHT, CAM_AHEAD)
    floor = hit == 0
    img, _ = _floor_colour(track, fx, fy, floor)
    img = np.where(floor[None], img, pb.WALL_RGB[:, None, None] * (0.9 + 0.1 * np.clip(_CV / max(HORIZON, 1), 0, 1)))
    for i, (_, _, _, kind) in enumerate(track.obstacles):
        on = hit == 2 + i
        if not on.any():
            continue
        col = np.array(KINDS[kind]["colour"])[:, None, None]
        lit = 0.65 + 0.45 * np.clip(nx * _LIGHT[0] + ny * _LIGHT[1], 0, 1)
        lit = np.where(top, 1.25, lit)
        shine = 0.25 * KINDS[kind]["gloss"] * np.clip(nx * _LIGHT[0] + ny * _LIGHT[1], 0, 1) ** 8   # a glossy highlight
        img = np.where(on[None], col * lit + shine, img)
    img = img * track.light * (1 + track.slope * (np.arange(CAM_W) / (CAM_W - 1) - 0.5))[None, None, :]
    k = int(round(c["blur"] * speed))
    if k > 0:                                                    # motion blur: the scene moves down the frame as the car advances
        acc = img.copy()
        for j in range(1, k + 1):
            acc[:, j:, :] += img[:, :-j, :]
            acc[:, :j, :] += img[:, :1, :]
        img = acc / (k + 1)
    img = img + rng.normal(0, c["px"], img.shape) / max(track.light, 0.3) ** 0.5   # darker room: noisier pixels
    return np.round(255 * np.clip(img, 0, 1)).astype(np.uint8).transpose(1, 2, 0)


def render_depth(track, x, y, th, rng=None, level=None):
    """The depth image at the car's pose: ``(DEP_H, DEP_W)`` float32, the
    distance along the camera's axis in metres, 0 where there is no reading:
    outside 0.6 - 8 m (VENDOR), on glossy black surfaces, in sun on the
    floor (the projected infrared pattern is drowned), and at random by noise
    level; the error grows with the distance squared from the vendor's 3 mm at
    1 m."""
    rng = np.random.default_rng() if rng is None else rng
    c = NOISE[track.level if level is None else level]
    t, hit, fx, fy, nx, ny, top = _trace(track, x, y, th, _DF, _DL, _DU, CAM_HEIGHT, CAM_AHEAD)
    z = t * F_DEP                                                # along the camera's axis
    z = z + rng.normal(0, 1, z.shape) * DEP_ACC * c["d_noise"] * z ** 2
    hole = (z < DEP_MIN) | (z > DEP_MAX) | (rng.random(z.shape) < c["d_hole"])
    for i, (_, _, _, kind) in enumerate(track.obstacles):
        if KINDS[kind]["gloss"] > 0:
            hole |= (hit == 2 + i) & (rng.random(z.shape) < GLOSS_HOLE_DEPTH * KINDS[kind]["gloss"])
    if track.sun and c["d_sun"] > 0:
        _, sun = _floor_colour(track, fx, fy, hit == 0)
        hole |= (hit == 0) & (rng.random(z.shape) < c["d_sun"] * np.clip(2 * sun, 0, 1))
    return np.where(hole, 0.0, z).astype(np.float32)


def _ray_box(ox, oy, ang, room):
    x0, x1, y0, y1 = room
    dx, dy = np.cos(ang), np.sin(ang)
    with np.errstate(divide="ignore"):
        tx = np.where(dx > 0, (x1 - ox) / dx, np.where(dx < 0, (x0 - ox) / dx, np.inf))
        ty = np.where(dy > 0, (y1 - oy) / dy, np.where(dy < 0, (y0 - oy) / dy, np.inf))
    return np.minimum(tx, ty)


def render_scan(track, x, y, th, rng=None, level=None):
    """One LiDAR scan at the car's pose: ``LIDAR_BEAMS`` ranges in metres,
    beam ``i`` at ``i`` degrees anticlockwise from straight ahead, 0 where
    nothing returns (as the RPLIDAR reports). Only obstacles taller than the
    scan plane return; glossy black ones lose most beams; the room's walls
    return too."""
    rng = np.random.default_rng() if rng is None else rng
    c = NOISE[track.level if level is None else level]
    ox, oy = x + np.cos(th) * LIDAR_AHEAD, y + np.sin(th) * LIDAR_AHEAD
    ang = th + np.radians(np.arange(LIDAR_BEAMS) * 360.0 / LIDAR_BEAMS)
    d = _ray_box(ox, oy, ang, track.room)
    for (px, py, r, kind) in track.obstacles:
        if KINDS[kind]["height"] > LIDAR_HEIGHT:
            hit = pb._ray_circle(ox, oy, ang, px, py, r)
            lost = (rng.random(LIDAR_BEAMS) < GLOSS_DROP_LIDAR * KINDS[kind]["gloss"]) & np.isfinite(hit)
            d = np.where(lost & (hit < d), 0.0, np.minimum(d, hit))     # a lost beam on the obstacle returns nothing
    d = d * (1 + c["l_noise"] * rng.normal(size=d.shape))
    d = np.where(rng.random(LIDAR_BEAMS) < c["l_drop"], 0.0, d)
    return np.where((d >= LIDAR_MIN) & (d <= LIDAR_MAX), d, 0.0).astype(np.float32)


# ====================================================================== the car
class R2:
    """The ROSMASTER R2 on a track: ``cmd(speed, wheel_angle)`` sets the
    speed it approaches (time constant :data:`SPEED_LAG`, no faster than the
    tyres allow) and the front wheels' angle (clipped to :data:`STEER_MAX`),
    ``step(dt)`` moves it by the kinematic bicycle model - Ex11's equations
    with the R2's numbers. ``rgb()``, ``depth()`` and ``scan()`` read the
    sensors at the current pose; ``gap()`` is the smallest gap between the
    body and any obstacle (negative: a collision)."""

    def __init__(self, track, s=0.0, d=0.0, dth=0.0, speed=0.0, seed=0, level=None):
        self.track = track
        self.level = track.level if level is None else level
        x, y = track.point(s, d)
        self.x, self.y, self.th, self.v = float(x), float(y), track.heading(s) + dth, float(speed)
        self.v_set, self.delta = float(speed), 0.0
        self.rng = np.random.default_rng([seed, 31])
        self._obs = np.array([o[:3] for o in track.obstacles]).reshape(-1, 3)

    def cmd(self, speed, wheel_angle):
        self.v_set = float(np.clip(speed, 0.0, V_MAX))
        self.delta = float(np.clip(wheel_angle, -STEER_MAX, STEER_MAX))

    def step(self, dt):
        dv = np.clip((self.v_set - self.v) / SPEED_LAG * dt, -MU * G * dt, MU * G * dt)   # no faster than the tyres allow
        self.v += dv
        self.x += self.v * np.cos(self.th) * dt
        self.y += self.v * np.sin(self.th) * dt
        self.th += self.v / WHEELBASE * np.tan(self.delta) * dt

    def gap(self):
        """The body as a capsule from REAR behind the rear axle to FRONT ahead
        of the front axle, HALF_W wide: the smallest gap to any obstacle, m."""
        if len(self._obs) == 0:
            return np.inf
        c, s = np.cos(self.th), np.sin(self.th)
        a = np.array([self.x - c * REAR, self.y - s * REAR])
        L = REAR + WHEELBASE + FRONT
        P = self._obs[:, :2] - a
        u = np.clip(P @ np.array([c, s]), 0, L)
        q = a + u[:, None] * np.array([c, s])
        return float((np.hypot(*(self._obs[:, :2] - q).T) - self._obs[:, 2] - HALF_W).min())

    def rgb(self):
        return render_rgb(self.track, self.x, self.y, self.th, self.rng, self.v, self.level)

    def depth(self):
        return render_depth(self.track, self.x, self.y, self.th, self.rng, self.level)

    def scan(self):
        return render_scan(self.track, self.x, self.y, self.th, self.rng, self.level)


def check_kinematics(T=6.0, dt=2e-3):
    """The zero-change check: with the JetRacer's numbers the R2's ``step``
    must reproduce Ex11's ``VirtualJetRacer.step`` exactly. Drives both with
    the same commands for ``T`` s; returns the largest difference in position, m."""
    import types
    jr = pb.VirtualJetRacer(0.0, 0.0, 0.0)
    tr = types.SimpleNamespace(point=lambda s, d: (0.0, 0.0), heading=lambda s: 0.0, obstacles=[], level=0)
    car = R2(tr)
    g = globals()
    keep = dict(WHEELBASE=g["WHEELBASE"], STEER_MAX=g["STEER_MAX"], V_MAX=g["V_MAX"])
    g.update(WHEELBASE=pb.WHEELBASE, STEER_MAX=pb.STEER_MAX, V_MAX=10.0)
    worst = 0.0
    try:
        for k in range(int(T / dt)):
            t = k * dt
            jr.steering, jr.throttle = 0.8 * np.sin(1.3 * t), 0.4 + 0.2 * np.cos(0.7 * t)
            car.cmd(pb.V_PER_DUTY * jr.throttle * jr.throttle_gain, jr.wheel_angle())
            jr.step(dt); car.step(dt)
            worst = max(worst, np.hypot(jr.x - car.x, jr.y - car.y))
    finally:
        g.update(keep)
    return worst


def check_twin(verbose=True):
    """Physics checks of the twin, each against a value computed by hand:
    the R2's motion with the JetRacer's numbers against Ex11's car (zero
    change); a noiseless LiDAR range and depth reading against the distance to
    an obstacle; the depth of the bare floor against h / sin(angle); the lap
    of an empty track at a constant speed against its length over the speed."""
    out = {"kinematics": check_kinematics()}
    tr = Track(0, 0, n_obstacles=0)
    x, y = tr.point(1.0); th = tr.heading(1.0)
    ox, oy = tr.point(3.0); r_ = 0.09
    tr.obstacles = [(float(ox), float(oy), r_, "box")]
    noiseless = dict(NOISE[0], l_noise=0.0, d_noise=0.0)
    keep = NOISE[0]
    NOISE[0] = noiseless
    try:
        sc = render_scan(tr, x, y, th, np.random.default_rng(0))
        lx, ly = x + np.cos(th) * LIDAR_AHEAD, y + np.sin(th) * LIDAR_AHEAD
        brg = int(round(np.degrees(np.arctan2(oy - ly, ox - lx) - th))) % 360
        out["lidar"] = abs(float(sc[brg]) - float(pb._ray_circle(lx, ly, np.radians(brg) + th, ox, oy, r_)))
        dep = render_depth(Track(0, 0, n_obstacles=0), x, y, th, np.random.default_rng(0))
        c = DEP_W // 2
        ang = np.arctan2(-_DU[:, c], _DF[:, c])                           # each row's ray below the horizontal
        z_all = np.where(ang > 0, CAM_HEIGHT / np.sin(np.abs(ang)) * np.cos(ang - CAM_PITCH), np.inf)   # along the camera's axis
        row = int(np.argmin(np.abs(z_all - 1.0)))                         # the row that sees the floor about 1 m away
        z_exact = z_all[row]
        out["depth floor"] = abs(float(dep[row, DEP_W // 2]) - z_exact) if dep[row, DEP_W // 2] > 0 else np.nan
    finally:
        NOISE[0] = keep
    empty = Track(0, 0, n_obstacles=0)
    lap = drive(empty, oracle(empty), FieldAround(), 1.0, need=(), a_lat=None)
    out["empty lap"] = lap["time"] / (empty.length / 1.0) - 1
    if verbose:
        print(f"  the R2's motion with the JetRacer's numbers against Ex11's car: {out['kinematics']:.1e} m apart at most")
        print(f"  LiDAR range to an obstacle 2 m ahead, noise off, against the exact distance: {out['lidar']:.1e} m")
        print(f"  depth of the bare floor about 1 m ahead (row {row}), noise off, against h/sin: {out['depth floor']:.1e} m")
        print(f"  empty track, 1 m/s, no bend limit: lap {lap['time']:.3f} s against L/v = {empty.length:.3f} s "
              f"({100 * out['empty lap']:+.2f} %: pure pursuit does not drive exactly on the centre line)")
    return out


# ====================================================================== what is true: the labels
def _in_zone(f, s, r, grow=0.0):
    """Whether an obstacle at ``(f, s)`` from the rear axle, of radius ``r``,
    blocks the path; ``grow`` moves the zone's far end, its sides and the
    edges of the camera's view outwards (positive) or inwards, in metres
    (degrees for the view: 25 times ``grow``)."""
    fc = f - CAM_AHEAD
    dist = np.hypot(fc, s)
    in_view = fc > 0 and abs(np.arctan2(s, fc)) - np.arcsin(min(1.0, r / dist)) < CAM_HFOV / 2 + np.radians(25 * grow)
    return bool(WHEELBASE < f < WHEELBASE + RANGE + grow and abs(s) < LANE + r + grow / 4 and in_view)


def truth(track, x, y, th):
    """The answer a perfect perception gives at the pose: ``blocked`` (1 when
    an obstacle stands ahead, its centre between the front axle and
    :data:`RANGE` beyond it, within the zone :data:`LANE` to either side and in
    the colour camera's view), and the nearest such obstacle's ``forward`` and
    ``sideways`` distance from the rear axle in the car's frame (m) and its
    ``kind``; nan and "" when free. ``edge`` is True when an obstacle stands
    within 0.2 m of the zone's far end or within 5 degrees of the view's edge,
    where either answer is fair: such frames are left out of the counts of
    missed obstacles and false stops."""
    best, edge = None, False
    for (ox, oy, r, kind) in track.obstacles:
        dx, dy = ox - x, oy - y
        f, s = dx * np.cos(th) + dy * np.sin(th), -dx * np.sin(th) + dy * np.cos(th)
        if _in_zone(f, s, r):
            if best is None or f < best[0]:
                best = (f, s, kind)
        if _in_zone(f, s, r, 0.2) != _in_zone(f, s, r, -0.2):
            edge = True
    if best is None:
        return dict(blocked=0, forward=np.nan, sideways=np.nan, kind="", edge=edge)
    return dict(blocked=1, forward=float(best[0]), sideways=float(best[1]), kind=best[2], edge=edge)


def oracle(track):
    """Perfect perception for :func:`drive`: ``perceive(car, sensors) -> (blocked, forward, sideways)``."""
    def perceive(car, sensors):
        t = truth(track, car.x, car.y, car.th)
        return t["blocked"], t["forward"], t["sideways"]
    return perceive


# ====================================================================== the conventional rules, one per sensor
R_TYP = 0.085             #: m, a typical obstacle radius: the rules place the centre this far behind the nearest surface
#: where the camera rule looks: the floor of the lane, from the nearest row to RANGE + 0.3 m ahead of the rear axle
LANE_MASK = _FLOOR & (FLOOR_F < RANGE + WHEELBASE + 0.3) & (np.abs(FLOOR_L) < LANE + 0.05)


def camera_rule(frame, threshold=0.15, share=0.004):
    """Ex11's dark-pixel rule on the R2's colour frame ``(CAM_H, CAM_W, 3)``:
    blocked when more than ``share`` of the lane's floor pixels are darker
    than the frame's floor (its median) by ``threshold``; where: the dark
    pixel lowest in the frame is the obstacle's foot, read off the flat floor,
    and the centre is :data:`R_TYP` further. Returns ``(blocked, forward,
    sideways)``."""
    g = np.asarray(frame, np.float32).mean(axis=-1) / 255
    floor = np.median(g[_FLOOR])
    dark = LANE_MASK & (g < floor - threshold)
    if dark.mean() / LANE_MASK.mean() <= share:
        return 0, np.nan, np.nan
    rows = np.where(dark.any(axis=1))[0]
    r = rows.max()                                             # the lowest dark row: the foot, nearest the car
    cols = np.where(dark[r])[0]
    cc = int(np.round(np.median(cols)))
    return 1, float(FLOOR_F[r, cc] + R_TYP), float(FLOOR_L[r, cc])


def lidar_rule(scan):
    """Blocked when a return falls in the lane ahead within range; where: the
    nearest such return, :data:`R_TYP` further."""
    a = np.radians(np.arange(LIDAR_BEAMS) * 360.0 / LIDAR_BEAMS)
    f, s = scan * np.cos(a) + LIDAR_AHEAD, scan * np.sin(a)
    hit = (scan > 0) & (f > WHEELBASE) & (f < RANGE + WHEELBASE) & (np.abs(s) < LANE + 0.05)
    if not hit.any():
        return 0, np.nan, np.nan
    i = np.argmin(np.where(hit, scan, np.inf))
    k = (scan[i] + R_TYP) / scan[i]
    return 1, float((f[i] - LIDAR_AHEAD) * k + LIDAR_AHEAD), float(s[i] * k)


def depth_points(depth):
    """The depth image as points in the car's frame: forward, left, up (m), and a mask of valid pixels."""
    z = np.asarray(depth, float)
    t = z / F_DEP
    return _DF * t + CAM_AHEAD, _DL * t, CAM_HEIGHT + _DU * t, z > 0


def depth_rule(depth, above=0.04, count=6):
    """Blocked when at least ``count`` valid pixels stand more than ``above``
    over the floor, in the lane ahead within range; where: the nearest of them,
    :data:`R_TYP` further. Returns ``(blocked, forward, sideways)``."""
    f, l, u, ok = depth_points(depth)
    m = ok & (u > above) & (u < 0.6) & (f > WHEELBASE) & (f < RANGE + WHEELBASE) & (np.abs(l) < LANE + 0.05)
    if m.sum() < count:
        return 0, np.nan, np.nan
    i = np.argmin(np.where(m, f, np.inf))
    return 1, float(f.flat[i] + R_TYP), float(l.flat[i])


def rules_or(*answers):
    """Several rules' answers by OR: blocked if any says so; where from the
    nearest of those that say so."""
    hits = [a for a in answers if a[0]]
    if not hits:
        return 0, np.nan, np.nan
    return min(hits, key=lambda a: a[1])


def conventional(names=("camera",)):
    """The conventional perception for :func:`drive`: the rules ``names``
    ("camera", "depth", "lidar") by OR."""
    def perceive(car, sensors):
        out = []
        for n in names:
            if n == "camera":
                out.append(camera_rule(sensors["rgb"]))
            elif n == "depth":
                out.append(depth_rule(sensors["depth"]))
            else:
                out.append(lidar_rule(sensors["scan"]))
        return rules_or(*out)
    return perceive


# ====================================================================== planning in the track's frame
AIM_F = 0.55              #: m: the potential field is read this far ahead of the rear axle
K_FIELD = 0.03            #: the field's strength, m (Ex11's)
K_WALL = 0.01             #: the push from the taped edges, m (ASSUMED)
R_ASSUMED = 0.10          #: m: the planners take every obstacle this large (the largest on a track)
D_LIMIT = W_HALF - HALF_W - 0.03   #: m: the farthest from the centre line the car's centre may aim
A_LAT = 1.0               #: m/s^2: the sideways acceleration allowed in a bend; it sets the speed there (ASSUMED)


def _pursuit(car, track, s, d_ref, speed):
    """Pure pursuit: the wheel angle that points the car at the point
    ``d_ref`` beside the centre line, one look-ahead further along."""
    look = max(0.45, 0.40 * max(car.v, speed))
    tx, ty = track.point(s + look, d_ref)
    alpha = np.arctan2(ty - car.y, tx - car.x) - car.th
    return float(np.arctan(2 * WHEELBASE * np.sin(alpha) / max(np.hypot(tx - car.x, ty - car.y), 1e-3)))


def world(car, forward, sideways):
    """An answer in the car's frame as a floor point."""
    c, s = np.cos(car.th), np.sin(car.th)
    return car.x + forward * c - sideways * s, car.y + forward * s + sideways * c


def _ahead(track, s, so):
    """Signed distance along the track from ``s`` (unwrapped) to the station ``so``."""
    return (so - s + track.length / 2) % track.length - track.length / 2


class SteerAround:
    """Ex11's steer-around rule on the track: when blocked, aim at a lane
    :data:`OFFSET` beside the obstacle on the side the car is already on (the
    other side when the track leaves no room there), slowed to 60 %, until the
    car's rear is past it, then back to the centre line."""

    OFFSET = 0.40

    def __init__(self):
        self.d_ref, self.hold = 0.0, -1.0

    def __call__(self, car, track, s, d, answer, speed):
        b, f, sd = answer
        so = do = None
        if b and np.isfinite(f):
            so, do = track.frenet(*world(car, f, sd))
            so = s + _ahead(track, s, float(so))
            if s < self.hold and so > self.hold:                   # a farther obstacle while passing one: it waits
                b = 0
        if b and np.isfinite(f):
            first = 1.0 if d >= do else -1.0                       # the side the car is already on, as Ex11's rule
            for side in (first, -first):                           # ... unless the track leaves no room there
                self.d_ref = float(np.clip(do + side * self.OFFSET, -D_LIMIT, D_LIMIT))
                if abs(self.d_ref - do) >= R_ASSUMED + HALF_W + 0.08:
                    break
            self.hold = so + R_ASSUMED + REAR + 0.10
        if s >= self.hold:
            self.d_ref = 0.0
        return (0.6 * speed if s < self.hold else speed), self.d_ref


class FieldAround:
    """Ex11's potential field on the track: an obstacle reported twice
    within :data:`CONFIRM` decisions is remembered at the mean of its sightings (reports nearer than :data:`NEAR`
    are not trusted and left out; taken :data:`R_ASSUMED` large, with
    :data:`MARGIN` for the perception's position error). The car aims at the
    lowest point across the track of ``U(d) = d**2 + sum K/gap_i(d) +
    K_w/edge gap(d)``, ``gap_i`` the sideways gap between the car's side and
    obstacle i, summed over the obstacles ahead until the car's rear is past
    them; each passed on the side chosen when the obstacle is first met (the one that needs the smaller swerve from
    where the car is, of those with room) and kept while it has room, so that a
    position that wavers from one decision to the next cannot swing the car
    across its path. Full speed,
    unless an obstacle is within :data:`SLOW` and the car is not yet clear of it."""

    MERGE = 0.40          #: m: a report this close to a remembered obstacle is the same obstacle (they stand 1.6 m apart or more)
    CONFIRM = 10          #: decisions within which a first report must be seen again to count (1 s at 10 a second; 0: no confirmation)
    RECHOOSE = 1.2        #: m: farther than this, a side left with no room is chosen again; nearer, the car keeps it
    NEAR = 0.70           #: m ahead of the rear axle: nearer, an obstacle is partly out of the frame and its position is not trusted
    MARGIN = 0.10         #: m: kept clear beyond the obstacle's assumed radius and the car's half-width
    SLOW, SLOW_FACTOR = 1.0, 0.5   #: an obstacle closer than SLOW m that the car is not yet clear of: speed times SLOW_FACTOR

    def __init__(self, k=K_FIELD):
        self.seen, self.candidates, self.k = [], [], k

    def __call__(self, car, track, s, d, answer, speed):
        b, f, sd = answer
        if b and np.isfinite(f) and f > self.NEAR:
            X, Y = world(car, f, sd)
            near = [i for i, o in enumerate(self.seen) if np.hypot(X - o[0], Y - o[1]) < self.MERGE]
            if near:                                               # seen again: the mean of every sighting
                o = self.seen[near[0]]
                o[3] += 1
                o[0] += (X - o[0]) / o[3]; o[1] += (Y - o[1]) / o[3]
            else:
                cand = [c for c in self.candidates if np.hypot(X - c[0], Y - c[1]) < self.MERGE]
                if self.CONFIRM == 0:                              # no confirmation asked: remembered at the first report
                    self.seen.append([X, Y, 0.0, 1])
                elif cand:                                         # a second sighting confirms it: now an obstacle
                    c = cand[0]
                    self.seen.append([(X + c[0]) / 2, (Y + c[1]) / 2, 0.0, 2])   # x, y, the side to pass on (chosen when first met), sightings
                    self.candidates.remove(c)
                else:
                    self.candidates.append([X, Y, 0])
        self.candidates = [[cx, cy, age + 1] for cx, cy, age in self.candidates if age < self.CONFIRM]   # a second report within CONFIRM decisions
        ds = np.linspace(-D_LIMIT, D_LIMIT, 161)
        U = ds ** 2 + K_WALL / np.maximum(W_HALF - HALF_W - np.abs(ds), 0.01)
        c = R_ASSUMED + HALF_W + self.MARGIN
        allowed = np.ones(ds.shape, bool)
        active = []
        for o in self.seen:
            so, do = track.frenet(o[0], o[1])
            ahead = _ahead(track, s, float(so))
            if -(R_ASSUMED + REAR + 0.10) < ahead < RANGE + WHEELBASE + 0.5:
                active.append((ahead, float(do), o))
        for ahead, do, o in sorted(active, key=lambda a: a[0]):    # every obstacle until the car's rear is past it
            gap = np.abs(ds - do) - c
            U = U + self.k / np.maximum(gap, 0.005) + 100.0 * np.maximum(-gap, 0.0)   # inside the margin: pushed out, steeply
            if o[2] != 0.0 and ahead > self.RECHOOSE and not (allowed & ((ds - do) * o[2] >= c)).any():
                o[2] = 0.0                                         # its position has moved and the side has no room left: choose again
            if o[2] == 0.0:                                        # first met: the side that needs the smaller swerve,
                options = []                                       # of those with room, beside those already chosen
                for side in (1.0, -1.0):
                    ok = allowed & ((ds - do) * side >= c)
                    if ok.any():
                        options.append((np.abs(ds[ok] - d).min(), side))
                o[2] = min(options)[1] if options else (1.0 if do < 0 else -1.0)
            ok = allowed & ((ds - do) * o[2] > 0)                  # stay on the chosen side
            allowed = ok if ok.any() else allowed
            if 0 < ahead < self.SLOW and abs(d - do) < c:
                speed = self.SLOW_FACTOR * speed                   # close, and not yet clear of it: slow down to gain time to steer
        U = np.where(allowed, U, np.inf)
        return speed, float(ds[np.argmin(U)])


# ====================================================================== one lap in closed loop
def bend_speed(track, s, a_lat=A_LAT, look=1.2):
    """The largest speed for the bends of the next ``look`` metres: the
    sideways acceleration v^2 kappa stays under ``a_lat``."""
    k = np.abs(np.interp(np.mod(s + np.linspace(0, look, 13), track.length), track.s, track.kappa)).max()
    return float(np.sqrt(a_lat / max(k, 1e-6)))


def drive(track, perceive, planner=None, speed=1.0, rate_hz=RATE_HZ, dt=2e-3, seed=0, t_max=None, record=False,
          need=("rgb", "depth", "scan"), a_lat=A_LAT):
    """Drive one lap of ``track`` from A (the start line) to B (the same line,
    one lap on). At every decision ``perceive(car, sensors)`` returns
    ``(blocked, forward, sideways)``; ``sensors`` holds the latest colour frame
    and depth image (30 a second) and LiDAR scan (5.5 a second), as listed in
    ``need``. ``planner`` (:class:`SteerAround` by default, or
    :class:`FieldAround`) turns the answer into a speed and an offset from the
    centre line; pure pursuit steers there; in a bend the speed is held under
    :func:`bend_speed` (``a_lat=None`` lifts it). Computing a decision takes one
    period: the wheels follow the command from the decision before. A car that
    touches an obstacle or leaves the track (its centre 0.1 m beyond an edge)
    stops the run. Returns ``dict(time, finished, collided, left, clearance,
    decisions, path)``, and with ``record`` the poses every 0.04 s."""
    planner = SteerAround() if planner is None else planner
    car = R2(track, 0.0, 0.0, 0.0, speed=speed, seed=seed)
    t_max = 3.0 * track.length / speed + 5.0 if t_max is None else t_max
    t, next_dec, next_scan, next_rec = 0.0, 0.0, 0.0, 0.0
    sensors, pending, clear, travelled = {}, (speed, 0.0), np.inf, 0.0
    s_prev = 0.0
    path, poses, seen_log = [(car.x, car.y)], [], []
    collided = left = False
    n_dec, seen = 0, 0
    while t < t_max:
        if record and t >= next_rec - 1e-12:
            poses.append((t, car.x, car.y, car.th, car.delta, car.v, seen))
            next_rec += 0.04
        if t >= next_scan - 1e-12 and "scan" in need:
            sensors["scan"], next_scan = car.scan(), next_scan + 1 / LIDAR_HZ
        if t >= next_dec - 1e-12:
            car.cmd(*pending)                                      # the command computed from the decision before
            if "rgb" in need:
                sensors["rgb"] = car.rgb()
            if "depth" in need:
                sensors["depth"] = car.depth()
            s_now, d_now = track.frenet(car.x, car.y)
            answer = perceive(car, sensors)
            seen = int(answer[0])
            v_cmd, d_ref = planner(car, track, travelled, float(d_now), answer, speed)
            if a_lat:
                v_cmd = min(v_cmd, bend_speed(track, travelled, a_lat))
            pending = (v_cmd, _pursuit(car, track, travelled, d_ref, v_cmd))
            next_dec += 1.0 / rate_hz
            n_dec += 1
        car.step(dt)
        t += dt
        s_now, d_now = track.frenet(car.x, car.y)
        ds = (float(s_now) - s_prev + track.length / 2) % track.length - track.length / 2
        travelled += ds; s_prev = float(s_now)
        g = car.gap()
        clear = min(clear, g)
        if g < 0:
            collided = True
            break
        if abs(float(d_now)) > W_HALF + 0.10:
            left = True
            break
        if travelled >= track.length:
            t -= (travelled - track.length) / max(car.v, 1e-6)       # back to the moment the line was crossed
            break
        if abs(car.x - path[-1][0]) + abs(car.y - path[-1][1]) > 0.03:
            path.append((car.x, car.y))
    finished = travelled >= track.length and not collided and not left
    out = dict(time=t if finished else np.inf, finished=finished, collided=collided, left=left,
               clearance=clear, decisions=n_dec, path=np.array(path))
    if record:
        poses.append((t, car.x, car.y, car.th, car.delta, car.v, seen))
        out["poses"] = np.array(poses)
    return out


# ====================================================================== labelled data
def samples(seeds, n_per_track, levels=(0,), seed=0, need=("rgb", "depth", "scan"), p_ahead=0.65, kinds=(0.35, 0.25, 0.20, 0.20)):
    """Labelled sensor data on the tracks ``seeds`` at the noise ``levels``:
    ``n_per_track`` poses per track and level, on the track and off its centre
    line, turned, at a speed from 0 to 1.8 m/s (for the motion blur); with
    probability ``p_ahead`` an obstacle stands 0.2 to 1.8 m ahead. Returns a
    dict of arrays: ``rgb`` (n, 3, CAM_H, CAM_W) uint8, ``depth`` (n, DEP_H,
    DEP_W) float32, ``scan`` (n, 360) float32, as asked in ``need``;
    ``blocked``, ``where`` (forward, sideways; nan when free), ``kind``,
    ``edge`` (an obstacle near the zone's edge: see :func:`truth`), ``level``, ``track``, ``shadow`` (a shadow in the lane ahead with no
    obstacle) and ``sun`` (a sun patch in the lane ahead)."""
    rng = np.random.default_rng([seed, 7])
    out = {k: [] for k in ("blocked", "where", "kind", "edge", "level", "track", "shadow", "sun", "pose") + tuple(need)}
    for sd in seeds:
        for lv in levels:
            tr = Track(sd, lv, kinds=kinds)
            for _ in range(n_per_track):
                if rng.random() < p_ahead and tr.obstacle_s:
                    so, do = tr.obstacle_s[int(rng.integers(len(tr.obstacle_s)))]
                    s = so - WHEELBASE - rng.uniform(0.2, 1.8)
                else:
                    s = rng.uniform(0, tr.length)
                d = rng.uniform(-0.35, 0.35)
                x, y = tr.point(s, d)
                th = tr.heading(s) + rng.uniform(-0.3, 0.3)
                v = rng.uniform(0.0, V_MAX)
                r = np.random.default_rng([sd, lv, int(rng.integers(1 << 30))])
                if "rgb" in need:
                    out["rgb"].append(render_rgb(tr, x, y, th, r, v).transpose(2, 0, 1))
                if "depth" in need:
                    out["depth"].append(render_depth(tr, x, y, th, r))
                if "scan" in need:
                    out["scan"].append(render_scan(tr, x, y, th, r))
                t = truth(tr, x, y, th)
                out["blocked"].append(t["blocked"]); out["where"].append((t["forward"], t["sideways"])); out["kind"].append(t["kind"])
                out["edge"].append(t["edge"])
                out["level"].append(lv); out["track"].append(sd); out["pose"].append((x, y, th, v))
                rel = [((a - x) * np.cos(th) + (b - y) * np.sin(th), -(a - x) * np.sin(th) + (b - y) * np.cos(th)) for a, b, _ in tr.shadows]
                out["shadow"].append(bool(any(0 < f < RANGE + 0.3 and abs(q) < LANE for f, q in rel)) and not t["blocked"])
                rel = [((a - x) * np.cos(th) + (b - y) * np.sin(th), -(a - x) * np.sin(th) + (b - y) * np.cos(th)) for a, b, _ in tr.sun]
                out["sun"].append(bool(any(0 < f < RANGE + 0.3 and abs(q) < LANE for f, q in rel)))
    D = {}
    for k, v in out.items():
        if k == "kind":
            D[k] = np.array(v, dtype="<U5")
        elif k in ("rgb",):
            D[k] = np.array(v, np.uint8)
        elif k in ("depth", "scan", "where"):
            D[k] = np.array(v, np.float32)
        elif k == "pose":
            D[k] = np.array(v)
        else:
            D[k] = np.array(v)
    return D


def rates(blocked_pred, D, where_pred=None):
    """Scores of a perception on labelled data ``D``: missed obstacles per
    kind, false stops on free poses and on shadow poses, the share right -
    all on the frames whose answer is clear (``edge`` frames left out) - and
    with ``where_pred`` the mean position error per kind, cm, on every frame
    both call blocked."""
    clear = ~D["edge"] if "edge" in D else np.ones(len(D["blocked"]), bool)
    b = D["blocked"] == 1
    out = {f"missed {k}": float(np.mean(blocked_pred[clear & b & (D["kind"] == k)] == 0)) if np.any(clear & b & (D["kind"] == k)) else np.nan
           for k in KIND_NAMES}
    out["false stops"] = float(np.mean(blocked_pred[clear & ~b] == 1))
    out["false stops, shadow"] = float(np.mean(blocked_pred[clear & D["shadow"]] == 1)) if np.any(clear & D["shadow"]) else np.nan
    out["right"] = float(np.mean(blocked_pred[clear] == D["blocked"][clear]))
    if where_pred is not None:
        both = b & (blocked_pred == 1)
        for k in KIND_NAMES:
            m = both & (D["kind"] == k)
            out[f"where {k} [cm]"] = float(100 * np.mean(np.hypot(*(where_pred[m] - D["where"][m]).T))) if m.any() else np.nan
    return out


# ====================================================================== pictures
def draw_track(track, ax=None, paths=None, dark=False, cars=()):
    """The track from above: the taped edges, the start line A = B, the
    obstacles (colour by kind, the car's half-width round each), the shadows
    and sun patches, and, if given, ``paths`` ``{label: (n, 2)}`` and car poses."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))
    fg = "white" if dark else "black"
    for side in (-1, 1):
        ex, ey = track.point(track.s, side * W_HALF)
        ax.plot(np.r_[ex, ex[:1]], np.r_[ey, ey[:1]], color="0.55", lw=2)
    ax.plot(track.x, track.y, color="0.8", lw=0.6, ls="--")
    ax0, ay0 = track.point(0.0, -W_HALF); ax1, ay1 = track.point(0.0, W_HALF)
    ax.plot([ax0, ax1], [ay0, ay1], color="tab:blue", lw=3)
    ax.text(*track.point(-0.25, -W_HALF - 0.25), "A = B", color="tab:blue", ha="center", fontsize=10, weight="bold")
    col = {"box": "#222222", "grey": "#9A948A", "low": "saddlebrown", "black": "#000000"}
    for (sx, sy, sz) in track.shadows:
        ax.add_patch(plt.Circle((sx, sy), sz, color="0.3", alpha=0.25, lw=0))
    for (gx, gy, gz) in track.sun:
        ax.add_patch(plt.Circle((gx, gy), gz * 1.3, color="gold", alpha=0.5, lw=0))
    for (ox, oy, r, kind) in track.obstacles:
        ax.add_patch(plt.Circle((ox, oy), r + HALF_W, color="tab:red", alpha=0.12, lw=0))
        ax.add_patch(plt.Circle((ox, oy), r, color=col[kind], ec="tab:red" if kind == "black" else None, lw=1))
    for label, p in (paths or {}).items():
        p = np.asarray(p)
        ax.plot(p[:, 0], p[:, 1], lw=1.4, label=label)
    for (x, y, th) in cars:
        ax.annotate("", (x + 0.35 * np.cos(th), y + 0.35 * np.sin(th)), (x, y), arrowprops=dict(arrowstyle="->", color=fg, lw=1.8))
    if paths:
        ax.legend(fontsize=8, loc="center")
    ax.set_aspect("equal"); ax.set_xlabel("x [m]", color=fg); ax.set_ylabel("y [m]", color=fg)
    ax.tick_params(colors=fg)
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], marker="o", ls="", color=col[k], label=k) for k in KIND_NAMES]
    ax.legend(handles=hs + ax.get_legend_handles_labels()[0], fontsize=7, loc="center", ncol=2)
    return ax


def show_sensors(track, x, y, th, speed=1.0, seed=0, axes=None, title=True):
    """The three sensors at one pose side by side: the colour frame, the
    depth image (black: no reading) and the LiDAR scan from above, with the
    car's lane and :data:`RANGE` drawn on it."""
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(seed)
    if axes is None:
        fig = plt.figure(figsize=(13, 3.2))
        axes = [fig.add_subplot(1, 3, 1), fig.add_subplot(1, 3, 2), fig.add_subplot(1, 3, 3)]
    a0, a1, a2 = axes
    a0.imshow(render_rgb(track, x, y, th, rng, speed)); a0.axis("off")
    dep = render_depth(track, x, y, th, rng)
    im = a1.imshow(np.where(dep > 0, dep, np.nan), cmap="viridis_r", vmin=0, vmax=4); a1.axis("off")
    a1.set_facecolor("black")
    sc = render_scan(track, x, y, th, rng)
    a = np.radians(np.arange(LIDAR_BEAMS))
    ok = sc > 0
    a2.scatter(sc[ok] * np.sin(a[ok]) * -1, sc[ok] * np.cos(a[ok]) + LIDAR_AHEAD, s=3, color="tab:red")
    a2.add_patch(plt.Rectangle((-LANE, WHEELBASE), 2 * LANE, RANGE, fill=False, ls="--", color="0.4"))
    a2.add_patch(plt.Rectangle((-HALF_W, -REAR), 2 * HALF_W, REAR + WHEELBASE + FRONT, color="tab:blue", alpha=0.6))
    a2.set_xlim(-2, 2); a2.set_ylim(-0.6, 3); a2.set_aspect("equal")
    a2.set_xlabel("to the right [m]"); a2.set_ylabel("ahead [m]")
    if title:
        a0.set_title(f"colour frame {CAM_W} x {CAM_H}", fontsize=9)
        a1.set_title(f"depth image {DEP_W} x {DEP_H} (white: no reading)", fontsize=9)
        a2.set_title("LiDAR scan; dashed: the lane within RANGE", fontsize=9)
    return axes


# ====================================================================== the R2 in 3-D
def _r2_body():
    """Everything but the wheels, in the car's frame (x forward from the rear
    axle, y left, z up), after the vendor's photograph: the aluminium chassis,
    the black top deck on standoffs, the Jetson with its fan, the 7-inch
    screen, the LiDAR on the top deck and the Astra Pro Plus at the front. A
    DRAWING to the twin's assumed numbers; only the depth camera's size
    (165 x 48 x 40 mm) is the vendor's."""
    _box, _tube, _prism = pb._box, pb._tube, pb._prism
    P = []
    L0, L1 = -REAR, WHEELBASE + FRONT
    P.append((_box(L0, L1, -0.075, 0.075, 0.040, 0.046), "#B8BCC2"))                     # the aluminium chassis plate
    P.append((_box(L0 + 0.01, L1 - 0.03, -0.065, 0.065, 0.046, 0.075), "#2B2B2E"))       # motors, battery and the expansion board
    P.append((_box(0.02, 0.17, -0.055, 0.055, 0.075, 0.080), "#123B7A"))                 # the expansion board, blue
    for (sx, sy) in ((L0 + 0.02, -0.06), (L0 + 0.02, 0.06), (L1 - 0.05, -0.06), (L1 - 0.05, 0.06)):
        P.append((_tube((sx, sy, 0.075), (sx, sy, 0.135), 0.004, 8), "#C9A227"))        # brass standoffs
    P.append((_box(L0, L1 - 0.02, -0.085, 0.085, 0.135, 0.141), "#151515"))              # the top deck
    P.append((_box(0.00, 0.10, -0.05, 0.05, 0.141, 0.150), "#0E3B1E"))                   # the Jetson Orin Nano module
    for k in range(7):                                                                   # its heat sink
        xx = 0.008 + k * 0.012
        P.append((_box(xx, xx + 0.004, -0.045, 0.045, 0.150, 0.170), "#1B1B1B"))
    fa = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    P.append(([np.column_stack([0.05 + 0.03 * np.cos(fa), 0.03 * np.sin(fa), np.full(20, 0.171)])], "#2C2C2C"))
    # the 7-inch screen, standing at the back, leaning back
    sc = [np.array([[L0 + 0.02, -0.085, 0.141], [L0 + 0.02, 0.085, 0.141], [L0 - 0.01, 0.085, 0.245], [L0 - 0.01, -0.085, 0.245]])]
    P.append((sc, "#0A0A0A"))
    P.append(([sc[0] + np.array([0.0015, 0, 0]) * 1 + np.array([[0.002, 0.008, 0.006], [0.002, -0.008, 0.006], [0.002, -0.008, -0.006], [0.002, 0.008, -0.006]])], "#24476B"))
    # the LiDAR (RPLIDAR A1) on the top deck: a dark base and its round rotating head, the scan plane at LIDAR_HEIGHT
    lx = LIDAR_AHEAD
    P.append((_box(lx - 0.035, lx + 0.035, -0.035, 0.035, 0.141, 0.160), "#1A1A1A"))
    P.append((_tube((lx, 0, 0.160), (lx, 0, LIDAR_HEIGHT + 0.012), 0.033, 22), "#222222"))
    P.append((_tube((lx, 0, LIDAR_HEIGHT - 0.008), (lx, 0, LIDAR_HEIGHT + 0.008), 0.0335, 22, caps=False), "#555A60"))
    # the Astra Pro Plus at the front, on a bracket: 165 x 48 x 40 mm (VENDOR)
    cx = CAM_AHEAD
    P.append((_box(cx - 0.05, cx - 0.01, -0.012, 0.012, 0.046, CAM_HEIGHT - 0.024), "#8E949B"))   # the bracket
    P.append((_box(cx - 0.040, cx, -0.0825, 0.0825, CAM_HEIGHT - 0.024, CAM_HEIGHT + 0.024), "#141414"))
    for yy, col in ((-0.045, "#3A3F47"), (0.0, "#25334A"), (0.04, "#3A3F47")):       # the projector, the colour lens, the infrared camera
        P.append((_tube((cx, yy, CAM_HEIGHT), (cx + 0.004, yy, CAM_HEIGHT), 0.008, 14), col))
    P.append((_box(L1 - 0.02, L1 + 0.01, -0.08, 0.08, 0.02, 0.05), "#202020"))           # the front bumper
    return P


def _r2_wheel():
    """One 65 mm wheel at the origin, its axle along y, outer face at +y."""
    R, W, n = WHEEL_D / 2, 0.028, 26
    out = [(pb._tube((0, -W / 2, 0), (0, W / 2, 0), R, n, caps=False), "#161616")]
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = lambda r, y: np.column_stack([r * np.sin(a), np.full(n, y), r * np.cos(a)])
    for y, sgn in ((W / 2, 1), (-W / 2, -1)):
        o, i_ = ring(R, y), ring(0.020, y)
        out.append(([np.array([o[k], o[(k + 1) % n], i_[(k + 1) % n], i_[k]])[::sgn] for k in range(n)], "#1E1E1E"))
    y = W / 2 + 0.0008
    spokes = []
    for k in range(6):
        b = 2 * np.pi * k / 6
        c0, c1 = np.array([np.sin(b), 0, np.cos(b)]), np.array([np.cos(b), 0, -np.sin(b)])
        spokes.append(np.array([0.004 * c0 - 0.004 * c1, 0.020 * c0 - 0.003 * c1, 0.020 * c0 + 0.003 * c1, 0.004 * c0 + 0.004 * c1]) + [0, y, 0])
    out.append((spokes, "#D6D9DD"))
    out.append(([ring(0.020, y - 0.001)], "#3A3A3A"))
    return out


_BODY = _WHEEL = None


def r2_model(x, y, th, delta=0.0, spin=0.0):
    """The ROSMASTER R2 drawn in 3-D at the pose: ``(faces, colour)`` pairs.
    The rear axle is at (x, y); ``delta`` turns the front wheels, ``spin``
    (rad) turns all four. A drawing, not a measurement."""
    global _BODY, _WHEEL
    if _BODY is None:
        _BODY, _WHEEL = _r2_body(), _r2_wheel()
    Rz, t = pb._rot("z", th), np.array([x, y, 0.0])
    out = [([f @ Rz.T + t for f in faces], col) for faces, col in _BODY]
    for ax_, steer in ((0.0, 0.0), (WHEELBASE, delta)):
        for sgn in (1, -1):
            M = pb._rot("z", steer) @ (np.eye(3) if sgn > 0 else pb._rot("z", np.pi)) @ pb._rot("y", spin * sgn)
            c = np.array([ax_, sgn * TRACK_W / 2, WHEEL_D / 2])
            for faces, col in _WHEEL:
                out.append(([(f @ M.T + c) @ Rz.T + t for f in faces], col))
    return out


def view3d(track, x, y, th, ax=None, delta=0.0, spin=0.0, trail=None, title="", beams=True, seed=0):
    """The R2 in 3-D at the pose on ``track``, seen from behind and a little
    to its left: the floor, the taped edges, shadows and sun, the obstacles,
    and, with ``beams``, the LiDAR's scan plane as red rays to what it hits.
    Draws into a 3-D ``ax`` (``projection="3d", computed_zorder=False``) or a
    new figure; returns the axes. The film (:func:`film`) is made of these."""
    import warnings
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
    from matplotlib.colors import LightSource
    warnings.filterwarnings("ignore", "overflow encountered", RuntimeWarning, "mpl_toolkits")
    if ax is None:
        ax = plt.figure(figsize=(5.5, 5)).add_axes([0, 0, 1, 1], projection="3d", computed_zorder=False)
    hx, hy = np.cos(th), np.sin(th)
    cx, cy = x + 0.45 * hx, y + 0.45 * hy
    R_ = 0.8
    ax.cla()
    ax.set_facecolor("#DADDE2")
    corners = [(-0.7, -3.0), (6.0, -3.0), (6.0, 3.0), (-0.7, 3.0)]
    fl = [[(x + a * hx - b * hy, y + a * hy + b * hx, 0) for a, b in corners]]
    ax.add_collection3d(Poly3DCollection(fl, facecolor="#BDB8AE", edgecolor="none", zorder=0), autolim=False)
    s0, _ = track.frenet(x, y)
    ss = float(s0) + np.arange(-0.8, 6.0, 0.05)
    flat = []
    for side in (-1, 1):                                         # the taped edges, as strips near the car
        a_ = np.column_stack(track.point(ss, side * (W_HALF - EDGE_HALF)))
        b_ = np.column_stack(track.point(ss, side * (W_HALF + EDGE_HALF)))
        for i in range(len(ss) - 1):
            flat.append([(*a_[i], 4e-4), (*a_[i + 1], 4e-4), (*b_[i + 1], 4e-4), (*b_[i], 4e-4)])
    l0, l1 = np.array(track.point(0.0, -W_HALF)), np.array(track.point(0.0, W_HALF))
    n0 = np.array([np.cos(track.heading(0.0)), np.sin(track.heading(0.0))]) * 0.03
    flat.append([(*(l0 - n0), 5e-4), (*(l0 + n0), 5e-4), (*(l1 + n0), 5e-4), (*(l1 - n0), 5e-4)])
    ax.add_collection3d(Poly3DCollection(flat, facecolor="#F7F7F2", edgecolor="none", zorder=1), autolim=False)
    for (sx, sy, sz) in track.shadows:
        ax.add_collection3d(Poly3DCollection(pb._ellipse(sx, sy, sz, sz * 0.8, 0.0), facecolor=(0, 0, 0, 0.28), edgecolor="none", zorder=1), autolim=False)
    for (gx, gy, gz) in track.sun:
        ax.add_collection3d(Poly3DCollection(pb._ellipse(gx, gy, gz * 1.6, gz * 1.2, 0.0), facecolor=(1, 1, 0.9, 0.85), edgecolor="none", zorder=1), autolim=False)
    ax.add_collection3d(Poly3DCollection(pb._ellipse(x + 0.12 * hx + 0.02, y + 0.12 * hy - 0.02, 0.24, 0.14, th), facecolor=(0, 0, 0, 0.22), edgecolor="none", zorder=1), autolim=False)
    if trail is not None and len(trail):
        tr = np.asarray(trail)
        ax.plot(tr[:, 0], tr[:, 1], 0.001, color="#FF8F00", lw=1.5, zorder=2)
    if beams:
        sc = render_scan(track, x, y, th, np.random.default_rng(seed))
        a = th + np.radians(np.arange(LIDAR_BEAMS))
        ox, oy = x + hx * LIDAR_AHEAD, y + hy * LIDAR_AHEAD
        segs = [[(ox, oy, LIDAR_HEIGHT), (ox + r * np.cos(b), oy + r * np.sin(b), LIDAR_HEIGHT)]
                for r, b in zip(sc[::3], a[::3]) if 0 < r < 2.5]
        ax.add_collection3d(Line3DCollection(segs, colors=(0.9, 0.1, 0.1, 0.35), linewidths=0.6, zorder=4), autolim=False)
    faces, cols = [], []
    for (ox_, oy_, r, k) in track.obstacles:
        if np.hypot(ox_ - x, oy_ - y) < 6 and (ox_ - x) * hx + (oy_ - y) * hy > -0.2:   # not those behind the car: they would hide it
            f = pb._tube((ox_, oy_, 0), (ox_, oy_, KINDS[k]["height"]), r, 24)
            faces += list(f); cols += [{"box": "#262626", "grey": "#8A8478", "low": "#3E2723", "black": "#050505"}[k]] * len(f)
    for f, col in r2_model(x, y, th, delta, spin):
        faces += list(f); cols += [col] * len(f)
    ax.add_collection3d(Poly3DCollection(faces, facecolors=cols, edgecolor="none", shade=True,
                                         lightsource=LightSource(azdeg=225, altdeg=50), zorder=3), autolim=False)
    ax.set_xlim(cx - R_, cx + R_); ax.set_ylim(cy - R_, cy + R_); ax.set_zlim(0, 0.5)
    ax.set_box_aspect((2 * R_, 2 * R_, 0.5), zoom=1.6)
    ax.set_proj_type("persp", focal_length=0.35)
    ax.view_init(elev=18, azim=np.degrees(th) + 180 + 25)
    ax.set_axis_off()
    if title:
        ax.text2D(0.03, 0.95, title, transform=ax.transAxes, fontsize=10, weight="bold")
    return ax


def film(run, track, title="", fps=10, every=1, start=0.0, seconds=None):
    """A film of a lap of :func:`drive` (run with ``record=True``): the R2 in
    3-D with its LiDAR's beams, beside its colour frame, its depth image and
    the track from above. ``every`` keeps every n-th recorded pose (0.04 s
    apart), from ``start`` s for ``seconds``. Returns a matplotlib animation;
    ``HTML(anim.to_jshtml())`` plays it in a notebook."""
    import matplotlib.pyplot as plt
    from matplotlib import animation
    P = run["poses"]
    keep = (P[:, 0] >= start) & ((P[:, 0] <= start + seconds) if seconds else True)
    idx = np.where(keep)[0][::every]
    fig = plt.figure(figsize=(12.8, 6.0), facecolor="white")
    ax3 = fig.add_axes([0.0, 0.0, 0.48, 1.0], projection="3d", computed_zorder=False)
    axc = fig.add_axes([0.50, 0.55, 0.24, 0.40]); axd = fig.add_axes([0.75, 0.55, 0.24, 0.40])
    axt = fig.add_axes([0.52, 0.04, 0.46, 0.46])
    path = P[:, 1:3]
    spin = np.r_[0.0, np.cumsum(np.hypot(*np.diff(path, axis=0).T))] / (WHEEL_D / 2)

    def draw(j):
        i = idx[j]
        t, x, y, th, delta, v, seen = P[i]
        rng = np.random.default_rng(i)
        view3d(track, x, y, th, ax3, delta, spin[i], trail=path[: i + 1], seed=i)
        ax3.text2D(0.03, 0.95, title, transform=ax3.transAxes, fontsize=12, weight="bold")
        ax3.text2D(0.03, 0.89, f"t = {t:4.2f} s   speed {v:4.2f} m/s   wheels {np.degrees(delta):+5.1f} deg"
                   + ("   path blocked" if seen else ""), transform=ax3.transAxes, fontsize=9)
        axc.cla(); axc.imshow(render_rgb(track, x, y, th, rng, v)); axc.axis("off"); axc.set_title("colour frame", fontsize=9)
        dep = render_depth(track, x, y, th, rng)
        axd.cla(); axd.set_facecolor("white"); axd.imshow(np.where(dep > 0, dep, np.nan), cmap="viridis_r", vmin=0, vmax=4)
        axd.axis("off"); axd.set_title("depth image (white: no reading)", fontsize=9)
        axt.cla(); draw_track(track, axt); axt.plot(path[: i + 1, 0], path[: i + 1, 1], color="#FF8F00", lw=1.2)
        axt.plot(x, y, "o", color="#B22222", ms=5); axt.set_xlabel(""); axt.set_ylabel(""); axt.tick_params(labelsize=7)
        return []

    anim = animation.FuncAnimation(fig, draw, frames=len(idx), interval=1000 / fps, blit=False)
    plt.close(fig)
    return anim
