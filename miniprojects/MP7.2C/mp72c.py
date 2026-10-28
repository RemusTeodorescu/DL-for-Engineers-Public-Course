r"""MP7.2C - Where was the panel struck? Ex07.2's tensioned panel, now with a
bolt and a reinforcing strip, struck by a hammer nobody saw; a few sensors
record its displacement for 20 ms. Find where it was struck and how hard.
The physics, the reference model (least squares) and the ground truth.

*Deep Learning for Engineering* - MSc, Aalborg University.
Remus Teodorescu (ret@et.aau.dk), with support from Research Assistant
Noman Khan (nomank@energy.aau.dk).

This module extends Ex07.2's ``problem.py`` (course policy C10 rule 1) and is
used by the notebook of MP7.2C (course policy C14).

## What changes against Ex07.2, and why

Ex07.2's panel (notebook 3) is a uniform square membrane, 0.40 m, wave speed
80 m/s, clamped on its edges; any motion of it is a sum of its modes
sin(m pi x/L) sin(n pi y/L), exactly. A real panel is fastened and stiffened:

* a **reinforcing strip** bonded across it, along y at x = 0.27 m, doubles its
  mass per area on the strip's centre line, tapering off over about 12 mm
  either side (a Gaussian profile, ASSUMED); the tension is unchanged, so the
  wave is slower over the strip;
* a **bolt** through a hole at (0.10, 0.20) m carries a small bracket: the
  bolt, its nut and washers and the bracket's foot add ten times the panel's
  own mass per area over a disc of 15 mm radius (a smooth plateau, ASSUMED).

With them the modes of the square no longer separate and **no exact solution
exists**. S is the strength of both: at S = 0 there is no strip and no bolt,
and the solver must reproduce Ex07.2's exact sum of modes
(:func:`check_zero_strength`).

## The problem, in metres of displacement

    m(x, y) u_tt = c^2 lap(u)                        on the panel
    m(x, y) = 1 + S (beta_s g(x) + beta_b chi(x, y))   the strip (beta_s = 1) and the bolt (beta_b = 10)
    u = 0 on the edges,  u(x, y, 0) = 0,  u_t(x, y, 0) = v exp(-r^2 / (2 s^2))

m is the mass per area relative to Ex07.2's uniform panel; the tension, and
with it c = 80 m/s away from the strip and the bolt, is unchanged.
r the distance from the strike (x0, y0), s = 25 mm the hammer's footprint
(known), v the peak velocity. Hidden: x0, y0 and v.

## Solvers

* the **forward model** (:class:`Panel`): finite differences, the five-point
  Laplacian and the leapfrog in time. Second order; many strikes at once.
  Since the panel is linear and its operator symmetric (with the mass as the
  weight), the readings of any strike are also the inner product of the
  strike with the motion that an impulse at the sensor makes
  (:class:`Readings`, reciprocity): six runs give the readings of any number
  of strikes.
* the **reference model** (:func:`least_squares_fit`): scipy's
  least_squares over (x0, y0, v), the forward model on the engineering grid.
* the **ground truth** (:func:`ground_truth`): the forward model on 2 mm and
  1 mm grids, Richardson-extrapolated (:func:`convergence_study`).
"""

from __future__ import annotations

import time

import numpy as np

import problem as pb

# ------------------------------------------------------------------ the data
L = pb.L_PANEL                 #: m, the panel's side, 0.40 (Ex07.2)
C = pb.C_WAVE                  #: m/s, the wave speed away from the strip (Ex07.2)
T_END = pb.WAVE_T_END          #: s, 20 ms (Ex07.2)
S_CHANGE = 1.0                 #: the strength of the strip and the bolt: 0 is Ex07.2's uniform panel
STRIP_X, STRIP_SIGMA, STRIP_BETA = 0.27, 0.012, 1.0   #: m, m, -: the strip's line, its half-width, its extra mass (ASSUMED)
BOLT, BOLT_R, BOLT_BETA = (0.10, 0.20), 0.015, 10.0  #: m, m, -: the bolt, the radius of its disc, its extra mass (ASSUMED)
HAMMER = 0.025                 #: m, the strike's footprint s (known; ASSUMED for a rubber-tipped hammer)
X_RANGE = (0.06, 0.34)         #: m, where a strike may land, both coordinates
V_RANGE = (0.5, 1.5)           #: m/s, its peak velocity
BOLT_CLEAR = 0.04              #: m, a strike is at least this far from the bolt
T_REC = np.round(np.arange(0.0, T_END + 1e-12, 1e-4), 7)   #: s, the sensors' instants: every 0.1 ms, 201
T_MOTION = np.round(np.arange(0.0, T_END + 1e-12, 5e-4), 7)   #: s, the instants the motion is scored at, 41
H_MOTION = 0.008               #: m, the motion is scored on the nodes every 8 mm, 51 x 51
#: the six sensor positions [m]: all on the 8 mm grid, away from the bolt
SENSORS = np.array([[0.080, 0.080], [0.200, 0.104], [0.320, 0.080],
                    [0.080, 0.320], [0.192, 0.296], [0.320, 0.312]])
#: the test cases, of rising difficulty: which sensors are read, the noise (m), the hidden strike (x0, y0 [m], v [m/s])
CASES = [dict(name="1", sensors=(0, 1, 2, 3, 4, 5), noise=2e-6, strike=(0.13, 0.27, 1.00)),
         dict(name="2", sensors=(0, 2, 3, 5), noise=5e-6, strike=(0.25, 0.15, 0.80)),
         dict(name="3", sensors=(0, 1, 4), noise=5e-6, strike=(0.30, 0.31, 1.20)),
         dict(name="4", sensors=(1, 2, 4), noise=10e-6, strike=(0.17, 0.12, 0.60))]


def mass(x, y, s=S_CHANGE):
    """m(x, y): the panel's mass per area relative to Ex07.2's, raised by the strip and by the bolt."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    strip = STRIP_BETA * np.exp(-0.5 * ((x - STRIP_X) / STRIP_SIGMA) ** 2)
    bolt = BOLT_BETA * np.exp(-(np.hypot(x - BOLT[0], y - BOLT[1]) / BOLT_R) ** 6)   # a plateau, edges over ~3 mm
    return 1.0 + s * (strip + bolt)


def strike_velocity(x, y, strike):
    """m/s, the initial velocity of the strike (x0, y0, v) at points x, y; the
    strikes may be stacked: strike of shape (n, 3) gives (n,) + x.shape."""
    s = np.atleast_2d(np.asarray(strike, float))
    sh = (len(s),) + (1,) * np.ndim(x)
    x0, y0, v = (s[:, i].reshape(sh) for i in range(3))
    out = v * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * HAMMER ** 2))
    return out if np.ndim(strike) > 1 else out[0]


# ---------------------------------------------------------- the forward model
class Panel:
    """Finite differences on the nodes every h of the panel, edges included
    (u = 0 there), leapfrog with a step that divides the sensors' 0.1 ms and
    keeps c sqrt(2) dt / h = 0.71 (stable below 1). Built once per grid;
    ``run`` marches any number of initial velocities at once."""

    def __init__(self, h, s=S_CHANGE):
        self.h, self.s = h, s
        self.n = round(L / h)
        self.x = np.linspace(0.0, L, self.n + 1)
        X, Y = np.meshgrid(self.x, self.x)                     # row = y
        self.X, self.Y = X, Y
        self.k = max(1, round(0.016 / h))                    # steps per 0.1 ms: c sqrt(2) dt / h = 0.71 on every grid
        self.dt = 1e-4 / self.k
        self.m = mass(X, Y, s)                                 # (n+1, n+1)
        self.cm = (C * self.dt / h) ** 2 / self.m[1:-1, 1:-1]  # (c dt / h)^2 / m on the interior nodes

    def index(self, pts):
        """Row, column of the nodes at points pts (n, 2) [m]; they must be nodes."""
        ij = np.asarray(pts) / self.h
        if not np.allclose(ij, np.round(ij), atol=1e-6):
            raise ValueError("the points are not nodes of this grid")
        ij = np.round(ij).astype(int)
        return ij[:, 1], ij[:, 0]

    def march(self, v0, sensors=None, motion=False, keep=False):
        """March initial velocities v0 (nb, n+1, n+1) [m/s] (edges ignored).
        Returns a dict with any of: 'readings' (nb, 201, len(sensors)) [m] at the
        sensors every 0.1 ms; 'motion' (nb, 41, 51, 51) [m] on the 8 mm nodes
        every 0.5 ms; 'fields' (nb, 201, n+1, n+1) [m] everywhere every 0.1 ms."""
        v = np.asarray(v0, float)[:, 1:-1, 1:-1]
        nb = len(v)

        def lap(u):
            p = np.pad(u, ((0, 0), (1, 1), (1, 1)))
            return p[:, 2:, 1:-1] + p[:, :-2, 1:-1] + p[:, 1:-1, 2:] + p[:, 1:-1, :-2] - 4 * u
        dt = self.dt
        old = np.zeros_like(v)
        u = dt * v + dt / 6 * self.cm * lap(v)               # u(dt) = dt v0 + dt^3/6 (c^2/m) lap v0: Taylor's series
        out = {}
        if sensors is not None:
            ri, ci = self.index(sensors)
            out["readings"] = np.zeros((nb, len(T_REC), len(ri)))
        step_m = round(H_MOTION / self.h)
        if motion:
            out["motion"] = np.zeros((nb, len(T_MOTION), self.n // step_m + 1, self.n // step_m + 1))
        if keep:
            out["fields"] = np.zeros((nb, len(T_REC), self.n + 1, self.n + 1), dtype=np.float32)
        km = round(5e-4 / dt)
        nsteps = round(T_END / dt)
        for s in range(1, nsteps + 1):
            if s % self.k == 0:
                full = np.pad(u, ((0, 0), (1, 1), (1, 1)))
                if sensors is not None:
                    out["readings"][:, s // self.k] = full[:, ri, ci]
                if keep:
                    out["fields"][:, s // self.k] = full
            if motion and s % km == 0:
                out["motion"][:, s // km] = np.pad(u, ((0, 0), (1, 1), (1, 1)))[:, ::step_m, ::step_m]
            if s == nsteps:
                break
            new = 2 * u - old + self.cm * lap(u)
            old, u = u, new
        return out

    def run(self, strikes, sensors=SENSORS, motion=False):
        """March strikes (n, 3) [m, m, m/s]. Returns the readings (n, 201,
        sensors) [m] and, if ``motion``, the motion (n, 41, 51, 51) [m]."""
        v0 = strike_velocity(self.X, self.Y, np.atleast_2d(np.asarray(strikes, float)))
        out = self.march(v0, sensors, motion)
        return (out["readings"], out["motion"]) if motion else out["readings"]


class Readings:
    """The readings of any strike at the six sensors by reciprocity. For each
    sensor, one march from an impulse of velocity 1/m at the sensor's node:
    its field G_s(node, t) at the 201 instants. Because the scheme is
    symmetric with the mass as the weight, the reading of sensor s at t for a
    strike of initial velocity v0 is sum over the nodes of G_s(node, t) m(node)
    v0(node): exactly the forward model's reading. Six marches, then any
    strike costs one weighted sum."""

    def __init__(self, h=0.002, sensors=SENSORS):
        self.panel = p = Panel(h)
        ri, ci = p.index(sensors)
        imp = np.zeros((len(ri), p.n + 1, p.n + 1))
        imp[np.arange(len(ri)), ri, ci] = 1.0 / p.m[ri, ci]
        G = p.march(imp, keep=True)["fields"]                  # (sensors, 201, n+1, n+1) float32
        self.G = (G * p.m[None, None].astype(np.float32)).reshape(len(ri), len(T_REC), -1)   # m-weighted
        self.G = np.ascontiguousarray(self.G.transpose(2, 1, 0).reshape(-1, len(T_REC) * len(ri)))  # (nodes, 201 * sensors)
        self.ns = len(ri)

    def __call__(self, strikes, batch=256):
        """Clean readings (n, 201, sensors) [m] of strikes (n, 3)."""
        strikes = np.atleast_2d(np.asarray(strikes, float))
        out = np.empty((len(strikes), len(T_REC), self.ns))
        X, Y = self.panel.X.ravel(), self.panel.Y.ravel()
        for b in range(0, len(strikes), batch):
            v0 = strike_velocity(X, Y, strikes[b:b + batch]).astype(np.float32)   # (batch, nodes)
            out[b:b + batch] = (v0 @ self.G).reshape(-1, len(T_REC), self.ns)
        return out


# ------------------------------------------------------ Ex07.2's exact motion
def modal_sum(strike, x, y, t, n_modes=80, n_quad=800):
    """The exact motion of Ex07.2's uniform panel (no strip, no bolt) struck by
    (x0, y0, v): the sum of n_modes x n_modes modes, the strike's sine
    transform taken on n_quad x n_quad midpoints. x, y: points (p,) [m]; t: (k,)
    [s]. Returns (k, p) [m]."""
    xq = (np.arange(n_quad) + 0.5) * L / n_quad
    Xq, Yq = np.meshgrid(xq, xq)
    v0 = strike_velocity(Xq, Yq, strike)
    m = np.arange(1, n_modes + 1)
    S = np.sin(np.outer(m, xq) * np.pi / L)
    b = 4 / L ** 2 * (S @ v0 @ S.T) * (L / n_quad) ** 2          # b[j, i]: mode i along x, j along y
    w = np.pi * C / L * np.sqrt(m[None, :] ** 2 + m[:, None] ** 2)
    sx, sy = np.sin(np.outer(x, m) * np.pi / L), np.sin(np.outer(y, m) * np.pi / L)
    return np.array([np.einsum("pi,ji,pj->p", sx, b / w * np.sin(w * tk), sy) for tk in t])


def check_zero_strength(grids=(0.002, 0.001), strike=(0.13, 0.27, 1.0), verbose=True):
    """At S = 0 the panel is Ex07.2's: the march, extrapolated from two grids,
    must reproduce the exact sum of modes at the six sensors. Returns the
    largest difference as a share of the largest reading, on the finer grid
    alone and extrapolated."""
    r = [Panel(h, s=0.0).run([strike])[0] for h in grids]
    ex = modal_sum(strike, SENSORS[:, 0], SENSORS[:, 1], T_REC)
    n = np.abs(ex).max()
    e_fine, e_rich = float(np.abs(r[1] - ex).max() / n), float(np.abs((4 * r[1] - r[0]) / 3 - ex).max() / n)
    if verbose:
        print(f"   S = 0: largest difference from Ex07.2's exact sum of modes {e_fine*100:.3f} % of the largest reading "
              f"on the {grids[1]*1e3:g} mm grid, {e_rich*100:.4f} % extrapolated")
    return e_fine, e_rich


# ------------------------------------------------------- the reference model
REF_H = 0.004                  #: m, the engineering grid: 100 x 100 cells, chosen in the convergence study


def bounds():
    """Lower and upper bounds of (x0, y0, v)."""
    return np.array([X_RANGE[0], X_RANGE[0], V_RANGE[0]]), np.array([X_RANGE[1], X_RANGE[1], V_RANGE[1]])


def default_starts():
    """Nine starting strikes on a 3 x 3 grid over the panel at the middle velocity: a wave's misfit has many
    valleys, and a fit from one start can end in the wrong one."""
    g = np.linspace(0.1, 0.3, 3)
    return [np.array([x, y, 1.0]) for y in g for x in g]


def least_squares_fit(readings, sensors, noise, h=REF_H, panel=None, starts=None, max_nfev=40):
    """The reference model: scipy.optimize.least_squares over (x0, y0, v), the
    readings predicted by the forward model on the engineering grid, the
    residuals divided by the noise; from each of nine starts, the best fit
    kept. Returns the fitted strike, the cost, the number of forward runs
    and the wall-clock time [s]."""
    from scipy.optimize import least_squares
    p = Panel(h) if panel is None else panel
    t0 = time.perf_counter()
    count = [0]
    lo, hi = bounds()

    def res(q):
        count[0] += 1
        return ((p.run([q], sensors)[0] - readings) / noise).ravel()

    best = None
    for q0 in (starts if starts is not None else default_starts()):
        f = least_squares(res, q0, bounds=(lo, hi), x_scale=np.array([0.01, 0.01, 0.1]), diff_step=1e-3,
                          max_nfev=max_nfev)
        if best is None or f.cost < best.cost:
            best = f
    return dict(strike=best.x, cost=best.cost, runs=count[0], seconds=time.perf_counter() - t0)


def information_bound(strike, sensors, noise, h=REF_H, motion_h=0.008):
    """The Cramer-Rao bound at a strike: the smallest standard deviation any
    unbiased method can reach, sigma^2 (J^T J)^-1, J the derivatives of the
    readings with respect to (x0, y0, v) by central differences on the
    forward model. Returns the position's (mm, the two coordinates combined),
    the velocity's (per cent) and the motion's largest standard deviation as a
    share of its peak (per cent, from the motion on the 8 mm nodes)."""
    p = Panel(h)
    q = np.asarray(strike, float)
    steps = np.array([1e-3, 1e-3, 1e-2])
    J, Jm = [], []
    for i in range(3):
        dq = np.zeros(3); dq[i] = steps[i]
        rp, mp_ = p.run([q + dq], sensors, motion=True)
        rm, mm_ = p.run([q - dq], sensors, motion=True)
        J.append(((rp - rm) / (2 * steps[i])).ravel()); Jm.append(((mp_ - mm_) / (2 * steps[i])).ravel())
    J, Jm = np.array(J).T, np.array(Jm).T
    cov = noise ** 2 * np.linalg.inv(J.T @ J)
    m0 = np.abs(p.run([q], sensors, motion=True)[1]).max()
    sd_motion = np.sqrt(np.einsum("ij,jk,ik->i", Jm, cov, Jm)).max() / m0
    return (float(np.hypot(np.sqrt(cov[0, 0]), np.sqrt(cov[1, 1])) * 1e3), float(np.sqrt(cov[2, 2]) / q[2] * 100),
            float(sd_motion * 100))


# -------------------------------------------------------- the ground truth
TRUTH_H = (0.008, 0.004, 0.002, 0.001)   #: m, four grids each twice as fine


def convergence_study(strike, grids=TRUTH_H, verbose=True):
    """The readings at the six sensors on grids each twice as fine; the change
    from the grid before as a share of the largest reading, and the
    Richardson extrapolation from the last two. Returns rows (h, largest
    reading, change, seconds) and the extrapolated readings."""
    rows, last = [], []
    for h in grids:
        t0 = time.perf_counter()
        r = Panel(h).run([strike])[0]
        n = np.abs(r).max()
        change = float(np.abs(r - last[-1]).max() / n) if last else np.nan
        rows.append((h, n, change, time.perf_counter() - t0))
        last.append(r)
        if verbose:
            print(f"   {h*1e3:.0f} mm: largest reading {n*1e3:.4f} mm, change from the grid before {change*100:.3f} % of it "
                  f"({rows[-1][3]:.1f} s)", flush=True)
    rich = (4 * last[-1] - last[-2]) / 3
    if verbose:
        c = [r[2] for r in rows[1:]]
        print("   ratios of the changes: " + ", ".join(f"{a/b:.2f}" for a, b in zip(c[:-1], c[1:]))
              + f"; Richardson moves the readings by {np.abs(rich - last[-1]).max()/rows[-1][1]*100:.4f} % of the largest")
    return rows, rich


def ground_truth(strike, sensors=SENSORS, grids=TRUTH_H[-2:]):
    """The clean readings (201, sensors) [m] and the motion on the 8 mm nodes
    (41, 51, 51) [m] of a strike, Richardson-extrapolated from the 2 mm and
    1 mm grids."""
    r = [Panel(h).run([strike], sensors, motion=True) for h in grids]
    return (4 * r[1][0][0] - r[0][0][0]) / 3, (4 * r[1][1][0] - r[0][1][0]) / 3


def motion(strike, grids=TRUTH_H[-2:]):
    """The motion (41, 51, 51) [m] on the 8 mm nodes every 0.5 ms of a strike
    (x0, y0, v), as the ground truth computes it: for scoring an estimate."""
    return ground_truth(strike, SENSORS[:1], grids)[1]


def motion_error(estimate, truth_motion):
    """The largest difference of the motion an estimated strike implies from
    the true motion, as a share of the true motion's peak."""
    return float(np.abs(motion(estimate) - truth_motion).max() / np.abs(truth_motion).max())


# ------------------------------------------------------- the training data
def sample_strikes(n, seed=72):
    """n strikes (x0, y0, v) drawn uniformly over the range, at least 40 mm from the bolt."""
    rng = np.random.default_rng(seed)
    out = np.empty((0, 3))
    while len(out) < n:
        q = np.c_[rng.uniform(*X_RANGE, 2 * n), rng.uniform(*X_RANGE, 2 * n), rng.uniform(*V_RANGE, 2 * n)]
        q = q[np.hypot(q[:, 0] - BOLT[0], q[:, 1] - BOLT[1]) >= BOLT_CLEAR]
        out = np.r_[out, q]
    return out[:n]


def simulate(strikes, h=0.002, operator=None):
    """Clean readings (n, 201, 6) [m] of many strikes at the six sensors, by
    reciprocity on the grid h (:class:`Readings`): the training data of an
    amortised network. The six marches take some seconds on the 2 mm grid; then
    about a millisecond per strike."""
    op = Readings(h) if operator is None else operator
    return op(strikes)


# ------------------------------------------------------------------ pictures
def draw_panel(ax=None, sensors=SENSORS, strike=None, label_sensors=True):
    """The panel of MP7.2C: the strip and the bolt (their extra mass in grey),
    the six sensors, and a strike (x0, y0, v) if given."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 5.0))
    g = np.linspace(0, L, 201)
    X, Y = np.meshgrid(g, g)
    ax.contourf(X * 1e3, Y * 1e3, mass(X, Y), levels=[1.05, 1.3, 1.6, 2.01, 6.0, 11.01],
                colors=["#e0e0e0", "#c8c8c8", "#b0b0b0", "#909090", "#606060"])   # the extra mass: strip and bolt
    ax.add_patch(plt.Rectangle((0, 0), L * 1e3, L * 1e3, fill=False, lw=3, ec="tab:blue"))
    ax.text(STRIP_X * 1e3, L * 1e3 + 6, "strip", ha="center", va="bottom", fontsize=8)
    ax.text(BOLT[0] * 1e3, BOLT[1] * 1e3 - 24, "bolt", ha="center", va="top", fontsize=8)
    S = np.asarray(sensors)
    ax.scatter(S[:, 0] * 1e3, S[:, 1] * 1e3, marker="s", s=36, c="tab:red", zorder=3, label="sensors")
    if label_sensors:
        for j, (x, y) in enumerate(SENSORS * 1e3):
            if any(np.allclose(SENSORS[j], q) for q in S):
                ax.text(x + 6, y + 6, str(j + 1), color="tab:red", fontsize=8)
    if strike is not None:
        ax.add_patch(plt.Circle((strike[0] * 1e3, strike[1] * 1e3), HAMMER * 1e3, fill=False, ec="tab:orange", lw=1.5))
        ax.plot(strike[0] * 1e3, strike[1] * 1e3, "*", color="tab:orange", ms=12, label="strike")
    ax.set_aspect("equal"); ax.set_xlim(-10, L * 1e3 + 10); ax.set_ylim(-10, L * 1e3 + 20)
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
    return ax


# ----------------------------------------------------------- the truth file
TRUTH_FILE = "MP7.2C_truth.npz"


def write_truth(path=TRUTH_FILE):
    """The test cases: the readings at the sensors read (seeded noise), the
    hidden strike, its clean readings and its motion from the ground truth;
    the convergence table and the zero-strength check."""
    print("zero-strength check")
    zs = check_zero_strength()
    print("convergence, case 1's strike")
    rows, _ = convergence_study(CASES[0]["strike"])
    out = dict(t=T_REC, t_motion=T_MOTION, x_motion=np.arange(0, L + 1e-9, H_MOTION), sensors=SENSORS,
               cases=np.array([c["name"] for c in CASES]))
    for c in CASES:
        n = c["name"]
        clean, mot = ground_truth(c["strike"])
        seed = 720 + int(n)
        rng = np.random.default_rng(seed)
        mask = np.isin(np.arange(len(SENSORS)), c["sensors"]).astype(float)
        noisy = (clean + rng.normal(0, c["noise"], clean.shape)) * mask
        out.update({f"case{n}_readings": noisy, f"case{n}_mask": mask, f"case{n}_noise": np.array(c["noise"]),
                    f"case{n}_seed": np.array(seed), f"case{n}_strike": np.array(c["strike"]),
                    f"case{n}_clean": clean, f"case{n}_motion": mot})
        print(f"   case {n}: {int(mask.sum())} sensors, noise {c['noise']*1e6:.0f} um; strike at ({c['strike'][0]*1e3:.0f}, "
              f"{c['strike'][1]*1e3:.0f}) mm, {c['strike'][2]:.2f} m/s; largest reading {np.abs(clean).max()*1e3:.3f} mm, "
              f"largest motion {np.abs(mot).max()*1e3:.3f} mm", flush=True)
    out["conv_h_mm"] = np.array([r[0] * 1e3 for r in rows]); out["conv_peak"] = np.array([r[1] for r in rows])
    out["conv_change"] = np.array([r[2] for r in rows]); out["conv_seconds"] = np.array([r[3] for r in rows])
    out["zero_strength_fine"], out["zero_strength_extrapolated"] = np.array(zs[0]), np.array(zs[1])
    out["params"] = np.array([f"{k} = {v}" for k, v in dict(
        PANEL="0.40 x 0.40 m, clamped edges, c = 80 m/s (Ex07.2)", STRIP="along y at x = 0.27 m, mass x2 on its line, "
        "Gaussian profile sigma 12 mm, ASSUMED", BOLT="at (0.10, 0.20) m, mass x11 over a 15 mm radius, ASSUMED",
        HAMMER="Gaussian footprint s = 25 mm, ASSUMED", STRIKE_RANGE="x0, y0 0.06-0.34 m, 40 mm from the bolt; v 0.5-1.5 m/s",
        NOISE="2, 5 or 10 um, one standard deviation, ASSUMED for displacement sensors").items()])
    out["README"] = np.array(
        "MP7.2C test cases: the 0.40 m panel of Ex07.2 with a strip and a bolt, struck at t = 0. sensors: (x, y) of the six "
        "sensors [m]; t: their instants [s] (every 0.1 ms). OBSERVED: case<n>_readings[k, j]: displacement [m] at t[k] and "
        "sensor j with noise of standard deviation case<n>_noise [m] from seed case<n>_seed, zero where case<n>_mask is 0 "
        "(sensor not read). HIDDEN (for scoring): case<n>_strike = (x0 [m], y0 [m], v [m/s]); case<n>_clean: the readings "
        "without noise [m]; case<n>_motion[k, i, j]: displacement [m] at t_motion[k] (every 0.5 ms) on the nodes "
        "x_motion[j], x_motion[i] (every 8 mm). Ground truth: finite differences on 2 mm and 1 mm grids, Richardson-"
        "extrapolated. conv_*: case 1's strike on four grids (cell mm, largest reading m, change from the grid before as a "
        "share of it, seconds). zero_strength_*: S = 0 against Ex07.2's exact sum of modes, as a share of the largest reading.")
    np.savez_compressed(path, **out)
    print("written", path)


if __name__ == "__main__":
    import sys
    if "--save" in sys.argv:
        write_truth()
    else:
        check_zero_strength()
        convergence_study(CASES[0]["strike"], TRUTH_H[:3])
