r"""MP7.1B - How old is the insulation? The averaged slot of Ex07.1 with a
hidden disc of aged insulation, found from a few temperature sensors.
The physics, the reference model (least squares) and the ground truth.

*Deep Learning for Engineering* - MSc, Aalborg University.
Remus Teodorescu (ret@et.aau.dk), with support from Research Assistant
Noman Khan (nomank@energy.aau.dk).

This module extends Ex07.1's ``problem.py`` (course policy C10 rule 1) and is
used by the notebook of MP7.1B (course policy C14).

## What changes against Ex07.1, and why

Ex07.1's slot is one averaged material, k_eff = 0.70 W/(m K), and has an exact
double sine series. Insulation ages: under heat the impregnating resin cracks
and delaminates, and air (0.03 W/(m K)) replaces part of it, so the bundle's
conductivity falls locally. Here a disc of the slot, centre (x_d, y_d) and
radius r_d, has a lower conductivity k_d. The conductivity jumps at the
disc's edge, the field is no longer a sum of the rectangle's modes, and **no
exact solution exists**. The strength of the change is 1 - k_d / k_eff; at
k_d = k_eff (strength zero) the solver must reproduce Ex07.1's exact series
(:func:`check_zero_strength`).

The disc is to be found from sensors: thermocouples or fibre-optic probes
placed in the slot at a few points, read at four currents.

## The problem, in kelvin above the wall

    div(k grad theta) + q = 0       in the slot,  q = FILL (c0 + c1 theta)  (Ex07.1's heat)
    k = k_d inside the disc, k_eff outside
    theta = 0                       on the four walls

## Three solvers

* the **forward model** (:class:`Slot`): cell-centred finite volumes on the
  whole slot, the conductivity of a cell cut by the disc's edge taken from the
  share of the cell inside the disc (a straight ramp across one cell), so the
  answer changes smoothly as the disc moves - which least squares needs.
* the **reference model** (:func:`least_squares_fit`): scipy's
  least_squares over the four disc parameters, with the forward model on the
  engineering grid, timed per new set of readings.
* the **ground truth** (:func:`ground_truth`): the forward model on 0.025 mm
  cells, checked on grids each twice as fine.
"""

from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import problem as pb

# ------------------------------------------------------------------ the data
K_EFF = pb.K_EFF                    #: W/(m K), the healthy averaged bundle, Ex07.1's 0.70
CURRENTS = (15.0, 25.0, pb.I_RATED, 40.0)   #: A, the four currents the sensors are read at
SENSORS = np.array([[-2.0, -4.5], [2.0, -4.5], [-2.0, 0.0],
                    [2.0, 0.0], [-2.0, 4.5], [2.0, 4.5]]) * 1e-3   #: m, the six sensor positions: two columns of three
FOUR = np.array([0, 1, 2, 3])       #: the four sensors kept when only four are read (the top pair missing)
#: the range of the hidden disc: centre anywhere with the disc inside the slot, radius, conductivity
R_RANGE = (1.0e-3, 2.5e-3)          #: m
K_RANGE = (0.28, 0.49)              #: W/(m K): 40 % to 70 % of k_eff (aged; "halved" is 0.35)   [ASSUMED]
NOISE_RANGE = (0.05, 0.10)          #: K, one standard deviation of the sensor noise
#: the agreed accuracy, set from what the readings hold (the Cramer-Rao bound) and what least squares reaches
CENTRE_TOL = 1.2e-3                 #: m, the disc's centre
STRENGTH_TOL = 0.35                 #: the disc's strength, relative
HOT_TOL = 0.3                       #: K, the hot spot at rated current


def heat_coefficients(I):
    """(c0, c1) of Ex07.1's averaged heat, q = c0 + c1 theta [W/m^3, W/(m^3 K)]."""
    return pb.heat_coefficients(I)


# ---------------------------------------------------------- the forward model
class Slot:
    """Finite volumes on the whole slot, square cells of side h, walls at
    theta = 0. Built once per grid; ``solve(disc, currents)`` assembles the
    conductivity of one disc and solves every current."""

    def __init__(self, h):
        self.h = h
        self.nx, self.ny = round(2 * pb.A_HALF / h), round(2 * pb.B_HALF / h)
        self.xc = -pb.A_HALF + (np.arange(self.nx) + 0.5) * h
        self.yc = -pb.B_HALF + (np.arange(self.ny) + 0.5) * h
        self.X, self.Y = np.meshgrid(self.xc, self.yc)
        n = self.nx * self.ny
        idx = np.arange(n).reshape(self.ny, self.nx)
        self.ix = (idx[:, :-1].ravel(), idx[:, 1:].ravel())
        self.iy = (idx[:-1, :].ravel(), idx[1:, :].ravel())
        self.n = n

    def conductivity(self, disc):
        """k per cell [W/(m K)] for disc = (x, y, r, k) [m, m, m, W/(m K)]:
        the share f of a cell inside the disc rises linearly from 0 to 1 over
        one cell across the edge; the cell's k is the harmonic mean of k_d
        and k_eff weighted by f (the two in series)."""
        if disc is None:
            return np.full(self.X.shape, K_EFF)
        x, y, r, kd = disc
        d = np.hypot(self.X - x, self.Y - y)
        f = np.clip((r - d) / self.h + 0.5, 0.0, 1.0)
        return 1.0 / (f / kd + (1 - f) / K_EFF)

    def solve(self, disc, currents=CURRENTS):
        """theta [K] per cell, shape (len(currents), ny, nx)."""
        k = self.conductivity(disc)
        hm = lambda a, b: 2 * a * b / (a + b)
        gx, gy = hm(k[:, :-1], k[:, 1:]).ravel(), hm(k[:-1, :], k[1:, :]).ravel()
        wall = np.zeros_like(k)
        wall[:, 0] += 2 * k[:, 0]; wall[:, -1] += 2 * k[:, -1]
        wall[0, :] += 2 * k[0, :]; wall[-1, :] += 2 * k[-1, :]
        diag = wall.ravel().copy()
        np.add.at(diag, self.ix[0], gx); np.add.at(diag, self.ix[1], gx)
        np.add.at(diag, self.iy[0], gy); np.add.at(diag, self.iy[1], gy)
        rows = np.r_[self.ix[0], self.ix[1], self.iy[0], self.iy[1]]
        cols = np.r_[self.ix[1], self.ix[0], self.iy[1], self.iy[0]]
        K = (sp.csc_matrix((-np.r_[gx, gx, gy, gy], (rows, cols)), shape=(self.n, self.n))
             + sp.diags(diag)).tocsc()
        lu = spla.splu(K)                                    # conduction alone, factorised once per disc
        a = self.h ** 2
        self._u = lu.solve(np.full(self.n, a))                # the field of a unit heat source everywhere
        out = []
        self.balance = []
        for I in currents:
            c0, c1 = heat_coefficients(I)
            # the heat grows by c1 per kelvin (0.3 % of c0): a few fixed-point steps on the one factorisation,
            # each multiplying the remaining error by about c1 / (k_eff lambda_1) = 0.04
            th = c0 * self._u                              # the heat at the wall temperature alone ...
            for _ in range(4):                             # ... then its growth with the rise, c1 per kelvin
                th = c0 * self._u + c1 * lu.solve(a * th)
            th = th.reshape(self.ny, self.nx)
            made = float((c0 + c1 * th).sum() * a)
            self.balance.append((made, float((wall * th).sum())))   # W/m: made, through the walls
            out.append(th)
        return np.array(out)

    def weights(self, pts):
        """The bilinear weights that read a field at points pts (n, 2) [m]
        from the cell centres, the walls (theta = 0) included: a sparse
        (n, cells) matrix, built once per set of points."""
        xp = np.r_[-pb.A_HALF, self.xc, pb.A_HALF]
        yp = np.r_[-pb.B_HALF, self.yc, pb.B_HALF]
        rows, cols, vals = [], [], []
        for k, (x, y) in enumerate(np.asarray(pts)):
            i = int(np.clip(np.searchsorted(xp, x) - 1, 0, len(xp) - 2))
            j = int(np.clip(np.searchsorted(yp, y) - 1, 0, len(yp) - 2))
            tx = (x - xp[i]) / (xp[i + 1] - xp[i]); ty = (y - yp[j]) / (yp[j + 1] - yp[j])
            for di, dj, w in ((0, 0, (1 - tx) * (1 - ty)), (1, 0, tx * (1 - ty)), (0, 1, (1 - tx) * ty), (1, 1, tx * ty)):
                ci, cj = i + di - 1, j + dj - 1                  # cell indices; -1 or n is a wall
                if 0 <= ci < self.nx and 0 <= cj < self.ny and w != 0:
                    rows.append(k); cols.append(cj * self.nx + ci); vals.append(w)
        return sp.csr_matrix((vals, (rows, cols)), shape=(len(pts), self.n))

    def at(self, th, pts):
        """A field (ny, nx) [K] at points pts (n, 2) [m], bilinear between cell
        centres, with the walls (theta = 0) added to the grid."""
        return self.weights(pts) @ th.ravel()

    def readings(self, disc, sensors=SENSORS, currents=CURRENTS):
        """The clean sensor readings [K], shape (len(currents), len(sensors)),
        and the hot spot at each current [K]."""
        key = np.asarray(sensors).tobytes()
        if getattr(self, "_key", None) != key:
            self._key, self._w = key, self.weights(sensors)
        th = self.solve(disc, currents)
        flat = th.reshape(len(currents), -1)
        return (self._w @ flat.T).T, flat.max(1)


# ------------------------------------------------------- the reference model
REF_H = 0.25e-3         #: m, the engineering grid: 40 x 80 cells on the whole slot, chosen in the convergence study


def least_squares_fit(readings, sensors=SENSORS, noise=0.1, h=REF_H, x0=None, slot=None, starts=None):
    """The reference model: scipy.optimize.least_squares over the disc's four
    parameters (x, y, r, k), the readings predicted by the forward model on the
    engineering grid, the residuals divided by the noise. Several starting
    points (the disc must be found anywhere in the slot); the best fit wins.
    Returns the fitted disc, its hot spot at each current, the cost, the
    number of forward solves and the wall-clock time (s)."""
    from scipy.optimize import least_squares
    s = Slot(h) if slot is None else slot
    t0 = time.perf_counter()
    count = [0]
    lo, hi = bounds()

    def res(p):
        count[0] += 1
        r, _ = s.readings(p, sensors)
        return ((r - readings) / noise).ravel()

    best = None
    for p0 in (starts if starts is not None else default_starts()):
        f = least_squares(res, p0, bounds=(lo, hi), x_scale=np.array([1e-3, 1e-3, 1e-3, 0.1]),
                          diff_step=1e-3, max_nfev=60)
        if best is None or f.cost < best.cost:
            best = f
    _, hot = s.readings(best.x, sensors)
    return dict(disc=best.x, hot=hot, cost=best.cost, solves=count[0], seconds=time.perf_counter() - t0)


def bounds():
    """Lower and upper bounds of (x, y, r, k): the centre at least the
    smallest radius inside the walls."""
    lo = np.array([-pb.A_HALF + R_RANGE[0], -pb.B_HALF + R_RANGE[0], R_RANGE[0], K_RANGE[0]])
    hi = np.array([pb.A_HALF - R_RANGE[0], pb.B_HALF - R_RANGE[0], R_RANGE[1], K_RANGE[1]])
    return lo, hi


def default_starts():
    """Four starting discs, one in each quarter of the slot, radius 1.75 mm, k 0.385 W/(m K): the middle of
    the range. The readings alone do not say in which quarter the disc is, so a fit from one start can end
    in the wrong one."""
    return [np.array([sx * 1.5e-3, sy * 4.0e-3, 1.75e-3, 0.385]) for sx in (-1, 1) for sy in (-1, 1)]


def strength(r, k):
    """The disc's strength [mm²]: its area times the contrast (k_eff - k)/(k_eff + k). Far from the disc,
    a disc of lower conductivity changes the field like a dipole whose size is this product, so the sensors
    see the product far better than the radius and the conductivity apart (a large mild disc and a small
    strong one look alike)."""
    return np.pi * (np.asarray(r) * 1e3) ** 2 * (K_EFF - np.asarray(k)) / (K_EFF + np.asarray(k))


def information_bound(disc, sensors, noise, slot=None):
    """The Cramer-Rao bound at a disc: the smallest standard deviation any
    unbiased method can reach from these readings, sigma^2 (J^T J)^-1 with
    J the derivatives of the readings with respect to (x, y, r, k), by
    finite differences on the forward model. Returns the centre's (mm), the
    strength's (per cent) and the hot spot's at rated current (K)."""
    s = Slot(REF_H) if slot is None else slot
    d = np.array(disc, float)
    r0, h0 = s.readings(d, sensors)
    J, Jh, Js = [], [], []
    for i, step in enumerate((0.05e-3, 0.05e-3, 0.05e-3, 0.01)):
        dp = d.copy(); dp[i] += step
        r1, h1 = s.readings(dp, sensors)
        J.append(((r1 - r0) / step).ravel()); Jh.append((h1[2] - h0[2]) / step)
        Js.append((strength(dp[2], dp[3]) - strength(d[2], d[3])) / step)
    J = np.array(J).T
    cov = noise ** 2 * np.linalg.pinv(J.T @ J)
    Jh, Js = np.array(Jh), np.array(Js)
    return (float(np.hypot(np.sqrt(cov[0, 0]), np.sqrt(cov[1, 1])) * 1e3),
            float(np.sqrt(Js @ cov @ Js) / strength(d[2], d[3]) * 100), float(np.sqrt(Jh @ cov @ Jh)))


def draw_slot(ax=None, disc=None, sensors=SENSORS):
    """The averaged slot, its 32 wires faintly, a disc of aged insulation
    (x, y, r, k) if given, and the sensors."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle
    if ax is None:
        _, ax = plt.subplots(figsize=(3.6, 6.4))
    a, b = pb.A_HALF * 1e3, pb.B_HALF * 1e3
    ax.add_patch(Rectangle((-a, -b), 2 * a, 2 * b, facecolor="#f2e6c9", edgecolor="#1f77b4", lw=2.4, zorder=0))
    for i in range(pb.N_COLS):
        for j in range(pb.N_ROWS):
            ax.add_patch(Circle((-a + (i + 0.5) * 2 * a / pb.N_COLS, -b + (j + 0.5) * 2 * b / pb.N_ROWS),
                                pb.WIRE_D / 2 * 1e3, facecolor="#d9772b", alpha=0.18, edgecolor="none", zorder=1))
    if disc is not None:
        ax.add_patch(Circle((disc[0] * 1e3, disc[1] * 1e3), disc[2] * 1e3, facecolor="#7b3294", alpha=0.45,
                            edgecolor="#7b3294", lw=1.2, zorder=2, label=f"aged, k = {disc[3]:.2f} W/(m·K)"))
    ax.scatter(np.asarray(sensors)[:, 0] * 1e3, np.asarray(sensors)[:, 1] * 1e3, marker="s", s=40, c="r", zorder=3, label="sensors")
    for j, (x, y) in enumerate(np.asarray(sensors) * 1e3):
        ax.text(x + 0.4, y + 0.4, str(j + 1), fontsize=8, color="r", zorder=4)
    ax.set_aspect("equal"); ax.set_xlim(-a - 0.5, a + 0.5); ax.set_ylim(-b - 0.5, b + 0.5)
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]"); ax.legend(fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    return ax


# -------------------------------------------------------- the ground truth
TRUTH_H = (0.1e-3, 0.05e-3, 0.025e-3, 0.0125e-3)   #: m, four grids each twice as fine
TRUTH_GRID = 0.025e-3   #: m, the ground truth's cells: 400 x 800 on the whole slot


def convergence_study(disc, I=pb.I_RATED, hs=TRUTH_H, verbose=True):
    """The hot spot, two sensors' readings and the energy balance on grids each
    twice as fine. Returns rows of (h [m], cells, hot spot [K], sensor 1 [K],
    sensor 4 [K], balance residual [-], seconds)."""
    rows = []
    for h in hs:
        t0 = time.perf_counter()
        s = Slot(h)
        th = s.solve(disc, (I,))[0]
        made, out = s.balance[0]
        r = s.at(th, SENSORS)
        rows.append((h, s.n, float(th.max()), r[0], r[3], (made - out) / made, time.perf_counter() - t0))
        if verbose:
            print(f"   cells {h*1e3:.4f} mm: {s.n:8d} cells, hot spot {th.max():.4f} K, sensors 1 and 4 "
                  f"{r[0]:.4f} and {r[3]:.4f} K, balance off by {abs(rows[-1][5]):.1e}  ({rows[-1][6]:.1f} s)")
    if verbose:
        for i in range(1, len(rows)):
            print(f"   change, grid {i} to {i + 1}: hot spot {rows[i][2] - rows[i-1][2]:+.4f} K, "
                  f"sensor 4 {rows[i][4] - rows[i-1][4]:+.4f} K")
    return rows


def ground_truth(disc, sensors=SENSORS, currents=CURRENTS, h=TRUTH_GRID):
    """Clean readings [K] (currents x sensors) and hot spots [K] on the truth grid."""
    return Slot(h).readings(disc, sensors, currents)


def check_zero_strength(h=0.025e-3, I=pb.I_RATED, verbose=True):
    """At k_d = k_eff the slot is Ex07.1's; the finite volumes must reproduce
    its exact series. Returns the largest difference [K]."""
    s = Slot(h)
    th = s.solve((1e-3, 2e-3, 1.5e-3, K_EFF), (I,))[0]
    ex = pb.exact_solution(I, s.xc, s.yc)
    err = float(np.abs(th - ex).max())
    if verbose:
        print(f"   k_d = k_eff, cells {h*1e3:.4f} mm: hot spot {th.max():.4f} K against Ex07.1's exact "
              f"{ex.max():.4f} K; largest difference {err:.1e} K")
    return err


# ------------------------------------------------------- the training data
def sample_discs(n, seed=72):
    """n discs drawn uniformly over the range: (x, y, r, k) per row."""
    rng = np.random.default_rng(seed)
    r = rng.uniform(*R_RANGE, n)
    x = rng.uniform(-pb.A_HALF + r, pb.A_HALF - r)
    y = rng.uniform(-pb.B_HALF + r, pb.B_HALF - r)
    return np.c_[x, y, r, rng.uniform(*K_RANGE, n)]


def simulate(discs, h=REF_H, verbose=True):
    """Clean readings (n, currents, 6 sensors) [K] and hot spots (n, currents)
    [K] of many discs, by the forward model on the grid h. The training data
    of an amortised network; about 10 ms a disc on the engineering grid."""
    s = Slot(h)
    R, H = [], []
    t0 = time.perf_counter()
    for i, d in enumerate(discs):
        r, hot = s.readings(d)
        R.append(r); H.append(hot)
        if verbose and (i + 1) % 1000 == 0:
            print(f"   {i + 1} discs, {time.perf_counter() - t0:.0f} s")
    return np.array(R), np.array(H)


# ----------------------------------------------------------- the test cases
#: rising difficulty: smaller discs, more noise, fewer sensors. The disc is hidden in the truth file.
CASES = [
    dict(name="1", disc=(-1.0e-3, -2.0e-3, 2.0e-3, 0.35), sensors="six", noise=0.05),
    dict(name="2", disc=(1.5e-3, -2.5e-3, 1.75e-3, 0.40), sensors="six", noise=0.10),
    dict(name="3", disc=(-1.5e-3, 3.0e-3, 1.75e-3, 0.35), sensors="six", noise=0.10),
    dict(name="4", disc=(-1.5e-3, -4.0e-3, 2.0e-3, 0.35), sensors="four", noise=0.05),
    dict(name="5", disc=(1.0e-3, -3.0e-3, 1.75e-3, 0.35), sensors="four", noise=0.10),
]
TRUTH_FILE = "MP7.1B_truth.npz"


def write_truth(path=TRUTH_FILE):
    """The test cases: readings at the six sensors (seeded noise), the hidden
    disc, its hot spots and the clean readings from the ground truth; plus the
    convergence table and the zero-strength check."""
    rows = convergence_study(CASES[0]["disc"], verbose=True)
    out = dict(current=np.array(CURRENTS), sensors=SENSORS, four=FOUR)
    for c in CASES:
        n = c["name"]
        clean, hot = ground_truth(c["disc"])
        rng = np.random.default_rng(710 + int(n))
        noisy = clean + rng.normal(0, c["noise"], clean.shape)
        used = np.arange(6) if c["sensors"] == "six" else FOUR
        mask = np.isin(np.arange(6), used).astype(float)
        out.update({f"case{n}_readings": noisy * mask, f"case{n}_mask": mask, f"case{n}_noise": np.array(c["noise"]),
                    f"case{n}_seed": np.array(710 + int(n)), f"case{n}_disc": np.array(c["disc"]),
                    f"case{n}_hot": hot, f"case{n}_clean": clean})
        print(f"   case {n}: disc ({c['disc'][0]*1e3:.2f}, {c['disc'][1]*1e3:.2f}) mm, r {c['disc'][2]*1e3:.2f} mm, "
              f"k {c['disc'][3]:.2f} W/(m K); hot spot at rated {hot[2]:.2f} K")
    healthy, hot0 = ground_truth(None)
    out.update(healthy_readings=healthy, healthy_hot=hot0, cases=np.array([c["name"] for c in CASES]))
    out["conv_h_mm"] = np.array([r[0] * 1e3 for r in rows]); out["conv_cells"] = np.array([r[1] for r in rows])
    out["conv_hot"] = np.array([r[2] for r in rows]); out["conv_sensor4"] = np.array([r[4] for r in rows])
    out["conv_balance"] = np.array([r[5] for r in rows]); out["conv_seconds"] = np.array([r[6] for r in rows])
    out["zero_strength_error_K"] = np.array(check_zero_strength(verbose=False))
    out["params"] = np.array([f"{k} = {v}" for k, v in dict(
        K_EFF="0.70 W/(m K), Ex07.1", SLOT="10 x 20 mm, walls at 90 C", CURRENTS="15, 25, 31.42, 40 A",
        R_RANGE="1.0-2.5 mm", K_RANGE="0.28-0.49 W/(m K), ASSUMED range of aged insulation", SENSORS="six at x = -2, 2 mm, y = -4.5, 0, 4.5 mm",
        NOISE="0.05 or 0.10 K, one standard deviation, ASSUMED for thermocouples").items()])
    out["case_sensors"] = np.array([c["sensors"] for c in CASES])
    out["README"] = np.array(
        "MP7.1B test cases, averaged slot (k_eff 0.70 W/(m K)) with a disc of lower conductivity. "
        "sensors: (x, y) of the six sensors [m]; current: the four currents [A]. "
        "OBSERVED: case<n>_readings[i, j]: rise [K] at current i and sensor j with noise of standard deviation "
        "case<n>_noise [K] from seed case<n>_seed, zero where case<n>_mask is 0 (sensor not fitted). "
        "HIDDEN (for scoring): case<n>_disc = (x [m], y [m], r [m], k [W/(m K)]); case<n>_hot: hot spot at each "
        "current [K]; case<n>_clean: readings without noise [K]. healthy_*: the slot with no disc. "
        "case_sensors: six or four per case. Ground truth on 0.025 mm cells. conv_*: case 1 at rated current on four grids (cell mm, cells, hot spot K, "
        "sensor 4 K, balance residual, solve s). zero_strength_error_K: k_d = k_eff against Ex07.1's exact series.")
    np.savez_compressed(path, **out)
    print("written", path)


if __name__ == "__main__":
    import sys
    print("zero-strength check")
    check_zero_strength()
    print("convergence, case 1, rated current")
    convergence_study(CASES[0]["disc"])
    if "--save" in sys.argv:
        write_truth()
