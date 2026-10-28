r"""MP8.1B — Two modules on one sink, with the air warming as it crosses the
sink. The physics, the reference model and the ground truth.

*Deep Learning for Engineering* — MSc, Aalborg University.
Remus Teodorescu (ret@et.aau.dk), with support from Research Assistant
Noman Khan (nomank@energy.aau.dk).

This module extends Ex08's ``heatsink.py`` (course policy C10 rule 1: one set
of physics and constants for the exercise and its mini project) and is used by
the notebook of MP8.1B (course policy C14).

## What changes against Ex08, and why

Two modules of 75 W each, 15 mm wide, sit on Ex08's base plate symmetrically
about its centre, a gap ``g`` apart (0 to 40 mm). Ex08 cools the fin side with
one coefficient in air of one temperature. Here the fan blows the air across
the sink in the direction of x, from x = 0 to x = W, and two things change
along its path:

* **the air warms.** The heat the fins give up at x is carried downstream by
  the air, whose heat capacity flow is C = m_dot c_p (W/K). Its rise above the
  inlet, θ_a(x), obeys the air's energy balance

      C dθ_a/dx = S · D · h_eff(x) (θ(x, 0) - θ_a(x)),     θ_a(0) = 0

  where D is the sink's depth and θ the rise of the base above the inlet air.
* **the coefficient falls.** The boundary layer on the fins grows along the
  flow, so h is highest at the inlet. A linear fall is assumed,

      h(x) = H_AIR (1 + S · B_DROP (1 - 2 x / W))

  50 W/(m²K) at the inlet and 30 at the outlet at S = 1, the mean unchanged.

The fin-side condition becomes k ∂θ/∂z = h_eff(x)(θ - θ_a(x)), with a
coefficient that varies along x and an air temperature that depends on the
solution itself. Cosines in x are no longer eigenfunctions of the problem, the
series of Ex08 does not separate, and **no exact solution exists**. ``S`` is
the strength of the change: at ``S = 0`` the air keeps its inlet temperature
and h = H_AIR everywhere, which is Ex08's problem with two footprints, and the
solver must reproduce Ex08's exact series with those patches
(:func:`check_zero_strength`).

Values: C = 15 W/K (about 0.015 kg/s of air, 4 m/s through the fins' open
area; ASSUMED, chosen so that the air warms by 10 K over the sink at 150 W);
B_DROP = 0.25 (ASSUMED, a developing boundary layer).

## Two solvers

* the **reference model** (:func:`reference` and :func:`reference_best_gap`):
  finite volumes on an engineering grid, solved for one gap; the best gap by a
  bounded one-dimensional search, which solves the plate once per trial gap.
* the **ground truth** (:func:`ground_truth`): the same scheme on grids each
  twice as fine, converged and extrapolated (:func:`convergence_study`).
"""

from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import minimize_scalar

import heatsink as hs

# ------------------------------------------------------------------ the data
P_MODULE = 75.0           #: W, each module
A_MODULE = 0.015          #: m, each module's width (half of Ex08's footprint)
FLUX = P_MODULE / (A_MODULE * hs.DEPTH)   #: W/m², 50 kW/m², Ex08's flux
G_MIN, G_MAX = 0.0, 0.040   #: m, the range of the gap, an input of the network
G_TEST = (0.005, 0.0175, 0.0325)   #: m, gaps whose fields are kept in the truth file
X_HOT = (0.045, 0.095)    #: m, the hottest point lies under the downstream module, in this span, at every gap
C_AIR = 15.0              #: W/K, heat capacity flow of the air, m_dot c_p        [ASSUMED]
B_DROP = 0.25             #: relative fall of h from inlet to outlet, ±25 %      [ASSUMED]
S_AIR = 1.0               #: strength of the change: 0 is Ex08's air


def patches(gap):
    """The two footprints (x0, x1, flux) for a gap (m), symmetric about W/2."""
    c = hs.W_BASE / 2
    return ((c - gap / 2 - A_MODULE, c - gap / 2, FLUX), (c + gap / 2, c + gap / 2 + A_MODULE, FLUX))


def h_air(x, s=S_AIR):
    """The air's coefficient on the fins at x (m): highest at the inlet."""
    return hs.H_AIR * (1.0 + s * B_DROP * (1.0 - 2.0 * np.asarray(x, dtype=float) / hs.W_BASE))


def h_eff(x, s=S_AIR):
    """The effective coefficient of the finned face at x: Ex08's fin calculation with h(x)."""
    return hs.fin_efficiency(h_air(x, s))[2]


def flux_top(x, gap):
    """The module flux (W/m²) on the top face at x."""
    x = np.asarray(x, dtype=float); q = np.zeros_like(x)
    for lo, hi, f in patches(gap):
        q = q + np.where((x >= lo) & (x <= hi), f, 0.0)
    return q


def _cover(edges, gap):
    """The flux averaged over cells with the given x edges: each cell gets the
    share of it the footprints cover (W/m²). Keeps second order when a
    footprint's edge falls inside a cell."""
    q = np.zeros(len(edges) - 1)
    for lo, hi, f in patches(gap):
        q += f * np.clip(np.minimum(edges[1:], hi) - np.maximum(edges[:-1], lo), 0, None) / np.diff(edges)
    return q


# ------------------------------------------------------------- finite volumes
class Plate:
    """Finite volumes on nx by nz cells of the base plate, coupled to the air's
    energy balance along the fin side (one air temperature per cell face in x).
    Unknowns: the rise of every cell above the inlet air, then θ_a at the nx+1
    faces. Built once per grid; :meth:`solve` takes a gap."""

    def __init__(self, nx, nz, s=S_AIR):
        self.nx, self.nz, self.s = nx, nz, s
        dx, dz = hs.W_BASE / nx, hs.T_BASE / nz
        self.dx, self.dz = dx, dz
        self.xe = np.linspace(0, hs.W_BASE, nx + 1)
        self.xc = 0.5 * (self.xe[1:] + self.xe[:-1])
        self.zc = (np.arange(nz) + 0.5) * dz
        k = hs.K_AL
        n = nx * nz
        I = lambda i, j: j * nx + i                          # cell (i across, j up)
        rows, cols, vals = [], [], []
        def add(r, c, v):
            rows.append(r); cols.append(c); vals.append(v)
        gx, gz = k * dz / dx, k * dx / dz                    # W/(m K) per metre of depth
        for j in range(nz):
            for i in range(nx):
                p = I(i, j)
                if i > 0: add(p, p, gx); add(p, I(i - 1, j), -gx)
                if i < nx - 1: add(p, p, gx); add(p, I(i + 1, j), -gx)
                if j > 0: add(p, p, gz); add(p, I(i, j - 1), -gz)
                if j < nz - 1: add(p, p, gz); add(p, I(i, j + 1), -gz)
        # the fin side: each bottom cell to the air at its centre, θ_a there the mean of its two faces
        he = h_eff(self.xc, s)
        g = dx / (1.0 / he + dz / (2 * k))                   # W/(m K) per metre of depth
        self.g = g
        na = nx + 1
        for i in range(nx):
            p = I(i, 0)
            add(p, p, g[i]); add(p, n + i, -0.5 * g[i]); add(p, n + i + 1, -0.5 * g[i])
        # the air: θ_a(0) = 0; C (θ_a,i+1 - θ_a,i) = S D g_i (θ_i - (θ_a,i + θ_a,i+1)/2)
        add(n, n, 1.0)
        D = hs.DEPTH
        for i in range(nx):
            r = n + i + 1
            add(r, n + i + 1, C_AIR + 0.5 * s * D * g[i]); add(r, n + i, -C_AIR + 0.5 * s * D * g[i])
            add(r, I(i, 0), -s * D * g[i])
        self.A = sp.csc_matrix(sp.coo_matrix((vals, (rows, cols)), shape=(n + na, n + na)))
        self.lu = spla.splu(self.A)
        self.n = n

    def solve(self, gap):
        """Returns the rise of every cell (nz, nx) and θ_a at the faces (nx+1), K."""
        b = np.zeros(self.A.shape[0])
        b[(self.nz - 1) * self.nx: self.n] = _cover(self.xe, gap) * self.dx    # W per metre of depth
        u = self.lu.solve(b)
        return u[: self.n].reshape(self.nz, self.nx), u[self.n:]

    def top_face(self, th, gap):
        """The rise on the module face at the cell centres: the top cell plus half a cell of the flux's slope."""
        return th[-1] + _cover(self.xe, gap) * self.dz / (2 * hs.K_AL)

    def hottest(self, th, gap):
        """The hottest point of the module face (K) and its x (m), by a parabola
        through the hottest cell and its two neighbours."""
        f = self.top_face(th, gap)
        i = int(np.clip(np.argmax(f), 1, self.nx - 2))
        a, b, c = f[i - 1], f[i], f[i + 1]
        den = a - 2 * b + c
        s = 0.0 if den == 0 else np.clip(0.5 * (a - c) / den, -1, 1)
        return float(b - 0.25 * (a - c) * s), float(self.xc[i] + s * self.dx)

    def balance(self, th, ta):
        """Heat in from the modules, heat taken by the air (C θ_a at the outlet), W for the depth."""
        q_in = 2 * P_MODULE
        q_fins = hs.DEPTH * np.sum(self.g * (th[0] - 0.5 * (ta[1:] + ta[:-1])))
        return dict(heat_in=q_in, heat_fins=q_fins, heat_air=C_AIR * ta[-1])


# ------------------------------------------------------- the reference model
REF_GRID = (50, 5)        #: cells of 2 mm, the coarsest that meets 0.1 K at every gap (the convergence study)
_CACHE = {}


def _plate(grid, s=S_AIR):
    key = (grid, s)
    if key not in _CACHE:
        _CACHE[key] = Plate(*grid, s=s)
    return _CACHE[key]


def reference(gap, grid=REF_GRID):
    """The reference model for one gap (m): the hottest point (K), its x (m),
    and the wall-clock time (s). The matrix does not depend on the gap and is
    factorised once, as an engineer's solver would; a new gap is one solve."""
    m = _plate(grid)
    t0 = time.perf_counter()
    th, ta = m.solve(gap)
    hot, xh = m.hottest(th, gap)
    return dict(hottest=hot, x_hot=xh, theta=th, theta_air=ta, x=m.xc, z=m.zc, seconds=time.perf_counter() - t0)


def reference_best_gap(grid=REF_GRID, xatol=1e-4):
    """The best gap by the reference model: a bounded scalar search (Brent) of
    the hottest point over the gap, one plate solve per trial. Returns the gap
    (m), the hottest point there (K), the number of solves and the time (s)."""
    m = _plate(grid)
    t0 = time.perf_counter()
    res = minimize_scalar(lambda g: m.hottest(m.solve(g)[0], g)[0], bounds=(G_MIN, G_MAX),
                          method="bounded", options=dict(xatol=xatol))
    return dict(gap=float(res.x), hottest=float(res.fun), solves=int(res.nfev), seconds=time.perf_counter() - t0)


# -------------------------------------------------------- the ground truth
TRUTH_GRIDS = ((100, 10), (200, 20), (400, 40), (800, 80))


def convergence_study(gap=G_TEST[1], grids=TRUTH_GRIDS, verbose=True):
    """The hottest point at one gap on grids each twice as fine; second order
    expected; the last two extrapolated by Richardson."""
    rows = []
    for g in grids:
        t0 = time.perf_counter()
        m = Plate(*g); th, ta = m.solve(gap)
        hot, xh = m.hottest(th, gap)
        b = m.balance(th, ta)
        rows.append((g, hot, xh, b["heat_air"] / b["heat_in"] - 1, time.perf_counter() - t0))
        if verbose:
            print(f"   {g[0]:4d} x {g[1]:3d} cells: hottest {hot:.4f} K at x = {xh*1e3:.2f} mm; air takes "
                  f"{b['heat_air']:.4f} W of {b['heat_in']:.0f} W ({rows[-1][4]:.1f} s)")
    ext = rows[-1][1] + (rows[-1][1] - rows[-2][1]) / 3
    if verbose:
        d = np.abs(np.diff([r[1] for r in rows]))
        print("   change of the hottest between grids:", np.array2string(d, precision=5),
              " ratios", np.array2string(d[:-1] / d[1:], precision=2))
        print(f"   extrapolated hottest {ext:.4f} K")
    return rows, ext


def truth_curve(gaps, grid=TRUTH_GRIDS[2], fine=TRUTH_GRIDS[3]):
    """The hottest point against the gap on two grids, and its Richardson
    extrapolation (the scoring value)."""
    m1, m2 = Plate(*grid), Plate(*fine)
    h1 = np.array([m1.hottest(m1.solve(g)[0], g)[0] for g in gaps])
    h2 = np.array([m2.hottest(m2.solve(g)[0], g)[0] for g in gaps])
    return h1, h2, h2 + (h2 - h1) / 3


def truth_best_gap(grid=TRUTH_GRIDS[3]):
    """The best gap on the finest grid, to 0.01 mm, with the Richardson-corrected
    hottest point there."""
    m = Plate(*grid)
    res = minimize_scalar(lambda g: m.hottest(m.solve(g)[0], g)[0], bounds=(G_MIN, G_MAX),
                          method="bounded", options=dict(xatol=1e-5))
    return float(res.x), float(res.fun)


def check_zero_strength(grid=(800, 80), gaps=(0.0, 0.0175, 0.035), verbose=True):
    """At S = 0 the air keeps its inlet temperature and h = H_AIR: Ex08's
    problem with two footprints, whose exact series (hs.exact with patches) the
    finite volumes must reproduce. Returns the largest difference over the
    plate (K)."""
    m = Plate(*grid, s=0.0)
    err = 0.0
    X, Z = np.meshgrid(m.xc, m.zc)
    for g in gaps:
        th, _ = m.solve(g)
        ex = hs.exact(X, Z, patches=patches(g))
        err = max(err, float(np.abs(th - ex).max()))
    if verbose:
        print(f"   S = 0, {grid[0]} x {grid[1]} cells: largest difference from Ex08's exact series {err:.2e} K")
    return err


def ex08_series(x, z, gap, n=200):
    """Ex08's exact steady rise (K) with the two footprints - the problem at
    S = 0, air at the inlet temperature and h = H_AIR everywhere."""
    return hs.exact(x, z, n=n, patches=patches(gap))


# ------------------------------------------------------- the training points
def sample_interior(n, seed=81):
    """n points (x, z, gap) in the plate, in metres."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0, hs.W_BASE, n), rng.uniform(0, hs.T_BASE, n), rng.uniform(G_MIN, G_MAX, n)


def sample_top(n, seed=82, edges=0.2):
    """n points (x, gap) on the module face; a share ``edges`` within 1.5 mm of
    a footprint's edge, where the flux steps."""
    rng = np.random.default_rng(seed)
    g = rng.uniform(G_MIN, G_MAX, n)
    x = rng.uniform(0, hs.W_BASE, n)
    k = int(n * edges)
    e = np.array([np.array(patches(gg))[:, :2].ravel()[rng.integers(0, 4)] for gg in g[:k]])
    x[:k] = e + 3e-3 * (rng.random(k) - 0.5)
    return np.clip(x, 0, hs.W_BASE), g


def sample_fin_side(n, seed=83):
    """n points (x, gap) on the fin side."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0, hs.W_BASE, n), rng.uniform(G_MIN, G_MAX, n)


# ----------------------------------------------------------- the truth file
TRUTH_FILE = "MP8.1B_truth.npz"


def write_truth(path=TRUTH_FILE):
    """The hottest point against the gap (0 to 40 mm every 0.5 mm) on the
    400 x 40 and 800 x 80 grids and extrapolated; the best gap; the fields at
    the three test gaps on 400 x 40; the convergence table; the zero-strength
    check; every parameter with its unit."""
    rows, ext = convergence_study(verbose=True)
    gaps = np.round(np.arange(0.0, 0.040 + 1e-9, 0.0005), 6)
    h1, h2, he = truth_curve(gaps)
    gb, hb = truth_best_gap()
    out = dict(gaps=gaps, hottest_400=h1, hottest_800=h2, hottest=he, best_gap=np.array(gb), best_hottest=np.array(hb),
               test_gaps=np.array(G_TEST))
    m = Plate(*TRUTH_GRIDS[2])
    out["x"], out["z"] = m.xc, m.zc
    for g in G_TEST:
        th, ta = m.solve(g)
        key = f"{g*1e3:.1f}mm"
        out[f"field_{key}"] = th; out[f"air_{key}"] = ta
    out["x_faces"] = m.xe
    out["conv_grids"] = np.array([r[0] for r in rows], dtype=float)
    out["conv_hottest"] = np.array([r[1] for r in rows]); out["conv_balance"] = np.array([r[3] for r in rows])
    out["conv_hottest_extrapolated"] = np.array(ext)
    out["zero_strength_error_K"] = np.array(check_zero_strength(verbose=False))
    # the same curve with Ex08's air (S = 0), for comparison
    m0 = Plate(*TRUTH_GRIDS[2], s=0.0)
    out["hottest_S0_400"] = np.array([m0.hottest(m0.solve(g)[0], g)[0] for g in gaps])
    out["params"] = np.array([f"{k} = {v}" for k, v in dict(
        P_MODULE="75 W each", A_MODULE="15 mm wide, 100 mm deep", GAP="0-40 mm, symmetric about x = 50 mm",
        C_AIR="15 W/K, ASSUMED", B_DROP="0.25: h from 50 to 30 W/(m2K) along x, ASSUMED", S_AIR=S_AIR,
        H_AIR=f"{hs.H_AIR} W/(m2K) mean", K=f"{hs.K_AL} W/(m K)").items()])
    out["README"] = np.array(
        "MP8.1B ground truth. gaps: m; hottest_400/_800: hottest rise of the module face above the inlet air on 400x40 and "
        "800x80 cells, K; hottest: Richardson-extrapolated (the score); best_gap (m) and best_hottest (K) on 800x80; "
        "field_<g>mm: (nz, nx) cell rises on 400x40 at the test gaps, K, cell centres x, z (m); air_<g>mm: air rise at the "
        "cell faces x_faces, K; conv_*: convergence at 17.5 mm; zero_strength_error_K: S = 0 against Ex08's exact series; "
        "hottest_S0_400: the curve with Ex08's air; params: the data.")
    np.savez_compressed(path, **out)
    print("written", path)


if __name__ == "__main__":
    print("zero-strength check")
    check_zero_strength()
    print("convergence at a 17.5 mm gap")
    convergence_study()
    import sys
    if "--save" in sys.argv:
        write_truth()
