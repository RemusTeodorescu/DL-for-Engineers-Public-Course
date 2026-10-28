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
the sink in the direction of x, from x = 0 to x = W, and three things change:

* **the air warms.** The heat the fins give up at x is carried downstream by
  the air, whose heat capacity flow is C = m_dot c_p (W/K). Its rise above the
  inlet, θ_a(x), obeys the air's energy balance

      C dθ_a/dx = S · D · h_eff(x, ΔT) (θ(x, 0) - θ_a(x)),     θ_a(0) = 0

  where D is the sink's depth and θ the rise of the base above the inlet air.
* **the fan's coefficient falls along the flow.** The boundary layer on the
  fins grows, so h is highest at the inlet: H_AIR (1 + S B_DROP (1 - 2x/W)),
  50 W/(m²K) at the inlet and 30 at the outlet at S = 1.
* **natural convection** adds S C_NAT (ΔT / L_fin)^(1/4) (MP8.2A's law), with
  ΔT = θ(x, 0) - θ_a(x) the fin root's rise above the LOCAL air: the hotter
  the fins, the stronger the cooling.

The fin-side condition becomes k ∂θ/∂z = h_eff(x, ΔT) ΔT: a coefficient that
varies along x, an air temperature that depends on the solution, and a
coefficient that depends on the solution itself - the problem is
**nonlinear**. Cosines in x are no longer eigenfunctions, superposition fails,
and **no exact solution exists**. Every solver must iterate for every gap.
``S`` is the strength of all three changes: at ``S = 0`` the air keeps its
inlet temperature and h = H_AIR everywhere, which is Ex08's linear problem
with two footprints, and the solver must reproduce Ex08's exact series with
those patches (:func:`check_zero_strength`).

Values: C = 15 W/K (about 0.015 kg/s of air, 4 m/s through the fins' open
area; ASSUMED, chosen so that the air warms by 10 K over the sink at 150 W);
B_DROP = 0.25 (ASSUMED, a developing boundary layer); C_NAT = 1.42
W/(m^(7/4) K^(5/4)) (laminar vertical plate in air, Incropera & DeWitt ch. 9,
from memory, ASSUMED; MP8.2A's value). The conductivity of the aluminium is
kept constant (Ex08's 167 W/(m·K)).

## Two solvers

* the **reference model** (:func:`reference` and :func:`reference_best_gap`):
  finite volumes on an engineering grid, solved for one gap by Newton's method
  (a few iterations, each assembling and factorising the Jacobian); the best
  gap by a bounded one-dimensional search, one nonlinear solve per trial gap.
  Chord iterations with one kept factorisation (:meth:`Plate.solve_chord`)
  are the cheaper alternative and are timed too.
* the **ground truth**: the same scheme, Newton to 1e-10 K, on grids each
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
X_HOT = (0.050, 0.083)    #: m, the hottest point lies under the downstream module, in this span (52.6 to 79.5 mm), at every gap
C_AIR = 15.0              #: W/K, heat capacity flow of the air, m_dot c_p        [ASSUMED]
B_DROP = 0.25             #: relative fall of h from inlet to outlet, ±25 %      [ASSUMED]
C_NAT = 1.42              #: W/(m^(7/4) K^(5/4)), natural convection on the fins, MP8.2A's law   [ASSUMED]
S_AIR = 1.0               #: strength of the change: 0 is Ex08's air (no warming, no fall of h, no natural convection)


def patches(gap):
    """The two footprints (x0, x1, flux) for a gap (m), symmetric about W/2."""
    c = hs.W_BASE / 2
    return ((c - gap / 2 - A_MODULE, c - gap / 2, FLUX), (c + gap / 2, c + gap / 2 + A_MODULE, FLUX))


def h_air(x, dT=0.0, s=S_AIR):
    """The air's coefficient on the fins (W/(m²K)) at x (m) and at a rise dT (K)
    of the fin root above the local air: the fan's part, highest at the inlet
    and falling along the flow, plus natural convection, which grows with the
    fourth root of the rise (MP8.2A's law)."""
    x = np.asarray(x, dtype=float)
    dT = np.maximum(np.asarray(dT, dtype=float), 0.0)
    return hs.H_AIR * (1.0 + s * B_DROP * (1.0 - 2.0 * x / hs.W_BASE)) + s * C_NAT * (dT / hs.L_FIN) ** 0.25


def h_eff(x, dT=0.0, s=S_AIR):
    """The effective coefficient of the finned face at x and a root rise dT:
    Ex08's fin calculation with h(x, dT)."""
    return hs.fin_efficiency(h_air(x, dT, s))[2]


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
    Unknowns u: the rise of every cell above the inlet air, then θ_a at the
    nx+1 faces.

    The equations are  L u + P G(Δ) = b : L is linear (conduction, and the
    air's C dθ_a/dx), G_i = g(x_i, Δ_i) Δ_i is the heat (W per metre of depth)
    bottom cell i gives to the air, with Δ_i = θ_i - (θ_a,i + θ_a,i+1)/2 its
    rise above the local air and g the conductance of half a cell of metal in
    series with h_eff(x_i, Δ_i); P puts +G_i in the cell's balance and
    -S D G_i in the air's. h_eff depends on Δ, so the system is NONLINEAR and
    is solved by Newton's method (:meth:`solve`) or by chord iterations with
    one fixed factorisation (:meth:`solve_chord`)."""

    def __init__(self, nx, nz, s=S_AIR):
        self.nx, self.nz, self.s = nx, nz, s
        dx, dz = hs.W_BASE / nx, hs.T_BASE / nz
        self.dx, self.dz = dx, dz
        self.xe = np.linspace(0, hs.W_BASE, nx + 1)
        self.xc = 0.5 * (self.xe[1:] + self.xe[:-1])
        self.zc = (np.arange(nz) + 0.5) * dz
        k = hs.K_AL
        n = nx * nz
        self.n = n
        na = nx + 1
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
        add(n, n, 1.0)                                       # θ_a(0) = 0
        for i in range(nx):                                  # C (θ_a,i+1 - θ_a,i) ...
            add(n + i + 1, n + i + 1, C_AIR); add(n + i + 1, n + i, -C_AIR)
        N = n + na
        self.L = sp.csr_matrix(sp.coo_matrix((vals, (rows, cols)), shape=(N, N)))
        ib = np.arange(nx)                                   # bottom cells
        # P (N x nx): +1 in the cell's row, -S D in the air row i+1
        self.P = sp.csr_matrix((np.r_[np.ones(nx), np.full(nx, -s * hs.DEPTH)],
                                (np.r_[ib, n + ib + 1], np.r_[ib, ib])), shape=(N, nx))
        # E (nx x N): Δ = E u
        self.E = sp.csr_matrix((np.r_[np.ones(nx), np.full(nx, -0.5), np.full(nx, -0.5)],
                                (np.r_[ib, ib, ib], np.r_[ib, n + ib, n + ib + 1])), shape=(nx, N))
        self.N = N
        self._chord = None

    # --- the nonlinear fin side
    def g(self, D):
        """W/(m K) per metre of depth: half a cell of metal in series with h_eff(x, Δs). The coefficient is
        evaluated at the rise of the SURFACE above the air, Δs = Δ (1/h_eff) / (1/h_eff + dz/2k), not at the
        cell centre's Δ (that would cost the scheme an order); one fixed-point pass finds Δs."""
        r = self.dz / (2 * hs.K_AL)
        he = h_eff(self.xc, D, self.s)
        he = h_eff(self.xc, D / (1.0 + r * he), self.s)
        return self.dx / (1.0 / he + r)

    def G(self, D):
        return self.g(D) * D

    def dG(self, D, eps=1e-4):
        return (self.G(D + eps) - self.G(D - eps)) / (2 * eps)

    def rhs(self, gap):
        b = np.zeros(self.N)
        b[(self.nz - 1) * self.nx: self.n] = _cover(self.xe, gap) * self.dx    # W per metre of depth
        return b

    def residual(self, u, b):
        return self.L @ u + self.P @ self.G(self.E @ u) - b

    def solve(self, gap, u0=None, tol=1e-10, maxit=30):
        """Newton's method: at every iteration the Jacobian L + P diag(G') E is
        assembled and factorised, and u is corrected until no unknown changes
        by more than ``tol`` kelvin. Starts from ``u0`` or from the field of a
        uniform 45 K rise (a warm start from a neighbouring gap converges in
        fewer iterations). Returns the cells' rise (nz, nx) and θ_a (nx+1);
        ``self.iterations`` counts the Newton steps."""
        b = self.rhs(gap)
        u = np.full(self.N, 45.0) if u0 is None else u0.copy()
        if u0 is None:
            u[self.n:] = np.linspace(0, 2 * P_MODULE / C_AIR * self.s, self.nx + 1)
        for it in range(1, maxit + 1):
            D = self.E @ u
            J = (self.L + self.P @ sp.diags(self.dG(D)) @ self.E).tocsc()
            du = spla.spsolve(J, -self.residual(u, b))
            u += du
            if np.abs(du).max() < tol:
                break
        self.iterations, self.u = it, u
        return u[: self.n].reshape(self.nz, self.nx), u[self.n:]

    def solve_chord(self, gap, u0=None, tol=1e-6, maxit=200, D_lin=45.0):
        """Chord iterations: the Jacobian at a uniform rise D_lin is factorised
        ONCE per grid and reused for every iteration and every gap; each
        iteration is one back-substitution. Converges linearly."""
        if self._chord is None:
            D0 = np.full(self.nx, D_lin)
            self._chord = spla.splu((self.L + self.P @ sp.diags(self.dG(D0)) @ self.E).tocsc())
        b = self.rhs(gap)
        u = np.full(self.N, 45.0) if u0 is None else u0.copy()
        for it in range(1, maxit + 1):
            du = self._chord.solve(-self.residual(u, b))
            u += du
            if np.abs(du).max() < tol:
                break
        self.iterations, self.u = it, u
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
        """Heat in from the modules, heat given by the fins, heat taken by the air (C θ_a at the outlet), W for the depth."""
        D = th[0] - 0.5 * (ta[1:] + ta[:-1])
        return dict(heat_in=2 * P_MODULE, heat_fins=hs.DEPTH * np.sum(self.G(D)), heat_air=C_AIR * ta[-1])


# ------------------------------------------------------- the reference model
REF_GRID = (50, 5)        #: cells of 2 mm, the coarsest that meets 0.1 K at every gap (the convergence study)
REF_TOL = 1e-3            #: K, Newton stops when no unknown changes by more than this
_CACHE = {}


def _plate(grid, s=S_AIR):
    key = (grid, s)
    if key not in _CACHE:
        _CACHE[key] = Plate(*grid, s=s)
    return _CACHE[key]


def reference(gap, grid=REF_GRID, method="newton"):
    """The reference model for one new gap (m): finite volumes on the
    engineering grid, Newton's method from a cold start (``method="chord"``:
    chord iterations with the one factorisation kept from earlier calls).
    Returns the hottest point (K), its x (m), the fields, the iterations and the
    wall-clock time (s)."""
    m = _plate(grid)
    t0 = time.perf_counter()
    th, ta = (m.solve if method == "newton" else m.solve_chord)(gap, tol=REF_TOL)
    hot, xh = m.hottest(th, gap)
    return dict(hottest=hot, x_hot=xh, theta=th, theta_air=ta, x=m.xc, z=m.zc, iterations=m.iterations,
                seconds=time.perf_counter() - t0)


def reference_best_gap(grid=REF_GRID, xatol=1e-4, method="newton"):
    """The best gap by the reference model: a bounded scalar search (Brent) of
    the hottest point over the gap, one nonlinear solve per trial gap, each
    warm-started from the previous trial's field. Returns the gap (m), the
    hottest point there (K), the number of solves and of iterations, and the
    time (s)."""
    m = _plate(grid)
    state = dict(u=None, its=0)
    def f(g):
        th, ta = (m.solve if method == "newton" else m.solve_chord)(g, u0=state["u"], tol=REF_TOL)
        state["u"] = m.u; state["its"] += m.iterations
        return m.hottest(th, g)[0]
    t0 = time.perf_counter()
    res = minimize_scalar(f, bounds=(G_MIN, G_MAX), method="bounded", options=dict(xatol=xatol))
    return dict(gap=float(res.x), hottest=float(res.fun), solves=int(res.nfev), iterations=state["its"],
                seconds=time.perf_counter() - t0)


# -------------------------------------------------------- the ground truth
TRUTH_GRIDS = ((100, 10), (200, 20), (400, 40), (800, 80))


def convergence_study(gap=G_TEST[1], grids=TRUTH_GRIDS, verbose=True):
    """Two quantities at one gap on grids each twice as fine: the plate's mean
    rise, a smooth quantity whose changes show the scheme's order (second), and
    the hottest point, the scored quantity, read by a parabola through the
    hottest cells (that reading adds a small noise of a few 1e-04 K, so its
    changes do not fall cleanly). The last two hottest points are extrapolated
    by Richardson with order 2."""
    rows = []
    for g in grids:
        t0 = time.perf_counter()
        m = Plate(*g); th, ta = m.solve(gap)
        hot, xh = m.hottest(th, gap)
        b = m.balance(th, ta)
        rows.append((g, hot, xh, b["heat_air"] / b["heat_in"] - 1, time.perf_counter() - t0, float(th.mean())))
        if verbose:
            print(f"   {g[0]:4d} x {g[1]:3d} cells: hottest {hot:.4f} K at x = {xh*1e3:.2f} mm, plate's mean rise "
                  f"{th.mean():.6f} K; air takes {b['heat_air']:.4f} W of {b['heat_in']:.0f} W ({rows[-1][4]:.1f} s)")
    ext = rows[-1][1] + (rows[-1][1] - rows[-2][1]) / 3
    dm = np.abs(np.diff([r[5] for r in rows]))
    order = float(np.log2(dm[-2] / dm[-1]))
    if verbose:
        d = np.abs(np.diff([r[1] for r in rows]))
        print("   change of the mean rise between grids:", np.array2string(dm, precision=7),
              " ratios", np.array2string(dm[:-1] / dm[1:], precision=2), f" observed order {order:.2f}")
        print("   change of the hottest between grids:", np.array2string(d, precision=5))
        print(f"   extrapolated hottest {ext:.4f} K")
    return rows, ext, order


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
    rows, ext, order = convergence_study(verbose=True)
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
    out["conv_mean_rise"] = np.array([r[5] for r in rows]); out["conv_order"] = np.array(order)
    out["zero_strength_error_K"] = np.array(check_zero_strength(verbose=False))
    # the same curve with Ex08's air (S = 0), for comparison
    m0 = Plate(*TRUTH_GRIDS[2], s=0.0)
    out["hottest_S0_400"] = np.array([m0.hottest(m0.solve(g)[0], g)[0] for g in gaps])
    out["params"] = np.array([f"{k} = {v}" for k, v in dict(
        P_MODULE="75 W each", A_MODULE="15 mm wide, 100 mm deep", GAP="0-40 mm, symmetric about x = 50 mm",
        C_AIR="15 W/K, ASSUMED", B_DROP="0.25: h from 50 to 30 W/(m2K) along x, ASSUMED", S_AIR=S_AIR,
        C_NAT="1.42 W/(m^1.75 K^1.25): natural convection h = C_NAT (dT/L_fin)^(1/4), MP8.2A's law, ASSUMED",
        H_AIR=f"{hs.H_AIR} W/(m2K) mean", K=f"{hs.K_AL} W/(m K)").items()])
    out["README"] = np.array(
        "MP8.1B ground truth. gaps: m; hottest_400/_800: hottest rise of the module face above the inlet air on 400x40 and "
        "800x80 cells, K; hottest: Richardson-extrapolated (the score); best_gap (m) and best_hottest (K) on 800x80; "
        "field_<g>mm: (nz, nx) cell rises on 400x40 at the test gaps, K, cell centres x, z (m); air_<g>mm: air rise at the "
        "cell faces x_faces, K; conv_*: convergence at 17.5 mm (hottest, plate's mean rise, observed order of the mean, balance); zero_strength_error_K: S = 0 against Ex08's exact series; "
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
