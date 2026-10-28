r"""MP10.2A — The channel across its height, and the electrode under it. The
physics, the reference model and the ground truth.

*Deep Learning for Engineering* — MSc, Aalborg University.
Remus Teodorescu (ret@et.aau.dk), with support from Research Assistant
Noman Khan (nomank@energy.aau.dk).

This module extends Ex10.2's ``problem.py`` (course policy C10 rule 1: one set
of physics and constants for the exercise and its mini project) and is used by
the notebook of MP10.2A (course policy C14). It takes over the model of
``archive/miniprojects/ex102_truth.py`` (MP10.2A) and writes its solver again,
faster: Newton's method on the current instead of bisection.

## What changes against Ex10.2, and why

Ex10.2's channel is one-dimensional: the steam fraction is one number at each
``x``, as if the gas were mixed across the 1 mm channel, and the steam the
porous fuel electrode cannot bring to the reaction is priced by a
**gas-transport loss** with an estimated limiting current density
(4 A/cm² at the inlet). Here both are resolved:

* the steam fraction ``y(x, z)`` across the channel's height, carried along by
  Poiseuille's profile and diffused across, as in Ex09.2;
* the steam through the 0.5 mm porous electrode to the reaction at the
  electrolyte, by diffusion with an effective diffusivity ``D_eff``. Steam in,
  hydrogen out, one for one, and no reaction inside the electrode: its profile
  across is a straight line, so the steam at the reaction site is
  ``y_r = y_w - K_E i`` with ``y_w`` the steam at the electrode's surface and
  ``K_E = delta / (2 F c D_eff)`` per A/cm².

The charge transfer is evaluated **at the reaction site**: the Nernst
potential at ``y_r``, the activation and the ohmic losses of Ex10.2, and no
gas-transport loss term, because the gas transport is now resolved. The
limiting current is no longer an input: it is ``y_w / K_E``, 3.94 A/cm² at the
inlet, which is where the estimate ``D_eff`` came from. The cell is held at
800 degC (isothermal): the heat of Ex10.2 is left out so that the project is
about the gas.

## The problem, scaled

    w(Z) y_X = alpha y_ZZ                         in the channel, 0 < X < 1, 0 < Z < 1
    y = 0.90                                     at X = 0 (the inlet)
    y_Z = beta i                                 at Z = 0 (the electrode's surface: steam taken)
    y_Z = 0                                      at Z = 1 (the interconnect)
    V = E(y_r) + (RT/F) asinh(i / 2 i_0) + i ASR,   y_r = y(X, 0) - K_E i     at every X

with ``X = x/L``, ``Z = z/h``, ``w = 6 Z (1 - Z)`` Poiseuille's profile over the
mean speed, ``alpha = L D / (U h^2)`` = 80 and ``beta = h / (2 F c D)`` per
A/cm². Integrated over the height with the flow's weight, the equation gives
the **balance of the mean**, ``d ybar/dX = - K_I i`` with ``K_I = alpha beta``,
Ex10.2's steam balance without axial diffusion.

## No exact solution

The current at each ``x`` is the root of the nonlinear charge transfer at the
reaction site, which depends on the unknown at the wall; the wall condition
therefore depends on the solution, and the profile across the channel changes
along it. No closed form exists.

## The zero-strength check

The strength ``S`` multiplies both transport resistances: the electrode's drop
``K_E`` and the channel's resistance across, ``1/alpha``. ``S = 1`` is this
project. As ``S`` goes to zero the gas is mixed across the channel and reaches
the reaction site without loss, and the model becomes Ex10.2's
one-dimensional steam balance and charge transfer at 800 degC with the
gas-transport loss switched off (``i_L`` infinite) and without axial
diffusion: an ordinary differential equation, solved here by RK4 to rounding
(:func:`one_d`). :func:`check_zero_strength` shows the march approaching it in
proportion to ``S``.

## Two solvers

* the **reference model** (:func:`reference`): Crank–Nicolson along the
  channel with Rannacher's start, central differences across, the current by
  Newton's method at the wall node, on an engineering grid. It is what the
  network is timed against.
* the **ground truth** (:func:`ground_truth`): the same scheme on grids each
  twice as fine, converged (:func:`convergence_study`).
"""

from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import problem as pb

# ------------------------------------------------------------------ the data
F, R = pb.FARADAY, pb.R_GAS
T_CELL = 1073.15        #: K, the cell held at 800 degC (isothermal)
DELTA = 0.5e-3          #: m, thickness of the porous fuel electrode   [estimate]
D_EFF = 1.0e-5          #: m^2/s, effective diffusivity of steam in it (porosity over tortuosity times D)   [estimate]
ALPHA = pb.L_CHANNEL * pb.D_GAS / (pb.U_GAS * pb.H_CHANNEL ** 2)    #: 80: diffusion across over convection along
BETA = pb.H_CHANNEL * 1e4 / (2 * F * pb.C_TOT * pb.D_GAS)           #: wall slope dy/dZ per (A/cm^2), 5.70e-03
K_E = DELTA * 1e4 / (2 * F * pb.C_TOT * D_EFF)                       #: steam drop across the electrode per (A/cm^2), 0.228
K_I = ALPHA * BETA                                                   #: steam used per (A/cm^2) along X, Ex10.2's K_I
V_LOW, V_HIGH = pb.V_LOW, pb.V_HIGH                                  #: V, the range of cell voltages, a network input
V_TEST = (1.18, 1.27, 1.33)                                          #: V, the voltages the answers are scored at
RT_F = R * T_CELL / F
ASR_C = float(pb.asr(T_CELL))                                        #: ohm cm^2 at 800 degC, 0.15
E0 = 1.253 - 2.4516e-4 * T_CELL + 0.5 * RT_F * np.log(pb.P_O2 ** 0.5)   #: the Nernst potential's constant part [V]

#: The agreed accuracy (part A): half of what the plug-flow hand calculation misses (0.13 to 0.19 %, 5e-04 to
#: 1.3e-03 and 1.7e-03 to 3.1e-03 over the range), so that the profile across the channel must be resolved.
TOL_CURRENT = 0.001     #: relative, 0.1 % of the cell current
TOL_MEAN = 5e-4         #: in the steam fraction, the outlet's mixing-cup mean
TOL_SITE = 1e-3         #: in the steam fraction at the reaction site, beyond the first 5 mm
X_SCORE = 0.05          #: the reaction site is scored beyond the first 5 mm, as in Ex09.2 and MP9.2A: at the inlet
                        #: corner the flat feed meets the electrode's flux and the field is singular


def phi(Z):
    """The fully developed profile across the channel for a uniform wall flux,
    Ex09.2's: phi = Z^3 - Z^4/2 - Z + 13/35, slope -1 at Z = 0 and 0 at Z = 1,
    flow-weighted mean 0. y = ybar - beta i phi(Z) holds far from the inlet if
    i changes slowly."""
    return Z ** 3 - Z ** 4 / 2 - Z + 13.0 / 35.0


def v_site(i, y_r, s=1.0):
    """The cell voltage [V] that drives current density ``i`` [A/cm^2] when the
    steam fraction at the reaction site is ``y_r``: the Nernst potential there,
    the activation loss and the ohmic loss of Ex10.2 at 800 degC; no
    gas-transport loss term. NumPy or torch."""
    return pb.nernst(y_r, T_CELL) + RT_F * pb._asinh(i / (2 * pb.I_0)) + i * ASR_C


def current_at(V, y_w, s=1.0, i0=None, iters=40):
    """The current density [A/cm^2] at cell voltage ``V`` where the steam at the
    electrode's surface is ``y_w``: the root of V = v_site(i, y_w - s K_E i) on
    [0, y_w / (s K_E)), by Newton's method kept inside the bracket (bisection
    when a step leaves it). Also returns di/dy_w. NumPy, any shape."""
    y_w = np.asarray(y_w, float)
    ke = s * K_E
    hi = y_w / ke * (1 - 1e-12) if ke > 0 else np.full_like(y_w, np.inf)
    lo = np.zeros_like(y_w)
    i = np.zeros_like(y_w) + (0.5 if i0 is None else i0)
    i = np.where(i < hi, i, 0.5 * (lo + np.minimum(hi, 10.0)))
    for _ in range(iters):
        yr = y_w - ke * i
        f = E0 + 0.5 * RT_F * np.log((1 - yr) / yr) + RT_F * np.arcsinh(i / (2 * pb.I_0)) + i * ASR_C - V
        dEdy = -0.5 * RT_F * (1 / (1 - yr) + 1 / yr)
        df = -ke * dEdy + RT_F / np.sqrt((i / (2 * pb.I_0)) ** 2 + 1) / (2 * pb.I_0) + ASR_C
        lo = np.where(f < 0, i, lo); hi = np.where(f > 0, i, hi)
        new = i - f / df
        bad = (new <= lo) | (new >= hi) | ~np.isfinite(new)
        new = np.where(bad, 0.5 * (lo + np.minimum(hi, lo + 20.0)), new)
        if np.all(np.abs(new - i) < 1e-14):
            i = new
            break
        i = new
    yr = y_w - ke * i
    dEdy = -0.5 * RT_F * (1 / (1 - yr) + 1 / yr)
    df = -ke * dEdy + RT_F / np.sqrt((i / (2 * pb.I_0)) ** 2 + 1) / (2 * pb.I_0) + ASR_C
    return i, -dEdy / df


def site(y_w, i, s=1.0):
    """The steam fraction at the reaction site."""
    return np.asarray(y_w) - s * K_E * np.asarray(i)


# ---------------------------------------------------------- the march
class March:
    """Crank–Nicolson along the channel on an nx by nz grid (X and Z, walls
    included), Rannacher's start (the first ``start`` steps as two fully
    implicit half steps each), the current by Newton's method at the wall node.
    Only the wall row is nonlinear, so each Newton step is one solve with the
    matrix factorised once and a rank-one correction (Sherman–Morrison). The
    velocity row is normalised so that its trapezoid integral is exactly one:
    the grid carries exactly the flow fed, and the balance of the mean holds
    on any grid. ``s`` is the strength of the transport resistances."""

    def __init__(self, nx, nz, s=1.0, start=2):
        self.nx, self.nz, self.s, self.start = nx, nz, s, start
        self.X, self.Z = np.linspace(0, 1, nx), np.linspace(0, 1, nz)
        self.hx, self.hz = 1.0 / (nx - 1), 1.0 / (nz - 1)
        self.alpha = ALPHA / s
        self.beta = BETA * s
        A = sp.diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(nz, nz)).tolil()
        A[0, 1] = 2.0; A[-1, -2] = 2.0                        # both walls by a ghost node
        self.A = A.tocsc() / self.hz ** 2
        self.e0 = np.zeros(nz); self.e0[0] = 1.0
        self.cw = 2.0 * self.beta / self.hz                   # the wall flux through its ghost node: -cw i on row 0
        w = 6 * self.Z * (1 - self.Z) * self.hz; w[[0, -1]] *= 0.5
        self.flow = w.sum()
        self.wmean = w / w.sum()                              # the mixing-cup mean's weights
        M = sp.diags(6 * self.Z * (1 - self.Z) / self.flow).tocsc()
        self.M = M
        c = 0.5 * self.hx * self.alpha                        # CN and the half steps of the start share M - c A
        self.lu = spla.splu((M - c * self.A).tocsc())
        self.Rm = (M + c * self.A).tocsc()
        self.c = c
        self.w0 = self.lu.solve(self.e0)

    def _solve(self, rhs, V, guess, i_guess, iters=6):
        """Solve (M - c A) y + c cw i(y0) e0 = rhs by Newton's method on y0."""
        base = self.lu.solve(rhs)
        y, i = guess, i_guess
        for _ in range(iters):
            i, di = current_at(V, y[0], self.s, i0=i)
            d = self.c * self.cw * di
            shift = self.c * self.cw * (i - di * y[0])
            x = base - shift * self.w0
            y_new = x - d * x[0] / (1.0 + d * self.w0[0]) * self.w0
            if abs(y_new[0] - y[0]) < 1e-14:
                y = y_new
                break
            y = y_new
        i, _ = current_at(V, y[0], self.s, i0=i)
        return y, float(i)

    def run(self, V):
        """March at cell voltage V [V]. Returns y (nz, nx) and i (nx)."""
        y = np.full(self.nz, pb.Y_IN)
        i, _ = current_at(V, y[0], self.s)
        i = float(i)
        Y = np.empty((self.nz, self.nx)); Y[:, 0] = y
        I = np.empty(self.nx); I[0] = i
        for n in range(1, self.nx):
            if n <= self.start:
                for _ in range(2):                            # two implicit half steps
                    y, i = self._solve(self.M @ y, V, y, i)
            else:
                rhs = self.Rm @ y - self.c * self.cw * i * self.e0
                y, i = self._solve(rhs, V, y, i)
            Y[:, n] = y; I[n] = i
        self.y, self.i = Y, I
        return Y, I

    def mean(self, Y=None):
        """The mixing-cup (flow-weighted) mean of every column."""
        return self.wmean @ (self.y if Y is None else Y)

    def current_A(self, I=None):
        """The cell current [A]: the current density integrated over the 100 cm^2 (trapezoid along X)."""
        return float(np.trapezoid(self.i if I is None else I, self.X) * pb.A_CELL)

    def balance(self):
        """The balance of the mean at the outlet: the mean's fall from the field,
        against K_I times the integral of the current (trapezoid along X)."""
        return float(pb.Y_IN - self.mean()[-1]), float(K_I * np.trapezoid(self.i, self.X))


def faraday(mean_out):
    """The cell current [A] from the steam converted, by Faraday's law: the
    mean's fall over the channel over K_I, times the 100 cm^2."""
    return float((pb.Y_IN - mean_out) / K_I * pb.A_CELL)


def summary(m, V):
    """The scored quantities of one march: the cell current [A], the outlet
    mean, and the steam at the reaction site along X."""
    mean = m.mean()
    return dict(current=faraday(mean[-1]), mean_out=float(mean[-1]), y_site=site(m.y[0], m.i, m.s), mean=mean,
                y_wall=m.y[0].copy(), i=m.i.copy(), X=m.X)


# ---------------------------------------------- the one-dimensional model (S -> 0)
def one_d(V, nx=4001, gas_loss=False):
    """Ex10.2's one-dimensional steam balance and charge transfer at 800 degC,
    without axial diffusion: dy/dX = -K_I i(y), y(0) = 0.90, by RK4 on nx nodes.
    ``gas_loss=False`` is the limit of infinitely fast transport (S = 0): the
    Nernst potential at the channel's steam and no gas-transport loss.
    ``gas_loss=True`` is Ex10.2's own charge transfer with its estimated
    limiting current (pb.current). Returns X, y, i."""
    if gas_loss:
        f = lambda y: -K_I * float(pb.current(V, y, T_CELL))
    else:
        f = lambda y: -K_I * float(current_at(V, y, 0.0)[0])
    h = 1.0 / (nx - 1); y = pb.Y_IN; ys = [y]
    for _ in range(nx - 1):
        k1 = f(y); k2 = f(y + h * k1 / 2); k3 = f(y + h * k2 / 2); k4 = f(y + h * k3)
        y += h * (k1 + 2 * k2 + 2 * k3 + k4) / 6; ys.append(y)
    ys = np.array(ys)
    i = pb.current(V, ys, T_CELL) if gas_loss else current_at(V, ys, 0.0)[0]
    return np.linspace(0, 1, nx), ys, i


def plug_flow(V, nx=2001):
    """The hand calculation: the gas mixed across the channel (plug flow), the
    electrode resolved: dybar/dX = -K_I i, with i the root of the charge
    transfer at the reaction site ybar - K_E i. By RK4 on nx nodes. Returns X,
    the mean, the current density and the reaction-site steam."""
    f = lambda y: -K_I * float(current_at(V, y, 1.0)[0])
    h = 1.0 / (nx - 1); y = pb.Y_IN; ys = [y]
    for _ in range(nx - 1):
        k1 = f(y); k2 = f(y + h * k1 / 2); k3 = f(y + h * k2 / 2); k4 = f(y + h * k3)
        y += h * (k1 + 2 * k2 + 2 * k3 + k4) / 6; ys.append(y)
    ys = np.array(ys); i = current_at(V, ys, 1.0)[0]
    return np.linspace(0, 1, nx), ys, i, ys - K_E * i


def check_zero_strength(V=1.29, strengths=(1e-1, 1e-2, 1e-3), grid=(801, 41), verbose=True):
    """The march with both transport resistances times S, against the
    one-dimensional model of infinitely fast transport: the difference in the
    cell current [A] and in the outlet mean must fall in proportion to S."""
    X1, y1, i1 = one_d(V)
    I1 = faraday(y1[-1])
    rows = []
    for s in strengths:
        m = March(*grid, s=s); m.run(V)
        I = faraday(m.mean()[-1])
        dI, dm = abs(I - I1), abs(m.mean()[-1] - y1[-1])
        rows.append((s, dI, dm))
        if verbose:
            print(f"   S = {s:.0e}: cell current {I:.4f} A against the 1-D model's {I1:.4f} A "
                  f"(off by {dI:.2e} A), outlet mean off by {dm:.2e}")
    return rows


# ------------------------------------------------------- the reference model
REF_GRID = (21, 5)      #: nx, nz: the engineering grid, the coarsest that meets the agreed accuracy over the range


def reference(V, grid=None):
    """The reference model at one cell voltage: the march on the engineering
    grid. Returns its summary and the wall-clock time [s]."""
    t0 = time.perf_counter()
    m = March(*(grid or REF_GRID))
    m.run(V)
    out = summary(m, V)
    out["seconds"] = time.perf_counter() - t0
    return out


# -------------------------------------------------------- the ground truth
TRUTH_GRIDS = ((101, 11), (201, 21), (401, 41), (801, 81), (1601, 161))
TRUTH_GRID = TRUTH_GRIDS[-1]    #: the grid the truth file is written on


def convergence_study(V=1.33, grids=TRUTH_GRIDS, verbose=True):
    """The cell current, the outlet mean and the reaction-site steam at the
    outlet on grids each twice as fine in X and Z; Crank–Nicolson and central
    differences are second order, so the change between grids should fall by
    about four each time. Returns the rows and the Richardson extrapolations."""
    rows = []
    for g in grids:
        t0 = time.perf_counter()
        m = March(*g); m.run(V)
        sec = time.perf_counter() - t0
        out = summary(m, V)
        a, b = m.balance()
        rows.append(dict(grid=g, current=out["current"], mean_out=out["mean_out"], site_out=float(out["y_site"][-1]),
                         balance=a - b, seconds=sec))
        if verbose:
            print(f"   grid {g[0]:5d} x {g[1]:4d}: cell current {out['current']:.6f} A, outlet mean {out['mean_out']:.7f}, "
                  f"reaction site at the outlet {out['y_site'][-1]:.7f}, balance off by {abs(a - b):.1e}   ({sec:.2f} s)")
    ext = {k: rows[-1][k] + (rows[-1][k] - rows[-2][k]) / 3 for k in ("current", "mean_out", "site_out")}
    if verbose:
        for k in ("current", "mean_out", "site_out"):
            d = np.abs(np.diff([r[k] for r in rows]))
            print(f"   change of {k} between grids: " + " ".join(f"{v:.1e}" for v in d)
                  + "; observed order " + " ".join(f"{np.log2(d[j] / d[j + 1]):.2f}" for j in range(len(d) - 1)))
        print(f"   Richardson: cell current {ext['current']:.6f} A, outlet mean {ext['mean_out']:.7f}, "
              f"reaction site at the outlet {ext['site_out']:.7f}")
    return rows, ext


def ground_truth(V, grid=TRUTH_GRID):
    """The fields at one cell voltage on the truth grid."""
    m = March(*grid); m.run(V)
    out = summary(m, V)
    out.update(Z=m.Z, y=m.y)
    return out


# ------------------------------------------------------- the training points
def sample_channel(n, seed=88):
    """n points (X, Z, V) over the channel and the range of voltages, a Latin
    hypercube; a fifth of them in the first 5 % of the length, where the
    profile across the channel develops from the flat inlet."""
    rng = np.random.default_rng(seed)
    u = (rng.permuted(np.tile(np.arange(n), (3, 1)), axis=1).T + rng.random((n, 3))) / n
    u[: n // 5, 0] *= 0.05
    u[:, 2] = V_LOW + (V_HIGH - V_LOW) * u[:, 2]
    return u


def sample_faces(n, seed=88):
    """Points on the three faces that carry a condition, at random voltages:
    (inlet, electrode, interconnect), each (n, 3) as (X, Z, V)."""
    rng = np.random.default_rng(seed + 1)
    v = lambda m: V_LOW + (V_HIGH - V_LOW) * rng.random(m)
    inlet = np.c_[np.zeros(n), rng.random(n), v(n)]
    x = rng.random(n); x[: n // 5] *= 0.05
    electrode = np.c_[x, np.zeros(n), v(n)]
    x2 = rng.random(n); x2[: n // 5] *= 0.05
    interconnect = np.c_[x2, np.ones(n), v(n)]
    return inlet, electrode, interconnect


# ----------------------------------------------------------- the truth file
TRUTH_FILE = "MP10.2A_truth.npz"
V_SWEEP = np.round(np.linspace(1.15, 1.35, 21), 3)   #: V, the sweep of the cell's polarisation curve
V_OUTSIDE = (1.12, 1.38)                             #: V, outside the trained range


def write_truth(path=TRUTH_FILE, verbose=True):
    """The ground truth on the 1601 x 161 grid: at the three test voltages the
    field restricted to 401 x 41 nodes and, on all 1601 nodes along X, the
    steam at the electrode's surface and at the reaction site, the current
    density and the mixing-cup mean; the cell current and the outlet mean at
    21 voltages and two outside the range; the convergence table at 1.33 V;
    the zero-strength check; Ex10.2's 1-D model and the plug-flow hand
    calculation for comparison; every parameter with its unit."""
    t0 = time.perf_counter()
    rows, ext = convergence_study(1.33, verbose=verbose)
    out = dict(V_test=np.array(V_TEST), grid=np.array(TRUTH_GRID, dtype=float))
    Xs, Zs = np.linspace(0, 1, 401), np.linspace(0, 1, 41)
    for V in V_TEST:
        g = ground_truth(V)
        k = f"{V:.2f}"
        ix = np.rint(Xs * (len(g["X"]) - 1)).astype(int); iz = np.rint(Zs * (len(g["Z"]) - 1)).astype(int)
        out[f"field_{k}"] = g["y"][np.ix_(iz, ix)]
        for q in ("y_wall", "y_site", "i", "mean"):
            out[f"{q}_{k}"] = g[q]
        out[f"current_{k}"] = np.array(g["current"]); out[f"mean_out_{k}"] = np.array(g["mean_out"])
        out["X_line"] = g["X"]
    out["X"], out["Z"] = Xs, Zs
    sweep = [ground_truth(V) for V in V_SWEEP]
    out["V_sweep"] = V_SWEEP
    out["current_sweep"] = np.array([g["current"] for g in sweep]); out["mean_out_sweep"] = np.array([g["mean_out"] for g in sweep])
    out["V_outside"] = np.array(V_OUTSIDE)
    for V in V_OUTSIDE:
        g = ground_truth(V)
        for q in ("y_site", "i"):
            out[f"{q}_{V:.2f}"] = g[q]
        out[f"current_{V:.2f}"] = np.array(g["current"]); out[f"mean_out_{V:.2f}"] = np.array(g["mean_out"])
    one, plug = [], []
    for V, g in zip(V_SWEEP, sweep):
        X1, y1, i1 = one_d(V, gas_loss=True)
        one.append(faraday(y1[-1]))
        Xp, yp, ip, sp_ = plug_flow(V)
        k = g["X"] >= X_SCORE
        plug.append([faraday(yp[-1]) / g["current"] - 1, yp[-1] - g["mean_out"],
                     np.abs(np.interp(g["X"], Xp, sp_) - g["y_site"])[k].max()])
    out["current_sweep_ex102_1d"] = np.array(one)
    out["plug_errors_sweep"] = np.array(plug)
    out["conv_grids"] = np.array([r["grid"] for r in rows], dtype=float)
    for q in ("current", "mean_out", "site_out", "balance"):
        out[f"conv_{q}"] = np.array([r[q] for r in rows])
    for q in ("current", "mean_out", "site_out"):
        out[f"conv_{q}_extrapolated"] = np.array(ext[q])
    z = check_zero_strength(verbose=verbose)
    out["zero_strength"] = np.array(z)
    out["params"] = np.array([f"{k} = {v}" for k, v in dict(
        cell="Ex10.2's cell, 100 x 100 mm, isothermal at 800 degC", channel="100 mm long, 1 mm high, 1 m/s mean, Poiseuille",
        feed="0.90 steam fraction", D=f"{pb.D_GAS} m2/s, estimate (Ex10.2)", electrode=f"{DELTA*1e3} mm thick, D_eff {D_EFF} m2/s, estimate",
        kinetics=f"i0 {pb.I_0} A/cm2, ASR {ASR_C:.3f} ohm cm2 at 800 degC, estimates (Ex10.2)",
        alpha=f"{ALPHA:.1f}", beta=f"{BETA:.4e} per A/cm2", K_E=f"{K_E:.4f} per A/cm2", K_I=f"{K_I:.4f} per A/cm2",
        V="1.15-1.35 V, test 1.18, 1.27, 1.33 V").items()])
    out["README"] = np.array(
        "MP10.2A ground truth (Crank-Nicolson march, Rannacher start, 1601 x 161 nodes). V_test [V]; field_<V>: steam "
        "fraction [-] on (Z, X), 41 x 401 nodes (X = x/L, Z = z/h, 0 at the electrode); on X_line (1601 nodes): y_wall_<V> "
        "steam at the electrode's surface, y_site_<V> at the reaction site, i_<V> current density [A/cm2], mean_<V> the "
        "mixing-cup mean; current_<V> cell current [A] (Faraday, from the steam converted); mean_out_<V> outlet mean; "
        "V_sweep, current_sweep, mean_out_sweep: 21 voltages; V_outside: 1.12 and 1.38 V; current_sweep_ex102_1d: Ex10.2's "
        "1-D model (isothermal, no axial diffusion, its gas-transport loss) [A]; plug_errors_sweep: the plug-flow hand "
        "calculation's errors (relative current, outlet mean, reaction site beyond 5 mm); conv_*: convergence at 1.33 V on "
        "conv_grids (nx, nz), conv_balance the trapezoid current against Faraday's in the steam fraction; zero_strength: "
        "rows (S, cell current difference [A], outlet mean difference) against the 1-D model of infinitely fast transport; "
        "params: the data.")
    np.savez_compressed(path, **out)
    print("written", path, f"({time.perf_counter() - t0:.0f} s)")


if __name__ == "__main__":
    import sys
    print("zero-strength check")
    check_zero_strength()
    print("convergence at 1.33 V")
    convergence_study()
    if "--save" in sys.argv:
        write_truth()
