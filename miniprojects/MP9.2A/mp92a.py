r"""MP9.2A — The SOEC steam channel, with the current following the steam at
the electrode. The physics, the reference model and the ground truth.

*Deep Learning for Engineering* — MSc, Aalborg University.
Remus Teodorescu (ret@et.aau.dk), with support from Research Assistant
Noman Khan (nomank@energy.aau.dk).

This module extends Ex09.2's ``channel.py`` (course policy C10 rule 1: one set
of physics and constants for the exercise and its mini project) and is used by
the notebook of MP9.2A (course policy C14).

## What changes against Ex09.2, and why

Ex09.2 runs the cell as a fuel cell with a **given, uniform** current density,
0.5 A/cm², so the wall flux N = i/2F is a constant and the field has an exact
solution. Here the same channel runs as an **electrolyser** (SOEC): 90 % steam
is fed, steam is taken at the electrode and hydrogen returned, one for one.
And the current is no longer given: an electrolyser is run at a fixed cell
voltage, and where the steam at the electrode is depleted the concentration
overpotential rises and the local current falls. The reduced model used here
(ASSUMED, a saturating Langmuir-type form; no kinetic constant is taken from a
measurement) is

    N(y_w) = N0 [ (1 - B) + B r(y_w) / r(Y_IN) ],     r(y) = y / (y + K_HALF)

with y_w the steam fraction at the electrode, N0 = i0 / 2F the flux at the
inlet's composition (i0 = 0.5 A/cm², Ex09.2's value), K_HALF = 0.2 the steam
fraction at which r is half its maximum, and B the strength of the change. At
B = 0 the current is uniform, the problem is Ex09.2's with the sign of the flux
reversed, and the march must reproduce Ex09.2's exact solution with that sign
(:func:`check_zero_strength`). At B = 1 the current falls along the channel
with the steam: the wall condition depends on the unknown, the problem is
nonlinear, and **no exact solution exists**. The form also keeps the steam
fraction positive: as y_w goes to zero, so does the current.

## The problem, scaled as in Ex09.2

    6 Z (1 - Z) G s θ_X = θ_ZZ                   in the channel, 0 < X < 1, 0 < Z < 1
    θ = 0                                        at X = 0 (the inlet: 90 % steam)
    θ_Z = + G g(θ)                               at Z = 0 (the electrode: steam taken)
    θ_Z = 0                                      at Z = 1 (the interconnect)

with θ = (y - Y_IN)/Y, Y = N0 L/(u_ref h c) = 0.228 as in Ex09.2, s = u/u_ref,
and g = N/N0 = (1 - B) + B r(Y_IN + Y θ)/r(Y_IN). Integrated over the height
with the two wall conditions the equation gives the **balance of the mean**,

    s dθ̄/dX = - g(θ(X, 0))                      θ̄: the flow-weighted (mixing-cup) mean

the steam taken at the wall leaves the flow; :func:`March` checks it.

## Two solvers

* the **reference model** (:func:`reference`): Crank-Nicolson along the
  channel with Rannacher's start (two fully implicit columns first, as Ex09.2
  found), central differences across, the nonlinear wall by Newton's method on
  the electrode node, on an engineering grid; for the design speed, the secant
  method on the speed with that march. It is what the network is timed against.
* the **ground truth** (:func:`ground_truth`): the same scheme on grids each
  twice as fine, converged and extrapolated (:func:`convergence_study`).
"""

from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import channel as ch

# ------------------------------------------------------------------ the data
Y_IN = 0.90             #: steam fraction fed: 90 % steam, 10 % hydrogen (a typical SOEC feed)   [typical value]
K_HALF = 0.20           #: the steam fraction at which r(y) = y/(y + K) is half its maximum   [ASSUMED]
B_COUPLE = 1.0          #: strength of the change: 0 is Ex09.2's uniform current, reversed
N0 = ch.N_WALL          #: mol/(m² s), steam taken at the inlet's composition, i0/2F with i0 = 0.5 A/cm²
Y = ch.Y_SCALE          #: the mean's fall over the channel at 1 m/s and uniform current, 0.228
G = ch.G                #: Pe across the channel times the aspect ratio, 0.0125
S_MIN, S_MAX = 0.2, 1.2 #: the range of feed speeds the network is trained over, m/s (s = u / 1 m/s)
S_TEST = (0.25, 0.55, 0.95)   #: m/s, the speeds the answers are scored at (not trained on)
X_SCORE = 0.05          #: the electrode fraction is scored beyond the first 5 mm, as in Ex09.2
CONVERSION = 0.70       #: the design asks for 70 % of the steam fed to be converted
TARGET_Y = 5e-4         #: the agreed accuracy in the steam fraction at the electrode: 0.05 percentage point
MU_FEED = 4.0e-5        #: Pa s, the viscosity of 90 % steam at 800 degC   [estimate]
RHO_FEED = ch.C_TOT * (Y_IN * ch.M_H2O + (1 - Y_IN) * ch.M_H2)   #: kg/m³ of the feed, 0.19


def reynolds(s):
    """The Reynolds number on the channel height at the feed speed s (m/s), with the feed's density."""
    return RHO_FEED * s * ch.U_REF * ch.H_CHANNEL / MU_FEED


def pressure_drop(s):
    """Poiseuille's pressure drop over the channel, Pa, with the feed's viscosity."""
    return 12 * MU_FEED * s * ch.U_REF * ch.L_CHANNEL / ch.H_CHANNEL ** 2


def g_flux(theta_w, b=B_COUPLE):
    """g = N/N0 at the electrode, as a function of the scaled steam there."""
    y = np.maximum(Y_IN + Y * np.asarray(theta_w, dtype=float), 0.0)
    r = y / (y + K_HALF)
    return (1.0 - b) + b * r / (Y_IN / (Y_IN + K_HALF))


def dg_flux(theta_w, b=B_COUPLE):
    """dg/dθ at the electrode (for Newton's method)."""
    y = np.maximum(Y_IN + Y * np.asarray(theta_w, dtype=float), 0.0)
    return b * Y * K_HALF / (y + K_HALF) ** 2 / (Y_IN / (Y_IN + K_HALF))


def steam(theta):
    """Scaled field back to the steam fraction."""
    return Y_IN + Y * np.asarray(theta)


def current_density(theta_w, b=B_COUPLE):
    """The local current density, A/cm², at the electrode's scaled steam."""
    return ch.I_CELL / 1e4 * g_flux(theta_w, b)


def plug_flow(X, s, b=B_COUPLE, lib=np, iters=6):
    """The hand calculation of the mean: the steam fraction of a gas mixed
    across the channel (plug flow), with the current at that fraction,

        s dθ̄/dX = -g(θ̄),   θ̄ = 0 at X = 0.

    Separable: with y = Y_IN + Y θ̄, r_in = r(Y_IN) and a = b / r_in,
    (1 - b) and b parts give  dX = -(s / Y) dy / ((1 - b) + a y / (y + K)).
    At b = 1 this integrates to  (Y_IN - y) + K ln(Y_IN / y) = Y X / (s a), a Lambert W;
    at b = 0 to  θ̄ = -X / s, Ex09.2's mean reversed. In general it is solved
    for u = ln y by Newton's method, which converges from u = ln Y_IN
    downwards for any b in [0, 1]; six steps reach rounding over the whole
    range of speeds (four leave 3e-06 in the steam fraction). ``lib`` is numpy or torch (every step is
    differentiable, so a network can build it in and autograd differentiate it).
    Returns θ̄(X, s)."""
    if b == 0.0:
        return -X / s                                        # Ex09.2's mean, reversed (it may go below zero: the uniform current does not stop)
    r_in = Y_IN / (Y_IN + K_HALF)
    a = b / r_in
    c = 1.0 - b
    # F(y) = ∫_y^Y_IN dy' / (c + a y'/(y' + K)) = Y X / s; written out for c > 0 and c = 0 together
    def F(y):
        if c == 0.0:
            return ((Y_IN - y) + K_HALF * (np.log(Y_IN) - lib.log(y))) / a
        # 1/(c + a y/(y+K)) = (y + K) / ((c + a) y + c K)
        q = c + a
        return (Y_IN - y) / q + (K_HALF - c * K_HALF / q) / q * (np.log(q * Y_IN + c * K_HALF) - lib.log(q * y + c * K_HALF))
    def dF(y):
        return -(y + K_HALF) / ((c + a) * y + c * K_HALF)
    rhs = Y * X / s
    u = lib.log(Y_IN + 0.0 * rhs)
    for _ in range(iters):
        y = lib.exp(u)
        u = u - (F(y) - rhs) / (dF(y) * y)                   # Newton on u = ln y: F decreasing and convex in u
    return (lib.exp(u) - Y_IN) / Y


def exact_zero_strength(X, Z, s):
    """Ex09.2's exact solution with the flux reversed: the field at B = 0
    beyond the entrance, θ = -(X/s + G φ(Z))."""
    return -ch.exact(X, Z, s)


# ---------------------------------------------------------- the march
class March:
    """Crank-Nicolson along the channel on an nx by nz grid, the electrode's
    nonlinear flux by Newton's method on its node. Built once per grid; run
    for any speed. Only the electrode row is nonlinear, so each Newton step is
    one solve with the factorised linear matrix and a rank-one correction
    (Sherman-Morrison): no refactorisation."""

    def __init__(self, nx, nz, b=B_COUPLE, start=2):
        self.nx, self.nz, self.b, self.start = nx, nz, b, start
        self.X, self.Z = np.linspace(0, 1, nx), np.linspace(0, 1, nz)
        self.hx, self.hz = 1.0 / (nx - 1), 1.0 / (nz - 1)
        A = sp.diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(nz, nz)).tolil()
        A[0, 1] = 2.0; A[-1, -2] = 2.0                        # both walls by a ghost node
        self.A = (A.tocsc() / self.hz ** 2)
        self.e0 = np.zeros(nz); self.e0[0] = 1.0
        self.cw = 2.0 * G / self.hz                           # the electrode's flux through its ghost: -cw g(θ0) on row 0
        # the trapezoid weights of the flow-weighted mean, normalised (Ex09.2's mixing_cup)
        w = ch.poiseuille(self.Z) * self.hz; w[[0, -1]] *= 0.5
        self.flow = w.sum()                                   # the trapezoid integral of 6 Z (1 - Z): 1 - hz²
        self.wmean = w / w.sum()
        self._s = None

    def _factor(self, s):
        if self._s == s:
            return
        # the velocity in front of θ_X, zero at the walls, normalised so that its trapezoid integral over the
        # height is exactly 1: the grid then carries exactly the flow fed, and the mean's balance holds on any
        # grid (without it, 11 nodes carry 1 % less flow and the steam is used up 1 % too fast)
        M = sp.diags(ch.poiseuille(self.Z) / self.flow * G * s).tocsc()
        I = lambda c: spla.splu((M - c * self.A).tocsc())
        self.cn, self.be = I(0.5 * self.hx), I(0.5 * self.hx)  # CN and the implicit half step share the matrix M - (hx/2) A
        self.R = (M + 0.5 * self.hx * self.A).tocsc()
        self.M = M
        self.w0 = self.cn.solve(self.e0)                       # for the rank-one Newton correction
        self._s = s

    def _solve(self, lu, w0, rhs, c, guess, iters=3):
        """Solve (M - c A) θ + c cw g(θ0) e0 = rhs by Newton on θ0."""
        th = guess
        base = lu.solve(rhs)
        for _ in range(iters):
            # linearise c cw g(θ0) about the current θ0: g + g' (θ0 - θ0*)
            g0, dg0 = g_flux(th[0], self.b), dg_flux(th[0], self.b)
            d = c * self.cw * dg0
            r_shift = c * self.cw * (g0 - dg0 * th[0])
            x = base - r_shift * w0                             # (M - cA)^-1 (rhs - r_shift e0)
            th = x - d * x[0] / (1.0 + d * w0[0]) * w0         # Sherman-Morrison for + d e0 e0^T
        return th

    def run(self, s, iters=3):
        """March at the feed speed s (m/s). Returns θ (nz, nx)."""
        self._factor(s)
        nx, hx = self.nx, self.hx
        th = np.zeros((self.nz, nx))
        for n in range(1, nx):
            old = th[:, n - 1]
            if n <= self.start:
                y = old
                for _ in range(2):                              # two implicit half steps: the wall condition holds on this column
                    y = self._solve(self.be, self.w0, self.M @ y, 0.5 * hx, y, iters)
                th[:, n] = y
            else:
                rhs = self.R @ old - 0.5 * hx * self.cw * g_flux(old[0], self.b) * self.e0
                th[:, n] = self._solve(self.cn, self.w0, rhs, 0.5 * hx, old, iters)
        self.theta = th
        return th

    def mean(self, th=None):
        """The flow-weighted mean θ̄ of every column."""
        th = self.theta if th is None else th
        return self.wmean @ th

    def balance(self, s, th=None):
        """The balance of the mean at the outlet: θ̄(1) from the field against
        -(1/s) ∫ g(θ_wall) dX from the wall flux (trapezoid along X)."""
        th = self.theta if th is None else th
        gw = g_flux(th[0], self.b)
        flux = -np.trapezoid(gw, self.X) / s
        return float(self.mean(th)[-1]), float(flux)


def conversion_from(m, s, th):
    """The share of the fed steam converted over the channel, from the wall
    flux (the more exact of the two balance sides)."""
    _, mean_out = m.balance(s, th)
    return -Y * mean_out / Y_IN


def electrode(th):
    """The steam fraction along the electrode."""
    return steam(th[0])


# ------------------------------------------------------- the reference model
REF_GRID = (21, 3)      #: nx, nz: the engineering grid, the coarsest that meets 0.1 percentage point at every speed of the range


def reference(s, grid=None):
    """The reference model at one feed speed: the march on the engineering
    grid. Returns X, the electrode's steam fraction, the conversion and the
    wall-clock time (s)."""
    m = March(*(grid or REF_GRID))
    t0 = time.perf_counter()
    th = m.run(s)
    conv = conversion_from(m, s, th)
    sec = time.perf_counter() - t0
    return dict(X=m.X, y_wall=electrode(th), conversion=conv, seconds=sec, theta=th, Z=m.Z)


def speed_for_conversion(target=CONVERSION, grid=None, s0=None, tol=1e-6, verbose=False):
    """The feed speed (m/s) at which the march converts ``target`` of the
    steam: the secant method on the speed, one march per step, started from
    the uniform-current hand calculation. Returns the speed, the marches it
    took and the wall-clock time (s)."""
    m = March(*(grid or REF_GRID))
    t0 = time.perf_counter()
    f = lambda s: conversion_from(m, s, m.run(s)) - target
    a = s0 or Y / (target * Y_IN)                             # the hand calculation with the inlet's current
    b = 0.9 * a
    fa, fb = f(a), f(b); n = 2
    for _ in range(30):
        c = b - fb * (b - a) / (fb - fa)
        a, fa = b, fb
        b, fb = c, f(c); n += 1
        if verbose:
            print(f"   s = {b:.6f} m/s, conversion off by {fb:.2e}")
        if abs(b - a) < tol * b:
            break
    return b, n, time.perf_counter() - t0


# -------------------------------------------------------- the ground truth
TRUTH_GRIDS = ((201, 11), (401, 21), (801, 41), (1601, 81), (3201, 161))
TRUTH_GRID = TRUTH_GRIDS[-2]    #: the grid the truth file is written on


def on_grid(th, X, Z, Xc, Zc):
    """Restrict a field on (X, Z) to a coarser grid (Xc, Zc) that shares its nodes."""
    ix = np.rint(Xc * (len(X) - 1)).astype(int); iz = np.rint(Zc * (len(Z) - 1)).astype(int)
    return th[np.ix_(iz, ix)]


def convergence_study(s=0.55, grids=TRUTH_GRIDS, verbose=True):
    """The electrode's steam fraction and the conversion on grids each twice as
    fine in X and Z. Crank-Nicolson and central differences are second order,
    so the change between grids should fall by about four each time; the last
    two are extrapolated by Richardson. Returns the rows and the extrapolated
    electrode fraction on the coarsest grid's X and the conversion."""
    rows = []
    for g in grids:
        m = March(*g)
        t0 = time.perf_counter()
        th = m.run(s)
        sec = time.perf_counter() - t0
        c = conversion_from(m, s, th)
        mean_field, mean_flux = m.balance(s, th)
        Xc = np.linspace(0, 1, grids[0][0])
        yw = steam(on_grid(th, m.X, m.Z, Xc, np.array([0.0]))[0])
        rows.append(dict(grid=g, y_wall=yw, conversion=c, balance=mean_field - mean_flux, seconds=sec))
        if verbose:
            print(f"   grid {g[0]:5d} x {g[1]:4d}: steam at the electrode at the outlet {100 * yw[-1]:.5f} %, "
                  f"conversion {100 * c:.5f} %, balance off by {abs(mean_field - mean_flux):.1e}   ({sec:.2f} s)")
    yw_e = rows[-1]["y_wall"] + (rows[-1]["y_wall"] - rows[-2]["y_wall"]) / 3
    c_e = rows[-1]["conversion"] + (rows[-1]["conversion"] - rows[-2]["conversion"]) / 3
    if verbose:
        Xc = np.linspace(0, 1, grids[0][0]); k = Xc >= X_SCORE
        for i in range(1, len(rows)):
            d = np.abs(rows[i]["y_wall"] - rows[i - 1]["y_wall"])[k].max()
            print(f"   change of the electrode fraction beyond 5 mm, grid {i} to {i + 1}: {d:.2e}")
        print(f"   extrapolated: steam at the electrode at the outlet {100 * yw_e[-1]:.5f} %, conversion {100 * c_e:.5f} %")
    return rows, yw_e, c_e


def ground_truth(s, grid=TRUTH_GRID):
    """The field at a speed on the truth grid."""
    m = March(*grid)
    th = m.run(s)
    return dict(X=m.X, Z=m.Z, theta=th, y_wall=electrode(th), conversion=conversion_from(m, s, th))


def check_zero_strength(grid=(1601, 81), speeds=(0.35, 0.55, 0.95), verbose=True):
    """At B = 0 the current is uniform and the problem is Ex09.2's with the
    flux reversed: the march must reproduce -ch.exact beyond the entrance."""
    worst = 0.0
    for s in speeds:
        m = March(*grid, b=0.0)
        th = m.run(s)
        XX, ZZ = np.meshgrid(m.X, m.Z)
        e = Y * np.abs(th - exact_zero_strength(XX, ZZ, s))[XX >= X_SCORE].max()
        worst = max(worst, e)
        if verbose:
            print(f"   B = 0, u = {s:.2f} m/s: largest difference from Ex09.2's exact solution, reversed: {e:.1e} in the steam fraction")
    return worst


# ------------------------------------------------------- the training points
def sample_channel(n, seed=88):
    """n points (X, Z, s) over the channel and the range of speeds, Latin
    hypercube; a fifth of them in the first 5 % of the length, where the
    profile develops (Ex09.2's sampler on this range of speeds)."""
    rng = np.random.default_rng(seed)
    u = (rng.permuted(np.tile(np.arange(n), (3, 1)), axis=1).T + rng.random((n, 3))) / n
    u[: n // 5, 0] *= 0.05
    u[:, 2] = S_MIN + (S_MAX - S_MIN) * u[:, 2]
    return u


def sample_faces(n, seed=88):
    """Points on the three faces that carry a condition, at random speeds:
    (inlet, electrode, interconnect), each (n, 3) as (X, Z, s)."""
    rng = np.random.default_rng(seed + 1)
    s = lambda m: S_MIN + (S_MAX - S_MIN) * rng.random(m)
    inlet = np.c_[np.zeros(n), rng.random(n), s(n)]
    x = rng.random(n); x[: n // 5] *= 0.05
    electrode_ = np.c_[x, np.zeros(n), s(n)]
    x2 = rng.random(n); x2[: n // 5] *= 0.05
    interconnect = np.c_[x2, np.ones(n), s(n)]
    return inlet, electrode_, interconnect


# ----------------------------------------------------------- the truth file
TRUTH_FILE = "MP9.2A_truth.npz"


def write_truth(path=TRUTH_FILE, grid=TRUTH_GRID):
    """The ground truth at the three test speeds and at the design speed: the
    field on the 1601 x 81 grid restricted to 401 x 41 nodes, the electrode's
    steam fraction on all 1601 nodes, the conversion, with the convergence
    table, the zero-strength check and every parameter with its unit."""
    rows, yw_e, c_e = convergence_study(0.55, verbose=True)
    s_star, n_march, _ = speed_for_conversion(grid=grid)
    out = dict(speeds=np.array(S_TEST), grid=np.array(grid, dtype=float), speed_70=np.array(s_star))
    Xs, Zs = np.linspace(0, 1, 401), np.linspace(0, 1, 41)
    for s in S_TEST + (s_star,):
        gt = ground_truth(s, grid)
        key = "design" if s == s_star else f"{s:.2f}"
        out[f"field_{key}"] = steam(on_grid(gt["theta"], gt["X"], gt["Z"], Xs, Zs))
        out[f"y_wall_{key}"] = gt["y_wall"]
        out[f"conversion_{key}"] = np.array(gt["conversion"])
        out["X_wall"] = gt["X"]
    out["X"], out["Z"] = Xs, Zs
    out["conv_grids"] = np.array([r["grid"] for r in rows], dtype=float)
    out["conv_y_wall_outlet"] = np.array([r["y_wall"][-1] for r in rows])
    out["conv_conversion"] = np.array([r["conversion"] for r in rows])
    out["conv_balance"] = np.array([r["balance"] for r in rows])
    out["conv_y_wall_outlet_extrapolated"] = np.array(yw_e[-1]); out["conv_conversion_extrapolated"] = np.array(c_e)
    out["zero_strength_error"] = np.array(check_zero_strength(verbose=False))
    out["params"] = np.array([f"{k} = {v}" for k, v in dict(
        Y_IN="0.90 steam fraction fed, typical", K_HALF="0.20, ASSUMED", B_COUPLE=B_COUPLE,
        I0="0.5 A/cm2 at the inlet's composition, typical", N0=f"{N0:.4e} mol/(m2 s)", D=f"{ch.D_GAS} m2/s, estimate",
        L="0.10 m", H="1.0e-3 m", T="800 degC", P="1 atm", C=f"{ch.C_TOT:.3f} mol/m3", S_RANGE="0.2-1.2 m/s").items()])
    out["README"] = np.array(
        "MP9.2A ground truth. speeds: the three test feed speeds (m/s); field_<s>: steam fraction (nz=41, nx=401) on Z, X "
        "(scaled height and length); y_wall_<s>: steam fraction at the electrode on X_wall (1601 nodes); conversion_<s>: "
        "share of the fed steam converted; *_design: at speed_70, the feed speed (m/s) converting 70 %; conv_*: the "
        "convergence study at 0.55 m/s, grids (nx, nz); zero_strength_error: the march at B = 0 against Ex09.2's exact "
        "solution reversed, in the steam fraction; params: the data.")
    np.savez_compressed(path, **out)
    print("written", path, f"; speed for 70 % conversion {s_star:.5f} m/s ({n_march} marches)")


if __name__ == "__main__":
    import sys
    print("zero-strength check")
    check_zero_strength()
    print("convergence at 0.55 m/s")
    convergence_study()
    if "--save" in sys.argv:
        write_truth()
