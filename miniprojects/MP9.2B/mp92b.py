r"""MP9.2B — The air channel of the SOFC, with the current following the oxygen
at the electrode. The physics, the reference model and the ground truth.

*Deep Learning for Engineering* — MSc, Aalborg University.
Remus Teodorescu (ret@et.aau.dk), with support from Research Assistant
Noman Khan (nomank@energy.aau.dk).

This module extends Ex09.2's ``channel.py`` (course policy C10 rule 1: one set
of physics and constants for the exercise and its mini project) and is used by
the notebook of MP9.2B (course policy C14).

## What changes against Ex09.2, and why

Ex09.2 is the fuel channel: hydrogen taken, steam returned, one for one, at a
given uniform current; the molar flow is constant and the field has an exact
solution. The **air channel** on the other side of the same cell differs in
three ways:

1. **Oxygen is taken and nothing is returned.** The molar flow falls along the
   channel (the nitrogen is conserved, the oxygen leaves through the wall), so
   the mean velocity falls, the gas has a velocity towards the electrode (the
   Stefan velocity, v = -N/c at the wall), and the diffusive flux at the wall
   carries only the share 1 - y_w of the oxygen flux: D c y_z = N (1 - y_w).
   (J. Khazaei's correction of 7 October: the outlet fraction of the mean is
   (y_in - Y)/(1 - Y), not y_in - Y.)
2. **Air, not hydrogen:** 21 % oxygen, D = 1.9e-04 m²/s for oxygen in
   nitrogen at 800 degC (an estimate), four electrons per oxygen molecule.
3. **The current follows the oxygen at the electrode** (ASSUMED reduced
   model, a saturating Langmuir-type form; no kinetic constant is taken from a
   measurement):

       N(y_w) = N0 [ (1 - B) + B r(y_w) / r(Y_IN) ],     r(y) = y / (y + K_O2)

   with N0 = i0 / 4F, i0 = 0.5 A/cm² (Ex09.2's value) and K_O2 = 0.05.

Two strengths switch the changes off: ``B`` (the coupling) and ``S`` (the
one-sided exchange: the falling flow, the Stefan velocity and the 1 - y_w
factor). At B = 0 and S = 0 the problem is Ex09.2's with the air's numbers and
the sign of the flux reversed, and the march must reproduce Ex09.2's exact
solution (:func:`check_zero_strength`). At B = 1 and S = 1 the wall condition
and the velocity depend on the unknown, and **no exact solution exists**.

## The problem, scaled as in Ex09.2

X = x/L, Z = z/h, s = u_in / (1 m/s), θ = (y - Y_IN)/Y with Y = N0 L/(u_ref h c)
= 0.114, G = u_ref h²/(D L) = 0.0526, g = N/N0, y_w = Y_IN + Y θ(X, 0):

    6 Z(1-Z) G s σ(X) θ_X - S G Y g(θ_w) f(Z) θ_Z = θ_ZZ      f(Z) = 1 - 3Z² + 2Z³
    θ = 0 at X = 0;   θ_Z = G g(θ_w) (1 - S y_w) at Z = 0;   θ_Z = 0 at Z = 1
    σ(X) = (1 - S Y_IN) / (1 - S ȳ(X))                         nitrogen conserved

σ is the molar flow left, relative to the inlet's, and ȳ the flow-weighted
(mixing-cup) mean of the oxygen fraction. The velocity is Poiseuille's profile
with the local mean speed s σ(X): the suction at the wall is a thousandth of
the axial speed (its Reynolds number is about 0.01), so the profile is not
distorted; the cross velocity v follows from continuity, v = -(N/c) f(Z).

The balance of the oxygen's molar flow, integrated over the height:

    s σ dθ̄/dX = - g(θ_w) (1 - S ȳ)

## Two solvers

* the **reference model** (:func:`reference`): Crank-Nicolson along the
  channel with Rannacher's start, central differences across, the nonlinear
  wall and velocity by three iterations per step, on an engineering grid; for
  the design speed, the secant method with that march. Timed against the
  network.
* the **ground truth** (:func:`ground_truth`): the same scheme on grids each
  twice as fine, converged and extrapolated (:func:`convergence_study`).
"""

from __future__ import annotations

import time

import numpy as np
from scipy.linalg import solve_banded

import channel as ch

# ------------------------------------------------------------------ the data
Y_IN = 0.21             #: oxygen fraction in the air fed
D_AIR = 1.9e-4          #: m²/s, oxygen in nitrogen at 800 degC   [estimate]
MU_AIR = 4.4e-5         #: Pa s, air at 800 degC   [estimate]
RHO_AIR = ch.C_TOT * (Y_IN * 32e-3 + (1 - Y_IN) * 28e-3)   #: kg/m³ of the air fed, 0.33
K_O2 = 0.05             #: the oxygen fraction at which r(y) = y/(y + K) is half its maximum   [ASSUMED]
B_COUPLE = 1.0          #: strength of the coupling: 0 is Ex09.2's uniform current
STEFAN = 1.0            #: strength of the one-sided exchange: 0 is Ex09.2's constant molar flow
N0 = ch.I_CELL / (4 * ch.FARADAY)   #: mol/(m² s), oxygen taken at the inlet's composition: four electrons per molecule
Y = N0 * ch.L_CHANNEL / (ch.U_REF * ch.H_CHANNEL * ch.C_TOT)   #: the scale of the fall of the oxygen fraction, 0.114
G = ch.U_REF * ch.H_CHANNEL ** 2 / (D_AIR * ch.L_CHANNEL)       #: Pe across times the aspect ratio, 0.0526
S_MIN, S_MAX = 1.0, 3.0         #: the range of air speeds at the inlet, m/s
S_TEST = (1.2, 1.9, 2.7)        #: m/s, the speeds the answers are scored at (not trained on)
X_SCORE = 0.05                  #: scored beyond the first 5 mm, as in Ex09.2
O2_TARGET = 0.15                #: the design asks for 15 % oxygen at the electrode at the outlet
TARGET_Y = 1e-3                 #: the agreed accuracy in the oxygen fraction at the electrode: 0.1 percentage point


def stoichiometry(s):
    """The air fed over the oxygen the inlet's current would take: λ = u h c y_in / (N0 L)."""
    return s * Y_IN / Y


def speed_for_stoichiometry(lam):
    return lam * Y / Y_IN


def reynolds(s):
    """The Reynolds number on the channel height, air fed at s m/s."""
    return RHO_AIR * s * ch.U_REF * ch.H_CHANNEL / MU_AIR


def pressure_drop(s):
    """Poiseuille's pressure drop over the channel (Pa), the inlet's speed throughout (an upper bound)."""
    return 12 * MU_AIR * s * ch.U_REF * ch.L_CHANNEL / ch.H_CHANNEL ** 2


def g_flux(theta_w, b=B_COUPLE):
    """g = N/N0 at the electrode's scaled oxygen fraction."""
    y = np.maximum(Y_IN + Y * np.asarray(theta_w, dtype=float), 0.0)
    return (1.0 - b) + b * (y / (y + K_O2)) / (Y_IN / (Y_IN + K_O2))


def oxygen(theta):
    """Scaled field back to the oxygen fraction."""
    return Y_IN + Y * np.asarray(theta)


def current_density(theta_w, b=B_COUPLE):
    """The local current density, A/cm²."""
    return ch.I_CELL / 1e4 * g_flux(theta_w, b)


def f_cross(Z):
    """The shape of the cross velocity: v = -(N/c) f(Z), f = 1 at the electrode, 0 at the interconnect.
    Takes a NumPy array or a torch tensor."""
    if not hasattr(Z, "requires_grad"):
        Z = np.asarray(Z, dtype=float)
    return 1.0 - 3.0 * Z ** 2 + 2.0 * Z ** 3


def exact_zero_strength(X, Z, s):
    """At B = 0 and S = 0: Ex09.2's exact solution with the air's G and the flux
    reversed, θ = -(X/s + G φ(Z)) (channel.phi is the same profile)."""
    return -(np.asarray(X, dtype=float) / s + G * ch.phi(Z))


def plug_flow(X, s, lib=np, iters=6):
    """The hand calculation of the mean, at B = 1 and S = 1: the gas mixed
    across the channel (y_w = ȳ), the current following it, the molar flow
    falling. Separable:

        (y + K) dy / (y (1 - y)²) = - Y dX / (r_in s (1 - Y_IN))

    whose left side integrates to F(y) = K ln y - K ln(1 - y) + (1 + K)/(1 - y).
    Solved for u = ln y by Newton's method from u = ln Y_IN; six steps reach
    rounding. ``lib`` is numpy or torch (differentiable). Returns θ̄(X, s)."""
    K, r_in = K_O2, Y_IN / (Y_IN + K_O2)
    F = lambda y: K * lib.log(y) - K * lib.log(1 - y) + (1 + K) / (1 - y)
    F_in = K * np.log(Y_IN) - K * np.log(1 - Y_IN) + (1 + K) / (1 - Y_IN)
    rhs = F_in - Y * X / (r_in * s * (1 - Y_IN))
    u = np.log(Y_IN) + 0.0 * rhs
    for _ in range(iters):
        y = lib.exp(u)
        u = u - (F(y) - rhs) / ((y + K) / (1 - y) ** 2)       # dF/du = y F'(y) = (y + K)/(1 - y)²
    return (lib.exp(u) - Y_IN) / Y


def sigma_plug(X, s, lib=np):
    """The molar flow left, relative to the inlet's, from the hand calculation's
    mean: σ_p = (1 - Y_IN)/(1 - y_p), nitrogen conserved. The physics loss of
    the notebook uses it in place of the network's own mean, which would need a
    quadrature column at every point (see check_sigma_plug for what it costs)."""
    return (1 - Y_IN) / (1 - (Y_IN + Y * plug_flow(X, s, lib=lib)))


# ---------------------------------------------------------- the march
class March:
    """Crank-Nicolson along the channel on an nx by nz grid, with the molar
    flow σ marched beside the field. Each step solves a tridiagonal system
    three times: the electrode's flux linearised on its node (Newton), the
    cross velocity and σ at the last estimate (they change by parts in ten
    thousand within a step). Built once per grid; run for any speed."""

    def __init__(self, nx, nz, b=B_COUPLE, stefan=STEFAN, cross=None, start=2, iters=3, sigma_from_plug=False):
        self.nx, self.nz, self.b, self.start, self.iters = nx, nz, b, start, iters
        self.sigma_from_plug = sigma_from_plug             # σ from the hand calculation instead of the march's own balance
        self.S = stefan                                     # the one-sided exchange: the 1 - y_w at the wall and the molar flow falling
        self.F = stefan                                     # (the two go together: one without the other loses oxygen)
        self.V = stefan if cross is None else cross         # the Stefan velocity's term in the equation, v θ_Z
        self.X, self.Z = np.linspace(0, 1, nx), np.linspace(0, 1, nz)
        self.hx, self.hz = 1.0 / (nx - 1), 1.0 / (nz - 1)
        w = ch.poiseuille(self.Z) * self.hz; w[[0, -1]] *= 0.5
        self.flow = w.sum()                                 # the trapezoid integral of 6 Z (1 - Z): normalised to carry the flow fed exactly
        self.wmean = w / w.sum()
        self.u = ch.poiseuille(self.Z) / self.flow * G      # the axial velocity row, times G, per unit speed
        self.f = f_cross(self.Z)

    def _wall(self, t0):
        """θ_Z at the electrode, G g (1 - S y_w), and its derivative in θ_0."""
        q = lambda t: G * g_flux(t, self.b) * (1 - self.S * oxygen(t))
        e = 1e-7
        return q(t0), (q(t0 + e) - q(t0 - e)) / (2 * e)

    def _L_matrix(self, a, d0):
        """The banded matrix of L(θ) = θ_ZZ + a θ_Z (a: the cross-velocity coefficient
        at each node), with the electrode's flux linearised: d0 is added to its node."""
        nz, hz = self.nz, self.hz
        lo, di, up = np.full(nz, 1 / hz ** 2) - a / (2 * hz), np.full(nz, -2 / hz ** 2), np.full(nz, 1 / hz ** 2) + a / (2 * hz)
        di[0] += d0
        up[0], lo[-1] = 2 / hz ** 2, 2 / hz ** 2            # the ghost nodes: the slope's part goes to the right-hand side
        return lo, di, up

    def _L_apply(self, th, a):
        """L(θ) for a column, with the electrode's flux from its own value."""
        hz = self.hz
        out = np.empty_like(th)
        out[1:-1] = (th[2:] - 2 * th[1:-1] + th[:-2]) / hz ** 2 + a[1:-1] * (th[2:] - th[:-2]) / (2 * hz)
        q, _ = self._wall(th[0])
        out[0] = (2 * th[1] - 2 * th[0]) / hz ** 2 - 2 * q / hz + a[0] * q
        out[-1] = (2 * th[-2] - 2 * th[-1]) / hz ** 2
        return out

    def _step(self, th, sig, s, h, cn):
        """One step of length h from the column (th, sig): Crank-Nicolson (cn) or backward Euler."""
        c = 0.5 if cn else 1.0
        a_old = self.V * G * Y * g_flux(th[0], self.b) * self.f
        L_old = self._L_apply(th, a_old) if cn else 0.0
        g_old = g_flux(th[0], self.b)
        new, sig_new = th.copy(), sig
        for _ in range(self.iters):
            g_new = g_flux(new[0], self.b)
            sig_new = sig - h * self.F * Y * ((1 - c) * g_old + c * g_new) / s
            if self.sigma_from_plug:
                sig, sig_new = (float(sigma_plug(x, s)) for x in (self._x, self._x + h))
            M = self.u * s * 0.5 * (sig + sig_new) / h        # the march term, the velocity at the step's middle
            a = self.V * G * Y * g_new * self.f
            q, dq = self._wall(new[0])
            wq = q * (a[0] - 2 / self.hz)                      # the flux's part of row 0
            dw = dq * (a[0] - 2 / self.hz)
            lo, di, up = self._L_matrix(a, 0.0)
            ab = np.zeros((3, self.nz))
            ab[0, 1:] = -c * up[:-1]; ab[1] = M - c * (di + np.r_[dw, np.zeros(self.nz - 1)]); ab[2, :-1] = -c * lo[1:]
            rhs = M * th + (1 - c) * L_old
            rhs[0] += c * (wq - dw * new[0])
            new = solve_banded((1, 1), ab, rhs)
        g_new = g_flux(new[0], self.b)
        sig_new = sig - h * self.F * Y * ((1 - c) * g_old + c * g_new) / s
        self._x += h
        return new, sig_new

    def run(self, s):
        """March at the air speed s (m/s). Returns θ (nz, nx) and σ (nx)."""
        th = np.zeros((self.nz, self.nx)); sig = np.ones(self.nx)
        self._x = 0.0
        for n in range(1, self.nx):
            col, sg = th[:, n - 1], sig[n - 1]
            if n <= self.start:
                for _ in range(2):                              # Rannacher's start: two backward-Euler half steps
                    col, sg = self._step(col, sg, s, 0.5 * self.hx, cn=False)
            else:
                col, sg = self._step(col, sg, s, self.hx, cn=True)
            th[:, n], sig[n] = col, sg
        self.theta, self.sigma, self.s = th, sig, s
        return th, sig

    def mean(self, th=None):
        th = self.theta if th is None else th
        return self.wmean @ th

    def balances(self):
        """Two checks at the outlet: the oxygen's molar flow (s σ ȳ at the outlet
        against s Y_IN minus what the wall took) and the nitrogen's (σ (1 - S ȳ)
        against 1 - S Y_IN). Returns both residuals, in units of the inlet's flow."""
        yb = oxygen(self.mean())
        taken = Y * np.trapezoid(g_flux(self.theta[0], self.b), self.X)
        o2 = (self.s * self.sigma[-1] * yb[-1] - (self.s * Y_IN - self.F * taken)) / (self.s * Y_IN)
        n2 = self.sigma[-1] * (1 - self.S * yb[-1]) - (1 - self.S * Y_IN)
        return float(o2), float(n2)


def electrode(th):
    return oxygen(th[0])


# ------------------------------------------------------- the reference model
REF_GRID = (21, 4)      #: nx, nz: the engineering grid, the coarsest that meets 0.1 percentage point and the speed within 1 %


def reference(s, grid=None):
    """The reference model at one air speed: the march on the engineering grid."""
    m = March(*(grid or REF_GRID))
    t0 = time.perf_counter()
    th, sig = m.run(s)
    sec = time.perf_counter() - t0
    return dict(X=m.X, y_wall=electrode(th), seconds=sec, theta=th, sigma=sig, Z=m.Z)


def speed_for_oxygen(target=O2_TARGET, grid=None, tol=1e-6):
    """The air speed (m/s) at which the electrode sees ``target`` oxygen at the
    outlet: the secant method, one march per step, started from the uniform-
    current, constant-flow hand calculation. Returns the speed, the marches, the time (s)."""
    m = March(*(grid or REF_GRID))
    t0 = time.perf_counter()
    f = lambda s: oxygen(m.run(s)[0][0, -1]) - target
    a = Y / (Y_IN - target); b = 1.1 * a
    fa, fb = f(a), f(b); n = 2
    for _ in range(30):
        c = b - fb * (b - a) / (fb - fa)
        a, fa = b, fb
        b, fb = c, f(c); n += 1
        if abs(b - a) < tol * b:
            break
    return b, n, time.perf_counter() - t0


# -------------------------------------------------------- the ground truth
TRUTH_GRIDS = ((201, 11), (401, 21), (801, 41), (1601, 81), (3201, 161))
TRUTH_GRID = TRUTH_GRIDS[-2]


def on_grid(th, X, Z, Xc, Zc):
    ix = np.rint(Xc * (len(X) - 1)).astype(int); iz = np.rint(Zc * (len(Z) - 1)).astype(int)
    return th[np.ix_(iz, ix)]


def convergence_study(s=1.9, grids=TRUTH_GRIDS, verbose=True):
    """The electrode's oxygen fraction on grids each twice as fine; Richardson on the last two."""
    rows = []
    Xc = np.linspace(0, 1, grids[0][0])
    for g in grids:
        m = March(*g)
        t0 = time.perf_counter()
        th, sig = m.run(s)
        sec = time.perf_counter() - t0
        yw = oxygen(on_grid(th, m.X, m.Z, Xc, np.array([0.0]))[0])
        o2, n2 = m.balances()
        rows.append(dict(grid=g, y_wall=yw, sigma_out=sig[-1], o2=o2, n2=n2, seconds=sec))
        if verbose:
            print(f"   grid {g[0]:5d} x {g[1]:4d}: oxygen at the electrode at the outlet {100 * yw[-1]:.5f} %, "
                  f"molar flow left {100 * sig[-1]:.4f} %, balances off by {abs(o2):.1e} (oxygen), {abs(n2):.1e} (nitrogen)   ({sec:.2f} s)")
    yw_e = rows[-1]["y_wall"] + (rows[-1]["y_wall"] - rows[-2]["y_wall"]) / 3
    if verbose:
        k = Xc >= X_SCORE
        for i in range(1, len(rows)):
            d = np.abs(rows[i]["y_wall"] - rows[i - 1]["y_wall"])[k].max()
            print(f"   change of the electrode fraction beyond 5 mm, grid {i} to {i + 1}: {d:.2e}")
        print(f"   extrapolated: oxygen at the electrode at the outlet {100 * yw_e[-1]:.5f} %")
    return rows, yw_e


def ground_truth(s, grid=TRUTH_GRID, **kw):
    m = March(*grid, **kw)
    th, sig = m.run(s)
    return dict(X=m.X, Z=m.Z, theta=th, sigma=sig, y_wall=electrode(th), y_mean=oxygen(m.mean()), balances=m.balances())


def check_zero_strength(grid=(1601, 81), speeds=(1.2, 1.9, 2.7), verbose=True):
    """At B = 0 and S = 0 the problem is Ex09.2's with the air's numbers, reversed:
    the march must reproduce exact_zero_strength beyond the entrance."""
    worst = 0.0
    for s in speeds:
        m = March(*grid, b=0.0, stefan=0.0)
        th, _ = m.run(s)
        XX, ZZ = np.meshgrid(m.X, m.Z)
        e = Y * np.abs(th - exact_zero_strength(XX, ZZ, s))[XX >= X_SCORE].max()
        worst = max(worst, e)
        if verbose:
            print(f"   B = 0, S = 0, u = {s:.1f} m/s: largest difference from Ex09.2's exact solution (air, reversed): {e:.1e} in the oxygen fraction")
    return worst


def check_sigma_plug(s=1.2, grid=(801, 41)):
    """The largest change of the electrode's oxygen fraction (beyond 5 mm) when
    the march takes the molar flow σ from the hand calculation's mean instead
    of its own balance: what the notebook's physics loss gives up by doing so."""
    a = ground_truth(s, grid)["y_wall"]
    b = ground_truth(s, grid, sigma_from_plug=True)["y_wall"]
    X = np.linspace(0, 1, grid[0])
    return float(np.abs(a - b)[X >= X_SCORE].max())


def stefan_study(s, grid=TRUTH_GRID):
    """The oxygen at the electrode at the outlet (i) with the whole one-sided
    exchange, (ii) with the molar flow falling and the 1 - y_w at the wall but
    without the Stefan velocity's term v θ_Z in the equation, and (iii) with
    neither, Ex09.2's equimolar model with the air's numbers. Returns the three."""
    full = ground_truth(s, grid)["y_wall"][-1]
    no_cross = ground_truth(s, grid, cross=0.0)["y_wall"][-1]
    none = ground_truth(s, grid, stefan=0.0)["y_wall"][-1]
    return full, no_cross, none


# ------------------------------------------------------- the training points
def sample_channel(n, seed=88):
    """n points (X, Z, s) over the channel and the range of speeds, Latin
    hypercube; a fifth of them in the first 5 % of the length."""
    rng = np.random.default_rng(seed)
    u = (rng.permuted(np.tile(np.arange(n), (3, 1)), axis=1).T + rng.random((n, 3))) / n
    u[: n // 5, 0] *= 0.05
    u[:, 2] = S_MIN + (S_MAX - S_MIN) * u[:, 2]
    return u


def sample_faces(n, seed=88):
    """Points on the inlet, the electrode and the interconnect, at random speeds."""
    rng = np.random.default_rng(seed + 1)
    s = lambda m: S_MIN + (S_MAX - S_MIN) * rng.random(m)
    inlet = np.c_[np.zeros(n), rng.random(n), s(n)]
    x = rng.random(n); x[: n // 5] *= 0.05
    electrode_ = np.c_[x, np.zeros(n), s(n)]
    x2 = rng.random(n); x2[: n // 5] *= 0.05
    interconnect = np.c_[x2, np.ones(n), s(n)]
    return inlet, electrode_, interconnect


# ----------------------------------------------------------- the truth file
TRUTH_FILE = "MP9.2B_truth.npz"


def write_truth(path=TRUTH_FILE, grid=TRUTH_GRID):
    rows, yw_e = convergence_study(1.9, verbose=True)
    s_star, n_march, _ = speed_for_oxygen(grid=grid)
    s5 = speed_for_stoichiometry(5.0)
    out = dict(speeds=np.array(S_TEST), grid=np.array(grid, dtype=float), speed_15=np.array(s_star), speed_stoich5=np.array(s5))
    Xs, Zs = np.linspace(0, 1, 401), np.linspace(0, 1, 41)
    for s, key in [(v, f"{v:.1f}") for v in S_TEST] + [(s_star, "design"), (s5, "stoich5")]:
        gt = ground_truth(s, grid)
        out[f"field_{key}"] = oxygen(on_grid(gt["theta"], gt["X"], gt["Z"], Xs, Zs))
        out[f"y_wall_{key}"] = gt["y_wall"]; out[f"y_mean_{key}"] = gt["y_mean"]; out[f"sigma_{key}"] = gt["sigma"]
        out["X_wall"] = gt["X"]
    full, flow_only, none = stefan_study(s5, grid)
    out["stefan_stoich5"] = np.array([full, flow_only, none])
    out["X"], out["Z"] = Xs, Zs
    out["conv_grids"] = np.array([r["grid"] for r in rows], dtype=float)
    out["conv_y_wall_outlet"] = np.array([r["y_wall"][-1] for r in rows])
    out["conv_balance_o2"] = np.array([r["o2"] for r in rows]); out["conv_balance_n2"] = np.array([r["n2"] for r in rows])
    out["conv_y_wall_outlet_extrapolated"] = np.array(yw_e[-1])
    out["zero_strength_error"] = np.array(check_zero_strength(verbose=False))
    out["params"] = np.array([f"{k} = {v}" for k, v in dict(
        Y_IN="0.21 oxygen fraction of the air fed", D_AIR="1.9e-4 m2/s, estimate", MU_AIR="4.4e-5 Pa s, estimate",
        K_O2="0.05, ASSUMED", B_COUPLE=B_COUPLE, STEFAN=STEFAN, I0="0.5 A/cm2 at the inlet's composition, typical",
        N0=f"{N0:.4e} mol/(m2 s), i0/4F", L="0.10 m", H="1.0e-3 m", T="800 degC", P="1 atm", C=f"{ch.C_TOT:.3f} mol/m3",
        S_RANGE="1.0-3.0 m/s").items()])
    out["README"] = np.array(
        "MP9.2B ground truth. speeds: the three test air speeds at the inlet (m/s); field_<s>: oxygen fraction (nz=41, nx=401) "
        "on Z, X (scaled); y_wall_<s>, y_mean_<s>, sigma_<s>: oxygen fraction at the electrode, flow-weighted mean, and molar "
        "flow left relative to the inlet, on X_wall (1601 nodes); *_design: at speed_15, the air speed (m/s) giving 15 % oxygen "
        "at the electrode at the outlet; *_stoich5: at speed_stoich5, stoichiometry 5; stefan_stoich5: oxygen at the electrode "
        "at the outlet at stoichiometry 5 with the full one-sided exchange, without the Stefan velocity's term in the equation, and with neither (Ex09.2's equimolar model); "
        "conv_*: the convergence study at 1.9 m/s, grids (nx, nz), balances of oxygen and nitrogen at the outlet; "
        "zero_strength_error: the march at B = 0, S = 0 against Ex09.2's exact solution (air, reversed); params: the data.")
    np.savez_compressed(path, **out)
    print("written", path, f"; speed for 15 % at the electrode {s_star:.5f} m/s ({n_march} marches)")


if __name__ == "__main__":
    import sys
    print("zero-strength check")
    check_zero_strength()
    print("convergence at 1.9 m/s")
    convergence_study()
    if "--save" in sys.argv:
        write_truth()
