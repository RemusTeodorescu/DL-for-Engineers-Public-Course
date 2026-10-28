r"""MP7.2A - A die with a moving hot spot. Ex07.2's silicon die, now powered by
three blocks that switch on in turn, with silicon's conductivity falling as it
warms. The physics, the reference model and the ground truth.

*Deep Learning for Engineering* - MSc, Aalborg University.
Remus Teodorescu (ret@et.aau.dk), with support from Research Assistant
Noman Khan (nomank@energy.aau.dk).

This module extends Ex07.2's ``problem.py`` (course policy C10 rule 1: one
set of physics and constants for the exercise and its mini project) and is
used by the notebook of MP7.2A (course policy C14).

## What changes against Ex07.2, and why

Ex07.2's die (notebook 1) is released from a given field and cools with one
constant diffusivity, 8.8e-5 m²/s; its solution is two decaying modes. A
working die is powered: here three 2 x 2 mm blocks of P watts each switch on
in turn for 60 ms, and silicon's conductivity falls as it warms,

    k(T) = K300 (300 K / T)^(1.3 S)        (T in kelvin)

the usual fit to the measured conductivity of pure silicon (Glassbrenner and
Slack, Phys. Rev. 134 (1964) A1058; the fit from memory, ASSUMED until
checked). S is the strength of the change: at S = 0 the conductivity is the
constant 148 W/(m K) of Ex07.2's diffusivity, and with Ex07.2's initial field
and no power the solver must reproduce Ex07.2's exact two modes
(:func:`check_zero_strength`). At S = 1 the heat equation is nonlinear and
**no exact solution exists**.

## The problem, in kelvin above the package (65 C)

    rho_c dtheta/dt = div(k(T) grad theta) + q(x, y, t; P)    in the die
    theta = 0 on the four edges,  theta = 0 at t = 0

q is P / (A_block * thickness) inside the block that is on, zero elsewhere.

## Two solvers, one scheme

Vertex-centred finite volumes (one control volume round every node, so the
nodes of a coarse grid are nodes of every finer one and the block centres
and edges are nodes), Crank-Nicolson in time with the conductivity taken
from the field extrapolated to the half step (second order, one linear solve
per step), and Rannacher's start - two backward-Euler half steps - after
every switch. Each linear system is symmetric and solved by conjugate
gradients.

* the **reference model** (:func:`reference`): the scheme on an engineering
  grid, timed.
* the **ground truth** (:func:`ground_truth`): the scheme on grids each twice
  as fine, Richardson-extrapolated (:func:`convergence_study`).
"""

from __future__ import annotations

import time

import numpy as np

import problem as pb

# ------------------------------------------------------------------ the data
L = pb.L_DIE                   #: m, the die's edge, 10 mm (Ex07.2)
K300 = 148.0                   #: W/(m K), silicon at 300 K (Glassbrenner and Slack 1964; ASSUMED until checked)
K_EXP = 1.3                    #: the exponent of the fit k = K300 (300/T)^1.3 (ASSUMED until checked)
RHO_C = K300 / pb.ALPHA        #: J/(m^3 K), 1.68 MJ/(m^3 K): Ex07.2's diffusivity at 300 K
S_K = 1.0                      #: the strength of k(T): 0 is Ex07.2's constant conductivity
T_PKG = pb.T_PACKAGE           #: C, the package temperature the edges are held at (65 C)
THICK = 0.5e-3                 #: m, the die's thickness (ASSUMED, a typical thinned die)
BLOCK = 2.0e-3                 #: m, a block's side
#: the three blocks: centre (m) and the instants they switch on and off (s)
BLOCKS = (((2.5e-3, 2.5e-3), 0.00, 0.06), ((7.5e-3, 2.5e-3), 0.06, 0.12), ((5.0e-3, 7.5e-3), 0.12, 0.18))
P_MIN, P_MAX = 5.0, 10.0       #: W per block, the range of the power, an input of the network (ASSUMED range)
P0 = 8.0                       #: W per block, the worked example: 200 W/cm^2 through the die (ASSUMED)
P_TEST = (5.5, 7.25, 9.5)      #: W, the powers the answers are scored at (not trained on)
P_OUT = (4.0, 11.0)            #: W, two powers outside the range, for a question of the report
T_END = 0.20                   #: s, the window
T_REC = np.round(np.arange(0.0, T_END + 1e-9, 0.002), 6)   #: s, the recorded instants: every 2 ms
SWITCHES = np.array(sorted({b[1] for b in BLOCKS} | {b[2] for b in BLOCKS}))   #: s, 0, 60, 120, 180 ms
H_SCORE = 0.25e-3              #: m, the scoring nodes: every 0.25 mm, edges included (41 x 41)
#: the agreed accuracy (part A)
PEAK_TOL, PEAK_T_TOL, FIELD_TOL = 1.0, 0.002, 2.0     #: K, s, K


def q_block(P):
    """W/m^3, the heat density in a block of power P (W): 4e9 at 8 W."""
    return P / (BLOCK * BLOCK * THICK)


def k_si(theta, s=S_K):
    """W/(m K), silicon's conductivity at the rise theta (K) above the package."""
    return K300 * (300.0 / (T_PKG + 273.15 + theta)) ** (K_EXP * s)


def dk_si(theta, s=S_K):
    """W/(m K^2), dk/dtheta."""
    return -K_EXP * s * k_si(theta, s) / (T_PKG + 273.15 + theta)


def power_on(t):
    """Which block is on at the instant t (s): an integer 0, 1, 2, or -1 for none."""
    for i, (_, on, off) in enumerate(BLOCKS):
        if on <= t < off:
            return i
    return -1


def block_share(x, y, i, h=0.0):
    """The share of the control volume of side h round each node (x, y) [m]
    that lies in block i: 1 inside, 1/2 on an edge, 1/4 on a corner (h > 0),
    or the plain indicator (h = 0)."""
    (cx, cy), _, _ = BLOCKS[i]

    def frac(z, c):
        if h == 0.0:
            return (np.abs(z - c) < BLOCK / 2).astype(float)
        lo, hi = np.maximum(z - h / 2, c - BLOCK / 2), np.minimum(z + h / 2, c + BLOCK / 2)
        return np.clip(hi - lo, 0.0, None) / h
    return frac(np.asarray(x, float), cx) * frac(np.asarray(y, float), cy)


# ---------------------------------------------------------- the time march
class Die:
    """Vertex-centred finite volumes on n x n cells of side h = L/n (nodes at
    i h, edges at 0 and n), Crank-Nicolson in steps of dt with Rannacher's
    start after every switch. Built once per grid; ``run`` marches for any
    power."""

    def __init__(self, n, dt, s=S_K):
        self.n, self.h, self.dt, self.s = n, L / n, dt, s
        self.x = np.linspace(0.0, L, n + 1)                       # nodes, edges included
        X, Y = np.meshgrid(self.x[1:-1], self.x[1:-1])            # the unknowns: interior nodes (row = y)
        self.share = [block_share(X, Y, i, self.h) for i in range(3)]
        self.cg_iters = 0

    # the conduction operator, -div(k grad theta), on the interior nodes [W/m^3]
    def faces(self, th):
        """Face conductivities (arithmetic mean of the two nodes' k) for the
        field th (interior nodes), with the edges at theta = 0."""
        p = np.pad(th, 1)                                         # the edge nodes, theta = 0
        k = k_si(p, self.s)
        return 0.5 * (k[:, 1:] + k[:, :-1]), 0.5 * (k[1:, :] + k[:-1, :])   # kx (m+2, m+1), ky (m+1, m+2)

    def apply(self, th, kx, ky):
        p = np.pad(th, 1)
        fx = kx * (p[:, 1:] - p[:, :-1])                          # k dtheta between neighbours along x
        fy = ky * (p[1:, :] - p[:-1, :])
        return -((fx[1:-1, 1:] - fx[1:-1, :-1]) + (fy[1:, 1:-1] - fy[:-1, 1:-1])) / self.h ** 2

    def diag(self, kx, ky):
        return (kx[1:-1, 1:] + kx[1:-1, :-1] + ky[1:, 1:-1] + ky[:-1, 1:-1]) / self.h ** 2

    def solve(self, c, w, kx, ky, rhs, x0, tol=1e-11):
        """(c I + w K) x = rhs by conjugate gradients with the diagonal as
        preconditioner; symmetric positive definite."""
        Minv = 1.0 / (c + w * self.diag(kx, ky))
        x = x0.copy()
        r = rhs - (c * x + w * self.apply(x, kx, ky))
        z = Minv * r
        p = z.copy()
        rz = np.vdot(r, z)
        bnorm = np.linalg.norm(rhs) + 1e-300
        for it in range(2000):
            Ap = c * p + w * self.apply(p, kx, ky)
            a = rz / np.vdot(p, Ap)
            x += a * p
            r -= a * Ap
            if np.linalg.norm(r) < tol * bnorm:
                break
            z = Minv * r
            rz_new = np.vdot(r, z)
            p = z + (rz_new / rz) * p
            rz = rz_new
        self.cg_iters += it + 1
        return x

    def wall_flow(self, th, kx, ky):
        """W per metre of depth, the heat leaving through the four edges (the
        flux k dtheta/dn at each edge face, as the control volumes see it)."""
        p = np.pad(th, 1)
        out = (kx[1:-1, 0] * p[1:-1, 1]).sum() + (kx[1:-1, -1] * p[1:-1, -2]).sum() \
            + (ky[0, 1:-1] * p[1, 1:-1]).sum() + (ky[-1, 1:-1] * p[-2, 1:-1]).sum()
        return out                                                # the h of the face length cancels the 1/h of the slope

    def run(self, P=P0, record=T_REC, theta0=None, keep_peak=True):
        """March the window at a block power P (W). ``record``: instants (s,
        multiples of dt) at which the field (n+1, n+1, edges included) is kept.
        ``theta0``: an initial field on the interior nodes (default zero).
        Returns a dict: t, fields, the peak (K) and its instant (s), the
        energy balance (J, through the die's thickness)."""
        m = self.n - 1
        dt, h = self.dt, self.h
        th = np.zeros((m, m)) if theta0 is None else np.array(theta0, float)
        prev = th.copy()
        nt = int(round(T_END / dt))
        rec = {int(round(r / dt)): r for r in record}
        fields, times = [], []
        if 0 in rec:
            fields.append(np.pad(th, 1)); times.append(0.0)
        peak, t_peak = float(th.max()), 0.0
        heat_in = heat_out = 0.0
        stored0 = RHO_C * th.sum() * h * h
        last = None
        for s in range(1, nt + 1):
            t_mid = (s - 0.5) * dt
            b = power_on(t_mid)
            q = q_block(P) * self.share[b] if b >= 0 else np.zeros((m, m))
            if b != last:
                # Rannacher's start: the source has just jumped; two backward-Euler half steps damp
                # what Crank-Nicolson would carry on as an undamped oscillation
                new = th
                for _ in range(2):
                    kx, ky = self.faces(new)
                    c = RHO_C / (dt / 2)
                    nxt = self.solve(c, 1.0, kx, ky, c * new + q, new)
                    heat_out += (dt / 2) * self.wall_flow(nxt, kx, ky)
                    new = nxt
            else:
                star = 1.5 * th - 0.5 * prev                       # the field extrapolated to the half step
                kx, ky = self.faces(star)
                c = RHO_C / dt
                rhs = c * th - 0.5 * self.apply(th, kx, ky) + q
                new = self.solve(c, 0.5, kx, ky, rhs, 2 * th - prev)
                heat_out += 0.5 * dt * (self.wall_flow(th, kx, ky) + self.wall_flow(new, kx, ky))
            heat_in += dt * q.sum() * h * h
            last = b
            prev, th = th, new
            if keep_peak and th.max() > peak:
                peak, t_peak = float(th.max()), s * dt
            if s in rec:
                fields.append(np.pad(th, 1)); times.append(rec[s])
        stored = RHO_C * th.sum() * h * h - stored0
        bal = dict(heat_in=heat_in * THICK, heat_out=heat_out * THICK, stored=stored * THICK)
        bal["residual"] = bal["heat_in"] - bal["heat_out"] - bal["stored"]
        return dict(t=np.array(times), fields=np.array(fields), peak=peak, t_peak=t_peak, balance=bal, x=self.x)


def on_score(field, n):
    """A field on the nodes of an n-cell grid, (n+1, n+1) or (k, n+1, n+1),
    read at the scoring nodes, every 0.25 mm (41 x 41): every n/40-th node
    when n is a multiple of 40, else bilinear between the grid's nodes."""
    ns = round(L / H_SCORE)
    if n % ns == 0:
        return field[..., ::n // ns, ::n // ns]
    x, xs = np.linspace(0, L, n + 1), np.linspace(0, L, ns + 1)
    i = np.clip(np.searchsorted(x, xs, side="right") - 1, 0, n - 1)
    w = (xs - x[i]) / (x[1] - x[0])
    f = field[..., i, :] * (1 - w)[:, None] + field[..., i + 1, :] * w[:, None]       # along y (rows)
    return f[..., i] * (1 - w) + f[..., i + 1] * w                                       # along x (columns)


# ------------------------------------------------- the linear part, exactly
def linear_modes(P=P0, M=120, k_const=None):
    """The exact solution of the problem with one constant conductivity
    k_const (default: silicon's at the package temperature, 126.7 W/(m K)),
    as a sum of the die's modes sin(m pi x/L) sin(n pi y/L), m, n = 1..M.
    Each block's heat is projected on the modes; each mode then rises and
    falls as a first-order system. Returns a function theta(x, y, t) for 1-D
    node coordinates x, y (m) and an instant t (s), the field (len(y), len(x)) [K]."""
    k = k_si(0.0) if k_const is None else k_const
    m = np.arange(1, M + 1)
    lam = (np.pi / L) ** 2 * (m[:, None] ** 2 + m[None, :] ** 2)       # (n, m) 1/m^2
    kappa = k * lam / RHO_C                                            # 1/s, each mode's decay rate
    proj = []
    for (cx, cy), on, off in BLOCKS:
        ix = L / (m * np.pi) * (np.cos(m * np.pi * (cx - BLOCK / 2) / L) - np.cos(m * np.pi * (cx + BLOCK / 2) / L))
        iy = L / (m * np.pi) * (np.cos(m * np.pi * (cy - BLOCK / 2) / L) - np.cos(m * np.pi * (cy + BLOCK / 2) / L))
        proj.append((4 / L ** 2 * q_block(P) * iy[:, None] * ix[None, :] / (k * lam), on, off))   # (n, m) K

    def coeffs(t):
        c = np.zeros_like(lam)
        for a, on, off in proj:
            if t > on:
                c += a * (1 - np.exp(-kappa * (t - on)))
            if t > off:
                c -= a * (1 - np.exp(-kappa * (t - off)))
        return c

    def theta(x, y, t):
        sx = np.sin(np.outer(x, m) * np.pi / L)                        # (len(x), M)
        sy = np.sin(np.outer(y, m) * np.pi / L)
        return sy @ coeffs(t) @ sx.T
    theta.coeffs, theta.m = coeffs, m
    return theta


def check_zero_strength(n=160, dt=0.5e-3, verbose=True):
    """Two checks at S = 0 (constant conductivity, Ex07.2's 148 W/(m K)).
    (1) No power, Ex07.2's initial field: the march must give Ex07.2's exact
    two decaying modes. (2) The blocks on, from zero: the march must give
    the exact modal sum of :func:`linear_modes`. Returns the two largest
    differences over the recorded instants (K)."""
    d = Die(n, dt, s=0.0)
    X, Y = np.meshgrid(d.x[1:-1], d.x[1:-1])
    r = d.run(P=0.0, theta0=pb.heat_initial(X, Y), keep_peak=False)
    XX, YY = np.meshgrid(d.x, d.x)
    e1 = max(float(np.abs(f - pb.heat_exact(XX, YY, t)).max()) for f, t in zip(r["fields"], r["t"]))
    r = d.run(P=P0, keep_peak=False)
    lin = linear_modes(P0, M=400, k_const=K300)
    e2 = max(float(np.abs(f - lin(d.x, d.x, t)).max()) for f, t in zip(r["fields"][::10], r["t"][::10]))
    if verbose:
        print(f"   S = 0, no power, Ex07.2's start: largest difference from Ex07.2's exact modes {e1:.2e} K "
              f"(on a 60 K start)")
        print(f"   S = 0, the blocks on: largest difference from the exact modal sum {e2:.2e} K")
    return e1, e2


# ------------------------------------------------------- the reference model
REF_GRID = (20, 2.0e-3)        #: cells across, time step (s): the engineering grid, chosen in the convergence study


def reference(P, grid=REF_GRID):
    """The reference model: the march on the engineering grid at the power P.
    Returns the fields at the recorded instants on the scoring nodes (101, 41,
    41) [K], the peak [K], its instant [s] and the wall-clock time [s]."""
    t0 = time.perf_counter()
    r = Die(*grid).run(P)
    sec = time.perf_counter() - t0
    return dict(fields=on_score(r["fields"], grid[0]), peak=r["peak"], t_peak=r["t_peak"], seconds=sec)


# -------------------------------------------------------- the ground truth
TRUTH_GRIDS = ((40, 2.0e-3), (80, 1.0e-3), (160, 0.5e-3), (320, 0.25e-3))   #: cells, dt (s): each twice as fine


def convergence_study(P=P0, grids=TRUTH_GRIDS, verbose=True):
    """The peak and the field at the scoring nodes on grids each twice as fine
    in space and time. Returns rows (cells, dt, peak, t_peak, field change
    from the grid before [K], energy residual [-], CG iterations per step,
    seconds) and the fields of the last two grids (for Richardson)."""
    rows, last = [], []
    for n, dt in grids:
        t0 = time.perf_counter()
        d = Die(n, dt)
        r = d.run(P)
        f = on_score(r["fields"], n)
        change = float(np.abs(f - last[-1]).max()) if last else np.nan
        res = r["balance"]["residual"] / r["balance"]["heat_in"]
        rows.append((n, dt, r["peak"], r["t_peak"], change, res, d.cg_iters / round(T_END / dt), time.perf_counter() - t0))
        last.append(f)
        if verbose:
            print(f"   {n:4d} x {n} cells of {L/n*1e3:.4f} mm, dt {dt*1e3:.2f} ms: peak {r['peak']:.4f} K at "
                  f"{r['t_peak']*1e3:.2f} ms, field change {change:.4f} K, balance off by {abs(res):.1e}, "
                  f"{rows[-1][6]:.0f} CG iterations a step  ({rows[-1][7]:.1f} s)", flush=True)
    return rows, last


def ground_truth(P, grids=TRUTH_GRIDS[-2:]):
    """The ground truth at the power P: the march on the two finest grids,
    combined by Richardson extrapolation at order two, (4 fine - coarse) / 3,
    at the scoring nodes and for the peak. Returns the fields (101, 41, 41)
    [K], the peak [K], its instant [s], the energy balance of the finest run
    and the largest Richardson correction [K]."""
    runs = [Die(*g).run(P) for g in grids]
    f = [on_score(r["fields"], g[0]) for r, g in zip(runs, grids)]
    field = (4 * f[1] - f[0]) / 3
    peak = (4 * runs[1]["peak"] - runs[0]["peak"]) / 3
    return dict(fields=field, peak=peak, t_peak=runs[1]["t_peak"], balance=runs[1]["balance"],
                correction=float(np.abs(field - f[1]).max()))


def linear_peak(P=P0, n=401):
    """The peak with silicon's conductivity held at its value at the package
    temperature (126.7 W/(m K)): the largest value of the exact modal sum on
    an n x n grid at the end of the third pulse [K]. What a linear model gives."""
    x = np.linspace(0.0, L, n)
    return float(linear_modes(P, M=400)(x, x, BLOCKS[2][2]).max())


# ------------------------------------------------------- the training points
class Linear:
    """The exact solution of the linear problem (the conductivity held at its
    value at the package temperature) at P0, and its derivatives, at any
    points: a sum of M x M modes of the die. Scale by P / P0 for another
    power. Usable to build the linear part into a network (part B)."""

    def __init__(self, M=60):
        self.M = M
        m = np.arange(1, M + 1)
        self.m = m
        self.k = k_si(0.0)
        self.lam = (np.pi / L) ** 2 * (m[:, None] ** 2 + m[None, :] ** 2)
        self.kappa = self.k * self.lam / RHO_C
        self.proj = []
        for (cx, cy), on, off in BLOCKS:
            ix = L / (m * np.pi) * (np.cos(m * np.pi * (cx - BLOCK / 2) / L) - np.cos(m * np.pi * (cx + BLOCK / 2) / L))
            iy = L / (m * np.pi) * (np.cos(m * np.pi * (cy - BLOCK / 2) / L) - np.cos(m * np.pi * (cy + BLOCK / 2) / L))
            self.proj.append((4 / L ** 2 * q_block(P0) * iy[:, None] * ix[None, :] / (self.k * self.lam), on, off))

    def coeffs(self, t):
        """The modal amplitudes at the instants t (n,): (n, M, M) [K]."""
        t = np.atleast_1d(np.asarray(t, float))
        c = np.zeros((len(t), self.M, self.M))
        for a, on, off in self.proj:
            c += a * (1 - np.exp(-self.kappa * np.clip(t - on, 0, None)[:, None, None]))
            c -= a * (1 - np.exp(-self.kappa * np.clip(t - off, 0, None)[:, None, None]))
        return c

    def at(self, x, y, t, chunk=2000):
        """theta, dtheta/dx, dtheta/dy and the Laplacian at points (x, y, t)
        [m, m, s], each (n,): an array (4, n) [K, K/m, K/m, K/m^2] at P0."""
        x, y, t = (np.asarray(v, float).ravel() for v in (x, y, t))
        out = np.zeros((4, len(x)))
        w = self.m * np.pi / L
        for s in range(0, len(x), chunk):
            sl = slice(s, s + chunk)
            C = self.coeffs(t[sl])
            sx, sy = np.sin(np.outer(x[sl], w)), np.sin(np.outer(y[sl], w))
            cx, cy = np.cos(np.outer(x[sl], w)) * w, np.cos(np.outer(y[sl], w)) * w
            out[0, sl] = np.einsum("pn,pnm,pm->p", sy, C, sx)
            out[1, sl] = np.einsum("pn,pnm,pm->p", sy, C, cx)
            out[2, sl] = np.einsum("pn,pnm,pm->p", cy, C, sx)
            out[3, sl] = np.einsum("pn,pnm,pm->p", sy, -self.lam * C, sx)
        return out

    def at_points(self, x, y, ts):
        """theta at P0 at the points (x, y) [m] (1-D each) at every instant of
        ts [s]: (len(ts), len(x)) [K]."""
        w = self.m * np.pi / L
        sx, sy = np.sin(np.outer(x, w)), np.sin(np.outer(y, w))
        return np.einsum("tpm,pm->tp", np.matmul(sy[None], self.coeffs(ts)), sx)

    def grid(self, xs, ts):
        """theta at P0 on the nodes xs x xs (1-D, m) at the instants ts (s):
        (len(ts), len(xs), len(xs)) [K]."""
        w = self.m * np.pi / L
        S = np.sin(np.outer(xs, w))
        return np.einsum("in,tnm,jm->tij", S, self.coeffs(ts), S)


def sample_times(n, rng, near=0.3, spread=0.005):
    """n instants in the window (s): a share ``near`` of them just after a
    switch, exponentially distributed with mean ``spread`` (s); the rest uniform."""
    k = int(n * near)
    after = SWITCHES[rng.integers(0, SWITCHES.size, k)] + rng.exponential(spread, k)
    return np.clip(np.r_[after, rng.uniform(0, T_END, n - k)], 0.0, T_END)


def sample_interior(n, seed=72, near=0.6, near_t=0.3):
    """n training points (x, y, t, P) [m, m, s, W]: instants by
    :func:`sample_times`; a share ``near`` of the points within 2.5 mm of the
    centre of the block that is on (or was on last), where the field is
    steepest, the rest uniform over the die; P uniform over the range."""
    rng = np.random.default_rng(seed)
    t = sample_times(n, rng, near_t)
    x, y = rng.uniform(0, L, n), rng.uniform(0, L, n)
    close = rng.random(n) < near
    for i, ((cx, cy), on, off) in enumerate(BLOCKS):
        sel = close & (t >= on) & ((t < off) | (i == len(BLOCKS) - 1))
        x[sel] = cx + rng.uniform(-2.5e-3, 2.5e-3, sel.sum())
        y[sel] = cy + rng.uniform(-2.5e-3, 2.5e-3, sel.sum())
    return x, y, t, rng.uniform(P_MIN, P_MAX, n)


def heat(x, y, t, P):
    """W/m^3, the heat density q at points (x, y, t) for the power P, (n,) each."""
    x, y, t, P = (np.broadcast_to(np.asarray(v, float), np.shape(x)) for v in (x, y, t, P))
    q = np.zeros(np.shape(x))
    for i, ((cx, cy), on, off) in enumerate(BLOCKS):
        sel = (t >= on) & (t < off) & (np.abs(x - cx) < BLOCK / 2) & (np.abs(y - cy) < BLOCK / 2)
        q[sel] = q_block(P[sel])
    return q


def score_nodes():
    """The 41 x 41 scoring nodes, every 0.25 mm, edges included: x, y (m), 1-D."""
    return np.linspace(0.0, L, round(L / H_SCORE) + 1)


# ------------------------------------------------------------------ pictures
def draw_die(ax=None):
    """The die of MP7.2A: the 10 mm square of silicon with its edges held at
    the package temperature, and the three blocks with the instants they are on."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(5.4, 4.6))
    Lm = L * 1e3
    ax.add_patch(plt.Rectangle((0, 0), Lm, Lm, facecolor="#eef0f3", lw=4, ec="tab:blue"))
    colours = ("#d62728", "#ff7f0e", "#9467bd")
    for i, ((cx, cy), on, off) in enumerate(BLOCKS):
        x0, y0 = (cx - BLOCK / 2) * 1e3, (cy - BLOCK / 2) * 1e3
        ax.add_patch(plt.Rectangle((x0, y0), BLOCK * 1e3, BLOCK * 1e3, facecolor=colours[i], alpha=0.75, ec="k", lw=0.8))
        ax.text(cx * 1e3, cy * 1e3, f"{i + 1}", ha="center", va="center", color="w", fontsize=12, fontweight="bold")
        ax.text(cx * 1e3, y0 - 0.25, f"on {on*1e3:.0f}-{off*1e3:.0f} ms", ha="center", va="top", fontsize=8.5)
    ax.text(Lm / 2, -0.7, f"edges held at the package's {T_PKG:.0f} °C", ha="center", va="top", color="tab:blue", fontsize=9.5)
    ax.annotate("", xy=(0, Lm + 0.6), xytext=(Lm, Lm + 0.6), arrowprops=dict(arrowstyle="<->", lw=0.8, color="k"))
    ax.text(Lm / 2, Lm + 0.8, f"{Lm:.0f} mm of silicon, {THICK*1e3:.1f} mm thick", ha="center", va="bottom", fontsize=9.5)
    ax.text(Lm + 0.6, 0.62 * Lm, f"three blocks,\n{BLOCK*1e3:.0f} × {BLOCK*1e3:.0f} mm,\n{P_MIN:.0f} to {P_MAX:.0f} W each,\n"
            "one at a time", fontsize=9, va="center")
    ax.set_xlim(-0.6, Lm + 4.6); ax.set_ylim(-2.0, Lm + 2.0); ax.set_aspect("equal"); ax.axis("off")
    return ax


# ----------------------------------------------------------- the truth file
TRUTH_FILE = "MP7.2A_truth.npz"


def power_key(P):
    """The key of a power in the truth file: 8 W -> '8W', 7.25 W -> '7.25W'."""
    return f"{P:g}W"


def write_truth(path=TRUTH_FILE):
    """The ground truth at the worked example (8 W) and the three test powers:
    the field at the scoring nodes at every recorded instant, the peak and its
    instant; the convergence study at 8 W, the zero-strength checks, the
    reference model's grid, and every parameter with its unit."""
    print("zero-strength checks")
    e1, e2 = check_zero_strength()
    print("convergence at 8 W")
    rows, last = convergence_study(P0)
    out = dict(x=score_nodes(), t=T_REC, powers=np.array((P0,) + P_TEST + P_OUT))
    rich = (4 * last[-1] - last[-2]) / 3
    pe = (4 * rows[-1][2] - rows[-2][2]) / 3
    k8 = power_key(P0)
    out[f"field_{k8}"], out[f"peak_{k8}"], out[f"t_peak_{k8}"] = rich, np.array(pe), np.array(rows[-1][3])
    out[f"correction_{k8}"] = np.array(np.abs(rich - last[-1]).max())
    for P in P_TEST + P_OUT:
        g = ground_truth(P)
        key = power_key(P)
        out[f"field_{key}"], out[f"peak_{key}"], out[f"t_peak_{key}"] = g["fields"], np.array(g["peak"]), np.array(g["t_peak"])
        out[f"balance_{key}"] = np.array([g["balance"][k] for k in ("heat_in", "heat_out", "stored", "residual")])
        out[f"correction_{key}"] = np.array(g["correction"])
        print(f"   {P:.2f} W: peak {g['peak']:.4f} K at {g['t_peak']*1e3:.2f} ms; Richardson correction "
              f"{g['correction']:.4f} K; balance off by {abs(g['balance']['residual'] / g['balance']['heat_in']):.1e}", flush=True)
    out[f"linear_peak_{k8}"] = np.array(linear_peak(P0))
    out["conv_cells"] = np.array([r[0] for r in rows]); out["conv_dt"] = np.array([r[1] for r in rows])
    out["conv_peak"] = np.array([r[2] for r in rows]); out["conv_t_peak"] = np.array([r[3] for r in rows])
    out["conv_field_change"] = np.array([r[4] for r in rows]); out["conv_balance"] = np.array([r[5] for r in rows])
    out["conv_cg_per_step"] = np.array([r[6] for r in rows]); out["conv_seconds"] = np.array([r[7] for r in rows])
    out["zero_strength_modes_K"], out["zero_strength_blocks_K"] = np.array(e1), np.array(e2)
    out["ref_grid"] = np.array(REF_GRID, dtype=float)
    out["blocks"] = np.array([[c[0], c[1], on, off] for c, on, off in BLOCKS])
    out["params"] = np.array([f"{k} = {v}" for k, v in dict(
        DIE="10 x 10 mm silicon, edges at 65 C (Ex07.2)", THICK="0.5 mm, ASSUMED", BLOCK="2 x 2 mm, three, on 60 ms each in turn",
        POWER="5-10 W per block, ASSUMED range; 8 W = 200 W/cm^2", K="148 (300 K / T)^1.3 W/(m K), Glassbrenner and Slack 1964 fit, ASSUMED until checked",
        RHO_C=f"{RHO_C:.4g} J/(m^3 K), from Ex07.2's diffusivity 8.8e-5 m^2/s at 300 K").items()])
    out["README"] = np.array(
        "MP7.2A ground truth at 8 W (the worked example), 5.5, 7.25 and 9.5 W (the test powers) and 4 and 11 W (outside the range). x: the scoring nodes along each edge [m] (41, every 0.25 mm, edges included; the field "
        "is indexed [instant, y node, x node]); t: the recorded instants [s] (every 2 ms). field_<P>W: rise above the "
        "package (65 C) [K] at P W per block, Richardson-extrapolated from 160 x 160 and 320 x 320 cells; peak_<P>W: the "
        "largest rise anywhere in the die over the window [K], extrapolated; t_peak_<P>W: its instant [s]; correction_<P>W: "
        "the largest Richardson correction [K]; balance_<P>W: heat in, out through the edges, stored, residual [J]. "
        "linear_peak_8W: the peak with k held at its 65 C value [K]. conv_*: the 8 W convergence study (cells across, "
        "dt s, peak K, its instant s, largest change of the field at the scoring nodes from the grid before K, energy "
        "residual relative, CG iterations per step, seconds). zero_strength_*: S = 0 against Ex07.2's exact modes and "
        "the exact modal sum with the blocks [K]. blocks: centre x, y [m], on, off [s]. ref_grid: cells, dt [s].")
    np.savez_compressed(path, **out)
    print("written", path)


if __name__ == "__main__":
    import sys
    if "--save" in sys.argv:
        write_truth()
    else:
        check_zero_strength()
        convergence_study(P0, TRUTH_GRIDS[:3])
