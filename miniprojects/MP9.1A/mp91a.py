r"""MP9.1A — One bend of the serpentine. The physics, the reference model and
the ground truth.

*Deep Learning for Engineering* — MSc, Aalborg University.
Remus Teodorescu (ret@et.aau.dk), with support from Research Assistant
Noman Khan (nomank@energy.aau.dk).

This module extends Ex09.1's ``pipe.py`` (course policy C10 rule 1: one set of
physics and constants for the exercise and its mini project) and is used by
the notebook of MP9.1A (course policy C14).

## What changes against Ex09.1, and why

Ex09.1 solves one straight leg of the serpentine, far from the bends, where the
velocity is the same at every cross-section; its steady limit is
Hagen–Poiseuille's parabola, an exact solution. Here the coolant goes through
**one U-bend** of the serpentine (12 mm radius to the tube's axis, 180°)
between a 20 mm straight leg and a 110 mm straight leg. In the bend the
centrifugal force throws the fast liquid of the middle outwards; it returns
along the walls, two counter-rotating (Dean) vortices form beside the axial
flow, and after the bend they decay while the parabola re-forms. The velocity
now changes along the tube and across it, the equations are nonlinear, and
**no exact solution exists**.

The strength of the change is the curvature ratio ``delta = a / R_c`` (tube
radius over bend radius, 0.25). At ``delta = 0`` the bend is straight and the
march must keep Ex09.1's parabola at every section
(:func:`check_zero_strength`). In a bend that never ends the march must reach
the fully developed bend of ``dean.py`` (:func:`check_long_bend`).

## The model: Dean's equations, parabolised

Lengths over the tube radius ``a``, the arc length ``s`` along the tube's axis,
the axial velocity ``w`` over the straight tube's centre speed ``V`` at the same
flow, the secondary (cross-section) velocity over ``nu / a``, with ``x`` in the
section pointing away from the bend's centre and ``Re = V a / nu`` (equal to
the exercise's Reynolds number on the mean speed and the diameter):

    Re w w_s + u . grad w = pi(s) + lap w                       axial momentum
    Re w om_s + u . grad om = lap om - kappa(s) C d(w^2)/dy     the curl of the secondary momentum
    lap psi = -om,   u = (psi_y, -psi_x)                        the secondary flow
    w = psi = d psi/dn = 0 on the wall;   flow rate fixed       no slip; 0.3 L/min

with ``C = delta Re^2`` and ``kappa = 1`` in the bend, 0 in the legs. Two
approximations, both stated in the notebook:

* **Dean's** (small ``delta``): curvature enters only through the centrifugal
  term. ``delta`` = 0.25 is at the edge of where it holds (Berger, Talbot and
  Yao, Annu. Rev. Fluid Mech. 1983, from memory).
* **Parabolised**: diffusion along ``s`` is dropped and nothing downstream acts
  upstream, so the 3-D problem becomes a march of 2-D sections. The secondary
  flow is kept divergence-free; the small cross-flow that the change of the
  axial velocity along the tube drives (of order ``Re dw/ds``) is left out.

The ground truth is the converged solution of **this model**, not of the full
Navier–Stokes equations; the network is scored against the same model.

## Two solvers

* the **reference model** (:func:`reference`): the march on an engineering
  grid of the section, timed against the network.
* the **ground truth** (:func:`ground_truth`): the same march on a grid twice
  and four times as fine, converged (:func:`convergence_study`).

Values (C10 rule 5): the tube and the coolant are Ex09.1's (typical values,
from memory); the bend radius 12 mm and the legs 20 mm and 110 mm are
**assumed** (the serpentine of Ex09.1's drawing).
"""

from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import pipe as pp

# ------------------------------------------------------------------ the data
A_TUBE = pp.R_TUBE                 #: m, tube radius, 3 mm (Ex09.1)
R_BEND = pp.BEND_R                 #: m, bend radius to the tube's axis, 12 mm   [assumed, Ex09.1's drawing]
DELTA = A_TUBE / R_BEND            #: the curvature ratio a / R_c, 0.25: the strength of the change
LEG_IN, LEG_OUT = 0.020, 0.110     #: m, the straight legs before and after the bend   [assumed]
S1 = LEG_IN / A_TUBE               #: scaled arc length where the bend starts, 6.67
S2 = S1 + np.pi * R_BEND / A_TUBE  #: where it ends, 19.23
S3 = S2 + LEG_OUT / A_TUBE         #: the outlet, 55.90
Q_NOMINAL = 0.3                    #: L/min, the exercise's flow
Q_MIN, Q_MAX = 0.15, 0.45          #: L/min, the range of flows the network takes as an input (laminar: Re 226 to 678)
Q_TRAIN = (0.15, 0.175, 0.225, 0.25, 0.275, 0.325, 0.35, 0.375, 0.425, 0.45)   #: L/min, the flows the training data come from
Q_TEST = (0.20, 0.30, 0.40)        #: L/min, the flows the answers are scored at (not trained on)
SECTIONS = {"bend middle": 0.5 * (S1 + S2), "bend exit": S2, "1 diameter after": S2 + 2.0,
            "5 diameters after": S2 + 10.0, "outlet": S3}   #: scaled arc lengths of the five scored sections
RAMP = 1.0                         #: tube radii over which the curvature rises to 1/R_c (and falls back), centred on each end of the bend   [assumed]
TARGET_W = 0.02                    #: the agreed accuracy: the axial velocity within 2 % of the centre speed, root mean square over each section
TARGET_K = 0.05                    #: and the loss coefficient within 5 %


def curvature(s, s1=S1, s2=S2, ramp=RAMP):
    """kappa(s): the curvature over 1/R_c along the tube. It rises from 0 to 1
    over ``ramp`` tube radii centred on the bend's start and falls back the same
    way at its end (a smooth cubic), so the total turn is still 180 degrees. A
    parabolised march carries nothing upstream; in the real tube the bend is felt
    about a radius ahead, and a jump in curvature would put a singular layer of
    vorticity on the wall at the bend's start."""
    s = np.asarray(s, dtype=float)
    def up(z):
        z = np.clip(z, 0.0, 1.0)
        return z * z * (3 - 2 * z)
    if ramp <= 0:
        return ((s > s1) & (s < s2)).astype(float)
    return up((s - s1) / ramp + 0.5) - up((s - s2) / ramp + 0.5)


def reynolds(q_lmin):
    """Re = V a / nu = U D / nu at the flow ``q_lmin`` (L/min): the exercise's Reynolds number."""
    return pp.RE * q_lmin / Q_NOMINAL


def centre_speed(q_lmin):
    """m/s, the straight tube's centre speed V at the flow ``q_lmin``: the velocity scale."""
    return pp.V * q_lmin / Q_NOMINAL


def dean_number(q_lmin):
    """Re sqrt(a/R_c), Re on the mean speed and the diameter."""
    return reynolds(q_lmin) * np.sqrt(DELTA)


def k_from_pi(pi, s, re):
    """The bend's loss coefficient: the pressure drop over the whole tube in
    excess of a straight tube of the same length, over rho U^2 / 2:
    K = (8 / Re) integral (pi - 4) ds."""
    return 8.0 / re * np.trapezoid(np.asarray(pi) - 4.0, s)


# ---------------------------------------------------------- the section's grid
class Section:
    """The cross-section on Dean's finite-volume grid (``dean.py``, the same
    discretisation): ``n`` rings of cells, centres at r = (i + 1/2) h, ``m``
    cells round, no node on the axis, the wall a face of the last ring. Holds
    the operators of the march as sparse matrices."""

    def __init__(self, n, m):
        self.n, self.m = n, m
        self.N = N = n * m
        h = self.h = 1.0 / n
        self.r = (np.arange(n) + 0.5) * h
        self.th = np.arange(m) * 2 * np.pi / m
        dth = self.dth = 2 * np.pi / m
        R, TH = np.meshgrid(self.r, self.th, indexing="ij")
        self.R, self.TH = R.ravel(), TH.ravel()
        self.area = self.R * h * dth                             # cell areas
        idx = np.arange(N).reshape(n, m)
        k = idx.ravel()
        jp, jm = idx[:, (np.arange(m) + 1) % m].ravel(), idx[:, (np.arange(m) - 1) % m].ravel()
        I = np.repeat(np.arange(n), m)
        # the Laplacian, zero on the wall half a cell away (dean.Grid.L0)
        ro, ri = (I + 1) * h, I * h
        cr_o = ro / (self.R * h * h); cr_i = ri / (self.R * h * h); ct = 1.0 / (self.R * dth) ** 2
        inner = I < n - 1
        rows = [k, k, k[inner], k[I > 0], k]
        cols = [jp, jm, k[inner] + m, k[I > 0] - m, k]
        diag = -2 * ct - cr_i - np.where(inner, cr_o, 2.0 * cr_o)
        vals = [ct, ct, cr_o[inner], cr_i[I > 0], diag]
        self.L = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N))
        # first derivatives at the centres (dean.Grid.d_r and d_th): central, across the axis by the
        # opposite cell, and one-sided on the last ring with the wall value zero half a cell out
        self.Dth = sp.csr_matrix((np.r_[np.full(N, 1 / (2 * dth)), np.full(N, -1 / (2 * dth))],
                                  (np.r_[k, k], np.r_[jp, jm])), shape=(N, N))
        mid = (I > 0) & (I < n - 1)
        first, last = I == 0, I == n - 1
        mirror = idx[0, (np.arange(m) + m // 2) % m]
        rows = np.r_[k[mid], k[mid], k[first], k[first], k[last]]
        cols = np.r_[k[mid] + m, k[mid] - m, k[first] + m, mirror, k[last] - m]
        vals = np.r_[np.full(mid.sum(), 1 / (2 * h)), np.full(mid.sum(), -1 / (2 * h)),
                     np.full(m, 1 / (2 * h)), np.full(m, -1 / (2 * h)), np.full(m, -1 / (1.5 * h))]
        self.Dr = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))
        self.Dy = (sp.diags(np.sin(self.TH)) @ self.Dr + sp.diags(np.cos(self.TH) / self.R) @ self.Dth).tocsr()
        # Thom's wall vorticity, -8 psi / h^2 from the last centre, entering the last ring through the wall face
        lastk = k[last]
        self.Wpsi = sp.csr_matrix((np.full(m, 2.0 / (self.r[-1] * h * h) * (-8.0 / h ** 2)), (lastk, lastk)), shape=(N, N))
        # the grid's own parabola: lap w = -4, so that a straight tube keeps it exactly (pi = 4)
        self.w0 = spla.spsolve(self.L.tocsc(), -4.0 * np.ones(N))
        self.Q = float(self.area @ self.w0)                      # the flow rate held at every section
        self.I = sp.identity(N, format="csr")


# ---------------------------------------------------------- the march
class March:
    """The parabolised march along the tube on an n x m section grid with steps
    ``ds`` (in tube radii): implicit in s (BDF2 after a first backward-Euler
    step and after each end of the bend), the secondary velocity taken from the
    section before (one linearisation per step), the flow rate held exactly by
    superposing two solves. Built once per grid; run for any flow."""

    def __init__(self, n, m, ds, delta=DELTA, s_bend=(S1, S2), s_end=S3, ramp=RAMP):
        self.sec = S = Section(n, m)
        self.delta, self.ds, self.ramp = delta, ds, ramp
        self.s1, self.s2 = s_bend
        self.s = np.linspace(0.0, s_end, int(round(s_end / ds)) + 1)

    def _adv(self, psi):
        """u . grad as a matrix, for the secondary velocity of the stream function psi."""
        S = self.sec
        ur = (S.Dth @ psi) / S.R                                 # u_r = psi_theta / r
        ut = -(S.Dr @ psi)                                       # u_theta = -psi_r
        return sp.diags(ur) @ S.Dr + sp.diags(ut / S.R) @ S.Dth

    def run(self, q_lmin=Q_NOMINAL, keep=None):
        """March at the flow ``q_lmin`` (L/min). ``keep``: scaled arc lengths at
        which to keep w and psi (the nearest station). Sets ``pi`` (the axial
        pressure gradient at every station), ``w``, ``psi`` at the outlet, and
        ``kept`` (dict s -> (w, psi)); returns self."""
        S, s, N = self.sec, self.s, self.sec.N
        re = reynolds(q_lmin)
        C = self.delta * re ** 2
        w = S.w0.copy(); om = np.zeros(N); psi = np.zeros(N)
        prev = None
        pis = [4.0]
        want = [] if keep is None else sorted(keep)
        kept, last3 = {}, [(s[0], w, psi)]
        for v in want:
            if v <= s[0]:
                kept[v] = (w.copy(), psi.copy())
        for k in range(1, len(s)):
            ds = s[k] - s[k - 1]
            kap = curvature(s[k], self.s1, self.s2, self.ramp)     # implicit: the curvature at the new section
            if prev is not None:
                w1, om1, psi1 = prev                             # BDF2: second order in s
                c0, rw, ro = 1.5 / ds, (4 * w - w1) / (2 * ds), (4 * om - om1) / (2 * ds)
                wg, pg = 2 * w - w1, 2 * psi - psi1
            else:                                                # backward Euler: the first step
                c0, rw, ro = 1.0 / ds, w / ds, om / ds
                wg, pg = w, psi
            c = re * np.maximum(wg, 0.0)                         # the marching coefficient Re w, never divided by
            A = (sp.diags(c * c0) + self._adv(pg) - S.L).tocsc()
            lu = spla.splu(A)
            x1 = lu.solve(c * rw); x2 = lu.solve(np.ones(N))     # the axial momentum, for pi = 0 and for pi = 1
            pi = (S.Q - S.area @ x1) / (S.area @ x2)             # the pressure gradient that keeps the flow rate
            wn = x1 + pi * x2
            M = sp.bmat([[A, -S.Wpsi], [S.I, S.L]], format="csc")  # vorticity and stream function together
            rhs = np.r_[c * ro - kap * C * (S.Dy @ (wn ** 2)), np.zeros(N)]
            sol = spla.splu(M).solve(rhs)
            prev = (w, om, psi)
            w, om, psi = wn, sol[:N], sol[N:]
            pis.append(pi)
            last3 = (last3 + [(s[k], w, psi)])[-3:]
            for v in want:                                       # a section between stations: quadratic through the last three
                if v not in kept and v <= s[k] + 1e-12 and len(last3) == 3:
                    (sa, wa, pa), (sb, wb, pb), (sc, wc, pc) = last3
                    la = (v - sb) * (v - sc) / ((sa - sb) * (sa - sc)); lb = (v - sa) * (v - sc) / ((sb - sa) * (sb - sc))
                    lc = (v - sa) * (v - sb) / ((sc - sa) * (sc - sb))
                    kept[v] = (la * wa + lb * wb + lc * wc, la * pa + lb * pb + lc * pc)
        self.pi, self.w, self.om, self.psi, self.kept, self.re = np.array(pis), w, om, psi, kept, re
        self.flow_error = abs(S.area @ w - S.Q) / S.Q
        return self

    def K(self):
        return k_from_pi(self.pi, self.s, self.re)


# ---------------------------------------------------------- sampling a section
def sample(sec, f, x, y):
    """A cell-centred field ``f`` (n*m) at points (x, y) of the section, by
    linear interpolation in r and theta; the axis value is the mean of the first
    ring, the wall value zero (w and psi vanish on the wall). Second order."""
    from scipy.interpolate import RegularGridInterpolator
    F = f.reshape(sec.n, sec.m)
    rr = np.r_[0.0, sec.r, 1.0]
    tt = np.r_[sec.th, 2 * np.pi]
    G = np.zeros((sec.n + 2, sec.m + 1))
    G[0] = F[0].mean(); G[1:-1, :-1] = F; G[1:-1, -1] = F[:, 0]
    r = np.clip(np.hypot(x, y), 0, 1); t = np.mod(np.arctan2(y, x), 2 * np.pi)
    return RegularGridInterpolator((rr, tt), G)(np.c_[r, t])


def score_points(n=25):
    """The points of a section where the velocity is scored: an n x n square
    grid in x and y, kept inside r <= 1 (the wall included, where w = 0)."""
    g = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(g, g, indexing="ij")
    keep = np.hypot(X, Y) <= 1.0 + 1e-12
    return X[keep], Y[keep]


# ------------------------------------------------------- the reference model
REF_GRID = (16, 32, 0.2)   #: n rings, m cells round, step ds (tube radii): the engineering grid


def reference(q_lmin, grid=REF_GRID, keep=tuple(SECTIONS.values())):
    """The reference model at one flow: the march on the engineering grid.
    Returns the sections' axial velocity and stream function on the score
    points, the loss coefficient and the wall-clock time (s)."""
    t0 = time.perf_counter()
    m = March(*grid).run(q_lmin, keep=keep)
    x, y = score_points()
    out = {name: (sample(m.sec, m.kept[sv][0], x, y), sample(m.sec, m.kept[sv][1], x, y)) for name, sv in SECTIONS.items()}
    sec = time.perf_counter() - t0
    return dict(sections=out, K=m.K(), pi=m.pi, s=m.s, seconds=sec, march=m)


# -------------------------------------------------------- the ground truth
TRUTH_GRIDS = ((16, 32, 0.1), (32, 64, 0.1), (64, 128, 0.1))   #: each twice as fine across the section
TRUTH_GRID = TRUTH_GRIDS[-1]
STEP_GRIDS = ((32, 64, 0.4), (32, 64, 0.2), (32, 64, 0.1), (32, 64, 0.05))   #: the step along the tube halved three times


def convergence_study(q_lmin=Q_NOMINAL, grids=TRUTH_GRIDS, verbose=True, label="grid"):
    """The axial velocity at the five sections (on the score points) and the
    loss coefficient on grids each twice as fine. Returns rows (grid, values
    per section, K, flow error, seconds)."""
    x, y = score_points()
    rows = []
    for g in grids:
        t0 = time.perf_counter()
        m = March(*g).run(q_lmin, keep=tuple(SECTIONS.values()))
        sec = time.perf_counter() - t0
        vals = {name: sample(m.sec, m.kept[sv][0], x, y) for name, sv in SECTIONS.items()}
        rows.append(dict(grid=g, w=vals, K=m.K(), flow_error=m.flow_error, seconds=sec,
                         wmax_exit=float(vals["bend exit"].max())))
        if verbose:
            print(f"   {label} {g[0]:3d} x {g[1]:3d}, ds {g[2]:.2f}: K {rows[-1]['K']:.4f}, fastest liquid at the bend exit "
                  f"{rows[-1]['wmax_exit']:.4f} V, flow rate held to {m.flow_error:.0e}   ({sec:.0f} s)", flush=True)
    for i in range(1, len(rows)):
        rows[i]["change_max"] = max(np.abs(rows[i]["w"][k] - rows[i - 1]["w"][k]).max() for k in SECTIONS)
        rows[i]["change_rms"] = max(rms(rows[i]["w"][k] - rows[i - 1]["w"][k]) for k in SECTIONS)
        if verbose:
            print(f"   change of the axial velocity at the five sections, {label} {i} to {i + 1}: largest {rows[i]['change_max']:.2e} V, "
                  f"root mean square over a section {rows[i]['change_rms']:.2e} V (worst section)")
    if len(rows) >= 3:
        k0, k1, k2 = (r["K"] for r in rows[-3:])
        same_sign = (k0 - k1) * (k1 - k2) > 0
        p = np.log(abs(k0 - k1) / abs(k1 - k2)) / np.log(2) if same_sign else np.nan   # the observed order
        rows[-1]["K_order"] = p
        if same_sign and p > 0.5:
            rows[-1]["K_extrapolated"] = k2 + (k2 - k1) / (2 ** p - 1)       # Richardson with that order
            if verbose:
                print(f"   K: observed order {p:.2f}, Richardson extrapolation {rows[-1]['K_extrapolated']:.4f}")
        else:                                                                # the changes do not fall: no extrapolation
            rows[-1]["K_extrapolated"] = np.nan
            if verbose:
                print(f"   K: the changes do not fall ({abs(k0 - k1):.4f}, {abs(k1 - k2):.4f}): no order, no extrapolation; "
                      f"its uncertainty is taken as the last change, {abs(k1 - k2) / k2 * 100:.1f} %")
    return rows


def rms(e):
    """The root mean square over the score points of a section: as they are an
    even grid in x and y, it is the root mean square over the section's area."""
    return float(np.sqrt(np.mean(np.asarray(e) ** 2)))


def ground_truth(q_lmin, grid=TRUTH_GRID):
    """The truth at one flow: the march on the finest grid, the five sections
    on the score points, w along the axis and pi along the tube."""
    x, y = score_points()
    m = March(*grid).run(q_lmin, keep=tuple(SECTIONS.values()))
    return dict(sections={name: (sample(m.sec, m.kept[sv][0], x, y), sample(m.sec, m.kept[sv][1], x, y)) for name, sv in SECTIONS.items()},
                K=m.K(), pi=m.pi, s=m.s, march=m)


# ------------------------------------------------------------- the checks
def check_zero_strength(grid=(16, 32, 0.4), q_lmin=Q_NOMINAL, verbose=True):
    """delta = 0: the bend is straight, and the march must keep Ex09.1's
    steady limit, the parabola w = 1 - r^2, with pi = 4, at every section."""
    m = March(*grid, delta=0.0).run(q_lmin)
    S = m.sec
    e_grid = np.abs(m.w - S.w0).max()
    e_exact = np.abs(m.w - (1 - S.R ** 2)).max()
    if verbose:
        print(f"   delta = 0: the outlet against the grid's parabola {e_grid:.1e} V, against Hagen-Poiseuille's 1 - r^2 "
              f"{e_exact:.1e} V ({S.n} rings: the grid's own error), pressure gradient off 4 by {np.abs(m.pi - 4).max():.1e}, "
              f"secondary flow {np.abs(m.psi).max():.1e}")
    return e_grid, e_exact


def check_long_bend(n=16, m=32, ds=0.5, length=300.0, q_lmin=Q_NOMINAL, verbose=True):
    """A bend that never ends: after 300 radii the march must reach the fully
    developed bend that dean.py solves on the same grid."""
    import dean
    g, w, psi, om, ratio = dean.at_flow(reynolds(q_lmin), DELTA, n, m)
    mm = March(n, m, ds, s_bend=(-1.0, 2 * length), s_end=length).run(q_lmin)
    dw = np.abs(mm.w - w.ravel() / ratio).max()
    dpsi = np.abs(mm.psi - psi.ravel()).max() / np.abs(psi).max()
    if verbose:
        print(f"   long bend ({length:.0f} radii): friction over the straight tube's {mm.pi[-1] / 4:.4f} (dean.py {1 / ratio:.4f}); "
              f"axial velocity off by {dw:.1e} V; secondary flow off by {dpsi:.1e} of its largest value")
    return mm.pi[-1] / 4, 1 / ratio, dw, dpsi


def check_symmetry(m):
    """The bend is symmetric about its plane (y -> -y): w even, psi odd."""
    S = m.sec
    j = (-np.arange(S.m)) % S.m                                  # theta -> -theta
    W = m.w.reshape(S.n, S.m); P = m.psi.reshape(S.n, S.m)
    return np.abs(W - W[:, j]).max(), np.abs(P + P[:, j]).max()


# ------------------------------------------------------- the training points
def sample_tube(n, seed=88, q_range=(Q_MIN, Q_MAX)):
    """n points (s, x, y, q) in the tube from the bend's start to the outlet and
    over the range of flows: s in tube radii, (x, y) uniform over the section,
    q in L/min. Half the arc lengths in the bend and the first 10 radii after it,
    where the flow changes fastest. The inlet leg needs no points: there the
    parabola is kept exactly (nothing downstream acts upstream)."""
    rng = np.random.default_rng(seed)
    k = n // 2
    s = np.r_[rng.uniform(S1, S2 + 10.0, k), rng.uniform(S1, S3, n - k)]
    r = np.sqrt(rng.random(n)); t = 2 * np.pi * rng.random(n)
    return np.c_[s, r * np.cos(t), r * np.sin(t), rng.uniform(*q_range, n)]


DATA_GRID = (32, 64, 0.2)   #: the grid the training data is computed on: four times the reference model's cells


def training_data(q_list=Q_TRAIN, grid=DATA_GRID, every=0.5, n_pts=25, verbose=True):
    """Data the network may train on, computed offline with the march at the
    training flows (none of the test flows): the axial velocity w and the
    stream function psi on the score points of a section every ``every`` tube
    radii from the bend's start to the outlet, and the axial pressure gradient
    pi at every station. Returns (rows (s, x, y, q, w, psi), pi rows (s, q, pi),
    seconds)."""
    x, y = score_points(n_pts)
    stations = np.arange(S1, S3 + 1e-9, every)
    rows, pis = [], []
    t0 = time.perf_counter()
    for q in q_list:
        m = March(*grid).run(q, keep=tuple(stations))
        for sv in stations:
            w, psi = m.kept[sv]
            rows.append(np.c_[np.full(x.size, sv), x, y, np.full(x.size, q), sample(m.sec, w, x, y), sample(m.sec, psi, x, y)])
        pis.append(np.c_[m.s, np.full(m.s.size, q), m.pi])
    sec = time.perf_counter() - t0
    if verbose:
        print(f"   training data: {len(q_list)} flows x {stations.size} sections x {x.size} points, on {grid[0]} x {grid[1]} cells, in {sec:.0f} s")
    return np.concatenate(rows), np.concatenate(pis), sec


# ----------------------------------------------------------- the truth file
TRUTH_FILE = "MP9.1A_truth.npz"


def write_truth(path=TRUTH_FILE):
    """The ground truth at the three test flows on the 64 x 128 grid (the five
    sections on the score points, pi along the tube, K), the convergence study
    at 0.3 L/min across the section and along the tube, the checks and every
    parameter with its unit."""
    out = dict(flows=np.array(Q_TEST), grid=np.array(TRUTH_GRID, dtype=float),
               section_names=np.array(list(SECTIONS)), section_s=np.array(list(SECTIONS.values())))
    x, y = score_points()
    out["x"], out["y"] = x, y
    print("convergence across the section at 0.3 L/min")
    rows = convergence_study(Q_NOMINAL, TRUTH_GRIDS)
    print("convergence along the tube at 0.3 L/min")
    srows = convergence_study(Q_NOMINAL, STEP_GRIDS, label="step")
    for tag, rr in (("conv", rows), ("step", srows)):
        out[f"{tag}_grids"] = np.array([r["grid"] for r in rr], dtype=float)
        out[f"{tag}_K"] = np.array([r["K"] for r in rr])
        out[f"{tag}_wmax_exit"] = np.array([r["wmax_exit"] for r in rr])
        out[f"{tag}_change"] = np.array([r["change_max"] for r in rr[1:]])
        out[f"{tag}_change_rms"] = np.array([r["change_rms"] for r in rr[1:]])
        out[f"{tag}_K_extrapolated"] = np.array(rr[-1]["K_extrapolated"]); out[f"{tag}_K_order"] = np.array(rr[-1]["K_order"])
        out[f"{tag}_flow_error"] = np.array([r["flow_error"] for r in rr])
        out[f"{tag}_seconds"] = np.array([r["seconds"] for r in rr])
    print("the checks")
    for name in SECTIONS:
        for i, r in enumerate(rows):
            out[f"conv{i}_w_{name.replace(' ', '_')}"] = r["w"][name]          # the 0.3 L/min sections on each grid
    out["zero_strength"] = np.array(check_zero_strength())
    out["long_bend"] = np.array(check_long_bend())
    for q in Q_TEST:
        t0 = time.perf_counter()
        g = ground_truth(q)
        key = f"{q:.2f}"
        for name in SECTIONS:
            tag = name.replace(" ", "_")
            out[f"w_{key}_{tag}"], out[f"psi_{key}_{tag}"] = g["sections"][name]
        out[f"K_{key}"] = np.array(g["K"]); out[f"pi_{key}"] = g["pi"]; out["s"] = g["s"]
        mm = g["march"]
        out[f"axis_{key}"] = np.array([mm.kept[v][0][: mm.sec.m].mean() for v in SECTIONS.values()])
        sym = check_symmetry(mm)
        out[f"symmetry_{key}"] = np.array(sym)
        print(f"   truth at {q:.2f} L/min: K {g['K']:.4f}, fastest at the bend exit {g['sections']['bend exit'][0].max():.4f} V, "
              f"flow rate held to {mm.flow_error:.0e}, symmetry {sym[0]:.0e} / {sym[1]:.0e}  ({time.perf_counter() - t0:.0f} s)", flush=True)
    out["params"] = np.array([f"{k} = {v}" for k, v in dict(
        A="3 mm tube radius (Ex09.1, typical)", R_BEND="12 mm to the tube's axis, assumed", DELTA=f"{DELTA}",
        LEGS="20 mm before, 110 mm after, assumed", RHO=f"{pp.RHO} kg/m3", MU=f"{pp.MU} Pa s (50/50 water-glycol, 40 degC, typical)",
        FLOWS="0.15-0.45 L/min, Re 226-678 (on the mean speed and the diameter)").items()])
    out["README"] = np.array(
        "MP9.1A ground truth: the parabolised Dean march on 64 x 128 cells, steps of 0.1 tube radii. flows: the test flows "
        "(L/min); x, y: the score points of a section (over the tube radius, x away from the bend's centre); "
        "w_<q>_<section>, psi_<q>_<section>: axial velocity over the straight tube's centre speed V(q) and secondary stream "
        "function over nu, on x, y, at the sections section_names at scaled arc lengths section_s (tube radii from the "
        "inlet; bend from 6.67 to 19.23); K_<q>: the loss coefficient; pi_<q>: the axial pressure gradient (4 = straight) on "
        "s; axis_<q>: w on the axis at the five sections; conv_*, step_*: the convergence study at 0.3 L/min across the "
        "section and along the tube (grids n, m, ds; change, change_rms: largest and worst-section root-mean-square change of w between grids, in V; K_extrapolated, K_order: Richardson); "
        "zero_strength: delta = 0 against the grid's and the exact parabola; long_bend: friction ratio of the march and of "
        "dean.py, velocity and stream-function differences; symmetry_<q>: w even, psi odd; params: the data.")
    np.savez_compressed(path, **out)
    print("written", path)


if __name__ == "__main__":
    import sys
    if "--checks" in sys.argv:
        check_zero_strength(); check_long_bend()
    if "--conv" in sys.argv:
        convergence_study(); convergence_study(grids=STEP_GRIDS, label="step")
    if "--save" in sys.argv:
        write_truth()
