r"""MP7.1A - The slot with its insulation: the 32 wires of Ex07.1 resolved.
The physics, the reference model and the ground truth.

*Deep Learning for Engineering* - MSc, Aalborg University.
Remus Teodorescu (ret@et.aau.dk), with support from Research Assistant
Noman Khan (nomank@energy.aau.dk).

This module extends Ex07.1's ``problem.py`` (course policy C10 rule 1: one set
of physics and constants for the exercise and its mini project) and is used by
the notebook of MP7.1A (course policy C14).

## What changes against Ex07.1, and why

Ex07.1 averages the winding into one material, k_eff = 0.70 W/(m K), with the
heat of the 32 wires spread evenly over the slot. Its rectangle with constant
coefficients then has an exact double sine series. Here every wire is drawn
as it is: a copper core of radius 1.000 mm (k = 400 W/(m K)), its grade 2
enamel coat to 1.0375 mm (IEC 60317: a 2.000 mm conductor gains at least
0.075 mm on its diameter), and impregnating resin in the rest of the slot
(enamel and resin 0.20 W/(m K), typical values, ASSUMED). The heat is made in
the copper only. The conductivity jumps by a factor of 2000 at every wire's
edge, the field is no longer a sum of the rectangle's modes, and **no exact
solution exists**.

``S`` is the strength of the change. Every cell's conductivity and heat are
mixed between the averaged slot (S = 0) and the resolved slot (S = 1); at
S = 0 the solver must reproduce Ex07.1's exact series
(:func:`check_zero_strength`).

## The problem, in kelvin above the wall

    div(k grad theta) + q = 0                in the slot
    q = c0(I) + c1(I) theta                  in the copper, 0 elsewhere
    theta = 0                                on the four walls (iron at 90 C)

with c0 = rho(T_WALL) J^2 and c1 = rho_20 alpha J^2, J = I / A_wire, as in
Ex07.1. The slot is symmetric about both its axes, so the quarter x > 0, y > 0
is solved, with zero heat flow across the two symmetry lines.

## Two solvers

* the **reference model** (:func:`reference`): the same finite volumes on an
  engineering grid, 0.083 mm cells (60 x 120 on the quarter), the coarsest
  that meets the agreed accuracy. It is what the network is timed against.
* the **ground truth** (:func:`ground_truth`): 0.0125 mm cells, checked on
  grids each twice as fine (:func:`convergence_study`).
"""

from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import problem as pb

# ------------------------------------------------------------------ the data
K_CU = 400.0            #: W/(m K), copper (Incropera & DeWitt, table A.1: 401 at 300 K)
K_ENAMEL = 0.20         #: W/(m K), polyamide-imide / polyester-imide enamel           [ASSUMED]
K_RESIN = 0.20          #: W/(m K), unfilled impregnating resin                          [ASSUMED]
R_CU = pb.WIRE_D / 2    #: m, the copper core, 1.000 mm
R_ENAMEL = R_CU + 0.075e-3 / 2   #: m, 1.0375 mm: grade 2 enamel, IEC 60317-0-1 minimum increase 0.075 mm
PITCH_X = 2 * pb.A_HALF / pb.N_COLS   #: m, 2.5 mm between wire centres across the slot
PITCH_Y = 2 * pb.B_HALF / pb.N_ROWS   #: m, 2.5 mm down it
CENTRES = np.array([(-pb.A_HALF + (i + 0.5) * PITCH_X, -pb.B_HALF + (j + 0.5) * PITCH_Y)
                    for i in range(pb.N_COLS) for j in range(pb.N_ROWS)])   #: m, the 32 wire centres
QUARTER = CENTRES[(CENTRES[:, 0] > 0) & (CENTRES[:, 1] > 0)]   #: m, the 8 wires of the quarter x > 0, y > 0
I_MIN, I_MAX = pb.I_RANGE          #: A, 10 to 45 A per wire (RMS), Ex07.1's range
I_TEST = (17.5, pb.I_RATED, 41.5)  #: A, the currents the answers are scored at (not trained on)
I_OUTSIDE = (5.0, 50.0)            #: A, outside the range, for the extrapolation question
I_TRUTH = (5.0, 10.0, 17.5, 25.0, pb.I_RATED, 38.0, 41.5, 45.0, 50.0)   #: A, the currents of the truth file


def heat_coefficients(I):
    """(c0, c1): the heat per unit volume of COPPER at the wall temperature
    [W/m^3], and its growth per kelvin of rise [W/(m^3 K)], at current I [A]."""
    J2 = (I / pb.A_WIRE) ** 2
    return pb.resistivity(pb.T_WALL) * J2, pb.RHO_CU * pb.ALPHA_CU * J2


def copper_fraction(xc, yc, h, sub=8):
    """The share of each square cell (centres xc, yc, side h) that is copper,
    counted on sub x sub points in every cell the copper's edge crosses."""
    offs = (np.arange(sub) + 0.5) / sub - 0.5
    X, Y = np.meshgrid(xc, yc)
    f = np.zeros_like(X)
    for cx, cy in CENTRES:
        r2 = (X - cx) ** 2 + (Y - cy) ** 2
        near = r2 < (R_CU + h) ** 2
        if not near.any():
            continue
        inside = r2 < (R_CU - h) ** 2
        f[inside] = 1.0
        band = near & ~inside
        for ox in offs:
            for oy in offs:
                f[band] += (((X + ox * h - cx) ** 2 + (Y + oy * h - cy) ** 2) < R_CU ** 2)[band] / sub ** 2
    return f


def material_map(xc, yc):
    """The material at each cell centre: 0 copper, 1 enamel, 2 resin (for drawing)."""
    X, Y = np.meshgrid(xc, yc)
    m = np.full(X.shape, 2)
    for cx, cy in CENTRES:
        r2 = (X - cx) ** 2 + (Y - cy) ** 2
        m[r2 < R_ENAMEL ** 2] = 1
        m[r2 < R_CU ** 2] = 0
    return m


def cell_conductivity(f, S=1.0):
    """The conductivity of a cell that is a share f copper and the rest
    insulation [W/(m K)]: the harmonic mean, the conductivity of the two in
    series, which is how heat crosses a thin copper edge into resin. Mixed
    with Ex07.1's k_eff by the strength S (S = 0: the averaged slot)."""
    k_res = 1.0 / (f / K_CU + (1 - f) / K_RESIN)
    return (1 - S) * pb.K_EFF + S * k_res


class Quarter:
    """Cell-centred finite volumes on the quarter slot, square cells of side
    h. The matrix is assembled once per grid; a current is one sparse solve.

    Every cell exchanges heat with its four neighbours through a conductance
    equal to the harmonic mean of their conductivities (the two half cells in
    series), so the heat flow is continuous across every material edge. A cell
    on a wall exchanges with the wall through a half cell, conductance 2k; the
    two symmetry lines x = 0 and y = 0 carry no heat."""

    def __init__(self, h, S=1.0):
        self.h, self.S = h, S
        self.nx, self.ny = round(pb.A_HALF / h), round(pb.B_HALF / h)
        self.xe = np.linspace(0, pb.A_HALF, self.nx + 1)
        self.ye = np.linspace(0, pb.B_HALF, self.ny + 1)
        self.xc = (self.xe[:-1] + self.xe[1:]) / 2
        self.yc = (self.ye[:-1] + self.ye[1:]) / 2
        f = copper_fraction(self.xc, self.yc, h)
        self.frac = (1 - S) * pb.FILL + S * f                  # the share of each cell that makes heat
        self.k = cell_conductivity(f, S)
        k, ny, nx = self.k, self.ny, self.nx
        idx = np.arange(nx * ny).reshape(ny, nx)
        hm = lambda a, b: 2 * a * b / (a + b)
        gx, gy = hm(k[:, :-1], k[:, 1:]), hm(k[:-1, :], k[1:, :])
        diag = np.zeros((ny, nx))
        diag[:, :-1] += gx; diag[:, 1:] += gx; diag[:-1, :] += gy; diag[1:, :] += gy
        self.wall = np.zeros((ny, nx))
        self.wall[:, -1] += 2 * k[:, -1]; self.wall[-1, :] += 2 * k[-1, :]   # x = A and y = B: the iron
        diag += self.wall
        rows = np.concatenate([idx[:, :-1].ravel(), idx[:, 1:].ravel(), idx[:-1, :].ravel(), idx[1:, :].ravel()])
        cols = np.concatenate([idx[:, 1:].ravel(), idx[:, :-1].ravel(), idx[1:, :].ravel(), idx[:-1, :].ravel()])
        vals = -np.concatenate([gx.ravel(), gx.ravel(), gy.ravel(), gy.ravel()])
        self.K = (sp.coo_matrix((vals, (rows, cols)), shape=(nx * ny, nx * ny)) + sp.diags(diag.ravel())).tocsc()
        self.n = nx * ny

    def solve(self, I):
        """The rise theta [K] per cell at current I [A], shape (ny, nx); also
        sets ``balance``: heat made and heat leaving through the walls [W/m]."""
        c0, c1 = heat_coefficients(I)
        a = self.h ** 2
        A = self.K - sp.diags((self.frac * c1 * a).ravel())
        th = spla.spsolve(A.tocsc(), (self.frac * c0 * a).ravel()).reshape(self.ny, self.nx)
        made = float((self.frac * (c0 + c1 * th)).sum() * a)
        out = float((self.wall * th).sum())
        self.balance = dict(made=4 * made, out=4 * out)        # W per metre of slot, the whole slot
        return th


def hot_spot(th):
    return float(th.max())


def theta_scale(I):
    """A scale of the rise at current I [K]: the averaged heat over the
    half-width squared over twice k_eff, FILL c0 A^2 / (2 k_eff); 19.7 K at
    rated current against an actual hot spot of 21.14 K."""
    return pb.FILL * heat_coefficients(I)[0] * pb.A_HALF ** 2 / (2 * pb.K_EFF)


# ---------------------------------------------------------- reading a field
def at(xc, yc, th, x, y):
    """The field ``th`` on cell centres xc, yc interpolated bilinearly to
    points (x, y) in the quarter [m]. The grid is padded first with the
    walls (theta = 0 at x = A and y = B) and mirrored at the symmetry lines,
    so every point of the quarter lies between known values."""
    from scipy.interpolate import RegularGridInterpolator
    xp = np.r_[-xc[0], xc, pb.A_HALF]
    yp = np.r_[-yc[0], yc, pb.B_HALF]
    t = np.pad(th, ((1, 1), (1, 1)))
    t[0, 1:-1], t[1:-1, 0] = th[0], th[:, 0]                  # mirrored across y = 0 and x = 0
    t[0, 0] = th[0, 0]
    t[-1, :], t[:, -1] = 0.0, 0.0                              # the walls
    f = RegularGridInterpolator((yp, xp), t)
    return f(np.c_[np.ravel(y), np.ravel(x)]).reshape(np.shape(x))


# ------------------------------------------------------- the reference model
REF_H = pb.A_HALF / 60  #: m, the engineering grid: 60 x 120 cells (0.083 mm) on the quarter, the coarsest that meets the agreed accuracy
HOT_TOL, FIELD_TOL = 0.1, 0.5   #: K, the agreed accuracy: hot spot, and the field at every scoring point


def reference(I, grid=None, h=REF_H):
    """The reference model: finite volumes on the engineering grid. ``grid``
    is a :class:`Quarter` built once (the geometry does not change with the
    current); a new current is one sparse solve, which is what is timed.
    Returns a dict with the field, its cell centres, the hot spot and the
    wall-clock time of the solve (s)."""
    q = Quarter(h) if grid is None else grid
    t0 = time.perf_counter()
    th = q.solve(I)
    wall = time.perf_counter() - t0
    return dict(theta=th, x=q.xc, y=q.yc, hot=hot_spot(th), seconds=wall)


def reference_at(I, grid, xs, ys):
    """The reference model's answer to a new current at the points xs, ys [m]:
    one solve on the engineering grid ``grid`` and its field read there [K]."""
    th = grid.solve(I)
    return at(grid.xc, grid.yc, th, xs, ys), hot_spot(th)


# -------------------------------------------------------- the ground truth
TRUTH_H = (0.1e-3, 0.05e-3, 0.025e-3, 0.0125e-3, 0.00625e-3)   #: m, five grids each twice as fine
TRUTH_GRID = 0.0125e-3   #: m, the ground truth's cells: 400 x 800 on the quarter


def convergence_study(I=pb.I_RATED, hs=TRUTH_H, verbose=True):
    """The hot spot and the energy balance on grids each twice as fine.
    Returns rows of (h [m], cells, hot spot [K], balance residual [-], seconds)."""
    rows, fields = [], []
    X, Y = np.meshgrid(*score_axes())
    m = score_mask(X, Y); X, Y = X[m], Y[m]
    for h in hs:
        t0 = time.perf_counter()
        q = Quarter(h)
        th = q.solve(I)
        b = q.balance
        rows.append((h, q.n, hot_spot(th), (b["made"] - b["out"]) / b["made"], time.perf_counter() - t0))
        fields.append(at(q.xc, q.yc, th, X, Y))
        if verbose:
            print(f"   cells {h*1e3:.4f} mm: {q.n:7d} cells, hot spot {rows[-1][2]:.4f} K, "
                  f"energy balance off by {abs(rows[-1][3]):.1e}  ({rows[-1][4]:.1f} s)")
    change = [0.0] + [float(np.abs(fields[i] - fields[i - 1]).max()) for i in range(1, len(fields))]
    rows = [r + (c,) for r, c in zip(rows, change)]
    if verbose:
        for i in range(1, len(rows)):
            print(f"   grid {i} to {i + 1}: hot spot changes {rows[i][2] - rows[i-1][2]:+.4f} K, "
                  f"the field at the scoring points by {change[i]:.4f} K at most")
        print(f"   (the field is compared at {X.size} points farther than {EDGE_GAP*1e3:.2f} mm from every copper edge)")
    return rows


def ground_truth(I, h=TRUTH_GRID):
    q = Quarter(h)
    th = q.solve(I)
    return dict(theta=th, x=q.xc, y=q.yc, hot=hot_spot(th), balance=q.balance, frac=q.frac, k=q.k)


def check_zero_strength(h=0.025e-3, I=pb.I_RATED, verbose=True):
    """At S = 0 every cell has Ex07.1's k_eff and its share of the heat: the
    averaged slot, whose exact series is pb.exact_solution. The finite volumes
    must reproduce it. Returns the largest difference over the quarter [K]."""
    q = Quarter(h, S=0.0)
    th = q.solve(I)
    ex = pb.exact_solution(I, q.xc, q.yc)
    err = float(np.abs(th - ex).max())
    if verbose:
        print(f"   S = 0, cells {h*1e3:.4f} mm: hot spot {th.max():.4f} K against Ex07.1's exact {ex.max():.4f} K; "
              f"largest difference {err:.1e} K")
    return err


def copper_spread(g):
    """The largest spread of temperature inside one copper core [K]: how
    nearly isothermal each wire is."""
    X, Y = np.meshgrid(g["x"], g["y"])
    out = 0.0
    for cx, cy in QUARTER:
        m = (X - cx) ** 2 + (Y - cy) ** 2 < (R_CU - 0.05e-3) ** 2
        out = max(out, float(g["theta"][m].max() - g["theta"][m].min()))
    return out


def wire_temperatures(g):
    """The mean rise of each of the quarter's 8 copper cores [K], in the
    order of QUARTER (column by column, from the slot's centre outward)."""
    X, Y = np.meshgrid(g["x"], g["y"])
    return np.array([g["theta"][(X - cx) ** 2 + (Y - cy) ** 2 < (R_CU - 0.05e-3) ** 2].mean() for cx, cy in QUARTER])


# ------------------------------------------------------- the scoring points
def score_axes():
    """x and y [m] of the points the field is scored at: 100 x 200 on the
    quarter, 0.05 mm apart, from 0.025 mm off the symmetry lines to 0.025 mm
    off the walls. They are cell centres of the 0.05 mm grid and lie between
    the cell centres of the 0.0125 mm ground truth."""
    return (np.arange(100) + 0.5) * 0.05e-3, (np.arange(200) + 0.5) * 0.05e-3


EDGE_GAP = 0.05e-3      #: m, scoring points closer than this to a copper edge are left out


def score_mask(X, Y):
    """True at the points scored: farther than EDGE_GAP from every copper
    edge. At an edge the field has a kink (the slope jumps 2000 times), and a
    value interpolated between two grid cells across it is an error of the
    interpolation, not of the solution."""
    r = np.min([np.hypot(X - cx, Y - cy) for cx, cy in CENTRES], 0)
    return np.abs(r - R_CU) > EDGE_GAP


def score_points(G):
    """The scored points as flat arrays x, y [m] and the ground truth's field
    there (current, points) [K], interpolated from its 0.0125 mm cells."""
    X, Y = np.meshgrid(*score_axes())
    m = score_mask(X, Y)
    return X[m], Y[m], np.array([at(G["x"], G["y"], t, X[m], Y[m]) for t in G["theta"]])


# ------------------------------------------------------- the training points
def in_resin(x, y, margin=0.0):
    """True where (x, y) [m] is outside every copper core (by ``margin`` m)."""
    x, y = np.asarray(x), np.asarray(y)
    ok = np.ones(x.shape, bool)
    for cx, cy in CENTRES:
        ok &= (x - cx) ** 2 + (y - cy) ** 2 >= (R_CU + margin) ** 2
    return ok


def sample_resin(n, seed=71, near=0.5, band=0.3e-3):
    """n points (x, y) [m] in the insulation of the quarter, and n currents
    I [A] drawn uniformly over 10-45 A. A share ``near`` of the points lie
    within ``band`` of a copper core, where the field bends most."""
    rng = np.random.default_rng(seed)
    k = int(n * near)
    w = rng.integers(0, len(QUARTER), k)
    r = R_CU + band * rng.random(k)
    a = 2 * np.pi * rng.random(k)
    xn, yn = QUARTER[w, 0] + r * np.cos(a), QUARTER[w, 1] + r * np.sin(a)
    x, y = rng.random(4 * n) * pb.A_HALF, rng.random(4 * n) * pb.B_HALF   # uniform, then the copper left out
    ok = in_resin(x, y)
    x = np.r_[xn, x[ok][: n - k]]
    y = np.r_[yn, y[ok][: n - k]]
    return x, y, rng.uniform(I_MIN, I_MAX, n)


def sample_copper(n, seed=73):
    """n points (x, y) [m] inside the copper cores of the quarter, uniform by
    area, and n currents I [A]."""
    rng = np.random.default_rng(seed)
    w = rng.integers(0, len(QUARTER), n)
    r = R_CU * np.sqrt(rng.random(n)); a = 2 * np.pi * rng.random(n)
    return QUARTER[w, 0] + r * np.cos(a), QUARTER[w, 1] + r * np.sin(a), rng.uniform(I_MIN, I_MAX, n)


def sample_edges(n, seed=74):
    """n points on the copper's edges (r = 1.000 mm), with the outward unit
    normal of each (pointing from the copper into the insulation), and n
    currents I [A]. Returns x, y [m], nx, ny [-], I [A]."""
    rng = np.random.default_rng(seed)
    w = rng.integers(0, len(QUARTER), n); a = 2 * np.pi * rng.random(n)
    return (QUARTER[w, 0] + R_CU * np.cos(a), QUARTER[w, 1] + R_CU * np.sin(a),
            np.cos(a), np.sin(a), rng.uniform(I_MIN, I_MAX, n))


def sample_walls(n, seed=75):
    """n points on the two walls of the quarter (x = 5 mm and y = 10 mm, by
    length), and n currents I [A]."""
    rng = np.random.default_rng(seed)
    s = rng.random(n) * (pb.A_HALF + pb.B_HALF)
    side = s < pb.B_HALF
    x = np.where(side, pb.A_HALF, s - pb.B_HALF); y = np.where(side, s, pb.B_HALF)
    return x, y, rng.uniform(I_MIN, I_MAX, n)


def sample_symmetry(n, seed=76):
    """n points on the two symmetry lines (x = 0 and y = 0) in the insulation,
    with the unit normal of the line, and n currents I [A]."""
    rng = np.random.default_rng(seed)
    s = rng.random(3 * n) * (pb.A_HALF + pb.B_HALF)
    side = s < pb.B_HALF
    x = np.where(side, 0.0, s - pb.B_HALF); y = np.where(side, s, 0.0)
    ok = in_resin(x, y)
    x, y, side = x[ok][:n], y[ok][:n], side[ok][:n]
    return x, y, side.astype(float), 1.0 - side, rng.uniform(I_MIN, I_MAX, len(x))


# ------------------------------------------------------------ quadrature
HALF_PITCH = PITCH_X / 2   #: m, 1.25 mm: each core sits in its own 2.5 mm square cell of the winding


def resin_quadrature(nr=24, nt=24):
    """Gauss points and weights covering the insulation of the quarter
    exactly: each core's 2.5 mm square, minus the core, in polar coordinates
    about its centre - the angle split into eight octants (on each the square's
    edge is a smooth function of the angle), nt points per octant; the radius
    from the copper's edge to the square's, nr points, graded towards the
    copper where the field bends most. Returns x, y [m] and weights [m^2];
    the weights sum to the insulation's area, 24.87 mm^2."""
    gr, wr = np.polynomial.legendre.leggauss(nr)
    gt, wt = np.polynomial.legendre.leggauss(nt)
    pts, wts = [], []
    for k in range(8):
        a0, a1 = k * np.pi / 4, (k + 1) * np.pi / 4
        th = a0 + (gt + 1) / 2 * (a1 - a0); wth = wt / 2 * (a1 - a0)
        rmax = HALF_PITCH / np.maximum(np.abs(np.cos(th)), np.abs(np.sin(th)))
        for t, w, rm in zip(th, wth, rmax):
            s = (gr + 1) / 2
            r = R_CU + (rm - R_CU) * s ** 2                   # graded: r = R + (r_max - R) s^2
            w_r = wr / 2 * (rm - R_CU) * 2 * s
            pts.append(np.c_[r * np.cos(t), r * np.sin(t)]); wts.append(w * w_r * r)
    p, w = np.concatenate(pts), np.concatenate(wts)
    xy = np.concatenate([p + c for c in QUARTER])
    return xy[:, 0], xy[:, 1], np.tile(w, len(QUARTER))


def copper_quadrature(nr=8, nt=32):
    """Gauss points and weights over the eight copper cores of the quarter:
    x, y [m], weights [m^2] summing to 8 pi R^2 = 25.13 mm^2."""
    gr, wr = np.polynomial.legendre.leggauss(nr)
    r = (gr + 1) / 2 * R_CU; w_r = wr / 2 * R_CU * r
    th = np.arange(nt) * 2 * np.pi / nt
    R, TH = np.meshgrid(r, th); W = np.outer(np.full(nt, 2 * np.pi / nt), w_r)
    xy = np.concatenate([np.c_[c[0] + (R * np.cos(TH)).ravel(), c[1] + (R * np.sin(TH)).ravel()] for c in QUARTER])
    return xy[:, 0], xy[:, 1], np.tile(W.ravel(), len(QUARTER))


# ------------------------------------------------------------ pictures
def draw_slot(ax=None, quarter=True, zoom=None):
    """The slot with every wire resolved: copper cores, their enamel coats and
    the resin between, to scale; the quarter that is solved outlined.
    ``zoom`` = (x0, x1, y0, y1) in mm shows only that window."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle
    if ax is None:
        _, ax = plt.subplots(figsize=(4.2, 7.0))
    a, b = pb.A_HALF * 1e3, pb.B_HALF * 1e3
    ax.add_patch(Rectangle((-a, -b), 2 * a, 2 * b, facecolor="#f2e6c9", edgecolor="#1f77b4", lw=2.4, zorder=0))
    for cx, cy in CENTRES * 1e3:
        ax.add_patch(Circle((cx, cy), R_ENAMEL * 1e3, facecolor="#2e7d32", edgecolor="none", zorder=1))
        ax.add_patch(Circle((cx, cy), R_CU * 1e3, facecolor="#d9772b", edgecolor="#8f2f18", lw=0.4, zorder=2))
    if quarter:
        ax.add_patch(Rectangle((0, 0), a, b, facecolor="none", edgecolor="k", lw=1.6, ls="--", zorder=3))
        ax.text(a / 2, b + 0.5, "the quarter solved", ha="center", va="bottom", fontsize=8)
    ax.plot([], [], "s", color="#d9772b", label="copper, 400 W/(m·K)")
    ax.plot([], [], "s", color="#2e7d32", label="enamel, 0.20 W/(m·K)")
    ax.plot([], [], "s", color="#f2e6c9", label="resin, 0.20 W/(m·K)")
    ax.set_aspect("equal")
    if zoom:
        ax.set_xlim(zoom[0], zoom[1]); ax.set_ylim(zoom[2], zoom[3])
    else:
        ax.set_xlim(-a - 0.5, a + 0.5); ax.set_ylim(-b - 0.5, b + 1.5)
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
    return ax


# ----------------------------------------------------------- the truth file
TRUTH_FILE = "MP7.1A_truth.npz"


def write_truth(path=TRUTH_FILE):
    """The ground truth at seven currents on 400 x 800 cells of the quarter,
    with the convergence table, the zero-strength check and every parameter."""
    rows = convergence_study(verbose=True)
    out = dict(current=np.array(I_TRUTH), test_currents=np.array(I_TEST), reference_h_mm=np.array(REF_H * 1e3))
    th, hot, bal, wires = [], [], [], []
    for I in I_TRUTH:
        g = ground_truth(I)
        th.append(g["theta"].astype(np.float32)); hot.append(g["hot"])
        bal.append([g["balance"]["made"], g["balance"]["out"]]); wires.append(wire_temperatures(g))
        print(f"   {I:5.2f} A: hot spot {g['hot']:.4f} K")
    out.update(theta=np.stack(th), hot_spot=np.array(hot), balance=np.array(bal), wire_theta=np.array(wires),
               x=g["x"], y=g["y"], k=g["k"].astype(np.float32), copper_fraction=g["frac"].astype(np.float32),
               wire_centres=QUARTER, copper_spread_K=copper_spread(g))
    out["conv_h_mm"] = np.array([r[0] * 1e3 for r in rows]); out["conv_cells"] = np.array([r[1] for r in rows])
    out["conv_hot"] = np.array([r[2] for r in rows]); out["conv_balance"] = np.array([r[3] for r in rows])
    out["conv_seconds"] = np.array([r[4] for r in rows]); out["conv_field_change"] = np.array([r[5] for r in rows])
    out["zero_strength_error_K"] = np.array(check_zero_strength(verbose=False))
    out["params"] = np.array([f"{k} = {v}" for k, v in dict(
        K_CU="400 W/(m K), Incropera & DeWitt table A.1", K_ENAMEL="0.20 W/(m K), ASSUMED",
        K_RESIN="0.20 W/(m K), ASSUMED", R_CU="1.000 mm", R_ENAMEL="1.0375 mm, IEC 60317 grade 2 minimum",
        PITCH="2.5 mm both ways", SLOT="10 x 20 mm, walls at 90 C", RHO_CU="1.72e-8 Ohm m at 20 C",
        ALPHA_CU="3.93e-3 1/K", I_RANGE="10-45 A RMS per wire").items()])
    out["README"] = np.array(
        "MP7.1A ground truth, finite volumes on the quarter slot x >= 0, y >= 0 (symmetric about both axes). "
        "theta[i, row, col]: rise above the 90 C wall [K] at current[i] [A] on cell centres x[col], y[row] [m], "
        "0.0125 mm cells (float32). hot_spot[i]: largest theta [K]. wire_theta[i, j]: mean rise of copper core j [K], "
        "centres wire_centres[j] [m]. balance[i]: heat made and heat leaving through the walls, whole slot [W/m]. "
        "k: cell conductivity [W/(m K)]; copper_fraction: share of each cell that is copper (where heat is made). "
        "conv_*: at rated current on five grids (the last finer than the ground truth): cell size mm, cells, hot spot K, energy balance residual, solve s, "
        "largest change of the field at the 100 x 200 scoring points from the grid before K. "
        "zero_strength_error_K: at S = 0 the solver against Ex07.1's exact series. test_currents: scored, not trained on. "
        "params: the data and their sources.")
    np.savez_compressed(path, **out)
    print("written", path)


if __name__ == "__main__":
    import sys
    print("zero-strength check")
    check_zero_strength()
    print("convergence at rated current")
    convergence_study()
    if "--save" in sys.argv:
        write_truth()
