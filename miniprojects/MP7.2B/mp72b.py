r"""MP7.2B - The eigenvalues of a real slot. Ex07.2's stator slot (notebook 2)
with its sides tapered, as the slots between parallel-sided teeth are: the
first three eigenvalues and modes of its Helmholtz operator, and the current
at which the copper's feedback runs away, for any taper. The physics, the
reference model and the ground truth.

*Deep Learning for Engineering* - MSc, Aalborg University.
Remus Teodorescu (ret@et.aau.dk), with support from Research Assistant
Noman Khan (nomank@energy.aau.dk).

This module extends Ex07.2's ``slot_problem.py`` (course policy C10 rule 1)
and is used by the notebook of MP7.2B (course policy C14).

## What changes against Ex07.2, and why

Ex07.2's slot is a 10 x 20 mm rectangle, and its eigenvalues are exact,
lambda_mn = pi^2 (m^2/W^2 + n^2/H^2). A real slot between teeth with
parallel sides widens towards the back of the stator. Here the slot keeps its
20 mm depth and its 200 mm^2 (mean width 10 mm, so the copper fill and the
heat per volume are Ex07.2's) and tapers linearly from the opening, width
W_o, to the back, width W_b = 20 mm - W_o. The taper's strength is
tau = (W_b - 10 mm) / 10 mm: at tau = 0 the slot is Ex07.2's rectangle and
the solver must reproduce the exact eigenvalues (:func:`check_zero_strength`).
With tau > 0 the walls are slanted, the modes do not separate, and **no exact
solution exists**.

## The problem

    -lap(u) = lambda u     in the slot,   u = 0 on its four walls

the eigenpairs (lambda_k, u_k), k = 1, 2, 3, lambda in 1/mm^2. The first sets
the runaway current (Ex07.2, notebook 2): the copper's heat grows with its
temperature by c1 per kelvin, and when c1 reaches k_eff lambda_1 no steady
state exists, at

    I_run = A_wire sqrt(k_eff lambda_1 / (FILL rho_20 alpha_Cu))

## Two solvers, one scheme

The slot is mapped onto a square, xi = x / a(y), eta = y / B, with a(y) the
half-width, and covered by a grid of nodes in (xi, eta): its columns follow
the slanted walls, so every wall node is on a wall. On that grid the
eigenproblem is solved by bilinear finite elements: the stiffness and mass
of every four-node cell, a generalised sparse eigenproblem K u = lambda M u,
the three smallest by shift-and-invert (scipy's eigsh). Second order in the
node spacing, and Richardson extrapolation applies. (Finite differences on a
square grid cut the slanted walls into stair steps, and their eigenvalues
then converge irregularly: :func:`staircase` shows it.)

* the **reference model** (:func:`reference`): the scheme on an engineering grid, timed.
* the **ground truth** (:func:`ground_truth`): the scheme on grids each twice
  as fine, down to 0.03125 mm across the slot, Richardson-extrapolated.
"""

from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import slot_problem as sl

# ------------------------------------------------------------------ the data
B = sl.B_HALF                  #: m, half the depth: the slot runs from y = -B (the opening) to y = +B (the back)
DEPTH = 2 * B                  #: m, 20 mm (Ex07.2)
W_MEAN = 2 * sl.A_HALF         #: m, the mean width, 10 mm: the area stays Ex07.2's 200 mm²
WB_RANGE = (10.0e-3, 12.5e-3)  #: m, the width at the back: the network's input range (the opening is 20 mm minus it)
WB_TEST = (10.5e-3, 11.25e-3, 12.0e-3)   #: m, the widths the answers are scored at (12 mm: the 8-to-12 mm slot)
WB_OUT = (9.5e-3, 13.0e-3)     #: m, two widths outside the range, for a question of the report
#: the agreed accuracy (part A): relative error of lambda_1, of lambda_2 and lambda_3, and the runaway current
LAM1_TOL, LAM23_TOL, RUN_TOL = 0.003, 0.01, 0.5


def taper(wb):
    """The taper's strength tau = (W_b - 10 mm) / 10 mm for a back width wb [m]."""
    return (wb - W_MEAN) / W_MEAN                              # NumPy or torch


def half_width(y, wb):
    """m, the slot's half-width at y (from -B at the opening to +B at the back) for a back width wb. NumPy or torch."""
    return 0.5 * W_MEAN * (1 + taper(wb) * y / B)


def rectangle_eigenvalues(width=W_MEAN, depth=DEPTH, k=3):
    """The k smallest exact eigenvalues [1/m²] of a rectangle held at zero on its walls."""
    lam = sorted(np.pi ** 2 * (m ** 2 / width ** 2 + n ** 2 / depth ** 2) for m in range(1, 8) for n in range(1, 12))
    return np.array(lam[:k])


def runaway(lam1):
    """A, the runaway current for a first eigenvalue lam1 [1/m²] (Ex07.2's formula with lambda_1 in place
    of the rectangle's)."""
    return sl.A_WIRE * np.sqrt(sl.K_EFF * np.asarray(lam1) / (sl.FILL * sl.RHO_CU * sl.ALPHA_CU))


# --------------------------------------------------- the boundary-fitted grid
class TaperedSlot:
    """Bilinear finite elements on a grid of nx x 2 nx cells that follows the
    walls of the slot with back width wb. ``eigen(k)`` returns the k smallest
    eigenvalues [1/m²] and the modes on the nodes (k, 2 nx + 1, nx + 1),
    normalised to a unit integral of u² and positive where largest."""

    def __init__(self, nx, wb):
        self.nx, self.ny, self.wb = nx, 2 * nx, wb
        self.xi = np.linspace(-1.0, 1.0, nx + 1)
        self.eta = np.linspace(-1.0, 1.0, self.ny + 1)
        XI, ETA = np.meshgrid(self.xi, self.eta)                 # (ny + 1, nx + 1), row = eta
        self.Y = B * ETA
        self.X = half_width(self.Y, wb) * XI
        self._assemble()

    def _assemble(self):
        nx, ny = self.nx, self.ny
        idx = np.arange((nx + 1) * (ny + 1)).reshape(ny + 1, nx + 1)
        corners = np.stack([idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]], -1).reshape(-1, 4)   # anticlockwise
        xc, yc = self.X.ravel()[corners], self.Y.ravel()[corners]           # (cells, 4)
        g = 1 / np.sqrt(3.0)
        Kl = np.zeros((len(corners), 4, 4)); Ml = np.zeros_like(Kl)
        sr, ss = np.array([-1, 1, 1, -1.0]), np.array([-1, -1, 1, 1.0])
        for r in (-g, g):                                       # 2 x 2 Gauss points of each cell
            for s in (-g, g):
                N = 0.25 * (1 + sr * r) * (1 + ss * s)
                dNr, dNs = 0.25 * sr * (1 + ss * s), 0.25 * ss * (1 + sr * r)
                J11, J12 = xc @ dNr, yc @ dNr                   # dx/dr, dy/dr
                J21, J22 = xc @ dNs, yc @ dNs                   # dx/ds, dy/ds
                det = J11 * J22 - J12 * J21
                dNx = (J22[:, None] * dNr - J12[:, None] * dNs) / det[:, None]
                dNy = (-J21[:, None] * dNr + J11[:, None] * dNs) / det[:, None]
                Kl += det[:, None, None] * (dNx[:, :, None] * dNx[:, None, :] + dNy[:, :, None] * dNy[:, None, :])
                Ml += det[:, None, None] * N[None, :, None] * N[None, None, :]
        rows = np.repeat(corners, 4, axis=1).ravel(); cols = np.tile(corners, (1, 4)).ravel()
        n = (nx + 1) * (ny + 1)
        K = sp.csr_matrix((Kl.ravel(), (rows, cols)), shape=(n, n))
        M = sp.csr_matrix((Ml.ravel(), (rows, cols)), shape=(n, n))
        wall = np.zeros((ny + 1, nx + 1), bool)
        wall[[0, -1], :] = True; wall[:, [0, -1]] = True
        self.inner = np.flatnonzero(~wall.ravel())
        self.K = K[self.inner][:, self.inner].tocsc()
        self.M = M[self.inner][:, self.inner].tocsc()
        self.Mfull = M

    def eigen(self, k=3):
        lam, vec = spla.eigsh(self.K, k=k, M=self.M, sigma=0.0, which="LM")
        o = np.argsort(lam)
        lam, vec = lam[o], vec[:, o]
        modes = np.zeros((k, (self.ny + 1) * (self.nx + 1)))
        modes[:, self.inner] = vec.T
        for i in range(k):                                      # unit integral of u² [1/m²], positive where largest
            modes[i] /= np.sqrt(modes[i] @ (self.Mfull @ modes[i]))
            modes[i] *= np.sign(modes[i][np.argmax(np.abs(modes[i]))])
        return lam, modes.reshape(k, self.ny + 1, self.nx + 1)


def staircase(h, wb, k=3):
    """For comparison only: the five-point finite differences of Ex07.2 on a
    square grid of spacing h [m], the nodes inside the tapered slot kept and
    the walls cut into stair steps. Returns the k smallest eigenvalues [1/m²]."""
    x = np.arange(-W_MEAN, W_MEAN + h / 2, h)
    y = np.arange(-B, B + h / 2, h)
    X, Y = np.meshgrid(x, y)
    inside = (np.abs(X) < half_width(Y, wb) - 1e-12) & (np.abs(Y) < B - 1e-12)
    ids = -np.ones(inside.shape, int)
    ids[inside] = np.arange(inside.sum())
    r, c = np.nonzero(inside)
    rows, cols, vals = [ids[r, c]], [ids[r, c]], [np.full(len(r), 4 / h ** 2)]
    for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        rr, cc = r + dr, c + dc
        ok = inside[rr, cc]
        rows.append(ids[r[ok], c[ok]]); cols.append(ids[rr[ok], cc[ok]]); vals.append(np.full(ok.sum(), -1 / h ** 2))
    A = sp.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))))
    return np.sort(spla.eigsh(A, k=k, sigma=0.0, which="LM")[0])


def check_zero_strength(nx=(160, 320), verbose=True):
    """At tau = 0 (a back width of 10 mm) the slot is Ex07.2's rectangle: the
    scheme, extrapolated from two grids, must give the exact eigenvalues.
    Returns the largest relative difference of the three."""
    lam = [TaperedSlot(n, W_MEAN).eigen()[0] for n in nx]
    rich = lam[1] + (lam[1] - lam[0]) / 3
    ex = rectangle_eigenvalues()
    err = float(np.abs(rich / ex - 1).max())
    if verbose:
        print("   tau = 0: " + ", ".join(f"{l/1e6:.6f}" for l in rich) + " per mm² against Ex07.2's exact "
              + ", ".join(f"{l/1e6:.6f}" for l in ex) + f"; largest relative difference {err:.1e}")
    return err


# ------------------------------------------------------- the reference model
REF_NX = 20                    #: cells across the slot (0.5 mm), 40 along it: the engineering grid, chosen in the convergence study


def reference(wb, nx=REF_NX):
    """The reference model: the scheme on the engineering grid for a back
    width wb [m]. Returns the eigenvalues [1/m²], the runaway current [A],
    the modes on its nodes (3, 2 nx + 1, nx + 1) and the wall-clock time [s]."""
    t0 = time.perf_counter()
    lam, modes = TaperedSlot(nx, wb).eigen()
    return dict(lam=lam, runaway=float(runaway(lam[0])), modes=modes, seconds=time.perf_counter() - t0)


# -------------------------------------------------------- the ground truth
TRUTH_NX = (40, 80, 160, 320)  #: cells across: 0.25, 0.125, 0.0625 and 0.03125 mm (at the mean width)


def convergence_study(wb, grids=TRUTH_NX, verbose=True):
    """The three eigenvalues on grids each twice as fine; the observed order
    from the last three, and the Richardson extrapolation from the last two at
    order two. Returns rows (nx, lam (3,), seconds), the order (3,), the
    extrapolated eigenvalues (3,) and the modes of the finest grid."""
    rows, modes = [], None
    for n in grids:
        t0 = time.perf_counter()
        lam, modes = TaperedSlot(n, wb).eigen()
        rows.append((n, lam, time.perf_counter() - t0))
        if verbose:
            print(f"   {n:4d} x {2*n} cells ({W_MEAN/n*1e3:.5f} mm across at the mean width): lambda = "
                  + ", ".join(f"{l/1e6:.6f}" for l in lam) + f" per mm²  ({rows[-1][2]:.1f} s)", flush=True)
    a, b, c = (r[1] for r in rows[-3:])
    order = np.log2((b - a) / (c - b))
    rich = c + (c - b) / 3
    if verbose:
        print("   observed order " + ", ".join(f"{o:.2f}" for o in order) + "; extrapolated "
              + ", ".join(f"{l/1e6:.6f}" for l in rich) + f" per mm²; runaway {runaway(rich[0]):.2f} A")
    return rows, order, rich, modes


def ground_truth(wb, grids=TRUTH_NX[-2:]):
    """The extrapolated eigenvalues [1/m²] and the runaway current [A] at a back
    width wb, and the modes of the finest grid read at the reference grid's
    nodes (3, 41, 21) [1/m] (the grids are nested)."""
    lam = [TaperedSlot(n, wb).eigen() for n in grids]
    rich = lam[1][0] + (lam[1][0] - lam[0][0]) / 3
    step = grids[1] // REF_NX
    return dict(lam=rich, runaway=float(runaway(rich[0])), modes=lam[1][1][:, ::step, ::step],
                change=lam[1][0] - lam[0][0])


# ------------------------------------------------------- the training points
def to_xy(xi, eta, wb):
    """The point (x, y) [m] of the slot of back width wb at mapped coordinates
    (xi, eta) in [-1, 1]²: x = a(y) xi, y = B eta. NumPy or torch."""
    y = B * eta
    return 0.5 * W_MEAN * (1 + (wb - W_MEAN) / W_MEAN * eta) * xi, y


def quadrature(n_xi=24, n_eta=48):
    """Gauss-Legendre points (xi, eta) on [-1, 1]² and weights, (n,) each: the
    integral over the slot of back width wb is sum(w * f * a(y) * B), with a
    the half-width at the point (:func:`half_width`)."""
    g1, w1 = np.polynomial.legendre.leggauss(n_xi)
    g2, w2 = np.polynomial.legendre.leggauss(n_eta)
    XI, ETA = np.meshgrid(g1, g2)
    return XI.ravel(), ETA.ravel(), (w2[:, None] * w1[None, :]).ravel()


def sample_widths(n, seed=72):
    """n back widths [m], uniform over the training range."""
    return np.random.default_rng(seed).uniform(*WB_RANGE, n)


def ref_nodes(wb, nx=REF_NX):
    """The reference grid's nodes for a back width wb: x, y [m], (2 nx + 1, nx + 1) each.
    The ground truth's modes are given there."""
    XI, ETA = np.meshgrid(np.linspace(-1, 1, nx + 1), np.linspace(-1, 1, 2 * nx + 1))
    return to_xy(XI, ETA, wb)


# ------------------------------------------------------------------ pictures
def draw_slot(ax=None, wb=12e-3, modes=None, wires=True):
    """The tapered slot of back width wb [mm drawn], with Ex07.2's 32 wires
    faintly and the rectangle of Ex07.2 dashed; or, with ``modes`` (a field on
    the reference nodes), that field in colour."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Circle
    if ax is None:
        _, ax = plt.subplots(figsize=(3.6, 6.4))
    a0, ab = half_width(-B, wb) * 1e3, half_width(B, wb) * 1e3
    b = B * 1e3
    corners = [(-a0, -b), (a0, -b), (ab, b), (-ab, b)]
    if modes is None:
        ax.add_patch(Polygon(corners, closed=True, facecolor="#f2e6c9", edgecolor="none", zorder=1))
    ax.add_patch(Polygon(corners, closed=True, facecolor="none", edgecolor="#1f77b4", lw=2.4, zorder=3))
    ax.plot([-5, 5, 5, -5, -5], [-b, -b, b, b, -b], "k--", lw=0.8, zorder=4, label="Ex07.2's rectangle")
    if modes is not None:                                  # a signed field: zero is white, red positive, blue negative
        X, Y = ref_nodes(wb)
        m = float(np.abs(modes).max()) or 1.0
        ax.contourf(X * 1e3, Y * 1e3, modes, levels=np.linspace(-m, m, 22), cmap="RdBu_r", zorder=1)
    elif wires:
        for i in range(sl.N_COLS):
            for j in range(sl.N_ROWS):
                yc = -b + (j + 0.5) * 2 * b / sl.N_ROWS
                ax.add_patch(Circle(((-1 + (i + 0.5) * 2 / sl.N_COLS) * half_width(yc / 1e3, wb) * 1e3, yc), sl.WIRE_D / 2 * 1e3,
                                    facecolor="#d9772b", alpha=0.25, edgecolor="none", zorder=2))
    ax.text(0, -b - 0.6, f"opening {2*a0:.1f} mm", ha="center", va="top", fontsize=8)
    ax.text(0, b + 0.6, f"back {2*ab:.1f} mm", ha="center", va="bottom", fontsize=8)
    ax.set_aspect("equal"); ax.set_xlim(-7.5, 7.5); ax.set_ylim(-b - 2.2, b + 2.2)
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
    return ax


# ----------------------------------------------------------- the truth file
TRUTH_FILE = "MP7.2B_truth.npz"


def width_key(wb):
    """The key of a back width in the truth file: 12 mm -> '12mm', 11.25 mm -> '11.25mm'."""
    return f"{wb*1e3:g}mm"


def write_truth(path=TRUTH_FILE):
    """The ground truth at the worked example and test widths and the two
    outside the range: eigenvalues, runaway current, modes on the reference
    nodes; the convergence study at 12 mm, the zero-strength check and the
    staircase comparison; every parameter with its unit."""
    print("zero-strength check")
    zs = check_zero_strength()
    print("convergence at 12 mm")
    rows, order, rich, _ = convergence_study(12e-3)
    widths = sorted(set(WB_TEST + WB_OUT))
    out = dict(widths=np.array(widths), rect_lam=rectangle_eigenvalues(), rect_runaway=np.array(sl.runaway_current()))
    for wb in widths:
        g = ground_truth(wb)
        k = width_key(wb)
        out[f"lam_{k}"], out[f"runaway_{k}"], out[f"modes_{k}"] = g["lam"], np.array(g["runaway"]), g["modes"]
        out[f"change_{k}"] = g["change"]
        print(f"   back {wb*1e3:5.2f} mm: lambda " + ", ".join(f"{l/1e6:.6f}" for l in g["lam"])
              + f" per mm², runaway {g['runaway']:.2f} A; last change of the grids "
              + ", ".join(f"{c/l:.1e}" for c, l in zip(g["change"], g["lam"])), flush=True)
    out["conv_nx"] = np.array([r[0] for r in rows]); out["conv_lam"] = np.array([r[1] for r in rows])
    out["conv_seconds"] = np.array([r[2] for r in rows]); out["conv_order"] = order; out["conv_extrapolated"] = rich
    stair_h = (0.25e-3, 0.125e-3, 0.0625e-3, 0.03125e-3)
    out["stair_h"] = np.array(stair_h)
    out["stair_lam"] = np.array([staircase(h, 12e-3) for h in stair_h])
    out["zero_strength_rel"] = np.array(zs)
    out["params"] = np.array([f"{k} = {v}" for k, v in dict(
        SLOT="20 mm deep, area 200 mm^2 (Ex07.2), opening 20 mm - back width", K_EFF="0.70 W/(m K) (Ex07.2)",
        FILL="0.503 (Ex07.2)", RHO_CU="1.72e-8 Ohm m at 20 C, +0.393 %/K (Ex07.2)", WB_RANGE="10-12.5 mm, the network's input").items()])
    out["README"] = np.array(
        "MP7.2B ground truth: the first three eigenvalues of -lap u = lambda u, u = 0 on the walls, in a slot 20 mm deep "
        "tapering linearly from the opening (y = -10 mm) to the back (y = +10 mm), mean width 10 mm. widths: back widths [m]. "
        "lam_<w>mm: the three eigenvalues [1/m^2], Richardson-extrapolated from 160 x 320 and 320 x 640 bilinear elements "
        "on a grid that follows the walls; change_<w>mm: their change between those two grids [1/m^2]; runaway_<w>mm: the "
        "runaway current [A]; modes_<w>mm: the three modes (3, 41, 21) [1/m] on the reference grid's nodes "
        "(mp72b.ref_nodes), unit integral of u^2, positive where largest. rect_*: Ex07.2's 10 x 20 mm rectangle, exact. "
        "conv_*: the 12 mm convergence study (cells across, eigenvalues 1/m^2, seconds, observed order, extrapolated). "
        "stair_*: five-point finite differences on a square grid at 12 mm, the walls as stair steps (spacing m, "
        "eigenvalues 1/m^2). zero_strength_rel: at 10 mm against the exact rectangle, the largest relative difference.")
    np.savez_compressed(path, **out)
    print("written", path)


if __name__ == "__main__":
    import sys
    if "--save" in sys.argv:
        write_truth()
    else:
        check_zero_strength()
        convergence_study(12e-3, TRUTH_NX[:3])
