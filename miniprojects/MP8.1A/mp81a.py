r"""MP8.1A — The fins resolved: steady conduction in the whole cross-section of
the heat sink, base plate and ten fins, with the air's coefficient on every
face in the air. The physics, the reference model and the ground truth.

*Deep Learning for Engineering* — MSc, Aalborg University.
Remus Teodorescu (ret@et.aau.dk), with support from Research Assistant
Noman Khan (nomank@energy.aau.dk).

This module extends Ex08's ``heatsink.py`` (course policy C10 rule 1: one set
of physics and constants for the exercise and its mini project) and is used by
the notebook of MP8.1A (course policy C14).

## What changes against Ex08, and why

Ex08 collapses the ten fins into one effective coefficient on the fin side of
the base plate, h_eff = 322 W/(m²K), by the textbook fin efficiency. That
makes the base plate a rectangle with an exact series solution. The collapse
assumes that every fin root sits at the local temperature of a smooth fin
side and that heat leaves the base evenly across it. In the real sink the heat
leaves through ten 2 mm roots, and the metal between them is bare. Here the
fins are resolved: the domain is the whole comb, base plate and fins, and the
air's coefficient h acts on every face in the air - the fins' two sides and
their tips, and the bare base between the fins. The domain is no longer a
rectangle, separation of variables fails at the fin roots, and **no exact
solution exists**.

The change is a change of geometry, not of a coefficient, so it has no
strength parameter that tends continuously to Ex08. The solver is checked
instead in Ex08's own limit: run with the fins removed and h_eff on the fin
side of the base (``collapsed=True``), the same code must reproduce Ex08's
exact series (:func:`check_collapsed`).

## The problem, in kelvin above the air

    k (∂²θ/∂x² + ∂²θ/∂z²) = 0         in the metal: base plate 0 < z < t_b, fins -L_fin < z < 0
    k ∂θ/∂n = (P / P0) q''            on the module's footprint, z = t_b
    ∂θ/∂n = 0                         on the rest of the top face and at the base's two ends
    -k ∂θ/∂n = h θ                    on every face in the air: fin sides, fin tips, bare base

n is the outward normal. P, the module's power, and h, the air's coefficient,
are the two inputs of the network: P from 100 to 200 W, h from 20 to 60 W/(m²K)
(a slow and a fast fan).

## Two solvers

* the **reference model** (:func:`reference`): finite volumes on square cells
  of 1 mm, the coarsest grid that meets the agreed accuracy.
* the **ground truth** (:func:`ground_truth`): the same scheme on cells each
  half the size, converged and extrapolated (:func:`convergence_study`).
"""

from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import heatsink as hs

# ------------------------------------------------------------------ the data
P0 = hs.POWER                  #: W, the power Ex08's flux is set for (150 W)
P_MIN, P_MAX = 100.0, 200.0    #: W, the range of the module's power, an input of the network
H_MIN, H_MAX = 20.0, 60.0      #: W/(m²K), the range of the air's coefficient, an input   [typical values, ASSUMED]
TESTS = ((115.0, 25.0), (165.0, 45.0), (195.0, 55.0))   #: (P W, h W/(m²K)), the cases scored (not trained on)
Z_BOTTOM = -hs.L_FIN           #: m, the fin tips
S_RES = hs.Q_MEAN * hs.T_BASE / hs.K_AL   #: K, the rise across the base's thickness at 150 W (0.90 K): the residuals' scale
FIN_CENTRES = (np.arange(hs.N_FINS) + 0.5) * hs.PITCH   #: m, the fins' centre lines


def in_metal(x, z):
    """True where (x, z) (m) is in the metal: the base plate or a fin."""
    x = np.asarray(x, dtype=float); z = np.asarray(z, dtype=float)
    base = (z >= 0) & (z <= hs.T_BASE) & (x >= 0) & (x <= hs.W_BASE)
    xl = np.mod(x, hs.PITCH) - hs.PITCH / 2                       # m, across the nearest fin from its centre line
    fin = (z < 0) & (z >= Z_BOTTOM) & (np.abs(xl) <= hs.T_FIN / 2) & (x >= 0) & (x <= hs.W_BASE)
    return base | fin


# ------------------------------------------------------------- finite volumes
class Comb:
    """Finite volumes on square cells of side ``dx`` over the comb, per metre of
    depth. The geometry and the conduction between cells are built once; a
    new air coefficient changes only the diagonal, so :meth:`solve` assembles
    that and solves. ``collapsed=True`` removes the fins and puts Ex08's
    h_eff(h) on the fin side of the base instead: Ex08's problem, for the check."""

    def __init__(self, dx, collapsed=False):
        self.dx, self.collapsed = dx, collapsed
        W, tb = hs.W_BASE, hs.T_BASE
        nx = int(round(W / dx))
        nzb = int(round(tb / dx))
        nzf = 0 if collapsed else int(round(hs.L_FIN / dx))
        nz = nzb + nzf
        self.xc = (np.arange(nx) + 0.5) * dx                      # m, cell centres across
        self.zc = -nzf * dx + (np.arange(nz) + 0.5) * dx          # m, cell centres up
        X, Z = np.meshgrid(self.xc, self.zc)
        metal = in_metal(X, Z) if not collapsed else (Z > 0)
        self.X, self.Z, self.metal = X, Z, metal
        idx = -np.ones(metal.shape, int); idx[metal] = np.arange(metal.sum())
        self.idx, self.n = idx, int(metal.sum())
        k = hs.K_AL
        rows, cols = [], []
        cond_diag = np.zeros(self.n)
        air_count = np.zeros(self.n)                              # faces of each cell in the air
        insul = np.zeros(self.n)                                  # (diagnostic) faces insulated
        for axis, step in ((0, 1), (0, -1), (1, 1), (1, -1)):
            nb = np.zeros_like(metal)
            sl_src = [slice(None), slice(None)]; sl_dst = [slice(None), slice(None)]
            if step == 1:
                sl_src[axis] = slice(0, -1); sl_dst[axis] = slice(1, None)
            else:
                sl_src[axis] = slice(1, None); sl_dst[axis] = slice(0, -1)
            nb[tuple(sl_src)] = metal[tuple(sl_dst)]              # is the neighbour in this direction metal?
            both = metal & nb
            i = idx[both]
            jj = np.full(metal.shape, -1); jj[tuple(sl_src)] = idx[tuple(sl_dst)]
            rows += list(i); cols += list(jj[both])
            np.add.at(cond_diag, i, 1.0)
            open_ = metal & ~nb                                   # a face of a metal cell with no metal beyond
            # which of those faces are in the air: not the top face, not the base's two ends
            if axis == 0 and step == 1:                           # +z: the top face of the base
                air = open_ & (Z < tb - dx)                       # (never: the comb has no overhang)
            elif axis == 1:                                       # ±x: the base's ends are insulated; fin sides are air
                air = open_ & (Z < 0)
            else:                                                 # -z: the fin tips and the bare base between fins
                air = open_
            if collapsed and axis == 0 and step == -1:
                air = open_                                       # the fin side of the base carries h_eff
            np.add.at(air_count, idx[air], 1.0)
            np.add.at(insul, idx[open_ & ~air], 1.0)
        self.Kc = (sp.coo_matrix((-k * np.ones(len(rows)), (rows, cols)), shape=(self.n, self.n)).tocsr()
                   + sp.diags(k * cond_diag)).tocsc()             # conduction between cells, W/(m K) per metre of depth
        self.air_count = air_count
        # the module's flux into the top cells of the footprint
        top = metal & np.isclose(Z, tb - dx / 2)
        self.top = top
        foot = top & (X > hs.MODULE[0]) & (X < hs.MODULE[1])
        self.b0 = np.zeros(self.n); self.b0[idx[foot]] = hs.FLUX * dx   # W per metre of depth at P0
        self.flux_top = np.where(foot, hs.FLUX, 0.0)[top]               # W/m² on each top cell at P0
        self.x_top = X[top]

    def g_air(self, h):
        """The conductance of one face in the air, W/(m K) per metre of depth:
        half a cell of metal in series with the air's coefficient."""
        if self.collapsed:
            h = hs.fin_efficiency(h)[2]
        return h * self.dx / (1.0 + h * self.dx / (2 * hs.K_AL))

    def solve(self, power=P0, h=hs.H_AIR):
        """The rise (K) in every metal cell, as a flat vector, for the module's
        power ``power`` (W) and the air's coefficient ``h`` (W/(m²K))."""
        A = self.Kc + sp.diags(self.g_air(h) * self.air_count)
        return spla.spsolve(A.tocsc(), (power / P0) * self.b0)

    def field(self, th):
        """The rise on the rectangle of cells, NaN in the air."""
        out = np.full(self.metal.shape, np.nan); out[self.metal] = th
        return out

    def top_face(self, th, power=P0):
        """The rise on the top face of the base at the top cells' x: the cell
        value plus half a cell of the flux's slope (K)."""
        return th[self.idx[self.top]] +(power / P0) * self.flux_top * self.dx / (2 * hs.K_AL)

    def hottest(self, th, power=P0):
        """The rise at the centre of the module's face, x = W/2, z = t_b (K):
        the hottest point, from the two top cells either side of it."""
        tf = self.top_face(th, power)
        return float(np.interp(hs.W_BASE / 2, self.x_top, tf))

    def heat_out(self, th, h=hs.H_AIR):
        """The heat leaving through every face in the air, W for the 100 mm depth."""
        return float(hs.DEPTH * np.sum(self.g_air(h) * self.air_count * th))


# ------------------------------------------------------- the reference model
REF_DX = 1.0e-3        #: m, the engineering grid, chosen in the convergence study
_REF = {}


def _comb(dx):
    if dx not in _REF:
        _REF[dx] = Comb(dx)
    return _REF[dx]


def reference(power=P0, h=hs.H_AIR, dx=REF_DX):
    """The reference model: finite volumes on cells of ``dx``. The geometry is
    built once and cached, as an engineer's solver would; a new case assembles
    the air faces for its h and solves. Returns the hottest point (K), the heat
    out (W), the cell centres and the rise in each metal cell (K), and the
    wall-clock time of the new case (s)."""
    m = _comb(dx)
    t0 = time.perf_counter()
    th = m.solve(power, h)
    hot = m.hottest(th, power)
    wall = time.perf_counter() - t0
    return dict(hottest=hot, heat_out=m.heat_out(th, h), x=m.X[m.metal], z=m.Z[m.metal], theta=th, seconds=wall)


# -------------------------------------------------------- the ground truth
TRUTH_DX = (1.0e-3, 0.5e-3, 0.25e-3, 0.125e-3)   #: m, four grids each twice as fine


def convergence_study(power=P0, h=hs.H_AIR, grids=TRUTH_DX, verbose=True):
    """The hottest point and the heat out on grids each twice as fine. Finite
    volumes are second order where the field is smooth, but the twenty
    re-entrant corners at the fin roots make the field singular in its
    derivatives (it goes as r^(2/3) there), which lowers the observed order
    to about 1.4. The last three grids give that order, and the last two are
    extrapolated by Richardson with it. Returns the rows (dx, cells, hottest, heat out, balance
    error, seconds) and the extrapolated hottest point."""
    rows = []
    for dx in grids:
        t0 = time.perf_counter()
        m = Comb(dx); th = m.solve(power, h)
        q = m.heat_out(th, h)
        rows.append((dx, m.n, m.hottest(th, power), q, q / power - 1.0, time.perf_counter() - t0))
        if verbose:
            print(f"   cells of {dx*1e3:.3f} mm: {m.n:7d} cells, hottest {rows[-1][2]:.4f} K, heat out {q:.4f} W "
                  f"(balance {rows[-1][4]:+.1e}), {rows[-1][5]:.1f} s")
    h1, h2, h3 = (r[2] for r in rows[-3:])
    order = float(np.log2(abs(h2 - h1) / abs(h3 - h2)))         # the observed order of the last three grids
    ext = h3 + (h3 - h2) / (2 ** order - 1)                      # Richardson with the observed order
    if verbose:
        d = np.abs(np.diff([r[2] for r in rows]))
        print("   change of the hottest between grids:", np.array2string(d, precision=5),
              " ratios", np.array2string(d[:-1] / d[1:], precision=2))
        print(f"   observed order {order:.2f}; extrapolated hottest {ext:.4f} K")
    return rows, ext, order


def ground_truth(power=P0, h=hs.H_AIR, dx=TRUTH_DX[-1]):
    """The rise in every metal cell on the finest grid, with its hottest point and heat out."""
    m = Comb(dx); th = m.solve(power, h)
    return dict(x=m.X[m.metal], z=m.Z[m.metal], theta=th, hottest=m.hottest(th, power),
                heat_out=m.heat_out(th, h), dx=dx)


def check_collapsed(dx=0.125e-3, verbose=True):
    """Ex08's limit: the same finite volumes with the fins removed and h_eff on
    the fin side of the base must reproduce Ex08's exact series. Returns the
    largest difference over the base (K) and at the hottest point."""
    m = Comb(dx, collapsed=True)
    th = m.solve(P0, hs.H_AIR)
    ex = hs.exact(m.X[m.metal], m.Z[m.metal])
    err = float(np.abs(th - ex).max())
    hot_err = abs(m.hottest(th) - hs.hottest())
    if verbose:
        print(f"   fins collapsed, cells of {dx*1e3:.3f} mm: largest difference from Ex08's exact series {err:.2e} K, "
              f"at the hottest point {hot_err:.2e} K ({hs.hottest():.2f} K)")
    return err, hot_err


# ----------------------------------------------- Ex08's model, for comparison
def collapsed(x, z, power=P0, h=hs.H_AIR, n=200):
    """Ex08's collapsed model extended into the fins, in kelvin: the exact
    series of the base plate with h_eff(h) on its fin side, and below it each
    fin as the textbook 1-D fin from its root's temperature (the root at the
    fin's centre line). It is the model the resolved comb is compared with."""
    x = np.asarray(x, dtype=float); z = np.asarray(z, dtype=float)
    m, eta, he = hs.fin_efficiency(h)
    q, lam = hs._modes(n)
    q = q * power / P0
    def series(xx, zz):
        Z = np.empty((n, zz.size)); Z[0] = q[0] * (zz / hs.K_AL + 1 / he)
        l = lam[1:, None]; r = he / (hs.K_AL * l)
        num = (1 + r) * np.exp(l * (zz - hs.T_BASE)) + (1 - r) * np.exp(-l * (zz + hs.T_BASE))
        den = (1 + r) - (1 - r) * np.exp(-2 * l * hs.T_BASE)
        Z[1:] = q[1:, None] / (hs.K_AL * l) * num / den
        return np.sum(np.cos(np.outer(lam, xx)) * Z, axis=0)
    base = series(x, np.clip(z, 0, None))
    xc = FIN_CENTRES[_fin_index(x)]
    root = series(xc, np.zeros_like(z))
    s = hs.L_FIN + np.minimum(z, 0); r = h / (m * hs.K_AL)
    phi = (np.cosh(m * s) + r * np.sinh(m * s)) / (np.cosh(m * hs.L_FIN) + r * np.sinh(m * hs.L_FIN))
    return np.where(z >= 0, base, root * phi)


def air_quadrature(n_side=80, n_tip=4, n_bare=16):
    """Midpoints and lengths (m) of segments covering every face in the air,
    with a flag 1 on a fin face and 0 on the bare base: a sum of
    weight × h × rise over them, times the depth, is the heat out (W)."""
    xs, zs, ws, fin = [], [], [], []
    for c in FIN_CENTRES:
        zz = Z_BOTTOM * (np.arange(n_side) + 0.5) / n_side
        for sgn in (-1, 1):
            xs += list(np.full(n_side, c + sgn * hs.T_FIN / 2)); zs += list(zz); ws += [hs.L_FIN / n_side] * n_side
        xs += list(c - hs.T_FIN / 2 + hs.T_FIN * (np.arange(n_tip) + 0.5) / n_tip); zs += [Z_BOTTOM] * n_tip
        ws += [hs.T_FIN / n_tip] * n_tip; fin += [1.0] * (2 * n_side + n_tip)
    edges = [0.0] + sum([[c - hs.T_FIN / 2, c + hs.T_FIN / 2] for c in FIN_CENTRES], []) + [hs.W_BASE]
    for a, b in zip(edges[0::2], edges[1::2]):
        k = max(2, int(round(n_bare * (b - a) / (hs.PITCH - hs.T_FIN))))
        xs += list(a + (b - a) * (np.arange(k) + 0.5) / k); zs += [0.0] * k; ws += [(b - a) / k] * k; fin += [0.0] * k
    return np.array(xs), np.array(zs), np.array(ws), np.array(fin)


# ------------------------------------------------------- the training points
def _fin_index(x):
    return np.clip(np.floor(np.asarray(x) / hs.PITCH).astype(int), 0, hs.N_FINS - 1)


def sample_cases(n, rng):
    """n cases (P W, h W/(m²K)), uniform over the ranges."""
    return rng.uniform(P_MIN, P_MAX, n), rng.uniform(H_MIN, H_MAX, n)


def sample_interior(n, seed=81, fin_share=0.4, near_root=0.5, depth=1.5e-3):
    """n points (x, z) in the metal (m), with a case (P, h) each: a share
    ``fin_share`` in the fins, the rest in the base plate. In each, a share
    ``near_root`` lies close to the fin roots (z = 0), exponentially
    distributed in z with mean ``depth``: the field changes fastest there."""
    rng = np.random.default_rng(seed)
    nf = int(n * fin_share); nb = n - nf
    near = lambda k: rng.exponential(depth, k)
    kb = int(nb * near_root); kf = int(nf * near_root)
    xb = rng.uniform(0, hs.W_BASE, nb)
    zb = np.r_[rng.uniform(0, hs.T_BASE, nb - kb), np.clip(near(kb), 0, hs.T_BASE)]
    i = rng.integers(0, hs.N_FINS, nf)
    xf = FIN_CENTRES[i] + hs.T_FIN * (rng.random(nf) - 0.5)
    zf = np.r_[Z_BOTTOM * rng.random(nf - kf), -np.clip(near(kf), 0, hs.L_FIN)]
    P, h = sample_cases(n, rng)
    return np.r_[xb, xf], np.r_[zb, zf], P, h


def sample_roots(n, seed=84):
    """n points (x) across the fin roots (z = 0, under a fin), with a case (P, h):
    where a design that treats the base and the fins apart joins them."""
    rng = np.random.default_rng(seed)
    i = rng.integers(0, hs.N_FINS, n)
    P, h = sample_cases(n, rng)
    return FIN_CENTRES[i] + hs.T_FIN * (rng.random(n) - 0.5), P, h


def sample_air_faces(n, seed=82):
    """n points on the faces in the air, with their outward normal (nx, nz),
    the face's length weight (m of face per point, so a sum is an integral)
    and a case (P, h): fin sides, fin tips and the bare base between fins, in
    proportion to their length."""
    rng = np.random.default_rng(seed)
    L_side = 2 * hs.N_FINS * hs.L_FIN                           # m, both sides of every fin
    L_tip = hs.N_FINS * hs.T_FIN
    L_bare = hs.W_BASE - hs.N_FINS * hs.T_FIN
    L = L_side + L_tip + L_bare
    kind = rng.choice(3, n, p=[L_side / L, L_tip / L, L_bare / L])
    x = np.empty(n); z = np.empty(n); nx = np.zeros(n); nz = np.zeros(n)
    i = rng.integers(0, hs.N_FINS, n)
    side = np.where(rng.random(n) < 0.5, -1.0, 1.0)
    s = kind == 0                                                # fin sides
    x[s] = FIN_CENTRES[i[s]] + side[s] * hs.T_FIN / 2; z[s] = Z_BOTTOM * rng.random(s.sum()); nx[s] = side[s]
    s = kind == 1                                                # fin tips
    x[s] = FIN_CENTRES[i[s]] + hs.T_FIN * (rng.random(s.sum()) - 0.5); z[s] = Z_BOTTOM; nz[s] = -1.0
    s = kind == 2                                                # bare base: between the fins, and the half gaps at the ends
    x[s] = _bare_positions(rng.random(s.sum()) * L_bare); z[s] = 0.0; nz[s] = -1.0
    P, h = sample_cases(n, rng)
    return x, z, nx, nz, np.full(n, L / n), P, h


def _bare_positions(u):
    """Map u in (0, L_bare) to x on the bare base: the gaps between fins in order."""
    edges = [0.0]
    for c in FIN_CENTRES:
        edges += [c - hs.T_FIN / 2, c + hs.T_FIN / 2]
    edges.append(hs.W_BASE)
    gaps = [(edges[2 * j], edges[2 * j + 1]) for j in range(len(edges) // 2)]
    lens = np.array([b - a for a, b in gaps]); cum = np.r_[0, np.cumsum(lens)]
    j = np.clip(np.searchsorted(cum, u, side="right") - 1, 0, len(gaps) - 1)
    return np.array([gaps[k][0] for k in j]) + (u - cum[j])


def sample_top(n, seed=83, edges=0.1):
    """n points x (m) on the top face of the base, with a case (P, h); a share
    ``edges`` within 1.5 mm of the footprint's edges, where the flux steps."""
    rng = np.random.default_rng(seed)
    k = int(n * edges)
    x = np.r_[rng.uniform(0, hs.W_BASE, n - k),
              np.where(rng.random(k) < 0.5, hs.MODULE[0], hs.MODULE[1]) + 3e-3 * (rng.random(k) - 0.5)]
    P, h = sample_cases(n, rng)
    return np.clip(x, 0, hs.W_BASE), P, h


def truth_at(x_cells, z_cells, theta, dx, xq, zq):
    """The ground truth's rise (K) at points (xq, zq) (m) in the metal, by
    bilinear interpolation between the truth's cell centres (spacing dx).
    Used to score a coarser grid's cells against the truth."""
    from scipy.interpolate import RegularGridInterpolator
    xs = np.unique(np.round(x_cells / dx * 2).astype(int)) * dx / 2
    zs = np.unique(np.round(z_cells / dx * 2).astype(int)) * dx / 2
    F = np.zeros((zs.size, xs.size))
    F[np.searchsorted(zs, z_cells - 1e-12), np.searchsorted(xs, x_cells - 1e-12)] = theta
    return RegularGridInterpolator((zs, xs), F)(np.c_[zq, xq])


# ----------------------------------------------------------- the truth file
TRUTH_FILE = "MP8.1A_truth.npz"


def write_truth(path=TRUTH_FILE, dx=TRUTH_DX[2]):
    """The ground truth at the three test cases on cells of 0.25 mm (their
    hottest points within 0.01 K of the extrapolated values), with the
    convergence table at Ex08's case and every parameter with its unit."""
    rows, ext, order = convergence_study(P0, hs.H_AIR, TRUTH_DX, verbose=True)
    out = dict(cases=np.array(TESTS), dx=np.array(dx))
    m = Comb(dx)
    out["x"], out["z"] = m.X[m.metal], m.Z[m.metal]
    for P, h in TESTS:
        th = m.solve(P, h)
        key = f"{int(P)}W_{int(h)}h"
        out[f"theta_{key}"] = th
        out[f"hottest_{key}"] = np.array(m.hottest(th, P))
        out[f"heat_out_{key}"] = np.array(m.heat_out(th, h))
        # the hottest point on the finest grid and extrapolated, for the scoring of the hottest point
        mf = Comb(TRUTH_DX[-1]); hf = mf.hottest(mf.solve(P, h), P)
        out[f"hottest_fine_{key}"] = np.array(hf)
        out[f"hottest_extrapolated_{key}"] = np.array(hf + (hf - out[f"hottest_{key}"]) / (2 ** order - 1))
    out["conv_dx"] = np.array([r[0] for r in rows]); out["conv_cells"] = np.array([r[1] for r in rows])
    out["conv_hottest"] = np.array([r[2] for r in rows]); out["conv_heat_out"] = np.array([r[3] for r in rows])
    out["conv_balance"] = np.array([r[4] for r in rows])
    out["conv_hottest_extrapolated"] = np.array(ext); out["conv_order"] = np.array(order)
    err, hot_err = check_collapsed(verbose=False)
    out["collapsed_error_K"] = np.array(err)
    out["params"] = np.array([f"{k} = {v}" for k, v in dict(
        P_RANGE="100-200 W", H_RANGE="20-60 W/(m2K), ASSUMED (a slow and a fast fan)", K=f"{hs.K_AL} W/(m K)",
        BASE=f"{hs.W_BASE*1e3:.0f} x {hs.T_BASE*1e3:.0f} mm", FINS=f"{hs.N_FINS} of {hs.T_FIN*1e3:.0f} x {hs.L_FIN*1e3:.0f} mm, pitch {hs.PITCH*1e3:.0f} mm",
        MODULE=f"{hs.MODULE[0]*1e3:.0f}-{hs.MODULE[1]*1e3:.0f} mm", DEPTH=f"{hs.DEPTH*1e3:.0f} mm").items()])
    out["README"] = np.array(
        "MP8.1A ground truth. cases: (P W, h W/(m2K)) scored; x, z: centres of the metal cells (m), cells of dx m, z = 0 the fin "
        "side of the base, z < 0 the fins; theta_<P>W_<h>h: rise above the air in each cell, K; hottest_*: rise at x = W/2 on the "
        "module face on this grid, K; hottest_fine_* on 0.125 mm cells, hottest_extrapolated_* by Richardson (the score); "
        "conv_order: observed order of the hottest point (the fin roots' corners lower it from 2); heat_out_*: W through the air faces for 100 mm of depth; conv_*: the convergence study at 150 W, 40 W/(m2K); "
        "collapsed_error_K: the same solver with the fins collapsed against Ex08's exact series; params: the data.")
    np.savez_compressed(path, **out)
    print("written", path)


if __name__ == "__main__":
    print("Ex08's limit (fins collapsed)")
    check_collapsed()
    print("convergence at 150 W, h = 40 W/(m²K)")
    convergence_study()
    import sys
    if "--save" in sys.argv:
        write_truth()
