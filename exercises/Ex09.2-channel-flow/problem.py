r"""Ex_09.2 — the problem: the mean flow past a tube in a water duct.

*Deep Learning for Engineering* — MSc, Aalborg University.
Remus Teodorescu (ret@et.aau.dk), with support from Research Assistant
Noman Khan (nomank@energy.aau.dk).

**Reference texts.** Pope, *Turbulent Flows* (2000), ch. 4, 7 and 10; Liu,
*PINN with Python: An Introduction* (2025); Raissi, Perdikaris & Karniadakis,
*Physics-informed neural networks*, J. Comput. Phys. **378** (2019) 686–707.
These are the works to read for the theory. The code, the problem and the
exposition here are original to this course.

## The problem

Water runs through a duct 100 mm high at a mean speed of 0.25 m/s, past a tube
of 40 mm that crosses it. The Reynolds number is 25 000, so the flow is
turbulent, and what is solved for is its **mean**: the Reynolds-averaged
equations with the eddy-viscosity closure of L9.2,

    (u . grad) u = - grad p / rho + nu_eff lap u,     div u = 0,
    nu_eff = nu + nu_t

with ONE number for the eddy viscosity over the whole duct, nu_t = 0.02 U H -
five hundred times the viscosity of water. That number is a choice, made so
that the modelled mean flow is steady and smooth; a real closure varies in
space. Saying what the answer is worth with a closure like that is what the
exercise is about, and section 6 of the notebook finds the number from
measurements instead.

In scaled units - lengths over the duct height H, speeds over the mean speed
U, pressure over rho U^2 - the duct is 4 x 1, the tube a circle of radius 0.2
at (1.2, 0.5), the inflow 6 y (1 - y), and the only number left is
nu_eff / (U H) = 0.02.

## What is in here

The data; the samplers of the network's points; and the two classical
solvers, which are too long for a notebook cell:

    reference(spacing)     quadratic finite elements on a mesh FITTED to the
                           tube, the nonlinear system solved by Newton's
                           method. The reference: the problem has no exact
                           solution.
    finite_volumes(n_y)    a staggered grid of square cells, the tube as a
                           STAIRCASE, marched in time to the steady state.
                           The classical method the network is compared with.

There is no network here: the trial stream function and the residuals are
written in the notebook.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "H_DUCT", "U_MEAN", "RHO", "NU_WATER", "NU_EFF", "NU_EFF_SI", "RE_DUCT",
    "P_SCALE", "F_SCALE", "CHANNEL", "TUBE", "PROBES", "NOISE",
    "inflow", "psi_inflow", "sample_fluid", "sample_tube", "sample_outlet",
    "reference", "reference_at", "finite_volumes", "mean_over_height",
    "network_numbers", "probe_readings", "drag_coefficient", "describe_problem",
]

# ----------------------------------------------------------------- the data
H_DUCT = 0.10            #: m, the height of the duct
U_MEAN = 0.25            #: m/s, the mean speed of the water
RHO = 998.0              #: kg/m^3, water at 20 degC
NU_WATER = 1.0e-6        #: m^2/s, water at 20 degC
RE_DUCT = U_MEAN * H_DUCT / NU_WATER          #: 25 000: turbulent

#: The effective viscosity nu + nu_t in scaled units, nu_eff / (U H). The
#: closure of this set: one number for the whole duct.
NU_EFF = 0.02
NU_EFF_SI = NU_EFF * U_MEAN * H_DUCT          #: m^2/s, 5.0e-04

P_SCALE = RHO * U_MEAN ** 2                   #: Pa, the pressure scale, 62.38
F_SCALE = RHO * U_MEAN ** 2 * H_DUCT          #: N per metre of tube, the force scale, 6.24

#: The duct and the tube in scaled units.
CHANNEL = dict(L=4.0, H=1.0)
TUBE = dict(xc=1.2, yc=0.5, r=0.2)

#: Where section 6's twelve velocity probes sit, in the wake of the tube, and
#: the noise of their readings in units of U.
PROBES = np.array([[x, y] for x in (1.6, 2.0, 2.4, 2.8) for y in (0.3, 0.5, 0.7)])
NOISE = 0.01


def inflow(y):
    """The inflow profile, 6 y (1 - y): mean 1, zero on both walls. The
    developed profile of a duct with a uniform viscosity."""
    return 6.0 * y * (1.0 - y)


def psi_inflow(y):
    """The stream function of the inflow, 3 y^2 - 2 y^3: its y-derivative is
    :func:`inflow`, it is 0 on the lower wall and 1 on the upper."""
    return 3.0 * y ** 2 - 2.0 * y ** 3


def drag_coefficient(force):
    """A scaled force per unit depth as a drag coefficient on the tube's
    diameter and the mean speed: F / (1/2 rho U^2 D)."""
    return force / (0.5 * 2.0 * TUBE["r"])


def _outside(p, margin=0.0):
    return np.hypot(p[:, 0] - TUBE["xc"], p[:, 1] - TUBE["yc"]) > TUBE["r"] + margin


# ------------------------------------------------------------------- samplers
def sample_fluid(n, seed=88):
    """``n`` collocation points in the water: half spread over the whole duct,
    half over the stretch round the tube and its wake (x from 0.7 to 2.2),
    where the flow bends. Points inside the tube are rejected."""
    rng = np.random.default_rng(seed)

    def draw(m, x0, x1):
        out = np.empty((0, 2))
        while len(out) < m:
            p = np.c_[rng.uniform(x0, x1, 2 * m), rng.uniform(0.0, CHANNEL["H"], 2 * m)]
            out = np.vstack([out, p[_outside(p)]])
        return out[:m]

    return np.vstack([draw(n // 2, 0.0, CHANNEL["L"]), draw(n - n // 2, 0.7, 2.2)])


def sample_tube(n):
    """``n`` points on the surface of the tube, evenly spaced."""
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.c_[TUBE["xc"] + TUBE["r"] * np.cos(t), TUBE["yc"] + TUBE["r"] * np.sin(t)]


def sample_outlet(n):
    """``n`` points on the outlet, x = L, evenly spaced from wall to wall."""
    return np.c_[np.full(n, CHANNEL["L"]), np.linspace(0.0, CHANNEL["H"], n)]


def mean_over_height(y, f):
    """The mean of ``f`` over the height of the duct, by the trapezoidal rule."""
    k = np.argsort(y)
    y, f = np.asarray(y)[k], np.asarray(f)[k]
    return float(np.sum(0.5 * (f[1:] + f[:-1]) * np.diff(y)) / (y[-1] - y[0]))


# ------------------------------------------------------------- the reference
# A 7-point quadrature rule on the triangle, exact to degree 5.
_A1, _B1 = 0.0597158717897698, 0.4701420641051151
_A2, _B2 = 0.7974269853530873, 0.1012865073234563
_QP = np.array([[1 / 3, 1 / 3, 1 / 3],
                [_A1, _B1, _B1], [_B1, _A1, _B1], [_B1, _B1, _A1],
                [_A2, _B2, _B2], [_B2, _A2, _B2], [_B2, _B2, _A2]])
_QW = np.array([0.225] + [0.1323941527885062] * 3 + [0.1259391805448271] * 3)


def _mesh(spacing, fine=4.0, growth=1.35):
    """Vertices and triangles: a square grid of the given spacing, and rings
    of nodes round the tube that start ``fine`` times finer on its surface and
    grow outwards. A Delaunay triangulation joins them."""
    from scipy.spatial import Delaunay
    nx, ny = int(round(CHANNEL["L"] / spacing)), int(round(CHANNEL["H"] / spacing))
    X, Y = np.meshgrid(np.linspace(0, CHANNEL["L"], nx + 1), np.linspace(0, CHANNEL["H"], ny + 1))
    rings, rad, s = [], TUBE["r"], spacing / fine
    while s < spacing:
        n = int(np.ceil(2 * np.pi * rad / s))
        t = np.linspace(0, 2 * np.pi, n, endpoint=False) + (0.5 * np.pi / n) * len(rings)
        rings.append(np.c_[TUBE["xc"] + rad * np.cos(t), TUBE["yc"] + rad * np.sin(t)])
        rad += 0.9 * s
        s *= growth
    P = np.c_[X.ravel(), Y.ravel()]
    P = np.vstack([P[_outside(P, rad - TUBE["r"] - 0.9 * s / growth + 0.7 * spacing)]] + rings)
    tri = Delaunay(P).simplices
    tri = tri[_outside(P[tri].mean(axis=1))]
    x, y = P[tri, 0], P[tri, 1]
    det = (x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0]) - (x[:, 2] - x[:, 0]) * (y[:, 1] - y[:, 0])
    tri, det = tri[np.abs(det) > 1e-14], det[np.abs(det) > 1e-14]
    tri[det < 0] = tri[det < 0][:, [0, 2, 1]]                # counter-clockwise
    return P, tri


def reference(spacing=1 / 40, nu=NU_EFF, tol=1e-10):
    """The reference solution: Taylor-Hood finite elements (quadratic velocity,
    linear pressure) on a mesh fitted to the tube, the steady equations solved
    by Newton's method from rest.

    The inflow, the walls and the tube are imposed at the nodes; the outlet is
    free (zero traction, the natural condition of the method). Returns a dict:
    the nodes ``P`` (vertices first, then edge midpoints), the triangles, the
    nodal ``u``, ``v`` and ``p``, the force on the tube per unit depth
    (``drag``, ``lift``), the pressure drop ``dp`` from inlet to outlet, and
    the number of unknowns - everything in scaled units.

    Its accuracy is shown in the notebook, by refining ``spacing``. Measured
    once beyond what the notebook runs: 1/40 is within 0.004 U of 1/80 in
    velocity everywhere, and its drag and pressure drop within 0.01 %.
    """
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    P1, tri = _mesh(spacing)
    # the edge midpoints: the extra nodes of the quadratic element
    e = np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    uniq, inv = np.unique(np.sort(e, axis=1), axis=0, return_inverse=True)
    inv = inv.ravel()
    nt, n1 = len(tri), len(P1)
    P = np.vstack([P1, 0.5 * (P1[uniq[:, 0]] + P1[uniq[:, 1]])])
    t6 = np.c_[tri, n1 + inv[:nt], n1 + inv[nt:2 * nt], n1 + inv[2 * nt:]]
    n2 = len(P)

    x, y = P1[tri, 0], P1[tri, 1]
    area = 0.5 * ((x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0]) - (x[:, 2] - x[:, 0]) * (y[:, 1] - y[:, 0]))
    b = np.stack([y[:, 1] - y[:, 2], y[:, 2] - y[:, 0], y[:, 0] - y[:, 1]], 1) / (2 * area[:, None])
    c = np.stack([x[:, 2] - x[:, 1], x[:, 0] - x[:, 2], x[:, 1] - x[:, 0]], 1) / (2 * area[:, None])
    lam = _QP
    N = np.c_[lam * (2 * lam - 1), 4 * lam[:, 0] * lam[:, 1], 4 * lam[:, 1] * lam[:, 2], 4 * lam[:, 2] * lam[:, 0]]

    def dN(g):
        out = np.empty((nt, len(_QW), 6))
        for i in range(3):
            out[:, :, i] = (4 * lam[None, :, i] - 1) * g[:, None, i]
        for k, (i, j) in enumerate(((0, 1), (1, 2), (2, 0))):
            out[:, :, 3 + k] = 4 * (lam[None, :, i] * g[:, None, j] + lam[None, :, j] * g[:, None, i])
        return out

    Nx, Ny = dN(b), dN(c)
    w = area[:, None] * _QW[None, :]
    I6, J6 = np.repeat(t6, 6, axis=1).ravel(), np.tile(t6, (1, 6)).ravel()

    def mat66(Ke):
        return sp.coo_matrix((Ke.ravel(), (I6, J6)), shape=(n2, n2)).tocsr()

    K = mat66(np.einsum("eq,eqi,eqj->eij", w, Nx, Nx) + np.einsum("eq,eqi,eqj->eij", w, Ny, Ny))
    I63, J63 = np.repeat(t6, 3, axis=1).ravel(), np.tile(tri, (1, 6)).ravel()
    Bx = sp.coo_matrix((np.einsum("eq,eqi,qk->eik", w, Nx, lam).ravel(), (I63, J63)), shape=(n2, n1)).tocsr()
    By = sp.coo_matrix((np.einsum("eq,eqi,qk->eik", w, Ny, lam).ravel(), (I63, J63)), shape=(n2, n1)).tocsr()

    # where the velocity is imposed: the inlet, the two walls, the tube
    eps = 1e-9
    on_in = P[:, 0] < eps
    on_wall = (P[:, 1] < eps) | (P[:, 1] > CHANNEL["H"] - eps)
    ring = np.abs(np.hypot(P1[:, 0] - TUBE["xc"], P1[:, 1] - TUBE["yc"]) - TUBE["r"]) < 1e-9
    e3 = np.vstack([t6[:, [0, 1, 3]], t6[:, [1, 2, 4]], t6[:, [2, 0, 5]]])
    mid = e3[ring[e3[:, 0]] & ring[e3[:, 1]], 2]
    mid = np.unique(mid[np.bincount(e3[:, 2], minlength=n2)[mid] == 1])   # edges ON the tube belong to one triangle
    on_tube = np.zeros(n2, bool)
    on_tube[:n1] = ring
    on_tube[mid] = True
    fixed = on_in | on_wall | on_tube
    u, v, p = np.zeros(n2), np.zeros(n2), np.zeros(n1)
    u[on_in] = inflow(P[on_in, 1])
    u[on_wall | on_tube] = 0.0
    free_u = np.where(~fixed)[0]
    free = np.r_[free_u, n2 + free_u, 2 * n2 + np.arange(n1)]

    def system(u, v, p, jacobian=True):
        ue, ve = u[t6], v[t6]
        uq, vq = np.einsum("qi,ei->eq", N, ue), np.einsum("qi,ei->eq", N, ve)
        A = nu * K + mat66(np.einsum("eq,qi,eqj->eij", w * uq, N, Nx) + np.einsum("eq,qi,eqj->eij", w * vq, N, Ny))
        R = np.r_[A @ u - Bx @ p, A @ v - By @ p, -(Bx.T @ u + By.T @ v)]
        if not jacobian:
            return R, None
        ux, uy = np.einsum("eqi,ei->eq", Nx, ue), np.einsum("eqi,ei->eq", Ny, ue)
        vx, vy = np.einsum("eqi,ei->eq", Nx, ve), np.einsum("eqi,ei->eq", Ny, ve)
        M = lambda f: mat66(np.einsum("eq,qi,qj->eij", w * f, N, N))
        J = sp.bmat([[A + M(ux), M(uy), -Bx], [M(vx), A + M(vy), -By], [-Bx.T, -By.T, None]], format="csr")
        return R, J

    for it in range(25):                                     # Newton's method
        R, J = system(u, v, p)
        if np.abs(R[free]).max() < tol:
            break
        d = np.zeros(2 * n2 + n1)
        d[free] = spla.spsolve(J[free][:, free].tocsc(), -R[free])
        u += d[:n2]
        v += d[n2:2 * n2]
        p += d[2 * n2:]
    R, _ = system(u, v, p, jacobian=False)                   # the reaction at the tube's nodes is the force on it
    ref = dict(P=P, P1=P1, tri=tri, t6=t6, u=u, v=v, p=p, nu=nu, newton=it, n_unknowns=len(free),
               drag=float(-R[:n2][on_tube].sum()), lift=float(-R[n2:2 * n2][on_tube].sum()))
    k_in, k_out = np.where(P1[:, 0] < eps)[0], np.where(P1[:, 0] > CHANNEL["L"] - eps)[0]
    ref["dp"] = mean_over_height(P1[k_in, 1], p[k_in]) - mean_over_height(P1[k_out, 1], p[k_out])
    return ref


def reference_at(ref, x, y):
    """The reference's ``(u, v, p)`` at the points ``(x, y)``, with the shape
    functions of the triangle that holds each point (quadratic for the
    velocity, linear for the pressure). NaN inside the tube."""
    from matplotlib.tri import Triangulation
    if "_finder" not in ref:
        ref["_finder"] = Triangulation(ref["P1"][:, 0], ref["P1"][:, 1], ref["tri"]).get_trifinder()
    x, y = np.asarray(x, float), np.asarray(y, float)
    shape = x.shape
    x, y = x.ravel(), y.ravel()
    e = ref["_finder"](x, y)
    ok = e >= 0
    tri, t6, P1 = ref["tri"][e[ok]], ref["t6"][e[ok]], ref["P1"]
    x0, y0 = P1[tri[:, 0], 0], P1[tri[:, 0], 1]
    x1, y1 = P1[tri[:, 1], 0], P1[tri[:, 1], 1]
    x2, y2 = P1[tri[:, 2], 0], P1[tri[:, 2], 1]
    det = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    l1 = ((x[ok] - x0) * (y2 - y0) - (x2 - x0) * (y[ok] - y0)) / det
    l2 = ((x1 - x0) * (y[ok] - y0) - (x[ok] - x0) * (y1 - y0)) / det
    lam = np.c_[1 - l1 - l2, l1, l2]
    N = np.c_[lam * (2 * lam - 1), 4 * lam[:, 0] * lam[:, 1], 4 * lam[:, 1] * lam[:, 2], 4 * lam[:, 2] * lam[:, 0]]
    out = []
    for vals, shp, conn in ((ref["u"], N, t6), (ref["v"], N, t6), (ref["p"], lam, tri)):
        f = np.full(len(x), np.nan)
        f[ok] = (shp * vals[conn]).sum(axis=1)
        out.append(f.reshape(shape))
    return out


# ------------------------------------------------- the classical method: a grid
def finite_volumes(n_y, nu=NU_EFF, tol=1e-6, t_max=40.0):
    """The mean flow on a staggered grid of square cells, ``n_y`` across the
    duct: ``u`` on the vertical faces, ``v`` on the horizontal ones, ``p`` at
    the centres. Central differences. The tube is a STAIRCASE: the velocity is
    set to zero on every face inside it. Marched in time to the steady state
    by a projection step - an explicit step of convection and viscosity, then
    a pressure that makes the result divergence-free, its matrix factorised
    once.

    Returns a dict: the face coordinates, ``u``, ``v``, ``p``, the masks of the
    faces inside the tube, the force on the tube per unit depth (``drag``, the
    momentum the staircase removes), the pressure drop ``dp``, and the time
    ``t`` at which the flow stopped changing - in scaled units."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    h = CHANNEL["H"] / n_y
    nx, ny = int(round(CHANNEL["L"] / h)), n_y
    xu, yu = np.arange(nx + 1) * h, (np.arange(ny) + 0.5) * h
    xv, yv = (np.arange(nx) + 0.5) * h, np.arange(ny + 1) * h
    XU, YU = np.meshgrid(xu, yu)
    XV, YV = np.meshgrid(xv, yv)
    inside = lambda X, Y: (X - TUBE["xc"]) ** 2 + (Y - TUBE["yc"]) ** 2 < TUBE["r"] ** 2
    su, sv = inside(XU, YU), inside(XV, YV)
    u_in = inflow(yu)
    # the pressure matrix: a five-point Laplacian, zero gradient on the walls and the inlet, p = 0 on the outlet
    idx = np.arange(nx * ny).reshape(ny, nx)
    rows, cols, diag = [], [], np.zeros((ny, nx))
    for a, b in ((idx[:, :-1], idx[:, 1:]), (idx[:-1, :], idx[1:, :])):
        rows += [a.ravel(), b.ravel()]
        cols += [b.ravel(), a.ravel()]
    diag[:, :-1] -= 1; diag[:, 1:] -= 1; diag[:-1, :] -= 1; diag[1:, :] -= 1; diag[:, -1] -= 2
    r, c = np.concatenate(rows), np.concatenate(cols)
    A = sp.coo_matrix((np.ones(len(r)), (r, c)), shape=(nx * ny, nx * ny)) + sp.diags(diag.ravel())
    lu = spla.splu((A / h ** 2).tocsc())
    dt = min(0.2 * h / 2.5, 0.2 * h * h / nu)                # the stable step: convection and viscosity
    u = np.tile(u_in[:, None], (1, nx + 1))
    u[su] = 0.0
    v = np.zeros((ny + 1, nx))
    t = 0.0
    while t < t_max:
        # ghost values: no slip on the walls, the inflow, zero gradient at the outlet
        U = np.zeros((ny + 2, nx + 3)); U[1:-1, 1:-1] = u
        U[1:-1, 0] = 2 * u_in - u[:, 1]; U[1:-1, -1] = u[:, -1]
        U[0, :] = -U[1, :]; U[-1, :] = -U[-2, :]
        V = np.zeros((ny + 3, nx + 2)); V[1:-1, 1:-1] = v
        V[1:-1, 0] = -v[:, 0]; V[1:-1, -1] = v[:, -1]
        uc, vc = U[1:-1, 1:-1], V[1:-1, 1:-1]
        vbar = 0.25 * (V[1:-2, 1:] + V[1:-2, :-1] + V[2:-1, 1:] + V[2:-1, :-1])       # v at the u faces
        ubar = 0.25 * (U[1:, 1:-2] + U[1:, 2:-1] + U[:-1, 1:-2] + U[:-1, 2:-1])       # u at the v faces
        conv_u = uc * (U[1:-1, 2:] - U[1:-1, :-2]) / (2 * h) + vbar * (U[2:, 1:-1] - U[:-2, 1:-1]) / (2 * h)
        conv_v = ubar * (V[1:-1, 2:] - V[1:-1, :-2]) / (2 * h) + vc * (V[2:, 1:-1] - V[:-2, 1:-1]) / (2 * h)
        lap_u = (U[1:-1, 2:] + U[1:-1, :-2] + U[2:, 1:-1] + U[:-2, 1:-1] - 4 * uc) / h ** 2
        lap_v = (V[1:-1, 2:] + V[1:-1, :-2] + V[2:, 1:-1] + V[:-2, 1:-1] - 4 * vc) / h ** 2
        us, vs = uc + dt * (nu * lap_u - conv_u), vc + dt * (nu * lap_v - conv_v)
        us[:, 0] = u_in; vs[0, :] = 0.0; vs[-1, :] = 0.0
        drag = us[su].sum() * h * h / dt                     # the momentum the tube's faces take out
        us[su] = 0.0; vs[sv] = 0.0
        div = ((us[:, 1:] - us[:, :-1]) + (vs[1:, :] - vs[:-1, :])) / h
        p = lu.solve((div / dt).ravel()).reshape(ny, nx)     # the pressure that removes the divergence
        pe = np.hstack([p, -p[:, -1:]])
        un, vn = us.copy(), vs.copy()
        un[:, 1:] -= dt * (pe[:, 1:] - pe[:, :-1]) / h
        vn[1:-1, :] -= dt * (p[1:, :] - p[:-1, :]) / h
        drag += un[su].sum() * h * h / dt
        un[su] = 0.0; vn[sv] = 0.0
        change = max(np.abs(un - u).max(), np.abs(vn - v).max()) / dt
        u, v, t = un, vn, t + dt
        if change < tol:
            break
    return dict(xu=xu, yu=yu, xv=xv, yv=yv, u=u, v=v, p=p, in_tube_u=su, in_tube_v=sv, h=h, t=t,
                drag=float(drag), dp=float(p[:, 0].mean() - p[:, -1].mean()))


# ------------------------------------------------------ numbers from a network
def network_numbers(fields, nu=NU_EFF, n=400):
    """The two engineering numbers from a network. ``fields`` maps points to
    ``(u, v, p)``. Returns the force on the tube per unit depth - the traction
    ``-p n + nu (grad u + grad u^T) n`` averaged over ``n`` surface points,
    times the circumference - and the pressure drop, the mean pressure over
    the inlet minus the mean over the outlet. Scaled units."""
    from course_core import to_numpy, to_tensor
    from pinn_core import grad
    xy = to_tensor(sample_tube(n), requires_grad=True)
    u, v, p = fields(xy)
    gu, gv = grad(u, xy), grad(v, xy)
    nx = (xy[:, 0:1] - TUBE["xc"]) / TUBE["r"]               # the normal, out of the tube
    ny = (xy[:, 1:2] - TUBE["yc"]) / TUBE["r"]
    tx = -p * nx + nu * (2 * gu[:, 0:1] * nx + (gu[:, 1:2] + gv[:, 0:1]) * ny)
    drag = float(tx.mean().item()) * 2.0 * np.pi * TUBE["r"]
    y = np.linspace(0.0, CHANNEL["H"], 201)
    ends = []
    for x in (0.0, CHANNEL["L"]):
        q = to_tensor(np.c_[np.full(len(y), x), y], requires_grad=True)
        ends.append(mean_over_height(y, to_numpy(fields(q)[2]).ravel()))
    return drag, ends[0] - ends[1]


def probe_readings(ref, seed=92):
    """What twelve velocity probes in the wake would read: the reference's
    ``u`` at :data:`PROBES`, with noise of :data:`NOISE` (1 % of the mean
    speed). Synthetic measurements: the reference, sampled."""
    rng = np.random.default_rng(seed)
    return reference_at(ref, PROBES[:, 0], PROBES[:, 1])[0] + rng.normal(0.0, NOISE, len(PROBES))


def describe_problem() -> None:
    """Print the duct and the numbers it implies."""
    D = 2 * TUBE["r"] * H_DUCT
    print(f"  duct             : {H_DUCT * 1e3:.0f} mm high, {CHANNEL['L'] * H_DUCT * 1e3:.0f} mm long; water at {U_MEAN:.2f} m/s (mean)")
    print(f"  tube             : {D * 1e3:.0f} mm across the duct, its centre {TUBE['xc'] * H_DUCT * 1e3:.0f} mm from the inlet, on the centreline")
    print(f"  Reynolds number  : U H / nu = {RE_DUCT:.0f}  ->  turbulent; the mean flow is solved for")
    print(f"  closure          : nu_eff = nu + nu_t = {NU_EFF:.2f} U H = {NU_EFF_SI:.1e} m^2/s  ({NU_EFF_SI / NU_WATER:.0f} times water's)")
    print(f"  the model's Re   : U H / nu_eff = {1 / NU_EFF:.0f}, and {2 * TUBE['r'] / NU_EFF:.0f} on the tube's diameter")
    print(f"  pressure scale   : rho U^2 = {P_SCALE:.2f} Pa        force scale: rho U^2 H = {F_SCALE:.2f} N per metre of tube")
