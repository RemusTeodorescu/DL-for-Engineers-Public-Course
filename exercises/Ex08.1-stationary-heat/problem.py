r"""Ex_08.1 — the physics: steady conduction in a plate with a cooling hole.

*Deep Learning for Engineering* — MSc, Aalborg University.
Remus Teodorescu (ret@et.aau.dk), with support from Research Assistant
Noman Khan (nomank@energy.aau.dk).

**Reference texts.** Liu, *PINN with Python: An Introduction* (2025); Raissi,
Perdikaris & Karniadakis, *Physics-informed neural networks*, J. Comput. Phys.
**378** (2019) 686–707. These are the works to read for the theory. The code,
the problem and the exposition here are original to this course.

## The problem

A square aluminium plate generates heat evenly and is cooled through an
elliptical channel at its centre. Its outer edges are insulated.

    k (T_xx + T_yy) + Q = 0     in the plate
    T = T_coolant               on the wall of the channel
    dT/dn = 0                   on the four outer edges

## Scaled units

Lengths are divided by the plate's side ``L`` and the temperature rise above
the coolant by ``DELTA_T = Q L^2 / k``. The scaled rise ``theta`` then obeys

    theta_xx + theta_yy + 1 = 0,   theta = 0 on the channel,   d(theta)/dn = 0 on the edges

on the unit square, and ``T = T_COOLANT + DELTA_T * theta``. Everything in this
module works in the scaled units; :func:`kelvin` converts a rise back.

## What this module gives the notebook

* the data of the plate, as constants;
* the geometry: :func:`ellipse_phi` (the level set, also the boundary
  multiplier), :func:`ellipse_normal`, and the three samplers;
* :func:`reference` — the reference solution, by linear finite elements on a
  mesh fitted to the channel, and :func:`reference_at` to read it at any point;
* :func:`heat_out` — the heat a network sends through the channel wall, for the
  energy balance.

The samplers return NumPy, like the ones in ``pinn_core``: call ``to_tensor``
at the point of use, with ``requires_grad=True`` where the network is
differentiated. Nothing at module level imports torch.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "L_PLATE", "K_PLATE", "Q_PLATE", "T_COOLANT", "DELTA_T", "HOLE", "DOMAIN",
    "kelvin", "ellipse_phi", "hole_multiplier", "ellipse_normal",
    "sample_plate_with_hole", "sample_ellipse_boundary", "sample_outer_edges",
    "hole_perimeter", "plate_area", "reference", "reference_at", "heat_out",
    "describe_problem",
]

# ------------------------------------------------------------------ the data
L_PLATE = 0.100          #: m, the side of the plate
K_PLATE = 167.0          #: W/(m·K), aluminium alloy 6061
Q_PLATE = 1.0e6          #: W/m^3, the heat generated in the plate
T_COOLANT = 40.0         #: degC, the wall of the channel
DELTA_T = Q_PLATE * L_PLATE ** 2 / K_PLATE   #: K, the temperature scale, 59.88

#: The cooling channel in scaled units: an ellipse of 36 x 22 mm at the centre.
HOLE = dict(xc=0.5, yc=0.5, a=0.18, b=0.11)

#: ``((x_lo, x_hi), (y_lo, y_hi))`` — the plate in scaled units, before the hole.
DOMAIN = ((0.0, 1.0), (0.0, 1.0))


def kelvin(theta):
    """A scaled rise as kelvin above the coolant."""
    return DELTA_T * theta


# ------------------------------------------------------------------- geometry
def _phi_np(xy, xc, yc, a, b):
    """The level set on a flat ``(N, 2)`` NumPy array. Used by the samplers."""
    return ((xy[:, 0] - xc) / a) ** 2 + ((xy[:, 1] - yc) / b) ** 2 - 1.0


def ellipse_phi(xy, hole=None):
    """Level set of the channel: negative inside it, zero on its wall, positive
    in the plate. Shape ``(N, 1)``. Plain arithmetic, so NumPy in gives NumPy
    out and a tensor in gives a tensor out."""
    h = hole or HOLE
    return (((xy[:, 0:1] - h["xc"]) / h["a"]) ** 2
            + ((xy[:, 1:2] - h["yc"]) / h["b"]) ** 2 - 1.0)


def hole_multiplier(xy, hole=None):
    """The boundary multiplier of L8.1: zero on the channel wall, positive in
    the plate. For a shape with a formula it is the level set itself."""
    return ellipse_phi(xy, hole)


def ellipse_normal(xy, hole=None):
    """Unit normal on the channel wall, pointing OUT of the plate (into the
    channel). Torch. Returns ``(nx, ny)``, each shaped ``(N, 1)``."""
    import torch
    h = hole or HOLE
    gx = 2.0 * (xy[:, 0:1] - h["xc"]) / h["a"] ** 2
    gy = 2.0 * (xy[:, 1:2] - h["yc"]) / h["b"] ** 2
    norm = torch.sqrt(gx ** 2 + gy ** 2)
    return -gx / norm, -gy / norm


# ------------------------------------------------------------------- samplers
def sample_plate_with_hole(n, domain=DOMAIN, hole=None, margin=1.05, seed=0):
    """``n`` points uniform in the plate, by rejection: drawn over the square,
    kept where the level set (inflated by ``margin``) is positive."""
    h = hole or HOLE
    rng = np.random.default_rng(seed)
    (x0, x1), (y0, y1) = domain
    out = np.empty((0, 2))
    while len(out) < n:
        p = np.c_[rng.uniform(x0, x1, 2 * n), rng.uniform(y0, y1, 2 * n)]
        keep = _phi_np(p, h["xc"], h["yc"], h["a"] * margin, h["b"] * margin) > 0
        out = np.vstack([out, p[keep]])
    return out[:n]


def sample_ellipse_boundary(n, hole=None):
    """``n`` points on the channel wall at equal steps of ARC LENGTH. Equal
    steps of angle would crowd the flat sides and starve the ends, where the
    wall bends most and the flux is largest."""
    h = hole or HOLE
    t = np.linspace(0.0, 2.0 * np.pi, 20001)
    arc = np.r_[0.0, np.cumsum(np.hypot(np.diff(h["a"] * np.cos(t)),
                                        np.diff(h["b"] * np.sin(t))))]
    th = np.interp(np.linspace(0.0, arc[-1], n, endpoint=False), arc, t)
    return np.c_[h["xc"] + h["a"] * np.cos(th), h["yc"] + h["b"] * np.sin(th)]


def sample_outer_edges(n_per_edge, domain=DOMAIN):
    """Points on the four insulated edges, evenly spaced. Returns a list of four
    ``(points, axis)`` pairs: ``axis`` is 0 where the normal is along x (the
    left and right edges) and 1 where it is along y."""
    (x0, x1), (y0, y1) = domain
    s = np.linspace(0.0, 1.0, n_per_edge)
    xs, ys = x0 + (x1 - x0) * s, y0 + (y1 - y0) * s
    return [(np.c_[np.full(n_per_edge, x0), ys], 0), (np.c_[np.full(n_per_edge, x1), ys], 0),
            (np.c_[xs, np.full(n_per_edge, y0)], 1), (np.c_[xs, np.full(n_per_edge, y1)], 1)]


def hole_perimeter(hole=None) -> float:
    """Ramanujan's approximation to the perimeter of the channel wall."""
    h = hole or HOLE
    return float(np.pi * (3 * (h["a"] + h["b"])
                          - np.sqrt((3 * h["a"] + h["b"]) * (h["a"] + 3 * h["b"]))))


def plate_area(hole=None) -> float:
    """Area of the unit plate with the channel removed: 0.9378."""
    h = hole or HOLE
    return float(1.0 - np.pi * h["a"] * h["b"])


# ------------------------------------------------------------- the reference
def reference(spacing=1 / 200, hole=None):
    """The reference solution, by linear finite elements.

    The mesh is FITTED to the channel: nodes sit on its wall at equal arc
    length, a regular grid of the same spacing fills the plate, and a Delaunay
    triangulation joins them, with the triangles inside the channel removed.
    The wall nodes are held at ``theta = 0``; the insulated edges need nothing
    (zero flux is the natural condition of the method).

    Returns a dict: ``P`` the nodes, ``tri`` the triangles, ``theta`` the
    scaled rise at the nodes, ``heat_out`` the heat through the channel wall
    and ``generated`` the heat generated, both in scaled units.

    Its accuracy is shown in the notebook, by refining ``spacing``: 1/200
    (38 009 nodes) is within 0.002 K of 1/400 everywhere.
    """
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    from scipy.spatial import Delaunay

    h = hole or HOLE
    g = np.linspace(0.0, 1.0, int(round(1.0 / spacing)) + 1)
    X, Y = np.meshgrid(g, g)
    P = np.c_[X.ravel(), Y.ravel()]
    # clear a band of half a spacing round the channel, so wall nodes are not crowded
    P = P[_phi_np(P, h["xc"], h["yc"], h["a"] + 0.5 * spacing, h["b"] + 0.5 * spacing) > 0]
    n_in = len(P)
    P = np.vstack([P, sample_ellipse_boundary(int(np.ceil(hole_perimeter(h) / spacing)), h)])
    tri = Delaunay(P).simplices
    tri = tri[_phi_np(P[tri].mean(axis=1), h["xc"], h["yc"], h["a"], h["b"]) > 0]
    x, y = P[tri, 0], P[tri, 1]
    area = 0.5 * np.abs((x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0])
                        - (x[:, 2] - x[:, 0]) * (y[:, 1] - y[:, 0]))
    ok = area > 1e-14
    tri, x, y, area = tri[ok], x[ok], y[ok], area[ok]
    # the gradients of the three hat functions on each triangle
    b = np.stack([y[:, 1] - y[:, 2], y[:, 2] - y[:, 0], y[:, 0] - y[:, 1]], 1) / (2 * area[:, None])
    c = np.stack([x[:, 2] - x[:, 1], x[:, 0] - x[:, 2], x[:, 1] - x[:, 0]], 1) / (2 * area[:, None])
    Ke = area[:, None, None] * (b[:, :, None] * b[:, None, :] + c[:, :, None] * c[:, None, :])
    n = len(P)
    K = sp.coo_matrix((Ke.ravel(), (np.repeat(tri, 3, axis=1).ravel(),
                                    np.tile(tri, (1, 3)).ravel())), shape=(n, n)).tocsr()
    f = np.zeros(n)
    np.add.at(f, tri.ravel(), np.repeat(area / 3.0, 3))          # the source, 1 per unit area
    free = np.arange(n_in)
    theta = np.zeros(n)
    theta[free] = spla.spsolve(K[free][:, free].tocsc(), f[free])
    out = float(-(K[n_in:] @ theta - f[n_in:]).sum())            # the reaction at the wall nodes
    return dict(P=P, tri=tri, theta=theta, heat_out=out, generated=float(area.sum()))


def reference_at(ref, x, y):
    """The reference's scaled rise at the points ``(x, y)``, by linear
    interpolation on its triangles. NaN inside the channel."""
    from matplotlib.tri import LinearTriInterpolator, Triangulation
    P = ref["P"]
    f = LinearTriInterpolator(Triangulation(P[:, 0], P[:, 1], ref["tri"]), ref["theta"])
    return f(np.asarray(x), np.asarray(y)).filled(np.nan)


# ---------------------------------------------------------- the energy balance
def heat_out(trial, n=400, hole=None):
    """The heat a network sends through the channel wall, in scaled units:
    Fourier's flux along the outward normal, its mean over ``n`` wall points
    at equal arc length, times the perimeter. ``trial`` maps points to the
    scaled rise. In steady state it must equal :func:`plate_area`, the heat
    generated."""
    from course_core import to_tensor
    from pinn_core import grad
    h = hole or HOLE
    xy = to_tensor(sample_ellipse_boundary(n, h), requires_grad=True)
    g = grad(trial(xy), xy)                      # the temperature gradient on the wall
    nx, ny = ellipse_normal(xy, h)
    q_n = -(g[:, 0:1] * nx + g[:, 1:2] * ny)     # Fourier's flux, out of the plate
    return float(q_n.mean().item()) * hole_perimeter(h)


def describe_problem() -> None:
    """Print the plate and the numbers it implies."""
    h = HOLE
    L = L_PLATE * 1e3
    print(f"  plate            : {L:.0f} x {L:.0f} mm aluminium, k = {K_PLATE:.0f} W/(m·K)")
    print(f"  heat generated   : Q = {Q_PLATE:.1e} W/m^3, evenly")
    print(f"  cooling channel  : ellipse {2 * h['a'] * L:.0f} x {2 * h['b'] * L:.0f} mm at the centre, "
          f"wall at {T_COOLANT:.0f} degC")
    print("  outer edges      : insulated")
    print(f"  temperature scale: Q L^2 / k = {DELTA_T:.2f} K")
    print(f"  plate area       : {plate_area():.4f} L^2  ->  {Q_PLATE * plate_area() * L_PLATE ** 2:.0f} W "
          f"per metre of depth must leave through the channel")
    print(f"  channel wall     : {hole_perimeter() * L:.1f} mm round (Ramanujan)")
