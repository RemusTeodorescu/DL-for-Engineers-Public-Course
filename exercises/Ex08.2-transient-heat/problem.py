r"""Ex_08.2 — the physics: the plate of Ex_08.1, switched on.

*Deep Learning for Engineering* — MSc, Aalborg University.
Remus Teodorescu (ret@et.aau.dk), with support from Research Assistant
Noman Khan (nomank@energy.aau.dk).

**Reference texts.** Liu, *PINN with Python: An Introduction* (2025); Raissi,
Perdikaris & Karniadakis, *Physics-informed neural networks*, J. Comput. Phys.
**378** (2019) 686–707. These are the works to read for the theory. The code,
the problem and the exposition here are original to this course.

## The problem

The aluminium plate of Ex_08.1, with its cooling channel, is at the coolant's
temperature. At ``t = 0`` its heat generation is switched on.

    rho c_p T_t = k (T_xx + T_yy) + Q     in the plate, for t > 0
    T = T_coolant                         on the wall of the channel
    dT/dn = 0                             on the four outer edges
    T = T_coolant                         everywhere at t = 0

## Scaled units

Lengths are divided by the side ``L``, the rise above the coolant by
``DELTA_T = Q L^2 / k``, and time by the window ``T_END``. The scaled rise
``theta(x, y, tau)`` then obeys

    theta_tau = C (theta_xx + theta_yy + 1),      C = alpha T_END / L^2 = 0.55

with ``theta = 0`` on the channel and at ``tau = 0`` and zero normal gradient
on the edges. ``alpha = k / (rho c_p)`` is the diffusivity. Everything in this
module works in the scaled units.

## What this module gives the notebook

* the data of the plate, as constants;
* the geometry of Ex_08.1: the level set, the samplers;
* :func:`reference` — the reference solution, by linear finite elements on a
  mesh fitted to the channel and Crank-Nicolson in time, and
  :func:`reference_at` to read one instant of it at any point;
* :func:`sensor_readings` — what four thermocouples would record, for the
  inverse problem.

The samplers return NumPy: call ``to_tensor`` at the point of use. Nothing at
module level imports torch.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "L_PLATE", "K_PLATE", "RHO_CP", "Q_PLATE", "T_COOLANT", "DELTA_T", "ALPHA",
    "T_END", "C_SCALED", "HOLE", "DOMAIN", "SENSORS",
    "kelvin", "ellipse_phi", "hole_multiplier",
    "sample_spacetime", "sample_edges_spacetime",
    "reference", "reference_at", "sensor_readings", "describe_problem",
]

# ------------------------------------------------------------------ the data
L_PLATE = 0.100          #: m, the side of the plate
K_PLATE = 167.0          #: W/(m K), aluminium alloy 6061
RHO_CP = 2.43e6          #: J/(m^3 K), its heat capacity per volume
Q_PLATE = 1.0e6          #: W/m^3, the heat generated, switched on at t = 0
T_COOLANT = 40.0         #: degC, the channel wall and the start
DELTA_T = Q_PLATE * L_PLATE ** 2 / K_PLATE   #: K, the temperature scale, 59.88
ALPHA = K_PLATE / RHO_CP                     #: m^2/s, the diffusivity, 6.87e-05
T_END = 80.0             #: s, the window
C_SCALED = ALPHA * T_END / L_PLATE ** 2      #: the diffusivity in scaled units, 0.55

#: The cooling channel in scaled units: an ellipse of 36 x 22 mm at the centre.
HOLE = dict(xc=0.5, yc=0.5, a=0.18, b=0.11)
DOMAIN = ((0.0, 1.0), (0.0, 1.0))

#: Four thermocouples, in scaled coordinates: two corners' neighbours, two near the channel.
SENSORS = np.array([[0.10, 0.10], [0.90, 0.85], [0.50, 0.80], [0.80, 0.50]])


def kelvin(theta):
    """A scaled rise as kelvin above the coolant."""
    return DELTA_T * theta


# ------------------------------------------------------------------- geometry
def _phi_np(xy, xc, yc, a, b):
    return ((xy[:, 0] - xc) / a) ** 2 + ((xy[:, 1] - yc) / b) ** 2 - 1.0


def ellipse_phi(xy, hole=None):
    """Level set of the channel: negative inside it, zero on its wall, positive
    in the plate. Shape ``(N, 1)``; NumPy in, NumPy out; tensor in, tensor out.
    Only the first two columns of ``xy`` are read, so space-time points work."""
    h = hole or HOLE
    return (((xy[:, 0:1] - h["xc"]) / h["a"]) ** 2
            + ((xy[:, 1:2] - h["yc"]) / h["b"]) ** 2 - 1.0)


def hole_multiplier(xy, hole=None):
    """The boundary multiplier of L8.1: zero on the channel wall, positive in
    the plate, and bounded: ``phi / (1 + phi)`` runs from 0 on the wall towards
    1 far from it, so the network behind it stays of order one."""
    f = ellipse_phi(xy, hole)
    return f / (1.0 + f)


def _ellipse_points(n, h):
    t = np.linspace(0.0, 2.0 * np.pi, 20001)
    arc = np.r_[0.0, np.cumsum(np.hypot(np.diff(h["a"] * np.cos(t)),
                                        np.diff(h["b"] * np.sin(t))))]
    th = np.interp(np.linspace(0.0, arc[-1], n, endpoint=False), arc, t)
    return np.c_[h["xc"] + h["a"] * np.cos(th), h["yc"] + h["b"] * np.sin(th)]


def _perimeter(h):
    return float(np.pi * (3 * (h["a"] + h["b"])
                          - np.sqrt((3 * h["a"] + h["b"]) * (h["a"] + 3 * h["b"]))))


# ------------------------------------------------------------------- samplers
def sample_spacetime(n, hole=None, margin=1.05, early=2.0, seed=0):
    """``n`` points ``(x, y, tau)``: uniform in the plate by rejection, and in
    scaled time ``tau = u**early`` with ``u`` uniform, so ``early = 2`` puts
    more points at early times, where the field changes fastest."""
    h = hole or HOLE
    rng = np.random.default_rng(seed)
    out = np.empty((0, 2))
    while len(out) < n:
        p = rng.uniform(0.0, 1.0, (2 * n, 2))
        out = np.vstack([out, p[_phi_np(p, h["xc"], h["yc"], h["a"] * margin, h["b"] * margin) > 0]])
    return np.c_[out[:n], rng.uniform(0.0, 1.0, n) ** early]


def sample_edges_spacetime(n_per_edge, early=2.0, seed=1):
    """Points on the four insulated edges at random times. Returns a list of
    four ``(points, axis)`` pairs; ``axis`` is the direction of the normal."""
    rng = np.random.default_rng(seed)
    out = []
    for fixed, value, axis in ((0, 0.0, 0), (0, 1.0, 0), (1, 0.0, 1), (1, 1.0, 1)):
        p = np.empty((n_per_edge, 3))
        p[:, fixed] = value
        p[:, 1 - fixed] = rng.uniform(0.0, 1.0, n_per_edge)
        p[:, 2] = rng.uniform(0.0, 1.0, n_per_edge) ** early
        out.append((p, axis))
    return out


# ------------------------------------------------------------- the reference
def reference(spacing=1 / 100, steps=160, keep=40, c=None, hole=None):
    """The reference solution: linear finite elements in space on a mesh
    fitted to the channel, Crank-Nicolson in time.

    ``spacing`` is the node spacing, ``steps`` the number of time steps over
    the window, ``keep`` how many equally spaced instants are stored (the
    start is added). ``c`` is the scaled diffusivity, ``C_SCALED`` by default.

    Returns a dict: ``P`` the nodes, ``tri`` the triangles, ``tau`` the stored
    scaled times and ``theta`` the scaled rise, one row per stored time.

    Its accuracy is shown in the notebook, by refining both ``spacing`` and
    ``steps`` together.
    """
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    from scipy.spatial import Delaunay

    h = hole or HOLE
    c = C_SCALED if c is None else c
    g = np.linspace(0.0, 1.0, int(round(1.0 / spacing)) + 1)
    X, Y = np.meshgrid(g, g)
    P = np.c_[X.ravel(), Y.ravel()]
    P = P[_phi_np(P, h["xc"], h["yc"], h["a"] + 0.5 * spacing, h["b"] + 0.5 * spacing) > 0]
    n_in = len(P)
    P = np.vstack([P, _ellipse_points(int(np.ceil(_perimeter(h) / spacing)), h)])
    tri = Delaunay(P).simplices
    tri = tri[_phi_np(P[tri].mean(axis=1), h["xc"], h["yc"], h["a"], h["b"]) > 0]
    x, y = P[tri, 0], P[tri, 1]
    area = 0.5 * np.abs((x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0])
                        - (x[:, 2] - x[:, 0]) * (y[:, 1] - y[:, 0]))
    ok = area > 1e-14
    tri, x, y, area = tri[ok], x[ok], y[ok], area[ok]
    b = np.stack([y[:, 1] - y[:, 2], y[:, 2] - y[:, 0], y[:, 0] - y[:, 1]], 1) / (2 * area[:, None])
    cc = np.stack([x[:, 2] - x[:, 1], x[:, 0] - x[:, 2], x[:, 1] - x[:, 0]], 1) / (2 * area[:, None])
    Ke = area[:, None, None] * (b[:, :, None] * b[:, None, :] + cc[:, :, None] * cc[:, None, :])
    Me = area[:, None, None] * (np.ones((3, 3)) + np.eye(3))[None] / 12.0
    rows, cols = np.repeat(tri, 3, axis=1).ravel(), np.tile(tri, (1, 3)).ravel()
    n = len(P)
    K = sp.coo_matrix((Ke.ravel(), (rows, cols)), shape=(n, n)).tocsr()   # conduction
    M = sp.coo_matrix((Me.ravel(), (rows, cols)), shape=(n, n)).tocsr()   # storage
    f = np.zeros(n)
    np.add.at(f, tri.ravel(), np.repeat(area / 3.0, 3))                  # the source
    free = np.arange(n_in)                                                # wall nodes stay at zero
    Kf, Mf, ff = c * K[free][:, free], M[free][:, free], c * f[free]
    dt = 1.0 / steps
    lu = spla.splu((Mf + 0.5 * dt * Kf).tocsc())                          # factorised once
    B = (Mf - 0.5 * dt * Kf).tocsr()
    u = np.zeros(n_in)
    assert steps % keep == 0, "steps must be a multiple of keep"
    every = steps // keep
    taus, fields = [0.0], [np.zeros(n)]
    for k in range(1, steps + 1):
        u = lu.solve(B @ u + dt * ff)                                     # one Crank-Nicolson step
        if k % every == 0:
            full = np.zeros(n)
            full[free] = u
            taus.append(k * dt)
            fields.append(full)
    return dict(P=P, tri=tri, tau=np.array(taus), theta=np.array(fields))


def reference_at(ref, x, y, tau):
    """The reference's scaled rise at the points ``(x, y)`` and the stored
    instant nearest to ``tau``. NaN inside the channel."""
    from matplotlib.tri import LinearTriInterpolator, Triangulation
    P = ref["P"]
    k = int(np.abs(ref["tau"] - tau).argmin())
    f = LinearTriInterpolator(Triangulation(P[:, 0], P[:, 1], ref["tri"]), ref["theta"][k])
    return f(np.asarray(x), np.asarray(y)).filled(np.nan)


def sensor_readings(ref, noise=0.05, seed=3):
    """What the four thermocouples of ``SENSORS`` record: the reference at
    their positions at every stored instant, in kelvin above the coolant, plus
    Gaussian noise of ``noise`` kelvin. Returns ``(tau, readings)`` with
    ``readings`` shaped ``(instants, 4)``."""
    rng = np.random.default_rng(seed)
    r = np.array([kelvin(reference_at(ref, SENSORS[:, 0], SENSORS[:, 1], t)) for t in ref["tau"]])
    return ref["tau"], r + noise * rng.standard_normal(r.shape)


def describe_problem() -> None:
    """Print the plate and the numbers it implies."""
    h = HOLE
    L = L_PLATE * 1e3
    print(f"  plate            : {L:.0f} x {L:.0f} mm aluminium, k = {K_PLATE:.0f} W/(m K), "
          f"rho c_p = {RHO_CP / 1e6:.2f} MJ/(m^3 K)")
    print(f"  diffusivity      : alpha = k / (rho c_p) = {ALPHA:.2e} m^2/s")
    print(f"  heat generated   : Q = {Q_PLATE:.1e} W/m^3, switched on at t = 0")
    print(f"  cooling channel  : ellipse {2 * h['a'] * L:.0f} x {2 * h['b'] * L:.0f} mm at the centre, "
          f"wall at {T_COOLANT:.0f} degC")
    print(f"  start            : the whole plate at {T_COOLANT:.0f} degC; outer edges insulated")
    print(f"  temperature scale: Q L^2 / k = {DELTA_T:.2f} K")
    print(f"  time for heat to cross the plate: L^2 / alpha = {L_PLATE ** 2 / ALPHA:.2f} s")
    print(f"  window           : 0 to {T_END:.0f} s  ->  scaled diffusivity C = {C_SCALED:.2f}")
