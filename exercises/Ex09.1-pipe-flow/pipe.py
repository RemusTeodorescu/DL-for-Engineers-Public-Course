r"""Ex_09.1 (pipe version, October 2026) - the coolant in one tube of a cold plate, when the pump starts.

*Deep Learning for Engineering* - MSc, Aalborg University.
Remus Teodorescu (ret@et.aau.dk), with support from Research Assistant
Noman Khan (nomank@energy.aau.dk).

**Reference texts.** White, *Fluid Mechanics*, ch. 6 (pipe flow, the Darcy
friction factor); Liu, *PINN with Python* (2025). The code, the problem and the
exposition here are original to this course.

## The problem

A liquid cold plate carries a serpentine of copper tube, 6 mm inside
diameter. A 50/50 mix of water and ethylene glycol at 40 degC is pumped
through it at 0.3 L/min. In one straight leg of the serpentine, far enough
from the bends, the flow is laminar and the same at every cross-section, so
only the axial velocity ``u(r, t)`` is left, and Newton's law for the liquid
reduces to

    rho u_t = G + mu (u_rr + u_r / r),     u = 0 on the wall,  u = 0 at t = 0

with ``G`` the pressure drop per metre, switched on at ``t = 0`` and then
held. Its steady state is Hagen-Poiseuille's parabola, and Darcy's friction
factor ``f = 64 / Re`` follows from it. The start-up has an exact solution as
a Bessel series (Szymanski 1932): the parabola less the part that has not yet
arrived.

## Scaled units

Radius over ``R``, written through ``s = (r / R)^2`` so that the axis needs no
special treatment; time over the window ``T_END``; velocity over the steady
centre speed ``V``. Then

    u_t = A + 4 C (s u_ss + u_s),      A = 4 C,   C = nu T_END / R^2

``C`` is the viscosity in these units - the number section 6 finds.

## The numbers are typical values, not measurements

The tube, the coolant's density and viscosity, and the flow rate are values of
the right order, typed from memory; check them before quoting a result. The
pressure gradient is taken to be held from ``t = 0``: a pump settles in a small
fraction of the 4 s it takes the coolant to follow.
"""

from __future__ import annotations

import numpy as np
from scipy.special import j0, j1, jn_zeros

__all__ = [
    "R_TUBE", "D_TUBE", "RHO", "MU", "NU", "FLOW", "AREA", "U_MEAN", "RE", "G", "V", "T_END", "C_SCALED", "A_SCALED",
    "ENTRANCE", "darcy", "exact", "exact_mean", "fdm", "fdm_mean", "flow_meter", "describe_problem",
]

# ------------------------------------------------------------------ the data (typical values, from memory)
R_TUBE = 0.003                 #: m, inside radius of the tube
D_TUBE = 2 * R_TUBE            #: m, 6 mm inside
RHO = 1065.0                   #: kg/m^3, 50/50 water-glycol at 40 degC
MU = 2.5e-3                    #: Pa s, its viscosity at 40 degC
NU = MU / RHO                  #: m^2/s, kinematic viscosity
FLOW = 0.3e-3 / 60.0           #: m^3/s, 0.3 L/min
AREA = np.pi * R_TUBE ** 2     #: m^2
U_MEAN = FLOW / AREA           #: m/s, mean speed, steady
RE = U_MEAN * D_TUBE / NU      #: Reynolds number, about 450
G = 8 * MU * U_MEAN / R_TUBE ** 2   #: Pa/m, the pressure drop per metre that drives it (Poiseuille)
V = G * R_TUBE ** 2 / (4 * MU) #: m/s, the steady centre speed: the velocity scale, twice the mean
T_END = 4.0                    #: s, the window after the pump starts
C_SCALED = NU * T_END / R_TUBE ** 2   #: the viscosity in scaled units
A_SCALED = G * T_END / (RHO * V)      #: the pressure push in scaled units (= 4 C for the nominal coolant)
ENTRANCE = 0.05 * RE * D_TUBE  #: m, the length after a bend before the parabola has formed

_LAM = jn_zeros(0, 400)        # the zeros of J0


def darcy(re):
    """Darcy's friction factor of laminar pipe flow, ``64 / Re``."""
    return 64.0 / re


def exact(s, t, c=None, a=None, n=400):
    """The exact start-up velocity, scaled, at ``s = (r/R)^2`` and scaled time
    ``t = t_seconds / T_END``: Szymanski's Bessel series. ``c`` is the scaled
    viscosity (``C_SCALED`` by default) and ``a`` the scaled push
    (``A_SCALED``); the steady centre speed is ``a / (4 c)``."""
    c = C_SCALED if c is None else c
    a = A_SCALED if a is None else a
    s, t = np.broadcast_arrays(np.asarray(s, float), np.asarray(t, float))
    eta = np.sqrt(np.clip(s, 0.0, 1.0))
    lam = _LAM[:n]
    terms = 8 * j0(np.multiply.outer(eta, lam)) / (lam ** 3 * j1(lam)) * np.exp(-np.multiply.outer(4 * c * t, lam ** 2 / 4))
    return a / (4 * c) * ((1 - s) - terms.sum(axis=-1))


def exact_mean(t, c=None, a=None, n=400):
    """The exact mean velocity over the section - what a flow meter reads, scaled."""
    c = C_SCALED if c is None else c
    a = A_SCALED if a is None else a
    lam = _LAM[:n]
    t = np.asarray(t, float)
    return a / (4 * c) * (0.5 - (16 / lam ** 4 * np.exp(-np.multiply.outer(c * t, lam ** 2))).sum(axis=-1))


def fdm(n, steps, keep=40, c=None, a=None):
    """Finite differences: ``n`` nodes from the axis to the wall in ``r``, central
    differences, the axis by symmetry (the Laplacian there is ``2 u_rr``), and
    Crank-Nicolson in time with the matrix factorised once. Returns the node
    radii ``eta = r / R``, the stored scaled times and the velocity, one row per
    stored time."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    c = C_SCALED if c is None else c
    a = A_SCALED if a is None else a
    eta = np.linspace(0.0, 1.0, n)
    h = eta[1]
    m = n - 1                                     # unknowns: every node but the wall
    main = np.full(m, -2.0 / h ** 2)
    up = np.zeros(m - 1); lo = np.zeros(m - 1)
    j = np.arange(1, m)
    up[1:] = 1 / h ** 2 + 1 / (2 * h * eta[j[:-1]])  # u_rr + u_r / r at the interior nodes
    lo[:] = 1 / h ** 2 - 1 / (2 * h * eta[j])
    up[0] = 4 / h ** 2; main[0] = -4 / h ** 2      # the axis: 2 u_rr, with the mirror node
    L = sp.diags([lo, main, up], [-1, 0, 1], format="csc") * (c / 1.0)   # the operator in eta, times c
    # u_t = a + c (u_rr + u_r / r) in eta; s-form and eta-form are the same equation
    dt = 1.0 / steps
    I = sp.identity(m, format="csc")
    lu = spla.splu((I - 0.5 * dt * L).tocsc())
    B = (I + 0.5 * dt * L).tocsr()
    u = np.zeros(m)
    every = max(steps // keep, 1)
    ts, out = [0.0], [np.zeros(n)]
    for k in range(1, steps + 1):
        u = lu.solve(B @ u + dt * a)
        if k % every == 0:
            ts.append(k * dt); out.append(np.r_[u, 0.0])
    return eta, np.array(ts), np.array(out)


def fdm_mean(eta, field):
    """The mean velocity of a finite-difference profile: ``2 * integral u eta d eta``, by the trapezoidal rule."""
    return 2 * np.trapezoid(field * eta, eta, axis=-1)


def flow_meter(c_true, t_end_record=1.0, every=0.05, noise=0.01, seed=7):
    """A flow meter's record after the pump starts: the mean velocity (scaled)
    every ``every`` of the window up to ``t_end_record``, for a coolant whose
    scaled viscosity is ``c_true``, with noise of ``noise`` times the nominal
    steady mean. **Synthetic**: the exact solution, with the true viscosity."""
    rng = np.random.default_rng(seed)
    t = np.arange(every, t_end_record + 1e-9, every)
    return t, exact_mean(t, c=c_true) + noise * 0.5 * rng.standard_normal(len(t))


def describe_problem() -> None:
    """Print the data and the numbers it implies."""
    print("  values are TYPICAL, typed from memory - check before quoting")
    print(f"  tube            : {D_TUBE * 1e3:.0f} mm inside, one straight leg of a cold plate's serpentine")
    print(f"  coolant         : 50/50 water-glycol at 40 degC, rho = {RHO:.0f} kg/m^3, mu = {MU * 1e3:.2f} mPa s, nu = {NU:.2e} m^2/s")
    print(f"  flow            : {FLOW * 6e4:.1f} L/min -> mean speed {U_MEAN:.3f} m/s, Re = {RE:.0f} (laminar below about 2300)")
    print(f"  steady          : centre speed {V:.3f} m/s; pressure drop {G:.0f} Pa per metre; Darcy f = 64/Re = {darcy(RE):.3f}")
    print(f"  after a bend    : about 0.05 Re D = {ENTRANCE * 1e3:.0f} mm before the parabola has formed")
    print(f"  start-up        : R^2 / nu = {R_TUBE ** 2 / NU:.2f} s; window {T_END:.0f} s; scaled viscosity C = {C_SCALED:.3f}")


# ------------------------------------------------------------------ the cold plate, for the drawings (illustrative sizes)
PLATE = (0.250, 0.130, 0.012)  #: m, aluminium plate: length, width, thickness
LEG_X = (0.010, 0.240)         #: m, where the straight legs start and end along the plate
BEND_R = 0.012                 #: m, radius of a U-bend, to the tube's axis
LEG_Y = (0.029, 0.053, 0.077, 0.101)   #: m, the four legs, 2 x BEND_R apart
TUBE_Z = 0.006                 #: m, the tube's axis, at mid-thickness
MODULE = (0.095, 0.155, 0.035, 0.095)  #: m, the power module's footprint on the top face: x0, x1, y0, y1
#: m, the section A-A: a cut across the second leg (y = 53 mm) at x = 48 mm. The second leg runs from
#: its bend at x = 240 mm back towards x = 10 mm; the parabola has re-formed by x = 240 - 136 = 104 mm,
#: so A-A lies in the fully developed part, 56 mm beyond that point.
SECTION_AA = (0.048, 0.053)


def serpentine():
    """The tube's axis through the plate: a list of ``(kind, x, y)`` pieces,
    legs and bends in the order the coolant meets them."""
    pieces = []
    for k, y in enumerate(LEG_Y):
        x = np.linspace(*LEG_X, 50) if k % 2 == 0 else np.linspace(*LEG_X[::-1], 50)
        pieces.append(("leg", x, np.full_like(x, y)))
        if k < len(LEG_Y) - 1:
            a = np.linspace(-np.pi / 2, np.pi / 2, 40)
            if k % 2 == 0:                                   # a bend at the far end, turning back
                pieces.append(("bend", LEG_X[1] + BEND_R * np.cos(a), y + BEND_R + BEND_R * np.sin(a)))
            else:                                            # a bend at the near end
                pieces.append(("bend", LEG_X[0] - BEND_R * np.cos(a), y + BEND_R + BEND_R * np.sin(a)))
    return pieces


def draw_domain():
    """The physical domain, full width: the cold plate with its serpentine in
    3-D, the part of one leg this notebook solves, and the tube's
    cross-section with what is known on it."""
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(15, 5.6))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    L, W, H = (1e3 * v for v in PLATE)
    for z in (0, H):                                         # the plate's outline, top and bottom
        ax.plot([0, L, L, 0, 0], [0, 0, W, W, 0], [z] * 5, color="0.55", lw=1)
    for x, y in ((0, 0), (L, 0), (L, W), (0, W)):
        ax.plot([x, x], [y, y], [0, H], color="0.55", lw=1)
    m = [1e3 * v for v in MODULE]
    ax.plot([m[0], m[1], m[1], m[0], m[0]], [m[2], m[2], m[3], m[3], m[2]], [H] * 5, color="tab:red", lw=2)
    ax.text(m[1] + 4, m[3], H + 2, "power module", color="tab:red", fontsize=9)
    zt = 1e3 * TUBE_Z
    for kind, x, y in serpentine():
        ax.plot(1e3 * x, 1e3 * y, zt, color="tab:blue" if kind == "leg" else "tab:purple", lw=3)
    xs = np.linspace(1e3 * LEG_X[0], 1e3 * (LEG_X[1] - ENTRANCE), 20)   # leg 2 runs back from its bend; the parabola needs ENTRANCE first
    ax.plot(xs, np.full_like(xs, 1e3 * LEG_Y[1]), zt, color="tab:orange", lw=6)
    ax.text(1e3 * LEG_X[0], 1e3 * LEG_Y[1] - 12, zt, "solved here", color="tab:orange", fontsize=9)
    xa, ya = (1e3 * v for v in SECTION_AA)                    # the section A-A: a vertical plane across the leg
    ax.plot([xa, xa, xa, xa, xa], [ya - 9, ya + 9, ya + 9, ya - 9, ya - 9], [0, 0, H, H, 0], color="k", lw=1.5)
    ax.text(xa, ya + 11, H + 1, "A-A", fontsize=10)
    ax.text(-25, 1e3 * LEG_Y[0], zt, "in", fontsize=9)
    ax.text(-25, 1e3 * LEG_Y[-1], zt, "out", fontsize=9)
    ax.set_xlabel("x  [mm]"); ax.set_ylabel("y  [mm]"); ax.set_zlabel("z  [mm]")
    ax.set_box_aspect((L, W, 40)); ax.view_init(28, -62)
    ax.set_title(f"cold plate {L:.0f} x {W:.0f} x {H:.0f} mm; a {1e3 * D_TUBE:.0f} mm tube: four legs (blue), three bends (purple)", fontsize=10)

    ax = fig.add_subplot(1, 2, 2)
    a = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(a), np.sin(a), color="k", lw=3)           # the wall
    for rr in (0.25, 0.5, 0.75):
        ax.plot(rr * np.cos(a), rr * np.sin(a), color="0.85", lw=0.8)
    ax.annotate("", xy=(np.cos(0.6), np.sin(0.6)), xytext=(0, 0), arrowprops=dict(arrowstyle="->"))
    ax.text(0.42, 0.40, "R = 3 mm", fontsize=10)
    ax.annotate("", xy=(0.55, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="tab:green"))
    ax.text(0.20, -0.14, "r", color="tab:green", fontsize=11)
    ax.plot(0, 0, "o", color="tab:orange")
    ax.text(-0.95, -0.30, "axis: u largest, du/dr = 0", fontsize=9, color="tab:orange")
    ax.text(-1.0, -1.25, "wall: u = 0, the liquid sticks to it", fontsize=10)
    ax.text(-1.0, 1.10, "u points along the tube, out of the page", fontsize=10)
    eta = np.linspace(-1, 1, 41)                              # the steady profile across the section, drawn to the right
    ax.plot(1.5 + 0.45 * (1 - eta ** 2), eta, color="tab:blue", lw=2)
    ax.plot([1.5, 1.5], [-1, 1], color="0.5", lw=0.8)
    for e in np.linspace(-0.9, 0.9, 7):
        ax.annotate("", xy=(1.5 + 0.45 * (1 - e ** 2), e), xytext=(1.5, e), arrowprops=dict(arrowstyle="->", color="tab:blue", lw=0.8))
    ax.text(1.42, 1.10, "u(r), once steady", color="tab:blue", fontsize=10)
    ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(-1.3, 2.3); ax.set_ylim(-1.4, 1.3)
    ax.set_title(f"cross-section A-A (x = {1e3 * SECTION_AA[0]:.0f} mm on the second leg), and what is known on it", fontsize=10)
    plt.tight_layout()
    return fig


def draw_methods(n_fdm, pts_s, pts_t):
    """Where each method puts its unknowns: the leg in the plate (from
    above), the finite-difference nodes across the section, and the network's
    collocation points over radius and time."""
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(15, 4.6))
    ax = fig.add_subplot(1, 3, 1)
    L, W = 1e3 * PLATE[0], 1e3 * PLATE[1]
    ax.plot([0, L, L, 0, 0], [0, 0, W, W, 0], color="0.55", lw=1)
    for kind, x, y in serpentine():
        ax.plot(1e3 * x, 1e3 * y, color="tab:blue" if kind == "leg" else "tab:purple", lw=2)
    xs = np.linspace(1e3 * LEG_X[0], 1e3 * (LEG_X[1] - ENTRANCE), 20)
    ax.plot(xs, np.full_like(xs, 1e3 * LEG_Y[1]), color="tab:orange", lw=5, alpha=0.6)
    xc, yc = (1e3 * v for v in SECTION_AA)                    # the section A-A, as in draw_domain
    ax.plot([xc, xc], [yc - 9, yc + 9], color="k", lw=2)
    ax.text(xc + 2, yc + 9, "A-A", fontsize=9)
    ax.set_aspect("equal"); ax.set_xlabel("x  [mm]"); ax.set_ylabel("y  [mm]")
    ax.set_title("the plate from above; the solved part (orange), the section A-A", fontsize=10)
    ax = fig.add_subplot(1, 3, 2)
    a = np.linspace(0, 2 * np.pi, 300)
    eta = np.linspace(0, 1, n_fdm)
    for e in eta[1:-1]:
        ax.plot(e * np.cos(a), e * np.sin(a), color="0.6", lw=0.5)   # each node stands for a ring: u is the same all round it
    ax.plot(np.cos(a), np.sin(a), color="k", lw=2)
    ax.plot(eta, 0 * eta, "o", ms=3, color="tab:blue")
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"finite differences: {n_fdm} nodes from the axis to the wall, each a ring", fontsize=10)
    ax = fig.add_subplot(1, 3, 3)
    ax.plot(np.sqrt(pts_s), pts_t * T_END, ".", ms=1.5, color="tab:orange")
    ax.set_xlabel("r / R"); ax.set_ylabel("t  [s]")
    ax.set_title(f"the PINN: {len(pts_s)} collocation points over radius and time", fontsize=10)
    plt.tight_layout()
    return fig


def draw_sections(fields, times):
    """Colour maps of the velocity across the tube's section: one row per
    entry of ``fields`` (a name and a function of ``(s, t_scaled)`` giving
    m/s), one column per instant in ``times`` (s). A row whose name contains
    "error" is drawn in its own colour scale."""
    import matplotlib.pyplot as plt
    r = np.linspace(0, 1, 81); a = np.linspace(0, 2 * np.pi, 121)
    RR, AA = np.meshgrid(r, a)
    X, Y = RR * np.cos(AA), RR * np.sin(AA)
    re_ = np.linspace(0, 1, 82); ae = np.linspace(0, 2 * np.pi, 122)   # cell edges, so pcolormesh needs no guessing
    RE, AE = np.meshgrid(re_, ae); XE, YE = RE * np.cos(AE), RE * np.sin(AE)
    rows = list(fields.items())
    fig, axes = plt.subplots(len(rows), len(times), figsize=(3.3 * len(times) + 1.0, 3.0 * len(rows)), squeeze=False)
    speeds = [float(np.max(f(r ** 2, times[-1] / T_END))) for name, f in rows if "error" not in name]
    vmax = max(speeds) if speeds else None
    for i, (name, f) in enumerate(rows):
        err = "error" in name
        for j, t in enumerate(times):
            ax = axes[i, j]
            rc = 0.5 * (re_[1:] + re_[:-1])
            vals = np.tile(f(rc ** 2, t / T_END), (len(ae) - 1, 1))
            im = ax.pcolormesh(XE, YE, vals, cmap="magma" if err else "viridis", shading="auto", vmin=0, vmax=None if err else vmax)
            ax.set_aspect("equal"); ax.axis("off")
            ax.set_title(f"{name}, t = {t:g} s", fontsize=9)
            plt.colorbar(im, ax=ax, shrink=0.75, format="%.0e" if err else "%.2f", label="" if err else "u  [m/s]")
    plt.tight_layout()
    return fig


# ------------------------------------------------------------------ the same methods on one device (GPU on Colab)
def fdm_torch(n, steps, keep=40, c=None, a=None, device=None):
    """The finite differences of :func:`fdm`, written in torch so that they run
    on the same device as the network: the same nodes, the same axis rule, the
    same Crank-Nicolson step, the matrix LU-factorised once on the device.
    Returns tensors ``(eta, t, U)`` on the device."""
    import torch
    c = C_SCALED if c is None else c
    a = A_SCALED if a is None else a
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dt_ = torch.float64
    eta = torch.linspace(0.0, 1.0, n, dtype=dt_, device=dev)
    h = 1.0 / (n - 1)
    m = n - 1
    L = torch.zeros(m, m, dtype=dt_, device=dev)
    j = torch.arange(1, m, device=dev)
    L[j, j] = -2.0 / h ** 2
    L[j, j - 1] = 1 / h ** 2 - 1 / (2 * h * eta[j])
    L[j[:-1], j[:-1] + 1] = 1 / h ** 2 + 1 / (2 * h * eta[j[:-1]])
    L[0, 0], L[0, 1] = -4.0 / h ** 2, 4.0 / h ** 2       # the axis: 2 u_rr with the mirror node
    L = c * L
    dt = 1.0 / steps
    I = torch.eye(m, dtype=dt_, device=dev)
    lu, piv = torch.linalg.lu_factor(I - 0.5 * dt * L)
    B = I + 0.5 * dt * L
    u = torch.zeros(m, 1, dtype=dt_, device=dev)
    every = max(steps // keep, 1)
    out = [torch.zeros(n, dtype=dt_, device=dev)]
    for k in range(1, steps + 1):
        u = torch.linalg.lu_solve(lu, piv, B @ u + dt * a)
        if k % every == 0:
            out.append(torch.cat([u[:, 0], torch.zeros(1, dtype=dt_, device=dev)]))
    ts = torch.arange(len(out), dtype=dt_, device=dev) * every * dt
    return eta, ts, torch.stack(out)


def box_explicit(d, n, t_end=1.0, kappa=1.0, device=None):
    """The scaling test: the same kind of equation, u_t = 1 + kappa lap u, in a
    unit box of ``d`` dimensions with u = 0 on its walls and at the start, by
    explicit finite differences on ``n`` nodes per direction - the way a grid
    runs on a GPU, every node updated at once. The step is the largest that is
    stable, h^2 / (2 d kappa), so the number of steps grows with n^2. Returns
    the field at ``t_end`` and the number of steps."""
    import torch
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    h = 1.0 / (n - 1)
    dt = 0.9 * h * h / (2 * d * kappa)
    steps = int(np.ceil(t_end / dt)); dt = t_end / steps
    u = torch.zeros((n,) * d, dtype=torch.float64, device=dev)
    inner = (slice(1, -1),) * d
    for _ in range(steps):
        lap = -2 * d * u[inner]
        for ax in range(d):
            lo = [slice(1, -1)] * d; hi = [slice(1, -1)] * d
            lo[ax] = slice(0, -2); hi[ax] = slice(2, None)
            lap = lap + u[tuple(lo)] + u[tuple(hi)]
        u[inner] = u[inner] + dt * (1.0 + kappa * lap / h ** 2)
    return u, steps
