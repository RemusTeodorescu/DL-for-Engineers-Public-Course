r"""Ex_07.1 — the physics: a stator slot in steady state.

*Deep Learning for Engineering* — MSc, Aalborg University.
Remus Teodorescu (ret@et.aau.dk), with support from Research Assistant
Noman Khan (nomank@energy.aau.dk).

**Reference texts.** Liu, *PINN with Python: An Introduction* (2025); Raissi,
Perdikaris & Karniadakis, *Physics-informed neural networks*, J. Comput. Phys.
**378** (2019) 686–707. These are the works to read for the theory. The code,
the problem and the exposition here are original to this course, written from
the 2019 paper and the PyTorch documentation and not derived from any
publisher's code listings. Where a symbol matches a textbook's it is because
both follow the standard notation of the field.

## The problem

A slot in the stator of an electrical machine, 10 mm wide and 20 mm deep,
holding a winding of 32 round copper wires, 2 mm bare, in 4 columns of 8.
Each wire carries the same current I, and each makes R I^2 of heat. The slot
walls are held at the temperature of the surrounding iron, T_WALL = 90 C,
which is cooled. In steady state the temperature rise theta = T - T_WALL obeys

    k_eff lap(theta) + q(theta, I) = 0      in the slot
    theta = 0                               on all four walls

**The winding is treated as one material.** Modelling every wire would make
the conductivity jump from about 400 W/m.K in the copper to about 0.2 in the
enamel and resin around it, at every wire edge. Machine designers average it
instead: one effective conductivity for the bundle, K_EFF = 0.70 W/m.K, and
the heat of the 32 wires spread over the slot,

    q = FILL * rho(T) * J^2,    J = I / A_WIRE,    FILL = 0.503

**Copper's resistance rises with temperature**, 0.39 % per kelvin, so
rho(T) = RHO_CU (1 + ALPHA_CU (T - 20)) and the heat depends on theta. That
makes theta grow faster than I^2 - about 5 % faster at 45 A - but q is still
affine in theta, so for any one current the problem stays a single linear
system. :func:`fdm_solve` solves it; :func:`ground_truth` makes it as
accurate as finite differences can.

**AC at 50 Hz heats like DC at its RMS value.** The slot's thermal time
constant is about 30 s against a 20 ms period, and copper's skin depth at
50 Hz is 9.3 mm against a 1 mm wire radius, so I here is the RMS current.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "A_HALF", "B_HALF", "K_EFF", "T_WALL", "DOMAIN", "RHO_CU",
    "N_COLS", "N_ROWS", "N_WIRES", "WIRE_D", "A_WIRE", "FILL", "ALPHA_CU",
    "I_RATED", "I_RANGE", "resistivity", "heat_per_wire", "heat_source",
    "heat_coefficients", "runaway_current", "describe_slot", "exact_solution",
    "fdm_grid", "fdm_matrix", "fdm_solve", "ground_truth",
    "hard_bc_factor", "plot_field", "plot_fdm_grid", "plot_slot_geometry",
]

# ── geometry and material ─────────────────────────────────────────────────

#: Half-width of the slot [m]. The slot is 10 mm across.
A_HALF = 0.005

#: Half-depth of the slot [m]. The slot is 20 mm deep.
B_HALF = 0.010

#: Effective conductivity of the impregnated bundle [W/m·K]. Not copper's 400
#: — this is a composite of strands, enamel, epoxy and voids, and the figure
#: is the one that matters for the hot spot.
K_EFF = 0.70


#: Slot-wall temperature [°C] — the local iron temperature.
T_WALL = 90.0

#: ``((x_lo, x_hi), (y_lo, y_hi))`` — the slot cross-section [m].
DOMAIN = ((-A_HALF, A_HALF), (-B_HALF, B_HALF))

#: Resistivity of copper at 20 °C [Ω·m], for the sanity check in §6.
RHO_CU = 1.72e-8


# ── the hard-wall mask ───────────────────────────────────────────────────────

def hard_bc_factor(x, y):
    """The factor that vanishes on all four walls, for hard enforcement.

    A network output multiplied by this **cannot** violate the boundary
    condition, whatever it learns:

        θ̂(x, y) = (1 − ξ²)(1 − η²) · N(x, y)

    Compare that with adding a penalty to the loss and hoping. Notebook 02
    measures the difference.
    """
    xi, eta = x / A_HALF, y / B_HALF
    return (1 - xi ** 2) * (1 - eta ** 2)


# ── the real slot: 32 wires, a current, and copper that heats up ─────────

#: The winding: 4 columns by 8 rows of round copper wire.
N_COLS, N_ROWS = 4, 8
N_WIRES = N_COLS * N_ROWS

#: Bare diameter of one wire [m], and its copper cross-section [m^2], 3.14 mm^2.
WIRE_D = 2.0e-3
A_WIRE = np.pi * (WIRE_D / 2) ** 2

#: The copper fraction of the slot: 32 x 3.14 mm^2 over 200 mm^2 = 0.503. The
#: rest is enamel, the slot liner and impregnating resin.
FILL = N_WIRES * A_WIRE / ((2 * A_HALF) * (2 * B_HALF))

#: Copper's temperature coefficient of resistance [1/K], referred to 20 C.
ALPHA_CU = 3.93e-3

#: Rated current per wire [A, RMS]: 10 A/mm^2 in a 2 mm wire, 31.4 A. The
#: notebooks use I_RANGE, from light load to 1.4 times rated.
I_RATED = 10e6 * A_WIRE
I_RANGE = (10.0, 45.0)


def resistivity(T):
    """Copper's resistivity at T [C], rising 0.39 % per kelvin above 20 C."""
    return RHO_CU * (1 + ALPHA_CU * (T - 20.0))


def heat_per_wire(I, T):
    """R I^2 of one wire at temperature T [C], per metre of slot [W/m]."""
    return resistivity(T) / A_WIRE * I ** 2


def heat_source(theta, I, T_wall=None):
    """The winding's heat, spread over the slot [W/m^3].

    Each wire makes R I^2 per metre, and the 32 together, spread over the
    slot's cross-section, make FILL * rho(T) * J^2 with J = I / A_WIRE. The
    copper sits at T = T_WALL + theta, so the heat rises as the slot warms.

    ``T_wall`` is the iron temperature [C]; T_WALL when left out.
    Polynomial in theta, so it works unchanged on arrays and tensors.
    """
    T_wall = T_WALL if T_wall is None else T_wall
    J = I / A_WIRE
    return FILL * resistivity(T_wall + theta) * J ** 2


def heat_coefficients(I, T_wall=None):
    """``(c0, c1)`` with q = c0 + c1 theta: the heat at the wall temperature
    [W/m^3], and how much it grows per kelvin of rise [W/m^3/K]."""
    T_wall = T_WALL if T_wall is None else T_wall
    J2 = (I / A_WIRE) ** 2
    return (FILL * resistivity(T_wall) * J2,
            FILL * RHO_CU * ALPHA_CU * J2)


def runaway_current() -> float:
    """The current above which the slot has no steady state [A].

    The heat grows by c1 per kelvin; conduction removes it at a rate set by
    the slowest cooling pattern of the slot, K_EFF * lambda_1, with
    lambda_1 = pi^2 (1/W^2 + 1/H^2) for a rectangle held at zero on its
    walls. When c1 reaches it, every extra kelvin makes more heat than it can
    shed. A property of this model, not a rating - real insulation fails far
    earlier.
    """
    lam1 = np.pi ** 2 * (1 / (2 * A_HALF) ** 2 + 1 / (2 * B_HALF) ** 2)
    return float(A_WIRE * np.sqrt(K_EFF * lam1 / (FILL * RHO_CU * ALPHA_CU)))


def describe_slot(I=None) -> None:
    """Print the slot, the winding and the heat, at current I (rated if None)."""
    I = I_RATED if I is None else I
    c0, c1 = heat_coefficients(I)
    print(f"  slot            : {2*A_HALF*1e3:.0f} x {2*B_HALF*1e3:.0f} mm, walls held at {T_WALL:.0f} C")
    print(f"  winding         : {N_COLS} x {N_ROWS} = {N_WIRES} wires of {WIRE_D*1e3:.0f} mm, "
          f"copper fill {FILL:.2f}")
    print(f"  bundle k_eff    : {K_EFF:.2f} W/m.K   (copper alone: about 400)")
    print(f"  current         : {I:.1f} A per wire = {I/A_WIRE/1e6:.1f} A/mm^2   "
          f"(the notebooks use {I_RANGE[0]:.0f} to {I_RANGE[1]:.0f} A)")
    print(f"  one wire at {T_WALL:.0f} C: R = {resistivity(T_WALL)/A_WIRE*1e3:.2f} mOhm/m, "
          f"R I^2 = {heat_per_wire(I, T_WALL):.2f} W/m")
    print(f"  whole slot      : {N_WIRES*heat_per_wire(I, T_WALL):.0f} W per metre of slot")
    print(f"  heat source q   : {c0/1e6:.2f} MW/m^3 at the wall temperature, "
          f"+{c1/c0*100:.2f} % per kelvin of rise")
    print(f"  thermal runaway : {runaway_current():.0f} A, "
          f"{runaway_current()/I_RANGE[1]:.1f} x the largest current used")


def fdm_grid(nx: int):
    """The grid :func:`fdm_solve` uses: nx nodes across the slot and 2 nx - 1
    down it, walls included, so every cell is square."""
    return (np.linspace(-A_HALF, A_HALF, nx),
            np.linspace(-B_HALF, B_HALF, 2 * nx - 1))


def fdm_matrix(I, nx: int = 41, T_wall=None):
    """The finite-difference system for one current: ``(A, rhs)`` with
    A theta = rhs over the interior nodes, walls left out (they are zero).

    The five-point stencil stands in for the Laplacian at every interior node,

        (east + west - 2 centre) / h^2  +  (north + south - 2 centre) / h^2,

    and because the heat is affine in theta, q = c0 + c1 theta, the equation
    k_eff lap(theta) + q = 0 becomes (k_eff L + c1) theta = -c0: one sparse
    matrix with at most five entries in a row.
    """
    import warnings
    import scipy.sparse as sp
    x, y = fdm_grid(nx)
    hx, hy = x[1] - x[0], y[1] - y[0]
    mx, my = nx - 2, len(y) - 2
    with warnings.catch_warnings():            # scipy's own diags() notice
        warnings.simplefilter("ignore")
        Dx = sp.diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(mx, mx)) / hx ** 2
        Dy = sp.diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(my, my)) / hy ** 2
        L = sp.kronsum(Dx, Dy, format="csc")    # rows of x, stacked down y
        c0, c1 = heat_coefficients(I, T_wall)
        A = (K_EFF * L + c1 * sp.identity(mx * my, format="csc")).tocsc()
    return A, -c0 * np.ones(mx * my)


def fdm_solve(I, nx: int = 41, T_wall=None):
    """The slot's temperature rise by finite differences [K].

    Builds :func:`fdm_matrix` on an nx x (2 nx - 1) grid and solves it.
    Returns ``(X, Y, theta)``, each of shape (2 nx - 1, nx), walls included,
    laid out exactly as ``pinn_core.grid_points(nx, 2 nx - 1, DOMAIN)`` lays
    them, so ``theta.ravel()`` goes straight into :func:`plot_field`.
    """
    import scipy.sparse.linalg as spla
    x, y = fdm_grid(nx)
    A, rhs = fdm_matrix(I, nx, T_wall)
    theta = np.zeros((len(y), nx))
    theta[1:-1, 1:-1] = spla.spsolve(A, rhs).reshape(len(y) - 2, nx - 2)
    X, Y = np.meshgrid(x, y)
    return X, Y, theta


def ground_truth(I, nx: int = 321, T_wall=None):
    """theta on the nx grid, as accurate as finite differences can make it [K].

    The stencil's error falls as h^2, so a solve on this grid and one on a
    grid twice as fine, combined as (4 fine - coarse) / 3, cancel the leading
    error (Richardson extrapolation). At nx = 321 the fine solve and the
    extrapolation agree to about 1e-5 K on a 19 K rise. A few seconds.
    """
    X, Y, coarse = fdm_solve(I, nx, T_wall)
    _, _, fine = fdm_solve(I, 2 * nx - 1, T_wall)
    return X, Y, (4 * fine[::2, ::2] - coarse) / 3


def exact_solution(I, x, y, n_terms: int = 399, T_wall=None):
    """The slot's temperature rise, exactly, as a double sine series [K].

    With q = c0 + c1 theta the equation k_eff lap(theta) + c1 theta = -c0 has
    constant coefficients, and the slot is a rectangle held at zero on its
    walls, so theta is a sum of the slot's modes
    sin(m pi x'/W) sin(n pi y'/H), x' = x + W/2, y' = y + H/2, each with
    lap = -lambda_mn times itself. The constant source is 16/(pi^2 m n) times
    each odd mode, so

        theta = sum over odd m, n of 16 c0 / (pi^2 m n (k_eff lambda_mn - c1)) * mode.

    ``x`` and ``y`` are 1-D node coordinates [m]; returns an array of shape
    (len(y), len(x)), laid out as :func:`fdm_solve` lays its answer. Terms up
    to ``n_terms`` in each direction: at 399 the sum is within about 1e-5 K.
    The first term, m = n = 1, is the dome of notebook section 2.
    """
    c0, c1 = heat_coefficients(I, T_wall)
    W, H = 2 * A_HALF, 2 * B_HALF
    m = np.arange(1, n_terms + 1, 2)
    lam = (np.pi * m[:, None] / W) ** 2 + (np.pi * m[None, :] / H) ** 2
    A = 16 * c0 / (np.pi ** 2 * m[:, None] * m[None, :] * (K_EFF * lam - c1))
    Sx = np.sin(np.outer(np.asarray(x) + A_HALF, np.pi * m / W))     # (nx, M)
    Sy = np.sin(np.outer(np.asarray(y) + B_HALF, np.pi * m / H))     # (ny, N)
    return Sy @ A.T @ Sx.T


# ── pictures ──────────────────────────────────────────────────────────────

def _mm(pts):
    return pts * 1e3


def plot_field(values, nx: int = 121, ny: int = 121, ax=None,
               title: str = "", label: str = "θ  [K]", cmap: str = "inferno",
               wires: bool = False):
    """Filled contours of a field sampled on :func:`pinn_core.grid_points`.

    ``wires=True`` outlines the 32 wires on top. The field itself does not
    show them: the winding is averaged into one material, so the heat is
    spread evenly and the outlines only say where the copper sits.
    """
    import matplotlib.pyplot as plt
    from course_core import new_axes
    from pinn_core import grid_points
    values = np.asarray(values).ravel()
    if values.size != nx * ny:            # sampled on another square grid
        n = int(round(np.sqrt(values.size)))
        if n * n == values.size:
            nx = ny = n
    X, Y, _ = grid_points(nx, ny, DOMAIN)
    ax = new_axes(ax, figsize=(4.6, 6.2))
    v = np.asarray(values).reshape(X.shape)
    # a diverging map means a signed field: centre it on zero, or zero lands
    # wherever the data range puts it and is drawn red or blue
    signed = cmap in ("coolwarm", "RdBu", "RdBu_r", "bwr", "seismic")
    if signed:
        m = float(np.nanmax(np.abs(v))) or 1.0
        levels = np.linspace(-m, m, 25)          # zero is the white midpoint
    else:
        levels = 24
    c = ax.contourf(_mm(X), _mm(Y), v, levels=levels, cmap=cmap)
    if signed and np.nanmin(v) < 0 < np.nanmax(v):
        # the zero line: where the model is exactly right
        ax.contour(_mm(X), _mm(Y), v, levels=[0.0], colors="k", linewidths=1.0)
    if wires:
        from matplotlib.patches import Circle
        a, b, rad = A_HALF * 1e3, B_HALF * 1e3, WIRE_D / 2 * 1e3
        for i in range(N_COLS):
            for j in range(N_ROWS):
                ax.add_patch(Circle((-a + (i + 0.5) * 2 * a / N_COLS,
                                     -b + (j + 0.5) * 2 * b / N_ROWS), rad,
                                    facecolor="none", edgecolor="white",
                                    lw=0.8, alpha=0.75))
    ax.set_aspect("equal")
    ax.set_xlabel("x  [mm]"); ax.set_ylabel("y  [mm]")
    ax.set_title(title)
    ticks = None
    if signed:                       # round labels inside the same range
        from matplotlib.ticker import MaxNLocator
        ticks = [t for t in MaxNLocator(7, symmetric=True).tick_values(-m, m)
                 if abs(t) <= m * 1.0001]
    plt.colorbar(c, ax=ax, label=label, shrink=0.85, ticks=ticks)
    return ax


def plot_fdm_grid(nx: int = 7, ax=None, node=(2, 4)):
    """What finite differences does to the slot, on a grid small enough to count.

    The slot rectangle with the :func:`fdm_grid` nodes on it: the wall nodes are
    known (theta = 0), every interior node is one unknown, and one node is
    picked out with the four neighbours its equation uses. ``node`` is that
    node's (column, row), counted from the bottom-left wall node.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Circle
    from course_core import new_axes

    x, y = (_mm(v) for v in fdm_grid(nx))
    ny = len(y)
    X, Y = np.meshgrid(x, y)
    wall = np.zeros(X.shape, bool)
    wall[[0, -1], :] = True
    wall[:, [0, -1]] = True
    n_unknown = int((~wall).sum())

    WALL, NODE, P, NB, CU = "#1f77b4", "#4a4f57", "#d94f2b", "#e8a33d", "#d94f2b"
    ax = new_axes(ax, figsize=(4.6, 6.2))

    # the slot, and its wires faintly behind the grid
    a, b = A_HALF * 1e3, B_HALF * 1e3
    ax.add_patch(Rectangle((-a, -b), 2 * a, 2 * b, facecolor="#fdf6ee",
                           edgecolor=WALL, lw=2.4, zorder=0))
    rad = WIRE_D / 2 * 1e3
    for i in range(N_COLS):
        for j in range(N_ROWS):
            ax.add_patch(Circle((-a + (i + 0.5) * 2 * a / N_COLS,
                                 -b + (j + 0.5) * 2 * b / N_ROWS), rad,
                                facecolor=CU, edgecolor="none", alpha=0.13,
                                zorder=0))
    for xv in x:
        ax.plot([xv, xv], [y[0], y[-1]], color="#c9ccd1", lw=0.6, zorder=1)
    for yv in y:
        ax.plot([x[0], x[-1]], [yv, yv], color="#c9ccd1", lw=0.6, zorder=1)

    # the nodes: known on the walls, unknown inside
    ax.scatter(X[wall], Y[wall], s=34, marker="s", facecolor="white",
               edgecolor=WALL, lw=1.2, zorder=3,
               label="on the wall: θ = 0, known")
    ax.scatter(X[~wall], Y[~wall], s=30, color=NODE, zorder=3,
               label="inside: one unknown θ each (%d)" % n_unknown)

    # one node and the four neighbours its equation uses
    ci, cj = node
    for di, dj, name in ((1, 0, "E"), (-1, 0, "W"), (0, 1, "N"), (0, -1, "S")):
        ax.plot([x[ci], x[ci + di]], [y[cj], y[cj + dj]], color=NB, lw=2.2,
                zorder=4)
        ax.scatter([x[ci + di]], [y[cj + dj]], s=70, color=NB, zorder=5,
                   label="its four neighbours" if name == "E" else None)
        ax.text(x[ci + di] + 0.35 * di + 0.30 * abs(dj),
                y[cj + dj] + 0.35 * dj + 0.30 * abs(di), name, color=NB,
                fontsize=9, fontweight="bold", ha="left" if di >= 0 else "right",
                va="bottom" if dj >= 0 else "top", zorder=6)
    ax.scatter([x[ci]], [y[cj]], s=110, color=P, zorder=6,
               label="one node: its θ is set by them\nand by its own heat")

    ax.set_aspect("equal")
    ax.set_xlim(-a - 1.2, a + 1.2)
    ax.set_ylim(-b - 1.2, b + 1.2)
    ax.set_xlabel("x  [mm]"); ax.set_ylabel("y  [mm]")
    ax.set_title("a %d × %d grid on the slot: %d unknowns" % (nx, ny, n_unknown))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), fontsize=8,
              frameon=False, ncol=1)
    return ax


def plot_fdm_mesh(nx: int = 161, ax=None, corner: float = 0.5e-3):
    """The mesh :func:`fdm_solve` really uses, with one corner enlarged.

    Left: the whole slot with its 32 wires and a small box in the bottom-left
    corner. The mesh itself is too fine to see at this scale, so the box is
    drawn again, enlarged, in an inset: there the real nodes are visible - the
    wall nodes (known, theta = 0), the interior nodes (the unknowns), and one
    node with the four neighbours its equation uses. ``corner`` is the side of
    the enlarged box [m].
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Circle
    from course_core import new_axes

    x, y = fdm_grid(nx)
    h = x[1] - x[0]
    a, b, c = A_HALF * 1e3, B_HALF * 1e3, corner * 1e3
    WALL, NODE, P, NB, CU = "#1f77b4", "#4a4f57", "#d94f2b", "#e8a33d", "#d94f2b"
    ax = new_axes(ax, figsize=(4.6, 6.2))

    def slot(axis, faint=True):
        axis.add_patch(Rectangle((-a, -b), 2 * a, 2 * b, facecolor="#fdf6ee",
                                 edgecolor=WALL, lw=2.4, zorder=0))
        rad = WIRE_D / 2 * 1e3
        for i in range(N_COLS):
            for j in range(N_ROWS):
                axis.add_patch(Circle((-a + (i + 0.5) * 2 * a / N_COLS,
                                       -b + (j + 0.5) * 2 * b / N_ROWS), rad,
                                      facecolor=CU, edgecolor="none",
                                      alpha=0.13 if faint else 0.25, zorder=0))

    slot(ax)
    ax.set_aspect("equal")
    ax.set_xlim(-a - 0.8, a + 0.8); ax.set_ylim(-b - 0.8, b + 0.8)
    ax.set_xlabel("x  [mm]"); ax.set_ylabel("y  [mm]")
    n_unknown = (nx - 2) * (len(y) - 2)
    ax.set_title(f"the mesh: {nx} x {len(y)} nodes, h = {h * 1e3:.4f} mm\n"
                 f"{n_unknown:,} unknowns - the boxed corner enlarged", fontsize=10)

    ins = ax.inset_axes([0.30, 0.30, 0.62, 0.40])
    slot(ins, faint=False)
    xs, ys = x[x <= -A_HALF + corner + 1e-12] * 1e3, y[y <= -B_HALF + corner + 1e-12] * 1e3
    for xv in xs:
        ins.plot([xv, xv], [ys[0], ys[-1]], color="#c9ccd1", lw=0.6, zorder=1)
    for yv in ys:
        ins.plot([xs[0], xs[-1]], [yv, yv], color="#c9ccd1", lw=0.6, zorder=1)
    X, Y = np.meshgrid(xs, ys)
    wall = np.isclose(X, -a) | np.isclose(Y, -b)
    ins.scatter(X[wall], Y[wall], s=22, marker="s", facecolor="white", edgecolor=WALL,
                lw=1.0, zorder=3, label="wall node: θ = 0, known")
    ins.scatter(X[~wall], Y[~wall], s=16, color=NODE, zorder=3, label="interior node: one unknown")
    ci = cj = len(xs) // 2
    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        ins.plot([xs[ci], xs[ci + di]], [ys[cj], ys[cj + dj]], color=NB, lw=2.0, zorder=4)
        ins.scatter([xs[ci + di]], [ys[cj + dj]], s=40, color=NB, zorder=5,
                    label="its four neighbours" if (di, dj) == (1, 0) else None)
    ins.scatter([xs[ci]], [ys[cj]], s=60, color=P, zorder=6, label="one node's equation")
    ins.set_xlim(xs[0] - 0.04, xs[-1] + 0.02); ins.set_ylim(ys[0] - 0.04, ys[-1] + 0.02)
    ins.set_aspect("equal"); ins.set_xticks([]); ins.set_yticks([])
    ins.set_title(f"the bottom-left corner, {c:g} x {c:g} mm", fontsize=8)
    ax.indicate_inset_zoom(ins, edgecolor="k", alpha=0.8)
    for sp in ins.spines.values():
        sp.set_edgecolor("k"); sp.set_linewidth(1.2)
    ins.legend(loc="upper center", bbox_to_anchor=(0.5, -0.04), fontsize=7, frameon=False)
    return ax


def plot_slot_geometry(ax=None, strands: bool = True, n_slots: int = 36):
    """Where the problem sits: one stator slot, and half of each neighbour.

    A sector of the stator, cut on two radial lines through the middle of the
    neighbouring slots - the repeating section a machine drawing shows. The
    slot is drawn from A_HALF and B_HALF with *parallel* sides, so it is the
    rectangle the notebooks solve on, put back where it belongs. The strand
    circles are the 32 wires of 2 mm, to scale.

    ``n_slots`` is the slot count of the machine and sets the tooth width;
    36 over this bore leaves a tooth about as wide as the slot, which is
    normal for a machine of this size.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Circle
    from matplotlib.path import Path
    from course_core import new_axes

    a, depth = A_HALF * 1e3, 2 * B_HALF * 1e3   # 5 mm half-width, 20 mm deep
    r_bore, neck, a_neck, yoke = 110.0, 2.0, 2.0, 15.0
    r1 = r_bore + neck                 # slot starts above the tooth tips
    r2 = r1 + depth                    # slot bottom
    r_out = r2 + yoke                  # back of the yoke
    pitch = 2 * np.pi / n_slots
    half = pitch                       # the cut: one pitch either side
    mid = np.pi / 2                    # the centre slot points up

    IRON, EDGE, CU, WALL = "#b9bec6", "#6b7079", "#d94f2b", "#1f77b4"
    CUT, VOID = "#0f9d58", "#fdf6ee"
    ax = new_axes(ax, figsize=(6.6, 4.9))

    def local(phi, xs, rs):
        """Points given as (tangential offset, radius) about angle ``phi``."""
        ur = np.array([np.cos(phi), np.sin(phi)])
        ut = np.array([-np.sin(phi), np.cos(phi)])
        return np.array([r * ur + x * ut for x, r in zip(xs, rs)])

    def arc(r, a0, a1, n=80):
        th = np.linspace(a0, a1, n)
        return np.c_[r * np.cos(th), r * np.sin(th)]

    # the sector of iron, and the clip that cuts the half slots at its edges
    sector = np.vstack([arc(r_out, mid - half, mid + half),
                        arc(r_bore, mid + half, mid - half)])
    ax.add_patch(Polygon(sector, closed=True, facecolor=IRON,
                         edgecolor=EDGE, lw=1.2, zorder=1))
    clip = Polygon(sector, closed=True, transform=ax.transData)

    def slot(phi, centre):
        """One slot: the body, its semiclosed neck, and what fills it."""
        body = local(phi, [-a, a, a, -a], [r1, r1, r2, r2])
        nk = local(phi, [-a_neck, a_neck, a_neck, -a_neck],
                   [r_bore, r_bore, r1, r1])
        for poly, lw in ((nk, 1.0), (body, 2.4 if centre else 1.0)):
            pt = Polygon(poly, closed=True, facecolor=VOID, zorder=2,
                         edgecolor=(WALL if centre else EDGE), lw=lw)
            pt.set_clip_path(clip)
            ax.add_patch(pt)
        if not strands:
            return
        ncol, nrow = N_COLS, N_ROWS               # the real winding, to scale
        rad = WIRE_D / 2 * 1e3
        for i in range(ncol):
            for j in range(nrow):
                c = local(phi, [-a + (i + 0.5) * 2 * a / ncol],
                          [r1 + (j + 0.5) * depth / nrow])[0]
                ci = Circle(c, rad, facecolor=CU, zorder=3,
                            edgecolor="#8f2f18", lw=0.4,
                            alpha=1.0 if centre else 0.45)
                ci.set_clip_path(clip)
                ax.add_patch(ci)

    for k in (-1, 0, 1):
        slot(mid + k * pitch, centre=(k == 0))

    # the two cut lines, and the bore
    for s in (-1, 1):
        ph = mid + s * half
        ax.plot([r_bore * np.cos(ph), r_out * np.cos(ph)],
                [r_bore * np.sin(ph), r_out * np.sin(ph)],
                color=CUT, lw=1.6, ls=(0, (6, 2, 1, 2)), zorder=5)
    ax.text(*(local(mid + half, [0.0], [r_out + 4.5])[0]), "cut",
            color=CUT, fontsize=8.5, ha="center", va="center")
    ax.text(*(local(mid - half, [0.0], [r_out + 4.5])[0]), "cut",
            color=CUT, fontsize=8.5, ha="center", va="center")
    bore = arc(r_bore, mid - half * 1.06, mid + half * 1.06)
    ax.plot(bore[:, 0], bore[:, 1], color=EDGE, lw=0.9, ls=(0, (4, 3)))

    # what everything is - text in the corners, leaders into the drawing
    ax.annotate("slot wall:  \u03b8 = 0\n(iron held at %.0f \u00b0C)" % T_WALL,
                xy=local(mid, [-a], [r1 + 0.62 * depth])[0],
                xytext=(0.985, 0.74), textcoords="axes fraction",
                color=WALL, fontsize=8.5, ha="right", va="center",
                arrowprops=dict(arrowstyle="->", color=WALL, lw=1.2,
                                shrinkB=2))
    ax.annotate("32 copper wires of 2 mm\neach makes R I\u00b2 of heat",
                xy=local(mid, [0.60 * a], [r1 + 0.22 * depth])[0],
                xytext=(0.015, 0.22), textcoords="axes fraction",
                color="#8f2f18", fontsize=8.5, ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color="#8f2f18", lw=1.2,
                                shrinkB=2))
    ax.text(*(local(mid, [0.0], [r2 + yoke * 0.55])[0]), "stator yoke",
            fontsize=8.5, ha="center", va="center", color="#4a4f57")
    for s in (-1, 1):
        ax.text(*(local(mid + s * pitch / 2, [0.0], [r1 + 0.52 * depth])[0]),
                "tooth", fontsize=7.5, ha="center", va="center",
                color="#4a4f57",
                rotation=np.degrees(mid + s * pitch / 2) - 90)
        ax.text(*(local(mid + s * pitch, [0.0], [r_bore - 6.0])[0]),
                "half of\nthe next", fontsize=7, ha="center", va="top",
                color="#8a8f97")
    ax.text(*(local(mid, [0.0], [r_bore - 6.0])[0]),
            "air gap \u00b7 rotor below", fontsize=8, ha="center",
            va="top", color=EDGE, style="italic")

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("One stator slot of %d, with half of each neighbour\n"
                 "the slot is %.0f \u00d7 %.0f mm \u2014 that rectangle is the "
                 "domain" % (n_slots, 2 * a, depth), fontsize=9.5)
    ax.set_xlim(-0.30 * r_out, 0.30 * r_out)
    ax.set_ylim(r_bore - 17.0, r_out + 10.0)
    return ax
