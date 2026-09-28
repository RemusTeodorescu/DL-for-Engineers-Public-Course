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

A slot in the stator of an electrical machine, filled with an impregnated
copper winding bundle. Current through the winding dissipates heat; the slot
walls are held at the temperature of the surrounding iron, which is cooled.
In steady state the excess temperature θ = T − T_wall satisfies Poisson's
equation

    k_eff ∇²θ + q(x, y) = 0        on the slot cross-section
    θ = 0                          on all four walls

The bundle is a **poor conductor** — epoxy, enamel and trapped air between
strands give an effective conductivity below 1 W/m·K, two to three orders
below solid copper. That is why a realistic current density produces a
temperature rise you can measure, and it is why slot hot spots are what
actually limit a machine's rating.

## The manufactured solution

The source is chosen so the exact answer is known, which is what lets the
notebooks measure **true error** rather than estimate it:

    θ(x, y) = ΔT (1 − ξ²)(1 − η²)(1 + s ξ),     ξ = x/a,  η = y/b

Zero on all four walls by construction. Polynomial rather than trigonometric,
so a network cannot do well by discovering a single Fourier mode. Skewed by
``s`` toward one side — a real slot is hotter near the closed end — so there is
no symmetry to exploit either.

The required source follows by differentiating:

    q = −k_eff ∇²θ
      = −k_eff ΔT [ (−2 − 6 s ξ)(1 − η²)/a²
                    + (1 − ξ²)(1 + s ξ)(−2/b²) ]

With ``s ≤ 1/3`` the bracket is negative everywhere, so **q ≥ 0 over the whole
slot** — the source is heating everywhere, as ohmic dissipation must be. A
manufactured problem that quietly requires negative heat generation is a
mathematics exercise wearing an engineering costume; this one does not.

Everything here is polynomial, so :func:`source` and :func:`theta_exact` work
unchanged on NumPy arrays and on torch tensors. No branching on type.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "A_HALF", "B_HALF", "K_EFF", "DELTA_T", "SKEW", "T_WALL",
    "DOMAIN", "RHO_CU", "FILL_FACTOR",
    "theta_exact", "temperature_exact", "source",
    "hard_bc_factor", "equivalent_current_density", "describe_problem",
    "plot_field", "plot_error", "plot_source", "plot_slot_geometry",
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

#: Peak excess temperature of the manufactured solution [K].
DELTA_T = 20.0

#: Skew of the profile toward +x. Must satisfy ``SKEW <= 1/3`` or the implied
#: source goes negative near a corner. See the module docstring.
SKEW = 0.30

#: Slot-wall temperature [°C] — the local iron temperature.
T_WALL = 90.0

#: ``((x_lo, x_hi), (y_lo, y_hi))`` — the slot cross-section [m].
DOMAIN = ((-A_HALF, A_HALF), (-B_HALF, B_HALF))

#: Resistivity of copper at 20 °C [Ω·m], for the sanity check in §6.
RHO_CU = 1.72e-8

#: Slot fill factor — the fraction of the slot area that is copper.
FILL_FACTOR = 0.5


# ── the manufactured solution and the source it implies ───────────────────

def theta_exact(x, y):
    """Excess temperature above the slot wall [K]. NumPy or torch."""
    xi, eta = x / A_HALF, y / B_HALF
    return DELTA_T * (1 - xi ** 2) * (1 - eta ** 2) * (1 + SKEW * xi)


def temperature_exact(x, y):
    """Absolute temperature [°C]."""
    return T_WALL + theta_exact(x, y)


def source(x, y):
    """Volumetric heat generation q [W/m³] — NumPy or torch.

    Derived by differentiating :func:`theta_exact`, so the pair is consistent
    by construction rather than by hope. Non-negative everywhere for
    ``SKEW <= 1/3``.
    """
    xi, eta = x / A_HALF, y / B_HALF
    lap = DELTA_T * ((-2 - 6 * SKEW * xi) * (1 - eta ** 2) / A_HALF ** 2
                     + (1 - xi ** 2) * (1 + SKEW * xi) * (-2 / B_HALF ** 2))
    return -K_EFF * lap


def hard_bc_factor(x, y):
    """The factor that vanishes on all four walls, for hard enforcement.

    A network output multiplied by this **cannot** violate the boundary
    condition, whatever it learns:

        θ̂(x, y) = (1 − ξ²)(1 − η²) · N(x, y)

    Compare that with adding a penalty to the loss and hoping. Notebook 02 is
    the measurement of the difference.
    """
    xi, eta = x / A_HALF, y / B_HALF
    return (1 - xi ** 2) * (1 - eta ** 2)


# ── keeping the physics honest ────────────────────────────────────────────

def equivalent_current_density(q) -> float:
    """The current density in the copper implied by a source q [A/mm²].

    q = ρ J² over the copper, and only ``FILL_FACTOR`` of the slot is copper,
    so a slot-averaged q corresponds to q/FILL in the strands themselves.

    Print this. A manufactured problem is only worth solving if the numbers it
    implies are ones a machine designer would recognise, and this is the line
    that checks.
    """
    return float(np.sqrt(np.asarray(q) / (RHO_CU * FILL_FACTOR)) / 1e6)


def describe_problem() -> None:
    """Print the geometry, the material and the numbers they imply."""
    from pinn_core import grid_points
    _, _, pts = grid_points(201, 201, DOMAIN)
    q = source(pts[:, 0], pts[:, 1])
    print(f"  slot            : {2*A_HALF*1e3:.0f} x {2*B_HALF*1e3:.0f} mm"
          f"   (half-width {A_HALF*1e3:.0f} mm, half-depth {B_HALF*1e3:.0f} mm)")
    print(f"  bundle k_eff    : {K_EFF:.2f} W/m.K")
    print(f"  wall            : {T_WALL:.0f} C")
    print(f"  peak rise       : {DELTA_T:.0f} K   -> hot spot {T_WALL+DELTA_T:.0f} C")
    print(f"  source q        : {q.min()/1e6:.2f} .. {q.max()/1e6:.2f} MW/m^3"
          f"   (mean {q.mean()/1e6:.2f})")
    print(f"  non-negative    : {bool(q.min() >= 0)}")
    print(f"  implied J       : {equivalent_current_density(q.mean()):.1f} A/mm^2"
          f"   in the copper   (machines run 5-20)")


# ── pictures ──────────────────────────────────────────────────────────────

def _mm(pts):
    return pts * 1e3


def plot_field(values, nx: int = 121, ny: int = 121, ax=None,
               title: str = "", label: str = "θ  [K]", cmap: str = "inferno"):
    """Filled contours of a field sampled on :func:`pinn_core.grid_points`."""
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


def plot_error(predicted, nx: int = 121, ny: int = 121, ax=None,
               title: str = "signed error, θ̂ − θ  [K]"):
    """Where the model is wrong, and by how much, in kelvin.

    Signed and in physical units on purpose. A relative L2 norm of 1e-3 tells
    you the fit is good; this tells you whether the residual error sits at the
    hot spot, which is the only place a machine designer cares about.
    """
    from pinn_core import grid_points
    predicted = np.asarray(predicted).ravel()
    if predicted.size != nx * ny:         # sampled on another square grid
        n = int(round(np.sqrt(predicted.size)))
        if n * n == predicted.size:
            nx = ny = n
    X, Y, pts = grid_points(nx, ny, DOMAIN)
    err = predicted - theta_exact(pts[:, 0], pts[:, 1])
    return plot_field(err, nx, ny, ax, title, label="error  [K]", cmap="coolwarm")


def plot_slot_geometry(ax=None, strands: bool = True, n_slots: int = 36):
    """Where the problem sits: one stator slot, and half of each neighbour.

    A sector of the stator, cut on two radial lines through the middle of the
    neighbouring slots - the repeating section a machine drawing shows. The
    slot is drawn from A_HALF and B_HALF with *parallel* sides, so it is the
    rectangle the notebooks solve on, put back where it belongs. The strand
    circles cover FILL_FACTOR of the slot at the radius drawn.

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
        ncol, nrow = 5, 10
        rad = np.sqrt(FILL_FACTOR * (2 * a) * depth / (np.pi * ncol * nrow))
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
    ax.annotate("copper strands in epoxy\nohmic heat:  q(x, y) > 0",
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


def plot_source(nx: int = 121, ny: int = 121, ax=None):
    """The manufactured source, in MW/m³."""
    from pinn_core import grid_points
    _, _, pts = grid_points(nx, ny, DOMAIN)
    q = source(pts[:, 0], pts[:, 1]) / 1e6
    return plot_field(q, nx, ny, ax, "manufactured source q",
                      label="q  [MW/m³]", cmap="magma")
