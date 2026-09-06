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
    "plot_field", "plot_error", "plot_source",
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
    X, Y, _ = grid_points(nx, ny, DOMAIN)
    ax = new_axes(ax, figsize=(4.6, 6.2))
    c = ax.contourf(_mm(X), _mm(Y), np.asarray(values).reshape(X.shape),
                    levels=24, cmap=cmap)
    ax.set_aspect("equal")
    ax.set_xlabel("x  [mm]"); ax.set_ylabel("y  [mm]")
    ax.set_title(title)
    plt.colorbar(c, ax=ax, label=label, shrink=0.85)
    return ax


def plot_error(predicted, nx: int = 121, ny: int = 121, ax=None,
               title: str = "signed error, θ̂ − θ  [K]"):
    """Where the model is wrong, and by how much, in kelvin.

    Signed and in physical units on purpose. A relative L2 norm of 1e-3 tells
    you the fit is good; this tells you whether the residual error sits at the
    hot spot, which is the only place a machine designer cares about.
    """
    from pinn_core import grid_points
    X, Y, pts = grid_points(nx, ny, DOMAIN)
    err = np.asarray(predicted).ravel() - theta_exact(pts[:, 0], pts[:, 1])
    return plot_field(err, nx, ny, ax, title, label="error  [K]", cmap="coolwarm")


def plot_source(nx: int = 121, ny: int = 121, ax=None):
    """The manufactured source, in MW/m³."""
    from pinn_core import grid_points
    _, _, pts = grid_points(nx, ny, DOMAIN)
    q = source(pts[:, 0], pts[:, 1]) / 1e6
    return plot_field(q, nx, ny, ax, "manufactured source q",
                      label="q  [MW/m³]", cmap="magma")
