r"""Ex_07.2 — the physics: time enters, twice.

*Deep Learning for Engineering* — MSc, Aalborg University.
Remus Teodorescu (ret@et.aau.dk), with support from Research Assistant
Noman Khan (nomank@energy.aau.dk).

**Reference texts.** Liu, *PINN with Python: An Introduction* (2025); Raissi,
Perdikaris & Karniadakis, *Physics-informed neural networks*, J. Comput. Phys.
**378** (2019) 686–707. These are the works to read for the theory. The code,
the problems and the exposition here are original to this course, written from
the 2019 paper and the PyTorch documentation and not derived from any
publisher's code listings.

Two problems, chosen so that the difference between them is the lesson.

## 1 · A silicon die cooling down — parabolic, ONE initial condition

A 10 x 10 mm die has just finished a power pulse. Its edges are clamped to the
package temperature by the lid; the stored heat diffuses out.

    theta_t = alpha (theta_xx + theta_yy)
    theta = 0 on the edges,  theta(x, y, 0) = given

The initial field carries **two modes**: a broad hot region from the bulk
dissipation, and a sharper feature left by a localised structure. They decay at
different rates — the sharp one 2.5x faster — so the field does not simply
shrink, it *changes shape*. A model that matches the late field can still be
badly wrong early, and notebook 02 measures exactly that.

The exact solution is a sum of separable modes, which is what makes the error
measurable:

    theta = A1 sin(pi X) sin(pi Y) exp(-2 pi^2 alpha t / L^2)
          + A2 sin(2 pi X) sin(pi Y) exp(-5 pi^2 alpha t / L^2)

with X = x/L, Y = y/L.

## 2 · A struck panel — hyperbolic, TWO initial conditions

A tensioned panel, clamped on all four edges, struck at t = 0. It starts
**flat** and **moving**.

    u_tt = c^2 (u_xx + u_yy)
    u = 0 on the edges
    u(x, y, 0) = 0            <- displacement
    u_t(x, y, 0) = v0 * mode  <- velocity

    u = (v0/omega) sin(omega t) sin(pi X) sin(pi Y),  omega = sqrt(2) pi c / L

This is where a second-order-in-time problem stops forgiving you. Give the
network only the displacement condition and ``u = 0`` everywhere, for all time,
satisfies the PDE exactly, both boundary conditions exactly, and the one
initial condition you supplied exactly. The loss goes to zero. The panel never
moves.

Notebook 03 asks students to do exactly that and watch it happen. A residual
near machine precision is not evidence that the problem was posed correctly,
and this is the cheapest possible demonstration.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "L_DIE", "ALPHA", "A_SLOW", "A_FAST", "T_PACKAGE", "HEAT_T_END",
    "HEAT_DOMAIN", "heat_exact", "heat_initial", "heat_rates",
    "L_PANEL", "C_WAVE", "V_STRIKE", "OMEGA", "WAVE_T_END",
    "WAVE_DOMAIN", "wave_exact", "wave_velocity", "wave_period",
    "hard_bc_factor", "describe_problem",
    "plot_slice", "plot_time_history",
]

# ══════════════════════════════════════════════════════════════════════════
# 1 · the die
# ══════════════════════════════════════════════════════════════════════════

L_DIE = 0.010            #: die edge length [m] — 10 mm square
ALPHA = 8.8e-5           #: thermal diffusivity of silicon [m^2/s]
A_SLOW = 40.0            #: amplitude of the broad mode at t = 0 [K]
A_FAST = 20.0            #: amplitude of the sharp mode at t = 0 [K]
T_PACKAGE = 65.0         #: package temperature the edges are clamped to [C]

#: Long enough for the slow mode to fall by about 97%.
HEAT_T_END = 0.20

#: ``((x_lo, x_hi), (y_lo, y_hi))`` — the die [m].
HEAT_DOMAIN = ((0.0, L_DIE), (0.0, L_DIE))


def heat_rates():
    """Decay rates of the two modes [1/s], slow first.

    Mode ``(m, n)`` on a square of side L decays at
    ``(m^2 + n^2) pi^2 alpha / L^2``. So (1,1) gives 2 and (2,1) gives 5 — the
    sharp mode dies two and a half times faster, and that ratio is the whole
    point of using two of them.
    """
    base = np.pi ** 2 * ALPHA / L_DIE ** 2
    return 2.0 * base, 5.0 * base


def heat_exact(x, y, t):
    """Excess temperature above the package [K]. NumPy or torch."""
    X, Y = x / L_DIE, y / L_DIE
    k_slow, k_fast = heat_rates()
    sin = np.sin if isinstance(x, np.ndarray) else __import__("torch").sin
    exp = np.exp if isinstance(x, np.ndarray) else __import__("torch").exp
    return (A_SLOW * sin(np.pi * X) * sin(np.pi * Y) * exp(-k_slow * t)
            + A_FAST * sin(2 * np.pi * X) * sin(np.pi * Y) * exp(-k_fast * t))


def heat_initial(x, y):
    """The field at t = 0 [K] — the two modes superposed."""
    return heat_exact(x, y, 0.0 * x)


# ══════════════════════════════════════════════════════════════════════════
# 2 · the panel
# ══════════════════════════════════════════════════════════════════════════

L_PANEL = 0.40           #: panel edge length [m]
C_WAVE = 80.0            #: wave speed [m/s] — sqrt(tension / areal density)
V_STRIKE = 0.50          #: peak initial velocity [m/s]

#: Fundamental angular frequency of a square membrane, mode (1,1).
OMEGA = np.sqrt(2.0) * np.pi * C_WAVE / L_PANEL

#: About three periods.
WAVE_T_END = 0.020

#: ``((x_lo, x_hi), (y_lo, y_hi))`` — the panel [m].
WAVE_DOMAIN = ((0.0, L_PANEL), (0.0, L_PANEL))


def wave_period() -> float:
    """The fundamental period [s]."""
    return float(2.0 * np.pi / OMEGA)


def wave_exact(x, y, t):
    """Transverse displacement [m]. NumPy or torch.

    Zero at t = 0 everywhere. The panel is flat and moving.
    """
    X, Y = x / L_PANEL, y / L_PANEL
    sin = np.sin if isinstance(x, np.ndarray) else __import__("torch").sin
    return (V_STRIKE / OMEGA) * sin(OMEGA * t) * sin(np.pi * X) * sin(np.pi * Y)


def wave_velocity(x, y, t):
    """Transverse velocity [m/s] — the second initial condition, and the one
    a student is invited to omit in notebook 03."""
    X, Y = x / L_PANEL, y / L_PANEL
    sin = np.sin if isinstance(x, np.ndarray) else __import__("torch").sin
    cos = np.cos if isinstance(x, np.ndarray) else __import__("torch").cos
    return V_STRIKE * cos(OMEGA * t) * sin(np.pi * X) * sin(np.pi * Y)


# ══════════════════════════════════════════════════════════════════════════
# shared
# ══════════════════════════════════════════════════════════════════════════

def hard_bc_factor(x, y, domain):
    """Vanishes on all four edges of ``domain`` — for hard spatial enforcement.

    Ex_07.1 used the same idea on a slot. Here the edges are at 0 and L rather
    than symmetric about the origin, so the factor is written in the scaled
    coordinate directly: ``X (1 - X) Y (1 - Y)``, up to a constant.
    """
    (x0, x1), (y0, y1) = domain[0], domain[1]
    X, Y = (x - x0) / (x1 - x0), (y - y0) / (y1 - y0)
    return 16.0 * X * (1 - X) * Y * (1 - Y)


def describe_problem() -> None:
    """Print both problems and the numbers a reader should sanity-check."""
    k_slow, k_fast = heat_rates()
    print("  THE DIE  (parabolic, one initial condition)")
    print(f"    size            : {L_DIE*1e3:.0f} x {L_DIE*1e3:.0f} mm silicon")
    print(f"    diffusivity     : {ALPHA:.2e} m^2/s")
    print(f"    edges clamped to : {T_PACKAGE:.0f} C")
    print(f"    initial peak    : {A_SLOW + A_FAST:.0f} K above package")
    print(f"    slow mode (1,1) : rate {k_slow:7.2f} 1/s   tau {1/k_slow*1e3:6.1f} ms")
    print(f"    fast mode (2,1) : rate {k_fast:7.2f} 1/s   tau {1/k_fast*1e3:6.1f} ms"
          f"   ({k_fast/k_slow:.1f}x faster)")
    print(f"    window          : 0 .. {HEAT_T_END*1e3:.0f} ms"
          f"   ({HEAT_T_END*k_slow:.1f} slow time constants)")
    print()
    print("  THE PANEL  (hyperbolic, TWO initial conditions)")
    print(f"    size            : {L_PANEL*1e2:.0f} x {L_PANEL*1e2:.0f} cm")
    print(f"    wave speed      : {C_WAVE:.0f} m/s")
    print(f"    fundamental     : {OMEGA/(2*np.pi):.1f} Hz"
          f"   period {wave_period()*1e3:.2f} ms")
    print(f"    struck at       : {V_STRIKE:.2f} m/s peak velocity")
    print(f"    peak deflection : {V_STRIKE/OMEGA*1e3:.3f} mm")
    print(f"    window          : 0 .. {WAVE_T_END*1e3:.0f} ms"
          f"   ({WAVE_T_END/wave_period():.1f} periods)")
    print()
    print("    u(x, y, 0) = 0 everywhere. The panel starts flat and moving,")
    print("    which is what makes the second initial condition load-bearing.")


def plot_slice(values, domain, nx=121, ny=121, ax=None, title="",
               label="", cmap="inferno", scale=1.0):
    """Filled contours of a field on a spatial slice."""
    import matplotlib.pyplot as plt
    from course_core import new_axes
    from pinn_core import grid_points
    X, Y, _ = grid_points(nx, ny, domain)
    ax = new_axes(ax, figsize=(5.0, 4.6))
    c = ax.contourf(X * scale, Y * scale,
                    np.asarray(values).reshape(X.shape), levels=24, cmap=cmap)
    ax.set_aspect("equal")
    ax.set_title(title)
    plt.colorbar(c, ax=ax, label=label, shrink=0.85)
    return ax


def plot_time_history(times, curves, ax=None, title="", ylabel=""):
    """Several quantities against time on one pair of axes."""
    from course_core import new_axes, CYCLE
    ax = new_axes(ax, figsize=(7.2, 4.2))
    for i, (name, y) in enumerate(curves.items()):
        ax.plot(np.asarray(times) * 1e3, np.asarray(y), lw=1.9,
                color=CYCLE[i % len(CYCLE)], label=name)
    ax.set_xlabel("t  [ms]"); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(frameon=False, fontsize=9); ax.grid(alpha=0.25)
    return ax
