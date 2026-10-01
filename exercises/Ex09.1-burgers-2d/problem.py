r"""Ex_09.1 — the problem: a front carried by a flow (2-D coupled Burgers).

*Deep Learning for Engineering* — MSc, Aalborg University.
Remus Teodorescu (ret@et.aau.dk), with support from Research Assistant
Noman Khan (nomank@energy.aau.dk).

**Reference texts.** Liu, *PINN with Python: An Introduction* (2025); Raissi,
Perdikaris & Karniadakis, *Physics-informed neural networks*, J. Comput. Phys.
**378** (2019) 686–707. These are the works to read for the theory. The code,
the problem and the exposition here are original to this course.

## The problem

    u_t + u u_x + v u_y = nu (u_xx + u_yy)
    v_t + u v_x + v v_y = nu (v_xx + v_yy)

on the unit square, for 0 <= t <= 1. These are the momentum equations of a
flow with the pressure and the incompressibility constraint left out: they
keep the convective term, which is nonlinear and couples the two components,
and the viscous term. Speeds are in units of a reference speed U and lengths
in units of the square's side L, so the only number left is the Reynolds
number, Re = U L / nu = 1 / nu.

They have an exact solution (checked against both equations):

    u = 3/4 - 1 / (4 (1 + exp(s))),    v = 3/4 + 1 / (4 (1 + exp(s))),
    s = ((y - x) - t/4) / (8 nu)

a single **front** along the line y = x + t/4, which moves across the square
as time runs. Its width in (y - x) is 8 nu = 8/Re: 0.40 at Re = 20 and 0.016
at Re = 500. And u + v = 3/2 everywhere and always, which is a check that
needs no exact solution.

The values on the four edges and at t = 0 are taken from the exact solution,
so every error the notebook measures is the method's and never the data's.

## What is in here

The data, the exact solution, the points a network is trained on and the
points it is scored on. No solver: the finite-difference solver and the
network are written in the notebook.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "DOMAIN", "T_END", "RE", "NU", "RE_HIGH", "NU_HIGH", "JUMP",
    "exact", "front_width", "sample_spacetime", "sample_edges_and_start",
    "check_points", "describe_problem",
]

#: ``((x_lo, x_hi), (y_lo, y_hi))`` — the unit square, in units of L.
DOMAIN = ((0.0, 1.0), (0.0, 1.0))

#: End of the time window, in units of L/U. The front moves a quarter of a side in it.
T_END = 1.0

#: The Reynolds number of sections 4 and 5, and the viscosity it implies.
RE = 20.0
NU = 1.0 / RE

#: The Reynolds number of section 6.
RE_HIGH = 500.0
NU_HIGH = 1.0 / RE_HIGH

#: The change of u (and of v) across the front, in units of U.
JUMP = 0.25


def exact(x, y, t, nu=NU):
    """The exact solution: ``(u, v)`` at the points ``(x, y)`` and time ``t``.
    NumPy in, NumPy out; the arrays keep the shape of ``x``."""
    s = np.clip(((y - x) - 0.25 * t) / (8.0 * nu), -500.0, 500.0)   # clipped so exp cannot overflow
    e = JUMP / (1.0 + np.exp(s))
    return 0.75 - e, 0.75 + e


def front_width(nu=NU):
    """The width of the front in (y - x): 8 nu, which is 8/Re."""
    return 8.0 * nu


# ------------------------------------------------------------------- samplers
def sample_spacetime(n, seed=88):
    """``n`` collocation points in the square and the time window, shaped
    ``(n, 3)`` with time last. A Latin hypercube: each coordinate is cut into
    ``n`` slices and every slice is used once, so no region is left empty."""
    rng = np.random.default_rng(seed)
    cols = [(rng.permutation(n) + rng.random(n)) / n for _ in range(3)]
    return np.c_[cols[0], cols[1], cols[2] * T_END]


def sample_edges_and_start(n_edge=20, n_times=12, n_start=400, seed=88):
    """The points where the solution is given: ``n_edge`` on each of the four
    edges at each of ``n_times`` instants, and ``n_start`` in the square at
    t = 0. Shaped ``(N, 3)``, the edges first."""
    rng = np.random.default_rng(seed + 5)
    s = np.linspace(0.0, 1.0, n_edge)
    out = []
    for t in np.linspace(0.0, T_END, n_times):
        tt = np.full(n_edge, t)
        out += [np.c_[s, 0 * s, tt], np.c_[s, 0 * s + 1, tt], np.c_[0 * s, s, tt], np.c_[0 * s + 1, s, tt]]
    out.append(np.c_[rng.random(n_start), rng.random(n_start), np.zeros(n_start)])
    return np.vstack(out)


def check_points(nu=NU, n=20000, instants=21, seed=0):
    """The points a network is scored on, shaped ``(N, 3)``: at each of
    ``instants`` times, ``n`` points spread over the square and up to ``n``
    more laid ACROSS the front, within three front widths of it. A regular
    grid would not do: at a high Reynolds number the front passes between its
    nodes and the error at the front is never seen."""
    rng = np.random.default_rng(seed)
    out = []
    for t in np.linspace(0.0, T_END, instants):
        x = rng.random(n)
        y = x + 0.25 * t + (rng.random(n) - 0.5) * 6.0 * front_width(nu)
        keep = (y > 0.0) & (y < 1.0)
        p = np.vstack([np.c_[x[keep], y[keep]], rng.random((n, 2))])
        out.append(np.c_[p, np.full(len(p), t)])
    return np.vstack(out)


def describe_problem(nu=NU) -> None:
    """Print the problem and the numbers it implies."""
    g = np.linspace(0.0, 1.0, 101)
    X, Y = np.meshgrid(g, g)
    u0, v0 = exact(X, Y, 0.0, nu)
    print("  equations       : the coupled Burgers equations, two components (u, v)")
    print(f"  region, window  : the unit square, t from 0 to {T_END:.0f}  (lengths in L, speeds in U, time in L/U)")
    print(f"  Reynolds number : Re = U L / nu = {1.0 / nu:.0f}, so nu = {nu:.2g} in scaled units")
    print(f"  front           : along y = x + t/4, {front_width(nu):.2g} wide in (y - x); "
          f"it moves {0.25 * T_END:.2f} of a side in the window")
    print(f"  u at t = 0      : {u0.min():.2f} to {u0.max():.2f}        v at t = 0: {v0.min():.2f} to {v0.max():.2f}")
    print("  a free check    : u + v = 1.50 everywhere and always")
