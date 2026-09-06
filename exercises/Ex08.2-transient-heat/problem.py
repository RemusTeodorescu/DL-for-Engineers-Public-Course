r"""Ex_08.2 — the physics: a plate, a hole, and time.

*Deep Learning for Engineering* — MSc, Aalborg University.
Remus Teodorescu (ret@et.aau.dk), with support from Research Assistant
Noman Khan (nomank@energy.aau.dk).

**Reference texts.** Liu, *PINN with Python: An Introduction* (2025); Raissi,
Perdikaris & Karniadakis, *Physics-informed neural networks*, J. Comput. Phys.
**378** (2019) 686–707. These are the works to read for the theory. The code,
the problems and the exposition here are original to this course, written from
the 2019 paper and the PyTorch documentation and not derived from any
publisher's code listings.

Two problems share one file, because they share one plate.

## 1 · The square plate, cooling — parabolic, ONE initial condition

The unit square, held at zero on all four edges, released from a single
Helmholtz mode:

    T_t = c (T_xx + T_yy)
    T = 0 on the edges,  T(x, y, 0) = sin(pi x) sin(pi y)

The exact solution is that mode with a decaying amplitude:

    T = exp(-2 pi^2 c t) sin(pi x) sin(pi y)

The eigenvalue of the lowest mode of the square, 2 pi^2, *is* the decay rate.
The **shape never changes**, only the amplitude — which is what makes it a
clean benchmark, and also what makes the number reported alongside it matter:
by t = 1 the amplitude is about 3e-9, so a relative error of 100% there can
mean an absolute error of 1e-8. :func:`error_vs_time` returns both, always.

The time constant is ``tau = L^2 / (pi^2 c)``, and the transient is
essentially over by ``4 tau``. Choose ``t_end`` from that number rather than
from habit: solve for very much longer and you have computed a steady state,
correctly, at great expense.

## 2 · The same plate with an elliptical hole

Carried over from Ex_08.1 and now given a time coordinate. The hole is a
level set,

    phi(x, y) = ((x - xc)/a)^2 + ((y - yc)/b)^2 - 1

negative inside the hole, zero on it, positive in the material. Two things
follow, and both are used:

* ``phi`` **is** the hard-boundary multiplier. For a shape with a closed-form
  level set no multiplier network needs training — the analytic shortcut of
  L8.1 slide 15.
* ``grad phi`` gives the normal. The material's outward normal is the hole's
  *inward* normal, so :func:`ellipse_normal` carries a minus sign. Getting it
  backwards produces a plausible, inverted field and a flux balance that is
  wrong by exactly a factor of −1.

The hole is sampled by **arc length**, not by angle. Uniform angular spacing
under-samples the high-curvature ends of an ellipse, which is where the flux
concentrates.

## 3 · What the notebooks add

Notebook 04 makes ``c`` unknown and recovers it from a synthetic cooling
curve. Nothing in this file changes for that: the diffusivity is an argument
everywhere it appears, never a module constant, which is what lets a notebook
pass a trainable tensor in its place.

## A note on NumPy and torch

The reference solutions are called from both worlds — from NumPy in a plotting
or checking cell, from torch inside a residual. Each one dispatches on the type
of its first argument through :func:`_lib`, so ``exact_transient`` works on
arrays, on tensors and on plain floats without a second version of the formula
existing anywhere.

The samplers here follow the convention of ``pinn_core``: **they return NumPy**.
Call ``to_tensor(pts, requires_grad=True)`` at the point of use.
"""

from __future__ import annotations

import numpy as np
import torch

from course_core import SEED, new_axes, to_numpy, to_tensor
from pinn_core import (grad, grid_points, interior_points, max_abs_error,
                       relative_l2)

__all__ = [
    "PLATE_DOMAIN", "HOLE",
    "ellipse_phi", "hole_multiplier", "ellipse_normal",
    "plate_points", "hole_points", "plate_spacetime_points",
    "exact_manufactured", "source_manufactured", "flux_balance",
    "steady_grid_masked",
    "exact_transient", "time_constant", "fourier_number",
    "slice_at_time", "error_vs_time",
    "describe_problem", "plot_slice",
]

#: ``((x_lo, x_hi), (y_lo, y_hi))`` — the plate, non-dimensional.
PLATE_DOMAIN = ((0.0, 1.0), (0.0, 1.0))

#: Centre and semi-axes of the elliptical hole, in the same coordinates.
HOLE = dict(xc=0.5, yc=0.5, a=0.18, b=0.11)


def _lib(x):
    """The array library ``x`` belongs to — ``torch`` for a tensor, else NumPy.

    One formula, two worlds. A plain Python float goes to NumPy, which is the
    behaviour you want when a notebook writes ``exact_transient(X, Y, 0.0)``.
    """
    return torch if isinstance(x, torch.Tensor) else np


# ══════════════════════════════════════════════════════════════════════════
# 1 · the geometry: a plate with an elliptical hole
# ══════════════════════════════════════════════════════════════════════════

def _phi_flat(xy, xc, yc, a, b):
    """The level set on flat ``(N, 2)`` columns — used by the samplers."""
    return ((xy[:, 0] - xc) / a) ** 2 + ((xy[:, 1] - yc) / b) ** 2 - 1.0


def ellipse_phi(xy, hole=None):
    """Level-set function: negative inside the hole, zero on it, positive out.

    Keeps its column shape, so the result is ``(N, 1)`` and can be multiplied
    straight into a network output. NumPy or torch.
    """
    h = hole or HOLE
    return (((xy[:, 0:1] - h["xc"]) / h["a"]) ** 2
            + ((xy[:, 1:2] - h["yc"]) / h["b"]) ** 2 - 1.0)


def hole_multiplier(xy, hole=None):
    """Vanishes on the hole boundary, positive in the material.

    This is the analytic shortcut of L8.1 slide 15: for a shape with a
    closed-form level set, no multiplier network needs training.
    """
    return ellipse_phi(xy, hole)


def ellipse_normal(xy, hole=None):
    """Unit normal on the hole, pointing OUT of the material (into the hole).

    Sign matters: the material's outward normal is the inward normal of the
    hole. Getting it backwards produces a plausible but inverted field.
    """
    h = hole or HOLE
    lib = _lib(xy)
    gx = 2.0 * (xy[:, 0:1] - h["xc"]) / h["a"] ** 2
    gy = 2.0 * (xy[:, 1:2] - h["yc"]) / h["b"] ** 2
    n = lib.sqrt(gx ** 2 + gy ** 2) + 1e-12
    return -gx / n, -gy / n          # minus: out of the material


# ══════════════════════════════════════════════════════════════════════════
# 2 · sampling the geometry
# ══════════════════════════════════════════════════════════════════════════
#
# The library's samplers cover a box. These three cover what a box cannot: a
# domain with a piece removed, and its curved boundary. Everything else the
# notebooks need — the outer edges, the initial slice, the boundary in time —
# is a rectangle, so it comes straight from ``pinn_core``.

def plate_points(n, domain=PLATE_DOMAIN, hole=None, margin=1.05,
                 method="lhs", seed=None):
    """``n`` interior points on the plate, rejecting anything inside the hole.

    Rejection rather than a mapped grid: the hole is one shape today and a
    different one next week, and rejection does not care which. ``margin``
    pushes the rejected region slightly beyond the hole so that no collocation
    point lands on the boundary itself, where the multiplier is zero and the
    residual therefore says nothing.

    Returns ``(n, 2)`` NumPy. ``seed=None`` means the course seed, so a
    notebook that does not ask for a seed still gets the same points twice.
    """
    h = hole or HOLE
    rng = np.random.default_rng(SEED if seed is None else seed)
    keep = np.empty((0, 2))
    while len(keep) < n:
        cand = interior_points(max(2 * n, 512), domain, method,
                               seed=int(rng.integers(2 ** 32)))
        ok = _phi_flat(cand, h["xc"], h["yc"],
                       h["a"] * margin, h["b"] * margin) > 0
        keep = np.vstack([keep, cand[ok]])
    return keep[:n]


def hole_points(n, hole=None):
    """Points on the hole, spaced by arc length rather than by angle.

    Uniform angular spacing under-samples the high-curvature ends of an
    ellipse, which is where the flux concentrates. Returns ``(n, 2)`` NumPy;
    wrap with ``requires_grad=True`` if you are going to take a flux through
    them.
    """
    h = hole or HOLE
    t = np.linspace(0, 2 * np.pi, 4000)
    x, y = h["a"] * np.cos(t), h["b"] * np.sin(t)
    s = np.concatenate([[0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    tt = np.interp(np.linspace(0, s[-1], n, endpoint=False), s, t)
    return np.column_stack([h["xc"] + h["a"] * np.cos(tt),
                            h["yc"] + h["b"] * np.sin(tt)])


def plate_spacetime_points(n, t_end=1.0, with_hole=False, domain=PLATE_DOMAIN,
                           hole=None, margin=1.05, method="lhs", seed=None):
    """``n`` points scattered through the plate AND through time, ``(n, 3)``.

    Time is the last column. With ``with_hole=False`` this is exactly
    ``spacetime_points``; with ``with_hole=True`` it is that, minus the
    cylinder swept by the hole.
    """
    h = hole or HOLE
    slab = tuple(domain) + ((0.0, float(t_end)),)
    rng = np.random.default_rng(SEED if seed is None else seed)
    if not with_hole:
        return interior_points(n, slab, method,
                               seed=int(rng.integers(2 ** 32)))
    keep = np.empty((0, 3))
    while len(keep) < n:
        cand = interior_points(max(2 * n, 512), slab, method,
                               seed=int(rng.integers(2 ** 32)))
        ok = _phi_flat(cand, h["xc"], h["yc"],
                       h["a"] * margin, h["b"] * margin) > 0
        keep = np.vstack([keep, cand[ok]])
    return keep[:n]


# ══════════════════════════════════════════════════════════════════════════
# 3 · the steady problem — manufactured solution and a flux check
# ══════════════════════════════════════════════════════════════════════════

def exact_manufactured(X, Y):
    """T = sin(pi x) sin(pi y) — the field used throughout L7 and Ex_07.1."""
    lib = _lib(X)
    return lib.sin(np.pi * X) * lib.sin(np.pi * Y)


def source_manufactured(xy):
    """Q/k that makes :func:`exact_manufactured` solve T_xx + T_yy + Q/k = 0."""
    lib = _lib(xy)
    return (2.0 * np.pi ** 2
            * lib.sin(np.pi * xy[:, 0:1]) * lib.sin(np.pi * xy[:, 1:2]))


def flux_balance(model, xy_hole, q_source, k=1.0, hole=None, trial=None):
    """Heat leaving through the hole versus heat generated inside the plate.

    In steady state these must match. A mismatch of more than a few percent
    means the flux condition was never really learned.

    ``xy_hole`` must be a tensor with ``requires_grad=True`` — the flux is a
    derivative of the field, so the points are differentiated through.
    """
    h = hole or HOLE
    T = model(xy_hole) if trial is None else trial(model, xy_hole)
    g = grad(T, xy_hole)
    nx, ny = ellipse_normal(xy_hole, h)
    q_n = -k * (g[:, 0:1] * nx + g[:, 1:2] * ny)
    perim = np.pi * (3 * (h["a"] + h["b"])
                     - np.sqrt((3 * h["a"] + h["b"]) * (h["a"] + 3 * h["b"])))
    out = float(q_n.mean().item()) * perim
    area = 1.0 - np.pi * h["a"] * h["b"]
    return out, q_source * area


def steady_grid_masked(model, k=161, domain=PLATE_DOMAIN, hole=None,
                       trial=None):
    """Evaluate a steady model on a grid, masking points inside the hole.

    Returns ``(X, Y, T)`` with ``NaN`` in the hole, which ``contourf`` leaves
    blank — the picture then shows the geometry rather than hiding it.
    """
    h = hole or HOLE
    X, Y, pts = grid_points(k, k, domain)
    with torch.no_grad():
        xy = to_tensor(pts)
        T = to_numpy(model(xy) if trial is None else trial(model, xy))
    T = T.reshape(X.shape)
    T[_phi_flat(pts, h["xc"], h["yc"], h["a"], h["b"]).reshape(X.shape) < 0] = np.nan
    return X, Y, T


# ══════════════════════════════════════════════════════════════════════════
# 4 · the transient problem — reference solution and timescales
# ══════════════════════════════════════════════════════════════════════════

def exact_transient(X, Y, t, c=1.0):
    """u = exp(-2 pi^2 c t) sin(pi x) sin(pi y).

    The fundamental mode of the 2-D heat equation on the unit square with
    zero edges, obtained by separation of variables. Standard since Fourier;
    it is the exact solution this set checks the network against.

    The spatial factor is the lowest Helmholtz mode of the square; the
    eigenvalue 2 pi^2 becomes the decay rate. The SHAPE never changes, only
    the amplitude — which is what makes it a clean benchmark.
    """
    lib = _lib(X)
    return (lib.exp(-2.0 * np.pi ** 2 * c * t)
            * lib.sin(np.pi * X) * lib.sin(np.pi * Y))


def time_constant(c=1.0, L=1.0):
    """tau = L^2 / (pi^2 alpha). Steady state is reached at roughly 4 tau."""
    return L ** 2 / (np.pi ** 2 * c)


def fourier_number(t, c=1.0, L=1.0):
    """Dimensionless time. Fo >~ 1 means the transient is essentially over."""
    return c * t / L ** 2


def slice_at_time(model, t, k=161, domain=PLATE_DOMAIN, trial=None,
                  mask_hole=False, hole=None):
    """Evaluate a transient model on a ``k`` by ``k`` spatial grid at one instant.

    Returns ``(X, Y, T)``. Pass ``trial=`` for a hard-condition model, where
    the field is the trial solution rather than the raw network output.
    """
    h = hole or HOLE
    X, Y, pts = grid_points(k, k, domain, t=float(t))
    with torch.no_grad():
        xyt = to_tensor(pts)
        T = to_numpy(model(xyt) if trial is None else trial(model, xyt))
    T = T.reshape(X.shape)
    if mask_hole:
        T[_phi_flat(pts, h["xc"], h["yc"], h["a"], h["b"]).reshape(X.shape) < 0] = np.nan
    return X, Y, T


def error_vs_time(model, ts=None, c=1.0, trial=None, k=101,
                  domain=PLATE_DOMAIN):
    """Relative L2 error at a sequence of instants, plus the absolute error.

    Report BOTH. By t = 1 the exact amplitude is about 5e-9, so a relative
    error of 100% there may mean an absolute error of 1e-8. Where the
    reference has decayed into the noise the relative figure is returned as
    ``NaN`` rather than as a large number that means nothing.
    """
    ts = np.linspace(0.0, 1.0, 11) if ts is None else np.asarray(ts)
    rel, absol = [], []
    for t in ts:
        X, Y, U = slice_at_time(model, t, k=k, domain=domain, trial=trial)
        E = exact_transient(X, Y, t, c)
        good = np.linalg.norm(E) > 1e-12
        rel.append(relative_l2(U, E) if good else np.nan)
        absol.append(max_abs_error(U, E))
    return ts, np.array(rel), np.array(absol)


# ══════════════════════════════════════════════════════════════════════════
# 5 · reporting
# ══════════════════════════════════════════════════════════════════════════

def describe_problem(c=1.0, t_end=1.0) -> None:
    """Print both problems and the numbers a reader should sanity-check."""
    tau = time_constant(c)
    print("  THE SQUARE PLATE  (parabolic, one initial condition)")
    print(f"    domain          : {PLATE_DOMAIN}  x  t in (0, {t_end})")
    print(f"    diffusivity     : c = {c:g}")
    print("    edges           : T = 0 on all four, for all t")
    print("    initial field   : sin(pi x) sin(pi y)   (one mode, amplitude 1)")
    print(f"    time constant   : tau = {tau:.4f}"
          f"   steady by about 4 tau = {4 * tau:.3f}")
    print(f"    window          : {t_end / tau:.1f} time constants")
    print()
    print("    t        Fo       amplitude")
    for t in (0.0, 0.05, 0.1, 0.5, 1.0):
        print(f"    {t:<5}    {fourier_number(t, c):<6.3f}   "
              f"{np.exp(-2 * np.pi ** 2 * c * t):.3e}")
    print()
    a, b = HOLE["a"], HOLE["b"]
    perim = np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b)))
    print("  THE PLATE WITH A HOLE  (same equation, harder domain)")
    print(f"    hole            : ellipse at ({HOLE['xc']}, {HOLE['yc']}), "
          f"semi-axes {a} x {b}")
    print(f"    hole area       : {np.pi * a * b:.4f}"
          f"   ({np.pi * a * b * 100:.1f}% of the plate)")
    print(f"    hole perimeter  : {perim:.4f}   (Ramanujan)")
    print("    enforced by     : the level set itself, no trained multiplier")


def plot_slice(X, Y, values, ax=None, title="", label="T", cmap="magma",
               levels=50):
    """Filled contours of one spatial slice. ``NaN`` in the hole shows through."""
    import matplotlib.pyplot as plt
    ax = new_axes(ax, figsize=(5.0, 4.6))
    c = ax.contourf(X, Y, np.asarray(values), levels, cmap=cmap)
    ax.set_aspect("equal")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_title(title)
    plt.colorbar(c, ax=ax, label=label, shrink=0.85)
    return ax
