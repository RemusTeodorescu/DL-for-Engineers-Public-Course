r"""Ex_08.1 — the physics: stationary conduction on a plate with a hole.

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

A square plate with a cooling hole through it, in steady state with uniform
internal heat generation:

    -k (T_xx + T_yy) = Q        in the material
    T = 0                       on the hole wall
    ∂T/∂n = 0                   on the four outer edges

Nothing in the *equation* is new — it is a Poisson problem, as in Ex_07.1.
What is new is **geometry**. The domain is not a rectangle, so the samplers in
``pinn_core`` do not describe it; the boundary that carries the interesting
condition is curved; and the outward normal there is a function of position
rather than a constant per edge.

## What this module adds

Four things, and they are the four things any non-rectangular PINN needs.

1. **A level set.** :func:`ellipse_phi` is negative inside the hole, zero on it
   and positive in the material. One formula answers "is this point in the
   domain?" and "how far in?" at the same time.
2. **A hard-enforcement multiplier.** :func:`hole_multiplier` is that same
   level set, used as a factor: a network multiplied by it *cannot* be non-zero
   on the hole wall, whatever it learns. This is the analytic shortcut of L8.1
   slide 15 — for a shape with a closed-form level set, no multiplier network
   needs training.
3. **Samplers that respect the geometry.** Rejection sampling for the interior;
   arc-length spacing on the ellipse, because uniform spacing *in the angle*
   starves the high-curvature ends, which is exactly where the flux
   concentrates.
4. **A flux balance.** In steady state the heat leaving through the hole must
   equal the heat generated in the plate. That is a check which has to come out
   right for physical reasons, and it is worth more than an error norm that
   looks small.

## The manufactured solution

Before any geometry, notebook 01 verifies the formulation on the unit square,
where the answer is known:

    T(x, y) = sin(πx) sin(πy)     ⇒     Q/k = 2π² sin(πx) sin(πy)

Zero on all four edges of the unit square, so the polynomial lift
``x(1-x)y(1-y)`` enforces the boundary condition exactly. An implementation
error found here costs minutes; the same error found on the plate costs an
afternoon.

## Conventions

**The samplers here return NumPy, like the ones in ``pinn_core``.** Points can
then be plotted, saved, diffed and checked without a device or a graph. Call
:func:`to_tensor` at the point of use — with ``requires_grad=True`` for every
set the network is differentiated at, which here means the interior points,
the outer edges (the flux loss differentiates there) and the hole points passed
to :func:`flux_balance`. Forgetting it is the single most common first error.

:func:`ellipse_phi` and :func:`hole_multiplier` are plain arithmetic and work
on NumPy arrays and on torch tensors unchanged. :func:`ellipse_normal`,
:func:`source_manufactured`, :func:`flux_balance` and :func:`eval_grid_masked`
are torch; :func:`exact_manufactured` is NumPy. The docstrings say which.

Nothing at module level imports torch, so the geometry can be checked in an
environment that has only NumPy.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "HOLE", "DOMAIN",
    "ellipse_phi", "hole_multiplier", "ellipse_normal",
    "sample_plate_with_hole", "sample_ellipse_boundary", "sample_outer_edges",
    "exact_manufactured", "source_manufactured",
    "hole_perimeter", "plate_area",
    "flux_balance", "eval_grid_masked", "describe_problem",
]

#: The cooling hole — an ellipse, centred, wider than it is tall.
HOLE = dict(xc=0.5, yc=0.5, a=0.18, b=0.11)

#: ``((x_lo, x_hi), (y_lo, y_hi))`` — the plate, before the hole is removed.
DOMAIN = ((0.0, 1.0), (0.0, 1.0))


# ------------------------------------------------------------------- geometry
def _phi_np(xy, xc, yc, a, b):
    """The level set on a flat ``(N, 2)`` NumPy array. Used by the samplers."""
    return ((xy[:, 0] - xc) / a) ** 2 + ((xy[:, 1] - yc) / b) ** 2 - 1.0


def ellipse_phi(xy, hole=None):
    """Level-set function: negative inside the hole, zero on it, positive out.

    Returns shape ``(N, 1)``. Plain arithmetic, so NumPy in gives NumPy out and
    a tensor in gives a tensor out — the trial solution calls it on tensors,
    notebook 00 calls it on arrays, and it is the same function.
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

    Torch — pass ``to_tensor(...)`` points. Returns ``(nx, ny)``, each shaped
    ``(N, 1)``.

    Sign matters: the material's outward normal is the inward normal of the
    hole. Getting it backwards produces a plausible but inverted field.
    """
    import torch
    h = hole or HOLE
    gx = 2.0 * (xy[:, 0:1] - h["xc"]) / h["a"] ** 2
    gy = 2.0 * (xy[:, 1:2] - h["yc"]) / h["b"] ** 2
    n = torch.sqrt(gx ** 2 + gy ** 2) + 1e-12
    return -gx / n, -gy / n          # minus: out of the material


# ------------------------------------------------------------------- sampling
def sample_plate_with_hole(n, domain=DOMAIN, hole=None, margin=1.05,
                           seed=None) -> np.ndarray:
    """``n`` interior points on the plate, rejecting anything inside the hole.

    Rejection sampling on top of :func:`pinn_core.interior_points`: draw a
    stratified batch on the whole rectangle, keep what lands in the material,
    repeat until there are enough. ``margin`` inflates the rejected ellipse
    slightly, so no collocation point sits on the hole wall where the
    multiplier — and with it the trial solution — is identically zero.

    Returns ``(n, 2)`` NumPy. Wrap with ``to_tensor(..., requires_grad=True)``
    before differentiating the network at these points.
    """
    from course_core import SEED
    from pinn_core import interior_points

    h = hole or HOLE
    rng = np.random.default_rng(SEED if seed is None else seed)
    keep = np.empty((0, 2))
    while len(keep) < n:
        cand = interior_points(max(2 * n, 512), domain, "lhs",
                               seed=int(rng.integers(0, 2 ** 31 - 1)))
        ok = _phi_np(cand, h["xc"], h["yc"], h["a"] * margin, h["b"] * margin) > 0
        keep = np.vstack([keep, cand[ok]])
    return keep[:n]


def sample_ellipse_boundary(n, hole=None) -> np.ndarray:
    """Points on the hole, spaced by arc length rather than by angle.

    Uniform angular spacing under-samples the high-curvature ends of an
    ellipse, which is where the flux concentrates. Notebook 00 measures the
    difference between the two spacings.

    Returns ``(n, 2)`` NumPy, and is deterministic — there is no randomness
    here to seed.
    """
    h = hole or HOLE
    t = np.linspace(0, 2 * np.pi, 4000)
    x, y = h["a"] * np.cos(t), h["b"] * np.sin(t)
    s = np.concatenate([[0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    tt = np.interp(np.linspace(0, s[-1], n, endpoint=False), s, t)
    return np.column_stack([h["xc"] + h["a"] * np.cos(tt),
                            h["yc"] + h["b"] * np.sin(tt)])


def sample_outer_edges(n_per_edge, domain=DOMAIN, seed=None) -> np.ndarray:
    """Points on the four outer edges of the plate, ``n_per_edge`` on each.

    A thin wrapper on :func:`pinn_core.boundary_points`, so that the three
    point sets of this problem are named symmetrically in the notebooks. The
    seed defaults to the course seed rather than to ``None``, so that a plain
    ``sample_outer_edges(30)`` is reproducible.

    Returns ``(4 * n_per_edge, 2)`` NumPy. The flux condition differentiates
    the network here, so wrap with ``requires_grad=True``.
    """
    from course_core import SEED
    from pinn_core import boundary_points
    return boundary_points(n_per_edge, domain,
                           seed=SEED if seed is None else seed)


# ------------------------------------------------- manufactured verification
def exact_manufactured(X, Y):
    """T = sin(pi x) sin(pi y) — the standard manufactured field of L7.

    NumPy, and meant for a grid: pass the ``X`` and ``Y`` returned by
    :func:`pinn_core.grid_points`.
    """
    return np.sin(np.pi * X) * np.sin(np.pi * Y)


def source_manufactured(xy):
    """Q/k that makes exact_manufactured the solution of T_xx + T_yy + Q/k = 0.

    Torch — it is evaluated inside the residual, on the collocation tensor.
    Returns shape ``(N, 1)``.
    """
    import torch
    return (2.0 * np.pi ** 2
            * torch.sin(np.pi * xy[:, 0:1]) * torch.sin(np.pi * xy[:, 1:2]))


# -------------------------------------------------------------------- checks
def hole_perimeter(hole=None) -> float:
    """Ramanujan's approximation to the perimeter of the elliptical hole.

    Exact enough here — its relative error for an ellipse this round is far
    below anything a trained network will contribute to the flux balance.
    """
    h = hole or HOLE
    return float(np.pi * (3 * (h["a"] + h["b"])
                          - np.sqrt((3 * h["a"] + h["b"])
                                    * (h["a"] + 3 * h["b"]))))


def plate_area(hole=None) -> float:
    """Area of the unit plate with the elliptical hole removed."""
    h = hole or HOLE
    return float(1.0 - np.pi * h["a"] * h["b"])


def flux_balance(model, xy_hole, q_source, k=1.0, hole=None, trial=None):
    """Heat leaving through the hole versus heat generated inside the plate.

    In steady state these must match. A mismatch of more than a few percent
    means the flux condition was never really learned.

    ``xy_hole`` must be a tensor made with ``requires_grad=True``: the normal
    flux is a derivative of the network, so the points are differentiated
    through. Returns ``(out, generated)``.
    """
    from pinn_core import grad

    h = hole or HOLE
    T = model(xy_hole) if trial is None else trial(model, xy_hole)
    g = grad(T, xy_hole)
    nx, ny = ellipse_normal(xy_hole, h)
    q_n = -k * (g[:, 0:1] * nx + g[:, 1:2] * ny)
    perim = hole_perimeter(h)
    out = float(q_n.mean().item()) * perim
    area = plate_area(h)
    return out, q_source * area


def eval_grid_masked(model, k=161, hole=None, trial=None):
    """Evaluate on a grid, masking points inside the hole with NaN.

    Returns ``(X, Y, T)``, with ``T`` shaped like ``X`` so it can go straight
    into ``contourf``. The mask uses the *un*-inflated ellipse, so the picture
    shows the hole at its true size even though the collocation points were
    kept clear of it by ``margin``.
    """
    import torch
    from course_core import to_numpy, to_tensor
    from pinn_core import grid_points

    h = hole or HOLE
    X, Y, pts = grid_points(k, k, DOMAIN)
    xy = to_tensor(pts)
    with torch.no_grad():
        T = to_numpy(model(xy) if trial is None else trial(model, xy))
    T = T.reshape(X.shape)
    inside = _phi_np(pts, h["xc"], h["yc"], h["a"], h["b"]).reshape(X.shape) < 0
    T[inside] = np.nan
    return X, Y, T


def describe_problem() -> None:
    """Print the geometry and the numbers it implies. NumPy only."""
    h = HOLE
    hole_area = float(np.pi * h["a"] * h["b"])
    print("  plate           : 1.000 x 1.000   (unit square)")
    print(f"  hole            : ellipse at ({h['xc']:.2f}, {h['yc']:.2f}),"
          f"  a = {h['a']:.2f},  b = {h['b']:.2f}")
    print(f"  hole area       : {hole_area:.5f}"
          f"   = {100 * hole_area:.2f}% of the plate")
    print(f"  material area   : {plate_area():.5f}")
    print(f"  hole perimeter  : {hole_perimeter():.5f}   (Ramanujan)")
    print(f"  aspect ratio a/b: {h['a'] / h['b']:.3f}"
          f"   -> the ends are {(h['a'] / h['b']) ** 3:.2f}x as curved"
          f" as the sides")
    print(f"  generated heat  : Q/k x {plate_area():.5f}"
          f"   must leave through {hole_perimeter():.5f} of wall")
