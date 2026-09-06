# ---------------------------------------------------------------------------
# GENERATED COPY - do not edit.
#
# The original is tools/pinn/pinn_core.py. Edit that and run
#     python3 tools/pinn/sync_cores.py
# Edits made here are silently overwritten the next time anyone does.
# ---------------------------------------------------------------------------

r"""Physics-informed neural networks — the shared machinery for Part 2.

*Deep Learning for Engineering* — MSc, Aalborg University.
Remus Teodorescu (ret@et.aau.dk), with support from Research Assistant
Noman Khan (nomank@energy.aau.dk). The code here is original to this course;
see docs/PROVENANCE.md for what each reference text is cited for.

Written for *Deep Learning for Engineering*, Aalborg University.

Everything here is problem-independent: a network, the derivative helpers that
make a PDE residual expressible, samplers for a rectangular domain, a
two-stage optimiser and the error metrics. The physics lives in the notebooks.

## Where this comes from

The method is Raissi, Perdikaris & Karniadakis, *Physics-informed neural
networks*, J. Comput. Phys. **378** (2019) 686–707 — a network is made a
differentiable function of the coordinates, the PDE is imposed by penalising
its residual at sampled points, and automatic differentiation supplies the
derivatives. The two-stage Adam-then-L-BFGS schedule is the practice that grew
up around that paper.

The implementation is this course's own, written against that paper and the
PyTorch documentation. Names continue Part 1 rather than any textbook:
``MLP``, ``grad``, ``d2`` and ``set_seed`` mean here exactly what they meant in
``Ex_2_core.py``, so a student arriving from Ex_02 already knows four of them.

## Design decisions worth knowing

**float64 by default.** A second derivative of a network is a difference of
differences; in float32 the residual can be dominated by rounding long before
it is dominated by the model. The cost is roughly a factor of two in speed on
CPU and much more on most GPUs, which is why single precision is normal in
deep learning and wrong here.

**Samplers return NumPy, not tensors.** So they can be plotted, saved, diffed
and unit-tested without a device or a graph. Call :func:`to_tensor` at the
point of use.

**tanh, not ReLU.** ReLU's second derivative is zero wherever it is defined, so
a second-order residual has nothing to work with. L4.1 slide 12 makes this
argument eight weeks before it is needed; this is where it is needed.

Importing ``pinn_core`` pulls ``course_core`` in with it, so one line is
enough:

    from pinn_core import *
    set_seed(88)

    model = MLP(n_in=2, n_hidden=40, n_layers=4)
    xy    = interior_points(2000, DOMAIN_UNIT_SQUARE)

    def residual(model, xy):
        u = model(xy)
        return d2(u, xy, 0) + d2(u, xy, 1) + source(xy)

    history = train_two_stage(model, lambda: total_loss(model))
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

# Everything the whole course shares lives one layer down. Re-exported here so
# a Part 2 notebook needs a single import line.
from course_core import (DEVICE, SEED, set_seed, personal_seed,
                         to_tensor, to_numpy, MLP, parameter_count, mse,
                         check, check_shape, error_table, plot_curves,
                         CYCLE, new_axes)

torch.set_default_dtype(torch.float64)

__all__ = [
    "DEVICE", "SEED", "set_seed", "personal_seed", "to_tensor", "to_numpy",
    "MLP", "parameter_count", "mse", "check", "check_shape",
    "error_table", "plot_curves", "CYCLE", "new_axes",
    "describe", "pseudo_dimension",
    "grad", "d2",
    "DOMAIN_UNIT_SQUARE", "interior_points", "boundary_points",
    "spacetime_points", "initial_points", "boundary_points_in_time",
    "grid_points", "latin_hypercube",
    "train_two_stage",
    "relative_l2", "max_abs_error",
]


# ──────────────────────────────────────────────────────────────────────────
# 1 · the network
# ──────────────────────────────────────────────────────────────────────────

def describe(model: "MLP", n_collocation: Optional[int] = None) -> int:
    """Print the model's size and, if given, the collocation-to-parameter ratio.

    The ratio is the number worth watching. A network with more parameters than
    the residual has sample points can satisfy the equations at every point you
    tested and do anything at all between them — the PINN version of fitting
    noise, except there is no held-out set to catch it. Report it.

    Returns the parameter count.
    """
    p = parameter_count(model)
    p_star = pseudo_dimension(model)
    print(f"  network        : {model.n_in} -> "
          f"{model.n_hidden} x {model.n_layers} -> {model.n_out}   (tanh)")
    print(f"  parameters     : {p}")
    print(f"  P*             : {p_star}   (the capacity measure L7.1 uses)")
    if n_collocation is not None:
        print(f"  collocation    : {n_collocation}"
              f"   =  {n_collocation / p_star:.1f} x P*"
              f"   =  {n_collocation / p:.2f} x parameters")
        if n_collocation < p_star:
            print("                   BELOW P* — undersampled, see L7.1")
    print(f"  device         : {DEVICE}   dtype: {torch.get_default_dtype()}")
    return p


def pseudo_dimension(model: "MLP") -> int:
    """A capacity measure for a fully connected network, ``(p+1) + (N+1)L``.

    Reported alongside the parameter count because the two disagree by more
    than an order of magnitude and the course uses both. The parameter count
    is what the optimiser must fit; ``P*`` is the sampling target L7.1 sets,
    and a collocation set below it is undersampled whatever the parameter
    ratio says.

    Printing only one of them is what made Ex_08.1 and Ex_07.1 appear to
    contradict each other. They do not; they were quoting different
    denominators.
    """
    return (model.n_in + 1) + (model.n_hidden + 1) * model.n_layers


# ──────────────────────────────────────────────────────────────────────────
# 2 · derivatives
# ──────────────────────────────────────────────────────────────────────────

def grad(u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """∂u/∂x for a scalar field, returned with the same shape as ``x``.

    Column ``i`` is the derivative with respect to input ``i``, so for a
    space–time problem sampled as ``(x, y, t)`` the time derivative is
    ``grad(u, xyt)[:, 2:3]``.

    ``create_graph=True`` keeps the result differentiable, which is what makes
    :func:`d2` possible. Without it a second call returns ``None``.
    """
    return torch.autograd.grad(
        u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]


def d2(u: torch.Tensor, x: torch.Tensor, i: int) -> torch.Tensor:
    """The second derivative ∂²u/∂xᵢ², shaped ``(N, 1)``.

    Two nested calls, taking the ``i``-th column in between. Written out rather
    than hidden so that the cost is visible: a second-order residual in three
    inputs builds a graph three times the depth of the forward pass, and that,
    not the network size, is what makes PINNs slow.
    """
    first = grad(u, x)[:, i:i + 1]
    return grad(first, x)[:, i:i + 1]


# ──────────────────────────────────────────────────────────────────────────
# 3 · sampling a rectangular domain
# ──────────────────────────────────────────────────────────────────────────

#: ``((x_lo, x_hi), (y_lo, y_hi))`` — the default domain, the unit square.
DOMAIN_UNIT_SQUARE = ((0.0, 1.0), (0.0, 1.0))


def latin_hypercube(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` points in the unit ``d``-cube, one per stratum along every axis.

    Plain uniform sampling leaves gaps and clusters that a residual notices;
    stratifying each axis and shuffling the strata independently spreads the
    points at the same cost. Returns shape ``(n, d)`` in ``[0, 1)``.
    """
    cut = (np.arange(n)[:, None] + rng.random((n, d))) / n
    for j in range(d):
        rng.shuffle(cut[:, j])
    return cut


def _scale(unit: np.ndarray, domain: Sequence[Sequence[float]]) -> np.ndarray:
    lo = np.array([a for a, _ in domain], dtype=float)
    hi = np.array([b for _, b in domain], dtype=float)
    return lo + unit * (hi - lo)


def interior_points(n: int, domain=DOMAIN_UNIT_SQUARE, method: str = "lhs",
                    seed: Optional[int] = None) -> np.ndarray:
    """``n`` collocation points strictly inside ``domain``.

    ``method`` is ``"lhs"`` (stratified, the default) or ``"uniform"``.
    Returns ``(n, d)`` NumPy; wrap with :func:`to_tensor` with
    ``requires_grad=True`` before differentiating through it.
    """
    rng = np.random.default_rng(seed)
    d = len(domain)
    unit = (latin_hypercube(n, d, rng) if method == "lhs"
            else rng.random((n, d)))
    return _scale(unit, domain)


def boundary_points(n_per_edge: int, domain=DOMAIN_UNIT_SQUARE,
                    seed: Optional[int] = None) -> np.ndarray:
    """Points on the four edges of a rectangle, ``n_per_edge`` on each.

    Corners are included twice, once from each edge that meets there. That is
    deliberate: a corner is where two boundary conditions must agree, and
    weighting it twice is a cheap way of saying so.
    """
    rng = np.random.default_rng(seed)
    (x0, x1), (y0, y1) = domain[0], domain[1]
    s = rng.random(n_per_edge)
    x = x0 + s * (x1 - x0)
    t = rng.random(n_per_edge)
    y = y0 + t * (y1 - y0)
    edges = [np.stack([x, np.full(n_per_edge, y0)], axis=1),
             np.stack([x, np.full(n_per_edge, y1)], axis=1),
             np.stack([np.full(n_per_edge, x0), y], axis=1),
             np.stack([np.full(n_per_edge, x1), y], axis=1)]
    return np.concatenate(edges, axis=0)


def spacetime_points(n: int, domain=DOMAIN_UNIT_SQUARE,
                     t_span: Tuple[float, float] = (0.0, 1.0),
                     method: str = "lhs",
                     seed: Optional[int] = None) -> np.ndarray:
    """``n`` points in the space–time slab, shaped ``(n, d + 1)``.

    Time is the **last** column, by convention, everywhere in this course.
    """
    return interior_points(n, tuple(domain) + (t_span,), method, seed)


def initial_points(n: int, domain=DOMAIN_UNIT_SQUARE, t0: float = 0.0,
                   method: str = "lhs",
                   seed: Optional[int] = None) -> np.ndarray:
    """``n`` points on the ``t = t0`` slice, shaped ``(n, d + 1)``."""
    xy = interior_points(n, domain, method, seed)
    return np.concatenate([xy, np.full((len(xy), 1), float(t0))], axis=1)


def boundary_points_in_time(n_per_edge: int, n_times: int,
                            domain=DOMAIN_UNIT_SQUARE,
                            t_span: Tuple[float, float] = (0.0, 1.0),
                            seed: Optional[int] = None) -> np.ndarray:
    """The spatial boundary, repeated at ``n_times`` instants."""
    rng = np.random.default_rng(seed)
    xy = boundary_points(n_per_edge, domain, seed)
    times = np.linspace(t_span[0], t_span[1], n_times)
    return np.concatenate(
        [np.concatenate([xy, np.full((len(xy), 1), t)], axis=1) for t in times],
        axis=0)


def grid_points(nx: int, ny: int, domain=DOMAIN_UNIT_SQUARE,
                t: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A regular ``nx`` by ``ny`` grid, for plotting and for scoring error.

    Returns ``(X, Y, points)`` — the two meshgrid arrays for ``contourf``, and
    the same points flattened to ``(nx * ny, d)`` ready for the model. Pass
    ``t`` to append a constant time column.

    Never train on this. A grid is the wrong sample for a residual and the
    right one for a picture.
    """
    (x0, x1), (y0, y1) = domain[0], domain[1]
    X, Y = np.meshgrid(np.linspace(x0, x1, nx), np.linspace(y0, y1, ny))
    pts = np.stack([X.ravel(), Y.ravel()], axis=1)
    if t is not None:
        pts = np.concatenate([pts, np.full((len(pts), 1), float(t))], axis=1)
    return X, Y, pts


# ──────────────────────────────────────────────────────────────────────────
# 4 · training
# ──────────────────────────────────────────────────────────────────────────

def train_two_stage(model: nn.Module,
                    loss_fn: Callable[[], torch.Tensor],
                    adam_steps: int = 2000,
                    lbfgs_steps: int = 200,
                    lr: float = 1e-3,
                    report_every: int = 500) -> Dict[str, np.ndarray]:
    """Adam first, then L-BFGS. The recipe every exercise in Part 2 uses.

    ``loss_fn`` takes no arguments and returns the total loss as a scalar
    tensor — it closes over the model and whatever points it needs, so this
    function never has to know how many terms the loss has.

    Why two stages, in one line each: Adam is robust a long way from a solution
    and slow to polish; L-BFGS uses curvature to polish quickly and is
    unreliable far from a minimum. Ex_06 notebook 02 measures both claims.

    L-BFGS is full-batch by construction. It calls ``loss_fn`` several times
    per step during its line search, so the points must not change between
    calls — resample before the stage, never inside it.

    Returns ``{"adam": ..., "lbfgs": ...}``, the loss at every step.
    """
    history: Dict[str, list] = {"adam": [], "lbfgs": []}

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for step in range(adam_steps):
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        opt.step()
        history["adam"].append(loss.item())
        if report_every and step % report_every == 0:
            print(f"    adam  {step:>6d}   loss {loss.item():.6e}")

    opt = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=20,
                            history_size=50,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        return loss

    for step in range(lbfgs_steps):
        opt.step(closure)
        # NOT under torch.no_grad(). A PDE residual differentiates the network
        # with respect to its inputs, so evaluating the loss without building
        # a graph raises "does not require grad and does not have a grad_fn".
        # Recording the curve costs one extra forward-and-backward per step;
        # a silent crash in every Part 2 exercise would cost rather more.
        history["lbfgs"].append(float(loss_fn().detach()))
        if report_every and step % max(1, report_every // 10) == 0:
            print(f"    lbfgs {step:>6d}   loss {history['lbfgs'][-1]:.6e}")

    last = (history["lbfgs"] or history["adam"])
    if last:
        print(f"  final loss: {last[-1]:.6e}")
    else:
        print("  no steps were run — both adam_steps and lbfgs_steps were zero")
    return {k: np.asarray(v) for k, v in history.items()}


# ──────────────────────────────────────────────────────────────────────────
# 5 · scoring
# ──────────────────────────────────────────────────────────────────────────

def relative_l2(prediction, reference) -> float:
    """‖prediction − reference‖₂ / ‖reference‖₂, as a plain float.

    The standard PINN error measure, and dimensionless, so it can be compared
    between problems. It is undefined for a reference that is identically zero
    and misleading for one that is nearly so — say which you have.
    """
    p = np.asarray(prediction, dtype=float).ravel()
    r = np.asarray(reference, dtype=float).ravel()
    denom = np.linalg.norm(r)
    if denom == 0.0:
        raise ValueError("relative_l2 needs a reference that is not all zero")
    return float(np.linalg.norm(p - r) / denom)


def max_abs_error(prediction, reference) -> float:
    """The largest single error, in the units of the field.

    Quote it beside :func:`relative_l2`. A good norm hides a bad corner, and
    the corner is usually where the engineering is.
    """
    p = np.asarray(prediction, dtype=float).ravel()
    r = np.asarray(reference, dtype=float).ravel()
    return float(np.max(np.abs(p - r)))
