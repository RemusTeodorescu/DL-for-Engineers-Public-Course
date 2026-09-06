# ---------------------------------------------------------------------------
# GENERATED COPY - do not edit.
#
# The original is tools/pinn/course_core.py. Edit that and run
#     python3 tools/pinn/sync_cores.py
# Edits made here are silently overwritten the next time anyone does.
# ---------------------------------------------------------------------------

r"""Shared machinery for every exercise set in the course.

*Deep Learning for Engineering* — MSc, Aalborg University.

Remus Teodorescu (ret@et.aau.dk), with support from Research Assistant
Noman Khan (nomank@energy.aau.dk).

This module holds what every exercise set needs regardless of its physics: a
seed, a network, tensor conversion, the checking helpers and the reporting
helpers. Part 2 adds ``pinn_core.py`` on top of it; each set adds a
``problem.py`` of its own.

Before this existed, ``set_seed`` was defined fifteen times across the course
and ``grad`` ten times. One definition means one place to fix a bug, and it
means ``MLP`` in Ex_10 is demonstrably the same object as ``MLP`` in Ex_02.

Generated copy in each exercise folder — edit ``tools/pinn/course_core.py``
and run ``python3 tools/pinn/sync_cores.py``.
"""

from __future__ import annotations

import hashlib
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

__all__ = [
    "DEVICE", "SEED", "set_seed", "personal_seed",
    "to_tensor", "to_numpy",
    "MLP", "parameter_count",
    "mse", "check", "check_shape",
    "error_table", "plot_curves", "CYCLE", "new_axes",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 88


def set_seed(seed: int = SEED) -> None:
    """Seed NumPy and PyTorch together, so a run can be repeated.

    Report the seed with any result. A network's final loss varies between
    initialisations by more than most people expect, and a single number
    without its seed is not a measurement.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)


def personal_seed(student_number) -> int:
    """A seed of your own, derived from your study number.

    The notebooks fix the seed so the printed "what you should see" blocks are
    true on every machine. That is right for checking your work and wrong for
    reporting it: with one seed the whole cohort produces identical numbers.
    The report notebooks ask for numbers from this seed instead.

        SEED = personal_seed("20241234")

    Non-digits are ignored, so ``"aau-20241234"`` and ``20241234`` agree.

    SHA-256 rather than Python's built-in :func:`hash`, which is salted per
    process — ``hash("20241234")`` differs in every session, and a seed that
    changes between runs is not a seed.
    """
    digits = "".join(ch for ch in str(student_number) if ch.isdigit())
    if not digits:
        raise ValueError(
            "personal_seed needs a study number containing at least one "
            f"digit; got {student_number!r}")
    return int(hashlib.sha256(digits.encode("utf-8")).hexdigest()[:8], 16) % 100_000


def to_tensor(a, requires_grad: bool = False) -> torch.Tensor:
    """NumPy array to a tensor on :data:`DEVICE`, shaped ``(N, d)``.

    Set ``requires_grad=True`` for coordinates you will differentiate the
    network output with respect to — every set of collocation points. Omitting
    it is the most common reason an autograd call returns ``None``.

    A **1-D input becomes a column**, ``(N,) -> (N, 1)``, never a row. That
    distinction is not cosmetic: a per-bus vector of length 2 silently
    transposed to ``(1, 2)`` will broadcast against a coupling matrix and
    produce a plausible, wrong answer with no error anywhere.
    """
    a = np.asarray(a, dtype=float)
    if a.ndim == 0:
        a = a.reshape(1, 1)
    elif a.ndim == 1:
        a = a.reshape(-1, 1)
    return torch.as_tensor(a, device=DEVICE).requires_grad_(requires_grad)


def to_numpy(t: torch.Tensor) -> np.ndarray:
    """Detach a tensor to NumPy, whatever device and graph it is attached to."""
    return t.detach().cpu().numpy()


class MLP(nn.Module):
    """A fully connected network, ``tanh`` throughout, linear at the output.

    ``tanh`` and not ReLU, everywhere in this course. ReLU's second derivative
    is zero wherever it is defined, so a second-order PDE residual has nothing
    to work with. L4.1 slide 12 makes the argument eight weeks before Part 2
    needs it.

    Xavier initialisation: the default for ``nn.Linear`` is tuned for ReLU and
    leaves a deep tanh network's later layers close to saturated, which shows
    up as a loss that will not move for the first several hundred steps.
    """

    def __init__(self, n_in: int = 1, n_out: int = 1, n_hidden: int = 40,
                 n_layers: int = 4, activation=nn.Tanh):
        super().__init__()
        widths = [n_in] + [n_hidden] * n_layers + [n_out]
        layers: list[nn.Module] = []
        for i in range(len(widths) - 1):
            layers.append(nn.Linear(widths[i], widths[i + 1]))
            if i < len(widths) - 2:
                layers.append(activation())
        self.net = nn.Sequential(*layers)
        self.n_in, self.n_out = n_in, n_out
        self.n_hidden, self.n_layers = n_hidden, n_layers
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
        self.to(DEVICE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parameter_count(model: nn.Module) -> int:
    """Number of trainable numbers in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def mse(residual: torch.Tensor) -> torch.Tensor:
    """Mean of the squares. Every loss term in this course is one of these."""
    return torch.mean(residual ** 2)


def check(name: str, got, want, tol: float = 1e-10) -> bool:
    """Compare a computed value against the expected one and report.

    Prints the number that decided the verdict, which is more useful than a
    bare PASS.
    """
    g = np.asarray(got, dtype=float)
    w = np.asarray(want, dtype=float)
    if g.shape != w.shape:
        print(f"  FAIL  {name}: shape {g.shape}, expected {w.shape}")
        return False
    err = float(np.max(np.abs(g - w))) if g.size else 0.0
    verdict = "PASS" if err <= tol else "FAIL"
    print(f"  {verdict}  {name}: max abs error {err:.3e}  (tolerance {tol:.1e})")
    return err <= tol


def check_shape(name: str, got, want_shape: Sequence[int]) -> bool:
    """Compare an array's shape against the expected one and report."""
    g = np.asarray(got)
    want = tuple(int(v) for v in want_shape)
    verdict = "PASS" if g.shape == want else "FAIL"
    print(f"  {verdict}  {name}: shape {g.shape}, expected {want}")
    return g.shape == want


def error_table(rows, headers) -> str:
    """Format a small table as Markdown, for pasting into the report."""
    headers = list(headers)
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join([" --- "] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


CYCLE = ["#d94f2b", "#1f77b4", "#f4a300", "#0f9d58", "#7b61a8", "#00838f"]


def new_axes(ax=None, figsize=(7.2, 4.4)):
    """An axes to draw on, creating a figure only if one was not supplied."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    return ax


def plot_curves(history: Dict[str, np.ndarray], ax=None,
                title: str = "Training loss", logy: bool = True):
    """Loss curves on one pair of axes.

    A standing requirement of this course: every training run is reported with
    its curve, not with a final number alone.
    """
    ax = new_axes(ax)
    for i, (name, curve) in enumerate(history.items()):
        ax.plot(np.asarray(curve), lw=1.8, color=CYCLE[i % len(CYCLE)], label=name)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25, which="both")
    return ax
