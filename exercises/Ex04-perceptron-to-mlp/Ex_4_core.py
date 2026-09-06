r"""Shared code for Ex_04 — Perceptron to MLP.

*Deep Learning for Engineering* — MSc, Aalborg University.
Remus Teodorescu (ret@et.aau.dk), with support from Research Assistant
Noman Khan (nomank@energy.aau.dk). The code here is original to this course;
see docs/PROVENANCE.md for what each reference text is cited for.

    Ex04_00_environment_check.ipynb        # 0 — check the tools, run this first
    Ex04_01_perceptron_and_xor.ipynb       # 1 — NumPy perceptron; AND, then XOR
    Ex04_02_pytorch_mlp.ipynb              # 2 — the same network, in PyTorch
    Ex04_03_counting_kinks.ipynb           # 3 — sweep D, count the kinks
    Ex04_04_depth_vs_width.ipynb           # 4 — one budget, spent three ways
    Ex04_05_overfit_then_regularise.ipynb  # 5 — the lecture's eleven points
    Ex04_06_report.ipynb                   # 6 — the report

This module is complete. You are not expected to change anything in it. Your
work is in the `# TODO:` cells of the notebooks.

Three datasets live here:

* the four rows of a logic gate, for the perceptron in notebook 01;
* a wiggly one-dimensional curve, for the capacity experiments in notebooks 03
  and 04;
* **the eleven noisy samples from L4.2 slides 11-13**, reproduced exactly —
  same truth, same eleven noise values — so that the overfitting lab in
  notebook 05 draws the figure the lecture drew.

Everything uses `torch`, `numpy` and `matplotlib` only, on a CPU, in seconds.
"""

from __future__ import annotations

import copy
import os
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn


SEED = 0

#: Where notebooks write anything a later notebook reads.
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "Ex04_outputs")


def set_seed(seed: int = SEED) -> None:
    """Seed NumPy and PyTorch together, so a run can be repeated."""
    np.random.seed(seed)
    torch.manual_seed(seed)

def personal_seed(student_number) -> int:
    """A seed of your own, derived from your study number.

    Every notebook in this course fixes the seed to 0 so that the printed
    "what you should see" blocks are true on every machine. That is right for
    checking your work and wrong for reporting it: with one seed, every student
    in the cohort produces identical numbers, and a report can be copied
    without leaving a trace.

    So the report asks for numbers from **your** seed:

        SEED = core.personal_seed("20241234")   # your AAU study number

    Non-digits are ignored, so ``"aau-20241234"`` and ``20241234`` agree.

    Uses SHA-256 rather than Python's built-in :func:`hash`, which is salted
    per process — ``hash("20241234")`` gives a different answer in every
    session, and a seed that changes between runs is not a seed.
    """
    import hashlib
    digits = "".join(ch for ch in str(student_number) if ch.isdigit())
    if not digits:
        raise ValueError(
            "personal_seed needs a study number containing at least one digit; "
            f"got {student_number!r}")
    return int(hashlib.sha256(digits.encode("utf-8")).hexdigest()[:8], 16) % 100_000


# ──────────────────────────────────────────────────────────────────────────
# 1 · logic gates — notebooks 01 and 02
# ──────────────────────────────────────────────────────────────────────────

_GATES = {
    "AND":  [0.0, 0.0, 0.0, 1.0],
    "OR":   [0.0, 1.0, 1.0, 1.0],
    "NAND": [1.0, 1.0, 1.0, 0.0],
    "XOR":  [0.0, 1.0, 1.0, 0.0],
}

#: The four input rows, in the order (0,0), (0,1), (1,0), (1,1).
LOGIC_INPUTS = np.array([[0.0, 0.0],
                         [0.0, 1.0],
                         [1.0, 0.0],
                         [1.0, 1.0]])


def logic_dataset(gate: str = "AND") -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(X, y)`` for a two-input logic gate.

    ``X`` has shape (4, 2) and ``y`` shape (4,), both float. The whole dataset
    is four rows: this is the smallest interesting classification problem there
    is, and XOR is the smallest one a single neuron cannot solve.
    """
    gate = gate.upper()
    if gate not in _GATES:
        raise ValueError(f"gate must be one of {sorted(_GATES)}")
    return LOGIC_INPUTS.copy(), np.array(_GATES[gate])


def truth_table(y_pred: np.ndarray, y_true: np.ndarray,
                title: str = "") -> None:
    """Print the four rows, the prediction and the target, side by side."""
    if title:
        print(title)
    print("  x1  x2   target   predicted")
    for i in range(4):
        mark = " " if y_pred[i] == y_true[i] else "   <-- wrong"
        print(f"  {LOGIC_INPUTS[i, 0]:.0f}   {LOGIC_INPUTS[i, 1]:.0f}"
              f"      {y_true[i]:.0f}         {y_pred[i]:.0f}{mark}")
    print(f"  {int(np.sum(y_pred == y_true))} of 4 correct")


def plot_logic(X: np.ndarray, y: np.ndarray, ax=None, title: str = ""):
    """Scatter the four rows of a logic problem: filled = 1, hollow = 0."""
    ax = _new_axes(ax, figsize=(4.6, 4.4))
    for i in range(len(y)):
        if y[i] > 0.5:
            ax.plot(X[i, 0], X[i, 1], "o", ms=13, color="#1f77b4", zorder=4)
        else:
            ax.plot(X[i, 0], X[i, 1], "o", ms=13, mfc="white",
                    mec="#d94f2b", mew=2.2, zorder=4)
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylim(-0.35, 1.35)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.set_aspect("equal")
    return ax


def plot_decision_line(ax, w: Sequence[float], b: float,
                       colour: str = "#0f9d58", label: Optional[str] = None):
    """Draw the line w1 x1 + w2 x2 + b = 0 on an existing logic plot.

    This is what a single perceptron computes: everything on one side of this
    line is classified 1, everything on the other side 0.
    """
    w = np.asarray(w, dtype=float).ravel()
    xs = np.linspace(-0.35, 1.35, 10)
    if abs(w[1]) > 1e-9:
        ax.plot(xs, -(w[0] * xs + b) / w[1], lw=2.0, color=colour, label=label)
    elif abs(w[0]) > 1e-9:
        ax.axvline(-b / w[0], lw=2.0, color=colour, label=label)
    return ax


def plot_boundary(ax, predict: Callable[[np.ndarray], np.ndarray],
                  n: int = 220):
    """Shade the region a classifier calls 1.

    ``predict`` takes an array of shape (N, 2) and returns N values; anything
    above 0.5 counts as class 1.
    """
    g = np.linspace(-0.35, 1.35, n)
    xx, yy = np.meshgrid(g, g)
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1)
    zz = np.asarray(predict(grid), dtype=float).reshape(xx.shape)
    ax.contourf(xx, yy, zz, levels=[-1e9, 0.5, 1e9],
                colors=["#fdece7", "#e6f0f8"], zorder=0)
    ax.contour(xx, yy, zz, levels=[0.5], colors=["#555555"], linewidths=1.2,
               zorder=1)
    return ax


def plot_errors(errors: Sequence[float], ax=None,
                title: str = "Misclassified rows per epoch"):
    """Errors against epoch for the perceptron learning rule.

    A converged perceptron reaches zero and stays there. One that cannot
    represent the problem never does, and the shape of *that* curve is the
    point of notebook 01.
    """
    ax = _new_axes(ax, figsize=(6.6, 3.6))
    ax.plot(np.arange(len(errors)), errors, lw=1.6, color="#d94f2b")
    ax.set_ylim(-0.2, 4.2)
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_xlabel("epoch")
    ax.set_ylabel("rows wrong (of 4)")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    return ax


# ──────────────────────────────────────────────────────────────────────────
# 2 · the regression targets — notebooks 03, 04 and 05
# ──────────────────────────────────────────────────────────────────────────

#: The domain of the capacity target. Centred on zero deliberately: a network
#: trains noticeably better on inputs that are centred, and Ex_04 does not want
#: to spend a notebook on input scaling.
WIGGLY_DOMAIN = (-1.0, 1.0)


def wiggly_truth(x: np.ndarray) -> np.ndarray:
    """A smooth curve with two length scales, on [-1, 1].

    Used for the capacity experiments. It is deliberately harder than a single
    hump: a network needs several kinks before it can follow the small ripple
    as well as the large oscillation, which is what makes the sweep in notebook
    03 worth plotting.
    """
    x = np.asarray(x, dtype=float)
    return np.sin(np.pi * x) + 0.35 * np.sin(3.0 * np.pi * x)


def wiggly_dataset(n: int = 400, noise: float = 0.06,
                   seed: int = 7) -> Tuple[np.ndarray, np.ndarray]:
    """``n`` samples of :func:`wiggly_truth` on [-1, 1], evenly spaced."""
    x = np.linspace(*WIGGLY_DOMAIN, n)
    rng = np.random.default_rng(seed)
    y = wiggly_truth(x) + rng.normal(0.0, noise, size=x.shape)
    return x, y


# ── the lecture's own dataset ─────────────────────────────────────────────
# L4.2 slides 11-13 fit one dataset three ways: too rigid, about right, too
# flexible. These are its eleven points, copied from the generator
# (lectures/L04.2-deep-neural-networks/src/L4.2_generator.js) so that notebook
# 05 draws the same picture on the student's own screen.

#: The eleven noise values, verbatim from the lecture generator.
LECTURE_NOISE = np.array([0.06, -0.09, 0.05, 0.11, -0.07, 0.02,
                          -0.10, 0.08, -0.04, 0.09, -0.06])

#: Standard deviation of the lecture's noise, used for the held-out set.
LECTURE_NOISE_SD = 0.07


def lecture_truth(x: np.ndarray) -> np.ndarray:
    """The smooth process behind the lecture's eleven samples.

    ``0.30 + 0.52 sin(2.9 x) + 0.10 x`` — the grey dashed curve on L4.2 slide
    11, which the lecture points out you never see in real life.
    """
    x = np.asarray(x, dtype=float)
    return 0.30 + 0.52 * np.sin(2.9 * x) + 0.10 * x


def lecture_dataset() -> Tuple[np.ndarray, np.ndarray]:
    """The eleven points from L4.2 slides 11-13, exactly.

    ``x_i = i / 10`` for i = 0 … 10, and ``y_i = truth(x_i) + noise_i`` with the
    eleven noise values fixed in the lecture generator. No random number
    generator is involved: these are the same eleven numbers on every machine.
    """
    x = np.arange(11) / 10.0
    y = lecture_truth(x) + LECTURE_NOISE
    return x, y


def lecture_validation(n: int = 40, seed: int = 11,
                       noise: float = LECTURE_NOISE_SD
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """A held-out set from the same process — a second measurement campaign.

    Same truth, same instrument, different samples. This is the data the
    lecture keeps back on slide 14, and it is the only thing that can tell you
    that the flexible model on slide 13 is worse than the one on slide 12.
    """
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0.0, 1.0, size=n))
    y = lecture_truth(x) + rng.normal(0.0, noise, size=x.shape)
    return x, y


# ──────────────────────────────────────────────────────────────────────────
# 3 · networks
# ──────────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    """A fully connected network with one input and one output.

    ``MLP(hidden=(64, 64), activation="tanh")`` is two hidden layers of 64
    units with tanh activations. ``hidden=(D,)`` is the shallow network of L4.1
    slides 13-15 — one hidden layer of D units.

    ``activation="relu"`` gives a piecewise linear function, which is the right
    choice when you want to see kinks. ``activation="tanh"`` gives a smooth one,
    which is the right choice for notebook 05 and for every network in Part 2
    of this course.
    """

    def __init__(self, hidden: Sequence[int] = (64,),
                 activation: str = "relu",
                 n_in: int = 1, n_out: int = 1) -> None:
        super().__init__()
        acts = {"relu": nn.ReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid}
        if activation not in acts:
            raise ValueError(f"activation must be one of {sorted(acts)}")
        act = acts[activation]

        layers: list = []
        prev = n_in
        for width in hidden:
            layers += [nn.Linear(prev, width), act()]
            prev = width
        layers += [nn.Linear(prev, n_out)]
        self.net = nn.Sequential(*layers)
        self.hidden = tuple(hidden)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def count_parameters(model: nn.Module) -> int:
    """Number of trainable numbers in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def parameter_count(n_in: int, hidden: Sequence[int], n_out: int = 1) -> int:
    """Parameters of an MLP with these layer sizes, without building it.

    Each ``Linear(a, b)`` contributes ``a*b`` weights and ``b`` biases. Useful
    for the equal-budget sweep in notebook 04, where you want to choose widths
    before committing to them.
    """
    sizes = [n_in] + list(hidden) + [n_out]
    return sum(sizes[i] * sizes[i + 1] + sizes[i + 1]
               for i in range(len(sizes) - 1))


def train(model: nn.Module,
          x: np.ndarray,
          y: np.ndarray,
          *,
          epochs: int = 2000,
          lr: float = 1e-2,
          weight_decay: float = 0.0,
          x_val: Optional[np.ndarray] = None,
          y_val: Optional[np.ndarray] = None,
          keep_best: bool = False,
          verbose_every: int = 0) -> Dict[str, object]:
    """Full-batch training with Adam and a mean-squared-error loss.

    ``x`` and ``y`` are 1-D NumPy arrays. Returns a history dictionary with
    ``"epoch"``, ``"train"`` and, when a validation set is given, ``"val"``.

    With ``keep_best=True`` the history also carries ``"best_epoch"``,
    ``"best_val"`` and ``"best_state"``, a copy of the weights at the epoch with
    the lowest validation loss. That is early stopping, implemented honestly:
    you train to the end and then go back, rather than stopping the moment the
    curve ticks upwards, which on a noisy validation curve stops far too early.
    """
    xt = torch.tensor(np.asarray(x, dtype=np.float32).reshape(-1, 1))
    yt = torch.tensor(np.asarray(y, dtype=np.float32).reshape(-1, 1))
    has_val = x_val is not None and y_val is not None
    if has_val:
        xv = torch.tensor(np.asarray(x_val, dtype=np.float32).reshape(-1, 1))
        yv = torch.tensor(np.asarray(y_val, dtype=np.float32).reshape(-1, 1))

    opt = torch.optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    hist_train, hist_val = [], []
    best_val, best_epoch, best_state = float("inf"), -1, None

    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(xt), yt)
        loss.backward()
        opt.step()
        hist_train.append(float(loss.item()))

        if has_val:
            model.eval()
            with torch.no_grad():
                v = float(loss_fn(model(xv), yv).item())
            hist_val.append(v)
            if keep_best and v < best_val:
                best_val, best_epoch = v, epoch
                best_state = copy.deepcopy(model.state_dict())

        if verbose_every and (epoch % verbose_every == 0 or epoch == epochs - 1):
            msg = f"epoch {epoch:6d}   train {hist_train[-1]:.6f}"
            if has_val:
                msg += f"   val {hist_val[-1]:.6f}"
            print(msg)

    history: Dict[str, object] = {
        "epoch": np.arange(epochs),
        "train": np.asarray(hist_train, dtype=float),
    }
    if has_val:
        history["val"] = np.asarray(hist_val, dtype=float)
    if keep_best and best_state is not None:
        history["best_epoch"] = best_epoch
        history["best_val"] = best_val
        history["best_state"] = best_state
    return history


def restore_best(model: nn.Module, history: Dict[str, object]) -> nn.Module:
    """Load the weights from the best validation epoch back into ``model``."""
    if "best_state" not in history:
        raise KeyError("this history has no best_state — train with keep_best=True")
    model.load_state_dict(history["best_state"])
    return model


def predict(model: nn.Module, x: np.ndarray) -> np.ndarray:
    """Evaluate a 1-D model on a 1-D array and return a 1-D array."""
    model.eval()
    with torch.no_grad():
        xt = torch.tensor(np.asarray(x, dtype=np.float32).reshape(-1, 1))
        return model(xt).numpy().ravel()


def mse(prediction: np.ndarray, target: np.ndarray) -> float:
    """Mean squared error, as a plain float."""
    prediction = np.asarray(prediction, dtype=float).ravel()
    target = np.asarray(target, dtype=float).ravel()
    return float(np.mean((prediction - target) ** 2))


def evaluate(model: nn.Module, x: np.ndarray, y: np.ndarray) -> float:
    """Mean squared error of ``model`` on ``(x, y)``."""
    return mse(predict(model, x), y)


# ──────────────────────────────────────────────────────────────────────────
# 4 · kinks — notebook 03
# ──────────────────────────────────────────────────────────────────────────

def numeric_kinks(x: np.ndarray, y: np.ndarray,
                  threshold: float = 0.02) -> np.ndarray:
    """Find kinks in a sampled piecewise-linear curve, numerically.

    A kink is a place where the slope changes. This estimates the slope by
    first differences and reports the midpoints where the slope changes by more
    than ``threshold`` times the total slope range.

    It is an estimate: two kinks closer together than the sample spacing look
    like one, and a very shallow kink looks like none. Notebook 03 asks you to
    compute the exact positions from the weights instead, and to compare.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    slope = np.diff(y) / np.diff(x)
    change = np.abs(np.diff(slope))
    scale = float(np.ptp(slope)) if np.ptp(slope) > 0 else 1.0
    idx = np.nonzero(change > threshold * scale)[0] + 1
    if idx.size == 0:
        return np.array([])

    # A single kink that falls between two samples shows up as a change spread
    # over two neighbouring intervals, so merge runs of adjacent indices and
    # report the centre of mass of each run.
    positions = []
    run = [idx[0]]
    for j in idx[1:]:
        if j == run[-1] + 1:
            run.append(j)
        else:
            positions.append(run)
            run = [j]
    positions.append(run)

    out = []
    for run in positions:
        w = change[np.asarray(run) - 1]
        out.append(float(np.sum(x[run] * w) / np.sum(w)))
    return np.asarray(out)


def plot_kinks(ax, kinks: Sequence[float], colour: str = "#7b61a8",
               label: Optional[str] = None):
    """Draw a vertical line at every kink position."""
    for i, k in enumerate(np.atleast_1d(np.asarray(kinks, dtype=float))):
        ax.axvline(float(k), color=colour, lw=1.0, ls=":", alpha=0.9,
                   label=label if i == 0 else None, zorder=1)
    return ax


# ──────────────────────────────────────────────────────────────────────────
# 5 · figures
# ──────────────────────────────────────────────────────────────────────────

def _new_axes(ax, figsize=(7.2, 4.4)):
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    return ax


def plot_fit(x: np.ndarray, y: np.ndarray,
             curves: Optional[Dict[str, np.ndarray]] = None,
             x_curve: Optional[np.ndarray] = None,
             truth: Optional[Callable[[np.ndarray], np.ndarray]] = None,
             x_val: Optional[np.ndarray] = None,
             y_val: Optional[np.ndarray] = None,
             ax=None, title: str = "", ylim=None):
    """Training points, optional held-out points, optional truth, and fits."""
    ax = _new_axes(ax)
    if x_curve is None:
        x_curve = np.linspace(float(np.min(x)), float(np.max(x)), 400)
    if truth is not None:
        ax.plot(x_curve, truth(x_curve), lw=1.5, ls="--", color="#888888",
                label="truth", zorder=2)
    if x_val is not None and y_val is not None:
        ax.plot(x_val, y_val, "s", ms=4, mfc="none", mec="#0f9d58", mew=1.2,
                label="held out", zorder=3)
    ax.plot(x, y, "o", ms=7, color="#111111", label="training", zorder=5)
    for i, (label, values) in enumerate(sorted((curves or {}).items())):
        ax.plot(x_curve, np.asarray(values).ravel(), lw=2.0,
                color=_CYCLE[i % len(_CYCLE)], label=label, zorder=4)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25)
    return ax


_CYCLE = ["#d94f2b", "#1f77b4", "#f4a300", "#0f9d58", "#7b61a8", "#00838f"]


def plot_curves(history: Dict[str, object], ax=None,
                title: str = "Training and validation loss",
                mark_best: bool = True):
    """Training and validation loss on one pair of axes, log scale.

    This is a standing requirement of the course: **every** training run in
    Ex_04 is reported with both curves on the same axes. Two separate figures,
    or a training curve alone, hide exactly the gap you are looking for.
    """
    ax = _new_axes(ax, figsize=(6.6, 4.2))
    ax.plot(history["epoch"], history["train"], lw=1.7, color="#1f77b4",
            label="training")
    if "val" in history:
        ax.plot(history["epoch"], history["val"], lw=1.7, color="#d94f2b",
                label="validation")
    if mark_best and "best_epoch" in history:
        ax.axvline(float(history["best_epoch"]), color="#0f9d58", lw=1.2,
                   ls="--",
                   label=f"best validation (epoch {history['best_epoch']})")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("mean squared error")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25, which="both")
    return ax


def error_table(rows, headers) -> str:
    """Format a small table as Markdown, for pasting into the report."""
    headers = list(headers)
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join([" --- "] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────
# 6 · the four questions
# ──────────────────────────────────────────────────────────────────────────

FOUR_QUESTIONS = (
    "What is the input, precisely?",
    "What is the loss — what single number was minimised?",
    "Where did the data come from, and who paid for it?",
    "What happens when it is wrong?",
)


def four_questions() -> Tuple[str, ...]:
    """The four questions from L3.2 slide 2, in order."""
    return FOUR_QUESTIONS
