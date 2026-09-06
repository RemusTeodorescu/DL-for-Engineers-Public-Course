r"""Shared code for Ex_06 — Training Lab.

*Deep Learning for Engineering* — MSc, Aalborg University.
Remus Teodorescu (ret@et.aau.dk), with support from Research Assistant
Noman Khan (nomank@energy.aau.dk). The code here is original to this course;
see docs/PROVENANCE.md for what each reference text is cited for.

    Ex06_00_environment_check.ipynb        # 0 — check the tools, run this first
    Ex06_01_loss_functions.ipynb           # 1 — likelihood, cross entropy, robustness
    Ex06_02_optimiser_comparison.ipynb     # 2 — SGD, momentum, Adam, and the
                                           #     Adam-to-L-BFGS handoff Part 2 uses
    Ex06_03_overfit_then_regularise.ipynb  # 3 — twenty points, too much capacity
    Ex06_04_quantisation.ipynb             # 4 — smaller and faster, and what it costs
    Ex06_05_report.ipynb                   # 5 — the report

Notebooks 02 to 05 are **not written yet**; see this folder's README. The four
datasets they are specified against live here already, so they behave the same
way when those notebooks arrive as they do in 00 and 01 today.

This module is complete. You are not expected to change anything in it. Your
work is in the ``# TODO:`` cells of the notebooks.

Four datasets, all generated on your own machine — **nothing is downloaded**,
for the same reason as in Ex_05: a lecture theatre's network is not to be
trusted.

* **a load-cell calibration line** — forty readings against applied load, with
  Gaussian instrument noise of known standard deviation. Notebook 01 uses it to
  show that least squares *is* maximum likelihood under that noise model, and
  that the noise level you assume changes the likelihood but not the fit.
* **vibration classes** — three hundred and sixty machine-vibration signatures
  in two features, labelled balanced, imbalance or bearing fault. Deliberately
  overlapping: a class boundary a linear model gets exactly right teaches
  nothing about calibration.
* **a damped structural response** — two hundred **noiseless** samples. Noise
  is left out on purpose. When you compare optimisers, noisy data drives every
  optimiser to the same floor and the comparison measures nothing but the
  noise.
* **fatigue measurements** — twenty training points and sixty held out from the
  same curve, used by notebook 02 as a second regression target.
* **a second machine** — the same three vibration classes measured through a
  different sensor on a different mounting, so the features are rotated,
  scaled and offset. Notebook 03 transfers the network trained on machine A
  to machine B using ten labelled samples per class, which is the realistic
  amount an engineer gets.

Everything uses ``torch``, ``numpy`` and ``matplotlib`` only, on a CPU, in
minutes.
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn


SEED = 0

#: Where notebooks write anything a later notebook reads.
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "Ex06_outputs")


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


def count_parameters(model: nn.Module) -> int:
    """Number of trainable numbers in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ──────────────────────────────────────────────────────────────────────────
# 1 · the load-cell calibration line
# ──────────────────────────────────────────────────────────────────────────

#: The straight line the instrument really follows, and its noise level.
TRUE_SLOPE = 2.4
TRUE_INTERCEPT = 0.8
TRUE_SIGMA = 0.35

# Seed chosen so the default draw is a *typical* one: least squares recovers
# 2.395 / 0.787 and the residual standard deviation comes out at 0.348 against
# a true 0.35. Close but not exact, which is the honest lesson. The previous
# default happened to sit 1.7 sigma low and read like a bug.
CALIBRATION_SEED = 51


def calibration_dataset(n: int = 40, seed: int = CALIBRATION_SEED,
                        sigma: float = TRUE_SIGMA
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """``n`` readings from a load cell: ``y = a x + b`` plus Gaussian noise.

    The noise is Gaussian **by construction**, which is what makes notebook 01
    work: least squares is the maximum-likelihood estimator for exactly this
    noise model, and the notebook demonstrates that rather than asserting it.

    Returns ``(x, y)``, both shape ``(n,)``, float64.
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(0.05, 1.95, n)
    y = TRUE_SLOPE * x + TRUE_INTERCEPT + rng.normal(0.0, sigma, size=n)
    return x, y


def add_outlier(x: np.ndarray, y: np.ndarray, index: int = 7,
                offset: float = 4.0) -> Tuple[np.ndarray, np.ndarray]:
    """One reading recorded wrongly. Returns copies; the inputs are untouched.

    A transcription error, a loose connector, an operator reading the wrong
    dial — a single bad number in forty. Notebook 01 fits the corrupted set
    under both squared and absolute error, and the difference is the point.
    """
    x_bad = np.array(x, dtype=float, copy=True)
    y_bad = np.array(y, dtype=float, copy=True)
    y_bad[index] += offset
    return x_bad, y_bad


# ──────────────────────────────────────────────────────────────────────────
# 2 · vibration classes
# ──────────────────────────────────────────────────────────────────────────

VIBRATION_CLASSES = ("balanced", "imbalance", "bearing fault")

#: Class centres in (1x amplitude, high-frequency band amplitude), normalised.
_VIB_CENTRES = np.array([[0.25, 0.20],    # balanced: both small
                         [1.00, 0.30],    # imbalance: large at running speed
                         [0.35, 1.00]])   # bearing fault: energy up in the band
_VIB_SPREAD = 0.22


def vibration_dataset(n_per_class: int = 120, seed: int = 11
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """Machine-vibration signatures in two features, three classes.

    Feature 0 is the amplitude at running speed; feature 1 is the amplitude in
    the high-frequency band where bearing defects show up. Imbalance lifts the
    first, a spalled bearing lifts the second, and a healthy machine lifts
    neither.

    The classes **overlap on purpose**. A separable problem would let any loss
    reach perfect accuracy, and notebook 01 is about what the losses disagree
    on once accuracy has stopped distinguishing them.

    Returns ``(X, y)`` with ``X`` shape ``(3 * n_per_class, 2)`` float32 and
    ``y`` int64. The rows are **shuffled**, so any contiguous slice — such as
    the 240/120 split notebook 01 takes — contains all three classes.
    """
    rng = np.random.default_rng(seed)
    X = np.concatenate([
        centre + _VIB_SPREAD * rng.standard_normal((n_per_class, 2))
        for centre in _VIB_CENTRES])
    y = np.repeat(np.arange(len(VIBRATION_CLASSES)), n_per_class)

    order = rng.permutation(len(y))
    return X[order].astype(np.float32), y[order].astype(np.int64)


#: How machine B differs from machine A. A different accelerometer with a
#: different gain, bolted to a different part of the frame, so the two axes
#: are mixed and the whole cloud sits somewhere else. The *classes* are the
#: same physical faults — that is what makes transfer the right tool rather
#: than starting again.
MACHINE_B_GAIN = np.array([1.45, 0.70])
MACHINE_B_MIX = 0.35          # radians of rotation between the two axes
MACHINE_B_OFFSET = np.array([0.30, -0.15])


def _machine_b_transform(X: np.ndarray) -> np.ndarray:
    c, s = np.cos(MACHINE_B_MIX), np.sin(MACHINE_B_MIX)
    R = np.array([[c, -s], [s, c]])
    return (X * MACHINE_B_GAIN) @ R.T + MACHINE_B_OFFSET


def machine_b_dataset(n_per_class: int = 60, seed: int = 13
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """The same three faults, measured on a second machine.

    Same physics, different instrument: the features are scaled per axis,
    rotated, and offset. A classifier trained on machine A will be *wrong*
    here, but not uninformative — the structure it learned still applies, and
    that gap between "wrong" and "uninformative" is what notebook 03 measures.

    Returns ``(X, y)`` shaped like :func:`vibration_dataset`, shuffled.
    """
    rng = np.random.default_rng(seed)
    X = np.concatenate([
        centre + _VIB_SPREAD * rng.standard_normal((n_per_class, 2))
        for centre in _VIB_CENTRES])
    y = np.repeat(np.arange(len(VIBRATION_CLASSES)), n_per_class)

    X = _machine_b_transform(X)
    order = rng.permutation(len(y))
    return X[order].astype(np.float32), y[order].astype(np.int64)


def few_shot_split(X: np.ndarray, y: np.ndarray, per_class: int = 10,
                   seed: int = 5
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split into a small labelled set and everything else.

    ``per_class`` labelled samples of each class, chosen at random; the rest
    is held out. Ten per class is not a teaching convenience — it is roughly
    what you get when labelling means running a machine to failure.

    Returns ``(X_few, y_few, X_rest, y_rest)``.
    """
    rng = np.random.default_rng(seed)
    take = []
    for k in range(len(VIBRATION_CLASSES)):
        idx = np.flatnonzero(y == k)
        take.append(rng.choice(idx, size=per_class, replace=False))
    take = np.concatenate(take)
    rest = np.setdiff1d(np.arange(len(y)), take)
    return X[take], y[take], X[rest], y[rest]


def one_hot(y: np.ndarray, n_classes: int = len(VIBRATION_CLASSES)
            ) -> np.ndarray:
    """Integer labels to one-hot rows, shape ``(len(y), n_classes)``."""
    y = np.asarray(y).astype(int).ravel()
    out = np.zeros((len(y), n_classes), dtype=np.float32)
    out[np.arange(len(y)), y] = 1.0
    return out


def plot_classes(X: np.ndarray, y: np.ndarray, ax=None, title: str = "",
                 predictions: Optional[np.ndarray] = None):
    """Scatter the two vibration features, coloured by class.

    Pass ``predictions`` to ring the points the model gets wrong.
    """
    ax = _new_axes(ax, figsize=(6.0, 4.4))
    X = np.asarray(X)
    y = np.asarray(y).astype(int)
    for k, name in enumerate(VIBRATION_CLASSES):
        m = y == k
        ax.plot(X[m, 0], X[m, 1], "o", ms=4.5, alpha=0.85,
                color=_CYCLE[k], label=name)
    if predictions is not None:
        wrong = np.asarray(predictions).astype(int) != y
        ax.plot(X[wrong, 0], X[wrong, 1], "o", ms=10, mfc="none",
                mec="#111111", mew=1.4,
                label=f"wrong ({int(wrong.sum())})")
    ax.set_xlabel("amplitude at running speed")
    ax.set_ylabel("high-frequency band amplitude")
    ax.set_title(title or "vibration classes")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)
    return ax


# ──────────────────────────────────────────────────────────────────────────
# 3 · the damped structural response — noiseless, on purpose
# ──────────────────────────────────────────────────────────────────────────

RESPONSE_SPAN = (0.0, 4.0)


def damped_response(x) -> np.ndarray:
    """A damped oscillation: ``exp(-0.9 x) sin(4 x)``.

    Deterministic and noiseless. ``response_dataset`` returns exactly this
    function sampled on a grid, and notebook 00 checks that it does.
    """
    x = np.asarray(x, dtype=float)
    return np.exp(-0.9 * x) * np.sin(4.0 * x)


def response_dataset(n: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """``n`` noiseless samples of the damped response, evenly spaced.

    No noise. When you are comparing optimisers you want the differences
    between them to be the only thing moving; noise puts a floor under every
    optimiser at the same height and hides exactly what the notebook is
    measuring.

    Returns ``(x, y)``, both shape ``(n,)``, float64.
    """
    x = np.linspace(RESPONSE_SPAN[0], RESPONSE_SPAN[1], n)
    return x, damped_response(x)


# ──────────────────────────────────────────────────────────────────────────
# 4 · fatigue measurements — few points, plenty of capacity
# ──────────────────────────────────────────────────────────────────────────

FATIGUE_SIGMA = 0.045


def fatigue_truth(t) -> np.ndarray:
    """Normalised fatigue strength against normalised life, on ``[0, 1]``.

    Smooth, monotone-ish and slightly wavy — enough structure that twenty
    noisy samples do not pin it down, which is what notebook 03 needs.
    """
    t = np.asarray(t, dtype=float)
    return 0.70 * np.exp(-1.8 * t) + 0.22 * np.cos(5.0 * t) + 0.08


#: Seed chosen for even coverage: the largest gap between consecutive training
#: specimens is 0.107 of the range, so a model that fits badly between points
#: is overfitting rather than extrapolating into a hole in the data.
FATIGUE_SEED = 214


def fatigue_dataset(n: int = 20, seed: int = FATIGUE_SEED
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """``n`` noisy specimens — the training set. Twenty by default.

    Twenty is not an accident. It is few enough that a network with a few
    hundred parameters can drive its training loss to zero while getting the
    curve between the points badly wrong.
    """
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0.0, 1.0, size=n))
    return t, fatigue_truth(t) + rng.normal(0.0, FATIGUE_SIGMA, size=n)


def fatigue_validation(n: int = 60, seed: int = FATIGUE_SEED + 1
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """``n`` further specimens from the same curve, held out.

    Same population, same noise. Held out means held out: fit on
    ``fatigue_dataset`` only, and report on this.
    """
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0.0, 1.0, size=n))
    return t, fatigue_truth(t) + rng.normal(0.0, FATIGUE_SIGMA, size=n)


# ──────────────────────────────────────────────────────────────────────────
# 5 · a small network, and tensors to feed it
# ──────────────────────────────────────────────────────────────────────────

def to_tensor(a) -> torch.Tensor:
    """NumPy vector to a float32 column tensor of shape ``(N, 1)``.

    Column, not flat. ``nn.Linear`` and ``nn.MSELoss`` both want a trailing
    feature dimension, and a flat target against a column prediction
    broadcasts into an ``(N, N)`` loss that trains to something meaningless
    without ever raising.
    """
    a = np.asarray(a, dtype=np.float32).reshape(-1, 1)
    return torch.from_numpy(a)


class MLP(nn.Module):
    """A plain fully connected network. ``tanh`` by default, and deliberately.

    Part 2 differentiates the network's output with respect to its input, and
    ReLU's second derivative is zero everywhere it is defined — a second-order
    PDE residual would have nothing to work with. Using ``tanh`` here means the
    networks in this exercise set are the same objects you meet in L7. See
    L4.1 slide 12.
    """

    def __init__(self, in_dim: int = 1, out_dim: int = 1,
                 hidden: Sequence[int] = (16, 16),
                 activation=nn.Tanh):
        super().__init__()
        layers = []
        width = in_dim
        for h in hidden:
            layers += [nn.Linear(width, h), activation()]
            width = h
        layers.append(nn.Linear(width, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ──────────────────────────────────────────────────────────────────────────
# 6 · what a deployed model costs
# ──────────────────────────────────────────────────────────────────────────

def model_size_bytes(model: nn.Module) -> int:
    """Serialised size of the model's weights, in bytes.

    Measured the way it is measured in practice — by writing the state
    dictionary out and looking at how many bytes appeared — rather than by
    counting parameters and multiplying by four. Quantised tensors do not obey
    that multiplication, and the whole point of notebook 04 is the number you
    would actually copy onto the device.
    """
    import io
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getbuffer().nbytes


def time_forward(model: nn.Module, X: torch.Tensor, repeats: int = 50,
                 warmup: int = 5) -> float:
    """Median wall-clock seconds for one forward pass over ``X``.

    Median, not mean: on a shared laptop one scheduling hiccup ruins a mean
    and leaves a median untouched. ``warmup`` passes are run and discarded,
    because the first call through a freshly built graph is never
    representative.

    Timings on a laptop under load are noisy. Report the ratio between two
    models measured in the same session, never an absolute figure.
    """
    import time
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(X)
        samples = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            model(X)
            samples.append(time.perf_counter() - t0)
    return float(np.median(samples))


# ──────────────────────────────────────────────────────────────────────────
# 6 · reporting
# ──────────────────────────────────────────────────────────────────────────

_CYCLE = ["#d94f2b", "#1f77b4", "#f4a300", "#0f9d58", "#7b61a8", "#00838f"]


def _new_axes(ax, figsize=(7.2, 4.4)):
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    return ax


def plot_curves(history: Dict[str, np.ndarray], ax=None,
                title: str = "Training and validation loss"):
    """Loss curves on one pair of axes, log scale.

    A standing requirement of this course: **every** training run is reported
    with its curve, not with a final number alone.
    """
    ax = _new_axes(ax)
    for i, (name, curve) in enumerate(history.items()):
        ax.plot(np.asarray(curve), lw=1.8, color=_CYCLE[i % len(_CYCLE)],
                label=name)
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25, which="both")
    return ax


def mse(prediction: np.ndarray, target: np.ndarray) -> float:
    """Mean squared error, as a plain float."""
    prediction = np.asarray(prediction, dtype=float).ravel()
    target = np.asarray(target, dtype=float).ravel()
    return float(np.mean((prediction - target) ** 2))


def error_table(rows, headers) -> str:
    """Format a small table as Markdown, for pasting into the report."""
    headers = list(headers)
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join([" --- "] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)
