r"""Shared code for Ex_06 — Training Lab.

*Deep Learning for Engineering* — MSc, Aalborg University.
Remus Teodorescu (ret@et.aau.dk), with support from Research Assistant
Noman Khan (nomank@energy.aau.dk). The code here is original to this course;
see docs/PROVENANCE.md for what each reference text is cited for.

    Ex06_00_environment_check.ipynb        # 0 — check the tools, run this first
    Ex06_01_loss_functions.ipynb           # 1 — likelihood, cross entropy, robustness
    Ex06_02_optimiser_comparison.ipynb     # 2 — SGD, momentum, Adam, and the
                                           #     Adam-to-L-BFGS handoff Part 2 uses
    Ex06_03_transfer_and_fine_tuning.ipynb # 3 — a second machine, ten labels per class
    Ex06_04_battery_arbitrage.ipynb        # 4 — a battery that learns to trade: RL,
                                           #     the exact optimum, and imitation of it
    Ex06_05_report.ipynb                   # 5 — the report

The quantisation notebook moved to Ex11.1 on 22 September 2026, with the
size and timing helpers it used (``quantisation_core.py`` there).

This module is complete. You are not expected to change anything in it. Your
work is in the ``# TODO:`` cells of the notebooks.

Six datasets, all generated on your own machine — **nothing is downloaded**,
because a lecture theatre's network is not to be trusted.

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
  same curve. Kept for optional use; no notebook uses it.
* **a second machine** — the same three vibration classes measured through a
  different sensor on a different mounting, so the features are rotated,
  scaled and offset. Notebook 03 transfers the network trained on machine A
  to machine B using ten labelled samples per class, which is the realistic
  amount an engineer gets.
* **day-ahead electricity prices** — synthetic 24-hour price curves with a
  morning and an evening peak and a solar dip at noon. Notebook 04 trades a
  200 kWh battery on them by reinforcement learning, against the exact optimum
  from a linear program.

Everything uses ``torch``, ``numpy`` and ``matplotlib``, and ``scipy`` for
notebook 04's one linear program, on a CPU, in minutes.
"""

from __future__ import annotations

import os
import re
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn


SEED = 0

#: Where notebooks write anything a later notebook reads.
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "Ex06_outputs")

# ── keeping results between notebooks ─────────────────────────────────────
#
# Later notebooks read what earlier ones saved. On Google Colab every notebook
# runs on its own temporary machine, so a file saved in one notebook is not
# there when the next one opens, and it is gone from its own machine once that
# runtime is recycled. ``keep_outputs()`` therefore moves OUTPUT_DIR into the
# student's Google Drive. When Drive is declined or unavailable, ``saved()``
# downloads each result file and ``needed()`` asks for it back. Locally none
# of this does anything.

DRIVE_ROOT = "DL4Eng"


def on_colab() -> bool:
    """True when running on Google Colab."""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def _in_drive(directory=None) -> bool:
    directory = OUTPUT_DIR if directory is None else directory
    return bool(directory) and str(directory).startswith("/content/drive/")


def keep_outputs(local_dir=None) -> str:
    """Make ``OUTPUT_DIR`` survive from one notebook to the next on Colab.

    ``local_dir`` is the results folder beside the notebooks, e.g.
    ``"Ex06_outputs"``. On Colab this mounts Google Drive and points
    OUTPUT_DIR at ``MyDrive/DL4Eng/<local_dir>``; approve the access request
    when Colab shows it. Locally, and when Drive is declined or unavailable,
    OUTPUT_DIR is ``local_dir`` itself and ``saved`` / ``needed`` fall back to
    downloading the files and asking for them back. Returns OUTPUT_DIR.
    """
    global OUTPUT_DIR
    if local_dir is not None:
        OUTPUT_DIR = local_dir
    if OUTPUT_DIR is None:
        raise ValueError("keep_outputs() needs the name of the results folder")
    if not on_colab() or _in_drive():
        return OUTPUT_DIR
    name = os.path.basename(os.path.normpath(OUTPUT_DIR))
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        OUTPUT_DIR = os.path.join("/content/drive/MyDrive", DRIVE_ROOT, name)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print("results are kept in your Google Drive, in",
              DRIVE_ROOT + "/" + name)
    except Exception as exc:  # declined, no Google account, or Drive is down
        text = str(exc).strip()
        reason = text.splitlines()[0] if text else type(exc).__name__
        print("Google Drive is not available (" + reason + ").")
        print("Results will be downloaded to your computer instead. Keep the")
        print("files: the notebook that needs them asks for them.")
    return OUTPUT_DIR


def output_path(name: str) -> str:
    """``OUTPUT_DIR/name``, creating the folder."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return os.path.join(OUTPUT_DIR, name)


def saved(*paths) -> None:
    """Call after writing result files. On Colab without Drive, download them.

    The machine the files were written on will not exist when the next
    notebook runs, so the student keeps a copy and ``needed`` asks for it.
    """
    for path in paths:
        print("saved:", path)
    if on_colab() and paths and not _in_drive(os.path.dirname(paths[0])):
        from google.colab import files
        print("This Colab machine is temporary, so the file(s) above are being "
              "downloaded. Keep them for the later notebooks.")
        for path in paths:
            files.download(path)


def needed(*names, directory=None) -> list:
    """Call before reading result files. Returns the names still missing.

    On Colab without Drive, missing files are requested as one upload: the
    student picks the copies that ``saved`` downloaded earlier (or cancels,
    if that notebook was never run). Elsewhere this only reports.
    """
    directory = OUTPUT_DIR if directory is None else directory
    missing = [n for n in names if not os.path.exists(os.path.join(directory, n))]
    if missing and on_colab() and not _in_drive(directory):
        from google.colab import files
        print("Not on this machine:", ", ".join(missing))
        print("Upload the copies downloaded by the notebooks that wrote them "
              "(cancel if you never ran those notebooks).")
        uploaded = files.upload()
        os.makedirs(directory, exist_ok=True)
        for fname, data in uploaded.items():
            # browsers rename a second download to "name (1).ext"
            clean = re.sub(r" \(\d+\)(?=\.[A-Za-z0-9]+$)", "", os.path.basename(fname))
            with open(os.path.join(directory, clean), "wb") as fh:
                fh.write(data)
            print("received", clean)
        missing = [n for n in names if not os.path.exists(os.path.join(directory, n))]
    return missing



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
    L4.1's Activation Function slide.
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
    """Format a small table as Markdown, for pasting into the report.

    Every column is padded to one width, so the table also reads straight
    when it is simply printed: a Markdown renderer ignores the padding, a
    printed cell output does not, and unpadded the numbers started wherever
    their row's name happened to end. A column whose cells are all numbers
    (or a placeholder "-") is right-aligned, and its separator says so, so
    the rendered table agrees with the printed one.
    """
    headers = [str(h) for h in headers]
    body = [[str(c) for c in row] for row in rows]
    n = len(headers)

    def cell(row, j):
        return row[j] if j < len(row) else ""

    def is_number(s):
        s = s.strip()
        if s in ("", "-", "–", "—"):
            return True
        try:
            float(s.rstrip("%"))
            return True
        except ValueError:
            return False

    width = [max([len(headers[j])] + [len(cell(r, j)) for r in body]) for j in range(n)]
    right = [bool(body) and all(is_number(cell(r, j)) for r in body) for j in range(n)]

    def line(row):
        return "| " + " | ".join(cell(row, j).rjust(width[j]) if right[j]
                                 else cell(row, j).ljust(width[j])
                                 for j in range(n)) + " |"

    rule = "|" + "|".join("-" * (width[j] + 1) + ":" if right[j] else "-" * (width[j] + 2)
                          for j in range(n)) + "|"
    return "\n".join([line(headers), rule] + [line(r) for r in body])


# ──────────────────────────────────────────────────────────────────────────
#  Notebook 04 — a battery that trades on the day-ahead price
# ──────────────────────────────────────────────────────────────────────────

#: The battery: 200 kWh, 50 kW (a four-hour battery), 95 % efficient each way,
#: 0.02 EUR of ageing for every kWh that passes through it, half full at the
#: start of the day and to be at least half full again at the end.
BATTERY = dict(E_MAX=200.0, P_MAX=50.0, ETA=0.95, C_DEG=0.02, SOC0=100.0)
HOURS = 24
#: The price of finishing the day below the starting charge, per kWh short.
#: Without it the cheapest policy is to empty the battery and walk away.
SHORTFALL_PRICE = 0.30


def arbitrage_prices(n_days: int, seed: int) -> np.ndarray:
    """``n_days`` day-ahead price curves, EUR/kWh, shape ``(n_days, 24)``.

    A morning and an evening peak and a midday dip where solar pushes the
    price down, scaled day by day between 0.7 and 1.3, shifted by up to two
    hours, with a little hour-to-hour noise. Synthetic, and shaped like a
    European day-ahead market; the numbers are not any market's.
    """
    rng = np.random.default_rng(seed)
    h = np.arange(HOURS)
    base = (0.10 + 0.10 * np.exp(-0.5 * ((h - 8) / 1.5) ** 2)
            + 0.22 * np.exp(-0.5 * ((h - 19) / 1.8) ** 2)
            - 0.05 * np.exp(-0.5 * ((h - 13) / 2.0) ** 2))
    scale = rng.uniform(0.7, 1.3, (n_days, 1))
    shift = rng.integers(-2, 3, n_days)
    p = np.stack([np.roll(base, s) for s in shift]) * scale
    p = p + rng.normal(0.0, 0.015, (n_days, HOURS))
    return np.clip(p, 0.01, None)


def lp_schedule(prices: np.ndarray) -> Tuple[float, np.ndarray]:
    """The best possible day for this battery, by linear programming.

    Decision variables: the energy bought from the grid, ``c_t``, and sold to
    it, ``d_t``, in each hour, both between 0 and ``P_MAX``. The charge after
    hour t is ``SOC0 + ETA * sum(c) - sum(d) / ETA``, kept between 0 and
    ``E_MAX`` and ending no lower than it started. The objective is the
    day's revenue minus ageing. Returns ``(profit_eur, grid_power)`` with
    ``grid_power = d - c`` in kW, positive when selling.

    This is the expert the notebook's network imitates: exact, because the
    battery's model is known and linear, and given the whole day's prices.
    """
    from scipy.optimize import linprog           # Colab and Anaconda both ship it
    b = BATTERY
    p = np.asarray(prices, dtype=float)
    cost = np.concatenate([p + b["C_DEG"], -p + b["C_DEG"]])
    L = np.tril(np.ones((HOURS, HOURS)))
    soc = np.hstack([b["ETA"] * L, -L / b["ETA"]])        # charge above SOC0 after hour t
    A = np.vstack([soc, -soc, -soc[-1:]])
    rhs = np.concatenate([np.full(HOURS, b["E_MAX"] - b["SOC0"]),
                          np.full(HOURS, b["SOC0"]), [0.0]])
    res = linprog(cost, A_ub=A, b_ub=rhs, bounds=[(0, b["P_MAX"])] * (2 * HOURS),
                  method="highs")
    return float(-res.fun), res.x[HOURS:] - res.x[:HOURS]


def battery_features(soc: torch.Tensor, t: int, prices: torch.Tensor) -> torch.Tensor:
    """What the policy sees in hour ``t``, for a batch of days: shape (B, 7).

    The charge as a fraction of capacity; the hour as a point on a circle;
    the price now; how far it sits from the day's mean; how much dearer the
    dearest hour still to come is; and how much cheaper the cheapest. Scaled
    to be of order one. The day-ahead market publishes tomorrow's prices
    today, so looking ahead within the day is not cheating.
    """
    B = prices.shape[0]
    now, ahead = prices[:, t], prices[:, t:]
    angle = 2 * np.pi * t / HOURS
    return torch.stack([
        soc / BATTERY["E_MAX"],
        torch.full((B,), np.sin(angle), dtype=prices.dtype),
        torch.full((B,), np.cos(angle), dtype=prices.dtype),
        now / 0.3,
        (now - prices.mean(1)) / 0.1,
        (ahead.max(1).values - now) / 0.1,
        (now - ahead.min(1).values) / 0.1], 1).float()


def plot_battery_day(prices, power, ax=None, title: str = ""):
    """One day: the price, and the battery's grid power as bars (+ selling)."""
    ax = _new_axes(ax)
    h = np.arange(HOURS)
    ax2 = ax.twinx()
    ax2.bar(h, power, color=np.where(np.asarray(power) >= 0, "#2e9e5b", "#d9822b"),
            alpha=0.55, width=0.8)
    ax2.set_ylabel("grid power, kW (+ sell, - buy)")
    ax2.set_ylim(-1.3 * BATTERY["P_MAX"], 1.3 * BATTERY["P_MAX"])
    ax.plot(h, prices, color="#1f77b4", lw=2.0)
    ax.set_xlabel("hour")
    ax.set_ylabel("price, EUR/kWh")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    return ax
