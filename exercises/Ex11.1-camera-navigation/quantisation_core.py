r"""The quantisation notebook's data, network and measurements.

*Deep Learning for Engineering* — MSc, Aalborg University.

Remus Teodorescu (ret@et.aau.dk), with support from Research Assistant
Noman Khan (nomank@energy.aau.dk).

Used by ``Ex11.1_01_quantisation.ipynb``. The vibration classes, the second
machine and the damped response are the ones of Ex_06, and the code here is
Ex_6_core's own, so the numbers the notebook prints are the numbers Ex_06's
notebooks printed. What is new is what a deployed model costs:
``model_size_bytes`` and ``time_forward``.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


SEED = 0


def set_seed(seed: int = SEED) -> None:
    """Seed NumPy and PyTorch together, so a run can be repeated."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def count_parameters(model: nn.Module) -> int:
    """Number of trainable numbers in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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
# 4 · a small network, and tensors to feed it
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
# 5 · what a deployed model costs
# ──────────────────────────────────────────────────────────────────────────

def model_size_bytes(model: nn.Module) -> int:
    """Serialised size of the model's weights, in bytes.

    Measured the way it is measured in practice — by writing the state
    dictionary out and looking at how many bytes appeared — rather than by
    counting parameters and multiplying by four. Quantised tensors do not obey
    that multiplication, and the whole point of the quantisation notebook is the number you
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


def error_table(rows, headers) -> str:
    """Format a small table as Markdown, for pasting into the report."""
    headers = list(headers)
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join([" --- "] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)
