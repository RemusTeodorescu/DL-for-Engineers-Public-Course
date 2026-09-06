r"""Shared code for Ex_05 — CNN and GNN.

*Deep Learning for Engineering* — MSc, Aalborg University.
Remus Teodorescu (ret@et.aau.dk), with support from Research Assistant
Noman Khan (nomank@energy.aau.dk). The code here is original to this course;
see docs/PROVENANCE.md for what each reference text is cited for.

    Ex05_00_environment_check.ipynb        # 0 — check the tools, run this first
    Ex05_01_cnn_image_classification.ipynb # 1 — a convolution, by hand and then learned
    Ex05_02_graph_basics.ipynb             # 2 — adjacency, neighbours, message passing
    Ex05_03_gnn_six_bus_network.ipynb      # 3 — node regression on six buses
    Ex05_04_sequence_model.ipynb           # 4 — a recurrent net, and attention
    Ex05_05_report.ipynb                   # 5 — the report

This module is complete. You are not expected to change anything in it. Your
work is in the `# TODO:` cells of the notebooks.

Four datasets live here, all generated on your own machine — **nothing is
downloaded**, because a lecture theatre's network is not to be trusted and a
flaky MNIST mirror has ruined more exercise sessions than any bug:

* **weld radiographs**, 16 x 16 greyscale, three classes — clean, crack, pit.
  Procedural, in the spirit of the weld-inspection example from L3.2.
* **a six-bus power network** — the adjacency matrix, the line susceptances and
  a small synthetic load-flow dataset. This is deliberately the same object
  **Ex_12.1 in Part 2** works on, so that when L12.1 says "you met graph neural
  networks in Part 1" you remember not only the architecture but the same six
  buses.
* **a load profile** — a synthetic thirty-day substation demand trace for the
  sequence notebook.
* **a handful of fixed convolution kernels**, so that notebook 01 can check a
  hand-written convolution against numbers that are the same on every machine.

Everything uses `torch`, `numpy` and `matplotlib` only, on a CPU, in minutes.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn


SEED = 0

#: Where notebooks write anything a later notebook reads.
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "Ex05_outputs")


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
# 1 · the weld images — notebook 01
# ──────────────────────────────────────────────────────────────────────────

IMG_SIZE = 16
CLASS_NAMES = ("clean", "crack", "pit")


def _plate(rng: np.random.Generator) -> np.ndarray:
    """A blank plate: a slow brightness ramp plus film grain.

    Real radiographs are never uniform — the film, the source distance and the
    plate thickness all put a gradient across the image. Reproducing that is
    the point: a classifier that keys on overall brightness will fail here.
    """
    yy, xx = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE] / (IMG_SIZE - 1.0)
    a, b = rng.uniform(-0.25, 0.25, size=2)
    base = 0.55 + a * (xx - 0.5) + b * (yy - 0.5)
    grain = rng.normal(0.0, 0.05, size=(IMG_SIZE, IMG_SIZE))
    return base + grain


def _draw_crack(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A thin dark line at a random angle — one pixel wide, slightly wavy."""
    yy, xx = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE].astype(float)
    theta = rng.uniform(0.0, np.pi)
    cx, cy = rng.uniform(4.0, 11.0, size=2)
    # signed distance from the line through (cx, cy) with direction theta
    d = (xx - cx) * np.sin(theta) - (yy - cy) * np.cos(theta)
    # a small wave along the line, so cracks are not perfectly straight
    s = (xx - cx) * np.cos(theta) + (yy - cy) * np.sin(theta)
    d = d + 0.8 * np.sin(s / 3.0)
    depth = rng.uniform(0.30, 0.45)
    img = img - depth * np.exp(-(d ** 2) / (2 * 0.55 ** 2))
    return img


def _draw_pit(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A round dark blob — porosity, two or three pixels across."""
    yy, xx = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE].astype(float)
    cx, cy = rng.uniform(3.5, 12.5, size=2)
    r = rng.uniform(1.1, 1.9)
    depth = rng.uniform(0.30, 0.45)
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    return img - depth * np.exp(-d2 / (2 * r ** 2))


def weld_images(n_per_class: int = 200, seed: int = 3
                ) -> Tuple[np.ndarray, np.ndarray]:
    """Synthetic 16 x 16 radiographs of a weld seam.

    Returns ``(X, y)`` with ``X`` of shape ``(N, 1, 16, 16)`` and float values
    roughly in [0, 1], and ``y`` of shape ``(N,)`` holding 0, 1 or 2 for
    *clean*, *crack*, *pit*. ``N = 3 * n_per_class``, and the rows are
    shuffled.

    The channel dimension of size one is not decoration: `torch.nn.Conv2d`
    expects ``(batch, channels, height, width)`` and a greyscale image has one
    channel. Getting this shape wrong is the most common first error with a
    convolutional network.

    **What is real and what is not.** The three defect *types* — a thin linear
    indication, a round pore, and sound material — are the real vocabulary of
    radiographic weld inspection. The images are not: they are drawn with two
    Gaussian profiles and film grain, and no radiographer would mistake them
    for the genuine article. What survives the simplification is the property
    the notebook is about: **the defect can be anywhere on the plate, and it is
    the same defect wherever it is.**
    """
    rng = np.random.default_rng(seed)
    images, labels = [], []
    for label in range(3):
        for _ in range(n_per_class):
            img = _plate(rng)
            if label == 1:
                img = _draw_crack(img, rng)
            elif label == 2:
                img = _draw_pit(img, rng)
            images.append(np.clip(img, 0.0, 1.0))
            labels.append(label)
    X = np.asarray(images, dtype=np.float32)[:, None, :, :]
    y = np.asarray(labels, dtype=np.int64)
    order = rng.permutation(len(y))
    return X[order], y[order]


def split_data(X: np.ndarray, y: np.ndarray, frac: float = 0.7,
               seed: int = 5) -> Tuple[np.ndarray, ...]:
    """Split into a training and a held-out set, shuffling first."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    cut = int(frac * len(y))
    tr, te = idx[:cut], idx[cut:]
    return X[tr], y[tr], X[te], y[te]


#: Three fixed kernels, so that a hand-written convolution can be checked
#: against numbers that are identical on every machine.
KERNELS: Dict[str, np.ndarray] = {
    "vertical edge": np.array([[-1.0, 0.0, 1.0],
                               [-2.0, 0.0, 2.0],
                               [-1.0, 0.0, 1.0]]),
    "horizontal edge": np.array([[-1.0, -2.0, -1.0],
                                 [0.0, 0.0, 0.0],
                                 [1.0, 2.0, 1.0]]),
    "blob": np.array([[-1.0, -1.0, -1.0],
                      [-1.0, 8.0, -1.0],
                      [-1.0, -1.0, -1.0]]),
    "blur": np.ones((3, 3)) / 9.0,
}

#: A 5 x 5 test patch with a bright vertical bar down the middle column.
TEST_PATCH = np.array([[0.0, 0.0, 1.0, 0.0, 0.0],
                       [0.0, 0.0, 1.0, 0.0, 0.0],
                       [0.0, 0.0, 1.0, 0.0, 0.0],
                       [0.0, 0.0, 1.0, 0.0, 0.0],
                       [0.0, 0.0, 1.0, 0.0, 0.0]])


def plot_images(X: np.ndarray, y: np.ndarray, n: int = 12, ax=None,
                title: str = "", predictions: Optional[np.ndarray] = None):
    """A row-major grid of the first ``n`` images, labelled."""
    cols = 6
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(1.5 * cols, 1.7 * rows))
    axes = np.atleast_1d(axes).ravel()
    for k in range(len(axes)):
        axes[k].axis("off")
        if k >= n:
            continue
        axes[k].imshow(X[k, 0], cmap="gray", vmin=0.0, vmax=1.0)
        name = CLASS_NAMES[int(y[k])]
        if predictions is not None:
            pred = CLASS_NAMES[int(predictions[k])]
            mark = "" if pred == name else "  X"
            axes[k].set_title(f"{name}\n-> {pred}{mark}", fontsize=8)
        else:
            axes[k].set_title(name, fontsize=9)
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig


def show_kernel(kernel: np.ndarray, ax=None, title: str = ""):
    """Draw a small kernel as a heat map with its numbers written on it."""
    ax = _new_axes(ax, figsize=(3.0, 3.0))
    lim = float(np.max(np.abs(kernel))) or 1.0
    ax.imshow(kernel, cmap="RdBu_r", vmin=-lim, vmax=lim)
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            ax.text(j, i, f"{kernel[i, j]:.2f}", ha="center", va="center",
                    fontsize=9, color="#111111")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10)
    return ax


def plot_feature_maps(maps: np.ndarray, title: str = "",
                      image: Optional[np.ndarray] = None):
    """Show one input image and the feature maps a conv layer produced from it.

    ``maps`` has shape ``(n_channels, h, w)``.
    """
    n = maps.shape[0]
    cols = n + (1 if image is not None else 0)
    fig, axes = plt.subplots(1, cols, figsize=(1.7 * cols, 2.1))
    axes = np.atleast_1d(axes).ravel()
    k = 0
    if image is not None:
        axes[0].imshow(image, cmap="gray")
        axes[0].set_title("input", fontsize=9)
        axes[0].axis("off")
        k = 1
    for c in range(n):
        axes[k + c].imshow(maps[c], cmap="viridis")
        axes[k + c].set_title(f"channel {c}", fontsize=9)
        axes[k + c].axis("off")
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Three-by-three confusion matrix, rows = truth, columns = prediction."""
    m = np.zeros((3, 3), dtype=int)
    for t, p in zip(np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()):
        m[int(t), int(p)] += 1
    return m


def print_confusion(m: np.ndarray) -> None:
    """Print a confusion matrix with class names on both axes."""
    print("            predicted")
    print("           " + "".join(f"{n:>8s}" for n in CLASS_NAMES))
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name:8s} " + "".join(f"{m[i, j]:8d}" for j in range(3)))
    total = m.sum()
    print(f"\n  accuracy {np.trace(m)}/{total} = {np.trace(m) / total:.3f}")


# ──────────────────────────────────────────────────────────────────────────
# 2 · the six-bus network — notebooks 02 and 03
# ──────────────────────────────────────────────────────────────────────────

#: The six buses. Two generators, three load centres, one HVDC infeed —
#: the shape of network Ex_12.1 works on.
BUS_NAMES = ("B1 gen", "B2 gen", "B3 load", "B4 load", "B5 load", "B6 HVDC")

#: Which of the three roles each bus has, in the same order.
BUS_ROLE = ("gen", "gen", "load", "load", "load", "hvdc")

#: The eight lines, as (from, to, susceptance in per unit). Susceptance is
#: 1/X for the line; the numbers are plausible for a transmission network at
#: this voltage level, and they are made up rather than measured.
SIX_BUS_LINES: Tuple[Tuple[int, int, float], ...] = (
    (0, 1, 4.0),
    (0, 2, 3.0),
    (1, 2, 3.5),
    (1, 3, 2.5),
    (2, 4, 3.0),
    (3, 4, 3.5),
    (3, 5, 2.0),
    (4, 5, 2.5),
)

#: Fixed drawing coordinates, so every student's picture of the network looks
#: the same and buses can be discussed by position.
BUS_XY = np.array([[0.0, 1.0],
                   [1.0, 1.6],
                   [0.4, 0.0],
                   [2.0, 1.2],
                   [1.6, 0.2],
                   [2.8, 0.7]])

N_BUS = 6

#: Bus 0 carries the angle reference. Angles are only defined up to a common
#: offset, so one bus has to be nominated; in a real system it is a large
#: generator, and here it is bus 0.
REFERENCE_BUS = 0


def six_bus_edges(lines=SIX_BUS_LINES) -> List[Tuple[int, int]]:
    """The lines as (from, to) pairs, without the susceptances."""
    return [(a, b) for a, b, _ in lines]


def lines_without(index: int) -> Tuple[Tuple[int, int, float], ...]:
    """The line list with line ``index`` removed — a single-line contingency.

    Tripping one line is the standard "N-1" case every transmission operator
    plans against. The remaining network is still connected for every index on
    this system, so a load flow still has a solution. Notebook 03 uses it to ask
    a question no accuracy figure can answer: what happens to a model when the
    graph it was trained on changes?
    """
    return tuple(l for k, l in enumerate(SIX_BUS_LINES) if k != index)


def six_bus_adjacency(lines=SIX_BUS_LINES) -> np.ndarray:
    """The 6 x 6 adjacency matrix: 1 where a line exists, 0 elsewhere.

    Symmetric, zero on the diagonal. A transmission line carries power both
    ways, so the graph is undirected — which is not true of every engineering
    graph, and is worth noticing.
    """
    A = np.zeros((N_BUS, N_BUS))
    for a, b in six_bus_edges(lines):
        A[a, b] = 1.0
        A[b, a] = 1.0
    return A


def susceptance_matrix(lines=SIX_BUS_LINES) -> np.ndarray:
    r"""The DC power-flow matrix :math:`B`, built from the line susceptances.

    :math:`B_{ij} = -b_{ij}` off the diagonal and :math:`B_{ii} = \sum_j b_{ij}`
    on it — a weighted graph Laplacian. This is genuine power-system
    engineering: it is the matrix in the linearised ("DC") power flow, and it
    is singular by construction, because angles are only defined up to a common
    offset.
    """
    B = np.zeros((N_BUS, N_BUS))
    for a, b, susc in lines:
        B[a, b] -= susc
        B[b, a] -= susc
        B[a, a] += susc
        B[b, b] += susc
    return B


def plot_graph(A: Optional[np.ndarray] = None,
               node_values: Optional[np.ndarray] = None,
               ax=None, title: str = "", labels: Sequence[str] = BUS_NAMES,
               cmap: str = "coolwarm"):
    """Draw the network, optionally colouring the buses by a value each."""
    A = six_bus_adjacency() if A is None else np.asarray(A)
    ax = _new_axes(ax, figsize=(6.4, 4.2))
    n = A.shape[0]
    xy = BUS_XY if n == N_BUS else _ring(n)
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j]:
                ax.plot(xy[[i, j], 0], xy[[i, j], 1], "-", lw=1.8,
                        color="#999999", zorder=1)
    if node_values is None:
        ax.scatter(xy[:, 0], xy[:, 1], s=760, c="#e6f0f8",
                   edgecolors="#1f77b4", linewidths=2.0, zorder=3)
    else:
        v = np.asarray(node_values, dtype=float).ravel()
        sc = ax.scatter(xy[:, 0], xy[:, 1], s=760, c=v, cmap=cmap,
                        edgecolors="#333333", linewidths=1.4, zorder=3)
        plt.colorbar(sc, ax=ax, shrink=0.85)
    for i in range(n):
        ax.text(xy[i, 0], xy[i, 1] - 0.28, labels[i] if i < len(labels)
                else str(i), ha="center", va="top", fontsize=9)
        ax.text(xy[i, 0], xy[i, 1], str(i), ha="center", va="center",
                fontsize=10, zorder=4)
    ax.set_xlim(xy[:, 0].min() - 0.5, xy[:, 0].max() + 0.5)
    ax.set_ylim(xy[:, 1].min() - 0.7, xy[:, 1].max() + 0.4)
    ax.set_title(title)
    ax.axis("off")
    return ax


def _ring(n: int) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([np.cos(t), np.sin(t)], axis=1)


def show_matrix(M: np.ndarray, ax=None, title: str = "", fmt: str = "{:.2f}",
                cmap: str = "Blues", labels: Optional[Sequence[str]] = None):
    """Print a small matrix as a heat map with its entries written on it."""
    M = np.asarray(M, dtype=float)
    ax = _new_axes(ax, figsize=(5.0, 4.4))
    ax.imshow(M, cmap=cmap)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center",
                    fontsize=8,
                    color="#111111" if M[i, j] < 0.6 * M.max() else "#ffffff")
    ticks = np.arange(M.shape[0])
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    if labels is not None:
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(title, fontsize=10)
    return ax


# ── the synthetic load-flow dataset ───────────────────────────────────────

#: How strongly line loading pulls the voltage magnitude down. A stand-in for
#: the resistive losses a full AC solve would produce — see `six_bus_dataset`.
SAG = 0.12

#: How strongly reactive injections move the voltage magnitude.
Q_GAIN = 0.4


def _injections(rng: np.random.Generator, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Draw ``n`` operating points: active and reactive injections per bus.

    Generation is positive, load is negative, and the active injections are
    shifted so they sum to zero — a network cannot produce and consume
    different totals, which is Kirchhoff's current law written for the system
    as a whole.
    """
    P = np.zeros((n, N_BUS))
    Q = np.zeros((n, N_BUS))
    for i, role in enumerate(BUS_ROLE):
        if role == "gen":
            P[:, i] = rng.uniform(0.4, 1.4, size=n)
            Q[:, i] = rng.uniform(0.0, 0.5, size=n)
        elif role == "load":
            P[:, i] = -rng.uniform(0.3, 1.1, size=n)
            Q[:, i] = -rng.uniform(0.05, 0.45, size=n)
        else:                                   # HVDC infeed
            P[:, i] = rng.uniform(0.1, 0.8, size=n)
            Q[:, i] = rng.uniform(-0.1, 0.1, size=n)
    P = P - P.mean(axis=1, keepdims=True)       # balance the system
    return P, Q


def dc_power_flow(P: np.ndarray, lines=SIX_BUS_LINES) -> np.ndarray:
    r"""The linearised ("DC") power flow: solve :math:`P = B\theta`.

    Accepts one case of shape ``(6,)`` or a batch of shape ``(n, 6)`` and
    returns the angles with bus 0 as the reference. :math:`B` is singular — a
    common shift of every angle changes nothing physical — so the solve uses a
    pseudo-inverse.

    This is the classical approximation, in daily use for market clearing and
    contingency screening. Notebook 03 uses it as the **baseline your network
    has to beat**, which is the same role weighted least squares plays in
    Ex_12.1.
    """
    P = np.asarray(P, dtype=float)
    Bp = np.linalg.pinv(susceptance_matrix(lines))
    theta = P @ Bp.T
    return theta - theta[..., [0]]


def ac_angles(P: np.ndarray, lines=SIX_BUS_LINES, tol: float = 1e-12,
              max_iter: int = 60) -> np.ndarray:
    r"""Solve the lossless active power flow for the bus angles.

    .. math:: P_i = \sum_{j \in \mathcal{N}(i)} b_{ij}\,\sin(\theta_i-\theta_j)

    by Newton-Raphson, starting from the DC solution and with bus 0 fixed as
    the angle reference. This equation is **real power-system physics**: it is
    the active-power half of the AC power flow with line resistance neglected
    and every voltage magnitude held at one per unit. Dropping the sine is what
    turns it into :func:`dc_power_flow`.

    Accepts ``(6,)`` or ``(n, 6)`` and returns the same shape.
    """
    P = np.asarray(P, dtype=float)
    single = P.ndim == 1
    P = np.atleast_2d(P)
    Bs = np.zeros((N_BUS, N_BUS))
    for a, b, susc in lines:
        Bs[a, b] = susc
        Bs[b, a] = susc

    out = np.empty_like(P)
    for k in range(P.shape[0]):
        theta = dc_power_flow(P[k], lines)
        for _ in range(max_iter):
            d = theta[:, None] - theta[None, :]
            f = (Bs * np.sin(d)).sum(axis=1) - P[k]
            J = -Bs * np.cos(d)
            np.fill_diagonal(J, (Bs * np.cos(d)).sum(axis=1))
            step = np.zeros(N_BUS)
            step[1:] = np.linalg.solve(J[1:, 1:], -f[1:])
            theta = theta + step
            if np.max(np.abs(step)) < tol:
                break
        out[k] = theta - theta[0]
    return out[0] if single else out


def six_bus_dataset(n_cases: int = 800, seed: int = 12, noise: float = 5e-4,
                    lines=SIX_BUS_LINES) -> Dict[str, np.ndarray]:
    r"""A small synthetic load-flow dataset on the six-bus network.

    Returns a dictionary with

    * ``X``  — node features, shape ``(n_cases, 6, 5)``:
      ``[P_i, Q_i, is_generator, is_load, is_reference]``
    * ``Y``  — node targets, shape ``(n_cases, 6, 2)``: ``[theta_i, V_i - 1]``,
      the bus voltage angle in radians and the deviation of the voltage
      magnitude from nominal, in per unit
    * ``Y_clean`` — the same targets before measurement noise was added
    * ``A``  — the adjacency matrix, ``(6, 6)``
    * ``P``, ``Q`` — the raw injections, for plotting

    **What is real physics and what is a teaching simplification.** Read this
    before you quote a number from this notebook anywhere.

    *Real.* The topology. The susceptance matrix :math:`B`, and the fact that
    it is a weighted graph Laplacian. The requirement that active injections
    sum to zero. The sign conventions — generation positive, load negative,
    angle leading at the generators. And, most importantly, the **angles**:
    they come from :func:`ac_angles`, which solves the lossless active power
    flow :math:`P_i=\sum_j b_{ij}\sin(\theta_i-\theta_j)` by Newton-Raphson.
    That equation is the real thing with resistance neglected.

    *Simplified.* Everything about voltage **magnitude**. A full AC power flow
    solves for :math:`V` and :math:`\theta` together, using resistances as well
    as reactances, and enforces generator reactive limits. Here the magnitude
    is a stand-in built from two effects: a linear response to reactive
    injection, and a **sag proportional to the square of each line's angle
    difference**, imitating the way losses depress voltage on a heavily loaded
    corridor. Right qualitative behaviour, wrong numbers.

    *Also invented:* the line susceptances, the injection ranges, and the
    measurement noise level. Ex_12.1 in Part 2 makes the same kind of
    declaration about its own network, and for the same reason — the discipline
    is not avoiding assumptions, it is labelling them.

    *Why it is still worth training on.* The map from injections to state is
    **nonlinear**, **local** — a bus is affected most by its neighbours — and
    **defined on a graph**. Those three properties are what the architecture in
    notebook 03 exploits, and all three are real.
    """
    rng = np.random.default_rng(seed)
    P, Q = _injections(rng, n_cases)
    Bp = np.linalg.pinv(susceptance_matrix(lines))

    theta = ac_angles(P, lines)

    # voltage magnitude: reactive support, minus a loading-dependent sag
    sag = np.zeros_like(theta)
    for a, b, susc in lines:
        d2 = susc * (theta[:, a] - theta[:, b]) ** 2
        sag[:, a] += d2
        sag[:, b] += d2
    V = 1.0 + Q_GAIN * (Q @ Bp.T) - SAG * sag

    is_gen = np.array([1.0 if r == "gen" else 0.0 for r in BUS_ROLE])
    is_load = np.array([1.0 if r == "load" else 0.0 for r in BUS_ROLE])
    # The angle reference has to be a *feature*, not a convention. A model with
    # no notion of bus number cannot know which bus the angles are measured
    # from unless you tell it, and if you tell it by position the model stops
    # being permutation equivariant. See notebook 03, section 4.
    is_ref = np.zeros(N_BUS)
    is_ref[REFERENCE_BUS] = 1.0
    X = np.stack([P, Q,
                  np.tile(is_gen, (n_cases, 1)),
                  np.tile(is_load, (n_cases, 1)),
                  np.tile(is_ref, (n_cases, 1))], axis=2)
    Y_clean = np.stack([theta, V - 1.0], axis=2)
    Y = Y_clean + rng.normal(0.0, noise, size=Y_clean.shape)

    return {"X": X.astype(np.float32), "Y": Y.astype(np.float32),
            "Y_clean": Y_clean.astype(np.float32),
            "A": six_bus_adjacency(lines), "P": P, "Q": Q,
            "lines": lines,
            "noise": noise,
            "Y_scale": Y_clean.reshape(-1, 2).std(axis=0).astype(np.float32),
            "feature_names": np.array(["P", "Q", "is_gen", "is_load",
                                       "is_reference"]),
            "target_names": np.array(["theta [rad]", "V - 1 [p.u.]"])}


def normalised_adjacency(A: np.ndarray) -> np.ndarray:
    r"""Kipf and Welling's :math:`\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}`.

    Self-loops are added first (:math:`\tilde{A} = A + I`), so a node keeps its
    own features, and then rows and columns are scaled by the square root of
    the degree, so that a high-degree node does not simply shout louder than a
    low-degree one. Notebook 02 asks you to build this yourself and compare.
    """
    A = np.asarray(A, dtype=float)
    A_tilde = A + np.eye(A.shape[0])
    d = A_tilde.sum(axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(d)
    return A_tilde * d_inv_sqrt[:, None] * d_inv_sqrt[None, :]


def permutation_matrix(perm: Sequence[int]) -> np.ndarray:
    """The matrix ``P`` with ``P[i, perm[i]] = 1``.

    Left-multiplying node features by ``P`` relabels the nodes: row ``i`` of
    ``P X`` is row ``perm[i]`` of ``X``.
    """
    perm = np.asarray(perm, dtype=int)
    n = len(perm)
    P = np.zeros((n, n))
    P[np.arange(n), perm] = 1.0
    return P


def train_graph(model: nn.Module,
                A_hat: np.ndarray,
                X: np.ndarray,
                Y: np.ndarray,
                *,
                epochs: int = 400,
                lr: float = 5e-3,
                X_val: Optional[np.ndarray] = None,
                Y_val: Optional[np.ndarray] = None,
                verbose_every: int = 0) -> Dict[str, np.ndarray]:
    """Full-batch Adam training for a node-regression model.

    ``model(A_hat_tensor, X_tensor)`` must return a tensor shaped like ``Y``.
    Returns a history dictionary with ``"epoch"``, ``"train"`` and, when a
    validation set is given, ``"val"``.
    """
    At = torch.tensor(np.asarray(A_hat, dtype=np.float32))
    Xt = torch.tensor(np.asarray(X, dtype=np.float32))
    Yt = torch.tensor(np.asarray(Y, dtype=np.float32))
    has_val = X_val is not None and Y_val is not None
    if has_val:
        Xv = torch.tensor(np.asarray(X_val, dtype=np.float32))
        Yv = torch.tensor(np.asarray(Y_val, dtype=np.float32))

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    hist_train, hist_val = [], []

    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(At, Xt), Yt)
        loss.backward()
        opt.step()
        hist_train.append(float(loss.item()))
        if has_val:
            model.eval()
            with torch.no_grad():
                hist_val.append(float(loss_fn(model(At, Xv), Yv).item()))
        if verbose_every and (epoch % verbose_every == 0
                              or epoch == epochs - 1):
            msg = f"epoch {epoch:5d}   train {hist_train[-1]:.6f}"
            if has_val:
                msg += f"   val {hist_val[-1]:.6f}"
            print(msg)

    out = {"epoch": np.arange(epochs),
           "train": np.asarray(hist_train, dtype=float)}
    if has_val:
        out["val"] = np.asarray(hist_val, dtype=float)
    return out


# ──────────────────────────────────────────────────────────────────────────
# 3 · the load profile — notebook 04
# ──────────────────────────────────────────────────────────────────────────

STEPS_PER_DAY = 24


def load_profile(n_days: int = 40, seed: int = 21) -> np.ndarray:
    """A synthetic hourly substation demand trace, in per unit of peak.

    Three ingredients, all of them things a real feeder actually has:

    * a **daily cycle** with a morning shoulder and an evening peak;
    * a **weekly cycle** — weekends are about fifteen per cent lighter;
    * **correlated noise**, an AR(1) process rather than independent draws,
      because weather and human behaviour are correlated from hour to hour.

    Independent noise would make the task far too easy: a model could average
    it away. Correlated noise cannot be averaged away, and it is what makes the
    one-step forecast in notebook 04 a real forecast.
    """
    rng = np.random.default_rng(seed)
    n = n_days * STEPS_PER_DAY
    t = np.arange(n)
    hour = t % STEPS_PER_DAY
    day = t // STEPS_PER_DAY

    daily = (0.55
             + 0.18 * np.sin(2 * np.pi * (hour - 7) / STEPS_PER_DAY)
             + 0.12 * np.sin(4 * np.pi * (hour - 3) / STEPS_PER_DAY))
    weekly = np.where(day % 7 >= 5, 0.85, 1.0)

    noise = np.zeros(n)
    for k in range(1, n):
        noise[k] = 0.75 * noise[k - 1] + rng.normal(0.0, 0.020)

    return (daily * weekly + noise).astype(np.float64)


def make_windows(series: np.ndarray, window: int = 24, horizon: int = 1
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """Turn a 1-D series into ``(inputs, targets)`` for one-step forecasting.

    ``inputs`` has shape ``(N, window, 1)`` — batch, time, features — which is
    the shape every recurrent layer in PyTorch expects when
    ``batch_first=True``. ``targets`` has shape ``(N, 1)``.
    """
    series = np.asarray(series, dtype=np.float32).ravel()
    n = len(series) - window - horizon + 1
    X = np.stack([series[i:i + window] for i in range(n)])[:, :, None]
    y = series[window + horizon - 1: window + horizon - 1 + n][:, None]
    return X.astype(np.float32), y.astype(np.float32)


def split_series(X: np.ndarray, y: np.ndarray, frac: float = 0.75
                 ) -> Tuple[np.ndarray, ...]:
    """Split windows in **time order** — never shuffle a forecasting split.

    Shuffling before splitting lets the model see the future of its own test
    set, because two overlapping windows share twenty-three of their
    twenty-four values. It is the single most common way a published time
    series result turns out to be worthless.
    """
    cut = int(frac * len(y))
    return X[:cut], y[:cut], X[cut:], y[cut:]


def plot_series(series: np.ndarray, ax=None, title: str = "",
                highlight: Optional[Tuple[int, int]] = None):
    """Plot a demand trace with light day boundaries."""
    ax = _new_axes(ax, figsize=(9.0, 3.4))
    ax.plot(np.arange(len(series)), series, lw=1.2, color="#1f77b4")
    for d in range(0, len(series), STEPS_PER_DAY):
        ax.axvline(d, color="#dddddd", lw=0.7, zorder=0)
    if highlight is not None:
        ax.axvspan(highlight[0], highlight[1], color="#f4a300", alpha=0.20)
    ax.set_xlabel("hour")
    ax.set_ylabel("demand [p.u.]")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    return ax


def plot_forecast(y_true: np.ndarray, curves: Dict[str, np.ndarray],
                  ax=None, title: str = "", n: int = 120):
    """The first ``n`` steps of the held-out target and one or more forecasts."""
    ax = _new_axes(ax, figsize=(9.0, 3.6))
    y_true = np.asarray(y_true).ravel()[:n]
    ax.plot(y_true, lw=2.0, color="#111111", label="measured")
    for i, (label, values) in enumerate(sorted(curves.items())):
        ax.plot(np.asarray(values).ravel()[:n], lw=1.5,
                color=_CYCLE[i % len(_CYCLE)], label=label)
    ax.set_xlabel("hour of the held-out period")
    ax.set_ylabel("demand [p.u.]")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25)
    return ax


# ──────────────────────────────────────────────────────────────────────────
# 4 · figures and tables
# ──────────────────────────────────────────────────────────────────────────

_CYCLE = ["#d94f2b", "#1f77b4", "#f4a300", "#0f9d58", "#7b61a8", "#00838f"]


def _new_axes(ax, figsize=(7.2, 4.4)):
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    return ax


def plot_curves(history: Dict[str, np.ndarray], ax=None,
                title: str = "Training and validation loss"):
    """Training and validation loss on one pair of axes, log scale.

    A standing requirement of this course: **every** training run is reported
    with both curves on the same axes. A training curve on its own hides
    exactly the gap you are looking for.
    """
    ax = _new_axes(ax, figsize=(6.6, 4.2))
    ax.plot(history["epoch"], history["train"], lw=1.7, color="#1f77b4",
            label="training")
    if "val" in history:
        ax.plot(history["epoch"], history["val"], lw=1.7, color="#d94f2b",
                label="validation")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("mean squared error")
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


# ──────────────────────────────────────────────────────────────────────────
# 5 · the four questions
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
