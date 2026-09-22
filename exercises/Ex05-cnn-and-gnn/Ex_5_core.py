r"""Shared code for Ex_05 — CNN and GNN.

*Deep Learning for Engineering* — MSc, Aalborg University.
Remus Teodorescu (ret@et.aau.dk), with support from Research Assistant
Noman Khan (nomank@energy.aau.dk). The code here is original to this course;
see docs/PROVENANCE.md for what each reference text is cited for.

    Ex05_00_environment_check.ipynb        # 0 — check the tools, run this first
    Ex05_01_cnn_image_classification.ipynb # 1 — weld defects: a CNN against a dense network
    Ex05_02_graph_basics.ipynb             # 2 — adjacency, neighbours, message passing
    Ex05_03_gnn_six_bus_network.ipynb      # 3 — node regression on six buses
    Ex05_04_sequence_model.ipynb           # 4 — a recurrent net and an LSTM
    Ex05_05_report.ipynb                   # 5 — the report

This module is complete. You are not expected to change anything in it. Your
work is in the `# TODO:` cells of the notebooks.

Four datasets live here, all generated on your own machine — **nothing is
downloaded**, because a lecture theatre's network is not to be trusted and a
flaky MNIST mirror has ruined more exercise sessions than any bug:

* **weld radiographs**, 16 x 16 greyscale, three classes — clean, crack, pit.
  Procedural, in the spirit of the weld-inspection example from L3.2.
* **a six-bus power network** — the adjacency matrix, the line impedances and
  a small load-flow dataset solved with a full AC power flow.
* **a load profile** — a synthetic forty-day substation demand trace for the
  sequence notebook.
* **a handful of fixed convolution kernels**, so that notebook 01 can set a
  convolution layer's weights by hand before it lets training choose them.

Everything uses `torch`, `numpy` and `matplotlib` only, on a CPU, in minutes.
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn


SEED = 0

#: Where notebooks write anything a later notebook reads.
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "Ex05_outputs")

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
    ``"Ex07.1_outputs"``. On Colab this mounts Google Drive and points
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


#: Four fixed kernels — hand-set weights for a convolution layer, the same
#: numbers on every machine.
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
                    fontsize=9,
                    color="white" if abs(kernel[i, j]) > 0.6 * lim else "#111111")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10)
    return ax


def plot_feature_maps(maps: np.ndarray, title: str = "",
                      image: Optional[np.ndarray] = None):
    """Show one input image and the feature maps a conv layer produced from it.

    ``maps`` has shape ``(n_channels, h, w)``. Each map is drawn with a
    diverging colour scale of its own, symmetric about zero — red above zero,
    blue below, white at zero — and its largest absolute value is written
    under it, so a map that looks strong but is small can be recognised.
    """
    n = maps.shape[0]
    cols = n + (1 if image is not None else 0)
    fig, axes = plt.subplots(1, cols, figsize=(1.7 * cols, 2.3))
    axes = np.atleast_1d(axes).ravel()
    k = 0
    if image is not None:
        axes[0].imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        axes[0].set_title("input", fontsize=9)
        axes[0].axis("off")
        k = 1
    for c in range(n):
        lim = float(np.max(np.abs(maps[c]))) or 1.0
        axes[k + c].imshow(maps[c], cmap="RdBu_r", vmin=-lim, vmax=lim)
        axes[k + c].set_title(f"channel {c}", fontsize=9)
        axes[k + c].set_xticks([]); axes[k + c].set_yticks([])
        axes[k + c].set_xlabel(f"±{lim:.2f}", fontsize=8)
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig


def plot_responses(images: np.ndarray, maps: np.ndarray,
                   row_names: Sequence[str], col_names: Sequence[str],
                   title: str = ""):
    """Images down the left, and each image's feature maps to its right.

    ``images`` has shape ``(n_images, h, w)`` and ``maps`` has shape
    ``(n_images, n_kernels, h', w')``. The image is drawn in grey, because it
    is brightness. Each kernel's column has its own diverging colour scale,
    symmetric about zero and shared down the column, with its colour bar
    underneath: red is a positive response, blue a negative one, white none.
    """
    n_img, n_k = maps.shape[:2]
    fig, axes = plt.subplots(n_img, n_k + 1,
                             figsize=(2.1 * (n_k + 1), 2.0 * n_img + 0.9),
                             squeeze=False)
    for r in range(n_img):
        axes[r, 0].imshow(images[r], cmap="gray", vmin=0.0, vmax=1.0)
        axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        axes[r, 0].set_ylabel(row_names[r], fontsize=10)
        if r == 0:
            axes[r, 0].set_title("image", fontsize=10)
    for c in range(n_k):
        lim = float(np.max(np.abs(maps[:, c]))) or 1.0
        for r in range(n_img):
            im = axes[r, c + 1].imshow(maps[r, c], cmap="RdBu_r",
                                       vmin=-lim, vmax=lim)
            axes[r, c + 1].axis("off")
            if r == 0:
                axes[r, c + 1].set_title(col_names[c], fontsize=10)
        fig.colorbar(im, ax=axes[:, c + 1], orientation="horizontal",
                     fraction=0.05, pad=0.04, label="response")
    fig.colorbar(plt.cm.ScalarMappable(cmap="gray",
                                       norm=plt.Normalize(0.0, 1.0)),
                 ax=axes[:, 0], orientation="horizontal", fraction=0.05,
                 pad=0.04, label="brightness")
    if title:
        fig.suptitle(title, fontsize=12)
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


def plot_confusions(matrices: Dict[str, np.ndarray], title: str = ""):
    """Confusion matrices side by side, each cell labelled with its count."""
    n = len(matrices)
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 3.8), squeeze=False)
    for ax, (name, m) in zip(axes[0], matrices.items()):
        ax.imshow(m, cmap="Blues", vmin=0, vmax=int(m.sum(axis=1).max()))
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                ax.text(j, i, str(int(m[i, j])), ha="center", va="center",
                        fontsize=12,
                        color="white" if m[i, j] > 0.6 * m.sum(axis=1).max()
                        else "#111111")
        ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES)
        ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
        ax.set_xlabel("predicted class")
        ax.set_ylabel("true class")
        acc = np.trace(m) / m.sum()
        ax.set_title(f"{name}: {np.trace(m)}/{m.sum()} correct ({acc:.3f})",
                     fontsize=10)
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return fig


def train_classifier(model: nn.Module,
                     X_train: np.ndarray, y_train: np.ndarray,
                     X_test: np.ndarray, y_test: np.ndarray,
                     *, epochs: int = 1000, lr: float = 1e-3,
                     record_every: int = 10) -> Dict[str, np.ndarray]:
    """Full-batch Adam on cross entropy — the one recipe notebook 01 uses.

    Both networks in the notebook are trained with this function, so they
    get the same data, the same optimiser, the same learning rate and the
    same number of epochs. Returns a dictionary with

    * ``"loss"`` — training cross entropy at every epoch;
    * ``"epoch"``, ``"acc_train"``, ``"acc_test"`` — both accuracies, every
      ``record_every`` epochs and at the last;
    * ``"seconds"`` — wall-clock time spent on the training steps alone (the
      accuracy bookkeeping is not counted);
    * ``"pred_train"``, ``"pred_test"`` — the final predicted classes.
    """
    import time
    Xt, yt = torch.tensor(X_train), torch.tensor(y_train)
    Xv, yv = torch.tensor(X_test), torch.tensor(y_test)
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    losses, epochs_rec, acc_tr, acc_te = [], [], [], []
    seconds = 0.0
    for epoch in range(epochs):
        t0 = time.perf_counter()
        optimiser.zero_grad()
        loss = loss_fn(model(Xt), yt)
        loss.backward()
        optimiser.step()
        seconds += time.perf_counter() - t0
        losses.append(float(loss.item()))
        if epoch % record_every == 0 or epoch == epochs - 1:
            with torch.no_grad():
                epochs_rec.append(epoch + 1)
                acc_tr.append(float((model(Xt).argmax(1) == yt).float().mean()))
                acc_te.append(float((model(Xv).argmax(1) == yv).float().mean()))
    with torch.no_grad():
        pred_train = model(Xt).argmax(1).numpy()
        pred_test = model(Xv).argmax(1).numpy()
    return {"loss": np.asarray(losses), "epoch": np.asarray(epochs_rec),
            "acc_train": np.asarray(acc_tr), "acc_test": np.asarray(acc_te),
            "seconds": seconds,
            "pred_train": pred_train, "pred_test": pred_test}


# ──────────────────────────────────────────────────────────────────────────
# 2 · the six-bus network — notebooks 02 and 03
# ──────────────────────────────────────────────────────────────────────────

#: The six buses. Two generators, three load centres, one HVDC infeed.
#: The bus number is the index, 0 to 5; the name says what is there.
BUS_NAMES = ("gen (slack)", "gen (PV)", "load", "load", "load", "HVDC")

#: Which of the three roles each bus has, in the same order.
BUS_ROLE = ("gen", "gen", "load", "load", "load", "hvdc")

#: The power-flow bus type, in the same order.
#:   slack — |V| and angle given (angle 0, the reference); P and Q are solved
#:   PV    — P and |V| given; Q and the angle are solved
#:   PQ    — P and Q given; |V| and the angle are solved
#: The HVDC converter is a PQ bus: it is told how much P and Q to deliver.
BUS_TYPE = ("slack", "PV", "PQ", "PQ", "PQ", "PQ")

#: Voltage set-points of the two generator buses, in per unit. Fixed for every
#: operating point: the slack bus holds 1.03 p.u. and the PV bus 1.02 p.u.
V_SET = np.array([1.03, 1.02, 1.0, 1.0, 1.0, 1.0])

#: The power base. Every P, Q, R and X here is in per unit on 100 MVA, so
#: 1 p.u. of active power is 100 MW and 1 p.u. of reactive power 100 Mvar.
S_BASE_MVA = 100.0

#: The eight lines, as (from, to, R, X): series resistance and reactance in per
#: unit. The line's impedance is z = R + jX and its admittance y = 1/z. The
#: numbers are plausible for a transmission network (X/R about 8) and are made
#: up rather than measured. Line charging (shunt capacitance) is neglected.
SIX_BUS_LINES: Tuple[Tuple[int, int, float, float], ...] = (
    (0, 1, 0.008, 0.06),
    (0, 2, 0.009, 0.07),
    (1, 2, 0.008, 0.06),
    (1, 3, 0.010, 0.08),
    (2, 4, 0.009, 0.07),
    (3, 4, 0.008, 0.06),
    (3, 5, 0.013, 0.10),
    (4, 5, 0.010, 0.08),
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


def _branches(lines) -> List[Tuple[int, int, float, float]]:
    """Lines as (from, to, R, X). Older (from, to, 1/X) triples are accepted."""
    out = []
    for l in lines:
        if len(l) == 3:
            out.append((int(l[0]), int(l[1]), 0.0, 1.0 / float(l[2])))
        else:
            out.append((int(l[0]), int(l[1]), float(l[2]), float(l[3])))
    return out


def six_bus_edges(lines=SIX_BUS_LINES) -> List[Tuple[int, int]]:
    """The lines as (from, to) pairs, without their impedances."""
    return [(l[0], l[1]) for l in lines]


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


def admittance_matrix(lines=SIX_BUS_LINES) -> np.ndarray:
    r"""The bus admittance matrix :math:`Y_{bus}`, complex, 6 x 6.

    Each line has impedance :math:`z = R + jX` and admittance :math:`y = 1/z`.
    Off the diagonal :math:`Y_{ij} = -y_{ij}`; on it :math:`Y_{ii} = \sum_j
    y_{ij}`. It is the matrix of the AC power flow, :math:`I = Y_{bus} V`.
    """
    Y = np.zeros((N_BUS, N_BUS), dtype=complex)
    for a, b, r, x in _branches(lines):
        y = 1.0 / complex(r, x)
        Y[a, b] -= y
        Y[b, a] -= y
        Y[a, a] += y
        Y[b, b] += y
    return Y


def edge_features(lines=SIX_BUS_LINES) -> np.ndarray:
    """E, one row per line: ``[R, X]`` in per unit, shape ``(n_lines, 2)``."""
    return np.array([[r, x] for _, _, r, x in _branches(lines)])


def line_table(lines=SIX_BUS_LINES) -> str:
    """The line data as a Markdown table: impedance and admittance per line."""
    rows = []
    for k, (a, b, r, x) in enumerate(_branches(lines)):
        y = 1.0 / complex(r, x)
        rows.append([k, f"{a} - {b}", f"{r:.3f}", f"{x:.3f}",
                     f"{y.real:.2f} {'-' if y.imag < 0 else '+'} "
                     f"j{abs(y.imag):.2f}"])
    return error_table(rows, ["line", "buses", "R [p.u.]", "X [p.u.]",
                              "admittance y = 1/(R + jX) [p.u.]"])


def plot_graph(A: Optional[np.ndarray] = None,
               node_values: Optional[np.ndarray] = None,
               ax=None, title: str = "", labels: Sequence[str] = BUS_NAMES,
               cmap: str = "coolwarm", cbar_label: str = "",
               annotate: str = ""):
    """Draw the network, optionally colouring the buses by a value each.

    ``cbar_label`` names the colour bar, with its unit. ``annotate`` is a
    format such as ``"{:+.2f}"``: when given, each bus's value is written
    beside it.
    """
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
        lim = np.abs(v).max() if cmap == "coolwarm" else None
        sc = ax.scatter(xy[:, 0], xy[:, 1], s=760, c=v, cmap=cmap,
                        vmin=-lim if lim else None, vmax=lim,
                        edgecolors="#333333", linewidths=1.4, zorder=3)
        cb = plt.colorbar(sc, ax=ax, shrink=0.85)
        if cbar_label:
            cb.set_label(cbar_label)
        if annotate:
            for i in range(n):
                ax.text(xy[i, 0] + 0.2, xy[i, 1] + 0.12,
                        annotate.format(v[i]), ha="left", va="bottom",
                        fontsize=9, fontweight="bold")
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


# ── the load-flow dataset ─────────────────────────────────────────────────

def _injections(rng: np.random.Generator, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Draw ``n`` operating points: the P and Q each bus is **told** to inject.

    Generation is positive, load is negative, in per unit on 100 MVA.

    * bus 1, the PV generator: P from 0.3 to 1.0 (30 to 100 MW)
    * buses 2, 3 and 4, the loads: P from 0.5 to 1.2 drawn, Q from 0.1 to 0.4
    * bus 5, the HVDC converter: P from 0.1 to 0.5 delivered, Q from -0.1 to 0.1

    Each number is drawn independently and uniformly. Bus 0, the slack bus, is
    left at zero: its P and Q are not chosen but **solved**, since it supplies
    whatever the others do not, losses included. The Q of the PV bus is solved
    too, so it is zero here as well.
    """
    P = np.zeros((n, N_BUS))
    Q = np.zeros((n, N_BUS))
    for i, role in enumerate(BUS_ROLE):
        if BUS_TYPE[i] == "slack":
            continue
        if role == "gen":                       # PV bus: P given, Q solved
            P[:, i] = rng.uniform(0.3, 1.0, size=n)
        elif role == "load":
            P[:, i] = -rng.uniform(0.5, 1.2, size=n)
            Q[:, i] = -rng.uniform(0.1, 0.4, size=n)
        else:                                   # HVDC converter, a PQ bus
            P[:, i] = rng.uniform(0.1, 0.5, size=n)
            Q[:, i] = rng.uniform(-0.1, 0.1, size=n)
    return P, Q


def ac_power_flow(P: np.ndarray, Q: np.ndarray, lines=SIX_BUS_LINES,
                  v_set: np.ndarray = V_SET, tol: float = 1e-10,
                  max_iter: int = 30):
    r"""The full AC power flow, by Newton-Raphson, for one operating point.

    Solves, at every bus :math:`i`,

    .. math:: P_i + jQ_i = V_i \sum_k \overline{Y_{ik} V_k}

    with the unknowns set by the bus type (:data:`BUS_TYPE`): the slack bus has
    :math:`|V|` = ``v_set`` and angle 0; the PV bus has P and :math:`|V|` =
    ``v_set``; the PQ buses have P and Q. Only the given entries of ``P`` and
    ``Q`` are read.

    Returns ``(Vm, theta, P, Q)``: the magnitude in per unit, the angle in
    radians, and the P and Q at **every** bus once solved — including the
    slack bus's P and Q and the PV bus's Q.
    """
    Y = admittance_matrix(lines)
    pv = [i for i, t in enumerate(BUS_TYPE) if t == "PV"]
    pq = [i for i, t in enumerate(BUS_TYPE) if t == "PQ"]
    pvpq = pv + pq
    Vm = np.where(np.isin(BUS_TYPE, ["slack", "PV"]), v_set, 1.0).astype(float)
    th = np.zeros(N_BUS)
    for _ in range(max_iter):
        V = Vm * np.exp(1j * th)
        S = V * np.conj(Y @ V)
        mis = np.r_[P[pvpq] - S.real[pvpq], Q[pq] - S.imag[pq]]
        if np.max(np.abs(mis)) < tol:
            break
        # the Jacobian, in the compact complex form: dS/dtheta and dS/d|V|
        I = Y @ V
        dS_dth = 1j * np.diag(V) @ np.conj(np.diag(I) - Y @ np.diag(V))
        dS_dVm = (np.diag(V) @ np.conj(Y @ np.diag(V / Vm))
                  + np.conj(np.diag(I)) @ np.diag(V / Vm))
        J = np.block([[dS_dth.real[np.ix_(pvpq, pvpq)], dS_dVm.real[np.ix_(pvpq, pq)]],
                      [dS_dth.imag[np.ix_(pq, pvpq)], dS_dVm.imag[np.ix_(pq, pq)]]])
        dx = np.linalg.solve(J, mis)
        th[pvpq] += dx[:len(pvpq)]
        Vm[pq] += dx[len(pvpq):]
    else:
        raise RuntimeError("AC power flow did not converge")
    V = Vm * np.exp(1j * th)
    S = V * np.conj(Y @ V)
    return Vm, th, S.real, S.imag


def six_bus_dataset(n_cases: int = 800, seed: int = 12, noise: float = 0.0,
                    lines=SIX_BUS_LINES) -> Dict[str, np.ndarray]:
    r"""A load-flow dataset on the six-bus network: ``n_cases`` operating points.

    Each case draws the given injections at random (:func:`_injections`) and
    solves the full AC power flow (:func:`ac_power_flow`) for them. What varies
    from case to case is only what is drawn: the PV generator's P, the three
    loads' P and Q, and the HVDC converter's P and Q. The network and the two
    voltage set-points are the same in every case.

    Returns a dictionary with

    * ``X`` — node features, ``(n_cases, 6, 6)``: what is **given** at each
      bus, ``[P, Q, V_set, is_slack, is_PV, is_PQ]``. A quantity that is solved
      rather than given (the slack bus's P and Q, the PV bus's Q, a PQ bus's
      |V|) is entered as 0; the bus-type flags say which entries are real.
    * ``Y`` — node targets, ``(n_cases, 6, 2)``: ``[theta, |V|]``, the solved
      angle in radians and magnitude in per unit
    * ``A`` — the adjacency matrix, ``(6, 6)``
    * ``E`` — edge features, ``(n_lines, 2)``: ``[R, X]`` of each line
    * ``P``, ``Q`` — the solved injections at every bus, ``(n_cases, 6)``
    * ``Y_mean``, ``Y_scale`` — per-channel mean and standard deviation of Y

    **Real and invented.** The equations are the real AC power flow. The line
    impedances, the injection ranges and the set-points are invented, line
    charging is neglected, and the generators' reactive limits are not
    enforced. ``noise`` adds Gaussian noise to Y (default none).
    """
    rng = np.random.default_rng(seed)
    P_giv, Q_giv = _injections(rng, n_cases)
    Vm = np.empty((n_cases, N_BUS))
    th = np.empty((n_cases, N_BUS))
    P = np.empty((n_cases, N_BUS))
    Q = np.empty((n_cases, N_BUS))
    for k in range(n_cases):
        Vm[k], th[k], P[k], Q[k] = ac_power_flow(P_giv[k], Q_giv[k], lines)

    is_slack = np.array([t == "slack" for t in BUS_TYPE], dtype=float)
    is_pv = np.array([t == "PV" for t in BUS_TYPE], dtype=float)
    is_pq = np.array([t == "PQ" for t in BUS_TYPE], dtype=float)
    v_given = np.where(is_pq > 0, 0.0, V_SET)
    tile = lambda v: np.tile(v, (n_cases, 1))
    X = np.stack([P_giv, Q_giv, tile(v_given),
                  tile(is_slack), tile(is_pv), tile(is_pq)], axis=2)
    Y_clean = np.stack([th, Vm], axis=2)
    Y = Y_clean + rng.normal(0.0, noise, size=Y_clean.shape) if noise else Y_clean

    return {"X": X.astype(np.float32), "Y": Y.astype(np.float32),
            "Y_clean": Y_clean.astype(np.float32),
            "A": six_bus_adjacency(lines), "E": edge_features(lines),
            "P": P, "Q": Q,
            "lines": lines,
            "noise": noise,
            "Y_mean": Y_clean.reshape(-1, 2).mean(axis=0).astype(np.float32),
            "Y_scale": Y_clean.reshape(-1, 2).std(axis=0).astype(np.float32),
            "feature_names": np.array(["P", "Q", "V_set", "is_slack",
                                       "is_PV", "is_PQ"]),
            "target_names": np.array(["theta [rad]", "|V| [p.u.]"])}


def bus_table(P, Q, Vm, theta, extra=None) -> str:
    """One operating point, per bus, as a Markdown table.

    ``P``, ``Q`` are the solved injections (positive into the network) and are
    split into generation and load; ``theta`` is in radians and printed in
    degrees. ``extra`` is an optional ``{column name: values}`` dict appended
    on the right, for a model's prediction beside the reference.
    """
    rows = []
    for i in range(N_BUS):
        row = [i, BUS_TYPE[i], BUS_NAMES[i],
               f"{max(P[i], 0):.3f}", f"{max(Q[i], 0):.3f}",
               f"{max(-P[i], 0):.3f}", f"{max(-Q[i], 0):.3f}",
               f"{Vm[i]:.4f}", f"{np.degrees(theta[i]):.2f}"]
        for vals in (extra or {}).values():
            row.append(f"{vals[i]:.4f}")
        rows.append(row)
    return error_table(rows, ["bus", "type", "what", "P_gen [p.u.]",
                              "Q_gen [p.u.]", "P_load [p.u.]", "Q_load [p.u.]",
                              "|V| [p.u.]", "theta [deg]"] + list(extra or {}))


def plot_power_flow(Vm_ref, theta_ref, predictions: Dict[str, Tuple],
                    title: str = ""):
    """The usual power-flow picture: |V| and angle per bus, then the error.

    Three panels sharing the bus axis. Top, the voltage profile in per unit
    with dashed limits at 0.95 and 1.05. Middle, the angle profile in degrees,
    the slack bus at 0. Bottom, each prediction's error. ``Vm_ref`` and
    ``theta_ref`` (radians) are the reference power flow; ``predictions``
    maps a label to ``(Vm, theta)`` in the same units.
    """
    bus = np.arange(N_BUS)
    k = len(predictions)
    w = 0.8 / (k + 1)
    colours = ["#1f77b4", "#d94f2b", "#0f9d58", "#8e44ad"]
    fig, axes = plt.subplots(3, 1, figsize=(8.2, 8.4), sharex=True)
    axes[0].bar(bus - 0.4 + w / 2, Vm_ref, w, color="#444444",
                label="reference (AC power flow)")
    axes[1].bar(bus - 0.4 + w / 2, np.degrees(theta_ref), w, color="#444444",
                label="reference (AC power flow)")
    for j, (name, (vm, th)) in enumerate(predictions.items()):
        x = bus - 0.4 + w * (j + 1.5)
        c = colours[j % len(colours)]
        axes[0].bar(x, vm, w, color=c, label=name)
        axes[1].bar(x, np.degrees(th), w, color=c, label=name)
        axes[2].plot(bus, np.degrees(np.asarray(th) - theta_ref), "o-", color=c,
                     label=f"{name}: angle error [deg]")
        axes[2].plot(bus, 100 * (np.asarray(vm) - Vm_ref), "s--", color=c,
                     label=f"{name}: |V| error [% of 1 p.u.]")
    axes[0].axhline(0.95, color="#777777", ls="--", lw=1.1)
    axes[0].axhline(1.05, color="#777777", ls="--", lw=1.1,
                    label="limits 0.95 and 1.05 p.u.")
    lo = min(0.93, np.min(Vm_ref) - 0.01)
    axes[0].set_ylim(lo, 1.07)
    axes[0].set_ylabel("|V| [p.u.]")
    axes[0].set_title(title or "Voltage profile")
    axes[1].axhline(0, color="#777777", lw=0.8)
    axes[1].set_ylabel("angle theta [deg]")
    axes[2].axhline(0, color="#777777", lw=0.8)
    axes[2].set_ylabel("prediction - reference")
    axes[2].set_xlabel("bus number")
    axes[2].set_xticks(bus)
    axes[2].set_xticklabels([f"{i}\n{BUS_TYPE[i]}" for i in bus])
    for a in axes:
        a.grid(alpha=0.25)
        a.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    return axes


# ── accuracy against speed: the AC power flow and a graph network, timed ───

def synthetic_grid(n: int, seed: int = 0) -> Dict[str, np.ndarray]:
    """A synthetic ``n``-bus transmission grid, for timing only.

    A ring of ``n`` buses plus chords, so the grid is meshed and each bus has
    about three lines, like a real transmission network. Each chord joins a bus
    to the one about sqrt(n) buses further round, which keeps the grid's
    diameter near sqrt(n). Line impedances are drawn around the six-bus
    network's (X from 0.05 to 0.10 p.u., X/R = 8). Bus 0 is the slack bus
    (1.03 p.u.); every fourth bus is a PV generator (1.02 p.u.); the others are
    PQ loads drawing 0.3 to 0.6 p.u. of P at a power factor near 0.95. The PV
    generators together supply the whole load, shared equally, so the power
    flows are local and the slack bus supplies only the losses — a grid of any
    size then stays within its voltage limits and the power flow converges.

    Returns ``lines`` (``(from, to, R, X)`` tuples), ``types``, ``v_set``, and
    the given ``P`` and ``Q`` of one operating point.
    """
    rng = np.random.default_rng(seed)
    edges = {(i, (i + 1) % n) for i in range(n)} if n > 2 else {(0, 1)}
    step = max(2, int(round(np.sqrt(n))))
    for i in range(0, n, 3):
        j = (i + step) % n
        if j != i:
            edges.add((min(i, j), max(i, j)))
    edges = sorted({(min(a, b), max(a, b)) for a, b in edges})
    x = rng.uniform(0.05, 0.10, size=len(edges))
    lines = tuple((a, b, float(xx) / 8.0, float(xx)) for (a, b), xx in zip(edges, x))

    types = np.array(["PQ"] * n, dtype=object)
    types[::4] = "PV"
    types[0] = "slack"
    v_set = np.where(types == "slack", 1.03, np.where(types == "PV", 1.02, 1.0))
    P = np.zeros(n)
    Q = np.zeros(n)
    pq = types == "PQ"
    P[pq] = -rng.uniform(0.3, 0.6, size=pq.sum())
    Q[pq] = 0.33 * P[pq]
    pv = types == "PV"
    P[pv] = (-P[pq].sum()) / max(pv.sum(), 1)
    return {"lines": lines, "types": types, "v_set": v_set, "P": P, "Q": Q}


def newton_raphson_dense(Y, types, v_set, P, Q, tol: float = 1e-8,
                         max_iter: int = 30):
    """:func:`newton_raphson` with dense NumPy matrices: faster on a small grid.

    ``Y`` is the admittance matrix as a dense complex array. On a few dozen
    buses the bookkeeping of sparse matrices costs more than it saves, so the
    timing uses whichever of the two is faster at each size.
    """
    types = np.asarray(types)
    pv = np.flatnonzero(types == "PV")
    pq = np.flatnonzero(types == "PQ")
    pvpq = np.r_[pv, pq]
    Vm = np.where(types == "PQ", 1.0, v_set).astype(float)
    th = np.zeros(len(types))
    for it in range(max_iter):
        V = Vm * np.exp(1j * th)
        I = Y @ V
        S = V * np.conj(I)
        mis = np.r_[P[pvpq] - S.real[pvpq], Q[pq] - S.imag[pq]]
        if np.max(np.abs(mis)) < tol:
            break
        dS_dth = 1j * V[:, None] * np.conj(np.diag(I) - Y * V[None, :])
        Vn = V / np.abs(V)
        dS_dVm = V[:, None] * np.conj(Y * Vn[None, :]) + np.diag(np.conj(I) * Vn)
        J = np.block([[dS_dth.real[np.ix_(pvpq, pvpq)], dS_dVm.real[np.ix_(pvpq, pq)]],
                      [dS_dth.imag[np.ix_(pq, pvpq)], dS_dVm.imag[np.ix_(pq, pq)]]])
        dx = np.linalg.solve(J, mis)
        th[pvpq] += dx[:len(pvpq)]
        Vm[pq] += dx[len(pvpq):]
    else:
        raise RuntimeError("AC power flow did not converge")
    return Vm, th, it


def newton_raphson(lines, types, v_set, P, Q, tol: float = 1e-8,
                   max_iter: int = 30, Y=None):
    r"""The AC power flow by Newton-Raphson, with sparse matrices.

    The same equations and the same method as :func:`ac_power_flow`, written
    for a grid of any size: the admittance matrix and the Jacobian are
    ``scipy.sparse`` matrices, and each Newton step is one sparse LU solve
    (``scipy.sparse.linalg.spsolve``). This is how production power-flow
    programs do it, and it is what makes the comparison with a graph network
    fair on a large grid. Pass a prebuilt ``Y`` to leave its construction out
    of a timing. Returns ``(Vm, theta, iterations)``.
    """
    import scipy.sparse as sp
    from scipy.sparse.linalg import spsolve
    types = np.asarray(types)
    n = len(types)
    if Y is None:
        Y = sparse_admittance(lines, n)
    pv = np.flatnonzero(types == "PV")
    pq = np.flatnonzero(types == "PQ")
    pvpq = np.r_[pv, pq]
    Vm = np.where(types == "PQ", 1.0, v_set).astype(float)
    th = np.zeros(n)
    V = Vm * np.exp(1j * th)
    for it in range(max_iter):
        I = Y @ V
        S = V * np.conj(I)
        mis = np.r_[P[pvpq] - S.real[pvpq], Q[pq] - S.imag[pq]]
        if np.max(np.abs(mis)) < tol:
            break
        dV = sp.diags(V)
        dS_dth = 1j * dV @ (sp.diags(I) - Y @ dV).conj()
        Vn = sp.diags(V / np.abs(V))
        dS_dVm = dV @ (Y @ Vn).conj() + sp.diags(np.conj(I)) @ Vn
        dS_dth, dS_dVm = dS_dth.tocsr(), dS_dVm.tocsr()
        J = sp.vstack([sp.hstack([dS_dth.real[pvpq][:, pvpq], dS_dVm.real[pvpq][:, pq]]),
                       sp.hstack([dS_dth.imag[pq][:, pvpq], dS_dVm.imag[pq][:, pq]])],
                      format="csc")
        dx = spsolve(J, mis)
        th[pvpq] += dx[:len(pvpq)]
        Vm[pq] += dx[len(pvpq):]
        V = Vm * np.exp(1j * th)
    else:
        raise RuntimeError("AC power flow did not converge")
    return Vm, th, it


def sparse_admittance(lines, n: int):
    """The bus admittance matrix of ``lines`` as a ``scipy.sparse`` CSR matrix."""
    import scipy.sparse as sp
    br = _branches(lines)
    a = np.array([l[0] for l in br])
    b = np.array([l[1] for l in br])
    y = 1.0 / (np.array([l[2] for l in br]) + 1j * np.array([l[3] for l in br]))
    rows = np.r_[a, b, a, b]
    cols = np.r_[b, a, a, b]
    vals = np.r_[-y, -y, y, y]
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))


class _TimingGNN(nn.Module):
    """Notebook 03's graph network, with the neighbour sum done over an edge list.

    Same layers — own state through one linear map, the sum of the neighbours'
    states through another, tanh — and the same sizes. The sum over neighbours
    is an ``index_add_`` over the lines instead of a product with the dense
    adjacency matrix, so the cost grows with the number of lines rather than
    with the number of buses squared: the fair way to run a GNN on a big grid.
    """

    def __init__(self, n_in=6, hidden=32, n_out=2, n_layers=3):
        super().__init__()
        sizes = [n_in] + [hidden] * n_layers
        self.self_lin = nn.ModuleList([nn.Linear(sizes[k], sizes[k + 1])
                                       for k in range(n_layers)])
        self.neigh_lin = nn.ModuleList([nn.Linear(sizes[k], sizes[k + 1], bias=False)
                                        for k in range(n_layers)])
        self.head = nn.Linear(hidden, n_out)

    def forward(self, src, dst, H):                  # H is (cases, buses, features)
        for ls, ln in zip(self.self_lin, self.neigh_lin):
            m = ln(H)
            agg = torch.zeros_like(m).index_add_(1, dst, m[:, src])
            H = torch.tanh(ls(H) + agg)
        return self.head(H)


def median_time(fn, repeats: int = 5, warmup: int = 2) -> float:
    """Seconds taken by ``fn()``: the median of ``repeats`` runs after ``warmup`` runs."""
    import time
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def time_power_flow(n: int, batch: int = 256, nr_cases: int = 5,
                    repeats: int = 5, device: str = "cpu") -> Dict[str, float]:
    """Time the AC power flow and an untrained graph network on an ``n``-bus grid.

    * **AC power flow**: Newton-Raphson from a flat start to a mismatch below
      1e-8 p.u., one case after another (each case has its own Jacobian, so
      there is nothing to batch). Timed both with sparse matrices
      (:func:`newton_raphson`) and, up to 400 buses, with dense ones
      (:func:`newton_raphson_dense`), and the faster is kept — the fair
      comparison. ``nr_cases`` operating points, all injections scaled by a
      common 0.9 to 1.1 and a per-bus 0.95 to 1.05; the admittance matrix is
      built once, outside the timing. Seconds per case.
    * **Graph network**: :class:`_TimingGNN` — 3 layers, 32 numbers per bus —
      with random weights: the cost of a forward pass does not depend on what
      the weights are, so training it first would change nothing here. Timed on
      one case (``single``) and on ``batch`` cases in one forward pass, divided
      by ``batch`` (``batched``). Seconds per case. Timed a second time with
      as many layers as the farthest bus is hops from the slack bus
      (``reach``): the depth a GNN needs for every bus to hear from every
      other, which grows with the grid.

    Each figure is the median of ``repeats`` runs after two warm-up runs.
    """
    g = synthetic_grid(n)
    types, v_set, lines = g["types"], g["v_set"], g["lines"]
    Y = sparse_admittance(lines, n)
    rng = np.random.default_rng(1)
    cases = []
    for _ in range(nr_cases):
        f = rng.uniform(0.9, 1.1) * rng.uniform(0.95, 1.05, n)
        cases.append((g["P"] * f, g["Q"] * f))
    iters = []

    def run_sparse():
        iters.clear()
        for P, Q in cases:
            iters.append(newton_raphson(lines, types, v_set, P, Q, Y=Y)[2])

    t_sparse = median_time(run_sparse, repeats=repeats, warmup=1) / nr_cases
    t_dense = np.inf
    if n <= 400:                        # beyond that a dense solve is only slower
        Yd = Y.toarray()

        def run_dense():
            for P, Q in cases:
                newton_raphson_dense(Yd, types, v_set, P, Q)

        t_dense = median_time(run_dense, repeats=repeats, warmup=1) / nr_cases
    t_nr = min(t_sparse, t_dense)

    br = _branches(lines)
    src = torch.tensor([l[0] for l in br] + [l[1] for l in br], device=device)
    dst = torch.tensor([l[1] for l in br] + [l[0] for l in br], device=device)
    torch.manual_seed(0)
    X1 = torch.randn(1, n, 6, device=device)
    XB = torch.randn(batch, n, 6, device=device)
    hops = _hops_from(0, n, br)

    def gnn(model, Xt):
        def f():
            with torch.no_grad():
                model(src, dst, Xt)
            if device != "cpu":
                torch.cuda.synchronize()
        return f

    three = _TimingGNN(n_layers=3).to(device).eval()
    reach = _TimingGNN(n_layers=max(3, hops)).to(device).eval()
    return {"n": n, "lines": len(lines), "hops": hops,
            "nr_iterations": float(np.mean(iters)),
            "t_nr": t_nr, "t_nr_dense": float(t_dense), "t_nr_sparse": t_sparse,
            "t_gnn_single": median_time(gnn(three, X1), repeats=repeats),
            "t_gnn_batched": median_time(gnn(three, XB), repeats=repeats) / batch,
            "t_gnn_reach_single": median_time(gnn(reach, X1), repeats=repeats),
            "t_gnn_reach_batched": median_time(gnn(reach, XB), repeats=repeats) / batch}


def _hops_from(start: int, n: int, br) -> int:
    """How many lines the farthest bus is from ``start`` (breadth-first search)."""
    nbrs = [[] for _ in range(n)]
    for a, b, _, _ in br:
        nbrs[a].append(b)
        nbrs[b].append(a)
    dist = np.full(n, -1)
    dist[start] = 0
    frontier = [start]
    while frontier:
        nxt = []
        for i in frontier:
            for j in nbrs[i]:
                if dist[j] < 0:
                    dist[j] = dist[i] + 1
                    nxt.append(j)
        frontier = nxt
    return int(dist.max())


def speed_sweep(sizes=(6, 12, 24, 48, 96, 192), **kw) -> List[Dict[str, float]]:
    """:func:`time_power_flow` for each grid size, printed as it goes."""
    out = []
    for n in sizes:
        r = time_power_flow(n, **kw)
        out.append(r)
        print(f"  {n:5d} buses, {r['lines']:5d} lines, {r['hops']:3d} hops:  AC power flow "
              f"{1e3 * r['t_nr']:8.3f} ms   GNN one case {1e3 * r['t_gnn_single']:7.3f} ms"
              f"   GNN batched {1e3 * r['t_gnn_batched']:7.4f} ms"
              f"   GNN batched, {max(3, r['hops'])} layers (one per hop) {1e3 * r['t_gnn_reach_batched']:8.4f} ms")
    return out


def crossover(sweep, key: str) -> Optional[float]:
    """The grid size at which the GNN's ``key`` time first drops below Newton-Raphson's.

    Interpolated on log axes between the last size where the power flow was
    faster and the first where the GNN was. ``None`` if the GNN is never
    faster in the sweep; the smallest size if it is faster everywhere.
    """
    n = np.array([r["n"] for r in sweep], dtype=float)
    d = np.log(np.array([r[key] for r in sweep])) - np.log(np.array([r["t_nr"] for r in sweep]))
    if d[0] < 0:
        return float(n[0])
    for k in range(1, len(n)):
        if d[k] < 0:
            f = d[k - 1] / (d[k - 1] - d[k])
            return float(np.exp(np.log(n[k - 1]) + f * (np.log(n[k]) - np.log(n[k - 1]))))
    return None


def plot_speed(sweep, ax=None, title: str = "Time per case against grid size"):
    """Log-log plot of seconds per case against buses, crossovers marked."""
    ax = _new_axes(ax, figsize=(8.4, 5.2))
    n = [r["n"] for r in sweep]
    ax.loglog(n, [r["t_nr"] for r in sweep], "o-", color="#444444", lw=1.8,
              label="AC power flow (Newton-Raphson, dense or sparse, the faster)")
    ax.loglog(n, [r["t_gnn_single"] for r in sweep], "s-", color="#1f77b4", lw=1.8,
              label="graph network, one case at a time")
    ax.loglog(n, [r["t_gnn_batched"] for r in sweep], "^-", color="#d94f2b", lw=1.8,
              label="graph network, cases in a batch")
    ax.loglog(n, [r["t_gnn_reach_batched"] for r in sweep], "v--", color="#8e44ad", lw=1.8,
              label="graph network, batch, one layer per hop of the grid")
    names = {"t_gnn_single": "one case", "t_gnn_batched": "batched",
             "t_gnn_reach_batched": "batched, one layer per hop"}
    for key, c in (("t_gnn_single", "#1f77b4"), ("t_gnn_batched", "#d94f2b"),
                   ("t_gnn_reach_batched", "#8e44ad")):
        x = crossover(sweep, key)
        mode = names[key]
        if x is None:
            ax.plot([], [], " ", label=f"{mode}: the GNN is never faster here")
        elif x > n[0]:
            ax.axvline(x, color=c, ls="--", lw=1.1,
                       label=f"crossover, {mode}: about {x:.0f} buses")
        else:
            ax.plot([], [], " ", label=f"{mode}: the GNN is faster at every size here")
    ax.set_xlabel("buses in the grid, N")
    ax.set_ylabel("time per case [s]")
    ax.set_title(title)
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=False, fontsize=8)
    return ax


def normalised_adjacency(A: np.ndarray) -> np.ndarray:
    r"""Kipf and Welling's :math:`\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}`.

    Self-loops are added first (:math:`\tilde{A} = A + I`), so a node keeps its
    own features, and then rows and columns are scaled by the square root of
    the degree, so that a high-degree node does not simply shout louder than a
    low-degree one. Notebooks 02 and 03 no longer use it — their layer is plain
    message passing over ``A`` — and it is kept for the environment check.
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
    ax.set_xlabel("time [h]")
    ax.set_ylabel("demand [p.u.]")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    return ax


def plot_forecast(y_true: np.ndarray, curves: Dict[str, np.ndarray],
                  ax=None, title: str = "", n: int = 120):
    """The first ``n`` steps of the held-out target and one or more forecasts.

    The curves are drawn in the order the dictionary gives them.
    """
    ax = _new_axes(ax, figsize=(9.0, 3.6))
    y_true = np.asarray(y_true).ravel()[:n]
    ax.plot(y_true, lw=2.0, color="#111111", label="measured")
    for i, (label, values) in enumerate(curves.items()):
        ax.plot(np.asarray(values).ravel()[:n], lw=1.5,
                color=_CYCLE[i % len(_CYCLE)], label=label)
    ax.set_xlabel("time into the held-out period [h]")
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
