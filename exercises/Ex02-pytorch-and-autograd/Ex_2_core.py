"""Ex_2 - PyTorch and autograd core library.

*Deep Learning for Engineering* — MSc, Aalborg University.
Remus Teodorescu (ret@et.aau.dk), with support from Research Assistant
Noman Khan (nomank@energy.aau.dk). The code here is original to this course;
see docs/PROVENANCE.md for what each reference text is cited for.

Problem-independent machinery for the second exercise set: seeding, tensor
conversion, the two autograd helpers, a pass/fail checker, a small multi-layer
perceptron, a finite-difference reference, and the plotting conventions.

The two helpers that matter are `grad` and `d2`. They are deliberately the same
two functions, with the same names and the same signatures, that appear in the
core library of every Part 2 exercise:

    grad(y, x)          dy/dx, differentiating a network output with respect
                        to its INPUT, with the graph kept so it can be
                        differentiated again
    d2(y, x)            the second derivative, by calling grad twice

Everything in L7 to L12 - the Poisson residual, the heat equation, the
Navier-Stokes momentum balance, the power-flow equations - is those two
functions applied to a network's output and combined into a loss. Notebook 02
is where you meet them, and it is the most important notebook in Part 1.

    DEVICE, DTYPE           the conventions used everywhere in this course
    set_seed                seeds Python, NumPy and PyTorch in one call
    as_input                a column tensor of shape (n, 1) that requires grad
    grad, d2                first and second derivatives w.r.t. an input
    describe_tensor         shape, dtype, device, requires_grad, grad_fn
    check, check_shape      pass/fail with the number that decided it
    finite_difference       a derivative that is NOT autograd, for comparison
    mlp, count_parameters   a small tanh network and its parameter count
    vibration_data          the curve fitted in notebook 03
    engineering_axes        axis labels with units, grid, no box
    plot_loss               loss against epoch, on a log axis, always

Usage in a notebook:

    from Ex_2_core import *          # or:  import Ex_2_core as core

===============================================================================
HOW TO RUN Ex_2
===============================================================================

WHAT YOU NEED

    Python 3.9+, with:  torch  numpy  matplotlib
    Google Colab has all three preinstalled. Nothing else is required, and no
    GPU is needed - every problem in this exercise runs on a CPU in seconds.
    The largest network trained here has about two thousand parameters.

    If you are working locally and torch is missing, the CPU build is the one
    you want:

        pip install torch --index-url https://download.pytorch.org/whl/cpu

FILES IN THIS EXERCISE

    modules (read, do not edit):
      Ex_2_core.py

    notebooks (run in this order):
      Ex02_00_environment_check.ipynb     environment check, read-only
      Ex02_01_tensors.ipynb               TODOs
      Ex02_02_autograd_by_hand.ipynb      TODOs - the important one
      Ex02_03_training_loop.ipynb         TODOs

A NOTE ON THE FILE NAMES

    Notebooks are named with the exercise number, Ex01_ / Ex02_, but the
    importable module uses underscores: Ex_2_core.py. This is not an
    inconsistency for its own sake - a Python module name cannot contain a
    dot, because a dot means "inside a package". The import in every setup
    cell therefore reads Ex_2_core, with underscores.

RUNNING ON GOOGLE COLAB

    1. Upload ALL files from this folder to your Colab session:
       open the Files pane (folder icon, left-hand side) and drag them in.
    2. Open notebook 00 and run it top to bottom. It checks your environment
       and will tell you if something is missing.
    3. Work through the numbered notebooks in order.

    Uploaded files vanish when the Colab runtime restarts. If you get
    "FileNotFoundError" partway through a session, re-upload the module -
    the first cell of every notebook will prompt you automatically.

    To avoid re-uploading, put the folder in Google Drive and mount it:

        from google.colab import drive
        drive.mount('/content/drive')
        %cd /content/drive/MyDrive/Ex02-pytorch-and-autograd

RUNNING LOCALLY (Jupyter, VS Code)

    Keep every file in one folder and launch from that folder:

        cd Ex02-pytorch-and-autograd
        jupyter lab

    The notebooks import the module by name, so the working directory must be
    the folder containing it.

HOW THE EXERCISE IS STRUCTURED

    The module is complete and working. You are not asked to rewrite it: it
    contains nothing specific to any one problem - seeding, conversion,
    plotting, and the two autograd helpers that every Part 2 exercise also
    imports.

    Your work is in the numbered notebooks, marked by

        # TODO: ...

    and a raise NotImplementedError immediately below. Delete the raise, write
    your answer in its place, and run the cell. They are short - a few lines
    each - and that is the point.

    Ex_2 carries far more explanatory text than the Part 2 exercises do. That
    is deliberate: L2 is a recorded lecture, and a student who missed it should
    still be able to work through this from the notebook alone.

    Run notebooks in order. Notebook 02 assumes the tensor vocabulary of 01,
    and notebook 03 assumes the autograd of 02.

WHY NOTEBOOK 02 IS THE ONE THAT MATTERS

    Almost everyone arrives believing that autograd is a neural-network
    mechanism: something that computes gradients of a loss with respect to
    weights so that an optimiser can take a step. That is one use of it and it
    is not what it is.

    Autograd is a general differentiation engine. It will differentiate any
    expression built from differentiable operations, with respect to any tensor
    that took part in it - including the INPUT. So if u is a network's output
    at coordinate x, then

        u_x  = grad(u, x)
        u_xx = grad(u_x, x)

    are the first and second derivatives of the network's output field, exact
    to machine precision, with no finite differences and no mesh. Write down

        u_xx + k**2 * u

    and you have the residual of the Helmholtz equation, evaluated anywhere you
    like. Minimise its square and you have solved a differential equation with
    no data at all.

    That is the whole idea of Part 2, and notebook 02 is where it is planted.
    Do not skip it, and do not skip the last section of it, which shows that
    the second derivative of a ReLU network is identically zero - the reason
    every network in L7 to L12 uses tanh.

EXPECTED RUNTIME

    Notebooks 00 to 02 are instant: no training happens in any of them.
    Notebook 03 runs six short trainings - the main fit, a deliberately broken
    one, three learning rates - each a thousand to three thousand Adam steps on
    a network of about 2,200 parameters. That is well under a minute in total
    on a CPU. Budget half an hour for the set, nearly all of it reading.

TROUBLESHOOTING

    NameError: name 'null' is not defined
        You ran a .ipynb file as if it were Python, e.g. with %run. Notebooks
        are opened, not executed as source. Use the import in the setup cell.

    FileNotFoundError: ... must sit next to this notebook
        The module is not in the working directory. On Colab, re-upload.
        Locally, launch Jupyter from the folder containing every file.

    NotImplementedError
        Expected. You have reached a TODO cell that is yours to complete.

    RuntimeError: element 0 of tensors does not require grad and does not have
    a grad_fn
        You asked for the derivative with respect to a tensor that was not
        marked. Create it with requires_grad=True, or use as_input().

    RuntimeError: Trying to backward through the graph a second time
        The graph was freed after the first backward pass. Either you meant to
        pass create_graph=True - which is what grad() does, and what makes a
        second derivative possible - or you are calling backward twice on one
        forward pass, which is a real bug.

    grad returns None, or a tensor of zeros
        The output does not actually depend on the input you differentiated
        with respect to, or you detached the graph somewhere in between.
        .detach(), .item(), .numpy() and torch.no_grad() all cut the graph.

    The second derivative is exactly zero everywhere
        If your activation is ReLU, that is the correct answer and it is the
        subject of the last section of notebook 02. If it is tanh, you have
        lost the graph: the first grad call needs create_graph=True.

    Loss does not fall at all
        Nine times in ten it is a missing optimizer.zero_grad(), so gradients
        accumulate across steps, or a learning rate that is orders of magnitude
        too large. Print the loss every hundred steps and look at the shape of
        the curve, not at one number.

    Loss becomes nan
        A learning rate far too large, or a division by zero in the loss.
        Halve the learning rate first; it is the cause more often than
        anything else.

    Results differ slightly from the numbers quoted in the notebook
        Expected. A different torch version or a different machine moves the
        last digits. What must hold is the ordering of results and anything
        that should be exactly zero.
"""

from __future__ import annotations

import random
import sys

import numpy as np
import torch
from torch import nn

__all__ = [
    "personal_seed",
    "DEVICE",
    "DTYPE",
    "SEED",
    "as_input",
    "banner",
    "check",
    "check_shape",
    "count_parameters",
    "d2",
    "describe_tensor",
    "engineering_axes",
    "finite_difference",
    "grad",
    "mlp",
    "plot_loss",
    "set_seed",
    "to_numpy",
    "vibration_data",
    "vibration_truth",
]

SEED = 88

# Every problem in Ex_2 is small enough that moving it to a GPU would cost more
# in transfers than it saves in arithmetic, so the device is fixed to the CPU.
# Part 2 uses the same name for a device that may well be a GPU; notebook 01
# shows how that line is normally written.
DEVICE = torch.device("cpu")

# PyTorch's default floating type is float32, and NumPy's is float64. Mixing
# them raises in some places and silently downcasts in others, so this course
# states the type explicitly everywhere.
DTYPE = torch.float32


# ---------------------------------------------------------------------------
# printing and seeding
# ---------------------------------------------------------------------------

def banner(text: str, width: int = 74) -> None:
    """Print a section heading, so a long cell output stays readable."""
    print("=" * width)
    print(text)
    print("=" * width)


def set_seed(seed: int = SEED) -> None:
    """Seed Python, NumPy and PyTorch so that a run is reproducible.

    Three generators have to be seeded, because three libraries draw from
    three different streams: Python's `random`, NumPy's global generator, and
    PyTorch's, which is the one that decides your initial weights.

    Reproducibility on one machine is what this buys. It does not make results
    bit-identical across torch versions or across CPU and GPU.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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


# ---------------------------------------------------------------------------
# tensors
# ---------------------------------------------------------------------------

def to_numpy(t) -> np.ndarray:
    """Detach a tensor from the graph and return it as a NumPy array.

    The three steps matter and are worth reading in order: `.detach()` cuts the
    tensor out of the autograd graph, `.cpu()` moves it to host memory if it
    was elsewhere, `.numpy()` shares the buffer. Forget `.detach()` on a tensor
    that requires grad and PyTorch raises rather than silently handing you
    something that cannot be differentiated.
    """
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return np.asarray(t)


def as_input(x, requires_grad: bool = True) -> torch.Tensor:
    """Return `x` as a column tensor of shape (n, 1) that requires grad.

    Shape (n, 1) rather than (n,) is the convention of every exercise in this
    course, because a network with one input takes a batch of n samples each of
    dimension 1, and `nn.Linear` expects that trailing feature axis. Keeping
    the convention everywhere means the (n,) against (n, 1) broadcasting trap
    from Ex_1 notebook 02 has no opportunity to appear.

    requires_grad=True marks the tensor as something autograd should track, so
    that grad(u, x) is possible. That single flag is the difference between an
    array library and a differentiation engine.
    """
    t = torch.as_tensor(x, dtype=DTYPE, device=DEVICE).detach().reshape(-1, 1).clone()
    t.requires_grad_(requires_grad)
    return t


def describe_tensor(t: torch.Tensor, name: str = "tensor") -> None:
    """Print everything worth knowing about a tensor, shape first."""
    print(f"  {name}")
    print(f"      shape          {tuple(t.shape)}")
    print(f"      dtype          {t.dtype}")
    print(f"      device         {t.device}")
    print(f"      requires_grad  {t.requires_grad}")
    print(f"      is_leaf        {t.is_leaf}")
    print(f"      grad_fn        {t.grad_fn}")


# ---------------------------------------------------------------------------
# autograd - the two functions the whole of Part 2 is built on
# ---------------------------------------------------------------------------

def grad(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """dy/dx, with the graph kept so that the result can be differentiated again.

    `torch.autograd.grad` computes a vector-Jacobian product, not a Jacobian.
    `grad_outputs=ones_like(y)` selects the vector of ones, which for an
    elementwise map y_i = f(x_i) gives exactly the vector of derivatives
    dy_i/dx_i - one derivative per sample, which is what a residual needs.

    `create_graph=True` is what makes second derivatives possible: it records
    the operations performed while computing the gradient, so the gradient is
    itself a differentiable function of x. Without it, d2 returns None or
    raises, and this is the single most common mistake in a first PINN.

    `allow_unused=True`, with zeros substituted for a missing gradient, is here
    for one specific reason: ReLU. The first derivative of a ReLU network does
    not depend on x anywhere in the graph - it is a product of weight matrices
    and a constant mask - so asking for a second derivative would otherwise
    raise rather than return the mathematically correct zero. See the last
    section of notebook 02. The price is that differentiating with respect to a
    genuinely unrelated tensor now returns zeros instead of complaining, so if
    a derivative surprises you by being exactly zero, check first that the
    output really does depend on that input.

    Returns a tensor with the same shape as x.
    """
    g = torch.autograd.grad(
        y, x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )[0]
    return torch.zeros_like(x) if g is None else g


def d2(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """d2y/dx2, by differentiating twice. Same shape as x."""
    return grad(grad(y, x), x)


def finite_difference(f, x, h: float = 1e-4) -> np.ndarray:
    """Central-difference derivative of a plain NumPy function, for comparison.

    This is the method autograd replaces. It is deliberately included so that
    notebook 02 can show what you give up: the answer depends on h, it loses
    accuracy at both ends of the h range - truncation error for large h,
    cancellation in floating point for small h - and it costs two function
    evaluations per derivative per dimension, which is what makes it hopeless
    for a network with three inputs and a second-order operator.
    """
    x = np.asarray(x, dtype=float)
    return (f(x + h) - f(x - h)) / (2.0 * h)


# ---------------------------------------------------------------------------
# checking
# ---------------------------------------------------------------------------

def check(name: str, got, want, tol: float = 1e-5) -> bool:
    """Compare a computed value against the expected one and report.

    Accepts tensors, arrays and scalars in any combination, and prints the
    largest absolute difference, which is the number that decided the verdict.

    The default tolerance is 1e-5 rather than the 1e-10 of Ex_1 because these
    are float32 tensors: float32 carries roughly seven significant decimal
    digits, so agreement to 1e-5 on numbers of order one is as good as it gets.
    Asking for more is asking the arithmetic for something it does not have.
    """
    g = np.asarray(to_numpy(got), dtype=float)
    w = np.asarray(to_numpy(want), dtype=float)
    if g.shape != w.shape:
        print(f"  FAIL  {name}: shape {g.shape}, expected {w.shape}")
        return False
    err = float(np.max(np.abs(g - w))) if g.size else 0.0
    verdict = "PASS" if err <= tol else "FAIL"
    print(f"  {verdict}  {name}: max abs error {err:.3e}  (tolerance {tol:.1e})")
    return err <= tol


def check_shape(name: str, got, want_shape) -> bool:
    """Compare a tensor's or array's shape against the expected one."""
    shape = tuple(got.shape)
    want = tuple(int(v) for v in want_shape)
    verdict = "PASS" if shape == want else "FAIL"
    print(f"  {verdict}  {name}: shape {shape}, expected {want}")
    return shape == want


# ---------------------------------------------------------------------------
# a small network
# ---------------------------------------------------------------------------

def mlp(n_in: int = 1, n_out: int = 1, n_hidden: int = 32, n_layers: int = 3,
        activation: str = "tanh") -> nn.Sequential:
    """A plain fully connected network: n_layers hidden layers of n_hidden units.

    `activation` is "tanh" or "relu". The default is tanh, and the reason is
    the subject of the last section of notebook 02: a ReLU network is
    piecewise linear, so its second derivative is zero almost everywhere and it
    cannot represent the curvature that a second-order differential operator
    asks for. Every network in L7 to L12 uses tanh for that reason and no
    other.

    The network is small on purpose. Four hidden layers of 20 to 50 units is
    the usual size for the whole of Part 2, and a bigger one is generally a
    slower way of getting the same answer.
    """
    act = {"tanh": nn.Tanh, "relu": nn.ReLU, "sigmoid": nn.Sigmoid}[activation]
    layers: list = [nn.Linear(n_in, n_hidden), act()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(n_hidden, n_hidden), act()]
    layers += [nn.Linear(n_hidden, n_out)]
    return nn.Sequential(*layers).to(DEVICE)


def count_parameters(model: nn.Module) -> int:
    """Total number of trainable parameters, weights and biases together."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# the dataset fitted in notebook 03
# ---------------------------------------------------------------------------

def vibration_data(n: int = 60, noise: float = 0.03, seed: int = SEED,
                   t_max: float = 2.0):
    """Free vibration of a lightly damped single-degree-of-freedom system.

        y(t) = exp(-0.5 t) sin(2 pi t)

    which is an accelerometer trace after an impact test: two cycles, decaying.
    Returns (t, y) as float32 column tensors of shape (n, 1), t sorted, y with
    additive Gaussian measurement noise of standard deviation `noise` in the
    same units as y.

    A straight line cannot fit this and neither can a quadratic, so it is a
    fair test of whether the network in notebook 03 learned anything - while
    still being small enough to fit in a few thousand Adam steps on a CPU.
    """
    g = torch.Generator().manual_seed(seed)
    t = torch.rand(n, 1, generator=g) * t_max
    t, _ = torch.sort(t, dim=0)
    clean = torch.exp(-0.5 * t) * torch.sin(2 * np.pi * t)
    y = clean + noise * torch.randn(n, 1, generator=g)
    return t.to(DTYPE), y.to(DTYPE)


def vibration_truth(t):
    """The noise-free curve behind vibration_data, for a NumPy array of times."""
    t = np.asarray(t, dtype=float)
    return np.exp(-0.5 * t) * np.sin(2 * np.pi * t)


# ---------------------------------------------------------------------------
# plotting conventions
# ---------------------------------------------------------------------------

def engineering_axes(ax, xlabel: str, ylabel: str, title: str | None = None,
                     grid: bool = True, legend: bool = False):
    """Apply this course's figure conventions to one matplotlib Axes.

    The same helper as in Ex_1: labels carry units, the grid is light, the box
    is gone. An engineering figure with an unlabelled axis is a decoration.
    """
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if grid:
        ax.grid(True, alpha=0.3, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if legend:
        ax.legend(frameon=False)
    return ax


def plot_loss(history, ax=None, label: str = "loss", title: str | None = None):
    """Plot a loss history against epoch, on a logarithmic y-axis.

    Always logarithmic. A loss falls by orders of magnitude, and on a linear
    axis everything after the first decade is pressed flat against the
    baseline: you see a horizontal line and conclude that training stalled,
    when in fact the loss fell by a factor of a thousand. Every loss curve in
    this course, and in Part 2, is plotted this way.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.plot(np.arange(1, len(history) + 1), np.asarray(history), label=label)
    ax.set_yscale("log")
    engineering_axes(ax, "epoch [-]", "loss [-]", title=title, legend=True)
    return ax
