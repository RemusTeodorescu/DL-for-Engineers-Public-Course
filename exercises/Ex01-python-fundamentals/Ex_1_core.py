"""Ex_1 - Python fundamentals core library.

*Deep Learning for Engineering* — MSc, Aalborg University.
Remus Teodorescu (ret@et.aau.dk), with support from Research Assistant
Noman Khan (nomank@energy.aau.dk). The code here is original to this course;
see docs/PROVENANCE.md for what each reference text is cited for.

Problem-independent machinery for the first exercise set: environment reporting,
array description, the broadcasting alignment table, a pass/fail checker, a
timing helper, small engineering datasets and the plotting conventions used for
every figure in this course.

Nothing here is specific to deep learning. The same file would serve any first
week of scientific Python. It exists so that the numbered notebooks contain
only the ideas you are meant to think about.

    banner                  section headings, so printed output stays readable
    check_environment       versions and a verdict, used by notebook 00
    set_seed                reproducibility, one call at the top of every run
    describe                shape, dtype, ndim, size, memory for one array
    broadcast_table         the right-aligned alignment table - the core device
    can_broadcast           the same question as a boolean
    check / check_shape     pass/fail with the number that decides it
    time_call               a fair best-of-N timing of a callable
    compare_timings         a timing table with speed-up factors
    beam_deflection         cantilever under uniform load, the running example
    measured_deflection     the same, with instrument noise
    sensor_matrix           a (n_sensors, n_samples) array for broadcasting work
    engineering_axes        axis labels with units, grid, sensible limits
    traceback_demo          raises from three frames down, on purpose
    broken_average          a function with a real bug, for you to diagnose

Usage in a notebook:

    from Ex_1_core import *          # or:  import Ex_1_core as core

===============================================================================
HOW TO RUN Ex_1
===============================================================================

WHAT YOU NEED

    Python 3.9+, with:  numpy  matplotlib
    Notebook 00 also looks for torch, because Ex_2 needs it and it is better to
    find out now than in week two. Nothing in Ex_1 imports torch.

    Google Colab has all three preinstalled. No GPU is needed - nothing in this
    exercise takes longer than a few seconds to run.

FILES IN THIS EXERCISE

    modules (read, do not edit):
      Ex_1_core.py

    notebooks (run in this order):
      Ex01_00_environment_check.ipynb      environment check, read-only
      Ex01_01_python_basics.ipynb          TODOs
      Ex01_02_numpy_and_broadcasting.ipynb TODOs - the important one
      Ex01_03_plotting.ipynb               TODOs

A NOTE ON THE FILE NAMES

    Notebooks are named with the exercise number, Ex01_ / Ex02_, but the
    importable module uses underscores: Ex_1_core.py. This is not an
    inconsistency for its own sake - a Python module name cannot contain a
    dot, because a dot means "inside a package". The import in every setup
    cell therefore reads Ex_1_core, with underscores.

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
        %cd /content/drive/MyDrive/Ex01-python-fundamentals

RUNNING LOCALLY (Jupyter, VS Code)

    Keep every file in one folder and launch from that folder:

        cd Ex01-python-fundamentals
        jupyter lab

    The notebooks import the module by name, so the working directory must be
    the folder containing it.

HOW THE EXERCISE IS STRUCTURED

    The module is complete and working. You are not asked to rewrite it: it
    contains nothing that carries an idea - printing, timing, plotting
    conventions and two small synthetic datasets.

    Your work is in the numbered notebooks, marked by

        # TODO: ...

    and a raise NotImplementedError immediately below. Delete the raise, write
    your answer in its place, and run the cell. Each one is a few lines.

    Ex_1 carries far more explanatory text than the Part 2 exercises do. That
    is deliberate. It is written to be readable on its own, so that a student
    who missed the recorded lecture can still work through it. From Ex_7
    onwards the prose thins out considerably.

    Run notebooks in order. Notebook 02 assumes the vocabulary of notebook 01,
    and notebook 03 plots arrays built the way notebook 02 builds them.

WHY BROADCASTING GETS A WHOLE NOTEBOOK

    Every shape bug in the Part 2 exercises - and there will be many - is a
    broadcasting misunderstanding wearing a disguise. The specific one that
    costs the most time is the difference between an array of shape (N,) and
    one of shape (N, 1): subtract them and NumPy hands you an (N, N) matrix
    without complaint, your loss becomes a number that is silently wrong, and
    nothing raises. Twenty minutes here saves an exercise session in week
    seven.

EXPECTED RUNTIME

    Under five minutes per notebook, including the timed comparison in
    notebook 02. The difficulty is conceptual, not computational.

TROUBLESHOOTING

    NameError: name 'null' is not defined
        You ran a .ipynb file as if it were Python, e.g. with %run. Notebooks
        are opened, not executed as source. Use the import in the setup cell.

    FileNotFoundError: ... must sit next to this notebook
        The module is not in the working directory. On Colab, re-upload.
        Locally, launch Jupyter from the folder containing every file.

    NotImplementedError
        Expected. You have reached a TODO cell that is yours to complete.
        Delete the raise and write your answer above it.

    NameError: name 'x' is not defined, in a cell that used to work
        You restarted the kernel, or ran cells out of order. A notebook keeps
        one namespace for the whole file, and it is built by running cells top
        to bottom. Use Run All rather than hunting for the missing cell.

    ValueError: operands could not be broadcast together with shapes ...
        The good case: NumPy caught it. Print both shapes, then feed them to
        broadcast_table() and read the alignment.

    A result that is an (N, N) matrix when you expected a vector of length N
        The bad case: broadcasting succeeded and gave you something you did not
        want. This is the (N,) versus (N, 1) trap. Print .shape at every step.

    Figures do not appear
        In Jupyter and Colab a figure is drawn when the cell ends. If you build
        a figure across several cells it will not show. Keep one figure to one
        cell, and call plt.show() at the end of it.
"""

from __future__ import annotations

import random
import sys
import time
from typing import Callable, Iterable, Sequence

import numpy as np

__all__ = [
    "personal_seed",
    "SEED",
    "banner",
    "beam_deflection",
    "broadcast_table",
    "broken_average",
    "can_broadcast",
    "check",
    "check_environment",
    "check_shape",
    "compare_timings",
    "describe",
    "engineering_axes",
    "measured_deflection",
    "sensor_matrix",
    "set_seed",
    "time_call",
    "traceback_demo",
]

SEED = 88


# ---------------------------------------------------------------------------
# printing
# ---------------------------------------------------------------------------

def banner(text: str, width: int = 74) -> None:
    """Print a section heading, so a long cell output stays readable."""
    print("=" * width)
    print(text)
    print("=" * width)


# ---------------------------------------------------------------------------
# environment
# ---------------------------------------------------------------------------

def check_environment(verbose: bool = True) -> dict:
    """Report the interpreter and package versions, and return them as a dict.

    torch is reported but not required: nothing in Ex_1 imports it. It is
    checked here so that a missing install is discovered in week one rather
    than in the middle of Ex_2.
    """
    info = {
        "python": ".".join(str(v) for v in sys.version_info[:3]),
        "numpy": np.__version__,
        "matplotlib": None,
        "torch": None,
        "colab": "google.colab" in sys.modules,
    }
    try:
        import matplotlib

        info["matplotlib"] = matplotlib.__version__
    except ImportError:
        pass
    try:
        import torch

        info["torch"] = torch.__version__
    except ImportError:
        pass

    if verbose:
        banner("Environment")
        print(f"  python      {info['python']}")
        print(f"  numpy       {info['numpy']}")
        print(f"  matplotlib  {info['matplotlib'] or 'MISSING'}")
        print(f"  torch       {info['torch'] or 'not installed (needed from Ex_2)'}")
        print(f"  on Colab    {info['colab']}")
        print()
        major, minor = sys.version_info[:2]
        ok = (major, minor) >= (3, 9) and info["matplotlib"] is not None
        print("  verdict:    " + ("ready for Ex_1" if ok else "something is missing, see above"))
    return info


def set_seed(seed: int = SEED) -> None:
    """Seed Python's and NumPy's generators so a run is reproducible.

    Call this once, at the top of a notebook. Reproducibility is not a nicety:
    without it you cannot tell a real change in a result from the noise of a
    different random draw.
    """
    random.seed(seed)
    np.random.seed(seed)

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
# arrays: describing and checking
# ---------------------------------------------------------------------------

def describe(a, name: str = "array") -> None:
    """Print everything worth knowing about an array in one line block.

    Shape first, because shape is what goes wrong.
    """
    a = np.asarray(a)
    print(f"  {name}")
    print(f"      shape   {a.shape}")
    print(f"      ndim    {a.ndim}")
    print(f"      dtype   {a.dtype}")
    print(f"      size    {a.size} elements, {a.nbytes} bytes")
    flat = a.ravel()
    preview = ", ".join(f"{v}" for v in flat[:4])
    if flat.size > 4:
        preview += ", ..."
    print(f"      first   [{preview}]")


def can_broadcast(shape_a: Sequence[int], shape_b: Sequence[int]) -> bool:
    """True if two shapes broadcast against each other."""
    for x, y in zip(reversed(tuple(shape_a)), reversed(tuple(shape_b))):
        if x != y and x != 1 and y != 1:
            return False
    return True


def broadcast_table(shape_a: Sequence[int], shape_b: Sequence[int]) -> tuple | None:
    """Print the alignment table for two shapes and return the result shape.

    This is the single most useful diagnostic in the whole of Part 1. It writes
    the two shapes out right-aligned, pads the shorter one with 1s, and marks
    each column as compatible or not. Returns the broadcast shape, or None if
    the shapes are incompatible.

        >>> broadcast_table((3, 1), (4,))
        ...
        result: (3, 4)
    """
    a = tuple(int(v) for v in shape_a)
    b = tuple(int(v) for v in shape_b)
    n = max(len(a), len(b))
    pa = (1,) * (n - len(a)) + a
    pb = (1,) * (n - len(b)) + b

    cols, out, ok = [], [], True
    for x, y in zip(pa, pb):
        if x == y:
            cols.append("same")
            out.append(x)
        elif x == 1:
            cols.append("stretch A")
            out.append(y)
        elif y == 1:
            cols.append("stretch B")
            out.append(x)
        else:
            cols.append("CLASH")
            out.append(None)
            ok = False

    w = max(8, max(len(str(v)) for v in pa + pb) + 2, max(len(c) for c in cols) + 2)
    head = "".join(f"{'dim -' + str(n - i):>{w}}" for i in range(n))
    print(f"  {'':<12}{head}")
    print(f"  {'A ' + str(a):<12}" + "".join(f"{str(v):>{w}}" for v in pa))
    print(f"  {'B ' + str(b):<12}" + "".join(f"{str(v):>{w}}" for v in pb))
    print(f"  {'':<12}" + "".join(f"{c:>{w}}" for c in cols))
    if ok:
        result = tuple(out)
        print(f"  result: {result}")
        return result
    print("  result: ValueError - operands could not be broadcast together")
    return None


def check(name: str, got, want, tol: float = 1e-10) -> bool:
    """Compare a computed value against the expected one and report.

    Prints the number that decided the verdict, which is more useful than a
    bare PASS. Handles scalars, arrays and anything np.asarray accepts.
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


# ---------------------------------------------------------------------------
# timing
# ---------------------------------------------------------------------------

def time_call(fn: Callable, *args, repeat: int = 5, **kwargs) -> float:
    """Return the best of `repeat` wall-clock timings of fn(*args, **kwargs).

    Best, not mean: the slow runs are contaminated by whatever else the machine
    was doing, and the fastest run is the closest thing to the cost of the code
    itself. The result is one call, in seconds.
    """
    fn(*args, **kwargs)                        # warm up, then discard
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        best = min(best, time.perf_counter() - t0)
    return best


def compare_timings(results: Iterable[tuple]) -> None:
    """Print a timing table from (label, seconds) pairs, with speed-up factors.

    The slowest entry is the reference, so every factor is at least 1.
    """
    rows = list(results)
    slowest = max(t for _, t in rows)
    print(f"  {'variant':<28}{'seconds':>12}{'speed-up':>12}")
    print("  " + "-" * 52)
    for label, t in rows:
        print(f"  {label:<28}{t:>12.6f}{slowest / t:>11.1f}x")


# ---------------------------------------------------------------------------
# small engineering datasets
# ---------------------------------------------------------------------------

def beam_deflection(x, length: float = 2.0, load: float = 500.0,
                    E: float = 210e9, I: float = 8.0e-6):
    """Downward deflection of a cantilever under a uniformly distributed load.

    Parameters
    ----------
    x       distance from the fixed end, in metres, array-like
    length  span of the cantilever, m       (default 2.0)
    load    distributed load, N/m           (default 500, a light shelf)
    E       Young's modulus, Pa             (default 210 GPa, structural steel)
    I       second moment of area, m^4      (default 8e-6, a small I-section)

    Returns the deflection in MILLIMETRES, positive downwards, because that is
    how a deflection is reported on a drawing. The standard result is

        y(x) = w x^2 (6 L^2 - 4 L x + x^2) / (24 E I)

    Keeping real units and a real formula matters: it forces the plots in
    notebook 03 to carry units, and it gives the broadcasting drills numbers
    with a physical meaning rather than arange(12).
    """
    x = np.asarray(x, dtype=float)
    y = load * x ** 2 * (6 * length ** 2 - 4 * length * x + x ** 2) / (24 * E * I)
    return y * 1000.0


def measured_deflection(x, noise_mm: float = 0.04, seed: int = SEED, **kwargs):
    """beam_deflection with additive Gaussian instrument noise, in millimetres.

    A dial gauge reading to 0.01 mm has a repeatability of roughly 0.04 mm once
    you include how the operator reads it, which is where the default comes
    from.
    """
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    return beam_deflection(x, **kwargs) + rng.normal(0.0, noise_mm, size=x.shape)


def sensor_matrix(n_sensors: int = 6, n_samples: int = 240, seed: int = SEED):
    """A (n_sensors, n_samples) array of strain-gauge readings, in microstrain.

    Each row is one gauge. Every gauge has its own zero offset and its own gain
    error, which is exactly the situation that makes per-row broadcasting
    useful: to remove the offsets you subtract a column vector of shape
    (n_sensors, 1), and to correct the gains you divide by one.

    Returns (readings, offsets, gains) so that a notebook can check an answer
    against the truth.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 4.0, n_samples)                       # seconds
    true_signal = 120.0 * np.sin(2 * np.pi * 0.75 * t) + 40.0 * np.sin(2 * np.pi * 2.5 * t)
    offsets = rng.uniform(-30.0, 30.0, size=(n_sensors, 1))    # microstrain
    gains = rng.uniform(0.9, 1.1, size=(n_sensors, 1))         # dimensionless
    noise = rng.normal(0.0, 3.0, size=(n_sensors, n_samples))
    readings = gains * true_signal[None, :] + offsets + noise
    return readings, offsets, gains


# ---------------------------------------------------------------------------
# plotting conventions
# ---------------------------------------------------------------------------

def engineering_axes(ax, xlabel: str, ylabel: str, title: str | None = None,
                     grid: bool = True, legend: bool = False):
    """Apply this course's figure conventions to one matplotlib Axes.

    Axis labels carry units in brackets, the grid is light rather than loud,
    and the top and right spines are removed so the data is the darkest thing
    on the page. Pass the labels with their units already in them:

        engineering_axes(ax, "position x [m]", "deflection [mm]")

    An engineering figure with an unlabelled axis is not a figure, it is a
    decoration. Every plot you hand in for this course carries units.
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


# ---------------------------------------------------------------------------
# deliberately broken things, for the traceback exercise
# ---------------------------------------------------------------------------

def traceback_demo() -> None:
    """Raise an IndexError from three frames down, on purpose.

    Used by notebook 01. Call it, then read the traceback from the BOTTOM
    upwards: the last line is what went wrong, and the frame directly above it
    is where. The frames above that are how you got there.
    """
    def outer():
        return middle([2.0, 4.0, 6.0])

    def middle(values):
        return inner(values, 5)

    def inner(values, index):
        return values[index]

    outer()


def broken_average(values) -> float:
    """Return the mean of `values`. Contains one real bug. Do not fix it here.

    Notebook 01 asks you to call this, read the traceback, say what is wrong in
    one sentence, and write a corrected version in your own cell. The bug is
    the kind you will write yourself at least once this semester.
    """
    total = 0.0
    for i in range(1, len(values)):
        total += values[i]
    return total / len(values)
