# Ex01 — Python Fundamentals

**Paired with lecture block L1 · Part 1**

Self-check notebook, broadcasting drills, first engineering plots. No deep
learning yet.

## Goals

By the end of this exercise set you can

1. write and debug Python — types, containers, functions — and read a traceback
   well enough to act on it rather than guess;
2. use NumPy arrays deliberately: dtype, shape, slicing, broadcasting, and
   replace a loop with a vectorised expression;
3. produce an engineering figure that could go into a report — labelled axes,
   units, a legend that means something;
4. recognise the arithmetic traps that arrive with you from MATLAB, in
   particular that `**` is the power operator and `^` is not;
5. check a computed result against a hand calculation instead of trusting it.

## State

**Written.** One module and four notebooks. Every notebook has been executed
end to end with reference answers substituted for the TODO cells, so the check
cells are known to pass once a correct answer is supplied.

## Notebooks

Run in order; later notebooks assume the vocabulary of earlier ones.

```
Ex01_00_environment_check.ipynb      read-only. Versions, seeds, first figure.
Ex01_01_python_basics.ipynb          5 TODOs. Types, containers, control flow,
                                     functions, imports, reading a traceback.
Ex01_02_numpy_and_broadcasting.ipynb 7 TODOs. Arrays vs lists, dtype and shape,
                                     indexing, BROADCASTING, vectorisation with
                                     a timed comparison.
Ex01_03_plotting.ipynb               2 TODOs. Figure and axes, units on every
                                     label, log scales, the residual plot.
```

**Notebook 02 is the one to protect if a student is short of time.** Every shape
bug in the Part 2 exercises is a broadcasting misunderstanding, and the specific
failure it drills — `(N,)` against `(N, 1)` producing an `(N, N)` matrix and a
plausible wrong number, with nothing raised — is the one that costs an
afternoon in week seven.

## Module

`Ex_1_core.py` carries the long **HOW TO RUN Ex_1** docstring, in the same shape
as the Part 2 core libraries: what you need, the file list, the note on module
naming, Colab and local instructions, how the exercise is structured, expected
runtime, and a troubleshooting section that covers the traceback students will
actually see.

Its contents: `banner`, `check_environment`, `set_seed`, `describe`,
`broadcast_table` (the right-aligned alignment table that is the teaching device
of notebook 02), `can_broadcast`, `check` and `check_shape`, `time_call` and
`compare_timings`, the cantilever dataset (`beam_deflection`,
`measured_deflection`), the strain-gauge dataset (`sensor_matrix`),
`engineering_axes` for the figure conventions, and two deliberately broken
things — `traceback_demo` and `broken_average` — for the debugging section.

## The running example

A 2 m steel cantilever under a uniform distributed load, in real units, with a
dial-gauge noise level of 0.04 mm. It appears in all four notebooks: as an array
to describe, as the source of the `(N,)` versus `(N, 1)` trap, as a load sweep
to plot, and as a data-and-residual pair. Using one physical object throughout
means the shapes always mean something.

## Conventions

Same as every Part 2 exercise set:

- Self-contained folder. Requires `torch`, `numpy`, `matplotlib` — all
  preinstalled on Google Colab. No GPU needed. Nothing in Ex01 actually imports
  `torch`; notebook 00 checks for it so that a missing install is found in week
  one rather than in the middle of Ex02.
- Notebook `00` is a read-only environment check. Run it first.
- Modules (`Ex_1_core.py`) are complete and are **not** to be rewritten by
  students. The work is in `# TODO:` cells in the numbered notebooks, each
  followed by `raise NotImplementedError`.
- Module names use underscores because a Python module name cannot contain a
  dot: `Ex_1_core.py`, imported as `Ex_1_core`.
- Every cell that produces output is followed by a **"What you should see"**
  note, so a student working alone can tell a correct run from a broken one.

## Deliberately unlike Part 2

The Part 2 exercises are terse: a paragraph of physics, then a TODO. Ex01 and
Ex02 are not. Every TODO here is preceded by several paragraphs explaining what
is being asked, why it is being asked, and what a correct answer looks like,
because L1 and L2 are recorded rather than live and a student who missed them
has no room to ask. The prose thins out from Ex03 and is gone by Ex07.

## Expected runtime

CPU only, under five minutes per notebook including the timed comparison in
notebook 02. The difficulty is conceptual, not computational.
