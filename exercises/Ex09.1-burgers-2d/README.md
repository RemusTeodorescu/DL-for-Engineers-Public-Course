# Ex_09.1 — Laminar Flow: 2-D coupled Burgers

**Paired with L9.1 · Laminar Flow · Part 2**

The first nonlinear, **vector-valued** problem of the course, and the last stop
before Navier–Stokes. Two coupled equations, two outputs from one network, and
a single dimensionless number — the Reynolds number — that decides whether the
method works at all.

Everything before this was solved once and reported. This set is built the
other way round: an **interactive control panel** exposes the collocation
count, the Reynolds number, the architecture and the optimiser budget as
sliders, and a report is assembled from every run you record. The question is
not "can you solve it" but **"where does it stop working, and how would you
know?"**

## Goals

By the end you can

1. write the residual of a **coupled, nonlinear system** — two equations
   sharing two unknowns — from a single gradient call per component;
2. build a network with several outputs and say what a shared trunk assumes
   about the physics, then test that assumption against two separate networks
   at matched parameter count;
3. run a parameter study as a study rather than as a sequence of edits, and
   report each run with its loss curve and its error against time;
4. connect a dimensionless group to a length scale the network must resolve —
   here, a front of e-folding width `8 nu` — and predict from it where the
   collocation set will become too coarse;
5. **recognise the characteristic PINN failure at high Re**: not a divergence,
   but a smooth, plausible and wrong field whose loss looks perfectly healthy.

## The problem

```
u_t + u u_x + v u_y = nu (u_xx + u_yy)
v_t + u v_x + v v_y = nu (v_xx + v_yy)      on the unit square, 0 <= t <= 1
```

The vector-valued, nonlinear, coupled stepping stone to Navier–Stokes: it keeps
convection and the coupling between the components, and drops the pressure and
the incompressibility constraint. Everything that makes a flow solver hard
except the constraint that makes it slow.

The exact solution is known, so the notebooks measure **true error** rather
than estimating it:

```
u = 3/4 - 1/(4 (1 + exp((-4x + 4y - t)/(32 nu))))
v = 3/4 + 1/(4 (1 + exp((-4x + 4y - t)/(32 nu))))
```

Both components are one tanh-like front. Rewriting the exponent as
`((y - x) - t/4) / (8 nu)` says everything about the difficulty: the front lies
on **y = x + t/4**, it translates across the square as time runs, and its
e-folding width in `y - x` is **8 nu**. With U = L = 1 the Reynolds number is
simply 1/nu, so raising Re thins the front in exact proportion — a width of
0.4 at Re = 20, against a unit side, and 0.016 at Re = 500.

Boundary and initial data are both taken from the exact solution, which makes
this a **pure verification problem**: every error reported is the model's,
never the data's. And `u + v = 3/2` identically, for all x, y and t — a free
diagnostic nothing in the loss asks for.

## The notebooks

Run them in order; later ones load results the earlier ones saved into
`Ex09.1_outputs/`.

```
Ex09.1_00_setup_check.ipynb      the environment, the samplers, the reference solution
Ex09.1_01_burgers2d.ipynb        write the residual and the loss — the only TODOs
Ex09.1_02_control_panel.ipynb    the interactive parameter study
Ex09.1_03_reynolds_sweep.ipynb   find the Reynolds limit yourself
Ex09.1_04_report.ipynb           assemble the report for submission
```

**Notebooks 02 and 03 ask you to paste your `residual_fn` and
`loss_fn_factory` from notebook 01.** They are the physics, and they stay
yours; everything else in the set is scaffolding that calls them.

## Files

Every Part 2 exercise has the same three modules beside it. Only the third
differs between sets.

| | |
|---|---|
| `course_core.py` | shared by the whole course — `set_seed`, `MLP`, `to_tensor`, `check` |
| `pinn_core.py` | the PDE machinery — `grad`, `d2`, samplers, `train_two_stage` |
| `problem.py` | **this** problem — exact solution, `FlowPINN`, the control panel, the report |

The first two are generated. Edit `tools/pinn/*.py` and run
`python3 tools/pinn/sync_cores.py`; never edit a copy.

**Colab needs the whole folder**, not just a notebook — the imports expect the
three modules beside them. The control panel also needs `ipywidgets`, which is
preinstalled on Colab; locally, `pip install ipywidgets`.

## Expected runtime

CPU only; no GPU is needed anywhere in Part 2. A single run at the default
configuration is a few minutes. Notebook 02 launches one run per press of the
button and notebook 03 launches seven in a row, so **time one run before you
commit to a sweep** and budget accordingly.

Two things make a run longer than the Part 1 figures would suggest. An L-BFGS
*step* in `train_two_stage` is an outer step of up to twenty inner iterations
with a strong-Wolfe line search, so `lbfgs_epochs = 200` is a substantial
stage, not a footnote. And the residual here is second order in three inputs
for **two** coupled fields, which is the most autograd work of any set in the
course so far.

If a cell seems stuck, it probably is not: the L-BFGS stage prints only every
few dozen steps, so a long silence after the Adam output is normal.

## What to hand in

- the residual and the loss you wrote, with the error against the exact
  solution, and the loss curve for every run you quote
- the control-panel study, with the parameter you found mattered most
- the Reynolds number at which your solution stops being trustworthy, and the
  evidence you used to decide that
- the five questions at the end of the generated report, answered

## Things that go wrong, and what they mean

**The solution looks right and the error is large.** Compare against the exact
solution the module provides, not against your intuition about what a flow
should look like. Plausible-looking flow fields are easy.

**Raising Reynolds makes training fail rather than gradually degrade.** That is
the honest finding, and the sweep exists to let you report where it happens
rather than quoting a number from a paper.

**The loss falls as far as a low-Re run but the error is far worse.** The model
satisfies the equations at the points you gave it and does something else
between them. Compare the front width `8 nu` with the spacing of your
collocation set before blaming the optimiser.

**The control panel does not render.** `ipywidgets` is missing, or the notebook
needs re-running after installing it. It is a display problem, not a physics
one.

**`NotImplementedError`.** Expected. You have reached a TODO cell, or a paste
cell in notebook 02 or 03 that is waiting for your notebook 01 functions.

**The loss becomes `nan`.** Almost always a residual that divides by zero or
takes a log or square root of a negative number. Print the residual on a
handful of points before training.

## How this is meant to be used

The modules are complete and working — you are not asked to rewrite them. Your
work is in the cells marked `# TODO`, which are the residual and the loss.
They are short by design, so your time goes on the parts that carry the ideas
rather than on tensor plumbing.

You *are* expected to read `problem.py`. The laboratory around the physics —
the sampling, the runner, the panel, the report — is the reference
implementation your work is judged against, and the module docstring states
the problem more precisely than this file does.

## Reference texts

Liu, G.R., *PINN with Python: An Introduction* (2025).
Raissi, Perdikaris & Karniadakis, *Physics-informed neural networks*,
J. Comput. Phys. **378** (2019) 686–707.

These are the works to read for the theory. **The code, the problem and the
exposition in this exercise set are original to this course** — written from the
2019 paper and the PyTorch documentation, and not derived from any publisher's
code listings. Where a symbol matches a textbook's, it is because both follow
the standard notation of the field.

## Before this is assigned

This set was migrated onto the shared `course_core.py` / `pinn_core.py` from
its own private core, in an environment where PyTorch could not be installed.
The physics — the exact solution, the Reynolds relation, the geometry and the
time window — is unchanged from the working version. **Nothing that requires
torch has been executed.** Run notebooks 00 and 01 end to end before this goes
to students, and in particular confirm that the reference-solution residual
check in notebook 00 still passes.
