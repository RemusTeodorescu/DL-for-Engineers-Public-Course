# Ex_07.1 — Fundamentals of PINNs: a stator slot

**Paired with L7.1 · Fundamentals of PINNs · Part 2**

The first exercise of Part 2, and the one that establishes the machinery the
rest reuse: a residual built by automatic differentiation, a composite loss, and
the Adam-then-L-BFGS schedule from Ex_06.

**On the relationship with Ex_03.** Students have already seen a differential
equation in a loss function — Ex_03 put a damped oscillator there and measured
what it bought. That was deliberate, and it means the *idea* is not new here.
What is new is that they build it. In Ex_03 the residual is written for them in
`Ex_3_core.py` and the exercise is to turn the weights; here the first TODO is
to write the residual themselves, on a PDE with two independent variables, and
nothing is supplied but the domain and the source term.

The two also differ in what a condition is. Ex_03 has an initial condition
imposed as one more loss term. Here the boundary condition is imposed twice —
once softly and once exactly by construction — and comparing the two is the
whole point of the exercise.

The question it answers is narrow and it matters everywhere afterwards:
**when you know something about the solution, where should you put it?**

## Goals

By the end you can

1. write a PDE residual with automatic differentiation and check it against an
   analytic derivative;
2. assemble a composite loss whose terms have been non-dimensionalised, and say
   what the weight between them means;
3. enforce a Dirichlet condition **softly**, as a penalty, and measure what it
   costs;
4. enforce the same condition **exactly**, by construction, and say what that
   costs instead;
5. score a solution in two currencies — a dimensionless norm and the worst
   error in kelvin — and explain why one alone is not enough.

## The problem

A slot in the stator of an electrical machine, packed with an impregnated
copper winding. Current heats the bundle; the slot walls sit at the local iron
temperature. In steady state,

```
k_eff ∇²θ + q(x, y) = 0     on a 10 × 20 mm slot
θ = 0                        on all four walls
```

The bundle is a poor conductor — enamel, epoxy and trapped air give an
effective conductivity of about 0.70 W/m·K, three orders below solid copper.
That is why a realistic current density produces a temperature rise worth
computing, and it is why slot hot spots limit a machine's rating.

The source is **manufactured**, so the exact solution is known and the
notebooks can measure true error rather than estimate it:

```
θ(x, y) = ΔT (1 − ξ²)(1 − η²)(1 + 0.3 ξ)     ξ = x/a, η = y/b
```

Polynomial rather than trigonometric, so no single Fourier mode gets a network
most of the way. Skewed toward one side, so there is no symmetry to exploit.
And the skew is capped at 1/3, which is what keeps the implied source
non-negative everywhere — a manufactured problem that quietly requires heat to
be removed from part of the domain is a mathematics exercise in fancy dress.

The numbers it implies are checked aloud in notebook 00: a peak of
2.13 MW/m³, equivalent to **10.4 A/mm² in the copper**, where machines run
between 5 and 20.

## The notebooks

Run in order; later notebooks load results saved by earlier ones.

```
Ex07.1_00_environment_check.ipynb    tools, autograd, samplers, the problem itself
Ex07.1_01_slot_soft_bc.ipynb         the walls as a penalty — and the weight sweep
Ex07.1_02_slot_hard_bc.ipynb         the walls built into the function space
Ex07.1_03_compare_and_report.ipynb   both under a sparse sample, and the report
```

## Files

Every Part 2 exercise has the same three modules beside it. Only the third
differs between sets.

| | |
|---|---|
| `course_core.py` | shared by the whole course — `set_seed`, `MLP`, `to_tensor`, `check` |
| `pinn_core.py` | the PDE machinery — `grad`, `d2`, samplers, `train_two_stage` |
| `problem.py` | **this** problem — geometry, exact solution, source, plots |

The first two are generated. Edit `tools/pinn/*.py` and run
`python3 tools/pinn/sync_cores.py`; never edit a copy.

**Colab needs the whole folder**, not just a notebook — the imports expect the
three modules beside them.

## Expected runtime

CPU only. Notebooks 00 and 02 are a couple of minutes each; notebook 01's
weight sweep is six training runs and notebook 03's scarcity sweep is ten, so
budget five to ten minutes for each of those. No GPU is required anywhere in
Part 2.

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

Written in an environment where PyTorch could not be installed. The NumPy half
is verified: the manufactured source and exact solution agree to 7.7e-13
relative, the boundary values are exactly zero on all four walls, the hard-BC
factor is exact, and the samplers place points where they claim. **Nothing that
requires torch has been executed.** Run notebook 00 end to end before this goes
to students. See `docs/REPO_NOTES_PART1.md` §9.
