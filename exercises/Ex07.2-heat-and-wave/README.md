# Ex_07.2 — Fundamental PDEs: a die and a panel

**Paired with L7.2 · Fundamental PDEs · Part 2**

Time enters. Everything in Ex_07.1 was steady — one field, two coordinates, no
history. Here the network takes $(x, y, t)$ and the question becomes **how many
conditions a problem needs**, and what happens when it does not get them.

## Goals

By the end you can

1. write first- and second-order-in-time residuals and say which column of the
   gradient is which;
2. assemble a three- or four-term loss with every term non-dimensionalised;
3. enforce an initial condition exactly, by construction, and state precisely
   what that costs the network;
4. explain why a diffusion error shrinks with time and a wave error grows;
5. **recognise an under-determined problem from its symptoms**, and give the
   counting rule that would have prevented it.

## The two problems

**The die — parabolic, one initial condition.** A 10 × 10 mm silicon die after
a power pulse, edges clamped to the package. The initial field carries two
modes decaying at 17.4 and 43.4 s⁻¹, so the sharp feature dies 2.5× faster and
the field **changes shape** rather than merely shrinking. A model that matches
the late field can still be badly wrong early.

**The panel — hyperbolic, two initial conditions.** A 40 × 40 cm tensioned
panel, 141 Hz fundamental, struck at 0.5 m/s to a peak deflection of 0.56 mm.
It starts **flat and moving**, which is what makes the second condition
load-bearing.

## The notebooks

```
Ex07.2_00_environment_check.ipynb    space-time sampling, derivatives in t
Ex07.2_01_die_soft_ic.ipynb          three soft terms — and where the error goes
Ex07.2_02_die_hard_ic.ipynb          θ₀ + t·D·N, and what N pays for it
Ex07.2_03_panel_both_ics.ipynb       the panel solved properly, over three periods
Ex07.2_04_missing_condition.ipynb    the panel that never moved
Ex07.2_05_compare_and_report.ipynb   the report
```

**Notebook 04 is the one that matters.** Solve the panel correctly in every
respect except the velocity condition, and `u ≡ 0` satisfies the PDE, both
boundaries and the initial displacement — exactly. The loss converges *better*
than the correct run. Three diagnostics pass magnificently and the fourth, the
one you left out of the loss, is the only one that knows.

> A converged residual tells you the network solves the problem you posed. It
> says nothing about whether you posed the right problem.

That is the habit L8 onward depends on, because from there you rarely have an
exact solution to check against.

## Files

| | |
|---|---|
| `course_core.py` | shared by the whole course |
| `pinn_core.py` | the PDE machinery |
| `problem.py` | both problems — exact solutions, initial fields, plots |

The first two are generated: edit `tools/pinn/*.py` and run
`python3 tools/pinn/sync_cores.py`. **Colab needs the whole folder.**

## Expected runtime

CPU only. The panel is the expensive one — a space–time slab, 6000 collocation
points and a fifth hidden layer — so budget ten minutes for notebooks 03 and 04
each. The die notebooks are three to five minutes.

## Reference texts

Liu, G.R., *PINN with Python: An Introduction* (2025).
Raissi, Perdikaris & Karniadakis, *Physics-informed neural networks*,
J. Comput. Phys. **378** (2019) 686–707.

These are the works to read for the theory. **The code, the problems and the
exposition in this exercise set are original to this course** — written from the
2019 paper and the PyTorch documentation, and not derived from any publisher's
code listings. Where a symbol matches a textbook's, it is because both follow
the standard notation of the field.

## Before this is assigned

The NumPy half is verified: both exact solutions satisfy their PDEs to ~5e-6
relative under central differences, the die's edges are exactly zero, the
panel's initial displacement is exactly zero and its initial velocity is
0.4999 m/s against a nominal 0.5. **Nothing requiring torch has been executed.**
Run notebook 00 end to end before this goes to students.
