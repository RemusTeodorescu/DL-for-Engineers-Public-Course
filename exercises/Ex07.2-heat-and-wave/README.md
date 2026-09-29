# Ex_07.2 — Fundamental PDEs: Heat, Wave and Helmholtz

**Paired with L7.2 · Fundamental PDEs · Part 2**

Time enters. A steady problem has one field, two coordinates and no history. Here the network takes $(x, y, t)$ and the question becomes **how many
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

## The two problems, and the equations they solve

**The heat equation, θ_t = α ∇²θ — the die; parabolic, one initial condition.** A 10 × 10 mm silicon die after
a power pulse, edges clamped to the package. The initial field carries two
modes decaying at 17.4 and 43.4 s⁻¹, so the sharp feature dies 2.5× faster and
the field **changes shape** rather than merely shrinking. A model that matches
the late field can still be badly wrong early.

**The wave equation, u_tt = c² ∇²u — the panel; hyperbolic, two initial conditions.** A 40 × 40 cm tensioned
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

## Supplement to PDE Recap (Optional)

```
Ex07.2_supplement_helmholtz.ipynb         the exercise: one TODO, the loss
Ex07.2_supplement_helmholtz_light.ipynb   every cell written out
```

A standalone notebook beside the die and the panel, not part of the report
(C11 allows one optional supplement per set). It takes L7.2's Helmholtz
equation to the stator slot of an electrical machine: hot copper turns Poisson into Helmholtz
(q = c0 + c1 θ), the slot's first eigenvalue λ1 sets how far the temperature
is pushed up and where thermal runaway would be (158 A), a hard-enforced
network finds λ1 with and without the unit-norm safeguard against u ≡ 0, and
three design levers move the feedback ratio. Two questions, tagged L7.2 Q3 and
Q5. About a minute on a CPU. It stands on its own: `slot_problem.py` and
`Ex07.2_slot.png` are this folder's copies of Ex_07.1's `problem.py` and slot
figure, so nothing is needed from Ex_07.1. MP7.2C grows out of it. Built by
`tools/exercises/ex072/build_ex072_supplement.py`.

## Files

| | |
|---|---|
| `course_core.py` | shared by the whole course |
| `pinn_core.py` | the PDE machinery |
| `problem.py` | both problems — exact solutions, initial fields, plots |

The first two are generated: edit `tools/pinn/*.py` and run
`python3 tools/pinn/sync_cores.py`. **On Colab nothing needs uploading** — each notebook fetches them.

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

## Results between notebooks on Colab

Later notebooks read `.npz` / `.pkl` files that earlier ones write into
`Ex07.2_outputs/`. On Google Colab every notebook runs on its own temporary machine,
so those files would not survive from one notebook to the next. Notebooks
01 to 05 therefore start with an `outputs-cell` that calls `keep_outputs()` from
`course_core.py`: on Colab it mounts the student's Google Drive and moves the results
folder to `MyDrive/DL4Eng/Ex07.2_outputs`. If the student declines the Drive request or
has no Google account, `saved()` downloads each result file when it is written
and `needed()` asks for the files to be uploaded before they are read. Locally
the cell does nothing. The report notebook writes its `.md` and `.pdf` into the
same folder.

## Mini project proposal

The set ends with three mini projects (notebook 05, the last section). Each student chooses one
mini project from the Part 2 sets and solves it individually in one month. The
ground truth is given, built by `tools/miniprojects/ex072_truth.py` under policy C10
(`COURSE_POLICIES.md`), with a worked example of each.

| | the problem | the deep learning | the ground truth given | required |
|---|---|---|---|---|
| **MP7.2A · A die with a moving hot spot** | a 10 × 10 mm die; three 2 × 2 mm blocks of 8 W switching on in turn; silicon's k(T) | a PINN for θ(x, y, t) with the power map as input, the initial condition built in, points that follow the power | finite volumes, backward Euler, 0.05 mm and 0.5 ms; worked example: peak 33.79 K at 180 ms, k(T) adds 1.66 K | peak within 1 K and 2 ms; field within 2 K; 200 ms in under 1 s |
| **MP7.2B · Where was the panel struck?** | a strike nobody saw; 3 to 6 sensors record the 0.40 m panel's displacement for 20 ms | an inverse PINN: u(x, y, t) and the initial velocity, fitted to the wave equation and the sensors | the exact modal sum, 80 × 80 modes; three cases; worked example: peak 0.23 mm, a mirror strike 2e-19 m apart on the diagonal | strike within 10 mm, velocity within 10 %, motion within 5 % of peak |
| **MP7.2C · The eigenvalues of a real slot** | a slot tapered from 8 to 12 mm, 20 mm deep; the first three Helmholtz eigenpairs | an eigenvalue PINN with a rebuilt mask, a norm term, a trainable λ and orthogonal higher modes | finite differences to 0.03125 mm, extrapolated; worked example: λ₁ 0.1226 per mm², 0.7 % below the rectangle; runaway 157.9 A | λ₁ within 0.3 %, λ₂ and λ₃ within 1 %, runaway within 0.5 A |
