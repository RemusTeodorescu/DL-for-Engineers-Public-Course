# Ex_07.2 (Optional) — Benchmark PDEs: Parabolic, Elliptic and Hyperbolic

**Paired with L7.2 · Fundamental PDEs · Part 2**

Three problems, one of each type of second-order equation, each in a notebook of
its own. Every notebook tells the same story (course policy C13): the problem,
its data, its physics, the exact solution and finite differences on a mesh
chosen for an agreed accuracy, one physics-informed network, the comparison on
accuracy against the exact solution and on computing time against finite
differences, and what the comparison says. Then a short report, and a mini
project.

The three are independent: run them in any order, alone. None needs Ex_07.1.

## The notebooks

```
Ex07.2_1_die_pulse.ipynb        parabolic   the silicon die after a power pulse
Ex07.2_2_slot_runaway.ipynb     elliptic    the stator slot and its thermal runaway
Ex07.2_3_panel_strike.ipynb     hyperbolic  the wave of a struck panel
```

Each comes in two forms: the exercise, with two TODO cells (the answer in the
comment directly above the line to write), and `_light`, written out.

| | equation | conditions | exact solution | agreed accuracy, and the mesh that meets it | the network |
|---|---|---|---|---|---|
| **1 · die** | heat, $\theta_t = \alpha\nabla^2\theta$ | edges at 65 °C, **one** initial field | two decaying modes | 0.02 K: 41 × 41 nodes, 200 Crank-Nicolson steps | 4 × 32; start and edges built in |
| **2 · slot** | $k\nabla^2\theta + c_0 + c_1\theta = 0$, Helmholtz once hot copper feeds back | walls at 90 °C | a series of the slot's modes | 0.1 K at 10-100 A: 41 × 81 nodes | 4 × 32, every current at once; walls built in |
| **3 · panel** | wave, $u_{tt} = c^2\nabla^2 u$ | edges clamped, **two** initial conditions | the first mode | 1 % of the 0.56 mm peak: 41 × 41 nodes, leapfrog | 5 × 48; edges and flat start built in, the velocity in the loss |

### What the set measures (CPU, seed 88)

| | finite differences: error, one run | PINN: error, training |
|---|---|---|
| 1 · die | 1.58e-02 K, about 30 ms | 1.65e-02 K, about 3 min; then about 300 ms to evaluate the window |
| 2 · slot | 4.08e-02 K at worst, about 8 ms per current | within 0.1 K to 80 A, 0.28 K at 100 A; about 2 ms per current after about 3 min |
| 3 · panel | 0.30 % of the peak, about 10 ms | 0.49 %, about 14 min; then about 400 ms to evaluate the window |

The notebooks print the exact figures for the machine they run on.

## Files

| | |
|---|---|
| `course_core.py` | shared by the whole course |
| `pinn_core.py` | the PDE machinery |
| `problem.py` | the die and the panel - exact solutions, initial fields |
| `slot_problem.py` | the stator slot - its heat, the exact series, finite differences; a copy of Ex_07.1's `problem.py`, so notebook 2 stands alone |
| `Ex07.2_slot.png` | the slot figure notebook 2 shows |

The first two are generated: edit `tools/pinn/*.py` and run
`python3 tools/pinn/sync_cores.py`. **The notebooks are generated too**, both
forms of all three, by `tools/exercises/ex072/build_ex072.py`: edit the builder
and rerun it rather than editing a notebook. **On Colab nothing needs uploading**
- each notebook fetches its files.

## Expected runtime

CPU only. The die and the slot take about five minutes each, most of it
training; the panel about fifteen - a wave over three periods is the hardest of
the three for a network.

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

Every light notebook has been run end to end on a local CPU and the exercise
forms stop at their first TODO. Still to do, as for every set (C8): a run from
a fresh Colab runtime, and a review by someone other than the author.

## Mini project proposal

The set ends with three mini projects (one at the end of each notebook, for its own problem). Each student chooses one
mini project from the Part 2 sets and solves it individually in one month. The
ground truth is given, built by `tools/miniprojects/ex072_truth.py` under policy C10
(`COURSE_POLICIES.md`), with a worked example of each.

| | the problem | the deep learning | the ground truth given | required |
|---|---|---|---|---|
| **MP7.2A · A die with a moving hot spot** | a 10 × 10 mm die; three 2 × 2 mm blocks of 8 W switching on in turn; silicon's k(T) | a PINN for θ(x, y, t) with the power map as input, the initial condition built in, points that follow the power | finite volumes, backward Euler, 0.05 mm and 0.5 ms; worked example: peak 33.79 K at 180 ms, k(T) adds 1.66 K | peak within 1 K and 2 ms; field within 2 K; 200 ms in under 1 s |
| **MP7.2B · The eigenvalues of a real slot** | a slot tapered from 8 to 12 mm, 20 mm deep; the first three Helmholtz eigenpairs | an eigenvalue PINN with a rebuilt mask, a norm term, a trainable λ and orthogonal higher modes | finite differences to 0.03125 mm, extrapolated; worked example: λ₁ 0.1226 per mm², 0.7 % below the rectangle; runaway 157.9 A | λ₁ within 0.3 %, λ₂ and λ₃ within 1 %, runaway within 0.5 A |
| **MP7.2C · Where was the panel struck?** | a strike nobody saw; 3 to 6 sensors record the 0.40 m panel's displacement for 20 ms | an inverse PINN: u(x, y, t) and the initial velocity, fitted to the wave equation and the sensors | the exact modal sum, 80 × 80 modes; three cases; worked example: peak 0.23 mm, a mirror strike 2e-19 m apart on the diagonal | strike within 10 mm, velocity within 10 %, motion within 5 % of peak |
