# Ex_09.1 — A Front Carried by a Flow

**Paired with L9.1 · Laminar Flow · Part 2**

The coupled Burgers equations in **one notebook** (course policies C11 and
C13): the momentum equations of a flow with the pressure and the
incompressibility constraint left out. Two streams of the same fluid meet:
below a slanted line the fluid moves at (0.50, 1.00), towards the line; above
it at (0.75, 0.75), along it. Where they meet the velocity changes in a thin
band, a front, whose width is set by the Reynolds number, and the front creeps
upwards. Students compute the answer
two ways — finite differences on a grid, and a physics-informed network with
two outputs — score both against an exact solution, and compare them on
accuracy and on time. Then they raise the Reynolds number from 20 to 500 and do
it again.

## The pipe version, for inspection

`Ex09.1_pipe_startup_new.ipynb` (with its light form) sits beside the Burgers
notebook until the lecturer chooses one. A cold plate, 250 × 130 × 12 mm, has
a 6 mm copper tube in a serpentine - four legs of 230 mm, three U-bends of
12 mm radius - drawn in 3-D in section 2. The notebook solves the coolant
(50/50 water–glycol at 40 °C, 0.3 L/min, Re = 452) in the fully developed
part of one leg when the pump starts. Section 3 reduces Navier–Stokes to
$
ho u_t = G + \mu(u_{rr} + u_r/r)$ step by step (round and straight, far
from the bends, the volume kept), lists the conditions (no slip, symmetry on
the axis, at rest, the pump's push), says why this case can be solved by
hand, gives Hagen–Poiseuille's parabola and Darcy's law ($K = R^2/8$, exact
because the inertia is zero), and the start-up's exact series, of which the
parabola is the limit. Finite differences meet 1e-03 with 21 nodes × 80
steps in 1 ms; the PINN (wall and start built in, $s = (r/R)^2$ as input) is
within 6e-04 after about 40 s of training, 0.4 ms online. Section 5 draws the
velocity across the section in colour, exact, grid and network, and their
errors. Section 6 finds the coolant's viscosity from a flow meter's first
second: Darcy's law on the last reading +15.5 % (the flow has not settled);
least squares on the exact series, least squares with the grid as the model
(5 runs, about 13 ms) and the network (about 40 s of training) all within
2 % - an inverse problem is not a network's privilege, and for this tube the
grid in an optimiser is the cheapest. Section 5 also times both methods **on one device**
(the GPU on Colab; the grid in PyTorch, `pp.fdm_torch`): on a laptop GPU the
grid takes 9.6 ms, the trained network answers in 0.31 ms - 31 times faster -
but its 97 s of training are repaid only after about 10 500 answers; the
inverse grid fit is 555 times faster than the network's. A scaling cell
measures the same equation in a box in 1, 2 and 3 dimensions and
extrapolates: a 3-D grid of 161 nodes per direction about half an hour per
answer, a three-parameter sweep a thousand runs - the network's case is many
queries and many dimensions, not one small solve.

Its mini projects: **MP9.1A, one bend of the serpentine, in 3-D** - a U-bend
between a 20 mm and a 110 mm leg; the students write a marching
(parabolised) solver, with guidelines in the brief, prove it against two
supplied limits (the notebook's parabola, and the fully developed bend from
`tools/miniprojects/dean.py`: fastest liquid 0.74 R towards the outer wall,
friction twice the leg's) and then train a 3-D PINN against it, reporting the
speed maps through and after the bend, the vortices' decay and the bend's
loss coefficient; **MP9.1B, the pump's
speed reference** - the least-pumping flow over 30 minutes for a staircase of
module power, keeping the module at or below 80 °C: the flow must rise eight
minutes before a step, and a controller that sees only the present power
reaches 86.47 °C. Both truths are in `tools/miniprojects/pipe_truth.py`. **All
sizes and the coolant's properties are typical values typed from memory, and
so are the heat transfer and friction correlations used.**

## The problem

```
u_t + u u_x + v u_y = ν (u_xx + u_yy)
v_t + u v_x + v v_y = ν (v_xx + v_yy)        on the unit square, 0 ≤ t ≤ 1
u, v given on the four edges and at t = 0     (a weighted loss term)
```

Lengths are in units of the side L and speeds in units of a reference speed U,
so the one number left is Re = UL/ν = 1/ν. The exact solution is a single
front along y = x + t/4, of width 8ν = 8/Re: 0.40 at Re = 20 and 0.016 at
Re = 500. It also satisfies u + v = 3/2 everywhere, a check that needs no
exact solution. It is a model problem, not a device: the units stay scaled.

## The notebook

```
Ex09.1_pinn_burgers_front.ipynb         the exercise: two TODO cells, the answer in the comment above each
Ex09.1_pinn_burgers_front_light.ipynb   the same notebook with every cell written out
```

| section | what the student does |
|---|---|
| 1 – 3 | the problem, its data, its physics: convection, viscosity, the coupling, the front's width 8/Re |
| 4 | the exact solution as the reference; finite differences (central, Heun in time) on finer grids, for an agreed 1e-03 |
| 5 | the PINN: one network, two outputs; the two residuals (TODO 1), the loss (TODO 2), training; the two methods compared |
| 6 | the Reynolds number raised to 500: both methods again, nothing changed but ν |
| 7 | what the notebook says |
| 8 | the report and its PDF |
| 9 | mini project proposal |

Speeds are printed with at most two decimals and errors as powers of ten (C12).
The control panel, the shared-against-separate network comparison and the
seven-point Reynolds sweep of the earlier five-notebook version are gone.

### What it measures (CPU, seed 88)

| Re | method | worst error | one solution | training |
|---|---|---|---|---|
| 20 | finite differences, 6 × 6 nodes (the coarsest for 1e-03) | 5.46 × 10⁻⁴ | milliseconds | — |
| 20 | PINN, 4 × 32, two outputs, 4000 points | 6.72 × 10⁻⁴ | 0.02 s | 40 s |
| 500 | finite differences, 321 × 321 nodes (the coarsest for 1e-03) | 3.08 × 10⁻⁴ | about 10 s | — |
| 500 | PINN, the same network and the same 4000 points | 9.68 × 10⁻⁴ | 0.02 s | about 1 min |

The network keeps u + v within 7 × 10⁻⁴ of 3/2 at Re = 20 and 9 × 10⁻⁴ at
Re = 500 without being told to.

**The network shows no Reynolds ceiling on this problem**, and the notebook
says why: the exact solution is a tanh of a straight line in (x, y, t), which
one tanh neuron represents exactly, so the network does not have to resolve
the front with points. Outside the notebook the same network and points gave
2.4 × 10⁻³ at Re = 2000, where central differences need more than 321 nodes a
side. The grid's cost is what grows with the Reynolds number here: about five
nodes across the front, 6 a side at Re = 20 and 321 at Re = 500. The earlier
notebooks told students to expect the network to fail quietly at high Re; that
was never measured, and on this problem it does not happen. A flow whose thin
feature is not a single tanh is Ex_09.2 and mini project MP9.1A.

## Files

| | |
|---|---|
| `course_core.py` | shared by the whole course — `set_seed`, `MLP`, `to_tensor`, `check` |
| `pinn_core.py` | the PDE machinery — `grad`, `d2`, `train_two_stage` |
| `problem.py` | **this** problem — the data, the exact solution, the collocation and data points, the points a network is scored on |

The first two are generated. Edit `tools/pinn/*.py` and run
`python3 tools/pinn/sync_cores.py`; never edit a copy.

**The notebook is generated too**, both forms from one source,
`tools/exercises/ex091/build_ex091.py`, with the cells every one-notebook set
shares in `tools/exercises/part2_notebook.py`. Edit the builder and rerun it
rather than editing a notebook, or the two forms drift apart.

**On Colab nothing needs uploading**: the first code cell fetches the three
modules from the public course repository, afresh on every run.

## Expected runtime

CPU only; a GPU is slower on problems this small. About two minutes in all on
a laptop, most of it the two trainings of sections 5 and 6 and the 321 × 321
grid of section 6.

## Reference texts

Liu, G.R., *PINN with Python: An Introduction* (2025).
Raissi, Perdikaris & Karniadakis, *Physics-informed neural networks*,
J. Comput. Phys. **378** (2019) 686–707.

These are the works to read for the theory. **The code, the problem and the
exposition in this exercise set are original to this course.**

## Before this is assigned

The light version has been executed end to end on a local CPU; the exercise
version stops at its first TODO. Still to do, as for every set (C8): a run from
a fresh Colab runtime, and a review by someone other than the author.

## Mini project proposal

The set ends with two mini projects (section 9). Each student chooses one
mini project from the Part 2 sets and solves it individually in one month. The
ground truth is given, built by `tools/miniprojects/ex091_truth.py` under policy C10
(`COURSE_POLICIES.md`), with a worked example of each.

| | the problem | the deep learning | the ground truth given | required |
|---|---|---|---|---|
| **MP9.1A · Two fronts that meet** | two streams collide into fronts along x = ½ and y = ½ that cross at the centre; Re 100 and 200 | a PINN for u, v with points that follow the fronts, or training in time order | finite differences, 257 × 257, upwind, RK3; worked example: steepest du/dx 4.63 (Re 100) and 10.24 (Re 200) at t = 1 | u, v within 0.02 (Re 100) and 0.05 (Re 200); steepest gradient within 10 % |
| **MP9.1B · What is the viscosity?** | a few probes record u, v; the viscosity, and so Re, is unknown; three cases | an inverse PINN with a trainable ν, fitted to the equations and the probes | the exact solution at a hidden ν, sampled with noise 1e-03; worked example: case 3's probes see a thousandth of the noise | ν within 5 % (cases 1, 2); case 3 shown unidentifiable |
