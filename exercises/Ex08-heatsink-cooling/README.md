# Ex_08 — Heatsink Cooling of a Power Module

**Paired with L8.1 · Stationary Heat and L8.2 · Transient Heat · Part 2**

One notebook for both lectures of block 8. A 150 W power module sits on an
aluminium heat sink with ten fins in forced air. **Part 1 (L8.1)** asks how hot
the module's footprint gets in steady operation; **Part 2 (L8.2)** switches the
module on for 20 s, asks how hot it gets and how fast the sink recovers, and
then finds the fins' heat transfer coefficient from one thermocouple. Both
parts have an **exact solution**, so finite differences (FDM) and a
physics-informed neural network (PINN) are scored against the truth, and their
times are compared on what runs online, the training reported apart.

**One notebook, one session.** Part 2 continues where Part 1 stops: the
libraries, the heat sink (`heatsink.py`) and Part 1's results stay in memory,
and nothing is loaded again. Part 1's results are kept in `R1`, Part 2's in
`R2`, because Part 2 reuses Part 1's variable names; the report at the end
gathers both.

## The problem

An extruded heat sink: a base plate 100 × 10 mm of aluminium 6061
(k = 167 W/(m·K)), ten straight fins of 2 × 40 mm at a 10 mm pitch, forced air
at 40 °C with h = 40 W/(m²K). A module 30 mm wide in the middle of the top face
dissipates 150 W per 100 mm of depth. The fins collapse into one effective
coefficient on the base, h_eff = 322 W/(m²K), by the exact fin efficiency, so
the base plate is a rectangle:

```
k (T_xx + T_zz) = 0                 in the base plate       (Part 1)
rho c_p T_t = k (T_xx + T_zz)       in the base plate       (Part 2, the pulse)
k T_z = q''(x) s(t)                 on the module's face, s = 1 while the power is on
k T_z = h_eff (T - T_air)           on the fin side
T_x = 0                             at the two ends
```

The steady field is a cosine series in elementary functions, summed in closed
form where it converges slowly; the transient is the steady field less what
has not yet arrived, in the eigenfunctions of the convecting face. Both are
exact to rounding (`hs.exact`, `hs.exact_transient`). The material and air
values are typical, not measured on a product.

## The notebook

```
Ex08_heatsink_cooling.ipynb         the exercise: four TODO cells, the answer in the comment of each
Ex08_heatsink_cooling_light.ipynb   the same notebook with every cell written out
```

| section | what the student does |
|---|---|
| Part 1 · 1 – 3 | the problem, its data, its physics: the fins, the base plate, the exact solution |
| 4 | FDM on a grid, refined until an agreed 0.01 K; the grid and the PINN's points drawn |
| 5 – 6 | a PINN with the hand calculation built in (**TODO 1, 2**), the comparison, what it says |
| Part 2 · 7 – 9 | the pulse: storage, the time scales, the exact transient |
| 10 | Crank–Nicolson, refined until an agreed 0.05 K |
| 11 | a PINN in time with the lumped response built in (**TODO 3, 4**), and the comparison |
| 12 – 13 | the inverse problem: h_eff from one thermocouple; what it says |
| 14 | the report: four questions per lecture and one to conclude, one PDF for Moodle |
| 15 | the four mini projects |

Measured (exercise run on 5 October 2026): the hottest point 92.45 °C, 4.98 K
above the hand calculation; FDM meets 0.01 K at 81 × 17 nodes, the PINN about
0.3 K with an energy balance of 150 W. In the pulse the module peaks at
56.19 °C at 20 s and the thermocouple 4 s after the switch-off; Crank–Nicolson
meets 0.05 K at 321 × 65 nodes and 1200 steps, the PINN only after the pulse;
the inverse problem finds h_eff about 5 % low from one thermocouple.

## Files

| | |
|---|---|
| `course_core.py` | shared by the whole course — `set_seed`, `MLP`, `to_tensor`, `check`, `DEVICE` |
| `pinn_core.py` | the PDE machinery — `grad`, `d2`, samplers, `train_two_stage` |
| `heatsink.py` | **this** problem — the data, the exact steady and transient fields, FDM, the samplers, the drawings |

The first two are generated. Edit `tools/pinn/*.py` and run
`python3 tools/pinn/sync_cores.py`; never edit a copy.

**The notebook is generated too**, both forms from one source,
`tools/exercises/ex08/build_ex08.py`, with the cells every one-notebook set
shares in `tools/exercises/part2_notebook.py`. Edit the builder and rerun it
rather than editing a notebook, or the two forms drift apart.

**On Colab nothing needs uploading**: the first code cell fetches the three
modules from the public course repository, afresh on every run. The notebook
asks Colab for a GPU runtime (T4); if it opens on a CPU, the setup cell says so.

## Expected runtime

About twenty-five minutes for the whole notebook, most of it the three trainings
(steady, in time, inverse). Measured on 5 October 2026 on a laptop GPU: 407, 517
and 585 s; the laptop's CPU is no slower, because networks this small do not fill
a GPU. Colab's T4 against Colab's two-core CPU has not been measured yet.

## Reference texts

Liu, G.R., *PINN with Python: An Introduction* (2025).
Raissi, Perdikaris & Karniadakis, *Physics-informed neural networks*,
J. Comput. Phys. **378** (2019) 686–707.
Incropera & DeWitt, *Fundamentals of Heat and Mass Transfer*, ch. 3 (fins) and
ch. 5 (transient conduction).

These are the works to read for the theory. **The code, the problem and the
exposition in this exercise set are original to this course.**

## Before this is assigned

The light version has been executed end to end locally; the exercise version
stops at its first TODO. Still to do, as for every set (C8): a run from a fresh
Colab runtime, and a review by someone other than the author.

## Mini projects

Each mini project takes this set's problem further - a harder version of it, or its
inverse - with no exact solution. Each student chooses **one** mini project from the
Part 2 sets and solves it individually in one month; the dates and the hand-in are
given in L13. Each mini project is **one Colab notebook in five parts** under course
policy C14: A the problem, B the data and the physics and C the reference model and
the ground truth are given and solved; D the physics-informed network and E the
report are yours. The goal is the reference model's accuracy with an online answer
at least ten times faster (for an inverse problem, ten times faster than a new
least-squares fit); each project's README states the problem, what changed, the
ground truth, a worked example and the requirements.

| mini project | new against the exercise | required | notebook |
|---|---|---|---|
| [**MP8.1A · The Fins Resolved**](https://github.com/RemusTeodorescu/DL-for-Engineers-Public-Course/tree/main/miniprojects/MP8.1A) | the whole comb of base and fins, the air's coefficient on every face; power and coefficient as inputs | hottest point within 0.15 K, field within 0.2 K, heat out within 1 % | [open in Colab](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/miniprojects/MP8.1A/MP8.1A_fins_resolved.ipynb) |
| [**MP8.1B · Two Modules on One Sink**](https://github.com/RemusTeodorescu/DL-for-Engineers-Public-Course/tree/main/miniprojects/MP8.1B) | two modules a gap apart, the air warming along the sink, the fan's coefficient falling along the flow and natural convection on the fins: nonlinear, the reference iterates for every gap | hottest point within 0.1 K at every gap, best gap within 2 mm by a gradient, one training, 10× faster than Newton's method | [open in Colab](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/miniprojects/MP8.1B/MP8.1B_two_modules.ipynb) |
| [**MP8.2A · Thermal Cycling with Natural Convection**](https://github.com/RemusTeodorescu/DL-for-Engineers-Public-Course/tree/main/miniprojects/MP8.2A) | ten 20 s pulses a minute apart, 100 to 200 W, natural convection making the fin side nonlinear | every peak and trough within 0.3 K, field at 600 s within 0.5 K | [open in Colab](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/miniprojects/MP8.2A/MP8.2A_thermal_cycling.ipynb) |
| [**MP8.2B · Heat Capacity and Coefficient Together**](https://github.com/RemusTeodorescu/DL-for-Engineers-Public-Course/tree/main/miniprojects/MP8.2B) | the heat capacity and the fan's coefficient both hidden, natural convection added (inverse) | both numbers within 5 % on five records | [open in Colab](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/miniprojects/MP8.2B/MP8.2B_heat_capacity_and_coefficient.ipynb) |
