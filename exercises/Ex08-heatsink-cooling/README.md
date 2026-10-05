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
| 15 | four mini project proposals |

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

## Mini project proposal

The set ends with four mini projects (section 15 of the notebook). Each student chooses one
mini project from the Part 2 sets and solves it individually in one month. The
ground truth is given, built by `tools/miniprojects/ex08_truth.py` under policy C10
(`COURSE_POLICIES.md`), with a worked example of each.

| | the problem | the deep learning | the ground truth given | required |
|---|---|---|---|---|
| **MP8.1A · The fins resolved** | the heat sink of the set, fins resolved: conduction in a comb with h on every air face | a PINN over the comb, points in the metal only, one weighted convection term over all the air faces | finite volumes, 28 800 cells of 0.25 mm; worked example: hottest 51.68 K against the collapsed model's 52.45 K | hottest within 0.1 K, field within 0.2 K, heat out within 1 % |
| **MP8.1B · Two modules on one sink** | two 75 W modules on the sink of the set; the gap between them from 0 to 40 mm | one PINN with the gap as a third input, then the best gap by a gradient through it | the exact series by superposition; worked example: 52.45 K at no gap, 49.28 K at 30 mm | hottest within 0.1 K at every gap, the optimum within 2 mm, one training |
| **MP8.2A · Thermal cycling** | the heat sink of the set under a train of ten 20 s pulses, one a minute | a PINN with the lumped train built in, trained period by period or with the cycle as an input | the exact series by superposition; worked example: the swing settles at 13.45 K by the fifth cycle | peaks and troughs within 0.3 K, the settling cycle exact, the final field within 0.5 K |
| **MP8.2B · Heat capacity and coefficient together** | the thermocouple record of the set with both rho c_p and h_eff unknown | the inverse PINN with two trainable numbers, and the sensitivity of the fit to each | the record and the true values; worked example: the rise fixes rho c_p, the tail h_eff | both within 5 %, the two misfit plots, the paragraph |
