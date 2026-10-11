# Ex_10.2 — Steam, Charge and Heat in a Solid Oxide Electrolysis Cell

**Paired with L10.2 · Solid Oxide Cells · Part 2**

A solid oxide electrolysis cell, 100 × 100 mm of electrode, in **one
notebook** (course policies C11 and C13). Along the fuel channel three
quantities change together - the steam fraction, the local current density
and the temperature - and each depends on the others: a **multiphysics**
problem. **Part A** trains one physics-informed network for every operating
point (cell voltage 1.15 to 1.35 V, inlet temperature 750 to 850 °C) and
measures what it gains over coupled finite volumes when many operating points
are asked for. **Part B** defines the cost of an hour of operation
(electricity, hydrogen sold, ageing of the stack) and finds the optimal cell
voltage under a 10 K temperature limit by a gradient through the network.

Restructured on 11 October 2026; the single-equation version (isothermal, one
resistance, the voltage as the only input) is in git history.

## The problem

```
steam balance      y_x = y_xx / Pe - K_I i                        Pe = 125
energy balance     theta_x = theta_xx / Pe_T + S_Q i (V - V_tn)   Pe_T = 16, theta = (T - T_in) / 50 K
charge transfer    V = E(y, T) + (RT/F) asinh(i / 2 i_0) + i ASR(T) - (RT/2F) ln(1 - i / i_L(y))
y = 0.90, theta = 0 at the inlet (built into the network); zero slopes at the outlet (a loss term)
```

The charge transfer has no derivative: at every point it is an equation for
the current density given the steam fraction and the temperature. Its
current uses steam and releases (above $V_{tn}$ = 1.29 V) or absorbs (below)
heat, and the temperature changes the resistance, the kinetics and the Nernst
potential in turn.

**The kinetic and transport parameters, the heat transport, the ageing law
and the prices are estimates chosen for the exercise**, of the right order
and not measured on any cell; `problem.py` marks each `[estimate]` and the
notebook's data table says so. The thermodynamics is standard.

## The notebook

```
Ex10.2_pinn_steam_channel.ipynb         the exercise: two TODO cells, the answer in the comment above each
Ex10.2_pinn_steam_channel_light.ipynb   the same notebook with every cell written out
```

| section | what the student does |
|---|---|
| 1 – 3 | the cell, the PEM and alkaline comparison, the data, the three laws, and the four difficulties of the coupling, with a drawing of what each law passes to the others |
| 4 | the reference (SciPy's boundary-value solver, checked at two tolerances); coupled finite volumes, Newton on steam and temperature together, for an agreed 0.01 in steam fraction and 0.5 K; what leaving the heat out changes |
| 5 | Part A: the parametric network (x, V, T_in in; y, T, i out; the inlet and the current limit built in), the two balances (TODO 1) and the charge transfer scaled by 50 mV (TODO 2); a short training, then the saved weights; twelve operating points scored; 1000 operating points timed against finite volumes |
| 6 | Part B: the cost of an hour, the 10 K limit as a penalty, gradient descent on the voltage through the network, checked by 81 voltages solved by finite volumes |
| 7 | what the notebook says |
| 8 | the report and its PDF |
| 9 | mini project proposal |

### What it measures (2-core CPU, seed 88, 11 October 2026)

| | worst error over 12 operating points: steam fraction | temperature | current density | 1000 operating points |
|---|---|---|---|---|
| finite volumes, 50 cells (the coarsest for 0.01 and 0.5 K) | 0.0068 | 0.40 K | 0.023 A/cm² | 25 s (25 ms each) |
| PINN, 4 × 64, every operating point (saved weights) | 0.0022 | 0.19 K | 0.016 A/cm² | 0.05 s in one batch, about 500 times faster |

The saved weights took 9000 Adam steps and L-BFGS, 33 minutes on the CPU,
repaid after about 80 000 operating points. Leaving the heat out overstates
the cell current at 1.15 V by 8 %. At an inlet of 800 °C the cost of an hour
is lowest at 1.167 V, where the cell cools by 25 K; with the 10 K limit the
optimum is 1.250 V. The network's gradient and the sweep of finite volumes
agree on both.

## Files

| | |
|---|---|
| `course_core.py` | shared by the whole course — `set_seed`, `MLP`, `to_tensor`, `check` |
| `pinn_core.py` | the PDE machinery — `grad`, `d2`, `train_two_stage` |
| `problem.py` | **this** problem — the data, the three laws (NumPy or torch), the local current by bisection, the boundary-value reference, the cost of an hour, the sampler, the drawings |
| `Ex10.2_weights.pt` | the trained network the notebook loads; written by the notebook itself with `FULL_TRAINING = True` |

The first two are generated. Edit `tools/pinn/*.py` and run
`python3 tools/pinn/sync_cores.py`; never edit a copy.

**The notebook is generated too**, both forms from one source,
`tools/exercises/ex102/build_ex102.py`, with the cells every one-notebook set
shares in `tools/exercises/part2_notebook.py`. Edit the builder and rerun it
rather than editing a notebook, or the two forms drift apart. If the network
or its loss changes, the weights must be trained again: run the light
notebook with `FULL_TRAINING = True`.

**On Colab nothing needs uploading**: the first code cell fetches the three
modules and the weights from the public course repository, afresh on every
run.

## Expected runtime

About two and a half minutes on a CPU, a minute and a half of it the short
training of section 5 and 25 s the timing of finite volumes. With
`FULL_TRAINING = True`, about half an hour on a CPU.

## Reference texts

Liu, G.R., *PINN with Python: An Introduction* (2025).
Raissi, Perdikaris & Karniadakis, *Physics-informed neural networks*,
J. Comput. Phys. **378** (2019) 686–707.

These are the works to read for the theory. **The code, the problem and the
exposition in this exercise set are original to this course.**

## Before this is assigned

The light version has been executed end to end on a local CPU; the exercise
version stops at its TODO cells. Still to do, as for every set (C8): a run from
a fresh Colab runtime, and a review by someone other than the author - in
particular of the estimated parameters.

## Mini project proposal

The set ends with two mini projects (section 9). Each student chooses one
mini project from the Part 2 sets and solves it individually in one month. The
ground truth is given, built by `tools/miniprojects/ex102_truth.py` under policy C10
(`COURSE_POLICIES.md`), with a worked example of each.

| | the problem | the deep learning | the ground truth given | required |
|---|---|---|---|---|
| **MP10.2A · The channel across its height, and the electrode under it** | the notebook's cell, isothermal, with the steam resolved across the 1 mm channel and through the 0.5 mm porous electrode | a PINN for y(x, Z) and i(x), the charge transfer at the reaction site as a loss term | Crank–Nicolson march with Rannacher's start; worked example: the 1-D model overstates the current by 5 to 9 % | total current within 0.5 %, outlet mean within 0.005, reaction-site steam within 0.01 |
| **MP10.2B · A day of operation under an hourly price** | a day under an hourly electricity price: V and T_in for each of 24 hours, a hydrogen contract, the 10 K limit | the notebook's parametric network as the model of every hour; gradient descent on 48 numbers | coupled finite volumes on 81 × 41 operating points, the contract by a Lagrange multiplier; worked example: 0.9 % below the best constant point | the day's cost within 0.5 %, the contract within 0.5 %, the limit checked by finite volumes |

