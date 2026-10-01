# Ex_12.1 — The State of a Grid

**Paired with L12.1 · Power Grid Stability Estimation · Part 2**

State estimation on a six-bus transmission grid, in **one notebook** (course
policies C11 and C13). A control room sees a few meters and has to say what
the voltage is at every bus. Students do that with the estimator every
utility runs, weighted least squares, and with a network that proposes the
state and is judged by the power flow equations, and compare the two on
accuracy and on time. Then a line faults, the local plant swings, and its
inertia is found from the frequency it recorded.

## The problem

```
P_i = V_i Σ_k V_k (G_ik cos θ_ik + B_ik sin θ_ik)        the power flow equations,
Q_i = V_i Σ_k V_k (G_ik sin θ_ik − B_ik cos θ_ik)        at every bus
z_j = h_j(V, θ) + noise                                   every reading is a function of the state
dδ/dt = ω − ω_s                                           the swing equation of the plant
(2H/ω_s) dω/dt = P_m − P_e(δ) − D (ω − ω_s)/ω_s
```

Six buses, six lines, per unit on 100 MVA; the state is six voltage
magnitudes and five angles. **The grid is representative of eastern Denmark,
not a model of it**: its line impedances are textbook values, because no
operator publishes measured ones, and the accuracy of a forecast (0.05 p.u.)
is assumed. Results that depend on the impedances must say so.

## The notebook

```
Ex12.1_pinn_grid_state.ipynb         the exercise: two TODO cells, the answer in the comment above each
Ex12.1_pinn_grid_state_light.ipynb   the same notebook with every cell written out
```

| section | what the student does |
|---|---|
| 1 – 3 | the problem, its data, its physics: the state, the three sets of readings, the power flow equations |
| 4 | the reference (Newton's power flow on 300 operating points) and weighted least squares on the full, the thin and the forecast set, for an agreed 0.01 p.u. |
| 5 | a network from 14 readings to the state, the power flow in torch (TODO 1), the loss with no true state in it (TODO 2); the two compared; then a line out that neither is told of |
| 6 | the inverse problem: the inertia of the plant from a recorded disturbance, by shooting and by a physics-informed fit, from two starts, after the fault and before it |
| 7 | what the notebook says |
| 8 | the report and its PDF |
| 9 | mini project proposal |

It replaces notebooks 00 to 05. Left to the lecture: bad-data detection by
normalised residuals, the placement of one more meter, the critical clearing
time, and the rate of change of frequency on the Nordic record.

### What it measures (CPU, seed 88)

| worst bus, mean over 300 operating points | magnitude | angle | one estimate | training |
|---|---|---|---|---|
| weighted least squares, full set (14 readings) | 0.0030 p.u. | 0.07° | 1.1 ms | — |
| weighted least squares, thin set (6 readings, 5 of 11 unknowns) | 0.0248 p.u. | 1.82° | 0.8 ms | — |
| weighted least squares, forecast set | 0.0050 p.u. | 0.15° | 1.1 ms | — |
| network, 9995 weights, forecast set, no true state | 0.0043 p.u. | 0.15° | 0.26 ms; 2 µs each in a batch of 300 | 13 s |

With line 1–2 out and neither told: least squares 0.0512 p.u., the network
0.0516 p.u. (least squares told of it: 0.0070 p.u.); the network's own check
rises from 0.28 to 2.35.

| inertia (the record was made with 4.00 s) | started at 8 s | started at 2 s |
|---|---|---|
| after the fault, shooting and least squares | 7.41 s (misfit 0.451 Hz) | 1.83 s (0.457 Hz) |
| after the fault, physics-informed | 3.89 s (0.002 Hz) | 3.90 s (0.002 Hz) |
| before the fault, shooting and least squares | 50.00 s, the bound | 1.94 s |
| before the fault, physics-informed | 8.35 s | 2.10 s |

**Three things worth knowing.**

*The network does not beat least squares on accuracy, and the notebook does
not say it does.* From the same readings the two agree; the network's gain is
the time of one estimate. The earlier notebook 02 reported its estimator as
far better than least squares on the thin set. That estimator had the state
itself as trainable numbers, and its physics term penalised the injections
only at the buses where they were metered - the same information the
measurement term already had. What kept its unmetered buses near 1.0 p.u. was
the logistic bound and the flat start, not the physics.

*The earlier inertia fit (`pb.identify_inertia`) does not recover the
inertia.* Run on the old notebook's own window it returns 73 s and 127 s for
a machine of 4 s: its residual is written as dω/dt = (…)/H, which a very
large H satisfies, and it applies the post-fault network to the fault itself.
The fit in section 6 writes the equation with H on the left and uses only the
record after the fault is cleared. `identify_inertia`, `dynamic_pinn` and
`algebraic_pinn` are still in `problem.py`, unused by the notebook.

*Shooting has local minima here and the physics-informed fit did not.* That
is a finding about this record and these two starts, not a general law.

## Files

| | |
|---|---|
| `course_core.py` | shared by the whole course — `set_seed`, `MLP`, `to_tensor`, `check` |
| `pinn_core.py` | the Part 2 helpers — `grad`, `train_two_stage` |
| `problem.py` | **this** problem — the grid, the power flow, the meters, weighted least squares, the machines and the swing equation; section 9 holds what the one notebook adds: operating points, the three sets of readings, the recorded disturbance and the classical inertia fit |

The first two are generated. Edit `tools/pinn/*.py` and run
`python3 tools/pinn/sync_cores.py`; never edit a copy.

**The notebook is generated too**, both forms from one source,
`tools/exercises/ex121/build_ex121.py`, with the cells every one-notebook set
shares in `tools/exercises/part2_notebook.py`. Edit the builder and rerun it
rather than editing a notebook, or the two forms drift apart.

**On Colab nothing needs uploading**: the first code cell fetches the three
modules from the public course repository, afresh on every run.

## Expected runtime

CPU only. About a minute and a half: 15 s for the estimator, about a minute
for the eight inertia fits.

## Reference texts

Liu, G.R., *PINN with Python: An Introduction* (2025).
Raissi, Perdikaris & Karniadakis, *Physics-informed neural networks*,
J. Comput. Phys. **378** (2019) 686–707.
For the power-system side: Schweppe & Wildes (1970); Abur & Expósito,
*Power System State Estimation* (2004); Kundur, *Power System Stability and
Control* (1994), ch. 11 – 13.

These are the works to read for the theory. **The code, the problem and the
exposition in this exercise set are original to this course.**

## Before this is assigned

The light version has been executed end to end on a local CPU; the exercise
version stops at its TODO cells. Still to do, as for every set (C8): a run from
a fresh Colab runtime, and a review by someone other than the author - in
particular by someone who runs a state estimator for a living.

## Mini project proposal

The set ends with two mini projects (section 9). Each student chooses one
mini project from the Part 2 sets and solves it individually in one month. The
ground truth is given, built by `tools/miniprojects/ex121_truth.py` under policy C10
(`COURSE_POLICIES.md`), with a worked example of each.

| | the problem | the deep learning | the ground truth given | required |
|---|---|---|---|---|
| **MP12.1A · A bigger grid, many machines** | the WSCC 9-bus system, 3 machines; a fault at bus 7 cleared after 0.10 s | a PINN for every machine's angle and speed, coupled swing equations as the residual | RK4 at 0.5 ms on the reduced network; worked example: machine 2 swings to 83.04°, CCT 0.183 s | angles within 1°, speeds within 0.01 Hz |
| **MP12.1B · The inertia of every machine** | the 9-bus system; H and D of all three machines hidden; frequency after three faults | an inverse PINN with every H and D trainable | RK4 records every 10 ms, noise 1 mHz; worked example: inertias move the records 84-437 mHz, dampings 7-31 | inertias within 5 %, dampings within 20 %, total within 2 % |
