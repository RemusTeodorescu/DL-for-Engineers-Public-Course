# Ex_12 — Power Grid Stability (the course's sets)

**Paired with L12.1 and L12.2 · Part 2.** Two sets on the same six-bus grid, side by side in
this folder. Each is one notebook and its light form, with its own physics module.

| set | lecture | notebook | physics | question |
|---|---|---|---|---|
| **Ex12.1** | L12.1 Estimation | `Ex12.1_pinn_grid_state.ipynb` (+ `_light`) | `problem_12_1.py` | the state of the grid from its meters, and a machine's inertia |
| **Ex12.2** | L12.2 Prediction | `Ex12.2_gnn_contingency_screening.ipynb` (+ `_light`) | `problem_12_2.py`, `andes_compare.py` | is the grid secure after each line fault: simulation, ANDES and a graph network |

`course_core.py` and `pinn_core.py` are shared by both. The larger 39-bus grid with
inference on the Jetson boards is optional and solved, in `exercises/Ex12_optional`.

## Ex_12.1 — The State of a Grid

**Paired with L12.1 · Power Grid Stability Estimation · Part 2**

State estimation on a six-bus transmission grid, in **one notebook** (course
policies C11 and C13). The state of the grid - the voltage magnitude and
phase angle at every bus - is computed from a small number of noisy
measurements and the power flow equations. Students do it with weighted least
squares, the method used in control centres, and with a neural network that
returns the state and is trained with the power flow equations, and compare
the two on accuracy and computing time. Then a short circuit on a line makes
the rotor angle of the local generator oscillate, and the generator's inertia
constant is estimated from the frequency recorded at its terminals.

### The problem

```
P_i = V_i Σ_k V_k (G_ik cos θ_ik + B_ik sin θ_ik)        the power flow equations,
Q_i = V_i Σ_k V_k (G_ik sin θ_ik − B_ik cos θ_ik)        at every bus
z_j = h_j(V, θ) + noise                                   every measurement is a function of the state
dδ/dt = ω − ω_s                                           the swing equation of the plant
(2H/ω_s) dω/dt = P_m − P_e(δ) − D (ω − ω_s)/ω_s
```

Six buses, six lines, per unit on 100 MVA; the state is six voltage
magnitudes and five angles. **The grid is representative of eastern Denmark,
not a model of it**: its line impedances are textbook values, because no
operator publishes measured ones, and the accuracy of a forecast (0.05 p.u.)
is assumed. Results that depend on the impedances must say so.

### The notebook

```
Ex12.1_pinn_grid_state.ipynb         the exercise: two TODO cells, the answer in the comment above each
Ex12.1_pinn_grid_state_light.ipynb   the same notebook with every cell written out
```

| section | what the student does |
|---|---|
| 1 – 3 | the problem, its data, its physics: the state, the three sets of readings, the power flow equations |
| 4 | the reference (the power flow solved by Newton–Raphson on 300 operating points) and weighted least squares on the full, the thin and the forecast set, for an agreed 0.01 p.u. |
| 5 | a network from 14 readings to the state, the power flow in torch (TODO 1), the loss with no true state in it (TODO 2); the two compared; then a line out that neither is told of |
| 6 | the inverse problem: the inertia of the plant from a recorded disturbance, by shooting and by a physics-informed fit, from two starts, after the fault and before it |
| 7 | what the notebook says |
| 8 | the report and its PDF |
| 9 | mini project proposal |

It replaces notebooks 00 to 05. Left to the lecture: bad-data detection by
normalised residuals, the placement of one more meter, the critical clearing
time, and the rate of change of frequency on the Nordic record.

#### What it measures (CPU, seed 88)

| worst bus, mean over 300 operating points | magnitude | angle | one estimate | training |
|---|---|---|---|---|
| weighted least squares, full set (14 readings) | 0.0030 p.u. | 0.07° | 2.9 ms | — |
| weighted least squares, thin set (6 readings, 5 of 11 unknowns) | 0.0240 p.u. | 1.81° | 1.8 ms | — |
| weighted least squares, forecast set | 0.0050 p.u. | 0.15° | 1.8 ms | — |
| neural network, 9995 weights, forecast set, trained with the power flow, no true state | 0.0043 p.u. | 0.15° | 0.36 ms; 3.5 µs each in a batch of 300 | 36 s |

With line 1–2 out and neither told: least squares 0.0512 p.u., the network
0.0516 p.u. (least squares told of it: 0.0070 p.u.); the network's own check
rises from 0.29 to 2.35.

| inertia (the record was made with 4.00 s) | started at 8 s | started at 2 s |
|---|---|---|
| after the fault, shooting and least squares | 7.41 s (misfit 0.451 Hz) | 1.83 s (0.457 Hz) |
| after the fault, physics-informed | 3.89 s (0.002 Hz) | 3.89 s (0.002 Hz) |
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
`algebraic_pinn` are still in `problem_12_1.py`, unused by the notebook.

*Shooting has local minima here and the physics-informed fit did not.* That
is a finding about this record and these two starts, not a general law.

### Files

| | |
|---|---|
| `course_core.py` | shared by the whole course — `set_seed`, `MLP`, `to_tensor`, `check` |
| `pinn_core.py` | the Part 2 helpers — `grad`, `train_two_stage` |
| `problem_12_1.py` | **this** problem — the grid, the power flow, the meters, weighted least squares, the machines and the swing equation; section 9 holds what the one notebook adds: operating points, the three sets of readings, the recorded disturbance and the classical inertia fit |

The first two are generated. Edit `tools/pinn/*.py` and run
`python3 tools/pinn/sync_cores.py`; never edit a copy.

**The notebook is generated too**, both forms from one source,
`tools/exercises/ex121/build_ex121.py`, with the cells every one-notebook set
shares in `tools/exercises/part2_notebook.py`. Edit the builder and rerun it
rather than editing a notebook, or the two forms drift apart.

**On Colab nothing needs uploading**: the first code cell fetches the three
modules from the public course repository, afresh on every run.

### Expected runtime

CPU only. About two minutes: about half a minute for the estimator, a minute
and a half for the eight inertia fits.

### Reference texts

Liu, G.R., *PINN with Python: An Introduction* (2025).
Raissi, Perdikaris & Karniadakis, *Physics-informed neural networks*,
J. Comput. Phys. **378** (2019) 686–707.
For the power-system side: Schweppe & Wildes (1970); Abur & Expósito,
*Power System State Estimation* (2004); Kundur, *Power System Stability and
Control* (1994), ch. 11 – 13.

These are the works to read for the theory. **The code, the problem and the
exposition in this exercise set are original to this course.**

### Before this is assigned

The light version has been executed end to end on a local CPU; the exercise
version stops at its TODO cells. Still to do, as for every set (C8): a run from
a fresh Colab runtime, and a review by someone other than the author - in
particular by someone who runs a state estimator for a living.

### Mini project proposal

The set ends with two mini projects (section 9). Each student chooses one
mini project from the Part 2 sets and solves it individually in one month. The
ground truth is given, built by `tools/miniprojects/ex121_truth.py` under policy C10
(`COURSE_POLICIES.md`), with a worked example of each.

| | the problem | the deep learning | the ground truth given | required |
|---|---|---|---|---|
| **MP12.1A · The state of the 39-bus grid** | the IEEE 39-bus grid, 77 unknowns from 90 readings: PMUs, machine meters, forecasts, known zeros | the set's neural network trained with the power flow equations, scaled to 39 buses | true states by pandapower's power flow; worked example: least squares within 0.0041 p.u., 3.3 ms | worst bus within 0.01 p.u.; faster than least squares; the line out seen |
| **MP12.1B · The inertia of every machine** | the IEEE 39-bus grid; H and D of nine machines hidden; frequency after three faults | the set's inverse problem for ten coupled machines, every H and D trainable | RK4 records every 20 ms, noise 1 mHz; worked example: inertias move the records 25-92 mHz, dampings 1.8-9.4 | inertias within 5 %, dampings within 20 %, total within 2 % |

---

## Ex_12.2 — Screening Contingencies

**Paired with L12.2 · Prediction of Power Grid Stability · Part 2**

N-1 screening on the six-bus grid of Ex_12.1, in **one notebook** (course
policies C11 and C13). A line faults and the protection needs 140 ms to clear
it: does the local plant stay in step? The answer is the critical clearing
time, and finding it takes a simulation in time for every line and every
operating point. Students compute it by simulation and by a graph network
trained on simulations, compare the two on accuracy and on time, and then ask
the network about a line it has never seen out.

### The problem

```
dδ_i/dt = ω_i − ω_s                                        the swing equation of each machine
(2H_i/ω_s) dω_i/dt = P_m,i − P_e,i(δ) − D_i (ω_i − ω_s)/ω_s
CCT = the longest fault for which the angle between the machines stays under 180°
insecure  ⇔  CCT ≤ 140 ms                                  the protection's clearing time
```

180 operating points × 6 contingencies = 1080 cases; 16 % are insecure.
**The grid is representative of eastern Denmark, not a model of it**, the
plant's inertia, damping and reactance are estimates, and the labels are the
classical machine model's: no exciter, no governor, first swing only.

### The notebook

```
Ex12.2_gnn_contingency_screening.ipynb         the exercise: two TODO cells, the answer in the comment above each
Ex12.2_gnn_contingency_screening_light.ipynb   the same notebook with every cell written out
```

| section | what the student does |
|---|---|
| 1 – 3 | the problem, its data, its physics: the contingencies, the swing equation, one case cleared 10 ms before and after its critical time |
| 4 | the reference (RK4 on a 2 ms step, bisected to 2 ms, all 1080 cases) and the same simulation on steps of 5, 10 and 20 ms, for an agreed 5 ms |
| 5 | the worst contingency in ANDES, a dedicated stability solver: the classical machine and a machine with exciter and governor (added 13 October 2026) |
| 6 | a graph network: the graph layer (TODO 1), the loss (TODO 2), trained on 144 operating points and tested on 36 held out; the three compared |
| 7 | the same network trained without line 0 out and asked about it; the threshold raised until nothing is missed |
| 8 | what the notebook says |
| 9 | the report and its PDF |
| 10 | mini project proposal |

It replaces notebooks 00 to 05. Left to the lecture: the dense baseline and
the relabelling test (no dense network is trained here, so the notebook does
not show that a graph network beats one), and the hand-computed label.

**Section 5 measures, for the worst contingency (line 0 out, base operating
point):** the course's simulation 219.7 ms (0.3 s to find), ANDES with the
classical machine 209.0 ms (11 s), ANDES with GENROU, ESST1A and TGOV1
162.6 ms (19 s). (`andes_compare.py` is from Shahariya Rasheed's ANDES comparison, whose separate supplement notebook was removed on 13 October 2026 when its essentials moved into section 5.)

#### What it measures (CPU, seed 88)

| 216 held-out cases | mean error | worst | insecure missed | false alarms | one case | training |
|---|---|---|---|---|---|---|
| reference, step 2 ms | — | — | — | — | 3.0 ms | — |
| simulation, step 5 ms (the coarsest for 5 ms) | 1.66 ms | 4.88 ms | 0 of 26 | 5 | 1.1 ms | — |
| graph network, 6721 weights | 0.58 ms | 4.76 ms | 0 of 26 | 0 | 7 µs | 24 s |

Over all 1080 cases the 5 ms step changes 19 verdicts, the 10 ms step 53.

| line 0 out, 36 held-out cases | mean error | insecure missed |
|---|---|---|
| network that has seen line 0 out | 0.68 ms | 0 of 11 |
| network trained without it | 56.24 ms, all on the unsafe side | 10 of 11 (11 in an earlier run) |

With the threshold raised from 140 to 185 ms the second network misses none
and raises no false alarm: its error is a nearly constant offset, so it still
ranks the cases correctly. The 185 ms was found with the truth in hand.

**Two things worth knowing.**

*The labels are simulated for every case at once.* `pb.build_cases` runs the
same Runge–Kutta integration and the same bisection as `pb.label_case` with a
leading case axis, and returns the same 1080 labels to the last bit in a few
seconds instead of three minutes. The per-case times in the table are that batched
simulation's; one case on its own, by `pb.label_case`, takes about 0.2 s.

*The lecture's four-way table has one row measured here.* L12.2 says of a
graph network on an unseen topology that structure helps. This notebook
measures that row and finds 57 ms of error on the unsafe side; it does not
train the dense network, so it cannot say whether that is better or worse
than a dense one would do. The physics-informed graph network is not built.

### Files

| | |
|---|---|
| `course_core.py` | shared by the whole course — `set_seed`, `MLP`, `to_tensor`, `check` |
| `pinn_core.py` | the Part 2 helpers — `train_two_stage` |
| `problem_12_2.py` | **this** problem — the grid, the machines, the fault model, the contingencies, the operating points, the labels (`label_case`, `build_cases`), the features and the adjacency |
| `andes_compare.py` | the six-bus grid in ANDES, used by section 5 |

The first two are generated. Edit `tools/pinn/*.py` and run
`python3 tools/pinn/sync_cores.py`; never edit a copy.

**The notebook is generated too**, both forms from one source,
`tools/exercises/ex122/build_ex122.py`. Edit the builder and rerun it rather
than editing a notebook. Edge inference on the Jetson boards is in the optional
39-bus Ex12.2 (`exercises/Ex12_optional`).

### Expected runtime

About a minute and a half on a CPU: two trainings of about 25 s, four runs of
the simulation, and about 30 s of ANDES (installed by the notebook on Colab, about 30 s more).

### Reference texts

Liu, G.R., *PINN with Python: An Introduction* (2025).
Raissi, Perdikaris & Karniadakis, *Physics-informed neural networks*,
J. Comput. Phys. **378** (2019) 686–707.
Kundur, *Power System Stability and Control* (1994), ch. 2 and 13.

These are the works to read for the theory. **The code, the problem and the
exposition in this exercise set are original to this course.**

### Before this is assigned

The light version of the study notebook has been executed end to end on a
local CPU; the exercise version stops at its TODO cells. Still to do: a run
from a fresh Colab runtime, a review by someone other than the author, and a
run of the latency notebook on the Thor.

### Mini project proposal

The set ends with two mini projects (section 10). Each student chooses one
mini project from the Part 2 sets and solves it individually in one month. The
ground truth is given, built by `tools/miniprojects/ex122_truth.py` under policy C10
(`COURSE_POLICIES.md`), with a worked example of each.

| | the problem | the deep learning | the ground truth given | required |
|---|---|---|---|---|
| **MP12.2A · Contingency screening on the 39-bus grid** | the IEEE 39-bus grid, 10 classical machines, 35 contingencies, 150 operating points | the set's graph network, scaled to 39 buses | 5250 critical clearing times by RK4 on 2 ms; worked example: 11.4 % insecure, four contingencies carry most | no insecure case missed, at most 5 % false alarms, mean error under 5 ms |
| **MP12.2B · The same screening with converters** | the grid of MP12.2A with 0 to 5 machines replaced by converters with half the inertia | the graph network with the inertia as a bus input | 5250 critical clearing times with converters; worked example: 22.2 % insecure, 569 hidden from a screen without converters | no insecure case missed, at most 5 % false alarms |
