# Ex_12.1 (new) — Basin Stability of a Grid

**Paired with L12.1 · Power Grid Stability Estimation · Part 2**

*Written on 13 October 2026 beside the six-bus Ex12.1 (`Ex12.1-power-grid-stability-estimation`,
state estimation), which stays as it is. The folder and the set key carry an `N` until the
lecturer chooses which of the two is assigned.*

Basin stability on the IEEE 39-bus grid, in **one notebook** (policies C11 and C13). After a
large disturbance, does the grid come back into step, and from which buses is it easiest to
throw out of step? Students see what the conventional studies say (the eigenvalues of the
linearised model; the critical clearing time of a fault by bisection), compute the
single-node basin stability of every bus by Monte Carlo, and train a graph network that
estimates the whole map from the grid's data; the two are compared on accuracy and on time,
then on renewable shares and tripped branches the network never saw.

## The problem

```
dδ_i/dt = ω_i,   M_i dω_i/dt = P_i − D_i ω_i − P_e,i                machines and converters
T_i dE_i/dt = E_f,i − E_i + X_i Σ_j B_ij E_j cos(δ_i − δ_j)        machines only
D_i dδ_i/dt = P_i − P_e,i                                           load buses (no inertia)
P_e,i = Σ_j E_i E_j B_ij sin(δ_i − δ_j)
S_i = the share of random kicks at bus i (angle anywhere, frequency up to 1 Hz)
      after which the grid is back in step 10 s later
```

The physics is `grid39.py`, shared with the new Ex12.2. The network and power flow are
MATPOWER's `case39`; the machines' $H$, $x_d$, $x'_d$, $T'_{d0}$ are Pai (1989), **typed from
memory and to be checked against the book**. **Assumed, not data**: a load bus has no inertia
and follows its neighbours within 10 ms; the machines' damping ($D/M$ = 1 1/s); a converter's
inertia (2 s) and damping ($D/M$ = 1 1/s).

## The notebook

```
Ex12.1_basin_stability.ipynb         the exercise: two TODO cells, the answer in the comment above each
Ex12.1_basin_stability_light.ipynb   the same notebook with every cell written out
Ex12.1_snbs_maps.npz                 the maps the network learns from (tools/exercises/grid39/make_snbs_data.py)
```

| section | what the student does |
|---|---|
| 1 – 3 | the problem, its data, its physics: the model's right-hand side (TODO 1); the eigenvalues, the critical clearing time of one fault, two kicks of one machine |
| 4 | basin stability by Monte Carlo: the base map with 100 kicks per bus, and 15 test scenarios against the reference (300 kicks) |
| 5 | a graph network for the map (TODO 2: the graph layer), trained on 200 maps; compared with Monte Carlo |
| 6 | two held-out groups: 45 to 70 % converters, and a branch never out in training |
| 7 | what the notebook says |
| 8 | the report and its PDF |
| 9 | mini project proposal |

### What it measures (CPU, one thread, seed 88)

Every eigenvalue has a negative real part (largest −0.23 1/s); the machines swing at 0.48 to
1.41 Hz. A fault at bus 6 cleared by tripping branch 5-6: critical clearing time 219 ms, eight
simulations. Machine 2 kicked by 1.5 rad returns; by 2.0 rad it falls out of step.

Base map: every load bus 1.00; machine 2 (bus 31) 0.43, machine 5 (bus 34) 0.50, the others
0.62 to 0.72, machines 1 and 10 at 1.00. Over the training data a converter's bus averages 0.64
against 0.69 for a machine's.

| 15 test scenarios, against the reference (300 kicks) | mean | at the machines | worst | one map |
|---|---|---|---|---|
| Monte Carlo, 100 kicks per bus | 0.010 | 0.035 | 0.17 | about 15 s |
| graph network, 42 497 weights | 0.006 | 0.019 | 0.08 | 0.32 ms |

Training: about 150 s, offline. Held out: 45 to 70 % converters 0.019 at the machines (worst
0.11, bias +0.004); a branch never out in training 0.020 (worst 0.09).

**Three things worth knowing.**

*The network is closer to the reference than Monte Carlo with 100 kicks.* It learned the
average over 200 scenarios, which smooths the sampling noise of any one map. Its labels are
themselves 100-kick maps; the reference it is scored against has 300.

*The held-out groups did not find the network's limit.* In this model a converter lowers its
bus's basin stability by only about 0.05, and the training data have converters at all nine
places, so 45 to 70 % is a modest extrapolation. The notebook says so rather than claiming
the network generalises; MP12.1A is a harder test.

*The load model was chosen for Ex12.2 and holds here.* Load buses with inertia made the model
ring at 5 to 45 Hz after every event; first-order load buses do not ring, and moved the
machines' basin stability by at most about 0.07 against the inertial-load version (60 kicks per
bus, different draws).

## Files

| | |
|---|---|
| `course_core.py`, `pinn_core.py` | generated (`tools/pinn/sync_cores.py`) |
| `grid39.py` | generated: the IEEE 39-bus grid and its model, from `tools/exercises/grid39/grid39_src.py` by `sync.py` |
| `problem.py` | this set: the maps, the numbers a bus is described by, the graph |

**The notebook is generated**, both forms from `tools/exercises/ex121n/build_ex121n.py`; the
maps by `tools/exercises/grid39/make_snbs_data.py` (about 75 minutes on a laptop CPU, seeded).

## Before this is assigned

The light version has been executed end to end on a CPU. Still to do (C8): a run from a fresh
Colab T4 runtime (with the GPU timings), a review by someone other than the author, the machine
data checked against Pai (1989). L12.1 does not yet carry this set's story.

## Reference texts

Kundur, *Power System Stability and Control* (1994), ch. 12–13. Menck, Heitzig, Kurths &
Schellnhuber, *How dead ends undermine power grid stability*, Nat. Commun. 5 (2014) 3969.
Nauck et al., *Predicting basin stability of power grids using graph neural networks*, New J.
Phys. 24 (2022) 043041. Liu, *PINN with Python* (2025). **The code, the problem and the
exposition in this exercise set are original to this course.**

## Mini project proposal

The set ends with two mini projects (section 9). Each student chooses one
mini project from the Part 2 sets and solves it individually in one month. The
ground truth is given, built by `tools/miniprojects/ex121n_truth.py` under policy C10
(`COURSE_POLICIES.md`), with a worked example of each.

| | the problem | the deep learning | the ground truth given | required |
|---|---|---|---|---|
| **MP12.1A · Where to put grid-forming converters** | six machines replaced by converters; make two of them grid-forming - which two? | the set's graph network with grid-forming as an input, ranking all 15 placements | basin-stability maps of all 16 choices, 300 kicks per bus; worked example: 0.63 at the converters with none, 0.72 with the best pair | maps within 0.05; a pair within 0.02 of the best mean at the converters |
| **MP12.1B · The inertia of every machine** | H and D/M of the nine local machines hidden; frequency after three branch trips | an inverse physics-informed network with every H and D/M trainable | RK4 records every 20 ms, noise 1 mHz; worked example: inertias move the records 1.22-9.04 mHz, dampings 0.15-1.30 | inertias within 5 %, dampings within 20 %, total within 2 % |

