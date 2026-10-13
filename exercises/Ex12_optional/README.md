# Ex_12 optional — The IEEE 39-Bus Grid and Edge Inference

**Optional and solved.** The 39-bus, edge-inference extension of the course's six-bus Ex12.1
and Ex12.2 (`exercises/Ex12_power_grid`), for students who want the larger grid. Two sets
side by side in this folder, each one notebook and its light form, with its own physics module.

| set | lecture | notebook | physics | question |
|---|---|---|---|---|
| **Ex12.1 · 39** | L12.1 | `Ex12.1_39_edge_optional_basin_stability.ipynb` (+ `_light`) | `problem_12_1_39.py`, `Ex12.1_39_edge_snbs_maps.npz` | basin stability of every bus, by Monte Carlo and by a graph network |
| **Ex12.2 · 39** | L12.2 | `Ex12.2_39_edge_optional_realtime_screening.ipynb` (+ `_light`) | `problem_12_2_39.py`, `Ex12.2_39_edge_surrogate.pt`, `board/` | N-1 screening by a graph-network surrogate, in a real-time loop on a virtual board |

`grid39.py`, `course_core.py` and `pinn_core.py` are shared by both. The `board/` scripts run
the loop on the real Jetson AGX Orin (PMU player) and AGX Thor (screener) in the mini projects.

## Ex_12.1_39_edge_optional — Basin Stability of a Grid

**Paired with L12.1 · Power Grid Stability Estimation · Part 2**

*Optional and solved: the 39-bus, edge-inference extension of the six-bus Ex12.1, written on 13 October 2026. The six-bus set is the one the course assigns; this one is for students who want the larger grid.*

Basin stability on the IEEE 39-bus grid, in **one notebook** (policies C11 and C13). After a
large disturbance, does the grid come back into step, and from which buses is it easiest to
throw out of step? Students see what the conventional studies say (the eigenvalues of the
linearised model; the critical clearing time of a fault by bisection), compute the
single-node basin stability of every bus by Monte Carlo, and train a graph network that
estimates the whole map from the grid's data; the two are compared on accuracy and on time,
then on renewable shares and tripped branches the network never saw.

### The problem

```
dδ_i/dt = ω_i,   M_i dω_i/dt = P_i − D_i ω_i − P_e,i                machines and converters
T_i dE_i/dt = E_f,i − E_i + X_i Σ_j B_ij E_j cos(δ_i − δ_j)        machines only
D_i dδ_i/dt = P_i − P_e,i                                           load buses (no inertia)
P_e,i = Σ_j E_i E_j B_ij sin(δ_i − δ_j)
S_i = the share of random kicks at bus i (angle anywhere, frequency up to 1 Hz)
      after which the grid is back in step 10 s later
```

The physics is `grid39.py`, shared with the new Ex12.2. The network and power flow are
MATPOWER's `case39`; the machines' $H$, $x_d$, $x'_d$, $T'_{d0}$ are Pai (1989), checked on
13 October 2026 against the IEEE 39-bus case shipped with ANDES. **Assumed, not data**: a load bus has no inertia
and follows its neighbours within 10 ms; the machines' damping ($D/M$ = 1 1/s); a converter's
inertia (2 s) and damping ($D/M$ = 1 1/s).

### The notebook

```
Ex12.1_39_edge_optional_basin_stability.ipynb         the exercise: two TODO cells, the answer in the comment above each
Ex12.1_39_edge_optional_basin_stability_light.ipynb   the same notebook with every cell written out
Ex12.1_39_edge_snbs_maps.npz                 the maps the network learns from (tools/exercises/grid39/make_snbs_data.py)
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

#### What it measures (CPU, one thread, seed 88)

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
the network generalises; MP12.1A-39 is a harder test.

*The load model was chosen for Ex12.2 and holds here.* Load buses with inertia made the model
ring at 5 to 45 Hz after every event; first-order load buses do not ring, and moved the
machines' basin stability by at most about 0.07 against the inertial-load version (60 kicks per
bus, different draws).

### Files

| | |
|---|---|
| `course_core.py`, `pinn_core.py` | generated (`tools/pinn/sync_cores.py`) |
| `grid39.py` | generated: the IEEE 39-bus grid and its model, from `tools/exercises/grid39/grid39_src.py` by `sync.py` |
| `problem_12_1_39.py` | this set: the maps, the numbers a bus is described by, the graph |

**The notebook is generated**, both forms from `tools/exercises/ex121_39_edge_optional/build_ex121_39_edge_optional.py`; the
maps by `tools/exercises/grid39/make_snbs_data.py` (about 75 minutes on a laptop CPU, seeded).

### Before this is assigned

The light version has been executed end to end on a CPU. Still to do (C8): a run from a fresh
Colab T4 runtime (with the GPU timings) and a review by someone other than the author. L12.1 does not yet carry this set's story.

### Reference texts

Kundur, *Power System Stability and Control* (1994), ch. 12–13. Menck, Heitzig, Kurths &
Schellnhuber, *How dead ends undermine power grid stability*, Nat. Commun. 5 (2014) 3969.
Nauck et al., *Predicting basin stability of power grids using graph neural networks*, New J.
Phys. 24 (2022) 043041. Liu, *PINN with Python* (2025). **The code, the problem and the
exposition in this exercise set are original to this course.**

### Mini project proposal

The set ends with two mini projects (section 9). Each student chooses one
mini project from the Part 2 sets and solves it individually in one month. The
ground truth is given, built by `tools/miniprojects/ex121_39_edge_optional_truth.py` under policy C10
(`COURSE_POLICIES.md`), with a worked example of each.

| | the problem | the deep learning | the ground truth given | required |
|---|---|---|---|---|
| **MP12.1A-39 · Where to put grid-forming converters** | six machines replaced by converters; make two of them grid-forming - which two? | the set's graph network with grid-forming as an input, ranking all 15 placements | basin-stability maps of all 16 choices, 300 kicks per bus; worked example: 0.63 at the converters with none, 0.72 with the best pair | maps within 0.05; a pair within 0.02 of the best mean at the converters |
| **MP12.1B-39 · The inertia of every machine** | H and D/M of the nine local machines hidden; frequency after three branch trips | an inverse physics-informed network with every H and D/M trainable | RK4 records every 20 ms, noise 1 mHz; worked example: inertias move the records 1.22-9.04 mHz, dampings 0.15-1.30 | inertias within 5 %, dampings within 20 %, total within 2 % |

---

## Ex_12.2_39_edge_optional — Real-Time Stability Screening on the Edge

**Paired with L12.2 · Prediction of Power Grid Stability · Part 2**

*Optional and solved: the 39-bus, edge-inference extension of the six-bus Ex12.2, written on 13 October 2026. The six-bus set is the one the course assigns; this notebook has its own two mini projects (MP12.2A-39 the 39-bus grid with more converters, MP12.2B-39 the loop on the Orin and the Thor).*

N-1 screening in real time on the IEEE 39-bus grid, in **one notebook** (policies C11 and C13).
Every 20 ms the PMUs send a frame; for each of 35 credible faults, would the grid stay in step?
Students compute it by simulation (RK4 on the third-order model) and by a graph-network
surrogate of the dynamics that steps 50 ms, compare the two on verdicts and on time, and run
the screen in real time: a PMU player and a screener talking over UDP - the same two scripts
that run on the Jetson AGX Orin and Thor in the mini projects - on a virtual board in Colab.

### The problem

```
dδ_i/dt = ω_i                                         machines and converters
M_i dω_i/dt = P_i − D_i ω_i − P_e,i
T_i dE_i/dt = E_f,i − E_i + X_i Σ_j B_ij E_j cos(δ_i − δ_j)     machines only
D_i dδ_i/dt = P_i − P_e,i                             load buses
P_e,i = Σ_j E_i E_j B_ij sin(δ_i − δ_j)
a contingency: a solid fault at the first bus of a branch, cleared after 150 ms by tripping it
insecure  ⇔  some machine more than 180° from the centre of inertia within 3 s
```

35 contingencies, 75 operating states (load 0.9 to 1.15, up to 45 % of the local power from
converters): 2 625 cases, 108 insecure (4.1 %). The physics is `grid39.py`, shared with the new
Ex12.1; **every value marked assumed there is assumed here**: a load bus has no inertia and
follows its neighbours within 10 ms, the machines' and converters' damping, the converters'
inertia (2 s). The machines' data (Pai 1989) were checked against the IEEE 39-bus case shipped with ANDES.

### The notebook

```
Ex12.2_39_edge_optional_realtime_screening.ipynb         the exercise: two TODO cells, the answer in the comment above each
Ex12.2_39_edge_optional_realtime_screening_light.ipynb   the same notebook with every cell written out
Ex12.2_39_edge_surrogate.pt                     the trained surrogate (written by the notebook with FULL_TRAINING = True)
board/pmu_player.py                     the PMU player: the Orin in the mini projects
board/screener.py                       the screener: the Thor in the mini projects
board/surrogate.py, board/frames.py     the surrogate and the screen; the two UDP packets
```

| section | what the student does |
|---|---|
| 1 – 3 | the problem, its data, its physics; a fault the base case survives and another state does not |
| 4 | the reference (RK4 on 10 ms; 5 ms changes no verdict) and why its step cannot grow (20 ms: 2 verdicts wrong, separations off by thousands of degrees; 25 ms: 432 of 525 wrong) |
| 5 | the surrogate: a graph network stepping 50 ms after the fault is cleared; data loss over six steps plus the physics residual (TODO 1) |
| 6 | the screen for one frame (TODO 2), against the reference on verdicts and on time |
| 7 | the real-time loop: player and screener over UDP on the virtual board |
| 8 | what the notebook says |
| 9 | the report and its PDF |
| 10 | mini project proposal |

#### What it measures (CPU, one thread, seed 88)

| 15 test states, 525 cases | insecure caught | missed | false alarms | one state's screen |
|---|---|---|---|---|
| RK4 on 10 ms (reference) | 21 of 21 | 0 | 0 | about 130 ms |
| graph-network surrogate, 8 195 weights | 21 of 21 | 0 | 6 | about 70 ms |

On the secure cases the surrogate's largest separation is 3.82° off on average and 21.13° at
worst (the shipped weights after the notebook's 300 further steps). **Trained from scratch
without the physics term** it catches the same cases with the same false alarms, and is 8.34°
off on average and 64.67° at worst, against 4.47° and 25.05° from scratch with it: the physics
term halved the error behind the verdicts without changing them (one training run each). Full
training: 4 000 Adam steps, about two minutes.

Virtual board on the same CPU: the screener's processing 75 ms (median), round trip 85 ms; 376
of 1 500 frames answered, 1 124 dropped because a newer frame had arrived.

**Three things worth knowing.**

*The surrogate is only modestly faster on a CPU, and the notebook says so.* It takes 57 steps
where RK4 takes 285 after the fault, but does more arithmetic in each step. A CPU's time goes
into arithmetic; a GPU's goes into the number of steps (each one a chain of kernel launches),
so the gain should be larger on a GPU. **The GPU numbers have not been measured yet**: the
first Colab T4 run must fill them in, and the Thor's are MP12.2B-39's to measure.

*The fault itself stays with RK4.* A first version let the surrogate run through the fault too,
with a flag on the faulted bus: it learned nothing about the fault (3 frames in 60) and caught
none of the insecure cases. Started from the state at clearing, on the network without the
tripped branch, it caught all of them.

*The load buses decided the model.* With load buses carrying inertia, the model rang at 5 to 45
Hz after every event; no 50 ms step can follow that, and the surrogate's error was as large as
the swings. First-order load buses (no inertia, a 10 ms relaxation, assumed) have real
eigenvalues: the fast part decays without ringing, which is what lets a learned 50 ms step
work - and what holds explicit RK4 below about 14 ms.

### Files

| | |
|---|---|
| `course_core.py`, `pinn_core.py` | generated (`tools/pinn/sync_cores.py`) |
| `grid39.py` | generated: the IEEE 39-bus grid and its model, from `tools/exercises/grid39/grid39_src.py` by `sync.py` |
| `problem_12_2_39.py` | this set: operating states, the fault phase, the reference, the verdict, the surrogate's inputs, the PMU recording |

**The notebook is generated**, both forms from `tools/exercises/ex122_39_edge_optional/build_ex122_39_edge_optional.py`. Edit the
builder and rerun it.

### On the boards (the mini projects)

```
# on the Thor (screener), from this folder:
python3 board/screener.py --reply-to <orin-ip> --weights Ex12.2_39_edge_surrogate.pt --device cuda [--half]
# on the Orin (player):
python3 board/pmu_player.py --to <thor-ip> --recording MP12.2B-39_recording.npz
```

The player sends on a fixed 20 ms clock and measures every round trip on its own clock; the
screener always screens the newest frame and counts the ones it drops. Both write a log
(`player_log.npz`, `screener_log.npz`). Neither has been run on the boards yet.

### Before this is assigned

The light version has been executed end to end on a CPU. Still to do (C8): a run from a fresh
Colab T4 runtime, which also gives the GPU timings the text asks for; a review by someone other
than the author; the scripts run on the Orin and
the Thor. L12.2 does not yet carry this set's story.

### Reference texts

Kundur, *Power System Stability and Control* (1994), ch. 13. Liu, *PINN with Python* (2025).
Raissi, Perdikaris & Karniadakis, J. Comput. Phys. **378** (2019) 686–707. **The code, the
problem and the exposition in this exercise set are original to this course.**

### Mini project proposal

The set ends with two mini projects (section 10). Each student chooses one
mini project from the Part 2 sets and solves it individually in one month. The
ground truth is given, built by `tools/miniprojects/ex122_39_edge_optional_truth.py` under policy C10
(`COURSE_POLICIES.md`), with a worked example of each.

| | the problem | the deep learning | the ground truth given | required |
|---|---|---|---|---|
| **MP12.2A-39 · A bigger grid with more renewable power** | the IEEE 39-bus grid, 50 to 80 % converters with low inertia and a current limit in faults | a retrained surrogate with the converters' behaviour, screening on the boards | 20 states x 35 contingencies x two fault behaviours, RK4 on 5 ms; worked example: 18 % insecure if they keep their power, none if they limit it | every insecure case caught, at most 5 % false alarms, within 20 ms |
| **MP12.2B-39 · The real-time loop on the two boards** | the Orin plays ten minutes of PMU frames, the Thor screens 35 contingencies per frame | the set's surrogate, made fast on the Thor (half precision, compiled graph) | reference verdicts for one frame a second, RK4 on 10 ms; worked example: two contingencies insecure in every frame | verdicts agree; round trip under 20 ms for 99 % of frames |
