# Ex_12.2 (new) — Real-Time Stability Screening on the Edge

**Paired with L12.2 · Prediction of Power Grid Stability · Part 2**

*Written on 13 October 2026 beside the six-bus Ex12.2 (`Ex12.2-power-grid-stability-prediction`),
which stays as it is. The folder and the set key carry an `N` until the lecturer chooses which
of the two is assigned.*

N-1 screening in real time on the IEEE 39-bus grid, in **one notebook** (policies C11 and C13).
Every 20 ms the PMUs send a frame; for each of 35 credible faults, would the grid stay in step?
Students compute it by simulation (RK4 on the third-order model) and by a graph-network
surrogate of the dynamics that steps 50 ms, compare the two on verdicts and on time, and run
the screen in real time: a PMU player and a screener talking over UDP - the same two scripts
that run on the Jetson AGX Orin and Thor in the mini projects - on a virtual board in Colab.

## The problem

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
inertia (2 s). The machines' data (Pai 1989) are typed from memory and must be checked.

## The notebook

```
Ex12.2_realtime_screening.ipynb         the exercise: two TODO cells, the answer in the comment above each
Ex12.2_realtime_screening_light.ipynb   the same notebook with every cell written out
Ex12.2_surrogate.pt                     the trained surrogate (written by the notebook with FULL_TRAINING = True)
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

### What it measures (CPU, one thread, seed 88)

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
first Colab T4 run must fill them in, and the Thor's are MP12.2A's to measure.

*The fault itself stays with RK4.* A first version let the surrogate run through the fault too,
with a flag on the faulted bus: it learned nothing about the fault (3 frames in 60) and caught
none of the insecure cases. Started from the state at clearing, on the network without the
tripped branch, it caught all of them.

*The load buses decided the model.* With load buses carrying inertia, the model rang at 5 to 45
Hz after every event; no 50 ms step can follow that, and the surrogate's error was as large as
the swings. First-order load buses (no inertia, a 10 ms relaxation, assumed) have real
eigenvalues: the fast part decays without ringing, which is what lets a learned 50 ms step
work - and what holds explicit RK4 below about 14 ms.

## Files

| | |
|---|---|
| `course_core.py`, `pinn_core.py` | generated (`tools/pinn/sync_cores.py`) |
| `grid39.py` | generated: the IEEE 39-bus grid and its model, from `tools/exercises/grid39/grid39_src.py` by `sync.py` |
| `problem.py` | this set: operating states, the fault phase, the reference, the verdict, the surrogate's inputs, the PMU recording |

**The notebook is generated**, both forms from `tools/exercises/ex122n/build_ex122n.py`. Edit the
builder and rerun it.

## On the boards (the mini projects)

```
# on the Thor (screener), from this folder:
python3 board/screener.py --reply-to <orin-ip> --weights Ex12.2_surrogate.pt --device cuda [--half]
# on the Orin (player):
python3 board/pmu_player.py --to <thor-ip> --recording MP12.2A_recording.npz
```

The player sends on a fixed 20 ms clock and measures every round trip on its own clock; the
screener always screens the newest frame and counts the ones it drops. Both write a log
(`player_log.npz`, `screener_log.npz`). Neither has been run on the boards yet.

## Before this is assigned

The light version has been executed end to end on a CPU. Still to do (C8): a run from a fresh
Colab T4 runtime, which also gives the GPU timings the text asks for; a review by someone other
than the author; the machine data checked against Pai (1989); the scripts run on the Orin and
the Thor. L12.2 does not yet carry this set's story.

## Reference texts

Kundur, *Power System Stability and Control* (1994), ch. 13. Liu, *PINN with Python* (2025).
Raissi, Perdikaris & Karniadakis, J. Comput. Phys. **378** (2019) 686–707. **The code, the
problem and the exposition in this exercise set are original to this course.**

## Mini project proposal

The set ends with two mini projects (section 10). Each student chooses one
mini project from the Part 2 sets and solves it individually in one month. The
ground truth is given, built by `tools/miniprojects/ex122n_truth.py` under policy C10
(`COURSE_POLICIES.md`), with a worked example of each.

| | the problem | the deep learning | the ground truth given | required |
|---|---|---|---|---|
| **MP12.2A · The real-time loop on the two boards** | the Orin plays ten minutes of PMU frames, the Thor screens 35 contingencies per frame | the set's surrogate, made fast on the Thor (half precision, compiled graph) | reference verdicts for one frame a second, RK4 on 10 ms; worked example: two contingencies insecure in every frame | verdicts agree; round trip under 20 ms for 99 % of frames |
| **MP12.2B · More renewable power, and what converters do in a fault** | 50 to 80 % converters with low inertia and a current limit in faults | a retrained surrogate with the converters' behaviour, screening on the boards | 20 states x 35 contingencies x two fault behaviours, RK4 on 5 ms; worked example: 18 % insecure if they keep their power, none if they limit it | every insecure case caught, at most 5 % false alarms, within 20 ms |

