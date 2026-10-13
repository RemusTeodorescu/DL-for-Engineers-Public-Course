# Ex_12.2 — Screening Contingencies

**Paired with L12.2 · Prediction of Power Grid Stability · Part 2**

N-1 screening on the six-bus grid of Ex_12.1, in **one notebook** (course
policies C11 and C13). A line faults and the protection needs 140 ms to clear
it: does the local plant stay in step? The answer is the critical clearing
time, and finding it takes a simulation in time for every line and every
operating point. Students compute it by simulation and by a graph network
trained on simulations, compare the two on accuracy and on time, and then ask
the network about a line it has never seen out.

## The problem

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

## The notebooks

```
Ex12.2_gnn_contingency_screening.ipynb         the exercise: two TODO cells, the answer in the comment above each
Ex12.2_gnn_contingency_screening_light.ipynb   the same notebook with every cell written out
Ex12.2_on_the_board.ipynb                      the latency notebook: on Colab, and on the Jetson Thor in L13
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
point):** the course's simulation 219.7 ms (0.8 s to find), ANDES with the
classical machine 209.0 ms (18 s), ANDES with GENROU, ESST1A and TGOV1
162.6 ms (21 s). (`andes_compare.py` is from Shahariya Rasheed's ANDES comparison, whose separate supplement notebook was removed on 13 October 2026 when its essentials moved into section 5.)

### What it measures (CPU, seed 88)

| 216 held-out cases | mean error | worst | insecure missed | false alarms | one case | training |
|---|---|---|---|---|---|---|
| reference, step 2 ms | — | — | — | — | 1.7 ms | — |
| simulation, step 5 ms (the coarsest for 5 ms) | 1.66 ms | 4.88 ms | 0 of 26 | 5 | 0.6 ms | — |
| graph network, 6721 weights | 0.57 ms | 4.31 ms | 0 of 26 | 0 | 3 µs | 11 s |

Over all 1080 cases the 5 ms step changes 19 verdicts, the 10 ms step 53.

| line 0 out, 36 held-out cases | mean error | insecure missed |
|---|---|---|
| network that has seen line 0 out | 0.69 ms | 0 of 11 |
| network trained without it | 57.21 ms, all on the unsafe side | 11 of 11 |

With the threshold raised from 140 to 185 ms the second network misses none
and raises no false alarm: its error is a nearly constant offset, so it still
ranks the cases correctly. The 185 ms was found with the truth in hand.

**Two things worth knowing.**

*The labels are simulated for every case at once.* `pb.build_cases` runs the
same Runge–Kutta integration and the same bisection as `pb.label_case` with a
leading case axis, and returns the same 1080 labels to the last bit in 1.7 s
instead of three minutes. The per-case times in the table are that batched
simulation's; one case on its own, by `pb.label_case`, takes about 0.2 s.

*The lecture's four-way table has one row measured here.* L12.2 says of a
graph network on an unseen topology that structure helps. This notebook
measures that row and finds 57 ms of error on the unsafe side; it does not
train the dense network, so it cannot say whether that is better or worse
than a dense one would do. The physics-informed graph network is not built.

## Files

| | |
|---|---|
| `course_core.py` | shared by the whole course — `set_seed`, `MLP`, `to_tensor`, `check` |
| `pinn_core.py` | the Part 2 helpers — `train_two_stage` |
| `problem.py` | **this** problem — the grid, the machines, the fault model, the contingencies, the operating points, the labels (`label_case`, `build_cases`), the features and the adjacency |
| `andes_compare.py` | the six-bus grid in ANDES, used by section 5 |

The first two are generated. Edit `tools/pinn/*.py` and run
`python3 tools/pinn/sync_cores.py`; never edit a copy.

**The study notebook is generated too**, both forms from one source,
`tools/exercises/ex122/build_ex122.py`. Edit the builder and rerun it rather
than editing a notebook. The latency notebook is edited
by hand; neither has a light form.

## Expected runtime

About two minutes on a CPU: two trainings of about ten seconds, four runs of
the simulation, and about 40 s of ANDES (installed by the notebook on Colab).

## Reference texts

Liu, G.R., *PINN with Python: An Introduction* (2025).
Raissi, Perdikaris & Karniadakis, *Physics-informed neural networks*,
J. Comput. Phys. **378** (2019) 686–707.
Kundur, *Power System Stability and Control* (1994), ch. 2 and 13.

These are the works to read for the theory. **The code, the problem and the
exposition in this exercise set are original to this course.**

## Before this is assigned

The light version of the study notebook has been executed end to end on a
local CPU; the exercise version stops at its TODO cells. Still to do: a run
from a fresh Colab runtime, a review by someone other than the author, and a
run of the latency notebook on the Thor.

## Mini project proposal

The set ends with two mini projects (section 10). Each student chooses one
mini project from the Part 2 sets and solves it individually in one month. The
ground truth is given, built by `tools/miniprojects/ex122_truth.py` under policy C10
(`COURSE_POLICIES.md`), with a worked example of each.

| | the problem | the deep learning | the ground truth given | required |
|---|---|---|---|---|
| **MP12.2A · A bigger grid with more renewable power** | the IEEE 39-bus grid, 50 to 80 % converters with low inertia and a current limit in faults | a retrained surrogate with the converters' behaviour, screening on the boards | 20 states x 35 contingencies x two fault behaviours, RK4 on 5 ms; worked example: 18 % insecure if they keep their power, none if they limit it | every insecure case caught, at most 5 % false alarms, within 20 ms |
| **MP12.2B · The real-time loop on the two boards** | the Orin plays ten minutes of PMU frames, the Thor screens 35 contingencies per frame | the optional notebook's surrogate, made fast on the Thor (half precision, compiled graph) | reference verdicts for one frame a second, RK4 on 10 ms; worked example: two contingencies insecure in every frame | verdicts agree; round trip under 20 ms for 99 % of frames |
