# Ex_08.1 — A Heated Plate with a Cooling Channel

**Paired with L8.1 · Stationary Heat · Part 2**

Steady conduction with real units and a real geometry, in **one notebook**
(course policies C11 and C13). An aluminium plate generates heat and is cooled
through an elliptical channel at its centre; the question is how hot it gets,
and where. Students compute the answer three ways — a finite-element reference
on a mesh fitted to the channel, finite differences on a square grid, and a
physics-informed network with the channel wall built in — and compare them on
accuracy and on time.

## The problem

A 100 × 100 mm plate of aluminium 6061 (k = 167 W/m·K) generating 1 MW/m³
evenly, with an elliptical cooling channel of 36 × 22 mm at its centre whose
wall the coolant holds at 40 °C. The outer edges are insulated.

```
k ∇²T + Q = 0        in the plate
T = 40 °C            on the channel wall     (built into the network)
∂T/∂n = 0            on the four outer edges (a weighted loss term)
```

It is solved in scaled units: lengths over the side L, the rise above the
coolant over ΔT = QL²/k = 59.88 K, so that θ_xx + θ_yy + 1 = 0 on the unit
square. Of L8.1's three mechanisms only conduction is solved; the coolant's
convection enters as the fixed wall temperature. The problem has no exact
solution, which is the point of the reference.

## The notebook

```
Ex08.1_pinn_plate_with_channel.ipynb         the exercise: two TODO cells, the answer in the comment above each
Ex08.1_pinn_plate_with_channel_light.ipynb   the same notebook with every cell written out
```

| section | what the student does |
|---|---|
| 1 – 3 | the problem, its data, its physics, the scaled units and the level set |
| 4 | the reference: finite elements on a fitted mesh, refined until it stops moving; then finite differences on a staircase grid, for an agreed 0.10 K |
| 5 | the PINN: the trial function S φ 𝒩 (TODO 1), the residual (TODO 2), the insulated edges as a weighted term, training; the three answers compared, and the energy balance |
| 6 | what the comparison says |
| 7 | the report and its PDF |
| 8 | mini project proposal |

Numbers are printed with at most two decimals (C12). The manufactured
solution, the sweep of the flux weight and the geometry check of the earlier
five-notebook version are left to the lecture.

### What it measures (CPU, seed 88)

| | hottest point | worst error | one solve | training |
|---|---|---|---|---|
| reference, fitted mesh L/200 (38 009 nodes) | 49.56 °C | within 0.004 K of L/400 | about 0.6 s | — |
| finite differences, 641 × 641 (the coarsest for 0.10 K) | 49.60 °C | 6.43 × 10⁻² K | about 3 s | — |
| PINN, 4 × 32, channel built in | 49.56 °C | 1.84 × 10⁻² K | about 60 ms | about 2 min |

The network's energy balance is 99.51 %: the heat it sends through the channel
wall against the heat generated. On the staircase grid finite differences gain
at best a factor of two per halving, where the fitted mesh gains more.

## Files

| | |
|---|---|
| `course_core.py` | shared by the whole course — `set_seed`, `MLP`, `to_tensor`, `check` |
| `pinn_core.py` | the PDE machinery — `grad`, `d2`, samplers, `train_two_stage` |
| `problem.py` | **this** problem — the data, the level set and its samplers, the finite-element reference, the energy balance |

The first two are generated. Edit `tools/pinn/*.py` and run
`python3 tools/pinn/sync_cores.py`; never edit a copy.

**The notebook is generated too**, both forms from one source,
`tools/exercises/ex081/build_ex081.py`, with the cells every one-notebook set
shares in `tools/exercises/part2_notebook.py`. Edit the builder and rerun it
rather than editing a notebook, or the two forms drift apart.

**On Colab nothing needs uploading**: the first code cell fetches the three
modules from the public course repository, afresh on every run.

## Expected runtime

CPU only; a GPU is slower on problems this small. About four minutes in all,
half of it the training of section 5.

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

The set ends with two mini projects (section 8). Each student chooses one
mini project from the Part 2 sets and solves it individually in one month. The
ground truth is given, built by `tools/miniprojects/ex081_truth.py` under policy C10
(`COURSE_POLICIES.md`), with a worked example of each.

| | the problem | the deep learning | the ground truth given | required |
|---|---|---|---|---|
| **MP8.1A · Cooled by a fluid, through several bores** | a 100 × 100 mm aluminium section, 1 MW/m³; two water bores and an oil bore, each a Robin wall | a PINN with a weighted Robin term on every curved wall, sampled by arc length, checked by the heat balance | finite elements, 38,033 nodes; worked example: hottest 74.66 °C, 4238.6 / 4238.6 / 894.9 W/m through the bores | hottest point within 0.2 K, field within 0.5 K, each bore's heat within 2 % |
| **MP8.1B · Where should the bore go?** | one water bore anywhere in the block; a hot 25 × 25 mm component in the corner | a parametric PINN with the bore's position as input, optimised through with autograd | finite elements at 121 positions; worked example: best (57, 50) mm at 117.02 °C, centre 118.77 °C | within 0.5 K at every position; chosen design within 0.3 K of the best; 121 designs in under 0.1 s |
