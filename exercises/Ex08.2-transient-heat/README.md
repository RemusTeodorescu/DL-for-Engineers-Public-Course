# Ex_08.2 — The Plate Switched On

**Paired with L8.2 · Transient Heat · Part 2**

The plate of Ex_08.1 with a clock, in **one notebook** (course policies C11
and C13). The plate sits at the coolant's temperature and its power is
switched on; the question is how fast it warms and when it is steady. Students
compute the answer three ways — a finite-element reference, finite differences
on a square grid, and a physics-informed network with the start and the
channel wall built in — and compare them on accuracy and on time. Then the
same network recovers the plate's diffusivity from four thermocouples.

## The problem

The 100 × 100 mm aluminium plate of Ex_08.1 (k = 167 W/m·K, ρc_p = 2.43
MJ/m³·K, so α = 6.87 × 10⁻⁵ m²/s), its elliptical cooling channel held at
40 °C, its outer edges insulated. At t = 0 the plate is at 40 °C and its heat
generation, 1 MW/m³, is switched on. The window is 80 s.

```
ρc_p T_t = k ∇²T + Q     in the plate, t > 0
T = 40 °C                on the channel wall, and everywhere at t = 0   (both built into the network)
∂T/∂n = 0                on the four outer edges                        (a weighted loss term)
```

In scaled units — lengths over L, the rise over ΔT = QL²/k = 59.88 K, time
over the window — it reads θ_τ = C (θ_xx + θ_yy + 1) with C = α·80 s/L² = 0.55.

## The notebook

```
Ex08.2_pinn_plate_switched_on.ipynb         the exercise: two TODO cells, the answer in the comment above each
Ex08.2_pinn_plate_switched_on_light.ipynb   the same notebook with every cell written out
```

| section | what the student does |
|---|---|
| 1 – 3 | the problem, its data, its physics: storage, the diffusivity, the time L²/α, the scaled units |
| 4 | what finite elements and finite differences each ask of the equation, with both meshes drawn; the reference: finite elements on the fitted mesh with Crank–Nicolson, refined in space and time together; then finite differences on the staircase grid, for an agreed 0.20 K |
| 5 | the PINN: the trial function S τ φ 𝒩 (TODO 1), the residual (TODO 2), training; the three answers compared over the window |
| 6 | the inverse problem: the diffusivity as one more trainable number, found from four thermocouples |
| 7 | what the notebook says |
| 8 | the report and its PDF |
| 9 | mini project proposal |

Numbers are printed with at most two decimals (C12). The square-plate
benchmark, the soft start against the hard start, and the choice of sensor
window of the earlier six-notebook version are left to the lecture.

### What it measures (CPU, seed 88)

| | unknowns | hottest point at 80 s | worst error over the window | time to the answer |
|---|---|---|---|---|
| reference, fitted mesh L/200, 320 steps | 38 009 | 49.42 °C | within 0.01 K of L/100 | 2 to 3 s |
| finite differences, 321 × 321, 320 steps (the coarsest for 0.20 K) | 96 664 | 49.49 °C | 1.25 × 10⁻¹ K | 5 to 15 s |
| PINN, 4 × 32, start and channel built in | 3 329 | 49.49 °C | 1.13 × 10⁻¹ K | about 2 min, all of it training; the trained network then gives the 41 instants in about 0.4 s |

"Time to the answer" counts everything from the problem to the field over
the window, so the network's training is in it. The network is neither more
exact than finite differences in any way that matters (0.11 against 0.13 K,
both inside the agreed 0.20 K) nor faster: it takes twenty to fifty times
longer. The notebook says so in as many words.

The hottest point reaches 9.42 K above the coolant at 80 s (9.56 K when
steady) and half of that after about 13 s. The inverse problem returns
α = 6.92 × 10⁻⁵ m²/s against the true 6.87 × 10⁻⁵, 0.72 % high, in about two
and a half minutes, from readings with 0.05 K of noise and a start 45 % low.

## Files

| | |
|---|---|
| `course_core.py` | shared by the whole course — `set_seed`, `MLP`, `to_tensor`, `check` |
| `pinn_core.py` | the PDE machinery — `grad`, `d2`, samplers, `train_two_stage` |
| `problem.py` | **this** problem — the data, the level set, the space-time samplers, the finite-element reference in time, the thermocouple readings |

The first two are generated. Edit `tools/pinn/*.py` and run
`python3 tools/pinn/sync_cores.py`; never edit a copy.

**The notebook is generated too**, both forms from one source,
`tools/exercises/ex082/build_ex082.py`, with the cells every one-notebook set
shares in `tools/exercises/part2_notebook.py`. Edit the builder and rerun it
rather than editing a notebook, or the two forms drift apart.

**On Colab nothing needs uploading**: the first code cell fetches the three
modules from the public course repository, afresh on every run.

## Expected runtime

CPU only; a GPU is slower on problems this small. About eight minutes in all,
most of it the two trainings of sections 5 and 6.

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

The set ends with two mini projects (section 9). Each student chooses one
mini project from the Part 2 sets and solves it individually in one month. The
ground truth is given, built by `tools/miniprojects/ex082_truth.py` under policy C10
(`COURSE_POLICIES.md`), with a worked example of each.

| | the problem | the deep learning | the ground truth given | required |
|---|---|---|---|---|
| **MP8.2A · The cold plate in three dimensions, switched on** | the cold plate of MP8.1B, cold at the start, its 150 W chip switched on; 0 to 20 s | a PINN for T(x, y, z, t), the start built in, the flux and Robin walls as weighted loss terms | an exact series in space and time; worked example: half of the chip's rise after 0.48 s, 90 % after 4.51 s | hottest point within 0.5 K at every instant, field within 1 K, time to 90 % within 0.3 s |
| **MP8.2B · Finding a hidden flaw by thermography** | a flash on a 4 mm laminate; a delamination at 1 or 2 mm depth holds heat back; an infrared camera for 10 s | an inverse PINN: θ(x, z, t) and α(x, z), fitted to the heat equation and the camera frames | finite volumes, 0.1 × 0.025 mm; three cases; worked example: 1.14 K warmer over the flaw at 2.86 s | flaw edges within 1 mm, depth within 0.2 mm, surface within 0.05 K |
