# Ex_09.1 — Liquid Serpentine Cooling Plate

**Paired with L9.1 · Laminar Flow · Part 2**

The coolant in one tube of a liquid cold plate, when the pump starts, in **one
notebook** (course policies C11 and C13). A 250 × 130 × 12 mm aluminium plate
carries a 6 mm copper tube in a serpentine - four legs of 230 mm, three
U-bends of 12 mm radius. A 50/50 water–glycol coolant at 40 °C is pumped
through it at 0.3 L/min (Re = 452, laminar). The notebook solves the flow in
the fully developed part of one leg, at the section A-A, as it starts from
rest.

## The problem

```
ρ u_t = G + μ (u_rr + u_r / r)        Newton's law along the tube: the pressure's push against the drag
u = 0 on the wall                      no slip                          (built into the PINN)
∂u/∂r = 0 on the axis                  symmetry                         (built in by the input s = r²/R²)
u = 0 at t = 0                         the coolant at rest              (built into the PINN)
```

Section 3 builds this from Navier–Stokes step by step: a parcel; round and
straight, so no swirl; far from the bends, so fully developed; the volume
kept, so no radial velocity - and then the convective acceleration is zero,
not neglected. The steady limit is **Hagen–Poiseuille's parabola** and
**Darcy's law** with the permeability $R^2/8$ of a round tube (the friction
factor 64/Re); the start-up is the exact Bessel series of which the parabola
is the limit. The entrance length after a bend, 0.05 Re D = 136 mm, says
which part of each leg this applies to.

**All sizes and the coolant's properties are typical values typed from
memory**, and so are the heat transfer and friction correlations used in the
mini projects. Check them before quoting a result.

## The notebook

```
Ex09.1_cooling_plate.ipynb         the exercise: two TODO cells, the answer in the comment above each
Ex09.1_cooling_plate_light.ipynb   the same notebook with every cell written out
```

| section | what the student does |
|---|---|
| 1 – 3 | the problem; the plate and its serpentine drawn in 3-D, the section A-A; the physics step by step, the conditions, Hagen–Poiseuille, Darcy, the exact start-up |
| 4 | the exact solution as the reference; finite differences across the tube for an agreed 1e-03, with the grid and the PINN's points drawn |
| 5 | a PINN with the wall and the start built in (TODO 1 the trial function, TODO 2 the residual); the comparison, the section in colour; both methods timed on one device, and how the costs grow in 1, 2 and 3 dimensions |
| 6 | the inverse problem: the coolant's viscosity from a flow meter's first second, four ways |
| 7 | what the notebook says |
| 8 | the report and its PDF |
| 9 | mini project proposal |

### What it measures

| | worst error | time |
|---|---|---|
| finite differences, 21 nodes × 80 Crank–Nicolson steps | 9.1 × 10⁻⁴ | about 1 ms on a CPU |
| PINN, 4 × 32, wall and start built in | 6.2 × 10⁻⁴ | about 40 s of training on a CPU; 0.3 – 0.4 ms per answer |

Online, inference only, at the same accuracy and on the same device, section
5 times one FDM solve against one evaluation of the trained PINN; the
training is reported apart, as the offline cost, and never set against a
solve. The numbers are those of the runtime you run it on. A scaling test of
the same equation in a box shows the grid's cost growing with the dimension
where the network's does not. The PINN's case is many queries and many
dimensions, not one small solve.

| the viscosity from the first second (true 3.50 mPa·s) | found | off by |
|---|---|---|
| Darcy's law on the last reading, as if steady | 4.04 mPa·s | +15.5 % |
| least squares on the exact series | 3.55 mPa·s | +1.4 % |
| least squares with FDM as the model, 5 runs | 3.53 mPa·s | +1.0 % |
| the PINN | 3.56 mPa·s | +1.6 % |

An inverse problem is not a PINN's privilege: any model that predicts the
readings can be fitted to them.

## Files

| | |
|---|---|
| `course_core.py` | shared by the whole course — `set_seed`, `MLP`, `to_tensor`, `check` |
| `pinn_core.py` | the Part 2 helpers — `grad`, `train_two_stage`, `describe` |
| `pipe.py` | **this** problem — the data, the exact start-up, finite differences in NumPy and in PyTorch, the flow meter, the scaling test, and the drawings of the plate, the methods and the section |

The first two are generated. Edit `tools/pinn/*.py` and run
`python3 tools/pinn/sync_cores.py`; never edit a copy.

**The notebook is generated too**, both forms from one source,
`tools/exercises/ex091/build_ex091.py`. Edit the builder and rerun it rather
than editing a notebook. On Colab nothing needs uploading: the first code cell
fetches the three modules from the public course repository; choose a GPU
runtime for the timing of section 5.

## Expected runtime

About two minutes on a CPU, most of it the PINN's two trainings.

## Reference texts

White, *Fluid Mechanics*, ch. 6 (pipe flow), and *Viscous Fluid Flow*, ch. 3
(the start-up); Liu, *PINN with Python: An Introduction* (2025).

These are the works to read for the theory. **The code, the problem and the
exposition in this exercise set are original to this course.**

## Before this is assigned

The light version has been executed end to end on a local CPU and on a local
GPU; the exercise version stops at its TODO cells. Still to do: a run from a
fresh Colab runtime, a review by someone other than the author, and the
property values checked against a supplier's data.

It replaced, on 5 October 2026, the notebook on the coupled Burgers equations
(a front carried by a flow), whose ideas - the Reynolds number as the width of
a front, the PINN's points against a thin feature - are on L9.1's Advanced
Topics slide.

## Mini project proposal

The set ends with two mini projects (section 9 of the notebook). Each student chooses one
mini project from the Part 2 sets and solves it individually in one month. The
ground truth is given, built by `tools/miniprojects/ex091_truth.py` under policy C10
(`COURSE_POLICIES.md`), with a worked example of each.

| | the problem | the deep learning | the ground truth given | required |
|---|---|---|---|---|
| **MP9.1A · One bend of the serpentine, in 3-D** | one 12 mm U-bend between a 20 mm and a 110 mm leg at 0.3 L/min; the velocity in 3-D, the recovery after the bend, the bend's loss coefficient | a marching (parabolised) solver written by the student, then a 3-D PINN scored against it | the two limits the solver must reach: the fully developed bend (Dean, checked against Dean's series) and the notebook's parabola; no 3-D reference | three checks on the solver (parabola 1e-04, long bend 1 %, flow rate 1e-06), PINN within 2 % of it, K with a grid study |
| **MP9.1B · The pump's speed reference** | the flow to send to the pump, over 30 minutes, for a staircase of module power; the module at or below 80 °C, the least pumping; case 2 with a 700 W step that only turbulent flow can cool | a network for the flow reference in time, bounded by its last layer, with the plate's energy balance and the limit in the loss | direct transcription and SLSQP on the same model; worked example: the flow raised before the step, the reactive controller 6.5 K over | module at most 80.2 °C, pumping within 5 %, the flow raised before the step |
