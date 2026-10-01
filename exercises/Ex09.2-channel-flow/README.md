# Ex_09.2 — The Mean Flow Past a Tube

**Paired with L9.2 · Turbulent Flow · Part 2**

A turbulent mean flow with an eddy-viscosity closure, in **one notebook**
(course policies C11 and C13). Water runs through a duct and past a tube that
crosses it; the pressure drop and the drag on the tube are wanted. Students
compute them three ways — a finite-element reference on a mesh fitted to the
tube, finite volumes on a staircase grid, and a physics-informed network with
the stream function — and compare them on accuracy and on time. Then the same
network finds the eddy viscosity from twelve velocity probes.

## The problem

A duct 100 mm high and 400 mm long carries water at a mean speed of 0.25 m/s
past a 40 mm tube on its centreline, 120 mm from the inlet. The Reynolds number
is 25 000, so the flow is turbulent and its **mean** is solved for:

```
(u·∇)u = −∇p/ρ + ν_eff ∇²u,   ∇·u = 0,   ν_eff = ν + ν_t
u = v = 0 on the walls, u = 6y(1−y) at the inlet      (built into the network)
u = v = 0 on the tube; p = 0, v = 0 at the outlet     (weighted loss terms)
```

**The closure is one number**, ν_t = 0.02 UH = 5.0 × 10⁻⁴ m²/s, five hundred
times the viscosity of water. It is a choice, made so that the modelled mean
flow is steady and smooth; it is not a measured or a calibrated value, and a
real eddy viscosity varies in space. The notebook says so, and its last
bullet and mini project MP9.2B take it up. In scaled units — lengths over H,
speeds over U, pressure over ρU² = 62.38 Pa — the duct is 4 × 1, the tube a
circle of radius 0.2 at (1.2, 0.5), and ν_eff/(UH) = 0.02.

## The notebook

```
Ex09.2_pinn_flow_past_tube.ipynb         the exercise: two TODO cells, the answer in the comment above each
Ex09.2_pinn_flow_past_tube_light.ipynb   the same notebook with every cell written out
```

| section | what the student does |
|---|---|
| 1 – 3 | the problem, its data, its physics: why the mean, the closure problem, the eddy viscosity, the scaled units |
| 4 | the reference: quadratic finite elements on the fitted mesh, Newton's method, refined twice; then finite volumes on the staircase grid, for an agreed 5 % on the pressure drop and the drag |
| 5 | the PINN: the trial stream function with the walls and the inflow built in (TODO 1), the two momentum residuals (TODO 2), training; the three answers compared |
| 6 | the inverse problem: the eddy viscosity as one more trainable number, found from twelve probes |
| 7 | what the notebook says |
| 8 | the report and its PDF |
| 9 | mini project proposal |

Numbers are printed with at most two decimals (C12). The five obstacle
shapes, the control panel and the shape comparison of the earlier
five-notebook version are gone; the tube is the only obstacle.

### What it measures (CPU, seed 88)

| | pressure drop | drag on the tube | worst velocity error | one solution | training |
|---|---|---|---|---|---|
| reference, fitted mesh H/40, 58 772 unknowns | 218.25 Pa | 11.40 N/m (coefficient 9.14) | within 0.004 U of H/80 | about 6 s | — |
| finite volumes, 320 × 80 cells (the coarsest for 5 %) | 212.01 Pa (−2.86 %) | 10.94 N/m (−4.03 %) | 0.26 U, beside the staircase | about 16 s | — |
| PINN, 4 × 32, stream function, 2000 points | 217.22 Pa (−0.48 %) | 11.28 N/m (−1.00 %) | 0.02 U | 0.01 s | about 2.5 min |

The inverse problem returns ν_eff = 5.14 × 10⁻⁴ m²/s against the true
5.00 × 10⁻⁴, 2.70 % high, in about two minutes, from twelve readings of u with
noise of 1 % of the mean speed and a start twice too high; with it the
pressure drop is 2.11 % and the drag 0.76 % from the reference.

**Two things that cost time and are not obvious.** The inverse problem's
residuals are **divided by the trainable viscosity**: without that, on some
starts ν_eff runs to zero (it did, to 6.5 × 10⁻⁸ m²/s, with a loss that looked
healthy), because an almost inviscid flow fits twelve probes as well. With the
division three seeds gave +2.1 to +3.3 %. And the reference's velocity must be
evaluated with its own quadratic shape functions: linear interpolation between
its nodes is wrong by 0.02 U beside the walls, which is as large as the
network's error.

## Files

| | |
|---|---|
| `course_core.py` | shared by the whole course — `set_seed`, `MLP`, `to_tensor`, `check` |
| `pinn_core.py` | the PDE machinery — `grad`, `d2`, `train_two_stage` |
| `problem.py` | **this** problem — the data, the samplers, the finite-element reference (`reference`, `reference_at`), the finite-volume solver (`finite_volumes`), the drag and pressure drop of a network, the probe readings |

The first two are generated. Edit `tools/pinn/*.py` and run
`python3 tools/pinn/sync_cores.py`; never edit a copy.

**The notebook is generated too**, both forms from one source,
`tools/exercises/ex092/build_ex092.py`, with the cells every one-notebook set
shares in `tools/exercises/part2_notebook.py`. Edit the builder and rerun it
rather than editing a notebook, or the two forms drift apart.

**On Colab nothing needs uploading**: the first code cell fetches the three
modules from the public course repository, afresh on every run.

See also `MiniProject_Turbulence.md`, the earlier outline of an L13 project
on learned closures; mini project MP9.2B below is its first track, with a
ground truth.

## Expected runtime

CPU only; a GPU is slower on problems this small. About six minutes in all,
most of it the two trainings of sections 5 and 6.

## Reference texts

Pope, S.B., *Turbulent Flows* (2000), ch. 4, 7, 10.
Liu, G.R., *PINN with Python: An Introduction* (2025).
Raissi, Perdikaris & Karniadakis, *Physics-informed neural networks*,
J. Comput. Phys. **378** (2019) 686–707.

These are the works to read for the theory. **The code, the problem and the
exposition in this exercise set are original to this course.**

## Before this is assigned

The light version has been executed end to end on a local CPU; the exercise
version stops at its TODO cells. Still to do, as for every set (C8): a run from
a fresh Colab runtime, and a review by someone other than the author.

## Mini project proposal

The set ends with two mini projects (section 9). Each student chooses one
mini project from the Part 2 sets and solves it individually in one month. The
ground truth is given, built by `tools/miniprojects/ex092_truth.py` under policy C10
(`COURSE_POLICIES.md`), with a worked example of each.

| | the problem | the deep learning | the ground truth given | required |
|---|---|---|---|---|
| **MP9.2A · The wake starts to shed** | the Schäfer & Turek cylinder at Re = 100: an unsteady, shedding wake | a PINN in (x, y, t) with the stream function, over several shedding periods | the published benchmark values (St 0.30, cD max 3.23, cL max 1.00); the course solver's fields, first order, St 0.272 at D/40 | St within 3 %, drag within 5 %, lift within 10 % |
| **MP9.2B · The eddy viscosity from measurements** | this set's channel; the eddy viscosity hidden; u at 12 probes, noise 0.01 | an inverse PINN: the flow and ν_t(x, y), fitted to the mean-flow equations and the probes | the steady flow by the course solver, 320 × 80; worked example: ν_t adds 8 % to the drag and moves the probes 0.025-0.70 | ν_t peak within 25 % and 0.2 units; probes within 0.02 |
