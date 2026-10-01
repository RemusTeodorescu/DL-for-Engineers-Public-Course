# Ex_10.1 — A Particle in a Battery Cell

**Paired with L10.1 · Ionic Diffusion and Charge Conservation · Part 2**

The single particle model in **one notebook** (course policies C11 and C13).
An LG M50 cell is discharged at 1C; in its negative electrode the lithium has
to diffuse to the surface of a graphite particle before it can leave. Students
compute the concentration in the particle three ways — an exact series, finite
volumes, and a physics-informed network with the start built in — and compare
them on accuracy and on time. Then the same network recovers the diffusivity
from the surface concentration, which is what a battery management system
could see.

## The problem

```
c_t = D_s (c_rr + (2/r) c_r)     in the particle, 0 < t < 3600 s
−D_s c_r = j                     on the surface            (a weighted loss term)
c_r = 0                          at the centre             (follows from the residual, multiplied by r)
c = c_0                          at t = 0                  (built into the network)
```

Graphite particle of radius 5.86 µm, D_s = 3.3 × 10⁻¹⁴ m²/s, starting at
29 866 mol/m³; 5 A for an hour gives a surface flux of 1.54 × 10⁻⁵ mol/(m² s).
In scaled units — radius over R, time over the hour, the concentration lost
over c_ref = j t_end/R = 9476 mol/m³ — it reads u_τ = C (u_rr + (2/r) u_r),
C u_r = 1 on the surface, with C = D_s t_end/R² = 3.46. The mean of u over the
sphere is exactly 3τ: the mass balance, which does not contain the diffusivity.

**The parameter values are those of Chen et al. (2020) as distributed with
PyBaMM, typed in from memory.** PyBaMM is not needed to run the notebook.
Check them against `pybamm.ParameterValues("Chen2020")` before the set is
assigned.

## The notebook

```
Ex10.1_pinn_particle_diffusion.ipynb         the exercise: two TODO cells, the answer in the comment above each
Ex10.1_pinn_particle_diffusion_light.ipynb   the same notebook with every cell written out
```

| section | what the student does |
|---|---|
| 1 – 3 | the problem, its data, its physics: diffusion in a sphere, a flux at the surface, the scales, the mass balance |
| 4 | the exact series as the reference; finite volumes on shells (Crank–Nicolson), for an agreed 30 mol/m³ |
| 5 | the PINN: the trial function S τ 𝒩 (TODO 1), the residual multiplied by r (TODO 2), the surface flux and the mass balance as loss terms; the three compared |
| 6 | the inverse problem: the diffusivity as one more trainable number, found from 37 readings of the surface concentration |
| 7 | what the notebook says |
| 8 | the report and its PDF |
| 9 | mini project proposal |

The SPMe, the thermal model and the control panel of the earlier six-notebook
version, which needed PyBaMM, are left to the lecture and to mini project
MP10.1A.

### What it measures (CPU, seed 88)

| | surface at 3600 s | worst error over the hour | the hour | training |
|---|---|---|---|---|
| exact series | 891 mol/m³ (centre 2260) | — | — | — |
| finite volumes, 40 shells, 144 steps (the coarsest for 30 mol/m³) | 890 mol/m³ | 29 mol/m³ | a few ms | — |
| PINN, 4 × 32, start built in | 899 mol/m³ | 21 mol/m³ | under 1 ms | about 45 s |

The inverse problem returns D_s = 3.29 × 10⁻¹⁴ m²/s against the true
3.30 × 10⁻¹⁴, 0.38 % low, in under a minute, from readings with 50 mol/m³ of
noise and a start at half the true value.

**Three things that cost time and are not obvious.** The surface points must
not include τ = 0: there the trial function has no gradient, the flux
condition cannot be met, and one point in two hundred holds the loss at 0.05.
The **mass balance has to be a loss term**: without it the network is late
with the flux in the first instants, loses 0.007 of a unit and carries that to
the end (78 mol/m³). And the residual is **not** divided by the trainable C
here, unlike Ex_08.2: divided, C runs to infinity, where a uniform particle
fits the surface readings; with the mass balance in the loss the undivided
form gave −1.2 to −1.9 % from starts at half and at twice the true value.

## Files

| | |
|---|---|
| `course_core.py` | shared by the whole course — `set_seed`, `MLP`, `to_tensor`, `check` |
| `pinn_core.py` | the PDE machinery — `grad`, `d2`, `train_two_stage` |
| `problem.py` | **this** problem — the data, the series solution, the samplers, the quadrature of the mass balance, the surface readings |

The first two are generated. Edit `tools/pinn/*.py` and run
`python3 tools/pinn/sync_cores.py`; never edit a copy.

**The notebook is generated too**, both forms from one source,
`tools/exercises/ex101/build_ex101.py`, with the cells every one-notebook set
shares in `tools/exercises/part2_notebook.py`. Edit the builder and rerun it
rather than editing a notebook, or the two forms drift apart.

**On Colab nothing needs uploading**: the first code cell fetches the three
modules from the public course repository, afresh on every run.

## Expected runtime

CPU only; a GPU is slower on problems this small. About two minutes in all,
nearly all of it the two trainings of sections 5 and 6.

## Reference texts

Liu, G.R., *PINN with Python: An Introduction* (2025).
Chen et al., *Development of experimental techniques for parameterization of
multi-scale lithium-ion battery models*, J. Electrochem. Soc. **167** (2020) 080534.
Crank, J., *The Mathematics of Diffusion* (1975).

These are the works to read for the theory. **The code, the problem and the
exposition in this exercise set are original to this course.**

## Before this is assigned

The light version has been executed end to end on a local CPU; the exercise
version stops at its TODO cells. Still to do, as for every set (C8): the
parameter check against PyBaMM named above, a run from a fresh Colab runtime,
and a review by someone other than the author.

## Mini project proposal

The set ends with two mini projects (section 9). Each student chooses one
mini project from the Part 2 sets and solves it individually in one month. The
ground truth is given, built by `tools/miniprojects/ex101_truth.py` under policy C10
(`COURSE_POLICIES.md`), with a worked example of each.

| | the problem | the deep learning | the ground truth given | required |
|---|---|---|---|---|
| **MP10.1A · The full model, where the SPMe stops** | the LG M50 at 1C, 2C, 3C: the Doyle-Fuller-Newman model, electrolyte and particles coupled | coupled PINNs for electrolyte and particles, the terminal voltage from them | PyBaMM's DFN, mesh ×4; worked example: at 3C the DFN gives 2.335 Ah, the SPMe 0.247 Ah | voltage within 20 mV (1C, 2C) and 50 mV (3C); 3C capacity within 2 % |
| **MP10.1B · How much has the cell aged?** | an aged M50: slower diffusion and lost lithium, seen only through discharge voltages | an inverse PINN on the SPMe with the two ageing factors trainable | PyBaMM SPMe discharges of the hidden-aged cell; worked example: 5.015 → 4.455 Ah; the two effects correlate at 0.81-0.87 | both factors within 5 %, capacity within 1 % (case 1) |
