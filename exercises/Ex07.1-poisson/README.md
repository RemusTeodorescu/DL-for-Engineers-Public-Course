# Ex_07.1 — Heat in a stator slot, by finite differences and by a PINN

**Paired with L7.1 · Fundamentals of PINNs · Part 2**

The first exercise of Part 2, in **one notebook** (course policy C11). A
winding in the slot of an electrical machine carries a current and heats up;
the question is how hot it gets inside. Students compute the answer three ways
— by finite differences, and by a physics-informed network whose walls are
imposed by soft and by hard enforcement — and score all three against the
exact solution, which this rectangular slot has as a double sine series.

It establishes the machinery the rest of Part 2 reuses: a residual built by
automatic differentiation, collocation points, a composite loss, soft and hard
enforcement, and the Adam-then-L-BFGS schedule of L6.1.

## The problem

A 10 × 20 mm slot holding 32 round copper wires of 2 mm, in 4 columns of 8,
all carrying the same current, 10 to 45 A (rated: 10 A/mm², 31.4 A). The walls
are held at the cooled iron's 90 °C. In steady state the rise θ = T − 90 °C obeys

```
k_eff ∇²θ + q(θ, I) = 0      in the slot
θ = 0                        on all four walls
q = fill · ρ(T) · J²,   J = I / A_wire,   ρ(T) = ρ20 [1 + α (T − 20)]
```

The winding is averaged into one material — k_eff = 0.70 W/m·K, copper fill
0.503 — as machine designers do. Copper's resistance rises 0.39 % per kelvin,
so q = c0 + c1 θ: a positive feedback that amplifies the rise by
1/(1 − c1/(k_eff λ1)), with λ1 the slot's first eigenvalue, and has no steady
state (thermal runaway) at 158 A. Because c0 and c1 are constants, the equation
has an exact solution, `problem.exact_solution`.

## The notebook

```
Ex07.1_slot.ipynb         the exercise: three TODO cells, the answer in the comment above each
Ex07.1_slot_light.ipynb   the same notebook with every cell written out
```

| section | what the student does |
|---|---|
| 1 | the slot and the heat its winding makes — TODO 1, the heat source |
| 2 | autograd checked exact on the grid's 3,081 nodes, with the slot's own mode; λ1 and the amplification |
| 3 | the exact solution, a sum of the slot's modes |
| 4 | the temperature by finite differences on 41 × 81, the mesh drawn in the cell |
| 5 | the right mesh density: 11 to 321 nodes across, against the exact solution — TODO 2 |
| 6 | the PINN: collocation points, network size against N* and the FDM unknowns, the residual (TODO 3), training with soft and hard enforcement |
| 7 | the three answers compared against the exact solution at six currents |
| 8 | the report and its PDF |
| 9 | mini project proposal |

### What it measures (CPU, seed 88)

| | worst error, six currents | per current | training |
|---|---|---|---|
| finite differences, 41 × 81 | 0.007 K | about 7 ms | — |
| PINN, soft enforcement | 1.5 K | about 1 ms | about 80 s |
| PINN, hard enforcement | 0.018 K | about 1 ms | about 60 s |

The exact hot spot at rated current is 18.740 K above the wall. Hard
enforcement beats soft by almost a hundred times; finite differences are more
accurate still for one current, and the trained network repays its training
after about ten thousand currents.

## Files

| | |
|---|---|
| `course_core.py` | shared by the whole course — `set_seed`, `MLP`, `to_tensor`, `check` |
| `pinn_core.py` | the PDE machinery — `grad`, `d2`, samplers, `train_two_stage` |
| `problem.py` | **this** problem — the slot, the winding and its heat, the finite-difference solver, plots |
| `Ex07.1_slot.png` | the slot figure notebook 00 embeds, drawn by `problem.plot_slot_geometry()` |

The first two are generated. Edit `tools/pinn/*.py` and run
`python3 tools/pinn/sync_cores.py`; never edit a copy.

**The notebook is generated too**, both forms from one source,
`tools/exercises/ex071/build_ex071.py`, with the shared cells in `common.py`
and `_cells.py`. Edit the builder and rerun it rather than editing a notebook,
or the two forms drift apart.

**On Colab nothing needs uploading**: the first code cell fetches the three
modules from the public course repository, afresh on every run.

## Expected runtime

CPU only; a GPU is slower on problems this small. About three minutes in all,
most of it the two trainings of section 6.

## Reference texts

Liu, G.R., *PINN with Python: An Introduction* (2025).
Raissi, Perdikaris & Karniadakis, *Physics-informed neural networks*,
J. Comput. Phys. **378** (2019) 686–707.

These are the works to read for the theory. **The code, the problem and the
exposition in this exercise set are original to this course** — written from the
2019 paper and the PyTorch documentation, and not derived from any publisher's
code listings. Where a symbol matches a textbook's, it is because both follow
the standard notation of the field.

## Before this is assigned

The light version has been executed end to end on a local CPU; the exercise
version stops at its first TODO. Still to do, as for every set (C8): a run from
a fresh Colab runtime, and a review by someone other than the author.

## Mini project proposal

The notebook ends with two mini projects (section 9). Each student
chooses one mini project from the Part 2 sets and solves it individually in
one month. The course provides the ground truth once a project is chosen,
built by a classical solver under policy C10 (`COURSE_POLICIES.md`), so the month
goes into the deep learning.

| | the problem | the deep learning | the ground truth provided |
|---|---|---|---|
| **A · The slot with its insulation** | the 32 wires resolved: 2 mm copper, a 0.04 mm grade 2 enamel coat (IEC 60317), impregnating resin between them; the conductivity drops about 2000-fold at every copper edge | a domain-decomposed PINN, one network per material, with temperature and heat flow matched across every edge | finite differences with several nodes across the enamel, per-cell conductivity, the currents of this set, with a grid-refinement table |
| **B · How old is the insulation?** | aged insulation conducts less where the resin has come away and air has taken its place; only a few sensors in the slot, read at a few currents | an inverse PINN: one network for θ(x, y, I), one for k(x, y), fitted to the physics and the sensors; then the hot spot and the margin to the class limit | slots with a hidden degraded region: sensor readings with positions and noise, and the true fields for scoring; cases of rising difficulty |
