# Ex_07.1 — Fundamentals of PINNs: a stator slot

**Paired with L7.1 · Fundamentals of PINNs, and L7.2 · Fundamental PDEs · Part 2**

The first exercise of Part 2. A winding in the slot of an electrical machine
carries a current and heats up; the question is how hot it gets inside, where
nobody can put a thermometer everywhere. Students compute the answer three
ways — by finite differences, and by a physics-informed network whose walls are
imposed first softly and then hard — and measure which is more accurate and
which is faster. On the way they find that hot copper turns the slot's Poisson
equation into a Helmholtz equation.

It establishes the machinery the rest of Part 2 reuses: a residual built by
automatic differentiation, a composite loss, hard and soft conditions, and the
Adam-then-L-BFGS schedule of L6.1.

## Goals

By the end you can

1. compute the heat a winding makes from its current, $RI^2$, with copper's
   resistance rising as it warms;
2. solve the slot's temperature by finite differences, and show that the
   answer has converged;
3. train a PINN with its walls imposed softly and hard, and score both against
   the finite-difference answer;
4. compare the methods on accuracy and computing time, and say when a PINN is
   worth its training;
5. explain why hot copper turns the slot's Poisson equation into a Helmholtz
   equation, find its eigenvalue with a PINN, and say how a designer keeps its
   effect small.

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
0.503 — as machine designers do; modelling every wire would make the
conductivity jump 2000-fold at every wire edge. Copper's resistance rises
0.39 % per kelvin, so the heat depends on θ: that is a positive feedback, the
rise grows about 5 % faster than I² at 45 A, and it is what turns the equation
into a Helmholtz equation in notebook 03. The source is the real one; the
ground truth is finite differences, made converged in notebook 01.

## The notebooks

Run in order; later notebooks load results saved by earlier ones.

```
Ex07.1_00_environment_check.ipynb    the slot, its winding and heat, the tools
Ex07.1_01_fdm_ground_truth.ipynb     the ground truth by finite differences, and proof it has converged
Ex07.1_02_pinn_soft_and_hard.ipynb   a PINN with soft and with hard walls, against FDM on accuracy and time
Ex07.1_03_helmholtz.ipynb            hot copper, the Helmholtz equation, and the slot's eigenvalue
Ex07.1_04_report.ipynb               the results, the answers, and the report as a PDF
```

Notebooks 01 to 03 each come in two forms: the exercise, with a few lines to
write and the answer in the comment above them, and `_light`, written out.

### What the set measures (CPU, seed 88)

| | |
|---|---|
| hot spot at rated current | 18.74 K above the wall, at the centre |
| rise at 45 A, against I² scaling | +4.9 % |
| PINN, soft walls: worst error / on the walls | 1.45 K / 0.17 K |
| PINN, hard walls: worst error | 0.020 K |
| smallest good network | 16 × 3, 625 parameters, 0.031 K |
| the slot's first eigenvalue, network against formula | 3.0842 against 3.0843 (scaled) |
| thermal runaway / class F reached | 158 A / 56 A |

The verdict is finite differences on accuracy and on a single solve; the
trained network is several times faster per current, but its training only
pays after about ten thousand currents. Notebook 02 prints the exact
figures for the machine it runs on.

## Files

| | |
|---|---|
| `course_core.py` | shared by the whole course — `set_seed`, `MLP`, `to_tensor`, `check` |
| `pinn_core.py` | the PDE machinery — `grad`, `d2`, samplers, `train_two_stage` |
| `problem.py` | **this** problem — the slot, the winding and its heat, the finite-difference solver, plots |
| `Ex07.1_slot.png` | the slot figure notebook 00 embeds, drawn by `problem.plot_slot_geometry()` |

The first two are generated. Edit `tools/pinn/*.py` and run
`python3 tools/pinn/sync_cores.py`; never edit a copy.

**The notebooks are generated too**, both forms of each from one source in
`tools/exercises/ex071/` (`build_nb01.py` … `build_nb04.py`, with the shared
cells in `common.py` and `_cells.py`). Edit the builder and rerun it rather
than editing a notebook, or the two forms drift apart. Notebook 00 has no
builder and is edited in place.

**On Colab nothing needs uploading**: each notebook's first code cell fetches
the three modules from the public course repository, afresh on every run.

## Expected runtime

CPU only; a GPU is slower on problems this small. Notebook 00 runs in seconds,
01 in about half a minute, 03 in about a minute, and 02 in about four — it
trains four networks.

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

Every notebook has been executed end to end on a local CPU, both forms: the
light versions run to the end, the exercise versions stop at their first TODO.
Still to do, as for every set (C8): a run from a fresh Colab runtime, and a
review by someone other than the author.

## Results between notebooks on Colab

Later notebooks read `.npz` files that earlier ones write into
`Ex07.1_outputs/`. On Google Colab every notebook runs on its own temporary
machine, so notebooks 01 to 04 start with an `outputs-cell` that calls
`keep_outputs()` from `course_core.py`: on Colab it mounts the student's Google
Drive and keeps the results in `MyDrive/DL4Eng/Ex07.1_outputs`. If the student
declines the Drive request, `saved()` downloads each result file when it is
written and `needed()` asks for the files to be uploaded before they are read.
Locally the cell does nothing. The report notebook writes its `.md` and `.pdf`
into the same folder.

## Mini project proposal

The set ends with two mini projects (notebook 04, section 6). Each student
chooses one mini project from the Part 2 sets and solves it individually in
one month. The course provides the ground truth once a project is chosen,
built by a classical solver under policy C10 (`COURSE_POLICIES.md`), so the month
goes into the deep learning.

| | the problem | the deep learning | the ground truth provided |
|---|---|---|---|
| **A · The slot with its insulation** | the 32 wires resolved: copper, a thin enamel coat, air between them; the conductivity jumps 10³–10⁴-fold at every edge | a domain-decomposed PINN, one network per material, with temperature and heat flow matched across every edge | finite differences with several nodes across the enamel, per-cell conductivity, the currents of this set, with a grid-refinement table |
| **B · How old is the insulation?** | aged insulation conducts less where it has degraded; only a few sensors in the slot, read at a few currents | an inverse PINN: one network for θ(x, y, I), one for k(x, y), fitted to the physics and the sensors; then the hot spot and the margin to the class limit | slots with a hidden degraded region: sensor readings with positions and noise, and the true fields for scoring; cases of rising difficulty |
