# Ex_10.2 — The Gas Channel of a Solid Oxide Cell

**Paired with L10.2 · Solid Oxide Cells and Optimisation · Part 2**

A solid oxide electrolysis cell at 800 °C, in **one notebook** (course
policies C11 and C13). Steam enters a gas channel and is split into hydrogen
along the electrode; the current is not given but follows from the cell
voltage, and it falls along the channel as the steam runs short. Students
compute the steam fraction three ways — a boundary-value solver, finite
volumes, and a physics-informed network trained over a whole range of cell
voltages — and compare them on accuracy and on time. Then the network's own
gradient finds the voltage at which an hour of operation is worth most.

## The problem

```
u y_x = D y_xx − i(y) / (2F h c_tot)       the steam fraction y along the channel
i(y) = (V − E(y)) / ASR                    the local current density
E(y) = E_0 + (RT/2F) ln((1−y) √p_O2 / y)   the Nernst potential
y = 0.90 at the inlet                      (built into the network)
y_x = 0 at the outlet                      (a loss term)
```

100 × 100 mm of electrode, a channel 1 mm high, gas at 1 m/s, 800 °C. In
scaled units it reads y_x = y_xx/Pe − K (V − E(y)) with Pe = 125 and
K = 1.83 per volt.

**Two of the numbers are estimates chosen for the exercise**, not measured on
any cell: the area-specific resistance, 0.25 Ω cm², which stands for all the
losses of the polarisation curve at once, and the gas diffusivity,
8 × 10⁻⁴ m²/s. So are the prices of section 6 (hydrogen 3.00 €/kg,
electricity 0.065 €/kWh). The thermodynamics is standard. The notebook and
`problem.py` say this where the numbers appear.

## The notebook

```
Ex10.2_pinn_steam_channel.ipynb         the exercise: two TODO cells, the answer in the comment above each
Ex10.2_pinn_steam_channel_light.ipynb   the same notebook with every cell written out
```

| section | what the student does |
|---|---|
| 1 – 3 | the problem, its data, its physics: the Nernst potential, the local current, the steam balance, Pe and K |
| 4 | the reference: SciPy's boundary-value solver, checked at two tolerances; then upwind finite volumes with Newton's method, for an agreed 0.005 in steam fraction |
| 5 | the PINN: the cell voltage as a second input, the trial function y_in + x 𝒩 (TODO 1), the residual (TODO 2), training; the three compared at 1.29 V and the network checked at four other voltages |
| 6 | the best cell voltage: the worth of an hour as a function of V, its gradient by autograd through the network, checked against 61 runs of the solver |
| 7 | what the notebook says |
| 8 | the report and its PDF |
| 9 | mini project proposal |

The button-cell fit, the degradation sweep and the 24-hour schedule of the
earlier six-notebook version are left to the lecture and to the mini projects.

### What it measures (CPU, seed 88)

| at 1.29 V | steam at the outlet | hydrogen | power | worst error | one solution | training |
|---|---|---|---|---|---|---|
| reference, boundary-value solver | 0.27 (69 % used) | 5.21 g/h | 178 W | 2 × 10⁻¹² between tolerances | 0.3 s | — |
| finite volumes, 50 cells (the coarsest for 0.005) | 0.28 | 5.20 g/h | 178 W | 0.0042 | 2 ms | — |
| PINN, 4 × 32, any voltage from 1.10 to 1.40 V | 0.27 | 5.21 g/h | 178 W | 0.0005 | under 1 ms | 20 s |

The same network is within 0.0013 of the reference at 1.10, 1.20, 1.35 and
1.40 V. The best voltage is 1.30 V by the network's gradient (200 steps, 0.2 s)
and 1.30 V by 61 runs of the solver (4 s), worth 0.41 cent an hour for this
one cell; the thermoneutral voltage is 1.29 V.

## Files

| | |
|---|---|
| `course_core.py` | shared by the whole course — `set_seed`, `MLP`, `to_tensor`, `check` |
| `pinn_core.py` | the PDE machinery — `grad`, `d2`, `train_two_stage` |
| `problem.py` | **this** problem — the data, the Nernst potential (NumPy or torch), the boundary-value reference, the numbers read off a solution, the samplers |
| `cell_model.py` | the fuller cell model of the earlier set — three overpotentials, heat, a degradation law. The notebook does not use it; the mini projects and `tools/miniprojects/ex102_truth.py` do |

The first two are generated. Edit `tools/pinn/*.py` and run
`python3 tools/pinn/sync_cores.py`; never edit a copy.

**The notebook is generated too**, both forms from one source,
`tools/exercises/ex102/build_ex102.py`, with the cells every one-notebook set
shares in `tools/exercises/part2_notebook.py`. Edit the builder and rerun it
rather than editing a notebook, or the two forms drift apart.

**On Colab nothing needs uploading**: the first code cell fetches the three
modules the notebook uses from the public course repository, afresh on every
run.

## Expected runtime

CPU only. About half a minute in all, most of it the one training of
section 5.

## Reference texts

Liu, G.R., *PINN with Python: An Introduction* (2025).
Raissi, Perdikaris & Karniadakis, *Physics-informed neural networks*,
J. Comput. Phys. **378** (2019) 686–707.

These are the works to read for the theory. **The code, the problem and the
exposition in this exercise set are original to this course.**

## Before this is assigned

The light version has been executed end to end on a local CPU; the exercise
version stops at its TODO cells. Still to do, as for every set (C8): a run from
a fresh Colab runtime, and a review by someone other than the author - in
particular of the two estimated parameters.

## Mini project proposal

The set ends with two mini projects (section 9). Each student chooses one
mini project from the Part 2 sets and solves it individually in one month. The
ground truth is given, built by `tools/miniprojects/ex102_truth.py` under policy C10
(`COURSE_POLICIES.md`), with a worked example of each.

| | the problem | the deep learning | the ground truth given | required |
|---|---|---|---|---|
| **MP10.2A · The channel with its temperature** | a 10 × 10 cm electrolyser at 1.20, 1.29 and 1.35 V; its temperature and current along the channel, coupled | a PINN for steam fraction and temperature along x, the local current from the cell model | an ODE march, tolerance 1e-10; worked example: at 1.20 V the cell cools to 761.20 °C and its current nearly halves | outlet within 1 K, profile within 2 K, total current within 1 % |
| **MP10.2B · The degradation law from long records** | the degradation law's k₀, E, n hidden; 3000 h voltage records at up to four test points | the cell model with trainable degradation parameters, fitted to the drift | the course's model over 3000 h, noise 0.5 mV; worked example: from one temperature, E is exactly unidentifiable | E and n within 10 % (case 1); cases 2, 3 shown unidentifiable |
