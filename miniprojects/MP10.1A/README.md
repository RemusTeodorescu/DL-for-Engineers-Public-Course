# MP10.1A · The Full Model, Where the SPMe Stops

**Mini project of Ex10.1, PINN Particle Diffusion · Part 2 · course policy C14**

One Colab notebook, `MP10.1A_dfn_lfp.ipynb` (GPU runtime, T4), in five parts:

| part | given | what it holds |
|---|---|---|
| A · The problem | solved | why the voltage and capacity at any rate matter, the energy-format LFP cell, the Doyle-Fuller-Newman model with every condition, why there is no exact solution and where the SPMe fails, what is asked |
| B · The data and the physics | solved | the network's inputs and outputs, why a physics-only PINN of the DFN is not attempted, the physics built in (lithium balance, Ex10.1's exact particle series, the open-circuit potentials, the electrolyte's zero flux and conservation), the training discharges, the loss terms |
| C · The reference model and the ground truth | solved | the DFN scheme step by step, convergence on four meshes with Richardson extrapolation, the lithium balances, the zero-strength check against Ex10.1's exact series, the reference model's accuracy and time |
| D · The network | TODO cells | your network, your training data and training, your figures; a given cell evaluates and times it |
| E · The report | TODO cells | the goal metrics against the ground truth and the reference model, and ten questions |

Its first cell installs PyBaMM (about a minute on Colab); the next fetches `mp101a.py`, the ground truth `MP10.1A_truth.npz` and Ex10.1's library files. The results pass from part to part in memory: nothing to save, mount or upload.

## The problem

Constant-current discharges of an energy-format LFP cell from full charge to 2.0 V, at any rate from 0.5C to 6C: the terminal voltage, the capacity delivered and the electrolyte concentration across the cell. **New against Ex10.1:** the electrolyte is resolved and coupled to a particle at every place of each electrode - the full Doyle-Fuller-Newman model (DFN) instead of Ex10.1's single particle. The DFN is nonlinear and coupled, so **no exact solution exists**. The strength of the change is how slowly the electrolyte transports lithium: with the electrolyte and the solids 1 000 times faster, the DFN's graphite particles carry one flux and follow Ex10.1's exact series.

| | value |
|---|---|
| cell | PyBaMM `Prada2013` (A123 26650, LiFePO4/graphite; Prada et al. 2013) with **both electrode thicknesses × 3** (assumed: an energy-format cell, not a datasheet) |
| layers | graphite 102 µm, separator 25 µm, LiFePO4 240 µm; electrode area 0.18 m² |
| particles | graphite R = 5 µm, D_s = 3.0e-15 m²/s; LiFePO4 R = 50 nm, D_s = 5.9e-18 m²/s |
| electrolyte | 1 200 mol/m³ at the start, D_e = 2.0e-10 m²/s, t+ = 0.36 |
| capacity | 6.9 Ah nominal (1C = 6.9 A), cut-off 2.0 V, 25 °C isothermal |
| open-circuit potentials | graphite Chen et al. 2020, LFP Afshar et al. 2017 (as PyBaMM ships Prada2013) |

Where the SPMe stops: at 5C it delivers 1.39 Ah against the DFN's 2.84 Ah, because the electrolyte in the LiFePO4 next to its collector falls to a few mol/m³.

## What you build

One network, trained once over 0.5C to 6C, answering `net.voltage(q, C)` and `net.electrolyte(x, q, C)`; the capacity is where its voltage reaches 2.0 V. A physics-only PINN of the full DFN (four coupled fields, the particle field four-dimensional with the rate, porosity jumps at the separator, an event at the cut-off) is not a sound plan for a month on Colab, so the project is **hybrid**: the physics with a closed form is built into the network, the rest is learnt from DFN discharges simulated at rates the student chooses, never within 0.25C of a test rate.

## The reference model and the ground truth

* **Reference model** (timed): PyBaMM's DFN on its default mesh (20 points in each layer and particle radius), built once, one solve per new rate. Up to 95 % of the capacity within 2.2, 17.8 and 44.2 mV of the ground truth at 1C, 3C and 5C; capacity +0.04, +0.33 and +1.17 %; electrolyte within 1.2, 5.2 and 14.9 mol/m³; 35 to 78 ms per new rate on one CPU core.
* **Ground truth** (scoring), `MP10.1A_truth.npz`: the DFN on a mesh eight times finer (160 points), tolerances 1e-9. At 5C the capacity changes by 0.0254, 0.0063 and 0.0016 Ah on meshes ×1 → ×8 (observed order 2.01, 2.00); within 0.018 % of the Richardson value 2.8372 Ah. Lithium in the electrolyte and in the particles kept to 1e-13 or better, the Faraday balance of the graphite to 3e-6. At zero strength the graphite's surface concentration follows Ex10.1's exact series to 1.5e-3, 3.7e-4 and 9.2e-5 of its drop on meshes ×2, ×4, ×8.

## Worked example (from the ground truth)

| rate | capacity, DFN | capacity, SPMe | lowest electrolyte concentration at 95 % of the capacity |
|---|---|---|---|
| 1C | 5.81 Ah | 5.81 Ah | 831 mol/m³ |
| 3C | 3.94 Ah | 3.95 Ah | 157 mol/m³ |
| 5C | 2.84 Ah | 1.39 Ah | 4 mol/m³ |

## Requirements

* the voltage at 1C and 3C within **20 mV**, at 5C within **50 mV** of the ground truth, up to 95 % of its capacity; none of the three rates trained on;
* the 5C capacity within **2 %**;
* the electrolyte concentration at 1C, 3C and 5C, at 25, 50, 75 and 95 % of the capacity, within **30 mol/m³** everywhere;
* a new rate answered at least **10 times faster** than the reference model on the same Colab runtime; simulation and training reported apart, as the offline cost.

**Set in this conversion:** the voltage is scored up to 95 % of the capacity (the proposal named no window: in the last few per cent the voltage falls by hundreds of millivolts per per cent of charge, and that error is the capacity's, scored on its own), and the electrolyte requirement was added. With them the proposal's numbers stand; the reference model meets every one, with the least margin at 3C (17.8 of 20 mV).

## The solved version

`MP10.1A_dfn_lfp_solved.ipynb` (private): the voltage as the surface open-circuit voltage (lithium balance and Ex10.1's exact series at the particles' surfaces, which carries the graphite's steps and the knee) minus a 3 × 64 tanh network of (q, ln C); the electrolyte as a 32-term cosine series whose coefficients come from a second 3 × 64 network, with zero flux at the collectors, salt conservation and the start built in; 19 169 weights. Eleven DFN discharges at 0.5 to 6C on the mesh twice as fine as the default, Adam then L-BFGS, 167 to 229 s offline on one CPU thread. At the test rates: voltage within 4.07, 8.87 and 14.87 mV, capacity +0.02, +0.08 and +0.22 %, electrolyte within 7.98, 17.38 and 19.60 mol/m³; online 1.1 to 1.8 ms against 63 to 72 ms for the reference on the same CPU runtime (39 to 57 times, depending on the load). Without the built-in surface open-circuit voltage the same network reached only 22.5 mV at 3C; with five training rates instead of eleven, 9.0 mV.

## What you hand in

The PDF written at the end of part E: the results table, your figures, and the ten answers.
