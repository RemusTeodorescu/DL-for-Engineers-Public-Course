# MP10.1B · How Much Has the Cell Aged?

**Mini project of Ex10.1, PINN Particle Diffusion · Part 2 · course policy C14**

One Colab notebook, `MP10.1B_cell_ageing.ipynb` (GPU runtime, T4), in five parts:

| part | given | what it holds |
|---|---|---|
| A · The problem | solved | why the ageing mechanism matters, the LG M50 cell and its two hidden ageing factors, the SPMe and where the factors enter, Ex10.1's series for the particle's surface, why the two effects look alike, what is asked |
| B · The data and the physics | solved | the amortised network's inputs and outputs, how the forward model makes the training data, the noise and missing rates |
| C · The reference model and the ground truth | solved | the forward model step by step, convergence on four meshes, the lithium balances, the zero-strength check, least squares on the four cases with its time, and the Cramér-Rao bound |
| D · The network | TODO cells | your network, your training data and training, your figures; a given cell evaluates and times it |
| E · The report | TODO cells | the goal metrics against the hidden values and least squares, and ten questions |

Its first cell installs PyBaMM (about a minute on Colab); the next fetches `mp101b.py`, the test cases `MP10.1B_truth.npz` and Ex10.1's library files. The results pass from part to part in memory: nothing to save, mount or upload.

## The problem

An aged LG M50 cell (PyBaMM's `Chen2020`, NMC811/graphite-SiOx, 5 Ah; the single particle model with electrolyte, SPMe, 25 °C) whose ageing is seen only through the voltage of constant-current discharges. Two factors of the graphite electrode are hidden: **f_D**, its particle diffusivity relative to the fresh cell's, and **f_L**, its lithium at full charge relative to the fresh cell's (lost cyclable lithium). Find both, and the capacity at 0.5C. The SPMe (Butler-Volmer kinetics, nonlinear open-circuit potentials, the electrolyte) has **no exact solution**; at f_D = f_L = 1 (zero strength) it reproduces the fresh cell.

| | value |
|---|---|
| cell | LG M50, 5 Ah nominal (1C = 5 A), cut-off 2.5 V; graphite R = 5.86 µm, D_s = 3.3e-14 m²/s (Chen et al. 2020) |
| f_D (hidden) | 0.3 to 1.0 (assumed range) |
| f_L (hidden) | 0.80 to 1.00 (assumed range: up to 20 % of the cyclable lithium lost) |
| records | discharges from full charge to 2.5 V at 0.5C, 1C or 2C: the voltage every 0.05 Ah discharged, and the charge at the cut-off |
| noise | 5 or 10 mV on every voltage, 5 mAh on the charge, one standard deviation, seeded (assumed for a cell tester) |

Four test cases of rising difficulty (three rates at 5 mV; 1C and 2C at 5 mV; three rates at 10 mV; 2C only at 10 mV), records and hidden answers both in the public file.

## What you build

One **amortised** network, trained once over the whole range of ageing on cells simulated with the forward model, that answers a new set of records by being evaluated: f_D, f_L and the capacity at 0.5C, from the first 64 readings and the last 24 before the cut-off at each recorded rate, the charges at the cut-offs and which rates were recorded.

## The reference model and the ground truth

* **Reference model** (timed): `scipy.optimize.least_squares` over (ln f_D, f_L), the SPMe on PyBaMM's default mesh inside (built once, about 10 ms a discharge). On the four cases: f_D within 0.1 to 1.4 %, f_L within 0.16 %, the capacity within 0.12 %; 46 to 82 forward solves, 0.5 to 1.1 s per set of records on one CPU core (median 0.64 and 0.72 s in two runs).
* **Ground truth**, `MP10.1B_truth.npz`: the SPMe on a mesh eight times finer (160 points per particle radius and layer), tolerances 1e-9. At 2C on case 1's cell the capacity changes by 0.40, 0.10 and 0.025 mAh on meshes ×1 → ×8 (observed order 1.99, 2.00), within 0.01 mAh of Richardson; the Faraday balance of the graphite holds to the solver's tolerance; at f_D = f_L = 1 the model gives the fresh cell to 1e-6 V. The reference model's readings differ from the ground truth's by up to 1.3 to 4.1 mV, below the noise.
* **What the records can tell**: the Cramér-Rao bound gives one standard deviation of 0.59 to 1.46 % on f_D, 0.03 to 0.06 % on f_L and 0.02 to 0.04 % on the capacity with two or three rates; with the 2C record alone 21 % on f_D, 2.6 % on f_L and 2.1 % on the capacity, the two estimates correlating at -0.9998.

## Worked example (case 1, from the ground truth)

f_D = 0.55 and f_L = 0.90: the capacity at 0.5C falls from 5.02 to 4.46 Ah (to 4.36 Ah at 1C and 4.12 Ah at 2C). The voltage stays within about 10 mV of the fresh cell's through the first 2 Ah and falls up to 670 mV below it where the aged cell's discharge ends first. Slower diffusion alone (f_D = 0.5) changes a discharge by up to 26 mV at 0.5C and 96 mV at 2C, lost lithium alone (f_L = 0.9) by up to 462 and 343 mV, and the shapes of the two changes correlate at 0.90 to 0.92.

## Requirements

| quantity | cases 1, 2, 3 | case 4 (2C only) |
|---|---|---|
| f_D | within **5 %** | reported, not required |
| f_L | within **1 %** | within **5 %** |
| capacity at 0.5C | within **1 %** | within **5 %** |
| online time per new set of records | at least **10 times shorter** than a new least-squares fit on the same Colab runtime | the same |

**Changed from the first proposal** (both factors within 5 %, capacity within 1 %, on case 1): f_D and the capacity keep their tolerances, now on cases 1 to 3; f_L is asked within 1 % because its whole range is 0.80 to 1.00; case 4 is new, and the Cramér-Rao bound shows a single 2C record cannot separate the two factors for any method, so there f_L and the capacity are asked within 5 % (about two bound standard deviations) and f_D is reported only. The proposal's inverse PINN on the SPMe became an amortised network, as policy C14 asks for an inverse problem.

## The solved version

`MP10.1B_cell_ageing_solved.ipynb` (private): a dense tanh network, 3 × 256, 201 731 weights, on the first 64 and the last 24 readings of each rate and the charges at the cut-offs, all as changes from the fresh cell's, and the rate mask; 16 000 simulated cells (405 to 538 s) and 400 epochs of Adam with fresh noise and random subsets of rates (all three 40 %, each pair 15 %, each single rate 5 %), 628 to 824 s offline in all on one shared CPU thread. On the four cases: f_D +1.81, -2.77, +2.72 % (and +9.72 % on case 4, not required), f_L within 0.25 % (1.23 % on case 4), the capacity within 0.18 % (1.02 % on case 4); 0.14 to 0.15 ms per set of records against 0.64 to 0.72 s for least squares on the same runtime (about 4 600 to 4 700 times). On 100 noisy repeats of case 1 its f_D spreads by 2.27 %, about four times the Cramér-Rao bound: amortised, it is fast and inside the requirements, not at the information limit, which least squares comes closer to.

## What you hand in

The PDF written at the end of part E: the results table, your figures, and the ten answers.
