# MP7.1A · The Slot with Its Insulation

**Mini project of Ex07.1, PINN Thermal Prediction · Part 2 · course policy C14**

One Colab notebook, `MP7.1A_slot_with_insulation.ipynb` (GPU runtime, T4), in five parts:

| part | given | what it holds |
|---|---|---|
| A · The problem | solved | why the hot spot matters, the slot with every wire resolved, what is new against Ex07.1, the conditions on the copper edges, why there is no exact solution, what is asked |
| B · The data and the physics | solved | the network's inputs and output, the equations in each material, the edge conditions and the loss terms, what can be built in, the training points and a quadrature over the insulation |
| C · The reference model and the ground truth | solved | the finite-volume scheme step by step, convergence on five grids, the energy balance, the zero-strength check, the reference model's accuracy and time |
| D · The physics-informed network | TODO cells | your network (one per material), your training, your figures; a given cell evaluates and times it |
| E · The report | TODO cells | the goal metrics against the ground truth and the reference model, and ten questions |

Its first cell fetches `mp71a.py`, the ground truth `MP7.1A_truth.npz` and Ex07.1's library files. The results pass from part to part in memory: nothing to save, mount or upload.

## The problem

The stator slot of Ex07.1, 10 × 20 mm, its walls held at 90 °C, with **every wire resolved** instead of averaged: 32 copper cores, each in its enamel coat, in impregnating resin. The heat $\rho(T) J^2$ is made in the copper only, and the conductivity jumps by a factor of 2000 at every copper edge, so **no exact solution exists**. Every cell is mixed between the averaged slot (strength S = 0, Ex07.1's exact series) and the resolved one (S = 1).

| | value |
|---|---|
| slot | 10 × 20 mm, walls at 90 °C (Ex07.1) |
| winding | 4 × 8 wires, centres 2.5 mm apart both ways |
| copper core | radius 1.000 mm, k = 400 W/(m·K) (Incropera & DeWitt, table A.1) |
| enamel, grade 2 | to radius 1.0375 mm (IEC 60317 minimum), k = 0.20 W/(m·K), assumed |
| resin | everywhere else, k = 0.20 W/(m·K), assumed |
| current | 10 to 45 A per wire (RMS), a network input; rated 31.42 A = 10 A/mm² |
| copper's resistivity | 1.72e-8 Ω·m at 20 °C, +0.39 % per kelvin (Ex07.1) |

## What you build

One network, θ(x, y, I), decomposed by material - a network for the copper and one for the insulation, joined by the same temperature and the same heat flow on every copper edge - trained once over the whole range of currents. The design, what is built in, the training and the figures are yours.

## The reference model and the ground truth

* **Reference model** (timed): cell-centred finite volumes on the quarter slot (symmetric about both axes), 60 × 120 cells of 0.083 mm, the copper's share of each cell counted and its conductivity the two materials in series. At the test currents: hot spot within 0.034 K, field within 0.30 K of the ground truth; 25 to 47 ms per new current on the authors' CPU (one sparse solve, the field read at the scoring points).
* **Ground truth** (scoring), `MP7.1A_truth.npz`: the same scheme on 400 × 800 cells of 0.0125 mm at nine currents (5 to 50 A). Five grids each twice as fine change the hot spot by 0.096, 0.005, 0.0008 and 0.0014 K and the field at the scoring points by 0.31, 0.010, 0.0027 and 0.0016 K; the energy balance closes to 1e-9; at S = 0 the scheme reproduces Ex07.1's exact series to 2e-4 K. The field is scored at 17 824 points 0.05 mm apart, those within 0.05 mm of a copper edge left out (the slope jumps there, and an interpolated value would measure the interpolation).

## Worked example (rated current, 31.42 A, from the ground truth)

Hot spot 21.14 K above the wall, against 18.74 K for Ex07.1's averaged slot: resolving the wires raises it 12.8 % (the averaged slot gives the resolved hot spot with k_eff = 0.62 instead of 0.70 W/(m·K)). The eight cores of the quarter run at 21.14, 19.70, 16.02, 8.08, 10.68, 10.07, 8.48 and 4.74 K; each core is within 0.03 K of one temperature (0.08 K at 45 A). The slot makes 228.89 W per metre, all of it leaving through the walls.

## Requirements

* the hot spot within **0.1 K** of the ground truth at 17.5, 31.42 and 41.5 A, which are not trained on;
* the field within **0.5 K** at every scoring point at the same currents;
* the online answer to a new current at least **10 times faster** than the reference model on the same Colab runtime; the training reported apart, as the offline cost.

The first proposal's targets are kept. The reference grid was chosen as the coarsest that meets both accuracies (on 50 × 100 cells the hot spot is 0.19 K off).

## The solved version

`MP7.1A_slot_with_insulation_solved.ipynb` (private) uses the linearity of the insulation: θ = T_j(I) plus a small dome in core j, and Σ T_j(I) φ_j(x, y) in the insulation, where the eight unit fields φ_j are one tanh network (4 × 48, 7 688 weights) with the cores, the walls and the symmetry lines built in, trained by minimising the trace of the cores' conductance matrix (Dirichlet's principle) on a Gauss quadrature of the insulation; the eight core temperatures for any current come from an 8 × 8 heat balance. Measured on one CPU thread: hot spot within 0.073 K and field within 0.21 K at the test currents (0.12 K on the hot spot at 50 A, outside the range); training 17 minutes; a new current about 1 ms against 25 to 47 ms for the reference model.

## What you hand in

The PDF written at the end of part E: the results table, your figures, and the ten answers.
