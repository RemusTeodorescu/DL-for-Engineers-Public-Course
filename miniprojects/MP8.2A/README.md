# MP8.2A · Thermal Cycling with Natural Convection

**Mini project of Ex08, Heatsink Cooling of a Power Module · Part 2 · course policy C14**

Five Colab notebooks, run in order (GPU runtime, T4):

| notebook | given | what it holds |
|---|---|---|
| `MP8.2A_00_problem.ipynb` | solved | why the history matters, the heat sink, what is new against Ex08, the domain and its conditions, why there is no exact solution, what is asked |
| `MP8.2A_01_data_and_physics.ipynb` | solved | the network's inputs and output, the scaled equation, the loss terms and what can be built in, the training points, optional thermocouple data |
| `MP8.2A_02_reference.ipynb` | solved | the reference model and the ground truth: the scheme step by step, convergence, energy balance, the zero-strength check, the reference's accuracy and time |
| `MP8.2A_03_pinn.ipynb` | TODO cells | your network, your training, your figures; a given cell evaluates and times it |
| `MP8.2A_04_report.ipynb` | TODO cells | the goal metrics against the ground truth and the reference, and eighteen questions |

## The problem

The heat sink of Ex08 under a duty cycle: the module on for 20 s every 60 s, ten cycles, 600 s in all, at any power from 100 to 200 W. **New against Ex08:** the fins are cooled by the fan *and* by natural convection, h = 40 + 1.42 (ΔT/L_fin)^¼ W/(m²K) (laminar vertical plate in air, Incropera & DeWitt ch. 9; assumed), so the fin-side condition is nonlinear and **no exact solution exists**. At zero strength of the natural-convection term the problem is Ex08's.

| | value |
|---|---|
| base plate | 100 × 10 mm, aluminium 6061, k = 167 W/(m·K), ρc_p = 2.43 MJ/(m³K) |
| fins | ten, 2 × 40 mm, pitch 10 mm, collapsed into h_eff(ΔT): 322 W/(m²K) at no rise, 370 at 20 K |
| module | 30 mm footprint, 100–200 W (a network input); 50 kW/m² at 150 W |
| air | 40 °C |
| duty cycle | 20 s on, 40 s off, ten cycles |

## What you build

One physics-informed network θ(x, z, t, P) for the whole window and the whole range of powers, trained once. The choice of network, of what to build into it, of the training and of the figures is yours.

## The reference model and the ground truth

* **Reference model** (timed): Crank-Nicolson finite differences on 11 × 3 nodes with a 4 s step, the nonlinear fin side by chord iterations, Rannacher's start after each switch. Worst peak or trough at the test powers 0.21 K; 26 to 57 ms per new power on the authors' CPUs.
* **Ground truth** (scoring), `MP8.2A_truth.npz`: the same scheme on 161 × 33 nodes with a 0.125 s step. Four grids each twice as fine change the tenth peak by 0.0037, 0.0009 and 0.0002 K (second order); within 0.001 K of the Richardson extrapolation; energy balance closed to 1e-6; at zero strength it reproduces Ex08's exact series to better than 0.001 K.

## Worked example (150 W, from the ground truth)

Peaks of the ten cycles 16.02, 20.44, 22.22, 22.93, 23.21, 23.32, 23.37, 23.39, 23.39, 23.40 K; troughs 5.98 … 9.99 K; the swing settles at 13.4 K by the seventh cycle. With the fan alone (Ex08) the tenth peak is 25.26 K: natural convection lowers it by 1.9 K.

## Requirements

* every cycle's peak and trough within **0.3 K** of the ground truth at 115, 165 and 195 W, which are not trained on;
* the field at 600 s within **0.5 K**;
* the online answer to a new power at least **10 times faster** than the reference model on the same Colab runtime; the training reported apart, as the offline cost.

## What you hand in

The PDF written by notebook 04: the results table, your figures, and the eighteen answers.
