# MP8.2B · Heat Capacity and Coefficient Together

**Mini project of Ex08, Heatsink Cooling of a Power Module · Part 2 · course policy C14**

One Colab notebook, `MP8.2B_heat_capacity_and_coefficient.ipynb` (GPU runtime, T4), in five parts:

| part | given | what it holds |
|---|---|---|
| A · The problem | solved | why two numbers from one record, the sink and the pulse, natural convection on the fins, what each number does to the record, why there is no exact solution, the five test records, what is asked |
| B · The data and the physics | solved | two designs (a network from the record to the numbers; a network conditioned on the numbers), the sensitivity of the record to each number, the training records the physics generates |
| C · The reference model and the ground truth | solved | the march, its convergence and Ex08's limit, the least-squares reference on the five test records, its accuracy and time |
| D · The network | TODO cells | your network and `estimate(readings)`, your training, your figures; a given cell evaluates and times it |
| E · The report | TODO cells | the goal metrics against the hidden values and the least-squares reference, and eighteen questions |

Its first cell fetches `mp82b.py`, the ground truth `MP8.2B_truth.npz` and Ex08's library files. The results pass from part to part in memory.

## The problem

Ex08 Part 2's pulse - 150 W for 20 s, a 150 s window, the thermocouple at x = 80 mm on the module face read every 2 s with 0.05 K of noise - with **two numbers hidden**: the base's heat capacity ρc_p and the fan's coefficient h_fan. **New against Ex08:** the fins are cooled by the fan and by natural convection, h = h_fan + 1.42 (ΔT/L_fin)^¼ W/(m²K) (MP8.2A's law; assumed), so the fin side is nonlinear and **no exact solution exists**: the records are made by the converged finite-difference march. At S = 0 it reproduces Ex08's exact pulse.

| | value |
|---|---|
| base plate | 100 × 10 mm, aluminium 6061, k = 167 W/(m·K) |
| pulse | 150 W for 20 s, window 150 s |
| thermocouple | x = 80 mm on the module face, 75 readings every 2 s, noise 0.05 K |
| hidden ρc_p | 1.95–2.92 MJ/(m³K) (±20 % about 2.43; assumed range) |
| hidden h_fan | 25–55 W/(m²K) (about Ex08's 40; assumed range) |
| test records | five pairs and noise seeds, observed readings and hidden values in `MP8.2B_truth.npz` |

## What you build

One network, trained once over the ranges of the two hidden numbers, that answers a new noisy record with both numbers. The solved design maps the record to the two numbers (as scaled logarithms), trained on 2 000 records the physics generates (the march on 41 × 5 nodes, 0.5 s, within 0.01 K of the truth) with fresh noise at every step. A physics-informed network conditioned on the two numbers, with a fast fit through it, is the alternative the notebook describes.

## The reference model, the least-squares reference and the ground truth

* **Reference model**: Crank-Nicolson on 21 × 3 nodes with a 1 s step, the nonlinear fin side by chord iterations, Rannacher's start after each switch.
* **Least-squares reference** (timed): `scipy.optimize.least_squares` on the logarithms of the two numbers, the reference model as the forward model, from Ex08's values. Both numbers within 0.4 % on the five test records with a misfit equal to the noise; 9 to 21 forward runs, 0.3 to 1.2 s per record on one CPU core depending on the machine.
* **Ground truth**, `MP8.2B_truth.npz`: the same march on 161 × 17 nodes with a 0.125 s step. Four grids each twice as fine change the record by 0.022, 0.0045 and 0.0011 K (second order); within 4e-04 K of the Richardson extrapolation; at S = 0 within 4e-04 K of Ex08's exact pulse.

## Worked example (Ex08's values, from the ground truth)

At ρc_p = 2.43 MJ/(m³K) and h_fan = 40 W/(m²K) the thermocouple peaks at 9.78 K at 24 s and reads 1.64 K at 150 s. A 5 % change of ρc_p moves the record by up to 0.44 K at the switch-off; a 5 % change of h_fan by up to 0.17 K in the tail near 78 s; natural convection lowers it by up to 0.49 K.

## Requirements

* both numbers within **5 %** of the hidden values on the five test records;
* the online answer to a new record at least **10 times faster** than a new least-squares fit with the reference model on the same Colab runtime; the training (and the building of the training records) reported apart, as the offline cost.

## What you hand in

The PDF written at the end of part E: the results table, your figures, and the eighteen answers.
