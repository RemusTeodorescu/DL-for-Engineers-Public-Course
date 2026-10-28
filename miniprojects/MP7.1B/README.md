# MP7.1B · How Old Is the Insulation?

**Mini project of Ex07.1, PINN Thermal Prediction · Part 2 · course policy C14**

One Colab notebook, `MP7.1B_insulation_ageing.ipynb` (GPU runtime, T4), in five parts:

| part | given | what it holds |
|---|---|---|
| A · The problem | solved | why aged insulation matters, the slot with a hidden disc and its sensors, what is new against Ex07.1, why the inverse problem is hard, what is asked |
| B · The data and the physics | solved | the amortised network's inputs and outputs, how the forward model makes the training data, the noise and missing sensors |
| C · The reference model and the ground truth | solved | the forward model step by step, convergence, energy balance, the zero-strength check, least squares on the five cases with its time, and the Cramér-Rao bound |
| D · The network | TODO cells | your network, your training data and training, your figures; a given cell evaluates and times it |
| E · The report | TODO cells | the goal metrics against the hidden values and least squares, and ten questions |

Its first cell fetches `mp71b.py`, the test cases `MP7.1B_truth.npz` and Ex07.1's library files. The results pass from part to part in memory: nothing to save, mount or upload.

## The problem

The averaged slot of Ex07.1 (k_eff = 0.70 W/(m·K), walls at 90 °C) with a hidden **disc of aged insulation** of lower conductivity. Sensors read the rise at four currents, with noise. Find where the disc is, how strong it is, and how hot the slot's hot spot now runs. The disc's edge breaks Ex07.1's separable rectangle, so **no exact solution exists**; at k_d = k_eff (strength zero) the solver reproduces Ex07.1's exact series.

| | value |
|---|---|
| slot | 10 × 20 mm, k_eff = 0.70 W/(m·K), heat as Ex07.1 |
| aged disc (hidden) | centre anywhere with the disc inside; radius 1.0 to 2.5 mm; k_d = 0.28 to 0.49 W/(m·K), 40 to 70 % of k_eff (assumed) |
| sensors | six at x = ±2 mm, y = -4.5, 0, 4.5 mm; four (top pair missing) in cases 4 and 5 |
| currents | 15, 25, 31.42, 40 A |
| noise | 0.05 or 0.10 K, one standard deviation, seeded (assumed for thermocouples) |

Five test cases of rising difficulty (six sensors at 0.05 K; six at 0.10 K, twice; four at 0.05 K; four at 0.10 K), readings and hidden answers both in the public file.

## What you build

One **amortised** network, trained once over the whole range of discs on slots simulated with the forward model, that answers a new set of readings by being evaluated: the disc's centre, radius, conductivity and strength, and the hot spot at the four currents.

## The reference model and the ground truth

* **Reference model** (timed): `scipy.optimize.least_squares` over (x, y, r, k) from four starting discs, the forward model inside on 40 × 80 cells of 0.25 mm. On the five cases: centre within 0.22 to 1.00 mm, strength within 15 %, hot spot within 0.29 K; 243 to 897 forward solves, 4 to 39 s per set of readings on one CPU core (median 5.9 s alone on the machine, 16.9 s shared).
* **Ground truth**, `MP7.1B_truth.npz`: cell-centred finite volumes on the whole slot, 400 × 800 cells of 0.025 mm, the disc's edge ramped across one cell. Four grids each twice as fine change the hot spot by 0.0040, 0.0008 and 0.0007 K and sensor 4 by 0.0011, 0.0006 and 0.0004 K (order about one, from the ramped edge); energy balance closed to 7e-8; at k_d = k_eff Ex07.1's exact series to 2e-4 K.
* **What the readings can tell**: the Cramér-Rao bound at each case's disc gives one standard deviation of 0.21 to 0.71 mm on the centre, 12 to 36 % on the strength and 0.23 to 0.35 K on the hot spot, for any unbiased method.

## Worked example (case 1, from the ground truth)

A disc of radius 2.00 mm and k = 0.35 W/(m·K) centred at (-1.0, -2.0) mm, strength 4.19 mm². It raises the hot spot at rated current from 18.74 to 20.29 K (+1.55 K). Without noise it raises sensor 4 by 0.08 K at 15 A, 0.39 K at rated current and 0.68 K at 40 A, and lowers sensor 1, on the far side, by 0.01 to 0.09 K: the heat the disc holds back goes round it.

## Requirements

* in every case: the disc's centre within **1.2 mm**, its strength πr²(k_eff - k_d)/(k_eff + k_d) within **35 %**, the hot spot at rated current within **0.3 K**;
* a new set of readings answered at least **10 times faster** than a new least-squares fit on the same Colab runtime; the simulation and training reported apart, as the offline cost.

**Changed from the first proposal** (disc within 0.5 mm, k within 20 %, hot spot within 0.1 K): the Cramér-Rao bound shows the readings cannot support those numbers for any method, and least squares misses them. The tolerances were set at about one and a half to two bound standard deviations of the hardest case, so that both least squares and the solved network meet them on all five cases. The strength replaces the radius and k apart, which the readings trade against each other; both are still reported.

## The solved version

`MP7.1B_insulation_ageing_solved.ipynb` (private): a dense tanh network, 3 × 256, 141 833 weights, on the readings relative to the healthy slot's and the sensor mask; 20 000 simulated slots and 300 epochs of Adam with fresh noise and missing sensors (627 s offline alone on the machine, 1187 s shared, one CPU thread). On the five cases: centre within 0.89 mm, strength within 31 %, hot spot within 0.25 K; 0.16 to 0.20 ms per set of readings against 5.9 to 16.9 s for least squares on the same runtime.

## What you hand in

The PDF written at the end of part E: the results table, your figures, and the ten answers.
