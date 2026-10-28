# MP10.2B · A Day of Operation Under an Hourly Price

**Mini project of Ex10.2, Steam, Charge and Heat in a Solid Oxide Electrolysis Cell · Part 2 · course policy C14**

One Colab notebook, `MP10.2B_day_of_operation.ipynb` (GPU runtime, T4), in five parts:

| part | given | what it holds |
|---|---|---|
| A · The problem | solved | why an electrolyser follows the hourly price, the cell of Ex10.2, the contract and the 10 K limit, the model of an hour and its map over the operating range, why there is no exact solution and what is exact (a Lagrange multiplier separates the hours), what is asked |
| B · The data and the physics | solved | Ex10.2's parametric network as the model of every hour, its accuracy against the ground truth, the optimisation it is used for and its choices (bounds, contract, limit, optimiser), the four price curves |
| C · The reference model and the ground truth | solved | the ground truth step by step, its convergence and checks, the optimal day on each curve, the reference model (SLSQP with the finite volumes) run, judged and timed |
| D · The network and the optimiser | TODO cells | your model of an hour, your optimiser through it, your figures; a given cell runs it on the four curves, times it and judges every schedule by the finite volumes |
| E · The report | TODO cells | the goal metrics against the ground truth and the reference model, and twelve questions |

Its first cell fetches `mp102b.py`, the ground truth `MP10.2B_truth.npz` and Ex10.2's library files (`problem.py`, `course_core.py`, `pinn_core.py`); the setup cell fetches Ex10.2's trained network, `Ex10.2_weights.pt`, from the exercise's folder. The results pass from part to part in memory: nothing to save, mount or upload.

## The problem

One cell of Ex10.2 (100 cm², its three coupled laws, its cost of an hour and its 10 K limit) is run for a day under an hourly electricity price. For each of the 24 hours the operator chooses the cell voltage $V_t$ (1.15 to 1.35 V) and the inlet temperature $T_t$ (750 to 850 °C): 48 numbers. The hydrogen of the day is sold under a contract, what 24 hours at 1.20 V and 800 °C make, 93.33 g; the schedule minimises the day's cost of electricity and ageing, $C = \sum_t (100\,p_t E_t + A_t)$ [cent], with the contract met and the temperature change along the cell at most 10 K in every hour. Each hour is quasi-steady. **No exact solution exists**: every hour's numbers come from Ex10.2's coupled model; once a Lagrange multiplier prices the hydrogen, the hours are independent, and the contract fixes the multiplier.

| | value |
|---|---|
| cell | Ex10.2's, 100 cm²; every kinetic, transport and ageing value Ex10.2's estimate |
| decisions | $V_t$ 1.15–1.35 V and $T_t$ 750–850 °C for each of 24 hours |
| contract | 93.33 g of hydrogen a day (24 h at 1.20 V and 800 °C, by finite volumes on 800 cells) |
| limit | 10 K along the cell in every hour (Ex10.2's estimate); the 10 K band of voltages is 1.238–1.320 V at 750 °C |
| prices | four curves, invented for the project: autumn weekday 2–14, sunny spring day 0–18, windy night 0.8–15, winter cold snap 9.5–33 cent/kWh |

## What you build

A schedule for any price curve, computed through a network: Ex10.2's parametric network $(x, V, T_\mathrm{in}) \mapsto (y, \theta, i)$, trained once (Ex10.2: 2005 s on a 2-core CPU), is the model of every hour, and the 48 numbers are found by a gradient-based optimiser through it, its gradients by autograd. Your choices: the model (Ex10.2's network as it is, trained further, or your own), how the bounds, the contract and the limit are held, the optimiser, the figures. Every schedule is judged by the finite volumes, not by the network.

## The reference model and the ground truth

* **Reference model** (timed): SLSQP on the 48 numbers with Ex10.2's coupled finite volumes on 50 cells (its agreed accuracy) as the model of every hour, forward differences (three solves an hour per gradient), started from 1.25 V and 800 °C. 9 to 13 iterations, 8 to 18 s per price curve on the authors' CPU (shared with other jobs). Judged on 800 cells: the day's cost 0.15 to 0.21 % above the optimum, the hydrogen 0.15 % above the contract, the largest change 10.11 K.
* **Ground truth** (scoring), `MP10.2B_truth.npz`: the finite volumes on 800 cells tabulated on 81 × 41 operating points (2.5 mV × 2.5 K), bicubic splines, each hour minimised for a given multiplier, the multiplier that makes the contract by bisection, the schedule checked again hour by hour on 800 cells. The autumn weekday's cost is 32.7263, 32.7259, 32.7265 and 32.7271 cent with tables on 100, 200, 400 and 800 cells (changes below 2e-05 relative) and changes by less than 6e-06 cent between operating grids of 10 and 2.5 mV; the finite volumes conserve steam and heat to rounding; under a flat price the optimum is one operating point.

## Worked example (from the ground truth)

| price curve | day's cost | best constant point | what the schedule saves | multiplier |
|---|---|---|---|---|
| autumn weekday | 32.73 cent | 33.83 cent (1.28 V, 750 °C) | 3.3 % | 5.12 euro/kg |
| sunny spring day | 33.69 cent | 35.67 cent | 5.5 % | 5.59 euro/kg |
| windy night | 33.24 cent | 35.03 cent | 5.1 % | 5.90 euro/kg |
| winter cold snap | 68.82 cent | 71.67 cent | 4.0 % | 9.39 euro/kg |

Each optimal day makes exactly the contract and touches the 10 K limit (10.00 K by the finite volumes). The voltage moves between the edges of the 10 K band at 750 °C, 1.237 V in the dear hours and 1.32 V in the cheap ones; the inlet temperature stays at 750 °C almost all day (the ageing rises steeply with temperature), raised to 752–757 °C only in the cheapest night hours of the windy night and the cold snap. Each hour makes 3.36 to 4.62 g.

## Requirements

* the day's cost of your schedule, judged by the finite volumes on 800 cells, within **0.5 %** of the ground truth's on each of the four price curves;
* the hydrogen of the day within **0.5 %** of the contract, judged the same way;
* the largest temperature change in any hour at most **10.2 K** (the 10 K limit and a tolerance of 0.2 K), judged the same way;
* the online answer to a new price curve at least **10 times faster** than the reference model on the same Colab runtime; the training of the model reported apart, as the offline cost.

The solved version uses Ex10.2's shipped weights unchanged (within 0.16 % in the hydrogen and 0.20 K over the range) and SLSQP with autograd's gradients through the network: the cost within 0.02 %, the contract within 0.03 %, 9.94 K, 0.08 to 0.12 s per price curve, 149 to 158 times faster than the reference model on the authors' CPU.

**What changed against the proposal of Ex10.2.** The contract is now what 1.20 V at 800 °C make in a day (the proposal: 1.25 V). With 1.25 V the cell ran at the top of the 10 K band in nearly every hour and the hourly schedule saved only 0.9 % against a constant operating point, so a constant answer would almost pass the 0.5 %; with 1.20 V it saves 3.4 to 5.9 %. The day is solved for four price curves, none used to build anything, so that the online comparison is a new case for both methods; the ground truth interpolates its table by splines instead of picking grid nodes, so the contract is met exactly. The requirements are the proposal's, with the limit's tolerance set at 0.2 K.

## What you hand in

The PDF written at the end of part E: the results table, your figures, and the twelve answers.
