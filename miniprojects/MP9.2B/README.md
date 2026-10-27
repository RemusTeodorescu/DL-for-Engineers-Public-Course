# MP9.2B · The Air Channel

**Mini project of Ex09.2, Hydrogen Fuel Channel for SOFC/SOEC · Part 2 · course policy C14**

One Colab notebook, `MP9.2B_air_channel.ipynb` (GPU runtime, T4), in five parts:

| part | given | what it holds |
|---|---|---|
| A · The problem | solved | why the air flow matters, what the one-sided exchange and the current following the oxygen change against Ex09.2, the domain and its conditions, why there is no exact solution, what is asked |
| B · The data and the physics | solved | the network's inputs and output, the scaled equation with the falling flow and the Stefan term, the loss terms, the oxygen balance, the plug-flow hand calculation, the training points |
| C · The reference model and the ground truth | solved | the scheme step by step, convergence, both balances, the zero-strength check, the Stefan question from the ground truth, the reference model's accuracy and time |
| D · The physics-informed network | TODO cells | your network, your training, your figures; a given cell evaluates it, finds the speed for 15 % by Newton's method through it, and times it |
| E · The report | TODO cells | the goal metrics against the ground truth and the reference model, and ten questions |

Its first cell fetches `mp92b.py`, the ground truth `MP9.2B_truth.npz` and Ex09.2's library files (`channel.py`, `course_core.py`, `pinn_core.py`). The results pass from part to part in memory.

## The problem

The air channel on the other side of Ex09.2's cell, at any air speed from 1.0 to 3.0 m/s at the inlet (stoichiometry 1.84 to 5.52). **New against Ex09.2:** oxygen is taken and nothing is returned, so the molar flow falls along the channel (the nitrogen is conserved), the gas has a velocity towards the electrode (the Stefan velocity) and the diffusive flux at the wall carries only $1 - y_w$ of the oxygen taken (J. Khazaei's review of 7 October 2026); and the current follows the oxygen at the electrode, $N/N_0 = (1 - B) + B\,r(y_w)/r(0.21)$ with $r(y) = y/(y + 0.05)$ (a saturating, Langmuir-type form; assumed). The wall condition, the velocity and the equation depend on the unknown, and **no exact solution exists**. At $B = 0$ and $S = 0$ (no coupling, no one-sided exchange) the problem is Ex09.2's with the air's numbers.

| | value |
|---|---|
| channel | 100 mm long, 1 mm high, 800 °C, 1 atm, $c$ = 11.36 mol/m³ (Ex09.2) |
| air | 21 % oxygen; $D$ = 1.9e-04 m²/s, viscosity 4.4e-05 Pa s (estimates); 0.33 kg/m³; Re 7–22 |
| speed | 1.0–3.0 m/s at the inlet (a network input); stoichiometry 5 at 2.72 m/s |
| current | 0.5 A/cm² at the inlet's composition, $N_0 = i_0/4F$ = 0.0130 mol/(m² s); $K$ = 0.05 (assumed), $B$ = 1, $S$ = 1 |
| scales | $Y$ = 0.114, $G$ = 0.0526 |

## What you build

One physics-informed network θ(X, Z, s) for the whole channel and the whole range of air speeds, trained once, with the falling flow and the Stefan term in its loss; trained a second time without the Stefan term to answer the Stefan question with a number; the speed for 15 % oxygen at the electrode at the outlet by Newton's method through the network.

## The reference model and the ground truth

* **Reference model** (timed): Crank–Nicolson along the channel on 21 × 4 nodes with the molar flow marched beside the field, Rannacher's start, three iterations per step for the nonlinear wall, Stefan term and flow, the discrete flow normalised; the 15 % speed by the secant method (eight marches). Worst error at the electrode 0.022 percentage points; the 15 % speed 0.35 % off; 6 to 17 ms per new speed and 50 to 110 ms per speed search on the authors' CPU (shared with other jobs).
* **Ground truth** (scoring), `MP9.2B_truth.npz`: the same scheme on 1601 × 81 nodes. Five grids each twice as fine change the electrode's outlet value with ratios 4.1, 4.0, 4.0 (second order); within 2e-07 of the Richardson extrapolation; the oxygen and nitrogen balances close to 1e-08; at $B = 0$, $S = 0$ it reproduces Ex09.2's exact solution (air's numbers) to 8.5e-06 beyond 5 mm (3.9e-07 beyond 10 mm, the rest is the entrance).

## Worked example (from the ground truth)

| air speed | stoichiometry | oxygen at the electrode at the outlet | mixing-cup mean at the outlet | molar flow left |
|---|---|---|---|---|
| 1.2 m/s | 2.21 | 12.96 % | 13.13 % | 90.94 % |
| 1.9 m/s | 3.50 | 15.92 % | 16.10 % | 94.16 % |
| 2.7 m/s | 4.97 | 17.41 % | 17.58 % | 95.85 % |

At stoichiometry 5 (2.72 m/s) the electrode sees 17.43 % oxygen at the outlet. Without the Stefan term in the equation, 17.42 % (0.0076 point less); with Ex09.2's equimolar model, 16.69 % (0.74 point less). The one-sided exchange matters through the falling flow and the $1 - y_w$ at the wall, not through the cross velocity. The speed for 15 % at the electrode at the outlet is **1.6064 m/s** (stoichiometry 2.96); Ex09.2's balance would say 1.90 m/s. The plug-flow hand calculation misses the electrode by 0.18 point.

## Requirements

* the oxygen fraction at the electrode beyond the first 5 mm within **0.1 percentage point** of the ground truth at 1.2, 1.9 and 2.7 m/s, which are not trained on;
* the Stefan question answered **with a number**: the change of the oxygen at the electrode at the outlet at stoichiometry 5, from the network trained with and without the term;
* the air speed for 15 % at the electrode at the outlet within **1 %**, by a gradient through the network;
* the online answer to a new speed at least **10 times faster** than the reference model on the same Colab runtime; the training reported apart.

## What you hand in

The PDF written at the end of part E: the results table, your figures, and the ten answers.
