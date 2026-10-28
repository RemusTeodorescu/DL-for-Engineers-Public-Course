# MP10.2A · The Channel Across Its Height, and the Electrode Under It

**Mini project of Ex10.2, Steam, Charge and Heat in a Solid Oxide Electrolysis Cell · Part 2 · course policy C14**

One Colab notebook, `MP10.2A_channel_and_electrode.ipynb` (GPU runtime, T4), in five parts:

| part | given | what it holds |
|---|---|---|
| A · The problem | solved | why the steam at the reaction site matters, the cell of Ex10.2 with its channel and porous electrode resolved, the domain and its conditions, why there is no exact solution, the zero-strength limit, what is asked |
| B · The data and the physics | solved | the network's inputs and outputs, the scaled equations, the six loss terms and their scales, the balance of the mean, what can be built in, the plug-flow hand calculation, the training points |
| C · The reference model and the ground truth | solved | the march step by step, convergence, conservation, the zero-strength check, the reference model's accuracy and time |
| D · The physics-informed network | TODO cells | your network, your training, your figures; a given cell evaluates it and times it |
| E · The report | TODO cells | the goal metrics against the ground truth and the reference model, and twelve questions |

Its first cell fetches `mp102a.py`, the ground truth `MP10.2A_truth.npz` and Ex10.2's library files (`problem.py`, `course_core.py`, `pinn_core.py`). The results pass from part to part in memory: nothing to save, mount or upload.

## The problem

Ex10.2's electrolysis cell, held at 800 °C, at any cell voltage from 1.15 to 1.35 V. **New against Ex10.2:** the steam fraction is resolved across the 1 mm fuel channel (Poiseuille's profile, diffusion across) and through the 0.5 mm porous fuel electrode to the reaction at the electrolyte, where the charge transfer is evaluated (Nernst potential at the reaction site, Ex10.2's activation and ohmic losses, no gas-transport loss term). Steam in and hydrogen out of the electrode one for one, nothing reacting in the pores, so the steam at the reaction site is $y_r = y_w - K_E\,i$. The current density at every $x$ is the root of a nonlinear equation in the steam at the wall: **no exact solution exists**. As both transport resistances go to zero (strength $S \to 0$) the model becomes Ex10.2's one-dimensional steam balance and charge transfer without gas-transport loss.

| | value |
|---|---|
| cell | Ex10.2's, 100 × 100 mm, isothermal at 800 °C |
| fuel channel | 100 mm long, 1 mm high, 1 m/s mean speed, Poiseuille; $D$ = 8.0e-04 m²/s (Ex10.2's estimate) |
| feed | 90 % steam, 10 % hydrogen |
| fuel electrode | 0.5 mm thick, $D_\mathrm{eff}$ = 1.0e-05 m²/s (estimate); $K_E$ = 0.228 per A/cm², limiting current 3.94 A/cm² at the inlet (Ex10.2 assumed 4.0) |
| kinetics | $i_0$ = 0.5 A/cm², ASR = 0.15 Ω cm² at 800 °C (Ex10.2's estimates) |
| scales | $\alpha = LD/(Uh^2)$ = 80, $\beta = h/(2FcD)$ = 5.70e-03 per A/cm², $K_I$ = 0.456 per A/cm² |
| cell voltage | 1.15 to 1.35 V, a network input; test voltages 1.18, 1.27 and 1.33 V |

## What you build

One physics-informed network $(X, Z, V) \mapsto (y, i)$ for the steam fraction in the channel and the current density along it, over the whole range of voltages, trained once with the transport equation, its three boundary conditions, the charge transfer at the reaction site and the balance of the mean as its loss. The choice of network, of what to build into it, of the training and of the figures is yours.

## The reference model and the ground truth

* **Reference model** (timed): Crank–Nicolson along the channel on 21 × 5 nodes, central differences across, Rannacher's start, the current density by Newton's method at the wall node, the discrete flow normalised so the grid carries exactly the flow fed. The coarsest grid that meets the three requirements at 21 voltages over the range: at most 0.05 % in the current, 3.2e-04 in the outlet mean, 1.1e-04 at the reaction site. 4 to 11 ms per new voltage on the authors' CPU, shared with other jobs (the solved network: 0.27 to 0.30 ms, 15 to 32 times faster).
* **Ground truth** (scoring), `MP10.2A_truth.npz`: the same scheme on 1601 × 161 nodes. Five grids each twice as fine change the cell current at 1.33 V by 3.3e-03, 8.3e-04, 2.1e-04 and 5.2e-05 A and the outlet mean by 1.5e-05 to 2.4e-07 (observed order 2.00); within 2e-05 A of the Richardson extrapolation; the trapezoid integral of the current density agrees with Faraday's law to 3.1e-07; at $S$ = 0.1, 0.01 and 0.001 the difference from the one-dimensional limit falls from 3.04 to 0.32 to 0.03 A, in proportion to $S$.

## Worked example (from the ground truth)

| cell voltage | cell current | Ex10.2's 1-D model | outlet mean | electrode's surface at the outlet | reaction site at the outlet |
|---|---|---|---|---|---|
| 1.18 V | 93.84 A | 102.48 A (+9.2 %) | 0.4718 | 0.4701 | 0.2905 |
| 1.27 V | 122.22 A | 130.92 A (+7.1 %) | 0.3423 | 0.3402 | 0.1200 |
| 1.33 V | 138.97 A | 146.57 A (+5.5 %) | 0.2659 | 0.2638 | 0.0391 |

The current density falls along the channel from 1.11, 1.46 and 1.71 A/cm² at the inlet to 0.79, 0.97 and 0.98 at the outlet. At the outlet the electrode's surface lies 1.7 to 2.1 thousandths below the mixing-cup mean; the reaction site 0.18 to 0.22 below the surface. The plug-flow hand calculation (gas mixed across the channel, electrode resolved) misses by up to 0.19 % in the current, 1.27e-03 in the outlet mean and 3.01e-03 at the reaction site; Ex10.2's one-dimensional model overstates the current by 4.9 to 9.9 % over the range.

## Requirements

* the cell current within **0.1 %**, the outlet's mixing-cup steam fraction within **5e-04**, and the steam at the reaction site beyond the first 5 mm within **1e-03** of the ground truth, at 1.18, 1.27 and 1.33 V, which are not trained on;
* the online answer to a new voltage at least **10 times faster** than the reference model on the same Colab runtime; the training reported apart, as the offline cost.

**What changed against the proposal of Ex10.2.** The proposal asked for 0.5 % in the current, 0.005 in the outlet mean and 0.01 at the reaction site. The plug-flow hand calculation alone meets those (0.19 %, 1.3e-03, 3.0e-03), so a network that never resolved the profile across the channel would pass. The requirements are now about half of what the hand calculation misses; the reaction site is scored beyond the first 5 mm, as in Ex09.2 and MP9.2A, because at the inlet corner the flat feed meets the electrode's flux and the field is singular. The voltage is the network's input, so that a new voltage is answered by one evaluation.

## What you hand in

The PDF written at the end of part E: the results table, your figures, and the twelve answers.
