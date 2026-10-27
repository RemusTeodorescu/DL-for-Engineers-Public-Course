# MP9.2A · The SOEC Steam Channel

**Mini project of Ex09.2, Hydrogen Fuel Channel for SOFC/SOEC · Part 2 · course policy C14**

One Colab notebook, `MP9.2A_soec_steam_channel.ipynb` (GPU runtime, T4), in five parts:

| part | given | what it holds |
|---|---|---|
| A · The problem | solved | why the feed speed matters, the channel run as an electrolyser, the current following the steam, the domain and its conditions, why there is no exact solution, what is asked |
| B · The data and the physics | solved | the network's inputs and output, the scaled equation, the loss terms, the balance of the mean, the plug-flow hand calculation that can be built in, the training points |
| C · The reference model and the ground truth | solved | the scheme step by step, convergence, the balance of the mean, the zero-strength check, the reference model's accuracy and time |
| D · The physics-informed network | TODO cells | your network, your training, your figures; a given cell evaluates it, finds the speed for 70 % by Newton's method through it, and times it |
| E · The report | TODO cells | the goal metrics against the ground truth and the reference model, and ten questions |

Its first cell fetches `mp92a.py`, the ground truth `MP9.2A_truth.npz` and Ex09.2's library files (`channel.py`, `course_core.py`, `pinn_core.py`). The results pass from part to part in memory: nothing to save, mount or upload.

## The problem

Ex09.2's channel run as a solid oxide electrolysis cell: 90 % steam fed, steam taken at the electrode and hydrogen returned, one for one, at any feed speed from 0.2 to 1.2 m/s. **New against Ex09.2:** the current is not given but follows the steam at the electrode, $N/N_0 = (1 - B) + B\,r(y_w)/r(0.90)$ with $r(y) = y/(y + 0.2)$ (a saturating, Langmuir-type form; assumed), so the electrode condition is nonlinear and **no exact solution exists**. At $B = 0$ the problem is Ex09.2's with the flux reversed.

| | value |
|---|---|
| channel | 100 mm long, 1 mm high, 800 °C, 1 atm, $c$ = 11.36 mol/m³ (Ex09.2) |
| feed | 90 % steam, 10 % hydrogen; speed 0.2–1.2 m/s (a network input) |
| gas | $D$ = 8.0e-04 m²/s (Ex09.2's estimate); viscosity 4.0e-05 Pa s (estimate); Re 0.93–5.59 |
| current | 0.5 A/cm² at the inlet's composition, $N_0 = i_0/2F$ = 0.0259 mol/(m² s); $K$ = 0.2 (assumed), $B$ = 1 |
| scales | $Y$ = 0.228, $G$ = 0.0125 (Ex09.2's) |

## What you build

One physics-informed network θ(X, Z, s) for the whole channel and the whole range of feed speeds, trained once; the steam fraction along the electrode for a new speed from one evaluation, and the speed for 70 % conversion by Newton's method through the network, its derivative by autograd. The choice of network, of what to build into it, of the training and of the figures is yours.

## The reference model and the ground truth

* **Reference model** (timed): Crank–Nicolson along the channel on 21 × 3 nodes, Rannacher's start (two implicit columns), the nonlinear electrode by Newton's method on its node, the discrete flow normalised so the grid carries exactly the flow fed; the speed for 70 % by the secant method (six marches). Worst error at the electrode 0.036 percentage points at the test speeds; 2 to 6 ms per new speed and 25 to 31 ms per speed search on the authors' CPU (shared with other jobs).
* **Ground truth** (scoring), `MP9.2A_truth.npz`: the same scheme on 1601 × 81 nodes. Five grids each twice as fine change the electrode's outlet value by 5.3e-06, 1.3e-06, 3.3e-07 and 8.2e-08 (second order); within 1e-07 of the Richardson extrapolation; the balance of the mean closes to 1e-07; at $B = 0$ it reproduces Ex09.2's exact solution with the flux reversed to 1.5e-07.

## Worked example (from the ground truth)

| feed speed | steam converted | steam at the electrode at the outlet | hand calculation (plug flow) |
|---|---|---|---|
| 0.25 m/s | 83.63 % | 14.68 % | 14.70 % |
| 0.55 m/s | 43.59 % | 50.67 % | 50.75 % |
| 0.95 m/s | 25.93 % | 66.56 % | 66.65 % |

The speed for 70 % conversion is **0.3201 m/s**; the uniform current would say 0.362 m/s. There the current density falls from 0.50 A/cm² at the inlet to 0.35 at the outlet, where the electrode sees 26.93 % steam, 0.074 percentage points below the mean. The plug-flow hand calculation misses the electrode by up to 0.105 points.

## Requirements

* the steam fraction at the electrode beyond the first 5 mm within **0.05 percentage point** of the ground truth at 0.25, 0.55 and 0.95 m/s, which are not trained on;
* the feed speed for 70 % conversion within **1 %**, by a gradient through the network;
* the online answer to a new speed at least **10 times faster** than the reference model on the same Colab runtime; the training reported apart, as the offline cost.

The topic's requirement was 0.1 percentage point; it was tightened to 0.05 because the plug-flow hand calculation alone misses by only 0.105 points, so at 0.1 a network that barely learnt the profile across the channel would pass.

## What you hand in

The PDF written at the end of part E: the results table, your figures, and the ten answers.
