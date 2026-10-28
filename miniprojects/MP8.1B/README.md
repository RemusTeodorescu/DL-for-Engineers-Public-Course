# MP8.1B · Two Modules on One Sink

**Mini project of Ex08, Heatsink Cooling of a Power Module · Part 2 · course policy C14**

One Colab notebook, `MP8.1B_two_modules.ipynb` (GPU runtime, T4), in five parts:

| part | given | what it holds |
|---|---|---|
| A · The problem | solved | why the gap matters, the sink with two modules, the warming air and the falling coefficient, the coupled equations, why there is no exact solution, the hottest point against the gap, what is asked |
| B · The data and the physics | solved | the networks' inputs and outputs, the loss terms and what can be built in (Ex08's exact series with two footprints), the training points |
| C · The reference model and the ground truth | solved | the finite-volume scheme coupled to the air step by step, convergence, the energy balance, Ex08's limit, the reference model's accuracy and time (one gap, and the best gap by a 1-D search) |
| D · The physics-informed network | TODO cells | your networks, your training, your figures, and the best gap by a gradient through the network; a given cell evaluates and times it |
| E · The report | TODO cells | the goal metrics against the ground truth and the reference model, and eighteen questions |

Its first cell fetches `mp81b.py`, the ground truth `MP8.1B_truth.npz` and Ex08's library files. The results pass from part to part in memory.

## The problem

Two 75 W modules, 15 mm wide each, on Ex08's base plate, symmetric about its centre, a gap g of 0 to 40 mm apart. **New against Ex08:** the fan blows the air across the sink along x, and three things change. The air **warms** as it takes up the heat (heat capacity flow C = 15 W/K, so it warms by 10 K across the sink; assumed). The fan's coefficient **falls** along the flow as the boundary layer grows (from 50 W/(m²K) at the inlet to 30 at the outlet, mean Ex08's 40; assumed). **Natural convection** on the hot fins adds 1.42 (ΔT/L_fin)^¼ W/(m²K), with ΔT the fin root's rise above the local air (MP8.2A's law; assumed); about 8 W/(m²K) at a 45 K rise. The fin-side coefficient depends on the place and on the solution itself: the problem is **nonlinear** and coupled to the air's energy balance, and **no exact solution exists**. At zero strength of the change (S = 0) the problem is Ex08's linear one with two footprints, and the solver reproduces Ex08's exact series.

| | value |
|---|---|
| base plate | 100 × 10 mm, aluminium 6061, k = 167 W/(m·K) (constant); fins collapsed as in Ex08 |
| modules | two of 75 W on 15 mm each, 50 kW/m²; gap 0–40 mm (a network input) |
| air | C = 15 W/K, θ_a(0) = 0 at the inlet, 10 K at the outlet |
| coefficient | h(x, ΔT) = 40 (1 + 0.25 (1 − 2x/W)) + 1.42 (ΔT/0.04)^¼ W/(m²K); h_eff about 448 at the inlet and 309 at the outlet at a 45 K rise |

## What you build

One physics-informed network for the plate's rise θ(x, z, g) and one for the air's rise θ_a(x, g), trained once with the gap as an input; then the best gap from the gradient of the hottest point with respect to the gap, through the network.

## The reference model and the ground truth

* **Reference model** (timed): finite volumes on 50 × 5 cells of 2 mm coupled to the air cell by cell, solved by **Newton's method** for every gap (three iterations from a cold start, each assembling and factorising the Jacobian); the best gap by Brent's bounded search, ten nonlinear solves warm-started from one another. Hottest point within 0.027 K of the truth at every gap (25 × 3 cells: 0.12 K); best gap 28.87 mm, 0.01 mm off. About 2.6 to 2.9 ms per gap and 19 to 21 ms per best gap on one CPU core. Chord iterations with one kept factorisation, timed beside it, are about seven times cheaper (0.4 ms per gap, 2.7 ms per best gap).
* **Ground truth** (scoring), `MP8.1B_truth.npz`: the same scheme, Newton to 1e-10 K, on 400 × 40 and 800 × 80 cells, extrapolated by Richardson, at 81 gaps every 0.5 mm. Four grids at a 17.5 mm gap change the plate's mean rise by 2.7e-05, 5.7e-06 and 1.9e-06 K (ratios 4.7 and 2.9) and the hottest point by 3e-03, 4e-04 and 3e-04 K; the two finest differ by less than 3.3e-04 K at every gap; the air takes the 150 W in to 1e-11; at S = 0 it reproduces Ex08's exact series with two footprints to 0.002 K.

## Worked example (from the ground truth)

The hottest point is 50.86 K above the inlet air with no gap, falls to **48.80 K at the best gap, 28.86 mm**, and is 49.07 K at 40 mm. With Ex08's air (S = 0) the best gap is 35 mm at 49.22 K: the warming air moves the best gap closer, because spreading the modules pushes the downstream one into the warmest air at the weakest coefficient. Every gap from 22.5 to 35.5 mm is within 0.1 K of the best.

## Requirements

* the hottest point within **0.1 K** of the ground truth at every gap, 0 to 40 mm every 0.5 mm;
* the best gap, by a gradient through the network, within **2 mm** of 28.86 mm;
* **one** training, the gap an input;
* the online answer at least **10 times faster** than the reference model (Newton) on the same Colab runtime, for a new gap and for the best gap; the training reported apart, as the offline cost.

## What you hand in

The PDF written at the end of part E: the results table, your figures, and the eighteen answers.
