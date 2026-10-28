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

Two 75 W modules, 15 mm wide each, on Ex08's base plate, symmetric about its centre, a gap g of 0 to 40 mm apart. **New against Ex08:** the fan blows the air across the sink along x, and the air **warms** as it takes up the heat (its heat capacity flow C = 15 W/K, so it warms by 10 K across the sink; assumed), while the fins' coefficient **falls** along the flow as the boundary layer grows (h from 50 W/(m²K) at the inlet to 30 at the outlet, mean Ex08's 40; assumed). The fin-side condition couples the plate to the air's energy balance, with a coefficient that varies along x: **no exact solution exists**. At zero strength of the change (S = 0) the problem is Ex08's with two footprints, and the solver reproduces Ex08's exact series.

| | value |
|---|---|
| base plate | 100 × 10 mm, aluminium 6061, k = 167 W/(m·K); fins collapsed as in Ex08 |
| modules | two of 75 W on 15 mm each, 50 kW/m²; gap 0–40 mm (a network input) |
| air | C = 15 W/K, θ_a(0) = 0 at the inlet, 10 K at the outlet |
| coefficient | h(x) = 40 (1 + 0.25 (1 − 2x/W)) W/(m²K); h_eff 393 at the inlet, 248 at the outlet |

## What you build

One physics-informed network for the plate's rise θ(x, z, g) and one for the air's rise θ_a(x, g), trained once with the gap as an input; then the best gap by a gradient of the hottest point with respect to the gap, through the network.

## The reference model and the ground truth

* **Reference model** (timed): finite volumes on 50 × 5 cells of 2 mm coupled to the air cell by cell, the matrix factorised once (it does not depend on the gap); the best gap by Brent's bounded search, about ten solves. Hottest point within 0.025 K of the truth at every gap; best gap 28.64 mm, 0.48 mm off; 0.05 to 0.1 ms per gap and 0.6 to 1 ms per best gap on one CPU core.
* **Ground truth** (scoring), `MP8.1B_truth.npz`: the same scheme on 400 × 40 and 800 × 80 cells, extrapolated by Richardson, at 81 gaps every 0.5 mm. Four grids at a 17.5 mm gap change the hottest point by 3e-03, 4e-04 and 3e-04 K; the two finest differ by less than 3e-04 K at every gap; the air takes the 150 W in to 1e-11; at S = 0 it reproduces Ex08's exact series with two footprints to 0.002 K.

## Worked example (from the ground truth)

The hottest point is 57.90 K above the inlet air with no gap, falls to **55.91 K at the best gap, 28.17 mm**, and rises to 56.22 K at 40 mm. With Ex08's air (S = 0) the whole curve is about 6.5 K lower and its best gap is 35 mm (49.22 K): the warming air moves the best gap closer, because spreading the modules pushes the downstream one into the warmest air at the weakest coefficient.

## Requirements

* the hottest point within **0.1 K** of the ground truth at every gap, 0 to 40 mm every 0.5 mm;
* the best gap, by a gradient through the network, within **2 mm** of 28.17 mm;
* **one** training, the gap an input;
* the online answer at least **10 times faster** than the reference model on the same Colab runtime, for a new gap and for the best gap; the training reported apart, as the offline cost.

## What you hand in

The PDF written at the end of part E: the results table, your figures, and the eighteen answers.
