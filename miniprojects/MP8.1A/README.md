# MP8.1A · The Fins Resolved

**Mini project of Ex08, Heatsink Cooling of a Power Module · Part 2 · course policy C14**

One Colab notebook, `MP8.1A_fins_resolved.ipynb` (GPU runtime, T4), in five parts:

| part | given | what it holds |
|---|---|---|
| A · The problem | solved | why resolve the fins, the comb and its data, the conditions, the fin roots' corners, why there is no exact solution, how far Ex08's collapsed model is off, what is asked |
| B · The data and the physics | solved | the network's inputs and output, the loss terms and what can be built in (Ex08's collapsed model among them), the energy balance, the training points |
| C · The reference model and the ground truth | solved | the finite-volume scheme step by step, convergence, Ex08's limit, the reference model's accuracy and time |
| D · The physics-informed network | TODO cells | your network, your training, your figures; a given cell evaluates and times it |
| E · The report | TODO cells | the goal metrics against the ground truth and the reference model, and eighteen questions |

Its first cell fetches `mp81a.py`, the ground truth `MP8.1A_truth.npz` and Ex08's library files. The results pass from part to part in memory: nothing to save, mount or upload.

## The problem

Ex08's heat sink in steady operation, with the **fins resolved**: the whole cross-section - the base plate and the ten fins - as one piece of metal, with the air's coefficient on every face in the air (fin sides, fin tips, the bare base between the fins). Ex08 collapsed the fins into one coefficient on the base's underside; here nothing is collapsed. The comb is not a rectangle, so **no exact solution exists**; the solver is checked in Ex08's limit (fins collapsed), where it reproduces Ex08's exact series. Two inputs: the module's power and the air's coefficient.

| | value |
|---|---|
| base plate | 100 × 10 mm, aluminium 6061, k = 167 W/(m·K) |
| fins | ten, 2 × 40 mm, pitch 10 mm, resolved |
| module | 30 mm footprint, P = 100–200 W (a network input) |
| air | 40 °C, h = 20–60 W/(m²K) on every face in the air (a network input; range assumed) |
| test cases | (115 W, 25), (165 W, 45), (195 W, 55 W/(m²K)), not trained on |

## What you build

One physics-informed network θ(x, z, P, h) over the whole comb and both ranges, trained once. The twenty re-entrant corners at the fin roots, where the gradient is unbounded, are the difficulty.

## The reference model and the ground truth

* **Reference model** (timed): finite volumes on 1 mm square cells (1 800 cells, two across a fin), the coarsest useful grid (2 mm cells are 3 K off). Hottest point 0.07–0.12 K and field at most 0.13 K off at the test cases; heat out exact; about 2 ms per new case on one CPU core.
* **Ground truth** (scoring), `MP8.1A_truth.npz`: the same scheme on 0.25 mm cells (28 800) for the fields and 0.125 mm (115 200) for the hottest point. Four grids each twice as fine change the hottest point by 0.058, 0.022 and 0.009 K: observed order 1.36 (the fin roots' corners, where the field goes as r^(2/3)); Richardson with the observed order; heat out equal to heat in to 1e-11; with the fins collapsed it reproduces Ex08's exact series to 0.002 K.

## Worked example (150 W, h = 40 W/(m²K), from the ground truth)

Hottest point 52.71 K above the air (extrapolated), against 52.45 K for Ex08's collapsed model: the resolved comb is 0.26 K hotter over the base, colder right under each fin root and warmer between roots, up to 0.49 K apart. Heat out 150.00 W.

## Requirements

The proposal asked for 0.1 K on the hottest point. The 1 mm reference model reaches 0.07–0.12 K, and the next finer grid costs four times as much for an accuracy no fan's coefficient is known to, so the agreed engineering accuracy is:

* the hottest point within **0.15 K** of the ground truth at the three test cases;
* the field within **0.2 K** everywhere in the comb;
* the heat out through the air faces within **1 %**;
* the online answer to a new case at least **10 times faster** than the reference model on the same Colab runtime; the training reported apart, as the offline cost.

## What you hand in

The PDF written at the end of part E: the results table, your figures, and the eighteen answers.
