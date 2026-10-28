# MP7.2C · Where Was the Panel Struck?

**Mini project of Ex07.2, Benchmark PDEs (notebook 3, the struck panel) · Part 2 · course policy C14**

One Colab notebook, `MP7.2C_panel_strike.ipynb` (GPU runtime, T4), in five parts:

| part | given | what it holds |
|---|---|---|
| A · The problem | solved | why an unseen blow must be located, the panel with its strip and its bolt, what is new against Ex07.2, the wave equation and its conditions, why the inverse problem is hard, what is asked |
| B · The data and the physics | solved | what the amortised network takes and returns, the physics in the training data, reciprocity (six marches give the readings of any strike), the training data |
| C · The reference model and the ground truth | solved | the finite-difference scheme step by step, convergence on four grids, the zero-strength check, the least-squares reference on the four test cases, the Cramér-Rao bound |
| D · The network | TODO cells | your network, your training, your figures; a given cell evaluates and times it |
| E · The report | TODO cells | the goal metrics against the hidden answers and the least-squares reference, and eight questions |

Its first cell fetches `mp72c.py`, the ground truth `MP7.2C_truth.npz` and Ex07.2's library files. The results pass from part to part in memory: nothing to save, mount or upload.

## The problem

Ex07.2's 0.40 × 0.40 m panel (clamped edges, wave speed 80 m/s) with a **reinforcing strip** along y at x = 0.27 m (mass per area × 2 on its line, a 12 mm Gaussian profile) and a **bolt** carrying a bracket at (0.10, 0.20) m (mass per area × 11 over a 15 mm radius), both assumed. A hammer blow nobody saw - a Gaussian of initial velocity, footprint 25 mm (known), centre (x0, y0) and peak v hidden - and three to six sensors that record the displacement every 0.1 ms for 20 ms with 2 to 10 µm of noise. With the strip and the bolt the panel's mass depends on the place, its modes do not separate and **no exact solution exists**; at zero strength of both the solver reproduces Ex07.2's exact sum of modes.

| | value |
|---|---|
| panel | 0.40 × 0.40 m, clamped, c = 80 m/s (Ex07.2) |
| strip, bolt | as above (assumed) |
| strike | x0, y0 from 0.06 to 0.34 m, at least 40 mm from the bolt; v 0.5 to 1.5 m/s; footprint 25 mm (assumed) |
| sensors | six positions, 3 to 6 read; 201 readings each over 20 ms |
| noise | 2, 5 or 10 µm, one standard deviation (assumed) |

## What you build

One amortised network, trained once over the range of strikes on simulated readings, that answers a new set of readings (any three to six sensors) with (x0, y0, v) in one evaluation. Its inputs, size, training and figures are yours; an inverse PINN may be compared in the report.

## The reference model and the ground truth

* **Forward model**: finite differences, the five-point Laplacian over the node's mass and the leapfrog (c√2 Δt/h = 0.71), started by Taylor's series. Training data by **reciprocity**: six marches from impulses at the sensors on the 2 mm grid give the readings of any strike as one weighted sum, identical to the march to single precision.
* **Reference model** (timed): scipy's least_squares over (x0, y0, v), the forward model on the 4 mm grid, from nine starts (the misfit has many valleys): 545 to 657 forward runs and 30 to 61 s per set of readings on one CPU thread. On the four test cases: position within 0.10 to 0.90 mm, velocity within 0.42 % (2.18 % in case 4), the motion its answer implies within 0.26 to 3.03 % of the peak.
* **Ground truth** (scoring), `MP7.2C_truth.npz`: the forward model on 2 mm and 1 mm grids, Richardson-extrapolated. Four grids each twice as fine change the readings by 10.5, 2.8 and 0.71 % of the largest (second order); at zero strength the scheme reproduces Ex07.2's exact sum of modes to 0.15 % on 1 mm and 0.007 % extrapolated. The file holds, per case, the noisy readings (seeded), the mask of the sensors read, the noise and seed (observed), and the strike, the clean readings and the motion on the nodes every 8 mm every 0.5 ms (hidden).

## Worked example (case 1, from the ground truth)

A strike at (130, 270) mm at 1.00 m/s: the six sensors read up to 0.136 mm, first moving 1 to 3 ms after the blow; the panel's largest displacement is 0.230 mm. On Ex07.2's uniform panel the same strike gives readings up to 0.18 mm different: the strip and the bolt change the motion completely.

## Requirements

* the strike's position within **10 mm** and its velocity within **10 %** in all four cases;
* the motion of the panel that the answer implies (the ground truth's solver run with the strike found) within **5 %** of the true motion's peak;
* a new set of readings answered at least **10 times faster** than a new least-squares fit with the reference model on the same Colab runtime; simulation and training reported apart, as the offline cost.

The proposal's targets are kept. The Cramér-Rao bound (part C) leaves 0.11 to 1.30 mm on the position, 0.13 to 1.34 % on the velocity and 0.18 to 2.01 % on the motion: the motion is the binding requirement, 2.5 standard deviations in case 4.

## The solved version

`MP7.2C_panel_strike_solved.ipynb` (private): a dense network (three tanh layers of 512, 256 and 256 neurons, single precision, 512 259 weights) on the six records at 0.2 ms, each divided by its own root mean square, with the logarithm of that size and the mask as inputs; trained on 40 000 strikes simulated by reciprocity, fresh noise (2 to 10 µm) and a random three to six sensors at every batch, Adam for 200 epochs; 14 to 16 minutes offline on one CPU thread, simulation included. Its answer is polished by two Gauss–Newton steps through the reciprocity operator (four weighted sums each), which take it from the right valley of the misfit to its bottom: the network alone was within 1.95 mm and 2.83 % and missed case 4's motion by 5.22 %. With the steps: position within 0.04 to 0.86 mm, velocity within 0.38 % (2.15 % in case 4), motion within 0.21 to 2.94 % of the peak, the least-squares reference's accuracy; on 100 noisy repeats of case 4 the spread is at the Cramér-Rao bound. A new set of readings in 77 to 135 ms against 35 to 41 s for least squares on the same CPU, 256 to 529 times faster.

## What you hand in

The PDF written at the end of part E: the results table, your figures, and the eight answers.
