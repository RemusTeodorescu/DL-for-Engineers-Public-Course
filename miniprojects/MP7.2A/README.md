# MP7.2A · A Die with a Moving Hot Spot

**Mini project of Ex07.2, Benchmark PDEs (notebook 1, the silicon die) · Part 2 · course policy C14**

One Colab notebook, `MP7.2A_moving_hot_spot.ipynb` (GPU runtime, T4), in five parts:

| part | given | what it holds |
|---|---|---|
| A · The problem | solved | why the peak of a die matters, the die and its three blocks, what is new against Ex07.2, the equation and its conditions, why there is no exact solution, what is asked |
| B · The data and the physics | solved | the network's inputs and output, the scaled equation and the loss terms, what can be built in (the exact linear part), the training points |
| C · The reference model and the ground truth | solved | the finite-volume scheme step by step, convergence on four grids, the energy balance, the zero-strength checks, the reference model's accuracy and time |
| D · The physics-informed network | TODO cells | your network, your training, your figures; a given cell evaluates and times it |
| E · The report | TODO cells | the goal metrics against the ground truth and the reference model, and eight questions |

Its first cell fetches `mp72a.py`, the ground truth `MP7.2A_truth.npz` and Ex07.2's library files. The results pass from part to part in memory: nothing to save, mount or upload.

## The problem

Ex07.2's 10 × 10 mm silicon die, its edges held at the package's 65 °C, now **powered**: three 2 × 2 mm blocks of P watts each switch on in turn for 60 ms, and the die cools for the last 20 ms. **New against Ex07.2:** the die starts at the package temperature and is heated, and silicon's conductivity falls as it warms, k = 148 (300 K/T)^1.3 W/(m·K), so the heat equation is nonlinear and **no exact solution exists**. With the strength S of k(T) at zero (k = 148 W/(m·K), Ex07.2's diffusivity) the solver reproduces Ex07.2's exact modes.

| | value |
|---|---|
| die | 10 × 10 mm silicon, 0.5 mm thick (assumed); edges at 65 °C (Ex07.2) |
| blocks | 2 × 2 mm, centred at (2.5, 2.5), (7.5, 2.5) and (5.0, 7.5) mm; on 0–60, 60–120 and 120–180 ms |
| power | P = 5 to 10 W per block, a network input (assumed range); 8 W = 200 W/cm² |
| silicon | k = 148 (300 K/T)^1.3 W/(m·K): 126.7 at 65 °C, 111.4 at 100 °C (the usual fit to Glassbrenner and Slack 1964, from memory, assumed until checked); ρc = 1.68 MJ/(m³K), from Ex07.2's diffusivity |
| window | 0 to 200 ms, from the package temperature; the field scored every 2 ms |

## What you build

One network θ(x, y, t, P) for the whole window and the whole range of powers, trained once. The choice of network, of what to build into it (the edges, the start, the exact linear part of part B), of the training and of the figures is yours.

## The reference model and the ground truth

* **Reference model** (timed): vertex-centred finite volumes, Crank–Nicolson with the conductivity extrapolated to the half step and Rannacher's start after each switch, conjugate gradients; 20 × 20 cells of 0.5 mm, steps of 2 ms. At the test powers: peak 0.28 to 0.52 K low, field within 0.88 to 1.60 K; 64 to 160 ms per new power on one CPU thread, depending on the load of the machine.
* **Ground truth** (scoring), `MP7.2A_truth.npz`: the same scheme on 160 × 160 and 320 × 320 cells (0.0625 and 0.03125 mm) with steps of 0.5 and 0.25 ms, Richardson-extrapolated. Four grids each twice as fine change the field at the scoring nodes by 0.22, 0.066 and 0.017 K (second order) and the peak by 0.19, 0.014 and 0.0035 K; the largest Richardson correction is 0.0057 K; the energy balance closes to 1e-10; at S = 0 the scheme reproduces Ex07.2's exact two modes to 2.8e-3 K on a 60 K start, and the exact modal sum with the blocks on to 4.5e-3 K. The field is given at the 41 × 41 nodes every 0.25 mm at 101 instants, at 8 W, the three test powers and 4 and 11 W.

## Worked example (8 W per block, from the ground truth)

The peak is 33.84 K above the package, 98.84 °C, at 180 ms, the end of the third pulse; the largest rise falls by about half in the 10 ms after each switch. With the conductivity held at its 65 °C value (the exact modal sum) the peak is 32.17 K: **silicon's k(T) adds 1.67 K**. The blocks put 1.44 J into the die; 1.06 J leaves through the edges by 200 ms.

## Requirements

* the peak within **1 K** and its instant within **2 ms** of the ground truth at 5.5, 7.25 and 9.5 W, which are not trained on;
* the field within **2 K** at every scoring node and every recorded instant at the same powers;
* the online answer to a new power - the field at the 1 681 scoring nodes at 101 instants, and the peak - at least **10 times faster** than the reference model on the same Colab runtime; the training reported apart, as the offline cost.

The first proposal's targets are kept. The reference grid was chosen as the coarsest that meets them (on 10 × 10 cells the field is 3.05 K off).

## The solved version

`MP7.2A_moving_hot_spot_solved.ipynb` (private) builds in the exact linear part, (P/8 W) θ_lin(x, y, t), the modal sum of part B with k held at 65 °C (60 × 60 modes), which carries every jump of the heat in space and in time; a network learns only what k(T) adds, δ, as a sum of 24 products ψ_k(x, y) a_k(t, P) of two tanh networks (3 × 48 and 3 × 32, 9 024 weights), with the edges and the start built in. It is trained on the residual of δ's equation divided by 0.1 q_block: Adam (2 000 steps on fresh points) then L-BFGS (600 iterations on one fixed set), 6.5 to 8.5 minutes on one CPU thread. Measured: peak within 0.05 K and at the right instant, field within 0.13 K at the test powers (peak 0.10 K and field 0.16 K at 11 W, outside the range); a new power in 4.0 ms against 59 ms for the reference model on the same CPU, 15 times faster (the target of 10 is for a Colab T4).

## What you hand in

The PDF written at the end of part E: the results table, your figures, and the eight answers.
