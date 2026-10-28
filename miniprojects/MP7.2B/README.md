# MP7.2B · The Eigenvalues of a Real Slot

**Mini project of Ex07.2, Benchmark PDEs (notebook 2, the stator slot and its thermal runaway) · Part 2 · course policy C14**

One Colab notebook, `MP7.2B_slot_eigenvalues.ipynb` (GPU runtime, T4), in five parts:

| part | given | what it holds |
|---|---|---|
| A · The problem | solved | why the slot's eigenvalues matter (the runaway current), the tapered slot, what is new against Ex07.2, the eigenproblem and its conditions, why there is no exact solution, what is asked |
| B · The data and the physics | solved | the network's inputs and outputs, the map that makes every slot the same square, the two routes to a loss (the residual with trainable eigenvalues, and Rayleigh's quotient), what can be built in, the Gauss points |
| C · The reference model and the ground truth | solved | the wall-following grid and the bilinear elements step by step, convergence on four grids, the zero-strength check, the stair steps of finite differences, the reference model's accuracy and time |
| D · The physics-informed network | TODO cells | your network, your training, your figures; a given cell evaluates and times it |
| E · The report | TODO cells | the goal metrics against the ground truth and the reference model, and eight questions |

Its first cell fetches `mp72b.py`, the ground truth `MP7.2B_truth.npz` and Ex07.2's library files (`slot_problem.py`). The results pass from part to part in memory: nothing to save, mount or upload.

## The problem

Ex07.2's stator slot, 20 mm deep with 200 mm² of averaged winding, **tapered**: W_b wide at the back and 20 mm - W_b at the opening, as a slot between parallel-sided teeth is. Wanted: the first three eigenpairs of -∇²u = λu with u = 0 on the walls, and the runaway current I_run = A_wire (k_eff λ1 / (FILL ρ20 α))^½ of Ex07.2, **for any taper**. The slanted walls stop the equation separating, so **no exact solution exists**; at zero taper (W_b = 10 mm) the slot is Ex07.2's rectangle and the solver reproduces its exact eigenvalues.

| | value |
|---|---|
| slot | 20 mm deep, mean width 10 mm (area 200 mm², Ex07.2), straight tapered sides |
| taper | width at the back W_b = 10 to 12.5 mm, a network input; opening 20 mm - W_b |
| winding | k_eff = 0.70 W/(m·K), copper fill 0.503, 2 mm wires (Ex07.2) |
| copper | 1.72e-8 Ω·m at 20 °C, +0.393 % per kelvin (Ex07.2) |
| rectangle (Ex07.2) | λ = 0.1234, 0.1974, 0.3208 per mm², runaway 158.38 A, exact |

## What you build

One network that returns the three modes at any point of any slot of the range, u_k(x, y; W_b), and the three eigenvalues λ_k(W_b), trained once. The route to the loss (the residual with trainable eigenvalues, or Rayleigh's quotient), what to build in and the figures are yours.

## The reference model and the ground truth

* **Reference model** (timed): the slot mapped onto a square and covered by a grid that follows the walls, bilinear finite elements, the three smallest eigenvalues by shift-and-invert (scipy's eigsh); 20 × 40 cells (0.5 mm across at the mean width). At the test widths: λ1 0.18 %, λ2 0.21 %, λ3 0.39 % high, runaway current 0.14 A high; 7 to 22 ms per new taper on one CPU thread, depending on the load of the machine.
* **Ground truth** (scoring), `MP7.2B_truth.npz`: the same scheme on 160 × 320 and 320 × 640 cells (0.0625 and 0.03125 mm), Richardson-extrapolated. Four grids each twice as fine change the eigenvalues by 3.3e-4, 8.2e-5 and 2.0e-5 of λ1 (observed order 2.00 for all three); at zero taper the scheme gives Ex07.2's exact rectangle to 4e-10. Five-point finite differences on a square grid, for comparison, cut the slanted walls into stair steps and are still 0.36 % low on λ1 at 0.03125 mm (first order). Eigenvalues, runaway current and the three modes on the reference grid's nodes at 10.5, 11.25 and 12 mm (test), and at 9.5 and 13 mm (outside the range).

## Worked example (the 8-to-12 mm slot, W_b = 12 mm, from the ground truth)

λ = 0.1226, 0.2018 and 0.3253 per mm²: λ1 0.6 % below the rectangle's, λ2 and λ3 2.2 and 1.4 % above. The runaway current falls from 158.38 A to 157.90 A. The 9.5 mm slot is the 10.5 mm slot turned upside down and has the same eigenvalues.

## Requirements

* λ1 within **0.3 %**, λ2 and λ3 within **1 %**, the runaway current within **0.5 A** of the ground truth at 10.5, 11.25 and 12 mm, which are not trained on;
* the online answer to a new taper - the eigenvalues, the runaway current and the three modes at the reference grid's 861 nodes - at least **10 times faster** than the reference model on the same Colab runtime; the training reported apart, as the offline cost.

The first proposal's targets are kept. The reference grid was chosen as the coarsest that meets them (on 10 × 20 cells λ1 is 0.70 % and the runaway current 0.55 A off).

## The solved version

`MP7.2B_slot_eigenvalues_solved.ipynb` (private): the slot mapped onto the square (ξ = x/a(y), η = y/B), the walls built in by (1 - ξ²)(1 - η²), the modes even in x by seeing ξ²; one tanh network (3 × 32) of (ξ², η, τ) for the three modes and a small one (2 × 16) of τ for the eigenvalues. Trained by route (b): the Rayleigh quotients of the modes made orthonormal in order (Gram–Schmidt), weighted by 1/λ_k of the rectangle, with the eigenvalue network fitted to the Ritz values; Adam (1 000 steps, eight random widths each) then L-BFGS on twelve fixed widths, then the eigenvalue network alone; 3 to 5 minutes on one CPU thread. Measured at the test widths: λ1 within 0.006 %, λ2 within 0.05 %, λ3 within 0.02 %, the runaway current within 0.004 A, the modes within 0.21 % of their peaks; at 13 mm, outside the range, λ3 0.33 % off. A new taper in 0.85 ms against 19.9 ms for the reference model on the same CPU, 23 times faster (the target of 10 is for a Colab T4).

## What you hand in

The PDF written at the end of part E: the results table, your figures, and the eight answers.
