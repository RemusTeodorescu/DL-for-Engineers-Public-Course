# MP12B · Stability Prediction after a Contingency

**Mini project of Ex12.2, Screening Contingencies · Part 2 · course policy C14**

One Colab notebook, `MP12B_stability_prediction.ipynb` (GPU runtime, T4), in five parts:

| part | given | what it holds |
|---|---|---|
| A · The problem | solved | why the swings after a fault must be predicted early, the grid, the PMU window and its noise, the swing equations and their conditions, why there is no exact solution, what is asked |
| B · The data and the physics | solved | the network's inputs and outputs, the grid features that carry the swing equations' coefficients, the loss, the 52 200 training cases simulated in the notebook (about 7 minutes) |
| C · The reference model and the ground truth | solved | the scheme step by step, convergence, the one-machine and energy checks, ANDES, the reference model's accuracy and time |
| D · The recurrent network | TODO cells | your network, your training and verdict threshold, your figures; a given cell evaluates and times it |
| E · The report | TODO cells | the goal metrics against the ground truth and the reference model, and ten questions |

Its first cell fetches `mp12b.py`, `grid39c.py` (the 39-bus grid, MP12A's), the ground truth `MP12B_truth.npz` and Ex12.2's library files (`problem_12_2.py`, `course_core.py`, `pinn_core.py`). The results pass from part to part in memory.

## The problem

MP12A's grid - IEEE 39-bus, ten classical machines, 0 to 60 % of the generation from converters with no inertia - **after** a fault has been cleared. From the first **200 ms** of PMU-like measurements (11 frames at 50 per second: the rotor angles of machines 2 to 10 against machine 1 and the speeds of all ten, with noise), predict the rotor angles **to 3 s** and the **verdict**: will two machines separate by more than 180°? **New against Ex12.2:** the question after the fault rather than before it, a measured and noisy start, trajectories rather than one number. The multi-machine swing has no closed form; the ground truth is a converged simulation.

| | value |
|---|---|
| a case | an operating point (load 0.85-1.15 of base, jitter 10 %, converters 0/20/40/60 %), one of 35 contingencies, a clearing time 60-260 ms |
| measurements | angles and speeds every 20 ms, 0 to 200 ms after clearing; noise 0.5° and 5 mHz (assumed) |
| prediction | angles at 70 instants, 240 ms to 3 s, every 40 ms; the verdict |
| data | 52 200 training cases (1 740 operating points) simulated in part B; 1 800 test cases (60 other operating points) in the truth file |
| model | classical machines, $D = H$, constant-impedance loads and converters (all assumed, as MP12A) |

## What you build

A network that reads the 11 frames and the context of the case - which machines are converters, their pre-fault power, the contingency, the clearing time, and 129 grid features from the post-fault network (one Kron reduction) - and returns the angles over the next 2.8 s and the verdict; and the verdict's threshold, chosen on the training cases.

## The reference model and the ground truth

* **Reference model** (timed): one Kron reduction, then RK4 on 10 ms from the last measured frame (with its noise) to 3 s. On the 1 800 test cases: every verdict right; the largest angle error of a case that holds is 1.15° in the median case and within 2.28° in 90 % of them (from the true state it is exact to 1e-04°); about 0.1 to 0.2 s per case on the authors' shared CPU.
* **Ground truth** (scoring), `MP12B_truth.npz`: RK4 on 2 ms from the pre-fault equilibrium, the fault cleared exactly, no noise. Halving the step from 10 ms changes no angle on 200 test cases by more than 6e-05°; the one-machine check within 0.02 ms of the equal area criterion; the transient energy conserved to 4e-11; **ANDES** (GENCLS, implicit trapezoidal, 2 ms) on 8 stable test cases agrees to 0.01° over 3 s in 7, and differs by 2.88° in the one where it needed a milder fault after clearing.

## Worked example (from the ground truth)

20.1 % of the test cases lose synchronism: 16.4, 23.6, 27.8 and 12.4 % at 0, 20, 40 and 60 % converters. In the cases that hold, the largest angle against machine 1 is about 80° in the median case.

## Requirements

* the verdict: at most **1 %** of the test cases that lose synchronism missed, at most **5 %** of those that hold flagged;
* the angles of the test cases that hold: the largest error over the 3 s of each case within **5°** in the median case and within **15°** in 90 % of them;
* the online answer to a new case at least **10 times faster** than the reference model on the same Colab runtime; the training reported apart.

The trajectory requirement was relaxed: the network was first asked to match the reference model (1.15° and 2.28°), which a network trained on 52 200 cases did not reach; the agreed accuracy is set where both can meet it.

## What you hand in

The PDF written at the end of part E: the results table, your figures, and the ten answers.
