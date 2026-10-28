# MP12A · Stability Estimation after Contingencies

**Mini project of Ex12.2, Screening Contingencies · Part 2 · course policy C14**

One Colab notebook, `MP12A_stability_estimation.ipynb` (GPU runtime, T4), in five parts:

| part | given | what it holds |
|---|---|---|
| A · The problem | solved | why the screen must be fast, the IEEE 39-bus grid and its converters, the swing equations and their conditions, why there is no exact solution, what is asked |
| B · The data and the physics | solved | the graph network's inputs and output, the 16 800 cases, how the converters move the critical clearing times, the physics features that can be built in, the censored loss |
| C · The reference model and the ground truth | solved | the scheme step by step, convergence in the time step and the search, the one-machine check, the energy check, ANDES, the reference model's accuracy and time |
| D · The graph neural network | TODO cells | your network, your training and screen margin, your figures; a given cell evaluates and times it |
| E · The report | TODO cells | the goal metrics against the ground truth and the reference model, and ten questions |

Its first cell fetches `mp12a.py`, `grid39c.py` (the 39-bus grid), the ground truth `MP12A_truth.npz` and Ex12.2's library files (`problem_12_2.py`, `course_core.py`, `pinn_core.py`). The results pass from part to part in memory: nothing to save, mount or upload.

## The problem

Ex12.2's screen by the critical clearing time (CCT), on the **IEEE 39-bus grid** (MATPOWER `case39`; ten classical machines, $H$ and $x'_d$ of Pai 1989) with **0, 20, 40 or 60 % of the generation from converters** placed at random among the nine local machines. **New against Ex12.2:** a grid six times larger with ten machines, 35 contingencies, and converters - no inertia, modelled as a negative constant impedance (assumed). The multi-machine CCT has no closed form; the one-machine case of the equal area criterion checks the solver.

| | value |
|---|---|
| grid | 39 buses, 46 branches, 21 loads, 10 machines; 100 MVA, 50 Hz (assumed; case39 is 60 Hz) |
| machines | classical, $H$ = 24 to 42 s (machine 1, the interconnection, 500 s), damping $D = H$ (assumed) |
| loads, converters | constant impedances at the pre-fault voltage (assumed) |
| contingency | a solid fault (1e-04 p.u. reactance) at the first bus of one of the 35 branches whose loss leaves the grid whole, cleared by tripping it |
| operating points | 480: load 0.85-1.15 of the base case, each load jittered 10 %; 120 at each converter share; 432 for training, 48 for the test |
| insecure | CCT of 140 ms or less (Ex12.2's protection time); stable means no two machines 180° apart within 3 s |

## What you build

A graph neural network that takes one case - the 39 buses' numbers before the fault (power flow, machines, converters, inertia, the faulted bus, and four optional physics features from two Kron reductions) and the grid after the trip - and returns its CCT; and a screen margin chosen on the training cases. The choice of network, training and figures is yours.

## The reference model and the ground truth

* **Reference model** (timed): RK4 on 10 ms (half a cycle), the fault cleared exactly at the trial time, bisection from [0, 500 ms] to an interval of 7.8 ms; all 35 contingencies of a new operating point at once. On the 1 680 test cases: mean error 1.75 ms, worst 3.90 ms, no insecure case missed, 2 false alarms; about 1 s per operating point on the authors' CPU (shared).
* **Ground truth** (scoring), `MP12A_truth.npz`: RK4 on 2 ms, bisection to 0.01 ms, 16 800 cases. Halving the step from 40 ms changes no CCT by more than 0.013 ms, and the change falls about tenfold per halving; refining the bisection tenfold changes the CCT by 0.46, 0.027 and 0.003 ms. Checks: the one-machine case against the equal area criterion within 0.02 ms; the transient energy of the lossless, undamped grid conserved to 2e-10; **ANDES** (GENCLS, its own implicit trapezoidal method) on 24 random test cases: 20 answered, the 13 run with the same fault throughout within 0.5 ms (median 0.25 ms), the 7 where ANDES needed a milder fault after clearing 0.2 to 11 ms off; power flows within 3e-05 degrees.

## Worked example (from the ground truth)

| converter share | insecure cases | median CCT | at 500 ms or more |
|---|---|---|---|
| 0 % | 9.8 % | 239 ms | 9.9 % |
| 20 % | 17.9 % | 212 ms | 6.7 % |
| 40 % | 15.9 % | 226 ms | 6.8 % |
| 60 % | 13.8 % | 231 ms | 7.8 % |

The reference model on the same test cases: mean error 1.75 ms, 197 insecure cases all found, 2 false alarms of 1 483.

## Requirements

* the CCT of the 1 680 test cases within **5 ms on average** (Ex12.2's agreed accuracy; the reference model meets it);
* **no insecure test case missed**, at most **5 %** of the secure ones flagged, with a margin chosen on the training cases;
* the online screen of a new operating point at least **10 times faster** than the reference model on the same Colab runtime; the training reported apart, as the offline cost.

## What you hand in

The PDF written at the end of part E: the results table, your figures, and the ten answers.
