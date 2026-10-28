# MP9.1A · One Bend of the Serpentine

**Mini project of Ex09.1, Liquid Serpentine Cooling Plate · Part 2 · course policy C14**

One Colab notebook, `MP9.1A_u_bend.ipynb` (GPU runtime, T4), in five parts:

| part | given | what it holds |
|---|---|---|
| A · The problem | solved | why the bend matters, the domain and what is new against Ex09.1, the parabolised Dean equations and their conditions, why there is no exact solution, what is asked |
| B · The data and the physics | solved | the network's inputs and outputs, the momentum residual, the training data computed offline by the march, what can be built in (wall, inlet, symmetry, flow rate), the training points |
| C · The reference model and the ground truth | solved | the march step by step, convergence across the section and along the tube, the zero-strength and long-bend checks, the reference model's accuracy and time |
| D · The physics-informed network | TODO cells | your network, your training, your figures; a given cell evaluates it and times it |
| E · The report | TODO cells | the goal metrics against the ground truth and the reference model, and ten questions |

Its first cell fetches `mp91a.py`, the ground truth `MP9.1A_truth.npz` and Ex09.1's library files (`pipe.py`, `course_core.py`, `pinn_core.py`). The results pass from part to part in memory: nothing to save, mount or upload.

## The problem

One U-bend of Ex09.1's serpentine with a straight leg on each side, at any flow from 0.15 to 0.45 L/min. In the bend the centrifugal force moves the fastest liquid to the outer wall and drives two Dean vortices; after it they decay and the parabola re-forms. Wanted: the velocity in three dimensions, the recovery after the bend, and the bend's loss coefficient $K$. **New against Ex09.1:** the flow changes along the tube and has a secondary flow across it; the equations are coupled and nonlinear, and **no exact solution exists**. The strength of the change is the curvature ratio $\delta = a/R_c$ = 0.25; at $\delta = 0$ the problem is Ex09.1's steady limit.

The model is Dean's (curvature only through the centrifugal force) and **parabolised** (no diffusion along the tube, nothing acts upstream): a march of cross-sections. The curvature rises over one tube radius at each end of the bend (assumed). The ground truth is the converged solution of this model, not of the full Navier–Stokes equations.

| | value |
|---|---|
| tube, coolant | Ex09.1's: 6 mm inside, 50/50 water–glycol at 40 °C (typical values) |
| domain | 20 mm straight, 180° bend of 12 mm radius to the axis (37.7 mm), 110 mm straight: 55.90 tube radii (assumed) |
| flow | 0.15 to 0.45 L/min, a network input: Re 226 to 678, Dean number 113 to 339 |
| inlet | Ex09.1's parabola |
| scored | five sections: bend middle, bend exit, 1 and 5 diameters after, outlet; 441 points each |

## What you build

One network $(w, \psi, \pi)(s, x, y, Q)$ - the axial velocity, the secondary stream function and the pressure gradient along the tube - for the whole piece of tube and the whole range of flows, trained once on the momentum equation along the tube and on data computed offline by the march at ten training flows. The choice of network, of what to build into it, of the training and of the figures is yours.

## The reference model and the ground truth

* **Reference model** (timed): the march on 16 × 32 cells of the section (`dean.py`'s finite-volume grid), steps of 0.2 tube radii, BDF2 along the tube, the flow rate held exactly at every section. The axial velocity within 0.80, 1.03 and 1.60 % of the centre speed (root mean square, worst section) at 0.20, 0.30 and 0.40 L/min, $K$ within 3.7, 3.1 and 1.2 %; 1.3 to 2.1 s per new flow on the authors' CPU, shared with other jobs.
* **Ground truth** (scoring), `MP9.1A_truth.npz`: the same march on 64 × 128 cells, steps of 0.1 radii. Across the section (16, 32, 64 rings) the largest change of the axial velocity falls from 3.1e-02 to 1.1e-02 of the centre speed, the root mean square from 7.2e-03 to 3.4e-03 (order about 1.1 to 1.4: the thin layer at the outer wall); $K$ = 1.2200, 1.2021, 1.1793, no clean order, uncertainty about 2 %. Along the tube (steps 0.4 to 0.05) the march is second order (observed 2.13). At $\delta = 0$ it keeps the parabola to 1.6e-14; in a bend 300 radii long it reaches `dean.py`'s fully developed bend (friction 2.0152 times the straight tube's) to 2.4e-08; symmetry and flow rate hold to rounding.

## Worked example (from the ground truth)

| flow | Reynolds number | loss coefficient $K$ | axial velocity on the axis: bend middle / exit / 1 D after / 5 D after / outlet |
|---|---|---|---|
| 0.20 L/min | 301 | 1.27 | 0.44 / 0.66 / 0.70 / 0.80 / 0.98 V |
| 0.30 L/min | 452 | 1.18 | 0.39 / 0.65 / 0.70 / 0.83 / 0.97 V |
| 0.40 L/min | 603 | 1.15 | 0.37 / 0.63 / 0.70 / 0.82 / 0.94 V |

The pressure gradient along the tube rises from Poiseuille's 4 to about 8 in the bend and decays back over the outlet leg. At 0.3 L/min the fastest liquid at the bend's exit is 0.84 V, pressed towards the outer wall; the axis sees 0.65 V.

## Requirements

* the axial velocity at the five sections within **2 % of the centre speed** in root mean square over each section (the largest point error reported) at 0.20, 0.30 and 0.40 L/min, which are not trained on;
* the loss coefficient $K$ within **5 %**;
* the online answer to a new flow at least **10 times faster** than the reference model on the same Colab runtime; the training and the computing of its data reported apart, as the offline cost.

**What changed against the proposal of Ex09.1.** The students no longer write the marching solver: it is supplied as the reference model, checked against both limits, and its converged form is the ground truth. The network takes the flow as an input, so that a new flow is answered online. The proposal's "within 2 % of the solver's velocity" is kept as 2 % of the centre speed in root mean square over each section: at the worst point the ground truth itself is only known to about 1 %, which a pointwise 2 % could not judge.

## What you hand in

The PDF written at the end of part E: the results table, your figures, and the ten answers.
