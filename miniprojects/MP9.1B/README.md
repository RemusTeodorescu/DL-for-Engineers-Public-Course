# MP9.1B · The Pump's Speed Reference

**Mini project of Ex09.1, Liquid Serpentine Cooling Plate · Part 2 · course policy C14**

One Colab notebook, `MP9.1B_pump_speed_reference.ipynb` (GPU runtime, T4), in five parts:

| part | given | what it holds |
|---|---|---|
| A · The problem | solved | why the pump's flow must look ahead, the cold plate as one thermal mass, the heat transfer and friction in both regimes, why a 700 W step needs turbulent flow, why there is no exact solution, what is asked |
| B · The data and the physics | solved | the network's inputs and output, its bounded last layer, the plate's energy balance integrated exactly inside the loss, the two loss terms, what can be built in, the training step powers |
| C · The reference model and the ground truth | solved | direct transcription and SLSQP step by step, convergence, energy balance, the reference model's accuracy and time, a reactive controller for contrast |
| D · The physics-informed network | TODO cells | your network, your training, your figures; a given cell evaluates it with an independent NumPy integration and times it |
| E · The report | TODO cells | the goal metrics against the ground truth and the reference model, and ten questions |

Its first cell fetches `mp91b.py`, the ground truth `MP9.1B_truth.npz` and Ex09.1's library files (`pipe.py`, `course_core.py`, `pinn_core.py`). The results pass from part to part in memory: nothing to save, mount or upload.

## The problem

The cold plate of Ex09.1 carries a power module whose power steps: 120 W, then a step of $P_\mathrm{step}$ from 1200 to 1600 s, then 150 W to 2000 s. Wanted: the flow to send to the pump at every instant, so that the module stays at or below 80 °C with the least pumping energy. **New against Ex09.1:** the tube is a heat exchanger and its flow the control input; the plate is one thermal mass, $C\,dT/dt = P - G(Q)(T - T_\mathrm{in})$; the flow may be laminar or turbulent. The optimal schedule is the solution of an optimal control problem with a state constraint and regime-switching nonlinearities: **no exact solution exists**.

| | value |
|---|---|
| plate | aluminium 250 × 130 × 12 mm, one thermal mass, 948 J/K |
| tube | 6 mm inside, 1.03 m in the plate (Ex09.1's serpentine) |
| coolant | 50/50 water–glycol, supplied at 25 °C; $c_p$ = 3400 J/(kg·K), $k$ = 0.40 W/(m·K), Pr = 21.2 (typical) |
| module | 0.05 K/W to the plate, limit 80 °C (assumed) |
| pump | 0.05 to 4 L/min (Re 75 to 6000) |
| heat transfer | Sieder–Tate below Re 2300 (thermally developing), Dittus–Boelter above 4000, a straight blend between |
| friction | 64/Re, Blasius above 4000, the same blend |
| step power | 300 to 800 W, a network input |

All values typical or assumed, and the correlations quoted from memory (Incropera & DeWitt ch. 8, White ch. 6): check them before designing with them.

## What you build

One network $Q(t, P_\mathrm{step})$ for the flow reference over the whole window and the whole range of step powers, bounded by its last layer to the pump's range, trained once on the model alone: the plate's energy balance integrated exactly inside the loss, the pumping energy as the objective and the module's limit as a penalty. The choice of network, of the training and of the figures is yours.

## The reference model and the ground truth

* **Reference model** (timed): direct transcription, the flow held on 40 s intervals (50 decisions), the plate integrated exactly between them, the limit at 4 instants per interval, SLSQP started from 2 L/min. Pumping within 0.52 % of the ground truth at the test steps, the module at 80.00 °C; 0.5 to 1.3 s per new staircase (48 to 72 SLSQP iterations) on the authors' CPU, shared with other jobs.
* **Ground truth** (scoring), `MP9.1B_truth.npz`: the same on 10 s intervals, the limit at 10 instants per interval (checked at 50). At 700 W the pumping is 61.71, 61.59, 61.34 and 61.27 J on 80, 40, 20 and 10 s intervals (observed order 1.92 over the last two halvings; Richardson 61.25 J, the ground truth 0.04 % above it); the plate's exact integration agrees with a Runge–Kutta march to 5e-07 K; the energy balance closes to 1e-03 J at every step power.

## Worked example (from the ground truth)

| step power | least pumping | most flow (Reynolds number) | flow 60 s before the step |
|---|---|---|---|
| 300 W | 0.98 J | 0.32 L/min (488, laminar) | 0.14 L/min |
| 550 W | 28.32 J | 2.10 L/min (3159, transition) | 0.47 L/min |
| 700 W | 61.27 J | 2.10 L/min (3158, transition) | 0.44 L/min |
| 800 W | 123.56 J | 2.50 L/min (3762, transition) | 0.61 L/min |

Under the 120 W base the flow sits at 0.05 L/min; **minutes before the step it starts to rise** (at 700 W: 0.09 L/min at 600 s, 0.19 at 900 s, 0.38 at 1100 s) so that the plate is cool when the power arrives; during the step it holds the steady flow and the module sits at 80.00 °C. A controller that sees only the present power reaches 86.52 °C at 300 W. Laminar flow holds about 460 W at most; the least steady flow has a knee near 500 W, where the turbulent heat transfer takes over.

## Requirements

* the module at most **80.2 °C** under your schedule at the step powers 350, 550, 650 and 750 W, which are not trained on (integrated exactly by the given evaluation);
* the pumping energy within **5 %** of the ground truth's at the same steps;
* the flow raised before the step;
* the online answer to a new staircase at least **10 times faster** than the reference model on the same Colab runtime; the training reported apart, as the offline cost.

**What changed against the proposal of Ex09.1.** The proposal's two cases, a 300 W and a 700 W step, are now one range of step powers, 300 to 800 W, and the step power is the network's input: a new staircase is answered by one evaluation, which is what makes the online comparison with the reference model fair. The requirements are the proposal's.

## What you hand in

The PDF written at the end of part E: the results table, your figures, and the ten answers.
