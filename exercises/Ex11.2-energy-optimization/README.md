# Ex_11.2 — The Cheapest Lap

**Paired with L11.2 · Dynamics, Energy and Efficient Driving · Part 2**

Ex_11.1 asked whether the car could drive the course. This set asks what a lap
costs. It has **one study notebook** (course policies C11 and C13), which runs
on Colab or a laptop, and **one on-car notebook**, which is the hardware
procedure of the competition and runs only on the JetRacer.

```
Ex11.2_pinn_cheapest_lap.ipynb         the study notebook: two TODO cells, the answer in the comment above each
Ex11.2_pinn_cheapest_lap_light.ipynb   the same notebook with every cell written out
Ex11.2_on_the_car.ipynb                on the car: the sensor, the coast-down, the identification, the scored laps
```

## The problem

Which speed at every point of a closed course makes a lap cost least,

```
J = E + w T,        w = 5 J/s fixed by the competition
F = c_rr m g + ½ ρ C_d A v² + m v dv/ds        the force the motor supplies
i = max(F/K, 0)                                no regeneration
P = i² R_a + K i v + P_hotel                   winding heat, traction, electronics
v ≤ min(√(μg/κ), v_max)                        the grip limit                 (built into the network)
```

Driving slowly pays the electronics for longer; driving fast pays in winding
heat and in the kinetic energy thrown away before each corner.

**Every number of the car in the study notebook is a placeholder of the right
order, not a measurement**: 1.05 kg, μ = 0.65, c_rr = 0.02, K = 1.0 N/A,
R_a = 1.5 Ω, C_d A = 0.02 m², 7 W of electronics, a top speed of 3.0 m/s, and an 18 m course of
three corners. They are in `problem.py`, marked as placeholders, and the
notebook says so where it prints them. The on-car notebook measures your own;
no result of the study notebook is your car's until you have put them in.

## The study notebook

| section | what the student does |
|---|---|
| 1 – 3 | the problem, its data, its physics: the force, the power, no regeneration, the grip limit |
| 4 | the reference and the classical optimiser: one unknown per step, L-BFGS-B with bounds, the step halved four times, for an agreed 1 % of the cost |
| 5 | the network: the speed as a function of arc length, the grip limit as its last layer (TODO 1), the lap cost in torch (TODO 2); the two compared |
| 6 | the inverse problem: the rolling coefficient from a coast-down, by least squares and by a physics-informed fit with one more trainable number |
| 7 | what the notebook says |
| 8 | the report and its PDF |
| 9 | mini project proposal |

It replaces the earlier `Ex11.2_20_energy_optimization_pinn.ipynb`.

### What it measures (CPU, seed 88, the placeholder car)

| | energy | lap time | cost J | against the reference | computed in |
|---|---|---|---|---|---|
| reference, step 0.0625 m (288 unknowns) | 55.99 J | 6.50 s | 88.50 J | — | about 10 s |
| classical optimiser, step 0.25 m (the coarsest for 1 %) | 55.60 J | 6.49 s | 88.03 J | −0.53 % | under 1 s |
| network, 3 × 32, six harmonics, grip limit built in | 55.66 J | 6.48 s | 88.08 J | −0.48 % | about 3 s |

The network is 0.05 % above the classical optimiser on the same step and is
never above the ceiling. The coast-down returns c_rr = 0.0200 by least squares
and 0.0199 by the physics-informed fit, against the 0.0200 the synthetic log
was made with.

**Two things worth knowing.** With 7 W of electronics and 5 J for every
second, time dominates the cost, so the best lap is close to the fastest the
tyres allow: the interior minimum the lecture describes needs a smaller hotel
load or a smaller w to show clearly. And the earlier notebook 20 told
students to expect the network a few per cent worse than SciPy, because a
sigmoid only approaches its ceiling; with six harmonics as inputs and L-BFGS
it is 0.05 % worse. The earlier notebook also wrote the traction power as
F·v with F allowed to be negative, which credits braking as regeneration; the
power here is K·i·v with the current clamped at zero.

## The on-car notebook

`Ex11.2_on_the_car.ipynb` is unchanged in what it does: it configures the
INA219, measures the hotel load, logs a coast-down, identifies the
parameters with `scipy.optimize.least_squares`, commits an energy prediction
before the run, and scores three clean laps on two boards. It needs the car,
the Waveshare image and a working Ex_11.1 policy; it has no light form.
**The car has no encoder**, so its speed comes from lap timing or from the
camera, which bounds every identified parameter.

## Files

| | |
|---|---|
| `course_core.py` | shared by the whole course — `set_seed`, `MLP`, `to_tensor`, `check` |
| `pinn_core.py` | the training helpers — `grad`, `train_two_stage` |
| `problem.py` | **this** problem — the placeholder car, the course, the lap cost (NumPy or torch), the classical optimiser, the synthetic coast-down |

The first two are generated. Edit `tools/pinn/*.py` and run
`python3 tools/pinn/sync_cores.py`; never edit a copy.

**The study notebook is generated too**, both forms from one source,
`tools/exercises/ex112/build_ex112.py`. Edit the builder and rerun it rather
than editing a notebook. The on-car notebook is edited by hand.

## Expected runtime

The study notebook: about twenty seconds on a CPU, half of it the reference on
its finest step. The on-car notebook: one lab session.

## Reference texts

Liu, G.R., *PINN with Python: An Introduction* (2025).
Raissi, Perdikaris & Karniadakis, *Physics-informed neural networks*,
J. Comput. Phys. **378** (2019) 686–707.

These are the works to read for the theory. **The code, the problem and the
exposition in this exercise set are original to this course.**

## Before this is assigned

The light version of the study notebook has been executed end to end on a
local CPU; the exercise version stops at its TODO cells. Still to do: a run
from a fresh Colab runtime, a review by someone other than the author, and -
as before - a first run of the on-car notebook on a car.

## Mini project proposal

The set ends with two mini projects (section 9). Each student chooses one
mini project from the Part 2 sets and solves it individually in one month. The
ground truth is given, built by `tools/miniprojects/ex112_truth.py` under policy C10
(`COURSE_POLICIES.md`), with a worked example of each.

| | the problem | the deep learning | the ground truth given | required |
|---|---|---|---|---|
| **MP11.2A · The energy-optimal lap, with the battery** | a fixed 7.20 m lap; grip-limited turns, copper loss, the pack's sag | section 5's network with the pack, the grip limit as a layer | direct collocation, 50 to 200 nodes; worked example: the cheapest lap is 3.4 s at 26.16 J; the pack adds under 1 % | energy within 2 %, the cheapest lap within 0.1 s, the curve in under 1 s |
| **MP11.2B · What the model misses** | an extra low-speed friction the four-parameter model does not have; coast-downs from 1, 2, 3 m/s | a neural ODE: the vehicle model plus a small network for what it misses | simulated coast-downs, noise 0.02 m/s; worked example: the model misses up to 0.081 m/s, at low speed only | extra loss within 20 %, parameters within 10 %, speed within 0.02 m/s |
