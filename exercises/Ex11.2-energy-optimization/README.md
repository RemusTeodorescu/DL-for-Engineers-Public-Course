# Ex_11.2 — Dynamics and Energy: making the policy efficient

**Paired with L11.2 · Dynamics and Energy · Part 2**

Ex_11.1 asked whether the car could drive the course. This asks what it cost.
You identify the vehicle's dynamics and electrical parameters from measured
data — an inverse problem, the same shape as Ex_08.2's — predict the energy of
a lap **before** you run it, then optimise the speed profile.

Prerequisite: a working Ex_11.1 policy and its submission file.

## The two notebooks, and why there are two

`Ex11.2_10_energy_optimization.ipynb` runs **on the car**. It identifies the
parameters with `scipy.optimize.least_squares` and finds the speed profile with
`scipy.optimize.minimize`. For four parameters and a one-dimensional profile
those are the right tools, and this is the notebook you are scored on.

`Ex11.2_20_energy_optimization_pinn.ipynb` solves the **same optimisation** as a
neural network: the profile becomes a function of arc length, the grip limit
becomes a layer, and autograd supplies the gradient. Run it off the car, after
notebook 10, using the parameters you identified there.

The comparison is the point. Expect the network to land a few percent worse and
to take longer, and expect the grip ceiling never to be violated. The report asks
you to name one change to the problem that would reverse the verdict.

## Goals

By the end you can

1. configure a current sensor deliberately — averaging, conversion time — and
   justify the configuration against the switching frequency it is measuring;
2. identify vehicle parameters from a coast-down, and **report as lumped what
   the data cannot separate**;
3. state a prediction before the measurement that would confirm or refute it,
   and be scored on the gap;
4. find the energy-optimal lap time and explain why the minimum is interior
   rather than at either extreme;
5. distinguish traction energy from hotel load, and show which one puts the
   minimum where it is.

## The problem

Total lap cost is

```
J = E_lap + w · T_lap,     w = 5 J/s fixed
```

and there are two scores: efficiency `η = J₀ / J`, and model fidelity
`φ = E / (E + |E_pred − E|)` with the prediction **timestamped before** the run.
The second is the one that rewards understanding rather than tuning.

Driving slowly wastes energy on the hotel load — Nano, camera, OLED — for
longer; driving quickly wastes it on drag and on braking. The minimum sits
between, and it moves when the hotel load changes. That is why the shunt has to
see the **whole** vehicle rather than the traction branch alone: wire it wrong
and the interior minimum disappears from your data entirely.

## The notebook

```
Ex11.2_10_energy_optimization.ipynb   the whole exercise, run on the car
```

Sections: the pack you actually have, the INA219 configured deliberately, the
instrumented loop, the coast-down, parameter identification, the energy-optimal
speed profile, the committed prediction, and the scored run.

## Files

| | |
|---|---|
| `course_core.py` | shared by the whole course |

Self-contained otherwise: the INA219 and JetRacer libraries live on the car.
`course_core.py` is generated — edit `tools/pinn/course_core.py` and run
`python3 tools/pinn/sync_cores.py`.

## What to hand in

- both indices, with the prediction's timestamp
- your identified parameters, stated as lumped quantities where the data cannot
  separate them
- how you measured speed, given there is no encoder
- the energy-versus-lap-time curve with your chosen operating point marked

## Things that go wrong, and what they mean

**Your energy trace is noise.** The drive is switched at kilohertz and you
sampled at 20 Hz with a short conversion window, so each sample landed wherever
it liked in the PWM cycle. Set on-chip averaging so each reading integrates over
many switching periods, and record the configuration.

**You cannot separate the torque constant from the gearing.** You cannot. The
data does not distinguish them — fit the lumped constant, report it as lumped,
and say which physical parameters it contains.

**The hotel load is invisible to your shunt.** Then the shunt is in the wrong
place, and the interior minimum in the energy-versus-lap-time curve disappears
with it.

**There is no encoder and you assumed one.** The Pro has none. Speed comes from
segment timing, optical flow, or a sensor you add — state which, because it
bounds everything downstream.

## Expected runtime

An afternoon on the car. The coast-down is ten minutes of measurement and the
identification fits in seconds; the scored runs take the rest.

## Reference texts

Raissi, Perdikaris & Karniadakis, *Physics-informed neural networks*,
J. Comput. Phys. **378** (2019) 686–707 — for the inverse-problem formulation.
Prince, S.J.D., *Understanding Deep Learning* (MIT Press, 2023), Ch. 6.

These are the works to read for the theory. **The code, the problem and the
exposition in this exercise set are original to this course** and are not
derived from any publisher's code listings.

## Before this is assigned

The pack is 2S2P at 8.4 V and no motor count is claimed — earlier drafts quoted
the plain JetRacer's figures. **Nothing in this set has been executed on a car
this term.**
