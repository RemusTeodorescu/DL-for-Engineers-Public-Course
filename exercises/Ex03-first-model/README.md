# Ex03 — Your First Model

**Paired with lecture block L3 · Part 1**

Predict how a vibrating mass moves, four ways — a textbook formula, a
regression, a neural network, and a PINN — and discover that the model which
fits the measurements best is the worst one.

## Goals

By the end of this exercise a student can

1. compute a system's natural frequency and damping ratio from its physical
   constants, and say what those two numbers predict about its behaviour;
2. linearise an exponential envelope and recover a physical constant by least
   squares — the logarithmic decrement, as used on real structures;
3. train a network on measurements alone, and demonstrate that it has fitted
   sensor noise and sensor bias along with the signal;
4. add a differential equation to a loss function, and explain what collocation
   points and initial conditions each contribute;
5. choose a physics weight from evidence, and argue for that choice in a case
   where no ground truth is available;
6. state, with their own numbers, why training error is not a measure of whether
   a model is right.

## State

**Written.** One notebook and one module. No solutions notebook; the `# TODO`
blocks are the exercise.

Replaces an earlier five-notebook version built on exponential cooling. That
system was too smooth to make the case — a first-order monotonic decay is fitted
almost perfectly by a plain network, so the PINN had nothing to demonstrate. A
second-order oscillator does, and it also makes extrapolation fail visibly
rather than subtly.

## The notebook

```
Ex03_vibrating_mass.ipynb    the whole exercise, about 90 minutes
Ex_3_core.py                 data, network, training loop — complete
```

One notebook, deliberately. Every notebook opened in Colab runs on its own
machine, so a set that passes files between notebooks cannot work there without
mounting Drive. Nothing here is written to disk except the final report, which
downloads through the browser.

**Running it in Colab.** Open the notebook from the badge at its top and run the
first code cell: it fetches `Ex_3_core.py` from the public course repository if
the file is not already beside the notebook. That is the whole setup - no Drive
mount, no clone, no upload. If the runtime is recycled the fetched file goes
with it, and the same cell fetches it again.

## The system

A mass on a spring with a damper, released from 50 mm and left to ring:

```
m x'' + c x' + k x = 0        m = 0.5 kg, k = 20 N/m, c = 0.215 Ns/m
                              omega_n = 6.32 rad/s, period ~1 s, zeta ~ 0.034
```

Five seconds of measurements at 50 ms — 101 samples — and predictions wanted out
to twelve seconds. More than half the answer lies beyond the data.

Two things are deliberately wrong with what the student is given, and both drive
the argument.

**The catalogue damping is 13% high.** `SYSTEM["damping_Ns_m"]` is 0.215; the
real value in `_C_TRUE` is 0.190. Dampers are made to a tolerance and catalogue
figures are measured somewhere that is not your laboratory. The physics is right
and the constants are only approximately right, which is the normal engineering
situation.

**The sensor is noisy and was never zeroed.** 1.0 mm of noise plus a constant
1.5 mm offset. The bias is the more interesting of the two: it damages the
regression specifically, because lifting every peak by a fixed amount flattens
the apparent decay and makes the system look less damped than it is.

Everything is generated from a fixed seed, so every student fits the same
numbers. `Ex_3_core.truth` holds the real motion and exists only for scoring.

## The arc

| | model | knows | fitted parameters |
|---|---|---|---|
| **1** | closed-form solution | physics + catalogue constants | 0 |
| **2** | logarithmic decrement | the measurements | 2 |
| **3** | neural network | the measurements | 9 473 |
| **4** | PINN | measurements + equation + initial conditions | 9 473 |
| **5** | PINN with `w_data = 0` | equation + initial conditions only | 9 473 |

Models 3 and 4 use **the same network**. Only the loss differs, so any
difference in the result is attributable to the physics rather than to
architecture.

## The measured result

| model | error inside the data | error outside |
|---|---|---|
| 1 — analytic | 1.261 mm | 1.342 mm |
| 2 — regression | 3.065 mm | 3.034 mm |
| 3 — network | 1.762 mm | **30.256 mm** |
| 4 — PINN | **0.355 mm** | **0.590 mm** |
| 5 — PINN, no measurements | 1.300 mm | 1.407 mm |

Four things fall out of this table, each supported by a number the student
generates.

**The network fits the measurements better than they deserve.** It agrees with
them to 0.618 mm, while the measurements are wrong by 1.803 mm. It has fitted
the noise and the bias, because nothing told it which parts of the data to
believe.

**And it is the worst model** — fifty times worse than the PINN outside the
data. Training error cannot reveal this.

**The PINN declines to fit the errors.** It disagrees with the measurements by
1.551 mm, against a measurement error of 1.803 mm. It fits the data *worse* and
is closer to the truth. Physics in the loss is not only about extrapolation; it
is a statement about which parts of a dataset to trust.

**Model 5 lands where model 1 does**, for the same reason: both used the
catalogue damping and neither had anything to correct it with. That is the
closing argument — the measurements in model 4 never taught the network the
*shape* of the motion, because the physics did that. What they contributed was a
correction to a constant.

## The report

Five questions, answered in the student's own words and assembled into
`Ex03_report.md`, which downloads through the browser. They are about
interpretation rather than recall: how a model can fit best and be worst, which
models the sensor bias damaged, what the measurements contributed given that
model 5 worked without them, how to choose a physics weight when no truth is
available, and when a PINN is not worth the trouble.

## Two traps, solved in the module

Both are commented in place, because both are the usual reason a first PINN
disappoints — and a student who meets either unaided will conclude, correctly
from the evidence in front of them, that PINNs do not work.

**The residual is divided by `omega_n**2`.** Undivided, the physics term is
roughly forty times the data term, the measurements are drowned out, and the
model underfits everything.

**All three loss terms are divided by the amplitude.** This makes the weights
scale-free: `w_physics = 10` means the same thing whether the mass swings 50 mm
or 50 m. Without it the weights need retuning for every new problem, silently.

A third is handled rather than commented away. `core.FourierFeatures` gives the
network sin and cos of a spread of fixed frequencies, because a tanh network is
strongly biased towards smooth functions and will relax to a flat line outside
its data whatever the physics says — a property of the network that would
otherwise be blamed on the method. The frequencies are a spread, not the
system's own, so the answer is not smuggled in; the highest must exceed
`omega_n`, which is why `n_features` is 40.

## Conventions

Same as every Part 2 exercise set:

- Self-contained folder. Requires `torch`, `numpy`, `matplotlib` — all
  preinstalled on Google Colab. No GPU needed.
- `Ex_3_core.py` is complete and is **not** to be rewritten by students. The
  work is in five `# TODO` blocks, each with the answer hinted on the same line
  and a guard below that stops with an instruction if left unfilled.
- Module names use underscores because a Python module name cannot contain a
  dot: `Ex_3_core.py`, imported as `Ex_3_core`.
- Every TODO is preceded by enough prose that the notebook works for a student
  who missed the lecture. Part 1 exercises are deliberately more discursive than
  the Part 2 ones, and this one assumes very little prior Python.
- British spelling throughout.

## Depends on

- **L3.1** slide 17 (the modelling spectrum) and slide 25 (this exercise).
- **L3.2** slide 2 (the four questions) and slide 24 (this exercise, restated).

**Both cross-references need checking against the rewrite.** The slides describe
three models on a cooling curve; this is four models on an oscillator, and the
modelling spectrum now has a fourth position on it. Feeds **L4.1 slide 23**,
which explains the extrapolation figure.

## Expected runtime

CPU only. About five minutes of compute: four training runs of 6 000 epochs at
roughly thirty seconds each, plus a four-point weight sweep. The difficulty is
conceptual, not computational.
