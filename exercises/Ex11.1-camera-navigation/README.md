# Ex_11.1 — Vision-Based Navigation: driving a course from pixels

**Paired with L11.1 · Vision-Based Navigation · Part 2**

The first exercise in the course where the model has to act. Everything up to
here was judged against a stored answer; here it is judged against a car that
either makes the corner or does not. You collect your own dataset, train a
regression from image to steering target, and drive a scored lap.

**This runs on the car, not on Colab.** Before you open the notebook the
Waveshare image must be flashed, the Nano must be in 5 W mode, and the stock
motion notebook must turn the wheels. Debugging a network on a car that has
never driven is a wasted afternoon — do `SETUP.md` first.

## Goals

By the end you can

1. collect a labelled dataset from a physical sensor and say honestly what it
   does and does not cover;
2. train an image-to-steering regression and evaluate it on **held-out** data
   rather than on the frames it was fitted to;
3. instrument a control loop — measure its rate, and convert that rate into
   the blind distance `d = v/f` the car travels between decisions;
4. design and demonstrate a failsafe, and treat it as a gate rather than a
   nicety;
5. report a scored result with its uncertainty, from the median of repeated
   clean laps rather than the best one.

## The problem

The car is a Waveshare JetRacer Pro: a Jetson Nano, a wide-angle camera, a
steering servo and a 2S2P 18650 pack at 8.4 V. There is **no encoder**, which
constrains every speed-dependent claim you can make downstream.

The score is the Navigation Index

```
N = T₀ / (T̄ + 2c)
```

with `T̄` the median of three clean laps and `c` the penalty per contact. The
blind distance enters as a hard cap: if `d = v/f` exceeds half the narrowest
clearance on the course you are capped at `N = 1.00` however fast you drove.
Raise the loop rate or slow down — those are the only two levers the algebra
allows.

## The notebook

```
Ex11.1_10_camera_navigation.ipynb    the whole exercise, run on the car
```

It is one notebook rather than six because the car is the bottleneck: the
sections are hardware check, data collection, training with a held-out split,
the instrumented control loop, the failsafe gate, and the scored run.

## Files

| | |
|---|---|
| `SETUP.md` | flash, configure and verify the car — **do this first** |
| `course_core.py` | shared by the whole course |

`course_core.py` is generated: edit `tools/pinn/course_core.py` and run
`python3 tools/pinn/sync_cores.py`.

## What to hand in

- your Navigation Index, median of three clean laps
- your measured loop rate and the blind distance that follows from it
- evidence the failsafe works
- held-out validation error, not training error

## Things that go wrong, and what they mean

**`jetracer` imports but the wheels do not turn.** You have NVIDIA's motor code
rather than Waveshare's. Remove the `jetracer` folder and reinstall the
Waveshare version.

**The car resets under load.** 5 W power mode is not set, and it does not always
survive a reboot. Re-run the hardware check after *every* restart.

**The car follows the racing line and cannot recover.** Your dataset contains
only the racing line. Drive off-line deliberately and label the correction —
that is where the useful gradient lives.

**Your validation error is suspiciously low.** Consecutive frames are nearly
identical, so a random split leaks. Hold out whole runs, not whole frames.

## Expected runtime

An afternoon on the car. Data collection is twenty minutes, training two to
five minutes on the Nano, and the rest is driving.

## Reference texts

Prince, S.J.D., *Understanding Deep Learning* (MIT Press, 2023), Ch. 10.
Goodfellow, Bengio & Courville, *Deep Learning* (MIT Press, 2016), Ch. 9.

These are the works to read for the theory. **The code, the problem and the
exposition in this exercise set are original to this course** and are not
derived from any publisher's code listings.

## Before this is assigned

The scoring algebra and the blind-distance cap are checked by hand. **Nothing
in this set has been executed on a car this term** — one JetRacer must complete
notebook 10 end to end before it goes to students.
