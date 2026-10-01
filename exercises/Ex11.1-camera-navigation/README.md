# Ex_11.1 — Steering from a Frame

**Paired with L11.1 · Vision-Based Navigation · Part 2**

The first exercise in the course where the model has to act. The set has
**one study notebook** (course policies C11 and C13), which runs on Colab or a
laptop, and **one on-car notebook**, which is the hardware procedure of the
competition and runs only on the JetRacer.

```
Ex11.1_cnn_steering_from_a_frame.ipynb         the study notebook: two TODO cells, the answer in the comment above each
Ex11.1_cnn_steering_from_a_frame_light.ipynb   the same notebook with every cell written out
Ex11.1_on_the_car.ipynb                        on the car: hardware check, collection, training, the loop, the scored laps
SETUP.md                                       flash, configure and verify the car - do this before the on-car notebook
```

## The problem

A car follows a line of tape. Its camera gives a frame; the policy returns
where the line is a little way ahead, one number between −1 and +1, and the
steering follows from it. Between two decisions the car is blind for

```
d = v / f          speed over decisions a second
```

**The study notebook's frames are synthetic**: 48 × 64 grey pixels drawn by
`problem.py` from the car's offset, its heading and the curve of the road,
with a brightness slope, pixel noise and, on some frames, a patch of glare.
That is what makes the true target known. **The car in its section 6 is
assumed too**: a bicycle model with a wheelbase of 0.20 m and a steering limit
of 30° on a stadium track of 7.20 m, none of it measured. The on-car notebook
is where real frames and a real loop rate come from.

## The study notebook

| section | what the student does |
|---|---|
| 1 – 3 | the problem, its data, the geometry of the loop: the target, the steering it implies, the blind distance |
| 4 | the reference (the true target) and the classical detector - bright pixels in a band of rows - at four resolutions, for an agreed mean error of 0.03, one pixel |
| 5 | a convolutional network (TODO 1) and its training step (TODO 2), trained on frames collected along the line and on frames collected deliberately; the three compared |
| 6 | the loop rate: a simulated lap at eight combinations of speed and decisions a second; then the network's weights stored in 8 to 2 bits |
| 7 | what the notebook says |
| 8 | the report and its PDF |
| 9 | mini project proposal |

It replaces `Ex11.1_01_quantisation.ipynb`, the quantisation notebook that
came from Ex06 on 22 September 2026. Of that notebook, the rounding of a
regression's weights is kept (section 6); PyTorch's dynamic quantisation of a
classifier, its comparison with a narrower float model, and the
signal-to-noise line are not.

### What it measures (CPU, seed 88)

| mean error on 1000 held-out frames | along the line | beside the line | with glare | one frame | training |
|---|---|---|---|---|---|
| classical detector, 12 × 16 pixels (the coarsest for 0.03) | 0.024 | 0.026 | 0.369 | 0.04 ms | — |
| network, 55 233 weights, trained along the line | 0.002 | 0.136 | 0.149 | 0.19 ms | 3 s |
| the same network, trained deliberately | 0.006 | 0.008 | 0.009 | 0.19 ms | 3 s |

The simulated car holds the line within about 2 cm for a blind distance up to
10 cm, is 3 cm off at 13 cm and leaves the course at 17 cm; 1 m/s at 10 Hz and
2 m/s at 20 Hz give the same result. Weights stored in 8, 6, 5, 4, 3 and 2
bits give errors of 0.009, 0.008, 0.013, 0.021, 0.149 and 0.234. The timings
are a laptop's, not the Nano's.

## The on-car notebook

**This runs on the car, not on Colab.** Before you open it the Waveshare image
must be flashed, the Nano must be in 5 W mode, and the stock motion notebook
must turn the wheels - do `SETUP.md` first. Its stages are the hardware check,
data collection, training with a held-out split, the instrumented control
loop, the failsafe gate, and the scored run. It has no light form.

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

### What to hand in from the car

- `SUBMISSION_Ex11.1_<team>_car<NN>.json`, `logs/latency.json`, `logs/training.json`, `model_best.pth`
- two paragraphs: what limited you - data, latency, or grip - and how you know
- the measured loop rate and the blind distance at your speed
- held-out validation error, not training error

### Things that go wrong, and what they mean

**`jetracer` imports but the wheels do not turn.** You have NVIDIA's motor code
rather than Waveshare's. Remove the `jetracer` folder and reinstall the
Waveshare version.

**The car resets under load.** 5 W power mode is not set, and it does not always
survive a reboot. Re-run the hardware check after *every* restart.

**The car follows the racing line and cannot recover.** Your dataset contains
only the racing line. Drive off-line deliberately and label the correction -
section 5 of the study notebook shows what that is worth.

**Your validation error is suspiciously low.** Consecutive frames are nearly
identical, so a random split leaks. Hold out whole runs, not whole frames.

## Files

| | |
|---|---|
| `SETUP.md` | flash, configure and verify the car |
| `course_core.py` | shared by the whole course — `set_seed`, `to_tensor`, `parameter_count` |
| `pinn_core.py` | the Part 2 helpers; the study notebook uses only its imports |
| `problem.py` | **this** problem — the frame generator, the classical detector, the simulated lap, the rounding of weights |

The two cores are generated. Edit `tools/pinn/*.py` and run
`python3 tools/pinn/sync_cores.py`; never edit a copy.

**The study notebook is generated too**, both forms from one source,
`tools/exercises/ex111/build_ex111.py`. Edit the builder and rerun it rather
than editing a notebook. The on-car notebook is edited by hand.

## Expected runtime

The study notebook: about fifteen seconds on a CPU. The on-car notebook: an
afternoon - data collection twenty minutes, training two to five minutes on
the Nano, and the rest is driving.

## Reference texts

Prince, S.J.D., *Understanding Deep Learning* (MIT Press, 2023), Ch. 10.
Goodfellow, Bengio & Courville, *Deep Learning* (MIT Press, 2016), Ch. 9.

These are the works to read for the theory. **The code, the problem and the
exposition in this exercise set are original to this course** and are not
derived from any publisher's code listings.

## Before this is assigned

The light version of the study notebook has been executed end to end on a
local CPU; the exercise version stops at its TODO cells. Still to do: a run
from a fresh Colab runtime and a review by someone other than the author.
The scoring algebra and the blind-distance cap of the on-car notebook are
checked by hand, but **nothing in it has been executed on a car this term** -
one JetRacer must complete it end to end before it goes to students.

## Mini project proposal

The set ends with two mini projects (section 9). Each student chooses one
mini project from the Part 2 sets and solves it individually in one month. The
ground truth is given, built by `tools/miniprojects/ex111_truth.py` under policy C10
(`COURSE_POLICIES.md`), with a worked example of each.

| | the problem | the deep learning | the ground truth given | required |
|---|---|---|---|---|
| **MP11.1A · Where will the car be?** | predict the car's position and heading one second ahead on the 3 × 2 m track, from its commands | a network constrained by the kinematic bicycle model, commands as input | the bicycle model on the stadium track; worked example: a wheelbase 5 % wrong puts the car 2.82 cm off after 1 s | 2 cm and 2° one second ahead; under 5 ms on the Nano |
| **MP11.1B · How late does the car react?** | the loop's delay and the servo's lag, hidden, from logged commands and heading | an inverse model: bicycle kinematics with a trainable delay and lag | simulated laps, heading noise 0.5°; worked example: a weave separates them 6 times better than a smooth lap | delay and lag within 10 ms |
