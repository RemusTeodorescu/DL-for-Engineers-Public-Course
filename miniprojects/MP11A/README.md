# MP11A · Faster from A to B

**Mini project of Ex11, Vision-Based Navigation · Part 2 · course policy C14**

One Colab notebook, `MP11A_faster_a_to_b.ipynb` (GPU runtime, T4), in five parts:

| part | given | what it holds |
|---|---|---|
| A · The problem | solved | why a car must say where an obstacle stands, the ROSMASTER R2 twin (what the vendor states, what is assumed), the track and its four kinds of obstacle, the five noise levels, how the car drives, what is asked |
| B · The data | solved | the network's input and outputs, the two loss terms, what the flat floor gives, 14 400 training frames and 1 200 held-out frames rendered from the twin |
| C · The reference model and the ground truth | solved | the conventional camera rule step by step, the perfect-perception lap and its convergence, the twin's physics checks, the rule's accuracy, time and closed-loop laps, a film |
| D · The network | TODO cells | your network, your training, your figures; a given cell scores it on the held-out frames, drives the 10 test tracks with it and times one decision |
| E · The report | TODO cells | the goal metrics against the ground truth and the reference model, and eighteen questions |

Its first cell fetches `mp11a.py`, the twin `r2twin.py`, the ground truth `MP11A_truth.npz` and Ex11's library files (`problem.py`, `course_core.py`, `pinn_core.py`). Everything runs on the digital twin: no real car and no ROS (the lecturer, 27 October 2026). The results pass from part to part in memory.

## The problem

The Yahboom ROSMASTER R2 drives one lap of a closed track at 1.5 m/s, deciding ten times a second from **its colour camera alone**: is the path blocked, and **where** does the obstacle stand. A planner (Ex11's potential field, given) steers round what is reported. **New against Ex11:** the car (the R2's twin, `r2twin.py`, shared with MP11B), a closed track with bends, a chicane and an S-bend instead of a straight line, 4 to 8 obstacles of four kinds, five seeded noise levels, and the position of the obstacle, which Ex11's race took from the course.

| | value |
|---|---|
| car | ROSMASTER R2: Ackermann steering, 65 mm wheels, at most 1.8 m/s (vendor); wheelbase 0.23 m, width 0.24 m, steering limit 25 degrees, speed lag 0.3 s, friction 0.6 (ASSUMED) |
| camera | Astra Pro Plus colour camera, 66.1 × 40.2 degrees (vendor), used at 112 × 64 pixels; 17 cm high, 10 degrees down (ASSUMED) |
| localisation | odometry and IMU (vendor), exact in the twin: the car knows the track, not the obstacles |
| track | closed, about 21 m, 1.1 m wide between taped edges, chicane jog 0.55 - 0.75 m, tightest radius about 1 m; drawn from a seed |
| obstacles | 4 - 8, at least 1.6 m apart, radius 7 - 10 cm: dark `box` 30 cm, `grey` (the floor's colour) 30 cm, `low` 10 cm, glossy `black` 30 cm |
| blocked | an obstacle's centre between the front axle and 1.5 m beyond it, within 0.55 m plus its radius of the car's axis, in the camera's view |
| noise levels (ASSUMED) | 0 clean lab … 4 extreme: light 1 to 0.35, pixel noise 1 to 7 %, 0 to 5 shadows, 0 to 4 sun patches, motion blur up to 2.4 rows per m/s |
| drive | 10 decisions a second, one period late; sideways acceleration under 1 m/s² in the bends |

## What you build

One convolutional network: a 64 × 112 frame in, the logit of *blocked* and the obstacle's position (forward, sideways, metres) out - Ex11's network with a position head. It is trained on frames rendered from 120 tracks at noise levels 0 to 2 and drives the potential field. The design, training and figures are yours.

## The reference model and the ground truth

* **Reference model** (conventional vision, timed): Ex11's dark-pixel rule on the R2's frame (pixels 0.15 darker than the frame's median floor in the zone; blocked above 0.4 % of it), the obstacle's foot read off the flat floor. On the 900 held-out frames of levels 0 - 2 it misses 92.5 % of the grey boxes, 8.2 % of the black ones and 3.5 % of the dark boxes, and stops for 6.7 % of the frames with only a shadow ahead; it places what it finds within 3 to 9 cm (7 to 26 cm at level 3). In closed loop on the 10 test tracks at levels 0 - 2 it collides on 23 of 30 laps with Ex11's steer-around rule at 1.0 m/s and on 21 of 30 with the field at 1.5 m/s, nearly always on a grey box. One decision takes 0.11 to 0.15 ms on a CPU.
* **Ground truth** (`MP11A_truth.npz`): the twin's geometry (labels exact) and one lap of each test track with perfect perception: potential field at 1.5 m/s, mean lap **18.13 s**, smallest gap 9.7 cm, no collision; steer-around at 1.0 m/s, 28.54 s. The time step converges: mean laps 18.1345, 18.1254, 18.1243 and 18.1229 s at 4, 2, 1 and 0.5 ms (changes 9e-03, 1e-03, 1.4e-03 s: no regular order, since the decisions every 0.1 s make the lap a non-smooth function of the step; the 2 ms used is 2.4 ms from the finest). Physics checks: with the JetRacer's numbers the R2's motion is Ex11's car exactly (0 m: the limiting case that reproduces the exercise); a noiseless LiDAR range and depth reading match the exact distances to 2e-08 and 5e-08 m; an empty lap at 1 m/s takes 0.6 % more than L/v (pure pursuit is not exactly on the centre line).

## Worked example (from the ground truth)

Test track 1000 (lap 20.91 m, three dark boxes and a low block): perfect perception with the field at 1.5 m/s laps it in 17.69 s. Over the 10 test tracks, 16.39 to 19.97 s, mean 18.13 s, smallest gap 9.7 cm.

## Requirements

* **no collision** on the 10 test tracks (seeds 1000 - 1009, never trained on) at noise levels 0, 1 and 2, the potential field at 1.5 m/s (30 laps);
* the mean lap within **5 %** of perfect perception's 18.13 s;
* on 900 held-out frames at levels 0 - 2: at most **5 %** missed of **every** kind, at most **5 %** false stops (frames with an obstacle at the zone's edge left out);
* the mean position error at most **10 cm** for every kind;
* one decision at most **10 ms** on the Colab T4 (a tenth of the 100 ms between decisions).

Noise level 3 (never trained on) is reported, not required. The network is not faster than the dark-pixel rule (a few array operations); the comparison asked is the per-decision time of both, and the requirement is the loop's budget.

## What you hand in

The PDF written at the end of part E: the results table, your figures, and the eighteen answers.
