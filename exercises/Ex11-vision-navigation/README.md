# Ex_11 — Vision-Based Navigation, from A to B

**Paired with L11.1 · Vision-Based Navigation and L11.2 · Obstacle Avoidance · Part 2**

One notebook for both lectures (course policies C11 and C13), as Ex08 is for
L8.1 and L8.2. **It runs on Colab or a laptop and needs no car**: it drives a
**virtual JetRacer**, a digital twin of the lab's Waveshare JetRacer Pro with
the same software interface - `car.steering` and `car.throttle` from −1 to 1,
with the JetRacer library's `steering_gain`, `steering_offset` and
`throttle_gain`, and `camera.read()` returning a 224 × 224 colour frame, as
`CSICamera(width=224, height=224)` does on the car. Each frame is rendered from
where the car stands on the floor, so the true answer to every question is
known. The real car is a demonstration in the lab; the two mini projects run on
a digital twin of another car, the ROSMASTER R2 (below).

```
Ex11_vision_navigation.ipynb         the notebook: three TODO cells, the answer in the comment inside each
Ex11_vision_navigation_light.ipynb   the same notebook with every cell written out
problem.py                           the virtual JetRacer, the courses, the rules, the sensors, the film
Ex11_on_the_car.ipynb, SETUP.md      the real JetRacer: for the lab demonstration
```

## The problem

A JetRacer drives 6 m from A to B along a taped line, past three obstacles
standing on it, deciding from its camera. The two lectures ask four questions
on the way, and the notebook answers each with a measurement:

| question | lecture | sections |
|---|---|---|
| where is the path? | L11.1 | 2 – 4 |
| how do I steer onto it? | L11.1 | 1, 7 |
| is the path blocked? | L11.2 | 5 |
| what do I do about it? | L11.2 | 6 – 7 |

**From the vendor** (`docs/hardware/jetracer-pro.md`): Jetson Nano 4 GB,
IMX219-160 camera, Ackermann steering with a 6 kg·cm servo, RC380 motor, 2S2P
pack at 8.4 V, no encoder. **Assumed in the twin, to be measured on the car:**
the camera (10 cm high, tilted 20° down, a 120° pinhole for the 160° lens,
12 cm ahead of the rear axle), the wheelbase (0.20 m), the wheel angle at full
servo (35°, limit 30°), the speed at full duty (5 m/s) and how fast it is
reached (0.3 s), the friction (0.6), the half-width (10 cm), the 0.7 m within
which an obstacle blocks the path, and the course.

## The notebook

| section | what the student does |
|---|---|
| 1 | the JetRacer and its twin; the course from A to B; pure pursuit and the blind distance |
| 2 | the frames on the course: the car at four places and what its camera sees from each |
| 3 | the classical line detector at four resolutions, for an agreed mean error of 0.03 |
| 4 | the convolution of L5.1; **TODO 1**, the network (one network, one output here, two in section 5), trained on two drives a person makes round the lap, smooth and with swerves; its learned kernels |
| 5 | **TODO 2**, the cross-entropy; the network against the dark-pixel rule; the threshold by the expected cost |
| 6 | **TODO 3**, the potential field; stop, steer around and the field on the course |
| 7 | the network driving the twin round the lap at eight loop rates, the stopping distance and the safe speed, multiply-adds and CPU against GPU, the weights in fewer bits |
| 8 | the race: a conventional and a deep-learning JetRacer from A to B, each seeing with its own camera, and the winner celebrated |
| 9 | what the notebook says |
| 10 | the report and its PDF: four questions per lecture and one to conclude |
| 11 | the mini projects |

### What it measures (CPU, seed 88; about 5.5 minutes, most of it training)

| finding the line: mean error on 300 test frames | car on the line | beside it | with glare |
|---|---|---|---|
| classical detector, 112 × 112 | 0.007 | 0.021 | 0.083 |
| network, smooth drive | 0.010 | 0.195 | 0.194 |
| network, drive with swerves | 0.021 | 0.036 | 0.050 |

As a root mean square in centimetres across the floor 40 cm ahead (L11.1,
*Steering From a Frame*): detector 0.8 / 4.5 / 14.1, network trained
on the drive with swerves 1.8 / 3.6 / 5.3.

| free or blocked: 800 held-out frames | right | missed | false stops | grey boxes missed |
|---|---|---|---|---|
| dark-pixel rule | 85.1 % | 19.7 % | 13.5 % | 70.5 % |
| network, threshold 0.5 | 96.6 % | 14.6 % | 0.2 % | 43.2 % |
| network, threshold 0.05 (least expected cost) | 95.4 % | 3.9 % | 4.8 % | 11.4 % |

| strategy, 10 decisions a second | time to B | smallest gap |
|---|---|---|
| stop | never | — |
| steer around | 8.30 s | 0.6 cm; collides on 2 of 20 runs with the network's errors |
| potential field | 6.30 s | 6.7 cm; none of 20 |

With the network steering the twin round the lap, the car stays within about
5 cm of the line up to 17 cm blind distance and swings 15 cm at 20 cm. The
largest safe speed within 0.7 m is 1.93, 2.34 and 2.68 m/s at 5, 10 and 30 Hz.
One frame is 16.8 million multiply-adds. The steering network keeps its error
down to 5 bits and loses it at 4; the obstacle network keeps its accuracy down
to 4 bits.

L11.1's and L11.2's measured figures and numbers come from the same
`problem.py` with the same seeds (`tools/deck/make_l11_figs.py`).

### The race

`pb.race` sends two virtual JetRacers along `pb.RACE_COURSE` (a dark box, a
grey box the colour of the floor, a dark box, two shadows, two patches of
sun): one sees with the classical detector and the dark-pixel rule, one with
the two networks; both pass obstacles by the same rule. The conventional car
runs into the grey box; the deep-learning car reaches B in about 8.4 s and
`pb.race_film` celebrates it on the podium with champagne and confetti.

### The film

`pb.film(pb.avoid("field", 10))` films a run: the JetRacer in 3-D, modelled on
the vendor's photograph (chassis and foam bumper, white ten-spoke rims, blue
shock caps, battery holder, board with USB ports, the Jetson's heat sink and
fan, the green camera bracket, antennas), lit and with its shadow, followed
from behind as it drives the course; beside it the frame its camera sees and
the course from above. The shapes are a drawing to the twin's own numbers, not
a measurement of the car. About 25 s to draw.

## The car, for the mini projects

The mini projects, MP11A and MP11B (course policy C14), run **on a digital twin
only**, in Colab: no real car and no ROS (the lecturer, 27 October 2026). The
twin is of the lab's **Yahboom ROSMASTER R2**, Superior version, and lives in
`miniprojects/MP11A/r2twin.py`, shared by both projects: Ackermann steering,
65 mm wheels, at most 1.8 m/s, its built-in Astra Pro Plus depth camera (colour
and depth) and SLAM A1 LiDAR (Slamtec RPLIDAR A1), odometry and an IMU, on a
closed track with seeded noise at rising levels. The R2 has no ultrasonic
sensor, and the twin has none. What the vendor states is in
`docs/hardware/rosmaster-r2.md` in the course repository; everything else in the
twin - size, wheelbase, steering limit, sensor mounting, speed lag, friction,
noise - is assumed and listed by `r2twin.spec()`. The twin extends this set's
`problem.py`: with the JetRacer's numbers its motion is Ex11's car exactly.

The JetRacer Pro stays for the lecture's demonstration: `SETUP.md` flashes and
checks it, and `Ex11_on_the_car.ipynb` collects frames, trains, and runs the
instrumented loop on it.

## Mini projects

Each mini project takes this set's problem further - a harder version of it, or its
inverse - with no exact solution. Each student chooses **one** mini project from the
Part 2 sets and solves it individually in one month; the dates and the hand-in are
given in L13. Each mini project is **one Colab notebook in five parts** under course
policy C14: A the problem, B the data and the physics and C the reference model and
the ground truth are given and solved; D the physics-informed network and E the
report are yours. The goal is the reference model's accuracy with an online answer
at least ten times faster (for an inverse problem, ten times faster than a new
least-squares fit); each project's README states the problem, what changed, the
ground truth, a worked example and the requirements.

| mini project | new against the exercise | required | notebook |
|---|---|---|---|
| [**MP11A · Faster from A to B**](https://github.com/RemusTeodorescu/DL-for-Engineers-Public-Course/tree/main/miniprojects/MP11A) | the ROSMASTER R2's digital twin, camera only, on a closed track with 4 - 8 obstacles of four kinds; where the obstacle stands | no collision, lap within 5 % of 18.13 s, at most 5 % missed of every kind, position within 10 cm, a decision in 10 ms | [open in Colab](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/miniprojects/MP11A/MP11A_faster_a_to_b.ipynb) |
| [**MP11B · Fusion of Camera, Depth Camera and LiDAR**](https://github.com/RemusTeodorescu/DL-for-Engineers-Public-Course/tree/main/miniprojects/MP11B) | the twin's three sensors fused in difficult conditions: a dim room, sun on the floor, obstacles each sensor finds hard | as MP11A at noise levels 2 and 3: no collision, 5 % per kind, within 10 cm, 10 ms | [open in Colab](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/miniprojects/MP11B/MP11B_sensor_fusion.ipynb) |
