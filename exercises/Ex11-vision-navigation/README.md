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
known. The real car is a demonstration in the lab, and the platform of the two
mini projects.

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
| 11 | the mini project proposals |

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

The mini projects run on the lab's **Yahboom ROSMASTER R2**, Superior version
(26 October 2026): Jetson Orin Nano Super 8 GB, Ackermann steering, 65 mm
wheels, at most 1.8 m/s, a SLAM A1 LiDAR (Slamtec RPLIDAR A1), an Astra Pro
Plus depth camera, an IMU and odometry, ROS2 Humble (vendor; see
`docs/hardware/rosmaster-r2.md` in the course repository for what the vendor
does and does not state). Its depth camera and LiDAR are built in; the DFRobot
URM09 ultrasonic sensor is an optional add-on for MP11B, on the Orin Nano's
I2C bus. The car is driven through ROS2 with Yahboom's
course materials, not through the JetRacer library, so code written for the
twin keeps its network and decision rule and changes only how it reads the
camera and sends the steering and speed. The twin's camera mount, steering and
speed response are a JetRacer's and assumed: measure the R2's and set them in
`problem.py` before training on rendered frames.

The JetRacer Pro stays for the lecture's demonstration: `SETUP.md` flashes and
checks it, and `Ex11_on_the_car.ipynb` collects frames, trains, and runs the
instrumented loop on it.

## Mini project proposal

The set ends with two mini projects (in this README, as the set has no report notebook). Each student chooses one
mini project from the Part 2 sets and solves it individually in one month. The
ground truth is given, built by `tools/miniprojects/ex11_truth.py` under policy C10
(`COURSE_POLICIES.md`), with a worked example of each.

| | the problem | the deep learning | the ground truth given | required |
|---|---|---|---|---|
| **MP11A · Faster from A to B, on the ROSMASTER R2** | faster from A to B on the ROSMASTER R2, camera only: better obstacle avoidance and a higher speed | the notebook's network with a position head (blocked and where), feeding the potential field or your own planner | the measured course on the floor; on the twin, perfect perception's 8.76 s (steer around) and 4.24 s (field, 1.5 m/s) | twin: no collision, at most 5.0 s, position within 5 cm; car: five runs without contact, filmed |
| **MP11B · Depth camera and LiDAR in difficult conditions, on the ROSMASTER R2** | the ROSMASTER R2 with its built-in depth camera and LiDAR, a URM09 sonar optional, in difficult conditions: grey and low obstacles, shadows, glare, a dim room | one network, three branches (frame, scan, range), heads for blocked and for where; twin first, then the car | the measured course on the floor; on the twin, perfect perception's 8.70 s on hard courses; all three rules by OR collide on 12 of 20 | twin: 5 % per kind, no collision, within 10 %; car: five runs without contact in the dimmed lab, filmed |
