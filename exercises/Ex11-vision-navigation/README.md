# Ex_11 — Vision-Based Navigation, from A to B

**Paired with L11.1 · Vision-Based Navigation and L11.2 · Obstacle Avoidance · Part 2**

One notebook for both lectures (course policies C11 and C13), as Ex08 is for
L8.1 and L8.2. **It runs on Colab or a laptop and needs no car**: the frames
are drawn by `problem.py` and the car is simulated, so the true answer to
every question is known. The same task on the real JetRacer is a
demonstration in the lab. The car and two extra sensors are for the mini
projects.

```
Ex11_vision_navigation.ipynb         the notebook: three TODO cells, the answer in the comment inside each
Ex11_vision_navigation_light.ipynb   the same notebook with every cell written out
problem.py                           the frames, the rules, the simulated car, the course, the sensors
Ex11_on_the_car.ipynb, SETUP.md      the JetRacer: for the mini projects only, not for the notebook
```

## The problem

A small car drives 6 m from A to B along a taped line, past three obstacles
standing on it, deciding from one camera. The two lectures ask four questions
on the way, and the notebook answers each with a measurement:

| question | lecture | sections |
|---|---|---|
| where is the path? | L11.1 | 2 – 4 |
| how do I steer onto it? | L11.1 | 1, 7 |
| is the path blocked? | L11.2 | 5 |
| what do I do about it? | L11.2 | 6 – 7 |

**Assumed, not measured:** the course and its obstacles, the car's wheelbase
(0.20 m), steering limit (30°) and half-width (10 cm), the 0.7 m the camera
judges free or blocked, the tyres' friction (0.6), and the cost of a
collision against a false stop (100 to 1).

## The notebook

| section | what the student does |
|---|---|
| 1 | the course from A to B, pure pursuit, the blind distance |
| 2 | the drawn frames: the line, glare, obstacles and shadows |
| 3 | the classical line detector at four resolutions, for an agreed mean error of 0.03 |
| 4 | **TODO 1**, the network (one network, one output here, two in section 5), trained on frames collected along the line and deliberately |
| 5 | **TODO 2**, the cross-entropy; the network against the dark-pixel rule; the threshold by the expected cost |
| 6 | **TODO 3**, the potential field; stop, steer around and the field on the course |
| 7 | the blind distance on a lap, the stopping distance and the safe speed, multiply-adds and CPU against GPU, the weights in fewer bits |
| 8 | what the notebook says |
| 9 | the report and its PDF: four questions per lecture and one to conclude |
| 10 | the mini project proposals |

### What it measures (CPU, seed 88; about 90 s)

| finding the line: mean error on 1000 held-out frames | along the line | beside it | with glare |
|---|---|---|---|
| classical detector, 12 × 16 pixels | 0.024 | 0.026 | 0.369 |
| network, trained along the line | 0.002 | 0.136 | 0.149 |
| network, trained deliberately | 0.006 | 0.008 | 0.009 |

| free or blocked: 2000 held-out frames | right | missed | false stops | shadows called blocked |
|---|---|---|---|---|
| dark-pixel rule | 84.2 % | 19.9 % | 14.5 % | 13.5 % |
| network, threshold 0.5 | 98.4 % | 3.0 % | 1.2 % | 0 % |
| network, threshold 0.10 (least expected cost) | 98.5 % | 0 % | 2.0 % | |

| strategy, 10 decisions a second | time to B | smallest gap |
|---|---|---|
| stop | never | — |
| steer around | 8.66 s | 2.2 cm (touches at 5 Hz) |
| potential field | 6.33 s | 10.8 cm |

The car holds the line within about 2 cm up to 10 cm blind and leaves the
course at 17 cm; the largest safe speed within 0.7 m is 1.93, 2.34 and
2.68 m/s at 5, 10 and 30 Hz. One frame is 645,152 multiply-adds. The steering
network keeps its error to 5 bits and loses it at 3; the obstacle network
keeps its accuracy to 4 bits.

L11.1's and L11.2's measured figures come from the same `problem.py` with
the same seeds (`tools/deck/make_l111_figs.py`, `make_l112_figs.py`).

## The car, for the mini projects

The JetRacer Pro, the RPLIDAR A1 and the DFRobot URM09 are available for the
mini projects. `SETUP.md` flashes and checks the car; `Ex11_on_the_car.ipynb`
collects frames, trains, and runs the instrumented loop on it. The car has
**no encoder**, so speeds on the floor are timed, not measured.

## Mini project proposal

The set ends with two mini projects (in this README, as the set has no report notebook). Each student chooses one
mini project from the Part 2 sets and solves it individually in one month. The
ground truth is given, built by `tools/miniprojects/ex11_truth.py` under policy C10
(`COURSE_POLICIES.md`), with a worked example of each.

| | the problem | the deep learning | the ground truth given | required |
|---|---|---|---|---|
| **MP11A · The camera with a LiDAR** | random 6 m courses; dark, grey and low obstacles and shadows; one camera frame and one 360° LiDAR scan | one network, two branches (2-D over the frame, 1-D over the scan), heads for blocked and for where | exact geometry from Ex11's problem.py; 6000 + 1500 poses, 20 courses to drive; worked example: camera OR LiDAR collides on 11 of 20 | each kind missed at most 5 %, position within 4 cm, no collision, time within 10 %; one run on the car |
| **MP11B · The camera with an ultrasonic sensor** | the courses of MP11A; one camera frame and one ultrasonic range from a ±30° cone | one network: the frame through convolutions, the range joined before the dense layers; blocked and where | as MP11A, with the ultrasonic range; worked example: the ultrasonic rule is 87 % right and collides on 20 of 20 | each kind missed at most 5 %, position within 5 cm, no collision, time within 10 %; one run on the car |
