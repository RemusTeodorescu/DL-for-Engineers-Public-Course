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
Ex11_on_the_car.ipynb, SETUP.md      the real JetRacer: for the mini projects only
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
| 4 | the convolution of L5.1; **TODO 1**, the network (one network, one output here, two in section 5), trained on frames collected along the line and deliberately; its learned kernels |
| 5 | **TODO 2**, the cross-entropy; the network against the dark-pixel rule; the threshold by the expected cost |
| 6 | **TODO 3**, the potential field; stop, steer around and the field on the course |
| 7 | the network driving the twin round the lap at eight loop rates, the stopping distance and the safe speed, multiply-adds and CPU against GPU, the weights in fewer bits |
| 8 | what the notebook says |
| 9 | the report and its PDF: four questions per lecture and one to conclude |
| 10 | the mini project proposals |

### What it measures (CPU, seed 88; about 5.5 minutes, most of it training)

| finding the line: mean error on 300 held-out frames | along the line | beside it | with glare |
|---|---|---|---|
| classical detector, 112 × 112 | 0.007 | 0.021 | 0.083 |
| network, trained along the line | 0.006 | 0.185 | 0.186 |
| network, trained deliberately | 0.018 | 0.035 | 0.055 |

| free or blocked: 800 held-out frames | right | missed | false stops | grey boxes missed |
|---|---|---|---|---|
| dark-pixel rule | 85.1 % | 19.7 % | 13.5 % | 70.5 % |
| network, threshold 0.5 | 95.1 % | 21.9 % | 0 % | 70.5 % |
| network, threshold 0.05 (least expected cost) | 94.9 % | 3.9 % | 5.5 % | 11.4 % |

| strategy, 10 decisions a second | time to B | smallest gap |
|---|---|---|
| stop | never | — |
| steer around | 8.30 s | 0.6 cm; collides on 2 of 20 runs with the network's errors |
| potential field | 6.30 s | 6.7 cm; none of 20 |

With the network steering the twin round the lap, the car stays within about
6 cm of the line up to 17 cm blind distance and swings 15 cm at 20 cm. The
largest safe speed within 0.7 m is 1.93, 2.34 and 2.68 m/s at 5, 10 and 30 Hz.
One frame is 16.8 million multiply-adds. The steering network keeps its error
down to 5 bits and loses it at 4; the obstacle network keeps its accuracy down
to 4 bits.

L11.1's and L11.2's measured figures and numbers come from the same
`problem.py` with the same seeds (`tools/deck/make_l11_figs.py`).

### The film

`pb.film(pb.avoid("field", 10))` films a run: the JetRacer in 3-D seen from
behind, the frame its camera sees, and the course from above.

## The car, for the mini projects

The JetRacer Pro, the RPLIDAR A1 and the DFRobot URM09 are available for the
mini projects, which are done on the real car. `SETUP.md` flashes and checks
the car; `Ex11_on_the_car.ipynb` collects frames, trains, and runs the
instrumented loop on it. The car has **no encoder**, so speeds on the floor are
timed, not measured. Code written for the twin runs on the car with the import
changed: `from jetracer.nvidia_racecar import NvidiaRacecar` and
`from jetcam.csi_camera import CSICamera`.

## Mini project proposal

The set ends with two mini projects (in this README, as the set has no report notebook). Each student chooses one
mini project from the Part 2 sets and solves it individually in one month. The
ground truth is given, built by `tools/miniprojects/ex11_truth.py` under policy C10
(`COURSE_POLICIES.md`), with a worked example of each.

| | the problem | the deep learning | the ground truth given | required |
|---|---|---|---|---|
| **MP11A · The camera with a LiDAR, on the JetRacer** | the real JetRacer with an RPLIDAR A1; a taped 6 m course with dark, grey and low obstacles and dark paper | one network, two branches (2-D over the frame, 1-D over the scan), heads for blocked and for where; twin first, then the car | the measured course on the floor; on the twin, the classical rules and perfect perception's 8.76 s; camera OR LiDAR collides on 13 of 20 | twin: 5 % per kind, no collision, within 10 %; car: five runs without contact, filmed |
| **MP11B · The camera with an ultrasonic sensor, on the JetRacer** | the real JetRacer with a URM09 on I2C; the course of MP11A | one network: the frame through convolutions, the range joined before the dense layers; twin first, then the car | as MP11A; on the twin the ultrasonic rule is 86 % right and collides on 20 of 20 | as MP11A |
