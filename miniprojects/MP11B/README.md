# MP11B · Fusion of Camera, Depth Camera and LiDAR

**Mini project of Ex11, Vision-Based Navigation · Part 2 · course policy C14**

One Colab notebook, `MP11B_sensor_fusion.ipynb` (GPU runtime, T4), in five parts:

| part | given | what it holds |
|---|---|---|
| A · The problem | solved | why fuse, the ROSMASTER R2 twin and its three sensors, which sensor sees which kind of obstacle, the five noise levels, what is asked |
| B · The data and the fusion | solved | **a primer on sensor fusion** (why sensors fail differently, early, late and intermediate fusion, modality dropout), the network's inputs and outputs, 16 000 training poses and 900 held-out poses of all three sensors, and a worked fusion rule by the sensors' measured reliabilities |
| C · The reference model and the ground truth | solved | the three conventional rules, their fusion by OR and by reliability, on the held-out poses and in closed loop; the perfect-perception lap and its convergence; the twin's physics checks; a film |
| D · The network | TODO cells | your fusion network (three branches, heads for blocked and where), your training, your figures; a given cell scores it, drives the 10 test tracks and times one decision |
| E · The report | TODO cells | the goal metrics against the ground truth and the references, and eighteen questions |

Its first cells fetch `mp11b.py`, the ground truth `MP11B_truth.npz`, Ex11's library files, and the twin `r2twin.py` **from MP11A's folder** (one source for both projects). Everything runs on the digital twin: no real car, no ROS.

## The problem

The ROSMASTER R2 drives one lap of MP11A's closed track at 1.5 m/s in **difficult conditions** and must fuse its three sensors into one answer, ten times a second: is the path blocked, and where does the obstacle stand. **New against Ex11 and MP11A:** the depth camera and the LiDAR (the R2 has no ultrasonic sensor), more of the obstacles each sensor finds hard (15 % dark boxes, 30 % grey, 30 % low, 25 % glossy black), and the required noise levels are 2 and 3 (a dim room with sun on the floor).

| sensor | here | vendor | where it fails |
|---|---|---|---|
| colour camera | 64 × 112 frame | 66.1 × 40.2 degrees | grey boxes, shadows, dim light, sun |
| depth camera (Astra Pro Plus) | 48 × 64 distances, 0 = no reading | 0.6 - 8 m, 58.4 × 45.5 degrees, ±3 mm at 1 m | glossy black (85 % holes, ASSUMED), sun on the floor, nearer than 0.6 m |
| LiDAR (SLAM A1 = RPLIDAR A1) | 360 ranges, 0 = no return | 0.15 - 12 m, 5.5 scans/s | anything under its plane at 20 cm (ASSUMED mount): the low blocks; glossy black (85 % dropouts, ASSUMED) |

The car, the track, the zone called blocked, the noise levels and the drive are MP11A's (its README has the tables); all values not stated by the vendors are ASSUMED and listed by `r2twin.spec()`.

## What you build

One network with three branches - the frame, the depth image, the scan - whose features are joined (intermediate fusion) and read by heads for blocked and for where. Trained on 100 tracks at noise levels 0 to 3. The design (masks for missing readings, coordinates, modality dropout), training and figures are yours.

## The reference model and the ground truth

* **Reference model** (timed): the three conventional rules of `r2twin` - dark pixels on the frame, depth points standing over the floor, LiDAR returns in the zone - joined by **OR** (Ex11's proposal), and, as a worked example of late fusion, by their **measured reliabilities** (naive Bayes, section B5). On the 600 held-out poses of levels 2 and 3, OR misses no obstacle but stops on 21.9 % of the free poses and places obstacles 21 to 25 cm off at level 3; the reliability fusion stops on 4.0 % but misses 46.5 % of the grey boxes. In closed loop on the 10 test tracks at levels 2 and 3, OR collides on 13 of 20 laps with the steer-around rule at 1.0 m/s and on 5 of 20 with the field at 1.5 m/s, the reliability fusion on 5 of 20. One decision of the pipeline takes 0.18 to 0.27 ms on a CPU.
* **Ground truth** (`MP11B_truth.npz`): the twin's geometry and one lap of each test track with perfect perception. The test tracks place their obstacles exactly as MP11A's (only the kinds differ), so the laps are the same: field at 1.5 m/s mean **18.13 s**, smallest gap 9.7 cm, no collision; convergence in the time step and the physics checks as in MP11A.

## Worked example

On test track 1000 of this project (a dark box, two grey boxes and a low block), perfect perception laps in 17.69 s; over the 10 test tracks 16.39 to 19.97 s, mean 18.13 s. Section B5 measures each rule's hit and false rates on 4 000 training poses (camera 74.8 % and 1.8 %, depth 55.7 % and 3.3 %, LiDAR 64.0 % and 18.9 %) and works one pose through by hand: a low block 1.69 m ahead that the camera and the LiDAR miss and the depth camera sees. Log-odds -1.36 (camera free) + 2.83 (depth blocked) - 0.81 (LiDAR free) = +0.66: probability 0.66, blocked. OR says blocked, AND says free.

## Requirements

* **no collision** on the 10 test tracks (seeds 1000 - 1009) at noise levels 2 and 3, the field at 1.5 m/s (20 laps);
* the mean lap within **5 %** of perfect perception's 18.13 s;
* on 600 held-out poses at levels 2 and 3: at most **5 %** missed of **every** kind and at most **5 %** false stops;
* the mean position error at most **10 cm** for every kind;
* one decision at most **10 ms** on the Colab T4.

Level 4 (never trained on) is reported, not required, as is each sensor's contribution (the network with one sensor blanked).

## What you hand in

The PDF written at the end of part E: the results table, your figures, and the eighteen answers.
