r"""MP11A — Faster from A to B, camera only, on the ROSMASTER R2 twin. The data,
the reference model (conventional vision) and the ground truth (perfect
perception).

*Deep Learning for Engineering* — MSc, Aalborg University.
Remus Teodorescu (ret@et.aau.dk), with support from Research Assistant
Noman Khan (nomank@energy.aau.dk).

This module extends Ex11's ``problem.py`` through ``r2twin.py``, the digital
twin of the Yahboom ROSMASTER R2 shared with MP11B (course policy C10 rule 1:
one set of physics for the exercise and its mini projects), and is used by the
notebook of MP11A (course policy C14: twin only, in Colab).

## What changes against Ex11, and why

Ex11 drives a JetRacer 6 m along a straight taped line past three obstacles
whose positions the race takes from the course. Here the car is the
ROSMASTER R2 (its colour camera only), the course is a closed 21 m track with
bends, a chicane and an S-bend, 4 to 8 obstacles of four kinds, and the noise
of the room rises in five seeded levels. The network must say not only
whether the path is blocked but **where** the obstacle stands, so that a
planner can pass it at speed.

## Two references

* the **reference model** (conventional vision): Ex11's dark-pixel rule on the
  R2's frame, with the obstacle's foot read off the flat floor
  (``r2twin.camera_rule``), driving Ex11's steer-around rule at 1.0 m/s - and,
  for a fair comparison, the potential field at 1.5 m/s;
* the **ground truth**: the twin's own geometry (every label exact) and the
  lap driven with perfect perception and the same planner, converged in the
  time step (:func:`convergence_study`).
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

import problem as pb                                   # Ex11's virtual JetRacer
import r2twin as rt                                    # the ROSMASTER R2 twin, its sensors, the track and the planners

TRAIN_TRACKS = range(0, 120)        #: the tracks the training frames come from
TRAIN_LEVELS = (0, 1, 2)            #: the noise levels trained on
PER_TRACK = 40                      #: poses per track and level for training
TEST_SAMPLE_TRACKS = range(900, 920)   #: held-out tracks for the frame-by-frame test
TEST_SAMPLE_PER = 15                #: poses per test track and level
TEST_LEVELS = (0, 1, 2, 3)          #: 0 - 2 trained on (other tracks), 3 never seen
TEST_TRACKS = range(1000, 1010)     #: held-out tracks driven in closed loop
REQUIRED_LEVELS = (0, 1, 2)         #: the levels the requirements hold at
SPEED = 1.5                         #: m/s, the speed asked of the network's car (the R2's largest is 1.8)
REF_SPEED = 1.0                     #: m/s, Ex11's speed for the steer-around rule
THRESHOLD = 0.5                     #: the network's probability above which the path counts as blocked
DT = 2e-3                           #: s, the time step of the drive (converged: convergence_study)
KINDS = rt.KIND_NAMES


def training_data(tracks=TRAIN_TRACKS, per=PER_TRACK, levels=TRAIN_LEVELS, seed=5):
    """Labelled colour frames from the training tracks (see ``r2twin.samples``)."""
    return rt.samples(tracks, per, levels=levels, seed=seed, need=("rgb",))


def test_samples(seed=9):
    """The held-out frames: 20 tracks never trained on, 15 poses each, at noise levels 0 to 3."""
    return rt.samples(TEST_SAMPLE_TRACKS, TEST_SAMPLE_PER, levels=TEST_LEVELS, seed=seed, need=("rgb",))


def reference_answers(D):
    """The conventional rule on every frame of ``D``: ``(blocked, forward, sideways)`` rows."""
    return np.array([rt.camera_rule(f.transpose(1, 2, 0)) for f in D["rgb"]], float)


def network_perceive(net, device="cpu", threshold=THRESHOLD):
    """A trained ``net(frames) -> (n, 3)`` (logit, forward, sideways in metres)
    as a perception for ``r2twin.drive``."""
    import torch

    def perceive(car, sensors):
        x = torch.as_tensor(np.ascontiguousarray(sensors["rgb"].transpose(2, 0, 1)[None]), device=device)
        with torch.no_grad():
            o = net(x)[0].detach().cpu().numpy()
        return int(1 / (1 + np.exp(-o[0])) > threshold), float(o[1]), float(o[2])
    return perceive


def closed_loop(make_perceive, planner=rt.FieldAround, speed=SPEED, tracks=TEST_TRACKS, levels=REQUIRED_LEVELS,
                need=("rgb",), dt=DT, verbose=False):
    """Drive one lap of every track at every level. ``make_perceive(track)``
    returns the perception. Returns arrays ``time``, ``gap``, ``collided``,
    ``left`` of shape (levels, tracks)."""
    T, Gp, C, Lf = (np.zeros((len(levels), len(tracks))) for _ in range(4))
    for i, lv in enumerate(levels):
        for j, sd in enumerate(tracks):
            tr = rt.Track(sd, lv)
            r = rt.drive(tr, make_perceive(tr), planner(), speed, seed=sd, dt=dt, need=need)
            T[i, j], Gp[i, j], C[i, j], Lf[i, j] = r["time"], r["clearance"], r["collided"], r["left"]
        if verbose:
            ok = np.isfinite(T[i])
            print(f"  level {lv}: {int(C[i].sum())} collision(s) in {len(tracks)} laps, mean lap "
                  f"{T[i][ok].mean() if ok.any() else np.inf:.2f} s, smallest gap {100 * Gp[i].min():.1f} cm")
    return dict(time=T, gap=Gp, collided=C.astype(bool), left=Lf.astype(bool), levels=np.array(levels), tracks=np.array(list(tracks)))


# ====================================================================== the ground truth
def oracle_laps(dt=DT, planner=rt.FieldAround, speed=SPEED, tracks=TEST_TRACKS):
    """Perfect perception on the test tracks (noise does not change it, so level 0)."""
    return closed_loop(rt.oracle, planner, speed, tracks, levels=(0,), need=(), dt=dt)


def convergence_study(dts=(4e-3, 2e-3, 1e-3, 5e-4), verbose=True):
    """The perfect-perception lap on the test tracks with the potential field
    at :data:`SPEED`, the time step halved three times: the mean lap time on
    each, the change between steps, the observed order and, where it is
    regular, Richardson's extrapolation (the car is moved by explicit Euler,
    first order)."""
    rows = []
    for dt in dts:
        r = oracle_laps(dt)
        rows.append((dt, float(np.mean(r["time"])), float(r["gap"].min()), int(r["collided"].sum())))
        if verbose:
            print(f"  dt {1e3 * dt:4.1f} ms: mean lap {rows[-1][1]:.4f} s, smallest gap {100 * rows[-1][2]:.2f} cm, "
                  f"collisions {rows[-1][3]}")
    m = np.array([r[1] for r in rows])
    d = np.abs(np.diff(m))
    order = np.log2(d[:-1] / d[1:])
    regular = np.all(np.abs(order - 1.0) < 0.5)               # explicit Euler: first order, if the lap were smooth in the step
    rich = m[-1] + (m[-1] - m[-2]) if regular else m[-1]      # Richardson for order 1; otherwise the finest step stands
    if verbose:
        print("  change between steps:", " ".join(f"{v:.1e}" for v in d), "s; observed order", np.round(order, 2))
        print(f"  {'Richardson, order 1' if regular else 'no regular order (the decisions every 0.1 s make the lap a non-smooth function of the step): the finest step'}"
              f" {rich:.4f} s; the step used, {1e3 * DT:.0f} ms, is {1e3 * abs(m[list(dts).index(DT)] - rich):.1f} ms from it")
    return dict(table=np.array(rows), order=order, richardson=rich)


check_sensors = rt.check_twin                    #: the physics checks of the twin (r2twin.check_twin)


def commit():
    try:
        root = Path(__file__).resolve().parents[2]
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip() or "unknown"
    except OSError:
        return "unknown"


TRUTH_FILE = "MP11A_truth.npz"


def write_truth(path=TRUTH_FILE):
    """The ground truth file: perfect perception on the test tracks with both
    planners, the convergence study, and the labels of the held-out frames
    (the frames themselves are rendered again in the notebook from the same
    seeds; the labels check that they are the same)."""
    conv = convergence_study()
    field = oracle_laps()
    steer = oracle_laps(planner=rt.SteerAround, speed=REF_SPEED)
    D = rt.samples(TEST_SAMPLE_TRACKS, TEST_SAMPLE_PER, levels=TEST_LEVELS, seed=9, need=())   # the labels of test_samples()
    readme = (f"MP11A ground truth on the ROSMASTER R2 twin (r2twin.py), miniprojects/MP11A/mp11a.py at commit {commit()}. "
              "test_tracks: the seeds of r2twin.Track(seed, level), held out. field_time, field_gap: one lap with perfect "
              f"perception and the potential field (FieldAround) at {SPEED} m/s, dt {1e3 * DT:.0f} ms [s, m]; steer_time, "
              f"steer_gap: the same with the steer-around rule at {REF_SPEED} m/s. Perfect perception does not depend on "
              "the noise level. conv_dt, conv_time, conv_gap: the convergence study (time step [s], mean lap [s], "
              "smallest gap [m]); conv_richardson [s]. sample_*: the labels of the held-out frames of mp11a.test_samples(): "
              "blocked (0/1), where (forward, sideways from the rear axle [m]), kind, edge (near the zone's edge, left out of "
              "the counts), level, track. Every value of the "
              "car and its sensors is VENDOR or ASSUMED as r2twin.spec() lists.")
    np.savez_compressed(path, test_tracks=np.array(list(TEST_TRACKS)), field_time=field["time"][0], field_gap=field["gap"][0],
                        steer_time=steer["time"][0], steer_gap=steer["gap"][0], conv_dt=conv["table"][:, 0],
                        conv_time=conv["table"][:, 1], conv_gap=conv["table"][:, 2], conv_richardson=conv["richardson"],
                        sample_blocked=D["blocked"], sample_where=D["where"], sample_kind=D["kind"], sample_edge=D["edge"], sample_level=D["level"],
                        sample_track=D["track"], speed=SPEED, ref_speed=REF_SPEED, dt=DT, README=readme)
    print("written", path)


if __name__ == "__main__":
    import sys
    if "check" in sys.argv:
        check_sensors()
    else:
        write_truth()
