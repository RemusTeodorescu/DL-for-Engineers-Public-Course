r"""MP11B — Fusion of the camera, the depth camera and the LiDAR in difficult
conditions, on the ROSMASTER R2 twin. The data, the conventional rules and
their fusion (the reference model) and the ground truth (perfect perception).

*Deep Learning for Engineering* — MSc, Aalborg University.
Remus Teodorescu (ret@et.aau.dk), with support from Research Assistant
Noman Khan (nomank@energy.aau.dk).

This module extends Ex11's ``problem.py`` through ``r2twin.py``, the digital
twin of the Yahboom ROSMASTER R2 kept in MP11A's folder and shared by both
projects (course policy C10 rule 1), and is used by the notebook of MP11B
(course policy C14: twin only, in Colab).

## What changes against Ex11 and MP11A, and why

Ex11's camera misses a grey box the colour of the floor and stops for
shadows; its mini project proposal MP11B asked what a LiDAR and a second
range sensor add. The R2 carries a SLAM A1 LiDAR and an Astra Pro Plus depth
camera, and no ultrasonic sensor. Here the tracks are MP11A's with **more of
the obstacles each sensor finds hard** (grey boxes, low blocks under the
LiDAR's plane, glossy black ones that return little light) and the room is
**harder**: the required noise levels are 2 and 3, a dim room with sun on the
floor, and level 4 is reported, never trained on.

## Two references

* the **reference model**: the three conventional rules of ``r2twin`` -
  ``camera_rule`` (Ex11's dark pixels), ``depth_rule`` (points standing over
  the floor), ``lidar_rule`` (returns in the zone) - joined by OR, Ex11's
  proposal, and as a worked example of late fusion, joined by their measured
  reliabilities (:func:`trust_fusion`);
* the **ground truth**: the twin's geometry (exact labels) and the lap driven
  with perfect perception and the same planner, converged in the time step.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

_here = Path(__file__).resolve().parent
if not (_here / "r2twin.py").exists() and (_here.parent / "MP11A" / "r2twin.py").exists():
    sys.path.insert(0, str(_here.parent / "MP11A"))      # in the repository: the twin lives in MP11A's folder

import problem as pb                                   # Ex11's virtual JetRacer
import r2twin as rt                                    # the ROSMASTER R2 twin, its sensors, the track and the planners

KINDS_B = (0.15, 0.30, 0.30, 0.25)  #: the weights of box, grey, low and black: more of the hard kinds than MP11A's
TRAIN_TRACKS = range(0, 100)        #: the tracks the training data come from
TRAIN_LEVELS = (0, 1, 2, 3)         #: the noise levels trained on
PER_TRACK = 40                      #: poses per track and level for training
TEST_SAMPLE_TRACKS = range(900, 920)   #: held-out tracks for the frame-by-frame test
TEST_SAMPLE_PER = 15                #: poses per test track and level
TEST_LEVELS = (2, 3, 4)             #: 2 and 3 trained on (other tracks), 4 never seen
TEST_TRACKS = range(1000, 1010)     #: held-out tracks driven in closed loop
REQUIRED_LEVELS = (2, 3)            #: the levels the requirements hold at
SPEED = 1.5                         #: m/s, the speed asked of the network's car
REF_SPEED = 1.0                     #: m/s, Ex11's speed for the steer-around rule
THRESHOLD = 0.5                     #: the network's probability above which the path counts as blocked
DT = 2e-3                           #: s, the time step of the drive (converged: convergence_study)
SENSORS = ("camera", "depth", "lidar")


def track(seed, level):
    """A track of this project: MP11A's, with the hard kinds more likely."""
    return rt.Track(seed, level, kinds=KINDS_B)


def training_data(tracks=TRAIN_TRACKS, per=PER_TRACK, levels=TRAIN_LEVELS, seed=5):
    """Labelled readings of all three sensors from the training tracks (see ``r2twin.samples``)."""
    return rt.samples(tracks, per, levels=levels, seed=seed, kinds=KINDS_B)


def test_samples(seed=9):
    """The held-out readings: 20 tracks never trained on, 15 poses each, at noise levels 2, 3 and 4."""
    return rt.samples(TEST_SAMPLE_TRACKS, TEST_SAMPLE_PER, levels=TEST_LEVELS, seed=seed, kinds=KINDS_B)


def rule_answers(D):
    """Every conventional rule on every pose of ``D``: a dict of ``(n, 3)``
    arrays ``(blocked, forward, sideways)`` for "camera", "depth", "lidar"
    and "OR"."""
    out = {"camera": np.array([rt.camera_rule(f.transpose(1, 2, 0)) for f in D["rgb"]], float),
           "depth": np.array([rt.depth_rule(d) for d in D["depth"]], float),
           "lidar": np.array([rt.lidar_rule(s) for s in D["scan"]], float)}
    out["OR"] = np.array([rt.rules_or(a, b, c) for a, b, c in zip(out["camera"], out["depth"], out["lidar"])], float)
    return out


# ====================================================================== a worked example of late fusion
def trust_table(D, answers):
    """How far each rule can be trusted, measured on labelled data: the share
    of blocked poses it calls blocked (``hit``) and the share of free poses it
    calls blocked (``false``), clear poses only."""
    clear = ~D["edge"]
    b = D["blocked"] == 1
    return {n: dict(hit=float(np.clip(np.mean(answers[n][clear & b, 0]), 0.005, 0.995)),
                    false=float(np.clip(np.mean(answers[n][clear & ~b, 0]), 0.005, 0.995))) for n in SENSORS}


def trust_fusion(answers, trust, prior=0.5):
    """Late fusion by the rules' measured reliabilities (naive Bayes): start
    from the prior odds of blocked, multiply by ``hit/false`` for every rule
    that says blocked and by ``(1 - hit)/(1 - false)`` for every rule that
    says free; blocked when the odds exceed one. Where: from the rule that
    says blocked with the largest ``hit/false``. Returns ``(n, 3)`` and the
    probability of blocked."""
    n = len(answers["camera"])
    logodds = np.full(n, np.log(prior / (1 - prior)))
    for s in SENSORS:
        b = answers[s][:, 0] > 0.5
        h, f = trust[s]["hit"], trust[s]["false"]
        logodds += np.where(b, np.log(h / f), np.log((1 - h) / (1 - f)))
    p = 1 / (1 + np.exp(-logodds))
    out = np.full((n, 3), np.nan)
    out[:, 0] = p > 0.5
    order = sorted(SENSORS, key=lambda s: -trust[s]["hit"] / trust[s]["false"])   # the most telling rule first
    for i in np.where(out[:, 0] > 0)[0]:
        for s in order:
            if answers[s][i, 0] > 0.5:
                out[i, 1:] = answers[s][i, 1:]
                break
    return out, p


def trust_perceive(trust):
    """:func:`trust_fusion` as a perception for ``r2twin.drive``."""
    def perceive(car, s):
        a = {"camera": np.array([rt.camera_rule(s["rgb"])], float), "depth": np.array([rt.depth_rule(s["depth"])], float),
             "lidar": np.array([rt.lidar_rule(s["scan"])], float)}
        o, _ = trust_fusion(a, trust)
        return int(o[0, 0]), float(o[0, 1]), float(o[0, 2])
    return perceive


# ====================================================================== the network in the loop
def network_perceive(net, device="cpu", threshold=THRESHOLD):
    """A trained ``net(frame, depth, scan) -> (n, 3)`` (logit, forward, sideways
    in metres) as a perception for ``r2twin.drive``. The network gets the
    sensors as the R2 gives them: the frame (n, 3, 64, 112) uint8, the depth
    image (n, 48, 64) in metres (0: no reading), the scan (n, 360) in metres
    (0: no return)."""
    import torch

    def perceive(car, s):
        fr = torch.as_tensor(np.ascontiguousarray(s["rgb"].transpose(2, 0, 1)[None]), device=device)
        de = torch.as_tensor(s["depth"][None], device=device)
        sc = torch.as_tensor(s["scan"][None], device=device)
        with torch.no_grad():
            o = net(fr, de, sc)[0].detach().cpu().numpy()
        return int(1 / (1 + np.exp(-o[0])) > threshold), float(o[1]), float(o[2])
    return perceive


def closed_loop(make_perceive, planner=rt.FieldAround, speed=SPEED, tracks=TEST_TRACKS, levels=REQUIRED_LEVELS,
                need=("rgb", "depth", "scan"), dt=DT, verbose=False):
    """Drive one lap of every track of this project at every level.
    ``make_perceive(track)`` returns the perception. Returns arrays ``time``,
    ``gap``, ``collided``, ``left`` of shape (levels, tracks)."""
    T, Gp, C, Lf = (np.zeros((len(levels), len(tracks))) for _ in range(4))
    for i, lv in enumerate(levels):
        for j, sd in enumerate(tracks):
            tr = track(sd, lv)
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
    at :data:`SPEED`, the time step halved three times (explicit Euler, first
    order): the mean lap time, the change, the observed order, Richardson."""
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
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=_here.parents[1], capture_output=True, text=True).stdout.strip() or "unknown"
    except OSError:
        return "unknown"


TRUTH_FILE = "MP11B_truth.npz"


def write_truth(path=TRUTH_FILE):
    """The ground truth file: perfect perception on the test tracks with both
    planners, the convergence study, and the labels of the held-out readings
    (rendered again in the notebook from the same seeds)."""
    conv = convergence_study()
    field = oracle_laps()
    steer = oracle_laps(planner=rt.SteerAround, speed=REF_SPEED)
    D = rt.samples(TEST_SAMPLE_TRACKS, TEST_SAMPLE_PER, levels=TEST_LEVELS, seed=9, kinds=KINDS_B, need=())
    readme = (f"MP11B ground truth on the ROSMASTER R2 twin (MP11A/r2twin.py), miniprojects/MP11B/mp11b.py at commit {commit()}. "
              "test_tracks: the seeds of r2twin.Track(seed, level, kinds=KINDS_B), held out. field_time, field_gap: one lap with "
              f"perfect perception and the potential field (FieldAround) at {SPEED} m/s, dt {1e3 * DT:.0f} ms [s, m]; steer_time, "
              f"steer_gap: the same with the steer-around rule at {REF_SPEED} m/s. Perfect perception does not depend on the "
              "noise level. conv_dt, conv_time, conv_gap: the convergence study (time step [s], mean lap [s], smallest gap [m]); "
              "conv_richardson [s]. sample_*: the labels of the held-out readings of mp11b.test_samples(): blocked (0/1), where "
              "(forward, sideways from the rear axle [m]), kind, edge (near the zone's edge, left out of the counts), level, "
              "track. Every value of the car and its sensors is VENDOR or ASSUMED as r2twin.spec() lists.")
    np.savez_compressed(path, test_tracks=np.array(list(TEST_TRACKS)), field_time=field["time"][0], field_gap=field["gap"][0],
                        steer_time=steer["time"][0], steer_gap=steer["gap"][0], conv_dt=conv["table"][:, 0],
                        conv_time=conv["table"][:, 1], conv_gap=conv["table"][:, 2], conv_richardson=conv["richardson"],
                        sample_blocked=D["blocked"], sample_where=D["where"], sample_kind=D["kind"], sample_edge=D["edge"],
                        sample_level=D["level"], sample_track=D["track"], speed=SPEED, ref_speed=REF_SPEED, dt=DT,
                        kinds=np.array(KINDS_B), README=readme)
    print("written", path)


if __name__ == "__main__":
    write_truth()
