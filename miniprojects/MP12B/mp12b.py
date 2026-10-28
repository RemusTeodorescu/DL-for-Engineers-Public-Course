r"""MP12B — Stability prediction after a contingency: the rotor-angle
trajectories of the IEEE 39-bus grid with converters, predicted from the
first 200 ms of measurements after the fault is cleared. The data, the
reference model and the ground truth.

*Deep Learning for Engineering* — MSc, Aalborg University.
Remus Teodorescu (ret@et.aau.dk), with support from Research Assistant
Noman Khan (nomank@energy.aau.dk).

This module extends Ex12.2 (``problem_12_2.py``: the swing equations of
classical machines, the stability criterion) to the IEEE 39-bus grid of
``grid39c.py``, and is used by the notebook of MP12B (course policy C14).

## What changes against Ex12.2

Ex12.2 asks, before a fault, how long a fault the grid can withstand. Here
the fault has happened and been cleared, and the question is what happens
next: **will the machines stay in synchronism, and how will they swing?**
Phasor measurement units (PMUs) report the machines' rotor angles and speeds
50 times a second; after the first 200 ms of them (11 frames) the operator
wants the next 2.8 s, and the verdict, long before they arrive.

* the grid is the IEEE 39-bus system with ten classical machines and 0 to
  60 % of its generation from converters (``grid39c.py``);
* a case is an operating point, a contingency of the 35 and a clearing time
  between 60 and 260 ms; the training cases (1 740 operating points, 52 200
  cases) are simulated by :func:`training_data` in the notebook, the test
  cases (60 other operating points, 1 800 cases) are in the truth file;
* the measurements are the rotor angles of the nine machines against machine
  1 and the speeds of all ten, every 20 ms from the clearing to 200 ms after
  it, with noise of 0.5 degrees and 5 mHz (ASSUMED, a PMU-based estimate of
  the rotor angle; IEEE C37.118 allows 5 mHz of frequency error in steady
  state);
* the multi-machine swing has no solution in closed form; the ground truth is
  a converged simulation, checked against the exact one-machine case and the
  conservation of the transient energy (``grid39c.smib_check``,
  ``grid39c.energy_check``).

## Two solvers

* the **reference model** (:func:`reference`): what an engineer would run when
  the window closes - the swing equations by RK4 on 10 ms from the measured
  state at 200 ms (the last frame, with its noise) to 3 s, on the post-fault
  network of the known contingency;
* the **ground truth** (:func:`simulate_cases`): RK4 on 2 ms from the fault's
  start, noise-free, shown converged in the step (:func:`convergence_study`).
"""
from __future__ import annotations

import os
import time

import numpy as np

import grid39c as g

# ------------------------------------------------------------------ the data
N_TRAIN_OPS = 1740           #: training operating points (435 at each converter share)
N_TEST_OPS = 60              #: test operating points (15 at each share), drawn with other seeds
SEEDS = {"train": (41, 42, 43, 44), "test": (51, 52, 53, 54)}   #: operating points, converters, cases, noise
CASES_PER_OP = 30            #: cases (contingency, clearing time) per operating point
SHARES = g.SHARES
T_CLEAR = (0.06, 0.26)       #: s; the clearing time of a case, uniform
FRAME = 0.02                 #: s; a PMU frame, 50 frames per second
WINDOW = 0.20                #: s; the measurement window after clearing: frames 0 to 10
N_WIN = int(round(WINDOW / FRAME)) + 1
HORIZON = 3.0                #: s; the trajectory is predicted to 3 s after clearing
OUT_DT = 0.04                #: s; the trajectory is given and scored every 40 ms
T_OUT = np.round(np.arange(0.0, HORIZON + 1e-9, OUT_DT), 6)   #: 0 to 3 s after clearing, 76 instants
K_PRED = int(round(WINDOW / OUT_DT)) + 1   #: the prediction starts at T_OUT[6] = 0.24 s, after the window
SIGMA_ANGLE = 0.5            #: deg; noise of a rotor-angle measurement (ASSUMED)
SIGMA_FREQ = 0.005           #: Hz; noise of a speed measurement (ASSUMED; IEEE C37.118's 5 mHz)
TRUTH_DT = 0.002             #: s; the ground truth's RK4 step
REF_DT = 0.010               #: s; the reference model's RK4 step
MACHINES = np.arange(1, g.NM)   #: the nine machines whose angle is given against machine 1
NET_FEATURES = 129           #: numbers describing the post-fault grid of a case (:func:`grid_features`)


def share_of(i):
    """The converter share of operating point i."""
    return SHARES[i % len(SHARES)]


def build_ops(which):
    """The operating points ("train" or "test") and their converters."""
    s_op, s_pl, _, _ = SEEDS[which]
    ops = g.operating_points(N_TRAIN_OPS if which == "train" else N_TEST_OPS, seed=s_op)
    rng = np.random.default_rng(s_pl)
    conv = np.array([g.place_converters(op, share_of(i), rng) for i, op in enumerate(ops)])
    return ops, conv


def op_arrays(ops):
    return {k: np.array([op[k] for op in ops]) for k in ("V", "th", "P", "Q", "PL", "QL", "Pg", "Qg", "scale")}


def op_from_arrays(D, i):
    """Operating point i back from the truth file's arrays."""
    return {k: np.asarray(D[f"op_{k}"][i]) for k in ("V", "th", "P", "Q", "PL", "QL", "Pg", "Qg")}


def case_list(which, n_ops):
    """The cases: CASES_PER_OP random contingencies of every operating point,
    each with a clearing time drawn from T_CLEAR [s]."""
    rng = np.random.default_rng(SEEDS[which][2])
    op_id = np.repeat(np.arange(n_ops), CASES_PER_OP)
    cont_id = np.concatenate([rng.choice(g.N_CONT, CASES_PER_OP, replace=False) for _ in range(n_ops)])
    t_clear = np.round(rng.uniform(*T_CLEAR, len(op_id)), 4)
    return op_id, cont_id, t_clear


def out_index():
    """The positions of T_OUT in the 20 ms record."""
    return np.rint(T_OUT / FRAME).astype(int)


def simulate_cases(ops, conv, op_id, cont_id, t_clear, dt=TRUTH_DT, chunk=2000, full=False):
    """The ground truth of the cases: RK4 on ``dt`` from the fault's start,
    cleared exactly at t_clear, to 3 s after clearing, recorded every 20 ms.
    Returns the angles of machines 2 to 10 against machine 1 [deg] and the
    speeds of all ten [Hz above 50] in the window (n, 11, 9) and (n, 11, 10),
    the angles at T_OUT (n, 76, 9), and whether the case lost synchronism
    (two machines more than 180 deg apart) within 3 s; with ``full``, the
    angles on the whole 20 ms record (n, 151, 9) in place of T_OUT's."""
    wa, wf, tr, uns = [], [], [], []
    oi = out_index()
    for s in range(0, len(op_id), chunk):
        sl = slice(s, s + chunk)
        cs = g.stack([g.case_networks(ops[i], conv[i], *g.CONTINGENCIES[j]) for i, j in zip(op_id[sl], cont_id[sl])])
        top, t, d, w = g.simulate(cs, t_clear[sl], dt, HORIZON, record_dt=FRAME)
        a = np.degrees(d[:, :, MACHINES] - d[:, :, :1])
        a = np.where(conv[op_id[sl]][:, None, MACHINES], 0.0, a).astype(np.float32)
        wa.append(a[:, :N_WIN]); wf.append((w[:, :N_WIN] / (2 * np.pi)).astype(np.float32))
        tr.append(a if full else a[:, oi]); uns.append(top >= np.pi)
    return np.concatenate(wa), np.concatenate(wf), np.concatenate(tr), np.concatenate(uns)


def measure(win_a, win_f, conv_case, seed):
    """The PMU window: the true angles [deg] and speeds [Hz] with seeded
    Gaussian noise of SIGMA_ANGLE and SIGMA_FREQ; zero at converters."""
    rng = np.random.default_rng(seed)
    a = win_a + SIGMA_ANGLE * rng.standard_normal(win_a.shape)
    f = win_f + SIGMA_FREQ * rng.standard_normal(win_f.shape)
    a = np.where(conv_case[:, None, MACHINES], 0.0, a)
    f = np.where(conv_case[:, None, :], 0.0, f)
    return a.astype(np.float32), f.astype(np.float32)


def grid_features(op, conv, contingency):
    """What the operator's model says about the grid after the contingency,
    129 numbers: from the post-fault network reduced onto the machines, the
    couplings E_i E_j B_ij (45, upper triangle) and E_i E_j G_ij (45) and the
    machines' own E_i^2 G_ii (10) [p.u.]; the mechanical powers P_m (10)
    [p.u.]; 2H/w_s x 100 (10) [p.u. s²]; the pre-fault angles of machines 2
    to 10 against machine 1 (9) [rad]. Zero at converters. One Kron
    reduction; no simulation."""
    c = g.case_networks(op, conv, *contingency)
    iu = np.triu_indices(g.NM, 1)
    EE = np.outer(c["E"], c["E"])
    M = 2 * g.H_GEN / g.WS * (~np.asarray(conv, bool))
    return np.r_[(EE * c["post"].imag)[iu], (EE * c["post"].real)[iu], np.diag(EE * c["post"].real), c["Pm"], 100 * M,
                 c["d0"][1:] - c["d0"][0]]


def make_cases(which, n_ops=None, verbose=True):
    """All the data of the "train" or "test" cases: their definition, the
    true and the measured windows, the true angles at T_OUT, the verdicts,
    the grid features and the operating points. About 7 minutes on a CPU
    for the 52 200 training cases."""
    t0 = time.perf_counter()
    ops, conv = build_ops(which)
    if n_ops is not None:
        ops, conv = ops[:n_ops], conv[:n_ops]
    op_id, cont_id, t_clear = case_list(which, len(ops))
    wa, wf, traj, uns = simulate_cases(ops, conv, op_id, cont_id, t_clear)
    ma, mf = measure(wa, wf, conv[op_id], SEEDS[which][3])
    gf = np.array([grid_features(ops[i], conv[i], g.CONTINGENCIES[j]) for i, j in zip(op_id, cont_id)])
    if verbose:
        print(f"{which}: {len(op_id)} cases of {len(ops)} operating points, {100 * uns.mean():.1f} % unstable, "
              f"{time.perf_counter() - t0:.0f} s")
    out = dict(op_id=op_id, cont_id=cont_id, t_clear=t_clear, converters=conv, unstable=uns,
               window_angles=ma, window_freq=mf, window_true_angles=wa, window_true_freq=wf, traj=traj,
               grid_features=gf.astype(np.float32), share=np.array([share_of(i) for i in range(len(ops))]),
               share_actual=np.array([g.converter_share(ops[i], conv[i]) for i in range(len(ops))]))
    out.update({f"op_{k}": v for k, v in op_arrays(ops).items()})
    return out


def training_data(verbose=True):
    """The 52 200 training cases (:func:`make_cases` "train")."""
    return make_cases("train", verbose=verbose)


def reference(op, conv, contingency, angles, speeds, dt=REF_DT):
    """The reference model for one case: the post-fault network of the known
    contingency (a Kron reduction), then RK4 on ``dt`` from the measured state
    at the end of the window - the last frame's angles [deg] and speeds [Hz]
    - to 3 s after clearing. Returns the angles of machines 2 to 10 against
    machine 1 [deg] at T_OUT[K_PRED:], and whether two machines separate by
    more than 180 deg."""
    c = g.case_networks(op, conv, *contingency)
    d = np.zeros((1, g.NM)); d[0, MACHINES] = np.radians(angles)
    w = 2 * np.pi * np.asarray(speeds, float)[None]
    top, t, dd, _ = g.simulate(g.stack([c]), None, dt, HORIZON, record_dt=OUT_DT, t_from=WINDOW, x0=(d, w))
    a = np.degrees(dd[0][:, MACHINES] - dd[0][:, :1])
    return np.where(conv[MACHINES], 0.0, a)[1:], bool(top[0] >= np.pi)


def reference_batch(ops, conv, op_id, cont_id, angles, speeds, dt=REF_DT):
    """:func:`reference` for many cases at once (the same arithmetic,
    batched; used to score the reference model on every test case)."""
    cs = g.stack([g.case_networks(ops[i], conv[i], *g.CONTINGENCIES[j]) for i, j in zip(op_id, cont_id)])
    d = np.zeros((len(op_id), g.NM)); d[:, MACHINES] = np.radians(angles)
    w = 2 * np.pi * np.asarray(speeds, float)
    top, t, dd, _ = g.simulate(cs, None, dt, HORIZON, record_dt=OUT_DT, t_from=WINDOW, x0=(d, w))
    a = np.degrees(dd[:, :, MACHINES] - dd[:, :, :1])
    return np.where(conv[op_id][:, None, MACHINES], 0.0, a)[:, 1:], top >= np.pi


def convergence_study(ops, conv, op_id, cont_id, t_clear, steps=(0.01, 0.005, 0.002, 0.001), verbose=True):
    """The trajectories of the given cases on RK4 steps each about twice as
    fine: the largest change of an angle [deg] between steps, over the cases
    that stay in synchronism."""
    res = {dt: simulate_cases(ops, conv, op_id, cont_id, t_clear, dt=dt, full=True) for dt in steps}
    stable = ~res[steps[-1]][3]
    rows = []
    for a, b in zip(steps[:-1], steps[1:]):
        ch = float(np.abs(res[b][2].astype(float) - res[a][2])[stable].max())
        rows.append(ch)
        if verbose:
            print(f"  RK4 step {1000*a:.0f} -> {1000*b:.0f} ms: largest change of an angle {ch:.2e} deg "
                  f"({int((res[a][3] != res[b][3]).sum())} verdicts changed)")
    return np.array(steps), np.array(rows)


def andes_trajectory(op, conv, contingency, t_clear, t_fault=0.1):
    """ANDES's rotor angles of machines 2 to 10 against machine 1 [deg] at
    T_OUT after clearing (its GENCLS model and implicit trapezoidal method on
    2 ms), or None if ANDES could not finish the run."""
    import contextlib, io, logging
    import andes
    andes.config_logger(stream_level=50); logging.getLogger("andes").setLevel(logging.CRITICAL)
    for xf in (1e-4, 1e-3):
        ss = g.andes_system(op, conv, contingency[0], contingency[1], t_clear, t_fault, xf)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            ss.setup(); ss.PFlow.run()
            ss.TDS.config.tf = t_fault + t_clear + HORIZON; ss.TDS.config.tstep = 0.002; ss.TDS.config.no_tqdm = 1
            ss.TDS.init(); ss.TDS.run()
        t = np.array(ss.dae.ts.t)
        if abs(t[-1] - ss.TDS.config.tf) < 1e-6:
            d = np.array(ss.dae.ts.x[:, ss.GENCLS.delta.a])
            idx = list(np.nonzero(~conv)[0])                     # GENCLS devices in machine order, converters left out
            full = np.zeros((len(t), g.NM)); full[:, idx] = d
            out = np.array([np.interp(t_fault + t_clear + T_OUT, t, full[:, m] - full[:, 0]) for m in MACHINES]).T
            return np.where(conv[MACHINES], 0.0, np.degrees(out)), xf
    return None, None


TRUTH_FILE = "MP12B_truth.npz"


def write_truth(path=TRUTH_FILE, cache="mp12b_cache", n_andes=8):
    """Builds the test set and its checks, in pieces kept in ``cache``: the
    60 test operating points and their 1 800 cases (ground truth, PMU
    windows, grid features), the reference model on every test case, the
    convergence study and an ANDES comparison; then writes the truth file.
    The training cases are not in the file: :func:`training_data` makes them."""
    os.makedirs(cache, exist_ok=True)
    t_all = time.perf_counter()
    f = os.path.join(cache, "test.npz")
    if not os.path.exists(f):
        np.savez(f, **make_cases("test"))
    T = dict(np.load(f))
    ops = [op_from_arrays(T, i) for i in range(N_TEST_OPS)]
    conv = T["converters"]
    f = os.path.join(cache, "reference.npz")
    if not os.path.exists(f):
        t0 = time.perf_counter()
        ra, ru = reference_batch(ops, conv, T["op_id"], T["cont_id"], T["window_angles"][:, -1], T["window_freq"][:, -1])
        rt, rtu = reference_batch(ops, conv, T["op_id"], T["cont_id"], T["window_true_angles"][:, -1], T["window_true_freq"][:, -1])
        np.savez(f, ang=ra.astype(np.float32), uns=ru, ang_exact_start=rt.astype(np.float32), uns_exact_start=rtu)
        print(f"reference model on {len(ra)} test cases: {time.perf_counter() - t0:.0f} s", flush=True)
    RF = np.load(f)
    f = os.path.join(cache, "checks.npz")
    if not os.path.exists(f):
        sub = np.arange(0, len(T["op_id"]), 9)[:200]
        print("convergence study on 200 test cases", flush=True)
        steps, rows = convergence_study(ops, conv, T["op_id"][sub], T["cont_id"][sub], T["t_clear"][sub])
        rng = np.random.default_rng(5)
        picks = rng.choice(np.nonzero(~T["unstable"])[0], n_andes, replace=False)
        an, xfs = [], []
        for n in picks:
            a, xf = andes_trajectory(ops[T["op_id"][n]], conv[T["op_id"][n]], g.CONTINGENCIES[T["cont_id"][n]], T["t_clear"][n])
            an.append(np.full((len(T_OUT), len(MACHINES)), np.nan) if a is None else a); xfs.append(np.nan if xf is None else xf)
            e = np.nan if a is None else np.abs(a - T["traj"][n]).max()
            print(f"  ANDES, test case {n}: largest angle difference {e:.2f} deg over 3 s (fault reactance {xfs[-1]})", flush=True)
        np.savez(f, steps=steps, rows=rows, andes_cases=picks, andes_angles=np.array(an), andes_xf=np.array(xfs))
    C = np.load(f)
    out = dict(T)
    out.update(t_out=T_OUT, t_window=np.round(np.arange(N_WIN) * FRAME, 6), k_pred=np.array(K_PRED),
               ref_traj=RF["ang"], ref_unstable=RF["uns"], ref_traj_exact_start=RF["ang_exact_start"],
               ref_unstable_exact_start=RF["uns_exact_start"],
               conv_steps=C["steps"], conv_rows=C["rows"], andes_cases=C["andes_cases"], andes_angles=C["andes_angles"],
               andes_xf=C["andes_xf"], fault_bus=np.array([fb for fb, _ in g.CONTINGENCIES]),
               branch=np.array([k for _, k in g.CONTINGENCIES]), sigma_angle=np.array(SIGMA_ANGLE),
               sigma_freq=np.array(SIGMA_FREQ))
    out["params"] = np.array([f"{k} = {v}" for k, v in dict(
        grid="IEEE 39-bus, MATPOWER case39, 100 MVA, 50 Hz (ASSUMED)", machines="classical, Pai (1989), D = H (ASSUMED)",
        loads="constant impedance (ASSUMED)", converters="negative constant impedance, no inertia (ASSUMED)",
        fault="a reactance of 1e-4 p.u. to earth, cleared by tripping the branch",
        noise=f"angle {SIGMA_ANGLE} deg, speed {SIGMA_FREQ} Hz, Gaussian (ASSUMED)",
        truth=f"RK4 {TRUTH_DT} s from the fault, noise-free", reference=f"RK4 {REF_DT} s from the last noisy frame").items()])
    out["README"] = np.array(
        "MP12B ground truth: the 1800 test cases (60 operating points x 30; op_id, cont_id, t_clear [s]). "
        "window_angles (n, 11, 9): measured rotor angles of machines 2-10 against machine 1 [deg] at t_window after "
        "clearing, with noise; window_freq (n, 11, 10): measured speeds of machines 1-10 [Hz above 50]; window_true_*: "
        "without noise; zero at converters. traj (n, 76, 9): true angles [deg] at t_out (0 to 3 s after clearing, "
        "40 ms); unstable: two machines more than 180 deg apart within 3 s. grid_features (n, 129): mp12b.grid_features. "
        "ref_traj (n, 70, 9) and ref_unstable: the reference model from the last noisy frame, at t_out[k_pred:]; "
        "*_exact_start: the same from the true state. converters (60, 10), share: per operating point; op_*: power "
        "flows; conv_*: largest angle change [deg] between RK4 steps conv_steps on 200 test cases; andes_*: ANDES's "
        "angles at t_out for andes_cases (andes_xf: the fault reactance it needed); params. The training cases are "
        "made by mp12b.training_data().")
    np.savez_compressed(path, **out)
    print(f"written {path}: {len(T['op_id'])} test cases, {100 * T['unstable'].mean():.1f} % unstable, "
          f"{time.perf_counter() - t_all:.0f} s", flush=True)


if __name__ == "__main__":
    write_truth()
