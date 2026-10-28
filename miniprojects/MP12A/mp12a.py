r"""MP12A — Stability estimation after contingencies: the critical clearing
time of every contingency on the IEEE 39-bus grid with converters. The data,
the reference model and the ground truth.

*Deep Learning for Engineering* — MSc, Aalborg University.
Remus Teodorescu (ret@et.aau.dk), with support from Research Assistant
Noman Khan (nomank@energy.aau.dk).

This module extends Ex12.2 (``problem_12_2.py``: the screen by the critical
clearing time, its graph tools and its confusion table) to the IEEE 39-bus
grid of ``grid39c.py``, and is used by the notebook of MP12A (course policy
C14).

## What changes against Ex12.2

Ex12.2 screens a six-bus grid with two machines and six contingencies. Here:

* the grid is the IEEE 39-bus (New England) system: ten classical machines
  swinging against each other, 35 contingencies (a solid fault at the first
  bus of every branch whose loss leaves the grid whole, cleared by tripping
  that branch);
* **converters replace machines**: at every operating point a share of the
  generation (0, 20, 40 or 60 %) is delivered by converters with no inertia
  (``grid39c.py`` says how they are modelled), placed at random among the nine
  local machines; the screen must know where they are;
* the critical clearing time (CCT) has no closed form for ten machines
  (Ex12.2 already had none for two with damping), so the ground truth is a
  converged simulation; the one case with a closed form - one machine
  against an infinite bus, no damping, no fault power - checks the code
  (:func:`grid39c.smib_check`).

## Two solvers

* the **reference model** (:func:`reference`): what an engineer would run for
  a new operating point - the swing equations by RK4 on a 10 ms step (half a
  cycle, the usual step of a transient stability program), the CCT of each
  of the 35 contingencies by bisection from [0, 500 ms] to an interval of
  8 ms (six halvings);
* the **ground truth** (:func:`ground_truth`): RK4 on 2 ms, bisection to
  0.01 ms (sixteen halvings), shown converged in the step and in the
  bisection tolerance (:func:`convergence_study`), checked against the exact
  one-machine case, against conservation of the transient energy, and against
  ANDES on a subset (:func:`andes_check`).
"""
from __future__ import annotations

import os
import time

import numpy as np

import grid39c as g

# ------------------------------------------------------------------ the data
N_OPS = 480                  #: operating points: 120 at each converter share
OP_SEED = 12                 #: seed of the operating points (load level and jitter)
PLACE_SEED = 22              #: seed of the converters' placement
N_TRAIN_OPS = 432            #: the first 432 operating points are for training, the last 48 for the test
SHARES = g.SHARES            #: 0, 20, 40, 60 % of the generation from converters
PROTECTION_TIME = g.PROTECTION_TIME   #: 140 ms, Ex12.2's: a case is insecure if its CCT is 140 ms or less
CCT_MAX = g.CCT_MAX          #: the search stops at 500 ms; a CCT of 500 ms means 'at least 500 ms'
FEATURES = ("P [p.u.]", "Q [p.u.]", "|V| [p.u.]", "theta [rad]", "is_machine", "is_converter", "H [100 s]", "is_fault_bus")
N_FEATURE = len(FEATURES)

TRUTH_DT, TRUTH_TOL = 0.002, 1e-5    #: ground truth: RK4 step 2 ms, bisection to 0.01 ms
REF_DT, REF_TOL = 0.010, 0.008       #: reference model: RK4 step 10 ms, bisection to an interval under 8 ms (7.8 ms)
AGREED_MEAN = 0.005                  #: s; the agreed accuracy: the CCT within 5 ms on average over the test cases
MAX_FALSE_ALARMS = 0.05              #: at most 5 % of the secure test cases flagged insecure


def share_of(i):
    """The converter share of operating point i."""
    return SHARES[i % len(SHARES)]


def build_ops():
    """The 480 operating points (load level 0.85 to 1.15 of the base case,
    each load jittered by 10 %) and, for each, the converters: a boolean
    mask over the ten machines, a share SHARES[i % 4] of the generation."""
    ops = g.operating_points(N_OPS, seed=OP_SEED)
    rng = np.random.default_rng(PLACE_SEED)
    conv = np.array([g.place_converters(op, share_of(i), rng) for i, op in enumerate(ops)])
    return ops, conv


def op_arrays(ops):
    """The operating points as arrays (for the truth file)."""
    return {k: np.array([op[k] for op in ops]) for k in ("V", "th", "P", "Q", "PL", "QL", "Pg", "Qg", "scale")}


def op_from_arrays(D, i):
    """Operating point i back from the arrays of :func:`op_arrays` (or the truth file)."""
    return {k: np.asarray(D[f"op_{k}"][i]) for k in ("V", "th", "P", "Q", "PL", "QL", "Pg", "Qg")}


def features(op, conv, fault_bus):
    """The numbers known at every bus before the fault, (39, 8): the net
    injections P and Q and the voltage |V| and angle theta from the power
    flow, whether a machine swings at the bus, whether a converter is there,
    the inertia constant of the machine in units of 100 s (0 for none or a
    converter), and whether the fault is at this bus."""
    Xf = np.zeros((g.N, N_FEATURE))
    Xf[:, 0], Xf[:, 1], Xf[:, 2], Xf[:, 3] = op["P"], op["Q"], op["V"], op["th"]
    Xf[g.GEN_BUS, 4] = ~conv
    Xf[g.GEN_BUS, 5] = conv
    Xf[g.GEN_BUS, 6] = np.where(conv, 0.0, g.H_GEN / 100.0)
    Xf[fault_bus, 7] = 1.0
    return Xf


PHYSICS_FEATURES = ("fault-on acceleration [100 rad/s²]", "imbalance after clearing [100 rad/s²]", "P_m / 2H [p.u./s]",
                    "synchronising coefficient after clearing [100 1/s²]")


def physics_features(op, conv):
    """Four numbers per machine bus for each of the 35 contingencies, (35,
    39, 4), zero elsewhere and at converters, all at the pre-fault angles:
    the acceleration of the machine during the fault, (P_m - P_e^fault) w_s / 2H,
    and just after clearing, (P_m - P_e^post) w_s / 2H, in units of 100
    rad/s²; its power per inertia P_m / 2H; and its synchronising
    coefficient after clearing, dP_e,i/d(delta_i) w_s / 2H, the square of the
    angular frequency at which it would swing alone against the others, in
    units of 100 1/s². Two Kron reductions per contingency, no simulation."""
    cs = cases(op, conv)
    M = 2 * g.H_GEN / g.WS
    out = np.zeros((g.N_CONT, g.N, 4))
    on = ~np.asarray(conv, bool)
    d, E, Yp = cs["d0"], cs["E"], cs["post"]
    dd = d[:, :, None] - d[:, None, :]
    EE = E[:, :, None] * E[:, None, :]
    k = EE * (Yp.imag * np.cos(dd) - Yp.real * np.sin(dd))          # dP_e,i / d(delta_j) for j != i, with a minus sign
    ks = k.sum(-1) - np.diagonal(k, axis1=1, axis2=2)              # dP_e,i / d(delta_i)
    out[:, g.GEN_BUS, 0] = np.where(on, (cs["Pm"] - g.electrical_power(d, E, cs["fault"])) / M, 0.0) / 100
    out[:, g.GEN_BUS, 1] = np.where(on, (cs["Pm"] - g.electrical_power(d, E, Yp)) / M, 0.0) / 100
    out[:, g.GEN_BUS, 2] = np.where(on, cs["Pm"] / (2 * g.H_GEN), 0.0)
    out[:, g.GEN_BUS, 3] = np.where(on, ks / M, 0.0) / 100
    return out


def physics_features_all(D):
    """:func:`physics_features` for every case of the truth file, (16800, 39, 4)."""
    n = len(D["share"])
    return np.concatenate([physics_features(op_from_arrays(D, i), D["converters"][i]) for i in range(n)])


def weighted_adjacency():
    """The weight of every branch, its susceptance 1/x divided by the median
    over the branches (parallel branches add), as a 39 x 39 matrix: multiplied
    into the binary adjacency, a graph layer passes more along a strong
    branch than along a weak one."""
    W = np.zeros((g.N, g.N))
    for k in range(g.N_BRANCH):
        W[g.F[k], g.T[k]] += 1 / g.X[k]
        W[g.T[k], g.F[k]] = W[g.F[k], g.T[k]]
    return W / np.median(W[W > 0])


def graphs():
    """The 35 grids after their contingency: (35, 39, 39) adjacency matrices."""
    return np.array([g.adjacency((k,)) for _, k in g.CONTINGENCIES])


def cases(op, conv):
    """The 35 contingencies of one operating point as stacked reduced networks."""
    return g.stack([g.case_networks(op, conv, fb, k) for fb, k in g.CONTINGENCIES])


def reference(op, conv, dt=REF_DT, tol=REF_TOL):
    """The reference model: the CCT of the 35 contingencies of a new
    operating point [s], RK4 on 10 ms, bisection to 8 ms. Everything an
    engineer must compute after the power flow is inside: the 71 reduced
    networks and the simulations."""
    return g.cct(cases(op, conv), dt, tol)


def ground_truth(op, conv):
    """The ground truth: the same, RK4 on 2 ms, bisection to 0.01 ms."""
    return g.cct(cases(op, conv), TRUTH_DT, TRUTH_TOL)


def convergence_study(op, conv, steps=(0.04, 0.02, 0.01, 0.005, 0.002), tols=(1e-3, 1e-4, 1e-5, 1e-6), verbose=True):
    """The CCT of the 35 contingencies of one operating point (1) on RK4
    steps each about twice as fine, bisected to 1e-7 s so the bisection does
    not hide the step's error, and (2) on the truth's 2 ms step with the
    bisection tolerance falling tenfold. Returns both tables."""
    cs = cases(op, conv)
    by_step = {dt: g.cct(cs, dt, 1e-7) for dt in steps}
    by_tol = {tol: g.cct(cs, TRUTH_DT, tol) for tol in tols}
    inside = (by_step[steps[-1]] > 0) & (by_step[steps[-1]] < CCT_MAX)
    if verbose:
        print("  RK4 step [ms]   largest change of the CCT from the previous step [ms]")
        for a, b in zip(steps[:-1], steps[1:]):
            print(f"  {1000 * a:5.0f} -> {1000 * b:3.0f}     {1000 * np.abs(by_step[b] - by_step[a])[inside].max():.2e}")
        print("  bisection to [ms]   largest change of the CCT [ms]")
        for a, b in zip(tols[:-1], tols[1:]):
            print(f"  {1000 * a:7.3f} -> {1000 * b:7.4f}   {1000 * np.abs(by_tol[b] - by_tol[a])[inside].max():.2e}")
    return by_step, by_tol


def andes_check(ops, conv, picks, verbose=True):
    """ANDES's CCT (its GENCLS model, its implicit trapezoidal method on 2 ms,
    bisection to 1 ms) against the ground truth on the cases ``picks`` =
    (operating point, contingency) pairs. Returns ANDES's CCT [s] (NaN where
    ANDES could not finish a run) and the seconds each took."""
    out, sec, milder = [], [], []
    for i, j in picks:
        fb, k = g.CONTINGENCIES[j]
        t0 = time.perf_counter()
        c, m = g.andes_cct(ops[i], conv[i], fb, k)
        out.append(c); milder.append(m)
        sec.append(time.perf_counter() - t0)
        if verbose:
            print(f"  operating point {i}, share {100 * share_of(i):.0f} %, contingency {j}: ANDES {1000 * out[-1]:.1f} ms ({sec[-1]:.0f} s)"
                  + (", milder fault in a run" if milder[-1] else ""), flush=True)
    return np.array(out), np.array(sec), np.array(milder)


TRUTH_FILE = "MP12A_truth.npz"


def write_truth(path=TRUTH_FILE, cache="mp12a_cache", n_andes=24):
    """Builds the whole data set, in blocks kept in ``cache`` so a restarted
    run continues: the operating points, the features and graphs, the ground
    truth and the reference model's CCT of all 16800 cases, the convergence
    study, the checks and the ANDES subset; then writes the truth file."""
    os.makedirs(cache, exist_ok=True)
    t_all = time.perf_counter()
    ops, conv = build_ops()
    print(f"{N_OPS} operating points, {time.perf_counter() - t_all:.0f} s", flush=True)
    block = 24
    for s in range(0, N_OPS, block):
        f = os.path.join(cache, f"block_{s:03d}.npz")
        if os.path.exists(f):
            continue
        t0 = time.perf_counter()
        cs = g.stack([c for i in range(s, s + block) for c in
                      [g.case_networks(ops[i], conv[i], fb, k) for fb, k in g.CONTINGENCIES]])
        truth = g.cct(cs, TRUTH_DT, TRUTH_TOL)
        t1 = time.perf_counter()
        ref = g.cct(cs, REF_DT, REF_TOL)
        np.savez(f, truth=truth, ref=ref, t_truth=t1 - t0, t_ref=time.perf_counter() - t1)
        print(f"  block {s}: truth {t1 - t0:.0f} s, reference {time.perf_counter() - t1:.0f} s", flush=True)
    B = [np.load(os.path.join(cache, f"block_{s:03d}.npz")) for s in range(0, N_OPS, block)]
    cct = np.concatenate([b["truth"] for b in B]); cct_ref = np.concatenate([b["ref"] for b in B])
    sec_truth = sum(float(b["t_truth"]) for b in B) / len(cct)
    f = os.path.join(cache, "checks.npz")
    if not os.path.exists(f):
        print("convergence study, operating point 0 (no converters) and 3 (60 %)", flush=True)
        st0, tl0 = convergence_study(ops[0], conv[0])
        st3, tl3 = convergence_study(ops[3], conv[3])
        print("the one-machine case against the equal area criterion", flush=True)
        smib = g.smib_check()
        print("the transient energy, lossless and undamped", flush=True)
        en = [g.energy_check(ops[i], conv[i], *g.CONTINGENCIES[j]) for i, j in ((0, 3), (3, 20))]
        steps = np.array(sorted(st0, reverse=True)); tols = np.array(sorted(tl0, reverse=True))
        np.savez(f, steps=steps, tols=tols, step_cct=np.array([np.r_[st0[s], st3[s]] for s in steps]),
                 tol_cct=np.array([np.r_[tl0[t], tl3[t]] for t in tols]), smib=smib, energy=np.array(en))
    C = np.load(f)
    f = os.path.join(cache, "andes.npz")
    if not os.path.exists(f):
        rng = np.random.default_rng(7)
        test = np.arange(N_TRAIN_OPS, N_OPS)
        picks = [(int(test[q]), int(rng.integers(g.N_CONT))) for q in rng.choice(len(test), n_andes, replace=False)]
        a, sa, mi = andes_check(ops, conv, picks)
        pw = np.array([g.andes_power_flow(ops[i], conv[i]) for i, _ in picks])
        np.savez(f, picks=np.array(picks), cct=a, seconds=sa, pf=pw, milder=mi)
    A = np.load(f)
    X = np.array([features(ops[i], conv[i], fb) for i in range(N_OPS) for fb, _ in g.CONTINGENCIES], dtype=np.float32)
    op_id = np.repeat(np.arange(N_OPS), g.N_CONT); cont_id = np.tile(np.arange(g.N_CONT), N_OPS)
    picks = A["picks"]; idx = picks[:, 0] * g.N_CONT + picks[:, 1]
    out = dict(X=X, feature_names=np.array(FEATURES), A=graphs().astype(np.uint8), cct=cct, cct_ref=cct_ref,
               op_id=op_id, cont_id=cont_id, share=np.array([share_of(i) for i in range(N_OPS)]),
               share_actual=np.array([g.converter_share(ops[i], conv[i]) for i in range(N_OPS)]), converters=conv,
               n_train_ops=np.array(N_TRAIN_OPS), fault_bus=np.array([fb for fb, _ in g.CONTINGENCIES]),
               tripped=np.array([[g.F[k], g.T[k]] for _, k in g.CONTINGENCIES]), branch=np.array([k for _, k in g.CONTINGENCIES]),
               protection_time=np.array(PROTECTION_TIME), cct_max=np.array(CCT_MAX),
               truth_dt=np.array(TRUTH_DT), truth_tol=np.array(TRUTH_TOL), ref_dt=np.array(REF_DT), ref_tol=np.array(REF_TOL),
               truth_seconds_per_case=np.array(sec_truth),
               conv_steps=C["steps"], conv_step_cct=C["step_cct"], conv_tols=C["tols"], conv_tol_cct=C["tol_cct"],
               smib_check=C["smib"], energy_check=C["energy"],
               andes_picks=picks, andes_cct=A["cct"], andes_truth=cct[idx], andes_seconds=A["seconds"], andes_power_flow=A["pf"], andes_milder=A["milder"],
               **{f"op_{k}": v for k, v in op_arrays(ops).items()})
    out["params"] = np.array([f"{k} = {v}" for k, v in dict(
        grid="IEEE 39-bus, MATPOWER case39 (Athay et al. 1979), 100 MVA base, 50 Hz (ASSUMED; case39 is 60 Hz)",
        machines="classical, H and x'_d of Pai (1989), D = H (ASSUMED)", loads="constant impedance at the pre-fault voltage (ASSUMED)",
        converters="negative constant impedance carrying the replaced machine's P and Q, no inertia (ASSUMED)",
        fault="solid, a reactance of 1e-4 p.u. to earth at the faulted bus, cleared by tripping the branch",
        criterion="unstable if two machines separate by more than 180 deg within 3 s of clearing",
        operating_points="load 0.85-1.15 of base, each load jittered 10 %, PV machines scaled alike").items()])
    out["README"] = np.array(
        "MP12A ground truth. Case n = 35 * op_id + cont_id (16800 cases). X: (16800, 39, 8) bus features named in feature_names; "
        "A: (35, 39, 39) adjacency after each contingency (cont_id indexes it); cct: ground-truth critical clearing time [s] "
        "(RK4 2 ms, bisection 0.01 ms; 0.5 = at least 500 ms); cct_ref: the reference model's (RK4 10 ms, bisection 8 ms); "
        "share, share_actual, converters (480, 10): converter share and placement per operating point; op_*: the power flow "
        "of each operating point (p.u., rad); the first n_train_ops operating points are for training; fault_bus, branch, "
        "tripped (0-based buses): the contingencies; conv_*: convergence study (CCT [s] of the 70 cases of operating "
        "points 0 and 3 per RK4 step and per bisection tolerance); smib_check: (P_m, equal-area CCT, RK4 CCT) [p.u., s]; "
        "energy_check: relative drift of the transient energy; andes_*: ANDES cross-check (picks = (op, contingency); "
        "milder: a run of the bisection needed ANDES's fault of 1e-3 p.u.); params.")
    np.savez_compressed(path, **out)
    print(f"written {path}: {len(cct)} cases, {100 * (cct <= PROTECTION_TIME).mean():.1f} % insecure, "
          f"{time.perf_counter() - t_all:.0f} s", flush=True)


if __name__ == "__main__":
    write_truth()
