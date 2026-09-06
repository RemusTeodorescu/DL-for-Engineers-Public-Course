r"""Ex_12.1 — the physics: a six-bus transmission network and the machines on it.

*Deep Learning for Engineering* — MSc, Aalborg University.
Remus Teodorescu (ret@et.aau.dk), with support from Research Assistant
Noman Khan (nomank@energy.aau.dk).

**Reference texts.** Liu, *PINN with Python: An Introduction* (2025); Raissi,
Perdikaris & Karniadakis, *Physics-informed neural networks*, J. Comput. Phys.
**378** (2019) 686–707. These are the works to read for the theory. The code,
the problem and the exposition here are original to this course, written from
the 2019 paper and the PyTorch documentation and not derived from any
publisher's code listings. Where a symbol matches a textbook's it is because
both follow the standard notation of the field.

## The problem

Recover the state of a six-bus grid from too few measurements — first as an
algebraic problem (the power flow equations), then as a dynamic one (the swing
equation) — and finally identify a machine's inertia from a disturbance.

The algebraic half asks for the voltage magnitude and angle at every bus,

    P_i = V_i Σ_k V_k ( G_ik cos θ_ik + B_ik sin θ_ik )
    Q_i = V_i Σ_k V_k ( G_ik sin θ_ik − B_ik cos θ_ik ),    θ_ik = θ_i − θ_k

given readings at a few of them. The dynamic half asks for δ(t) and ω(t) at
every machine,

    dδ_i/dt = ω_i − ω_s
    dω_i/dt = (ω_s / 2H_i) ( P_m,i − P_e,i − D_i (ω_i − ω_s) / ω_s )

given a frequency trace at one of them. Both are the same manoeuvre: the
measurements are not enough, and the model equations are what fills the gap.

This is **not a PDE over a rectangle**, and it is the first set in Part 2 that
is not. There is no spatial domain to sample; the collocation points are
instants in time, and the "boundary" is a slack bus and an initial condition.
The shared rectangle samplers in ``pinn_core`` are therefore barely used here.
Everything they would have provided — where to put the residual, and how to
score what comes back — is provided instead by the network model below.

## Network provenance — read this before quoting a number

The six-bus case is **DK2-representative, not DK2**. Its structure follows
eastern Denmark in the ways that matter for this exercise — a few load centres,
two generation buses, and two HVDC links entering as fixed injections — but the
line impedances are plausible textbook values, **not** measured ones. No TSO
publishes a nodal model with impedances; that data is withheld for security.

Any result that depends on the impedances must say so. Results that depend only
on structure (which buses are observable, how error grows with distance from a
meter) are on firmer ground.

The same six-bus case appears in lecture L5.1 and again in L12.1. It does not
change between them, and it must not change here.

## Conventions

    per unit throughout, 100 MVA base
    bus 0 is the slack bus: theta = 0, V = 1.0
    angles in radians internally, degrees only for display
    injections are generation minus load, so a load bus has negative P

## A name that is deliberately shadowed

:func:`error_table` here is **not** ``course_core.error_table``. The shared one
formats a Markdown table; this one is the per-bus estimation report this
exercise insists on, split metered from unmetered. Both names survive because
the notebooks reach this one through ``pb.error_table`` and the shared one bare.
If you write ``error_table(V_true, ...)`` without the ``pb.`` you will get the
wrong function and a confusing error; that is the only trap in this file.

## The one habit this file is built around

An estimate at an **unmetered** bus is an inference from the model. An estimate
at a **metered** bus is supported by a reading. They are not the same thing,
and every table produced here separates them — mean and worst, both, every time.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

import numpy as np
import torch
import torch.nn as nn

from pinn_core import (DEVICE, MLP, describe, grad, parameter_count,
                       to_numpy, to_tensor, train_two_stage)

__all__ = [
    # network and power flow
    "BASE_MVA", "FAULT_FACTOR", "BUSES", "BRANCHES", "N_BUS",
    "build_ybus", "injections", "pq_from_state", "solve_power_flow",
    # measurements and the classical estimator
    "MeasurementSet", "default_measurements", "thin_measurements",
    "synth_measurements", "wls_estimate", "observable",
    # dynamics
    "Machines", "equilibrium", "critical_clearing_time", "reduced_admittance",
    "electrical_power", "swing_rhs", "simulate_swing",
    "frequency_from_omega", "centre_of_inertia", "kinetic_energy",
    "rocof_initial",
    # data
    "EDS_URL", "FINGRID_FREQ_HIST", "TIMEOUT",
    "dk2_load", "nordic_frequency", "from_csv", "find_event",
    "rocof_from_series", "scale_to_network",
    # physics-informed estimators
    "StateVariables", "algebraic_pinn", "TrajectoryNet", "dynamic_pinn",
    "identify_inertia",
    # lab: persistence, tables, plots, report
    "RESULTS", "use_course_style",
    "BG", "TXT", "MUTED", "CYAN", "AMBER", "ORANGE", "GREEN", "PURPLE",
    "save", "load", "reference_state", "error_table", "comparison_table",
    "plot_network_state", "plot_sweep", "plot_swing", "plot_frequency_event",
    "plot_loss", "make_report",
]


# ══════════════════════════════════════════════════════════════════════════
# 1 · the network
# ══════════════════════════════════════════════════════════════════════════

BASE_MVA = 100.0

#: Coupling retained during the fault, as a fraction of the post-fault network.
#: A close three-phase fault collapses the transfer capability; this crude
#: factor stands in for that. It sets the critical clearing time, so it is the
#: knob to turn if you want the exercise's fault to be more or less severe.
FAULT_FACTOR = 0.04

#: bus: name, type ("slack" | "gen" | "load"), P injection, Q injection (p.u.)
BUSES = [
    ("Slack / SE link", "slack",  0.00,  0.00),
    ("Gen east",        "gen",    1.15,  0.05),
    ("Load north",      "load",  -0.58, -0.20),
    ("Load city",       "load",  -1.05, -0.36),
    ("Load south",      "load",  -0.46, -0.16),
    ("HVDC / DE link",  "load",   0.36,  0.06),
]

#: branch: from, to, r, x, b_shunt (total line charging)
BRANCHES = [
    (0, 1, 0.0035, 0.0300, 0.030),
    (1, 2, 0.0063, 0.0385, 0.022),
    (2, 3, 0.0049, 0.0333, 0.026),
    (0, 4, 0.0077, 0.0455, 0.018),
    (4, 3, 0.0056, 0.0350, 0.024),
    (3, 5, 0.0070, 0.0420, 0.020),
]

N_BUS = len(BUSES)


def build_ybus(branches=None, n=N_BUS, outage=None):
    """Bus admittance matrix. `outage` removes one branch by index."""
    br = list(BRANCHES if branches is None else branches)
    if outage is not None:
        br = [b for i, b in enumerate(br) if i != outage]
    Y = np.zeros((n, n), dtype=complex)
    for (f, t, r, x, b) in br:
        y = 1.0 / complex(r, x)
        Y[f, f] += y + 1j * b / 2
        Y[t, t] += y + 1j * b / 2
        Y[f, t] -= y
        Y[t, f] -= y
    return Y


def injections(buses=None):
    b = BUSES if buses is None else buses
    P = np.array([x[2] for x in b], dtype=float)
    Q = np.array([x[3] for x in b], dtype=float)
    return P, Q


def pq_from_state(V, th, Y):
    """Real and reactive injection at every bus, from the voltages.

    This is the residual the estimator drives to zero; it is also the
    'h(x)' of the measurement model when the measurement is an injection.
    """
    G, B = Y.real, Y.imag
    dth = th[:, None] - th[None, :]
    VV = V[:, None] * V[None, :]
    P = (VV * (G * np.cos(dth) + B * np.sin(dth))).sum(axis=1)
    Q = (VV * (G * np.sin(dth) - B * np.cos(dth))).sum(axis=1)
    return P, Q


# ══════════════════════════════════════════════════════════════════════════
# 2 · power flow
# ══════════════════════════════════════════════════════════════════════════

def solve_power_flow(Y=None, P=None, Q=None, tol=1e-10, max_iter=40,
                     verbose=False):
    """Newton-Raphson. Returns (V, theta, converged, iterations).

    Bus 0 is slack. Gen buses are treated as PQ here to keep the exercise
    short; a PV bus would fix |V| and free Q, which is the standard next step
    and is left as an extension in notebook 00.
    """
    Y = build_ybus() if Y is None else Y
    if P is None or Q is None:
        P, Q = injections()
    n = Y.shape[0]
    V = np.ones(n)
    th = np.zeros(n)
    idx = np.arange(1, n)                      # unknowns: all but slack

    for it in range(1, max_iter + 1):
        Pc, Qc = pq_from_state(V, th, Y)
        dP, dQ = P[idx] - Pc[idx], Q[idx] - Qc[idx]
        mis = np.concatenate([dP, dQ])
        if np.max(np.abs(mis)) < tol:
            return V, th, True, it
        J = _jacobian(V, th, Y, idx)
        try:
            step = np.linalg.solve(J, mis)
        except np.linalg.LinAlgError:
            return V, th, False, it
        th[idx] += step[:len(idx)]
        V[idx] += step[len(idx):]
        if verbose:
            print(f"  it {it:2d}  max mismatch {np.max(np.abs(mis)):.3e}")
    return V, th, False, max_iter


def _jacobian(V, th, Y, idx, eps=1e-7):
    """Numerical Jacobian. Slower than the analytic one and far harder to get
    wrong; at six buses the difference is invisible."""
    def f(z):
        thz, Vz = th.copy(), V.copy()
        thz[idx] = z[:len(idx)]
        Vz[idx] = z[len(idx):]
        Pc, Qc = pq_from_state(Vz, thz, Y)
        return np.concatenate([Pc[idx], Qc[idx]])

    z0 = np.concatenate([th[idx], V[idx]])
    f0 = f(z0)
    J = np.zeros((len(f0), len(z0)))
    for k in range(len(z0)):
        zp = z0.copy()
        zp[k] += eps
        J[:, k] = (f(zp) - f0) / eps
    return J


# ══════════════════════════════════════════════════════════════════════════
# 3 · what is measured
# ══════════════════════════════════════════════════════════════════════════

class MeasurementSet:
    """Which quantities are measured, with what accuracy.

    types: "V" (magnitude), "P", "Q" (injections), "TH" (PMU angle)
    """

    def __init__(self, spec, sigma=None):
        self.spec = list(spec)                       # [(kind, bus), ...]
        default = {"V": 0.004, "P": 0.010, "Q": 0.010, "TH": 0.002}
        self.sigma = np.array([(sigma or default)[k] for k, _ in self.spec])

    def __len__(self):
        return len(self.spec)

    def h(self, V, th, Y):
        P, Q = pq_from_state(V, th, Y)
        out = []
        for kind, i in self.spec:
            out.append({"V": V[i], "TH": th[i], "P": P[i], "Q": Q[i]}[kind])
        return np.array(out)

    def measured_buses(self):
        return sorted({i for _, i in self.spec})

    def unmeasured_buses(self, n=N_BUS):
        m = set(self.measured_buses())
        return [i for i in range(n) if i not in m]


def default_measurements(pmu_buses=(0, 3), scada_buses=(0, 1, 2, 3, 4)):
    """A workable set: PMUs at two buses, SCADA injections at five.

    This is observable - check it with observable() rather than trusting the
    docstring - and it is what the WLS baseline in notebook 01 uses.
    """
    spec = []
    for i in pmu_buses:
        spec += [("V", i), ("TH", i)]
    for i in scada_buses:
        spec += [("P", i), ("Q", i)]
    return MeasurementSet(spec)


def thin_measurements(pmu_buses=(0,), scada_buses=(0, 3)):
    """Deliberately too few. Not observable - WLS needs pseudo-measurements
    here, and this is the case where the physics residual has something to do.

    Notebook 02 compares the two sets; the difference between them is the
    entire argument of the exercise.
    """
    spec = []
    for i in pmu_buses:
        spec += [("V", i), ("TH", i)]
    for i in scada_buses:
        spec += [("P", i), ("Q", i)]
    return MeasurementSet(spec)


def synth_measurements(V, th, Y, ms, rng=None, bad=None):
    """Generate readings from a known state. `bad` = (index, offset) injects
    one gross error, for the bad-data part of notebook 01."""
    rng = np.random.default_rng(0) if rng is None else rng
    z = ms.h(V, th, Y) + ms.sigma * rng.standard_normal(len(ms))
    if bad is not None:
        k, off = bad
        z[k] += off
    return z


# ══════════════════════════════════════════════════════════════════════════
# 4 · the classical estimator
# ══════════════════════════════════════════════════════════════════════════

def wls_estimate(z, ms, Y, max_iter=40, tol=1e-10, flat=True, ridge=1e-8):
    """Gauss-Newton weighted least squares - the baseline every utility runs.

    Returns (V, theta, info). info carries iterations, convergence, the
    normalised residuals and the objective, which notebook 01 uses for
    bad-data detection.
    """
    n = Y.shape[0]
    V = np.ones(n) if flat else None
    th = np.zeros(n)
    # The state vector is NOT symmetric in angle and magnitude, and this is
    # where state estimation differs from load flow.
    #
    # In load flow the slack bus fixes both quantities: its angle is the
    # reference and its magnitude is a specified boundary condition. Neither is
    # solved for, which is why the solver above excludes bus 0 from both.
    #
    # In state estimation only the angle reference is free to choose. Angles are
    # observable only relative to one another, so one must be pinned, and bus 0
    # is the conventional choice. Voltage MAGNITUDES are absolute physical
    # quantities: every one of them, including the slack bus, is estimated from
    # the measurements. Fixing |V| at bus 0 would assert a value the data was
    # never asked about.
    #
    # So for n buses the state has (n-1) angles and n magnitudes, 2n-1 in all.
    idx_th = np.arange(1, n)          # angles: all but the reference
    idx_V = np.arange(n)              # magnitudes: every bus
    W = np.diag(1.0 / ms.sigma ** 2)

    for it in range(1, max_iter + 1):
        r = z - ms.h(V, th, Y)
        H = _meas_jacobian(V, th, Y, ms, idx_th, idx_V)
        G = H.T @ W @ H + ridge * np.eye(H.shape[1])
        try:
            step = np.linalg.solve(G, H.T @ W @ r)
        except np.linalg.LinAlgError:
            return V, th, {"converged": False, "iterations": it}
        th[idx_th] += step[:len(idx_th)]
        V[idx_V] += step[len(idx_th):]
        if np.max(np.abs(step)) < tol:
            break

    r = z - ms.h(V, th, Y)
    H = _meas_jacobian(V, th, Y, ms, idx_th, idx_V)
    G = H.T @ W @ H + ridge * np.eye(H.shape[1])
    try:
        S = np.eye(len(z)) - H @ np.linalg.solve(G, H.T @ W)
        omega = np.diag(S @ np.diag(ms.sigma ** 2))
        rn = np.abs(r) / np.sqrt(np.maximum(omega, 1e-18))
    except np.linalg.LinAlgError:
        rn = np.full(len(z), np.nan)

    return V, th, {"converged": it < max_iter, "iterations": it,
                   "J": float(r @ W @ r), "residuals": r,
                   "normalised_residuals": rn}


def _meas_jacobian(V, th, Y, ms, idx_th, idx_V=None, eps=1e-7):
    """Numerical Jacobian of the measurement function.

    idx_th and idx_V are the buses whose angle and whose magnitude are state
    variables. They differ: the reference angle is excluded, every magnitude is
    included. Passing only idx_th reproduces the old behaviour, which is what
    the load-flow Jacobian wants.
    """
    idx_V = idx_th if idx_V is None else idx_V

    def f(zv):
        thz, Vz = th.copy(), V.copy()
        thz[idx_th] = zv[:len(idx_th)]
        Vz[idx_V] = zv[len(idx_th):]
        return ms.h(Vz, thz, Y)
    z0 = np.concatenate([th[idx_th], V[idx_V]])
    f0 = f(z0)
    H = np.zeros((len(f0), len(z0)))
    for k in range(len(z0)):
        zp = z0.copy()
        zp[k] += eps
        H[:, k] = (f(zp) - f0) / eps
    return H


def observable(ms, Y, idx_th=None, idx_V=None):
    """Rank test on the measurement Jacobian at flat start.

    The state is (n-1) angles and n magnitudes, so full rank is 2n-1. For the
    six-bus network that is 11, and a rank below it names the number of
    independent quantities the measurement set can actually determine.
    """
    n = Y.shape[0]
    idx_th = np.arange(1, n) if idx_th is None else idx_th
    idx_V = np.arange(n) if idx_V is None else idx_V
    H = _meas_jacobian(np.ones(n), np.zeros(n), Y, ms, idx_th, idx_V)
    rank = np.linalg.matrix_rank(H, tol=1e-8)
    return rank == H.shape[1], int(rank), int(H.shape[1])


# ══════════════════════════════════════════════════════════════════════════
# 5 · dynamics
# ══════════════════════════════════════════════════════════════════════════

class Machines:
    """Classical machine model for the two generation buses.

    H in seconds on machine base, D in p.u. torque per p.u. speed. Defaults are
    ESTIMATED - typical values for plant of this size, not measurements of any
    real unit. Notebook 04 asks you to recover them and compare.
    """

    def __init__(self, buses=(0, 1), H=(150.0, 4.0), D=(2.0, 1.2),
                 E=(1.05, 1.03), Pm=None, f0=50.0):
        """Machine 0 is the Nordic system seen through the Swedish link: a very
        large inertia that barely moves, i.e. an infinite bus in all but name.
        Machine 1 is the local plant, and it is the one that swings. H and D
        for machine 1 are ESTIMATED - plausible for a unit of this size, not
        measured - and notebook 04 asks you to recover them."""
        self.buses = list(buses)
        self.H = np.array(H, dtype=float)
        self.D = np.array(D, dtype=float)
        self.E = np.array(E, dtype=float)
        self.f0 = f0
        self.ws = 2 * np.pi * f0
        self.Pm = np.array([-1.0, 1.15]) if Pm is None else np.array(Pm, float)

    def __len__(self):
        return len(self.buses)


def equilibrium(machines, Yred, Pm1=None):
    """Self-consistent pre-fault operating point for the reduced system.

    Machine 0 is the angle reference (delta = 0). Solve for the other machines'
    angles so that their electrical power equals their mechanical power, then
    set machine 0's mechanical power to whatever it is delivering. The result
    is an exact equilibrium of the model you are about to integrate - start
    anywhere else and the machines swing before the fault, which makes every
    later plot unreadable.
    """
    from scipy.optimize import brentq
    nm = len(machines)
    Pm = machines.Pm.copy() if Pm1 is None else np.asarray(Pm1, float).copy()
    delta = np.zeros(nm)
    for k in range(1, nm):
        def r(d):
            dv = delta.copy(); dv[k] = d
            return electrical_power(dv, machines.E, Yred)[k] - Pm[k]
        lo, hi = -1.2, 1.2
        if r(lo) * r(hi) > 0:
            raise RuntimeError("no equilibrium for machine %d - Pm too large "
                               "for the coupling; lower Pm or strengthen the "
                               "network" % k)
        delta[k] = brentq(r, lo, hi, xtol=1e-12)
    Pe = electrical_power(delta, machines.E, Yred)
    Pm[0] = Pe[0]
    machines.Pm = Pm
    return delta, Pm


def critical_clearing_time(machines, Yred_pre, Yred_post, delta0,
                           Yred_fault=None, t_fault=0.5, t_end=4.0,
                           lo=0.0, hi=1.0, tol=1e-3, limit=np.pi):
    """Bisection on clearing time: the longest fault the system survives.

    'Survives' means the maximum angle separation stays below `limit`. Crude
    but honest, and it is the number notebook 03 compares its prediction to.
    """
    def stable(tc):
        T, D, W = simulate_swing(machines, Yred_pre, Yred_post, t_end=t_end,
                                 t_fault=t_fault, t_clear=t_fault + tc,
                                 Yred_fault=Yred_fault, delta0=delta0)
        sep = D.max(axis=1) - D.min(axis=1)
        return np.max(np.abs(sep)) < limit
    if not stable(lo):
        return 0.0
    if stable(hi):
        return hi
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if stable(mid):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def reduced_admittance(Y, machines, V, th, P, Q):
    """Kron-reduce the network onto the machine internal nodes.

    Loads become constant impedances at their operating point. This is the
    standard classical-model reduction, and it is a strong assumption: a real
    load is not a constant impedance, and the exercise asks you to say what
    that buys and costs.
    """
    n = Y.shape[0]
    Yl = Y.copy()
    for i in range(n):
        S = complex(-P[i], -Q[i])              # load draws power
        if abs(S) > 1e-12 and V[i] > 1e-9:
            Yl[i, i] += np.conj(S) / V[i] ** 2
    gen = list(machines.buses)
    load = [i for i in range(n) if i not in gen]
    if not load:
        return Yl
    A = Yl[np.ix_(gen, gen)]
    B = Yl[np.ix_(gen, load)]
    C = Yl[np.ix_(load, gen)]
    D = Yl[np.ix_(load, load)]
    return A - B @ np.linalg.solve(D, C)


def electrical_power(delta, E, Yred):
    """P_e for each machine given internal angles."""
    G, B = Yred.real, Yred.imag
    dd = delta[:, None] - delta[None, :]
    EE = E[:, None] * E[None, :]
    return (EE * (G * np.cos(dd) + B * np.sin(dd))).sum(axis=1)


def swing_rhs(delta, omega, machines, Yred, Pm=None):
    """First-order form. Returns (ddelta, domega) - never integrate the
    second-order form directly."""
    Pm = machines.Pm if Pm is None else Pm
    Pe = electrical_power(delta, machines.E, Yred)
    dd = omega - machines.ws
    dw = (machines.ws / (2 * machines.H)) * (Pm - Pe - machines.D * dd / machines.ws)
    return dd, dw


def simulate_swing(machines, Yred_pre, Yred_post, t_end=3.0, dt=0.002,
                   t_fault=0.5, t_clear=0.62, Yred_fault=None, delta0=None):
    """RK4 through pre-fault, fault-on and post-fault networks.

    Returns t, delta (nt, nm), omega (nt, nm). The fault is represented by a
    weakened network rather than by a bus short, which keeps the reduction
    valid and the code short; a real study would do better.
    """
    nm = len(machines)
    if Yred_fault is None:
        Yred_fault = Yred_post * FAULT_FACTOR
    if delta0 is None:
        delta0 = np.zeros(nm)
    d = np.array(delta0, dtype=float)
    w = np.full(nm, machines.ws)
    nt = int(round(t_end / dt)) + 1
    T = np.zeros(nt); D = np.zeros((nt, nm)); W = np.zeros((nt, nm))
    D[0], W[0] = d, w

    def Y_at(t):
        if t < t_fault:
            return Yred_pre
        if t < t_clear:
            return Yred_fault
        return Yred_post

    for k in range(1, nt):
        t = (k - 1) * dt
        Yk = Y_at(t)

        def f(dv, wv):
            return swing_rhs(dv, wv, machines, Yk)

        k1d, k1w = f(d, w)
        k2d, k2w = f(d + dt/2*k1d, w + dt/2*k1w)
        k3d, k3w = f(d + dt/2*k2d, w + dt/2*k2w)
        k4d, k4w = f(d + dt*k3d, w + dt*k3w)
        d = d + dt/6*(k1d + 2*k2d + 2*k3d + k4d)
        w = w + dt/6*(k1w + 2*k2w + 2*k3w + k4w)
        T[k], D[k], W[k] = k*dt, d, w
    return T, D, W


def frequency_from_omega(W, f0=50.0):
    """Machine speeds to frequency in Hz."""
    return f0 * W / (2 * np.pi * f0)


def centre_of_inertia(D, machines):
    Hs = machines.H
    return (D * Hs).sum(axis=1) / Hs.sum()


def kinetic_energy(machines, S_rating=None):
    """E_k = sum H_i S_i, in MW-seconds on the given ratings."""
    S = np.full(len(machines), BASE_MVA) if S_rating is None else np.asarray(S_rating)
    return float((machines.H * S).sum())


def rocof_initial(dP_mw, Ek_mws, f0=50.0):
    """Initial RoCoF for a power imbalance, from the kinetic energy."""
    return f0 * dP_mw / (2.0 * Ek_mws)


# ══════════════════════════════════════════════════════════════════════════
# 6 · real Danish and Nordic data, with an offline fallback
# ══════════════════════════════════════════════════════════════════════════
#
# Two public sources, both free and neither needing an API key:
#
#     Energinet Energi Data Service   production, consumption and exchanges per
#                                     price area, including DK2
#                                     https://www.energidataservice.dk/
#
#     Fingrid open data               frequency of the Nordic synchronous area,
#                                     historical and real time
#                                     https://data.fingrid.fi/en/datasets/339
#
# Why Nordic frequency is DK2's frequency
#     Denmark is split. Western Denmark (DK1) belongs to the Continental
#     European synchronous area; eastern Denmark (DK2, Zealand) belongs to the
#     Nordic one, connected to Sweden by AC. The two halves are joined only by
#     the Great Belt HVDC link, which transfers power but not synchronism. So
#     when a unit trips in Finland or Sweden, Zealand feels it and Jutland does
#     not, and the Nordic frequency measurement is the right one for a DK2
#     machine.
#
# Offline fallback
#     Every loader here falls back to a bundled synthetic series if the network
#     is unavailable - Colab sometimes is, exam conditions usually are. The
#     fallback is clearly flagged in the returned dict as source="synthetic".
#     A result computed on synthetic data is not a result about Denmark, and
#     the report template asks you to state which you used.

EDS_URL = "https://api.energidataservice.dk/dataset/"
FINGRID_FREQ_HIST = "https://data.fingrid.fi/en/datasets/339"
TIMEOUT = 20


def _get_json(url, timeout=TIMEOUT):
    req = urllib.request.Request(url, headers={"User-Agent": "AppliedPINN/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf8"))


def _offline(reason):
    print(f"  [data] falling back to synthetic series: {reason}")


def dk2_load(start="2026-01-15T00:00", end="2026-01-16T00:00", verbose=True):
    """Hourly DK2 consumption in MW.

    Returns {"hours": (24,), "load_mw": (24,), "source": "energinet"|"synthetic"}.

    The exercise uses the shape of this curve, scaled onto the six-bus network,
    so that the operating point you estimate is a real one rather than a number
    somebody invented.
    """
    url = (EDS_URL + "ConsumptionPerPriceArea?offset=0"
           f"&start={start}&end={end}"
           "&filter=%7B%22PriceArea%22:[%22DK2%22]%7D"
           "&sort=HourUTC%20ASC")
    try:
        js = _get_json(url)
        rec = js.get("records", [])
        if not rec:
            raise ValueError("no records returned")
        mw = np.array([r.get("ConsumptionMWh") or r.get("ConsumptionTotal") or 0.0
                       for r in rec], dtype=float)
        if verbose:
            print(f"  [data] Energinet: {len(mw)} hourly points for DK2")
        return {"hours": np.arange(len(mw)), "load_mw": mw, "source": "energinet"}
    except Exception as e:                       # noqa: BLE001 - any failure -> offline
        if verbose:
            _offline(f"{type(e).__name__}")
        return _synthetic_load()


def _synthetic_load():
    """A plausible DK2 winter weekday: night trough, morning ramp, evening peak.

    Magnitudes are of the right order for eastern Denmark but are NOT a
    measurement. Flagged as synthetic so it cannot be mistaken for one.
    """
    h = np.arange(24)
    base = 1500.0
    shape = (0.78 + 0.10 * np.sin((h - 3) / 24 * 2 * np.pi)
             + 0.16 * np.exp(-((h - 8.5) ** 2) / 6.0)
             + 0.26 * np.exp(-((h - 18.0) ** 2) / 7.0))
    return {"hours": h, "load_mw": base * shape, "source": "synthetic"}


def nordic_frequency(n_seconds=60.0, rate_hz=10.0, event=True, verbose=True):
    """Nordic synchronous area frequency around a disturbance.

    Returns {"t": (n,), "f_hz": (n,), "source": ...}.

    The live Fingrid API needs a key for bulk historical download, so this
    loader ships a synthetic event by default and tells you so. If you have
    downloaded a real series, pass it to `from_csv` below instead - the rest of
    the exercise does not care where the numbers came from, only that you say.
    """
    if verbose:
        print("  [data] using a synthetic frequency event; see from_csv() to "
              "use a real Fingrid download")
    return _synthetic_frequency(n_seconds, rate_hz, event)


def _synthetic_frequency(n_seconds=60.0, rate_hz=10.0, event=True, f0=50.0,
                         rocof0=-0.28, seed=7):
    """A generation trip, with the three regimes the lecture separates.

        0 to ~1 s     inertial response - a straight fall, slope set by E_k
        1 to ~30 s    primary reserve arrests it, frequency reaches a nadir
        30 s onward   secondary control walks it back

    Built as ONE smooth function rather than glued segments. A piecewise
    construction whose pieces do not match at the join puts a step in the
    series, and a step is exactly what a RoCoF estimator will lock onto - you
    would measure the seam, not the physics.

        f(t) = f0 - depth (1 - e^{-dt/tau_a}) + rec (1 - e^{-dt/tau_r})

    with depth/tau_a set so the initial slope is the RoCoF you asked for.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, n_seconds, 1.0 / rate_hz)
    f = np.full_like(t, f0)
    if event:
        t_ev = 5.0
        tau_a, tau_r = 1.47, 45.0
        rec = 0.23
        depth = (-rocof0 + rec / tau_r) * tau_a     # so f'(0) = rocof0
        m = t >= t_ev
        dt = t[m] - t_ev
        f[m] = (f0 - depth * (1 - np.exp(-dt / tau_a))
                + rec * (1 - np.exp(-dt / tau_r)))
    f = f + 0.004 * rng.standard_normal(t.shape)     # PMU-class noise
    return {"t": t, "f_hz": f, "source": "synthetic", "f0": f0,
            "true_rocof": rocof0 if event else 0.0,
            "true_event_time": 5.0 if event else None}


def from_csv(path, t_col=0, f_col=1, skip=1, verbose=True):
    """Load a frequency series you downloaded yourself.

    Expects two columns: seconds (or any monotonic time) and frequency in Hz.
    """
    arr = np.loadtxt(path, delimiter=",", skiprows=skip, usecols=(t_col, f_col))
    t, f = arr[:, 0], arr[:, 1]
    t = t - t[0]
    if verbose:
        print(f"  [data] {len(t)} points from {path}, "
              f"{t[-1]:.1f} s, f in [{f.min():.3f}, {f.max():.3f}] Hz")
    return {"t": t, "f_hz": f, "source": f"csv:{path}", "f0": 50.0}


def find_event(t, f, f0=50.0, drop=0.05):
    """First time the frequency falls `drop` Hz below nominal.

    Do NOT locate the event by the largest jump between samples - on a noisy
    PMU series that finds the noisiest sample, not the disturbance. Ask for a
    sustained departure instead.
    """
    below = np.where(f < f0 - drop)[0]
    if len(below) == 0:
        return None
    k = int(below[0])
    # step back to where the decline actually began
    while k > 0 and f[k - 1] < f0 - 0.005:
        k -= 1
    return float(t[k])


def rocof_from_series(t, f, window=1.0, t_event=None, f0=50.0):
    """Least-squares slope over `window` seconds after the event, in Hz/s.

    Differentiating noisy data sample by sample manufactures a number that is
    mostly noise. Fitting a straight line over a stated window is the least bad
    simple answer - and stating the window is the important part, because the
    value depends on it.
    """
    if t_event is None:
        t_event = find_event(t, f, f0)
    if t_event is None:
        return float("nan"), None
    m = (t >= t_event) & (t <= t_event + window)
    if m.sum() < 3:
        return float("nan"), t_event
    A = np.vstack([t[m] - t_event, np.ones(m.sum())]).T
    slope, _ = np.linalg.lstsq(A, f[m], rcond=None)[0]
    return float(slope), float(t_event)


def scale_to_network(load_mw, buses_p, hour=None, ref_hour=None):
    """Map the SHAPE of a real load curve onto the six-bus case.

    The six-bus network is a reduced representation, so its per-unit loading is
    not DK2's megawatts - dropping 1500 MW onto a case designed to carry 209 MW
    diverges the power flow, which is the first thing everyone tries.

    What transfers is the shape. The peak hour of the real curve is mapped onto
    the network's design loading, and every other hour scales in proportion, so
    the daily variation you estimate through is real even though the magnitude
    is the network's own.

    Returns (P_scaled, hour, implied_base_mva). Quote the implied base in your
    report - it is the honest statement of what the per-unit numbers mean.
    """
    load_mw = np.asarray(load_mw, dtype=float)
    p = np.array(buses_p, dtype=float)
    load_idx = p < 0
    ref = int(np.argmax(load_mw)) if ref_hour is None else int(ref_hour)
    hour = ref if hour is None else int(hour)

    design_total = -p[load_idx].sum()                 # p.u. at the design point
    factor = load_mw[hour] / load_mw[ref]             # real, relative to peak
    share = p[load_idx] / p[load_idx].sum()

    out = p.copy()
    out[load_idx] = -design_total * factor * share
    implied_base = load_mw[ref] / design_total
    return out, hour, float(implied_base)


# ══════════════════════════════════════════════════════════════════════════
# 7 · the physics-informed estimators
# ══════════════════════════════════════════════════════════════════════════
#
# Two notes on the plumbing, because both bite silently.
#
# **Shapes.** ``to_tensor`` promotes everything to at least two dimensions,
# which is right for collocation points shaped ``(N, d)`` and wrong for the
# per-bus and per-machine vectors used here. Those broadcast against
# ``(N, n_machine)`` arrays, and a stray leading axis transposes the physics
# without raising. :func:`_flat` is the one-line fix, used everywhere below.
#
# **L-BFGS steps.** ``train_two_stage`` runs ``lbfgs_steps`` *outer* steps of
# up to 20 inner iterations each, so ``lbfgs_steps=20`` here is about 400
# L-BFGS iterations. The numbers in the notebooks are chosen against that
# convention; multiply by twenty before comparing them with a plain
# ``max_iter``.


def _flat(a):
    """A one-dimensional tensor on :data:`DEVICE`, whatever ``a`` came in as.

    ``to_tensor`` is built for point sets and gives back ``(1, n)`` for a plain
    vector. Every per-bus and per-machine quantity in this section wants
    ``(n,)`` so that it broadcasts along the last axis of a ``(N, n)`` batch.
    """
    return to_tensor(a).reshape(-1)


# ── algebraic state estimation ────────────────────────────────────────────

class StateVariables(nn.Module):
    """The state itself as free parameters, with the reference built in.

    There is no neural network here, and that is deliberate. For a single grid
    at a single instant there is nothing to generalise over, so the honest
    parameterisation is the state vector itself - which makes the comparison
    with WLS a comparison of loss functions rather than of architectures.

    Hard constraints, as in L7.1:
        slack angle is not a variable at all, so it is exactly zero
        magnitudes pass through a scaled sigmoid, so they cannot leave the band
    """

    def __init__(self, n_bus, v_min=0.90, v_max=1.10, slack=0):
        super().__init__()
        self.n_bus, self.slack = n_bus, slack
        self.v_min, self.v_max = v_min, v_max
        self.raw_v = nn.Parameter(torch.zeros(n_bus))
        self.raw_th = nn.Parameter(torch.zeros(n_bus - 1))

    def forward(self):
        V = self.v_min + (self.v_max - self.v_min) * torch.sigmoid(self.raw_v)
        z = torch.zeros(1, dtype=self.raw_th.dtype, device=self.raw_th.device)
        th = torch.cat([z, self.raw_th]) if self.slack == 0 else None
        if th is None:
            raise NotImplementedError("slack bus other than 0")
        return V, th


def _pq_torch(V, th, G, B):
    dth = th[:, None] - th[None, :]
    VV = V[:, None] * V[None, :]
    P = (VV * (G * torch.cos(dth) + B * torch.sin(dth))).sum(dim=1)
    Q = (VV * (G * torch.sin(dth) - B * torch.cos(dth))).sum(dim=1)
    return P, Q


def algebraic_pinn(z, ms, Y, lam_pf=1.0, adam_steps=4000, lbfgs_steps=20,
                   lr=1e-2, v_band=(0.90, 1.10), verbose=True):
    """State estimation with the power flow equations as a residual.

    lam_pf is the whole experiment. At 0 this is least squares on the
    measurements alone, and unmetered buses are undetermined. Very large, and
    it becomes a power flow solution that ignores your meters. Sweep it.

    Returns ``(V, theta, history)``. The history is what ``train_two_stage``
    returned - ``"adam"`` and ``"lbfgs"`` - with the two loss terms ``"meas"``
    and ``"pf"`` recorded alongside, one entry per evaluation of the loss.
    Keeping the terms apart is the point: it is the term that *stops*
    improving that tells you what went wrong, and a single total hides it.
    """
    n = Y.shape[0]
    G = to_tensor(Y.real)
    B = to_tensor(Y.imag)
    zt = _flat(z)
    sig = _flat(ms.sigma)
    kinds = [k for k, _ in ms.spec]
    idxs = [i for _, i in ms.spec]

    sv = StateVariables(n, *v_band).to(DEVICE)

    def h_torch(V, th):
        P, Q = _pq_torch(V, th, G, B)
        out = []
        for k, i in zip(kinds, idxs):
            out.append({"V": V[i], "TH": th[i], "P": P[i], "Q": Q[i]}[k])
        return torch.stack(out)

    trace = {"meas": [], "pf": []}

    def terms():
        V, th = sv()
        r = (zt - h_torch(V, th)) / sig
        L_meas = (r ** 2).mean()
        P, Q = _pq_torch(V, th, G, B)
        # The residual: injections must be consistent with the voltages at
        # every bus with no injection source of its own. With no injection
        # measurement at a bus we still know the network equation holds; what
        # we do not know is the right-hand side, so we impose consistency of
        # the *network* by penalising mismatch against the measured injections
        # where they exist and leaving the rest free.
        mask = torch.zeros(n, dtype=zt.dtype, device=DEVICE)
        tgtP = torch.zeros(n, dtype=zt.dtype, device=DEVICE)
        tgtQ = torch.zeros(n, dtype=zt.dtype, device=DEVICE)
        for k, i, zz in zip(kinds, idxs, zt):
            if k == "P":
                mask[i] = 1.0
                tgtP[i] = zz
            elif k == "Q":
                tgtQ[i] = zz
        L_pf = (mask * ((P - tgtP) ** 2 + (Q - tgtQ) ** 2)).sum() / max(mask.sum(), 1)
        return {"meas": L_meas, "pf": lam_pf * L_pf}

    def loss_fn():
        # enable_grad, not because this residual differentiates through its
        # inputs — it does not — but because the two below do, and all three
        # are called the same way. See the note above :func:`_flat`.
        with torch.enable_grad():
            t = terms()
        trace["meas"].append(float(t["meas"]))
        trace["pf"].append(float(t["pf"]))
        return t["meas"] + t["pf"]

    if verbose:
        print(f"  algebraic estimator   lambda_pf = {lam_pf:g}")
        print(f"  free parameters : {parameter_count(sv)}"
              f"   ({n} magnitudes + {n - 1} angles, slack angle is not one)")
        print(f"  measurements    : {len(ms)}")

    history = train_two_stage(sv, loss_fn, adam_steps=adam_steps,
                              lbfgs_steps=lbfgs_steps, lr=lr,
                              report_every=max(adam_steps // 4, 1) if verbose else 0)
    history = dict(history)
    history["meas"] = np.asarray(trace["meas"])
    history["pf"] = np.asarray(trace["pf"])
    if verbose:
        print(f"  final terms     : meas {history['meas'][-1]:.3e}"
              f"   pf {history['pf'][-1]:.3e}")

    with torch.no_grad():
        V, th = sv()
    return to_numpy(V).ravel(), to_numpy(th).ravel(), history


# ── dynamic state estimation ──────────────────────────────────────────────

class TrajectoryNet(nn.Module):
    """delta(t) and omega(t) for every machine, with the initial condition
    built in rather than penalised - the L7.2 habit.

        delta(t) = delta0 + (1 - exp(-t/tau)) * net(t)

    so delta(0) = delta0 exactly, for any weights.
    """

    def __init__(self, n_machines, delta0, omega0, tau=0.05, width=64, depth=5):
        super().__init__()
        self.nm = n_machines
        self.register_buffer("delta0", _flat(delta0))
        self.register_buffer("omega0", _flat(omega0))
        self.tau = tau
        self.net = MLP(n_in=1, n_out=2 * n_machines,
                       n_hidden=width, n_layers=depth)

    def forward(self, t):
        out = self.net(t)
        gate = 1.0 - torch.exp(-t / self.tau)
        d = self.delta0 + gate * out[:, :self.nm]
        w = self.omega0 + gate * out[:, self.nm:]
        return d, w


def _swing_residual(net, tc, E, H, D, Pm, Gr, Br, ws, nm):
    """The swing equation, evaluated on collocation times ``tc``.

    Returns ``(r1, r2)`` — the two first-order residuals. Written once and
    called from both :func:`dynamic_pinn` and :func:`identify_inertia`, which
    differ only in whether H and D are constants or unknowns.

    ``tc`` must carry ``requires_grad=True``; it is what the time derivatives
    are taken with respect to.
    """
    d_c, w_c = net(tc)
    dd_dt = torch.stack([grad(d_c[:, k:k + 1], tc).squeeze(-1)
                         for k in range(nm)], dim=1)
    dw_dt = torch.stack([grad(w_c[:, k:k + 1], tc).squeeze(-1)
                         for k in range(nm)], dim=1)
    dd = d_c[:, :, None] - d_c[:, None, :]
    EE = E[None, :, None] * E[None, None, :]
    Pe = (EE * (Gr * torch.cos(dd) + Br * torch.sin(dd))).sum(dim=2)
    r1 = dd_dt - (w_c - ws)
    r2 = dw_dt - (ws / (2 * H)) * (Pm - Pe - D * (w_c - ws) / ws)
    return r1, r2


def dynamic_pinn(t_obs, f_obs, machines, Yred, delta0, omega0,
                 lam_dyn=1.0, n_coll=600, t_end=None, adam_steps=4000,
                 lbfgs_steps=20, lr=3e-3, verbose=True):
    """Fit machine trajectories to frequency observations plus the swing equation.

    t_obs, f_obs: observation times and measured frequency (Hz) - typically
    only for the machines that are actually metered.

    Returns ``(net, history)``, with ``"data"`` and ``"dyn"`` recorded in the
    history beside ``"adam"`` and ``"lbfgs"``.
    """
    t_end = float(t_obs.max()) if t_end is None else t_end
    nm = len(machines)
    net = TrajectoryNet(nm, delta0, omega0).to(DEVICE)

    Gr = to_tensor(Yred.real)
    Br = to_tensor(Yred.imag)
    E = _flat(machines.E)
    H = _flat(machines.H)
    D = _flat(machines.D)
    Pm = _flat(machines.Pm)
    ws = machines.ws

    tt = to_tensor(np.asarray(t_obs).reshape(-1, 1))
    ff = to_tensor(f_obs)
    # A regular grid, not a Latin hypercube: L-BFGS re-evaluates the loss
    # several times per step and must see the same points each time, and a
    # one-dimensional time axis is dense enough at 600 points that the
    # stratification the shared samplers buy has nothing left to buy.
    tc = to_tensor(np.linspace(0.0, t_end, n_coll).reshape(-1, 1),
                   requires_grad=True)

    trace = {"data": [], "dyn": []}

    def terms():
        _, w_o = net(tt)
        f_pred = w_o / (2 * np.pi)
        L_data = ((f_pred[:, :ff.shape[1]] - ff) ** 2).mean()
        r1, r2 = _swing_residual(net, tc, E, H, D, Pm, Gr, Br, ws, nm)
        L_dyn = (r1 ** 2).mean() + (r2 ** 2).mean() / (ws ** 2)
        return {"data": L_data, "dyn": lam_dyn * L_dyn}

    def loss_fn():
        # enable_grad is load-bearing: the swing residual differentiates the
        # network with respect to t, and ``train_two_stage`` evaluates the loss
        # once per L-BFGS step inside ``torch.no_grad()`` to record its history.
        # Without this the forward pass there builds no graph and ``grad``
        # raises. It costs nothing when grad is already on.
        with torch.enable_grad():
            t = terms()
        trace["data"].append(float(t["data"]))
        trace["dyn"].append(float(t["dyn"]))
        return t["data"] + t["dyn"]

    if verbose:
        print(f"  dynamic estimator   lambda_dyn = {lam_dyn:g}")
        describe(net.net, n_coll)

    history = train_two_stage(net, loss_fn, adam_steps=adam_steps,
                              lbfgs_steps=lbfgs_steps, lr=lr,
                              report_every=max(adam_steps // 4, 1) if verbose else 0)
    history = dict(history)
    history["data"] = np.asarray(trace["data"])
    history["dyn"] = np.asarray(trace["dyn"])
    if verbose:
        print(f"  final terms     : data {history['data'][-1]:.3e}"
              f"   dyn {history['dyn'][-1]:.3e}")
    return net, history


def identify_inertia(t_obs, f_obs, machines, Yred, delta0, omega0,
                     H_init=None, lam_dyn=1.0, n_coll=600, adam_steps=4000,
                     lbfgs_steps=15, lr=3e-3, verbose=True):
    """The inverse problem: H (and D) as trainable variables.

    Returns (H_hat, D_hat, net, history). Read the warning in the lecture
    before believing the answer: in a quiet window H barely enters the
    equations, and the optimiser will still return a confident number.

    H and D are carried as ``log`` parameters attached to the trajectory
    network itself. ``train_two_stage`` optimises ``model.parameters()``, and
    an ``nn.Parameter`` assigned to a module is registered by that assignment —
    so the unknown physical constants and the network weights go into one
    optimiser without the caller having to assemble a parameter list. The log
    keeps them positive, which a negative inertia would not be.
    """
    nm = len(machines)
    t_end = float(np.asarray(t_obs).max())
    net = TrajectoryNet(nm, delta0, omega0).to(DEVICE)

    H0 = machines.H if H_init is None else np.asarray(H_init, float)
    net.log_H = nn.Parameter(_flat(np.log(H0)).clone())
    net.log_D = nn.Parameter(_flat(np.log(np.maximum(machines.D, 1e-3))).clone())

    Gr, Br = to_tensor(Yred.real), to_tensor(Yred.imag)
    E, Pm = _flat(machines.E), _flat(machines.Pm)
    ws = machines.ws
    tt = to_tensor(np.asarray(t_obs).reshape(-1, 1))
    ff = to_tensor(f_obs)
    tc = to_tensor(np.linspace(0.0, t_end, n_coll).reshape(-1, 1),
                   requires_grad=True)

    trace = {"data": [], "dyn": []}

    def terms():
        H = torch.exp(net.log_H)
        D = torch.exp(net.log_D)
        _, w_o = net(tt)
        L_data = ((w_o[:, :ff.shape[1]] / (2 * np.pi) - ff) ** 2).mean()
        r1, r2 = _swing_residual(net, tc, E, H, D, Pm, Gr, Br, ws, nm)
        L_dyn = (r1 ** 2).mean() + (r2 ** 2).mean() / (ws ** 2)
        return {"data": L_data, "dyn": lam_dyn * L_dyn}

    def loss_fn():
        # enable_grad is load-bearing: the swing residual differentiates the
        # network with respect to t, and ``train_two_stage`` evaluates the loss
        # once per L-BFGS step inside ``torch.no_grad()`` to record its history.
        # Without this the forward pass there builds no graph and ``grad``
        # raises. It costs nothing when grad is already on.
        with torch.enable_grad():
            t = terms()
        trace["data"].append(float(t["data"]))
        trace["dyn"].append(float(t["dyn"]))
        return t["data"] + t["dyn"]

    if verbose:
        print(f"  inertia identification   lambda_dyn = {lam_dyn:g}")
        print(f"  H starts at     : {np.round(H0, 3)}")
        describe(net.net, n_coll)

    history = train_two_stage(net, loss_fn, adam_steps=adam_steps,
                              lbfgs_steps=lbfgs_steps, lr=lr,
                              report_every=max(adam_steps // 4, 1) if verbose else 0)
    history = dict(history)
    history["data"] = np.asarray(trace["data"])
    history["dyn"] = np.asarray(trace["dyn"])
    if verbose:
        print(f"  final terms     : data {history['data'][-1]:.3e}"
              f"   dyn {history['dyn'][-1]:.3e}")

    return (to_numpy(torch.exp(net.log_H)).ravel(),
            to_numpy(torch.exp(net.log_D)).ravel(), net, history)


# ══════════════════════════════════════════════════════════════════════════
# 8 · lab: persistence, tables, plots, report
# ══════════════════════════════════════════════════════════════════════════
#
# Nothing in this section is physics. It exists so the notebooks stay about the
# method rather than about matplotlib, and so every student's tables have the
# same columns and can be compared with each other's.
#
# The one opinion baked in: every table that reports estimation error splits
# metered from unmetered buses, and reports the worst bus as well as the mean.
# You can override that, but you will have to do it deliberately.

#: Where the notebooks write. Sequential by design — later notebooks load what
#: earlier ones saved.
RESULTS = "Ex12.1_outputs"

# course palette, so figures match the slides
BG, TXT, MUTED = "#07080B", "#E4ECF5", "#AFC0D2"
CYAN, AMBER, ORANGE, GREEN, PURPLE = "#8FE3FA", "#FFC98A", "#FF9E7A", "#9BEDB6", "#CBB6FF"


def use_course_style():
    """The dark slide palette. Every plot function here calls it for you."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG,
        "savefig.facecolor": BG,
        "text.color": TXT, "axes.labelcolor": TXT,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": "#39404F", "grid.color": "#242A36",
        "axes.grid": True, "grid.linewidth": 0.6,
        "font.size": 10, "axes.titlesize": 11,
        "legend.frameon": False, "figure.dpi": 110,
    })


# ── persistence ───────────────────────────────────────────────────────────

def save(name, **arrays):
    """Save results for a later notebook. Notebooks are sequential; this is how
    state crosses between them."""
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, name + ".npz")
    np.savez(path, **arrays)
    print(f"  saved -> {path}  ({', '.join(arrays)})")
    return path


def reference_state(save_it=True, verbose=True):
    """Rebuild what notebook 00 saves as ``00_reference``, from scratch.

    This is exactly the computation notebook 00 performs: build the admittance
    matrix, solve the base-case power flow, and fetch the DK2 load curve (with
    its flagged synthetic fallback when offline). It exists so that notebooks
    01 to 05 can run in a fresh runtime - each Colab tab is a separate machine,
    so a file saved in one tab does not exist in another. The network and the
    power flow are deterministic; only the load curve's *source* can differ
    between runs, and it is flagged either way.
    """
    Y = build_ybus()
    P, Q = injections()
    V, th, converged, _ = solve_power_flow(Y, P, Q)
    if not converged:
        raise RuntimeError("base-case power flow did not converge - "
                           "this should never happen on the six-bus network")
    day = dk2_load(verbose=verbose)
    ref = dict(V=V, th=th, P=P, Q=Q,
               load_mw=day["load_mw"], source=np.array([day["source"]]))
    if save_it:
        save("00_reference", **ref)
    return ref


# Results that load() can rebuild on the spot when the file is absent. Only
# the reference qualifies: it is cheap and deterministic. Everything else is
# the product of a training run or an estimate the student is meant to have
# made, and silently regenerating those would defeat the comparison in 05.
_REBUILDERS = {"00_reference": reference_state}


def load(name, required=True):
    path = os.path.join(RESULTS, name + ".npz")
    if not os.path.exists(path):
        if name in _REBUILDERS:
            print(f"  [load] {path} not found - rebuilding it now.")
            print(f"  [load] (Each Colab tab is a separate runtime, so files "
                  f"saved in another notebook's tab are not here. This rebuild "
                  f"is the same computation notebook 00 performs.)")
            return _REBUILDERS[name](save_it=True)
        if required:
            raise FileNotFoundError(
                f"{path} not found. Notebooks in this exercise are sequential - "
                f"run the earlier notebook that produces '{name}' first.\n"
                f"On Colab, note that every notebook tab is a separate runtime "
                f"with its own filesystem. Either run the notebooks one after "
                f"another in the same runtime, or download the "
                f"'{RESULTS}/' folder from the earlier tab (Files panel, right-"
                f"click, Download) and upload it here before this cell.")
        return None
    return dict(np.load(path, allow_pickle=True))


# ── tables ────────────────────────────────────────────────────────────────

def error_table(V_true, th_true, V_est, th_est, measured, label="", n_bus=None):
    """The table this exercise insists on: metered vs unmetered, worst vs mean.

    Not ``course_core.error_table`` — see the note in the module docstring.
    Reach this one through ``pb.error_table``.
    """
    n = len(V_true) if n_bus is None else n_bus
    meas = sorted(set(int(i) for i in measured))
    unmeas = [i for i in range(n) if i not in meas]
    eV = np.abs(np.asarray(V_est).ravel() - np.asarray(V_true).ravel())
    eT = np.abs(np.degrees(np.asarray(th_est).ravel() - np.asarray(th_true).ravel()))

    def block(idx):
        if not idx:
            return dict(n=0, V_mean=np.nan, V_worst=np.nan, V_worst_bus=-1,
                        TH_mean=np.nan, TH_worst=np.nan)
        return dict(n=len(idx),
                    V_mean=float(eV[idx].mean()), V_worst=float(eV[idx].max()),
                    V_worst_bus=int(idx[int(np.argmax(eV[idx]))]),
                    TH_mean=float(eT[idx].mean()), TH_worst=float(eT[idx].max()))

    res = {"label": label, "metered": block(meas), "unmetered": block(unmeas)}
    print(f"\n  {label}")
    print(f"  {'':<12}{'buses':>6}{'|V| mean':>11}{'|V| worst':>11}"
          f"{'  at bus':>8}{'ang mean°':>11}{'ang worst°':>12}")
    for k in ("metered", "unmetered"):
        b = res[k]
        if b["n"] == 0:
            print(f"  {k:<12}{0:>6}{'—':>11}")
            continue
        print(f"  {k:<12}{b['n']:>6}{b['V_mean']:>11.2e}{b['V_worst']:>11.2e}"
              f"{b['V_worst_bus']:>8d}{b['TH_mean']:>11.3f}{b['TH_worst']:>12.3f}")
    return res


def comparison_table(rows):
    """rows: [(name, dict from error_table), ...]"""
    print(f"\n  {'method':<22}{'metered worst':>15}{'unmetered worst':>17}")
    print("  " + "-" * 54)
    for name, r in rows:
        m = r["metered"]["V_worst"]
        u = r["unmetered"]["V_worst"]
        us = "—" if np.isnan(u) else f"{u:.2e}"
        print(f"  {name:<22}{m:>15.2e}{us:>17}")


# ── plots ─────────────────────────────────────────────────────────────────

def plot_network_state(V, th, V_ref=None, th_ref=None, measured=(), title=""):
    import matplotlib.pyplot as plt
    use_course_style()
    V = np.asarray(V).ravel()
    th = np.asarray(th).ravel()
    n = len(V)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.1))
    x = np.arange(n)
    cols = [CYAN if i in measured else ORANGE for i in range(n)]
    for a, (val, ref, lab) in zip(ax, [(V, V_ref, "|V|  (p.u.)"),
                                       (np.degrees(th),
                                        None if th_ref is None else np.degrees(th_ref),
                                        "angle  (deg)")]):
        if ref is not None:
            a.plot(x, np.asarray(ref).ravel(), "-", color=MUTED, lw=1.4,
                   label="truth")
        a.scatter(x, val, c=cols, s=42, zorder=3, label="estimate")
        a.set_xlabel("bus"); a.set_ylabel(lab); a.set_xticks(x)
    ax[0].legend(loc="best", fontsize=8)
    fig.suptitle(title + "   (cyan = metered, orange = unmetered)", fontsize=10)
    fig.tight_layout()
    return fig


def plot_sweep(lams, err_metered, err_unmetered, chosen=None):
    """The lambda sweep: the plot that shows the trade-off, if you plot the
    unmetered buses. On metered buses it looks flat, which is the point."""
    import matplotlib.pyplot as plt
    use_course_style()
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.loglog(lams, err_metered, "o-", color=CYAN, label="metered buses")
    ax.loglog(lams, err_unmetered, "o-", color=ORANGE, label="unmetered buses")
    if chosen is not None:
        ax.axvline(chosen, color=GREEN, ls="--", lw=1.2, label=f"chosen λ = {chosen:g}")
    ax.set_xlabel("λ  (weight on the power flow residual)")
    ax.set_ylabel("worst |V| error")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_swing(T, D, W, t_fault=None, t_clear=None, f0=50.0, title=""):
    import matplotlib.pyplot as plt
    use_course_style()
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
    for k in range(D.shape[1]):
        ax[0].plot(T, np.degrees(D[:, k]), lw=1.6,
                   color=[CYAN, AMBER, PURPLE][k % 3], label=f"machine {k}")
        ax[1].plot(T, W[:, k] / (2 * np.pi), lw=1.6,
                   color=[CYAN, AMBER, PURPLE][k % 3])
    for a in ax:
        if t_fault is not None:
            a.axvspan(t_fault, t_clear, color=ORANGE, alpha=0.16, lw=0)
    ax[0].set_xlabel("t  (s)"); ax[0].set_ylabel("rotor angle  (deg)")
    ax[1].set_xlabel("t  (s)"); ax[1].set_ylabel("frequency  (Hz)")
    ax[1].axhline(f0, color=MUTED, lw=0.8, ls=":")
    ax[0].legend(fontsize=8)
    fig.suptitle(title + "   (shaded = fault on)", fontsize=10)
    fig.tight_layout()
    return fig


def plot_frequency_event(t, f, t_event=None, window=None, f0=50.0, fit=None):
    import matplotlib.pyplot as plt
    use_course_style()
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.plot(t, f, lw=1.0, color=CYAN, label="measured")
    ax.axhline(f0, color=MUTED, lw=0.8, ls=":")
    if t_event is not None:
        ax.axvline(t_event, color=ORANGE, lw=1.2, ls="--", label="event")
        if window:
            ax.axvspan(t_event, t_event + window, color=GREEN, alpha=0.14, lw=0)
        if fit is not None:
            tt = np.linspace(t_event, t_event + (window or 1.0), 20)
            ax.plot(tt, f0 + fit * (tt - t_event), lw=1.6, color=AMBER,
                    label=f"RoCoF fit = {fit:+.3f} Hz/s")
    ax.set_xlabel("t  (s)"); ax.set_ylabel("frequency  (Hz)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_loss(history, title=""):
    """Every curve in a training history, on one log axis.

    ``history`` is what the estimators here return: ``"adam"`` and ``"lbfgs"``
    from ``train_two_stage``, plus the individual loss terms. The curves have
    **different lengths on purpose** — the optimiser traces carry one point per
    optimiser step, the term traces one point per evaluation of the loss, and
    L-BFGS evaluates the loss several times per step during its line search.
    Each is drawn against its own index, so read the shape rather than the
    x-position.

    The term curves are the ones to watch. A total that has stopped falling
    tells you nothing about *which* half stopped.
    """
    import matplotlib.pyplot as plt
    use_course_style()
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    colours = [MUTED, TXT, CYAN, AMBER, PURPLE, GREEN]
    for (name, curve), colour in zip(history.items(), colours):
        y = np.asarray(curve, dtype=float).ravel()
        if y.size == 0:
            continue
        style = "--" if name in ("adam", "lbfgs") else "-"
        ax.semilogy(np.arange(y.size), np.maximum(y, 1e-300), style,
                    color=colour, lw=1.3, label=name)
    ax.set_xlabel("loss evaluation  (each curve against its own index)")
    ax.set_ylabel("loss")
    ax.legend(fontsize=8)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


# ── report ────────────────────────────────────────────────────────────────

def make_report(sections, filename="Ex12.1_report.md", author=""):
    L = ["# Ex_12.1 - Power Grid Stability Estimation", ""]
    if author:
        L.append(f"**Author:** {author}  ")
    L += ["",
          "> The network is DK2-REPRESENTATIVE: real injections and real",
          "> frequency, but line impedances are plausible values rather than",
          "> measured ones. State which of your conclusions depend on them.",
          ""]
    for title, body in sections:
        L += [f"## {title}", "", body, ""]
    L += ["## The question that carries the marks", "",
          "Which of your numbers would you show a control-room operator, and",
          "which would you not? For each number you would withhold, say what",
          "would have to be true before you would show it.", ""]
    text = "\n".join(L)
    folder = os.path.dirname(filename)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(filename, "w", encoding="utf8") as fh:
        fh.write(text)
    print(f"  wrote {filename} ({len(text)} characters)")
    return filename
