r"""Ex_12.2 — the physics: N-1 screening on the same six-bus network.

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

Ex_12.1 asked what the grid is doing **now**. This one asks where it is
**heading**: given the present dispatch, which single-line outage would leave
the system unable to survive a fault?

That question is **N-1 screening**, and a control room answers it for every
credible contingency several times an hour. The honest answer is a transient
simulation per contingency, and the honest answer is too slow. So the exercise
is to build a surrogate — and to be precise about what it is allowed to
replace.

A contingency here is the textbook one:

    a solid three-phase fault at the sending end of branch k,
    cleared at time t_c by tripping branch k.

The system survives if the machines stay in step. The longest fault it
survives is the **critical clearing time**, and it is what everything in this
set predicts:

    CCT > PROTECTION_TIME   the protection is fast enough — secure
    CCT < PROTECTION_TIME   the fault outlasts the breakers — insecure

The base case is the same fault at the local plant's terminals cleared without
losing a line — a successful autoreclose — so that the CCT reduction caused by
actually losing the branch can be read off directly.

## Why the graph is an **input**, and not a label

A line outage changes the network's topology. A dense network sees a
fixed-length vector: the only way to tell it which line is out is a one-hot
flag, and the flag is a name, not a structure. A graph network is handed the
modified adjacency itself, so a line out is a **different input**, not a
different case identifier.

That is the whole reason the graph network from Ex_05 notebook 03 and L5.1 is
the right architecture here, and notebook 03 measures the consequence: permute
the bus numbering and the graph model's prediction does not move, while the
dense model's does.

## Network provenance — read this before quoting a number

The six-bus case is **DK2-representative, not DK2**, exactly as in Ex_12.1: the
structure follows eastern Denmark in the ways that matter — a few load centres,
two generation buses, two HVDC links entering as fixed injections — but the
line impedances are plausible textbook values, **not** measured ones. No TSO
publishes a nodal model with impedances.

Any result that depends on the impedances must say so. Results that depend only
on structure are on firmer ground.

:data:`BUSES` and :data:`BRANCHES` are **character-identical** to Ex_12.1's, and
:func:`check_continuity` prints them so you can see it rather than trust it.
The same object under three different questions — a supervised regression in
Ex_05, a state estimator in Ex_12.1, a stability screen here — is the point of
carrying it through the course.

## What is new here, and what was inherited

Inherited verbatim from ``Ex12.1/problem.py``, values unchanged:
:data:`BUSES`, :data:`BRANCHES`, :data:`BASE_MVA`, :data:`FAULT_FACTOR`,
:func:`build_ybus`, :func:`injections`, :func:`pq_from_state`,
:func:`solve_power_flow`, :class:`Machines`, :func:`equilibrium`,
:func:`reduced_admittance`, :func:`electrical_power`, :func:`swing_rhs`,
:func:`simulate_swing`, :func:`critical_clearing_time`.

New in this set: the transient reactances that make a **bus fault**
representable at all, the contingency enumeration, the operating-point
sampler, the labelling pipeline, the dataset, and the graph machinery.

## One refinement of the machine model, stated plainly

Ex_12.1 represented a fault by weakening the whole reduced network by
:data:`FAULT_FACTOR`. That is enough to make a swing curve, and it cannot
express *where* the fault is. Screening needs a fault location, so this set
adds the missing piece of the classical model: a **transient reactance**
between each machine's internal EMF and its terminal bus
(:data:`X_D_PRIME`). With that in place a fault is what it physically is — a
large shunt admittance at a bus (:data:`FAULT_ADMITTANCE`) — and the Kron
reduction takes care of the rest.

:data:`FAULT_FACTOR` is still here, unchanged, because
:func:`simulate_swing` falls back to it when no fault network is supplied.
Nothing in this set uses that fallback; every call passes ``Yred_fault``.

The internal EMFs are likewise computed from the power flow,
:math:`E = V + j x'_d (S/V)^*`, rather than fixed at Ex_12.1's ``(1.05, 1.03)``.
That matters here and did not there: the operating point is the input, so the
machine's internal state has to follow it.

## Conventions

    per unit throughout, 100 MVA base
    bus 0 is the slack bus: theta = 0, V = 1.0
    angles in radians internally, degrees only for display
    injections are generation minus load, so a load bus has negative P
    time in seconds; clearing times in seconds, quoted in ms where useful

## A name that is NOT shadowed

Ex_12.1 shadowed ``course_core.error_table`` and warned about it. This file
does not: :func:`screening_table` has its own name, and ``error_table`` in a
notebook here is always the shared one.
"""

from __future__ import annotations

import os
import time

import numpy as np
import torch
import torch.nn as nn

from pinn_core import MLP, parameter_count, to_numpy, to_tensor

__all__ = [
    # ── inherited from Ex_12.1, verbatim ──
    "BASE_MVA", "FAULT_FACTOR", "BUSES", "BRANCHES", "N_BUS",
    "build_ybus", "injections", "pq_from_state", "solve_power_flow",
    "Machines", "equilibrium", "reduced_admittance", "electrical_power",
    "swing_rhs", "simulate_swing", "critical_clearing_time",
    "check_continuity",
    # ── the classical model, completed ──
    "X_D_PRIME", "FAULT_ADMITTANCE",
    "augment_with_transient_reactance", "internal_emf", "machine_reduced",
    # ── the contingency set ──
    "is_connected", "islanding_outages", "contingencies", "N_CONTINGENCY",
    # ── operating points and labels ──
    "LOAD_BUSES", "LOAD_SCALE", "LOCAL_SHARE", "HVDC_RANGE",
    "T_FAULT", "T_END", "CCT_MAX", "CCT_TOL", "PROTECTION_TIME",
    "operating_points", "case_setup", "label_case", "secure",
    # ── the dataset ──
    "FEATURE_NAMES", "N_FEATURE", "adjacency", "node_features",
    "outage_onehot", "build_dataset", "dense_inputs",
    # ── graph machinery ──
    "normalised_adjacency", "permutation_matrix", "GraphLayer", "ScreeningGNN",
    # ── lab: persistence, tables, plots ──
    "RESULTS", "use_course_style",
    "BG", "TXT", "MUTED", "CYAN", "AMBER", "ORANGE", "GREEN", "PURPLE",
    "BUS_XY", "save", "load", "screening_table", "confusion",
    "plot_network", "plot_swing", "plot_screening", "plot_parity",
]


# ══════════════════════════════════════════════════════════════════════════
# 1 · the network                                    [verbatim from Ex_12.1]
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


def check_continuity(verbose=True):
    """Print the buses and the lines, so you can see this is the same network.

    **Why this function exists.** The six-bus case appears four times in this
    course: as a supervised node-regression problem in Ex_05 notebook 03, as
    the network in L5.1, as the object of a state estimator in Ex_12.1, and as
    the thing being screened here. A student who has to take on trust that they
    are the same network learns nothing from the repetition. A student who can
    print the bus names and the line impedances in two folders and diff them
    learns that a model is a *thing*, carried between problems, not a fresh
    invention each time.

    It also fails loudly if someone edits an impedance. Every number in
    notebooks 00–05 depends on :data:`BRANCHES`; a silent change to one entry
    would move every result in the set and leave the prose describing a network
    that no longer exists.

    Returns the ``(BUSES, BRANCHES)`` pair so a caller can diff them.
    """
    if verbose:
        print("  buses")
        for i, (name, kind, p, q) in enumerate(BUSES):
            print(f"    {i}  {name:<18s} {kind:<6s}"
                  f"  P {p:+.2f}   Q {q:+.2f}   p.u.")
        print("  lines")
        for k, (f, t, r, x, b) in enumerate(BRANCHES):
            print(f"    {k}  {f} - {t}   r {r:.4f}   x {x:.4f}   b {b:.3f}"
                  f"   |z| {abs(complex(r, x)):.4f} p.u.")
        print(f"  {N_BUS} buses, {len(BRANCHES)} lines, {BASE_MVA:.0f} MVA base")
        print("  identical to Ex_12.1/problem.py — diff the two files if you "
              "want to check")
    return BUSES, BRANCHES


# ══════════════════════════════════════════════════════════════════════════
# 2 · power flow                                     [verbatim from Ex_12.1]
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
# 3 · dynamics                                       [verbatim from Ex_12.1]
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


# ══════════════════════════════════════════════════════════════════════════
# 4 · the classical model, completed: where the fault is
# ══════════════════════════════════════════════════════════════════════════
#
# Ex_12.1 put the machines' internal EMFs directly on buses 0 and 1 and
# represented the fault by multiplying the reduced network by FAULT_FACTOR.
# For a single swing curve that is enough. For screening it is not: a fault has
# a *location*, and a model with no reactance between the EMF and the bus
# cannot express one — a solid fault at a machine bus would be a short across
# the EMF itself.
#
# So this set inserts the piece the classical model is normally drawn with: a
# transient reactance x'_d from each internal node to its terminal bus. The
# augmented network has 8 nodes (6 buses + 2 internal), every one of the 6
# buses is now eliminable, and a fault anywhere is a large shunt on the
# diagonal. reduced_admittance is then called *unchanged* on the augmented
# matrix, with the two internal nodes as the retained set.

#: Transient reactance from each machine's internal EMF to its terminal bus,
#: p.u. on the 100 MVA system base.
#:
#: Machine 0 is not a machine: it is the Nordic system seen through the Swedish
#: AC link, and 0.02 p.u. is a stiff Thevenin equivalent for it — consistent
#: with H = 150 s, which is the same statement in the time domain.
#:
#: Machine 1 is the local plant. 0.30 p.u. on 100 MVA is about 0.25 p.u. on the
#: machine base of a unit large enough to carry the 115 MW of the base
#: dispatch, which is the textbook range for a salient-pole or cylindrical
#: rotor unit including its step-up transformer. ESTIMATED, like H and D — a
#: plausible number for plant of this size, not a measurement of any real unit.
X_D_PRIME = (0.02, 0.30)

#: Shunt admittance representing a solid three-phase fault, p.u. Equivalent to
#: a fault impedance of 1e-4 p.u., which on a 100 MVA, 400 kV base is about
#: 0.16 milliohm — a bolted fault in everything but name. Large enough that the
#: answer no longer depends on it (double it and the CCT does not move),
#: small enough that the 4x4 elimination stays well conditioned.
FAULT_ADMITTANCE = 1e4


class _InternalNodes:
    """The retained set for :func:`reduced_admittance` after augmentation.

    ``reduced_admittance`` only ever reads ``machines.buses``, so this three-
    line stand-in is all that is needed to point it at the internal nodes
    instead of the terminal buses. Using it means the reduction itself stays
    byte-identical to Ex_12.1's.
    """

    def __init__(self, buses):
        self.buses = list(buses)

    def __len__(self):
        return len(self.buses)


def augment_with_transient_reactance(Y, machines, xdp=X_D_PRIME):
    """Append one internal node per machine, tied to its bus through j·x'_d.

    Returns an ``(n + nm, n + nm)`` admittance matrix whose last ``nm`` rows
    are the machine internal nodes, in the order of ``machines.buses``.
    """
    n = Y.shape[0]
    nm = len(machines)
    Ya = np.zeros((n + nm, n + nm), dtype=complex)
    Ya[:n, :n] = Y
    for k, b in enumerate(machines.buses):
        y = 1.0 / complex(0.0, float(xdp[k]))
        i = n + k
        Ya[i, i] += y
        Ya[b, b] += y
        Ya[i, b] -= y
        Ya[b, i] -= y
    return Ya


def internal_emf(V, th, P, Q, machines, xdp=X_D_PRIME):
    """The complex EMF behind transient reactance, E = V + j x'_d (S/V)*.

    Computed from the **solved power flow**, so the machine's internal state
    follows the operating point. Ex_12.1 held ``E`` fixed at ``(1.05, 1.03)``,
    which was harmless there because the operating point never changed; here
    the operating point is the input and holding E fixed would throw away most
    of the signal the surrogate is supposed to learn.

    Pass the **computed** injections from :func:`pq_from_state`, not the
    scheduled ones — at the slack bus the scheduled injection is zero and the
    computed one is whatever the rest of the network needs.
    """
    E = []
    for k, b in enumerate(machines.buses):
        Vc = V[b] * np.exp(1j * th[b])
        S = complex(P[b], Q[b])
        E.append(Vc + 1j * float(xdp[k]) * np.conj(S / Vc))
    return np.array(E)


def machine_reduced(Y, machines, V, th, P, Q, fault_bus=None, xdp=X_D_PRIME):
    """Reduce onto the machine internal nodes, optionally with a bus fault.

    Three steps, none of them new physics:

    1. if ``fault_bus`` is given, add :data:`FAULT_ADMITTANCE` to that
       diagonal entry — that is what a solid three-phase fault *is*;
    2. augment with the transient reactances;
    3. call :func:`reduced_admittance` unchanged, retaining the internal nodes.

    The machine buses' own injections are zeroed before the reduction. Step 3
    turns every injection into a constant shunt impedance, which is right for a
    load and wrong for a generator: the generator is already represented by its
    EMF and x'_d, and converting its injection as well would count it twice.
    """
    n = Y.shape[0]
    nm = len(machines)
    Yf = np.array(Y, dtype=complex, copy=True)
    if fault_bus is not None:
        Yf[fault_bus, fault_bus] += FAULT_ADMITTANCE
    Ya = augment_with_transient_reactance(Yf, machines, xdp)

    Pe = np.concatenate([np.asarray(P, float), np.zeros(nm)])
    Qe = np.concatenate([np.asarray(Q, float), np.zeros(nm)])
    for b in machines.buses:
        Pe[b] = 0.0
        Qe[b] = 0.0
    Ve = np.concatenate([np.asarray(V, float), np.ones(nm)])
    the = np.concatenate([np.asarray(th, float), np.zeros(nm)])

    return reduced_admittance(Ya, _InternalNodes(range(n, n + nm)),
                              Ve, the, Pe, Qe)


# ══════════════════════════════════════════════════════════════════════════
# 5 · the contingency set
# ══════════════════════════════════════════════════════════════════════════

def is_connected(Y, tol=1e-9):
    """Is every bus reachable from bus 0 through non-zero off-diagonal terms?

    The admittance matrix **is** the graph: ``Y[i, j]`` is non-zero exactly
    when a branch joins i and j. So connectivity needs no separate edge list
    and cannot drift out of step with one, which is the reason the test is
    written this way rather than against :data:`BRANCHES`.
    """
    Y = np.asarray(Y)
    n = Y.shape[0]
    A = np.abs(Y - np.diag(np.diag(Y))) > tol
    seen = {0}
    stack = [0]
    while stack:
        i = stack.pop()
        for j in range(n):
            if A[i, j] and j not in seen:
                seen.add(j)
                stack.append(j)
    return len(seen) == n


def islanding_outages():
    """Branch indices whose removal splits the network into islands.

    Removing one of these leaves a bus with no path to the slack. There is no
    power flow solution — the Jacobian is singular — and there is no meaningful
    critical clearing time either, because the islanded bus has already lost
    supply before any machine has had time to swing.

    A real screening tool reports these separately as **loss-of-supply**
    contingencies. They are a different kind of problem with a different
    remedy, and folding them into a stability score would be a category error.
    """
    return [k for k in range(len(BRANCHES))
            if not is_connected(build_ybus(outage=k))]


def contingencies(verbose=False):
    """The N-1 list: the base case, then every non-islanding single outage.

    Each entry is a dict::

        {"index": position in this list,
         "outage": branch index removed, or None for the base case,
         "fault_bus": bus at which the three-phase fault is applied,
         "label": something printable}

    **What a contingency is here.** A solid three-phase fault at the sending
    end of branch k, cleared by tripping branch k. That is the textbook
    definition and it is why the fault location moves with the contingency: you
    do not clear a fault on one line by tripping a different one.

    The sending end is the first bus in the :data:`BRANCHES` entry. That list
    is ordered outward from the slack bus, so the sending end is the end nearer
    generation — the more severe of the two, and the one a planning study
    would pick.

    **The base case** is the same fault at the local plant's terminals (bus 1)
    cleared *without* losing a line: a successful autoreclose. It is the
    reference against which the cost of actually losing a branch is read.
    """
    skipped = islanding_outages()
    out = [{"index": 0, "outage": None, "fault_bus": 1,
            "label": "base — fault at Gen east, no line lost"}]
    for k, (f, t, *_rest) in enumerate(BRANCHES):
        if k in skipped:
            continue
        out.append({"index": len(out), "outage": k, "fault_bus": f,
                    "label": f"trip line {k}  ({BUSES[f][0]} – {BUSES[t][0]})"})
    if verbose:
        print(f"  {len(BRANCHES)} branches, {len(skipped)} skipped as islanding"
              f"  -> {len(out)} contingencies (base + {len(out)-1} outages)")
        for k in skipped:
            f, t = BRANCHES[k][0], BRANCHES[k][1]
            print(f"    skipped branch {k} ({BUSES[f][0]} – {BUSES[t][0]}): "
                  f"removing it islands bus {t} — a loss-of-supply case, not a "
                  f"stability case")
    return out


#: How many cases :func:`contingencies` returns. Computed once at import.
N_CONTINGENCY = len(contingencies())


# ══════════════════════════════════════════════════════════════════════════
# 6 · operating points
# ══════════════════════════════════════════════════════════════════════════

#: The three buses that carry load and are scaled together.
LOAD_BUSES = (2, 3, 4)

#: System load as a multiple of the base case. 0.85 is a summer night on
#: Zealand; 1.60 is a cold winter evening peak. Wider than a day's excursion on
#: purpose — a screening surrogate that has only seen the middle of the range
#: is exactly the surrogate that fails on the evening it is needed.
LOAD_SCALE = (0.85, 1.60)

#: Fraction of the load increment above base that Gen east picks up; the slack
#: bus (the Swedish AC link) covers the rest. This is the import-versus-local
#: generation axis, and it is the second thing a DK2 dispatcher actually turns.
LOCAL_SHARE = (0.55, 1.00)

#: Import on the German HVDC link at bus 5 [p.u.]. Base case is 0.36.
HVDC_RANGE = (0.15, 0.55)


def operating_points(n, seed=0, max_tries=200):
    """``n`` plausible pre-fault dispatches, as an ``(n, 2, N_BUS)`` array.

    ``op[0]`` is P at every bus and ``op[1]`` is Q, both p.u. on
    :data:`BASE_MVA`, in the sign convention of :data:`BUSES` — generation
    positive, load negative.

    **The ranges, and why.** Three independent knobs, drawn uniformly:

    ==================  ==================  =====================================
    ``LOAD_SCALE``      0.85 – 1.60         system load, as a multiple of base
    ``LOCAL_SHARE``     0.55 – 1.00         share of the load increment carried
                                            by Gen east rather than imported
    ``HVDC_RANGE``      0.15 – 0.55 p.u.    import on the German link at bus 5
    ==================  ==================  =====================================

    plus an independent ±6 % jitter on each load bus, so the three load centres
    do not move in lockstep — they never do, and a surrogate trained on
    perfectly correlated loads will discover a one-dimensional problem that is
    not there.

    Reactive power is scaled with active power at every bus, i.e. **constant
    power factor**. That is the usual planning assumption and it is a
    simplification: real load power factor moves with composition, and voltage
    stability lives in exactly that difference. Nothing in this set touches
    voltage stability, so the assumption is safe here and would not be in a
    Q-V study.

    A draw is rejected and redrawn if the power flow fails to converge or the
    lowest bus voltage falls below 0.85 p.u. — a case a control room would
    already be acting on, and not one to train a stability screen against.
    """
    rng = np.random.default_rng(seed)
    P0, Q0 = injections()
    base_load = float(-sum(P0[i] for i in LOAD_BUSES))
    ops = []
    tries = 0
    while len(ops) < n:
        tries += 1
        if tries > max_tries * max(n, 1):
            raise RuntimeError(
                f"only {len(ops)} of {n} operating points converged in "
                f"{tries} draws — the ranges above are too wide for this "
                f"network")
        s = rng.uniform(*LOAD_SCALE)
        share = rng.uniform(*LOCAL_SHARE)
        hvdc = rng.uniform(*HVDC_RANGE)
        P, Q = P0.copy(), Q0.copy()
        for i in LOAD_BUSES:
            j = 1.0 + rng.uniform(-0.06, 0.06)
            P[i] = P0[i] * s * j
            Q[i] = Q0[i] * s * j
        total = float(-sum(P[i] for i in LOAD_BUSES))
        P[1] = P0[1] + share * (total - base_load)
        Q[1] = Q0[1] * P[1] / P0[1]
        P[5] = hvdc
        Q[5] = Q0[5] * hvdc / P0[5]
        V, th, converged, _ = solve_power_flow(build_ybus(), P, Q)
        if not converged or V.min() < 0.85:
            continue
        ops.append(np.stack([P, Q]))
    return np.array(ops)


# ══════════════════════════════════════════════════════════════════════════
# 7 · labelling — the expensive truth
# ══════════════════════════════════════════════════════════════════════════

#: When the fault is applied [s]. Only long enough to show a flat pre-fault
#: trace on the plot; the pre-fault state is an exact equilibrium, so nothing
#: happens before this.
T_FAULT = 0.10

#: How long each trial is integrated [s]. The marginal case reaches its
#: greatest angle separation about 0.7 s after the fault, so 1.0 s of window
#: covers the first swing with room to spare. Longer is safer and linearly more
#: expensive; this is the shortest window that reproduces the CCT obtained with
#: a 4 s window to within the bisection tolerance.
T_END = 1.0

#: Upper end of the bisection [s]. A contingency that survives half a second of
#: solid three-phase fault is not a stability constraint — the protection would
#: have cleared it four times over — so the search stops there and reports
#: ``CCT_MAX``. Roughly one case in six is censored this way, and the notebooks
#: say so wherever it changes a number.
CCT_MAX = 0.5

#: Bisection tolerance [s]. 2 ms is 0.1 of a cycle at 50 Hz, and about 1.4 % of
#: :data:`PROTECTION_TIME` — finer than the label is meaningful, given that the
#: classical machine model is itself worth rather less than that.
CCT_TOL = 0.002

#: The clearing time the protection can actually deliver [s]. A contingency
#: with ``CCT < PROTECTION_TIME`` is **insecure**: the fault outlasts the
#: breakers and the plant slips a pole.
#:
#: 0.14 s is seven cycles at 50 Hz, and it is a budget rather than a guess:
#:
#:     1.5 cycles   30 ms   distance relay decision, zone 1
#:     2.5 cycles   50 ms   circuit-breaker interrupting time
#:     3.0 cycles   60 ms   margin — pole scatter, DC offset, and the slower
#:                          of the two ends in a permissive scheme
#:     ---------------------
#:     7.0 cycles  140 ms
#:
#: A modern transmission scheme clears faster than this; an older one with an
#: oil breaker does not. Choosing the pessimistic end is deliberate: the point
#: of the exercise is the asymmetry between a missed insecure case and a false
#: alarm, and a threshold that never bites teaches neither.
PROTECTION_TIME = 0.14


def case_setup(op, outage, fault_bus):
    """Everything a swing simulation of one case needs, as a dict.

    Split out from :func:`label_case` so that notebook 00 can plot the swing
    curves without re-deriving the networks, and so that the three admittance
    matrices — pre-fault, fault-on, post-fault — can be looked at.

    Returns ``None`` if the pre-fault power flow does not converge.
    """
    P = np.asarray(op, dtype=float)[0]
    Q = np.asarray(op, dtype=float)[1]
    machines = Machines()

    Y_pre = build_ybus()
    V, th, converged, iters = solve_power_flow(Y_pre, P, Q)
    if not converged:
        return None
    Pc, Qc = pq_from_state(V, th, Y_pre)

    machines.E = np.abs(internal_emf(V, th, Pc, Qc, machines))

    Yred_pre = machine_reduced(Y_pre, machines, V, th, Pc, Qc)
    Yred_fault = machine_reduced(Y_pre, machines, V, th, Pc, Qc,
                                 fault_bus=fault_bus)
    if outage is None:
        Yred_post = Yred_pre
    else:
        Yred_post = machine_reduced(build_ybus(outage=outage), machines,
                                    V, th, Pc, Qc)

    # The local plant's mechanical power is its scheduled dispatch. x'_d is
    # purely reactive, so the power at the internal node equals the power at
    # the terminal bus exactly; there is nothing to correct for.
    try:
        delta0, Pm = equilibrium(machines, Yred_pre, [-1.0, P[1]])
    except RuntimeError:
        return None

    return {"V": V, "th": th, "P": P, "Q": Q, "P_calc": Pc, "Q_calc": Qc,
            "machines": machines, "delta0": delta0, "Pm": Pm,
            "Yred_pre": Yred_pre, "Yred_fault": Yred_fault,
            "Yred_post": Yred_post, "outage": outage, "fault_bus": fault_bus,
            "iterations": iters}


def label_case(op, outage, fault_bus, return_detail=False):
    """The critical clearing time of one (operating point, contingency) pair.

    This is the **ground truth** of the whole exercise, and it is expensive:
    a power flow, three Kron reductions, an equilibrium solve, and then a
    bisection in which every trial is a full RK4 integration of the swing
    equations. Ten or eleven simulations per label, about a third of a second
    each on a laptop CPU.

    Multiply that by every contingency, every hour, in a control room with
    thousands of branches rather than six, and you have the reason surrogates
    for this exist at all. Notebook 00 times it so the number is yours rather
    than mine.

    Returns the CCT in seconds. ``0.0`` means the system does not survive even
    an instantaneous fault — the post-fault network cannot hold the pre-fault
    dispatch at all — and :data:`CCT_MAX` means the search reached its ceiling.
    With ``return_detail=True`` returns ``(cct, detail)`` where ``detail`` is
    the dict from :func:`case_setup`.
    """
    detail = case_setup(op, outage, fault_bus)
    if detail is None:
        return (np.nan, None) if return_detail else np.nan
    cct = critical_clearing_time(
        detail["machines"], detail["Yred_pre"], detail["Yred_post"],
        detail["delta0"], Yred_fault=detail["Yred_fault"],
        t_fault=T_FAULT, t_end=T_END, hi=CCT_MAX, tol=CCT_TOL)
    return (cct, detail) if return_detail else cct


def secure(cct, threshold=PROTECTION_TIME):
    """True where the protection is fast enough. The label notebooks 02-04 score.

    Strictly greater: a CCT exactly equal to the clearing time is a fault the
    breakers only just clear, with no margin at all, and a screening tool that
    called that secure would be lying by a rounding error.
    """
    return np.asarray(cct, dtype=float) > float(threshold)


# ══════════════════════════════════════════════════════════════════════════
# 8 · the dataset
# ══════════════════════════════════════════════════════════════════════════

#: Per-bus input channels. Six numbers per bus, six buses.
FEATURE_NAMES = ("P", "Q", "|V|", "theta", "is_machine", "is_fault_bus")

N_FEATURE = len(FEATURE_NAMES)


def adjacency(outage=None):
    """The 6x6 binary adjacency of the **post-fault** network.

    Symmetric, zero on the diagonal, built from :func:`build_ybus` so it cannot
    disagree with the electrical model. This is the object the graph network
    is handed and the dense network cannot be.
    """
    Y = build_ybus(outage=outage)
    A = (np.abs(Y - np.diag(np.diag(Y))) > 1e-9).astype(float)
    return A


def node_features(op, contingency, detail=None):
    """``(N_BUS, N_FEATURE)`` node features for one case.

    Channels, in order:

    ``P``, ``Q``
        the scheduled injection at the bus, p.u. — the dispatch, which is what
        an operator knows before anything happens;
    ``|V|``, ``theta``
        the solved **pre-fault** state, from the power flow. This is the output
        of Ex_12.1: the estimated state is the input to the prediction, and
        that is the seam between the two exercise sets;
    ``is_machine``
        1 at buses 0 and 1. Structural, not operational;
    ``is_fault_bus``
        1 at the bus where the fault is applied. Part of the *contingency*, not
        of the operating point.

    What is **not** here is any encoding of which line is out. That is carried
    entirely by the adjacency, which is the argument of the whole exercise.
    """
    if detail is None:
        detail = case_setup(op, contingency["outage"], contingency["fault_bus"])
    if detail is None:
        return None
    X = np.zeros((N_BUS, N_FEATURE))
    X[:, 0] = detail["P"]
    X[:, 1] = detail["Q"]
    X[:, 2] = detail["V"]
    X[:, 3] = detail["th"]
    for b in Machines().buses:
        X[b, 4] = 1.0
    X[contingency["fault_bus"], 5] = 1.0
    return X


def outage_onehot(contingency):
    """A one-hot flag over the branches — all zeros for the base case.

    The **only** thing a dense network can be told about the topology. Notebook
    02 uses it and notebook 03 explains why it is not enough: the flag names
    the outage, it does not describe it. Two networks that differ by one line
    are, to a dense model, two arbitrary and unrelated bit patterns.
    """
    v = np.zeros(len(BRANCHES))
    if contingency["outage"] is not None:
        v[contingency["outage"]] = 1.0
    return v


def build_dataset(n_ops=180, seed=12, cache=True, verbose=True):
    """The labelled screening set. Cached to ``.npz`` — this is the slow step.

    Every operating point is crossed with every contingency, so the set has
    ``n_ops * N_CONTINGENCY`` rows. Returns a dict of arrays:

    ==============  ==========================  ==================================
    ``X``           ``(n, N_BUS, N_FEATURE)``   node features
    ``A``           ``(n, N_BUS, N_BUS)``       post-fault adjacency, per case
    ``onehot``      ``(n, len(BRANCHES))``      outaged branch, for the dense model
    ``cct``         ``(n,)``                    critical clearing time [s]
    ``secure``      ``(n,)``                    ``cct > PROTECTION_TIME``
    ``op_id``       ``(n,)``                    which operating point
    ``cont_id``     ``(n,)``                    which contingency
    ``ops``         ``(n_ops, 2, N_BUS)``       the dispatches themselves
    ==============  ==========================  ==================================

    **Split on ``op_id``, never at random.** The ``N_CONTINGENCY`` rows sharing
    an operating point share a power flow and most of their features; put some
    of them in training and the rest in test and you are measuring memorisation.
    Notebook 02 does the split and says so again there.

    The cache key is ``(n_ops, seed)``, so changing either writes a new file
    and nothing silently goes stale.
    """
    path = os.path.join(RESULTS, f"dataset_n{n_ops}_s{seed}.npz")
    if cache and os.path.exists(path):
        d = dict(np.load(path, allow_pickle=False))
        if verbose:
            print(f"  loaded cached dataset <- {path}"
                  f"   ({d['cct'].size} cases)")
        return d

    cases = contingencies()
    ops = operating_points(n_ops, seed=seed)
    n = n_ops * len(cases)
    X = np.zeros((n, N_BUS, N_FEATURE))
    A = np.zeros((n, N_BUS, N_BUS))
    OH = np.zeros((n, len(BRANCHES)))
    y = np.zeros(n)
    op_id = np.zeros(n, dtype=int)
    cont_id = np.zeros(n, dtype=int)

    t0 = time.time()
    row = 0
    for i, op in enumerate(ops):
        for c in cases:
            cct, detail = label_case(op, c["outage"], c["fault_bus"],
                                     return_detail=True)
            X[row] = node_features(op, c, detail=detail)
            A[row] = adjacency(c["outage"])
            OH[row] = outage_onehot(c)
            y[row] = cct
            op_id[row] = i
            cont_id[row] = c["index"]
            row += 1
        if verbose and (i + 1) % 30 == 0:
            done = (i + 1) * len(cases)
            rate = (time.time() - t0) / done
            print(f"    {done:5d} / {n} labels"
                  f"   {rate*1e3:.0f} ms each"
                  f"   {(n-done)*rate:5.0f} s remaining")
    elapsed = time.time() - t0

    d = {"X": X, "A": A, "onehot": OH, "cct": y,
         "secure": secure(y).astype(float),
         "op_id": op_id, "cont_id": cont_id, "ops": ops,
         "seconds": np.array(elapsed)}
    if cache:
        os.makedirs(RESULTS, exist_ok=True)
        np.savez(path, **d)
        if verbose:
            print(f"  saved -> {path}")
    if verbose:
        print(f"  {n} labels in {elapsed:.1f} s "
              f"({elapsed/n*1e3:.0f} ms each)")
        print(f"  CCT  min {y.min():.3f}   median {np.median(y):.3f}   "
              f"max {y.max():.3f}  s")
        print(f"  insecure at {PROTECTION_TIME*1e3:.0f} ms: "
              f"{int((y <= PROTECTION_TIME).sum())} of {n} "
              f"({100*(y <= PROTECTION_TIME).mean():.1f} %)")
        print(f"  censored at CCT_MAX = {CCT_MAX:.2f} s: "
              f"{int((y >= CCT_MAX).sum())} ({100*(y >= CCT_MAX).mean():.1f} %)")
    return d


def dense_inputs(X, onehot):
    """Flatten node features and staple the one-hot on: ``(n, 6*6 + 6)``.

    This is the dense baseline's entire view of the world, and writing it in
    one line makes the loss visible. Bus order is now a *convention encoded in
    the weights*: column 13 means "reactive power at bus 2" only because
    somebody listed the buses in that order.
    """
    X = np.asarray(X, dtype=float)
    onehot = np.asarray(onehot, dtype=float)
    return np.concatenate([X.reshape(X.shape[0], -1), onehot], axis=1)


# ══════════════════════════════════════════════════════════════════════════
# 9 · graph machinery
# ══════════════════════════════════════════════════════════════════════════

def normalised_adjacency(A):
    r"""Kipf and Welling's :math:`\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}`.

    Self-loops first (:math:`\tilde{A} = A + I`) so a bus keeps its own
    features, then rows and columns scaled by the square root of the degree so
    a well-connected bus does not simply shout louder than a poorly connected
    one.

    Identical in form to ``Ex_5_core.normalised_adjacency`` — the same
    normalisation you built by hand in Ex_05 notebook 02 — except that this one
    also accepts a **stack** of adjacencies, ``(n, N, N)``, because here the
    graph changes from case to case. That difference is the whole exercise in
    one function signature.
    """
    A = np.asarray(A, dtype=float)
    single = (A.ndim == 2)
    if single:
        A = A[None]
    At = A + np.eye(A.shape[-1])[None]
    d = At.sum(axis=-1)
    dis = 1.0 / np.sqrt(d)
    out = At * dis[:, :, None] * dis[:, None, :]
    return out[0] if single else out


def permutation_matrix(perm):
    """The matrix ``P`` with ``P[i, perm[i]] = 1``.

    Left-multiplying node features by ``P`` relabels the buses: row ``i`` of
    ``P X`` is row ``perm[i]`` of ``X``. Same definition as Ex_05's, so the
    permutation test in notebook 03 is the same test you ran in Part 1.
    """
    perm = np.asarray(perm, dtype=int)
    n = len(perm)
    P = np.zeros((n, n))
    P[np.arange(n), perm] = 1.0
    return P


class GraphLayer(nn.Module):
    r"""One message-passing step, with the self term kept separate.

    .. math::

        H^{(l+1)} = \tanh\!\left(H^{(l)} W_{\text{self}} + b
                    + \hat{A}\, H^{(l)} W_{\text{neigh}}\right)

    Two weight matrices, not one: what a bus does and what its neighbours do
    are physically different things, and pooling them before transforming
    forces the layer to express that difference through the normalisation
    constants alone. Ex_05 notebook 03 measures what that costs.

    ``A_hat`` may be ``(N, N)`` — one graph for every case — or ``(n, N, N)``,
    one graph per case. **In this exercise it is always the second**, because
    the contingency *is* the graph. ``H`` is ``(n, N, features)`` and
    ``A_hat @ H`` broadcasts correctly in both shapes.

    The activation is applied by the caller, not here, so that a residual
    connection can be wrapped around it.
    """

    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.self_lin = nn.Linear(n_in, n_out)
        self.neigh_lin = nn.Linear(n_in, n_out, bias=False)

    def forward(self, A_hat: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        return self.self_lin(H) + A_hat @ self.neigh_lin(H)


class ScreeningGNN(nn.Module):
    """Message passing, mean pooling, then an :class:`MLP` head — CCT out.

    The reference implementation. Notebook 03 asks you to write your own; this
    one is here so the shape check has something to agree with and so the
    parameter count in the notebook is not a number you have to take on trust.

    **Depth is a physical quantity.** One layer moves information one hop. The
    six-bus network has a diameter of three, so a machine at bus 1 cannot feel
    the loss of the line from bus 3 to bus 5 in fewer than three layers. That
    is the same argument Ex_05 notebook 03 made with a depth sweep, and it is
    why ``depth=3`` is the default rather than a number that was tuned.

    **Mean pooling, because the target is one number for the whole graph.**
    The CCT is a property of the system, not of a bus. Pooling by mean rather
    than by sum keeps the output on the same scale when a bus is added, which
    matters if you ever run this on a larger network — and running it on a
    larger network without retraining is exactly the claim the architecture
    makes.
    """

    def __init__(self, n_in: int = N_FEATURE, hidden: int = 32, depth: int = 3,
                 head_hidden: int = 32, head_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList(
            [GraphLayer(n_in if k == 0 else hidden, hidden)
             for k in range(depth)])
        self.head = MLP(n_in=hidden, n_out=1, n_hidden=head_hidden,
                        n_layers=head_layers)
        self.depth = depth

    def forward(self, A_hat: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        for k, layer in enumerate(self.layers):
            Z = torch.tanh(layer(A_hat, H))
            H = H + Z if k > 0 else Z          # residual after the first
        return self.head(H.mean(dim=-2))


# ══════════════════════════════════════════════════════════════════════════
# 10 · lab: persistence, tables, plots
# ══════════════════════════════════════════════════════════════════════════
#
# Nothing in this section is physics. It exists so the notebooks stay about the
# method rather than about matplotlib, and so every student's tables have the
# same columns and can be compared with each other's.
#
# The one opinion baked in: every table that scores a screen reports the two
# error kinds separately. A mean absolute error over both of them together
# hides the only distinction that matters here.

#: Where the notebooks write. Sequential by design — later notebooks load what
#: earlier ones saved.
RESULTS = "Ex12.2_outputs"

# course palette, so figures match the slides
BG, TXT, MUTED = "#07080B", "#E4ECF5", "#AFC0D2"
CYAN, AMBER, ORANGE, GREEN, PURPLE = "#8FE3FA", "#FFC98A", "#FF9E7A", "#9BEDB6", "#CBB6FF"

#: Fixed drawing coordinates, so every student's picture of the network looks
#: the same and buses can be discussed by position rather than by index.
BUS_XY = np.array([[0.0, 1.00],     # 0 slack / SE link
                   [1.1, 1.85],     # 1 gen east
                   [2.4, 1.85],     # 2 load north
                   [3.4, 0.95],     # 3 load city
                   [1.6, 0.05],     # 4 load south
                   [4.5, 0.30]])    # 5 HVDC / DE link


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


def load(name, required=True):
    path = os.path.join(RESULTS, name + ".npz")
    if not os.path.exists(path):
        if required:
            raise FileNotFoundError(
                f"{path} not found. Notebooks in this exercise are sequential - "
                f"run the earlier notebook that produces '{name}' first.\n"
                f"On Colab, note that every notebook tab is a separate runtime "
                f"with its own filesystem, so a file saved in another tab is "
                f"not here. Every result this exercise passes forward is a "
                f"training run of yours, so nothing is rebuilt automatically. "
                f"Either run the notebooks one after another in the same "
                f"runtime, or download the '{RESULTS}/' folder from the "
                f"earlier tab (Files panel, right-click, Download) and upload "
                f"it here before this cell.")
        return None
    return dict(np.load(path, allow_pickle=True))


# ── tables ────────────────────────────────────────────────────────────────

def confusion(cct_true, cct_pred, threshold=PROTECTION_TIME):
    """The 2x2 that a screening tool is actually judged on.

    Returns a dict with ``tp``, ``fp``, ``fn``, ``tn``, ``missed``, ``alarms``,
    ``recall`` and ``accuracy``, where **positive means insecure** — the thing
    you are hunting for.

    The naming is deliberate. ``missed`` is a contingency that would have taken
    the system down and was screened out; ``alarms`` is a contingency that was
    flagged and turned out to be fine. They cost different things and this
    function refuses to add them together.
    """
    t = np.asarray(cct_true, float)
    p = np.asarray(cct_pred, float)
    true_insecure = t <= threshold
    pred_insecure = p <= threshold
    tp = int((true_insecure & pred_insecure).sum())
    fp = int((~true_insecure & pred_insecure).sum())
    fn = int((true_insecure & ~pred_insecure).sum())
    tn = int((~true_insecure & ~pred_insecure).sum())
    n = tp + fp + fn + tn
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "missed": fn, "alarms": fp,
            "recall": tp / max(tp + fn, 1),
            "accuracy": (tp + tn) / max(n, 1),
            "n": n, "threshold": float(threshold)}


def screening_table(rows, headers=("model", "MAE [ms]", "worst [ms]",
                                   "missed", "false alarms", "accuracy")):
    """A Markdown table with the columns this exercise insists on.

    ``rows`` is a list of lists already formatted as strings. Uses the shared
    ``course_core.error_table`` underneath — this file does not shadow it.
    """
    from course_core import error_table
    return error_table(rows, headers)


# ── plots ─────────────────────────────────────────────────────────────────

def plot_network(outage=None, node_values=None, fault_bus=None, ax=None,
                 title="", label=""):
    """The six-bus network, with the outaged line struck through.

    ``node_values`` colours the buses by any per-bus quantity; ``fault_bus``
    rings one bus in orange. Draw this next to every screening result — a
    ranking of contingencies by index is unreadable, and the same ranking drawn
    on the network is obvious.
    """
    import matplotlib.pyplot as plt
    use_course_style()
    ax = ax if ax is not None else plt.subplots(figsize=(6.4, 4.0))[1]
    for k, (f, t, *_rest) in enumerate(BRANCHES):
        x = [BUS_XY[f, 0], BUS_XY[t, 0]]
        y = [BUS_XY[f, 1], BUS_XY[t, 1]]
        if k == outage:
            ax.plot(x, y, ls=":", lw=1.6, color=ORANGE, zorder=1)
            ax.plot(np.mean(x), np.mean(y), "x", ms=13, mew=2.6,
                    color=ORANGE, zorder=4)
        else:
            ax.plot(x, y, "-", lw=2.0, color="#39404F", zorder=1)
        ax.annotate(str(k), (np.mean(x), np.mean(y)), color=MUTED,
                    fontsize=8, xytext=(4, 4), textcoords="offset points")
    if node_values is None:
        colours = [CYAN if BUSES[i][1] != "load" else MUTED
                   for i in range(N_BUS)]
        sc = ax.scatter(BUS_XY[:, 0], BUS_XY[:, 1], s=380, c=colours,
                        zorder=3, edgecolors=BG, linewidths=1.6)
    else:
        sc = ax.scatter(BUS_XY[:, 0], BUS_XY[:, 1], s=380,
                        c=np.asarray(node_values).ravel(), cmap="viridis",
                        zorder=3, edgecolors=BG, linewidths=1.6)
        plt.colorbar(sc, ax=ax, label=label, shrink=0.82)
    if fault_bus is not None:
        ax.scatter([BUS_XY[fault_bus, 0]], [BUS_XY[fault_bus, 1]], s=760,
                   facecolors="none", edgecolors=ORANGE, linewidths=2.2,
                   zorder=2)
    for i, (name, *_rest) in enumerate(BUSES):
        ax.annotate(f"{i}  {name}", BUS_XY[i], color=TXT, fontsize=8,
                    xytext=(0, 16), textcoords="offset points", ha="center")
    ax.set_xlim(-0.7, 5.4)
    ax.set_ylim(-0.5, 2.5)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.set_title(title, fontsize=10)
    return ax


def plot_swing(T, D, W, t_fault=None, t_clear=None, f0=50.0, title="",
               limit=180.0):
    """Rotor angles and machine frequency, with the fault window shaded.

    Same figure as Ex_12.1's, plus the separation limit drawn as a line,
    because in this set the limit is the definition of the label.
    """
    import matplotlib.pyplot as plt
    use_course_style()
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
    for k in range(D.shape[1]):
        ax[0].plot(T, np.degrees(D[:, k] - D[:, 0]), lw=1.6,
                   color=[CYAN, AMBER, PURPLE][k % 3], label=f"machine {k}")
        ax[1].plot(T, W[:, k] / (2 * np.pi), lw=1.6,
                   color=[CYAN, AMBER, PURPLE][k % 3])
    for a in ax:
        if t_fault is not None and t_clear is not None:
            a.axvspan(t_fault, t_clear, color=ORANGE, alpha=0.16, lw=0)
    ax[0].axhline(limit, color=ORANGE, lw=1.0, ls="--")
    ax[0].set_xlabel("t  (s)")
    ax[0].set_ylabel("angle w.r.t. machine 0  (deg)")
    ax[1].set_xlabel("t  (s)"); ax[1].set_ylabel("frequency  (Hz)")
    ax[1].axhline(f0, color=MUTED, lw=0.8, ls=":")
    ax[0].legend(fontsize=8)
    fig.suptitle(title + "   (shaded = fault on, dashed = 180° limit)",
                 fontsize=10)
    fig.tight_layout()
    return fig


def plot_screening(cct_true, cct_pred=None, labels=None,
                   threshold=PROTECTION_TIME, title="", ax=None):
    """Contingencies ranked by clearing time — the control-room picture.

    Sorted by whichever series is being *used to decide*: the prediction if one
    is given, the truth otherwise. Anything left of the threshold line is a
    case somebody has to look at.
    """
    import matplotlib.pyplot as plt
    use_course_style()
    ax = ax if ax is not None else plt.subplots(figsize=(7.6, 3.6))[1]
    t = np.asarray(cct_true, float).ravel()
    key = t if cct_pred is None else np.asarray(cct_pred, float).ravel()
    order = np.argsort(key)
    x = np.arange(len(order))
    ax.bar(x, t[order] * 1e3, width=0.62, color=CYAN, label="true CCT")
    if cct_pred is not None:
        p = np.asarray(cct_pred, float).ravel()
        ax.plot(x, p[order] * 1e3, "o", ms=5, color=AMBER, label="predicted")
    ax.axhline(threshold * 1e3, color=ORANGE, lw=1.4, ls="--",
               label=f"protection {threshold*1e3:.0f} ms")
    if labels is not None:
        ax.set_xticks(x)
        ax.set_xticklabels([labels[i] for i in order], rotation=35,
                           ha="right", fontsize=8)
    ax.set_ylabel("clearing time  (ms)")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    return ax


def plot_parity(cct_true, cct_pred, threshold=PROTECTION_TIME, title="",
                ax=None):
    """Predicted against true CCT, with the two error quadrants shaded.

    The quadrant below-right is a **missed insecure case**; the quadrant
    above-left is a **false alarm**. Drawing them in different colours is the
    single most useful thing this file does, because the mean absolute error
    that everyone reports treats a point in one as identical to a point in the
    other.
    """
    import matplotlib.pyplot as plt
    use_course_style()
    ax = ax if ax is not None else plt.subplots(figsize=(4.8, 4.6))[1]
    t = np.asarray(cct_true, float).ravel() * 1e3
    p = np.asarray(cct_pred, float).ravel() * 1e3
    thr = threshold * 1e3
    hi = max(t.max(), p.max()) * 1.05
    ax.axhspan(0, thr, xmin=thr / hi, xmax=1.0, color=ORANGE, alpha=0.10, lw=0)
    ax.axhspan(thr, hi, xmin=0.0, xmax=thr / hi, color=CYAN, alpha=0.10, lw=0)
    ax.plot([0, hi], [0, hi], "-", lw=1.0, color=MUTED)
    ax.axhline(thr, color=ORANGE, lw=1.0, ls="--")
    ax.axvline(thr, color=ORANGE, lw=1.0, ls="--")
    ax.plot(t, p, "o", ms=4, alpha=0.65, color=CYAN)
    ax.set_xlim(0, hi); ax.set_ylim(0, hi)
    ax.set_xlabel("true CCT  (ms)"); ax.set_ylabel("predicted CCT  (ms)")
    ax.set_title(title, fontsize=10)
    return ax
