r"""MP9.1B — The pump's speed reference. The model, the reference model and
the ground truth.

*Deep Learning for Engineering* — MSc, Aalborg University.
Remus Teodorescu (ret@et.aau.dk), with support from Research Assistant
Noman Khan (nomank@energy.aau.dk).

This module extends Ex09.1's ``pipe.py`` (course policy C10 rule 1: one set of
physics and constants for the exercise and its mini project) and is used by
the notebook of MP9.1B (course policy C14). It takes over the model of
``tools/miniprojects/ex091_truth.py`` (MP9.1B, cases 1 and 2), unchanged.

## The problem

The serpentine's cold plate carries a power module whose power steps up and
down with the load. The pump's flow is what a controller can change: enough to
keep the module at or below 80 °C, no more, because pumping costs energy. Given
the module's power over the next 2000 s - a staircase - what flow should be
sent to the pump at every instant?

The cold plate is one thermal mass ``C`` (aluminium, 250 x 130 x 12 mm); the
module sits on it with a resistance ``R_m``; the coolant enters at 25 °C and
takes heat at a rate set by the flow ``Q``:

    C dT/dt = P(t) - G(Q) (T - T_in),        T(0) = T_in
    T_module = T + R_m P(t) <= 80 °C
    minimise   integral dp(Q) Q dt                               (the pump's hydraulic energy)

``G(Q) = m c_p (1 - exp(-hA / m c_p))`` is the heat the coolant takes per
kelvin of the plate's lead over the supply; ``hA`` comes from Sieder and
Tate's correlation (the tube is in its thermal entrance region) below Re 2300,
Dittus and Boelter's above Re 4000, a straight blend between; ``dp`` from
64/Re, Blasius's 0.316 Re^-1/4, the same blend (``pipe.pressure_drop``). All
correlations and values are from memory and flagged (see ex091_truth.py).

## The staircase as an input

The staircase is that of ex091_truth.py's case 2: 120 W from 0 s, a step
``P_step`` from 1200 s to 1600 s, then 150 W to 2000 s. ``P_step`` is the
**input** of the network and the parameter of a new case: 300 W (case 1, laminar
is enough) to 800 W (the flow must turn turbulent; case 2 is 700 W).

## No exact solution

The optimal flow schedule is the solution of an optimal control problem with a
state constraint and a nonlinear, regime-switching conductance and cost. It has
no closed form. The plate's energy balance alone is linear in T for a flow held
constant, so between two decisions it is integrated exactly; that is what the
two solvers below use.

## Two solvers

* the **reference model** (:func:`reference`): direct transcription - the flow
  held constant on 40 s intervals, the plate integrated exactly between them,
  the hottest module temperature checked at 4 points per interval, minimised by
  SLSQP. It is what an engineer would run for a new staircase, and what the
  network is timed against.
* the **ground truth** (:func:`ground_truth`): the same transcription on 10 s
  intervals, the constraint checked at 10 points per interval, the module
  temperature then checked at 50 points per interval; the convergence study on
  80, 40, 20 and 10 s intervals (:func:`convergence_study`).
"""

from __future__ import annotations

import time

import numpy as np
from scipy.optimize import minimize, brentq

import pipe as pp

# ------------------------------------------------------------------ the data (ex091_truth.py, from memory)
CP, K_FLUID = 3400.0, 0.40           #: J/(kg K), W/(m K): the coolant  [assumed, typical]
T_IN, T_LIMIT, R_MODULE = 25.0, 80.0, 0.05   #: degC supply, degC limit at the module, K/W module to plate  [assumed]
LEG_LEN = pp.LEG_X[1] - pp.LEG_X[0]  #: m, one straight leg
TUBE_LEN = len(pp.LEG_Y) * LEG_LEN + (len(pp.LEG_Y) - 1) * np.pi * pp.BEND_R   #: m, the serpentine, 1.03 m
C_PLATE = 2.43e6 * np.prod(pp.PLATE) #: J/K, the aluminium plate as one thermal mass (rho c_p 2.43 MJ/(m3 K))
A_TUBE = np.pi * pp.D_TUBE * TUBE_LEN  #: m^2, the tube's inner surface
PR = pp.NU / (K_FLUID / (pp.RHO * CP))  #: Prandtl number, about 21
LMIN = 1.0 / 6e4                     #: m^3/s per L/min
Q_MIN, Q_MAX = 0.05, 4.0             #: L/min, the pump's range: 0.05 L/min to 4 L/min (Re 75 to 6000)
T_END = 2000.0                       #: s, the window
T_UP, T_DOWN = 1200.0, 1600.0        #: s, the step's start and end
P_BASE, P_AFTER = 120.0, 150.0       #: W, before and after the step
P_MIN, P_MAX = 300.0, 800.0          #: W, the range of step powers the network takes as an input
P_TRAIN = (300.0, 400.0, 500.0, 600.0, 700.0, 800.0)   #: W, the step powers the training may use
P_TEST = (350.0, 550.0, 650.0, 750.0)   #: W, the step powers the answers are scored at (not trained on)
T_TOL = 0.2                          #: K, the agreed tolerance on the limit: the module at most 80.2 degC
PUMP_TOL = 0.05                      #: the agreed tolerance on the pumping energy: within 5 % of the ground truth


def power(t, p_step):
    """W, the module's power at time ``t`` (s) for a step of ``p_step`` W."""
    t = np.asarray(t, dtype=float)
    return np.where((t >= T_UP) & (t < T_DOWN), p_step, np.where(t < T_UP, P_BASE, P_AFTER))


def reynolds(q_lmin):
    """The Reynolds number in the tube at the flow ``q_lmin`` (L/min)."""
    return pp.reynolds(np.asarray(q_lmin) * LMIN)


def blend(re):
    """0 below Re 2300 (laminar), 1 above Re 4000 (turbulent), a straight line between."""
    return np.clip((re - pp.RE_LAMINAR) / (pp.RE_TURBULENT - pp.RE_LAMINAR), 0.0, 1.0)


def h_area(q_lmin):
    """W/K, h A of the tube: Sieder and Tate (developing, never below 3.66) below
    Re 2300, Dittus and Boelter above 4000, a straight blend between."""
    re = reynolds(q_lmin)
    nu_lam = np.maximum(3.66, 1.86 * (re * PR * pp.D_TUBE / TUBE_LEN) ** (1 / 3))
    nu_tur = 0.023 * re ** 0.8 * PR ** 0.4
    w = blend(re)
    return ((1 - w) * nu_lam + w * nu_tur) * K_FLUID / pp.D_TUBE * A_TUBE


def conductance(q_lmin):
    """W/K from the plate to the coolant at the flow ``q_lmin`` (L/min): the
    coolant warms along the tube, so G = m c_p (1 - exp(-hA / m c_p))."""
    mc = np.asarray(q_lmin) * LMIN * pp.RHO * CP
    return mc * (1 - np.exp(-h_area(q_lmin) / mc))


def pump_power(q_lmin):
    """W, the pump's hydraulic power dp Q at the flow ``q_lmin`` (L/min), with the
    friction factor of the flow's regime (pipe.pressure_drop)."""
    q = np.asarray(q_lmin) * LMIN
    return pp.pressure_drop(q, TUBE_LEN) * q


def steady_flow(p):
    """L/min, the least flow that holds the module at its limit under a steady
    power ``p`` (W)."""
    f = lambda q: conductance(q) * (T_LIMIT - R_MODULE * p - T_IN) - p
    return brentq(f, Q_MIN / 10, Q_MAX) if f(Q_MAX) >= 0 else np.nan


def pumping_scale(p_step):
    """J, an estimate of the least pumping energy for a step of ``p_step`` W, from
    the physics alone: the pump's power at the least steady flow for that power
    (section A4 of the notebook), held for the step's 400 s. It is what the
    network's pumping term is divided by, so that every step power weighs alike."""
    p_step = np.atleast_1d(np.asarray(p_step, dtype=float))
    return np.array([(T_DOWN - T_UP) * float(pump_power(steady_flow(p))) for p in p_step])


# ---------------------------------------------------------- the plate's energy balance
def simulate(q, dt, p_step, sub=1, T0=T_IN):
    """The plate's temperature under flows ``q`` (L/min) held over intervals of
    ``dt`` s: exact between the decisions, since C dT/dt = P - G (T - T_in) is
    linear in T with P and G constant. Returns the times (s) and the module's
    temperature (degC) at ``sub`` points inside every interval, and the plate's
    temperature at the end of each interval."""
    q = np.asarray(q, dtype=float)
    n = q.size
    P = power((np.arange(n) + 0.5) * dt, p_step)
    G = conductance(q)
    T_inf = T_IN + P / G
    a = np.exp(-G * dt / C_PLATE)
    T_end = np.empty(n); T = T0
    for k in range(n):                                       # the plate at the end of each interval
        T = T_inf[k] + (T - T_inf[k]) * a[k]
        T_end[k] = T
    T_start = np.r_[T0, T_end[:-1]]
    tt = np.linspace(0, dt, sub + 1)[1:]
    Ts = T_inf[:, None] + (T_start - T_inf)[:, None] * np.exp(-np.outer(G, tt) / C_PLATE)
    times = (np.arange(n)[:, None] * dt + tt[None, :]).ravel()
    return times, (Ts + R_MODULE * P[:, None]).ravel(), T_end


def module_peak(q, dt, p_step, sub=50):
    """degC, the hottest module temperature over the window."""
    return float(simulate(q, dt, p_step, sub)[1].max())


def pumping(q, dt):
    """J, the pump's hydraulic energy for flows ``q`` (L/min) held over ``dt`` s."""
    return float(np.sum(pump_power(q)) * dt)


# ------------------------------------------------------- the optimisation
def optimise(p_step, dt, sub=4, x0=None, ftol=1e-12, maxiter=1000):
    """Direct transcription: the least pumping energy with the module at or below
    its limit at ``sub`` points of every interval, by SLSQP over the flows held on
    intervals of ``dt`` s. Returns the flows (L/min), the result and the seconds."""
    n = int(round(T_END / dt))
    cons = {"type": "ineq", "fun": lambda x: T_LIMIT - simulate(x * Q_MAX, dt, p_step, sub)[1]}
    cost = lambda x: pumping(x * Q_MAX, dt)
    x0 = np.full(n, 0.5) if x0 is None else np.asarray(x0) / Q_MAX
    t0 = time.perf_counter()
    res = minimize(cost, x0, method="SLSQP", constraints=[cons], bounds=[(Q_MIN / Q_MAX, 1.0)] * n,
                   options=dict(maxiter=maxiter, ftol=ftol))
    return res.x * Q_MAX, res, time.perf_counter() - t0


REF_DT = 40.0     #: s, the reference model's intervals
TRUTH_DT = 10.0   #: s, the ground truth's intervals
TRUTH_DTS = (80.0, 40.0, 20.0, 10.0)   #: the convergence study


def reference(p_step, dt=REF_DT):
    """The reference model for a new staircase: SLSQP on 40 s intervals, started
    from a constant 2 L/min. Returns the flows (L/min), the times, the hottest
    module (degC, checked at 50 points per interval), the pumping (J) and the
    wall-clock seconds."""
    q, res, sec = optimise(p_step, dt)
    return dict(q=q, t=np.arange(q.size) * dt, dt=dt, peak=module_peak(q, dt, p_step), pumping=pumping(q, dt),
                seconds=sec, iterations=res.nit)


def ground_truth(p_step, dt=TRUTH_DT):
    """The ground truth for a staircase: SLSQP on 10 s intervals with the limit at
    10 points per interval, started from the 20 s solution."""
    q20, _, _ = optimise(p_step, 2 * dt, sub=10)
    q, res, sec = optimise(p_step, dt, sub=10, x0=np.repeat(q20, 2))
    return dict(q=q, t=np.arange(q.size) * dt, dt=dt, peak=module_peak(q, dt, p_step), pumping=pumping(q, dt),
                seconds=sec, iterations=res.nit)


def convergence_study(p_step=700.0, dts=TRUTH_DTS, verbose=True):
    """The pumping energy and the hottest module on intervals each half as long."""
    rows, q_prev = [], None
    for dt in dts:
        x0 = None if q_prev is None else np.repeat(q_prev, int(round(rows[-1]["dt"] / dt)))
        q, res, sec = optimise(p_step, dt, sub=10, x0=x0)
        rows.append(dict(dt=dt, q=q, pumping=pumping(q, dt), peak=module_peak(q, dt, p_step), seconds=sec))
        q_prev = q
        if verbose:
            print(f"   intervals of {dt:4.0f} s: pumping {rows[-1]['pumping']:.3f} J, hottest module {rows[-1]['peak']:.3f} degC, "
                  f"flow at the step {q[int(T_UP / dt)]:.3f} L/min  ({sec:.0f} s)", flush=True)
    E = np.array([r["pumping"] for r in rows])
    d = np.abs(np.diff(E))
    if verbose:
        print("   change of the pumping energy between interval lengths:", " ".join(f"{v:.3f} J" for v in d),
              "; ratios", " ".join(f"{a / b:.1f}" for a, b in zip(d[:-1], d[1:])))
    p = np.log(d[-2] / d[-1]) / np.log(2)
    E_ex = E[-1] + (E[-1] - E[-2]) / (2 ** p - 1)
    if verbose:
        print(f"   observed order {p:.2f}; Richardson extrapolation {E_ex:.3f} J (10 s intervals {E[-1]:.3f} J, "
              f"{100 * (E[-1] / E_ex - 1):+.2f} %)")
    return rows, p, E_ex


# ------------------------------------------------------------- the checks
def check_constant_flow(q=2.0, p_step=700.0, dt=10.0):
    """The exact integration against a fine Runge-Kutta march of the same ODE."""
    from scipy.integrate import solve_ivp
    n = int(T_END / dt)
    t, Tm, _ = simulate(np.full(n, q), dt, p_step, sub=10)
    G = conductance(q)
    sol = solve_ivp(lambda tt, T: (power(tt, p_step) - G * (T - T_IN)) / C_PLATE, (0, T_END), [T_IN],
                    t_eval=t, max_step=1.0, rtol=1e-10, atol=1e-10)
    return float(np.abs(sol.y[0] + R_MODULE * power(t - 1e-9, p_step) - Tm).max())


def check_energy(q, dt, p_step):
    """The plate's energy balance over the window: heat in (the module) minus
    heat out (the coolant) equals heat stored, J."""
    n = q.size
    tt = np.linspace(0, dt, 201)
    t, _, T_end = simulate(q, dt, p_step)
    T_start = np.r_[T_IN, T_end[:-1]]
    G = conductance(q); P = power((np.arange(n) + 0.5) * dt, p_step); T_inf = T_IN + P / G
    out = 0.0
    for k in range(n):
        T = T_inf[k] + (T_start[k] - T_inf[k]) * np.exp(-G[k] * tt / C_PLATE)
        out += np.trapezoid(G[k] * (T - T_IN), tt)
    heat_in = float(np.sum(P) * dt)
    stored = C_PLATE * (T_end[-1] - T_IN)
    return heat_in, out, stored, heat_in - out - stored


def reactive(p_step, dt=10.0):
    """A controller that sees only the present power: in each interval, the least
    flow that brings the module to its limit by the interval's end."""
    n, T, q = int(round(T_END / dt)), T_IN, []
    for k in range(n):
        P = float(power(k * dt + 1e-9, p_step))
        def end_temp(qq):
            G = conductance(qq); T_inf = T_IN + P / G
            return T_inf + (T - T_inf) * np.exp(-G * dt / C_PLATE) + R_MODULE * P
        qk = Q_MIN if end_temp(Q_MIN) <= T_LIMIT else (Q_MAX if end_temp(Q_MAX) > T_LIMIT else brentq(lambda qq: end_temp(qq) - T_LIMIT, Q_MIN, Q_MAX))
        G = conductance(qk); T_inf = T_IN + P / G
        T = T_inf + (T - T_inf) * np.exp(-G * dt / C_PLATE)
        q.append(qk)
    q = np.array(q)
    return q, module_peak(q, dt, p_step)


# ------------------------------------------------------------ the truth file
TRUTH_FILE = "MP9.1B_truth.npz"


def write_truth(path=TRUTH_FILE):
    """The ground truth for the training and test step powers (flows on 10 s
    intervals, the module's temperature, the pumping), the convergence study at
    700 W, the steady flow against power, the checks and every parameter."""
    out = {}
    print("convergence at 700 W")
    rows, p, E_ex = convergence_study(700.0)
    out["conv_dt"] = np.array([r["dt"] for r in rows]); out["conv_pumping"] = np.array([r["pumping"] for r in rows])
    out["conv_peak"] = np.array([r["peak"] for r in rows]); out["conv_order"] = np.array(p); out["conv_extrapolated"] = np.array(E_ex)
    out["conv_seconds"] = np.array([r["seconds"] for r in rows])
    out["check_exact_integration"] = np.array(check_constant_flow())
    steps = sorted(set(P_TRAIN) | set(P_TEST))
    out["step_powers"] = np.array(steps, dtype=float)
    for P in steps:
        t0 = time.perf_counter()
        g = rows[-1] if P == 700.0 else None
        if g is not None:
            q = g["q"]
        else:
            q = ground_truth(P)["q"]
        key = f"{int(P)}W"
        t, Tm, _ = simulate(q, TRUTH_DT, P, sub=10)
        out[f"flow_{key}"] = q; out[f"module_{key}"] = Tm
        out[f"pumping_{key}"] = np.array(pumping(q, TRUTH_DT)); out[f"peak_{key}"] = np.array(Tm.max())
        out[f"balance_{key}"] = np.array(check_energy(q, TRUTH_DT, P))
        print(f"   truth at {P:.0f} W: pumping {pumping(q, TRUTH_DT):.3f} J, hottest {Tm.max():.3f} degC, most {q.max():.2f} L/min "
              f"(Re {reynolds(q.max()):.0f}), energy balance off by {out[f'balance_{key}'][-1]:.1e} J  ({time.perf_counter() - t0:.0f} s)", flush=True)
    out["t"] = np.arange(int(T_END / TRUTH_DT)) * TRUTH_DT
    out["t_module"] = simulate(np.ones(int(T_END / TRUTH_DT)), TRUTH_DT, 300.0, sub=10)[0]
    Ps = np.arange(100, 851, 10.0)
    out["steady_power"] = Ps; out["steady_flow"] = np.array([steady_flow(P) for P in Ps])
    out["params"] = np.array([f"{k} = {v}" for k, v in dict(
        C_PLATE=f"{C_PLATE:.1f} J/K (aluminium 250 x 130 x 12 mm, rho c_p 2.43 MJ/(m3 K))", R_MODULE="0.05 K/W, assumed",
        T_IN="25 degC supply, assumed", T_LIMIT="80 degC, assumed", CP="3400 J/(kg K), typical", K_FLUID="0.40 W/(m K), typical",
        TUBE=f"{TUBE_LEN:.3f} m of 6 mm tube", PUMP="0.05-4 L/min", POWER="120 W to 1200 s, P_step to 1600 s, 150 W to 2000 s",
        CORRELATIONS="Sieder-Tate, Dittus-Boelter, 64/Re, Blasius, straight blend 2300-4000: all from memory").items()])
    out["README"] = np.array(
        "MP9.1B ground truth: least pumping energy with the module at or below 80 degC, SLSQP on 10 s intervals. "
        "step_powers: the step powers P (W) solved; flow_<P>W: flow (L/min) held on [t, t + 10 s); module_<P>W: module "
        "temperature (degC) at t_module (10 points per interval); pumping_<P>W: J; peak_<P>W: degC; balance_<P>W: heat in, "
        "heat out to the coolant, heat stored, residual (J); conv_*: the 700 W study on 80, 40, 20, 10 s intervals; "
        "check_exact_integration: the exact interval integration against Runge-Kutta, K; steady_power, steady_flow: the "
        "least steady flow (L/min) holding the limit against a steady power (W); params: the data.")
    np.savez_compressed(path, **out)
    print("written", path)


# ------------------------------------------------------- the training points
def sample_steps(n, seed=88):
    """n step powers (W), uniform over the trained range."""
    return np.random.default_rng(seed).uniform(P_MIN, P_MAX, n)


if __name__ == "__main__":
    import sys
    if "--save" in sys.argv:
        write_truth()
    else:
        print("check: exact integration against Runge-Kutta", check_constant_flow())
        convergence_study()
