r"""Ex_11.2 — the problem: the cheapest lap of a JetRacer round a course.

*Deep Learning for Engineering* — MSc, Aalborg University.
Remus Teodorescu (ret@et.aau.dk), with support from Research Assistant
Noman Khan (nomank@energy.aau.dk).

**Reference texts.** Liu, *PINN with Python: An Introduction* (2025); Raissi,
Perdikaris & Karniadakis, *Physics-informed neural networks*, J. Comput. Phys.
**378** (2019) 686–707. These are the works to read for the theory. The code,
the problem and the exposition here are original to this course.

## The problem

A small electric car drives a closed course of straights and corners. At
every point of the lap it has a speed ``v(s)``; the question is which speed
profile makes a lap cost least, where the cost counts the energy from the
pack and the time:

    J = E + w T,          E = sum of P dt,   T = sum of dt,   dt = ds / v
    F = c_rr m g + (1/2) rho C_d A v^2 + m v dv/ds        the force the motor must supply
    i = max(F / K, 0)                                     the motor current; no regeneration
    P = i^2 R_a + K i v + P_hotel                         winding heat, traction, electronics
    v <= min(sqrt(mu g / kappa), v_max)                   the grip limit, and the car's top speed

Driving slowly pays the electronics for longer; driving fast pays in winding
heat and in the kinetic energy thrown away before each corner. The best lap
is in between.

## The numbers are placeholders

**The car's parameters below are placeholders of the right order, not
measurements.** The mass, the friction coefficient, the rolling coefficient,
the traction constant, the armature resistance, the top speed and the hotel
load are all measured on your own car in the on-car notebook of this set; put
your values here before you believe a joule of the answer. The course is a
placeholder too: survey your own.

## What is in here

The data, the course, the lap cost (for NumPy arrays and for torch tensors),
the classical optimiser, and the synthetic coast-down of section 6. The
network is written in the notebook.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "G", "MASS", "MU", "C_RR", "K_TRACTION", "R_ARM", "P_HOTEL", "V_MAX", "V_FLOOR", "W_TIME",
    "RHO_AIR", "CD_A", "SEGMENTS", "LAP_LENGTH",
    "course", "ceiling", "lap_cost", "optimise", "coast_down", "coast_down_model", "describe_problem",
]

# ------------------------------------------------- the car: PLACEHOLDERS, measure your own
G = 9.81                 #: m/s^2
MASS = 1.05              #: kg
MU = 0.65                #: tyre-floor friction coefficient
C_RR = 0.02              #: rolling resistance coefficient
K_TRACTION = 1.0         #: N/A, torque constant x gear ratio / wheel radius, lumped
R_ARM = 1.5              #: ohm, armature and wiring
P_HOTEL = 7.0            #: W, the Nano, the camera and the display
V_MAX = 3.0              #: m/s, top speed at the capped throttle
V_FLOOR = 0.3            #: m/s, below this the car stalls
RHO_AIR = 1.2            #: kg/m^3
CD_A = 0.02              #: m^2, drag coefficient x frontal area
W_TIME = 5.0             #: J/s, what a second of lap time is worth (fixed by the competition)

# ------------------------------------------------- the course: a PLACEHOLDER, survey your own
#: ``(length in m, curvature in 1/m)`` for each stretch of the lap, in order.
SEGMENTS = [(3.0, 0.0), (2.0, 0.9), (4.0, 0.0), (1.5, 1.4), (2.5, 0.0), (2.0, 0.6), (3.0, 0.0)]
LAP_LENGTH = float(sum(length for length, _ in SEGMENTS))          #: 18 m


def course(ds):
    """The lap cut into steps of ``ds`` metres: the arc length at the middle of
    each step, and the curvature there."""
    n = int(round(LAP_LENGTH / ds))
    s = (np.arange(n) + 0.5) * ds
    edges = np.cumsum([0.0] + [length for length, _ in SEGMENTS])
    curv = np.array([SEGMENTS[min(np.searchsorted(edges, x, side="right") - 1, len(SEGMENTS) - 1)][1] for x in s])
    return s, curv


def ceiling(curvature):
    """The highest speed allowed at each step: the grip limit
    ``sqrt(mu g / kappa)`` in a corner, the car's top speed on a straight."""
    k = np.maximum(np.asarray(curvature, float), 1e-9)
    return np.minimum(np.sqrt(MU * G / k), V_MAX)


def lap_cost(v, ds):
    """Energy in J, time in s and the cost ``J = E + w T`` of one lap driven at
    speeds ``v`` (one per step of ``ds``). The lap is closed: the step after
    the last is the first. Works on a NumPy array and on a torch tensor, so
    autograd can differentiate through it."""
    if isinstance(v, np.ndarray):
        roll, clip0, tot = np.roll, (lambda a: np.maximum(a, 0.0)), np.sum
    else:
        import torch
        roll, clip0, tot = torch.roll, (lambda a: torch.clamp(a, min=0.0)), torch.sum
    dv_ds = (roll(v, -1) - roll(v, 1)) / (2.0 * ds)              # central difference round the lap
    force = C_RR * MASS * G + 0.5 * RHO_AIR * CD_A * v ** 2 + MASS * v * dv_ds
    current = clip0(force / K_TRACTION)                          # braking draws no current and returns none
    power = current ** 2 * R_ARM + K_TRACTION * current * v + P_HOTEL
    dt = ds / v
    energy, time = tot(power * dt), tot(dt)
    return energy, time, energy + W_TIME * time


def optimise(ds, start=None):
    """The classical answer: the speed at every step as one unknown each,
    bounded between the stall speed and the ceiling, minimised by L-BFGS-B
    (SciPy). Returns ``(s, v, energy, time, cost)``."""
    from scipy.optimize import minimize
    s, curv = course(ds)
    top = ceiling(curv)
    x0 = np.minimum(top, 1.5) if start is None else np.minimum(top, start)
    res = minimize(lambda v: float(lap_cost(v, ds)[2]), x0, method="L-BFGS-B",
                   bounds=[(V_FLOOR, float(u)) for u in top],
                   options=dict(maxiter=20000, maxfun=2000000, ftol=1e-14, gtol=1e-9))
    e, t, j = lap_cost(res.x, ds)
    return s, res.x, float(e), float(t), float(j)


# ------------------------------------------------------------- the coast-down
def coast_down_model(t, c_rr, v0=2.5):
    """The speed of the car coasting from ``v0`` with the drive cut:
    ``m dv/dt = -c_rr m g - (1/2) rho C_d A v^2``, integrated (SciPy)."""
    from scipy.integrate import solve_ivp
    f = lambda _, v: -(c_rr * G + 0.5 * RHO_AIR * CD_A * v ** 2 / MASS)
    return solve_ivp(f, (0.0, float(t[-1])), [v0], t_eval=t, rtol=1e-10, atol=1e-12).y[0]


def coast_down(seed=112, noise=0.03):
    """A coast-down as the car would log it: the speed every 50 ms for 6 s,
    from 2.5 m/s, with noise of 0.03 m/s. **Synthetic**: the model above at
    :data:`C_RR`, with noise added. The car has no encoder, so on the real car
    the speed has to come from lap timing or from the camera. Returns
    ``(t, v)``."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 6.0, 121)
    return t, coast_down_model(t, C_RR) + rng.normal(0.0, noise, len(t))


def describe_problem() -> None:
    """Print the car, the course and the numbers they imply."""
    print("  the car's numbers are PLACEHOLDERS - measure your own in the on-car notebook")
    print(f"  car              : {MASS:.2f} kg, friction {MU:.2f}, rolling {C_RR:.2f}, top speed {V_MAX:.1f} m/s")
    print(f"  motor            : {K_TRACTION:.1f} N/A at the wheels, {R_ARM:.1f} ohm; electronics {P_HOTEL:.0f} W all the time")
    print(f"  course           : {LAP_LENGTH:.0f} m, three corners of radius "
          + ", ".join(f"{1 / k:.2f}" for _, k in SEGMENTS if k > 0) + " m")
    print("  grip limit       : " + ", ".join(f"{np.sqrt(MU * G / k):.2f}" for _, k in SEGMENTS if k > 0) + " m/s in the three corners")
    print(f"  cost             : J = E + w T with w = {W_TIME:.0f} J/s")
    for v in (0.5, 1.5, 3.0):
        f = C_RR * MASS * G + 0.5 * RHO_AIR * CD_A * v ** 2
        print(f"  steady {v:.1f} m/s   : traction {f * v + (f / K_TRACTION) ** 2 * R_ARM:.2f} W, electronics {P_HOTEL:.0f} W, "
              f"{(f * v + (f / K_TRACTION) ** 2 * R_ARM + P_HOTEL + W_TIME) / v:.1f} J per metre with the time counted")
