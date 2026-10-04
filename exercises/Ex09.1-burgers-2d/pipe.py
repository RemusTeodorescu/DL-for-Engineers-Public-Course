r"""Ex_09.1 (pipe version, October 2026) - the coolant in one tube of a cold plate, when the pump starts.

*Deep Learning for Engineering* - MSc, Aalborg University.
Remus Teodorescu (ret@et.aau.dk), with support from Research Assistant
Noman Khan (nomank@energy.aau.dk).

**Reference texts.** White, *Fluid Mechanics*, ch. 6 (pipe flow, the Darcy
friction factor); Liu, *PINN with Python* (2025). The code, the problem and the
exposition here are original to this course.

## The problem

A liquid cold plate carries a serpentine of copper tube, 6 mm inside
diameter. A 50/50 mix of water and ethylene glycol at 40 degC is pumped
through it at 0.3 L/min. In one straight leg of the serpentine, far enough
from the bends, the flow is laminar and the same at every cross-section, so
only the axial velocity ``u(r, t)`` is left, and Newton's law for the liquid
reduces to

    rho u_t = G + mu (u_rr + u_r / r),     u = 0 on the wall,  u = 0 at t = 0

with ``G`` the pressure drop per metre, switched on at ``t = 0`` and then
held. Its steady state is Hagen-Poiseuille's parabola, and Darcy's friction
factor ``f = 64 / Re`` follows from it. The start-up has an exact solution as
a Bessel series (Szymanski 1932): the parabola less the part that has not yet
arrived.

## Scaled units

Radius over ``R``, written through ``s = (r / R)^2`` so that the axis needs no
special treatment; time over the window ``T_END``; velocity over the steady
centre speed ``V``. Then

    u_t = A + 4 C (s u_ss + u_s),      A = 4 C,   C = nu T_END / R^2

``C`` is the viscosity in these units - the number section 6 finds.

## The numbers are typical values, not measurements

The tube, the coolant's density and viscosity, and the flow rate are values of
the right order, typed from memory; check them before quoting a result. The
pressure gradient is taken to be held from ``t = 0``: a pump settles in a small
fraction of the 4 s it takes the coolant to follow.
"""

from __future__ import annotations

import numpy as np
from scipy.special import j0, j1, jn_zeros

__all__ = [
    "R_TUBE", "D_TUBE", "RHO", "MU", "NU", "FLOW", "AREA", "U_MEAN", "RE", "G", "V", "T_END", "C_SCALED", "A_SCALED",
    "ENTRANCE", "darcy", "exact", "exact_mean", "fdm", "fdm_mean", "flow_meter", "describe_problem",
]

# ------------------------------------------------------------------ the data (typical values, from memory)
R_TUBE = 0.003                 #: m, inside radius of the tube
D_TUBE = 2 * R_TUBE            #: m, 6 mm inside
RHO = 1065.0                   #: kg/m^3, 50/50 water-glycol at 40 degC
MU = 2.5e-3                    #: Pa s, its viscosity at 40 degC
NU = MU / RHO                  #: m^2/s, kinematic viscosity
FLOW = 0.3e-3 / 60.0           #: m^3/s, 0.3 L/min
AREA = np.pi * R_TUBE ** 2     #: m^2
U_MEAN = FLOW / AREA           #: m/s, mean speed, steady
RE = U_MEAN * D_TUBE / NU      #: Reynolds number, about 450
G = 8 * MU * U_MEAN / R_TUBE ** 2   #: Pa/m, the pressure drop per metre that drives it (Poiseuille)
V = G * R_TUBE ** 2 / (4 * MU) #: m/s, the steady centre speed: the velocity scale, twice the mean
T_END = 4.0                    #: s, the window after the pump starts
C_SCALED = NU * T_END / R_TUBE ** 2   #: the viscosity in scaled units
A_SCALED = G * T_END / (RHO * V)      #: the pressure push in scaled units (= 4 C for the nominal coolant)
ENTRANCE = 0.05 * RE * D_TUBE  #: m, the length after a bend before the parabola has formed

_LAM = jn_zeros(0, 400)        # the zeros of J0


def darcy(re):
    """Darcy's friction factor of laminar pipe flow, ``64 / Re``."""
    return 64.0 / re


def exact(s, t, c=None, a=None, n=400):
    """The exact start-up velocity, scaled, at ``s = (r/R)^2`` and scaled time
    ``t = t_seconds / T_END``: Szymanski's Bessel series. ``c`` is the scaled
    viscosity (``C_SCALED`` by default) and ``a`` the scaled push
    (``A_SCALED``); the steady centre speed is ``a / (4 c)``."""
    c = C_SCALED if c is None else c
    a = A_SCALED if a is None else a
    s, t = np.broadcast_arrays(np.asarray(s, float), np.asarray(t, float))
    eta = np.sqrt(np.clip(s, 0.0, 1.0))
    lam = _LAM[:n]
    terms = 8 * j0(np.multiply.outer(eta, lam)) / (lam ** 3 * j1(lam)) * np.exp(-np.multiply.outer(4 * c * t, lam ** 2 / 4))
    return a / (4 * c) * ((1 - s) - terms.sum(axis=-1))


def exact_mean(t, c=None, a=None, n=400):
    """The exact mean velocity over the section - what a flow meter reads, scaled."""
    c = C_SCALED if c is None else c
    a = A_SCALED if a is None else a
    lam = _LAM[:n]
    t = np.asarray(t, float)
    return a / (4 * c) * (0.5 - (16 / lam ** 4 * np.exp(-np.multiply.outer(c * t, lam ** 2))).sum(axis=-1))


def fdm(n, steps, keep=40, c=None, a=None):
    """Finite differences: ``n`` nodes from the axis to the wall in ``r``, central
    differences, the axis by symmetry (the Laplacian there is ``2 u_rr``), and
    Crank-Nicolson in time with the matrix factorised once. Returns the node
    radii ``eta = r / R``, the stored scaled times and the velocity, one row per
    stored time."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    c = C_SCALED if c is None else c
    a = A_SCALED if a is None else a
    eta = np.linspace(0.0, 1.0, n)
    h = eta[1]
    m = n - 1                                     # unknowns: every node but the wall
    main = np.full(m, -2.0 / h ** 2)
    up = np.zeros(m - 1); lo = np.zeros(m - 1)
    j = np.arange(1, m)
    up[1:] = 1 / h ** 2 + 1 / (2 * h * eta[j[:-1]])  # u_rr + u_r / r at the interior nodes
    lo[:] = 1 / h ** 2 - 1 / (2 * h * eta[j])
    up[0] = 4 / h ** 2; main[0] = -4 / h ** 2      # the axis: 2 u_rr, with the mirror node
    L = sp.diags([lo, main, up], [-1, 0, 1], format="csc") * (c / 1.0)   # the operator in eta, times c
    # u_t = a + c (u_rr + u_r / r) in eta; s-form and eta-form are the same equation
    dt = 1.0 / steps
    I = sp.identity(m, format="csc")
    lu = spla.splu((I - 0.5 * dt * L).tocsc())
    B = (I + 0.5 * dt * L).tocsr()
    u = np.zeros(m)
    every = max(steps // keep, 1)
    ts, out = [0.0], [np.zeros(n)]
    for k in range(1, steps + 1):
        u = lu.solve(B @ u + dt * a)
        if k % every == 0:
            ts.append(k * dt); out.append(np.r_[u, 0.0])
    return eta, np.array(ts), np.array(out)


def fdm_mean(eta, field):
    """The mean velocity of a finite-difference profile: ``2 * integral u eta d eta``, by the trapezoidal rule."""
    return 2 * np.trapezoid(field * eta, eta, axis=-1)


def flow_meter(c_true, t_end_record=1.0, every=0.05, noise=0.01, seed=7):
    """A flow meter's record after the pump starts: the mean velocity (scaled)
    every ``every`` of the window up to ``t_end_record``, for a coolant whose
    scaled viscosity is ``c_true``, with noise of ``noise`` times the nominal
    steady mean. **Synthetic**: the exact solution, with the true viscosity."""
    rng = np.random.default_rng(seed)
    t = np.arange(every, t_end_record + 1e-9, every)
    return t, exact_mean(t, c=c_true) + noise * 0.5 * rng.standard_normal(len(t))


def describe_problem() -> None:
    """Print the data and the numbers it implies."""
    print("  values are TYPICAL, typed from memory - check before quoting")
    print(f"  tube            : {D_TUBE * 1e3:.0f} mm inside, one straight leg of a cold plate's serpentine")
    print(f"  coolant         : 50/50 water-glycol at 40 degC, rho = {RHO:.0f} kg/m^3, mu = {MU * 1e3:.2f} mPa s, nu = {NU:.2e} m^2/s")
    print(f"  flow            : {FLOW * 6e4:.1f} L/min -> mean speed {U_MEAN:.3f} m/s, Re = {RE:.0f} (laminar below about 2300)")
    print(f"  steady          : centre speed {V:.3f} m/s; pressure drop {G:.0f} Pa per metre; Darcy f = 64/Re = {darcy(RE):.3f}")
    print(f"  after a bend    : about 0.05 Re D = {ENTRANCE * 1e3:.0f} mm before the parabola has formed")
    print(f"  start-up        : R^2 / nu = {R_TUBE ** 2 / NU:.2f} s; window {T_END:.0f} s; scaled viscosity C = {C_SCALED:.3f}")
