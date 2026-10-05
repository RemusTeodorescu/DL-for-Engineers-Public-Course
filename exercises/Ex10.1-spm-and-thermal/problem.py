r"""Ex_10.1 — the problem: lithium leaving a graphite particle during a discharge.

*Deep Learning for Engineering* — MSc, Aalborg University.
Remus Teodorescu (ret@et.aau.dk), with support from Research Assistant
Noman Khan (nomank@energy.aau.dk).

**Reference texts.** Liu, *PINN with Python: An Introduction* (2025); Raissi,
Perdikaris & Karniadakis, *Physics-informed neural networks*, J. Comput. Phys.
**378** (2019) 686–707; Crank, *The Mathematics of Diffusion* (1975), for the
series solution; Chen et al., J. Electrochem. Soc. **167** (2020) 080534, for
the cell's parameters. These are the works to read for the theory. The code,
the problem and the exposition here are original to this course.

## The problem

The negative electrode of an LG M50 21700 cell is graphite, and in the single
particle model (L10.1) the whole electrode is one representative particle, a
sphere of radius 5.86 µm. During a 1C discharge, 5 A for one hour, lithium
leaves through the particle's surface at a constant rate and the lithium
inside has to diffuse outwards to replace it:

    c_t = D_s (c_rr + (2/r) c_r)          in the particle, 0 < t < 3600 s
    -D_s c_r = j                          on the surface, r = R
    c_r = 0                               at the centre (symmetry)
    c = c_0                               at t = 0

The reaction sees the concentration at the SURFACE, so what matters is how
far the surface runs ahead of the rest of the particle.

## Scaled units

Radius over R, time over the hour, and the concentration lost over
``c_ref = j t_end / R``, the scale the mass balance sets:

    u = (c_0 - c) / c_ref,    rho = r / R,    tau = t / t_end
    u_tau = C (u_rr + (2/rho) u_rho),    C u_rho = 1 at rho = 1,    u = 0 at tau = 0
    C = D_s t_end / R^2 = 3.46

The mean of ``u`` over the sphere is exactly ``3 tau``, whatever the
diffusivity: what has left the particle is the current times the time.

## The data are typed in, not read from PyBaMM

The parameter values below are those of Chen et al. (2020) as distributed
with PyBaMM, **written from memory**. Check them against
``pybamm.ParameterValues("Chen2020")`` before the set is assigned.

## What is in here

The data, the exact solution (a series), the points a network is trained on,
the quadrature of the mass balance, and the synthetic readings of section 6.
No solver: the finite-volume solver and the network are written in the
notebook.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "FARADAY", "R_PARTICLE", "D_SOLID", "C_MAX", "C_START", "CURRENT", "L_ELECTRODE",
    "AREA", "EPS_ACTIVE", "A_SPECIFIC", "J_SURFACE", "T_END", "C_REF", "C_SCALED", "NOISE",
    "exact", "concentration", "sample_particle", "surface_points", "mass_quadrature",
    "surface_readings", "describe_problem", "draw_particle",
]

# ----------------------------------------------------------------- the data
FARADAY = 96485.0        #: C/mol
R_PARTICLE = 5.86e-6     #: m, radius of a graphite particle
D_SOLID = 3.3e-14        #: m^2/s, diffusivity of lithium in the particle
C_MAX = 33133.0          #: mol/m^3, the most lithium the graphite can hold
C_START = 29866.0        #: mol/m^3, the concentration in the charged cell
CURRENT = 5.0            #: A, a 1C discharge of the 5 Ah cell
L_ELECTRODE = 85.2e-6    #: m, thickness of the negative electrode
AREA = 0.1027            #: m^2, area of the electrode sheet (1.58 m x 65 mm)
EPS_ACTIVE = 0.75        #: volume fraction of graphite in the electrode
T_END = 3600.0           #: s, the length of the discharge

A_SPECIFIC = 3.0 * EPS_ACTIVE / R_PARTICLE                           #: 1/m, particle surface per volume of electrode
J_SURFACE = CURRENT / (FARADAY * A_SPECIFIC * AREA * L_ELECTRODE)    #: mol/(m^2 s), lithium leaving the surface
C_REF = J_SURFACE * T_END / R_PARTICLE                               #: mol/m^3, the concentration scale
C_SCALED = D_SOLID * T_END / R_PARTICLE ** 2                         #: the scaled diffusivity, 3.46

#: Noise of the surface readings of section 6, mol/m^3.
NOISE = 50.0


def concentration(u):
    """A scaled loss ``u`` as a concentration in mol/m^3."""
    return C_START - C_REF * np.asarray(u)


# ------------------------------------------------------------ the exact solution
def _roots(n=60):
    """The first ``n`` positive roots of tan(x) = x, by bisection."""
    out, k = [], 1
    while len(out) < n:
        lo, hi = k * np.pi + 1e-9, (k + 0.5) * np.pi - 1e-9
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            if (np.tan(lo) - lo) * (np.tan(mid) - mid) <= 0:
                hi = mid
            else:
                lo = mid
        out.append(0.5 * (lo + hi))
        k += 1
    return np.array(out)


_ROOTS = _roots()


def exact(rho, tau, c=None):
    """The exact scaled loss ``u(rho, tau)``: a sphere with a constant flux
    through its surface (Crank, *The Mathematics of Diffusion*). ``c`` is the
    scaled diffusivity; :data:`C_SCALED` unless given. NumPy in, NumPy out."""
    c = C_SCALED if c is None else c
    rho, tau = np.broadcast_arrays(np.asarray(rho, float), np.asarray(tau, float))
    safe = np.where(rho < 1e-9, 1e-9, rho)
    t = c * tau
    s = np.zeros_like(safe)
    for ln in _ROOTS:
        s += np.sin(ln * safe) / (ln ** 2 * np.sin(ln)) * np.exp(-ln ** 2 * t)
    return (3.0 * t + safe ** 2 / 2.0 - 0.3 - 2.0 * s / safe) / c


# ------------------------------------------------------------------- samplers
def sample_particle(n, seed=88):
    """``n`` collocation points ``(rho, tau)`` in the particle and the hour,
    more of them early, where the field changes fastest."""
    rng = np.random.default_rng(seed)
    return np.c_[rng.random(n), rng.random(n) ** 1.5]


def surface_points(n=200):
    """``n`` points on the surface, rho = 1, at evenly spaced instants after
    tau = 0. Not AT tau = 0: there the trial function has no gradient and the
    flux condition cannot be met."""
    return np.c_[np.ones(n), np.linspace(0.0, 1.0, n + 1)[1:]]


def mass_quadrature(n_r=41, n_t=20):
    """The mass balance as a quadrature: points ``(rho, tau)`` on ``n_t``
    instants, weights such that the weighted sum over one instant is the mean
    of ``u`` over the sphere, and the value that mean must have, ``3 tau``.
    Returns ``(points, weights, targets)``; the points are ordered instant by
    instant, ``n_r`` to each."""
    rq = np.linspace(0.0, 1.0, n_r)
    tq = np.linspace(0.0, 1.0, n_t + 1)[1:]
    w = np.full(n_r, 1.0 / (n_r - 1))
    w[0] = w[-1] = 0.5 / (n_r - 1)                                    # trapezoidal rule in rho
    pts = np.array([[r, t] for t in tq for r in rq])
    return pts, np.tile(3.0 * rq ** 2 * w, n_t).reshape(-1, 1), 3.0 * tq.reshape(-1, 1)


def surface_readings(seed=101):
    """What a battery management system would have: the surface concentration
    every 100 s, in mol/m^3, with noise of :data:`NOISE`. Synthetic: the exact
    solution, sampled. Returns ``(tau, c_surface)``.

    A real system reads a voltage and gets the surface concentration from it
    through the open-circuit curve; that step is taken as done here."""
    rng = np.random.default_rng(seed)
    tau = np.linspace(0.0, 1.0, 37)
    return tau, concentration(exact(1.0, tau)) + rng.normal(0.0, NOISE, len(tau))


def describe_problem() -> None:
    """Print the particle and the numbers it implies."""
    print(f"  particle         : graphite, radius {R_PARTICLE * 1e6:.2f} um, D_s = {D_SOLID:.1e} m^2/s")
    print(f"  discharge        : {CURRENT:.0f} A (1C) for {T_END:.0f} s")
    print(f"  surface flux     : j = I / (F a_s A L) = {J_SURFACE:.2e} mol/(m^2 s) leaving the particle")
    print(f"  start            : {C_START:.0f} mol/m^3 everywhere ({C_START / C_MAX:.2f} of the maximum {C_MAX:.0f})")
    print(f"  diffusion time   : R^2 / D_s = {R_PARTICLE ** 2 / D_SOLID:.0f} s")
    print(f"  scales           : c_ref = j t_end / R = {C_REF:.0f} mol/m^3;  C = D_s t_end / R^2 = {C_SCALED:.2f}")
    print(f"  mass balance     : the mean concentration falls by 3 c_ref = {3 * C_REF:.0f} mol/m^3 over the hour")


def draw_particle(ax=None):
    """The particle of the single particle model: the electrode it stands for,
    and the particle itself, coloured by the exact concentration at the end
    of the hour, with the lithium leaving through its surface."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(9.0, 3.8))
    # the electrode: a slab of particles, one of which stands for all
    ax.add_patch(plt.Rectangle((0, 0), 2.0, 4.0, fill=False, lw=1.2, ec="0.4"))
    rng = np.random.default_rng(3)
    for cx, cy in zip(rng.uniform(0.25, 1.75, 24), rng.uniform(0.25, 3.75, 24)):
        ax.add_patch(plt.Circle((cx, cy), 0.2, fc="0.85", ec="0.5", lw=0.6))
    ax.add_patch(plt.Circle((1.0, 2.0), 0.24, fc="tab:orange", ec="k", lw=1.0))
    ax.text(1.0, -0.25, f"negative electrode,\n{L_ELECTRODE * 1e6:.0f} µm of graphite", ha="center", va="top", fontsize=10)
    ax.plot([1.24, 4.0], [2.0, 3.9], color="0.5", lw=0.8, ls="--")
    ax.plot([1.24, 4.0], [2.0, 0.1], color="0.5", lw=0.8, ls="--")
    # the particle, in rings of the exact concentration at the end of the hour
    cx, cy, Rd = 6.0, 2.0, 1.9
    c = concentration(exact(np.linspace(0, 1, 41), 1.0))
    cmap, lo, hi = plt.get_cmap("viridis"), c.min(), c.max()
    for k in range(40, 0, -1):
        ax.add_patch(plt.Circle((cx, cy), Rd * k / 40, fc=cmap((c[k] - lo) / (hi - lo)), ec="none"))
    ax.add_patch(plt.Circle((cx, cy), Rd, fill=False, ec="k", lw=1.2))
    for a in np.linspace(0, 2 * np.pi, 9)[:-1]:
        ax.annotate("", xy=(cx + 2.45 * np.cos(a), cy + 2.45 * np.sin(a)), xytext=(cx + 1.95 * np.cos(a), cy + 1.95 * np.sin(a)),
                    arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.2))
    ax.annotate("", xy=(cx + Rd * np.cos(2.4), cy + Rd * np.sin(2.4)), xytext=(cx, cy), arrowprops=dict(arrowstyle="->", color="w", lw=1))
    ax.text(cx, cy - 0.45, f"R = {R_PARTICLE * 1e6:.2f} µm", color="w", fontsize=10, ha="center", va="top")
    ax.text(8.7, 3.4, "lithium leaves\nthrough the surface:\na 1C discharge", color="tab:red", fontsize=10, va="center")
    ax.text(8.7, 1.9, f"surface {c[-1]:.0f} mol/m³\ncentre {c[0]:.0f} mol/m³\nafter the hour", fontsize=10, va="center")
    ax.text(8.7, 0.5, "centre: symmetry", fontsize=10, va="center")
    ax.set_xlim(-0.2, 11.6); ax.set_ylim(-1.2, 4.6); ax.set_aspect("equal"); ax.axis("off")
    return ax.figure
