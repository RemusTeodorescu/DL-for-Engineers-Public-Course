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
    "surface_readings", "describe_problem", "draw_particle", "draw_sandwich",
    "animate_discharge",
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
    # the electrode: a slab of identical, evenly spaced particles - the model's
    # own assumption - one of which stands for all
    ax.add_patch(plt.Rectangle((0, 0), 2.0, 4.0, fill=False, lw=1.2, ec="0.4"))
    for i, cx in enumerate((0.4, 1.0, 1.6)):
        for k, cy in enumerate(np.linspace(0.35, 3.65, 7)):
            if i == 1 and k == 3:
                continue                                     # the representative particle goes here
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


def draw_sandwich(ax=None):
    """The whole cell and the mesh of the full (Doyle-Fuller-Newman) model:
    collector, electrode, separator, electrode, collector, the electrolyte
    filling the pores, and one representative particle at every node of the
    thickness. The cell is BYD's 4680 cylindrical LFP cell (15 Ah); the LFP
    particle radius, 50 nm, is PyBaMM's Prada2013 value. Every particle is
    drawn at ONE display size so the drawing stays uniform - an LFP particle
    is a hundred times smaller than section 1's graphite particle."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(10.5, 3.6))
    x_an, x_sep, x_cat = 34.0, 25.0, 80.0                    # drawn proportions of the stack
    x0, x1, x2, x3 = 0.0, x_an, x_an + x_sep, x_an + x_sep + x_cat
    H = 30.0
    # the electrolyte is the continuous phase: it fills both electrodes and the separator
    ax.add_patch(plt.Rectangle((x0, 0), x3, H, fc="#dceaf5", ec="none"))
    ax.add_patch(plt.Rectangle((x1, 0), x_sep, H, fc="#c3d6e8", ec="none", hatch="///"))
    # the collectors
    ax.add_patch(plt.Rectangle((-10, 0), 10, H, fc="#c88a5a", ec="0.3", lw=0.8))
    ax.add_patch(plt.Rectangle((x3, 0), 10, H, fc="0.75", ec="0.3", lw=0.8))
    ax.text(-5, H / 2, "Cu", rotation=90, ha="center", va="center", fontsize=9)
    ax.text(x3 + 5, H / 2, "Al", rotation=90, ha="center", va="center", fontsize=9)
    for x in (x0, x1, x2, x3):
        ax.plot([x, x], [0, H], color="0.3", lw=0.8)
    # the mesh: dashed node boundaries, one representative particle per node,
    # every particle at the same display size (rings: each carries its own radial grid)
    for k in range(1, 3):                                    # 3 nodes in the graphite
        ax.plot([x0 + k * x_an / 3] * 2, [0, H], color="0.45", lw=0.7, ls="--")
    for k in range(1, 5):                                    # 5 nodes in the LFP
        ax.plot([x2 + k * x_cat / 5] * 2, [0, H], color="0.45", lw=0.7, ls="--")
    def particle(cx, shades):
        for rr, fc in zip((4.5, 3.0, 1.5), shades):
            ax.add_patch(plt.Circle((cx, H / 2), rr, fc=fc, ec="0.4", lw=0.6))
    for k in range(3):                                       # graphite, grey as in section 1's drawing
        particle(x0 + (2 * k + 1) * x_an / 6, ("0.80", "0.86", "0.92"))
    for k in range(5):                                       # LFP, orange
        particle(x2 + (2 * k + 1) * x_cat / 10, ("#f5b16a", "#f8c68f", "#fbdbb5"))
    # where the lithium goes on discharge
    ax.annotate("", xy=(x2 + 10, H + 6), xytext=(x1 - 10, H + 6), arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1.2))
    ax.text((x1 + x2) / 2, H + 8, "Li$^+$ in the electrolyte", color="tab:blue", fontsize=9, ha="center", va="bottom")
    ax.annotate("", xy=(-12, H + 6), xytext=(16, H + 6), arrowprops=dict(arrowstyle="->", color="0.35", lw=1.0))
    ax.text(2, H + 8, "e$^-$ through the solid", fontsize=9, ha="center", va="bottom")
    ax.text(x_an / 2, -2, "graphite", fontsize=9, ha="center", va="top")
    ax.text((x1 + x2) / 2, -2, "separator", fontsize=9, ha="center", va="top")
    ax.text(x2 + x_cat / 2, -2, "LFP, R = 50 nm", fontsize=9, ha="center", va="top")
    ax.text((x0 + x3) / 2, -8, "particles drawn at one size: an LFP particle is a hundred times\n"
            "smaller than the graphite particle of section 1", fontsize=9, ha="center", va="top", style="italic")
    ax.set_xlim(-14, 156); ax.set_ylim(-17, 42); ax.set_aspect("equal"); ax.axis("off")
    return ax.figure


# ------------------------------------------- the second electrode, for the film
R_LFP = 5.0e-8           #: m, radius of an LFP particle (PyBaMM, Prada 2013)
D_LFP = 5.9e-18          #: m^2/s, diffusivity of lithium in LFP (PyBaMM, Prada 2013)
C_MAX_LFP = 22806.0      #: mol/m^3, the most lithium the LFP can hold (PyBaMM, Prada 2013)
CP_SCALED = D_LFP * T_END / R_LFP ** 2                               #: the LFP scaled diffusivity, 8.50
STO_P0, STO_P1 = 0.05, 0.95   #: the LFP lithiation window over the discharge (chosen)


def ocp_graphite(sto):
    """Open-circuit potential of the LG M50 graphite against lithiation,
    the fit of Chen et al. (2020), as distributed with PyBaMM."""
    sto = np.maximum(np.asarray(sto, float), 1e-4)
    return (1.9793 * np.exp(-39.3631 * sto) + 0.2482
            - 0.0909 * np.tanh(29.8538 * (sto - 0.1234))
            - 0.04478 * np.tanh(14.9159 * (sto - 0.2769))
            - 0.0205 * np.tanh(30.4444 * (sto - 0.6103)))


def ocp_lfp(sto):
    """Open-circuit potential of LFP against lithiation, the fit of
    Afshar et al. (2017), as distributed with PyBaMM's Prada 2013 set."""
    sto = np.asarray(sto, float)
    return 3.4077 - 0.020269 * sto + 0.5 * np.exp(-150 * sto) - 0.9 * np.exp(-30 * (1 - sto))


def animate_discharge(crate=1.0, seconds=10.0, fps=24):
    """The discharge at ``crate`` C played in ``seconds``: both electrodes of
    an LFP cell - this set's graphite emptying while an LFP particle fills -
    on ONE lithiation colour scale (0 to 1, the same at every rate), the two
    states of charge falling, and the terminal voltage read from the two
    open-circuit curves at the two surfaces (Butler-Volmer overpotentials and
    ohmic drops left out). The flux scales with the rate and the window is
    ``T_END / crate``, so the same charge moves; when the voltage dives to
    the 2.0 V cut-off before the bulk is drained (above about 1.25C here,
    because the graphite surface runs empty ahead of the mean), the film
    stops there and says how much charge was left undelivered. Returns a
    ``matplotlib.animation.FuncAnimation``; in a notebook, show it with
    ``HTML(anim.to_html5_video())``."""
    import matplotlib.pyplot as plt
    from matplotlib import animation, cm, colors
    n_r, frames = 41, int(round(seconds * fps))
    rho = np.linspace(0.0, 1.0, n_r)
    taus = np.linspace(0.0, 1.0, frames)                     # time over the window T_END / crate
    Un = np.array([exact(rho, t, c=C_SCALED / crate) for t in taus])    # the graphite field
    Up = np.array([exact(rho, t, c=CP_SCALED / crate) for t in taus])   # the LFP field (lithium coming IN)
    sto_n = (C_START - C_REF * Un) / C_MAX                   # lithiation of the graphite, falling
    sto_p = STO_P0 + (STO_P1 - STO_P0) * Up / 3.0            # lithiation of the LFP, rising
    volt = ocp_lfp(sto_p[:, -1]) - ocp_graphite(sto_n[:, -1])   # the voltage, from the two surfaces
    low = np.nonzero((volt <= 2.0) | (sto_n[:, -1] <= 1e-3))[0]  # the 2.0 V cut-off: the graphite surface is nearly empty
    cut = int(low[0]) if len(low) else None
    if cut is not None:                                      # stop at the cut-off, hold the last frame a moment
        taus, sto_n, sto_p, volt = taus[: cut + 1], sto_n[: cut + 1], sto_p[: cut + 1], volt[: cut + 1]
        frames = len(taus) + int(round(1.5 * fps))
    t_s = taus * T_END / crate                               # time in seconds
    norm = colors.Normalize(0.0, 1.0)                        # one lithiation scale for both particles
    cmap = plt.get_cmap("viridis")
    fig, (axL, axV) = plt.subplots(1, 2, figsize=(11.2, 4.4), gridspec_kw=dict(width_ratios=[1.55, 1.0]))
    rings_n, rings_p = [], []
    for cx, rings in ((2.0, rings_n), (6.4, rings_p)):       # the two particles, in rings, surface drawn first
        for k in range(n_r - 1, 0, -1):
            rings.append(plt.Circle((cx, 4.6), 1.9 * k / (n_r - 1), ec="none"))
            axL.add_patch(rings[-1])
        axL.add_patch(plt.Circle((cx, 4.6), 1.9, fill=False, ec="k", lw=1.2))
    axL.text(2.0, 2.45, "graphite - emptying", fontsize=10, ha="center", va="top")
    axL.text(6.4, 2.45, "LFP - filling", fontsize=10, ha="center", va="top")
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=axL, fraction=0.045, pad=0.02,
                 label="lithiation  c / c$_{max}$")
    bars, pcts = [], []
    for y, colour, name in ((1.15, "tab:blue", "SOC, bulk"), (0.30, "tab:orange", "SOC, surface")):
        axL.text(0.15, y + 0.21, name, fontsize=9, va="center")
        bars.append(axL.add_patch(plt.Rectangle((2.7, y), 0.0, 0.42, fc=colour)))
        axL.add_patch(plt.Rectangle((2.7, y), 4.0, 0.42, fill=False, ec="0.4", lw=0.8))
        pcts.append(axL.text(6.85, y + 0.21, "", va="center", fontsize=10))
    axL.text(0.15, 7.0, f"a {crate:g}C discharge: {T_END / crate:.0f} s played in {seconds:.0f} s",
             fontsize=10, va="bottom")
    t_cut = axL.text(0.15, -0.35, "", fontsize=10, va="top", color="tab:red")
    axL.set_xlim(-0.1, 8.8); axL.set_ylim(-1.3, 7.6); axL.set_aspect("equal"); axL.axis("off")
    # the voltage panel: the whole curve in grey, the film drawing it in
    axV.plot(t_s, volt, color="0.8", lw=1.2)
    v_line, = axV.plot([], [], color="tab:green", lw=1.8)
    v_dot, = axV.plot([], [], "o", color="tab:green", ms=5)
    v_text = axV.text(0.03, 0.06, "", transform=axV.transAxes, fontsize=10)
    axV.set_xlabel("t  [s]"); axV.set_ylabel("cell voltage  [V]")
    axV.set_xlim(0, T_END / crate); axV.set_ylim(1.9, 3.5)
    fig.tight_layout()

    def frame(i):
        i = min(i, len(taus) - 1)                            # past the cut-off the last frame is held
        for k, rn, rp in zip(range(n_r - 1, 0, -1), rings_n, rings_p):
            rn.set_facecolor(cmap(norm(max(sto_n[i, k], 0.0))))
            rp.set_facecolor(cmap(norm(min(sto_p[i, k], 1.0))))
        for soc, bar, txt in zip((1.0 - taus[i], 1.0 - Un[i, -1] / 3.0), bars, pcts):
            bar.set_width(4.0 * max(soc, 0.0))
            txt.set_text(f"{100 * soc:.0f} %")
        v_line.set_data(t_s[: i + 1], volt[: i + 1])
        v_dot.set_data([t_s[i]], [volt[i]])
        v_text.set_text(f"t = {t_s[i]:.0f} s,  V = {volt[i]:.2f} V")
        if cut is not None and i == len(taus) - 1:
            t_cut.set_text(f"cut-off at 2.0 V: the graphite surface is nearly empty -\n{100 * (1.0 - taus[i]):.0f} % of the charge undelivered")
        return rings_n + rings_p + bars + pcts + [v_line, v_dot, v_text, t_cut]

    anim = animation.FuncAnimation(fig, frame, frames=frames, interval=1000.0 / fps, blit=False)
    plt.close(fig)                                           # the animation carries the figure; no still copy
    return anim
