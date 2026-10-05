r"""Ex_09.2 (hydrogen-channel version) — the physics: hydrogen carried along the
fuel channel of a solid oxide fuel cell, from the CFD side only.

*Deep Learning for Engineering* — MSc, Aalborg University.
Remus Teodorescu (ret@et.aau.dk), with support from Research Assistant
Noman Khan (nomank@energy.aau.dk).

**Reference texts.** Incropera & DeWitt, *Fundamentals of Heat and Mass
Transfer*, ch. 8 (internal flow) and ch. 14 (diffusion mass transfer); Bird,
Stewart & Lightfoot, *Transport Phenomena*, ch. 17–18; Shah & London,
*Laminar Flow Forced Convection in Ducts* (the constant-flux Nusselt number
5.385 for parallel plates, one wall active); Liu, *PINN with Python* (2025).
The code, the problem and the exposition here are original to this course.

## The problem

A gas channel ``L`` long and ``h`` high at 800 degC and 1 atm. Humidified
hydrogen enters at the mean speed ``u``. Along the electrode wall (``z = 0``)
hydrogen is taken up and steam returned at a given, uniform rate, set by the
current density through Faraday's law and by nothing else: ``N = i / 2F`` moles
per square metre and second. The other wall (the interconnect, ``z = h``) is
impermeable. Wanted: the steam fraction ``y`` everywhere, the fuel used, and
the inlet speed that keeps enough hydrogen at the electrode.

    u(z) y_x = D y_zz                    in the channel,  u(z) = 6 u (z/h)(1 - z/h)
    y = y_in                             at x = 0
    -D c y_z = N                         at z = 0   (steam enters the gas from the electrode)
    y_z = 0                              at z = h

Diffusion along the channel is dropped: the Peclet number along it is 125,
so the term is a ten-thousandth of the one across, and dropping it is what
the Graetz problem does. What is left is parabolic in ``x`` - it marches
from the inlet like the heat equation marches in time, and needs no outlet
condition. (Kept, the term would need a fully developed outflow, ``y_xx = 0``;
a zero-slope outlet would be wrong, because the mean keeps rising.)

Three facts from the lecture make this the whole model. The gas is ideal and
the Mach number is 1e-03, so it is incompressible; the exchange at the wall
is one molecule for one, so the molar concentration ``c = p/RT`` and the
velocity field are unchanged by the reaction, and the species is written in
mole fractions; and Re is about 1 with an entrance length of 0.07 mm, so the
velocity is Poiseuille's parabola from the inlet.

## Scaled units

``X = x/L``, ``Z = z/h``, and the rise of the steam fraction over
``Y = N L / (u_ref h c)``, the rise of the mixing-cup mean over the whole
channel at the reference speed (1 m/s): ``theta = (y - y_in) / Y``. With
``s = u / u_ref`` the inlet speed as a free parameter, ``EPS = h/L`` and
``G = u_ref h^2 / (D L)`` (the Peclet number across the channel at the
reference speed, times the aspect ratio),

    6 Z (1 - Z) G s theta_X = theta_ZZ
    theta = 0 at X = 0;   theta_Z = -G at Z = 0;   theta_Z = 0 at Z = 1

## The exact solution

Beyond an entrance region a fraction of a millimetre long, the field is
exactly linear along the channel and a quartic across it:

    theta(X, Z) = X / s + G phi(Z),      phi(Z) = Z^3 - Z^4/2 - Z + 13/35

``X/s`` is the mixing-cup balance, the hand calculation; ``phi`` is what the
hand calculation misses, the profile across the channel, found by integrating
the equation twice in ``Z`` with the two wall conditions and fixing the
constant so that the flow-weighted mean of ``phi`` is zero, which the
integral balance fixes for every ``X``. Its Sherwood number on the hydraulic diameter ``2h`` is
``2 x 35/13 = 5.385``, the published constant for parallel plates with one
wall at constant flux and the other impermeable (Shah & London; Incropera's
table 8.1 gives the same number for heat). :func:`exact` is that field;
:func:`fdm` marches the full problem from the inlet on a grid, entrance included.

Values (C10 rule 5), typical and ASSUMED until cited: a solid oxide cell
(100 mm long, 1 mm channel, 800 degC, 1 atm; D = 8.0e-04 m^2/s for steam in
hydrogen, an estimate); 97 % hydrogen at the inlet; 0.5 A/cm^2; a gas
viscosity of 2.2e-05 Pa s.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

__all__ = [
    "FARADAY", "R_GAS", "T_CELL", "P_CELL", "C_TOT", "L_CHANNEL", "H_CHANNEL", "W_CELL",
    "Y_IN", "D_GAS", "MU_GAS", "M_H2", "M_H2O", "RHO_IN", "I_CELL", "N_WALL", "U_REF",
    "S_MIN", "S_MAX", "EPS", "G", "Y_SCALE", "PE_H", "PE_L", "RE", "ENTRANCE_MM", "SHERWOOD",
    "utilisation", "speed_for_utilisation", "pressure_drop", "poiseuille", "phi", "exact", "exact_wall_outlet",
    "fdm", "sample_channel", "sample_faces", "draw_channel", "draw_cell", "describe_problem", "kelvin_free",
]

# ------------------------------------------------------------------ the data
FARADAY = 96485.0          #: C/mol
R_GAS = 8.314              #: J/(mol K)
T_CELL = 1073.15           #: K, 800 degC
P_CELL = 101325.0          #: Pa
C_TOT = P_CELL / (R_GAS * T_CELL)   #: mol/m³, the gas at 1 atm and 800 degC, 11.36
L_CHANNEL = 0.10           #: m
H_CHANNEL = 1.0e-3         #: m, channel height
W_CELL = 0.10              #: m, the cell's width into the page
Y_IN = 0.03                #: steam fraction at the inlet: 97 % hydrogen, humidified
D_GAS = 8.0e-4             #: m²/s, steam in hydrogen at 800 degC   [estimate]
MU_GAS = 2.2e-5            #: Pa s, the mixture's viscosity at 800 degC   [typical value]
M_H2, M_H2O = 2.016e-3, 18.015e-3   #: kg/mol
RHO_IN = C_TOT * ((1 - Y_IN) * M_H2 + Y_IN * M_H2O)   #: kg/m³ at the inlet, 0.028
I_CELL = 5000.0            #: A/m², 0.5 A/cm², uniform along the electrode   [typical value]
N_WALL = I_CELL / (2 * FARADAY)     #: mol/(m² s), steam returned (hydrogen taken) at the electrode, 0.0259
U_REF = 1.0                #: m/s, the reference inlet speed; s = u / U_REF
S_MIN, S_MAX = 0.3, 1.5    #: the range of inlet speeds the network is trained over, in units of U_REF

EPS = H_CHANNEL / L_CHANNEL                           #: the aspect ratio, 0.01
G = U_REF * H_CHANNEL ** 2 / (D_GAS * L_CHANNEL)      #: Pe across the channel times the aspect ratio, 0.0125
Y_SCALE = N_WALL * L_CHANNEL / (U_REF * H_CHANNEL * C_TOT)   #: the rise of the mixing-cup mean at U_REF over the channel, 0.228
PE_H = U_REF * H_CHANNEL / D_GAS                      #: Peclet across the channel at U_REF, 1.25
PE_L = U_REF * L_CHANNEL / D_GAS                      #: Peclet along the channel at U_REF, 125
RE = RHO_IN * U_REF * H_CHANNEL / MU_GAS              #: Reynolds number at U_REF, about 1.3
ENTRANCE_MM = 0.05 * RE * H_CHANNEL * 1e3             #: mm, the hydrodynamic entrance length at U_REF
SHERWOOD = 2 * 35.0 / 13.0                            #: on the hydraulic diameter 2h: 5.385


def utilisation(s):
    """The share of the inlet hydrogen used over the channel, at the speed s U_REF."""
    return N_WALL * L_CHANNEL / (s * U_REF * H_CHANNEL * C_TOT * (1 - Y_IN))


def speed_for_utilisation(u_target):
    """The inlet speed, in units of U_REF, that gives the utilisation u_target."""
    return N_WALL * L_CHANNEL / (u_target * U_REF * H_CHANNEL * C_TOT * (1 - Y_IN))


def pressure_drop(s):
    """Poiseuille's pressure drop over the channel, Pa, at the speed s U_REF."""
    return 12 * MU_GAS * s * U_REF * L_CHANNEL / H_CHANNEL ** 2


def poiseuille(Z):
    """u(z)/u_mean = 6 Z (1 - Z)."""
    return 6.0 * Z * (1.0 - Z)


def phi(Z):
    """The profile across the channel, in units of G: zero flow-weighted mean,
    slope -1 at the electrode, 0 at the interconnect."""
    Z = np.asarray(Z, dtype=float)
    return Z ** 3 - Z ** 4 / 2 - Z + 13.0 / 35.0


def exact(X, Z, s):
    """The scaled rise theta beyond the entrance: X/s + G phi(Z). Exact to
    rounding for X above a few millimetres over L."""
    return np.asarray(X, dtype=float) / s + G * phi(Z)


def exact_wall_outlet(s):
    """The steam fraction at the electrode at the outlet, and the hydrogen fraction there."""
    y = Y_IN + Y_SCALE * exact(1.0, 0.0, s)
    return y, 1.0 - y


def kelvin_free(theta):
    """Scaled rise back to a steam fraction."""
    return Y_IN + Y_SCALE * np.asarray(theta)


# ----------------------------------------------------------- finite differences
def fdm(nx, nz, s, start=2):
    """The full problem marched from the inlet (FDM): Crank-Nicolson along the
    channel (second order), central differences across, ghost nodes for the
    two walls. The equation is parabolic in X, so x is stepped like time and
    no outlet condition is needed. Returns X, Z (1-D) and theta (nz, nx).

    At the walls the velocity is zero, so a wall row has no march term left:
    it is the wall condition alone, and Crank-Nicolson holds it only as the
    AVERAGE of the old and the new column. The inlet column breaks it (the
    gas enters with zero slope where the electrode asks for -G), and that
    error then flips sign at every step and never decays: a sawtooth on the
    electrode node. So the first ``start`` columns are each taken by two
    fully implicit half steps (Rannacher's start), which satisfy the wall
    condition on every column; Crank-Nicolson keeps it from there, and the
    scheme converges at second order. ``start=0`` is plain Crank-Nicolson."""
    hx, hz = 1.0 / (nx - 1), 1.0 / (nz - 1)
    X, Z = np.linspace(0, 1, nx), np.linspace(0, 1, nz)
    A = sp.diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(nz, nz)).tolil()
    A[0, 1] = 2.0; A[-1, -2] = 2.0                             # both walls by a ghost node
    A = A.tocsc() / hz ** 2
    b = np.zeros(nz); b[0] = 2.0 * G / hz                      # the electrode's flux through its ghost: theta_Z = -G
    M = sp.diags(poiseuille(Z) * G * s).tocsc()                # the velocity in front of theta_X; zero at the walls
    L = spla.splu((M - 0.5 * hx * A).tocsc())
    R = (M + 0.5 * hx * A).tocsc()
    Lb = spla.splu((M - 0.5 * hx * A).tocsc())                 # a fully implicit half step (hx/2), for the start
    th = np.zeros((nz, nx))
    for n in range(1, nx):
        if n <= start:
            y = th[:, n - 1]
            for _ in range(2):
                y = Lb.solve(M @ y + 0.5 * hx * b)             # two implicit half steps: the wall condition holds on this column
            th[:, n] = y
        else:
            th[:, n] = L.solve(R @ th[:, n - 1] + hx * b)      # Crank-Nicolson
    return X, Z, th


# ------------------------------------------------------------------- sampling
def sample_channel(n, seed=88):
    """n points (X, Z, s) over the channel and the range of speeds, Latin
    hypercube; a fifth of them in the first 5 % of the length, where the
    profile develops."""
    rng = np.random.default_rng(seed)
    u = (rng.permuted(np.tile(np.arange(n), (3, 1)), axis=1).T + rng.random((n, 3))) / n
    u[: n // 5, 0] *= 0.05
    u[:, 2] = S_MIN + (S_MAX - S_MIN) * u[:, 2]
    return u


def sample_faces(n, seed=88):
    """Points on the four sides at random speeds: (inlet, electrode, interconnect, outlet), each (m, 3)."""
    rng = np.random.default_rng(seed + 1)
    s = lambda m: S_MIN + (S_MAX - S_MIN) * rng.random(m)
    z = rng.random(n); inlet = np.c_[np.zeros(n), z, s(n)]
    x = rng.random(n); x[: n // 5] *= 0.05
    electrode = np.c_[x, np.zeros(n), s(n)]
    x2 = rng.random(n); x2[: n // 5] *= 0.05
    interconnect = np.c_[x2, np.ones(n), s(n)]
    z2 = rng.random(n // 2); outlet = np.c_[np.ones(n // 2), z2, s(n // 2)]
    return inlet, electrode, interconnect, outlet


# ------------------------------------------------------------------ drawing
def draw_channel(ax):
    """The channel in millimetres, not to scale across: electrode, interconnect, the flow, the exchange."""
    import matplotlib.patches as mpatches
    L, h = L_CHANNEL * 1e3, 10.0                                 # the height drawn ten times too large, and said so
    ax.add_patch(mpatches.Rectangle((0, -4), L, 4, facecolor="0.75", edgecolor="k", lw=1))
    ax.text(L / 2, -2, "electrode: hydrogen taken, steam returned, i = 0.5 A/cm²", ha="center", va="center", fontsize=9)
    ax.add_patch(mpatches.Rectangle((0, h), L, 4, facecolor="0.55", edgecolor="k", lw=1))
    ax.text(L / 2, h + 2, "interconnect: impermeable wall", ha="center", va="center", fontsize=9, color="w")
    ax.add_patch(mpatches.Rectangle((0, 0), L, h, facecolor="#dbe9f6", edgecolor="k", lw=1))
    zz = np.linspace(0, 1, 9)[1:-1]
    for z in zz:
        ax.annotate("", xy=(6 + 14 * poiseuille(z), z * h), xytext=(6, z * h), arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1))
    ax.text(28, h / 2, "Poiseuille's profile, Re ≈ 1", va="center", fontsize=8.5, color="tab:blue")
    for xa in np.linspace(45, L - 5, 6):
        ax.annotate("", xy=(xa, h / 2 + 1.5), xytext=(xa, 0.4), arrowprops=dict(arrowstyle="->", color="tab:orange", lw=1))
        ax.annotate("", xy=(xa + 2, 0.4), xytext=(xa + 2, h / 2 + 1.5), arrowprops=dict(arrowstyle="->", color="tab:green", lw=1))
    ax.text(70, h / 2 + 2.6, "steam up, hydrogen down: one for one", ha="center", fontsize=8.5)
    ax.text(-2, h / 2, f"97 % H₂\nu = 0.3 to 1.5 m/s", ha="right", va="center", fontsize=8.5)
    ax.text(L + 2, h / 2, "outlet", ha="left", va="center", fontsize=8.5)
    ax.annotate("", xy=(0, h + 6), xytext=(L, h + 6), arrowprops=dict(arrowstyle="<->", lw=0.8)); ax.text(L / 2, h + 7, "100 mm", ha="center", va="bottom", fontsize=8.5)
    ax.annotate("", xy=(L + 12, 0), xytext=(L + 12, h), arrowprops=dict(arrowstyle="<->", lw=0.8)); ax.text(L + 14, h / 2, "1 mm", va="center", fontsize=8.5)
    ax.set_xlim(-22, L + 26); ax.set_ylim(-6, h + 11); ax.set_aspect("equal")
    ax.set_xlabel("x  [mm]"); ax.set_yticks([]); ax.set_title("the fuel channel, per metre of width; the height drawn ten times too large", fontsize=10)


def draw_cell():
    """What happens behind the electrode wall, in both directions of the same cell.

    A cross-section through a solid oxide cell, not to scale: the fuel channel
    of this notebook on top, then the porous fuel electrode, the electrolyte
    (it conducts oxide ions O2- and nothing else), the air electrode and the
    air channel. Left, the fuel cell (SOFC): oxide ions cross the electrolyte
    upwards and burn the hydrogen that diffuses into the fuel electrode, the
    steam goes back into the channel, the electrons go round the external
    circuit through a load. Right, the electrolyser (SOEC): a power supply
    drives the same reactions backwards - steam is split at the fuel
    electrode, the hydrogen goes back into the channel, the oxide ions cross
    downwards and leave the air electrode as oxygen."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))
    layers = [  # (bottom, top, colour, name)
        (0.0, 1.3, "#eaf3e6", "air channel"),
        (1.3, 2.1, "#c9d9b9", "air electrode"),
        (2.1, 2.5, "#e8d6a8", "electrolyte: conducts O$^{2-}$ only"),
        (2.5, 3.5, "#b9c3cf", "fuel electrode (porous nickel and ceramic)"),
        (3.5, 5.5, "#dbe9f6", "fuel channel: this notebook"),
        (5.5, 6.1, "0.55", "interconnect"),
    ]

    def arrow(ax, x0, y0, x1, y1, col, text=None, tx=None, ty=None):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=2.0, mutation_scale=16))
        if text:
            ax.text(tx if tx is not None else x1 + 0.15, ty if ty is not None else (y0 + y1) / 2,
                    text, color=col, fontsize=10.5, va="center", fontweight="bold")

    for ax, mode in zip(axes, ("SOFC", "SOEC")):
        fc = mode == "SOFC"
        for y0, y1, col, name in layers:
            ax.add_patch(mpatches.Rectangle((0, y0), 9.0, y1 - y0, facecolor=col, edgecolor="k", lw=0.8))
            ax.text(8.9, y1 - 0.2 if "channel" in name else (y0 + y1) / 2, name, ha="right", va="center", fontsize=8.5,
                    color="w" if name == "interconnect" else "0.25")
        # the gas along the channel, and the air
        ax.annotate("", xy=(2.2, 5.05), xytext=(0.2, 5.05), arrowprops=dict(arrowstyle="-|>", color="tab:blue", lw=1.5))
        ax.text(0.2, 4.72, "97 % H$_2$, 3 % H$_2$O in" if fc else "90 % H$_2$O, 10 % H$_2$ in", fontsize=9.5, color="tab:blue")
        ax.annotate("", xy=(2.2, 0.85), xytext=(0.2, 0.85), arrowprops=dict(arrowstyle="-|>", color="tab:green", lw=1.5))
        ax.text(0.2, 0.42, "air in: O$_2$ taken" if fc else "air in: O$_2$ added", fontsize=9.5, color="tab:green")
        # the exchange at the fuel electrode
        if fc:
            arrow(ax, 3.0, 4.3, 3.0, 3.05, "tab:green", "H$_2$", tx=2.35, ty=3.95)
            arrow(ax, 4.1, 3.05, 4.1, 4.3, "tab:orange", "H$_2$O", tx=4.25, ty=3.95)
            arrow(ax, 3.55, 1.7, 3.55, 2.95, "tab:red", "O$^{2-}$", tx=3.7, ty=2.3)
            arrow(ax, 3.55, 0.95, 3.55, 1.6, "tab:green", "O$_2$", tx=3.7, ty=1.1)
            ax.text(5.0, 3.62, "at the fuel electrode:\nH$_2$ + O$^{2-}$ → H$_2$O + 2e$^-$", fontsize=10.5, va="bottom")
            ax.text(5.0, 0.1, "at the air electrode:\n½O$_2$ + 2e$^-$ → O$^{2-}$", fontsize=10.5, va="bottom")
        else:
            arrow(ax, 3.0, 4.3, 3.0, 3.05, "tab:orange", "H$_2$O", tx=2.15, ty=3.95)
            arrow(ax, 4.1, 3.05, 4.1, 4.3, "tab:green", "H$_2$", tx=4.25, ty=3.95)
            arrow(ax, 3.55, 2.95, 3.55, 1.7, "tab:red", "O$^{2-}$", tx=3.7, ty=2.3)
            arrow(ax, 3.55, 1.6, 3.55, 0.95, "tab:green", "O$_2$", tx=3.7, ty=1.1)
            ax.text(5.0, 3.62, "at the fuel electrode:\nH$_2$O + 2e$^-$ → H$_2$ + O$^{2-}$", fontsize=10.5, va="bottom")
            ax.text(5.0, 0.1, "at the air electrode:\nO$^{2-}$ → ½O$_2$ + 2e$^-$", fontsize=10.5, va="bottom")
        # the external circuit: electrons from one electrode to the other
        ax.plot([9.0, 9.9, 9.9, 9.0], [3.0, 3.0, 1.7, 1.7], color="k", lw=1.2)
        ax.add_patch(mpatches.Rectangle((9.6, 2.05), 0.6, 0.6, facecolor="w", edgecolor="k", lw=1.2))
        ax.text(10.35, 2.35, "load" if fc else "power\nsupply", fontsize=9.5, va="center")
        if fc:
            arrow(ax, 9.9, 1.95, 9.9, 1.75, "k")
            ax.text(10.05, 3.15, "e$^-$ out of the fuel electrode", fontsize=9)
        else:
            arrow(ax, 9.9, 2.75, 9.9, 2.95, "k")
            ax.text(10.05, 3.15, "e$^-$ into the fuel electrode", fontsize=9)
        ax.set_title(("fuel cell (SOFC): hydrogen burnt, steam returned, power out" if fc else
                      "electrolyser (SOEC): steam split, hydrogen returned, power in"), fontsize=11)
        ax.set_xlim(-0.1, 12.2); ax.set_ylim(-0.1, 6.2); ax.set_axis_off()
    fig.suptitle("behind the electrode wall: one molecule out of the channel, one back, for every two electrons "
                 "(layers not to scale)", fontsize=10.5)
    fig.tight_layout()
    return fig


def draw_coordinates():
    """The channel's coordinates and the three velocity components: the channel
    in 3-D with x along it, z across the gap and the width into the page, and
    the gap seen from the side. At one parcel the three components are drawn:
    u along the channel (kept), v across the gap and w across the width (both
    zero: the walls and the volume condition kill v, and a channel a hundred
    times wider than it is high has nothing varying across its width)."""
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(14, 4.6))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    L, W, H = 3.0, 1.6, 0.9                                     # the height drawn far too large, as in the channel figure
    for z in (0, H):
        ax.plot([0, L, L, 0, 0], [0, 0, W, W, 0], [z] * 5, color="0.6", lw=1.2 if z == 0 else 0.8)
    for x, y in ((0, 0), (L, 0), (L, W), (0, W)):
        ax.plot([x, x], [y, y], [0, H], color="0.6", lw=0.8)
    ax.text(L + 0.1, W + 0.1, -0.02, "electrode (z = 0)", fontsize=9, color="0.3")
    ax.text(L + 0.1, W + 0.1, H + 0.08, "interconnect (z = h)", fontsize=9, color="0.3")
    ax.quiver(-0.3, 0, 0, 1.0, 0, 0, color="k", lw=1.2, arrow_length_ratio=0.15); ax.text(0.8, -0.15, 0.0, "x  along", fontsize=10)
    ax.quiver(0, 0, 0, 0, 0, 1.0, color="k", lw=1.2, arrow_length_ratio=0.15); ax.text(0, 0.05, 1.1, "z  across the gap", fontsize=10)
    ax.quiver(0, 0, 0, 0, 0.8, 0, color="k", lw=1.2, arrow_length_ratio=0.15); ax.text(-0.9, 0.95, -0.12, "across the width (100 mm)", fontsize=10)
    x0, y0, z0 = 2.1, 0.5, 0.55                                 # one parcel
    ax.plot([x0], [y0], [z0], "o", color="tab:orange", ms=7)
    ax.quiver(x0, y0, z0, 0.8, 0, 0, color="tab:blue", lw=2, arrow_length_ratio=0.2); ax.text(x0 + 0.85, y0, z0 + 0.05, "u", color="tab:blue", fontsize=13)
    ax.quiver(x0, y0, z0, 0, 0, 0.35, color="tab:green", lw=2, arrow_length_ratio=0.3); ax.text(x0, y0, z0 + 0.45, "v", color="tab:green", fontsize=13)
    ax.quiver(x0, y0, z0, 0, 0.4, 0, color="tab:purple", lw=2, arrow_length_ratio=0.3); ax.text(x0, y0 + 0.5, z0, "w", color="tab:purple", fontsize=13)
    ax.set_xlim(-0.3, L + 0.3); ax.set_ylim(-0.3, W + 0.3); ax.set_zlim(-0.1, 1.2)
    ax.set_box_aspect((L + 0.6, W + 0.6, 1.3), zoom=1.05); ax.view_init(24, -58); ax.axis("off")
    ax.set_title("the channel's coordinates: x along, z across the 1 mm gap, the third direction across its 100 mm width", fontsize=10)

    ax = fig.add_subplot(1, 2, 2)
    ax.plot([0, 3], [0, 0], color="k", lw=3); ax.plot([0, 3], [1, 1], color="k", lw=3)
    ax.text(1.5, -0.12, "electrode, u = 0", ha="center", va="top", fontsize=9.5)
    ax.text(1.5, 1.06, "interconnect, u = 0", ha="center", va="bottom", fontsize=9.5)
    zz = np.linspace(0, 1, 11)[1:-1]
    for z in zz:
        ax.annotate("", xy=(0.5 + 1.2 * poiseuille(z) / 1.5, z), xytext=(0.5, z), arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1))
    zc = np.linspace(0, 1, 100); ax.plot(0.5 + 1.2 * poiseuille(zc) / 1.5, zc, color="tab:blue", lw=1.5)
    ax.plot(2.3, 0.55, "o", color="tab:orange", ms=9); ax.plot(2.3, 0.55, "o", color="tab:blue", ms=4)
    ax.annotate("", xy=(2.3, 0.85), xytext=(2.3, 0.55), arrowprops=dict(arrowstyle="->", color="tab:green", lw=2)); ax.text(2.36, 0.78, "v", color="tab:green", fontsize=13)
    ax.text(2.36, 0.46, "u  (out of the page: w)", color="tab:blue", fontsize=10)
    for yy, col, txt in ((1.15, "tab:blue", "u  along the channel:  kept, Poiseuille's parabola"),
                         (0.80, "tab:green", "v  across the gap:  zero - no flow through either wall,\n     and the volume condition with u unchanging along"),
                         (0.35, "tab:purple", "w  across the width:  zero - a hundred times wider than\n     high, nothing varies across it except at the edges"),
                         (-0.15, "k", "so the velocity is u(z), the same at every x: the flow is\n     solved; the notebook's unknown is the steam fraction")):
        ax.text(3.3, yy, txt, color=col, fontsize=10, va="center")
    ax.axis("off"); ax.set_xlim(-0.1, 9.0); ax.set_ylim(-0.45, 1.45)
    ax.set_title("the gap seen from the side: the three components at one parcel", fontsize=10)
    plt.tight_layout()
    return fig


def describe_problem():
    print(f"  channel   : {L_CHANNEL * 1e3:.0f} mm long, {H_CHANNEL * 1e3:.0f} mm high, {T_CELL - 273.15:.0f} °C, 1 atm: c = {C_TOT:.2f} mol/m³")
    print(f"  gas       : {100 * (1 - Y_IN):.0f} % hydrogen at the inlet, ρ = {RHO_IN:.3f} kg/m³, μ = {MU_GAS:.1e} Pa s, D = {D_GAS:.1e} m²/s (estimates)")
    print(f"  electrode : {I_CELL / 1e4:.1f} A/cm² uniform: N = i/2F = {N_WALL:.4f} mol/(m² s) of hydrogen taken and steam returned")
    print(f"  at 1 m/s  : Re = {RE:.2f}, entrance length {ENTRANCE_MM:.2f} mm; Pe = {PE_H:.2f} across, {PE_L:.0f} along; "
          f"mixing length across h²u/D = {H_CHANNEL ** 2 * U_REF / D_GAS * 1e3:.2f} mm; Δp = {pressure_drop(1.0):.2f} Pa")
    print(f"  scales    : the mixing-cup steam fraction rises by Y = {Y_SCALE:.3f} over the channel at 1 m/s; across the channel the "
          f"profile is G phi, G = {G:.4f}; utilisation {100 * utilisation(1.0):.1f} % at 1 m/s, {100 * utilisation(S_MIN):.1f} % at {S_MIN:.1f} m/s")
