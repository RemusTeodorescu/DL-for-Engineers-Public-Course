r"""Ex_10.2 — the problem: steam along the gas channel of a solid oxide electrolysis cell.

*Deep Learning for Engineering* — MSc, Aalborg University.
Remus Teodorescu (ret@et.aau.dk), with support from Research Assistant
Noman Khan (nomank@energy.aau.dk).

**Reference texts.** Liu, *PINN with Python: An Introduction* (2025); Raissi,
Perdikaris & Karniadakis, *Physics-informed neural networks*, J. Comput. Phys.
**378** (2019) 686–707. These are the works to read for the theory. The code,
the problem and the exposition here are original to this course.

## The problem

A solid oxide cell at 800 °C, run as an electrolyser at a fixed cell voltage
``V``. Steam enters a gas channel 100 mm long and 1 mm high (Ex_09.2's channel) and is split into
hydrogen as it flows along the electrode. With ``y`` the steam fraction:

    u y_x = D y_xx - i(y) / (2 F h c_tot)        along the channel
    i(y)  = (V - E(y)) / ASR                     the local current density
    E(y)  = E_0 + (R T / 2F) ln((1 - y) sqrt(p_O2) / y)      the Nernst potential
    y = 0.90 at the inlet,  y_x = 0 at the outlet

The current is not an input. It is whatever the cell voltage drives against
the local Nernst potential, and that potential rises as the steam is used up:
the cell starves towards the outlet. In scaled units, ``x`` over the channel
length,

    y_x = y_xx / Pe - K (V - E(y)),    Pe = u L / D = 125,   K = L / (u h c_tot 2F ASR) = 1.83 per volt

## Which numbers are measured and which are not

The thermodynamics is standard: the Nernst potential, Faraday's law, the
thermoneutral voltage of 1.29 V. The cell's resistance (``ASR``, 0.25 ohm cm²),
the gas diffusivity (8e-4 m²/s) and the prices of section 6 are **estimates
chosen for the exercise**, of the right order and not measured on any cell.
The area-specific resistance stands for all three losses of L10.2's
polarisation curve at once, which is a linear model of a curve that is not
linear. There is no community reference code for solid oxide cells as PyBaMM
is for batteries; a result that depends on these numbers must say so.

## What is in here

The data; the Nernst potential, written so that it takes NumPy arrays and
torch tensors alike; the reference, a boundary-value solver; the numbers an
engineer reads off a solution; and the samplers. The finite-volume solver and
the network are written in the notebook.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "FARADAY", "R_GAS", "T_CELL", "P_O2", "Y_IN", "L_CHANNEL", "H_CHANNEL", "W_CELL", "U_GAS",
    "D_GAS", "ASR", "E_0", "C_TOT", "PE", "K_REACT", "V_LOW", "V_HIGH", "V_TN",
    "PRICE_H2", "PRICE_EL", "M_H2",
    "nernst", "voltage", "reference", "cell_numbers", "value_per_hour",
    "sample_channel", "outlet_points", "describe_problem", "draw_cell",
]

# ----------------------------------------------------------------- the data
FARADAY = 96485.0        #: C/mol
R_GAS = 8.314            #: J/(mol K)
T_CELL = 1073.15         #: K, 800 degC
P_O2 = 0.21              #: atm, oxygen on the air side
Y_IN = 0.90              #: steam fraction at the inlet; the rest is hydrogen
L_CHANNEL = 0.10         #: m
H_CHANNEL = 1.0e-3       #: m, height of the gas channel
W_CELL = 0.10            #: m, width of the cell: 100 x 100 mm of electrode
U_GAS = 1.0              #: m/s, gas speed in the channel
D_GAS = 8.0e-4           #: m^2/s, steam in hydrogen at 800 degC   [estimate]
ASR = 0.25e-4            #: ohm m^2 (0.25 ohm cm^2), all losses in one resistance   [estimate]
M_H2 = 2.016e-3          #: kg/mol

E_0 = 1.253 - 2.4516e-4 * T_CELL              #: V, standard potential at 800 degC: 0.99
C_TOT = 101325.0 / (R_GAS * T_CELL)           #: mol/m^3, gas at 1 atm and 800 degC
PE = U_GAS * L_CHANNEL / D_GAS                #: the Peclet number, 125
K_REACT = L_CHANNEL / (U_GAS * H_CHANNEL * C_TOT * 2 * FARADAY * ASR)   #: 1/V, 1.83
V_TN = 248000.0 / (2 * FARADAY)               #: V, the thermoneutral voltage, 1.29

#: The range of cell voltage the network is trained over.
V_LOW, V_HIGH = 1.10, 1.40

#: The prices of section 6.   [chosen for the exercise]
PRICE_H2 = 3.00          #: euro per kg of hydrogen
PRICE_EL = 0.065         #: euro per kWh of electricity


def voltage(v):
    """The cell voltage for a scaled voltage ``v`` in [0, 1]."""
    return V_LOW + (V_HIGH - V_LOW) * v


def nernst(y):
    """The Nernst potential at steam fraction ``y``, in volts. NumPy in gives
    NumPy out and a tensor in gives a tensor out. ``y`` is held inside
    (1e-4, 1 - 1e-4), so that a network's first guesses cannot reach the
    logarithm's ends."""
    try:
        import torch
        if torch.is_tensor(y):
            y = torch.clamp(y, 1e-4, 1.0 - 1e-4)
            return E_0 + R_GAS * T_CELL / (2 * FARADAY) * torch.log((1.0 - y) * P_O2 ** 0.5 / y)
    except ImportError:                              # pragma: no cover
        pass
    y = np.clip(y, 1e-4, 1.0 - 1e-4)
    return E_0 + R_GAS * T_CELL / (2 * FARADAY) * np.log((1.0 - y) * P_O2 ** 0.5 / y)


# ------------------------------------------------------------- the reference
def reference(V, tol=1e-10):
    """The reference solution at cell voltage ``V``: SciPy's boundary-value
    solver, a collocation method on a mesh it refines itself until the
    residual is below ``tol``. Returns a function ``y(x)`` for ``x`` in [0, 1]
    (a NumPy array in, an array out)."""
    from scipy.integrate import solve_bvp
    x = np.linspace(0.0, 1.0, 2001)
    f = lambda x, s: np.vstack([s[1], PE * (s[1] + K_REACT * (V - nernst(s[0])))])
    bc = lambda a, b: np.array([a[0] - Y_IN, b[1]])
    guess = np.vstack([Y_IN - 0.5 * x, -0.5 * np.ones_like(x)])
    sol = solve_bvp(f, bc, x, guess, tol=tol, max_nodes=200000)
    assert sol.success, sol.message
    return lambda xx: sol.sol(np.asarray(xx, float))[0]


def cell_numbers(x, y, V):
    """What an engineer reads off a solution ``y`` on points ``x`` (scaled,
    increasing) at cell voltage ``V``. Returns a dict: the steam fraction at
    the outlet, the share of the steam used, the mean current density in
    A/cm², the hydrogen made in g/h, and the electrical power in W."""
    i = (V - nernst(y)) / ASR                                   # A/m^2, the local current density
    i_mean = float(np.sum(0.5 * (i[1:] + i[:-1]) * np.diff(x)) / (x[-1] - x[0]))
    current = i_mean * L_CHANNEL * W_CELL                       # A
    return dict(outlet=float(y[-1]), used=float(1.0 - y[-1] / Y_IN), i_mean=i_mean / 1e4,
                hydrogen=current / (2 * FARADAY) * M_H2 * 1000 * 3600, power=float(V * current))


def value_per_hour(hydrogen_g_h, power_w):
    """What an hour of operation is worth, in euro cents: the hydrogen made,
    at :data:`PRICE_H2`, less the electricity used, at :data:`PRICE_EL`."""
    return 100.0 * (PRICE_H2 * hydrogen_g_h / 1000.0 - PRICE_EL * power_w / 1000.0)


# ------------------------------------------------------------------- samplers
def sample_channel(n, seed=88):
    """``n`` collocation points ``(x, v)``: a position along the channel and a
    scaled cell voltage, both uniform in [0, 1]."""
    rng = np.random.default_rng(seed)
    return np.c_[rng.random(n), rng.random(n)]


def outlet_points(n=100):
    """``n`` points on the outlet, x = 1, at evenly spaced voltages."""
    return np.c_[np.ones(n), np.linspace(0.0, 1.0, n)]


def describe_problem() -> None:
    """Print the cell and the numbers it implies."""
    print(f"  cell             : solid oxide, electrolysis, {T_CELL - 273.15:.0f} degC, {L_CHANNEL * 1e3:.0f} x {W_CELL * 1e3:.0f} mm of electrode")
    print(f"  gas channel      : {H_CHANNEL * 1e3:.0f} mm high, gas at {U_GAS:.1f} m/s, {100 * Y_IN:.0f} % steam at the inlet")
    print(f"  resistance       : ASR = {ASR * 1e4:.2f} ohm cm^2, all losses in one number   [estimate]")
    print(f"  Nernst potential : {nernst(np.array(Y_IN)):.2f} V at the inlet composition; standard potential {E_0:.2f} V")
    print(f"  thermoneutral    : {V_TN:.2f} V")
    print(f"  scaled numbers   : Pe = u L / D = {PE:.0f};  K = L / (u h c 2F ASR) = {K_REACT:.2f} per volt")
    print(f"  steam supplied   : {U_GAS * H_CHANNEL * W_CELL * C_TOT * Y_IN * 1000:.2f} mmol/s; using all of it would take "
          f"{U_GAS * H_CHANNEL * W_CELL * C_TOT * Y_IN * 2 * FARADAY:.0f} A")


def draw_soec(ax=None):
    """The whole electrolysis cell in cross-section, not to scale, as Ex_09.2
    drew it: the hydrogen (fuel) channel of this notebook on top, the porous
    fuel electrode, the electrolyte (it conducts oxide ions O2- and nothing
    else), the air electrode and the air channel. A power supply drives
    electrons into the fuel electrode; steam is split there, the hydrogen goes
    back into the channel, the oxide ions cross the electrolyte downwards and
    leave the air electrode as oxygen."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    if ax is None:
        _, ax = plt.subplots(figsize=(10.5, 5.6))
    layers = [  # (bottom, top, colour, name)
        (0.0, 1.3, "#eaf3e6", "air channel"),
        (1.3, 2.1, "#c9d9b9", "air electrode (oxygen side)"),
        (2.1, 2.5, "#e8d6a8", "electrolyte: conducts O$^{2-}$ only"),
        (2.5, 3.5, "#b9c3cf", "fuel electrode (porous nickel and ceramic)"),
        (3.5, 5.5, "#dbe9f6", "hydrogen (fuel) channel: this notebook"),
        (5.5, 6.1, "0.55", "interconnect"),
    ]

    def arrow(x0, y0, x1, y1, col, text=None, tx=None, ty=None):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=2.0, mutation_scale=16))
        if text:
            ax.text(tx, ty, text, color=col, fontsize=10.5, va="center", fontweight="bold")

    for y0, y1, col, name in layers:
        ax.add_patch(mpatches.Rectangle((0, y0), 9.0, y1 - y0, facecolor=col, edgecolor="k", lw=0.8))
        ax.text(8.9, y1 - 0.2 if "channel" in name else (y0 + y1) / 2, name, ha="right", va="center", fontsize=8.5,
                color="w" if name == "interconnect" else "0.25")
    # the gas along the channel, and the air
    ax.annotate("", xy=(2.2, 5.05), xytext=(0.2, 5.05), arrowprops=dict(arrowstyle="-|>", color="tab:blue", lw=1.5))
    ax.text(0.2, 4.72, f"{100 * Y_IN:.0f} % H$_2$O, {100 * (1 - Y_IN):.0f} % H$_2$ in", fontsize=9.5, color="tab:blue")
    ax.annotate("", xy=(8.8, 5.05), xytext=(6.8, 5.05), arrowprops=dict(arrowstyle="-|>", color="tab:blue", lw=1.5))
    ax.text(6.5, 4.72, "less steam, more H$_2$ out", fontsize=9.5, color="tab:blue")
    ax.annotate("", xy=(2.2, 0.85), xytext=(0.2, 0.85), arrowprops=dict(arrowstyle="-|>", color="tab:green", lw=1.5))
    ax.text(0.2, 0.42, "air in: O$_2$ added", fontsize=9.5, color="tab:green")
    # the exchange at the fuel electrode and through the electrolyte
    arrow(3.0, 4.3, 3.0, 3.05, "tab:orange", "H$_2$O", tx=2.15, ty=3.95)
    arrow(4.1, 3.05, 4.1, 4.3, "tab:green", "H$_2$", tx=4.25, ty=3.95)
    arrow(3.55, 2.95, 3.55, 1.7, "tab:red", "O$^{2-}$", tx=3.7, ty=2.3)
    arrow(3.55, 1.6, 3.55, 0.95, "tab:green", "O$_2$", tx=3.7, ty=1.1)
    ax.text(5.0, 3.62, "at the fuel electrode:\nH$_2$O + 2e$^-$ → H$_2$ + O$^{2-}$", fontsize=10.5, va="bottom")
    ax.text(5.0, 0.1, "at the air electrode:\nO$^{2-}$ → ½O$_2$ + 2e$^-$", fontsize=10.5, va="bottom")
    # the external circuit: electrons driven into the fuel electrode
    ax.plot([9.0, 9.9, 9.9, 9.0], [3.0, 3.0, 1.7, 1.7], color="k", lw=1.2)
    ax.add_patch(mpatches.Rectangle((9.6, 2.05), 0.6, 0.6, facecolor="w", edgecolor="k", lw=1.2))
    ax.text(10.35, 2.35, f"power supply\n{V_LOW:.2f} to {V_HIGH:.2f} V", fontsize=9.5, va="center")
    arrow(9.9, 2.75, 9.9, 2.95, "k")
    ax.text(10.05, 3.15, "e$^-$ into the fuel electrode", fontsize=9)
    ax.set_title(f"solid oxide electrolysis cell at {T_CELL - 273.15:.0f} °C: steam split, hydrogen returned, "
                 "power in (layers not to scale)", fontsize=10.5)
    ax.set_xlim(-0.1, 12.6); ax.set_ylim(-0.1, 6.2); ax.set_axis_off()
    return ax.figure


def draw_cell(ax=None):
    """The steam channel of the electrolysis cell along its length: steam in
    on the left, split at the fuel electrode below it, the oxygen ions through
    the electrolyte to the air side. The height is drawn ten times too large."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 3.8))
    L, h = L_CHANNEL * 1e3, 10.0 * H_CHANNEL * 1e3
    ax.add_patch(plt.Rectangle((0, h), L, 2.5, fc="0.75", ec="k", lw=0.8))
    ax.text(L / 2, h + 1.25, "interconnect", ha="center", va="center", fontsize=9)
    ax.add_patch(plt.Rectangle((0, 0), L, h, fc="#DDEEFF", ec="k", lw=0.8))
    ax.add_patch(plt.Rectangle((0, -3.5), L, 3.5, fc="0.55", ec="k", lw=0.8))
    ax.text(L / 2, -1.75, "fuel electrode:  H₂O + 2e⁻ → H₂ + O²⁻", ha="center", va="center", fontsize=9, color="w")
    ax.add_patch(plt.Rectangle((0, -7.0), L, 3.5, fc="#F3E3C3", ec="k", lw=0.8))
    ax.text(L / 2, -5.25, "electrolyte: O²⁻ to the air side", ha="center", va="center", fontsize=9)
    for xa in np.linspace(30, 92, 6):
        ax.annotate("", xy=(xa, 0.4), xytext=(xa, h / 2 + 1.5), arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1))
        ax.annotate("", xy=(xa + 2.5, h / 2 + 1.5), xytext=(xa + 2.5, 0.4), arrowprops=dict(arrowstyle="->", color="tab:green", lw=1))
    ax.text(61, h / 2 + 2.6, "steam down, hydrogen up: one for one", ha="center", fontsize=9)
    for z in np.linspace(0.15, 0.85, 5):
        ax.annotate("", xy=(16, z * h), xytext=(4, z * h), arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1))
    ax.text(-2, h / 2, f"{100 * Y_IN:.0f} % steam\n{U_GAS:.1f} m/s", ha="right", va="center", fontsize=9)
    ax.text(L + 2, h / 2, "outlet", ha="left", va="center", fontsize=9)
    ax.annotate("", xy=(0, h + 5), xytext=(L, h + 5), arrowprops=dict(arrowstyle="<->", lw=0.8, color="k"))
    ax.text(L / 2, h + 5.6, f"{L:.0f} mm", ha="center", va="bottom", fontsize=9)
    ax.annotate("", xy=(L + 13, 0), xytext=(L + 13, h), arrowprops=dict(arrowstyle="<->", lw=0.8, color="k"))
    ax.text(L + 15, h / 2, f"{H_CHANNEL * 1e3:.0f} mm", va="center", fontsize=9)
    ax.text(L + 2, -5.0, f"cell voltage\n{V_LOW:.2f} to {V_HIGH:.2f} V", ha="left", va="center", fontsize=9)
    ax.set_xlim(-22, L + 26); ax.set_ylim(-8, h + 9); ax.set_aspect("equal"); ax.axis("off")
    return ax.figure
