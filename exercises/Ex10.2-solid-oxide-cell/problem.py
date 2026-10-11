r"""Ex_10.2 — the problem: steam, charge and heat along the fuel channel of a solid oxide electrolysis cell.

*Deep Learning for Engineering* — MSc, Aalborg University.
Remus Teodorescu (ret@et.aau.dk), with support from Research Assistant
Noman Khan (nomank@energy.aau.dk).

**Reference texts.** Liu, *PINN with Python: An Introduction* (2025); Raissi,
Perdikaris & Karniadakis, *Physics-informed neural networks*, J. Comput. Phys.
**378** (2019) 686–707. These are the works to read for the theory. The code,
the problem and the exposition here are original to this course.

## The problem

A solid oxide cell, 100 × 100 mm of electrode, run as an electrolyser at a
fixed cell voltage ``V`` with gas entering at ``T_in``. Steam enters the
hydrogen (fuel) channel, 100 mm long and 1 mm high (Ex_09.2's channel), and is
split into hydrogen along the fuel electrode. Three fields along the channel,
``x`` over its length:

* ``y``  the steam mole fraction                       [-]
* ``T``  the temperature of cell and gas, one value    [K]
* ``i``  the local current density                     [A/cm²]

and three laws that couple them:

    steam balance      y_x = y_xx / Pe - K_I i                       (uses i)
    energy balance     theta_x = theta_xx / Pe_T + S_Q i (V - V_tn)  (uses i)
    charge transfer    V = E(y, T) + (RT/F) asinh(i / 2 i_0)
                           + i ASR(T) - (RT/2F) ln(1 - i / i_L(y))   (uses y and T)

with ``theta = (T - T_in) / 50 K``; ``y = 0.90`` and ``theta = 0`` at the
inlet and zero slopes at the outlet. The charge transfer has no derivative in
it: at every point it is an equation for ``i`` given ``y`` and ``T``. The heat
source ``i (V - V_tn)`` cools the cell below the thermoneutral voltage and
heats it above.

## Which numbers are measured and which are not

The thermodynamics is standard: the Nernst potential, Faraday's law, the
thermoneutral voltage of 1.29 V. The kinetic and transport parameters (the
area-specific resistance and its activation energy, the exchange current
density, the limiting current density), the gas diffusivity, the heat
capacity flow and the solid's conduction, the stack cost and the ageing law,
and the two prices are **estimates chosen for the exercise**, of the right
order and not measured on any cell; each is marked ``[estimate]`` where it is
defined. There is no community reference code for solid oxide cells as
PyBaMM is for batteries; a result that depends on these numbers must say so.

## What is in here

The data; the cell's laws, written so that they take NumPy arrays and torch
tensors alike; the local current from the charge transfer, by bisection; the
reference, a boundary-value solver; the cost of an hour of operation; the
sampler; and the drawings. The finite-volume solver and the network are
written in the notebook.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "FARADAY", "R_GAS", "P_O2", "Y_IN", "L_CHANNEL", "H_CHANNEL", "W_CELL", "A_CELL", "U_GAS", "D_GAS",
    "T_REF", "ASR_0", "E_ASR", "I_0", "I_LIM_0", "GAMMA", "LAMBDA_T", "DT_SCALE", "C_TOT",
    "PE", "PE_T", "K_I", "S_Q", "V_TN", "V_LOW", "V_HIGH", "T_LOW", "T_HIGH",
    "M_H2", "PRICE_H2", "PRICE_EL", "STACK_COST", "P_RATED", "DASR_EOL", "K_DEG", "E_DEG", "N_DEG", "DT_MAX",
    "voltage", "inlet_temperature", "scaled", "nernst", "asr", "i_lim", "cell_voltage", "current",
    "reference", "cost_per_hour", "sample_points", "describe_problem",
    "draw_soec", "draw_cell", "draw_coupling",
]

# ----------------------------------------------------------------- the data
FARADAY = 96485.0        #: C/mol
R_GAS = 8.314            #: J/(mol K)
P_O2 = 0.21              #: atm, oxygen on the air side
Y_IN = 0.90              #: steam fraction at the inlet; the rest is hydrogen
L_CHANNEL = 0.10         #: m, length of the channel and the cell
H_CHANNEL = 1.0e-3       #: m, height of the gas channel
W_CELL = 0.10            #: m, width of the cell
A_CELL = 100.0           #: cm^2 of electrode, 100 x 100 mm
U_GAS = 1.0              #: m/s, gas speed in the channel
D_GAS = 8.0e-4           #: m^2/s, steam in hydrogen at 800 degC   [estimate]
M_H2 = 2.016e-3          #: kg/mol

T_REF = 1073.15          #: K, 800 degC, where the resistance below is given
ASR_0 = 0.15             #: ohm cm^2, ohmic resistance of electrolyte and contacts at 800 degC   [estimate]
E_ASR = 80000.0          #: J/mol, its activation energy: the electrolyte conducts better when hot   [estimate]
I_0 = 0.5                #: A/cm^2, exchange current density of the electrodes   [estimate]
I_LIM_0 = 4.0            #: A/cm^2, limiting current density at the inlet steam fraction   [estimate]

GAMMA = 4.0              #: W/(m K) per m of cell width: heat capacity flow of the fuel and air gases   [estimate]
LAMBDA_T = 0.025         #: W/K: in-plane conduction of the cell and interconnect, k times thickness   [estimate]
DT_SCALE = 50.0          #: K, the temperature scale: theta = (T - T_in) / 50 K

C_TOT = 101325.0 / (R_GAS * T_REF)                       #: mol/m^3, gas at 1 atm, held at 800 degC
PE = U_GAS * L_CHANNEL / D_GAS                           #: the Peclet number of the steam, 125
PE_T = GAMMA * L_CHANNEL / LAMBDA_T                      #: the Peclet number of the heat, 16
K_I = L_CHANNEL / (U_GAS * H_CHANNEL * C_TOT * 2 * FARADAY) * 1e4   #: steam used per (A/cm^2), scaled
S_Q = L_CHANNEL * 1e4 / (GAMMA * DT_SCALE)               #: heat per (W/cm^2), scaled: 5.0
V_TN = 248000.0 / (2 * FARADAY)                          #: V, the thermoneutral voltage, 1.29

#: The operating range the network is trained over: cell voltage and inlet temperature.
V_LOW, V_HIGH = 1.15, 1.35
T_LOW, T_HIGH = 1023.15, 1123.15                         #: K, 750 to 850 degC

#: The cost of an hour of operation, section 6.   [chosen for the exercise]
PRICE_H2 = 5.00          #: euro per kg of hydrogen
PRICE_EL = 0.05          #: euro per kWh of electricity
STACK_COST = 2000.0      #: euro per kW of rated cell power   [estimate]
P_RATED = 0.25           #: kW, rated power of one 100 cm^2 cell
DASR_EOL = 0.15          #: ohm cm^2: end of life when the resistance has grown by this much (doubled)   [estimate]
K_DEG, E_DEG, N_DEG = 0.184, 100000.0, 1.5   #: ageing law dASR/dt = K e^(-E/RT) i^n, ohm cm^2 per hour   [estimate]
DT_MAX = 10.0            #: K, the largest temperature change along the cell allowed in operation   [estimate]


# ------------------------------------------------------------ numpy or torch
def _torch(z):
    """True if ``z`` is a torch tensor: each law below is written once and
    serves the solver (NumPy) and the network (torch) alike."""
    try:
        import torch
        return torch.is_tensor(z)
    except ImportError:                              # pragma: no cover
        return False


def _clip(z, lo, hi=None):
    return z.clamp(lo, hi) if _torch(z) else np.clip(z, lo, hi)


def _exp(z):
    return z.exp() if _torch(z) else np.exp(z)


def _log(z):
    return z.log() if _torch(z) else np.log(z)


def _asinh(z):
    return z.asinh() if _torch(z) else np.arcsinh(z)


def voltage(v):
    """The cell voltage in volts for a scaled voltage ``v`` in [0, 1]."""
    return V_LOW + (V_HIGH - V_LOW) * v


def inlet_temperature(tau):
    """The inlet temperature in kelvin for a scaled temperature ``tau`` in [0, 1]."""
    return T_LOW + (T_HIGH - T_LOW) * tau


def scaled(V, T_in):
    """The scaled inputs ``(v, tau)`` of a cell voltage [V] and an inlet temperature [K]."""
    return (V - V_LOW) / (V_HIGH - V_LOW), (T_in - T_LOW) / (T_HIGH - T_LOW)


# ------------------------------------------------------------- the cell's laws
def nernst(y, T):
    """The Nernst potential [V] at steam fraction ``y`` and temperature ``T``
    [K]. ``y`` is held inside (1e-4, 1 - 1e-4), so that a network's first
    guesses cannot reach the logarithm's ends."""
    y = _clip(y, 1e-4, 1 - 1e-4)
    return 1.253 - 2.4516e-4 * T + R_GAS * T / (2 * FARADAY) * _log((1 - y) * P_O2 ** 0.5 / y)


def asr(T):
    """The area-specific (ohmic) resistance [ohm cm^2] at temperature ``T`` [K]: Arrhenius."""
    return ASR_0 * _exp(E_ASR / R_GAS * (1 / T - 1 / T_REF))


def i_lim(y):
    """The limiting current density [A/cm^2]: the most steam the porous
    electrode can take in, proportional to the steam fraction in the channel."""
    return I_LIM_0 * _clip(y, 1e-4) / Y_IN


def cell_voltage(i, y, T):
    """The cell voltage [V] that drives current density ``i`` [A/cm^2] where the
    steam fraction is ``y`` and the temperature ``T`` [K]: the Nernst
    potential, the activation loss (Butler–Volmer, symmetric), the ohmic loss
    and the gas-transport loss."""
    gas = _clip(1 - i / i_lim(y), 1e-6)
    return (nernst(y, T) + R_GAS * T / FARADAY * _asinh(i / (2 * I_0))
            + i * asr(T) - R_GAS * T / (2 * FARADAY) * _log(gas))


def current(V, y, T, steps=60):
    """The local current density [A/cm^2] at cell voltage ``V`` where the steam
    fraction is ``y`` and the temperature ``T``: the charge transfer solved
    for ``i``, point by point, by bisection on [0, i_L). ``cell_voltage`` rises
    with ``i`` on that interval, so the root is unique. NumPy only."""
    y, T = np.broadcast_arrays(np.asarray(y, float), np.asarray(T, float))
    lo = np.zeros_like(y); hi = i_lim(y) * (1 - 1e-12)
    for _ in range(steps):
        mid = 0.5 * (lo + hi); above = cell_voltage(mid, y, T) > V
        hi = np.where(above, mid, hi); lo = np.where(above, lo, mid)
    return 0.5 * (lo + hi)


# ------------------------------------------------------------- the reference
def reference(V, T_in, tol=1e-9):
    """The reference solution at cell voltage ``V`` [V] and inlet temperature
    ``T_in`` [K]: SciPy's boundary-value solver, a collocation method on a
    mesh it refines itself until the residual is below ``tol``, with the
    current solved at every point. Returns a function of ``x`` in [0, 1] that
    gives ``(y, T, i)``."""
    from scipy.integrate import solve_bvp

    def f(x, s):
        y, y_x, th, th_x = s
        i = current(V, y, T_in + DT_SCALE * th)
        return np.vstack([y_x, PE * (y_x + K_I * i), th_x, PE_T * (th_x - S_Q * i * (V - V_TN))])

    bc = lambda a, b: np.array([a[0] - Y_IN, b[1], a[2], b[3]])
    x = np.linspace(0.0, 1.0, 401)
    guess = np.vstack([Y_IN - 0.4 * x, -0.4 * np.ones_like(x), 0 * x, 0 * x])
    sol = solve_bvp(f, bc, x, guess, tol=tol, max_nodes=500000)
    assert sol.success, sol.message

    def fields(xx):
        y, _, th, _ = sol.sol(np.asarray(xx, float))
        T = T_in + DT_SCALE * th
        return y, T, current(V, y, T)
    return fields


# ---------------------------------------------------------- the cost of an hour
def cost_per_hour(x, i, T, V):
    """The cost of an hour of operation of one cell, in euro cents, from the
    current density ``i`` [A/cm^2] and the temperature ``T`` [K] on points
    ``x`` (scaled, increasing, from 0 to 1, along the first axis) at cell
    voltage ``V``: the electricity, less the hydrogen sold, plus the share of
    the stack that ages in the hour. NumPy in gives NumPy out, tensors give
    tensors, so the cost can be differentiated through a network. Returns a
    dict: the total and its three parts [cent/h], the current [A] and the
    life of the cell at this operating point [h]."""
    if _torch(i):
        w = x[1:] - x[:-1]                                    # the trapezoidal rule along the channel
        integral = lambda f: (0.5 * w * (f[1:] + f[:-1])).sum(0)
    else:
        integral = lambda f: np.trapezoid(f, x, axis=0)
    current_A = integral(i) * A_CELL                          # A
    hydrogen = current_A / (2 * FARADAY) * M_H2 * 3600        # kg/h, by Faraday's law
    power = V * current_A / 1000                              # kW
    ageing = integral(K_DEG * _exp(-E_DEG / (R_GAS * T)) * _clip(i, 0.0) ** N_DEG)   # ohm cm^2 per hour
    c_el, c_h2 = 100 * PRICE_EL * power, 100 * PRICE_H2 * hydrogen
    c_age = 100 * STACK_COST * P_RATED * ageing / DASR_EOL
    return dict(total=c_el - c_h2 + c_age, electricity=c_el, hydrogen=c_h2, ageing=c_age,
                current=current_A, life_h=DASR_EOL / ageing)


# ------------------------------------------------------------------- sampler
def sample_points(n, seed=88, edges=0.2):
    """``n`` collocation points ``(x, v, tau)``: a position along the channel,
    a scaled cell voltage and a scaled inlet temperature, uniform in [0, 1].
    A share ``edges`` of them is put on the edges of the operating range, half
    at the lowest or highest voltage and half at the lowest or highest inlet
    temperature, where the cell runs short of steam or cools most."""
    rng = np.random.default_rng(seed)
    p = rng.random((n, 3)); m = int(n * edges / 2)
    p[:m, 1] = rng.integers(0, 2, m)
    p[m:2 * m, 2] = rng.integers(0, 2, m)
    return p


def describe_problem() -> None:
    """Print the cell and the numbers it implies."""
    print(f"  cell             : solid oxide, electrolysis, {L_CHANNEL * 1e3:.0f} x {W_CELL * 1e3:.0f} mm of electrode")
    print(f"  fuel channel     : {H_CHANNEL * 1e3:.0f} mm high, gas at {U_GAS:.1f} m/s, {100 * Y_IN:.0f} % steam at the inlet")
    print(f"  operating range  : cell voltage {V_LOW:.2f} to {V_HIGH:.2f} V, inlet {T_LOW - 273.15:.0f} to {T_HIGH - 273.15:.0f} degC")
    print(f"  resistance       : ASR = {ASR_0:.2f} ohm cm^2 at 800 degC, {asr(T_LOW):.2f} at 750, {asr(T_HIGH):.2f} at 850   [estimate]")
    print(f"  electrodes       : exchange current density {I_0:.2f} A/cm^2; limiting current {I_LIM_0:.2f} A/cm^2 at the inlet steam   [estimate]")
    print(f"  Nernst potential : {float(nernst(np.array(Y_IN), T_REF)):.2f} V at the inlet, 800 degC; thermoneutral voltage {V_TN:.2f} V")
    print(f"  steam            : Pe = u L / D = {PE:.0f};  K_I = {K_I:.3f} per (A/cm^2)")
    print(f"  heat             : Pe_T = Gamma L / (k t) = {PE_T:.0f};  S_Q = {S_Q:.2f} per (W/cm^2);  theta = (T - T_in) / {DT_SCALE:.0f} K")
    supply = U_GAS * H_CHANNEL * W_CELL * C_TOT * Y_IN
    print(f"  steam supplied   : {supply * 1000:.2f} mmol/s; using all of it would take {supply * 2 * FARADAY:.0f} A")


# ------------------------------------------------------------------ drawings
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
    ax.annotate("", xy=(2.2, 5.05), xytext=(0.2, 5.05), arrowprops=dict(arrowstyle="-|>", color="tab:blue", lw=1.5))
    ax.text(0.2, 4.72, f"{100 * Y_IN:.0f} % H$_2$O, {100 * (1 - Y_IN):.0f} % H$_2$ in", fontsize=9.5, color="tab:blue")
    ax.annotate("", xy=(8.8, 5.05), xytext=(6.8, 5.05), arrowprops=dict(arrowstyle="-|>", color="tab:blue", lw=1.5))
    ax.text(6.5, 4.72, "less steam, more H$_2$ out", fontsize=9.5, color="tab:blue")
    ax.annotate("", xy=(2.2, 0.85), xytext=(0.2, 0.85), arrowprops=dict(arrowstyle="-|>", color="tab:green", lw=1.5))
    ax.text(0.2, 0.42, "air in: O$_2$ added", fontsize=9.5, color="tab:green")
    arrow(3.0, 4.3, 3.0, 3.05, "tab:orange", "H$_2$O", tx=2.15, ty=3.95)
    arrow(4.1, 3.05, 4.1, 4.3, "tab:green", "H$_2$", tx=4.25, ty=3.95)
    arrow(3.55, 2.95, 3.55, 1.7, "tab:red", "O$^{2-}$", tx=3.7, ty=2.3)
    arrow(3.55, 1.6, 3.55, 0.95, "tab:green", "O$_2$", tx=3.7, ty=1.1)
    ax.text(5.0, 3.62, "at the fuel electrode:\nH$_2$O + 2e$^-$ → H$_2$ + O$^{2-}$", fontsize=10.5, va="bottom")
    ax.text(5.0, 0.1, "at the air electrode:\nO$^{2-}$ → ½O$_2$ + 2e$^-$", fontsize=10.5, va="bottom")
    ax.plot([9.0, 9.9, 9.9, 9.0], [3.0, 3.0, 1.7, 1.7], color="k", lw=1.2)
    ax.add_patch(mpatches.Rectangle((9.6, 2.05), 0.6, 0.6, facecolor="w", edgecolor="k", lw=1.2))
    ax.text(10.35, 2.35, f"power supply\n{V_LOW:.2f} to {V_HIGH:.2f} V", fontsize=9.5, va="center")
    arrow(9.9, 2.75, 9.9, 2.95, "k")
    ax.text(10.05, 3.15, "e$^-$ into the fuel electrode", fontsize=9)
    ax.set_title("solid oxide electrolysis cell at 750 to 850 °C: steam split, hydrogen returned, "
                 "power in (layers not to scale)", fontsize=10.5)
    ax.set_xlim(-0.1, 12.6); ax.set_ylim(-0.1, 6.2); ax.set_axis_off()
    return ax.figure


def draw_cell(ax=None):
    """The hydrogen (fuel) channel of the electrolysis cell along its length:
    steam in on the left, split at the fuel electrode below it, the oxide ions
    through the electrolyte to the air side, and the three fields the notebook
    solves for. The height is drawn ten times too large."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4.2))
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
    ax.text(-2, h / 2, f"{100 * Y_IN:.0f} % steam\n{U_GAS:.1f} m/s\nat $T_{{in}}$", ha="right", va="center", fontsize=9)
    ax.text(L + 2, h / 2, "outlet", ha="left", va="center", fontsize=9)
    ax.annotate("", xy=(0, h + 5), xytext=(L, h + 5), arrowprops=dict(arrowstyle="<->", lw=0.8, color="k"))
    ax.text(L / 2, h + 5.6, f"{L:.0f} mm:  steam fraction $y(x)$,  temperature $T(x)$,  current density $i(x)$",
            ha="center", va="bottom", fontsize=9)
    ax.annotate("", xy=(L + 13, 0), xytext=(L + 13, h), arrowprops=dict(arrowstyle="<->", lw=0.8, color="k"))
    ax.text(L + 15, h / 2, f"{H_CHANNEL * 1e3:.0f} mm", va="center", fontsize=9)
    ax.text(L + 2, -5.0, f"cell voltage\n{V_LOW:.2f} to {V_HIGH:.2f} V", ha="left", va="center", fontsize=9)
    ax.set_xlim(-22, L + 26); ax.set_ylim(-8, h + 9); ax.set_aspect("equal"); ax.axis("off")
    return ax.figure


def draw_coupling(ax=None):
    """The three laws of the channel and what each passes to the others: the
    multiphysics of the cell in one picture. The steam balance and the energy
    balance are differential equations along x; the charge transfer is an
    algebraic equation at every point, and it takes from both and gives to
    both."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 5.0))
    boxes = [  # (x, y, w, colour, title, equation, kind)
        (0.0, 2.7, 3.8, "#dbe9f6", "steam balance", r"$y_x = y_{xx}/\mathrm{Pe} - K_I\,i$", "differential, along $x$"),
        (3.2, 0.0, 5.2, "#f6e3c7", "charge transfer",
         r"$V = E(y,T) + \eta_{act}(i,T) + i\,\mathrm{ASR}(T) + \eta_{gas}(i,y)$", "algebraic, at every point"),
        (7.8, 2.7, 3.8, "#f6d6d6", "energy balance", r"$\theta_x = \theta_{xx}/\mathrm{Pe}_T + S_Q\,i\,(V - V_{tn})$",
         "differential, along $x$"),
    ]
    H = 1.55
    for x0, y0, w, col, title, eq, kind in boxes:
        ax.add_patch(FancyBboxPatch((x0, y0), w, H, boxstyle="round,pad=0.05", fc=col, ec="k", lw=1))
        ax.text(x0 + w / 2, y0 + H - 0.3, title, ha="center", va="center", fontsize=11, fontweight="bold")
        ax.text(x0 + w / 2, y0 + H / 2 - 0.05, eq, ha="center", va="center", fontsize=10)
        ax.text(x0 + w / 2, y0 + 0.22, kind, ha="center", va="center", fontsize=8.5, color="0.35")

    def arrow(x0, y0, x1, y1, text, tx, ty, col, ha):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.8, mutation_scale=15))
        ax.text(tx, ty, text, ha=ha, va="center", fontsize=9, color=col)

    arrow(1.2, 2.65, 3.3, 1.2, "$y$: Nernst potential,\nlimiting current", 1.6, 1.4, "tab:blue", "right")
    arrow(4.3, 1.62, 2.8, 2.65, "$i$: steam used", 3.9, 2.25, "tab:orange", "left")
    arrow(7.3, 1.62, 8.8, 2.65, "$i$: heat $i\\,(V - V_{tn})$", 7.7, 2.25, "tab:orange", "right")
    arrow(10.4, 2.65, 8.3, 1.2, "$T$: Nernst potential,\nkinetics, resistance", 10.0, 1.4, "tab:red", "left")
    ax.text(5.8, 4.75, "two-way coupling: the current depends on the steam and the temperature, and changes both",
            ha="center", va="center", fontsize=9.5, style="italic")
    ax.set_xlim(-0.3, 11.9); ax.set_ylim(-0.3, 5.0); ax.set_axis_off()
    return ax.figure
