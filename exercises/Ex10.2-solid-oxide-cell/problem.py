r"""Ex_10.2 — the physics: a reversible solid oxide cell, SOFC and SOEC in one.

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

The same hardware runs both ways, so the sign of the current selects the mode
and nothing else changes. Fuel-cell mode has V < E; electrolysis has V > E.

    Nernst potential          thermodynamics of H2 + 1/2 O2 <-> H2O
    three overpotentials      activation, ohmic, concentration
    thermoneutral voltage     V_tn = dH / 2F  (~1.29 V for steam at 800 C)
    heat generation           q = i (V - V_tn)   - zero at the thermoneutral point
    degradation               Arrhenius in T, power law in |i|

Sign convention, everywhere in this file: **i > 0 is electrolysis (SOEC), i < 0
is fuel cell (SOFC)**. The losses always oppose the useful direction, so they
add in electrolysis and subtract in fuel-cell mode, and one expression handles
both by using the signed current directly.

Every function that a residual or an objective might have to differentiate
through dispatches on ``torch.is_tensor`` and works unchanged on NumPy arrays
and on torch tensors. Nothing here needs a copy for each backend.

## Parameter provenance

Every default below carries a source tag. Values marked ESTIMATED are
order-of-magnitude placeholders chosen so the model reproduces published
polarisation behaviour — they are NOT measured, and any result that depends on
them must say so. Unlike the battery half of L10, there is no community
reference implementation to fall back on: **there is no PyBaMM for solid oxide
cells**, and this module is a course implementation rather than a validated
community code. Knowing which of your results are defensible is the point of
the exercise.

## The lab half

Below the physics sits the control panel, the plots and the report writer.
There is no physics in that half: the residuals, the fit and the objective are
written by the student. This module runs cases, draws the polarisation and
trajectory plots, and assembles the report.

## Sources

    IEA SOFC Benchmark Test 1 (Achenbach, 1994/96) - single cell on hydrogen;
        the report and data are in the public domain.
    Published button-cell polarisation fits give activation energies of order
        100 kJ/mol (fuel electrode) and 66 kJ/mol (oxygen electrode).
    SOEC thermoneutral operation at 1.29 V is standard practice; degradation
        rises with temperature.
"""

from __future__ import annotations

import os

import numpy as np
import torch

__all__ = [
    "F_CONST", "R_GAS", "MODES", "CHANNEL_DOMAIN",
    "SOCParams", "nernst", "eta_activation", "eta_ohmic",
    "eta_concentration", "cell_voltage", "thermoneutral_voltage",
    "heat_generation", "polarisation_curve", "degradation_rate",
    "integrate_degradation", "channel_profile", "hydrogen_rate",
    "control_panel", "plot_polarisation", "plot_heat", "plot_channel",
    "plot_trajectory", "price_profile", "evaluate_trajectory", "make_report",
]

F_CONST = 96485.0
R_GAS = 8.314
MODES = ("SOFC", "SOEC")

#: ``((x_lo, x_hi),)`` — the channel coordinate, normalised to [0, 1].
#: The same coordinate :func:`channel_profile` uses, so a 1-D channel PINN and
#: the reference profile can be compared point for point.
CHANNEL_DOMAIN = ((0.0, 1.0),)


class SOCParams:
    """Operating point and cell properties for a reversible solid oxide cell."""

    def __init__(self, mode="SOEC", T=1073.15, p_H2=0.10, p_H2O=0.90,
                 p_O2=0.21, ASR_0=0.25, i_L=2.5, i_0=0.5, utilisation=0.6,
                 area=0.01):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.mode = mode
        self.T = float(T)                 # K
        self.p_H2 = float(p_H2)           # fuel-side partial pressures, atm
        self.p_H2O = float(p_H2O)
        self.p_O2 = float(p_O2)           # air side
        self.ASR_0 = float(ASR_0)         # ohm cm2 at T_ref   [ESTIMATED]
        self.i_L = float(i_L)             # limiting current, A/cm2 [ESTIMATED]
        self.i_0 = float(i_0)             # exchange current, A/cm2 [ESTIMATED]
        self.utilisation = float(utilisation)
        self.area = float(area)           # m2
        self.T_ref = 1073.15
        self.E_act_ASR = 80000.0          # J/mol  [ESTIMATED, in the published range]
        self.E_deg = 100000.0             # J/mol  [ESTIMATED]
        self.deg_exponent = 1.5           # [ESTIMATED]
        self.deg_k0 = 1.84e-1             # [ESTIMATED, gives ~1%/kh at 800 C]

    @property
    def T_celsius(self):
        return self.T - 273.15

    def __repr__(self):
        return (f"SOCParams({self.mode}, T={self.T_celsius:.0f}C, "
                f"pH2={self.p_H2:.2f}, pH2O={self.p_H2O:.2f}, "
                f"ASR0={self.ASR_0:.3f})")


# ------------------------------------------------------------ thermodynamics
def nernst(par, p_H2=None, p_H2O=None):
    """Reversible cell voltage for H2 + 1/2 O2 <-> H2O.

    E0(T) is the standard linear fit, about 0.99 V at 800 C.
    """
    p_H2 = par.p_H2 if p_H2 is None else p_H2
    p_H2O = par.p_H2O if p_H2O is None else p_H2O
    E0 = 1.253 - 2.4516e-4 * par.T
    return E0 + (R_GAS * par.T) / (2 * F_CONST) * np.log(
        np.maximum(p_H2, 1e-12) * np.sqrt(par.p_O2) / np.maximum(p_H2O, 1e-12))


def thermoneutral_voltage(T=1073.15):
    """V_tn = dH / 2F.

    Using dH ~ 248 kJ/mol for steam at high temperature gives about 1.285 V,
    which is the 1.29 V quoted throughout the SOEC literature. Compare 1.48 V
    for low-temperature liquid-water electrolysis.
    """
    dH = 248000.0                      # J/mol, steam, high temperature
    return dH / (2 * F_CONST)


# ------------------------------------------------------------- overpotentials
def eta_activation(i, par):
    """Symmetric Butler-Volmer, inverted to an arcsinh.

    Better conditioned than the exponential form and it cannot overflow -
    the same argument as the battery kinetics in L10.1.
    """
    bk = torch if torch.is_tensor(i) else np
    asinh = bk.asinh if hasattr(bk, "asinh") else bk.arcsinh
    return (R_GAS * par.T) / F_CONST * asinh(i / (2.0 * par.i_0))


def eta_ohmic(i, par, T=None):
    """i * ASR(T), with ASR falling as temperature rises (Arrhenius)."""
    T = par.T if T is None else T
    bk = torch if torch.is_tensor(i) else np
    asr = par.ASR_0 * bk.exp(par.E_act_ASR / R_GAS * (1.0 / T - 1.0 / par.T_ref))
    return i * asr


def eta_concentration(i, par):
    """Mass-transport limitation as the current approaches i_L."""
    bk = torch if torch.is_tensor(i) else np
    x = 1.0 - bk.abs(i) / par.i_L
    x = bk.clip(x, 1e-4, None) if bk is np else torch.clamp(x, min=1e-4)
    return -(R_GAS * par.T) / (2 * F_CONST) * bk.log(x)


def cell_voltage(i, par, p_H2=None, p_H2O=None):
    """Cell voltage at current density i [A/cm2].

    Sign convention: i > 0 is electrolysis (SOEC), i < 0 is fuel cell (SOFC).
    The losses always oppose the useful direction, so they add in electrolysis
    and subtract in fuel-cell mode - which one expression handles by using the
    signed current directly.
    """
    E = nernst(par, p_H2, p_H2O)
    return E + eta_activation(i, par) + eta_ohmic(i, par) + \
        np.sign(i) * eta_concentration(i, par) if not torch.is_tensor(i) else \
        E + eta_activation(i, par) + eta_ohmic(i, par) + \
        torch.sign(i) * eta_concentration(i, par)


def polarisation_curve(par, i_min=-1.5, i_max=1.5, n=201):
    """Voltage against current density across both operating modes."""
    i = np.linspace(i_min, i_max, n)
    return i, cell_voltage(i, par)


# -------------------------------------------------------------------- energy
def heat_generation(i, par):
    """q = i (V - V_tn) [W/cm2].

    Collapses all three overpotentials and the reaction entropy into the
    deviation from the thermoneutral voltage. Negative means the cell absorbs
    heat (endothermal electrolysis); zero is the thermoneutral point.
    """
    return i * (cell_voltage(i, par) - thermoneutral_voltage(par.T))


def hydrogen_rate(i, par):
    """Hydrogen production rate [mol/s] from Faraday's law. Positive in SOEC."""
    return i * (par.area * 1e4) / (2.0 * F_CONST)


# --------------------------------------------------------------- degradation
def degradation_rate(i, par, T=None):
    """d(ASR)/dt: Arrhenius in temperature, power law in current density.

    An empirical fit, not a mechanism. Published rates vary by orders of
    magnitude, so use this for trends only - never quote a predicted lifetime.
    """
    T = par.T if T is None else T
    return (par.deg_k0 * np.exp(-par.E_deg / (R_GAS * T))
            * np.abs(i) ** par.deg_exponent)


def integrate_degradation(i_traj, T_traj, par, dt_hours):
    """Accumulate ASR growth along an operating trajectory."""
    i_traj = np.atleast_1d(i_traj)
    T_traj = np.atleast_1d(np.broadcast_to(T_traj, i_traj.shape))
    rates = np.array([degradation_rate(ii, par, TT)
                      for ii, TT in zip(i_traj, T_traj)])
    return par.ASR_0 + np.cumsum(rates * dt_hours)


# ----------------------------------------------------------------- 1-D channel
def channel_profile(par, i_mean, n=80, n_H2O_in=None):
    """Composition along the channel, and the local Nernst potential.

    Reactant is consumed along the flow, so the local Nernst potential falls
    from inlet to outlet. The applied cell voltage is essentially uniform along
    the cell; what varies is the local current density, because the local
    driving force differs. This is what a zero-dimensional model cannot
    represent.
    """
    x = np.linspace(0.0, 1.0, n)

    # Utilisation from the operating condition rather than as a free input.
    # The current draws reactant at I/(2F); the inlet supplies n_H2O_in.
    # Supplying n_H2O_in makes i_mean matter, which it did not before.
    if n_H2O_in is not None and n_H2O_in > 0.0:
        I = float(i_mean) * (par.area * 1e4)          # A, from A/cm2
        U = (I / (2.0 * F_CONST)) / n_H2O_in
        U = float(np.clip(U, 0.0, 0.95))
    else:
        U = par.utilisation

    if par.mode == "SOEC":
        p_H2O = par.p_H2O * (1.0 - U * x)
        p_H2 = par.p_H2 + par.p_H2O * U * x
    else:
        p_H2 = par.p_H2 * (1.0 - U * x)
        p_H2O = par.p_H2O + par.p_H2 * U * x
    E_local = np.array([nernst(par, a, b) for a, b in zip(p_H2, p_H2O)])
    return x, p_H2, p_H2O, E_local


# ------------------------------------------------------------ operating value
def price_profile(n=24, kind="daily"):
    """A 24-hour electricity price in arbitrary units.

    'daily' has a night trough and two peaks - enough structure to make the
    optimal trajectory non-constant, which is the whole point.
    """
    h = np.arange(n)
    if kind == "flat":
        return np.ones(n)
    if kind == "volatile":
        rng = np.random.default_rng(7)
        return np.clip(1.0 + 0.7 * np.sin(2 * np.pi * h / 24)
                       + 0.5 * rng.standard_normal(n), 0.15, None)
    return (1.0 + 0.55 * np.sin(2 * np.pi * (h - 8) / 24)
            + 0.30 * np.sin(4 * np.pi * (h - 4) / 24))


def evaluate_trajectory(i_traj, T_traj, par, price, h_value=3.0, dt_hours=1.0):
    """Value produced, energy cost, and end-of-life resistance for a trajectory.

    Returns a dict. Positive current is electrolysis, so hydrogen value is
    earned and electricity is paid for.
    """
    i_traj = np.asarray(i_traj, float)
    T_traj = np.broadcast_to(np.asarray(T_traj, float), i_traj.shape)
    price = np.asarray(price, float)

    # ASR is a state: it grows with use, and the grown value sets the voltage
    # at the next step. Evaluating every step at ASR_0 would make degradation
    # invisible to the objective, which was a defect in an earlier version.
    asr_state = par.ASR_0
    V, asr_hist = [], []
    for ii, TT in zip(i_traj, T_traj):
        V.append(cell_voltage(ii, SOCParams(par.mode, TT, par.p_H2, par.p_H2O,
                                            par.p_O2, asr_state, par.i_L,
                                            par.i_0, par.utilisation, par.area)))
        asr_state = asr_state + degradation_rate(ii, par, TT) * dt_hours
        asr_hist.append(asr_state)
    V = np.array(V)
    n_H2 = np.array([hydrogen_rate(ii, par) for ii in i_traj]) * 3600 * dt_hours
    P_elec = i_traj * (par.area * 1e4) * V * dt_hours / 1000.0     # kWh-ish
    asr = np.array(asr_hist)
    return {"V": V, "n_H2": n_H2, "revenue": float((h_value * n_H2).sum()),
            "cost": float((price * P_elec).sum()),
            "profit": float((h_value * n_H2 - price * P_elec).sum()),
            "ASR_end": float(asr[-1]), "ASR": asr, "i": i_traj, "T": T_traj}


# ---------------------------------------------------------------- pictures
def plot_polarisation(pars, labels=None):
    """Voltage against current density, one curve per operating point."""
    import matplotlib.pyplot as plt
    labels = labels or [repr(p) for p in pars]
    plt.figure(figsize=(7.5, 3.8))
    for p_, lb in zip(pars, labels):
        i, V = polarisation_curve(p_)
        plt.plot(i, V, label=lb)
    plt.axhline(thermoneutral_voltage(), ls="--", c="0.6", lw=1)
    plt.axvline(0, c="0.4", lw=0.8)
    plt.text(0.02, thermoneutral_voltage() + 0.02, "$V_{tn}$", fontsize=9)
    plt.xlabel("current density [A/cm$^2$]   (negative = SOFC, positive = SOEC)")
    plt.ylabel("cell voltage [V]"); plt.legend(fontsize=8)
    plt.tight_layout(); plt.show()


def plot_heat(par):
    """The polarisation curve beside the heat it generates."""
    import matplotlib.pyplot as plt
    i, V = polarisation_curve(par)
    q = np.array([heat_generation(ii, par) for ii in i])
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
    ax[0].plot(i, V); ax[0].axhline(thermoneutral_voltage(), ls="--", c="0.6")
    ax[0].set_xlabel("i [A/cm2]"); ax[0].set_ylabel("V [V]"); ax[0].set_title("polarisation")
    ax[1].plot(i, q); ax[1].axhline(0, c="0.4", lw=0.8)
    ax[1].set_xlabel("i [A/cm2]"); ax[1].set_ylabel("q [W/cm2]")
    ax[1].set_title("heat generation: negative = endothermal")
    plt.tight_layout(); plt.show()


def plot_channel(par, i_mean=1.0):
    """Composition and local Nernst potential along the flow."""
    import matplotlib.pyplot as plt
    x, pH2, pH2O, E = channel_profile(par, i_mean)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.2))
    ax[0].plot(x, pH2, label="$p_{H_2}$"); ax[0].plot(x, pH2O, label="$p_{H_2O}$")
    ax[0].set_xlabel("position along channel"); ax[0].set_ylabel("partial pressure")
    ax[0].legend()
    ax[1].plot(x, E); ax[1].set_xlabel("position along channel")
    ax[1].set_ylabel("local Nernst potential [V]")
    ax[1].set_title("inlet and outlet do not operate at the same voltage")
    plt.tight_layout(); plt.show()


def plot_trajectory(result, price):
    """Price, current, temperature and accumulated ASR over the horizon."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(4, 1, figsize=(9, 8), sharex=True)
    h = np.arange(len(result["i"]))
    ax[0].step(h, price, where="mid"); ax[0].set_ylabel("price")
    ax[1].step(h, result["i"], where="mid"); ax[1].set_ylabel("i [A/cm2]")
    ax[2].step(h, result["T"] - 273.15, where="mid"); ax[2].set_ylabel("T [degC]")
    ax[3].plot(h, result["ASR"]); ax[3].set_ylabel("ASR [ohm cm2]")
    ax[3].set_xlabel("hour")
    for a in ax:
        a.grid(alpha=0.25)
    plt.tight_layout(); plt.show()


# ------------------------------------------------------------- control panel
def control_panel(on_run):
    """Sliders for the operating point, and a button that runs your case."""
    import ipywidgets as W
    from IPython.display import display

    style = {"description_width": "165px"}; lay = W.Layout(width="430px")
    w = {
        "mode": W.Dropdown(options=list(MODES), value="SOEC",
                           description="Operating mode", style=style, layout=lay),
        "T_c": W.FloatSlider(value=800.0, min=600.0, max=900.0, step=10.0,
                             description="Temperature [degC]", style=style, layout=lay),
        "p_H2": W.FloatSlider(value=0.10, min=0.02, max=0.95, step=0.02,
                              description="Fuel-side p(H2)", style=style, layout=lay),
        "utilisation": W.FloatSlider(value=0.6, min=0.1, max=0.9, step=0.05,
                                     description="Reactant utilisation", style=style, layout=lay),
        "ASR_0": W.FloatSlider(value=0.25, min=0.08, max=0.80, step=0.01,
                               description="ASR at T_ref", style=style, layout=lay),
        "i_L": W.FloatSlider(value=2.5, min=0.8, max=4.0, step=0.1,
                             description="Limiting current", style=style, layout=lay),
        "h_value": W.FloatSlider(value=3.0, min=0.5, max=8.0, step=0.5,
                                 description="Hydrogen value", style=style, layout=lay),
        "price_kind": W.Dropdown(options=["daily", "flat", "volatile"], value="daily",
                                 description="Price profile", style=style, layout=lay),
    }
    readout = W.HTML()

    def _par():
        return SOCParams(w["mode"].value, w["T_c"].value + 273.15,
                         w["p_H2"].value, 1.0 - w["p_H2"].value, 0.21,
                         w["ASR_0"].value, w["i_L"].value, 0.5,
                         w["utilisation"].value)

    def _update(*_):
        par = _par()
        E = nernst(par); Vtn = thermoneutral_voltage()
        i_tn = None
        ii, VV = polarisation_curve(par)
        k = int(np.argmin(np.abs(VV - Vtn)))
        i_tn = ii[k]
        readout.value = (f"<div style='font-family:monospace'>OCV = {E:.3f} V"
                         f" &nbsp;|&nbsp; V_tn = {Vtn:.3f} V"
                         f" &nbsp;|&nbsp; i at V_tn = {i_tn:+.2f} A/cm2"
                         f" &nbsp;|&nbsp; mode = {par.mode}</div>")
    for x in w.values():
        x.observe(_update, "value")
    _update()

    run = W.Button(description="Run case", button_style="success", icon="play",
                   layout=W.Layout(width="170px"))
    out = W.Output()

    def _click(_):
        with out:
            out.clear_output()
            on_run(_par(), {k: v.value for k, v in w.items()})

    run.on_click(_click)
    display(W.VBox([
        W.HTML("<h3>Ex_10.2 &mdash; reversible solid oxide cell</h3>"
               "<p>Switch between fuel-cell and electrolysis mode, set the "
               "operating point, then <b>Run case</b>. The readout shows the "
               "open-circuit and thermoneutral voltages and the current at "
               "which heat generation vanishes.</p>"),
        W.HBox([W.VBox(list(w.values())[:4]), W.VBox(list(w.values())[4:])]),
        readout, run, out]))
    return w


# ------------------------------------------------------------------- report
def make_report(cases, filename="Ex10.2_report.md", author="", notes=""):
    """Assemble the submission report from the cases you ran."""
    L = ["# Ex_10.2 - Solid oxide cells: SOFC, SOEC and lifetime-aware operation", ""]
    if author:
        L.append(f"**Author:** {author}  ")
    L += [f"**Cases run:** {len(cases)}", "",
          "> Parameter provenance: values marked ESTIMATED in problem.py are",
          "> order-of-magnitude placeholders, not measurements. State this for any",
          "> result that depends on them.", "",
          "## Cases", "",
          "| # | mode | T [degC] | ASR_0 | utilisation | OCV [V] | i at V_tn | profit | ASR_end |",
          "|---|------|----------|-------|-------------|---------|-----------|--------|---------|"]
    for k, c in enumerate(cases, 1):
        p_ = c["par"]
        L.append(f"| {k} | {p_.mode} | {p_.T_celsius:.0f} | {p_.ASR_0:.3f} | "
                 f"{p_.utilisation:.2f} | {c.get('OCV', float('nan')):.3f} | "
                 f"{c.get('i_tn', float('nan')):+.2f} | "
                 f"{c.get('profit', float('nan')):.2f} | "
                 f"{c.get('ASR_end', float('nan')):.4f} |")
    L += ["", "## Your interpretation", "",
          "1. **Both modes.** Compare the polarisation curve either side of zero",
          "   current. Why do the overpotentials subtract in fuel-cell mode and add",
          "   in electrolysis, and what does that mean for efficiency?", "",
          "2. **Thermoneutral.** At what current density did heat generation vanish,",
          "   and did it match V_tn? What happens to the sign of q either side of it?", "",
          "3. **Utilisation.** How far did the local Nernst potential shift along the",
          "   channel? At what utilisation does a 0-D model become misleading?", "",
          "4. **The trade-off.** Show the optimal trajectory for a short required",
          "   lifetime and a long one. What changed, and why?", "",
          "5. **What you would not claim.** Which of your numbers rest on ESTIMATED",
          "   parameters, and what would you need to measure to defend them?", ""]
    if notes:
        L += ["## Notes", "", notes, ""]
    parent = os.path.dirname(filename)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(filename, "w") as fh:
        fh.write("\n".join(L))
    print(f"wrote {filename}  ({len(cases)} cases)")
    return filename
