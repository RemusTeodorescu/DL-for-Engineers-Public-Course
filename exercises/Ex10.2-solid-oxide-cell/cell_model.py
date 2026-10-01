r"""Ex_10.2 — the full cell model, for the mini projects (MP10.2A and MP10.2B).

*Deep Learning for Engineering* — MSc, Aalborg University.

The notebook of this set uses a simpler cell: one resistance for all losses
(``problem.py``). This module is the course's fuller model of a reversible
solid oxide cell - the Nernst potential, the three overpotentials of L10.2's
polarisation curve, the thermoneutral voltage and the heat, and an empirical
degradation law - and it is what ``tools/miniprojects/ex102_truth.py`` builds
the mini projects' ground truth with.

Sign convention: **i > 0 is electrolysis (SOEC), i < 0 is fuel cell (SOFC)**.

## Parameter provenance

Every default carries a source tag. Values marked ESTIMATED are
order-of-magnitude placeholders chosen so the model reproduces published
polarisation behaviour - they are NOT measured, and any result that depends
on them must say so. There is no community reference implementation for solid
oxide cells as PyBaMM is for batteries; this is a course implementation and
not a validated community code.

Every function a residual might have to differentiate through works
unchanged on NumPy arrays and on torch tensors.
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
    "price_profile", "evaluate_trajectory",
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
