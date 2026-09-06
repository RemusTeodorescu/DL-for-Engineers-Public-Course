r"""Ex_10.1 — the physics: a lithium-ion cell, from one particle to a hot can.

*Deep Learning for Engineering* — MSc, Aalborg University.
Remus Teodorescu (ret@et.aau.dk), with support from Research Assistant
Noman Khan (nomank@energy.aau.dk).

**Reference texts.** Liu, *PINN with Python: An Introduction* (2025); Raissi,
Perdikaris & Karniadakis, *Physics-informed neural networks*, J. Comput. Phys.
**378** (2019) 686–707. These are the works to read for the theory. The code,
the problems and the exposition here are original to this course, written from
the 2019 paper and the PyTorch documentation and not derived from any
publisher's code listings.

Physics from the Doyle-Fuller-Newman framework; thermal coupling and the heat
source decomposition from Gu & Wang (2000). Parameters and reference solutions
come from PyBaMM (LG M50 21700: Chen2020 isothermal, ORegan2022 thermal).

Everything is non-dimensional inside the solver. Raw SI values span roughly
twenty orders of magnitude - particle radii of 1e-6 m, diffusivities of 1e-14
m^2/s, concentrations of 1e4 mol/m^3 - and no loss weighting substitutes for
scaling them.

## 1 · One particle — parabolic, radial, and singular at the centre

The single-particle model reduces a whole electrode to one representative
sphere. In the dimensionless variables that is

    c_t = c_rr + (2/r) c_r,   c_r(0, t) = 0,  c_r(1, t) = 1,  c(r, 0) = 0

Two things separate this from every problem in Ex_07. **The domain is radial**:
one spatial coordinate, and the Laplacian carries the curvature term ``2/r``
which is singular at the centre. Sample away from ``r = 0`` or handle it
explicitly — it is a real numerical trap, not a formality. **And the boundary
condition at the surface is on the flux, not the value**: the current pulls
lithium out through the surface at a fixed rate, and nothing fixes the
concentration anywhere.

:func:`analytic_sphere` is the classical series solution for exactly this
problem (Crank, *Mathematics of Diffusion*). It needs no PyBaMM and no
training, so a student can confirm a residual before trusting anything else.

**Why some samplers live here and not in ``pinn_core``.** The shared library
samples rectangles: ``interior_points``, ``boundary_points`` and their
space-time relatives know about boxes and nothing else. The (r, t) slab *is* a
rectangle, so :func:`particle_points` is a thin wrapper over
``interior_points`` — but the edges of that rectangle are not four
interchangeable walls. One is the centre of a sphere, one is the reacting
surface, one is an instant in time; each carries a different kind of condition
and the centre one has to be held off the singularity. Those, and the series
reference solution for the sphere, are geometry, which means they belong to the
problem and not to the library.

## 2 · The whole cell — SPMe, and where the SPM stops being honest

The SPM assumes a uniform reaction rate and a uniform electrolyte. Both
assumptions fail as C-rate rises, and the SPMe differs from the SPM by exactly
the electrolyte physics that was dropped. :func:`pybamm_reference` supplies
both, so the failure can be measured in millivolts rather than argued about.

## 3 · The cell heats up — Gu & Wang's three sources

    q = a_s i_n eta  +  a_s i_n T dU/dT  +  sigma|grad phi_s|^2 + kappa|grad phi_e|^2
        \___________/    \______________/    \_________________________________/
        irreversible       reversible                     ohmic

Only the middle term changes sign with the current direction. Gu & Wang had to
neglect it for want of data; ORegan2022 measured it, so it can be restored
here. :func:`heat_source_terms` returns the three separately, because the
question worth asking is which one dominates.
"""

from __future__ import annotations

import time

import numpy as np
import torch
import matplotlib.pyplot as plt

from pinn_core import (SEED, MLP, describe, set_seed, to_tensor, to_numpy,
                       interior_points, train_two_stage, relative_l2,
                       max_abs_error)

__all__ = [
    "F_CONST", "R_GAS", "CellParams",
    "sphere_roots", "analytic_sphere",
    "R_CENTRE_EPS", "particle_points", "particle_surface_points",
    "particle_centre_points", "particle_initial_points",
    "ocp_negative", "ocp_positive", "butler_volmer_eta", "theta_from_c",
    "heat_source_terms", "arrhenius",
    "pybamm_reference", "PYBAMM_OK",
    "run_particle", "plot_particle", "lumped_temperature", "plot_rz_field",
    "mlp_parameter_count", "control_panel", "make_report",
]

F_CONST = 96485.0          # Faraday, C/mol
R_GAS = 8.314              # J/mol/K


# ══════════════════════════════════════════════════════════════════════════
# 1 · the cell
# ══════════════════════════════════════════════════════════════════════════

class CellParams:
    """LG M50 21700, order-of-magnitude values for scaling and defaults.

    Use PyBaMM for authoritative numbers; these exist so the notebooks run
    even when PyBaMM is unavailable, and so students can see the scales.
    """

    def __init__(self, c_rate=1.0, T_amb=298.15, soc0=1.0, h_cool=10.0):
        self.capacity = 5.0            # Ah, nominal
        self.c_rate = float(c_rate)
        self.T_amb = float(T_amb)      # K
        self.soc0 = float(soc0)
        self.h_cool = float(h_cool)    # W/m^2/K
        # geometry
        self.R_cell, self.H_cell = 0.0105, 0.070     # m, 21700
        self.Rs_n, self.Rs_p = 5.86e-6, 5.22e-6      # m, particle radii
        # transport
        self.Ds_n, self.Ds_p = 3.3e-14, 4.0e-15      # m^2/s
        self.cs_max_n, self.cs_max_p = 33133.0, 63104.0   # mol/m^3
        # thermal
        self.rho_cp = 2.5e6            # J/m^3/K
        self.lam_r, self.lam_z = 0.9, 30.0            # W/m/K, anisotropic
        self.E_act_Ds = 30000.0        # J/mol, Arrhenius activation energy

    @property
    def current(self):
        """Applied current in amperes."""
        return self.c_rate * self.capacity

    @property
    def t_discharge(self):
        """Nominal discharge duration in seconds."""
        return 3600.0 / max(self.c_rate, 1e-9)

    @property
    def tau_diff_n(self):
        """Particle diffusion time, Rs^2/Ds. The natural timescale of the SPM."""
        return self.Rs_n ** 2 / self.Ds_n

    @property
    def biot(self):
        """Bi = hR/lambda. Small means a lumped thermal model is honest."""
        return self.h_cool * self.R_cell / self.lam_r

    def __repr__(self):
        return (f"CellParams(C-rate={self.c_rate:g}, T_amb={self.T_amb - 273.15:.0f}C, "
                f"SOC0={self.soc0:.2f}, h={self.h_cool:g}, Bi={self.biot:.3f})")


# ══════════════════════════════════════════════════════════════════════════
# 2 · the analytic sphere solution
# ══════════════════════════════════════════════════════════════════════════

def sphere_roots(n=40):
    """Roots of tan(L) = L, needed by the constant-flux sphere solution.

    Bisection rather than scipy, to keep the dependency list to torch, numpy
    and matplotlib.
    """
    out = []
    k = 1
    while len(out) < n:
        lo, hi = k * np.pi + 1e-9, (k + 0.5) * np.pi - 1e-9
        f = lambda L: np.tan(L) - L
        a, b = f(lo), f(hi)
        if a * b < 0:
            for _ in range(200):
                m = 0.5 * (lo + hi)
                if f(lo) * f(m) <= 0:
                    hi = m
                else:
                    lo = m
            out.append(0.5 * (lo + hi))
        k += 1
    return np.array(out)


_ROOTS = sphere_roots(40)


def analytic_sphere(r, t, n_terms=40):
    """Dimensionless concentration in a sphere with unit constant surface flux.

    Solves  dc/dt = (1/r^2) d/dr (r^2 dc/dr),  dc/dr(0)=0,  dc/dr(1)=1,  c(r,0)=0.
    Classical result (Crank, *Mathematics of Diffusion*). Verified against the
    PDE to ~1e-6 by finite differences.

    This is the exercise's analytic check: it needs no PyBaMM and no training,
    so a student can confirm their residual before trusting anything else.
    """
    r = np.atleast_1d(np.asarray(r, float))
    t = np.atleast_1d(np.asarray(t, float))
    rr, tt = np.broadcast_arrays(r, t)
    safe = np.where(rr == 0.0, 1e-12, rr)
    s = np.zeros_like(safe)
    for ln in _ROOTS[:n_terms]:
        s += np.sin(ln * safe) / (ln ** 2 * np.sin(ln)) * np.exp(-ln ** 2 * tt)
    return 3.0 * tt + safe ** 2 / 2.0 - 0.3 - 2.0 * s / safe


# ══════════════════════════════════════════════════════════════════════════
# 3 · sampling the particle — the radial geometry the library does not know
# ══════════════════════════════════════════════════════════════════════════

#: The radius at which the centre condition is imposed. Not zero: the ``2/r``
#: term in the residual is singular there, and a boundary point sitting exactly
#: on a singularity produces a loss term that is infinite or NaN on the first
#: forward pass.
R_CENTRE_EPS = 1e-3


def particle_points(n, t_end=1.0, r_min=0.0, method="lhs", seed=None):
    """Collocation points in (r, t) on the unit sphere radius and time window.

    Returns ``(n, 2)`` NumPy, like every sampler in ``pinn_core`` — wrap it
    with ``to_tensor(..., requires_grad=True)`` before differentiating through
    it. Column 0 is r, column 1 is t.

    The (r, t) slab is a rectangle, so this is ``interior_points`` with a name
    that says what the columns mean. Raise ``r_min`` above zero if you would
    rather keep the collocation points off the centre than clamp the curvature
    term inside the residual — the two are different modelling choices and
    notebook 01 asks you to make one of them deliberately.

    The seed defaults to the course seed rather than to the system entropy, so
    an unseeded call is still reproducible.
    """
    return interior_points(n, ((r_min, 1.0), (0.0, float(t_end))),
                           method=method, seed=SEED if seed is None else seed)


def particle_surface_points(n=200, t_end=1.0):
    """``n`` points on r = 1, spanning the time window. The reacting surface.

    Where the flux condition ``c_r = 1`` lives — the current, arriving.
    """
    return np.column_stack([np.ones(n), np.linspace(0.0, t_end, n)])


def particle_centre_points(n=200, t_end=1.0, r_eps=R_CENTRE_EPS):
    """``n`` points just off the centre, spanning the time window.

    Where the symmetry condition ``c_r = 0`` lives. At ``r_eps`` and not at
    ``r = 0``; see :data:`R_CENTRE_EPS`.
    """
    return np.column_stack([np.full(n, r_eps), np.linspace(0.0, t_end, n)])


def particle_initial_points(n=200, r_eps=R_CENTRE_EPS):
    """``n`` points on the t = 0 slice, from ``r_eps`` to the surface."""
    return np.column_stack([np.linspace(r_eps, 1.0, n), np.zeros(n)])


# ══════════════════════════════════════════════════════════════════════════
# 4 · electrochemistry
# ══════════════════════════════════════════════════════════════════════════

def theta_from_c(c_surf, c_max):
    """Local state of charge from the surface concentration."""
    return c_surf / c_max


def ocp_negative(theta):
    """Graphite OCP, Doyle et al. (1996) Eq. 8 as quoted by Gu & Wang."""
    bk = torch if torch.is_tensor(theta) else np
    return 0.16 + 1.32 * bk.exp(-3.0 * theta) + 10.0 * bk.exp(-2000.0 * theta)


def ocp_positive(theta):
    """Manganese-oxide OCP, Doyle et al. (1996) Eq. 7 as quoted by Gu & Wang.

    Kept for continuity with the paper. For the LG M50 (NMC811) use the OCP
    supplied by the PyBaMM parameter set instead.
    """
    bk = torch if torch.is_tensor(theta) else np
    return (4.19829 + 0.0565661 * bk.tanh(-14.5546 * theta + 8.60942)
            - 0.157123 * bk.exp(-0.04738 * theta ** 8)
            + 0.810239 * bk.exp(-40.0 * (theta - 0.133875)))


def butler_volmer_eta(i_n, i_0, T, alpha=0.5):
    """Overpotential from a symmetric Butler-Volmer law, inverted analytically.

    For alpha_a = alpha_c = 0.5 the law inverts to an arcsinh, which is far
    better conditioned than solving the exponential form numerically - and it
    cannot overflow during early training.
    """
    bk = torch if torch.is_tensor(i_n) else np
    arg = i_n / (2.0 * i_0)
    asinh = bk.asinh if hasattr(bk, "asinh") else bk.arcsinh
    return (2.0 * R_GAS * T) / (alpha * 2.0 * F_CONST) * asinh(arg)


def arrhenius(phi_ref, E_act, T, T_ref=298.15):
    """Temperature dependence of a transport property, Gu & Wang Eq. 26."""
    bk = torch if torch.is_tensor(T) else np
    return phi_ref * bk.exp(E_act / R_GAS * (1.0 / T_ref - 1.0 / T))


def heat_source_terms(i_n, a_s, eta, T, dUdT, sigma_eff=None, grad_phi_s=None,
                      kappa_eff=None, grad_phi_e=None):
    """The three heat sources of Gu & Wang Eq. 25, returned separately.

    Returns (irreversible, reversible, ohmic).

    * irreversible  a_s i_n eta        always >= 0
    * reversible    a_s i_n T dU/dT    changes sign with the current
    * ohmic         sigma|grad phi_s|^2 + kappa|grad phi_e|^2

    Gu & Wang neglect the reversible term, stating that dU/dT was not known for
    lithium-ion systems and that the missing data is necessary for accurate
    local heat prediction. ORegan2022 measured it, so it can be restored here.
    """
    q_irr = a_s * i_n * eta
    q_rev = a_s * i_n * T * dUdT
    q_ohm = 0.0
    if sigma_eff is not None and grad_phi_s is not None:
        q_ohm = q_ohm + sigma_eff * (grad_phi_s ** 2)
    if kappa_eff is not None and grad_phi_e is not None:
        q_ohm = q_ohm + kappa_eff * (grad_phi_e ** 2)
    return q_irr, q_rev, q_ohm


# ══════════════════════════════════════════════════════════════════════════
# 5 · the PyBaMM reference
# ══════════════════════════════════════════════════════════════════════════

try:
    import pybamm                                    # noqa: F401
    PYBAMM_OK = True
except ImportError:
    PYBAMM_OK = False


def pybamm_reference(model="SPM", c_rate=1.0, parameter_set="Chen2020",
                     thermal=False, T_amb=298.15, t_eval=None):
    """Reference solution from PyBaMM. Returns a dict of numpy arrays.

    model          'SPM', 'SPMe' or 'DFN'
    parameter_set  'Chen2020' (isothermal) or 'ORegan2022' (has thermal data)
    thermal        add a lumped thermal model

    Chen2020 carries no thermal parameters - the LG M50 degradation study says
    so explicitly - so ask for thermal=True only with ORegan2022.
    """
    if not PYBAMM_OK:
        raise ImportError(
            "PyBaMM is not installed.  On Colab:  !pip install -q pybamm")
    import pybamm

    if thermal and parameter_set == "Chen2020":
        raise ValueError(
            "Chen2020 has no thermal parameters. Use parameter_set='ORegan2022' "
            "for thermal runs (see L10.1 slide 17).")

    options = {"thermal": "lumped"} if thermal else {}
    cls = {"SPM": pybamm.lithium_ion.SPM,
           "SPMe": pybamm.lithium_ion.SPMe,
           "DFN": pybamm.lithium_ion.DFN}[model]
    mdl = cls(options=options) if options else cls()

    params = pybamm.ParameterValues(parameter_set)
    params["Ambient temperature [K]"] = T_amb
    sim = pybamm.Simulation(mdl, parameter_values=params,
                            C_rate=c_rate) if hasattr(pybamm, "Simulation") else None
    if t_eval is None:
        t_eval = np.linspace(0, 3600.0 / max(c_rate, 1e-9), 300)
    sol = sim.solve(t_eval)

    out = {"t": sol["Time [s]"].entries,
           "V": sol["Terminal voltage [V]"].entries,
           "model": model, "parameter_set": parameter_set, "c_rate": c_rate}
    for name, key in [("T", "Volume-averaged cell temperature [K]"),
                      ("c_s_n_surf", "Negative particle surface concentration [mol.m-3]"),
                      ("c_s_p_surf", "Positive particle surface concentration [mol.m-3]"),
                      ("c_e", "Electrolyte concentration [mol.m-3]")]:
        try:
            out[name] = sol[key].entries
        except Exception:
            pass
    return out


# ══════════════════════════════════════════════════════════════════════════
# 6 · the lab — studies, plots and the report
#
# No physics below this line: the residuals and losses are written by the
# student in notebooks 01-03. This half runs studies, draws the r-z temperature
# field, and assembles the report.
# ══════════════════════════════════════════════════════════════════════════

def run_particle(cell, residual_fn, loss_fn_factory, n_coll=3000, n_hidden=32,
                 n_layers=4, adam=3000, lbfgs=300, t_end=1.0, verbose=True):
    """Train a PINN for the dimensionless particle-diffusion problem.

    ``loss_fn_factory(model, rt)`` returns the no-argument closure that
    ``train_two_stage`` calls; ``rt`` arrives as a tensor with
    ``requires_grad=True``, ready to differentiate through.
    """
    set_seed(88)
    model = MLP(n_in=2, n_hidden=n_hidden, n_layers=n_layers, n_out=1)
    rt = to_tensor(particle_points(n_coll, t_end=t_end), requires_grad=True)
    if verbose:
        describe(model, n_coll)
    t0 = time.time()
    hist = train_two_stage(model, loss_fn_factory(model, rt),
                           adam_steps=adam, lbfgs_steps=lbfgs, lr=1e-3,
                           report_every=750 if verbose else 0)
    wall = time.time() - t0

    rg = np.linspace(0.001, 1.0, 121)
    errs = []
    for t in (0.02, 0.1, 0.3, 0.6, 1.0):
        if t > t_end:
            continue
        P = to_tensor(np.column_stack([rg, np.full_like(rg, t)]))
        with torch.no_grad():
            c = to_numpy(model(P)).ravel()
        ex = analytic_sphere(rg, t)
        errs.append({"t": float(t),
                     "rel": relative_l2(c, ex),
                     "max": max_abs_error(c, ex),
                     "surf_err": float(abs(c[-1] - ex[-1]))})
    res = {"cell": cell, "model": model, "history": hist, "seconds": wall,
           "final_loss": float(hist["lbfgs"][-1]), "errors": errs,
           "mean_rel": float(np.mean([e["rel"] for e in errs])),
           "n_coll": n_coll, "n_hidden": n_hidden, "n_layers": n_layers}
    if verbose:
        print(f"\n{cell}\n  wall {wall:.1f}s  loss {res['final_loss']:.3e}  "
              f"mean rel L2 {res['mean_rel']:.3e}")
    return res


def plot_particle(result, times=(0.02, 0.1, 0.3, 0.6, 1.0)):
    """The trained profile against the analytic one, at several instants."""
    rg = np.linspace(0.001, 1.0, 200)
    plt.figure(figsize=(7.5, 3.6))
    for t in times:
        P = to_tensor(np.column_stack([rg, np.full_like(rg, t)]))
        with torch.no_grad():
            c = to_numpy(result["model"](P)).ravel()
        line, = plt.plot(rg, analytic_sphere(rg, t), lw=2, alpha=0.55)
        plt.plot(rg, c, "--", color=line.get_color(), label=f"t = {t}")
    plt.xlabel("r / R$_s$"); plt.ylabel("dimensionless concentration")
    plt.title("solid line: analytic     dashed: PINN")
    plt.legend(fontsize=8); plt.tight_layout(); plt.show()


def lumped_temperature(cell, q_total, t):
    """Lumped thermal response, for comparison with a distributed solution.

    rho c_p dT/dt = q - (hA/V)(T - T_amb).  Valid when the Biot number is
    small; print cell.biot before trusting it (L10.1 slide 14).
    """
    A_over_V = 2.0 / cell.R_cell + 2.0 / cell.H_cell
    tau = cell.rho_cp / (cell.h_cool * A_over_V)
    dT = q_total / (cell.h_cool * A_over_V)
    return cell.T_amb + dT * (1.0 - np.exp(-t / tau))


def plot_rz_field(cell, T_core_rise, k=(60, 90)):
    """Sketch the r-z temperature field of a cylindrical cell.

    A parabolic radial profile with an axial gradient towards the cooled tab -
    the qualitative picture of L10.1 slide 14. Replace with your trained field
    once notebook 03 works.
    """
    r = np.linspace(0, cell.R_cell, k[0])
    z = np.linspace(0, cell.H_cell, k[1])
    Rg, Zg = np.meshgrid(r, z)
    radial = 1.0 - (Rg / cell.R_cell) ** 2
    axial = 1.0 - 0.45 * (Zg / cell.H_cell)
    T = cell.T_amb + T_core_rise * radial * axial
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    im = ax[0].contourf(Rg * 1e3, Zg * 1e3, T - 273.15, 50, cmap="magma")
    ax[0].set_xlabel("r [mm]"); ax[0].set_ylabel("z [mm]")
    ax[0].set_title("temperature [degC]"); fig.colorbar(im, ax=ax[0])
    ax[1].plot(r * 1e3, T[len(z) // 2] - 273.15)
    ax[1].set_xlabel("r [mm]"); ax[1].set_ylabel("T [degC]")
    ax[1].set_title(f"radial profile at mid-height   (Bi = {cell.biot:.3f})")
    plt.tight_layout(); plt.show()
    return T


def mlp_parameter_count(n_hidden, n_layers, n_in=2, n_out=1):
    """Trainable parameters of the ``MLP`` the control panel would build.

    The same number ``parameter_count`` would return, computed from the widths
    alone so the readout can update on every slider move without constructing a
    network. The ratio that matters is collocation points per parameter, and
    ``describe`` prints the same one at the start of every run, so the panel and
    the training log cannot disagree. Below one the network has more parameters
    than the residual has sample points, which is where the panel warns.
    """
    widths = [n_in] + [n_hidden] * n_layers + [n_out]
    return sum((widths[i] + 1) * widths[i + 1] for i in range(len(widths) - 1))


def control_panel(on_run):
    """Sliders for the operating point and the numerics, plus a Run button."""
    import ipywidgets as W
    from IPython.display import display

    style = {"description_width": "160px"}; lay = W.Layout(width="420px")
    w = {
        "c_rate": W.FloatSlider(value=1.0, min=0.1, max=5.0, step=0.1,
                                description="C-rate", style=style, layout=lay),
        "T_amb": W.FloatSlider(value=25.0, min=-10.0, max=45.0, step=1.0,
                               description="Ambient temp [degC]", style=style, layout=lay),
        "soc0": W.FloatSlider(value=1.0, min=0.2, max=1.0, step=0.05,
                              description="Initial SOC", style=style, layout=lay),
        "h_cool": W.FloatLogSlider(value=10.0, base=10, min=-0.5, max=2.5, step=0.1,
                                   description="Cooling h [W/m2K]", style=style, layout=lay),
        "n_coll": W.IntSlider(value=3000, min=500, max=15000, step=500,
                              description="Collocation points", style=style, layout=lay),
        "n_hidden": W.IntSlider(value=32, min=10, max=80, step=2,
                                description="Neurons per layer", style=style, layout=lay),
        "n_layers": W.IntSlider(value=4, min=2, max=8, step=1,
                                description="Hidden layers", style=style, layout=lay),
        "adam": W.IntSlider(value=3000, min=500, max=10000, step=500,
                            description="Adam epochs", style=style, layout=lay),
    }
    readout = W.HTML()

    def _update(*_):
        cell = CellParams(w["c_rate"].value, w["T_amb"].value + 273.15,
                          w["soc0"].value, w["h_cool"].value)
        n_param = mlp_parameter_count(w["n_hidden"].value, w["n_layers"].value)
        ratio = w["n_coll"].value / n_param
        warn = ("" if ratio >= 1.0 else
                " &nbsp;<b style='color:#c60'>fewer points than parameters</b>")
        lump = "lumped OK" if cell.biot < 0.1 else "<b style='color:#c60'>gradient matters</b>"
        readout.value = (
            f"<div style='font-family:monospace'>I = {cell.current:.1f} A"
            f" &nbsp;|&nbsp; discharge {cell.t_discharge/60:.0f} min"
            f" &nbsp;|&nbsp; particle tau = {cell.tau_diff_n:.0f} s"
            f" &nbsp;|&nbsp; Bi = {cell.biot:.3f} ({lump})"
            f" &nbsp;|&nbsp; N/params = {ratio:.1f}{warn}</div>")
    for x in w.values():
        x.observe(_update, "value")
    _update()

    run = W.Button(description="Run study", button_style="success", icon="play",
                   layout=W.Layout(width="170px"))
    out = W.Output()

    def _click(_):
        with out:
            out.clear_output()
            cell = CellParams(w["c_rate"].value, w["T_amb"].value + 273.15,
                              w["soc0"].value, w["h_cool"].value)
            on_run(cell, {k: v.value for k, v in w.items()})

    run.on_click(_click)
    display(W.VBox([
        W.HTML("<h3>Ex_10.1 &mdash; LG M50 21700 control panel</h3>"
               "<p>Set the operating conditions and the numerics, then "
               "<b>Run study</b>. The readout shows the derived quantities that "
               "decide whether your model choices are sound.</p>"),
        W.HBox([W.VBox(list(w.values())[:4]), W.VBox(list(w.values())[4:])]),
        readout, run, out]))
    return w


def make_report(results, filename="Ex10.1_report.md", author="", notes=""):
    """Assemble the run table and the questions into a Markdown file."""
    L = ["# Ex_10.1 - Ionic diffusion, charge conservation and thermal coupling", ""]
    if author:
        L.append(f"**Author:** {author}  ")
    L += [f"**Runs recorded:** {len(results)}",
          f"**PyBaMM available:** {PYBAMM_OK}", "", "## Runs", "",
          "| # | C-rate | T_amb [degC] | SOC0 | h | Bi | N_f | net | wall (s) | final loss | mean rel L2 |",
          "|---|--------|--------------|------|---|----|-----|-----|----------|------------|-------------|"]
    for i, r in enumerate(results, 1):
        c = r["cell"]
        L.append(f"| {i} | {c.c_rate:g} | {c.T_amb-273.15:.0f} | {c.soc0:.2f} | "
                 f"{c.h_cool:g} | {c.biot:.3f} | {r['n_coll']} | "
                 f"{r['n_hidden']}x{r['n_layers']} | {r['seconds']:.1f} | "
                 f"{r['final_loss']:.3e} | {r['mean_rel']:.3e} |")
    if results and results[0].get("errors"):
        L += ["", "## Error against time", "",
              "| run | " + " | ".join(f"t={e['t']:g}" for e in results[0]["errors"]) + " |",
              "|---|" + "---|" * len(results[0]["errors"])]
        for i, r in enumerate(results, 1):
            L.append(f"| {i} | " + " | ".join(f"{e['rel']:.2e}" for e in r["errors"]) + " |")
    else:
        L += ["", "## Error against time", "",
              "_No runs recorded yet - run notebooks 01 and 04 first._"]
    L += ["", "## Your interpretation", "",
          "1. **Model choice.** At which C-rate did the SPM start to disagree with the",
          "   PyBaMM SPMe reference, and which physical effect was it missing?", "",
          "2. **Coupling.** Compare a run with temperature-dependent properties against",
          "   one with them frozen. Which way did the voltage move, and the temperature?",
          "   Does your result agree with Gu & Wang's coupled/decoupled comparison?", "",
          "3. **Heat sources.** Which of the three terms dominated, and which changed",
          "   sign? What happens to the reversible term on charge rather than discharge?", "",
          "4. **Thermal gradient.** What Biot number did you obtain, and was a lumped",
          "   model defensible? Where was the hot spot in the r-z plane?", "",
          "5. **Honest assessment.** PyBaMM solves all of this faster. State the case,",
          "   in one paragraph, for building the PINN anyway.", ""]
    if notes:
        L += ["## Notes", "", notes, ""]
    with open(filename, "w") as fh:
        fh.write("\n".join(L))
    print(f"wrote {filename}  ({len(results)} runs)")
    return filename
