r"""MP10.1A - The full model, where the SPMe stops. An energy-format LFP cell
discharged at 0.5C to 6C: the Doyle-Fuller-Newman model (DFN), electrolyte and
particles coupled. The physics, the reference model and the ground truth.

*Deep Learning for Engineering* - MSc, Aalborg University.
Remus Teodorescu (ret@et.aau.dk), with support from Research Assistant
Noman Khan (nomank@energy.aau.dk).

This module extends Ex10.1's ``problem.py`` (course policy C10 rule 1) and is
used by the notebook of MP10.1A (course policy C14). It needs PyBaMM
(``pip install pybamm``), which the notebook's first cell installs on Colab.

## What changes against Ex10.1, and why

Ex10.1 solved one graphite particle with a constant flux through its surface:
the single particle model, whose concentration has an exact series. That
model assumes the electrolyte is perfectly mixed and conducts perfectly, so
that every particle in an electrode carries the same current. Here the
electrolyte is resolved: the DFN couples the lithium concentration in the
electrolyte across the cell, c_e(x, t), the potentials of the electrolyte and
the solid, the reaction rate at every place in the electrodes and the
concentration in a particle at every place, c_s(x, r, t). The coupling is
nonlinear (Butler-Volmer kinetics, a concentration-dependent electrolyte
conductivity) and **no exact solution exists**.

The cell is PyBaMM's ``Prada2013`` parameter set (the A123 26650 LFP/graphite
cell, Prada et al., J. Electrochem. Soc. 160 (2013) A616; its open-circuit
potentials are Afshar 2017 for LFP and Chen 2020 for graphite, as PyBaMM
ships them) with both electrode thicknesses multiplied by three, to stand for
an energy-format LFP cell of 6.9 Ah nominal (ASSUMED: the tripling is a
device, not a datasheet). With thin electrodes the single particle model with
electrolyte (SPMe) stays within a few millivolts of the DFN even at 10C; with
thick ones the electrolyte runs out near the positive current collector at
high rates, the SPMe's linearised electrolyte fails, and only the DFN holds.

**The strength of the change** is how slowly the electrolyte transports
lithium: :func:`check_zero_strength` multiplies the electrolyte's diffusivity
and conductivity and the solids' conductivities by 10^3 (strength 1e-3), so
that every particle carries the same flux, and the DFN's negative particle
must then reproduce Ex10.1's exact series for a sphere at constant flux.

## Three solvers

* the **ground truth** (:func:`ground_truth`): PyBaMM's DFN on a mesh eight
  times finer than its default (160 points across each electrode and the
  separator, 160 in each particle's radius), solver tolerances 1e-9; checked
  on meshes x1, x2, x4, x8 with Richardson extrapolation of the capacity.
* the **reference model** (:func:`reference`): PyBaMM's DFN on its default
  mesh (20 points in each region and radius), built once, then solved and
  timed for each new C-rate - what an engineer runs for a new rate.
* the **training data** (:func:`simulate`): the DFN on a mesh twice as fine as
  the default, at the rates the student chooses.
"""

from __future__ import annotations

import time

import numpy as np

import problem as pb           # Ex10.1's module: Faraday's constant and the exact particle series

try:                           # PyBaMM is needed to solve the DFN; the notebook installs it on Colab
    import pybamm
except ImportError:            # pragma: no cover
    pybamm = None

# ------------------------------------------------------------------ the data
FARADAY = pb.FARADAY             #: C/mol
THICKEN = 3.0                    #: both electrode thicknesses x 3: the energy-format cell   [ASSUMED]
CAP_NOM = 2.3 * THICKEN          #: Ah, nominal: the A123's 2.3 Ah x 3 = 6.9 Ah
V_CUT = 2.0                      #: V, the lower cut-off voltage (Prada2013)
C_RANGE = (0.5, 6.0)             #: the C-rates the network must cover (1C = 6.9 A)
C_TEST = (1.0, 3.0, 5.0)         #: the rates the network is tested at, never trained on
C_GAP = 0.25                     #: no training discharge within this many C of a test rate
WINDOW = 0.95                    #: the voltage is scored up to this share of the ground truth's capacity
V_TOL = {1.0: 0.020, 3.0: 0.020, 5.0: 0.050}   #: V, the agreed voltage accuracy at each test rate
CAP_TOL = 0.02                   #: the agreed accuracy of the 5C capacity, relative
CE_TOL = 30.0                    #: mol/m^3, the agreed accuracy of the electrolyte concentration (2.5 % of 1200)
CE_SHARES = (0.25, 0.5, 0.75, 0.95)   #: shares of the capacity at which the electrolyte is scored

REFINE_REF = 1                   #: the reference model's mesh: PyBaMM's default
REFINE_DATA = 2                  #: the training data's mesh: twice as fine
REFINE_TRUTH = 8                 #: the ground truth's mesh
TOL_TRUTH = 1e-9                 #: the ground truth's solver tolerances (relative and absolute)


def parameters(current=None, fast_electrolyte=None):
    """PyBaMM's Prada2013 with both electrode thicknesses x 3. The current [A]
    is an input of the built model unless given; ``fast_electrolyte`` (a
    factor) multiplies the electrolyte's diffusivity and conductivity and the
    solids' conductivities, for the zero-strength check."""
    p = pybamm.ParameterValues("Prada2013")
    p["Negative electrode thickness [m]"] *= THICKEN
    p["Positive electrode thickness [m]"] *= THICKEN
    p["Current function [A]"] = "[input]" if current is None else current
    if fast_electrolyte:
        k = fast_electrolyte
        D, kap = p["Electrolyte diffusivity [m2.s-1]"], p["Electrolyte conductivity [S.m-1]"]
        p["Electrolyte diffusivity [m2.s-1]"] = D * k
        p["Electrolyte conductivity [S.m-1]"] = (lambda c, T, f=kap: f(c, T) * k) if callable(kap) else kap * k
        for e in ("Negative", "Positive"):
            p[f"{e} electrode conductivity [S.m-1]"] *= k
    return p


_P = None


def _values():
    global _P
    if _P is None:
        _P = parameters(current=1.0)
    return _P


def constants():
    """The cell's geometry and electrolyte, from the parameter set: a dict."""
    p = _values()
    L_n = p["Negative electrode thickness [m]"]; L_s = p["Separator thickness [m]"]; L_p = p["Positive electrode thickness [m]"]
    area = p["Electrode height [m]"] * p["Electrode width [m]"]
    out = dict(L_n=L_n, L_s=L_s, L_p=L_p, L=L_n + L_s + L_p, area=area,
               eps_n=p["Negative electrode porosity"], eps_s=p["Separator porosity"], eps_p=p["Positive electrode porosity"],
               brug=p["Separator Bruggeman coefficient (electrolyte)"],
               D_e=p["Electrolyte diffusivity [m2.s-1]"], c_e0=p["Initial concentration in electrolyte [mol.m-3]"],
               t_plus=p["Cation transference number"],
               R_n=p["Negative particle radius [m]"], R_p=p["Positive particle radius [m]"],
               D_n=p["Negative particle diffusivity [m2.s-1]"], D_p=p["Positive particle diffusivity [m2.s-1]"])
    for e, s in (("Negative", "n"), ("Positive", "p")):
        cmax = p[f"Maximum concentration in {e.lower()} electrode [mol.m-3]"]
        frac = p[f"{e} electrode active material volume fraction"]
        L = out[f"L_{s}"]
        out[f"cmax_{s}"] = cmax
        out[f"frac_{s}"] = frac
        out[f"sto0_{s}"] = p[f"Initial concentration in {e.lower()} electrode [mol.m-3]"] / cmax
        out[f"Q_{s}"] = FARADAY * cmax * frac * L * area / 3600.0     # Ah per unit of stoichiometry
    return out


K = None


def _k():
    global K
    if K is None:
        K = constants()
    return K


# --------------------------------------------- the open-circuit voltage (lithium balance)
def u_negative(sto, lib=np):
    """Graphite's open-circuit potential [V] against its stoichiometry: Chen et al.,
    J. Electrochem. Soc. 167 (2020) 080534, as PyBaMM's Prada2013 uses it. ``lib`` is
    numpy or torch."""
    return (1.9793 * lib.exp(-39.3631 * sto) + 0.2482 - 0.0909 * lib.tanh(29.8538 * (sto - 0.1234))
            - 0.04478 * lib.tanh(14.9159 * (sto - 0.2769)) - 0.0205 * lib.tanh(30.4444 * (sto - 0.6103)))


def u_positive(sto, lib=np):
    """LFP's open-circuit potential [V] against its stoichiometry: Afshar et al.,
    arXiv:1709.03970 (2017), as PyBaMM's Prada2013 uses it."""
    return 3.4077 - 0.020269 * sto + 0.5 * lib.exp(-150 * sto) - 0.9 * lib.exp(-30 * (1 - sto))


def stoichiometries(q):
    """The mean stoichiometry of each electrode after q [Ah] discharged: the
    lithium balance (what leaves the graphite enters the LFP), exact."""
    k = _k()
    return k["sto0_n"] - np.asarray(q) / k["Q_n"], k["sto0_p"] + np.asarray(q) / k["Q_p"]


def ocv(q, lib=np):
    """The open-circuit voltage [V] at the mean stoichiometries after q [Ah]
    discharged: what the cell would read if it rested with every particle mixed."""
    k = _k()
    return (u_positive(k["sto0_p"] + q / k["Q_p"], lib) - u_negative(k["sto0_n"] - q / k["Q_n"], lib))


_LAMBDA = pb._ROOTS[:40]           # the roots of tan(x) = x, from Ex10.1's exact series


def surface_shift(t, C, electrode="n", lib=np):
    """How far a particle's surface stoichiometry has moved ahead of its mean
    after t [s] of a discharge at C, if every particle of the electrode
    carries the same flux: Ex10.1's exact series for a sphere at constant flux,
    at its surface, minus its mean (``(jR/D)(1/5 - 2 sum exp(-l^2 D t/R^2)/l^2)``
    over c_max). Positive: the surface is emptier (negative electrode) or
    fuller (positive electrode) than the mean."""
    k = _k()
    R, D, frac, L = k[f"R_{electrode}"], k[f"D_{electrode}"], k[f"frac_{electrode}"], k[f"L_{electrode}"]
    j = C * CAP_NOM / (FARADAY * (3 * frac / R) * L * k["area"])   # mol/(m^2 s), the flux through each particle's surface
    tt = t * D / R ** 2
    lam = _LAMBDA if lib is np else lib.tensor(_LAMBDA, dtype=t.dtype, device=t.device)
    series = (lib.exp(-(lam ** 2) * tt[..., None]) / lam ** 2).sum(-1)
    return (j * R / D) * (0.2 - 2 * series) / k[f"cmax_{electrode}"]


def ocv_surface(q, C, lib=np):
    """The open-circuit voltage [V] at the particles' SURFACE stoichiometries
    after q [Ah] discharged at C, if every particle carries the same flux:
    the single particle model's voltage without kinetics or electrolyte. The
    mean stoichiometries are the lithium balance; the surfaces run ahead of
    them by Ex10.1's exact series (:func:`surface_shift`)."""
    k = _k()
    t = q * 3600.0 / (C * CAP_NOM)
    sn = k["sto0_n"] - q / k["Q_n"] - surface_shift(t, C, "n", lib)
    sp = k["sto0_p"] + q / k["Q_p"] + surface_shift(t, C, "p", lib)
    return u_positive(sp, lib) - u_negative(sn, lib)


# ------------------------------------------------------------------ the DFN
class DFN:
    """PyBaMM's DFN (or SPMe) of the energy-format LFP, built once on a mesh
    ``refine`` times PyBaMM's default, the current an input: ``discharge(C)``
    solves one constant-current discharge to the cut-off."""

    def __init__(self, refine=1, model="DFN", tol=1e-6, fast_electrolyte=None):
        m = {"DFN": pybamm.lithium_ion.DFN, "SPMe": pybamm.lithium_ion.SPMe, "SPM": pybamm.lithium_ion.SPM}[model]()
        pts = {k: v * refine for k, v in m.default_var_pts.items()}
        self.refine, self.model = refine, model
        self.sim = pybamm.Simulation(m, parameter_values=parameters(fast_electrolyte=fast_electrolyte), var_pts=pts,
                                     solver=pybamm.IDAKLUSolver(rtol=tol, atol=tol))
        self.sim.build()

    def solve(self, C):
        """The raw PyBaMM solution of a discharge at C [1/h] to the cut-off."""
        return self.sim.solve([0.0, 3600.0 / C], inputs={"Current function [A]": C * CAP_NOM})

    def discharge(self, C, fields=True):
        """One discharge at C: t [s], q [Ah], V [V], the capacity [Ah] and the
        solve time [s]; with ``fields`` also x [m] and c_e [mol/m^3], (x, t)."""
        t0 = time.perf_counter()
        sol = self.solve(C)
        dt = time.perf_counter() - t0
        out = dict(C=C, t=sol["Time [s]"].entries, q=sol["Discharge capacity [A.h]"].entries,
                   V=sol["Voltage [V]"].entries, seconds=dt)
        out["cap"] = float(out["q"][-1])
        if fields:
            out["x"] = sol["x [m]"].entries[:, 0]
            out["c_e"] = sol["Electrolyte concentration [mol.m-3]"].entries
            out["li_electrolyte"] = sol["Total lithium in electrolyte [mol]"].entries
            out["li_particles"] = sol["Total lithium in particles [mol]"].entries
            out["li_negative"] = sol["Total lithium in negative electrode [mol]"].entries
        return out


# ------------------------------------------------------- the reference model
_REF = None


def reference_model():
    """The reference model, built once: the DFN on PyBaMM's default mesh."""
    global _REF
    if _REF is None:
        _REF = DFN(REFINE_REF)
    return _REF


def reference(C):
    """The reference model's answer for a new C-rate: the discharge curve
    (q [Ah], V [V]) and the capacity [Ah], by the DFN on the default mesh."""
    r = reference_model().discharge(C, fields=False)
    return r["q"], r["V"], r["cap"]


# --------------------------------------------------------- the training data
_DATA = None


def simulate(rates, refine=REFINE_DATA, verbose=True):
    """Discharges at the given C-rates on the training mesh, for the network:
    a list of dicts (see :meth:`DFN.discharge`). Refuses a rate within
    ``C_GAP`` of a test rate or outside ``C_RANGE``."""
    global _DATA
    for C in rates:
        if min(abs(C - c) for c in C_TEST) < C_GAP:
            raise ValueError(f"{C}C is within {C_GAP}C of a test rate {C_TEST}: not allowed for training")
        if not C_RANGE[0] - 1e-9 <= C <= C_RANGE[1] + 1e-9:
            raise ValueError(f"{C}C is outside the range {C_RANGE}")
    if _DATA is None or _DATA.refine != refine:
        _DATA = DFN(refine)
    out = []
    t0 = time.perf_counter()
    for C in rates:
        out.append(_DATA.discharge(C))
        if verbose:
            print(f"   {C:.2f}C: capacity {out[-1]['cap']:.2f} Ah, {len(out[-1]['t'])} instants")
    if verbose:
        print(f"   {len(rates)} discharges in {time.perf_counter() - t0:.1f} s")
    return out


def field_at(d, shares):
    """The electrolyte concentration [mol/m^3] of a discharge ``d`` at the given
    shares of its capacity: (len(shares), nx), linear in time between the
    solver's instants."""
    qs = np.asarray(shares) * d["cap"]
    return np.array([[np.interp(qq, d["q"], row) for row in d["c_e"]] for qq in qs])


# -------------------------------------------------------- the ground truth
def convergence_study(C=5.0, refines=(1, 2, 4, REFINE_TRUTH), verbose=True):
    """The capacity and the voltage at 50 % and 90 % of the finest mesh's
    capacity, on meshes each twice as fine, at C. Returns rows (refine,
    points per region, capacity [Ah], V at 50 % [V], V at 90 % [V], seconds)."""
    runs = []
    for r in refines:
        d = DFN(r, tol=TOL_TRUTH).discharge(C, fields=False)
        runs.append((r, d))
    cap_f = runs[-1][1]["cap"]
    rows = []
    for r, d in runs:
        v50, v90 = np.interp([0.5 * cap_f, 0.9 * cap_f], d["q"], d["V"])
        rows.append((r, 20 * r, d["cap"], v50, v90, d["seconds"]))
        if verbose:
            print(f"   mesh x{r}: {20 * r:4d} points per region and radius, capacity {d['cap']:.4f} Ah, "
                  f"voltage at 50 % {v50:.4f} V and at 90 % {v90:.4f} V ({d['seconds']:.1f} s)")
    caps = np.array([r[2] for r in rows])
    dc = np.diff(caps)
    order = np.log2(np.abs(dc[:-1] / dc[1:]))
    rich = caps[-1] + dc[-1] / (2 ** order[-1] - 1)
    if verbose:
        print("   capacity change between meshes [Ah]:", " ".join(f"{x:+.5f}" for x in dc))
        print("   observed order:", " ".join(f"{x:.2f}" for x in order),
              f"| Richardson {rich:.4f} Ah; finest mesh off by {abs(caps[-1] - rich) / rich * 100:.3f} %")
    return rows, order, rich


_TRUTH = None


def ground_truth(C):
    """The ground truth's discharge at C: the DFN on the x8 mesh, tolerances 1e-9
    (the model built once)."""
    global _TRUTH
    if _TRUTH is None:
        _TRUTH = DFN(REFINE_TRUTH, tol=TOL_TRUTH)
    return _TRUTH.discharge(C)


def balances(d):
    """Conservation in a discharge ``d``: the largest relative change of the
    lithium in the electrolyte and in the particles, and the largest
    difference between the lithium that left the negative electrode and the
    charge passed (Faraday), relative."""
    e = np.abs(d["li_electrolyte"] / d["li_electrolyte"][0] - 1).max()
    s = np.abs(d["li_particles"] / d["li_particles"][0] - 1).max()
    left = d["li_negative"][0] - d["li_negative"]
    passed = d["q"] * 3600.0 / FARADAY
    f = np.abs(left[1:] - passed[1:]).max() / passed[-1]
    return float(e), float(s), float(f)


def check_zero_strength(C=1.0, factor=1e3, refines=(2, 4, REFINE_TRUTH), verbose=True):
    """At zero strength - the electrolyte and the solids conducting 10^3
    times better - every negative particle carries the same flux, and the DFN's
    surface concentration must follow Ex10.1's exact series for a sphere at a
    constant flux (problem.exact). Scored from 1 % of the discharge on: at the
    first instant the exact surface concentration falls with an infinite
    slope, which no mesh follows. Returns, for each mesh, the largest
    difference relative to the drop of the surface concentration, and the
    spread of the particles across the electrode, relative to the same drop."""
    k = _k()
    I = C * CAP_NOM
    a = 3 * k["frac_n"] / k["R_n"]                                    # 1/m, particle surface per electrode volume
    j = I / (FARADAY * a * k["L_n"] * k["area"])                       # mol/(m^2 s), the flux out of every particle
    out = []
    for r in refines:
        sol = DFN(r, tol=1e-8, fast_electrolyte=factor).solve(C)
        t = sol["Time [s]"].entries
        cs = sol["X-averaged negative particle surface concentration [mol.m-3]"].entries
        spread = sol["Negative particle surface concentration [mol.m-3]"].entries
        t_end = t[-1]
        c_ref = j * t_end / k["R_n"]                                  # Ex10.1's concentration scale
        D_hat = k["D_n"] * t_end / k["R_n"] ** 2                      # Ex10.1's scaled diffusivity
        ex = k["sto0_n"] * k["cmax_n"] - c_ref * pb.exact(1.0, t / t_end, D_hat)
        drop = cs[0] - cs[-1]
        late = t >= 0.01 * t_end
        err = float(np.abs(cs - ex)[late].max() / drop)
        sp = float((spread.max(0) - spread.min(0)).max() / drop)
        out.append((r, err, sp))
        if verbose:
            print(f"   strength 1e-3, mesh x{r}, {C:.0f}C: the surface concentration falls {drop:.0f} mol/m^3; the DFN "
                  f"against Ex10.1's exact series differs by at most {err:.1e} of that; the particles across the "
                  f"electrode differ by {sp:.1e}")
    return out


def spme(C):
    """The single particle model with electrolyte, on the default mesh: q [Ah], V [V], capacity [Ah]."""
    r = DFN(1, model="SPMe").discharge(C, fields=False)
    return r["q"], r["V"], r["cap"]


def shade_regions(ax, label=True):
    """Shade the negative electrode, the separator and the positive electrode
    on an axis whose x is the position through the cell in micrometres."""
    k = _k()
    a, b, c = k["L_n"] * 1e6, (k["L_n"] + k["L_s"]) * 1e6, k["L"] * 1e6
    for x0, x1, col, name in ((0, a, "0.85", "graphite"), (a, b, "#c3d6e8", "separator"), (b, c, "#fbdbb5", "LiFePO4")):
        ax.axvspan(x0, x1, color=col, alpha=0.5, lw=0)
        if label:
            ax.text((x0 + x1) / 2, 1.0, name, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8)
    ax.set_xlim(0, c)


# ------------------------------------------------------------- the scoring
def voltage_error(q_net, V_net, q_true, V_true, cap_true):
    """The largest |V_net - V_true| [V] at the same discharged charge, up to
    WINDOW of the ground truth's capacity (400 points)."""
    q = np.linspace(0.0, WINDOW * cap_true, 400)
    return float(np.abs(np.interp(q, q_net, V_net) - np.interp(q, q_true, V_true)).max())


def crossing(q, V, v_cut=V_CUT):
    """The discharged charge [Ah] where a curve V(q) first falls to the cut-off,
    by linear interpolation; the last q if it never does."""
    q, V = np.asarray(q, float).ravel(), np.asarray(V, float).ravel()
    below = np.nonzero(V <= v_cut)[0]
    if len(below) == 0 or below[0] == 0:
        return float(q[-1])
    i = below[0]
    return float(q[i - 1] + (q[i] - q[i - 1]) * (V[i - 1] - v_cut) / (V[i - 1] - V[i]))


TRUTH_FILE = "MP10.1A_truth.npz"


def write_truth(path=TRUTH_FILE):
    """The ground truth at 1C, 3C and 5C (curves and electrolyte fields), the
    SPMe beside it, the convergence table, the balances and the zero-strength
    check."""
    t0 = time.perf_counter()
    rows, order, rich = convergence_study(5.0)
    out = {}
    for C in C_TEST:
        d = ground_truth(C)
        tag = f"{C:.0f}C"
        e, s, f = balances(d)
        x8 = d["x"]
        out.update({f"t_{tag}": d["t"], f"q_{tag}": d["q"], f"V_{tag}": d["V"], f"cap_{tag}": np.array(d["cap"]),
                    f"ce_{tag}": field_at(d, CE_SHARES), f"balance_{tag}": np.array([e, s, f])})
        qs, Vs, cs = spme(C)
        out.update({f"spme_q_{tag}": qs, f"spme_V_{tag}": Vs, f"spme_cap_{tag}": np.array(cs)})
        print(f"   {tag}: capacity {d['cap']:.4f} Ah; electrolyte lithium kept to {e:.1e}, particles' to {s:.1e}, "
              f"Faraday to {f:.1e}; SPMe capacity {cs:.4f} Ah")
    out["x"] = x8
    out["ce_shares"] = np.array(CE_SHARES)
    z = check_zero_strength(verbose=True)
    out["zero_strength_refine"] = np.array([r[0] for r in z])
    out["zero_strength_error"] = np.array([r[1] for r in z])
    out["zero_strength_spread"] = np.array([r[2] for r in z])
    out["conv_refine"] = np.array([r[0] for r in rows]); out["conv_points"] = np.array([r[1] for r in rows])
    out["conv_cap"] = np.array([r[2] for r in rows]); out["conv_V50"] = np.array([r[3] for r in rows])
    out["conv_V90"] = np.array([r[4] for r in rows]); out["conv_seconds"] = np.array([r[5] for r in rows])
    out["conv_order"] = order; out["conv_richardson_cap"] = np.array(rich)
    k = _k()
    out["params"] = np.array([f"{a} = {b}" for a, b in dict(
        CELL="PyBaMM Prada2013 (A123 26650 LFP/graphite, Prada et al. 2013) with both electrode thicknesses x 3 (ASSUMED)",
        CAPACITY="6.9 Ah nominal (2.3 Ah x 3); 1C = 6.9 A", CUT_OFF="2.0 V", TEMPERATURE="298 K, isothermal",
        THICKNESS=f"negative {k['L_n']*1e6:.0f} um, separator {k['L_s']*1e6:.0f} um, positive {k['L_p']*1e6:.0f} um",
        ELECTROLYTE=f"{k['c_e0']:.0f} mol/m3 initially, D_e {k['D_e']:.1e} m2/s, t+ {k['t_plus']}",
        OCP="graphite Chen 2020, LFP Afshar 2017 (as PyBaMM ships Prada2013)",
        TRUTH=f"DFN on mesh x{REFINE_TRUTH} ({20 * REFINE_TRUTH} points per region and radius), tolerances {TOL_TRUTH}",
        PYBAMM=pybamm.__version__).items()])
    out["README"] = np.array(
        "MP10.1A ground truth: constant-current discharges of the energy-format LFP (Prada2013, electrode thicknesses x3) "
        "to 2.0 V by PyBaMM's DFN on mesh x8. For tag in 1C, 3C, 5C: t_<tag> [s], q_<tag> discharged charge [Ah], "
        "V_<tag> terminal voltage [V], cap_<tag> capacity [Ah], ce_<tag>[i, j] electrolyte concentration [mol/m3] at "
        "ce_shares[i] of the capacity and position x[j] [m] (0 at the negative current collector), balance_<tag>: largest "
        "relative change of lithium in the electrolyte and in the particles, and the Faraday balance of the negative "
        "electrode; spme_*: the SPMe on its default mesh. conv_*: 5C on meshes x1, x2, x4, x8 (refine, points per region, "
        "capacity Ah, V at 50 and 90 % of the finest capacity, solve s), conv_order, conv_richardson_cap [Ah]. "
        "zero_strength_*: electrolyte and solids 1e3 times faster at 1C on meshes zero_strength_refine: the negative "
        "particles' surface concentration against Ex10.1's exact series from 1 % of the discharge on (error) and the "
        "particles' spread across the electrode (spread), both relative to the drop of the surface concentration.")
    np.savez_compressed(path, **out)
    print(f"written {path} ({time.perf_counter() - t0:.0f} s)")


if __name__ == "__main__":
    import sys
    print("zero-strength check")
    check_zero_strength()
    print("convergence at 5C")
    convergence_study(5.0)
    if "--save" in sys.argv:
        write_truth()
