r"""MP10.1B - How much has the cell aged? An aged LG M50 cell whose ageing is
seen only through the voltage of its discharges. The physics, the reference
model (least squares) and the ground truth.

*Deep Learning for Engineering* - MSc, Aalborg University.
Remus Teodorescu (ret@et.aau.dk), with support from Research Assistant
Noman Khan (nomank@energy.aau.dk).

This module extends Ex10.1's ``problem.py`` (course policy C10 rule 1) and is
used by the notebook of MP10.1B (course policy C14). It needs PyBaMM
(``pip install pybamm``), which the notebook's first cell installs on Colab.

## What changes against Ex10.1, and why

Ex10.1 solved one graphite particle of a fresh cell and, in its section 6,
found the particle's diffusivity from noisy readings of its surface
concentration. A cell in service cannot be opened to read a concentration:
what is measured is its terminal voltage. Two ageing mechanisms of the
graphite electrode act on that voltage:

* **slower diffusion** in the graphite particles (cracking, film growth):
  the negative particle diffusivity multiplied by ``f_D`` < 1;
* **lost cyclable lithium** (lithium bound in the solid-electrolyte
  interphase): the lithium in the graphite at full charge multiplied by
  ``f_L`` < 1.

The cell is a different one from Ex10.1's: PyBaMM's ``Chen2020`` parameter
set of the LG M50 21700 (NMC811/graphite-SiOx, 5 Ah; Chen et al., J.
Electrochem. Soc. 167 (2020) 080534), because it has a complete, published
parameter set for the single particle model with electrolyte (SPMe), which is
the model here. The SPMe is nonlinear (Butler-Volmer kinetics, open-circuit
potentials, a concentration-dependent electrolyte) and has **no exact
solution**. **The strength of the change** is the ageing itself: at
``f_D = f_L = 1`` the model must reproduce the fresh cell
(:func:`check_zero_strength`), and the graphite particle at the mean flux
must follow Ex10.1's exact series with the diffusivity scaled by ``f_D``
(:func:`surface_lag`).

## Three solvers

* the **forward model** (:class:`SPMe`): PyBaMM's SPMe built once with
  ``f_D``, ``f_L`` and the current as inputs; one discharge in a few ms;
* the **reference model** (:func:`least_squares_fit`): scipy's least squares
  over the two factors with the forward model on PyBaMM's default mesh,
  timed per new set of records;
* the **ground truth** (:func:`ground_truth`): the SPMe on a mesh eight times
  finer, tolerances 1e-9, checked on meshes x1, x2, x4, x8.
"""

from __future__ import annotations

import time

import numpy as np

import problem as pb           # Ex10.1's module: the exact series of a particle at constant flux

try:                           # PyBaMM solves the SPMe; the notebook installs it on Colab
    import pybamm
except ImportError:            # pragma: no cover
    pybamm = None

# ------------------------------------------------------------------ the data
CAP_NOM = 5.0                    #: Ah, the M50's nominal capacity in Chen2020; 1C = 5 A
V_CUT = 2.5                      #: V, the lower cut-off (Chen2020)
RATES = (0.5, 1.0, 2.0)          #: the discharge rates a record may hold [1/h]
DQ = 0.05                        #: Ah, one reading every 1 % of the nominal capacity (every 36/C s)
N_READ = 64                      #: the first readings every record holds: up to 3.2 Ah, below every aged capacity
N_LATE = 24                      #: the last readings before the cut-off the network also reads: its last 1.2 Ah
N_MAX = 110                      #: the most readings a record holds (5.5 Ah)
FD_RANGE = (0.3, 1.0)            #: the diffusivity factor's range   [ASSUMED]
FL_RANGE = (0.8, 1.0)            #: the lithium factor's range: up to 20 % of the cyclable lithium lost   [ASSUMED]
NOISE_RANGE = (0.005, 0.010)     #: V, one standard deviation of the voltage noise   [ASSUMED for a cell tester]
CAP_NOISE = 0.005                #: Ah, one standard deviation of the charge counted to the cut-off (0.1 %)   [ASSUMED]
CAP_RATE = 0.5                   #: the capacity asked is the discharge capacity at 0.5C to 2.5 V
#: the agreed accuracy per test case, relative: (f_D, f_L, capacity at 0.5C); None = reported, not required.
#: Set from the Cramer-Rao bound of each case's records (part C of the notebook): a single 2C record (case 4)
#: leaves f_D with a 21 % standard deviation and the capacity with 2.1 %, whatever the method.
TOL = {"1": (0.05, 0.01, 0.01), "2": (0.05, 0.01, 0.01), "3": (0.05, 0.01, 0.01), "4": (None, 0.05, 0.05)}
REFINE_REF = 1                   #: the reference model's mesh: PyBaMM's default
REFINE_TRUTH = 8                 #: the ground truth's mesh
TOL_TRUTH = 1e-9                 #: the ground truth's solver tolerances

KEY_D = "Negative particle diffusivity [m2.s-1]"
KEY_L = "Initial concentration in negative electrode [mol.m-3]"


def parameters(inputs=True):
    """PyBaMM's Chen2020, with the two ageing factors f_D and f_L and the
    current as inputs of the built model (``inputs=False``: the fresh cell,
    nothing an input but the current)."""
    p = pybamm.ParameterValues("Chen2020")
    p["Current function [A]"] = "[input]"
    if inputs:
        d, l = p[KEY_D], p[KEY_L]
        p[KEY_D] = (lambda sto, T, d=d: d(sto, T) * pybamm.InputParameter("f_D")) if callable(d) else d * pybamm.InputParameter("f_D")
        p[KEY_L] = l * pybamm.InputParameter("f_L")
    return p


def constants():
    """The graphite particle and the cell, from the parameter set: a dict."""
    p = pybamm.ParameterValues("Chen2020")
    area = p["Electrode height [m]"] * p["Electrode width [m]"]
    return dict(R_n=p["Negative particle radius [m]"], D_n=p[KEY_D], c_n0=p[KEY_L],
                cmax_n=p["Maximum concentration in negative electrode [mol.m-3]"],
                frac_n=p["Negative electrode active material volume fraction"], L_n=p["Negative electrode thickness [m]"],
                area=area, c_p0=p["Initial concentration in positive electrode [mol.m-3]"],
                cmax_p=p["Maximum concentration in positive electrode [mol.m-3]"])


# ---------------------------------------------------------- the forward model
class SPMe:
    """PyBaMM's SPMe of the LG M50 built once on a mesh ``refine`` times the
    default, with f_D, f_L and the current as inputs."""

    def __init__(self, refine=1, tol=1e-6, inputs=True, model="SPMe"):
        m = {"SPMe": pybamm.lithium_ion.SPMe, "DFN": pybamm.lithium_ion.DFN}[model]()
        pts = {k: v * refine for k, v in m.default_var_pts.items()}
        self.inputs = inputs
        self.sim = pybamm.Simulation(m, parameter_values=parameters(inputs), var_pts=pts,
                                     solver=pybamm.IDAKLUSolver(rtol=tol, atol=tol))
        self.sim.build()

    def solve(self, f_D, f_L, rate, t_interp=None):
        inp = {"Current function [A]": rate * CAP_NOM}
        if self.inputs:
            inp.update(f_D=float(f_D), f_L=float(f_L))
        return self.sim.solve([0.0, 3600.0 / rate * 1.05], inputs=inp, t_interp=t_interp)

    def discharge(self, f_D, f_L, rate, n=400):
        """One discharge at ``rate`` to the cut-off: q [Ah] and V [V] at n even
        instants (the solver's own interpolation between its steps) and the
        cut-off, and the solve time [s]."""
        t0 = time.perf_counter()
        sol = self.solve(f_D, f_L, rate, np.linspace(0.0, 3600.0 / rate * 1.05, n))
        return sol["Discharge capacity [A.h]"].entries, sol["Voltage [V]"].entries, time.perf_counter() - t0

    def record(self, f_D, f_L, rate):
        """What a cell tester records: the voltage [V] every DQ Ah from DQ to
        the cut-off, read at those instants from the solver (not interpolated
        afterwards), and the charge [Ah] discharged at the cut-off."""
        tk = DQ * np.arange(0, 200) * 3600.0 / (rate * CAP_NOM)
        tk = tk[tk < 3600.0 / rate * 1.05]
        sol = self.solve(f_D, f_L, rate, tk)
        t, q, V = sol["Time [s]"].entries, sol["Discharge capacity [A.h]"].entries, sol["Voltage [V]"].entries
        cap = float(q[-1])
        k = t * rate * CAP_NOM / 3600.0 / DQ
        keep = (t > 0) & (np.abs(k - np.round(k)) < 1e-6) & (q < cap - 1e-9)
        return V[keep], cap


def q_readings(n):
    """The discharged charge [Ah] of the first n readings of a record."""
    return DQ * np.arange(1, n + 1)


# ------------------------------------------------------- the reference model
_REF = None


def reference_model():
    """The forward model on PyBaMM's default mesh, built once."""
    global _REF
    if _REF is None:
        _REF = SPMe(REFINE_REF)
    return _REF


def predict(f_D, f_L, rates, n_reads, model=None):
    """The readings the forward model predicts for the given records: for each
    rate, the voltage at the record's readings (``n_reads[i]`` of them; past
    the model's own cut-off the voltage is held at 2.5 V) and the capacity."""
    m = reference_model() if model is None else model
    out = []
    for rate, n in zip(rates, n_reads):
        V, cap = m.record(f_D, f_L, rate)
        V = np.r_[V, np.full(max(0, n - len(V)), V_CUT)][:n]          # past the model's own cut-off: held at 2.5 V
        out.append((V, cap))
    return out


def least_squares_fit(records, noise, starts=((0.6, 0.9),), model=None):
    """The reference model: scipy.optimize.least_squares over (ln f_D, f_L),
    the records predicted by the forward model on the default mesh, each
    voltage residual divided by the noise and each capacity by CAP_NOISE (the
    tester's charge counting, assumed). ``records`` is a list of (rate,
    voltages, capacity). Returns the factors, the capacity at 0.5C, the cost,
    the number of forward solves and the wall-clock time [s]."""
    from scipy.optimize import least_squares
    rates = [r[0] for r in records]
    n_reads = [len(r[1]) for r in records]
    count = [0]
    t0 = time.perf_counter()

    def res(p):
        count[0] += len(rates)
        pred = predict(np.exp(p[0]), p[1], rates, n_reads, model)
        return np.concatenate([np.r_[(Vp - r[1]) / noise, (cp - r[2]) / CAP_NOISE] for (Vp, cp), r in zip(pred, records)])

    lo = np.array([np.log(FD_RANGE[0] * 0.8), FL_RANGE[0] - 0.05])
    hi = np.array([np.log(1.2), 1.05])
    best = None
    for s in starts:
        f = least_squares(res, np.array([np.log(s[0]), s[1]]), bounds=(lo, hi), x_scale=np.array([0.3, 0.05]),
                          diff_step=1e-4, max_nfev=60)
        if best is None or f.cost < best.cost:
            best = f
    f_D, f_L = float(np.exp(best.x[0])), float(best.x[1])
    cap = capacity(f_D, f_L, model)
    return dict(f_D=f_D, f_L=f_L, cap=cap, cost=float(best.cost), solves=count[0] + 1,
                seconds=time.perf_counter() - t0)


def capacity(f_D, f_L, model=None):
    """The discharge capacity [Ah] at 0.5C to 2.5 V of a cell with these factors."""
    m = reference_model() if model is None else model
    return float(m.record(f_D, f_L, CAP_RATE)[1])


def information_bound(f_D, f_L, rates, noise, n_reads=None, model=None):
    """The Cramer-Rao bound at (f_D, f_L): the smallest standard deviations any
    unbiased method can reach from these records, sigma^2 (J^T J)^-1 with J the
    derivatives of the readings (voltages over sigma, capacities over 5 mAh)
    by central differences. Returns the relative standard deviations of f_D and
    f_L, that of the capacity at 0.5C, and the correlation of the two
    estimates."""
    m = reference_model() if model is None else model
    if n_reads is None:
        n_reads = [len(m.record(f_D, f_L, r)[0]) for r in rates]

    def vec(a, b):
        return np.concatenate([np.r_[V / noise, c / CAP_NOISE] for V, c in predict(a, b, rates, n_reads, m)])

    steps = (0.01 * f_D, 0.002)
    J = np.array([(vec(f_D + steps[0], f_L) - vec(f_D - steps[0], f_L)) / (2 * steps[0]),
                  (vec(f_D, f_L + steps[1]) - vec(f_D, f_L - steps[1])) / (2 * steps[1])]).T
    cov = np.linalg.inv(J.T @ J)
    g = np.array([(capacity(f_D + steps[0], f_L, m) - capacity(f_D - steps[0], f_L, m)) / (2 * steps[0]),
                  (capacity(f_D, f_L + steps[1], m) - capacity(f_D, f_L - steps[1], m)) / (2 * steps[1])])
    sd = np.sqrt(np.diag(cov))
    return (float(sd[0] / f_D), float(sd[1] / f_L), float(np.sqrt(g @ cov @ g) / capacity(f_D, f_L, m)),
            float(cov[0, 1] / (sd[0] * sd[1])))


def shape_correlation(rate, f_D=0.5, f_L=0.9, model=None):
    """How alike the two ageing effects look in a discharge: the correlation
    of the voltage change slower diffusion alone makes with the change lost
    lithium alone makes, at the same discharged charge."""
    m = reference_model() if model is None else model
    q0, V0, _ = m.discharge(1.0, 1.0, rate)
    qa, Va, _ = m.discharge(f_D, 1.0, rate)
    qb, Vb, _ = m.discharge(1.0, f_L, rate)
    qq = np.linspace(0, 0.98 * min(qa[-1], qb[-1]), 300)
    a = np.interp(qq, qa, Va) - np.interp(qq, q0, V0)
    b = np.interp(qq, qb, Vb) - np.interp(qq, q0, V0)
    return float(np.corrcoef(a, b)[0, 1]), float(np.abs(a).max()), float(np.abs(b).max())


# ------------------------------------------------- Ex10.1's particle, aged
def surface_lag(f_D, rate=1.0):
    """How far the graphite particle's surface concentration has fallen below
    its mean [mol/m^3] at the end of a discharge at ``rate`` (one hour at 1C), at
    its mean flux: Ex10.1's exact series, jR/(5 D) once the flux has been
    steady, with D = f_D D_n. Slower diffusion makes the surface run further ahead
    of the mean, which the voltage sees."""
    k = constants()
    j = rate * CAP_NOM / (pb.FARADAY * (3 * k["frac_n"] / k["R_n"]) * k["L_n"] * k["area"])
    D = f_D * k["D_n"]
    t = 3600.0 / rate
    c_ref = j * t / k["R_n"]                                          # Ex10.1's concentration scale
    return float(c_ref * (pb.exact(1.0, 1.0, D * t / k["R_n"] ** 2) - 3.0))   # the surface's loss minus the mean's, 3 tau


# -------------------------------------------------------- the ground truth
def convergence_study(f_D, f_L, rate=2.0, refines=(1, 2, 4, REFINE_TRUTH), verbose=True):
    """The capacity and the voltage at 1 and 3 Ah on meshes each twice as
    fine. Returns rows (refine, points per particle, capacity [Ah], V at 1 Ah
    [V], V at 3 Ah [V], seconds), the observed order and the Richardson value
    of the capacity."""
    rows = []
    for r in refines:
        q, V, s = SPMe(r, tol=TOL_TRUTH).discharge(f_D, f_L, rate)
        v1, v3 = np.interp([1.0, 3.0], q, V)
        rows.append((r, 20 * r, float(q[-1]), v1, v3, s))
        if verbose:
            print(f"   mesh x{r}: {20 * r:4d} points in a particle's radius, capacity {q[-1]:.5f} Ah, voltage at 1 Ah "
                  f"{v1:.5f} V and at 3 Ah {v3:.5f} V ({s:.2f} s)")
    caps = np.array([r[2] for r in rows])
    dc = np.diff(caps)
    order = np.log2(np.abs(dc[:-1] / dc[1:]))
    rich = caps[-1] + dc[-1] / (2 ** order[-1] - 1)
    if verbose:
        print("   capacity change between meshes [Ah]:", " ".join(f"{x:+.6f}" for x in dc), "| observed order",
              " ".join(f"{x:.2f}" for x in order), f"| Richardson {rich:.5f} Ah")
    return rows, order, rich


_TRUTH = None


def truth_model():
    global _TRUTH
    if _TRUTH is None:
        _TRUTH = SPMe(REFINE_TRUTH, tol=TOL_TRUTH)
    return _TRUTH


def ground_truth(f_D, f_L, rate):
    """A clean record (voltages every DQ Ah, capacity) on the ground truth's mesh."""
    return truth_model().record(f_D, f_L, rate)


def check_zero_strength(verbose=True):
    """At f_D = f_L = 1 the model with the ageing inputs must be the fresh
    cell's model, built from Chen2020 untouched. Returns the largest voltage
    difference [V] over the three rates."""
    aged, fresh = SPMe(1), SPMe(1, inputs=False)
    worst = 0.0
    for r in RATES:
        qa, Va, _ = aged.discharge(1.0, 1.0, r)
        qf, Vf, _ = fresh.discharge(1.0, 1.0, r)
        qq = np.linspace(0, min(qa[-1], qf[-1]), 500)
        worst = max(worst, float(np.abs(np.interp(qq, qa, Va) - np.interp(qq, qf, Vf)).max()))
    if verbose:
        print(f"   f_D = f_L = 1 against the fresh Chen2020 cell: largest voltage difference {worst:.1e} V")
    return worst


def balances(f_D, f_L, rate=2.0):
    """Faraday's balance on the ground truth: the lithium leaving the graphite
    against the charge passed, relative; and the lithium in the particles, kept."""
    sol = truth_model().sim.solve([0.0, 3600.0 / rate * 1.05], inputs={"Current function [A]": rate * CAP_NOM, "f_D": f_D, "f_L": f_L})
    neg = sol["Total lithium in negative electrode [mol]"].entries
    part = sol["Total lithium in particles [mol]"].entries
    q = sol["Discharge capacity [A.h]"].entries
    passed = q * 3600.0 / pb.FARADAY
    return float(np.abs((neg[0] - neg) - passed)[1:].max() / passed[-1]), float(np.abs(part / part[0] - 1).max())


# ------------------------------------------------------- the training data
def sample_factors(n, seed=101):
    """n cells drawn uniformly over the ranges: (f_D, f_L) per row."""
    rng = np.random.default_rng(seed)
    return np.c_[rng.uniform(*FD_RANGE, n), rng.uniform(*FL_RANGE, n)]


def simulate(factors, verbose=True):
    """Clean records of many cells by the forward model on the default mesh:
    readings (n, 3 rates, N_MAX) [V], NaN past each cut-off, and the charges
    at the cut-offs (n, 3) [Ah]; the first rate's charge is the capacity at
    0.5C. About 3 x 10 ms a cell."""
    m = reference_model()
    R = np.full((len(factors), len(RATES), N_MAX), np.nan); C = np.zeros((len(factors), len(RATES)))
    t0 = time.perf_counter()
    for i, (a, b) in enumerate(factors):
        for k, r in enumerate(RATES):
            V, cap = m.record(a, b, r)
            R[i, k, :len(V)] = V; C[i, k] = cap
        if verbose and (i + 1) % 2000 == 0:
            print(f"   {i + 1} cells, {time.perf_counter() - t0:.0f} s")
    return R, C


def early_late(R):
    """The two windows of records (..., N_MAX) [V] the network reads: the first
    N_READ readings, and the last N_LATE before each cut-off."""
    early = R[..., :N_READ]
    n = np.isfinite(R).sum(-1)                                        # readings per record
    idx = np.clip(n[..., None] - N_LATE + np.arange(N_LATE), 0, R.shape[-1] - 1)
    late = np.take_along_axis(R, idx, -1)
    return early, late


# ----------------------------------------------------------- the test cases
#: rising difficulty: fewer rates, more noise. The factors are hidden in the truth file.
CASES = [
    dict(name="1", f_D=0.55, f_L=0.90, rates=(0.5, 1.0, 2.0), noise=0.005),
    dict(name="2", f_D=0.40, f_L=0.95, rates=(1.0, 2.0), noise=0.005),
    dict(name="3", f_D=0.75, f_L=0.85, rates=(0.5, 1.0, 2.0), noise=0.010),
    dict(name="4", f_D=0.35, f_L=0.83, rates=(2.0,), noise=0.010),
]
TRUTH_FILE = "MP10.1B_truth.npz"


def write_truth(path=TRUTH_FILE):
    """The test cases: noisy records at their rates (seeded), the hidden
    factors, the clean records and the capacity at 0.5C from the ground truth;
    the fresh cell's records; the convergence table, the balances and the
    zero-strength check."""
    t0 = time.perf_counter()
    rows, order, rich = convergence_study(CASES[0]["f_D"], CASES[0]["f_L"])
    out = dict(rates=np.array(RATES), dq=np.array(DQ))
    for c in CASES:
        n = c["name"]
        rng = np.random.default_rng(1010 + int(n))
        V = np.full((len(RATES), N_MAX), np.nan); Vc = np.full((len(RATES), N_MAX), np.nan); caps = np.full(len(RATES), np.nan)
        for k, r in enumerate(RATES):
            clean, cap = ground_truth(c["f_D"], c["f_L"], r)
            Vc[k, :len(clean)] = clean
            if r in c["rates"]:
                V[k, :len(clean)] = clean + rng.normal(0, c["noise"], len(clean))
                caps[k] = cap + rng.normal(0, CAP_NOISE)
        cap05 = float(ground_truth(c["f_D"], c["f_L"], CAP_RATE)[1])
        out.update({f"case{n}_V": V, f"case{n}_cap": caps, f"case{n}_noise": np.array(c["noise"]),
                    f"case{n}_seed": np.array(1010 + int(n)), f"case{n}_mask": np.isin(RATES, c["rates"]).astype(float),
                    f"case{n}_f_D": np.array(c["f_D"]), f"case{n}_f_L": np.array(c["f_L"]),
                    f"case{n}_cap05": np.array(cap05), f"case{n}_clean": Vc, f"case{n}_clean_cap": np.array([ground_truth(c["f_D"], c["f_L"], r)[1] for r in RATES])})
        print(f"   case {n}: f_D {c['f_D']:.2f}, f_L {c['f_L']:.2f}, rates {c['rates']}, noise {c['noise']*1e3:.0f} mV; "
              f"capacity at 0.5C {cap05:.3f} Ah")
    fresh = np.full((len(RATES), N_MAX), np.nan); fcap = np.zeros(len(RATES))
    for k, r in enumerate(RATES):
        v, fcap[k] = ground_truth(1.0, 1.0, r)
        fresh[:, :][k, :len(v)] = v
    out.update(fresh_V=fresh, fresh_cap=fcap, cases=np.array([c["name"] for c in CASES]))
    out["conv_refine"] = np.array([r[0] for r in rows]); out["conv_points"] = np.array([r[1] for r in rows])
    out["conv_cap"] = np.array([r[2] for r in rows]); out["conv_V1"] = np.array([r[3] for r in rows])
    out["conv_V3"] = np.array([r[4] for r in rows]); out["conv_seconds"] = np.array([r[5] for r in rows])
    out["conv_order"] = order; out["conv_richardson_cap"] = np.array(rich)
    out["balance_faraday"], out["balance_particles"] = map(np.array, balances(CASES[0]["f_D"], CASES[0]["f_L"]))
    out["zero_strength_error_V"] = np.array(check_zero_strength(verbose=False))
    out["params"] = np.array([f"{a} = {b}" for a, b in dict(
        CELL="PyBaMM Chen2020: LG M50 21700, NMC811/graphite-SiOx, 5 Ah (Chen et al. 2020)", MODEL="SPMe, isothermal 25 C",
        CUT_OFF="2.5 V", F_D="factor on the negative particle diffusivity, 0.3-1.0 (ASSUMED range)",
        F_L="factor on the initial lithium concentration in the negative electrode, 0.8-1.0 (ASSUMED range)",
        READINGS="voltage every 0.05 Ah discharged (1 % of nominal) to the cut-off, and the charge at the cut-off",
        NOISE="0.005 or 0.010 V, one standard deviation (ASSUMED for a cell tester)",
        TRUTH=f"SPMe on mesh x{REFINE_TRUTH}, tolerances {TOL_TRUTH}", PYBAMM=pybamm.__version__).items()])
    out["README"] = np.array(
        "MP10.1B test cases: aged LG M50 (PyBaMM Chen2020, SPMe). rates: the three rates [1/h], row k of every (3, 110) "
        "array; dq: Ah between readings. OBSERVED: case<n>_V[k, i]: voltage [V] at (i + 1) dq Ah discharged at rate k, "
        "noise of standard deviation case<n>_noise [V] from seed case<n>_seed, NaN past the cut-off or where the rate "
        "was not recorded (case<n>_mask 0); case<n>_cap[k]: charge at the 2.5 V cut-off [Ah] with noise of standard deviation 0.005 Ah, NaN where not recorded. "
        "HIDDEN (for scoring): case<n>_f_D, case<n>_f_L: the factors on the negative particle diffusivity and on the "
        "initial lithium in the negative electrode; case<n>_cap05: capacity at 0.5C to 2.5 V [Ah]; case<n>_clean: the "
        "readings without noise at all three rates, case<n>_clean_cap their charges at the cut-off. fresh_V, fresh_cap: the fresh cell. conv_*: case 1 at 2C on meshes "
        "x1-x8 (refine, points per radius, capacity Ah, V at 1 and 3 Ah, solve s), conv_order, conv_richardson_cap. "
        "balance_*: Faraday and particle-lithium balances (relative). zero_strength_error_V: f_D = f_L = 1 against the "
        "fresh cell.")
    np.savez_compressed(path, **out)
    print(f"written {path} ({time.perf_counter() - t0:.0f} s)")


if __name__ == "__main__":
    import sys
    check_zero_strength()
    convergence_study(CASES[0]["f_D"], CASES[0]["f_L"])
    if "--save" in sys.argv:
        write_truth()
