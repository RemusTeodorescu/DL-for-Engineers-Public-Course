r"""MP8.2A — Thermal cycling of a power module on a heat sink, with natural
convection added to the fan's air. The physics, the reference model and the
ground truth.

*Deep Learning for Engineering* — MSc, Aalborg University.
Remus Teodorescu (ret@et.aau.dk), with support from Research Assistant
Noman Khan (nomank@energy.aau.dk).

This module extends Ex08's ``heatsink.py`` (course policy C10 rule 1: one set
of physics and constants for the exercise and its mini project) and is used by
the five notebooks of MP8.2A (course policy C14).

## What changes against Ex08, and why

Ex08 cools the fins with one constant coefficient, h = 40 W/(m²K). That makes
the problem linear, and its solution is an exact series. A real heat sink in
an enclosure is also cooled by the buoyancy of the air it warms: the hotter
the fins, the stronger the natural convection. The coefficient becomes

    h(ΔT) = H_AIR + S_NAT · C_NAT · (ΔT / L_FIN)^(1/4)

the forced part of Ex08 plus the laminar natural-convection part of a
vertical plate in air (Incropera & DeWitt, ch. 9, the simplified air
correlation; from memory, ASSUMED until checked). ΔT is the fin-root rise, the
fin side of the base plate. The fins' efficiency depends on h, so the
effective coefficient on the base, h_eff(ΔT), follows from Ex08's
``fin_efficiency``. The wall condition becomes nonlinear in the temperature,
and **no exact solution exists**. ``S_NAT`` is the strength of the change: at
``S_NAT = 0`` the problem is Ex08's and the solver must reproduce Ex08's exact
series (the check of :func:`check_zero_strength`).

## The problem, in kelvin above the air

    ρc_p ∂θ/∂t = k (∂²θ/∂x² + ∂²θ/∂z²)        in the base plate, 0 < x < W, 0 < z < t_b
    k ∂θ/∂z = (P / P0) q''(x) s(t)              at z = t_b: the module on its footprint
    k ∂θ/∂z = h_eff(θ) θ                        at z = 0: the fins
    ∂θ/∂x = 0                                   at x = 0 and x = W: insulated ends
    θ = 0                                       at t = 0

s(t) is the train: 1 for T_ON every PERIOD, CYCLES times. P is the module's
power, an input between P_MIN and P_MAX; q'' is Ex08's flux at P0 = 150 W.

## Two solvers

* the **reference model** (:func:`reference`): Crank-Nicolson finite
  differences on an engineering grid, the nonlinear wall by three chord
  iterations per step. It is what the network is timed against.
* the **ground truth** (:func:`ground_truth`): the same scheme on grids each
  twice as fine, converged and extrapolated (:func:`convergence_study`).
"""

from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import heatsink as hs

# ------------------------------------------------------------------ the data
C_NAT = 1.42            #: W/(m^(7/4) K^(5/4)): h = C_NAT (ΔT/L)^(1/4), laminar vertical plate in air   [ASSUMED]
S_NAT = 1.0             #: strength of the change: 0 is Ex08's linear problem
P0 = hs.POWER           #: W, the power Ex08's flux is set for (150 W)
P_MIN, P_MAX = 100.0, 200.0   #: W, the range of the module's power, an input of the network
P_TEST = (115.0, 165.0, 195.0)   #: W, the powers the answers are scored at (not trained on)
T_ON = 20.0             #: s, the module is on for 20 s ...
PERIOD = 60.0           #: s, ... every minute
CYCLES = 10             #: ten cycles
T_END = PERIOD * CYCLES   #: s, the window, 600 s
CENTRE = (hs.W_BASE / 2, hs.T_BASE)   #: m, the module's centre on the top face: the hottest point


def h_air(dT, s_nat=S_NAT):
    """The coefficient on the fins at a fin-root rise dT (K): forced plus natural."""
    dT = np.maximum(np.asarray(dT, dtype=float), 0.0)
    return hs.H_AIR + s_nat * C_NAT * (dT / hs.L_FIN) ** 0.25


def h_eff(dT, s_nat=S_NAT):
    """The effective coefficient the finned face puts on the base plate at a
    fin-root rise dT: Ex08's fin calculation with h(dT)."""
    return hs.fin_efficiency(h_air(dT, s_nat))[2]


def train_on(t):
    """s(t): 1 while the module is on, 0 while it is off."""
    t = np.asarray(t, dtype=float)
    return ((t % PERIOD) < T_ON) & (t < T_END)


# ---------------------------------------------------------- the time march
class March:
    """Crank-Nicolson on an nx by nz grid of the base plate, in kelvin, with
    the nonlinear fin face. Built once per grid and step; run for any power."""

    def __init__(self, nx, nz, dt, s_nat=S_NAT):
        self.nx, self.nz, self.dt, self.s_nat = nx, nz, dt, s_nat
        self.x = np.linspace(0, hs.W_BASE, nx)
        self.z = np.linspace(0, hs.T_BASE, nz)
        hx, hz = self.x[1], self.z[1]
        a = hs.ALPHA
        Dx = sp.diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(nx, nx)).tolil()
        Dx[0, 1] = 2.0; Dx[-1, -2] = 2.0                     # mirrored nodes: insulated ends
        Dz = sp.diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(nz, nz)).tolil()
        Dz[0, 1] = 2.0; Dz[-1, -2] = 2.0                     # ghost nodes on both faces; the fluxes go to vectors
        self.A = (a / hx ** 2) * sp.kron(sp.identity(nz), Dx.tocsr()) + (a / hz ** 2) * sp.kron(Dz.tocsr(), sp.identity(nx))
        self.A = self.A.tocsc()
        # the module's flux into the top row, through its ghost node
        flux = np.where((self.x >= hs.MODULE[0]) & (self.x <= hs.MODULE[1]), hs.FLUX, 0.0)
        edge = np.isclose(self.x, hs.MODULE[0]) | np.isclose(self.x, hs.MODULE[1])
        flux = np.where(edge, 0.5 * hs.FLUX, flux)            # a node on the footprint's edge carries half
        self.b = np.zeros(nx * nz)
        self.b[-nx:] = 2.0 * flux / (hs.RHO_CP * hz)
        self.cw = 2.0 / (hs.RHO_CP * hz)                       # the fin face: -cw h_eff(θ) θ on the bottom row
        # the chord matrix: the fin face linearised once, at a typical rise of 20 K
        d0 = self.cw * (h_eff(20.0, s_nat) * 1.25)
        D = sp.diags(np.r_[np.full(nx, d0), np.zeros(nx * (nz - 1))])
        I = sp.identity(nx * nz, format="csc")
        self.lu = spla.splu((I - 0.5 * dt * (self.A - D)).tocsc())
        self.D, self.R = D, (I + 0.5 * dt * self.A).tocsc()

    def wall(self, th):
        """The fin face's term on the bottom row: -cw h_eff(θ) θ."""
        out = np.zeros_like(th)
        t0 = th[: self.nx]
        out[: self.nx] = -self.cw * h_eff(t0, self.s_nat) * t0
        return out

    def run(self, power=P0, record=None, iters=3):
        """March the window at the module power ``power`` (W). Returns the
        times (s), the centre's rise (K) at every step, and the fields
        (len(record), nz, nx) at the instants of ``record`` (s, multiples of dt)."""
        nt = int(round(T_END / self.dt))
        scale = power / P0
        th = np.zeros(self.nx * self.nz)
        ic = (self.nz - 1) * self.nx + self.nx // 2           # the centre of the top face
        centre = np.zeros(nt + 1)
        rec = {} if record is None else {int(round(r / self.dt)): r for r in record}
        fields = []
        if 0 in rec:
            fields.append(th.reshape(self.nz, self.nx).copy())
        s_old = -1.0
        h2 = 0.5 * self.dt
        wx = np.full(self.nx, self.x[1]); wx[[0, -1]] *= 0.5     # trapezoid weights across the base
        wz = np.full(self.nz, self.z[1]); wz[[0, -1]] *= 0.5     # and through its thickness
        fin = lambda v: hs.DEPTH * np.sum(wx * h_eff(v[: self.nx], self.s_nat) * v[: self.nx])   # W through the fins
        q_in = q_out = 0.0
        for n in range(1, nt + 1):
            t_mid = (n - 0.5) * self.dt
            s = float(train_on(t_mid))                          # the train switches on whole steps
            if s != s_old:
                # Rannacher's start: after a switch the flux jumps, and Crank-Nicolson
                # would carry the jump as an undamped oscillation, costing an order.
                # Two backward-Euler half steps damp it; their matrix is CN's own.
                new = th
                for _ in range(2):
                    rhs = new + h2 * scale * s * self.b
                    nxt = new.copy()
                    for _ in range(iters):
                        nxt = self.lu.solve(rhs + h2 * (self.wall(nxt) + self.D @ nxt))
                    new = nxt
            else:
                rhs = self.R @ th + 0.5 * self.dt * self.wall(th) + self.dt * scale * s * self.b
                new = th.copy()
                for _ in range(iters):                          # chord iterations on the nonlinear face
                    new = self.lu.solve(rhs + 0.5 * self.dt * (self.wall(new) + self.D @ new))
            s_old = s
            q_in += self.dt * s * power                          # J, the module's heat
            q_out += 0.5 * self.dt * (fin(th) + fin(new))        # J, through the fins
            th = new
            centre[n] = th[ic]
            if n in rec:
                fields.append(th.reshape(self.nz, self.nx).copy())
        stored = hs.RHO_CP * hs.DEPTH * np.sum(np.outer(wz, wx) * th.reshape(self.nz, self.nx))
        #: the energy balance over the window: heat in, heat out through the fins, heat stored (J)
        self.balance = dict(heat_in=q_in, heat_out=q_out, stored=stored, residual=q_in - q_out - stored)
        return np.arange(nt + 1) * self.dt, centre, np.array(fields)


def cycle_extremes(t, centre):
    """The peak of the centre's rise in each cycle, and its trough: the rise at
    the end of the cycle, just before the module switches on again (K)."""
    peaks, troughs = [], []
    for c in range(CYCLES):
        m = (t > c * PERIOD) & (t <= (c + 1) * PERIOD)
        peaks.append(centre[m].max()); troughs.append(np.interp((c + 1) * PERIOD, t, centre))
    return np.array(peaks), np.array(troughs)


# ------------------------------------------------------- the reference model
REF_GRID = (11, 3, 4.0)    #: nx, nz, dt (s): the engineering grid, chosen in the convergence study


def reference(power=P0, record=(T_END,)):
    """The reference model: Crank-Nicolson on the engineering grid. Returns a
    dict with the times, the centre's rise, the cycle peaks and troughs, the
    fields at ``record``, and the wall-clock time of the run (s)."""
    m = March(*REF_GRID)
    t0 = time.perf_counter()
    t, c, f = m.run(power, record)
    wall = time.perf_counter() - t0
    p, q = cycle_extremes(t, c)
    return dict(t=t, centre=c, peaks=p, troughs=q, fields=f, x=m.x, z=m.z, seconds=wall)


# -------------------------------------------------------- the ground truth
TRUTH_GRIDS = ((41, 9, 0.5), (81, 17, 0.25), (161, 33, 0.125), (321, 65, 0.0625))


def convergence_study(power=P0, grids=TRUTH_GRIDS, verbose=True):
    """The cycle peaks and troughs on grids each twice as fine in x, z and t.
    Crank-Nicolson and central differences are second order, so the change
    between grids should fall by about four each time; the last two are
    extrapolated by Richardson. Returns the list of (grid, peaks, troughs,
    seconds) and the extrapolated peaks and troughs."""
    rows = []
    for g in grids:
        t0 = time.perf_counter()
        t, c, _ = March(*g).run(power)
        p, q = cycle_extremes(t, c)
        rows.append((g, p, q, time.perf_counter() - t0))
        if verbose:
            print(f"   grid {g[0]:4d} x {g[1]:3d}, dt {g[2]:.4f} s: peak of cycle 10 {p[-1]:.4f} K, "
                  f"trough {q[-1]:.4f} K   ({rows[-1][3]:.1f} s)")
    (_, p1, q1, _), (_, p2, q2, _) = rows[-2], rows[-1]
    pe, qe = p2 + (p2 - p1) / 3, q2 + (q2 - q1) / 3
    if verbose:
        for i in range(1, len(rows)):
            d = np.abs(rows[i][1] - rows[i - 1][1]).max()
            print(f"   change of the peaks, grid {i} to {i + 1}: {d:.4f} K")
        print(f"   extrapolated: peak of cycle 10 {pe[-1]:.4f} K, trough {qe[-1]:.4f} K")
    return rows, pe, qe


def ground_truth(power, grid=TRUTH_GRIDS[-1], record=(T_END,)):
    """The centre's rise and the final field at a power, on the finest grid."""
    m = March(*grid)
    t, c, f = m.run(power, record)
    p, q = cycle_extremes(t, c)
    return dict(t=t, centre=c, peaks=p, troughs=q, fields=f, x=m.x, z=m.z)


def check_zero_strength(grid=(161, 33, 0.125), verbose=True):
    """At S_NAT = 0 the problem is Ex08's, linear, and its train has an exact
    solution by superposition of Ex08's steps. The march must reproduce it."""
    t, c, _ = March(*grid, s_nat=0.0).run(P0)
    mu = hs.roots(120)
    x, z = np.array([CENTRE[0]]), np.array([CENTRE[1]])
    ts = np.arange(5.0, T_END + 1e-9, 5.0)
    ex = []
    for tt in ts:
        th = 0.0
        for k in range(CYCLES):
            t0 = k * PERIOD
            if tt > t0:
                th += hs._step(x, z, tt - t0, 300, 120, mu)[0]
            if tt > t0 + T_ON:
                th -= hs._step(x, z, tt - t0 - T_ON, 300, 120, mu)[0]
        ex.append(th)
    ex = np.array(ex)
    num = np.interp(ts, t, c)
    err = np.abs(num - ex).max()
    if verbose:
        print(f"   S_NAT = 0: largest difference from Ex08's exact series {err:.4f} K over the window")
    return err


# ------------------------------------------------------- the training points
SWITCHES = np.array(sorted({k * PERIOD for k in range(CYCLES)} | {k * PERIOD + T_ON for k in range(CYCLES)}))   #: s


def sample_times(n, rng, near=0.5, spread=4.0):
    """n instants in the window: a share ``near`` of them just after a switch
    (on or off), exponentially distributed with mean ``spread`` seconds,
    where the field changes fastest; the rest uniform."""
    k = int(n * near)
    after = SWITCHES[rng.integers(0, SWITCHES.size, k)] + rng.exponential(spread, k)
    return np.clip(np.r_[after, rng.uniform(0, T_END, n - k)], 0.0, T_END)


def sample_interior(n, seed=88, near=0.25):
    """n points (X, Z, t, P) inside the base plate, in the window and the power
    range: X = x/W and Z = z/t_b in (0, 1), t in s, P in W. ``near``: the share
    of instants just after a switch (see sample_times)."""
    rng = np.random.default_rng(seed)
    return rng.random(n), rng.random(n), sample_times(n, rng, near), rng.uniform(P_MIN, P_MAX, n)


def sample_top(n, seed=89, edges=0.1, near=0.25):
    """n points (X, t, P) on the module face; a share ``edges`` of them within
    2.5 mm of the footprint's edges, where the flux steps from zero to
    50 kW/m². A smooth network cannot follow that step exactly, so points
    there keep a residual no training removes; many of them can dominate the
    loss (set ``edges=0`` for a uniform face)."""
    rng = np.random.default_rng(seed)
    lo, hi = hs.MODULE[0] / hs.W_BASE, hs.MODULE[1] / hs.W_BASE
    k = int(n * edges)
    X = np.r_[rng.random(n - k), np.where(rng.random(k) < 0.5, lo, hi) + 0.05 * (rng.random(k) - 0.5)]
    return np.clip(X, 0, 1), sample_times(n, rng, near), rng.uniform(P_MIN, P_MAX, n)


def sample_bottom(n, seed=90, near=0.25):
    """n points (X, t, P) on the fin side."""
    rng = np.random.default_rng(seed)
    return rng.random(n), sample_times(n, rng, near), rng.uniform(P_MIN, P_MAX, n)


def thermocouple(power=P0, seed=91):
    """Measured data, if a design wants some: the thermocouple of Ex08 beside
    the module (x = 80 mm on the module face), every 2 s with 0.05 K of noise,
    from the ground truth at the power ``power``. Returns t (s) and the rise (K)."""
    m = March(*TRUTH_GRIDS[2])
    tt, _, f = m.run(power, record=np.arange(2.0, T_END + 1e-9, 2.0))
    ix = int(round(hs.SENSOR / hs.W_BASE * (m.nx - 1)))
    rng = np.random.default_rng(seed)
    return np.arange(2.0, T_END + 1e-9, 2.0), f[:, -1, ix] + hs.NOISE_K * rng.standard_normal(f.shape[0])


# ----------------------------------------------------------- the truth file
TRUTH_FILE = "MP8.2A_truth.npz"


def write_truth(path=TRUTH_FILE, grid=TRUTH_GRIDS[2]):
    """The ground truth at the three test powers: the centre's rise every
    0.125 s, the cycle peaks and troughs, and the field at the end, on the
    161 x 33 grid (its peaks within 0.001 K of the extrapolated values), with
    the convergence table and every parameter with its unit."""
    rows, pe, qe = convergence_study(P0, TRUTH_GRIDS, verbose=True)
    out = dict(powers=np.array(P_TEST), grid=np.array(grid, dtype=float))
    for P in P_TEST:
        g = ground_truth(P, grid)
        key = f"{int(P)}W"
        out[f"t"] = g["t"]; out[f"centre_{key}"] = g["centre"]
        out[f"peaks_{key}"] = g["peaks"]; out[f"troughs_{key}"] = g["troughs"]
        out[f"field_{key}"] = g["fields"][-1]
        out["x"], out["z"] = g["x"], g["z"]
    out["conv_grids"] = np.array([r[0] for r in rows], dtype=float)
    out["conv_peak10"] = np.array([r[1][-1] for r in rows]); out["conv_trough10"] = np.array([r[2][-1] for r in rows])
    out["conv_peak10_extrapolated"] = pe[-1]; out["zero_strength_error_K"] = check_zero_strength(verbose=False)
    out["params"] = np.array([f"{k} = {v}" for k, v in dict(
        C_NAT="1.42 W/(m^1.75 K^1.25), ASSUMED", S_NAT=S_NAT, T_ON="20 s", PERIOD="60 s", CYCLES=CYCLES,
        P_RANGE="100-200 W", H_AIR=f"{hs.H_AIR} W/(m2K)", K=f"{hs.K_AL} W/(m K)", RHO_CP=f"{hs.RHO_CP} J/(m3K)").items()])
    out["README"] = np.array(
        "MP8.2A ground truth. t: s; centre_<P>W: rise of the module centre on the top face, K above the air; "
        "peaks_/troughs_<P>W: per cycle, K (trough = rise at the end of the cycle); field_<P>W: (nz, nx) rise at 600 s, K, "
        "on x, z (m); conv_*: the 150 W convergence study, grids (nx, nz, dt s); params: the data.")
    np.savez_compressed(path, **out)
    print("written", path)


if __name__ == "__main__":
    print("zero-strength check")
    check_zero_strength()
    print("convergence at 150 W")
    convergence_study()
    import sys
    if "--save" in sys.argv:
        write_truth()
