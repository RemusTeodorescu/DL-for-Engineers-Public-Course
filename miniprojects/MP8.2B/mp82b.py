r"""MP8.2B — Heat capacity and fan coefficient together, from one
thermocouple record: an inverse problem, with natural convection on the fins.
The physics, the reference model, the least-squares reference and the
ground truth.

*Deep Learning for Engineering* — MSc, Aalborg University.
Remus Teodorescu (ret@et.aau.dk), with support from Research Assistant
Noman Khan (nomank@energy.aau.dk).

This module extends Ex08's ``heatsink.py`` (course policy C10 rule 1: one set
of physics and constants for the exercise and its mini project) and is used by
the notebook of MP8.2B (course policy C14).

## What changes against Ex08, and why

Ex08 Part 2 switches the 150 W module on for 20 s and records the
thermocouple glued beside it (x = 80 mm on the module face) every 2 s for
150 s, with 0.05 K of noise; its inverse problem finds h_eff with ρc_p known.
Here **two** numbers are hidden: the base's volumetric heat capacity ρc_p (the
alloy is not known exactly) and the fan's coefficient h_fan (the fan's speed
is not known). And the fins are cooled as in MP8.2A, by the fan and by natural
convection,

    h(ΔT) = h_fan + S_NAT · C_NAT · (ΔT / L_fin)^(1/4)

so the fin-side condition is nonlinear in the temperature and **no exact
solution exists**: the records are made by the converged finite-difference
march. ``S_NAT`` is the strength of the change; at ``S_NAT = 0`` with Ex08's
values the march reproduces Ex08's exact pulse (:func:`check_zero_strength`).

## The problem, in kelvin above the air

    ρc_p ∂θ/∂t = k (∂²θ/∂x² + ∂²θ/∂z²)         in the base plate
    k ∂θ/∂z = q''(x) s(t)                         at z = t_b: the module, on for 20 s
    k ∂θ/∂z = h_eff(θ; h_fan) θ                   at z = 0: the fins
    ∂θ/∂x = 0 at x = 0, W;      θ = 0 at t = 0

Hidden: ρc_p in 1.95 - 2.92 MJ/(m³K) (±20 % about Ex08's 2.43) and h_fan in
25 - 55 W/(m²K) (Ex08's 40 in the middle). Observed: 75 readings of the
thermocouple, at 2, 4, ..., 150 s, with 0.05 K of noise.

## Two solvers and the least-squares reference

* the **reference model** (:class:`March` on :data:`REF_GRID`): Crank-
  Nicolson on an engineering grid; it is the forward model of the
  **least-squares reference** (:func:`lsq_fit`), the conventional way to find
  two parameters from a record, timed per new record.
* the **ground truth**: the same march on grids each twice as fine
  (:func:`convergence_study`); the test records are made on its fine grid.
"""

from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import least_squares

import heatsink as hs

# ------------------------------------------------------------------ the data
C_NAT = 1.42              #: W/(m^(7/4) K^(5/4)), MP8.2A's natural convection   [ASSUMED]
S_NAT = 1.0               #: strength of the change: 0 is Ex08's linear problem
RHO_CP_RANGE = (1.95e6, 2.92e6)   #: J/(m³K), ±20 % about Ex08's 2.43e6
H_FAN_RANGE = (25.0, 55.0)        #: W/(m²K), about Ex08's 40
T_READ = np.arange(hs.DT_SENSOR, hs.T_END + 1e-9, hs.DT_SENSOR)   #: s, the 75 instants of the record
IX_SENSOR = hs.SENSOR     #: m, x = 80 mm on the module face
NOISE_K = hs.NOISE_K      #: K
#: the test records: hidden (ρc_p J/(m³K), h_fan W/(m²K)) and noise seed - not trained on
TESTS = ((2.43e6, 40.0, 101), (2.10e6, 30.0, 102), (2.75e6, 50.0, 103), (2.20e6, 52.0, 104), (2.80e6, 28.0, 105))


def h_air(dT, h_fan, s_nat=S_NAT):
    """The coefficient on the fins at a fin-root rise dT (K): fan plus natural convection."""
    dT = np.maximum(np.asarray(dT, dtype=float), 0.0)
    return h_fan + s_nat * C_NAT * (dT / hs.L_FIN) ** 0.25


def h_eff(dT, h_fan, s_nat=S_NAT):
    """The effective coefficient of the finned face on the base at a fin-root rise dT."""
    return hs.fin_efficiency(h_air(dT, h_fan, s_nat))[2]


# ---------------------------------------------------------- the time march
class March:
    """Crank-Nicolson on an nx by nz grid of the base plate, in kelvin, with the
    nonlinear fin face, for given ρc_p and h_fan. The matrices depend on both,
    so a new pair builds and factorises them again (part of the cost of every
    least-squares iteration)."""

    def __init__(self, nx, nz, dt, rho_cp=hs.RHO_CP, h_fan=hs.H_AIR, s_nat=S_NAT):
        self.nx, self.nz, self.dt = nx, nz, dt
        self.rho_cp, self.h_fan, self.s_nat = rho_cp, h_fan, s_nat
        self.x = np.linspace(0, hs.W_BASE, nx)
        self.z = np.linspace(0, hs.T_BASE, nz)
        hx, hz = self.x[1], self.z[1]
        a = hs.K_AL / rho_cp
        Dx = sp.diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(nx, nx)).tolil()
        Dx[0, 1] = 2.0; Dx[-1, -2] = 2.0                     # mirrored nodes: insulated ends
        Dz = sp.diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(nz, nz)).tolil()
        Dz[0, 1] = 2.0; Dz[-1, -2] = 2.0                     # ghost nodes on both faces; the fluxes go to vectors
        self.A = ((a / hx ** 2) * sp.kron(sp.identity(nz), Dx.tocsr())
                  + (a / hz ** 2) * sp.kron(Dz.tocsr(), sp.identity(nx))).tocsc()
        flux = np.where((self.x >= hs.MODULE[0]) & (self.x <= hs.MODULE[1]), hs.FLUX, 0.0)
        edge = np.isclose(self.x, hs.MODULE[0]) | np.isclose(self.x, hs.MODULE[1])
        flux = np.where(edge, 0.5 * hs.FLUX, flux)            # a node on the footprint's edge carries half
        self.b = np.zeros(nx * nz); self.b[-nx:] = 2.0 * flux / (rho_cp * hz)
        self.cw = 2.0 / (rho_cp * hz)                         # the fin face: -cw h_eff(θ) θ on the bottom row
        d0 = self.cw * h_eff(5.0, h_fan, s_nat) * 1.1         # chord: the fin face linearised once
        self.D = sp.diags(np.r_[np.full(nx, d0), np.zeros(nx * (nz - 1))])
        I = sp.identity(nx * nz, format="csc")
        self.lu = spla.splu((I - 0.5 * dt * (self.A - self.D)).tocsc())
        self.R = (I + 0.5 * dt * self.A).tocsc()
        self.isensor = (nz - 1) * nx + int(round(IX_SENSOR / hs.W_BASE * (nx - 1)))
        self.icentre = (nz - 1) * nx + nx // 2

    def wall(self, th):
        out = np.zeros_like(th)
        t0 = th[: self.nx]
        out[: self.nx] = -self.cw * h_eff(t0, self.h_fan, self.s_nat) * t0
        return out

    def run(self, iters=3, fields_at=()):
        """March the 150 s window. Returns the times (s), the sensor's rise and
        the module centre's rise at every step (K), and the fields at the
        instants ``fields_at`` (s)."""
        nt = int(round(hs.T_END / self.dt))
        th = np.zeros(self.nx * self.nz)
        sens = np.zeros(nt + 1); cent = np.zeros(nt + 1)
        rec = {int(round(r / self.dt)) for r in fields_at}; fields = []
        h2 = 0.5 * self.dt; s_old = -1.0
        for n in range(1, nt + 1):
            s = float((n - 0.5) * self.dt < hs.T_ON)
            if s != s_old:                                    # Rannacher's start after a switch
                new = th
                for _ in range(2):
                    rhs = new + h2 * s * self.b
                    nxt = new.copy()
                    for _ in range(iters):
                        nxt = self.lu.solve(rhs + h2 * (self.wall(nxt) + self.D @ nxt))
                    new = nxt
            else:
                rhs = self.R @ th + 0.5 * self.dt * self.wall(th) + self.dt * s * self.b
                new = th.copy()
                for _ in range(iters):
                    new = self.lu.solve(rhs + 0.5 * self.dt * (self.wall(new) + self.D @ new))
            s_old = s; th = new
            sens[n] = th[self.isensor]; cent[n] = th[self.icentre]
            if n in rec:
                fields.append(th.reshape(self.nz, self.nx).copy())
        return np.arange(nt + 1) * self.dt, sens, cent, np.array(fields)


def record(rho_cp, h_fan, grid, s_nat=S_NAT):
    """The thermocouple's rise (K) at the 75 instants of the record, noise-free."""
    t, s, _, _ = March(*grid, rho_cp=rho_cp, h_fan=h_fan, s_nat=s_nat).run()
    return np.interp(T_READ, t, s)


# ------------------------------------------------------- the reference model
REF_GRID = (21, 3, 1.0)    #: nx, nz, dt (s): the engineering grid, chosen in the convergence study


def reference(rho_cp, h_fan):
    """The reference model's record (K) for a pair, and its time (s)."""
    t0 = time.perf_counter()
    r = record(rho_cp, h_fan, REF_GRID)
    return r, time.perf_counter() - t0


def lsq_fit(readings, grid=REF_GRID, start=(hs.RHO_CP, hs.H_AIR), s_nat=S_NAT):
    """The least-squares reference: ρc_p and h_fan that make the reference
    model's record closest to the readings, by scipy's trust-region
    least_squares in the logarithms of both (scaled, positive), the Jacobian
    by finite differences. Starts from Ex08's values. Returns the pair, the
    number of forward runs and the wall-clock time (s)."""
    calls = [0]
    def res(p):
        calls[0] += 1
        return record(np.exp(p[0]), np.exp(p[1]), grid, s_nat) - readings
    t0 = time.perf_counter()
    lo = np.log([RHO_CP_RANGE[0] * 0.8, H_FAN_RANGE[0] * 0.8]); hi = np.log([RHO_CP_RANGE[1] * 1.2, H_FAN_RANGE[1] * 1.2])
    sol = least_squares(res, np.log(start), bounds=(lo, hi), x_scale=0.1, diff_step=1e-3, xtol=1e-6)
    return dict(rho_cp=float(np.exp(sol.x[0])), h_fan=float(np.exp(sol.x[1])), runs=calls[0],
                seconds=time.perf_counter() - t0, rms=float(np.sqrt(np.mean(sol.fun ** 2))))


# -------------------------------------------------------- the ground truth
TRUTH_GRIDS = ((21, 3, 1.0), (41, 5, 0.5), (81, 9, 0.25), (161, 17, 0.125))
TRUTH_GRID = TRUTH_GRIDS[-1]


def convergence_study(rho_cp=hs.RHO_CP, h_fan=hs.H_AIR, grids=TRUTH_GRIDS, verbose=True):
    """The sensor's record on grids each twice as fine in x, z and t; second
    order expected; the last two extrapolated by Richardson. Returns the rows
    (grid, record, seconds), the extrapolated record, and the energy balance
    on the finest grid."""
    rows = []
    for g in grids:
        t0 = time.perf_counter()
        r = record(rho_cp, h_fan, g)
        rows.append((g, r, time.perf_counter() - t0))
    ext = rows[-1][1] + (rows[-1][1] - rows[-2][1]) / 3
    if verbose:
        for i, (g, r, s) in enumerate(rows):
            msg = f"   grid {g[0]:4d} x {g[1]:3d}, dt {g[2]:.4f} s: sensor peak {r.max():.4f} K, at 150 s {r[-1]:.4f} K ({s:.1f} s)"
            if i:
                msg += f";  largest change from the grid before {np.abs(r - rows[i-1][1]).max():.5f} K"
            print(msg)
        print(f"   extrapolated: peak {ext.max():.4f} K; finest grid within {np.abs(rows[-1][1] - ext).max():.5f} K")
    return rows, ext


def check_zero_strength(grid=(161, 17, 0.125), verbose=True):
    """At S_NAT = 0 with Ex08's ρc_p and h the problem is Ex08's pulse, whose
    exact series (hs.exact_transient) the march must reproduce at the sensor."""
    r = record(hs.RHO_CP, hs.H_AIR, grid, s_nat=0.0)
    ex = np.array([hs.exact_transient(np.array([hs.SENSOR]), np.array([hs.T_BASE]), t)[0] for t in T_READ])
    err = float(np.abs(r - ex).max())
    if verbose:
        print(f"   S_NAT = 0: largest difference from Ex08's exact pulse at the sensor {err:.2e} K")
    return err


def noisy(clean, seed):
    """A record with the thermocouple's noise, 0.05 K, seeded."""
    return clean + NOISE_K * np.random.default_rng(seed).standard_normal(clean.shape)


# ------------------------------------------------------- training records
def sample_pairs(n, seed=82):
    """n hidden pairs (ρc_p, h_fan), uniform over the ranges."""
    rng = np.random.default_rng(seed)
    return rng.uniform(*RHO_CP_RANGE, n), rng.uniform(*H_FAN_RANGE, n)


def training_records(n, grid=TRUTH_GRIDS[1], seed=82, verbose=True):
    """Noise-free records of n random pairs, by the march on ``grid`` (41 x 5,
    0.5 s: within 0.01 K of the ground truth). One run each, a few tens of
    milliseconds; a network that learns from records is trained on these."""
    rc, hf = sample_pairs(n, seed)
    t0 = time.perf_counter()
    R = np.array([record(a, b, grid) for a, b in zip(rc, hf)])
    if verbose:
        print(f"   {n} records on {grid[0]} x {grid[1]}, dt {grid[2]} s, in {time.perf_counter() - t0:.0f} s")
    return rc, hf, R


# ----------------------------------------------------------- the truth file
TRUTH_FILE = "MP8.2B_truth.npz"


def write_truth(path=TRUTH_FILE):
    """The five test records (observed: readings with noise; hidden: the true
    pairs and the clean records), the convergence table, the zero-strength
    check, every parameter with its unit."""
    rows, ext = convergence_study(verbose=True)
    out = dict(t=T_READ, sensor_x=np.array(hs.SENSOR), noise_K=np.array(NOISE_K))
    hid = []
    for i, (rc, hf, seed) in enumerate(TESTS):
        clean = record(rc, hf, TRUTH_GRID)
        out[f"readings_{i}"] = noisy(clean, seed)
        out[f"clean_{i}"] = clean
        hid.append((rc, hf, seed))
    out["hidden_rho_cp"] = np.array([h[0] for h in hid]); out["hidden_h_fan"] = np.array([h[1] for h in hid])
    out["noise_seeds"] = np.array([h[2] for h in hid])
    out["conv_grids"] = np.array([r[0] for r in rows], dtype=float)
    out["conv_records"] = np.array([r[1] for r in rows]); out["conv_extrapolated"] = ext
    out["zero_strength_error_K"] = np.array(check_zero_strength(verbose=False))
    out["params"] = np.array([f"{k} = {v}" for k, v in dict(
        C_NAT="1.42 W/(m^1.75 K^1.25), ASSUMED", S_NAT=S_NAT, POWER=f"{hs.POWER} W for {hs.T_ON} s",
        WINDOW=f"{hs.T_END} s", SENSOR="x = 80 mm on the module face, every 2 s, 0.05 K noise",
        RHO_CP_RANGE="1.95e6-2.92e6 J/(m3K)", H_FAN_RANGE="25-55 W/(m2K)", K=f"{hs.K_AL} W/(m K)").items()])
    out["README"] = np.array(
        "MP8.2B ground truth. t: s, the 75 instants; readings_<i>: OBSERVED thermocouple rise above the air with noise, K, "
        "test record i = 0..4; clean_<i>: the same without noise (HIDDEN); hidden_rho_cp (J/(m3K)), hidden_h_fan (W/(m2K)): "
        "the HIDDEN pairs; noise_seeds; conv_*: the sensor record at Ex08's values on four grids (nx, nz, dt s) and "
        "extrapolated; zero_strength_error_K: S_NAT = 0 against Ex08's exact pulse; params: the data.")
    np.savez_compressed(path, **out)
    print("written", path)


if __name__ == "__main__":
    print("zero-strength check")
    check_zero_strength()
    print("convergence at Ex08's values")
    convergence_study()
    import sys
    if "--save" in sys.argv:
        write_truth()
