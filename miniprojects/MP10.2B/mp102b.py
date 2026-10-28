r"""MP10.2B — A day of operation under an hourly price. The model, the
reference model and the ground truth.

*Deep Learning for Engineering* — MSc, Aalborg University.
Remus Teodorescu (ret@et.aau.dk), with support from Research Assistant
Noman Khan (nomank@energy.aau.dk).

This module extends Ex10.2's ``problem.py`` (course policy C10 rule 1: one set
of physics and constants for the exercise and its mini project) and is used by
the notebook of MP10.2B (course policy C14). It takes over the model of
``tools/miniprojects/ex102_truth.py`` (MP10.2B): the notebook's coupled finite
volumes, a hydrogen contract for the day, the 10 K limit.

## The problem

One cell of Ex10.2 (100 cm²) is run for a day. The electricity price changes
every hour; the hydrogen made over the day is sold under a **contract**: a
fixed amount, here what the cell makes in 24 hours at 1.20 V with the gas
entering at 800 degC. For every hour the operator chooses the cell voltage
``V_t`` (1.15 to 1.35 V) and the inlet temperature ``T_t`` (750 to 850 degC):
48 numbers. The cost of the day is

    C = sum_t [ 100 p_t E_t + A_t ]                         [euro cent]

with ``p_t`` the hour's price [euro/kWh], ``E_t = V_t I_t / 1000`` the
electricity of the hour [kWh] and ``A_t`` the share of the stack that ages in
the hour [cent] (Ex10.2's ageing law). The hydrogen is sold at the contract's
price whatever the schedule, so it does not enter the choice. Subject to

    sum_t H_t = H_contract                                  (the contract)
    max_x |T(x) - T_t| <= 10 K   in every hour               (the ceramic layers)

with ``H_t`` the hydrogen of the hour by Faraday's law. Each hour is judged at
its steady state (quasi-steady, as Ex10.2: the thermal transient of moving
between operating points, a few minutes, is not modelled).

## No exact solution

Every hour's electricity, ageing, hydrogen and temperature change come from
the coupled steam, energy and charge balances of Ex10.2, which have no closed
form; the day is a constrained optimisation over 48 numbers of these. The
**structure** is exact: once a Lagrange multiplier ``lambda`` prices the
hydrogen, the hours are independent, each minimising
``100 p_t E_t + A_t - lambda H_t`` under its own temperature limit, and the
contract fixes ``lambda``. The ground truth uses it.

## Two solvers

* the **reference model** (:func:`reference`): SLSQP on the 48 numbers with
  the coupled finite volumes on Ex10.2's engineering grid (50 cells) as the
  model of every hour, the derivatives by forward differences (each hour's
  numbers depend only on its own two, so a gradient costs three solves an
  hour). It is what an engineer would run for a new price curve, and what the
  network is timed against.
* the **ground truth** (:func:`ground_truth`): the coupled finite volumes on
  800 cells tabulated on 81 x 41 operating points (2.5 mV x 2.5 K), smoothed by
  bicubic splines, each hour minimised on them for a given ``lambda``, the
  ``lambda`` that makes the contract by bisection, and the schedule found
  checked again hour by hour with the finite volumes on 800 cells.

Every price curve is **invented for the project**; every kinetic, transport
and ageing value is Ex10.2's estimate.
"""

from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import problem as pb

F = pb.FARADAY
HOURS = 24
LIMIT = pb.DT_MAX                   #: K, the largest temperature change along the cell, 10 K   [estimate, Ex10.2]
LIMIT_TOL = 0.2                     #: K, the agreed tolerance on the limit, checked by finite volumes on 800 cells
TOL_COST = 0.005                    #: the agreed accuracy of the day's cost, relative, 0.5 %
TOL_CONTRACT = 0.005                #: the agreed accuracy of the contract, relative, 0.5 %
REF_CELLS = 50                      #: Ex10.2's engineering grid: 0.01 in steam fraction, 0.5 K
TRUTH_CELLS = 800                   #: the ground truth's grid
TRUTH_CELL_GRIDS = (100, 200, 400, 800)
VB = np.linspace(pb.V_LOW, pb.V_HIGH, 81)        #: V, the operating grid of the table, 2.5 mV
TB = np.linspace(pb.T_LOW, pb.T_HIGH, 41)        #: K, 2.5 K

#: Hourly electricity prices, euro/kWh, hour 0 = 00:00 to 01:00. All four are invented for the project, with the
#: shapes of a day-ahead market: none of them is used to build the network, and each is a new case for every method.
PRICES = {
    "autumn weekday": np.array([0.060, 0.055, 0.050, 0.050, 0.055, 0.065, 0.085, 0.100, 0.095, 0.070, 0.045, 0.030,
                                0.020, 0.020, 0.025, 0.040, 0.065, 0.100, 0.140, 0.130, 0.110, 0.090, 0.075, 0.065]),
    "sunny spring day": np.array([0.080, 0.075, 0.070, 0.070, 0.075, 0.090, 0.120, 0.110, 0.060, 0.025, 0.008, 0.002,
                                  0.000, 0.000, 0.004, 0.020, 0.060, 0.130, 0.180, 0.170, 0.140, 0.110, 0.095, 0.085]),
    "windy night": np.array([0.015, 0.010, 0.008, 0.008, 0.010, 0.020, 0.060, 0.100, 0.110, 0.105, 0.095, 0.090,
                             0.085, 0.085, 0.090, 0.100, 0.115, 0.140, 0.150, 0.130, 0.090, 0.060, 0.035, 0.020]),
    "winter cold snap": np.array([0.110, 0.100, 0.095, 0.095, 0.100, 0.130, 0.200, 0.300, 0.280, 0.220, 0.180, 0.160,
                                  0.150, 0.150, 0.160, 0.190, 0.250, 0.320, 0.330, 0.290, 0.230, 0.180, 0.150, 0.130]),
}
PRICE_NAMES = tuple(PRICES)
V_CONTRACT, T_CONTRACT = 1.20, 1073.15   #: the contract is what 24 hours at 1.20 V and 800 degC make


# ------------------------------------------------------------ the model of an hour
def fvm(V, T_in, n=REF_CELLS, tol=1e-10):
    """Ex10.2's coupled finite volumes (its section 4): the steam fraction and
    the scaled temperature in n cells, upwind convection, central diffusion,
    the current from the charge transfer in every cell, Newton–Raphson on all
    2n unknowns. Returns the cell centres, y, T [K] and i [A/cm^2]."""
    h = 1.0 / n
    dy, dt = 1 / (pb.PE * h * h), 1 / (pb.PE_T * h * h)
    a_in = np.ones(n); a_in[0] = 2.0; a_out = np.ones(n); a_out[-1] = 0.0
    q = pb.S_Q * (pb.V_TN - V)
    y, th, e = np.full(n, pb.Y_IN), np.zeros(n), 1e-7
    for _ in range(30):
        T = T_in + pb.DT_SCALE * th; i = pb.current(V, y, T)
        dV = (pb.cell_voltage(i + e, y, T) - pb.cell_voltage(i - e, y, T)) / (2 * e)
        i_y = -(pb.cell_voltage(i, y + e, T) - pb.cell_voltage(i, y - e, T)) / (2 * e) / dV
        i_t = -(pb.cell_voltage(i, y, T + 1e-4) - pb.cell_voltage(i, y, T - 1e-4)) / 2e-4 / dV * pb.DT_SCALE
        yu, yd = np.r_[pb.Y_IN, y[:-1]], np.r_[y[1:], y[-1]]; tu, td = np.r_[0.0, th[:-1]], np.r_[th[1:], th[-1]]
        r_y = (y - yu) / h - dy * (a_out * (yd - y) - a_in * (y - yu)) + pb.K_I * i
        r_t = (th - tu) / h - dt * (a_out * (td - th) - a_in * (th - tu)) + q * i
        Jyy = sp.diags([-(1 / h + dy * a_in[1:]), 1 / h + dy * (a_out + a_in) + pb.K_I * i_y, -dy * a_out[:-1]], [-1, 0, 1])
        Jtt = sp.diags([-(1 / h + dt * a_in[1:]), 1 / h + dt * (a_out + a_in) + q * i_t, -dt * a_out[:-1]], [-1, 0, 1])
        step = spla.spsolve(sp.bmat([[Jyy, sp.diags(pb.K_I * i_t)], [sp.diags(q * i_y), Jtt]], format="csc"),
                            -np.r_[r_y, r_t])
        y, th = y + step[:n], th + step[n:]
        if np.abs(step).max() < tol:
            break
    T = T_in + pb.DT_SCALE * th
    return (np.arange(n) + 0.5) * h, y, T, pb.current(V, y, T)


def hour_from_fields(V, i, T, T_in):
    """The numbers of one hour from the fields on equal cells (midpoint rule):
    the current [A], the electricity [kWh], the hydrogen [kg], the ageing
    [cent] and the largest temperature change [K]."""
    current = float(np.mean(i) * pb.A_CELL)
    ageing = float(np.mean(pb.K_DEG * np.exp(-pb.E_DEG / (pb.R_GAS * T)) * np.clip(i, 0, None) ** pb.N_DEG))
    return dict(current=current, electricity=V * current / 1000, hydrogen=current / (2 * F) * pb.M_H2 * 3600,
                ageing=100 * pb.STACK_COST * pb.P_RATED * ageing / pb.DASR_EOL, dT=float(np.abs(T - T_in).max()))


def hour(V, T_in, n=TRUTH_CELLS):
    """One hour at cell voltage V [V] and inlet temperature T_in [K], by the
    coupled finite volumes on n cells."""
    _, _, T, i = fvm(V, T_in, n)
    return hour_from_fields(V, i, T, T_in)


def day(Vs, Ts, price, n=TRUTH_CELLS):
    """A schedule (24 voltages [V], 24 inlet temperatures [K]) judged by the
    finite volumes on n cells: the day's cost [cent], hydrogen [kg], each
    hour's largest temperature change [K] and the hours' numbers."""
    hrs = [hour(V, T, n) for V, T in zip(Vs, Ts)]
    cost = sum(100 * p * h["electricity"] + h["ageing"] for p, h in zip(price, hrs))
    return dict(cost=float(cost), hydrogen=float(sum(h["hydrogen"] for h in hrs)),
                dT=np.array([h["dT"] for h in hrs]), hours=hrs)


def contract(n=TRUTH_CELLS):
    """The hydrogen contract [kg/day]: what 24 hours at 1.20 V and 800 degC make."""
    return 24 * hour(V_CONTRACT, T_CONTRACT, n)["hydrogen"]


# ------------------------------------------------------- the reference model
def reference(price, n=REF_CELLS, demand=None, x0=None, verbose=False):
    """The reference model for one price curve: SLSQP on the 48 numbers
    (scaled to [0, 1]), the finite volumes on n cells as the model of every
    hour, forward differences for the derivatives (three solves an hour, as
    each hour depends only on its own two numbers), the contract as an
    equality and the 24 temperature limits as inequalities. Started from 1.25 V
    and 800 degC in every hour. Returns the schedule, its model values and
    the wall-clock time [s]."""
    from scipy.optimize import minimize
    t0 = time.perf_counter()
    demand = contract(n) if demand is None else demand
    eps = 1e-6
    cache = {}

    def unscale(z):
        return pb.V_LOW + (pb.V_HIGH - pb.V_LOW) * z[:HOURS], pb.T_LOW + (pb.T_HIGH - pb.T_LOW) * z[HOURS:]

    def evaluate(z):
        key = z.tobytes()
        if key in cache:
            return cache[key]
        Vs, Ts = unscale(z)
        base = np.array([[h["electricity"], h["ageing"], h["hydrogen"], h["dT"]]
                         for h in (hour(V, T, n) for V, T in zip(Vs, Ts))])
        dV = np.array([[h["electricity"], h["ageing"], h["hydrogen"], h["dT"]]
                       for h in (hour(V + eps * (pb.V_HIGH - pb.V_LOW), T, n) for V, T in zip(Vs, Ts))])
        dTn = np.array([[h["electricity"], h["ageing"], h["hydrogen"], h["dT"]]
                        for h in (hour(V, T + eps * (pb.T_HIGH - pb.T_LOW), n) for V, T in zip(Vs, Ts))])
        cache.clear(); cache[key] = (base, (dV - base) / eps, (dTn - base) / eps)
        return cache[key]

    def cost(z):
        b, _, _ = evaluate(z); return float(np.sum(100 * price * b[:, 0] + b[:, 1]))

    def cost_grad(z):
        _, gv, gt = evaluate(z); return np.r_[100 * price * gv[:, 0] + gv[:, 1], 100 * price * gt[:, 0] + gt[:, 1]]

    def h2(z):
        b, _, _ = evaluate(z); return np.array([1e3 * (np.sum(b[:, 2]) - demand)])     # in grams

    def h2_grad(z):
        _, gv, gt = evaluate(z); return 1e3 * np.r_[gv[:, 2], gt[:, 2]][None, :]

    def lim(z):
        b, _, _ = evaluate(z); return LIMIT - b[:, 3]

    def lim_grad(z):
        _, gv, gt = evaluate(z); return -np.c_[np.diag(gv[:, 3]), np.diag(gt[:, 3])]

    z0 = np.r_[np.full(HOURS, 0.5), np.full(HOURS, 0.5)] if x0 is None else x0
    res = minimize(cost, z0, jac=cost_grad, method="SLSQP", bounds=[(0, 1)] * (2 * HOURS),
                   constraints=[dict(type="eq", fun=h2, jac=h2_grad), dict(type="ineq", fun=lim, jac=lim_grad)],
                   options=dict(ftol=1e-10, maxiter=200, disp=verbose))
    Vs, Ts = unscale(res.x)
    sec = time.perf_counter() - t0
    return dict(V=Vs, T_in=Ts, cost_model=float(res.fun), iterations=int(res.nit), success=bool(res.success),
                seconds=sec, demand=demand)


# -------------------------------------------------------- the ground truth
def table(n=TRUTH_CELLS, VB=VB, TB=TB, verbose=False):
    """Every hour's numbers on the operating grid, by the finite volumes on n
    cells: electricity [kWh], ageing [cent], hydrogen [kg] and the largest
    temperature change [K], each an array [V, T_in]."""
    out = {k: np.zeros((len(VB), len(TB))) for k in ("electricity", "ageing", "hydrogen", "dT")}
    t0 = time.perf_counter()
    for a, V in enumerate(VB):
        for b, T in enumerate(TB):
            h = hour(V, T, n)
            for k in out:
                out[k][a, b] = h[k]
        if verbose and a % 20 == 0:
            print(f"   table, {n} cells: {a + 1} of {len(VB)} voltages, {time.perf_counter() - t0:.0f} s", flush=True)
    return out


class Smooth:
    """The table smoothed by bicubic splines in (V, T_in), with derivatives."""

    def __init__(self, tab, VB=VB, TB=TB):
        from scipy.interpolate import RectBivariateSpline
        self.f = {k: RectBivariateSpline(VB, TB, v, kx=3, ky=3) for k, v in tab.items()}

    def __call__(self, k, V, T, dx=0, dy=0):
        return self.f[k](V, T, dx=dx, dy=dy, grid=False)


def best_hour(S, p, lam, limit=LIMIT, starts=None):
    """The operating point of one hour at price p [euro/kWh] that minimises
    100 p E + A - lam H [cent] under the temperature limit, on the splines:
    SLSQP from the best node of a coarse search."""
    from scipy.optimize import minimize
    sv, st = pb.V_HIGH - pb.V_LOW, pb.T_HIGH - pb.T_LOW
    J = lambda V, T: 100 * p * S("electricity", V, T) + S("ageing", V, T) - lam * S("hydrogen", V, T)
    if starts is None:
        Vg, Tg = np.meshgrid(np.linspace(pb.V_LOW, pb.V_HIGH, 41), np.linspace(pb.T_LOW, pb.T_HIGH, 21), indexing="ij")
        ok = S("dT", Vg, Tg) <= limit
        j = np.where(ok, J(Vg, Tg), np.inf); a = np.unravel_index(np.argmin(j), j.shape)
        starts = [(Vg[a], Tg[a])]
    best = None
    for V0, T0 in starts:
        f = lambda z: float(J(pb.V_LOW + sv * z[0], pb.T_LOW + st * z[1]))
        g = lambda z: np.array([
            float(sv * (100 * p * S("electricity", pb.V_LOW + sv * z[0], pb.T_LOW + st * z[1], dx=1)
                        + S("ageing", pb.V_LOW + sv * z[0], pb.T_LOW + st * z[1], dx=1)
                        - lam * S("hydrogen", pb.V_LOW + sv * z[0], pb.T_LOW + st * z[1], dx=1))),
            float(st * (100 * p * S("electricity", pb.V_LOW + sv * z[0], pb.T_LOW + st * z[1], dy=1)
                        + S("ageing", pb.V_LOW + sv * z[0], pb.T_LOW + st * z[1], dy=1)
                        - lam * S("hydrogen", pb.V_LOW + sv * z[0], pb.T_LOW + st * z[1], dy=1)))])
        c = lambda z: np.array([limit - float(S("dT", pb.V_LOW + sv * z[0], pb.T_LOW + st * z[1]))])
        r = minimize(f, [(V0 - pb.V_LOW) / sv, (T0 - pb.T_LOW) / st], jac=g, method="SLSQP", bounds=[(0, 1), (0, 1)],
                     constraints=[dict(type="ineq", fun=c)], options=dict(ftol=1e-14, maxiter=200))
        if best is None or r.fun < best.fun:
            best = r
    return pb.V_LOW + sv * best.x[0], pb.T_LOW + st * best.x[1]


def schedule(S, price, lam, limit=LIMIT):
    pts = [best_hour(S, p, lam, limit) for p in price]
    Vs, Ts = np.array([q[0] for q in pts]), np.array([q[1] for q in pts])
    return Vs, Ts, float(np.sum(S("hydrogen", Vs, Ts)))


def optimum(S, price, demand, limit=LIMIT, lo=0.0, hi=3000.0, iters=50):
    """The multiplier lam [cent/kg] that makes the contract, by bisection on the
    hydrogen of the day (it rises with lam), and its schedule."""
    for _ in range(iters):
        lam = 0.5 * (lo + hi)
        _, _, made = schedule(S, price, lam, limit)
        lo, hi = (lam, hi) if made < demand else (lo, lam)
    lam = 0.5 * (lo + hi)
    Vs, Ts, made = schedule(S, price, lam, limit)
    return dict(lam=lam, V=Vs, T_in=Ts, hydrogen_spline=made)


def best_constant(S, price, demand, limit=LIMIT, nv=401, nt=201):
    """The best operating point held all day that makes the contract under the
    limit, on the splines (a fine search): what a schedule must beat."""
    Vg, Tg = np.meshgrid(np.linspace(pb.V_LOW, pb.V_HIGH, nv), np.linspace(pb.T_LOW, pb.T_HIGH, nt), indexing="ij")
    ok = (S("dT", Vg, Tg) <= limit) & (HOURS * S("hydrogen", Vg, Tg) >= demand)
    c = np.where(ok, 100 * np.sum(price) * S("electricity", Vg, Tg) + HOURS * S("ageing", Vg, Tg), np.inf)
    a = np.unravel_index(np.argmin(c), c.shape)
    return float(Vg[a]), float(Tg[a]), float(c[a])


def check_balances(V=1.32, T_in=1023.15, n=TRUTH_CELLS):
    """The finite volumes are conservative: the steam that leaves the channel
    (convected out, less what diffuses back in at the inlet) equals the steam
    the current splits, and the heat carried out equals the heat the current
    absorbs or releases, both in scaled units. Returns the two mismatches."""
    x, y, T, i = fvm(V, T_in, n)
    h = 1.0 / n; th = (T - T_in) / pb.DT_SCALE
    steam_in = 2 * (pb.Y_IN - y[0]) / (pb.PE * h)              # diffusion through the inlet face (half a cell)
    steam = (pb.Y_IN - y[-1]) + steam_in - pb.K_I * np.mean(i)
    heat_in = 2 * (0.0 - th[0]) / (pb.PE_T * h)
    heat = (th[-1] - 0.0) - heat_in - pb.S_Q * (V - pb.V_TN) * np.mean(i)
    return float(steam), float(heat)


TRUTH_FILE = "MP10.2B_truth.npz"
TRUTH_GRID_STEPS = (4, 2, 1)        #: the operating grids of the convergence study: every 4th, 2nd and every node


def write_truth(path=TRUTH_FILE, verbose=True):
    """The ground truth: the contract, the table on 800 cells and its
    convergence (cells 100 to 800; operating grids 10 mV x 10 K to 2.5 mV x
    2.5 K), the optimal day for every price curve checked hour by hour by the
    finite volumes on 800 cells, the best constant point, and the checks."""
    t0 = time.perf_counter()
    demand = contract()
    out = dict(contract_kg=np.array(demand), VB=VB, TB=TB, price_names=np.array(PRICE_NAMES),
               limit_K=np.array(LIMIT), limit_tol_K=np.array(LIMIT_TOL))
    tabs = {}
    for n in TRUTH_CELL_GRIDS:
        tabs[n] = table(n, verbose=verbose)
        if verbose:
            print(f"   table on {n} cells done, {time.perf_counter() - t0:.0f} s", flush=True)
    for k, v in tabs[TRUTH_CELLS].items():
        out[f"table_{k}"] = v
    # convergence of the day's cost (autumn weekday) in the cells and in the operating grid
    price = PRICES[PRICE_NAMES[0]]
    conv_cells, conv_grid = [], []
    for n in TRUTH_CELL_GRIDS:
        o = optimum(Smooth(tabs[n]), price, demand)
        S = Smooth(tabs[n]); c = float(np.sum(100 * price * S("electricity", o["V"], o["T_in"]) + S("ageing", o["V"], o["T_in"])))
        conv_cells.append(c)
        if verbose:
            print(f"   {n:4d} cells: day's cost {c:.4f} cent, multiplier {o['lam']:.2f} cent/kg", flush=True)
    for st in TRUTH_GRID_STEPS:
        sub = {k: v[::st, ::st] for k, v in tabs[TRUTH_CELLS].items()}
        S = Smooth(sub, VB[::st], TB[::st]); o = optimum(S, price, demand)
        c = float(np.sum(100 * price * S("electricity", o["V"], o["T_in"]) + S("ageing", o["V"], o["T_in"])))
        conv_grid.append(c)
        if verbose:
            print(f"   operating grid {len(VB[::st])} x {len(TB[::st])}: day's cost {c:.4f} cent", flush=True)
    out["conv_cells"] = np.array(TRUTH_CELL_GRIDS); out["conv_cells_cost"] = np.array(conv_cells)
    out["conv_grid_steps"] = np.array([2.5e-3 * s for s in TRUTH_GRID_STEPS]); out["conv_grid_cost"] = np.array(conv_grid)
    S = Smooth(tabs[TRUTH_CELLS])
    for name in PRICE_NAMES:
        price = PRICES[name]; key = name.replace(" ", "_")
        o = optimum(S, price, demand)
        chk = day(o["V"], o["T_in"], price)
        Vc, Tc, cc = best_constant(S, price, demand)
        cchk = day(np.full(HOURS, Vc), np.full(HOURS, Tc), price)
        out.update({f"{key}_price": price, f"{key}_V": o["V"], f"{key}_T_in": o["T_in"], f"{key}_lam": np.array(o["lam"]),
                    f"{key}_cost": np.array(chk["cost"]), f"{key}_hydrogen": np.array(chk["hydrogen"]),
                    f"{key}_dT": chk["dT"], f"{key}_constant_V": np.array(Vc), f"{key}_constant_T_in": np.array(Tc),
                    f"{key}_constant_cost": np.array(cchk["cost"]),
                    f"{key}_hydrogen_hour": np.array([h["hydrogen"] for h in chk["hours"]])})
        if verbose:
            print(f"   {name}: day's cost {chk['cost']:.2f} cent (best constant point {cchk['cost']:.2f}, "
                  f"{100 * (1 - chk['cost'] / cchk['cost']):.2f} % more), hydrogen {1e3 * chk['hydrogen']:.2f} g "
                  f"(contract {1e3 * demand:.2f}), largest change {chk['dT'].max():.2f} K, "
                  f"multiplier {o['lam'] / 100:.2f} euro/kg   ({time.perf_counter() - t0:.0f} s)", flush=True)
    flat = optimum(S, np.full(HOURS, 0.07), demand)
    out["flat_spread_V"] = np.array(np.ptp(flat["V"])); out["flat_spread_T"] = np.array(np.ptp(flat["T_in"]))
    out["balance_steam"], out["balance_heat"] = (np.array(v) for v in check_balances())
    out["params"] = np.array([f"{k} = {v}" for k, v in dict(
        cell="Ex10.2's cell, 100 cm2, every kinetic, transport and ageing value Ex10.2's estimate",
        V="1.15-1.35 V", T_in="750-850 degC", contract="24 h at 1.20 V and 800 degC, by finite volumes on 800 cells",
        limit="10 K, Ex10.2's estimate", prices="invented for the project, euro/kWh, hour 0 = 00:00-01:00",
        truth="finite volumes on 800 cells, table 81 x 41, bicubic splines, Lagrange multiplier by bisection").items()])
    out["README"] = np.array(
        "MP10.2B ground truth. contract_kg: hydrogen of the day [kg]; VB [V], TB [K]: the operating grid; "
        "table_<electricity [kWh], ageing [cent], hydrogen [kg], dT [K]>: one hour at each (V, T_in) by finite volumes "
        "on 800 cells; for each price curve <name> (spaces as _): _price [euro/kWh], _V [V] and _T_in [K] the optimal "
        "schedule, _lam [cent/kg] its multiplier, _cost [cent], _hydrogen [kg], _dT [K] (per hour) checked by finite "
        "volumes on 800 cells, _hydrogen_hour [kg], _constant_V/_T_in/_cost the best constant point; conv_cells(_cost): "
        "the autumn weekday's cost [cent] by tables on 100-800 cells; conv_grid_steps [V] (_cost): by operating grids "
        "of 10, 5 and 2.5 mV; flat_spread_V/T: the spread of the optimal schedule under a flat price (zero: one point); "
        "balance_steam/heat: the finite volumes' conservation, scaled; params: the data.")
    np.savez_compressed(path, **out)
    print("written", path, f"({time.perf_counter() - t0:.0f} s)")
    return out


if __name__ == "__main__":
    import sys
    print("balances (steam, heat):", check_balances())
    if "--save" in sys.argv:
        write_truth()
