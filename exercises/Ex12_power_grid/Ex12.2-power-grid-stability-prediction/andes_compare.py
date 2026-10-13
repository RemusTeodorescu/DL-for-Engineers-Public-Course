"""
andes_compare.py — ANDES implementation of the Ex12.1 / Ex12.2 six-bus case.

Drop-in alongside problem.py. Does not modify or replace anything.
Uses the same BUSES, BRANCHES, and machine parameters so every difference
in the results is attributable to the simulation engine, not the data.

Two models:
    build_classical()  — GENCLS, no exciter, no governor (fair comparison)
    build_detailed()   — GENROU + ESST1A + TGOV1 (what a real study would use)

Usage:
    import andes_compare as ac

    # One CCT, classical model
    cct = ac.compute_cct(fault_bus=0, trip_line=0, model="classical")

    # Compare with problem.py
    import problem as pb
    op = np.stack(pb.injections())
    cct_pb = pb.label_case(op, outage=0, fault_bus=0)
    print(f"problem.py: {cct_pb*1e3:.1f} ms   ANDES classical: {cct*1e3:.1f} ms")

    # Full dataset comparison
    results = ac.compare_all_contingencies(model="classical")

Requires: pip install andes
"""

import numpy as np

try:
    import andes
    andes.config_logger(stream_level=40)  # errors only
    HAS_ANDES = True
except ImportError:
    HAS_ANDES = False
    print("  [andes_compare] andes not installed — pip install andes")


# ═══════════════════════════════════════════════════════════════════════
# Network data — copied from problem.py so this file is self-contained
# ═══════════════════════════════════════════════════════════════════════

BASE_MVA = 100.0

BUSES = [
    ("Slack / SE link", "slack",  0.00,  0.00),
    ("Gen east",        "gen",    1.15,  0.05),
    ("Load north",      "load",  -0.58, -0.20),
    ("Load city",       "load",  -1.05, -0.36),
    ("Load south",      "load",  -0.46, -0.16),
    ("HVDC / DE link",  "load",   0.36,  0.06),
]

BRANCHES = [
    (0, 1, 0.0035, 0.0300, 0.030),
    (1, 2, 0.0063, 0.0385, 0.022),
    (2, 3, 0.0049, 0.0333, 0.026),
    (0, 4, 0.0077, 0.0455, 0.018),
    (4, 3, 0.0056, 0.0350, 0.024),
    (3, 5, 0.0070, 0.0420, 0.020),
]

N_BUS = len(BUSES)

# Machine parameters — same as problem.py Machines class
MACHINE_BUSES = [0, 1]
H_VALUES = [150.0, 4.0]
D_VALUES = [2.0, 1.2]
X_D_PRIME = 0.30         # same for both machines in problem.py

# Timing — same as problem.py Ex12.2
T_FAULT = 0.5
T_END = 5.0
CCT_MAX = 0.5
CCT_TOL = 0.002
PROTECTION_TIME = 0.14


# ═══════════════════════════════════════════════════════════════════════
# System builders
# ═══════════════════════════════════════════════════════════════════════

def _add_network(ss, P_override=None, Q_override=None):
    """Add buses, lines, loads, and static generators common to both models."""

    P_bus = [b[2] for b in BUSES] if P_override is None else list(P_override)
    Q_bus = [b[3] for b in BUSES] if Q_override is None else list(Q_override)

    # Buses (problem.py 0..5 → ANDES 1..6)
    for idx, (name, *_) in enumerate(BUSES, 1):
        ss.add("Bus", {"idx": idx, "name": name, "Vn": 230, "v0": 1.0})

    # Lines
    for i, (f, t, r, x, b) in enumerate(BRANCHES, 1):
        ss.add("Line", {"idx": i, "bus1": f + 1, "bus2": t + 1,
                         "r": r, "x": x, "b": b,
                         "Vn1": 230, "Vn2": 230, "rate_a": 999})

    # Static generators (interface for dynamic models)
    ss.add("Slack", {"idx": 1, "bus": 1, "Vn": 230,
                      "v0": 1.0, "p0": abs(P_bus[0]) if P_bus[0] != 0 else 0.58})
    ss.add("PV", {"idx": 2, "bus": 2, "Vn": 230,
                   "v0": 1.0, "p0": P_bus[1], "q0": Q_bus[1]})

    # Loads — positive p0 = consumption in ANDES
    load_buses = [i for i in range(N_BUS) if BUSES[i][1] == "load"]
    for k, bus in enumerate(load_buses, 1):
        p = -P_bus[bus] if P_bus[bus] < 0 else 0.0
        q = -Q_bus[bus] if Q_bus[bus] < 0 else 0.0
        if P_bus[bus] > 0:    # injection (HVDC)
            p = -P_bus[bus]   # negative consumption = generation
            q = -Q_bus[bus]
        ss.add("PQ", {"idx": k, "bus": bus + 1, "Vn": 230,
                       "p0": -P_bus[bus] if BUSES[bus][1] == "load" else 0,
                       "q0": -Q_bus[bus] if BUSES[bus][1] == "load" else 0})


def build_classical(P=None, Q=None):
    """GENCLS — classical model, constant E behind xd'.
    No exciter, no governor. Fair comparison with problem.py."""
    ss = andes.System(default_config=True)
    _add_network(ss, P, Q)

    for idx, (bus, H, D) in enumerate(
            zip(MACHINE_BUSES, H_VALUES, D_VALUES), 1):
        ss.add("GENCLS", {
            "idx": idx, "bus": bus + 1, "gen": idx,
            "Vn": 230, "Sn": BASE_MVA, "fn": 50,
            "M": 2 * H, "D": D,
            "xd1": X_D_PRIME,
            "ra": 0.003,
        })
    return ss


def build_detailed(P=None, Q=None):
    """GENROU + ESST1A + TGOV1 — what a real planning study would use."""
    ss = andes.System(default_config=True)
    _add_network(ss, P, Q)

    for idx, (bus, H, D) in enumerate(
            zip(MACHINE_BUSES, H_VALUES, D_VALUES), 1):
        ss.add("GENROU", {
            "idx": idx, "bus": bus + 1, "gen": idx,
            "Vn": 230, "Sn": BASE_MVA, "fn": 50,
            "M": 2 * H, "D": D,
            "xd": 1.8, "xq": 1.7,
            "xd1": 0.30, "xq1": 0.55,
            "xd2": 0.25, "xq2": 0.25,
            "Td10": 8.0, "Tq10": 0.4,
            "Td20": 0.03, "Tq20": 0.05,
            "ra": 0.003,
        })
        # Exciter
        ss.add("ESST1A", {
            "idx": idx, "syn": idx,
            "KA": 200, "TA": 0.01,
            "VRMAX": 7.0, "VRMIN": -6.0,
            "KC": 0.04,
        })
        # Governor
        ss.add("TGOV1", {
            "idx": idx, "syn": idx,
            "R": 0.05, "T1": 0.5,
            "VMAX": 1.2, "VMIN": 0.0,
            "Dt": 0.0, "T2": 2.5, "T3": 7.5,
        })
    return ss


# ═══════════════════════════════════════════════════════════════════════
# Simulation
# ═══════════════════════════════════════════════════════════════════════

def run_case(fault_bus, trip_line=None, t_clear=None,
             model="classical", P=None, Q=None, t_end=T_END):
    """Run one contingency. Returns dict with t, delta, omega, or None."""

    if not HAS_ANDES:
        raise ImportError("andes not installed")

    builder = build_classical if model == "classical" else build_detailed
    ss = builder(P, Q)

    t_clear = T_FAULT + 0.1 if t_clear is None else t_clear

    # Fault at the specified bus (problem.py index → ANDES index +1)
    ss.add("Fault", {"idx": 1, "bus": fault_bus + 1,
                      "tf": T_FAULT, "tc": t_clear,
                      "xf": 0.0001, "rf": 0.0})

    # Trip line at clearing time (if specified)
    if trip_line is not None:
        ss.add("Toggler", {"idx": 2, "model": "Line",
                            "dev": trip_line + 1, "t": t_clear})

    ss.setup()
    ss.PFlow.run()
    if not ss.PFlow.converged:
        return None

    ss.TDS.config.tf = t_end
    ss.TDS.config.tstep = 0.005
    ss.TDS.init()
    ss.TDS.run()

    # Get machine model
    if model == "classical":
        gen = ss.GENCLS
    else:
        gen = ss.GENROU

    t = np.array(ss.dae.ts.t)
    delta = np.array(ss.dae.ts.x[:, gen.delta.a])
    omega = np.array(ss.dae.ts.x[:, gen.omega.a])

    return {"t": t, "delta": delta, "omega": omega,
            "converged": ss.TDS.converged,
            "V": ss.Bus.v.v.copy(),
            "ang": np.degrees(ss.Bus.a.v.copy())}


def is_stable(result, limit=np.pi):
    """Check if the simulation stayed synchronized."""
    if result is None or not result["converged"]:
        return False
    sep = np.abs(result["delta"][:, 0] - result["delta"][:, 1])
    return float(np.max(sep)) < limit


def compute_cct(fault_bus, trip_line=None, model="classical",
                P=None, Q=None,
                lo=0.005, hi=CCT_MAX, tol=CCT_TOL, t_end=T_END):
    """Bisect on clearing time — same algorithm as problem.py."""

    def stable(tc):
        r = run_case(fault_bus, trip_line,
                     t_clear=T_FAULT + tc,
                     model=model, P=P, Q=Q, t_end=t_end)
        return is_stable(r)

    if not stable(lo):
        return 0.0
    if stable(hi):
        return hi
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if stable(mid):
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ═══════════════════════════════════════════════════════════════════════
# Contingency list — mirrors problem.py contingencies()
# ═══════════════════════════════════════════════════════════════════════

def contingencies():
    """Same contingency list as problem.py."""
    # Islanding check: line 5 (bus 3–5) islands bus 5
    islanding = {5}
    out = [{"index": 0, "outage": None, "fault_bus": 1,
            "label": "base — fault at Gen east, no line lost"}]
    for k, (f, t, *_) in enumerate(BRANCHES):
        if k in islanding:
            continue
        out.append({"index": len(out), "outage": k, "fault_bus": f,
                    "label": f"trip line {k}  ({BUSES[f][0]} – {BUSES[t][0]})"})
    return out


# ═══════════════════════════════════════════════════════════════════════
# Comparison tools
# ═══════════════════════════════════════════════════════════════════════

def compare_all_contingencies(model="classical", verbose=True):
    """Run all contingencies at base dispatch, return CCTs."""
    cases = contingencies()
    results = []

    for c in cases:
        cct = compute_cct(fault_bus=c["fault_bus"],
                          trip_line=c["outage"],
                          model=model)
        results.append({"case": c, "cct": cct,
                        "secure": cct > PROTECTION_TIME})
        if verbose:
            flag = "secure" if cct > PROTECTION_TIME else "INSECURE"
            ceil = "  (censored)" if cct >= CCT_MAX else ""
            print(f"  {c['label']:<46s} {cct*1e3:6.1f} ms   {flag}{ceil}")

    return results


def compare_with_problem(verbose=True):
    """Side-by-side: problem.py vs ANDES classical vs ANDES detailed."""
    try:
        import problem as pb
    except ImportError:
        print("problem.py not found — run from the exercise folder")
        return None

    cases = contingencies()
    op = np.stack(pb.injections())

    if verbose:
        print(f"  {'contingency':<46s}{'problem.py':>12s}{'ANDES cls':>11s}"
              f"{'ANDES det':>11s}")
        print("  " + "─" * 80)

    rows = []
    for c in cases:
        # problem.py
        cct_pb = pb.label_case(op, c["outage"], c["fault_bus"])

        # ANDES classical
        cct_cls = compute_cct(fault_bus=c["fault_bus"],
                              trip_line=c["outage"],
                              model="classical")

        # ANDES detailed
        cct_det = compute_cct(fault_bus=c["fault_bus"],
                              trip_line=c["outage"],
                              model="detailed")

        rows.append({"label": c["label"],
                     "problem_py": cct_pb,
                     "andes_classical": cct_cls,
                     "andes_detailed": cct_det})

        if verbose:
            print(f"  {c['label']:<46s}"
                  f"{cct_pb*1e3:>10.1f} ms"
                  f"{cct_cls*1e3:>10.1f} ms"
                  f"{cct_det*1e3:>10.1f} ms")

    return rows


def power_flow_comparison(verbose=True):
    """Compare power flow results."""
    ss_cls = build_classical()
    ss_cls.setup()
    ss_cls.PFlow.run()

    ss_det = build_detailed()
    ss_det.setup()
    ss_det.PFlow.run()

    if verbose:
        print(f"  {'Bus':>4}  {'ANDES cls':>10}  {'ANDES det':>10}")
        for i in range(N_BUS):
            print(f"  {i:>4}  {ss_cls.Bus.v.v[i]:>10.4f}  {ss_det.Bus.v.v[i]:>10.4f}")

    return {"classical": ss_cls.Bus.v.v.copy(),
            "detailed": ss_det.Bus.v.v.copy()}


# ═══════════════════════════════════════════════════════════════════════
# Main — run standalone for a quick check
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("=== Power Flow ===")
    power_flow_comparison()

    print("\n=== All Contingencies — Classical (GENCLS) ===")
    compare_all_contingencies(model="classical")

    print("\n=== All Contingencies — Detailed (GENROU+AVR+Gov) ===")
    compare_all_contingencies(model="detailed")

    print("\n=== Side-by-side (if problem.py is available) ===")
    try:
        compare_with_problem()
    except Exception as e:
        print(f"  skipped: {e}")
