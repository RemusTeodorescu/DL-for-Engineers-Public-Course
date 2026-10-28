r"""The IEEE 39-bus grid with classical machines and inertia-free converters,
for the mini projects MP12A and MP12B (course policy C14).

*Deep Learning for Engineering* — MSc, Aalborg University.
Remus Teodorescu (ret@et.aau.dk), with support from Research Assistant
Noman Khan (nomank@energy.aau.dk).

The six-bus model of Ex12.2 (``problem_12_2.py``) on the IEEE 39-bus
(New England) grid, self-contained: NumPy and PyTorch only.

* **The network** is MATPOWER's ``case39`` (Athay, Podmore & Virmani 1979, as
  distributed with MATPOWER and pandapower): 46 branches with their
  resistance, reactance, charging and transformer taps, 21 loads, ten
  machines. The data are typed below from that case; the power flow
  (:func:`power_flow`) is a Newton-Raphson written here and agrees with
  pandapower's to 1e-8 p.u. on the base case.
* **The machines are classical**: a constant voltage behind the transient
  reactance x'_d and the swing equation, no exciter, no governor - Ex12.2's
  model. H and x'_d of Pai (1989) on 100 MVA, checked against the IEEE 39-bus
  case that ships with ANDES (``ieee39_full.xlsx``, M = 2H and x'_d there on
  1000 MVA). Machine 1 at bus 39 (index 38) has H = 500 s: it stands for the
  rest of the interconnection, as the Nordic system does at bus 0 of the
  six-bus grid, and it is never replaced by a converter.
* **The damping is ASSUMED**: D_i = H_i, so a swing decays at about 0.5 1/s.
* **The loads are constant impedances** at their pre-fault voltage (ASSUMED,
  the usual choice for a classical stability study), and the network is
  reduced onto the machines' internal nodes (Kron reduction), as in Ex12.2.
* **A solid fault** is a reactance of 1e-4 p.u. from the faulted bus to earth,
  as ANDES writes it (Ex12.2 used a resistance of 1e-4 p.u.; the critical
  clearing time differs by about 1 ms); it is cleared by tripping one branch.
* **A converter** (a wind or solar plant connected through power
  electronics) replaces a machine at its bus. It injects the machine's
  pre-fault active and reactive power, so the power flow is unchanged, but it
  has **no inertia and no rotor angle**. It is modelled as a **negative
  constant impedance** - an injection that scales with the square of the
  local voltage (ASSUMED). It is the simplest model that keeps the network
  linear, so the Kron reduction still holds, and it does what a
  current-limited converter does to first order: during a fault, where the
  voltage collapses, it delivers little power, and it recovers with the
  voltage. It has no current limit, no fault ride-through control and no
  phase-locked loop; a real study would model those.

Every bus index here is 0-based (IEEE bus 39 is index 38). Units: power in
p.u. on 100 MVA, angles in rad, speeds as the deviation from synchronous
speed in rad/s, times in s, unless a name says otherwise.
"""
import itertools

import numpy as np

N = 39
BASE_MVA = 100.0
F0 = 50.0                                       #: Hz; the course's grids are 50 Hz (case39 is a 60 Hz grid: ASSUMED 50)
WS = 2 * np.pi * F0                             #: synchronous speed [rad/s]
FAULT_ADMITTANCE = -1e4j                        #: p.u.; a solid fault: a reactance of 1e-4 p.u. to earth (ANDES's form)

# ── the network: MATPOWER case39 ────────────────────────────────────────────
#  (from bus, to bus, r [p.u.], x [p.u.], total charging b [p.u.], tap ratio at the from bus)
BRANCHES = np.array([
    (0, 1, 0.0035, 0.0411, 0.6987, 1.000), (0, 38, 0.0010, 0.0250, 0.7500, 1.000),
    (1, 2, 0.0013, 0.0151, 0.2572, 1.000), (1, 24, 0.0070, 0.0086, 0.1460, 1.000),
    (2, 3, 0.0013, 0.0213, 0.2214, 1.000), (2, 17, 0.0011, 0.0133, 0.2138, 1.000),
    (3, 4, 0.0008, 0.0128, 0.1342, 1.000), (3, 13, 0.0008, 0.0129, 0.1382, 1.000),
    (4, 5, 0.0002, 0.0026, 0.0434, 1.000), (4, 7, 0.0008, 0.0112, 0.1476, 1.000),
    (5, 6, 0.0006, 0.0092, 0.1130, 1.000), (5, 10, 0.0007, 0.0082, 0.1389, 1.000),
    (6, 7, 0.0004, 0.0046, 0.0780, 1.000), (7, 8, 0.0023, 0.0363, 0.3804, 1.000),
    (8, 38, 0.0010, 0.0250, 1.2000, 1.000), (9, 10, 0.0004, 0.0043, 0.0729, 1.000),
    (9, 12, 0.0004, 0.0043, 0.0729, 1.000), (12, 13, 0.0009, 0.0101, 0.1723, 1.000),
    (13, 14, 0.0018, 0.0217, 0.3660, 1.000), (14, 15, 0.0009, 0.0094, 0.1710, 1.000),
    (15, 16, 0.0007, 0.0089, 0.1342, 1.000), (15, 18, 0.0016, 0.0195, 0.3040, 1.000),
    (15, 20, 0.0008, 0.0135, 0.2548, 1.000), (15, 23, 0.0003, 0.0059, 0.0680, 1.000),
    (16, 17, 0.0007, 0.0082, 0.1319, 1.000), (16, 26, 0.0013, 0.0173, 0.3216, 1.000),
    (20, 21, 0.0008, 0.0140, 0.2565, 1.000), (21, 22, 0.0006, 0.0096, 0.1846, 1.000),
    (22, 23, 0.0022, 0.0350, 0.3610, 1.000), (22, 35, 0.0005, 0.0272, 0.0000, 1.000),
    (24, 25, 0.0032, 0.0323, 0.5310, 1.000), (25, 26, 0.0014, 0.0147, 0.2396, 1.000),
    (25, 27, 0.0043, 0.0474, 0.7802, 1.000), (25, 28, 0.0057, 0.0625, 1.0290, 1.000),
    (27, 28, 0.0014, 0.0151, 0.2490, 1.000), (1, 29, 0.0000, 0.0181, 0.0000, 1.025),
    (5, 30, 0.0000, 0.0250, 0.0000, 1.070), (9, 31, 0.0000, 0.0200, 0.0000, 1.070),
    (11, 10, 0.0016, 0.0435, 0.0000, 1.006), (11, 12, 0.0016, 0.0435, 0.0000, 1.006),
    (18, 19, 0.0007, 0.0138, 0.0000, 1.060), (18, 32, 0.0007, 0.0142, 0.0000, 1.070),
    (19, 33, 0.0009, 0.0180, 0.0000, 1.009), (21, 34, 0.0000, 0.0143, 0.0000, 1.025),
    (24, 36, 0.0006, 0.0232, 0.0000, 1.025), (28, 37, 0.0008, 0.0156, 0.0000, 1.025)])
F = BRANCHES[:, 0].astype(int)
T = BRANCHES[:, 1].astype(int)
R, X, BC, TAP = BRANCHES[:, 2], BRANCHES[:, 3], BRANCHES[:, 4], BRANCHES[:, 5]
N_BRANCH = len(F)

#  the 21 loads: (bus, P [MW], Q [Mvar])
LOADS = np.array([
    (0, 97.6, 44.2), (2, 322.0, 2.4), (3, 500.0, 184.0), (6, 233.8, 84.0), (7, 522.0, 176.6), (8, 6.5, -66.6),
    (11, 8.53, 88.0), (14, 320.0, 153.0), (15, 329.0, 32.3), (17, 158.0, 30.0), (19, 680.0, 103.0),
    (20, 274.0, 115.0), (22, 247.5, 84.6), (23, 308.6, -92.2), (24, 224.0, 47.2), (25, 139.0, 17.0),
    (26, 281.0, 75.5), (27, 206.0, 27.6), (28, 283.5, 26.9), (30, 9.2, 4.6), (38, 1104.0, 250.0)])
LOAD_BUS = LOADS[:, 0].astype(int)
P_LOAD0, Q_LOAD0 = LOADS[:, 1] / BASE_MVA, LOADS[:, 2] / BASE_MVA

# ── the ten machines, in the order machine 1 ... 10 (IEEE buses 39, 31, ..., 38, 30) ──
GEN_BUS = np.array([38, 30, 31, 32, 33, 34, 35, 36, 37, 29])
P_GEN0 = np.array([1000.0, np.nan, 650.0, 632.0, 508.0, 650.0, 560.0, 540.0, 830.0, 250.0]) / BASE_MVA  #: set points (machine 2 is the slack)
V_GEN = np.array([1.03, 0.982, 0.9841, 0.9972, 1.0123, 1.0494, 1.0636, 1.0275, 1.0265, 1.0499])         #: voltage set points [p.u.]
SLACK = 30                                     #: bus 31 (index 30), machine 2
H_GEN = np.array([500, 30.3, 35.8, 28.6, 26.0, 34.8, 26.4, 24.3, 34.5, 42.0])   #: inertia constants [s] on 100 MVA (Pai 1989; ANDES)
XDP = np.array([0.006, 0.0697, 0.0531, 0.0436, 0.132, 0.05, 0.049, 0.057, 0.057, 0.031])   #: x'_d [p.u.] on 100 MVA
D_GEN = H_GEN.copy()                           #: damping [p.u.] - ASSUMED: D/(2H) = 0.5 1/s
NM = len(GEN_BUS)
LOCAL = np.arange(1, NM)                       #: the nine machines a converter may replace (not the interconnection at bus 39)


def ybus(out=()):
    """The bus admittance matrix [p.u.] with the branches ``out`` removed: the
    pi model of every line and transformer, taps and charging included."""
    Y = np.zeros((N, N), complex)
    for k in range(N_BRANCH):
        if k in out:
            continue
        ys = 1.0 / complex(R[k], X[k])
        t = TAP[k]
        a, b = F[k], T[k]
        Y[a, a] += (ys + 0.5j * BC[k]) / t ** 2
        Y[b, b] += ys + 0.5j * BC[k]
        Y[a, b] -= ys / t
        Y[b, a] -= ys / t
    return Y


def adjacency(out=()):
    """The 39 x 39 binary adjacency of the grid without the branches ``out``."""
    A = np.zeros((N, N))
    for k in range(N_BRANCH):
        if k not in out:
            A[F[k], T[k]] = A[T[k], F[k]] = 1.0
    return A


def _connected(out):
    A = adjacency(out)
    seen, stack = {0}, [0]
    while stack:
        u = stack.pop()
        for v in np.nonzero(A[u])[0]:
            if int(v) not in seen:
                seen.add(int(v)); stack.append(int(v))
    return len(seen) == N


#: The credible contingencies: a solid fault at the first bus of every branch
#: whose loss leaves the grid in one piece, cleared by tripping that branch -
#: (fault_bus, branch) pairs, 35 of them, as in the optional 39-bus Ex12.2.
CONTINGENCIES = [(int(F[k]), k) for k in range(N_BRANCH) if _connected((k,))]
N_CONT = len(CONTINGENCIES)


# ── the power flow ──────────────────────────────────────────────────────────

def power_flow(load=None, gen=None, tol=1e-10, max_iter=30):
    """Newton-Raphson power flow with every load scaled by ``load`` (21
    factors, constant power factor) and the nine PV machines' power by ``gen``
    (9 factors, machines 1, 3..10); the slack machine 2 takes the rest. PV
    buses hold their voltage set point (no reactive limits, as pandapower's
    default). Returns a dict - V [p.u.], th [rad], the net injections P, Q at
    every bus, the loads PL, QL per bus, the machines' Pg, Qg in GEN_BUS order -
    or None if Newton does not converge."""
    Y = ybus()
    G, B = Y.real, Y.imag
    PL = np.zeros(N); QL = np.zeros(N)
    lf = np.ones(len(LOAD_BUS)) if load is None else np.asarray(load, float)
    np.add.at(PL, LOAD_BUS, P_LOAD0 * lf); np.add.at(QL, LOAD_BUS, Q_LOAD0 * lf)
    pv_idx = [i for i in range(NM) if GEN_BUS[i] != SLACK]
    gf = np.ones(len(pv_idx)) if gen is None else np.asarray(gen, float)
    Psch = -PL.copy()
    for j, i in enumerate(pv_idx):
        Psch[GEN_BUS[i]] += P_GEN0[i] * gf[j]
    V = np.ones(N); th = np.zeros(N)
    V[GEN_BUS] = V_GEN
    pv = GEN_BUS[pv_idx]
    pq = np.array([b for b in range(N) if b not in GEN_BUS])
    ang = np.array([b for b in range(N) if b != SLACK])          # unknown angles
    for _ in range(max_iter):
        Vc = V * np.exp(1j * th)
        S = Vc * np.conj(Y @ Vc)
        dP = Psch - S.real
        dQ = -QL - S.imag
        mis = np.r_[dP[ang], dQ[pq]]
        if np.abs(mis).max() < tol:
            break
        # the Jacobian in polar form
        dS_dth = 1j * np.diag(Vc) @ np.conj(np.diag(Y @ Vc) - Y @ np.diag(Vc))
        dS_dV = np.diag(Vc) @ np.conj(Y @ np.diag(np.exp(1j * th))) + np.diag(np.exp(1j * th) * np.conj(Y @ Vc))
        J = np.block([[dS_dth.real[np.ix_(ang, ang)], dS_dV.real[np.ix_(ang, pq)]],
                      [dS_dth.imag[np.ix_(pq, ang)], dS_dV.imag[np.ix_(pq, pq)]]])
        dx = np.linalg.solve(J, mis)
        th[ang] += dx[:len(ang)]
        V[pq] += dx[len(ang):]
    else:
        return None
    Vc = V * np.exp(1j * th)
    S = Vc * np.conj(Y @ Vc)
    Pinj, Qinj = S.real, S.imag
    Pg = Pinj[GEN_BUS] + PL[GEN_BUS]
    Qg = Qinj[GEN_BUS] + QL[GEN_BUS]
    return dict(V=V, th=th, P=Pinj, Q=Qinj, PL=PL, QL=QL, Pg=Pg, Qg=Qg)


def operating_points(n, seed=0, load_range=(0.85, 1.15), jitter=0.10):
    """``n`` operating points: the system load scaled by one factor in
    ``load_range``, each load jittered by up to ``jitter``, the PV machines
    scaled by the same factor; the slack machine takes the rest. A list of
    :func:`power_flow` dicts, each with its ``scale``."""
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        s = rng.uniform(*load_range)
        op = power_flow(load=s * rng.uniform(1 - jitter, 1 + jitter, len(LOAD_BUS)), gen=np.full(NM - 1, s))
        if op is not None and op["V"].min() > 0.85:
            op["scale"] = s
            out.append(op)
    return out


# ── converters ──────────────────────────────────────────────────────────────

SHARES = (0.0, 0.2, 0.4, 0.6)                  #: converter shares of the generation
SHARE_TOL = 0.03                               #: a placement's share is within 3 points of its target


def place_converters(op, share, rng):
    """A random set of local machines whose generation is ``share`` of the
    total (within SHARE_TOL), replaced by converters: a boolean mask over the
    ten machines. Every subset of the nine local machines within the
    tolerance is equally likely; if none is, the nearest."""
    if share <= 0:
        return np.zeros(NM, bool)
    Pg = op["Pg"]
    subsets, gaps = [], []
    for k in range(1, len(LOCAL) + 1):
        for c in itertools.combinations(LOCAL, k):
            sh = Pg[list(c)].sum() / Pg.sum()
            subsets.append(c); gaps.append(abs(sh - share))
    gaps = np.array(gaps)
    ok = np.nonzero(gaps <= SHARE_TOL)[0]
    pick = subsets[int(rng.choice(ok))] if len(ok) else subsets[int(gaps.argmin())]
    mask = np.zeros(NM, bool); mask[list(pick)] = True
    return mask


def converter_share(op, conv):
    """The share of the generation delivered by converters."""
    return float(op["Pg"][conv].sum() / op["Pg"].sum())


# ── the machines, classical, and the reduced networks ───────────────────────

def internal_voltage(op):
    """E behind x'_d from the solved power flow, complex, in GEN_BUS order."""
    Vt = op["V"][GEN_BUS] * np.exp(1j * op["th"][GEN_BUS])
    return Vt + 1j * XDP * np.conj((op["Pg"] + 1j * op["Qg"]) / Vt)


def reduced(Y, op, conv, fault_bus=None):
    """Kron reduction onto the ten internal nodes. Loads are constant
    impedances at their pre-fault voltage; a converter at bus b is a negative
    constant impedance carrying the machine's pre-fault P and Q; a converter's
    internal node is disconnected (its row and column are zero); a solid
    fault is FAULT_ADMITTANCE at ``fault_bus``."""
    Ya = np.zeros((N + NM, N + NM), complex)
    Ya[:N, :N] = Y + np.diag((op["PL"] - 1j * op["QL"]) / op["V"] ** 2)
    for k, b in enumerate(GEN_BUS):
        if conv[k]:
            Ya[b, b] -= (op["Pg"][k] - 1j * op["Qg"][k]) / op["V"][b] ** 2
        else:
            y = 1.0 / complex(0.0, XDP[k])
            i = N + k
            Ya[i, i] += y; Ya[b, b] += y; Ya[i, b] -= y; Ya[b, i] -= y
    if fault_bus is not None:
        Ya[fault_bus, fault_bus] += FAULT_ADMITTANCE
    m, n = slice(N, N + NM), slice(0, N)
    return Ya[m, m] - Ya[m, n] @ np.linalg.solve(Ya[n, n], Ya[n, m])


def electrical_power(d, E, Yr):
    """P_e,i = sum_j E_i E_j (G_ij cos(d_i - d_j) + B_ij sin(d_i - d_j)),
    batched over leading axes: d (..., 10), E (..., 10), Yr (..., 10, 10)."""
    dd = d[..., :, None] - d[..., None, :]
    EE = E[..., :, None] * E[..., None, :]
    return (EE * (Yr.real * np.cos(dd) + Yr.imag * np.sin(dd))).sum(-1)


def case_networks(op, conv, fault_bus, branch):
    """The pre-fault, fault-on and post-fault reduced networks of one case,
    the internal voltages E, the initial angles d0 and the mechanical power
    Pm (the pre-fault balance), with converters' slots zero; ``active`` marks
    the machines that swing."""
    conv = np.asarray(conv, bool)
    Y0 = ybus()
    Ec = internal_voltage(op)
    E = np.where(conv, 0.0, np.abs(Ec))
    d0 = np.where(conv, 0.0, np.angle(Ec))
    pre = reduced(Y0, op, conv)
    return dict(E=E, d0=d0, pre=pre, fault=reduced(Y0, op, conv, fault_bus),
                post=reduced(ybus((branch,)), op, conv), Pm=electrical_power(d0, E, pre), active=~conv)


def stack(nets):
    """A list of :func:`case_networks` dicts as one dict of stacked arrays."""
    return {k: np.array([c[k] for c in nets]) for k in nets[0]}


# ── the swing equations, batched RK4 ────────────────────────────────────────

def simulate(cs, t_clear, dt, t_post=3.0, H=None, D=None, record_dt=None, t_from=0.0, x0=None, stop_unstable=False):
    """The swing equations of every case of ``cs`` (a :func:`stack` dict) at
    once, by the classical fourth-order Runge-Kutta method (RK4).

    The fault is applied at t = 0 and cleared at ``t_clear`` (one per case),
    **exactly**: the fault-on interval is divided into the same number of
    steps for every case, each of length t_clear/n <= dt; after clearing the
    step is ``dt`` for ``t_post`` seconds. Returns the largest angle spread
    between two machines that swing [rad] over the whole run, and if
    ``record_dt`` the times after clearing (0, record_dt, ...) with the
    angles d (n, k, 10) [rad] and speeds w (n, k, 10) [rad/s] at those times.

    ``x0 = (d, w)`` starts the post-fault simulation at ``t_from`` seconds
    after clearing from a given state instead (the fault is then not
    simulated). In torch, float64, on the CPU."""
    import torch
    Tt = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float64)
    n = len(cs["E"])
    Hs = np.tile(H_GEN, (n, 1)) if H is None else np.asarray(H)
    Ds = np.tile(D_GEN, (n, 1)) if D is None else np.asarray(D)
    E, M, Dm, Pm = Tt(cs["E"]), Tt(2 * Hs / WS), Tt(Ds), Tt(cs["Pm"])
    act = Tt(cs["active"]).bool()
    nets = {k: (Tt(cs[k].real), Tt(cs[k].imag)) for k in ("fault", "post")}
    big = torch.tensor(1e9, dtype=torch.float64)

    def spread(d):
        return torch.where(act, d, -big).max(1).values - torch.where(act, d, big).min(1).values

    def rk4(d, w, h, GB):
        G, B = GB
        def f(d, w):
            c, s = torch.cos(d) * E, torch.sin(d) * E
            Gc, Gs = torch.bmm(G, c[..., None])[..., 0], torch.bmm(G, s[..., None])[..., 0]
            Bc, Bs = torch.bmm(B, c[..., None])[..., 0], torch.bmm(B, s[..., None])[..., 0]
            pe = c * Gc + s * Gs + s * Bc - c * Bs
            return w, torch.where(act, (Pm - pe - Dm * w / WS) / M, torch.zeros_like(w))
        k1 = f(d, w)
        k2 = f(d + h / 2 * k1[0], w + h / 2 * k1[1])
        k3 = f(d + h / 2 * k2[0], w + h / 2 * k2[1])
        k4 = f(d + h * k3[0], w + h * k3[1])
        return d + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]), w + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])

    if x0 is None:
        d = Tt(cs["d0"]).clone(); w = torch.zeros_like(d)
        tc = np.broadcast_to(np.asarray(t_clear, float), (n,))
        nf = max(1, int(np.ceil(tc.max() / dt - 1e-9)))
        h = Tt(tc / nf)[:, None]
        top = spread(d)
        for _ in range(nf):                                  # the fault on, cleared exactly at t_clear
            d, w = rk4(d, w, h, nets["fault"])
            top = torch.maximum(top, spread(d))
    else:
        d, w = Tt(x0[0]).clone(), Tt(x0[1]).clone()
        top = spread(d)
    every = int(round(record_dt / dt)) if record_dt else 0
    rec_t, rec_d, rec_w = [t_from], [d.clone()], [w.clone()]
    nt = int(round((t_post - t_from) / dt))
    for k in range(nt):                                      # after clearing: the post-fault grid
        d, w = rk4(d, w, dt, nets["post"])
        top = torch.maximum(top, spread(d))
        if every and (k + 1) % every == 0:
            rec_t.append(t_from + (k + 1) * dt); rec_d.append(d.clone()); rec_w.append(w.clone())
        if stop_unstable and not every and k % 50 == 49 and bool((top > np.pi).all()):
            break
    if every:
        return top.numpy(), np.array(rec_t), torch.stack(rec_d, 1).numpy(), torch.stack(rec_w, 1).numpy()
    return top.numpy()


CCT_MAX = 0.5                                  #: s; the search for the critical clearing time stops at 500 ms
PROTECTION_TIME = 0.14                         #: s; Ex12.2's protection: insecure at 140 ms or less


def cct(cs, dt, tol, t_post=3.0, chunk=2000):
    """The critical clearing time of every case of ``cs`` [s], by bisection
    to ``tol``: the longest fault after which no two machines that swing
    separate by more than 180 degrees within ``t_post`` seconds of its
    clearing, simulated by RK4 on step ``dt``. 0 if even a fault of one
    tenth of ``tol`` loses synchronism, CCT_MAX if a fault of CCT_MAX does not."""
    n = len(cs["E"])
    out = np.zeros(n)
    for s in range(0, n, chunk):
        part = {k: v[s:s + chunk] for k, v in cs.items()}
        m = len(part["E"])
        ok = lambda tc: simulate(part, tc, dt, t_post) < np.pi
        always = ok(np.full(m, CCT_MAX))
        never = ~ok(np.full(m, 0.1 * tol))
        lo, hi = np.zeros(m), np.full(m, CCT_MAX)
        while hi[0] - lo[0] > tol:
            mid = 0.5 * (lo + hi)
            st = ok(mid)
            lo, hi = np.where(st, mid, lo), np.where(st, hi, mid)
        out[s:s + m] = np.where(never, 0.0, np.where(always, CCT_MAX, 0.5 * (lo + hi)))
    return out


# ── checks ──────────────────────────────────────────────────────────────────

def smib_check(dt=0.002, tol=1e-4, verbose=True):
    """The one case with an exact critical clearing time: one machine against
    an infinite bus, no damping, no losses, no electrical power during the
    fault. The equal area criterion gives the critical clearing angle in
    closed form, and with P_e = 0 during the fault the angle is a parabola in
    time, so the critical clearing time is

        t_c = sqrt( 4 H (d_c - d_0) / (w_s P_m) ),   cos d_c = P_m (d_u - d_0) / P_max + cos d_u

    with d_0 the pre-fault angle, d_u = pi - asin(P_m / P_max) the unstable
    equilibrium after clearing and P_max the post-fault transfer limit.
    The course's RK4 and bisection, given that case in the form of this
    module (slot 0 the infinite bus with H = 1e9 s, slot 1 the machine),
    must return it. Returns the difference [s] for three loadings."""
    rows = []
    for pm, x_pre, x_post in ((0.8, 0.45, 0.60), (0.9, 0.40, 0.50), (0.6, 0.50, 0.80)):
        E1 = 1.1
        pmax_pre, pmax_post = E1 / x_pre, E1 / x_post
        d0 = np.arcsin(pm / pmax_pre)
        du = np.pi - np.arcsin(pm / pmax_post)                 # the unstable equilibrium after clearing
        cdc = (pm * (du - d0) + pmax_post * np.cos(du)) / pmax_post
        dc = np.arccos(cdc)
        H1 = 3.5
        exact = np.sqrt(4 * H1 * (dc - d0) / (WS * pm))
        Yr = lambda x: np.array([[1 / (1j * x), -1 / (1j * x)], [-1 / (1j * x), 1 / (1j * x)]])
        pre, post = Yr(x_pre), Yr(x_post)
        fault = np.zeros((2, 2), complex); fault[0, 0] = 1 / (1j * x_pre)   # the machine cut off from the bus
        cs = dict(E=np.array([[1.0, E1]]), d0=np.array([[0.0, d0]]), pre=pre[None], fault=fault[None], post=post[None],
                  active=np.array([[True, True]]))
        Hs, Ds = np.array([[1e9, H1]]), np.zeros((1, 2))
        cs["Pm"] = electrical_power(cs["d0"], cs["E"], cs["pre"])
        lo, hi = 0.0, 0.5
        while hi - lo > tol:
            mid = 0.5 * (lo + hi)
            if simulate(cs, np.array([mid]), dt, 3.0, H=Hs, D=Ds)[0] < np.pi:
                lo = mid
            else:
                hi = mid
        rows.append((pm, exact, 0.5 * (lo + hi)))
        if verbose:
            print(f"  P_m = {pm:.2f} p.u.: equal area criterion {1000 * exact:.2f} ms, RK4 and bisection {1000 * 0.5 * (lo + hi):.2f} ms, "
                  f"difference {1000 * abs(0.5 * (lo + hi) - exact):.3f} ms")
    return np.array(rows)


def energy_check(op, conv, fault_bus, branch, t_clear=0.1, dt=0.002, verbose=True):
    """The conservation check: with the conductances of the reduced networks
    set to zero and no damping, the transient energy

        W = sum_i H_i w_i^2 / w_s - sum_i P_m,i d_i - sum_{i<j} E_i E_j B_ij cos(d_i - d_j)

    is constant after clearing. Returns its largest relative drift over 3 s
    (relative to the kinetic energy gained during the fault)."""
    c = case_networks(op, conv, fault_bus, branch)
    cs = stack([c])
    for k in ("pre", "fault", "post"):
        cs[k] = 1j * cs[k].imag                              # lossless
    cs["Pm"] = electrical_power(cs["d0"], cs["E"], cs["pre"])
    _, t, d, w = simulate(cs, np.array([t_clear]), dt, 3.0, D=np.zeros((1, NM)), record_dt=dt)
    d, w = d[0], w[0]
    act = cs["active"][0]
    B = cs["post"][0].imag
    EE = np.outer(cs["E"][0], cs["E"][0]) * B
    pot = -(d * cs["Pm"][0]).sum(1) - 0.5 * (EE[None] * np.cos(d[:, :, None] - d[:, None, :])).sum((1, 2))
    kin = (H_GEN[None] * w ** 2 / WS * act).sum(1)
    W = kin + pot
    drift = np.abs(W - W[0]).max() / kin[0]
    if verbose:
        print(f"  transient energy after clearing, lossless and undamped: kinetic energy at clearing {kin[0]:.3f} p.u.s, "
              f"largest change over 3 s {np.abs(W - W[0]).max():.2e} p.u.s ({drift:.1e} of it)")
    return drift


# ── a drawing of the grid ───────────────────────────────────────────────────
#: Bus positions for drawing (a stress-majorisation layout of the graph distances; not geography)
BUS_XY = np.array([(0.214, 0.234), (0.249, 0.352), (0.376, 0.372), (0.488, 0.317), (0.506, 0.191), (0.603, 0.108), (0.526, 0.017), (0.437, 0.089), (0.322, 0.056), (0.788, 0.247), (0.719, 0.143), (0.721, 0.242), (0.706, 0.343), (0.593, 0.396), (0.589, 0.522), (0.539, 0.642), (0.420, 0.603), (0.402, 0.488), (0.499, 0.763), (0.516, 0.885), (0.633, 0.703), (0.766, 0.719), (0.716, 0.791), (0.647, 0.675), (0.171, 0.451), (0.183, 0.578), (0.297, 0.626), (0.075, 0.597), (0.102, 0.671), (0.145, 0.307), (0.655, 0.000), (0.898, 0.221), (0.418, 0.840), (0.505, 1.000), (0.879, 0.750), (0.777, 0.891), (0.052, 0.422), (0.000, 0.726), (0.222, 0.117)])


def draw_grid(conv=None, contingency=None, ax=None, title=None):
    """The 39 buses and 46 branches; machines as circles (numbered 1 to 10),
    converters as squares; with ``contingency`` = (fault_bus, branch) the
    faulted bus as a star and the tripped branch dashed in red."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 6.5))
    conv = np.zeros(NM, bool) if conv is None else np.asarray(conv, bool)
    xy = BUS_XY
    for k in range(N_BRANCH):
        out = contingency is not None and k == contingency[1]
        ax.plot(xy[[F[k], T[k]], 0], xy[[F[k], T[k]], 1], "--" if out else "-", color="tab:red" if out else "0.55",
                lw=2.0 if out else 1.0, zorder=1)
    ax.scatter(xy[:, 0], xy[:, 1], s=14, color="0.3", zorder=2)
    for i, b in enumerate(GEN_BUS):
        ax.scatter(*xy[b], s=210, marker="s" if conv[i] else "o", color="tab:green" if conv[i] else "tab:blue",
                   edgecolor="k", zorder=3)
        ax.text(*xy[b], str(i + 1), ha="center", va="center", fontsize=7, color="w", zorder=4)
    for b in range(N):
        ax.text(xy[b, 0] + 0.012, xy[b, 1] + 0.012, str(b + 1), fontsize=6, color="0.35")
    if contingency is not None:
        ax.scatter(*xy[contingency[0]], s=260, marker="*", color="tab:red", zorder=5)
    ax.scatter([], [], s=80, marker="o", color="tab:blue", edgecolor="k", label="machine (number 1-10)")
    ax.scatter([], [], s=80, marker="s", color="tab:green", edgecolor="k", label="converter (no inertia)")
    if contingency is not None:
        ax.scatter([], [], s=120, marker="*", color="tab:red", label="fault; dashed: the branch tripped")
    ax.legend(loc="lower left", fontsize=7, frameon=False)
    ax.set_axis_off(); ax.set_aspect("equal")
    if title:
        ax.set_title(title, fontsize=9)
    return ax


# ── the cross-check: the same grid in ANDES ─────────────────────────────────
# ANDES (Cui, Li & Tomsovic, IEEE Trans. Power Systems, 2021) is an open-source
# power system simulator, the one Ex12.2 section 5 uses. Here it is given the
# same network, the same solved operating point and the same classical
# machines (its GENCLS model, M = 2H, D, x'_d), loads as constant impedances
# (its default in a time simulation), a converter as a load of negative power
# (constant impedance too), and integrates them with its own implicit
# trapezoidal method. Its fault is the same reactance of 1e-4 p.u. to earth,
# and 1e-3 p.u. when ANDES cannot converge after clearing with the smaller one
# (a milder fault: such a run is marked).

def andes_system(op, conv, fault_bus=None, branch=None, t_clear=None, t_fault=0.1, xf=1e-4):
    """The case as an ANDES system (not yet set up)."""
    import andes
    ss = andes.System(default_config=True)
    for b in range(N):
        ss.add("Bus", {"idx": b + 1, "Vn": 345.0, "v0": float(op["V"][b]), "a0": float(op["th"][b])})
    for k in range(N_BRANCH):
        ss.add("Line", {"idx": k + 1, "bus1": int(F[k]) + 1, "bus2": int(T[k]) + 1, "r": float(R[k]), "x": float(X[k]),
                        "b": float(BC[k]), "tap": float(TAP[k]), "Vn1": 345.0, "Vn2": 345.0})
    slack = 0 if conv[1] else 1                                 # machine 2 is the slack; if it is a converter, machine 1
    for i, b in enumerate(GEN_BUS):
        if conv[i]:
            ss.add("PQ", {"idx": f"C{i}", "bus": int(b) + 1, "Vn": 345.0, "p0": -float(op["Pg"][i]), "q0": -float(op["Qg"][i])})
        elif i == slack:
            ss.add("Slack", {"idx": f"G{i}", "bus": int(b) + 1, "Vn": 345.0, "v0": float(op["V"][b]),
                             "a0": float(op["th"][b]), "p0": float(op["Pg"][i])})
        else:
            ss.add("PV", {"idx": f"G{i}", "bus": int(b) + 1, "Vn": 345.0, "v0": float(op["V"][b]), "p0": float(op["Pg"][i])})
    for b in np.unique(LOAD_BUS):
        ss.add("PQ", {"idx": f"L{b}", "bus": int(b) + 1, "Vn": 345.0, "p0": float(op["PL"][b]), "q0": float(op["QL"][b])})
    for i, b in enumerate(GEN_BUS):
        if not conv[i]:
            ss.add("GENCLS", {"idx": i + 1, "bus": int(b) + 1, "gen": f"G{i}", "Vn": 345.0, "Sn": BASE_MVA, "fn": F0,
                              "M": 2 * float(H_GEN[i]), "D": float(D_GEN[i]), "xd1": float(XDP[i]), "ra": 0.0})
    if fault_bus is not None:
        ss.add("Fault", {"idx": 1, "bus": int(fault_bus) + 1, "tf": t_fault, "tc": t_fault + t_clear, "xf": xf, "rf": 0.0})
        ss.add("Toggler", {"idx": 1, "model": "Line", "dev": int(branch) + 1, "t": t_fault + t_clear})
    return ss


def andes_stable(op, conv, fault_bus, branch, t_clear, t_post=3.0, step=0.002, t_fault=0.1):
    """One ANDES run: (True, xf) if no two machines separate by more than 180
    degrees within t_post seconds of clearing, (False, xf) if they do, xf the
    fault reactance used. A run ANDES cannot finish is repeated with the
    larger fault reactance; if that fails too, (None, None)."""
    import contextlib, io, logging
    import andes
    andes.config_logger(stream_level=50); logging.getLogger("andes").setLevel(logging.CRITICAL)
    for xf in (1e-4, 1e-3):
        ss = andes_system(op, conv, fault_bus, branch, t_clear, t_fault, xf)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            ss.setup(); ss.PFlow.run()
            ss.TDS.config.tf = t_fault + t_clear + t_post; ss.TDS.config.tstep = step; ss.TDS.config.no_tqdm = 1
            ss.TDS.init(); ss.TDS.run()
        d = np.array(ss.dae.ts.x[:, ss.GENCLS.delta.a]); t = np.array(ss.dae.ts.t)
        if float((d.max(1) - d.min(1)).max()) >= np.pi:
            return False, xf
        if abs(t[-1] - ss.TDS.config.tf) < 1e-6:
            return True, xf
    return None, None


def andes_cct(op, conv, fault_bus, branch, tol=0.001):
    """The critical clearing time by ANDES, by the same bisection [s], and
    whether any run of it needed the milder fault; NaN if ANDES could not
    finish one of the runs the bisection needed."""
    top, xf = andes_stable(op, conv, fault_bus, branch, CCT_MAX)
    if top is None:
        return np.nan, True
    if top:
        return CCT_MAX, xf > 1e-4
    lo, hi, milder = 0.0, CCT_MAX, False
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        ok, xf = andes_stable(op, conv, fault_bus, branch, mid)
        if ok is None:
            return np.nan, True
        milder |= xf > 1e-4
        lo, hi = (mid, hi) if ok else (lo, mid)
    return 0.5 * (lo + hi), milder


def andes_power_flow(op, conv):
    """ANDES's power flow of the operating point: the largest differences
    from this module's in voltage [p.u.] and angle [deg]."""
    import contextlib, io
    ss = andes_system(op, conv)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        ss.setup(); ss.PFlow.run()
    return float(np.abs(ss.Bus.v.v - op["V"]).max()), float(np.degrees(np.abs(ss.Bus.a.v - op["th"])).max())
