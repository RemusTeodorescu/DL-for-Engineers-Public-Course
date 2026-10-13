r"""The IEEE 39-bus grid as a network of oscillators — the physics shared by
Ex_12.1 (basin stability) and Ex_12.2 (real-time screening).

*Deep Learning for Engineering* — MSc, Aalborg University.
Remus Teodorescu (ret@et.aau.dk), with support from Research Assistant
Noman Khan (nomank@energy.aau.dk).

This file is generated: edit ``tools/exercises/grid39/grid39_src.py`` and run
``python3 tools/exercises/grid39/sync.py``; never edit a copy in a set folder.

## The model

Every bus is an oscillator with an angle $\delta_i$ and a frequency
deviation $\omega_i$ (the Kuramoto model of a power grid, Filatrella et al.
2008). The ten machines are third order: their internal voltage $E_i$ behind
the transient reactance $x'_d$ also moves (Schmietendorf et al. 2014):

    dδ_i/dt = ω_i
    M_i dω_i/dt = P_i − D_i ω_i − Σ_j E_i E_j B_ij sin(δ_i − δ_j)
    T_i dE_i/dt = E_f,i − E_i + X_i Σ_j B_ij E_j cos(δ_i − δ_j)        machines only

with $M_i = 2H_i/\omega_s$, $B$ the lossless susceptance matrix of the
network (diagonal included), $X_i = x_d - x'_d$ and $T_i = T'_{d0}$. A load
bus keeps its voltage $E_i$ and has no inertia: its angle follows its power
balance, $D_i\,d\delta_i/dt = P_i - P_{e,i}$ (a frequency-dependent load). A **converter** (a renewable plant that replaced
a machine) keeps its voltage, has little inertia and no field winding.

## Where the numbers come from

* The network and its power flow: the IEEE 39-bus (New England) case as
  distributed with MATPOWER and pandapower (``case39``), solved there and
  copied below. Resistances are dropped: the model is lossless, and the
  losses (0.7 % of the load) are spread over the buses in proportion to
  their injection.
* The machines' $H$, $x_d$, $x'_d$, $T'_{d0}$ on 100 MVA: the ten-machine
  New England data of Pai, *Energy Function Analysis for Power System
  Stability* (1989), as reproduced in most textbooks. **Typed from memory and
  to be checked against the book before release.** Machine 1 (bus 39) is the
  equivalent of the rest of the system, $H$ = 500 s.
* **Assumed, not data:** a load bus has no inertia and follows its
  neighbours within about 10 ms ($D_i = \tau K_i$, $\tau$ = 10 ms,
  $K_i = -B_{ii}E_i^2$ the network's hold on it); the machines' damping
  ($D/M$ = 1 1/s); a converter's inertia (2 s) and damping ($D/M$ = 1 1/s). They are the model's choices and are flagged wherever a
  result depends on them.

Buses are numbered from 0 here; the IEEE numbering is one higher (bus 39 is
index 38).
"""
import numpy as np
import torch

# IEEE 39-bus (New England) case as in pandapower's case39 / MATPOWER, power flow solved there.
F = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 8, 9, 9, 12, 13, 14, 15, 15, 15, 15, 16, 16, 20, 21, 22, 22, 24, 25, 25, 25, 27, 1, 5, 9, 11, 11, 18, 18, 19, 21, 24, 28]
T = [1, 38, 2, 24, 3, 17, 4, 13, 5, 7, 6, 10, 7, 8, 38, 10, 12, 13, 14, 15, 16, 18, 20, 23, 17, 26, 21, 22, 23, 35, 25, 26, 27, 28, 28, 29, 30, 31, 10, 12, 19, 32, 33, 34, 36, 37]
X_BRANCH = [0.0411, 0.025, 0.0151, 0.0086, 0.0213, 0.0133, 0.0128, 0.0129, 0.0026, 0.0112, 0.0092, 0.0082, 0.0046, 0.0363, 0.025, 0.0043, 0.0043, 0.0101, 0.0217, 0.0094, 0.0089, 0.0195, 0.0135, 0.0059, 0.0082, 0.0173, 0.014, 0.0096, 0.035, 0.0272, 0.0323, 0.0147, 0.0474, 0.0625, 0.0151, 0.0181, 0.025, 0.02, 0.0435, 0.0435, 0.0138, 0.0142, 0.018, 0.0143, 0.0232, 0.0156]
B_CHARGING = [0.6987, 0.75, 0.2572, 0.146, 0.2214, 0.2138, 0.1342, 0.1382, 0.0434, 0.1476, 0.113, 0.1389, 0.078, 0.3804, 1.2, 0.0729, 0.0729, 0.1723, 0.366, 0.171, 0.1342, 0.304, 0.2548, 0.068, 0.1319, 0.3216, 0.2565, 0.1846, 0.361, 0, 0.531, 0.2396, 0.7802, 1.029, 0.249, -0, -0, -0, -0, -0, -0, -0, -0, -0, -0, -0]
TAP = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1.025, 1.07, 1.07, 1.006, 1.006, 1.06, 1.07, 1.009, 1.025, 1.025, 1.025]
V_PF = [1.039384, 1.048494, 1.030708, 1.00446, 1.006006, 1.008226, 0.9983973, 0.9978723, 1.038332, 1.017843, 1.013386, 1.000815, 1.014923, 1.012319, 1.016185, 1.03252, 1.034237, 1.031573, 1.050107, 0.9910105, 1.032319, 1.050143, 1.045145, 1.038001, 1.057683, 1.052561, 1.038345, 1.050374, 1.050115, 1.0499, 0.982, 0.9841, 0.9972, 1.0123, 1.0494, 1.0636, 1.0275, 1.0265, 1.03]
THETA_PF = [-0.2362583, -0.1707851, -0.2142633, -0.2203781, -0.1953432, -0.1816596, -0.2226277, -0.2327544, -0.2474605, -0.1426087, -0.1559795, -0.1570591, -0.1558566, -0.1870172, -0.1980146, -0.175115, -0.1940184, -0.2091981, -0.09442358, -0.119052, -0.1331467, -0.05555592, -0.0590144, -0.1730277, -0.1460707, -0.1647376, -0.198307, -0.1034694, -0.05532474, -0.128639, 0, -0.003288853, -0.00337153, -0.0284684, 0.03100589, 0.07798895, -0.02762679, 0.06794249, -0.2536881]
P_GEN = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.5, 6.778711, 6.5, 6.32, 5.08, 6.5, 5.6, 5.4, 8.3, 10]
Q_GEN = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.617616, 2.215745, 2.069649, 1.082928, 1.666884, 2.106614, 1.001648, -0.01369447, 0.2173273, 0.7846736]
P_LOAD = [0.976, 0, 3.22, 5, 0, 0, 2.338, 5.22, 0.065, 0, 0, 0.0853, 0, 0, 3.2, 3.29, 0, 1.58, 0, 6.8, 2.74, 0, 2.475, 3.086, 2.24, 1.39, 2.81, 2.06, 2.835, 0, 0.092, 0, 0, 0, 0, 0, 0, 0, 11.04]
SLACK = 30
XY = [[0.624, 0.257], [0.482, 0.06], [0.233, 0.108], [0.062, 0.283], [0.112, 0.528], [-0.012, 0.747], [0.19, 0.865], [0.308, 0.671], [0.541, 0.658], [-0.447, 0.61], [-0.251, 0.759], [-0.32, 0.576], [-0.358, 0.38], [-0.185, 0.206], [-0.261, -0.03], [-0.246, -0.284], [-0.0, -0.291], [0.109, -0.09], [-0.247, -0.536], [-0.212, -0.776], [-0.47, -0.273], [-0.675, -0.445], [-0.72, -0.279], [-0.46, -0.335], [0.56, -0.175], [0.458, -0.4], [0.212, -0.413], [0.495, -0.616], [0.64, -0.555], [0.705, 0.076], [-0.038, 0.981], [-0.634, 0.733], [-0.404, -0.694], [-0.262, -1.0], [-0.85, -0.595], [-0.949, -0.265], [0.798, -0.2], [0.785, -0.725], [0.686, 0.479]]

N = 39
WS = 2 * np.pi * 50                      # synchronous speed [rad/s]
LINES = list(zip(F, T))                  # branch k joins buses F[k] and T[k]

# ── the machines (Pai 1989, 100 MVA base; typed from memory - check) ──────
GEN_BUS = np.array([38, 30, 31, 32, 33, 34, 35, 36, 37, 29])    # machines 1..10
H_GEN = np.array([500, 30.3, 35.8, 28.6, 26.0, 34.8, 26.4, 24.3, 34.5, 42.0])   # [s]
XD = np.array([0.02, 0.295, 0.2495, 0.262, 0.67, 0.254, 0.295, 0.290, 0.2106, 0.10])
XDP = np.array([0.006, 0.0697, 0.0531, 0.0436, 0.132, 0.05, 0.049, 0.057, 0.057, 0.031])
TD0 = np.array([7.0, 6.56, 5.7, 5.69, 5.4, 7.3, 5.66, 6.7, 4.79, 10.2])           # [s]
LOCAL = [int(g) for g in GEN_BUS if g != 38]                    # the nine machines a converter may replace
IS_GEN = np.zeros(N, bool); IS_GEN[GEN_BUS] = True

# ── assumed, not data ─────────────────────────────────────────────────────
TAU_LOAD = 0.01       # a load bus has no inertia and follows its neighbours within about TAU_LOAD [s]
H_CONVERTER = 2.0     # a converter's inertia [s]
ALPHA_GEN = 1.0       # D/M of a machine [1/s]
ALPHA_CONVERTER = 1.0 # D/M of a converter [1/s]: no more damping than the machine it replaced


def _trippable():
    """The branches whose loss leaves the grid in one piece."""
    adj = {i: set() for i in range(N)}
    for a, b in LINES:
        adj[a].add(b); adj[b].add(a)
    out = []
    for k, (a, b) in enumerate(LINES):
        seen, stack = {a}, [a]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if (u, v) in ((a, b), (b, a)) and sum(1 for x in LINES if set(x) == {a, b}) == 1:
                    continue
                if v not in seen:
                    seen.add(v); stack.append(v)
        if len(seen) == N:
            out.append(k)
    return out


TRIPPABLE = _trippable()


def susceptance(out=()):
    """Lossless susceptance matrix [p.u.], branches ``out`` removed, each
    machine's transient reactance x'_d in series with the branch to its bus."""
    xdp = {int(g): x for g, x in zip(GEN_BUS, XDP) if g != 38}
    B = np.zeros((N, N))
    for k, (a, b) in enumerate(LINES):
        if k in out:
            continue
        y = 1.0 / (X_BRANCH[k] + xdp.get(a, 0.0) + xdp.get(b, 0.0))
        t = TAP[k]
        B[a, a] -= y / t ** 2; B[b, b] -= y
        B[a, b] += y / t; B[b, a] += y / t
        B[a, a] += B_CHARGING[k] / 2; B[b, b] += B_CHARGING[k] / 2
    return B


def electrical_power(B, d, E):
    """P_e,i = Σ_j E_i E_j B_ij sin(δ_i − δ_j), as two matrix products."""
    s, c = np.sin(d), np.cos(d)
    return E * (s * (B @ (E * c)) - c * (B @ (E * s)))


def operating_point(load=1.0, out=()):
    """The equilibrium: angles at which every bus's power balances, with the
    machines' internal voltages from the power flow. ``load`` scales every
    load and every machine together."""
    from scipy.optimize import fsolve
    V, th = np.array(V_PF), np.array(THETA_PF)
    E, d = V.copy(), th.copy()
    for g, xp in zip(GEN_BUS, XDP):
        if g == 38:
            continue
        Vt = V[g] * np.exp(1j * th[g])
        I = np.conj((P_GEN[g] + 1j * Q_GEN[g]) / Vt)
        Ei = Vt + 1j * xp * I                          # the voltage behind x'_d
        E[g], d[g] = abs(Ei), np.angle(Ei)
    P = load * (np.array(P_GEN) - np.array(P_LOAD))
    P = P - P.sum() * np.abs(P) / np.abs(P).sum()      # lossless: the losses spread over the buses
    B = susceptance(out)
    keep = np.arange(N) != SLACK
    def residual(x):
        dd = np.zeros(N); dd[keep] = x
        return (electrical_power(B, dd, E) - P)[keep]
    x, _, ok, _ = fsolve(residual, (d - d[SLACK])[keep], full_output=True, xtol=1e-12)
    dd = np.zeros(N); dd[keep] = x
    good = ok == 1 and np.abs(residual(x)).max() < 1e-8 and np.ptp(dd) < np.pi
    return B, P, E, dd, good


class Grid:
    """One state of the grid: load level, branches out, machines replaced by
    converters. Holds the model's arrays (NumPy) and, after ``.to()``, tensors."""

    def __init__(self, load=1.0, out=(), converters=()):
        out = tuple(int(k) for k in np.atleast_1d(out) if k >= 0)
        self.load, self.out, self.converters = float(load), out, tuple(int(c) for c in converters)
        B, P, E, d, ok = operating_point(load, out)
        H = np.zeros(N); H[GEN_BUS] = H_GEN
        X = np.zeros(N); X[GEN_BUS] = XD - XDP
        Tc = np.ones(N); Tc[GEN_BUS] = TD0
        K = -np.diag(B) * E ** 2                       # how strongly the network holds each bus [p.u./rad]
        field = IS_GEN.astype(float)
        for g in self.converters:                      # a converter: less inertia, no field winding
            H[g], X[g], field[g] = H_CONVERTER, 0.0, 0.0
        M = 2 * H / WS
        inertial = (H > 0).astype(float)               # 1 at a machine or converter, 0 at a load bus
        D = np.where(H > 0, np.where(np.isin(np.arange(N), self.converters), ALPHA_CONVERTER, ALPHA_GEN) * M,
                     TAU_LOAD * K)                     # a load bus: D_i dδ_i/dt = P_i − P_e,i
        self.ok = ok
        self.np = dict(B=B, P=P, M=np.where(H > 0, M, 1.0), D=D, inertial=inertial, X=X, T=Tc, field=field, d0=d, E0=E,
                       Ef=E - X * (np.cos(d) * (B @ (E * np.cos(d))) + np.sin(d) * (B @ (E * np.sin(d)))))
        self.H = H
        self.t = None

    @property
    def renewable_share(self):
        """The share of the nine local machines' generation now made by converters."""
        return float(np.sum(np.array(P_GEN)[list(self.converters)]) / np.sum(np.array(P_GEN)[LOCAL]))

    def to(self, device="cpu", dtype=torch.float32):
        self.t = {k: torch.as_tensor(v, device=device, dtype=dtype) for k, v in self.np.items()}
        self.device, self.dtype = torch.device(device), dtype
        return self


def rk4(f, z, dt):
    """One fourth-order Runge–Kutta step of dz/dt = f(z), z a tuple of tensors."""
    k1 = f(z)
    k2 = f(tuple(a + dt / 2 * b for a, b in zip(z, k1)))
    k3 = f(tuple(a + dt / 2 * b for a, b in zip(z, k2)))
    k4 = f(tuple(a + dt * b for a, b in zip(z, k3)))
    return tuple(a + dt / 6 * (b + 2 * c + 2 * e + h) for a, b, c, e, h in zip(z, k1, k2, k3, k4))


def perturbations(n, nodes, seed=0, wmax=2 * np.pi):
    """n random kicks at each of ``nodes``: the angle moved anywhere on the
    circle, the frequency by up to ``wmax`` [rad/s] (1 Hz by default)."""
    rng = np.random.default_rng(seed)
    nodes = np.repeat(np.asarray(nodes), n)
    return nodes, rng.uniform(-np.pi, np.pi, len(nodes)), rng.uniform(-wmax, wmax, len(nodes))


def returned(w, E, tol=1.0):
    """Back in step: every machine and converter within ``tol`` rad/s
    (0.16 Hz) of nominal, and every voltage above 0.2 p.u. A run that lost
    step keeps slipping at several hertz, so the verdict does not hang on the
    tolerance. (A load bus's frequency is not a state: it is zero here.)"""
    return (w.abs().amax(-1) < tol) & torch.isfinite(w).all(-1) & (E.amin(-1) > 0.2)


def draw_grid(values=None, grid=None, ax=None, label="", title="", cmap="viridis", vmin=0, vmax=1):
    """The 39 buses: machines as large dots, converters ringed in green, a
    tripped branch dotted in orange. ``values`` colours the buses."""
    import matplotlib.pyplot as plt
    use_course_style()
    xy = np.array(XY)
    ax = ax if ax is not None else plt.subplots(figsize=(6.4, 5.6))[1]
    out = grid.out if grid is not None else ()
    conv = grid.converters if grid is not None else ()
    for k, (a, b) in enumerate(LINES):
        if k in out:
            ax.plot(*xy[[a, b]].T, ls=":", lw=1.8, color=ORANGE, zorder=1)
            ax.plot(*xy[[a, b]].mean(0), "x", ms=11, mew=2.4, color=ORANGE, zorder=4)
        else:
            ax.plot(*xy[[a, b]].T, "-", lw=1.4, color=EDGE, zorder=1)
    size = np.where(IS_GEN, 230, 90)
    if values is None:
        sc = ax.scatter(*xy.T, s=size, c=np.where(IS_GEN, CYAN, MUTED), zorder=3, edgecolors=BG, linewidths=1.2)
    else:
        sc = ax.scatter(*xy.T, s=size, c=np.asarray(values), cmap=cmap, vmin=vmin, vmax=vmax, zorder=3,
                        edgecolors=BG, linewidths=1.2)
        plt.colorbar(sc, ax=ax, label=label, shrink=0.8)
    if conv:
        ax.scatter(*xy[list(conv)].T, s=520, facecolors="none", edgecolors=GREEN, linewidths=2.0, zorder=2)
    for g in GEN_BUS:
        ax.annotate(str(g + 1), xy[g], color=TXT, fontsize=8, xytext=(0, 10), textcoords="offset points", ha="center")
    ax.set_aspect("equal"); ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10, color=TXT)
    return ax


BG, TXT, MUTED, EDGE = "#07080B", "#E4ECF5", "#AFC0D2", "#5B6779"
CYAN, AMBER, ORANGE, GREEN, PURPLE = "#8FE3FA", "#FFC98A", "#FF9E7A", "#9BEDB6", "#CBB6FF"


def use_course_style():
    """The dark slide palette."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
        "text.color": TXT, "axes.labelcolor": TXT, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": "#39404F", "grid.color": "#242A36", "axes.grid": True, "grid.linewidth": 0.6,
        "font.size": 10, "axes.titlesize": 11, "legend.frameon": False, "figure.dpi": 110,
    })


# ── the dynamics, in PyTorch (batched: one row per run) ───────────────────

def rhs(g, z, fault=None):
    """The right-hand side of the model for the state z = (δ, ω, E), each of
    shape (runs, 39). ``fault``: a 0/1 mask of shape (39,) or (runs, 39); a
    faulted bus is held at zero voltage, so it exchanges no power."""
    d, w, E = z
    p = g.t
    Ev = E if fault is None else E * (1 - fault)
    s, c = torch.sin(d), torch.cos(d)
    Bs, Bc = (Ev * s) @ p["B"], (Ev * c) @ p["B"]            # Σ_j B_ij E_j sin δ_j and cos δ_j (B is symmetric)
    pe = Ev * (s * Bc - c * Bs)                               # Σ_j E_i E_j B_ij sin(δ_i − δ_j)
    q = c * Bc + s * Bs                                       # Σ_j B_ij E_j cos(δ_i − δ_j)
    a = p["inertial"]
    return (a * w + (1 - a) * (p["P"] - pe) / p["D"],               # a load bus moves with its power balance
            a * (p["P"] - p["D"] * w - pe) / p["M"],
            p["field"] * (p["Ef"] - E + p["X"] * q) / p["T"])


def start(g, runs=1):
    """The equilibrium state, repeated for ``runs`` runs."""
    p = g.t
    return (p["d0"].repeat(runs, 1).clone(), torch.zeros(runs, N, device=g.device, dtype=g.dtype),
            p["E0"].repeat(runs, 1).clone())


def kick(g, nodes, dd, dw):
    """The equilibrium with bus nodes[r] kicked by (dd[r], dw[r]) in run r."""
    z = start(g, len(nodes))
    r = torch.arange(len(nodes), device=g.device)
    n = torch.as_tensor(np.asarray(nodes), device=g.device)
    z[0][r, n] += torch.as_tensor(np.asarray(dd), device=g.device, dtype=g.dtype)
    z[1][r, n] += torch.as_tensor(np.asarray(dw), device=g.device, dtype=g.dtype) * g.t["inertial"][n]   # a load bus has no frequency of its own
    return z


def simulate(g, z, t_end, dt=0.01, f=None, keep_every=0):
    """Integrate from z for t_end seconds by RK4 on step dt. ``f`` replaces the
    right-hand side (the notebook passes its own). Returns the final state, and
    the states every ``keep_every`` steps if asked."""
    f = f or (lambda zz: rhs(g, zz))
    kept = [z] if keep_every else None
    for k in range(int(round(t_end / dt))):
        z = rk4(f, z, dt)
        if keep_every and (k + 1) % keep_every == 0:
            kept.append(z)
    return (z, kept) if keep_every else z


def basin_stability(g, n=100, seed=0, nodes=None, dt=0.01, t_end=10.0, f=None, chunk=20000):
    """Single-node basin stability: for every bus, the share of n random kicks
    after which the grid returns to step. Monte Carlo, n runs per bus."""
    nodes = list(range(N)) if nodes is None else list(nodes)
    at, dd, dw = perturbations(n, nodes, seed)
    ok = []
    for i in range(0, len(at), chunk):
        z = simulate(g, kick(g, at[i:i + chunk], dd[i:i + chunk], dw[i:i + chunk]), t_end, dt, f)
        ok.append(returned(z[1], z[2]).cpu().numpy())
    ok = np.concatenate(ok).reshape(len(nodes), n)
    return ok.mean(1)


def small_signal(g):
    """The eigenvalues of the model linearised at its equilibrium (NumPy)."""
    gg = Grid.__new__(Grid); gg.__dict__.update(g.__dict__); gg.to("cpu", torch.float64)
    z0 = torch.cat([gg.t["d0"], torch.zeros(N, dtype=torch.float64), gg.t["E0"]])
    def f(x):
        d, w, E = x[:N][None], x[N:2 * N][None], x[2 * N:][None]
        return torch.cat([v[0] for v in rhs(gg, (d, w, E))])
    J = torch.autograd.functional.jacobian(f, z0).numpy()
    keep = np.r_[np.ones(N, bool), gg.np["inertial"] > 0, np.ones(N, bool)]   # a load bus has no frequency state
    lam = np.linalg.eigvals(J[np.ix_(keep, keep)])
    return lam[np.abs(lam) > 1e-6]                       # the angle reference's zero eigenvalue removed


# ── a fault, cleared by tripping a branch: the conventional study ─────────

def branch_out(g, k):
    """The same machines, set-points and field voltages on the network with
    branch k out: what the grid is after the protection has tripped it."""
    h = Grid.__new__(Grid); h.__dict__.update(g.__dict__)
    h.np = dict(g.np); h.np["B"] = susceptance(tuple(g.out) + (int(k),))
    h.out = tuple(g.out) + (int(k),)
    return h.to(g.device, g.dtype) if g.t is not None else h


def fault_and_clear(g, fault_bus, trip, t_clear, t_after=5.0, dt=0.01, runs=None):
    """A solid fault at ``fault_bus`` from t = 0, cleared after ``t_clear``
    seconds by tripping branch ``trip``; then ``t_after`` seconds on the
    network without it. ``t_clear`` may be a tensor of clearing times, one run
    each. Returns the final state and whether the grid returned to step."""
    tc = torch.as_tensor(np.atleast_1d(t_clear), device=g.device, dtype=g.dtype)
    z = start(g, len(tc))
    mask = torch.zeros(N, device=g.device, dtype=g.dtype); mask[fault_bus] = 1
    post = branch_out(g, trip)
    t, steps = 0.0, int(np.ceil(float(tc.max()) / dt - 1e-9))
    for k in range(steps):                                  # during the fault, each run until its own clearing time
        on = (t < tc - 1e-9).to(g.dtype)[:, None]
        zn = rk4(lambda zz: rhs(g, zz, fault=mask), z, dt)
        z = tuple(on * a + (1 - on) * b for a, b in zip(zn, rk4(lambda zz: rhs(post, zz), z, dt)))
        t += dt
    z = simulate(post, z, t_after, dt)
    return z, returned(z[1], z[2])


def critical_clearing_time(g, fault_bus, trip, t_max=0.5, tol=0.005, dt=0.005, t_after=5.0):
    """The longest fault the grid survives, by bisection to ``tol`` seconds:
    one simulation per halving, as a control room's study does it."""
    lo, hi, runs = 0.0, t_max, 0
    if bool(fault_and_clear(g, fault_bus, trip, hi, t_after, dt)[1][0]):
        return t_max, 1
    while hi - lo > tol:
        mid = 0.5 * (lo + hi); runs += 1
        if bool(fault_and_clear(g, fault_bus, trip, mid, t_after, dt)[1][0]):
            lo = mid
        else:
            hi = mid
    return lo, runs + 1


# ── N-1 screening: every contingency at once (Ex_12.2) ────────────────────

def contingencies():
    """The credible contingencies: a solid fault at the first bus of each
    branch whose loss leaves the grid whole, cleared by tripping that branch.
    A list of (fault_bus, branch)."""
    return [(int(LINES[k][0]), int(k)) for k in TRIPPABLE]


def _times(x, B):
    """x @ B for x of shape (runs, 39) and B of shape (39, 39) or (runs, 39, 39)."""
    return x @ B if B.dim() == 2 else (x.unsqueeze(1) @ B).squeeze(1)


def rhs_batched(p, z, fault=None, B=None):
    """The model's right-hand side with a network B per run (runs, 39, 39)."""
    d, w, E = z
    B = p["B"] if B is None else B
    Ev = E if fault is None else E * (1 - fault)
    s, c = torch.sin(d), torch.cos(d)
    Bs, Bc = _times(Ev * s, B), _times(Ev * c, B)
    pe = Ev * (s * Bc - c * Bs)
    q = c * Bc + s * Bs
    a = p["inertial"]
    return (a * w + (1 - a) * (p["P"] - pe) / p["D"], a * (p["P"] - p["D"] * w - pe) / p["M"],
            p["field"] * (p["Ef"] - E + p["X"] * q) / p["T"])


def post_fault_networks(g, cases):
    """The network after each contingency's branch is tripped, (cases, 39, 39)."""
    return torch.stack([torch.as_tensor(susceptance(tuple(g.out) + (k,)), device=g.device, dtype=g.dtype)
                        for _, k in cases])


def screen(g, cases=None, t_clear=0.10, t_end=3.0, dt=0.01, keep_every=0, z0=None):
    """Every contingency simulated at once from the state z0 (the equilibrium
    by default): the fault for ``t_clear`` seconds, then the network without
    the tripped branch until ``t_end``. Returns whether each run stayed in
    step, and the states every ``keep_every`` steps if asked."""
    cases = contingencies() if cases is None else cases
    R = len(cases)
    z = start(g, R) if z0 is None else tuple(a.repeat(R, 1) if a.shape[0] == 1 else a for a in z0)
    mask = torch.zeros(R, N, device=g.device, dtype=g.dtype)
    mask[torch.arange(R), torch.as_tensor([b for b, _ in cases])] = 1
    Bpost = post_fault_networks(g, cases)
    kept = [z] if keep_every else None
    n_fault, n_all = int(round(t_clear / dt)), int(round(t_end / dt))
    for k in range(n_all):
        if k < n_fault:
            z = rk4(lambda zz: rhs_batched(g.t, zz, fault=mask), z, dt)
        else:
            z = rk4(lambda zz: rhs_batched(g.t, zz, B=Bpost), z, dt)
        if keep_every and (k + 1) % keep_every == 0:
            kept.append(z)
    ok = returned_soon(z)
    return (ok, kept) if keep_every else ok


def returned_soon(z, tol=1.0):
    """Still in step at the end of a screening run: no bus slipping by more
    than ``tol`` rad/s (0.16 Hz) against the grid's mean frequency, and every
    voltage above 0.2 p.u. Over a few seconds a run that lost step is
    slipping at hertz; one that held swings well under the tolerance."""
    d, w, E = z
    dev = (w - w[:, IS_GEN].mean(-1, keepdim=True))[:, IS_GEN]
    return (dev.abs().amax(-1) < tol) & torch.isfinite(w).all(-1) & (E.amin(-1) > 0.2)
