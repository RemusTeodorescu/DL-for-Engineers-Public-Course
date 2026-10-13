r"""Ex_12.2 — Real-Time Screening on the Edge: the problem.

*Deep Learning for Engineering* — MSc, Aalborg University.
Remus Teodorescu (ret@et.aau.dk), with support from Research Assistant
Noman Khan (nomank@energy.aau.dk).

**Reference texts.** Liu, *PINN with Python: An Introduction* (2025); Raissi,
Perdikaris & Karniadakis, J. Comput. Phys. **378** (2019) 686–707. For the
grid: Kundur, *Power System Stability and Control* (1994), ch. 13. The code,
the problem and the exposition here are original to this course.

The physics - the grid, the model, the simulation - is in ``grid39.py``,
shared with Ex_12.1. This file adds what Ex_12.2 needs: the operating states
the surrogate is trained on, the numbers a bus is described by at every step,
the graphs after each contingency, the verdict, and the PMU recording the
real-time loop replays. The board scripts are in ``board/``.

## The contingency and the verdict

A contingency is a solid fault at the first bus of a branch, cleared after
``T_CLEAR`` = 150 ms by tripping that branch: 35 of them, one per branch
whose loss leaves the grid whole. A case is **insecure** if, within
``T_HORIZON`` = 3 s of the fault, some machine's angle moves more than 180°
from the grid's centre of inertia: it has slipped a pole and lost step.
"""
import numpy as np
import torch

from grid39 import *          # noqa: F401,F403
from grid39 import N, IS_GEN, GEN_BUS, LOCAL, WS, Grid, susceptance, contingencies, rk4, rhs_batched, start

T_CLEAR = 0.15         # the protection clears the fault after 150 ms
T_HORIZON = 3.0        # the verdict is taken over 3 s
DT_REF = 0.01          # the reference step (RK4)
DT_NET = 0.05          # the surrogate's step
CASES = contingencies()
N_CASE = len(CASES)


def operating_states(n, seed=0, max_share=0.45, load=(0.9, 1.15), max_converters=4):
    """n states of the grid: load 0.9 to 1.15 of the case, up to four of the
    nine local machines replaced by converters (share up to 45 %)."""
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        g = Grid(rng.uniform(*load), (), list(rng.choice(LOCAL, rng.integers(0, max_converters + 1), replace=False)))
        if g.ok and g.renewable_share <= max_share:
            out.append(g)
    return out


def fault_masks(device, dtype, cases=CASES):
    m = torch.zeros(len(cases), N, device=device, dtype=dtype)
    m[torch.arange(len(cases)), torch.as_tensor([b for b, _ in cases])] = 1
    return m


def post_networks(g, cases=CASES):
    """The network after each contingency's branch is tripped, (cases, 39, 39)."""
    return torch.stack([torch.as_tensor(susceptance(tuple(g.out) + (k,)), device=g.device, dtype=g.dtype)
                        for _, k in cases])


def fault_phase(g, z0=None, dt=DT_REF, cases=CASES):
    """Every contingency's fault, from the state z0 (the equilibrium by
    default) until it is cleared: T_CLEAR seconds of RK4 on the faulted grid.
    Returns the state of every case at the instant of clearing."""
    R = len(cases)
    z = start(g, R) if z0 is None else tuple(a.expand(R, -1).clone() for a in z0)
    mask = fault_masks(g.device, g.dtype, cases)
    for _ in range(int(round(T_CLEAR / dt))):
        z = rk4(lambda zz: rhs_batched(g.t, zz, fault=mask), z, dt)
    return z


def after_clearing(g, z, B_post, t=T_HORIZON - T_CLEAR, dt=DT_REF, keep_every=0):
    """RK4 on each case's own network from the clearing state, for t seconds."""
    kept = [z] if keep_every else None
    for k in range(int(round(t / dt))):
        z = rk4(lambda zz: rhs_batched(g.t, zz, B=B_post), z, dt)
        if keep_every and (k + 1) % keep_every == 0:
            kept.append(z)
    return (z, kept) if keep_every else z


def separation(d, M_inertial):
    """The largest angle of a machine from the centre of inertia [rad];
    ``d`` (..., 39), ``M_inertial`` the inertia of every machine and
    converter (0 at a load bus)."""
    coi = (d * M_inertial).sum(-1, keepdim=True) / M_inertial.sum()
    return (d - coi)[..., GEN_BUS].abs().amax(-1)


def inertia_weights(g):
    return g.t["M"] * g.t["inertial"]


def reference(g, z0=None, dt=DT_REF):
    """The reference screen of one operating state: the fault and the 2.85 s
    after it by RK4 on ``dt``. Returns the trajectories every 50 ms from
    clearing, (cases, 58, 39, 3), and each case's largest separation."""
    z = fault_phase(g, z0, dt)
    _, kept = after_clearing(g, z, post_networks(g), dt=dt, keep_every=int(round(DT_NET / dt)))
    traj = torch.stack([torch.stack(s, -1) for s in kept], 1)
    return traj, separation(traj[..., 0], inertia_weights(g)).amax(1)


# ── what the surrogate is given ───────────────────────────────────────────

STATIC_NAMES = ("P", "log10(1 + H)", "D/M / 10", "machine", "x_d - x'_d", "T'd0 / 10", "E_f", "E_0")
STATE_NAMES = ("sin(angle - COI)", "cos(angle - COI)", "frequency / 5", "E")
N_FEATURE = len(STATIC_NAMES) + len(STATE_NAMES)
SCALE = torch.tensor([0.3, 3.0, 0.05], dtype=torch.float32)     # typical change of angle [rad], frequency [rad/s] and E [p.u.] in 50 ms


def statics(g):
    """The numbers that do not change during a run, (39, 8)."""
    p = g.np
    H = p["inertial"] * p["M"] * WS / 2
    return torch.as_tensor(np.stack([p["P"], np.log10(1 + H), p["inertial"] * p["D"] / p["M"] / 10, p["field"],
                                     p["X"], p["T"] / 10, p["Ef"], p["E0"]], -1), device=g.device, dtype=g.dtype)


def graphs(g, cases=CASES):
    """Each contingency's network after clearing as the graph the surrogate
    is handed: susceptance between buses, self-loops, scaled by the degrees.
    (cases, 39, 39)."""
    A = []
    for _, k in cases:
        B = susceptance(tuple(g.out) + (k,))
        W = np.where(np.eye(N, dtype=bool), 0.0, np.maximum(B, 0.0)) / 100 + np.eye(N)
        d = 1 / np.sqrt(W.sum(1))
        A.append(d[:, None] * W * d[None, :])
    return torch.as_tensor(np.stack(A), device=g.device, dtype=g.dtype)


def state_features(x, Mw):
    """The state of every bus as the surrogate sees it: x (..., 39, 3)."""
    d, w, E = x[..., 0], x[..., 1], x[..., 2]
    dd = d - (d * Mw).sum(-1, keepdim=True) / Mw.sum()
    return torch.stack([torch.sin(dd), torch.cos(dd), w / 5, E], -1)


# ── the PMU recording the real-time loop replays ──────────────────────────

def recording(seconds=60.0, seed=7, frame=0.02, device="cpu"):
    """A minute of the grid as PMUs see it at 50 frames a second: the load
    drifts by a few per cent, two converters' output swings with the wind,
    and every reading carries noise (angle 0.01 rad, frequency 0.01 rad/s).
    Returns the frames (frames, 39, 3) and their times."""
    rng = np.random.default_rng(seed)
    g = Grid(1.0, (), (31, 33)).to(device, torch.float32)
    P0 = g.t["P"].clone()
    z = start(g, 1)
    n = int(round(seconds / frame))
    load = 1 + 0.03 * np.sin(2 * np.pi * np.arange(n) * frame / 40 + rng.uniform(0, 6))
    wind = 1 + np.cumsum(rng.normal(0, 0.01, n)).clip(-0.25, 0.25)
    frames = []
    for k in range(n):
        scale = torch.ones(N, device=device, dtype=g.dtype)
        scale[~torch.as_tensor(IS_GEN)] = float(load[k])
        scale[[31, 33]] = float(wind[k])
        P = P0 * scale
        P = P - P.sum() * P.abs() / P.abs().sum()                  # balanced, as in the model
        g.t["P"] = P
        for _ in range(int(round(frame / DT_REF))):
            z = rk4(lambda zz: rhs_batched(g.t, zz), z, DT_REF)
        x = torch.stack(z, -1)[0].clone()
        noise = torch.randn(2, N, generator=torch.Generator().manual_seed(seed * 100000 + k)).to(device, g.dtype)
        x[:, 0] += 0.01 * noise[0]                                  # angle noise 0.01 rad
        x[:, 1] += 0.01 * noise[1] * g.t["inertial"]                # frequency noise 0.01 rad/s, where there is a frequency
        frames.append(x)
    g.t["P"] = P0
    return torch.stack(frames).cpu().numpy(), np.arange(n) * frame


def describe_problem():
    print("IEEE 39-bus grid, the model of Ex_12.1: machines 3rd order, converters 2nd order, loads 1st order")
    print(f"  {N_CASE} contingencies: a solid fault at the first bus of a branch, cleared after "
          f"{1000 * T_CLEAR:.0f} ms by tripping that branch")
    print(f"  insecure: a machine more than 180 deg from the centre of inertia within {T_HORIZON:.0f} s")
