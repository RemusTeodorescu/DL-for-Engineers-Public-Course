r"""Ex_12.1 — Basin Stability of a Grid: the problem.

*Deep Learning for Engineering* — MSc, Aalborg University.
Remus Teodorescu (ret@et.aau.dk), with support from Research Assistant
Noman Khan (nomank@energy.aau.dk).

**Reference texts.** Liu, *PINN with Python: An Introduction* (2025); Raissi,
Perdikaris & Karniadakis, J. Comput. Phys. **378** (2019) 686–707. For the
grid: Kundur, *Power System Stability and Control* (1994), ch. 12–13;
Menck, Heitzig, Kurths & Schellnhuber, *How dead ends undermine power grid
stability*, Nat. Commun. 5 (2014) 3969 (basin stability); Nauck et al.,
*Predicting basin stability of power grids using graph neural networks*,
New J. Phys. 24 (2022) 043041. The code, the problem and the exposition here
are original to this course.

The physics - the grid, the model, the simulation, the basin stability, the
critical clearing time - is in ``grid39.py``, shared with Ex_12.2. This file
adds what Ex_12.1 needs on top: the maps the network learns from, the
numbers a bus is described by, and the graph.
"""
import os
import urllib.request

import numpy as np
import torch

from grid39 import *          # noqa: F401,F403
from grid39 import N, IS_GEN, GEN_BUS, LINES, XY

DATA = "Ex12.1_39_edge_snbs_maps.npz"
URL = ("https://raw.githubusercontent.com/RemusTeodorescu/DL-for-Engineers-Public-Course/main/"
       "exercises/Ex12.1_39_edge_optional-basin-stability/" + DATA)
GROUPS = ("train", "test_in", "out_re", "out_line")


def load_maps():
    """The basin-stability maps built by tools/exercises/grid39/make_snbs_data.py:
    a dict per group (train, test_in, out_re, out_line) of arrays, one row per
    scenario."""
    if not os.path.exists(DATA):
        urllib.request.urlretrieve(URL, DATA)
    z = np.load(DATA, allow_pickle=False)
    maps = {grp: {k[len(grp) + 1:]: z[k] for k in z.files if k.startswith(grp + "_")} for grp in GROUPS}
    maps["held_lines"] = [int(k) for k in z["held_lines"]]
    return maps


FEATURE_NAMES = ("P", "log10(1 + H)", "machine", "converter", "E", "x_d - x'_d", "T'd0 / 10", "strength / 100")
N_FEATURE = len(FEATURE_NAMES)


def node_features(m):
    """The numbers each bus is described by, shape (scenarios, 39, 8): its
    power, its inertia (log), whether it is a machine or a converter, its
    voltage, its field constants, and how strongly the network holds it."""
    S = len(m["P"])
    strength = np.stack([-np.diag(susceptance(() if o < 0 else (int(o),))) for o in m["out"]])
    H = m["inertial"] * m["M"] * WS / 2                     # inertia constant [s]; 0 at a load bus
    return np.stack([m["P"], np.log10(1 + H), np.broadcast_to(IS_GEN, (S, N)) * (1 - m["converter"]),
                     m["converter"], m["E0"], m["X"], m["T"] / 10, strength / 100], axis=-1)


def adjacency(m):
    """The graph of each scenario, shape (scenarios, 39, 39): the branches'
    susceptance between buses (zero where there is no branch, or it is out),
    with self-loops, scaled by the buses' weighted degrees."""
    A = []
    for o in m["out"]:
        B = susceptance(() if o < 0 else (int(o),))
        W = np.where(np.eye(N, dtype=bool), 0.0, np.maximum(B, 0.0))
        W = W / W.max() + np.eye(N)
        d = 1 / np.sqrt(W.sum(1))
        A.append(d[:, None] * W * d[None, :])
    return np.stack(A)


def standard_error(p, n):
    """The Monte Carlo standard error of a share p from n kicks."""
    return np.sqrt(np.clip(p * (1 - p), 0, None) / n)


def describe_problem():
    print("IEEE 39-bus grid: 39 buses, 46 branches, 10 machines (bus 39 = the rest of the system)")
    print(f"  {len(TRIPPABLE)} branches can trip without splitting the grid")
    print("  machines: 3rd order (angle, frequency, voltage behind x'_d); converters: 2nd order; loads: 1st order")
    print("  assumed, not data: a load bus follows its neighbours within 10 ms; machine D/M 1 1/s; converter 2 s, D/M 1 1/s")
