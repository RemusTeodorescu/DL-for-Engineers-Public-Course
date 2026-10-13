"""The screening surrogate: a graph network that advances the grid's state
by 50 ms, and the screen that uses it. The same code runs in the Colab
notebook (section 7, the virtual board) and on the Jetson AGX Thor.

The network's weights are written by the notebook (section 5) as
``Ex12.2_39_edge_surrogate.pt``; the course ships a trained copy beside the notebook.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # problem_12_2_39.py and grid39.py, one folder up

import problem_12_2_39 as pb


class GraphLayer(torch.nn.Module):                            # a network: layers in __init__, the computation in forward
    def __init__(self, n_in, n_out):
        super().__init__()
        self.own = torch.nn.Linear(n_in, n_out)               # dense layer on a bus's own numbers, with the bias
        self.neighbours = torch.nn.Linear(n_in, n_out, bias=False)   # dense layer on what the neighbours send
    def forward(self, A_hat, H):
        return self.own(H) + A_hat @ self.neighbours(H)       # own numbers, plus the neighbours' summed along the branches


class Surrogate(torch.nn.Module):                             # a network: layers in __init__, the computation in forward
    """x(t + 50 ms) = x(t) + SCALE * network(graph, numbers at every bus)."""
    def __init__(self, hidden=32, depth=4):
        super().__init__()
        self.layers = torch.nn.ModuleList([GraphLayer(pb.N_FEATURE if k == 0 else hidden, hidden) for k in range(depth)])  # four graph layers
        self.head = torch.nn.Sequential(torch.nn.Linear(hidden, hidden), torch.nn.Tanh(), torch.nn.Linear(hidden, 3))   # dense layers: three changes per bus
    def forward(self, A_hat, H):
        for k, layer in enumerate(self.layers):
            Z = torch.tanh(layer(A_hat, H))                   # one graph layer, then tanh
            H = H + Z if k > 0 else Z                         # a residual connection after the first layer
        return self.head(H)


def step(net, x, A_hat, stat, Mw, scale):
    """One 50 ms step for every case at once: x (cases, 39, 3)."""
    H = torch.cat([pb.state_features(x, Mw), stat.expand(x.shape[0], -1, -1)], -1)
    return x + scale * net(A_hat, H)


class Screen:
    """Everything the screen needs for one operating state, prepared once:
    the grid, its 35 post-fault graphs, the static numbers of every bus."""
    def __init__(self, net, g):
        self.net, self.g = net, g
        self.A = pb.graphs(g).to(next(net.parameters()).dtype)
        self.stat = pb.statics(g).to(self.A.dtype)
        self.Mw = pb.inertia_weights(g).to(self.A.dtype)
        self.scale = pb.SCALE.to(g.device, self.A.dtype)
        self.steps = int(round((pb.T_HORIZON - pb.T_CLEAR) / pb.DT_NET))

    @torch.no_grad()                                          # no gradient tracking: evaluation only
    def __call__(self, frame):
        """frame: the state of the 39 buses, (39, 3) - angle, frequency, E.
        Returns each contingency's largest separation over the horizon [rad]."""
        x0 = torch.as_tensor(frame, device=self.g.device, dtype=self.g.dtype)[None]
        z = pb.fault_phase(self.g, tuple(x0[..., i] for i in range(3)))   # the fault, by RK4: 150 ms
        x = torch.stack(z, -1).to(self.A.dtype)
        worst = pb.separation(x[..., 0], self.Mw)
        for _ in range(self.steps):                           # then the surrogate, 50 ms a step
            x = step(self.net, x, self.A, self.stat, self.Mw, self.scale)
            worst = torch.maximum(worst, pb.separation(x[..., 0], self.Mw))
        return worst


def load(path, device="cpu", dtype=torch.float32):
    net = Surrogate().to(device=device, dtype=dtype)          # the network, on the device
    net.load_state_dict(torch.load(path, map_location=device))   # the trained weights
    return net.eval()                                         # evaluation mode
