"""Ex_03 — support code for the mass-spring-damper exercise.

*Deep Learning for Engineering* — MSc, Aalborg University.
Remus Teodorescu (ret@et.aau.dk), with support from Research Assistant
Noman Khan (nomank@energy.aau.dk). The code here is original to this course;
see docs/PROVENANCE.md for what each reference text is cited for.

You do not need to edit this file, but it is worth reading. Everything that is
machine-learning plumbing lives here, so the notebook can be about physics.
Where a choice matters, the comment says why.

The exercise asks one question four ways: given poor measurements of a
vibrating mass, and knowledge of the physics that governs it, what is the best
prediction of how it moves — including at times nobody measured?
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

# ─────────────────────────────────────────────────────────────────────────────
# 1 · The physical system
# ─────────────────────────────────────────────────────────────────────────────
# A mass on a spring with a damper. The standard second-order system: vehicle
# suspensions, machine mounts, buildings in wind, anything you want to keep
# still.
#
#     m x'' + c x' + k x = 0
#
# It is pulled to x = 0.05 m and released, so x'(0) = 0.

SYSTEM = {
    "mass_kg":       0.50,      # from the drawing
    "stiffness_N_m": 20.0,      # from the spring catalogue
    "damping_Ns_m":  0.215,     # from the damper datasheet
}

X0_M = 0.05        # released from here
V0_MS = 0.0        # from rest

# The real damping is not the datasheet damping. Dampers are made to a
# tolerance, they age, and the catalogue figure was measured at a temperature
# that is not your laboratory. 13% out is unremarkable.
#
# This is the honest version of a situation every engineer meets: the physics is
# right and the constants are only approximately right.
_C_TRUE = 0.190

# The instrument is a cheap displacement sensor: random noise, plus a constant
# offset because nobody zeroed it before the test.
_NOISE_M = 0.0010     # standard deviation, metres
_BIAS_M = 0.0015      # constant offset, metres — it reads high

T_DATA_S = 5.0        # measurements stop here
T_END_S = 12.0        # predictions are wanted out to here

_SEED = 3


# ─────────────────────────────────────────────────────────────────────────────
# 2 · Numbers that follow from the physics
# ─────────────────────────────────────────────────────────────────────────────

def natural_frequency(stiffness=None, mass=None) -> float:
    """omega_n = sqrt(k/m) in rad/s — how fast it wants to oscillate."""
    k = SYSTEM["stiffness_N_m"] if stiffness is None else stiffness
    m = SYSTEM["mass_kg"] if mass is None else mass
    return float(np.sqrt(k / m))


def damping_ratio(damping=None, stiffness=None, mass=None) -> float:
    """zeta = c / (2 sqrt(km)), dimensionless.

    Below 1 the mass oscillates as it settles; above 1 it creeps back without
    overshooting. This system is near 0.03 — very lightly damped, so it rings
    for many cycles, and that is what makes it hard to predict.
    """
    c = SYSTEM["damping_Ns_m"] if damping is None else damping
    k = SYSTEM["stiffness_N_m"] if stiffness is None else stiffness
    m = SYSTEM["mass_kg"] if mass is None else mass
    return float(c / (2.0 * np.sqrt(k * m)))


def free_response(t, damping):
    """Exact solution of m x'' + c x' + k x = 0 for our release conditions.

    Used twice: with the datasheet damping it is model 1; with the true damping
    it is the answer everything is scored against.
    """
    t = np.asarray(t, dtype=float)
    wn = natural_frequency()
    z = damping_ratio(damping)
    wd = wn * np.sqrt(1.0 - z * z)          # damped frequency, slightly lower
    return np.exp(-z * wn * t) * (
        X0_M * np.cos(wd * t) + ((V0_MS + z * wn * X0_M) / wd) * np.sin(wd * t))


def truth(t):
    """What the mass really did. No experiment gives you this.

    It exists so the notebook can score predictions honestly. Fitting the
    measurements well and being right are different things, and without this you
    cannot tell them apart — which is most of the point of the exercise.
    """
    return free_response(t, _C_TRUE)


# ─────────────────────────────────────────────────────────────────────────────
# 3 · The measurements
# ─────────────────────────────────────────────────────────────────────────────

def load_measurements():
    """Returns (t, x): 101 samples at 50 ms, from t = 0 to t = 5 s.

    Noisy, offset, and stopping long before we stop caring about the answer.
    """
    rng = np.random.default_rng(_SEED)
    t = np.arange(0.0, T_DATA_S + 1e-9, 0.05)
    x = truth(t) + rng.normal(0.0, _NOISE_M, t.shape) + _BIAS_M
    return t, x


def measurement_error() -> float:
    """How wrong the measurements are in total: sqrt(noise^2 + bias^2).

    Worth knowing. A model that disagrees with the measurements by about this
    much is not failing to fit them — it is declining to fit their errors.
    """
    return float(np.sqrt(_NOISE_M ** 2 + _BIAS_M ** 2))


def find_peaks(t, x, floor=0.002):
    """Indices of positive local maxima, ignoring anything too small.

    Peak-finding is fiddly and is not what this exercise is about, so it is done
    for you. `floor` stops noise late in the record being mistaken for peaks.
    """
    return np.array([i for i in range(1, len(x) - 1)
                     if x[i] > x[i - 1] and x[i] >= x[i + 1] and x[i] > floor])


# ─────────────────────────────────────────────────────────────────────────────
# 4 · The network
# ─────────────────────────────────────────────────────────────────────────────
# One architecture, used for both the plain network and the PINN. Deliberate:
# if the two differed in shape as well as in training, you could not say which
# change caused the difference in the answer.


class FourierFeatures(nn.Module):
    """Feeds the network sin(f t) and cos(f t) for a spread of fixed f.

    Why this exists: a plain tanh network is strongly biased towards smooth,
    slowly-varying functions. Ask one to represent an oscillation and it fights
    you — it manages where data forces it to, and relaxes back to something flat
    everywhere else. That is a property of the network, not of the physics, and
    it would otherwise be blamed on the PINN.

    Handing it oscillations as raw material removes the handicap. Note the
    frequencies are a spread, not the system's own frequency: the network is
    given the vocabulary to oscillate, not the answer.

    The highest frequency must comfortably exceed omega_n or the network cannot
    express the motion at all. n_features is chosen with that in mind: with
    t_max = 12 s, 40 features reach 10.5 rad/s against omega_n = 6.3 rad/s.
    """

    def __init__(self, n_features: int = 40, t_max: float = T_END_S):
        super().__init__()
        freqs = torch.arange(1, n_features + 1, dtype=torch.float32) * np.pi / t_max
        self.register_buffer("freqs", freqs)
        self.n_out = 1 + 2 * n_features

    def forward(self, t):
        a = t * self.freqs
        return torch.cat([t, torch.sin(a), torch.cos(a)], dim=1)


def build_network(hidden: int = 64, n_features: int = 40, seed: int = 0):
    """A small fully-connected network: time in, displacement out.

    The seed is fixed so two runs differ because of what you changed, not
    because of where the weights happened to start.
    """
    torch.manual_seed(seed)
    feats = FourierFeatures(n_features)
    return nn.Sequential(feats,
                         nn.Linear(feats.n_out, hidden), nn.Tanh(),
                         nn.Linear(hidden, hidden), nn.Tanh(),
                         nn.Linear(hidden, 1))


def physics_residual(model, t_collocation, damping=None):
    """How badly the network's output breaks m x'' + c x' + k x = 0.

    This is the one piece of genuine PINN machinery. `torch.autograd.grad` is
    the same differentiation PyTorch uses to train networks, turned on the
    output instead of the loss: it gives x'(t) and then x''(t) exactly, at any t
    we like, with no finite differences and no mesh.

    The residual is divided by omega_n^2 to put it on the scale of a
    displacement. Without that it is roughly forty times larger than the data
    term, the physics drowns out the measurements, and the model underfits
    everything. This is the usual reason a first PINN disappoints.
    """
    wn = natural_frequency()
    z = damping_ratio(damping)
    x = model(t_collocation)
    x_t = torch.autograd.grad(x, t_collocation, torch.ones_like(x),
                              create_graph=True)[0]
    x_tt = torch.autograd.grad(x_t, t_collocation, torch.ones_like(x_t),
                               create_graph=True)[0]
    return (x_tt + 2.0 * z * wn * x_t + wn ** 2 * x) / wn ** 2


def train(model, t_data, x_data, w_data=1.0, w_physics=0.0, w_initial=0.0,
          epochs=6000, lr=2e-3, n_collocation=400, damping=None, report=True):
    """Trains the network. The three weights decide what kind of model it is.

        w_data      how much to believe the measurements
        w_physics   how much to believe m x'' + c x' + k x = 0
        w_initial   how much to believe x(0) and x'(0)

    With w_physics = 0 this is an ordinary neural network fit. With all three
    non-zero it is a PINN. With w_data = 0 it is a differential-equation solver
    that has never seen a measurement.

    Collocation points are where the physics is checked. They are spread over
    the whole interval, including the long stretch with no measurements — which
    is exactly how a PINN can say anything sensible out there.

    All three terms are divided by the size of the motion before being weighted,
    so that a weight of 10 means the same thing whether the mass swings 50 mm or
    50 m. Without this the weights would silently need retuning for every new
    problem, which is a common and very confusing trap.
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    td = torch.tensor(np.asarray(t_data), dtype=torch.float32).reshape(-1, 1)
    xd = torch.tensor(np.asarray(x_data), dtype=torch.float32).reshape(-1, 1)
    tc = torch.linspace(0.0, T_END_S, n_collocation).reshape(-1, 1).requires_grad_(True)
    t0 = torch.zeros(1, 1, requires_grad=True)

    scale = X0_M                       # the size of the motion
    v_scale = X0_M * natural_frequency()   # and of its velocity

    for epoch in range(epochs):
        opt.zero_grad()
        loss = torch.zeros(())

        if w_data:
            loss = loss + w_data * (((model(td) - xd) / scale) ** 2).mean()

        if w_physics:
            r = physics_residual(model, tc, damping)
            loss = loss + w_physics * ((r / scale) ** 2).mean()

        if w_initial:
            x0 = model(t0)
            v0 = torch.autograd.grad(x0, t0, torch.ones_like(x0),
                                     create_graph=True)[0]
            loss = loss + w_initial * ((((x0 - X0_M) / scale) ** 2).mean()
                                       + (((v0 - V0_MS) / v_scale) ** 2).mean())

        loss.backward()
        opt.step()

        if report and (epoch % max(1, epochs // 5) == 0 or epoch == epochs - 1):
            print(f"    epoch {epoch:>5}   loss {loss.item():.3e}")

    return model


def predict(model, t):
    """Evaluates a trained network at times `t`, returning a numpy array."""
    tt = torch.tensor(np.asarray(t, dtype=float), dtype=torch.float32).reshape(-1, 1)
    return model(tt).detach().numpy().ravel()


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# 5 · Scoring and presentation
# ─────────────────────────────────────────────────────────────────────────────

def rmse(a, b) -> float:
    """Root mean squared difference, in metres."""
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def score(prediction_fn, label=""):
    """Scores a model in the two places that matter, against the truth.

    `prediction_fn` takes an array of times and returns displacements, so it
    works for a formula and for a trained network alike.

    Returns (inside, outside) in metres:
      inside   where measurements exist   — can it fit?
      outside  where they do not          — can it predict?
    """
    t_in, _ = load_measurements()
    t_out = np.linspace(T_DATA_S, T_END_S, 400)
    inside = rmse(prediction_fn(t_in), truth(t_in))
    outside = rmse(prediction_fn(t_out), truth(t_out))
    if label:
        print(f"  {label:<32} inside {inside*1000:7.3f} mm   "
              f"outside {outside*1000:7.3f} mm")
    return inside, outside


def error_table(rows, headers):
    """A plain-text table, so results can be pasted into a report."""
    cols = list(zip(*([headers] + [[str(c) for c in r] for r in rows])))
    w = [max(len(str(c)) for c in col) for col in cols]
    line = "  ".join("-" * n for n in w)
    out = ["  ".join(h.ljust(n) for h, n in zip(headers, w)), line]
    out += ["  ".join(str(c).ljust(n) for c, n in zip(r, w)) for r in rows]
    return "\n".join(out)


def plot_models(models, title="", show_truth=True, ax=None):
    """Draws predictions against the measurements and the truth.

    `models` is a list of (label, prediction_fn, colour). The shaded band is
    where measurements exist; everything right of it is prediction.
    """
    import matplotlib.pyplot as plt

    t_d, x_d = load_measurements()
    tt = np.linspace(0.0, T_END_S, 900)
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 3.6))

    ax.axvspan(0, T_DATA_S, color="#e8f0fe", zorder=0)
    ax.text(T_DATA_S / 2, 1.14 * X0_M, "measurements", ha="center",
            color="#5b6b7f", fontsize=9)
    ax.text((T_DATA_S + T_END_S) / 2, 1.14 * X0_M, "no measurements",
            ha="center", color="#5b6b7f", fontsize=9)
    if show_truth:
        ax.plot(tt, truth(tt), color="#9aa5b1", lw=3, label="truth", zorder=1)
    ax.plot(t_d, x_d, "o", ms=3, color="#1a1a1a", label="measured", zorder=4)
    for label, fn, colour in models:
        ax.plot(tt, fn(tt), color=colour, lw=2, label=label, zorder=3)

    ax.set_xlabel("time [s]")
    ax.set_ylabel("displacement [m]")
    ax.set_ylim(-1.4 * X0_M, 1.4 * X0_M)
    ax.legend(ncol=4, frameon=False, loc="lower right", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    if title:
        ax.set_title(title, loc="left", fontsize=11)
    return ax
