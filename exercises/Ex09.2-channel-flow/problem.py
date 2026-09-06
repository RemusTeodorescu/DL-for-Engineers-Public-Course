r"""Ex_09.2 — the physics: steady mean flow past an obstacle in a channel.

*Deep Learning for Engineering* — MSc, Aalborg University.
Remus Teodorescu (ret@et.aau.dk), with support from Research Assistant
Noman Khan (nomank@energy.aau.dk).

**Reference texts.** Liu, *PINN with Python: An Introduction* (2025); Raissi,
Perdikaris & Karniadakis, *Physics-informed neural networks*, J. Comput. Phys.
**378** (2019) 686–707. These are the works to read for the theory. The code,
the problem and the exposition here are original to this course, written from
the 2019 paper and the PyTorch documentation and not derived from any
publisher's code listings. Where a symbol matches a textbook's it is because
both follow the standard notation of the field.

## The problem

Steady mean flow through a rectangular channel containing one obstacle. The
student chooses the shape; the solver, the sampling and the reporting are the
same in every case.

    (u·∇)u = −∇p + ν_eff ∇²u,        ∇·u = 0

``ν_eff = ν + ν_t`` is the effective viscosity of L9.2. Prescribing ν_t turns
an unsolvable turbulent problem into the laminar one of L9.1 — which is the
whole trick, and also the whole weakness, and saying which of the two it is in
your own case is what this exercise is marked on.

Incompressibility is hard-enforced: the network outputs a **stream function**
ψ and a pressure p, and the velocity is taken as ``u = ψ_y``, ``v = −ψ_x``. So
``∇·u = 0`` holds identically and there is no continuity term in the loss at
all — the hard-enforcement route of L9.1 slide 8, applied to a constraint
rather than to a boundary condition.

## Why this file carries its own samplers

The domain is **not a rectangle**. It is a rectangle with a hole in it, and the
hole moves and changes shape under student control.

``pinn_core`` samples rectangles, and that is deliberate: a shared library that
tried to sample arbitrary geometry would either be a mesh generator or a lie.
An exercise set with real geometry is expected to supply its own samplers, and
these are this set's:

    sample_channel     interior points, rejecting anything inside the obstacle
    sample_walls       the two channel walls
    sample_inlet       x = 0
    sample_outlet      x = L
    sample_obstacle    the obstacle surface, plus a graded near-wall cloud

Only the first of them touches the shared machinery: it draws its candidate
batches with :func:`pinn_core.interior_points` over the bounding rectangle and
then rejects. The rest are one-dimensional and are written out here.

They follow the shared library's convention and **return NumPy arrays**. Wrap
them with :func:`to_tensor` at the point of use — with
``requires_grad=True`` for anything the network is differentiated at, which
here means all of them, because the velocity itself is a derivative of ψ.

## The geometry is a level set

Every shape exposes the same ``phi(x, y)``: negative inside, zero on the
surface, positive outside. The solver never asks which shape it was given, so
adding a sixth shape means writing one function and touching nothing else.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from pinn_core import (SEED, DEVICE, set_seed, to_tensor, to_numpy,
                       MLP, parameter_count, describe, grad,
                       interior_points, grid_points, train_two_stage)

__all__ = [
    "SHAPES", "CHANNEL", "DOMAIN", "LBFGS_INNER", "Obstacle", "ChannelPINN",
    "sample_channel", "sample_walls", "sample_inlet", "sample_outlet",
    "sample_obstacle", "inlet_profile", "eddy_viscosity",
    "drag_coefficient", "pressure_drop",
    "PipeConfig", "run_case", "control_panel", "plot_case", "plot_shape",
    "make_report",
]

#: The five obstacle shapes. Adding one means adding a branch to
#: :meth:`Obstacle.phi` and to :meth:`Obstacle.outline`, and nothing else.
SHAPES = ("circle", "square", "ellipse", "diamond", "aerofoil")

#: Channel length and height, non-dimensional.
CHANNEL = dict(L=4.0, H=1.0)

#: ``((x_lo, x_hi), (y_lo, y_hi))`` — the bounding rectangle of the channel,
#: in the form ``pinn_core``'s samplers and ``grid_points`` expect. The flow
#: domain is this box **minus** the obstacle; see :func:`sample_channel`.
DOMAIN = ((0.0, CHANNEL["L"]), (0.0, CHANNEL["H"]))

#: L-BFGS iterations inside one :func:`pinn_core.train_two_stage` step. The
#: shared optimiser takes ``lbfgs_steps`` outer steps of ``max_iter=20``, so a
#: configuration asking for 300 L-BFGS iterations wants 15 of them. Converting
#: here keeps ``PipeConfig.lbfgs_epochs`` meaning what it has always meant — a
#: number of L-BFGS iterations — and keeps the runtime what it has always been.
LBFGS_INNER = 20


class Obstacle:
    """Signed level set: negative inside, zero on the surface, positive outside.

    Every shape exposes the same interface, so the solver never needs to know
    which one was chosen.
    """

    def __init__(self, shape="circle", xc=1.2, yc=0.5, size=0.2, aspect=1.0):
        if shape not in SHAPES:
            raise ValueError(f"shape must be one of {SHAPES}")
        self.shape, self.xc, self.yc = shape, float(xc), float(yc)
        self.size, self.aspect = float(size), float(aspect)

    @property
    def D(self):
        """Projected height - the length scale for drag and Strouhal."""
        return 2 * self.size * (self.aspect if self.shape in ("ellipse", "aerofoil") else 1.0)

    def phi(self, x, y):
        """Level set, accepting numpy arrays or torch tensors."""
        bk = torch if torch.is_tensor(x) else np
        dx, dy = (x - self.xc) / self.size, (y - self.yc) / self.size
        if self.shape == "circle":
            return dx ** 2 + dy ** 2 - 1.0
        if self.shape == "ellipse":
            return dx ** 2 + (dy / self.aspect) ** 2 - 1.0
        if self.shape == "square":
            return bk.maximum(abs(dx), abs(dy)) - 1.0
        if self.shape == "diamond":
            return abs(dx) + abs(dy) - 1.0
        # Aerofoil: a smooth teardrop, blunt at the nose and tapered aft.
        # The thickness formula is only meaningful for |dx| <= 1, so the level
        # set is intersected with the chord - without that, points far
        # downstream on the centreline read as lying ON the surface.
        if bk is np:
            thick = np.clip(self.aspect * (1.0 - dx)
                            * np.sqrt(np.clip((dx + 1.0) / 2.0, 0.0, None)), 0.0, None)
            return np.maximum(np.abs(dy) - thick, np.abs(dx) - 1.0)
        thick = torch.clamp(self.aspect * (1.0 - dx)
                            * torch.sqrt(torch.clamp((dx + 1.0) / 2.0, min=0.0)), min=0.0)
        return torch.maximum(torch.abs(dy) - thick, torch.abs(dx) - 1.0)

    def inside(self, x, y):
        return self.phi(x, y) < 0

    def outline(self, n=400):
        """Polyline of the surface, for plotting."""
        t = np.linspace(0, 2 * np.pi, n)
        if self.shape == "circle":
            return self.xc + self.size * np.cos(t), self.yc + self.size * np.sin(t)
        if self.shape == "ellipse":
            return (self.xc + self.size * np.cos(t),
                    self.yc + self.size * self.aspect * np.sin(t))
        if self.shape == "square":
            c = np.maximum(np.abs(np.cos(t)), np.abs(np.sin(t)))
            return self.xc + self.size * np.cos(t) / c, self.yc + self.size * np.sin(t) / c
        if self.shape == "diamond":
            c = np.abs(np.cos(t)) + np.abs(np.sin(t))
            return self.xc + self.size * np.cos(t) / c, self.yc + self.size * np.sin(t) / c
        s = np.linspace(-1, 1, n // 2)
        th = self.aspect * (1 - s) * np.sqrt(np.clip((s + 1) / 2, 0, None))
        return (self.xc + self.size * np.concatenate([s, s[::-1]]),
                self.yc + self.size * np.concatenate([th, -th[::-1]]))


class ChannelPINN(torch.nn.Module):
    """Outputs the stream function and pressure, (psi, p).

    Taking the velocity from psi makes the flow divergence-free by
    construction - the hard-enforcement route of L9.1 slide 8.

    A thin wrapper around the shared :class:`MLP`: two inputs ``(x, y)``, two
    outputs ``(psi, p)``, tanh throughout. The size attributes are copied onto
    the wrapper so :func:`pinn_core.describe` can report on it directly.
    """

    def __init__(self, n_hidden=48, n_layers=6, activation=torch.nn.Tanh):
        super().__init__()
        self.net = MLP(n_in=2, n_out=2, n_hidden=n_hidden, n_layers=n_layers,
                       activation=activation)
        self.n_in, self.n_out = 2, 2
        self.n_hidden, self.n_layers = int(n_hidden), int(n_layers)

    def forward(self, xy):
        return self.net(xy)

    def velocity(self, xy):
        """``(u, v, p)`` at ``xy``, which must have ``requires_grad=True``.

        ``u`` and ``v`` are derivatives of the network output, so a tensor
        made without ``requires_grad`` gives an autograd error here rather
        than a wrong answer — which is the good failure mode.
        """
        out = self.net(xy)
        psi, p = out[:, 0:1], out[:, 1:2]
        g = grad(psi, xy)
        return g[:, 1:2], -g[:, 0:1], p        # u = psi_y, v = -psi_x


# ------------------------------------------------------------------- sampling
def sample_channel(n, obs, chan=None, seed=None):
    """Interior points, rejecting anything inside the obstacle.

    Candidate batches come from the shared rectangle sampler over the bounding
    box; the rejection test is this set's own, because the shared library has
    no notion of a hole. Returns ``(n, 2)`` NumPy.

    The threshold is ``phi > 0.04`` rather than ``phi > 0``: a collocation
    point sitting on the surface has no boundary layer on one side of it, and
    the residual there is dominated by the singularity rather than by the
    physics. The graded cloud from :func:`sample_obstacle` covers that band
    instead.
    """
    ch = chan or CHANNEL
    rng = np.random.default_rng(SEED if seed is None else seed)
    domain = ((0.0, ch["L"]), (0.0, ch["H"]))
    keep = np.empty((0, 2))
    while len(keep) < n:
        c = interior_points(max(2 * n, 1024), domain, method="lhs",
                            seed=int(rng.integers(0, 2 ** 31 - 1)))
        keep = np.vstack([keep, c[obs.phi(c[:, 0], c[:, 1]) > 0.04]])
    return keep[:n]


def sample_walls(n, chan=None):
    """The two channel walls, ``n`` points along each. NumPy, ``(2n, 2)``."""
    ch = chan or CHANNEL
    x = np.linspace(0, ch["L"], n)
    return np.concatenate([np.column_stack([x, np.zeros(n)]),
                           np.column_stack([x, np.full(n, ch["H"])])])


def sample_inlet(n, chan=None):
    """``n`` points on the inlet plane ``x = 0``. NumPy, ``(n, 2)``."""
    ch = chan or CHANNEL
    y = np.linspace(0, ch["H"], n)
    return np.column_stack([np.zeros(n), y])


def sample_outlet(n, chan=None):
    """``n`` points on the outlet plane ``x = L``. NumPy, ``(n, 2)``."""
    ch = chan or CHANNEL
    y = np.linspace(0, ch["H"], n)
    return np.column_stack([np.full(n, ch["L"]), y])


def sample_obstacle(n, obs, grade=2.2, seed=None):
    """Points on the obstacle surface, plus a graded cloud just outside it.

    The graded cloud is what resolves the boundary layer - see L9.1 slide 14.

    Returns ``(surface, near)``, both NumPy.
    """
    xs, ys = obs.outline(n)
    surf = np.column_stack([xs, ys])
    rng = np.random.default_rng((SEED + 7) if seed is None else seed)
    idx = rng.integers(0, len(surf), n * 3)
    off = np.power(rng.random(n * 3), grade) * obs.size * 1.6
    nx = surf[idx, 0] - obs.xc
    ny = surf[idx, 1] - obs.yc
    nn = np.hypot(nx, ny) + 1e-12
    near = np.column_stack([surf[idx, 0] + off * nx / nn,
                            surf[idx, 1] + off * ny / nn])
    near = near[obs.phi(near[:, 0], near[:, 1]) > 0.01]
    return surf, near


# -------------------------------------------------------------------- physics
def inlet_profile(y, U=1.0, chan=None, kind="poiseuille"):
    """Prescribed inlet velocity: uniform, or fully developed parabolic."""
    ch = chan or CHANNEL
    if kind == "uniform":
        return U * (torch.ones_like(y) if torch.is_tensor(y) else np.ones_like(y))
    eta = y / ch["H"]
    return 6.0 * U * eta * (1.0 - eta)


def eddy_viscosity(Re, model="none", Cmu=0.09):
    """Effective viscosity, non-dimensional (nu = 1/Re with U = D = 1).

    'none'    laminar, nu_eff = 1/Re
    'uniform' a crude constant eddy viscosity - the simplest closure of L9.2
    """
    nu = 1.0 / Re
    if model == "none":
        return nu
    if model == "uniform":
        return nu + Cmu * 0.01           # a deliberately crude closure
    raise ValueError("model must be 'none' or 'uniform'")


def drag_coefficient(model, obs, nu_eff, rho=1.0, U=1.0, n=400):
    """C_D from pressure and shear integrated over the obstacle surface.

    Approximate: it uses the surface normal from the level set and a
    one-sided velocity gradient. Good enough to compare shapes, not a
    certified value - see L9.2 slide 12 on the validation gap.
    """
    xs, ys = obs.outline(n)
    P = to_tensor(np.column_stack([xs, ys]), requires_grad=True)
    u, v, p = model.velocity(P)
    gx = torch.autograd.grad(obs.phi(P[:, 0:1], P[:, 1:2]).sum(), P,
                             create_graph=False, retain_graph=True)[0]
    nrm = torch.sqrt((gx ** 2).sum(dim=1, keepdim=True)) + 1e-12
    nx, ny = gx[:, 0:1] / nrm, gx[:, 1:2] / nrm
    gu = grad(u, P)
    shear = nu_eff * (gu[:, 0:1] * nx + gu[:, 1:2] * ny)
    ds = np.hypot(np.diff(xs, append=xs[0]), np.diff(ys, append=ys[0]))
    ds = to_tensor(ds.reshape(-1, 1))
    F = ((-p * nx + shear) * ds).sum()
    return float(F.item() / (0.5 * rho * U ** 2 * obs.D))


def pressure_drop(model, chan=None, n=120):
    """Mean pressure at the inlet minus mean pressure at the outlet."""
    ch = chan or CHANNEL
    A = to_tensor(sample_inlet(n, ch))
    B = to_tensor(sample_outlet(n, ch))
    with torch.no_grad():
        pa = model(A)[:, 1].mean().item()
        pb = model(B)[:, 1].mean().item()
    return float(pa - pb)


# -------------------------------------------------------- the control surface
class PipeConfig:
    """Everything the exercise exposes.

    The geometry itself is a parameter here — the shape, its size and its
    position are all under the student's control, which is what makes the
    shape comparison of notebook 03 possible and what makes it easy to compare
    two shapes at different blockages and think you have measured shape.
    """

    def __init__(self, shape="circle", size=0.2, aspect=1.0, x_pos=1.2,
                 reynolds=100.0, inlet_speed=1.0, inlet_kind="poiseuille",
                 closure="none", n_collocation=6000, n_surface=250,
                 n_hidden=48, n_layers=6, adam_epochs=3000, lbfgs_epochs=300,
                 seed=88):
        self.shape, self.size, self.aspect, self.x_pos = shape, float(size), float(aspect), float(x_pos)
        self.reynolds, self.inlet_speed = float(reynolds), float(inlet_speed)
        self.inlet_kind, self.closure = inlet_kind, closure
        self.n_collocation, self.n_surface = int(n_collocation), int(n_surface)
        self.n_hidden, self.n_layers = int(n_hidden), int(n_layers)
        self.adam_epochs, self.lbfgs_epochs = int(adam_epochs), int(lbfgs_epochs)
        self.seed = int(seed)

    @property
    def obstacle(self):
        return Obstacle(self.shape, self.x_pos, CHANNEL["H"] / 2, self.size, self.aspect)

    @property
    def nu_eff(self):
        return eddy_viscosity(self.reynolds, self.closure)

    @property
    def blockage(self):
        return self.obstacle.D / CHANNEL["H"]

    def __repr__(self):
        return (f"PipeConfig({self.shape}, Re={self.reynolds:g}, "
                f"blockage={self.blockage:.2f}, closure={self.closure}, "
                f"N_f={self.n_collocation})")


def run_case(cfg, residual_fn, loss_fn_factory, verbose=True):
    """Train one configuration and return the engineering quantities.

    The samplers hand back NumPy; every point set is wrapped here with
    ``requires_grad=True``, including the boundary sets, because the velocity
    is itself a derivative of the network output and so even a no-slip term
    differentiates through its points.
    """
    set_seed(cfg.seed)
    obs = cfg.obstacle
    model = ChannelPINN(cfg.n_hidden, cfg.n_layers).to(DEVICE)
    surf, near = sample_obstacle(cfg.n_surface, obs, seed=cfg.seed)
    pts = {
        "f": to_tensor(sample_channel(cfg.n_collocation, obs, seed=cfg.seed),
                       requires_grad=True),
        "walls": to_tensor(sample_walls(120), requires_grad=True),
        "inlet": to_tensor(sample_inlet(80), requires_grad=True),
        "outlet": to_tensor(sample_outlet(80), requires_grad=True),
        "surf": to_tensor(surf, requires_grad=True),
        "near": to_tensor(near, requires_grad=True),
    }
    if verbose:
        describe(model, cfg.n_collocation)

    t0 = time.time()
    hist = train_two_stage(model, loss_fn_factory(model, pts, cfg),
                           adam_steps=cfg.adam_epochs,
                           lbfgs_steps=max(1, cfg.lbfgs_epochs // LBFGS_INNER),
                           lr=1e-3,
                           report_every=750 if verbose else 0)
    wall = time.time() - t0

    try:
        cd = drag_coefficient(model, obs, cfg.nu_eff, U=cfg.inlet_speed)
    except Exception as e:
        cd = float("nan")
        if verbose:
            print("  drag failed:", e)
    dp = pressure_drop(model)

    res = {"config": cfg, "model": model, "history": hist, "seconds": wall,
           "final_loss": float(hist["lbfgs"][-1]), "C_D": cd, "dp": dp,
           "n_params": parameter_count(model)}
    if verbose:
        print(f"\n{cfg}\n  wall {wall:.1f}s  loss {res['final_loss']:.3e}  "
              f"C_D {cd:.3f}  dp {dp:.4f}")
    return res


# -------------------------------------------------------------------- pictures
def plot_shape(cfg):
    """Preview the geometry and the sample distribution before training."""
    import matplotlib.pyplot as plt
    obs = cfg.obstacle
    pts = sample_channel(2500, obs, seed=cfg.seed)
    surf, near = sample_obstacle(cfg.n_surface, obs, seed=cfg.seed)
    xs, ys = obs.outline()
    plt.figure(figsize=(11, 3.0))
    plt.scatter(pts[:, 0], pts[:, 1], s=2, alpha=0.4, label="interior")
    plt.scatter(near[:, 0], near[:, 1], s=3, alpha=0.6, label="graded near-wall")
    plt.plot(xs, ys, "w-", lw=1.5)
    plt.xlim(0, CHANNEL["L"]); plt.ylim(0, CHANNEL["H"])
    plt.gca().set_aspect("equal"); plt.legend(fontsize=8)
    plt.title(f"{cfg.shape}, blockage {cfg.blockage:.2f}")
    plt.tight_layout(); plt.show()


def plot_case(result, k=(360, 90)):
    """Speed, pressure and streamwise velocity on a grid, obstacle masked out."""
    import matplotlib.pyplot as plt
    cfg = result["config"]; obs = cfg.obstacle
    X, Y, pts = grid_points(k[0], k[1], DOMAIN)
    P = to_tensor(pts, requires_grad=True)
    u, v, p = result["model"].velocity(P)
    U = to_numpy(u).reshape(X.shape)
    V = to_numpy(v).reshape(X.shape)
    PR = to_numpy(p).reshape(X.shape)
    mask = obs.phi(X, Y) < 0
    for A in (U, V, PR):
        A[mask] = np.nan
    xs, ys = obs.outline()
    fig, ax = plt.subplots(3, 1, figsize=(11, 7.2))
    for a, (D, ttl) in zip(ax, [(np.sqrt(U ** 2 + V ** 2), "speed"),
                                (PR, "pressure"), (U, "u")]):
        im = a.contourf(X, Y, D, 50, cmap="magma")
        a.plot(xs, ys, "w-", lw=1.4); a.set_aspect("equal"); a.set_title(ttl)
        fig.colorbar(im, ax=a)
    plt.tight_layout(); plt.show()
    return fig


def control_panel(on_run, defaults=None):
    """The widget panel. ``on_run(cfg)`` is called when Run case is pressed."""
    import ipywidgets as W
    from IPython.display import display

    d = defaults or PipeConfig()
    style = {"description_width": "140px"}; lay = W.Layout(width="400px")
    w = {
        "shape": W.Dropdown(options=list(SHAPES), value=d.shape,
                            description="Obstacle shape", style=style, layout=lay),
        "size": W.FloatSlider(value=d.size, min=0.05, max=0.35, step=0.01,
                              description="Size", style=style, layout=lay),
        "aspect": W.FloatSlider(value=d.aspect, min=0.3, max=2.5, step=0.1,
                                description="Aspect ratio", style=style, layout=lay),
        "x_pos": W.FloatSlider(value=d.x_pos, min=0.6, max=2.5, step=0.1,
                               description="Position along pipe", style=style, layout=lay),
        "reynolds": W.FloatLogSlider(value=d.reynolds, base=10, min=0.7, max=3.5, step=0.1,
                                     description="Reynolds number", style=style, layout=lay),
        "inlet_speed": W.FloatSlider(value=d.inlet_speed, min=0.2, max=3.0, step=0.1,
                                     description="Inlet speed U", style=style, layout=lay),
        "inlet_kind": W.Dropdown(options=["poiseuille", "uniform"], value=d.inlet_kind,
                                 description="Inlet profile", style=style, layout=lay),
        "closure": W.Dropdown(options=["none", "uniform"], value=d.closure,
                              description="Turbulence closure", style=style, layout=lay),
        "n_collocation": W.IntSlider(value=d.n_collocation, min=1000, max=25000, step=1000,
                                     description="Collocation points", style=style, layout=lay),
        "n_hidden": W.IntSlider(value=d.n_hidden, min=20, max=100, step=10,
                                description="Neurons per layer", style=style, layout=lay),
        "n_layers": W.IntSlider(value=d.n_layers, min=3, max=9, step=1,
                                description="Hidden layers", style=style, layout=lay),
        "adam_epochs": W.IntSlider(value=d.adam_epochs, min=500, max=10000, step=500,
                                   description="Adam epochs", style=style, layout=lay),
    }
    readout = W.HTML()

    def _update(*_):
        cfg = PipeConfig(**{k: v.value for k, v in w.items()})
        # P* is Liu's pseudo-dimension, (p + 1) + (N_n + 1) N_L with p = 2. It
        # counts neurons, not weights, and it is the number the sampling
        # condition is written against.
        p_star = 3 + (w["n_hidden"].value + 1) * w["n_layers"].value
        ratio = w["n_collocation"].value / p_star
        warn = "" if ratio >= 10 else " &nbsp;<b style='color:#c60'>low</b>"
        block = cfg.blockage
        bwarn = "" if block < 0.5 else " &nbsp;<b style='color:#c60'>very blocked</b>"
        readout.value = (f"<div style='font-family:monospace'>nu_eff = {cfg.nu_eff:.4g}"
                         f" &nbsp;|&nbsp; P* = {p_star} &nbsp;|&nbsp; N_f/P* = {ratio:.1f}{warn}"
                         f" &nbsp;|&nbsp; blockage = {block:.2f}{bwarn}</div>")
    for x in w.values():
        x.observe(_update, "value")
    _update()

    prev = W.Button(description="Preview geometry", icon="search",
                    layout=W.Layout(width="190px"))
    run = W.Button(description="Run case", button_style="success", icon="play",
                   layout=W.Layout(width="160px"))
    out = W.Output()

    def _cfg():
        return PipeConfig(**{k: v.value for k, v in w.items()})

    def _prev(_):
        with out:
            out.clear_output(); plot_shape(_cfg())

    def _run(_):
        with out:
            out.clear_output(); on_run(_cfg())

    prev.on_click(_prev); run.on_click(_run)
    display(W.VBox([
        W.HTML("<h3>Ex_09.2 &mdash; obstacle in a pipe</h3>"
               "<p>Choose a shape and set the flow. <b>Preview geometry</b> is free; "
               "<b>Run case</b> trains a model. Every run is recorded for the report.</p>"),
        W.HBox([W.VBox(list(w.values())[:6]), W.VBox(list(w.values())[6:])]),
        readout, W.HBox([prev, run]), out]))
    return w


def make_report(results, filename="Ex09.2_report.md", author="", notes=""):
    """Write the case table and the five report questions to a Markdown file."""
    L = ["# Ex_09.2 - Flow past an obstacle in a pipe", ""]
    if author:
        L.append(f"**Author:** {author}  ")
    L += [f"**Cases run:** {len(results)}", "", "## Cases", "",
          "| # | shape | blockage | Re | closure | N_f | params | wall (s) | final loss | C_D | dp |",
          "|---|-------|----------|----|---------|-----|--------|----------|------------|-----|----|"]
    for i, r in enumerate(results, 1):
        c = r["config"]
        L.append(f"| {i} | {c.shape} | {c.blockage:.2f} | {c.reynolds:g} | {c.closure} | "
                 f"{c.n_collocation} | {r['n_params']} | {r['seconds']:.1f} | "
                 f"{r['final_loss']:.3e} | {r['C_D']:.3f} | {r['dp']:.4f} |")
    L += ["", "## Your interpretation", "",
          "1. **Shape.** Compare two shapes at the same Reynolds number and the same",
          "   blockage. Which had the higher drag, and why physically? Refer to where",
          "   the flow separates.", "",
          "2. **Reynolds number.** How did drag and pressure drop vary with Re? At what",
          "   Re did the solution begin to degrade, and how did the degradation appear -",
          "   as an obvious failure, or as a plausible but smeared field?", "",
          "3. **Closure.** Compare `closure='none'` with `closure='uniform'` at the same",
          "   Re. What changed, and what does that tell you about the eddy-viscosity",
          "   hypothesis?", "",
          "4. **Sampling.** Did the near-wall grading change your drag figure? Why is",
          "   drag more sensitive to it than the pressure drop is?", "",
          "5. **Honest assessment.** Would you present any of these numbers to a client?",
          "   State what validation would be needed first.", ""]
    if notes:
        L += ["## Notes", "", notes, ""]
    with open(filename, "w") as fh:
        fh.write("\n".join(L))
    print(f"wrote {filename}  ({len(results)} cases)")
    return filename
