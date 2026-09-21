r"""Ex_09.1 — the physics: 2-D coupled Burgers, and a laboratory around it.

*Deep Learning for Engineering* — MSc, Aalborg University.
Remus Teodorescu (ret@et.aau.dk), with support from Research Assistant
Noman Khan (nomank@energy.aau.dk).

**Reference texts.** Liu, *PINN with Python: An Introduction* (2025); Raissi,
Perdikaris & Karniadakis, *Physics-informed neural networks*, J. Comput. Phys.
**378** (2019) 686–707. These are the works to read for the theory. The code,
the problems and the exposition here are original to this course, written from
the 2019 paper and the PyTorch documentation and not derived from any
publisher's code listings.

## The problem

    u_t + u u_x + v u_y = nu (u_xx + u_yy)
    v_t + u v_x + v v_y = nu (v_xx + v_yy)

The vector-valued, nonlinear, coupled stepping stone to Navier–Stokes: it keeps
convection and the coupling between the two components, and drops the pressure
and the incompressibility constraint. Everything that makes a flow solver hard
except the constraint that makes it *slow*.

Exact solution (verified against both equations):

    u = 3/4 - 1/(4 (1 + exp((-4x + 4y - t)/(32 nu))))
    v = 3/4 + 1/(4 (1 + exp((-4x + 4y - t)/(32 nu))))

Both components are a single tanh-like front. Writing the exponent as

    ((y - x) - t/4) / (8 nu)

says everything about the difficulty: the front lies along **y = x + t/4**, it
translates across the square as time runs, and its e-folding width in
``y - x`` is **8 nu**. With U = L = 1 the Reynolds number is simply 1/nu, so
raising Re thins the front in exact proportion. At Re = 20 that width is 0.4,
a substantial fraction of a unit side; at Re = 500 it is 0.016, and a
collocation set that resolved the first will not resolve the second. That
single number is what notebook 03 sweeps, and the reason a smooth network
eventually stops being able to follow the solution.

Both the boundary data and the initial data are taken from the exact solution,
so this is a **pure verification problem**: every error the notebooks measure
is the model's, never the data's.

## What else is in here

Everything below the physics is a laboratory: a configuration object, a
runner, an ipywidgets control panel, the plots and an automatic report
generator. None of it contains physics. The residual and the loss are written
by the student in notebook 01 and passed *in* — this module only calls them.

That separation is the point of the set. The knobs are exposed so that a
parameter study is a slider movement rather than an edit, and the report is
assembled mechanically so that the marks go on the interpretation.
"""

from __future__ import annotations

import os
import time

import numpy as np
import torch

from course_core import (DEVICE, SEED, MLP, parameter_count, set_seed,
                         to_numpy, to_tensor)
from pinn_core import (boundary_points_in_time, describe, grid_points,
                       initial_points, max_abs_error, relative_l2,
                       spacetime_points, train_two_stage)

__all__ = [
    "BURGERS_DOMAIN", "T_END", "NU_DEFAULT", "RE_DEFAULT", "U_REF", "L_REF",
    "exact_u", "exact_v", "reynolds_from_nu", "nu_from_reynolds",
    "describe_problem",
    "FlowPINN",
    "sample_domain", "sample_boundary_xt",
    "eval_field", "field_errors",
    "Config", "run_study", "control_panel",
    "plot_fields", "plot_error_vs_time", "make_report",
    "OUTPUT_DIR",
]

# ══════════════════════════════════════════════════════════════════════════
# 1 · the reference solution
# ══════════════════════════════════════════════════════════════════════════

#: ``((x_lo, x_hi), (y_lo, y_hi))`` — the unit square.
BURGERS_DOMAIN = ((0.0, 1.0), (0.0, 1.0))

#: End of the time window. The front crosses a quarter of the square in it.
T_END = 1.0

#: Reference velocity and length. Both unity, which is what makes Re = 1/nu.
U_REF = 1.0
L_REF = 1.0

#: The default Reynolds number, and the viscosity it implies.
RE_DEFAULT = 20.0
NU_DEFAULT = 0.05

#: Where the notebooks write everything they save.
OUTPUT_DIR = "Ex09.1_outputs"


def exact_u(x, y, t, nu):
    """The x-component of the exact solution. NumPy in, NumPy out."""
    return 0.75 - 0.25 / (1.0 + np.exp((-4 * x + 4 * y - t) / (32 * nu)))


def exact_v(x, y, t, nu):
    """The y-component of the exact solution. NumPy in, NumPy out."""
    return 0.75 + 0.25 / (1.0 + np.exp((-4 * x + 4 * y - t) / (32 * nu)))


def reynolds_from_nu(nu, U=U_REF, L=L_REF):
    """Re = UL/nu. With U = L = 1 the Reynolds number is simply 1/nu."""
    return U * L / nu


def nu_from_reynolds(Re, U=U_REF, L=L_REF):
    """The viscosity a given Reynolds number implies, at U = L = 1."""
    return U * L / Re


def describe_problem(nu: float = NU_DEFAULT) -> None:
    """Print the problem and the numbers a reader should sanity-check.

    Nothing here is asserted: the ranges are evaluated from the exact solution
    on the domain, so the printout is a measurement of the formulas above
    rather than a claim about them.
    """
    Re = reynolds_from_nu(nu)
    _, _, pts = grid_points(101, 101, BURGERS_DOMAIN)
    x, y = pts[:, 0], pts[:, 1]
    u0, v0 = exact_u(x, y, 0.0, nu), exact_v(x, y, 0.0, nu)
    u1, v1 = exact_u(x, y, T_END, nu), exact_v(x, y, T_END, nu)

    print("  2-D COUPLED BURGERS  (nonlinear, vector-valued, two equations)")
    print(f"    domain          : [{BURGERS_DOMAIN[0][0]:.0f}, "
          f"{BURGERS_DOMAIN[0][1]:.0f}] x [{BURGERS_DOMAIN[1][0]:.0f}, "
          f"{BURGERS_DOMAIN[1][1]:.0f}]")
    print(f"    time window     : 0 .. {T_END:.2f}")
    print(f"    viscosity       : nu = {nu:.4g}")
    print(f"    Reynolds        : Re = U L / nu = {Re:g}   (U = L = 1)")
    print("    front           : the line y = x + t/4")
    print(f"    front width     : {8 * nu:.4g} in (y - x)   [e-folding, = 8 nu]")
    print(f"    u at t = 0      : {u0.min():.4f} .. {u0.max():.4f}"
          f"        at t = {T_END:g}: {u1.min():.4f} .. {u1.max():.4f}")
    print(f"    v at t = 0      : {v0.min():.4f} .. {v0.max():.4f}"
          f"        at t = {T_END:g}: {v1.min():.4f} .. {v1.max():.4f}")
    print()
    print("    u + v = 3/2 everywhere and for all time, which is a free")
    print("    diagnostic: a model that violates it has not learned the")
    print("    coupling, whatever its residual says.")


# ══════════════════════════════════════════════════════════════════════════
# 2 · the model
# ══════════════════════════════════════════════════════════════════════════

class FlowPINN(torch.nn.Module):
    """A single trunk with two outputs, (u, v).

    One network rather than two: the components are physically coupled, so
    they should share hidden features. The control panel in notebook 02 asks
    you to test that claim, at matched parameter count.

    Built from :class:`MLP`, so the activation, the initialisation and the
    device are the course's everywhere-else defaults and not this set's own.
    The attributes ``n_in``, ``n_out``, ``n_hidden`` and ``n_layers`` are set
    so that :func:`describe` can report the network like any other.
    """

    def __init__(self, n_hidden=40, n_layers=5, activation=torch.nn.Tanh,
                 separate=False):
        super().__init__()
        self.separate = separate
        self.n_in, self.n_out = 3, 2
        self.n_hidden, self.n_layers = n_hidden, n_layers
        if separate:
            self.net_u = MLP(n_in=3, n_out=1, n_hidden=n_hidden,
                             n_layers=n_layers, activation=activation)
            self.net_v = MLP(n_in=3, n_out=1, n_hidden=n_hidden,
                             n_layers=n_layers, activation=activation)
        else:
            self.net = MLP(n_in=3, n_out=2, n_hidden=n_hidden,
                           n_layers=n_layers, activation=activation)
        self.to(DEVICE)

    def forward(self, xyt):
        if self.separate:
            return torch.cat([self.net_u(xyt), self.net_v(xyt)], dim=1)
        return self.net(xyt)


# ══════════════════════════════════════════════════════════════════════════
# 3 · sampling
# ══════════════════════════════════════════════════════════════════════════
#
# Both functions return **NumPy**, following the shared samplers they are
# built on. Wrap them at the point of use:
#
#     xyt_f = to_tensor(sample_domain(4000), requires_grad=True)
#     xyt_b = to_tensor(sample_boundary_xt(20, 12))
#
# The collocation set is differentiated through and needs the flag; the
# boundary set is only evaluated and does not.


def sample_domain(n, domain=BURGERS_DOMAIN, t_end=T_END, method="lhs",
                  seed=None):
    """``n`` collocation points through space and time, shaped ``(n, 3)``.

    Time is the last column, as everywhere in this course.
    """
    return spacetime_points(n, domain, (0.0, float(t_end)), method,
                            SEED if seed is None else seed)


def sample_boundary_xt(n_per_edge=20, n_t=12, domain=BURGERS_DOMAIN,
                       t_end=T_END, include_ic=True, n_ic=200, seed=None):
    """The four edges across time, plus the ``t = 0`` face.

    Both conditions are taken from the exact solution, so the exercise is a
    pure verification problem: any error is the model's, not the data's.

    Returns ``(N, 3)`` NumPy, the boundary tube first and the initial slice
    last, so a plot can slice the two apart.
    """
    s = SEED if seed is None else seed
    pts = [boundary_points_in_time(n_per_edge, n_t, domain,
                                   (0.0, float(t_end)), seed=s)]
    if include_ic:
        pts.append(initial_points(n_ic, domain, t0=0.0,
                                  seed=(SEED + 5) if seed is None else seed))
    return np.concatenate(pts, axis=0)


# ══════════════════════════════════════════════════════════════════════════
# 4 · evaluation
# ══════════════════════════════════════════════════════════════════════════

def eval_field(model, t, nu, k=101):
    """Predicted and exact (u, v) on a ``k`` x ``k`` grid at one instant.

    Returns ``(X, Y, U, V, U_exact, V_exact)``, every field shaped like the
    meshgrid, ready for ``contourf``.
    """
    X, Y, pts = grid_points(k, k, BURGERS_DOMAIN, t=float(t))
    with torch.no_grad():
        out = to_numpy(model(to_tensor(pts)))
    U = out[:, 0].reshape(X.shape)
    V = out[:, 1].reshape(X.shape)
    return X, Y, U, V, exact_u(X, Y, t, nu), exact_v(X, Y, t, nu)


def field_errors(model, nu, ts=(0.0, 0.25, 0.5, 0.75, 1.0), k=81):
    """Relative L2 and worst-point error of each component, instant by instant.

    One number for a whole space–time slab hides the thing worth knowing: a
    transient problem is not uniformly hard.
    """
    rows = []
    for t in ts:
        X, Y, U, V, Ue, Ve = eval_field(model, t, nu, k)
        rows.append({
            "t": float(t),
            "u_rel": relative_l2(U, Ue),
            "v_rel": relative_l2(V, Ve),
            "u_max": max_abs_error(U, Ue),
            "v_max": max_abs_error(V, Ve),
        })
    return rows


# ══════════════════════════════════════════════════════════════════════════
# 5 · the study
# ══════════════════════════════════════════════════════════════════════════

class Config:
    """Every knob the exercise exposes, in one place.

    ``adam_epochs`` and ``lbfgs_epochs`` are the two stages of
    :func:`train_two_stage`. An L-BFGS *step* there is an outer step of up to
    twenty inner iterations with a strong-Wolfe line search, so the stage is
    worth more per unit than the Adam one — which is the whole point of the
    handoff, and what the panel invites you to measure.
    """

    def __init__(self, n_collocation=4000, n_boundary_per_edge=20, n_times=12,
                 reynolds=RE_DEFAULT, n_hidden=40, n_layers=5,
                 separate_nets=False, adam_epochs=2000, lbfgs_epochs=200,
                 t_end=T_END, seed=88):
        self.n_collocation = int(n_collocation)
        self.n_boundary_per_edge = int(n_boundary_per_edge)
        self.n_times = int(n_times)
        self.reynolds = float(reynolds)
        self.n_hidden = int(n_hidden)
        self.n_layers = int(n_layers)
        self.separate_nets = bool(separate_nets)
        self.adam_epochs = int(adam_epochs)
        self.lbfgs_epochs = int(lbfgs_epochs)
        self.t_end = float(t_end)
        self.seed = int(seed)

    @property
    def nu(self):
        return nu_from_reynolds(self.reynolds)

    def __repr__(self):
        return (f"Config(Re={self.reynolds:g}, nu={self.nu:.4g}, "
                f"N_f={self.n_collocation}, net={self.n_hidden}x{self.n_layers}, "
                f"separate={self.separate_nets})")


def run_study(cfg, residual_fn, loss_fn_factory, verbose=True):
    """Train one model for the given configuration and return a result dict.

    The two callables are yours, written in notebook 01:

        residual_fn(model, xyt, nu) -> (res_u, res_v)
        loss_fn_factory(model, xyt_f, xyt_b, nu) -> callable taking no arguments

    The second returns a closure because that is what :func:`train_two_stage`
    expects: a zero-argument function returning a scalar tensor, called
    repeatedly inside the L-BFGS line search.

    Points are drawn once, before training, and never resampled. L-BFGS is
    full-batch by construction and a point set that moved under it would
    invalidate its curvature estimate.
    """
    set_seed(cfg.seed)
    model = FlowPINN(cfg.n_hidden, cfg.n_layers, separate=cfg.separate_nets)

    xyt_f = to_tensor(
        sample_domain(cfg.n_collocation, t_end=cfg.t_end, seed=cfg.seed),
        requires_grad=True)
    xyt_b = to_tensor(
        sample_boundary_xt(cfg.n_boundary_per_edge, cfg.n_times,
                           t_end=cfg.t_end, seed=cfg.seed))

    if verbose:
        print(f"  {cfg}")
        describe(model, cfg.n_collocation)

    # train_two_stage reports history["lbfgs"][-1] unconditionally, so a
    # zero-length L-BFGS stage would raise on the way out. One outer step is
    # the closest this can get to "no L-BFGS at all".
    lbfgs_steps = max(1, cfg.lbfgs_epochs)

    t0 = time.time()
    hist = train_two_stage(model,
                           loss_fn_factory(model, xyt_f, xyt_b, cfg.nu),
                           adam_steps=cfg.adam_epochs,
                           lbfgs_steps=lbfgs_steps,
                           lr=1e-3,
                           report_every=500 if verbose else 0)
    wall = time.time() - t0

    # Five instants spanning the window. At the default t_end = 1 these are
    # exactly field_errors' own defaults, 0, 0.25, 0.5, 0.75, 1.
    errs = field_errors(model, cfg.nu,
                        ts=tuple(np.linspace(0.0, cfg.t_end, 5)))
    res = {
        "config": cfg, "model": model, "history": hist, "seconds": wall,
        "errors": errs, "n_params": parameter_count(model),
        "final_loss": float(hist["lbfgs"][-1]),
        "mean_u_rel": float(np.mean([e["u_rel"] for e in errs])),
        "mean_v_rel": float(np.mean([e["v_rel"] for e in errs])),
        "n_boundary": int(xyt_b.shape[0]),
    }
    if verbose:
        print(f"\n{cfg}\n  wall {wall:.1f}s   final loss {res['final_loss']:.3e}"
              f"   mean rel L2  u {res['mean_u_rel']:.3e}"
              f"  v {res['mean_v_rel']:.3e}")
    return res


# ══════════════════════════════════════════════════════════════════════════
# 6 · the control panel
# ══════════════════════════════════════════════════════════════════════════

def _pseudo_dimension(n_in, n_hidden, n_layers):
    """P* = (p + 1) + (N_n + 1) N_L — Liu (2025), Eq. (2.19).

    A count of *neurons*, not of weights, and the quantity the sampling
    condition of L7.1 is stated in. The panel shows it beside the trainable
    parameter count, because the two say different things and the exercise
    quotes both.
    """
    return (n_in + 1) + (n_hidden + 1) * n_layers


def control_panel(on_run, defaults=None):
    """Build the ipywidgets panel. ``on_run(cfg)`` is called when Run is pressed."""
    import ipywidgets as W
    from IPython.display import display

    d = defaults or Config()
    style = {"description_width": "150px"}
    lay = W.Layout(width="420px")

    w = {
        "n_collocation": W.IntSlider(value=d.n_collocation, min=500, max=20000, step=500,
                                     description="Collocation points", style=style, layout=lay),
        "n_boundary_per_edge": W.IntSlider(value=d.n_boundary_per_edge, min=5, max=60, step=5,
                                           description="Boundary pts / edge", style=style, layout=lay),
        "reynolds": W.FloatLogSlider(value=d.reynolds, base=10, min=0.5, max=2.7, step=0.1,
                                     description="Reynolds number", style=style, layout=lay),
        "n_hidden": W.IntSlider(value=d.n_hidden, min=10, max=100, step=10,
                                description="Neurons per layer", style=style, layout=lay),
        "n_layers": W.IntSlider(value=d.n_layers, min=2, max=8, step=1,
                                description="Hidden layers", style=style, layout=lay),
        "separate_nets": W.Checkbox(value=d.separate_nets, description="Separate nets for u, v",
                                    style=style, layout=lay),
        "adam_epochs": W.IntSlider(value=d.adam_epochs, min=500, max=8000, step=500,
                                   description="Adam epochs", style=style, layout=lay),
        "lbfgs_epochs": W.IntSlider(value=d.lbfgs_epochs, min=0, max=800, step=50,
                                    description="L-BFGS epochs", style=style, layout=lay),
    }
    readout = W.HTML()

    def _update(*_):
        nu = nu_from_reynolds(w["reynolds"].value)
        p_star = _pseudo_dimension(3, w["n_hidden"].value, w["n_layers"].value)
        ratio = w["n_collocation"].value / p_star
        warn = "" if ratio >= 10 else " &nbsp;<b style='color:#c60'>low</b>"
        n_par = parameter_count(FlowPINN(w["n_hidden"].value, w["n_layers"].value,
                                         separate=w["separate_nets"].value))
        readout.value = (f"<div style='font-family:monospace'>nu = {nu:.4g} &nbsp;|&nbsp; "
                         f"front width = {8*nu:.3g} &nbsp;|&nbsp; "
                         f"P* = {p_star} &nbsp;|&nbsp; N_f / P* = {ratio:.1f}{warn}"
                         f" &nbsp;|&nbsp; parameters = {n_par} &nbsp;|&nbsp; "
                         f"N_f / params = {w['n_collocation'].value / n_par:.1f}</div>")
    for x in w.values():
        x.observe(_update, "value")
    _update()

    run = W.Button(description="Run study", button_style="success",
                   icon="play", layout=W.Layout(width="180px"))
    out = W.Output()

    def _click(_):
        with out:
            out.clear_output()
            cfg = Config(**{k: v.value for k, v in w.items()})
            on_run(cfg)

    run.on_click(_click)
    display(W.VBox([
        W.HTML("<h3>Ex_09.1 &mdash; 2-D Burgers control panel</h3>"
               "<p>Change any parameter, then press <b>Run study</b>. "
               "Each run is recorded so you can compare them in notebook 04.</p>"),
        W.HBox([W.VBox(list(w.values())[:4]), W.VBox(list(w.values())[4:])]),
        readout, run, out]))
    return w


# ══════════════════════════════════════════════════════════════════════════
# 7 · plots
# ══════════════════════════════════════════════════════════════════════════

def plot_fields(result, t=0.5):
    """Model, exact and |error| for both components, at one instant."""
    import matplotlib.pyplot as plt
    cfg = result["config"]
    X, Y, U, V, Ue, Ve = eval_field(result["model"], t, cfg.nu)
    fig, ax = plt.subplots(2, 3, figsize=(13, 6.4))
    for row, (P, E, nm) in enumerate([(U, Ue, "u"), (V, Ve, "v")]):
        for a, (D, ttl) in zip(ax[row], [(P, f"{nm} PINN"), (E, f"{nm} exact"),
                                         (np.abs(P - E), f"|{nm} error|")]):
            im = a.contourf(X, Y, D, 40, cmap="magma")
            a.set_title(f"{ttl}   (t = {t})"); a.set_aspect("equal")
            fig.colorbar(im, ax=a)
    plt.tight_layout(); plt.show()
    return fig


def plot_error_vs_time(results, labels=None):
    """One curve per run: relative L2 error in u against time."""
    import matplotlib.pyplot as plt
    from course_core import CYCLE
    labels = labels or [repr(r["config"]) for r in results]
    plt.figure(figsize=(7.5, 3.6))
    for i, (r, lb) in enumerate(zip(results, labels)):
        ts = [e["t"] for e in r["errors"]]
        ev = [e["u_rel"] for e in r["errors"]]
        plt.semilogy(ts, ev, "o-", color=CYCLE[i % len(CYCLE)], label=lb)
    plt.xlabel("t"); plt.ylabel("relative $L_2$ error in u")
    plt.grid(alpha=0.25, which="both")
    plt.legend(fontsize=8); plt.tight_layout(); plt.show()


# ══════════════════════════════════════════════════════════════════════════
# 8 · the report
# ══════════════════════════════════════════════════════════════════════════

def make_report(results, filename=None, author="", notes=""):
    """Assemble a markdown report from a list of run results.

    The numbers are produced mechanically. The five questions at the end are
    not, and they are what the set is marked on.
    """
    if filename is None:
        filename = os.path.join(OUTPUT_DIR, "Ex09.1_report.md")
    parent = os.path.dirname(filename)
    if parent:
        os.makedirs(parent, exist_ok=True)

    L = ["# Ex_09.1 - 2-D Burgers with a PINN", ""]
    if author:
        L.append(f"**Author:** {author}  ")
    L += [f"**Runs recorded:** {len(results)}", "",
          "## Configurations and results", "",
          "| # | Re | nu | N_f | net | params | wall (s) | final loss | mean rel L2 (u) | mean rel L2 (v) |",
          "|---|----|----|-----|-----|--------|----------|------------|-----------------|-----------------|"]
    for i, r in enumerate(results, 1):
        c = r["config"]
        L.append(f"| {i} | {c.reynolds:g} | {c.nu:.4g} | {c.n_collocation} | "
                 f"{c.n_hidden}x{c.n_layers}{' sep' if c.separate_nets else ''} | "
                 f"{r['n_params']} | {r['seconds']:.1f} | {r['final_loss']:.3e} | "
                 f"{r['mean_u_rel']:.3e} | {r['mean_v_rel']:.3e} |")
    L += ["", "## Error against time", "",
          "| run | " + " | ".join(f"t={e['t']:g}" for e in results[0]["errors"]) + " |",
          "|---|" + "---|" * len(results[0]["errors"])]
    for i, r in enumerate(results, 1):
        L.append(f"| {i} | " + " | ".join(f"{e['u_rel']:.2e}" for e in r["errors"]) + " |")
    L += ["", "## Your interpretation", "",
          "Answer each in a short paragraph, referring to the table above.", "",
          "1. **Collocation count.** How did the error respond as N_f rose? Where did it",
          "   plateau, and what does the plateau tell you about the limiting factor?", "",
          "2. **Reynolds number.** At what Re did your model stop converging? What did the",
          "   failure look like - a large error, or a plausible but wrong smooth field?", "",
          "3. **Architecture.** Did a shared trunk beat two separate networks at equal",
          "   parameter count? Why would you expect that, physically?", "",
          "4. **Optimiser.** How much of the final accuracy came from the L-BFGS stage?", "",
          "5. **Honest assessment.** For the case you ran, would you use a PINN or a",
          "   classical solver, and on what grounds?", ""]
    if notes:
        L += ["## Notes", "", notes, ""]
    with open(filename, "w") as fh:
        fh.write("\n".join(L))
    print(f"wrote {filename}  ({len(results)} runs)")
    return filename
