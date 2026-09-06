# Ex_09.2 — Turbulent flow: an obstacle in a channel

**Paired with L9.2 · Turbulent Flow · Part 2**

Flow past an obstacle in a channel, with **your choice of shape** — circle,
square, ellipse, diamond or aerofoil — and a control panel for geometry,
Reynolds number, inlet speed, closure and sampling.

This is the exercise where the right answer may be *refusing to answer*.
The marks are for saying precisely what would have to be true first.

## Goals

By the end you can

1. write the steady RANS momentum residual with automatic differentiation, and
   say why there is no continuity term in the loss;
2. hard-enforce incompressibility through a **stream function**, and explain
   what that buys compared with penalising $\nabla\cdot\mathbf{u}$;
3. sample a domain that is **not a rectangle** — reject the obstacle, grade the
   points towards its surface, and say what the grading is for;
4. turn a trained field into the two numbers an engineer asks for, drag
   coefficient and pressure drop, and name what makes each of them unreliable;
5. compare shapes **at matched blockage**, and recognise an unmatched
   comparison as a measurement of area rather than of shape;
6. state the Reynolds number beyond which you would not report your own result,
   and what validation would be needed to change that.

## The problem

Steady mean flow through a channel of length 4 and height 1, non-dimensional,
containing one obstacle:

```
(u·∇)u = −∇p + ν_eff ∇²u      in the channel, outside the obstacle
∇·u    = 0
u = v  = 0                     on the channel walls and the obstacle surface
u = prescribed profile, v = 0  at the inlet
p = 0                          at the outlet, which also fixes the pressure gauge
```

`ν_eff = ν + ν_t` is the effective viscosity of L9.2. Prescribing `ν_t` — here
either nothing at all (`closure="none"`, laminar) or a crude constant
(`closure="uniform"`) — turns an unsolvable turbulent problem into the laminar
one of L9.1. That substitution is the whole of the exercise's power and the
whole of its weakness.

Incompressibility is **hard-enforced**: the network outputs a stream function
and a pressure, and the velocity is taken as `u = ψ_y`, `v = −ψ_x`. So
`∇·u = 0` holds identically and there is no continuity term in the loss at all.

The geometry is a **signed level set** — negative inside, zero on the surface,
positive outside — so the solver never asks which shape it was given. Adding a
sixth shape means writing one function and touching nothing else. The domain is
therefore a rectangle with a moving hole in it, which is why `problem.py`
carries its own samplers: `pinn_core` samples rectangles, and a set with real
geometry supplies the rest itself.

There is **no exact solution here**. Nothing in this set measures true error;
it measures drag, pressure drop and your own judgement.

## The notebooks

Run in order; later notebooks load results saved by earlier ones.

```
Ex09.2_00_geometry_lab.ipynb       shapes, level sets and sampling — read-only
Ex09.2_01_channel_flow.ipynb       write the residual and the loss — has TODOs
Ex09.2_02_control_panel.ipynb      interactive parameter study
Ex09.2_03_shape_comparison.ipynb   compare shapes at matched blockage
Ex09.2_04_report.ipynb             assemble the report for submission
```

Everything they write goes to `Ex09.2_outputs/`.

## Files

Every Part 2 exercise has the same three modules beside it. Only the third
differs between sets.

| | |
|---|---|
| `course_core.py` | shared by the whole course — `set_seed`, `MLP`, `to_tensor`, `check` |
| `pinn_core.py` | the PDE machinery — `grad`, `d2`, samplers, `train_two_stage` |
| `problem.py` | **this** problem — shapes, level sets, the channel samplers, the control panel, drag and pressure drop |

The first two are generated. Edit `tools/pinn/*.py` and run
`python3 tools/pinn/sync_cores.py`; never edit a copy.

**Colab needs the whole folder**, not just a notebook — the imports expect the
three modules beside them.

Requires `torch`, `numpy`, `matplotlib` and — for notebook 02 — **`ipywidgets`**,
which is preinstalled on Colab and is `pip install ipywidgets` locally.

See also `MiniProject_Turbulence.md`, which defines the L13 project building on
this exercise and on Ex_09.1.

## Expected runtime

CPU only; no GPU is needed anywhere in Part 2. A single case is one to three
minutes. Notebook 03 runs five of them back to back, and the control panel in
notebook 02 runs as many as you press the button for, so budget 30–60 minutes
for the two of them together.

If a cell seems stuck it probably is not: L-BFGS reports rarely, so a long
silence after the Adam output is normal.

## What to hand in

- your residual and loss, and the flow field for at least two shapes
- the shape comparison **at matched blockage** — an unmatched comparison
  measures blockage, not shape
- the Reynolds number beyond which you would not report your result
- an explicit refusal: name a quantity your model produces that you would
  not hand to anyone, and say what would have to change first

## Things that go wrong, and what they mean

**Shapes rank differently than you expected.** Check the blockage ratio is
actually matched. Most surprising rankings are a comparison of areas.

**Drag comes out negative or absurd.** The integration path around the
obstacle is probably wrong or too coarse. Check it on the circle first,
where you have something to compare against.

**The closure makes little difference.** At the Reynolds numbers this
exercise can reach, it may genuinely not. Reporting that honestly is
better than tuning until it does.

**An autograd error inside `model.velocity`.** The points were wrapped without
`requires_grad=True`. The samplers return NumPy; the velocity is a derivative
of the stream function, so *every* point set the network sees needs the flag —
boundary sets included.

## How this is meant to be used

The modules are complete and working — you are not asked to rewrite them. Your
work is in the cells marked `# TODO`, which are the residual and the loss. They
are short by design, so your time goes on the parts that carry the ideas rather
than on tensor plumbing.

You *are* expected to read the modules. They contain the reference
implementations your work is judged against.

## Reference texts

Liu, G.R., *PINN with Python: An Introduction* (2025), Ch. 2–6.
Raissi, Perdikaris & Karniadakis, *Physics-informed neural networks*,
J. Comput. Phys. **378** (2019) 686–707.
Pope, *Turbulent Flows*, Cambridge (2000) — for the closure problem itself.

These are the works to read for the theory. **The code, the problem and the
exposition in this exercise set are original to this course** — written from the
2019 paper and the PyTorch documentation, and not derived from any publisher's
code listings. Where a symbol matches a textbook's, it is because both follow
the standard notation of the field.
