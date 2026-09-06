# Ex_08.1 — Stationary Heat: a plate with a cooling hole

**Paired with L8.1 · Stationary Heat Transfer · Part 2**

Steady conduction on a plate with a cooling hole: real geometry, a
hard-enforced curved boundary, and a flux balance as the physical check.

The flux balance is the point. It is a check that must come out right for
physical reasons, and it is worth more than an error norm that looks small.

## Goals

By the end you can

1. verify a PDE formulation on a manufactured solution before trusting it on
   real geometry, and say why that ordering saves an afternoon;
2. hard-enforce a fixed temperature on a **curved** boundary using its
   level-set function, with no multiplier network to pre-train;
3. impose a zero-flux condition where the outward normal changes at every
   point, and get its sign right;
4. sample a curved boundary by **arc length** rather than by angle, and measure
   the difference the choice makes;
5. weight a soft flux term against a PDE residual, and diagnose from the
   per-term loss history whether it was being ignored;
6. check a solution by **conservation** — heat out through the hole against
   heat generated in the plate — rather than by an error norm alone.

## The problem

A square plate with a cooling hole through it, generating heat internally, in
steady state:

```
∇²T + Q/k = 0        in the material
T = 0                on the hole wall        (hard-enforced)
∂T/∂n = 0            on the four outer edges (soft, weighted)
```

The hole is an ellipse at the centre of the unit square, `a = 0.18`,
`b = 0.11` — 6.22% of the plate's area, with 0.924 of wall through which
everything generated in the remaining 0.938 of area has to leave.

The equation is the elliptic problem of Ex_07.1 again. What is new is
**geometry**: the domain is not a rectangle, so `pinn_core`'s samplers do not
describe it; the boundary carrying the interesting condition is curved; and the
outward normal there is a function of position rather than one of four constant
vectors. `problem.py` supplies the four things that fixes — a level set, a
hard-enforcement multiplier, geometry-aware samplers, and the flux balance.

Notebook 01 comes first for a reason. It solves the same operator on the plain
unit square against a manufactured solution,

```
T(x, y) = sin(πx) sin(πy)   ⇒   Q/k = 2π² sin(πx) sin(πy)
```

where the answer is known and the boundary condition can be lifted exactly by
`x(1-x)y(1-y)`. An implementation error found there costs minutes; the same
error found on the plate costs an afternoon. Do not skip it.

## The notebooks

Run in order; later notebooks load results saved by earlier ones.

```
Ex08.1_00_geometry_check.ipynb      tools, geometry, samplers, multiplier, normals — read-only
Ex08.1_01_manufactured.ipynb        verification on a known solution — has TODOs
Ex08.1_02_plate_with_hole.ipynb     the real geometry — has TODOs
Ex08.1_03_weights_and_flux.ipynb    flux weighting and the energy balance — has TODOs
Ex08.1_04_compare_and_report.ipynb  comparison and the report
```

Everything the notebooks write goes to `Ex08.1_outputs/`.

## Files

Every Part 2 exercise has the same three modules beside it. Only the third
differs between sets.

| | |
|---|---|
| `course_core.py` | shared by the whole course — `set_seed`, `MLP`, `to_tensor`, `check` |
| `pinn_core.py` | the PDE machinery — `grad`, `d2`, samplers, `train_two_stage` |
| `problem.py` | **this** problem — the hole, the level-set multiplier, the normals, the flux balance |

The first two are generated. Edit `tools/pinn/*.py` and run
`python3 tools/pinn/sync_cores.py`; never edit a copy.

**Colab needs the whole folder**, not just a notebook — the imports expect the
three modules beside them. Uploaded files vanish when the runtime restarts; if
you get a `FileNotFoundError` partway through a session, re-upload, and the
first cell of every notebook will prompt you.

Requires `torch`, `numpy` and `matplotlib`, all preinstalled on Colab. **No GPU
is needed.**

## A note on the samplers

Everything that samples points — in `pinn_core` and in `problem.py` alike —
returns a **NumPy array**, not a tensor. Points can then be plotted, saved and
checked without a device or a graph. Wrap them at the point of use:

```python
xy_f = to_tensor(pb.sample_plate_with_hole(1500), requires_grad=True)
xy_o = to_tensor(pb.sample_outer_edges(30), requires_grad=True)   # flux: also grad
xy_h = to_tensor(pb.sample_ellipse_boundary(200), requires_grad=True)
```

`requires_grad=True` wherever the network is differentiated at those points,
which in this exercise means all three sets. Omitting it on `xy_o` is the
easiest mistake in notebook 02: the zero-flux condition is a first derivative,
so those points are differentiated too.

## Expected runtime

CPU only, no GPU anywhere. The retired core's run guide quoted a minute or two
per notebook, with a few minutes for the three-weight sweep in notebook 03.
Treat that as the order of magnitude rather than a measurement: the two-stage
schedule here spends more of its budget in L-BFGS than that guide's did, and
**nothing in this folder has been re-timed.** Time your first run.

If a cell seems stuck, it probably is not — L-BFGS prints sparsely, so a long
silence after the Adam output is normal.

## What to hand in

- the manufactured-solution verification, before anything else
- the plate solution, with the flux balance stated as a number
- the weight study, and what it did to the flux balance
- whether your solution conserves energy, and what you concluded if not

## Things that go wrong, and what they mean

**You skipped the manufactured solution.** Do not. It is the only place in this
exercise where you know the answer, and an implementation error found here
costs minutes rather than an afternoon.

**`grad` returned `None`, or the loss will not move.** A sampler handed you a
NumPy array and you passed it on without `to_tensor(..., requires_grad=True)`.
This is the single most common first error.

**The flux balance is off by a few per cent.** Look at the sampling near the
hole. A curved boundary needs points on it, spaced by arc length, and a uniform
sample of the rectangle gives you almost none.

**The level-set multiplier makes training unstable.** It goes to zero on the
boundary, so gradients there are small; that is the price of hard enforcement
and is usually worth paying. Check the multiplier is not *negative* anywhere
inside the domain — notebook 00 prints its minimum.

**The temperature field looks plausible but inverted.** The outward normal of
the material points *into* the hole. Notebook 00 draws the arrows; look at them.

## How this is meant to be used

The three modules are complete and working — you are not asked to rewrite them.
Your work is in the cells marked `# TODO`, which are the residual, the loss and
the trial solution. They are short by design, so your time goes on the parts
that carry the ideas rather than on tensor plumbing.

You *are* expected to read the modules. They contain the reference
implementations your work is judged against.

## Reference texts

Liu, G.R., *PINN with Python: An Introduction* (2025), Ch. 2–6.
Raissi, Perdikaris & Karniadakis, *Physics-informed neural networks*,
J. Comput. Phys. **378** (2019) 686–707.

These are the works to read for the theory. **The code, the problem and the
exposition in this exercise set are original to this course** — written from the
2019 paper and the PyTorch documentation, and not derived from any publisher's
code listings. Where a symbol matches a textbook's, it is because both follow
the standard notation of the field.

## Before this is assigned

Migrated to the shared `course_core` / `pinn_core` library in an environment
where PyTorch could not be installed. The NumPy half is verified: the level set
is machine-zero on the hole and positive throughout the material, the rejection
sampler keeps every collocation point clear of the inflated ellipse, the
arc-length spacing beats the angular spacing by 265×, and the Ramanujan
perimeter agrees with a two-million-segment numerical integration to 4e-7
relative. **Nothing that requires torch has been executed.** Run notebooks 00
to 03 end to end before this goes to students, and re-time them.
