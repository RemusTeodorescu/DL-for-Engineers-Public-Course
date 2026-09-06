# Ex_10.2 — Solid oxide cells: SOFC, SOEC and lifetime-aware operation

**Paired with L10.2 · Solid oxide cells · Part 2**

A reversible solid oxide cell in **both modes** — fuel cell and electrolyser —
ending in a lifetime-aware optimisation of a 24-hour operating trajectory. The
same hardware runs both ways, so the sign of the current selects the mode and
nothing else changes.

**There is no PyBaMM here.** No community implementation, no canonical
parameter file. The reference solver is a course implementation and every
parameter carries a source tag; values marked `ESTIMATED` are
order-of-magnitude placeholders, not measurements. Knowing which of your
results are defensible is what this set is built around.

## Goals

By the end you can

1. read a polarisation curve continuously through zero current, and say why
   the three overpotentials subtract in fuel-cell mode and add in electrolysis;
2. locate the thermoneutral point from $q = i(V - V_{tn})$ and predict the sign
   of the heat either side of it;
3. fit a 0-D button-cell model to a noisy polarisation curve and verify it
   against four independent checks rather than against its own residual;
4. write a 1-D convection–diffusion channel residual, and quantify the error a
   0-D model makes as reactant utilisation rises;
5. produce the temperature trade-off — production rate against lifetime — as a
   computed result, and test how far it moves with an `ESTIMATED` parameter;
6. optimise an operating trajectory by differentiating through the cell model,
   check it against brute force, and state which of your numbers you would
   defend.

## The problem

    Nernst potential          thermodynamics of H2 + 1/2 O2 <-> H2O
    three overpotentials      activation, ohmic, concentration
    thermoneutral voltage     V_tn = dH / 2F  (~1.29 V for steam at 800 C)
    heat generation           q = i (V - V_tn)   — zero at the thermoneutral point
    degradation               Arrhenius in T, power law in |i|

Sign convention throughout: **i > 0 is electrolysis (SOEC), i < 0 is fuel cell
(SOFC)**. One expression covers both, because the losses always oppose the
useful direction and the signed current already says which that is.

Temperature is the main lever and it pulls both ways at once: higher
temperature lowers the area-specific resistance **and** raises the degradation
rate. Notebook 00 shows both in two lines of output; notebook 03 turns it into
a curve; notebook 04 makes the optimiser choose.

Every function in `problem.py` dispatches on `torch.is_tensor` and works
unchanged on NumPy arrays and on torch tensors — which is what makes notebook
04's gradient-based optimisation possible without a second implementation of
the physics.

## The notebooks

Run in order; later notebooks load results saved by earlier ones.

```
Ex10.2_00_cell_lab.ipynb        device behaviour in both directions — read-only
Ex10.2_01_button_cell.ipynb     0-D fit and the four verification checks
Ex10.2_02_channel.ipynb         1-D channel and utilisation
Ex10.2_03_degradation.ipynb     the temperature trade-off
Ex10.2_04_optimisation.ipynb    lifetime-aware optimisation — the course finale
Ex10.2_05_report.ipynb          assemble the report for submission
```

Everything the notebooks save goes to `Ex10.2_outputs/`, and notebook 05 reads
that folder.

## Files

Every Part 2 exercise has the same three modules beside it. Only the third
differs between sets.

| | |
|---|---|
| `course_core.py` | shared by the whole course — `set_seed`, `MLP`, `to_tensor`, `check` |
| `pinn_core.py` | the PDE machinery — `grad`, `d2`, samplers, `train_two_stage` |
| `problem.py` | **this** problem — cell model, degradation, control panel, plots, report |

The first two are generated. Edit `tools/pinn/*.py` and run
`python3 tools/pinn/sync_cores.py`; never edit a copy.

**Colab needs the whole folder**, not just a notebook — the imports expect the
three modules beside them. Requires `torch`, `numpy` and `matplotlib`, all
preinstalled on Colab; the control panel also wants `ipywidgets`, which Colab
has and a local install may not (`pip install ipywidgets`).

## Expected runtime

CPU only; no GPU is needed anywhere. A single training run takes one to three
minutes. Notebooks 00 and 05 are seconds. The channel PINN in notebook 02 and
the optimisation in notebook 04 launch several runs each, so budget 30–60
minutes for the set as a whole. L-BFGS prints only at the end of its stage, so
a long silence after the Adam output is normal.

## What to hand in

- the button-cell fit and all four verification checks
- the channel profile and utilisation
- the optimised 24-hour trajectory, against the constant-current baseline
- **which of your results depend on a parameter marked `ESTIMATED`**, stated
  explicitly — this is the question the exercise is built around

## Things that go wrong, and what they mean

**You cannot recover parameters you generated yourself.** Then the fit is not
trustworthy on anything. Notebook 01 asks you to do this first for exactly that
reason.

**The optimiser beats the baseline by an implausible margin.** Check the
end-of-life constraint is actually binding. An unconstrained optimisation will
happily destroy the cell for profit.

**Heat generation does not cross zero where you expect.** The thermoneutral
voltage is where it vanishes; if your crossing is elsewhere, check the sign
convention for current in electrolysis mode.

**A sampler's output will not differentiate.** The samplers return NumPy
arrays, deliberately, so they can be plotted and checked without a device or a
graph. Wrap them: `to_tensor(pts, requires_grad=True)` for anything you
differentiate through, `to_tensor(pts)` otherwise.

**The loss becomes `nan`.** Almost always a residual that divides by zero or
takes a log or a square root of a negative. Print the residual on a handful of
points before training.

## How this is meant to be used

The modules are complete and working — you are not asked to rewrite them. Your
work is in the cells marked `# TODO`, which are the residual, the loss, the fit
and the objective. They are short by design, so your time goes on the parts
that carry the ideas rather than on tensor plumbing.

You *are* expected to read the modules. They contain the reference
implementations your work is judged against.

## Reference texts

Liu, G.R., *PINN with Python: An Introduction* (2025).
Raissi, Perdikaris & Karniadakis, *Physics-informed neural networks*,
J. Comput. Phys. **378** (2019) 686–707.

These are the works to read for the theory. **The code, the problem and the
exposition in this exercise set are original to this course** — written from the
2019 paper and the PyTorch documentation, and not derived from any publisher's
code listings. Where a symbol matches a textbook's, it is because both follow
the standard notation of the field.

The cell model itself follows the IEA SOFC Benchmark Test 1 (Achenbach,
1994/96, public domain) for the single-cell hydrogen case; published
button-cell polarisation fits give activation energies of order 100 kJ/mol
(fuel electrode) and 66 kJ/mol (oxygen electrode); SOEC thermoneutral operation
at 1.29 V is standard practice.

## Before this is assigned

Migrated to `course_core` / `pinn_core` in an environment where PyTorch could
not be installed. Every equation, constant and operating point is carried over
unchanged from the previous version of this set; only the library calls
changed. **Nothing has been executed.** Run notebooks 00 to 05 end to end
before this goes to students.
