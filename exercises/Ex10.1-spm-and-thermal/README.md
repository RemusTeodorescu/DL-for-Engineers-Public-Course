# Ex_10.1 — Battery models: SPM, SPMe and thermal coupling

**Paired with L10.1 · Battery models · Part 2**

SPM → SPMe → thermal coupling on the LG M50 21700, with **PyBaMM as the
reference** and a control panel for C-rate, ambient temperature, initial SOC
and cooling.

Having a community reference implementation to compare against is a luxury.
Ex_10.2 removes it, and the difference in how confident you can be is the point
of the pair.

## Goals

By the end you can

1. write a **radial** diffusion residual, including the $2/r$ curvature term,
   and say what you did about the singularity at the centre;
2. impose a **flux** condition on both ends of a domain where no value is
   prescribed anywhere, and explain why that is still a well-posed problem;
3. score a trained field against a series solution and report the **surface**
   error separately from the bulk, because the surface is what the reaction
   sees;
4. locate the operating point at which a reduced model — the SPM — measurably
   stops matching a fuller one, and name the physics it dropped;
5. decompose a coupled heat source into its irreversible, reversible and ohmic
   parts, and say which changes sign with the current;
6. judge from the Biot number whether a lumped thermal model is honest, and say
   what you would need to do if it is not.

## The problems

**One particle — parabolic, radial, singular at the centre.** The
single-particle model reduces an electrode to one representative sphere:
$c_t = c_{rr} + (2/r)c_r$, with zero flux at the centre, unit flux at the
surface and $c=0$ initially. Two things separate it from everything in Ex_07:
the Laplacian carries a curvature term that blows up at $r=0$, and **both**
boundary conditions are on the flux — nothing fixes the concentration anywhere.
`analytic_sphere` in `problem.py` is the classical series solution (Crank,
*Mathematics of Diffusion*), so the error is measurable without PyBaMM and
without training anything.

**The whole cell — where the SPM stops being honest.** The SPM assumes a
uniform reaction rate and a uniform electrolyte. Both fail as C-rate rises, and
the SPMe differs from the SPM by exactly the electrolyte physics that was
dropped. PyBaMM supplies both, so the failure is measured in millivolts rather
than argued about.

**The cell heats up — Gu & Wang's three sources.** Irreversible, reversible and
ohmic heating, computed separately, with the entropic term the 2000 paper had
to neglect for want of data and ORegan2022 measured.

## The notebooks

```
Ex10.1_00_reference.ipynb        analytic sphere + PyBaMM references — read-only
Ex10.1_01_spm.ipynb              the single particle — has TODOs
Ex10.1_02_spme.ipynb             where the SPM fails — has TODOs
Ex10.1_03_thermal.ipynb          Gu & Wang heat source and the r–z field — has TODOs
Ex10.1_04_control_panel.ipynb    interactive parameter study
Ex10.1_05_report.ipynb           assemble the report for submission
```

Run them in order — later ones load results the earlier ones saved into
`Ex10.1_outputs/`.

## Files

| | |
|---|---|
| `course_core.py` | shared by the whole course |
| `pinn_core.py` | the PDE machinery |
| `problem.py` | the cell — parameters, the analytic sphere, OCPs, heat sources, the PyBaMM reference, the control panel and the report |

The first two are generated: edit `tools/pinn/*.py` and run
`python3 tools/pinn/sync_cores.py`. **Colab needs the whole folder.**

The samplers in `pinn_core.py` know about rectangles. The (r, t) slab is one,
so `pb.particle_points` is a thin wrapper — but its four edges are the centre
of a sphere, a reacting surface and an instant in time, each carrying a
different kind of condition. Those samplers, and the series solution for the
sphere, live in `problem.py` because they are geometry, not machinery.

## What to hand in

- the SPM solution against the analytic sphere solution
- the operating point where the SPM measurably fails, and the SPMe result there
- the thermal r–z field, with the heat source stated
- the control-panel study, and which parameter dominated

## Expected runtime

CPU only; no GPU is needed. A single particle run is one to three minutes. The
control panel launches one run per click, so budget 30–60 minutes for a
worthwhile study. Notebooks 00 (section 3) and 02 additionally need
**`pybamm`** — `pip install -q pybamm`; notebook 01 works without it, using the
analytic sphere solution.

If a cell seems stuck, it probably is not: L-BFGS reports rarely, so a long
silence after the Adam output is normal.

## Things that go wrong, and what they mean

**PyBaMM will not install.** Notebook 01 does not need it — use the analytic
sphere solution and say in your report which reference you used.

**The loss becomes `nan` on the first step.** Almost always the $2/r$ term at a
collocation point sitting on, or extremely close to, the centre. Clamp it or
sample away from it.

**The SPM matches PyBaMM everywhere you look.** Then you have not yet found the
regime where it fails. Push the C-rate: the SPM neglects electrolyte transport,
so the discrepancy appears where that matters.

**The thermal solution has a hot spot in the wrong place.** Check the sign of
each heat-source term separately. Reversible heating changes sign with current
direction and is easy to get backwards.

## How this is meant to be used

The modules are complete and working — you are not asked to rewrite them. Your
work is in the cells marked `# TODO`, which are the residual, the loss and the
trial solution. They are short by design, so your time goes on the parts that
carry the ideas rather than on tensor plumbing.

You *are* expected to read `problem.py`. It contains the reference
implementations your work is judged against.

## Reference texts

Liu, G.R., *PINN with Python: An Introduction* (2025).
Raissi, Perdikaris & Karniadakis, *Physics-informed neural networks*,
J. Comput. Phys. **378** (2019) 686–707.

These are the works to read for the theory. **The code, the problems and the
exposition in this exercise set are original to this course** — written from the
2019 paper and the PyTorch documentation, and not derived from any publisher's
code listings. Where a symbol matches a textbook's, it is because both follow
the standard notation of the field.

The physics follows the Doyle-Fuller-Newman framework; the thermal coupling and
the heat-source decomposition follow Gu & Wang (2000). Parameters and reference
solutions come from PyBaMM (LG M50 21700: Chen2020 isothermal, ORegan2022
thermal).

## Before this is assigned

The NumPy half of `problem.py` is verified: `analytic_sphere` satisfies its PDE
to between 5e-8 and 1.5e-6 under central differences — finite-difference error,
not a flaw in the solution — and its surface flux is 0.99991 against a nominal
1 by a one-sided difference. `CellParams`, the three heat-source terms, the
lumped thermal response, the Arrhenius factor and the particle samplers all
run. **Nothing requiring torch has been executed** — this set was migrated onto
`course_core`/`pinn_core` without a torch runtime available. Run notebooks 00
and 01 end to end before this goes to students.
