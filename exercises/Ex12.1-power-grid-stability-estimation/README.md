# Ex_12.1 — Power Grid Stability Estimation

**Paired with L12.1 · Power Grid Stability Estimation · Part 2**

Recover the state of a small transmission network from too few measurements —
first as an algebraic problem, then as a dynamic one — and finish by
identifying the inertia of a machine from a disturbance.

This is the first exercise in the course where **most of the system cannot be
measured at all**, and where the physics is what fills the gap. Everything else
follows from that.

It is also the first set in Part 2 with no PDE and no rectangle in it. The
collocation points are buses and instants, not points in a domain, so the
shared samplers in `pinn_core` are barely used here. `grad`, `to_tensor`,
`MLP` and `train_two_stage` are used exactly as everywhere else.

## Goals

By the end you can

1. state what observability means as a rank condition on the measurement
   Jacobian, and say why it is a property of *where* the meters are rather than
   how many there are;
2. run weighted least squares as a baseline, read its normalised residuals, and
   identify a bad measurement from them;
3. add the power flow equations to the objective as a residual, and sweep the
   weight λ between measurements and physics — reading the trade-off off a
   curve, at the **unmetered** buses where it is visible;
4. represent a trajectory δ(t), ω(t) with a network, impose the swing equation
   by automatic differentiation, and build the initial condition into the trial
   solution instead of penalising it;
5. identify a physical parameter — machine inertia — as a trainable variable,
   and say from the fit alone whether the window you chose supports the answer;
6. report an estimate at an unmetered bus as what it is: an inference from the
   model, not a reading.

## The system you are estimating

A six-bus network. Bus 0 is the connection to the Nordic synchronous area
through Sweden and behaves as an infinite bus; bus 1 is local generation; buses
2–4 are load; bus 5 is the HVDC link to Germany, which enters as a fixed
injection rather than as a synchronous branch — a DC link transfers power but
not synchronism.

The algebraic half asks for |V| and θ at every bus given readings at a few of
them:

```
P_i = V_i Σ_k V_k ( G_ik cos θ_ik + B_ik sin θ_ik )
Q_i = V_i Σ_k V_k ( G_ik sin θ_ik − B_ik cos θ_ik )
```

The dynamic half asks for δ(t) and ω(t) at each machine given a frequency trace
at one of them:

```
dδ_i/dt = ω_i − ω_s
dω_i/dt = (ω_s / 2H_i) ( P_m,i − P_e,i − D_i (ω_i − ω_s) / ω_s )
```

**The network is DK2-representative, not DK2.** Its structure follows eastern
Denmark in the ways that matter for this exercise, but the line impedances are
plausible textbook values, *not* measured ones. No TSO publishes a nodal model
with impedances — that data is withheld for security. Real Danish injections
and real Nordic frequency are available and are used; the topology is not.

Say this in your report. Results that depend only on structure — which buses
are observable, how error grows with distance from a meter — are on firmer
ground than results that depend on a specific reactance.

The same six-bus case appears in L5.1 and again in L12.1. It does not change
between them.

## The notebooks

Run in order; later notebooks load results saved by earlier ones.

```
Ex12.1_00_system_check.ipynb        the network, its power flow, tools checked   read-only
Ex12.1_01_wls_baseline.ipynb        weighted least squares — the utility baseline  has TODOs
Ex12.1_02_algebraic_pinn.ipynb      the power flow equations as a residual         has TODOs
Ex12.1_03_dynamic_pinn.ipynb        the swing equation, a fault, a trajectory      has TODOs
Ex12.1_04_inertia.ipynb             the inverse problem: recover H and D           has TODOs
Ex12.1_05_compare_and_report.ipynb  tables, curves and the report questions        assemble
```

**The notebooks now sit beside the modules, not in a `notebooks/` subfolder.**
This set was the only one in the course arranged the other way; it no longer
is. If you have an older copy, move the notebooks up one level or the imports
will not find `problem.py`.

Everything each notebook writes goes to `Ex12.1_outputs/` — the saved states
that cross between notebooks, and the report skeleton notebook 05 generates.

## Files

Every Part 2 exercise has the same three modules beside it. Only the third
differs between sets.

| | |
|---|---|
| `course_core.py` | shared by the whole course — `set_seed`, `MLP`, `to_tensor`, `check` |
| `pinn_core.py` | the PDE machinery — `grad`, `d2`, samplers, `train_two_stage` |
| `problem.py` | **this** problem — network, power flow, WLS, dynamics, Danish data, estimators, plots |

The first two are generated. Edit `tools/pinn/*.py` and run
`python3 tools/pinn/sync_cores.py`; never edit a copy.

`problem.py` is large, because this set carries three things rather than one: a
six-bus network model with a Newton-Raphson power flow and a classical
estimator, a two-machine dynamic model with an RK4 reference integration, and a
loader for real Energinet and Fingrid data with an offline fallback. You are
not asked to modify it. You **are** asked to read it — it holds the reference
implementations your estimator is judged against, and it is written to be read.

One trap worth knowing about. `problem.py` defines an `error_table` that is not
`course_core`'s. The shared one formats a Markdown table; this one is the
per-bus estimation report, split metered from unmetered. Reach it as
`pb.error_table`, always.

## Getting started on Colab

1. Upload **every file in this folder** to one Colab directory, keeping them
   together. The notebooks import the modules by name from the working
   directory.
2. Run `Ex12.1_00_system_check.ipynb` first. It checks your environment, checks
   that automatic differentiation works, solves the reference power flow, and
   tells you if something is missing.
3. Work through the numbered notebooks in order, **in the same runtime**.
   Every Colab tab is a separate machine with its own filesystem, so a file
   saved by notebook 00 in one tab does not exist in another. The reference
   state repairs itself — `pb.load("00_reference")` rebuilds it if it is
   missing — but the training results of 01–04 are your runs and do not. To
   move them between runtimes, download the `Ex12.1_outputs/` folder from the
   Files panel and upload it in the new tab.

## Expected runtime

CPU only, no GPU anywhere. Notebooks 00, 01 and 05 are seconds to a minute —
they are NumPy. Notebooks 02, 03 and 04 each train, and each training run is a
few thousand Adam steps followed by L-BFGS; budget a few minutes per run.
Notebook 02's λ sweep is thirteen runs and is the longest cell in the set, so
allow ten to twenty minutes for it.

If a single cell runs much longer than about three minutes, stop it and read
the error rather than waiting.

Note `lbfgs_steps` in `train_two_stage` counts **outer** steps of up to twenty
inner L-BFGS iterations each, so `lbfgs_steps=15` is roughly three hundred
L-BFGS iterations. The values in the notebooks are chosen against that
convention.

## The one habit this exercise is trying to build

> An estimate at an **unmetered** bus is an inference from the model.
> An estimate at a **metered** bus is supported by a reading.
> They are not the same thing, and every table you produce must separate them.

The mean error across all buses will look reassuring and will hide the bus that
matters. Report the worst bus alongside the average, every time.

## What to hand in

Notebook 05 generates a report skeleton. It asks for:

- your WLS baseline numbers, with the measurement set stated
- the λ sweep from notebook 02, as a curve, with the value you chose and why
- estimation error split by **metered** and **unmetered** buses, worst and mean
- your identified inertia, with an honest statement of how much the disturbance
  window supports it
- which of your numbers you would show a control-room operator, and which you
  would not

The last question carries the marks. As in L9.2, credit is for saying precisely
what would have to be true first.

## Things that go wrong, and what they mean

**The power flow does not converge after an outage.** Some contingencies
disconnect a bus — removing the branch to the HVDC link islands bus 5, and
there is then no solution to find. This is a real result, not a bug. Report it
as an islanding case rather than tuning until it converges.

**WLS returns nonsense with the thin measurement set.** It should. That set is
deliberately unobservable, and a classical estimator has no way to determine
part of the state from the data alone. That failure is the motivation for
everything in notebook 02.

**The λ sweep looks flat.** Check that you are plotting error at *unmetered*
buses. At metered buses the two terms mostly agree, so the trade-off is
invisible — which is itself worth understanding.

**The identified inertia is confident and wrong.** Look at your window. Inertia
enters the swing equation through acceleration, so a quiet interval contains
almost no information about H, and the optimiser will still return a number —
one determined by your initialisation rather than by the grid. This is the
identifiability lesson from L11.2, in a third domain.

**Everything trains but the answer is smoothly wrong everywhere.** Check the
topology you gave the residual matches the one that generated the
measurements. A wrong admittance matrix produces a beautifully consistent
estimate of a network that does not exist, with small residuals throughout. It
is the most dangerous failure in the exercise because every diagnostic looks
healthy.

**`grad` returns `None`.** You built the collocation times without
`requires_grad=True`. The samplers and `np.linspace` both return NumPy; it is
`to_tensor(..., requires_grad=True)` that makes a coordinate differentiable.

## Reference texts

Liu, G.R., *PINN with Python: An Introduction* (2025).
Raissi, Perdikaris & Karniadakis, *Physics-informed neural networks*,
J. Comput. Phys. **378** (2019) 686–707.

These are the works to read for the theory. **The code, the problem and the
exposition in this exercise set are original to this course** — written from the
2019 paper and the PyTorch documentation, and not derived from any publisher's
code listings. Where a symbol matches a textbook's, it is because both follow
the standard notation of the field.

For the power-systems half, the three works L12.1 sends you to:

Schweppe & Wildes, *Power system static-state estimation*, IEEE Trans. Power
Apparatus and Systems (1970) — the paper that started the field, short and
readable.
Abur & Expósito, *Power System State Estimation: Theory and Implementation*
(2004) — weighted least squares, observability and bad-data detection.
Kundur, *Power System Stability and Control* (1994) — the swing equation,
machine models and transient stability; chapter 11 for this exercise.

## Status

The NumPy half is verified in this environment and reproduces every "expected
output" block in the notebooks: 18 non-zero entries in Y, power flow converged
in 5 iterations with voltages 0.9706–1.0000 and a mismatch at 3e-15,
observability rank 10/10 and 4/10, WLS objective J = 12.04 with worst errors
6.7e-4 / 6.4e-4 and 2.2e-2 / 3.0e-2, an implied base of 742 MVA, equilibrium
angles [0, 1.99]°, a critical clearing time of 0.330 s, and RoCoF values of
−0.213, −0.186 and −0.138 Hz/s over the three windows.

**Nothing that requires torch has been executed** — torch could not be
installed in the environment where this migration was done. That covers the
three estimators in `problem.py` (`algebraic_pinn`, `dynamic_pinn`,
`identify_inertia`) and everything in notebooks 02, 03 and 04 that calls them.
Treat their first run as a debugging session and report anything that breaks.
