# Ex_08.2 — Dynamic Heat: a plate, a hole, and time

**Paired with L8.2 · Dynamic Heat · Part 2**

Transient conduction on the Ex_08.1 geometry: soft versus hard initial
conditions, the plate with a hole in time, and recovering an unknown
diffusivity from a cooling curve.

The inverse problem at the end is the first time in the course that the network
is asked to find a **physical constant** rather than a field. It returns in
L11.2 and again in Ex_12.1.

## Goals

By the end you can

1. write a first-order-in-time residual on a space–time slab and say which
   column of the gradient is the time derivative;
2. assemble a three-term loss and report **which** term the optimiser is
   actually reducing, rather than only the total;
3. enforce an initial condition and a boundary condition exactly, by
   construction, and state precisely what that costs the network;
4. put the same equation on a domain with a hole, using a level set as both the
   rejection test and the hard-boundary multiplier;
5. recover an unknown diffusivity from noisy data, and decide whether the data
   supports the number you got — **identifiability**;
6. choose a time window from the diffusion time constant, and report relative
   *and* absolute error because at late times one of them is meaningless.

## The problem

**The square plate.** The unit square, zero on all four edges, released from a
single Helmholtz mode:

$$T_t = c\,(T_{xx}+T_{yy}), \qquad T(x,y,0)=\sin(\pi x)\sin(\pi y)$$

with the exact solution $T=e^{-2\pi^2 c t}\sin(\pi x)\sin(\pi y)$ — the fundamental mode, by separation of variables.
The eigenvalue of the lowest mode, $2\pi^2$, *is* the decay rate: the shape
never changes, only the amplitude. The time constant is
$\tau = L^2/(\pi^2 c) = 0.1013$ and the transient is essentially over by
$4\tau$. By $t = 1$ the amplitude is about $3\times10^{-9}$, which is why every
error in this set is reported twice.

**The plate with a hole.** The Ex_08.1 geometry, now transient: an elliptical
hole at the centre, held at zero, with a uniform source switched on at $t=0$.
The hole is a level set, so no multiplier network has to be trained — and there
is no exact solution to score against, which from here on is the normal
situation.

**The unknown diffusivity.** Five synthetic sensors, twelve readings each
through the early transient, with noise. $\alpha$ joins the trainable
parameters as $\log\alpha$ and the loss gains a data term. Then the notebook
asks you to move the sensors into the settled part of the curve and watch a
confident, wrong answer come back.

## The notebooks

```
Ex08.2_00_transient_check.ipynb    timescales, space-time sampling, the geometry
Ex08.2_01_soft_ic.ipynb            three soft terms - and where the error goes
Ex08.2_02_hard_ic.ipynb            (1-t) f_IC + t D N, and what N pays for it
Ex08.2_03_plate_transient.ipynb    the plate with a hole, in time
Ex08.2_04_inverse_alpha.ipynb      recovering the diffusivity
Ex08.2_05_compare_and_report.ipynb the report
```

Run them **in order** — notebooks 02 and 05 load results the earlier ones
saved, into a folder called `Ex08.2_outputs`. Notebook 00 is read-and-run;
notebooks 01 to 04 have `# TODO` cells that are yours to complete.

Your work is in those cells: the residual, the loss and the trial solution.
They are short by design, so your time goes on the parts that carry the ideas
rather than on tensor plumbing. You *are* expected to read `problem.py` — it
contains the reference implementations your work is judged against.

## Files

| | |
|---|---|
| `course_core.py` | shared by the whole course |
| `pinn_core.py` | the PDE machinery |
| `problem.py` | the geometry, the reference solution, the timescales and the plots |

The first two are generated: edit `tools/pinn/*.py` and run
`python3 tools/pinn/sync_cores.py`. **Colab needs the whole folder** — upload
every file to one directory, keeping them together, and run notebook 00 first.
Requires `torch`, `numpy` and `matplotlib`, all preinstalled on Colab. **No GPU
is needed.**

## Expected runtime

CPU only. Every training run is a space–time slab, so budget three to five
minutes each for notebooks 01, 02 and 03, and rather less for 04. An hour for
the full set is a safe estimate, carried over from the previous version of this
exercise. If a cell seems stuck it probably is not: each L-BFGS step runs up to
twenty inner iterations and only every fiftieth step is printed, so a long
silence after the Adam output is normal.

## Things that go wrong, and what they mean

**The recovered diffusivity is confident and wrong.** Check how much of your
cooling curve actually contains transient behaviour. Once the plate has
equilibrated the data says nothing about $\alpha$, and the optimiser will still
return a number. This is identifiability, and it recurs in every later inverse
problem in the course.

**Early-time error is much worse than late-time error.** Expected with a soft
IC. That is the comparison the exercise wants.

**The solution is smooth everywhere and matches nothing.** Check the timescale
printed by notebook 00 against the interval you are solving over. Solving for
far longer than the diffusion time gives you a steady state, correctly.

**`grad` returns `None`.** The points were not built with
`to_tensor(..., requires_grad=True)`. The samplers return NumPy arrays now; the
conversion is yours, at the point of use.

**The loss becomes `nan`.** Almost always a residual that divides by zero or
takes a log or a square root of a negative. Print the residual on a handful of
points before training.

## What to hand in

- soft versus hard IC, with the error at early time reported separately
- the transient plate solution, compared against your Ex_08.1 steady result
- your recovered diffusivity, with a statement of how much the data supports it
- what would happen to the inverse fit if the cooling curve were shorter

## Reference texts

Liu, G.R., *PINN with Python: An Introduction* (2025), Ch. 2–6.
Raissi, Perdikaris & Karniadakis, *Physics-informed neural networks*,
J. Comput. Phys. **378** (2019) 686–707.

These are the works to read for the theory. **The code, the problems and the
exposition in this exercise set are original to this course** — written from the
2019 paper and the PyTorch documentation, and not derived from any publisher's
code listings. Where a symbol matches a textbook's, it is because both follow
the standard notation of the field.

## Before this is assigned

The NumPy half is verified: the samplers reject every point inside the hole,
the arc-length hole sampler lands on the level set to 1e-15, and the timescale
table printed by `describe_problem` matches the closed form. **Nothing
requiring torch has been executed.** Run notebook 00 end to end before this
goes to students.
