# Ex_09.2 — Hydrogen Fuel Channel for SOFC/SOEC

**Paired with L9.2 · Gas Flow · Part 2**

The fuel channel of a solid oxide fuel cell, from the flow side, in **one
notebook** (course policies C11 and C13). Humidified hydrogen flows along a
1 mm channel over a 100 × 100 mm electrode at 800 °C; along the electrode wall
hydrogen is taken up and steam returned at a rate set by the current. How does
the gas change along and across the channel, how much of the fuel is used, and
how fast must it be fed so that the electrode at the outlet still sees enough
hydrogen?

## The problem

```
u(z) y_x = D y_zz                        the steam fraction y, carried along and diffused across
-D y_z = N/c at the electrode            hydrogen taken, steam returned: i / 2F
y_z = 0 at the interconnect              impermeable
y = 3 % at the inlet                     97 % hydrogen fed
```

Three facts reduce the gas to this: the Mach number is about 1e-03, so the
driving pressure does not change the density; the exchange is equimolar, so
the molar concentration and the velocity do not change (the mass density
does, 2.5 times along the channel at 1 m/s, which is why the balance is in
moles); and Re = 1.3, where the entrance length settles at about 0.6
hydraulic diameters, 1 % of the channel, so the velocity is Poiseuille's
profile from the inlet on. **Diffusion along the channel is
dropped** (Péclet 125 along it), which makes the equation parabolic in $x$:
it marches from the inlet and needs no outlet condition. Beyond the first
millimetre its solution is **exact**: linear along the channel and a quartic
across it, with a Sherwood number of 2 · 35/13 = 5.385.

The cell's current (0.5 A/cm²), the speeds (0.3 to 1.5 m/s) and the gas
properties are typical values, flagged where they appear.

## The notebook

```
Ex09.2_hydrogen_channel.ipynb         the exercise: two TODO cells, the answer in the comment above each
Ex09.2_hydrogen_channel_light.ipynb   the same notebook with every cell written out
```

| section | what the student does |
|---|---|
| 1 – 3 | the problem, its data and its physics: the gas, the flow, the species equation, the exact solution |
| 4 | finite differences marched along the channel (Crank–Nicolson), refined to an agreed 0.1 percentage point |
| 5 | one PINN for every inlet speed (the speed as an input; TODO 1 the trial function, TODO 2 the residual), and the comparison |
| 6 | the inlet speed for 25 % hydrogen at the electrode at the outlet, by Newton's method through the network |
| 7 | what the notebook says |
| 8 | the report and its PDF |
| 9 | mini project proposal |

### What it measures

The electrode sees 0.106 percentage points more steam than the mean at every
position and speed: the channel mixes across within a millimetre. Finite
differences (FDM), marched from the inlet with Crank–Nicolson and two
implicit steps first (without them the wall condition is held only on
average and the error on the electrode node oscillates from step to step),
converge at second order and meet 0.1 percentage point at 401 × 41 in a few
milliseconds; the PINN, with its flow-weighted mean held at the hand
calculation by a loss term, is within 2 to 6e-06 after some minutes of
training on a CPU, 25 to 120 times more exact. Both find the Sherwood
number to 0.3 % once the wall and the mean are read from the same computed
column (an earlier version mixed a computed wall with the exact mean and
reported 8 to 31 % errors that were the bookkeeping's). Online, no grid
tested is as exact as the PINN; against the finest the PINN returns the field
a few times faster and one number hundreds of times faster, inference only;
its larger gain is the speed as an input. Newton through autograd finds
0.3174 m/s for 25 % hydrogen at the electrode, the exact value.

## Files

| | |
|---|---|
| `course_core.py` | shared by the whole course |
| `pinn_core.py` | the Part 2 helpers |
| `channel.py` | **this** problem — the gas, the channel, the exact solution, the march |
| `MiniProject_Turbulence.md` | an outline for an L13 project on turbulence closure, kept from the earlier set |

The first two are generated: edit `tools/pinn/*.py` and run
`python3 tools/pinn/sync_cores.py`. **The notebook is generated too**, from
`tools/exercises/ex092/build_ex092.py`.

## Expected runtime

About six minutes on a CPU, most of it the PINN's training.

## Reference texts

Shah and London, *Laminar Flow Forced Convection in Ducts* (1978), for the
Sherwood number of a channel with one wall exchanging; Liu, *PINN with Python:
An Introduction* (2025).

These are the works to read for the theory. **The code, the problem and the
exposition in this exercise set are original to this course.**

## Before this is assigned

Executed end to end on a local CPU on 3 October 2026; the exercise version
stops at its TODO cells. Still to do: a run from a fresh Colab runtime and a
review by someone other than the author.

It replaced, on 5 October 2026, the notebook on the mean flow past a tube in a
duct with one eddy viscosity.

## Mini project proposal

The set ends with two mini projects (section 9 of the notebook). Each student chooses one
mini project from the Part 2 sets and solves it individually in one month. The
ground truth is given, built by `tools/miniprojects/ex092_truth.py` under policy C10
(`COURSE_POLICIES.md`), with a worked example of each.

| | the problem | the deep learning | the ground truth given | required |
|---|---|---|---|---|
| **MP9.2A · The SOEC steam channel** | the same channel as an electrolyser: 90 % steam fed, the wall flux reversed; the feed speed for 70 % conversion | the notebook's parametric PINN retrained with the sign and feed changed; the conversion by a gradient through it | the exact solution with the sign reversed; worked example: 0.362 m/s for 70 %, 73.13 % hydrogen at the electrode | speed within 1 %, electrode fraction within 0.1 point, one training |
| **MP9.2B · The air channel** | the SOFC's air channel: oxygen taken and not returned, so the molar flow falls; stoichiometry 5, D four times smaller | a PINN with the Stefan term in the equation and the wall condition, against the exact solution without it | the balance with the molar flow falling and the fully developed excess; worked example: 17.31 % oxygen at the electrode at the outlet, 0.74 points above the constant-flow balance, Stefan velocity 0.6 % of D/h | electrode fraction within 0.1 point, the Stefan question answered with a number, the speed for 15 % by a gradient |