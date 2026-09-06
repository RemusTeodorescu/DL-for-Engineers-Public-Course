# Ex05 — CNN and GNN

**Paired with lecture block L5 · Part 1**

One idea, three domains: **a learned local rule, applied everywhere.** A
convolution applies it on a grid, message passing applies it on a graph, a
recurrent cell applies it along time. Notebook 03 does node regression on a
six-bus power network — deliberately the same object Ex_12.1 uses, so the
continuity into Part 2 is felt rather than asserted.

## Goals

By the end of this exercise set you can

1. implement a 2-D convolution by hand and check it against fixed kernels;
2. justify the parameter argument for convolution over a dense layer with
   counts you computed yourself;
3. build an adjacency matrix, perform message passing by hand, and write a
   graph convolution layer in one matrix expression;
4. demonstrate permutation equivariance to machine precision, and show a dense
   baseline failing the same test;
5. write a recurrent cell, an LSTM cell and single-head self-attention, and
   compare all four models honestly on the same held-out data — including the
   one that wins for uninteresting reasons.

## State

**Written.** Six notebooks and one module. No solutions notebook; the `# TODO:`
cells are the exercise.

## Notebooks

Run in order; notebook 05 loads results saved by 01, 03 and 04.

```
Ex05_00_environment_check.ipynb          read only — versions, four datasets, three one-liners
Ex05_01_cnn_image_classification.ipynb   convolution by hand, then a CNN on synthetic weld radiographs
Ex05_02_graph_basics.ipynb               adjacency, neighbourhoods, message passing — NumPy only, no torch
Ex05_03_gnn_six_bus_network.ipynb        node regression on six buses; permutation equivariance measured
Ex05_04_sequence_model.ipynb             RNN, LSTM and self-attention written out; the honest comparison
Ex05_05_report.ipynb                     the two marked questions, plus the four from L3.2
```

Notebooks 01, 03 and 04 each write an `.npz` into `Ex05_outputs/`; notebook 05
reads all three and refuses to build a report without them.

## Nothing is downloaded

All four datasets are generated on the student's machine by `Ex_5_core.py`. The
usual first CNN exercise fetches MNIST from a mirror, and every year that mirror
is slow, blocked by a university firewall, or has moved. The weld radiographs are
twenty lines of NumPy and are identical on every machine in the room.

## Notebook 03 is the one that matters

`Ex_5_core.six_bus_dataset()` builds a small load-flow dataset on a six-bus
network with two generators, three load centres and an HVDC infeed. The module
docstring separates **real physics from teaching simplification** explicitly, and
notebook 03 section 1 repeats the separation in the student's face, because
Ex_12.1 makes the same kind of declaration about its own network and the habit is
the point:

- **Real.** The topology; the susceptance matrix as a weighted graph Laplacian;
  the bus angles, obtained by Newton-Raphson on the lossless active power flow
  `P_i = Σ b_ij sin(θ_i − θ_j)`; balanced injections; the sign conventions.
- **Simplified.** Everything about voltage magnitude — a linear response to
  reactive injection minus a sag proportional to squared line angle differences.
  Right qualitative behaviour, wrong numbers.
- **Invented.** Line susceptances, injection ranges, noise level, and the fact
  that bus 5 is an HVDC link.

### The result the notebook is built around

| model | parameters | angle RMSE [rad] | voltage RMSE [p.u.] | permutation gap [rad] |
|---|---|---|---|---|
| DC power flow (no training) | 0 | 0.00232 | not predicted | — |
| graph network, 3 layers | 4,578 | 0.00431 | 0.00170 | 1.7e-16 |
| dense network on the flattened state | 6,924 | 0.00163 | 0.00071 | 5.7e-01 |

**The dense baseline is the most accurate model, and the fifty-year-old linear
approximation beats the graph network on angles.** The notebook says so plainly
rather than arranging a win, because arranging a win teaches students to expect
one. What the graph network has instead: it predicts voltage magnitude, which the
DC solve does not; its parameter count does not grow with the network; and it is
**exactly** permutation equivariant, so relabelling the buses leaves its accuracy
unchanged to five decimal places while the dense network's error rises by two
orders of magnitude.

That is the L5.1 report question: *the more accurate model is broken by a
relabelling — which would you deploy, and what would have to be true for that to
be right?*

### Two further experiments in notebook 03

- **Depth sweep.** Angle RMSE 0.0304, 0.0075, 0.0043, 0.0055 for one to four
  layers. The graph's diameter is three, measured in notebook 02, and that is
  where the returns stop. The notebook is explicit that depth and parameter count
  moved together, so only the *shape* of the curve supports the diameter
  argument.
- **A line trip.** Both models degrade badly on a topology they never saw — the
  graph network to 0.115 rad, the dense one to 0.265, against a data spread of
  0.32. The conclusion is the one L12.1 slide 13 reaches: being able to *accept*
  a new topology is necessary for transfer and not sufficient. Training across
  topologies is the fix, and it is offered as an extension in notebook 05.

### The `is_reference` feature

Angles are defined only up to a common offset, so one bus must be nominated as
the reference. A permutation-equivariant model has no notion of bus number and
therefore cannot know which one — so the reference is supplied as a **node
feature** that permutes with everything else. This is the same design point
L12.1 makes about measurement masks: anything positional must be a feature.

## Notebook 04 ends where L5.2's closing slide does

Four architectures on 702 windows of a synthetic load profile:

| model | parameters | held-out MSE | seconds |
|---|---|---|---|
| perceptron | 417 | 0.000786 | 0.5 |
| recurrent | 305 | 0.001271 | 5.4 |
| LSTM | 1,169 | 0.000793 | 27.6 |
| attention | 409 | 0.000705 | 14.3 |
| persistence baseline | 0 | 0.003320 | — |

Attention looks best by ten per cent — so the notebook makes the student train
both contenders from three seeds. Perceptron: 0.000786, 0.000709, 0.000771.
Attention: 0.000705, 0.000778, 0.001066. **The distributions overlap completely
and the attention model's mean is worse.** The advantage was a property of seed
zero.

The RNN, the LSTM's four gates and scaled dot-product attention are all written
out by hand rather than called from `torch.nn`, because those five or six lines
are the entire content of each architecture.

**The residual check is deliberately a failure.** The perceptron's held-out MSE
looks excellent against persistence, and its residuals have a lag-1
autocorrelation of about 0.5 — the AR(1) noise in the generator is half
predictable and the model did not take it. Neither the loss curves nor the MSE
revealed this; only the two-line diagnostic did. That is the point of the
section, and it becomes report question 4.

## Notebook 01

2,019-parameter CNN reaches 0.983 held-out accuracy on the three-class weld task;
a 16,643-parameter dense network reaches 0.722, and is worse on the **training**
set too. The notebook is careful that the second fact is not overfitting: the
dense model has to learn translation invariance from data, separately in every
region of the image, and 420 images are not enough. It is also explicit that the
comparison is deliberately favourable, because the defects are uniformly
distributed over the plate.

## Notebook 02 uses no torch at all

Adjacency from an edge list, neighbourhoods, the diameter from powers of `A`,
message passing with loops and then as a matrix product, `Â = D̃^-1/2 Ã D̃^-1/2`
built by hand, and permutation equivariance verified on **random** weights — a
stronger statement than verifying it on a trained model. The over-smoothing
demonstration (a signal starting at bus 0 and flattening after four rounds) is
what motivates the residual connections in notebook 03.

## How the numbers in these notebooks were verified

Every notebook was executed end to end with reference solutions filled into the
`# TODO:` cells, and every figure quoted under a "What you should see" heading is
a number that run actually produced. Nothing here is estimated.

One caveat the reader should have. `torch` could not be installed in the machine
these notebooks were written on, so the runs used a **NumPy reverse-mode autograd
stand-in** implementing the slice of the torch API these notebooks touch
(`Linear`, `Conv2d`, `MaxPool2d`, `Flatten`, `Dropout`, the losses, `SGD`, `Adam`
and `LBFGS`), gradient-checked against finite differences. It is a test double,
not a copy of PyTorch, so:

- **Deterministic quantities are exact** and will match on any machine —
  parameter counts, shapes, hand-written convolutions, adjacency and permutation
  algebra, closed-form fits, softmax and cross-entropy values, quantisation
  arithmetic, and the analytic gradient tables.
- **Anything downstream of a random initialisation will differ**, because the
  stand-in draws its initial weights from NumPy's generator rather than
  PyTorch's. Trained losses and accuracies should land within the range each
  notebook states, and the notebooks say so wherever it matters.
- **The L-BFGS implementation uses a backtracking Armijo line search**, whereas
  `torch.optim.LBFGS` offers a strong-Wolfe search. The behaviour is the same and
  the ordering of the recipes is the same; the exact final losses will not be.

Re-run the notebooks once against real PyTorch before the first teaching session
and adjust any quoted figure that has moved outside the stated range.

## Conventions

Same as every Part 2 exercise set:

- Self-contained folder. Requires `torch`, `numpy`, `matplotlib` — all
  preinstalled on Google Colab. No GPU needed.
- Notebook `00` is a read-only environment check. Run it first.
- Modules (`Ex_5_core.py`) are complete and are **not** to be rewritten by
  students. The work is in `# TODO:` cells in the numbered notebooks, each
  followed by `raise NotImplementedError`.
- Module names use underscores because a Python module name cannot contain a
  dot: `Ex_5_core.py`, imported as `Ex_5_core`.
- Every `TODO` is preceded by enough prose that the notebook works for a student
  who missed the lecture. Part 1 exercises are deliberately more discursive than
  the Part 2 ones.
- Seeds are set everywhere, and notebook 04 makes the point that a single seed is
  not evidence by training three of them.
- British spelling throughout.

## The two marked questions

> **L5.1.** On the six-bus network the dense baseline was more accurate and the
> graph network survived a relabelling. Which would you deploy, and what would
> have to be true about the deployment for that to be the right choice?

> **L5.2.** Four architectures came within a factor of two of each other, and the
> differences were comparable to the seed-to-seed spread. What would have to
> change about the data for the ranking to become meaningful?

Notebook 05 also asks the four questions from L3.2 slide 2 about the model the
student chose, and includes a worked example answer so that the expected standard
is visible rather than guessed.

## Depends on

- **L5.1** — the opening framing (a learned local rule applied everywhere),
  invariance and equivariance, CNN mechanics, residual connections, message
  passing, the graph convolution in matrix form, and permutation equivariance.
  All six of L5.1's rendered equations appear in these notebooks.
- **L5.2** — sequential data in engineering, the recurrence relation, vanishing
  gradients, the LSTM's gates (GBC Ch. 10), scaled dot-product self-attention and
  positional encoding, and the honest closing slide.
- **L3.2** — the weld-inspection framing of notebook 01, and the four questions.
- **Ex_04**, whose seed-variation lesson notebook 04 repeats, and whose
  equal-budget sweep notebook 05 suggests as an extension.

## Forward to Part 2

- **L12.1 slide 12** says *"you met graph neural networks in Part 1"*. This is
  where. Ex_12.1's six buses are these six buses.
- **L12.2** on depth as a physical choice, on equivariance, and on the difference
  between avoiding retraining and avoiding recomputation — notebook 03 sections 7
  and 8 are the Part 1 version of both arguments.
- **L11.1** runs a CNN on the JetRacer camera; notebook 01 is the preparation.

## Expected runtime

CPU only. Notebooks 00 and 02 take seconds. Notebook 01 trains two models and
takes a minute or two. Notebook 03 trains six models and takes three to five
minutes, most of it in the depth sweep. Notebook 04 is the slowest — a Python
loop over twenty-four time steps cannot be vectorised away, which is one of
L5.2's own points — and takes about five minutes including the three-seed
comparison. The difficulty is conceptual, not computational.
