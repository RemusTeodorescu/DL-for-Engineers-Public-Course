# Ex05 — CNN and GNN

**Paired with lecture block L5 · Part 1**

One idea, three kinds of data: **a learned local rule, applied everywhere.** A
convolution applies it on a grid, message passing applies it on a graph, a
recurrent network applies it along time. Notebook 03 predicts the power flow on
a six-bus network with a graph network.

The set was simplified on 21 September 2026, with L5.1 and L5.2, after the first
teaching session: fewer TODOs, one straight line from problem to result in each
notebook, and only what the two lectures now teach — convolution, message
passing, the recurrent network and the LSTM.

## Goals

By the end of this exercise set you can

1. build a small CNN, set a convolution's weights by hand, and show that the
   trained CNN beats a dense network on the same images with far fewer
   parameters;
2. build an adjacency matrix and perform message passing by hand;
3. show permutation equivariance to machine precision, and a dense network
   failing the same test;
4. train a graph network to approximate the AC power flow, compare it with a
   dense network, and weigh its accuracy against its speed — also after a line
   trips;
5. build a recurrent network and an LSTM, and compare them with the persistence
   forecast on the same held-out data.

## State

**Written.** Six notebooks and one module. Every light notebook was run end to
end against PyTorch on 21 September 2026, and every number under a "What you
should see" heading is a number that run printed.

## Notebooks

Run in order; notebook 05 loads results saved by 01, 03 and 04.

```
Ex05_00_environment_check.ipynb          read only — versions, the three datasets, three one-liners
Ex05_01_cnn_image_classification.ipynb   a CNN on synthetic weld radiographs, against a dense network
Ex05_02_graph_basics.ipynb               adjacency, neighbours, message passing — NumPy only, no torch
Ex05_03_gnn_six_bus_network.ipynb        a graph network predicts the AC power flow on six buses
Ex05_04_sequence_model.ipynb             persistence, a recurrent network and an LSTM on hourly load
Ex05_05_report.ipynb                     the two marked questions, plus the four from L3.2
```

Notebooks 01 to 04 each have a `_light` twin with the same text and every cell
written out — `Ex05_01_cnn_image_classification_light.ipynb` and so on.
Notebook 00 and the report are the same for both. Start from notebook 00, which
links to whichever of the two you choose.

Notebooks 01, 03 and 04 each write an `.npz` into `Ex05_outputs/`; notebook 05
reads all three and refuses to build a report without them.

## Nothing is downloaded

All three datasets are generated on the student's machine by `Ex_5_core.py`. The
usual first CNN exercise fetches MNIST from a mirror, and every year that mirror
is slow, blocked by a university firewall, or has moved. The weld radiographs are
twenty lines of NumPy and are identical on every machine in the room.

## Notebook 01 — a CNN against a dense network

Three classes of 16 × 16 weld radiograph: clean, crack, pit. The notebook
defines a two-layer CNN (2,019 parameters), sets four classic 3 × 3 kernels into
a convolution by hand to show what a feature map is, trains the CNN, and then
trains a dense network (16,643 parameters) with exactly the same recipe:

| model | parameters | train accuracy | held-out accuracy |
|---|---|---|---|
| CNN | 2,019 | 1.000 | 0.989 |
| dense | 16,643 | 0.995 | 0.711 |

Both fit their training images; only the CNN carries what it learned to images it
has not seen, because it uses one kernel at every position. The notebook is also
honest about the cost: the CNN trains several times slower. A convolution saves
parameters, not computation. The comparison favours the CNN because the defects
are equally likely anywhere on the plate, and the notebook says so.

## Notebook 02 uses no torch at all

Adjacency from an edge list, neighbourhoods, the diameter (three hops) from
powers of `A`, one round of mean aggregation with loops and then as a matrix
product, and permutation equivariance of a message-passing layer,
`tanh(H W_self + A H W_neigh)`, verified on **random** weights — a stronger
statement than verifying it on a trained model. The gap is exactly zero. The
over-smoothing demonstration, a signal starting at bus 0 and flattening after a
few rounds, is what the depth sweep in notebook 03 measures.

## Notebook 03 is the one that matters

`Ex_5_core.six_bus_dataset()` solves 800 operating points of a six-bus network
with a full AC power flow (Newton-Raphson). **The AC power flow is the ground
truth; the graph network learns to approximate it, faster.** Bus 0 is the slack
bus (|V| = 1.03 p.u., θ = 0), bus 1 a PV generator (|V| = 1.02 p.u.), and buses
2 to 5 are PQ buses — three loads and an HVDC infeed. Each line is given by its
impedance R + jX. The notebook separates what is real from what is invented,
because labelling assumptions is a habit worth forming early:

- **Real.** The AC power-flow equations, the bus types, per unit on 100 MVA.
- **Invented.** The line impedances (X/R about 8), the injection ranges and the
  set-points. Line charging is neglected and generator reactive limits are not
  enforced.

The graph network is three message-passing layers — one per hop of the diameter —
with 32 numbers per bus. Its error against the AC power flow on the 200 held-out
cases, next to a dense network on the flattened state:

| model | parameters | angle RMSE [rad] | \|V\| RMSE [p.u.] | angle RMSE, buses renumbered |
|---|---|---|---|---|
| graph network, 3 layers | 4,642 | 0.00057 | 0.00033 | 0.00057 |
| dense network | 7,308 | 0.00078 | 0.00041 | 0.04860 |

**The graph network is the more accurate, and the only one that survives a
renumbering of the buses** — 0.03° on angles and 0.03 % of nominal on |V|.

**Accuracy against speed.** Time per case on the CPU, median of several runs,
from the notebook run of 21 September (timings vary between runs and machines;
read the ratios):

| | AC power flow | GNN, one case | GNN, 200 cases at once |
|---|---|---|---|
| six buses, trained network | 0.25 ms | 0.12 ms (about 2x faster) | 0.0023 ms per case (about 100x faster) |

A sweep over synthetic grids (`core.synthetic_grid`: a ring with chords, a PV
generator every fourth bus, the same three-layer network with random weights,
since a forward pass costs the same whatever the weights) asks how the two grow
with size. Newton-Raphson is timed dense or sparse, whichever is faster at that
size. The notebook runs 6 to 768 buses in about half a minute; a separate run
took it to 6,144:

| buses | AC power flow [ms] | GNN, one case [ms] | GNN, batch of 256 [ms per case] |
|---|---|---|---|
| 6 | 0.157 | 0.107 | 0.0029 |
| 96 | 1.51 | 0.20 | 0.014 |
| 768 | 14.2 | 0.76 | 0.32 |
| 6,144 | 184 | 4.0 | 3.1 |

In a batch the graph network is faster at every size, about fifty times at six
buses. One case at a time the two are level up to about fifty buses and the
graph network pulls ahead from about a hundred, to about twenty times at 768
and forty-five at 6,144. The notebook states the caveats: both sides are Python,
so the small-grid times are mostly overhead; a network that must reach across a
large grid needs about one layer per hop, which shrinks the batched margin to
about four times at 768 buses; and the timed networks are untrained.

Two further experiments:

- **Depth sweep.** Angle RMSE 0.00706, 0.00073, 0.00057, 0.00069 and 0.00277 for
  1, 2, 3, 4 and 6 layers. One layer cannot reach across the network; six
  over-smooth.
- **A line trips.** With line 1-3 out and no retraining, the error against a new
  AC power flow rises to 0.052 rad for the graph network (given the new
  adjacency) and 0.078 rad for the dense network. The power flow has voltages
  below 0.95 p.u. at buses 3 to 5 that both networks miss. Being able to
  *accept* a new topology is necessary for transfer and not sufficient; training
  across topologies is offered as an extension in notebook 05.

Fast and nearly right on the network it was trained on, wrong where a
contingency study needs it to be right: that is the L5.1 report question.

Because the model has no notion of bus number, the bus type — slack, PV or PQ —
reaches it as three flag features that move with the bus when the buses are
renumbered.

## Notebook 04 — persistence, a recurrent network and an LSTM

One-hour-ahead forecasting of forty days of synthetic substation load, 702
training windows of 24 hours and 234 held out, split in time order. The two
networks are `nn.RNN` and `nn.LSTM` (hidden size 16) with a linear head, each in
its own cell with its parameter count written beside it, trained with one recipe
(full batch, Adam, learning rate 0.01, 300 epochs):

| model | parameters | held-out MSE [p.u.²] | vs persistence |
|---|---|---|---|
| persistence (next hour = this hour) | 0 | 0.003320 | 1.00 |
| recurrent network | 321 | 0.001298 | 0.39 |
| LSTM | 1,233 | 0.000910 | 0.27 |

Both networks beat persistence; the LSTM beats the recurrent network with 3.8
times the parameters. The notebook says that this is one seed and that neither
network has converged at 300 epochs. The forecast plot shows where the gain
comes from: persistence is always an hour late on the morning rise and the
evening fall.

## Conventions

Same as every Part 2 exercise set:

- Self-contained folder. Requires `torch`, `numpy`, `matplotlib` — all
  preinstalled on Google Colab. No GPU needed.
- Notebook `00` is a read-only environment check. Run it first.
- The module (`Ex_5_core.py`) is complete and is **not** to be rewritten by
  students. The work is in the `TODO` cells of the numbered notebooks: three or
  four per notebook, one to three lines each.
- Module names use underscores because a Python module name cannot contain a
  dot: `Ex_5_core.py`, imported as `Ex_5_core`.
- Every `TODO` is preceded by enough prose that the notebook works for a student
  who missed the lecture.
- Seeds are set everywhere, so every printed number is the same on every
  machine; the training times are not.
- British spelling throughout.

## The two marked questions

The same questions as on the lectures' Exercise slides; the report in notebook 05
is built around them.

> **L5.1.** The graph network answers far faster than the AC power flow, and
> less exactly. When is the speed worth the error, before and after a line
> trips?

> **L5.2.** Three forecasts of the same load. Which would you deploy on a
> substation controller, and which number or plot decided it?

Notebook 05 also asks the four questions from L3.2 about the model the student
chose, and includes a worked example answer so that the expected standard is
visible rather than guessed. It then collects the student's answers to the four
"Before you move on" questions of each notebook.

## Depends on

- **L5.1** — a learned local rule applied everywhere, convolution, stride,
  padding, channels and pooling, graphs as X, A and E, message passing, and
  permutation equivariance.
- **L5.2** — sequences and the persistence baseline, the recurrent network and
  its hidden state, and the LSTM's cell state and gates (GBC ch. 10).
- **L3.2** — the weld-inspection framing of notebook 01, and the four questions.
- **L4.2** — depth against width at an equal parameter budget, the extension
  notebook 05 suggests for the depth sweep.

## Expected runtime

CPU only. Measured on 21 September 2026 on a desktop CPU, running the light
notebooks top to bottom: notebook 00 about 2 s, 01 about 10 s, 02 about 2 s, 03
under a minute (the depth sweep, five graph networks, and the speed sweep to 768
buses), 04 about 10 s, and the report 2 s. Allow several times that on a laptop or on Colab. The difficulty
is conceptual, not computational.

## Results between notebooks on Colab

Later notebooks read `.npz` files that earlier ones write into `Ex05_outputs/`.
On Google Colab every notebook runs on its own temporary machine, so those files
would not survive from one notebook to the next. Notebooks 01, 03, 04 and 05
therefore start with an `outputs-cell` that calls `keep_outputs()` from
`Ex_5_core.py`: on Colab it mounts the student's Google Drive and moves the
results folder to `MyDrive/DL4Eng/Ex05_outputs`. If the student declines the
Drive request or has no Google account, `saved()` downloads each result file when
it is written, and `needed()` asks for the files to be uploaded before they are
read. Locally the cell does nothing. The report notebook writes its `.md` and
`.pdf` into the same folder.
