# Ex_12.2 — Prediction of Power Grid Stability

**Paired with L12.2 · Prediction of Power Grid Stability · Part 2**

Ex_12.1 asked what the grid is doing **now**. This set asks where it is
**heading**: given the present operating point, which single-line outage would
leave the system unable to survive a fault?

That question is **N-1 screening**, and a control room answers it for every
credible contingency several times an hour. The honest answer is a transient
simulation per contingency, and the honest answer is too slow. So the exercise
is to build a surrogate, and to be exact about what it is allowed to replace.

The architectural question it turns on is narrow and it decides everything:

> **A line outage changes the network's topology.** A dense network sees a
> fixed-length vector and can only be told about the outage by a one-hot flag;
> a graph network is handed the modified adjacency directly. A line out is a
> *different input*, not a different label — and that is why the graph network
> from Ex_05 notebook 03 and L5.1 is the right architecture here.

## Running on Colab: one runtime, in order

Every Colab tab is a separate machine with its own filesystem. Notebooks 03,
04 and 05 load the training results of the notebooks before them, and every
one of those is a run of yours — so nothing is rebuilt automatically, and a
file saved in one tab does not exist in another. Work through the numbered
notebooks in order **in the same runtime**, or move results between runtimes
by downloading the `Ex12.2_outputs/` folder from the Files panel and
uploading it in the new tab.

## Notebook 06 — latency, and what a Colab number is worth

Everything in this exercise runs on Colab, including the latency measurements.
You do not need a Jetson Thor.

What notebook 06 measures and you can defend: how cost per contingency falls
with batch size, what half precision is worth as a ratio, how cost grows with
the number of buses, and how the learned screener compares with time-domain
integration on the same machine. Those are properties of the model and the
arithmetic, and they hold on any reasonable accelerator.

What it cannot tell you: absolute latency on edge hardware. The GPU Colab gives
you is not a Thor. Report those rows as pending rather than guessing.

Every timing is a median of repeated runs after a warm-up, reported with its
spread. A single Colab timing is noise.

The L13 mini-project runs this notebook unchanged on a real Thor and fills the
empty column.

## Goals

By the end you can

1. enumerate an N-1 contingency set from the connectivity of an admittance
   matrix, and say why an islanding outage is a different kind of event that
   does not belong on the same axis;
2. produce a critical clearing time by simulation — power flow, Kron reduction,
   equilibrium, bisection on the swing equations — and quote what one costs;
3. train a dense surrogate on the flattened state with a one-hot outage flag,
   score it in two currencies, and demonstrate the two failures that are
   properties of that representation rather than of the fit;
4. build a message-passing model that takes the adjacency as an input, and
   **measure** its permutation invariance to machine precision rather than
   asserting it;
5. choose a graph model's depth from the graph's diameter, and say what a model
   one layer short is structurally unable to represent;
6. set a screening threshold from the asymmetry between a missed insecure
   contingency and a false alarm, and defend the choice with numbers.

## The problem

The same six-bus, DK2-representative network as **Ex_12.1** — `BUSES` and
`BRANCHES` character-identical, and `problem.check_continuity()` prints them so
a student can diff rather than trust.

Ex_05 notebook 03 and L5.1 also work on a six-bus network, and it is **not this
one**: that is a DC, susceptance-only model with eight lines and buses named
`B1 gen … B6 HVDC`, chosen so message passing can be done by hand. This set and
Ex_12.1 use six `(r, x, b)` branches and named buses `Slack / SE link … HVDC /
DE link`. The continuity across the course is of the *object* — a small meshed
transmission system carried from a supervised regression to a state estimator to
a stability screen — not of the numbers. Do not expect a diff of the two to
match.

A contingency is the textbook one:

```
a solid three-phase fault at the sending end of branch k,
cleared at time t_c by tripping branch k
```

The system survives if the machines stay in step. The longest fault it survives
is the **critical clearing time**, and it is what everything here predicts:

```
CCT > PROTECTION_TIME   the protection is fast enough  — secure
CCT < PROTECTION_TIME   the fault outlasts the breakers — insecure
```

`PROTECTION_TIME = 0.14 s` is seven cycles at 50 Hz and it is a budget, not a
guess: 1.5 cycles of relay decision, 2.5 cycles of breaker interruption, and
three cycles of margin for pole scatter and the slower end of a permissive
scheme.

Six branches, one of which islands bus 5 and is dropped, so the set is the base
case plus five outages. Crossed with 180 plausible dispatches that is **1080
labels**, and building them takes about **five and a half minutes** of CPU —
which is the entire argument for a surrogate, stated in seconds.

Of those 1080 labels, **16.2 % are insecure**, so a model that says "secure"
for everything scores 83.8 %. Notebook 01 puts that number on the page before
any model is trained, which is what stops it being mistaken for a result later.

**Two things this set adds to Ex_12.1's physics.** A transient reactance between
each machine's internal EMF and its terminal bus, without which a solid fault at
a machine bus is a short across the EMF itself and cannot be represented; and
internal EMFs computed from the power flow rather than held fixed, because here
the operating point *is* the input. Both are documented in `problem.py` §4.

## The notebooks

Run in order; later notebooks load results saved by earlier ones.

```
Ex12.2_00_system_check.ipynb        the network, the contingency list, ONE CCT and what it cost
Ex12.2_01_contingency_set.ipynb     build the dataset; class balance; label one case by hand
Ex12.2_02_dense_baseline.ipynb      flatten the state, one-hot the outage — and the two failures
Ex12.2_03_graph_network.ipynb       the adjacency as an input, and the permutation test
Ex12.2_04_screening.ipynb           rank, threshold, and price the two kinds of error
Ex12.2_05_compare_and_report.ipynb  the held-out-topology table, and the report
```

Notebook 00 is read-only. Notebooks 01–05 have `# TODO:` cells.

## Files

Every Part 2 exercise has the same three modules beside it. Only the third
differs between sets.

| | |
|---|---|
| `course_core.py` | shared by the whole course — `set_seed`, `MLP`, `to_tensor`, `check` |
| `pinn_core.py` | the PDE machinery — `grad`, `d2`, samplers, `train_two_stage` |
| `problem.py` | **this** problem — network, contingencies, labels, graph layer, plots |

The first two are generated. Edit `tools/pinn/*.py` and run
`python3 tools/pinn/sync_cores.py`; never edit a copy.

`problem.py` is in four parts and it is worth knowing which is which:
sections 1–3 are **copied verbatim from `Ex12.1/problem.py`**, values unchanged;
section 4 completes the classical machine model so a fault has a location;
sections 5–8 are the contingency set, the operating points, the labelling and
the dataset; sections 9–10 are the graph machinery and the lab helpers.

Outputs — the dataset cache, the saved predictions and the report — go to
`Ex12.2_outputs/`.

**Colab needs the whole folder**, not just a notebook — the imports expect the
three modules beside them. `scipy` is required as well as `torch`; Colab has it.

## Expected runtime

CPU only; no GPU is required anywhere in Part 2.

| | |
|---|---|
| notebook 00 | two to three minutes — the simulations are small but there are several |
| notebook 01 | **five to six minutes** the first time, then instant from the cache |
| notebook 02 | a few minutes, one training run |
| notebook 03 | a few minutes, one training run |
| notebook 04 | seconds — it only reads predictions |
| notebook 05 | ten minutes — two more training runs for the held-out-topology table |

If notebook 01 is too slow for a lab session, `n_ops=60` runs in under two
minutes and everything downstream still works. Say so in the report if you do.

## Reference texts

Liu, G.R., *PINN with Python: An Introduction* (2025).
Raissi, Perdikaris & Karniadakis, *Physics-informed neural networks*,
J. Comput. Phys. **378** (2019) 686–707.

These are the works to read for the theory. **The code, the problem and the
exposition in this exercise set are original to this course** — written from the
2019 paper and the PyTorch documentation, and not derived from any publisher's
code listings. Where a symbol matches a textbook's, it is because both follow
the standard notation of the field.

## Before this is assigned

Written in an environment where **PyTorch could not be installed**. The split
between what has been executed and what has not is therefore sharp, and it is
worth reading before the set goes out.

**Executed, and every number in the notebooks' "what you should see" blocks
comes from these runs.** The whole NumPy/SciPy half:

* `build_ybus` with and without an outage — `Y[0,1] = -3.8367+32.8857j` intact
  and exactly `0j` with line 0 out, 18 non-zero entries falling to 16, both
  symmetric;
* `solve_power_flow` on the base case — converges in 5 iterations, `|V|` from
  0.9706 to 1.000, angles inside ±2.5°, slack injection +0.59505 p.u., losses
  +0.01505 p.u., residual mismatch 2.2e-14;
* `contingencies()` — 6 branches, 1 skipped as islanding (branch 5, the spur to
  bus 5), 6 contingencies;
* `check_continuity()` — prints the six buses and six lines; `BUSES` and
  `BRANCHES` verified character-identical to `Ex12.1/problem.py`;
* one full `label_case` — CCT 219.7 ms for the tie outage at the base dispatch,
  0.4 s of CPU, with two independent consistency checks passing (the machine is
  in equilibrium before the fault; the equilibrium angle equals `arg(E1)−arg(E0)`
  to 1e-9, both giving 18.6830°);
* the full `build_dataset(180, 12)` — **1080 labels in 335.1 s, 310 ms each**;
  CCT min 55.7 ms, median 226.6 ms, max 500 ms; 16.2 % insecure at 140 ms;
  17.2 % censored at the 500 ms ceiling;
* the linear/ridge reference results quoted in notebooks 02–05: test MAE
  20.72 ms against 108.21 ms for predicting the mean; the held-out-topology
  result (24.03 ms seen, 67.42 ms held out, +67.4 ms optimistic bias, 53 of 72
  insecure cases missed, recall 0.264); the permutation result (2873.3 ms
  maximum change, MAE 20.72 → 1709.9 ms); the threshold sweep (zero-miss at
  138 ms for three false alarms) and the top-k table (perfect ranking, 42 % of
  insecure cases caught at k = 1).

**Not executed. Nothing that requires torch has been run.** No network in this
set has been trained, and no notebook states a trained-model number. Where a
"what you should see" block covers a trained model it describes shapes,
parameter counts and qualitative behaviour, and says explicitly that the number
is yours. Two specific expectations are stated as predictions rather than
measurements and should be checked on the first real run:

* the graph model's permutation gap, argued to be at float64 round-off (order
  `1e-14` ms or smaller) because the invariance is exact in exact arithmetic;
* the parameter count of `ScreeningGNN(n_in=6, hidden=32, depth=3)`, derived by
  counting as **6,721** — 4,576 in the message-passing layers, 2,145 in the
  head. The notebook prints it; if it disagrees, the architecture drifted.

**Also verified without torch.** Every notebook parses as JSON with
`nbformat == 4`; every code cell passes `ast.parse`; `problem.py` passes
`ast.parse`; the setup cell is byte-identical to Ex_07.1's in all six notebooks;
no notebook carries a course-header cell (the stamping tool adds those); every
`pb.*` name referenced in a notebook exists in `problem.py`. Every code cell
that does not need torch was **executed**, including all of notebooks 00 and 01
except the two TODO-dependent cells, and every post-TODO cell in notebooks 04
and 05 was executed against a stand-in solution to confirm it uses the names the
TODO asks for and runs.

**One open question, and it is a real one.** Notebook 05's held-out-topology
table predicts that the graph model degrades less than the dense one. That is
the argument of L12.2 and it is what the representation implies, but it has not
been measured here and six contingencies is a very small number of topologies to
learn structure from. The notebook says so, tells the student to report a
negative if they get one, and lists the three things to check first. **Run
notebook 05 before this goes to students** — if the graph model does not win,
the text is already written to accommodate it, but the teaching staff should
know which way it went.

**One discrepancy worth recording.** The six-bus network in `Ex05-cnn-and-gnn/Ex_5_core.py`
is *not* this network: it has eight lines described by susceptances
(`SIX_BUS_LINES`), where Ex_12.1 and Ex_12.2 have six lines described by
`(r, x, b)`. The two share a bus count and a role, not a topology. This set is
byte-identical to **Ex_12.1**, which is what the brief required; the Ex_05
mismatch predates it and should be resolved in one direction or the other before
the continuity claim in L5.1 is repeated to students.

Run notebook 00 end to end before this goes out. See `docs/REPO_NOTES_PART1.md` §9.
