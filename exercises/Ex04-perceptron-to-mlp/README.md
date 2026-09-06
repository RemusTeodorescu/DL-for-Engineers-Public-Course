# Ex04 — Perceptron to MLP

**Paired with lecture block L4 · Part 1**

Demonstrate the XOR failure, then fix it. Sweep depth against width and find out
how small a network the problem actually needs — then overfit one deliberately
and regularise it back.

## Goals

By the end of this exercise set you can

1. implement a perceptron and its learning rule, and demonstrate the XOR
   failure rather than being told about it;
2. set by hand the nine parameters of a 2-2-1 network that computes XOR, and
   say what the hidden layer did;
3. count the kinks a shallow ReLU network produces and confirm the
   piecewise-linear picture that L4.1 argues for;
4. compare depth against width at a **fixed parameter budget**, as a controlled
   experiment with more than one seed;
5. overfit a network deliberately, recover it with weight decay and with early
   stopping, and measure how much of the damage each fix undid.

## State

**Written.** Seven notebooks and one module. No solutions notebook; the `# TODO:`
cells are the exercise.

## Notebooks

Run in order; later notebooks load results saved by earlier ones.

```
Ex04_00_environment_check.ipynb         read only — versions, three datasets, one run
Ex04_01_perceptron_and_xor.ipynb        NumPy perceptron: AND works, XOR does not
Ex04_02_pytorch_mlp.ipynb               the same nine parameters, found by gradient descent
Ex04_03_counting_kinks.ipynb            sweep D; the fit is piecewise linear with D kinks
Ex04_04_depth_vs_width.ipynb            one budget of 1000 parameters, spent three ways
Ex04_05_overfit_then_regularise.ipynb   the lecture's eleven points, overfitted then rescued
Ex04_06_report.ipynb                    the two marked questions, plus the four from L3.2
```

Notebooks 03, 04 and 05 each write an `.npz` into `Ex04_outputs/`; notebook 06
reads all three and refuses to build a report without them.

## Notebook 01 is meant to fail

Section 4 runs the perceptron learning rule on XOR for two hundred epochs and it
never converges. **This is intended.** The notebook says so three times — in the
header, in a block quote immediately before the cell, and in the note after it —
because every year somebody reports it as a bug. L4.1's own speaker notes say to
warn the room about this.

The failure is then made airtight two ways: the four-line algebraic contradiction
from Minsky and Papert (1969), and a brute-force search over 200,000 random
lines whose best score is 3 of 4. Only then does the student set the **nine
parameters** of a 2-2-1 network by hand — OR, NAND, and an AND of the two — and
see all four rows come out right. The hidden-space plot at the end is L4.1 slide
9's "bending the plane so a straight line suffices", drawn from the student's own
weights.

The nine numbers are set by hand rather than trained because in 1969 nobody knew
how to train them. Notebook 02 trains them, which is the point of notebook 02.

## The capacity target

`wiggly_truth` is `sin(πx) + 0.35 sin(3πx)` on **[-1, 1]**, used by notebooks 03
and 04. Two length scales, so a network needs more than three kinks before it can
follow it; centred on zero, because a ReLU network trains markedly better on
centred inputs and Ex_04 would rather not spend a notebook on input scaling.

Notebook 03 trains each `D` from four seeds and keeps the best, and says why in
the text: a single-seed sweep of shallow ReLU networks is mostly a picture of
initialisation luck, and units that die never recover. With that, the sweep is
monotone — training MSE 0.16, 0.15, 0.032, 0.032, 0.014, 0.007, 0.0006 for
`D = 1, 2, 3, 5, 10, 20, 40` against a noise floor of 0.0004 — and the kink count
is at most `D` everywhere, falling well below it for the larger networks.

Notebook 04's budget of 1000 parameters buys 1×333, 2×29 or 4×17 units. On this
target all three reach roughly the same best-seed validation error, near the noise
floor; what differs is how reliably they train. The notebook says so, and turns it
into the marked question rather than pretending depth won.

## Notebook 05 uses the lecture's own data

`Ex_4_core.lecture_dataset()` reproduces **the eleven points from L4.2 slides 11
to 13 exactly** — same truth `0.30 + 0.52 sin(2.9x) + 0.10x`, same eleven noise
values, copied from `lectures/L04.2-deep-neural-networks/src/L4.2_generator.js`.
If that generator ever changes, change `LECTURE_NOISE` and `lecture_truth` to
match. The whole point is that slide 13 appears on the student's own screen.

The held-out set is forty points from the same truth with the same noise
standard deviation — a second measurement campaign, and the only thing that can
reveal that the flexible model is worse.

The lab: 4353 parameters on eleven points, training loss driven to order 1e-7
against a noise variance of 0.0049; then a weight-decay sweep over seven values;
then early stopping via the best-validation epoch; then a table of how much of the
damage each fix recovers.

The recovery is scored on the **error against the true curve**, not on the
validation error, and the notebook is explicit that this is only possible because
the data is synthetic. The reason is worth knowing before marking: with eleven
training points and forty held-out points, the validation error moves from 0.0067
to 0.0040 — less than a factor of two, most of it measurement noise — while the
error against the truth moves from 0.0052 to 0.0002. Scoring on the validation
gap above the noise floor produces recoveries above 100 %, because a forty-point
sample fluctuates below its own floor.

**Every run is plotted with training and validation on the same axes** — a
fifteen-panel grid in notebook 04 and a seven-panel grid in notebook 05 — because
that is a standing requirement of the course and this is the notebook that shows
why.

## Conventions

Same as every Part 2 exercise set:

- Self-contained folder. Requires `torch`, `numpy`, `matplotlib` — all
  preinstalled on Google Colab. No GPU needed.
- Notebook `00` is a read-only environment check. Run it first.
- Modules (`Ex_4_core.py`) are complete and are **not** to be rewritten by
  students. The work is in `# TODO:` cells in the numbered notebooks, each
  followed by `raise NotImplementedError`.
- Module names use underscores because a Python module name cannot contain a
  dot: `Ex_4_core.py`, imported as `Ex_4_core`.
- Every `TODO` is preceded by enough prose that the notebook works for a student
  who missed the lecture. Part 1 exercises are deliberately more discursive than
  the Part 2 ones.
- Seeds are set everywhere, and notebook 02 makes the point that a single seed
  is not evidence by training ten of them.
- British spelling throughout.

## The two marked questions

From the two lecture decks, and the report in notebook 06 is built around them:

> **L4.1 slide 22.** At equal parameter count, which did better — deeper or
> wider? And why do you think so?

> **L4.2 slide 22.** Which model would you deploy, and what would have to be true
> for that to be right?

Notebook 06 also asks the four questions from L3.2 slide 2 about the model the
student chose to deploy, and includes a worked example answer so that the
expected standard is visible rather than guessed.

## Depends on

- **L4.1** slides 9 (the fix), 12 (why Part 2 uses tanh), 14–15 (kinks), 16
  (universal approximation and its five caveats), 21 (failure modes) and 22
  (this exercise).
- **L4.2** slides 3–6 (depth and regions), 10 (capacity as a dial), 11–13 (the
  eleven points), 14 (train/validation/test), 17–19 (regularisation) and 22
  (this exercise).
- **Ex_03**, whose extrapolation plot is explained again in notebook 03 section 4
  once the piecewise-linear picture is available.

## Expected runtime

CPU only. Notebooks 00 to 02 take well under a minute of compute each. Notebook 03
trains twenty-eight small networks and takes about a minute; notebook 04 runs
fifteen and takes two or three; notebook 05 runs nine and takes about one. The
difficulty is conceptual, not computational.
