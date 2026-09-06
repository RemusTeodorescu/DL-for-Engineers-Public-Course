# Ex06 — Training Lab

**Paired with lecture block L6 · Part 1 · the last exercise set before the physics arrives**

Where a loss comes from, how you get to the bottom of it, and what happens to a
trained model afterwards. Notebooks 01 and 02 belong to **L6.1 · Loss Functions
and Gradients**; 03 and 04 belong to **L6.2 · Post-Training and Reinforcement
Learning**.

## Goals

By the end of this exercise set you can

1. derive the mean squared error from a Gaussian noise assumption and the cross
   entropy from a categorical one, and verify both numerically;
2. change the noise assumption and watch a single bad reading move a fitted
   line — the point being that a loss is a statement about expected errors;
3. compare SGD, momentum, Adam and L-BFGS on one problem, and measure the
   **Adam-then-L-BFGS handoff** that every exercise in Part 2 uses;
4. transfer a trained classifier to a second machine with ten labels per class,
   and find the learning rate at which fine-tuning destroys what it was given;
5. quantise a deployed model and report size, latency and accuracy **together**,
   and say when a narrower float model would have been the better lever.

## State

**Built.** Six notebooks, `Ex_6_core.py` complete.

Not yet run against real PyTorch — see the warning at the bottom of this file.

## The notebooks

Run in order; later notebooks load results saved by earlier ones.

```
Ex06_00_environment_check.ipynb        # the tools, the four datasets, four optimisers
Ex06_01_loss_functions.ipynb           # a loss is a noise assumption written down
Ex06_02_optimiser_comparison.ipynb     # SGD, momentum, Adam, L-BFGS, and the handoff
Ex06_03_transfer_and_fine_tuning.ipynb # a second machine and ten labels per class
Ex06_04_quantisation.ipynb             # making it fit on the device
Ex06_05_report.ipynb                   # the report
```

### What each one is for

**01 · Where the loss comes from.** Derive the mean squared error from a
Gaussian assumption and the cross entropy from a categorical one, then change
the assumption and watch the fitted line move. Opens by pointing back at
Ex_03 notebook 01 rather than refitting the same line again — the fitting was
done there; the question here is *why that objective*.

**02 · Four optimisers, one problem.** The same network and the same noiseless
target, four update rules. Sweeps the learning rate until both failure
directions appear, then meets L-BFGS's closure interface and measures the
**Adam-then-L-BFGS handoff** that every exercise from Ex_07 onwards uses.

**03 · A second machine, and ten labels each.** A classifier trained on machine
A meets machine B, whose sensor has a different gain, mounting and orientation.
Train from scratch, retrain the last layer, or fine-tune everything — on thirty
labels. Finds the learning rate at which fine-tuning destroys what it was
given, and the label budget at which transfer stops paying. This is L11's
pretrained ResNet-18 argument at a size you can see.

**04 · Making it fit on the device.** Post-training dynamic quantisation.
Measures size, latency and accuracy before and after, then asks the question
that keeps the result honest: would a narrower float model have been smaller
still at the same accuracy?

**05 · The report.** Six questions, tight word limits, checked before assembly.

## Conventions

Same as every exercise set in this course:

- Self-contained folder. Requires `torch`, `numpy`, `matplotlib` — all
  preinstalled on Google Colab. No GPU needed.
- Notebook `00` is a read-only environment check. Run it first.
- `Ex_6_core.py` is complete and is **not** to be rewritten by students. The
  work is in the `# TODO:` cells, each followed by `raise NotImplementedError`.
- Module names use underscores because a Python module name cannot contain a
  dot: `Ex_6_core.py`, imported as `Ex_6_core`.
- **Colab needs the whole folder**, not just one notebook — the imports expect
  `Ex_6_core.py` beside them.

## The four datasets

All generated on your machine; nothing is downloaded.

| | |
|---|---|
| calibration line | 40 load-cell readings, Gaussian noise of known σ |
| vibration classes | 360 signatures in 2 features — balanced, imbalance, bearing fault |
| damped response | 200 **noiseless** samples of `exp(-0.9x) sin(4x)` |
| fatigue | 20 training points, 60 held out |
| machine B | the vibration classes again, through a different sensor |

Two of these are deliberately unusual and the notebooks say so aloud. The
damped response carries **no noise**, because when you compare optimisers,
noise puts every optimiser on the same floor and the comparison measures
nothing. The vibration classes deliberately **overlap**, because a separable
problem lets every loss reach perfect accuracy and notebook 01 is about what
the losses disagree on after accuracy has stopped distinguishing them.

## What this set does not repeat

Deliberate boundaries, so that a student does not meet the same exercise twice:

- **Overfitting, weight decay and early stopping** are **Ex_04 notebook 05**,
  not here. They moved to L4.2 in the course restructure; Ex06's original plan
  listed them and was out of date.
- **Writing a training loop** is Ex_02 notebook 03. Notebook 02 here assumes
  you can write one and asks what to put in the optimiser slot.
- **Least squares by hand** is Ex_03 notebook 01. Notebook 01 here derives why
  that objective, and verifies rather than refits.

## Expected runtime

CPU only, a few minutes per notebook. Notebook 02's learning-rate sweep is the
longest at around two minutes. The difficulty is conceptual, not
computational.

## Before this is assigned

`Ex_6_core.py` and notebooks 02 to 05 were written in an environment where
**PyTorch could not be installed**. The NumPy half of the module — every
dataset, every printed value in notebook 00 — has been verified line by line
against the notebooks' stated expectations. The torch half (`MLP`, `to_tensor`,
`model_size_bytes`, `time_forward`) is written but unexecuted.

Run notebook 00 once end to end on a machine with real PyTorch before this goes
to students. See `docs/REPO_NOTES_PART1.md` §9.
