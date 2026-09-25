# Ex06 — Training Lab

**Paired with lecture block L6 · Part 1 · the last exercise set before the physics arrives**

Where a loss comes from, how you get to the bottom of it, and what happens to a
trained model afterwards. Notebooks 01 and 02 belong to **L6.1 · Loss Functions
and Gradients**; 03 and 04 belong to **L6.2 · Training Philosophies**. The
quantisation notebook moved to Ex11.1 on 22 September 2026, with deployment.

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
5. train a battery to trade by reinforcement learning, measure it against the
   exact optimum of a linear program, and imitate the optimiser with a network
   that runs a hundred times faster.

## State

**Built.** Seven notebooks, `Ex_6_core.py` complete.

Notebook 00 has been run end to end against real PyTorch — see the note at the bottom of this file.

## The notebooks

Run in order; later notebooks load results saved by earlier ones.

```
Ex06_00_environment_check.ipynb        # the tools, three datasets, the L-BFGS closure
Ex06_01_loss_functions.ipynb           # a loss is a noise assumption written down
Ex06_02_optimiser_comparison.ipynb     # Adam, L-BFGS, and the handoff between them
Ex06_03_transfer_and_fine_tuning.ipynb # a second machine and ten labels per class
Ex06_04_battery_arbitrage.ipynb        # a battery that learns to trade
Ex06_05_report.ipynb                   # the report
```

### What each one is for

**01 · Where the loss comes from.** Derive the mean squared error from a
Gaussian assumption and the cross entropy from a categorical one, then change
the assumption and watch the fitted line move. Verifies rather than refits:
least squares by hand is assumed, and the question here is *why that
objective*.

**02 · Adam, then L-BFGS.** The same network and the same noiseless target,
trained with the two optimisers Part 2 uses: Adam alone, L-BFGS alone with
its closure interface, and the **Adam-then-L-BFGS handoff** that every
exercise in Part 2 uses, at two handoff points. Plain gradient descent, random
batches and momentum are taught in L6.1 and not repeated here.

**03 · A second machine, and ten labels each.** A classifier trained on machine
A meets machine B, whose sensor has a different gain, mounting and orientation.
Train from scratch, retrain the last layer, or fine-tune everything — on thirty
labels. Finds the learning rate at which fine-tuning destroys what it was
given, and the label budget at which transfer stops paying. This is L11's
pretrained ResNet-18 argument at a size you can see.

**04 · A battery that learns to trade.** A 200 kWh battery on the day-ahead
market. REINFORCE learns a policy from the reward alone and reaches about four
fifths of the optimum that a linear program computes in milliseconds; a network
trained to imitate the linear program reaches nineteen twentieths and runs
over a hundred times faster. Paired with L6.2: reinforcement learning, where it
does not fit, and approximate MPC.

**05 · The report.** Six questions, tight word limits, checked before assembly,
then every notebook's questions and one question to conclude.

## Conventions

Same as every exercise set in this course:

- Self-contained folder. Requires `torch`, `numpy`, `matplotlib`, and `scipy`
  for notebook 04's linear program — all preinstalled on Google Colab. No GPU
  needed.
- Notebook `00` is a read-only environment check. Run it first.
- `Ex_6_core.py` is complete and is **not** to be rewritten by students. The
  work is in the `# TODO:` cells, each followed by `raise NotImplementedError`.
- Module names use underscores because a Python module name cannot contain a
  dot: `Ex_6_core.py`, imported as `Ex_6_core`.
- **On Colab nothing needs uploading**: each notebook's first code cell fetches
  `Ex_6_core.py` from the public course repository.

## The six datasets

All generated on your machine; nothing is downloaded.

| | |
|---|---|
| calibration line | 40 load-cell readings, Gaussian noise of known σ |
| vibration classes | 360 signatures in 2 features — balanced, imbalance, bearing fault |
| damped response | 200 **noiseless** samples of `exp(-0.9x) sin(4x)` |
| fatigue | 20 training points, 60 held out — in `Ex_6_core.py` for optional use; no notebook uses it |
| machine B | the vibration classes again, through a different sensor |
| day-ahead prices | 24-hour price curves with two peaks and a solar dip — notebook 04 |

Two of these are deliberately unusual and the notebooks say so aloud. The
damped response carries **no noise**, because when you compare optimisers,
noise puts every optimiser on the same floor and the comparison measures
nothing. The vibration classes deliberately **overlap**, because a separable
problem lets every loss reach perfect accuracy and notebook 01 is about what
the losses disagree on after accuracy has stopped distinguishing them.

## What this set does not cover

Deliberate boundaries:

- **Overfitting, weight decay and early stopping** belong to L4.2 and are not
  treated here.
- **Writing a training loop** is assumed. Notebook 02 asks what to put in the
  optimiser slot.
- **Least squares by hand** is assumed. Notebook 01 derives why that
  objective, and verifies rather than refits.

## Expected runtime

CPU only, under a minute of compute per notebook (22 September 2026, the light
notebooks run end to end by `tools/exercises/run_light.py`: 01 8 s, 02 33 s,
03 12 s, 04 21 s). The difficulty is conceptual, not computational.

## Before this is assigned

`Ex_6_core.py` and notebooks 02 to 04 were written in an environment where
**PyTorch could not be installed**. On 19 September 2026 notebook 00 was run end
to end with PyTorch 2.14 (CPU): every cell runs, the printed datasets match the
notes, and `MLP` and `to_tensor` work. The run showed that L-BFGS without a line
search stalls on the smoke test (loss 0.061 against Adam's 0.010), so notebook 00
now uses `line_search_fn="strong_wolfe"`, as `pinn_core.py` does in Part 2
(loss 0.00075). Notebook 02 still builds L-BFGS without a line search.
Since 22 September every light notebook, 01 to 04, runs end to end, and the
report loads what they save.

Run notebooks 01 to 04 once, with the TODO cells completed, before this goes to
students. See `docs/REPO_NOTES_PART1.md` §9.

## Results between notebooks on Colab

Later notebooks read `.npz` / `.pkl` files that earlier ones write into
`Ex06_outputs/`. On Google Colab every notebook runs on its own temporary machine,
so those files would not survive from one notebook to the next. Notebooks
01 to 04 therefore start with an `outputs-cell` that calls `keep_outputs()` from
`Ex_6_core.py`: on Colab it mounts the student's Google Drive and moves the results
folder to `MyDrive/DL4Eng/Ex06_outputs`. If the student declines the Drive request or
has no Google account, `saved()` downloads each result file when it is written
and `needed()` asks for the files to be uploaded before they are read. Locally
the cell does nothing. The report notebook writes its `.md` and `.pdf` into the
same folder.
