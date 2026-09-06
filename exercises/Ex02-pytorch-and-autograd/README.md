# Ex02 — PyTorch and Autograd

**Paired with lecture block L2 · Part 1**

Differentiate an analytic function with respect to its INPUT and check against
the hand-derived derivative. This is the seed of every Part 2 exercise - do not
skip notebook 02.

## Goals

By the end of this exercise set you can

1. create and manipulate tensors, and say what dtype a tensor carries and which
   device it lives on;
2. differentiate a function with respect to its **input** with automatic
   differentiation, and check the result against a derivative you took by hand;
3. take a second derivative, and explain what `create_graph` is for and what
   happens without it;
4. write the five-line training loop from memory, and say precisely what
   `zero_grad` prevents;
5. state why ReLU's second derivative decides the activation choice for every
   network in Part 2.

## State

**Written.** One module and four notebooks. `Ex_2_core.py` compiles and its
public interface is fixed; the notebooks were checked cell by cell for
undefined names against that interface. See *Verification* below for what was
and was not executed.

## Notebooks

Run in order; later notebooks assume the vocabulary of earlier ones.

```
Ex02_00_environment_check.ipynb   read-only. Versions, seeds, the six-line
                                  autograd smoke test, the log-axis convention.
Ex02_01_tensors.ipynb             5 TODOs. dtype and device, the (n, 1) column
                                  convention, shapes, a linear layer written by
                                  hand, requires_grad and the three ways to lose
                                  the graph.
Ex02_02_autograd_by_hand.ipynb    7 TODOs. THE notebook. See below.
Ex02_03_training_loop.ipynb       3 TODOs. Forward, loss, zero_grad, backward,
                                  step; a curve fitted; the learning rate; and
                                  what happens outside the training data.
```

## Notebook 02, in detail

It is the most important notebook in Part 1, and it says so in its first line.
The argument it builds, in order:

1. **`torch.autograd.grad` on a scalar**, checked against a derivative written
   by hand.
2. **`grad_outputs`** — what a vector–Jacobian product is, and why a vector of
   ones is the right choice for an elementwise map and not merely the
   conventional one.
3. **An analytic function on a grid.** u(x) = e^(-x) sin(3x), differentiated by
   autograd and by the student, compared at 101 points.
4. **Second derivatives**, and what `create_graph=True` buys — demonstrated by
   removing it and reading the exception.
5. **Autograd against finite differences** over eleven decades of step size.
   Produces the check-mark error curve: truncation on one side, floating-point
   cancellation on the other, a best case near 1e-11, against an autograd line
   flat at machine epsilon. Run in `float64`, because the question is about
   numerical precision.
6. **A residual.** u'' + k²u evaluated on a solution (zero) and on a
   non-solution (not zero). The first physics-informed object in the course,
   and the notebook says plainly that this is Ex_7.1 with a different equation.
7. **The derivative of a network with respect to its input**, verified against
   a central difference of the network itself.
8. **ReLU against tanh.** Identical networks, identical seed. The tanh second
   derivative is of order 1; the ReLU second derivative is exactly zero,
   everywhere, and the notebook explains both why (piecewise linear ⇒ piecewise
   constant first derivative ⇒ zero second derivative) and what it costs (a
   second-order residual gets no gradient signal at all). **This is why every
   network in L7–L12 uses tanh**, and it closes the loop on the L4.1 bridge
   slide.

It ends with the eight-line skeleton of every Part 2 exercise, with the three
lines this notebook taught marked as such.

## Module

`Ex_2_core.py` carries the long **HOW TO RUN Ex_2** docstring in the same shape
as the Part 2 core libraries, including a troubleshooting section written around
the exceptions students will actually hit — `element 0 of tensors does not
require grad`, `Trying to backward through the graph a second time`, a second
derivative that is exactly zero, a loss that will not fall.

The two functions that matter are deliberately named and shaped exactly as in
the Part 2 core libraries:

```python
grad(y, x)      # dy/dx w.r.t. the input, with create_graph=True
d2(y, x)        # grad(grad(y, x), x)
```

so that the code a student writes in week two is the code they will read in
week seven.

**One deviation, deliberate.** `grad` passes `allow_unused=True` and substitutes
zeros for a missing gradient. Without it, the second derivative of a ReLU
network raises *One of the differentiated Tensors appears to not have been used
in the graph* rather than returning the mathematically correct zero — the ReLU
first derivative is a product of weight matrices and a constant mask, with no
path back to `x`. Notebook 02 §8 explains this and states the price: an
unexpected exact zero now needs checking rather than raising by itself.

Also in the module: `set_seed` (Python, NumPy and torch together), `as_input`
(the `(n, 1)` column convention), `describe_tensor`, `check` and `check_shape`
with a float32-appropriate default tolerance, `finite_difference` for the §5
comparison, `mlp` and `count_parameters`, `vibration_data` for notebook 03, and
`plot_loss`, which sets a logarithmic y-axis and is not optional.

## Conventions

Same as every Part 2 exercise set:

- Self-contained folder. Requires `torch`, `numpy`, `matplotlib` — all
  preinstalled on Google Colab. No GPU needed; `DEVICE` is fixed to the CPU
  because the largest network here has about 2,200 parameters.
- Notebook `00` is a read-only environment check. Run it first.
- Modules (`Ex_2_core.py`) are complete and are **not** to be rewritten by
  students. The work is in `# TODO:` cells in the numbered notebooks, each
  followed by `raise NotImplementedError`.
- Module names use underscores because a Python module name cannot contain a
  dot: `Ex_2_core.py`, imported as `Ex_2_core`.
- Every cell that produces output is followed by a **"What you should see"**
  note, including the tolerance used and why it is what it is — 1e-4 for a
  first derivative in float32, 1e-3 for a second, 5e-3 against a finite
  difference. Tolerances are set by the arithmetic, not by optimism, and the
  notebooks say so each time.

## Deliberately unlike Part 2

Every TODO is preceded by several paragraphs explaining what is being asked,
why, and what a correct answer looks like. L2 is recorded rather than live, so a
student who missed it has no room to ask a question; the notebooks are written
to work on their own. The prose thins out from Ex03 and is gone by Ex07.

## Verification

The notebook JSON is valid `nbformat` 4 and every code cell parses. Reference
answers were written for all 15 TODO cells and a static pass confirmed that no
cell references a name that has not been defined by an earlier cell or by the
module.

`torch` could not be installed in the environment these files were authored in,
so the Ex02 notebooks have **not** been executed end to end. Ex01 has. Before
this set is given to students, run all four notebooks once with a solution
copy — the numbers quoted in the "What you should see" notes are reasoned
estimates, not transcripts.

## Expected runtime

CPU only. Notebooks 00 to 02 do no training at all and are instant. Notebook 03
runs six short trainings totalling well under a minute. Budget half an hour for
the set, nearly all of it reading.
