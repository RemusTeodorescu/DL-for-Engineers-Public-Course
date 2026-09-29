# Course Policies

Thirteen rules for the lectures and exercises of *Deep Learning for Engineering*.
They replace the 35 separate slide policies (P1–P12, D1–D18, E1–E9) that grew
out of the reviews in September 2026; each old policy now sits under one of the
first eight, and its check still runs. C9 was added the same day, C10 to C12 on
29 September.

Agreed 28 September 2026, after the Part 1 student evaluation asked for better
lecture structure and time management, one consistent notation, and fewer
errors in the exercises.

| | Policy | Checked by |
|---|---|---|
| C1 | Lecture structure and time | `tools/deck/slide_policies.py` |
| C2 | One concept per slide, stated first | `tools/deck/slide_policies.py` |
| C3 | One notation for the whole course | `tools/deck/check_notation.py` |
| C4 | Self-contained references | `tools/deck/slide_policies.py`, `tools/deck/check_xrefs.py`, `tools/exercises/check_exercise_policy.py` |
| C5 | Plain, moderate language | `tools/deck/slide_policies.py` |
| C6 | Visual consistency | `tools/deck/slide_policies.py`, `tools/deck/figure_policy.py` |
| C7 | Layout passes the checkers | `tools/deck/slide_policies.py` and the fit/layout checkers |
| C8 | Exercises: few notebooks, only the essential ones | `tools/exercises/check_exercise_policy.py` |
| C9 | Notebooks: every deep-learning line carries a comment | `tools/exercises/comment_dl_lines.py` |
| C10 | Part 2: every set ends with a mini project proposal | `tools/exercises/check_exercise_policy.py` |
| C11 | Part 2: one notebook per set | `tools/exercises/check_exercise_policy.py` |
| C12 | Numbers with at most two decimals | by review |
| C13 | Part 2: a forward-problem notebook tells one story | by review |

`python3 tools/deck/slide_policies.py --list` prints the twelve with the old
policies under each; `--full` adds the review history behind every one.

## C1 · Lecture structure and time

- At most 20 slides per lecture. A longer deck declares `@slide-budget N — reason`.
- The order is fixed: title, Learning Objectives, the sections, Advanced Topics,
  Exercise, Key Takeaways, Questions.
- The Learning Objectives are the numbered sections, each with its reading (UDL chapter).
- Only the first lecture of a block opens with a Recap: the previous lecture's
  ten questions, verbatim.
- Two talk stops per lecture, each on a slide that defines a new concept: a
  general question first, then a question about a figure on the same slide.
- The Questions slide has ten exam questions, each with its reference.
- Every Key Takeaway answers an objective and is reproduced in a notebook.

*Replaces D3, D4, D6, D7, D8, D10, D13, D15, D16, P7, E4, E6.*

## C2 · One concept per slide, stated first

- The title names the concept in at most six plain words.
- A definition box follows directly under the title, two or three plain
  sentences simplified from UDL or GBC; the explanation follows it.
- A formula is followed by its PyTorch code.
- An architecture is defined before any metaphor is used for it.

*Replaces D1, D2, D9, D17, E1, E3, E5.*

## C3 · One notation for the whole course

- `docs/NOTATION.md` is binding for slides, speaker notes, notebooks and the
  variable names in code.
- Where UDL or GBC write another symbol, the course symbol is used and the
  notation table lists the difference.
- A new symbol is added to the table before it is used.

*New.*

## C4 · Self-contained references

- No slide numbers on slides, in speaker notes or in notebooks.
- Another lecture is named on at most two slides of a deck.
- An exercise points to a lecture by its code and the slide title or an
  equation, for example "L4.2, Backpropagation".

*Replaces P4, D5, and extends them to the exercises.*

## C5 · Plain, moderate language

- No superlatives, no absolutes, no dismissive asides.
- None of the words on the struck-word list (`JARGON` in `slide_policies.py`).
- Takeaways and questions say what is, what follows and what to check.

*Replaces P5, D18, E9.*

## C6 · Visual consistency

- A colour carries one meaning: cyan structure, amber attention, orange the
  trap, green the result, purple the subtlety. Code boxes are lilac.
- Left column: the what and the why. Right column, framed: the how, the cost
  and the result.
- A figure takes the idea from a book, never the picture (`docs/FIGURE_POLICY.md`).
- The title slide: byline under the title, the overview as the figure's caption.

*Replaces D11, D12, E2, E8.*

## C7 · Layout passes the checkers

No overlapping blocks, nothing past the content floor, nothing below 11 pt,
the course type settings declared. Fully automatic; `preflight.py`,
`check_fit.py`, `check_layout.py`, `check_figtext.py` and `validate_pptx.py`
run with it.

*Replaces P1, P2, P3, P6, P8, P9, P10, P11, P12.*

## C8 · Exercises: few notebooks, only the essential ones

- At most four working notebooks per set, plus the setup notebook (`00`) and
  the report notebook.
- A set does not cover the whole lecture. It takes the two or three ideas a
  student must be able to apply, preferably those Part 2 builds on; the rest
  stays in the lecture or on its Advanced Topics slide.
- Each working notebook answers one question in about 30 minutes. The
  Exercise slide lists that question; the report asks one closing question
  across the notebooks.
- Every working notebook has a light version, generated from the solved one.
- Wherever a notebook sends the reader to one that has two forms, it offers
  both, in the same words in both forms, so a reader in either can choose:

  > **Open notebook 02 as**
  > - **the exercise**, where you write the missing lines, or
  > - **the light version**, with every cell written out.

  A link into the report says the report has one form, so nobody looks for
  a light version that does not exist.
- The two forms read the same. A light version carries its exercise's
  exam-question tags, in the same order: when a lecture's Questions slide
  changes, both forms change. Sections are numbered once, in order; a TODO
  keeps its label inside the heading (`## 4 · TODO 1 — bad data`), and the
  light version says "section 4" where the exercise says "TODO 1".
  `tools/exercises/two_forms_handover.py --apply` writes both;
  `check_exercise_policy.py` fails a link to one form without the other.
- Before release a set runs from a fresh Colab runtime (`tools/light/solve_run.py`)
  and is reviewed once by a person other than its author.

*Replaces D14, E7. The limit, the scope rule, the release check and the two-form handover (28 September 2026) are new.*

## C9 · Notebooks: every deep-learning line carries a comment

- A code line that calls a deep-learning function carries a short comment
  saying what it does: a layer or activation, a loss, an optimiser, the
  training-step calls (`zero_grad`, `backward`, `step`), autograd and gradient
  tracking (`torch.autograd.grad`, `requires_grad`, `no_grad`, `detach`),
  training and evaluation mode, data loaders, saved weights, seeds, and the
  course libraries' own deep-learning helpers (`train_two_stage`, `grad`,
  `MLP`, the collocation samplers).
- Ordinary code gets no comment: loops, prints, plots, NumPy, tensor arithmetic
  and tensor creation. The comments are there to point at the lines that matter.
- The comment sits at the end of the line, or on the line directly above it
  when the line is long.
- `python3 tools/exercises/comment_dl_lines.py` lists the lines without one;
  `--apply` writes a first version, to be read and improved by hand.

*New, 28 September 2026.*

## C10 · Part 2: every set ends with a mini project proposal

- Every Part 2 exercise set (Ex07.1 to Ex12.2) ends with a section *Mini
  project proposal*: the last numbered section of the report notebook, and the
  same two proposals in the set's README. A set with no report notebook
  carries it in its README alone.
- Two proposals per set - or one per problem, where a set has more than two
  (Ex07.2: the die, the panel and the slot's Helmholtz equation) - each grown
  out of a problem the set has developed: a harder version of it, or its
  inverse. Ex07.1's are the slot
  with every wire and its insulation resolved (a domain-decomposed PINN), and
  the insulation's aging recovered from a few sensors (an inverse PINN).
- Each student chooses one mini project from all the Part 2 sets and solves it
  individually, in one month. The dates and the hand-in are set in L13.
- The two are named after the set, **MP7.1A** and **MP7.1B**, MP8.2A and so on.
- Each proposal is specific, with numbers, under these headings: **the
  problem** (every dimension, material value, range and noise level, in a
  table where there are several); why a plain PINN is not enough, where that is
  the point; **what you build** with deep learning; **the ground truth you
  get** (the solver, the grid, the file and its contents, the checks it
  passes); **a worked example** - the ground truth's first case, computed, with
  its numbers; **requirements** - how accurate the prediction must be, or how
  fast, in numbers; **what you hand in**.
- The ground truth is the course's, never the student's: the month goes into
  the deep learning. Its script is written, and its worked example run, when
  the proposal is written; the full data set is built when a student chooses
  the project. Both follow the rules below. Ex07.1's is
  `tools/miniprojects/ex071_truth.py`.

### Building a mini project's ground truth

1. **A classical solver, never a network.** Finite differences or volumes,
   finite elements, or an ODE integrator (`scipy`). Extend the set's own
   `problem.py` rather than copying it, so the exercise and the project share
   one set of physics and constants.
2. **One script per set**, `tools/miniprojects/<set>_truth.py`, covering both projects,
   seeded and deterministic, that writes the data file and prints the checks
   below. Say how long it runs; prefer under half an hour on a laptop CPU.
3. **Shown to have converged.** At least three grids, each twice as fine as
   the last. Print the quantity the student will be judged on (a hot spot, a
   flux, a trajectory) on each, show the error falling at the method's order,
   and use Richardson extrapolation where it applies. The ground truth must be
   at least ten times - better a hundred times - more accurate than what a
   good student's network can reach, or the error measured is the reference's.
4. **Checked against physics.** A conservation balance (heat made = heat
   leaving through the walls, mass in = mass out), any symmetry the problem
   has, and a limiting case that reproduces the exercise: the resolved slot
   with one conductivity everywhere must give Ex07.1's averaged field.
5. **Every value sourced.** Each material property and dimension is either
   taken from a named reference or marked *assumed* in the script and in the
   file. No invented hardware figures.
6. **One `.npz` file, self-describing.** Coordinates, fields, every physical
   parameter with its unit, the material or region map, the currents or other
   inputs, the convergence table, the seed and the git commit of the script,
   and a `README` string naming every key. Keep it under 50 MB.
7. **Inverse problems split what is observed from what is hidden.** The
   student fits the observations (sensor positions, readings, the noise level
   and its seed) and scores against the truth afterwards, in a separate file.
   Give cases of rising difficulty: fewer sensors, more noise, a smaller
   feature to find.
8. **Generalisation is tested outside the training range.** When the network
   is meant to work across a parameter (a current, a Reynolds number), keep a
   held-out range the student does not train on.
9. **Run by a second person** before the student gets it, like every set (C8).

*New, 29 September 2026.*

## C11 · Part 2: one notebook per set

- Every Part 2 exercise set is **one notebook**, in its two forms: the
  exercise, with its TODO cells, and `_light`, written out. No environment
  check notebook, no separate report notebook, no results passed between
  notebooks.
- The sections follow the lecture's story, in this order: the problem and its
  physics; the checks the method relies on (autograd, where the set uses it);
  the reference the methods are scored against — an exact solution where the
  problem has one; the classical solution and its mesh; the PINN, with its
  collocation points and size set against that mesh; the answers compared;
  the report; the mini project proposal (C10), last.
- The reference is named for what it is. A finite-difference answer is one of
  the methods, not a "ground truth", unless nothing better exists — and then
  the notebook says how its accuracy was shown.
- Every figure's drawing code is in the cell that shows it, unless it is a
  plot the whole course shares (`plot_field`).
- No jargon in the headings or the text: say what a section does.
- A set may add **one optional supplement**, a second notebook named
  `*_supplement_*` with its light version: material from the lecture the main
  notebook leaves out. It stands alone (its own setup, problem and questions),
  is marked *(Optional)* in its title, and is not required for the report.
  Ex07.2's supplement to the PDE recap (Helmholtz in the stator slot) is the
  first.
- A **benchmark set** - one that works through several equations side by side
  - is one notebook **per problem**, each standing alone with its own report
  and its own mini project. Ex07.2, *Benchmark PDEs*, is the one: the die
  (parabolic), the stator slot (elliptic) and the struck panel (hyperbolic).

*New, 29 September 2026. Ex07.1 is the first set under it; the others follow
as they are revised.*

## C12 · Numbers with at most two decimals

- A temperature or any other physical value, printed by a notebook or written
  on a slide, carries at most two decimals: 18.74 K, not 18.7403 K. More
  decimals are noise to the reader and invite comparing digits that mean
  nothing.
- A value smaller than 0.01 is written in powers of ten with a two-decimal
  mantissa (`3.02e-04 K`, $3\times10^{-4}$ K), not as 0.000302.
- Exceptions, where the extra digits are the point: a tolerance or a check
  (`1e-12`), a value quoted from a standard or a datasheet as it is given,
  and a learning rate or other hyperparameter.

*New, 29 September 2026.*

## C13 · Part 2: a forward-problem notebook tells one story

A notebook that solves a forward problem with a PINN - the field from the
equation and its conditions - follows the same seven steps as Ex07.1, in this
order, and nothing else:

1. **The problem** - what is being computed and why an engineer wants it.
2. **The data** - geometry, materials, loads, conditions, as numbers in a table.
3. **The physics** - the equation, its conditions, and what its type
   (elliptic, parabolic, hyperbolic) means for the answer.
4. **The reference and the classical solution** - the exact solution where
   there is one (otherwise a converged ground truth, C11), and finite
   differences on a mesh chosen for an agreed accuracy.
5. **The PINN** - one network, one way of meeting the conditions, trained with
   the course's recipe (Adam, then L-BFGS).
6. **The comparison** - the PINN and finite differences against the reference
   on **accuracy**, and the PINN against finite differences on **computing
   time**: per solve, and training.
7. **The discussion** - what the numbers say, in a few plain bullets.

Then the report and the mini project (C10). What stays out:

- **No alternatives explored.** One mesh study, one network, one way of
  enforcing the conditions. Soft against hard, a sweep of widths, a second
  optimiser, a demonstration that a concept is true - those are for the
  lecture, a supplement or a mini project, not the forward notebook.
- **Lean.** About twenty cells; at most two TODO cells, each on the physics or
  the network (the residual, the trial function), never on plotting; three or
  four report questions and the one that concludes.
- Every "What you should see" gives the numbers the notebook actually prints.

*New, 29 September 2026. Ex07.1 was written this way; Ex07.2's three notebooks
are the first built to it.*
