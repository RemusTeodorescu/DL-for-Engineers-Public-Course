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
  four questions with their answers, verbatim.
- **The Recap names the lecture it recaps (D24, Part 2, 2 October 2026).** Its
  title is `Recap of L7.2 Fundamental PDEs`: the lecture's number, as the
  slide's source line gives it, and its name, as its own title slide prints
  it. `slide_policies.py` reads the name from that lecture's generator, so a
  renamed lecture is reported on the next Recap until it is copied across.
  L7.1, L8.1 and L9.1 carry it; L10.1, L11.1 and L12.1 take it with their
  Recap slide.
- Two talk stops per lecture, each on a slide that defines a new concept: a
  general question first, then a question about a figure on the same slide.
- **The Questions slide (D21, 1 October 2026).** Four exam questions, two
  for each of the lecture's two main topics, headed `1 – 2 · TOPIC` and
  `3 – 4 · TOPIC`. Each is a `[question, answer, reference]` triple:
  - the **question** in bold, at most 125 characters, the most general one the
    topic can be asked - about a concept, not a detail;
  - the **answer** beside it, rich enough to fill the slide (60 to 520
    characters, three to five sentences), written as it should be said at the
    oral examination;
  - the **reference** to read, in italics after the answer: the book chapter
    and, where there is one, the notebook that works through it.
  It was ten questions with references and no answers. Students found 24
  lectures of ten too many to prepare, and a question with nothing beside it
  read as a second learning objective. Every deck from L1.1 to L12.2 is on
  the format; the next block's Recap copies the four verbatim (D16), and the
  notebooks' question tags (`→ L8.1 Q3`) run from Q1 to Q4.
  `slide_policies.py` checks the count, the halves, the lengths, the
  references, that the answers are drawn, and that each box holds its text.
- **What the four ask (10 October 2026, L10.1 the pattern).** Part 1 asks
  the general deep-learning questions. A Part 2 lecture's four ask **how deep
  learning solves that lecture's physics, with that physics' own
  difficulties**: what the network takes and returns, what is built in rather
  than learned, what the training must be told that the equation does not
  enforce by itself, what a parameter estimated with a network needs from
  the measurements, when the network pays against the classical solver. They
  are qualitative - the concept, not a formula - and in explicit language
  (C5, D25): "the concentration of lithium inside the particle along the
  radius", never "the lithium". The physics is the setting, not the subject;
  the answers may carry a number where it makes the point.
- Every Key Takeaway answers an objective and is reproduced in a notebook.

*Replaces D3, D4, D6, D7, D8, D10, D13, D15, D16, D21, D24, P7, E4, E6.*

## C2 · One concept per slide, stated first

- The title names the concept in at most six plain words.
- A definition box follows directly under the title, two or three plain
  sentences simplified from UDL or GBC; the explanation follows it.
- A formula is followed by its PyTorch code (Part 1).
- **In Part 2 a formula is followed by a figure of its concept, and by code
  only where the code is special (D23, 1 October 2026).** When an equation
  introduces a concept, the lines that compute it are usually the equation
  retyped: `q = h * (T_s - T_inf)` under Newton's law of cooling. That is
  ordinary arithmetic, it teaches nothing about deep learning, and it takes
  the room a picture could have. So in L7.1 to L12.2 (L13 is a special lecture
  and exempt):
  - a code box stays only if it calls something specific to deep learning:
    automatic differentiation (`grad`, `autograd`, `backward`,
    `requires_grad`, `detach`), an optimiser (`optim`, `Adam`, L-BFGS,
    `step`), or `torch.nn` (a module, `MLP`, a `Parameter`, an activation).
    Arithmetic, a mean, a square, a loop and a call of the network are not
    special;
  - otherwise the box is replaced by a **figure of the concept**, in the same
    place and the same footprint, drawn with the deck's own shapes
    (`D.figure` and the primitives beside it in `tools/deck/diagrams.js`);
  - the figure answers one question: *if you had to represent this concept by
    a drawing, what would you draw?* A hot surface with a stream over it for
    convection; a membrane sagging under a load for Poisson's equation; a
    factor that is zero on the wall, times any network, for a hard boundary.
    It is a picture of the idea, usually very abstract, not a plot of data
    and not a second table;
  - it carries a few words, at the figure size, and no equation of its own;
  - the narration says what the drawing shows, where it used to say "in code
    that is three lines";
  - a slide that has good reason to keep plain code writes
    `@keep-code — reason` in the generator.
  - a deck that had no code at all (L10.1 to L12.2) gets the figure all the
    same: every slide whose equation introduces a concept carries one. Where
    a slide had no room, the figure takes the place of a secondary block of
    text and that text moves to the narration;
  - a figure drawn by hand from literal shapes is declared in the generator,
    `/* @figure — what it draws */`, so that the check can see it.
- **A formula names its symbols (D22, 1 October 2026).** Under the equation
  there is a legend: every symbol in it, named in the equation's own typeface,
  with its unit where it has one ("$h$ the heat transfer coefficient,
  $T_\infty$ the fluid temperature"). Equations stacked on one slide may share
  one legend. A quantity that is new in the lecture is also explained in
  words on the slide where it first appears - on L8.1's boundary conditions
  that is the Robin condition, the heat transfer coefficient and the Biot
  number. A slide that names its symbols another way writes
  `@no-legend — reason` in the generator. Every deck is under it since
  1 October 2026.
- An architecture is defined before any metaphor is used for it.

*Replaces D1, D2, D9, D17, D22, E1, E3, E5; D23 is new.*

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
- **Explicit, never implicit (D25, 10 October 2026 — very important).** Clear,
  moderate academic language, with no jargon and no shorthand. Name the
  quantity, where it lives and what it depends on: not "learn the lithium
  inside a particle" but "learn the concentration of lithium inside the
  particle along the radius"; not "the pascals that drive the flow" but "the
  pressure drop driving the flow". A substance, a unit or a device never
  stands in for a quantity. It holds on the slides, in the narration and in
  the notebooks; `IMPLICIT` in `slide_policies.py` checks the shorthand struck
  so far on the slides, and the rest is read by review.

*Replaces P5, D18, E9; D25 added 10 October 2026.*

## C6 · Visual consistency

- A colour carries one meaning: cyan structure, amber attention, orange the
  trap, green the result, purple the subtlety. Code boxes are lilac.
- Left column: the what and the why. Right column, framed: the how, the cost
  and the result.
- A figure takes the idea from a book, never the picture (`docs/FIGURE_POLICY.md`).
- The title slide: byline under the title, the overview as the figure's caption.
- Maths in a definition box is set in the equation's font: every symbol and
  formula goes between dollars, `$u_{xx} + u_{yy} + f = 0$`, and `theme.js`
  `mathRuns()` sets it in mathematical italic with real subscripts and powers
  (D19, 29 September 2026). A lone variable is maths too: the field $u$, a
  factor $γ$, the $k$-th derivative (30 September).
- One maths font throughout: equations and every symbol in text, a table, a
  stack or a chain are set in Cambria Math; a symbol goes between dollars,
  never typed as a maths-italic character (D20, 30 September 2026).
- **Every symbol and formula on a slide is maths (D26, 10 October 2026).** In a
  definition, a bullet, a table, a caption, a figure label or a bare text box,
  a symbol or formula goes between dollars and is set in the equations' Cambria
  Math: `$x = 0$`, `$t_{end}$`, `$2π^2$`, never `x = 0` in the body face. Units
  with a superscript (m², mol/m³), powers of ten, chemical formulae and code are
  not maths. A label in spaced capitals cannot take maths and is reworded. The
  check reads the concept-figure functions above the first slide as well.

*Replaces D11, D12, D19, D20, E2, E8; D26 added 10 October 2026.*

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
- A supplement with nothing to fill in needs no light version (Ex12.2's
  comparison with ANDES).
- A **hardware set** keeps one hardware notebook beside its study notebook,
  named `*_on_the_car` (Ex11.1, Ex11.2: the JetRacer) or `*_on_the_board`
  (Ex12.2: the latency notebook L13 runs on the Thor). It is the procedure
  that needs the hardware, it has no light form, and it is not counted. The
  study notebook runs on Colab and says where its numbers are placeholders
  for ones the hardware notebook measures.

*New, 29 September 2026. It is the rule for **every** Part 2 set. Ex07.1 was
the first under it; Ex07.2 (three notebooks, a benchmark set), Ex08.1,
Ex08.2, Ex09.1 and Ex09.2 (1 October) follow, and Ex10.1 to Ex12.2 since
1 October: every Part 2 set holds under `check_exercise_policy.py`. On 5
October Ex08.1 and Ex08.2 became **one notebook for both lectures**, Ex08,
Heatsink Cooling of a Power Module: Part 1 (L8.1) and Part 2 (L8.2) in one
session, so that Part 2 loads nothing again. It is one notebook and its light
version, so it holds under C11 as it stands.*

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

- **A lecture that teaches an inverse problem gets one section for it**, after
  the comparison and before the discussion: the same network and residual with
  one more trainable number and one more loss term, written out, with no TODO.
  Ex08 recovers the fins' heat transfer coefficient that way, and Ex09.1 the
  coolant's viscosity.
  It is the lecture's second topic, not an alternative, so it does not break
  the rule above. Where the lecture's second topic is not an inverse problem
  the section takes that topic instead: Ex09.2 finds the inlet speed a design
  asks for.
- **An inverse problem divides its residual by the number it identifies.**
  Left undivided, the optimiser can shrink the trainable number and the
  residual with it: the diffusivity of the old Ex08.2 and the eddy viscosity
  of the old Ex09.2 both ran to zero that way, with a loss that looked healthy.
- **Where the problem has no exact solution**, section 4 builds a converged
  reference first - a different and better-suited method than the classical
  one it is compared with (until October 2026, Ex08.1's plate: finite
  elements on a mesh fitted to the hole, against FDM on a staircase grid;
  Ex09.2's tube in a duct: quadratic finite elements, against finite volumes
  on a staircase grid; every Part 2 set now has an exact solution) - and shows its accuracy by refinement, at least ten times better
  than the agreed accuracy. A solver too long for a cell lives in the set's
  `problem.py` and the notebook calls it.
- **A model problem stays in scaled units.** Real units and a real object are
  the rule; a problem that is an equation and not a device (the coupled
  Burgers equations Ex09.1 used until October 2026) says so and is not
  dressed as one.
- **A result that contradicts the lecture is reported, not hidden.** The
  Burgers network of the old Ex09.1 met no Reynolds ceiling on its problem;
  the notebook said so and said why, and the slide was corrected to match.

*New, 29 September 2026. Ex07.1 was written this way; Ex07.2's three notebooks
were the first built to it, then Ex08.1, Ex08.2, Ex09.1 and Ex09.2
(1 October), whose builders share `tools/exercises/part2_notebook.py`. Start a
new set's builder from `tools/exercises/ex08/build_ex08.py`, or from
`ex092/build_ex092.py` for a set with an inverse section.*
