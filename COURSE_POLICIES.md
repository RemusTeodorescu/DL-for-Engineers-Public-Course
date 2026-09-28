# Course Policies

Eight rules for the lectures and exercises of *Deep Learning for Engineering*.
They replace the 35 separate slide policies (P1–P12, D1–D18, E1–E9) that grew
out of the reviews in September 2026; each old policy now sits under one of the
eight, and its check still runs.

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

`python3 tools/deck/slide_policies.py --list` prints the eight with the old
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
- Before release a set runs from a fresh Colab runtime (`tools/light/solve_run.py`)
  and is reviewed once by a person other than its author.

*Replaces D14, E7. The limit, the scope rule and the release check are new.*
