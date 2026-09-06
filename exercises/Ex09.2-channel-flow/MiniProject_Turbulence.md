# L13 MiniProject — Turbulence Modelling with PINNs

Applied PINN for Energy · Aalborg University

## Why this topic

L9.2 stops where the research begins. Every closure used in that lecture is
*prescribed*: someone else calibrated it, and the network merely solves the
resulting equations. The open question — whether a network can **learn** the
closure, and whether what it learns generalises — is live, unresolved, and
well matched to a project.

You are not expected to solve it. You are expected to pose a precise question,
run a controlled study, and report honestly on what you found, including
failure.

## Prerequisites

Ex_9.1 and Ex_9.2 completed. You will reuse `Ex_9_2_pipe.py` and the control
panel; the geometry, the sampling and the reporting are already built.

## Tracks

Choose one. Each is a real question with no published answer for this setup.

### Track A — Learn the closure

Represent the eddy viscosity by a second network, ν_t = N(x), constrained by
the RANS residual and by reference data at one Reynolds number.

- Does the learned ν_t reproduce the qualitative structure a mixing-length
  model would predict — small near the centreline, large in the shear layer?
- **The real question:** does it generalise to a Reynolds number it never saw?
- Enforce ν_t ≥ 0 with a softplus output. Explain why a negative eddy viscosity
  destabilises training.

### Track B — Assimilate sparse data

Reconstruct the mean field from a handful of virtual sensors, sampled from a
converged Ex_9.2 run treated as ground truth.

- How few sensors before reconstruction fails, and *where* does it fail first?
- Does sensor placement matter more than sensor count?
- Compare against pure interpolation with no physics. Quantify what the
  residual term buys you.

### Track C — Enforce the wall law

Impose the logarithmic profile as a hard constraint near the wall instead of
resolving the sublayer.

- How much accuracy in drag does it buy per unit of compute?
- At what Reynolds number does resolving become infeasible and the wall
  function essential?
- Compare a hard-enforced wall law against a penalty term.

### Track D — Map the limit

Chart where a plain PINN stops working, across Reynolds number, geometry,
blockage and sampling density.

- Produce a map of the parameter space marked converged / degraded / failed,
  with a stated criterion for each.
- Characterise the *manner* of failure: does the error grow, or does the field
  stay plausible while becoming wrong?
- **A carefully established negative result is a full-credit outcome.** Knowing
  where a method fails is worth as much as making it succeed, and it is rarer.

## Deliverables

1. **A question**, stated in one sentence, with a criterion for answering it.
2. **A controlled study**: what varied, what was held fixed, what seed.
3. **A report** of 6–10 pages: method, results, interpretation, limitations.
4. **Runnable code**, built on the course modules.
5. **A 10-minute presentation.**

## Assessment

| Weight | Criterion |
|---|---|
| 25% | Is the question precise, and is the criterion for answering it stated? |
| 25% | Is the study controlled — one variable at a time, seeds fixed and reported? |
| 30% | Is the interpretation sound, and are the limitations stated honestly? |
| 20% | Is the code clear and reproducible? |

Note what is **not** assessed: whether the method worked. A study showing that
an approach fails, with the failure characterised and explained, scores as
highly as one showing success.

## Guidance

- **Scope down.** One Reynolds number, one geometry, one question answered
  properly beats four half-answered.
- **Fix and report your seed.** Results that move between runs are not results.
- **Verify before you interpret.** Reproduce a case from Ex_9.2 before changing
  anything.
- **Distrust a pretty picture.** A smooth, plausible, wrong field is the
  characteristic failure of a PINN on turbulent flow — L9.2 slide 12.

## Reading

- Pope, *Turbulent Flows* (2000) — Ch. 4 for Reynolds averaging, Ch. 10 for
  eddy-viscosity closures.
- Raissi, Perdikaris & Karniadakis (2019), *J. Comput. Phys.* 378, 686–707.
- Current literature on PINNs for RANS and on closure learning — search this
  yourself; the field moves quickly, and finding the state of the art is part
  of the project.
