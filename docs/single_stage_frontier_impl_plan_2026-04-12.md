# Single-Stage Frontier Formulation Implementation Plan

Date: 2026-04-12
Status: proposal only, not yet implemented
Scope: `examples/single_stage_optimization/` single-stage objective assembly, CLI semantics, ALM usage, and documentation

## Goal

Replace the current target-tracking single-stage formulation with a constrained frontier-search formulation that matches the stated design problem:

- maximize `iota`
- maximize nested-surface `volume`
- minimize QA error
- minimize engineering complexity
- require hardware-feasible, solver-valid final designs

This note does not change Stage 2 into a physics frontier search. Stage 2 should remain a field-error-plus-engineering cleanup stage unless a separate design decision says otherwise.

## Executive Summary

The current repo already contains most of the structural pieces needed for the proposed formulation:

- a base single-stage physics objective
- ALM constraint handling
- explicit search-time hardware policy with `hard`, `warn`, and `adaptive`
- final hard certification semantics

What is misaligned with the stated project goal is not the existence of constraints. It is the meaning of the physics objective:

- `iota` is currently encoded as a target penalty
- `volume` is currently encoded as an internal Boozer-surface target, not as an outer search objective
- `Boozer residual` is currently treated like an ordinary weighted scalar term even though it also acts as a solver-validity / trust metric

The proposed replacement is:

1. keep a scalarized frontier-search objective over `iota`, `volume`, QA, and complexity
2. keep hardware and buildability limits in ALM constraints
3. treat solver-invalid states as invalid-state guards, not ordinary inequalities
4. keep search-time traversal soft when needed and final certification hard

## Why This Change Is Needed

The current single-stage objective does not reflect the latest stated problem:

- there is no fixed target `iota`
- there is no fixed target `volume`
- the actual goal is to find the best feasible tradeoff among several competing physics and engineering quantities

When no fixed target exists, a target penalty injects arbitrary bias. A frontier-search or scalarized multi-objective formulation is a closer mathematical match.

## Current Repo Behavior

### Single-stage

Current single-stage objective assembly:

- `JnonQSRatio`
- `RES_WEIGHT * JBoozerResidual`
- `IOTAS_WEIGHT * Jiota`
- `LENGTH_WEIGHT * JCurveLength`
- `CC_WEIGHT * JCurveCurve`
- `CS_WEIGHT * JCurveSurface`
- `CURVATURE_WEIGHT * JCurvature`
- optional vessel-gap term

Code anchors:

- [examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py](/Users/suhjungdae/code/columbia/simsopt-surrogate/examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:1677)
- [examples/single_stage_optimization/banana_opt/single_stage_objectives.py](/Users/suhjungdae/code/columbia/simsopt-surrogate/examples/single_stage_optimization/banana_opt/single_stage_objectives.py:25)

Current target-oriented pieces:

- `Jiota = QuadraticPenalty(..., iota_target)`
- `BoozerSurface(..., vol_target, ...)`

Code anchors:

- [examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py](/Users/suhjungdae/code/columbia/simsopt-surrogate/examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:1188)
- [examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py](/Users/suhjungdae/code/columbia/simsopt-surrogate/examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:1687)

Current ALM support:

- `weighted_sum`: physics remains in base objective
- `thresholded_physics`: physics terms are promoted to inequality constraints

Code anchors:

- [examples/single_stage_optimization/banana_opt/single_stage_objectives.py](/Users/suhjungdae/code/columbia/simsopt-surrogate/examples/single_stage_optimization/banana_opt/single_stage_objectives.py:142)
- [examples/single_stage_optimization/banana_opt/single_stage_objectives.py](/Users/suhjungdae/code/columbia/simsopt-surrogate/examples/single_stage_optimization/banana_opt/single_stage_objectives.py:192)
- [examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py](/Users/suhjungdae/code/columbia/simsopt-surrogate/examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:808)

Current search-time hardware policy:

- `hard`
- `warn`
- `adaptive`

Code anchors:

- [examples/single_stage_optimization/banana_opt/single_stage_search_policy.py](/Users/suhjungdae/code/columbia/simsopt-surrogate/examples/single_stage_optimization/banana_opt/single_stage_search_policy.py:5)
- [examples/single_stage_optimization/README.md](/Users/suhjungdae/code/columbia/simsopt-surrogate/examples/single_stage_optimization/README.md:319)

### Stage 2

Stage 2 remains squared-flux plus engineering regularization / constraints:

- squared flux
- length
- coil-coil distance
- curvature
- banana current in ALM mode

Code anchors:

- [examples/single_stage_optimization/STAGE_2/banana_coil_solver.py](/Users/suhjungdae/code/columbia/simsopt-surrogate/examples/single_stage_optimization/STAGE_2/banana_coil_solver.py:620)
- [examples/single_stage_optimization/banana_opt/stage2_objectives.py](/Users/suhjungdae/code/columbia/simsopt-surrogate/examples/single_stage_optimization/banana_opt/stage2_objectives.py:399)

## Proposed Formulation

### Mathematical statement

Let `x` be the optimized design variables.

We want:

```text
maximize   [ iota(x), V(x), -Q(x), -C(x) ]
subject to g_j(x) <= 0
           B(x) <= eps_B   or equivalent solver-validity requirement
           x in D_valid
```

where:

- `iota(x)` is the chosen rotational-transform metric
- `V(x)` is the nested-surface volume metric
- `Q(x)` is the QA error metric
- `C(x)` is the engineering complexity metric
- `g_j(x)` are hardware / geometry constraints
- `B(x)` is a Boozer-validity or Boozer-residual gate
- `D_valid` is the set of states for which the model remains meaningful

### Scalarized implementation target

For a practical minimizer, use:

```text
minimize
    w_Q * Q_norm(x)
  + w_C * C_norm(x)
  - w_i * iota_norm(x)
  - w_V * V_norm(x)
  + w_B * Phi_B(B(x))

subject to
    g_j(x) <= 0
    x in D_valid
```

Interpretation:

- `Phi_B` should behave like a gate or steep trust penalty, not just another smooth preference term
- if a hard Boozer threshold is preferred, move `B(x)` into ALM instead

## Before vs After

### Before

```text
minimize
  J_QA
  + w_B * J_Boozer
  + w_iota * J_iota_target_penalty
  + w_len * J_length_penalty
  + w_cc * J_cc
  + w_cs * J_cs
  + w_curv * J_curvature
```

with:

- `iota` as target-matching penalty
- `volume` hidden inside `vol_target` in the Boozer solve
- `Boozer residual` as ordinary weighted term

### After

```text
minimize
  w_QA * QA_norm
  + w_complex * Complex_norm
  - w_iota * iota_norm
  - w_vol * volume_norm
  + w_B * Boozer_gate_penalty
```

subject to:

- ALM traversable hardware constraints
- invalid-state guards for solver-invalid states

## Constraint Taxonomy

### A. Objective terms

These should rank acceptable candidates:

- [ ] `iota`
- [ ] `volume`
- [ ] QA error
- [ ] engineering complexity

### B. ALM constraints

These may be violated during search but must be satisfied at convergence:

- [ ] length
- [ ] coil-coil distance
- [ ] coil-surface distance
- [ ] surface-surface / vessel distance
- [ ] curvature
- [ ] banana current
- [ ] any future strain / torsion hard limit

### C. Invalid-state guards

These are not ordinary inequalities:

- [ ] Boozer solve failure
- [ ] non-finite objective or gradient
- [ ] self-intersection or geometry degeneracy
- [ ] loss of meaningful nested-surface state for downstream metrics
- [ ] topology / tracing failure that makes the candidate untrustworthy rather than merely poor

## Implementation Plan

### Phase 0: Freeze intent and semantics

- [ ] Confirm that `iota` is a maximize quantity, not a target quantity
- [ ] Confirm that `volume` is a maximize quantity, not a hidden target quantity
- [ ] Confirm whether `Boozer residual` should be:
  - a hard validity threshold
  - or a steep trust penalty plus invalid-state reject on failure
- [ ] Confirm which complexity metrics should define `C(x)` in the first landed version

### Phase 1: Separate validity from ranking

- [ ] Audit all single-stage failure paths and classify them as:
  - invalid-state reject
  - modeled infeasibility
  - ordinary poor objective value
- [ ] Keep search-time `warn` / `adaptive` available for traversable infeasibility
- [ ] Keep final certification hard-feasible
- [ ] Document that invalid-state guards are distinct from ALM constraints

Primary touch points:

- [ ] [examples/single_stage_optimization/banana_opt/single_stage_search_policy.py](/Users/suhjungdae/code/columbia/simsopt-surrogate/examples/single_stage_optimization/banana_opt/single_stage_search_policy.py:1)
- [ ] [examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py](/Users/suhjungdae/code/columbia/simsopt-surrogate/examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:1544)

### Phase 2: Replace target-style `iota`

- [ ] Add a scalar reward formulation for `iota`
- [ ] Keep backward compatibility for old target-based runs behind a separate mode or flag
- [ ] Rename or deprecate `--iota-target` in frontier mode
- [ ] Introduce normalization for `iota` so its scale is comparable to QA and complexity terms

Possible implementation pattern:

- `--single-stage-goal-mode {target, frontier}`
- `target` keeps current behavior
- `frontier` switches to monotone `iota` reward

Primary touch points:

- [ ] [examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py](/Users/suhjungdae/code/columbia/simsopt-surrogate/examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:849)
- [ ] [examples/single_stage_optimization/banana_opt/single_stage_objectives.py](/Users/suhjungdae/code/columbia/simsopt-surrogate/examples/single_stage_optimization/banana_opt/single_stage_objectives.py:25)

### Phase 3: Expose `volume` as an outer objective

- [ ] Define a first-class `JVolume`
- [ ] Decide whether to use absolute volume, normalized volume, or volume relative to the seed / reference surface
- [ ] Keep internal Boozer-surface `vol_target` machinery only if required for the numerical solve
- [ ] Ensure frontier mode can reward larger nested volume even when no explicit target exists
- [ ] Thread `JVolume` into results reporting and checkpoint summaries

Primary touch points:

- [ ] [examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py](/Users/suhjungdae/code/columbia/simsopt-surrogate/examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:1188)
- [ ] [examples/single_stage_optimization/banana_opt/single_stage_geometry.py](/Users/suhjungdae/code/columbia/simsopt-surrogate/examples/single_stage_optimization/banana_opt/single_stage_geometry.py:32)
- [ ] [examples/single_stage_optimization/banana_opt/single_stage_objectives.py](/Users/suhjungdae/code/columbia/simsopt-surrogate/examples/single_stage_optimization/banana_opt/single_stage_objectives.py:142)

### Phase 4: Reframe Boozer residual

- [ ] Separate `Boozer failure` from `large but finite Boozer residual`
- [ ] Implement one of two explicit contracts:
  - hard threshold in ALM
  - trust-gate penalty in the frontier scalar objective
- [ ] Document the chosen contract in the single-stage README and CLI help
- [ ] Ensure invalid Boozer states do not masquerade as merely poor objective values

Primary touch points:

- [ ] [examples/single_stage_optimization/banana_opt/single_stage_objectives.py](/Users/suhjungdae/code/columbia/simsopt-surrogate/examples/single_stage_optimization/banana_opt/single_stage_objectives.py:192)
- [ ] [examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py](/Users/suhjungdae/code/columbia/simsopt-surrogate/examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:808)

### Phase 5: Keep ALM for traversable hardware constraints

- [ ] Retain ALM for buildability constraints
- [ ] Prefer `warn` or `adaptive` search-time policy when ALM is active
- [ ] Keep `thresholded_physics` optional rather than the default for frontier mode unless explicit physics thresholds are available
- [ ] Ensure final certification remains hard-feasible in all search modes

Primary touch points:

- [ ] [examples/single_stage_optimization/README.md](/Users/suhjungdae/code/columbia/simsopt-surrogate/examples/single_stage_optimization/README.md:281)
- [ ] [examples/single_stage_optimization/banana_opt/single_stage_objectives.py](/Users/suhjungdae/code/columbia/simsopt-surrogate/examples/single_stage_optimization/banana_opt/single_stage_objectives.py:192)

### Phase 6: CLI and documentation cleanup

- [ ] Add explicit frontier-mode docs
- [ ] Avoid ambiguous language like "target" when no target exists
- [ ] Distinguish:
  - objective terms
  - ALM constraints
  - invalid-state guards
- [ ] Add a short README statement:
  - "maximize `iota` and nested volume, minimize QA error and engineering complexity, subject to hardware and solver-validity constraints"

Primary touch points:

- [ ] [examples/single_stage_optimization/README.md](/Users/suhjungdae/code/columbia/simsopt-surrogate/examples/single_stage_optimization/README.md:1)
- [ ] repo root [README.md](/Users/suhjungdae/code/columbia/simsopt-surrogate/README.md:1) if public-facing wording should change there too

### Phase 7: Validation

- [ ] Unit tests for new frontier objective assembly
- [ ] Regression tests proving old target mode still reproduces prior behavior
- [ ] Tests for invalid-state classification
- [ ] Tests for search-time `warn` / `adaptive` behavior under frontier mode
- [ ] At least one end-to-end dry run showing frontier mode can traverse temporary infeasibility and still return a hard-feasible final result
- [ ] Confirm results metadata records enough information to reconstruct:
  - scalar weights
  - chosen goal mode
  - validity-gate settings
  - final constraint feasibility

Likely test files:

- [ ] [tests/geo/test_single_stage_example.py](/Users/suhjungdae/code/columbia/simsopt-surrogate/tests/geo/test_single_stage_example.py:1)
- [ ] [tests/geo/test_single_stage_alm_integration.py](/Users/suhjungdae/code/columbia/simsopt-surrogate/tests/geo/test_single_stage_alm_integration.py:1)
- [ ] [tests/geo/test_banana_objective_modules.py](/Users/suhjungdae/code/columbia/simsopt-surrogate/tests/geo/test_banana_objective_modules.py:1)

## Recommended First Landing

The smallest high-leverage first landing is:

- [ ] add `frontier` goal mode
- [ ] replace target-style `Jiota` with monotone `iota` reward in that mode
- [ ] add explicit `JVolume`
- [ ] keep current ALM hardware handling
- [ ] keep `Boozer residual` as a steep gate-like penalty first, not a hard ALM physics threshold

Why this first:

- it aligns the objective with the stated problem
- it preserves the current Stage 2 / ALM architecture
- it avoids forcing arbitrary physics thresholds before the project has them

## Risks

- Weight tuning can become arbitrary if normalization is poor
- Promoting `volume` to an outer objective may expose solver pathologies that were hidden by fixed `vol_target`
- Treating Boozer residual as too soft a penalty can admit misleading candidates
- Treating Boozer residual as too hard a threshold too early can overconstrain exploration
- Backward compatibility can be lost if target mode is removed instead of explicitly preserved

## Source Map

This table separates direct source support from repo-specific inference.

| Claim | Support type | Source |
| --- | --- | --- |
| Single-stage optimization combines physics and engineering in one optimization | direct literature support | Giuliani et al. 2020, "Single-stage gradient-based stellarator coil design: Optimization for near-axis quasi-symmetry" https://arxiv.org/pdf/2010.02033v2 |
| Boozer-coordinate / Boozer-surface quantities, quasisymmetry, rotational transform, and coil complexity can be optimized together | direct literature support | Giuliani et al. 2022, "Direct computation of magnetic surfaces in Boozer coordinates and coil optimization for quasi-symmetry" https://arxiv.org/pdf/2203.03753v2 |
| Stellarator optimization should explicitly support hard constraints on physics or design variables | direct literature support | Conlin et al. 2024, "Stellarator Optimization with Constraints" https://arxiv.org/pdf/2403.11033v1 |
| Volume of nested flux surfaces trades off against other physics / engineering goals | direct literature support | Lee et al. 2022, "Stellarator coil optimization supporting multiple magnetic configurations" https://arxiv.org/pdf/2208.01096v3 |
| Single-stage frameworks include confinement and engineering constraints together | direct literature support | Jorge et al. 2024, "Simplified and Flexible Coils for Stellarators using Single-Stage Optimization" https://arxiv.org/pdf/2406.07830v1 |
| CSX-specific optimization should derive objectives and constraints from project needs, not inherit a generic target-only formulation | direct literature support | Baillod et al. 2024, "Integrating Novel Stellarator Single-Stage Optimization Algorithms to Design the Columbia Stellarator Experiment" https://arxiv.org/pdf/2409.05261v1 |
| Upstream SIMSOPT single-stage examples separate plasma-stage and coil-stage ingredients | direct code / docs support | SIMSOPT single-stage docs https://simsopt.readthedocs.io/v1.2.0/example_single_stage.html and example code https://raw.githubusercontent.com/hiddenSymmetries/simsopt/master/examples/3_Advanced/single_stage_optimization.py |
| Upstream SIMSOPT stage-two examples are primarily flux plus engineering regularization | direct code / docs support | SIMSOPT coil docs https://simsopt.readthedocs.io/v1.2.0/example_coils.html and stage-two example code https://raw.githubusercontent.com/hiddenSymmetries/simsopt/master/examples/2_Intermediate/stage_two_optimization.py |
| This repo already separates base objective, ALM constraints, and search-time rejection policy | direct repo support | [single_stage_objectives.py](/Users/suhjungdae/code/columbia/simsopt-surrogate/examples/single_stage_optimization/banana_opt/single_stage_objectives.py:142), [single_stage_objectives.py](/Users/suhjungdae/code/columbia/simsopt-surrogate/examples/single_stage_optimization/banana_opt/single_stage_objectives.py:192), [single_stage_search_policy.py](/Users/suhjungdae/code/columbia/simsopt-surrogate/examples/single_stage_optimization/banana_opt/single_stage_search_policy.py:5) |
| Search-time infeasible traversal but hard final certification is already the repo's intended pattern | direct repo support | [examples/single_stage_optimization/README.md](/Users/suhjungdae/code/columbia/simsopt-surrogate/examples/single_stage_optimization/README.md:281) |
| If no fixed `iota` or `volume` targets exist, target penalties for those quantities are artificial and should be replaced by monotone rewards or frontier scalarization | inference from sources + project statement | inferred from your stated project goal plus the above code and literature |
| Boozer residual should partly act as a validity / trust gate, not only as a peer scalar objective | inference from repo role of Boozer solve + optimization design judgment | inferred from current Boozer-surface dependence and invalid-state behavior; not claimed as a direct quote from one paper |

## Acceptance Criteria

This plan should be considered complete only when all of the following are true:

- [ ] frontier mode exists and is documented
- [ ] target mode still exists or is intentionally removed with migration notes
- [ ] `iota` and `volume` are first-class outer search objectives in frontier mode
- [ ] hardware / buildability constraints remain explicit and final-hard
- [ ] invalid-state guards are documented and tested separately from ALM constraints
- [ ] results metadata is sufficient to reproduce the chosen frontier scalarization

