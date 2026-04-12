# Single-Stage Goal-Mode Comparison Plan

Date: 2026-04-12
Status: comparison protocol with the first frontier-specific iota-objective slice landed
Scope: `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py` runs and downstream result summaries

## Purpose

Add a new single-stage goal-mode seam without replacing the existing target-based workflow.

The comparison contract is:

- keep the current formulation as `--single-stage-goal-mode target`
- add the new formulation incrementally under `--single-stage-goal-mode frontier`
- run matched A/B comparisons before deciding whether frontier mode should become the preferred path

## Phase 1 Contract

The first landing established comparison infrastructure. The current landing adds the
first frontier-specific objective difference without replacing the legacy path.

- [x] `target` remains the default
- [x] `frontier` exists as a CLI-visible mode
- [x] run identity changes when goal mode changes, while explicit `target` remains fingerprint-compatible with legacy no-flag runs
- [x] results metadata records the selected goal mode
- [x] frontier objective semantics differ from target mode
- [x] direct single-stage CLI default is documented as `SINGLE_STAGE_GOAL_MODE`-overridable

Current frontier scope:

- [x] `frontier` swaps the single-stage `Jiota` term from target penalty to monotone iota reward
- [ ] `frontier` includes a volume reward rather than the legacy target-driven volume contract
- [ ] `frontier` revises search-time hardware/topology behavior
- [ ] `frontier` supports `ALM thresholded_physics`

## Matched-Run Rules

When comparing `target` and `frontier`, keep all of these fixed:

- [x] Stage 2 seed artifact or Stage 2 seed spec
- [x] equilibrium / plasma-surface file
- [x] hardware constraints
- [x] optimizer budget
- [x] ALM settings
- [x] search-time hardware mode
- [x] basin-hopping settings
- [x] Boozer stage / refinement settings
- [ ] random seed where applicable
- [x] multi-surface / topology / confinement settings
- [x] plasma-current and banana-surface-radius settings

Only the single-stage goal mode should differ.

## Required Reported Metrics

Every comparison summary should report, at minimum:

- [x] final feasibility pass / fail
- [x] final `iota`
- [x] final nested-surface volume
- [x] final QA error
- [x] final Boozer residual
- [x] final engineering-complexity metrics
- [x] final hardware margins:
  - coil length
  - coil-coil spacing
  - coil-surface spacing
  - surface-vessel spacing when applicable
  - curvature
  - shared Stage 2 banana current metadata when present
- [x] termination reason
- [x] optimizer success flag
- [x] count of invalid-state rejects
- [x] best-feasible checkpoint metrics when the run captures a best-feasible incumbent

## Run Metadata Requirements

The run record must distinguish:

- [x] selected goal mode
- [x] whether frontier objective semantics were actually active
- [x] `target`-specific fields still used for backward compatibility
- [x] enough information to reproduce the exact scalarization or target contract

## Decision Rule

Do not replace the target-based path until matched runs show that frontier mode is genuinely better for the project goal.

At minimum, a frontier-mode promotion case should show one of:

- [ ] better feasible `iota` at comparable QA / complexity
- [ ] better feasible volume at comparable QA / complexity
- [ ] better overall tradeoff on the agreed project scorecard
- [ ] comparable physics with better convergence or feasibility behavior

## Sources and Rationale

The comparison-first rollout is justified by:

- the repo already having a stable target-based baseline
- the new formulation being a modeling change, not just an implementation cleanup
- the need to separate "new physics objective" wins from accidental code-path drift

Supporting context:

- current frontier implementation plan: [docs/single_stage_frontier_impl_plan_2026-04-12.md](/Users/suhjungdae/code/columbia/simsopt-surrogate/docs/single_stage_frontier_impl_plan_2026-04-12.md)
- single-stage manual workflow notes: [examples/single_stage_optimization/README.md](/Users/suhjungdae/code/columbia/simsopt-surrogate/examples/single_stage_optimization/README.md)
