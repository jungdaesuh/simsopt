# Single-Stage Frontier V4 Requirements And Implementation Plan

Date: 2026-04-13
Status: `v4` proposal target; `frontier_v2` implemented, `frontier_v3` initial multilane campaign implemented, `frontier_v4` not yet implemented
Scope: `examples/single_stage_optimization/` frontier architecture, runtime contracts, archive semantics, and validation gates

## Goal

Define the end-state `frontier_v4` system for single-stage optimization.

`frontier_v4` is not a tuned variant of the current one-shot scalar frontier lane. It is a real constrained multi-objective frontier engine that:

- explores several tradeoff directions from the same seed family
- maintains a non-dominated feasible archive during search
- uses multiple smooth constrained local search lanes during optimization
- returns a frontier set plus one recommended incumbent
- preserves the current `target` mode as the single-point production baseline

This plan assumes the repo will reach the earlier frontier steps first:

- `frontier_v2`: smooth constrained scalar frontier search
- `frontier_v3`: multi-probe scalarization / epsilon sweep with archive

`frontier_v4` is the first version that should be allowed to claim it performs frontier search rather than scalar comparison.

## Prerequisite Versions

This document is the `v4` target architecture, not the near-term execution plan for the currently diagnosed frontier failure mode.

The prerequisite versions are:

- `frontier_v2`: one-lane smooth constrained frontier search that removes hard Boozer trust rejection from the search path, rescales frontier normalization so the lane objective is numerically usable by the existing local optimizer, and validates behavior on the canonical feasible seed
- `frontier_v3`: multi-lane local frontier campaign using Chebyshev or epsilon-style preference variation, certified archive filtering, and target-vs-frontier reporting from one campaign without requiring a new population engine

`frontier_v4` builds on those prerequisites by freezing the long-lived campaign, archive, recommendation, resume, and engine-abstraction contracts.

## Executive Summary

The current `frontier` mode is implemented as one scalarized objective with a Boozer trust reject layered on top:

- [examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py](/Users/suhjungdae/code/columbia/simsopt-surrogate/examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:1534)
- [examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py](/Users/suhjungdae/code/columbia/simsopt-surrogate/examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:1902)

That single-lane seam is useful, but it is not yet the full frontier engine described in this document because it does not:

- maintain a Pareto archive
- search multiple preference directions in one campaign
- measure progress by hypervolume or dominance
- separate soft search-time constraints from final certification in a frontier-native way

The repo now also contains an initial `frontier_v3` campaign wrapper that runs multiple frontier lanes from one seed, maintains a certified non-dominated archive, and emits recommendation/comparison artifacts. The remaining `v4` gap is the long-lived contract freeze plus resume, reporting, and engine-abstraction work.

`frontier_v4` should upgrade the system to a campaign-level optimizer with these properties:

1. a frontier campaign owns multiple candidate search lanes
2. each lane uses smooth constrained optimization rather than hard search-time rejection
3. the campaign maintains a non-dominated feasible archive in real time
4. the campaign outputs:
   - a frontier archive
   - one recommended incumbent
   - per-member tradeoff metadata
   - campaign-level frontier diagnostics

The preferred first `v4` engine is a multi-lane local frontier campaign over the existing smooth constrained optimizer stack. Later optional engines may add evolutionary or Bayesian frontier search, but they are not the first implementation target.

## Why V4 Exists

The project has already learned three things from the current frontier work:

1. score design and search design are different problems
2. a scalarized frontier score can be directionally reasonable while the search path is still weak
3. preserving a good feasible seed is not the same as discovering a better tradeoff set

`frontier_v4` exists to solve the third problem directly.

The end-state question is no longer:

- "How do we make one scalar frontier run less fragile?"

It is:

- "How do we return a small, reliable, reproducible set of feasible tradeoff candidates from one frontier campaign?"

## Design Principles

### P1. Frontier is a set problem

`frontier_v4` must return a set of feasible tradeoff solutions, not one scalar best point.

### P2. Search-time constraints must be smooth

Invalid states that make the physics model meaningless may still require hard invalidation, but trust-style, topology-style, and engineering-style constraints must enter search through smooth penalties, augmented Lagrangian terms, or epsilon-constraint routing wherever possible.

### P3. Final certification remains hard

The campaign may traverse soft infeasibility during search, but final archive membership requires hard feasibility and solver-validity certification.

### P4. Archive truth beats scalar truth

Scalar scores are lane-local decision aids only. The frontier archive and its metric vectors are the truth surface for final comparison.

### P5. `target` remains the single-point baseline

`target` is still the production single-point optimizer. `frontier_v4` is a separate campaign workflow.

### P6. Reproducibility is mandatory

Every frontier member must carry enough metadata to rerun that exact search lane.

## Pareto Objective Contract

The first `v4` implementation must freeze the Pareto truth surface instead of leaving it implicit.

Pareto objective vector for archive dominance:

- [ ] maximize final `iota`
- [ ] maximize final nested volume
- [ ] minimize final QA error
- [ ] minimize final Boozer residual

Constraint and certification metrics, not Pareto objectives:

- [ ] curve length
- [ ] coil-coil spacing
- [ ] coil-surface spacing
- [ ] surface-vessel spacing when applicable
- [ ] curvature
- [ ] topology / confinement validity
- [ ] solver-validity and finite-state checks

Dominance reproducibility rules:

- [ ] objective directions are frozen by schema version
- [ ] objective normalization references are frozen per campaign manifest
- [ ] equality tolerance for dominance is explicit and versioned
- [ ] near-duplicate suppression threshold is explicit and versioned

## Archive State Contract

The first `v4` implementation must distinguish working-state archive bookkeeping from final frontier truth.

Archive member states:

- [ ] `provisional`: candidate has a complete objective vector from a lane, but final hard certification is not yet complete
- [ ] `certified`: candidate passes final hard certification and is eligible for frontier truth, recommendation, and final reporting
- [ ] `rejected`: candidate failed certification or replay and is retained only for diagnostics

Archive state rules:

- [ ] progress artifacts may report provisional and certified members separately
- [x] final frontier archive contains certified members only
- [x] final recommendation policy can select certified members only
- [ ] final hypervolume / dominance summaries use certified members only
- [ ] provisional members must never silently enter final comparison outputs

## Requirements

## Functional Requirements

### FR1. Frontier campaign mode

The system must support an explicit frontier campaign mode that launches multiple coordinated search lanes from one seed artifact.

Required user-facing contract:

- [x] `--single-stage-goal-mode target` remains unchanged
- [x] `frontier_v4` is selected through a frontier campaign CLI, not by overloading the single-lane CLI alone

### FR2. Multiple preference directions

Each frontier campaign must explore multiple tradeoff directions.

At least one of these mechanisms must be supported:

- [ ] reference-point sweep
- [ ] achievement / Chebyshev scalarization sweep
- [ ] epsilon-constraint sweep
- [x] multi-lane local preference schedule over the existing single-stage optimizer

Plain fixed weighted-sum-only search is insufficient for the final `v4` contract.

### FR3. Non-dominated archive with certified truth

The campaign must maintain archive state throughout execution, but frontier truth is defined only on certified members.

Archive membership rules:

- [ ] provisional members may be tracked for progress reporting only
- [x] certified members pass final hard feasibility and solver-validity checks
- [x] certified member dominance uses the frozen Pareto objective vector only
- [x] certified members that are dominated by new certified members are removed
- [x] archive keeps enough diversity to avoid near-duplicate collapse

### FR4. Smooth search-time constraint handling

Search-time treatment for the following must be smooth or ALM-routed:

- [x] Boozer trust
- [ ] hardware engineering limits
- [ ] topology / confinement limits where differentiable or cheaply relaxable

Hard invalidation is allowed only for:

- [ ] non-finite objective or gradient states
- [ ] failed surface solve with no meaningful metric state
- [ ] geometry states the pipeline cannot represent or restore safely

Conditioning requirement:

- [x] frontier lane normalization must keep active objective and soft-constraint terms in a numerically usable range for the existing local optimizer near the seed and early accepted iterates
- [ ] normalization policy must be explicit enough to prevent one lane term from dominating search solely due to scale mismatch

### FR5. Recommendation policy

The campaign must output one recommended incumbent in addition to the archive.

The recommendation policy must be explicit and reproducible.

Required initial policies:

- [x] `balanced`

Deferred policies:

- [ ] `max_iota_under_safe_boozer`
- [ ] `max_volume_under_safe_hardware`
- [ ] `closest_to_seed`

### FR6. Frontier diagnostics

The campaign must report frontier-native diagnostics.

Required metrics:

- [x] archive size
- [ ] feasible lane count
- [ ] non-dominated count
- [x] dominance updates
- [ ] hypervolume or dominance-improvement history
- [x] per-member metric vectors
- [x] per-member distance from seed
- [x] recommendation rationale

### FR7. Seed reuse and warm-start support

The campaign must reuse existing seed / warm-start / preserved-state infrastructure where possible.

Required support:

- [x] common Stage 2 seed validation
- [x] common surface identity checks
- [ ] warm-start reuse across frontier lanes where valid
- [ ] solver-owned state snapshot / restore hooks

### FR8. Single-member rerun path

Every archive member must be rerunnable as a single lane using the saved lane contract.

### FR9. Comparison support against `target`

The system must support campaign-to-target comparison without pretending scalar objective values are comparable.

Required comparison outputs:

- [x] `target` final metrics
- [x] frontier recommended-member metrics
- [x] frontier archive best-by-metric summaries
- [x] target-vs-frontier metric deltas

### FR10. Failure tolerance

The campaign must survive partial lane failure and still emit a valid archive summary.

Required behavior:

- [x] failed lane does not kill the campaign by default
- [ ] partial artifacts from interrupted lanes are salvageable
- [x] final campaign summary records lane result source and failure reason

### FR11. Frontier engine abstraction

`v4` must define a stable engine interface so multiple frontier engines can share the same archive and reporting contracts.

Required first engine:

- [x] multi-lane local frontier campaign engine

Optional later engines:

- [ ] reference-guided evolutionary frontier engine
- [ ] Bayesian multi-objective campaign engine

### FR12. Stable result schema

Campaign-level and member-level JSON schemas must be versioned and backward-readable.

## Non-Functional Requirements

### NFR1. Bounded campaign cost

`frontier_v4` must expose explicit budget controls:

- [x] total lane budget
- [x] per-lane budget
- [x] concurrent lane count or probe count
- [ ] early-stop criteria

Operational note:

- [ ] campaign defaults should be chosen from an explicit runtime calibration pass on reduced and canonical fixtures rather than arbitrary lane-count or iteration-count guesses

### NFR2. Resume support

Interrupted campaigns must be resumable from disk without corrupting archive state.

### NFR3. Deterministic metadata

All stochastic choices must be recorded:

- [x] RNG seed
- [ ] reference-point sampler seed
- [ ] engine-specific random state

### NFR4. Observability

Long campaigns must produce incremental machine-readable progress artifacts.

### NFR5. No regression to `target`

`target` path behavior, fingerprints, and result semantics must remain stable.

### NFR6. Frontier isolation

Frontier-specific code should live in new `banana_opt/` helper modules rather than growing `single_stage_banana_example.py` into a second orchestration framework.

### NFR7. Testability

Every archive rule and recommendation rule must be unit-testable without running a full expensive campaign.

### NFR8. Physics-contract clarity

The final docs must clearly distinguish:

- objective preferences
- search-time soft constraints
- final hard certification rules

## Preferred V4 Architecture

## Engine choice

Preferred first `v4` engine:

- multi-lane local frontier campaign

Why:

- matches the existing lane contract and current single-stage optimizer architecture
- reuses the smooth constrained local search machinery already being hardened in `frontier_v2` and `frontier_v3`
- addresses the observed frontier failure mode first: search dynamics and rejection behavior, not missing population heuristics
- is cheaper to validate, resume, and salvage than a new population engine
- still yields a real frontier archive because the campaign explores multiple preference directions, not just one scalar lane

Deferred engines:

- reference-guided evolutionary frontier engine for broader non-convex exploration after the archive and reporting contracts are stable
- Bayesian frontier engine using qNParEGO / qNEHVI-style campaign logic after the repo is ready to absorb a surrogate stack

## Campaign structure

The campaign should have these layers:

1. campaign planner
2. lane executor
3. archive manager
4. recommendation policy
5. reporter / salvager

### Campaign planner

Responsibilities:

- validate the seed artifact
- select frontier engine
- instantiate lane-local preference directions
- assign lane budgets
- create campaign manifest

### Lane executor

Responsibilities:

- run one search lane
- expose smooth constrained objective
- emit incremental lane checkpoints
- publish provisional and certified candidate states to archive manager

### Archive manager

Responsibilities:

- check feasibility and certification eligibility
- evaluate certified-member dominance on the frozen Pareto vector
- maintain diversity guardrails
- compute hypervolume or dominance summaries

### Recommendation policy

Responsibilities:

- select one recommended incumbent from the archive
- store rationale and policy inputs

### Reporter / salvager

Responsibilities:

- write campaign summary from final or partial artifacts
- keep lane-level provenance and result source

## Proposed Module Layout

New modules to add under `examples/single_stage_optimization/banana_opt/`:

- `frontier_archive.py`
- `frontier_dominance.py`
- `frontier_scalarization.py`
- `frontier_constraints.py`
- `frontier_engine_base.py`
- `frontier_engine_multilane_local.py`
- `frontier_recommendation.py`
- `frontier_campaign_reporting.py`

New top-level workflow entrypoint:

- `examples/single_stage_optimization/run_single_stage_frontier_campaign.py`

Existing integration points:

- [examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py](/Users/suhjungdae/code/columbia/simsopt-surrogate/examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py)
- [examples/single_stage_optimization/banana_opt/single_stage_objectives.py](/Users/suhjungdae/code/columbia/simsopt-surrogate/examples/single_stage_optimization/banana_opt/single_stage_objectives.py:25)
- [examples/single_stage_optimization/alm_utils.py](/Users/suhjungdae/code/columbia/simsopt-surrogate/examples/single_stage_optimization/alm_utils.py:1154)
- [examples/single_stage_optimization/run_single_stage_goal_mode_comparison.py](/Users/suhjungdae/code/columbia/simsopt-surrogate/examples/single_stage_optimization/run_single_stage_goal_mode_comparison.py)

## Runtime Contracts

## Campaign manifest

Required `campaign_manifest.json` fields:

- [x] `FRONTIER_VERSION`
- [x] `FRONTIER_ENGINE`
- [x] `FRONTIER_CAMPAIGN_ID`
- [x] `SEED_ARTIFACT_PATH`
- [x] `SEED_RESULTS_PATH`
- [x] `SEED_SURFACE_IDENTITY`
- [x] `FRONTIER_REFERENCE_MODE`
- [x] `FRONTIER_REFERENCE_POINTS`
- [x] `FRONTIER_SCALARIZATION_FAMILY`
- [x] `FRONTIER_CONSTRAINT_MODE`
- [x] `PARETO_OBJECTIVE_VECTOR`
- [x] `PARETO_OBJECTIVE_NORMALIZATION`
- [x] `DOMINANCE_TOLERANCE`
- [x] `DUPLICATE_DISTANCE_THRESHOLD`
- [x] `FRONTIER_RECOMMENDATION_POLICY`
- [x] `LANE_BUDGET`
- [x] `TOTAL_BUDGET`
- [x] `RNG_SEED`
- [x] `CREATED_AT`

## Lane contract

Each lane must record:

- [ ] `lane_id`
- [ ] `campaign_id`
- [ ] `engine`
- [ ] `reference_point`
- [ ] `scalarization_type`
- [ ] `scalarization_params`
- [ ] `constraint_mode`
- [ ] `warm_start_source`
- [ ] `optimizer_budget`
- [ ] `rng_seed`
- [ ] `result_source`
- [ ] `termination_reason`
- [ ] `success`
- [ ] `provisional_member_ids`
- [ ] `certified_member_ids`
- [ ] `final_certified`

## Archive member contract

Each archive member must record:

- [x] `member_id`
- [x] `lane_id`
- [x] `campaign_id`
- [x] `archive_state`
- [x] `dominance_signature`
- [x] `objective_metrics`
- [x] `constraint_metrics`
- [x] `hard_certification_ok`
- [x] `soft_search_score`
- [x] `distance_from_seed`
- [x] `hypervolume_contribution`
- [x] `recommendation_flags`
- [x] `rerun_contract`

Required objective metrics:

- [x] final `iota`
- [x] final nested volume
- [x] final QA error
- [x] final Boozer residual

Required constraint metrics:

- [x] curve length
- [x] coil-coil spacing
- [x] coil-surface spacing
- [x] surface-vessel spacing when applicable
- [x] curvature

## Recommendation contract

`frontier_recommended.json` must record:

- [x] `recommended_member_id`
- [x] `policy_name`
- [x] `policy_inputs`
- [x] `policy_rationale`
- [x] `recommended_metrics`
- [x] `frontier_archive_size`

## CLI Requirements

New campaign CLI:

```bash
python examples/single_stage_optimization/run_single_stage_frontier_campaign.py \
  --stage2-path ... \
  --frontier-version v4 \
  --frontier-engine multilane_local \
  --frontier-reference-mode reference_points \
  --frontier-num-lanes 8 \
  --frontier-recommendation-policy balanced
```

Required flags:

- [x] `--frontier-version`
- [x] `--frontier-engine`
- [ ] `--frontier-reference-mode`
- [x] `--frontier-num-lanes`
- [x] `--frontier-total-budget`
- [x] `--frontier-lane-budget`
- [x] `--frontier-recommendation-policy`
- [ ] `--frontier-hypervolume-reference`
- [x] `--frontier-rng-seed`
- [ ] `--resume`

Optional engine-specific flags:

- [ ] `--frontier-reference-points-file`
- [ ] `--frontier-epsilon-spec-file`

Deferred engine-specific flags:

- [ ] `--frontier-population-size`
- [ ] `--frontier-mutation-scale`
- [ ] `--frontier-crossover-rate`

## Implementation Plan

## Phase 0. Contract freeze

Deliverables:

- [ ] freeze `v4` JSON schemas
- [ ] freeze campaign CLI names
- [ ] freeze archive membership rules
- [ ] freeze Pareto objective vector and normalization policy
- [ ] freeze archive state semantics
- [ ] freeze recommendation policy inputs

Files:

- [ ] new docs plan
- [ ] new lightweight schema helpers under `banana_opt/`

Acceptance gate:

- [ ] schema-only tests pass
- [ ] CLI parse tests pass

## Phase 1. Archive core

Implement:

- [x] dominance checks
- [x] non-dominated archive updates
- [x] diversity tie-break rules
- [x] archive serialization

Files:

- [x] `banana_opt/frontier_dominance.py`
- [x] `banana_opt/frontier_archive.py`
- [x] tests in `tests/geo/test_frontier_archive.py`

Acceptance gate:

- [x] deterministic archive updates from synthetic metric vectors
- [x] dominance edge cases covered

## Phase 2. Frontier scalarization and constraint helpers

Implement:

- [ ] achievement / Chebyshev scalarizations
- [ ] epsilon-constraint helper functions
- [x] smooth Boozer trust penalty helpers
- [ ] smooth hardware/topology routing helpers
- [x] explicit normalization / conditioning helpers for frontier lane objectives

Files:

- [ ] `banana_opt/frontier_scalarization.py`
- [ ] `banana_opt/frontier_constraints.py`
- [ ] integration hooks into `single_stage_objectives.py`

Acceptance gate:

- [ ] scalarization invariants tested
- [x] soft-constraint helpers produce finite values and gradients on representative states
- [ ] normalization / conditioning checks show lane objective terms stay within a usable scale range near the seed and first accepted moves

## Phase 3. Lane executor contract

Implement a reusable frontier lane executor on top of the current single-stage machinery.

Responsibilities:

- [ ] consume one lane contract
- [x] run one smooth constrained search
- [x] checkpoint lane state incrementally
- [ ] publish provisional and certified candidates to the archive manager

Files:

- [ ] `banana_opt/frontier_engine_base.py`
- [x] `single_stage_banana_example.py` integration hooks

Acceptance gate:

- [ ] one-lane frontier run can publish a certified member to a mock archive
- [x] interrupted run writes salvageable lane summary

## Phase 4. Multilane local frontier campaign

Implement the first real `v4` engine:

- [x] multi-lane local campaign executor
- [x] reference-point / scalarization / epsilon schedule generation
- [x] lane scheduling and budget allocation
- [x] archive update loop

Files:

- [x] `banana_opt/frontier_engine_multilane_local.py`
- [x] `run_single_stage_frontier_campaign.py`

Acceptance gate:

- [x] campaign runs multiple lanes from one seed
- [x] final archive contains more than one member on synthetic or reduced fixtures
- [x] campaign survives one failed lane without losing summary output

## Phase 4b. Deferred evolutionary engine

Implement only after the archive contracts, lane contracts, and reporting seams are stable:

- [ ] reference-guided evolutionary frontier engine
- [ ] population-to-archive adapter
- [ ] evolutionary-specific progress reporting

## Phase 5. Recommendation and reporting

Implement:

- [x] recommendation policies
- [ ] hypervolume / dominance progress reporting
- [x] final campaign summary
- [x] target-vs-frontier comparison adapter

Files:

- [x] `banana_opt/frontier_recommendation.py`
- [ ] `banana_opt/frontier_campaign_reporting.py`
- [x] `run_single_stage_goal_mode_comparison.py` comparison adapter updates

Acceptance gate:

- [x] recommendation policy is deterministic for fixed archive input
- [x] campaign summary and comparison summary are readable from partial lane artifacts

## Phase 6. Resume and persistence

Implement:

- [ ] campaign resume
- [ ] lane resume
- [ ] archive replay from partial checkpoints

Acceptance gate:

- [ ] interrupted campaign can resume and produce the same final archive as a clean uninterrupted run on deterministic smoke fixtures

## Phase 7. Expensive-fixture validation

Run controlled campaigns on known feasible seeds.

Required validation questions:

- [ ] does the archive contain more than one distinct certified member?
- [ ] does recommended frontier member beat or match `target` on the agreed scorecard?
- [ ] does frontier find tradeoff points `target` cannot reach in one run?
- [ ] does campaign reporting stay intact under interrupted lanes?

Acceptance gate:

- [ ] at least one canonical seed yields a nontrivial feasible frontier archive
- [ ] no regression to `target`

## Testing Plan

## Unit tests

Add:

- [x] `tests/geo/test_frontier_archive.py`
- [ ] `tests/geo/test_frontier_scalarization.py`
- [ ] `tests/geo/test_frontier_recommendation.py`

Coverage:

- [x] dominance rules
- [x] archive replacement
- [x] duplicate / near-duplicate handling
- [x] recommendation policy determinism
- [x] schema serialization

## Workflow helper tests

Add:

- [x] frontier campaign manifest writing
- [ ] partial lane salvage
- [ ] resume path
- [x] target-vs-frontier comparison adapter

## Integration tests

Add:

- [ ] reduced-fixture multi-lane frontier campaign
- [ ] interrupted campaign resume
- [x] archive emission from mixed success / failure lanes

## Regression tests

Preserve:

- [ ] current `target` behavior
- [x] current goal-mode comparison wrapper behavior
- [ ] salvage semantics from partial artifacts

## Risks

### R1. Compute cost explosion

Mitigation:

- explicit campaign budgets
- early archive-based stopping
- reduced-fixture validation first

### R2. Archive degenerates to near-duplicates

Mitigation:

- diversity threshold on metric vectors
- distance-to-seed plus hypervolume tie-breaks

### R3. Smooth constraint terms become numerically fragile

Mitigation:

- reuse ALM helpers where possible
- add finite-value / finite-gradient tests around thresholds

### R4. Campaign complexity leaks into `single_stage_banana_example.py`

Mitigation:

- keep campaign orchestration in new modules
- add only narrow lane hooks to the existing single-stage file

### R5. Users misread recommendation as frontier truth

Mitigation:

- always emit archive and recommendation together
- docs must state that recommendation is policy-dependent

## Out Of Scope

For the first `v4` landing:

- full Bayesian frontier engine
- reference-guided evolutionary frontier engine
- replacing `target` as the production default
- automatic Stage 2 frontier search
- global Pareto search across Stage 2 and single-stage jointly
- exact checkpoint continuation of every solver-internal state

## Decision Rules

Promote `frontier_v4` from proposal to active experimental workflow only when:

- [x] archive and recommendation schemas are implemented
- [x] multi-lane campaign survives partial failures
- [ ] reduced and real-fixture validation demonstrate nontrivial feasible archive output

Do not promote `frontier_v4` to replace `target` unless:

- [ ] the recommended frontier member is consistently competitive with or better than `target` on the agreed scorecard
- [ ] the campaign archive reveals useful tradeoff structure that one-shot target runs do not provide
- [ ] runtime and resume behavior are operationally acceptable

## Concrete First Landing Recommendation

The first implementable `v4` slice should be:

1. [x] archive core
2. [x] smooth frontier scalarization helpers
3. [x] multi-lane local frontier campaign runner
4. [x] recommendation policy
5. [ ] partial-artifact salvage and resume

This intentionally avoids a premature jump to a full Bayesian frontier stack.

## Related Notes

- [docs/single_stage_frontier_impl_plan_2026-04-12.md](/Users/suhjungdae/code/columbia/simsopt-surrogate/docs/single_stage_frontier_impl_plan_2026-04-12.md)
- [docs/single_stage_goal_mode_comparison_plan_2026-04-12.md](/Users/suhjungdae/code/columbia/simsopt-surrogate/docs/single_stage_goal_mode_comparison_plan_2026-04-12.md)
- [docs/single_stage_search_gate_plan_2026-04-08.md](/Users/suhjungdae/code/columbia/simsopt-surrogate/docs/single_stage_search_gate_plan_2026-04-08.md)
