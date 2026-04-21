# Single-Stage Frontier + Global Pareto Plan

Date: 2026-04-22  
Status: Merged execution plan after frontier validation and global-engine design review  
Scope: `examples/single_stage_optimization/SINGLE_STAGE/` and supporting `banana_opt/` frontier modules

## Goal

Build from the current certified local frontier workflow toward a reproducible, evaluator-backed global Pareto engine without destabilizing the existing frontier contract.

This plan intentionally sequences the work as:

1. Harden the current frontier contract.
2. Extract a deterministic evaluator seam.
3. Add the first global engine.
4. Escalate to surrogates or decomposition only if cost and results justify it.

## Executive Summary

The current frontier implementation is legitimate, tested, and useful, but it is still a lane-oriented local campaign rather than a true global Pareto engine. The right near-term move is not to jump straight to a population algorithm. The right move is to first remove contract ambiguities in frontier mode and extract a serializable, deterministic evaluator that can be called in-process without module-global state.

After that seam exists, `NSGA-III` is the first global engine to add. It is chosen because:

- it is reference-direction based, which matches the planned full-simplex frontier direction generation
- it fits `pymoo`'s feasibility-first / CV-oriented constrained workflow
- it requires a smaller architectural lift than `MOEA/D` from the repo's current lane/archive design
- official `pymoo` `MOEA/D` still rejects constrained problems outright[^ext-pymoo-moead]

This does **not** mean `NSGA-III` is the final engine. It means it is the best first benchmark once the evaluator seam is real.

## Current Context

### What already exists

- [x] A working frontier campaign runner with a certified archive
- [x] Schema-versioned frontier lane contracts and progress records
- [x] Hypervolume history reporting for lane-completion events
- [x] Search-time hard invalidation taxonomy
- [x] Search-time trust, topology, and hardware gating
- [x] In-process objective evaluation building blocks

### What does not exist yet

- [ ] A stateless, serializable, in-process frontier evaluator callable suitable for a population engine
- [ ] A global Pareto engine under `--frontier-engine`
- [ ] Full 4-objective shared-mode exploration without custom JSON inputs
- [ ] Frozen scalarization normalization as an explicit per-campaign contract
- [ ] A documented constrained decomposition engine path

## Why This Sequence

The frontier hardening items are small, high-signal fixes to the current local engine. The evaluator seam is the true blocker for any global engine. Without a deterministic evaluator, caching is unsound, cross-worker comparisons are noisy, and distributed execution is brittle. Only after that seam exists does it make sense to compare global engines.

## Current Code Anchors

### Frontier runner and archive

- [`examples/single_stage_optimization/run_single_stage_frontier_campaign.py`](../examples/single_stage_optimization/run_single_stage_frontier_campaign.py)
- [`examples/single_stage_optimization/banana_opt/frontier_engine_base.py`](../examples/single_stage_optimization/banana_opt/frontier_engine_base.py)
- [`examples/single_stage_optimization/banana_opt/frontier_archive.py`](../examples/single_stage_optimization/banana_opt/frontier_archive.py)
- [`examples/single_stage_optimization/banana_opt/frontier_campaign_reporting.py`](../examples/single_stage_optimization/banana_opt/frontier_campaign_reporting.py)
- [`examples/single_stage_optimization/banana_opt/frontier_recommendation.py`](../examples/single_stage_optimization/banana_opt/frontier_recommendation.py)
- [`examples/single_stage_optimization/banana_opt/frontier_dominance.py`](../examples/single_stage_optimization/banana_opt/frontier_dominance.py)

### Single-stage evaluation path

- [`examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py`](../examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py)
- [`examples/single_stage_optimization/banana_opt/single_stage_objectives.py`](../examples/single_stage_optimization/banana_opt/single_stage_objectives.py)
- [`examples/single_stage_optimization/banana_opt/single_stage_geometry.py`](../examples/single_stage_optimization/banana_opt/single_stage_geometry.py)
- [`examples/single_stage_optimization/banana_opt/frontier_constraints.py`](../examples/single_stage_optimization/banana_opt/frontier_constraints.py)
- [`examples/single_stage_optimization/topology_scorer.py`](../examples/single_stage_optimization/topology_scorer.py)
- [`examples/single_stage_optimization/banana_opt/coil_order_upgrade.py`](../examples/single_stage_optimization/banana_opt/coil_order_upgrade.py)

## Non-Goals

- Do not claim exact certified global Pareto optimality for this simulation stack.
- Do not refactor `thresholded_physics` and frontier together in the first pass.
- Do not add surrogate modeling until a cost gate says plain true-evaluation global search is too expensive.
- Do not relabel independent scalarized lanes as `MOEA/D`.
- Do not treat `NSGA-III` with one reference direction as equivalent to the current single frontier lane.

## Phase 1: Frontier Contract Hardening

Objective: remove known contract ambiguities and close small correctness-adjacent gaps without changing the overall engine model.

### 1.1 Epsilon contract hardening

- [ ] Make epsilon JSON schema fail loudly on unknown keys
- [ ] Make the parser reject unknown epsilon metric keys instead of silently ignoring them
- [ ] Keep certification limited to the currently implemented epsilon metrics until the schema is versioned for broader support

Rationale:

- Current certification only checks `qa_error` and `boozer_residual`, so the first fix should close the silent-ignore hole before expanding contract surface.

### 1.2 Lane-level scalarization parameters

- [ ] Move `_FRONTIER_CHEBYSHEV_SHARPNESS` into lane-level `scalarization_params`
- [ ] Move `_FRONTIER_EPSILON_PENALTY_WEIGHT` into lane-level `scalarization_params`
- [ ] Add CLI overrides
- [ ] Persist the values through rerun contracts, manifests, summaries, and resume/replay
- [ ] Add a replay round-trip test for non-default values

Rationale:

- These are scalarization semantics, not runtime-calibration knobs. The current leaks are specifically the private module globals `_FRONTIER_CHEBYSHEV_SHARPNESS` and `_FRONTIER_EPSILON_PENALTY_WEIGHT`, not the existing per-lane `scalarization_params` that already survive rerun and replay. `_FRONTIER_CHEBYSHEV_SHARPNESS` is consumed in the smooth-Chebyshev scalarization path, and `_FRONTIER_EPSILON_PENALTY_WEIGHT` is consumed in `_frontier_excess_penalty()`.

### 1.3 Recommendation gate centralization

- [ ] Define explicit gate rules per recommendation policy
- [ ] Keep current certification semantics for missing values:
  - trust missing -> not automatically unsafe
  - hardware missing -> unsafe
- [ ] Remove duplicated inline missing-value policy from individual recommendation functions

Rationale:

- The asymmetry is currently real and intentional. The fix is to centralize it, not to erase it. Current anchors: `banana_opt/frontier_recommendation.py:72` treats missing `frontier_trust_ok` as eligible, while `banana_opt/frontier_recommendation.py:104` treats missing `hardware_constraints_ok` as ineligible.

### 1.4 Frozen scalarization normalization

- [ ] Extend the existing manifest-level frozen scalarization-normalization contract instead of replacing it
- [ ] Keep the current seed-relative normalization kind as the default baseline
- [ ] Add additional normalization kinds behind an explicit kind selector
- [ ] Add optional pre-pass ideal/nadir normalization for campaigns that want the extra upfront cost
- [ ] Record the selected normalization method in the manifest
- [ ] Do not derive normalization adaptively from the live archive

Rationale:

- The frozen normalization contract is partially in place already; the manifest already freezes `kind` and `metric_rules`, and validation rejects metric-rule drift. The missing work is broader kind support plus a real ideal/nadir path. Adaptive archive-derived renormalization remains disallowed because it breaks reproducibility and can change scalarization meaning mid-campaign.

### 1.5 Full-simplex auto-generated achievement mode

- [ ] Add a new auto-generated achievement/Chebyshev mode using full-simplex reference directions
- [ ] Support Das-Dennis or equivalent reference-direction generation
- [ ] Leave legacy `shared` mode unchanged for backward compatibility
- [ ] Make the new mode the recommended path for true 4-objective exploration without handwritten JSON

Rationale:

- The current shared mode does not expose a true 4-axis lane family because only `iota_share` / `volume_share` vary lane-to-lane in `banana_opt/frontier_engine_multilane_local.py:63-75`; `effective_qs_weight=1.0` is fixed in `SINGLE_STAGE/single_stage_banana_example.py:2280`; and Boozer stays constant across shared lanes because `res_weight` is constant in the shared lane generator and then normalized in `SINGLE_STAGE/single_stage_banana_example.py:2281-2283`.

### 1.6 Scalarization/archive scale reconciliation

- [ ] Either unify scalarization scales and archive-distance scales under one documented contract
- [ ] Or document the intentional divergence and why duplicate detection / balanced scoring want different scaling

Trigger:

- This must land in the same milestone as 1.5 or be explicitly documented as an intentional divergence.

### Phase 1 exit criteria

- [ ] No epsilon key can be silently ignored
- [ ] Non-default scalarization params survive resume/replay unchanged
- [ ] Recommendation gating policy lives in one place
- [ ] New campaigns record frozen scalarization normalization explicitly
- [ ] A full-simplex achievement-mode path exists without breaking legacy shared mode
- [ ] The reference-direction generator used by the new achievement mode is reusable by the first global engine instead of being reimplemented a second time

## Phase 2: Evaluator Seam Extraction

Objective: create a deterministic, serializable evaluator suitable for population algorithms, caching, and distributed execution.

### 2.1 Decision-variable contract

- [ ] Define exactly which DOFs participate in global search
- [ ] Define bounds and whether local/global modes use the same bound contract
- [ ] Persist semantic role metadata for each DOF:
  - `phic(k)`
  - `phis(k)`
  - `thetac(k)`
  - `thetas(k)`
- [ ] Define structural invariants operators must preserve:
  - stellarator symmetry
  - `nfp` coupling
  - coil ordering / semantic rebinding invariants

Rationale:

- The `x` vector cannot be allowed to emerge implicitly from whichever module gets written first. The semantic DOF vocabulary already exists; this phase is about formalizing and persisting that contract rather than discovering it from scratch.

### 2.2 Serializable evaluator spec

- [x] Define `SingleStageFrontierEvaluatorSpec`
- [x] Add `from_spec()` constructor
- [x] Require that evaluator instances depend only on the spec and explicit runtime inputs
- [x] Remove dependence on live module globals from the evaluation path
- [ ] Make evaluator instances pickle-safe / serializable

Acceptance criterion:

- [x] A serialized evaluator spec can be re-instantiated in a fresh Python process and produce the same outputs for the same inputs

### 2.3 Fatal init vs per-candidate invalid separation

- [x] Add an explicit fatal initialization error type for bad specs or missing artifacts
- [x] Return `valid=False` plus diagnostics for candidate-specific failures
- [x] Never blur “campaign misconfigured” with “candidate infeasible / invalid”

### 2.4 Determinism contract

- [x] Define canonical fresh surface initialization per call
- [x] Use a fixed stored seed surface state as the initial guess for every evaluation unless the contract explicitly says otherwise
- [x] Do not reuse mutable state from the previous candidate evaluation
- [ ] Only introduce an evaluator RNG seed if a stochastic subroutine is actually admitted into the evaluation path

Rationale:

- The proven current determinism risk is hidden mutable surface/Boozer state, not necessarily RNG.

### 2.5 Cache and CV contract

- [x] Add evaluation caching keyed by the explicit evaluator contract plus candidate vector
- [x] Define `evaluate_batch(X)` semantics:
  - split cache hits and misses
  - evaluate misses only
  - merge results back in original order
- [x] Start with per-worker caching plus shared persistence, not shared mutable in-memory cache
- [x] Map current invalidation reasons into real-valued CV buckets

Suggested initial invalidation buckets:

- [x] `surface_solve_failed`
- [x] `geometry_state_unrestorable`
- [x] `missing_search_eval`
- [x] `nonfinite_evaluation`
- [x] topology-broken equivalent failures

### 2.6 Shadow-run validation

- [x] Run the extracted evaluator side-by-side against the current inline path
- [ ] Define a concrete sample budget before opening Phase 2 implementation tickets:
  - `N_seed_baselines`
  - `N_lane_optima`
  - `N_checkpoint_mid_iterates`
  - `N_known_failures`
  - `K_perturbations_per_point`
- [ ] Cover:
  - Stage 2 seed DOFs
  - prior frontier lane optima
  - intermediate lane checkpoint iterates
  - known-failed points
  - small perturbations around each category
- [ ] Define agreement tolerances before execution:
  - objective-metric tolerances
  - gradient tolerance, if gradients are compared
  - exact failure-path agreement criteria for invalid points
- [ ] Make parity scope explicit before execution:
  - objective / constraint parity is mandatory
  - gradient parity is required only if the extracted evaluator exposes gradients
- [ ] Validate both successful outputs and failure-path agreement

### Phase 2 exit criteria

- [ ] Decision-variable schema is explicit and documented
- [x] Evaluator is serializable and re-instantiable from spec
- [ ] Determinism holds for fixed spec plus candidate input
- [x] Cache semantics are explicit
- [ ] Shadow-run agreement passes on a mixed sample including failures

## Phase 3: First Global Engine

Objective: add the first true global Pareto engine with the smallest reasonable architectural lift.

### 3.1 Add `NSGA-III` first

- [ ] Add `nsga3` under `--frontier-engine`
- [ ] Keep `multilane_local` intact as the baseline engine
- [ ] Reuse the same objective and certification contracts as far as possible

### 3.2 Archive remains SSOT

- [ ] After each generation, propose certified non-dominated population members to `update_frontier_archive`
- [ ] Keep the frontier archive as the project source of truth for Pareto membership
- [ ] Persist population checkpoints separately from archive checkpoints

Rationale:

- The repo already has a tested archive update and replay seam. The first global engine should feed that seam, not replace it.

### 3.3 Domain-aware operators and repair

- [ ] Implement custom variation and/or repair that preserves:
  - stellarator symmetry subspaces
  - `nfp` lockstep structure
  - semantic coil-order invariants
- [ ] Add tests that operator output still satisfies those invariants
- [ ] Avoid treating default generic real-vector operators as sufficient

### 3.4 Telemetry and observability

- [ ] Persist per-generation hypervolume
- [ ] Persist CV distribution
- [ ] Persist failure-reason histograms
- [ ] Persist cache hit/miss stats
- [ ] Persist generation-level archive growth and certification counts

### 3.5 Seed-aware initialization

- [ ] Seed the initial population from validated Stage 2 / prior frontier artifacts where available
- [ ] Support ingesting an external validated seed bank if one is curated later
- [ ] Avoid starting from naive random DOFs as the only initialization path
- [ ] Freeze the seed corpus and initialization recipe before any `multilane_local` vs `nsga3` comparison is used for a go/no-go decision

### 3.6 Quantitative go/no-go gate

- [ ] Do not start the benchmark comparison until the seed corpus and initialization recipe are frozen
- [ ] Fix the true-evaluation budget before running the comparison
- [ ] Fix a minimum hypervolume improvement threshold over `multilane_local`
- [ ] Fix a minimum archive diversity threshold
- [ ] Fix a wall-clock budget
- [ ] Fix a reproducibility threshold across repeated runs

Example gate template:

- [ ] true-evaluation budget = `N`
- [ ] hypervolume improvement >= `target`
- [ ] non-dominated archive size >= `target`
- [ ] wall-clock <= `budget`

### Phase 3 exit criteria

- [ ] `nsga3` runs end-to-end through the existing archive path
- [ ] Telemetry is available for each generation
- [ ] The benchmark is judged against pre-committed thresholds, not post hoc intuition

## Phase 4: Cost-Driven Surrogate Escalation

Objective: escalate to surrogate assistance only if plain global true-evaluation search is too expensive.

### 4.1 Explicit cost trigger

- [ ] Estimate:
  - `eval_cost_per_candidate`
  - `target_pop`
  - `target_generations`
  - `available_cores`
  - `wall_clock_budget`
- [ ] Trigger surrogate exploration only when:

`eval_cost_per_candidate * target_pop * target_generations > wall_clock_budget * available_cores`

### 4.2 Surrogate guardrail

- [ ] Periodically re-evaluate top surrogate-ranked archive members with the true evaluator
- [ ] If true-vs-surrogate divergence exceeds threshold:
  - demote the affected members
  - retrain or refresh the surrogate
  - record the divergence event

### 4.3 Candidate frameworks

- [ ] Evaluate `ParMOO` / `libEnsemble` only after Phase 3 numbers exist
- [ ] Keep domain-specific surrogate design as a separate work item

### Phase 4 exit criteria

- [ ] Surrogates are introduced only if the cost trigger is hit
- [ ] True-evaluation audit is mandatory for surrogate-led archive decisions

## Phase 5: Decomposition Engine Follow-On

Objective: consider constrained decomposition only after the first global engine is in place and benchmarked.

### 5.1 Constrained `MOEA/D` design note

- [ ] Pick a constrained decomposition variant before implementation
- [ ] Pick an implementation path only after verifying backend capabilities
- [ ] Do not assume an off-the-shelf backend already matches the required constrained behavior

Candidate variant families to evaluate:

- [ ] CMOEA/D
- [ ] MOEA/D-CDP
- [ ] MOEA/D-SR

Candidate implementation paths to evaluate:

- [ ] custom in-repo implementation
- [ ] fork/adapt an existing library implementation
- [ ] wrap a backend only if its actual constraint semantics match requirements

### 5.2 Hybrid local refinement

- [ ] Add local refinement of top global-engine archive members using the current local frontier machinery
- [ ] Measure whether hybrid refinement improves true archive quality relative to global-only results

### Phase 5 exit criteria

- [ ] A constrained decomposition path is chosen by design note, not by assumption
- [ ] Hybrid local refinement shows measurable value on true-evaluation metrics

## Deferred Items and Triggers

### Hypervolume backend replacement

Deferred because:

- current archive sizes are still small enough for the in-tree implementation

Trigger:

- [ ] replace when contribution annotation exceeds about 1 second per lane/generation completion in integration runs

### `thresholded_physics` frontier compatibility

Deferred because:

- this is a formulation project, not a reorder/refactor of the current objective assembly

Trigger:

- [ ] open a dedicated design note once global-engine work is stable enough to justify a second objective-contract expansion

### Exact certified global Pareto

Deferred because:

- exact global certification is out of scope for this simulation stack and cost profile

## Why `NSGA-III` First

`NSGA-III` is chosen as the **first** global engine for this repo because it is the best fit for the current architecture, not because it is universally superior.

### Main reasons

- It is reference-direction based, which matches the planned full-simplex frontier-direction work.[^ext-pymoo-nsga3]
- It fits `pymoo`'s general feasibility-first / CV-based constraint handling model.[^ext-pymoo-constraints]
- It requires less architectural lift from the current lane/archive system than neighborhood-coupled `MOEA/D`.
- Official `pymoo` `MOEA/D` still asserts no constraint support in `_setup()`.[^ext-pymoo-moead]
- `pygmo` currently documents `moead` as multi-objective unconstrained (`M-U`), so it should not be treated as an already-validated constrained drop-in path.[^ext-pygmo-overview]

### What this does not mean

- It does not mean `NSGA-III` is the final engine.
- It does not mean `MOEA/D` is a bad fit forever.
- It does not mean `NSGA-III` with one reference direction is equivalent to a current frontier lane.
- It does not remove the need for surrogate assistance if evaluation cost is too high.

## Risks and Watch Points

- [ ] Seed quality may dominate first-pass global-engine results more than algorithm choice
- [ ] Evaluator extraction can expand in scope once hidden global-state dependencies are removed
- [ ] Domain-aware variation may become the true bottleneck after evaluator extraction
- [ ] Archive-scale vs scalarization-scale divergence can become a future maintenance trap if it is not documented
- [ ] Surrogate optimism can corrupt archive quality unless true-evaluation audits are mandatory

## Resources

### Repo-local resources

- [`run_single_stage_frontier_campaign.py`](../examples/single_stage_optimization/run_single_stage_frontier_campaign.py): frontier engine selection, lane execution, archive updates
- [`frontier_engine_base.py`](../examples/single_stage_optimization/banana_opt/frontier_engine_base.py): lane contracts, progress format, archive replay
- [`frontier_archive.py`](../examples/single_stage_optimization/banana_opt/frontier_archive.py): archive membership, epsilon certification, hypervolume plumbing
- [`frontier_recommendation.py`](../examples/single_stage_optimization/banana_opt/frontier_recommendation.py): current policy-specific gate asymmetry
- [`frontier_dominance.py`](../examples/single_stage_optimization/banana_opt/frontier_dominance.py): certification contract and Pareto semantics
- [`frontier_constraints.py`](../examples/single_stage_optimization/banana_opt/frontier_constraints.py): invalidation reason taxonomy
- [`frontier_solver_checkpoint.py`](../examples/single_stage_optimization/banana_opt/frontier_solver_checkpoint.py): checkpoint serialization and restored-incumbent seam used in determinism discussions
- [`single_stage_banana_example.py`](../examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py): current stateful evaluation path and search logic
- [`single_stage_geometry.py`](../examples/single_stage_optimization/banana_opt/single_stage_geometry.py): surface-state snapshot/restore and topology gate bridge
- [`single_stage_objectives.py`](../examples/single_stage_optimization/banana_opt/single_stage_objectives.py): in-process objective evaluation building blocks
- [`topology_scorer.py`](../examples/single_stage_optimization/topology_scorer.py): deterministic midplane topology seeding logic
- [`coil_order_upgrade.py`](../examples/single_stage_optimization/banana_opt/coil_order_upgrade.py): semantic DOF naming and symmetry-aware rebuild logic

### External resources

- [ ] `pymoo` NSGA-III docs and source
- [ ] `pymoo` constraint-handling docs
- [ ] `pymoo` MOEA/D source
- [ ] `pygmo` algorithm capability overview
- [ ] smooth Tchebycheff / achievement scalarization reference used in prior frontier validation

## Validation Todo for This Plan

- [ ] Keep fast-feedback tests for frontier contracts and archive behavior green while Phase 1 lands
- [ ] Add replay round-trip tests for non-default scalarization params
- [ ] Add evaluator determinism property tests in Phase 2
- [x] Add shadow-run agreement tests before enabling any global engine by default
- [ ] Add engine-comparison benchmark harness for Phase 3 go/no-go review

## Citations

[^ext-pymoo-moead]: Official `pymoo` `MOEA/D` source. `_setup()` asserts: the implementation does not support constrained problems. <https://pymoo.org/_modules/pymoo/algorithms/moo/moead.html>

[^ext-pymoo-nsga3]: Official `pymoo` `NSGA-III` documentation. `NSGA-III` is reference-direction based. <https://pymoo.org/algorithms/moo/nsga3.html>

[^ext-pymoo-constraints]: Official `pymoo` constraint-handling docs. Most algorithms in the framework use feasibility-first / CV-based handling. <https://www.pymoo.org/constraints/index.html>

[^ext-pygmo-overview]: Official `pygmo` overview. `moead` is currently classified as `M-U` in the capabilities table. <https://esa.github.io/pygmo2/overview.html>

[^ext-lin2024]: Lin et al., smooth Tchebycheff / achievement scalarization reference used during frontier validation. <https://arxiv.org/html/2402.19078v3>
