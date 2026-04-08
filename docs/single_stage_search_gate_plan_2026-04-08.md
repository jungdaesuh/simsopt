# Single-Stage Search Gate Plan

Date: 2026-04-08
Status: Follow-on plan after baseline hardware-search policy landing
Scope: `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py` and supporting `banana_opt/` modules

## Goal

Replace the current single-stage search-time hardware gate with a deterministic, replayable policy that:

- hard-rejects broken / untrustworthy states
- treats modeled infeasibility quantitatively
- supports a tiny rollback-protected soft traversal path for penalty mode
- keeps final certification hard
- stays aligned with the existing Stage 2 / ALM direction without pretending single-stage already has full restoration-phase behavior

## Already Landed

The baseline single-stage hardware-search policy seam is already implemented and committed:

- [x] Pure search-policy module exists in `banana_opt/single_stage_search_policy.py`
- [x] Single-stage CLI exposes:
  - `--hardware-search-mode`
  - `--hardware-search-soft-iterations`
- [x] Results metadata records:
  - `HARDWARE_SEARCH_MODE`
  - `HARDWARE_SEARCH_SOFT_ITERATIONS`
- [x] Search bookkeeping already distinguishes trial and accepted hardware status
- [x] Final hard certification remains in place

This document now tracks the remaining deeper adaptive-gate work beyond that landed baseline.

## Why This Is Needed

The current single-stage gate and the recent discussions exposed several issues:

- The existing adaptive seam is intentionally narrow.
  - `single_stage_search_policy.py` currently uses an accepted-iteration window and `gate_scale < 1.0`.
  - That is a pragmatic soft window, not a principled infeasibility policy with rollback/restoration.
- Broken states and modeled infeasibility need to be separated.
  - Boozer collapse, self-intersection, NaN / inf, invalid geometry state, and invalid topology evaluation are not useful search states.
  - Hardware-threshold misses and normal topology-gate failures are modeled infeasibility, not automatic corruption.
- Single-stage and Stage 2 do not currently model exactly the same hard constraints.
  - Single-stage hardware snapshot currently includes:
    - coil-coil distance
    - coil-surface distance
    - surface-vessel distance
    - curvature
  - Stage 2 explicit hard constraints currently include:
    - length
    - coil-coil distance
    - curvature
  - So the divergence is symmetric, not “Stage 2 is a superset”.
- Replay evidence showed that blanket warning-only traversal is not a safe general policy.
  - Hard mode preserved at least one archived feasible solution.
  - Weak soft traversal could drift into a different hardware-invalid basin.
- Micro-scan evidence still supports a narrow soft corridor.
  - Tiny infeasible near-passes can sit next to valid better families.
  - That supports a tiny, deterministic, rollback-protected soft traversal rule.

## Design Target

Use a 3-way state model:

1. `broken`
   - hard reject always
2. `modeled_infeasible`
   - policy-managed
3. `feasible`
   - normal accept path

Use accepted-state comparison for acceptance and best-feasible only for rollback / incumbent replacement.

Use soft budget counted in accepted infeasible transitions, not raw optimizer iterations.

Use outer-loop chunked `L-BFGS-B` only for `adaptive` mode.

- Keep `hard` mode as the current single `minimize()` call.
- In `adaptive` mode, use chunk boundaries to decide whether to continue from the current accepted endpoint, roll back to the best feasible incumbent, or terminate.
- Inside each chunk, keep `fun()` on the existing hard-reject path so infeasible-side gradients do not contaminate the quasi-Newton history.

## Non-Goals

- Do not add an LLM to the live gate.
- Do not ship permanent warning-only search as the default policy.
- Do not add a second ad hoc hardware loss on top of the existing objective / ALM structure.
- Do not treat topology-gate rejection as broken unless the topology evaluator itself is invalid.
- Do not assume current ALM already provides full mid-run restoration semantics in single-stage.
- Do not pay chunking overhead in default `hard` mode.
- Do not support `adaptive` with basin-hopping in v1.

## Current Relevant Files

- `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py`
- `examples/single_stage_optimization/banana_opt/single_stage_search_policy.py`
- `examples/single_stage_optimization/banana_opt/single_stage_geometry.py`
- `examples/single_stage_optimization/banana_opt/single_stage_objectives.py`
- `examples/single_stage_optimization/banana_opt/stage2_objectives.py`
- `examples/single_stage_optimization/alm_utils.py`
- `tests/geo/test_single_stage_example.py`
- `tests/geo/test_single_stage_alm_integration.py`
- `tests/geo/test_banana_objective_modules.py`

## Implementation Principles

- KISS: one small deterministic policy seam, using the existing two-phase `L-BFGS-B` restart pattern as precedent for adaptive chunking
- YAGNI: no generic multi-optimizer policy framework yet
- DRY: one source of truth for state classification and acceptance logic
- SSOT:
  - one source of truth for single-stage hardware snapshot
  - one source of truth for search-time gate policy
  - one source of truth for final hard certification
  - one shared feasible-incumbent abstraction across penalty-mode adaptive control and ALM
- Functional core:
  - pure classification and decision helpers
  - mutable solver shell only where needed

## Optimizer Ownership

- [ ] Keep `hard` mode on one `minimize(..., method="L-BFGS-B")` call
- [ ] Implement `adaptive` mode as outer-loop chunked control only
- [ ] Inside each adaptive chunk:
  - keep `fun()` on the hard-reject path
  - do not expose warning-only / softened gradients to the live quasi-Newton state
- [ ] Treat soft traversal as a next-chunk seeding decision, not as a change to `fun()` return values inside a chunk
- [ ] Reuse the existing phase1/phase2 handoff pattern in `single_stage_banana_example.py` as the restart template

## Required Decisions

### A. Length Semantics

- [ ] Decide whether single-stage `coil length` is:
  - a hard engineering constraint
  - or a soft objective term only
- [ ] If `length` becomes hard in single-stage:
  - add it to the search-time hardware snapshot
  - add it to final certification / results status
  - align tests and docs with that contract
- [ ] If `length` stays soft-only:
  - keep it out of hard hardware status
  - stop describing it as part of the single-stage hard gate

### B. Topology Semantics

- [ ] Keep normal topology-gate failure as `modeled_infeasible`
- [ ] Define and document what makes topology evaluation itself `broken`
  - tracing raises
  - invalid / nonsensical tracing output shape
  - corrupted evaluator state
  - NaN / inf in topology result

## State Model Todo

- [ ] Add an explicit 3-way state classifier:
  - `broken`
  - `modeled_infeasible`
  - `feasible`
- [ ] Classify as `broken` when any of these occur:
  - Boozer solve failure
  - NaN / inf in objective, gradients, or geometry metrics
  - self-intersection
  - invalid nesting / corrupted surface stack
  - topology evaluation raises
  - invalid topology evaluator output
- [ ] Classify as `modeled_infeasible` when:
  - hardware thresholds fail but evaluation is trustworthy
  - topology gate rejects candidate with valid output
- [ ] Classify as `feasible` only when modeled constraints pass

## Feasibility Metrics Todo

- [ ] Compute a normalized feasibility vector for the explicit single-stage hard constraint set
- [ ] Initial explicit set must match current single-stage hard gate exactly unless length is promoted:
  - coil-coil distance
  - coil-surface distance
  - surface-vessel distance
  - curvature
- [ ] If length is promoted, add normalized length violation to the same vector
- [ ] Define one explicit normalization convention and reuse existing ALM residual semantics where possible
  - lower-bound style residuals for minimum-distance constraints
  - upper-bound style residuals for curvature / length-style constraints
- [ ] Derive:
  - `max_violation`
  - per-constraint normalized residuals for reporting

## Acceptance Policy Todo

### Penalty Mode

- [ ] Replace current iteration-window adaptive policy with accepted-state-based policy
- [ ] Compare candidate against the last accepted state for acceptance
- [ ] Use best-feasible incumbent only for:
  - rollback
  - incumbent replacement
- [ ] Count soft budget in accepted infeasible transitions
- [ ] Add `adaptive` chunk controls:
  - `adaptive_chunk_maxiter`
  - chunk termination reason
  - remaining maxiter budget across chunks
- [ ] Add tiny deterministic soft traversal rule:
  - candidate must be trustworthy
  - candidate violation must stay below small cap
  - candidate must either:
    - improve feasibility relative to accepted state
    - or improve objective with only tiny feasibility worsening
- [ ] Reject otherwise
- [ ] Treat vanishing step size / step stall as a rollback-or-stop trigger, not an automatic continue signal

### ALM Mode

- [ ] Keep ALM wrapper thin
- [ ] In `constraint_method="alm"`:
  - hard-reject broken states
  - only use a severe-cap reject if it is explicitly defined in ALM-native feasibility units
    - reuse ALM's own feasibility values / max-feasibility-violation semantics
    - do not introduce a wrapper-only normalization that can disagree with the ALM subproblem
  - otherwise let ALM manage ordinary temporary infeasibility
- [ ] Document accurately that current ALM best-feasible restore is end-of-run behavior, not full mid-run restoration phase
- [ ] Extend ALM incumbent capture through a solver-owned snapshot hook rather than importing Boozer-specific state into `alm_utils.py`

## Rollback Todo

- [ ] Introduce one shared feasible-incumbent abstraction in `banana_opt/` for both penalty-mode adaptive control and ALM
  - `x`
  - solved surface / Boozer state
  - evaluation snapshot
  - hardware status
  - topology status
  - optional ALM multipliers / penalty
- [ ] Track best feasible incumbent explicitly in single-stage penalty mode through that shared abstraction
- [ ] Track accepted-state metrics separately:
  - accepted merit
  - accepted max violation
  - accepted infeasible transition count
  - nonimproving infeasible transition count
- [ ] Roll back on:
  - severe infeasibility
  - soft budget exhaustion
  - repeated accepted infeasible transitions without feasibility improvement
- [ ] Restore full solved state on rollback, not only `x`
  - DOFs alone are not the full state in this workflow
- [ ] Define rollback concretely as:
  - restore the full incumbent state
  - restart the next adaptive chunk with fresh quasi-Newton history
- [ ] Fix the same full-state restore defect in current ALM end-of-run incumbent restore

## Code Split Todo

- [ ] Extend `single_stage_search_policy.py` from the landed baseline seam into the real search-time decision logic
- [ ] Keep policy core pure:
  - config dataclasses
  - classification inputs
  - decision outputs
- [ ] Keep mutable solver operations in `single_stage_banana_example.py`
- [ ] Reuse `single_stage_geometry.py` as the hardware-snapshot SSOT
- [ ] Reuse existing objective/ALM outputs rather than inventing a second physics loss
- [ ] Add a shared incumbent module under `banana_opt/` rather than growing `alm_utils.py` with Boozer-specific state
- [ ] Keep state snapshot / restore boundaries solver-owned
  - `alm_utils.py` may store generic incumbent payloads or accept solver-owned snapshot hooks
  - Boozer / solved-surface mutation logic must remain in the solver layer and `banana_opt/`, not inside `alm_utils.py`
- [ ] Ensure final-state handoff is explicit:
  - `final_x = incumbent.x if rolled_back else res.x`
  - final surface-stack solve and hard certification must use that same chosen endpoint

## Observability Todo

- [ ] Add first-class gate trace fields for replay tuning
  - accepted infeasible transition count
  - accepted-step max violation
  - rollback count
  - rollback reasons
  - chunk termination reason
  - current gate mode
- [ ] Thread new adaptive tunables into run identity and results payload
  - violation cap
  - soft budget
  - chunk maxiter

## Testing Todo

### Unit / Focused Tests

- [ ] Add tests for 3-way state classification
- [ ] Add tests that broken states always reject
- [ ] Add tests that topology-gate rejection is modeled infeasibility unless topology evaluation is invalid
- [ ] Add tests that accepted-state comparison, not previous-trial comparison, drives adaptive acceptance
- [ ] Add tests that best-feasible is used for rollback, not primary acceptance
- [ ] Add tests that rollback restores full solved surface state
- [ ] Add tests that ALM mode does not reuse the penalty adaptive seam
- [ ] Add tests that the shared feasible-incumbent abstraction is used by both penalty adaptive control and ALM restore paths
- [ ] Add tests for the final-state invariant:
  - final certification runs against the rolled-back incumbent when rollback happened
  - `res.x` and accepted/final state cannot silently diverge

### Replay / Boundary Tests

- [ ] Add replay-based tests or replay harnesses for real archived boundary behavior
- [ ] Cover:
  - tiny infeasible near-pass
  - loose valid family
  - broken Boozer-init collapse
- [ ] Validate that the gate can distinguish:
  - useful tiny infeasible boundary candidates
  - catastrophic untrustworthy states
- [ ] Add a concrete row-470-style pass criterion:
  - no vanishing step-size loop
  - clean termination within configured budget
  - final accepted state hardware-feasible to archived tolerance

## Validation Todo

- [ ] `py_compile` on touched files
- [ ] `git diff --check`
- [ ] Focused single-stage unit / module tests
- [ ] `tests/geo/test_single_stage_example.py`
- [ ] `tests/geo/test_single_stage_alm_integration.py`
- [ ] Replay validation on real archived runs before changing defaults
- [ ] Validate deterministic chunk accounting and run-identity changes when adaptive tunables change

## Upstream / Downstream Contracts

- [ ] Keep wrapper-owned workflow policy separate from solver search policy
  - wrappers decide lane identity and artifact compatibility
  - solver decides search-time acceptance on normalized inputs
- [ ] Preserve current-contract SSOT in `banana_opt/current_contracts.py`
  - public `--plasma-current-A`
  - expert/internal `--boozer-I`
- [ ] Preserve artifact-contract SSOT in `banana_opt/artifact_contracts.py`
  - strict validator stays strict
  - only wrapper-owned flows may apply unambiguous legacy upgrades
- [ ] Keep downstream results/schema alignment explicit when adaptive behavior changes
  - update emitted fields
  - update replay tooling / docs
  - update focused tests that assert hardware-status reporting

## Proposed Implementation Order

- [ ] Step 1: settle single-stage length semantics
- [ ] Step 2: add 3-way state classification
- [ ] Step 3: add shared feasible-incumbent abstraction under `banana_opt/`
- [ ] Step 4: add accepted-state-based normalized feasibility metrics and accepted-state tracking
- [ ] Step 5: replace current adaptive soft-window seam with chunked adaptive control using those metrics
- [ ] Step 6: add best-feasible full-state rollback for penalty mode
- [ ] Step 7: fix ALM end-of-run incumbent restore to use the shared full-state incumbent
- [ ] Step 8: add replay-based boundary validation
- [ ] Step 9: replay-tune caps, budgets, and chunk size
- [ ] Step 10: only then consider more advanced filter/restoration refinements

## Recommended Defaults

- [ ] Keep `hard` as default mode
- [ ] Keep `warn` as debug / replay only
- [ ] Keep `adaptive` as penalty-only and experimental until replay evidence justifies broader use
- [ ] Keep final certification hard in all modes
- [ ] Keep `hard` mode on a single `minimize()` call
- [ ] Disallow `adaptive` with basin-hopping in v1

## Success Criteria

- [ ] Broken states are never consumed as soft traversal
- [ ] Modeled infeasibility is handled deterministically
- [ ] Soft traversal is tiny, quantitative, and rollback-protected
- [ ] Single-stage constraint semantics are internally consistent
- [ ] ALM path is not double-gated informally
- [ ] Adaptive mode does not contaminate quasi-Newton history with softened in-chunk gradients
- [ ] Penalty-mode adaptive control and ALM share one full-state incumbent contract
- [ ] Replay evidence supports the chosen caps / budget
- [ ] Replay evidence shows row-470-style stalls terminate cleanly
- [ ] Final endpoint certification remains strict

## Notes

- This plan supersedes the narrower checklist in `docs/single_stage_hardware_search_policy_todos_2026-04-08.md` for the actual adaptive-gate implementation discussion.
- The main architectural target is:
  - hard reject broken states
  - quantitative handling for modeled infeasibility
  - deterministic rollback
