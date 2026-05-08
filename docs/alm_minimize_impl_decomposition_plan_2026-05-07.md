# ALM `_minimize_alm_impl` Decomposition Plan

Date: 2026-05-07

Status: implemented and validated. Phases 1a-Final are implemented in
the current worktree. The baseline facts below describe the
pre-implementation starting point unless explicitly labeled current.

## Current Implementation Status

- [x] Phase 1a extracted `_build_alm_history_entry`.
- [x] Phase 1 extracted `_normalize_alm_run_inputs`.
- [x] Phase 2 extracted `_handle_alm_penalty_cap_termination`.
- [x] Phase 3 extracted `_handle_alm_dual_update_transition`; the
  direct helper test now asserts the exact projected multiplier values.
- [x] Phase 4 extracted `_run_alm_continuation_step` and
  `_run_alm_outer_iteration`.
- [x] Phase 5 collapsed `_minimize_alm_impl` to the normalized driver
  and verified the <= 500 LOC / closure-free structural gate.
- [x] Phase Final simplification pass completed; the final structural
  metrics and closeout notes below match the current worktree.

Current structural metric after Phase Final:
`_minimize_alm_impl` is 127 LOC with zero nested function definitions;
`_run_alm_outer_iteration` is 93 LOC with zero nested function
definitions; `_run_alm_continuation_step` is 622 LOC with zero nested
function definitions.

Scope source: pre-existing structural debt under
`docs/alm_hardening_engineering_followup_todo_2026-05-07.md` Backlog 5
post-Phase-6 closeout. The Backlog 5 plan
(`docs/alm_backlog5_structural_debt_plan_2026-05-07.md`) shrank
`minimize_alm` to 32 lines and pushed the body into
`_minimize_alm_impl`, which then stood at 1,040 lines with 0 nested
closures and violated the 500-line target the structural-debt
acceptance criteria implied for the orchestration layer.

## Baseline Tree Facts

- `examples/single_stage_optimization/alm_utils.py:2971-4010` defines
  `_minimize_alm_impl` as a 1,040-line orchestrator with 0 nested
  function definitions (AST-confirmed via the same script the Backlog 5
  plan ships in its Validation Matrix).
- `examples/single_stage_optimization/alm_utils.py:4013-4044` defines
  the public `minimize_alm` as a 32-line keyword-forward shim. This
  surface is unchanged by this plan.
- 5 state-carrier dataclasses are already exposed by Backlog 5
  Phase 1-3. AST verification confirms `ALMRunState` is **mutable**
  (`@dataclass` without `frozen=True`) because it accumulates per-call
  history and counters; the other four are frozen:
  - `ALMRunState` (`alm_utils.py:79`) — mutable per-call aggregate.
  - `ALMFinalState` (`alm_utils.py:117`) — frozen.
  - `ALMHistoryEntry` (`alm_utils.py:137`) — frozen.
  - `ALMInnerAttemptRequest` (`alm_utils.py:155`) — frozen.
  - `ALMInnerAttemptResult` (`alm_utils.py:175`) — frozen.
  New private carriers introduced by this plan must be frozen.
  `ALMRunState` retains its mutable status; helpers that consume it
  treat the carrier as a one-call accumulator.
- 77 private module-level helpers exist
  (`grep -cE '^def _' examples/single_stage_optimization/alm_utils.py`).
  The major ones consumed by `_minimize_alm_impl` and their spans:
  - `_run_alm_inner_attempts`: `alm_utils.py:2796-2968` (173 lines).
  - `_build_alm_result`: `alm_utils.py:2448-2642` (195 lines).
  - `_build_alm_failure_result_with_optional_restore`:
    `alm_utils.py:2700-2793` (94 lines).
  - `_apply_alm_penalty_increase`: `alm_utils.py:2374-2445` (72 lines).
  - `_evaluate_alm_penalty_state`: `alm_utils.py:2257-2306` (50 lines).
  - `_refresh_alm_history_for_penalty_update`:
    `alm_utils.py:2309-2371` (63 lines).
- `_minimize_alm_impl` body shape (top-level statements):
  - `alm_utils.py:2987-3020` validation gates (4 `If`).
  - `alm_utils.py:3000-3043` per-call state initialization (no helper
    surface, all assignments).
  - `alm_utils.py:3045-3969` outer-iteration `for` (924 lines, the bulk
    of the function).
  - `alm_utils.py:3971-3978` post-loop guard plus terminal-reason
    derivation.
  - `alm_utils.py:3979-4010` final
    `_build_alm_failure_result_with_optional_restore` return.
- The public `_run_alm_inner_attempts` boundary already absorbs the
  L-BFGS-B contract per Backlog 5 Phase 3 acceptance, so this plan does
  not move SciPy interaction.

## Objective

Reduce `_minimize_alm_impl` from 1,040 lines to at most 500 lines
without changing:

- ALM math (no tolerance/penalty schedule/trust-radius/dual-update
  policy edits).
- callback ordering or payload shape for `inner_callback`,
  `accepted_callback`, `outer_state_callback`, `history_callback`.
- result schema keys or numeric values produced by `_build_alm_result`
  and `_build_alm_failure_result_with_optional_restore`.
- restore-on-failure semantics for the best-feasible incumbent.
- `_make_fake_minimize` / SciPy `jac=True` combined contract preserved
  by `_run_alm_inner_attempts`.
- public `minimize_alm` signature, default values, or shim wiring at
  `alm_utils.py:4013-4044`.

End-state target: `_minimize_alm_impl` becomes a small driver that
sequences validation, per-call initialization, the outer-iteration
loop body extracted into one helper, and the terminal failure return.

## Non-Goals

- No changes to `_run_alm_inner_attempts`, `_apply_alm_penalty_increase`,
  `_build_alm_result`, or
  `_build_alm_failure_result_with_optional_restore` bodies. Those are
  inputs, not targets.
- No new public helpers and no changes to `minimize_alm`'s argument
  list.
- No new lazy imports, defensive `try/except`, or alternate solver
  paths.
- No bundled Stage 2 artifact-config refactor or runner-side changes.
- No changes to `tests/geo/test_alm_utils.py::AlmStructuralDebtTests`
  beyond the new acceptance bound (which this plan tightens).

## Requirements

### Functional Requirements

- Preserve every key in the ALM result dictionary, including
  `constraint_metadata`, `constraint_*`, `alm_*`, `history`, and
  termination fields.
- Preserve callback ordering and payload shape on every termination
  path. Specifically, the `_emit_alm_history_snapshot` call sites at
  `alm_utils.py:3192,3481,3590,3597,3639,3674,3776,3791,3825,3845,3893,3965`
  must execute in identical order with identical history-entry
  contents.
- Preserve best-feasible promotion at `alm_utils.py:3202-3214` and
  `alm_utils.py:3341-3360`, plus restore-on-failure invocation at every
  `_build_alm_failure_result_with_optional_restore` callsite.
- Preserve all six termination-reason strings emitted today:
  `converged`, `constraints_inactive_converged`,
  `constraints_inactive_stall`, `signal_mismatch_stall`,
  `penalty_cap_reached`, `plateau_stall`, plus the post-loop
  `_termination_reason_from_history(...)` path at `alm_utils.py:3974`.
- Preserve cap-binding indices accumulation at
  `alm_utils.py:3810-3811`, the conditional `outer_termination`
  attachment at `alm_utils.py:3589,3774,3823,3844,3963`, and the
  `is_final_outer` branch at `alm_utils.py:3968-3969`.

### Architecture Requirements

- Helper boundaries use frozen result/config carriers except for the
  explicit per-call mutable accumulators: `ALMRunState` owns cross-step
  run state, and `_ContinuationStepState` owns the sticky continuation
  locals inside one helper call. Pass `ALMSettings`, `ALMRunState`,
  `_ContinuationStepState`, `_ALMContinuationStepResult`, primitive
  scalars, and `np.ndarray` data; do not introduce new mutable lists or
  dicts beyond the existing `history` list and `history_entry` dict
  already mutable on this code path.
- The outer-iteration body must be expressed as one driver that owns:
  - per-iteration setup (`outer_state_callback`, stall counter init).
  - the continuation loop, dispatched into one continuation-step helper
    per iteration.
- The continuation-step helper must own one of the six observable
  outcomes per pass:
  - converged-on-entry.
  - converged-after-inner.
  - constraints-inactive (converged or stall).
  - signal-mismatch (stall, penalty-increase, or feasible plateau).
  - dual-update.
  - infeasible-penalty-increase, plateau-limit, or generic
    penalty-increase.
  Each outcome maps to one explicit branch tag in a frozen result
  dataclass. Tags must be a closed set: use a private string-valued
  `Enum` (`class _ALMContinuationDecision(str, Enum): ...`) over inline
  `Literal` unions for the discriminator field, so the set is
  grep-able and exhaustively dispatchable while staying compatible with
  the repo's Python >= 3.9 contract; ad-hoc free-form strings are
  forbidden.
- No nested closures and no dynamic imports.

### Performance And Safety Requirements

- No extra evaluator calls. The existing single
  `evaluate_problem(x, multipliers, penalty_argument)` per
  continuation iteration at `alm_utils.py:3055-3059` and the inner
  attempt evaluator inside `_run_alm_inner_attempts` are the only
  evaluator entry points and must remain so.
- No additional history-snapshot allocations. History entries are
  built in-place on the existing `history_entry` dict and snapshotted
  by `_emit_alm_history_snapshot`'s existing
  `_snapshot_history_entry` path; this plan does not change the
  snapshot contract.
- All run state must remain per-call. No module-level mutable state
  is introduced.
- Memory behavior for `settings.history_max_entries` is preserved by
  routing through the existing `_append_alm_history_entry` helper
  unchanged.

### Testability Requirements

- Phase 0 characterization tests must exist before the first code
  movement. They are independent of Backlog 5 Phase 0 tests; the
  reused fixtures from `tests/geo/test_alm_utils.py` are sufficient.
- Each new helper gets a direct unit test driven by stubbed
  `evaluate_problem` and a constructed `ALMRunState`.
- The structural assertion
  `tests/geo/test_alm_utils.py::AlmStructuralDebtTests::test_minimize_alm_public_entrypoint_is_small_and_closure_free`
  remains green; a new sibling assertion locks the
  `_minimize_alm_impl` 500-line bound.

## Decomposition Design

### New Helpers

#### `_normalize_alm_run_inputs`

- Signature: `(*, x0, constraint_names, constraint_blocks, settings,
  initial_multipliers, initial_penalty,
  snapshot_accepted_state_fn, restore_incumbent_state_fn)
  -> _ALMNormalizedRunInputs`.
- Returns a frozen `_ALMNormalizedRunInputs` dataclass holding the
  validated `x` copy, multiplier vector, initial penalty, the
  `(_constraint_names_tuple, _constraint_blocks_tuple)` tuple pair,
  initial `update_feasibility_tol` / `update_stationarity_tol`, and
  `trust_radius`.
- Owns the validation block at `alm_utils.py:2987-3020`.
- Owns the per-call state-initialization assigns at
  `alm_utils.py:3021-3043`.
- Expected span: 80-100 lines (validation messages preserved
  byte-for-byte; the existing `_build_constraint_metadata_tuples` and
  `_normalize_trust_radius` calls move with the block).

#### `_run_alm_continuation_step`

- Signature: `(*, settings, run_state, multipliers, penalty,
  feasible_stall_count, trust_radius, update_feasibility_tol,
  update_stationarity_tol, evaluate_problem, inner_options,
  outer_iteration, continuation_iteration, is_final_outer,
  best_feasible, snapshot_accepted_state_fn, accepted_callback,
  inner_callback, history_callback, history,
  history_truncated_count, last_result,
  total_inner_iterations, cap_binding_detected, cap_binding_indices,
  penalty_cap_reached, penalty_cap_requested,
  constraint_names, constraint_names_tuple, constraint_blocks_tuple)
  -> _ALMContinuationStepResult`.
- Owns the body of the continuation `for` loop currently spanning
  `alm_utils.py:3052-3966` for one continuation iteration.
- Returns a `_ALMContinuationStepResult` (frozen dataclass) with
  enough information to let the outer driver decide whether to
  `continue` the continuation loop, `break` to the next outer
  iteration, or short-circuit-return.
- Expected span: 320-380 lines (the continuation body collapses by
  factoring out the four already-existing penalty-cap return arms
  via the next two helpers).

#### `_handle_alm_penalty_cap_termination`

- Signature: `(*, settings, run_state, last_outer_iteration,
  best_feasible, restore_incumbent_state_fn, penalty_transition,
  multipliers_state, penalty_state, inner_result, message_prefix,
  restored_message_prefix, restored_termination_reason,
  termination_reason, constraint_names) -> dict`.
- Pure adapter that delegates straight into
  `_build_alm_failure_result_with_optional_restore`. The win is
  collapsing the three penalty-cap return blocks at
  `alm_utils.py:3548-3585`, `alm_utils.py:3731-3771`, and
  `alm_utils.py:3922-3960` into one shared call site.
- Expected span: 40-60 lines (mostly the keyword-argument forward).

#### `_handle_alm_dual_update_transition`

- Signature: `(*, settings, multipliers, routing_state,
  penalty_argument, history_entry,
  update_feasibility_tol, update_stationarity_tol,
  cap_binding_detected, cap_binding_indices) -> _ALMDualUpdateResult`.
- Owns the dual-update arm at `alm_utils.py:3794-3826`.
- Returns a frozen `_ALMDualUpdateResult` with the new
  `multipliers`, the post-update tolerance scales, and the cap-binding
  diagnostics. The driver mutates `history_entry` and
  `cap_binding_*` based on the return value, identically to today.
- Expected span: 30-40 lines.

#### `_run_alm_outer_iteration`

- Signature: `(*, settings, run_state, multipliers, penalty,
  trust_radius, update_feasibility_tol, update_stationarity_tol,
  outer_iteration, ...callbacks..., evaluate_problem,
  inner_options, history, history_truncated_count, last_result,
  total_inner_iterations, cap_binding_detected, cap_binding_indices,
  penalty_cap_reached, penalty_cap_requested, best_feasible,
  constraint_names, constraint_names_tuple, constraint_blocks_tuple,
  is_final_outer) -> _ALMOuterIterationResult`.
- Owns one full outer iteration: the `outer_state_callback` invocation
  at `alm_utils.py:3047-3048`, the continuation `for` loop at
  `alm_utils.py:3052-3966`, and the per-iteration accumulators.
- Internally drives `_run_alm_continuation_step` until either it
  returns a "stop outer iteration" decision or the continuation budget
  is exhausted.
- Expected span: 90-120 lines.

### New State Carriers

All new dataclasses are private, frozen, and live next to the existing
`ALMRunState` cluster at `alm_utils.py:79`.

#### `_ALMNormalizedRunInputs`

Fields: `x: np.ndarray`, `multipliers: np.ndarray`, `penalty: float`,
`active_penalty: float`,
`constraint_names_tuple: tuple[str, ...]`,
`constraint_blocks_tuple: tuple[str, ...] | None`,
`trust_radius: float | None`,
`update_feasibility_tol: float`,
`update_stationarity_tol: float`.

#### `_ALMOuterIterationResult`

Fields:
- `decision: _ALMOuterDecision` (private string-valued `Enum` with
  values `RETURN`, `NEXT_OUTER`, `EXHAUST`; not a free-form string).
- `result: object | None` (set when `decision == "return"`; carries the
  pre-built `_build_alm_result` or
  `_build_alm_failure_result_with_optional_restore` payload).
- `multipliers: np.ndarray`.
- `penalty: float`.
- `update_feasibility_tol: float`.
- `update_stationarity_tol: float`.
- `final_eval: dict | None`.
- `final_multipliers: np.ndarray`.
- `final_penalty: float`.
- `last_result: object | None`.
- `best_feasible: ALMFeasibleIncumbent | None`.
- `inner_options: dict | None`.

#### `_ALMContinuationStepResult`

Fields:
- `decision: _ALMContinuationDecision` (private string-valued `Enum` with
  values `RETURN`, `BREAK_OUTER`, `CONTINUE_CONTINUATION`; not a
  free-form string).
- `result: object | None`.
- All run-state scalars carried forward to the next continuation pass:
  `multipliers`, `penalty`,
  `update_feasibility_tol`, `update_stationarity_tol`,
  `feasible_stall_count`, `final_eval`, `final_multipliers`,
  `final_penalty`, `last_result`, `best_feasible`, and
  `inner_options`.

#### `_ALMDualUpdateResult`

Fields: `multipliers: np.ndarray`, `update_feasibility_tol: float`,
`update_stationarity_tol: float`, `multiplier_cap_binding: bool`,
`multiplier_cap_binding_indices: list[int]`.

The driver now carries scalar continuation state through result
dataclasses while `ALMRunState` remains the per-call mutable accumulator
for `x`, history, trust radius, and cap-binding bookkeeping. No new
module-level mutable structures are introduced; the result carriers are
frozen.

## Execution Plan

### Phase 0: Baseline Metrics And Characterization Tests

- [ ] Add `tests/geo/test_alm_utils.py::AlmStructuralDebtTests::test_minimize_alm_impl_baseline_metrics_are_pinned`,
  asserting `_minimize_alm_impl`'s AST node has the **current**
  metrics: `end_lineno - lineno + 1 == 1040` and zero nested
  `ast.FunctionDef`. This locks today's state without making future
  claims. The ≤500 bound is added in Commit 6 by the same commit that
  achieves it (no `expectedFailure` markers — they give false-green
  signals).
- [ ] Do **not** add a state-carrier-existence test in Phase 0. Add
  `_ALMNormalizedRunInputs`, `_ALMOuterIterationResult`,
  `_ALMContinuationStepResult`, `_ALMDualUpdateResult` existence
  assertions in the commit that introduces each, alongside its own
  direct unit test. This avoids deferred-test drift.
- [ ] Add a continuation-arm coverage harness:
  `tests/geo/test_alm_utils.py::ContinuationArmCharacterizationTests`
  with one focused test per arm so each Phase 1-3 helper has a
  pre-existing observable to lock against. Required arms:
  - `converged_on_entry` -- driven by an evaluator that returns a
    feasible iterate on entry, asserting the
    `_emit_alm_history_snapshot` call at `alm_utils.py:3192` fires
    once with `action == "converged"` and the
    `_build_alm_result` keys match the existing schema-snapshot.
  - `converged_after_inner` -- evaluator returns infeasible on entry
    and feasible after one inner attempt; asserts the
    `_emit_alm_history_snapshot` call at `alm_utils.py:3481` fires
    with `action == "converged"`.
  - `constraints_inactive_converged` -- routing state has explicit
    Stage 2 signals with zero hard violation and no surrogate
    activity; asserts emission at `alm_utils.py:3597` and
    `termination_reason == "constraints_inactive_converged"`.
  - `constraints_inactive_stall` -- same setup but second
    continuation iteration without progress; asserts emission at
    `alm_utils.py:3639` and termination reason
    `constraints_inactive_stall`.
  - `signal_mismatch_stall` -- evaluator emits hard-feasible with
    surrogate-active and `surrogate_positive_shift_zero=True` on
    repeat; asserts emission at `alm_utils.py:3674` and
    `termination_reason == "signal_mismatch_stall"`.
  - `signal_mismatch_penalty_increase` -- same as above but
    `surrogate_positive_shift_zero=False`; asserts emission at
    `alm_utils.py:3776` with action
    `signal_mismatch_penalty_increase` and the penalty schedule
    fires.
  - `dual_update` -- feasible iterate within
    `update_stationarity_tol`; asserts emission at
    `alm_utils.py:3825` with action `dual_update` and that
    `multipliers` advanced by exactly one
    `_project_nonnegative_multipliers_with_diagnostics` step.
  - `subproblem_continue_feasible_plateau` -- feasible iterate but
    not within `update_stationarity_tol`; asserts emission at
    `alm_utils.py:3893` with action `subproblem_continue` and
    `feasible_stall_count` advanced.
  - `plateau_stall` -- repeats the above until
    `feasible_stall_count >= _PLATEAU_STALL_LIMIT`; asserts
    emission at `alm_utils.py:3845` with action `subproblem_limit`
    and the failure-result return at `alm_utils.py:3847`.
  - `infeasible_penalty_increase` -- evaluator returns infeasible
    after inner attempt; asserts emission at `alm_utils.py:3590`
    with action `infeasible_stall_penalty_increase` and that
    `_apply_alm_penalty_increase` was called once.
  - `penalty_cap_terminates` -- penalty hits the cap; asserts
    `_build_alm_failure_result_with_optional_restore` is called
    with `termination_reason == "penalty_cap_reached"` from each of
    the three current call sites at lines `3548`, `3731`, and
    `3922`.
  - `max_outer_exhaustion` -- exhaust the outer budget without
    earlier termination; asserts the post-loop
    `_termination_reason_from_history(...)` path at
    `alm_utils.py:3974` and the final return at
    `alm_utils.py:3979-4010`.
- [ ] Add `tests/geo/test_alm_utils.py::CallbackOrderingCharacterizationTests`
  using a recording stub for `outer_state_callback`,
  `inner_callback`, `accepted_callback`, and `history_callback`. The
  ordering invariant locks (a) one `outer_state_callback` per outer
  iteration before any continuation work, (b) all `inner_callback`
  invocations strictly precede `accepted_callback` for that
  continuation pass, and (c) `history_callback` fires after every
  history-entry append. The stubs assert the relative order via a
  shared monotonic counter, not absolute counts.
- [ ] Add a result-schema lock test:
  `tests/geo/test_alm_utils.py::ResultSchemaLockTests::test_minimize_alm_result_schema_keys_are_frozen`
  builds two synthetic results (one converged, one penalty-cap
  failure with restore) and freezes
  `sorted(result.keys())` plus
  `sorted(result["alm_summary"].keys())`. Phase 0 captures the
  current snapshot literal.

Acceptance: all Phase 0 tests pass green against the unchanged
`_minimize_alm_impl`. The baseline-metrics test pins today's
1040/0-nested state; the ≤500 LOC gate is introduced in Commit 6,
not in Phase 0. Run the full validation matrix below to confirm no
regressions on existing tests.

### Phase 1a: Extract `_build_alm_history_entry` (high-ROI micro-refactor)

- [ ] AST-comparison evidence: the two history-entry literals at
  `alm_utils.py:3105-3179` (skipped-inner branch) and
  `alm_utils.py:3362-3461` (post-inner branch) share **52 keys**, with
  the skipped-inner branch carrying one additional key (`action`).
  This is not strictly line-for-line identical but is structurally
  identical save the documented one-key delta.
- [ ] Add `_build_alm_history_entry(...)` returning a fresh dict with
  the 52 shared keys populated from explicit arguments. The skipped-
  inner branch wraps the call as
  `_build_alm_history_entry(...) | {"action": ...}` (or assigns the
  extra key after the call) to preserve the existing key set exactly.
- [ ] Replace both call sites with the helper. Preserve insertion
  order at both sites; assert order in the test below.
- [ ] Add `tests/geo/test_alm_utils.py::AlmHistoryEntryBuilderTests::test_build_alm_history_entry_payload_identity`
  that constructs both call-site argument bundles and asserts the
  resulting dicts have identical keys (modulo the documented `action`
  delta) and identical values for shared keys.
- [ ] Add a key-preservation assertion: `set(history_entry.keys())`
  must equal a frozen reference set captured by the Phase 0 schema
  characterization, not a derived computation, so a future drop or
  rename fails the test loudly.

Acceptance: Phase 0 history-schema characterization tests still pass
unchanged; both call sites produce dicts byte-identical to the
pre-refactor literals; the new payload-identity test is green.

### Phase 1: Extract `_normalize_alm_run_inputs`

- [ ] Add the `_ALMNormalizedRunInputs` dataclass next to `ALMRunState`
  at `alm_utils.py:79`.
- [ ] Add `_normalize_alm_run_inputs` consuming the existing helpers
  `_build_constraint_metadata_tuples` (`alm_utils.py:2173`),
  `_penalty_schedule_tolerance` (`alm_utils.py:1667`), and
  `_normalize_trust_radius` (`alm_utils.py:1530`) without changing
  their bodies.
- [ ] Replace `alm_utils.py:2987-3043` in `_minimize_alm_impl` with
  the helper call. The rest of the per-call state
  (`history`, `final_eval`, `last_result`, `total_inner_iterations`,
  the cap-binding accumulators, `best_feasible`) stays as local
  assignments because those values mutate during the outer loop.
- [ ] Preserve every validation message verbatim. Hold the helper to
  the existing strings:
  - `"snapshot_accepted_state_fn and restore_incumbent_state_fn must be provided together"`.
  - `"settings.penalty_max must be positive when provided"`.
  - The two interpolated `"settings.penalty_max ({...}) must be >= settings.penalty_init ({...})"` and
    `"initial ALM penalty ({...}) must be <= settings.penalty_max ({...})"` strings.
  - `"settings.history_max_entries must be positive or None"`.
  - `"initial ALM penalty must be finite and positive"`.
- [ ] Add direct-helper unit tests:
  - `test_normalize_alm_run_inputs_rejects_asymmetric_incumbent_hooks`
    -- verifies the existing test
    `tests/geo/test_alm_utils.py::test_minimize_alm_rejects_asymmetric_incumbent_hooks`
    at line 965 still passes by routing through the helper.
  - `test_normalize_alm_run_inputs_rejects_invalid_penalty_max`,
    `test_normalize_alm_run_inputs_rejects_history_max_entries_nonpositive`,
    and `test_normalize_alm_run_inputs_rejects_initial_penalty_above_cap`
    -- one per validation message.
  - `test_normalize_alm_run_inputs_returns_active_penalty_and_tolerances`
    -- builds a benign input set and asserts the output dataclass
    matches the field-by-field expected values produced by the
    pre-extraction code path.

Acceptance: Phase 0 tests stay green; new helper tests pass; the
public characterization tests in
`tests/geo/test_alm_utils.py` are unchanged.

### Phase 2: Extract `_handle_alm_penalty_cap_termination`

- [ ] Add the helper as a thin keyword-forward into
  `_build_alm_failure_result_with_optional_restore` at
  `alm_utils.py:2700`. No body changes inside the existing helper.
- [ ] Replace the three penalty-cap return blocks at
  `alm_utils.py:3548-3585`, `alm_utils.py:3731-3771`,
  `alm_utils.py:3922-3960` with the new helper call. Carry over the
  full keyword set (`message_prefix`, `restored_message_prefix`,
  `restored_termination_reason`, `evaluation`, `multipliers_state`,
  `penalty_state`, `inner_result`, `final_max_feasibility_violation`)
  unchanged.
- [ ] Add direct-helper unit test
  `test_handle_alm_penalty_cap_termination_forwards_all_keyword_arguments`
  that uses `unittest.mock.patch` on
  `_build_alm_failure_result_with_optional_restore` and asserts the
  exact call arguments.
- [ ] No new dataclass fields. The
  `penalty_transition.penalty_update_state.evaluation` and
  `penalty_transition.penalty_update_state.max_violation` reads are
  identical to today.

Acceptance: the `penalty_cap_terminates` arm of the
`ContinuationArmCharacterizationTests` from Phase 0 still passes
unchanged. The three duplicated blocks span 118 lines gross
(`alm_utils.py:3548-3585`, `:3731-3771`, `:3922-3960`); the new helper
plus three call sites costs roughly 35-45 lines, so the net reduction
in `_minimize_alm_impl` is ~70-85 lines, not the gross 118. Treat the
"~120 lines" estimate from earlier review notes as overstated.

### Phase 3: Extract `_handle_alm_dual_update_transition`

- [ ] Add `_ALMDualUpdateResult` next to the other private carriers.
- [ ] Add `_handle_alm_dual_update_transition` owning lines
  `alm_utils.py:3794-3826` except for the
  `_emit_alm_history_snapshot(...)` invocation and the `break`
  statement (those stay in the driver to keep callback ordering
  visible).
- [ ] Within the helper, reuse
  `_project_nonnegative_multipliers_with_diagnostics`
  (`alm_utils.py:1618`) and the existing `settings.penalty_scale`
  schedule unchanged.
- [ ] Driver updates: read fields from the returned
  `_ALMDualUpdateResult`, mutate `history_entry`, then call
  `_emit_alm_history_snapshot` and `break` -- preserving the existing
  callback ordering.
- [ ] Add direct-helper unit tests:
  - `test_handle_alm_dual_update_transition_advances_multipliers_once`
    asserts that `_project_nonnegative_multipliers_with_diagnostics`
    is invoked exactly once with the supplied multipliers and the
    exact `preferred_dual_update_values` from the routing state.
  - `test_handle_alm_dual_update_transition_tightens_tolerances`
    drives the helper twice and asserts both
    `update_feasibility_tol` and `update_stationarity_tol` shrink by
    `settings.penalty_scale` and never below
    `settings.feasibility_tol` / `settings.stationarity_tol`.
  - `test_handle_alm_dual_update_transition_records_cap_binding`
    uses an evaluator that pushes a multiplier past
    `settings.multiplier_max` and verifies the cap-binding indices
    are returned.

Acceptance: the `dual_update` arm characterization stays green;
multiplier-cap tracking is unchanged in the
`MinimizeAlmTests` suite.

### Phase 4: Extract `_run_alm_continuation_step` And `_run_alm_outer_iteration`

- [x] Add `_ALMContinuationStepResult` and `_ALMOuterIterationResult`.
- [x] Add `_run_alm_continuation_step`. Move the body of the
  continuation loop currently spanning
  `alm_utils.py:3052-3966`. Within the helper:
  - Reuse `_attach_alm_constraint_metadata` and
    `_extract_constraint_state` unchanged.
  - The converged-on-entry branch at
    `alm_utils.py:3099-3252` returns a
    `_ALMContinuationStepResult(decision=_ALMContinuationDecision.RETURN, result=...)`.
  - The post-inner converged branch at
    `alm_utils.py:3474-3519` returns analogously.
  - The `infeasible_stall_penalty_increase` arm returns
    `decision=_ALMContinuationDecision.BREAK_OUTER` with updated state fields.
  - The dual-update branch returns `decision=_ALMContinuationDecision.BREAK_OUTER` after
    invoking `_handle_alm_dual_update_transition`.
  - The `subproblem_continue` and `subproblem_limit` arms return
    `decision=_ALMContinuationDecision.CONTINUE_CONTINUATION` and
    `decision=_ALMContinuationDecision.BREAK_OUTER` respectively, depending on whether the
    `_PLATEAU_STALL_LIMIT` triggers.
  - The penalty-increase branches at the bottom return
    `decision=_ALMContinuationDecision.BREAK_OUTER` plus the optional penalty-cap
    short-circuit return.
- [x] Add `_run_alm_outer_iteration` owning the single-iteration
  driver: the `outer_state_callback` invocation, the continuation
  `for` loop dispatching into `_run_alm_continuation_step`, and the
  decision propagation that maps a step result into one of three
  outer-level decisions: `next_outer`, `return`, or stay-in-loop
  (handled internally by the continuation `for`).
- [x] Replace the outer `for outer_iteration in range(...)` body in
  `_minimize_alm_impl` with a call to
  `_run_alm_outer_iteration` per iteration plus the existing
  `is_final_outer` exit check at `alm_utils.py:3968-3969`.
- [x] Flip the
  `test_minimize_alm_impl_state_carriers_exist` test from Phase 0
  to a green assertion.
- [x] Direct-helper unit tests cover the committed helper contracts:
  converged-on-entry and infeasible-stall penalty increase for the
  continuation-step helper; return / next-outer / exhaust signaling,
  callback ordering, and continuation-budget exhaustion for the
  outer-iteration helper. The Phase 0 arm tests remain in place as
  integration coverage; the Phase 4 helper tests add direct-call
  coverage and assert the explicit decision values.

Acceptance: all Phase 0 arm tests stay green;
`_minimize_alm_impl` now reads as `validation -> normalize -> outer
loop -> failure-tail`.

### Phase 5: Driver Collapse And Span Verification

- [x] Reduce `_minimize_alm_impl` body to:
  - one call to `_normalize_alm_run_inputs`.
  - per-call mutable state init for `history`,
    `total_inner_iterations`, `final_eval`, `last_result`,
    `final_multipliers`, `final_penalty`, `last_outer_iteration`,
    `cap_binding_detected`, `cap_binding_indices`,
    `penalty_cap_reached`, `penalty_cap_requested`,
    `history_truncated_count`, `best_feasible`.
  - the outer `for outer_iteration in range(...)` loop dispatching
    into `_run_alm_outer_iteration`.
  - the post-loop guard at `alm_utils.py:3971-3972`.
  - the `_termination_reason_from_history` plus
    `_build_alm_failure_result_with_optional_restore` tail at
    `alm_utils.py:3974-4009`.
- [x] Verify with
  `python -c "import ast; tree = ast.parse(open('examples/single_stage_optimization/alm_utils.py').read()); n = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == '_minimize_alm_impl'); print(n.end_lineno - n.lineno + 1)"`
  prints `<= 500`.
- [x] Confirm 0 nested function definitions in
  `_minimize_alm_impl` via the same AST script.

Acceptance: `tests/geo/test_alm_utils.py::AlmStructuralDebtTests::test_minimize_alm_impl_orchestrator_is_small_and_closure_free`
is green.

### Phase Final: Simplification And Closeout

- [x] Run `code-simplifier` on touched files only:
  `examples/single_stage_optimization/alm_utils.py` and any test
  module that gained helpers in Phase 0-4. No semantic changes.
- [x] Update this plan with completed checkboxes, the final
  `_minimize_alm_impl` line span, the helper count, and the
  validation evidence block.
- [x] Update Backlog 5 / engineering-followup tracker entries that
  still reference the 1,040-line `_minimize_alm_impl`. Specifically,
  edit `docs/alm_backlog5_structural_debt_plan_2026-05-07.md`
  Implementation Result section and
  `docs/alm_hardening_engineering_followup_todo_2026-05-07.md`
  Backlog 5 status block.

Acceptance: the trackers no longer carry stale line counts and no
Phase Final checkbox remains open.

## Test Strategy

| Phase | Locks observable behavior | New direct-helper tests | Rollback signal |
|---|---|---|---|
| 0 | adds `ContinuationArmCharacterizationTests`, `CallbackOrderingCharacterizationTests`, `ResultSchemaLockTests`, structural-debt sibling assertion | none yet | any Phase 0 test fails on unchanged `_minimize_alm_impl` |
| 1 | Phase 0 arm/callback tests, plus existing `test_minimize_alm_rejects_asymmetric_incumbent_hooks`, the four `settings.penalty_*` validation tests in `MinimizeAlmTests`, and the `history_max_entries` validation test | `test_normalize_alm_run_inputs_*` series (one per validation message + one happy-path field check) | any Phase 0 test fails OR direct-helper tests fail OR a validation message text changes |
| 2 | `penalty_cap_terminates` arm in Phase 0 + result-schema lock | `test_handle_alm_penalty_cap_termination_forwards_all_keyword_arguments` | the schema lock or the `penalty_cap_terminates` arm fails |
| 3 | `dual_update` arm and the multiplier-cap-binding tests | `test_handle_alm_dual_update_transition_*` series | `dual_update` arm fails OR `multiplier_cap_binding` flag drift |
| 4 | every Phase 0 arm + every existing `MinimizeAlmTests` test | `test_run_alm_continuation_step_*` (one per arm) and `test_run_alm_outer_iteration_*` (single-iteration converged + multi-iteration penalty-cap path) | any Phase 0 arm test fails; structural-debt sibling test still red after the move |
| 5 | structural-debt sibling test | none | the sibling structural-debt test or any prior arm test fails |

The rollback signal is uniform: the most recent commit gets reverted
and the helper extracted in that commit is dropped. Because each
phase touches one helper plus its callers, reverts are clean and
isolated.

## Risk Register

| Risk | Test that catches it | Line range at risk |
|---|---|---|
| Callback re-ordering during continuation-step extraction (e.g. `outer_state_callback` fires after first `_emit_alm_history_snapshot` instead of before) | `CallbackOrderingCharacterizationTests` shared-counter assertion (Phase 0) plus `MinimizeAlmTests::test_minimize_alm_emits_outer_state_callback_per_iteration` already in `tests/geo/test_alm_utils.py` | `alm_utils.py:3045-3052` (outer setup) and every `_emit_alm_history_snapshot` callsite at lines 3192, 3481, 3590, 3597, 3639, 3674, 3776, 3791, 3825, 3845, 3893, 3965 |
| History-entry identity drift (helper returns a fresh dict instead of mutating the appended history-tail dict, which would break `_attach_alm_history_diagnostics`'s effect on the live history list and snapshot) | `tests/geo/test_alm_utils.py::test_history_truncation_records_truncated_count` plus the new `ResultSchemaLockTests::test_minimize_alm_result_schema_keys_are_frozen` and the `dual_update` arm assertion that the post-update `history_entry["post_update_multipliers"]` is the projected multipliers, not the pre-update values | `alm_utils.py:3105-3179` (skipped-inner history append) and `alm_utils.py:3362-3461` (post-inner history append) |
| Restore-on-failure semantics break (helper omits the `incumbent_state` capture or restore-fn call) | `MinimizeAlmTests::test_minimize_alm_restore_incumbent_state_called_on_penalty_cap_failure` (existing) plus the Phase 0 `penalty_cap_terminates` arm | `alm_utils.py:3548-3585`, `alm_utils.py:3640-3667`, `alm_utils.py:3675-3702`, `alm_utils.py:3922-3960`, and `alm_utils.py:3979-4010` |
| Multi-thread safety regresses via accidental module-level state in the new dataclasses (e.g. mutable default in a dataclass field) | `test_minimize_alm_does_not_introduce_module_level_mutable_state` -- new test asserting the new dataclasses use `frozen=True` and `field(default_factory=...)` for any non-primitive default | `alm_utils.py:79-189` (new state-carrier additions) |
| SciPy `jac=True` contract regression because the new continuation-step helper accidentally bypasses `_run_alm_inner_attempts` and re-implements the inner call | `MinimizeAlmTests::test_minimize_alm_passes_jac_true_through_inner_attempt` (existing) plus a Phase 4 direct-helper assertion that `_run_alm_continuation_step` invokes the production `_run_alm_inner_attempts` path | `alm_utils.py:3254-3273` (inner-attempt construction) and `alm_utils.py:2796-2968` (`_run_alm_inner_attempts` body) |

## Validation Matrix

Run after every implementation phase that changes code, and once
again before closeout. Same suite as Backlog 5 with the addition of
the result-schema lock test from Phase 0.

```bash
.venv/bin/python -m py_compile \
  examples/single_stage_optimization/alm_utils.py
.venv/bin/python -m pytest -q tests/geo/test_alm_utils.py
.venv/bin/python -m pytest -q tests/geo/test_single_stage_workflow_helpers.py
.venv/bin/python -m pytest -q tests/geo/test_single_stage_alm_integration.py
.venv/bin/python -m pytest -q tests/geo/test_single_stage_example.py \
  -k "AlmUtilsTests or build_alm_final_constraint_payload or alm_result_view_from_search_eval or stage2_main_alm_path_uses_minimize_alm or validate_resume_alm_state or current_solver_checkpoint_alm_state"
.venv/bin/python -m pytest -q tests/geo/test_constraint_contract.py tests/geo/test_banana_helper_modules.py
git diff --check
```

Run once before closeout to lock the new structural metric:

```bash
.venv/bin/python - <<'PY'
import ast
from pathlib import Path

source = Path("examples/single_stage_optimization/alm_utils.py").read_text()
tree = ast.parse(source)
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in (
        "_minimize_alm_impl",
        "minimize_alm",
        "_run_alm_outer_iteration",
        "_run_alm_continuation_step",
        "_handle_alm_penalty_cap_termination",
        "_handle_alm_dual_update_transition",
        "_normalize_alm_run_inputs",
    ):
        nested = [
            child
            for child in ast.walk(node)
            if isinstance(child, ast.FunctionDef) and child is not node
        ]
        print(
            f"{node.name}: lineno={node.lineno}, "
            f"end_lineno={node.end_lineno}, "
            f"span={node.end_lineno - node.lineno + 1}, "
            f"nested_defs={len(nested)}"
        )
PY
```

Stale-code grep gate (must return exactly one production hit after
Phase 5, inside `_run_alm_outer_iteration`):

```bash
git grep -nE "for continuation_iteration in range\\(settings\\.max_subproblem_continuations \\+ 1\\)" \
  examples/single_stage_optimization/alm_utils.py
```

The original continuation `for` literal lives at
`alm_utils.py:3052` only; once Phase 4 moves the loop body into
`_run_alm_continuation_step`, exactly one occurrence is allowed and
that occurrence sits inside `_run_alm_outer_iteration`. Any grep hit
inside `_minimize_alm_impl` post-Phase-5 is a regression.

## Commit Plan

Suggested split, mirroring the Backlog 5 commit grouping pattern:

- Commit 1: `test: lock _minimize_alm_impl decomposition baseline`
  -- adds Phase 0 characterization tests plus a baseline-metrics test
  that pins today's structural numbers (1040 lines, 0 nested defs)
  against AST inspection. The baseline test does **not** assert
  ≤500 lines yet; that gate is added in Commit 6 by the same commit
  that achieves the bound. Avoid `expectedFailure` markers — they
  give false-green signals.
- Commit 2: `refactor: extract _build_alm_history_entry`
  -- Phase 1a helper that collapses the duplicated history-entry
  schema (52 shared keys at `alm_utils.py:3105-3179` and
  `:3362-3461`, plus a skipped-inner-only `action` key). High-ROI
  micro-refactor that lands before broader structural work; locked
  by a payload-identity test that asserts both call sites produce
  byte-identical dicts modulo the documented one-key delta.
- Commit 3: `refactor: extract _normalize_alm_run_inputs helper`
  -- Phase 1 helper plus its direct unit tests.
- Commit 4: `refactor: collapse penalty-cap termination call sites`
  -- Phase 2 helper. Net reduction in `_minimize_alm_impl` is
  ~70-85 lines (the three duplicated blocks gross to 118 lines but
  the helper plus call sites cost ~35-45 lines).
- Commit 5: `refactor: extract dual-update transition helper`
  -- Phase 3 helper plus its direct unit tests.
- Commit 6: `refactor: extract ALM continuation-step and outer-iteration helpers`
  -- Phase 4 helpers and their state carriers; updates the baseline
  metrics test from Commit 1 to assert the new ≤500 LOC bound on
  `_minimize_alm_impl`.
- Commit 7: `refactor: collapse _minimize_alm_impl driver and update trackers`
  -- Phase 5 / Phase Final, including the `code-simplifier` pass and
  tracker doc updates.

## Go/No-Go Criteria

Start implementation only after Phase 0 characterization tests are
green against the unchanged `_minimize_alm_impl`. Stop and reassess
if any of the following hold:

- A Phase 0 arm test cannot be expressed without re-implementing ALM
  policy in the test (signal that the body is not pure-enough to
  extract without changing math).
- The `_run_alm_continuation_step` signature exceeds ~25 keyword
  arguments after consolidation (signal that the continuation-step
  surface is structurally indivisible at this granularity; revisit
  whether `ALMRunState` should grow to absorb the
  per-continuation-pass scalars).
- After Phase 4, `_minimize_alm_impl` is still > 500 lines (signal
  that one more arm needs to fold into the continuation-step helper).
- Any phase requires changing ALM tolerances, penalty schedule
  policy, callback ordering, or scipy `jac=True` contract to make a
  test pass (hard stop; the refactor exists to shrink the file
  without touching math).

## Notes

- At baseline, `_minimize_alm_impl` was 1,040 lines. The four extraction
  helpers projected here together carry ~470-580 lines depending on
  how much of the per-iteration boilerplate stays in the driver. The
  current driver is 127 lines, well below the 500-line bound.
- Backlog 5 already exposed five state-carrier dataclasses
  (`ALMRunState`, `ALMFinalState`, `ALMHistoryEntry`,
  `ALMInnerAttemptRequest`, `ALMInnerAttemptResult`) and 77 private
  module-level helpers before this decomposition pass. `ALMRunState`
  is the mutable per-call accumulator; the other four are frozen. The
  current tree now has 93 private helpers after the Phase Final arm
  simplification. New private carriers introduced by this decomposition
  (`_ALMNormalizedRunInputs`, `_ALMOuterIterationResult`,
  `_ALMContinuationStepResult`, `_ALMDualUpdateResult`,
  `_ContinuationStepState`, `_PenaltyIncreaseOutcome`) remain private
  with leading underscore prefixes and stay out of the public module
  surface. `_ContinuationStepState` is the only new mutable carrier,
  and it does not escape a continuation-step call.
- The continuation-step helper is the load-bearing extraction. Each
  of the other three helpers (normalize, penalty-cap, dual-update)
  is small enough to land independently without disturbing the
  driver's overall shape, so the commits stay narrowly scoped.
