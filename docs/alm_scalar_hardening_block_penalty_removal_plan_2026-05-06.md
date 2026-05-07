# Scalar ALM Hardening and Block-Penalty Removal Plan

Date: 2026-05-06

Status: draft implementation plan after live-code review

Validated against: `simsopt-surrogate` HEAD `d116040a6`

## Scope

This plan hardens the live scalar augmented Lagrangian method (ALM) path, removes the dormant block-penalty control path, and locks the evaluator/checkpoint contracts that feed ALM.

It does not split diagnostic-only code into a new module. That is a separate cleanup after correctness and deletion stabilize.

## Review Update

Accepted review points:

- The strategic direction is correct: fix scalar ALM, keep constraint-block labels as diagnostics, and delete dormant block-penalty control.
- H4, H5, M2, and M6 are valid and are now explicit phases.
- The original phase ordering had a real contradiction: scalar tolerance/history work must not claim block-stall deletion before block penalties are removed.
- KKT residual work needs its own phase and commit.
- Resume hardening must include constraint names, not just multiplier length.
- `alm_block_penalties` needs a decision before coding. The decision here is: keep the field as a legacy nullable field populated as `None`.
- The final gate needs at least one lightweight end-to-end smoke run, not only unit tests.

Rejected or downgraded review points:

- H2 is stale as a live bug in the current tree. `frontier_evaluator.py` rejects `--single-stage-banana-current-mode=independent` before the runtime path that omits `banana_current_state` can run. Keep the regression guard, but do not schedule a live fix unless independent-mode frontier support is intentionally restored.
- `banana_opt/alm_fixture_benchmarking.py` currently uses `constraint_blocks` as metadata labels. It does not use active block-penalty internals. It still needs signal-contract updates before fallback removal.

## Contract Decisions

1. ALM penalty control is scalar-only for production.
2. `constraint_blocks` remain as diagnostic labels for summaries/history. They do not drive penalty updates.
3. `ALMResult.block_penalties`, the exported `ALM_BLOCK_PENALTIES` diagnostic key, and downstream `alm_block_penalties`/`alm_block_penalties_json` ledger fields remain as legacy schema fields with value `None`.
4. No `--alm-block-penalty-*` CLI flags are added.
5. Resume from ALM state is fail-fast when saved constraint names do not exactly match current constraint names.
6. No reset-on-mismatch fallback is introduced.
7. Evaluators must provide the normalized fields ALM consumes. Missing ALM signal fields are contract errors after migration.

## Audit Coverage Matrix

| ID | Disposition |
| --- | --- |
| C1 | Phase 2: all failure paths use one restore-aware builder, including penalty-cap termination. |
| C2 | Phase 7: delete `previous_violations_by_block` with dormant block penalties. Phase 3 verifies no scalar analogue exists. |
| C3 | Phase 3: scalar tolerance schedule is monotone nonincreasing; block schedule oscillation disappears with Phase 7 deletion. |
| C4 | Phase 4: KKT residual includes strongly active/violated constraints. The feasibility gate controls reporting, not active-set inclusion. |
| C5 | Phase 6: resume validates multiplier length and constraint-name identity. |
| C6 | Phase 2: restoration updates all result-relevant ALM state, not only `x`. |
| H1 | Phase 5: raw fields are only raw explicit fields. No normalized fallback is labeled raw. |
| H2 | Already guarded in current tree. Keep tests that reject independent-mode frontier evaluation unless support is reintroduced deliberately. |
| H3 | Phase 7: delete dormant block-penalty control. |
| H4 | Phase 8: prevent inherited `ALM_*` environment variables from silently overriding runner settings. |
| H5 | Phase 2: early-converged shortcut records `best_feasible` before returning success. |
| M1 | Phase 4: document and assert the `metric_grad` contract as objective gradient only, or stop using it for KKT residuals. |
| M2 | Phase 3: align progress/stall threshold semantics so steady small improvement is not treated as no progress. |
| M3 | Phase 5: remove fallback chains that can feed unnormalized values to multiplier updates. |
| M4 | Phase 3: scalar penalty growth publishes the updated penalty state before history emission. |
| M5 | Phase 2: restored failures get restored termination reasons consistently. |
| M6 | Phase 5: nonfinite evaluation sanitization owns/copies all ALM-consumed array fields. |
| M7 | Phase 7: delete the unused equality `augmented_objective` helper if it is not a public contract; otherwise fix equality feasibility semantics and keep tests. |

## Phase 0: Baseline Inventory

Goal: make the deletion and contract changes reviewable before behavior changes.

Tasks:

- Record `git status --short` and HEAD before implementation.
- Inventory every caller of `_extract_constraint_state` and `_extract_stage2_constraint_signal_state`.
- For each evaluator, verify it provides `constraint_values`, `feasibility_values`, `dual_update_values`, `hard_signed_constraint_values`, and `hard_dual_update_values` where ALM consumes them.
- Include `banana_opt/stage2_objectives.py`, `banana_opt/single_stage_objectives.py`, `banana_opt/alm_fixture_benchmarking.py`, single-stage callback metric construction, and minimal test fixtures.
- Produce a concrete list of evaluators and test fixtures that must be updated before Phase 5 fallback removal.
- Scan existing checkpoint/result JSONL files for ALM states that lack `constraint_names` or have multiplier lengths that would mismatch the current constraint builder.
- If more than 10 production-relevant checkpoints would fail the new resume contract, add a Phase 6.5 offline migration helper that strips/rebuilds ALM multiplier state. The solver still remains fail-fast; this is not a runtime fallback.
- Confirm no documented external consumer requires the block-penalty `ALMSettings` constructor fields. Current repo grep only found examples, tests, and historical docs.
- Verify current scalar stall tracking has no stale `previous_violation` analogue. Current grep only found block-specific `previous_violations_by_block` plus scalar `feasible_stall_count`.
- Preflight test fallout from restoration changes with `git grep -nE "penalty_cap_reached|restored_best_feasible|termination_reason" tests/`.
- Preflight strict signal migration with `git grep -nE "_extract_constraint_state|_extract_stage2_constraint_signal_state|dual_update_values|hard_dual_update_values|hard_signed_constraint_values|feasibility_values" examples/single_stage_optimization tests/geo`.
- Preflight block-penalty deletion with `git grep -nE "ALMBlockPenaltyState|block_penalty_state|block_penalties_enabled" examples/single_stage_optimization tests/geo docs/`.
- Inventory all ALM-prefixed identifiers before Phase 8 with `git grep -h -oE "ALM_[A-Z0-9_]+" examples/single_stage_optimization | sort -u`; the implementation strips by `ALM_` prefix, not by a hand-maintained allowlist.

Deliverable:

- A short implementation note or commit message section with the inventory result and checkpoint migration impact.

## Phase 1: Schema and Label Contract Lock

Goal: decide the compatibility surface before deleting code.

Tasks:

- Keep `constraint_blocks` as label metadata.
- Rename surviving block-label diagnostic helpers after block-penalty deletion, for example `_constraint_block_history_diagnostics` to `_constraint_label_history_diagnostics`, so `block` cannot be confused with active penalty groups.
- Keep `ALMResult.block_penalties` and exported result diagnostic key `ALM_BLOCK_PENALTIES` as legacy fields with value `None`.
- Keep downstream JSONL/registry columns such as `alm_block_penalties` and `alm_block_penalties_json` as legacy nullable output.
- Document that block labels are not penalty-control groups.

Acceptance:

- Existing downstream parsers that expect `alm_block_penalties` do not break.
- No live solver decision reads block penalty fields.

## Phase 2: Failure and Restore Unification

Goal: close C1, C6, H5, and M5 as one failure-result bug class.

Tasks:

- Replace parallel failure-result paths with one restore-aware builder.
- Route scalar penalty-cap termination through the restore-aware builder.
- Ensure restored failures use a restored termination reason suffix or explicit restored reason.
- When restoration fires, update all result-relevant ALM state consistently: `x`, multipliers, penalty, final evaluation, final multipliers, final penalty, and any remaining scalar penalty state.
- Record `best_feasible` before the early-converged shortcut returns success.
- Avoid adding fallback behavior. Restoration only uses a previously recorded feasible incumbent.

Tests:

- Penalty-cap failure restores a prior feasible incumbent.
- Early-converged success records a feasible incumbent.
- Returned state is internally consistent after restoration.
- Termination reason reflects restoration when restoration happened.

## Phase 3: Scalar Penalty, Tolerance, History, and Progress

Goal: make the scalar ALM update path monotone and correctly reported.

Tasks:

- Make update feasibility/stationarity tolerances monotone nonincreasing. Use `min(current_tol, scheduled_tol)` or an equivalent scalar schedule invariant.
- Refactor `_publish_current_penalty_state` to be scalar-pure before block-penalty deletion.
- Call the publish helper after scalar penalty growth and before history emission.
- Align progress/stall threshold semantics so objective/stationarity improvements are not held to a much looser 5 percent threshold while feasibility uses a near-zero relative threshold.
- Keep Phase 3 scalar-only. Do not delete block-stall code here; C2 closes in Phase 7.
- Add a regression test that constructs an outer-iteration sequence such as `dual_update`, `dual_update`, `dual_update`, `penalty_increase` and asserts stall classification uses current post-iteration progress, not a baseline frozen at the last penalty change.

Tests:

- Tolerances never loosen across dual-update and penalty-growth iterations.
- A scalar penalty-increase history row reports the updated penalty.
- A steady small improvement is not classified as no progress solely because it is below a hard-coded 5 percent threshold.

## Phase 4: KKT Stationarity and Gradient Contract

Goal: close C4 and M1.

Tasks:

- In `_kkt_stationarity_norm`, remove the active-set exclusion that skips strongly violated constraints.
- Use feasibility gating only to decide whether the residual is reported or trusted, not to omit violated constraints from the residual.
- Lock the `metric_grad` contract as objective-gradient-only for KKT diagnostics, or use the objective gradient source directly.
- KKT residuals are reported separately and do not reduce the stationarity value used for solver decisions. Convergence/stall decisions use the raw stationarity signal unless a later explicit contract changes that rule.

Tests:

- A strongly violated active inequality contributes to KKT stationarity.
- A bad or missing `metric_grad` contract cannot reduce solver-decision stationarity below the raw stationarity signal.

## Phase 5: Strict Signal Contract and Evaluation Ownership

Goal: close H1, M3, and M6.

Tasks:

- After Phase 0 inventory, remove fallback chains in `_extract_constraint_state` and `_extract_stage2_constraint_signal_state`.
- Require normalized ALM fields explicitly where ALM consumes normalized values.
- Raw result fields must come only from explicit raw keys. If the raw key is absent, the raw field is `None`.
- Update minimal test evaluators and `banana_opt/alm_fixture_benchmarking.py` to emit required ALM signal fields.
- Update single-stage callback metric construction so it does not rely on fallback routing for feasibility or dual-update values.
- In `_sanitize_nonfinite_inner_evaluation`, copy/own all ALM-consumed array fields and constraint-gradient lists when falling back to the previous finite evaluation.

Tests:

- Missing `dual_update_values` fails fast in ALM signal extraction.
- Missing raw fields produce `None`, not normalized values labeled raw.
- Sanitized fallback evaluation does not share mutable ALM array fields with the rejected evaluation.

## Phase 6: Resume and Checkpoint Contract

Goal: close C5 and prevent silent multiplier/constraint misalignment.

Tasks:

- Serialize `constraint_names` alongside `penalty` and `multipliers` in ALM checkpoint state.
- On resume, require exact element-wise match between saved and current constraint names.
- Require multiplier length to match current constraint count.
- Fail fast on mismatch with a direct error explaining saved and current constraint names.
- Keep the current independent-mode frontier rejection as the H2 guard unless independent-mode frontier support is intentionally restored.
- Add or retain `tests/geo/test_frontier_evaluator.py::test_frontier_runtime_rejects_independent_banana_current_mode` so independent banana-current mode is rejected before building a mismatched ALM constraint-name list.
- Add a checkpoint scan report from Phase 0 to the migration note.

Tests:

- Resume succeeds with matching constraint names and multiplier length.
- Resume fails with changed formulation, changed banana-current mode, changed surface stack, or changed JSurfSurf constraint count.
- Frontier evaluator still rejects independent banana-current mode unless explicitly supported.

## Phase 7: Remove Dormant Block-Penalty Control and Equality Helper

Goal: close C2/H3 and reduce the control-flow surface that caused several bug classes.

Split this into reviewable commits if the diff is large.

Tasks:

- Delete `ALMBlockPenaltyState`.
- Delete block-penalty settings from `ALMSettings` directly. If Phase 0 finds a documented external consumer outside examples/tests/historical docs, amend this plan before implementation.
- Delete block-penalty helpers: initialization, vector conversion, summaries, cap summaries, requested summaries, per-block max violations, next-state logic, cap-hit logic, validation, and block float-map helpers.
- Collapse `minimize_alm` to one scalar penalty path.
- Remove block-penalty tests and fixtures that test deleted control behavior.
- Preserve tests that cover `constraint_blocks` as diagnostic labels.
- Only constraint-label diagnostics may survive the deletion list above; rename `_constraint_block_history_diagnostics` to label terminology if it remains, and document any remaining `block` wording as diagnostic constraint labels, not penalty groups.
- Keep result `block_penalties`/`ALM_BLOCK_PENALTIES` as `None` only for legacy schema compatibility.
- Delete `augmented_objective` equality helper if it is not a public contract. If it must stay, fix equality feasibility to use absolute constraint violation and add the matching equality dual-update contract.
- Update equality-helper references in `tests/geo/test_alm_utils.py` and `tests/geo/test_single_stage_example.py` at the currently pinned `augmented_objective` tests.
- Check history-diagnostics test fallout around `_materialize_history_entry_diagnostics` and `_snapshot_history_entry` in `tests/geo/test_alm_utils.py`.

Tests:

- No production builder or runner can enable block penalties.
- `constraint_blocks` still appear in diagnostics/history where expected.
- No block-penalty control branch remains in `minimize_alm`.
- Equality-helper tests are either removed with the helper or updated for correct equality semantics.
- `git grep -nE "ALMBlockPenaltyState|block_penalty_state|block_penalties_enabled" examples/single_stage_optimization tests/geo docs/` returns zero matches outside historical docs and this plan.

## Phase 8: Runner Environment and Explicit ALM Settings

Goal: close H4.

Tasks:

- Update `workflow_runner_common.run_command` to pass an environment that strips inherited variables with the `ALM_` prefix unless a caller explicitly opts into them.
- Treat these known input defaults as audit anchors: `ALM_MAX_OUTER_ITERS`, `ALM_PENALTY_INIT`, `ALM_PENALTY_SCALE`, `ALM_PENALTY_MAX`, `ALM_FEAS_TOL`, `ALM_STATIONARITY_TOL`, `ALM_TRUST_RADIUS_INIT`, `ALM_TRUST_RADIUS_MIN`, `ALM_TRUST_RADIUS_SHRINK`, `ALM_TRUST_RADIUS_GROW`, `ALM_MAX_INNER_ATTEMPTS`, `ALM_MAX_SUBPROBLEM_CONTINUATIONS`, `ALM_DISTANCE_SMOOTHING`, `ALM_CURVATURE_SMOOTHING`, `ALM_FORMULATION`, `ALM_QS_THRESHOLD`, `ALM_BOOZER_THRESHOLD`, `ALM_IOTA_PENALTY_THRESHOLD`, `ALM_LENGTH_PENALTY_THRESHOLD`, and `ALM_TAYLOR_TEST_SEED`.
- The anchor list is not the strip boundary; any inherited environment variable whose name starts with `ALM_` is removed unless explicitly passed.
- Preserve non-ALM environment variables needed by subprocesses, including physics inputs such as `BOOZER_I`, `PLASMA_CURRENT_A`, and `PROXY_PLASMA_CURRENT_A`.
- Make `run_80ka_baseline_tradeoff_sweep.py` emit explicit single-stage `--alm-*` flags when using ALM, or route through the shared command builder that already emits them.
- Confirm Stage 2 command builders still emit explicit ALM flags.
- Confirm autoresearch launchers that already emit explicit ALM flags continue to do so.

Tests:

- A parent `ALM_PENALTY_INIT` does not change a spawned run unless explicitly passed.
- Summary-recorded ALM settings match subprocess command-line settings.

## Phase 9: Documentation and Migration Notes

Goal: make the new contract discoverable.

Tasks:

- Document scalar-only ALM control.
- Document `constraint_blocks` as labels only.
- Document `alm_block_penalties=None` as a legacy compatibility field.
- Document strict signal field requirements for evaluators.
- Document resume checkpoint migration: older checkpoints without `constraint_names` cannot be trusted for multiplier reuse.
- Include the Phase 0 checkpoint scan result.
- Note that block-penalty code is recoverable from git history if a future per-group penalty design is justified.

## Phase 10: Validation Gates

Required local validation:

```bash
.venv/bin/python -m pytest -q tests/geo/test_alm_utils.py
.venv/bin/python -m pytest -q tests/geo/test_single_stage_alm_integration.py
.venv/bin/python -m pytest -q tests/geo/test_banana_objective_modules.py
.venv/bin/python -m pytest -q tests/geo/test_frontier_evaluator.py
.venv/bin/python -m pytest -q tests/geo/test_alm_fixture_benchmarking.py tests/geo/test_alm_benchmarking.py
git diff --check
git grep -n "block_penalt" examples/single_stage_optimization tests/geo
```

Required smoke validation:

- Run the lightweight finite-current smoke path against a real seed with `examples/single_stage_optimization/run_finite_current_smoke.py`.
- Capture the exact command, seed/artifact path, and resulting ALM status in the final implementation note.

Acceptance:

- Unit tests pass.
- The smoke run completes or fails for a non-ALM reason that is documented with artifact path and command.
- No result schema parser breaks on `alm_block_penalties`.
- No ALM block-penalty control branch remains reachable.
- Final block-penalty grep contains only accepted legacy schema fields, historical docs, or this plan.

## Commit Plan

Recommended implementation commits:

1. `fix: unify ALM failure restoration`
2. `fix: tighten scalar ALM penalty schedule and progress checks`
3. `fix: correct ALM KKT stationarity contract`
4. `fix: enforce ALM signal field ownership`
5. `fix: enforce ALM resume constraint contract`
6. `fix: sanitize ALM runner environments`
7. `refactor: delete ALM block penalty state`
8. `refactor: collapse ALM scalar penalty path`
9. `refactor: remove block penalty tests and equality helper`
10. `docs: clarify ALM block diagnostics`

If review load is more important than commit count, commits 7 to 9 can be combined only after Phases 2 to 6 are green.

## Out of Scope

- Moving diagnostics/Taylor/KKT summaries into `alm_diagnostics.py`.
- Reintroducing per-block penalties behind CLI flags.
- Broad `alm_utils.py` style cleanup not required by the correctness fixes.
- Supporting independent banana-current frontier evaluation. Current behavior is rejection; adding support is a separate feature.
