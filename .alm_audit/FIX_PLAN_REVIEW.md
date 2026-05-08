# FIX_PLAN.md Adversarial Review

## Codex CLI Invocation / Reproducibility

Requested invocation:

```bash
codex exec --model gpt-5.5 -c reasoning.effort=xhigh --sandbox workspace-write --cd /Users/suhjungdae/code/columbia/simsopt-surrogate "<prompt>"
```

Model availability check:

```bash
codex debug models
```

Confirmed from `codex debug models`: model slug `gpt-5.5` is listed, and `xhigh` is one of its supported reasoning levels.

Execution note: the required `codex exec` invocation failed in this parent sandbox before model execution with `failed to initialize in-process app-server client: Operation not permitted`. Retrying with `CODEX_HOME=/private/tmp/codex-review-home` preserved the requested model/effort/sandbox/`--cd` flags and got past that permission failure, but then failed on network/DNS because this environment has network disabled (`failed to lookup address information ... api.openai.com`). Per the user's fallback instruction, this report was written locally from direct reads of the current working tree and `.alm_audit/*` audit documents.

## Per-Fix Verdict Table

| ID | Proposed fix | Verdict | One-line reason |
|---|---|---|---|
| H1 | Reject nonpositive ALM threshold flags and metadata thresholds | NEEDS-REVISION | Direction is right, but the stated "negative flags silently accepted" evidence is stale for `thresholded_physics`; zero/direct metadata paths remain the live root cause. |
| S1 | Auto-zero `LENGTH_WEIGHT` under `--constraint-method alm` | WRONG | Auto-mutating user input is the wrong guardrail posture, and the proposed condition is broader than the actual `weighted_sum` double-feed bug. |
| M1 | Document/accept surrogate-vs-hard hybrid signal and add mismatch test | APPROVED | Option A matches current engineering reality if the doc clearly says this is not classical ALM convergence. |
| M2 | Add final-outer labeling to `_emit_alm_subproblem_continue` | INSUFFICIENT-TESTS | The code fix is plausible, but tests cover only one of the two helper call paths. |
| M3 | Dual-update before subproblem-limit `BREAK_OUTER`; fix staged `gtol` ratchet | WRONG | The dual-update part violates the existing stationarity-gated dual-update contract; use penalty escalation or explicit subproblem-limit termination instead. |
| M4 | Block convergence when multiplier cap is binding | NEEDS-REVISION | Must distinguish current cap-active state from historical sticky cap detection and must gate both success arms, not just the regular converged arm. |
| M5 | Use `effective_feasibility_tol` consistently for routing | APPROVED | Fix addresses the exact root cause and has a focused characterization test. |
| M6 | Catch `nnls` errors in `_kkt_stationarity_norm` | NEEDS-REVISION | Catching `ValueError` hides programmer/data-shape bugs; test should force `RuntimeError`, not rely on rank deficiency. |
| M7 | Record scale-floor application in metadata source/flag | NEEDS-REVISION | Plan misses `hardware_constraint_schema.py`, where the original normalization audit also found hidden floor decisions. |
| M8 | Validate resume multipliers/penalty | NEEDS-REVISION | Fixes single-stage resume JSON only; direct `minimize_alm(initial_multipliers=...)` remains reachable and unvalidated. |
| L1 | Always shallow-copy in `_attach_alm_constraint_metadata` | APPROVED | Addresses the alias asymmetry narrowly and safely. |
| L2 | Remove live-history mutation in single-stage callback | WRONG | Deleting the mutation loses `smoothing_changed` from emitted history, and the proposed test cannot pass without changing ALM's callback contract. |
| L3 | Clone cached evaluator evaluation arrays | APPROVED | Correctly fixes the cache alias root cause with bounded copy scope. |
| T1 | Add KKT / invariance / sign / NaN / signal-mismatch property tests | WRONG | The proposed 2D KKT multiplier is mathematically wrong, and the normalization multiplier assertion is inverted. |

Verdict counts: APPROVED 4, NEEDS-REVISION 5, WRONG 4, INSUFFICIENT-TESTS 1.

## Non-Approved Details

### H1 - NEEDS-REVISION

What is wrong:

The plan's core evidence says negative threshold flags are silently accepted. In the current tree, `validate_single_stage_alm_formulation_args` already rejects negative thresholds for `thresholded_physics`: `single_stage_banana_example.py:2964-2970` builds `negative_thresholds` and raises `ValueError`. The live bug is narrower but still real: zero is accepted there because the check is `< 0.0`, and the metadata constructors still silently floor nonpositive or tiny direct inputs.

Evidence:

- `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:2964-2970` rejects negative `thresholded_physics` thresholds, contradicting the plan title.
- `examples/single_stage_optimization/banana_opt/single_stage_objectives.py:501-507` still uses `scale=max(raw_threshold, ALM_OBJECTIVE_SCALE_FLOOR)`.
- `examples/single_stage_optimization/banana_opt/stage2_objectives.py:719-728` still accepts the explicit Stage 2 iota threshold as `float(value)` and floors it.
- `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:1383-1413` parses env/default threshold values before the later single-stage formulation validator.

What to do instead:

Revise the section to say "nonpositive and direct metadata thresholds can silently floor; zero still accepted by CLI." Keep the `<= 0` validation, but avoid duplicate or contradictory validation text by updating `validate_single_stage_alm_formulation_args` from non-negative to positive or by routing all four threshold checks through one helper. Explicitly decide whether unused `weighted_sum` threshold flags should be rejected globally in `validate_alm_cli_args`; rejecting unused knobs is stricter than today's behavior and should be intentional.

### S1 - WRONG

What is wrong:

The double-feed exists in `weighted_sum` ALM only. The plan says to auto-zero `args.length_weight` whenever `args.constraint_method == "alm"`, which also changes `thresholded_physics` runs where `LENGTH_WEIGHT` is not part of the ALM objective because `evaluate_base_objective` sets `total=0.0` and `grad=0` for `thresholded_physics`.

The proposed auto-zero warning is also a silent config correction. That conflicts with the project posture: no fallback behavior, no hidden correction, and production-grade reproducibility. A warning on stderr does not make saved run identity, cached comparisons, or operator intent unambiguous.

Evidence:

- `single_stage_objectives.py:417-421` includes `LENGTH_WEIGHT * JCurveLength` in the base objective.
- `single_stage_objectives.py:427-432` uses that base objective only when `alm_formulation == "weighted_sum"`.
- `single_stage_objectives.py:781-786` independently adds the ALM `coil_length_upper_bound` constraint.
- `single_stage_banana_example.py:1472-1473` defaults `--length-weight` to `1`, so the double-feed is live by default.
- `single_stage_banana_example.py:8332` assigns `LENGTH_WEIGHT = args.length_weight`, so auto-zeroing mutates user-visible runtime state.

What to do instead:

Use a strict mode-aware resolver. For `constraint_method == "alm" and alm_formulation == "weighted_sum"`, either require effective `length_weight == 0.0` and raise otherwise, or change the ALM wrapper/default resolution so ALM weighted-sum defaults to `0.0` while explicit nonzero values fail. Do not apply the rule to `thresholded_physics`. Add tests for CLI/env/default resolution, result metadata/run identity, and wrapper paths that currently rely on the default.

### M2 - INSUFFICIENT-TESTS

What is wrong:

The implementation idea is reasonable: `_emit_alm_subproblem_continue` currently has no `is_final_outer` argument and writes no `outer_termination`, while `_termination_reason_from_history` falls back to the latest action string. The plan only tests the signal-mismatch call path, but the helper is shared by both signal-mismatch and feasible-update retry paths.

Evidence:

- `alm_utils.py:3457-3484` defines `_emit_alm_subproblem_continue` and sets `action="subproblem_continue"` without `outer_termination`.
- `alm_utils.py:4221-4226` calls it from the signal-mismatch progress path.
- `alm_utils.py:4325-4330` calls it from the feasible-update continuation path.
- `alm_utils.py:1677-1695` returns latest action if `outer_termination` is absent.

What to do instead:

Keep the helper signature change, but add a second final-outer test that reaches the feasible-update retry path at `alm_utils.py:4325-4330`. The midrun unchanged-label test is useful, but it does not prove both terminal callers were wired correctly.

### M3 - WRONG

What is wrong:

The plan's dual-update fix contradicts the existing state machine. The normal dual-update arm requires both `hard_feasible_for_update` and `stationarity_norm <= state.update_stationarity_tol`; the subproblem-limit branch at `alm_utils.py:4267` is reached after that predicate failed. Calling `_handle_alm_dual_update_transition` there updates multipliers without the stationarity gate that the rest of the algorithm treats as mandatory.

The original algorithm review recommended changing state before the next outer iteration, but specifically described penalty escalation for repeated subproblem-limit exhaustion, not a blind dual update. The plan's risk note says "inner_result.success and constraints are hard-feasible-for-update; that's already the gating condition," but current code proves that is not the dual-update gating condition.

Evidence:

- `alm_utils.py:4228-4265` is the only current dual-update arm and includes the stationarity predicate.
- `alm_utils.py:4267-4318` handles hard-feasible subproblem-limit/plateau after the stationarity-gated dual-update arm did not fire.
- `algorithm_review.md:46-70` recommends escalating the `max_subproblem_continuations` subproblem-limit arm to a penalty-increase path while preserving plateau-stall termination.
- `FIX_PLAN.md:342-346` instead requires a dual update before `BREAK_OUTER`.

What to do instead:

Split the M3 fix. Keep the `gtol` ratchet fix. For subproblem-limit `BREAK_OUTER`, either call the existing penalty-increase arm when the reason is `max_subproblem_continuations`, or return a distinct failure/termination reason that tells the operator the outer budget is being burned without a state change. Do not update multipliers unless the same stationarity gate used at `alm_utils.py:4228` is satisfied.

### M4 - NEEDS-REVISION

What is wrong:

The plan says to add `not multiplier_cap_binding_active` to the converged gate, but it does not define "active" precisely. Current code has sticky run-level diagnostics (`run_state.cap_binding_detected`, `run_state.cap_binding_indices`) that record whether the cap ever bound during a prior dual update. A previously capped multiplier can later move below the cap if subsequent dual-update values are negative; using the sticky flag as a convergence blocker would introduce false failures.

The plan also targets only the regular converged gate. There is a second success arm, `constraints_inactive_converged`, that also emits success and should be considered under the same cap-active policy.

Evidence:

- `alm_utils.py:4241-4251` records cap binding after a dual update and stores it in `run_state.cap_binding_detected`.
- `alm_utils.py:4091-4118` emits regular `converged`.
- `alm_utils.py:4138-4161` emits `constraints_inactive_converged`.
- `alm_utils.py:1556-1575` only tells you whether the last projected update exceeded the cap, not whether a historical cap is still physically active.

What to do instead:

Define a current cap-active predicate. At minimum, use the latest dual update's `multiplier_cap_binding` for the next outer convergence decision, or recompute whether current multipliers are at the cap while the relevant preferred dual update would push further positive. Gate both success arms with the same predicate, and include tests for regular convergence, constraints-inactive convergence, and a historical-cap-then-unbound case.

### M6 - NEEDS-REVISION

What is wrong:

The audit identified SciPy `nnls` `RuntimeError("too many iterations")`. The plan proposes catching `(RuntimeError, ValueError)`. Catching `ValueError` would hide shape and contract bugs in `active_matrix` or `total_grad_array`, which violates the guardrail against defensive catch-all behavior.

The proposed rank-deficient-matrix test is also weak. Rank deficiency does not reliably make SciPy `nnls` raise; it can return a least-squares solution. That test can pass without proving the crash path.

Evidence:

- `alm_utils.py:2070-2073` calls `nnls(active_matrix, -total_grad_array)` and returns the residual.
- `numerics_review.md:37-55` names the crash as SciPy `RuntimeError` from iteration exhaustion and recommends an explicit `maxiter`.

What to do instead:

Catch only `RuntimeError` from `nnls`, return `None`, and log/report a narrow diagnostic. Pass `maxiter=10 * active_matrix.shape[1]` or another explicit bounded value. Test by monkeypatching `module.nnls` to raise `RuntimeError("too many iterations")`; add a separate test proving shape mismatches still raise instead of being swallowed.

### M7 - NEEDS-REVISION

What is wrong:

The normalization audit's floor-source finding was not limited to physics metadata. `hardware_constraint_schema.py` also applies a physical scale floor and still emits `source="threshold:<schema_name>"` or `source="schema:<schema_name>.alm_scale"` with no floor flag. The fix plan says "SSOT" but only lists `_physics_alm_metadata` and `_stage2_alm_constraint_metadata`.

Adding a required field to `ALMConstraintMetadata` also has constructor blast radius. There are direct constructors in tests and production; the field needs a default, and artifact payloads need an explicit list if consumers are expected to see it.

Evidence:

- `hardware_constraint_schema.py:273-277` applies `ALM_PHYSICAL_SCALE_FLOOR`.
- `hardware_constraint_schema.py:330-334` emits the source string without floor information.
- `hardware_constraint_schema.py:46-58` defines `ALMConstraintMetadata`.
- `hardware_constraint_schema.py:383-408` emits metadata payload fields and currently has no `scale_floor_applied` output.
- `single_stage_objectives.py:501-507` and `stage2_objectives.py:723-728` are only two of the floor sites.

What to do instead:

Introduce one helper that returns `(scale, floor_applied, source)` for objective and physical scale floors, or two helpers with the same contract if the floor constants differ. Add `scale_floor_applied: bool = False` at the end of the dataclass to preserve existing constructors. Emit a payload field such as `constraint_scale_floor_applied`, and update single-stage, Stage 2, hardware schema, and tests.

### M8 - NEEDS-REVISION

What is wrong:

The proposed fix validates only single-stage resume JSON. The same corrupted multiplier vector can still be passed through the public ALM driver via `minimize_alm(..., initial_multipliers=...)`, because `_normalize_alm_run_inputs` copies `initial_multipliers` without checking shape, finiteness, or nonnegativity.

Evidence:

- `single_stage_banana_example.py:4342-4366` validates resume names/shape only and returns multipliers/penalty.
- `single_stage_banana_example.py:8661-8664` and `8951-8954` route checkpoint resume state through that helper.
- `alm_utils.py:2171-2175` copies `initial_multipliers` directly or initializes zeros.
- `alm_utils.py:2160-2164` already validates `penalty_max`, showing this is the right boundary for direct-driver validation.

What to do instead:

Add a shared helper in `alm_utils.py` for initial multiplier validation: exact shape, finite, and `>= 0.0` for inequality ALM. Use it from `_normalize_alm_run_inputs`, and either mirror the same helper in the single-stage module or have `validate_resume_alm_state` enforce the same rules before handing values to `minimize_alm`. Add both resume tests and direct `minimize_alm(initial_multipliers=...)` tests.

### L2 - WRONG

What is wrong:

The plan says the mutation is redundant because `update_single_stage_alm_smoothing_from_history` writes to `latest_history_entry`. That is only true for the owned snapshot passed to the callback, not for the live ALM history list. Removing the `history[-1]["smoothing_changed"] = ...` line means emitted partial-state `history` loses `smoothing_changed`; only `latest_history_entry` retains it.

The proposed test also does not match the proposed implementation. A test asserting that arbitrary callback mutation of `history[-1]` has no observable effect requires `_emit_alm_history_snapshot` to pass a copied history list, but the plan only removes this one downstream mutation.

Evidence:

- `alm_utils.py:2379-2385` passes live `history` plus a snapshotted `latest_entry`.
- `single_stage_banana_example.py:2151-2168` mutates the `history_entry` object passed to it.
- `single_stage_banana_example.py:9015-9021` copies `latest_history_entry["smoothing_changed"]` back into live `history[-1]`.
- `single_stage_banana_example.py:6507-6535` emits both `latest_history_entry` and full `history` in partial-state payloads.
- `tests/geo/test_alm_utils.py:2390` currently asserts callback `history` is the live `result.history` object.

What to do instead:

Choose one contract. Strict option: change `_emit_alm_history_snapshot` to pass a copied history list, update the identity test, and let callbacks mutate their copy. Narrow option: keep ALM callback history borrowed/read-only, but make the single-stage callback maintain an owned partial-history copy, then merge `smoothing_changed` into that owned copy before emitting payloads. Either way, add a test that both protects `result.history` and proves partial-state payloads still include `smoothing_changed` where consumers need it.

### T1 - WRONG

What is wrong:

The proposed KKT fixture has incorrect math. For `f(x,y)=0.5*((x-1)^2+(y-1)^2)` with constraint `x+y <= 1`, the active minimizer is `(0.5, 0.5)`, but the KKT multiplier is `lambda=0.5`, not `1.0`:

```text
grad f = (-0.5, -0.5)
grad c = (1, 1)
grad f + lambda grad c = 0 => lambda = 0.5
```

The proposed normalization property is also inverted. The project normalization convention is `lambda_norm = lambda_raw * scale`, so the raw-comparable value is `lambda_norm / scale`, not `lambda_norm * scale`.

The sign-flip test is underspecified. A flipped inequality can still converge successfully, just to the wrong feasible set. The test should compare the final solution and intended raw feasibility, not merely assert "does not label success."

Evidence:

- `normalization_review.md:90-125` states the multiplier conversion: equivalent normalized multiplier changes with scale, and the raw multiplier is recovered by dividing by scale.
- `test_coverage_review.md:280-306` recommends a KKT test and a sign-flip detector, but its scalar sign-flip sketch checks final location and multiplier, not just success.
- `FIX_PLAN.md:892-904` contains the incorrect lambda and inverted scale assertion.

What to do instead:

Correct the analytic KKT fixture before implementation. For the stated 2D fixture, assert `lambda ~= 0.5`; alternatively change the objective coefficient if `lambda ~= 1` is desired. For scale invariance, assert `result_a.multipliers / scale_a ~= result_b.multipliers / scale_b` and primal equality. For sign flip, assert that the known-bad flipped evaluator either fails or returns a point violating the intended original constraint; do not rely on success alone.

## Cross-Cutting Correctness

### H1 / M7 / M8

These fixes all touch validation/provenance boundaries, but the plan currently treats them as separate patches. H1 and M7 should share a scale-resolution helper contract so "reject nonpositive" and "record positive-but-floored" cannot drift. M8 should reuse ALM boundary validation, not duplicate single-stage-only checks.

### M2 / M3

M2 and M3 both affect continuation/outer-loop dispatch. M2 is a labeling fix for `CONTINUE_CONTINUATION` natural exhaustion. M3 proposes changing state advancement on a `BREAK_OUTER` arm, but its dual-update recommendation contradicts the current stationarity-gated dual-update branch at `alm_utils.py:4228-4265`.

### M4 success arms

M4 must cover both success arms: regular `converged` at `alm_utils.py:4091-4118` and `constraints_inactive_converged` at `alm_utils.py:4138-4161`. Gating only one leaves a reachable success path with cap-binding semantics unresolved.

### S1 / T1

S1 changes the objective composition and multiplier interpretation. T1's KKT and normalization tests are exactly the right category of protection, but their current math is wrong; if implemented as written they would either fail or validate the wrong multiplier conversion.

## Order-of-Fix Risk

The proposed order is not the lowest risk.

Recommended order:

1. Correct and land the smallest characterization tests first: fixed T1 KKT/invariance tests, M2 terminal-label tests for both call paths, M3 subproblem-limit characterization, M4 cap-active characterization, and L2 callback-contract tests.
2. Apply low-risk validation/provenance fixes: H1, M7, M8, L1, L3.
3. Apply control-flow changes: M5, M2, M4, corrected M3.
4. Apply design-call changes: S1 and M1 after their contract wording is settled.
5. Backfill the remaining broader T1 properties and run the targeted/full validation suite.

T1 should not be entirely last. The corrected KKT and normalization-invariance tests should land before M3/M4/S1, because those fixes can silently change convergence and multiplier semantics. Some T1 tests that depend on new failure reasons can land with their corresponding fixes.

## Design-Call Assessment

### S1 options A/B/C

I do not agree with recommended Option A as written. Auto-zeroing `LENGTH_WEIGHT` with a warning is a hidden config rewrite, and the condition is too broad because it applies to all ALM modes rather than the `weighted_sum` double-feed case. Prefer a strict Option B variant: make `weighted_sum` ALM require effective `length_weight == 0.0`, update wrappers/default resolution to provide that value explicitly, and fail on explicit nonzero input. Option C is valid only if the base length objective is moved to a separate soft preference target distinct from the hard ALM ceiling; otherwise it preserves the same double-feed.

### M1 options A/B

I agree with Option A, with stronger wording. Current code intentionally uses surrogate signals in the inner augmented objective and hard signals for dual updates; `alm_utils.py:2005-2026` detects hard/surrogate mismatch and `alm_utils.py:4091-4096` blocks regular convergence when mismatch is active. Option B would try to align the inner objective to hard signals after K outer iterations, but the hard signals may be nonsmooth or unavailable as stable gradients, which is exactly why the surrogate/hard split exists. The doc must explicitly say classical ALM convergence claims are not being made for this hybrid mode; the tests should pin deterministic termination and no false-success labels under sustained mismatch.

## Effort Sanity Check

Material underestimates:

| ID | Plan estimate | More realistic | Why |
|---|---:|---:|---|
| S1 | ~2 hr | 3-5 hr | Requires mode-aware default/explicit-input policy, wrapper updates, run identity/result metadata checks, and tests beyond one patched CLI path. |
| M3 | ~1 hr | 3-5 hr | The proposed behavior is wrong; correcting it requires deciding penalty escalation vs termination semantics and preserving existing stationarity-gated dual updates. |
| M4 | ~30 min | 2-4 hr | Needs current-vs-sticky cap semantics, both success arms, a failure helper, and at least three tests. |
| M6 | ~20 min | 1-2 hr | Requires precise exception scope, explicit `maxiter`, and a monkeypatched failure test plus non-swallowed shape-error test. |
| M7 | ~30 min | 2-4 hr | Dataclass schema change, all constructors, hardware schema, artifact payload, and tests must move together. |
| M8 | ~30 min | 1.5-3 hr | Must cover both resume JSON and direct `minimize_alm` API boundary. |
| L2 | ~30 min | 2-4 hr | Requires choosing and testing the callback ownership contract, not just deleting one mutation. |
| T1 | ~2 hr | 4-8 hr | Real ALM KKT fixtures need analytic correction, tolerance tuning, and stable sign/NaN/mismatch scenarios. |

## Missing Items From Original Audits

These are not all blockers for the first implementation wave, but a fix plan titled "Confirmed Issues & Fix Plan" should include them or explicitly defer them.

1. `math_review.md:F4` says `_kkt_stationarity_norm` is a misnamed/miscomputed diagnostic because it uses augmented `metric_grad` instead of bare `base_grad`. T1 adds tests but no code/doc fix for the diagnostic semantics.
2. `physics_review.md:Finding 2` says Stage 2 `--stage2-iota-tolerance` and single-stage `--alm-iota-penalty-threshold` use different operator-facing units. Not in the plan.
3. `numerics_review.md:1.3` and `1.4` identify remaining mutable-alias contracts in `_sanitize_nonfinite_inner_evaluation` and `_build_augmented_evaluation`. L3 fixes only evaluator cache aliasing.
4. `regression_review.md:Smoking Gun Candidate` offers two contract-level resolutions for live history callbacks. L2 addresses only one downstream write and does not settle the broader callback contract.
5. `test_coverage_review.md` missing-test items 6-8 and 10 are not covered by T1: best-feasible monotonicity, penalty-schedule monotonicity over many infeasible outers, full-driver stall-class coverage, and augmented-gradient Taylor consistency.
6. `regression_review.md:21` flags `trust_radius_grow` as CLI-only validation with programmatic `ALMSettings(trust_radius_grow=0.5)` still possible. Not in the plan.

## Final Verdict

This fix plan is not production-ready as written. It has several solid fixes, but S1, M3, L2, and T1 would either implement the wrong behavior or test the wrong mathematical contract, and M7/M8 leave root-cause paths reachable. Revise the plan before implementation begins, then land corrected characterization tests before changing the ALM control-flow and multiplier semantics.
