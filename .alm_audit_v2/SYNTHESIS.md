# ALM Audit v2 Synthesis — 2026-05-08

Branch: `surrogate-confinement-v2` · HEAD `e7b836464` · `examples/single_stage_optimization/alm_utils.py` (4847 LOC)

## Verdict

**ALM mode is NOT bug-free.** The math, algorithm, physics, and computation each have surviving defects after the v1 audit/fix cycle (`bf936a0a4`, `a169f296a`, `2e9acced2`). 8 specialist agents (Opus 4.7, max effort) audited the post-fix code in parallel. They cross-confirmed two ship-blocking issues from independent angles:

| Cross-confirmed bug | Confirming agents | Severity |
| --- | --- | --- |
| `DEFAULT_ALM_LENGTH_PENALTY_THRESHOLD = 0.0` violates `require_positive_alm_threshold` — default invocation crashes | runner, normalization, physics | **CRITICAL** |
| `--alm-iota-penalty-threshold` help+README documented wrong (claims `(ι−ι_target)²·iotas_weight`, actual is `0.5·(ι−ι_target)²` with no weight); single-stage path bypasses Stage-2's `2e9acced2` conversion | physics, runner, normalization | **CRITICAL** (physics) / **HIGH** (operator-facing) |
| M4 cap-active gate omitted from start-of-outer skipped-inner shortcut at `alm_utils.py:3968-4057` — `success=True, "converged"` emitted on cap-bound multipliers | math, algorithm | **HIGH** |
| M9 fix not propagated to `_surrogate_kkt_stationarity_norm` at `alm_utils.py:835-852` — surrogate-side KKT diagnostic collapses to ~0 once L-BFGS-B converges | math, algorithm, numerics | MEDIUM (diagnostic) |

The v1 fix commit (`bf936a0a4`, 11 audit findings closed) lands most fixes correctly: **M2, M3.a, M3.b, M6, M8, L1–L5 are confirmed correct at the cited sites**. Math primitives (augmented form, multiplier projection, penalty schedule, gradient sign) match Bertsekas (1982). Frame conventions, sign conventions (`c ≤ 0 ⟺ feasible`), and ACCEPT_OFFSPEC removal (`d61648f50`) are all clean.

The surviving issues fall into three buckets: (1) **incomplete fix application** to sibling sites — M4 missed L3968 shortcut, M5 missed L4250 history-attach, M9 missed `_surrogate_kkt_stationarity_norm`; (2) **operator-facing correctness** — broken default threshold, wrong iota help text, missing help= on 4 flags; (3) **regression-pin gaps** — M4/M5/M6/M9/L1 fixes have no test that would catch their regression.

---

## Severity-ranked fix list

### CRITICAL (ship blockers)

**C1. Runner default crashes against post-`a169f296a` validator** (`run_single_stage_thresholded_physics_alm.py:43`)
- `DEFAULT_ALM_LENGTH_PENALTY_THRESHOLD = 0.0` → `require_positive_alm_threshold` rejects `≤ 0` → `ValueError` on every default invocation.
- Same constant referenced from `run_stage2_to_single_stage.py:538-542` propagates to recovery handoff.
- Confirmed by: runner F1, normalization F1, physics F2.
- Fix: pick a defensible non-zero default, OR adopt `default=None + append_optional_flag` pattern from `run_single_stage_goal_mode_comparison.py:252-263` so the parser raises with an actionable "missing required flag" error.

**C2. Iota threshold operator-facing units are silently wrong by factor √(2·iotas_weight)** (`single_stage_banana_example.py:1397-1416`, `README.md:576-593`)
- Help+README claim constraint is `(ι − ι_target)² · iotas_weight ≤ T` and tell operator to convert via `T = d² · iotas_weight`.
- Actual constraint: `0.5·(ι − ι_target)² − T ≤ 0` (no `iotas_weight` factor) because `Jiota = QuadraticPenalty(iota_term, iota_target)` and the constraint is on `Jiota.J()` (= `0.5·diff²`), not `iotas_weight·Jiota`.
- With default `--iotas-weight 100` and target deviation 0.01, an operator following the README sets `T=1e-2` but the actual feasible band is `|ι − target| ≤ √(2·1e-2) ≈ 0.141` — **14× looser** than intended.
- Stage-2 path was correctly documented in `2e9acced2`; only single-stage was missed.
- Confirmed by: physics F1, runner F2, normalization F2.
- Fix: rewrite help+README to state `0.5·(ι−ι_target)² ≤ T → T = 0.5·d²`. Cross-link the `2e9acced2` Stage-2 doc and add a unit test that pins `Jiota.J() == 0.5*diff²` to prevent the help text drifting again.

### HIGH

**H1. M4 cap-binding gate omits the start-of-outer skipped-inner shortcut** (`alm_utils.py:3968-4057`)
- The third success arm (skipped-inner) emits `success=True, termination_reason="converged"` without checking `not run_state.last_cap_binding_active`. The other two converged arms (post-inner L4264-4274, constraints-inactive L4316-4321) carry the gate.
- Reachable: outer K dual-update clamps λ at `multiplier_max` → `last_cap_binding_active=True`; outer K+1 start-of-outer re-evaluates at same x with clamped multipliers → tolerances satisfied → shortcut fires → false `converged`.
- Downstream consumers (autoresearch promotion, frontier reporting, run_single_stage campaigns) gate on `result.success`. Silent wrong-answer at the success boundary.
- Confirmed by: math F1, algorithm F1.
- Fix: add `and not run_state.last_cap_binding_active` to the gate at L3968-3973, mirroring L4273. Pair with a regression test (see T1 below).

**H2. Stage-2 signal fields outside `_nonfinite_evaluation_fields` whitelist → silent NaN propagation** (`alm_utils.py:1271-1284`)
- `hard_signed_constraint_values`, `hard_violation_values`, `surrogate_signed_constraint_values`, `hard_dual_update_values` participate in routing/dual-update decisions but are not in the array-validation list. A NaN in `hard_dual_update_values` reaches `_handle_alm_dual_update_transition` unsanitized.
- Combined with H3 below, NaN multipliers silently propagate with `multiplier_cap_binding=False`.
- Confirmed by: numerics F2.
- Fix: add the four Stage-2 signal field names to the whitelist alongside the existing scalar fields.

**H3. `_project_nonnegative_multipliers_with_diagnostics` cap mask `updated > cap` is False for NaN** (`alm_utils.py:1655-1674`)
- `np.any(np.array([nan]) > 1.0) == False`; `np.minimum(NaN, cap) == NaN`. NaN multipliers pass through with `cap_binding_mask=False` and the diagnostic lies.
- Confirmed by: numerics F3.
- Fix: add an explicit `np.isfinite(updated).all()` assert (or `np.where(np.isnan(updated), cap, updated)` if a NaN-tolerance choice is made deliberately) before the cap comparison.

**H4. M9 fix not applied to `_surrogate_kkt_stationarity_norm`** (`alm_utils.py:835-852`)
- Sibling diagnostic still feeds augmented `metric_grad` (or `evaluation["grad"]`) into `_kkt_stationarity_norm`. Reported `surrogate_kkt_stationarity_norm` and `final_surrogate_kkt_stationarity_norm` collapse to ~gtol once L-BFGS-B converges, regardless of multiplier quality — the exact symptom M9 was written to fix on the non-surrogate side.
- Diagnostic-only — does not affect `result.success` directly, but operators rely on it to assess multiplier quality.
- Confirmed by: math F2, algorithm F3, numerics F1.
- Fix: mirror the M9 pattern — read `evaluation.get("base_grad", evaluation.get("metric_grad", evaluation["grad"]))`.

**H5. Hybrid signal contract doc line citations off by 64-130 lines** (`docs/alm_hybrid_signal_contract_2026-05-08.md`)
- Doc cites converged-gate guard at L4133-L4138; actual is L4264-L4274. `_extract_stage2_constraint_signal_state` cited at L1915-L1953; actual L1979-L2037. The doc is the contract SSOT; its "forbids future refactors" clause is defeated when the cited lines do not match the code.
- Confirmed by: hybrid F1.
- Fix: regenerate line citations against HEAD, add a CI check that the cited lines still parse to the named functions.

**H6. Dual-update arm at `alm_utils.py:4411` mutates λ under sustained mismatch when `hard_feasible_for_update=True ∧ hard_feasible_strict=False`**
- The signal-mismatch arm at L4364 only gates on `hard_feasible_strict`; the relaxed-tol regime continues dual updates while the inner solve was minimizing the surrogate Lagrangian. The contract doc claims the safeguard "structurally blocks false-success labeling" but does not document that dual mass continues to accumulate under mismatch.
- Confirmed by: hybrid F2.
- Fix: tighten the gate, OR document this regime explicitly in the contract doc and add a test that pins the behavior.

**H7. `_surrogate_hard_sign_mismatch` uses `np.sign(0.0) == 0.0` → spurious mismatch at exact boundary** (`alm_utils.py:826-832`)
- `hard=0.0, surrogate=-1e-3` flags `mismatch=True` (signs `0.0` vs `-1.0`). This diagnostic is consumed by adaptive smoothing in `single_stage_banana_example.py:2117-2132`, causing unwanted smoothing-shrink at exact-boundary iterates — a real production-mode bug.
- Confirmed by: hybrid F3.
- Fix: treat zero as match (`abs(hard) < tol` → not mismatched), or use `np.sign(hard) * np.sign(surrogate) < 0` semantics.

**H8. Help text stripped from 4 physics threshold flags** (`run_single_stage_thresholded_physics_alm.py:190-205`)
- `--alm-qs-threshold`, `--alm-boozer-threshold`, `--alm-iota-penalty-threshold`, `--alm-length-penalty-threshold` show no unit guidance to operators — explicit regression vs commit `2e9acced2`'s "operator-facing units" goal.
- Confirmed by: runner F3, runner F7.
- Fix: adopt `default=None + append_optional_flag` from `run_single_stage_goal_mode_comparison.py:252-263`.

**H9. `run_stage2_alm.py` basin-seed via `os.urandom(4)` breaks artifact reuse and reproducibility** (`run_stage2_alm.py:367-372`)
- `_normalize_basin_seed` calls `os.urandom(4)` when JSON spec sets `basin_hops > 0` with `basin_seed=None`. The seed is encoded into the artifact path via `format_stage2_basin_suffix`, so each run materializes a NEW path.
- `--basin-seed` is not a CLI flag; the JSON spec is the only entrypoint.
- Confirmed by: runner F4.
- Fix: expose `--basin-seed` as a CLI flag, default to a deterministic value, log the chosen seed.

**H10. M5 fix incomplete: `_attach_alm_history_diagnostics` still uses `state.update_feasibility_tol`** (`alm_utils.py:4250`)
- Post-inner `_constraint_routing_state` and `_stationarity_metrics` correctly switched to `effective_feasibility_tol`, but the immediately-following `_attach_alm_history_diagnostics` uses the unclamped `state.update_feasibility_tol`. History entry's `surrogate_kkt_stationarity_norm` uses a different active-set gate than the routing flags in the same entry.
- Confirmed by: algorithm F2.
- Fix: thread `effective_feasibility_tol` to `_attach_alm_history_diagnostics`.

**H11. M2 feasible-update retry call site is untested** (`alm_utils.py:4534`)
- `_emit_alm_subproblem_continue` is correctly threaded with `is_final_outer` at both L4408 (signal-mismatch retry) and L4534 (feasible-update retry). Tests cover the signal-mismatch path; the feasible-update path is **not** tested. v1 `FIX_PLAN_REVIEW.md` flagged this in v1 as INSUFFICIENT-TESTS; recommendation was not followed.
- Confirmed by: algorithm F4.
- Fix: add `test_alm_subproblem_continue_marks_max_outer_in_feasible_update_retry`.

**H12. `coil_length_upper_bound` and `length_penalty` both active in thresholded_physics mode** (`single_stage_banana_example.py:326-331`, `single_stage_objectives.py:787-792` and `:938-942`)
- Two ALM constraints enforce overlapping `L ≤ L_target` inequalities (linear m vs squared m²). Same redundancy as prior audit Finding 1 in weighted_sum mode; the prior fix only blocked weighted_sum mode.
- Confirmed by: physics F3.
- Fix: pick one formulation per mode, or scale them so they enforce different limits intentionally.

### MEDIUM

| ID | Issue | File:Line | Source |
| --- | --- | --- | --- |
| M1 | `validate_alm_cli_args` still missing positivity checks for the 4 physics thresholds (v1 H1 step 143) | `alm_utils.py:389-432` | normalization F3 |
| M2 | `scale_floor_applied: bool` not serialized in `alm_constraint_metadata_payload`; provenance only as `:floored` source-string suffix | `hardware_constraint_schema.py:413-438` | normalization F4 |
| M3 | Stage-2 hard signal normalization bypasses `normalize_alm_constraint_signals` SSOT | `stage2_objectives.py:1918-1921` | normalization F5 |
| M4 | Property test asserts re-run determinism but doesn't pin a specific terminator label | `tests/geo/test_alm_utils.py:2359-2491` | hybrid F4 |
| M5 | Property test only covers surrogate-active-hard-inactive polarity; opposite-polarity `signal_mismatch_stall` arm at L4367 unreachable through fixture | tests | hybrid F5 |
| M6 | Legacy non-explicit-stage2 path silently disables `signal_mismatch_active`; `alm_fixture_benchmarking.py:166-193` runs with no mismatch detection | `alm_utils.py:2017-2105` | hybrid F6 |
| M7 | No test asserts dual update specifically uses hard channel; fixture `_routing_state_with_preferred` sets `hard == surrogate` | tests | hybrid F7 |
| M8 | `ALMSettings.__post_init__` and `validate_alm_cli_args` use `<=`/`<` checks that silently accept NaN (`nan <= 0.0` is False) | `alm_utils.py:60-77, 410-432` | numerics F4 |
| M9 | `ALM_PHYSICAL_SCALE_FLOOR = sys.float_info.epsilon` (~2.22e-16) — user-supplied 1e-15 m threshold floors to ~eps; downstream amplification yields ~4.5e23 positive-shift | `hardware_constraint_schema.py:30` | numerics F5 |
| M10 | `_termination_reason_from_history` has no specific case for new `subproblem_limit_penalty_increase` action; returns bare `"max_outer"` | `alm_utils.py:1779-1795` | algorithm F5 |
| M11 | Runners emit summary JSON via `json.dump(... indent=2)` without `allow_nan=False`; bypass `workflow_runner_common.write_json` guard | runners | runner F5 |
| M12 | `run_stage2_alm.py` doesn't call `validate_alm_cli_args`; `--alm-distance-smoothing`, `--alm-curvature-smoothing` validated only inside subprocess after lock+spawn | `run_stage2_alm.py:798-812` | runner F6, normalization F7 |
| M13 | Stage-2 iota-penalty `activity_tolerance == scale` → normalized = 1.0 vs 1e-3 for hardware constraints; iota constraint effectively always active | `stage2_objectives.py:1303-1305` | normalization F8 |

### LOW

| ID | Issue | Source |
| --- | --- | --- |
| L1 | `_kkt_stationarity_norm` declares unused `feasibility_gate`/`feasibility_values` params | math F3 |
| L2 | `_explicit_raw_signed_constraint_values` returns the **surrogate** raw — misleading name | hybrid F8 |
| L3 | `_emit_alm_stall_failure_step` doesn't annotate `outer_termination` on history (asymmetric vs other arms) | hybrid F9 |
| L4 | `_directional_taylor_result` divides by `2*epsilon` without validating epsilon | numerics F6 |
| L5 | `alm_raw_dual_estimates` divides by `constraint_scales` without finiteness/positivity validation | numerics F7 |
| L6 | `_kkt_stationarity_norm` activity rule diverges from `_constraint_activity_mask` | numerics F8 |
| L7 | `_normalize_trust_radius` accepts `+inf`; profile selection silently changes | numerics F9 |
| L8 | `_resolved_threshold` skips positivity validation on user override (defended downstream) | normalization F6 |
| L9 | `alm_fixture_benchmarking` records `seed` but never seeds any RNG | runner F8 |
| L10 | Initial multiplier validation not bounded by `multiplier_max` | algorithm F6 |
| L11 | `x0` finiteness validation gap in `_normalize_alm_run_inputs` | algorithm F7 |

---

## Test-coverage regression risk

The test-coverage agent found **5 CRITICAL gaps** where v1 fixes have **no test that would catch a regression** of the fix. These are not bugs in the production code, but they mean the v1 fixes are unprotected:

| v1 fix | Production site | Test status |
| --- | --- | --- |
| **M4** (cap-active gates `converged` label) | `alm_utils.py:4264-4296`, `:4316-4346`, `:4434` | only diagnostic asserted; no negative test would catch deletion of the gate |
| **M5** (post-inner routing tol consistency) | `alm_utils.py:4118-4124` | comment-only (line 3867); no assertion |
| **M6** (`nnls` `RuntimeError` only) | `alm_utils.py:2170-2188` | catch arm dead from test perspective |
| **M9** (KKT diagnostic uses `base_grad`) | `alm_utils.py:2228-2231` | augmented-vs-bare gradient distinction not asserted |
| **L1** (no-blocks lane shallow-copy) | `alm_utils.py:2360-2364` | no aliasing test |

Plus 9 HIGH structural gaps surviving from v1 audit (no end-to-end multi-constraint multiplier-sign assertion, no normalization invariance under non-trivial scale, NaN robustness only covers `total`/`grad`, best-feasible monotonicity untested, penalty-schedule monotonicity unasserted, 12-constraint test is a smoke, `validate_alm_cli_args` patched-out everywhere, 14 `ALMSettings.__post_init__` rejection arms with only 3 tested, `_emit_alm_*` retry arms have no direct unit tests).

Worth noting: 5 outer-iteration tests (test_alm_utils.py:4584-4685) **mock `_run_alm_continuation_step` entirely** — the actual control-flow under audit is not exercised by them.

---

## Per-domain verdict

| Domain | Verdict | Notes |
| --- | --- | --- |
| **Math (KKT, dual update, augmented form, penalty schedule)** | Almost bug-free; 1 HIGH (H1 cap-binding gate) + 1 MEDIUM (H4 surrogate-side M9). Math primitives match Bertsekas (1982). | Fix H1 reduces math to bug-free at primitive level. |
| **Algorithm (control flow, retry arms, terminator labels)** | 3 surviving issues from v1 fix; M2/M3.a/M3.b/M6/M8 confirmed correct. | H1, H10, H11, M10. |
| **Physics (constraint formulation, sign conventions, units, frame)** | 1 CRITICAL operator-facing bug (C2) + 1 MEDIUM redundancy (H12). Sign conventions, MAJOR_RADIUS frame, ACCEPT_OFFSPEC removal all confirmed correct. | C2 is silent-wrong-physics. |
| **Computation (numerics, stability, NaN/Inf)** | Silent-NaN propagation chain (H2 + H3) + diagnostic-only H4 + M8/M9 NaN-permissive validation + scale floor too low. | F2+F3 chain is the most subtle path: NaN out of Stage-2 evaluator silently corrupts multipliers and flatters KKT diagnostic, with the only crash deferred to next-outer `_require_finite_evaluation`. |
| **Normalization (scale floors, block-penalty removal)** | Block-penalty removal complete. Scale-floor provenance landed (a169f296a). 1 SSOT bypass (M3) + 1 schema serialization gap (M2). | Mostly clean. |
| **Hybrid surrogate/hard contract** | Structural intent honored at production choke points. 3 HIGH (H5 doc drift, H6 dual update under mismatch, H7 sign(0)) + 4 MEDIUM coverage gaps. | Doc citations are stale by 64-130 lines. |
| **Test coverage** | 5 CRITICAL pin gaps where v1 fixes are unprotected. 9 HIGH structural gaps from v1 unaddressed. | Strongest fix-test coverage: M3.a, M3.b, M7, M8, T1.a, T1.e, H1, L2. |
| **Runner CLI surface** | 1 CRITICAL (C1 default crash) + 3 HIGH (C2/H8/H9). ACCEPT_OFFSPEC removal complete; constraint registration order deterministic. | C1 means default invocation is broken today. |

---

## Recommended fix order

1. **C1** (1-line default change + add CLI test) and **C2** (rewrite help+README + add `Jiota.J() == 0.5*diff²` pin test) — ship blockers, both <30 minutes each.
2. **H1** (1-line gate + regression test) — silent wrong-success boundary.
3. **H2 + H3 + H4** together — silent-NaN chain. Add the 4 Stage-2 fields to the whitelist, add `np.isfinite` assert before cap comparison, mirror M9 to surrogate sibling. Pair with NaN-injection tests.
4. **H10 + H11** — finish the M5/M2 work. One-line edits each.
5. **H5** — regenerate doc line citations + CI line-citation check.
6. **H6 + H7** — tighten dual-update gate semantics + zero-sign mismatch.
7. **H8 + H9** — runner CLI ergonomics.
8. **H12** — pick one length constraint formulation per mode.
9. **Test pin gaps T1–T5** for M4/M5/M6/M9/L1 — protect existing v1 fixes from regression.
10. MEDIUM/LOW backlog as time permits.

---

## Coverage gap

The Codex independent review (gpt-5.5 xhigh) **failed** — agent stalled in the writing phase >600s without progress. Cross-cutting bugs that would have surfaced from a Codex pass are partially substituted by the overlap among the 8 specialist reports: H1 (M4 cap-binding gate) was independently flagged by math + algorithm; H4 (M9 surrogate sibling) by math + algorithm + numerics; C1/C2 (zero-default + iota help) by 3 agents each. A future audit pass should re-run Codex with a tighter scope or via `codex:codex` skill rather than `codex:codex-rescue` agent.

---

## Files

- 8 specialist reports: `.alm_audit_v2/{math,algorithm,physics,numerics,normalization,hybrid_signal,test_coverage,runner}_review.md`
- Synthesis: `.alm_audit_v2/SYNTHESIS.md` (this file)
- Prior v1 audit and fix plan: `.alm_audit/`
- Hybrid signal contract: `docs/alm_hybrid_signal_contract_2026-05-08.md`
