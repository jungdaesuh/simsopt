# ALM Hybrid Surrogate/Hard Signal Contract Audit v2 — 2026-05-08

Branch: `surrogate-confinement-v2`
HEAD: `e7b836464`
Reviewer: Claude Code (Opus 4.7, 1M context)
Files audited:
- `docs/alm_hybrid_signal_contract_2026-05-08.md` (contract spec, lines 1-86)
- `examples/single_stage_optimization/alm_utils.py` (4847 LOC)
- `examples/single_stage_optimization/banana_opt/stage2_objectives.py` (signal-producer side)
- `examples/single_stage_optimization/banana_opt/single_stage_objectives.py` (metadata routing)
- `examples/single_stage_optimization/banana_opt/alm_fixture_benchmarking.py` (legacy-path caller)
- `tests/geo/test_alm_utils.py` (property test + supporting fixtures)
- `tests/geo/test_single_stage_alm_integration.py` (integration tests)

## Summary

The hybrid surrogate-vs-hard contract is structurally honored at the production
choke points: stage-2 evaluation feeds the augmented-Lagrangian inner objective
the **surrogate** signed values and stores the **hard** dual-update channel as a
side-by-side field; `_extract_stage2_constraint_signal_state` selects
`hard_dual_update_values` as `preferred_dual_update_values`; and the converged
gate is guarded by `not signal_mismatch_active`. The deterministic-termination
property test pins the blocking of the success label under sustained mismatch.

However, the contract has eight observable gaps, **two HIGH** and one **MEDIUM**
that materially weaken the safety rail; the rest are documentation / coverage
hygiene that may invite regressions:

1. **F1 (HIGH)** — Every line citation in
   `docs/alm_hybrid_signal_contract_2026-05-08.md` is wrong by 64-130 lines:
   the doc says `_extract_stage2_constraint_signal_state` is at L1915-1953
   but it's at L1979-2037; the converged-gate guard at L4133-4138 is at
   L4264-4274; etc. The doc is the SSOT of the contract and operators cite
   the wrong lines today, defeating its own forbids-future-refactors clause.
2. **F2 (HIGH)** — The dual-update arm at L4411 fires regardless of
   `signal_mismatch_active`. When `hard_feasible_for_update=True` (relaxed
   tol) but `hard_feasible_strict=False` (settings.feasibility_tol),
   λ updates under sustained mismatch even though the inner solve was
   minimizing the surrogate Lagrangian. The contract document (line 47)
   claims the safeguard "structurally blocks false-success labeling" but
   does NOT document that dual updates can still fire and accumulate
   λ-mass under mismatch. This is a subtle deviation from
   "the algorithm is operating outside its theory" toward "the algorithm
   is actively mutating dual state under mismatch."
3. **F3 (HIGH)** — `_surrogate_hard_sign_mismatch` (L826-832) uses
   `np.sign`, which returns 0.0 for exact zero. So `hard=0.0` and
   `surrogate=-1e-3` produces `mismatch=True` (signs `0.0` vs `-1.0`).
   This diagnostic is **consumed by adaptive smoothing** in
   `single_stage_banana_example.py:2117-2132` (`normalized_hard_surrogate_gap_counts`),
   where any per-constraint mismatch flag triggers smoothing-shrink.
   Spurious shrink at exact boundaries is a real production-mode bug.
4. **F4 (MEDIUM)** — The deterministic-termination property test at
   `tests/geo/test_alm_utils.py:2359-2491` asserts the two re-runs agree
   on the termination_reason but does NOT pin a specific label. A
   regression that changes the terminator (e.g., from `max_outer` to
   `signal_mismatch_stall`) passes the test silently. The test claim
   in its docstring (line 2367) says it "pins" the contract; the actual
   pin is weaker than claimed.
5. **F5 (MEDIUM)** — The property test only exercises one polarity of
   mismatch (surrogate active + hard inactive → `signal_mismatch_penalty_increase`
   arm). The opposite polarity (hard active + surrogate inactive →
   `signal_mismatch_stall` arm at L4367) is unreachable through the test
   fixture because the fixture sets `surrogate_signed_constraint_values=0.2`
   (positive shift > 0). The contract has two retry arms; the test covers one.
6. **F6 (MEDIUM)** — In the legacy non-explicit-stage2 path
   (`_extract_stage2_constraint_signal_state`, L2017-2021), the routing
   collapses: `hard_signed_constraint_values = dual_update_values`
   (which can itself be the SURROGATE per
   `single_stage_objectives.py:574`'s `dual_update_value_kind="surrogate"`
   metadata). Then `signal_mismatch_active` is gated on
   `explicit_stage2_signals` (L2105) so it can NEVER trigger in the
   legacy path. **The contract's safeguard is silently disabled for any
   evaluator that doesn't set the four explicit stage-2 fields.**
   `examples/single_stage_optimization/banana_opt/alm_fixture_benchmarking.py:166-193`
   is one such caller — it goes through the legacy path with no mismatch
   detection ever firing.
7. **F7 (MEDIUM)** — There is **no test** that constructs a routing
   state with `hard_signed_constraint_values ≠ surrogate_signed_constraint_values`
   and verifies the dual update specifically uses the **hard** channel.
   The dual-update unit tests
   (`tests/geo/test_alm_utils.py:969-1025`) all use a degenerate
   `_routing_state_with_preferred` fixture where
   `hard == surrogate == preferred`, so the test passes
   even if `_handle_alm_dual_update_transition` were to swap to the
   surrogate channel. This is the missing "is the hard signal actually
   reaching the dual update?" test.
8. **F8 (LOW)** — `_explicit_raw_signed_constraint_values` (L678-686)
   returns the SURROGATE raw signed values (falls back to
   `raw_surrogate_signed_constraint_values`), not the hard. The function
   name reads as "the explicit raw signed constraint" which to a reader
   sounds like the hard channel. It is diagnostic-only (consumed by
   `_constraint_history_diagnostics_source` and result-builders), but
   the misleading name invites future refactor confusion.
9. **F9 (LOW)** — `_emit_alm_stall_failure_step` (L3646-3693) does NOT
   accept `is_final_outer` and never sets `outer_termination="max_outer"`
   on the history entry. The `signal_mismatch_stall` and
   `constraints_inactive_stall` arms are RETURN-arms (build their own
   `failure_result`), so the missing annotation is observationally inert
   for `result.termination_reason`, but the history entry on the final
   outer iteration is missing the `outer_termination` field that all
   other terminal arms set. This is asymmetric history bookkeeping —
   a downstream consumer that relies on `outer_termination` as a
   "this was a terminal entry" marker will miss these two arms.

## Methodology

1. Read the contract document end-to-end (86 lines).
2. Loaded all line citations against the current alm_utils.py tree at
   HEAD `e7b836464` and recorded actual line numbers.
3. Walked the constraint-routing path from the inner objective at
   `stage2_objectives.py:1951-1958` through `_extract_stage2_constraint_signal_state`
   (`alm_utils.py:1979-2037`) to `_handle_alm_dual_update_transition`
   (`alm_utils.py:3090-3120`).
4. Walked the converged-gate guard, the constraints_inactive arm, the
   signal-mismatch retry/stall arms, and the dual-update arm in
   `_run_alm_continuation_step` (L3886-4542).
5. Traced `signal_mismatch_active` from construction
   (`_constraint_routing_state`, L2079-2137) through every consumer.
6. Checked `_surrogate_hard_sign_mismatch` for boundary edge cases
   (`np.sign(0.0)=0.0` flags spurious mismatches against negative
   surrogate).
7. Read the property test (`test_alm_utils.py:2359-2491`) and matched
   its fixture against the arm dispatch in `_run_alm_continuation_step`
   to identify which arm is actually exercised.
8. Read the prior audits at `.alm_audit/{algorithm,math,FIX_PLAN,FIX_PLAN_REVIEW}_review.md`
   to avoid duplicating findings.

Coverage verified by grep for: `_surrogate_hard_sign_mismatch`,
`_surrogate_kkt_stationarity_norm`, `_constraint_routing_state`,
`_explicit_raw_signed_constraint_values`, `alm_raw_dual_estimates`,
`_raw_dual_estimates`, `_extract_stage2_constraint_signal_state`,
`_handle_alm_dual_update_transition`, `signal_mismatch_active`,
`hard_dual_update_values`, `preferred_dual_update_values`,
`augmented_inequality_objective`, `dual_update_value_kind`,
`explicit_stage2_signals`.

## Findings

### F1: Every line citation in the hybrid contract document is stale [HIGH]

- **File**: `docs/alm_hybrid_signal_contract_2026-05-08.md`
- **Code (doc)**:
  > `examples/single_stage_optimization/alm_utils.py:1915-1953` — `_extract_stage2_constraint_signal_state` selects `hard_dual_update_values` as `preferred_dual_update_values`...
  >
  > `examples/single_stage_optimization/alm_utils.py:2988-2995` — `_handle_alm_dual_update_transition` projects new multipliers using `routing_state.signal_state.preferred_dual_update_values`...
  >
  > `examples/single_stage_optimization/alm_utils.py:4133-4138` — the converged branch requires `not signal_mismatch_active`...
  >
  > `examples/single_stage_optimization/alm_utils.py:2007-2022` — `_constraint_routing_state` builds both `hard_activity_mask` and `surrogate_activity_mask`...
  >
  > `examples/single_stage_optimization/alm_utils.py:4224-4244` — `signal_mismatch_stall` arm
  >
  > `examples/single_stage_optimization/alm_utils.py:3851-3855` — skipped-inner shortcut guard
  >
  > `examples/single_stage_optimization/banana_opt/stage2_objectives.py:1944` — `augmented_inequality_objective(...)` is called with `normalized_surrogate_signed_constraint_values`

- **Bug**: Every cite drifts by 7-130 lines against the current tree.
  Verified actuals at HEAD `e7b836464`:

  | Doc cite | Doc says | Actually at | Drift |
  |---|---|---|---|
  | `_extract_stage2_constraint_signal_state` | 1915-1953 | 1979-2037 | +64 |
  | `_handle_alm_dual_update_transition` body using preferred_dual_update_values | 2988-2995 | 3090-3120 (call at 3102) | +102 |
  | converged-gate guard | 4133-4138 | 4264-4274 | +131 |
  | `_constraint_routing_state` mismatch detection | 2007-2022 | 2079-2137 (mask compare at 2104-2105) | +82 |
  | boundary disagreement | 2036-2042 | 2119-2125 | +83 |
  | `signal_mismatch_stall` arm | 4224-4244 | 4364-4400 | +140 |
  | skipped-inner converged-gate | 3851-3855 | 3968-3973 | +118 |
  | `stage2_objectives.py` augmented_inequality_objective call | 1944 | 1951 | +7 |

- **Why**: The contract document was committed 2026-05-08 (today) but the
  actual line numbers reflect the post-refactor state at HEAD
  `e7b836464`. The doc was written against an earlier intermediate
  state and committed without re-resolving the line numbers. Recent
  commits (`bf936a0a4`, `a169f296a`, `2e9acced2`, `3671c479c`,
  `e7b836464`) refactored the ALM driver and ALL cited line numbers
  shifted.

- **Impact**: The contract document is the SSOT for the surrogate-vs-hard
  signal split. Its "What this contract forbids" section (lines 60-68)
  cites the converged-gate guard at L4133-4138 as load-bearing. An
  operator who reads the doc, navigates to L4133-4138, and sees
  unrelated code (this range is inside the converged-arm's
  `_emit_alm_converged_step` invocation, not the converged-arm gate
  itself) will conclude the contract is misdocumented or already
  violated. Worse, a future refactor that updates L4133-4138 (a
  different code region today) would not register as a contract change
  per the doc's text. The forbids-future-refactors clause is unable to
  protect a citation it cannot resolve.

- **Suggested fix**: Re-resolve every cite. Use line ranges rather
  than single lines to absorb minor refactors, and add a short
  function-name anchor inside each cite (e.g., `alm_utils.py:4264-4274
  inside _run_alm_continuation_step's converged arm`) so the
  citation survives line-number drift. Add a CI check that greps the
  cited symbols and asserts they exist somewhere in the file (cheap
  lifetime guard).

---

### F2: Dual update is not gated by `signal_mismatch_active` [HIGH]

- **File**: `examples/single_stage_optimization/alm_utils.py:4411-4452`
- **Code**:
  ```python
  if hard_feasible_for_update and stationarity_norm <= state.update_stationarity_tol:
      state.feasible_stall_count = 0
      dual_update = _handle_alm_dual_update_transition(
          multipliers=state.multipliers,
          routing_state=routing_state,
          ...
      )
      state.multipliers = dual_update.multipliers
  ```

- **Bug**: The dual-update arm at L4411 has no `signal_mismatch_active`
  guard. The signal-mismatch arm at L4364 only fires if
  `hard_feasible_strict` (`hard_max_violation ≤ settings.feasibility_tol`).
  When `hard_feasible_for_update=True` (relaxed tol up to
  `effective_feasibility_tol`) AND `hard_feasible_strict=False` AND
  `signal_mismatch_active=True` AND `stationarity_norm ≤ update_stationarity_tol`,
  the dual update fires under sustained mismatch.

- **Why**: This is the regime where (a) the hard violation has decreased
  enough to clear the relaxed gate but not the strict gate, (b) the
  inner subproblem converged on the surrogate to within the loose
  stationarity tol, and (c) the surrogate and hard activity masks
  disagree (e.g., surrogate says some constraint is active, hard says
  it's not). The signal-mismatch arm specifically excludes this case
  (it requires `hard_feasible_strict`). The constraints-inactive arm
  also excludes it (requires `not signal_mismatch_active`). So the
  control flow falls through to L4411, where λ is updated using
  `routing_state.signal_state.preferred_dual_update_values` (the hard
  channel) at an iterate `x_k` that minimized the **surrogate**
  Lagrangian.

- **Impact**: The contract document at line 47 claims the safeguard
  "structurally blocks false-success labeling" — true; but it doesn't
  block dual-state mutation. Under sustained mismatch in the
  loose-feasibility regime, λ accumulates updates each outer
  iteration. This can:
  1. Push λ toward the multiplier cap (L1556-1575 caps at
     `settings.multiplier_max=1e6` default), at which point M4's
     `last_cap_binding_active` blocks the converged label as a
     secondary safeguard.
  2. Drive `λ + μ * c_surrogate > 0` for constraints where hard says
     feasible (boundary mismatch becomes self-sustaining).
  3. Bias the inner subproblem's augmented-Lagrangian gradient at
     subsequent iterations (the augmenting term `(λ + μc)∇c` grows
     with λ).

  None of these are theory failures of the safeguard itself — the
  converged label is correctly blocked. But the dual state is allowed
  to drift in a direction informed by the hard signal at iterates that
  minimized the surrogate problem. This is exactly the "hybrid forfeits
  Bertsekas convergence" trade-off that the contract document forfeits
  on lines 28-35; but the doc does NOT explicitly state that **the
  dual update is allowed to fire under mismatch**, leaving an
  operator/refactorer to assume it is suppressed.

- **Suggested fix**: Either (a) add `not signal_mismatch_active` to the
  L4411 condition and route to a new "mismatch-feasible-update-blocked"
  arm that records the dual-update-skipped event in history (so the
  operator sees that the algorithm refused to update in mismatch),
  or (b) explicitly document in the contract doc, in a new "dual
  update mutation under mismatch" subsection, that the dual update
  fires under mismatch when `hard_feasible_for_update` is met and
  the converged guard catches the false-success at result-build time.
  Option (b) is the YAGNI choice; option (a) tightens the safety rail.

---

### F3: `_surrogate_hard_sign_mismatch` flags spurious mismatches at exact-zero hard values [HIGH]

- **File**: `examples/single_stage_optimization/alm_utils.py:826-832`
- **Code**:
  ```python
  def _surrogate_hard_sign_mismatch(
      surrogate_signed_values: np.ndarray,
      hard_signed_values: np.ndarray,
  ) -> list[bool]:
      surrogate_signs = np.sign(np.asarray(surrogate_signed_values, dtype=float))
      hard_signs = np.sign(np.asarray(hard_signed_values, dtype=float))
      return (surrogate_signs != hard_signs).tolist()
  ```

- **Bug**: `np.sign(0.0) = 0.0`, `np.sign(-1e-3) = -1.0`. So a constraint
  with `hard=0.0` (exactly satisfied at the boundary) and `surrogate=-1e-3`
  (active region of surrogate well inside the feasible side) gives
  `0.0 != -1.0 → mismatch=True`. Worse: `hard=+1e-15` (computational
  noise of zero) and `surrogate=-1e-3` gives `+1.0 != -1.0 →
  mismatch=True`. The threshold for "mismatch" is implicitly
  `|hard| > 0` rather than a tolerance.

- **Why**: `np.sign` is a strict three-valued function (-1, 0, +1)
  designed for sign extraction without tolerance. The downstream
  consumer (`normalized_hard_surrogate_gap_counts` at
  `single_stage_banana_example.py:2111-2132`) uses the diagnostic to
  drive `shrink_alm_smoothing_for_gap_count` (L2135-2137) — the
  smoothing parameter is reduced whenever any constraint reports
  mismatch. Reducing smoothing changes the surrogate aggressively
  toward the hard signal, which the algorithm-side wants only when
  the surrogate is materially diverging from the hard. Spurious
  flags at exact-zero hard values cause unnecessary smoothing
  shrinkage.

- **Impact**: Production-mode bug. Adaptive smoothing is intended to
  shrink only when the surrogate is materially disagreeing with the
  hard signal. Today, an iterate where some constraint sits exactly
  at its hard boundary (e.g., a binding distance constraint where
  `coil_coil_distance == coil_coil_distance_threshold` exactly) will
  trigger spurious mismatch and unwanted shrinkage. Over many outer
  iterations this can drive smoothing below `smoothing_min` and
  destabilize the surrogate gradient.

  The bug is also contagious to the integration test at
  `tests/geo/test_single_stage_alm_integration.py:572-616`
  (`test_single_stage_adaptive_smoothing_counts_normalized_hard_surrogate_gaps`)
  which uses a fixture
  `"surrogate_hard_sign_mismatch_by_constraint": [False, True, True]`
  — the fixture pre-builds the diagnostic and bypasses the producer,
  so the bug is invisible to that test.

- **Suggested fix**: Add an activity tolerance band:
  ```python
  def _surrogate_hard_sign_mismatch(
      surrogate_signed_values: np.ndarray,
      hard_signed_values: np.ndarray,
      activity_tolerances: np.ndarray | None = None,
  ) -> list[bool]:
      tol = (
          0.0 if activity_tolerances is None
          else np.asarray(activity_tolerances, dtype=float)
      )
      surrogate_signs = np.where(
          np.abs(surrogate_signed_values) <= tol, 0.0,
          np.sign(surrogate_signed_values)
      )
      hard_signs = np.where(
          np.abs(hard_signed_values) <= tol, 0.0,
          np.sign(hard_signed_values)
      )
      return (surrogate_signs != hard_signs).tolist()
  ```
  This makes hard-zero (or hard-within-tolerance) NOT count as a
  sign mismatch with a surrogate that is within the same tolerance
  band. Pass `evaluation["constraint_activity_tolerances"]` from the
  caller. Add a unit test:
  ```python
  def test_surrogate_hard_sign_mismatch_does_not_flag_exact_zero_hard():
      result = _surrogate_hard_sign_mismatch(
          surrogate_signed_values=np.array([-1e-3]),
          hard_signed_values=np.array([0.0]),
          activity_tolerances=np.array([1e-6]),
      )
      self.assertEqual(result, [False])
  ```

---

### F4: Property test does not pin a specific termination_reason [MEDIUM]

- **File**: `tests/geo/test_alm_utils.py:2464-2468`
- **Code**:
  ```python
  # Termination reason is stable across re-runs.
  self.assertEqual(
      first_result.termination_reason,
      second_result.termination_reason,
  )
  ```

- **Bug**: The test asserts the two re-runs produce equal
  `termination_reason`, but does NOT pin the value to a specific
  string like `"max_outer"`. Any deterministic label change (e.g.,
  refactor renaming `"max_outer"` to `"outer_iteration_cap_reached"`
  or routing fixture through `signal_mismatch_stall`) passes the
  test silently.

- **Why**: The test docstring at L2367-2368 says "must produce the
  same `termination_reason` and history action sequence across re-runs".
  This is the determinism property, which is correct as-stated. But
  the contract being pinned (per the doc at line 51) is "failure-labeling
  chatter under sustained mismatch": the documented bounded label set
  (`signal_mismatch_stall`, `signal_mismatch_penalty_increase` followed by
  `max_outer`, etc.) should be ASSERTED as one of, not just stable.

- **Impact**: A regression that silently moves the run from the
  `signal_mismatch_penalty_increase → max_outer` arm to the
  `signal_mismatch_stall` arm passes the test. The contract document
  (line 51) explicitly enumerates the expected label set; the test
  pins none of those labels. The "deterministic-termination property
  test" is named for and claims to pin the contract, but pins only
  the determinism axis.

- **Suggested fix**: Add an explicit label assertion. Given the test
  fixture's polarity (surrogate active, hard inactive,
  `surrogate_positive_shift_zero=False`, sustained), the expected
  arm is `signal_mismatch_penalty_increase`. The test should assert:
  ```python
  expected_terminator_set = {"max_outer", "max_outer_after_penalty_increase",
                             "penalty_cap_reached"}
  self.assertIn(first_result.termination_reason, expected_terminator_set)
  ```
  Or, more strictly, pin the exact label expected for THIS fixture:
  ```python
  self.assertEqual(first_result.termination_reason, "max_outer")
  self.assertEqual(
      [entry["action"] for entry in first_result.history],
      ["subproblem_continue", "signal_mismatch_penalty_increase",
       "subproblem_continue", "signal_mismatch_penalty_increase",
       "subproblem_continue", "signal_mismatch_penalty_increase"],
  )
  ```
  Adjust counts based on `max_outer_iterations=5` and observed
  arm dispatch.

---

### F5: Property test only covers one polarity of mismatch [MEDIUM]

- **File**: `tests/geo/test_alm_utils.py:2402-2411` (fixture)
- **Code**:
  ```python
  return self._stage2_signal_evaluation(
      ...
      hard_signed_constraint_values=np.array([-1.0e-2]),
      surrogate_signed_constraint_values=np.array([0.2]),
      hard_dual_update_values=np.array([-1.0e-2]),
      ...
  )
  ```

- **Bug**: The fixture sets surrogate active (positive 0.2), hard
  inactive (negative -0.01). With multipliers starting at zero,
  `surrogate_positive_shift = max(0, 0 + penalty * 0.2) > 0`. So
  `surrogate_positive_shift_zero=False`, which means the
  `signal_mismatch_stall` arm at L4367 (which requires
  `surrogate_positive_shift_zero=True`) is **unreachable** through
  this fixture.

- **Why**: The contract has two distinct retry arms under signal
  mismatch:
  - `signal_mismatch_penalty_increase` (L4385) — fires when surrogate
    has positive shift (active in augmenting term).
  - `signal_mismatch_stall` (L4367) — fires when surrogate has zero
    shift but masks still disagree (i.e., hard says active, surrogate
    says inactive).

  The test docstring at line 2486-2491 explicitly says "no chatter
  between continuation arms" and the comment at 2471-2472 references
  "signal_mismatch_stall arms" — but the fixture cannot trigger
  that arm.

- **Impact**: The other half of the contract's failure-label space
  is untested. A regression that breaks the `signal_mismatch_stall`
  arm (e.g., changes the `surrogate_positive_shift_zero` predicate
  computation) would not surface until production. Per the contract
  doc's "residual risk" line 51, both arms are part of the bounded
  label set; the property test only pins one.

- **Suggested fix**: Add a sister property test that flips polarity:
  ```python
  def test_alm_terminates_deterministically_under_sustained_signal_mismatch_inverse_polarity(self):
      # Hard active, surrogate inactive (hard violation reported as 0
      # so hard_feasible_strict trips, forcing the signal_mismatch arm,
      # then surrogate_positive_shift_zero=True triggers stall).
      ...
      hard_signed_constraint_values=np.array([1.0e-3]),
      surrogate_signed_constraint_values=np.array([-1.0e-2]),
      hard_dual_update_values=np.array([1.0e-3]),
      hard_violation_values=np.array([0.0]),  # hard "violation" zero so feasible
      ...
  ```
  Verify the run hits `signal_mismatch_stall` arm and terminates
  with the expected label.

---

### F6: Legacy non-explicit-stage2 path silently disables the safeguard [MEDIUM]

- **File**: `examples/single_stage_optimization/alm_utils.py:1979-2037`,
  L2105 (signal_mismatch_active gating).
- **Code**:
  ```python
  explicit_stage2_signals = any(
      key in evaluation
      for key in stage2_signal_fields
  )
  ...
  if explicit_stage2_signals:
      ...  # use the four explicit fields
  else:
      hard_signed_constraint_values = dual_update_values
      hard_violation_values = feasibility_values
      surrogate_signed_constraint_values = solver_constraint_values
      preferred_dual_update_values = dual_update_values
  ...
  signal_mismatch_active = signal_state.explicit_stage2_signals and masks_disagree
  ```

- **Bug**: When the evaluator does NOT set any of the four explicit
  stage-2 signal fields (`hard_signed_constraint_values`,
  `hard_violation_values`, `surrogate_signed_constraint_values`,
  `hard_dual_update_values`), `explicit_stage2_signals=False`. Then
  `hard_signed_constraint_values` is set to whatever `dual_update_values`
  is — **which itself can be the SURROGATE per metadata**
  (`single_stage_objectives.py:574`'s
  `dual_update_value_kind="surrogate" if not uses_hard_signal`).
  Worse, line 2105 hard-gates `signal_mismatch_active` on
  `explicit_stage2_signals`, so the safeguard is silently disabled.

- **Why**: The contract document forbids "Removing the
  `hard_dual_update_values` field from stage-2 evaluation output"
  (line 68: "raises `KeyError` when this field is missing"). The
  KeyError is conditional — it only fires when ANY of the four
  fields is present (line 1998-2006). An evaluator that omits ALL
  four falls through silently to the legacy mode where there's no
  hard channel at all. This is documented (line 68 mentions
  "strict-error behavior is part of the contract surface and must
  not be loosened") but the strict error is conditional, not
  absolute.

- **Impact**: Two production callers go through the legacy path:
  1. `examples/single_stage_optimization/banana_opt/alm_fixture_benchmarking.py:166-193`
     — `_evaluate_fixture` does NOT set any of the four explicit
     fields, so the benchmarking suite runs entirely without the
     safeguard. Mismatch detection cannot fire; the converged-gate
     guard is a no-op for these runs.
  2. Any legacy resume path or test fixture that uses
     `_complete_alm_evaluation` (`tests/geo/test_alm_utils.py`)
     without the stage-2 augmentation has the same hole. Greppable
     by `grep -rn "explicit_stage2_signals=True"` — production code
     never explicitly sets it; it's inferred from the field presence.

  This means the contract's safeguard is conditional on the caller's
  evaluator shape. An operator who reads the contract doc and
  trusts the safeguard fires for all callers is wrong.

- **Suggested fix**: Make the contract explicit. Either:
  (a) Add a flag `require_stage2_signals: bool = True` to
      `minimize_alm` and raise at run start if the first evaluator
      call doesn't return all four fields; or
  (b) Document in the contract doc, in a new "scope" subsection,
      that the safeguard only applies when the evaluator emits
      all four explicit stage-2 fields, and explicitly list the
      legacy-path callers that opt out.

  Option (a) is the SSOT-clean fix; option (b) accepts the gap
  and warns the operator.

---

### F7: No test asserts the dual update specifically uses the hard channel [MEDIUM]

- **File**: `tests/geo/test_alm_utils.py:946-1025`
- **Code**:
  ```python
  @staticmethod
  def _routing_state_with_preferred(module, preferred_dual_update_values):
      signal_state = module.ALMConstraintSignalState(
          explicit_stage2_signals=False,
          hard_signed_constraint_values=preferred_dual_update_values,
          hard_violation_values=np.zeros_like(preferred_dual_update_values),
          surrogate_signed_constraint_values=preferred_dual_update_values,
          preferred_dual_update_values=preferred_dual_update_values,
      )
      ...
  ```

- **Bug**: The fixture sets
  `hard_signed_constraint_values == surrogate_signed_constraint_values
  == preferred_dual_update_values`. The dual-update test at L969-1001
  uses this fixture and asserts the multipliers update. But because
  hard == surrogate in the fixture, the test would pass even if
  `_handle_alm_dual_update_transition` were re-routed to use
  `surrogate_signed_constraint_values` instead of
  `preferred_dual_update_values`. The contract's central claim — that
  the dual update uses the HARD channel — is **not pinned by any test**.

- **Why**: When the fixture was constructed (test_alm_utils.py:947),
  the goal was to exercise the projection logic with a known dual
  update vector. The author chose to set all three signal channels
  to the same value to keep the fixture small. This is fine for
  testing the projection arithmetic, but it loses the ability to
  detect a routing swap.

- **Impact**: The forbids-future-refactor clause in the contract
  doc (lines 60-64) says: "Routing surrogate signals into the dual
  update... requires a fresh dual-convergence analysis." But the
  test suite would not catch a refactor that does exactly that.
  The clause is unenforced.

- **Suggested fix**: Add a test that distinguishes the channels:
  ```python
  def test_dual_update_uses_hard_channel_not_surrogate(self):
      module = load_alm_utils_module()
      # Hard says +0.05 (active, positive violation).
      # Surrogate says -0.10 (inactive, well inside).
      # If the dual update uses hard, λ grows by μ*0.05.
      # If it incorrectly uses surrogate, λ stays at 0 (max(0, 0 + μ*(-0.10))).
      hard_values = np.array([0.05])
      surrogate_values = np.array([-0.10])
      signal_state = module.ALMConstraintSignalState(
          explicit_stage2_signals=True,
          hard_signed_constraint_values=hard_values,
          hard_violation_values=np.maximum(hard_values, 0.0),
          surrogate_signed_constraint_values=surrogate_values,
          preferred_dual_update_values=hard_values,
      )
      routing_state = module.ALMConstraintRoutingState(
          signal_state=signal_state,
          ...
      )
      result = module._handle_alm_dual_update_transition(
          multipliers=np.array([0.0]),
          routing_state=routing_state,
          penalty_argument=10.0,
          settings=settings,
          update_feasibility_tol=1e-2,
          update_stationarity_tol=1e-2,
      )
      np.testing.assert_allclose(result.multipliers, np.array([0.5]))
      # If surrogate were used: result.multipliers would be [0.0].
  ```
  Symmetric test for the surrogate-input check at the inner-objective
  side: assert that `augmented_inequality_objective` called with
  the surrogate values produces a gradient that matches `(λ + μ*c_surrogate) ∇c`.

---

### F8: `_explicit_raw_signed_constraint_values` returns surrogate, not hard [LOW]

- **File**: `examples/single_stage_optimization/alm_utils.py:678-686`
- **Code**:
  ```python
  def _explicit_raw_signed_constraint_values(evaluation: dict) -> np.ndarray | None:
      raw_constraint_values = _optional_float_array(evaluation, "raw_constraint_values", None)
      if raw_constraint_values is not None:
          return raw_constraint_values
      return _optional_float_array(
          evaluation,
          "raw_surrogate_signed_constraint_values",
          None,
      )
  ```

- **Bug**: Function name suggests it returns the explicit (hard) raw
  signed constraint values. Actual behavior: returns
  `raw_constraint_values` (which is set to
  `sanitized_surrogate_signed_constraint_values` at
  `stage2_objectives.py:1973`) or falls back to
  `raw_surrogate_signed_constraint_values`. The hard-side
  counterpart is `raw_hard_signed_constraint_values` (set at
  `stage2_objectives.py:1977`), which this function does NOT consult.

- **Why**: Historically, "raw_constraint_values" was the un-normalized
  signed value used by the inner objective — which, post-hybrid-split,
  is the surrogate. The function just exposes the raw of the inner
  channel for diagnostic display. The name was not updated when the
  hard/surrogate split was introduced.

- **Impact**: Diagnostic-only (consumed by
  `_constraint_history_diagnostics_source` at L917 and result-builder
  at L2753-2755). Does not affect dual updates or convergence. But
  it's misleading to a reader of the contract document who, scanning
  for "the raw hard channel," lands on this function name and gets
  the wrong answer.

- **Suggested fix**: Rename to
  `_explicit_inner_objective_raw_signed_constraint_values` (long but
  honest), or split into two helpers:
  `_raw_surrogate_signed_constraint_values(evaluation)` and
  `_raw_hard_signed_constraint_values(evaluation)`, both
  documented with their channel.

---

### F9: `_emit_alm_stall_failure_step` does not set `outer_termination` on final outer [LOW]

- **File**: `examples/single_stage_optimization/alm_utils.py:3646-3693`
- **Code**:
  ```python
  def _emit_alm_stall_failure_step(
      *,
      ...
      action: str,
      termination_reason: str,
      ...
  ) -> _ALMContinuationStepResult:
      history_entry["action"] = action
      history_entry["trust_radius"] = run_state.trust_radius
      _emit_alm_history_snapshot(...)
      return _finalize_continuation_step(
          state, _ALMContinuationDecision.RETURN, ...
      )
  ```

- **Bug**: The helper does not accept `is_final_outer` and never
  annotates `outer_termination="max_outer"` on the history entry.
  Two RETURN-arms feed it: `signal_mismatch_stall` (L4367) and
  `constraints_inactive_stall` (L4345). On the final outer iteration,
  these arms produce a history entry without `outer_termination`,
  while every BREAK_OUTER arm (penalty_increase, dual_update,
  subproblem_continue) sets the field via `_annotate_break_outer_history`.

- **Why**: The two stall arms RETURN with their own pre-built
  failure result that carries `termination_reason` directly — so
  `_termination_reason_from_history` is not consulted (the run
  finalizer uses the result's termination_reason field). The missing
  history annotation is observationally inert for `result.termination_reason`.

- **Impact**: Asymmetric history bookkeeping. A downstream consumer
  iterating over `result.history` and using
  `entry.get("outer_termination") == "max_outer"` as a "this entry
  ended the run" marker will miss the two stall-arm exits.
  Frontier-archive reporting and adaptive smoothing logic both
  consume `result.history`; whether either rely on this marker is
  callsite-dependent.

- **Suggested fix**: Thread `is_final_outer` into `_emit_alm_stall_failure_step`
  and set `history_entry["outer_termination"] = "max_outer"` when
  it's True. Keep `termination_reason` flowing through the result
  builder; the history entry annotation is supplementary.

---

## Contract Verification

Mapping each safeguard listed in `docs/alm_hybrid_signal_contract_2026-05-08.md`
to its actual current behavior:

| Doc Safeguard | Doc Citation | Current Location | Verdict | Notes |
|---|---|---|---|---|
| Inner objective fed surrogate | `stage2_objectives.py:1944` | `stage2_objectives.py:1951-1958` | CONFIRMED (line drift +7) | `augmented_inequality_objective(..., normalized_surrogate_signed_constraint_values, ...)` correctly uses surrogate. |
| Stage-2 evaluation stores both signals | `stage2_objectives.py:1955-1972` | `stage2_objectives.py:1959-1988` | CONFIRMED (line drift +4) | `hard_signed_constraint_values`, `hard_violation_values`, `surrogate_signed_constraint_values`, `hard_dual_update_values` all stored. |
| `_extract_stage2_constraint_signal_state` selects hard channel | `alm_utils.py:1915-1953` | `alm_utils.py:1979-2037` | CONFIRMED (line drift +64) | L2014-2016: `preferred_dual_update_values = _as_float_array(evaluation["hard_dual_update_values"])`. |
| `_handle_alm_dual_update_transition` uses preferred (hard) | `alm_utils.py:2988-2995` | `alm_utils.py:3090-3120` (call at 3102) | CONFIRMED (line drift +102) | Routing-state hard channel feeds `_project_nonnegative_multipliers_with_diagnostics`. |
| Mismatch detection | `alm_utils.py:2007-2022` | `alm_utils.py:2079-2137` (mask compare at 2104-2105) | CONFIRMED (line drift +82) | `signal_mismatch_active = explicit_stage2_signals and masks_disagree`. |
| Boundary disagreement detection | `alm_utils.py:2036-2042` | `alm_utils.py:2119-2125` | CONFIRMED (line drift +83) | `direct_boundary_mismatch = ... and np.any(surrogate_positive_shift > 0.0)`. |
| Mismatch flag in every history entry | `alm_utils.py:2806`, `alm_utils.py:3873` | (multiple) | CONFIRMED | `signal_mismatch_active` is annotated at L2447, L2917, L3849, L3990, L4190 in current tree. |
| Converged-gate guard | `alm_utils.py:4133-4138` | `alm_utils.py:4264-4274` | CONFIRMED (line drift +131) | `not signal_mismatch_active` AND `not run_state.last_cap_binding_active` (M4 addition). |
| Constraints-inactive guard | `alm_utils.py:4125-4131` | `alm_utils.py:4316-4321` and `4256-4262` (constraints_inactive_candidate gating) | CONFIRMED (line drift +191) | `constraints_inactive_candidate` requires `not signal_mismatch_active` (L4261). |
| Skipped-inner shortcut guard | `alm_utils.py:3851-3855` | `alm_utils.py:3968-3973` | CONFIRMED (line drift +118) | Same predicate at L3972: `not current_signal_mismatch_active`. |
| Strict KeyError on missing hard_dual_update_values | `alm_utils.py:1915-1923` | `alm_utils.py:1998-2006` | INCOMPLETE | KeyError fires only if ANY of the four fields is set; fully-empty evaluator silently passes through legacy mode. See F6. |
| Forbid hard signal in inner objective (refactor block) | (no test) | n/a | INCOMPLETE | No test exercises `augmented_inequality_objective` with hard input vs surrogate input; the function happily consumes whatever is passed. See F7's symmetric test. |
| Forbid surrogate signal in dual update (refactor block) | (no test) | n/a | INCOMPLETE | No test distinguishes the channels in the dual-update path; the test fixture has hard == surrogate. See F7. |
| Property test deterministic termination | `tests/geo/test_alm_utils.py::MinimizeAlmTests::test_alm_terminates_deterministically_under_sustained_signal_mismatch` | tests/geo/test_alm_utils.py:2359-2491 | INCOMPLETE | Test asserts re-run determinism but does NOT pin a specific terminator label, and only covers one mismatch polarity. See F4 and F5. |

## Confirmed-Correct Items

The following were verified line-by-line and produce no finding:

- **Inner objective signal routing** —
  `stage2_objectives.py:1951-1958` calls `augmented_inequality_objective`
  with `normalized_surrogate_signed_constraint_values`. The augmenting
  term (`positive_shift = max(0, λ + μ * c_surrogate)`) and the
  augmented gradient (`∇f + positive_shift * ∇c_surrogate`) both
  reflect the surrogate channel. Line 1954.
- **Dual-update value selection** —
  `_extract_stage2_constraint_signal_state` at L2014-2016 correctly
  reads from `evaluation["hard_dual_update_values"]` when explicit
  stage-2 signals are present. The four explicit fields are validated
  for shape consistency at L2022-2030.
- **Dual-update math** —
  `_handle_alm_dual_update_transition` at L3090-3120 correctly
  passes `routing_state.signal_state.preferred_dual_update_values` (the
  hard channel) to `_project_nonnegative_multipliers_with_diagnostics`
  at L3100-3105. Confirmed by reading both functions and the unit
  test at L969-1001 (which tests the projection arithmetic, not the
  channel selection — see F7).
- **Converged-gate guard text** —
  L4264-4274 includes `not signal_mismatch_active` and
  `not run_state.last_cap_binding_active` (M4 fix). The skipped-inner
  shortcut at L3968-3973 has the parallel guard. The
  constraints-inactive arm at L4316-4321 has the M4 cap-binding gate.
- **Mismatch flag plumbing** —
  `signal_mismatch_active` is annotated on every history entry
  (L2447, L2917, L3849, L3990, L4190) and surfaces in the result
  carrier (L2917). Verified by reading `_alm_history_entry_payload`,
  `_build_skipped_inner_history_entry`, and
  `_build_alm_failure_result_with_optional_restore`.
- **Mismatch is independent of multipliers (mask path)** —
  `hard_activity_mask` and `surrogate_activity_mask` (L2090, L2098)
  depend only on constraint values, feasibility values, and
  activity tolerances. No multiplier dependence; mismatch on the
  mask path is stable across dual updates.
- **Mismatch can switch on after dual update (boundary path)** —
  `direct_boundary_mismatch` (L2119-2125) depends on
  `surrogate_positive_shift > 0`, which depends on multipliers. After
  a dual update, λ grows; if `λ + μ * c_surrogate > 0` for any
  surrogate-active-but-hard-inactive constraint, mismatch flips on.
  This is **expected behavior** per the contract — a feature, not a
  bug — but worth noting.
- **Empty constraint vector** —
  `_constraint_routing_state` returns
  `signal_mismatch_active=False` for empty arrays (`np.array_equal(empty, empty)=True`).
  `_kkt_stationarity_norm` returns 0.0 or None for empty inputs
  (L2152-2153, L2166-2167). No NaN risk.
- **Mismatch detection gating on explicit_stage2_signals** —
  L2105 hard-gates the safeguard on `explicit_stage2_signals`. This
  is intentional; legacy callers without explicit signals collapse
  hard==surrogate and have no meaningful mismatch (since both
  channels are the same value). Subject to F6's documentation
  finding.
- **Augmented Lagrangian formula** — (cross-verified against
  `.alm_audit/math_review.md:9-39, 248-310`) the inequality augmented
  Lagrangian, gradient, dual projection, penalty schedule, and
  outer convergence test are mathematically correct. Hybrid forfeits
  rate-of-convergence theorems but the per-step math is right.

## Verdict

The hybrid surrogate-vs-hard contract is structurally **honored** at the
implementation level: the inner objective receives the surrogate, the dual
update consumes the hard channel via `preferred_dual_update_values`, the
mismatch detection plumbs through history, and the converged gate
correctly blocks `result.success` under sustained mismatch. The
forfeited Bertsekas/LANCELOT convergence guarantees (per `math_review.md`
F1) are correctly classified by the contract document.

That said, the contract document and supporting tests have **load-bearing
gaps**:

- **F1 (HIGH)** — Every line citation in the contract doc is stale by
  64-130 lines. The doc's "what this contract forbids" clauses cite
  line numbers that do not resolve in the current tree.
- **F2 (HIGH)** — Dual-update mutation is not gated by
  `signal_mismatch_active`; under loose-feasibility mismatch, λ
  accumulates updates while the safeguard only blocks the
  success label. The contract doc does not document this regime.
- **F3 (HIGH)** — `_surrogate_hard_sign_mismatch` flags spurious
  mismatches at exact-zero hard values, driving unwanted adaptive-smoothing
  shrinkage. Production-mode bug.
- **F4-F7 (MEDIUM)** — Test coverage is incomplete: property test does
  not pin a specific terminator label, covers only one mismatch
  polarity, has no test for hard-vs-surrogate channel routing in
  the dual update, and the legacy-path callers silently bypass the
  safeguard.
- **F8-F9 (LOW)** — Naming and history-bookkeeping hygiene.

Recommend fixing F1 immediately (re-resolve cites, add CI greps for
cited symbols), F3 immediately (add tolerance to sign mismatch),
F4+F5 in the next test pass (pin labels + add inverse-polarity
property test), F2 in the next contract revision (decide whether
to gate the dual update or document the regime), F6 as a contract
scope clarification, F7 as a regression-protection test addition,
F8+F9 as cleanup.

The contract is not violated today, but it is partially unprotected.
A future refactor that swaps the inner-objective channel or routes
the surrogate into the dual update could pass the test suite without
detection. The "fresh derivation required" clause is unenforced.

End of audit.
