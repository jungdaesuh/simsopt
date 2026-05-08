# ALM Numerics / JAX / Computation-Correctness Audit

**Scope.** `examples/single_stage_optimization/alm_utils.py` (4637 LOC), helpers
`examples/single_stage_optimization/banana_opt/alm_benchmarking.py` and
`examples/single_stage_optimization/banana_opt/alm_fixture_benchmarking.py`,
plus the `evaluate_problem` callable producer
`examples/single_stage_optimization/banana_opt/stage2_objectives.py`.

**Branch under audit.** `surrogate-confinement-v2` (HEAD `382d7a082`).

---

## 0. Executive summary

ALM on this branch is a **pure NumPy / SciPy** driver. There is no `import jax`,
no JIT-compiled inner solve, no device traffic, and no tracer surface in
`alm_utils.py`. The "native JAX single-stage ALM runtime" referenced by the
task brief (`f50c3aa0c`, `cca0cc104`, `f3f2f537c`) lives only on the parallel
worktree branch `gpu-purity-stage2-20260405` and was never merged into
`surrogate-confinement-v2`. On the audited branch, ALM evaluates a callback
that returns float64 NumPy arrays, runs `scipy.optimize.minimize(method="L-BFGS-B")`
synchronously, and aggregates results in NumPy — i.e. all eight items in the
task brief that key on JAX (dtype-on-tracer, ConcretizationTypeError, host-device
sync, device_put, pure_callback, etc.) **do not apply** to the current ALM code.

The remaining concerns — dtype, NaN/Inf handling, copy semantics, in-place
mutation, ill-conditioning, norm choice, accumulator overflow — are auditable.
Findings are numbered MED/LOW; no HIGH-severity numerical correctness defects
were found. Two MED items relate to defensive guards around external scipy
calls; two LOW items are aliasing / API consistency. Eight items are verified
correct.

---

## 1. Findings

### 1.1 MED — `nnls` failure can crash diagnostics path
- **File:line.** `examples/single_stage_optimization/alm_utils.py:2071` inside
  `_kkt_stationarity_norm`.
- **Symptom.** `scipy.optimize.nnls(active_matrix, -total_grad_array)` is invoked
  with no `try/except` and no `maxiter` guard. scipy's `nnls` raises
  `RuntimeError("too many iterations")` when the active-constraint Jacobian is
  ill-conditioned beyond the default 3·n iteration budget. This function is
  reached from `_stationarity_metrics`, which is in turn reached from every
  history entry build, every summary, and the inner-solve callback
  `_ALMInnerAttemptEvaluator.callback` (L208–222).
- **Root cause.** With `multiplier_max=1.0e6` and `penalty_max=1.0e8`, active
  constraint gradients can become near-parallel (the audit-cited "exact Jacobian
  ill-conditioning finding" applies analogously here whenever two constraints
  share a near-degenerate gradient direction). nnls then exhausts its iteration
  budget and raises.
- **Fix.** Wrap the nnls call with a `try / except RuntimeError` returning
  `None` for the kkt_stationarity_norm (callers already accept `None`). Pass an
  explicit `maxiter=10*active_matrix.shape[1]` to bound work.

### 1.2 MED — `_attach_alm_constraint_metadata` returns the original dict when no blocks
- **File:line.** `examples/single_stage_optimization/alm_utils.py:2209-2210`.
- **Symptom.** When `constraint_blocks_tuple is None`, the function returns
  `evaluation` directly (no copy). When non-None, it returns a fresh shallow
  `dict(evaluation)`. Callers — at L2401, L3157, L3770 — then assign into the
  result. None mutate today, but the dual-shape contract is fragile: any future
  change that adds a `result[...] = ...` write would silently corrupt the upstream
  evaluator's dict in the no-blocks lane only.
- **Root cause.** Branch on `constraint_blocks_tuple is None` returns alias
  rather than a copy.
- **Fix.** Always shallow-copy: `return dict(evaluation)` in the early-return
  arm. Negligible cost; eliminates the alias divergence.

### 1.3 LOW — `_sanitize_nonfinite_inner_evaluation` shares non-array fields with fallback
- **File:line.** `examples/single_stage_optimization/alm_utils.py:1228-1252`.
- **Symptom.** The shallow-dict-copy + selective-array-copy strategy
  (commit 70cb9770c "tighten ALM copy discipline") owns the 23 fields in
  `_OWNED_EVALUATION_ARRAY_FIELDS` plus `constraint_grads`, `constraint_names`,
  `constraint_blocks`, `constraint_scale_sources`. Any *other* field that
  happens to be a list / dict in the upstream evaluation (e.g.,
  `block_max_normalized_violation` dict, `nonfinite_input_fields` list) is
  shared with the fallback. No current downstream caller mutates these in-place,
  so this is latent.
- **Root cause.** Manual whitelist of "owned" fields needs to stay in sync with
  upstream evaluator output.
- **Fix.** Either: (a) add `_OWNED_EVALUATION_ARRAY_FIELDS` entries explicitly
  for any new mutable field; or (b) at the cost of one extra `dict.copy()` per
  sanitize, copy unknown list/dict values too.

### 1.4 LOW — `_build_augmented_evaluation` aliases caller's input arrays
- **File:line.** `examples/single_stage_optimization/alm_utils.py:499-525`.
- **Symptom.** `result["base_grad"]`, `result["grad"]`, `result["constraint_values"]`,
  `result["positive_shift_values"]`, `result["augmented_term_by_constraint"]`
  all use `np.asarray(..., dtype=float)` which is a *no-copy* call when the
  input is already a float64 ndarray. The dict therefore aliases the caller's
  arrays. The caller (`augmented_inequality_objective`) does pass freshly
  allocated arrays today, but this is an undocumented contract.
- **Root cause.** `np.asarray` is not a copy.
- **Fix.** Either document the contract that callers pass owned arrays, or
  switch to `.copy()` for the four most-commonly-read fields. Cost: ~6 array
  copies per evaluation, negligible relative to the L-BFGS-B inner cost.

### 1.5 LOW — Multiplier projection has no NaN guard
- **File:line.** `examples/single_stage_optimization/alm_utils.py:1543-1574`,
  `_updated_nonnegative_multipliers` and
  `_project_nonnegative_multipliers_with_diagnostics`.
- **Symptom.** A NaN in `dual_update_values` would silently produce NaN
  multipliers (`np.maximum(0, NaN) == NaN` in NumPy; `np.minimum(NaN, cap) == NaN`).
  `cap_binding_mask = updated > cap` evaluates to False for NaN, so cap-binding
  diagnostics also miss it. The next iteration's evaluation would then receive
  NaN multipliers and produce NaN gradients, which would be sanitized at the
  inner-loop sanitize boundary — but `cap_binding_indices=[]` would lie about
  whether a cap fired.
- **Root cause.** The pre-condition that `dual_update_values` is finite is
  enforced upstream by `_require_finite_evaluation` at L3775 and the inner
  sanitize at L3112 / L175 / L194, so today a NaN can't reach this code path.
  The defense-in-depth gap is theoretical.
- **Fix.** Add `if np.any(~np.isfinite(updated)):` and surface as
  `multiplier_cap_binding=True` with a synthetic flag. Or rely entirely on the
  upstream finite-check (current state). Document either way.

---

## 2. Verified correct

The following items were specifically checked against the task brief and
found to be implemented correctly on this branch:

1. **Dtype discipline.** Every numerical pathway routes through
   `np.asarray(values, dtype=float)` (93 occurrences; line spot-checked at
   L382-405, L443-470, L1548-1552, L2052, L2070-2073). `dtype=float` resolves
   to `np.float64` on every supported platform. The single `dtype=int` use
   (L657) is for a block-index array. There are no `astype(np.float32)` calls
   and no implicit-mix sites where a Python float scalar is combined with a
   non-float-cast array. Stage 2's evaluator
   (`stage2_objectives.evaluate_stage2_alm_problem` L1644-1820) likewise stays
   in float64 throughout.

2. **NaN/Inf detection.** `_nonfinite_evaluation_fields` (L1169-1214) inspects
   total, grad, optional scalars (`stationarity_norm`, `metric_stationarity_norm`,
   `max_violation`, `max_feasibility_violation`, `base_value`, `base_total`),
   optional arrays (`constraint_values`, `feasibility_values`,
   `dual_update_values`, `metric_grad`, `base_grad`, `constraint_activity_tolerances`),
   and each entry of the `constraint_grads` list. Two enforcement strategies:
   `_require_finite_evaluation` (raise) at outer-iterate evaluation (L3775) and
   penalty-update evaluation (L2406); `_sanitize_nonfinite_inner_evaluation`
   (fallback to current_eval with elevated total) at inner-attempt steps
   (L175, L194, L3112).

3. **Sanitize fallback total elevation.** `_elevated_rejection_total` (L1224)
   inflates the fallback total by `|current_total| + 1.0 + ATOL`, making the
   sanitized iterate strictly worse than current and ensuring it is rejected
   by `_candidate_is_acceptable` (L1369-1403). NaN at the *fallback* would
   propagate, but the outer-evaluation guard at L3775 catches it first.

4. **No JIT tracer leaks.** No `import jax` or `jnp` in `alm_utils.py`
   (verified by `grep -ncE 'jax|jnp'` returning 0). Therefore no
   `ConcretizationTypeError` surface. The `_ALMInnerAttemptEvaluator.fun` and
   `.callback` operate purely on NumPy arrays.

5. **No host-device sync.** Same — no JAX, so no `.item()` / `np.asarray(jax_array)`
   sync points at all.

6. **No in-place mutation that breaks JAX functional contract.** All `+=` /
   `[i] =` writes target NumPy arrays held in `ALMRunState` (mutable dataclass)
   or in fresh local arrays (`total_grad = base_grad_array.copy()` then
   `total_grad +=` at L394-400). No JAX traced array is ever the target of
   in-place assignment.

7. **Augmented-Lagrangian formulation avoids catastrophic cancellation.**
   `_augmented_terms` (L528-538) computes
   `0.5 * (s - λ) * (s + λ) / μ` where `s = max(0, λ + μ·c)`. When constraint
   active, `s - λ = μ·c` exactly, so `(s-λ)·(s+λ)/μ = c·(s+λ)` — no subtraction
   of two near-equal large quantities. When inactive, `s = 0`, term is
   `-λ²/(2μ)`, which is a single product divided by μ; no cancellation.

8. **Penalty-and-multiplier float-range hygiene.** With caps
   `multiplier_max=1.0e6`, `penalty_max=1.0e8`, worst-case
   `λ + μ·c = 1e6 + 1e8·c`. For `|c| ≤ 1` (post-normalization), this stays
   below 1e9 — twelve orders of magnitude below float64 overflow. Squaring
   for the augmented term yields ~1e16 / 1e8 = 1e8, still well within range.
   `_penalty_values` (L1578-1588) explicitly rejects non-finite or
   non-positive penalty (raises `ValueError`). `_next_penalty` (L1609-1625)
   detects `np.isfinite(requested_penalty)` and clips to `penalty_max`.

9. **Cache state invalidation is correct.** `_BOXED_INNER_PROFILES` is
   `MappingProxyType` (L265, hardened by commit `80e518337`). `ALMSettings`,
   `ALMInnerSolveProfile`, `_ALMNormalizedRunInputs`,
   `_ALMContinuationStepResult`, `_ALMOuterIterationResult` are all
   `@dataclass(frozen=True)`. `_BOXED_INNER_PROFILES` reads via dict-key lookup
   with a tuple constructed per call — no shared mutable state.
   `_build_inner_options` (L1628-1663) does `options = dict(inner_options)`
   on every call, never mutating the input.

10. **Trimmed-history defer is correct.** `_attach_alm_history_diagnostics`
    (L2347-2367) attaches a *source dict* of arrays under the
    `_HISTORY_DIAGNOSTICS_SOURCE_KEY` private key. The conversion to scalar
    lists happens in `_materialize_history_entry_diagnostics` (L983-989),
    which is called only at result-construction time (L2629-2630) and at
    snapshot time (L1255-1256). After truncation by `_append_alm_history_entry`
    (L2332-2344), the *retained* entries still carry their source dict, so
    diagnostics can be reproduced from the trimmed history. Trimming happens
    at append-time (i.e. after the entry is committed).

11. **Norm-order conventions are internally consistent.** All ALM stationarity
    gates use the L2 augmented gradient norm (L497, L1019, L2094-2099,
    `settings.stationarity_tol` compared against this). The single L∞ norm
    site (`_lbfgsb_projected_gradient_max_norm`, L795) is a separate
    diagnostic — labelled `inner_lbfgsb_projected_gradient_norm` — that
    matches scipy L-BFGS-B's own convergence criterion (sup-norm projected
    gradient). `||c||_∞` (max-violation) uses `_max_value` (L578-579), which
    is `np.max(values) if values.size > 0 else 0.0`; correct since
    `feasibility_values >= 0` post-`np.maximum(c, 0)` (L402).

12. **Inner solver receives `μ` and `λ` separately.** The `evaluate_problem`
    callback signature is `(inner_x, multipliers, penalty_argument)`. The
    augmented gradient is computed inside the user-supplied evaluator
    (e.g. `augmented_inequality_objective` at L382-401). ALM does not
    pre-multiply or fold μ and λ into a combined parameter — they are passed
    through as separate arguments to every `evaluate_problem` call (verified
    at L176-180 inside `_ALMInnerAttemptEvaluator.fun`, L195-199 inside
    `.callback`, and L3113-3117 in `_run_alm_inner_attempts`).

13. **No matrix factorisation in ALM driver.** The only solve in ALM is
    `nnls` for the active-set KKT diagnostic (see Finding 1.1). There is no
    direct `np.linalg.inv`, no `np.linalg.solve` without pivoting, no
    triangular solve. Stability of the inner L-BFGS-B is delegated to scipy's
    Fortran code, which is well-tested.

14. **Penalty growth termination is overflow-safe.** `_next_penalty`
    (L1609-1625) checks `np.isfinite(requested_penalty)` before comparing
    with `penalty_max`. If the requested penalty overflows to `inf`, the
    function returns `(penalty, True, requested_penalty)` (cap_hit=True),
    not a NaN. The penalty-cap termination handler converts this into a
    successful failure-result with `restored_best_feasible` if available.

15. **GPU correctness — N/A.** No GPU code in ALM driver; no `device_put`
    discipline needed. `evaluate_problem` may call into JAX-backed objectives
    (via simsopt geo modules), and those modules manage their own
    device-placement under the `transfer_guard` policy. ALM only sees host
    NumPy arrays at the API boundary.

---

## 3. JAX-on-other-branch reference

The native-JAX inner-solve plumbing on `gpu-purity-stage2-20260405` (commits
`f50c3aa0c`, `cca0cc104`, `f3f2f537c`) introduces these patterns *not present*
on `surrogate-confinement-v2`:

- `_build_target_inner_value_and_grad` wraps `evaluate_value_and_grad` with
  `jax.pure_callback` so the L-BFGS-on-device runtime can call the host
  evaluator without committing to traced execution.
- `jax.device_put(np.asarray(center, np.float64))` and `widths_jax` are
  device-placed once per outer iteration; the tanh trust-region
  parametrisation is fully on-device.
- The result_spec uses `jax.ShapeDtypeStruct((), np.float64)` and
  `jax.ShapeDtypeStruct(shape, np.float64)` — explicit float64.
- `_resolve_target_inner_optimizer` rejects anything but `lbfgs-ondevice`
  with `use_least_squares_objective=False`.

If/when this code is merged into `surrogate-confinement-v2`, the audit
should be re-run against the merged tip with these specific items checked:
(a) the `pure_callback` result_spec dtype must remain float64 to match host;
(b) `widths_jax * (1.0 - jnp.square(jnp.tanh(opt_x)))` is a chain-rule scale
factor — verify it isn't double-applied by the on-device optimizer;
(c) `evaluate_value_and_grad` running outside the JAX trace under
`pure_callback` will not cause concretization errors but will block
JIT optimization across the host call — confirm whether `target_minimize` is
itself jitted.

---

## 4. Cross-cutting observations

- **Pure-NumPy ALM is the right design choice for the current scope.** The
  inner solve is L-BFGS-B (scipy), which has no JAX equivalent in tree, and
  the ALM driver itself runs ~10-50 outer iterations × ~150 inner iterations
  per outer = ~5000 evaluations. Each evaluation is dominated by the
  user-supplied `evaluate_problem` (Stage 2 hardware/iota objectives). ALM's
  per-iteration overhead is dominated by `_constraint_history_diagnostics_source`
  (~30 array allocations per history append) — but defer-to-materialize
  (`451d7ab09`) and shallow-copy sanitize (`70cb9770c`) keep this small.

- **The frozen `ALMSettings` + `MappingProxyType` profile registry pattern
  (commit `80e518337`) eliminates an entire class of "stale-cache" bugs.**

- **`_OWNED_EVALUATION_ARRAY_FIELDS` is the load-bearing whitelist for copy
  discipline.** Tests `test_alm_utils.py` (4447 LOC) cover normalised
  constraints, dual updates, penalty caps, and cap-binding semantics. Future
  Stage-2 evaluator changes that introduce new mutable evaluation fields must
  be reflected in this tuple — that is the one piece of context the audit
  recommends documenting in `BANANA_OPTIMIZATION_TODOS.md` (the constants
  block at L271 is itself well-named, but a one-liner comment that reads
  "extend when adding mutable evaluation fields" would prevent regressions).

- **The exact-Newton-Jacobian ill-conditioning finding from
  `project_known_issues.md` does not affect ALM.** ALM does not compute or
  factor a Boozer-residual Jacobian; the only matrix decomposition in scope
  is scipy `nnls` over the active-constraint Jacobian (≤ 6 columns,
  see Finding 1.1).

---

## 5. Out-of-scope cross-references

The audit task explicitly excluded:
- *math correctness* (math agent) — augmented-Lagrangian formula
  (`0.5·(s²-λ²)/μ` decomposition at L533-538), penalty-tolerance schedule
  (L1605), ALM convergence criteria (L4091-4094) are mathematically
  standard but not re-derived here.
- *physics meaning* (physics agent) — constraint scaling, hard-vs-surrogate
  routing, banana-current upper bound semantics live in
  `stage2_objectives.py` and were not audited.
- *algorithmic control flow* (algorithm agent) — the
  `_ALMContinuationDecision` / `_ALMOuterDecision` state machine was not
  audited beyond verifying that it passes through arrays and dicts without
  introducing alias bugs.
- *test sufficiency* (test agent) — `test_alm_utils.py` and
  `test_alm_benchmarking.py` exist; coverage of the specific findings above
  was not assessed.
