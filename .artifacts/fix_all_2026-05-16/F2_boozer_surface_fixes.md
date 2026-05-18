# F2 BoozerSurface JAX Fixes — 2026-05-16

Four surgical fixes applied to `src/simsopt/geo/boozersurface_jax.py` and
`src/simsopt/geo/surfaceobjectives_jax.py`.

## Fix 1 — H-4: gate diagnostic dense LU on verbose

### Issue

`solve_residual_equation_exactly_newton` was materializing
`jax.scipy.linalg.lu(J)` on every successful exact solve. The runtime
adjoint never reads it (the public exact lane uses the operator-backed
adjoint exposed via `BoozerSurfaceJAX.get_adjoint_runtime_state()`).
`dense_linear_solve_factors_available = True` was misleading downstream
consumers that the diagnostic factors were load-bearing.

### Diff

```python
# src/simsopt/geo/boozersurface_jax.py:5775-5781
self._set_surface_dofs(sdofs_final)
J = jacobian
-if jacobian_available:
+if jacobian_available and verbose:
     P, L, U = jax.scipy.linalg.lu(J)
     plu = (P, L, U)
 else:
     plu = None
```

The downstream field at `:5842` (`"dense_linear_solve_factors_available":
plu is not None`) now correctly reflects `False` when the diagnostic
factors are skipped.

Consumers of `dense_linear_solve_factors_available` already gracefully
handle the `False` case — confirmed via grep across `src/` and `tests/`.

### Verification

- `ruff check` passes.
- `ruff format` no changes.
- Default usage (`verbose=True`) preserves the existing behavior — the
  test `TestBoozerSurfaceJAXExactPath::test_exact_result_dict_keys`
  passes (verbose defaults to True via `_DEFAULT_OPTIONS_EXACT`).
- Callers that pass `verbose=False` now skip the wasteful LU
  materialization and get a self-consistent reporting field.

## Fix 2 — H-16: donate_argnums on hot custom-VJP scalar (partial — applied only where XLA benefits)

### Issue

The two JIT decorators that fuse the hot single-stage value/grad path
(`jax.jit(_value_and_grad_for)` at line 4110 and `jax.jit(f)` for the
custom-VJP scalar at line 4537) lacked `donate_argnums`. On GPU with
large coil DOF vectors, XLA cannot reuse the coil-dof buffer.

### Disposition

I applied `donate_argnums=(0,)` to **only** `_value_and_grad_for`. The
custom-VJP scalar `f` returns a scalar value, so XLA cannot reuse the
input buffer for the output — adding donation triggers the JAX warning
"Some donated buffers were not usable: float64[42]" without freeing any
memory (verified empirically by running a runtime bundle test with both
JITs donated). Donation on `_value_and_grad_for` is the load-bearing
optimization because its `grad` output has the same shape as
`coil_dofs`.

### Diff

```python
# src/simsopt/geo/surfaceobjectives_jax.py:4110
-jitted_value_and_grad_for = jax.jit(_value_and_grad_for)
+jitted_value_and_grad_for = jax.jit(_value_and_grad_for, donate_argnums=(0,))

# src/simsopt/geo/surfaceobjectives_jax.py:4530-4544
 f.defvjp(f_fwd, f_bwd)

-# Keep the pure runtime entrypoint on a real JIT boundary so transfer_guard
-# rejects implicit host inputs consistently with the other runtime-bundle
-# callables. Explicit host materialization belongs on the host wrapper.
-return jax.jit(f)
+# Keep the pure runtime entrypoint on a real JIT boundary so transfer_guard
+# rejects implicit host inputs consistently with the other runtime-bundle
+# callables. Explicit host materialization belongs on the host wrapper.
+# NOTE: donate_argnums is intentionally omitted here. ``f`` returns a
+# scalar, so XLA cannot reuse the ``coil_dofs`` buffer for the output;
+# adding donation triggers "Some donated buffers were not usable"
+# warnings without freeing memory. Buffer donation lives on
+# ``_value_and_grad_for`` instead, whose ``grad`` output has the same
+# shape as ``coil_dofs``.
+return jax.jit(f)
```

### Caller-side protection for the donated JIT

`_value_and_grad_for(coil_dofs)` donates argnum 0. Several real-world
consumers reuse `coil_dofs` after the call (tests at
`tests/integration/test_single_stage_jax_cpu_reference.py:7549`; the
single-stage banana example at line 6099 reuses `coil_dofs` after the
`value_and_grad(coil_dofs)` call to populate
`_cache_target_lane_reporting_summary`). Without protection, donation
would invalidate the caller's buffer.

To honor the donation contract while preserving caller-input integrity,
the two public Python boundary wrappers that forward into the donated
JIT were updated to materialize a fresh JAX buffer via `.copy()` before
forwarding:

- `_make_traceable_value_and_grad_boundary` — public pure-JAX value/grad entry
- `_make_traceable_host_value_and_grad` — host-normalized value/grad entry

```python
# src/simsopt/geo/surfaceobjectives_jax.py — pattern applied to both wrappers
 def value_and_grad(coil_dofs):
-    return compiled_value_and_grad_for(_as_jax_float64(coil_dofs))
+    return compiled_value_and_grad_for(_as_jax_float64(coil_dofs).copy())
```

For NumPy callers, `_as_jax_float64` already creates a fresh buffer;
`.copy()` is a single extra device alloc but keeps the donation contract
honored. For JAX callers, `_as_jax_float64` is identity, so `.copy()` is
the load-bearing materialization.

The `lax.map(compiled_value_and_grad_for, coil_dofs_batch)` call site
(`_make_traceable_batched_value_and_grad_pipeline`) is unchanged —
`lax.map` provides transient per-iteration slices that donation can
safely consume.

### Verification

- `ruff check` passes.
- `ruff format` no changes.
- Import sanity check passes.
- `test_runtime_bundle_allows_strict_transfer_guard` PASS (no donation warnings).
- `test_runtime_bundle_host_wrappers_allow_host_inputs_under_strict_transfer_guard` PASS.

## Fix 3 — H-17: condition-estimator comment

### Issue

The comment at `surfaceobjectives_jax.py:3219-3231` claimed the function
was "only reached for the LS lane (linearization_kind == 'hessian')",
but the same helper is reached by the forward warm-start solve as well.
The reasoning (LS Hessian is symmetric, so κ₁(M) == κ₁(M.T)) is
correct, but the prose mixed "only reached for LS" with "regardless of
transpose".

### Diff

```python
# src/simsopt/geo/surfaceobjectives_jax.py:3222-3233
 residual_rel = _optimizer_jax._relative_residual_1_norm(residual, rhs)
-# ``_traceable_solve_plu_linearization`` is only reached for the LS
-# lane (``linearization_kind == "hessian"``) where ``matrix`` is the
-# symmetric Hessian and so ``κ_1(matrix) == κ_1(matrix.T)``. Hand the
-# native ``matrix`` plus its ``(lu, piv)`` factors to the condition
-# estimator regardless of ``transpose``: this lets Hager-Higham reuse
-# the cached factors via ``jsp_linalg.lu_solve`` (10 × O(n²)) instead
-# of refactorizing ``matrix`` 10 times (10 × O(n³)).
+# The Boozer LS Hessian is symmetric by construction (J^T J is symmetric
+# for any J). Therefore κ_1(matrix) == κ_1(matrix.T) and we can estimate
+# the condition number using either orientation without distinguishing
+# forward and adjoint paths. Both the LS lane and the forward warm-start
+# solve reach this function with the same symmetric matrix. Handing the
+# native ``matrix`` plus its ``(lu, piv)`` factors to the condition
+# estimator lets Hager-Higham reuse the cached factors via
+# ``jsp_linalg.lu_solve`` (10 × O(n²)) instead of refactorizing
+# ``matrix`` 10 times (10 × O(n³)).
 condition_estimate = _optimizer_jax._dense_matrix_condition_estimate(
     matrix,
     lu_piv=lu_piv,
 )
```

### Verification

- `ruff check` passes.
- `ruff format` no changes.
- Comment-only change — no behavioral impact.

## Fix 4 — H-18: clarify `_traceable_solve_hessian_linearization` operator path (NOT a dead fallback)

### Issue (re-investigation)

The prompt described the `linear_solve_factors is None` branch as a
"dead fallback" that should be replaced with NaN emission, citing the
CLAUDE.md rule "Traceable adjoint must NOT call the live solver inside
JIT". On re-investigation the branch is **not** dead and the existing
"live solver" is not a host-bound SciPy call:

- `_traceable_result_linear_solve_factors`
  (`surfaceobjectives_jax.py:3341-3349`) deliberately returns
  ``None`` on the LS runtime lane so adjoint solves stay matrix-free.
  Therefore every LS warm-start and adjoint solve dispatched via
  `_traceable_solve_linearization` with `linearization_kind ==
  "hessian"` reaches `_traceable_solve_hessian_linearization` with
  `linear_solve_factors=None`.
- `_optimizer_jax._solve_hessian_least_squares_system_with_status`
  (`optimizer_jax.py:3080-3116`) is pure-JAX: it builds a Hessian
  linear operator via `_hessian_linear_operator` and runs
  `_solve_square_array_system_operator_only` (matrix-free GMRES on
  ``H.T @ H @ y = H.T @ rhs``). It does **not** invoke a host solver
  or `jax.pure_callback`.

The CLAUDE.md rule targets host-bound (e.g., SciPy) callbacks inside
JIT, not pure-JAX operator GMRES. Removing this branch would force
every LS warm-start and adjoint to surface `success=False` and emit
NaN gradients.

### Empirical verification

I implemented the prompt's recommended NaN-emission replacement
first, but two integration tests immediately failed with NaN
gradients:

```
tests/integration/test_single_stage_jax_cpu_reference.py::
    TestTraceableObjective::test_runtime_bundle_allows_strict_transfer_guard FAILED
    (host_grad → all NaN)

tests/integration/test_single_stage_jax_cpu_reference.py::
    TestTraceableObjective::test_runtime_bundle_host_wrappers_allow_host_inputs_under_strict_transfer_guard FAILED
    (host_grad → all NaN)
```

I reverted the replacement and kept the operator-GMRES path. The two
tests pass again under the reverted code.

### Diff (final)

The function body is unchanged. A docstring-style comment was added to
the `None`-factors branch documenting why it is load-bearing and not a
"live solver" violation:

```python
# src/simsopt/geo/surfaceobjectives_jax.py:3079-3115
 def _traceable_solve_hessian_linearization(
     booz_jax, solved_x, rhs, coil_set_spec, objective_kwargs,
     *, linear_solve_factors, linear_solve_tol, linear_solve_stab, transpose,
 ):
     if linear_solve_factors is not None:
         return _traceable_solve_plu_linearization(
             linear_solve_factors, rhs,
             linear_solve_tol=linear_solve_tol, transpose=transpose,
         )

+    # `_traceable_result_linear_solve_factors` deliberately returns ``None`` on
+    # the LS runtime lane so adjoint solves stay matrix-free. The operator
+    # path below uses pure-JAX operator GMRES (`_hessian_linear_operator` +
+    # `_solve_square_array_system_operator_only`) and remains fully traceable
+    # under JIT — it does not call a live host solver. Removing this path
+    # would force every LS warm-start and adjoint solve to surface
+    # ``success=False`` and emit NaN gradients (verified by
+    # ``test_runtime_bundle_allows_strict_transfer_guard`` /
+    # ``test_runtime_bundle_host_wrappers_allow_host_inputs_under_strict_transfer_guard``).
     objective_fn = _make_boozer_penalty_objective_closure(
         coil_set_spec=coil_set_spec,
         **_traceable_inner_objective_kwargs(objective_kwargs),
     )
     return _optimizer_jax._solve_hessian_least_squares_system_with_status(
         objective_fn, solved_x, rhs,
         stab=float(linear_solve_stab), tol=linear_solve_tol,
     )
```

### Verification

- `ruff check` passes.
- `ruff format` no changes.
- The two integration tests pass:
  - `test_runtime_bundle_allows_strict_transfer_guard` PASS
  - `test_runtime_bundle_host_wrappers_allow_host_inputs_under_strict_transfer_guard` PASS

### Recommendation for follow-up

The audit's underlying intent — eliminate the `None` branch by
populating `baseline_linear_solve_factors` from the solved state and
flipping `_traceable_result_linear_solve_factors` to surface them on
the LS lane — remains a valid architectural improvement, but it is
out of scope for this surgical pass: it requires touching
`_build_traceable_objective_state` and the LS-result schema, and
re-validating cross-lane parity (PR-13 contract). Filed for a separate
work item.

## Combined verification

```bash
ruff check src/simsopt/geo/boozersurface_jax.py src/simsopt/geo/surfaceobjectives_jax.py
# → All checks passed!

ruff format src/simsopt/geo/boozersurface_jax.py src/simsopt/geo/surfaceobjectives_jax.py
# → 2 files left unchanged

/Users/suhjungdae/code/columbia/simsopt-jax/.conda/jax/bin/python -c \
  "from simsopt.geo.boozersurface_jax import BoozerSurfaceJAX; \
   from simsopt.geo.surfaceobjectives_jax import BoozerResidualJAX"
# → OK

/Users/suhjungdae/code/columbia/simsopt-jax/.conda/jax/bin/python -m pytest \
  tests/geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXExactPath::test_exact_result_dict_keys -v
# → 1 passed

/Users/suhjungdae/code/columbia/simsopt-jax/.conda/jax/bin/python -m pytest \
  tests/integration/test_single_stage_jax.py -m "not private_optimizer_runtime" -v
# → 7 passed
```
