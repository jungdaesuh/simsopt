# simsopt-jax traceable target-lane PR plan

**Date:** 2026-04-19  
**Status:** Active implementation tracker; PR 1 landed on 2026-04-23 and PR 2 fully closed on 2026-04-24
**Scope:** Single-stage target-lane `same_coils` hot-path cleanup plus the remaining work to make exact JAX adjoints and exact traceable wrappers strict native-JAX CPU/GPU lanes while preserving original CPU/C++ behavior

This note complements [docs/single_stage_banana_ondevice_hot_path_diagnosis_2026-04-19.md](docs/single_stage_banana_ondevice_hot_path_diagnosis_2026-04-19.md).

It records the validated two-PR plan after checking the current tree, the
single-stage target-lane launcher, the private on-device L-BFGS implementation,
and the former dense-PLU exact-adjoint contract.

## 2026-04-23 status update

This file is no longer just a historical design note. PR 1 landed on
2026-04-23 and PR 2 fully closed on 2026-04-24. The current strict exact-JAX
direction is:

- preserve original CPU/C++ `PLU` / `forward_backward` / `vjp` lanes unchanged
- keep dense JAX exact solves and upstream SIMSOPT dense solves as
  reference-oracle or metadata paths only
- make JAX-owned exact runtime paths native JAX on CPU/GPU only
- remove fallback or defensive branches from JAX exact adjoint, warm-start, and
  traceable wrapper behavior

### Primary references

Local current-tree references:

- `src/simsopt/geo/boozersurface_jax.py`
- `src/simsopt/geo/surfaceobjectives_jax.py`
- `src/simsopt/geo/optimizer_jax.py`
- `tests/geo/test_surface_objectives_jax.py`
- `benchmarks/validation_ladder_contract.py`

Upstream / external references:

- upstream SIMSOPT exact `PLU` result contract:
  `/Users/suhjungdae/code/opensource/simsopt/src/simsopt/geo/boozersurface.py`
- upstream SIMSOPT CPU wrapper adjoints through `forward_backward(...)`:
  `/Users/suhjungdae/code/opensource/simsopt/src/simsopt/geo/surfaceobjectives.py`
- upstream JAX `custom_linear_solve`:
  `/Users/suhjungdae/code/opensource/jax/jax/_src/lax/control_flow/solves.py`
- upstream JAX `gmres`:
  `/Users/suhjungdae/code/opensource/jax/jax/_src/scipy/sparse/linalg.py`
- upstream JAX custom-VJP and `while_loop` rationale:
  `/Users/suhjungdae/code/opensource/jax/docs/notebooks/Custom_derivative_rules_for_Python_code.md`
- official JAX docs:
  <https://docs.jax.dev/en/latest/_autosummary/jax.custom_vjp.html>,
  <https://docs.jax.dev/en/latest/_autosummary/jax.scipy.sparse.linalg.gmres.html>,
  <https://docs.jax.dev/en/latest/default_dtypes.html>
- official SIMSOPT docs:
  <https://simsopt.readthedocs.io/stable/simsopt.geo.html>
- NVIDIA CUDA floating-point docs:
  <https://docs.nvidia.com/cuda/floating-point/index.html>

### Landed from this plan

- [x] Private on-device L-BFGS accepts `initial_value_and_grad`.
- [x] Seeded L-BFGS finalization reuses the initial value/gradient when
  `state.k == 0` instead of always reevaluating.
- [x] `target_minimize(...)` threads `initial_value_and_grad` through the
  `lbfgs-ondevice` explicit value-and-grad lane.
- [x] The traceable single-stage runtime exposes
  `make_traceable_objective_seeded_value_and_grad(...)` for the optimizer-facing
  seeded path.
- [x] The single-stage example uses the seeded explicit value-and-grad helper on
  the target lane instead of routing the hot path through the older public
  baseline-aware boundary.
- [x] The runtime-state seam exists and is the intended JAX product seam through
  `get_adjoint_runtime_state()`, `solve_forward(_with_status)`,
  `solve_transpose_with_status`, and grouped VJP helpers.
- [x] The least-squares normal-equation solve is operator-only in
  `src/simsopt/geo/optimizer_jax.py`.
- [x] Parity mode now enforces `newton_stab=0.0`.
- [x] Synthetic gradient rescue on adjoint failure is gone. Failed adjoint
  solves now surface non-finite gradients rather than a synthetic direct-term
  substitute.

### Closed on 2026-04-23 for the strict exact-JAX contract

- [x] `dense_jax` is no longer a live exact-runtime backend in
  `src/simsopt/geo/boozersurface_jax.py`.
- [x] The traceable exact solve no longer dispatches on
  `linear_solve_backend` in `src/simsopt/geo/surfaceobjectives_jax.py`.
- [x] Adjoint-only failure now keeps the real primal value and returns a
  non-finite gradient instead of substituting `value=+inf`.
- [x] Exact warm-start, profile-suite solve, and ALM failure objects now keep
  finite values/status and preserve the failed predicted state instead of
  rebuilding a `+inf` sentinel or snapping back to `baseline_x`.
- [x] Exact warm-start prediction no longer silently falls back to `baseline_x`;
  exact warm-start failure now surfaces explicitly through an unsuccessful
  forward-result state.
- [x] Runtime solve status no longer depends on inferred finiteness where the
  exact runtime already has explicit `*_with_status` solve callbacks.

### Current decision boundary

- CPU/C++ behavior is preserved. Original upstream `PLU` and `forward_backward`
  lanes remain valid compatibility and parity-oracle surfaces.
- JAX-owned exact runtime behavior should be matrix-free and native on CPU/GPU.
- Dense exact solves are allowed only as reference or metadata, not as supported
  JAX runtime behavior.
- `benchmarks/validation_ladder_contract.py::PARITY_LADDER_TOLERANCES` remains
  the SSOT for lane-specific precision gates. The strict-runtime cleanup below
  must preserve the existing direct-kernel and LS-wrapper-gradient lanes.

## Confirmed current-tree facts

The table below mixes historical findings with current corrections. The status
update above is the current source of truth.

| Claim | Location | Status |
| --- | --- | --- |
| Private on-device L-BFGS seeds its state from an eager `value_and_grad` call before entering the jitted `while_loop` | `src/simsopt/geo/optimizer_jax_private/_lbfgs.py:351` | verified |
| Private on-device L-BFGS unconditionally reevaluates `value_and_grad_fun(state.x_k)` after the `while_loop` | `src/simsopt/geo/optimizer_jax_private/_lbfgs.py:667` | verified |
| The single-stage traceable forward path still carries a baseline-aware `same_coils` `lax.cond` | `src/simsopt/geo/surfaceobjectives_jax.py:2145-2235` | verified |
| The compiled single-stage bundle is built in one place and already separates public runtime boundaries from internal compiled closures | `src/simsopt/geo/surfaceobjectives_jax.py:2489-2569` and `:2693-2715` | verified |
| The single-stage example already has an explicit target-lane `value_and_grad` path | `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:5292-5348`, `:6660-6727` | verified |
| The exact-mode warm-start predictor still uses a forward PLU solve, not just the transposed adjoint solve | superseded | fixed: exact warm-start prediction uses `_solve_jacobian_system_with_status(..., transpose=False)` and failed exact warm-start solves now surface explicitly instead of silently reusing `baseline_x` |
| Exact-mode `run_code_traceable()` still defines success as `result["success"] & finite & jacobian_available_jax` | superseded | fixed: dense Jacobian availability is not required for traceable exact adjoint availability |
| Wrapper/runtime adjoint state is still hard-coded to `.plu` | superseded | fixed: JAX runtime state is operator-backed, exposes explicit status callbacks, and treats dense `PLU` only as metadata availability |

## Non-goals

- Do not move the hot-path baseline dispatch to Python host logic inside a jitted function.
- Do not rewrite the exact-Newton `lax.while_loop` as `scan`.
- Do not change the public `runtime_bundle["objective"]` or `runtime_bundle["value_and_grad"]` contracts in PR 1.
- Do not change upstream CPU/C++ `PLU` lane behavior while cleaning up the JAX exact lane.

## PR 1: Remove `same_coils` from the optimizer hot path

Status on 2026-04-23: landed.

### Goal

Keep the public traceable runtime bundle baseline-aware, but give the single-stage
target-lane optimizer a separate seeded explicit `value_and_grad` entrypoint that
never traces the baseline `lax.cond` in its hot path.

### Planned changes

1. Add optional seeding to private explicit-value-and-grad L-BFGS.

   Extend:

   - `src/simsopt/geo/optimizer_jax_private/_lbfgs.py:_minimize_lbfgs_private_impl`
   - `src/simsopt/geo/optimizer_jax_private/_lbfgs.py:_minimize_lbfgs_private_value_and_grad`

   with:

   ```python
   initial_value_and_grad: tuple[jax.Array, jax.Array] | None = None
   ```

   Behavior:

   - At solver entry, use `initial_value_and_grad` instead of calling
     `_coerce_value_and_grad_result(value_and_grad_fun, x0)` when a seed is provided.
   - After the `while_loop`, replace the unconditional reevaluation with a
     `lax.cond`:
     - reuse `state.f_k, state.g_k` when `state.k == 0` and a seed was provided
     - otherwise reevaluate as today

   Constraint:

   - Both `lax.cond` branches must return the same pytree structure and dtypes.
   - The reuse branch must return the normalized optimizer state payload shape,
     not a structurally different object.

2. Thread the seed through `target_minimize(...)`.

   Extend `src/simsopt/geo/optimizer_jax.py:target_minimize` with:

   ```python
   initial_value_and_grad=None
   ```

   and only honor it on:

   - `method == "lbfgs-ondevice"`
   - `value_and_grad is True`

   All other branches remain unchanged.

3. Add a general-only single-stage compiled path.

   In `src/simsopt/geo/surfaceobjectives_jax.py`, add a helper:

   ```python
   def _traceable_forward_result_general_only(...):
   ```

   whose body is exactly the current `general_case` branch from
   `_traceable_forward_result(...)`, with no outer `same_coils` `lax.cond`.

   Then extend
   `_build_traceable_objective_compiled_bundle_from_state(...)` to build:

   - `compiled_forward_result_general_only`
   - `compiled_value_and_grad_for_general_only`
   - `optimizer_initial_value_and_grad`

   with:

   ```python
   optimizer_initial_value_and_grad = (
       _as_jax_float64(baseline_value),
       baseline_gradient,
   )
   ```

   where `baseline_gradient` is the implicit total gradient at the cached baseline.

   Transfer-guard note:

   - The bundle state is hostified in `_build_traceable_objective_state(...)`.
   - If baseline gradient materialization requires host transfer during bundle
     construction, wrap that bootstrap in `with jax.transfer_guard("allow"):` in
     the same style already used for host-wrapper baseline gradient materialization
     at `surfaceobjectives_jax.py:2792`.

4. Add a seeded public helper for optimizer use.

   Next to `make_traceable_objective_value_and_grad(...)`, add:

   ```python
   class TraceableSeededValueAndGrad(NamedTuple):
       value_and_grad: Callable[[jax.Array], tuple[jax.Array, jax.Array]]
       initial_value_and_grad: tuple[jax.Array, jax.Array]
   ```

   and:

   ```python
   def make_traceable_objective_seeded_value_and_grad(...):
   ```

   This helper should return:

   - a boundary around `compiled_value_and_grad_for_general_only`
   - the cached baseline seed

   Rationale:

   - this keeps the existing runtime-bundle dict contract stable
   - the seeded helper is an optimizer-facing API, not a general runtime-bundle API

5. Wire only the single-stage target-lane explicit-VG path to the seeded helper.

   Update:

   - `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py`

   Specifically:

   - `build_target_lane_outer_objectives(...)`
   - `prepare_target_lane_outer_objectives(...)`
   - `run_single_stage_optimizer(...)`
   - `run_single_stage_target_lane_optimizer_with_retries(...)`

   Rules:

   - pass the seed only on the first optimizer attempt whose `x0` matches the
     bundle baseline
   - drop the seed on retries, because retries restart from an anchor state, not
     from the cached bundle baseline
   - this rule is not phase-specific; it is baseline-match-specific

   The actual `target_minimize(...)` call is assembled through
   `target_minimize_kwargs` at `single_stage_banana_example.py:6704-6716` and
   invoked at `:6723`.

6. Keep cache semantics unchanged.

   `_traceable_runtime_cache_key(...)` already keys the cached runtime entry by:

   - `id(booz_jax)`
   - `id(bs_jax)`
   - `booz_jax._solver_generation`
   - option/config signatures

   The seeded optimizer helper depends only on the same baseline state already
   represented by that key, so the existing cache key should remain valid.

### Tests for PR 1

1. Private L-BFGS tests in `tests/geo/test_boozersurface_jax_private.py`

   - seeded `maxiter=0` run does not reevaluate the objective
   - seeded run with iterations still reevaluates only when `state.k > 0`
   - `lax.cond` finalization returns matching pytree structure and dtypes on both branches

2. Traceable single-stage bundle tests in `tests/geo/test_surface_objectives_jax.py`

   - seeded helper returns the expected cached baseline seed
   - seeded helper `value_and_grad` matches the existing public `value_and_grad`
     on non-baseline inputs
   - runtime-bundle public keys remain unchanged
   - cache reuse returns the same cached seeded payload for the same runtime entry

3. Single-stage example tests in `tests/geo/test_single_stage_example.py`

   - target-lane explicit-VG path passes `initial_value_and_grad` into
     `target_minimize(...)`
   - retries do not reuse the original baseline seed after the first attempt
   - strict transfer-guard mode still succeeds during seeded bootstrap

4. Automated lowered-program regression

   Replace the one-off manual HLO check with an automated regression that proves
   the optimizer-facing seeded helper lowers through the general-only forward path
   rather than the baseline-aware `same_coils` `lax.cond`.

## PR 2: Finish the strict native-JAX exact-adjoint contract

Status on 2026-04-24: landed. The runtime-state seam, exact traceable wrapper
cleanup, failure semantics, and native CPU/GPU parity coverage now match the
strict operator-only exact-JAX contract.

### Goal

Preserve original CPU/C++ exact behavior while making JAX-owned exact adjoints,
exact warm-start prediction, and exact traceable wrappers strict native-JAX
CPU/GPU paths. Dense exact JAX solves and upstream SIMSOPT dense solves remain
reference oracles and metadata producers, not runtime fallbacks.

### Planned changes

1. Remove exact runtime backend selection.

   Update `src/simsopt/geo/boozersurface_jax.py` so exact runtime behavior no
   longer depends on:

   - `_DEFAULT_OPTIONS_EXACT["exact_adjoint_linear_solve_backend"]`
   - `_EXACT_ADJOINT_LINEAR_SOLVE_BACKENDS`
   - `_exact_adjoint_linear_solve_backend()`
   - the `dense_jax` branch inside `_build_runtime_linear_solve_callbacks(...)`

   Required result:

   - exact JAX runtime always reports `linear_solve_backend="operator"`
   - dense exact artifacts may still be emitted as metadata with
     `dense_linear_solve_factors_available`
   - original CPU/C++ exact `PLU` behavior is untouched

2. Collapse the traceable exact solve to the same operator seam.

   Update `src/simsopt/geo/surfaceobjectives_jax.py` so
   `_traceable_solve_exact_linearization(...)` and all downstream compiled state
   use only the operator exact linear solve.

   Required cleanup:

   - remove `linear_solve_backend` dispatch from exact traceable solve paths
   - stop threading exact backend policy through compiled bundle state, cache
     keys, and public traceable boundaries
   - keep dense exact solves available only in dedicated reference helpers/tests

3. Remove defensive status synthesis.

   In `src/simsopt/geo/boozersurface_jax.py`, delete inferred status wrappers
   for exact runtime behavior where explicit `*_with_status` solve callbacks
   already exist.

   Required result:

   - exact runtime status comes only from explicit solver status callbacks
   - the checked adjoint path does not reinterpret finiteness as success

4. Remove warm-start fallback-to-baseline behavior.

   In `src/simsopt/geo/surfaceobjectives_jax.py:_traceable_predict_warmstart_x`,
   remove the `baseline_x` rescue branch on operator failure.

   Required result:

   - exact warm-start predictor either produces a valid operator solve result or
     surfaces failure explicitly
   - exact warm-start behavior does not silently erase a failed linear solve

5. Tighten exact failure semantics in traceable wrappers.

   Update forward/value-and-grad/seeded/host/ALM JAX wrappers in
   `src/simsopt/geo/surfaceobjectives_jax.py` so:

   - `primal_success=True` and adjoint failure keeps the real primal value
   - adjoint failure returns a non-finite gradient
   - primal failure remains a failed state and must not be rewritten as a
     “successful but penalized” value

6. Remove `+inf` adjoint-only value substitution.

   Delete `_TRACEABLE_ADJOINT_FAIL_VALUE_SENTINEL` usage from the exact
   traceable wrapper paths and replace that signaling with explicit status plus
   non-finite gradient behavior.

7. Keep dense exact solves as reference-only validation tools.

   Status: landed for the well-conditioned exact oracle lane in
   `tests/geo/test_boozersurface_jax.py::
   test_exact_well_conditioned_operator_adjoint_matches_dense_reference_and_plu`.
   That fixture compares:

   - JAX operator solve on the native runtime path
   - dense JAX exact reference solve
   - upstream SIMSOPT `PLU` reference solve

   Dense exact solves remain reference-only validation tools and must not
   return as a supported runtime backend.

8. Add explicit native JAX CPU/GPU parity coverage on the exact lane.

   Required runtime lane:

   - `jax_enable_x64=True`
   - fixed seeds
   - recorded JAX/CUDA/device metadata
   - no host or SciPy fallback in the supported JAX lane

   Status: landed in `tests/geo/test_boozersurface_jax.py::
   test_exact_well_conditioned_operator_adjoint_cpu_gpu_same_state_parity`.
   Keep the reduced real public CPU/GPU parity coverage as-is; it complements
   the exact-lane same-state adjoint gate rather than replacing it.

### Acceptance criteria

- Exact JAX runtime has no live `dense_jax` backend.
- Exact traceable wrappers do not dispatch on backend choice.
- Exact adjoint failure never returns a synthetic finite gradient.
- Adjoint-only failure does not overwrite a valid primal value with `+inf`.
- Exact warm-start no longer silently falls back to `baseline_x`.
- CPU/C++ `PLU` and `forward_backward` lanes remain unchanged.
- JAX CPU and JAX GPU share the same operator-only exact implementation.

### Precision gates

- well-conditioned exact operator vs dense-JAX vs upstream-`PLU` reference:
  `rtol=1e-6`, `atol=1e-8`, residual `<= 1e-10`
- same-state JAX CPU vs JAX GPU forward:
  `rtol=1e-10`, `atol=1e-12`
- same-state JAX CPU vs JAX GPU gradients:
  `rtol=1e-8`, `atol=1e-10`
- whole-solve JAX CPU vs JAX GPU values:
  `rtol=1e-6`, `atol=1e-7`
- existing direct-kernel and LS-wrapper-gradient precision lanes remain governed
  by `benchmarks/validation_ladder_contract.py`

### Tests for PR 2

Replace the stale dense-vs-operator runtime test assumptions with the stricter
contract:

1. Exact runtime contract tests

   - exact runtime state reports `linear_solve_backend="operator"`
   - exact runtime state exposes explicit status-bearing solve callbacks
   - no runtime caller can select `dense_jax`

2. Well-conditioned exact reference-oracle tests

   - landed in `tests/geo/test_boozersurface_jax.py::
     test_exact_well_conditioned_operator_adjoint_matches_dense_reference_and_plu`
   - operator exact adjoint agrees with dense JAX reference and upstream SIMSOPT
     `PLU` on the same well-conditioned state to the precision gates above

3. Warm-start predictor tests

   - landed: failed exact warm-start solve does not silently reuse `baseline_x`
   - landed: successful exact warm-start solve agrees with the reference
     operator linearization on a branch-stable fixture in
     `tests/geo/test_surface_objectives_jax.py::
     test_traceable_exact_warmstart_success_matches_reference_operator_linearization`

4. Success-semantics tests

   - primal-success plus adjoint-fail keeps the real primal value and returns a
     non-finite gradient
   - primal failure remains a failed state and does not pass through a
     substitute-value branch

5. Native CPU/GPU parity tests

   - landed: reduced real public runtime lane has CPU/GPU parity coverage in
     `tests/integration/test_single_stage_jax_cpu_reference.py`
   - landed: the well-conditioned exact-adjoint lane has an explicit same-state
     CPU-vs-GPU parity assertion using the existing exact oracle fixture
   - landed: same-state adjoint parity on the operator-only exact lane
   - landed: same-state projected-gradient parity on the operator-only exact lane
   - retained separately: whole-solve parity at the looser GPU-runtime lane
     thresholds

## Final comparison against earlier drafts

This validated plan differs from earlier drafts in four important ways:

1. It does not attempt Python host dispatch inside traced optimizer code.
2. It does not treat the exact-Newton `while_loop` as the primary compile-graph bug.
3. It keeps PR 1 scoped to the single-stage target-lane seeded optimizer path.
4. It treats exact JAX strict-runtime cleanup as a removal of fallback behavior,
   not as an invitation to redesign or weaken the preserved CPU/C++ lanes.

## Recommended merge order

Historical recommendation from 2026-04-19:

1. PR 1: seeded explicit-VG single-stage target-lane optimizer path
2. PR 2a: exact runtime cleanup to operator-only behavior
3. PR 2b: exact traceable wrapper cleanup plus native CPU/GPU parity tests
