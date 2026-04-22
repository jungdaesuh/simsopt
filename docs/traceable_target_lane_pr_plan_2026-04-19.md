# simsopt-jax traceable target-lane PR plan

**Date:** 2026-04-19  
**Status:** Historical implementation plan; materially stale after 2026-04-22 landing work
**Scope:** Single-stage target-lane `same_coils` hot-path cleanup and exact-mode adjoint-state generalization

This note complements [docs/single_stage_banana_ondevice_hot_path_diagnosis_2026-04-19.md](docs/single_stage_banana_ondevice_hot_path_diagnosis_2026-04-19.md).

It records the validated two-PR plan after checking the current tree, the
single-stage target-lane launcher, the private on-device L-BFGS implementation,
and the current dense-PLU exact-adjoint contract.

## 2026-04-23 status update

The plan below remains useful as a design record, but it is no longer a pure
future-work list. The seeded target-lane seam has landed, and the adjoint
runtime seam has partially landed. Treat the rest of this file as historical
context plus remaining follow-up.

### Completed from this plan

- [x] Private on-device L-BFGS accepts `initial_value_and_grad`.
- [x] Seeded L-BFGS finalization reuses the initial value/gradient when
  `state.k == 0` instead of always reevaluating.
- [x] `target_minimize(...)` threads `initial_value_and_grad` through the
  `lbfgs-ondevice` explicit value-and-grad lane.
- [x] The traceable single-stage runtime exposes
  `make_traceable_objective_seeded_value_and_grad(...)` for the optimizer-facing
  seeded path.
- [x] The single-stage example uses the seeded explicit value-and-grad helper
  on the target lane instead of always routing through the older public
  baseline-aware boundary.
- [x] The main adjoint seam is now operator-first through
  `get_adjoint_runtime_state()`, `solve_forward(_with_status)`,
  `solve_transpose_with_status`, and `stream_group_vjps`.
- [x] Legacy CPU/reference surface-objective consumers were migrated off direct
  `res["PLU"]` / `res["vjp"]` access onto the runtime-state seam.
- [x] Much of the dense-specific traceable payload was renamed from `*_plu` to
  `*_linear_solve_factors`, matching the active seam more closely.

### Still open

- [ ] Matrix-free exact adjoint mode behind an explicit option is still not
  implemented.
- [ ] Exact-mode success semantics are still not fully split into separate
  primal-success and adjoint-state-availability contracts.
- [ ] Some compatibility payloads still expose legacy `plu` / `PLU` aliases,
  mainly in lower-level Boozer result dictionaries and CPU fallback wrappers.
- [ ] The historical line references and some wording below were validated on
  2026-04-19 and should not be treated as current line-accurate anchors.

## Confirmed current-tree facts

Historical snapshot from 2026-04-19. The status update above is the current
source of truth.

| Claim | Location | Status |
| --- | --- | --- |
| Private on-device L-BFGS seeds its state from an eager `value_and_grad` call before entering the jitted `while_loop` | `src/simsopt/geo/optimizer_jax_private/_lbfgs.py:351` | verified |
| Private on-device L-BFGS unconditionally reevaluates `value_and_grad_fun(state.x_k)` after the `while_loop` | `src/simsopt/geo/optimizer_jax_private/_lbfgs.py:667` | verified |
| The single-stage traceable forward path still carries a baseline-aware `same_coils` `lax.cond` | `src/simsopt/geo/surfaceobjectives_jax.py:2145-2235` | verified |
| The compiled single-stage bundle is built in one place and already separates public runtime boundaries from internal compiled closures | `src/simsopt/geo/surfaceobjectives_jax.py:2489-2569` and `:2693-2715` | verified |
| The single-stage example already has an explicit target-lane `value_and_grad` path | `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:5292-5348`, `:6660-6727` | verified |
| The exact-mode warm-start predictor still uses a forward PLU solve, not just the transposed adjoint solve | `src/simsopt/geo/surfaceobjectives_jax.py:2377` | verified |
| Exact-mode `run_code_traceable()` still defines success as `result["success"] & finite & jacobian_available_jax` | `src/simsopt/geo/boozersurface_jax.py:2387` | verified |
| Wrapper/runtime adjoint state is still hard-coded to `.plu` | `src/simsopt/geo/boozersurface_jax.py:180-187` and `src/simsopt/geo/surfaceobjectives_jax.py:963-973` | verified |

## Non-goals

- Do not move the hot-path baseline dispatch to Python host logic inside a jitted function.
- Do not rewrite the exact-Newton `lax.while_loop` as `scan`.
- Do not change the public `runtime_bundle["objective"]` or `runtime_bundle["value_and_grad"]` contracts in PR 1.
- Do not make matrix-free adjoints the default in PR 2.

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

## PR 2: Generalize exact-mode adjoint state and add matrix-free mode

Status on 2026-04-23: partially landed. The adjoint runtime seam and much of the
renaming/generalization are in place, but the matrix-free exact mode itself is
still pending.

### Goal

Keep dense-PLU exact adjoints as the default compatibility path, but introduce an
abstract adjoint-state contract that can also support a matrix-free exact adjoint
solve without forcing the exact primal path to fail at dense-Jacobian finalization.

### Planned changes

1. Generalize adjoint state first.

   Update:

   - `src/simsopt/geo/boozersurface_jax.py:_BoozerAdjointRuntimeState`

   so they no longer assume a concrete `.plu` payload as the only valid adjoint state.

2. Split success semantics in the same change that introduces matrix-free mode.

   The exact-mode result currently defines:

   ```python
   success = result["success"] & finite & jacobian_available_jax
   ```

   In matrix-free mode, `jacobian_available_jax` would otherwise stay false and
   silently force the outer objective down the failure-penalty path.

   Introduce:

   - `primal_success`
   - `adjoint_state_available`

   and keep legacy `success` only as compatibility glue during migration.

3. Rename internal dense-specific plumbing before changing behavior.

   In `src/simsopt/geo/surfaceobjectives_jax.py`, rename internal fields such as:

   - `baseline_plu`
   - `solved_plu`

   to abstract adjoint-state names, while keeping dense-PLU behavior intact in the
   first refactor.

4. Add symmetric solver dispatch points.

   Keep `_solve_boozer_adjoint(adjoint_state, rhs)` as the transpose solve seam,
   and add:

   ```python
   def _solve_boozer_forward(adjoint_state, rhs):
   ```

   because matrix-free mode must cover both:

   - the transposed adjoint solve currently routed through `_solve_boozer_adjoint`
   - the forward warm-start solve currently routed through
     `_solve_plu_with_refinement(*baseline_plu, -forcing)` at
     `surfaceobjectives_jax.py:2377`

5. Add matrix-free exact adjoint mode behind an option.

   In `BoozerSurfaceJAX`, add an exact-mode option such as:

   - `exact_adjoint_mode="dense_plu" | "matrix_free"`

   with dense PLU remaining the default.

   Dense mode:

   - preserves current `jacobian`, `plu`, and `scaling_limit` behavior

   Matrix-free mode:

   - skips dense-Jacobian finalization for adjoint availability
   - carries an abstract adjoint-state payload instead
   - solves linear systems through the new forward/adjoint dispatch layer

6. Route both exact-mode linear solves through the new abstraction.

   Dense mode:

   - `_solve_boozer_forward(...)` -> `_solve_plu_with_refinement(...)`
   - `_solve_boozer_adjoint(...)` -> `_solve_plu_transpose_with_refinement(...)`

   Matrix-free mode:

   - forward solve through a non-transposed linear operator
   - adjoint solve through the transposed operator

### Tests for PR 2

1. Dense-mode compatibility tests

   - existing exact-mode behavior and `scaling_limit` reports remain unchanged in dense mode

2. Matrix-free operator parity tests

   - for a small fixture, verify that the matrix-free transpose action matches the
     transpose defined by JAX autodiff on the same residual function

3. Warm-start predictor tests

   - matrix-free mode supports the forward warm-start solve path used by
     `_traceable_predict_warmstart_x(...)`

4. Success-semantics tests

   - matrix-free exact solves can report real objective values instead of being
     forced into failure-penalty mode by dense-Jacobian availability gating

5. Scaling tests

   - dense mode still reports `failure_category="scaling_limit"` and
     `failure_stage="dense_jacobian_finalization"` when appropriate
   - matrix-free mode avoids that dense-finalization failure path on the same fixture

## Final comparison against earlier drafts

This validated plan differs from earlier drafts in four important ways:

1. It does not attempt Python host dispatch inside traced optimizer code.
2. It does not treat the exact-Newton `while_loop` as the primary compile-graph bug.
3. It keeps PR 1 scoped to the single-stage target-lane seeded optimizer path.
4. It treats matrix-free exact adjoints as a separate architectural milestone,
   including both transpose and forward linear solves plus the success-contract split.

## Recommended merge order

Historical recommendation from 2026-04-19:

1. PR 1: seeded explicit-VG single-stage target-lane optimizer path
2. PR 2a: adjoint-state renaming/generalization with dense-only behavior
3. PR 2b: matrix-free exact adjoint mode behind a flag
