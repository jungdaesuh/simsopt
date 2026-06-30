# Single-Stage Convergence Root Solutions Implementation Plan

> Local adaptation: 2026-06-26 for
> `/Users/suhjungdae/code/columbia/simopt-jax-clean-local`.
> Source plan: sibling checkout
> `/Users/suhjungdae/code/columbia/simsopt-pr-jax-port-clean/docs/single_stage_convergence_root_solutions_implementation_plan_2026-06-26.md`
> at `fdc3ae192`.
> Local source checked at `951c0b71d`.
> Status: PLAN. This document records required fixes; it does not claim they are
> already implemented.

## Purpose

Make the sibling single-stage convergence plan executable in this checkout. The
root symptom remains a single-stage banana optimization that evaluates finite
L-BFGS-B trial points but can terminate with
`ABNORMAL_TERMINATION_IN_LNSRCH`. The local evidence still points at the
single-stage outer target-lane driver and its rejection path, not at the core
BiotSavart/Boozer JAX kernels.

This checkout did not previously contain this document. The local plan is the
source of truth for applying the convergence work here.

## Local Review State

- Current branch: `simopt-jax-clean-local`.
- Current HEAD: `951c0b71d Merge dense-LU / adjoint-gate stack from pr/jax-port-clean`.
- Dirty worktree at review time:
  `examples/single_stage_optimization/SINGLE_STAGE/run_single_stage_continuation.py`,
  `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py`,
  `tests/integration/test_continuation_ladder.py`, and
  `tests/integration/test_single_stage_physics_parity.py`.
- The dirty source slice adds or extends `scipy-jax-decomposed` entrypoint and
  continuation-ladder behavior. Preserve that slice; do not mix convergence-plan
  code fixes into it without an explicit staging decision.
- External JAX docs were refreshed through `ctx7` against `/jax-ml/jax`.
  The relevant benchmarking contract is still to separate device transfer,
  compilation, and execution timing, and to use `block_until_ready()` for JAX
  timing. That supports keeping compile/cold-start claims separate from
  optimizer convergence claims.

## Verified Local Anchors

- Objective-evaluation trace flags exist in
  `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:5120`
  and `:5129`; `--replay-objective-evaluation-trace` is at `:5145`.
- The target-lane hardware filter is still a boolean feasibility predicate in
  `single_stage_banana_example.py:7487-7679`.
- The self-intersection filter is a separate hard predicate in
  `single_stage_banana_example.py:6755-6817`; keep it separate unless a real
  differentiable surrogate is designed.
- The combined success predicate still `logical_and`s hardware and
  self-intersection filters in `single_stage_banana_example.py:7682-7732`.
- `_traceable_rejected_objective_value` still stop-gradients the candidate and
  returns a flat elevated value in
  `src/simsopt_jax_adapters/geo/surface_objectives.py:1038-1048`.
- The traceable forward path still gates the objective with a boolean
  `success_filter` and calls the rejected-value helper in
  `src/simsopt_jax_adapters/geo/surface_objectives_traceable.py:718-739`.
- Rejection gradients still fall back to baseline gradients in the fused and
  decomposed traceable paths:
  `surface_objectives_traceable.py:1410-1424`, `:2043-2056`,
  and `:3640-3648`.
- The dirty decomposed host fallback test currently pins baseline-gradient
  behavior in `tests/integration/test_single_stage_jax_cpu_reference.py:7112`.
- High-mpol `ftol_by_mpol` still falls below the proposed noise floor at
  mpol 15-18 in `single_stage_banana_example.py:14066-14078`, with fallback
  resolution at `:15357-15360`.
- The dense square-operator least-squares solve path is in
  `src/simsopt_jax/geo/optimizers/optimizer.py:4773-4803`. The float64
  numerical-safety guard already uses a condition screen and deliberately keeps
  the forward-error bound float32-only at `:4545-4598`.
- The Phase 6 contract contradiction is live: `_materialize_dense_hessian_host`
  delegates to `_materialize_dense_hessian` at
  `src/simsopt_jax/geo/optimizers/optimizer.py:3701-3702`, while
  `tests/geo/test_boozersurface_jax.py:2114-2138` expects host control not to
  call `_materialize_dense_hessian`. The item19 tests at
  `tests/geo/test_optimizer_jax_item19.py:32-58` expect host and device
  materializers to agree and no separate `_materialize_dense_linear_operator_host`
  helper to exist.

## Non-Goals

- Do not rewrite core JAX kernels, Boozer residual kernels, or BiotSavart
  kernels as part of this plan.
- Do not switch optimizer libraries as a substitute for fixing the rejected-step
  objective/gradient contract. The problem here is the target-lane objective
  surface presented to SciPy host control.
- Do not weaken tolerances to hide line-search stalls or adjoint failures.
- Do not fold self-intersection into the hardware smooth barrier without a
  separate differentiable surrogate design.
- Do not extend the float64 dense-solve forward-error gate without a failing
  production-scale repro. The local source already explains why that can
  false-reject large accurate float64 solves.

## Goals

- Produce a real converged single-stage result: monotone objective decrease,
  `status == 0`, at least one accepted L-BFGS-B step, decreasing projected
  gradient norm, and iota moving toward target.
- FD-certify the production operator/lstsq gradient at production-like scale.
- Replace the hard hardware-rejection plateau with a point-dependent smooth
  exterior penalty for distance/curvature violations.
- Keep true Boozer-solve failures loud and distinguish them from hardware-margin
  violations.
- Decide and implement one `newton_polish` host-materialization contract so the
  focused tests agree.

## Implementation Plan

### Phase 0 - Reconfirm the failing seed mechanism

1. Re-run the rejected single-stage config with
   `--record-objective-evaluation-trace`.
2. Omit `--compact-objective-evaluation-trace` when replay-grade vectors are
   needed.
3. Inspect `${OUT_DIR_ITER}/outer_optimizer_progress.json` objective-evaluation
   events for the outer trial `(x, f, g)` tuples.
4. Preserve the SciPy `message` and low-level task text. `status=2` alone is
   not enough proof of a line-search stall.
5. Keep `record_scipy_callback_trace` separate: it is adapter/Boozer metadata,
   not the outer single-stage line-search trace.
6. Measure the seed-local objective noise by evaluating fixed coils with two
   inner tolerances, for example `newton_tol=1e-11` and `1e-13`.

Expected classification: all finite trial values/gradients with tiny decrease
or plateau. If any trial produces NaN or non-finite gradient, reopen the adjoint
failure path before changing the barrier.

### Phase 1 - FD-certify the production gradient

Add a focused production-scale finite-difference gate, for example
`tests/integration/test_single_stage_production_gradient_fd.py`.

Required properties:

- Build a real mpol 8 fixture.
- Converge the inner Boozer solve.
- Sample 3 to 5 deterministic random unit coil-DOF directions.
- Compare central differences against the adjoint directional derivative.
- Re-solve the inner Boozer problem for each probe.
- Use an eps ladder in the production-noise window, e.g.
  `(3e-3, 1.5e-3, 7.5e-4)`, not the existing small toy ladders around
  `1e-4`.
- Prove the exercised path is the operator/lstsq path:
  `linear_solve_factors=None` reaching
  `_solve_dense_square_operator_least_squares_system_with_status`, not only an
  eager dense-PLU toy path.

Pass condition: Taylor-rate decrease plus an absolute directional-derivative
error near `1e-6`, adjusted only after measuring seed-local `deltaJ`.

### Phase 2 - Get a reduced-lane converged GPU result

Run the reduced `--optimizer-backend scipy-jax` lane first at `mpol <= 6`,
because it avoids the full production compile breadth while still exercising
the target-lane value/gradient contract.

Requirements:

- Use a persistent compile cache on a durable run volume, not `/tmp`.
- Record a warm-cache second run.
- Do not accept the convergence result as production-gradient proof until
  Phase 1 passes.
- Store the convergence table from the progress JSON:
  outer iteration, objective, projected-gradient or infinity-norm gradient,
  accepted step, status/message.

### Phase 3 - Replace the hardware rejection cliff

The local code currently has only a boolean hardware success filter and a
rejected-value helper that has no residual input. Therefore the fix must add a
real residual contract; simply dropping `stop_gradient` in
`_traceable_rejected_objective_value` is not sufficient.

1. Add a single-stage hardware constraint evaluator next to
   `build_single_stage_target_lane_hardware_success_filter`.
2. Return the four positive-when-violating residuals:
   `cc_dist - curve_curve_min_dist`,
   `cs_dist - curve_surface_min_dist`,
   `ss_dist - surface_vessel_min_dist`, and
   `max_curvature - curvature_threshold`.
3. Keep the boolean success predicate as a derived value or update every
   `success_filter` consumer/cache/test in one coherent contract change. Do not
   make a bool-typed `success_filter` silently return an array.
4. Feed residuals into a smooth exterior barrier:
   `base_objective + sum_k w_k * max(0, residual_k)^2`.
5. Propagate that barrier through
   `surface_objectives_traceable.py:718-739` and every rejected-gradient
   fallback so the rejected candidate returns a point-dependent gradient.
6. Update the tests that currently pin baseline-gradient behavior, especially
   the dirty decomposed host fallback test in
   `tests/integration/test_single_stage_jax_cpu_reference.py:7112`.
7. Preserve a separate failure handling path for true Boozer-solve failure.
   That path may return a loud failure penalty, but it must not be confused with
   hardware-margin rejection.

### Phase 4 - Match optimizer tolerances to measured noise

After Phase 0 measures seed-local `deltaJ`, floor high-mpol `ftol` at a value
that is safely above the objective noise floor. The sibling plan proposes
`ftol >= 1e-8` for high mpol; this checkout still sets mpol 15-18 below that.

Implementation choices:

- Adjust `ftol_by_mpol` directly, or
- Clamp at the resolution site that computes `ftol` from `args.outer_ftol`.

Do not tighten inner `newton_tol` by default until the measured `deltaJ` says it
is needed; it increases per-evaluation cost.

### Phase 5 - Treat dense-solve hardening as repro-first

The local dense-solve safety code is already more specific than the stale
high-level diagnosis. It condition-screens float64 solves and applies the
forward-error bound only to float32.

Before changing it:

1. Add a regression/probe that demonstrates
   `_dense_matrix_solve_numerically_safe` accepts a near-singular float64
   production iterate that should fail closed.
2. If the repro exists, choose the smallest fix, such as a dimension-aware
   condition criterion or relative stabilization floor.
3. Re-run the Phase 1 FD certificate after any dense-solve safety change.

### Phase 6 - Measure compile breadth before narrowing

This checkout already has `benchmarks/compile_breadth_probe.py`. Use it before
rewriting compile boundaries.

Run mpol 6, 8, and 10 probes under `JAX_LOG_COMPILES=1` and an XLA dump
directory. Record whether K1 forward solve, K2 solved-state value/gradient, or
the fused value/gradient kernel dominates.

Only after measurement:

- Promote the decomposed split if it proves to remove the dominant fused
  compile.
- Revisit dense-adjoint materialization gates only if the probe implicates that
  path.
- Use `benchmarks/check_cached_kernel_callback_compatibility.py` and
  `tests/test_check_cached_kernel_callback_compatibility.py` for callback-free
  cacheability checks.

### Phase 7 - Resolve `newton_polish` host materialization

The local focused test result is currently:

```text
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
/opt/homebrew/Caskroom/miniforge/base/bin/python -m pytest \
  tests/geo/test_boozersurface_jax.py::TestOptimizerAdapter::test_newton_polish_host_control_uses_host_dense_materialization \
  tests/geo/test_optimizer_jax_item19.py::test_item19_host_dense_hessian_reuses_chunked_device_materializer \
  tests/geo/test_optimizer_jax_item19.py::test_item19_host_dense_hessian_agrees_with_device_materializer \
  -q

F..  # 1 failed, 2 passed
```

Make one product decision:

- Keep `allow_host_control=True` on a true host materializer and update item19
  expectations accordingly, or
- Route host control through the existing chunked device materializer and update
  the host-control test/comment to assert that contract.

Do not leave both claims in the test suite.

## Validation Plan

- `git status --short` before and after edits; preserve unrelated dirty files.
- `rg` path checks for every referenced file and symbol in this document.
- Focused Phase 7 test above: expected to fail until the product decision is
  implemented.
- After Phase 3: targeted rejection-path tests in
  `tests/geo/test_surface_objectives_jax.py` and
  `tests/integration/test_single_stage_jax_cpu_reference.py`.
- After Phase 3/4: run a same-seed A/B with pre-fix `status=2, nfev=21` versus
  post-fix accepted-step trajectory.
- After Phase 1: production FD cert at mpol 8.
- After Phase 6: `benchmarks/compile_breadth_probe.py` result JSON committed or
  archived with the run command and environment.
- Before declaring completion: Stage-2 JAX and core Boozer/BiotSavart parity
  tests still pass, because this plan is not allowed to perturb core kernels.

## Completion Criteria

- A local converged single-stage GPU result exists with `status == 0` and at
  least one accepted step.
- The production-gradient FD certificate passes.
- Full-space target-lane runs no longer fail from the hardware rejection cliff.
- High-mpol `ftol` is tied to measured objective noise.
- Dense-solve safety remains repro-backed.
- The `newton_polish` host-materialization focused tests agree.
- The dirty `scipy-jax-decomposed` and continuation-ladder work is either kept
  separate or intentionally integrated with explicit staging evidence.

## Open Questions

- Is the immediate milestone a reduced-lane mpol <= 6 converged GPU result, or a
  full production mpol 10 result?
- Should the hardware residual/barrier contract live only in the single-stage
  example, or become an adapter-level single-stage constraint contract?
- Which `newton_polish` host-materialization contract is intended: independent
  host materializer, or shared chunked device materializer under host control?
- What is the measured seed-local `deltaJ` for the failing production seed?
