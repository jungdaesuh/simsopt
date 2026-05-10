# CPU/C++ vs JAX CPU Parity Drift Implementation Plan - 2026-05-07

Status: updated plan after validating the May 7 dense-Newton artifacts and the
current code paths.

Current tree inspected at `7f5e526ef`. The working tree was dirty when this
plan was written. This document is additive and does not classify unrelated
uncommitted changes outside the parity-drift surface.

## Goal

Close the remaining CPU/C++ SciPy reference vs JAX CPU fullgraph single-stage
drift without loosening parity tolerances.

The target is strict parity in this trust order:

1. Existing SIMSOPT C++/SciPy behavior.
2. JAX CPU matches the CPU/C++ reference lane.
3. JAX GPU matches the proven JAX CPU lane.

JAX-vs-JAX self-consistency is not enough for this plan.

## Current Evidence

Execution update from the current implementation pass:

- `.artifacts/parity/20260507-cpp-jaxcpu-single-stage-current-m1b/fullgraph-m1-trace.json`
  passed same-candidate replay and full `maxiter=1` CPU/C++ vs JAX CPU parity.
- `.artifacts/parity/20260507-cpp-jaxcpu-single-stage-current-m5/fullgraph-m5-trace.json`
  still fails the full `maxiter=5` gate:
  - `final_iota_abs_diff = 5.814942668005973e-10`
  - `final_volume_rel_diff = 1.2478977864730094e-10`
  - `field_error_rel_diff = 1.5525982944175097e-07`
- Same-candidate replay for the `maxiter=5` run passes with exact candidate
  matching:
  - `same_candidate_event_count = 23`
  - `max_candidate_abs_diff = 0.0`
  - `max_objective_abs_diff = 8.260059303211165e-14`
  - `max_optimizer_gradient_abs_diff = 3.107514245925813e-11`
  - `max_slice_gradient_owner = iota_penalty`
- The free-running optimizer path now has a diagnostic comparator. On the
  `maxiter=5` artifact, the first material candidate split is:
  - `pair_index = 8`
  - `accepted_iteration_target = 3`
  - `line_search_evaluation = 3`
  - `candidate_abs_diff = 2.9750871904932197e-12`
  - final-event `candidate_abs_diff = 1.235757207673771e-08`

Current best artifact:

- `.artifacts/parity/20260507-cpp-jaxcpu-single-stage-dense-newton-v8/fullgraph-m10-trace.json`

Dense-newton-v8 same-candidate replay now passes:

- `max_candidate_abs_diff = 0.0`
- `max_objective_abs_diff = 8.260059303211165e-14`
- `max_optimizer_gradient_abs_diff = 3.107514245925813e-11`
- `max_failure_scalar_abs_diff = 2.220446049250313e-15`
- `max_hardware_metric_abs_diff = 2.2737367544323206e-13`

The full gate still fails:

- `final_iota_abs_diff = 5.814942668005973e-10`
- `final_volume_rel_diff = 1.2478977864730094e-10`
- `field_error_rel_diff = 1.5525982944175097e-07`
- `max_surface_pointwise_abs = 0.0`

Older artifact, now stale as the primary target:

- `.artifacts/single_stage_full_run_rootcause/20260506-exact-replay-gate-maxiter5/cpp_jax_cpu_fullgraph_trace.json`
- first replay failure at pair 9
- `max_objective_abs_diff = 8.335470091935804e-10`
- `max_optimizer_gradient_abs_diff = 7.221488407260779e-09`

Intermediate artifact:

- `.artifacts/single_stage_full_run_rootcause/20260506-fullgraph-newtontol1e13-maxiter5/cpp_jax_cpu_fullgraph_trace.json`
- `max_objective_abs_diff = 9.93205517829665e-13`
- `max_optimizer_gradient_abs_diff = 3.81621512168806e-10`
- first failure became solver-success mismatch, not the old pair-9 gradient
  mismatch

## Corrected Diagnosis

The drift is mainly algorithmic solver-path drift amplified by L-BFGS-B, not a
plain floating-point reduction-order issue.

However, the latest dense-newton-v8 single-stage artifact is not using the
public Boozer exact Newton path as its primary Boozer initialization route. It
records:

- `boozer_optimizer_backend = scipy`
- `boozer_least_squares_algorithm = quasi-newton`
- `boozer_optimizer_method = bfgs`

That means the current P0 target is the Boozer LS route:

- CPU: `BoozerSurface.run_code()` does BFGS, then
  `minimize_boozer_penalty_constraints_newton()`.
- JAX: `BoozerSurfaceJAX.run_code()` does the corresponding JAX LS BFGS route,
  then `minimize_boozer_penalty_constraints_newton()`.

The exact Newton mismatch is real and should remain on a separate validation
lane, but it is not the current dense-newton-v8 root target.

## Relevant Code Paths

CPU LS solve:

- `src/simsopt/geo/boozersurface.py`
- `run_code()` routes LS through BFGS then Newton polish.
- `minimize_boozer_penalty_constraints_LBFGS()` runs SciPy BFGS or L-BFGS-B.
- `minimize_boozer_penalty_constraints_newton()` solves dense Hessian systems
  with `np.linalg.solve`.

JAX LS solve:

- `src/simsopt/geo/boozersurface_jax.py`
- `minimize_boozer_penalty_constraints_LBFGS()` routes through
  `reference_minimize` or `target_minimize`.
- `minimize_boozer_penalty_constraints_newton()` routes through
  `_run_newton_polish_for_method()`.
- For `optimizer_backend="scipy"`, `_run_newton_polish_for_method()` calls
  `newton_polish()` in `src/simsopt/geo/optimizer_jax.py`.

JAX dense Newton polish:

- `src/simsopt/geo/optimizer_jax.py`
- `newton_polish()` can materialize dense Hessian steps via
  `dense_newton_steps=True`.
- `_solve_dense_newton_step()` currently uses host `np.linalg.solve` and
  conditional refinement.

Exact Newton separate lane:

- CPU exact: `BoozerSurface.solve_residual_equation_exactly_newton()`
- JAX exact: `optimizer_jax.newton_exact()`
- This path has GMRES-vs-dense-LU and residual-monotone gate differences, but
  it is not the route shown by dense-newton-v8.

## Non-Goals

- [ ] Do not loosen scalar, gradient, or final-physics tolerances.
- [ ] Do not treat JAX CPU/GPU self-consistency as CPU/C++ parity proof.
- [ ] Do not patch `IotasJAX` just because it is visually suspicious.
- [ ] Do not replace the CPU/reference SciPy lane.
- [ ] Do not make rollback/state changes unless a trace proves stale state
      leakage.
- [ ] Do not make `newton_exact()` LU replacement the first implementation
      patch for the dense-newton-v8 failure.

## Implementation Plan

### Phase 0 - Lock The Evidence Baseline

- [ ] Record the exact current artifact path in the next validation note.
- [ ] Record the current commit hash and dirty-tree status before every run.
- [ ] Treat dense-newton-v8 as the current baseline, not the May 6 pair-9
      artifact.
- [ ] Preserve the May 6 artifacts only as historical regression evidence.
- [ ] Add a short helper summary in the comparator output with:
  - [ ] selected Boozer type (`ls` or `exact`)
  - [ ] Boozer optimizer backend
  - [ ] least-squares algorithm
  - [ ] Newton tolerance and maxiter
  - [ ] materialized dense Hessian/Jacobian flags
  - [ ] linear solve backend

Acceptance gate:

- [ ] Re-run the existing dense-newton-v8-equivalent replay and confirm the
      baseline still reports same-candidate replay pass with max gradient diff
      near `3.1075e-11`.

### Phase 1 - Add LS Solver/Polish Trace Metadata

Files:

- `src/simsopt/geo/boozersurface_jax.py`
- `src/simsopt/geo/boozersurface.py`
- `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py`
- `benchmarks/single_stage_init_parity.py`

Todos:

- [ ] Add missing CPU LS Newton polish metadata to the CPU result dictionary:
  - [ ] `linearization_kind = "hessian"`
  - [ ] `linear_solve_backend = "dense-lu"`
  - [ ] `dense_linear_solve_factors_available`
  - [ ] `dense_newton_steps_materialized = true`
  - [ ] `newton_iter`
  - [ ] final gradient norm
  - [ ] Hessian shape
  - [ ] whether iterative refinement ran on the final accepted Newton step
- [ ] Wire existing JAX LS Newton polish metadata into the single-stage trace
      event instead of re-deriving it:
  - [ ] `linearization_kind`
  - [ ] `linear_solve_backend`
  - [ ] `dense_newton_steps_materialized`
  - [ ] `hessian_materialized`
  - [ ] `dense_hessian_shape`
  - [ ] `dense_hessian_bytes`
- [ ] Add only the JAX fields that are not already present in the solver result:
  - [ ] `newton_iter`
  - [ ] final gradient norm
  - [ ] whether dense refinement ran
- [ ] Thread this metadata into the single-stage objective-evaluation trace.
- [ ] Teach `compare_same_candidate_objective_replay()` to compare metadata
      presence and exact enum fields.
- [ ] Keep numeric metadata comparisons separate from objective/gradient
      comparisons so the first failure remains readable.

Acceptance gate:

- [ ] `compare_same_candidate_objective_replay()` fails if CPU/JAX disagree on
      `linearization_kind`, `linear_solve_backend`,
      `dense_newton_steps_materialized`, or `dense_linear_solve_factors_available`
      at any same-candidate event.
- [ ] Same-candidate replay emits enough metadata to identify whether CPU and
      JAX used the same LS Newton-polish algorithm at each event.

### Phase 2 - Add Per-Term Gradient Slice Replay

Files:

- `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py`
- `benchmarks/single_stage_init_parity.py`
- likely objective wrapper owners under `src/simsopt/geo/surfaceobjectives_jax.py`

Todos:

- [ ] Extend `build_single_stage_objective_evaluation_trace_event()` to record
      per-term objective and gradient summaries.
- [ ] Include at least these slices:
  - [ ] Boozer residual
  - [ ] Iotas / iota penalty
  - [ ] NonQS
  - [ ] coil curvature
  - [ ] curve-curve distance
  - [ ] curve-surface distance
  - [ ] surface terms
  - [ ] any scalar hardware/failure penalty term included in the composite
- [ ] Record each slice in optimizer-coordinate space, not only native wrapper
      coordinates.
- [ ] In `compare_same_candidate_objective_replay()`, diff each slice before
      computing the aggregate failure summary.
- [ ] Report:
  - [ ] max slice objective abs diff
  - [ ] max slice gradient abs diff
  - [ ] owning slice name
  - [ ] pair index and line-search eval
- [ ] Do not patch any wrapper until this data identifies the owner.

Acceptance gate:

- [ ] The replay identifies whether the remaining `3.1e-11` aggregate gradient
      diff belongs to LS Boozer linearization, Iotas, BoozerResidual, NonQS, or
      another slice.

### Phase 3 - Direct LS Hessian/Gradient Oracle Comparison

Files:

- `tests/geo/test_boozersurface_jax.py`
- `tests/geo/test_boozersurface_jax_private.py`
- `tests/geo/boozersurface_jax_test_helpers.py`
- `src/simsopt/geo/boozersurface_jax.py`
- `src/simsopt/geo/optimizer_jax.py`

Todos:

- [ ] Build a fixed-state CPU/JAX fixture for the exact Boozer LS candidate
      states used in dense-newton-v8.
- [ ] Compare CPU `boozer_penalty_constraints_vectorized(..., derivatives=2)`
      against JAX value, gradient, and Hessian at identical state.
- [ ] Use the explicit JAX construction path for the oracle:
  - [ ] build the scalar objective with
        `BoozerSurfaceJAX._make_penalty_objective_with(...)`
  - [ ] get value/gradient with
        `optimizer_jax._cached_jit_value_and_grad(objective_fn)`
  - [ ] build HVPs with `optimizer_jax._hessian_vector_product_fn(objective_fn)`
  - [ ] materialize the dense Hessian with
        `optimizer_jax._materialize_dense_hessian(hvp_fn, x, symmetrize=False)`
  - [ ] compare the dense step through
        `optimizer_jax._solve_dense_newton_step(...)`
- [ ] Compare:
  - [ ] scalar objective
  - [ ] gradient vector
  - [ ] Hessian matrix
  - [ ] dense Newton step `solve(H, grad)`
  - [ ] refined Newton step when the refinement predicate is active
- [ ] At a fixed outer optimizer candidate `x`, run the inner BFGS stage to
      convergence on both lanes and compare the BFGS output state before
      treating Newton polish as the variable under test:
  - [ ] `||x_BFGS_cpu - x_BFGS_jax||_inf`
  - [ ] BFGS objective value
  - [ ] BFGS gradient norm
  - [ ] BFGS iteration count and success flag
- [ ] Verify CPU and JAX use the same optimize-G dimension and coordinate
      ordering.
- [ ] Add a regression test that fails if dense Newton polish silently falls
      back to operator GMRES for this parity lane.

Acceptance gate:

- [ ] Fixed-state LS Hessian/gradient parity is within the named parity ladder
      tolerance before rerunning the outer optimizer.
- [ ] Inner BFGS first-stage output parity is either within `1e-12` in
      infinity norm or explicitly identified as the owner of the remaining
      downstream drift.

### Phase 4 - Align The Proven Solver Layer

Only start this phase after Phases 1-3 identify the owner.

Possible patches:

- [ ] If LS Hessian values differ:
  - [ ] trace the residual scalarization path first
  - [ ] compare Boozer residual, volume label, and z-axis constraint terms
  - [ ] patch the residual/label derivative source, not the optimizer
- [ ] If Hessian values match but dense Newton steps differ:
  - [ ] align host solve dtype and factorization path
  - [ ] make JAX dense solve use the same host LAPACK contract as CPU for the
        CPU-parity lane
  - [ ] align iterative-refinement predicate with CPU LS Newton polish
- [ ] If dense Newton steps match but adjoint gradients differ:
  - [ ] compare runtime `linearization_kind`
  - [ ] compare PLU factor availability
  - [ ] compare `_solve_boozer_adjoint()` dense-vs-operator route
  - [ ] force parity lane to use the same dense factors when available
- [ ] If a wrapper slice owns the diff:
  - [ ] patch only that wrapper
  - [ ] add a fixed-state wrapper-gradient regression

Acceptance gate:

- [ ] Same-candidate replay remains exact-candidate matched and reduces
      `max_optimizer_gradient_abs_diff` to `<= 1e-12` without regressing
      objective or hardware comparisons.
- [ ] Stretch target: if the fixed-state Hessian/gradient oracle shows an
      `O(eps)` floor, reduce `max_optimizer_gradient_abs_diff` to `<= 1e-13`.

### Phase 5 - Optimizer Amplification Diagnostics

Files:

- `benchmarks/single_stage_init_parity.py`
- `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py`
- target/reference optimizer adapters under `src/simsopt/geo/optimizer_jax.py`
  and `src/simsopt/geo/optimizer_jax_reference.py`

Todos:

- [ ] Add accepted-step history comparison:
  - [ ] accepted iteration
  - [ ] candidate vector hash/summary
  - [ ] objective value
  - [ ] projected gradient norm
  - [ ] step length when available
  - [ ] termination message/status
- [ ] Add distance-to-threshold diagnostics for hardware predicates:
  - [ ] max curvature threshold margin
  - [ ] curve-curve distance margin
  - [ ] curve-surface distance margin
  - [ ] surface-vessel distance margin
- [ ] Compare post-optimization field-error reporting at identical final
      optimizer state before attributing `field_error_rel_diff` only to L-BFGS-B
      path amplification:
  - [ ] set both lanes to the same final optimizer `x`
  - [ ] recompute `norm_field_summary(...)`
  - [ ] compare `FIELD_ERROR`
  - [ ] compare the saved/reporting path value against the recomputed value
- [ ] Keep same-candidate replay separate from optimizer-path comparison.
- [ ] Do not infer a bad objective from a different accepted optimizer path
      unless same-candidate replay also fails.

Acceptance gate:

- [ ] The first full-optimizer path split is reported as either:
  - [ ] objective/gradient contract split at same candidate, or
  - [ ] optimizer acceptance split after same-candidate parity holds.

### Phase 6 - Cleanup Amplifiers

Failure penalty:

- [ ] Make `compute_single_stage_failure_penalty()` contribution
      class-exclusive after `reject_class` is selected.
- [ ] Add a test where solver, hardware, and self-intersection signals are all
      present and only the selected class contributes the class-specific score.
- [ ] Keep failure count and step-ratio terms independent if they are intended
      to be class-agnostic.

Discrete replay/hardware:

- [ ] Ensure all discrete replay paths compare `violation_keys` and numeric
      metrics, not formatted violation strings.
- [ ] Add numeric tolerance checks for hardware metrics if a discrete replay
      comparator still uses text.

Exact Newton separate lane:

- [ ] Add a dedicated exact-Boozer CPU/JAX parity test for
      `solve_residual_equation_exactly_newton()`.
- [ ] Compare CPU dense LU + refinement against JAX `newton_exact()`.
- [ ] If exact parity is still needed, implement a CPU-parity dense-LU exact
      Newton mode separately from the LS dense-newton-v8 fix.
- [ ] Treat exact-mode JAX parity as not release-complete until this lane
      passes; the dense-newton-v8 LS fix does not prove `boozer_type="exact"`
      users can rely on CPU/JAX parity.

Acceptance gate:

- [ ] Cleanup changes do not alter dense-newton-v8 same-candidate replay unless
      the change is intentionally part of the measured fix.

## Final Validation Ladder

Run in order:

- [ ] Fixed-state LS value/gradient/Hessian parity tests.
- [ ] Fixed-state wrapper-gradient slice parity tests.
- [ ] Same-candidate objective replay against the current dense-newton-v8
      settings.
- [ ] Full `maxiter=5` CPU/C++ vs JAX CPU parity run.
- [ ] Full `maxiter=10` CPU/C++ vs JAX CPU parity run.
- [ ] If CPU passes, repeat JAX CPU vs JAX GPU on hardware.

Required pass criteria:

- [ ] Same-candidate replay passes.
- [ ] Final iota diff <= configured `final_iota_abs_tol`.
- [ ] Final volume relative diff <= configured `final_volume_rel_tol`.
- [ ] Final field error relative diff <= configured `field_error_rel_tol`.
- [ ] Surface pointwise geometry diff remains zero or within configured
      geometry tolerance.
- [ ] No tolerance loosening was used to obtain the pass.

## Current Priority Order

1. [ ] Add LS solver/polish metadata to traces.
2. [ ] Add per-term gradient slice replay.
3. [ ] Add direct LS Hessian/gradient fixed-state oracle tests.
4. [ ] Add inner BFGS first-stage parity diagnostics.
5. [ ] Patch the proven LS/adjoint/wrapper layer.
6. [ ] Add optimizer amplification and final field-error reporting diagnostics.
7. [ ] Clean up failure penalty class exclusivity.
8. [ ] Validate exact Newton as a separate lane before claiming exact-mode
       CPU/JAX parity.
