# Boozer BFGS Pre-Newton Contract Alignment Plan - 2026-05-07

Status: implementation plan for the next repair slice after the parity bug
census. This plan is scoped to CPU/C++ vs JAX CPU `scipy-jax-fullgraph`
parity for the inner Boozer LS quasi-Newton/pre-Newton solve.

Current tree inspected with a dirty worktree. This document is additive and
does not classify unrelated modified or untracked files.

## Problem Statement

The May 7 parity bug census moved the remaining failure from a vague
`iota_penalty` gradient symptom to a concrete first divergence:

- Artifact:
  `.artifacts/parity/20260507-cpp-jaxcpu-bug-census-m5/fullgraph-m5-census.json`
- Same-candidate replay status: `pass`
- Same-candidate event count: `23`
- First native-gradient divergence:
  - family: `boozer_solve`
  - layer: `pre_newton_state`
  - pair index: `4`
  - line-search evaluation: `4`
- Largest observed native-gradient layer drift:
  - `boozer_solve.pre_newton_state`: `2.446160944635789e-08`
  - `iota_penalty.adjoint`: `1.2260386483831098e-08`
  - `iota_penalty.weighted_penalty_optimizer_gradient`:
    `3.107847312833201e-11`

Interpretation: the remaining gradient mismatch is downstream of the inner
Boozer pre-Newton solve state unless a narrower test disproves that. The next
repair must align the gradient-producing algorithm at that layer first, then
rerun the census to decide whether PLU/Hessian/adjoint/projection are separate
bugs or only downstream effects.

## Requirements

### Functional Requirements

- Match the CPU `BoozerSurface.run_code()` LS contract for the
  `scipy-jax-fullgraph` CPU-parity lane at the same outer candidate.
- Preserve the trust chain:
  1. Existing SIMSOPT CPU/C++/SciPy behavior is the oracle.
  2. JAX CPU parity matches CPU/C++.
  3. JAX GPU/target lanes match the proven JAX CPU contract where applicable.
- Align the pre-Newton contract before changing iota-adjoint or projection code.
- Compare and lock the following pre-Newton fields at fixed outer candidates:
  - exact backend mode and explicit outer-to-inner Boozer routing path
  - resolved Boozer method
  - SciPy options passed to the inner solve
  - SciPy callback kwarg at the parity boundary
  - initial decision vector layout
  - raw Boozer value/gradient kernel output before SciPy sees it
  - final pre-Newton decision vector
  - objective value
  - optimizer-gradient vector
  - success/status/message/iteration count
- Make the SciPy call contract observable in a durable artifact or spy, including
  method, options, callback, success, status, message, iteration count, function
  evaluation count, and gradient evaluation count. Numeric trace parity alone is
  not sufficient proof of this contract.
- Rerun the existing parity bug census after the first repair and classify
  remaining divergent layers as closed, downstream, or still-open.

### Non-Functional Requirements

- **KISS:** Prefer one CPU-parity adapter path or one method-resolution fix over
  a new solver family.
- **YAGNI:** Do not implement a full CPU-style mutable object graph in JAX.
- **DRY / SSOT:** Keep optimizer method and option resolution in the existing
  resolver surfaces:
  - `benchmarks/single_stage_backend_routing.py`
  - `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py`
  - `src/simsopt/geo/optimizer_jax.py`
  - `src/simsopt/geo/optimizer_jax_reference.py`
  - `src/simsopt/geo/boozersurface_jax.py`
  The runtime single-stage route and benchmark helper route must be tested for
  the same outer-to-inner Boozer backend mapping.
- **SOLID / SRP:** Keep benchmark replay, Boozer solve execution, optimizer
  adapters, and trace summarization as separate responsibilities.
- **Immutable / functional:** Preserve the JAX target architecture:
  immutable runtime specs, traceable kernels, cached bundles, and pure
  value/gradient functions. Mutable state remains limited to compatibility
  wrappers such as `BoozerSurfaceJAX.run_code()`.
- **Memory efficient:** Do not add per-iteration dense trace payloads by
  default. Full vector/Hessian payloads stay behind the explicit objective trace
  and parity benchmark paths.
- **Performant:** Do not route normal GPU/ondevice execution through host SciPy.
  Any host-SciPy parity behavior must be limited to the CPU-parity lane.
- **Fail-fast:** Do not add tolerance inflation, fallback solvers, broad
  try/except recovery, or silent method substitution.

## Current Code Facts

CPU reference path:

- `src/simsopt/geo/boozersurface.py`
- `BoozerSurface.run_code()` routes BoozerLS through
  `minimize_boozer_penalty_constraints_LBFGS()` and then Newton polish.
- CPU BoozerLS defaults `limited_memory=False`, so the first stage is SciPy
  `BFGS`; `limited_memory=True` selects SciPy `L-BFGS-B`.
- CPU first-stage options are:
  - `BFGS`: `{"maxiter": maxiter, "gtol": tol}`
  - `L-BFGS-B`: `{"maxiter": maxiter, "gtol": tol, "maxcor": 200, "ftol": tol}`

JAX compatibility path:

- `src/simsopt/geo/boozersurface_jax.py`
- `BoozerSurfaceJAX.run_code()` has the same high-level shape:
  first-stage quasi-Newton solve, then Newton polish.
- `BoozerSurfaceJAX` has a context-sensitive default: omitted
  `optimizer_backend` resolves to `"ondevice"` when the active simsopt backend
  is JAX, and to `"scipy"` in native CPU/reference backend mode. Do not change
  this global default for this repair.
- The `scipy-jax-fullgraph` single-stage parity path gets the inner host-SciPy
  Boozer route because `single_stage_banana_example.py` explicitly maps the
  outer fullgraph backend to inner `optimizer_backend="scipy"`.
- Once that explicit inner backend is `"scipy"` and `limited_memory=False`, the
  resolved inner method should be `bfgs`.
- The JAX first-stage path dispatches through `reference_minimize()` or
  `target_minimize()` depending on the resolved inner method.
- `src/simsopt/geo/optimizer_jax_reference.py` converts the JAX value/gradient
  callback to the SciPy `minimize(jac=True)` host contract.

Important distinction:

- The outer `scipy-jax-fullgraph` optimizer method is
  `lbfgs-scipy-jax-fullgraph`.
- The inner Boozer LS pre-Newton solver for the parity lane must still be
  checked and matched as its own contract. Do not assume the outer L-BFGS name
  means the inner Boozer solve is also L-BFGS.
- `benchmarks/single_stage_backend_routing.resolve_boozer_optimizer_method()`
  reports `"scipy"` for the inner SciPy backend, not the concrete semantic
  SciPy method. Tests that need `BFGS` vs `L-BFGS-B` must inspect
  `BoozerSurfaceJAX._resolve_optimizer_method()` or the result metadata.

## Non-Goals

- Do not change upstream CPU `BoozerSurface` behavior.
- Do not loosen `PARITY_LADDER_TOLERANCES`.
- Do not patch `IotasJAX`, coil VJP projection, or PLU solve first unless the
  post-repair census proves they remain independently divergent.
- Do not make the JAX target/ondevice lane call host SciPy.
- Do not use the CPU-parity host-SciPy inner Boozer route under CUDA/ondevice
  target execution.
- Do not close the `limited_memory=True` Boozer `L-BFGS-B` parity lane in this
  repair. This slice targets the default CPU-parity Boozer `BFGS` lane. If
  implementation touches shared option construction, it must still avoid
  regressing `L-BFGS-B` option construction and must not let `L-BFGS-B` options
  leak into `BFGS`.
- Do not add broad defensive checks, fallback modes, or runtime recovery.
- Do not rewrite `BoozerSurfaceJAX` into a CPU-style mutable object graph.

## Implementation Plan

### Phase 0 - Lock The Baseline

- [ ] Record current `git rev-parse HEAD` and dirty-tree status in the next
      validation note.
- [ ] Preserve the current bug-census artifact path as the baseline.
- [ ] Add or update a short machine-readable replay summary only if the current
      `parity_bug_census` output is missing required pre-Newton fields.
- [ ] Add durable pre-Newton proof fields to the trace/census artifact, or add a
      strict SciPy adapter spy used by the fixed-state contract test. The required
      fields are:
  - [ ] resolved inner Boozer backend and semantic method
  - [ ] exact SciPy method string
  - [ ] exact SciPy options dictionary after internal keys are stripped
  - [ ] SciPy callback kwarg
  - [ ] success, status, message, nit, nfev, and njev from the SciPy result
- [ ] Do not rely on the current census alone for method/options/status/message
      parity: the numeric decomposition comparator checks state/value/gradient
      layers, but it does not currently prove SciPy options or result
      status/message equality.
- [ ] Confirm the fixed-candidate trace includes native-gradient events only for
      this census.

Acceptance gate:

- [ ] Current code reproduces the census shape:
  first divergence at `boozer_solve.pre_newton_state`, with later iota layers
  still downstream.

### Phase 0.5 - Compare The Raw Inner Kernel Before BFGS

Files:

- `tests/geo/test_boozersurface_jax.py`
- `src/simsopt/geo/boozersurface_jax.py` only if an adapter seam must be
  exposed for the test without changing behavior.

Tasks:

- [ ] At the same inner decision vector, compare CPU
      `BoozerSurface.boozer_penalty_constraints_vectorized(..., derivatives=1)`
      with the JAX objective callable built by
      `BoozerSurfaceJAX._make_penalty_objective_with(...)`.
- [ ] Compare scalar value and optimizer-gradient vector before SciPy BFGS
      receives either callback.
- [ ] Assert dtype and shape explicitly:
  - [ ] CPU decision vector is `np.float64`.
  - [ ] JAX-packed decision vector is materialized through the same host NumPy
        conversion used at the SciPy boundary before dtype or byte-level
        comparisons.
  - [ ] JAX-packed decision vector materializes as float64 at the SciPy host
        boundary.
  - [ ] The byte-level `np.float64` decision vectors match before evaluation.
- [ ] Use existing parity-ladder lanes:
  - [ ] scalar value and solved-state fields: `direct_kernel`
        (`rtol=1e-10`, `atol=1e-12`)
  - [ ] optimizer-gradient vector: `ls_wrapper_gradient`
        (`rtol=1e-10`, `atol=1e-12`)
- [ ] If this kernel comparison fails, patch the first mismatching layer in the
      value/gradient construction before inspecting BFGS method/options.

Acceptance gate:

- [ ] Raw CPU/JAX inner objective value and gradient match at the same decision
      vector under the existing parity-ladder tolerances.
- [ ] If kernel reduction-order drift cannot be closed under those tolerances,
      stop and document the fork explicitly: align the JAX kernel reduction
      order to CPU or request a separate tolerance-contract decision. Do not
      silently continue into method/options work.

### Phase 1 - Add A Narrow Pre-Newton Contract Test

Files:

- `tests/geo/test_boozersurface_jax.py`
- `tests/test_benchmark_helpers.py`
- Possibly `benchmarks/single_stage_init_parity.py`

Tasks:

- [ ] Build a same-state CPU/JAX BoozerLS fixture using
      `_build_upstream_boozer_penalty_case()` for the live CPU/JAX Boozer
      objects, and use `UpstreamBoozerImmutableInputs` only for the immutable
      array/spec snapshot it actually owns.
- [ ] Assert CPU and JAX resolve the same inner method for the parity lane:
      `BFGS` on CPU and `bfgs` on JAX should be treated as the same semantic
      method.
- [ ] Assert live CPU and JAX Boozer objects both use
      `options["limited_memory"] == False` for this parity fixture before method
      and options assertions.
- [ ] Assert the resolved method token, the SciPy method string, and the SciPy
      option dictionary match CPU semantics modulo documented case differences.
- [ ] Assert SciPy options match the CPU contract exactly for the selected
      semantic method.
- [ ] For the BFGS lane, assert no L-BFGS-B-only options (`maxcor`, `ftol`,
      `maxfun`, `maxls`) spill into SciPy BFGS.
- [ ] Suppress progress callbacks in the fixed-state pre-Newton contract test
      and assert the SciPy adapter call receives `callback=None`. Do not treat
      stripping callback-like entries from `options` as sufficient proof.
- [ ] Assert result status/message/iteration fields are either identical or
      mapped through a documented semantic contract. Do not infer this from the
      numeric census.
- [ ] Assert the trace/census proof fields or SciPy adapter spy capture the same
      method/options/callback/status/message/nit/nfev/njev values used by the
      fixed-state comparison.
- [ ] Assert initial decision-vector layout matches:
      surface DOFs, iota, then optional `G`.
- [ ] Assert pre-Newton output state/value/gradient parity at a fixed state,
      before Newton polish and before iota-adjoint projection.
- [ ] Keep this test independent from outer optimizer trace drift so failures
      point directly at the inner Boozer contract.

Acceptance gate:

- [ ] The new test fails on the current pre-Newton drift or proves which lower
      sub-layer is drifting.
- [ ] The test uses the exact fullgraph parity route: outer
      `scipy-jax-fullgraph` maps to inner Boozer `optimizer_backend="scipy"`;
      direct JAX-backend constructor defaults remain `ondevice`.
- [ ] The runtime single-stage route in `single_stage_banana_example.py` and the
      benchmark route in `benchmarks/single_stage_backend_routing.py` resolve the
      same inner Boozer backend for `scipy-jax-fullgraph`.
- [ ] Existing Boozer public API tests still pass.

### Phase 2 - Align Method And Option Resolution

Files:

- `src/simsopt/geo/boozersurface_jax.py`
- `src/simsopt/geo/optimizer_jax_reference.py`
- `benchmarks/single_stage_backend_routing.py`
- `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py`
- `src/simsopt/geo/optimizer_jax.py` only if the fixed-state route test proves
  the outer optimizer resolver is incorrectly threading the inner Boozer route.

Tasks:

- [ ] If the test shows a method mismatch, fix the resolver so the
      `scipy-jax-fullgraph` Boozer parity lane resolves the same inner
      quasi-Newton method as CPU.
- [ ] If the test shows an option mismatch, mirror CPU's exact SciPy option
      dictionary from a JAX-side helper and lock it with a contract test. Do not
      move CPU option construction into shared code in this slice.
- [ ] Make option stripping method-aware: `BFGS` must not receive `L-BFGS-B`-only
      keys (`maxcor`, `ftol`, `maxfun`, `maxls`), and callback-like internal keys
      must not enter the SciPy `options` dictionary.
- [ ] Add a small same-state unit test for option stripping with an input dict
      containing `callback`, `progress_callback`, `failure_callback`, `maxcor`,
      `ftol`, `maxfun`, and `maxls`; for semantic method `BFGS`, the resulting
      SciPy options must contain none of those keys.
- [ ] If a shared option helper is modified in a way that affects
      `limited_memory=True`, preserve CPU `L-BFGS-B` option semantics including
      `ftol=tol`, or stop and split that repair into a separate limited-memory
      parity slice.
- [ ] Prefer fixing the single-stage outer-to-inner route or
      `BoozerSurfaceJAX._resolve_optimizer_method()` over editing
      `optimizer_jax.py` outer optimizer resolution.
- [ ] Keep method names lane-specific at public boundaries, but compare through
      a semantic contract where CPU `BFGS` equals JAX `bfgs`.
- [ ] Do not alter ondevice method resolution except for tests proving it remains
      unchanged.

Acceptance gate:

- [ ] The pre-Newton method/options test passes.
- [ ] Existing routing tests still prove:
  - CPU/reference routes stay CPU/reference.
  - JAX ondevice routes stay ondevice.
  - `scipy-jax-fullgraph` outer optimizer remains fullgraph SciPy-control.
- [ ] A concrete ondevice regression test still proves default JAX backend
      `BoozerSurfaceJAX` resolves to the expected `*-ondevice` method and never
      routes target/GPU execution through host SciPy.

### Phase 3 - Align Host Adapter Conversion

Files:

- `src/simsopt/geo/boozersurface_jax.py`
- `src/simsopt/geo/optimizer_jax_reference.py`

Tasks:

- [ ] After raw kernel parity passes, compare the raw CPU objective callback
      output with the JAX host-adapted callback at the same inner decision
      vector.
- [ ] Verify scalar dtype, gradient dtype, shape, and flattening order.
- [ ] Patch only the first mismatching adapter layer:
  - gradient conversion
  - scalar conversion
  - host materialization to `np.float64`
  - decision-vector packing/unpacking
  - iota and optional `G` slice positions
- [ ] Keep conversions explicit and local to the host SciPy adapter boundary.
- [ ] Do not change residual definitions, signs, weights, or kernel reduction
      order in Phase 3. Those belong to Phase 0.5 raw-kernel parity.

Acceptance gate:

- [ ] CPU and JAX first-stage callbacks match before SciPy sees them within the
      existing parity-ladder tolerances for this lane.
- [ ] Fixed-state pre-Newton state/value/gradient and
      method/options/status/message contracts match. "Improves" is not
      sufficient for this parity lane.

### Phase 3.5 - Document Newton Polish Trigger Contract

Files:

- `src/simsopt/geo/boozersurface.py`
- `src/simsopt/geo/optimizer_jax.py`
- `src/simsopt/geo/boozersurface_jax.py`

Tasks:

- [ ] Before rerunning the census, read CPU
      `minimize_boozer_penalty_constraints_newton()` and JAX `newton_polish()`.
- [ ] Document whether both paths use the same Newton step, stopping criterion,
      dense solve, and dense-refinement trigger (`norm < 1e-9` on CPU).
- [ ] If they differ, do not bundle a Newton-polish patch into the BFGS
      pre-Newton patch. Record the expected next first-divergence layer for the
      census.

Acceptance gate:

- [ ] The next census result can be interpreted unambiguously as either
      pre-Newton closure or a separate Newton/PLU/Hessian divergence.

### Phase 4 - Rerun The Bug Census

Command shape:

```bash
.conda/jax-0.9.2/bin/python benchmarks/single_stage_init_parity.py \
  --platform cpu \
  --optimizer-backend scipy-jax-fullgraph \
  --maxiter 5 \
  --reference-optimizer-method lbfgs \
  --record-objective-evaluation-trace \
  --case-artifacts-dir .artifacts/parity/20260507-bfgs-prenewton-fix-m5/cases \
  --output-json .artifacts/parity/20260507-bfgs-prenewton-fix-m5/result.json
```

Acceptance gate:

- [ ] Use the same `--maxiter 5` baseline as the May 7 bug-census artifact, and
      record the exact command with the result.
- [ ] `parity_bug_census` no longer reports
      `boozer_solve.pre_newton_state` as the first divergence.
- [ ] `parity_bug_census` reports no divergent `boozer_solve.pre_newton_*`
      layer above the existing diagnostic tolerance.
- [ ] Persist a structured diff against
      `.artifacts/parity/20260507-cpp-jaxcpu-bug-census-m5/fullgraph-m5-census.json`
      classifying each prior divergent layer as closed, downstream, or
      still-open.
- [ ] Persist the SciPy call-contract proof with the census rerun. If the census
      schema does not store method/options/callback/status/message/nit/nfev/njev
      directly, attach the fixed-state adapter-spy artifact path and the captured
      field values to the validation note.
- [ ] Use objective census-diff rules:
  - [ ] `closed`: the same layer is below the original diagnostic tolerance after
        the pre-Newton patch.
  - [ ] `downstream`: a later layer drops below the same diagnostic tolerance
        without any layer-specific patch once the upstream pre-Newton layer is
        closed.
  - [ ] `still-open`: the layer remains above the same diagnostic tolerance after
        the upstream pre-Newton layer is closed, or becomes the first remaining
        divergence.
- [ ] If full parity passes, stop and document the closure.
- [ ] If full parity still fails, classify remaining divergent layers:
  - `boozer_solve.linear_solve_factors`
  - `boozer_solve.final_hessian`
  - `iota_penalty.adjoint`
  - `iota_penalty.optimizer_projection_gradient`
  - `iota_penalty.weighted_penalty_optimizer_gradient`

### Phase 5 - Only Then Patch Remaining Layers

- [ ] If PLU/Hessian remains the first divergence, align dense Hessian and LU
      factor materialization in the CPU-parity lane.
- [ ] If adjoint remains the first divergence with matched PLU/Hessian, align
      transpose solve semantics.
- [ ] If projection remains the first divergence with matched adjoint, align
      coil VJP projection and optimizer-coordinate mapping.
- [ ] Each patch gets one fixed-state test plus one census rerun.

## Validation Plan

Run focused checks after the implementation patch:

```bash
.conda/jax-0.9.2/bin/python -m pytest tests/geo/test_boozersurface_jax.py -q
.conda/jax-0.9.2/bin/python -m pytest tests/test_benchmark_helpers.py -q
.conda/jax-0.9.2/bin/python -m pytest tests/geo/test_single_stage_example.py -q
python -m ruff check src/simsopt/geo/boozersurface_jax.py src/simsopt/geo/optimizer_jax.py src/simsopt/geo/optimizer_jax_reference.py benchmarks/single_stage_init_parity.py tests/geo/test_boozersurface_jax.py tests/test_benchmark_helpers.py
git diff --check
```

Then run the real maxiter-5 census command from Phase 4.

## Stop Conditions

- Stop after the pre-Newton layer is proven closed and the census identifies a
  different first divergence. Do not bundle a PLU/adjoint/projection rewrite
  into the same patch.
- Stop if the fixed-state test proves CPU and JAX pre-Newton contracts already
  match and the census artifact was stale. In that case, rerun the census and
  update this plan before editing solver code.
- Stop if the only remaining differences are below the existing parity ladder
  tolerances and final physics parity passes.
