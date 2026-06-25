# Banana Coil JAX Port Implementation Plan

## Purpose

This plan defines how to port the custom banana-coil optimization code to local JAX usage without directly porting the mutable SIMSOPT `Optimizable` object graph. It is intended to guide implementation, review, and parity validation for a frozen-spec, pure-kernel JAX path.

## Goals

- Provide a Stage 2 banana objective path where one host optimizer evaluation calls one compiled JAX value-and-gradient kernel.
- Freeze banana coil, current, surface, hardware, and weight state into explicit JAX specs that can be used without live SIMSOPT objects inside compiled kernels.
- Preserve current loader, CLI, artifact, VMEC/WOUT, CAD, plotting, and reporting behavior at the host boundary.
- Add direct parity tests against the existing `CurveCWSFourierCPP`, adapter-object objectives, and Stage 2 target-lane behavior before expanding the port.
- Keep the decision vector as the single source of truth for differentiable state.

## Non-Goals

- Do not port or replace the full SIMSOPT `Optimizable` DAG, recompute flags, dependency propagation, or cache invalidation model.
- Do not make CAD swept-solid hardware contact checks part of a JIT-compiled kernel.
- Do not move JSON/GSON loading, CLI parsing, VMEC/WOUT loading, GLB loading, plotting, or report generation onto device.
- Do not start with a repo-wide Equinox migration or a full replacement of existing JAX specs.
- Do not depend on JAXopt for new work; it is not a maintained target dependency.

## Current Context

- `src/simsopt_jax_adapters/geo/curvecwsfourier.py` currently defines `CurveCWSFourierCPP`, JAX-backed geometry methods, and `to_spec()` for RZ Fourier surfaces.
- `src/simsopt_jax/core/curve_kernels.py` currently provides pure JAX curve helpers including `curve_cws_rz_gamma_from_dofs`, `incremental_arclength_pure`, and `kappa_pure`.
- `examples/single_stage_optimization/banana_opt/jax_banana_drivers.py` currently builds banana optimization flows with host SIMSOPT objects and JAX adapters such as `BiotSavartJAX`, `BoozerSurfaceJAX`, `CurveLengthJAX`, `CurveCurveDistanceJAX`, `CurveSurfaceDistanceJAX`, `LpCurveCurvatureJAX`, and `SquaredFluxJAX`.
- `examples/single_stage_optimization/banana_opt/jax_banana_drivers.py` currently assembles weighted objective terms through `weighted_sum_objective()` and mutates `objective.x` inside SciPy L-BFGS-B callbacks.
- `examples/single_stage_optimization/banana_opt/jax_banana_types.py` currently holds banana constants, hardware thresholds, target widths/distances, and weight dataclasses.
- `src/simsopt_jax_adapters/objectives/stage2_target.py` currently demonstrates the target-lane pattern: structured optimizer state, frozen specs, `final_specs_from_dofs`, `jax.jit(jax.value_and_grad(...))`, and cacheable value-and-gradient callables.
- `src/simsopt_jax/core/specs.py` currently provides the existing frozen dataclass spec registration and serialization pattern for core JAX payloads.
- Existing Stage 2 and Single Stage CLIs already use `--backend` for the field/objective backend choice `cpu|jax`; any frozen-spec selector must not repurpose that flag without updating the existing CLI contract and tests.
- Operational caution: prior local runs have imported `simsopt` from a sibling checkout when the interpreter/path was not pinned. Validation must confirm `simsopt.__file__` before trusting parity results.
- Current review finding: the ambient `python` in this shell is not a valid parity environment. It is Python 3.14.3, does not find JAX, and cannot import this checkout's `simsopt` through `PYTHONPATH=src` because the available `simsoptpp` does not expose `Curve`.

## Rationale

The lowest-risk JAX port is to preserve host orchestration and replace per-term object churn with explicit specs plus pure kernels. Directly porting `Optimizable` would carry mutable DAG semantics, side effects, and cache invalidation into code that JAX wants to see as pure array transformations. The existing Stage 2 target lane already shows the right shape: freeze host objects into specs, keep a structured optimizer state, construct final specs from a decision vector, and expose compiled value-and-gradient functions.

The first implementation should therefore make the banana-specific JAX path narrow and testable: geometry and local penalties first, then pairwise distances, then field/flux/Boozer terms. This keeps each phase small enough to validate against existing behavior before moving more objective terms onto device.

## Assumptions

- The first supported pure CWS path can target `SurfaceRZFourier`; `CurveCWSFourierCPP.to_spec()` currently raises for non-RZ Fourier surfaces.
- Existing JAX specs and local optimizer utilities remain the primary integration layer; Equinox may be considered later only if it reduces boilerplate without breaking serialization or solver contracts.
- Stage 2 is the first production target; single-stage Boozer/iota terms come after Stage 2 geometry and flux parity are stable.
- A host optimizer calling a compiled JAX value-and-gradient function is acceptable as the first local usage milestone.
- The validation environment can be pinned to an interpreter where `simsopt.__file__` resolves inside this checkout.
- Runtime parity tests are blocked until that interpreter/env is selected; static path and document checks are still valid in the ambient shell.

## Implementation Plan

1. Define the banana frozen-spec boundary.
   - [ ] Add `src/simsopt_jax/core/banana.py` with `BananaSystemSpec`, `BananaObjectiveSpec`, and `BananaDecisionSpec` or an equivalent structured state model.
   - [ ] Include the CWS base curve spec, banana symmetry/current map, fixed TF/VF/proxy current metadata, winding/LCFS surface specs, hardware thresholds, objective weights, quadrature sizes, and optional Boozer solve configuration.
   - [ ] Keep differentiable array fields separate from static metadata so `jax.jit` cache keys are stable and intentional.
   - [ ] Add `as_dict` / `from_dict` compatibility only if the spec must cross existing GSON artifact boundaries; otherwise keep serialization host-side in the example layer.
   - [ ] If banana symbols are intended to be public through `simsopt_jax.core`, update `src/simsopt_jax/core/__init__.py` and the import smoke tests; otherwise keep imports explicit from `simsopt_jax.core.banana`.

2. Freeze the existing host banana setup into specs.
   - [ ] Add `examples/single_stage_optimization/banana_opt/jax_banana_specs.py`.
   - [ ] Convert `CurveCWSFourierCPP.to_spec()` output, current values, symmetry rotations, hardware limits, and weight dataclasses into a `BananaObjectiveSpec`.
   - [ ] Provide a host-only builder that accepts the current objects from `jax_banana_drivers.py` and returns `(spec, decision_vector)`.
   - [ ] Confirm the builder does not call JAX kernels from inside CLI parsing or artifact loading paths.

3. Add pure banana geometry kernels.
   - [ ] Implement `banana_geometry_from_dofs(spec, decision_vector)` using the CWS RZ spec and decision-vector curve DOFs.
   - [ ] Return base banana `gamma`, `gammadash`, `gammadashdash`, symmetry-expanded banana curves, and currents as explicit arrays or typed tuples.
   - [ ] Add length calculation with the same normalization as `CurveLengthJAX`: `curve_length_pure(incremental_arclength)` returns the mean incremental arclength, not the unnormalized sum.
   - [ ] Add maximum curvature and p-norm curvature penalties using `kappa_pure`.
   - [ ] Add poloidal extent, projected width, and self-distance kernels matching the current `PoloidalExtentJAX`, `ProjectedEllipseWidthJAX`, and `CurveSelfDistanceJAX` behavior.

4. Add the narrow Stage 2 objective kernel.
   - [ ] Implement `banana_stage2_terms(spec, decision_vector)` returning named raw terms in a stable order.
   - [ ] Implement `banana_stage2_value(spec, decision_vector)` as the weighted scalar objective.
   - [ ] Implement or expose `banana_stage2_value_and_grad(spec, decision_vector)` using `jax.jit(jax.value_and_grad(...))`.
   - [ ] Keep the host-facing wrapper thin: it should pass only the frozen spec and decision vector to compiled JAX code.

5. Connect the new path to `jax_banana_drivers.py`.
   - [ ] Add a typed objective-mode selector that distinguishes current adapter-object mode from frozen-spec kernel mode without changing the existing `--backend cpu|jax` field-backend meaning in Stage 2 or Single Stage CLIs.
   - [ ] For frozen-spec mode, replace `objective.x = dofs; objective.J(); objective.dJ()` inside the host optimizer with a direct call to the compiled value-and-gradient function.
   - [ ] Preserve the existing adapter-object path as the compatibility baseline.
   - [ ] Emit diagnostics showing which backend ran, the resolved `simsopt.__file__`, the decision-vector length, and the raw objective terms.

6. Port pairwise and finite-build terms.
   - [ ] Add coil-coil distance kernels using chunked `jax.lax.scan` or existing pairwise reduction helpers.
   - [ ] Add coil-surface distance kernels with chunked dense-vs-batched self-consistency checks.
   - [ ] Add finite-build pack metrics after geometry parity is stable.
   - [ ] Add hardware keepout proxy-field terms only after the distance kernels have direct parity tests.

7. Port field, flux, and Boozer terms.
   - [ ] Reuse existing grouped Biot-Savart specs and field kernels for Stage 2 flux terms.
   - [ ] Port `SquaredFluxJAX` behavior into the frozen-spec objective path without introducing host callbacks.
   - [ ] Add Boozer/iota/non-quasisymmetry terms for single-stage only after Stage 2 value-and-gradient parity passes.
   - [ ] Keep Boozer solve state explicit; do not hide solver-required state in a mutable object graph.

8. Move the outer optimizer only after the compiled kernel is stable.
   - [ ] Keep the first milestone as SciPy or the existing host optimizer calling one compiled JAX value-and-gradient function per evaluation.
   - [ ] After parity and transfer-guard tests pass, route the same value-and-gradient callable through the repo's on-device optimizer path.
   - [ ] Compare host-optimizer and on-device optimizer trajectories as different solver contracts, not as bit-identical behavior.

## Validation Plan

- [ ] Confirm import provenance before every local parity run:
  ```sh
  PYTHONPATH=src python - <<'PY'
  import simsopt
  print(simsopt.__file__)
  PY
  ```
  The command must print this checkout's `src/simsopt/__init__.py`; a missing import, sibling path, or `simsoptpp` import error blocks runtime parity.
- [ ] Add CWS geometry parity tests against `CurveCWSFourierCPP.gamma()`, `gammadash()`, `gammadashdash()`, and curvature for a fixed RZ surface and fixed banana DOFs.
- [ ] Add `to_spec()` roundtrip or extraction tests proving the frozen spec carries the same CWS dofs, quadrature points, surface metadata, `G`, and `H`.
- [ ] Add a numerical spot check proving the banana length term matches `CurveLengthJAX.J()` and does not accidentally use the raw sum of incremental arclengths.
- [ ] Add Stage 2 objective value parity between current adapter-object mode and frozen-spec kernel mode for a small fixed seed.
- [ ] Add Stage 2 gradient parity against the adapter-object `dJ()` and a finite-difference spot check on selected DOFs.
- [ ] Add static-shape smoke tests for `jax.jit(banana_stage2_value)` and `jax.jit(jax.value_and_grad(banana_stage2_value))`.
- [ ] Run transfer-guard tests around compiled kernels to prove there are no hidden host callbacks or broad host/device transfers.
- [ ] Add chunked-vs-dense consistency tests for self-distance, coil-coil distance, and coil-surface distance kernels.
- [ ] Add a smoke run proving the host optimizer calls one compiled value-and-gradient boundary per evaluation in frozen-spec mode.
- [ ] Add diagnostics artifact checks proving raw terms, weighted terms, decision-vector shape, and backend mode are recorded.
- [ ] Prefer extending existing test surfaces where possible: `tests/core/test_jax_core_specs.py` for spec registration, `tests/integration/test_stage2_jax.py` for Stage 2 target-lane parity, and `tests/test_banana_jax_soft_penalty_drivers.py` for banana soft-penalty driver smoke coverage.

## Risks and Mitigations

- Risk: The port accidentally preserves hidden mutable `Optimizable` state through a closure.
  Mitigation: Require compiled kernels to accept only frozen specs and decision vectors, and add transfer-guard tests that fail on host callbacks.

- Risk: Existing CWS support is narrower than the host object path.
  Mitigation: Start with `SurfaceRZFourier` only and mark non-RZ CWS surfaces out of scope until a direct spec and parity path exists.

- Risk: Pairwise distance kernels become memory-heavy for full hardware configurations.
  Mitigation: Implement chunked `lax.scan` reductions and require chunked-vs-dense parity on small cases before using full cases.

- Risk: Host optimizer and on-device optimizer trajectories differ.
  Mitigation: Validate objective and gradient parity first; treat optimizer trajectory comparison as a solver-contract check, not a bit-identity gate.

- Risk: Local import path contamination invalidates parity results.
  Mitigation: Pin the interpreter/env and record `simsopt.__file__` in every parity artifact.

- Risk: Serialization behavior changes when specs move from host objects to frozen payloads.
  Mitigation: Keep artifact I/O host-side for the first milestone and add `as_dict` / `from_dict` compatibility only where a spec must cross an existing artifact boundary.

## Completion Criteria

- [ ] `src/simsopt_jax/core/banana.py` exists and exposes pure, typed banana geometry and Stage 2 objective functions.
- [ ] `examples/single_stage_optimization/banana_opt/jax_banana_specs.py` exists and freezes the current host setup into specs plus a decision vector.
- [ ] The banana driver path can run the existing adapter-object objective mode and the new frozen-spec objective mode from the same initial seed without changing the existing `--backend cpu|jax` meaning.
- [ ] CWS geometry parity passes against `CurveCWSFourierCPP` for gamma, derivatives, and curvature.
- [ ] Stage 2 objective value and gradient parity pass for a small fixed seed.
- [ ] Compiled kernel smoke tests pass under strict transfer guard.
- [ ] Chunked pairwise distance kernels pass dense-vs-chunked self-consistency tests.
- [ ] The first local optimization milestone produces diagnostics and artifacts showing a host optimizer calling one compiled JAX value-and-gradient kernel per evaluation.

## Open Questions

- Which exact interpreter/env should be the required local validation command for this checkout?
- Should the first frozen-spec objective mode be exposed through a new CLI flag such as `--objective-backend` / `--jax-objective-mode`, or remain an internal driver option until parity is complete?
- Should `BananaSystemSpec` live permanently in `src/simsopt_jax/core/banana.py`, or should banana-specific spec construction remain under `examples/single_stage_optimization/banana_opt/` with only reusable kernels in `src/`?
- Which Stage 2 seed and surface artifact should become the canonical small parity fixture?
- Should Equinox be excluded from this slice entirely, or allowed only for new banana-local typed PyTree modules after the current spec pattern is matched?
