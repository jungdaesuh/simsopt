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
- `src/simsopt_jax_adapters/objectives/stage2_target.py` currently demonstrates the target-lane pattern: structured optimizer state, frozen specs, `final_specs_from_dofs`, `jax.jit(jax.value_and_grad(...))`, and cacheable value-and-gradient callables. It is not just a generic pattern: it already contains a **banana-specific** frozen-spec value-and-grad lane (`banana_curve_spec`, `banana_coil_specs`, `banana_curve_map`/`banana_current_map`, `banana_coil_dof_specs`, and `_banana_symmetry_runtime_inputs_from_coils` at `stage2_target.py:486`), exposed through `build_stage2_target_objective(...)`. This is shipped, CLI-wired code; the new `banana.py` must reuse or extend it rather than re-implement it (SSOT).
- `examples/single_stage_optimization/STAGE_2/banana_coil_solver.py` is the production Stage 2 banana CLI (argparse + `srun` entry in `banana-scan.sh`). It already drives the frozen-spec target lane: it gates on `should_build_stage2_target_objective(...)`, calls `build_stage2_target_objective(...)`, and consumes `target_objective_bundle.final_specs_from_dofs(...)`. By contrast `jax_banana_drivers.py` is an adapter-object module with **no** CLI. The plan's integration boundary (step 5) must say which of these the frozen-spec objective mode wires into.
- The single-stage example already has a frozen-spec seed-spec flow: `--jax-runtime-seed-spec` / `--compile-jax-runtime-seed-spec` (exercised end-to-end in `tests/test_banana_jax_soft_penalty_drivers.py`). A new frozen-spec objective mode must coexist with this flow, not duplicate it.
- `src/simsopt_jax/core/specs.py` currently provides the existing frozen dataclass spec registration and serialization pattern for core JAX payloads via the `@_register_jax_spec(data_fields=..., meta_fields=...)` decorator (`specs.py:167`), which freezes the dataclass, registers its JAX pytree partition, and auto-attaches `as_dict`/`from_dict`.
- Existing Stage 2 and Single Stage CLIs already use `--backend` for the field/objective backend choice `cpu|jax`; any frozen-spec selector must not repurpose that flag without updating the existing CLI contract and tests.
- Both CLIs also already ship a separate `--optimizer-backend` selector (Stage 2: `scipy`/`ondevice`/`scipy-jax`/`scipy-jax-fullgraph`/`optax-lbfgs`/`optimistix-lbfgs`; single-stage adds `host-jax`) whose help text already distinguishes "evaluate the mutable `Optimizable` graph" from "evaluate the JAX target-lane value/grad". A new objective-mode selector must reconcile against this precedent rather than add a third overlapping backend flag.
- Operational caution: prior local runs have imported `simsopt` from a sibling checkout when the interpreter/path was not pinned. Validation must confirm `simsopt.__file__` before trusting parity results.
- Current review finding: the ambient `python3` in this shell is Python 3.13.12 and can import both JAX and a `Curve`-exposing `simsoptpp`, but `PYTHONPATH=src python3 -c 'import simsopt'` still resolves `simsopt` to the sibling `/Users/suhjungdae/code/columbia/simsopt-surrogate/src` through the base-env editable redirect. Runtime parity must therefore use the explicit `python3.13 -S` command in the validation plan, which resolves `simsopt.__file__` inside this checkout and keeps `simsoptpp.Curve` available.

## Rationale

The lowest-risk JAX port is to preserve host orchestration and replace per-term object churn with explicit specs plus pure kernels. Directly porting `Optimizable` would carry mutable DAG semantics, side effects, and cache invalidation into code that JAX wants to see as pure array transformations. The existing Stage 2 target lane already shows the right shape: freeze host objects into specs, keep a structured optimizer state, construct final specs from a decision vector, and expose compiled value-and-gradient functions.

The first implementation should therefore make the banana-specific JAX path narrow and testable: geometry and local penalties first, then pairwise distances, then field/flux/Boozer terms. This keeps each phase small enough to validate against existing behavior before moving more objective terms onto device.

## Assumptions

- The first supported pure CWS path can target `SurfaceRZFourier`; `CurveCWSFourierCPP.to_spec()` currently raises for non-RZ Fourier surfaces.
- Existing JAX specs and local optimizer utilities remain the primary integration layer; Equinox may be considered later only if it reduces boilerplate without breaking serialization or solver contracts.
- Stage 2 is the first production target; single-stage Boozer/iota terms come after Stage 2 geometry and flux parity are stable.
- A host optimizer calling a compiled JAX value-and-gradient function is acceptable as the first local usage milestone.
- The validation environment can be pinned to an interpreter where `simsopt.__file__` resolves inside this checkout.
- Runtime parity tests are trusted only under the pinned `python3.13 -S` validation command; static path and document checks are still valid in the ambient shell.

## Implementation Plan

1. Define the banana frozen-spec boundary.
   - [x] First decide reuse-vs-reimplement: the `stage2_target.py` banana lane already defines `banana_curve_spec`, `banana_coil_specs`, `banana_curve_map`/`banana_current_map`, and `_banana_symmetry_runtime_inputs_from_coils`. Prefer extending/generalizing that machinery; do not duplicate it. Add a parity check between any new `banana.py` kernels and the existing `stage2_target` banana value/grad before new code is written.
   - [x] Add `src/simsopt_jax/core/banana.py` with `BananaSystemSpec`, `BananaObjectiveSpec`, and `BananaDecisionSpec` or an equivalent structured state model.
   - [x] Include the CWS base curve spec, banana symmetry/current map, fixed TF/VF/proxy current metadata, winding/LCFS surface specs, hardware thresholds, quadrature sizes, and optional Boozer solve configuration. `BananaSystemSpec` now carries the CWS curve, symmetry/current partition, fixed TF current state, fixed auxiliary coil set plus TF/proxy/VF role labels, and disabled-by-default explicit Boozer solve metadata; `BananaObjectiveSpec` carries LCFS/flux samples plus static quadrature counts. Focused tests assert auxiliary roles, Boozer config, quadrature metadata, GSON roundtrip, and canonical Biot-Savart freezer role population.
   - [x] Record the **stage-specific** objective weight set, not "objective weights" as one bundle: `Stage2Weights` and `SingleStageWeights` carry materially different values for the *same* geometry terms (`length` 2e-3 vs 5e-2, `ccdist` 1e6 vs 1e4, `width` 1e1 vs 1e2, `selfint` 1e1 vs 1e2), so the spec must say which stage's weights apply.
   - [x] Note that the hardware thresholds are not just stored — they are consumed as `QuadraticPenalty` thresholds (min/max/two-sided modes), so the spec must carry both the threshold value and the violation mode per term.
   - [x] Register specs with the repo's SSOT mechanism `@_register_jax_spec(data_fields=..., meta_fields=...)` (`specs.py:167`): put differentiable arrays in `data_fields` and static metadata in `meta_fields` so `jax.jit` cache keys are stable (data-field updates must not recompile; meta-field updates must). Getting this partition wrong breaks pytree flattening / cache keys.
   - [x] The `_register_jax_spec` decorator already auto-attaches `as_dict`/`from_dict`, so the "only if needed" caveat does not apply on that path. Only the direct `jax.tree_util.register_dataclass` pattern (e.g. `analytic_pure_fields.py`) skips serialization; choose explicitly which of the two patterns the banana spec uses.
   - [x] Banana symbols are intentionally kept private to explicit `simsopt_jax.core.banana` imports for this slice. Because they are not public through `simsopt_jax.core`, the fail-closed export contract does not need banana entries; `tests/test_jax_import_smoke.py::test_jax_core_public_contract` covers that `import simsopt_jax.core` and the frozen public export map still succeed.

2. Freeze the existing host banana setup into specs.
   - [x] Add `examples/single_stage_optimization/banana_opt/jax_banana_specs.py`.
   - [x] Convert `CurveCWSFourierCPP.to_spec()` output, current values, symmetry rotations, hardware limits, and weight dataclasses into a `BananaObjectiveSpec`. Reuse the existing frozen-spec primitives rather than re-deriving them: `make_curve_cwsfourier_rz_spec`/`CurveCWSFourierRZSpec` for the CWS curve, and `make_coil_symmetry_spec`/`apply_coil_symmetry` (`CoilSymmetrySpec`) as the frozen-spec equivalent of the host `coils_via_symmetries`. `stage2_target.py:486` (`_banana_symmetry_runtime_inputs_from_coils`) is the reference implementation.
   - [x] Provide a host-only builder that accepts the current objects from `jax_banana_drivers.py` and returns `(spec, decision_vector)`.
   - [x] The `decision_vector` is **not** curve DOFs only. When `*_fix_current=False`, `_scaled_current(...)` leaves the `ScaledCurrent` base `Current(1.0)` DOF free, so it becomes part of `objective.x` and the host optimizer mutates it each evaluation. `BananaDecisionSpec` now defines the proven Stage 2 local-objective partition for TF current, banana current, and curve DOFs (with the `KA_TO_A` scaling); VF/proxy current metadata remains part of the broader frozen-system item above and field/flux terms below.
   - [x] Confirm the builder does not call JAX kernels from inside CLI parsing or artifact loading paths.

3. Add pure banana geometry kernels.
   - [x] Implement `banana_geometry_from_dofs(spec, decision_vector)` using the CWS RZ spec and decision-vector curve DOFs.
   - [x] Return base banana `gamma`, `gammadash`, `gammadashdash`, symmetry-expanded banana curves, and currents as explicit arrays or typed tuples.
   - [x] Add length calculation with the same normalization as `CurveLengthJAX`: `curve_length_pure(incremental_arclength)` returns the mean incremental arclength, not the unnormalized sum. Note `curve_length_pure` already exists at `src/simsopt_jax_adapters/geo/curve_objectives.py:49` (it is **not** in `curve_kernels.py`, which has `incremental_arclength_pure`/`kappa_pure` only). Import that one; do not redefine it or import the legacy non-JAX twin `simsopt.geo.curveobjectives.curve_length_pure`.
   - [x] Add maximum curvature and p-norm curvature penalties using `kappa_pure`.
   - [x] Add poloidal extent, projected width, and self-distance kernels matching the current `PoloidalExtentJAX`, `ProjectedEllipseWidthJAX`, and `CurveSelfDistanceJAX` behavior.
   - [x] Port the `QuadraticPenalty` hinge as a pure per-term helper, not just the raw kernels: `0.5*max(J - t, 0)^2` ("max"), `0.5*min(J - t, 0)^2` ("min"), `0.5*(J - t)^2` (`"identity"` / two-sided). The Stage 2 local objective uses this helper for `length_max`/`length_min`, `width_max`/`width_min`, and current penalties; the helper also accepts an explicit `"two-sided"` alias for the future single-stage `iota` term. The actual Boozer/iota/non-QS objective terms remain tracked separately below.
   - [x] Port the coil-current magnitude penalties: `tf_current_max` and `banana_current_max` are `QuadraticPenalty(CurrentMagnitude(current), threshold_A, "max")` where `CurrentMagnitude.J() = abs(current.get_value())` with a sign-aware VJP (`current.vjp(sign)`), gated by `include_current_penalties`. These are differentiable objective terms over the current DOF, distinct from the *frozen* current metadata in step 1; they must be ported with the abs-then-hinge math and the sign-aware gradient.

4. Add the narrow Stage 2 objective kernel.
   - [x] Implement `banana_stage2_terms(spec, decision_vector)` returning named terms in a stable order. "Term" here is the per-`WeightedTerm` contribution *after* its `QuadraticPenalty` hinge (pre-weight), matching the live driver's term set (`length_max`/`length_min`, `width_max`/`width_min`, `poloidal`, curvature, `selfint`, `ccdist`, `csdist`, `tf_current_max`/`banana_current_max`, `sqflux`), so that the weighted sum reproduces `objective.J()`. Mirror the flag gates (`include_min_length`, `include_width`, `include_current_penalties`) so a config that disables a term is reproducible.
   - [x] Implement `banana_stage2_value(spec, decision_vector)` as the weighted scalar objective.
   - [x] Implement or expose `banana_stage2_value_and_grad(spec, decision_vector)` using `jax.jit(jax.value_and_grad(...))`.
   - [x] Keep the host-facing wrapper thin: it should pass only the frozen spec and decision vector to compiled JAX code.

5. Connect the new path to `jax_banana_drivers.py`.
   - [x] Fix the integration boundary first. `jax_banana_drivers.py` is an adapter-object module with no CLI; the production Stage 2 banana CLI is `examples/single_stage_optimization/STAGE_2/banana_coil_solver.py`, which drives the frozen-spec target lane via `build_stage2_target_objective`/`final_specs_from_dofs`. The first milestone is wired into the production solver through the Stage 2 target-objective lane; the adapter driver remains the compatibility/parity baseline.
   - [x] Add a typed objective-mode selector that distinguishes current adapter-object mode from frozen-spec kernel mode without changing the existing `--backend cpu|jax` field-backend meaning in Stage 2 or Single Stage CLIs. The resolved selector is the existing `--optimizer-backend` / `TargetOptimizerContract` path (`scipy` remains the reference lane; `ondevice`/`scipy-jax`/`scipy-jax-fullgraph`/`optax-lbfgs`/`optimistix-lbfgs` build the target lane), not a third overlapping backend flag.
   - [x] For frozen-spec mode, replace `objective.x = dofs; objective.J(); objective.dJ()` inside the host optimizer with a direct call to the compiled value-and-gradient function.
   - [x] Preserve the existing adapter-object path as the compatibility baseline.
   - [x] Emit diagnostics showing which backend ran, the resolved `simsopt.__file__`, the decision-vector length, and the raw objective terms.

6. Port pairwise and finite-build terms.
   - [x] Add coil-coil distance kernels using chunked `jax.lax.scan` or existing pairwise reduction helpers.
   - [x] Add coil-surface distance kernels with chunked dense-vs-batched self-consistency checks.
   - [x] Add finite-build pack metrics after geometry parity is stable. `src/simsopt_jax/core/banana.py` now has pure array kernels for the SIMSOPT-scale pack rotation/fold hinge, torsional strain/twist-strain hinge, projected finite-build reach, and rotation-aware curvature-excess metric; `tests/core/test_banana.py` covers closed-form values plus JIT/value-and-gradient smoke.
   - [x] Add hardware keepout proxy-field terms only after the distance kernels have direct parity tests. The core slice ports the point-cloud steering proxy as pure JAX array kernels: swept Type-KK channel surface sampling, one-curve point-cloud keepout, and symmetry-expanded/chunked point-cloud keepout. Host-side JSON/GLB loading, SDF manifest handling, and the direct swept-solid `hardware_contact_report` oracle remain outside compiled kernels.

7. Port field, flux, and Boozer terms.
   - [x] Reuse existing grouped Biot-Savart specs and field kernels for Stage 2 flux terms.
   - [x] Port `SquaredFluxJAX` behavior into the frozen-spec objective path without introducing host callbacks.
   - [x] Add Boozer/iota/non-quasisymmetry terms for single-stage only after Stage 2 value-and-gradient parity passes. The banana-local facade `build_banana_single_stage_solved_state_target(...)` reuses the existing full single-stage traceable objective lane for non-QS, Boozer residual, iota, length, distance, vessel, and curvature terms instead of reimplementing the solver math.
   - [x] Keep Boozer solve state explicit; do not hide solver-required state in a mutable object graph. `BananaSingleStageSolvedStateTarget` carries `coil_dofs`, `solved_x`, and `linear_solve_factors`, and `test_banana_solved_state_target_matches_single_stage_terms` monkeypatches `booz_jax.run_code` after construction to reject hidden resolves while matching the object-backed single-stage `JF` value.

8. Move the outer optimizer only after the compiled kernel is stable.
   - [x] Keep the first milestone as SciPy or the existing host optimizer calling one compiled JAX value-and-gradient function per evaluation.
   - [x] After parity and transfer-guard tests pass, route the same value-and-gradient callable through the repo's on-device optimizer path.
   - [x] Compare host-optimizer and on-device optimizer trajectories as different solver contracts, not as bit-identical behavior.

## Validation Plan

- [x] Confirm import provenance before every local parity run:
  ```sh
  PYTHONPATH=/Users/suhjungdae/code/columbia/simopt-jax-clean-local/src:/Users/suhjungdae/code/columbia/simsopt-jax/.miniforge/lib/python3.13/site-packages \
    /Users/suhjungdae/code/columbia/simsopt-jax/.miniforge/bin/python3.13 -S - <<'PY'
  import simsopt
  print(simsopt.__file__)
  PY
  ```
  The command must print this checkout's `src/simsopt/__init__.py`; a missing import, sibling path, or `simsoptpp` import error blocks runtime parity.
- [x] Add CWS geometry parity tests against `CurveCWSFourierCPP.gamma()`, `gammadash()`, `gammadashdash()`, and curvature for a fixed RZ surface and fixed banana DOFs.
- [x] Add `to_spec()` roundtrip or extraction tests proving the frozen spec carries the same CWS dofs, quadrature points, surface metadata, `G`, and `H`.
- [x] Add a numerical spot check proving the banana length term matches `CurveLengthJAX.J()` and does not accidentally use the raw sum of incremental arclengths.
- [x] Add Stage 2 objective value parity between current adapter-object mode and frozen-spec kernel mode for a small fixed seed.
- [x] Add Stage 2 gradient parity against the adapter-object `dJ()` and a finite-difference spot check on selected DOFs.
- [x] Add **per-term** parity, not only aggregate value/gradient: compare each `QuadraticPenalty`-wrapped term contribution (`length_max`/`length_min`, `width_max`/`width_min`, `tf_current_max`/`banana_current_max`, etc.) against its adapter `WeightedTerm`, so a dropped or mis-hinged term is named, not just caught as an aggregate mismatch.
- [x] Run the parity at both stages' weight vectors (`Stage2Weights` and `SingleStageWeights`) so the shared geometry kernel is validated under both, since the geometry-term weights differ by stage.
- [x] Add a gradient-parity case with `*_fix_current=False` so the free TF/banana current DOFs are exercised; the existing soft-penalty smoke fixes all currents, so the free-current path is otherwise untested.
- [x] Add a parity check between the new `banana.py` value/grad and the existing `stage2_target.py` banana lane (`build_stage2_target_objective`) to enforce SSOT before the new kernels are trusted.
- [x] Mirror `tests/core/test_jax_core_specs.py:338/374`: assert the banana spec's `data_fields` do not trigger `jax.jit` recompilation while `meta_fields` do (correct data/meta partition).
- [x] Keep banana symbols private to `simsopt_jax.core.banana` and assert `import simsopt_jax.core` still succeeds through the frozen public export contract.
- [x] Add static-shape smoke tests for `jax.jit(banana_stage2_value)` and `jax.jit(jax.value_and_grad(banana_stage2_value))`.
- [x] Run transfer-guard tests around compiled kernels to prove there are no hidden host callbacks or broad host/device transfers.
- [x] Add chunked-vs-dense consistency tests for self-distance, coil-coil distance, and coil-surface distance kernels.
- [x] Add a smoke run proving the host optimizer calls one compiled value-and-gradient boundary per evaluation in frozen-spec mode.
- [x] Add diagnostics artifact checks proving raw terms, weighted terms, decision-vector shape, and backend mode are recorded.
- [x] Prefer extending existing test surfaces where possible: `tests/core/test_jax_core_specs.py` for spec registration, `tests/integration/test_stage2_jax.py` / `tests/integration/test_stage2_target_banana_core.py` for Stage 2 target-lane parity and diagnostics, and new focused banana tests only where no existing surface owned the core local kernel contract.

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

- Risk: The frozen-spec kernel reproduces only the raw geometry kernels and silently drops the `QuadraticPenalty` hinge, the flag-gated terms (`length_min`, `width_min`, current penalties), or the free-current DOFs.
  Mitigation: Port the per-term hinge and full `WeightedTerm` set, mirror the `include_*` flags, and require per-term parity (not just aggregate) plus a `fix_current=False` gradient case.

- Risk: `banana.py` duplicates the banana frozen-spec value-and-grad machinery that already exists in `stage2_target.py`, violating SSOT.
  Mitigation: Make the reuse-vs-reimplement decision up front and add a parity check against `build_stage2_target_objective` before trusting any new kernels.

## Completion Criteria

- [x] `src/simsopt_jax/core/banana.py` exists and exposes pure, typed banana geometry and Stage 2 objective functions.
- [x] `examples/single_stage_optimization/banana_opt/jax_banana_specs.py` exists and freezes the current host setup into specs plus a decision vector.
- [x] The banana driver path can run the existing adapter-object objective mode and the new frozen-spec objective mode from the same initial seed without changing the existing `--backend cpu|jax` meaning.
- [x] CWS geometry parity passes against `CurveCWSFourierCPP` for gamma, derivatives, and curvature.
- [x] Stage 2 objective value and gradient parity pass for a small fixed seed, including per-term parity for the full live `WeightedTerm` set (`QuadraticPenalty`-wrapped `length_min`/`width_min`/current penalties included) under both `Stage2Weights` and `SingleStageWeights`. `tests/test_banana_jax_specs.py::test_banana_stage2_value_and_grad_matches_driver_full_stage2_terms` now compares the complete Stage 2 live term set for per-term values, per-term full-vector gradients, aggregate value, aggregate gradient, and finite-difference spot checks under `Stage2Weights` and a Stage2-shaped copy of `SingleStageWeights` geometry/current weights. Full single-stage Boozer/iota/non-QS parity remains tracked by the gated implementation rows above.
- [x] Compiled kernel smoke tests pass under strict transfer guard.
- [x] Chunked pairwise distance kernels pass dense-vs-chunked self-consistency tests.
- [x] The first local target-lane smoke produces diagnostics and artifacts showing the production Stage 2 path routes through the compiled JAX value-and-gradient lane. Verified by a pinned one-iteration Stage 2 smoke run with `--backend jax --optimizer-backend scipy-jax --skip-postprocess --comparison-results-json examples/single_stage_optimization/STAGE_2/outputs-wout_nfp22ginsburg_000_014417_iota15.nc/first_local_optimization_scipy_jax_comparison.json`; the comparison payload records `OBJECTIVE_DIAGNOSTICS.backend=jax`, `optimizer_backend=scipy-jax`, `objective_source=target-objective`, this checkout's `simsopt_file`, decision-vector shape/count, raw terms, weighted terms, and term gradient norms. That smoke is not an optimizer-success certificate: the checked artifact remains `accepted_artifact=false` if the one-step solver exits abnormally or returns a non-finite final gradient.

## Open Questions

- Which exact interpreter/env should be the required local validation command for this checkout? Current answer: use `/Users/suhjungdae/code/columbia/simsopt-jax/.miniforge/bin/python3.13 -S` with `PYTHONPATH=/Users/suhjungdae/code/columbia/simopt-jax-clean-local/src:/Users/suhjungdae/code/columbia/simsopt-jax/.miniforge/lib/python3.13/site-packages`. Without `-S`, the env-site editable redirect imports `simsopt` from `/Users/suhjungdae/code/columbia/simsopt-surrogate/src`, even when this checkout's `src` is first on `PYTHONPATH`. With `-S`, `simsopt.__file__` resolves to this checkout and `simsoptpp.Curve` is present.
- Should the first frozen-spec objective mode be exposed through a new CLI flag, or extend the existing `--optimizer-backend` selector / internal `use_target_lane` toggles? Current answer: extend the existing `--optimizer-backend` / target-lane contract. `--backend` already means field backend (`cpu|jax`) in both Stage 2 and single-stage CLIs, while `--optimizer-backend=scipy-jax` already names the host-control/JAX-value-grad route. Avoid a third overlapping `--objective-backend` selector unless the production solver needs a separate mode after parity is complete.
- Should `BananaSystemSpec` live permanently in `src/simsopt_jax/core/banana.py`, or should banana-specific spec construction remain under `examples/single_stage_optimization/banana_opt/` with only reusable kernels in `src/`? (Note `stage2_target.py` already hosts a banana-specific spec lane in the adapters layer; this question also decides what is deduplicated against it.)
- Which Stage 2 seed and surface artifact should become the canonical small parity fixture?
- Should Equinox be excluded from this slice entirely, or allowed only for new banana-local typed PyTree modules after the current spec pattern is matched?
