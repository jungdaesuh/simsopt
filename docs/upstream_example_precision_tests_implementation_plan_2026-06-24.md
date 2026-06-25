# Upstream Example Precision Tests Implementation Plan

## Purpose

Prepare direct precision-test coverage for the upstream-master SIMSOPT example files that currently lack JAX precision / CPU-C++ parity counterparts in this checkout.

## Goals

- Register every target upstream-master example in the non-banana example parity inventory with an explicit supported, partial, or unsupported-native-JAX classification.
- Add real CPU/C++ or CPU public-API versus JAX CPU x64 precision comparisons for every target example that has a native JAX comparison surface in the current tree.
- Preserve external solver boundaries: VMEC, SPEC, QSC, MPI orchestration, plotting, VTK, and file-writing side effects must be named as host-side or unsupported unless a true JAX-native counterpart exists.
- Produce JSON artifacts and focused integration tests that make the new coverage reproducible.

## Non-Goals

- Do not add CUDA or GPU parity claims for this plan.
- Do not port VMEC, SPEC, QSC, BOOZXFORM, or external MPI solver execution to JAX.
- Do not claim optimizer iterate identity or full example runtime equivalence where the current parity harness only supports fixed-state or reduced deterministic subproblems.
- Do not include local/custom examples such as `examples/single_stage_optimization/run_single_stage_thresholded_physics_alm.py` or `examples/single_stage_optimization/run_stage2_alm.py`.

## Current Context

- The target list is the 27 current `examples/**/*.py` files that are present in the available master refs but absent from the existing direct parity coverage. Verified refs: `upstream_check/master` at `631cbe7361927a4800691645b5cc86a9abeab95c`, `upstream_hss/master` at `fc28d62f8e84e8f194ac5d1e74e360693b0ec368`, and `origin/master` at `21117aa8d0175e157f4cec42f92eff74b8c07755`; their `examples/**/*.py` file sets match for this audit.
- The existing parity harness lives in `benchmarks/non_banana_example_parity_fixtures.py` and `benchmarks/non_banana_example_cpp_jax_cpu_parity.py`.
- `FixtureSpec` already records `source_example`, `classification`, `classification_reason`, `rationale`, `acceptance_criteria`, `inputs`, and `fixture_kind`.
- The current fixture classification constants are `SUPPORTED = "supported"` and `UNSUPPORTED_NATIVE_JAX = "unsupported_native_jax"`. `partial` is a runtime verdict derived from supported fixtures with named `unsupported_components`; it is not a separate classification constant.
- The runner forces the CPU precision lane through `JAX_PLATFORMS=cpu` and `JAX_ENABLE_X64=1` unless the float32 smoke mode is explicitly selected.
- `docs/jax_parity_manifest.md` records the current non-banana fixture inventory and keeps CUDA status out of scope for CPU-only rows.
- The harness source and `docs/jax_parity_manifest.md` reference `docs/non_banana_example_cpp_jax_cpu_parity_plan_2026-05-12.md`, `docs/non_banana_example_cpp_jax_cpu_parity_results_2026-05-12.md`, and `docs/example_cpp_jax_cpu_gpu_parity_expansion_plan_2026-05-14.md`, but those files are not present in this checkout. Treat the live harness, current manifest table, and this plan as the authority until those links are repaired or the missing artifacts are restored.
- Traceable least-squares support lives in `src/simsopt_jax/solve/serial.py`; MPI assembly support lives in `src/simsopt_jax_adapters/solve/mpi.py`.
- The current worktree is dirty with unrelated changes. This plan is additive and should not require reverting or normalizing existing dirty files.

### Verified Target Inventory

- `examples/1_Simple/just_a_quadratic.py`
- `examples/1_Simple/logger_example.py`
- `examples/1_Simple/minimize_curve_length.py`
- `examples/2_Intermediate/B_external_normal.py`
- `examples/2_Intermediate/QH_fixed_resolution.py`
- `examples/2_Intermediate/QH_fixed_resolution_boozer.py`
- `examples/2_Intermediate/QSC.py`
- `examples/2_Intermediate/boozerQA_ls_mpi.py`
- `examples/2_Intermediate/constrained_optimization.py`
- `examples/2_Intermediate/eliminate_magnetic_islands.py`
- `examples/2_Intermediate/free_boundary_vmec.py`
- `examples/2_Intermediate/resolution_increase.py`
- `examples/2_Intermediate/resolution_increase_boozer.py`
- `examples/2_Intermediate/stage_two_optimization_stochastic.py`
- `examples/2_Intermediate/vmec_adjoint.py`
- `examples/3_Advanced/optimize_qs_and_islands_simultaneously.py`
- `examples/3_Advanced/single_stage_optimization.py`
- `examples/3_Advanced/single_stage_optimization_finite_beta.py`
- `examples/stellarator_benchmarks/1DOF_circularCrossSection_varyAxis_targetIota.py`
- `examples/stellarator_benchmarks/1DOF_circularCrossSection_varyAxis_targetIota_spec.py`
- `examples/stellarator_benchmarks/1DOF_circularCrossSection_varyR0_targetVolume.py`
- `examples/stellarator_benchmarks/1DOF_circularCrossSection_varyR0_targetVolume_spec.py`
- `examples/stellarator_benchmarks/2DOF_circularCrossSection_varyAxis_targetIotaAndQuasisymmetry.py`
- `examples/stellarator_benchmarks/2DOF_specOnly_targetIotaAndVolume.py`
- `examples/stellarator_benchmarks/2DOF_vmecAndSpec.py`
- `examples/stellarator_benchmarks/2DOF_vmecOnly_targetIotaAndVolume.py`
- `examples/stellarator_benchmarks/7dof.py`

## Rationale

Extending the existing fixture registry is the lowest-risk path because it already owns the CPU/JAX lane metadata, tolerance routing, artifact schema, dirty-tree recording, and focused integration tests. A separate runner would duplicate parity policy and make it easier to overclaim unsupported surfaces.

The implementation should classify each example by the smallest native comparison surface that faithfully represents the upstream example. Examples that are mostly external solver orchestration should become partial or unsupported rows, not artificial JAX tests. Examples with curve, surface, magnetic-field, Boozer, or deterministic objective subproblems can become supported or partial precision fixtures by reusing the current public JAX wrappers.

## Assumptions

- CPU precision evidence means `cpu_cpp` versus `jax_cpu` with x64 enabled, using the existing tolerance ladder in `benchmarks/validation_ladder_contract.py`.
- CPU public-API oracles are acceptable for Python-only example surfaces that do not call into `simsoptpp`, but the fixture classification must say when the oracle is CPU Python rather than C++.
- Existing JAX wrappers for curve objectives, surface objectives, Biot-Savart, SquaredFlux, Boozer residual/wrappers, VMEC frozen diagnostics, SurfaceGarabedian, CurvePerturbed, and traceable least squares are the only allowed JAX surfaces unless implementation work adds a new native wrapper with tests.
- VMEC, SPEC, QSC, BOOZXFORM, MPI execution, plotting, VTK output, and full optimizer loops remain host-side unless a separate implementation plan explicitly ports them.

## Implementation Plan

1. Lock the target inventory and coverage contract.
   - [ ] Add a test-local constant for the 27 upstream-master uncovered example paths in `tests/integration/test_non_banana_example_cpp_jax_cpu_parity.py`, copied from the verified target inventory above.
   - [ ] Add a coverage test that requires every listed path to appear as a `FixtureSpec.source_example`, or in a deliberate no-numeric/unsupported registry row with a clear `classification_reason`.
   - [ ] Keep the coverage test scoped to upstream-master files only; do not include local/custom single-stage driver files.
   - [ ] Verify duplicate fixture rows for one example are allowed only when they exercise distinct inputs or modes, as with the existing CWS rows.

2. Extend registry classification without broad schema churn.
   - [ ] Prefer the existing `SUPPORTED` and `UNSUPPORTED_NATIVE_JAX` classifications unless a concrete runner behavior requires a new classification string.
   - [ ] For unsupported rows, use `_unsupported_classification_builder` and include the exact unsupported boundary in `classification_reason`.
   - [ ] For partial rows, keep `classification=SUPPORTED`, return native-supported comparisons, and list host-only surfaces in `unsupported_components`.
   - [ ] Add fixture IDs with stable, descriptive names; avoid names tied to implementation phases or temporary TODO wording.

3. Add low-cost scalar and curve precision fixtures.
   - [ ] `examples/1_Simple/just_a_quadratic.py`: add a reduced least-squares scalar fixture comparing the CPU `Identity` / `LeastSquaresProblem` objective and final solved `(x, y, z)` against either a `TraceableLeastSquaresProblem` in `src/simsopt_jax/solve/serial.py` or an explicit JAX x64 quadratic expression. Mark the oracle as CPU Python plus analytic/traceable JAX, not C++.
   - [ ] `examples/1_Simple/logger_example.py`: add an explicit no-numeric row. Do not fabricate a precision comparison for logging and MPI rank formatting.
   - [ ] `examples/1_Simple/minimize_curve_length.py`: add a `CurveRZFourier` fixed-state fixture comparing `CurveLength` / `CurveLengthJAX` value, gradient, active DOF hash, and a reduced one-step least-squares objective if the existing solver wrapper supports a clean CPU/JAX comparison.

4. Add virtual-casing and target-array rows.
   - [ ] `examples/2_Intermediate/B_external_normal.py`: add a row that records `VirtualCasing.from_vmec` and `VirtualCasing.load` as host-side, then compares a cached or temporary-file-loaded `B_external_normal` target shape/hash and its use inside a reduced `SquaredFlux` / `SquaredFluxJAX` target fixture. Do not let the fixture write `vcasing_li383_low_res_reference.nc` into `tests/test_files/` during normal validation.
   - [ ] Reuse the existing finite-beta target-array metadata pattern instead of introducing a second target-array convention.
   - [ ] Add fixture assertions that the target array is shared byte-for-byte by CPU and JAX lanes before any objective comparison is accepted.

5. Add deterministic coil/objective composite rows.
   - [ ] `examples/2_Intermediate/stage_two_optimization_stochastic.py`: build a reduced deterministic fixture with fixed `PCG64DXSM` seed, reduced sample count, `CurvePerturbed` geometry, sample-mean `SquaredFlux`, and the supported curve penalties.
   - [ ] For the stochastic row, list unsupported components such as `ArclengthVariation` or full out-of-sample evaluation unless a native JAX wrapper exists and is exercised.
   - [ ] `examples/3_Advanced/single_stage_optimization.py`: add a reduced fixed-state row for the coil-side `SquaredFlux(definition="local")` plus supported curve penalties. Keep VMEC, finite-difference surface coupling, MPI, and full optimizer execution outside the native precision claim.
   - [ ] `examples/3_Advanced/single_stage_optimization_finite_beta.py`: mirror the single-stage row with the finite-beta target-array path. Keep `VirtualCasing.from_vmec`, VMEC, and full optimizer execution listed as host-side components.

6. Add Boozer and quasisymmetry fixed-state rows where JAX wrappers exist.
   - [ ] `examples/2_Intermediate/QH_fixed_resolution_boozer.py`: identify a reduced solved or fixed-state Boozer/Quasisymmetry scalar surface that can reuse the existing `boozer_qa_wrappers` comparison style.
   - [ ] `examples/2_Intermediate/resolution_increase_boozer.py`: add a reduced fixed-state row for Boozer/Quasisymmetry scalar checks after the example's resolution-change boundary is reconstructed.
   - [ ] `examples/stellarator_benchmarks/2DOF_circularCrossSection_varyAxis_targetIotaAndQuasisymmetry.py`: compare the available Boozer/Iotas/Quasisymmetry scalar outputs where the current wrappers support them; list VMEC solving as host-side.
   - [ ] `examples/stellarator_benchmarks/7dof.py`: add a reduced SurfaceGarabedian + Boozer scalar row only for supported fixed-state quantities. Keep the full seven-DOF VMEC optimization host-side.

7. Add VMEC frozen-diagnostic rows and explicit solver boundaries.
   - [ ] `examples/2_Intermediate/QH_fixed_resolution.py`: compare supported frozen VMEC diagnostics or surface geometry only; list `Vmec.run`, MPI solve, and `QuasisymmetryRatioResidual` solve orchestration as host-side unless a native JAX wrapper is added.
   - [ ] `examples/2_Intermediate/constrained_optimization.py`: classify `ConstrainedProblem` / `constrained_mpi_solve` and VMEC execution boundaries explicitly; add only supported fixed-state diagnostics.
   - [ ] `examples/2_Intermediate/free_boundary_vmec.py`: add a host-only or frozen-output row; do not claim free-boundary VMEC solve parity.
   - [ ] `examples/2_Intermediate/resolution_increase.py`: compare the pre/post resolution surface state or frozen diagnostics if reproducible without full VMEC optimization; otherwise record as host-only.
   - [ ] `examples/2_Intermediate/vmec_adjoint.py`: classify the `IotaTargetMetric` adjoint solve boundary. Add a fixed-state value check only if the current VMEC diagnostic adapters expose the same quantity.
   - [ ] `examples/stellarator_benchmarks/1DOF_circularCrossSection_varyAxis_targetIota.py`: add a reduced VMEC frozen-diagnostic or SurfaceGarabedian geometry row.
   - [ ] `examples/stellarator_benchmarks/1DOF_circularCrossSection_varyR0_targetVolume.py`: add a reduced volume/surface metric row if the CPU and JAX surfaces can be reconstructed without running the full solver.
   - [ ] `examples/stellarator_benchmarks/2DOF_vmecOnly_targetIotaAndVolume.py`: add a reduced VMEC frozen-diagnostic row for target iota/volume quantities only if supported by current adapters.

8. Add SPEC and QSC classifications without overclaiming.
   - [ ] `examples/2_Intermediate/QSC.py`: classify QSC execution as external Python package behavior. Add a precision row only if the `QSCWrapper` objective can be evaluated through a genuine JAX-native expression; otherwise add an unsupported-native-JAX registry row.
   - [ ] `examples/2_Intermediate/eliminate_magnetic_islands.py`: classify `Spec` and `Residue` as host-side unless a native JAX SPEC/residue wrapper exists.
   - [ ] `examples/3_Advanced/optimize_qs_and_islands_simultaneously.py`: separate the VMEC/Boozer fixed-state pieces from the SPEC island-residue pieces; only the former may receive native comparisons.
   - [ ] `examples/stellarator_benchmarks/1DOF_circularCrossSection_varyAxis_targetIota_spec.py`: classify the SPEC target-iota path as host-side unless a native wrapper is added.
   - [ ] `examples/stellarator_benchmarks/1DOF_circularCrossSection_varyR0_targetVolume_spec.py`: classify the SPEC volume path as host-side unless a native wrapper is added.
   - [ ] `examples/stellarator_benchmarks/2DOF_specOnly_targetIotaAndVolume.py`: classify the SPEC-only row as unsupported-native-JAX unless native SPEC diagnostics are introduced.
   - [ ] `examples/stellarator_benchmarks/2DOF_vmecAndSpec.py`: split VMEC-supported diagnostics from SPEC host-only diagnostics in one partial row.

9. Update docs and artifacts.
   - [ ] Update `docs/jax_parity_manifest.md` with a new "Upstream Example Precision Expansion" subsection listing each new fixture, verdict, source example, comparisons, and unsupported components.
   - [ ] Repair or replace the current missing links in `docs/jax_parity_manifest.md` for the non-banana plan/results/expansion documents.
   - [ ] Update `docs/jax_parity_status.md` only with reproduced evidence, not planned rows.
   - [ ] Write a results document after implementation, for example `docs/upstream_example_precision_tests_results_2026-06-24.md`.
   - [ ] Store CPU JSON artifacts under `.artifacts/parity/20260624-upstream-example-precision/`.

10. Keep test and code changes scoped.
   - [ ] Put fixture builders in `benchmarks/non_banana_example_parity_fixtures.py` near the most similar existing builder.
   - [ ] Add comparison helpers to `benchmarks/non_banana_example_cpp_jax_cpu_parity.py` only when no existing `fixture_kind` can express the row.
   - [ ] Add focused integration assertions in `tests/integration/test_non_banana_example_cpp_jax_cpu_parity.py` for each new supported or partial fixture family.
   - [ ] Avoid importing example scripts directly when they create output directories, run MPI solvers, launch external codes, or write VTK files at import time.

## Validation Plan

- [ ] Run syntax checks for touched Python files:

  ```bash
  python -m py_compile benchmarks/non_banana_example_parity_fixtures.py benchmarks/non_banana_example_cpp_jax_cpu_parity.py tests/integration/test_non_banana_example_cpp_jax_cpu_parity.py
  ```

- [ ] Run the focused integration test file:

  ```bash
  PYTHONNOUSERSITE=1 PYTHONPATH=src pytest tests/integration/test_non_banana_example_cpp_jax_cpu_parity.py
  ```

- [ ] Run the new CPU precision fixture batch and save an artifact:

  ```bash
  PYTHONNOUSERSITE=1 PYTHONPATH=src python benchmarks/non_banana_example_cpp_jax_cpu_parity.py --fixtures <new-fixture-ids> --output-json .artifacts/parity/20260624-upstream-example-precision/cpu_cpp_vs_jax_cpu.json
  ```

- [ ] Run the full supported non-banana example batch after the focused fixtures pass:

  ```bash
  PYTHONNOUSERSITE=1 PYTHONPATH=src python benchmarks/non_banana_example_cpp_jax_cpu_parity.py --fixtures all-supported --output-json .artifacts/parity/20260624-upstream-example-precision/all-supported-cpu.json
  ```

- [ ] Run a full registry artifact that includes unsupported-native-JAX rows, so host-only classifications are serialized and visible:

  ```bash
  PYTHONNOUSERSITE=1 PYTHONPATH=src python benchmarks/non_banana_example_cpp_jax_cpu_parity.py --fixtures all --output-json .artifacts/parity/20260624-upstream-example-precision/all-registry-cpu.json
  ```

- [ ] Verify the target inventory coverage test fails if any of the 27 listed upstream examples are removed from the registry.
- [ ] Verify `all-registry-cpu.json` contains no `verdict: "fail"` rows and that every `verdict: "unsupported"` row has a non-empty `classification_reason` and `error`.
- [ ] Run `git diff --check`.
- [ ] Review the JSON artifact and docs to confirm unsupported components are explicit and no row claims CUDA evidence.

## Risks and Mitigations

- Risk: A host-only external solver example is accidentally reported as JAX precision coverage.
  Mitigation: Require unsupported components and classification reasons for VMEC, SPEC, QSC, MPI, BOOZXFORM, plotting, VTK, and full optimizer loops.

- Risk: Fixture builders import example scripts and trigger side effects.
  Mitigation: Reconstruct reduced fixture state from the example constants and inputs instead of importing top-level example modules.

- Risk: New rows duplicate existing low-level unit tests without tying coverage to the upstream example.
  Mitigation: Every new fixture must record `source_example`, input hashes where files are used, and acceptance criteria that name the example-derived state.

- Risk: A reduced fixture overclaims full example equivalence.
  Mitigation: Use `classification_reason`, `rationale`, and `unsupported_components` to distinguish fixed-state precision from solver/runtime parity.

- Risk: VMEC/SPEC/QSC rows become a large porting project.
  Mitigation: Keep this plan to registry classification and reduced native surfaces; require a separate design plan before adding new native solver wrappers.

## Completion Criteria

- [ ] All 27 target upstream-master examples appear in the parity registry through supported, partial, or unsupported-native-JAX rows.
- [ ] Every supported or partial row has runnable focused tests and a CPU JSON artifact with no native-supported comparison failures.
- [ ] Every unsupported or host-only row states the exact blocker and avoids fake numeric precision claims.
- [ ] `docs/jax_parity_manifest.md` and the new results document match the produced JSON artifact.
- [ ] Focused integration tests, fixture batch command, `py_compile`, and `git diff --check` pass.
- [ ] The worktree diff contains only the intended fixture, test, doc, and artifact-reference changes for this expansion.

## Open Questions

- Should CPU Python analytic oracles, such as the `just_a_quadratic.py` row, count toward the same completion bucket as CPU/C++ oracle rows, or should the docs split them into a separate "CPU public-API oracle" bucket?
- Should the VMEC frozen-diagnostic rows be considered precision coverage for VMEC examples, or should VMEC examples remain unsupported until a native JAX VMEC solver exists?
- Should SPEC-only rows be represented as unsupported registry rows now, or deferred until there is a concrete SPEC JAX adapter decision?
- What artifact size and runtime budget should apply to the stochastic and single-stage reduced fixtures in CI?
