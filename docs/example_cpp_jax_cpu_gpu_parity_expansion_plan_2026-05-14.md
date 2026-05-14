# Example C++/JAX CPU Parity Expansion Plan - 2026-05-14

Status: implementation in progress for expanding example-derived parity
evidence beyond the existing non-banana fixture set. This document is not a GPU
pass claim. The current CPU-only evidence remains in
`docs/non_banana_example_cpp_jax_cpu_parity_results_2026-05-12.md`; the
2026-05-14 expansion artifact for the current supported rows is
`.artifacts/parity/20260514-example-expansion/all-supported-cpu.json`; the
full registry, including support-gate rows, was first captured in
`.artifacts/parity/20260514-example-expansion/all-fixtures.json` and is
refreshed after the curve-objective closeout in
`.artifacts/parity/20260514-curve-objectives/all-fixtures-current.json`.

Implemented in the current expansion pass:

- [x] Wave 0 harness fields needed by the first expansion rows:
      family constants, fixture input/file-content hashes, CPU-only
      GPU-readiness metadata, version-probe metadata, `--lanes` selection plus
      a fail-closed `jax_gpu` CUDA follow-up lane that requires a CPU baseline
      artifact,
      source-example comparison fields, and no-fake-pass tests.
- [x] Wave 1 `surface_area_volume_simple`: fixed-state Area/Volume value and
      gradient parity for `examples/1_Simple/surf_vol_area.py`.
- [x] Wave 2 `qfm_surface`: fixed-state QFM residual, Bdotn, gradient, and
      Area/Volume/ToroidalFlux label parity for `examples/1_Simple/qfm.py`,
      with `QfmSurface_host_solver` explicitly listed as unsupported.
- [x] Wave 3 `pm_simple_fixed_state_gpmo_baseline`: reduced fixed-state
      `permanent_magnet_simple.py` payload, baseline GPMO final moment/residual,
      `R2_history`, `Bn_history`, `m_history`, and `DipoleField` vs
      `DipoleFieldJAX` parity at `K=4`.
- [x] Wave 3 `pm_qa_fixed_state_gpmo_arbvec_or_multi`: historical id retained,
      but the row now preserves the real `permanent_magnet_QA.py`
      `relax_and_split` algorithm family and reaches partial CPU/C++ vs JAX CPU
      parity for grid payloads, dense/proxy moments, residuals, objectives, and
      scalar and dense/proxy histories plus `DipoleField`/`DipoleFieldJAX`;
      coil-current optimization and output side effects remain explicitly
      unsupported.
- [x] Wave 3 `pm_muse_famus`: reduced MUSE FAMUS fixture preserves the
      example `ArbVec_backtracking` algorithm family and reaches pass
      CPU/C++ vs JAX CPU parity for grid payloads, final moments, residual,
      objective, `R2_history`, `Bn_history`, `m_history`, and
      `DipoleField`/`DipoleFieldJAX`.
- [x] Wave 3 `pm_pm4stell_backtracking`: reduced PM4Stell fixture preserves
      the example `ArbVec_backtracking` algorithm family with PM4Stell
      face/edge/corner triplet polarizations and reaches pass CPU/C++ vs JAX
      CPU parity for grid payloads, final moments, residual, objective,
      `R2_history`, `Bn_history`, `m_history`, and `DipoleField`/
      `DipoleFieldJAX`.
- [x] Wave 4 `wireframe_rcls_basic_fixed_state`: fixed-state RCLS matrix,
      objective, constraint, `WireframeField`, and Bnormal parity for
      `wireframe_rcls_basic.py`. The raw current vector is explicitly listed as
      `RCLS_current_vector_nonunique_nullspace` rather than treated as proof.
- [x] Wave 4 `wireframe_rcls_ports_constraint_gate`: reduced
      port-constrained RCLS fixture for `wireframe_rcls_with_ports.py`
      preserves port collision masks, the poloidal-current constraint, and
      the public `surf_plas` input mode. It reaches partial CPU/C++ vs JAX CPU
      parity for matrices, objective components, constraint shape and
      satisfaction, `WireframeField`/`WireframeFieldJAX`, and Bnormal, with
      raw current-vector identity still named as
      `RCLS_current_vector_nonunique_nullspace`.
- [x] Wave 4 `wireframe_gsco_sector_saddle_fixed_state`: reduced
      sector/saddle GSCO fixture for `wireframe_gsco_sector_saddle.py`
      preserves TF-coil initial currents, toroidal break free-cell masks,
      poloidal-current constraints, and the public `surf_plas` input mode. It
      reaches pass CPU/C++ vs JAX CPU parity for matrices, constraint flags
      and masks, final current state, histories, `WireframeField`/
      `WireframeFieldJAX`, and Bnormal.
- [x] Wave 4 `wireframe_gsco_multistep_reduced_diagnostic`: reduced
      first-step `wireframe_gsco_multistep.py` diagnostic preserves the
      public `surf_plas`/`ext_field` `optimize_wireframe` input mode and
      reaches partial CPU/C++ vs JAX CPU parity for matrices, GSCO flags,
      final current/loop-count state, and history arrays. The mutating
      multistep loop, small-coil pruning, final adjustment, and plot/VTK
      outputs remain explicitly unsupported.
- [x] Wave 5 `tracing_fieldlines_qa_reduced_endpoint`: reduced
      `tracing_fieldlines_QA.py` interpolated-field tracing fixture using
      `InterpolatedFieldJAX.jax_B_at`, with endpoint and phi-hit coordinate
      parity under the `event_time_tracing` lane. The example's raw
      `LevelsetStoppingCriterion(sc_fieldline.dist)` adapter and skip callback
      are now exercised.
- [x] Wave 5 `tracing_fieldlines_ncsx_reduced_endpoint`: reduced
      `tracing_fieldlines_NCSX.py` fixture using `get_data("ncsx")`,
      `InterpolatedFieldJAX.jax_B_at`, one magnetic-axis fieldline, and a
      pinned phi plane. It reaches CPU/C++ vs JAX CPU `pass` for field
      values, endpoint, final integration time/status, hit coordinates, and
      hit count while exercising the raw levelset-distance stopping adapter
      and skip callback.
- [x] Wave 5 `tracing_particle_gc_vac_reduced_endpoint`: reduced
      `tracing_particle.py` particle guiding-center fixture using
      `InterpolatedFieldJAX.jax_B_GradAbsB_at`, one axis-seeded particle,
      `mode='gc_vac'`, pinned phi planes, and `forget_exact_path=True`. It
      reaches CPU/C++ vs JAX CPU `pass` for interpolated B, GradAbsB,
      endpoint, final time/status, phi-hit rows, and hit count while exercising
      the example's raw `LevelsetStoppingCriterion(sc_particle.dist)` adapter.
- [x] Wave 5 `tracing_boozer_gc_reduced_endpoint`: reduced
      `tracing_boozer.py` Boozer guiding-center fixture using cached VMEC wout
      plus cached BOOZXFORM boozmn data and
      `InterpolatedBoozerFieldJAX` frozen-state scalar evaluators. It reaches
      partial CPU/C++ vs JAX CPU parity for modB, endpoint, final time/status,
      zeta-hit rows, and hit count; the input-file VMEC solve remains named
      unsupported because the local/CI fixture path has no VMEC/BOOZXFORM
      external-solver extension.
- [x] Wave 6 `finite_beta_target_flux`: W7-X finite-beta fixed-state
      `SquaredFlux` target-array parity using a deterministic cached
      `B_external_normal` virtual-casing target array. The curve-objective
      closeout adds the length identity penalties to the native JAX lane, so
      this row now reaches CPU/C++-vs-JAX CPU `pass`.
- [x] Wave 6 `coil_forces_support_gate`: reduced fixed-state force/energy
      subproblem for `coil_forces.py` now reaches public-wrapper vs
      explicit `LpCurveForceJAX` / `B2EnergyJAX` lane parity for force and
      magnetic-energy values, per-component gradients, and the weighted native
      subtotal. It also compares the public JAX force/energy values against
      independent CPU oracles.

Current trust chain:

```text
SIMSOPT CPU/C++ behavior -> JAX CPU matches at the same state -> later JAX GPU matches JAX CPU and preserves the same CPU/C++ oracle contract
```

The immediate implementation target is CPU/C++ vs JAX CPU parity. GPU proof is
a later gate and must not be inferred from JAX CPU success.

## Requirements

- [x] Reuse the existing example parity harness shape before adding a new
      benchmark framework.
- [x] Treat upstream SIMSOPT CPU/C++ behavior as the oracle.
- [x] Compare fixed states first: same geometry, quadrature, field points,
      target arrays, free/fixed DOFs, currents, solver parameters, and input
      vectors.
- [x] Compare JAX CPU against CPU/C++ before comparing any JAX GPU lane.
- [x] Keep full example optimizer trajectories as diagnostics unless the
      optimizer route itself is the component under test.
- [x] Avoid importing example scripts that execute optimizers, write VTK files,
      generate plots, or mutate global state at import time.
- [x] Reconstruct fixture state from the same inputs the examples use, with
      side-effect-free fixture builders.
- [x] Record every unsupported component by exact component name. Do not fill
      a JAX lane with CPU code and call it parity.
- [x] Keep mixed CPU/JAX diagnostic rows separate from native-JAX pass rows.
- [x] Require `JAX_ENABLE_X64=1` and `JAX_PLATFORMS=cpu` for CPU parity runs.
- [x] Synchronize JAX timing or completion-sensitive checks with
      `block_until_ready()` or `jax.block_until_ready(...)`.
- [x] Use transfer-guard or strict target-lane checks for GPU-readiness probes
      where the code claims no hidden host roundtrip.
- [x] Preserve the dirty-worktree policy: record dirty metadata, but do not
      stage unrelated user changes.

## Current Context

Existing evidence:

- [x] `benchmarks/non_banana_example_cpp_jax_cpu_parity.py` and
      `benchmarks/non_banana_example_parity_fixtures.py` already provide the
      fixture/reporting pattern to extend.
- [x] `tests/integration/test_non_banana_example_cpp_jax_cpu_parity.py` already
      enforces verdict semantics for `pass`, `partial`, and `unsupported`.
- [x] Existing non-banana rows cover Stage-II, Boozer, finite-beta support
      gates, finite-build, CWS, and QFM rows. In this expansion pass CWS moved
      to `pass`, finite-build first moved to `partial`, and QFM moved to `partial`
      with `QfmSurface_host_solver` still named unsupported. The later
      curve-objective closeout adds public JAX curve-objective wrappers and
      moves `minimal_stage2_flux_length_gap`, `full_stage2_composite`,
      `planar_stage2_composite`, `finite_beta_target_flux`, and
      `finitebuild_multifilament_support_gate`
      to CPU/C++-vs-JAX CPU `pass` in
      `.artifacts/parity/20260514-curve-objectives/local-converted-fixtures.json`.
      The refreshed full CPU/JAX matrix in
      `.artifacts/parity/20260514-curve-objectives/all-fixtures-current.json`
      is 27 fixtures = 21 pass / 6 partial / 0 fail; the remaining partial
      rows are named support gaps, not failing native-supported comparisons.
- [x] `docs/jax_parity_manifest.md` records surface/objective non-CUDA evidence
      and the current non-banana inventory.

Support that changed since the 2026-05-12 plan:

- [x] `AreaJAX`, `VolumeJAX`, and `QfmResidualJAX` exist in
      `src/simsopt/geo/surfaceobjectives_jax.py`.
- [x] Permanent-magnet JAX support now includes `PermanentMagnetGridJAX`,
      `DipoleFieldJAX`, and explicit `GPMO_*_jax` wrappers.
- [x] Wireframe JAX support now includes `WireframeFieldJAX`,
      `rcls_wireframe_jax`, `gsco_wireframe_jax`, and
      `optimize_wireframe_jax`.
- [x] Tracing support now includes JAX fieldline, guiding-center, full-orbit,
      and Boozer guiding-center routes.

Important carve-outs:

- [x] `QfmSurface` remains host solver orchestration over mutable
      `surface.x`; prove `QfmResidualJAX` and label metrics, not the SciPy
      QFM optimizer path.
- [x] The live mutable `PermanentMagnetGrid` host loop remains host
      orchestration; prove immutable `PermanentMagnetGridJAX` payloads and
      explicit algorithms, not a generic CPU-style dispatcher.
- [x] VMEC/SPEC/QSC examples remain external-solver workflows unless a fixture
      extracts a fixed native-JAX-supported subproblem.
- [x] VTK, plotting, logging, and file-output side effects are not parity
      surfaces.

## Official Documentation Checks

Sources checked during the document review:

- JAX official docs via Context7 for `/google/jax`, query `official docs x64
  default dtypes asynchronous dispatch block_until_ready transfer guard CUDA GPU
  platform parity`.
- Context7 resolved SIMSOPT as `/hiddensymmetries/simsopt`; public API details
  below were checked against the official Read the Docs pages.
- JAX installation docs: <https://docs.jax.dev/en/latest/installation.html>.
- SIMSOPT field docs: <https://simsopt.readthedocs.io/latest/simsopt.field.html>.
- SIMSOPT geometry docs: <https://simsopt.readthedocs.io/latest/simsopt.geo.html>.
- SIMSOPT solve docs: <https://simsopt.readthedocs.io/latest/simsopt.solve.html>.
- NVIDIA CUDA Programming Guide:
  <https://docs.nvidia.com/cuda/cuda-programming-guide/>.

Doc-backed constraints that affect this plan:

- [x] JAX x64 is controlled by `jax_enable_x64` or `JAX_ENABLE_X64`; parity
      runs must enable it before importing JAX and before creating arrays.
- [x] JAX defaults away from x64, so a run without x64 is a smoke test only.
- [x] JAX dispatch is asynchronous; measurements and completion-sensitive
      evidence must explicitly wait with `block_until_ready()`.
- [x] JAX transfer guard can disallow implicit transfers; strict GPU-readiness
      tests should use this to catch hidden host/device movement.
- [x] JAX GPU parity requires a supported JAX CUDA wheel/runtime combination
      and an active CUDA backend. A GPU row must fail when JAX is running on
      CPU.
- [x] JAX CUDA rows must record which supported install line they use:
  - [x] CUDA 12 wheel/runtime line: NVIDIA driver at least 525, CUDA at least
        12.1, and GPU compute capability at least SM 5.2.
  - [x] CUDA 13 wheel/runtime line: NVIDIA driver at least 580, CUDA at least
        13.0, and GPU compute capability at least SM 7.5.
- [x] SIMSOPT `MagneticField` comparisons must respect the documented
      `set_points(...)` contract: set identical point arrays immediately before
      each CPU/C++ and JAX field evaluation, using independent field instances
      per lane.
- [x] SIMSOPT wireframe rows must preserve the documented
      `optimize_wireframe(...)` input mode. Compare `surf_plas` paths against
      `surf_plas` paths, and compare precomputed `Amat`/`bvec` paths against
      the same frozen `Amat`/`bvec` paths.
- [x] SIMSOPT tracing rows must pin `mode`, stopping criteria, `phis`, and
      `forget_exact_path`; compare only endpoint/event arrays returned by that
      exact trace contract.
- [x] CUDA documentation provenance must use the current CUDA Programming Guide
      URL above. The old CUDA C++ Programming Guide index URL is legacy.

## Shared Harness Contract

Extend the existing harness rather than creating a parallel system.

- [x] Add new `FixtureSpec` rows to
      `benchmarks/non_banana_example_parity_fixtures.py` unless a fixture family
      requires a small helper module.
- [x] Keep one JSON schema for all example parity rows.
- [x] Keep the existing dataclass boundaries as the source of truth:
  - [x] `FixtureSpec`: `fixture_id`, `source_example`, `classification`,
        `classification_reason`, `inputs`, and `fixture_kind`.
  - [x] `FixtureBuild`: `spec`, `cpu_lane`, `jax_lane`,
        `unsupported_components`, optional native-subproblem evaluators, and
        optional `x0`.
  - [x] Output `FixtureResult`: `fixture_inputs`, `dof_contract`,
        `native_spec_contract`, `lanes`, `comparisons`,
        `unsupported_components`, diagnostics, `verdict`, `passed`,
        `failures`, and optional `error`.
  - [x] Run artifact: top-level `metadata` plus the fixture result list.
- [x] Add family-specific comparison helpers only when the existing comparison
      shape cannot express the result.
- [x] Store CPU lane and JAX lane artifacts in separate objects. Do not share
      mutable surfaces, fields, grids, or wireframe instances between lanes.
- [x] Add one integration test per new fixture verdict and one aggregate
      smoke test for the expanded fixture set.

Required comparison fields:

- [x] `quantity`
- [x] `component`
- [x] `source_example`
- [x] `cpu_cpp_value`
- [x] `jax_cpu_value`
- [x] `tolerance_bucket`
- [x] `rtol`
- [x] `atol`
- [x] `max_abs_diff`
- [x] `max_rel_diff`
- [x] `argmax_index`
- [x] `verdict`

Required metadata:

- [x] Git SHA and dirty-tree summary.
- [x] `JAX_ENABLE_X64` and active platform.
- [x] JAX version, backend, devices, and device kind.
- [x] For GPU rows: JAX CUDA wheel/runtime line, CUDA runtime version visible
      to JAX, NVIDIA driver version, device name, and compute capability.
- [x] SIMSOPT example source path.
- [x] Input files, file-content hashes for file-backed fixtures, generated
      fixture hashes, quadrature sizes, and seeds.
- [x] Explicit statement of whether the row is CPU-only, GPU-ready, or
      GPU-proven.

No-fake-test rules:

- [x] A pass row must compare CPU/C++ oracle output against independently built
      JAX CPU output.
- [x] JAX-vs-JAX comparisons may appear only as transform or GPU follow-up
      checks, never as the CPU/C++ preservation oracle.
- [x] A test that compares a function to itself is not parity evidence.
- [x] A test that only asserts the harness records an unsupported verdict is a
      support-gate test, not a port-correctness pass.
- [x] Tolerances must come from the existing validation ladder or be justified
      in the fixture row.
- [x] Do not loosen tolerances in the same commit that introduces a failing
      comparison.

## Wave 0 - Harness Preparation

Rationale: the next examples span surface scalars, QFM residuals,
permanent-magnet optimization, wireframe solves, and tracing. A small set of
shared helpers avoids one-off tests and keeps verdict semantics consistent.

Implementation checklist:

- [x] Add fixture-family constants for `surface_scalar`, `qfm`, `pm`,
      `wireframe`, and `tracing`.
- [x] Add helpers for recording fixed-state input hashes for dense arrays and
      content hashes for file-backed fixtures.
- [x] Add a helper for recording device/completion metadata after JAX values are
      blocked.
- [x] Add a helper that marks a fixture `gpu_ready=false` until a real GPU lane
      artifact exists.
- [x] Add version-probe guidance to every runnable command:
      `conda run -n jax-0.9.2 python -c "import jax, jaxlib; print(jax.__version__, jaxlib.__version__)"`.
- [x] Add fail-closed lane-selection scaffolding before documenting an
      executable GPU artifact command:
  - [x] Add a `--lanes` CLI option or an equivalent explicit lane selector.
  - [x] Plumb lane selection through `run_fixtures`; requested lanes are
        recorded in run metadata and unselected lane payloads/pairwise
        comparisons are omitted from the emitted artifact. A selection that
        omits either side of the `cpu_cpp`/`jax_cpu` parity pair fails closed
        instead of preserving a `pass` or `partial` verdict.
  - [x] Reject requested `jax_gpu` rows until a real CUDA execution/artifact
        path exists; this is a fail-closed guard, not GPU proof.
  - [x] Keep the current CPU-only runner contract intact for CPU parity rows.
- [x] Add a test that every new fixture has a non-empty `source_example`,
      `rationale`, and `acceptance_criteria`.
- [x] Add a test that every `pass` or `partial` row has at least one
      CPU/C++-vs-JAX CPU numeric comparison.
- [x] Add a test that no row uses a `cpu_fallback`, `host_fallback`, or
      `jax_self_reference` verdict marker.

Acceptance criteria:

- [x] Existing 2026-05-12 fixture ids still produce valid pass/partial/
      unsupported verdicts under the expanded schema; rows intentionally
      upgraded in this pass have their new verdicts recorded in the manifest.
- [x] The existing fixture regression suite is run before and after adding new
      fixture families.
- [x] New schema fields are present for old and new fixtures.
- [x] Fixture-level tests fail if a pass row has zero CPU/C++ oracle
      comparisons.

## Wave 1 - `examples/1_Simple/surf_vol_area.py`

Context:

- Source example: `examples/1_Simple/surf_vol_area.py`.
- CPU path: `SurfaceRZFourier`, `Area`, `Volume`, and the example's surface
  optimization objective.
- JAX path: `AreaJAX`, `VolumeJAX`, and `SurfaceRZFourier` JAX geometry helpers.

Rationale:

- This is the cleanest low-cost example-derived fixture.
- It tests scalar surface objectives and gradients without coils, external
  solvers, plotting, or optimizer-path ambiguity.
- It provides a small regression guard for the newly added Area/Volume wrapper
  parity.

Implementation checklist:

- [x] Add fixture `surface_area_volume_simple`.
- [x] Rebuild the example's initial `SurfaceRZFourier` state in a fixture
      builder instead of importing the script.
- [x] Build independent CPU and JAX surfaces from identical DOFs.
- [x] Compare `Area(s).J()` against `AreaJAX(s_jax).J()`.
- [x] Compare `Volume(s).J()` against `VolumeJAX(s_jax).J()`.
- [x] Compare `Area(s).dJ()` against `AreaJAX(s_jax).dJ()`.
- [x] Compare `Volume(s).dJ()` against `VolumeJAX(s_jax).dJ()`.
- [x] Add deterministic perturbed-surface checks for at least two nontrivial
      DOF perturbations.
- [x] Record the surface resolution, active DOF names, and target area/volume
      values from the example.
- [x] Keep any optimizer final-state check diagnostic-only until fixed-state
      parity passes.

Acceptance criteria:

- [x] Fixture verdict is `pass`.
- [x] Value and gradient comparisons pass under the `direct_kernel` or
      `derivative_heavy` tolerance bucket already used for surface objectives.
- [x] The fixture fails if the CPU and JAX lanes share the same surface object.
- [x] The JSON artifact reports zero unsupported components.
- [x] Later GPU gate can reuse the same fixture without changing fixture inputs.

## Wave 2 - `examples/1_Simple/qfm.py`

Context:

- Source example: `examples/1_Simple/qfm.py`.
- CPU path: `QfmResidual`, `QfmSurface`, `Volume`, `ToroidalFlux`, and `Area`.
- JAX path: `QfmResidualJAX`, `VolumeJAX`, `AreaJAX`, and existing
  `ToroidalFlux` JAX helper coverage.
- Current status: the 2026-05-12 harness reports `qfm_surface` as unsupported
  because fixture wiring is missing, not because the residual kernel is absent.

Rationale:

- This closes the current unsupported QFM row without claiming the host SciPy
  `QfmSurface` solver.
- It validates field-coupled surface residual parity plus label metrics used by
  the example.

Implementation checklist:

- [x] Replace the current `qfm_surface` unsupported builder with a fixed-state
      residual/label builder.
- [x] Rebuild the example's circular coils and initial surface independently for
      CPU and JAX lanes.
- [x] Compare `QfmResidual(s, bs).J()` against `QfmResidualJAX(s_jax, bs_jax).J()`.
- [x] Compare `QfmResidual.dJ()` against `QfmResidualJAX.dJ()` in the same
      surface DOF basis.
- [x] Compare the label metrics used by the example: `Volume`, `Area`, and
      `ToroidalFlux`.
- [x] Add separate fixed-state rows for the initial state and one
      post-constraint target state if the latter can be reconstructed without
      running SciPy in the JAX lane. Current fixture records
      `post_constraint_target_state=not_reconstructable_without_host_scipy_QfmSurface`;
      the example's post-constraint state is only created by mutable
      `QfmSurface.minimize_*` CPU solver orchestration, so no JAX fixed-state
      post-target row is claimed.
- [x] Keep `QfmSurface.minimize_*` results as CPU-only diagnostics unless a
      separate plan ports the solver orchestration.

Acceptance criteria:

- [x] Fixture verdict changes from `unsupported` to `pass` or `partial`.
- [x] Any remaining unsupported item is named as `QfmSurface_host_solver`, not
      hidden inside the QFM residual comparison.
- [x] Residual vector, residual norm, and surface gradient comparisons pass.
- [x] Area/Volume/ToroidalFlux label comparisons pass under their existing
      tolerance buckets.
- [x] No test imports `examples/1_Simple/qfm.py` as a module.

## Wave 3 - Permanent-Magnet Examples

Source examples:

- `examples/1_Simple/permanent_magnet_simple.py`
- `examples/2_Intermediate/permanent_magnet_QA.py`
- `examples/2_Intermediate/permanent_magnet_MUSE.py`
- `examples/2_Intermediate/permanent_magnet_PM4Stell.py`

Context:

- CPU path: `PermanentMagnetGrid`, `DipoleField`, and public
  `GPMO(pm_opt, algorithm=...)`.
- JAX path: immutable `PermanentMagnetGridJAX`, `DipoleFieldJAX`, and explicit
  `GPMO_baseline_jax`, `GPMO_multi_jax`, `GPMO_ArbVec_jax`,
  `GPMO_backtracking_jax`, `GPMO_ArbVec_backtracking_jax`.
- Current carve-out: the live mutable `PermanentMagnetGrid` host-loop workflow
  is not a JAX target. The supported JAX contract consumes immutable fixed-state
  payloads.
- Mapping note: CPU exposes one dispatcher with algorithm strings; JAX exposes
  algorithm-specific wrappers. Fixture rows compare one CPU algorithm selection
  against its matching JAX wrapper, not generic dispatcher parity.

Rationale:

- These examples exercise native C++/simsoptpp permanent-magnet code paths that
  now have explicit JAX ports.
- They provide strong evidence that real example geometry and FAMUS-derived
  inputs can be represented without relying on toy synthetic grids.

Implementation checklist:

- [x] Add fixture `pm_simple_fixed_state_gpmo_baseline`.
- [x] Add support-gate row `pm_qa_fixed_state_gpmo_arbvec_or_multi` pending
      exact algorithm selection and reduced deterministic inputs.
- [x] Upgrade `pm_qa_fixed_state_gpmo_arbvec_or_multi` to a partial fixture
      after choosing the exact algorithm used by the example (`relax_and_split`;
      the fixture id is historical).
- [x] Add support-gate rows for `pm_muse_famus` and
      `pm_pm4stell_backtracking`.
- [x] Upgrade `pm_muse_famus` to a partial fixture with input-file
      availability, deterministic CI reductions, and example
      `ArbVec_backtracking` routing pinned.
- [x] Upgrade `pm_pm4stell_backtracking` to partial after input-file
      availability and deterministic CI reductions are pinned.
- [x] Preserve the public example algorithm selection before creating each
      fixture:
  - [x] `permanent_magnet_simple.py` default `GPMO(pm_opt, **kwargs)` maps to
        the documented baseline algorithm.
  - [x] `permanent_magnet_QA.py` uses `relax_and_split`, not a GPMO family;
        the historical fixture id does not define the algorithm contract.
  - [x] MUSE row uses `ArbVec_backtracking` on both CPU/C++ and JAX lanes.
  - [x] PM4Stell row that uses `ArbVec_backtracking` must stay in that
        algorithm family.
  - [x] No parity row may compare baseline CPU history to backtracking JAX
        history.
- [x] Build the CPU `PermanentMagnetGrid` from the example input files.
- [x] Convert the CPU grid to `PermanentMagnetGridJAX.from_cpu(...)` or an
      equivalent fixed-state constructor.
- [x] Compare immutable grid payload fields: geometry arrays, normal field
      vector, maxima, active mask, and operator shapes.
- [x] Compare CPU `DipoleField` against `DipoleFieldJAX` for final and
      selected intermediate moment arrays at representative surface points.
- [x] Compare the explicit algorithm outputs against the CPU algorithm variant:
      residual histories, selected dipoles, selected components/vectors, final
      moment array, and final normal-field residual.
- [x] Record the exact GPMO algorithm variant. Do not test the generic JAX
      dispatcher because none is exported by design.
- [x] Use reduced deterministic `K` values for CI and keep full example-size
      runs as optional artifacts.

Acceptance criteria:

- [x] At least one permanent-magnet fixture reaches `pass` with CPU/C++ vs JAX
      CPU comparisons.
- [x] The fixture JSON names the algorithm variant and does not claim generic
      `GPMO` dispatcher parity.
- [x] `DipoleField` vs `DipoleFieldJAX` field comparisons pass for the selected
      final moment state.
- [x] Algorithm histories match within the same tolerance contract used by the
      existing permanent-magnet JAX tests, including CPU `R2_history`,
      `Bn_history`, and `m_history` counterparts where the public CPU function
      returns them.
  - [x] `pm_simple_fixed_state_gpmo_baseline` now compares CPU and JAX
        `R2_history`, `Bn_history`, and `m_history`.
  - [x] `pm_muse_famus` and `pm_pm4stell_backtracking` now compare
        `R2_history`, `Bn_history`, and `m_history` for the
        `ArbVec_backtracking` family.
  - [x] `ArbVec_backtracking` `m_history` is compared after the reduced CI
        fixtures use `K=5`, avoiding the CPU oracle's duplicate terminal
        `k=K-1` history write for `K=max_nMagnets=4`.
  - [x] `relax_and_split` scalar `RS_history` now compares the CPU
        relax-and-split scalar cost; the JAX inner-solver residual trace remains
        available separately as `residual_history`.
- [x] CI fixtures do not require writing Poincare plots, FAMUS output files, or
      VTK files.

GPU-later acceptance:

- [ ] Re-run the same immutable grid fixture on JAX GPU.
- [ ] Compare JAX GPU to JAX CPU for payload-preserving algorithm outputs.
- [ ] Keep CPU/C++ vs JAX CPU as the oracle; GPU success is a third-lane
      consistency check, not a replacement oracle.

## Wave 4 - Wireframe Examples

Source examples:

- `examples/2_Intermediate/wireframe_rcls_basic.py`
- `examples/2_Intermediate/wireframe_rcls_with_ports.py`
- `examples/2_Intermediate/wireframe_gsco_modular.py`
- `examples/2_Intermediate/wireframe_gsco_sector_saddle.py`
- `examples/3_Advanced/wireframe_gsco_multistep.py`

Context:

- CPU path: `ToroidalWireframe`, `WireframeField`, and public
  `optimize_wireframe`.
- JAX path: `WireframeFieldJAX`, `bnorm_obj_matrices_jax`,
  `rcls_wireframe_jax`, `gsco_wireframe_jax`, and `optimize_wireframe_jax`.

Rationale:

- Wireframe examples now have a direct JAX solve wrapper and a field wrapper.
- They are high-value because they exercise C++ wireframe field kernels,
  constraint matrices, and optimization wrappers with real example surfaces.

Implementation checklist:

- [x] Add fixture `wireframe_rcls_basic_fixed_state`.
- [x] Add reduced partial fixture `wireframe_rcls_ports_constraint_gate`.
- [x] Add fixture `wireframe_gsco_modular_fixed_state`.
- [x] Add reduced pass fixture `wireframe_gsco_sector_saddle_fixed_state`.
- [x] Add fixture `wireframe_gsco_multistep_reduced_diagnostic`.
- [x] Build CPU and JAX wireframes independently from the example surface and
      wireframe resolution.
- [x] Preserve the documented `optimize_wireframe(...)` input mode for each
      row:
  - [x] If the example uses a plasma surface and field objects, compare through
        the same `surf_plas`/field path.
  - [x] If the fixture precomputes `Amat` and `bvec`, compare CPU and JAX
        through the same frozen matrices.
  - [x] Do not compare a CPU `surf_plas` path against a JAX precomputed-matrix
        path in the same row.
- [x] Compare `WireframeField.B` and `WireframeField.dB_by_dX` against
      `WireframeFieldJAX` at representative plasma-surface points.
- [x] Compare `bnorm_obj_matrices` CPU outputs against
      `bnorm_obj_matrices_jax` for the same `surf_plas`, `ext_field`,
      target, and weighting.
- [x] Compare RCLS `f_B`, `f_R`, total `f`, and constraint satisfaction.
- [x] Do not claim raw RCLS current-vector identity unless a unique solution
      basis is proven; list nullspace-equivalent current-vector mismatch
      explicitly when fields/objectives agree.
- [x] Compare GSCO history fields where deterministic: `iter_hist`,
      `curr_hist`, `loop_hist`, `f_B_hist`, `f_S_hist`, `f_hist`, final `x`, and
      final loop count.
- [x] Record and preserve algorithm flags that change constraints:
  - [x] `assume_no_crossings`,
  - [x] `no_crossing`,
  - [x] `match_current`,
  - [x] current constraints and fixed/current-carrying segment masks.
- [x] Treat plotting, `to_vtk`, `make_plot_2d`, and Mayavi output as non-parity
      side effects.
- [x] For the multistep example, start with a reduced first-step diagnostic
      before claiming the full multistep procedure.

Acceptance criteria:

- [x] RCLS basic fixture reaches `partial` with native matrix/objective/field
      comparisons passing and raw current-vector identity listed as
      `RCLS_current_vector_nonunique_nullspace`.
- [x] RCLS ports fixture reaches `partial` with port collision constraints,
      poloidal-current constraints, constraint matrix shape/satisfaction, and
      fields compared, while raw current-vector identity remains listed as
      `RCLS_current_vector_nonunique_nullspace`.
- [x] Sector/saddle GSCO fixture reaches `pass` with TF-coil initial currents,
      toroidal break free-cell masks, final currents, histories, field B, and
      Bnormal compared.
- [x] At least one GSCO fixture reaches `pass` or `partial` with named
      unsupported side effects only.
- [x] The multistep GSCO row reaches `partial` for the reduced first-step
      diagnostic, with the mutating loop and post-processing stages named
      separately in `unsupported_components`.
- [x] Matrix, field, and optimization-result comparisons are separately
      recorded so a failing solve can be localized.
- [x] Constraint handling is tested by value, not by a tautological call to the
      same helper on both lanes.
- [x] No test imports the example scripts directly.

GPU-later acceptance:

- [ ] GPU row uses the same `Amat`/`bvec` and wireframe fixed state.
- [ ] JAX GPU and JAX CPU agree for field matrices and solve outputs.
- [ ] Any non-deterministic GSCO tie-breaking is pinned by explicit tie-break
      inputs or kept CPU-only until deterministic.

## Wave 5 - Tracing Examples

Source examples:

- `examples/1_Simple/tracing_fieldlines_QA.py`
- `examples/1_Simple/tracing_fieldlines_NCSX.py`
- `examples/1_Simple/tracing_particle.py`
- `examples/2_Intermediate/tracing_boozer.py`

Context:

- CPU path: `compute_fieldlines`, `trace_particles`,
  `trace_particles_starting_on_curve`, and `trace_particles_boozer`.
- JAX path: JAX fieldline, guiding-center, full-orbit, and Boozer
  guiding-center drivers routed through JAX-native fields.
- Existing tests already cover low-level endpoint/event parity for analytic
  and wrapper routes; examples can add real example initial-state coverage.

Rationale:

- Tracing examples validate event buffers, endpoint states, and field wrapper
  compatibility under realistic example setup.
- Full plots and VTK outputs do not prove numerical parity, so this wave keeps
  the fixture surface reduced and deterministic.

Implementation checklist:

- [x] Add fixture `tracing_fieldlines_qa_reduced_endpoint`.
- [x] Add reduced partial fixture `tracing_fieldlines_ncsx_reduced_endpoint`.
- [x] Add support-gate row `tracing_particle_gc_vac_reduced_endpoint`.
- [x] Add support-gate row `tracing_boozer_gc_reduced_endpoint`.
- [x] Upgrade `tracing_particle_gc_vac_reduced_endpoint` to a partial fixture
      after adding the `InterpolatedFieldJAX` B + GradAbsB hook needed by the
      guiding-center route.
- [x] Upgrade `tracing_boozer_gc_reduced_endpoint` to a partial fixture after
      routing `trace_particles_boozer` through `InterpolatedBoozerFieldJAX`
      frozen-state scalar evaluators from cached boozmn data.
- [x] Use reduced `tmax`, fieldline count, particle count, and event-buffer
      sizes suitable for CI.
- [x] Pin trace-return semantics before comparing rows:
  - [x] `mode`,
  - [x] stopping criteria,
  - [x] `phis`,
  - [x] `forget_exact_path`.
- [x] Compare final trajectory state, valid trajectory length (`t_final`),
      status code, phi/zeta hit count, and hit rows. Adaptive accepted-step
      row counts remain diagnostic-only because CPU and JAX controllers may
      choose different step counts under the same event-time contract.
- [x] Compare event rows only where both lanes report the same event semantic.
- [x] Use `InterpolatedFieldJAX`, `BiotSavartJAX`,
      `BoozerRadialInterpolantJAX`, or `InterpolatedBoozerFieldJAX` only when
      the fixture can build a native JAX field without CPU callbacks.
- [x] Keep `particles_to_vtk`, `plot_poincare_data`, and Poincare image output
      outside the parity contract.

Acceptance criteria:

- [x] At least one fieldline example fixture reaches `pass` with native
      field and event-output comparisons passing, including the raw
      levelset-distance adapter used by the example script.
- [x] NCSX fieldline fixture reaches `pass` with native field values,
      endpoint, final time/status, phi-hit coordinates, and hit count compared,
      while raw levelset-distance stopping and skip callbacks are exercised.
- [x] Particle and Boozer guiding-center fixtures reach `partial` with native
      field and endpoint/event comparisons passing; the particle row is now
      `pass`, and the remaining tracing blocker is the Boozer example's
      external VMEC solve.
- [x] The artifact records ODE tolerances, `tmax`, initial conditions, event
      planes, and max-hit buffer size.
- [x] No fixture treats matching plot files as numerical parity.
- [x] CPU/C++ endpoint and event output is the oracle for JAX CPU.

GPU-later acceptance:

- [ ] JAX GPU uses the same JAX-native field payload as JAX CPU.
- [ ] Endpoint/event comparisons are blocked before writing the artifact.
- [ ] GPU rows remain separate from CPU rows because ODE branching can amplify
      small numerical differences.

## Wave 6 - Lower-Priority Blocked Examples

These are useful after the prerequisite support surface is complete.

### `examples/2_Intermediate/stage_two_optimization_finite_beta.py`

Context:

- Uses `VirtualCasing.B_external_normal` as a nonzero target normal field.
- Current harness blocker is deterministic fixture wiring for the virtual
  casing target.

Implementation checklist:

- [x] Check in or generate a deterministic cached virtual-casing target artifact
      under a documented fixture path.
- [x] Hash the target array and shape before comparing objectives.
- [x] Compare `SquaredFlux` CPU against `SquaredFluxJAX` with the target array.
- [x] Include fixed-state length identity penalties through `CurveLengthJAX`
      once native coverage is available in the fixture.

Acceptance criteria:

- [x] Fixture moves from `unsupported` to `partial`, then to `pass` after the
      curve-objective closeout.
- [x] Target-array hash is recorded and stable in CI.

### `examples/3_Advanced/stage_two_optimization_finitebuild.py`

Context:

- Uses multifilament finite-build coils plus flux and geometric penalties.
- Existing support gate says filament base curves expose native specs; missing
  item is a full composite constructor in the harness.

Implementation checklist:

- [x] Add a reduced multifilament fixture constructor mirroring the example.
- [x] Compare native-supported field, flux, length-penalty, and curve-distance
      subproblems.
- [x] Record all previously CPU-only penalties as compared JAX components once
      `CurveLengthJAX` and `CurveCurveDistanceJAX` are available.

Acceptance criteria:

- [x] Fixture moves from `unsupported` to `partial`, then to `pass` after the
      curve-objective closeout.
- [x] Every formerly unsupported penalty has a named component row on both CPU
      and JAX lanes.

### `examples/1_Simple/optimize_coil_position_orientation.py`

Context:

- Uses oriented curve construction for TF/windowpane coil position and
  orientation DOFs.
- Current status: `OrientedCurveXYZFourier` exposes an immutable native spec,
  and the reduced TF/windowpane fixed-state flux fixture passes.

Implementation checklist:

- [x] Implement or verify `OrientedCurveXYZFourier` immutable spec support in a
      separate code plan.
- [x] Add fixed-state flux comparison once the spec exists.
- [x] Do not emulate orientation DOFs by pre-materializing a CPU curve in the
      JAX lane.

Acceptance criteria:

- [x] Support gate moves from `unsupported` to `partial`/`pass`.
- [x] Active free-DOF mapping is explicitly recorded.

### `examples/3_Advanced/curves_CWS_example.py`

Context:

- Uses saved optimized CWS `BiotSavart` artifacts.
- Legacy `simsopt.load()` reconstruction of saved `CurveCWSFourier` artifacts
  is now root-fixed for the example JSON schema.

Implementation checklist:

- [x] Fix or work around the deserializer in the CPU loader path only if the
      loader contract is root-caused.
- [x] Add fixed-state local `SquaredFlux` and `BdotN` comparisons after load
      succeeds.

Acceptance criteria:

- [x] `cws_saved_local_flux_nfp2` and `cws_saved_local_flux_nfp3` move from
      `unsupported` to `pass`.

### `examples/2_Intermediate/strain_optimization.py`

Context:

- Uses framed-curve strain objectives.
- There is known historical test-quality risk around framed-curve wrapper
  tests, so this is not a first-wave parity proof.

Implementation checklist:

- [x] Replace any same-function or JAX-vs-JAX-only framed-curve tests before
      adding this example as pass evidence.
- [x] Add CPU oracle value/gradient checks for strain quantities.
- [x] Add fixed-state example fixture only after the independent oracle tests
      pass.

Acceptance criteria:

- [x] Register `strain_optimization_support_gate` so the blocked row is
      explicit in the harness.
- [x] Public `FrameRotationJAX` / `FramedCurveCentroidJAX` VJP wrapper path is
      exercised through the public strain penalty classes.
- [x] Mark the example row `pass`: the fixed-state fixture now checks
      CPU/public framed-curve strain arrays, native penalty values, and
      rotation-gradient values against the public JAX framed-curve wrappers.

### `examples/3_Advanced/coil_forces.py`

Context:

- Uses `LpCurveForce` and `B2Energy` force/energy penalties.
- The optimizer, VTK output, and CPU-only geometric penalties remain out of
  this row's native force/energy subproblem.

Implementation checklist:

- [x] Identify the native JAX force/energy support surface first:
      `LpCurveForceJAX` and `B2EnergyJAX` now resolve through the public
      `simsopt.field` lazy exports as explicit aliases for the existing
      JAX-kernel-backed wrappers.
- [x] Add CPU-only regression artifact if useful:
      `tests/field/test_force_item09_closeout.py` pins production-scale
      `LpCurveForce` value/gradient behavior with a finite-difference oracle and
      strict transfer-guard coverage for the forward path.
- [x] Add public-wrapper JAX parity evidence with independent oracle coverage:
      the reduced fixture builds independent CPU and JAX coil trees and
      compares `LpCurveForce`, `B2Energy`, per-component gradients, the weighted
      native subtotal, and independent CPU force/energy oracle values.

Acceptance criteria:

- [x] Register `coil_forces_support_gate` as a supported fixed-state
      force/energy row in the harness.
- [x] Force/energy CPU oracle coverage exists for the fixed-state row:
      `RegularizedCoil.force` integration gates force and a NumPy
      inductance-matrix loop gates energy.
- [x] The row is `pass` with 8 native comparisons and no unsupported
      components.

## GPU Expansion Gate Plumbing

GPU proof starts only after the corresponding CPU/C++ vs JAX CPU fixture passes
or has a precise `partial` verdict for unrelated CPU-only components.

Status boundary: this section tracks runner/schema plumbing only. It is not a
completed JAX GPU proof. The current artifacts in
`.artifacts/parity/20260514-example-expansion/` select only `cpu_cpp,jax_cpu`
and record `jax_gpu.status = runtime_required`; the Wave 3-5 GPU-later
acceptance boxes above stay unchecked until a real CUDA artifact is produced.

Implementation checklist:

- [x] Add an explicit `jax_gpu` lane to the artifact schema.
- [x] Store GPU metadata: platform, device kind, JAX version, CUDA runtime
      evidence available from JAX, x64 status, and transfer-guard settings.
- [x] Use the explicit GPU parity backend contract:
  - [x] `SIMSOPT_BACKEND_MODE=jax_gpu_parity`
  - [x] `SIMSOPT_JAX_PLATFORM=cuda`
  - [x] `SIMSOPT_EXAMPLE_PARITY_JAX_PLATFORM=cuda`
  - [x] `JAX_PLATFORMS=cuda`
  - [x] `JAX_ENABLE_X64=1`
- [x] Fail the GPU parity run when the active JAX backend is CPU.
- [x] Do not use `jax_gpu_fast` for first proof. `jax_gpu_fast` is a later
      performance lane after CPU/C++ vs JAX CPU vs JAX GPU parity is closed.
- [x] Record CUDA/JAX provenance for every GPU artifact row:
  - [x] JAX and jaxlib versions.
  - [x] JAX CUDA wheel/runtime line.
  - [x] CUDA runtime version visible to JAX.
  - [x] NVIDIA driver version.
  - [x] Device name.
  - [x] Compute capability.
- [x] Reuse exactly the same fixture input hash as the JAX CPU lane.
- [x] Compare JAX GPU against JAX CPU for each native-supported component by
      requiring `--baseline-json` and emitting `jax_cpu_vs_jax_gpu` comparisons
      for the same fixture input hash.
- [x] Keep CPU/C++ vs JAX CPU as the preservation oracle.
- [x] Add memory budget metadata for large PM, wireframe, and tracing fixtures.
- [x] Block all GPU arrays before writing comparison results.
- [x] Run strict transfer-guard checks on JIT-compiled kernels that claim no
      host roundtrip.
- [x] Keep reductions and ODE/tracing outputs under tolerance buckets that name
      the expected sensitivity.

CPU-local runner/schema acceptance criteria:

- [x] A GPU row cannot exist without a matching CPU fixture input hash.
- [x] A GPU row cannot upgrade a CPU `unsupported` fixture to pass.
- [x] GPU failures include first failing component, max difference, and device
      metadata.
- [x] GPU artifacts are separate from CPU artifacts and never overwrite CPU
      evidence.

## Implementation Order

Recommended order:

1. [x] Wave 0 harness preparation.
2. [x] Wave 1 `surf_vol_area` pass fixture.
3. [x] Wave 2 QFM residual/label fixture.
4. [x] Wave 4 `wireframe_rcls_basic` fixture.
5. [x] Wave 3 `permanent_magnet_simple` fixture.
6. [x] Wave 5 reduced fieldline fixture.
7. [x] Remaining PM/wireframe/tracing rows.
8. [x] Wave 6 blocked rows only after their prerequisite support surfaces land.
9. [x] GPU runner/schema plumbing only after CPU rows are stable.

The first three waves are intentionally small. They should expose any harness
schema mistakes before PM, wireframe, and tracing add larger fixtures.

## Definition Of Done

- [x] New doc or manifest entry lists every added fixture, source example, and
      verdict.
- [x] Every pass/partial row has CPU/C++-vs-JAX CPU numeric comparisons.
- [x] Every unsupported row has an exact blocker and no fake comparison.
- [x] Existing non-banana fixtures still pass or retain their expected
      unsupported verdicts.
- [x] Focused tests cover the new fixture builders and verdict semantics.
- [x] A full expanded CPU command is documented and writes a JSON artifact.
- [x] No GPU proof is claimed until real GPU artifacts exist.
- [x] No tolerance is loosened without a separate rationale and review.
- [x] No test relies on output files, plots, logging, or script import side
      effects as parity evidence.

## Proposed Commands

CPU expanded fixture tests:

```bash
conda run -n jax-0.9.2 python -c \
    "import jax, jaxlib; print(jax.__version__, jaxlib.__version__)"

JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
    conda run -n jax-0.9.2 python -m pytest tests/integration/test_non_banana_example_cpp_jax_cpu_parity.py -v
```

CPU expanded artifact:

```bash
conda run -n jax-0.9.2 python -c \
    "import jax, jaxlib; print(jax.__version__, jaxlib.__version__)"

JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
    conda run -n jax-0.9.2 python benchmarks/non_banana_example_cpp_jax_cpu_parity.py \
        --fixtures all \
        --lanes cpu_cpp,jax_cpu \
        --git-sha "$(git rev-parse HEAD)" \
        --dirty-policy record \
        --output-json .artifacts/parity/20260514-example-expansion/all-fixtures.json
```

Later GPU artifact, after CPU rows pass:

The runner now supports a separate CUDA follow-up lane. It must be launched in a
CUDA process with the explicit parity environment and the CPU artifact from the
same fixture inputs. The GPU artifact remains unproven until this command runs
on real NVIDIA/CUDA hardware.

```bash
SIMSOPT_BACKEND_MODE=jax_gpu_parity \
SIMSOPT_JAX_PLATFORM=cuda \
SIMSOPT_EXAMPLE_PARITY_JAX_PLATFORM=cuda \
JAX_PLATFORMS=cuda \
JAX_ENABLE_X64=1 \
conda run -n jax-0.9.2 python -c \
    "import jax, jaxlib; print(jax.__version__, jaxlib.__version__)"

SIMSOPT_BACKEND_MODE=jax_gpu_parity \
SIMSOPT_JAX_PLATFORM=cuda \
SIMSOPT_EXAMPLE_PARITY_JAX_PLATFORM=cuda \
JAX_PLATFORMS=cuda \
JAX_ENABLE_X64=1 \
conda run -n jax-0.9.2 python benchmarks/non_banana_example_cpp_jax_cpu_parity.py \
    --fixtures all-supported \
    --lanes cpu_cpp,jax_gpu \
    --baseline-json .artifacts/parity/20260514-example-expansion/all-supported-cpu.json \
    --git-sha "$(git rev-parse HEAD)" \
    --dirty-policy record \
    --output-json .artifacts/parity/20260514-example-expansion/all-supported-gpu.json
```

The local macOS CPU environment is still not valid GPU proof; this command must
run on a CUDA-enabled host. The current hardware blocker and exact closeout
boundary are recorded in
`.artifacts/parity/20260514-example-expansion/GPU_ACCEPTANCE_BLOCKER.md`.
