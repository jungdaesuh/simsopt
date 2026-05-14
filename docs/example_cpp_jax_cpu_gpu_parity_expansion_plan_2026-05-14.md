# Example C++/JAX CPU Parity Expansion Plan - 2026-05-14

Status: planning document for expanding example-derived parity evidence beyond
the existing non-banana fixture set. This document is intentionally a plan, not
a pass claim. The current CPU-only evidence remains in
`docs/non_banana_example_cpp_jax_cpu_parity_results_2026-05-12.md`.

Current trust chain:

```text
SIMSOPT CPU/C++ behavior -> JAX CPU matches at the same state -> later JAX GPU matches JAX CPU and preserves the same CPU/C++ oracle contract
```

The immediate implementation target is CPU/C++ vs JAX CPU parity. GPU proof is
a later gate and must not be inferred from JAX CPU success.

## Requirements

- [ ] Reuse the existing example parity harness shape before adding a new
      benchmark framework.
- [ ] Treat upstream SIMSOPT CPU/C++ behavior as the oracle.
- [ ] Compare fixed states first: same geometry, quadrature, field points,
      target arrays, free/fixed DOFs, currents, solver parameters, and input
      vectors.
- [ ] Compare JAX CPU against CPU/C++ before comparing any JAX GPU lane.
- [ ] Keep full example optimizer trajectories as diagnostics unless the
      optimizer route itself is the component under test.
- [ ] Avoid importing example scripts that execute optimizers, write VTK files,
      generate plots, or mutate global state at import time.
- [ ] Reconstruct fixture state from the same inputs the examples use, with
      side-effect-free fixture builders.
- [ ] Record every unsupported component by exact component name. Do not fill
      a JAX lane with CPU code and call it parity.
- [ ] Keep mixed CPU/JAX diagnostic rows separate from native-JAX pass rows.
- [ ] Require `JAX_ENABLE_X64=1` and `JAX_PLATFORMS=cpu` for CPU parity runs.
- [ ] Synchronize JAX timing or completion-sensitive checks with
      `block_until_ready()` or `jax.block_until_ready(...)`.
- [ ] Use transfer-guard or strict target-lane checks for GPU-readiness probes
      where the code claims no hidden host roundtrip.
- [ ] Preserve the dirty-worktree policy: record dirty metadata, but do not
      stage unrelated user changes.

## Current Context

Existing evidence:

- [x] `benchmarks/non_banana_example_cpp_jax_cpu_parity.py` and
      `benchmarks/non_banana_example_parity_fixtures.py` already provide the
      fixture/reporting pattern to extend.
- [x] `tests/integration/test_non_banana_example_cpp_jax_cpu_parity.py` already
      enforces verdict semantics for `pass`, `partial`, and `unsupported`.
- [x] Existing non-banana rows cover Stage-II, Boozer, finite-beta support
      gates, finite-build support gates, CWS deserializer gates, and QFM as an
      unsupported fixture.
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

- [ ] `QfmSurface` remains host solver orchestration over mutable
      `surface.x`; prove `QfmResidualJAX` and label metrics, not the SciPy
      QFM optimizer path.
- [ ] The live mutable `PermanentMagnetGrid` host loop remains host
      orchestration; prove immutable `PermanentMagnetGridJAX` payloads and
      explicit algorithms, not a generic CPU-style dispatcher.
- [ ] VMEC/SPEC/QSC examples remain external-solver workflows unless a fixture
      extracts a fixed native-JAX-supported subproblem.
- [ ] VTK, plotting, logging, and file-output side effects are not parity
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

- [ ] JAX x64 is controlled by `jax_enable_x64` or `JAX_ENABLE_X64`; parity
      runs must enable it before importing JAX and before creating arrays.
- [ ] JAX defaults away from x64, so a run without x64 is a smoke test only.
- [ ] JAX dispatch is asynchronous; measurements and completion-sensitive
      evidence must explicitly wait with `block_until_ready()`.
- [ ] JAX transfer guard can disallow implicit transfers; strict GPU-readiness
      tests should use this to catch hidden host/device movement.
- [ ] JAX GPU parity requires a supported JAX CUDA wheel/runtime combination
      and an active CUDA backend. A GPU row must fail when JAX is running on
      CPU.
- [ ] JAX CUDA rows must record which supported install line they use:
  - [ ] CUDA 12 wheel/runtime line: NVIDIA driver at least 525, CUDA at least
        12.1, and GPU compute capability at least SM 5.2.
  - [ ] CUDA 13 wheel/runtime line: NVIDIA driver at least 580, CUDA at least
        13.0, and GPU compute capability at least SM 7.5.
- [ ] SIMSOPT `MagneticField` comparisons must respect the documented
      `set_points(...)` contract: set identical point arrays immediately before
      each CPU/C++ and JAX field evaluation, using independent field instances
      per lane.
- [ ] SIMSOPT wireframe rows must preserve the documented
      `optimize_wireframe(...)` input mode. Compare `surf_plas` paths against
      `surf_plas` paths, and compare precomputed `Amat`/`bvec` paths against
      the same frozen `Amat`/`bvec` paths.
- [ ] SIMSOPT tracing rows must pin `mode`, stopping criteria, `phis`, and
      `forget_exact_path`; compare only endpoint/event arrays returned by that
      exact trace contract.
- [ ] CUDA documentation provenance must use the current CUDA Programming Guide
      URL above. The old CUDA C++ Programming Guide index URL is legacy.

## Shared Harness Contract

Extend the existing harness rather than creating a parallel system.

- [ ] Add new `FixtureSpec` rows to
      `benchmarks/non_banana_example_parity_fixtures.py` unless a fixture family
      requires a small helper module.
- [ ] Keep one JSON schema for all example parity rows.
- [ ] Keep the existing dataclass boundaries as the source of truth:
  - [ ] `FixtureSpec`: `fixture_id`, `source_example`, `classification`,
        `classification_reason`, `inputs`, and `fixture_kind`.
  - [ ] `FixtureBuild`: `spec`, `cpu_lane`, `jax_lane`,
        `unsupported_components`, optional native-subproblem evaluators, and
        optional `x0`.
  - [ ] Output `FixtureResult`: `fixture_inputs`, `dof_contract`,
        `native_spec_contract`, `lanes`, `comparisons`,
        `unsupported_components`, diagnostics, `verdict`, `passed`,
        `failures`, and optional `error`.
  - [ ] Run artifact: top-level `metadata` plus the fixture result list.
- [ ] Add family-specific comparison helpers only when the existing comparison
      shape cannot express the result.
- [ ] Store CPU lane and JAX lane artifacts in separate objects. Do not share
      mutable surfaces, fields, grids, or wireframe instances between lanes.
- [ ] Add one integration test per new fixture verdict and one aggregate
      smoke test for the expanded fixture set.

Required comparison fields:

- [ ] `quantity`
- [ ] `component`
- [ ] `source_example`
- [ ] `cpu_cpp_value`
- [ ] `jax_cpu_value`
- [ ] `tolerance_bucket`
- [ ] `rtol`
- [ ] `atol`
- [ ] `max_abs_diff`
- [ ] `max_rel_diff`
- [ ] `argmax_index`
- [ ] `verdict`

Required metadata:

- [ ] Git SHA and dirty-tree summary.
- [ ] `JAX_ENABLE_X64` and active platform.
- [ ] JAX version, backend, devices, and device kind.
- [ ] For GPU rows: JAX CUDA wheel/runtime line, CUDA runtime version visible
      to JAX, NVIDIA driver version, device name, and compute capability.
- [ ] SIMSOPT example source path.
- [ ] Input files, generated fixture hashes, quadrature sizes, and seeds.
- [ ] Explicit statement of whether the row is CPU-only, GPU-ready, or
      GPU-proven.

No-fake-test rules:

- [ ] A pass row must compare CPU/C++ oracle output against independently built
      JAX CPU output.
- [ ] JAX-vs-JAX comparisons may appear only as transform or GPU follow-up
      checks, never as the CPU/C++ preservation oracle.
- [ ] A test that compares a function to itself is not parity evidence.
- [ ] A test that only asserts the harness records an unsupported verdict is a
      support-gate test, not a port-correctness pass.
- [ ] Tolerances must come from the existing validation ladder or be justified
      in the fixture row.
- [ ] Do not loosen tolerances in the same commit that introduces a failing
      comparison.

## Wave 0 - Harness Preparation

Rationale: the next examples span surface scalars, QFM residuals,
permanent-magnet optimization, wireframe solves, and tracing. A small set of
shared helpers avoids one-off tests and keeps verdict semantics consistent.

Implementation checklist:

- [ ] Add fixture-family constants for `surface_scalar`, `qfm`, `pm`,
      `wireframe`, and `tracing`.
- [ ] Add a helper for recording fixed-state input hashes for dense arrays.
- [ ] Add a helper for recording device/completion metadata after JAX values are
      blocked.
- [ ] Add a helper that marks a fixture `gpu_ready=false` until a real GPU lane
      artifact exists.
- [ ] Add version-probe guidance to every runnable command:
      `conda run -n jax-0.9.2 python -c "import jax, jaxlib; print(jax.__version__, jaxlib.__version__)"`.
- [ ] Add a platform-aware GPU runner extension before documenting an
      executable GPU artifact command:
  - [ ] Add a `--lanes` CLI option or an equivalent explicit lane selector.
  - [ ] Plumb lane selection through `run_fixtures`.
  - [ ] Make `jax_gpu` rows fail if `jax.devices()[0].platform` is not `cuda`
        or `gpu`.
  - [ ] Keep the current CPU-only runner contract intact for CPU parity rows.
- [ ] Add a test that every new fixture has a non-empty `source_example`,
      `rationale`, and `acceptance_criteria`.
- [ ] Add a test that every `pass` or `partial` row has at least one
      CPU/C++-vs-JAX CPU numeric comparison.
- [ ] Add a test that no row uses a `cpu_fallback`, `host_fallback`, or
      `jax_self_reference` verdict marker.

Acceptance criteria:

- [ ] Existing 2026-05-12 fixtures still produce the same verdict categories.
- [ ] The existing fixture regression suite is run before and after adding new
      fixture families.
- [ ] New schema fields are present for old and new fixtures.
- [ ] Fixture-level tests fail if a pass row has zero CPU/C++ oracle
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

- [ ] Add fixture `surface_area_volume_simple`.
- [ ] Rebuild the example's initial `SurfaceRZFourier` state in a fixture
      builder instead of importing the script.
- [ ] Build independent CPU and JAX surfaces from identical DOFs.
- [ ] Compare `Area(s).J()` against `AreaJAX(s_jax).J()`.
- [ ] Compare `Volume(s).J()` against `VolumeJAX(s_jax).J()`.
- [ ] Compare `Area(s).dJ()` against `AreaJAX(s_jax).dJ()`.
- [ ] Compare `Volume(s).dJ()` against `VolumeJAX(s_jax).dJ()`.
- [ ] Add deterministic perturbed-surface checks for at least two nontrivial
      DOF perturbations.
- [ ] Record the surface resolution, active DOF names, and target area/volume
      values from the example.
- [ ] Keep any optimizer final-state check diagnostic-only until fixed-state
      parity passes.

Acceptance criteria:

- [ ] Fixture verdict is `pass`.
- [ ] Value and gradient comparisons pass under the `direct_kernel` or
      `derivative_heavy` tolerance bucket already used for surface objectives.
- [ ] The fixture fails if the CPU and JAX lanes share the same surface object.
- [ ] The JSON artifact reports zero unsupported components.
- [ ] Later GPU gate can reuse the same fixture without changing fixture inputs.

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

- [ ] Replace the current `qfm_surface` unsupported builder with a fixed-state
      residual/label builder.
- [ ] Rebuild the example's circular coils and initial surface independently for
      CPU and JAX lanes.
- [ ] Compare `QfmResidual(s, bs).J()` against `QfmResidualJAX(s_jax, bs_jax).J()`.
- [ ] Compare `QfmResidual.dJ()` against `QfmResidualJAX.dJ()` in the same
      surface DOF basis.
- [ ] Compare the label metrics used by the example: `Volume`, `Area`, and
      `ToroidalFlux`.
- [ ] Add separate fixed-state rows for the initial state and one
      post-constraint target state if the latter can be reconstructed without
      running SciPy in the JAX lane.
- [ ] Keep `QfmSurface.minimize_*` results as CPU-only diagnostics unless a
      separate plan ports the solver orchestration.

Acceptance criteria:

- [ ] Fixture verdict changes from `unsupported` to `pass` or `partial`.
- [ ] Any remaining unsupported item is named as `QfmSurface_host_solver`, not
      hidden inside the QFM residual comparison.
- [ ] Residual vector, residual norm, and surface gradient comparisons pass.
- [ ] Area/Volume/ToroidalFlux label comparisons pass under their existing
      tolerance buckets.
- [ ] No test imports `examples/1_Simple/qfm.py` as a module.

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

- [ ] Add fixture `pm_simple_fixed_state_gpmo_baseline`.
- [ ] Add fixture `pm_qa_fixed_state_gpmo_arbvec_or_multi` after choosing the
      exact algorithm used by the example.
- [ ] Add support-gate or pass fixtures for `pm_muse_famus` and
      `pm_pm4stell_backtracking`, depending on whether their input files are
      available in the repo and deterministic under CI.
- [ ] Preserve the public example algorithm selection before creating each
      fixture:
  - [ ] `permanent_magnet_simple.py` default `GPMO(pm_opt, **kwargs)` maps to
        the documented baseline algorithm.
  - [ ] MUSE and PM4Stell rows that use `ArbVec_backtracking` must stay in that
        algorithm family.
  - [ ] No parity row may compare baseline CPU history to backtracking JAX
        history.
- [ ] Build the CPU `PermanentMagnetGrid` from the example input files.
- [ ] Convert the CPU grid to `PermanentMagnetGridJAX.from_cpu(...)` or an
      equivalent fixed-state constructor.
- [ ] Compare immutable grid payload fields: geometry arrays, normal field
      vector, maxima, active mask, and operator shapes.
- [ ] Compare CPU `DipoleField` against `DipoleFieldJAX` for final and
      selected intermediate moment arrays at representative surface points.
- [ ] Compare the explicit algorithm outputs against the CPU algorithm variant:
      residual histories, selected dipoles, selected components/vectors, final
      moment array, and final normal-field residual.
- [ ] Record the exact GPMO algorithm variant. Do not test the generic JAX
      dispatcher because none is exported by design.
- [ ] Use reduced deterministic `K` values for CI and keep full example-size
      runs as optional artifacts.

Acceptance criteria:

- [ ] At least one permanent-magnet fixture reaches `pass` with CPU/C++ vs JAX
      CPU comparisons.
- [ ] The fixture JSON names the algorithm variant and does not claim generic
      `GPMO` dispatcher parity.
- [ ] `DipoleField` vs `DipoleFieldJAX` field comparisons pass for the selected
      final moment state.
- [ ] Algorithm histories match within the same tolerance contract used by the
      existing permanent-magnet JAX tests, including CPU `R2_history`,
      `Bn_history`, and `m_history` counterparts where the public CPU function
      returns them.
- [ ] CI fixtures do not require writing Poincare plots, FAMUS output files, or
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

- [ ] Add fixture `wireframe_rcls_basic_fixed_state`.
- [ ] Add fixture `wireframe_rcls_ports_constraint_gate`.
- [ ] Add fixture `wireframe_gsco_modular_fixed_state`.
- [ ] Add fixture `wireframe_gsco_sector_saddle_fixed_state`.
- [ ] Add fixture `wireframe_gsco_multistep_reduced_diagnostic`.
- [ ] Build CPU and JAX wireframes independently from the example surface and
      wireframe resolution.
- [ ] Preserve the documented `optimize_wireframe(...)` input mode for each
      row:
  - [ ] If the example uses a plasma surface and field objects, compare through
        the same `surf_plas`/field path.
  - [ ] If the fixture precomputes `Amat` and `bvec`, compare CPU and JAX
        through the same frozen matrices.
  - [ ] Do not compare a CPU `surf_plas` path against a JAX precomputed-matrix
        path in the same row.
- [ ] Compare `WireframeField.B` and `WireframeField.dB_by_dX` against
      `WireframeFieldJAX` at representative plasma-surface points.
- [ ] Compare `bnorm_obj_matrices` CPU outputs against
      `bnorm_obj_matrices_jax` for the same `surf_plas`, `ext_field`,
      target, and weighting.
- [ ] Compare RCLS output current vector, `f_B`, `f_R`, total `f`, and constraint
      satisfaction.
- [ ] Compare GSCO history fields where deterministic: `iter_hist`,
      `curr_hist`, `loop_hist`, `f_B_hist`, `f_S_hist`, `f_hist`, final `x`, and
      final loop count.
- [ ] Record and preserve algorithm flags that change constraints:
  - [ ] `assume_no_crossings`,
  - [ ] `no_crossing`,
  - [ ] `match_current`,
  - [ ] current constraints and fixed/current-carrying segment masks.
- [ ] Treat plotting, `to_vtk`, `make_plot_2d`, and Mayavi output as non-parity
      side effects.
- [ ] For the multistep example, start with a reduced first-step diagnostic
      before claiming the full multistep procedure.

Acceptance criteria:

- [ ] RCLS basic fixture reaches `pass`.
- [ ] At least one GSCO fixture reaches `pass` or `partial` with named
      unsupported side effects only.
- [ ] Matrix, field, and optimization-result comparisons are separately
      recorded so a failing solve can be localized.
- [ ] Constraint handling is tested by value, not by a tautological call to the
      same helper on both lanes.
- [ ] No test imports the example scripts directly.

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

- [ ] Add fixture `tracing_fieldlines_qa_reduced_endpoint`.
- [ ] Add fixture `tracing_fieldlines_ncsx_reduced_endpoint`.
- [ ] Add fixture `tracing_particle_gc_vac_reduced_endpoint`.
- [ ] Add fixture `tracing_boozer_gc_reduced_endpoint`.
- [ ] Use reduced `tmax`, fieldline count, particle count, and event-buffer
      sizes suitable for CI.
- [ ] Pin trace-return semantics before comparing rows:
  - [ ] `mode`,
  - [ ] stopping criteria,
  - [ ] `phis`,
  - [ ] `forget_exact_path`.
- [ ] Compare final trajectory state, valid trajectory length, status code,
      phi/zeta hit count, and hit rows.
- [ ] Compare event rows only where both lanes report the same event semantic.
- [ ] Use `InterpolatedFieldJAX`, `BiotSavartJAX`, or
      `BoozerRadialInterpolantJAX` only when the fixture can build a native JAX
      field without CPU callbacks.
- [ ] Keep `particles_to_vtk`, `plot_poincare_data`, and Poincare image output
      outside the parity contract.

Acceptance criteria:

- [ ] At least one fieldline example fixture reaches `pass`.
- [ ] At least one particle/guiding-center fixture reaches `pass` or records a
      precise unsupported native-field reason.
- [ ] The artifact records ODE tolerances, `tmax`, initial conditions, event
      planes, and max-hit buffer size.
- [ ] No fixture treats matching plot files as numerical parity.
- [ ] CPU/C++ endpoint and event output is the oracle for JAX CPU.

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

- [ ] Check in or generate a deterministic cached virtual-casing target artifact
      under a documented fixture path.
- [ ] Hash the target array and shape before comparing objectives.
- [ ] Compare `SquaredFlux` CPU against `SquaredFluxJAX` with the target array.
- [ ] Keep length terms in `unsupported_components` until native coverage is
      available in the fixture.

Acceptance criteria:

- [ ] Fixture moves from `unsupported` to `partial`.
- [ ] Target-array hash is recorded and stable in CI.

### `examples/3_Advanced/stage_two_optimization_finitebuild.py`

Context:

- Uses multifilament finite-build coils plus flux and geometric penalties.
- Existing support gate says filament base curves expose native specs; missing
  item is a full composite constructor in the harness.

Implementation checklist:

- [ ] Add a reduced multifilament fixture constructor mirroring the example.
- [ ] Compare native-supported field and flux subproblems.
- [ ] Record all CPU-only penalties explicitly.

Acceptance criteria:

- [ ] Fixture moves from `unsupported` to `partial`.
- [ ] Every unsupported penalty has a named component row.

### `examples/1_Simple/optimize_coil_position_orientation.py`

Context:

- Uses oriented curve construction for TF/windowpane coil position and
  orientation DOFs.
- Current blocker is immutable native-spec support for
  `OrientedCurveXYZFourier`.

Implementation checklist:

- [ ] Implement or verify `OrientedCurveXYZFourier` immutable spec support in a
      separate code plan.
- [ ] Add fixed-state flux comparison once the spec exists.
- [ ] Do not emulate orientation DOFs by pre-materializing a CPU curve in the
      JAX lane.

Acceptance criteria:

- [ ] Support gate moves from `unsupported` to `partial`.
- [ ] Active free-DOF mapping is explicitly recorded.

### `examples/3_Advanced/curves_CWS_example.py`

Context:

- Uses saved optimized CWS `BiotSavart` artifacts.
- Current blocker is upstream `simsopt.load()` reconstruction of
  `CurveCWSFourier`.

Implementation checklist:

- [ ] Fix or work around the deserializer in the CPU loader path only if the
      loader contract is root-caused.
- [ ] Add fixed-state local `SquaredFlux` and `BdotN` comparisons after load
      succeeds.

Acceptance criteria:

- [ ] `cws_saved_local_flux_nfp2` and `cws_saved_local_flux_nfp3` move from
      `unsupported` to `pass`.

### `examples/2_Intermediate/strain_optimization.py`

Context:

- Uses framed-curve strain objectives.
- There is known historical test-quality risk around framed-curve wrapper
  tests, so this is not a first-wave parity proof.

Implementation checklist:

- [ ] Replace any same-function or JAX-vs-JAX-only framed-curve tests before
      adding this example as pass evidence.
- [ ] Add CPU oracle value/gradient checks for strain quantities.
- [ ] Add fixed-state example fixture only after the independent oracle tests
      pass.

Acceptance criteria:

- [ ] No strain example row can be marked `pass` until independent CPU/C++
      oracle tests exist.

### `examples/3_Advanced/coil_forces.py`

Context:

- Uses `LpCurveForce` and `B2Energy` force/energy penalties.
- This is a good future CPU regression target but not a clean current JAX
  example proof.

Implementation checklist:

- [ ] Identify the native JAX force/energy support surface first.
- [ ] Add CPU-only regression artifact if useful.
- [ ] Add JAX parity only after force/energy wrappers have independent oracle
      tests.

Acceptance criteria:

- [ ] No JAX pass claim until force/energy CPU/C++ oracle coverage exists.

## GPU Expansion Gate

GPU proof starts only after the corresponding CPU/C++ vs JAX CPU fixture passes
or has a precise `partial` verdict for unrelated CPU-only components.

Implementation checklist:

- [ ] Add an explicit `jax_gpu` lane to the artifact schema.
- [ ] Store GPU metadata: platform, device kind, JAX version, CUDA runtime
      evidence available from JAX, x64 status, and transfer-guard settings.
- [ ] Use the explicit GPU parity backend contract:
  - [ ] `SIMSOPT_BACKEND_MODE=jax_gpu_parity`
  - [ ] `SIMSOPT_JAX_PLATFORM=cuda`
  - [ ] `JAX_PLATFORMS=cuda`
  - [ ] `JAX_ENABLE_X64=1`
- [ ] Fail the GPU parity run when the active JAX backend is CPU.
- [ ] Do not use `jax_gpu_fast` for first proof. `jax_gpu_fast` is a later
      performance lane after CPU/C++ vs JAX CPU vs JAX GPU parity is closed.
- [ ] Record CUDA/JAX provenance for every GPU artifact row:
  - [ ] JAX and jaxlib versions.
  - [ ] JAX CUDA wheel/runtime line.
  - [ ] CUDA runtime version visible to JAX.
  - [ ] NVIDIA driver version.
  - [ ] Device name.
  - [ ] Compute capability.
- [ ] Reuse exactly the same fixture input hash as the JAX CPU lane.
- [ ] Compare JAX GPU against JAX CPU for each native-supported component.
- [ ] Keep CPU/C++ vs JAX CPU as the preservation oracle.
- [ ] Add memory budget metadata for large PM, wireframe, and tracing fixtures.
- [ ] Block all GPU arrays before writing comparison results.
- [ ] Run strict transfer-guard checks on JIT-compiled kernels that claim no
      host roundtrip.
- [ ] Keep reductions and ODE/tracing outputs under tolerance buckets that name
      the expected sensitivity.

Acceptance criteria:

- [ ] A GPU row cannot exist without a matching CPU fixture input hash.
- [ ] A GPU row cannot upgrade a CPU `unsupported` fixture to pass.
- [ ] GPU failures include first failing component, max difference, and device
      metadata.
- [ ] GPU artifacts are separate from CPU artifacts and never overwrite CPU
      evidence.

## Implementation Order

Recommended order:

1. [ ] Wave 0 harness preparation.
2. [ ] Wave 1 `surf_vol_area` pass fixture.
3. [ ] Wave 2 QFM residual/label fixture.
4. [ ] Wave 4 `wireframe_rcls_basic` fixture.
5. [ ] Wave 3 `permanent_magnet_simple` fixture.
6. [ ] Wave 5 reduced fieldline fixture.
7. [ ] Remaining PM/wireframe/tracing rows.
8. [ ] Wave 6 blocked rows only after their prerequisite support surfaces land.
9. [ ] GPU rows only after CPU rows are stable.

The first three waves are intentionally small. They should expose any harness
schema mistakes before PM, wireframe, and tracing add larger fixtures.

## Definition Of Done

- [ ] New doc or manifest entry lists every added fixture, source example, and
      verdict.
- [ ] Every pass/partial row has CPU/C++-vs-JAX CPU numeric comparisons.
- [ ] Every unsupported row has an exact blocker and no fake comparison.
- [ ] Existing non-banana fixtures still pass or retain their expected
      unsupported verdicts.
- [ ] Focused tests cover the new fixture builders and verdict semantics.
- [ ] A full expanded CPU command is documented and writes a JSON artifact.
- [ ] No GPU proof is claimed until real GPU artifacts exist.
- [ ] No tolerance is loosened without a separate rationale and review.
- [ ] No test relies on output files, plots, logging, or script import side
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
        --git-sha "$(git rev-parse HEAD)" \
        --dirty-policy record \
        --output-json .artifacts/parity/20260514-example-expansion/all-cpu.json
```

Later GPU artifact, after CPU rows pass:

This is aspirational until the Wave 0 GPU runner extension exists. Do not run
the current CPU-only benchmark with GPU lanes; it does not accept `--lanes` and
it forces `JAX_PLATFORMS=cpu`.

```bash
# ASPIRATIONAL - runner extension required before this becomes executable.
SIMSOPT_BACKEND_MODE=jax_gpu_parity \
SIMSOPT_JAX_PLATFORM=cuda \
JAX_PLATFORMS=cuda \
JAX_ENABLE_X64=1 \
conda run -n jax-0.9.2 python -c \
    "import jax, jaxlib; print(jax.__version__, jaxlib.__version__)"
```

The executable GPU artifact command should be added only after the runner
supports a `jax_gpu` lane and runtime device rejection. The current CPU
benchmark runner is not valid GPU proof.
