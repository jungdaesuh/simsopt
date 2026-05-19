# JAX Parity Manifest

Base parity matrix snapshot as of 2026-04-10.

Boozer rows refreshed on 2026-05-05 after the CPU closure in
[`boozer_full_parity_plan_2026-05-04.md`](boozer_full_parity_plan_2026-05-04.md).
This refresh does not claim CUDA hardware parity.

Banana Stage 2 and single-stage JAX closure remains split between local CPU/JAX
evidence and CUDA evidence. Local non-CUDA surface/objective implementation
evidence closes CPU/JAX rows only; it does not close any CUDA status. CUDA rows
remain incomplete until current-tree GPU artifacts are captured by checked-in
local or CI proof entrypoints.

Exact parity means the mirrored JAX test runs with:

- `jax_enable_x64=True` before test arrays are created
- strict backend parity lanes
- explicit host materialization / device sync before parity assertions
- fixed seeded fixtures across CPU and JAX lanes

Solver-level parity stays contract-based: convergence success, residual norm,
final objective, and final physics quantities must match within the documented
acceptance envelope, but iterate-by-iterate identity is not required.

## Parity Test Matrix

This matrix is the SSOT for the mirrored parity surface requested by
`jax_parity_reduction_todos_2026-04-10.md`.

Current Boozer CPU closure is tracked in
[`boozer_full_parity_plan_2026-05-04.md`](boozer_full_parity_plan_2026-05-04.md),
including the explicit mutable-identity exclusions and the current pass/fail
watermark.

| Upstream test | JAX test | Status | Notes |
| --- | --- | --- | --- |
| `tests/field/test_biotsavart.py` | `tests/field/test_biotsavart_jax_parity.py` | exact | Pure-kernel mirror for `A/B`, spatial derivatives, Hessians, and VJP identities. |
| `tests/objectives/test_fluxobjective.py` | `tests/objectives/test_fluxobjective_jax_parity.py` | contract-complete | Dedicated mirrored wrapper coverage for definitions, derivatives, target handling, degenerate normals, singular zero-field behavior, native-contract rejection, non-RZ surface cases, and mutation/layout guards. Value/gradient parity is exact where the upstream wrapper contract is defined. The bounded CPU-`nan` reproducer search found no finite-input reproducer against the repo-local simsoptpp build, and direct C++/JAX boundary tests now pin the documented degenerate contracts (`0.0` for zero-normal quadratic flux, `inf` for normalized/local zero-field singularities). |
| `tests/objectives/test_fluxobjective.py` | `tests/integration/test_stage2_jax.py` | partial | Integration coverage for mixed quadrature and native-spec rejection behavior complements the dedicated object-level parity file. This row intentionally remains partial after the flux/kernel reconciliation because it tracks Stage 2 integration scope, not low-level `integral_BdotN` boundary behavior. The target bundle now has fixed-state reporting parity, target-lane snapshot/accepted-partial reporting avoids the legacy distance culler, spec restart rehydration is covered, and the reduced `SIMSOPT_TARGET_LANE_STRICT=1` CPU/JAX run passes. CUDA evidence remains open under the hardware gate. |
| `tests/objectives/test_integral_bdotn_jax.py` | `tests/objectives/test_integral_bdotn_jax.py` | contract-complete | Exact on regular inputs, including direct `simsoptpp.integral_BdotN` parity. Documented degenerate contracts are covered for zero-normal quadratic flux (`0.0`) and normalized/local zero-field singularities (`inf`), and direct C++/JAX boundary tests pin those cases. The bounded CPU-`nan` reproducer search found no finite-input reproducer against the repo-local simsoptpp build. |
| `tests/geo/test_surface_rzfourier.py` | `tests/geo/test_surface_rzfourier_jax.py` | partial | Strict tolerance-based CPU/JAX parity for the RZ geometry/object API surface (`surface_spec`/`to_spec`, `*_jax`, DOF round-trips, gradients, loaders, and `copy`). Non-RZ geometry/object API and added surface-objective wrappers now have local non-CUDA evidence in the section below. Conditional VTK, label, and higher paired-point rows remain outside the banana contract unless the scope expands to claim them. |
| `tests/geo/test_surface_objectives.py::ToroidalFlux*` | `tests/geo/test_surface_objectives_jax.py` | complete | Upstream `ToroidalFlux` value, first-derivative, Hessian, coil-derivative, constant-in-index, and Taylor checks are mirrored across the surface-type and `stellsym` sweep with tolerance-based CPU/JAX parity. Scalar value tolerances are owned by the `derivative_heavy` ladder lane; this is tolerance-based parity, not exact arithmetic parity. |
| `tests/geo/test_boozersurface.py` | `tests/geo/test_boozersurface_jax.py` | cpu-contract-complete | Boozer CPU parity closure is complete for math kernels, solver results, guard behavior, derivatives/adjoints, and supported public APIs. Parity remains contract-based: solved-state quality and public result semantics are the oracle, not mutable object identity or iterate-by-iterate solver trajectory. |
| `tests/geo/test_single_stage_example.py` and single-stage Boozer integration slices | `tests/integration/test_single_stage_jax_cpu_reference.py` | cpu-contract-complete | Dedicated CPU/JAX Boozer integration tests compare convergence success, residual norms, final solver objective, and final physics quantities (`iota`, `G`, label value/error, anchored axis-z). CUDA Boozer parity is not claimed by this CPU closure and still requires the optional hardware validation gate in the Boozer plan. |

## Surface And Objective Non-CUDA Evidence

This section is documentary. The machine-checked banana product inventory is the
next section.

| Scope | Evidence | CPU/JAX status | CUDA status | Remaining carve-out |
| --- | --- | --- | --- | --- |
| RZ and non-RZ surface geometry, derivatives, scalar metrics, forms, curvatures, and explicit heavy Hessian APIs | `tests/geo/test_surface_rzfourier_jax.py`<br>`tests/geo/test_surface_fourier_jax.py`<br>`tests/geo/test_surface_taylor.py` | local non-CUDA covered for implemented Set B rows | not claimed | Conditional VTK/file-output rows and higher paired-point APIs remain unclaimed unless full legacy I/O scope explicitly requires them. |
| Non-RZ object API breadth for `SurfaceXYZFourier` and `SurfaceXYZTensorFourier` | `tests/geo/test_surface_fourier_jax.py`<br>`tests/geo/test_surface_xyzfourier.py` | local non-CUDA covered for copy, JSON/GSON, conversion, fitting, scaling, and normal-extension paths | not claimed | VTK smoke coverage remains conditional on an I/O parity claim. |
| Added surface objective wrappers and helper APIs | `tests/geo/test_surface_objectives_jax.py`<br>`tests/geo/test_surface_objectives.py` | local non-CUDA covered for `AspectRatioJAX`, `MajorRadiusJAX`, `QfmResidualJAX`, and `PrincipalCurvatureJAX` | not claimed | `aspect_ratio` Boozer label support remains conditional on full label-test parity scope. |

## Banana Coverage Inventory

This table is the machine-checked banana parity inventory. Keep unsupported
families as explicit carve-outs rather than folding them into a complete row.
For the banana optimization product surface, all required non-CUDA C++ /
`simsoptpp` lanes in this table are covered by the listed CPU/JAX parity
contracts. Any remaining non-P5 `partial` status is limited to Python surface
object/API breadth outside banana scope, not to an uncovered required C++ lane.

| Coverage row | Upstream Python test file | Upstream C++ implementation file | JAX implementation file | JAX parity test file | Tolerance lane | CPU/JAX status | CUDA status | Known carve-out |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Biot-Savart field kernels | `tests/field/test_biotsavart.py` | `src/simsoptpp/biot_savart_impl.h` | `src/simsopt/jax_core/biotsavart.py` | `tests/field/test_biotsavart_jax_parity.py` | `direct_kernel` | exact | open under P5 | none |
| Flux objective and integral_BdotN | `tests/objectives/test_fluxobjective.py` | `src/simsoptpp/integral_BdotN.cpp` | `src/simsopt/objectives/fluxobjective_jax.py`<br>`src/simsopt/objectives/integral_bdotn_jax.py` | `tests/objectives/test_fluxobjective_jax_parity.py`<br>`tests/objectives/test_integral_bdotn_jax.py` | `direct_kernel` | contract-complete | open under P5 | none |
| Stage 2 target reporting and strict reduced run | `tests/objectives/test_fluxobjective.py` | `src/simsoptpp/integral_BdotN.cpp`<br>`src/simsoptpp/biot_savart_impl.h` | `src/simsopt/objectives/stage2_target_objective_jax.py`<br>`examples/single_stage_optimization/STAGE_2/banana_coil_solver.py` | `tests/integration/test_stage2_jax.py`<br>`tests/integration/test_stage2_target_lane_purity.py` | `reporting_contract` | reduced-strict-complete | open under P5 | none |
| Surface RZ Fourier banana geometry | `tests/geo/test_surface_rzfourier.py` | N/A: not a required banana C++ oracle lane | `src/simsopt/jax_core/surface_rzfourier.py` | `tests/geo/test_surface_rzfourier_jax.py` | `direct_kernel` | contract-complete | open under P5 | none |
| SurfaceXYZTensorFourier single-stage seed geometry | `tests/geo/test_surface_xyzfourier.py` | `src/simsoptpp/surfacexyztensorfourier.h` | `src/simsopt/jax_core/surface_fourier.py`<br>`src/simsopt/geo/surface_fourier_jax.py` | `tests/geo/test_surface_fourier_jax.py`<br>`tests/test_benchmark_helpers.py` | `direct_kernel` | contract-complete | not claimed | none |
| ToroidalFlux surface objective | `tests/geo/test_surface_objectives.py` | N/A: Python surface objective contract, not a required banana C++ oracle lane | `src/simsopt/geo/surfaceobjectives_jax.py` | `tests/geo/test_surface_objectives_jax.py` | `derivative_heavy` | complete | not claimed | none |
| Boozer CPU contract for banana integration | `tests/geo/test_boozersurface.py` | `src/simsoptpp/boozerresidual_impl.h` | `src/simsopt/geo/boozersurface_jax.py` | `tests/geo/test_boozersurface_jax.py` | `exact_well_conditioned_adjoint` | cpu-contract-complete | open under P5 | none |

## Boozer Non-CUDA Lane Status

These rows intentionally exclude `gpu_runtime` and `reduction_cpu_gpu`. CUDA
lanes remain gated by the hardware validation commands in the Boozer plan.
The result-contract cleanup rerun on 2026-05-05 passed the non-CUDA bundle:
758 passed, 1 skipped, 65 deselected, 56 subtests passed.
The Boozer Hessian oracle addendum on 2026-05-05 then passed the focused CPU
Hessian/derivative slices: 16 `penalty_hessian` cases, and 45 combined
`hessian or derivative` cases.

| Lane | Status | Evidence |
| --- | --- | --- |
| `direct_kernel` | complete | `tests/geo/test_boozer_residual_jax.py`, `tests/geo/test_boozersurface_jax.py`, and label/residual kernel parity in `docs/boozer_full_parity_plan_2026-05-04.md`. |
| `ls_wrapper_gradient` | complete | `tests/integration/test_single_stage_jax_cpu_reference.py::test_real_fixture_ondevice_parity_and_wrapper_gradients` and wrapper-gradient slices for `IotasJAX`, `NonQuasiSymmetricRatioJAX`, and `BoozerResidualJAX`. |
| `derivative_heavy` | complete | `tests/geo/test_boozer_derivatives_jax.py`, direct Boozer derivative matrix checks, and batched adjoint RHS checks. |
| `direct_hessian_oracle` | complete | `tests/geo/test_boozersurface_jax.py::TestUpstreamFactoryBoozerMatrix::test_penalty_hessian_column_complete_cpu_parity_matrix`; same-state CPU/C++ Hessian oracle vs JAX HVP basis sweep at `rtol=1e-8`, `atol=1e-10`. |
| `exact_well_conditioned_adjoint` | complete | `tests/geo/test_boozersurface_jax.py::test_exact_well_conditioned_operator_adjoint_matches_dense_reference_and_plu`; operator callbacks are the runtime path and dense PLU is metadata only. |
| `exact_ill_conditioned_adjoint` | residual/failure-only | `tests/integration/test_single_stage_jax_cpu_reference.py::test_operator_adjoint_signoff_gate_on_exact_state`; mixed RHS fixture keeps residual-success and residual-failure behavior explicit without vector-parity claims. |
| `branch_stable_resolve` | complete | Branch-stable re-solve FD and exact/LS branch tests in `tests/integration/test_single_stage_jax_cpu_reference.py` and `tests/geo/test_boozersurface_jax.py`. |
| `fd_gradient` | complete | Directional finite-difference/Taylor evidence in `TestIotasJAXResolveFD`, `TestNonQSRatioJAXResolveFD`, and the fixed-state coil VJP directional FD slice. |

## Non-Banana Example CPU/JAX CPU Inventory

CPU-only fixed-state evidence for non-banana SIMSOPT examples. Implementation
plan: [`non_banana_example_cpp_jax_cpu_parity_plan_2026-05-12.md`](non_banana_example_cpp_jax_cpu_parity_plan_2026-05-12.md).
Results document: [`non_banana_example_cpp_jax_cpu_parity_results_2026-05-12.md`](non_banana_example_cpp_jax_cpu_parity_results_2026-05-12.md).
Expansion plan: [`example_cpp_jax_cpu_gpu_parity_expansion_plan_2026-05-14.md`](example_cpp_jax_cpu_gpu_parity_expansion_plan_2026-05-14.md).
JSON artifacts: `.artifacts/parity/20260512-non-banana-examples/`,
`.artifacts/parity/20260514-example-expansion/`, and
`.artifacts/parity/20260514-curve-objectives/` for the expanded fixture run
(`all-supported-cpu.json` for supported rows, `all-fixtures-current.json` for
the refreshed full registry including support gates).

These rows are independent of the banana inventory above; CUDA status is
explicitly `out of scope` for this CPU-only plan.

| Fixture | Verdict | Source example | Comparisons | Unsupported components | Notes |
| --- | --- | --- | --- | --- | --- |
| `minimal_stage2_flux_length_gap` | pass | `examples/1_Simple/stage_two_optimization_minimal.py` | 7 native pass | none | `QuadraticPenalty(sum(CurveLengthJAX), "max")` is now included in the CPU/JAX objective and gradient comparison; seed=1 Taylor diagnostic passes at `abs_diff < 1e-6` per eps. |
| `surface_area_volume_simple` | pass | `examples/1_Simple/surf_vol_area.py` | 8 native pass | none | Fixed initial `SurfaceRZFourier` state only; compares surface geometry, Area/Volume values, surface-DOF gradients, and perturbed Area/Volume values. Optimizer, save/load, and centered-difference solve are side-effect diagnostics, not parity surfaces. |
| `pm_simple_fixed_state_gpmo_baseline` | pass | `examples/1_Simple/permanent_magnet_simple.py` | 14 native pass | none | Reduced FAMUS-derived fixed-state fixture (`nphi=ntheta=2`, `downsample=100`, `K=4`) compares `PermanentMagnetGridJAX` payloads, baseline GPMO final moments/residual/objective, `R2_history`, `Bn_history`, `m_history`, and final `DipoleField`/`DipoleFieldJAX` B and Bdotn. This row does not claim generic `GPMO` dispatcher parity or full-size example runtime. |
| `pm_qa_fixed_state_gpmo_arbvec_or_multi` | partial | `examples/2_Intermediate/permanent_magnet_QA.py` | 19 native pass | `qa_coil_current_optimization`, `qa_plot_and_famus_outputs` | Historical fixture id retained, but the example uses `relax_and_split`, not GPMO. Reduced fixed-state QA fixture (`nphi=ntheta=4`, `max_iter=2`, `max_iter_RS=2`, two threshold passes) compares grid payloads, dense/proxy moments, scalar `RS_history`, dense/proxy moment histories, residuals, objectives, and `DipoleField`/`DipoleFieldJAX` B and Bdotn. Host coil-current optimization and output writing remain unsupported. |
| `pm_muse_famus` | pass | `examples/2_Intermediate/permanent_magnet_MUSE.py` | 14 native pass | none | Reduced MUSE FAMUS fixture (`nphi=ntheta=2`, `downsample=100`, `K=5`, `backtracking=2`, `max_nMagnets=4`) preserves the example `ArbVec_backtracking` family and compares grid payloads, final moments, residual, objective, `R2_history`, `Bn_history`, `m_history`, and `DipoleField`/`DipoleFieldJAX` B and Bdotn. `K=5` avoids the CPU oracle's duplicate terminal `k=K-1` history write for `K=max_nMagnets=4`. |
| `pm_pm4stell_backtracking` | pass | `examples/2_Intermediate/permanent_magnet_PM4Stell.py` | 14 native pass | none | Reduced PM4Stell fixture (`nphi=ntheta=2`, `downsample=100`, `K=5`, `backtracking=2`, `max_nMagnets=4`) preserves the example `ArbVec_backtracking` family with face/edge/corner triplet polarizations and compares grid payloads, final moments, residual, objective, `R2_history`, `Bn_history`, `m_history`, and `DipoleField`/`DipoleFieldJAX` B and Bdotn. `K=5` avoids the CPU oracle's duplicate terminal `k=K-1` history write for `K=max_nMagnets=4`. |
| `wireframe_rcls_basic_fixed_state` | partial | `examples/2_Intermediate/wireframe_rcls_basic.py` | 11 native pass | `RCLS_current_vector_nonunique_nullspace` | Reduced fixed-state RCLS fixture preserves the example `surf_plas` input mode and compares `Amat`, `bvec`, objective components, constraint satisfaction, `WireframeField`/`WireframeFieldJAX` B, dB/dX, and Bnormal. Raw current-vector identity is not claimed because equivalent RCLS solutions can differ in the nullspace while preserving fields/objectives. |
| `wireframe_rcls_ports_constraint_gate` | partial | `examples/2_Intermediate/wireframe_rcls_with_ports.py` | 12 native pass | `RCLS_current_vector_nonunique_nullspace` | Reduced port-constrained RCLS fixture preserves the example port collision masks, poloidal-current constraint, and `surf_plas` input mode; compares matrices, objective components, constraint shape/satisfaction, `WireframeField`/`WireframeFieldJAX` B, dB/dX, and Bnormal. Raw current-vector identity is not claimed because equivalent RCLS solutions can differ in the nullspace. |
| `wireframe_gsco_modular_fixed_state` | pass | `examples/2_Intermediate/wireframe_gsco_modular.py` | 12 native pass | none | Reduced deterministic GSCO fixed-state fixture compares C++ `simsoptpp.GSCO` against `greedy_stellarator_coil_optimization_jax` for `Amat`, `bvec`, constraint flags, final `x`, final loop count, `iter_hist`, `curr_hist`, `loop_hist`, `f_B_hist`, `f_S_hist`, and `f_hist`. |
| `wireframe_gsco_sector_saddle_fixed_state` | pass | `examples/2_Intermediate/wireframe_gsco_sector_saddle.py` | 17 native pass | none | Reduced sector/saddle GSCO fixture (`wf_n_phi=18`, `wf_n_theta=8`, `plas_n=4`, `max_iter=5`) preserves TF-coil initial currents, toroidal break free-cell masks, poloidal-current constraints, and the public `surf_plas` path; compares matrices, constraint flags/masks, final current state, histories, `WireframeField`/`WireframeFieldJAX` B, and Bnormal. |
| `wireframe_gsco_multistep_reduced_diagnostic` | partial | `examples/3_Advanced/wireframe_gsco_multistep.py` | 12 native pass | `wireframe_multistep_mutation_loop`, `wireframe_small_coil_pruning`, `wireframe_final_adjustment_step`, `wireframe_plot_and_vtk_outputs` | Reduced first-step fixture (`wf_n_phi=24`, `wf_n_theta=8`, `plas_n=4`, `max_iter=5`) preserves the example `surf_plas`/`ext_field` public `optimize_wireframe` path and compares `Amat`, `bvec`, GSCO flags, final current state, final loop count, and history arrays. Full multistep mutation and output stages remain unsupported. |
| `tracing_fieldlines_qa_reduced_endpoint` | pass | `examples/1_Simple/tracing_fieldlines_QA.py` | 6 native pass | none | Reduced `InterpolatedField` fixture (`nphi=32`, `ntheta=16`, interpolation grid `5x8x4`, one fieldline, `tmax=20`, `tol=1e-12`) compares `InterpolatedFieldJAX.jax_B_at`-routed field values, fieldline endpoint, final integration time/status, first phi-hit coordinates, and hit count under the `event_time_tracing` tolerance lane. The fixture now exercises the example's raw `LevelsetStoppingCriterion(sc_fieldline.dist)` spelling and SurfaceClassifier-based skip callback through the JAX route. |
| `tracing_fieldlines_ncsx_reduced_endpoint` | pass | `examples/1_Simple/tracing_fieldlines_NCSX.py` | 6 native pass | none | Reduced NCSX `InterpolatedField` fixture (`surface_nphi=32`, `surface_ntheta=12`, interpolation grid `5x8x4`, one fieldline, `tmax=20`, `tol=1e-12`) uses `simsopt.configs.get_data("ncsx")` and compares native JAX field values, fieldline endpoint, final integration time/status, phi-hit coordinates, and hit count while exercising the example's raw levelset-distance stopping adapter and skip callback. |
| `tracing_particle_gc_vac_reduced_endpoint` | pass | `examples/1_Simple/tracing_particle.py` | 7 native pass | none | Reduced NCSX particle guiding-center fixture (`surface_nphi=32`, `surface_ntheta=12`, interpolation grid `5x10x2`, one particle, `tmax=1e-7`, `tol=1e-9`, `mode='gc_vac'`) compares `InterpolatedFieldJAX` B and GradAbsB, endpoint, final integration time/status, phi-hit rows, and hit count while exercising the example's raw `LevelsetStoppingCriterion(sc_particle.dist)` adapter. |
| `tracing_boozer_gc_reduced_endpoint` | partial | `examples/2_Intermediate/tracing_boozer.py` | 6 native pass | `VMEC_input_external_solver` | Reduced Boozer guiding-center fixture uses cached VMEC wout plus cached BOOZXFORM boozmn data to preserve the example's `BoozerRadialInterpolant -> InterpolatedBoozerField -> trace_particles_boozer` path without requiring the VMEC or BOOZXFORM Python extensions at parity-run time. CPU `InterpolatedBoozerField` and `InterpolatedBoozerFieldJAX` compare modB, endpoint, final integration time/status, zeta-hit rows, and hit count; the unavailable input-file external-solver path remains named unsupported. |
| `cws_saved_local_flux_nfp2` | pass | `examples/3_Advanced/curves_CWS_example.py` | 7 native pass | none | Legacy `CurveCWSFourier` JSON reconstruction now returns live CWS curves; saved local-flux fixture compares surface geometry, `BiotSavart` field, `Bdotn`, `SquaredFlux`, native subtotal, and gradient. |
| `cws_saved_local_flux_nfp3` | pass | `examples/3_Advanced/curves_CWS_example.py` | 7 native pass | none | Same fixed-state saved local-flux contract as nfp=2, using the nfp=3/maxmode4 saved artifact. |
| `full_stage2_composite` | pass | `examples/2_Intermediate/stage_two_optimization.py` | 7 native pass | none | Full weighted composite now includes `CurveLengthJAX`, `CurveCurveDistanceJAX`, `CurveSurfaceDistanceJAX`, `LpCurveCurvatureJAX`, and `MeanSquaredCurvatureJAX` quadratic penalties on the JAX lane; raw and weighted component values are recorded on both lanes. |
| `planar_stage2_composite` | pass | `examples/2_Intermediate/stage_two_optimization_planar_coils.py` | 7 native pass | none | Full weighted planar composite now includes `CurveLengthJAX`, `CurveCurveDistanceJAX`, `CurveSurfaceDistanceJAX`, `LpCurveCurvatureJAX`, `MeanSquaredCurvatureJAX`, and `LinkingNumberJAX` on the JAX lane; raw and weighted component values are recorded on both lanes. |
| `position_orientation_flux_support_gate` | pass | `examples/1_Simple/optimize_coil_position_orientation.py` | 7 native pass | none | `OrientedCurveXYZFourier` now exposes `to_spec()`; reduced TF/windowpane fixed-state `SquaredFlux` fixture preserves active position/orientation DOFs and passes CPU/C++ vs JAX CPU field/objective/gradient comparisons. |
| `boozer_surface_basic` | pass | `examples/2_Intermediate/boozer.py` | 7 native pass | none | NCSX fixed-state residual + Area/Volume/ToroidalFlux at `direct_kernel`; `boozer_residual` max_abs=2.7e-14, labels ≤ 9e-16 |
| `boozer_qa_wrappers` | pass | `examples/2_Intermediate/boozerQA.py` | 7 native pass | none | NCSX solved-state Iotas / MajorRadius / NonQuasiSymmetricRatio / sum(CurveLength) scalar values at `direct_kernel`; JAX lane uses copied solved-state DOFs, pure-JAX surface helpers, and `CurveLengthJAX` over an independently loaded NCSX curve tree. This row still does not claim public `BoozerSurfaceJAX` wrapper/adjoint parity. |
| `finite_beta_target_flux` | pass | `examples/2_Intermediate/stage_two_optimization_finite_beta.py` | 7 native pass | none | W7-X finite-beta fixed-state fixture loads deterministic cached target `tests/test_files/finite_beta_w7x_B_external_normal_nphi32_ntheta32.npy` (`B_external_normal`, shape 32x32, SHA256 `ae4f35b773e2db9b2feb566d7fdbea7545f63e06cbbf1872ca7ec7ce46b7d658`) and compares the full target-flux plus `QuadraticPenalty(CurveLengthJAX, "identity")` objective/gradient. |
| `finitebuild_multifilament_support_gate` | pass | `examples/3_Advanced/stage_two_optimization_finitebuild.py` | 7 native pass | none | Reduced finite-build multifilament fixture compares symmetry-expanded filament `BiotSavart`, `Bdotn`, `SquaredFlux`, native subtotal, and gradient with `CurveLengthJAX` max penalties and `CurveCurveDistanceJAX` included in the weighted objective. `CurveFilament` now uses full-graph DOFs for spec-based field evaluation. |
| `strain_optimization_support_gate` | pass | `examples/2_Intermediate/strain_optimization.py` | 6 native pass | none | HSX fixed-state rotation-only fixture compares public CPU framed-curve strain arrays, torsional/binormal strain penalties, native subtotal, and rotation gradient against public `FrameRotationJAX` / `FramedCurveCentroidJAX` wrappers consumed by the same public strain penalty classes. |
| `coil_forces_support_gate` | pass | `examples/3_Advanced/coil_forces.py` | 8 native pass | none | Reduced fixed-state force/energy subproblem compares public `LpCurveForce`/`B2Energy` on independent coil trees and gates JAX public values against independent CPU oracles: `RegularizedCoil.force` integration for force and a NumPy inductance-matrix loop for energy. Optimizer execution, VTK output, and CPU-only geometric penalties remain out of scope. |
| `qfm_surface` | partial | `examples/1_Simple/qfm.py` | 9 native pass | `QfmSurface_host_solver` | Fixed initial NCSX surface only; compares QfmResidual value, Bdotn, surface gradient, Area, Volume, and ToroidalFlux. Host `QfmSurface` LBFGS/SLSQP solver orchestration is not claimed as JAX parity. |

`partial` rows above all pass their native-supported tolerance buckets. The
2026-05-14 curve-objective closeout artifact
`.artifacts/parity/20260514-curve-objectives/local-converted-fixtures.json`
records `minimal_stage2_flux_length_gap`, `full_stage2_composite`,
`planar_stage2_composite`, `finite_beta_target_flux`, and
`finitebuild_multifilament_support_gate` as `pass` with zero unsupported
components and zero failing CPU/C++ vs JAX CPU comparisons. The refreshed full
matrix artifact
`.artifacts/parity/20260514-curve-objectives/all-fixtures-current.json` records
27 fixtures = 21 pass / 6 partial / 0 fail. No native-supported comparison
fails.
