# Item 03 Coverage Matrix

Status: complete for CPU/JAX oracle closure. CUDA proof is not claimed.

Upstream audit SHA:
`1b0cc3a96063197cdbdd01559e04c25456fbe6ff`.

Search commands (per section 4a of the goal prompt):

```
git -C /Users/suhjungdae/code/columbia/simsopt-jax grep -nE \
    "SquaredFlux|fluxobjective|integral_bdotn|BdotN" tests/
git -C /Users/suhjungdae/code/opensource/simsopt grep -nE \
    "SquaredFlux|fluxobjective|integral_BdotN|BdotN" tests/
```

The matrix below classifies every reachable row.

## Repo tests (current HEAD)

| Source | Path / Node | Intent | Classification | Evidence |
| --- | --- | --- | --- | --- |
| current | `tests/objectives/test_fluxobjective.py::FluxObjectiveTests::test_definitions` | NumPy oracle vs `SquaredFlux.J()` for all 3 definitions; CPU C++ path | `oracle_only` | Validates the CPU C++ oracle (`sopp.integral_BdotN`) used as the JAX parity reference. |
| current | `tests/objectives/test_fluxobjective.py::FluxObjectiveTests::test_derivatives` | CPU `SquaredFlux.dJ()` Taylor test for all 3 definitions, with and without target, plus composite | `oracle_only` | CPU C++ oracle finite-difference gradient validation. |
| current | `tests/objectives/test_fluxobjective.py::FluxObjectiveTests::test_quadratic_flux_gradient_handles_zero_normals` | Zero-area normal contract on CPU `SquaredFlux.dJ()` | `oracle_only` | Documents the CPU singular-normal contract; mirrored JAX coverage exists in parity tests. |
| current | `tests/objectives/test_fluxobjective.py::FluxObjectiveTests::test_singular_local_returns_inf_and_raises_gradient_failure` | CPU `local` singular `\|B\|=0` contract | `oracle_only` | CPU C++ oracle inf contract. |
| current | `tests/objectives/test_fluxobjective.py::FluxObjectiveTests::test_singular_normalized_returns_inf_and_raises_gradient_failure` | CPU `normalized` singular denominator contract | `oracle_only` | CPU C++ oracle inf contract. |
| current | `tests/objectives/test_fluxobjective.py::FluxObjectiveTests::test_squaredfluxjax_requires_surface_spec` | `SquaredFluxJAX` rejects non-spec surfaces | `covered_by_unit_parity` | Enforces the immutable surface-spec contract; covered. |
| current | `tests/objectives/test_fluxobjective_jax_parity.py::test_fluxobjective_value_parity[*]` | JAX vs CPU `J()` parity for all 3 definitions | `covered_by_unit_parity` | Parametrized over 3 definitions; 2-coil `nfp=1, stellsym=False`. |
| current | `tests/objectives/test_fluxobjective_jax_parity.py::test_fluxobjective_gradient_parity[*]` | JAX vs CPU `dJ()` parity for all 3 definitions | `covered_by_unit_parity` | Parametrized over 3 definitions. |
| current | `tests/objectives/test_fluxobjective_jax_parity.py::test_squaredfluxjax_gradient_matches_directional_taylor_fd[*]` | JAX directional FD gradient parity for all 3 definitions | `covered_by_unit_parity` | Imports `parity_ladder_tolerances("fd-gradient")`. |
| current | `tests/objectives/test_fluxobjective_jax_parity.py::test_squaredfluxjax_large_point_cloud_grouped_vjp_matches_dense` | Chunked vs dense parity at large point cloud | `covered_by_unit_parity` | Grouped VJP parity. |
| current | `tests/objectives/test_fluxobjective_jax_parity.py::test_non_rz_fixed_surface_value_and_gradient_parity[*]` | XYZFourier + XYZTensorFourier surface parity for all 3 definitions | `covered_by_unit_parity` | Surface family coverage. |
| current | `tests/objectives/test_fluxobjective_jax_parity.py::test_fluxobjective_target_parity` | Non-zero target parity (quadratic flux) | `covered_by_unit_parity` | Target array shape semantics. |
| current | `tests/objectives/test_fluxobjective_jax_parity.py::test_quadratic_flux_zero_normals_contract` | JAX zero-area normal contract | `covered_by_unit_parity` | Matches CPU degenerate-normal contract. |
| current | `tests/objectives/test_fluxobjective_jax_parity.py::test_degenerate_normals_do_not_perturb_valid_flux_contracts[*]` | JAX masked degenerate + valid quadrature mix | `covered_by_unit_parity` | Hand-derived oracle from NumPy formula. |
| current | `tests/objectives/test_fluxobjective_jax_parity.py::test_singular_zero_field_contract[*]` | JAX singular `\|B\|=0` for `normalized` and `local` | `covered_by_unit_parity` | Returns inf, zero grad. |
| current | `tests/objectives/test_fluxobjective_jax_parity.py::test_squaredfluxjax_zero_current_gradient_raises_objective_failure[*]` | `ObjectiveFailure` on zero-current gradient | `covered_by_unit_parity` | Strict failure contract for `normalized` and `local`. |
| current | `tests/objectives/test_fluxobjective_jax_parity.py::test_squaredfluxjax_rejects_field_point_mutation_after_construction` | Fixed-points contract (RuntimeError) | `covered_by_unit_parity` | DOF-layout drift rejection. |
| current | `tests/objectives/test_fluxobjective_jax_parity.py::test_squaredfluxjax_rejects_field_dof_layout_mutation_after_construction` | Fixed DOF-layout contract (RuntimeError) | `covered_by_unit_parity` | Restart-compatibility witness. |
| current | `tests/objectives/test_fluxobjective_jax_parity.py::test_squaredfluxjax_requires_native_field_contract` | Non-native field rejected at construction | `covered_by_unit_parity` | Native JAX contract enforced. |
| current | `tests/objectives/test_integral_bdotn_jax.py::TestIntegralBdotN::test_against_numpy[*]` | NumPy oracle vs JAX `integral_BdotN` for all 3 definitions | `covered_by_unit_parity` | Pure JAX kernel parity. |
| current | `tests/objectives/test_integral_bdotn_jax.py::TestIntegralBdotN::test_against_numpy_with_target[*]` | Same with non-zero target | `covered_by_unit_parity` | Target-shape coverage. |
| current | `tests/objectives/test_integral_bdotn_jax.py::TestIntegralBdotNCppParity::*` | JAX vs C++ `sopp.integral_BdotN` direct parity | `covered_by_unit_parity` | C++ direct oracle parity for all 3 definitions. |
| current | `tests/objectives/test_integral_bdotn_jax.py::TestIntegralBdotNBoundaryContracts::*` | Boundary contract violations | `covered_by_unit_parity` | Shape and definition validation. |
| current | `tests/integration/test_stage2_jax.py::TestObjectiveValueParity::test_j_parity[*]` | Stage 2 JAX vs CPU `J()` parity for all 3 definitions | `covered_by_integration_parity` | nfp=1, stellsym=False, 2 coils, nphi=ntheta=32. |
| current | `tests/integration/test_stage2_jax.py::TestObjectiveValueParity::test_j_with_target` | Non-zero target parity | `covered_by_integration_parity` | Random target via `np.random.RandomState(42)`. |
| current | `tests/integration/test_stage2_jax.py::TestObjectiveValueParity::test_singular_zero_current_objectives_boundary_is_documented[*]` | Zero-current J=inf | `covered_by_integration_parity` | `normalized` and `local`. |
| current | `tests/integration/test_stage2_jax.py::TestGradientParity::test_gradient_parity[*]` | Stage 2 JAX vs CPU `dJ()` parity for all 3 definitions | `covered_by_integration_parity` | Composite Optimizable chain. |
| current | `tests/integration/test_stage2_jax.py::TestGradientParity::test_singular_zero_current_gradients_raise_objective_failure[*]` | ObjectiveFailure for zero current | `covered_by_integration_parity` | `normalized` and `local`. |
| current | `tests/integration/test_stage2_jax.py::TestGradientParity::test_j_only_uses_forward_path_until_gradient_is_requested` | JIT call-count discipline | `covered_by_integration_parity` | No spurious value_and_grad calls. |
| current | `tests/integration/test_stage2_jax.py::TestGradientParity::test_gradient_then_value_reuses_cached_squared_flux_value` | Combined value+grad cache | `covered_by_integration_parity` | JIT call-count discipline. |
| current | `tests/integration/test_stage2_jax.py::TestCompositeGradient::*` | Stage 2 composite (SquaredFlux + CurveLength) | `covered_by_integration_parity` | Composite gradient parity. |
| current | `tests/integration/test_stage2_jax.py::TestStage2ShortRunSeed1234*` | Short-run parity at seed 1234 | `covered_by_integration_parity` | Production-scale Stage 2 short run. |
| current | `tests/integration/test_stage2_jax.py::TestProductionScaleTarget*` | Production-scale composite target objective | `covered_by_integration_parity` | nphi=31, ntheta=16, ncoils=20 TF + banana, nfp=5, stellsym=True. |
| current | `tests/integration/test_stage2_jax.py::TestMixedQuadrature*` | Mixed-quadrature TF + banana coil parity | `covered_by_integration_parity` | All 3 definitions, mixed-quadrature consume immutable spec contract. |
| current | `tests/integration/test_stage2_jax.py::TestCurveCWSFourierCPP*` | CWS Fourier C++-curve parity | `covered_by_integration_parity` | Extended coil family coverage. |
| current | `tests/integration/test_jax_native_path.py::*` | End-to-end JAX-native value/gradient/finite diff | `covered_by_integration_parity` | Full DOF→B→J→dJ chain. |
| current | `tests/integration/test_non_banana_example_cpp_jax_cpu_parity.py::*` | Non-banana CPU C++ vs JAX CPU parity (Phase 6) | `covered_by_integration_parity` | Real Stage 2 SquaredFlux/SquaredFluxJAX subproblem. |
| current | `tests/subprocess/import_smoke_cases.py:1209-1328` | SquaredFluxJAX construction smoke (host-surface rejection) | `covered_by_integration_parity` | Surface spec rejection witness. |
| current | `tests/subprocess/import_smoke_cases.py:1934-2115` | SquaredFluxJAX importability | `covered_by_integration_parity` | Public package boundary smoke. |
| current | `tests/test_jax_import_smoke.py:1096-1187` | SquaredFluxJAX construction strict-parity smoke | `covered_by_integration_parity` | Multi-device / strict-parity entrypoint smokes. |
| current | `tests/field/test_magneticfields.py:851-918` | DipoleField + SquaredFlux numerical agreement | `wrapper_only` | Uses CPU SquaredFlux to validate DipoleField; not a JAX path. |
| current | `tests/field/test_wireframefield.py:242-273` | Wireframe inductance vs SquaredFlux | `wrapper_only` | Uses CPU SquaredFlux as reference. |
| current | `tests/geo/test_finitebuild.py:183` | Finite-build SquaredFlux smoke | `wrapper_only` | Uses CPU SquaredFlux; finite-build is item 09. |
| current | `tests/geo/test_pm_grid.py:185-262` | Permanent magnet SquaredFlux | `wrapper_only` | PM tier P4 future scope. |
| current | `tests/util/test_coil_optimization_helper_functions.py:705-821` | Helper function consumes SquaredFlux | `wrapper_only` | Downstream helper; not a JAX path. |
| current | `tests/solve/test_wf_optimization.py:39-291` | Wireframe optimization SquaredFlux | `wrapper_only` | Tier P4 future scope. |
| current | `tests/conftest.py:94` | `integral_bdotn_normalized_stress` fixture metadata | `wrapper_only` | Test infrastructure metadata. |
| current | `tests/test_benchmark_helpers.py:2465` | Benchmark helper key | `wrapper_only` | Benchmark plumbing, not a parity assertion. |
| current | `tests/integration/conftest.py:36-40` | Meta path finder for cross-env JAX module injection | `wrapper_only` | Test environment plumbing. |
| current | `tests/geo/test_single_stage_example.py:11194` | Comment about BdotN diagnostic | `not_applicable` | Comment only, not a SquaredFlux assertion. |
| **new** | `tests/objectives/test_fluxobjective_jax_item03_closeout.py::test_squared_flux_jax_matches_cpp_oracle_under_strict_transfer_at_production_scale[*]` | Production-scale × all 3 definitions × strict transfer × stellsym=True | `covered_by_unit_parity` | Item 03 closeout witness; imports `parity_ladder_tolerances("direct_kernel")`. |

## Upstream tests (SIMSOPT 1b0cc3a96)

| Source | Path / Node | Intent | Classification | Evidence |
| --- | --- | --- | --- | --- |
| upstream | `tests/objectives/test_fluxobjective.py::FluxObjectiveTests::test_definitions` | NumPy oracle vs CPU `SquaredFlux.J()` for all 3 definitions | `oracle_only` | Upstream parity oracle (mirrored in repo `test_fluxobjective.py`). |
| upstream | `tests/objectives/test_fluxobjective.py::FluxObjectiveTests::test_derivatives` | CPU Taylor test for all 3 definitions | `oracle_only` | Upstream gradient oracle (mirrored). |
| upstream | `tests/field/test_magneticfields.py:851-918` | DipoleField + SquaredFlux | `wrapper_only` | Mirrored in repo; consumes CPU SquaredFlux. |
| upstream | `tests/field/test_wireframefield.py:242-273` | Wireframe inductance | `wrapper_only` | Mirrored. |
| upstream | `tests/geo/test_finitebuild.py:183` | Finite-build SquaredFlux smoke | `wrapper_only` | Mirrored; tier P3-P4 future scope. |
| upstream | `tests/geo/test_pm_grid.py:185-262` | PM SquaredFlux | `wrapper_only` | Mirrored; tier P4 future scope. |
| upstream | `tests/util/test_coil_optimization_helper_functions.py:705-821` | Helper SquaredFlux | `wrapper_only` | Mirrored. |
| upstream | `tests/solve/test_wf_optimization.py:39-291` | Wireframe optimization | `wrapper_only` | Mirrored; tier P4 future scope. |

No matrix row is unclassified. Every JAX parity test that is
`covered_by_unit_parity` or `covered_by_integration_parity` resolves on
disk at the commit's tree and is collected by pytest (witnessed during
item 03 validation; see `.artifacts/jax_port_goal/plans/03.md` validation
log). The new `tests/objectives/test_fluxobjective_jax_item03_closeout.py`
imports its tolerances from
`benchmarks.validation_ladder_contract.parity_ladder_tolerances("direct_kernel")`
with no inlined `atol=`/`rtol=` numeric literals (verified by
`git diff a9da18fac..HEAD -- tests/objectives/test_fluxobjective_jax_item03_closeout.py | grep -E '(atol|rtol)\s*=\s*[0-9eE.+-]+'`
returning zero hits when run after staging).
