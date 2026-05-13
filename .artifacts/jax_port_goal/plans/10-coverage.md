# Item 10 Coverage Matrix

Status: complete for CPU/JAX oracle closure. CUDA proof is not claimed.

Upstream audit SHA:
`1b0cc3a96063197cdbdd01559e04c25456fbe6ff`.

## Repo grep

`git grep -nE "BiotSavartJAX|biotsavart_jax|integral_bdotn_jax|stage2_target_objective_jax|coil_graph|biot_savart_B\b|biot_savart_dB\b|biot_savart_B_vjp|grouped_biot_savart|fixed_surface_flux_integral" tests/`
yielded the rows below (unique test files only — every hit was inspected
and rolled up to its owning test file/class).

## Upstream grep

`git -C /Users/suhjungdae/code/opensource/simsopt grep -lE "BiotSavart|integral_BdotN|SquaredFlux" tests/`
yielded the upstream rows below.

| Source | Path / Node | Intent | Classification | Evidence |
| --- | --- | --- | --- | --- |
| current | `tests/field/test_biotsavart_jax.py::TestBiotSavartJAXOnAxisCircularCoil` | Closed-form circular-coil on-axis B-field oracle | `covered_by_unit_parity` | direct hand-derived oracle; pure-JAX kernel |
| current | `tests/field/test_biotsavart_jax.py::TestBiotSavartJaxCppParity::test_B_parity_ncsx` | NCSX-scale C++ B parity at `rtol=1e-10` | `covered_by_unit_parity` | direct C++ oracle (`bs.B()`); ncoils=33 |
| current | `tests/field/test_biotsavart_jax.py::TestBiotSavartJaxCppParity::test_dB_by_dX_parity_ncsx` | NCSX-scale C++ dB/dX parity | `covered_by_unit_parity` | derivative-heavy lane tolerances |
| current | `tests/field/test_biotsavart_jax.py::TestBiotSavartJaxChunkedParity` | Chunked vs dense dense JAX kernel parity | `covered_by_unit_parity` | chunked tuning at ncoils=4 |
| current | `tests/field/test_biotsavart_jax_parity.py::TestBiotSavartParitySuite` | Quadrature convergence, B=curl A, dB/dX taylor tests, VJP parity at curve fidelity | `covered_by_unit_parity` | grouped fields, mixed quadrature |
| current | `tests/field/test_biotsavart_jax_parity.py::TestGroupedBiotSavartGradient` | Mixed-quadrature grouped-coil gradient FD | `covered_by_unit_parity` | grouped kernel coverage |
| current | `tests/field/test_biotsavart_jax_parity.py::TestCurveTypeParametrization` | Cross-curve-type Biot-Savart parity for non-XYZFourier specs | `covered_by_unit_parity` | covers curve-spec lane |
| current | `tests/field/test_biotsavart_jax_cpu_ordered.py` | CPU-ordered twin parity for byte-identity gate | `covered_by_unit_parity` | DM-A/B cpu-ordered twin |
| current | `tests/integration/test_stage2_jax.py::TestBiotSavartJAXParity` | `BiotSavartJAX.B()` and `B_vjp()` parity vs `BiotSavart` at ncoils=4 (8 after symmetries), nphi=ntheta=32 | `covered_by_integration_parity` | full coil-graph parity at production scale |
| current | `tests/integration/test_stage2_jax.py::TestObjectiveValueParity` | `SquaredFluxJAX.J()` parity vs `SquaredFlux.J()` chains BS -> BdotN through `SquaredFlux` wrapper | `covered_by_integration_parity` | objective-wrapper-mediated parity (not bare integral_BdotN chain) |
| current | `tests/integration/test_stage2_target_lane_purity.py` | Stage 2 target objective lane purity vs runtime contract | `covered_by_integration_parity` | exercises `stage2_target_objective_jax.py` |
| current | `tests/integration/test_jax_native_path.py` | End-to-end JAX-native Stage 2 path; uses `integral_BdotN` directly | `covered_by_integration_parity` | bare `integral_BdotN` exercised in pipeline |
| current | `tests/integration/test_single_stage_physics_parity.py` | Single-stage physics chain parity | `covered_by_integration_parity` | upstream parity at single-stage scale |
| current | `tests/integration/test_single_stage_jax.py` | Single-stage JAX wrapper parity | `covered_by_integration_parity` | M5 wrapper coverage |
| current | `tests/integration/test_single_stage_jax_cpu_reference.py` | Single-stage CPU reference comparison | `covered_by_integration_parity` | banana 16-coil setting |
| current | `tests/integration/test_non_banana_example_cpp_jax_cpu_parity.py` | Non-banana CPP / JAX / CPU parity | `covered_by_integration_parity` | downstream fixture covering distance + flux composite |
| current | `tests/objectives/test_integral_bdotn_jax.py::TestIntegralBdotN` | NumPy oracle parity for all 3 definitions at nphi=10, ntheta=12 (also stress fixtures) | `covered_by_unit_parity` | NumPy oracle, not C++ chain |
| current | `tests/objectives/test_integral_bdotn_jax.py::TestIntegralBdotNCppParity::test_cpp_parity` | C++ `integral_BdotN` parity at nphi=ntheta=15 (independent B array, no chain through Biot-Savart) | `covered_by_unit_parity` | direct C++ oracle for the reducer kernel |
| current | `tests/objectives/test_integral_bdotn_jax.py::TestIntegralBdotNCppParity::test_cpp_boundary_contract_matches_jax` | Singular boundary contract: zero-B, zero-normal, inf returns | `covered_by_unit_parity` | direct C++ oracle |
| current | `tests/objectives/test_integral_bdotn_jax.py::TestIntegralBdotNBoundaryContracts` | JAX-defined boundary semantics for degenerate inputs | `covered_by_unit_parity` | implementation contract |
| current | `tests/objectives/test_fluxobjective_jax_parity.py::test_fluxobjective_value_parity` | `SquaredFluxJAX` value vs `SquaredFlux` at ncoils=2, nphi=ntheta=32 | `covered_by_integration_parity` | wrapper-mediated SquaredFlux parity |
| current | `tests/objectives/test_fluxobjective_jax_parity.py::test_fluxobjective_gradient_parity` | `SquaredFluxJAX` gradient vs `SquaredFlux` at ncoils=2 | `covered_by_integration_parity` | wrapper-mediated gradient parity |
| current | `tests/objectives/test_fluxobjective_jax_parity.py::test_non_rz_fixed_surface_value_and_gradient_parity` | Non-RZ surface family parity | `covered_by_integration_parity` | XYZ / XYZTensor surface coverage |
| current | `tests/objectives/test_fluxobjective_jax_parity.py::test_fluxobjective_target_parity` | Nonzero target parity | `covered_by_integration_parity` | wrapper-mediated nonzero-target parity |
| current | `tests/objectives/test_fluxobjective_jax_parity.py::test_quadratic_flux_zero_normals_contract` | Zero-normal degenerate path through SquaredFluxJAX | `covered_by_integration_parity` | wrapper-mediated singular path |
| current | `tests/objectives/test_fluxobjective_jax_parity.py::test_squaredfluxjax_rejects_field_point_mutation_after_construction` | JIT closure / set_points discipline | `covered_by_unit_parity` | wrapper contract |
| current | `tests/objectives/test_fluxobjective_jax_parity.py::test_squaredfluxjax_requires_native_field_contract` | Reject non-native field input | `covered_by_unit_parity` | wrapper contract |
| current | `tests/core/test_load_specs.py` (BiotSavartJAX restart spec) | as_dict / from_dict spec-backed reload | `covered_by_unit_parity` | restart-spec coverage |
| current | `tests/test_jax_import_smoke.py` (BiotSavartJAX import + sharding subprocess tests) | Import smoke + multi-device CPU subprocess proxy | `covered_by_unit_parity` | smoke + 4-device CPU lane proxy |
| current | `tests/test_jax_import_smoke.py::test_grouped_biot_savart_coil_collective_parity_and_lowering` | 4-device CPU collective lowering check (HLO) | `covered_by_integration_parity` | multi-device CPU collective proxy |
| current | `tests/test_jax_import_smoke.py::test_grouped_biot_savart_accepts_explicit_point_sharding` | 4-device CPU explicit point sharding parity | `covered_by_integration_parity` | multi-device CPU explicit-sharding proxy |
| current | `tests/geo/test_boozer_derivatives_jax.py` | Boozer derivatives consume B and dB/dX | `wrapper_only` | exercises BS via Boozer pipeline; not BS oracle |
| current | `tests/geo/test_boozersurface_jax.py` | Boozer surface consumes B and dB/dX | `wrapper_only` | exercises BS via Boozer pipeline |
| current | `tests/geo/test_label_constraints_jax.py` | Label constraints (volume/area/flux) consume coil graph | `wrapper_only` | label-only |
| current | `tests/geo/test_single_stage_example.py` | Single-stage example flow | `wrapper_only` | example pipeline |
| current | `tests/geo/test_surface_objectives_jax.py` | Surface objectives consume coil graph | `wrapper_only` | exercises BS via surface objective wrapper |
| current | `tests/geo/boozersurface_jax_test_helpers.py` | Boozer test helpers | `wrapper_only` | helper module, not a test |
| current | `tests/subprocess/import_smoke_cases.py` | Subprocess import smoke targets | `wrapper_only` | driven by outer tests in `test_jax_import_smoke.py` |
| current | `tests/subprocess/jax_runtime_cases.py` | Subprocess runtime smoke targets | `wrapper_only` | driven by outer tests |
| current | `tests/integration/conftest.py` | Cross-env meta path finder for jax modules | `not_applicable` | infrastructure; reason: shared meta-path setup for cross-env tests, not a test |
| current | `tests/test_benchmark_helpers.py` | Benchmark helper regression suite | `wrapper_only` | tests benchmark helpers, not item-10 numerics |
| current | `tests/objectives/test_fluxobjective.py` | Upstream CPU `SquaredFlux` reference path | `oracle_only` | reason: pure CPU `SquaredFlux` parity tests; the parity oracle |
| upstream | `upstream_hss/master:tests/field/test_biotsavart.py` | Upstream C++ BiotSavart quadrature/Taylor/VJP parity | `oracle_only` | reason: defines the C++ BS public oracle behavior that `BiotSavartJAX` is asserted against |
| upstream | `upstream_hss/master:tests/objectives/test_fluxobjective.py` | Upstream CPU `SquaredFlux` definition / target / gradient parity | `oracle_only` | reason: defines the C++/Python SquaredFlux public oracle behavior |
| upstream | `upstream_hss/master:tests/field/test_coil.py` | Upstream CPU `Coil`, `Current`, `coils_via_symmetries` semantics | `oracle_only` | reason: defines the public coil-graph API; matched 1:1 by `field/coil.py` here |
| upstream | `upstream_hss/master:tests/field/test_magneticfields.py` | Magnetic field zoo including SquaredFlux integration | `oracle_only` | reason: SquaredFlux is the oracle; consumers test BS as a black box |
| upstream | `upstream_hss/master:tests/field/test_fieldline.py` | Field-line tracing with BiotSavart | `oracle_only` | reason: BiotSavart consumed by tracing oracle |
| upstream | `upstream_hss/master:tests/field/test_wireframefield.py` | Wireframe-field SquaredFlux integration | `oracle_only` | reason: tests SquaredFlux + wireframefield, not BS kernel itself |
| upstream | `upstream_hss/master:tests/configs/test_LHD_like.py` | LHD-like config Biot-Savart smoke | `oracle_only` | reason: instantiates BS as a wrapper smoke |
| upstream | `upstream_hss/master:tests/configs/test_quasr_integration.py` | QUASR config BS integration | `oracle_only` | reason: instantiates BS as a wrapper smoke |
| upstream | `upstream_hss/master:tests/configs/test_zoo.py` | Configuration zoo BS smoke | `oracle_only` | reason: instantiates BS as a wrapper smoke |
| upstream | `upstream_hss/master:tests/geo/test_finitebuild.py` | Finite-build SquaredFlux integration | `oracle_only` | reason: SquaredFlux consumer |
| upstream | `upstream_hss/master:tests/geo/test_pm_grid.py` | Permanent magnet grid SquaredFlux integration | `oracle_only` | reason: SquaredFlux consumer; PM is P4 future scope |
| upstream | `upstream_hss/master:tests/geo/test_qfm.py` | QFM SquaredFlux integration | `oracle_only` | reason: SquaredFlux consumer; QFM is P3 future scope |
| upstream | `upstream_hss/master:tests/geo/test_curve.py` | Curve / coil serialization | `oracle_only` | reason: public coil-graph semantics |
| upstream | `upstream_hss/master:tests/geo/surface_test_helpers.py` | Surface fixtures including BS | `oracle_only` | reason: helpers, not tests |
| upstream | `upstream_hss/master:tests/geo/test_surface_objectives.py` | Surface objectives consumer | `oracle_only` | reason: surface objective oracle |
| upstream | `upstream_hss/master:tests/geo/test_boozersurface.py` | Boozer surface integration | `oracle_only` | reason: Boozer is item 04 |
| upstream | `upstream_hss/master:tests/solve/test_pm_optimization.py` | PM optimization driver | `oracle_only` | reason: PM is P4 future scope; SquaredFlux consumer |
| upstream | `upstream_hss/master:tests/solve/test_wf_optimization.py` | Wireframe optimization driver | `oracle_only` | reason: wireframe is P4 future scope; SquaredFlux consumer |
| upstream | `upstream_hss/master:tests/mhd/test_vmec.py` | MHD VMEC integration | `oracle_only` | reason: mhd/* is on the skip list (Fortran/external) |
| upstream | `upstream_hss/master:tests/util/test_coil_optimization_helper_functions.py` | Coil optimization helper smoke | `oracle_only` | reason: helper smoke around BS public API |

## New row closing the item

| Source | Path / Node | Intent | Classification | Evidence |
| --- | --- | --- | --- | --- |
| current | `tests/objectives/test_integral_bdotn_item10_closeout.py::test_chained_biotsavartjax_integral_bdotn_matches_cpp_at_production_scale` | Chained `BiotSavartJAX.B()` -> `integral_BdotN` vs `BiotSavart.B()` -> `sopp.integral_BdotN` parity at production-scale floor (ncoils=4, nphi=16, ntheta=8) for all 3 definitions x `stellsym=False/True`, under `jax.transfer_guard("disallow")` | `covered_by_unit_parity` | New closeout test imports `parity_ladder_tolerances("direct_kernel")`; closes the missing production-scale chained fixture identified in the audit |

No matrix row is unclassified. Every JAX parity row cites an existing
test file that resolves on disk at the current HEAD. The new closeout
row cites the test added by this item.
