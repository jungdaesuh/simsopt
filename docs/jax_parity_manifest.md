# JAX Parity Manifest

Base parity matrix snapshot as of 2026-04-10.

Boozer rows refreshed on 2026-05-05 after the CPU closure in
[`boozer_full_parity_plan_2026-05-04.md`](boozer_full_parity_plan_2026-05-04.md).
This refresh does not claim CUDA hardware parity.

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
| `tests/objectives/test_fluxobjective.py` | `tests/objectives/test_fluxobjective_jax_parity.py` | partial (pending reproducer search) | Dedicated mirrored wrapper coverage for definitions, derivatives, target handling, degenerate normals, singular zero-field behavior, native-contract rejection, non-RZ surface cases, and mutation/layout guards. Value/gradient parity is exact where the upstream wrapper contract is defined. Current kernel inspection and tests cover the documented degenerate contracts (`0.0` for zero-normal quadratic flux, `inf` for normalized/local zero-field singularities); no pinned CPU `nan` reproducer is recorded in this manifest yet. |
| `tests/objectives/test_fluxobjective.py` | `tests/integration/test_stage2_jax.py` | partial | Integration coverage for mixed quadrature and native-spec rejection behavior complements the dedicated object-level parity file. This row intentionally remains partial after the flux/kernel reconciliation because it tracks Stage 2 integration scope, not low-level `integral_BdotN` boundary behavior. |
| `tests/objectives/test_integral_bdotn_jax.py` | `tests/objectives/test_integral_bdotn_jax.py` | partial (pending reproducer search) | Exact on regular inputs, including direct `simsoptpp.integral_BdotN` parity. Documented degenerate contracts are covered for zero-normal quadratic flux (`0.0`) and normalized/local zero-field singularities (`inf`). The remaining manifest task is to either add a concrete CPU `nan` reproducer if one exists, or promote this row after that stale qualifier is ruled out. |
| `tests/geo/test_surface_rzfourier.py` | `tests/geo/test_surface_rzfourier_jax.py` | partial | Strict tolerance-based CPU/JAX parity for the RZ geometry/object API surface (`surface_spec`/`to_spec`, `*_jax`, DOF round-trips, gradients, loaders, and `copy`). Non-RZ `SurfaceXYZFourier` / `SurfaceXYZTensorFourier` geometry/spec parity exists separately, but broad XYZ object/I/O utility mirroring remains outside the banana contract unless the scope expands to full legacy surface parity. |
| `tests/geo/test_surface_objectives.py::ToroidalFlux*` | `tests/geo/test_surface_objectives_jax.py` | partial | Upstream `ToroidalFlux` first-derivative, Hessian, coil-derivative, constant-in-index, and Taylor checks are mirrored across the surface-type and `stellsym` sweep with tolerance-based CPU/JAX parity. The remaining bounded gap is a value-parity matrix over the same surface-type and `stellsym` sweep, not exact arithmetic parity. |
| `tests/geo/test_boozersurface.py` | `tests/geo/test_boozersurface_jax.py` | cpu-contract-complete | Boozer CPU parity closure is complete for math kernels, solver results, guard behavior, derivatives/adjoints, and supported public APIs. Parity remains contract-based: solved-state quality and public result semantics are the oracle, not mutable object identity or iterate-by-iterate solver trajectory. |
| `tests/integration/test_single_stage_example.py` and single-stage Boozer integration slices | `tests/integration/test_single_stage_jax_cpu_reference.py` | cpu-contract-complete | Dedicated CPU/JAX Boozer integration tests compare convergence success, residual norms, final solver objective, and final physics quantities (`iota`, `G`, label value/error, anchored axis-z). CUDA Boozer parity is not claimed by this CPU closure and still requires the optional hardware validation gate in the Boozer plan. |

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
