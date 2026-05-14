# JAX ↔ C++ Parity-Test Coverage Audit — 2026-05-13

## Executive Summary

Audited every `**/*_jax*.py` module under `src/simsopt/{field,geo,objectives,solve}/` plus
`src/simsopt/jax_core/`. Of the public/semantic-bearing JAX symbols audited
(~70 functions/classes with a clear C++ counterpart in `src/simsoptpp/`):

- **COVERED** (direct C++ oracle parity asserted, machine-precision tolerance lane): ~36
- **PARTIAL** (C++ parity for some but not all public surface): ~10
- **INDIRECT** (only transitively tested through Stage 2 / single-stage integration): ~4
- **MISSING** (no C++ oracle parity test in the repo): ~12
- **NO C++ COUNTERPART** (JAX-only orchestration; implicit-diff plumbing): ~15

Top 5 HIGH-severity MISSING items:

1. **`biotsavart_jax.biot_savart_A` / `biot_savart_dA_by_dX`** — direct C++ oracle parity vs `simsopt.field.BiotSavart.A()` / `.dA_by_dX()` is NOT exercised in `tests/field/test_biotsavart_jax.py::TestBiotSavartJaxCppParity` (only `B` and `dB/dX` are). Coverage exists only via dense-chunked self-consistency probes (`TestBiotSavartJaxChunkedParity`) which use the JAX dense kernel as the oracle — this is **tautological** under `REVIEWER_ORACLE_LINT.md`. HIGH severity for the `direct_kernel` lane.
2. **`biotsavart_jax.biot_savart_d2B_by_dXdX` / `biot_savart_d2A_by_dXdX`** — no test imports both the JAX symbol and `BiotSavart.d2B_by_dXdX` / `.d2A_by_dXdX` from a `simsoptpp`-backed wrapper at the `derivative-heavy` lane. The acknowledged backlog note (`CLAUDE.md` "BiotSavartJAX missing d2B_by_dXdX/A/compute") is still not bridged at the kernel level.
3. **`biotsavart_jax.biot_savart_B_vjp`** — no direct test compares the JAX VJP against the C++ `BiotSavart` VJP at the `derivative-heavy` lane. The only coverage in `test_biotsavart_jax.py::TestBiotSavartJaxChunkedParity::test_B_vjp_rebuilds_when_tuning_changes_in_process` (lines 548–599) uses `_dense_B_vjp(chunked_bs, …)` — i.e. the JAX dense path — as the reference. **Tautological** by `REVIEWER_ORACLE_LINT.md`.
4. **`boozer_residual_jax.boozer_residual_scalar` / `_grad` / `_hessian` / `_vector`** — the docstring at `tests/geo/test_boozer_residual_jax.py:8` claims "C++ parity (when simsoptpp is available)" but no class in that file imports a `simsoptpp` Boozer residual symbol. The C++ oracle `simsoptpp.boozer_residual` (declared at `src/simsoptpp/boozerresidual_py.h`) is never co-imported with the JAX scalar/vector/grad/hessian helpers. Only the wrapper-level CPU oracle (`BoozerSurface.boozer_penalty_constraints_vectorized`) is compared in `test_boozersurface_jax.py::TestUpstreamFactoryBoozerMatrix`. HIGH severity — this is the primary kernel claim.
5. **`fluxobjective_jax.SquaredFluxJAX` gradient parity at the `direct-kernel` lane** — the explicit `TestObjectiveValueParity` (`test_stage2_jax.py:1026`) covers value parity but the gradient comparison (`TestGradientParity`, line 1091) is a tier-2 Stage-2 e2e comparison (`tier1_stage2_value_gradient`, rtol=1e-9), not a `direct-kernel` rtol=1e-10 contract; the only `direct-kernel` lane SquaredFlux gradient assertion lives at `test_fluxobjective_jax_parity.py::test_fluxobjective_gradient_parity`, which routes through the JAX kernel on both sides under a strict-JAX backend (the `_flux_kernel_value_and_grad` host helper at line 279 builds gradients by autodiffing the same kernel). Without a `BiotSavart.B_vjp`-mediated CPU gradient, this is a `direct_kernel` lane gap.

Artifact path: `/Users/suhjungdae/code/columbia/simsopt-jax/.artifacts/jax_port_gap_audit_2026-05-13/jax_cpp_parity_test_gaps.md`.

---

## Per-Module Coverage Tables

### `src/simsopt/field/biotsavart_jax.py` (compat shim into `simsopt.jax_core.biotsavart`)

| Function | Status | Test path:line | Severity | Tolerance lane |
|---|---|---|---|---|
| `biot_savart_B` | COVERED | `tests/field/test_biotsavart_jax.py:480-503` (`TestBiotSavartJaxCppParity.test_B_parity_ncsx`) — `np.testing.assert_allclose(B_jax, bs.B(), rtol=1e-10)` | — | `direct_kernel` |
| `biot_savart_dB_by_dX` | COVERED | `tests/field/test_biotsavart_jax.py:505-523` (`test_dB_by_dX_parity_ncsx`) | — | `derivative_heavy` (first_derivative_rtol/atol) |
| `biot_savart_B_and_dB` | INDIRECT | `tests/field/test_biotsavart_jax.py:421-428` only checks against JAX `biot_savart_B` + `biot_savart_dB_by_dX`. Tautology by REVIEWER_ORACLE_LINT type "jax_path == host_path"; transitively covered by `test_B_parity_ncsx` + `test_dB_by_dX_parity_ncsx`. | LOW | — |
| `biot_savart_A` | **MISSING** | No `BiotSavart.A()` co-import in any `TestBiotSavartJaxCppParity` test. Only self-consistency vs `_one_point_dense` dense JAX kernel in `TestBiotSavartJaxChunkedParity:601-651`. | **HIGH** | `direct_kernel` |
| `biot_savart_dA_by_dX` | **MISSING** | Same as above. Dense self-consistency only. | **HIGH** | `derivative_heavy` |
| `biot_savart_d2B_by_dXdX` | **MISSING** | No co-import with `BiotSavart.d2B_by_dXdX()` anywhere. Only `_assert_second_derivative_taylor_convergence` Taylor-FD check in `test_biotsavart_jax_parity.py:390`. | **HIGH** | `derivative_heavy` (second_derivative) |
| `biot_savart_d2A_by_dXdX` | **MISSING** | No tests at all. | **HIGH** | `derivative_heavy` (second_derivative) |
| `biot_savart_B_vjp` | **MISSING** | Only dense JAX self-consistency probes (`tests/field/test_biotsavart_jax.py:585-599`, where `_dense_B_vjp` is a JAX-only reference). No C++ VJP oracle. | **MED** | `derivative_heavy` |
| `grouped_biot_savart_B` | PARTIAL | Self-grouping consistency in `test_biotsavart_jax_parity.py::TestGroupedBiotSavartGradient`; FD gradient parity only. No C++ multi-coil oracle at `direct_kernel`. | MED | `fd_gradient` only |
| `grouped_biot_savart_A` | **MISSING** | No tests. | MED | — |
| `group_coil_data` | NO C++ COUNTERPART | Pure JAX pytree packing; no C++ analog. | — | — |
| `invalidate_kernel_cache` | NO C++ COUNTERPART | Cache control only. | — | — |

### `src/simsopt/field/biotsavart_jax_backend.py` (`BiotSavartJAX(Optimizable)`)

| Function | Status | Test path:line | Severity | Tolerance lane |
|---|---|---|---|---|
| `BiotSavartJAX.B/dB_by_dX/A/dA_by_dX` | COVERED (`B`, `dB`); PARTIAL (`A`, `dA`) | `tests/integration/test_stage2_jax.py:1026-1100` (`TestObjectiveValueParity`, `TestGradientParity`) compares `BiotSavartJAX`-mediated `SquaredFluxJAX` against `SquaredFlux(BiotSavart)`. Direct symbol parity for `BiotSavartJAX.A()` is not exercised. | MED | `tier1_stage2_value_gradient` (e2e) |
| `BiotSavartJAX.B_vjp` | INDIRECT | Only via Stage 2 e2e gradient parity above. No direct VJP-vs-C++ assertion. | MED | — |

### `src/simsopt/field/_jax_common.py`

| Function | Status | Test path:line | Severity |
|---|---|---|---|
| `_as_*_float64*`, dtype helpers | NO C++ COUNTERPART | Internal dtype/device plumbing. | — |

### `src/simsopt/field/circular_coil_jax.py`

| Function | Status | Test path:line | Severity | Tolerance lane |
|---|---|---|---|---|
| `circular_coil_B/dB/A` (kernels) | COVERED | `tests/field/test_circular_coil_jax.py:114-210` (`TestKernelParity`) compares against `simsopt.field.CircularCoil` CPU wrapper via `B_cpu`/`dB_cpu`. | — | `direct_kernel` (`_RTOL`, `_ATOL`) |
| `CircularCoilJAX` wrapper | COVERED | `TestWrapperParity:216-263`. | — | `direct_kernel` |
| Rotation helpers | COVERED | `TestRotationHelpers:264-308` vs C++ `sopp` rotation. | — | `direct_kernel` |

### `src/simsopt/field/dipole_field_jax.py`

| Function | Status | Test path:line | Severity | Tolerance lane |
|---|---|---|---|---|
| `dipole_field_B/A/dB/dA` JAX kernels | COVERED | `tests/jax_core/test_dipole_field_jax_item24.py:69-107` (`test_dipole_field_jax_vs_cpp_direct_kernel`) imports `simsoptpp as sopp` and compares all four against `sopp.dipole_field_{B,A,dB,dA}`. | — | `direct_kernel` |
| `DipoleFieldJAX` wrapper | COVERED | `tests/field/test_dipole_field_jax_item26.py:78-196` (`TestDipoleFieldJAXParity`). | — | `direct_kernel` |

### `src/simsopt/field/dommaschk_jax.py`

| Function | Status | Test path:line | Severity | Tolerance lane |
|---|---|---|---|---|
| `dommaschk_field_B/dB` kernels | COVERED | `tests/jax_core/test_analytic_fields_item11.py:153-194` (`test_dommaschk_cpp_cross_oracle`) imports `simsoptpp as sopp` and compares against `sopp.DommaschkB / sopp.DommaschkdB`. | — | `direct_kernel` |
| `DommaschkJAX` wrapper | COVERED | `tests/field/test_magneticfieldclasses_jax_item15.py:239` (`TestDommaschkJAX`). | — | `direct_kernel` |

### `src/simsopt/field/reiman_jax.py`

| Function | Status | Test path:line | Severity | Tolerance lane |
|---|---|---|---|---|
| `reiman_B/dB` kernels | COVERED | `tests/jax_core/test_analytic_fields_item11.py:278-330` (`test_reiman_cpp_cross_oracle`) vs `sopp.ReimanB / sopp.ReimandB`. | — | `direct_kernel` |
| `ReimanJAX` wrapper | COVERED | `tests/field/test_magneticfieldclasses_jax_item15.py:333` (`TestReimanJAX`). | — | `direct_kernel` |

### `src/simsopt/field/toroidal_field_jax.py` / `poloidal_field_jax.py` / `mirror_model_jax.py`

| Function | Status | Test path:line | Severity | Tolerance lane |
|---|---|---|---|---|
| `ToroidalFieldJAX` | COVERED | `tests/field/test_magneticfieldclasses_jax_item15.py:80-142` (`TestToroidalFieldJAX`) vs `simsopt.field.ToroidalField` CPU. | — | `direct_kernel` |
| `PoloidalFieldJAX` | COVERED | `tests/field/test_magneticfieldclasses_jax_item15.py:144-183` (`TestPoloidalFieldJAX`). | — | `direct_kernel` |
| `MirrorModelJAX` | COVERED | `tests/field/test_magneticfieldclasses_jax_item15.py:185-237`. | — | `direct_kernel` |

### `src/simsopt/field/scalar_potential_rz_jax.py`

| Function | Status | Test path:line | Severity | Tolerance lane |
|---|---|---|---|---|
| `ScalarPotentialRZMagneticFieldJAX.B/dB` | COVERED | `tests/field/test_scalar_potential_rz_jax_item23.py:47-66` (`test_scalar_potential_rz_jax_matches_cpu_B_and_dB`) — compares against CPU-side `ScalarPotentialRZMagneticField` which uses sympy lambda; oracle is the upstream Python wrapper (sympy-derived), not C++. Acceptable as `closed-form analytic expression` (type 2) but not a C++ oracle. | LOW | — |

### `src/simsopt/field/wireframefield_jax.py`

| Function | Status | Test path:line | Severity | Tolerance lane |
|---|---|---|---|---|
| Public B / dB / segment contributions | COVERED | `tests/field/test_wireframefield_jax_item30.py:55-110` vs CPU wrapper that delegates to `simsoptpp.WireframeField`. Direct C++ kernel parity in `tests/jax_core/test_wireframe_jax_item29.py:125-251` (`test_closed_loop_B_dB_parity` + `test_multi_halfperiod_seg_signs_parity`) vs `sopp.WireframeField`. | — | `direct_kernel` |
| Normal-field matrix | COVERED | `tests/field/test_wireframefield_jax_item30.py:88-110`. | — | `direct_kernel` |

### `src/simsopt/field/boozermagneticfield_jax.py`

| Function | Status | Test path:line | Severity | Tolerance lane |
|---|---|---|---|---|
| `BoozerAnalyticJAX` public API methods (`modB`, `psip`, …) | COVERED | `tests/field/test_boozer_analytic_jax.py:170-298` — `_compare_all_methods` compares against `sopp.BoozerAnalytic` CPU. | — | `direct_kernel` |

### `src/simsopt/field/interpolated_boozer_field_jax.py`

| Function | Status | Test path:line | Severity | Tolerance lane |
|---|---|---|---|---|
| `InterpolatedBoozerFieldJAX` public methods | COVERED | `tests/field/test_interpolated_boozer_field_jax.py:103-219` (`_compare_all_methods` vs `BoozerRadialInterpolant`). | — | depends on case (mostly `direct_kernel`/`reporting_contract`) |

### `src/simsopt/field/interpolated_field_jax.py`

| Function | Status | Test path:line | Severity | Tolerance lane |
|---|---|---|---|---|
| `InterpolatedFieldJAX` public B/GradAbsB with NFP/stellsym folding | COVERED | `tests/field/test_interpolated_field_jax_item15.py:172-280` (`TestInterpolatedFieldJAXParity`) vs `simsopt.field.InterpolatedField` CPU which itself routes through `simsoptpp.RegularGridInterpolant3D`. | — | `direct_kernel` |

### `src/simsopt/field/magneticfieldclasses_jax.py`

| Function | Status | Test path:line | Severity |
|---|---|---|---|
| MagneticField composition / sums / products | COVERED | `tests/field/test_magnetic_field_composition_jax.py:112-308` (`TestMagneticFieldSumJAXParity`, `TestMagneticFieldMultiplyJAXParity`) compares JAX composition against composed CPU `MagneticField` wrappers. The components are themselves C++-backed. | — |

### `src/simsopt/field/sampling_jax.py`

| Function | Status | Test path:line | Severity |
|---|---|---|---|
| `sample_weighted_indices_jax`, draw helpers | PARTIAL (statistical only) | `tests/field/test_sampling_jax_item22.py:158-358`. Uses moment-matching against upstream rejection sampling rather than a numerical C++ oracle. Acceptable for a sampler. | LOW |

### `src/simsopt/objectives/integral_bdotn_jax.py`

| Function | Status | Test path:line | Severity | Tolerance lane |
|---|---|---|---|---|
| `integral_BdotN` (three definitions) | COVERED | `tests/objectives/test_integral_bdotn_jax.py:303-329` (`TestIntegralBdotNCppParity.test_cpp_parity`) imports `simsoptpp as sopp` and asserts `np.testing.assert_allclose(J_jax, J_cpp, rtol=1e-13)`. Also boundary contracts in same class. Production-scale closeout in `tests/objectives/test_integral_bdotn_item10_closeout.py:175` via `sopp.integral_BdotN`. | — | `direct_kernel` |
| `signed_BdotN_flux` | NO C++ COUNTERPART | JAX-only convenience; tested via closed torus closure analytic identity. | — | — |
| `residual_BdotN` (internal) | NO C++ COUNTERPART | Internal residual builder. | — | — |

### `src/simsopt/objectives/fluxobjective_jax.py` (`SquaredFluxJAX`)

| Function | Status | Test path:line | Severity | Tolerance lane |
|---|---|---|---|---|
| `SquaredFluxJAX.J` (value) | COVERED | `tests/integration/test_stage2_jax.py:1026-1085` (`TestObjectiveValueParity`) vs `SquaredFlux(BiotSavart)`; closeout production-scale parity at `tests/objectives/test_fluxobjective_jax_item03_closeout.py:98-131` at `direct_kernel` rtol. | — | `direct_kernel` |
| `SquaredFluxJAX.dJ` (gradient) | PARTIAL | `tests/integration/test_stage2_jax.py:1091-1130` (`TestGradientParity`) vs `SquaredFlux(BiotSavart).dJ()` at e2e Stage-2 tolerance (`_STAGE2_GRADIENT_PARITY_RTOL=1e-11`, `atol=1e-15`). The `direct_kernel` lane gradient (`rtol=1e-10`) is not explicitly asserted by a dedicated assertion. | **HIGH** | `direct_kernel` requires explicit assertion (see item 5 above) |

### `src/simsopt/objectives/stage2_target_objective_jax.py`

| Function | Status | Test path:line | Severity |
|---|---|---|---|
| `build_stage2_target_objective` / helpers | NO C++ COUNTERPART | Composite objective with quadratic penalties, target labels, curvature, etc. The constituent kernels (`SquaredFluxJAX`, curve-objectives JAX) are individually tested; the composite is a JAX-only orchestration. Tested e2e in `test_stage2_jax.py`. | — |

### `src/simsopt/geo/surface_fourier_jax.py`

| Function | Status | Test path:line | Severity | Tolerance lane |
|---|---|---|---|---|
| `surface_gamma` (SurfaceXYZTensorFourier eval) | COVERED | `tests/geo/test_surface_fourier_jax.py:327-358` (`TestSurfaceFourierJaxCppParity.test_gamma_match`) — imports `SurfaceXYZTensorFourier` (C++-backed) and asserts `assert_allclose(gamma_jax, gamma_cpp, atol=1e-13)`. | — | `direct_kernel` |
| `surface_gammadash1/2`, `surface_normal`, `surface_unitnormal`, `surface_gammadash1dash1/2dash2/mixed` | COVERED | `tests/geo/test_surface_fourier_jax.py:375-540` (`TestSurfaceFourierJaxCppParity.test_gammadash_parity`, `TestSurfaceXYZFourierJaxCppParity:460-540`); spec sweep in `TestSurfaceFourierSpecCppParity:765-953`. | — | `direct_kernel` (value), `derivative_heavy` (first derivative), `derivative_heavy.second_derivative` (mixed) |
| `surface_area`, `surface_volume` | COVERED | `tests/geo/test_surface_fourier_jax.py:875-886` (`test_spec_geometry_and_normals_match_cpp`, area/volume blocks). | — | `derivative_heavy.scalar_value` |
| `darea_by_dcoeff`, `dvolume_by_dcoeff`, `d2area/d2volume` | COVERED | `tests/geo/test_surface_fourier_jax.py:955-1061` (`test_area_volume_derivatives_match_cpp`) — sweeps `SurfaceXYZFourier` and `SurfaceXYZTensorFourier`, both `stellsym ∈ {True, False}`, asserts `darea_by_dcoeff` and `d2area_by_dcoeffdcoeff` at `derivative_heavy.first/second_derivative` lanes. | — | `derivative_heavy` (first + second) |
| `dgamma_by_dcoeff` etc. (M3 jacobians) | COVERED | `tests/geo/test_boozer_derivatives_jax.py::TestDgammaByDcoeff`, `TestDgammaByDcoeffStellsym` (`tests/geo/test_boozer_derivatives_jax.py:435-563`). FD validated and compared against `SurfaceXYZTensorFourier.dgamma_by_dcoeff()` (C++ via `_simsoptpp`). | — | `derivative_heavy.first_derivative` |
| Clamped-dims variants (`_hats_with_clamping`, etc.) | COVERED | `tests/geo/test_surface_xyz_tensor_clamped_jax.py:61-172` (`test_unclamped_baseline_gamma_parity`, `test_clamped_combination_gamma_parity`, `test_clamped_normal_and_unitnormal_parity`). | — | `direct_kernel`, `derivative_heavy` |
| Stellsym scatter / scattering helpers | COVERED | `tests/geo/test_boozersurface_jax.py::TestStellsymScatterIndices:797-855`; `TestStellsymMaskCPUJAXParity:8102` for CPU/JAX scatter parity. | — | `direct_kernel` |

### `src/simsopt/geo/surface_fourier_jax_cpu_ordered.py`

| Function | Status | Test path:line | Severity |
|---|---|---|---|
| `_eval_hat_cpu_ordered`, etc. | COVERED | `tests/geo/test_surface_fourier_jax_cpu_ordered.py` compares ordered-reduction JAX against C++ surface in fixed-state byte-identity flavor; auxiliary to the strict-identity parity gate. | — |

### `src/simsopt/geo/boozer_residual_jax.py`

| Function | Status | Test path:line | Severity | Tolerance lane |
|---|---|---|---|---|
| `boozer_residual_scalar` | **MISSING** (claimed but not delivered) | Docstring (`tests/geo/test_boozer_residual_jax.py:8`) says "C++ parity (when simsoptpp is available)" but **no class in the file co-imports the C++ `boozer_residual` symbol** (`src/simsoptpp/boozerresidual_py.h` declares `boozer_residual` / `boozer_residual_ds_dc`). All assertions in `TestBoozerResidualScalar`, `TestBoozerResidualGradient`, `TestBoozerResidualHessian` compare against a NumPy reproduction (`_numpy_boozer_residual_reference`, lines 89-97) of the same formula — **tautological** by `REVIEWER_ORACLE_LINT.md` type "NumPy reproduction == JAX kernel". | **HIGH** | `direct_kernel` claim is not actually pinned |
| `boozer_residual_vector` | **MISSING** | Same as scalar — only NumPy-reproduction reference. | **HIGH** | `direct_kernel` |
| `boozer_residual_grad` | PARTIAL | FD vs JAX value in `TestBoozerResidualGradient` (`rtol=1e-5`). No C++ gradient oracle. | MED | `fd_gradient` only |
| `boozer_residual_hessian` | PARTIAL | Symmetry + FD-of-grad in `TestBoozerResidualHessian` (`rtol=1e-4`). No C++ Hessian oracle. | MED | `fd_gradient` only |
| `boozer_residual_scalar_and_grad_cpu_ordered` | COVERED | `tests/geo/test_boozersurface_jax.py:7729-7771` (`TestUpstreamFactoryBoozerMatrix.test_penalty_cpu_ordered_value_and_grad_cpu_parity_fixed_state`) — compares value+gradient against `BoozerSurface.boozer_penalty_constraints_vectorized(derivatives=1)`. | — | `direct_kernel` (value), `ls_wrapper_gradient` (gradient) |
| `boozer_penalty_composed` (M3) | COVERED | `tests/geo/test_boozer_derivatives_jax.py::TestBoozerPenaltyGradComposed:565`; CPU/JAX matrix parity via the wrapper class in `tests/geo/test_boozersurface_jax.py::TestUpstreamFactoryBoozerMatrix:7667-7727`. | — | `direct_kernel` (value), `ls_wrapper_gradient` (gradient), `direct_hessian_oracle` (Hessian sweep) |
| `boozer_penalty_grad_composed` | COVERED | Same suite as `boozer_penalty_composed`. | — | `ls_wrapper_gradient` |
| `boozer_residual_jacobian_composed` | COVERED | `tests/geo/test_boozer_derivatives_jax.py::TestBoozerResidualJacobianComposed:629`. Also compared to CPU exact-constraints Jacobian in `tests/geo/test_boozersurface_jax.py::TestUpstreamFactoryBoozerMatrix.test_exact_constraints_residual_and_jvp_cpu_parity_matrix:7918-7953`. | — | `derivative_heavy.first_derivative` |
| `boozer_residual_coil_vjp` | COVERED | `tests/geo/test_boozer_derivatives_jax.py::TestBoozerResidualCoilVJP:756-885`. | — | `derivative_heavy.first_derivative` |

### `src/simsopt/geo/boozersurface_jax.py` (`BoozerSurfaceJAX(Optimizable)`)

| Function | Status | Test path:line | Severity | Tolerance lane |
|---|---|---|---|---|
| Penalty objective `_make_penalty_objective_with` | COVERED | `tests/geo/test_boozersurface_jax.py::TestUpstreamFactoryBoozerMatrix.test_penalty_raw_inner_callback_cpu_parity_fixed_state:7682-7727` and `test_penalty_cpu_ordered_value_and_grad_cpu_parity_fixed_state:7729-7771`. Matrix sweep at `test_penalty_value_and_gradient_cpu_parity_matrix:7667-7680`. | — | `direct_kernel` (value), `ls_wrapper_gradient` (gradient) |
| Penalty Hessian (column-complete) | COVERED | `test_penalty_hessian_column_complete_cpu_parity_matrix:7821-7867` and `test_penalty_hessian_directional_cpu_parity_matrix:7776-7816`. | — | `direct_hessian_oracle`, `fd_gradient` (directional) |
| `_solve_dense_newton_step` | COVERED | `test_penalty_dense_newton_step_cpu_parity_fixed_state:7869-7913`. | — | `direct_hessian_oracle` |
| Exact constraints residual + JVP | COVERED | `test_exact_constraints_residual_and_jvp_cpu_parity_matrix:7918-7953`. | — | `derivative_heavy.first_derivative` |
| LS solve via `optimizer_backend="scipy"` (default reference path) | COVERED | The "raw inner callback" parity (above) is the gold check; the CPU/JAX LS wrapper parity gate `_pre_newton_census_gate_failures` (`benchmarks/single_stage_init_parity.py`) anchors strict byte parity. | — | `ls_wrapper_gradient` |
| LS solve via `optimizer_backend="ondevice"` / `hybrid` | INDIRECT | `tests/integration/test_single_stage_jax_cpu_reference.py` (whole single-stage outer-loop scenarios). No direct LS-trajectory-vs-CPU bit parity test besides the gate. | MED | — |
| `solve_residual_equation_exactly_newton` | PARTIAL | Exact KKT residual/Jacobian parity matched (above); operator-vs-dense parity in `test_exact_well_conditioned_operator_adjoint_cpu_gpu_same_state_parity:5047`. Adjoint vector parity at `exact_well_conditioned_adjoint` lane. | LOW | `exact_well_conditioned_adjoint` |
| `_compute_stellsym_mask_indices` | COVERED | `tests/geo/test_boozersurface_jax.py::TestStellsymMaskCPUJAXParity:8102`. | — | `direct_kernel` |
| `build_boozer_surface_runtime_state` | COVERED | `TestBuildBoozerSurfaceRuntimeState:8232`. | — | — |

### `src/simsopt/geo/surfaceobjectives_jax.py` (M5 IFT wrappers)

| Function | Status | Test path:line | Severity | Tolerance lane |
|---|---|---|---|---|
| `BoozerResidualJAX.J` | COVERED | `tests/integration/test_single_stage_jax_cpu_reference.py:546-588, 1686-1689` compares `BoozerResidualJAX.J()` against `BoozerResidual(booz_cpu, bs_cpu).J()`. | — | `direct_kernel` (residual value) |
| `BoozerResidualJAX.dJ` (adjoint via IFT) | COVERED | Same test file at `1598-1690` via `_real_resolve_fd_*` real-resolve reduced-FD sweep. Documented in `CLAUDE.md` (fixed-surface FD, branch-stable-resolve, exact-well-conditioned-adjoint). | — | `branch_stable_resolve`, `fd_gradient` |
| `IotasJAX.J` | COVERED | `test_single_stage_jax_cpu_reference.py:557-560, 1556`. | — | `direct_kernel` |
| `IotasJAX.dJ` | COVERED | Same as residual dJ — IFT adjoint coverage in real-resolve FD lane and exact-well-conditioned-adjoint lane. | — | `exact_well_conditioned_adjoint` for well-conditioned exact fixtures |
| `NonQuasiSymmetricRatioJAX.J / dJ` | COVERED | Same suite as above. | — | `branch_stable_resolve`, `fd_gradient` |
| Composite (BoozerResidualJAX + Iotas penalty) | COVERED | `test_single_stage_jax_cpu_reference.py:1649-1671` (`_real_resolve_fd_composite_*`). | — | `branch_stable_resolve` |
| Implicit-diff plumbing (operator adjoint, solver state) | NO C++ COUNTERPART | JAX-only IFT pipeline; only operator-vs-dense self-consistency required. | — | — |

### `src/simsopt/geo/optimizer_jax.py` / `optimizer_jax_reference.py`

| Function | Status | Test path:line | Severity |
|---|---|---|---|
| `jax_minimize`, `newton_polish`, `newton_exact` | NO C++ COUNTERPART | JAX-side line-search/Newton solver, no C++ analog. Tested against SciPy BFGS in `test_boozersurface_jax.py::TestOptimizerAdapter:978`. | — |
| Result converters | NO C++ COUNTERPART | `tests/geo/test_optimizer_result_converters.py`. | — |

### `src/simsopt/geo/label_constraints_jax.py`

| Function | Status | Test path:line | Severity | Tolerance lane |
|---|---|---|---|---|
| `toroidal_flux_jax` | PARTIAL | `tests/geo/test_label_constraints_jax.py:171-235` (`test_toroidal_flux_invariance`, `test_toroidal_flux_gradient_fd`). Both use FD-from-JAX (and an analytic 1% invariance test). No C++ toroidal-flux oracle co-import. | MED | `fd_gradient` only |
| `volume_jax`, `area_jax` (via `surface_volume`, `surface_area`) | COVERED | Indirectly through `tests/geo/test_surface_fourier_jax.py::TestSurfaceFourierSpecCppParity:875-886` (area/volume vs C++). | — | `derivative_heavy.scalar_value` |
| `compute_G_from_currents` | NO C++ COUNTERPART | Trivial sum-and-rescale helper. | — | — |

### `src/simsopt/geo/_distance_jax.py`

| Function | Status | Test path:line | Severity | Tolerance lane |
|---|---|---|---|---|
| `get_close_candidates_within_collection` | COVERED | `tests/geo/test_distance_jax.py:30-58` vs `sopp.get_pointclouds_closer_than_threshold_within_collection`. | — | `direct_kernel` |
| `get_close_candidates_between_collections` | COVERED | `tests/geo/test_distance_jax.py:60-95` vs `sopp.get_pointclouds_closer_than_threshold_between_two_collections`. | — | `direct_kernel` |

### `src/simsopt/geo/framedcurve_jax.py`

| Function | Status | Test path:line | Severity | Tolerance lane |
|---|---|---|---|---|
| Rotated centroid / Frenet frame helpers | PARTIAL | `tests/geo/test_framedcurve_jax_wrappers_item18.py:82, 191` — uses closed-form planar-circle analytic frame. No `simsoptpp.framedcurve`-backed C++ oracle, but framedcurve has no C++ implementation — these are sympy-based Python wrappers. | LOW | analytic |

### `src/simsopt/geo/permanent_magnet_grid_jax.py`

| Function | Status | Test path:line | Severity |
|---|---|---|---|
| Grid construction / projections | COVERED | `tests/geo/test_permanent_magnet_grid_jax_item27.py` (imports `simsoptpp as sopp`). | — |

### `src/simsopt/geo/optimizer_jax_reference.py`

| Function | Status | Test path:line | Severity |
|---|---|---|---|
| Reference SciPy adapter | NO C++ COUNTERPART | SciPy-only; tested against `scipy.optimize.minimize` ground truth. | — |

### `src/simsopt/solve/permanent_magnet_optimization_jax.py`

| Function | Status | Test path:line | Severity | Tolerance lane |
|---|---|---|---|---|
| MwPGP algorithm | COVERED | `tests/solve/test_permanent_magnet_optimization_jax_item28.py:351` vs `simsoptpp.MwPGP_algorithm`. Helpers vs `sopp` CPU oracles at line 178+. | — | `pm_mwpgp_fixed_step` |
| GPMO | COVERED | Same file at `587` vs `simsoptpp.GPMO_ArbVec`. | — | `pm_mwpgp_fixed_step` |

### `src/simsopt/solve/wireframe_optimization_jax.py`

| Function | Status | Test path:line | Severity | Tolerance lane |
|---|---|---|---|---|
| Regularized constrained LS / RCLS / GSCO | COVERED | `tests/solve/test_wireframe_optimization_jax_item31.py:236-623` — multiple `sopp.GSCO` and CPU-wrapper parity matrix tests. | — | `direct_kernel` |

### `src/simsopt/jax_core/biotsavart_cpu_ordered.py`

| Function | Status | Test path:line | Severity |
|---|---|---|---|
| CPU-ordered Biot-Savart reductions | COVERED | `tests/field/test_biotsavart_jax_cpu_ordered.py`. Strict byte-identity vs C++ used by the dual-mode parity gate. | — |

### `src/simsopt/jax_core/curve_geometry.py` / `surface_rzfourier.py` / `surface_henneberg.py` / `surface_classifier.py` / etc.

| Function | Status | Test path:line | Severity | Tolerance lane |
|---|---|---|---|---|
| `CurveXYZFourierSymmetries.spec_gamma/_gammadash/_gammadashdash` | COVERED | `tests/geo/test_curvexyzfouriersymmetries_spec_jax.py:294-329` vs `curve.gamma()`, `curve.gammadash()`, `curve.gammadashdash()` (all C++-backed). | — | `direct_kernel`, `derivative_heavy` |
| `SurfaceRZFourier` JAX kernels | COVERED | `tests/geo/test_surface_rzfourier_jax.py:241-416` direct value/derivative parity vs CPU `surface.gamma() / gammadash1/2 / normal()`; production-scale closeout in `tests/geo/test_surface_rzfourier_jax_item06_closeout.py:98-159`. | — | `direct_kernel`, `derivative_heavy` |
| `SurfaceHenneberg` JAX | COVERED | `tests/geo/test_surface_henneberg_jax.py:142-288` (`test_gamma_matches_cpu_oracle*`, `test_gammadash1/2`, `test_normal`, `test_unitnormal`, `test_area/volume`). | — | `direct_kernel`, `derivative_heavy` |
| `SurfaceGarabedian` to-RZ converter | COVERED | `tests/geo/test_surface_garabedian_jax.py:109-138`. | — | `direct_kernel` |
| `BoozerRadialInterpolant` JAX wrapper | COVERED | `tests/jax_core/test_boozer_radial_interp_jax_item32.py:210-386` direct parity vs `sopp.compute_kmnc_kmns`, `sopp.compute_kmns`, `sopp.fourier_transform_{odd,even}`. | — | `direct_kernel` |
| `BoozerRadialInterpolant` fixed-state evaluator | COVERED | `tests/jax_core/test_boozer_fixed_state_jax_item33.py:246-309` vs sopp inverse-Fourier oracle. | — | `direct_kernel` |
| `analytic_fields` (Dommaschk, Reiman) | COVERED | See dommaschk/reiman entries above. | — | — |
| `regular_grid_interp` | COVERED | `tests/jax_core/test_regular_grid_interp_item13.py` vs `sopp.RegularGridInterpolant3D`. | — | `direct_kernel` |
| `tracing.py` (RK4/Dopri5 + event localization) | PARTIAL | `tests/jax_core/test_tracing_jax_item14.py:147-462`. RK step `direct_kernel` vs analytic toroidal-axis solution; endpoint vs `sopp` field-line tracing only matched at `event_time_tracing` lane (loose tolerance — accepted by lane contract). | — | `event_time_tracing` |
| Guiding-center / gc Boozer / full-orbit tracing | COVERED at `event_time_tracing` | `tests/jax_core/test_tracing_jax_guiding_center.py:184-274` (`test_trace_guiding_center_endpoint_matches_upstream_particle_tracing`) vs `sopp.particle_guiding_center_tracing`. | — | `event_time_tracing` |
| `finitebuild` filament helpers | PARTIAL | `tests/geo/test_finitebuild_jax_item20.py:73-252` uses closed-form planar-circle analytic oracle + FD gradient. No direct `simsoptpp.coil.finitebuild` C++ symbol (finite-build is Python-only upstream). | LOW | analytic + `fd_gradient` |
| `sharding.py`, `reductions.py`, `_device_scalars.py`, `_math_utils.py`, `_elliptic.py` | NO C++ COUNTERPART | JAX-only utilities. | — | — |
| `wireframe.py` JAX core kernels | COVERED | See `test_wireframe_jax_item29.py` entry above. | — | — |
| `sampling.py` | PARTIAL (statistical) | See sampling_jax entry above. | LOW | — |
| `scalar_potential_rz.py` JAX kernels | PARTIAL | See scalar_potential_rz_jax entry above. The CPU wrapper is sympy-based, not C++; the oracle is the sympy-derived NumPy lambda — acceptable as type 2 (closed-form) per `REVIEWER_ORACLE_LINT.md`. | LOW | — |
| `objectives_flux.py` (`flux_value_from_spec`, `flux_gradient_from_spec`) | PARTIAL | `tests/objectives/test_fluxobjective_jax_parity.py` uses strict-JAX kernels on both sides of the comparison — host-helper `_flux_kernel_value_and_grad` routes through the same JAX path. `direct-kernel` parity for value is delivered via `integral_BdotN`. Gradient parity at `direct-kernel` lane needs a `B_vjp`-mediated CPU oracle. | **HIGH** for gradient | `direct_kernel` |
| `specs.py` (spec dataclasses) | NO C++ COUNTERPART | Pure pytree contracts. | — | — |
| `_sympy_to_jax.py` (sympy lambdify shim) | NO C++ COUNTERPART | Sympy translation; tested via roundtrip identity. | — | — |
| `field.py` (composition) | NO C++ COUNTERPART | JAX composition adapters; tested e2e in field-composition tests. | — | — |
| `surface_classifier.py` | PARTIAL | `tests/jax_core/test_tracing_jax_levelset_events.py:355+`. No direct `simsoptpp.SurfaceClassifier` co-import in level-set tests, but classifier itself wraps a C++ interpolant (`simsoptpp.RegularGridInterpolant3D`) which is parity-tested elsewhere. | LOW | — |
| `pm_optimization.py` | COVERED | See solve entry above. | — | — |

### `src/simsopt/backend.py`, `src/simsopt/jax_core/_device_scalars.py`, etc.

| Function | Status | Severity |
|---|---|---|
| Backend selection / device residency helpers | NO C++ COUNTERPART | Plumbing. |

---

## MISSING HIGH-severity — Test Sketches

### M-1. `biotsavart_jax.biot_savart_A` direct C++ parity

**Sketch:**
```python
# tests/field/test_biotsavart_jax.py — extend TestBiotSavartJaxCppParity
def test_A_parity_ncsx(self):
    bs, points_np, gammas_np, gds_np, currents_np = _ncsx_biotsavart_parity_fixture()
    A_ref = bs.A()  # C++ via simsopt.field.BiotSavart
    A_jax = biot_savart_A(
        jnp.array(points_np), jnp.array(gammas_np),
        jnp.array(gds_np), jnp.array(currents_np),
    )
    np.testing.assert_allclose(np.array(A_jax), A_ref, rtol=1e-10, atol=1e-12)
```

- **Fixture**: existing `_ncsx_biotsavart_parity_fixture` (line 324) already builds the simsoptpp-backed `BiotSavart`.
- **Oracle**: type 1, `simsopt.field.BiotSavart` (C++ via `simsoptpp.BiotSavart`).
- **Lane**: `direct_kernel` (`rtol=1e-10`, `atol=1e-12`).
- **Add**: a mirror `test_dA_by_dX_parity_ncsx` against `bs.dA_by_dX()` at the `derivative_heavy.first_derivative` lane.

### M-2. `biot_savart_d2B_by_dXdX` / `d2A_by_dXdX` direct C++ parity

**Sketch:**
```python
def test_d2B_by_dXdX_parity_ncsx(self):
    bs, points_np, gammas_np, gds_np, currents_np = _ncsx_biotsavart_parity_fixture()
    d2B_ref = bs.d2B_by_dXdX()  # add to BiotSavart if not surfaced; sopp exposes it
    d2B_jax = biot_savart_d2B_by_dXdX(
        jnp.array(points_np), jnp.array(gammas_np),
        jnp.array(gds_np), jnp.array(currents_np),
    )
    np.testing.assert_allclose(
        np.array(d2B_jax), d2B_ref,
        rtol=_DERIVATIVE_HEAVY_TOLS["second_derivative_rtol"],
        atol=_DERIVATIVE_HEAVY_TOLS["second_derivative_atol"],
    )
```

- **Oracle**: type 1, `simsoptpp.BiotSavart.d2B_by_dXdX()`. The C++ symbol exists at `src/simsoptpp/biot_savart_py.cpp` (BiotSavart class) but the Python `BiotSavart.d2B_by_dXdX()` wrapper must be confirmed live in `simsopt.field.biotsavart`.
- **Lane**: `derivative_heavy.second_derivative` (`rtol=1e-6`, `atol=1e-8`).
- **Risk** if absent: silent drift in 2nd-order field derivatives used by exact Boozer Newton refinement and Hessian probes.

### M-3. `biot_savart_B_vjp` direct C++ parity

**Sketch:**
```python
def test_B_vjp_parity_ncsx(self):
    bs, points_np, gammas_np, gds_np, currents_np = _ncsx_biotsavart_parity_fixture()
    rng = np.random.default_rng(20260513)
    v = rng.standard_normal(points_np.shape)
    # CPU oracle: contract B against v, autodiff via the C++ wrapper
    bs.set_points(points_np)
    B_cpu = bs.B()
    cpu_grad_gammas, cpu_grad_gds, cpu_grad_I = bs.B_vjp(v)  # CPU C++ VJP
    jax_grad_gammas, jax_grad_gds, jax_grad_I = biot_savart_B_vjp(
        jnp.array(points_np), jnp.array(v),
        jnp.array(gammas_np), jnp.array(gds_np), jnp.array(currents_np),
    )
    np.testing.assert_allclose(
        np.array(jax_grad_gammas), cpu_grad_gammas,
        rtol=_DERIVATIVE_HEAVY_TOLS["first_derivative_rtol"],
        atol=_DERIVATIVE_HEAVY_TOLS["first_derivative_atol"],
    )
    # same for gds and I
```

- **Oracle**: type 1, `simsoptpp.BiotSavart.B_vjp(v)` (exposed via `simsoptpp` Python bindings; see `src/simsoptpp/biot_savart_vjp_py.h`).
- **Lane**: `derivative_heavy.first_derivative`.
- **Important**: today's chunked-self-consistency tests use the JAX dense kernel as the reference. Per `REVIEWER_ORACLE_LINT.md`, that is **tautological** — flag and replace.

### M-4. `boozer_residual_scalar/_vector/_grad/_hessian` direct C++ parity

**Sketch:**
```python
# tests/geo/test_boozer_residual_jax.py — add TestBoozerResidualCppParity
class TestBoozerResidualCppParity:
    @pytest.fixture(autouse=True)
    def _require_simsoptpp(self):
        pytest.importorskip("simsoptpp")

    def test_residual_vector_parity(self):
        import simsoptpp as sopp
        B, xphi, xtheta = _make_synthetic_data(8, 10)
        G, iota = 1.5, 0.3
        weight_inv_modB = True
        # JAX side
        r_jax = boozer_residual_vector(G, iota, B, xphi, xtheta,
                                       weight_inv_modB=weight_inv_modB)
        # C++ oracle (sopp.boozer_residual returns scalar + residual vector)
        B_np = host_array(B); xphi_np = host_array(xphi); xtheta_np = host_array(xtheta)
        r_cpp = sopp.boozer_residual(G, iota, B_np, xphi_np, xtheta_np, weight_inv_modB)
        np.testing.assert_allclose(np.array(r_jax), r_cpp, rtol=1e-10, atol=1e-12)

    def test_residual_scalar_parity(self):
        # same fixture, compare scalar value
        ...

    def test_residual_grad_parity(self):
        # compare boozer_residual_grad vs sopp.boozer_residual_ds_dc (or equivalent)
        ...
```

- **Oracle**: type 1, `simsoptpp.boozer_residual` / `simsoptpp.boozer_residual_ds_dc` (declared at `src/simsoptpp/boozerresidual_py.h` and exposed through `python.cpp`). Confirm the exact Python entry-point name; if absent at the package surface, use `simsoptpp.BoozerSurfaceCpp.boozer_residual` (via `_simsoptpp_boozer_compat`).
- **Lane**: `direct_kernel` (rtol=1e-10) for value/vector; `derivative_heavy.first_derivative` for `_grad`; `direct_hessian_oracle` for `_hessian` (matching the existing column-complete Hessian convention).
- **Why HIGH**: removes the largest tautology in the JAX Boozer test surface and closes the gap claimed by the docstring at `tests/geo/test_boozer_residual_jax.py:8`.

### M-5. `SquaredFluxJAX.dJ` `direct_kernel` lane gradient parity

**Sketch:**
```python
# tests/integration/test_stage2_jax.py — add to TestGradientParity
def test_dJ_direct_kernel_lane_parity_ncsx(self):
    # Build a fixed-state CPU vs JAX setup; do NOT route through Stage 2 e2e.
    bs_cpu = BiotSavart(coils)
    bs_cpu.set_points(surf.gamma().reshape((-1, 3)))
    jf_cpu = SquaredFlux(surf, bs_cpu, definition="quadratic flux")
    grad_cpu = np.asarray(jf_cpu.dJ())  # CPU C++ VJP chain
    # JAX side via the native fluxobjective wrapper
    sf_jax = SquaredFluxJAX.from_specs(...)  # see existing factories
    grad_jax = np.asarray(sf_jax.dJ())
    tols = parity_ladder_tolerances("direct_kernel")
    np.testing.assert_allclose(grad_jax, grad_cpu, rtol=tols["rtol"], atol=tols["atol"])
```

- **Oracle**: type 1, the CPU `SquaredFlux(surf, bs_cpu).dJ()` which composes `BiotSavart.B_vjp` (C++) with `integral_BdotN_vjp` (C++). The two C++ kernels are individually validated by other tests, but their composed gradient is currently only checked at Stage-2 e2e tolerance.
- **Lane**: `direct_kernel` (rtol=1e-10) — same lane the value test already targets.

---

## Tautological-Coverage Flags (Violations of `REVIEWER_ORACLE_LINT.md`)

These tests claim parity but the "oracle" is the JAX kernel under another name or a NumPy reproduction of the same formula. Per the lint policy these must be replaced.

1. **`tests/geo/test_boozer_residual_jax.py:89-97`** — `_numpy_boozer_residual_reference` is a literal NumPy reimplementation of the formula in `boozer_residual_scalar`/`boozer_residual_vector`. Every assertion in `TestBoozerResidualScalar` (lines 162-348), `TestBoozerResidualGradient`, and `TestBoozerResidualParityStress` (lines 387-450) compares JAX against this reproduction. Violation type: "NumPy reproduction of the JAX formula". Fix: replace with C++ `sopp.boozer_residual` co-import (sketch M-4).
2. **`tests/field/test_biotsavart_jax.py:585-599`** (`test_B_vjp_rebuilds_when_tuning_changes_in_process`) — the `_dense_B_vjp(chunked_bs, …)` reference (helper at line 294) executes the same JAX `_one_point_dense` kernel inside `chunked_bs`. The assertion `assert_allclose(chunked_leaf, dense_leaf)` compares a JAX chunked path against a JAX dense path. Violation type: "jax_path(x) == host_path(x) where host_path routes through JAX". Fix: add `sopp.BiotSavart.B_vjp` co-import (sketch M-3).
3. **`tests/field/test_biotsavart_jax.py:421-428`** (`test_B_and_dB_consistent`) — compares `biot_savart_B_and_dB` to `biot_savart_B` + `biot_savart_dB_by_dX` (same JAX kernel, different entry points). Tautology by construction. Acceptable as a "different entry point" smoke check, but should not be cited as a parity oracle.
4. **`tests/objectives/test_fluxobjective_jax_parity.py:279-308`** — `_flux_kernel_value_and_grad` helper builds its reference by autodiffing the same JAX flux kernel as the production path. Any `value`/`gradient` parity check that uses this helper as the oracle is tautological. Use a `BiotSavart`-backed `SquaredFlux` (C++ chain) as the reference instead.
5. **`tests/field/test_biotsavart_jax.py:638-651`** (`test_two_chunk_coil_and_quadrature_paths_match_dense_reference`) — `dense_B = _dense_reference_fields(chunked_bs, …)` is itself a JAX call into `chunked_bs._one_point_dense`. This is "JAX vs JAX" chunking parity, not a C++ oracle parity. Acceptable as an internal consistency check; do not let it stand in for a `direct_kernel` lane C++ assertion.

---

## Notes on Status Definitions

- **COVERED** requires (a) the JAX symbol and a C++ symbol both imported in the same test, and (b) `np.testing.assert_allclose` against the C++ result at the lane tolerance from `benchmarks/validation_ladder_contract.py::PARITY_LADDER_TOLERANCES`.
- **PARTIAL** = same as COVERED but only for a subset of the public API (e.g., value but not gradient; one surface family but not all variants; FD or analytic oracle only, no C++).
- **INDIRECT** = the JAX symbol's parity is implicit (a higher-level integration test exercises it transitively) but no direct kernel oracle is asserted.
- **MISSING** = no test imports both the JAX symbol and a C++ analog. Severity = HIGH for core kernels, MED for derivatives/VJPs, LOW for adapter glue.
- **NO C++ COUNTERPART** = JAX-only orchestration (composition adapters, IFT plumbing, JIT/cache helpers, pytree dataclasses, sympy lambdify shim, sharding, reductions).

## Cross-References

- `tests/REVIEWER_ORACLE_LINT.md` — oracle-lint policy.
- `benchmarks/validation_ladder_contract.py::PARITY_LADDER_TOLERANCES` — tolerance lanes.
- `benchmarks/single_stage_init_parity.py::_pre_newton_census_gate_failures` — the production strict byte-identity gate (separate from the per-kernel parity tests audited here).
- `.artifacts/jax-test-audit-2026-05-13/TEST_QUALITY_TODOS.md` — companion test-quality audit listing Tier-1 tautologies.
