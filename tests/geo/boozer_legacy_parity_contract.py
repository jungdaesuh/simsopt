"""Static BoozerSurface legacy-to-JAX parity contract matrix."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ParityCategory = Literal[
    "strict_parity",
    "jax_native_equivalent",
    "intentional_exclusion",
    "unsupported_jax_contract",
]

ToleranceLane = Literal[
    "direct_kernel",
    "ls_wrapper_gradient",
    "derivative_heavy",
    "exact_well_conditioned_adjoint",
    "exact_ill_conditioned_adjoint",
    "branch_stable_resolve",
    "fd_gradient",
    "gpu_runtime",
    "reduction_cpu_gpu",
]


@dataclass(frozen=True)
class BoozerLegacyParityEntry:
    legacy_test: str
    category: ParityCategory
    owner_file: str
    owner_test: str
    tolerance_lane: ToleranceLane | None
    requires_simsoptpp: bool
    requires_cuda: bool
    notes: str


BOOZER_LEGACY_PARITY_CONTRACT: tuple[BoozerLegacyParityEntry, ...] = (
    BoozerLegacyParityEntry(
        legacy_test="test_call_boozer_residual_falls_back_to_alpha_only_signature",
        category="unsupported_jax_contract",
        owner_file="src/simsopt/geo/_simsoptpp_boozer_compat.py",
        owner_test="cpu_cpp_compat_only_no_jax_fallback",
        tolerance_lane=None,
        requires_simsoptpp=True,
        requires_cuda=False,
        notes="Legacy alpha-only simsoptpp compatibility shim; JAX must not add a fallback.",
    ),
    BoozerLegacyParityEntry(
        legacy_test="test_call_boozer_residual_ds_falls_back_to_alpha_only_signature",
        category="unsupported_jax_contract",
        owner_file="src/simsopt/geo/_simsoptpp_boozer_compat.py",
        owner_test="cpu_cpp_compat_only_no_jax_fallback",
        tolerance_lane=None,
        requires_simsoptpp=True,
        requires_cuda=False,
        notes="Legacy derivative shim is CPU/C++ compatibility only.",
    ),
    BoozerLegacyParityEntry(
        legacy_test="test_call_boozer_residual_ds2_falls_back_to_alpha_only_signature",
        category="unsupported_jax_contract",
        owner_file="src/simsopt/geo/_simsoptpp_boozer_compat.py",
        owner_test="cpu_cpp_compat_only_no_jax_fallback",
        tolerance_lane=None,
        requires_simsoptpp=True,
        requires_cuda=False,
        notes="Legacy Hessian shim is CPU/C++ compatibility only.",
    ),
    BoozerLegacyParityEntry(
        legacy_test="test_solver_signatures_do_not_expose_vectorize",
        category="strict_parity",
        owner_file="tests/geo/test_boozersurface_jax.py",
        owner_test="TestBoozerSurfaceJAXClass::test_public_solver_signatures_do_not_expose_vectorize",
        tolerance_lane=None,
        requires_simsoptpp=False,
        requires_cuda=False,
        notes="Public JAX solver signatures do not expose the legacy vectorize knob.",
    ),
    BoozerLegacyParityEntry(
        legacy_test="test_constructor_rejects_spoof_surface_names",
        category="strict_parity",
        owner_file="tests/geo/test_boozersurface_jax.py",
        owner_test="TestBoozerSurfaceJAXClass::test_instantiation_rejects_grouped_extractor_only_adapter; TestBoozerSurfaceJAXClass::test_instantiation_rejects_hidden_coils_list_adapter; TestNegativeCases::test_extract_grouped_coil_set_spec_rejects_legacy_coils_fallback",
        tolerance_lane=None,
        requires_simsoptpp=False,
        requires_cuda=False,
        notes="JAX constructor boundaries must reject unsupported adapters by concrete contract, not spoofed names.",
    ),
    BoozerLegacyParityEntry(
        legacy_test="test_duplicate_import_surface_helpers_accept_canonical_surface_shapes",
        category="unsupported_jax_contract",
        owner_file="src/simsopt/geo/_simsoptpp_boozer_compat.py",
        owner_test="cpu_cpp_compat_only_no_hidden_jax_import_fallback",
        tolerance_lane=None,
        requires_simsoptpp=True,
        requires_cuda=False,
        notes="Duplicate-import helper acceptance belongs to the CPU helper layer only.",
    ),
    BoozerLegacyParityEntry(
        legacy_test="test_run_code_rejects_G_none_with_free_currents",
        category="strict_parity",
        owner_file="tests/geo/test_boozersurface_jax.py",
        owner_test="TestBoozerSurfaceJAXClass::test_run_code_rejects_G_none_with_free_currents",
        tolerance_lane=None,
        requires_simsoptpp=False,
        requires_cuda=False,
        notes="JAX public run_code rejects free public coil currents when G is implicit.",
    ),
    BoozerLegacyParityEntry(
        legacy_test="test_run_code_rejects_G_none_with_free_legacy_currents",
        category="strict_parity",
        owner_file="tests/geo/test_boozersurface_jax.py",
        owner_test="TestBoozerSurfaceJAXClass::test_run_code_rejects_G_none_with_free_legacy_currents",
        tolerance_lane=None,
        requires_simsoptpp=False,
        requires_cuda=False,
        notes="Existing legacy-current guard coverage owns this path.",
    ),
    BoozerLegacyParityEntry(
        legacy_test="test_none_G_coil_gradient_callback_rejects_free_currents",
        category="strict_parity",
        owner_file="tests/geo/test_boozersurface_jax.py",
        owner_test="TestBoozerSurfaceJAXClass::test_none_G_coil_gradient_callback_rejects_free_currents",
        tolerance_lane=None,
        requires_simsoptpp=False,
        requires_cuda=False,
        notes="Adjoint callback guard must use the same fixed/free current contract as run_code.",
    ),
    BoozerLegacyParityEntry(
        legacy_test="test_none_G_coil_gradient_callback_allows_explicit_G",
        category="strict_parity",
        owner_file="tests/geo/test_boozersurface_jax.py",
        owner_test="TestBoozerSurfaceJAXClass::test_none_G_coil_gradient_callback_allows_explicit_G",
        tolerance_lane=None,
        requires_simsoptpp=False,
        requires_cuda=False,
        notes="Explicit G is allowed even when currents are not all fixed.",
    ),
    BoozerLegacyParityEntry(
        legacy_test="test_residual",
        category="strict_parity",
        owner_file="tests/geo/test_boozersurface_jax.py",
        owner_test="TestUpstreamFactoryBoozerMatrix::test_exact_surface_scalar_residual_matches_legacy_cpu_state",
        tolerance_lane="direct_kernel",
        requires_simsoptpp=True,
        requires_cuda=False,
        notes="Same exact NCSX state scalar residual parity against the CPU/C++ oracle.",
    ),
    BoozerLegacyParityEntry(
        legacy_test="test_boozer_penalty_constraints_gradient",
        category="strict_parity",
        owner_file="tests/geo/test_boozersurface_jax.py",
        owner_test="TestUpstreamFactoryBoozerMatrix::test_penalty_value_and_gradient_cpu_parity_matrix",
        tolerance_lane="ls_wrapper_gradient",
        requires_simsoptpp=True,
        requires_cuda=False,
        notes="Penalty value and gradient parity over both supported upstream surface families, stellsym modes, and optimize_G modes.",
    ),
    BoozerLegacyParityEntry(
        legacy_test="test_boozer_penalty_constraints_hessian",
        category="strict_parity",
        owner_file="tests/geo/test_boozersurface_jax.py",
        owner_test="TestUpstreamFactoryBoozerMatrix::test_penalty_hessian_directional_cpu_parity_matrix",
        tolerance_lane="fd_gradient",
        requires_simsoptpp=True,
        requires_cuda=False,
        notes="Same-state CPU Hessian against JAX HVP directional parity over both upstream surface families, stellsym modes, and optimize_G modes.",
    ),
    BoozerLegacyParityEntry(
        legacy_test="test_boozer_constrained_jacobian",
        category="strict_parity",
        owner_file="tests/geo/test_boozersurface_jax.py",
        owner_test="TestUpstreamFactoryBoozerMatrix::test_exact_constraints_residual_and_jvp_cpu_parity_matrix",
        tolerance_lane="derivative_heavy",
        requires_simsoptpp=True,
        requires_cuda=False,
        notes="Exact constrained residual and seeded Jacobian-vector parity against the CPU oracle over both upstream surface families, stellsym modes, and optimize_G modes.",
    ),
    BoozerLegacyParityEntry(
        legacy_test="test_boozer_surface_optimisation_convergence",
        category="strict_parity",
        owner_file="tests/integration/test_single_stage_jax_cpu_reference.py",
        owner_test="TestRunCodeLSParity::test_ls_solve_parity; TestExactSolveCPUJAXParity::test_exact_solve_parity",
        tolerance_lane="branch_stable_resolve",
        requires_simsoptpp=True,
        requires_cuda=False,
        notes="Solver-result parity is final public metrics, not iteration identity.",
    ),
    BoozerLegacyParityEntry(
        legacy_test="test_boozer_serialization",
        category="intentional_exclusion",
        owner_file="tests/geo/test_boozersurface_jax.py",
        owner_test="TestBuildBoozerSurfaceRuntimeState::test_runtime_state_round_trip_compares_values_not_identity",
        tolerance_lane=None,
        requires_simsoptpp=False,
        requires_cuda=False,
        notes="Mutable object graph identity is excluded; immutable runtime-state round trip is required.",
    ),
    BoozerLegacyParityEntry(
        legacy_test="test_run_code",
        category="strict_parity",
        owner_file="tests/geo/test_boozersurface_jax.py",
        owner_test="TestBoozerSurfaceJAXClass::test_run_code_ls_converges; TestBoozerSurfaceJAXClass::test_run_code_idempotent; TestBoozerSurfaceJAXExactPath::test_run_code_exact_converges",
        tolerance_lane="branch_stable_resolve",
        requires_simsoptpp=True,
        requires_cuda=False,
        notes="Split into solver-result parity, immutable idempotence, G=None guard, and exact solve coverage.",
    ),
    BoozerLegacyParityEntry(
        legacy_test="test_minimize_boozer_penalty_constraints_ls_manual",
        category="strict_parity",
        owner_file="tests/geo/test_boozersurface_jax.py",
        owner_test="TestBoozerSurfaceJAXClass::test_public_manual_ls_api_supports_baseline_demo_sequence; TestBoozerSurfaceJAXClass::test_public_manual_ls_api_matches_legacy_manual_linear_contract",
        tolerance_lane="branch_stable_resolve",
        requires_simsoptpp=True,
        requires_cuda=False,
        notes="Manual LS remains a supported JAX public API path.",
    ),
    BoozerLegacyParityEntry(
        legacy_test="test_need_to_run_code_false",
        category="jax_native_equivalent",
        owner_file="tests/geo/test_boozersurface_jax.py",
        owner_test="TestBoozerSurfaceJAXClass::test_run_code_idempotent; TestBoozerSurfaceJAXClass::test_same_input_same_result_without_res_identity; TestBoozerSurfaceJAXExactPath::test_exact_idempotent",
        tolerance_lane="branch_stable_resolve",
        requires_simsoptpp=False,
        requires_cuda=False,
        notes="Object identity of cached res is excluded; same-input same-result behavior is the JAX contract.",
    ),
    BoozerLegacyParityEntry(
        legacy_test="test_minimize_boozer_exact_constraints_newton_G_None",
        category="strict_parity",
        owner_file="tests/geo/test_boozersurface_jax.py",
        owner_test="TestBoozerSurfaceJAXClass::test_public_exact_constraints_newton_restores_cpu_api[False-True]; TestBoozerSurfaceJAXClass::test_public_exact_constraints_newton_restores_cpu_api[False-False]",
        tolerance_lane="exact_well_conditioned_adjoint",
        requires_simsoptpp=True,
        requires_cuda=False,
        notes="Exact Newton with implicit G needs a public JAX result contract or explicit unsupported-JAX proof.",
    ),
    BoozerLegacyParityEntry(
        legacy_test="test_minimize_boozer_exact_constraints_newton_stellsym_false",
        category="strict_parity",
        owner_file="tests/geo/test_boozersurface_jax.py",
        owner_test="TestBoozerSurfaceJAXClass::test_public_exact_constraints_newton_nonstellsym_stays_native_without_root; TestBoozerSurfaceJAXClass::test_public_exact_constraints_newton_nonstellsym_uses_full_jacobian_solve",
        tolerance_lane="exact_well_conditioned_adjoint",
        requires_simsoptpp=True,
        requires_cuda=False,
        notes="Non-stellsym exact path is native JAX and must compare final public metrics.",
    ),
    BoozerLegacyParityEntry(
        legacy_test="test_boozer_surface_quadpoints",
        category="strict_parity",
        owner_file="tests/geo/test_boozersurface_jax.py",
        owner_test="TestMixedQuadratureBoozer::test_instantiation; TestStellsymMaskCPUJAXParity::test_mask_matches_cpu_surface",
        tolerance_lane="direct_kernel",
        requires_simsoptpp=True,
        requires_cuda=False,
        notes="Quadpoint and stellsym mask behavior must stay one-for-one with legacy setup.",
    ),
    BoozerLegacyParityEntry(
        legacy_test="test_boozer_surface_type_assert",
        category="strict_parity",
        owner_file="tests/geo/test_boozersurface_jax.py",
        owner_test="TestBoozerSurfaceJAXClass::test_constructor_rejects_spoof_surface_names; TestBoozerSurfaceJAXClass::test_constructor_rejects_unknown_explicit_surface_kind; TestBoozerSurfaceJAXClass::test_surface_geometry_rejects_unknown_surface_kind; TestUpstreamFactoryBoozerMatrix::test_runtime_state_accepts_supported_surface_family_matrix; TestUpstreamFactoryBoozerMatrix::test_exact_surface_factory_rejects_surface_xyzfourier",
        tolerance_lane=None,
        requires_simsoptpp=False,
        requires_cuda=False,
        notes="Accepted/rejected JAX surface families use real classes or explicit jax_surface_kind contracts.",
    ),
)

BOOZER_LEGACY_PARITY_BY_TEST = {
    entry.legacy_test: entry for entry in BOOZER_LEGACY_PARITY_CONTRACT
}
