"""Dependency-light ladder contract helpers shared by launchers and probes."""

from __future__ import annotations

from typing import Union


SHORT_RUN_SMOKE_MAXITER = 20
_SMOKE_STAGE2_RUNG_NAMES = ("stage2_cold", "stage2_warm")
_GEOMETRY_REPRO_STAGE2_RUNG_NAME = "stage2_warm_repro"
TIER3_SINGLE_STAGE_OUTER_LOOP_RUNG = "tier3_single_stage_outer_loop"

ParityToleranceValue = Union[float, bool, None]

OPTIMIZER_DRIFT_TOLERANCES = {
    "tier1_stage2_value_gradient": {
        "objective_rel_tol": 1e-10,
        "gradient_rtol": 1e-9,
        "gradient_atol": 1e-12,
    },
    "tier2_stage2_e2e": {
        "final_objective_rel_tol_20_iter": 5e-4,
        "final_objective_rel_tol_default": 1e-4,
        "field_error_rel_tol": 1e-4,
        "geometry_rel_tol_20_iter": None,
        "geometry_rel_tol_default": 1e-6,
    },
    "tier3_single_stage_init": {
        "final_iota_abs_tol": 1e-10,
        "final_volume_rel_tol": 1e-10,
        "field_error_rel_tol": 1e-8,
        "surface_geometry_rel_tol": 1e-9,
    },
    "tier4_adjoint_fd": {
        "adjoint_residual_rel_tol": 1e-10,
        "recomposed_total_rel_tol": 1e-12,
        "fixed_surface_fd_rel_tol": 1e-3,
        "fixed_surface_fd_abs_tol": 1e-8,
        "full_resolve_fd_rel_tol": 1e-2,
        "full_resolve_fd_abs_tol": 1e-8,
    },
    "optimizer_state_parity": {
        "x_rtol": 1e-6,
        "x_atol": 1e-8,
        "objective_rel_tol": 1e-6,
        "gradient_rtol": 1e-6,
        "gradient_atol": 1e-8,
        "jac_norm_inf_abs_tol": 1e-8,
    },
}

PARITY_LADDER_TOLERANCES: dict[str, dict[str, ParityToleranceValue]] = {
    "direct_kernel": {
        "rtol": 1e-10,
        "atol": 1e-12,
        "requires_same_state": True,
        "requires_direct_cpp_oracle": True,
        "vector_parity_required": True,
    },
    "ls_wrapper_gradient": {
        "rtol": 1e-10,
        "atol": 1e-12,
        "requires_same_state": True,
        "requires_direct_cpp_oracle": True,
        "vector_parity_required": True,
    },
    "derivative_heavy": {
        "scalar_value_rtol": 1e-10,
        "scalar_value_atol": 1e-12,
        "first_derivative_rtol": 1e-8,
        "first_derivative_atol": 1e-10,
        "second_derivative_rtol": 1e-6,
        "second_derivative_atol": 1e-8,
        "requires_same_input": True,
        "requires_direct_cpp_oracle": True,
        "fd_validation_secondary": True,
    },
    "reporting_contract": {
        "scalar_value_rtol": 1e-10,
        "scalar_value_atol": 1e-12,
        "distance_rtol": 1e-10,
        "distance_atol": 1e-12,
        "requires_same_state": True,
        "host_materialization_allowed": True,
    },
    "direct_hessian_oracle": {
        "second_derivative_rtol": 1e-8,
        "second_derivative_atol": 1e-10,
        "requires_same_state": True,
        "requires_direct_cpp_oracle": True,
        "full_matrix_required": True,
    },
    "exact_well_conditioned_adjoint": {
        "adjoint_rtol": 1e-6,
        "adjoint_atol": 1e-8,
        "gradient_rtol": 1e-6,
        "gradient_atol": 1e-8,
        "residual_rel_tol": 1e-10,
        "requires_same_state": True,
        "requires_well_conditioned_jacobian": True,
        "vector_parity_required": True,
    },
    "exact_ill_conditioned_adjoint": {
        "adjoint_rtol": None,
        "adjoint_atol": None,
        "gradient_rtol": None,
        "gradient_atol": None,
        "residual_rel_tol": 1e-10,
        "requires_same_state": True,
        "requires_well_conditioned_jacobian": False,
        "operator_failure_allowed": True,
        "vector_parity_required": False,
    },
    "branch_stable_resolve": {
        "core_value_rtol": 1e-6,
        "core_value_atol": 1e-7,
        "derived_value_rtol": 5e-5,
        "derived_value_atol": 1e-7,
        "requires_branch_stable_state": True,
        "branch_divergence_downgrades_to_health_only": True,
    },
    "fd_gradient": {
        "directional_fd_rtol": 1e-5,
        "directional_fd_atol": 1e-7,
        "directional_derivative_floor": 1e-12,
        "central_fd_error_rate": 0.4,
        "central_fd_min_stable_eps": 3,
        "direction_seed": 1729,
        "direction_count": 5,
        "max_direction_rejection_fraction": 0.2,
        "requires_branch_stable_state": True,
        "compares_directional_derivative": True,
    },
    "gpu_runtime": {
        "same_state_forward_rtol": 1e-10,
        "same_state_forward_atol": 1e-12,
        "same_state_gradient_rtol": 1e-8,
        "same_state_gradient_atol": 1e-10,
        "whole_solve_value_rtol": 1e-6,
        "whole_solve_value_atol": 1e-7,
        "requires_x64": True,
        "requires_fixed_seed": True,
        "requires_runtime_metadata": True,
    },
    "reduction_cpu_gpu": {
        "rtol": 1e-12,
        "atol": 1e-12,
        "requires_x64": True,
        "requires_cpu_gpu_devices": True,
        "uses_cancellation_stress": True,
    },
    # Scientific-equivalence ladder lanes per
    # docs/parity_scientific_equivalence_contract_2026-05-09.md §2 + §9.
    # These lanes are reporting-only at Phase 0/1: the parity arbiter does
    # not yet enforce these thresholds, and the existing
    # ``linear_solve_factors`` byte-parity probe and the pre-Newton hard
    # gate ``_pre_newton_census_gate_failures`` in
    # ``benchmarks/single_stage_init_parity.py`` remain authoritative.
    # ``*_condition_estimate_present`` is True because the JAX-native
    # Hager–Higham helper now populates dense compatibility solves.
    # Individual solve results may still emit ``None`` when their dense
    # compatibility operator is intentionally unavailable.
    "ls_solve_quality": {
        "ls_hessian_symmetry_rel_tol": 1e-10,
        "ls_hessian_action_max_rel_tol": 1e-8,
        "ls_newton_linear_residual_rel_tol": 1e-8,
        "ls_newton_step_abs_diff_rel_tol": 1e-8,
        "ls_condition_estimate_present": True,
        "requires_same_state": True,
        "reporting_only": True,
    },
    "exact_solve_quality": {
        "exact_jacobian_action_max_rel_tol": 1e-8,
        "exact_newton_linear_residual_rel_tol": 1e-8,
        "exact_refinement_correction_rel_tol": 1e-9,
        "exact_adjoint_solve_residual_rel_tol": 1e-8,
        "exact_condition_estimate_present": True,
        "requires_same_state": True,
        "reporting_only": True,
    },
}

# ---------------------------------------------------------------------------
# Reporting-only context (NOT a tolerance lane; does NOT gate)
#
# This dict augments gate FAILURE MESSAGES with empirical-baseline severity
# context. The gate's pass/fail decision is unchanged. Per-layer thresholds
# are populated by Slice DM-B once Phase 4 produces the first passing
# strict-gate artifact (corpus is empty as of 2026-05-08).
#
# See docs/parity_dual_mode_contract_2026-05-08.md §2.3 for the design
# rationale and §11 for the threshold-derivation methodology DM-B uses to
# fill in `per_layer`.
PARITY_LADDER_REPORTING_CONTEXT: dict[str, dict[str, object]] = {
    "pre_newton_state_empirical": {
        "threshold_kind": "empirical_per_layer",
        "purpose": "report_severity",  # NOT "gate"
        "source_artifacts": [],  # populated by DM-B from passing artifacts
        "per_layer": {},  # empty skeleton; DM-B populates from corpus
        "requires_byte_identity": False,
    },
}

CI_REPRODUCIBILITY_CONTRACT = {
    "gpu_reduction_order_max_ulp": 10,
    "gpu_reduction_order_rel_tol": 1e-12,
    "gpu_reduction_order_sample_size": 1000,
    "gpu_reproducibility_seed": 1729,
    "gpu_reproducibility_sample_size": 1000,
    "tolerance_ratchet_factor": 10.0,
}

# Initial reduced-fixture ratchet for the grouped-adjoint memory probe.
GROUPED_ADJOINT_MEMORY_BUDGETS = {
    "real_single_stage_init": {
        "cpu": {
            "max_peak_rss_mb": 8192.0,
            "max_peak_gpu_memory_mb": None,
        },
        "cuda": {
            "max_peak_rss_mb": 8192.0,
            "max_peak_gpu_memory_mb": 12288.0,
        },
    }
}

# Stage 2 floors mirror docs/source/jax_acceptance.rst.
TIER5_PERFORMANCE_BUDGETS = {
    "stable_hardware_weekly": {
        "tier2_stage2_e2e": {
            "min_outer_speedup_vs_cpu": 1.25,
            "min_warm_speedup_vs_cpu": 1.25,
            "max_compile_overhead_s": 60.0,
        }
    }
}

SINGLE_STAGE_PROOF_CONTRACTS = {
    TIER3_SINGLE_STAGE_OUTER_LOOP_RUNG: {
        "default_maxiter": 10,
        "min_iterations": 10,
        "require_objective_decrease": True,
        "required_outer_optimizer_method": "lbfgs-ondevice",
        "required_result_keys": (
            "FINAL_IOTA",
            "FINAL_VOLUME",
            "FIELD_ERROR",
            "MAX_CURVATURE",
        ),
    }
}

GPU_PROOF_PARITY_CONTRACTS = {
    "stage2": {
        "value_lane": "tier2_stage2_e2e",
        "value_contract_key": "final_objective_rel_tol",
        "gradient_lane": "tier1_stage2_value_gradient",
        "gradient_contract_key": "gradient_rtol",
    },
    "single_stage": {
        "value_lane": "tier3_single_stage_init",
        "value_contract_key": "field_error_rel_tol",
        "gradient_lane": "gpu_runtime",
        "gradient_contract_key": "same_state_gradient_rtol",
    },
}


def _normalize_contract_key(value: str) -> str:
    return str(value).strip().lower().replace("-", "_")


def _normalize_platform_key(value: str) -> str:
    platform_key = _normalize_contract_key(value)
    if platform_key == "gpu":
        return "cuda"
    return platform_key


def parity_ladder_tolerances(lane: str) -> dict[str, ParityToleranceValue]:
    """Return the precision contract for a named parity-validation lane."""
    lane_key = _normalize_contract_key(lane)
    if lane_key not in PARITY_LADDER_TOLERANCES:
        valid = ", ".join(sorted(PARITY_LADDER_TOLERANCES))
        raise ValueError(
            f"Unknown parity ladder lane {lane!r}. Expected one of: {valid}."
        )
    return dict(PARITY_LADDER_TOLERANCES[lane_key])


def grouped_adjoint_memory_budget(
    *,
    fixture: str,
    platform: str,
) -> dict[str, float | None]:
    fixture_key = _normalize_contract_key(fixture)
    if fixture_key not in GROUPED_ADJOINT_MEMORY_BUDGETS:
        valid = ", ".join(sorted(GROUPED_ADJOINT_MEMORY_BUDGETS))
        raise ValueError(
            f"Unknown grouped-adjoint fixture {fixture!r}. Expected one of: {valid}."
        )
    fixture_budgets = GROUPED_ADJOINT_MEMORY_BUDGETS[fixture_key]
    platform_key = _normalize_platform_key(platform)
    if platform_key == "auto":
        platform_key = "cpu"
    if platform_key not in fixture_budgets:
        valid = ", ".join(sorted(fixture_budgets))
        raise ValueError(
            f"Unknown grouped-adjoint platform {platform!r} for fixture "
            f"{fixture!r}. Expected one of: {valid}."
        )
    return dict(fixture_budgets[platform_key])


def evaluate_grouped_adjoint_memory_budget(
    metrics: dict[str, object],
    budget: dict[str, float | None],
) -> list[str]:
    failures: list[str] = []
    peak_rss_mb = metrics.get("peak_rss_mb")
    max_peak_rss_mb = budget.get("max_peak_rss_mb")
    if (
        max_peak_rss_mb is not None
        and peak_rss_mb is not None
        and float(peak_rss_mb) > float(max_peak_rss_mb)
    ):
        failures.append(
            "Grouped-adjoint memory probe peak RSS "
            f"{float(peak_rss_mb):.2f} MB exceeded checked-in budget "
            f"{float(max_peak_rss_mb):.2f} MB."
        )
    peak_gpu_memory_mb = metrics.get("peak_gpu_memory_mb")
    max_peak_gpu_memory_mb = budget.get("max_peak_gpu_memory_mb")
    if (
        max_peak_gpu_memory_mb is not None
        and peak_gpu_memory_mb is not None
        and float(peak_gpu_memory_mb) > float(max_peak_gpu_memory_mb)
    ):
        failures.append(
            "Grouped-adjoint memory probe peak GPU memory "
            f"{float(peak_gpu_memory_mb):.2f} MB exceeded checked-in budget "
            f"{float(max_peak_gpu_memory_mb):.2f} MB."
        )
    return failures


def tier5_performance_budget(
    *,
    profile: str,
) -> dict[str, dict[str, float | None]]:
    profile_key = _normalize_contract_key(profile)
    if profile_key not in TIER5_PERFORMANCE_BUDGETS:
        valid = ", ".join(sorted(TIER5_PERFORMANCE_BUDGETS))
        raise ValueError(
            f"Unknown Tier 5 performance budget profile {profile!r}. "
            f"Expected one of: {valid}."
        )
    return {
        rung: dict(rung_budget)
        for rung, rung_budget in TIER5_PERFORMANCE_BUDGETS[profile_key].items()
    }


def single_stage_proof_contract(
    rung: str = TIER3_SINGLE_STAGE_OUTER_LOOP_RUNG,
) -> dict[str, object]:
    """Return the documented contract for reduced single-stage proof rungs."""
    if rung not in SINGLE_STAGE_PROOF_CONTRACTS:
        valid = ", ".join(sorted(SINGLE_STAGE_PROOF_CONTRACTS))
        raise ValueError(
            f"Unknown single-stage proof rung {rung!r}. Expected one of: {valid}."
        )
    contract = dict(SINGLE_STAGE_PROOF_CONTRACTS[rung])
    contract["required_result_keys"] = tuple(contract["required_result_keys"])
    return contract


def gpu_proof_parity_contract(
    probe_kind: str,
    *,
    maxiter: int | None = None,
) -> dict[str, float | str]:
    """Return the explicit value/gradient tolerance schema for HF GPU proof."""
    probe_key = _normalize_contract_key(probe_kind)
    if probe_key not in GPU_PROOF_PARITY_CONTRACTS:
        valid = ", ".join(sorted(GPU_PROOF_PARITY_CONTRACTS))
        raise ValueError(
            f"Unknown GPU proof parity probe kind {probe_kind!r}. "
            f"Expected one of: {valid}."
        )

    contract = dict(GPU_PROOF_PARITY_CONTRACTS[probe_key])
    value_lane = str(contract["value_lane"])
    value_contract_key = str(contract["value_contract_key"])
    gradient_lane = str(contract["gradient_lane"])
    gradient_contract_key = str(contract["gradient_contract_key"])

    value_tolerances = optimizer_drift_tolerances(value_lane, maxiter=maxiter)
    if gradient_lane in OPTIMIZER_DRIFT_TOLERANCES:
        gradient_tolerances = optimizer_drift_tolerances(gradient_lane)
    else:
        gradient_tolerances = parity_ladder_tolerances(gradient_lane)

    contract["value_rtol"] = float(value_tolerances[value_contract_key])
    contract["gradient_rtol"] = float(gradient_tolerances[gradient_contract_key])
    return contract


def evaluate_tier5_performance_budget(
    summary_by_name: dict[str, dict[str, object]],
    budget: dict[str, dict[str, float | None]],
) -> list[str]:
    failures: list[str] = []
    for rung_name, rung_budget in budget.items():
        rung_summary = summary_by_name.get(rung_name)
        if rung_summary is None:
            failures.append(
                f"Tier 5 performance budget references missing summary rung {rung_name!r}."
            )
            continue
        min_outer_speedup = rung_budget.get("min_outer_speedup_vs_cpu")
        outer_speedup = rung_summary.get("outer_speedup_vs_cpu")
        if min_outer_speedup is not None:
            if outer_speedup is None or float(outer_speedup) < float(min_outer_speedup):
                failures.append(
                    f"{rung_name} outer first-run wall-clock speedup "
                    f"{'n/a' if outer_speedup is None else f'{float(outer_speedup):.2f}x'} "
                    f"fell below checked-in floor {float(min_outer_speedup):.2f}x."
                )
        min_warm_speedup = rung_budget.get("min_warm_speedup_vs_cpu")
        warm_speedup = rung_summary.get("warm_speedup_vs_cpu")
        if min_warm_speedup is not None:
            if warm_speedup is None or float(warm_speedup) < float(min_warm_speedup):
                failures.append(
                    f"{rung_name} warm steady-state speedup "
                    f"{'n/a' if warm_speedup is None else f'{float(warm_speedup):.2f}x'} "
                    f"fell below checked-in floor {float(min_warm_speedup):.2f}x."
                )
        max_compile_overhead = rung_budget.get("max_compile_overhead_s")
        compile_overhead = rung_summary.get("lane_compile_overhead_s")
        if max_compile_overhead is not None:
            if compile_overhead is None or float(compile_overhead) > float(
                max_compile_overhead
            ):
                failures.append(
                    f"{rung_name} compile overhead "
                    f"{'n/a' if compile_overhead is None else f'{float(compile_overhead):.2f}s'} "
                    f"exceeded checked-in ceiling {float(max_compile_overhead):.2f}s."
                )
    return failures


def resolve_probe_lane(*, optimizer_backend: str | None = None) -> str:
    """Map benchmark/probe options to the intended lane label."""
    if optimizer_backend not in {
        None,
        "scipy",
        "ondevice",
        "scipy-jax",
        "scipy-jax-fullgraph",
    }:
        raise ValueError(
            "optimizer_backend must be one of: scipy, ondevice, scipy-jax, "
            "scipy-jax-fullgraph."
        )
    if optimizer_backend == "ondevice":
        return "private-optimizer"
    if optimizer_backend == "scipy-jax":
        return "target-scipy-control"
    if optimizer_backend == "scipy-jax-fullgraph":
        return "target-scipy-fullgraph-control"
    return "trusted-public-reference"


def short_run_geometry_rel_tolerance(
    maxiter: int,
    explicit_tol: float | None = None,
) -> float | None:
    """Return the end-state geometry gate for Stage 2 ladder runs."""
    if explicit_tol is not None:
        return float(explicit_tol)
    if maxiter <= SHORT_RUN_SMOKE_MAXITER:
        return None
    return 1e-6


def short_run_stage2_final_objective_rel_tolerance(maxiter: int) -> float:
    """Return the Stage 2 endpoint-objective gate for a given iteration budget."""
    if maxiter <= SHORT_RUN_SMOKE_MAXITER:
        return 5e-4
    return 1e-4


def ci_reproducibility_contract() -> dict[str, float | int]:
    """Return the JAX CI reproducibility contract for GPU parity lanes."""
    return dict(CI_REPRODUCIBILITY_CONTRACT)


def ratchet_rel_tol(
    current_rel_tol: float,
    achieved_rel_err: float,
    *,
    factor: float,
) -> float:
    """Tighten a relative tolerance gate to the requested ratchet factor."""
    return min(float(current_rel_tol), float(factor) * float(achieved_rel_err))


def parity_ladder_ratchet_rel_tol(
    lane: str,
    current_rel_tol: float,
    achieved_rel_err: float,
    *,
    branch_divergent: bool = False,
    factor: float | None = None,
) -> float:
    """Return the ratcheted tolerance allowed by a parity-ladder lane.

    Lanes without vector parity, and branch-divergent branch-stable samples,
    keep their current tolerance even if one run reports a smaller error.
    """
    tolerances = parity_ladder_tolerances(lane)
    if branch_divergent or tolerances.get("vector_parity_required") is False:
        return float(current_rel_tol)

    ratchet_factor = (
        CI_REPRODUCIBILITY_CONTRACT["tolerance_ratchet_factor"]
        if factor is None
        else factor
    )
    return ratchet_rel_tol(
        current_rel_tol,
        achieved_rel_err,
        factor=float(ratchet_factor),
    )


def _smoke_geometry_override_error(maxiter: int) -> str:
    return (
        "Explicit --geometry-rel-tol conflicts with the maxiter="
        f"{int(maxiter)} Stage 2 smoke contract; omit the override or use "
        "a longer Stage 2 reproducibility rung."
    )


def stage2_geometry_repro_supported(maxiter: int) -> bool:
    """Return whether the HF harness permits an explicit Stage 2 repro rung."""
    return int(maxiter) > SHORT_RUN_SMOKE_MAXITER


def validate_stage2_hf_plan(
    maxiter: int,
    geometry_rel_tol: float | None,
) -> None:
    """Validate the requested HF Stage 2 rung shape against the ladder contract."""
    if geometry_rel_tol is None:
        return
    if stage2_geometry_repro_supported(maxiter):
        return
    raise ValueError(_smoke_geometry_override_error(maxiter))


def build_stage2_hf_plan(
    maxiter: int,
    geometry_rel_tol: float | None,
) -> dict[str, object]:
    """Return the HF Stage 2 rung plan derived from the ladder contract SSOT."""
    validate_stage2_hf_plan(maxiter, geometry_rel_tol)
    default_geometry_rel_tol = short_run_geometry_rel_tolerance(int(maxiter))
    explicit_geometry_repro = geometry_rel_tol is not None
    stage2_rungs = list(_SMOKE_STAGE2_RUNG_NAMES)
    if explicit_geometry_repro:
        stage2_rungs.append(_GEOMETRY_REPRO_STAGE2_RUNG_NAME)
    effective_geometry_rel_tol = (
        float(geometry_rel_tol) if explicit_geometry_repro else default_geometry_rel_tol
    )
    if explicit_geometry_repro:
        geometry_policy = "explicit-repro-gate"
    elif effective_geometry_rel_tol is None:
        geometry_policy = "report-only"
    else:
        geometry_policy = "default-long-run-gate"
    return {
        "stage2_rungs": tuple(stage2_rungs),
        "explicit_geometry_repro": explicit_geometry_repro,
        "geometry_rel_tol": (
            None if geometry_rel_tol is None else float(geometry_rel_tol)
        ),
        "effective_geometry_rel_tol": effective_geometry_rel_tol,
        "geometry_policy": geometry_policy,
        "smoke_budget": int(maxiter) <= SHORT_RUN_SMOKE_MAXITER,
        "supports_geometry_repro": stage2_geometry_repro_supported(maxiter),
    }


def optimizer_drift_tolerances(
    rung: str,
    *,
    maxiter: int | None = None,
) -> dict[str, float | None]:
    """Return the documented optimizer-replacement tolerances for a ladder rung."""
    if rung not in OPTIMIZER_DRIFT_TOLERANCES:
        valid = ", ".join(sorted(OPTIMIZER_DRIFT_TOLERANCES))
        raise ValueError(
            f"Unknown optimizer-drift rung {rung!r}. Expected one of: {valid}."
        )
    tolerances = dict(OPTIMIZER_DRIFT_TOLERANCES[rung])
    if rung == "tier2_stage2_e2e":
        tolerances.pop("final_objective_rel_tol_20_iter", None)
        tolerances.pop("final_objective_rel_tol_default", None)
        tolerances["final_objective_rel_tol"] = (
            short_run_stage2_final_objective_rel_tolerance(
                21 if maxiter is None else int(maxiter)
            )
        )
        tolerances.pop("geometry_rel_tol_20_iter", None)
        tolerances.pop("geometry_rel_tol_default", None)
        tolerances["geometry_rel_tol"] = short_run_geometry_rel_tolerance(
            21 if maxiter is None else int(maxiter)
        )
    return tolerances
