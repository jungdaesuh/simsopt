"""Dependency-light ladder contract helpers shared by launchers and probes."""

from __future__ import annotations

from typing import Mapping, TypedDict

from simsopt_jax.parity_tolerances import (
    PARITY_LADDER_TOLERANCES as PARITY_LADDER_TOLERANCES,
    ParityToleranceValue as ParityToleranceValue,
    normalize_contract_key as _normalize_contract_key,
    parity_ladder_tolerances as parity_ladder_tolerances,
)


SHORT_RUN_SMOKE_MAXITER = 20
TIER3_SINGLE_STAGE_OUTER_LOOP_RUNG = "tier3_single_stage_outer_loop"


class SingleStageProofContract(TypedDict):
    default_maxiter: int
    min_iterations: int
    require_objective_decrease: bool
    required_outer_optimizer_method: str
    required_result_keys: tuple[str, ...]


def _outer_optimizer_backend_choices():
    from simsopt_jax.geo._optimizer_backend_choices import (
        VALID_OUTER_OPTIMIZER_BACKENDS,
        render_invalid_optimizer_backend_message,
    )

    return VALID_OUTER_OPTIMIZER_BACKENDS, render_invalid_optimizer_backend_message


OPTIMIZER_DRIFT_TOLERANCES: dict[str, dict[str, float | None]] = {
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

QUANTITY_TOLERANCE_BUCKETS: dict[str, str] = {
    "field_B": "direct_kernel",
    "field_GradAbsB": "direct_kernel",
    "field_modB": "direct_kernel",
    "surface_gamma": "direct_kernel",
    "surface_unit_normal": "direct_kernel",
    "Bdotn": "direct_kernel",
    "objective_native_subtotal": "ls_wrapper_gradient",
    "SquaredFlux": "ls_wrapper_gradient",
    "SquaredFluxJAX": "ls_wrapper_gradient",
    "gradient": "ls_wrapper_gradient",
    "boozer_residual": "direct_kernel",
    "area": "direct_kernel",
    "volume": "direct_kernel",
    "area_gradient": "derivative_heavy",
    "volume_gradient": "derivative_heavy",
    "qfm_residual": "direct_kernel",
    "qfm_gradient": "derivative_heavy",
    "pm_grid_payload": "direct_kernel",
    "pm_moments": "direct_kernel",
    "pm_residual": "direct_kernel",
    "pm_proxy_residual": "direct_kernel",
    "pm_objective": "direct_kernel",
    "pm_proxy_objective": "direct_kernel",
    "pm_history": "direct_kernel",
    "pm_dipole_field_B": "direct_kernel",
    "pm_proxy_dipole_field_B": "direct_kernel",
    "pm_dipole_Bdotn": "direct_kernel",
    "pm_proxy_dipole_Bdotn": "direct_kernel",
    "wireframe_matrix": "direct_kernel",
    "wireframe_current": "direct_kernel",
    "wireframe_objective": "direct_kernel",
    "wireframe_constraints": "direct_kernel",
    "wireframe_field_B": "direct_kernel",
    "wireframe_field_dB_by_dX": "derivative_heavy",
    "wireframe_Bnormal": "direct_kernel",
    "wireframe_gsco_flags": "direct_kernel",
    "wireframe_gsco_history": "direct_kernel",
    "wireframe_gsco_solution": "direct_kernel",
    "trajectory_endpoint": "event_time_tracing",
    "trajectory_t_final": "event_time_tracing",
    "trajectory_status_code": "direct_kernel",
    "phi_hit_xyz": "event_time_tracing",
    "phi_hit_count": "direct_kernel",
    "toroidal_flux": "direct_kernel",
    "LpCurveForce": "direct_kernel",
    "B2Energy": "direct_kernel",
    "lp_curve_force_gradient": "derivative_heavy",
    "b2_energy_gradient": "derivative_heavy",
    "iota": "direct_kernel",
    "major_radius": "direct_kernel",
    "nq_symmetric_ratio": "direct_kernel",
}
FLOAT32_SMOKE_TOLERANCE_TIER = "float32_smoke"
STRICT_RUNTIME_TOLERANCE_TIERS = frozenset(("cpu_reference", "parity", "fast"))
FLOAT32_SMOKE_OBJECTIVE_QUANTITIES = frozenset(
    {
        "objective_native_subtotal",
        "SquaredFlux",
        "SquaredFluxJAX",
    }
)

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

CI_REPRODUCIBILITY_CONTRACT: dict[str, float | int] = {
    "gpu_reduction_order_max_ulp": 10,
    "gpu_reduction_order_rel_tol": 1e-12,
    "gpu_reduction_order_sample_size": 1000,
    "gpu_reproducibility_seed": 1729,
    "gpu_reproducibility_sample_size": 1000,
    "tolerance_ratchet_factor": 10.0,
}

# Initial reduced-fixture ratchet for the grouped-adjoint memory probe.
GROUPED_ADJOINT_MEMORY_BUDGETS: dict[str, dict[str, dict[str, float | None]]] = {
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

# Stable-hardware Stage 2 performance floors.
TIER5_PERFORMANCE_BUDGETS: dict[str, dict[str, dict[str, float | None]]] = {
    "stable_hardware_weekly": {
        "tier2_stage2_e2e": {
            "min_outer_speedup_vs_cpu": 1.25,
            "min_warm_speedup_vs_cpu": 1.25,
            "max_compile_overhead_s": 60.0,
        }
    }
}

SINGLE_STAGE_PROOF_CONTRACTS: dict[str, SingleStageProofContract] = {
    TIER3_SINGLE_STAGE_OUTER_LOOP_RUNG: {
        "default_maxiter": 10,
        "min_iterations": 10,
        "require_objective_decrease": True,
        "required_outer_optimizer_method": "lbfgs-scipy-jax",
        "required_result_keys": (
            "FINAL_IOTA",
            "FINAL_VOLUME",
            "FIELD_ERROR",
            "MAX_CURVATURE",
        ),
    }
}

GPU_PROOF_PARITY_CONTRACTS: dict[str, dict[str, str]] = {
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


# Schema for compile-timing evidence artifacts (e.g.
# ``.artifacts/lm_minpack_port_plan_2026-05-16/track1_g5_local_cpu_compile_smoke.json``).
# These JSON sidecars are evidence-class, not promotion-class: they document a
# measured first-trace compile time without claiming a parity-ladder lane.
# Each value declares which gate the measurement does and does not certify.
EVIDENCE_ARTIFACT_COMPILE_SCOPES: dict[str, str] = {
    "local_cpu_smoke_not_cuda_gate": (
        "Local CPU cold-compile timing for a target-lane least-squares method. "
        "Records elapsed seconds after ``jax.clear_caches()`` with explicit "
        "result synchronization. Does NOT certify CUDA first-compile "
        "performance or the stable-hardware performance budget."
    ),
}


def evidence_artifact_compile_scope(scope: str) -> str:
    """Return the documented meaning of a compile-timing evidence scope."""
    scope_key = _normalize_contract_key(scope)
    if scope_key not in EVIDENCE_ARTIFACT_COMPILE_SCOPES:
        valid = ", ".join(sorted(EVIDENCE_ARTIFACT_COMPILE_SCOPES))
        raise ValueError(
            f"Unknown compile-scope marker {scope!r}. Expected one of: {valid}."
        )
    return EVIDENCE_ARTIFACT_COMPILE_SCOPES[scope_key]


def _normalize_platform_key(value: str) -> str:
    platform_key = _normalize_contract_key(value)
    if platform_key == "gpu":
        return "cuda"
    return platform_key


def _float_contract_value(values: Mapping[str, object], key: str) -> float:
    value = values[key]
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise TypeError(f"Contract key {key!r} must hold a numeric value.")
    return float(value)


def _optional_float_contract_value(
    values: Mapping[str, object],
    key: str,
) -> float | None:
    value = values.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise TypeError(f"Contract key {key!r} must hold a numeric value or None.")
    return float(value)


def quantity_uses_gradient_tolerance(quantity: str) -> bool:
    """Return whether a quantity selects the float32 gradient tolerance lane."""
    return "gradient" in quantity.lower()


def _quantity_uses_float32_objective_tolerance(quantity: str, bucket: str) -> bool:
    quantity_lower = quantity.lower()
    return (
        quantity in FLOAT32_SMOKE_OBJECTIVE_QUANTITIES
        or "objective" in quantity_lower
        or (
            bucket == "ls_wrapper_gradient"
            and not quantity_uses_gradient_tolerance(quantity)
        )
    )


def quantity_parity_tolerance(
    quantity: str,
    *,
    runtime_tier: str,
) -> tuple[str, float, float]:
    """Return ``(bucket, rtol, atol)`` for a named parity quantity.

    The mapping preserves the generic parity-harness contract: strict runtime
    tiers route through quantity buckets, while the float32 smoke tier selects
    value/objective/gradient tolerances from its runtime lane.
    """
    bucket = QUANTITY_TOLERANCE_BUCKETS.get(quantity, "direct_kernel")
    runtime_tier_key = _normalize_contract_key(runtime_tier)
    if runtime_tier_key == FLOAT32_SMOKE_TOLERANCE_TIER:
        tolerances = parity_ladder_tolerances(runtime_tier_key)
        if quantity_uses_gradient_tolerance(quantity):
            return (
                runtime_tier_key,
                _float_contract_value(tolerances, "gradient_rtol"),
                _float_contract_value(tolerances, "gradient_atol"),
            )
        if _quantity_uses_float32_objective_tolerance(quantity, bucket):
            return (
                runtime_tier_key,
                _float_contract_value(tolerances, "objective_rtol"),
                _float_contract_value(tolerances, "objective_atol"),
            )
        return (
            runtime_tier_key,
            _float_contract_value(tolerances, "rtol"),
            _float_contract_value(tolerances, "atol"),
        )
    if runtime_tier_key not in STRICT_RUNTIME_TOLERANCE_TIERS:
        raise RuntimeError(
            f"Unsupported runtime tolerance tier {runtime_tier!r} for "
            "generic example parity harness."
        )

    tolerances = parity_ladder_tolerances(bucket)
    if bucket == "event_time_tracing":
        return (
            bucket,
            _float_contract_value(tolerances, "state_vector_rtol"),
            _float_contract_value(tolerances, "state_vector_atol"),
        )
    if "first_derivative_rtol" in tolerances and "rtol" not in tolerances:
        return (
            bucket,
            _float_contract_value(tolerances, "first_derivative_rtol"),
            _float_contract_value(tolerances, "first_derivative_atol"),
        )
    return (
        bucket,
        _float_contract_value(tolerances, "rtol"),
        _float_contract_value(tolerances, "atol"),
    )


def comparison_failure_gates_verdict(entry: Mapping[str, object]) -> bool:
    """Return whether a failed comparison should fail the fixture verdict."""
    return entry.get("verdict") == "fail" and entry.get("diagnostic_only") is not True


def comparison_failure_is_diagnostic(entry: Mapping[str, object]) -> bool:
    """Return whether a failed comparison is recorded as diagnostic-only."""
    return entry.get("verdict") == "fail" and entry.get("diagnostic_only") is True


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
    peak_rss_mb = _optional_float_contract_value(metrics, "peak_rss_mb")
    max_peak_rss_mb = budget.get("max_peak_rss_mb")
    if (
        max_peak_rss_mb is not None
        and peak_rss_mb is not None
        and peak_rss_mb > max_peak_rss_mb
    ):
        failures.append(
            "Grouped-adjoint memory probe peak RSS "
            f"{peak_rss_mb:.2f} MB exceeded checked-in budget "
            f"{max_peak_rss_mb:.2f} MB."
        )
    peak_gpu_memory_mb = _optional_float_contract_value(metrics, "peak_gpu_memory_mb")
    max_peak_gpu_memory_mb = budget.get("max_peak_gpu_memory_mb")
    if (
        max_peak_gpu_memory_mb is not None
        and peak_gpu_memory_mb is not None
        and peak_gpu_memory_mb > max_peak_gpu_memory_mb
    ):
        failures.append(
            "Grouped-adjoint memory probe peak GPU memory "
            f"{peak_gpu_memory_mb:.2f} MB exceeded checked-in budget "
            f"{max_peak_gpu_memory_mb:.2f} MB."
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
    contract = SINGLE_STAGE_PROOF_CONTRACTS[rung]
    return {
        "default_maxiter": contract["default_maxiter"],
        "min_iterations": contract["min_iterations"],
        "require_objective_decrease": contract["require_objective_decrease"],
        "required_outer_optimizer_method": contract["required_outer_optimizer_method"],
        "required_result_keys": tuple(contract["required_result_keys"]),
    }


def gpu_proof_parity_contract(
    probe_kind: str,
    *,
    maxiter: int | None = None,
) -> dict[str, float | str]:
    """Return the explicit value/gradient tolerance schema for GPU proof."""
    probe_key = _normalize_contract_key(probe_kind)
    if probe_key not in GPU_PROOF_PARITY_CONTRACTS:
        valid = ", ".join(sorted(GPU_PROOF_PARITY_CONTRACTS))
        raise ValueError(
            f"Unknown GPU proof parity probe kind {probe_kind!r}. "
            f"Expected one of: {valid}."
        )

    contract: dict[str, float | str] = {
        key: value for key, value in GPU_PROOF_PARITY_CONTRACTS[probe_key].items()
    }
    value_lane = str(contract["value_lane"])
    value_contract_key = str(contract["value_contract_key"])
    gradient_lane = str(contract["gradient_lane"])
    gradient_contract_key = str(contract["gradient_contract_key"])

    value_tolerances = optimizer_drift_tolerances(value_lane, maxiter=maxiter)
    if gradient_lane in OPTIMIZER_DRIFT_TOLERANCES:
        gradient_tolerances = optimizer_drift_tolerances(gradient_lane)
    else:
        gradient_tolerances = parity_ladder_tolerances(gradient_lane)

    contract["value_rtol"] = _float_contract_value(
        value_tolerances,
        value_contract_key,
    )
    contract["gradient_rtol"] = _float_contract_value(
        gradient_tolerances,
        gradient_contract_key,
    )
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
        outer_speedup = _optional_float_contract_value(
            rung_summary,
            "outer_speedup_vs_cpu",
        )
        if min_outer_speedup is not None:
            if outer_speedup is None or outer_speedup < min_outer_speedup:
                failures.append(
                    f"{rung_name} outer first-run wall-clock speedup "
                    f"{'n/a' if outer_speedup is None else f'{outer_speedup:.2f}x'} "
                    f"fell below checked-in floor {min_outer_speedup:.2f}x."
                )
        min_warm_speedup = rung_budget.get("min_warm_speedup_vs_cpu")
        warm_speedup = _optional_float_contract_value(
            rung_summary,
            "warm_speedup_vs_cpu",
        )
        if min_warm_speedup is not None:
            if warm_speedup is None or warm_speedup < min_warm_speedup:
                failures.append(
                    f"{rung_name} warm steady-state speedup "
                    f"{'n/a' if warm_speedup is None else f'{warm_speedup:.2f}x'} "
                    f"fell below checked-in floor {min_warm_speedup:.2f}x."
                )
        max_compile_overhead = rung_budget.get("max_compile_overhead_s")
        compile_overhead = _optional_float_contract_value(
            rung_summary,
            "lane_compile_overhead_s",
        )
        if max_compile_overhead is not None:
            if compile_overhead is None or compile_overhead > max_compile_overhead:
                failures.append(
                    f"{rung_name} compile overhead "
                    f"{'n/a' if compile_overhead is None else f'{compile_overhead:.2f}s'} "
                    f"exceeded checked-in ceiling {max_compile_overhead:.2f}s."
                )
    return failures


def resolve_probe_lane(*, optimizer_backend: str | None = None) -> str:
    """Map benchmark/probe options to the intended lane label."""
    (
        valid_outer_optimizer_backends,
        render_invalid_optimizer_backend_message,
    ) = _outer_optimizer_backend_choices()
    if (
        optimizer_backend is not None
        and optimizer_backend not in valid_outer_optimizer_backends
    ):
        raise ValueError(render_invalid_optimizer_backend_message("outer"))
    if optimizer_backend == "ondevice":
        return "private-optimizer"
    if optimizer_backend == "scipy-jax":
        return "target-scipy-control"
    if optimizer_backend == "host-jax":
        return "host-jax-kernelized-control"
    if optimizer_backend == "scipy-jax-fullgraph":
        return "target-scipy-fullgraph-control"
    if optimizer_backend == "optax-lbfgs":
        return "target-optax-lbfgs"
    if optimizer_backend == "optimistix-lbfgs":
        return "target-optimistix-lbfgs"
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
