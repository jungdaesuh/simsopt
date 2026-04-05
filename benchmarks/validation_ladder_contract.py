"""Dependency-light ladder contract helpers shared by launchers and probes."""

from __future__ import annotations


SHORT_RUN_SMOKE_MAXITER = 20
_SMOKE_STAGE2_RUNG_NAMES = ("stage2_cold", "stage2_warm")
_GEOMETRY_REPRO_STAGE2_RUNG_NAME = "stage2_warm_repro"

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
        "final_iota_abs_tol": 1e-3,
        "final_volume_rel_tol": 1e-6,
        "field_error_rel_tol": 1e-4,
        "surface_geometry_rel_tol": 1e-5,
    },
    "tier4_adjoint_fd": {
        "adjoint_residual_rel_tol": 1e-10,
        "recomposed_total_rel_tol": 1e-12,
        "fixed_surface_fd_rel_tol": 1e-3,
        "fixed_surface_fd_abs_tol": 1e-8,
        "full_resolve_fd_rel_tol": 1e-2,
        "full_resolve_fd_abs_tol": 1e-8,
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


def _normalize_contract_key(value: str) -> str:
    return str(value).strip().lower().replace("-", "_")


def _normalize_platform_key(value: str) -> str:
    platform_key = _normalize_contract_key(value)
    if platform_key == "gpu":
        return "cuda"
    return platform_key


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
                    f"{rung_name} cold end-to-end speedup "
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
            if (
                compile_overhead is None
                or float(compile_overhead) > float(max_compile_overhead)
            ):
                failures.append(
                    f"{rung_name} compile overhead "
                    f"{'n/a' if compile_overhead is None else f'{float(compile_overhead):.2f}s'} "
                    f"exceeded checked-in ceiling {float(max_compile_overhead):.2f}s."
                )
    return failures


def resolve_probe_lane(*, optimizer_backend: str | None = None) -> str:
    """Map benchmark/probe options to the intended lane label."""
    if optimizer_backend in {"hybrid", "ondevice"}:
        return "private-optimizer"
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
        float(geometry_rel_tol)
        if explicit_geometry_repro
        else default_geometry_rel_tol
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
