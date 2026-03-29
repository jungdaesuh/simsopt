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
