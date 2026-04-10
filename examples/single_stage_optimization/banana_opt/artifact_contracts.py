from __future__ import annotations

import math
from pathlib import Path

from .current_contracts import BANANA_CURRENT_HARD_LIMIT_A, resolve_effective_current_mode

DEFAULT_LEGACY_BANANA_INIT_CURRENT_A = 1.0e4


def _upgrade_legacy_banana_current_metadata(upgraded_results: dict) -> None:
    banana_current_A = upgraded_results.get("BANANA_CURRENT_A")
    stage2_seed_path = upgraded_results.get("STAGE2_BS_PATH")
    if upgraded_results.get("BANANA_INIT_CURRENT_A") is None:
        if stage2_seed_path in {None, ""}:
            upgraded_results["BANANA_INIT_CURRENT_A"] = (
                DEFAULT_LEGACY_BANANA_INIT_CURRENT_A
            )
        elif upgraded_results.get("init_only") and banana_current_A is not None:
            upgraded_results["BANANA_INIT_CURRENT_A"] = float(banana_current_A)
    if upgraded_results.get("BANANA_CURRENT_MAX_A") is None:
        realized_current_abs_A = (
            0.0 if banana_current_A is None else abs(float(banana_current_A))
        )
        upgraded_results["BANANA_CURRENT_MAX_A"] = max(
            BANANA_CURRENT_HARD_LIMIT_A,
            realized_current_abs_A,
        )


def upgrade_legacy_stage2_artifact_results(
    stage2_artifact_results: dict,
    *,
    known_num_tf_coils: int | None = None,
) -> dict:
    upgraded_results = dict(stage2_artifact_results)
    if upgraded_results.get("NUM_TF_COILS") is None and known_num_tf_coils is not None:
        upgraded_results["NUM_TF_COILS"] = int(known_num_tf_coils)
    if upgraded_results.get("TF_CURRENT_SUM_ABS_A") is None:
        tf_current_A = upgraded_results.get("TF_CURRENT_A")
        num_tf_coils = upgraded_results.get("NUM_TF_COILS")
        if tf_current_A is not None and num_tf_coils is not None:
            upgraded_results["TF_CURRENT_SUM_ABS_A"] = abs(float(tf_current_A)) * float(
                num_tf_coils
            )
    _upgrade_legacy_banana_current_metadata(upgraded_results)
    return upgraded_results


def require_stage2_artifact_float(
    stage2_artifact_results: dict,
    key: str,
    *,
    context_message: str,
) -> float:
    value = stage2_artifact_results.get(key)
    if value is None:
        raise ValueError(
            f"Stage 2 artifact results.json is missing {key}; cannot {context_message}."
        )
    return float(value)


def resolve_expected_stage2_tf_current_A(stage2_artifact_results: dict) -> float:
    return require_stage2_artifact_float(
        stage2_artifact_results,
        "TF_CURRENT_A",
        context_message="validate smoke provenance against the loaded artifact",
    )


def resolve_expected_stage2_tf_current_sum_abs_A(stage2_artifact_results: dict) -> float:
    return require_stage2_artifact_float(
        stage2_artifact_results,
        "TF_CURRENT_SUM_ABS_A",
        context_message="validate smoke provenance against the loaded artifact",
    )


def validate_smoke_results(
    results: dict,
    *,
    requested_current_A: float,
    expected_boozer_I: float,
    expected_stage2_tf_current_A: float,
    expected_stage2_tf_current_sum_abs_A: float,
) -> dict:
    expected_effective_mode = resolve_effective_current_mode(expected_boozer_I)
    required_keys = (
        "PLASMA_CURRENT_A",
        "PLASMA_CURRENT_INPUT_SOURCE",
        "BOOZER_I",
        "EFFECTIVE_CURRENT_MODE",
        "STAGE2_TF_CURRENT_A",
        "STAGE2_TF_CURRENT_SUM_ABS_A",
        "FINITE_CURRENT_MODE",
    )
    missing_keys = [key for key in required_keys if key not in results]
    checks = {
        "missing_keys": missing_keys,
        "plasma_current_matches": math.isclose(
            float(results.get("PLASMA_CURRENT_A", float("nan"))),
            requested_current_A,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "boozer_I_matches": math.isclose(
            float(results.get("BOOZER_I", float("nan"))),
            expected_boozer_I,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "stage2_tf_current_matches": math.isclose(
            float(results.get("STAGE2_TF_CURRENT_A", float("nan"))),
            expected_stage2_tf_current_A,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "stage2_tf_current_sum_abs_matches": math.isclose(
            float(results.get("STAGE2_TF_CURRENT_SUM_ABS_A", float("nan"))),
            expected_stage2_tf_current_sum_abs_A,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "input_source_matches": results.get("PLASMA_CURRENT_INPUT_SOURCE") == "physical_A",
        "mode_matches": results.get("FINITE_CURRENT_MODE") == "boozer_surrogate",
        "effective_mode_matches": results.get("EFFECTIVE_CURRENT_MODE")
        == expected_effective_mode,
    }
    checks["passed"] = not missing_keys and all(
        value for key, value in checks.items() if key not in {"missing_keys", "passed"}
    )
    return checks


def basin_metadata_from_config(config) -> dict:
    return {
        "basin_hops": config.basin_hops,
        "basin_stepsize": None if config.basin_hops == 0 else config.basin_stepsize,
        "basin_temperature": (
            None if config.basin_hops == 0 else config.basin_temperature
        ),
        "basin_niter_success": (
            None
            if config.basin_hops == 0 or config.basin_niter_success <= 0
            else config.basin_niter_success
        ),
        "basin_seed": None if config.basin_hops == 0 else config.basin_seed,
    }


def expected_locked_baseline_stage2_artifact_metadata(
    config,
    *,
    num_tf_coils: int,
) -> dict:
    return {
        "PLASMA_SURF_FILENAME": config.plasma_surf_filename,
        "TF_CURRENT_A": config.tf_current_A,
        "TF_CURRENT_SUM_ABS_A": num_tf_coils * config.tf_current_A,
        "NUM_TF_COILS": num_tf_coils,
        "BANANA_INIT_CURRENT_A": config.banana_init_current_A,
        "BANANA_CURRENT_MAX_A": config.banana_current_max_A,
        "MAJOR_RADIUS": config.major_radius,
        "TOROIDAL_FLUX": config.toroidal_flux,
        "LENGTH_WEIGHT": config.length_weight,
        "CC_WEIGHT": config.cc_weight,
        "CC_THRESHOLD": config.cc_threshold,
        "CURVATURE_WEIGHT": config.curvature_weight,
        "CURVATURE_THRESHOLD": config.curvature_threshold,
        "banana_surf_radius": config.banana_surf_radius,
        "order": config.order,
        "CONSTRAINT_METHOD": config.constraint_method,
        **basin_metadata_from_config(config),
        "init_only": config.init_only,
    }


def validate_stage2_artifact_metadata(
    stage2_results_path: Path,
    stage2_artifact_results: dict,
    *,
    expected_metadata: dict,
    owner_label: str,
    experiment_family: str,
) -> None:
    for key, expected in expected_metadata.items():
        actual = stage2_artifact_results.get(key)
        if actual is None and expected is not None:
            raise ValueError(
                f"Stage 2 artifact results.json is missing {key}; cannot verify the "
                f"locked {experiment_family} identity."
            )
        if isinstance(expected, float):
            if not math.isclose(
                float(actual),
                expected,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"{owner_label} is locked to {key}={expected!r} "
                    f"for the {experiment_family} lane, but {stage2_results_path} "
                    f"reports {actual!r}."
                )
            continue
        if actual != expected:
            raise ValueError(
                f"{owner_label} is locked to {key}={expected!r} "
                f"for the {experiment_family} lane, but {stage2_results_path} "
                f"reports {actual!r}."
            )
