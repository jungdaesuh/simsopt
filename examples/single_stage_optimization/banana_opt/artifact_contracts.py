from __future__ import annotations

import math
from pathlib import Path


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
            upgraded_results["TF_CURRENT_SUM_ABS_A"] = float(tf_current_A) * float(
                num_tf_coils
            )
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
    required_keys = (
        "PLASMA_CURRENT_A",
        "PLASMA_CURRENT_INPUT_SOURCE",
        "BOOZER_I",
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
    }
    checks["passed"] = not missing_keys and all(
        value for key, value in checks.items() if key not in {"missing_keys", "passed"}
    )
    return checks


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
        "basin_hops": config.basin_hops,
        "basin_stepsize": None if config.basin_hops == 0 else config.basin_stepsize,
        "basin_seed": config.basin_seed,
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
