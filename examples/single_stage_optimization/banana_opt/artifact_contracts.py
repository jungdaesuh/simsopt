from __future__ import annotations

import math
from pathlib import Path

from .current_contracts import (
    BANANA_CURRENT_HARD_LIMIT_A,
    DEFAULT_FINITE_CURRENT_MODE,
    physical_current_to_boozer_I,
    resolve_boozer_current_convention,
    resolve_effective_current_mode,
)
from .hardware_contracts import fixed_stage2_artifact_hardware_contract
from .hardware_constraint_schema import build_bootability_recovery_payload_fields
from workflow_helpers import canonical_stage2_iota_constraint_weight

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


def _upgrade_legacy_stage2_hardware_contract_metadata(upgraded_results: dict) -> None:
    for key, value in fixed_stage2_artifact_hardware_contract().items():
        if upgraded_results.get(key) is None:
            upgraded_results[key] = float(value)


def _upgrade_legacy_bootability_recovery_metadata(upgraded_results: dict) -> None:
    for key, value in build_bootability_recovery_payload_fields(
        None,
        recovery_attempted=False,
        recovery_succeeded=False,
    ).items():
        if upgraded_results.get(key) is None:
            upgraded_results[key] = value


def _upgrade_legacy_stage2_iota_report_metadata(upgraded_results: dict) -> None:
    defaults = {
        "STAGE2_ROOT_FIX_ENABLED": False,
        "STAGE2_IOTA_MODE": "off",
        "STAGE2_IOTA_TARGET": None,
        "STAGE2_IOTA_TOLERANCE": None,
        "STAGE2_IOTA_WEIGHT": 1.0,
        "STAGE2_IOTA_VOL_TARGET": 0.10,
        "STAGE2_IOTA_CONSTRAINT_WEIGHT": 1.0,
        "STAGE2_IOTA_NUM_TF_COILS": 20,
        "STAGE2_IOTA_NPHI": 91,
        "STAGE2_IOTA_NTHETA": 32,
        "STAGE2_IOTA_MPOL": 8,
        "STAGE2_IOTA_NTOR": 6,
        "STAGE2_IOTA_PROBE_SECONDS": None,
        "BOOTABILITY_STAGE2_BS_PATH": None,
        "BOOTABILITY_STAGE2_RESULTS_PATH": None,
        "STAGE2_IOTA_HOT_LOOP_ENABLED": False,
        "STAGE2_IOTA_BOOTSTRAP_SECONDS": None,
        "STAGE2_IOTA_RUNTIME_SECONDS": None,
        "STAGE2_IOTA_RUNTIME_CALLS": None,
        "STAGE2_IOTA_INITIAL": None,
        "STAGE2_IOTA_INITIAL_PENALTY": None,
        "STAGE2_IOTA_FINAL": None,
        "STAGE2_IOTA_FINAL_PENALTY": None,
        "STAGE2_IOTA_PENALTY_THRESHOLD": None,
        "STAGE2_SECONDARY_ARTIFACT_PRESERVED": False,
        "STAGE2_SECONDARY_ARTIFACT_REASON": None,
        "STAGE2_SECONDARY_ARTIFACT_SOURCE": None,
        "STAGE2_SECONDARY_BS_PATH": None,
        "STAGE2_SECONDARY_RESULTS_PATH": None,
    }
    for key, value in defaults.items():
        if upgraded_results.get(key) is None:
            upgraded_results[key] = value
    upgraded_results["STAGE2_IOTA_CONSTRAINT_WEIGHT"] = (
        canonical_stage2_iota_constraint_weight(
            upgraded_results.get("STAGE2_IOTA_CONSTRAINT_WEIGHT")
        )
    )


def _upgrade_legacy_finite_current_metadata(upgraded_results: dict) -> None:
    finite_current_mode = upgraded_results.get("FINITE_CURRENT_MODE")
    if finite_current_mode in {None, ""}:
        finite_current_mode = DEFAULT_FINITE_CURRENT_MODE
        upgraded_results["FINITE_CURRENT_MODE"] = finite_current_mode
    recorded_boozer_current_convention = upgraded_results.get("BOOZER_CURRENT_CONVENTION")
    if recorded_boozer_current_convention in {None, ""}:
        plasma_current_A = upgraded_results.get("PLASMA_CURRENT_A")
        boozer_I = upgraded_results.get("BOOZER_I")
        resolved_boozer_current_convention = resolve_boozer_current_convention(
            finite_current_mode
        )
        if plasma_current_A is not None and boozer_I is not None:
            expected_mu0_boozer_I = physical_current_to_boozer_I(
                plasma_current_A,
                convention="mu0",
            )
            expected_mu0_over_2pi_boozer_I = physical_current_to_boozer_I(
                plasma_current_A,
                convention="mu0_over_2pi",
            )
            matches_mu0 = math.isclose(
                float(boozer_I),
                expected_mu0_boozer_I,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            matches_mu0_over_2pi = math.isclose(
                float(boozer_I),
                expected_mu0_over_2pi_boozer_I,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            if matches_mu0 and not matches_mu0_over_2pi:
                resolved_boozer_current_convention = "mu0"
            elif matches_mu0_over_2pi and not matches_mu0:
                resolved_boozer_current_convention = "mu0_over_2pi"
        upgraded_results["BOOZER_CURRENT_CONVENTION"] = (
            resolved_boozer_current_convention
        )
    else:
        upgraded_results["BOOZER_CURRENT_CONVENTION"] = recorded_boozer_current_convention
    if upgraded_results.get("NUM_PROXY_COILS") is None:
        upgraded_results["NUM_PROXY_COILS"] = 0
    if upgraded_results.get("NUM_VF_COILS") is None:
        upgraded_results["NUM_VF_COILS"] = 0
    if upgraded_results.get("PROXY_PLASMA_CURRENT_A") is None:
        upgraded_results["PROXY_PLASMA_CURRENT_A"] = 0.0
    if upgraded_results.get("VF_CURRENT_A") is None:
        upgraded_results["VF_CURRENT_A"] = 0.0
    if upgraded_results.get("VF_TEMPLATE_PATH") is None:
        upgraded_results["VF_TEMPLATE_PATH"] = None
    if upgraded_results.get("NUM_BANANA_COILS") is None:
        nfp = upgraded_results.get("NFP")
        if nfp is not None:
            upgraded_results["NUM_BANANA_COILS"] = 2 * int(nfp)


def upgrade_legacy_stage2_artifact_results(
    stage2_artifact_results: dict,
    *,
    known_num_tf_coils: int | None = None,
    known_tf_current_A: float | None = None,
) -> dict:
    upgraded_results = dict(stage2_artifact_results)
    if upgraded_results.get("TF_CURRENT_A") is None and known_tf_current_A is not None:
        upgraded_results["TF_CURRENT_A"] = float(known_tf_current_A)
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
    _upgrade_legacy_stage2_hardware_contract_metadata(upgraded_results)
    _upgrade_legacy_bootability_recovery_metadata(upgraded_results)
    _upgrade_legacy_stage2_iota_report_metadata(upgraded_results)
    _upgrade_legacy_finite_current_metadata(upgraded_results)
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
    expected_finite_current_mode: str = DEFAULT_FINITE_CURRENT_MODE,
    expected_boozer_current_convention: str | None = None,
) -> dict:
    resolved_boozer_current_convention = (
        resolve_boozer_current_convention(expected_finite_current_mode)
        if expected_boozer_current_convention is None
        else expected_boozer_current_convention
    )
    expected_effective_mode = resolve_effective_current_mode(
        expected_boozer_I,
        finite_current_mode=expected_finite_current_mode,
    )
    required_keys = (
        "PLASMA_CURRENT_A",
        "PLASMA_CURRENT_INPUT_SOURCE",
        "BOOZER_I",
        "EFFECTIVE_CURRENT_MODE",
        "STAGE2_TF_CURRENT_A",
        "STAGE2_TF_CURRENT_SUM_ABS_A",
        "FINITE_CURRENT_MODE",
        "BOOZER_CURRENT_CONVENTION",
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
        "mode_matches": results.get("FINITE_CURRENT_MODE") == expected_finite_current_mode,
        "boozer_current_convention_matches": results.get("BOOZER_CURRENT_CONVENTION")
        == resolved_boozer_current_convention,
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
        "FINITE_CURRENT_MODE": config.finite_current_mode,
        "BOOZER_CURRENT_CONVENTION": resolve_boozer_current_convention(
            config.finite_current_mode
        ),
        "PROXY_PLASMA_CURRENT_A": config.proxy_plasma_current_A,
        "VF_CURRENT_A": config.vf_current_A,
        "VF_TEMPLATE_PATH": config.vf_template_path,
        "MAJOR_RADIUS": config.major_radius,
        "TOROIDAL_FLUX": config.toroidal_flux,
        "LENGTH_WEIGHT": config.length_weight,
        "CC_WEIGHT": config.cc_weight,
        "CC_THRESHOLD": config.cc_threshold,
        "CURVATURE_WEIGHT": config.curvature_weight,
        "CURVATURE_THRESHOLD": config.curvature_threshold,
        **fixed_stage2_artifact_hardware_contract(),
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
