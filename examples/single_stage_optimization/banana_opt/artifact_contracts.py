from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from .current_contracts import (
    BANANA_CURRENT_HARD_LIMIT_A,
    DEFAULT_FINITE_CURRENT_MODE,
    FINITE_CURRENT_MODE_SOURCE_ARTIFACT_METADATA,
    FINITE_CURRENT_MODE_SOURCE_LEGACY_ASSUMED_DEFAULT,
    physical_current_to_boozer_I,
    resolve_boozer_current_convention,
    resolve_effective_current_mode,
)
from .finite_current_profiles import (
    FINITE_CURRENT_PROFILES,
    JHALPERN30_FINITE_CURRENT_MODE,
)
from .hardware_contracts import (
    fixed_stage2_artifact_hardware_contract,
    stage2_artifact_hardware_contract,
)
from .hardware_constraint_schema import build_bootability_recovery_payload_fields
from .constraint_contract import CONSTRAINT_SCHEMA_VERSION
from .wout_convention import wout_convention_artifact_fields
from workflow_helpers import canonical_stage2_iota_constraint_weight


DEFAULT_LEGACY_BANANA_INIT_CURRENT_A = -1.0e4
_BOOZER_CURRENT_CONVENTION_INFERENCE_ABS_TOL = 1.0e-12
STAGE2_BS_SHA256_KEY = "STAGE2_BS_SHA256"
LEGACY_CONSTRAINT_CONTRACT_SCHEMA_VERSION = 0
CURRENT_CONSTRAINT_CONTRACT_SCHEMA_VERSION = CONSTRAINT_SCHEMA_VERSION
STAGE2_SEED_CONTRACT_HASH_KEY = "STAGE2_SEED_CONTRACT_HASH"


def compute_stage2_bs_sha256(stage2_bs_path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(stage2_bs_path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        "STAGE2_IOTA_TARGET": None,
        "STAGE2_IOTA_TOLERANCE": None,
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
        "IOTA_NEAR_TARGET": None,
        "STAGE2_IOTA_OBJECTIVE_COUPLED": False,
        "STAGE2_IOTA_HOT_LOOP_ENABLED": False,
        "STAGE2_IOTA_BOOTSTRAP_SECONDS": None,
        "STAGE2_IOTA_RUNTIME_SECONDS": None,
        "STAGE2_IOTA_RUNTIME_CALLS": None,
        "STAGE2_IOTA_INITIAL": None,
        "STAGE2_IOTA_INITIAL_PENALTY": None,
        "STAGE2_IOTA_FINAL": None,
        "STAGE2_IOTA_FINAL_PENALTY": None,
        "STAGE2_IOTA_FINAL_ABS_ERROR": None,
        "STAGE2_IOTA_FINAL_FEASIBLE": None,
        "STAGE2_IOTA_FINAL_SOLVE_FAILED": None,
        "STAGE2_IOTA_VALUE": None,
        "STAGE2_IOTA_PENALTY": None,
        "STAGE2_IOTA_ABS_ERROR": None,
        "STAGE2_IOTA_FEASIBLE": None,
        "STAGE2_IOTA_PENALTY_THRESHOLD": None,
        "STAGE2_SECONDARY_ARTIFACT_PRESERVED": False,
        "STAGE2_SECONDARY_ARTIFACT_REASON": None,
        "STAGE2_SECONDARY_ARTIFACT_SOURCE": None,
        "STAGE2_SECONDARY_BS_PATH": None,
        "STAGE2_SECONDARY_RESULTS_PATH": None,
        # Phase 1 Boozer residual trust gate keys. Legacy artifacts predate
        # the trust contract; default to ``None`` so downstream consumers can
        # detect the absence rather than read a fabricated value.
        "BOOZER_SOLVE_SUCCESS": None,
        "BOOZER_SELF_INTERSECTING": None,
        "BOOZER_CONSTRAINED_RESIDUAL_NORM": None,
        "BOOZER_TRUSTED": None,
        "IOTA_OBJECTIVE_ACTIVE": None,
        "BOOZER_TRUST_REASON": None,
        "BOOZER_TRUST_TOL": None,
        # Phase 3 non-Boozer topology bridge diagnostics. Always emitted with
        # ``None`` placeholders when not computed so the artifact schema is
        # identical across lanes.
        "FIELDLINE_IOTA_PROXY": None,
        "FIELDLINE_IOTA_PROXY_VALID": None,
        "HELICAL_FIELD_CONTENT": None,
        "S_HEL_OBJECTIVE_WEIGHT": None,
        "PRE_BOOZER_TOPOLOGY_SCORE": None,
    }
    for key, value in defaults.items():
        if upgraded_results.get(key) is None:
            upgraded_results[key] = value
    upgraded_results["STAGE2_IOTA_CONSTRAINT_WEIGHT"] = (
        canonical_stage2_iota_constraint_weight(
            upgraded_results.get("STAGE2_IOTA_CONSTRAINT_WEIGHT")
        )
    )


def _backfill_unprefixed_count_from_stage2(
    upgraded_results: dict,
    *,
    count_key: str,
) -> None:
    if upgraded_results.get(count_key) is not None:
        return
    stage2_count = upgraded_results.get(f"STAGE2_{count_key}")
    if stage2_count is not None:
        upgraded_results[count_key] = int(stage2_count)


def _infer_legacy_boozer_current_convention(
    *,
    finite_current_mode: str,
    plasma_current_A,
    boozer_I,
) -> str:
    resolved_boozer_current_convention = resolve_boozer_current_convention(
        finite_current_mode
    )
    if plasma_current_A is None or boozer_I is None:
        return resolved_boozer_current_convention

    expected_boozer_I_by_convention = {
        "mu0": physical_current_to_boozer_I(plasma_current_A, convention="mu0"),
        "mu0_over_2pi": physical_current_to_boozer_I(
            plasma_current_A,
            convention="mu0_over_2pi",
        ),
    }
    matching_conventions = {
        convention
        for convention, expected_boozer_I in expected_boozer_I_by_convention.items()
        if math.isclose(
            float(boozer_I),
            expected_boozer_I,
            rel_tol=0.0,
            abs_tol=_BOOZER_CURRENT_CONVENTION_INFERENCE_ABS_TOL,
        )
    }
    if len(matching_conventions) != 1:
        return resolved_boozer_current_convention
    return next(iter(matching_conventions))


def _upgrade_legacy_finite_current_metadata(upgraded_results: dict) -> None:
    finite_current_mode = upgraded_results.get("FINITE_CURRENT_MODE")
    finite_current_mode_source = upgraded_results.get("FINITE_CURRENT_MODE_SOURCE")
    if finite_current_mode in {None, ""}:
        finite_current_mode = DEFAULT_FINITE_CURRENT_MODE
        upgraded_results["FINITE_CURRENT_MODE"] = finite_current_mode
        if finite_current_mode_source in {None, ""}:
            upgraded_results["FINITE_CURRENT_MODE_SOURCE"] = (
                FINITE_CURRENT_MODE_SOURCE_LEGACY_ASSUMED_DEFAULT
            )
    elif finite_current_mode_source in {None, ""}:
        upgraded_results["FINITE_CURRENT_MODE_SOURCE"] = (
            FINITE_CURRENT_MODE_SOURCE_ARTIFACT_METADATA
        )
    is_jhalpern30_mode = finite_current_mode == JHALPERN30_FINITE_CURRENT_MODE
    recorded_boozer_current_convention = upgraded_results.get("BOOZER_CURRENT_CONVENTION")
    if recorded_boozer_current_convention in {None, ""}:
        if not is_jhalpern30_mode:
            upgraded_results["BOOZER_CURRENT_CONVENTION"] = (
                _infer_legacy_boozer_current_convention(
                    finite_current_mode=finite_current_mode,
                    plasma_current_A=upgraded_results.get("PLASMA_CURRENT_A"),
                    boozer_I=upgraded_results.get("BOOZER_I"),
                )
            )
    else:
        upgraded_results["BOOZER_CURRENT_CONVENTION"] = recorded_boozer_current_convention
    for count_key in (
        "NUM_BANANA_COILS",
        "NUM_PROXY_COILS",
        "NUM_VF_COILS",
    ):
        _backfill_unprefixed_count_from_stage2(
            upgraded_results,
            count_key=count_key,
        )
    if upgraded_results.get("NUM_PROXY_COILS") is None and not is_jhalpern30_mode:
        upgraded_results["NUM_PROXY_COILS"] = 0
    if upgraded_results.get("NUM_VF_COILS") is None and not is_jhalpern30_mode:
        upgraded_results["NUM_VF_COILS"] = 0
    if (
        upgraded_results.get("PROXY_PLASMA_CURRENT_A") is None
        and not is_jhalpern30_mode
    ):
        upgraded_results["PROXY_PLASMA_CURRENT_A"] = 0.0
    if upgraded_results.get("VF_CURRENT_A") is None and not is_jhalpern30_mode:
        upgraded_results["VF_CURRENT_A"] = 0.0
    if upgraded_results.get("VF_TEMPLATE_PATH") is None and not is_jhalpern30_mode:
        upgraded_results["VF_TEMPLATE_PATH"] = None

    profile = FINITE_CURRENT_PROFILES.get(finite_current_mode)
    if profile is not None and not is_jhalpern30_mode:
        policy_defaults = {
            "G0_POLICY": profile.g0_policy,
            "PROXY_PLACEMENT_MODE": profile.proxy_placement_policy,
            "PROXY_VF_CURRENT_SCALAR_POLICY": profile.proxy_vf_current_scalar_policy,
            "VF_TEMPLATE_SHA256": profile.vf_template_sha256,
            "VF_CURRENT_SIGN_POLICY": profile.vf_current_sign_policy,
            "VF_CURRENT_MUTABILITY": profile.vf_current_mutability,
        }
        for key, value in policy_defaults.items():
            if upgraded_results.get(key) is None:
                upgraded_results[key] = value
    if upgraded_results.get("VF_CURRENT_MAX_A") is None:
        upgraded_results["VF_CURRENT_MAX_A"] = BANANA_CURRENT_HARD_LIMIT_A
    if upgraded_results.get("NUM_BANANA_COILS") is None and not is_jhalpern30_mode:
        nfp = upgraded_results.get("NFP")
        if nfp is not None:
            upgraded_results["NUM_BANANA_COILS"] = 2 * int(nfp)


def _upgrade_legacy_wout_convention_metadata(upgraded_results: dict) -> None:
    """Backfill WOUT_CONVENTION/WOUT_OFF_SPEC for legacy artifacts.

    Backfill — producers now stamp at write time per task #40 (2026-05-21);
    this remains for legacy artifacts and round-trip safety.

    Stage 2 artifacts emitted by ``banana_coil_solver`` stamp both keys, but
    single-stage outputs (regular ``results.json`` and preserved-timeout
    sidecars such as ``results_best_feasible.partial.json``) historically did
    not. The Stage 2 seed-contract validator now requires both keys, so any
    seed re-promoted from a single-stage artifact needs them backfilled here
    rather than re-stamped at each producer site.

    Backfill is atomic — both keys are computed together from
    ``PLASMA_SURF_PATH`` and ``TF_CURRENT_A`` already present in the dict.
    If either input is absent or the WOUT file is not yet on disk, no fields
    are added; the downstream validator surfaces the precise gap.
    Already-recorded values are preserved so the upgrader cannot mask a
    drifted producer-side stamp.
    """
    if "WOUT_CONVENTION" in upgraded_results or "WOUT_OFF_SPEC" in upgraded_results:
        return
    plasma_surf_path = upgraded_results.get("PLASMA_SURF_PATH")
    tf_current_A = upgraded_results.get("TF_CURRENT_A")
    if plasma_surf_path is None or tf_current_A is None:
        return
    if not Path(str(plasma_surf_path)).is_file():
        return
    upgraded_results.update(
        wout_convention_artifact_fields(
            wout_path=str(plasma_surf_path),
            tf_current_A=float(tf_current_A),
        )
    )


def upgrade_legacy_stage2_artifact_results(
    stage2_artifact_results: dict,
    *,
    known_num_tf_coils: int | None = None,
    known_tf_current_A: float | None = None,
) -> dict:
    upgraded_results = dict(stage2_artifact_results)
    _backfill_unprefixed_count_from_stage2(
        upgraded_results,
        count_key="NUM_TF_COILS",
    )
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
    _upgrade_legacy_wout_convention_metadata(upgraded_results)
    _upgrade_legacy_constraint_contract_metadata(upgraded_results)
    return upgraded_results


def _upgrade_legacy_constraint_contract_metadata(upgraded_results: dict) -> None:
    if "CONTRACT_SCHEMA_VERSION" not in upgraded_results:
        upgraded_results["CONTRACT_SCHEMA_VERSION"] = (
            LEGACY_CONSTRAINT_CONTRACT_SCHEMA_VERSION
        )
    if "CONSTRAINT_PROFILE" not in upgraded_results:
        upgraded_results["CONSTRAINT_PROFILE"] = None
    if "EFFECTIVE_VALUES" not in upgraded_results:
        upgraded_results["EFFECTIVE_VALUES"] = None
    if "OVERRIDE_REASON" not in upgraded_results:
        upgraded_results["OVERRIDE_REASON"] = None


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


def _iter_json_objects(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _iter_json_objects(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_json_objects(child)


def validate_vacuum_boozer_surface_payload(payload: dict) -> dict:
    objects = list(_iter_json_objects(payload))
    plain_boozer_surfaces = [
        item
        for item in objects
        if item.get("@module") == "simsopt.geo.boozersurface"
        and item.get("@class") == "BoozerSurface"
    ]
    finite_i_boozer_surfaces = [
        item
        for item in objects
        if item.get("@class") == "BoozerSurfaceFiniteI"
        or item.get("@module") == "banana_opt.boozer_finite_current"
    ]
    i_fields = [item["I"] for item in objects if "I" in item]
    checks = {
        "has_plain_boozer_surface": bool(plain_boozer_surfaces),
        "no_finite_i_boozer_surface": not finite_i_boozer_surfaces,
        "no_i_field": not i_fields,
        "finite_i_boozer_surface_count": len(finite_i_boozer_surfaces),
        "i_field_count": len(i_fields),
    }
    checks["passed"] = (
        checks["has_plain_boozer_surface"]
        and checks["no_finite_i_boozer_surface"]
        and checks["no_i_field"]
    )
    return checks


def validate_vacuum_boozer_surface_json(surface_path: str | Path) -> dict:
    payload = json.loads(Path(surface_path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Boozer surface artifact must be a JSON object: {surface_path}")
    return validate_vacuum_boozer_surface_payload(payload)


def validate_boozer_surface_json_current_lineage(surface_path: str | Path) -> dict:
    payload = json.loads(Path(surface_path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Boozer surface artifact must be a JSON object: {surface_path}")
    i_values = [
        float(item["I"])
        for item in _iter_json_objects(payload)
        if "I" in item
    ]
    if not i_values:
        return validate_vacuum_boozer_surface_payload(payload)
    vacuum_i_values = [
        value
        for value in i_values
        if resolve_effective_current_mode(value) == "vacuum"
    ]
    if vacuum_i_values:
        return validate_vacuum_boozer_surface_payload(payload)
    return {
        "passed": True,
        "vacuum_i_field_count": 0,
        "finite_i_field_count": len(i_values),
    }


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
        "TF_CURRENT_SUM_ABS_A": num_tf_coils * abs(config.tf_current_A),
        "NUM_TF_COILS": num_tf_coils,
        "BANANA_INIT_CURRENT_A": config.banana_init_current_A,
        "BANANA_CURRENT_MAX_A": config.banana_current_max_A,
        "FINITE_CURRENT_MODE": config.finite_current_mode,
        "BOOZER_CURRENT_CONVENTION": resolve_boozer_current_convention(
            config.finite_current_mode
        ),
        "PROXY_PLASMA_CURRENT_A": config.proxy_plasma_current_A,
        "VF_CURRENT_A": config.vf_current_A,
        "VF_TEMPLATE_PATH": config.effective_vf_template_path,
        "MAJOR_RADIUS": config.major_radius,
        "TOROIDAL_FLUX": config.toroidal_flux,
        "LENGTH_WEIGHT": config.length_weight,
        "CC_WEIGHT": config.cc_weight,
        "CC_THRESHOLD": config.cc_threshold,
        "CURVATURE_WEIGHT": config.curvature_weight,
        "CURVATURE_THRESHOLD": config.curvature_threshold,
        **stage2_artifact_hardware_contract(config.length_target),
        "banana_surf_radius": config.banana_surf_radius,
        "order": config.order,
        "CONSTRAINT_METHOD": config.constraint_method,
        **basin_metadata_from_config(config),
        "init_only": config.init_only,
    }


def _allows_missing_legacy_zero_vf_template_path(
    *,
    metadata_key: str,
    expected_value,
    actual_value,
    stage2_artifact_results: dict,
) -> bool:
    return (
        metadata_key == "VF_TEMPLATE_PATH"
        and actual_value is None
        and expected_value is not None
        and float(stage2_artifact_results.get("VF_CURRENT_A", 0.0)) == 0.0
        and int(stage2_artifact_results.get("NUM_VF_COILS", 0)) == 0
    )


def validate_constraint_contract_schema_version(
    stage2_results_path: Path,
    stage2_artifact_results: dict,
    *,
    owner_label: str,
) -> None:
    """Enforce forward compatibility of the persisted constraint contract.

    * Absent / legacy schema (``CONTRACT_SCHEMA_VERSION=0``) is rejected
      because the artifact cannot be bound to the current constraint contract.
    * The current schema version is accepted silently.
    * Any other value — typically a future version produced by newer
      code — is rejected so we never silently misread an artifact whose
      schema we do not understand.
    """
    raw_version = stage2_artifact_results.get("CONTRACT_SCHEMA_VERSION")
    if raw_version is None:
        recorded_version = LEGACY_CONSTRAINT_CONTRACT_SCHEMA_VERSION
    else:
        recorded_version = int(raw_version)
    if recorded_version == CURRENT_CONSTRAINT_CONTRACT_SCHEMA_VERSION:
        return
    if recorded_version == LEGACY_CONSTRAINT_CONTRACT_SCHEMA_VERSION:
        raise ValueError(
            f"{owner_label}: {stage2_results_path} uses legacy constraint "
            f"contract schema v{LEGACY_CONSTRAINT_CONTRACT_SCHEMA_VERSION} "
            "(no CONTRACT_HASH); cannot verify constraint-contract binding."
        )
    raise ValueError(
        f"{owner_label}: {stage2_results_path} reports CONTRACT_SCHEMA_VERSION="
        f"{recorded_version!r}, which is incompatible with the current schema "
        f"v{CURRENT_CONSTRAINT_CONTRACT_SCHEMA_VERSION}. Upgrade the reader "
        "before loading this artifact."
    )


def validate_stage2_artifact_metadata(
    stage2_results_path: Path,
    stage2_artifact_results: dict,
    *,
    expected_metadata: dict,
    owner_label: str,
    experiment_family: str,
) -> None:
    validate_constraint_contract_schema_version(
        stage2_results_path,
        stage2_artifact_results,
        owner_label=owner_label,
    )
    for key, expected in expected_metadata.items():
        actual = stage2_artifact_results.get(key)
        if _allows_missing_legacy_zero_vf_template_path(
            metadata_key=key,
            expected_value=expected,
            actual_value=actual,
            stage2_artifact_results=stage2_artifact_results,
        ):
            continue
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
