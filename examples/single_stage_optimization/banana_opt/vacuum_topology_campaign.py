from __future__ import annotations

import hashlib
import json
import shlex
from pathlib import Path
from typing import Mapping, Sequence

STRICT_VACUUM_CURRENT_LINEAGE = "strict_vacuum"
STRICT_VACUUM_SEED_LINEAGE_RECENT_STAGE1_CANDIDATE = "recent_stage1_candidate"
STRICT_VACUUM_SEED_LINEAGE_LEGACY_CONTROL = "legacy_control"
STRICT_VACUUM_SEED_LINEAGES = (
    STRICT_VACUUM_SEED_LINEAGE_RECENT_STAGE1_CANDIDATE,
    STRICT_VACUUM_SEED_LINEAGE_LEGACY_CONTROL,
)
STRICT_VACUUM_BOOZER_SURFACE_CLASS = "BoozerSurface"
STRICT_VACUUM_BOOZER_SURFACE_MODULE = "simsopt.geo.boozersurface"
STRICT_VACUUM_BOOZER_INTERCHANGE_SCHEMA_VERSION = "strict_vacuum_boozer_interchange_v1"
STRICT_VACUUM_SEED_MANIFEST_SCHEMA_VERSION = "strict_vacuum_seed_manifest_v1"
STRICT_VACUUM_SOURCE_CURRENT_GROUP_PROJECTION = "tf_banana_only"

STRICT_VACUUM_FORBIDDEN_COMMAND_FLAGS = (
    "--boozer-I",
    "--plasma-current-A",
    "--finite-current-mode",
    "--proxy-plasma-current-A",
    "--vf-current-A",
)
STRICT_VACUUM_FORBIDDEN_COMMAND_SUBSTRINGS = ("BoozerSurfaceFiniteI",)
STRICT_VACUUM_ZERO_TOL = 1.0e-14


def _argument_uses_flag(argument: str, flag: str) -> bool:
    return argument == flag or argument.startswith(f"{flag}=")


def validate_strict_vacuum_command(command_args: Sequence[str]) -> dict[str, object]:
    """Return validation evidence for the strict-vacuum command-line contract."""
    command_tokens = [str(argument) for argument in command_args]
    forbidden_flag_tokens = [
        argument
        for argument in command_tokens
        if any(
            _argument_uses_flag(argument, flag)
            for flag in STRICT_VACUUM_FORBIDDEN_COMMAND_FLAGS
        )
    ]
    forbidden_substring_tokens = [
        argument
        for argument in command_tokens
        if any(
            forbidden_substring in argument
            for forbidden_substring in STRICT_VACUUM_FORBIDDEN_COMMAND_SUBSTRINGS
        )
    ]
    checks = {
        "forbidden_flags": forbidden_flag_tokens,
        "forbidden_substrings": forbidden_substring_tokens,
    }
    checks["passed"] = not forbidden_flag_tokens and not forbidden_substring_tokens
    return checks


def _is_zero_number(value: object) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return abs(float(value)) <= STRICT_VACUUM_ZERO_TOL
    return False


def _is_negative_number(value: object) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return float(value) < -STRICT_VACUUM_ZERO_TOL
    return False


def _number_or_none(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _is_zero_or_missing(value: object) -> bool:
    return value is None or value == "" or _is_zero_number(value)


def _is_zero_count_or_missing(value: object) -> bool:
    if value is None or value == "":
        return True
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return value == 0
    return False


def _has_only_strict_boozer_values(value: object, expected_value: str) -> bool:
    if isinstance(value, (list, tuple)):
        return all(item == expected_value for item in value)
    return value == expected_value


def _signed_tf_current_negative(results: Mapping[str, object]) -> bool:
    tf_current = results.get("TF_CURRENT_A", results.get("STAGE2_TF_CURRENT_A"))
    return _is_negative_number(tf_current)


def _independent_banana_currents_have_signed_max_abs_control(
    results: Mapping[str, object],
    banana_currents: Sequence[object],
) -> bool:
    if results.get("BANANA_CURRENT_MODE") != "independent":
        return False
    if results.get("BANANA_CURRENT_CONTROL_METRIC") != "max_abs":
        return False
    current_values: list[float] = []
    for current in banana_currents:
        current_value = _number_or_none(current)
        if current_value is None:
            return False
        current_values.append(current_value)
    if not current_values:
        return False
    recorded_max_abs = _number_or_none(results.get("BANANA_CURRENT_MAX_ABS_A"))
    if recorded_max_abs is None:
        return False
    actual_max_abs = max(abs(current) for current in current_values)
    max_abs_tolerance = max(1.0e-9, 1.0e-12 * actual_max_abs)
    return (
        current_values[0] < -STRICT_VACUUM_ZERO_TOL
        and all(abs(current) > STRICT_VACUUM_ZERO_TOL for current in current_values)
        and abs(recorded_max_abs - actual_max_abs) <= max_abs_tolerance
    )


def _shared_mode_banana_currents_signed(banana_currents) -> bool:
    # Shared-mode sign contract (program Hard Invariant 12): one shared current
    # DOF drives every coil through ScaledCurrent(shared, +/-1), so the realized
    # per-coil list is values in {-I, +I} with a NEGATIVE base (CW lane). A fresh
    # shared state reports the control value only (all -I); a resumed seed
    # reports realized per-coil values (alternating -I/+I). Both are valid here:
    # this metadata check enforces the sign convention (negative base, single
    # shared magnitude); alternation-vs-flattened ORDERING is enforced by the
    # seed-loading contract, which sees the coil graph rather than a bare list.
    values: list[float] = []
    for current in banana_currents:
        if not isinstance(current, (int, float)) or isinstance(current, bool):
            return False
        values.append(float(current))
    if not values or values[0] >= 0.0:
        return False
    magnitude = abs(values[0])
    tolerance = 1e-6 * max(magnitude, 1.0)
    return all(abs(abs(value) - magnitude) <= tolerance for value in values)


def _signed_banana_current_negative(results: Mapping[str, object]) -> bool:
    banana_currents = results.get("BANANA_CURRENTS_A")
    if isinstance(banana_currents, (list, tuple)):
        if not banana_currents:
            return False
        if _independent_banana_currents_have_signed_max_abs_control(
            results,
            banana_currents,
        ):
            return True
        if results.get("BANANA_CURRENT_MODE") == "shared":
            return _shared_mode_banana_currents_signed(banana_currents)
        return all(_is_negative_number(current) for current in banana_currents)
    observed_currents: list[object] = []
    for key in ("BANANA_INIT_CURRENT_A", "BANANA_CURRENT_A"):
        value = results.get(key)
        if value not in {None, ""}:
            observed_currents.append(value)
    return bool(observed_currents) and all(
        _is_negative_number(current) for current in observed_currents
    )


def strict_vacuum_metadata_status(results: Mapping[str, object]) -> dict[str, object]:
    """Validate the result metadata required for a production strict-vacuum run."""
    seed_lineage = results.get("STRICT_VACUUM_SEED_LINEAGE")
    checks = {
        "strict_vacuum_current_true": results.get("STRICT_VACUUM_CURRENT") is True,
        "current_lineage_matches": (
            results.get("CURRENT_LINEAGE") == STRICT_VACUUM_CURRENT_LINEAGE
        ),
        "seed_lineage_recorded": seed_lineage in STRICT_VACUUM_SEED_LINEAGES,
        "recent_stage1_candidate_id_recorded": (
            seed_lineage != STRICT_VACUUM_SEED_LINEAGE_RECENT_STAGE1_CANDIDATE
            or bool(results.get("STAGE1_CANDIDATE_ID"))
        ),
        "effective_mode_vacuum": results.get("EFFECTIVE_CURRENT_MODE") == "vacuum",
        "finite_current_mode_absent": results.get("FINITE_CURRENT_MODE")
        in {None, "", "vacuum"},
        "signed_tf_current_negative": _signed_tf_current_negative(results),
        "signed_banana_current_negative": _signed_banana_current_negative(results),
        "plasma_current_zero": _is_zero_or_missing(results.get("PLASMA_CURRENT_A")),
        "boozer_i_zero": _is_zero_or_missing(results.get("BOOZER_I")),
        "proxy_current_zero": _is_zero_or_missing(
            results.get("PROXY_PLASMA_CURRENT_A")
        ),
        "vf_current_zero": _is_zero_or_missing(results.get("VF_CURRENT_A")),
        "num_proxy_coils_zero": _is_zero_count_or_missing(
            results.get("NUM_PROXY_COILS")
        ),
        "num_vf_coils_zero": _is_zero_count_or_missing(results.get("NUM_VF_COILS")),
        "boozer_surface_class_plain": _has_only_strict_boozer_values(
            results.get("BOOZER_SURFACE_CLASSES", results.get("BOOZER_SURFACE_CLASS")),
            STRICT_VACUUM_BOOZER_SURFACE_CLASS,
        ),
        "boozer_surface_module_plain": _has_only_strict_boozer_values(
            results.get("BOOZER_SURFACE_MODULES", results.get("BOOZER_SURFACE_MODULE")),
            STRICT_VACUUM_BOOZER_SURFACE_MODULE,
        ),
    }
    checks["passed"] = all(checks.values())
    return checks


def strict_vacuum_seed_input_status(
    seed_results: Mapping[str, object],
    *,
    num_proxy_coils: int,
    num_vf_coils: int,
    projected_source_current_groups: bool = False,
) -> dict[str, object]:
    """Validate source-current provenance for a strict-vacuum accepted output."""
    seed_records_finite_current = seed_results.get("FINITE_CURRENT_MODE") not in {
        None,
        "",
        "vacuum",
    }
    current_metadata_projected = (
        bool(projected_source_current_groups) or seed_records_finite_current
    )
    checks = {
        "effective_mode_vacuum_or_projected": (
            seed_results.get("EFFECTIVE_CURRENT_MODE") in {None, "", "vacuum"}
            or current_metadata_projected
        ),
        "plasma_current_zero_or_projected": (
            _is_zero_or_missing(seed_results.get("PLASMA_CURRENT_A"))
            or current_metadata_projected
        ),
        "boozer_i_zero_or_projected": (
            _is_zero_or_missing(seed_results.get("BOOZER_I"))
            or current_metadata_projected
        ),
        "proxy_current_zero_or_projected": (
            _is_zero_or_missing(seed_results.get("PROXY_PLASMA_CURRENT_A"))
            or current_metadata_projected
        ),
        "vf_current_zero_or_projected": (
            _is_zero_or_missing(seed_results.get("VF_CURRENT_A"))
            or current_metadata_projected
        ),
        "metadata_num_proxy_coils_zero_or_projected": (
            _is_zero_count_or_missing(seed_results.get("NUM_PROXY_COILS"))
            or projected_source_current_groups
        ),
        "metadata_num_vf_coils_zero_or_projected": (
            _is_zero_count_or_missing(seed_results.get("NUM_VF_COILS"))
            or projected_source_current_groups
        ),
        "loaded_num_proxy_coils_zero_or_projected": int(num_proxy_coils) == 0
        or projected_source_current_groups,
        "loaded_num_vf_coils_zero_or_projected": int(num_vf_coils) == 0
        or projected_source_current_groups,
    }
    checks["passed"] = all(checks.values())
    return checks


def failed_strict_vacuum_checks(checks: Mapping[str, object]) -> list[str]:
    return [
        key
        for key, value in checks.items()
        if key != "passed" and value is not True and value not in ([], ())
    ]


def strict_vacuum_boozer_interchange_manifest() -> dict[str, object]:
    return {
        "schema_version": STRICT_VACUUM_BOOZER_INTERCHANGE_SCHEMA_VERSION,
        "interchange_mode": STRICT_VACUUM_CURRENT_LINEAGE,
        "current_lineage": STRICT_VACUUM_CURRENT_LINEAGE,
        "baseline_replayable": True,
        "requires_boozer_surface_module": STRICT_VACUUM_BOOZER_SURFACE_MODULE,
        "requires_boozer_surface_class": STRICT_VACUUM_BOOZER_SURFACE_CLASS,
        "requires_no_boozer_surface_i_field": True,
        "requires_no_boozer_surface_finite_i": True,
        "biot_savart_current_contract": (
            "signed_negative_tf_and_banana_currents_embedded"
        ),
        "boozer_state_sidecar_fields": ["iota", "G"],
    }


def format_captured_command(executable: str, command_args: Sequence[str]) -> str:
    return shlex.join([executable, *[str(argument) for argument in command_args]])


def _file_manifest(path: str | Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    artifact_path = Path(path)
    if not artifact_path.is_file():
        return {
            "path": str(artifact_path),
            "exists": False,
            "bytes": None,
            "sha256": None,
        }
    return {
        "path": str(artifact_path),
        "exists": True,
        "bytes": artifact_path.stat().st_size,
        "sha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
    }


def build_strict_vacuum_seed_manifest(
    *,
    command_args: Sequence[str],
    strict_vacuum_seed_lineage: str,
    stage1_candidate_id: str | None,
    seed_biot_savart_path: str | Path,
    seed_results_path: str | Path,
    warm_start_surface_path: str | Path | None,
    plasma_target_path: str | Path,
    output_results_path: str | Path,
    results: Mapping[str, object],
    seed_results: Mapping[str, object],
    seed_artifact_role: str,
    projected_source_current_groups: bool,
) -> dict[str, object]:
    if strict_vacuum_seed_lineage not in STRICT_VACUUM_SEED_LINEAGES:
        raise ValueError(
            "Strict-vacuum seed lineage must be one of "
            f"{', '.join(STRICT_VACUUM_SEED_LINEAGES)}; "
            f"got {strict_vacuum_seed_lineage!r}."
        )
    seed_finite_current_mode = seed_results.get("FINITE_CURRENT_MODE")
    inherited_seed_caveat = (
        None
        if seed_finite_current_mode in {None, "", "vacuum"}
        else (
            "The seed records finite-current/proxy provenance and is used only "
            "as a warm-start/control input; strict-vacuum acceptance is based on "
            "the new run command, Boozer lineage, and result metadata."
        )
    )
    return {
        "schema_version": STRICT_VACUUM_SEED_MANIFEST_SCHEMA_VERSION,
        "lineage": strict_vacuum_seed_lineage,
        "current_lineage": STRICT_VACUUM_CURRENT_LINEAGE,
        "baseline_replayable": True,
        "boozer_interchange_manifest": strict_vacuum_boozer_interchange_manifest(),
        "stage1_candidate_id": stage1_candidate_id,
        "production_candidate": (
            strict_vacuum_seed_lineage
            == STRICT_VACUUM_SEED_LINEAGE_RECENT_STAGE1_CANDIDATE
        ),
        "control_only": (
            strict_vacuum_seed_lineage == STRICT_VACUUM_SEED_LINEAGE_LEGACY_CONTROL
        ),
        "seed_artifact_role": seed_artifact_role,
        "inherited_seed_caveat": inherited_seed_caveat,
        "source_current_group_projection": (
            STRICT_VACUUM_SOURCE_CURRENT_GROUP_PROJECTION
            if projected_source_current_groups
            else None
        ),
        "command": {
            "argv": [str(argument) for argument in command_args],
            "validation": validate_strict_vacuum_command(command_args),
        },
        "result_metadata_validation": strict_vacuum_metadata_status(results),
        "seed_input_validation": strict_vacuum_seed_input_status(
            seed_results,
            num_proxy_coils=int(results.get("STAGE2_SOURCE_NUM_PROXY_COILS", 0)),
            num_vf_coils=int(results.get("STAGE2_SOURCE_NUM_VF_COILS", 0)),
            projected_source_current_groups=projected_source_current_groups,
        ),
        "source_files": {
            "seed_biot_savart": _file_manifest(seed_biot_savart_path),
            "seed_results": _file_manifest(seed_results_path),
            "warm_start_surface": _file_manifest(warm_start_surface_path),
            "plasma_target": _file_manifest(plasma_target_path),
            "output_results": _file_manifest(output_results_path),
        },
        "seed_metadata": {
            "FINITE_CURRENT_MODE": seed_finite_current_mode,
            "EFFECTIVE_CURRENT_MODE": seed_results.get("EFFECTIVE_CURRENT_MODE"),
            "PLASMA_CURRENT_A": seed_results.get("PLASMA_CURRENT_A"),
            "BOOZER_I": seed_results.get("BOOZER_I"),
            "PROXY_PLASMA_CURRENT_A": seed_results.get("PROXY_PLASMA_CURRENT_A"),
            "VF_CURRENT_A": seed_results.get("VF_CURRENT_A"),
            "NUM_PROXY_COILS": seed_results.get("NUM_PROXY_COILS"),
            "NUM_VF_COILS": seed_results.get("NUM_VF_COILS"),
        },
        "accepted_metadata": {
            "NUM_PROXY_COILS": results.get("NUM_PROXY_COILS"),
            "NUM_VF_COILS": results.get("NUM_VF_COILS"),
            "CURRENT_LINEAGE": results.get("CURRENT_LINEAGE"),
            "STRICT_VACUUM_SEED_LINEAGE": results.get("STRICT_VACUUM_SEED_LINEAGE"),
            "STAGE1_CANDIDATE_ID": results.get("STAGE1_CANDIDATE_ID"),
            "ACCEPT_OFFSPEC_COIL_LENGTH": results.get("ACCEPT_OFFSPEC_COIL_LENGTH"),
            "COIL_LENGTH_HARD_LIMIT_M": results.get("COIL_LENGTH_HARD_LIMIT_M"),
            "OFFSPEC_COIL_LENGTH_TARGET_M": results.get("OFFSPEC_COIL_LENGTH_TARGET_M"),
        },
    }


def write_strict_vacuum_seed_manifest(
    manifest_path: str | Path,
    manifest: Mapping[str, object],
) -> None:
    Path(manifest_path).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
