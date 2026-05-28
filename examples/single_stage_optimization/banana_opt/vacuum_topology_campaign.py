from __future__ import annotations

import hashlib
import json
import shlex
from pathlib import Path
from typing import Mapping, Sequence

STRICT_VACUUM_CURRENT_LINEAGE = "strict_vacuum"
STRICT_VACUUM_BOOZER_SURFACE_CLASS = "BoozerSurface"
STRICT_VACUUM_BOOZER_SURFACE_MODULE = "simsopt.geo.boozersurface"
STRICT_VACUUM_SEED_MANIFEST_SCHEMA_VERSION = "strict_vacuum_seed_manifest_v1"

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
    checks["passed"] = (
        not forbidden_flag_tokens and not forbidden_substring_tokens
    )
    return checks


def _is_zero_number(value: object) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return abs(float(value)) <= STRICT_VACUUM_ZERO_TOL
    return False


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


def strict_vacuum_metadata_status(results: Mapping[str, object]) -> dict[str, object]:
    """Validate the result metadata required for a production strict-vacuum run."""
    checks = {
        "strict_vacuum_current_true": results.get("STRICT_VACUUM_CURRENT") is True,
        "current_lineage_matches": (
            results.get("CURRENT_LINEAGE") == STRICT_VACUUM_CURRENT_LINEAGE
        ),
        "effective_mode_vacuum": results.get("EFFECTIVE_CURRENT_MODE") == "vacuum",
        "finite_current_mode_absent": results.get("FINITE_CURRENT_MODE")
        in {None, "", "vacuum"},
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
) -> dict[str, object]:
    """Validate that a warm-start seed carries no active proxy/VF/current sources."""
    checks = {
        "effective_mode_vacuum_or_missing": seed_results.get("EFFECTIVE_CURRENT_MODE")
        in {None, "", "vacuum"},
        "plasma_current_zero_or_missing": _is_zero_or_missing(
            seed_results.get("PLASMA_CURRENT_A")
        ),
        "boozer_i_zero_or_missing": _is_zero_or_missing(seed_results.get("BOOZER_I")),
        "proxy_current_zero_or_missing": _is_zero_or_missing(
            seed_results.get("PROXY_PLASMA_CURRENT_A")
        ),
        "vf_current_zero_or_missing": _is_zero_or_missing(
            seed_results.get("VF_CURRENT_A")
        ),
        "metadata_num_proxy_coils_zero_or_missing": _is_zero_count_or_missing(
            seed_results.get("NUM_PROXY_COILS")
        ),
        "metadata_num_vf_coils_zero_or_missing": _is_zero_count_or_missing(
            seed_results.get("NUM_VF_COILS")
        ),
        "loaded_num_proxy_coils_zero": int(num_proxy_coils) == 0,
        "loaded_num_vf_coils_zero": int(num_vf_coils) == 0,
    }
    checks["passed"] = all(checks.values())
    return checks


def failed_strict_vacuum_checks(checks: Mapping[str, object]) -> list[str]:
    return [
        key
        for key, value in checks.items()
        if key != "passed" and value is not True and value not in ([], ())
    ]


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
    seed_biot_savart_path: str | Path,
    seed_results_path: str | Path,
    warm_start_surface_path: str | Path | None,
    plasma_target_path: str | Path,
    output_results_path: str | Path,
    results: Mapping[str, object],
    seed_results: Mapping[str, object],
    seed_artifact_role: str,
) -> dict[str, object]:
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
        "current_lineage": STRICT_VACUUM_CURRENT_LINEAGE,
        "seed_artifact_role": seed_artifact_role,
        "inherited_seed_caveat": inherited_seed_caveat,
        "command": {
            "argv": [str(argument) for argument in command_args],
            "validation": validate_strict_vacuum_command(command_args),
        },
        "result_metadata_validation": strict_vacuum_metadata_status(results),
        "seed_input_validation": strict_vacuum_seed_input_status(
            seed_results,
            num_proxy_coils=int(results.get("STAGE2_NUM_PROXY_COILS", 0)),
            num_vf_coils=int(results.get("STAGE2_NUM_VF_COILS", 0)),
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
    }


def write_strict_vacuum_seed_manifest(
    manifest_path: str | Path,
    manifest: Mapping[str, object],
) -> None:
    Path(manifest_path).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
