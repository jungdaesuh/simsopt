"""Hardware-first configuration boundary for DESC joint banana runs.

The DESC-joint runner carries hardware paths and provenance here, while the
actual threshold values remain in ``hardware_constraint_schema``.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal

from banana_opt.hardware_constraint_schema import (
    HardwareConstraintSpec,
    get_hardware_constraint_spec,
    hardware_constraint_artifact_payload_field_names,
    hardware_constraint_specs,
)
from banana_opt.hardware_keepout import (
    hardware_keepout_metadata,
    hardware_sdf_metadata,
)
from banana_opt.desc_joint_io import (
    read_json_mapping,
    sha256_file as _sha256_file,
)

DESC_JOINT_HARDWARE_SPEC_SCHEMA_VERSION = "desc_joint_hardware_spec_v1"

CoilGroupPolicy = Literal["fixed", "optimized", "excluded"]
_COIL_GROUP_POLICIES = frozenset({"fixed", "optimized", "excluded"})

DESC_JOINT_REQUIRED_HARDWARE_CONSTRAINTS: tuple[str, ...] = (
    "coil_length",
    "coil_length_min",
    "coil_coil_spacing",
    "coil_surface_spacing",
    "max_curvature",
    "banana_current",
    "tf_current",
    "width_min",
    "width_max",
    "hardware_keepout",
)

_REQUIRED_TOP_LEVEL_KEYS = (
    "schema_version",
    "hardware_sources",
    "coil_group_policy",
)
_TOP_LEVEL_KEYS = _REQUIRED_TOP_LEVEL_KEYS + ("constraint_names",)
_REQUIRED_HARDWARE_SOURCE_KEYS = ("glb", "final_oracle")
_HARDWARE_SOURCE_KEYS = _REQUIRED_HARDWARE_SOURCE_KEYS + (
    "hardware_keepout_json",
    "hardware_sdf",
)
_REQUIRED_COIL_GROUP_POLICY_KEYS = ("tf", "banana", "proxy", "vf")


@dataclass(frozen=True, slots=True)
class DescJointHardwareSpec:
    """Resolved DESC-joint hardware input contract.

    Paths are resolved relative to the JSON spec file. Keepout/SDF freshness is
    already checked against ``glb_path`` when this object is constructed.
    """

    spec_path: Path
    glb_path: Path
    final_oracle_path: Path
    hardware_keepout_json_path: Path | None
    hardware_sdf_manifest_path: Path | None
    coil_group_policy: Mapping[str, CoilGroupPolicy]
    constraint_names: tuple[str, ...]
    hardware_metadata: Mapping[str, object]

    def constraint_specs(self) -> tuple[HardwareConstraintSpec, ...]:
        return hardware_constraint_specs(
            applies_to="artifact",
            names=self.constraint_names,
        )

    def threshold_by_name(self) -> dict[str, float]:
        return {
            spec.name: float(spec.threshold)
            for spec in self.constraint_specs()
        }

    def artifact_payload_field_names(self, *, prefix: str = "") -> tuple[str, ...]:
        return hardware_constraint_artifact_payload_field_names(
            prefix=prefix,
            names=self.constraint_names,
        )

    def source_checksums(self) -> dict[str, str]:
        checksums = {
            "glb": _sha256_file(self.glb_path),
            "final_oracle": _sha256_file(self.final_oracle_path),
        }
        if self.hardware_keepout_json_path is not None:
            checksums["hardware_keepout_json"] = _sha256_file(
                self.hardware_keepout_json_path
            )
        if self.hardware_sdf_manifest_path is not None:
            checksums["hardware_sdf_manifest"] = _sha256_file(
                self.hardware_sdf_manifest_path
            )
        return checksums

    def to_input_contract(self) -> dict[str, object]:
        return {
            "schema_version": DESC_JOINT_HARDWARE_SPEC_SCHEMA_VERSION,
            "spec_path": os.fspath(self.spec_path),
            "hardware_sources": {
                "glb": os.fspath(self.glb_path),
                "hardware_keepout_json": _optional_fspath(
                    self.hardware_keepout_json_path
                ),
                "hardware_sdf": _optional_fspath(self.hardware_sdf_manifest_path),
                "final_oracle": os.fspath(self.final_oracle_path),
            },
            "coil_group_policy": dict(self.coil_group_policy),
            "constraint_names": list(self.constraint_names),
            "constraint_thresholds": self.threshold_by_name(),
            "artifact_payload_fields": list(self.artifact_payload_field_names()),
            "source_checksums": self.source_checksums(),
            "hardware_metadata": dict(self.hardware_metadata),
        }


def load_desc_joint_hardware_spec(path: str | Path) -> DescJointHardwareSpec:
    spec_path = Path(path).expanduser().resolve()
    payload = _read_json_mapping(spec_path)
    _require_schema_version(payload)
    _require_keys(payload, _REQUIRED_TOP_LEVEL_KEYS, owner="hardware spec")
    _reject_unknown_keys(payload, _TOP_LEVEL_KEYS, owner="hardware spec")

    hardware_sources = _require_mapping(payload, "hardware_sources")
    _require_keys(
        hardware_sources,
        _REQUIRED_HARDWARE_SOURCE_KEYS,
        owner="hardware_sources",
    )
    _reject_unknown_keys(
        hardware_sources,
        _HARDWARE_SOURCE_KEYS,
        owner="hardware_sources",
    )
    glb_path = _require_existing_path(
        spec_path,
        hardware_sources,
        "glb",
    )
    final_oracle_path = _require_existing_path(
        spec_path,
        hardware_sources,
        "final_oracle",
    )
    keepout_path = _optional_existing_path(
        spec_path,
        hardware_sources,
        "hardware_keepout_json",
    )
    sdf_path = _optional_existing_path(spec_path, hardware_sources, "hardware_sdf")
    if keepout_path is None and sdf_path is None:
        raise ValueError(
            "DESC-joint hardware spec requires hardware_sources.hardware_keepout_json "
            "or hardware_sources.hardware_sdf."
        )

    policy = _coerce_coil_group_policy(_require_mapping(payload, "coil_group_policy"))
    constraint_names = _coerce_constraint_names(payload.get("constraint_names"))
    metadata = _resolve_hardware_metadata(
        glb_path=glb_path,
        keepout_path=keepout_path,
        sdf_path=sdf_path,
    )
    return DescJointHardwareSpec(
        spec_path=spec_path,
        glb_path=glb_path,
        final_oracle_path=final_oracle_path,
        hardware_keepout_json_path=keepout_path,
        hardware_sdf_manifest_path=sdf_path,
        coil_group_policy=MappingProxyType(policy),
        constraint_names=constraint_names,
        hardware_metadata=MappingProxyType(metadata),
    )


def _resolve_hardware_metadata(
    *,
    glb_path: Path,
    keepout_path: Path | None,
    sdf_path: Path | None,
) -> dict[str, object]:
    metadata: dict[str, object] = {}
    if keepout_path is not None:
        metadata.update(hardware_keepout_metadata(keepout_path, glb_path=glb_path))
    if sdf_path is not None:
        metadata.update(hardware_sdf_metadata(sdf_path, glb_path=glb_path))
    return metadata


def _read_json_mapping(path: Path) -> Mapping[str, object]:
    return read_json_mapping(
        path,
        error_message=lambda payload: (
            "DESC-joint hardware spec must be a JSON object; got "
            f"{type(payload).__name__}."
        ),
    )


def _require_schema_version(payload: Mapping[str, object]) -> None:
    version = payload.get("schema_version")
    if version != DESC_JOINT_HARDWARE_SPEC_SCHEMA_VERSION:
        raise ValueError(
            "DESC-joint hardware spec schema_version must be "
            f"{DESC_JOINT_HARDWARE_SPEC_SCHEMA_VERSION!r}; got {version!r}."
        )


def _require_keys(
    payload: Mapping[str, object],
    required_keys: Sequence[str],
    *,
    owner: str,
) -> None:
    missing = [key for key in required_keys if key not in payload]
    if missing:
        raise ValueError(f"{owner} missing required keys: {', '.join(missing)}.")


def _reject_unknown_keys(
    payload: Mapping[str, object],
    known_keys: Sequence[str],
    *,
    owner: str,
) -> None:
    unknown = sorted(set(payload) - set(known_keys))
    if unknown:
        raise ValueError(f"{owner} has unknown keys: {', '.join(unknown)}.")


def _require_mapping(payload: Mapping[str, object], field_name: str) -> Mapping[str, object]:
    value = payload.get(field_name)
    if not isinstance(value, Mapping):
        raise ValueError(f"DESC-joint hardware spec field {field_name!r} must be an object.")
    return value


def _resolve_path(spec_path: Path, raw_path: object, field_name: str) -> Path:
    if not isinstance(raw_path, str) or raw_path == "":
        raise ValueError(f"hardware_sources.{field_name} must be a nonempty path string.")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = spec_path.parent / path
    return path.resolve()


def _require_existing_path(
    spec_path: Path,
    payload: Mapping[str, object],
    field_name: str,
) -> Path:
    path = _resolve_path(spec_path, payload.get(field_name), field_name)
    if not path.exists():
        raise ValueError(f"hardware_sources.{field_name} does not exist: {path}.")
    if not path.is_file():
        raise ValueError(f"hardware_sources.{field_name} must be a file: {path}.")
    return path


def _optional_existing_path(
    spec_path: Path,
    payload: Mapping[str, object],
    field_name: str,
) -> Path | None:
    raw_path = payload.get(field_name)
    if raw_path is None:
        return None
    return _require_existing_path(spec_path, payload, field_name)


def _coerce_coil_group_policy(
    payload: Mapping[str, object],
) -> dict[str, CoilGroupPolicy]:
    _require_keys(
        payload,
        _REQUIRED_COIL_GROUP_POLICY_KEYS,
        owner="coil_group_policy",
    )
    unknown = sorted(set(payload) - set(_REQUIRED_COIL_GROUP_POLICY_KEYS))
    if unknown:
        raise ValueError(f"coil_group_policy has unknown groups: {', '.join(unknown)}.")
    policy: dict[str, CoilGroupPolicy] = {}
    for group_name in _REQUIRED_COIL_GROUP_POLICY_KEYS:
        policy[group_name] = _coerce_coil_group_policy_value(
            group_name,
            payload[group_name],
        )
    return policy


def _coerce_coil_group_policy_value(
    group_name: str,
    raw_value: object,
) -> CoilGroupPolicy:
    if raw_value == "fixed":
        return "fixed"
    if raw_value == "optimized":
        return "optimized"
    if raw_value == "excluded":
        return "excluded"
    choices = ", ".join(sorted(_COIL_GROUP_POLICIES))
    raise ValueError(
        f"coil_group_policy.{group_name} must be one of {{{choices}}}; "
        f"got {raw_value!r}."
    )


def _coerce_constraint_names(raw_names: object) -> tuple[str, ...]:
    if raw_names is None:
        names = DESC_JOINT_REQUIRED_HARDWARE_CONSTRAINTS
    else:
        if isinstance(raw_names, str) or not isinstance(raw_names, Sequence):
            raise ValueError("constraint_names must be a list of schema names.")
        names = tuple(_coerce_constraint_name(name) for name in raw_names)
    if len(set(names)) != len(names):
        raise ValueError("constraint_names contains duplicate entries.")
    for name in names:
        get_hardware_constraint_spec(name)
    missing_required = sorted(set(DESC_JOINT_REQUIRED_HARDWARE_CONSTRAINTS) - set(names))
    if missing_required:
        raise ValueError(
            "DESC-joint hardware spec missing required hardware constraints: "
            f"{', '.join(missing_required)}."
        )
    return tuple(names)


def _coerce_constraint_name(raw_name: object) -> str:
    if not isinstance(raw_name, str) or raw_name == "":
        raise ValueError("constraint_names entries must be nonempty strings.")
    return raw_name



def _optional_fspath(path: Path | None) -> str | None:
    return None if path is None else os.fspath(path)


__all__ = [
    "CoilGroupPolicy",
    "DESC_JOINT_HARDWARE_SPEC_SCHEMA_VERSION",
    "DESC_JOINT_REQUIRED_HARDWARE_CONSTRAINTS",
    "DescJointHardwareSpec",
    "load_desc_joint_hardware_spec",
]
