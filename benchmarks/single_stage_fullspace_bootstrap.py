"""Canonical feasible-bootstrap artifact for the full-space GPU lane.

The producer runs the authoritative host adapter once, then seals the exact
joint state, target scalars, layout, masks, and source/runtime identity.  The
validator reconstructs every binary64 fingerprint and revalidates the runtime
evidence before accepting the artifact.
"""

from __future__ import annotations

import hashlib
import math
import os
from collections.abc import Callable, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Final, TypeAlias, cast

import jax
import numpy as np
from simsopt_jax_adapters.geo.single_stage_fullspace import (
    SingleStageFullSpaceBootstrap,
    bootstrap_target_payload,
    build_single_stage_fullspace_bootstrap,
)

from benchmarks.single_stage_fullspace_snapshot import (
    RUNTIME_EVIDENCE_SCHEMA_VERSION,
    ArtifactRef,
    JsonValue,
    RuntimeEvidence,
    RuntimeIdentity,
    SourceIdentity,
    canonical_json_bytes,
    load_canonical_json_bytes,
    runtime_identity_from_payload,
    source_identity_from_payload,
    validate_runtime_evidence,
)

SCHEMA_VERSION: Final = "single-stage-fullspace-bootstrap-v1"
AUTHORITATIVE_DTYPE: Final = "float64"
_ORDERING: Final = ("coil_dofs", "surface_dofs", "iota", "G")
_TARGET_NAMES: Final = (
    "volume_target",
    "iota_target",
    "major_radius_target",
    "length_target",
)
_SHA256_HEX: Final = frozenset("0123456789abcdef")
BootstrapFactory: TypeAlias = Callable[[], SingleStageFullSpaceBootstrap]


class BootstrapArtifactError(ValueError):
    """A bootstrap artifact failed its integrity or semantic contract."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _float64_bytes(values: np.ndarray) -> bytes:
    return np.ascontiguousarray(values, dtype="<f8").tobytes()


def _int32_bytes(values: np.ndarray) -> bytes:
    return np.ascontiguousarray(values, dtype="<i4").tobytes()


def _identity_payload(
    identity: SourceIdentity | RuntimeIdentity,
) -> dict[str, JsonValue]:
    return cast(dict[str, JsonValue], asdict(identity))


def _artifact_payload(reference: ArtifactRef) -> dict[str, JsonValue]:
    return cast(dict[str, JsonValue], asdict(reference))


def _reject_symlink_components(path: Path) -> None:
    current = path.absolute()
    while True:
        if current.is_symlink():
            raise BootstrapArtifactError("bootstrap artifact path contains a symlink")
        if current.parent == current:
            return
        current = current.parent


def _bootstrap_payload(
    bootstrap: SingleStageFullSpaceBootstrap,
    *,
    source_identity: SourceIdentity,
    runtime_identity: RuntimeIdentity,
    runtime_evidence: ArtifactRef,
) -> dict[str, JsonValue]:
    layout = bootstrap.problem.layout
    raw_z0 = np.asarray(jax.device_get(bootstrap.z0))
    if raw_z0.dtype != np.dtype(np.float64):
        raise BootstrapArtifactError("bootstrap joint state must use float64")
    z0 = np.asarray(raw_z0, dtype=np.float64)
    if z0.shape != (layout.total_dof_count,):
        raise BootstrapArtifactError("bootstrap joint state does not match its layout")
    if not np.all(np.isfinite(z0)):
        raise BootstrapArtifactError("bootstrap joint state is not finite")
    raw_mask = np.asarray(jax.device_get(bootstrap.problem.exact_mask_indices))
    if raw_mask.dtype != np.dtype(np.int32):
        raise BootstrapArtifactError("bootstrap exact mask must use int32")
    mask = np.asarray(raw_mask, dtype=np.int32)
    if mask.shape != (254,):
        raise BootstrapArtifactError("bootstrap exact mask must contain 254 indices")
    coil_end = layout.coil_dof_count
    surface_end = coil_end + layout.surface_dof_count
    component_ranges = {
        "coil_dofs": (0, coil_end),
        "surface_dofs": (coil_end, surface_end),
        "iota": (surface_end, surface_end + 1),
        "G": (surface_end + 1, surface_end + 2),
    }
    component_sha256 = {
        name: _sha256(_float64_bytes(z0[start:stop]))
        for name, (start, stop) in component_ranges.items()
    }
    target_payload = bootstrap_target_payload(bootstrap)
    return {
        "exact_mask": {
            "dtype": "int32",
            "little_endian_sha256": _sha256(_int32_bytes(mask)),
            "values": mask.tolist(),
        },
        "layout": {
            "coil_dof_count": layout.coil_dof_count,
            "component_ranges": {
                name: [start, stop] for name, (start, stop) in component_ranges.items()
            },
            "equality_count": mask.size + 1,
            "ordering": list(_ORDERING),
            "surface_dof_count": layout.surface_dof_count,
            "total_dof_count": layout.total_dof_count,
        },
        "runtime_evidence": _artifact_payload(runtime_evidence),
        "runtime_identity": _identity_payload(runtime_identity),
        "schema_version": SCHEMA_VERSION,
        "source_identity": _identity_payload(source_identity),
        "state": {
            "component_little_endian_sha256": component_sha256,
            "dtype": AUTHORITATIVE_DTYPE,
            "little_endian_sha256": _sha256(_float64_bytes(z0)),
            "values": z0.tolist(),
        },
        "targets": cast(JsonValue, target_payload),
    }


def _runtime_evidence_for_ref(
    reference: ArtifactRef,
    *,
    campaign_root: Path,
    snapshot_root: Path,
) -> RuntimeEvidence:
    if reference.schema_version != RUNTIME_EVIDENCE_SCHEMA_VERSION:
        raise BootstrapArtifactError("runtime evidence reference schema is invalid")
    evidence_path = reference.resolve_and_validate(campaign_root)
    return validate_runtime_evidence(
        evidence_path,
        snapshot_root=snapshot_root,
        campaign_root=campaign_root,
    )


def publish_bootstrap_artifact(
    path: Path,
    *,
    campaign_root: Path,
    snapshot_root: Path,
    runtime_evidence: ArtifactRef,
    bootstrap_factory: BootstrapFactory = build_single_stage_fullspace_bootstrap,
) -> ArtifactRef:
    """Run the exact bootstrap and exclusively seal its provenance-bound state."""

    campaign = campaign_root.resolve(strict=True)
    output = path.absolute()
    _reject_symlink_components(output)
    if not campaign.is_dir() or not output.parent.resolve(strict=True).is_relative_to(
        campaign
    ):
        raise BootstrapArtifactError("bootstrap output must be campaign-local")
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    evidence = _runtime_evidence_for_ref(
        runtime_evidence,
        campaign_root=campaign,
        snapshot_root=snapshot_root,
    )
    payload = canonical_json_bytes(
        _bootstrap_payload(
            bootstrap_factory(),
            source_identity=evidence.source_identity,
            runtime_identity=evidence.observation.runtime_identity,
            runtime_evidence=runtime_evidence,
        )
    )
    with output.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    output.chmod(0o444)
    return ArtifactRef(
        relative_path=output.relative_to(campaign).as_posix(),
        sha256=_sha256(payload),
        size_bytes=len(payload),
        schema_version=SCHEMA_VERSION,
    )


def _mapping(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise BootstrapArtifactError(f"{context} must be an object")
    return value


def _exact_keys(
    value: Mapping[str, object], expected: frozenset[str], context: str
) -> None:
    if frozenset(value) != expected:
        raise BootstrapArtifactError(f"{context} keys do not match schema")


def _integer(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BootstrapArtifactError(f"{context} must be an integer")
    return value


def _sha(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_HEX for character in value)
    ):
        raise BootstrapArtifactError(f"{context} must be a lowercase SHA-256")
    return value


def _float(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BootstrapArtifactError(f"{context} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise BootstrapArtifactError(f"{context} must be finite")
    return result


def _artifact_ref(value: object) -> ArtifactRef:
    mapping = _mapping(value, "runtime_evidence")
    _exact_keys(
        mapping,
        frozenset(("relative_path", "schema_version", "sha256", "size_bytes")),
        "runtime_evidence",
    )
    relative_path = mapping["relative_path"]
    schema_version = mapping["schema_version"]
    if not isinstance(relative_path, str) or not isinstance(schema_version, str):
        raise BootstrapArtifactError("runtime evidence paths/schema must be strings")
    return ArtifactRef(
        relative_path=relative_path,
        schema_version=schema_version,
        sha256=_sha(mapping["sha256"], "runtime_evidence.sha256"),
        size_bytes=_integer(mapping["size_bytes"], "runtime_evidence.size_bytes"),
    )


def _validate_target_fingerprints(value: object) -> None:
    payload = _mapping(value, "targets")
    _exact_keys(
        payload,
        frozenset(
            (
                "schema_version",
                "targets",
                "first_base_current",
                "initial_boozer_residual_norm",
                "joint_dof_count",
                "equality_count",
            )
        ),
        "targets",
    )
    if payload["schema_version"] != "single-stage-fullspace-bootstrap-targets-v1":
        raise BootstrapArtifactError("target schema version is invalid")
    targets = payload["targets"]
    if not isinstance(targets, list) or len(targets) != len(_TARGET_NAMES):
        raise BootstrapArtifactError("target list does not match the frozen contract")
    entries = [*targets, payload["first_base_current"]]
    expected_names = (*_TARGET_NAMES, "first_base_current")
    for index, (raw_entry, expected_name) in enumerate(
        zip(entries, expected_names, strict=True)
    ):
        entry = _mapping(raw_entry, f"target[{index}]")
        _exact_keys(
            entry,
            frozenset(("name", "value", "hexadecimal", "little_endian_sha256")),
            f"target[{index}]",
        )
        scalar = _float(entry["value"], f"target[{index}].value")
        if entry["name"] != expected_name or entry["hexadecimal"] != scalar.hex():
            raise BootstrapArtifactError(f"target[{index}] scalar identity differs")
        if _sha256(_float64_bytes(np.asarray([scalar]))) != _sha(
            entry["little_endian_sha256"], f"target[{index}].little_endian_sha256"
        ):
            raise BootstrapArtifactError(f"target[{index}] fingerprint differs")
    residual_norm = _float(
        payload["initial_boozer_residual_norm"], "initial_boozer_residual_norm"
    )
    if residual_norm < 0.0:
        raise BootstrapArtifactError("initial Boozer residual norm must be nonnegative")
    if _integer(payload["joint_dof_count"], "joint_dof_count") != 716:
        raise BootstrapArtifactError("joint DOF count differs")
    if _integer(payload["equality_count"], "equality_count") != 255:
        raise BootstrapArtifactError("equality count differs")


def validate_bootstrap_artifact(
    path: Path,
    *,
    campaign_root: Path,
    snapshot_root: Path,
) -> dict[str, JsonValue]:
    """Revalidate canonical bytes, binary identities, layout, and provenance."""

    campaign = campaign_root.resolve(strict=True)
    _reject_symlink_components(path)
    artifact_path = path.resolve(strict=True)
    if not artifact_path.is_file() or not artifact_path.is_relative_to(campaign):
        raise BootstrapArtifactError("bootstrap artifact must be campaign-local")
    if artifact_path.stat().st_mode & 0o222:
        raise BootstrapArtifactError("bootstrap artifact must be read-only")
    document_value = load_canonical_json_bytes(artifact_path.read_bytes())
    document = _mapping(document_value, "bootstrap artifact")
    _exact_keys(
        document,
        frozenset(
            (
                "schema_version",
                "source_identity",
                "runtime_identity",
                "runtime_evidence",
                "layout",
                "state",
                "targets",
                "exact_mask",
            )
        ),
        "bootstrap artifact",
    )
    if document["schema_version"] != SCHEMA_VERSION:
        raise BootstrapArtifactError("bootstrap schema version is invalid")
    runtime_ref = _artifact_ref(document["runtime_evidence"])
    evidence = _runtime_evidence_for_ref(
        runtime_ref, campaign_root=campaign, snapshot_root=snapshot_root
    )
    source = source_identity_from_payload(document["source_identity"])
    runtime = runtime_identity_from_payload(document["runtime_identity"])
    if source != evidence.source_identity:
        raise BootstrapArtifactError("bootstrap source identity differs")
    if runtime != evidence.observation.runtime_identity:
        raise BootstrapArtifactError("bootstrap runtime identity differs")

    layout = _mapping(document["layout"], "layout")
    _exact_keys(
        layout,
        frozenset(
            (
                "ordering",
                "coil_dof_count",
                "surface_dof_count",
                "total_dof_count",
                "equality_count",
                "component_ranges",
            )
        ),
        "layout",
    )
    expected_ranges = {
        "coil_dofs": [0, 461],
        "surface_dofs": [461, 714],
        "iota": [714, 715],
        "G": [715, 716],
    }
    if (
        layout["ordering"] != list(_ORDERING)
        or layout["component_ranges"] != expected_ranges
        or _integer(layout["coil_dof_count"], "coil_dof_count") != 461
        or _integer(layout["surface_dof_count"], "surface_dof_count") != 253
        or _integer(layout["total_dof_count"], "total_dof_count") != 716
        or _integer(layout["equality_count"], "equality_count") != 255
    ):
        raise BootstrapArtifactError("bootstrap layout differs from frozen layout")

    state = _mapping(document["state"], "state")
    _exact_keys(
        state,
        frozenset(
            (
                "dtype",
                "values",
                "little_endian_sha256",
                "component_little_endian_sha256",
            )
        ),
        "state",
    )
    values = state["values"]
    if state["dtype"] != AUTHORITATIVE_DTYPE or not isinstance(values, list):
        raise BootstrapArtifactError("bootstrap state dtype or values are invalid")
    z0 = np.asarray(
        [_float(value, f"state.values[{index}]") for index, value in enumerate(values)],
        dtype=np.float64,
    )
    if z0.shape != (716,) or _sha256(_float64_bytes(z0)) != _sha(
        state["little_endian_sha256"], "state.little_endian_sha256"
    ):
        raise BootstrapArtifactError("bootstrap state fingerprint differs")
    component_hashes = _mapping(
        state["component_little_endian_sha256"], "state component fingerprints"
    )
    _exact_keys(component_hashes, frozenset(_ORDERING), "state component fingerprints")
    for name, (start, stop) in {
        "coil_dofs": (0, 461),
        "surface_dofs": (461, 714),
        "iota": (714, 715),
        "G": (715, 716),
    }.items():
        if _sha256(_float64_bytes(z0[start:stop])) != _sha(
            component_hashes[name], f"state component {name}"
        ):
            raise BootstrapArtifactError(f"state component {name} fingerprint differs")

    mask_payload = _mapping(document["exact_mask"], "exact_mask")
    _exact_keys(
        mask_payload,
        frozenset(("dtype", "values", "little_endian_sha256")),
        "exact_mask",
    )
    raw_mask = mask_payload["values"]
    if mask_payload["dtype"] != "int32" or not isinstance(raw_mask, list):
        raise BootstrapArtifactError("exact mask dtype or values are invalid")
    mask = np.asarray(
        [
            _integer(value, f"exact_mask.values[{index}]")
            for index, value in enumerate(raw_mask)
        ],
        dtype=np.int32,
    )
    if (
        mask.shape != (254,)
        or np.any(mask < 0)
        or len(set(mask.tolist())) != mask.size
        or _sha256(_int32_bytes(mask))
        != _sha(mask_payload["little_endian_sha256"], "exact mask fingerprint")
    ):
        raise BootstrapArtifactError("exact mask fingerprint or indices differ")
    _validate_target_fingerprints(document["targets"])
    return cast(dict[str, JsonValue], document)


__all__ = (
    "AUTHORITATIVE_DTYPE",
    "SCHEMA_VERSION",
    "BootstrapArtifactError",
    "publish_bootstrap_artifact",
    "validate_bootstrap_artifact",
)
