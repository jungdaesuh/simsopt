"""Artifact metadata shared by DESC bridge conversion reports."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

_SHA256_HEXDIGEST_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class DescBridgeSourceChecksum:
    name: str
    source_path: str
    sha256: str

    def __post_init__(self) -> None:
        _require_nonempty_name(self.name)
        resolved_path = Path(self.source_path).expanduser().resolve()
        if str(resolved_path) != self.source_path:
            raise ValueError(
                f"{self.name}.source_path must be an absolute resolved path."
            )
        _require_sha256_hexdigest(self.sha256, field_name=f"{self.name}.sha256")
        live_sha256 = _sha256_file(resolved_path)
        if live_sha256 != self.sha256:
            raise ValueError(
                f"{self.name}.sha256 does not match source_path content."
            )

    def to_json_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "source_path": self.source_path,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class DescBridgeConversionResidualSummary:
    max_fit_residual_m: float
    rms_fit_residual_m: float
    max_abs_length_delta_m: float
    max_abs_curvature_delta_inv_m: float
    min_coil_distance_delta_m: float | None
    max_field_sample_delta_T: float | None

    def __post_init__(self) -> None:
        _require_finite_nonnegative(
            self.max_fit_residual_m,
            field_name="max_fit_residual_m",
        )
        _require_finite_nonnegative(
            self.rms_fit_residual_m,
            field_name="rms_fit_residual_m",
        )
        _require_finite_nonnegative(
            self.max_abs_length_delta_m,
            field_name="max_abs_length_delta_m",
        )
        _require_finite_nonnegative(
            self.max_abs_curvature_delta_inv_m,
            field_name="max_abs_curvature_delta_inv_m",
        )
        if self.min_coil_distance_delta_m is not None and not math.isfinite(
            self.min_coil_distance_delta_m
        ):
            raise ValueError("min_coil_distance_delta_m must be finite when set.")
        if self.max_field_sample_delta_T is not None:
            _require_finite_nonnegative(
                self.max_field_sample_delta_T,
                field_name="max_field_sample_delta_T",
            )

    def to_json_dict(self) -> dict[str, float | None]:
        return {
            "max_fit_residual_m": self.max_fit_residual_m,
            "rms_fit_residual_m": self.rms_fit_residual_m,
            "max_abs_length_delta_m": self.max_abs_length_delta_m,
            "max_abs_curvature_delta_inv_m": self.max_abs_curvature_delta_inv_m,
            "min_coil_distance_delta_m": self.min_coil_distance_delta_m,
            "max_field_sample_delta_T": self.max_field_sample_delta_T,
        }


@dataclass(frozen=True, slots=True)
class DescBridgeSourceCoilGroup:
    role: str
    start: int
    count: int

    def __post_init__(self) -> None:
        _require_nonempty_name(self.role)
        _require_nonnegative_int(self.start, field_name=f"{self.role}.start")
        _require_nonnegative_int(self.count, field_name=f"{self.role}.count")

    @property
    def stop(self) -> int:
        return self.start + self.count

    def to_json_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "start": self.start,
            "count": self.count,
        }


@dataclass(frozen=True, slots=True)
class DescBridgeBananaPackMetadata:
    finite_build_enabled: bool | None = None
    filaments_per_banana: int | None = None
    numfilaments_n: int | None = None
    numfilaments_b: int | None = None

    def __post_init__(self) -> None:
        if self.finite_build_enabled is not None and not isinstance(
            self.finite_build_enabled,
            bool,
        ):
            raise ValueError("finite_build_enabled must be boolean when set.")
        _require_optional_positive_int(
            self.filaments_per_banana,
            field_name="filaments_per_banana",
        )
        _require_optional_positive_int(
            self.numfilaments_n,
            field_name="numfilaments_n",
        )
        _require_optional_positive_int(
            self.numfilaments_b,
            field_name="numfilaments_b",
        )
        if (
            self.filaments_per_banana is not None
            and self.numfilaments_n is not None
            and self.numfilaments_b is not None
        ):
            derived_filaments = self.numfilaments_n * self.numfilaments_b
            if self.filaments_per_banana != derived_filaments:
                raise ValueError(
                    "filaments_per_banana must equal "
                    f"numfilaments_n*numfilaments_b: "
                    f"{self.filaments_per_banana} vs {derived_filaments}."
                )
        if self.finite_build_enabled is True and self.filaments_per_banana is None:
            raise ValueError(
                "finite_build_enabled requires filaments_per_banana, or "
                "numfilaments_n and numfilaments_b from which it can be derived."
            )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "finite_build_enabled": self.finite_build_enabled,
            "filaments_per_banana": self.filaments_per_banana,
            "numfilaments_n": self.numfilaments_n,
            "numfilaments_b": self.numfilaments_b,
        }


@dataclass(frozen=True, slots=True)
class DescBridgeSourceIdentity:
    coil_names: tuple[str, ...]
    coil_group_manifest: tuple[DescBridgeSourceCoilGroup, ...]
    nfp: int | None = None
    stellarator_symmetry: bool | None = None
    banana_pack_metadata: DescBridgeBananaPackMetadata | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.coil_names, tuple):
            raise ValueError("DESC bridge source coil_names must be an immutable tuple.")
        for coil_name in self.coil_names:
            _require_nonempty_name(coil_name)
        if not isinstance(self.coil_group_manifest, tuple):
            raise ValueError(
                "DESC bridge source coil_group_manifest must be an immutable tuple."
            )
        cursor = 0
        for group in self.coil_group_manifest:
            if not isinstance(group, DescBridgeSourceCoilGroup):
                raise ValueError(
                    "DESC bridge source coil_group_manifest must contain "
                    "DescBridgeSourceCoilGroup entries."
                )
            if group.start != cursor:
                raise ValueError(
                    "DESC bridge source coil_group_manifest must be contiguous: "
                    f"{group.role} starts at {group.start}, expected {cursor}."
                )
            cursor = group.stop
        if self.coil_names and cursor != len(self.coil_names):
            raise ValueError(
                "DESC bridge source coil_group_manifest total must match "
                f"coil_names length: {cursor} vs {len(self.coil_names)}."
            )
        _require_optional_positive_int(self.nfp, field_name="nfp")
        if self.stellarator_symmetry is not None and not isinstance(
            self.stellarator_symmetry,
            bool,
        ):
            raise ValueError("stellarator_symmetry must be boolean when set.")
        if (
            self.banana_pack_metadata is not None
            and not isinstance(
                self.banana_pack_metadata,
                DescBridgeBananaPackMetadata,
            )
        ):
            raise ValueError(
                "banana_pack_metadata must be DescBridgeBananaPackMetadata when set."
            )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "coil_names": list(self.coil_names),
            "coil_group_manifest": [
                group.to_json_dict()
                for group in self.coil_group_manifest
            ],
            "nfp": self.nfp,
            "stellarator_symmetry": self.stellarator_symmetry,
            "banana_pack_metadata": (
                None
                if self.banana_pack_metadata is None
                else self.banana_pack_metadata.to_json_dict()
            ),
        }


class DescBridgeConversionResidualEntry(Protocol):
    max_fit_residual_m: float
    rms_fit_residual_m: float
    length_delta_m: float
    max_curvature_delta_inv_m: float
    field_sample_delta_T: float | None


@dataclass(frozen=True, slots=True)
class DescBridgeArtifactMetadata:
    desc_optimizer_version: str | None = None
    desc_commit: str | None = None
    source_artifact_checksums: tuple[DescBridgeSourceChecksum, ...] = ()
    source_identity: DescBridgeSourceIdentity | None = None
    conversion_residuals: DescBridgeConversionResidualSummary | None = None

    def __post_init__(self) -> None:
        if (
            self.desc_optimizer_version is not None
            and self.desc_optimizer_version == ""
        ):
            raise ValueError(
                "DESC bridge metadata desc_optimizer_version must be nonempty."
            )
        if self.desc_commit is not None and self.desc_commit == "":
            raise ValueError("DESC bridge metadata desc_commit must be nonempty.")
        if not isinstance(self.source_artifact_checksums, tuple):
            raise ValueError(
                "DESC bridge source_artifact_checksums must be an immutable tuple."
            )
        for checksum in self.source_artifact_checksums:
            if not isinstance(checksum, DescBridgeSourceChecksum):
                raise ValueError(
                    "DESC bridge source_artifact_checksums must contain "
                    "DescBridgeSourceChecksum entries."
                )
        names = [checksum.name for checksum in self.source_artifact_checksums]
        duplicates = sorted(name for name in set(names) if names.count(name) > 1)
        if duplicates:
            raise ValueError(
                "DESC bridge source checksum names must be unique: "
                f"{', '.join(duplicates)}."
            )
        if (
            self.source_identity is not None
            and not isinstance(self.source_identity, DescBridgeSourceIdentity)
        ):
            raise ValueError(
                "DESC bridge source_identity must be a DescBridgeSourceIdentity."
            )
        if (
            self.conversion_residuals is not None
            and not isinstance(
                self.conversion_residuals,
                DescBridgeConversionResidualSummary,
            )
        ):
            raise ValueError(
                "DESC bridge conversion_residuals must be a "
                "DescBridgeConversionResidualSummary."
            )

    def checksum_map(self) -> dict[str, str]:
        return {
            checksum.name: checksum.sha256
            for checksum in self.source_artifact_checksums
        }

    def source_path_map(self) -> dict[str, str]:
        return {
            checksum.name: checksum.source_path
            for checksum in self.source_artifact_checksums
        }

    def validate_source_artifacts_current(self) -> None:
        for checksum in self.source_artifact_checksums:
            live_sha256 = _sha256_file(Path(checksum.source_path))
            if live_sha256 != checksum.sha256:
                raise ValueError(
                    "DESC bridge source artifact checksum is stale for "
                    f"{checksum.name}: {checksum.source_path}."
                )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "desc_optimizer_version": self.desc_optimizer_version,
            "desc_commit": self.desc_commit,
            "source_artifact_paths": self.source_path_map(),
            "source_artifact_checksums": self.checksum_map(),
            "source_identity": (
                None
                if self.source_identity is None
                else self.source_identity.to_json_dict()
            ),
            "conversion_residuals": (
                None
                if self.conversion_residuals is None
                else self.conversion_residuals.to_json_dict()
            ),
        }


EMPTY_DESC_BRIDGE_ARTIFACT_METADATA = DescBridgeArtifactMetadata()


def desc_bridge_source_checksums(
    source_artifacts: Mapping[str, str | Path],
) -> tuple[DescBridgeSourceChecksum, ...]:
    return tuple(
        DescBridgeSourceChecksum(
            name=_require_nonempty_name(name),
            source_path=str(Path(path).expanduser().resolve()),
            sha256=_sha256_file(Path(path).expanduser()),
        )
        for name, path in source_artifacts.items()
    )


def desc_bridge_metadata_with_residuals(
    artifact_metadata: DescBridgeArtifactMetadata,
    *,
    entries: tuple[DescBridgeConversionResidualEntry, ...],
    min_coil_distance_delta_m: float | None,
) -> DescBridgeArtifactMetadata:
    return DescBridgeArtifactMetadata(
        desc_optimizer_version=artifact_metadata.desc_optimizer_version,
        desc_commit=artifact_metadata.desc_commit,
        source_artifact_checksums=artifact_metadata.source_artifact_checksums,
        source_identity=artifact_metadata.source_identity,
        conversion_residuals=desc_bridge_conversion_residual_summary(
            entries=entries,
            min_coil_distance_delta_m=min_coil_distance_delta_m,
        ),
    )


def desc_bridge_conversion_residual_summary(
    *,
    entries: tuple[DescBridgeConversionResidualEntry, ...],
    min_coil_distance_delta_m: float | None,
) -> DescBridgeConversionResidualSummary:
    if not entries:
        return DescBridgeConversionResidualSummary(
            max_fit_residual_m=0.0,
            rms_fit_residual_m=0.0,
            max_abs_length_delta_m=0.0,
            max_abs_curvature_delta_inv_m=0.0,
            min_coil_distance_delta_m=min_coil_distance_delta_m,
            max_field_sample_delta_T=None,
        )
    rms_residuals = [entry.rms_fit_residual_m for entry in entries]
    field_sample_deltas = [
        entry.field_sample_delta_T
        for entry in entries
        if entry.field_sample_delta_T is not None
    ]
    return DescBridgeConversionResidualSummary(
        max_fit_residual_m=max(
            entry.max_fit_residual_m for entry in entries
        ),
        rms_fit_residual_m=math.sqrt(
            sum(residual * residual for residual in rms_residuals)
            / float(len(rms_residuals))
        ),
        max_abs_length_delta_m=max(
            abs(entry.length_delta_m) for entry in entries
        ),
        max_abs_curvature_delta_inv_m=max(
            abs(entry.max_curvature_delta_inv_m) for entry in entries
        ),
        min_coil_distance_delta_m=min_coil_distance_delta_m,
        max_field_sample_delta_T=(
            None if not field_sample_deltas else max(field_sample_deltas)
        ),
    )


def _require_nonempty_name(name: str) -> str:
    if not isinstance(name, str) or name == "":
        raise ValueError("DESC bridge source artifact names must be nonempty strings.")
    return name


def _require_nonnegative_int(value: int, *, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a nonnegative integer.")


def _require_optional_positive_int(value: int | None, *, field_name: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer when set.")


def _require_sha256_hexdigest(value: str, *, field_name: str) -> None:
    if not isinstance(value, str) or not _SHA256_HEXDIGEST_RE.fullmatch(value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hex digest.")


def _require_finite_nonnegative(value: float, *, field_name: str) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{field_name} must be finite and nonnegative.")


def _sha256_file(path: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f"DESC bridge source artifact must be a file: {resolved}.")
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "EMPTY_DESC_BRIDGE_ARTIFACT_METADATA",
    "DescBridgeArtifactMetadata",
    "DescBridgeBananaPackMetadata",
    "DescBridgeConversionResidualEntry",
    "DescBridgeConversionResidualSummary",
    "DescBridgeSourceCoilGroup",
    "DescBridgeSourceChecksum",
    "DescBridgeSourceIdentity",
    "desc_bridge_conversion_residual_summary",
    "desc_bridge_metadata_with_residuals",
    "desc_bridge_source_checksums",
]
