"""Import sampled DESC coil results back into SIMSOPT-compatible artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from simsopt.field import BiotSavart
from simsopt.field.coil import Coil, Current

from banana_opt.desc_bridge.artifact_metadata import (
    DescBridgeArtifactMetadata,
    desc_bridge_metadata_with_residuals,
)
from banana_opt.desc_bridge.coil_export import DESC_EXPORT_GROUP_ORDER
from banana_opt.desc_bridge.coil_geometry import (
    curve_xyz_fourier_from_periodic_xyz_coefficients,
    fit_periodic_xyz,
    max_periodic_curvature,
    min_pairwise_periodic_distance,
    periodic_length,
)
from banana_opt.desc_bridge.coil_report_utils import (
    artifact_metadata_or_empty,
    coil_convention_report,
    current_sign,
    current_sign_counts,
    group_counts,
    optional_delta,
)
from banana_opt.desc_joint_validation import validate_desc_joint_final_oracle_evidence


@dataclass(frozen=True, slots=True)
class DescSampledCoil:
    group: str
    group_index: int
    name: str
    current_A: float
    coords_xyz: np.ndarray


@dataclass(frozen=True, slots=True)
class DescCoilImportEntry:
    group: str
    group_index: int
    import_index: int
    name: str
    current_A: float
    current_sign: str
    sample_count: int
    simsopt_fourier_order: int
    max_fit_residual_m: float
    rms_fit_residual_m: float
    source_length_m: float
    reconstructed_length_m: float
    length_delta_m: float
    source_max_curvature_inv_m: float
    reconstructed_max_curvature_inv_m: float
    max_curvature_delta_inv_m: float
    field_sample_delta_T: float | None = None

    def to_json_dict(self) -> dict[str, object]:
        return {
            "group": self.group,
            "group_index": self.group_index,
            "import_index": self.import_index,
            "name": self.name,
            "current_A": self.current_A,
            "current_sign": self.current_sign,
            "sample_count": self.sample_count,
            "simsopt_fourier_order": self.simsopt_fourier_order,
            "max_fit_residual_m": self.max_fit_residual_m,
            "rms_fit_residual_m": self.rms_fit_residual_m,
            "source_length_m": self.source_length_m,
            "reconstructed_length_m": self.reconstructed_length_m,
            "length_delta_m": self.length_delta_m,
            "source_max_curvature_inv_m": self.source_max_curvature_inv_m,
            "reconstructed_max_curvature_inv_m": self.reconstructed_max_curvature_inv_m,
            "max_curvature_delta_inv_m": self.max_curvature_delta_inv_m,
            "field_sample_delta_T": self.field_sample_delta_T,
        }


@dataclass(frozen=True, slots=True)
class DescCoilImportReport:
    sample_count: int
    simsopt_fourier_order: int
    artifact_metadata: DescBridgeArtifactMetadata
    group_order: tuple[str, ...]
    group_counts: Mapping[str, int]
    current_sign_counts: Mapping[str, int]
    source_min_coil_distance_m: float | None
    reconstructed_min_coil_distance_m: float | None
    min_coil_distance_delta_m: float | None
    entries: tuple[DescCoilImportEntry, ...]
    hardware_oracle_status: str
    final_oracle_evidence_path: str | None
    exported_artifact_paths: tuple[str, ...]

    def to_json_dict(self) -> dict[str, object]:
        return {
            "sample_count": self.sample_count,
            "simsopt_fourier_order": self.simsopt_fourier_order,
            "artifact_metadata": self.artifact_metadata.to_json_dict(),
            "group_order": list(self.group_order),
            "group_counts": dict(self.group_counts),
            "current_sign_counts": dict(self.current_sign_counts),
            "coil_conventions": coil_convention_report(),
            "source_min_coil_distance_m": self.source_min_coil_distance_m,
            "reconstructed_min_coil_distance_m": (
                self.reconstructed_min_coil_distance_m
            ),
            "min_coil_distance_delta_m": self.min_coil_distance_delta_m,
            "entries": [entry.to_json_dict() for entry in self.entries],
            "hardware_oracle_status": self.hardware_oracle_status,
            "final_oracle_evidence_path": self.final_oracle_evidence_path,
            "exported_artifact_paths": list(self.exported_artifact_paths),
        }


@dataclass(frozen=True, slots=True)
class DescCoilImportResult:
    biot_savart: BiotSavart
    coil_groups: Mapping[str, tuple[Coil, ...]]
    report: DescCoilImportReport


def import_desc_sampled_coils_to_simsopt(
    sampled_coils: Sequence[DescSampledCoil],
    *,
    simsopt_fourier_order: int,
    sample_count: int,
    coil_group_manifest: Mapping[str, int] | None = None,
    artifact_metadata: DescBridgeArtifactMetadata | None = None,
    hardware_oracle_status: str = "not_run",
    final_oracle_evidence_path: str | None = None,
    exported_artifact_paths: Sequence[str] = (),
) -> DescCoilImportResult:
    if simsopt_fourier_order <= 0:
        raise ValueError("simsopt_fourier_order must be positive.")
    if sample_count <= 2 * simsopt_fourier_order + 1:
        raise ValueError(
            "sample_count must exceed 2 * simsopt_fourier_order + 1 for a "
            "determined Fourier fit."
        )
    resolved_artifact_metadata = artifact_metadata_or_empty(artifact_metadata)
    validate_hardware_oracle_binding(
        hardware_oracle_status=hardware_oracle_status,
        final_oracle_evidence_path=final_oracle_evidence_path,
        artifact_metadata=resolved_artifact_metadata,
        exported_artifact_paths=exported_artifact_paths,
    )
    entries: list[DescCoilImportEntry] = []
    group_order = _import_group_order(sampled_coils, coil_group_manifest)
    _validate_group_manifest(sampled_coils, coil_group_manifest)
    ordered_sources = _ordered_sampled_coils(sampled_coils, group_order=group_order)
    grouped: dict[str, list[Coil]] = {group: [] for group in group_order}
    output_coils: list[Coil] = []
    source_curve_samples: list[np.ndarray] = []
    reconstructed_curve_samples: list[np.ndarray] = []
    for import_index, sampled_coil in enumerate(ordered_sources):
        coords = _coerce_coords(sampled_coil.coords_xyz, sample_count=sample_count)
        fit = fit_periodic_xyz(coords, order=simsopt_fourier_order)
        curve = curve_xyz_fourier_from_periodic_xyz_coefficients(
            fit.coefficients_xyz,
            sample_count=sample_count,
            order=simsopt_fourier_order,
        )
        coil = Coil(curve, Current(float(sampled_coil.current_A)))
        grouped[sampled_coil.group].append(coil)
        output_coils.append(coil)
        source_length = periodic_length(coords)
        reconstructed_length = periodic_length(fit.reconstructed_xyz)
        source_curvature = max_periodic_curvature(coords)
        reconstructed_curvature = max_periodic_curvature(fit.reconstructed_xyz)
        source_curve_samples.append(coords)
        reconstructed_curve_samples.append(fit.reconstructed_xyz)
        entries.append(
            DescCoilImportEntry(
                group=sampled_coil.group,
                group_index=sampled_coil.group_index,
                import_index=import_index,
                name=sampled_coil.name,
                current_A=float(sampled_coil.current_A),
                current_sign=current_sign(float(sampled_coil.current_A)),
                sample_count=sample_count,
                simsopt_fourier_order=simsopt_fourier_order,
                max_fit_residual_m=fit.max_residual_m,
                rms_fit_residual_m=fit.rms_residual_m,
                source_length_m=source_length,
                reconstructed_length_m=reconstructed_length,
                length_delta_m=reconstructed_length - source_length,
                source_max_curvature_inv_m=source_curvature,
                reconstructed_max_curvature_inv_m=reconstructed_curvature,
                max_curvature_delta_inv_m=reconstructed_curvature - source_curvature,
            )
        )
    source_min_distance = min_pairwise_periodic_distance(source_curve_samples)
    reconstructed_min_distance = min_pairwise_periodic_distance(
        reconstructed_curve_samples
    )
    min_distance_delta = optional_delta(
        reconstructed_min_distance,
        source_min_distance,
    )
    report_entries = tuple(entries)
    return DescCoilImportResult(
        biot_savart=BiotSavart(output_coils),
        coil_groups={group: tuple(grouped[group]) for group in group_order},
        report=DescCoilImportReport(
            sample_count=sample_count,
            simsopt_fourier_order=simsopt_fourier_order,
            artifact_metadata=desc_bridge_metadata_with_residuals(
                resolved_artifact_metadata,
                entries=report_entries,
                min_coil_distance_delta_m=min_distance_delta,
            ),
            group_order=group_order,
            group_counts=group_counts(entries, group_order=group_order),
            current_sign_counts=current_sign_counts(entries),
            source_min_coil_distance_m=source_min_distance,
            reconstructed_min_coil_distance_m=reconstructed_min_distance,
            min_coil_distance_delta_m=min_distance_delta,
            entries=report_entries,
            hardware_oracle_status=hardware_oracle_status,
            final_oracle_evidence_path=final_oracle_evidence_path,
            exported_artifact_paths=tuple(exported_artifact_paths),
        ),
    )


def validate_hardware_oracle_binding(
    *,
    hardware_oracle_status: str,
    final_oracle_evidence_path: str | None,
    artifact_metadata: DescBridgeArtifactMetadata | None = None,
    exported_artifact_paths: Sequence[str] = (),
) -> None:
    if hardware_oracle_status == "passed" and not final_oracle_evidence_path:
        raise ValueError(
            "DESC-imported artifacts cannot be marked hardware-clean without "
            "final_oracle_evidence_path."
        )
    if (
        hardware_oracle_status == "passed"
        and not Path(final_oracle_evidence_path).expanduser().is_file()
    ):
        raise ValueError(
            "DESC-imported artifacts cannot be marked hardware-clean with a missing "
            f"final_oracle_evidence_path: {final_oracle_evidence_path}."
        )
    if hardware_oracle_status == "passed":
        resolved_metadata = artifact_metadata_or_empty(artifact_metadata)
        if not resolved_metadata.source_artifact_checksums:
            raise ValueError(
                "DESC-imported artifacts cannot be marked hardware-clean without "
                "checksum-bound artifact_metadata."
            )
        resolved_metadata.validate_source_artifacts_current()
        validate_desc_joint_final_oracle_evidence(
            final_oracle_evidence_path,
            expected_source_artifact_checksums=resolved_metadata.checksum_map(),
            expected_exported_artifact_paths=exported_artifact_paths,
        )
    if hardware_oracle_status not in {"not_run", "passed", "failed", "blocked"}:
        raise ValueError(
            "hardware_oracle_status must be one of "
            "{not_run, passed, failed, blocked}."
        )


def _ordered_sampled_coils(
    sampled_coils: Sequence[DescSampledCoil],
    *,
    group_order: Sequence[str],
) -> tuple[DescSampledCoil, ...]:
    return tuple(
        sorted(
            sampled_coils,
            key=lambda coil: (
                group_order.index(coil.group),
                coil.group_index,
            ),
        )
    )


def _import_group_order(
    sampled_coils: Sequence[DescSampledCoil],
    coil_group_manifest: Mapping[str, int] | None,
) -> tuple[str, ...]:
    if coil_group_manifest is not None:
        _validate_manifest_names_and_counts(coil_group_manifest)
        return tuple(coil_group_manifest)
    return _ordered_group_names(sampled_coils)


def _validate_group_manifest(
    sampled_coils: Sequence[DescSampledCoil],
    coil_group_manifest: Mapping[str, int] | None,
) -> None:
    if coil_group_manifest is None:
        return
    _validate_sampled_group_identity(sampled_coils)
    observed_counts = _sampled_group_counts(sampled_coils)
    for group in observed_counts:
        if group not in coil_group_manifest:
            raise ValueError(
                "DESC sampled coil group is missing from the original manifest: "
                f"{group}."
            )
    for group, expected_count in coil_group_manifest.items():
        observed_count = observed_counts.get(group, 0)
        if observed_count != expected_count:
            raise ValueError(
                "DESC sampled coil group count does not match the original "
                f"manifest for {group}: observed {observed_count}, expected "
                f"{expected_count}."
            )
        observed_indices = sorted(
            sampled_coil.group_index
            for sampled_coil in sampled_coils
            if sampled_coil.group == group
        )
        expected_indices = list(range(expected_count))
        if observed_indices != expected_indices:
            raise ValueError(
                "DESC sampled coil group indices do not match the original "
                f"manifest for {group}: observed {observed_indices}, expected "
                f"{expected_indices}."
            )


def _validate_manifest_names_and_counts(
    coil_group_manifest: Mapping[str, int],
) -> None:
    for group, count in coil_group_manifest.items():
        if not isinstance(group, str) or group == "":
            raise ValueError("DESC original coil group names must be nonempty strings.")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(
                "DESC original coil group counts must be nonnegative integers."
            )


def _sampled_group_counts(
    sampled_coils: Sequence[DescSampledCoil],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sampled_coil in sampled_coils:
        counts[sampled_coil.group] = counts.get(sampled_coil.group, 0) + 1
    return counts


def _validate_sampled_group_identity(
    sampled_coils: Sequence[DescSampledCoil],
) -> None:
    for sampled_coil in sampled_coils:
        if not isinstance(sampled_coil.group, str) or sampled_coil.group == "":
            raise ValueError("DESC sampled coil group names must be nonempty strings.")
        if (
            isinstance(sampled_coil.group_index, bool)
            or not isinstance(sampled_coil.group_index, int)
            or sampled_coil.group_index < 0
        ):
            raise ValueError(
                "DESC sampled coil group_index values must be nonnegative integers."
            )


def _ordered_group_names(
    sampled_coils: Sequence[DescSampledCoil],
) -> tuple[str, ...]:
    for sampled_coil in sampled_coils:
        if not isinstance(sampled_coil.group, str) or sampled_coil.group == "":
            raise ValueError("DESC sampled coil group names must be nonempty strings.")
    present_groups = {sampled_coil.group for sampled_coil in sampled_coils}
    known_groups = [
        group
        for group in DESC_EXPORT_GROUP_ORDER
        if group in present_groups
    ]
    auxiliary_groups: list[str] = []
    for sampled_coil in sampled_coils:
        if (
            sampled_coil.group not in DESC_EXPORT_GROUP_ORDER
            and sampled_coil.group not in auxiliary_groups
        ):
            auxiliary_groups.append(sampled_coil.group)
    return tuple([*known_groups, *auxiliary_groups])


def _coerce_coords(coords: np.ndarray, *, sample_count: int) -> np.ndarray:
    coerced = np.asarray(coords, dtype=float)
    if coerced.shape != (sample_count, 3):
        raise ValueError(
            "DESC sampled coil coordinates must have shape "
            f"({sample_count}, 3); got {coerced.shape}."
        )
    if not np.isfinite(coerced).all():
        raise ValueError("DESC sampled coil coordinates must be finite.")
    return np.ascontiguousarray(coerced)


__all__ = [
    "DescCoilImportEntry",
    "DescCoilImportReport",
    "DescCoilImportResult",
    "DescSampledCoil",
    "import_desc_sampled_coils_to_simsopt",
    "validate_hardware_oracle_binding",
]
