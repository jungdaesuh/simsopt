"""Convert SIMSOPT coil groups into DESC FourierXYZ coils."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar, runtime_checkable

import numpy as np
from simsopt.field import BiotSavart
from simsopt.field.coil import Coil, Current
from simsopt.geo import CurveXYZFourier

from banana_opt.desc_bridge.artifact_metadata import (
    DescBridgeArtifactMetadata,
    DescBridgeBananaPackMetadata,
    DescBridgeSourceCoilGroup,
    DescBridgeSourceIdentity,
    desc_bridge_metadata_with_residuals,
)
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
    validate_group_names,
)

DESC_EXPORT_GROUP_ORDER: tuple[str, ...] = ("tf", "banana", "proxy", "vf")
DescCoilT = TypeVar("DescCoilT")


class DescFourierXyzCoilFactory(Protocol[DescCoilT]):
    @classmethod
    def from_values(
        cls,
        current_A: float,
        coords: np.ndarray,
        *,
        N: int,
        basis: str,
        name: str,
    ) -> DescCoilT:
        ...


class SimsoptCurve(Protocol):
    def gamma(self) -> np.ndarray:
        ...


@runtime_checkable
class ResampleableSimsoptCurve(SimsoptCurve, Protocol):
    quadpoints: np.ndarray

    def set_points(self, points: np.ndarray) -> None:
        ...


class SimsoptCurrent(Protocol):
    def get_value(self) -> float:
        ...


class SimsoptCoilLike(Protocol):
    curve: SimsoptCurve
    current: SimsoptCurrent


class Stage2CoilPartitions(Protocol):
    tf_coils: Sequence[SimsoptCoilLike]
    banana_coils: Sequence[SimsoptCoilLike]
    proxy_coils: Sequence[SimsoptCoilLike]
    vf_coils: Sequence[SimsoptCoilLike]


@dataclass(frozen=True, slots=True)
class DescCoilExportEntry:
    group: str
    group_index: int
    export_index: int
    name: str
    current_A: float
    current_sign: str
    sample_count: int
    coordinate_basis: str
    desc_fourier_order: int
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
            "export_index": self.export_index,
            "name": self.name,
            "current_A": self.current_A,
            "current_sign": self.current_sign,
            "sample_count": self.sample_count,
            "coordinate_basis": self.coordinate_basis,
            "desc_fourier_order": self.desc_fourier_order,
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
class DescCoilExportReport:
    coordinate_basis: str
    sample_count: int
    desc_fourier_order: int
    artifact_metadata: DescBridgeArtifactMetadata
    group_order: tuple[str, ...]
    group_counts: Mapping[str, int]
    current_sign_counts: Mapping[str, int]
    source_min_coil_distance_m: float | None
    reconstructed_min_coil_distance_m: float | None
    min_coil_distance_delta_m: float | None
    entries: tuple[DescCoilExportEntry, ...]

    def to_json_dict(self) -> dict[str, object]:
        return {
            "coordinate_basis": self.coordinate_basis,
            "sample_count": self.sample_count,
            "desc_fourier_order": self.desc_fourier_order,
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
        }


@dataclass(frozen=True, slots=True)
class DescCoilExportResult(Generic[DescCoilT]):
    desc_coils: tuple[DescCoilT, ...]
    report: DescCoilExportReport


def coil_groups_from_stage2_partitions(
    partitions: Stage2CoilPartitions,
) -> dict[str, tuple[SimsoptCoilLike, ...]]:
    return {
        "tf": tuple(partitions.tf_coils),
        "banana": tuple(partitions.banana_coils),
        "proxy": tuple(partitions.proxy_coils),
        "vf": tuple(partitions.vf_coils),
    }


def export_simsopt_coil_groups_to_desc(
    coil_groups: Mapping[str, Sequence[SimsoptCoilLike]],
    *,
    desc_fourier_xyz_coil_cls: type[DescFourierXyzCoilFactory[DescCoilT]],
    desc_fourier_order: int,
    sample_count: int,
    artifact_metadata: DescBridgeArtifactMetadata | None = None,
    source_nfp: int | None = None,
    source_stellarator_symmetry: bool | None = None,
    banana_pack_metadata: DescBridgeBananaPackMetadata | None = None,
    source_group_order: Sequence[str] | None = None,
) -> DescCoilExportResult[DescCoilT]:
    if desc_fourier_order <= 0:
        raise ValueError("desc_fourier_order must be positive.")
    if sample_count <= 2 * desc_fourier_order + 1:
        raise ValueError(
            "sample_count must exceed 2 * desc_fourier_order + 1 for a determined "
            "Fourier fit."
        )
    desc_coils: list[DescCoilT] = []
    entries: list[DescCoilExportEntry] = []
    source_curve_samples: list[np.ndarray] = []
    reconstructed_curve_samples: list[np.ndarray] = []
    export_index = 0
    group_order = _ordered_group_names(coil_groups)
    source_identity = _source_identity_from_coil_groups(
        coil_groups,
        group_order=_source_identity_group_order(
            coil_groups,
            source_group_order=source_group_order,
        ),
        nfp=source_nfp,
        stellarator_symmetry=source_stellarator_symmetry,
        banana_pack_metadata=banana_pack_metadata,
    )
    resolved_artifact_metadata = artifact_metadata_or_empty(artifact_metadata)
    source_bound_metadata = DescBridgeArtifactMetadata(
        desc_optimizer_version=resolved_artifact_metadata.desc_optimizer_version,
        desc_commit=resolved_artifact_metadata.desc_commit,
        source_artifact_checksums=resolved_artifact_metadata.source_artifact_checksums,
        source_identity=source_identity,
        conversion_residuals=resolved_artifact_metadata.conversion_residuals,
    )
    for group_name in group_order:
        for group_index, coil in enumerate(tuple(coil_groups.get(group_name, ()))):
            current_A = _coil_current_A(coil)
            source_samples = _sample_simsopt_curve_xyz(coil, sample_count=sample_count)
            fit = fit_periodic_xyz(source_samples, order=desc_fourier_order)
            name = f"{group_name}_{group_index:03d}"
            desc_coil = _build_desc_fourier_xyz_coil(
                desc_fourier_xyz_coil_cls,
                current_A=current_A,
                coords=source_samples,
                desc_fourier_order=desc_fourier_order,
                name=name,
            )
            source_length = periodic_length(source_samples)
            reconstructed_length = periodic_length(fit.reconstructed_xyz)
            source_curvature = max_periodic_curvature(source_samples)
            reconstructed_curvature = max_periodic_curvature(fit.reconstructed_xyz)
            field_sample_delta_T = _field_sample_delta_T(
                coil,
                fit.coefficients_xyz,
                source_samples=source_samples,
                sample_count=sample_count,
                order=desc_fourier_order,
                current_A=current_A,
            )
            source_curve_samples.append(source_samples)
            reconstructed_curve_samples.append(fit.reconstructed_xyz)
            desc_coils.append(desc_coil)
            entries.append(
                DescCoilExportEntry(
                    group=group_name,
                    group_index=group_index,
                    export_index=export_index,
                    name=name,
                    current_A=current_A,
                    current_sign=current_sign(current_A),
                    sample_count=sample_count,
                    coordinate_basis="xyz",
                    desc_fourier_order=desc_fourier_order,
                    max_fit_residual_m=fit.max_residual_m,
                    rms_fit_residual_m=fit.rms_residual_m,
                    source_length_m=source_length,
                    reconstructed_length_m=reconstructed_length,
                    length_delta_m=reconstructed_length - source_length,
                    source_max_curvature_inv_m=source_curvature,
                    reconstructed_max_curvature_inv_m=reconstructed_curvature,
                    max_curvature_delta_inv_m=(
                        reconstructed_curvature - source_curvature
                    ),
                    field_sample_delta_T=field_sample_delta_T,
                )
            )
            export_index += 1

    source_min_distance = min_pairwise_periodic_distance(source_curve_samples)
    reconstructed_min_distance = min_pairwise_periodic_distance(
        reconstructed_curve_samples
    )
    min_distance_delta = optional_delta(
        reconstructed_min_distance,
        source_min_distance,
    )
    report_entries = tuple(entries)
    return DescCoilExportResult(
        desc_coils=tuple(desc_coils),
        report=DescCoilExportReport(
            coordinate_basis="xyz",
            sample_count=sample_count,
            desc_fourier_order=desc_fourier_order,
            artifact_metadata=desc_bridge_metadata_with_residuals(
                source_bound_metadata,
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
        ),
    )


def _source_identity_from_coil_groups(
    coil_groups: Mapping[str, Sequence[SimsoptCoilLike]],
    *,
    group_order: tuple[str, ...],
    nfp: int | None,
    stellarator_symmetry: bool | None,
    banana_pack_metadata: DescBridgeBananaPackMetadata | None,
) -> DescBridgeSourceIdentity:
    coil_names: list[str] = []
    source_groups: list[DescBridgeSourceCoilGroup] = []
    cursor = 0
    for group_name in group_order:
        group_coils = tuple(coil_groups.get(group_name, ()))
        source_groups.append(
            DescBridgeSourceCoilGroup(
                role=group_name,
                start=cursor,
                count=len(group_coils),
            )
        )
        for group_index, coil in enumerate(group_coils):
            coil_names.append(
                _source_coil_name(
                    coil,
                    fallback_name=f"{group_name}_{group_index:03d}",
                )
            )
        cursor += len(group_coils)
    return DescBridgeSourceIdentity(
        coil_names=tuple(coil_names),
        coil_group_manifest=tuple(source_groups),
        nfp=nfp,
        stellarator_symmetry=stellarator_symmetry,
        banana_pack_metadata=banana_pack_metadata,
    )


def _source_identity_group_order(
    coil_groups: Mapping[str, Sequence[SimsoptCoilLike]],
    *,
    source_group_order: Sequence[str] | None,
) -> tuple[str, ...]:
    if source_group_order is None:
        return tuple(coil_groups)
    if isinstance(source_group_order, str):
        raise ValueError("source_group_order must be a sequence of group names.")
    group_order = tuple(source_group_order)
    for group_name in group_order:
        if not isinstance(group_name, str) or group_name == "":
            raise ValueError("source_group_order entries must be nonempty strings.")
    duplicates = sorted(
        group_name
        for group_name in set(group_order)
        if group_order.count(group_name) > 1
    )
    if duplicates:
        raise ValueError(
            "source_group_order entries must be unique: "
            f"{', '.join(duplicates)}."
        )
    coil_group_names = tuple(coil_groups)
    missing = [
        group_name
        for group_name in coil_group_names
        if group_name not in group_order
    ]
    extra = [
        group_name
        for group_name in group_order
        if group_name not in coil_groups
    ]
    if missing or extra:
        raise ValueError(
            "source_group_order must cover exactly the provided coil groups; "
            f"missing={missing}, extra={extra}."
        )
    return group_order


def _source_coil_name(coil: SimsoptCoilLike, *, fallback_name: str) -> str:
    raw_name = getattr(coil, "name", None)
    if isinstance(raw_name, str) and raw_name:
        return raw_name
    curve_name = getattr(coil.curve, "name", None)
    if isinstance(curve_name, str) and curve_name:
        return curve_name
    return fallback_name


def _build_desc_fourier_xyz_coil(
    desc_fourier_xyz_coil_cls: type[DescFourierXyzCoilFactory[DescCoilT]],
    *,
    current_A: float,
    coords: np.ndarray,
    desc_fourier_order: int,
    name: str,
) -> DescCoilT:
    return desc_fourier_xyz_coil_cls.from_values(
        current_A,
        coords,
        N=desc_fourier_order,
        basis="xyz",
        name=name,
    )


def _field_sample_delta_T(
    source_coil: SimsoptCoilLike,
    coefficients_xyz: np.ndarray,
    *,
    source_samples: np.ndarray,
    sample_count: int,
    order: int,
    current_A: float,
) -> float | None:
    if not isinstance(source_coil, Coil):
        return None
    reconstructed_curve = curve_xyz_fourier_from_periodic_xyz_coefficients(
        coefficients_xyz,
        sample_count=sample_count,
        order=order,
    )
    probe_points = _field_probe_points_xyz(source_samples)
    source_field = _evaluate_single_coil_field(source_coil, probe_points)
    reconstructed_field = _evaluate_single_coil_field(
        Coil(reconstructed_curve, Current(current_A)),
        probe_points,
    )
    return float(np.max(np.linalg.norm(source_field - reconstructed_field, axis=1)))


def _field_probe_points_xyz(source_samples: np.ndarray) -> np.ndarray:
    centroid = np.mean(source_samples, axis=0)
    radius = float(np.max(np.linalg.norm(source_samples - centroid, axis=1)))
    if radius <= 0.0:
        radius = 1.0
    offsets = radius * np.asarray(
        [
            (0.0, 0.0, 0.0),
            (0.25, 0.0, 0.15),
            (-0.25, 0.0, 0.15),
            (0.0, 0.25, -0.15),
            (0.0, -0.25, -0.15),
        ],
        dtype=float,
    )
    return np.ascontiguousarray(centroid + offsets)


def _evaluate_single_coil_field(coil: Coil, points_xyz: np.ndarray) -> np.ndarray:
    field = BiotSavart([coil])
    field.set_points(points_xyz)
    return np.asarray(field.B(), dtype=float)


def _coil_current_A(coil: SimsoptCoilLike) -> float:
    return float(coil.current.get_value())


def _ordered_group_names(
    coil_groups: Mapping[str, Sequence[SimsoptCoilLike]]
) -> tuple[str, ...]:
    validate_group_names(coil_groups, context="DESC export")
    known_groups = [group for group in DESC_EXPORT_GROUP_ORDER if group in coil_groups]
    auxiliary_groups = [
        group
        for group in coil_groups
        if group not in DESC_EXPORT_GROUP_ORDER
    ]
    return tuple([*known_groups, *auxiliary_groups])


def _sample_simsopt_curve_xyz(coil: SimsoptCoilLike, *, sample_count: int) -> np.ndarray:
    curve = coil.curve
    requested_points = np.linspace(0.0, 1.0, sample_count, endpoint=False)
    if isinstance(curve, ResampleableSimsoptCurve):
        original_points = curve.quadpoints
        curve.set_points(requested_points)
        try:
            coords = np.asarray(curve.gamma(), dtype=float)
        finally:
            curve.set_points(original_points)
    elif isinstance(curve, CurveXYZFourier):
        coords = _sample_curve_xyz_fourier(curve, requested_points=requested_points)
    else:
        coords = np.asarray(curve.gamma(), dtype=float)
        if coords.shape[0] != sample_count:
            raise ValueError(
                "SIMSOPT curve cannot be re-evaluated at the requested DESC "
                "sample_count without lossy linear resampling: "
                f"curve={type(curve).__module__}.{type(curve).__qualname__}, "
                f"native_samples={coords.shape[0]}, requested_samples={sample_count}."
            )
    if coords.shape != (sample_count, 3):
        raise ValueError(
            "SIMSOPT curve gamma() must return shape "
            f"({sample_count}, 3); got {coords.shape}."
        )
    if not np.isfinite(coords).all():
        raise ValueError("SIMSOPT curve samples must be finite.")
    return np.ascontiguousarray(coords)


def _sample_curve_xyz_fourier(
    curve: CurveXYZFourier,
    *,
    requested_points: np.ndarray,
) -> np.ndarray:
    dofs = np.asarray(curve.get_dofs(), dtype=float)
    per_coordinate_dofs, remainder = divmod(dofs.size, 3)
    if remainder != 0 or per_coordinate_dofs % 2 != 1:
        raise ValueError(
            "CurveXYZFourier dof layout must be 3*(2*order+1) to preserve "
            f"exact resampling; got {dofs.size} dofs."
        )
    order = (per_coordinate_dofs - 1) // 2
    sampled_curve = CurveXYZFourier(requested_points, order)
    sampled_curve.set_dofs(dofs)
    return np.asarray(sampled_curve.gamma(), dtype=float)


__all__ = [
    "DESC_EXPORT_GROUP_ORDER",
    "DescCoilExportEntry",
    "DescCoilExportReport",
    "DescCoilExportResult",
    "DescFourierXyzCoilFactory",
    "SimsoptCoilLike",
    "coil_groups_from_stage2_partitions",
    "export_simsopt_coil_groups_to_desc",
]
