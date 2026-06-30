"""Materialize conversion-only DESC joint runner artifacts.

The helpers in this module execute the SIMSOPT -> sampled-DESC-coil ->
SIMSOPT bridge without claiming that DESC has optimized the coils.  They are
used by the Lane A smoke path to prove artifact plumbing, residual reports, and
loadable exported SIMSOPT fields before real DESC runtime optimization is
enabled.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from simsopt import load
from simsopt.field import BiotSavart
from simsopt.field.coil import Coil

from banana_opt.desc_bridge.artifact_metadata import (
    DescBridgeArtifactMetadata,
    DescBridgeBananaPackMetadata,
    desc_bridge_source_checksums,
)
from banana_opt.desc_bridge.coil_export import (
    DescCoilExportEntry,
    DescCoilExportReport,
    export_simsopt_coil_groups_to_desc,
)
from banana_opt.desc_bridge.coil_import import (
    DescCoilImportReport,
    DescSampledCoil,
    import_desc_sampled_coils_to_simsopt,
)

DESC_CONVERSION_ONLY_COIL_SCHEMA_VERSION = "desc_joint_conversion_only_coils_v1"


@dataclass(frozen=True, slots=True)
class SerializableDescFourierXYZCoil:
    current_A: float
    coords_xyz: np.ndarray
    order: int
    basis: str
    name: str

    @classmethod
    def from_values(
        cls,
        current_A: float,
        coords: np.ndarray,
        *,
        N: int,
        basis: str,
        name: str,
    ) -> "SerializableDescFourierXYZCoil":
        return cls(
            current_A=float(current_A),
            coords_xyz=np.asarray(coords, dtype=float).copy(),
            order=int(N),
            basis=basis,
            name=name,
        )

    def to_json_dict(self, *, entry: DescCoilExportEntry) -> dict[str, object]:
        return {
            "group": entry.group,
            "group_index": entry.group_index,
            "export_index": entry.export_index,
            "name": self.name,
            "current_A": self.current_A,
            "basis": self.basis,
            "order": self.order,
            "coords_xyz": self.coords_xyz.tolist(),
        }


@dataclass(frozen=True, slots=True)
class DescConversionOnlyArtifacts:
    desc_coils_path: Path
    export_report_path: Path
    exported_biot_savart_path: Path
    import_report_path: Path

    def artifact_paths(self) -> tuple[str, ...]:
        return (
            os.fspath(self.desc_coils_path),
            os.fspath(self.export_report_path),
            os.fspath(self.exported_biot_savart_path),
            os.fspath(self.import_report_path),
        )


def materialize_conversion_only_artifacts(
    *,
    source_field_path: Path,
    source_artifacts: Mapping[str, Path],
    coil_group_counts: Mapping[str, int],
    output_root: Path,
    desc_fourier_order: int,
    sample_count: int,
    simsopt_fourier_order: int,
    source_nfp: int | None = None,
    source_stellarator_symmetry: bool | None = None,
) -> DescConversionOnlyArtifacts:
    if not coil_group_counts:
        raise ValueError(
            "conversion-only execution requires explicit seed coil_group counts."
        )
    output_root.mkdir(parents=True, exist_ok=True)
    artifact_metadata = DescBridgeArtifactMetadata(
        source_artifact_checksums=desc_bridge_source_checksums(source_artifacts),
    )
    biot_savart = load_simsopt_biot_savart(source_field_path)
    coil_groups = coil_groups_from_biot_savart(
        biot_savart,
        coil_group_counts=coil_group_counts,
    )
    export_result = export_simsopt_coil_groups_to_desc(
        coil_groups,
        desc_fourier_xyz_coil_cls=SerializableDescFourierXYZCoil,
        desc_fourier_order=desc_fourier_order,
        sample_count=sample_count,
        artifact_metadata=artifact_metadata,
        source_nfp=source_nfp,
        source_stellarator_symmetry=source_stellarator_symmetry,
        banana_pack_metadata=_banana_pack_metadata_from_source_artifacts(
            source_artifacts,
        ),
        source_group_order=tuple(coil_group_counts),
    )
    desc_coils_path = output_root / "desc_coils_conversion_only.json"
    _write_json(
        desc_coils_path,
        _desc_coils_payload(
            desc_coils=export_result.desc_coils,
            report=export_result.report,
        ),
    )
    export_report_path = output_root / "desc_coil_export_report.json"
    _write_json(export_report_path, export_result.report.to_json_dict())

    exported_biot_savart_path = output_root / "biot_savart_desc_export.json"
    import_result = import_desc_sampled_coils_to_simsopt(
        _sampled_coils_from_export(
            desc_coils=export_result.desc_coils,
            report=export_result.report,
        ),
        simsopt_fourier_order=simsopt_fourier_order,
        sample_count=sample_count,
        coil_group_manifest=dict(coil_group_counts),
        artifact_metadata=export_result.report.artifact_metadata,
        hardware_oracle_status="not_run",
        exported_artifact_paths=(os.fspath(exported_biot_savart_path),),
    )
    import_result.biot_savart.save(str(exported_biot_savart_path))
    import_report_path = output_root / "desc_coil_import_report.json"
    _write_json(import_report_path, import_result.report.to_json_dict())
    return DescConversionOnlyArtifacts(
        desc_coils_path=desc_coils_path,
        export_report_path=export_report_path,
        exported_biot_savart_path=exported_biot_savart_path,
        import_report_path=import_report_path,
    )


def load_simsopt_biot_savart(path: Path) -> BiotSavart:
    loaded = load(str(path))
    if not isinstance(loaded, BiotSavart):
        raise TypeError(f"DESC joint seed field must load as BiotSavart: {path}.")
    return loaded


def coil_groups_from_biot_savart(
    biot_savart: BiotSavart,
    *,
    coil_group_counts: Mapping[str, int],
) -> dict[str, tuple[Coil, ...]]:
    coils = tuple(biot_savart.coils)
    grouped: dict[str, tuple[Coil, ...]] = {}
    cursor = 0
    for group_name, group_count in coil_group_counts.items():
        if group_count < 0:
            raise ValueError("DESC joint coil group counts must be nonnegative.")
        next_cursor = cursor + group_count
        grouped[group_name] = _require_simsopt_coils(coils[cursor:next_cursor])
        cursor = next_cursor
    if cursor != len(coils):
        raise ValueError(
            "DESC joint coil group counts do not cover the loaded BiotSavart "
            f"coils: grouped {cursor}, loaded {len(coils)}."
        )
    return grouped


def _require_simsopt_coils(coils: Sequence[Coil]) -> tuple[Coil, ...]:
    typed_coils: list[Coil] = []
    for coil in coils:
        if not isinstance(coil, Coil):
            raise TypeError("DESC joint BiotSavart entries must be simsopt Coil objects.")
        typed_coils.append(coil)
    return tuple(typed_coils)


def _desc_coils_payload(
    *,
    desc_coils: tuple[SerializableDescFourierXYZCoil, ...],
    report: DescCoilExportReport,
) -> dict[str, object]:
    return {
        "schema_version": DESC_CONVERSION_ONLY_COIL_SCHEMA_VERSION,
        "coordinate_basis": report.coordinate_basis,
        "sample_count": report.sample_count,
        "desc_fourier_order": report.desc_fourier_order,
        "group_order": list(report.group_order),
        "group_counts": dict(report.group_counts),
        "artifact_metadata": report.artifact_metadata.to_json_dict(),
        "coils": [
            coil.to_json_dict(entry=entry)
            for coil, entry in zip(desc_coils, report.entries, strict=True)
        ],
    }


def _sampled_coils_from_export(
    *,
    desc_coils: tuple[SerializableDescFourierXYZCoil, ...],
    report: DescCoilExportReport,
) -> tuple[DescSampledCoil, ...]:
    return tuple(
        DescSampledCoil(
            group=entry.group,
            group_index=entry.group_index,
            name=coil.name,
            current_A=coil.current_A,
            coords_xyz=coil.coords_xyz,
        )
        for coil, entry in zip(desc_coils, report.entries, strict=True)
    )


def _banana_pack_metadata_from_source_artifacts(
    source_artifacts: Mapping[str, Path],
) -> DescBridgeBananaPackMetadata | None:
    source_results_path = source_artifacts.get("seed_source_results")
    if source_results_path is None:
        return None
    payload = json.loads(source_results_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("seed_source_results must contain a JSON object.")
    finite_build_enabled = payload.get("FINITE_BUILD_ENABLED")
    filaments_per_banana = _optional_positive_int(
        payload.get("FINITEBUILD_FILAMENTS_PER_BANANA"),
        field_name="FINITEBUILD_FILAMENTS_PER_BANANA",
    )
    numfilaments_n = _optional_positive_int(
        payload.get("FINITEBUILD_NUMFILAMENTS_N"),
        field_name="FINITEBUILD_NUMFILAMENTS_N",
    )
    numfilaments_b = _optional_positive_int(
        payload.get("FINITEBUILD_NUMFILAMENTS_B"),
        field_name="FINITEBUILD_NUMFILAMENTS_B",
    )
    if (
        finite_build_enabled is None
        and filaments_per_banana is None
        and numfilaments_n is None
        and numfilaments_b is None
    ):
        return None
    if filaments_per_banana is None and (
        numfilaments_n is not None and numfilaments_b is not None
    ):
        filaments_per_banana = numfilaments_n * numfilaments_b
    return DescBridgeBananaPackMetadata(
        finite_build_enabled=(
            None if finite_build_enabled is None else _require_bool(
                finite_build_enabled,
                field_name="FINITE_BUILD_ENABLED",
            )
        ),
        filaments_per_banana=filaments_per_banana,
        numfilaments_n=numfilaments_n,
        numfilaments_b=numfilaments_b,
    )


def _require_bool(value: object, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be boolean when set.")
    return value


def _optional_positive_int(value: object, *, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer when set.")
    return value


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


__all__ = [
    "DESC_CONVERSION_ONLY_COIL_SCHEMA_VERSION",
    "DescConversionOnlyArtifacts",
    "SerializableDescFourierXYZCoil",
    "coil_groups_from_biot_savart",
    "load_simsopt_biot_savart",
    "materialize_conversion_only_artifacts",
]
