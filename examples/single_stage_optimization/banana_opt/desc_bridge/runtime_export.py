"""Export optimized DESC coilsets back to SIMSOPT artifacts."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from banana_opt.desc_bridge.artifact_metadata import (
    DescBridgeArtifactMetadata,
    desc_bridge_source_checksums,
)
from banana_opt.desc_bridge.coil_import import (
    DescCoilImportReport,
    DescSampledCoil,
    import_desc_sampled_coils_to_simsopt,
)
from banana_opt.desc_bridge.runtime_imports import activate_desc_source_root

DescOptimizedSimsoptExportStatus = Literal["passed", "failed"]
DescOptimizedSurfaceExportStatus = Literal["passed", "failed"]
DESC_OPTIMIZED_SURFACE_MAX_FIT_RESIDUAL_M = 1.0e-6


@dataclass(frozen=True, slots=True)
class DescOptimizedSimsoptExportReport:
    status: DescOptimizedSimsoptExportStatus
    reason: str
    optimized_coilset_type: str | None
    sample_count: int
    simsopt_fourier_order: int
    coil_group_counts: Mapping[str, int]
    artifact_metadata: DescBridgeArtifactMetadata | None
    optimized_coilset_source_path: Path | None
    exported_biot_savart_path: Path | None
    import_report_path: Path | None

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": "desc_optimized_simsopt_export_report_v1",
            "status": self.status,
            "reason": self.reason,
            "optimized_coilset_type": self.optimized_coilset_type,
            "sample_count": self.sample_count,
            "simsopt_fourier_order": self.simsopt_fourier_order,
            "coil_group_counts": dict(self.coil_group_counts),
            "artifact_metadata": (
                None
                if self.artifact_metadata is None
                else self.artifact_metadata.to_json_dict()
            ),
            "optimized_coilset_source_path": (
                None
                if self.optimized_coilset_source_path is None
                else os.fspath(self.optimized_coilset_source_path)
            ),
            "exported_biot_savart_path": (
                None
                if self.exported_biot_savart_path is None
                else os.fspath(self.exported_biot_savart_path)
            ),
            "import_report_path": (
                None if self.import_report_path is None else os.fspath(self.import_report_path)
            ),
        }


@dataclass(frozen=True, slots=True)
class DescOptimizedSimsoptExportArtifacts:
    exported_biot_savart_path: Path
    import_report_path: Path
    export_report_path: Path
    import_report: DescCoilImportReport
    report: DescOptimizedSimsoptExportReport


@dataclass(frozen=True, slots=True)
class _DescCurveSampleGrid:
    nodes: np.ndarray
    NFP: int = 1
    sym: bool = False


class DescOptimizedSimsoptExportError(RuntimeError):
    def __init__(self, report: DescOptimizedSimsoptExportReport) -> None:
        super().__init__(report.reason)
        self.report = report


@dataclass(frozen=True, slots=True)
class DescOptimizedSurfaceExportReport:
    status: DescOptimizedSurfaceExportStatus
    reason: str
    optimized_equilibrium_type: str | None
    optimized_equilibrium_source_path: Path | None
    exported_surface_path: Path | None
    nfp: int
    stellarator_symmetry: bool
    mpol: int
    ntor: int
    sample_count_phi: int
    sample_count_theta: int
    max_fit_residual_m: float | None
    max_fit_residual_threshold_m: float
    mean_fit_residual_m: float | None
    rms_fit_residual_m: float | None

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": "desc_optimized_surface_export_report_v1",
            "status": self.status,
            "reason": self.reason,
            "optimized_equilibrium_type": self.optimized_equilibrium_type,
            "optimized_equilibrium_source_path": (
                None
                if self.optimized_equilibrium_source_path is None
                else os.fspath(self.optimized_equilibrium_source_path)
            ),
            "exported_surface_path": (
                None
                if self.exported_surface_path is None
                else os.fspath(self.exported_surface_path)
            ),
            "nfp": self.nfp,
            "stellarator_symmetry": self.stellarator_symmetry,
            "mpol": self.mpol,
            "ntor": self.ntor,
            "sample_count_phi": self.sample_count_phi,
            "sample_count_theta": self.sample_count_theta,
            "max_fit_residual_m": self.max_fit_residual_m,
            "max_fit_residual_threshold_m": self.max_fit_residual_threshold_m,
            "mean_fit_residual_m": self.mean_fit_residual_m,
            "rms_fit_residual_m": self.rms_fit_residual_m,
        }


@dataclass(frozen=True, slots=True)
class DescOptimizedSurfaceExportArtifacts:
    exported_surface_path: Path
    export_report_path: Path
    report: DescOptimizedSurfaceExportReport


class DescOptimizedSurfaceExportError(RuntimeError):
    def __init__(self, report: DescOptimizedSurfaceExportReport) -> None:
        super().__init__(report.reason)
        self.report = report


def materialize_optimized_desc_equilibrium_surface_simsopt_export(
    *,
    optimized_equilibrium_path: Path | None,
    output_root: Path,
    desc_source_root: Path | None = None,
    nfp: int,
    stellarator_symmetry: bool,
    mpol: int,
    ntor: int,
) -> DescOptimizedSurfaceExportArtifacts:
    """Load a saved DESC Equilibrium and export its LCFS to SIMSOPT surface JSON."""

    output_root.mkdir(parents=True, exist_ok=True)
    export_report_path = output_root / "desc_optimized_surface_export_report.json"
    exported_surface_path = output_root / "surf_desc_equilibrium_export.json"
    source_path: Path | None = None
    sample_count_phi = _surface_sample_count_phi(ntor)
    sample_count_theta = _surface_sample_count_theta(mpol)
    try:
        _validate_surface_export_inputs(
            nfp=nfp,
            stellarator_symmetry=stellarator_symmetry,
            mpol=mpol,
            ntor=ntor,
        )
        if optimized_equilibrium_path is None:
            raise ValueError("DESC runtime solve did not record desc_equilibrium.h5.")
        source_path = optimized_equilibrium_path.expanduser().resolve()
        equilibrium = load_desc_optimized_equilibrium_artifact(
            source_path,
            desc_source_root=desc_source_root,
        )
        desc_surface = _equilibrium_surface(equilibrium)
        exported_surface, residuals = _fit_simsopt_surface_to_desc_surface(
            desc_surface,
            desc_source_root=desc_source_root,
            nfp=nfp,
            stellarator_symmetry=stellarator_symmetry,
            mpol=mpol,
            ntor=ntor,
            sample_count_phi=sample_count_phi,
            sample_count_theta=sample_count_theta,
        )
        max_fit_residual_m = float(np.max(residuals))
        if max_fit_residual_m > DESC_OPTIMIZED_SURFACE_MAX_FIT_RESIDUAL_M:
            raise ValueError(
                "SIMSOPT surface fit residual exceeds threshold: "
                f"max_fit_residual_m={max_fit_residual_m:.6g} > "
                f"{DESC_OPTIMIZED_SURFACE_MAX_FIT_RESIDUAL_M:.6g}."
            )
        exported_surface.save(os.fspath(exported_surface_path))
        report = DescOptimizedSurfaceExportReport(
            status="passed",
            reason="Optimized DESC Equilibrium LCFS exported to SIMSOPT surface.",
            optimized_equilibrium_type=_qualified_type_name(equilibrium),
            optimized_equilibrium_source_path=source_path,
            exported_surface_path=exported_surface_path.resolve(),
            nfp=nfp,
            stellarator_symmetry=stellarator_symmetry,
            mpol=mpol,
            ntor=ntor,
            sample_count_phi=sample_count_phi,
            sample_count_theta=sample_count_theta,
            max_fit_residual_m=max_fit_residual_m,
            max_fit_residual_threshold_m=DESC_OPTIMIZED_SURFACE_MAX_FIT_RESIDUAL_M,
            mean_fit_residual_m=float(np.mean(residuals)),
            rms_fit_residual_m=float(np.sqrt(np.mean(residuals * residuals))),
        )
        export_report_path.write_text(
            _json_dumps(report.to_json_dict()),
            encoding="utf-8",
        )
        return DescOptimizedSurfaceExportArtifacts(
            exported_surface_path=exported_surface_path,
            export_report_path=export_report_path,
            report=report,
        )
    except Exception as exc:
        report = DescOptimizedSurfaceExportReport(
            status="failed",
            reason=f"{type(exc).__name__}: {exc}",
            optimized_equilibrium_type=None,
            optimized_equilibrium_source_path=source_path,
            exported_surface_path=(
                exported_surface_path.resolve()
                if exported_surface_path.is_file()
                else None
            ),
            nfp=nfp,
            stellarator_symmetry=stellarator_symmetry,
            mpol=mpol,
            ntor=ntor,
            sample_count_phi=sample_count_phi,
            sample_count_theta=sample_count_theta,
            max_fit_residual_m=None,
            max_fit_residual_threshold_m=DESC_OPTIMIZED_SURFACE_MAX_FIT_RESIDUAL_M,
            mean_fit_residual_m=None,
            rms_fit_residual_m=None,
        )
        export_report_path.write_text(
            _json_dumps(report.to_json_dict()),
            encoding="utf-8",
        )
        raise DescOptimizedSurfaceExportError(report) from exc


def materialize_optimized_desc_coilset_simsopt_export(
    *,
    optimized_coilset: object,
    source_artifacts: Mapping[str, Path],
    coil_group_counts: Mapping[str, int],
    output_root: Path,
    sample_count: int,
    simsopt_fourier_order: int,
    optimized_coilset_source_path: Path | None = None,
) -> DescOptimizedSimsoptExportArtifacts:
    """Sample an optimized DESC CoilSet and save a loadable SIMSOPT BiotSavart."""

    output_root.mkdir(parents=True, exist_ok=True)
    export_report_path = output_root / "desc_optimized_simsopt_export_report.json"
    exported_biot_savart_path = output_root / "biot_savart_desc_export.json"
    import_report_path = output_root / "desc_coil_import_report.json"
    artifact_metadata: DescBridgeArtifactMetadata | None = None
    try:
        _validate_export_inputs(
            coil_group_counts=coil_group_counts,
            sample_count=sample_count,
            simsopt_fourier_order=simsopt_fourier_order,
        )
        sampled_coils = sample_desc_coilset_unique_coils(
            optimized_coilset,
            coil_group_counts=coil_group_counts,
            sample_count=sample_count,
        )
        artifact_metadata = DescBridgeArtifactMetadata(
            source_artifact_checksums=desc_bridge_source_checksums(
                source_artifacts,
            ),
        )
        import_result = import_desc_sampled_coils_to_simsopt(
            sampled_coils,
            simsopt_fourier_order=simsopt_fourier_order,
            sample_count=sample_count,
            coil_group_manifest=coil_group_counts,
            artifact_metadata=artifact_metadata,
            hardware_oracle_status="not_run",
            exported_artifact_paths=(os.fspath(exported_biot_savart_path),),
        )
        artifact_metadata = import_result.report.artifact_metadata
        import_result.biot_savart.save(os.fspath(exported_biot_savart_path))
        import_report_path.write_text(
            _json_dumps(import_result.report.to_json_dict()),
            encoding="utf-8",
        )
        report = DescOptimizedSimsoptExportReport(
            status="passed",
            reason="Optimized DESC CoilSet exported to loadable SIMSOPT BiotSavart.",
            optimized_coilset_type=_qualified_type_name(optimized_coilset),
            sample_count=sample_count,
            simsopt_fourier_order=simsopt_fourier_order,
            coil_group_counts=dict(coil_group_counts),
            artifact_metadata=artifact_metadata,
            optimized_coilset_source_path=(
                None
                if optimized_coilset_source_path is None
                else optimized_coilset_source_path.resolve()
            ),
            exported_biot_savart_path=exported_biot_savart_path.resolve(),
            import_report_path=import_report_path.resolve(),
        )
        export_report_path.write_text(
            _json_dumps(report.to_json_dict()),
            encoding="utf-8",
        )
        return DescOptimizedSimsoptExportArtifacts(
            exported_biot_savart_path=exported_biot_savart_path,
            import_report_path=import_report_path,
            export_report_path=export_report_path,
            import_report=import_result.report,
            report=report,
        )
    except Exception as exc:
        report = DescOptimizedSimsoptExportReport(
            status="failed",
            reason=f"{type(exc).__name__}: {exc}",
            optimized_coilset_type=(
                None if optimized_coilset is None else _qualified_type_name(optimized_coilset)
            ),
            sample_count=sample_count,
            simsopt_fourier_order=simsopt_fourier_order,
            coil_group_counts=dict(coil_group_counts),
            artifact_metadata=artifact_metadata,
            optimized_coilset_source_path=(
                None
                if optimized_coilset_source_path is None
                else optimized_coilset_source_path.resolve()
            ),
            exported_biot_savart_path=(
                exported_biot_savart_path.resolve()
                if exported_biot_savart_path.is_file()
                else None
            ),
            import_report_path=(
                import_report_path.resolve() if import_report_path.is_file() else None
            ),
        )
        export_report_path.write_text(
            _json_dumps(report.to_json_dict()),
            encoding="utf-8",
        )
        raise DescOptimizedSimsoptExportError(report) from exc


def materialize_optimized_desc_coil_artifact_simsopt_export(
    *,
    optimized_coilset_path: Path | None,
    source_artifacts: Mapping[str, Path],
    coil_group_counts: Mapping[str, int],
    output_root: Path,
    sample_count: int,
    simsopt_fourier_order: int,
    desc_source_root: Path | None = None,
) -> DescOptimizedSimsoptExportArtifacts:
    """Load a saved DESC CoilSet artifact and export it to SIMSOPT format."""

    output_root.mkdir(parents=True, exist_ok=True)
    export_report_path = output_root / "desc_optimized_simsopt_export_report.json"
    source_path: Path | None = None
    try:
        _validate_export_inputs(
            coil_group_counts=coil_group_counts,
            sample_count=sample_count,
            simsopt_fourier_order=simsopt_fourier_order,
        )
        if optimized_coilset_path is None:
            raise ValueError("DESC runtime solve did not record desc_coils.h5.")
        source_path = optimized_coilset_path.expanduser().resolve()
        optimized_coilset = load_desc_optimized_coilset_artifact(
            source_path,
            desc_source_root=desc_source_root,
        )
        return materialize_optimized_desc_coilset_simsopt_export(
            optimized_coilset=optimized_coilset,
            source_artifacts=source_artifacts,
            coil_group_counts=coil_group_counts,
            output_root=output_root,
            sample_count=sample_count,
            simsopt_fourier_order=simsopt_fourier_order,
            optimized_coilset_source_path=source_path,
        )
    except DescOptimizedSimsoptExportError:
        raise
    except Exception as exc:
        report = DescOptimizedSimsoptExportReport(
            status="failed",
            reason=f"{type(exc).__name__}: {exc}",
            optimized_coilset_type=None,
            sample_count=sample_count,
            simsopt_fourier_order=simsopt_fourier_order,
            coil_group_counts=dict(coil_group_counts),
            artifact_metadata=None,
            optimized_coilset_source_path=source_path,
            exported_biot_savart_path=None,
            import_report_path=None,
        )
        export_report_path.write_text(
            _json_dumps(report.to_json_dict()),
            encoding="utf-8",
        )
        raise DescOptimizedSimsoptExportError(report) from exc


def load_desc_optimized_coilset_artifact(
    optimized_coilset_path: Path,
    *,
    desc_source_root: Path | None = None,
) -> object:
    """Load a saved DESC CoilSet-like artifact through DESC runtime APIs."""

    path = optimized_coilset_path.expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"DESC optimized coil artifact does not exist: {path}.")
    with activate_desc_source_root(desc_source_root):
        from desc.io import load as desc_load

        optimized_coilset = desc_load(os.fspath(path))
    _unique_coils(optimized_coilset)
    return optimized_coilset


def load_desc_optimized_equilibrium_artifact(
    optimized_equilibrium_path: Path,
    *,
    desc_source_root: Path | None = None,
) -> object:
    """Load a saved DESC Equilibrium-like artifact through DESC runtime APIs."""

    path = optimized_equilibrium_path.expanduser().resolve()
    if not path.is_file():
        raise ValueError(
            f"DESC optimized equilibrium artifact does not exist: {path}."
        )
    with activate_desc_source_root(desc_source_root):
        from desc.io import load as desc_load

        equilibrium = desc_load(os.fspath(path))
    _equilibrium_surface(equilibrium)
    return equilibrium


def sample_desc_coilset_unique_coils(
    coilset: object,
    *,
    coil_group_counts: Mapping[str, int],
    sample_count: int,
) -> tuple[DescSampledCoil, ...]:
    """Sample unique DESC coils in the source coil-group order."""

    _validate_group_counts(coil_group_counts)
    if isinstance(sample_count, bool) or not isinstance(sample_count, int):
        raise ValueError("sample_count must be a positive integer.")
    if sample_count <= 0:
        raise ValueError("sample_count must be a positive integer.")
    coils = _unique_coils(coilset)
    expected_count = sum(coil_group_counts.values())
    if len(coils) != expected_count:
        raise ValueError(
            "optimized DESC CoilSet coil count does not match source groups: "
            f"observed {len(coils)}, expected {expected_count}."
        )
    sampled: list[DescSampledCoil] = []
    cursor = 0
    for group, count in coil_group_counts.items():
        for group_index in range(count):
            coil = coils[cursor]
            sampled.append(
                DescSampledCoil(
                    group=group,
                    group_index=group_index,
                    name=_coil_name(coil, group=group, group_index=group_index),
                    current_A=_coil_current_A(coil),
                    coords_xyz=_coil_coords_xyz(coil, sample_count=sample_count),
                )
            )
            cursor += 1
    return tuple(sampled)


def _unique_coils(coilset: object) -> tuple[object, ...]:
    coils = getattr(coilset, "coils", None)
    if isinstance(coils, Sequence) and not isinstance(coils, (str, bytes)):
        return tuple(coils)
    if isinstance(coilset, Sequence) and not isinstance(coilset, (str, bytes)):
        return tuple(coilset)
    raise TypeError("optimized DESC CoilSet must expose a sequence of unique coils.")


def _coil_coords_xyz(coil: object, *, sample_count: int) -> np.ndarray:
    compute_position = getattr(coil, "_compute_position", None)
    if not callable(compute_position):
        raise TypeError("optimized DESC coil does not expose _compute_position.")
    coords = np.asarray(
        compute_position(
            grid=_desc_curve_sample_grid(sample_count=sample_count),
            basis="xyz",
        ),
        dtype=float,
    )
    if coords.ndim == 3 and coords.shape[0] == 1 and coords.shape[2] == 3:
        coords = coords[0]
    if coords.shape != (sample_count, 3):
        raise ValueError(
            "optimized DESC coil samples must have shape "
            f"({sample_count}, 3); got {coords.shape}."
        )
    if not np.isfinite(coords).all():
        raise ValueError("optimized DESC coil samples must be finite.")
    return np.ascontiguousarray(coords)


def _desc_curve_sample_grid(*, sample_count: int) -> object:
    """Build a DESC curve grid with exactly sample_count periodic zeta nodes."""

    zeta = np.linspace(0.0, 2.0 * np.pi, sample_count, endpoint=False)
    try:
        from desc.grid import LinearGrid
    except ModuleNotFoundError as exc:
        if exc.name != "desc":
            raise
        nodes = np.column_stack(
            (np.zeros_like(zeta), np.zeros_like(zeta), zeta)
        )
        return _DescCurveSampleGrid(nodes=nodes)
    return LinearGrid(zeta=zeta, NFP=1, sym=False)


def _coil_current_A(coil: object) -> float:
    current = getattr(coil, "current", None)
    if current is None:
        raise TypeError("optimized DESC coil does not expose current.")
    value = float(np.asarray(current, dtype=float).reshape(()))
    if not np.isfinite(value):
        raise ValueError("optimized DESC coil current must be finite.")
    return value


def _coil_name(coil: object, *, group: str, group_index: int) -> str:
    name = getattr(coil, "name", "")
    if isinstance(name, str) and name != "":
        return name
    return f"{group}_{group_index:03d}"


def _validate_export_inputs(
    *,
    coil_group_counts: Mapping[str, int],
    sample_count: int,
    simsopt_fourier_order: int,
) -> None:
    _validate_group_counts(coil_group_counts)
    if (
        isinstance(simsopt_fourier_order, bool)
        or not isinstance(simsopt_fourier_order, int)
        or simsopt_fourier_order <= 0
    ):
        raise ValueError("simsopt_fourier_order must be a positive integer.")
    if isinstance(sample_count, bool) or not isinstance(sample_count, int):
        raise ValueError("sample_count must be a positive integer.")
    if sample_count <= 2 * simsopt_fourier_order + 1:
        raise ValueError(
            "sample_count must exceed 2 * simsopt_fourier_order + 1 for "
            "SIMSOPT Fourier fitting."
        )


def _validate_group_counts(coil_group_counts: Mapping[str, int]) -> None:
    if not coil_group_counts:
        raise ValueError("optimized DESC export requires explicit coil group counts.")
    for group, count in coil_group_counts.items():
        if not isinstance(group, str) or group == "":
            raise ValueError("optimized DESC export group names must be nonempty.")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(
                "optimized DESC export group counts must be nonnegative integers."
            )


def _fit_simsopt_surface_to_desc_surface(
    desc_surface: object,
    *,
    desc_source_root: Path | None,
    nfp: int,
    stellarator_symmetry: bool,
    mpol: int,
    ntor: int,
    sample_count_phi: int,
    sample_count_theta: int,
) -> tuple[object, np.ndarray]:
    with activate_desc_source_root(desc_source_root):
        from desc.grid import Grid
        from simsopt.geo import SurfaceXYZTensorFourier

        quadpoints_phi = np.linspace(
            0.0,
            1.0 / float(nfp),
            sample_count_phi,
            endpoint=False,
        )
        quadpoints_theta = np.linspace(0.0, 1.0, sample_count_theta, endpoint=False)
        desc_xyz = _sample_desc_surface_xyz(
            desc_surface,
            desc_grid_cls=Grid,
            quadpoints_phi=quadpoints_phi,
            quadpoints_theta=quadpoints_theta,
        )
        simsopt_surface = SurfaceXYZTensorFourier(
            nfp=nfp,
            stellsym=stellarator_symmetry,
            mpol=mpol,
            ntor=ntor,
            quadpoints_phi=quadpoints_phi,
            quadpoints_theta=quadpoints_theta,
        )
        simsopt_surface.least_squares_fit(desc_xyz)
        fit_xyz = np.asarray(simsopt_surface.gamma(), dtype=float)
    if fit_xyz.shape != desc_xyz.shape:
        raise ValueError(
            "SIMSOPT fitted surface gamma() shape does not match DESC samples: "
            f"{fit_xyz.shape} vs {desc_xyz.shape}."
        )
    residuals = np.linalg.norm(fit_xyz - desc_xyz, axis=2)
    if not np.isfinite(residuals).all():
        raise ValueError("SIMSOPT surface fit residuals must be finite.")
    return simsopt_surface, residuals


def _sample_desc_surface_xyz(
    desc_surface: object,
    *,
    desc_grid_cls: object,
    quadpoints_phi: np.ndarray,
    quadpoints_theta: np.ndarray,
) -> np.ndarray:
    zeta_rad = 2.0 * np.pi * quadpoints_phi
    theta_rad = 2.0 * np.pi * quadpoints_theta
    zeta_grid, theta_grid = np.meshgrid(zeta_rad, theta_rad, indexing="ij")
    nodes = np.column_stack(
        (
            np.ones(zeta_grid.size),
            theta_grid.reshape(-1),
            zeta_grid.reshape(-1),
        )
    )
    grid = desc_grid_cls(nodes, sort=False, jitable=True)
    compute = getattr(desc_surface, "compute", None)
    if not callable(compute):
        raise TypeError("DESC equilibrium surface must expose compute().")
    data = compute("x", grid=grid, basis="xyz")
    if not isinstance(data, Mapping):
        raise TypeError("DESC surface compute('x') must return a mapping.")
    desc_xyz = np.asarray(data.get("x"), dtype=float)
    expected_flat_shape = (zeta_grid.size, 3)
    if desc_xyz.shape != expected_flat_shape:
        raise ValueError(
            "DESC surface x samples have unexpected shape: "
            f"{desc_xyz.shape} vs {expected_flat_shape}."
        )
    if not np.isfinite(desc_xyz).all():
        raise ValueError("DESC surface x samples must be finite.")
    return np.ascontiguousarray(
        desc_xyz.reshape((quadpoints_phi.size, quadpoints_theta.size, 3))
    )


def _equilibrium_surface(equilibrium: object) -> object:
    surface = getattr(equilibrium, "surface", None)
    if surface is None:
        raise TypeError("optimized DESC Equilibrium must expose .surface.")
    return surface


def _surface_sample_count_phi(ntor: int) -> int:
    return max(2 * max(ntor, 0) + 1, 3)


def _surface_sample_count_theta(mpol: int) -> int:
    return max(2 * max(mpol, 1) + 1, 9)


def _validate_surface_export_inputs(
    *,
    nfp: int,
    stellarator_symmetry: bool,
    mpol: int,
    ntor: int,
) -> None:
    if isinstance(nfp, bool) or not isinstance(nfp, int) or nfp <= 0:
        raise ValueError("nfp must be a positive integer.")
    if not isinstance(stellarator_symmetry, bool):
        raise ValueError("stellarator_symmetry must be boolean.")
    if isinstance(mpol, bool) or not isinstance(mpol, int) or mpol < 0:
        raise ValueError("mpol must be a nonnegative integer.")
    if isinstance(ntor, bool) or not isinstance(ntor, int) or ntor < 0:
        raise ValueError("ntor must be a nonnegative integer.")


def _qualified_type_name(value: object) -> str:
    return f"{type(value).__module__}.{type(value).__qualname__}"


def _json_dumps(payload: Mapping[str, object]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)


__all__ = [
    "DescOptimizedSimsoptExportArtifacts",
    "DescOptimizedSimsoptExportError",
    "DescOptimizedSimsoptExportReport",
    "DescOptimizedSurfaceExportArtifacts",
    "DescOptimizedSurfaceExportError",
    "DescOptimizedSurfaceExportReport",
    "load_desc_optimized_coilset_artifact",
    "load_desc_optimized_equilibrium_artifact",
    "materialize_optimized_desc_coil_artifact_simsopt_export",
    "materialize_optimized_desc_coilset_simsopt_export",
    "materialize_optimized_desc_equilibrium_surface_simsopt_export",
    "sample_desc_coilset_unique_coils",
]
