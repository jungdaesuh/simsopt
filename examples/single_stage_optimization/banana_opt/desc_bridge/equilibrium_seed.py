"""Explicit equilibrium seed provenance contract for DESC joint runs."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np

from banana_opt.desc_bridge.runtime_imports import activate_desc_source_root

DESC_EQUILIBRIUM_SEED_SCHEMA_VERSION = "desc_equilibrium_seed_v1"

EquilibriumSeedKind = Literal["vmec_wout", "desc_h5", "simsopt_surface"]
_EQUILIBRIUM_SEED_KINDS = frozenset({"vmec_wout", "desc_h5", "simsopt_surface"})


@dataclass(frozen=True, slots=True)
class DescEquilibriumSeedSpec:
    spec_path: Path
    source_kind: EquilibriumSeedKind
    source_path: Path
    nfp: int
    stellarator_symmetry: bool
    handedness: str
    angular_convention: str
    major_radius_m: float
    minor_radius_m: float
    lcfs_mpol: int
    lcfs_ntor: int
    target_lcfs_G: float | None = None

    def to_input_contract(self) -> dict[str, object]:
        return {
            "schema_version": DESC_EQUILIBRIUM_SEED_SCHEMA_VERSION,
            "spec_path": os.fspath(self.spec_path),
            "source_kind": self.source_kind,
            "source_path": os.fspath(self.source_path),
            "nfp": self.nfp,
            "stellarator_symmetry": self.stellarator_symmetry,
            "handedness": self.handedness,
            "angular_convention": self.angular_convention,
            "major_radius_m": self.major_radius_m,
            "minor_radius_m": self.minor_radius_m,
            "lcfs_mpol": self.lcfs_mpol,
            "lcfs_ntor": self.lcfs_ntor,
            "target_lcfs_G": self.target_lcfs_G,
        }

    def with_target_lcfs_G(self, target_lcfs_G: float) -> "DescEquilibriumSeedSpec":
        return replace(
            self,
            target_lcfs_G=_finite_float_value(
                target_lcfs_G,
                field_name="target_lcfs_G",
            ),
        )

    def with_lcfs_resolution(
        self,
        *,
        lcfs_mpol: int,
        lcfs_ntor: int,
    ) -> "DescEquilibriumSeedSpec":
        return replace(
            self,
            lcfs_mpol=_nonnegative_int_value(lcfs_mpol, field_name="lcfs_mpol"),
            lcfs_ntor=_nonnegative_int_value(lcfs_ntor, field_name="lcfs_ntor"),
        )


@dataclass(frozen=True, slots=True)
class DescLcfsParityReport:
    source_surface_type: str
    desc_surface_type: str
    sample_count_theta: int
    sample_count_phi: int
    comparison_sample_count: int
    max_source_parameter_phi_delta_rad: float
    max_xyz_delta_m: float
    mean_xyz_delta_m: float
    rms_xyz_delta_m: float

    def to_json_dict(self) -> dict[str, object]:
        return {
            "source_surface_type": self.source_surface_type,
            "desc_surface_type": self.desc_surface_type,
            "sample_count_theta": self.sample_count_theta,
            "sample_count_phi": self.sample_count_phi,
            "comparison_sample_count": self.comparison_sample_count,
            "max_source_parameter_phi_delta_rad": (
                self.max_source_parameter_phi_delta_rad
            ),
            "max_xyz_delta_m": self.max_xyz_delta_m,
            "mean_xyz_delta_m": self.mean_xyz_delta_m,
            "rms_xyz_delta_m": self.rms_xyz_delta_m,
        }


@dataclass(frozen=True, slots=True)
class DescLcfsGScaleReport:
    """Records the Psi scale used to match a SIMSOPT seed's Boozer LCFS G."""

    target_lcfs_G: float
    unscaled_lcfs_G: float
    scaled_lcfs_G: float
    psi_before_Wb: float
    psi_after_Wb: float
    psi_scale_factor: float
    relative_G_error: float

    def to_json_dict(self) -> dict[str, float]:
        return {
            "target_lcfs_G": self.target_lcfs_G,
            "unscaled_lcfs_G": self.unscaled_lcfs_G,
            "scaled_lcfs_G": self.scaled_lcfs_G,
            "psi_before_Wb": self.psi_before_Wb,
            "psi_after_Wb": self.psi_after_Wb,
            "psi_scale_factor": self.psi_scale_factor,
            "relative_G_error": self.relative_G_error,
        }


@dataclass(frozen=True, slots=True)
class _SimsoptLcfsSamples:
    coords_rpz: np.ndarray
    theta_rad: np.ndarray
    zeta_rad: np.ndarray
    source_xyz: np.ndarray
    sample_count_theta: int
    sample_count_phi: int
    max_source_parameter_phi_delta_rad: float


@dataclass(frozen=True, slots=True)
class DescEquilibriumRuntimeLoadReport:
    spec_path: Path
    source_kind: EquilibriumSeedKind
    source_path: Path
    loader: str
    status: Literal["passed", "failed"]
    reason: str
    desc_source_root: Path | None
    desc_version: str | None
    equilibrium_type: str | None
    requested_resolution: Mapping[str, int | str]
    lcfs_parity: DescLcfsParityReport | None
    lcfs_G_scaling: DescLcfsGScaleReport | None

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": "desc_equilibrium_runtime_load_report_v1",
            "spec_path": os.fspath(self.spec_path),
            "source_kind": self.source_kind,
            "source_path": os.fspath(self.source_path),
            "loader": self.loader,
            "status": self.status,
            "reason": self.reason,
            "desc_source_root": (
                None if self.desc_source_root is None else os.fspath(self.desc_source_root)
            ),
            "desc_version": self.desc_version,
            "equilibrium_type": self.equilibrium_type,
            "requested_resolution": dict(self.requested_resolution),
            "lcfs_parity": (
                None if self.lcfs_parity is None else self.lcfs_parity.to_json_dict()
            ),
            "lcfs_G_scaling": (
                None
                if self.lcfs_G_scaling is None
                else self.lcfs_G_scaling.to_json_dict()
            ),
        }


@dataclass(frozen=True, slots=True)
class DescLoadedEquilibriumSeed:
    equilibrium: object
    report: DescEquilibriumRuntimeLoadReport


class DescEquilibriumRuntimeLoadError(RuntimeError):
    def __init__(self, report: DescEquilibriumRuntimeLoadReport) -> None:
        super().__init__(report.reason)
        self.report = report


def load_desc_equilibrium_seed_spec(path: str | Path) -> DescEquilibriumSeedSpec:
    spec_path = Path(path).expanduser().resolve()
    payload = _read_json_mapping(spec_path)
    if payload.get("schema_version") != DESC_EQUILIBRIUM_SEED_SCHEMA_VERSION:
        raise ValueError("Unexpected DESC equilibrium seed schema_version.")
    source_kind = _coerce_source_kind(payload.get("source_kind"))
    target_lcfs_G = _optional_finite_float(payload, "target_lcfs_G")
    if target_lcfs_G is not None and source_kind != "simsopt_surface":
        raise ValueError(
            "DESC equilibrium seed target_lcfs_G is only supported for "
            "source_kind='simsopt_surface'."
        )
    source_path = _require_existing_source_path(spec_path, payload.get("source_path"))
    return DescEquilibriumSeedSpec(
        spec_path=spec_path,
        source_kind=source_kind,
        source_path=source_path,
        nfp=_positive_int(payload, "nfp"),
        stellarator_symmetry=_require_bool(payload, "stellarator_symmetry"),
        handedness=_nonempty_string(payload, "handedness"),
        angular_convention=_nonempty_string(payload, "angular_convention"),
        major_radius_m=_positive_float(payload, "major_radius_m"),
        minor_radius_m=_positive_float(payload, "minor_radius_m"),
        lcfs_mpol=_nonnegative_int(payload, "lcfs_mpol"),
        lcfs_ntor=_nonnegative_int(payload, "lcfs_ntor"),
        target_lcfs_G=target_lcfs_G,
    )


def load_desc_equilibrium_seed_runtime(
    spec: DescEquilibriumSeedSpec,
    *,
    desc_source_root: Path | None = None,
) -> DescLoadedEquilibriumSeed:
    loader = _runtime_loader_name(spec.source_kind)
    requested_resolution = _runtime_requested_resolution(spec)
    try:
        lcfs_parity: DescLcfsParityReport | None = None
        lcfs_G_scaling: DescLcfsGScaleReport | None = None
        with activate_desc_source_root(desc_source_root):
            if spec.source_kind == "desc_h5":
                import desc
                from desc.io import load as desc_load

                equilibrium = desc_load(os.fspath(spec.source_path))
                _apply_loaded_desc_h5_resolution(equilibrium, spec=spec)
            elif spec.source_kind == "vmec_wout":
                import desc
                from desc.vmec import VMECIO

                equilibrium = VMECIO.load(
                    os.fspath(spec.source_path),
                    L=max(1, spec.lcfs_mpol),
                    M=spec.lcfs_mpol,
                    N=spec.lcfs_ntor,
                    spectral_indexing="ansi",
                    profile="iota",
                )
                lcfs_parity = _compute_vmec_lcfs_parity(
                    equilibrium,
                    spec=spec,
                    vmec_io_cls=VMECIO,
                )
            else:
                import desc
                from desc.equilibrium import Equilibrium
                from desc.geometry import FourierRZToroidalSurface
                from desc.grid import Grid, LinearGrid
                from simsopt import load as simsopt_load

                simsopt_surface = _load_simsopt_surface(
                    spec.source_path,
                    simsopt_load=simsopt_load,
                )
                desc_surface, lcfs_parity = _fit_desc_lcfs_from_simsopt_surface(
                    simsopt_surface,
                    spec=spec,
                    desc_fourier_rz_surface_cls=FourierRZToroidalSurface,
                    desc_grid_cls=Grid,
                )
                equilibrium = Equilibrium(
                    surface=desc_surface,
                    NFP=spec.nfp,
                    **_desc_equilibrium_resolution(spec),
                    sym=spec.stellarator_symmetry,
                    spectral_indexing="ansi",
                    check_orientation=False,
                    ensure_nested=False,
                )
                if spec.target_lcfs_G is not None:
                    lcfs_G_scaling = _scale_equilibrium_psi_to_target_lcfs_G(
                        equilibrium,
                        target_lcfs_G=spec.target_lcfs_G,
                        desc_linear_grid_cls=LinearGrid,
                    )
            report = _runtime_load_report(
                spec=spec,
                loader=loader,
                status="passed",
                reason="DESC equilibrium seed loaded.",
                desc_source_root=desc_source_root,
                desc_version=_desc_version(desc),
                equilibrium=equilibrium,
                requested_resolution=requested_resolution,
                lcfs_parity=lcfs_parity,
                lcfs_G_scaling=lcfs_G_scaling,
            )
            return DescLoadedEquilibriumSeed(equilibrium=equilibrium, report=report)
    except Exception as exc:
        report = _runtime_load_report(
            spec=spec,
            loader=loader,
            status="failed",
            reason=f"{type(exc).__name__}: {exc}",
            desc_source_root=desc_source_root,
            desc_version=None,
            equilibrium=None,
            requested_resolution=requested_resolution,
            lcfs_parity=None,
            lcfs_G_scaling=None,
        )
        raise DescEquilibriumRuntimeLoadError(report) from exc


def _apply_loaded_desc_h5_resolution(
    equilibrium: object,
    *,
    spec: DescEquilibriumSeedSpec,
) -> None:
    resolution = _desc_equilibrium_resolution(spec)
    change_resolution = getattr(equilibrium, "change_resolution", None)
    if not callable(change_resolution):
        raise TypeError("DESC desc_h5 equilibrium must expose change_resolution().")
    current_L = _optional_int_attr(equilibrium, "L")
    if current_L is not None and resolution["L"] < current_L:
        get_surface_at = getattr(equilibrium, "get_surface_at", None)
        if not callable(get_surface_at):
            raise TypeError(
                "DESC desc_h5 equilibrium must expose get_surface_at() when "
                "reducing radial resolution."
            )
        equilibrium.surface = get_surface_at(rho=1.0)
    change_resolution(**resolution)


def _scale_equilibrium_psi_to_target_lcfs_G(
    equilibrium: object,
    *,
    target_lcfs_G: float,
    desc_linear_grid_cls: object,
) -> DescLcfsGScaleReport:
    target = _finite_float_value(target_lcfs_G, field_name="target_lcfs_G")
    unscaled_lcfs_G = _compute_equilibrium_lcfs_G(
        equilibrium,
        desc_linear_grid_cls=desc_linear_grid_cls,
    )
    psi_before = _finite_float_array_scalar(
        getattr(equilibrium, "Psi"),
        field_name="Psi",
    )
    if unscaled_lcfs_G == 0.0:
        raise ValueError("DESC equilibrium unscaled LCFS G must be nonzero.")
    scale_factor = target / unscaled_lcfs_G
    if not math.isfinite(scale_factor) or scale_factor <= 0.0:
        raise ValueError(
            "DESC equilibrium LCFS G scaling requires a finite positive Psi scale "
            f"factor; got {scale_factor!r}."
        )
    equilibrium.Psi = psi_before * scale_factor
    psi_after = _finite_float_array_scalar(
        getattr(equilibrium, "Psi"),
        field_name="Psi",
    )
    scaled_lcfs_G = _compute_equilibrium_lcfs_G(
        equilibrium,
        desc_linear_grid_cls=desc_linear_grid_cls,
    )
    relative_G_error = abs(scaled_lcfs_G - target) / max(abs(target), 1.0)
    return DescLcfsGScaleReport(
        target_lcfs_G=target,
        unscaled_lcfs_G=unscaled_lcfs_G,
        scaled_lcfs_G=scaled_lcfs_G,
        psi_before_Wb=psi_before,
        psi_after_Wb=psi_after,
        psi_scale_factor=scale_factor,
        relative_G_error=relative_G_error,
    )


def _compute_equilibrium_lcfs_G(
    equilibrium: object,
    *,
    desc_linear_grid_cls: object,
) -> float:
    grid = desc_linear_grid_cls(
        rho=1.0,
        M=0,
        N=0,
        NFP=getattr(equilibrium, "NFP"),
        sym=getattr(equilibrium, "sym"),
    )
    data = equilibrium.compute("G", grid=grid)
    if not isinstance(data, Mapping):
        raise TypeError("DESC equilibrium compute('G') must return a mapping.")
    return _finite_float_array_scalar(data.get("G"), field_name="G")


def _load_simsopt_surface(
    path: Path,
    *,
    simsopt_load: Callable[[str], object],
) -> object:
    loaded = simsopt_load(str(path))
    surface = getattr(loaded, "surface", loaded)
    for method_name in ("gamma",):
        if not callable(getattr(surface, method_name, None)):
            raise TypeError(f"SIMSOPT artifact must expose surface.{method_name}().")
    for attr_name in ("quadpoints_phi", "quadpoints_theta"):
        if getattr(surface, attr_name, None) is None:
            raise TypeError(f"SIMSOPT surface must expose {attr_name}.")
    return surface


@dataclass(frozen=True, slots=True)
class _DescLcfsParameterSamples:
    theta_rad: np.ndarray
    zeta_rad: np.ndarray
    sample_count_theta: int
    sample_count_phi: int


def _fit_desc_lcfs_from_simsopt_surface(
    simsopt_surface: object,
    *,
    spec: DescEquilibriumSeedSpec,
    desc_fourier_rz_surface_cls: object,
    desc_grid_cls: object,
) -> tuple[object, DescLcfsParityReport]:
    _validate_simsopt_surface_metadata(simsopt_surface, spec=spec)
    samples = _sample_simsopt_lcfs(simsopt_surface)
    desc_surface = desc_fourier_rz_surface_cls.from_values(
        samples.coords_rpz,
        samples.theta_rad,
        M=spec.lcfs_mpol,
        N=spec.lcfs_ntor,
        NFP=spec.nfp,
        sym=spec.stellarator_symmetry,
        check_orientation=False,
    )
    desc_xyz = _compute_desc_surface_xyz(
        desc_surface,
        desc_grid_cls=desc_grid_cls,
        theta_rad=samples.theta_rad,
        zeta_rad=samples.zeta_rad,
    )
    source_xyz = samples.source_xyz
    deltas = np.linalg.norm(desc_xyz - source_xyz, axis=1)
    return (
        desc_surface,
        DescLcfsParityReport(
            source_surface_type=_qualified_type_name(simsopt_surface),
            desc_surface_type=_qualified_type_name(desc_surface),
            sample_count_theta=samples.sample_count_theta,
            sample_count_phi=samples.sample_count_phi,
            comparison_sample_count=int(deltas.size),
            max_source_parameter_phi_delta_rad=(
                samples.max_source_parameter_phi_delta_rad
            ),
            max_xyz_delta_m=float(np.max(deltas)),
            mean_xyz_delta_m=float(np.mean(deltas)),
            rms_xyz_delta_m=float(np.sqrt(np.mean(deltas * deltas))),
        ),
    )


def _compute_vmec_lcfs_parity(
    equilibrium: object,
    *,
    spec: DescEquilibriumSeedSpec,
    vmec_io_cls: object,
) -> DescLcfsParityReport | None:
    compute_coord_surfaces = getattr(vmec_io_cls, "compute_coord_surfaces", None)
    if not callable(compute_coord_surfaces):
        return None
    sample_count_theta = _desc_lcfs_sample_count_theta(spec)
    sample_count_phi = _desc_lcfs_sample_count_phi(spec)
    coords = compute_coord_surfaces(
        equilibrium,
        os.fspath(spec.source_path),
        Nr=2,
        Nt=2,
        Nz=sample_count_phi,
        num_theta=sample_count_theta,
    )
    desc_r = _as_finite_float_array(coords["Rr_desc"][-1], field_name="Rr_desc")
    desc_z = _as_finite_float_array(coords["Zr_desc"][-1], field_name="Zr_desc")
    source_r = _as_finite_float_array(coords["Rr_vmec"][-1], field_name="Rr_vmec")
    source_z = _as_finite_float_array(coords["Zr_vmec"][-1], field_name="Zr_vmec")
    expected_shape = (sample_count_theta, sample_count_phi)
    for field_name, values in (
        ("Rr_desc", desc_r),
        ("Zr_desc", desc_z),
        ("Rr_vmec", source_r),
        ("Zr_vmec", source_z),
    ):
        if values.shape != expected_shape:
            raise ValueError(
                f"VMEC LCFS parity field {field_name} has unexpected shape: "
                f"{values.shape} vs {expected_shape}."
            )
    deltas = np.sqrt((desc_r - source_r) ** 2 + (desc_z - source_z) ** 2)
    desc_surface = getattr(equilibrium, "surface", equilibrium)
    return DescLcfsParityReport(
        source_surface_type="desc.vmec.VMECIO.compute_coord_surfaces:Rr_vmec/Zr_vmec",
        desc_surface_type=_qualified_type_name(desc_surface),
        sample_count_theta=sample_count_theta,
        sample_count_phi=sample_count_phi,
        comparison_sample_count=int(deltas.size),
        max_source_parameter_phi_delta_rad=0.0,
        max_xyz_delta_m=float(np.max(deltas)),
        mean_xyz_delta_m=float(np.mean(deltas)),
        rms_xyz_delta_m=float(np.sqrt(np.mean(deltas * deltas))),
    )


def _sample_desc_lcfs_parameters(
    spec: DescEquilibriumSeedSpec,
) -> _DescLcfsParameterSamples:
    sample_count_theta = _desc_lcfs_sample_count_theta(spec)
    sample_count_phi = _desc_lcfs_sample_count_phi(spec)
    theta = np.linspace(0.0, 2.0 * np.pi, sample_count_theta, endpoint=False)
    zeta = np.linspace(
        0.0,
        2.0 * np.pi / spec.nfp,
        sample_count_phi,
        endpoint=False,
    )
    zeta_grid, theta_grid = np.meshgrid(zeta, theta, indexing="ij")
    return _DescLcfsParameterSamples(
        theta_rad=theta_grid.reshape(-1),
        zeta_rad=zeta_grid.reshape(-1),
        sample_count_theta=sample_count_theta,
        sample_count_phi=sample_count_phi,
    )


def _desc_lcfs_sample_count_theta(spec: DescEquilibriumSeedSpec) -> int:
    return max(2 * max(1, spec.lcfs_mpol) + 1, 9)


def _desc_lcfs_sample_count_phi(spec: DescEquilibriumSeedSpec) -> int:
    return max(2 * spec.lcfs_ntor + 1, 1)


def _as_finite_float_array(value: object, *, field_name: str) -> np.ndarray:
    if hasattr(value, "filled"):
        value = value.filled(np.nan)
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"VMEC LCFS parity field {field_name} must be finite.")
    return array


def _validate_simsopt_surface_metadata(
    simsopt_surface: object,
    *,
    spec: DescEquilibriumSeedSpec,
) -> None:
    source_nfp = _optional_int_attr(simsopt_surface, "nfp")
    if source_nfp is None:
        source_nfp = _optional_int_attr(simsopt_surface, "NFP")
    if source_nfp is not None and source_nfp != spec.nfp:
        raise ValueError(
            "SIMSOPT surface NFP does not match equilibrium seed spec: "
            f"{source_nfp} vs {spec.nfp}."
        )
    source_symmetry = _optional_bool_attr(simsopt_surface, "stellsym")
    if source_symmetry is None:
        source_symmetry = _optional_bool_attr(simsopt_surface, "sym")
    if (
        source_symmetry is not None
        and source_symmetry != spec.stellarator_symmetry
    ):
        raise ValueError(
            "SIMSOPT surface stellarator symmetry does not match equilibrium seed "
            f"spec: {source_symmetry} vs {spec.stellarator_symmetry}."
        )


def _sample_simsopt_lcfs(simsopt_surface: object) -> _SimsoptLcfsSamples:
    source_xyz_grid = np.asarray(simsopt_surface.gamma(), dtype=float)
    phi = np.asarray(simsopt_surface.quadpoints_phi, dtype=float)
    theta = np.asarray(simsopt_surface.quadpoints_theta, dtype=float)
    expected_shape = (phi.size, theta.size, 3)
    if source_xyz_grid.shape != expected_shape:
        raise ValueError(
            "SIMSOPT surface gamma() shape does not match quadpoint metadata: "
            f"{source_xyz_grid.shape} vs {expected_shape}."
        )
    source_xyz = source_xyz_grid.reshape((-1, 3))
    theta_rad_grid = np.broadcast_to(
        (2.0 * np.pi * theta)[None, :],
        source_xyz_grid.shape[:2],
    )
    source_parameter_phi_rad_grid = np.broadcast_to(
        (2.0 * np.pi * phi)[:, None],
        source_xyz_grid.shape[:2],
    )
    raw_cylindrical_phi_rad_grid = np.mod(
        np.arctan2(source_xyz_grid[:, :, 1], source_xyz_grid[:, :, 0]),
        2.0 * np.pi,
    )
    cylindrical_phi_rad_grid = source_parameter_phi_rad_grid + _wrapped_angle_delta(
        raw_cylindrical_phi_rad_grid,
        source_parameter_phi_rad_grid,
    )
    radial_grid = np.linalg.norm(source_xyz_grid[:, :, :2], axis=2)
    coords_rpz = np.column_stack(
        (
            radial_grid.reshape(-1),
            cylindrical_phi_rad_grid.reshape(-1),
            source_xyz_grid[:, :, 2].reshape(-1),
        )
    )
    return _SimsoptLcfsSamples(
        coords_rpz=coords_rpz,
        theta_rad=theta_rad_grid.reshape(-1),
        zeta_rad=cylindrical_phi_rad_grid.reshape(-1),
        source_xyz=source_xyz,
        sample_count_theta=int(theta.size),
        sample_count_phi=int(phi.size),
        max_source_parameter_phi_delta_rad=float(
            np.max(
                np.abs(
                    cylindrical_phi_rad_grid - source_parameter_phi_rad_grid
                )
            )
        ),
    )


def _compute_desc_surface_xyz(
    desc_surface: object,
    *,
    desc_grid_cls: object,
    theta_rad: np.ndarray,
    zeta_rad: np.ndarray,
) -> np.ndarray:
    nodes = np.column_stack((np.ones_like(theta_rad), theta_rad, zeta_rad))
    grid = desc_grid_cls(nodes, sort=False, jitable=True)
    data = desc_surface.compute("x", grid=grid, basis="xyz")
    if not isinstance(data, Mapping):
        raise TypeError("DESC surface compute('x') must return a mapping.")
    desc_xyz = np.asarray(data.get("x"), dtype=float)
    if desc_xyz.shape != (theta_rad.size, 3):
        raise ValueError(
            "DESC surface x samples have unexpected shape: "
            f"{desc_xyz.shape} vs {(theta_rad.size, 3)}."
        )
    if not np.all(np.isfinite(desc_xyz)):
        raise ValueError("DESC surface x samples must be finite.")
    return desc_xyz


def _wrapped_angle_delta(value: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return (value - reference + np.pi) % (2.0 * np.pi) - np.pi


def _optional_int_attr(obj: object, attr_name: str) -> int | None:
    value = getattr(obj, attr_name, None)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"SIMSOPT surface attribute {attr_name} must be an integer.")
    return value


def _optional_bool_attr(obj: object, attr_name: str) -> bool | None:
    value = getattr(obj, attr_name, None)
    if value is None:
        return None
    if not isinstance(value, bool):
        raise TypeError(f"SIMSOPT surface attribute {attr_name} must be boolean.")
    return value


def _qualified_type_name(obj: object) -> str:
    return f"{type(obj).__module__}.{type(obj).__qualname__}"


def _read_json_mapping(path: Path) -> Mapping[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(
            f"DESC equilibrium seed spec must be a JSON object; got "
            f"{type(payload).__name__}."
        )
    return payload


def _coerce_source_kind(raw_kind: object) -> EquilibriumSeedKind:
    if raw_kind == "vmec_wout":
        return "vmec_wout"
    if raw_kind == "desc_h5":
        return "desc_h5"
    if raw_kind == "simsopt_surface":
        return "simsopt_surface"
    choices = ", ".join(sorted(_EQUILIBRIUM_SEED_KINDS))
    raise ValueError(
        f"DESC equilibrium seed source_kind must be one of {{{choices}}}; "
        f"got {raw_kind!r}."
    )


def _require_existing_source_path(spec_path: Path, raw_path: object) -> Path:
    if not isinstance(raw_path, str) or raw_path == "":
        raise ValueError("DESC equilibrium seed source_path must be a nonempty string.")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = spec_path.parent / path
    resolved = path.resolve()
    if not resolved.exists():
        raise ValueError(f"DESC equilibrium seed source_path does not exist: {resolved}.")
    if not resolved.is_file():
        raise ValueError(f"DESC equilibrium seed source_path must be a file: {resolved}.")
    return resolved


def _nonempty_string(payload: Mapping[str, object], field_name: str) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str) or value == "":
        raise ValueError(f"DESC equilibrium seed field {field_name!r} must be nonempty.")
    return value


def _require_bool(payload: Mapping[str, object], field_name: str) -> bool:
    value = payload.get(field_name)
    if not isinstance(value, bool):
        raise ValueError(f"DESC equilibrium seed field {field_name!r} must be boolean.")
    return value


def _positive_float(payload: Mapping[str, object], field_name: str) -> float:
    raw_value = payload.get(field_name)
    if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
        raise ValueError(f"DESC equilibrium seed field {field_name!r} must be numeric.")
    value = float(raw_value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(
            f"DESC equilibrium seed field {field_name!r} must be finite and positive."
        )
    return value


def _optional_finite_float(
    payload: Mapping[str, object],
    field_name: str,
) -> float | None:
    raw_value = payload.get(field_name)
    if raw_value is None:
        return None
    return _finite_float_value(raw_value, field_name=field_name)


def _finite_float_value(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"DESC equilibrium seed field {field_name!r} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"DESC equilibrium seed field {field_name!r} must be finite.")
    return result


def _finite_float_array_scalar(value: object, *, field_name: str) -> float:
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size != 1:
        raise ValueError(
            f"DESC equilibrium computed field {field_name!r} must be scalar; "
            f"got {array.size} values."
        )
    return _finite_float_value(array[0], field_name=field_name)


def _positive_int(payload: Mapping[str, object], field_name: str) -> int:
    raw_value = payload.get(field_name)
    if isinstance(raw_value, bool) or not isinstance(raw_value, int):
        raise ValueError(f"DESC equilibrium seed field {field_name!r} must be an integer.")
    value = raw_value
    if value <= 0:
        raise ValueError(f"DESC equilibrium seed field {field_name!r} must be positive.")
    return value


def _nonnegative_int(payload: Mapping[str, object], field_name: str) -> int:
    raw_value = payload.get(field_name)
    return _nonnegative_int_value(raw_value, field_name=field_name)


def _nonnegative_int_value(value: object, *, field_name: str) -> int:
    raw_value = value
    if isinstance(raw_value, bool) or not isinstance(raw_value, int):
        raise ValueError(f"DESC equilibrium seed field {field_name!r} must be an integer.")
    value = raw_value
    if value < 0:
        raise ValueError(
            f"DESC equilibrium seed field {field_name!r} must be nonnegative."
        )
    return value


def _runtime_loader_name(source_kind: EquilibriumSeedKind) -> str:
    if source_kind == "desc_h5":
        return "desc.io.load"
    if source_kind == "vmec_wout":
        return "desc.vmec.VMECIO.load"
    return "simsopt_surface_to_desc_equilibrium"


def _runtime_requested_resolution(
    spec: DescEquilibriumSeedSpec,
) -> Mapping[str, int | str]:
    if spec.source_kind == "desc_h5":
        return _desc_equilibrium_resolution(spec)
    if spec.source_kind == "vmec_wout":
        return {
            "L": max(1, spec.lcfs_mpol),
            "M": spec.lcfs_mpol,
            "N": spec.lcfs_ntor,
            "spectral_indexing": "ansi",
            "profile": "iota",
        }
    return {
        "lcfs_mpol": spec.lcfs_mpol,
        "lcfs_ntor": spec.lcfs_ntor,
    }


def _desc_equilibrium_resolution(spec: DescEquilibriumSeedSpec) -> dict[str, int]:
    radial_order = max(1, spec.lcfs_mpol)
    toroidal_grid_order = max(1, spec.lcfs_ntor)
    return {
        "L": radial_order,
        "M": spec.lcfs_mpol,
        "N": spec.lcfs_ntor,
        "L_grid": 2 * radial_order,
        "M_grid": 2 * radial_order,
        "N_grid": 2 * toroidal_grid_order,
    }


def _runtime_load_report(
    *,
    spec: DescEquilibriumSeedSpec,
    loader: str,
    status: Literal["passed", "failed"],
    reason: str,
    desc_source_root: Path | None,
    desc_version: str | None,
    equilibrium: object | None,
    requested_resolution: Mapping[str, int | str],
    lcfs_parity: DescLcfsParityReport | None,
    lcfs_G_scaling: DescLcfsGScaleReport | None,
) -> DescEquilibriumRuntimeLoadReport:
    return DescEquilibriumRuntimeLoadReport(
        spec_path=spec.spec_path,
        source_kind=spec.source_kind,
        source_path=spec.source_path,
        loader=loader,
        status=status,
        reason=reason,
        desc_source_root=None if desc_source_root is None else desc_source_root.resolve(),
        desc_version=desc_version,
        equilibrium_type=(
            None
            if equilibrium is None
            else f"{type(equilibrium).__module__}.{type(equilibrium).__qualname__}"
        ),
        requested_resolution=requested_resolution,
        lcfs_parity=lcfs_parity,
        lcfs_G_scaling=lcfs_G_scaling,
    )


def _desc_version(desc_module: object) -> str | None:
    version = getattr(desc_module, "__version__", None)
    if isinstance(version, str) and version != "":
        return version
    return None


__all__ = [
    "DESC_EQUILIBRIUM_SEED_SCHEMA_VERSION",
    "DescEquilibriumRuntimeLoadError",
    "DescEquilibriumRuntimeLoadReport",
    "DescEquilibriumSeedSpec",
    "DescLcfsGScaleReport",
    "DescLcfsParityReport",
    "DescLoadedEquilibriumSeed",
    "EquilibriumSeedKind",
    "load_desc_equilibrium_seed_spec",
    "load_desc_equilibrium_seed_runtime",
]
