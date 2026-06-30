"""Runtime DESC coilset construction from SIMSOPT seed fields."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence, Sized
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path

import numpy as np

from banana_opt.desc_bridge.artifact_metadata import (
    DescBridgeArtifactMetadata,
    desc_bridge_source_checksums,
)
from banana_opt.desc_bridge.coil_export import DescCoilExportReport
from banana_opt.desc_bridge.conversion_artifacts import (
    coil_groups_from_biot_savart,
    load_simsopt_biot_savart,
)
from banana_opt.desc_bridge.coil_export import export_simsopt_coil_groups_to_desc
from banana_opt.desc_bridge.runtime_imports import activate_desc_source_root

_FIELD_PARITY_PROBE_POINTS_XYZ = (
    (0.78, 0.0, 0.0),
    (0.84, 0.02, 0.01),
    (0.91, -0.02, -0.01),
    (0.95, 0.03, 0.0),
)
EXPANDED_SIMSOPT_FIELD_COILSET_NFP = 1
EXPANDED_SIMSOPT_FIELD_COILSET_STELLARATOR_SYMMETRY = False
DESC_RUNTIME_FIELD_PARITY_MAX_DELTA_T = 2.0
DESC_RUNTIME_IMPORT_MAX_FIT_RESIDUAL_M = 1.0e-4
DESC_RUNTIME_IMPORT_MAX_LENGTH_DELTA_M = 1.0e-4
DESC_RUNTIME_IMPORT_MAX_CURVATURE_DELTA_INV_M = 5.0
DEFAULT_OPTIMIZED_COIL_GROUPS: tuple[str, ...] = ("banana",)


@dataclass(frozen=True, slots=True)
class DescRuntimeCoilsetBuildReport:
    source_field_path: Path
    status: str
    reason: str
    desc_source_root: Path | None
    desc_version: str | None
    coilset_type: str | None
    desc_fourier_order: int
    sample_count: int
    field_sample_source_grid: int
    field_sample_chunk_size: int
    field_sample_probe_points_xyz: tuple[tuple[float, float, float], ...]
    max_desc_simsopt_field_sample_delta_T: float | None
    mean_desc_simsopt_field_sample_delta_T: float | None
    max_desc_simsopt_field_sample_delta_threshold_T: float
    source_nfp: int
    source_stellarator_symmetry: bool
    coilset_nfp: int
    coilset_stellarator_symmetry: bool
    nfp: int
    stellarator_symmetry: bool
    export_report: DescCoilExportReport | None

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": "desc_runtime_coilset_build_report_v1",
            "source_field_path": os.fspath(self.source_field_path),
            "status": self.status,
            "reason": self.reason,
            "desc_source_root": (
                None if self.desc_source_root is None else os.fspath(self.desc_source_root)
            ),
            "desc_version": self.desc_version,
            "coilset_type": self.coilset_type,
            "desc_fourier_order": self.desc_fourier_order,
            "sample_count": self.sample_count,
            "field_sample_source_grid": self.field_sample_source_grid,
            "field_sample_chunk_size": self.field_sample_chunk_size,
            "field_sample_probe_points_xyz": [
                list(point)
                for point in self.field_sample_probe_points_xyz
            ],
            "max_desc_simsopt_field_sample_delta_T": (
                self.max_desc_simsopt_field_sample_delta_T
            ),
            "mean_desc_simsopt_field_sample_delta_T": (
                self.mean_desc_simsopt_field_sample_delta_T
            ),
            "max_desc_simsopt_field_sample_delta_threshold_T": (
                self.max_desc_simsopt_field_sample_delta_threshold_T
            ),
            "source_nfp": self.source_nfp,
            "source_stellarator_symmetry": self.source_stellarator_symmetry,
            "coilset_nfp": self.coilset_nfp,
            "coilset_stellarator_symmetry": self.coilset_stellarator_symmetry,
            "nfp": self.nfp,
            "stellarator_symmetry": self.stellarator_symmetry,
            "export_report": (
                None if self.export_report is None else self.export_report.to_json_dict()
            ),
        }


@dataclass(frozen=True, slots=True)
class DescRuntimeCoilsetBuildResult:
    coilset: object
    report: DescRuntimeCoilsetBuildReport


@dataclass(frozen=True, slots=True)
class DescCoilsetOptimizationScope:
    group_order: tuple[str, ...]
    group_counts: dict[str, int]
    optimized_group_names: tuple[str, ...]
    fixed_group_names: tuple[str, ...]
    optimized_unique_coil_indices: tuple[int, ...]
    fixed_unique_coil_indices: tuple[int, ...]
    unique_coil_count: int
    optimized_unique_coil_count: int
    fixed_unique_coil_count: int

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": "desc_coilset_optimization_scope_v1",
            "group_order": list(self.group_order),
            "group_counts": dict(self.group_counts),
            "optimized_group_names": list(self.optimized_group_names),
            "fixed_group_names": list(self.fixed_group_names),
            "optimized_unique_coil_indices": list(self.optimized_unique_coil_indices),
            "fixed_unique_coil_indices": list(self.fixed_unique_coil_indices),
            "unique_coil_count": self.unique_coil_count,
            "optimized_unique_coil_count": self.optimized_unique_coil_count,
            "fixed_unique_coil_count": self.fixed_unique_coil_count,
        }


@dataclass(frozen=True, slots=True)
class DescScopedCoilsetOptimization:
    coilset: object
    scope: DescCoilsetOptimizationScope


class DescRuntimeCoilsetBuildError(RuntimeError):
    def __init__(self, report: DescRuntimeCoilsetBuildReport) -> None:
        super().__init__(report.reason)
        self.report = report


def build_desc_runtime_coilset_from_simsopt_field(
    *,
    source_field_path: Path,
    source_artifacts: Mapping[str, Path],
    coil_group_counts: Mapping[str, int],
    desc_fourier_order: int,
    sample_count: int,
    source_nfp: int | None,
    source_stellarator_symmetry: bool | None,
    desc_source_root: Path | None = None,
    field_sample_chunk_size: int = 10,
) -> DescRuntimeCoilsetBuildResult:
    field_sample_delta: tuple[float, float] | None = None
    export_report: DescCoilExportReport | None = None
    try:
        source_nfp = _require_source_nfp(source_nfp)
        source_stellarator_symmetry = _require_source_stellarator_symmetry(
            source_stellarator_symmetry,
        )
        _validate_positive_int(field_sample_chunk_size, "field_sample_chunk_size")
        with activate_desc_source_root(desc_source_root):
            import desc
            from desc.coils import CoilSet, FourierXYZCoil

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
                desc_fourier_xyz_coil_cls=FourierXYZCoil,
                desc_fourier_order=desc_fourier_order,
                sample_count=sample_count,
                artifact_metadata=artifact_metadata,
                source_nfp=source_nfp,
                source_stellarator_symmetry=source_stellarator_symmetry,
                source_group_order=tuple(coil_group_counts),
            )
            export_report = export_result.report
            _validate_desc_import_geometry_fidelity(export_report)
            coilset = CoilSet(
                *export_result.desc_coils,
                NFP=EXPANDED_SIMSOPT_FIELD_COILSET_NFP,
                sym=EXPANDED_SIMSOPT_FIELD_COILSET_STELLARATOR_SYMMETRY,
                name="desc_joint_seed_coilset",
                check_intersection=False,
            )
            field_sample_delta = _desc_simsopt_field_sample_delta_T(
                biot_savart=biot_savart,
                coilset=coilset,
                source_grid=sample_count,
                chunk_size=field_sample_chunk_size,
            )
            _validate_desc_simsopt_field_sample_delta(field_sample_delta)
            report = _runtime_coilset_report(
                source_field_path=source_field_path,
                status="passed",
                reason="DESC runtime coilset built from SIMSOPT seed field.",
                desc_source_root=desc_source_root,
                desc_version=_desc_version(desc),
                coilset=coilset,
                desc_fourier_order=desc_fourier_order,
                sample_count=sample_count,
                field_sample_source_grid=sample_count,
                field_sample_chunk_size=field_sample_chunk_size,
                field_sample_delta=field_sample_delta,
                source_nfp=source_nfp,
                source_stellarator_symmetry=source_stellarator_symmetry,
                coilset_nfp=EXPANDED_SIMSOPT_FIELD_COILSET_NFP,
                coilset_stellarator_symmetry=(
                    EXPANDED_SIMSOPT_FIELD_COILSET_STELLARATOR_SYMMETRY
                ),
                export_report=export_report,
            )
            return DescRuntimeCoilsetBuildResult(coilset=coilset, report=report)
    except Exception as exc:
        report = _runtime_coilset_report(
            source_field_path=source_field_path,
            status="failed",
            reason=f"{type(exc).__name__}: {exc}",
            desc_source_root=desc_source_root,
            desc_version=None,
            coilset=None,
            desc_fourier_order=desc_fourier_order,
            sample_count=sample_count,
            field_sample_source_grid=sample_count,
            field_sample_chunk_size=field_sample_chunk_size,
            field_sample_delta=field_sample_delta,
            source_nfp=_report_source_nfp(source_nfp),
            source_stellarator_symmetry=_report_source_stellarator_symmetry(
                source_stellarator_symmetry,
            ),
            coilset_nfp=EXPANDED_SIMSOPT_FIELD_COILSET_NFP,
            coilset_stellarator_symmetry=(
                EXPANDED_SIMSOPT_FIELD_COILSET_STELLARATOR_SYMMETRY
            ),
            export_report=export_report,
        )
        raise DescRuntimeCoilsetBuildError(report) from exc


def load_desc_runtime_coilset_checkpoint(
    *,
    checkpoint_path: Path,
    coil_group_counts: Mapping[str, int],
    source_nfp: int | None,
    source_stellarator_symmetry: bool | None,
    desc_source_root: Path | None = None,
    desc_fourier_order: int = 0,
    sample_count: int = 0,
    field_sample_chunk_size: int = 10,
) -> DescRuntimeCoilsetBuildResult:
    """Load a saved DESC CoilSet checkpoint for optimizer continuation."""

    try:
        source_nfp = _require_source_nfp(source_nfp)
        source_stellarator_symmetry = _require_source_stellarator_symmetry(
            source_stellarator_symmetry,
        )
        _validate_nonnegative_int(desc_fourier_order, "desc_fourier_order")
        _validate_nonnegative_int(sample_count, "sample_count")
        _validate_positive_int(field_sample_chunk_size, "field_sample_chunk_size")
        checkpoint = checkpoint_path.expanduser().resolve()
        if not checkpoint.is_file():
            raise ValueError(f"DESC CoilSet checkpoint does not exist: {checkpoint}.")
        with activate_desc_source_root(desc_source_root):
            import desc
            from desc.coils import CoilSet
            from desc.io import load as desc_load

            coilset = desc_load(os.fspath(checkpoint))
            if not isinstance(coilset, CoilSet):
                raise TypeError(
                    "DESC CoilSet checkpoint must load as desc.coils.CoilSet; got "
                    f"{type(coilset).__module__}.{type(coilset).__qualname__}."
                )
            grouped_coil_count = _coil_group_count_total(coil_group_counts)
            unique_coil_count = _unique_coil_count(coilset)
            if grouped_coil_count != unique_coil_count:
                raise ValueError(
                    "DESC CoilSet checkpoint unique-coil count does not match the "
                    "seed coil group manifest: "
                    f"grouped {grouped_coil_count}, coilset {unique_coil_count}."
                )
            report = _runtime_coilset_report(
                source_field_path=checkpoint,
                status="passed",
                reason="DESC runtime coilset loaded from DESC checkpoint.",
                desc_source_root=desc_source_root,
                desc_version=_desc_version(desc),
                coilset=coilset,
                desc_fourier_order=desc_fourier_order,
                sample_count=sample_count,
                field_sample_source_grid=sample_count,
                field_sample_chunk_size=field_sample_chunk_size,
                field_sample_delta=None,
                source_nfp=source_nfp,
                source_stellarator_symmetry=source_stellarator_symmetry,
                coilset_nfp=_coilset_nfp(coilset),
                coilset_stellarator_symmetry=_coilset_stellarator_symmetry(coilset),
                export_report=None,
            )
            return DescRuntimeCoilsetBuildResult(coilset=coilset, report=report)
    except Exception as exc:
        report = _runtime_coilset_report(
            source_field_path=checkpoint_path,
            status="failed",
            reason=f"{type(exc).__name__}: {exc}",
            desc_source_root=desc_source_root,
            desc_version=None,
            coilset=None,
            desc_fourier_order=desc_fourier_order,
            sample_count=sample_count,
            field_sample_source_grid=sample_count,
            field_sample_chunk_size=field_sample_chunk_size,
            field_sample_delta=None,
            source_nfp=_report_source_nfp(source_nfp),
            source_stellarator_symmetry=_report_source_stellarator_symmetry(
                source_stellarator_symmetry,
            ),
            coilset_nfp=EXPANDED_SIMSOPT_FIELD_COILSET_NFP,
            coilset_stellarator_symmetry=(
                EXPANDED_SIMSOPT_FIELD_COILSET_STELLARATOR_SYMMETRY
            ),
            export_report=None,
        )
        raise DescRuntimeCoilsetBuildError(report) from exc


def scope_desc_coilset_optimization_to_groups(
    *,
    coilset: object,
    coil_group_counts: Mapping[str, int],
    optimized_group_names: Sequence[str] = DEFAULT_OPTIMIZED_COIL_GROUPS,
    desc_source_root: Path | None = None,
) -> DescScopedCoilsetOptimization:
    """Expose selected coil groups to DESC's optimizer while keeping the full field."""

    scope = _coilset_optimization_scope(
        coil_group_counts=coil_group_counts,
        optimized_group_names=optimized_group_names,
        unique_coil_count=_unique_coil_count(coilset),
    )
    with activate_desc_source_root(desc_source_root):
        from desc.backend import jnp
        from desc.coils import CoilSet

        if not isinstance(coilset, CoilSet):
            raise TypeError(
                "DESC coil optimization scoping requires a desc.coils.CoilSet; "
                f"got {type(coilset).__module__}.{type(coilset).__qualname__}."
            )

        class ScopedDescCoilSet(CoilSet):
            _static_attrs = [
                *getattr(CoilSet, "_static_attrs", []),
                "_optimized_unique_coil_indices",
            ]

            def __init__(
                self,
                full_coilset: object,
                optimization_scope: DescCoilsetOptimizationScope,
            ) -> None:
                self._optimized_unique_coil_indices = (
                    optimization_scope.optimized_unique_coil_indices
                )
                self._coils = list(full_coilset.coils)
                self._NFP = int(full_coilset.NFP)
                self._sym = bool(full_coilset.sym)
                self._name = f"{full_coilset.name}_scoped_to_banana"

            @property
            def coils(self) -> list[object]:
                return self._coils

            @property
            def NFP(self) -> int:
                return self._NFP

            @property
            def sym(self) -> bool:
                return self._sym

            @property
            def name(self) -> str:
                return self._name

            @property
            def desc_joint_optimization_scope(
                self,
            ) -> DescCoilsetOptimizationScope:
                return scope

            @property
            def optimizable_params(self) -> list[object]:
                return [
                    self[index].optimizable_params
                    for index in self._optimized_unique_coil_indices
                ]

            @property
            def params_dict(self) -> list[dict[str, object]]:
                return [
                    self[index].params_dict
                    for index in self._optimized_unique_coil_indices
                ]

            @params_dict.setter
            def params_dict(self, params_dict: object) -> None:
                params = _require_sequence_length(
                    params_dict,
                    expected_length=len(self._optimized_unique_coil_indices),
                    field_name="params_dict",
                )
                for coil_index, coil_params in zip(
                    self._optimized_unique_coil_indices,
                    params,
                ):
                    self[coil_index].params_dict = dict(coil_params)

            @property
            def dimensions(self) -> list[dict[str, int]]:
                return [
                    self[index].dimensions
                    for index in self._optimized_unique_coil_indices
                ]

            @property
            def x_idx(self) -> list[dict[str, object]]:
                x_idx = [
                    self[index].x_idx
                    for index in self._optimized_unique_coil_indices
                ]
                dim_offsets = jnp.concatenate(
                    [
                        jnp.array([0]),
                        jnp.cumsum(
                            jnp.array(
                                [
                                    self[index].dim_x
                                    for index in self._optimized_unique_coil_indices
                                ],
                            ),
                        )[:-1],
                    ],
                )
                for dim_offset, coil_indices in zip(dim_offsets, x_idx):
                    for key in coil_indices:
                        coil_indices[key] += dim_offset
                return x_idx

            @property
            def dim_x(self) -> int:
                return int(
                    sum(
                        self[index].dim_x
                        for index in self._optimized_unique_coil_indices
                    ),
                )

            def pack_params(self, params: object) -> object:
                scoped_params = _require_sequence_length(
                    params,
                    expected_length=len(self._optimized_unique_coil_indices),
                    field_name="params",
                )
                return jnp.concatenate(
                    [
                        self[coil_index].pack_params(dict(coil_params))
                        for coil_index, coil_params in zip(
                            self._optimized_unique_coil_indices,
                            scoped_params,
                        )
                    ],
                )

            def unpack_params(self, x: object) -> list[dict[str, object]]:
                split_indices = np.cumsum(
                    [
                        self[index].dim_x
                        for index in self._optimized_unique_coil_indices
                    ],
                )[:-1]
                split_x = jnp.split(x, split_indices)
                return [
                    self[coil_index].unpack_params(coil_x)
                    for coil_index, coil_x in zip(
                        self._optimized_unique_coil_indices,
                        split_x,
                    )
                ]

            def compute(
                self,
                names: object,
                grid: object = None,
                params: object = None,
                transforms: object = None,
                data: object = None,
                **kwargs: object,
            ) -> object:
                return CoilSet.compute(
                    self,
                    names,
                    grid=grid,
                    params=self._merge_params(params),
                    transforms=transforms,
                    data=data,
                    **kwargs,
                )

            def _compute_position(
                self,
                params: object = None,
                grid: object = None,
                dx1: bool = False,
                **kwargs: object,
            ) -> object:
                return CoilSet._compute_position(
                    self,
                    params=self._merge_params(params),
                    grid=grid,
                    dx1=dx1,
                    **kwargs,
                )

            def compute_magnetic_field(
                self,
                coords: object,
                params: object = None,
                basis: str = "rpz",
                source_grid: object = None,
                transforms: object = None,
                chunk_size: int | None = None,
            ) -> object:
                return CoilSet.compute_magnetic_field(
                    self,
                    coords,
                    params=self._merge_params(params),
                    basis=basis,
                    source_grid=source_grid,
                    transforms=transforms,
                    chunk_size=chunk_size,
                )

            def compute_magnetic_vector_potential(
                self,
                coords: object,
                params: object = None,
                basis: str = "rpz",
                source_grid: object = None,
                transforms: object = None,
                chunk_size: int | None = None,
            ) -> object:
                return CoilSet.compute_magnetic_vector_potential(
                    self,
                    coords,
                    params=self._merge_params(params),
                    basis=basis,
                    source_grid=source_grid,
                    transforms=transforms,
                    chunk_size=chunk_size,
                )

            def _all_currents(self, currents: object = None) -> object:
                if currents is None:
                    return CoilSet._all_currents(self)
                merged_currents = list(self.current)
                scoped_currents = jnp.atleast_1d(currents).flatten()
                if scoped_currents.size != len(self._optimized_unique_coil_indices):
                    raise ValueError(
                        "Scoped DESC coilset expected one current per optimized "
                        f"unique coil; got {scoped_currents.size} currents for "
                        f"{len(self._optimized_unique_coil_indices)} coils."
                    )
                for scoped_index, coil_index in enumerate(
                    self._optimized_unique_coil_indices,
                ):
                    merged_currents[coil_index] = scoped_currents[scoped_index]
                return CoilSet._all_currents(self, merged_currents)

            def save(self, path: str) -> None:
                full_coilset = CoilSet(
                    *self.coils,
                    NFP=self.NFP,
                    sym=self.sym,
                    name=self.name,
                    check_intersection=False,
                )
                full_coilset.save(path)

            def copy(self, deepcopy: bool = True) -> object:
                copied_coils = tuple(coil.copy(deepcopy=deepcopy) for coil in self.coils)
                full_coilset = CoilSet(
                    *copied_coils,
                    NFP=self.NFP,
                    sym=self.sym,
                    name=self.name,
                    check_intersection=False,
                )
                return type(self)(
                    full_coilset,
                    scope,
                )

            def _merge_params(self, params: object) -> object:
                if params is None:
                    return None
                if (
                    not isinstance(params, Mapping)
                    and not isinstance(params, str)
                    and isinstance(params, Sequence)
                    and len(params) == len(self.coils)
                ):
                    for item in params:
                        if not isinstance(item, Mapping):
                            raise ValueError(
                                "params entries must be parameter dictionaries."
                            )
                    return params
                scoped_params = _require_sequence_length(
                    params,
                    expected_length=len(self._optimized_unique_coil_indices),
                    field_name="params",
                )
                full_params = [coil.params_dict for coil in self.coils]
                for scoped_param, coil_index in zip(
                    scoped_params,
                    self._optimized_unique_coil_indices,
                ):
                    full_params[coil_index] = dict(scoped_param)
                return full_params

        return DescScopedCoilsetOptimization(
            coilset=ScopedDescCoilSet(coilset, scope),
            scope=scope,
        )


def _runtime_coilset_report(
    *,
    source_field_path: Path,
    status: str,
    reason: str,
    desc_source_root: Path | None,
    desc_version: str | None,
    coilset: object | None,
    desc_fourier_order: int,
    sample_count: int,
    field_sample_source_grid: int,
    field_sample_chunk_size: int,
    field_sample_delta: tuple[float, float] | None,
    source_nfp: int,
    source_stellarator_symmetry: bool,
    coilset_nfp: int,
    coilset_stellarator_symmetry: bool,
    export_report: DescCoilExportReport | None,
) -> DescRuntimeCoilsetBuildReport:
    return DescRuntimeCoilsetBuildReport(
        source_field_path=source_field_path.resolve(),
        status=status,
        reason=reason,
        desc_source_root=None if desc_source_root is None else desc_source_root.resolve(),
        desc_version=desc_version,
        coilset_type=(
            None
            if coilset is None
            else f"{type(coilset).__module__}.{type(coilset).__qualname__}"
        ),
        desc_fourier_order=desc_fourier_order,
        sample_count=sample_count,
        field_sample_source_grid=field_sample_source_grid,
        field_sample_chunk_size=field_sample_chunk_size,
        field_sample_probe_points_xyz=_FIELD_PARITY_PROBE_POINTS_XYZ,
        max_desc_simsopt_field_sample_delta_T=(
            None if field_sample_delta is None else field_sample_delta[0]
        ),
        mean_desc_simsopt_field_sample_delta_T=(
            None if field_sample_delta is None else field_sample_delta[1]
        ),
        max_desc_simsopt_field_sample_delta_threshold_T=(
            DESC_RUNTIME_FIELD_PARITY_MAX_DELTA_T
        ),
        source_nfp=source_nfp,
        source_stellarator_symmetry=source_stellarator_symmetry,
        coilset_nfp=coilset_nfp,
        coilset_stellarator_symmetry=coilset_stellarator_symmetry,
        nfp=source_nfp,
        stellarator_symmetry=source_stellarator_symmetry,
        export_report=export_report,
    )


def _desc_simsopt_field_sample_delta_T(
    *,
    biot_savart: object,
    coilset: object,
    source_grid: int,
    chunk_size: int,
) -> tuple[float, float]:
    probe_points = np.asarray(_FIELD_PARITY_PROBE_POINTS_XYZ, dtype=float)
    biot_savart.set_points(probe_points)
    simsopt_field = np.asarray(biot_savart.B(), dtype=float)
    desc_field = np.asarray(
        coilset.compute_magnetic_field(
            probe_points,
            basis="xyz",
            source_grid=source_grid,
            chunk_size=chunk_size,
        ),
        dtype=float,
    )
    if simsopt_field.shape != desc_field.shape:
        raise ValueError(
            "DESC/SIMSOPT field parity sample shape mismatch: "
            f"SIMSOPT {simsopt_field.shape}, DESC {desc_field.shape}."
        )
    deltas = np.linalg.norm(simsopt_field - desc_field, axis=1)
    if not np.isfinite(deltas).all():
        raise ValueError("DESC/SIMSOPT field parity sample produced non-finite deltas.")
    return float(np.max(deltas)), float(np.mean(deltas))


def _validate_desc_simsopt_field_sample_delta(
    field_sample_delta: tuple[float, float],
) -> None:
    max_delta_T, mean_delta_T = field_sample_delta
    if max_delta_T > DESC_RUNTIME_FIELD_PARITY_MAX_DELTA_T:
        raise ValueError(
            "DESC/SIMSOPT field parity exceeded the runtime coilset threshold: "
            f"max_delta_T={max_delta_T:.6g}, mean_delta_T={mean_delta_T:.6g}, "
            f"threshold_T={DESC_RUNTIME_FIELD_PARITY_MAX_DELTA_T:.6g}. "
            "SIMSOPT seed fields are expanded physical coil sets, so DESC "
            "CoilSet symmetry virtualization must not change the field."
        )


def _validate_desc_import_geometry_fidelity(
    export_report: DescCoilExportReport,
) -> None:
    """Reject SIMSOPT-to-DESC imports that already corrupt coil geometry."""

    residuals = export_report.artifact_metadata.conversion_residuals
    if residuals is None:
        raise ValueError(
            "DESC runtime coilset import fidelity requires conversion residuals."
        )
    failures: list[str] = []
    if residuals.max_fit_residual_m > DESC_RUNTIME_IMPORT_MAX_FIT_RESIDUAL_M:
        failures.append(
            "max_fit_residual_m="
            f"{residuals.max_fit_residual_m:.6g} > "
            f"{DESC_RUNTIME_IMPORT_MAX_FIT_RESIDUAL_M:.6g}"
        )
    if residuals.max_abs_length_delta_m > DESC_RUNTIME_IMPORT_MAX_LENGTH_DELTA_M:
        failures.append(
            "max_abs_length_delta_m="
            f"{residuals.max_abs_length_delta_m:.6g} > "
            f"{DESC_RUNTIME_IMPORT_MAX_LENGTH_DELTA_M:.6g}"
        )
    if (
        residuals.max_abs_curvature_delta_inv_m
        > DESC_RUNTIME_IMPORT_MAX_CURVATURE_DELTA_INV_M
    ):
        failures.append(
            "max_abs_curvature_delta_inv_m="
            f"{residuals.max_abs_curvature_delta_inv_m:.6g} > "
            f"{DESC_RUNTIME_IMPORT_MAX_CURVATURE_DELTA_INV_M:.6g}"
        )
    if failures:
        raise ValueError(
            "DESC runtime coilset import fidelity exceeded threshold: "
            f"{'; '.join(failures)}. Preserve the source parameterization or "
            "increase the conversion Fourier order before optimization."
        )


def _require_source_nfp(source_nfp: int | None) -> int:
    if source_nfp is None:
        raise ValueError(
            "DESC runtime coilset assembly requires explicit seed source NFP."
        )
    if isinstance(source_nfp, bool) or not isinstance(source_nfp, int) or source_nfp <= 0:
        raise ValueError("DESC runtime coilset source NFP must be a positive integer.")
    return source_nfp


def _require_source_stellarator_symmetry(
    source_stellarator_symmetry: bool | None,
) -> bool:
    if source_stellarator_symmetry is None:
        raise ValueError(
            "DESC runtime coilset assembly requires explicit seed source "
            "stellarator_symmetry."
        )
    if not isinstance(source_stellarator_symmetry, bool):
        raise ValueError(
            "DESC runtime coilset source stellarator_symmetry must be boolean."
        )
    return source_stellarator_symmetry


def _validate_positive_int(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer.")


def _validate_nonnegative_int(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a nonnegative integer.")


def _coil_group_count_total(coil_group_counts: Mapping[str, int]) -> int:
    if not coil_group_counts:
        raise ValueError("DESC CoilSet checkpoint requires coil group counts.")
    total = 0
    for group_name, group_count in coil_group_counts.items():
        if not isinstance(group_name, str) or group_name == "":
            raise ValueError("DESC coil group names must be nonempty strings.")
        if (
            isinstance(group_count, bool)
            or not isinstance(group_count, int)
            or group_count < 0
        ):
            raise ValueError("DESC coil group counts must be nonnegative integers.")
        total += int(group_count)
    return total


def _coilset_optimization_scope(
    *,
    coil_group_counts: Mapping[str, int],
    optimized_group_names: Sequence[str],
    unique_coil_count: int,
) -> DescCoilsetOptimizationScope:
    if not coil_group_counts:
        raise ValueError("DESC coil optimization scoping requires coil group counts.")
    group_order = tuple(coil_group_counts)
    counts: dict[str, int] = {}
    for group_name in group_order:
        count = coil_group_counts[group_name]
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError("DESC coil group counts must be nonnegative integers.")
        counts[group_name] = int(count)
    if sum(counts.values()) != unique_coil_count:
        raise ValueError(
            "DESC coil group counts do not match the CoilSet unique-coil count: "
            f"grouped {sum(counts.values())}, coilset {unique_coil_count}."
        )
    optimized_groups = _normalized_group_names(
        optimized_group_names,
        field_name="optimized_group_names",
    )
    unknown_groups = tuple(group for group in optimized_groups if group not in counts)
    if unknown_groups:
        raise ValueError(
            "DESC optimized coil groups are absent from the seed groups: "
            f"{', '.join(unknown_groups)}."
        )

    optimized_indices: list[int] = []
    fixed_indices: list[int] = []
    fixed_groups: list[str] = []
    cursor = 0
    for group_name in group_order:
        group_count = counts[group_name]
        group_indices = range(cursor, cursor + group_count)
        if group_name in optimized_groups:
            optimized_indices.extend(group_indices)
        else:
            fixed_groups.append(group_name)
            fixed_indices.extend(group_indices)
        cursor += group_count
    if not optimized_indices:
        raise ValueError("DESC coil optimization scope selected zero coils.")
    return DescCoilsetOptimizationScope(
        group_order=group_order,
        group_counts=counts,
        optimized_group_names=optimized_groups,
        fixed_group_names=tuple(fixed_groups),
        optimized_unique_coil_indices=tuple(optimized_indices),
        fixed_unique_coil_indices=tuple(fixed_indices),
        unique_coil_count=unique_coil_count,
        optimized_unique_coil_count=len(optimized_indices),
        fixed_unique_coil_count=len(fixed_indices),
    )


def _normalized_group_names(
    group_names: Sequence[str],
    *,
    field_name: str,
) -> tuple[str, ...]:
    if isinstance(group_names, str) or not isinstance(group_names, Sequence):
        raise ValueError(f"{field_name} must be a sequence of group names.")
    normalized = tuple(group_name.strip() for group_name in group_names)
    if not normalized or any(group_name == "" for group_name in normalized):
        raise ValueError(f"{field_name} must contain nonempty group names.")
    duplicates = sorted(
        group_name for group_name in set(normalized) if normalized.count(group_name) > 1
    )
    if duplicates:
        raise ValueError(f"{field_name} contains duplicate groups: {duplicates}.")
    return normalized


def _unique_coil_count(coilset: object) -> int:
    if not isinstance(coilset, Sized):
        raise TypeError("DESC coil optimization scoping requires a sized CoilSet.")
    return len(coilset)


def _coilset_nfp(coilset: object) -> int:
    value = getattr(coilset, "NFP", None)
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError("DESC CoilSet checkpoint NFP must be a positive integer.")
    return int(value)


def _coilset_stellarator_symmetry(coilset: object) -> bool:
    value = getattr(coilset, "sym", None)
    if not isinstance(value, bool | np.bool_):
        raise ValueError("DESC CoilSet checkpoint sym must be boolean.")
    return bool(value)


def _require_sequence_length(
    value: object,
    *,
    expected_length: int,
    field_name: str,
) -> Sequence[Mapping[str, object]]:
    if isinstance(value, Mapping) or isinstance(value, str):
        raise ValueError(f"{field_name} must be a sequence of parameter dictionaries.")
    if not isinstance(value, Sequence):
        raise ValueError(f"{field_name} must be a sequence of parameter dictionaries.")
    if len(value) != expected_length:
        raise ValueError(
            f"{field_name} must contain {expected_length} parameter dictionaries; "
            f"got {len(value)}."
        )
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError(f"{field_name} entries must be parameter dictionaries.")
    return value


def _report_source_nfp(source_nfp: object) -> int:
    if isinstance(source_nfp, bool) or not isinstance(source_nfp, int):
        return 0
    return source_nfp


def _report_source_stellarator_symmetry(
    source_stellarator_symmetry: object,
) -> bool:
    if not isinstance(source_stellarator_symmetry, bool):
        return False
    return source_stellarator_symmetry


def _desc_version(desc_module: object) -> str | None:
    version = getattr(desc_module, "__version__", None)
    if isinstance(version, str) and version != "":
        return version
    return None


__all__ = [
    "DEFAULT_OPTIMIZED_COIL_GROUPS",
    "DescCoilsetOptimizationScope",
    "DescRuntimeCoilsetBuildError",
    "DescRuntimeCoilsetBuildReport",
    "DescRuntimeCoilsetBuildResult",
    "DescScopedCoilsetOptimization",
    "build_desc_runtime_coilset_from_simsopt_field",
    "load_desc_runtime_coilset_checkpoint",
    "scope_desc_coilset_optimization_to_groups",
]
