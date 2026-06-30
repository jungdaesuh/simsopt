import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_ROOT = REPO_ROOT / "examples" / "single_stage_optimization"
EXAMPLES_ROOT_STR = str(EXAMPLES_ROOT)
if EXAMPLES_ROOT_STR not in sys.path:
    sys.path.insert(0, EXAMPLES_ROOT_STR)

from banana_opt.desc_bridge.coil_export import (  # noqa: E402
    coil_groups_from_stage2_partitions,
    export_simsopt_coil_groups_to_desc,
)
from banana_opt.desc_bridge.artifact_metadata import (  # noqa: E402
    DescBridgeArtifactMetadata,
    DescBridgeBananaPackMetadata,
    DescBridgeSourceChecksum,
    desc_bridge_source_checksums,
)
from banana_opt.desc_bridge.coil_geometry import (  # noqa: E402
    min_pairwise_periodic_distance,
    periodic_length,
)
from banana_opt.desc_bridge.coil_import import (  # noqa: E402
    DescSampledCoil,
    import_desc_sampled_coils_to_simsopt,
    validate_hardware_oracle_binding,
)
from banana_opt.desc_bridge.runtime_coilset import (  # noqa: E402
    DESC_RUNTIME_IMPORT_MAX_FIT_RESIDUAL_M,
    _validate_desc_import_geometry_fidelity,
)
from banana_opt.desc_bridge.runtime_export import (  # noqa: E402
    sample_desc_coilset_unique_coils,
)
from banana_opt.desc_bridge.coil_report_utils import (  # noqa: E402
    coil_convention_report,
)
from banana_opt.desc_joint_validation import (  # noqa: E402
    DESC_JOINT_FINAL_ORACLE_EVIDENCE_SCHEMA_VERSION,
)
from simsopt.field import BiotSavart  # noqa: E402
from simsopt.field.coil import Coil, Current  # noqa: E402
from simsopt.geo import CurveXYZFourier  # noqa: E402


@dataclass(frozen=True, slots=True)
class _FakeDescFourierXYZCoil:
    current_A: float
    coords: np.ndarray
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
    ) -> "_FakeDescFourierXYZCoil":
        return cls(
            current_A=float(current_A),
            coords=np.asarray(coords, dtype=float).copy(),
            order=int(N),
            basis=basis,
            name=name,
        )


@dataclass(frozen=True, slots=True)
class _Partitions:
    tf_coils: tuple[object, ...]
    banana_coils: tuple[object, ...]
    proxy_coils: tuple[object, ...]
    vf_coils: tuple[object, ...]


class _MutableSamplingCurve:
    def __init__(self) -> None:
        self.quadpoints = np.linspace(0.0, 1.0, 11, endpoint=False)

    def set_points(self, points: np.ndarray) -> None:
        self.quadpoints = np.asarray(points, dtype=float)

    def gamma(self) -> np.ndarray:
        angle = 2.0 * np.pi * self.quadpoints
        return np.column_stack(
            (
                0.8 + 0.02 * np.cos(angle),
                0.02 * np.sin(angle),
                0.01 * np.sin(2.0 * angle),
            )
        )


@dataclass(frozen=True, slots=True)
class _MutableSamplingCoil:
    curve: _MutableSamplingCurve
    current: Current


class _FixedSamplingCurve:
    def __init__(self) -> None:
        self.quadpoints = np.linspace(0.0, 1.0, 11, endpoint=False)

    def gamma(self) -> np.ndarray:
        angle = 2.0 * np.pi * self.quadpoints
        return np.column_stack(
            (
                0.75 + 0.02 * np.cos(angle),
                0.02 * np.sin(angle),
                0.01 * np.sin(2.0 * angle),
            )
        )


@dataclass(frozen=True, slots=True)
class _FixedSamplingCoil:
    curve: _FixedSamplingCurve
    current: Current


@dataclass(frozen=True, slots=True)
class _FakeDescRuntimeCoil:
    name: str
    current: float
    coords_xyz: np.ndarray

    def _compute_position(self, *, grid: object, basis: str) -> np.ndarray:
        return self.coords_xyz


@dataclass(frozen=True, slots=True)
class _GridAwareDescRuntimeCoil:
    name: str
    current: float
    radius: float

    def _compute_position(self, *, grid: object, basis: str) -> np.ndarray:
        assert basis == "xyz"
        assert not isinstance(grid, int)
        nodes = np.asarray(getattr(grid, "nodes"), dtype=float)
        zeta = nodes[:, 2]
        return np.column_stack(
            (
                self.radius * np.cos(zeta),
                self.radius * np.sin(zeta),
                0.01 * np.sin(2.0 * zeta),
            )
        )


@dataclass(frozen=True, slots=True)
class _FakeDescRuntimeCoilSet:
    coils: tuple[_FakeDescRuntimeCoil | _GridAwareDescRuntimeCoil, ...]


@dataclass(frozen=True, slots=True)
class _NamedSamplingCoil:
    curve: _FixedSamplingCurve | _MutableSamplingCurve
    current: Current
    name: str


def _coil(current_A: float, *, x_offset: float, sample_count: int = 32) -> Coil:
    curve = CurveXYZFourier(sample_count, 1)
    dofs = curve.get_dofs().copy()
    dofs[:] = 0.0
    dofs[0] = x_offset
    dofs[1] = 0.02
    dofs[4] = 0.02
    dofs[8] = 0.01
    curve.set_dofs(dofs)
    return Coil(curve, Current(current_A))


def _higher_order_coil(
    current_A: float,
    *,
    x_offset: float,
    sample_count: int = 64,
) -> Coil:
    curve = CurveXYZFourier(sample_count, 3)
    dofs = curve.get_dofs().copy()
    dofs[:] = 0.0
    dofs[0] = x_offset
    dofs[2] = 0.02
    dofs[8] = 0.02
    dofs[20] = 0.006
    curve.set_dofs(dofs)
    return Coil(curve, Current(current_A))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_final_oracle_evidence(
    path: Path,
    *,
    source_artifact_checksums: dict[str, str],
    exported_artifact_paths: tuple[str, ...] = (),
) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": DESC_JOINT_FINAL_ORACLE_EVIDENCE_SCHEMA_VERSION,
                "source": "direct_loaded_artifact_hardware_contact_oracle",
                "passed": True,
                "exported_artifact_paths": list(exported_artifact_paths),
                "exported_artifact_checksums": {
                    artifact_path: _sha256(Path(artifact_path))
                    for artifact_path in exported_artifact_paths
                },
                "source_artifact_checksums": source_artifact_checksums,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def test_export_simsopt_coils_to_desc_preserves_group_order_and_current_signs():
    tf_coil = _coil(-8.0e4, x_offset=0.9)
    banana_positive = _coil(1.2e4, x_offset=0.8)
    banana_negative = _coil(-9.5e3, x_offset=0.82)
    vf_coil = _coil(250.0, x_offset=1.1)
    coil_groups = {
        "banana": (banana_positive, banana_negative),
        "tf": (tf_coil,),
        "proxy": (),
        "vf": (vf_coil,),
    }

    result = export_simsopt_coil_groups_to_desc(
        coil_groups,
        desc_fourier_xyz_coil_cls=_FakeDescFourierXYZCoil,
        desc_fourier_order=3,
        sample_count=32,
    )

    assert [entry.group for entry in result.report.entries] == [
        "tf",
        "banana",
        "banana",
        "vf",
    ]
    assert [entry.current_A for entry in result.report.entries] == [
        -8.0e4,
        1.2e4,
        -9.5e3,
        250.0,
    ]
    assert [coil.current_A for coil in result.desc_coils] == [
        -8.0e4,
        1.2e4,
        -9.5e3,
        250.0,
    ]
    assert [coil.name for coil in result.desc_coils] == [
        "tf_000",
        "banana_000",
        "banana_001",
        "vf_000",
    ]
    assert result.report.group_order == ("tf", "banana", "proxy", "vf")
    assert result.report.group_counts == {
        "tf": 1,
        "banana": 2,
        "proxy": 0,
        "vf": 1,
    }
    assert result.report.current_sign_counts == {
        "negative": 2,
        "zero": 0,
        "positive": 2,
    }
    assert result.report.to_json_dict()["coil_conventions"] == coil_convention_report()
    assert all(entry.sample_count == 32 for entry in result.report.entries)
    assert all(entry.coordinate_basis == "xyz" for entry in result.report.entries)


def test_export_report_records_fit_length_and_curvature_residuals():
    result = export_simsopt_coil_groups_to_desc(
        {
            "tf": (),
            "banana": (
                _coil(1.0e4, x_offset=0.8),
                _coil(-1.0e4, x_offset=0.85),
            ),
            "proxy": (),
            "vf": (),
        },
        desc_fourier_xyz_coil_cls=_FakeDescFourierXYZCoil,
        desc_fourier_order=3,
        sample_count=32,
    )
    entry = result.report.entries[0]

    assert entry.group == "banana"
    assert entry.max_fit_residual_m >= 0.0
    assert entry.rms_fit_residual_m >= 0.0
    assert entry.source_length_m > 0.0
    assert entry.reconstructed_length_m > 0.0
    assert np.isfinite(entry.length_delta_m)
    assert entry.source_max_curvature_inv_m >= 0.0
    assert entry.reconstructed_max_curvature_inv_m >= 0.0
    assert np.isfinite(entry.max_curvature_delta_inv_m)
    assert entry.field_sample_delta_T is not None
    assert entry.field_sample_delta_T <= 1.0e-12
    assert result.report.source_min_coil_distance_m is not None
    assert result.report.reconstructed_min_coil_distance_m is not None
    assert result.report.min_coil_distance_delta_m is not None
    assert np.isfinite(result.report.min_coil_distance_delta_m)


def test_export_restores_mutable_curve_sampling_points():
    curve = _MutableSamplingCurve()
    original_points = curve.quadpoints.copy()

    export_simsopt_coil_groups_to_desc(
        {
            "tf": (),
            "banana": (_MutableSamplingCoil(curve=curve, current=Current(1.0e4)),),
            "proxy": (),
            "vf": (),
        },
        desc_fourier_xyz_coil_cls=_FakeDescFourierXYZCoil,
        desc_fourier_order=3,
        sample_count=32,
    )

    np.testing.assert_allclose(curve.quadpoints, original_points, rtol=0.0, atol=0.0)


def test_export_re_evaluates_sparse_curve_xyz_fourier_without_linear_resampling():
    sparse_coil = _coil(1.0e4, x_offset=0.8, sample_count=15)
    expected_curve = CurveXYZFourier(64, 1)
    expected_curve.set_dofs(sparse_coil.curve.get_dofs())

    result = export_simsopt_coil_groups_to_desc(
        {"tf": (), "banana": (sparse_coil,), "proxy": (), "vf": ()},
        desc_fourier_xyz_coil_cls=_FakeDescFourierXYZCoil,
        desc_fourier_order=1,
        sample_count=64,
    )

    np.testing.assert_allclose(
        result.desc_coils[0].coords,
        expected_curve.gamma(),
        atol=1.0e-14,
        rtol=0.0,
    )
    assert result.desc_coils[0].coords.shape == (64, 3)
    assert result.report.entries[0].sample_count == 64
    assert result.report.entries[0].max_fit_residual_m <= 1.0e-12
    assert result.report.entries[0].field_sample_delta_T is not None
    assert np.isfinite(result.report.entries[0].field_sample_delta_T)


def test_export_rejects_fixed_native_curve_when_requested_sample_count_differs():
    with np.testing.assert_raises_regex(ValueError, "lossy linear resampling"):
        export_simsopt_coil_groups_to_desc(
            {
                "tf": (),
                "banana": (
                    _FixedSamplingCoil(
                        curve=_FixedSamplingCurve(),
                        current=Current(1.0e4),
                    ),
                ),
                "proxy": (),
                "vf": (),
            },
            desc_fourier_xyz_coil_cls=_FakeDescFourierXYZCoil,
            desc_fourier_order=3,
            sample_count=32,
        )


def test_runtime_import_fidelity_gate_rejects_underfit_desc_coils():
    underfit = export_simsopt_coil_groups_to_desc(
        {
            "tf": (),
            "banana": (_higher_order_coil(1.0e4, x_offset=0.8),),
            "proxy": (),
            "vf": (),
        },
        desc_fourier_xyz_coil_cls=_FakeDescFourierXYZCoil,
        desc_fourier_order=1,
        sample_count=64,
    )

    residuals = underfit.report.artifact_metadata.conversion_residuals
    assert residuals is not None
    assert residuals.max_fit_residual_m > DESC_RUNTIME_IMPORT_MAX_FIT_RESIDUAL_M
    with np.testing.assert_raises_regex(ValueError, "import fidelity exceeded"):
        _validate_desc_import_geometry_fidelity(underfit.report)

    faithful = export_simsopt_coil_groups_to_desc(
        {
            "tf": (),
            "banana": (_higher_order_coil(1.0e4, x_offset=0.8),),
            "proxy": (),
            "vf": (),
        },
        desc_fourier_xyz_coil_cls=_FakeDescFourierXYZCoil,
        desc_fourier_order=3,
        sample_count=64,
    )

    _validate_desc_import_geometry_fidelity(faithful.report)


def test_sample_desc_coilset_unique_coils_rejects_wrong_desc_grid_shape():
    desc_coilset = _FakeDescRuntimeCoilSet(
        coils=(
            _FakeDescRuntimeCoil(
                name="banana_000",
                current=1.0e4,
                coords_xyz=_coil(1.0e4, x_offset=0.8, sample_count=11).curve.gamma(),
            ),
        )
    )

    with np.testing.assert_raises_regex(ValueError, "optimized DESC coil samples"):
        sample_desc_coilset_unique_coils(
            desc_coilset,
            coil_group_counts={"banana": 1},
            sample_count=32,
        )


def test_sample_desc_coilset_unique_coils_uses_explicit_desc_grid_sample_count():
    desc_coilset = _FakeDescRuntimeCoilSet(
        coils=(
            _GridAwareDescRuntimeCoil(
                name="banana_000",
                current=1.0e4,
                radius=0.8,
            ),
        )
    )

    sampled = sample_desc_coilset_unique_coils(
        desc_coilset,
        coil_group_counts={"banana": 1},
        sample_count=32,
    )

    assert len(sampled) == 1
    assert sampled[0].coords_xyz.shape == (32, 3)
    assert sampled[0].current_A == 1.0e4


def test_coil_groups_from_stage2_partitions_uses_explicit_group_attributes():
    partitions = _Partitions(
        tf_coils=(_coil(-8.0e4, x_offset=0.9),),
        banana_coils=(_coil(1.0e4, x_offset=0.8),),
        proxy_coils=(_coil(5.0, x_offset=0.7),),
        vf_coils=(),
    )

    groups = coil_groups_from_stage2_partitions(partitions)

    assert list(groups) == ["tf", "banana", "proxy", "vf"]
    assert len(groups["tf"]) == 1
    assert len(groups["banana"]) == 1
    assert len(groups["proxy"]) == 1
    assert groups["vf"] == ()


def test_export_preserves_auxiliary_group_names_after_standard_groups():
    result = export_simsopt_coil_groups_to_desc(
        {
            "banana": (_coil(1.2e4, x_offset=0.8),),
            "auxiliary": (_coil(0.0, x_offset=1.2),),
        },
        desc_fourier_xyz_coil_cls=_FakeDescFourierXYZCoil,
        desc_fourier_order=3,
        sample_count=32,
    )

    assert result.report.group_order == ("banana", "auxiliary")
    assert result.report.group_counts == {"banana": 1, "auxiliary": 1}
    assert [entry.group for entry in result.report.entries] == [
        "banana",
        "auxiliary",
    ]
    assert [entry.name for entry in result.report.entries] == [
        "banana_000",
        "auxiliary_000",
    ]


def test_conversion_reports_record_desc_metadata_and_source_checksums(tmp_path):
    surface_path = tmp_path / "surf_opt_boozer_surface.json"
    field_path = tmp_path / "biot_savart_opt.json"
    surface_path.write_text('{"surface": true}\n', encoding="utf-8")
    field_path.write_text('{"field": true}\n', encoding="utf-8")
    artifact_metadata = DescBridgeArtifactMetadata(
        desc_optimizer_version="0.13.1",
        desc_commit="abc123def",
        source_artifact_checksums=desc_bridge_source_checksums(
            {
                "surface": surface_path,
                "field": field_path,
            }
        ),
    )
    exported = export_simsopt_coil_groups_to_desc(
        {
            "tf": (_coil(-8.0e4, x_offset=0.9),),
            "banana": (_coil(1.2e4, x_offset=0.8),),
        },
        desc_fourier_xyz_coil_cls=_FakeDescFourierXYZCoil,
        desc_fourier_order=3,
        sample_count=32,
        artifact_metadata=artifact_metadata,
    )

    expected_metadata_payload = {
        "desc_optimizer_version": "0.13.1",
        "desc_commit": "abc123def",
        "source_artifact_paths": {
            "surface": str(surface_path.resolve()),
            "field": str(field_path.resolve()),
        },
        "source_artifact_checksums": {
            "surface": _sha256(surface_path),
            "field": _sha256(field_path),
        },
    }
    exported_metadata = exported.report.to_json_dict()["artifact_metadata"]
    assert exported_metadata["desc_optimizer_version"] == (
        expected_metadata_payload["desc_optimizer_version"]
    )
    assert exported_metadata["desc_commit"] == expected_metadata_payload["desc_commit"]
    assert exported_metadata["source_artifact_paths"] == (
        expected_metadata_payload["source_artifact_paths"]
    )
    assert exported_metadata["source_artifact_checksums"] == (
        expected_metadata_payload["source_artifact_checksums"]
    )
    assert exported_metadata["conversion_residuals"]["max_fit_residual_m"] <= 1.0e-12
    assert exported_metadata["conversion_residuals"]["rms_fit_residual_m"] <= 1.0e-12
    assert exported_metadata["conversion_residuals"]["max_abs_length_delta_m"] <= (
        1.0e-12
    )
    assert exported_metadata["conversion_residuals"]["max_field_sample_delta_T"] <= (
        1.0e-12
    )
    sampled_coils = tuple(
        DescSampledCoil(
            group=entry.group,
            group_index=entry.group_index,
            name=entry.name,
            current_A=entry.current_A,
            coords_xyz=exported.desc_coils[entry.export_index].coords,
        )
        for entry in exported.report.entries
    )
    imported = import_desc_sampled_coils_to_simsopt(
        sampled_coils,
        simsopt_fourier_order=3,
        sample_count=32,
        coil_group_manifest=exported.report.group_counts,
        artifact_metadata=artifact_metadata,
        hardware_oracle_status="not_run",
    )

    imported_metadata = imported.report.to_json_dict()["artifact_metadata"]
    assert imported_metadata["desc_optimizer_version"] == (
        expected_metadata_payload["desc_optimizer_version"]
    )
    assert imported_metadata["desc_commit"] == expected_metadata_payload["desc_commit"]
    assert imported_metadata["source_artifact_paths"] == (
        expected_metadata_payload["source_artifact_paths"]
    )
    assert imported_metadata["source_artifact_checksums"] == (
        expected_metadata_payload["source_artifact_checksums"]
    )
    assert imported_metadata["conversion_residuals"]["max_fit_residual_m"] <= 1.0e-12
    assert imported_metadata["conversion_residuals"]["rms_fit_residual_m"] <= 1.0e-12
    assert imported_metadata["conversion_residuals"]["max_field_sample_delta_T"] is None


def test_conversion_reports_preserve_source_identity_metadata():
    banana_pack_metadata = DescBridgeBananaPackMetadata(
        finite_build_enabled=True,
        filaments_per_banana=2,
        numfilaments_n=1,
        numfilaments_b=2,
    )
    exported = export_simsopt_coil_groups_to_desc(
        {
            "tf": (
                _NamedSamplingCoil(
                    curve=_MutableSamplingCurve(),
                    current=Current(-8.0e4),
                    name="tf-source",
                ),
            ),
            "banana": (
                _NamedSamplingCoil(
                    curve=_MutableSamplingCurve(),
                    current=Current(1.2e4),
                    name="banana-source",
                ),
            ),
        },
        desc_fourier_xyz_coil_cls=_FakeDescFourierXYZCoil,
        desc_fourier_order=3,
        sample_count=32,
        source_nfp=5,
        source_stellarator_symmetry=True,
        banana_pack_metadata=banana_pack_metadata,
    )

    source_identity = exported.report.to_json_dict()["artifact_metadata"][
        "source_identity"
    ]
    assert source_identity == {
        "coil_names": ["tf-source", "banana-source"],
        "coil_group_manifest": [
            {"role": "tf", "start": 0, "count": 1},
            {"role": "banana", "start": 1, "count": 1},
        ],
        "nfp": 5,
        "stellarator_symmetry": True,
        "banana_pack_metadata": {
            "finite_build_enabled": True,
            "filaments_per_banana": 2,
            "numfilaments_n": 1,
            "numfilaments_b": 2,
        },
    }

    sampled_coils = tuple(
        DescSampledCoil(
            group=entry.group,
            group_index=entry.group_index,
            name=entry.name,
            current_A=entry.current_A,
            coords_xyz=exported.desc_coils[entry.export_index].coords,
        )
        for entry in exported.report.entries
    )
    imported = import_desc_sampled_coils_to_simsopt(
        sampled_coils,
        simsopt_fourier_order=3,
        sample_count=32,
        coil_group_manifest=exported.report.group_counts,
        artifact_metadata=exported.report.artifact_metadata,
        hardware_oracle_status="not_run",
    )

    assert (
        imported.report.to_json_dict()["artifact_metadata"]["source_identity"]
        == source_identity
    )


def test_source_identity_preserves_source_group_order_separate_from_export_order():
    exported = export_simsopt_coil_groups_to_desc(
        {
            "banana": (
                _NamedSamplingCoil(
                    curve=_MutableSamplingCurve(),
                    current=Current(1.2e4),
                    name="banana-source",
                ),
            ),
            "tf": (
                _NamedSamplingCoil(
                    curve=_MutableSamplingCurve(),
                    current=Current(-8.0e4),
                    name="tf-source",
                ),
            ),
        },
        desc_fourier_xyz_coil_cls=_FakeDescFourierXYZCoil,
        desc_fourier_order=3,
        sample_count=32,
    )

    assert exported.report.group_order == ("tf", "banana")
    source_identity = exported.report.to_json_dict()["artifact_metadata"][
        "source_identity"
    ]
    assert source_identity["coil_names"] == ["banana-source", "tf-source"]
    assert source_identity["coil_group_manifest"] == [
        {"role": "banana", "start": 0, "count": 1},
        {"role": "tf", "start": 1, "count": 1},
    ]


def test_banana_pack_metadata_rejects_inconsistent_filament_counts():
    with np.testing.assert_raises_regex(ValueError, "numfilaments_n"):
        DescBridgeBananaPackMetadata(
            finite_build_enabled=True,
            filaments_per_banana=3,
            numfilaments_n=1,
            numfilaments_b=2,
        )


def test_conversion_metadata_rejects_missing_source_artifact(tmp_path):
    with np.testing.assert_raises_regex(ValueError, "source artifact must be a file"):
        desc_bridge_source_checksums({"surface": tmp_path / "missing.json"})


def test_conversion_metadata_rejects_forged_source_checksum(tmp_path):
    surface_path = tmp_path / "surface.json"
    surface_path.write_text('{"surface": true}\n', encoding="utf-8")
    with np.testing.assert_raises_regex(ValueError, "SHA-256"):
        DescBridgeArtifactMetadata(
            source_artifact_checksums=(
                DescBridgeSourceChecksum(
                    name="surface",
                    source_path=str(surface_path.resolve()),
                    sha256="not-a-sha",
                ),
            )
        )
    with np.testing.assert_raises_regex(ValueError, "does not match"):
        DescBridgeArtifactMetadata(
            source_artifact_checksums=(
                DescBridgeSourceChecksum(
                    name="surface",
                    source_path=str(surface_path.resolve()),
                    sha256="a" * 64,
                ),
            )
        )


def test_import_sampled_desc_coils_preserves_groups_currents_and_coordinates():
    tf_coil = _coil(-8.0e4, x_offset=0.9)
    banana_positive = _coil(1.2e4, x_offset=0.8)
    banana_negative = _coil(-9.5e3, x_offset=0.82)
    exported = export_simsopt_coil_groups_to_desc(
        {
            "tf": (tf_coil,),
            "banana": (banana_positive, banana_negative),
            "proxy": (),
            "vf": (),
        },
        desc_fourier_xyz_coil_cls=_FakeDescFourierXYZCoil,
        desc_fourier_order=3,
        sample_count=32,
    )
    sampled_coils = tuple(
        DescSampledCoil(
            group=entry.group,
            group_index=entry.group_index,
            name=entry.name,
            current_A=entry.current_A,
            coords_xyz=exported.desc_coils[entry.export_index].coords,
        )
        for entry in exported.report.entries
    )

    imported = import_desc_sampled_coils_to_simsopt(
        sampled_coils,
        simsopt_fourier_order=3,
        sample_count=32,
        coil_group_manifest=exported.report.group_counts,
        hardware_oracle_status="not_run",
    )

    assert imported.report.group_order == ("tf", "banana", "proxy", "vf")
    assert [entry.group for entry in imported.report.entries] == [
        "tf",
        "banana",
        "banana",
    ]
    assert [entry.current_A for entry in imported.report.entries] == [
        -8.0e4,
        1.2e4,
        -9.5e3,
    ]
    assert len(imported.coil_groups["tf"]) == 1
    assert len(imported.coil_groups["banana"]) == 2
    assert imported.coil_groups["proxy"] == ()
    assert imported.coil_groups["vf"] == ()
    assert imported.report.group_counts == {
        "tf": 1,
        "banana": 2,
        "proxy": 0,
        "vf": 0,
    }
    assert imported.report.current_sign_counts == {
        "negative": 2,
        "zero": 0,
        "positive": 1,
    }
    assert imported.report.to_json_dict()["coil_conventions"] == (
        coil_convention_report()
    )
    assert imported.report.min_coil_distance_delta_m is not None
    assert len(imported.biot_savart.coils) == 3
    expected_lengths = [periodic_length(coil.coords_xyz) for coil in sampled_coils]
    np.testing.assert_allclose(
        [entry.source_length_m for entry in imported.report.entries],
        expected_lengths,
        atol=1.0e-14,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        [entry.reconstructed_length_m for entry in imported.report.entries],
        expected_lengths,
        atol=1.0e-12,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        [entry.length_delta_m for entry in imported.report.entries],
        np.zeros(len(imported.report.entries)),
        atol=1.0e-12,
        rtol=0.0,
    )
    expected_min_distance = min_pairwise_periodic_distance(
        [coil.coords_xyz for coil in sampled_coils]
    )
    assert expected_min_distance is not None
    np.testing.assert_allclose(
        imported.report.source_min_coil_distance_m,
        expected_min_distance,
        atol=1.0e-14,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        imported.report.reconstructed_min_coil_distance_m,
        expected_min_distance,
        atol=1.0e-12,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        imported.report.min_coil_distance_delta_m,
        0.0,
        atol=1.0e-12,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        imported.coil_groups["banana"][0].curve.gamma(),
        sampled_coils[1].coords_xyz,
        atol=1.0e-12,
        rtol=0.0,
    )
    probe_points = np.asarray(
        [
            (0.78, 0.0, 0.0),
            (0.84, 0.02, 0.01),
            (0.91, -0.02, -0.01),
            (0.95, 0.03, 0.0),
        ],
        dtype=float,
    )
    source_biot_savart = BiotSavart((tf_coil, banana_positive, banana_negative))
    source_biot_savart.set_points(probe_points)
    imported.biot_savart.set_points(probe_points)
    np.testing.assert_allclose(
        imported.biot_savart.B(),
        source_biot_savart.B(),
        atol=1.0e-12,
        rtol=0.0,
    )


def test_import_rejects_group_manifest_mismatches():
    sampled_coils = (
        DescSampledCoil(
            group="banana",
            group_index=0,
            name="banana_000",
            current_A=1.0e4,
            coords_xyz=_coil(1.0e4, x_offset=0.8).curve.gamma(),
        ),
    )

    with np.testing.assert_raises_regex(ValueError, "missing from the original"):
        import_desc_sampled_coils_to_simsopt(
            sampled_coils,
            simsopt_fourier_order=3,
            sample_count=32,
            coil_group_manifest={"tf": 0},
            hardware_oracle_status="not_run",
        )

    with np.testing.assert_raises_regex(ValueError, "does not match"):
        import_desc_sampled_coils_to_simsopt(
            sampled_coils,
            simsopt_fourier_order=3,
            sample_count=32,
            coil_group_manifest={"banana": 2},
            hardware_oracle_status="not_run",
        )
    duplicate_index_coils = (
        sampled_coils[0],
        DescSampledCoil(
            group="banana",
            group_index=0,
            name="banana_001",
            current_A=-1.0e4,
            coords_xyz=_coil(-1.0e4, x_offset=0.82).curve.gamma(),
        ),
    )
    with np.testing.assert_raises_regex(ValueError, "indices"):
        import_desc_sampled_coils_to_simsopt(
            duplicate_index_coils,
            simsopt_fourier_order=3,
            sample_count=32,
            coil_group_manifest={"banana": 2},
            hardware_oracle_status="not_run",
        )


def test_import_sampled_desc_coils_preserves_auxiliary_group():
    sampled_coils = (
        DescSampledCoil(
            group="auxiliary",
            group_index=0,
            name="auxiliary_000",
            current_A=0.0,
            coords_xyz=_coil(0.0, x_offset=1.2).curve.gamma(),
        ),
    )

    imported = import_desc_sampled_coils_to_simsopt(
        sampled_coils,
        simsopt_fourier_order=3,
        sample_count=32,
        hardware_oracle_status="not_run",
    )

    assert imported.report.group_order == ("auxiliary",)
    assert imported.report.group_counts == {"auxiliary": 1}
    assert imported.report.current_sign_counts == {
        "negative": 0,
        "zero": 1,
        "positive": 0,
    }
    assert imported.report.to_json_dict()["coil_conventions"] == (
        coil_convention_report()
    )
    assert len(imported.coil_groups["auxiliary"]) == 1


def test_import_rejects_hardware_clean_status_without_oracle_evidence(tmp_path):
    with np.testing.assert_raises_regex(ValueError, "final_oracle_evidence_path"):
        validate_hardware_oracle_binding(
            hardware_oracle_status="passed",
            final_oracle_evidence_path=None,
        )
    with np.testing.assert_raises_regex(ValueError, "missing"):
        validate_hardware_oracle_binding(
            hardware_oracle_status="passed",
            final_oracle_evidence_path="/tmp/desc_joint_missing_oracle_report.json",
        )

    oracle_path = tmp_path / "desc_joint_existing_oracle_report.json"
    oracle_path.write_text("{}\n", encoding="utf-8")
    source_artifact_path = tmp_path / "exported_surface.json"
    source_artifact_path.write_text('{"surface": true}\n', encoding="utf-8")
    source_artifact_metadata = DescBridgeArtifactMetadata(
        source_artifact_checksums=desc_bridge_source_checksums(
            {"surface": source_artifact_path}
        )
    )
    exported_artifact_path = tmp_path / "exported_biot_savart.json"
    exported_artifact_path.write_text('{"field": true}\n', encoding="utf-8")
    exported_artifact_paths = (str(exported_artifact_path),)
    with np.testing.assert_raises_regex(ValueError, "schema_version"):
        validate_hardware_oracle_binding(
            hardware_oracle_status="passed",
            final_oracle_evidence_path=str(oracle_path),
            artifact_metadata=source_artifact_metadata,
            exported_artifact_paths=exported_artifact_paths,
        )
    valid_oracle_path = _write_final_oracle_evidence(
        tmp_path / "desc_joint_valid_oracle_report.json",
        source_artifact_checksums=source_artifact_metadata.checksum_map(),
        exported_artifact_paths=exported_artifact_paths,
    )
    with np.testing.assert_raises_regex(ValueError, "checksum-bound"):
        validate_hardware_oracle_binding(
            hardware_oracle_status="passed",
            final_oracle_evidence_path=str(valid_oracle_path),
            exported_artifact_paths=exported_artifact_paths,
        )
    with np.testing.assert_raises_regex(ValueError, "at least one exported artifact"):
        validate_hardware_oracle_binding(
            hardware_oracle_status="passed",
            final_oracle_evidence_path=str(valid_oracle_path),
            artifact_metadata=source_artifact_metadata,
        )
    validate_hardware_oracle_binding(
        hardware_oracle_status="passed",
        final_oracle_evidence_path=str(valid_oracle_path),
        artifact_metadata=source_artifact_metadata,
        exported_artifact_paths=exported_artifact_paths,
    )
    with np.testing.assert_raises_regex(ValueError, "do not match"):
        validate_hardware_oracle_binding(
            hardware_oracle_status="passed",
            final_oracle_evidence_path=str(valid_oracle_path),
            artifact_metadata=DescBridgeArtifactMetadata(
                source_artifact_checksums=desc_bridge_source_checksums(
                    {"surface": valid_oracle_path}
                )
            ),
            exported_artifact_paths=exported_artifact_paths,
        )
