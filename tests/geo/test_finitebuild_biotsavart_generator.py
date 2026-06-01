import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from simsopt._core.optimizable import load
from simsopt.field import BiotSavart, Coil, Current, coils_via_symmetries
from simsopt.geo import CurveXYZFourier


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_ROOT = REPO_ROOT / "examples" / "single_stage_optimization"
WRAPPER_PATH = EXAMPLES_ROOT / "generate_finitebuild_biotsavart.py"
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

from banana_opt.coil_groups import build_contiguous_manifest  # noqa: E402
from banana_opt.finitebuild_export import (  # noqa: E402
    FiniteBuildExportConfig,
    default_output_path,
    export_finitebuild_biot_savart,
    metadata_output_path,
    normalize_biot_savart_source,
)


def _make_curve(index: int) -> CurveXYZFourier:
    curve = CurveXYZFourier(24, 1)
    curve.set("xc(0)", 0.02 * float(index))
    curve.set("xc(1)", 0.10)
    curve.set("ys(1)", 0.10)
    curve.fix_all()
    return curve


def _make_current(current_A: float) -> Current:
    current = Current(float(current_A))
    current.fix_all()
    return current


def _make_coil(index: int, current_A: float) -> Coil:
    return Coil(_make_curve(index), _make_current(current_A))


def _make_hbt_banana_coils(master_current_A: float) -> tuple[Coil, ...]:
    return tuple(
        coils_via_symmetries(
            [_make_curve(200)],
            [_make_current(master_current_A)],
            nfp=5,
            stellsym=True,
        )
    )


def _make_hbt_biot_savart(
    *,
    master_banana_current_A: float = -1.0e4,
    include_proxy_vf: bool = False,
) -> BiotSavart:
    tf_coils = [_make_coil(index, -8.0e4) for index in range(20)]
    banana_coils = list(_make_hbt_banana_coils(master_banana_current_A))
    proxy_coils = [_make_coil(400, -350.0)] if include_proxy_vf else []
    vf_coils = (
        [
            _make_coil(500 + index, 20.0 if index % 2 == 0 else -20.0)
            for index in range(20)
        ]
        if include_proxy_vf
        else []
    )
    return BiotSavart(tf_coils + banana_coils + proxy_coils + vf_coils)


def _save_biot_savart(path: Path, biot_savart: BiotSavart) -> Path:
    biot_savart.save(str(path))
    return path


def _coil_currents(biot_savart: BiotSavart) -> list[float]:
    return [float(coil.current.get_value()) for coil in biot_savart.coils]


def _finitebuild_config(source_path: Path, **overrides) -> FiniteBuildExportConfig:
    values = {
        "biot_savart_file": source_path,
        "output": overrides.pop("output", None),
        "numfilaments_n": overrides.pop("numfilaments_n", 2),
        "numfilaments_b": overrides.pop("numfilaments_b", 3),
        "gapsize_n": overrides.pop("gapsize_n", 0.003),
        "gapsize_b": overrides.pop("gapsize_b", 0.004),
        "rotation_order": overrides.pop("rotation_order", None),
        "frame": overrides.pop("frame", "centroid"),
        "banana_current_A": overrides.pop("banana_current_A", None),
        "stage2_results": overrides.pop("stage2_results", None),
        "finite_current_mode": overrides.pop("finite_current_mode", None),
        "num_tf_coils": overrides.pop("num_tf_coils", 20),
        "nfp": overrides.pop("nfp", 5),
        "stellsym": overrides.pop("stellsym", True),
        "overwrite": overrides.pop("overwrite", False),
    }
    assert not overrides
    return FiniteBuildExportConfig(**values)


def test_vacuum_profile_fallback_converts_and_preserves_current(tmp_path):
    source = _make_hbt_biot_savart(master_banana_current_A=-1.0e4)
    source_path = _save_biot_savart(tmp_path / "biot_savart_opt.json", source)

    result = export_finitebuild_biot_savart(_finitebuild_config(source_path))

    assert result.output_counts == {
        "tf": 20,
        "banana": 60,
        "proxy": 0,
        "vf": 0,
        "total": 80,
    }
    assert result.source_counts == {
        "tf": 20,
        "banana": 10,
        "proxy": 0,
        "vf": 0,
        "total": 30,
    }
    assert result.output_path.name == "biot_savart_finitebuild_opt.json"
    assert result.metadata_path == metadata_output_path(result.output_path)

    output = load(str(result.output_path))
    output_currents = _coil_currents(output)
    source_banana_currents = _coil_currents(source)[20:30]
    nfilaments = 6
    for symmetry_index, source_current_A in enumerate(source_banana_currents):
        start = 20 + symmetry_index * nfilaments
        stop = start + nfilaments
        assert sum(output_currents[start:stop]) == pytest.approx(source_current_A)

    output.set_points(np.array([[0.55, 0.02, 0.03]]))
    field = output.B()
    assert field.shape == (1, 3)
    assert np.all(np.isfinite(field))

    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["EXPORT_SCOPE"] == "field_evaluation_only_not_optimization_seed"
    assert metadata["FINITE_CURRENT_MODE"] == "vacuum"
    assert metadata["BANANA_CURRENT_OVERRIDE"] is False
    assert metadata["BANANA_SOURCE_TOTAL_CURRENT_A"] == pytest.approx(
        metadata["BANANA_OUTPUT_TOTAL_CURRENT_A"]
    )
    assert metadata["FINITEBUILD_FILAMENT_SETTINGS"]["numfilaments_n"] == 2
    assert metadata["OUTPUT_COIL_COUNTS"]["banana"] == 60


def test_finite_current_fallback_requires_mode_for_ambiguous_51_coil_artifact(tmp_path):
    source_path = _save_biot_savart(
        tmp_path / "biotsavart.json",
        _make_hbt_biot_savart(include_proxy_vf=True),
    )

    with pytest.raises(ValueError, match="ambiguous"):
        export_finitebuild_biot_savart(_finitebuild_config(source_path))

    result = export_finitebuild_biot_savart(
        _finitebuild_config(
            source_path,
            finite_current_mode="wataru_proxy_field",
            output=tmp_path / "explicit_output.json",
        )
    )
    assert result.output_counts == {
        "tf": 20,
        "banana": 60,
        "proxy": 1,
        "vf": 20,
        "total": 101,
    }
    output_currents = _coil_currents(load(str(result.output_path)))
    source_currents = _coil_currents(load(str(source_path)))
    assert output_currents[:20] == pytest.approx(source_currents[:20])
    assert output_currents[80:] == pytest.approx(source_currents[30:])


def test_metadata_free_profile_selection_rejects_count_mismatch(tmp_path):
    source_path = _save_biot_savart(
        tmp_path / "vacuum_biot_savart.json",
        _make_hbt_biot_savart(),
    )

    with pytest.raises(ValueError, match="expects 51 coils"):
        export_finitebuild_biot_savart(
            _finitebuild_config(
                source_path,
                finite_current_mode="wataru_proxy_field",
            )
        )


def test_metadata_free_profile_selection_rejects_unsupported_count(tmp_path):
    source = BiotSavart([*_make_hbt_biot_savart().coils, _make_coil(700, 12.0)])
    source_path = _save_biot_savart(tmp_path / "unsupported_biot_savart.json", source)

    with pytest.raises(
        ValueError,
        match="Cannot partition metadata-free artifact with 31 coils",
    ):
        export_finitebuild_biot_savart(_finitebuild_config(source_path))


def test_stage2_results_manifest_controls_nonstandard_partition(tmp_path):
    source = BiotSavart(
        [
            _make_coil(0, -8.0e4),
            _make_coil(1, -8.0e4),
            _make_coil(2, -1200.0),
            _make_coil(3, 33.0),
            _make_coil(4, -44.0),
        ]
    )
    source_path = _save_biot_savart(tmp_path / "seed.json", source)
    manifest = build_contiguous_manifest(
        num_tf_coils=2,
        num_banana_coils=1,
        num_proxy_coils=1,
        num_vf_coils=1,
    )
    results_path = tmp_path / "results.json"
    results_path.write_text(
        json.dumps(
            {
                "FINITE_CURRENT_MODE": "vacuum",
                "NUM_TF_COILS": 2,
                "COIL_GROUPS": manifest.to_json_payload(),
            }
        ),
        encoding="utf-8",
    )

    result = export_finitebuild_biot_savart(
        _finitebuild_config(
            source_path,
            stage2_results=results_path,
            num_tf_coils=2,
            nfp=1,
            stellsym=False,
            numfilaments_n=2,
            numfilaments_b=2,
        )
    )

    assert result.source_counts == {
        "tf": 2,
        "banana": 1,
        "proxy": 1,
        "vf": 1,
        "total": 5,
    }
    assert result.output_counts == {
        "tf": 2,
        "banana": 4,
        "proxy": 1,
        "vf": 1,
        "total": 8,
    }
    output_currents = _coil_currents(load(str(result.output_path)))
    assert output_currents[:2] == pytest.approx([-8.0e4, -8.0e4])
    assert sum(output_currents[2:6]) == pytest.approx(-1200.0)
    assert output_currents[6:] == pytest.approx([33.0, -44.0])

    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["PARTITION_SOURCE"] == "stage2_results"
    assert metadata["SOURCE_MANIFEST_IS_LEGACY_INFERRED"] is False
    assert metadata["SOURCE_STAGE2_RESULTS_PATH"] == str(results_path.resolve())
    assert "not a new physics model" in metadata["PROVENANCE_NOTE"]


def test_banana_current_override_changes_only_finitebuild_banana_total(tmp_path):
    source = _make_hbt_biot_savart(
        master_banana_current_A=-1.0e4, include_proxy_vf=True
    )
    source_path = _save_biot_savart(tmp_path / "biot_savart.json", source)

    result = export_finitebuild_biot_savart(
        _finitebuild_config(
            source_path,
            finite_current_mode="jhalpern30_proxy_field",
            banana_current_A=2400.0,
        )
    )

    output_currents = _coil_currents(load(str(result.output_path)))
    assert sum(output_currents[20:26]) == pytest.approx(2400.0)
    assert sum(output_currents[26:32]) == pytest.approx(-2400.0)
    assert output_currents[:20] == pytest.approx(_coil_currents(source)[:20])
    assert output_currents[80:] == pytest.approx(_coil_currents(source)[30:])
    assert _coil_currents(source)[20] == pytest.approx(-1.0e4)
    assert result.current_override is True


def test_default_output_naming_and_input_overwrite_rejection(tmp_path):
    assert (
        default_output_path(Path("/tmp/biot_savart_opt.json")).name
        == "biot_savart_finitebuild_opt.json"
    )
    assert (
        default_output_path(Path("/tmp/biotsavart.json")).name
        == "biotsavart_finitebuild.json"
    )
    assert (
        default_output_path(Path("/tmp/source.json")).name == "source_finitebuild.json"
    )

    source_path = _save_biot_savart(
        tmp_path / "biot_savart.json",
        _make_hbt_biot_savart(),
    )
    with pytest.raises(ValueError, match="must not overwrite"):
        export_finitebuild_biot_savart(
            _finitebuild_config(source_path, output=source_path)
        )

    existing_output = tmp_path / "existing.json"
    existing_output.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        export_finitebuild_biot_savart(
            _finitebuild_config(source_path, output=existing_output)
        )


def test_normalize_biot_savart_source_accepts_boozer_style_wrapper():
    biot_savart = _make_hbt_biot_savart()

    assert normalize_biot_savart_source(biot_savart) is biot_savart
    assert (
        normalize_biot_savart_source(SimpleNamespace(biotsavart=biot_savart))
        is biot_savart
    )
    from_coils = normalize_biot_savart_source(SimpleNamespace(coils=biot_savart.coils))
    assert len(from_coils.coils) == len(biot_savart.coils)


def test_direct_wrapper_cli_writes_output_and_metadata(tmp_path):
    source = BiotSavart(
        [
            _make_coil(0, -8.0e4),
            _make_coil(1, -8.0e4),
            _make_coil(2, -900.0),
            _make_coil(3, 11.0),
            _make_coil(4, -12.0),
        ]
    )
    source_path = _save_biot_savart(tmp_path / "seed.json", source)
    output_path = tmp_path / "converted.json"
    manifest = build_contiguous_manifest(
        num_tf_coils=2,
        num_banana_coils=1,
        num_proxy_coils=1,
        num_vf_coils=1,
    )
    results_path = tmp_path / "results.json"
    results_path.write_text(
        json.dumps(
            {
                "FINITE_CURRENT_MODE": "vacuum",
                "NUM_TF_COILS": 2,
                "COIL_GROUPS": manifest.to_json_payload(),
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(WRAPPER_PATH),
            str(source_path),
            "--output",
            str(output_path),
            "--numfilaments-n",
            "2",
            "--numfilaments-b",
            "2",
            "--gapsize-n",
            "0.003",
            "--gapsize-b",
            "0.004",
            "--stage2-results",
            str(results_path),
            "--num-tf-coils",
            "2",
            "--nfp",
            "1",
            "--no-stellsym",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert output_path.is_file()
    assert metadata_output_path(output_path).is_file()
    assert f"output: {output_path.resolve()}" in completed.stdout
    assert "banana_total_current_A: source -900, output -900" in completed.stdout
    output_currents = _coil_currents(load(str(output_path)))
    assert len(output_currents) == 8
    assert sum(output_currents[2:6]) == pytest.approx(-900.0)
