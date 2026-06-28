import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_ROOT = REPO_ROOT / "examples" / "single_stage_optimization"
EXAMPLES_ROOT_STR = str(EXAMPLES_ROOT)
if EXAMPLES_ROOT_STR not in sys.path:
    sys.path.insert(0, EXAMPLES_ROOT_STR)

from banana_opt.desc_joint_field_inventory import (  # noqa: E402
    load_desc_joint_field_inventory,
)
from banana_opt.desc_joint_seed_manifest import (  # noqa: E402
    DESC_JOINT_SEED_MANIFEST_SCHEMA_VERSION,
    load_desc_joint_seed_manifest,
)


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _write_boozer_surface_fixture(path: Path) -> Path:
    return _write_json(
        path,
        {
            "@class": "BoozerSurface",
            "field": {"@class": "BiotSavart"},
        },
    )


def _write_biot_savart_fixture(
    path: Path,
    *,
    curve_classes: tuple[str | None, ...],
    currents_A: tuple[float, ...] = (-8.0e4, 1.2e4, 0.0),
) -> Path:
    if len(curve_classes) != len(currents_A):
        raise ValueError("curve_classes length must match currents_A length.")
    objects: dict[str, object] = {}
    coil_refs: list[dict[str, str]] = []
    for index, current_A in enumerate(currents_A, start=1):
        current_name = f"Current{index}"
        coil_name = f"Coil{index}"
        objects[current_name] = {
            "@class": "Current",
            "current": current_A,
        }
        coil_payload: dict[str, object] = {
            "@class": "Coil",
            "current": {"$type": "ref", "value": current_name},
        }
        curve_class = curve_classes[index - 1]
        if curve_class is not None:
            curve_name = f"Curve{index}"
            objects[curve_name] = {"@class": curve_class}
            coil_payload["curve"] = {"$type": "ref", "value": curve_name}
        objects[coil_name] = coil_payload
        coil_refs.append({"$type": "ref", "value": coil_name})
    objects["BiotSavart1"] = {
        "@class": "BiotSavart",
        "coils": coil_refs,
    }
    return _write_json(
        path,
        {
            "@class": "SIMSON",
            "simsopt_objs": objects,
        },
    )


def _write_materialized_cws_results(path: Path) -> Path:
    return _write_json(
        path,
        {
            "SINGLE_STAGE_BANANA_GEOMETRY_MODE": "materialized_cws",
            "COIL_GROUPS": [
                {"role": "tf", "start": 0, "count": 1},
                {"role": "banana", "start": 1, "count": 2},
            ],
        },
    )


def _write_manifest(
    path: Path,
    *,
    surface_path: Path,
    field_path: Path,
    source_results_path: Path,
) -> Path:
    return _write_json(
        path,
        {
            "schema_version": DESC_JOINT_SEED_MANIFEST_SCHEMA_VERSION,
            "candidates": [
                {
                    "label": "slidclean_chomp",
                    "group": "slid_clean",
                    "surface": str(surface_path),
                    "field": str(field_path),
                    "surface_kind": "boozer_surface",
                    "source_results": str(source_results_path),
                },
            ],
        },
    )


def _add_unreferenced_curve_objects(path: Path) -> None:
    field_payload = json.loads(path.read_text(encoding="utf-8"))
    field_objects = field_payload["simsopt_objs"]
    assert isinstance(field_objects, dict)
    field_objects["UnusedCurveCWSFourierCPP"] = {
        "@class": "CurveCWSFourierCPP",
    }
    field_objects["UnusedCurveXYZFourier"] = {"@class": "CurveXYZFourier"}
    _write_json(path, field_payload)


def test_desc_joint_field_inventory_counts_only_referenced_field_curves(tmp_path):
    field_path = tmp_path / "biot_savart.json"
    _write_biot_savart_fixture(
        field_path,
        curve_classes=(
            "CurveXYZFourier",
            "CurveCWSFourierCPP",
            None,
        ),
    )
    _add_unreferenced_curve_objects(field_path)

    inventory = load_desc_joint_field_inventory(field_path)

    assert inventory.coil_count == 3
    assert inventory.cws_curve_count == 1
    assert inventory.xyz_curve_count == 1
    assert inventory.current_values_A == (-8.0e4, 1.2e4, 0.0)


def test_materialized_cws_manifest_rejects_flattened_xyz_field(tmp_path):
    surface_path = tmp_path / "surf_chomp_boozer_surface.json"
    field_path = tmp_path / "biot_savart_opt.json"
    source_results_path = tmp_path / "results.json"
    _write_boozer_surface_fixture(surface_path)
    _write_biot_savart_fixture(
        field_path,
        curve_classes=(
            "CurveXYZFourier",
            "CurveXYZFourier",
            "CurveXYZFourier",
        ),
    )
    _write_materialized_cws_results(source_results_path)
    manifest_path = _write_manifest(
        tmp_path / "seed_manifest.json",
        surface_path=surface_path,
        field_path=field_path,
        source_results_path=source_results_path,
    )

    with pytest.raises(ValueError, match="0 CurveCWSFourierCPP"):
        load_desc_joint_seed_manifest(manifest_path)


def test_materialized_cws_manifest_ignores_unreferenced_cws_objects(tmp_path):
    surface_path = tmp_path / "surf_chomp_boozer_surface.json"
    field_path = tmp_path / "biot_savart_opt.json"
    source_results_path = tmp_path / "results.json"
    _write_boozer_surface_fixture(surface_path)
    _write_biot_savart_fixture(
        field_path,
        curve_classes=(
            "CurveXYZFourier",
            "CurveXYZFourier",
            "CurveXYZFourier",
        ),
    )
    _add_unreferenced_curve_objects(field_path)
    _write_materialized_cws_results(source_results_path)
    manifest_path = _write_manifest(
        tmp_path / "seed_manifest.json",
        surface_path=surface_path,
        field_path=field_path,
        source_results_path=source_results_path,
    )

    with pytest.raises(ValueError, match="0 CurveCWSFourierCPP"):
        load_desc_joint_seed_manifest(manifest_path)


def test_materialized_cws_manifest_accepts_referenced_cws_field(tmp_path):
    surface_path = tmp_path / "surf_chomp_boozer_surface.json"
    field_path = tmp_path / "slid_cws_field_chomp.json"
    source_results_path = tmp_path / "results.json"
    _write_boozer_surface_fixture(surface_path)
    _write_biot_savart_fixture(
        field_path,
        curve_classes=(
            "CurveXYZFourier",
            "CurveCWSFourierCPP",
            "CurveCWSFourierCPP",
        ),
    )
    _write_materialized_cws_results(source_results_path)
    manifest_path = _write_manifest(
        tmp_path / "seed_manifest.json",
        surface_path=surface_path,
        field_path=field_path,
        source_results_path=source_results_path,
    )

    candidate = load_desc_joint_seed_manifest(manifest_path).candidate_by_label(
        "slidclean_chomp"
    )

    assert candidate.coil_group_source == "source_results"
    assert [coil_group.to_json_dict() for coil_group in candidate.coil_groups] == [
        {"name": "tf", "count": 1},
        {"name": "banana", "count": 2},
    ]
