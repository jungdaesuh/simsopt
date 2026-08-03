import json
import sys
from pathlib import Path

import numpy as np


HBT_DIR = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "single_stage_optimization"
    / "HBT_BANANA"
)
sys.path.insert(0, str(HBT_DIR / "utils"))

from hardware_metrics import (  # noqa: E402
    boozer_json_vacuum_lineage,
    curve_poloidal_half_extent,
    surface_shape_metrics,
    surface_vessel_clearance,
)


class _Curve:
    def __init__(self, gamma):
        self._gamma = np.asarray(gamma, dtype=float)

    def gamma(self):
        return self._gamma


class _Surface:
    def __init__(self, gamma):
        self._gamma = np.asarray(gamma, dtype=float)

    def gamma(self):
        return self._gamma

    def volume(self):
        return 0.09

    def major_radius(self):
        return 0.92

    def minor_radius(self):
        return 0.15


def test_curve_poloidal_half_extent_uses_inboard_midplane_convention():
    curve = _Curve([
        [0.903 - 0.142, 0.0, 0.0],
        [0.903 - 0.142 / np.sqrt(2.0), 0.0, 0.142 / np.sqrt(2.0)],
        [0.903, 0.0, 0.142],
    ])

    extent = curve_poloidal_half_extent(curve, winding_major_radius=0.903)

    assert extent == np.pi / 2.0


def test_surface_vessel_clearance_reports_minimum_positive_margin():
    surface = _Surface(np.array([
        [[0.976 + 0.10, 0.0, 0.0], [0.976, 0.0, 0.15]],
        [[0.976 - 0.18, 0.0, 0.0], [0.976, 0.0, -0.12]],
    ]))

    clearance = surface_vessel_clearance(
        surface, vessel_major_radius=0.976, vessel_minor_radius=0.222
    )

    assert np.isclose(clearance, 0.042)


def test_surface_shape_metrics_uses_surface_scalar_accessors():
    surface = _Surface(np.zeros((1, 1, 3)))

    assert surface_shape_metrics(surface) == {
        "final_volume": 0.09,
        "surface_major_radius_m": 0.92,
        "surface_minor_radius_m": 0.15,
    }


def test_boozer_json_vacuum_lineage_accepts_plain_boozer_surface(tmp_path):
    path = tmp_path / "boozersurface.json"
    path.write_text(json.dumps({
        "@module": "simsopt._core.json",
        "@class": "SIMSON",
        "simsopt_objs": {
            "BoozerSurface1": {
                "@module": "simsopt.geo.boozersurface",
                "@class": "BoozerSurface",
            }
        },
    }))

    lineage = boozer_json_vacuum_lineage(path)

    assert lineage["vacuum_lineage_ok"] is True
    assert lineage["boozer_surface_module"] == "simsopt.geo.boozersurface"
    assert lineage["boozer_json_has_I_field"] is False
    assert lineage["boozer_json_references_finite_i"] is False


def test_boozer_json_vacuum_lineage_rejects_finite_current_payload(tmp_path):
    path = tmp_path / "boozersurface.json"
    path.write_text(json.dumps({
        "simsopt_objs": {
            "BoozerSurfaceFiniteI1": {
                "@module": "banana_opt.boozer_finite_current",
                "@class": "BoozerSurfaceFiniteI",
                "I": 0.0,
            }
        },
    }))

    lineage = boozer_json_vacuum_lineage(path)

    assert lineage["vacuum_lineage_ok"] is False
    assert lineage["boozer_json_has_I_field"] is True
    assert lineage["boozer_json_references_finite_i"] is True


def test_boozer_json_vacuum_lineage_ignores_unrelated_I_metadata(tmp_path):
    path = tmp_path / "boozersurface.json"
    path.write_text(json.dumps({
        "simsopt_objs": {
            "BoozerSurface1": {
                "@module": "simsopt.geo.boozersurface",
                "@class": "BoozerSurface",
                "provenance": {"I": "metadata-not-enclosed-current"},
            }
        },
    }))

    lineage = boozer_json_vacuum_lineage(path)

    assert lineage["vacuum_lineage_ok"] is True
    assert lineage["boozer_json_has_I_field"] is False
    assert lineage["boozer_json_references_finite_i"] is False
