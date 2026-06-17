"""Tests for CAD SDF-backed hardware keep-out steering."""

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

EXAMPLES_ROOT = (
    Path(__file__).resolve().parents[2] / "examples" / "single_stage_optimization"
)
EXAMPLES_ROOT_STR = str(EXAMPLES_ROOT)
if EXAMPLES_ROOT_STR not in sys.path:
    sys.path.insert(0, EXAMPLES_ROOT_STR)

from banana_opt.hardware_contracts import (  # noqa: E402
    HARDWARE_KEEPOUT_MIN_DISTANCE_M,
    TYPE_KK_OUTER_CHANNEL_HALF_DEPTH_NORMAL_M,
)
from banana_opt.hardware_keepout import (  # noqa: E402
    CurveHardwareKeepout,
    CurveHardwareSdfKeepout,
    hardware_sdf_metadata,
    load_hardware_sdf,
)
from simsopt.geo import CurveXYZFourier  # noqa: E402

SIGNED_TEST_METHOD = "watertight_contains_trimesh_nearest"
COMPONENT_UNION_TEST_METHOD = "component_union_watertight_contains_trimesh_nearest"


def _circle_curve(radius=1.0, order=3, quadpoints=64, seed=None):
    curve = CurveXYZFourier(quadpoints, order)
    curve.x = np.zeros(curve.dof_size)
    curve.set("xc(1)", radius)
    curve.set("ys(1)", radius)
    if seed is not None:
        rng = np.random.default_rng(seed)
        curve.x = curve.x + 1e-3 * rng.standard_normal(curve.dof_size)
    return curve


def _plane_sdf_grid(origin, spacing, shape, plane_x):
    x = origin[0] + spacing * np.arange(shape[0])
    y = origin[1] + spacing * np.arange(shape[1])
    z = origin[2] + spacing * np.arange(shape[2])
    xx, _, _ = np.meshgrid(x, y, z, indexing="ij")
    return plane_x - xx


def _write_sdf_payload(
    root,
    *,
    grid,
    origin,
    spacing,
    effective_margin,
    units="m",
    static_keys=("sensors", "frame", "sample"),
    documented_gate_only=None,
    covered_by_other_in_loop=None,
    sign_method=SIGNED_TEST_METHOD,
    safety_margin=None,
    error_budget=None,
    group_effective_margin=None,
    narrow_band=0.02,
    group_narrow_band=None,
    glb_path=None,
):
    data_path = root / "hardware_sdf.npz"
    np.savez(data_path, sensors=grid)
    data_sha = hashlib.sha256(data_path.read_bytes()).hexdigest()
    if documented_gate_only is None:
        documented_gate_only = {
            "frame": {
                "reason": "not represented in this test SDF",
                "covered_by": "hardware_contact_report oracle",
            },
            "sample": {
                "reason": "not represented in this test SDF",
                "covered_by": "hardware_contact_report oracle",
            },
        }
    provenance = {"generator": "tests/geo/test_hardware_sdf_keepout.py"}
    if glb_path is not None:
        provenance.update(
            {
                "glb": str(glb_path),
                "glb_sha256": hashlib.sha256(glb_path.read_bytes()).hexdigest(),
            }
        )
    if safety_margin is None:
        safety_margin = effective_margin
    if error_budget is None:
        error_budget = {
            "e_sweep_sample_m": 0.0,
            "e_grid_m": 0.0,
            "e_sign_m": 0.0,
            "e_mesh_m": 0.0,
            "e_oracle_mapping_m": 0.0,
            "e_total_m": 0.0,
        }
    if group_effective_margin is None:
        group_effective_margin = effective_margin
    manifest = {
        "schema_version": 1,
        "kind": "hardware_sdf",
        "frame": "machine_metres_zup",
        "units": units,
        "data_file": data_path.name,
        "data_sha256": data_sha,
        "safety_margin_m": safety_margin,
        "error_budget_m": error_budget,
        "effective_margin_m": effective_margin,
        "groups": [
            {
                "label": "sensors",
                "field_key": "sensors",
                "origin_m": list(origin),
                "spacing_m": spacing,
                "shape": list(grid.shape),
                "sign_method": sign_method,
                "effective_margin_m": group_effective_margin,
            }
        ],
        "documented_gate_only": documented_gate_only,
        "provenance": provenance,
    }
    if narrow_band is not None:
        manifest["narrow_band_m"] = narrow_band
    if group_narrow_band is not None:
        manifest["groups"][0]["narrow_band_m"] = group_narrow_band
    if static_keys is not None:
        manifest["static_hardware_keys"] = list(static_keys)
    if covered_by_other_in_loop is not None:
        manifest["covered_by_other_in_loop"] = covered_by_other_in_loop
    manifest_path = root / "hardware_sdf.json"
    manifest_path.write_text(json.dumps(manifest))
    return manifest_path


class HardwareSdfKeepoutTests(unittest.TestCase):
    def test_loader_rejects_millimetre_units(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            grid = np.ones((3, 3, 3), dtype=float)
            manifest = _write_sdf_payload(
                root,
                grid=grid,
                origin=(0.0, 0.0, 0.0),
                spacing=0.01,
                effective_margin=0.005,
                units="mm",
            )

            with self.assertRaisesRegex(ValueError, "frame/units mismatch"):
                load_hardware_sdf(manifest)

    def test_sdf_penalty_uses_dimensionless_metre_ratio(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            grid = np.zeros((5, 5, 5), dtype=float)
            manifest = _write_sdf_payload(
                root,
                grid=grid,
                origin=(-0.1, -0.1, -0.1),
                spacing=0.05,
                effective_margin=0.005,
            )
            objective = CurveHardwareSdfKeepout(
                [_circle_curve(radius=0.02)],
                load_hardware_sdf(manifest),
                winding_r0=0.0,
            )

            self.assertAlmostEqual(objective.J(), 1.0, places=12)

    def test_loader_fail_closes_static_hardware_coverage(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            grid = np.ones((3, 3, 3), dtype=float)
            manifest = _write_sdf_payload(
                root,
                grid=grid,
                origin=(0.0, 0.0, 0.0),
                spacing=0.01,
                effective_margin=0.005,
                static_keys=("sensors", "frame", "sample", "solenoid"),
            )

            with self.assertRaisesRegex(ValueError, "does not cover"):
                load_hardware_sdf(manifest)

    def test_loader_requires_static_hardware_key_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            grid = np.ones((3, 3, 3), dtype=float)
            manifest = _write_sdf_payload(
                root,
                grid=grid,
                origin=(0.0, 0.0, 0.0),
                spacing=0.01,
                effective_margin=0.005,
                static_keys=None,
            )

            with self.assertRaisesRegex(ValueError, "static_hardware_keys"):
                load_hardware_sdf(manifest)

    def test_loader_rejects_unsigned_sdf_sign_method(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            grid = np.ones((3, 3, 3), dtype=float)
            manifest = _write_sdf_payload(
                root,
                grid=grid,
                origin=(0.0, 0.0, 0.0),
                spacing=0.01,
                effective_margin=0.005,
                sign_method="unsigned_conservative",
            )

            with self.assertRaisesRegex(ValueError, "optimizer-safe"):
                load_hardware_sdf(manifest)

    def test_loader_accepts_component_union_sign_method(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            grid = np.ones((3, 3, 3), dtype=float)
            manifest = _write_sdf_payload(
                root,
                grid=grid,
                origin=(0.0, 0.0, 0.0),
                spacing=0.01,
                effective_margin=0.005,
                sign_method=COMPONENT_UNION_TEST_METHOD,
            )

            sdf_data = load_hardware_sdf(manifest)

            self.assertEqual(
                sdf_data.sign_methods,
                {"sensors": COMPONENT_UNION_TEST_METHOD},
            )

    def test_loader_validates_error_budget_components(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            grid = np.ones((3, 3, 3), dtype=float)
            manifest = _write_sdf_payload(
                root,
                grid=grid,
                origin=(0.0, 0.0, 0.0),
                spacing=0.01,
                effective_margin=0.005,
                error_budget={
                    "e_sweep_sample_m": 0.0,
                    "e_grid_m": 0.001,
                    "e_sign_m": 0.0,
                    "e_mesh_m": 0.0,
                    "e_oracle_mapping_m": 0.0,
                    "e_total_m": 0.0,
                },
            )

            with self.assertRaisesRegex(ValueError, "e_total_m"):
                load_hardware_sdf(manifest)

    def test_loader_rejects_group_margin_below_error_budget_floor(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            grid = np.ones((3, 3, 3), dtype=float)
            manifest = _write_sdf_payload(
                root,
                grid=grid,
                origin=(0.0, 0.0, 0.0),
                spacing=0.01,
                effective_margin=0.008,
                safety_margin=0.005,
                error_budget={
                    "e_sweep_sample_m": 0.0,
                    "e_grid_m": 0.003,
                    "e_sign_m": 0.0,
                    "e_mesh_m": 0.0,
                    "e_oracle_mapping_m": 0.0,
                    "e_total_m": 0.003,
                },
                group_effective_margin=0.001,
            )

            with self.assertRaisesRegex(ValueError, "safety_margin_m plus"):
                load_hardware_sdf(manifest)

    def test_loader_rejects_missing_or_underwide_narrow_band(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            grid = np.ones((3, 3, 3), dtype=float)
            manifest = _write_sdf_payload(
                root,
                grid=grid,
                origin=(0.0, 0.0, 0.0),
                spacing=0.01,
                effective_margin=0.005,
                narrow_band=None,
            )

            with self.assertRaisesRegex(ValueError, "narrow_band_m"):
                load_hardware_sdf(manifest)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            grid = np.ones((3, 3, 3), dtype=float)
            manifest = _write_sdf_payload(
                root,
                grid=grid,
                origin=(0.0, 0.0, 0.0),
                spacing=0.01,
                effective_margin=0.005,
                narrow_band=0.004,
            )

            with self.assertRaisesRegex(ValueError, "narrow_band_m"):
                load_hardware_sdf(manifest)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            grid = np.ones((3, 3, 3), dtype=float)
            manifest = _write_sdf_payload(
                root,
                grid=grid,
                origin=(0.0, 0.0, 0.0),
                spacing=0.01,
                effective_margin=0.005,
                group_narrow_band=0.004,
            )

            with self.assertRaisesRegex(ValueError, "narrow_band_m"):
                load_hardware_sdf(manifest)

    def test_loader_rejects_nonfinite_sdf_numbers(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            grid = np.ones((3, 3, 3), dtype=float)
            manifest = _write_sdf_payload(
                root,
                grid=grid,
                origin=(0.0, 0.0, 0.0),
                spacing=0.01,
                effective_margin=float("nan"),
                safety_margin=0.005,
            )

            with self.assertRaisesRegex(ValueError, "effective_margin_m"):
                load_hardware_sdf(manifest)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            grid = np.ones((3, 3, 3), dtype=float)
            manifest = _write_sdf_payload(
                root,
                grid=grid,
                origin=(0.0, 0.0, 0.0),
                spacing=float("inf"),
                effective_margin=0.005,
            )

            with self.assertRaisesRegex(ValueError, "spacing_m"):
                load_hardware_sdf(manifest)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            grid = np.ones((3, 3, 3), dtype=float)
            manifest = _write_sdf_payload(
                root,
                grid=grid,
                origin=(0.0, 0.0, 0.0),
                spacing=0.01,
                effective_margin=0.005,
                error_budget={
                    "e_sweep_sample_m": 0.0,
                    "e_grid_m": float("inf"),
                    "e_sign_m": 0.0,
                    "e_mesh_m": 0.0,
                    "e_oracle_mapping_m": 0.0,
                    "e_total_m": float("inf"),
                },
            )

            with self.assertRaisesRegex(ValueError, "finite"):
                load_hardware_sdf(manifest)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            grid = np.full((3, 3, 3), np.inf, dtype=float)
            manifest = _write_sdf_payload(
                root,
                grid=grid,
                origin=(0.0, 0.0, 0.0),
                spacing=0.01,
                effective_margin=0.005,
            )

            with self.assertRaisesRegex(ValueError, "grid values"):
                load_hardware_sdf(manifest)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            grid = np.ones((3, 3, 3), dtype=float)
            manifest = _write_sdf_payload(
                root,
                grid=grid,
                origin=(float("nan"), 0.0, 0.0),
                spacing=0.01,
                effective_margin=0.005,
            )

            with self.assertRaisesRegex(ValueError, "origin_m"):
                load_hardware_sdf(manifest)

    def test_metadata_sha_binds_data_and_live_glb(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            glb = root / "hbt_assembly.glb"
            glb.write_bytes(b"current-cad")
            grid = np.ones((3, 3, 3), dtype=float)
            manifest = _write_sdf_payload(
                root,
                grid=grid,
                origin=(0.0, 0.0, 0.0),
                spacing=0.01,
                effective_margin=0.005,
                glb_path=glb,
            )

            metadata = hardware_sdf_metadata(manifest, glb_path=glb)

            self.assertEqual(metadata["HARDWARE_KEEPOUT_BACKEND"], "sdf")
            self.assertEqual(metadata["HARDWARE_SDF_GROUPS"], ["sensors"])
            self.assertEqual(
                metadata["DOCUMENTED_GATE_ONLY_GROUPS"], ["frame", "sample"]
            )
            self.assertEqual(
                metadata["HARDWARE_SDF_LIVE_GLB_SHA256"],
                hashlib.sha256(glb.read_bytes()).hexdigest(),
            )
            self.assertEqual(metadata["HARDWARE_SDF_SAFETY_MARGIN_M"], 0.005)
            self.assertEqual(
                metadata["HARDWARE_SDF_ERROR_BUDGET_M"]["e_total_m"], 0.0
            )

            glb.write_bytes(b"changed-cad")
            with self.assertRaisesRegex(ValueError, "stale hardware SDF manifest"):
                load_hardware_sdf(manifest, glb_path=glb)

    def test_loader_records_vessel_covered_by_other_in_loop(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            grid = np.ones((3, 3, 3), dtype=float)
            manifest = _write_sdf_payload(
                root,
                grid=grid,
                origin=(0.0, 0.0, 0.0),
                spacing=0.01,
                effective_margin=0.005,
                static_keys=("sensors", "vessel", "frame", "sample"),
                covered_by_other_in_loop={
                    "vessel": {
                        "reason": "analytic vessel term in-loop",
                        "covered_by": "CurveVesselEnvelopeKeepout",
                    }
                },
            )

            metadata = hardware_sdf_metadata(manifest)

            self.assertEqual(
                metadata["HARDWARE_SDF_OTHER_IN_LOOP_GROUPS"],
                ["vessel"],
            )

    def test_plane_sdf_objective_inactive_active_and_outside_grid(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            origin = np.array([-1.2, -1.2, -0.1], dtype=float)
            spacing = 0.02
            shape = (121, 121, 11)
            plane_x = 1.012
            margin = 0.005
            grid = _plane_sdf_grid(origin, spacing, shape, plane_x)
            manifest = _write_sdf_payload(
                root,
                grid=grid,
                origin=origin,
                spacing=spacing,
                effective_margin=margin,
            )
            sdf_data = load_hardware_sdf(manifest)

            clear_curve = _circle_curve(radius=0.4)
            active_curve = _circle_curve(radius=1.0)
            clear_objective = CurveHardwareSdfKeepout(
                [clear_curve], sdf_data, winding_r0=0.0
            )
            active_objective = CurveHardwareSdfKeepout(
                [active_curve], sdf_data, winding_r0=0.0
            )

            self.assertEqual(clear_objective.J(), 0.0)
            self.assertGreater(active_objective.J(), 0.0)
            self.assertAlmostEqual(
                active_objective.shortest_clearance(),
                plane_x - (1.0 + TYPE_KK_OUTER_CHANNEL_HALF_DEPTH_NORMAL_M),
                places=12,
            )

            tiny_grid = np.ones((3, 3, 3), dtype=float)
            tiny_manifest = _write_sdf_payload(
                root,
                grid=tiny_grid,
                origin=(10.0, 10.0, 10.0),
                spacing=0.01,
                effective_margin=margin,
            )
            tiny_objective = CurveHardwareSdfKeepout(
                [clear_curve], load_hardware_sdf(tiny_manifest), winding_r0=0.0
            )
            self.assertEqual(tiny_objective.J(), 0.0)
            self.assertEqual(tiny_objective.shortest_clearance(), margin)

    def test_plane_sdf_gradient_matches_finite_difference(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            origin = np.array([-1.2, -1.2, -0.1], dtype=float)
            spacing = 0.02
            shape = (121, 121, 11)
            plane_x = 1.012
            margin = 0.005
            grid = _plane_sdf_grid(origin, spacing, shape, plane_x)
            manifest = _write_sdf_payload(
                root,
                grid=grid,
                origin=origin,
                spacing=spacing,
                effective_margin=margin,
            )
            curve = _circle_curve(seed=33)
            objective = CurveHardwareSdfKeepout(
                [curve], load_hardware_sdf(manifest), winding_r0=0.0
            )
            x0 = np.asarray(curve.x, dtype=float).copy()
            rng = np.random.default_rng(34)
            direction = rng.standard_normal(x0.size)
            direction /= np.linalg.norm(direction)
            analytic = float(np.dot(objective.dJ(), direction))
            self.assertNotEqual(analytic, 0.0)

            eps = 1e-7
            curve.x = x0 + eps * direction
            j_plus = objective.J()
            curve.x = x0 - eps * direction
            j_minus = objective.J()
            curve.x = x0
            fd = (j_plus - j_minus) / (2.0 * eps)

            self.assertAlmostEqual(
                analytic,
                fd,
                delta=2e-4 * max(1.0, abs(analytic)),
                msg=f"SDF gradient analytic {analytic} vs finite-difference {fd}",
            )

    def test_no_solver_smoke_constructs_point_cloud_and_sdf_backends(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            origin = np.array([-1.2, -1.2, -0.1], dtype=float)
            spacing = 0.02
            shape = (121, 121, 11)
            plane_x = 1.012
            margin = 0.005
            grid = _plane_sdf_grid(origin, spacing, shape, plane_x)
            manifest = _write_sdf_payload(
                root,
                grid=grid,
                origin=origin,
                spacing=spacing,
                effective_margin=margin,
            )

            point_cloud_curve = _circle_curve(radius=1.0, seed=35)
            point_cloud_points = np.array([
                [1.0 + 0.011, 0.0, 0.0],
                [0.0, 1.0 + 0.012, 0.0],
            ])
            point_cloud_objective = CurveHardwareKeepout(
                [point_cloud_curve],
                point_cloud_points,
                HARDWARE_KEEPOUT_MIN_DISTANCE_M,
                1e-4,
                winding_r0=0.0,
            )

            sdf_curve = _circle_curve(radius=1.0, seed=36)
            sdf_objective = CurveHardwareSdfKeepout(
                [sdf_curve], load_hardware_sdf(manifest), winding_r0=0.0
            )

            smoke = {
                "point_cloud_J": point_cloud_objective.J(),
                "point_cloud_grad_norm": float(
                    np.linalg.norm(point_cloud_objective.dJ())
                ),
                "sdf_J": sdf_objective.J(),
                "sdf_grad_norm": float(np.linalg.norm(sdf_objective.dJ())),
            }

            for name, value in smoke.items():
                self.assertTrue(
                    np.isfinite(value), msg=f"{name} is non-finite in {smoke}"
                )
                self.assertGreater(value, 0.0, msg=f"{name} inactive in {smoke}")


if __name__ == "__main__":
    unittest.main()
