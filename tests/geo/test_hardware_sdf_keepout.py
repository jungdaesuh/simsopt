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
    TYPE_KK_OUTER_CHANNEL_HALF_DEPTH_NORMAL_M,
)
from banana_opt.hardware_keepout import (  # noqa: E402
    CurveHardwareSdfKeepout,
    hardware_sdf_metadata,
    load_hardware_sdf,
)
from simsopt.geo import CurveXYZFourier  # noqa: E402


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
    manifest = {
        "schema_version": 1,
        "kind": "hardware_sdf",
        "frame": "machine_metres_zup",
        "units": units,
        "data_file": data_path.name,
        "data_sha256": data_sha,
        "static_hardware_keys": list(static_keys),
        "groups": [
            {
                "label": "sensors",
                "field_key": "sensors",
                "origin_m": list(origin),
                "spacing_m": spacing,
                "shape": list(grid.shape),
                "sign_method": "analytic_test_plane",
                "effective_margin_m": effective_margin,
            }
        ],
        "documented_gate_only": documented_gate_only,
        "provenance": provenance,
    }
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
            self.assertGreater(tiny_objective.J(), 0.0)
            self.assertLess(tiny_objective.shortest_clearance(), 0.0)

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


if __name__ == "__main__":
    unittest.main()
