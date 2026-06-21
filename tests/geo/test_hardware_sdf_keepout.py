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
    HARDWARE_KEEPOUT_SAFETY_MARGIN_M,
    TYPE_KK_OUTER_CHANNEL_HALF_DEPTH_NORMAL_M,
)
from banana_opt.hardware_keepout import (  # noqa: E402
    CurveHardwareKeepout,
    CurveHardwareSdfClearanceHinge,
    CurveHardwareSdfFreeSpaceReward,
    CurveHardwareSdfKeepout,
    SPARSE_NARROW_BAND_SHELL_LAYOUT,
    SPARSE_NARROW_BAND_SHELL_STORAGE_KIND,
    _weighted_soft_min,
    curvature_arc_weight,
    hardware_sdf_metadata,
    load_hardware_sdf,
    resolve_clearance_arc_weight,
    smooth_toroidal_rate_arc_weight,
)
from banana_opt.single_stage_objectives import build_total_objective  # noqa: E402
from simsopt.geo import CurveXYZFourier  # noqa: E402

SIGNED_TEST_METHOD = "watertight_contains_trimesh_nearest"
COMPONENT_UNION_TEST_METHOD = "component_union_watertight_contains_trimesh_nearest"
PSEUDONORMAL_TEST_METHOD = "watertight_pseudonormal_libigl"
COMPONENT_UNION_PSEUDONORMAL_TEST_METHOD = (
    "component_union_watertight_pseudonormal_libigl"
)


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
    patches=None,
):
    data_path = root / "hardware_sdf.npz"
    payload = {"sensors": grid}
    patch_specs = []
    for index, patch in enumerate(() if patches is None else patches):
        patch_label = str(patch.get("label", f"patch_{index}"))
        field_key = str(patch.get("field_key", patch_label))
        payload[field_key] = np.asarray(patch["grid"], dtype=float)
        patch_spec = {
            "label": patch_label,
            "field_key": field_key,
            "origin_m": list(patch["origin"]),
            "spacing_m": patch["spacing"],
            "shape": list(np.asarray(patch["grid"]).shape),
        }
        if "sign_method" in patch:
            patch_spec["sign_method"] = patch["sign_method"]
        if "effective_margin" in patch:
            patch_spec["effective_margin_m"] = patch["effective_margin"]
        if "narrow_band" in patch:
            patch_spec["narrow_band_m"] = patch["narrow_band"]
        patch_specs.append(patch_spec)
    np.savez(data_path, **payload)
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
    group_spec = {
        "label": "sensors",
        "field_key": "sensors",
        "origin_m": list(origin),
        "spacing_m": spacing,
        "shape": list(grid.shape),
        "sign_method": sign_method,
        "effective_margin_m": group_effective_margin,
    }
    if patch_specs:
        group_spec["patches"] = patch_specs
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
        "groups": [group_spec],
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


def _write_sparse_sdf_payload(
    root,
    *,
    shape,
    origin,
    spacing,
    effective_margin,
    sparse_indices,
    sparse_values,
    stored_cell_count=None,
    sparse_default_value=None,
    sparse_indices_key="sensors_sparse_indices",
    sparse_values_key="sensors_sparse_values",
    omit_sparse_indices_payload=False,
    omit_sparse_values_payload=False,
    omit_sparse_default_value=False,
):
    data_path = root / "hardware_sdf.npz"
    payload = {}
    if not omit_sparse_indices_payload:
        payload[sparse_indices_key] = np.asarray(sparse_indices, dtype=np.int64)
    if not omit_sparse_values_payload:
        payload[sparse_values_key] = np.asarray(sparse_values, dtype=float)
    np.savez(data_path, **payload)
    data_sha = hashlib.sha256(data_path.read_bytes()).hexdigest()
    if stored_cell_count is None:
        stored_cell_count = len(sparse_values)
    if sparse_default_value is None:
        sparse_default_value = effective_margin
    group_spec = {
        "label": "sensors",
        "storage_layout": SPARSE_NARROW_BAND_SHELL_LAYOUT,
        "origin_m": list(origin),
        "spacing_m": spacing,
        "shape": list(shape),
        "sign_method": SIGNED_TEST_METHOD,
        "effective_margin_m": effective_margin,
        "narrow_band_m": 0.02,
        "sparse_indices_key": sparse_indices_key,
        "sparse_values_key": sparse_values_key,
        "stored_cell_count": stored_cell_count,
    }
    if not omit_sparse_default_value:
        group_spec["sparse_default_value_m"] = sparse_default_value
    manifest = {
        "schema_version": 1,
        "kind": "hardware_sdf",
        "storage_kind": SPARSE_NARROW_BAND_SHELL_STORAGE_KIND,
        "storage_layout": SPARSE_NARROW_BAND_SHELL_LAYOUT,
        "frame": "machine_metres_zup",
        "units": "m",
        "data_file": data_path.name,
        "data_sha256": data_sha,
        "safety_margin_m": effective_margin,
        "error_budget_m": {
            "e_sweep_sample_m": 0.0,
            "e_grid_m": 0.0,
            "e_sign_m": 0.0,
            "e_mesh_m": 0.0,
            "e_oracle_mapping_m": 0.0,
            "e_total_m": 0.0,
        },
        "effective_margin_m": effective_margin,
        "narrow_band_m": 0.02,
        "static_hardware_keys": ["sensors", "frame", "sample"],
        "groups": [group_spec],
        "documented_gate_only": {
            "frame": {
                "reason": "not represented in this test SDF",
                "covered_by": "hardware_contact_report oracle",
            },
            "sample": {
                "reason": "not represented in this test SDF",
                "covered_by": "hardware_contact_report oracle",
            },
        },
        "provenance": {"generator": "tests/geo/test_hardware_sdf_keepout.py"},
    }
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

    def test_sdf_free_space_reward_clips_inside_and_saturates_at_margin(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            margin = 0.004
            curve = _circle_curve(radius=0.02)
            inside_root = root / "inside"
            half_margin_root = root / "half_margin"
            saturated_root = root / "saturated"
            inside_root.mkdir()
            half_margin_root.mkdir()
            saturated_root.mkdir()

            inside_manifest = _write_sdf_payload(
                inside_root,
                grid=np.full((5, 5, 5), -0.001, dtype=float),
                origin=(-0.1, -0.1, -0.1),
                spacing=0.05,
                effective_margin=margin,
            )
            half_margin_manifest = _write_sdf_payload(
                half_margin_root,
                grid=np.full((5, 5, 5), 0.5 * margin, dtype=float),
                origin=(-0.1, -0.1, -0.1),
                spacing=0.05,
                effective_margin=margin,
            )
            saturated_manifest = _write_sdf_payload(
                saturated_root,
                grid=np.full((5, 5, 5), 2.0 * margin, dtype=float),
                origin=(-0.1, -0.1, -0.1),
                spacing=0.05,
                effective_margin=margin,
            )

            inside_objective = CurveHardwareSdfFreeSpaceReward(
                [curve], load_hardware_sdf(inside_manifest), winding_r0=0.0
            )
            half_margin_objective = CurveHardwareSdfFreeSpaceReward(
                [curve], load_hardware_sdf(half_margin_manifest), winding_r0=0.0
            )
            saturated_objective = CurveHardwareSdfFreeSpaceReward(
                [curve], load_hardware_sdf(saturated_manifest), winding_r0=0.0
            )

            self.assertEqual(inside_objective.J(), 0.0)
            self.assertAlmostEqual(half_margin_objective.J(), -0.25, places=12)
            self.assertAlmostEqual(saturated_objective.J(), -1.0, places=12)

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

    def test_loader_accepts_pseudonormal_libigl_sign_methods(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for sign_method in (
                PSEUDONORMAL_TEST_METHOD,
                COMPONENT_UNION_PSEUDONORMAL_TEST_METHOD,
            ):
                case_root = root / sign_method
                case_root.mkdir()
                manifest = _write_sdf_payload(
                    case_root,
                    grid=np.ones((3, 3, 3), dtype=float),
                    origin=(0.0, 0.0, 0.0),
                    spacing=0.01,
                    effective_margin=0.005,
                    sign_method=sign_method,
                )

                sdf_data = load_hardware_sdf(manifest)

                self.assertEqual(sdf_data.sign_methods, {"sensors": sign_method})

    def test_loader_accepts_sparse_narrow_band_shell_payload(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            margin = 0.006
            shape = (4, 4, 4)
            sparse_indices = np.array([[1, 1, 1], [2, 1, 1]], dtype=np.int64)
            sparse_values = np.array([-0.003, 0.011], dtype=float)
            sparse_default_value = margin + 0.004
            manifest = _write_sparse_sdf_payload(
                root,
                shape=shape,
                origin=(0.0, 0.0, 0.0),
                spacing=0.004,
                effective_margin=margin,
                sparse_indices=sparse_indices,
                sparse_values=sparse_values,
                sparse_default_value=sparse_default_value,
            )

            sdf_data = load_hardware_sdf(manifest)
            grid = sdf_data.groups[0].grid

            self.assertEqual(grid.shape, shape)
            self.assertEqual(sdf_data.group_labels, ("sensors",))
            self.assertAlmostEqual(grid[1, 1, 1], sparse_values[0])
            self.assertAlmostEqual(grid[2, 1, 1], sparse_values[1])
            self.assertAlmostEqual(grid[0, 0, 0], sparse_default_value)
            self.assertAlmostEqual(sdf_data.groups[0].spacing_m, 0.004)

    def test_loader_rejects_sparse_payload_without_default_value(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = _write_sparse_sdf_payload(
                root,
                shape=(4, 4, 4),
                origin=(0.0, 0.0, 0.0),
                spacing=0.004,
                effective_margin=0.006,
                sparse_indices=np.array([[1, 1, 1]], dtype=np.int64),
                sparse_values=np.array([0.001], dtype=float),
                omit_sparse_default_value=True,
            )

            with self.assertRaisesRegex(ValueError, "sparse_default_value_m"):
                load_hardware_sdf(manifest)

    def test_loader_rejects_sparse_default_below_effective_margin(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = _write_sparse_sdf_payload(
                root,
                shape=(4, 4, 4),
                origin=(0.0, 0.0, 0.0),
                spacing=0.004,
                effective_margin=0.006,
                sparse_indices=np.array([[1, 1, 1]], dtype=np.int64),
                sparse_values=np.array([0.001], dtype=float),
                sparse_default_value=0.005,
            )

            with self.assertRaisesRegex(ValueError, "sparse_default_value_m"):
                load_hardware_sdf(manifest)

    def test_loader_rejects_sparse_payload_with_missing_key(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = _write_sparse_sdf_payload(
                root,
                shape=(4, 4, 4),
                origin=(0.0, 0.0, 0.0),
                spacing=0.004,
                effective_margin=0.006,
                sparse_indices=np.array([[1, 1, 1]], dtype=np.int64),
                sparse_values=np.array([0.001], dtype=float),
                omit_sparse_indices_payload=True,
            )

            with self.assertRaisesRegex(ValueError, "missing sparse field"):
                load_hardware_sdf(manifest)

    def test_loader_rejects_sparse_stored_cell_count_mismatch(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = _write_sparse_sdf_payload(
                root,
                shape=(4, 4, 4),
                origin=(0.0, 0.0, 0.0),
                spacing=0.004,
                effective_margin=0.006,
                sparse_indices=np.array([[1, 1, 1]], dtype=np.int64),
                sparse_values=np.array([0.001], dtype=float),
                stored_cell_count=2,
            )

            with self.assertRaisesRegex(ValueError, "stored_cell_count mismatch"):
                load_hardware_sdf(manifest)

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

    def test_loader_records_patch_metadata(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = _write_sdf_payload(
                root,
                grid=np.full((5, 5, 5), 0.02, dtype=float),
                origin=(-2.0, -2.0, -1.0),
                spacing=1.0,
                effective_margin=0.02,
                narrow_band=0.2,
                patches=[
                    {
                        "label": "near_contact",
                        "grid": np.full((21, 41, 21), -0.002, dtype=float),
                        "origin": (0.9, -0.2, -0.1),
                        "spacing": 0.01,
                        "effective_margin": 0.02,
                        "narrow_band": 0.2,
                    }
                ],
            )

            metadata = hardware_sdf_metadata(manifest)
            sdf_data = load_hardware_sdf(manifest)

            self.assertEqual(metadata["HARDWARE_SDF_PATCH_COUNT"], 1)
            self.assertEqual(
                metadata["HARDWARE_SDF_PATCH_LABELS"],
                {"sensors": ["near_contact"]},
            )
            self.assertEqual(
                metadata["HARDWARE_SDF_PATCH_SPACING_M"],
                {"sensors": {"near_contact": 0.01}},
            )
            self.assertEqual(sdf_data.groups[0].patches[0].label, "near_contact")

    def test_patch_grid_overrides_base_grid_for_covering_samples(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = _write_sdf_payload(
                root,
                grid=np.full((5, 5, 5), 0.02, dtype=float),
                origin=(-2.0, -2.0, -1.0),
                spacing=1.0,
                effective_margin=0.02,
                narrow_band=0.2,
                patches=[
                    {
                        "label": "near_contact",
                        "grid": np.full((21, 41, 21), -0.002, dtype=float),
                        "origin": (0.9, -0.2, -0.1),
                        "spacing": 0.01,
                        "effective_margin": 0.02,
                        "narrow_band": 0.2,
                    }
                ],
            )
            sdf_data = load_hardware_sdf(manifest)
            curve = _circle_curve(radius=1.0)
            objective = CurveHardwareSdfKeepout(
                [curve], sdf_data, winding_r0=0.0
            )

            self.assertAlmostEqual(objective.shortest_clearance(), -0.002)
            self.assertGreater(objective.J(), 0.0)

    def test_finest_covering_patch_wins(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = _write_sdf_payload(
                root,
                grid=np.full((5, 5, 5), 0.02, dtype=float),
                origin=(-2.0, -2.0, -1.0),
                spacing=1.0,
                effective_margin=0.02,
                narrow_band=0.2,
                patches=[
                    {
                        "label": "coarse_near_contact",
                        "grid": np.full((11, 21, 11), 0.004, dtype=float),
                        "origin": (0.9, -0.2, -0.1),
                        "spacing": 0.02,
                        "effective_margin": 0.02,
                        "narrow_band": 0.2,
                    },
                    {
                        "label": "fine_near_contact",
                        "grid": np.full((21, 41, 21), -0.003, dtype=float),
                        "origin": (0.9, -0.2, -0.1),
                        "spacing": 0.01,
                        "effective_margin": 0.02,
                        "narrow_band": 0.2,
                    },
                ],
            )
            sdf_data = load_hardware_sdf(manifest)
            curve = _circle_curve(radius=1.0)
            objective = CurveHardwareSdfKeepout(
                [curve], sdf_data, winding_r0=0.0
            )

            self.assertAlmostEqual(objective.shortest_clearance(), -0.003)

    def test_loader_rejects_patch_that_is_not_finer_than_base_grid(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = _write_sdf_payload(
                root,
                grid=np.full((5, 5, 5), 0.02, dtype=float),
                origin=(-2.0, -2.0, -1.0),
                spacing=1.0,
                effective_margin=0.02,
                narrow_band=0.2,
                patches=[
                    {
                        "label": "not_finer",
                        "grid": np.full((3, 3, 3), 0.01, dtype=float),
                        "origin": (-1.0, -1.0, -1.0),
                        "spacing": 1.0,
                        "effective_margin": 0.02,
                        "narrow_band": 0.2,
                    }
                ],
            )

            with self.assertRaisesRegex(ValueError, "must be finer"):
                load_hardware_sdf(manifest)

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


def _wavy_curve(order=3, quadpoints=80):
    """Closed curve whose curvature varies along the arc (unlike a circle).

    The second-harmonic terms create distinct high- and low-curvature stations
    so curvature-based arc weighting is non-uniform.
    """
    curve = CurveXYZFourier(quadpoints, order)
    curve.x = np.zeros(curve.dof_size)
    curve.set("xc(1)", 1.0)
    curve.set("ys(1)", 1.0)
    curve.set("xc(2)", 0.18)
    curve.set("zs(2)", 0.05)
    return curve


class ClearanceArcWeightTests(unittest.TestCase):
    """Curvature-based arc weighting on the SDF clearance terms."""

    def _active_plane_sdf(self, root):
        # A large effective margin keeps the clip band wide so the clearance
        # fraction varies (rather than saturating) across the whole arc, which
        # makes the arc-weighting effect observable. This is a math fixture, not
        # a physical margin.
        origin = np.array([-1.3, -1.3, -0.3], dtype=float)
        spacing = 0.02
        shape = (131, 131, 31)
        plane_x = 1.21
        margin = 0.4
        grid = _plane_sdf_grid(origin, spacing, shape, plane_x)
        manifest = _write_sdf_payload(
            root,
            grid=grid,
            origin=origin,
            spacing=spacing,
            effective_margin=margin,
            narrow_band=0.8,
        )
        return load_hardware_sdf(manifest)

    def test_resolve_clearance_arc_weight_default_and_validation(self):
        self.assertIsNone(resolve_clearance_arc_weight(1.0, 1.0))
        self.assertEqual(
            resolve_clearance_arc_weight(0.2, 5.0),
            ("curvature", 0.2, 5.0),
        )
        self.assertEqual(
            resolve_clearance_arc_weight(0.2, 5.0, "smooth_toroidal_rate"),
            ("smooth_toroidal_rate", 0.2, 5.0),
        )
        for bad in [(0.0, 1.0), (1.0, -1.0)]:
            with self.assertRaisesRegex(ValueError, "must be positive"):
                resolve_clearance_arc_weight(*bad)
        with self.assertRaisesRegex(ValueError, "profile must be one of"):
            resolve_clearance_arc_weight(0.2, 5.0, "step")

    def test_curvature_arc_weight_maps_min_to_leg_max_to_uturn(self):
        w = curvature_arc_weight(np.array([10.0, 20.0, 30.0]), 0.5, 2.0)
        self.assertTrue(np.allclose(w, [0.5, 1.25, 2.0]))
        flat = curvature_arc_weight(np.array([7.0, 7.0]), 0.5, 2.0)
        self.assertTrue(np.allclose(flat, [0.5, 0.5]))

    def test_smooth_toroidal_rate_arc_weight_is_smooth_and_periodic(self):
        phi = np.array([0.0, 0.08, 0.31, 1.15, 2.2, 4.0])
        gamma = np.column_stack(
            [np.cos(phi), np.sin(phi), np.zeros_like(phi)]
        )
        w = smooth_toroidal_rate_arc_weight(gamma, 0.5, 2.0)
        self.assertAlmostEqual(float(np.min(w)), 0.5)
        self.assertAlmostEqual(float(np.max(w)), 2.0)
        self.assertGreater(np.ptp(w), 0.0)

        circle_phi = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
        circle = np.column_stack(
            [np.cos(circle_phi), np.sin(circle_phi), np.zeros_like(circle_phi)]
        )
        flat = smooth_toroidal_rate_arc_weight(circle, 0.5, 2.0)
        self.assertTrue(np.allclose(flat, 0.5))

    def test_uniform_weights_are_byte_identical_to_default(self):
        with tempfile.TemporaryDirectory() as td:
            sdf = self._active_plane_sdf(Path(td))
            curve = _wavy_curve()
            for cls in (CurveHardwareSdfKeepout, CurveHardwareSdfFreeSpaceReward):
                default = cls([curve], sdf, winding_r0=0.0)
                uniform = cls(
                    [curve], sdf, winding_r0=0.0,
                    clearance_leg_weight=1.0, clearance_uturn_weight=1.0,
                )
                self.assertIsNone(default._arc_weight)
                self.assertIsNone(default._station_weights[0])
                self.assertEqual(
                    default.J(), uniform.J(),
                    msg=f"{cls.__name__}: 1.0/1.0 must equal the default",
                )

    def test_arc_weight_keys_off_curvature_not_applied_blindly(self):
        # A circle has uniform curvature -> the arc weight collapses to a uniform
        # array, so even uturn != leg leaves J unchanged. Proves the weighting is
        # driven by curvature variation, not blindly multiplied in.
        with tempfile.TemporaryDirectory() as td:
            sdf = self._active_plane_sdf(Path(td))
            circle = _circle_curve(radius=1.0)
            uniform = CurveHardwareSdfFreeSpaceReward(
                [circle], sdf, winding_r0=0.0
            )
            flared = CurveHardwareSdfFreeSpaceReward(
                [circle], sdf, winding_r0=0.0,
                clearance_leg_weight=0.2, clearance_uturn_weight=5.0,
            )
            w = flared._station_weights[0]
            self.assertTrue(np.allclose(w, w[0]), "circle -> uniform weights")
            self.assertAlmostEqual(uniform.J(), flared.J(), places=12)

    def test_arc_weight_changes_objective_on_varying_field(self):
        with tempfile.TemporaryDirectory() as td:
            sdf = self._active_plane_sdf(Path(td))
            curve = _wavy_curve()
            uniform = CurveHardwareSdfFreeSpaceReward([curve], sdf, winding_r0=0.0)
            flared = CurveHardwareSdfFreeSpaceReward(
                [curve], sdf, winding_r0=0.0,
                clearance_leg_weight=0.2, clearance_uturn_weight=5.0,
            )
            # Frozen profile matches the documented curvature map.
            expected = curvature_arc_weight(curve.kappa(), 0.2, 5.0)
            self.assertTrue(np.allclose(flared._station_weights[0], expected))
            self.assertGreater(
                np.ptp(flared._station_weights[0]), 0.0,
                "wavy curve must give a non-uniform weight profile",
            )
            self.assertNotAlmostEqual(
                uniform.J(), flared.J(), places=6,
                msg="arc weighting must change J on a varying clearance field",
            )

    def test_smooth_toroidal_rate_profile_freezes_expected_weights(self):
        with tempfile.TemporaryDirectory() as td:
            sdf = self._active_plane_sdf(Path(td))
            curve = _wavy_curve()
            flared = CurveHardwareSdfFreeSpaceReward(
                [curve], sdf, winding_r0=0.0,
                clearance_leg_weight=0.2, clearance_uturn_weight=5.0,
                clearance_arc_weight_profile="smooth_toroidal_rate",
            )
            expected = smooth_toroidal_rate_arc_weight(
                curve.gamma(), 0.2, 5.0
            )
            self.assertEqual(
                flared._arc_weight,
                ("smooth_toroidal_rate", 0.2, 5.0),
            )
            self.assertTrue(np.allclose(flared._station_weights[0], expected))

    def test_arc_weighted_gradient_matches_finite_difference(self):
        # The decisive guard for the freeze-at-construction design: weights are
        # held fixed, so the analytic dJ (which differentiates only the clearance)
        # must match a finite difference of J. A live-recomputed weight would fail.
        with tempfile.TemporaryDirectory() as td:
            sdf = self._active_plane_sdf(Path(td))
            curve = _wavy_curve()
            rng = np.random.default_rng(7)
            curve.x = curve.x + 1e-3 * rng.standard_normal(curve.dof_size)
            obj = CurveHardwareSdfFreeSpaceReward(
                [curve], sdf, winding_r0=0.0,
                clearance_leg_weight=0.2, clearance_uturn_weight=5.0,
            )
            x0 = np.asarray(curve.x, dtype=float).copy()
            direction = rng.standard_normal(x0.size)
            direction /= np.linalg.norm(direction)
            analytic = float(np.dot(obj.dJ(), direction))
            self.assertNotEqual(analytic, 0.0)

            eps = 1e-7
            curve.x = x0 + eps * direction
            j_plus = obj.J()
            curve.x = x0 - eps * direction
            j_minus = obj.J()
            curve.x = x0
            fd = (j_plus - j_minus) / (2.0 * eps)
            self.assertAlmostEqual(
                analytic, fd,
                delta=2e-4 * max(1.0, abs(analytic)),
                msg=f"arc-weighted gradient analytic {analytic} vs FD {fd}",
            )


class _FakeScalarObjective:
    """Minimal +/* /.J() stand-in for the default-off graph-equivalence test."""

    def __init__(self, value):
        self._value = float(value)

    def J(self):
        return self._value

    def __add__(self, other):
        if other == 0:
            return self
        return _FakeScalarObjective(self._value + other._value)

    __radd__ = __add__

    def __mul__(self, scalar):
        return _FakeScalarObjective(self._value * float(scalar))

    __rmul__ = __mul__


def _base_total_objective_args():
    """Positional build_total_objective args from neutral fake physics terms."""
    zero = _FakeScalarObjective(0.0)
    return (
        zero,  # JnonQSRatio
        0.0,  # RES_WEIGHT
        zero,  # JBoozerResidual
        0.0,  # IOTAS_WEIGHT
        zero,  # Jiota
        0.0,  # VOLUME_WEIGHT
        None,  # JVolume
        0.0,  # LENGTH_WEIGHT
        zero,  # JCurveLength
        0.0,  # CC_WEIGHT
        zero,  # JCurveCurve
        0.0,  # CS_WEIGHT
        zero,  # JCurveSurface
        0.0,  # CURVATURE_WEIGHT
        zero,  # JCurvature
    )


class ClearanceHingeTests(unittest.TestCase):
    """One-sided hinge closing the worst sampled swept Type KK SDF clearance."""

    def _uniform_sdf(self, root, *, clearance, margin):
        # A constant SDF grid sets phi == clearance at every sampled surface
        # point, so the hinge value is fully determined by clearance vs target.
        manifest = _write_sdf_payload(
            root,
            grid=np.full((5, 5, 5), float(clearance), dtype=float),
            origin=(-0.1, -0.1, -0.1),
            spacing=0.05,
            effective_margin=margin,
        )
        return load_hardware_sdf(manifest)

    def test_zero_when_clear_and_positive_when_close(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            margin = 0.02
            target = 0.005
            curve = _circle_curve(radius=0.02)

            clear_root = root / "clear"
            close_root = root / "close"
            clear_root.mkdir()
            close_root.mkdir()

            # Clearance above target everywhere -> no shortfall -> J == 0.
            clear = self._uniform_sdf(
                clear_root, clearance=0.01, margin=margin
            )
            # Clearance below target everywhere -> uniform shortfall -> J > 0.
            close = self._uniform_sdf(
                close_root, clearance=0.001, margin=margin
            )

            clear_obj = CurveHardwareSdfClearanceHinge(
                [curve], clear, target_margin=target, winding_r0=0.0
            )
            close_obj = CurveHardwareSdfClearanceHinge(
                [curve], close, target_margin=target, winding_r0=0.0
            )

            self.assertEqual(clear_obj.J(), 0.0)
            self.assertGreater(close_obj.J(), 0.0)
            # Uniform phi == 0.001, target 0.005 -> shortfall 0.004 everywhere.
            self.assertAlmostEqual(close_obj.J(), (target - 0.001) ** 2, places=12)

    def test_accepts_per_quadpoint_target_margin_vector(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            margin = 0.02
            clearance = 0.001
            curve = _circle_curve(radius=0.02, quadpoints=8)
            sdf = self._uniform_sdf(root, clearance=clearance, margin=margin)
            target = np.linspace(0.003, 0.006, curve.gamma().shape[0])

            objective = CurveHardwareSdfClearanceHinge(
                [curve], sdf, target_margin=target, winding_r0=0.0
            )

            expected = float(np.mean((target - clearance) ** 2))
            self.assertAlmostEqual(objective.J(), expected, places=12)

    def test_uniform_target_margin_vector_matches_scalar_value_and_gradient(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            origin = np.array([-1.2, -1.2, -0.1], dtype=float)
            spacing = 0.02
            shape = (121, 121, 11)
            plane_x = 1.012
            margin = 0.005
            target = 0.05
            grid = _plane_sdf_grid(origin, spacing, shape, plane_x)
            manifest = _write_sdf_payload(
                root,
                grid=grid,
                origin=origin,
                spacing=spacing,
                effective_margin=margin,
                narrow_band=0.2,
            )
            sdf = load_hardware_sdf(manifest)
            curve = _circle_curve(seed=33)
            scalar = CurveHardwareSdfClearanceHinge(
                [curve], sdf, target_margin=target, winding_r0=0.0
            )
            vector = CurveHardwareSdfClearanceHinge(
                [curve],
                sdf,
                target_margin=np.full(curve.gamma().shape[0], target),
                winding_r0=0.0,
            )

            self.assertAlmostEqual(vector.J(), scalar.J(), places=12)
            self.assertTrue(np.allclose(vector.dJ(), scalar.dJ(), atol=1e-12))

    def test_softmin_hinge_tracks_per_group_min_clearance(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            origin = np.array([-1.2, -1.2, -0.1], dtype=float)
            spacing = 0.02
            shape = (121, 121, 11)
            plane_x = 1.02
            margin = 0.005
            target = 0.05
            grid = _plane_sdf_grid(origin, spacing, shape, plane_x)
            manifest = _write_sdf_payload(
                root,
                grid=grid,
                origin=origin,
                spacing=spacing,
                effective_margin=margin,
                narrow_band=0.2,
            )
            sdf = load_hardware_sdf(manifest)
            curve = _circle_curve(radius=1.0)
            diagnostic = CurveHardwareSdfKeepout([curve], sdf, winding_r0=0.0)
            objective = CurveHardwareSdfClearanceHinge(
                [curve],
                sdf,
                target_margin=target,
                winding_r0=0.0,
                soft_min_temperature=1.0e-6,
            )

            min_clearance = min(diagnostic.per_group_min_clearance().values())
            expected = max(target - min_clearance, 0.0) ** 2
            self.assertEqual(
                objective.per_group_min_clearance(),
                diagnostic.per_group_min_clearance(),
            )
            self.assertAlmostEqual(
                objective.shortest_clearance(),
                min_clearance,
                places=12,
            )
            self.assertGreaterEqual(objective.J(), expected)
            self.assertAlmostEqual(
                objective.J(),
                expected,
                delta=5.0e-4 * max(1.0e-12, expected),
            )

    def test_softmin_lower_bound_keeps_isolated_violation_active(self):
        values = np.full(64 * 26, 0.01, dtype=float)
        values[0] = -0.001

        soft_min = float(_weighted_soft_min(values, None, 0.001))

        self.assertLessEqual(soft_min, values.min())
        self.assertGreater(max(-soft_min, 0.0) ** 2, 0.0)

    def test_decreases_as_coil_pushed_into_clearance(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            origin = np.array([-1.2, -1.2, -0.1], dtype=float)
            spacing = 0.02
            shape = (121, 121, 11)
            # Plane SDF: phi = plane_x - x, so clearance shrinks toward +x. With
            # the plane just outboard of the unit circle, the worst (+x) station
            # sits inside the target band and the hinge is active.
            plane_x = 1.02
            margin = 0.005
            target = 0.05
            grid = _plane_sdf_grid(origin, spacing, shape, plane_x)
            manifest = _write_sdf_payload(
                root,
                grid=grid,
                origin=origin,
                spacing=spacing,
                effective_margin=margin,
                narrow_band=0.2,
            )
            sdf = load_hardware_sdf(manifest)
            curve = _circle_curve(radius=1.0)
            objective = CurveHardwareSdfClearanceHinge(
                [curve], sdf, target_margin=target, winding_r0=0.0
            )
            j_start = objective.J()
            self.assertGreater(j_start, 0.0)

            # Shrink the circle radius (xc(1)) -> the +x station retreats from the
            # plane -> clearance grows -> the hinge shortfall drops.
            x0 = np.asarray(curve.x, dtype=float).copy()
            xc1_index = list(curve.local_dof_names).index("xc(1)")
            moved = x0.copy()
            moved[xc1_index] -= 0.02
            curve.x = moved
            j_pushed = objective.J()
            curve.x = x0

            self.assertLess(j_pushed, j_start)

    def test_gradient_matches_finite_difference(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            origin = np.array([-1.2, -1.2, -0.1], dtype=float)
            spacing = 0.02
            shape = (121, 121, 11)
            plane_x = 1.012
            margin = 0.005
            # A wide target keeps the hinge active across the sampled stations so
            # the analytic gradient (differentiating only the clearance) is
            # exercised; this is the decisive contrast vs a gradient-dead hard min.
            target = 0.05
            grid = _plane_sdf_grid(origin, spacing, shape, plane_x)
            manifest = _write_sdf_payload(
                root,
                grid=grid,
                origin=origin,
                spacing=spacing,
                effective_margin=margin,
                narrow_band=0.2,
            )
            curve = _circle_curve(seed=33)
            objective = CurveHardwareSdfClearanceHinge(
                [curve], load_hardware_sdf(manifest),
                target_margin=target, winding_r0=0.0,
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
                msg=f"hinge gradient analytic {analytic} vs finite-difference {fd}",
            )

    def test_rejects_non_positive_target_margin(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sdf = self._uniform_sdf(root, clearance=0.01, margin=0.02)
            curve = _circle_curve(radius=0.02)
            for bad in (0.0, -0.005, float("nan"), float("inf")):
                with self.assertRaisesRegex(ValueError, "target_margin"):
                    CurveHardwareSdfClearanceHinge(
                        [curve], sdf, target_margin=bad, winding_r0=0.0
                    )

    def test_rejects_invalid_vector_margin_and_softmin_temperature(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sdf = self._uniform_sdf(root, clearance=0.01, margin=0.02)
            curve = _circle_curve(radius=0.02, quadpoints=8)

            with self.assertRaisesRegex(ValueError, "target_margin vector length"):
                CurveHardwareSdfClearanceHinge(
                    [curve],
                    sdf,
                    target_margin=np.full(curve.gamma().shape[0] - 1, 0.005),
                    winding_r0=0.0,
                )
            with self.assertRaisesRegex(ValueError, "target_margin vector entries"):
                CurveHardwareSdfClearanceHinge(
                    [curve],
                    sdf,
                    target_margin=np.full(curve.gamma().shape[0], np.nan),
                    winding_r0=0.0,
                )
            for bad in (0.0, -1.0e-3, float("nan"), float("inf")):
                with self.assertRaisesRegex(ValueError, "soft_min_temperature"):
                    CurveHardwareSdfClearanceHinge(
                        [curve],
                        sdf,
                        target_margin=0.005,
                        winding_r0=0.0,
                        soft_min_temperature=bad,
                    )

    def test_build_total_objective_default_off_leaves_graph_unchanged(self):
        reward = _FakeScalarObjective(-0.25)
        baseline = build_total_objective(*_base_total_objective_args())
        default_off = build_total_objective(
            *_base_total_objective_args(),
            JCurveHardwareSdfClearanceHinge=None,
            CLEARANCE_HINGE_WEIGHT=7.0,
        )
        self.assertEqual(default_off.J(), baseline.J())

        # A non-None term with weight does move the objective (guards the wiring).
        active = build_total_objective(
            *_base_total_objective_args(),
            JCurveHardwareSdfClearanceHinge=reward,
            CLEARANCE_HINGE_WEIGHT=7.0,
        )
        self.assertAlmostEqual(active.J() - baseline.J(), 7.0 * reward.J())


if __name__ == "__main__":
    unittest.main()
