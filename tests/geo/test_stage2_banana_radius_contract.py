import inspect
import sys
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_ROOT = REPO_ROOT / "examples" / "single_stage_optimization"
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

from STAGE_2 import banana_coil_solver as stage2_solver  # noqa: E402
from banana_opt.current_contracts import (  # noqa: E402
    resolve_loaded_tf_current_A,
    resolve_penalty_traversal_forbidden_box_bounds,
)
from banana_opt.hardware_contracts import VACUUM_VESSEL_MINOR_RADIUS_M  # noqa: E402


class FakeCurrent:
    def __init__(self, value):
        self.value = value

    def get_value(self):
        return self.value


class FakeCoil:
    def __init__(self, current_A):
        self.current = FakeCurrent(current_A)


class Stage2BananaRadiusContractTests(unittest.TestCase):
    def test_stage2_main_uses_shared_winding_radius_contract(self):
        source = inspect.getsource(stage2_solver.main)

        self.assertIn(
            "validate_banana_winding_surface_radius(args.banana_surf_radius)",
            source,
        )
        self.assertNotIn(
            "Stage 2 banana winding surface must remain concentric",
            source,
        )

    def test_cli_accepts_exploratory_radius_inside_vessel(self):
        with mock.patch.object(
            sys,
            "argv",
            ["banana_coil_solver.py", "--banana-surf-radius", "0.22"],
        ):
            args = stage2_solver.parse_args()

        self.assertEqual(args.banana_surf_radius, 0.22)
        self.assertLess(args.banana_surf_radius, VACUUM_VESSEL_MINOR_RADIUS_M)

    def test_offspec_major_radius_requires_explicit_replay_flag(self):
        source = inspect.getsource(stage2_solver.main)

        self.assertIn("args.accept_offspec_major_radius", source)
        with mock.patch.object(
            sys,
            "argv",
            [
                "banana_coil_solver.py",
                "--major-radius",
                "0.915",
                "--accept-offspec-major-radius",
            ],
        ):
            args = stage2_solver.parse_args()

        self.assertEqual(args.major_radius, 0.915)
        self.assertTrue(args.accept_offspec_major_radius)

    def test_stage2_parse_args_accepts_legacy_working_surface_scaling_mode(self):
        source = inspect.getsource(stage2_solver.main)

        self.assertIn("load_plasma_geometry_for_working_major_radius", source)
        self.assertIn('args.stage2_plasma_scaling_mode != "working"', source)
        with mock.patch.object(
            sys,
            "argv",
            [
                "banana_coil_solver.py",
                "--stage2-plasma-scaling-mode",
                "working",
            ],
        ):
            args = stage2_solver.parse_args()

        self.assertEqual(args.stage2_plasma_scaling_mode, "working")

    def test_positive_banana_current_sign_requires_explicit_replay_flag(self):
        with mock.patch.object(
            sys,
            "argv",
            [
                "banana_coil_solver.py",
                "--banana-init-current-A",
                "10000",
            ],
        ):
            with self.assertRaises(SystemExit):
                stage2_solver.parse_args()

        with mock.patch.object(
            sys,
            "argv",
            [
                "banana_coil_solver.py",
                "--banana-init-current-A",
                "10000",
                "--accept-offspec-banana-current-sign",
            ],
        ):
            args = stage2_solver.parse_args()

        self.assertEqual(args.banana_init_current_A, 10000)
        self.assertTrue(args.accept_offspec_banana_current_sign)

    def test_positive_tf_current_sign_requires_explicit_replay_flag(self):
        with self.assertRaises(ValueError):
            stage2_solver.validate_stage2_tf_current_value(
                80000.0,
                accepts_offspec_sign=False,
                accepts_offspec_magnitude=False,
                field_name="loaded Stage 2 seed TF current",
            )

        stage2_solver.validate_stage2_tf_current_value(
            80000.0,
            accepts_offspec_sign=True,
            accepts_offspec_magnitude=False,
            field_name="loaded Stage 2 seed TF current",
        )

        with mock.patch.object(
            sys,
            "argv",
            [
                "banana_coil_solver.py",
                "--tf-current-A",
                "80000",
            ],
        ):
            with self.assertRaises(SystemExit):
                stage2_solver.parse_args()

        with mock.patch.object(
            sys,
            "argv",
            [
                "banana_coil_solver.py",
                "--tf-current-A",
                "80000",
                "--accept-offspec-tf-current-sign",
            ],
        ):
            args = stage2_solver.parse_args()

        self.assertEqual(args.tf_current_A, 80000)
        self.assertTrue(args.accept_offspec_tf_current_sign)

    def test_offspec_current_magnitudes_require_explicit_replay_flags(self):
        with self.assertRaises(ValueError):
            stage2_solver.validate_stage2_tf_current_value(
                100000.0,
                accepts_offspec_sign=True,
                accepts_offspec_magnitude=False,
                field_name="loaded Stage 2 seed TF current",
            )

        stage2_solver.validate_stage2_tf_current_value(
            100000.0,
            accepts_offspec_sign=True,
            accepts_offspec_magnitude=True,
            field_name="loaded Stage 2 seed TF current",
        )

        with mock.patch.object(
            sys,
            "argv",
            [
                "banana_coil_solver.py",
                "--tf-current-A",
                "100000",
                "--accept-offspec-tf-current-sign",
            ],
        ):
            with self.assertRaises(SystemExit):
                stage2_solver.parse_args()

        with mock.patch.object(
            sys,
            "argv",
            [
                "banana_coil_solver.py",
                "--tf-current-A",
                "100000",
                "--banana-current-max-A",
                "20000",
                "--accept-offspec-tf-current-sign",
                "--accept-offspec-tf-current-magnitude",
                "--accept-offspec-banana-current-max",
            ],
        ):
            args = stage2_solver.parse_args()

        self.assertEqual(args.tf_current_A, 100000)
        self.assertEqual(args.banana_current_max_A, 20000)
        self.assertTrue(args.accept_offspec_tf_current_magnitude)
        self.assertTrue(args.accept_offspec_banana_current_max)

    def test_offspec_banana_current_flag_controls_penalty_runtime_bound(self):
        requested_thresholds = {"banana_current": 20000.0}

        self.assertEqual(
            resolve_penalty_traversal_forbidden_box_bounds(requested_thresholds),
            {"banana_current": 16000.0},
        )
        self.assertEqual(
            resolve_penalty_traversal_forbidden_box_bounds(
                requested_thresholds,
                allow_offspec_threshold_names=frozenset({"banana_current"}),
            ),
            {"banana_current": 20000.0},
        )

    def test_bootability_reload_accepts_explicit_offspec_tf_current_contract(self):
        coils = [FakeCoil(100000.0), FakeCoil(100000.0)]

        with self.assertRaises(ValueError):
            resolve_loaded_tf_current_A(100000.0, coils)

        self.assertEqual(
            resolve_loaded_tf_current_A(
                100000.0,
                coils,
                allow_offspec_current_contract=True,
            ),
            100000.0,
        )

    def test_bootability_probe_receives_offspec_current_replay_flag(self):
        source = inspect.getsource(stage2_solver.build_stage2_iota_report_payload)

        self.assertIn("allow_offspec_current_contract", source)

    def test_loaded_stage2_seed_uses_explicit_offspec_tf_replay_flags(self):
        source = inspect.getsource(stage2_solver.main)

        self.assertIn("loaded Stage 2 seed TF current", source)
        self.assertIn("validate_stage2_tf_current_value", source)


if __name__ == "__main__":
    unittest.main()
