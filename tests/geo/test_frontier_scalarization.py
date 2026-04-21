import importlib
import importlib.util
import json
import sys
import tempfile
import unittest
import uuid
from pathlib import Path


EXAMPLE_ROOT = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "single_stage_optimization"
)
FRONTIER_CAMPAIGN_PATH = EXAMPLE_ROOT / "run_single_stage_frontier_campaign.py"
GOAL_MODE_COMPARISON_PATH = (
    EXAMPLE_ROOT / "run_single_stage_goal_mode_comparison.py"
)
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))


def load_module(path: Path, stem: str):
    spec = importlib.util.spec_from_file_location(f"{stem}_{uuid.uuid4().hex}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_frontier_scalarization_module():
    return importlib.import_module("banana_opt.frontier_scalarization")


def load_frontier_campaign_module():
    return load_module(FRONTIER_CAMPAIGN_PATH, "run_single_stage_frontier_campaign")


def load_goal_mode_comparison_module():
    return load_module(
        GOAL_MODE_COMPARISON_PATH,
        "run_single_stage_goal_mode_comparison",
    )


class FrontierScalarizationTests(unittest.TestCase):
    def test_generate_frontier_lane_specs_shared_mode_matches_legacy_schedule(self):
        module = load_frontier_scalarization_module()

        lane_specs = module.generate_frontier_lane_specs(
            reference_mode=module.FRONTIER_REFERENCE_MODE_SHARED,
            num_lanes=3,
            iotas_weight=100.0,
            frontier_volume_weight=200.0,
            res_weight=1000.0,
            lane_budget=300,
            stage2_results=None,
            reference_points_file=None,
            epsilon_spec_file=None,
        )

        self.assertEqual([lane.lane_id for lane in lane_specs], ["lane_01", "lane_02", "lane_03"])
        self.assertEqual(
            [lane.scalarization_type for lane in lane_specs],
            ["weight_schedule_v1", "weight_schedule_v1", "weight_schedule_v1"],
        )
        self.assertAlmostEqual(lane_specs[0].iotas_weight, 60.0)
        self.assertAlmostEqual(lane_specs[0].frontier_volume_weight, 240.0)
        self.assertEqual(module.frontier_scalarization_family(lane_specs), "weight_schedule_v1")

    def test_reference_point_lane_specs_use_file_contract_and_weights(self):
        module = load_frontier_scalarization_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            reference_points_path = Path(tmpdir) / "reference_points.json"
            reference_points_path.write_text(
                json.dumps(
                    {
                        "schema_version": "frontier_reference_points_v1",
                        "lanes": [
                            {
                                "lane_id": "lane_iota",
                                "reference_point": {
                                    "iota": 0.165,
                                    "volume": 0.103,
                                    "qa_error": 0.011,
                                    "boozer_residual": 0.0075,
                                },
                                "iota_share": 0.8,
                                "volume_share": 0.2,
                                "frontier_reference_iota_scale": 0.03,
                                "frontier_reference_volume_scale": 0.012,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            lane_specs = module.generate_frontier_lane_specs(
                reference_mode=module.FRONTIER_REFERENCE_MODE_REFERENCE_POINTS,
                num_lanes=1,
                iotas_weight=100.0,
                frontier_volume_weight=200.0,
                res_weight=1000.0,
                lane_budget=250,
                stage2_results=None,
                reference_points_file=str(reference_points_path),
                epsilon_spec_file=None,
            )

        self.assertEqual(len(lane_specs), 1)
        lane_spec = lane_specs[0]
        self.assertEqual(lane_spec.lane_id, "lane_iota")
        self.assertEqual(
            lane_spec.scalarization_type,
            module.FRONTIER_REFERENCE_MODE_REFERENCE_POINTS,
        )
        self.assertAlmostEqual(lane_spec.iotas_weight, 240.0)
        self.assertAlmostEqual(lane_spec.frontier_volume_weight, 60.0)
        self.assertEqual(lane_spec.lane_budget, 250)
        self.assertEqual(lane_spec.scalarization_params["frontier_reference_iota"], 0.165)
        self.assertEqual(lane_spec.scalarization_params["frontier_reference_volume"], 0.103)
        self.assertEqual(lane_spec.scalarization_params["frontier_reference_qa"], 0.011)
        self.assertEqual(lane_spec.scalarization_params["frontier_reference_boozer"], 0.0075)
        self.assertEqual(lane_spec.scalarization_params["frontier_reference_iota_scale"], 0.03)
        self.assertEqual(lane_spec.scalarization_params["frontier_reference_volume_scale"], 0.012)

    def test_epsilon_lane_specs_project_constraints_into_lane_contract(self):
        module = load_frontier_scalarization_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            epsilon_path = Path(tmpdir) / "epsilon.json"
            epsilon_path.write_text(
                json.dumps(
                    {
                        "schema_version": "frontier_epsilon_spec_v1",
                        "lanes": [
                            {
                                "lane_id": "lane_safe_iota",
                                "objective": "iota",
                                "epsilon_constraints": {
                                    "qa_error": 0.011,
                                    "boozer_residual": 0.007,
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            lane_specs = module.generate_frontier_lane_specs(
                reference_mode=module.FRONTIER_REFERENCE_MODE_EPSILON,
                num_lanes=1,
                iotas_weight=100.0,
                frontier_volume_weight=150.0,
                res_weight=900.0,
                lane_budget=275,
                stage2_results={
                    "FINAL_IOTA": 0.15,
                    "FINAL_VOLUME": 0.10,
                    "NONQS_RATIO": 0.012,
                    "BOOZER_RESIDUAL": 0.008,
                },
                reference_points_file=None,
                epsilon_spec_file=str(epsilon_path),
            )

        self.assertEqual(len(lane_specs), 1)
        lane_spec = lane_specs[0]
        self.assertEqual(
            lane_spec.scalarization_type,
            module.FRONTIER_REFERENCE_MODE_EPSILON,
        )
        self.assertAlmostEqual(lane_spec.iotas_weight, 250.0)
        self.assertAlmostEqual(lane_spec.frontier_volume_weight, 0.0)
        self.assertEqual(lane_spec.lane_budget, 275)
        self.assertEqual(lane_spec.scalarization_params["frontier_reference_iota"], 0.15)
        self.assertEqual(lane_spec.scalarization_params["frontier_reference_volume"], 0.10)
        self.assertEqual(lane_spec.scalarization_params["frontier_reference_qa"], 0.011)
        self.assertEqual(lane_spec.scalarization_params["frontier_reference_boozer"], 0.007)
        self.assertEqual(
            lane_spec.scalarization_params["frontier_boozer_trust_threshold"],
            0.007,
        )
        self.assertEqual(
            lane_spec.scalarization_params["epsilon_constraint_qa_max"],
            0.011,
        )
        self.assertEqual(
            lane_spec.scalarization_params["epsilon_constraint_boozer_max"],
            0.007,
        )

    def test_achievement_chebyshev_lane_specs_use_reference_points_file(self):
        module = load_frontier_scalarization_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            achievement_path = Path(tmpdir) / "achievement.json"
            achievement_path.write_text(
                json.dumps(
                    {
                        "schema_version": "frontier_achievement_spec_v1",
                        "lanes": [
                            {
                                "lane_id": "lane_tradeoff",
                                "reference_point": {
                                    "iota": 0.17,
                                    "volume": 0.105,
                                    "qa_error": 0.011,
                                    "boozer_residual": 0.007,
                                },
                                "metric_weights": {
                                    "iota": 2.0,
                                    "volume": 1.5,
                                    "qa_error": 1.0,
                                    "boozer_residual": 0.5,
                                },
                                "rho": 0.02,
                                "iota_share": 0.7,
                                "volume_share": 0.3,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            lane_specs = module.generate_frontier_lane_specs(
                reference_mode=module.FRONTIER_REFERENCE_MODE_ACHIEVEMENT,
                num_lanes=1,
                iotas_weight=100.0,
                frontier_volume_weight=200.0,
                res_weight=1000.0,
                lane_budget=250,
                stage2_results=None,
                reference_points_file=str(achievement_path),
                epsilon_spec_file=None,
            )

        self.assertEqual(len(lane_specs), 1)
        lane_spec = lane_specs[0]
        self.assertEqual(
            lane_spec.scalarization_type,
            module.FRONTIER_REFERENCE_MODE_ACHIEVEMENT,
        )
        self.assertEqual(lane_spec.lane_id, "lane_tradeoff")
        self.assertAlmostEqual(lane_spec.iotas_weight, 210.0)
        self.assertAlmostEqual(lane_spec.frontier_volume_weight, 90.0)
        self.assertEqual(lane_spec.lane_budget, 250)
        self.assertEqual(lane_spec.scalarization_params["frontier_reference_iota"], 0.17)
        self.assertEqual(lane_spec.scalarization_params["frontier_reference_volume"], 0.105)
        self.assertEqual(lane_spec.scalarization_params["frontier_reference_qa"], 0.011)
        self.assertEqual(lane_spec.scalarization_params["frontier_reference_boozer"], 0.007)
        self.assertEqual(lane_spec.scalarization_params["frontier_chebyshev_rho"], 0.02)
        self.assertEqual(
            lane_spec.scalarization_params["frontier_chebyshev_weight_iota"],
            2.0,
        )
        self.assertEqual(
            lane_spec.scalarization_params["frontier_chebyshev_weight_volume"],
            1.5,
        )

    def test_frontier_campaign_threads_scalarization_overrides_into_single_stage_command(self):
        frontier_campaign_module = load_frontier_campaign_module()
        goal_mode_module = load_goal_mode_comparison_module()

        args = frontier_campaign_module.parse_args(
            [
                "--plasma-surf-filename",
                "demo.nc",
                "--stage2-bs-path",
                "/tmp/demo/biot_savart_opt.json",
            ]
        )
        lane_spec = frontier_campaign_module.FrontierLaneSpec(
            lane_id="lane_01",
            scalarization_type="reference_point_sweep_v1",
            scalarization_params={
                "frontier_reference_iota": 0.17,
                "frontier_reference_volume": 0.104,
                "frontier_reference_qa": 0.011,
                "frontier_reference_boozer": 0.0075,
                "frontier_boozer_trust_threshold": 0.009,
            },
            iotas_weight=180.0,
            frontier_volume_weight=120.0,
            res_weight=1000.0,
            lane_budget=275,
        )

        lane_args = frontier_campaign_module.build_frontier_lane_args(args, lane_spec)
        command = goal_mode_module.build_single_stage_goal_mode_command(
            lane_args,
            goal_mode="frontier",
            stage2_bs_path=Path("/tmp/demo/biot_savart_opt.json"),
            case_output_root=Path("/tmp/frontier_case"),
        )

        self.assertEqual(
            command[command.index("--frontier-reference-iota") + 1],
            "0.17",
        )
        self.assertEqual(
            command[command.index("--frontier-reference-volume") + 1],
            "0.104",
        )
        self.assertEqual(
            command[command.index("--frontier-reference-qa") + 1],
            "0.011",
        )
        self.assertEqual(
            command[command.index("--frontier-reference-boozer") + 1],
            "0.0075",
        )
        self.assertEqual(
            command[command.index("--frontier-boozer-trust-threshold") + 1],
            "0.009",
        )

    def test_frontier_campaign_threads_chebyshev_lane_contract_into_single_stage_command(self):
        frontier_campaign_module = load_frontier_campaign_module()
        goal_mode_module = load_goal_mode_comparison_module()

        args = frontier_campaign_module.parse_args(
            [
                "--plasma-surf-filename",
                "demo.nc",
                "--stage2-bs-path",
                "/tmp/demo/biot_savart_opt.json",
            ]
        )
        lane_spec = frontier_campaign_module.FrontierLaneSpec(
            lane_id="lane_tradeoff",
            scalarization_type="achievement_chebyshev_sweep_v1",
            scalarization_params={
                "frontier_reference_iota": 0.17,
                "frontier_reference_volume": 0.104,
                "frontier_reference_qa": 0.011,
                "frontier_reference_boozer": 0.0075,
                "frontier_chebyshev_rho": 0.02,
                "frontier_chebyshev_weight_iota": 2.0,
                "frontier_chebyshev_weight_volume": 1.5,
                "frontier_chebyshev_weight_qa": 1.0,
                "frontier_chebyshev_weight_boozer": 0.5,
            },
            iotas_weight=180.0,
            frontier_volume_weight=120.0,
            res_weight=1000.0,
            lane_budget=275,
        )

        lane_args = frontier_campaign_module.build_frontier_lane_args(args, lane_spec)
        command = goal_mode_module.build_single_stage_goal_mode_command(
            lane_args,
            goal_mode="frontier",
            stage2_bs_path=Path("/tmp/demo/biot_savart_opt.json"),
            case_output_root=Path("/tmp/frontier_case"),
        )

        self.assertEqual(
            command[command.index("--frontier-scalarization-type") + 1],
            "achievement_chebyshev_sweep_v1",
        )
        self.assertEqual(command[command.index("--frontier-chebyshev-rho") + 1], "0.02")
        self.assertEqual(
            command[command.index("--frontier-chebyshev-weight-iota") + 1],
            "2.0",
        )

    def test_frontier_campaign_threads_epsilon_lane_contract_into_single_stage_command(self):
        frontier_campaign_module = load_frontier_campaign_module()
        goal_mode_module = load_goal_mode_comparison_module()

        args = frontier_campaign_module.parse_args(
            [
                "--plasma-surf-filename",
                "demo.nc",
                "--stage2-bs-path",
                "/tmp/demo/biot_savart_opt.json",
            ]
        )
        lane_spec = frontier_campaign_module.FrontierLaneSpec(
            lane_id="lane_safe_iota",
            scalarization_type="epsilon_constraint_sweep_v1",
            scalarization_params={
                "frontier_reference_iota": 0.15,
                "frontier_reference_volume": 0.10,
                "frontier_reference_qa": 0.011,
                "frontier_reference_boozer": 0.007,
                "epsilon_constraint_qa_max": 0.011,
                "epsilon_constraint_boozer_max": 0.007,
            },
            iotas_weight=250.0,
            frontier_volume_weight=0.0,
            res_weight=900.0,
            lane_budget=275,
        )

        lane_args = frontier_campaign_module.build_frontier_lane_args(args, lane_spec)
        command = goal_mode_module.build_single_stage_goal_mode_command(
            lane_args,
            goal_mode="frontier",
            stage2_bs_path=Path("/tmp/demo/biot_savart_opt.json"),
            case_output_root=Path("/tmp/frontier_case"),
        )

        self.assertEqual(
            command[command.index("--frontier-scalarization-type") + 1],
            "epsilon_constraint_sweep_v1",
        )
        self.assertEqual(
            command[command.index("--epsilon-constraint-qa-max") + 1],
            "0.011",
        )
        self.assertEqual(
            command[command.index("--epsilon-constraint-boozer-max") + 1],
            "0.007",
        )


    def test_lane_rng_seed_uses_explicit_lane_index(self):
        frontier_campaign_module = load_frontier_campaign_module()

        self.assertEqual(
            frontier_campaign_module.lane_rng_seed(42, lane_index=0),
            42,
        )
        self.assertEqual(
            frontier_campaign_module.lane_rng_seed(42, lane_index=3),
            45,
        )


if __name__ == "__main__":
    unittest.main()
