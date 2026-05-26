import json
import tempfile
import unittest
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np

from geo._frontier_test_helpers import (
    load_frontier_campaign_execution_module,
    load_frontier_campaign_module,
    load_frontier_scalarization_module,
    load_goal_mode_comparison_module,
)


class FrontierScalarizationTests(unittest.TestCase):
    STAGE2_RESULTS = {
        "FINAL_IOTA": 0.15,
        "FINAL_VOLUME": 0.10,
        "NONQS_RATIO": 0.012,
        "BOOZER_RESIDUAL": 0.008,
    }

    def _write_spec(self, tmpdir, filename, payload):
        path = Path(tmpdir) / filename
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def _generate_lanes(self, module, reference_mode, **overrides):
        options = {
            "num_lanes": 1,
            "iotas_weight": 100.0,
            "frontier_volume_weight": 200.0,
            "res_weight": 1000.0,
            "lane_budget": 250,
            "stage2_results": None,
            "reference_points_file": None,
            "epsilon_spec_file": None,
        }
        options.update(overrides)
        return module.generate_frontier_lane_specs(
            reference_mode=reference_mode,
            **options,
        )

    def _frontier_lane_command(
        self,
        frontier_campaign_module,
        *,
        scalarization_type,
        scalarization_params,
        lane_id="lane_01",
        iotas_weight=180.0,
        frontier_volume_weight=120.0,
        res_weight=1000.0,
        lane_budget=275,
    ):
        goal_mode_module = load_goal_mode_comparison_module()
        args = frontier_campaign_module.parse_args(
            [
                "--plasma-surf-filename",
                "demo.nc",
                "--stage2-bs-path",
                "/tmp/demo/biot_savart_opt.json",
                "--frontier-hypervolume-reference",
                "auto:seed",
            ]
        )
        lane_spec = frontier_campaign_module.FrontierLaneSpec(
            lane_id=lane_id,
            scalarization_type=scalarization_type,
            scalarization_params=scalarization_params,
            iotas_weight=iotas_weight,
            frontier_volume_weight=frontier_volume_weight,
            res_weight=res_weight,
            lane_budget=lane_budget,
        )
        execution_module = load_frontier_campaign_execution_module()
        lane_args = execution_module.build_frontier_lane_args(args, lane_spec)
        return goal_mode_module.build_single_stage_goal_mode_command(
            lane_args,
            goal_mode="frontier",
            stage2_bs_path=Path("/tmp/demo/biot_savart_opt.json"),
            case_output_root=Path("/tmp/frontier_case"),
        )

    def _assert_params(self, lane_spec, expected):
        self.assertEqual({key: lane_spec.scalarization_params[key] for key in expected}, expected)

    def _assert_command_values(self, command, expected):
        self.assertEqual({flag: command[command.index(flag) + 1] for flag in expected}, expected)

    def test_generate_frontier_lane_specs_shared_mode_matches_legacy_schedule(self):
        module = load_frontier_scalarization_module()

        lane_specs = self._generate_lanes(
            module,
            reference_mode=module.FRONTIER_REFERENCE_MODE_SHARED,
            num_lanes=3,
            lane_budget=300,
        )

        self.assertEqual([lane.lane_id for lane in lane_specs], ["lane_01", "lane_02", "lane_03"])
        self.assertEqual([lane.scalarization_type for lane in lane_specs], ["weight_schedule_v1", "weight_schedule_v1", "weight_schedule_v1"])
        self.assertAlmostEqual(lane_specs[0].iotas_weight, 60.0)
        self.assertAlmostEqual(lane_specs[0].frontier_volume_weight, 240.0)
        self.assertEqual(module.frontier_scalarization_family(lane_specs), "weight_schedule_v1")

    def test_generate_frontier_lane_specs_dispatch_registry_covers_supported_modes(self):
        module = load_frontier_scalarization_module()

        self.assertIsInstance(module._FRONTIER_LANE_SPEC_GENERATORS, MappingProxyType)
        self.assertEqual(
            set(module._FRONTIER_LANE_SPEC_GENERATORS),
            set(module.SUPPORTED_FRONTIER_REFERENCE_MODES),
        )
        with self.assertRaises(TypeError):
            module._FRONTIER_LANE_SPEC_GENERATORS[module.FRONTIER_REFERENCE_MODE_SHARED] = (
                module._shared_lane_specs_from_request
            )

    def test_multilane_share_sweep_normalization(self):
        module = load_frontier_scalarization_module()

        for num_lanes in (1, 2, 3, 5, 10):
            lane_specs = module.generate_multilane_local_specs(
                num_lanes=num_lanes,
                iotas_weight=100.0,
                frontier_volume_weight=200.0,
                res_weight=1000.0,
                lane_budget=300,
            )

            self.assertEqual(len(lane_specs), num_lanes)
            for lane_spec in lane_specs:
                iota_share = lane_spec.scalarization_params["iota_share"]
                volume_share = lane_spec.scalarization_params["volume_share"]
                self.assertGreaterEqual(iota_share, 0.0)
                self.assertGreaterEqual(volume_share, 0.0)
                self.assertAlmostEqual(iota_share + volume_share, 1.0)

    def test_multilane_share_sweep_boundary(self):
        module = load_frontier_scalarization_module()

        lane_specs = module.generate_multilane_local_specs(
            num_lanes=2,
            iotas_weight=100.0,
            frontier_volume_weight=200.0,
            res_weight=1000.0,
            lane_budget=300,
        )

        observed = [
            (
                lane.scalarization_params["iota_share"],
                lane.scalarization_params["volume_share"],
            )
            for lane in lane_specs
        ]
        self.assertAlmostEqual(observed[0][0], 0.2)
        self.assertAlmostEqual(observed[0][1], 0.8)
        self.assertAlmostEqual(observed[1][0], 0.8)
        self.assertAlmostEqual(observed[1][1], 0.2)

    def test_reference_point_lane_specs_use_file_contract_and_weights(self):
        module = load_frontier_scalarization_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            reference_points_path = self._write_spec(
                tmpdir,
                "reference_points.json",
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
                },
            )

            lane_specs = self._generate_lanes(
                module,
                reference_mode=module.FRONTIER_REFERENCE_MODE_REFERENCE_POINTS,
                reference_points_file=str(reference_points_path),
            )

        self.assertEqual(len(lane_specs), 1)
        lane_spec = lane_specs[0]
        self.assertEqual(lane_spec.lane_id, "lane_iota")
        self.assertEqual(lane_spec.scalarization_type, module.FRONTIER_REFERENCE_MODE_REFERENCE_POINTS)
        self.assertAlmostEqual(lane_spec.iotas_weight, 240.0)
        self.assertAlmostEqual(lane_spec.frontier_volume_weight, 60.0)
        self.assertEqual(lane_spec.lane_budget, 250)
        self._assert_params(
            lane_spec,
            {
                "frontier_reference_iota": 0.165,
                "frontier_reference_volume": 0.103,
                "frontier_reference_qa": 0.011,
                "frontier_reference_boozer": 0.0075,
                "frontier_reference_iota_scale": 0.03,
                "frontier_reference_volume_scale": 0.012,
            },
        )

    def test_epsilon_lane_specs_project_constraints_into_lane_contract(self):
        module = load_frontier_scalarization_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            epsilon_path = self._write_spec(
                tmpdir,
                "epsilon.json",
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
                },
            )

            lane_specs = self._generate_lanes(
                module,
                reference_mode=module.FRONTIER_REFERENCE_MODE_EPSILON,
                frontier_volume_weight=150.0,
                res_weight=900.0,
                lane_budget=275,
                stage2_results=self.STAGE2_RESULTS,
                epsilon_spec_file=str(epsilon_path),
            )

        self.assertEqual(len(lane_specs), 1)
        lane_spec = lane_specs[0]
        self.assertEqual(lane_spec.scalarization_type, module.FRONTIER_REFERENCE_MODE_EPSILON)
        self.assertAlmostEqual(lane_spec.iotas_weight, 250.0)
        self.assertAlmostEqual(lane_spec.frontier_volume_weight, 0.0)
        self.assertEqual(lane_spec.lane_budget, 275)
        self._assert_params(
            lane_spec,
            {
                "frontier_reference_iota": 0.15,
                "frontier_reference_volume": 0.10,
                "frontier_reference_qa": 0.011,
                "frontier_reference_boozer": 0.007,
                "frontier_boozer_trust_threshold": 0.007,
                "epsilon_constraint_qa_max": 0.011,
                "epsilon_constraint_boozer_max": 0.007,
            },
        )

    def test_epsilon_lane_specs_reject_unknown_epsilon_metric_keys(self):
        module = load_frontier_scalarization_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            epsilon_path = self._write_spec(
                tmpdir,
                "epsilon.json",
                {
                    "schema_version": "frontier_epsilon_spec_v1",
                    "lanes": [
                        {
                            "lane_id": "lane_bad",
                            "objective": "iota",
                            "epsilon_constraints": {
                                "qa_error": 0.011,
                                "unknown_metric": 0.5,
                            },
                        }
                    ],
                },
            )

            with self.assertRaisesRegex(ValueError, "unsupported keys"):
                self._generate_lanes(
                    module,
                    reference_mode=module.FRONTIER_REFERENCE_MODE_EPSILON,
                    frontier_volume_weight=150.0,
                    res_weight=900.0,
                    lane_budget=275,
                    stage2_results=self.STAGE2_RESULTS,
                    epsilon_spec_file=str(epsilon_path),
                )

    def test_achievement_chebyshev_lane_specs_use_reference_points_file(self):
        module = load_frontier_scalarization_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            achievement_path = self._write_spec(
                tmpdir,
                "achievement.json",
                {
                    "schema_version": module.FRONTIER_ACHIEVEMENT_SPEC_SCHEMA_VERSION,
                    "lanes": [
                        {
                            "lane_id": "lane_tradeoff",
                            "reference_point": {
                                "iota": 0.17,
                                "volume": 0.105,
                                "qa_error": 0.011,
                                "boozer_residual": 0.007,
                            },
                            "frontier_reference_qa_scale": 0.011 * 0.25,
                            "frontier_reference_boozer_scale": 0.007 * 0.25,
                            "metric_weights": {
                                "iota": 2.0,
                                "volume": 1.5,
                                "qa_error": 1.0,
                                "boozer_residual": 0.5,
                            },
                            "rho": 0.02,
                            "sharpness": 18.0,
                            "iota_share": 0.7,
                            "volume_share": 0.3,
                        }
                    ],
                },
            )

            lane_specs = self._generate_lanes(
                module,
                reference_mode=module.FRONTIER_REFERENCE_MODE_ACHIEVEMENT,
                reference_points_file=str(achievement_path),
            )

        self.assertEqual(len(lane_specs), 1)
        lane_spec = lane_specs[0]
        self.assertEqual(lane_spec.scalarization_type, module.FRONTIER_REFERENCE_MODE_ACHIEVEMENT)
        self.assertEqual(lane_spec.lane_id, "lane_tradeoff")
        self.assertAlmostEqual(lane_spec.iotas_weight, 210.0)
        self.assertAlmostEqual(lane_spec.frontier_volume_weight, 90.0)
        self.assertEqual(lane_spec.lane_budget, 250)
        self._assert_params(
            lane_spec,
            {
                "frontier_reference_iota": 0.17,
                "frontier_reference_volume": 0.105,
                "frontier_reference_qa": 0.011,
                "frontier_reference_qa_scale": 0.011 * 0.25,
                "frontier_reference_boozer": 0.007,
                "frontier_reference_boozer_scale": 0.007 * 0.25,
                "frontier_chebyshev_rho": 0.02,
                "frontier_chebyshev_sharpness": 18.0,
                "frontier_chebyshev_weight_iota": 2.0,
                "frontier_chebyshev_weight_volume": 1.5,
            },
        )

    def test_frontier_chebyshev_goal_rejects_nonpositive_sharpness(self):
        module = load_frontier_scalarization_module()

        def config(sharpness):
            return SimpleNamespace(
                chebyshev_weight_iota=1.0,
                chebyshev_weight_volume=1.0,
                chebyshev_weight_qa=1.0,
                chebyshev_weight_boozer=1.0,
                iota_reference=0.18,
                volume_reference=0.12,
                qs_reference=0.010,
                boozer_reference=0.006,
                iota_scale=0.03,
                volume_scale=0.02,
                qs_scale=0.0025,
                boozer_scale=0.0015,
                chebyshev_sharpness=sharpness,
                chebyshev_rho=0.0,
            )

        objective_eval = {
            "J_iota_metric": 0.20,
            "J_volume_metric": 0.13,
            "J_QS": 0.009,
            "J_Boozer": 0.005,
            "dJ_iota_metric": [0.0],
            "dJ_volume_metric": [0.0],
            "dJ_QS": [0.0],
            "dJ_Boozer": [0.0],
        }
        for invalid in (0.0, -1.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                module._frontier_chebyshev_goal(objective_eval, config(invalid))

    def test_chebyshev_goal_uses_qs_scale_distinct_from_qs_reference(self):
        """F1.1: chebyshev gradient and delta divide by qs_scale, not qs_reference."""
        module = load_frontier_scalarization_module()

        def config(qs_scale_value):
            return SimpleNamespace(
                chebyshev_weight_iota=1.0,
                chebyshev_weight_volume=1.0,
                chebyshev_weight_qa=1.0,
                chebyshev_weight_boozer=1.0,
                iota_reference=0.18,
                volume_reference=0.12,
                qs_reference=0.010,
                boozer_reference=0.006,
                iota_scale=0.03,
                volume_scale=0.02,
                qs_scale=qs_scale_value,
                boozer_scale=0.0015,
                chebyshev_sharpness=12.0,
                chebyshev_rho=0.0,
            )

        objective_eval = {
            "J_iota_metric": 0.18,
            "J_volume_metric": 0.12,
            "J_QS": 0.020,
            "J_Boozer": 0.006,
            "dJ_iota_metric": [0.0],
            "dJ_volume_metric": [0.0],
            "dJ_QS": [1.0],
            "dJ_Boozer": [0.0],
        }
        # Halving qs_scale must double the qa-axis gradient contribution.
        # If the implementation still divided by qs_reference, halving
        # qs_scale would have no effect on the gradient. The softmax weight
        # of the QA axis nudges by O(1/sharpness) when scale halves, so we
        # require ~6 decimal places, not bit-exact equality.
        result_a = module._frontier_chebyshev_goal(objective_eval, config(0.005))
        result_b = module._frontier_chebyshev_goal(objective_eval, config(0.0025))
        grad_a = float(np.asarray(result_a["frontier_scalarization_grad"])[0])
        grad_b = float(np.asarray(result_b["frontier_scalarization_grad"])[0])
        self.assertAlmostEqual(grad_b / grad_a, 2.0, places=6)

    def test_excess_penalty_uses_qs_scale_distinct_from_qs_reference(self):
        """F1.3: epsilon excess penalty divides by qs_scale, not qs_reference."""
        module = load_frontier_scalarization_module()

        config = SimpleNamespace(
            qs_reference=0.010,
            boozer_reference=0.006,
            qs_scale=0.005,
            boozer_scale=0.003,
            epsilon_constraint_qa_max=0.012,
            epsilon_constraint_boozer_max=None,
            epsilon_penalty_weight=4.0,
        )
        objective_eval = {
            "J_QS": 0.020,
            "J_Boozer": 0.005,
            "dJ_QS": np.asarray([1.0]),
            "dJ_Boozer": np.asarray([0.0]),
        }
        penalties = module._frontier_epsilon_penalties(
            objective_eval,
            frontier_goal_config=config,
        )
        excess = 0.020 - 0.012
        expected_ratio = excess / config.qs_scale
        expected_penalty = config.epsilon_penalty_weight * expected_ratio ** 2
        self.assertAlmostEqual(
            penalties["qa_error"]["penalty"],
            expected_penalty,
            places=12,
        )

    def test_achievement_scalarization_scaling_invariance(self):
        module = load_frontier_scalarization_module()
        candidates = {"member_a": (0.20, 0.13, 0.009, 0.005), "member_b": (0.16, 0.11, 0.011, 0.007)}

        def config(scale):
            return SimpleNamespace(
                chebyshev_weight_iota=1.0,
                chebyshev_weight_volume=1.0,
                chebyshev_weight_qa=1.0,
                chebyshev_weight_boozer=1.0,
                iota_reference=0.18 * scale,
                volume_reference=0.12 * scale,
                qs_reference=0.010 * scale,
                boozer_reference=0.006 * scale,
                iota_scale=0.03 * scale,
                volume_scale=0.02 * scale,
                qs_scale=0.0025 * scale,
                boozer_scale=0.0015 * scale,
                chebyshev_sharpness=12.0,
                chebyshev_rho=0.02,
            )

        def total(metrics, scale):
            iota, volume, qa_error, boozer_residual = metrics
            objective_eval = {
                "J_iota_metric": iota * scale,
                "J_volume_metric": volume * scale,
                "J_QS": qa_error * scale,
                "J_Boozer": boozer_residual * scale,
                "dJ_iota_metric": [0.0],
                "dJ_volume_metric": [0.0],
                "dJ_QS": [0.0],
                "dJ_Boozer": [0.0],
            }
            return module._frontier_chebyshev_goal(objective_eval, config(scale))["frontier_scalarization_total"]

        def selected(scale):
            return min(candidates, key=lambda member_id: total(candidates[member_id], scale))

        self.assertEqual((selected(1.0), selected(2.0)), ("member_a", "member_a"))

    def test_frontier_campaign_threads_lane_contracts_into_single_stage_command(self):
        frontier_campaign_module = load_frontier_campaign_module()
        cases = [
            (
                "reference_point_sweep_v1",
                {},
                {
                    "frontier_reference_iota": 0.17,
                    "frontier_reference_volume": 0.104,
                    "frontier_reference_qa": 0.011,
                    "frontier_reference_boozer": 0.0075,
                    "frontier_boozer_trust_threshold": 0.009,
                },
                {
                    "--frontier-reference-iota": "0.17",
                    "--frontier-reference-volume": "0.104",
                    "--frontier-reference-qa": "0.011",
                    "--frontier-reference-boozer": "0.0075",
                    "--frontier-boozer-trust-threshold": "0.009",
                },
            ),
            (
                "achievement_chebyshev_sweep_v1",
                {"lane_id": "lane_tradeoff"},
                {
                    "frontier_reference_iota": 0.17,
                    "frontier_reference_volume": 0.104,
                    "frontier_reference_qa": 0.011,
                    "frontier_reference_boozer": 0.0075,
                    "frontier_chebyshev_rho": 0.02,
                    "frontier_chebyshev_sharpness": 18.0,
                    "frontier_chebyshev_weight_iota": 2.0,
                    "frontier_chebyshev_weight_volume": 1.5,
                    "frontier_chebyshev_weight_qa": 1.0,
                    "frontier_chebyshev_weight_boozer": 0.5,
                },
                {
                    "--frontier-scalarization-type": "achievement_chebyshev_sweep_v1",
                    "--frontier-chebyshev-rho": "0.02",
                    "--frontier-chebyshev-sharpness": "18.0",
                    "--frontier-chebyshev-weight-iota": "2.0",
                },
            ),
            (
                "epsilon_constraint_sweep_v1",
                {
                    "lane_id": "lane_safe_iota",
                    "iotas_weight": 250.0,
                    "frontier_volume_weight": 0.0,
                    "res_weight": 900.0,
                },
                {
                    "frontier_reference_iota": 0.15,
                    "frontier_reference_volume": 0.10,
                    "frontier_reference_qa": 0.011,
                    "frontier_reference_boozer": 0.007,
                    "epsilon_constraint_qa_max": 0.011,
                    "epsilon_constraint_boozer_max": 0.007,
                    "frontier_epsilon_penalty_weight": 9.0,
                },
                {
                    "--frontier-scalarization-type": "epsilon_constraint_sweep_v1",
                    "--epsilon-constraint-qa-max": "0.011",
                    "--epsilon-constraint-boozer-max": "0.007",
                    "--frontier-epsilon-penalty-weight": "9.0",
                },
            ),
        ]
        for scalarization_type, overrides, params, expected_flags in cases:
            with self.subTest(scalarization_type=scalarization_type):
                command = self._frontier_lane_command(
                    frontier_campaign_module,
                    scalarization_type=scalarization_type,
                    scalarization_params=params,
                    **overrides,
                )
                self._assert_command_values(command, expected_flags)

    def test_generate_frontier_lane_specs_full_simplex_mode_uses_seed_reference_metrics(self):
        module = load_frontier_scalarization_module()

        # 4 directions == C(1+4-1, 4-1) is the partition=1 Das-Dennis count.
        # Selection-by-rounding over a larger family is no longer permitted; the
        # request must match a Das-Dennis count exactly when partitions is None.
        lane_specs = self._generate_lanes(
            module,
            reference_mode=module.FRONTIER_REFERENCE_MODE_ACHIEVEMENT_FULL_SIMPLEX,
            num_lanes=4,
            stage2_results={"FINAL_IOTA": 0.17, "FINAL_VOLUME": 0.105, "NONQS_RATIO": 0.011, "BOOZER_RESIDUAL": 0.007},
        )

        self.assertEqual(len(lane_specs), 4)
        self.assertTrue(all(lane.scalarization_type == module.FRONTIER_REFERENCE_MODE_ACHIEVEMENT for lane in lane_specs))
        self.assertTrue(all(lane.scalarization_params["frontier_reference_iota"] == 0.17 and lane.scalarization_params["frontier_reference_volume"] == 0.105 for lane in lane_specs))
        self.assertTrue(all(lane.scalarization_params["frontier_chebyshev_sharpness"] == 12.0 for lane in lane_specs))
        self.assertTrue(
            all(
                lane.scalarization_params["frontier_chebyshev_weight_iota"] > 0.0
                and lane.scalarization_params["frontier_chebyshev_weight_volume"] > 0.0
                and lane.scalarization_params["frontier_chebyshev_weight_qa"] > 0.0
                and lane.scalarization_params["frontier_chebyshev_weight_boozer"] > 0.0
                for lane in lane_specs
            )
        )

    def test_generate_frontier_lane_specs_full_simplex_rejects_non_dasdennis_count(self):
        module = load_frontier_scalarization_module()

        with self.assertRaisesRegex(ValueError, "does not match any Das-Dennis"):
            self._generate_lanes(
                module,
                reference_mode=module.FRONTIER_REFERENCE_MODE_ACHIEVEMENT_FULL_SIMPLEX,
                num_lanes=5,
                stage2_results={"FINAL_IOTA": 0.17, "FINAL_VOLUME": 0.105, "NONQS_RATIO": 0.011, "BOOZER_RESIDUAL": 0.007},
            )

    def test_generate_frontier_lane_specs_full_simplex_partitions_emit_full_direction_family(self):
        module = load_frontier_scalarization_module()

        lane_specs = self._generate_lanes(
            module,
            reference_mode=module.FRONTIER_REFERENCE_MODE_ACHIEVEMENT_FULL_SIMPLEX,
            num_lanes=4,
            stage2_results={"FINAL_IOTA": 0.17, "FINAL_VOLUME": 0.105, "NONQS_RATIO": 0.011, "BOOZER_RESIDUAL": 0.007},
            full_simplex_partitions=1,
        )
        directions = module.generate_frontier_reference_directions(
            requested_num_directions=4,
            n_dim=4,
            partitions=1,
        )

        self.assertEqual(len(lane_specs), 4)
        for lane in lane_specs:
            weights = (
                lane.scalarization_params["frontier_chebyshev_weight_iota"],
                lane.scalarization_params["frontier_chebyshev_weight_volume"],
                lane.scalarization_params["frontier_chebyshev_weight_qa"],
                lane.scalarization_params["frontier_chebyshev_weight_boozer"],
            )
            self.assertAlmostEqual(sum(weights), 1.0, places=15)
            for weight in weights:
                self.assertGreaterEqual(weight, module._WEIGHT_FLOOR / 2.0)
        # Each direction places ~1.0 on a single axis; flooring three other
        # components at ``_WEIGHT_FLOOR`` and renormalizing to a unit simplex
        # leaves the dominant axis at ``1 / (1 + 3·_WEIGHT_FLOOR)``.
        dominant_weight = 1.0 / (1.0 + 3.0 * module._WEIGHT_FLOOR)
        floor_weight = module._WEIGHT_FLOOR / (1.0 + 3.0 * module._WEIGHT_FLOOR)
        observed_weight_vectors = {
            (
                lane.scalarization_params["frontier_chebyshev_weight_iota"],
                lane.scalarization_params["frontier_chebyshev_weight_volume"],
                lane.scalarization_params["frontier_chebyshev_weight_qa"],
                lane.scalarization_params["frontier_chebyshev_weight_boozer"],
            )
            for lane in lane_specs
        }
        expected_weight_vectors = {
            (dominant_weight, floor_weight, floor_weight, floor_weight),
            (floor_weight, dominant_weight, floor_weight, floor_weight),
            (floor_weight, floor_weight, dominant_weight, floor_weight),
            (floor_weight, floor_weight, floor_weight, dominant_weight),
        }
        observed_sorted = sorted(observed_weight_vectors, key=np.argmax)
        expected_sorted = sorted(expected_weight_vectors, key=np.argmax)
        self.assertEqual(len(observed_sorted), len(expected_sorted))
        for observed, expected in zip(observed_sorted, expected_sorted):
            np.testing.assert_allclose(observed, expected, rtol=0.0, atol=1.0e-15)
        self.assertEqual(len(directions), 4)
        self.assertEqual(
            set(directions),
            {
                (1.0, 0.0, 0.0, 0.0),
                (0.0, 1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            },
        )

    def test_lane_rng_seed_uses_explicit_lane_index(self):
        execution_module = load_frontier_campaign_execution_module()

        self.assertEqual(
            (
                execution_module.lane_rng_seed(42, lane_index=0),
                execution_module.lane_rng_seed(42, lane_index=3),
            ),
            (42, 45),
        )

    def test_chebyshev_goal_uses_separate_qs_scale(self):
        # F1.1: ``qs_scale`` decouples the Chebyshev gradient step from
        # ``qs_reference``. To isolate the denominator, set every metric
        # exactly at its reference (deltas all zero, softmax weights uniform);
        # the only term that varies between configurations is the
        # ``1 / qs_scale`` factor in the gradient.
        module = load_frontier_scalarization_module()

        def config(qs_scale):
            return SimpleNamespace(
                chebyshev_weight_iota=1.0,
                chebyshev_weight_volume=1.0,
                chebyshev_weight_qa=1.0,
                chebyshev_weight_boozer=1.0,
                iota_reference=0.18,
                volume_reference=0.12,
                qs_reference=1.0,
                qs_scale=qs_scale,
                boozer_reference=0.006,
                boozer_scale=0.0015,
                iota_scale=0.03,
                volume_scale=0.02,
                chebyshev_sharpness=12.0,
                chebyshev_rho=1.0e-3,
            )

        objective_eval = {
            "J_iota_metric": 0.18,
            "J_volume_metric": 0.12,
            "J_QS": 1.0,
            "J_Boozer": 0.006,
            "dJ_iota_metric": [0.0, 0.0],
            "dJ_volume_metric": [0.0, 0.0],
            "dJ_QS": [1.0, 0.0],
            "dJ_Boozer": [0.0, 0.0],
        }
        eval_half = module._frontier_chebyshev_goal(objective_eval, config(qs_scale=0.5))
        eval_one = module._frontier_chebyshev_goal(objective_eval, config(qs_scale=1.0))
        self.assertAlmostEqual(
            float(eval_half["frontier_scalarization_grad"][0])
            / float(eval_one["frontier_scalarization_grad"][0]),
            2.0,
            places=10,
        )

    def test_excess_penalty_uses_qs_scale_not_qs_reference(self):
        # F1.3: epsilon excess penalty divides by ``qs_scale``, not
        # ``qs_reference``. Holding qs_reference fixed and varying qs_scale
        # must change the penalty value.
        module = load_frontier_scalarization_module()

        def config(qs_scale):
            return SimpleNamespace(
                qs_reference=0.010,
                boozer_reference=0.006,
                qs_scale=qs_scale,
                boozer_scale=0.003,
                epsilon_constraint_qa_max=0.012,
                epsilon_constraint_boozer_max=None,
                epsilon_penalty_weight=4.0,
            )

        objective_eval = {
            "J_QS": 0.020,
            "J_Boozer": 0.005,
            "dJ_QS": np.asarray([1.0]),
            "dJ_Boozer": np.asarray([0.0]),
        }
        penalty_at_scale_one = module._frontier_epsilon_penalties(
            objective_eval,
            frontier_goal_config=config(0.005),
        )
        penalty_at_scale_two = module._frontier_epsilon_penalties(
            objective_eval,
            frontier_goal_config=config(0.010),
        )
        # Doubling qs_scale halves the excess_ratio; penalty ∝ ratio^2 so
        # penalty quarters.
        self.assertAlmostEqual(
            penalty_at_scale_two["qa_error"]["penalty"] * 4.0,
            penalty_at_scale_one["qa_error"]["penalty"],
            places=12,
        )

    def test_achievement_spec_v1_loads_fail(self):
        # F1.1 / F1.3: the schema bump from v1 to v2 is a hard load-time
        # failure. Old specs must be migrated, not silently accepted.
        module = load_frontier_scalarization_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            stale_path = self._write_spec(
                tmpdir,
                "achievement_v1.json",
                {
                    "schema_version": "frontier_achievement_spec_v1",
                    "lanes": [
                        {
                            "lane_id": "lane_a",
                            "reference_point": {
                                "iota": 0.17,
                                "volume": 0.105,
                                "qa_error": 0.011,
                                "boozer_residual": 0.007,
                            },
                        }
                    ],
                },
            )

            with self.assertRaisesRegex(
                ValueError,
                "frontier_achievement_spec_v2",
            ):
                self._generate_lanes(
                    module,
                    reference_mode=module.FRONTIER_REFERENCE_MODE_ACHIEVEMENT,
                    reference_points_file=str(stale_path),
                )

    def test_physics_total_matches_across_alm_and_non_alm_branches(self):
        # F1.2: ``physics_total`` (ALM) and ``total`` (non-ALM) must both
        # include the full geometry-penalty bundle so a fixed physical
        # configuration produces identical pre-replacement values.
        module = load_frontier_scalarization_module()

        grad_dim = 3

        def make_eval():
            return {
                "J_QS_objective": 0.4,
                "dJ_QS_objective": np.zeros(grad_dim),
                "J_Boozer_objective": 0.3,
                "dJ_Boozer_objective": np.zeros(grad_dim),
                "J_iota": 0.1,
                "dJ_iota": np.zeros(grad_dim),
                "J_volume": 0.2,
                "dJ_volume": np.zeros(grad_dim),
                "J_len": 5.0,
                "dJ_len": np.ones(grad_dim),
                "J_cc": 1.5,
                "dJ_cc": np.full(grad_dim, 0.5),
                "J_cs": 0.75,
                "dJ_cs": np.full(grad_dim, 0.25),
                "J_curvature": 2.0,
                "dJ_curvature": np.full(grad_dim, 0.1),
                "J_poloidal_extent": 0.3,
                "dJ_poloidal_extent": np.full(grad_dim, 0.05),
                "total": 0.0,
                "grad": np.zeros(grad_dim),
            }

        config = SimpleNamespace(
            scalarization_type="weight_schedule_v1",
        )
        common_args = dict(
            enabled=True,
            frontier_goal_config=config,
            surface_iota_term=SimpleNamespace(
                J=lambda: 0.18,
                dJ=lambda: np.zeros(grad_dim),
            ),
            surface_volume_term=SimpleNamespace(
                J=lambda: 0.12,
                dJ=lambda: np.zeros(grad_dim),
            ),
            effective_res_weight=1.0,
            effective_iotas_weight=1.0,
            effective_volume_weight=1.0,
            length_weight=2.0,
            cc_weight=0.5,
            cs_weight=0.25,
            curvature_weight=0.1,
            poloidal_extent_weight=0.025,
        )

        non_alm_eval = module.apply_frontier_scalarization_override(
            make_eval(),
            **common_args,
        )

        alm_payload = make_eval()
        alm_payload["constraint_values"] = np.zeros(2)
        alm_payload["constraint_grads"] = np.zeros((2, grad_dim))
        alm_eval = module.apply_frontier_scalarization_override(
            alm_payload,
            alm_formulation="weighted_sum",
            alm_multipliers=np.zeros(2),
            alm_penalty=1.0,
            **common_args,
        )

        # Same physical configuration must yield the same pre-replacement
        # geometry+frontier total; non-ALM stores it under ``total`` and ALM
        # under ``physics_total`` / ``base_total``.
        self.assertAlmostEqual(
            float(non_alm_eval["total"]),
            float(alm_eval["physics_total"]),
            places=12,
        )
        self.assertAlmostEqual(
            float(alm_eval["physics_total"]),
            float(alm_eval["base_total"]),
            places=12,
        )

    def test_frontier_geometry_bundle_includes_width_and_self_intersection(self):
        module = load_frontier_scalarization_module()
        grad_dim = 2
        objective_eval = {
            "J_QS_objective": 0.0,
            "dJ_QS_objective": np.zeros(grad_dim),
            "J_Boozer_objective": 0.0,
            "dJ_Boozer_objective": np.zeros(grad_dim),
            "J_iota": 0.0,
            "dJ_iota": np.zeros(grad_dim),
            "J_volume": 0.0,
            "dJ_volume": np.zeros(grad_dim),
            "J_len": 0.0,
            "dJ_len": np.zeros(grad_dim),
            "J_cc": 0.0,
            "dJ_cc": np.zeros(grad_dim),
            "J_cs": 0.0,
            "dJ_cs": np.zeros(grad_dim),
            "J_curvature": 0.0,
            "dJ_curvature": np.zeros(grad_dim),
            "J_poloidal_extent": 0.0,
            "dJ_poloidal_extent": np.zeros(grad_dim),
            "J_coil_width": 0.02,
            "dJ_coil_width": np.array([2.0, 3.0]),
            "coil_width_min_threshold": 0.05,
            "coil_width_max_threshold": 0.17,
            "J_self_intersect": 0.2,
            "dJ_self_intersect": np.array([0.5, -0.25]),
            "total": 0.0,
            "grad": np.zeros(grad_dim),
        }
        scalarized = module.apply_frontier_scalarization_override(
            objective_eval,
            enabled=True,
            frontier_goal_config=SimpleNamespace(
                scalarization_type="weight_schedule_v1",
            ),
            surface_iota_term=SimpleNamespace(
                J=lambda: 0.0,
                dJ=lambda: np.zeros(grad_dim),
            ),
            surface_volume_term=SimpleNamespace(
                J=lambda: 0.0,
                dJ=lambda: np.zeros(grad_dim),
            ),
            effective_res_weight=0.0,
            effective_iotas_weight=0.0,
            effective_volume_weight=0.0,
            length_weight=0.0,
            cc_weight=0.0,
            cs_weight=0.0,
            curvature_weight=0.0,
            poloidal_extent_weight=0.0,
            width_weight=10.0,
            selfint_weight=4.0,
        )

        self.assertAlmostEqual(float(scalarized["total"]), 0.8045)
        np.testing.assert_allclose(
            scalarized["grad"],
            np.array([1.4, -1.9]),
            rtol=0.0,
            atol=1.0e-12,
        )
