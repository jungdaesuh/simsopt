import unittest

import numpy as np

from geo._frontier_test_helpers import (
    load_frontier_constraints_module,
    load_search_evaluation_module,
    load_search_policy_module,
)


class FrontierConstraintTests(unittest.TestCase):
    def _hardware_result(self, **overrides):
        result = {
            "success": False,
            "violations": ["coil_coil_min_dist low", "max_curvature high"],
            "curve_curve_min_dist": 0.06,
            "cc_dist": 0.08,
            "curve_surface_min_dist": 0.04,
            "cs_dist": 0.03,
            "surface_vessel_min_dist": 0.05,
            "ss_dist": 0.05,
            "max_curvature": 48.0,
            "curvature_threshold": 40.0,
        }
        result.update(overrides)
        return result

    def _surface_status(self, **overrides):
        status = {
            "solve_success": [True, True],
            "self_intersections": [False, False],
            "volumes_ordered": True,
            "gap_ok": True,
            "vessel_gap_ok": True,
            "nesting_ok": True,
        }
        status.update(overrides)
        return status

    def test_annotate_search_evaluation_finiteness_flags_nonfinite_fields(self):
        module = load_search_evaluation_module()

        evaluation = module.annotate_search_evaluation_finiteness(
            {
                "total": float("nan"),
                "grad": np.array([1.0, np.inf]),
                "J_Boozer": 1.0e-5,
                "dJ_Boozer": np.array([1.0, 2.0]),
                "constraint_values": np.array([0.0, np.nan]),
                "constraint_grads": [
                    np.array([0.0, 1.0]),
                    np.array([1.0, np.nan]),
                ],
            }
        )

        self.assertFalse(evaluation["finite_eval_ok"])
        self.assertTrue(evaluation["nonfinite_evaluation"])
        self.assertEqual(
            evaluation["nonfinite_fields"],
            ["total", "grad", "constraint_values", "constraint_grads[1]"],
        )

    def test_annotate_search_evaluation_finiteness_flags_derived_frontier_fields(self):
        module = load_search_evaluation_module()

        evaluation = module.annotate_search_evaluation_finiteness(
            {
                "frontier_rank_total": float("inf"),
                "frontier_base_total": 1.0,
                "frontier_scalarization_total": float("nan"),
                "frontier_goal_grad": np.array([1.0, np.nan]),
                "frontier_scalarization_grad": np.array([1.0, 2.0]),
            }
        )

        self.assertFalse(evaluation["finite_eval_ok"])
        self.assertEqual(
            evaluation["nonfinite_fields"],
            [
                "frontier_rank_total",
                "frontier_scalarization_total",
                "frontier_goal_grad",
            ],
        )

    def test_evaluate_frontier_trust_penalty_matches_threshold_relative_contract(self):
        module = load_frontier_constraints_module()

        evaluation = module.annotate_frontier_search_eval(
            {
                "total": 2.0,
                "grad": np.array([1.0, -2.0]),
                "J_Boozer": 2.5e-5,
                "dJ_Boozer": np.array([2.0e-6, -1.0e-6]),
            },
            enabled=True,
            threshold=1.0e-5,
            penalty_scale=5.0e-5,
        )

        self.assertTrue(evaluation["finite_eval_ok"])
        self.assertFalse(evaluation["frontier_trust_ok"])
        self.assertAlmostEqual(evaluation["frontier_boozer_trust_excess"], 1.5e-5)
        self.assertAlmostEqual(evaluation["frontier_boozer_trust_excess_ratio"], 0.3)
        self.assertAlmostEqual(evaluation["frontier_trust_penalty"], 0.09)
        self.assertAlmostEqual(evaluation["frontier_rank_total"], 2.09)
        np.testing.assert_allclose(
            evaluation["grad"],
            np.array([1.024, -2.012]),
        )

    def test_evaluate_frontier_hardware_search_contract_reports_normalized_violations(self):
        module = load_frontier_constraints_module()
        search_policy_module = load_search_policy_module()

        contract = module.evaluate_frontier_hardware_search_contract(
            self._hardware_result(),
            policy=search_policy_module.HardwareSearchPolicy("hard", 0),
            context=search_policy_module.SearchContext(
                accepted_iterations=2,
                gate_scale=1.0,
                previous_objective=3.5,
            ),
        )

        self.assertTrue(contract["reject"])
        self.assertFalse(contract["warning_only"])
        self.assertAlmostEqual(contract["violation_ratios"]["curve_curve_min_dist"], 0.25)
        self.assertAlmostEqual(contract["violation_ratios"]["max_curvature"], 0.2)
        self.assertAlmostEqual(contract["max_violation_ratio"], 0.25)
        self.assertAlmostEqual(contract["rejection_increment"], 3.5)

    def test_frontier_hardware_ratios_ignore_diagnostic_surface_vessel_gap(self):
        module = load_frontier_constraints_module()

        penalty = module.evaluate_frontier_hardware_search_penalty(
            self._hardware_result(
                violations=["surface_vessel_min_dist below diagnostic reference"],
                curve_curve_min_dist=0.08,
                max_curvature=40.0,
                surface_vessel_min_dist=0.0,
                constraints={
                    "surface_vessel_min_dist": {
                        "threshold": 0.04,
                        "violation": 0.04,
                    }
                },
                violation_ratios={
                    "surface_vessel_min_dist": 1.0,
                    "plasma_vessel_spacing": 1.0,
                },
            ),
            previous_objective=3.5,
            penalty_scale=4.0,
        )

        self.assertNotIn("surface_vessel_min_dist", penalty["violation_ratios"])
        self.assertNotIn("plasma_vessel_spacing", penalty["violation_ratios"])
        self.assertAlmostEqual(penalty["max_violation_ratio"], 0.0)
        self.assertAlmostEqual(penalty["penalty"], 0.0)

    def test_evaluate_frontier_topology_search_contract_uses_existing_deficit_scaling(self):
        module = load_frontier_constraints_module()

        contract = module.evaluate_frontier_topology_search_contract(
            {
                "enabled": True,
                "success": False,
                "survival_fraction": 0.5,
                "survival_threshold": 0.75,
            },
            previous_objective=42.0,
            penalty_scale=4.0,
        )

        self.assertTrue(contract["reject"])
        self.assertAlmostEqual(contract["deficit"], 0.25)
        self.assertAlmostEqual(contract["rejection_increment"], 84.0)

    def test_evaluate_frontier_hardware_search_penalty_cases(self):
        module = load_frontier_constraints_module()
        zero_penalties = {
            "coil_coil_spacing_penalty": 0.0,
            "max_curvature_penalty": 0.0,
        }
        cases = [
            ("normalized_current_values", self._hardware_result(), {}, 0.25, 3.5),
            (
                "explicit_violation_ratios",
                self._hardware_result(
                    violations=["coil_coil_spacing penalty 2.5e-1 exceeds 0"],
                    violation_ratios={
                        "coil_coil_spacing_penalty": 0.25,
                        "max_curvature_penalty": 0.0,
                    },
                    curve_curve_min_dist=None,
                ),
                {"coil_coil_spacing_penalty": 0.25, "max_curvature_penalty": 0.0},
                0.25,
                3.5,
            ),
            (
                "explicit_plus_current_ratios",
                self._hardware_result(
                    violations=["|banana_current| exceeds threshold"],
                    violation_ratios=zero_penalties,
                    banana_current_A=2.4e4,
                    banana_current_max_A=1.6e4,
                ),
                {"banana_current": 0.5, "coil_coil_spacing_penalty": 0.0},
                0.5,
                7.0,
            ),
            (
                "schema_constraint_ratios",
                self._hardware_result(
                    violations=["|banana_current| exceeds threshold"],
                    constraints={
                        "banana_current": {"threshold": 1.6e4, "violation": 4.0e3}
                    },
                    violation_ratios=zero_penalties,
                ),
                {"banana_current": 0.25},
                0.25,
                3.5,
            ),
        ]
        for name, result, expected_ratios, expected_max, expected_penalty in cases:
            with self.subTest(name=name):
                penalty = module.evaluate_frontier_hardware_search_penalty(
                    result,
                    previous_objective=3.5,
                    penalty_scale=4.0,
                )
                for ratio_name, expected in expected_ratios.items():
                    self.assertAlmostEqual(penalty["violation_ratios"][ratio_name], expected)
                self.assertAlmostEqual(penalty["max_violation_ratio"], expected_max)
                self.assertAlmostEqual(penalty["penalty"], expected_penalty)

    def test_evaluate_frontier_topology_search_penalty_scales_with_deficit(self):
        module = load_frontier_constraints_module()

        penalty = module.evaluate_frontier_topology_search_penalty(
            {
                "enabled": True,
                "success": False,
                "survival_fraction": 0.5,
                "survival_threshold": 0.75,
            },
            previous_objective=42.0,
            penalty_scale=4.0,
        )

        self.assertAlmostEqual(penalty["deficit"], 0.25)
        self.assertAlmostEqual(penalty["penalty"], 42.0)

    def test_evaluate_frontier_hard_invalidation_cases(self):
        module = load_frontier_constraints_module()
        cases = [
            (
                "nonfinite_search_eval",
                {"finite_eval_ok": False, "nonfinite_fields": ["total", "grad"]},
                True,
                None,
                "nonfinite_evaluation",
                ["total", "grad"],
            ),
            (
                "surface_solve_failure",
                {"finite_eval_ok": True},
                False,
                self._surface_status(solve_success=[False, True]),
                "surface_solve_failed",
                ["solve_success"],
            ),
            (
                "unrestorable_geometry",
                {"finite_eval_ok": True},
                False,
                self._surface_status(self_intersections=[True, False]),
                "geometry_state_unrestorable",
                ["self_intersections"],
            ),
        ]
        for name, search_eval, surface_success, surface_status, reason, fields in cases:
            with self.subTest(name=name):
                invalidation = module.evaluate_frontier_hard_invalidation(
                    search_eval=search_eval,
                    surface_success=surface_success,
                    surface_status=surface_status,
                )
                self.assertTrue(invalidation["invalid"])
                self.assertEqual(invalidation["reason"], reason)
                self.assertEqual(invalidation["fields"], fields)
