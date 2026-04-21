import importlib
import sys
import unittest
from pathlib import Path

import numpy as np


EXAMPLE_ROOT = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "single_stage_optimization"
)
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))


def load_frontier_constraints_module():
    return importlib.import_module("banana_opt.frontier_constraints")


def load_search_policy_module():
    return importlib.import_module("banana_opt.single_stage_search_policy")


class FrontierConstraintTests(unittest.TestCase):
    def test_annotate_search_evaluation_finiteness_flags_nonfinite_fields(self):
        module = load_frontier_constraints_module()

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
            {
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
            },
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

    def test_evaluate_frontier_hardware_search_penalty_scales_with_max_violation_ratio(self):
        module = load_frontier_constraints_module()

        penalty = module.evaluate_frontier_hardware_search_penalty(
            {
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
            },
            previous_objective=3.5,
            penalty_scale=4.0,
        )

        self.assertAlmostEqual(penalty["max_violation_ratio"], 0.25)
        self.assertAlmostEqual(penalty["penalty"], 3.5)

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

    def test_evaluate_frontier_hard_invalidation_rejects_nonfinite_search_eval(self):
        module = load_frontier_constraints_module()

        invalidation = module.evaluate_frontier_hard_invalidation(
            search_eval={
                "finite_eval_ok": False,
                "nonfinite_fields": ["total", "grad"],
            },
            surface_success=True,
        )

        self.assertTrue(invalidation["invalid"])
        self.assertEqual(invalidation["reason"], "nonfinite_evaluation")
        self.assertEqual(invalidation["fields"], ["total", "grad"])

    def test_evaluate_frontier_hard_invalidation_classifies_surface_solve_failure(self):
        module = load_frontier_constraints_module()

        invalidation = module.evaluate_frontier_hard_invalidation(
            search_eval={"finite_eval_ok": True},
            surface_success=False,
            surface_status={
                "solve_success": [False, True],
                "self_intersections": [False, False],
                "volumes_ordered": True,
                "gap_ok": True,
                "vessel_gap_ok": True,
                "nesting_ok": True,
            },
        )

        self.assertTrue(invalidation["invalid"])
        self.assertEqual(invalidation["reason"], "surface_solve_failed")
        self.assertEqual(invalidation["fields"], ["solve_success"])

    def test_evaluate_frontier_hard_invalidation_classifies_unrestorable_geometry(self):
        module = load_frontier_constraints_module()

        invalidation = module.evaluate_frontier_hard_invalidation(
            search_eval={"finite_eval_ok": True},
            surface_success=False,
            surface_status={
                "solve_success": [True, True],
                "self_intersections": [True, False],
                "volumes_ordered": True,
                "gap_ok": True,
                "vessel_gap_ok": True,
                "nesting_ok": True,
            },
        )

        self.assertTrue(invalidation["invalid"])
        self.assertEqual(invalidation["reason"], "geometry_state_unrestorable")
        self.assertEqual(invalidation["fields"], ["self_intersections"])
