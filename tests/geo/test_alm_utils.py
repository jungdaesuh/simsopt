import importlib.util
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


ALM_UTILS_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "single_stage_optimization"
    / "alm_utils.py"
)


def load_alm_utils_module():
    spec = importlib.util.spec_from_file_location(
        f"alm_utils_{uuid.uuid4().hex}",
        ALM_UTILS_MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ResidualHelperTests(unittest.TestCase):
    def test_bound_residuals_clamp_satisfied_constraints(self):
        module = load_alm_utils_module()

        self.assertEqual(module.upper_bound_residual(1.0, 2.0), 0.0)
        self.assertEqual(module.upper_bound_residual(2.5, 2.0), 0.5)
        self.assertEqual(module.lower_bound_residual(2.5, 2.0), 0.0)
        self.assertEqual(module.lower_bound_residual(1.5, 2.0), 0.5)

    def test_normalized_quadratic_penalty_residual_undoes_extra_square(self):
        module = load_alm_utils_module()

        residual, grad = module.normalized_quadratic_penalty_residual(
            penalty_value=18.0,
            penalty_grad=np.array([4.0, -2.0]),
            normalization=2.0,
        )

        self.assertAlmostEqual(residual, 3.0)
        np.testing.assert_allclose(grad, np.array([1.0 / 3.0, -1.0 / 6.0]))

    def test_normalized_lp_penalty_residual_extracts_single_level_violation(self):
        module = load_alm_utils_module()

        residual, grad = module.normalized_lp_penalty_residual(
            penalty_value=12.0,
            penalty_grad=np.array([24.0]),
            p=4,
            normalization=3.0,
        )

        self.assertAlmostEqual(residual, 2.0)
        np.testing.assert_allclose(grad, np.array([1.0]))

    def test_augmented_inequality_objective_uses_projected_multiplier_shift(self):
        module = load_alm_utils_module()

        evaluation = module.augmented_inequality_objective(
            base_value=3.0,
            base_grad=np.array([1.0, -1.0]),
            constraint_values=np.array([-1.0, 0.5]),
            constraint_grads=[np.array([2.0, 0.0]), np.array([0.0, 4.0])],
            multipliers=np.array([0.5, 1.0]),
            penalty=2.0,
        )

        self.assertAlmostEqual(evaluation["total"], 3.6875)
        np.testing.assert_allclose(evaluation["grad"], np.array([1.0, 7.0]))
        self.assertAlmostEqual(evaluation["max_violation"], 0.5)


class MinimizeAlmTests(unittest.TestCase):
    def test_build_inner_options_caps_boxed_inner_work_in_feasible_continuations(self):
        module = load_alm_utils_module()

        options = module._build_inner_options(
            {"maxiter": 300, "ftol": 1e-15, "gtol": 1e-15},
            update_stationarity_tol=1.0,
            trust_radius=0.05,
            continuation_iteration=1,
            feasible_enough=True,
        )

        self.assertEqual(options["maxiter"], 8)
        self.assertEqual(options["maxls"], 8)
        self.assertEqual(options["maxfun"], 128)
        self.assertGreaterEqual(options["ftol"], 1e-11)
        self.assertGreaterEqual(options["gtol"], 1e-4)

    def test_build_inner_options_leaves_unbounded_solves_on_default_linesearch_budget(self):
        module = load_alm_utils_module()

        options = module._build_inner_options(
            {"maxiter": 300, "ftol": 1e-15, "gtol": 1e-15},
            update_stationarity_tol=1.0,
            trust_radius=None,
            continuation_iteration=0,
            feasible_enough=False,
        )

        self.assertEqual(options["maxiter"], 300)
        self.assertEqual(options["maxls"], 20)
        self.assertNotIn("maxfun", options)
        self.assertGreaterEqual(options["gtol"], 1e-4)

    def test_minimize_alm_keeps_current_iterate_after_all_trials_are_rejected(self):
        module = load_alm_utils_module()
        settings = module.ALMSettings(
            max_outer_iterations=1,
            max_subproblem_continuations=1,
            trust_radius_init=0.1,
            trust_radius_min=0.01,
            trust_radius_shrink=0.5,
            trust_radius_grow=1.5,
            max_inner_attempts=2,
        )

        def evaluate_problem(x, multipliers, penalty):
            value = float(np.dot(x, x))
            return {
                "total": value,
                "grad": 2.0 * np.asarray(x, dtype=float),
                "constraint_values": np.array([0.0]),
                "stationarity_norm": float(np.linalg.norm(2.0 * np.asarray(x, dtype=float))),
            }

        def fake_minimize(fun, x, jac, method, bounds, callback, options):
            return SimpleNamespace(
                x=np.array([0.6]),
                nit=1,
                success=False,
                message="ITERATION LIMIT",
            )

        with patch.object(module, "minimize", side_effect=fake_minimize):
            result = module.minimize_alm(
                np.array([0.2]),
                ["demo_constraint"],
                evaluate_problem,
                settings,
                {"maxiter": 5, "ftol": 1e-12, "gtol": 1e-12},
            )

        self.assertFalse(result.success)
        np.testing.assert_allclose(result.x, np.array([0.2]))
        self.assertEqual(result.nit, 4)
        self.assertEqual(result.history[0]["action"], "subproblem_continue")
        self.assertEqual(result.history[0]["inner_attempts"], 2)
        self.assertFalse(result.history[0]["meaningful_progress"])
        self.assertAlmostEqual(result.history[0]["accepted_move_norm"], 0.0)
        self.assertEqual(result.history[1]["action"], "subproblem_limit")

    def test_minimize_alm_retries_with_smaller_trust_radius_after_abnormal_step(self):
        module = load_alm_utils_module()
        settings = module.ALMSettings(
            max_outer_iterations=1,
            trust_radius_init=0.1,
            trust_radius_min=0.01,
            trust_radius_shrink=0.5,
            trust_radius_grow=1.5,
            max_inner_attempts=2,
        )
        minimize_calls = []

        def evaluate_problem(x, multipliers, penalty):
            value = float(np.dot(x, x))
            return {
                "total": value,
                "grad": 2.0 * np.asarray(x, dtype=float),
                "constraint_values": np.array([0.0]),
                "stationarity_norm": float(np.linalg.norm(2.0 * np.asarray(x, dtype=float))),
            }

        def fake_minimize(fun, x, jac, method, bounds, callback, options):
            minimize_calls.append(bounds)
            if len(minimize_calls) == 1:
                return SimpleNamespace(
                    x=np.asarray(x, dtype=float),
                    nit=0,
                    success=False,
                    message="ABNORMAL: line search failed",
                )
            return SimpleNamespace(
                x=np.array([0.05]),
                nit=1,
                success=True,
                message="CONVERGENCE",
            )

        with patch.object(module, "minimize", side_effect=fake_minimize):
            result = module.minimize_alm(
                np.array([0.2]),
                ["demo_constraint"],
                evaluate_problem,
                settings,
                {"maxiter": 5, "ftol": 1e-12, "gtol": 1e-12},
            )

        self.assertFalse(result.success)
        self.assertEqual(len(minimize_calls), 2)
        self.assertEqual(minimize_calls[0], [(0.1, 0.30000000000000004)])
        self.assertEqual(minimize_calls[1], [(0.15000000000000002, 0.25)])
        self.assertEqual(result.history[0]["inner_attempts"], 2)
        self.assertAlmostEqual(result.history[0]["accepted_move_norm"], 0.15)
        self.assertAlmostEqual(result.trust_radius, 0.075)

    def test_minimize_alm_reports_feasibility_values_separately_from_solver_residuals(self):
        module = load_alm_utils_module()
        settings = module.ALMSettings(max_outer_iterations=1)

        def evaluate_problem(x, multipliers, penalty):
            return {
                "total": 0.0,
                "grad": np.zeros(1),
                "constraint_values": np.array([3.0]),
                "feasibility_values": np.array([1.0]),
                "max_feasibility_violation": 1.0,
                "stationarity_norm": 0.0,
            }

        result = module.minimize_alm(
            np.array([0.0]),
            ["demo_constraint"],
            evaluate_problem,
            settings,
            {"maxiter": 5, "ftol": 1e-12, "gtol": 1e-12},
        )

        self.assertFalse(result.success)
        self.assertEqual(result.constraint_values, [1.0])
        self.assertEqual(result.solver_constraint_values, [3.0])
        self.assertEqual(result.history[0]["constraint_values"], [1.0])
        self.assertEqual(result.history[0]["solver_constraint_values"], [3.0])

    def test_minimize_alm_increases_penalty_only_when_feasibility_is_bad(self):
        module = load_alm_utils_module()
        settings = module.ALMSettings(
            max_outer_iterations=2,
            penalty_init=1.0,
            penalty_scale=10.0,
            feasibility_tol=1e-12,
            stationarity_tol=1e-12,
        )

        def evaluate_problem(x, multipliers, penalty):
            return {
                "total": 0.0,
                "grad": np.zeros(1),
                "constraint_values": np.array([2.0]),
                "stationarity_norm": 0.0,
            }

        result = module.minimize_alm(
            np.array([0.0]),
            ["demo_constraint"],
            evaluate_problem,
            settings,
            {"maxiter": 5, "ftol": 1e-12, "gtol": 1e-12},
        )

        self.assertFalse(result.success)
        self.assertEqual(result.penalty, 10.0)
        self.assertEqual(result.history[0]["action"], "penalty_increase")
        self.assertEqual(result.history[1]["action"], "max_outer")

    def test_minimize_alm_updates_duals_without_growing_penalty_after_meaningful_progress(self):
        module = load_alm_utils_module()
        settings = module.ALMSettings(
            max_outer_iterations=2,
            penalty_init=1.0,
            penalty_scale=10.0,
            feasibility_tol=1e-8,
            stationarity_tol=1e-8,
        )

        def evaluate_problem(x, multipliers, penalty):
            x = np.asarray(x, dtype=float)
            return {
                "total": float(np.dot(x, x) + np.dot(multipliers, np.array([0.25]))),
                "grad": 2.0 * x,
                "constraint_values": np.array([0.25]),
                "stationarity_norm": float(np.linalg.norm(2.0 * x)),
            }

        def fake_minimize(fun, x, jac, method, bounds, callback, options):
            return SimpleNamespace(
                x=np.array([0.0]),
                nit=1,
                success=True,
                message="CONVERGENCE",
            )

        with patch.object(module, "minimize", side_effect=fake_minimize):
            result = module.minimize_alm(
                np.array([0.1]),
                ["demo_constraint"],
                evaluate_problem,
                settings,
                {"maxiter": 5, "ftol": 1e-12, "gtol": 1e-12},
            )

        self.assertFalse(result.success)
        self.assertEqual(result.penalty, 1.0)
        self.assertEqual(result.history[0]["action"], "dual_update")
        self.assertTrue(result.history[0]["meaningful_progress"])
        self.assertEqual(result.history[1]["multipliers"], [0.25])

    def test_minimize_alm_does_not_dual_update_after_zero_work_feasible_stall(self):
        module = load_alm_utils_module()
        settings = module.ALMSettings(
            max_outer_iterations=2,
            max_subproblem_continuations=1,
            trust_radius_init=0.1,
            trust_radius_min=0.01,
            trust_radius_shrink=0.5,
            trust_radius_grow=1.5,
            max_inner_attempts=1,
            penalty_init=1.0,
            penalty_scale=10.0,
            feasibility_tol=1e-8,
            stationarity_tol=1e-8,
        )

        def evaluate_problem(x, multipliers, penalty):
            return {
                "total": 0.0,
                "grad": np.array([0.02]),
                "constraint_values": np.array([0.0]),
                "stationarity_norm": 0.02,
            }

        def fake_minimize(fun, x, jac, method, bounds, callback, options):
            return SimpleNamespace(
                x=np.asarray(x, dtype=float),
                nit=0,
                success=False,
                message="ABNORMAL: line search failed",
            )

        with patch.object(module, "minimize", side_effect=fake_minimize):
            result = module.minimize_alm(
                np.array([0.0]),
                ["demo_constraint"],
                evaluate_problem,
                settings,
                {"maxiter": 5, "ftol": 1e-12, "gtol": 1e-15},
            )

        self.assertFalse(result.success)
        self.assertEqual(result.history[0]["action"], "subproblem_continue")
        self.assertEqual(result.history[0]["multipliers"], [0.0])
        self.assertAlmostEqual(result.history[0]["trust_radius"], 0.15000000000000002)
        self.assertEqual(result.history[1]["action"], "subproblem_limit")
        self.assertAlmostEqual(result.trust_radius, 0.15000000000000002)

    def test_minimize_alm_treats_same_point_after_nonzero_iterations_as_stall(self):
        module = load_alm_utils_module()
        settings = module.ALMSettings(
            max_outer_iterations=2,
            max_subproblem_continuations=1,
            trust_radius_init=0.1,
            trust_radius_min=0.01,
            trust_radius_shrink=0.5,
            trust_radius_grow=1.5,
            max_inner_attempts=1,
            penalty_init=1.0,
            penalty_scale=10.0,
            feasibility_tol=1e-8,
            stationarity_tol=1e-8,
        )

        def evaluate_problem(x, multipliers, penalty):
            return {
                "total": 0.0,
                "grad": np.array([0.02]),
                "constraint_values": np.array([0.0]),
                "stationarity_norm": 0.02,
            }

        def fake_minimize(fun, x, jac, method, bounds, callback, options):
            return SimpleNamespace(
                x=np.asarray(x, dtype=float),
                nit=3,
                success=False,
                message="STOP: no further improvement",
            )

        with patch.object(module, "minimize", side_effect=fake_minimize):
            result = module.minimize_alm(
                np.array([0.0]),
                ["demo_constraint"],
                evaluate_problem,
                settings,
                {"maxiter": 300, "ftol": 1e-15, "gtol": 1e-15},
            )

        self.assertFalse(result.success)
        self.assertEqual(result.history[0]["action"], "subproblem_continue")
        self.assertFalse(result.history[0]["meaningful_progress"])
        self.assertEqual(result.history[0]["inner_iterations"], 3)
        self.assertAlmostEqual(result.history[0]["accepted_move_norm"], 0.0)
        self.assertEqual(result.history[1]["action"], "subproblem_limit")
        self.assertAlmostEqual(result.trust_radius, 0.15000000000000002)

    def test_minimize_alm_limits_feasible_plateau_before_full_continuation_budget(self):
        module = load_alm_utils_module()
        settings = module.ALMSettings(
            max_outer_iterations=1,
            max_subproblem_continuations=20,
            trust_radius_init=0.1,
            trust_radius_min=0.01,
            trust_radius_shrink=0.5,
            trust_radius_grow=1.5,
            max_inner_attempts=1,
            penalty_init=1.0,
            penalty_scale=10.0,
            feasibility_tol=1e-8,
            stationarity_tol=1e-8,
        )

        def evaluate_problem(x, multipliers, penalty):
            return {
                "total": 0.0,
                "grad": np.array([0.02]),
                "constraint_values": np.array([0.0]),
                "stationarity_norm": 0.02,
            }

        def fake_minimize(fun, x, jac, method, bounds, callback, options):
            return SimpleNamespace(
                x=np.asarray(x, dtype=float),
                nit=2,
                success=False,
                message="STOP: plateau",
            )

        with patch.object(module, "minimize", side_effect=fake_minimize):
            result = module.minimize_alm(
                np.array([0.0]),
                ["demo_constraint"],
                evaluate_problem,
                settings,
                {"maxiter": 300, "ftol": 1e-15, "gtol": 1e-15},
            )

        self.assertFalse(result.success)
        self.assertEqual(len(result.history), 2)
        self.assertEqual(result.history[0]["action"], "subproblem_continue")
        self.assertEqual(result.history[0]["feasible_stall_count"], 1)
        self.assertEqual(result.history[1]["action"], "subproblem_limit")
        self.assertEqual(result.history[1]["feasible_stall_count"], 2)

    def test_minimize_alm_retries_subproblem_before_escalating_penalty(self):
        module = load_alm_utils_module()
        settings = module.ALMSettings(
            max_outer_iterations=2,
            penalty_init=1.0,
            penalty_scale=10.0,
            feasibility_tol=1e-8,
            stationarity_tol=1e-8,
        )

        def evaluate_problem(x, multipliers, penalty):
            return {
                "total": 0.0,
                "grad": np.array([2.0]),
                "constraint_values": np.array([0.25]),
                "stationarity_norm": 2.0,
            }

        result = module.minimize_alm(
            np.array([0.0]),
            ["demo_constraint"],
            evaluate_problem,
            settings,
            {"maxiter": 5, "ftol": 1e-12, "gtol": 1e-12},
        )

        self.assertFalse(result.success)
        self.assertEqual(result.penalty, 1.0)
        self.assertEqual(result.history[0]["action"], "subproblem_continue")

    def test_minimize_alm_accepts_kkt_stationarity_at_nearly_active_inequality_boundary(self):
        module = load_alm_utils_module()
        settings = module.ALMSettings(
            max_outer_iterations=1,
            max_subproblem_continuations=1,
            feasibility_tol=1e-8,
            stationarity_tol=1e-8,
        )

        def evaluate_problem(x, multipliers, penalty):
            return {
                "total": 0.0,
                "grad": np.array([1.0]),
                "constraint_values": np.array([-1.0e-4]),
                "dual_update_values": np.array([-1.0e-4]),
                "feasibility_values": np.array([0.0]),
                "constraint_grads": [np.array([-1.0])],
                "constraint_activity_tolerances": np.array([1.0e-3]),
                "stationarity_norm": 1.0,
            }

        def fake_minimize(fun, x, jac, method, bounds, callback, options):
            return SimpleNamespace(
                x=np.asarray(x, dtype=float),
                nit=0,
                success=False,
                message="ABNORMAL: line search failed",
            )

        with patch.object(module, "minimize", side_effect=fake_minimize):
            result = module.minimize_alm(
                np.array([0.0]),
                ["demo_constraint"],
                evaluate_problem,
                settings,
                {"maxiter": 5, "ftol": 1e-12, "gtol": 1e-15},
            )

        self.assertTrue(result.success)
        self.assertEqual(result.history[0]["action"], "converged")
        self.assertAlmostEqual(result.history[0]["raw_stationarity_norm"], 1.0)
        self.assertAlmostEqual(result.history[0]["kkt_stationarity_norm"], 0.0)
        self.assertAlmostEqual(result.history[0]["stationarity_norm"], 0.0)

    def test_minimize_alm_interrupts_inner_solver_when_kkt_gate_is_hit_in_callback(self):
        module = load_alm_utils_module()
        settings = module.ALMSettings(
            max_outer_iterations=1,
            max_subproblem_continuations=1,
            feasibility_tol=1e-8,
            stationarity_tol=1e-8,
        )

        def evaluate_problem(x, multipliers, penalty):
            return {
                "total": 0.0,
                "grad": np.array([1.0]),
                "constraint_values": np.array([-1.0e-4]),
                "dual_update_values": np.array([-1.0e-4]),
                "feasibility_values": np.array([0.0]),
                "constraint_grads": [np.array([-1.0])],
                "constraint_activity_tolerances": np.array([1.0e-3]),
                "stationarity_norm": 1.0,
            }

        def fake_minimize(fun, x, jac, method, bounds, callback, options):
            callback(np.asarray(x, dtype=float))
            raise AssertionError("callback should have terminated the solve")

        with patch.object(module, "minimize", side_effect=fake_minimize):
            result = module.minimize_alm(
                np.array([0.0]),
                ["demo_constraint"],
                evaluate_problem,
                settings,
                {"maxiter": 5, "ftol": 1e-12, "gtol": 1e-15},
            )

        self.assertTrue(result.success)
        self.assertIn("KKT stationarity gate", result.history[0]["inner_message"])

    def test_minimize_alm_skips_inner_solver_when_current_iterate_already_converged(self):
        module = load_alm_utils_module()
        settings = module.ALMSettings(
            max_outer_iterations=1,
            max_subproblem_continuations=1,
            feasibility_tol=1e-8,
            stationarity_tol=1e-8,
        )

        def evaluate_problem(x, multipliers, penalty):
            return {
                "total": 0.0,
                "grad": np.array([1.0]),
                "constraint_values": np.array([-1.0e-4]),
                "dual_update_values": np.array([-1.0e-4]),
                "feasibility_values": np.array([0.0]),
                "constraint_grads": [np.array([-1.0])],
                "constraint_activity_tolerances": np.array([1.0e-3]),
                "stationarity_norm": 1.0,
            }

        with patch.object(module, "minimize", side_effect=AssertionError("minimize should not run")):
            result = module.minimize_alm(
                np.array([0.0]),
                ["demo_constraint"],
                evaluate_problem,
                settings,
                {"maxiter": 5, "ftol": 1e-12, "gtol": 1e-15},
            )

        self.assertTrue(result.success)
        self.assertEqual(result.history[0]["inner_iterations"], 0)
        self.assertIn("current iterate already satisfies", result.history[0]["inner_message"])


if __name__ == "__main__":
    unittest.main()
