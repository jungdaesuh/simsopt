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
        np.testing.assert_allclose(evaluation["dual_update_values"], np.array([-1.0, 0.5]))
        np.testing.assert_allclose(evaluation["feasibility_values"], np.array([0.0, 0.5]))
        np.testing.assert_allclose(
            evaluation["constraint_grads"],
            [np.array([2.0, 0.0]), np.array([0.0, 4.0])],
        )
        self.assertAlmostEqual(evaluation["max_feasibility_violation"], 0.5)

    def test_incumbent_objective_value_prefers_promoted_physics_total(self):
        module = load_alm_utils_module()

        self.assertAlmostEqual(
            module._incumbent_objective_value(
                {
                    "physics_total": 7.5,
                    "base_value": 0.0,
                    "base_total": 0.0,
                    "total": 12.0,
                }
            ),
            7.5,
        )

    def test_augmented_objective_exposes_solver_constraint_metadata(self):
        module = load_alm_utils_module()

        evaluation = module.augmented_objective(
            base_value=3.0,
            base_grad=np.array([1.0, -1.0]),
            constraint_values=[0.5, 0.0],
            constraint_grads=[np.array([2.0, 0.0]), np.array([0.0, 0.0])],
            multipliers=np.array([1.0, 7.0]),
            penalty=10.0,
        )

        self.assertAlmostEqual(evaluation["total"], 4.75)
        np.testing.assert_allclose(evaluation["grad"], np.array([13.0, -1.0]))
        np.testing.assert_allclose(evaluation["dual_update_values"], np.array([0.5, 0.0]))
        np.testing.assert_allclose(evaluation["feasibility_values"], np.array([0.5, 0.0]))
        np.testing.assert_allclose(
            evaluation["constraint_grads"],
            [np.array([2.0, 0.0]), np.array([0.0, 0.0])],
        )
        self.assertAlmostEqual(evaluation["max_feasibility_violation"], 0.5)

    def test_project_nonnegative_multipliers_enforces_nonnegativity_and_cap(self):
        module = load_alm_utils_module()

        projected = module._project_nonnegative_multipliers(
            np.array([0.2, 0.0]),
            np.array([0.5, -1.0]),
            penalty=2.0,
            multiplier_max=1.0,
        )

        np.testing.assert_allclose(projected, np.array([1.0, 0.0]))


class MinimizeAlmTests(unittest.TestCase):
    @staticmethod
    def _quadratic_taylor_evaluation(x, gradient_scale: float):
        x = np.asarray(x, dtype=float)
        return {
            "total": 0.5 * float(np.dot(x, x)),
            "grad": float(gradient_scale) * x,
            "constraint_values": np.zeros(1),
            "stationarity_norm": float(np.linalg.norm(float(gradient_scale) * x)),
        }

    @staticmethod
    def _stage2_signal_evaluation(**overrides):
        evaluation = {
            "total": 0.0,
            "grad": np.zeros(1),
            "constraint_values": np.array([-0.2]),
            "dual_update_values": np.array([-0.2]),
            "feasibility_values": np.array([0.0]),
            "hard_signed_constraint_values": np.array([-0.2]),
            "hard_violation_values": np.array([0.0]),
            "surrogate_signed_constraint_values": np.array([-0.2]),
            "hard_dual_update_values": np.array([-0.2]),
            "constraint_grads": [np.array([-1.0])],
            "constraint_activity_tolerances": np.array([1.0e-3]),
            "stationarity_norm": 0.0,
        }
        evaluation.update(overrides)
        return evaluation

    def test_select_inner_solve_profile_uses_explicit_boxed_feasible_continuation_profile(self):
        module = load_alm_utils_module()

        profile = module._select_inner_solve_profile(
            trust_radius=0.05,
            continuation_iteration=1,
            feasible_enough=True,
        )

        self.assertEqual(profile.name, "boxed_feasible_continuation")

    def test_zero_trust_radius_disables_box_bounds_and_boxed_profile(self):
        module = load_alm_utils_module()

        self.assertIsNone(module._build_box_bounds(np.array([1.0, -2.0]), 0.0))
        profile = module._select_inner_solve_profile(
            trust_radius=0.0,
            continuation_iteration=0,
            feasible_enough=False,
        )

        self.assertEqual(profile.name, "unbounded")

    def test_minimize_alm_rejects_asymmetric_incumbent_hooks(self):
        module = load_alm_utils_module()
        settings = module.ALMSettings(max_outer_iterations=1)

        def evaluate_problem(x, multipliers, penalty):
            del multipliers, penalty
            x = np.asarray(x, dtype=float)
            return {
                "total": float(np.dot(x, x)),
                "grad": 2.0 * x,
                "constraint_values": np.zeros(1),
                "stationarity_norm": float(np.linalg.norm(2.0 * x)),
            }

        with self.assertRaisesRegex(ValueError, "must be provided together"):
            module.minimize_alm(
                np.array([0.0]),
                ["dummy"],
                evaluate_problem,
                settings,
                {"maxiter": 1},
                snapshot_accepted_state_fn=lambda: {"x": 0.0},
            )

    def test_minimize_alm_keeps_positional_snapshot_hook_compatibility(self):
        module = load_alm_utils_module()
        settings = module.ALMSettings(max_outer_iterations=1)

        def evaluate_problem(x, multipliers, penalty):
            del multipliers, penalty
            x = np.asarray(x, dtype=float)
            return {
                "total": 0.0,
                "grad": np.zeros_like(x),
                "constraint_values": np.zeros(1),
                "stationarity_norm": 0.0,
            }

        snapshot_calls = []
        restore_calls = []

        def snapshot_state():
            snapshot_calls.append("snapshot")
            return {"x": 0.0}

        def restore_state(state):
            restore_calls.append(state)

        result = module.minimize_alm(
            np.array([0.0]),
            ["dummy"],
            evaluate_problem,
            settings,
            {"maxiter": 1},
            None,
            None,
            None,
            snapshot_state,
            restore_state,
        )

        self.assertTrue(result.success)
        self.assertEqual(snapshot_calls, [])
        self.assertEqual(restore_calls, [])

    def test_build_inner_options_caps_boxed_inner_work_in_feasible_continuations(self):
        module = load_alm_utils_module()
        profile = module._select_inner_solve_profile(
            trust_radius=0.05,
            continuation_iteration=1,
            feasible_enough=True,
        )
        options = module._build_inner_options(
            {"maxiter": 300, "ftol": 1e-15, "gtol": 1e-15},
            update_stationarity_tol=1.0,
            profile=profile,
        )

        self.assertEqual(options["maxiter"], 80)
        self.assertEqual(options["maxls"], 40)
        self.assertEqual(options["maxfun"], 6400)
        self.assertGreaterEqual(options["ftol"], 1e-11)
        self.assertGreaterEqual(options["gtol"], 1e-4)

    def test_build_inner_options_leaves_unbounded_solves_on_default_linesearch_budget(self):
        module = load_alm_utils_module()

        profile = module._select_inner_solve_profile(
            trust_radius=None,
            continuation_iteration=0,
            feasible_enough=False,
        )
        options = module._build_inner_options(
            {"maxiter": 300, "ftol": 1e-15, "gtol": 1e-15},
            update_stationarity_tol=1.0,
            profile=profile,
        )

        self.assertEqual(options["maxiter"], 300)
        self.assertEqual(options["maxls"], 20)
        self.assertNotIn("maxfun", options)
        self.assertGreaterEqual(options["gtol"], 1e-4)

    def test_classify_infeasible_inner_stall_scales_move_tolerance_with_iterate_norm(self):
        module = load_alm_utils_module()

        current_eval = {
            "total": 10.0,
            "grad": np.array([1.0]),
            "constraint_values": np.array([0.5]),
            "feasibility_values": np.array([0.5]),
            "dual_update_values": np.array([0.5]),
            "stationarity_norm": 1.0,
        }
        candidate_eval = {
            "total": 10.0,
            "grad": np.array([1.0]),
            "constraint_values": np.array([0.5]),
            "feasibility_values": np.array([0.5]),
            "dual_update_values": np.array([0.5]),
            "stationarity_norm": 1.0,
        }

        stalled, false_success, reason = module._classify_infeasible_inner_stall(
            current_eval,
            candidate_eval,
            SimpleNamespace(success=False, message="STOP: plateau"),
            moved_norm=5.0e-8,
            move_tolerance=module._move_tolerance(np.array([1.0e4])),
            feasibility_gate=1.0e-6,
        )

        self.assertTrue(stalled)
        self.assertFalse(false_success)
        self.assertEqual(reason, "failed_inner_solve_without_feasibility_gain")

    def test_conditioning_metrics_capture_penalty_dominance(self):
        module = load_alm_utils_module()

        metrics = module._conditioning_metrics(
            {
                "total": 100.0,
                "base_total": 1.0,
                "grad": np.array([10.0, 0.0]),
                "base_grad": np.array([1.0, 0.0]),
            }
        )

        self.assertAlmostEqual(metrics["conditioning_base_objective"], 1.0)
        self.assertAlmostEqual(metrics["conditioning_penalty_objective"], 99.0)
        self.assertAlmostEqual(metrics["conditioning_penalty_objective_ratio"], 99.0)
        self.assertAlmostEqual(metrics["conditioning_total_grad_norm"], 10.0)
        self.assertAlmostEqual(metrics["conditioning_base_grad_norm"], 1.0)
        self.assertAlmostEqual(metrics["conditioning_penalty_grad_norm"], 9.0)
        self.assertAlmostEqual(metrics["conditioning_penalty_grad_ratio"], 9.0)
        self.assertAlmostEqual(metrics["penalty_gradient_norm"], 9.0)

    def test_minimize_alm_sanitizes_nonfinite_candidate_evaluations(self):
        module = load_alm_utils_module()
        settings = module.ALMSettings(
            max_outer_iterations=1,
            penalty_init=1.0,
            penalty_scale=10.0,
            feasibility_tol=1e-8,
            stationarity_tol=1e-8,
            trust_radius_init=0.1,
            trust_radius_min=0.01,
            trust_radius_shrink=0.5,
            trust_radius_grow=1.5,
        )
        candidate_x = np.array([0.2])

        def evaluate_problem(x, multipliers, penalty):
            del multipliers, penalty
            x = np.asarray(x, dtype=float)
            if np.array_equal(x, candidate_x):
                return {
                    "total": np.nan,
                    "grad": np.array([np.nan]),
                    "constraint_values": np.array([0.5]),
                    "feasibility_values": np.array([0.5]),
                    "dual_update_values": np.array([0.5]),
                    "stationarity_norm": 1.0,
                }
            return {
                "total": float(np.dot(x, x)),
                "base_value": float(np.dot(x, x)),
                "base_grad": 2.0 * x,
                "grad": 2.0 * x,
                "constraint_values": np.array([0.5]),
                "feasibility_values": np.array([0.5]),
                "dual_update_values": np.array([0.5]),
                "stationarity_norm": float(np.linalg.norm(2.0 * x)),
            }

        def fake_minimize(fun, x, jac, method, bounds, callback, options):
            del jac, method, bounds, callback, options
            fun(candidate_x)
            return SimpleNamespace(
                x=candidate_x.copy(),
                nit=1,
                success=False,
                message="ABNORMAL",
            )

        with patch.object(module, "minimize", side_effect=fake_minimize):
            result = module.minimize_alm(
                np.array([0.0]),
                ["demo_constraint"],
                evaluate_problem,
                settings,
                {"maxiter": 5, "ftol": 1e-12, "gtol": 1e-12},
            )

        self.assertEqual(result.history[0]["action"], "penalty_increase")
        self.assertTrue(result.history[0]["nonfinite_candidate_evaluation"])
        self.assertEqual(result.history[0]["nonfinite_candidate_fields"], ["total", "grad"])
        np.testing.assert_allclose(result.x, np.array([0.0]))

    def test_candidate_is_acceptable_allows_near_equal_feasible_trial(self):
        module = load_alm_utils_module()

        current_eval = {
            "total": 1.0e-3,
            "grad": np.array([0.1]),
            "constraint_values": np.array([0.0]),
            "stationarity_norm": 0.1,
        }
        candidate_eval = {
            "total": 1.0005e-3,
            "grad": np.array([0.08]),
            "constraint_values": np.array([0.0]),
            "stationarity_norm": 0.08,
        }

        acceptable = module._candidate_is_acceptable(
            current_eval,
            candidate_eval,
            SimpleNamespace(success=False, nit=2, message="plateau"),
            moved_norm=0.0,
            update_feasibility_tol=1e-6,
        )

        self.assertTrue(acceptable)

    def test_directional_taylor_test_passes_for_consistent_quadratic_gradient(self):
        module = load_alm_utils_module()

        def evaluate_problem(x, multipliers, penalty):
            return self._quadratic_taylor_evaluation(x, 1.0)

        result = module.run_directional_taylor_test(
            evaluate_problem,
            np.array([0.2, -0.4]),
            np.zeros(1),
            1.0,
            seed=7,
        )

        self.assertTrue(result["passed"])
        self.assertTrue(result["max_ratio"] is None or result["max_ratio"] < 0.3)

    def test_directional_taylor_test_flags_inconsistent_gradient(self):
        module = load_alm_utils_module()

        def evaluate_problem(x, multipliers, penalty):
            return self._quadratic_taylor_evaluation(x, 2.0)

        result = module.run_directional_taylor_test(
            evaluate_problem,
            np.array([0.2, -0.4]),
            np.zeros(1),
            1.0,
            seed=7,
        )

        self.assertFalse(result["passed"])
        self.assertGreater(result["max_ratio"], result["ratio_threshold"])

    def test_minimize_alm_solves_simple_quadratic_with_signed_upper_bound_constraint(self):
        module = load_alm_utils_module()
        settings = module.ALMSettings(
            max_outer_iterations=6,
            penalty_init=1.0,
            penalty_scale=10.0,
            feasibility_tol=1e-8,
            stationarity_tol=1e-8,
        )

        def evaluate_problem(x, multipliers, penalty):
            value = 0.5 * (x[0] - 2.0) ** 2
            grad = np.array([x[0] - 2.0])
            signed_constraint_value = np.array([x[0] - 1.0])
            constraint_grad = [np.array([1.0])]
            return module.augmented_inequality_objective(
                value,
                grad,
                signed_constraint_value,
                constraint_grad,
                multipliers,
                penalty,
            )

        result = module.minimize_alm(
            np.array([0.0]),
            ["x_upper_bound"],
            evaluate_problem,
            settings,
            {"maxiter": 50, "maxcor": 20, "ftol": 1e-12, "gtol": 1e-12},
        )

        self.assertTrue(result.success)
        self.assertLessEqual(result.x[0], 1.0 + 1e-6)
        self.assertLessEqual(result.constraint_values[0], 1e-6)
        self.assertAlmostEqual(result.history[0]["kkt_stationarity_norm"], 0.0)

    def test_minimize_alm_requires_signed_constraint_activity_for_boundary_kkt_success(self):
        module = load_alm_utils_module()
        settings = module.ALMSettings(
            max_outer_iterations=1,
            penalty_init=1.0,
            penalty_scale=10.0,
            feasibility_tol=1e-12,
            stationarity_tol=1e-12,
        )

        def evaluate_problem(x, multipliers, penalty):
            value = 0.5 * (x[0] - 2.0) ** 2
            grad = np.array([x[0] - 2.0])
            residual = module.upper_bound_residual(x[0], 1.0)
            constraint_grad = np.array([1.0]) if residual > 0.0 else np.array([0.0])
            return module.augmented_objective(
                value,
                grad,
                [residual],
                [constraint_grad],
                multipliers,
                penalty,
            )

        result = module.minimize_alm(
            np.array([0.0]),
            ["x_upper_bound"],
            evaluate_problem,
            settings,
            {"maxiter": 50, "maxcor": 20, "ftol": 1e-12, "gtol": 1e-12},
        )

        self.assertFalse(result.success)
        self.assertAlmostEqual(result.x[0], 1.0)
        self.assertEqual(result.multipliers, [0.0])

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
        self.assertEqual(result.nit, 2)
        self.assertEqual(result.history[0]["action"], "dual_update")
        self.assertEqual(result.history[0]["inner_attempts"], 2)
        self.assertFalse(result.history[0]["meaningful_progress"])
        self.assertAlmostEqual(result.history[0]["accepted_move_norm"], 0.0)
        self.assertEqual(result.history[0]["outer_termination"], "max_outer")

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
        self.assertEqual(result.hard_signed_constraint_values, [3.0])
        self.assertEqual(result.hard_violation_values, [1.0])
        self.assertEqual(result.surrogate_signed_constraint_values, [3.0])
        self.assertEqual(result.history[0]["constraint_values"], [1.0])
        self.assertEqual(result.history[0]["solver_constraint_values"], [3.0])
        self.assertEqual(result.history[0]["hard_signed_constraint_values"], [3.0])
        self.assertEqual(result.history[0]["hard_violation_values"], [1.0])
        self.assertEqual(result.history[0]["surrogate_signed_constraint_values"], [3.0])

    def test_stationarity_metrics_uses_raw_norm_when_stage2_signals_disagree(self):
        module = load_alm_utils_module()
        evaluation = {
            "total": 0.0,
            "grad": np.array([1.0]),
            "metric_grad": np.array([1.0]),
            "constraint_values": np.array([0.2]),
            "dual_update_values": np.array([0.2]),
            "feasibility_values": np.array([0.0]),
            "hard_signed_constraint_values": np.array([-1.0e-2]),
            "hard_violation_values": np.array([0.0]),
            "surrogate_signed_constraint_values": np.array([0.2]),
            "hard_dual_update_values": np.array([-1.0e-2]),
            "constraint_grads": [np.array([-1.0])],
            "constraint_activity_tolerances": np.array([1.0e-3]),
            "stationarity_norm": 1.0,
            "metric_stationarity_norm": 1.0,
        }

        routing_state = module._constraint_routing_state(
            evaluation,
            np.zeros(1),
            1.0,
            1.0e-8,
        )
        raw_stationarity_norm, kkt_stationarity_norm, effective_stationarity_norm, mismatch = (
            module._stationarity_metrics(
                evaluation,
                routing_state,
                1.0e-8,
            )
        )

        self.assertTrue(mismatch)
        self.assertIsNone(kkt_stationarity_norm)
        self.assertAlmostEqual(raw_stationarity_norm, 1.0)
        self.assertAlmostEqual(effective_stationarity_norm, 1.0)

    def test_constraint_routing_state_flags_boundary_mismatch_when_surrogate_shift_is_live(self):
        module = load_alm_utils_module()
        evaluation = self._stage2_signal_evaluation(
            total=5.0e-7,
            grad=np.array([1.0e-3]),
            constraint_values=np.array([1.0e-3]),
            dual_update_values=np.array([1.0e-3]),
            hard_signed_constraint_values=np.array([-5.0e-4]),
            surrogate_signed_constraint_values=np.array([1.0e-3]),
            hard_dual_update_values=np.array([-5.0e-4]),
            constraint_grads=[np.array([1.0])],
            stationarity_norm=1.0e-3,
        )

        routing_state = module._constraint_routing_state(
            evaluation,
            np.zeros(1),
            1.0,
            1.0e-8,
        )

        self.assertEqual(routing_state.hard_activity_mask.tolist(), [True])
        self.assertEqual(routing_state.surrogate_activity_mask.tolist(), [True])
        self.assertTrue(routing_state.signal_mismatch_active)
        self.assertTrue(routing_state.hard_positive_shift_zero)
        self.assertFalse(routing_state.surrogate_positive_shift_zero)

    def test_minimize_alm_returns_constraints_inactive_converged_for_stage2_zero_shift(self):
        module = load_alm_utils_module()
        settings = module.ALMSettings(
            max_outer_iterations=2,
            max_subproblem_continuations=1,
            feasibility_tol=1.0e-8,
            stationarity_tol=1.0e-8,
        )
        minimize_calls = []

        def evaluate_problem(x, multipliers, penalty):
            del x, multipliers, penalty
            return self._stage2_signal_evaluation()

        def fake_minimize(fun, x, jac, method, bounds, callback, options):
            del fun, jac, method, bounds, callback, options
            minimize_calls.append(np.asarray(x, dtype=float).copy())
            return SimpleNamespace(
                x=np.asarray(x, dtype=float).copy(),
                nit=1,
                success=True,
                message="CONVERGENCE",
            )

        with patch.object(module, "minimize", side_effect=fake_minimize):
            result = module.minimize_alm(
                np.array([0.0]),
                ["demo_constraint"],
                evaluate_problem,
                settings,
                {"maxiter": 5, "ftol": 1e-12, "gtol": 1e-12},
            )

        self.assertTrue(result.success)
        self.assertEqual(result.termination_reason, "constraints_inactive_converged")
        self.assertEqual(len(minimize_calls), 1)
        self.assertEqual(result.history[0]["action"], "constraints_inactive_converged")
        self.assertTrue(result.history[0]["hard_positive_shift_zero"])
        self.assertFalse(result.history[0]["signal_mismatch_active"])
        self.assertIsNone(result.history[0]["active_constraint_name"])
        self.assertEqual(result.hard_signed_constraint_values, [-0.2])
        self.assertEqual(result.hard_violation_values, [0.0])
        self.assertEqual(result.surrogate_signed_constraint_values, [-0.2])

    def test_minimize_alm_returns_constraints_inactive_stall_after_repeat_without_progress(self):
        module = load_alm_utils_module()
        settings = module.ALMSettings(
            max_outer_iterations=3,
            max_subproblem_continuations=1,
            max_inner_attempts=1,
            feasibility_tol=1.0e-8,
            stationarity_tol=1.0e-8,
        )
        minimize_calls = {"count": 0}

        def evaluate_problem(x, multipliers, penalty):
            del x, multipliers, penalty
            return self._stage2_signal_evaluation(
                grad=np.array([1.0]),
                stationarity_norm=1.0,
            )

        def fake_minimize(fun, x, jac, method, bounds, callback, options):
            del fun, jac, method, bounds, callback, options
            minimize_calls["count"] += 1
            return SimpleNamespace(
                x=np.asarray(x, dtype=float).copy(),
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
                {"maxiter": 5, "ftol": 1e-12, "gtol": 1e-12},
            )

        self.assertFalse(result.success)
        self.assertEqual(result.termination_reason, "constraints_inactive_stall")
        self.assertEqual(minimize_calls["count"], 3)
        self.assertEqual(result.outer_iterations, 2)
        self.assertEqual(result.history[0]["action"], "dual_update")
        self.assertEqual(result.history[1]["action"], "subproblem_continue")
        self.assertEqual(result.history[2]["action"], "constraints_inactive_stall")
        self.assertTrue(result.history[2]["hard_positive_shift_zero"])
        self.assertFalse(result.history[2]["signal_mismatch_active"])
        self.assertIsNone(result.history[2]["active_constraint_name"])

    def test_minimize_alm_escalates_penalty_after_repeated_stage2_signal_mismatch(self):
        module = load_alm_utils_module()
        settings = module.ALMSettings(
            max_outer_iterations=3,
            max_subproblem_continuations=2,
            trust_radius_init=0.1,
            trust_radius_min=0.01,
            trust_radius_shrink=0.5,
            trust_radius_grow=1.5,
            max_inner_attempts=1,
            feasibility_tol=1.0e-8,
            stationarity_tol=1.0e-8,
        )
        minimize_calls = {"count": 0}

        def evaluate_problem(x, multipliers, penalty):
            del multipliers, penalty
            point = float(np.asarray(x, dtype=float)[0])
            if point < 0.5:
                total = 2.0
                grad = np.array([1.0])
                stationarity_norm = 1.0
            else:
                total = 1.0
                grad = np.array([0.5])
                stationarity_norm = 0.5
            return self._stage2_signal_evaluation(
                total=total,
                grad=grad,
                constraint_values=np.array([0.2]),
                dual_update_values=np.array([0.2]),
                hard_signed_constraint_values=np.array([-1.0e-2]),
                surrogate_signed_constraint_values=np.array([0.2]),
                hard_dual_update_values=np.array([-1.0e-2]),
                stationarity_norm=stationarity_norm,
            )

        def fake_minimize(fun, x, jac, method, bounds, callback, options):
            del fun, jac, method, bounds, callback, options
            minimize_calls["count"] += 1
            if minimize_calls["count"] == 1:
                return SimpleNamespace(
                    x=np.array([1.0]),
                    nit=1,
                    success=True,
                    message="CONVERGENCE",
                )
            return SimpleNamespace(
                x=np.asarray(x, dtype=float).copy(),
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
                {"maxiter": 5, "ftol": 1e-12, "gtol": 1e-12},
            )

        self.assertFalse(result.success)
        self.assertEqual(result.termination_reason, "max_outer")
        self.assertEqual(minimize_calls["count"], 4)
        self.assertEqual(result.history[0]["action"], "subproblem_continue")
        self.assertTrue(result.history[0]["signal_mismatch_active"])
        self.assertEqual(result.history[1]["action"], "signal_mismatch_penalty_increase")
        self.assertTrue(result.history[1]["signal_mismatch_active"])
        self.assertTrue(result.history[1]["hard_positive_shift_zero"])
        self.assertFalse(result.history[1]["surrogate_max_value"] <= 0.0)

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
        self.assertEqual(result.termination_reason, "max_outer_after_infeasible_stall")
        self.assertEqual(result.history[0]["action"], "infeasible_stall_penalty_increase")
        self.assertEqual(result.history[1]["action"], "infeasible_stall_penalty_increase")
        self.assertEqual(result.history[1]["outer_termination"], "max_outer")

    def test_minimize_alm_short_circuits_zero_step_infeasible_stall(self):
        module = load_alm_utils_module()
        settings = module.ALMSettings(
            max_outer_iterations=1,
            max_subproblem_continuations=3,
            trust_radius_init=0.1,
            trust_radius_min=0.01,
            trust_radius_shrink=0.5,
            trust_radius_grow=1.5,
            max_inner_attempts=3,
            penalty_init=10.0,
            penalty_scale=10.0,
            feasibility_tol=1e-8,
            stationarity_tol=1e-8,
        )
        minimize_calls = []
        history_snapshots = []

        def evaluate_problem(x, multipliers, penalty):
            del multipliers, penalty
            x = np.asarray(x, dtype=float)
            return {
                "total": 1.0,
                "grad": np.zeros_like(x),
                "constraint_values": np.array([2.0]),
                "stationarity_norm": 0.0,
            }

        def fake_minimize(fun, x, jac, method, bounds, callback, options):
            del fun, jac, method, callback, options
            minimize_calls.append(bounds)
            return SimpleNamespace(
                x=np.asarray(x, dtype=float),
                nit=5,
                success=False,
                message="ABNORMAL: line search failed",
            )

        def history_callback(history, latest_entry, multipliers, penalty):
            history_snapshots.append(
                {
                    "history": history,
                    "latest_entry": latest_entry,
                    "multipliers": multipliers.tolist(),
                    "penalty": float(penalty),
                }
            )

        with patch.object(module, "minimize", side_effect=fake_minimize):
            result = module.minimize_alm(
                np.array([0.2]),
                ["demo_constraint"],
                evaluate_problem,
                settings,
                {"maxiter": 300, "ftol": 1e-15, "gtol": 1e-15},
                history_callback=history_callback,
            )

        self.assertFalse(result.success)
        self.assertEqual(result.termination_reason, "max_outer_after_infeasible_stall")
        self.assertEqual(len(minimize_calls), 1)
        self.assertEqual(len(history_snapshots), 1)
        self.assertEqual(result.history[0]["action"], "infeasible_stall_penalty_increase")
        self.assertTrue(result.history[0]["infeasible_stall"])
        self.assertEqual(result.history[0]["inner_attempts"], 1)
        self.assertIs(history_snapshots[-1]["history"], result.history)
        self.assertIsNot(history_snapshots[-1]["latest_entry"], result.history[0])
        self.assertEqual(history_snapshots[-1]["latest_entry"]["action"], "infeasible_stall_penalty_increase")
        self.assertEqual(history_snapshots[-1]["latest_entry"]["outer_termination"], "max_outer")
        history_snapshots[-1]["latest_entry"]["action"] = "mutated_by_callback_owner"
        history_snapshots[-1]["latest_entry"]["constraint_values"][0] = 99.0
        self.assertEqual(result.history[0]["action"], "infeasible_stall_penalty_increase")
        self.assertEqual(result.history[0]["constraint_values"], [2.0])

    def test_sanitize_nonfinite_evaluation_copies_only_owned_gradient_arrays(self):
        module = load_alm_utils_module()
        fallback_grad = np.array([1.0, 2.0])
        fallback_metric_grad = np.array([3.0, 4.0])
        fallback_base_grad = np.array([5.0, 6.0])
        borrowed_metadata = {"constraint": "borrowed"}
        fallback_evaluation = {
            "total": 1.0,
            "grad": fallback_grad,
            "metric_grad": fallback_metric_grad,
            "base_grad": fallback_base_grad,
            "constraint_values": np.array([0.25]),
            "metadata": borrowed_metadata,
        }

        sanitized = module._sanitize_nonfinite_inner_evaluation(
            {"total": np.nan, "grad": np.array([np.nan, 0.0])},
            fallback_evaluation=fallback_evaluation,
        )

        self.assertGreater(sanitized["total"], fallback_evaluation["total"])
        self.assertIs(sanitized["metadata"], borrowed_metadata)
        self.assertIs(sanitized["constraint_values"], fallback_evaluation["constraint_values"])
        for field, fallback_array in (
            ("grad", fallback_grad),
            ("metric_grad", fallback_metric_grad),
            ("base_grad", fallback_base_grad),
        ):
            self.assertIsNot(sanitized[field], fallback_array)
            np.testing.assert_allclose(sanitized[field], fallback_array)
        sanitized["grad"][0] = 99.0
        self.assertEqual(fallback_grad[0], 1.0)

    def test_minimize_alm_classifies_relative_reduction_false_success_as_infeasible_stall(self):
        module = load_alm_utils_module()
        settings = module.ALMSettings(
            max_outer_iterations=1,
            max_subproblem_continuations=1,
            trust_radius_init=0.1,
            trust_radius_min=0.01,
            trust_radius_shrink=0.5,
            trust_radius_grow=1.5,
            max_inner_attempts=1,
            penalty_init=10.0,
            penalty_scale=10.0,
            feasibility_tol=1e-8,
            stationarity_tol=1e-8,
            relaxed_feasibility_gate_cap=1e-2,
        )

        def evaluate_problem(x, multipliers, penalty):
            del x, multipliers, penalty
            return {
                "total": 3.778792e-03,
                "grad": np.zeros(1),
                "constraint_values": np.array([2.461054010712543e-2]),
                "stationarity_norm": 2.1899001642505436,
            }

        def fake_minimize(fun, x, jac, method, bounds, callback, options):
            del fun, jac, method, bounds, callback, options
            return SimpleNamespace(
                x=np.asarray(x, dtype=float),
                nit=351,
                success=True,
                message="CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH",
            )

        with patch.object(module, "minimize", side_effect=fake_minimize):
            result = module.minimize_alm(
                np.array([0.0]),
                ["coil_coil_spacing"],
                evaluate_problem,
                settings,
                {"maxiter": 300, "ftol": 1e-15, "gtol": 1e-15},
            )

        self.assertFalse(result.success)
        self.assertEqual(result.termination_reason, "max_outer_after_infeasible_stall")
        self.assertEqual(result.history[0]["action"], "infeasible_stall_penalty_increase")
        self.assertTrue(result.history[0]["infeasible_stall"])
        self.assertTrue(result.history[0]["inner_false_success"])
        self.assertEqual(
            result.history[0]["inner_stall_reason"],
            "relative_objective_termination_without_feasibility_gain",
        )
        self.assertEqual(result.history[0]["active_constraint_name"], "coil_coil_spacing")
        self.assertAlmostEqual(result.history[0]["effective_feasibility_tolerance"], 1.0e-2)
        self.assertAlmostEqual(result.history[0]["feasibility_delta"], 0.0)
        self.assertEqual(result.history[0]["outer_termination"], "max_outer")

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
                "total": float(np.dot(x, x) + np.dot(multipliers, np.array([5.0e-3]))),
                "grad": 2.0 * x,
                "constraint_values": np.array([5.0e-3]),
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
        self.assertEqual(result.history[1]["multipliers"], [5.0e-3])

    def test_minimize_alm_applies_multiplier_cap_on_dual_update(self):
        module = load_alm_utils_module()
        settings = module.ALMSettings(
            max_outer_iterations=2,
            penalty_init=1.0,
            penalty_scale=10.0,
            feasibility_tol=1e-8,
            stationarity_tol=1e-8,
            relaxed_feasibility_gate_cap=1.0,
            multiplier_max=0.2,
        )

        def evaluate_problem(x, multipliers, penalty):
            x = np.asarray(x, dtype=float)
            return {
                "total": float(np.dot(x, x)),
                "grad": 2.0 * x,
                "constraint_values": np.array([0.5]),
                "stationarity_norm": float(np.linalg.norm(2.0 * x)),
            }

        def fake_minimize(fun, x, jac, method, bounds, callback, options):
            del fun, jac, method, bounds, callback, options
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
        self.assertEqual(result.history[0]["action"], "dual_update")
        self.assertEqual(result.history[0]["multipliers"], [0.0])
        self.assertEqual(result.history[0]["post_update_multipliers"], [0.2])
        self.assertTrue(result.history[0]["multiplier_cap_binding"])
        self.assertEqual(result.history[0]["multiplier_cap_binding_indices"], [0])
        self.assertEqual(result.history[1]["multipliers"], [0.2])
        self.assertTrue(result.multiplier_cap_binding)
        self.assertEqual(result.multiplier_cap_binding_indices, [0])

    def test_minimize_alm_history_entry_has_stable_schema(self):
        module = load_alm_utils_module()
        settings = module.ALMSettings(
            max_outer_iterations=1,
            penalty_init=1.0,
            penalty_scale=10.0,
            feasibility_tol=1e-8,
            stationarity_tol=1e-8,
            relaxed_feasibility_gate_cap=1.0,
        )

        def evaluate_problem(x, multipliers, penalty):
            x = np.asarray(x, dtype=float)
            return {
                "total": float(np.dot(x, x)),
                "grad": 2.0 * x,
                "constraint_values": np.array([0.5]),
                "stationarity_norm": float(np.linalg.norm(2.0 * x)),
            }

        def fake_minimize(fun, x, jac, method, bounds, callback, options):
            del fun, jac, method, bounds, callback, options
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

        self.assertEqual(
            set(result.history[0]),
            {
                "outer_iteration",
                "continuation_iteration",
                "inner_iterations",
                "inner_success",
                "inner_message",
                "penalty",
                "max_violation",
                "stationarity_norm",
                "raw_stationarity_norm",
                "kkt_stationarity_norm",
                "constraint_values",
                "solver_constraint_values",
                "hard_signed_constraint_values",
                "hard_violation_values",
                "surrogate_signed_constraint_values",
                "hard_max_violation",
                "surrogate_max_value",
                "hard_positive_shift_zero",
                "signal_mismatch_active",
                "multipliers",
                "post_update_multipliers",
                "feasibility_tolerance",
                "effective_feasibility_tolerance",
                "stationarity_tolerance",
                "trust_radius",
                "inner_maxiter",
                "inner_maxls",
                "inner_maxfun",
                "inner_profile",
                "inner_attempts",
                "accepted_move_norm",
                "accepted_move_norm_scaled",
                "infeasible_stall_move_tolerance",
                "objective_delta",
                "feasibility_delta",
                "feasibility_delta_tolerance",
                "stationarity_delta",
                "meaningful_progress",
                "feasible_stall_count",
                "infeasible_stall",
                "inner_false_success",
                "inner_stall_reason",
                "active_violation_index",
                "active_constraint_name",
                "nonfinite_candidate_evaluation",
                "nonfinite_candidate_fields",
                "multiplier_cap_binding",
                "multiplier_cap_binding_indices",
                "conditioning_base_objective",
                "conditioning_penalty_objective",
                "conditioning_penalty_objective_ratio",
                "conditioning_total_grad_norm",
                "conditioning_base_grad_norm",
                "conditioning_penalty_grad_norm",
                "conditioning_penalty_grad_ratio",
                "penalty_gradient_norm",
                "action",
                "outer_termination",
            },
        )

    def test_minimize_alm_tracks_active_constraint_switching(self):
        module = load_alm_utils_module()
        settings = module.ALMSettings(
            max_outer_iterations=2,
            penalty_init=1.0,
            penalty_scale=10.0,
            feasibility_tol=1e-8,
            stationarity_tol=1e-8,
            relaxed_feasibility_gate_cap=1.0,
        )

        def evaluate_problem(x, multipliers, penalty):
            del x, penalty
            if float(multipliers[0]) <= 0.0:
                constraint_values = np.array([0.7, 0.1])
            else:
                constraint_values = np.array([0.05, 0.6])
            return {
                "total": 0.0,
                "grad": np.array([0.0]),
                "constraint_values": constraint_values,
                "stationarity_norm": 0.0,
            }

        def fake_minimize(fun, x, jac, method, bounds, callback, options):
            del fun, jac, method, bounds, callback, options
            return SimpleNamespace(
                x=np.array([0.0]),
                nit=1,
                success=True,
                message="CONVERGENCE",
            )

        with patch.object(module, "minimize", side_effect=fake_minimize):
            result = module.minimize_alm(
                np.array([0.0]),
                ["constraint_a", "constraint_b"],
                evaluate_problem,
                settings,
                {"maxiter": 5, "ftol": 1e-12, "gtol": 1e-12},
            )

        self.assertEqual(result.history[0]["active_constraint_name"], "constraint_a")
        self.assertEqual(result.history[1]["active_constraint_name"], "constraint_b")

    def test_minimize_alm_handles_high_dimension_many_constraints_smoke(self):
        module = load_alm_utils_module()
        settings = module.ALMSettings(
            max_outer_iterations=1,
            max_inner_attempts=1,
            penalty_init=1.0,
            penalty_scale=10.0,
            feasibility_tol=1e-8,
            stationarity_tol=1e-8,
            relaxed_feasibility_gate_cap=1.0,
        )
        constraint_names = [f"constraint_{index}" for index in range(12)]
        x0 = np.zeros(128)

        def evaluate_problem(x, multipliers, penalty):
            del multipliers, penalty
            return {
                "total": float(np.dot(x, x)),
                "grad": np.zeros_like(x),
                "constraint_values": np.linspace(1.0e-3, 1.2e-2, len(constraint_names)),
                "stationarity_norm": 0.0,
            }

        def fake_minimize(fun, x, jac, method, bounds, callback, options):
            del fun, jac, method, bounds, callback, options
            return SimpleNamespace(
                x=np.asarray(x, dtype=float),
                nit=1,
                success=True,
                message="CONVERGENCE",
            )

        with patch.object(module, "minimize", side_effect=fake_minimize):
            result = module.minimize_alm(
                x0,
                constraint_names,
                evaluate_problem,
                settings,
                {"maxiter": 5, "ftol": 1e-12, "gtol": 1e-12},
            )

        self.assertEqual(result.x.shape, (128,))
        self.assertEqual(len(result.history[0]["constraint_values"]), len(constraint_names))

    def test_minimize_alm_caps_relaxed_feasibility_gate_before_dual_update(self):
        module = load_alm_utils_module()
        settings = module.ALMSettings(
            max_outer_iterations=1,
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
            relaxed_feasibility_gate_cap=1e-2,
        )

        def evaluate_problem(x, multipliers, penalty):
            return {
                "total": 0.0,
                "grad": np.array([0.2]),
                "constraint_values": np.array([2.5e-2]),
                "stationarity_norm": 0.2,
            }

        def fake_minimize(fun, x, jac, method, bounds, callback, options):
            callback(np.asarray(x, dtype=float))
            return SimpleNamespace(
                x=np.asarray(x, dtype=float),
                nit=1,
                success=False,
                message="STOP: plateau",
            )

        with patch.object(module, "minimize", side_effect=fake_minimize):
            result = module.minimize_alm(
                np.array([0.0]),
                ["demo_constraint"],
                evaluate_problem,
                settings,
                {"maxiter": 30, "ftol": 1e-12, "gtol": 1e-12},
        )

        self.assertFalse(result.success)
        self.assertEqual(result.penalty, 1.0)
        self.assertEqual(result.termination_reason, "max_outer_after_infeasible_stall")
        self.assertEqual(result.history[0]["action"], "infeasible_stall_penalty_increase")
        self.assertEqual(result.history[0]["inner_message"], "STOP: plateau")
        self.assertEqual(result.history[0]["inner_profile"], "boxed_infeasible_initial")
        self.assertAlmostEqual(result.history[0]["feasibility_tolerance"], 1.0)
        self.assertAlmostEqual(result.history[0]["effective_feasibility_tolerance"], 1.0e-2)
        self.assertEqual(result.history[0]["outer_termination"], "max_outer")

    def test_minimize_alm_dual_updates_after_zero_work_feasible_stall_when_tolerances_are_met(self):
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
        self.assertEqual(result.history[0]["action"], "dual_update")
        self.assertEqual(result.history[0]["multipliers"], [0.0])
        self.assertAlmostEqual(result.history[0]["trust_radius"], 0.1)
        self.assertEqual(result.history[1]["action"], "dual_update")
        self.assertAlmostEqual(result.trust_radius, 0.1)

    def test_minimize_alm_dual_updates_same_point_after_nonzero_iterations_when_tolerances_are_met(self):
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
        self.assertEqual(result.history[0]["action"], "dual_update")
        self.assertFalse(result.history[0]["meaningful_progress"])
        self.assertEqual(result.history[0]["inner_iterations"], 3)
        self.assertAlmostEqual(result.history[0]["accepted_move_norm"], 0.0)
        self.assertEqual(result.history[1]["action"], "dual_update")
        self.assertAlmostEqual(result.trust_radius, 0.1)

    def test_minimize_alm_dual_updates_immediately_when_feasible_stall_meets_current_tolerances(self):
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
        self.assertEqual(len(result.history), 1)
        self.assertEqual(result.history[0]["action"], "dual_update")
        self.assertEqual(result.history[0]["feasible_stall_count"], 0)
        self.assertEqual(result.history[0]["outer_termination"], "max_outer")

    def test_minimize_alm_uses_full_outer_budget_after_tolerance_tightens(self):
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
        minimize_calls = {"count": 0}

        def evaluate_problem(x, multipliers, penalty):
            x = np.asarray(x, dtype=float)
            if float(multipliers[0]) <= 0.0:
                return {
                    "total": float(np.dot(x, x)),
                    "grad": 2.0 * x,
                    "constraint_values": np.array([0.01]),
                    "stationarity_norm": float(np.linalg.norm(2.0 * x)),
                }
            return {
                "total": 0.0,
                "grad": np.array([0.02]),
                "constraint_values": np.array([0.0]),
                "stationarity_norm": 0.02,
            }

        def fake_minimize(fun, x, jac, method, bounds, callback, options):
            minimize_calls["count"] += 1
            if minimize_calls["count"] == 1:
                return SimpleNamespace(
                    x=np.array([0.0]),
                    nit=1,
                    success=True,
                    message="CONVERGENCE",
                )
            return SimpleNamespace(
                x=np.asarray(x, dtype=float),
                nit=2,
                success=False,
                message="STOP: plateau",
            )

        with patch.object(module, "minimize", side_effect=fake_minimize):
            result = module.minimize_alm(
                np.array([0.1]),
                ["demo_constraint"],
                evaluate_problem,
                settings,
                {"maxiter": 300, "ftol": 1e-15, "gtol": 1e-15},
            )

        self.assertFalse(result.success)
        self.assertEqual(result.history[0]["action"], "dual_update")
        self.assertEqual(result.history[1]["action"], "dual_update")
        self.assertEqual(result.history[1]["outer_termination"], "max_outer")
        self.assertEqual(result.outer_iterations, 2)
        self.assertIn("max outer iterations", result.message)

    def test_minimize_alm_keeps_outer_loop_running_after_feasible_stall(self):
        module = load_alm_utils_module()
        settings = module.ALMSettings(
            max_outer_iterations=4,
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
        minimize_calls = {"count": 0}

        def evaluate_problem(x, multipliers, penalty):
            x = np.asarray(x, dtype=float)
            if float(multipliers[0]) <= 0.0:
                return {
                    "total": float(np.dot(x, x)),
                    "grad": 2.0 * x,
                    "constraint_values": np.array([0.01]),
                    "stationarity_norm": float(np.linalg.norm(2.0 * x)),
                }
            return {
                "total": 0.0,
                "grad": np.array([0.02]),
                "constraint_values": np.array([0.0]),
                "stationarity_norm": 0.02,
            }

        def fake_minimize(fun, x, jac, method, bounds, callback, options):
            minimize_calls["count"] += 1
            if minimize_calls["count"] == 1:
                return SimpleNamespace(
                    x=np.array([0.0]),
                    nit=1,
                    success=True,
                    message="CONVERGENCE",
                )
            return SimpleNamespace(
                x=np.asarray(x, dtype=float),
                nit=2,
                success=False,
                message="STOP: plateau",
            )

        with patch.object(module, "minimize", side_effect=fake_minimize):
            result = module.minimize_alm(
                np.array([0.1]),
                ["demo_constraint"],
                evaluate_problem,
                settings,
                {"maxiter": 300, "ftol": 1e-15, "gtol": 1e-15},
            )

        self.assertFalse(result.success)
        self.assertEqual(minimize_calls["count"], 6)
        self.assertEqual(result.outer_iterations, 4)
        self.assertEqual(result.history[0]["action"], "dual_update")
        self.assertEqual(result.history[1]["action"], "dual_update")
        self.assertEqual(result.history[-2]["action"], "subproblem_continue")
        self.assertEqual(result.history[-1]["action"], "subproblem_limit")
        self.assertEqual(result.history[-1]["outer_termination"], "max_outer")

    def test_minimize_alm_escalates_penalty_for_material_violation_above_capped_gate(self):
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
        self.assertEqual(result.penalty, 10.0)
        self.assertEqual(result.termination_reason, "max_outer_after_infeasible_stall")
        self.assertEqual(result.history[0]["action"], "infeasible_stall_penalty_increase")
        self.assertEqual(result.history[0]["effective_feasibility_tolerance"], 1.0e-2)
        self.assertEqual(result.history[1]["action"], "infeasible_stall_penalty_increase")
        self.assertEqual(result.history[1]["outer_termination"], "max_outer")

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

    def test_minimize_alm_restores_best_feasible_incumbent_by_base_objective(self):
        module = load_alm_utils_module()
        settings = module.ALMSettings(
            max_outer_iterations=3,
            max_subproblem_continuations=0,
            trust_radius_init=0.1,
            trust_radius_min=0.01,
            trust_radius_shrink=0.5,
            trust_radius_grow=1.5,
            max_inner_attempts=1,
            penalty_init=1.0,
            penalty_scale=10.0,
            feasibility_tol=1e-8,
            stationarity_tol=1e-3,
        )
        minimize_calls = {"count": 0}

        def evaluate_problem(x, multipliers, penalty):
            point = float(np.asarray(x, dtype=float)[0])
            if point < 0.5:
                return {
                    "total": 20.0,
                    "base_value": 20.0,
                    "grad": np.array([0.5]),
                    "constraint_values": np.array([0.5]),
                    "stationarity_norm": 0.5,
                }
            if point < 1.5:
                return {
                    "total": 10.0,
                    "base_value": 10.0,
                    "grad": np.zeros(1),
                    "constraint_values": np.array([0.5]),
                    "stationarity_norm": 0.0,
                }
            if point < 2.5:
                return {
                    "total": 1.0,
                    "base_value": 1.0,
                    "grad": np.array([0.05]),
                    "constraint_values": np.array([0.0]),
                    "stationarity_norm": 0.05,
                }
            return {
                "total": 0.5,
                "base_value": 5.0,
                "grad": np.array([0.2]),
                "constraint_values": np.array([0.0]),
                "stationarity_norm": 0.2,
            }

        def fake_minimize(fun, x, jac, method, bounds, callback, options):
            minimize_calls["count"] += 1
            return SimpleNamespace(
                x=np.array([float(minimize_calls["count"])]),
                nit=1,
                success=True,
                message="CONVERGENCE",
            )

        with patch.object(module, "minimize", side_effect=fake_minimize):
            result = module.minimize_alm(
                np.array([0.0]),
                ["demo_constraint"],
                evaluate_problem,
                settings,
                {"maxiter": 5, "ftol": 1e-12, "gtol": 1e-12},
            )

        self.assertFalse(result.success)
        np.testing.assert_allclose(result.x, np.array([2.0]))
        np.testing.assert_allclose(result.inner_result.x, np.array([2.0]))
        self.assertTrue(result.restored_best_feasible)
        self.assertEqual(
            result.restored_best_feasible_reason,
            "final_iterate_worse_than_best_feasible",
        )
        self.assertEqual(result.termination_reason, "max_outer_restored_best_feasible")
        self.assertEqual(minimize_calls["count"], 3)
        self.assertEqual(result.history[0]["action"], "penalty_increase")
        self.assertEqual(result.history[1]["action"], "dual_update")
        self.assertEqual(result.history[2]["action"], "subproblem_limit")
        self.assertEqual(
            result.history[2]["subproblem_limit_reason"],
            "max_subproblem_continuations",
        )
        self.assertEqual(result.history[2]["outer_termination"], "max_outer")

    def test_minimize_alm_restores_best_feasible_solver_owned_incumbent_state(self):
        module = load_alm_utils_module()
        settings = module.ALMSettings(
            max_outer_iterations=3,
            max_subproblem_continuations=0,
            trust_radius_init=0.1,
            trust_radius_min=0.01,
            trust_radius_shrink=0.5,
            trust_radius_grow=1.5,
            max_inner_attempts=1,
            penalty_init=1.0,
            penalty_scale=10.0,
            feasibility_tol=1e-8,
            stationarity_tol=1e-3,
        )
        minimize_calls = {"count": 0}
        restored = {"state": None}
        snapshot_calls = {"count": 0}

        def evaluate_problem(x, multipliers, penalty):
            point = float(np.asarray(x, dtype=float)[0])
            if point < 0.5:
                return {
                    "total": 20.0,
                    "base_value": 20.0,
                    "grad": np.array([0.5]),
                    "constraint_values": np.array([0.5]),
                    "stationarity_norm": 0.5,
                }
            if point < 1.5:
                return {
                    "total": 10.0,
                    "base_value": 10.0,
                    "grad": np.zeros(1),
                    "constraint_values": np.array([0.5]),
                    "stationarity_norm": 0.0,
                }
            if point < 2.5:
                return {
                    "total": 1.0,
                    "base_value": 1.0,
                    "grad": np.array([0.05]),
                    "constraint_values": np.array([0.0]),
                    "stationarity_norm": 0.05,
                }
            return {
                "total": 0.5,
                "base_value": 5.0,
                "grad": np.array([0.2]),
                "constraint_values": np.array([0.0]),
                "stationarity_norm": 0.2,
            }

        def fake_minimize(fun, x, jac, method, bounds, callback, options):
            minimize_calls["count"] += 1
            return SimpleNamespace(
                x=np.array([float(minimize_calls["count"])]),
                nit=1,
                success=True,
                message="CONVERGENCE",
            )

        def snapshot_state():
            snapshot_calls["count"] += 1
            return {"accepted_point": float(minimize_calls["count"])}

        def restore_state(state):
            restored["state"] = state

        with patch.object(module, "minimize", side_effect=fake_minimize):
            result = module.minimize_alm(
                np.array([0.0]),
                ["demo_constraint"],
                evaluate_problem,
                settings,
                {"maxiter": 5, "ftol": 1e-12, "gtol": 1e-12},
                snapshot_accepted_state_fn=snapshot_state,
                restore_incumbent_state_fn=restore_state,
            )

        np.testing.assert_allclose(result.x, np.array([2.0]))
        self.assertTrue(result.restored_best_feasible)
        self.assertEqual(restored["state"], {"accepted_point": 2.0})
        self.assertEqual(snapshot_calls["count"], 1)

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

    def test_minimize_alm_reports_inner_and_accepted_callbacks_separately(self):
        module = load_alm_utils_module()
        settings = module.ALMSettings(
            max_outer_iterations=1,
            max_subproblem_continuations=1,
            feasibility_tol=1e-8,
            stationarity_tol=1e-8,
        )
        inner_points = []
        accepted_points = []

        def evaluate_problem(x, multipliers, penalty):
            x = np.asarray(x, dtype=float)
            return {
                "total": 0.5 * float(np.dot(x, x)),
                "grad": x.copy(),
                "constraint_values": np.zeros(1),
                "stationarity_norm": float(np.linalg.norm(x)),
            }

        def fake_minimize(fun, x, jac, method, bounds, callback, options):
            callback(np.array([2.0]))
            return SimpleNamespace(
                x=np.array([0.0]),
                nit=1,
                success=True,
                message="CONVERGENCE",
            )

        with patch.object(module, "minimize", side_effect=fake_minimize):
            result = module.minimize_alm(
                np.array([1.0]),
                ["demo_constraint"],
                evaluate_problem,
                settings,
                {"maxiter": 5, "ftol": 1e-12, "gtol": 1e-12},
                inner_callback=lambda x: inner_points.append(float(np.asarray(x)[0])),
                accepted_callback=lambda x: accepted_points.append(float(np.asarray(x)[0])),
            )

        self.assertTrue(result.success)
        self.assertEqual(inner_points, [2.0])
        self.assertEqual(accepted_points, [0.0])

    def test_minimize_alm_uses_metric_gradient_for_convergence_diagnostics(self):
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
                "grad": np.array([10.0]),
                "metric_grad": np.array([0.0]),
                "constraint_values": np.array([-1.0e-4]),
                "dual_update_values": np.array([-1.0e-4]),
                "feasibility_values": np.array([0.0]),
                "constraint_grads": [np.array([-1.0])],
                "constraint_activity_tolerances": np.array([1.0e-3]),
                "stationarity_norm": 10.0,
                "metric_stationarity_norm": 0.0,
            }

        with patch.object(module, "minimize", side_effect=AssertionError("minimize should not run")):
            result = module.minimize_alm(
                np.array([0.0]),
                ["demo_constraint"],
                evaluate_problem,
                settings,
                {"maxiter": 5, "ftol": 1e-12, "gtol": 1e-12},
            )

        self.assertTrue(result.success)
        self.assertAlmostEqual(result.history[0]["raw_stationarity_norm"], 0.0)
        self.assertAlmostEqual(result.history[0]["stationarity_norm"], 0.0)

    def test_next_penalty_caps_requested_growth(self):
        module = load_alm_utils_module()

        next_penalty, cap_hit, requested_penalty = module._next_penalty(
            1.0e8,
            penalty_scale=10.0,
            penalty_max=1.0e8,
        )

        self.assertEqual(next_penalty, 1.0e8)
        self.assertTrue(cap_hit)
        self.assertEqual(requested_penalty, 1.0e9)

    def test_next_penalty_caps_on_overflow_when_no_max(self):
        module = load_alm_utils_module()

        next_penalty, cap_hit, requested_penalty = module._next_penalty(
            1.0e308,
            penalty_scale=100.0,
            penalty_max=None,
        )

        self.assertEqual(next_penalty, 1.0e308)
        self.assertTrue(cap_hit)
        self.assertTrue(np.isinf(requested_penalty))

    def test_minimize_alm_stops_when_penalty_cap_blocks_further_growth(self):
        module = load_alm_utils_module()
        settings = module.ALMSettings(
            max_outer_iterations=2,
            penalty_init=1.0e8,
            penalty_scale=10.0,
            penalty_max=1.0e8,
            feasibility_tol=1.0e-8,
            stationarity_tol=1.0e-8,
            trust_radius_init=0.1,
            trust_radius_min=0.01,
            trust_radius_shrink=0.5,
            trust_radius_grow=1.5,
        )
        candidate_x = np.array([0.2])

        def evaluate_problem(x, multipliers, penalty):
            del x, multipliers, penalty
            return {
                "total": 1.0,
                "base_value": 0.1,
                "base_grad": np.array([0.0]),
                "grad": np.array([0.0]),
                "constraint_values": np.array([0.5]),
                "feasibility_values": np.array([0.5]),
                "dual_update_values": np.array([0.5]),
                "stationarity_norm": 0.0,
            }

        def fake_minimize(fun, x, jac, method, bounds, callback, options):
            del jac, method, bounds, callback, options
            fun(candidate_x)
            return SimpleNamespace(
                x=candidate_x.copy(),
                nit=1,
                success=False,
                message="ABNORMAL",
            )

        with patch.object(module, "minimize", side_effect=fake_minimize):
            result = module.minimize_alm(
                np.array([0.0]),
                ["demo_constraint"],
                evaluate_problem,
                settings,
                {"maxiter": 5, "ftol": 1.0e-12, "gtol": 1.0e-12},
            )

        self.assertFalse(result.success)
        self.assertEqual(result.termination_reason, "penalty_cap_reached")
        self.assertTrue(result.penalty_cap_reached)
        self.assertEqual(result.penalty_max, 1.0e8)
        self.assertEqual(result.penalty_cap_requested, 1.0e9)
        self.assertEqual(result.history[0]["action"], "penalty_cap_reached")
        self.assertIn("configured penalty cap", result.message)


if __name__ == "__main__":
    unittest.main()
