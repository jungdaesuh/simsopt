import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


EXAMPLES_ROOT = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "single_stage_optimization"
)
EXAMPLES_ROOT_STR = str(EXAMPLES_ROOT)
EXAMPLES_ROOT_INSERTED = EXAMPLES_ROOT_STR not in sys.path
if EXAMPLES_ROOT_INSERTED:
    sys.path.insert(0, EXAMPLES_ROOT_STR)
import alm_utils as _alm_utils  # noqa: E402
if EXAMPLES_ROOT_INSERTED:
    sys.path.remove(EXAMPLES_ROOT_STR)


def load_alm_utils_module():
    return _alm_utils


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
        np.testing.assert_allclose(evaluation["positive_shift_values"], np.array([0.0, 2.0]))
        np.testing.assert_allclose(
            evaluation["augmented_term_by_constraint"],
            np.array([-0.0625, 0.75]),
        )
        np.testing.assert_allclose(
            evaluation["constraint_grads"],
            [np.array([2.0, 0.0]), np.array([0.0, 4.0])],
        )
        self.assertAlmostEqual(evaluation["max_feasibility_violation"], 0.5)

    def test_augmented_inequality_objective_accepts_vector_penalty(self):
        module = load_alm_utils_module()

        evaluation = module.augmented_inequality_objective(
            base_value=3.0,
            base_grad=np.array([1.0, -1.0]),
            constraint_values=np.array([-1.0, 0.5]),
            constraint_grads=[np.array([2.0, 0.0]), np.array([0.0, 4.0])],
            multipliers=np.array([0.5, 1.0]),
            penalty=np.array([2.0, 4.0]),
        )

        self.assertAlmostEqual(evaluation["total"], 3.9375)
        np.testing.assert_allclose(evaluation["grad"], np.array([1.0, 11.0]))
        np.testing.assert_allclose(evaluation["dual_update_values"], np.array([-1.0, 0.5]))
        np.testing.assert_allclose(evaluation["feasibility_values"], np.array([0.0, 0.5]))
        np.testing.assert_allclose(evaluation["positive_shift_values"], np.array([0.0, 3.0]))
        np.testing.assert_allclose(
            evaluation["augmented_term_by_constraint"],
            np.array([-0.0625, 1.0]),
        )

    def test_augmented_inequality_objective_one_block_vector_matches_scalar(self):
        module = load_alm_utils_module()

        scalar = module.augmented_inequality_objective(
            base_value=3.0,
            base_grad=np.array([1.0, -1.0]),
            constraint_values=np.array([-1.0, 0.5]),
            constraint_grads=[np.array([2.0, 0.0]), np.array([0.0, 4.0])],
            multipliers=np.array([0.5, 1.0]),
            penalty=4.0,
        )
        vector = module.augmented_inequality_objective(
            base_value=3.0,
            base_grad=np.array([1.0, -1.0]),
            constraint_values=np.array([-1.0, 0.5]),
            constraint_grads=[np.array([2.0, 0.0]), np.array([0.0, 4.0])],
            multipliers=np.array([0.5, 1.0]),
            penalty=np.array([4.0, 4.0]),
        )

        self.assertAlmostEqual(vector["total"], scalar["total"])
        np.testing.assert_allclose(vector["grad"], scalar["grad"])

    def test_project_nonnegative_multipliers_uses_vector_penalty(self):
        module = load_alm_utils_module()

        updated, cap_binding, cap_indices = (
            module._project_nonnegative_multipliers_with_diagnostics(
                multipliers=np.array([1.0, 2.0]),
                dual_update_values=np.array([0.5, -1.0]),
                penalty=np.array([4.0, 10.0]),
                multiplier_max=2.5,
            )
        )

        np.testing.assert_allclose(updated, np.array([2.5, 0.0]))
        self.assertTrue(cap_binding)
        self.assertEqual(cap_indices, [0])

    def test_block_penalty_growth_targets_only_stalled_violated_blocks(self):
        module = load_alm_utils_module()
        settings = module.ALMSettings(
            penalty_init=1.0,
            penalty_scale=10.0,
            feasibility_tol=0.1,
            block_penalties_enabled=True,
        )
        state = module._initial_block_penalty_state(
            settings,
            ["geometry", "current"],
            initial_penalty=1.0,
        )

        next_state, grown_blocks, cap_hit_blocks, requested = module._next_block_penalty_state(
            state,
            {"geometry": 0.5, "current": 0.05},
            settings,
        )

        self.assertEqual(grown_blocks, ["geometry"])
        self.assertEqual(cap_hit_blocks, [])
        self.assertEqual(requested, {"geometry": 10.0})
        self.assertEqual(next_state.penalties_by_block["geometry"], 10.0)
        self.assertEqual(next_state.penalties_by_block["current"], 1.0)

    def test_block_penalty_growth_reports_caps_per_block(self):
        module = load_alm_utils_module()
        settings = module.ALMSettings(
            penalty_init=10.0,
            penalty_scale=10.0,
            penalty_max=1.0e8,
            feasibility_tol=0.1,
            block_penalties_enabled=True,
            block_penalty_max={"geometry": 50.0, "current": 1.0e8},
        )
        state = module._initial_block_penalty_state(
            settings,
            ["geometry", "current"],
            initial_penalty=10.0,
        )

        next_state, grown_blocks, cap_hit_blocks, requested = module._next_block_penalty_state(
            state,
            {"geometry": 0.5, "current": 0.0},
            settings,
        )

        self.assertEqual(grown_blocks, [])
        self.assertEqual(cap_hit_blocks, ["geometry"])
        self.assertEqual(requested, {"geometry": 100.0})
        self.assertEqual(next_state.penalties_by_block["geometry"], 10.0)
        self.assertTrue(next_state.cap_reached_by_block["geometry"])

    def test_block_penalty_state_reuses_state_when_values_do_not_change(self):
        module = load_alm_utils_module()
        settings = module.ALMSettings(
            penalty_init=1.0,
            penalty_scale=10.0,
            feasibility_tol=0.1,
            block_penalties_enabled=True,
        )
        state = module._initial_block_penalty_state(
            settings,
            ["geometry", "current"],
            initial_penalty=1.0,
        )
        state, _grown_blocks, _cap_hit_blocks, _requested = (
            module._next_block_penalty_state(
                state,
                {"geometry": 0.0, "current": 0.0},
                settings,
            )
        )

        next_state, grown_blocks, cap_hit_blocks, requested = (
            module._next_block_penalty_state(
                state,
                {"geometry": 0.0, "current": 0.0},
                settings,
            )
        )

        self.assertIs(next_state, state)
        self.assertEqual(grown_blocks, [])
        self.assertEqual(cap_hit_blocks, [])
        self.assertEqual(requested, {})
        self.assertFalse(next_state.cap_reached_by_block["current"])

    def test_block_penalty_state_exposes_read_only_maps(self):
        module = load_alm_utils_module()
        settings = module.ALMSettings(
            penalty_init=1.0,
            penalty_scale=10.0,
            feasibility_tol=0.1,
            block_penalties_enabled=True,
        )
        state = module._initial_block_penalty_state(
            settings,
            ["geometry"],
            initial_penalty=1.0,
        )

        with self.assertRaises(TypeError):
            state.penalties_by_block["geometry"] = 2.0

        next_state, _grown_blocks, _cap_hit_blocks, _requested = (
            module._next_block_penalty_state(
                state,
                {"geometry": 0.5},
                settings,
            )
        )

        self.assertEqual(next_state.penalties_by_block["geometry"], 10.0)
        self.assertEqual(state.penalties_by_block["geometry"], 1.0)
        with self.assertRaises(TypeError):
            next_state.stall_counts_by_block["geometry"] = 4

    def test_block_penalty_init_rejects_cap_below_initial_penalty(self):
        module = load_alm_utils_module()
        settings = module.ALMSettings(
            penalty_init=10.0,
            block_penalties_enabled=True,
            block_penalty_max={"geometry": 5.0},
        )

        with self.assertRaisesRegex(ValueError, "block_penalty_max\\['geometry'\\]"):
            module._initial_block_penalty_state(
                settings,
                ["geometry"],
                initial_penalty=10.0,
            )

    def test_block_penalty_init_rejects_invalid_hysteresis_settings(self):
        module = load_alm_utils_module()

        with self.assertRaisesRegex(
            ValueError,
            "block_penalty_improvement_fraction",
        ):
            module._initial_block_penalty_state(
                module.ALMSettings(
                    block_penalties_enabled=True,
                    block_penalty_improvement_fraction=1.0,
                ),
                ["geometry"],
                initial_penalty=1.0,
            )

        with self.assertRaisesRegex(ValueError, "block_penalty_patience"):
            module._initial_block_penalty_state(
                module.ALMSettings(
                    block_penalties_enabled=True,
                    block_penalty_patience=0,
                ),
                ["geometry"],
                initial_penalty=1.0,
            )

    def test_minimize_alm_requires_blocks_when_block_penalties_are_enabled(self):
        module = load_alm_utils_module()

        def evaluate_problem(x, multipliers, penalty):
            del x, multipliers, penalty
            return module.augmented_inequality_objective(
                base_value=0.0,
                base_grad=np.zeros(1),
                constraint_values=np.array([1.0]),
                constraint_grads=[np.zeros(1)],
                multipliers=np.zeros(1),
                penalty=1.0,
            )

        with self.assertRaisesRegex(ValueError, "constraint_blocks"):
            module.minimize_alm(
                np.zeros(1),
                ["gap"],
                evaluate_problem,
                module.ALMSettings(block_penalties_enabled=True),
                {"maxiter": 1},
            )

    def test_normalize_alm_constraints_scales_values_grads_and_tolerances(self):
        module = load_alm_utils_module()

        payload = module.normalize_alm_constraints(
            signed_values=np.array([1000.0, 1.0]),
            constraint_grads=[np.array([16.0, -8.0]), np.array([2.0, 4.0])],
            feasibility_values=np.array([500.0, 0.5]),
            activity_tolerances=np.array([16.0, 0.04]),
            scales=np.array([16000.0, 40.0]),
        )

        np.testing.assert_allclose(
            payload["normalized_signed_values"],
            [0.0625, 0.025],
        )
        np.testing.assert_allclose(
            payload["normalized_constraint_grads"],
            [np.array([0.001, -0.0005]), np.array([0.05, 0.1])],
        )
        np.testing.assert_allclose(
            payload["normalized_feasibility_values"],
            [0.03125, 0.0125],
        )
        np.testing.assert_allclose(
            payload["normalized_activity_tolerances"],
            [0.001, 0.001],
        )

    def test_normalize_alm_constraints_accepts_empty_constraint_set(self):
        module = load_alm_utils_module()

        payload = module.normalize_alm_constraints([], [], [], [], [])

        self.assertEqual(payload["normalized_signed_values"].shape, (0,))
        self.assertEqual(payload["normalized_constraint_grads"], [])
        self.assertEqual(payload["normalized_feasibility_values"].shape, (0,))
        self.assertEqual(payload["normalized_activity_tolerances"].shape, (0,))

    def test_normalize_alm_constraints_rejects_nonfinite_or_nonpositive_scales(self):
        module = load_alm_utils_module()

        for scales in ([1.0, np.nan], [1.0, 0.0], [1.0, -2.0]):
            with self.subTest(scales=scales):
                with self.assertRaisesRegex(ValueError, "finite and positive"):
                    module.normalize_alm_constraints(
                        signed_values=[1.0, 2.0],
                        constraint_grads=[np.array([1.0]), np.array([2.0])],
                        feasibility_values=[1.0, 2.0],
                        activity_tolerances=[0.1, 0.2],
                        scales=scales,
                    )

    def test_normalize_alm_constraints_rejects_shape_mismatch(self):
        module = load_alm_utils_module()

        with self.assertRaisesRegex(ValueError, "signed_values shape"):
            module.normalize_alm_constraints(
                signed_values=[1.0],
                constraint_grads=[np.array([1.0]), np.array([2.0])],
                feasibility_values=[1.0, 2.0],
                activity_tolerances=[0.1, 0.2],
                scales=[1.0, 2.0],
            )

        with self.assertRaisesRegex(ValueError, "feasibility_values shape"):
            module.normalize_alm_constraints(
                signed_values=[1.0, 2.0],
                constraint_grads=[np.array([1.0]), np.array([2.0])],
                feasibility_values=[1.0],
                activity_tolerances=[0.1, 0.2],
                scales=[1.0, 2.0],
            )

        with self.assertRaisesRegex(ValueError, "activity_tolerances shape"):
            module.normalize_alm_constraints(
                signed_values=[1.0, 2.0],
                constraint_grads=[np.array([1.0]), np.array([2.0])],
                feasibility_values=[1.0, 2.0],
                activity_tolerances=[0.1],
                scales=[1.0, 2.0],
            )

        with self.assertRaisesRegex(ValueError, "constraint_grads length"):
            module.normalize_alm_constraints(
                signed_values=[1.0, 2.0],
                constraint_grads=[np.array([1.0])],
                feasibility_values=[1.0, 2.0],
                activity_tolerances=[0.1, 0.2],
                scales=[1.0, 2.0],
            )

    def test_constraint_history_diagnostics_groups_values_by_block(self):
        module = load_alm_utils_module()
        evaluation = {
            "total": 0.0,
            "grad": np.zeros(1),
            "constraint_values": np.array([0.2, -0.1, 0.5]),
            "feasibility_values": np.array([0.2, 0.0, 0.5]),
            "dual_update_values": np.array([0.2, -0.1, 0.5]),
            "constraint_grads": [np.zeros(1), np.zeros(1), np.zeros(1)],
            "raw_dual_update_values": np.array([2.0, -1.0, 1000.0]),
            "raw_hard_violation_values": np.array([2.0, 0.0, 1000.0]),
            "normalized_signed_constraint_values": np.array([0.2, -0.1, 0.5]),
            "normalized_feasibility_values": np.array([0.2, 0.0, 0.5]),
            "constraint_scales": [10.0, 10.0, 2000.0],
            "constraint_blocks": ["geometry", "geometry", "current"],
        }
        multipliers = np.array([0.1, 0.2, 0.3])
        penalty = 4.0
        routing_state = module._constraint_routing_state(
            evaluation,
            multipliers,
            penalty,
            feasibility_gate=1.0,
        )

        diagnostics = module._constraint_history_diagnostics(
            evaluation,
            multipliers,
            penalty,
            ["gap", "length", "current"],
            np.array([0.2, -0.1, 0.5]),
            np.array([0.2, 0.0, 0.5]),
            routing_state,
            1.0,
        )

        self.assertEqual(
            diagnostics["block_max_normalized_violation"],
            {"geometry": 0.2, "current": 0.5},
        )
        self.assertEqual(
            diagnostics["block_max_raw_hard_violation"],
            {"geometry": 2.0, "current": 1000.0},
        )
        self.assertEqual(diagnostics["blocking_constraint_name"], "current")
        self.assertEqual(diagnostics["blocking_constraint_block"], "current")
        np.testing.assert_allclose(
            diagnostics["raw_dual_estimates"],
            [0.01, 0.02, 0.00015],
        )
        np.testing.assert_allclose(
            diagnostics["active_pressure_by_constraint"],
            [0.18, 0.0, 1.15],
        )
        np.testing.assert_allclose(
            diagnostics["surrogate_minus_hard_normalized_gap"],
            [0.0, 0.0, 0.0],
        )
        self.assertEqual(
            diagnostics["surrogate_hard_sign_mismatch_by_constraint"],
            [False, False, False],
        )
        self.assertIsNone(diagnostics["objective_to_augmented_term_ratio"])
        self.assertAlmostEqual(diagnostics["augmented_gradient_norm"], 0.0)
        self.assertAlmostEqual(diagnostics["surrogate_kkt_stationarity_norm"], 0.0)
        self.assertEqual(
            diagnostics["multiplier_interpretation"],
            "differentiable_alm_multipliers",
        )

    def test_objective_to_augmented_term_ratio_requires_explicit_base_objective(self):
        module = load_alm_utils_module()

        self.assertIsNone(
            module._objective_to_augmented_term_ratio(
                {"total": 3.0},
                np.array([3.0]),
            )
        )
        self.assertAlmostEqual(
            module._objective_to_augmented_term_ratio(
                {"total": 4.0, "base_total": 2.0},
                np.array([1.0]),
            ),
            2.0,
        )

    def test_alm_summary_uses_summary_diagnostics_without_full_history_payload(self):
        module = load_alm_utils_module()
        evaluation = {
            "total": 0.0,
            "grad": np.zeros(1),
            "constraint_values": np.array([0.2, -0.1, 0.5]),
            "feasibility_values": np.array([0.2, 0.0, 0.5]),
            "dual_update_values": np.array([0.2, -0.1, 0.5]),
            "constraint_grads": [np.zeros(1), np.zeros(1), np.zeros(1)],
            "raw_hard_violation_values": np.array([2.0, 0.0, 1000.0]),
            "constraint_blocks": ["geometry", "geometry", "current"],
        }
        multipliers = np.array([0.1, 0.2, 0.3])
        penalty = 4.0
        solver_constraint_values = np.array([0.2, -0.1, 0.5])
        feasibility_values = np.array([0.2, 0.0, 0.5])
        routing_state = module._constraint_routing_state(
            evaluation,
            multipliers,
            penalty,
            feasibility_gate=1.0,
        )

        with patch.object(
            module,
            "_constraint_history_diagnostics",
            side_effect=AssertionError("full diagnostics should not run"),
        ):
            summary = module._alm_summary(
                termination_reason="max_outer_iterations",
                evaluation=evaluation,
                multipliers=multipliers,
                penalty=penalty,
                constraint_names=["gap", "length", "current"],
                routing_state=routing_state,
                feasibility_values=feasibility_values,
                solver_constraint_values=solver_constraint_values,
                final_stationarity_norm=0.0,
                final_feasibility_tolerance=1.0,
                multiplier_cap_binding=False,
                penalty_cap_reached=False,
                history=[],
                history_truncated_count=0,
            )

        self.assertEqual(summary["blocking_constraint_name"], "current")
        self.assertEqual(
            summary["block_max_normalized_violation"],
            {"geometry": 0.2, "current": 0.5},
        )
        self.assertEqual(
            summary["max_raw_hard_violation_by_constraint"],
            {"gap": 2.0, "length": 0.0, "current": 1000.0},
        )
        self.assertEqual(
            summary["multiplier_interpretation"],
            "differentiable_alm_multipliers",
        )

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

    def test_multiplier_interpretation_marks_mixed_value_sources_as_search_multipliers(self):
        module = load_alm_utils_module()

        self.assertEqual(
            module._multiplier_interpretation(
                {
                    "gradient_value_kinds": ["surrogate", "surrogate"],
                    "dual_update_value_kinds": ["surrogate", "hard"],
                }
            ),
            "search_multipliers",
        )
        self.assertEqual(
            module._multiplier_interpretation(
                {
                    "gradient_value_kinds": ["surrogate", "hard"],
                    "dual_update_value_kinds": ["surrogate", "hard"],
                }
            ),
            "differentiable_alm_multipliers",
        )

    def test_surrogate_kkt_stationarity_uses_surrogate_feasibility_gate(self):
        module = load_alm_utils_module()
        evaluation = {
            "total": 0.0,
            "grad": np.array([1.0]),
            "constraint_values": np.array([0.0]),
            "feasibility_values": np.array([2.0]),
            "hard_signed_constraint_values": np.array([2.0]),
            "hard_violation_values": np.array([2.0]),
            "surrogate_signed_constraint_values": np.array([0.0]),
            "constraint_grads": [np.array([-1.0])],
            "constraint_activity_tolerances": np.array([0.0]),
        }
        routing_state = module._constraint_routing_state(
            evaluation,
            np.zeros(1),
            1.0,
            feasibility_gate=1.0,
        )

        self.assertAlmostEqual(
            module._surrogate_kkt_stationarity_norm(
                evaluation,
                routing_state,
                feasibility_gate=1.0,
            ),
            0.0,
        )

    def test_lbfgsb_projected_gradient_max_norm_uses_projected_infinity_norm(self):
        module = load_alm_utils_module()

        self.assertAlmostEqual(
            module._lbfgsb_projected_gradient_max_norm(
                np.array([5.0, -7.0, 3.0, -4.0]),
                np.array([0.0, 1.0, 0.5, 2.0]),
                [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (None, None)],
            ),
            4.0,
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

    def test_scaled_inequality_fixture_preserves_raw_feasibility_and_dual_conversion(self):
        module = load_alm_utils_module()

        raw_violation = 1000.0
        scale = 16000.0
        normalized_violation = raw_violation / scale
        lambda_raw = 0.25
        lambda_norm = lambda_raw * scale

        evaluation = module.augmented_inequality_objective(
            base_value=0.0,
            base_grad=np.array([0.0]),
            constraint_values=np.array([normalized_violation]),
            constraint_grads=[np.array([1.0 / scale])],
            multipliers=np.array([lambda_norm]),
            penalty=2.0,
        )

        self.assertAlmostEqual(raw_violation, 1000.0)
        self.assertAlmostEqual(evaluation["max_feasibility_violation"], 0.0625)
        self.assertAlmostEqual(lambda_raw, lambda_norm / scale)

    def test_scale_one_fixture_matches_existing_scalar_alm_behavior(self):
        module = load_alm_utils_module()

        raw_evaluation = module.augmented_inequality_objective(
            base_value=3.0,
            base_grad=np.array([1.0]),
            constraint_values=np.array([0.5]),
            constraint_grads=[np.array([2.0])],
            multipliers=np.array([0.25]),
            penalty=4.0,
        )
        scale_one_evaluation = module.augmented_inequality_objective(
            base_value=3.0,
            base_grad=np.array([1.0]),
            constraint_values=np.array([0.5 / 1.0]),
            constraint_grads=[np.array([2.0 / 1.0])],
            multipliers=np.array([0.25 * 1.0]),
            penalty=4.0,
        )

        self.assertEqual(raw_evaluation["total"], scale_one_evaluation["total"])
        np.testing.assert_allclose(raw_evaluation["grad"], scale_one_evaluation["grad"])
        np.testing.assert_allclose(
            raw_evaluation["dual_update_values"],
            scale_one_evaluation["dual_update_values"],
        )


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
    def _quartic_taylor_evaluation(x):
        x = np.asarray(x, dtype=float)
        grad = 4.0 * x**3
        return {
            "total": float(np.sum(x**4)),
            "grad": grad,
            "constraint_values": np.zeros(1),
            "stationarity_norm": float(np.linalg.norm(grad)),
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

    def test_conditioning_metrics_penalty_ratio_preserves_small_base_signal(self):
        module = load_alm_utils_module()

        metrics = module._conditioning_metrics(
            {
                "total": 0.101,
                "base_total": 1.0e-3,
                "grad": np.array([0.0]),
            }
        )

        self.assertAlmostEqual(metrics["conditioning_penalty_objective"], 0.1)
        self.assertAlmostEqual(metrics["conditioning_penalty_objective_ratio"], 100.0)

    def test_conditioning_metrics_penalty_ratio_is_none_for_zero_base_objective(self):
        module = load_alm_utils_module()

        metrics = module._conditioning_metrics(
            {
                "total": 0.1,
                "base_total": 0.0,
                "grad": np.array([0.0]),
            }
        )

        self.assertIsNone(metrics["conditioning_penalty_objective_ratio"])

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

    def test_directional_taylor_test_passes_for_consistent_gradient(self):
        module = load_alm_utils_module()

        def evaluate_problem(x, multipliers, penalty):
            del multipliers, penalty
            return self._quartic_taylor_evaluation(x)

        result = module.run_directional_taylor_test(
            evaluate_problem,
            np.array([0.2, -0.4]),
            np.zeros(1),
            1.0,
            seed=7,
        )

        self.assertTrue(result["passed"])
        self.assertEqual(result["direction_count"], 4)
        self.assertIsNotNone(result["max_ratio"])
        self.assertLess(result["max_ratio"], result["ratio_threshold"])

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

    def test_scale_one_minimize_alm_matches_scalar_history_and_convergence(self):
        module = load_alm_utils_module()
        settings = module.ALMSettings(
            max_outer_iterations=8,
            penalty_init=1.0,
            penalty_scale=10.0,
            feasibility_tol=1e-8,
            stationarity_tol=1e-8,
            relaxed_feasibility_gate_cap=1.0,
            multiplier_max=1.0e6,
            history_max_entries=None,
        )
        inner_options = {"maxiter": 80, "ftol": 1e-12, "gtol": 1e-10, "maxls": 20}

        def raw_evaluate_problem(x, multipliers, penalty):
            x = np.asarray(x, dtype=float)
            signed_constraint_value = np.array([x[0] - 0.25])
            return module.augmented_inequality_objective(
                0.5 * (x[0] - 1.0) ** 2,
                np.array([x[0] - 1.0]),
                signed_constraint_value,
                [np.array([1.0])],
                multipliers,
                penalty,
            )

        def scale_one_evaluate_problem(x, multipliers, penalty):
            x = np.asarray(x, dtype=float)
            scale = 1.0
            signed_constraint_value = np.array([(x[0] - 0.25) / scale])
            return module.augmented_inequality_objective(
                0.5 * (x[0] - 1.0) ** 2,
                np.array([x[0] - 1.0]),
                signed_constraint_value,
                [np.array([1.0 / scale])],
                multipliers,
                penalty,
            )

        raw_result = module.minimize_alm(
            np.array([0.0]),
            ["upper"],
            raw_evaluate_problem,
            settings,
            inner_options,
        )
        scale_one_result = module.minimize_alm(
            np.array([0.0]),
            ["upper"],
            scale_one_evaluate_problem,
            settings,
            inner_options,
        )

        self.assertTrue(raw_result.success)
        self.assertEqual(raw_result.termination_reason, "converged")
        self.assertEqual(raw_result.termination_reason, scale_one_result.termination_reason)
        np.testing.assert_allclose(raw_result.x, scale_one_result.x)
        np.testing.assert_allclose(raw_result.multipliers, scale_one_result.multipliers)
        self.assertEqual(raw_result.penalty, scale_one_result.penalty)
        self.assertEqual(raw_result.history, scale_one_result.history)

    def test_block_penalties_disabled_ignores_block_inputs_and_matches_scalar_alm(self):
        module = load_alm_utils_module()
        scalar_settings = module.ALMSettings(
            max_outer_iterations=8,
            penalty_init=1.0,
            penalty_scale=10.0,
            feasibility_tol=1e-8,
            stationarity_tol=1e-8,
            relaxed_feasibility_gate_cap=1.0,
            multiplier_max=1.0e6,
            history_max_entries=None,
        )
        disabled_block_settings = module.ALMSettings(
            max_outer_iterations=8,
            penalty_init=1.0,
            penalty_scale=10.0,
            feasibility_tol=1e-8,
            stationarity_tol=1e-8,
            relaxed_feasibility_gate_cap=1.0,
            multiplier_max=1.0e6,
            block_penalties_enabled=False,
            block_penalty_init={"current": 2.0},
            block_penalty_scale={"current": 3.0},
            history_max_entries=None,
        )
        inner_options = {"maxiter": 80, "ftol": 1e-12, "gtol": 1e-10, "maxls": 20}

        def evaluate_problem(x, multipliers, penalty):
            x = np.asarray(x, dtype=float)
            return module.augmented_inequality_objective(
                0.5 * (x[0] - 1.0) ** 2,
                np.array([x[0] - 1.0]),
                np.array([x[0] - 0.25]),
                [np.array([1.0])],
                multipliers,
                penalty,
            )

        scalar_result = module.minimize_alm(
            np.array([0.0]),
            ["upper"],
            evaluate_problem,
            scalar_settings,
            inner_options,
        )
        disabled_block_result = module.minimize_alm(
            np.array([0.0]),
            ["upper"],
            evaluate_problem,
            disabled_block_settings,
            inner_options,
            constraint_blocks=["current"],
        )

        self.assertTrue(scalar_result.success)
        self.assertEqual(scalar_result.termination_reason, "converged")
        self.assertEqual(
            scalar_result.termination_reason,
            disabled_block_result.termination_reason,
        )
        np.testing.assert_allclose(scalar_result.x, disabled_block_result.x)
        np.testing.assert_allclose(
            scalar_result.multipliers,
            disabled_block_result.multipliers,
        )
        self.assertEqual(scalar_result.penalty, disabled_block_result.penalty)
        self.assertEqual(scalar_result.history, disabled_block_result.history)

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

    def test_minimize_alm_reports_history_truncation_count(self):
        module = load_alm_utils_module()
        settings = module.ALMSettings(
            max_outer_iterations=3,
            penalty_init=1.0,
            penalty_scale=10.0,
            feasibility_tol=1e-12,
            stationarity_tol=1e-12,
            history_max_entries=1,
        )

        def evaluate_problem(x, multipliers, penalty):
            del x, multipliers, penalty
            return {
                "total": 0.0,
                "grad": np.zeros(1),
                "constraint_values": np.array([2.0]),
                "stationarity_norm": 0.0,
            }

        with patch.object(
            module,
            "_constraint_history_diagnostics_from_source",
            wraps=module._constraint_history_diagnostics_from_source,
        ) as materialize_diagnostics:
            result = module.minimize_alm(
                np.array([0.0]),
                ["demo_constraint"],
                evaluate_problem,
                settings,
                {"maxiter": 5, "ftol": 1e-12, "gtol": 1e-12},
            )

        self.assertFalse(result.success)
        self.assertEqual(len(result.history), 1)
        self.assertEqual(result.alm_summary["history_truncated_count"], 2)
        self.assertEqual(result.history[0]["outer_termination"], "max_outer")
        self.assertEqual(materialize_diagnostics.call_count, 1)
        self.assertNotIn(
            module._HISTORY_DIAGNOSTICS_SOURCE_KEY,
            result.history[0],
        )

    def test_history_diagnostic_materialization_is_pure_and_snapshots_are_owned(self):
        module = load_alm_utils_module()
        source_key = module._HISTORY_DIAGNOSTICS_SOURCE_KEY
        diagnostic_array = np.array([3.0])
        entry = {
            "action": "subproblem_continue",
            "constraint_values": [1.0],
            "block_penalties": {"geometry": {"penalty": 2.0}},
            source_key: {"source": "deferred"},
        }

        with patch.object(
            module,
            "_constraint_history_diagnostics_from_source",
            return_value={
                "constraint_blocks": ["geometry"],
                "nested_diagnostic": {"values": [4.0]},
                "diagnostic_array": diagnostic_array,
            },
        ):
            materialized = module._materialize_history_entry_diagnostics(entry)
            snapshot = module._snapshot_history_entry(entry)

        self.assertIn(source_key, entry)
        self.assertNotIn(source_key, materialized)
        self.assertNotIn(source_key, snapshot)
        self.assertEqual(materialized["constraint_blocks"], ["geometry"])

        snapshot["constraint_values"][0] = 99.0
        snapshot["block_penalties"]["geometry"]["penalty"] = 99.0
        snapshot["nested_diagnostic"]["values"][0] = 99.0
        snapshot["diagnostic_array"][0] = 99.0

        self.assertEqual(entry["constraint_values"], [1.0])
        self.assertEqual(entry["block_penalties"]["geometry"]["penalty"], 2.0)
        self.assertEqual(diagnostic_array[0], 3.0)

    def test_minimize_alm_short_circuits_zero_step_infeasible_stall(self):
        module = load_alm_utils_module()
        source_key = module._HISTORY_DIAGNOSTICS_SOURCE_KEY
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
                    "history_had_deferred_source": source_key in history[-1],
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
        self.assertTrue(history_snapshots[-1]["history_had_deferred_source"])
        self.assertNotIn(source_key, result.history[0])
        self.assertNotIn(
            source_key,
            history_snapshots[-1]["latest_entry"],
        )
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

    def test_minimize_alm_applies_multiplier_cap_in_normalized_units(self):
        module = load_alm_utils_module()
        settings = module.ALMSettings(
            max_outer_iterations=2,
            penalty_init=4.0,
            penalty_scale=10.0,
            feasibility_tol=1e-8,
            stationarity_tol=1e-8,
            relaxed_feasibility_gate_cap=1.0,
            multiplier_max=0.2,
        )
        scale = 8.0
        raw_violation = 0.5
        normalized_violation = raw_violation / scale

        def evaluate_problem(x, multipliers, penalty):
            del multipliers, penalty
            x = np.asarray(x, dtype=float)
            return {
                "total": float(np.dot(x, x)),
                "grad": 2.0 * x,
                "constraint_values": np.array([normalized_violation]),
                "normalized_signed_constraint_values": np.array([normalized_violation]),
                "normalized_feasibility_values": np.array([normalized_violation]),
                "raw_dual_update_values": np.array([raw_violation]),
                "raw_hard_violation_values": np.array([raw_violation]),
                "constraint_scales": np.array([scale]),
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
                ["normalized_current"],
                evaluate_problem,
                settings,
                {"maxiter": 5, "ftol": 1e-12, "gtol": 1e-12},
            )

        self.assertEqual(result.history[0]["post_update_multipliers"], [0.2])
        self.assertTrue(result.history[0]["multiplier_cap_binding"])
        self.assertEqual(result.history[0]["multiplier_cap_binding_indices"], [0])
        np.testing.assert_allclose(result.multipliers, [0.2])
        np.testing.assert_allclose(result.raw_dual_estimates, [0.2 / scale])
        self.assertTrue(result.multiplier_cap_binding)

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
                "constraint_names",
                "inner_iterations",
                "inner_success",
                "inner_message",
                "penalty",
                "penalty_values",
                "block_penalties",
                "max_violation",
                "stationarity_norm",
                "raw_stationarity_norm",
                "kkt_stationarity_norm",
                "constraint_values",
                "solver_constraint_values",
                "hard_signed_constraint_values",
                "hard_violation_values",
                "surrogate_signed_constraint_values",
                "raw_signed_constraint_values",
                "normalized_signed_constraint_values",
                "raw_hard_violation_values",
                "normalized_feasibility_values",
                "constraint_scales",
                "constraint_blocks",
                "normalized_multipliers",
                "raw_dual_estimates",
                "positive_shift_values",
                "augmented_term_by_constraint",
                "active_pressure_by_constraint",
                "surrogate_minus_hard_normalized_gap",
                "surrogate_hard_sign_mismatch_by_constraint",
                "objective_to_augmented_term_ratio",
                "augmented_gradient_norm",
                "surrogate_kkt_stationarity_norm",
                "multiplier_interpretation",
                "max_raw_hard_violation",
                "block_max_raw_hard_violation",
                "block_max_normalized_violation",
                "block_augmented_term",
                "block_positive_shift_norm",
                "blocking_constraint_name",
                "blocking_constraint_block",
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
                "inner_lbfgsb_projected_gradient_norm",
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
        self.assertEqual(
            result.alm_summary["termination_reason"],
            result.termination_reason,
        )
        self.assertIn("max_normalized_violation", result.alm_summary)
        self.assertIn("blocking_constraint_name", result.alm_summary)
        result_fields = module.alm_result_diagnostics_fields(result)
        self.assertIs(result_fields["ALM_SUMMARY"], result.alm_summary)
        self.assertEqual(
            result_fields["ALM_MULTIPLIER_INTERPRETATION"],
            "differentiable_alm_multipliers",
        )
        self.assertIn("ALM_FINAL_AUGMENTED_GRADIENT_NORM", result_fields)
        self.assertIn("ALM_FINAL_SURROGATE_KKT_STATIONARITY_NORM", result_fields)

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

    def test_minimize_alm_rejects_initial_penalty_above_cap(self):
        module = load_alm_utils_module()
        settings = module.ALMSettings(
            penalty_init=1.0,
            penalty_max=5.0,
        )

        def evaluate_problem(x, multipliers, penalty):
            del x, multipliers, penalty
            raise AssertionError("initial penalty validation should run before evaluation")

        with self.assertRaisesRegex(ValueError, "initial ALM penalty"):
            module.minimize_alm(
                np.zeros(1),
                ["demo_constraint"],
                evaluate_problem,
                settings,
                {"maxiter": 1},
                initial_penalty=10.0,
            )

    def test_minimize_alm_reports_block_penalty_caps_per_block(self):
        module = load_alm_utils_module()
        settings = module.ALMSettings(
            max_outer_iterations=1,
            max_subproblem_continuations=0,
            penalty_init=1.0,
            penalty_scale=10.0,
            penalty_max=1.0e8,
            feasibility_tol=1e-8,
            stationarity_tol=1e-8,
            max_inner_attempts=1,
            block_penalties_enabled=True,
            block_penalty_max={"geometry": 5.0, "current": 1.0e8},
        )
        penalty_arguments = []

        def evaluate_problem(x, multipliers, penalty):
            del x
            penalty_arguments.append(np.asarray(penalty, dtype=float).copy())
            return module.augmented_inequality_objective(
                base_value=0.0,
                base_grad=np.zeros(1),
                constraint_values=np.array([0.5, 0.0]),
                constraint_grads=[np.zeros(1), np.zeros(1)],
                multipliers=multipliers,
                penalty=penalty,
            )

        def fake_minimize(fun, x, jac, method, bounds, callback, options):
            del fun, jac, method, bounds, callback, options
            return SimpleNamespace(
                x=np.asarray(x, dtype=float).copy(),
                nit=1,
                success=True,
                message="CONVERGENCE",
            )

        with patch.object(module, "minimize", side_effect=fake_minimize):
            result = module.minimize_alm(
                np.zeros(1),
                ["gap", "current"],
                evaluate_problem,
                settings,
                {"maxiter": 1},
                constraint_blocks=["geometry", "current"],
            )

        np.testing.assert_allclose(penalty_arguments[0], [1.0, 1.0])
        self.assertEqual(result.termination_reason, "penalty_cap_reached")
        self.assertTrue(result.penalty_cap_reached)
        self.assertEqual(result.block_penalties, {"geometry": 1.0, "current": 1.0})
        self.assertEqual(result.penalty_values, [1.0, 1.0])
        self.assertEqual(
            result.block_penalty_cap_reached,
            {"geometry": True, "current": False},
        )
        self.assertEqual(
            result.block_penalty_cap_requested,
            {"geometry": 10.0, "current": None},
        )
        self.assertEqual(
            result.history[0]["block_penalty_growth_blocks"],
            [],
        )
        self.assertEqual(
            result.history[0]["block_penalty_cap_reached"],
            {"geometry": True, "current": False},
        )

    def test_block_penalty_cap_branch_refreshes_simultaneous_growth_history(self):
        module = load_alm_utils_module()
        settings = module.ALMSettings(
            max_outer_iterations=1,
            max_subproblem_continuations=0,
            penalty_init=1.0,
            feasibility_tol=1.0e-8,
            stationarity_tol=1.0e-8,
            max_inner_attempts=1,
            block_penalties_enabled=True,
            block_penalty_init={"geometry": 1.0, "current": 1.0},
            block_penalty_scale={"geometry": 10.0, "current": 100.0},
            block_penalty_max={"geometry": 5.0, "current": 1.0e8},
        )

        def evaluate_problem(x, multipliers, penalty):
            del x
            return module.augmented_inequality_objective(
                base_value=0.0,
                base_grad=np.zeros(1),
                constraint_values=np.array([0.5, 0.5]),
                constraint_grads=[np.ones(1), np.ones(1)],
                multipliers=multipliers,
                penalty=penalty,
            )

        def fake_minimize(fun, x, jac, method, bounds, callback, options):
            del fun, jac, method, bounds, callback, options
            return SimpleNamespace(
                x=np.asarray(x, dtype=float).copy(),
                nit=1,
                success=True,
                message="CONVERGENCE",
            )

        with patch.object(module, "minimize", side_effect=fake_minimize):
            result = module.minimize_alm(
                np.zeros(1),
                ["gap", "current"],
                evaluate_problem,
                settings,
                {"maxiter": 1},
                constraint_blocks=["geometry", "current"],
            )

        self.assertFalse(result.success)
        self.assertEqual(result.termination_reason, "penalty_cap_reached")
        self.assertEqual(result.block_penalties, {"geometry": 1.0, "current": 100.0})
        self.assertEqual(result.penalty, 100.0)
        self.assertEqual(result.penalty_values, [1.0, 100.0])
        self.assertAlmostEqual(result.final_objective, 12.625)
        self.assertAlmostEqual(result.final_augmented_gradient_norm, 50.5)
        self.assertEqual(result.history[0]["action"], "penalty_cap_reached")
        self.assertEqual(result.history[0]["block_penalty_growth_blocks"], ["current"])
        self.assertEqual(
            result.history[0]["block_penalty_cap_reached"],
            {"geometry": True, "current": False},
        )
        self.assertEqual(result.history[0]["block_penalties"], {"geometry": 1.0, "current": 100.0})
        self.assertEqual(result.history[0]["penalty"], 100.0)
        self.assertEqual(result.history[0]["penalty_values"], [1.0, 100.0])
        self.assertEqual(result.history[0]["positive_shift_values"], [0.5, 50.0])
        self.assertAlmostEqual(result.history[0]["augmented_gradient_norm"], 50.5)

    def test_block_penalty_cap_branch_refreshes_growth_tolerance_schedule(self):
        module = load_alm_utils_module()
        settings = module.ALMSettings(
            max_outer_iterations=1,
            max_subproblem_continuations=0,
            penalty_init=1.0,
            feasibility_tol=1.0e-8,
            stationarity_tol=1.0e-8,
            relaxed_feasibility_gate_cap=10.0,
            max_inner_attempts=1,
            block_penalties_enabled=True,
            block_penalty_init={"geometry": 100.0, "current": 1.0},
            block_penalty_scale={"geometry": 10.0, "current": 2.0},
            block_penalty_max={"geometry": 500.0, "current": 1.0e8},
        )

        def evaluate_problem(x, multipliers, penalty):
            del x
            return module.augmented_inequality_objective(
                base_value=0.0,
                base_grad=np.zeros(1),
                constraint_values=np.array([2.0, 2.0]),
                constraint_grads=[np.ones(1), np.ones(1)],
                multipliers=multipliers,
                penalty=penalty,
            )

        def fake_minimize(fun, x, jac, method, bounds, callback, options):
            del fun, jac, method, bounds, callback, options
            return SimpleNamespace(
                x=np.asarray(x, dtype=float).copy(),
                nit=1,
                success=True,
                message="CONVERGENCE",
            )

        with patch.object(module, "minimize", side_effect=fake_minimize):
            result = module.minimize_alm(
                np.zeros(1),
                ["gap", "current"],
                evaluate_problem,
                settings,
                {"maxiter": 1},
                constraint_blocks=["geometry", "current"],
            )

        self.assertEqual(result.termination_reason, "penalty_cap_reached")
        self.assertEqual(result.block_penalties, {"geometry": 100.0, "current": 2.0})
        self.assertEqual(result.history[0]["block_penalty_growth_blocks"], ["current"])
        self.assertEqual(
            result.history[0]["block_penalty_cap_reached"],
            {"geometry": True, "current": False},
        )
        self.assertAlmostEqual(result.history[0]["feasibility_tolerance"], 0.5)
        self.assertAlmostEqual(result.history[0]["effective_feasibility_tolerance"], 0.5)
        self.assertAlmostEqual(result.history[0]["stationarity_tolerance"], 0.5)

    def test_block_penalty_growth_refreshes_result_and_history_penalties(self):
        module = load_alm_utils_module()
        settings = module.ALMSettings(
            max_outer_iterations=1,
            max_subproblem_continuations=0,
            penalty_init=1.0,
            feasibility_tol=1.0e-8,
            stationarity_tol=1.0e-8,
            max_inner_attempts=1,
            block_penalties_enabled=True,
            block_penalty_init={"geometry": 1.0, "current": 1.0},
            block_penalty_scale={"geometry": 10.0, "current": 2.0},
        )

        def evaluate_problem(x, multipliers, penalty):
            del x
            return module.augmented_inequality_objective(
                base_value=0.0,
                base_grad=np.zeros(1),
                constraint_values=np.array([0.5, 0.0]),
                constraint_grads=[np.ones(1), np.zeros(1)],
                multipliers=multipliers,
                penalty=penalty,
            )

        def fake_minimize(fun, x, jac, method, bounds, callback, options):
            del fun, jac, method, bounds, callback, options
            return SimpleNamespace(
                x=np.asarray(x, dtype=float).copy(),
                nit=1,
                success=True,
                message="CONVERGENCE",
            )

        with patch.object(module, "minimize", side_effect=fake_minimize):
            result = module.minimize_alm(
                np.zeros(1),
                ["gap", "current"],
                evaluate_problem,
                settings,
                {"maxiter": 1},
                constraint_blocks=["geometry", "current"],
            )

        self.assertFalse(result.success)
        self.assertEqual(result.termination_reason, "max_outer_after_infeasible_stall")
        self.assertEqual(result.block_penalties, {"geometry": 10.0, "current": 1.0})
        self.assertEqual(result.penalty, 10.0)
        self.assertEqual(result.penalty_values, [10.0, 1.0])
        self.assertAlmostEqual(result.final_objective, 1.25)
        self.assertAlmostEqual(result.final_augmented_gradient_norm, 5.0)
        self.assertEqual(result.history[0]["action"], "infeasible_stall_penalty_increase")
        self.assertEqual(result.history[0]["block_penalty_growth_blocks"], ["geometry"])
        self.assertEqual(result.history[0]["block_penalties"], {"geometry": 10.0, "current": 1.0})
        self.assertEqual(result.history[0]["penalty"], 10.0)
        self.assertEqual(result.history[0]["penalty_values"], [10.0, 1.0])
        self.assertEqual(result.history[0]["positive_shift_values"], [5.0, 0.0])
        self.assertAlmostEqual(result.history[0]["augmented_gradient_norm"], 5.0)

    def test_block_penalty_tolerance_schedule_tracks_slowest_block_penalty(self):
        module = load_alm_utils_module()
        settings = module.ALMSettings(
            max_outer_iterations=2,
            max_subproblem_continuations=0,
            penalty_init=1.0,
            penalty_scale=10.0,
            feasibility_tol=1.0e-8,
            stationarity_tol=1.0e-8,
            relaxed_feasibility_gate_cap=10.0,
            max_inner_attempts=1,
            block_penalties_enabled=True,
            block_penalty_init={"geometry": 100.0, "current": 1.0},
            block_penalty_scale={"geometry": 10.0, "current": 2.0},
        )
        penalty_arguments = []

        def evaluate_problem(x, multipliers, penalty):
            del x
            penalty_arguments.append(np.asarray(penalty, dtype=float).copy())
            return module.augmented_inequality_objective(
                base_value=0.0,
                base_grad=np.zeros(1),
                constraint_values=np.array([0.5, 0.0]),
                constraint_grads=[np.zeros(1), np.zeros(1)],
                multipliers=multipliers,
                penalty=penalty,
            )

        def fake_minimize(fun, x, jac, method, bounds, callback, options):
            del fun, jac, method, bounds, callback, options
            return SimpleNamespace(
                x=np.asarray(x, dtype=float).copy(),
                nit=1,
                success=True,
                message="CONVERGENCE",
            )

        with patch.object(module, "minimize", side_effect=fake_minimize):
            result = module.minimize_alm(
                np.zeros(1),
                ["gap", "current"],
                evaluate_problem,
                settings,
                {"maxiter": 1},
                constraint_blocks=["geometry", "current"],
            )

        np.testing.assert_allclose(penalty_arguments[0], [100.0, 1.0])
        self.assertEqual(result.penalty, 100.0)
        self.assertEqual(result.history[0]["penalty"], 100.0)
        self.assertEqual(result.history[0]["penalty_values"], [100.0, 1.0])
        self.assertEqual(
            result.history[0]["block_penalties"],
            {"geometry": 100.0, "current": 1.0},
        )
        self.assertEqual(result.history[0]["action"], "dual_update")
        self.assertAlmostEqual(result.history[0]["feasibility_tolerance"], 1.0)
        self.assertAlmostEqual(result.history[0]["effective_feasibility_tolerance"], 1.0)
        self.assertAlmostEqual(result.history[0]["stationarity_tolerance"], 1.0)
        self.assertEqual(result.history[1]["action"], "dual_update")
        self.assertAlmostEqual(result.history[1]["feasibility_tolerance"], 0.5)
        self.assertAlmostEqual(result.history[1]["effective_feasibility_tolerance"], 0.5)
        self.assertAlmostEqual(result.history[1]["stationarity_tolerance"], 0.5)


if __name__ == "__main__":
    unittest.main()
