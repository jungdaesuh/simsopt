import ast
import dataclasses
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


def _read_cited_source(repo_root: Path, citation: str) -> str:
    path_text, line_text = citation.rsplit(":", 1)
    start_text, end_text = line_text.split("-")
    start = int(start_text)
    end = int(end_text)
    source_lines = (repo_root / path_text).read_text(encoding="utf-8").splitlines()
    return "\n".join(source_lines[start - 1:end])


class AlmHybridSignalContractDocTests(unittest.TestCase):
    def test_documented_code_citations_still_point_to_contract_anchors(self):
        repo_root = Path(__file__).resolve().parents[2]
        doc_path = repo_root / "docs" / "alm_hybrid_signal_contract_2026-05-08.md"
        doc = doc_path.read_text(encoding="utf-8")
        citations = {
            "examples/single_stage_optimization/banana_opt/stage2_objectives.py:2165-2172": (
                "augmented_inequality_objective("
            ),
            "examples/single_stage_optimization/banana_opt/stage2_objectives.py:2173-2201": (
                '"hard_dual_update_values"'
            ),
            "examples/single_stage_optimization/alm_utils.py:2183-2235": (
                "def _extract_stage2_constraint_signal_state"
            ),
            "examples/single_stage_optimization/alm_utils.py:3303-3319": (
                "routing_state.signal_state.preferred_dual_update_values"
            ),
            "examples/single_stage_optimization/alm_utils.py:2277-2335": (
                "def _constraint_routing_state"
            ),
            "examples/single_stage_optimization/alm_utils.py:4485-4495": (
                "and not signal_mismatch_active"
            ),
            "examples/single_stage_optimization/alm_utils.py:4181-4194": (
                "and not run_state.last_cap_binding_active"
            ),
        }
        for citation, anchor in citations.items():
            with self.subTest(citation=citation):
                self.assertIn(citation, doc)
                self.assertIn(anchor, _read_cited_source(repo_root, citation))


def _complete_alm_evaluation(evaluation: dict) -> dict:
    completed = dict(evaluation)
    if "constraint_values" in completed:
        constraint_values = np.asarray(completed["constraint_values"], dtype=float)
        if "feasibility_values" not in completed:
            completed["feasibility_values"] = np.maximum(constraint_values, 0.0)
        if "dual_update_values" not in completed:
            completed["dual_update_values"] = constraint_values.copy()
    return completed


class _CountingName:
    def __init__(self, name: str) -> None:
        self._name = name
        self.str_calls = 0

    def __str__(self) -> str:
        self.str_calls += 1
        return self._name


class _CountingBlock:
    def __init__(self, block: str) -> None:
        self._block = block
        self.str_calls = 0

    def __str__(self) -> str:
        self.str_calls += 1
        return self._block


class _CountingBlockContainer:
    def __init__(self, items) -> None:
        self._items = tuple(str(item) for item in items)
        self.iter_calls = 0

    def __iter__(self):
        self.iter_calls += 1
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index):
        return self._items[index]


def _make_fake_minimize():
    def fake_minimize(fun, x, jac, method, bounds, callback, options):
        del jac, method, bounds, callback
        inner_max = int(options.get("maxiter", 1))
        x_arr = np.asarray(x, dtype=float)
        for _ in range(inner_max):
            fun(x_arr)
        return SimpleNamespace(
            x=x_arr,
            nit=inner_max,
            success=False,
            message="MAXITER",
        )

    return fake_minimize


class ResidualHelperTests(unittest.TestCase):
    def test_bound_residuals_clamp_satisfied_constraints(self):
        module = load_alm_utils_module()

        self.assertEqual(module.upper_bound_residual(1.0, 2.0), 0.0)
        self.assertEqual(module.upper_bound_residual(2.5, 2.0), 0.5)
        self.assertEqual(module.lower_bound_residual(2.5, 2.0), 0.0)
        self.assertEqual(module.lower_bound_residual(1.5, 2.0), 0.5)

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

    def test_extract_constraint_state_requires_dual_update_values(self):
        module = load_alm_utils_module()

        with self.assertRaisesRegex(KeyError, "dual_update_values"):
            module._extract_constraint_state(
                {
                    "constraint_values": np.array([0.5]),
                    "feasibility_values": np.array([0.5]),
                }
            )

    def test_meaningful_progress_accepts_low_single_digit_objective_drop(self):
        module = load_alm_utils_module()

        self.assertTrue(
            module._made_meaningful_inner_progress(
                np.zeros(1),
                np.zeros(1),
                100.0,
                98.0,
                0.25,
                0.25,
                1.0,
                1.0,
            )
        )

    def test_meaningful_progress_rejects_objective_noise_at_fixed_feasibility(self):
        module = load_alm_utils_module()

        self.assertFalse(
            module._made_meaningful_inner_progress(
                np.zeros(1),
                np.zeros(1),
                100.0,
                100.0 - 1.0e-12 * 100.0,
                0.25,
                0.25,
                1.0,
                1.0,
            )
        )

    def test_meaningful_progress_accepts_objective_drop_above_rtol_edge(self):
        module = load_alm_utils_module()

        self.assertTrue(
            module._made_meaningful_inner_progress(
                np.zeros(1),
                np.zeros(1),
                100.0,
                100.0 - 2.0 * module._INFEASIBLE_STALL_OBJECTIVE_RTOL * 100.0,
                0.25,
                0.25,
                1.0,
                1.0,
            )
        )

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

    def test_project_nonnegative_multipliers_with_diagnostics_rejects_nonfinite(self):
        # H3: NaN/Inf in updated multipliers must surface as ValueError.
        # `updated > cap` is False for NaN, and `np.minimum(NaN, cap)` is NaN,
        # so without the H3 guard a non-finite multiplier would propagate
        # silently with `cap_binding=False`. Defense-in-depth after H2 closes
        # the upstream `_nonfinite_evaluation_fields` whitelist gap.
        module = load_alm_utils_module()
        with self.assertRaisesRegex(ValueError, "non-finite"):
            module._project_nonnegative_multipliers_with_diagnostics(
                multipliers=np.array([0.0]),
                dual_update_values=np.array([np.nan]),
                penalty=1.0,
                multiplier_max=10.0,
            )
        with self.assertRaisesRegex(ValueError, "non-finite"):
            module._project_nonnegative_multipliers_with_diagnostics(
                multipliers=np.array([np.inf]),
                dual_update_values=np.array([0.0]),
                penalty=1.0,
                multiplier_max=10.0,
            )

    def test_minimize_alm_records_constraint_blocks_as_diagnostics_only(self):
        module = load_alm_utils_module()
        settings = module.ALMSettings(
            penalty_init=1.0,
            penalty_scale=10.0,
            feasibility_tol=1e-8,
            stationarity_tol=1e-8,
            max_outer_iterations=1,
            history_max_entries=None,
        )
        penalty_arguments = []

        def evaluate_problem(x, multipliers, penalty):
            del x
            penalty_arguments.append(float(penalty))
            return module.augmented_inequality_objective(
                base_value=0.0,
                base_grad=np.zeros(1),
                constraint_values=np.array([0.0, 0.0]),
                constraint_grads=[np.zeros(1), np.zeros(1)],
                multipliers=multipliers,
                penalty=penalty,
            )

        result = module.minimize_alm(
            np.zeros(1),
            ["gap", "current"],
            evaluate_problem,
            settings,
            {"maxiter": 1},
            constraint_blocks=["geometry", "current"],
        )

        self.assertTrue(result.success)
        self.assertEqual(penalty_arguments, [1.0])
        self.assertIsNone(result.block_penalties)
        self.assertIsNone(result.block_penalty_cap_reached)
        self.assertEqual(result.penalty_values, [1.0, 1.0])
        self.assertEqual(result.history[0]["constraint_blocks"], ["geometry", "current"])
        self.assertIsNone(result.history[0]["block_penalties"])
        self.assertEqual(
            result.history[0]["block_max_normalized_violation"],
            {"geometry": 0.0, "current": 0.0},
        )

    def test_minimize_alm_rejects_mismatched_constraint_block_metadata(self):
        module = load_alm_utils_module()

        def evaluate_problem(x, multipliers, penalty):
            del x, multipliers, penalty
            return module.augmented_inequality_objective(
                base_value=0.0,
                base_grad=np.zeros(1),
                constraint_values=np.array([0.0]),
                constraint_grads=[np.zeros(1)],
                multipliers=np.zeros(1),
                penalty=1.0,
            )

        with self.assertRaisesRegex(ValueError, "constraint_blocks length"):
            module.minimize_alm(
                np.zeros(1),
                ["gap"],
                evaluate_problem,
                module.ALMSettings(),
                {"maxiter": 1},
                constraint_blocks=["geometry", "current"],
            )

    def test_build_constraint_metadata_tuples_stringifies_each_input_once(self):
        module = load_alm_utils_module()
        names = [_CountingName(f"c_{index}") for index in range(3)]
        blocks = [_CountingBlock(f"g_{index}") for index in range(3)]

        names_tuple, blocks_tuple = module._build_constraint_metadata_tuples(
            names,
            blocks,
        )

        self.assertIsInstance(names_tuple, tuple)
        self.assertIsInstance(blocks_tuple, tuple)
        self.assertEqual(names_tuple, ("c_0", "c_1", "c_2"))
        self.assertEqual(blocks_tuple, ("g_0", "g_1", "g_2"))
        for counter in names:
            self.assertEqual(counter.str_calls, 1)
        for counter in blocks:
            self.assertEqual(counter.str_calls, 1)

    def _run_metadata_names_with_maxiter(self, maxiter: int) -> int:
        module = load_alm_utils_module()
        settings = module.ALMSettings(
            max_outer_iterations=1,
            max_subproblem_continuations=1,
            max_inner_attempts=1,
            penalty_init=1.0,
            penalty_scale=10.0,
            feasibility_tol=1.0e-8,
            stationarity_tol=1.0e-8,
        )
        names = [_CountingName(f"c_{index}") for index in range(3)]

        def evaluate_problem(x, multipliers, penalty):
            del multipliers, penalty
            x = np.asarray(x, dtype=float)
            return _complete_alm_evaluation(
                {
                    "total": float(np.dot(x, x)),
                    "grad": np.zeros_like(x),
                    "constraint_values": np.full(len(names), 0.5),
                    "stationarity_norm": 0.0,
                }
            )

        with patch.object(module, "minimize", side_effect=_make_fake_minimize()):
            module.minimize_alm(
                np.array([0.0]),
                names,
                evaluate_problem,
                settings,
                {"maxiter": maxiter, "ftol": 1.0e-12, "gtol": 1.0e-12},
                constraint_blocks=["geometry", "current", "topology"],
            )
        return max(counter.str_calls for counter in names)

    def test_attach_constraint_metadata_names_does_not_scale_with_inner_iterations(self):
        short = self._run_metadata_names_with_maxiter(5)
        long = self._run_metadata_names_with_maxiter(100)

        self.assertEqual(
            long,
            short,
            (
                "constraint_names str() scales with inner iterations: "
                f"short(maxiter=5)={short}, long(maxiter=100)={long}"
            ),
        )

    def _run_metadata_blocks_with_maxiter(self, maxiter: int) -> int:
        module = load_alm_utils_module()
        settings = module.ALMSettings(
            max_outer_iterations=1,
            max_subproblem_continuations=1,
            max_inner_attempts=1,
            penalty_init=1.0,
            penalty_scale=10.0,
            feasibility_tol=1.0e-8,
            stationarity_tol=1.0e-8,
        )
        counting_blocks = _CountingBlockContainer(["geometry", "current"])

        def injecting_helper(constraint_names, constraint_blocks):
            del constraint_blocks
            names_tuple = tuple(str(name) for name in constraint_names)
            return names_tuple, counting_blocks

        def evaluate_problem(x, multipliers, penalty):
            del multipliers, penalty
            x = np.asarray(x, dtype=float)
            return _complete_alm_evaluation(
                {
                    "total": float(np.dot(x, x)),
                    "grad": np.zeros_like(x),
                    "constraint_values": np.array([0.5, 0.25]),
                    "stationarity_norm": 0.0,
                }
            )

        with patch.object(
            module,
            "_build_constraint_metadata_tuples",
            side_effect=injecting_helper,
        ), patch.object(module, "minimize", side_effect=_make_fake_minimize()):
            module.minimize_alm(
                np.array([0.0]),
                ["c_0", "c_1"],
                evaluate_problem,
                settings,
                {"maxiter": maxiter, "ftol": 1.0e-12, "gtol": 1.0e-12},
                constraint_blocks=["geometry", "current"],
            )
        return counting_blocks.iter_calls

    def test_attach_constraint_metadata_blocks_does_not_scale_with_inner_iterations(self):
        short = self._run_metadata_blocks_with_maxiter(5)
        long = self._run_metadata_blocks_with_maxiter(100)

        self.assertEqual(
            long,
            short,
            (
                "constraint_blocks iteration scales with inner iterations: "
                f"short(maxiter=5)={short}, long(maxiter=100)={long}"
            ),
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
        # KKT residual is unavailable when grad_f is structurally zero (e.g.
        # `thresholded_physics` mode), because nnls collapses to lambda=0,
        # residual=0 regardless of multiplier quality.
        self.assertIsNone(diagnostics["surrogate_kkt_stationarity_norm"])
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
            "dual_update_values": np.array([2.0]),
            "hard_signed_constraint_values": np.array([2.0]),
            "hard_violation_values": np.array([2.0]),
            "surrogate_signed_constraint_values": np.array([0.0]),
            "hard_dual_update_values": np.array([2.0]),
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

    def test_surrogate_kkt_stationarity_norm_prefers_base_grad(self):
        # H4 (mirror of M9 from bf936a0a4): the surrogate KKT diagnostic must
        # consume bare grad_f (`base_grad`), not the augmented gradient
        # (`metric_grad`/`grad`). Otherwise the residual collapses to ~0 once
        # L-BFGS-B converges and hides multiplier-quality defects, even though
        # the dual problem is far from optimal.
        module = load_alm_utils_module()
        # base_grad differs from grad. NNLS for `1.0 * lambda = -base_grad[0]`
        # with lambda >= 0 clamps to 0 (RHS positive impossible), so the
        # residual norm is 2.
        # If the helper used `grad` (augmented, zero at inner-solve minimum),
        # the residual would be 0.
        evaluation_with_base = {
            "total": 0.0,
            "grad": np.array([0.0]),
            "base_grad": np.array([2.0]),
            "constraint_values": np.array([0.0]),
            "feasibility_values": np.array([0.0]),
            "dual_update_values": np.array([0.0]),
            "hard_signed_constraint_values": np.array([0.0]),
            "hard_violation_values": np.array([0.0]),
            "surrogate_signed_constraint_values": np.array([0.0]),
            "hard_dual_update_values": np.array([0.0]),
            "constraint_grads": [np.array([1.0])],
            "constraint_activity_tolerances": np.array([0.0]),
        }
        routing_state_with_base = module._constraint_routing_state(
            evaluation_with_base,
            np.zeros(1),
            1.0,
            feasibility_gate=1.0,
        )

        self.assertAlmostEqual(
            module._surrogate_kkt_stationarity_norm(
                evaluation_with_base,
                routing_state_with_base,
                feasibility_gate=1.0,
            ),
            2.0,
        )

        # Backward-compat: when `base_grad` is absent, fall back to
        # `metric_grad` then `grad` (preserves the pre-H4 contract). Use a
        # non-zero `grad` so the diagnostic is meaningful; a structurally
        # zero gradient (e.g. `thresholded_physics` mode) returns `None`.
        evaluation_no_base = dict(evaluation_with_base)
        del evaluation_no_base["base_grad"]
        evaluation_no_base["grad"] = np.array([3.0])
        routing_state_no_base = module._constraint_routing_state(
            evaluation_no_base,
            np.zeros(1),
            1.0,
            feasibility_gate=1.0,
        )

        # NNLS for `1.0 * lambda = -3.0` with lambda >= 0 clamps to 0,
        # leaving residual norm 3.
        self.assertAlmostEqual(
            module._surrogate_kkt_stationarity_norm(
                evaluation_no_base,
                routing_state_no_base,
                feasibility_gate=1.0,
            ),
            3.0,
        )

        # Structural-zero gradient (e.g. `thresholded_physics`): `None`, not 0.
        evaluation_zero_grad = dict(evaluation_with_base)
        del evaluation_zero_grad["base_grad"]
        routing_state_zero_grad = module._constraint_routing_state(
            evaluation_zero_grad,
            np.zeros(1),
            1.0,
            feasibility_gate=1.0,
        )
        self.assertIsNone(
            module._surrogate_kkt_stationarity_norm(
                evaluation_zero_grad,
                routing_state_zero_grad,
                feasibility_gate=1.0,
            ),
        )

    def test_alm_subproblem_continue_marks_max_outer_when_is_final_outer(self):
        # Audit-v2 H11: `_emit_alm_subproblem_continue` is called from both the
        # signal-mismatch retry arm and the feasible-update retry arm; when
        # `is_final_outer=True` it must annotate `outer_termination="max_outer"`
        # on the history entry so `_termination_reason_from_history` reports a
        # `max_outer*` label rather than surfacing the bare action string.
        # The signal-mismatch path is exercised by
        # `test_minimize_alm_escalates_penalty_after_repeated_stage2_signal_mismatch`;
        # this test pins the helper directly so the feasible-update retry call
        # site cannot regress without flipping a test.
        module = load_alm_utils_module()
        state = module._ContinuationStepState(
            multipliers=np.zeros(1),
            penalty=1.0,
            update_feasibility_tol=1.0,
            update_stationarity_tol=1.0,
            feasible_stall_count=2,
            last_result=None,
            final_eval=None,
            final_multipliers=np.zeros(1),
            final_penalty=1.0,
            best_feasible=None,
            inner_options=None,
        )
        run_state = module.ALMRunState(
            x=np.zeros(1),
            total_inner_iterations=0,
            trust_radius=0.5,
            history=[],
            history_truncated_count=0,
            cap_binding_detected=False,
            cap_binding_indices=set(),
            penalty_cap_reached=False,
            penalty_cap_requested=None,
        )

        def emit_history_entry(*, is_final_outer):
            history_entry = {}
            module._emit_alm_subproblem_continue(
                state=state,
                run_state=run_state,
                history_entry=history_entry,
                history_callback=None,
                is_final_outer=is_final_outer,
            )
            return history_entry

        history_entry_final = emit_history_entry(is_final_outer=True)
        self.assertEqual(history_entry_final["action"], "subproblem_continue")
        self.assertEqual(history_entry_final["outer_termination"], "max_outer")
        self.assertEqual(history_entry_final["feasible_stall_count"], 2)

        history_entry_mid = emit_history_entry(is_final_outer=False)
        self.assertEqual(history_entry_mid["action"], "subproblem_continue")
        self.assertNotIn("outer_termination", history_entry_mid)

    def test_surrogate_hard_sign_mismatch_does_not_flag_exact_boundary(self):
        module = load_alm_utils_module()

        self.assertEqual(
            module._surrogate_hard_sign_mismatch(
                np.array([-1.0e-3, 1.0e-3, -1.0e-3]),
                np.array([0.0, -2.0e-3, 3.0e-3]),
            ),
            [False, True, True],
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

    def test_augmented_inequality_objective_exposes_solver_constraint_metadata(self):
        module = load_alm_utils_module()

        evaluation = module.augmented_inequality_objective(
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


class AlmStructuralDebtTests(unittest.TestCase):
    def _top_level_node(self, name: str):
        source_path = EXAMPLES_ROOT / "alm_utils.py"
        tree = ast.parse(source_path.read_text())
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name:
                return node
        self.fail(f"{name} not found in alm_utils.py")

    def test_minimize_alm_public_entrypoint_is_small_and_closure_free(self):
        node = self._top_level_node("minimize_alm")
        nested_functions = [
            child
            for child in ast.walk(node)
            if isinstance(child, ast.FunctionDef) and child is not node
        ]

        self.assertLessEqual(node.end_lineno - node.lineno + 1, 500)
        self.assertEqual(nested_functions, [])

    def test_extracted_alm_state_carriers_exist(self):
        module = load_alm_utils_module()

        self.assertTrue(hasattr(module, "ALMRunState"))
        self.assertTrue(hasattr(module, "ALMFinalState"))
        self.assertTrue(hasattr(module, "ALMHistoryEntry"))
        self.assertTrue(hasattr(module, "ALMInnerAttemptRequest"))
        self.assertTrue(hasattr(module, "ALMInnerAttemptResult"))

    def test_minimize_alm_impl_orchestrator_is_small_and_closure_free(self):
        node = self._top_level_node("_minimize_alm_impl")
        nested_functions = [
            child
            for child in ast.walk(node)
            if isinstance(child, ast.FunctionDef) and child is not node
        ]

        self.assertLessEqual(node.end_lineno - node.lineno + 1, 500)
        self.assertEqual(nested_functions, [])

    def test_minimize_alm_impl_phase4_helpers_exist_and_closure_free(self):
        for name in (
            "_run_alm_continuation_step",
            "_run_alm_outer_iteration",
            "_ALMContinuationDecision",
            "_ALMOuterDecision",
            "_ALMContinuationStepResult",
            "_ALMOuterIterationResult",
        ):
            node = self._top_level_node(name)
            if isinstance(node, ast.FunctionDef):
                nested = [
                    child
                    for child in ast.walk(node)
                    if isinstance(child, ast.FunctionDef) and child is not node
                ]
                self.assertEqual(nested, [], f"{name} must be closure-free")

    def test_phase4_result_carriers_are_frozen_dataclasses_without_mutable_defaults(self):
        module = load_alm_utils_module()
        for name in (
            "_ALMContinuationStepResult",
            "_ALMOuterIterationResult",
        ):
            carrier = getattr(module, name)
            self.assertTrue(dataclasses.is_dataclass(carrier), name)
            self.assertTrue(carrier.__dataclass_params__.frozen, name)
            for field in dataclasses.fields(carrier):
                if field.default is not dataclasses.MISSING:
                    self.assertNotIsInstance(
                        field.default,
                        (dict, list, set),
                        f"{name}.{field.name} has a mutable default",
                    )


_HISTORY_ENTRY_SHARED_KEYS = frozenset({
    "outer_iteration", "continuation_iteration", "constraint_names",
    "inner_iterations", "inner_success", "inner_message",
    "penalty", "penalty_values", "block_penalties",
    "max_violation", "stationarity_norm", "raw_stationarity_norm",
    "kkt_stationarity_norm", "constraint_values", "solver_constraint_values",
    "hard_signed_constraint_values", "hard_violation_values",
    "surrogate_signed_constraint_values", "hard_max_violation",
    "surrogate_max_value", "hard_positive_shift_zero", "signal_mismatch_active",
    "multipliers", "post_update_multipliers", "feasibility_tolerance",
    "effective_feasibility_tolerance", "stationarity_tolerance", "trust_radius",
    "inner_maxiter", "inner_maxls", "inner_maxfun", "inner_profile",
    "inner_lbfgsb_projected_gradient_norm", "inner_attempts",
    "accepted_move_norm", "accepted_move_norm_scaled",
    "infeasible_stall_move_tolerance", "objective_delta", "feasibility_delta",
    "feasibility_delta_tolerance", "stationarity_delta", "meaningful_progress",
    "feasible_stall_count", "infeasible_stall", "inner_false_success",
    "inner_stall_reason", "active_violation_index", "active_constraint_name",
    "nonfinite_candidate_evaluation", "nonfinite_candidate_fields",
    "multiplier_cap_binding", "multiplier_cap_binding_indices",
})


class AlmDualUpdateTransitionTests(unittest.TestCase):
    @staticmethod
    def _routing_state_with_preferred(module, preferred_dual_update_values):
        signal_state = module.ALMConstraintSignalState(
            explicit_stage2_signals=False,
            hard_signed_constraint_values=preferred_dual_update_values,
            hard_violation_values=np.zeros_like(preferred_dual_update_values),
            surrogate_signed_constraint_values=preferred_dual_update_values,
            preferred_dual_update_values=preferred_dual_update_values,
        )
        return module.ALMConstraintRoutingState(
            signal_state=signal_state,
            hard_activity_mask=np.zeros_like(preferred_dual_update_values, dtype=bool),
            surrogate_activity_mask=np.zeros_like(preferred_dual_update_values, dtype=bool),
            signal_mismatch_active=False,
            hard_positive_shift=np.zeros_like(preferred_dual_update_values),
            surrogate_positive_shift=np.zeros_like(preferred_dual_update_values),
            hard_positive_shift_zero=False,
            surrogate_positive_shift_zero=False,
            hard_max_violation=0.0,
            surrogate_max_value=0.0,
        )

    def test_dual_update_projects_multipliers_and_tightens_tolerances(self):
        module = load_alm_utils_module()
        settings = module.ALMSettings(
            penalty_init=1.0,
            penalty_scale=10.0,
            penalty_max=1.0e6,
            feasibility_tol=1.0e-8,
            stationarity_tol=1.0e-8,
            multiplier_max=10.0,
        )
        multipliers = np.array([0.1, 0.2])
        routing_state = self._routing_state_with_preferred(module, np.array([0.05, 0.05]))

        result = module._handle_alm_dual_update_transition(
            multipliers=multipliers,
            routing_state=routing_state,
            penalty_argument=2.0,
            settings=settings,
            update_feasibility_tol=1.0e-2,
            update_stationarity_tol=1.0e-2,
        )

        # Tolerances must shrink by penalty_scale, floored at the absolute settings tol.
        self.assertAlmostEqual(result.update_feasibility_tol, 1.0e-3)
        self.assertAlmostEqual(result.update_stationarity_tol, 1.0e-3)
        # Multipliers must be projected non-negative; original positive values move
        # by penalty * preferred_dual_update_values consistent with the underlying
        # _project_nonnegative_multipliers_with_diagnostics contract.
        self.assertEqual(result.multipliers.shape, multipliers.shape)
        np.testing.assert_allclose(result.multipliers, np.array([0.2, 0.3]))
        self.assertTrue(np.all(result.multipliers >= 0.0))
        self.assertIsInstance(result.multiplier_cap_binding, bool)
        self.assertIsInstance(result.multiplier_cap_binding_indices, list)

    def test_dual_update_floors_tolerance_at_settings_value(self):
        module = load_alm_utils_module()
        settings = module.ALMSettings(
            penalty_init=1.0,
            penalty_scale=10.0,
            feasibility_tol=1.0e-2,
            stationarity_tol=1.0e-2,
        )
        routing_state = self._routing_state_with_preferred(module, np.array([0.0]))

        result = module._handle_alm_dual_update_transition(
            multipliers=np.array([0.0]),
            routing_state=routing_state,
            penalty_argument=1.0,
            settings=settings,
            update_feasibility_tol=1.0e-3,
            update_stationarity_tol=1.0e-3,
        )

        # update_feasibility_tol/penalty_scale = 1e-4 < 1e-2 floor, so floor wins.
        self.assertEqual(result.update_feasibility_tol, 1.0e-2)
        self.assertEqual(result.update_stationarity_tol, 1.0e-2)


class AlmPenaltyCapTerminationTests(unittest.TestCase):
    def test_handle_alm_penalty_cap_termination_forwards_all_keyword_arguments(self):
        module = load_alm_utils_module()
        captured = {}

        def fake_failure(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(termination_reason=kwargs["termination_reason"])

        penalty_update_state = SimpleNamespace(
            evaluation={"total": 1.0},
            max_violation=0.5,
        )
        penalty_transition = SimpleNamespace(
            requested_penalty=1.0e9,
            penalty_update_state=penalty_update_state,
        )
        multipliers_in = np.array([0.1])

        with patch.object(
            module, "_build_alm_failure_result_with_optional_restore", fake_failure
        ):
            module._handle_alm_penalty_cap_termination(
                settings="SETTINGS",
                constraint_names=["c0"],
                run_state="RUNSTATE",
                last_outer_iteration=4,
                best_feasible="BEST_FEASIBLE_SENTINEL",
                restore_incumbent_state_fn="RESTORE_FN_SENTINEL",
                penalty_transition=penalty_transition,
                multipliers=multipliers_in,
                penalty=1.0e3,
                result="INNER_RESULT",
            )

        # Hardcoded contract values
        self.assertEqual(captured["termination_reason"], "penalty_cap_reached")
        self.assertEqual(
            captured["restored_termination_reason"],
            "penalty_cap_reached_restored_best_feasible",
        )
        self.assertIs(captured["evaluation"], penalty_update_state.evaluation)
        self.assertEqual(
            captured["final_max_feasibility_violation"],
            penalty_update_state.max_violation,
        )
        # Message-prefix substring guards (catches accidental string drift)
        self.assertIn("ALM stopped after the requested penalty update", captured["message_prefix"])
        self.assertIn("1.000e+09", captured["message_prefix"])
        self.assertIn("1.000e+03", captured["message_prefix"])
        self.assertIn(
            "ALM stopped at the penalty cap after restoring best",
            captured["restored_message_prefix"],
        )
        # All other forwarded kwargs (catches a kwarg rename like result→inner_result)
        self.assertEqual(captured["settings"], "SETTINGS")
        self.assertEqual(captured["constraint_names"], ["c0"])
        self.assertEqual(captured["run_state"], "RUNSTATE")
        self.assertEqual(captured["last_outer_iteration"], 4)
        self.assertEqual(captured["best_feasible"], "BEST_FEASIBLE_SENTINEL")
        self.assertEqual(captured["restore_incumbent_state_fn"], "RESTORE_FN_SENTINEL")
        np.testing.assert_array_equal(captured["multipliers_state"], multipliers_in)
        self.assertEqual(captured["penalty_state"], 1.0e3)
        self.assertEqual(captured["inner_result"], "INNER_RESULT")


class AlmNormalizeRunInputsValidationTests(unittest.TestCase):
    @staticmethod
    def _settings(module, **overrides):
        defaults = dict(
            penalty_init=1.0,
            penalty_scale=10.0,
            penalty_max=1.0e6,
            feasibility_tol=1.0e-6,
            stationarity_tol=1.0e-6,
        )
        defaults.update(overrides)
        return module.ALMSettings(**defaults)

    def _normalize(self, module, settings, **overrides):
        kwargs = dict(
            x0=np.zeros(1),
            constraint_names=["c0"],
            settings=settings,
            initial_multipliers=None,
            initial_penalty=None,
            constraint_blocks=None,
            snapshot_accepted_state_fn=None,
            restore_incumbent_state_fn=None,
        )
        kwargs.update(overrides)
        return module._normalize_alm_run_inputs(**kwargs)

    def test_normalize_rejects_unpaired_snapshot_and_restore(self):
        module = load_alm_utils_module()
        settings = self._settings(module)

        with self.assertRaisesRegex(
            ValueError,
            "snapshot_accepted_state_fn and restore_incumbent_state_fn must be provided together",
        ):
            self._normalize(
                module,
                settings,
                snapshot_accepted_state_fn=lambda: None,
                restore_incumbent_state_fn=None,
            )

    def test_settings_construction_rejects_nonpositive_penalty_max(self):
        # L5: ALMSettings.__post_init__ validates penalty_max — previously this
        # check lived in _normalize_alm_run_inputs, allowing programmatic
        # ALMSettings(penalty_max=0.0) to silently sit in memory.
        module = load_alm_utils_module()

        with self.assertRaisesRegex(
            ValueError, r"ALMSettings\.penalty_max must be positive when provided"
        ):
            self._settings(module, penalty_max=0.0)

    def test_settings_construction_rejects_penalty_max_below_init(self):
        module = load_alm_utils_module()

        with self.assertRaisesRegex(
            ValueError,
            r"ALMSettings\.penalty_max \(5\.0\) must be >= penalty_init \(10\.0\)",
        ):
            self._settings(module, penalty_init=10.0, penalty_max=5.0)

    def test_settings_construction_rejects_nonpositive_history_max_entries(self):
        module = load_alm_utils_module()

        with self.assertRaisesRegex(
            ValueError,
            r"ALMSettings\.history_max_entries must be positive when provided",
        ):
            self._settings(module, history_max_entries=0)

    def test_settings_construction_rejects_nonfinite_scalar_fields(self):
        module = load_alm_utils_module()
        scalar_fields = (
            "max_outer_iterations",
            "max_subproblem_continuations",
            "penalty_init",
            "penalty_scale",
            "penalty_max",
            "feasibility_tol",
            "stationarity_tol",
            "trust_radius_init",
            "trust_radius_min",
            "trust_radius_shrink",
            "trust_radius_grow",
            "max_inner_attempts",
            "relaxed_feasibility_gate_cap",
            "multiplier_max",
            "history_max_entries",
        )

        for field in scalar_fields:
            with self.subTest(field=field):
                with self.assertRaisesRegex(
                    ValueError,
                    rf"ALMSettings\.{field} must be finite",
                ):
                    self._settings(module, **{field: float("nan")})

    def test_settings_construction_rejects_noninteger_loop_fields(self):
        module = load_alm_utils_module()
        integer_fields = (
            "max_outer_iterations",
            "max_subproblem_continuations",
            "max_inner_attempts",
            "history_max_entries",
        )

        for field in integer_fields:
            with self.subTest(field=field):
                with self.assertRaisesRegex(
                    ValueError,
                    rf"ALMSettings\.{field} must be an integer",
                ):
                    self._settings(module, **{field: 1.5})

    def test_normalize_rejects_nonfinite_initial_penalty(self):
        module = load_alm_utils_module()
        settings = self._settings(module)

        with self.assertRaisesRegex(
            ValueError, r"initial ALM penalty must be finite and positive"
        ):
            self._normalize(module, settings, initial_penalty=float("inf"))

        with self.assertRaisesRegex(
            ValueError, r"initial ALM penalty must be finite and positive"
        ):
            self._normalize(module, settings, initial_penalty=0.0)

    def test_normalize_rejects_initial_penalty_above_cap(self):
        module = load_alm_utils_module()
        settings = self._settings(module, penalty_max=5.0)

        with self.assertRaisesRegex(
            ValueError,
            r"initial ALM penalty \(10\.0\) must be <= settings\.penalty_max \(5\.0\)",
        ):
            self._normalize(module, settings, initial_penalty=10.0)

    def test_normalize_rejects_initial_multipliers_wrong_shape(self):
        module = load_alm_utils_module()
        settings = self._settings(module)

        with self.assertRaisesRegex(
            ValueError, r"ALM initial_multipliers shape \(2,\) != \(1,\)"
        ):
            self._normalize(module, settings, initial_multipliers=np.array([0.0, 1.0]))

    def test_normalize_rejects_initial_multipliers_with_nan(self):
        module = load_alm_utils_module()
        settings = self._settings(module)

        with self.assertRaisesRegex(
            ValueError, r"ALM initial_multipliers non-finite at indices \[0\]"
        ):
            self._normalize(module, settings, initial_multipliers=np.array([float("nan")]))

    def test_normalize_rejects_initial_multipliers_with_inf(self):
        module = load_alm_utils_module()
        settings = self._settings(module)

        with self.assertRaisesRegex(
            ValueError, r"ALM initial_multipliers non-finite at indices \[0\]"
        ):
            self._normalize(module, settings, initial_multipliers=np.array([float("inf")]))

    def test_normalize_rejects_negative_initial_multipliers(self):
        module = load_alm_utils_module()
        settings = self._settings(module)

        with self.assertRaisesRegex(
            ValueError, r"ALM initial_multipliers negative at indices \[0\]"
        ):
            self._normalize(module, settings, initial_multipliers=np.array([-1.0]))

    def test_normalize_accepts_zero_initial_multipliers(self):
        module = load_alm_utils_module()
        settings = self._settings(module)

        normalized = self._normalize(
            module, settings, initial_multipliers=np.array([0.0])
        )
        np.testing.assert_array_equal(normalized.multipliers, np.array([0.0]))

    def test_normalize_accepts_positive_initial_multipliers(self):
        module = load_alm_utils_module()
        settings = self._settings(module)

        normalized = self._normalize(
            module, settings, initial_multipliers=np.array([2.5])
        )
        np.testing.assert_array_equal(normalized.multipliers, np.array([2.5]))


class RequirePositiveAlmThresholdTests(unittest.TestCase):
    def test_accepts_positive_finite_value(self):
        module = load_alm_utils_module()
        self.assertEqual(module.require_positive_alm_threshold("qs", 0.5), 0.5)

    def test_accepts_subnormal_positive_value(self):
        module = load_alm_utils_module()
        # Real geometric tolerances can be ~1e-9; helper must not floor.
        self.assertEqual(
            module.require_positive_alm_threshold("micro", 1.0e-15), 1.0e-15
        )

    def test_rejects_zero(self):
        module = load_alm_utils_module()
        with self.assertRaisesRegex(
            ValueError, r"ALM threshold 'qs' must be a finite positive value"
        ):
            module.require_positive_alm_threshold("qs", 0.0)

    def test_rejects_negative(self):
        module = load_alm_utils_module()
        with self.assertRaisesRegex(
            ValueError, r"ALM threshold 'qs' must be a finite positive value"
        ):
            module.require_positive_alm_threshold("qs", -1.0)

    def test_rejects_nan(self):
        module = load_alm_utils_module()
        with self.assertRaisesRegex(
            ValueError, r"ALM threshold 'qs' must be a finite positive value"
        ):
            module.require_positive_alm_threshold("qs", float("nan"))

    def test_rejects_positive_inf(self):
        module = load_alm_utils_module()
        with self.assertRaisesRegex(
            ValueError, r"ALM threshold 'qs' must be a finite positive value"
        ):
            module.require_positive_alm_threshold("qs", float("inf"))


class ValidateAlmCliArgsTests(unittest.TestCase):
    @staticmethod
    def _args(**overrides):
        defaults = dict(
            alm_max_outer_iters=10,
            alm_max_subproblem_continuations=20,
            alm_penalty_init=1.0,
            alm_penalty_scale=10.0,
            alm_penalty_max=1.0e8,
            alm_feas_tol=1.0e-6,
            alm_stationarity_tol=1.0e-6,
            alm_trust_radius_init=0.05,
            alm_trust_radius_min=1.0e-4,
            alm_trust_radius_shrink=0.5,
            alm_trust_radius_grow=1.5,
            alm_max_inner_attempts=4,
            alm_curvature_smoothing=0.25,
            alm_distance_smoothing=0.005,
        )
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def test_rejects_nonfinite_scalar_fields(self):
        module = load_alm_utils_module()
        field_to_flag = {
            "alm_max_outer_iters": "--alm-max-outer-iters",
            "alm_max_subproblem_continuations": "--alm-max-subproblem-continuations",
            "alm_penalty_init": "--alm-penalty-init",
            "alm_penalty_scale": "--alm-penalty-scale",
            "alm_penalty_max": "--alm-penalty-max",
            "alm_feas_tol": "--alm-feas-tol",
            "alm_stationarity_tol": "--alm-stationarity-tol",
            "alm_trust_radius_init": "--alm-trust-radius-init",
            "alm_trust_radius_min": "--alm-trust-radius-min",
            "alm_trust_radius_shrink": "--alm-trust-radius-shrink",
            "alm_trust_radius_grow": "--alm-trust-radius-grow",
            "alm_max_inner_attempts": "--alm-max-inner-attempts",
            "alm_curvature_smoothing": "--alm-curvature-smoothing",
            "alm_distance_smoothing": "--alm-distance-smoothing",
        }

        for field, flag_name in field_to_flag.items():
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, rf"{flag_name} must be finite"):
                    module.validate_alm_cli_args(
                        self._args(**{field: float("nan")})
                    )

    def test_rejects_noninteger_loop_fields(self):
        module = load_alm_utils_module()
        field_to_flag = {
            "alm_max_outer_iters": "--alm-max-outer-iters",
            "alm_max_subproblem_continuations": "--alm-max-subproblem-continuations",
            "alm_max_inner_attempts": "--alm-max-inner-attempts",
        }

        for field, flag_name in field_to_flag.items():
            with self.subTest(field=field):
                with self.assertRaisesRegex(
                    ValueError,
                    rf"{flag_name} must be an integer",
                ):
                    module.validate_alm_cli_args(self._args(**{field: 1.5}))


class ValidateInitialMultipliersTests(unittest.TestCase):
    def test_accepts_zero_and_positive(self):
        module = load_alm_utils_module()
        result = module.validate_initial_multipliers([0.0, 1.5, 0.0], 3)
        np.testing.assert_array_equal(result, np.array([0.0, 1.5, 0.0]))

    def test_returns_owned_copy(self):
        module = load_alm_utils_module()
        source = np.array([0.5, 1.0])
        result = module.validate_initial_multipliers(source, 2)
        result[0] = 99.0
        # Caller's array must not be mutated.
        self.assertEqual(source[0], 0.5)

    def test_rejects_shape_mismatch_too_short(self):
        module = load_alm_utils_module()
        with self.assertRaisesRegex(
            ValueError, r"ALM initial_multipliers shape \(1,\) != \(3,\)"
        ):
            module.validate_initial_multipliers([0.0], 3)

    def test_rejects_shape_mismatch_too_long(self):
        module = load_alm_utils_module()
        with self.assertRaisesRegex(
            ValueError, r"ALM initial_multipliers shape \(4,\) != \(3,\)"
        ):
            module.validate_initial_multipliers([0.0, 1.0, 2.0, 3.0], 3)

    def test_rejects_2d_shape(self):
        module = load_alm_utils_module()
        with self.assertRaisesRegex(
            ValueError, r"ALM initial_multipliers shape \(1, 2\) != \(2,\)"
        ):
            module.validate_initial_multipliers([[0.0, 1.0]], 2)

    def test_rejects_nan_with_indices(self):
        module = load_alm_utils_module()
        with self.assertRaisesRegex(
            ValueError, r"ALM initial_multipliers non-finite at indices \[1, 3\]"
        ):
            module.validate_initial_multipliers(
                [0.0, float("nan"), 1.0, float("inf")], 4
            )

    def test_rejects_negative_with_indices(self):
        module = load_alm_utils_module()
        with self.assertRaisesRegex(
            ValueError, r"ALM initial_multipliers negative at indices \[0, 2\]"
        ):
            module.validate_initial_multipliers([-1.0, 0.0, -0.5, 1.0], 4)


class AlmHistoryEntryBuilderTests(unittest.TestCase):
    @staticmethod
    def _make_routing_state(module):
        signal_state = module.ALMConstraintSignalState(
            explicit_stage2_signals=False,
            hard_signed_constraint_values=np.zeros(2),
            hard_violation_values=np.zeros(2),
            surrogate_signed_constraint_values=np.zeros(2),
            preferred_dual_update_values=np.zeros(2),
        )
        return module.ALMConstraintRoutingState(
            signal_state=signal_state,
            hard_activity_mask=np.zeros(2, dtype=bool),
            surrogate_activity_mask=np.zeros(2, dtype=bool),
            signal_mismatch_active=False,
            hard_positive_shift=np.zeros(2),
            surrogate_positive_shift=np.zeros(2),
            hard_positive_shift_zero=False,
            surrogate_positive_shift_zero=False,
            hard_max_violation=0.1,
            surrogate_max_value=0.2,
        )

    def _common_kwargs(self, module):
        return dict(
            outer_iteration=2,
            continuation_iteration=1,
            constraint_names=["c0", "c1"],
            penalty=4.0,
            penalty_argument=4.0,
            multipliers=np.array([0.1, 0.2]),
            post_update_multipliers=np.array([0.15, 0.25]),
            routing_state=self._make_routing_state(module),
            feasibility_values=np.array([0.01, 0.02]),
            solver_constraint_values=np.array([-1.0, 1.0]),
            max_feasibility_violation=0.02,
            stationarity_norm=0.5,
            raw_stationarity_norm=0.6,
            kkt_stationarity_norm=0.4,
            signal_mismatch_active=False,
            update_feasibility_tol=1e-3,
            effective_feasibility_tol=1e-2,
            update_stationarity_tol=1e-4,
            trust_radius=0.05,
            start_x=np.array([1.0, 2.0]),
        )

    def test_build_alm_history_entry_returns_all_52_shared_keys(self):
        module = load_alm_utils_module()
        kwargs = self._common_kwargs(module)
        kwargs.update(
            inner_iterations=0,
            inner_success=True,
            inner_message="ok",
            inner_maxiter=None,
            inner_maxls=None,
            inner_maxfun=None,
            inner_profile=None,
            inner_lbfgsb_projected_gradient_norm=None,
            inner_attempts=0,
            accepted_move_norm=0.0,
            objective_delta=0.0,
            feasibility_delta=0.0,
            feasibility_delta_tolerance=0.0,
            stationarity_delta=0.0,
            meaningful_progress=False,
            feasible_stall_count=0,
            infeasible_stall=False,
            inner_false_success=False,
            inner_stall_reason=None,
            active_violation_index=None,
            active_constraint_name=None,
            nonfinite_candidate_evaluation=False,
            nonfinite_candidate_fields=None,
        )

        entry = module._build_alm_history_entry(**kwargs)

        self.assertEqual(set(entry.keys()), set(_HISTORY_ENTRY_SHARED_KEYS))
        self.assertNotIn("action", entry)

    def test_build_alm_history_entry_skipped_inner_payload_matches_post_inner_shape(self):
        module = load_alm_utils_module()

        skipped_kwargs = self._common_kwargs(module) | dict(
            inner_iterations=0, inner_success=True, inner_message="skipped",
            inner_maxiter=None, inner_maxls=None, inner_maxfun=None,
            inner_profile=None, inner_lbfgsb_projected_gradient_norm=None,
            inner_attempts=0, accepted_move_norm=0.0, objective_delta=0.0,
            feasibility_delta=0.0, feasibility_delta_tolerance=0.0,
            stationarity_delta=0.0, meaningful_progress=False,
            feasible_stall_count=0, infeasible_stall=False,
            inner_false_success=False, inner_stall_reason=None,
            active_violation_index=None, active_constraint_name=None,
            nonfinite_candidate_evaluation=False,
            nonfinite_candidate_fields=None,
        )
        post_kwargs = self._common_kwargs(module) | dict(
            inner_iterations=7, inner_success=False, inner_message="MAXITER",
            inner_maxiter=10, inner_maxls=20, inner_maxfun=50,
            inner_profile="boxed", inner_lbfgsb_projected_gradient_norm=0.3,
            inner_attempts=2, accepted_move_norm=0.04, objective_delta=0.01,
            feasibility_delta=0.001, feasibility_delta_tolerance=1e-6,
            stationarity_delta=0.05, meaningful_progress=True,
            feasible_stall_count=1, infeasible_stall=False,
            inner_false_success=False, inner_stall_reason=None,
            active_violation_index=0, active_constraint_name="c0",
            nonfinite_candidate_evaluation=False,
            nonfinite_candidate_fields=None,
        )

        skipped = module._build_alm_history_entry(**skipped_kwargs)
        post = module._build_alm_history_entry(**post_kwargs)

        self.assertEqual(set(skipped.keys()), set(post.keys()))
        skipped_with_action = skipped | {"action": "converged"}
        self.assertEqual(
            set(skipped_with_action.keys()),
            set(_HISTORY_ENTRY_SHARED_KEYS) | {"action"},
        )


class MinimizeAlmTests(unittest.TestCase):
    @staticmethod
    def _quadratic_taylor_evaluation(x, gradient_scale: float):
        x = np.asarray(x, dtype=float)
        return _complete_alm_evaluation({
            "total": 0.5 * float(np.dot(x, x)),
            "grad": float(gradient_scale) * x,
            "constraint_values": np.zeros(1),
            "stationarity_norm": float(np.linalg.norm(float(gradient_scale) * x)),
        })

    @staticmethod
    def _quartic_taylor_evaluation(x):
        x = np.asarray(x, dtype=float)
        grad = 4.0 * x**3
        return _complete_alm_evaluation({
            "total": float(np.sum(x**4)),
            "grad": grad,
            "constraint_values": np.zeros(1),
            "stationarity_norm": float(np.linalg.norm(grad)),
        })

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
            return _complete_alm_evaluation({
                "total": float(np.dot(x, x)),
                "grad": 2.0 * x,
                "constraint_values": np.zeros(1),
                "stationarity_norm": float(np.linalg.norm(2.0 * x)),
            })

        with self.assertRaisesRegex(ValueError, "must be provided together"):
            module.minimize_alm(
                np.array([0.0]),
                ["dummy"],
                evaluate_problem,
                settings,
                {"maxiter": 1},
                snapshot_accepted_state_fn=lambda: {"x": 0.0},
            )

    def test_minimize_alm_reports_constraint_blocks_from_evaluation_metadata(self):
        module = load_alm_utils_module()
        settings = module.ALMSettings(max_outer_iterations=1)

        def evaluate_problem(x, multipliers, penalty):
            del x, multipliers, penalty
            return _complete_alm_evaluation({
                "total": 0.0,
                "grad": np.zeros(1),
                "constraint_values": np.zeros(2),
                "constraint_grads": [np.zeros(1), np.zeros(1)],
                "stationarity_norm": 0.0,
                "constraint_blocks": ["geometry", "current"],
            })

        result = module.minimize_alm(
            np.array([0.0]),
            ["coil_surface_spacing", "banana_current_upper_bound"],
            evaluate_problem,
            settings,
            {"maxiter": 1},
        )

        self.assertTrue(result.success)
        self.assertEqual(result.constraint_blocks, ["geometry", "current"])
        self.assertEqual(
            result.history[0]["constraint_blocks"],
            ["geometry", "current"],
        )

    def test_minimize_alm_keeps_positional_snapshot_hook_compatibility(self):
        module = load_alm_utils_module()
        settings = module.ALMSettings(max_outer_iterations=1)

        def evaluate_problem(x, multipliers, penalty):
            del multipliers, penalty
            x = np.asarray(x, dtype=float)
            return _complete_alm_evaluation({
                "total": 0.0,
                "grad": np.zeros_like(x),
                "constraint_values": np.zeros(1),
                "stationarity_norm": 0.0,
            })

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
        self.assertEqual(snapshot_calls, ["snapshot"])
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

    def test_build_inner_options_rejects_noninteger_work_limits(self):
        module = load_alm_utils_module()
        profile = module._select_inner_solve_profile(
            trust_radius=0.05,
            continuation_iteration=1,
            feasible_enough=True,
        )
        field_to_error = {
            "maxiter": "inner_options.maxiter must be an integer",
            "maxls": "inner_options.maxls must be an integer",
            "maxfun": "inner_options.maxfun must be an integer",
        }

        for field, error in field_to_error.items():
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, error):
                    module._build_inner_options(
                        {field: 1.5},
                        update_stationarity_tol=1.0,
                        profile=profile,
                    )

    def test_build_inner_options_rejects_nonfinite_tolerances(self):
        module = load_alm_utils_module()
        profile = module._select_inner_solve_profile(
            trust_radius=None,
            continuation_iteration=0,
            feasible_enough=False,
        )
        field_to_error = {
            "ftol": "inner_options.ftol must be finite",
            "gtol": "inner_options.gtol must be finite",
        }

        for field, error in field_to_error.items():
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, error):
                    module._build_inner_options(
                        {field: float("nan")},
                        update_stationarity_tol=1.0,
                        profile=profile,
                    )

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

        current_eval = _complete_alm_evaluation({
            "total": 1.0e-3,
            "grad": np.array([0.1]),
            "constraint_values": np.array([0.0]),
            "stationarity_norm": 0.1,
        })
        candidate_eval = _complete_alm_evaluation({
            "total": 1.0005e-3,
            "grad": np.array([0.08]),
            "constraint_values": np.array([0.0]),
            "stationarity_norm": 0.08,
        })

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

        self.assertFalse(result.success)
        self.assertLessEqual(result.x[0], 1.0 + 1e-6)
        self.assertLessEqual(result.constraint_values[0], 1e-6)
        self.assertAlmostEqual(result.final_raw_stationarity_norm, 1.0)
        self.assertAlmostEqual(result.final_kkt_stationarity_norm, 0.0)

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
            return module.augmented_inequality_objective(
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
            return _complete_alm_evaluation({
                "total": value,
                "grad": 2.0 * np.asarray(x, dtype=float),
                "constraint_values": np.array([0.0]),
                "stationarity_norm": float(np.linalg.norm(2.0 * np.asarray(x, dtype=float))),
            })

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
            return _complete_alm_evaluation({
                "total": value,
                "grad": 2.0 * np.asarray(x, dtype=float),
                "constraint_values": np.array([0.0]),
                "stationarity_norm": float(np.linalg.norm(2.0 * np.asarray(x, dtype=float))),
            })

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
            return _complete_alm_evaluation({
                "total": 0.0,
                "grad": np.zeros(1),
                "constraint_values": np.array([3.0]),
                "feasibility_values": np.array([1.0]),
                "max_feasibility_violation": 1.0,
                "stationarity_norm": 0.0,
            })

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
        stationarity_norm, kkt_stationarity_norm, mismatch = module._stationarity_metrics(
            evaluation,
            routing_state,
            1.0e-8,
        )

        self.assertTrue(mismatch)
        self.assertIsNone(kkt_stationarity_norm)
        self.assertAlmostEqual(stationarity_norm, 1.0)

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
        # The terminal action is `signal_mismatch_penalty_increase`, so the
        # outer-cap reason carries that label (audit-v2 M10) rather than the
        # bare `max_outer` fallback.
        self.assertEqual(
            result.termination_reason,
            "max_outer_after_signal_mismatch_penalty_increase",
        )
        self.assertEqual(minimize_calls["count"], 4)
        self.assertEqual(result.history[0]["action"], "subproblem_continue")
        self.assertTrue(result.history[0]["signal_mismatch_active"])
        self.assertEqual(result.history[1]["action"], "signal_mismatch_penalty_increase")
        self.assertTrue(result.history[1]["signal_mismatch_active"])
        self.assertTrue(result.history[1]["hard_positive_shift_zero"])
        self.assertFalse(result.history[1]["surrogate_max_value"] <= 0.0)

    def test_relaxed_stage2_signal_mismatch_does_not_dual_update(self):
        module = load_alm_utils_module()
        settings = module.ALMSettings(
            max_outer_iterations=1,
            max_subproblem_continuations=1,
            max_inner_attempts=1,
            feasibility_tol=1.0e-6,
            stationarity_tol=1.0e-6,
            penalty_init=1.0,
            relaxed_feasibility_gate_cap=1.0e-2,
            history_max_entries=None,
        )

        def evaluate_problem(x, multipliers, penalty):
            del x, multipliers, penalty
            return self._stage2_signal_evaluation(
                constraint_values=np.array([1.0e-3]),
                dual_update_values=np.array([1.0e-3]),
                feasibility_values=np.array([1.0e-3]),
                hard_signed_constraint_values=np.array([-1.0e-3]),
                hard_violation_values=np.array([1.0e-3]),
                surrogate_signed_constraint_values=np.array([1.0e-3]),
                hard_dual_update_values=np.array([1.0e-3]),
                constraint_grads=[np.array([0.0])],
                constraint_activity_tolerances=np.array([0.0]),
                stationarity_norm=0.0,
            )

        def fake_minimize(fun, x, jac, method, bounds, callback, options):
            del jac, method, bounds, callback, options
            x_array = np.asarray(x, dtype=float)
            fun(x_array)
            return SimpleNamespace(
                x=x_array.copy(),
                nit=0,
                success=True,
                message="CONVERGENCE",
            )

        with patch.object(module, "minimize", side_effect=fake_minimize):
            result = module.minimize_alm(
                np.array([0.0]),
                ["demo_constraint"],
                evaluate_problem,
                settings,
                {"maxiter": 1},
            )

        self.assertFalse(result.success)
        self.assertEqual(result.history[0]["action"], "signal_mismatch_penalty_increase")
        self.assertTrue(result.history[0]["signal_mismatch_active"])
        np.testing.assert_allclose(result.history[0]["post_update_multipliers"], [0.0])
        np.testing.assert_allclose(result.multipliers, np.array([0.0]))

    def test_alm_terminates_deterministically_under_sustained_signal_mismatch(self):
        """Pin the deterministic-termination property of the hybrid signal contract.

        Contract: when the hard activity mask disagrees with the surrogate
        activity mask on every iteration, ``minimize_alm`` must terminate
        within the outer iteration cap, must label the result as a failure
        (``result.success is False``), and must produce the same
        ``termination_reason`` and history action sequence across re-runs of
        the same fixture. See ``docs/alm_hybrid_signal_contract_2026-05-08.md``
        for the contract this test pins.
        """
        module = load_alm_utils_module()

        def _build_settings():
            return module.ALMSettings(
                max_outer_iterations=5,
                max_subproblem_continuations=2,
                trust_radius_init=0.1,
                trust_radius_min=0.01,
                trust_radius_shrink=0.5,
                trust_radius_grow=1.5,
                max_inner_attempts=1,
                feasibility_tol=1.0e-8,
                stationarity_tol=1.0e-8,
            )

        def _build_evaluator():
            def evaluate_problem(x, multipliers, penalty):
                del multipliers, penalty
                point = float(np.asarray(x, dtype=float)[0])
                # Sustained mismatch: hard says feasible/inactive
                # (-0.01 < -tol_band 1e-3 ⇒ mask False), surrogate says
                # active and violating (0.2 ≥ -tol_band ⇒ mask True). Both
                # masks disagree on every iteration regardless of the
                # iterate, so signal_mismatch_active is True throughout.
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

            return evaluate_problem

        def _build_fake_minimize():
            call_count = {"count": 0}

            def fake_minimize(fun, x, jac, method, bounds, callback, options):
                del fun, jac, method, bounds, callback, options
                call_count["count"] += 1
                if call_count["count"] == 1:
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

            return fake_minimize

        def _run_once():
            with patch.object(module, "minimize", side_effect=_build_fake_minimize()):
                return module.minimize_alm(
                    np.array([0.0]),
                    ["demo_constraint"],
                    _build_evaluator(),
                    _build_settings(),
                    {"maxiter": 5, "ftol": 1e-12, "gtol": 1e-12},
                )

        first_result = _run_once()
        second_result = _run_once()

        # Termination is bounded by the outer iteration cap.
        self.assertLessEqual(first_result.outer_iterations, 5)
        self.assertLessEqual(second_result.outer_iterations, 5)

        # Mismatch blocks the success label on both runs.
        self.assertFalse(first_result.success)
        self.assertFalse(second_result.success)

        # Mismatch was actually present throughout (no false negative
        # for the property being pinned).
        self.assertTrue(
            all(entry["signal_mismatch_active"] for entry in first_result.history)
        )

        # Termination reason is stable across re-runs.
        self.assertEqual(
            first_result.termination_reason,
            second_result.termination_reason,
        )

        # Action sequence is identical across re-runs (no chatter between
        # subproblem_continue / signal_mismatch_penalty_increase /
        # signal_mismatch_stall arms under the same fixture).
        first_actions = [entry["action"] for entry in first_result.history]
        second_actions = [entry["action"] for entry in second_result.history]
        self.assertEqual(first_actions, second_actions)

        # Outer-iteration counts are identical across re-runs.
        self.assertEqual(
            first_result.outer_iterations,
            second_result.outer_iterations,
        )

        # No success arm was reached: every action must come from the
        # mismatch-or-continuation set, not from the converged or
        # constraints_inactive_converged arms.
        forbidden_actions = {
            "converged",
            "constraints_inactive_converged",
        }
        for action in first_actions:
            self.assertNotIn(action, forbidden_actions)

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
            return _complete_alm_evaluation({
                "total": 0.0,
                "grad": np.zeros(1),
                "constraint_values": np.array([2.0]),
                "stationarity_norm": 0.0,
            })

        result = module.minimize_alm(
            np.array([0.0]),
            ["demo_constraint"],
            evaluate_problem,
            settings,
            {"maxiter": 5, "ftol": 1e-12, "gtol": 1e-12},
        )

        self.assertFalse(result.success)
        self.assertEqual(result.penalty, 100.0)
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
            return _complete_alm_evaluation({
                "total": 0.0,
                "grad": np.zeros(1),
                "constraint_values": np.array([2.0]),
                "stationarity_norm": 0.0,
            })

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
            return _complete_alm_evaluation({
                "total": 1.0,
                "grad": np.zeros_like(x),
                "constraint_values": np.array([2.0]),
                "stationarity_norm": 0.0,
            })

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
        # L2: callback receives a defensive copy of the history list (not
        # the live ALM-internal list). The outer list identity must
        # differ; entries within remain shared references and may have
        # been mutated by ALM after the snapshot was taken (e.g., the
        # final action annotation), so a per-iteration value-equality
        # check against `result.history` is not appropriate. The length
        # at callback time is at most the final length.
        self.assertIsNot(history_snapshots[-1]["history"], result.history)
        self.assertLessEqual(
            len(history_snapshots[-1]["history"]), len(result.history)
        )
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

    def test_nonfinite_evaluation_fields_detects_stage2_signal_arrays(self):
        # H2: explicit stage-2 signal arrays must be in the
        # `_nonfinite_evaluation_fields` whitelist so a NaN in any of them is
        # caught at the `_require_finite_evaluation` boundary, not silently
        # propagated through the dual update via `_extract_stage2_constraint_signal_state`.
        module = load_alm_utils_module()
        base = {"total": 0.0, "grad": np.zeros(1)}
        for kind, sentinel in (("nan", np.nan), ("inf", np.inf)):
            for field in module._STAGE2_SIGNAL_ARRAY_FIELDS:
                with self.subTest(field=field, kind=kind):
                    evaluation = dict(base)
                    evaluation[field] = np.array([sentinel])
                    self.assertIn(
                        field,
                        module._nonfinite_evaluation_fields(evaluation),
                    )

        # Finite stage-2 fields are not flagged.
        finite = dict(base)
        for field in module._STAGE2_SIGNAL_ARRAY_FIELDS:
            finite[field] = np.array([0.0])
        self.assertEqual(module._nonfinite_evaluation_fields(finite), ())

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
        self.assertIsNot(sanitized["constraint_values"], fallback_evaluation["constraint_values"])
        np.testing.assert_allclose(
            sanitized["constraint_values"],
            fallback_evaluation["constraint_values"],
        )
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
            return _complete_alm_evaluation({
                "total": 3.778792e-03,
                "grad": np.zeros(1),
                "constraint_values": np.array([2.461054010712543e-2]),
                "stationarity_norm": 2.1899001642505436,
            })

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
            return _complete_alm_evaluation({
                "total": float(np.dot(x, x) + np.dot(multipliers, np.array([5.0e-3]))),
                "grad": 2.0 * x,
                "constraint_values": np.array([5.0e-3]),
                "stationarity_norm": float(np.linalg.norm(2.0 * x)),
            })

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

    def test_minimize_alm_uses_current_progress_for_three_dual_updates_before_penalty(self):
        module = load_alm_utils_module()
        settings = module.ALMSettings(
            max_outer_iterations=4,
            max_subproblem_continuations=1,
            penalty_init=1.0,
            penalty_scale=10.0,
            feasibility_tol=1.0e-8,
            stationarity_tol=1.0e-8,
        )

        def evaluate_problem(x, multipliers, penalty):
            del x, penalty
            multiplier = float(multipliers[0])
            if multiplier < 0.009:
                constraint_value = 1.0e-2
                stationarity_norm = 0.2
            elif multiplier < 0.019:
                constraint_value = 1.0e-2
                stationarity_norm = 0.02
            elif multiplier < 0.029:
                constraint_value = 1.0e-2
                stationarity_norm = 0.002
            else:
                constraint_value = 2.0e-2
                stationarity_norm = 0.0002
            return _complete_alm_evaluation({
                "total": 1.0,
                "grad": np.array([stationarity_norm]),
                "constraint_values": np.array([constraint_value]),
                "stationarity_norm": stationarity_norm,
            })

        def fake_minimize(fun, x, jac, method, bounds, callback, options):
            del fun, jac, method, bounds, callback, options
            return SimpleNamespace(
                x=np.asarray(x, dtype=float) + 1.0e-4,
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
                {"maxiter": 5, "ftol": 1.0e-12, "gtol": 1.0e-12},
            )

        self.assertFalse(result.success)
        self.assertEqual(
            [entry["action"] for entry in result.history],
            ["dual_update", "dual_update", "dual_update", "penalty_increase"],
        )
        self.assertEqual(result.history[-1]["outer_termination"], "max_outer")

    def test_minimize_alm_tolerances_nonincreasing_across_dual_penalty_dual(self):
        module = load_alm_utils_module()
        settings = module.ALMSettings(
            max_outer_iterations=3,
            max_subproblem_continuations=1,
            penalty_init=1.0,
            penalty_scale=10.0,
            feasibility_tol=1.0e-8,
            stationarity_tol=1.0e-8,
        )

        def evaluate_problem(x, multipliers, penalty):
            del x
            if float(penalty) <= 1.0 and float(multipliers[0]) <= 0.0:
                constraint_value = 1.0e-2
                stationarity_norm = 0.2
            elif float(penalty) <= 1.0:
                constraint_value = 0.2
                stationarity_norm = 0.2
            else:
                constraint_value = 1.0e-2
                stationarity_norm = 0.05
            return _complete_alm_evaluation({
                "total": 1.0,
                "grad": np.array([stationarity_norm]),
                "constraint_values": np.array([constraint_value]),
                "stationarity_norm": stationarity_norm,
            })

        def fake_minimize(fun, x, jac, method, bounds, callback, options):
            del fun, jac, method, bounds, callback, options
            return SimpleNamespace(
                x=np.asarray(x, dtype=float) + 1.0e-4,
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
                {"maxiter": 5, "ftol": 1.0e-12, "gtol": 1.0e-12},
            )

        self.assertFalse(result.success)
        self.assertEqual(
            [entry["action"] for entry in result.history],
            ["dual_update", "penalty_increase", "dual_update"],
        )
        for field in ("feasibility_tolerance", "stationarity_tolerance"):
            values = [entry[field] for entry in result.history]
            self.assertEqual(values, sorted(values, reverse=True))

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
            return _complete_alm_evaluation({
                "total": float(np.dot(x, x)),
                "grad": 2.0 * x,
                "constraint_values": np.array([0.5]),
                "stationarity_norm": float(np.linalg.norm(2.0 * x)),
            })

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
            return _complete_alm_evaluation({
                "total": float(np.dot(x, x)),
                "grad": 2.0 * x,
                "constraint_values": np.array([normalized_violation]),
                "normalized_signed_constraint_values": np.array([normalized_violation]),
                "normalized_feasibility_values": np.array([normalized_violation]),
                "raw_dual_update_values": np.array([raw_violation]),
                "raw_hard_violation_values": np.array([raw_violation]),
                "constraint_scales": np.array([scale]),
                "stationarity_norm": float(np.linalg.norm(2.0 * x)),
            })

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

    def test_minimize_alm_result_preserves_raw_signed_constraint_values(self):
        module = load_alm_utils_module()
        settings = module.ALMSettings(max_outer_iterations=1)
        raw_signed = -2.0
        raw_dual_update = -1.25
        scale = 5.0
        normalized_signed = raw_signed / scale

        def evaluate_problem(x, multipliers, penalty):
            del x, multipliers, penalty
            return {
                "total": 0.0,
                "grad": np.zeros(1),
                "constraint_values": np.array([normalized_signed]),
                "dual_update_values": np.array([raw_dual_update / scale]),
                "feasibility_values": np.array([0.0]),
                "constraint_grads": [np.zeros(1)],
                "normalized_signed_constraint_values": np.array([normalized_signed]),
                "normalized_feasibility_values": np.array([0.0]),
                "raw_constraint_values": np.array([raw_signed]),
                "raw_dual_update_values": np.array([raw_dual_update]),
                "raw_feasibility_values": np.array([0.0]),
                "raw_surrogate_signed_constraint_values": np.array([raw_signed]),
                "constraint_scales": np.array([scale]),
                "stationarity_norm": 0.0,
            }

        result = module.minimize_alm(
            np.zeros(1),
            ["signed_gap"],
            evaluate_problem,
            settings,
            {"maxiter": 1},
        )

        np.testing.assert_allclose(result.raw_constraint_values, [raw_signed])
        np.testing.assert_allclose(result.raw_solver_constraint_values, [raw_signed])
        np.testing.assert_allclose(result.raw_dual_update_values, [raw_dual_update])
        np.testing.assert_allclose(
            result.normalized_constraint_values,
            [normalized_signed],
        )
        np.testing.assert_allclose(result.normalized_feasibility_values, [0.0])

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
            return _complete_alm_evaluation({
                "total": float(np.dot(x, x)),
                "grad": 2.0 * x,
                "constraint_values": np.array([0.5]),
                "stationarity_norm": float(np.linalg.norm(2.0 * x)),
            })

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
            return _complete_alm_evaluation({
                "total": 0.0,
                "grad": np.array([0.0]),
                "constraint_values": constraint_values,
                "stationarity_norm": 0.0,
            })

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
            return _complete_alm_evaluation({
                "total": float(np.dot(x, x)),
                "grad": np.zeros_like(x),
                "constraint_values": np.linspace(1.0e-3, 1.2e-2, len(constraint_names)),
                "stationarity_norm": 0.0,
            })

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
            return _complete_alm_evaluation({
                "total": 0.0,
                "grad": np.array([0.2]),
                "constraint_values": np.array([2.5e-2]),
                "stationarity_norm": 0.2,
            })

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
        self.assertEqual(result.penalty, 10.0)
        self.assertEqual(result.termination_reason, "max_outer_after_infeasible_stall")
        self.assertEqual(result.history[0]["action"], "infeasible_stall_penalty_increase")
        self.assertEqual(result.history[0]["inner_message"], "STOP: plateau")
        self.assertEqual(result.history[0]["inner_profile"], "boxed_infeasible_initial")
        self.assertAlmostEqual(result.history[0]["feasibility_tolerance"], 0.1)
        self.assertAlmostEqual(result.history[0]["effective_feasibility_tolerance"], 1.0e-2)
        self.assertEqual(result.history[0]["outer_termination"], "max_outer")

    def test_post_inner_history_diagnostics_use_effective_feasibility_gate(self):
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
        captured_feasibility_gates = []
        original_attach = module._attach_alm_history_diagnostics

        def capture_attach(*args, **kwargs):
            captured_feasibility_gates.append(float(args[-1]))
            return original_attach(*args, **kwargs)

        def evaluate_problem(x, multipliers, penalty):
            del x, multipliers, penalty
            return _complete_alm_evaluation({
                "total": 0.0,
                "grad": np.array([0.2]),
                "constraint_values": np.array([2.5e-2]),
                "stationarity_norm": 0.2,
            })

        def fake_minimize(fun, x, jac, method, bounds, callback, options):
            del fun, jac, method, bounds, callback, options
            return SimpleNamespace(
                x=np.asarray(x, dtype=float),
                nit=1,
                success=False,
                message="STOP: plateau",
            )

        with patch.object(
            module,
            "_attach_alm_history_diagnostics",
            side_effect=capture_attach,
        ), patch.object(module, "minimize", side_effect=fake_minimize):
            module.minimize_alm(
                np.array([0.0]),
                ["demo_constraint"],
                evaluate_problem,
                settings,
                {"maxiter": 30, "ftol": 1e-12, "gtol": 1e-12},
            )

        self.assertGreaterEqual(len(captured_feasibility_gates), 1)
        self.assertAlmostEqual(captured_feasibility_gates[0], 1.0e-2)

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
            return _complete_alm_evaluation({
                "total": 0.0,
                "grad": np.array([0.02]),
                "constraint_values": np.array([0.0]),
                "stationarity_norm": 0.02,
            })

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
            return _complete_alm_evaluation({
                "total": 0.0,
                "grad": np.array([0.02]),
                "constraint_values": np.array([0.0]),
                "stationarity_norm": 0.02,
            })

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
            return _complete_alm_evaluation({
                "total": 0.0,
                "grad": np.array([0.02]),
                "constraint_values": np.array([0.0]),
                "stationarity_norm": 0.02,
            })

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
                return _complete_alm_evaluation({
                    "total": float(np.dot(x, x)),
                    "grad": 2.0 * x,
                    "constraint_values": np.array([0.01]),
                    "stationarity_norm": float(np.linalg.norm(2.0 * x)),
                })
            return _complete_alm_evaluation({
                "total": 0.0,
                "grad": np.array([0.02]),
                "constraint_values": np.array([0.0]),
                "stationarity_norm": 0.02,
            })

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
                return _complete_alm_evaluation({
                    "total": float(np.dot(x, x)),
                    "grad": 2.0 * x,
                    "constraint_values": np.array([0.01]),
                    "stationarity_norm": float(np.linalg.norm(2.0 * x)),
                })
            return _complete_alm_evaluation({
                "total": 0.0,
                "grad": np.array([0.02]),
                "constraint_values": np.array([0.0]),
                "stationarity_norm": 0.02,
            })

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
        self.assertEqual(minimize_calls["count"], 4)
        self.assertEqual(result.outer_iterations, 3)
        self.assertEqual(result.termination_reason, "plateau_stall")
        self.assertEqual(result.history[0]["action"], "dual_update")
        self.assertEqual(result.history[1]["action"], "dual_update")
        self.assertEqual(result.history[-2]["action"], "subproblem_continue")
        self.assertEqual(result.history[-1]["action"], "subproblem_limit")
        self.assertEqual(result.history[-1]["subproblem_limit_reason"], "plateau_stall")

    def test_minimize_alm_stops_on_long_feasible_no_progress_plateau(self):
        module = load_alm_utils_module()
        settings = module.ALMSettings(
            max_outer_iterations=20,
            max_subproblem_continuations=20,
            penalty_init=1.0,
            penalty_scale=10.0,
            feasibility_tol=1.0e-8,
            stationarity_tol=1.0e-8,
        )
        minimize_calls = {"count": 0}

        def evaluate_problem(x, multipliers, penalty):
            del x, multipliers, penalty
            return _complete_alm_evaluation({
                "total": 0.0,
                "grad": np.array([2.0]),
                "constraint_values": np.array([0.0]),
                "stationarity_norm": 2.0,
            })

        def fake_minimize(fun, x, jac, method, bounds, callback, options):
            del fun, jac, method, bounds, callback, options
            minimize_calls["count"] += 1
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
                {"maxiter": 5, "ftol": 1.0e-12, "gtol": 1.0e-12},
            )

        self.assertFalse(result.success)
        self.assertEqual(result.termination_reason, "plateau_stall")
        self.assertLess(result.outer_iterations, settings.max_outer_iterations)
        self.assertEqual(minimize_calls["count"], 2)
        self.assertEqual(
            [entry["action"] for entry in result.history],
            ["subproblem_continue", "subproblem_limit"],
        )
        self.assertEqual(result.history[-1]["subproblem_limit_reason"], "plateau_stall")

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
            return _complete_alm_evaluation({
                "total": 0.0,
                "grad": np.array([2.0]),
                "constraint_values": np.array([0.25]),
                "stationarity_norm": 2.0,
            })

        result = module.minimize_alm(
            np.array([0.0]),
            ["demo_constraint"],
            evaluate_problem,
            settings,
            {"maxiter": 5, "ftol": 1e-12, "gtol": 1e-12},
        )

        self.assertFalse(result.success)
        self.assertEqual(result.penalty, 100.0)
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

        self.assertFalse(result.success)
        self.assertEqual(result.termination_reason, "max_outer_after_dual_update")
        self.assertEqual(result.history[0]["action"], "dual_update")
        self.assertAlmostEqual(result.history[0]["raw_stationarity_norm"], 1.0)
        self.assertAlmostEqual(result.history[0]["kkt_stationarity_norm"], 0.0)
        self.assertAlmostEqual(result.history[0]["stationarity_norm"], 1.0)

    def test_minimize_alm_restores_best_feasible_incumbent_by_base_objective(self):
        module = load_alm_utils_module()
        settings = module.ALMSettings(
            max_outer_iterations=3,
            max_subproblem_continuations=1,
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
                return _complete_alm_evaluation({
                    "total": 20.0,
                    "base_value": 20.0,
                    "grad": np.array([0.5]),
                    "constraint_values": np.array([0.5]),
                    "stationarity_norm": 0.5,
                })
            if point < 1.5:
                return _complete_alm_evaluation({
                    "total": 10.0,
                    "base_value": 10.0,
                    "grad": np.zeros(1),
                    "constraint_values": np.array([0.5]),
                    "stationarity_norm": 0.0,
                })
            if point < 2.5:
                return _complete_alm_evaluation({
                    "total": 1.0,
                    "base_value": 1.0,
                    "grad": np.array([0.05]),
                    "constraint_values": np.array([0.0]),
                    "stationarity_norm": 0.05,
                })
            return _complete_alm_evaluation({
                "total": 0.5,
                "base_value": 5.0,
                "grad": np.array([0.2]),
                "constraint_values": np.array([0.0]),
                "stationarity_norm": 0.2,
            })

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
        # M5: post-inner routing now uses the clamped effective_feasibility_tol
        # consistent with pre-inner; the third outer iteration adds one
        # subproblem_continue before the subproblem_limit fires. Final
        # best-feasible-restoration outcome is unchanged.
        self.assertEqual(minimize_calls["count"], 4)
        self.assertEqual(result.history[0]["action"], "penalty_increase")
        self.assertEqual(result.history[1]["action"], "dual_update")
        self.assertEqual(result.history[2]["action"], "subproblem_continue")
        # M3.a: max_subproblem_continuations exhaustion now routes through the
        # penalty-increase arm; the action label and outer_termination
        # annotation come from `_emit_alm_penalty_increase_arm`.
        self.assertEqual(
            result.history[3]["action"], "subproblem_limit_penalty_increase"
        )
        self.assertEqual(
            result.history[3]["subproblem_limit_reason"],
            "max_subproblem_continuations",
        )
        self.assertEqual(result.history[3]["outer_termination"], "max_outer")

    def test_minimize_alm_restores_best_feasible_solver_owned_incumbent_state(self):
        module = load_alm_utils_module()
        settings = module.ALMSettings(
            max_outer_iterations=3,
            max_subproblem_continuations=1,
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
                return _complete_alm_evaluation({
                    "total": 20.0,
                    "base_value": 20.0,
                    "grad": np.array([0.5]),
                    "constraint_values": np.array([0.5]),
                    "stationarity_norm": 0.5,
                })
            if point < 1.5:
                return _complete_alm_evaluation({
                    "total": 10.0,
                    "base_value": 10.0,
                    "grad": np.zeros(1),
                    "constraint_values": np.array([0.5]),
                    "stationarity_norm": 0.0,
                })
            if point < 2.5:
                return _complete_alm_evaluation({
                    "total": 1.0,
                    "base_value": 1.0,
                    "grad": np.array([0.05]),
                    "constraint_values": np.array([0.0]),
                    "stationarity_norm": 0.05,
                })
            return _complete_alm_evaluation({
                "total": 0.5,
                "base_value": 5.0,
                "grad": np.array([0.2]),
                "constraint_values": np.array([0.0]),
                "stationarity_norm": 0.2,
            })

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
        self.assertAlmostEqual(result.history[0]["raw_stationarity_norm"], 1.0)
        self.assertAlmostEqual(result.history[0]["kkt_stationarity_norm"], 0.0)
        self.assertAlmostEqual(result.history[0]["stationarity_norm"], 1.0)

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
        self.assertEqual(result.history[0]["inner_iterations"], 0)
        self.assertEqual(result.history[0]["action"], "dual_update")
        self.assertAlmostEqual(result.history[0]["stationarity_norm"], 1.0)

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
            return _complete_alm_evaluation({
                "total": 0.5 * float(np.dot(x, x)),
                "grad": x.copy(),
                "constraint_values": np.zeros(1),
                "stationarity_norm": float(np.linalg.norm(x)),
            })

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
                {"maxiter": 5, "ftol": 1e-12, "gtol": 1e-12},
            )

        self.assertFalse(result.success)
        self.assertAlmostEqual(result.history[0]["raw_stationarity_norm"], 10.0)
        self.assertAlmostEqual(result.history[0]["stationarity_norm"], 10.0)

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
            return _complete_alm_evaluation({
                "total": 1.0,
                "base_value": 0.1,
                "base_grad": np.array([0.0]),
                "grad": np.array([0.0]),
                "constraint_values": np.array([0.5]),
                "feasibility_values": np.array([0.5]),
                "dual_update_values": np.array([0.5]),
                "stationarity_norm": 0.0,
            })

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

    def test_minimize_alm_preserves_user_gtol_anchor_across_outer_iterations(self):
        # M3.b regression: the user's `gtol` from inner_options must remain the
        # base of `_build_inner_options`'s `max(base_gtol, staged_gtol)` rule
        # across every outer iteration. A previous fix wrongly stripped gtol
        # from persisted state, causing base to fall back to the 1e-12 default
        # and forfeiting any user-supplied looser tolerance.
        module = load_alm_utils_module()
        settings = module.ALMSettings(
            max_outer_iterations=3,
            max_subproblem_continuations=1,
            penalty_init=1.0,
            penalty_scale=10.0,
            feasibility_tol=1e-8,
            stationarity_tol=1e-8,
        )
        gtol_user = 1.0e-3  # significantly looser than the 1e-12 default
        observed_gtols: list[float] = []

        def evaluate_problem(x, multipliers, penalty):
            return _complete_alm_evaluation({
                "total": 1.0,
                "base_value": 1.0,
                "grad": np.array([0.5]),
                "constraint_values": np.array([0.5]),
                "stationarity_norm": 0.5,
            })

        def fake_minimize(fun, x, jac, method, bounds, callback, options):
            observed_gtols.append(float(options["gtol"]))
            return SimpleNamespace(
                x=np.asarray(x, dtype=float),
                nit=1,
                success=True,
                message="CONVERGENCE",
            )

        with patch.object(module, "minimize", side_effect=fake_minimize):
            module.minimize_alm(
                np.array([0.0]),
                ["demo_constraint"],
                evaluate_problem,
                settings,
                {"maxiter": 5, "ftol": 1e-12, "gtol": gtol_user},
            )

        # Every observed inner-solve gtol must be at least the user's anchor.
        # `staged_gtol` may exceed it (looser still), but it must never drop
        # below — that would mean the schedule re-derived against a stripped
        # default rather than the user's input.
        self.assertGreater(len(observed_gtols), 1)
        for gtol_observed in observed_gtols:
            self.assertGreaterEqual(gtol_observed, gtol_user - 1e-15)

    def test_minimize_alm_converges_to_kkt_on_active_linear_inequality(self):
        # T1.a: gold-standard end-to-end KKT integration with corrected math.
        # min  f(x,y) = 0.5*((x-1)^2 + (y-1)^2)
        # s.t. c(x,y) = x + y - 1 <= 0
        # Active at minimizer (0.5, 0.5) with Lagrange multiplier lambda = 0.5.
        module = load_alm_utils_module()
        settings = module.ALMSettings(
            max_outer_iterations=20,
            max_subproblem_continuations=8,
            penalty_init=1.0,
            penalty_scale=10.0,
            feasibility_tol=1e-8,
            stationarity_tol=1e-8,
        )

        def evaluate_problem(x, multipliers, penalty):
            x_arr = np.asarray(x, dtype=float)
            base_value = 0.5 * float((x_arr[0] - 1.0) ** 2 + (x_arr[1] - 1.0) ** 2)
            base_grad = np.array([x_arr[0] - 1.0, x_arr[1] - 1.0], dtype=float)
            constraint_values = np.array([x_arr[0] + x_arr[1] - 1.0], dtype=float)
            constraint_grads = [np.array([1.0, 1.0], dtype=float)]
            penalty_argument = float(penalty)
            multipliers_arr = np.asarray(multipliers, dtype=float)
            augmented_term = (
                multipliers_arr[0] * max(constraint_values[0], 0.0)
                + 0.5 * penalty_argument * max(constraint_values[0], 0.0) ** 2
            )
            shift = max(0.0, multipliers_arr[0] / max(penalty_argument, 1e-12))
            shifted = constraint_values[0] + shift
            if shifted > 0.0:
                augmented_grad = (multipliers_arr[0] + penalty_argument * constraint_values[0]) * constraint_grads[0]
            else:
                augmented_grad = np.zeros(2, dtype=float)
            total = base_value + augmented_term
            grad = base_grad + augmented_grad
            stationarity_norm = float(np.linalg.norm(grad))
            return _complete_alm_evaluation({
                "total": float(total),
                "base_value": float(base_value),
                "base_grad": base_grad,
                "grad": grad,
                "constraint_values": constraint_values,
                "constraint_grads": constraint_grads,
                "stationarity_norm": stationarity_norm,
            })

        result = module.minimize_alm(
            np.array([0.0, 0.0]),
            ["xy_sum_upper_bound"],
            evaluate_problem,
            settings,
            {"maxiter": 200, "ftol": 1e-15, "gtol": 1e-15},
        )

        self.assertTrue(result.success)
        self.assertEqual(result.termination_reason, "converged")
        np.testing.assert_allclose(result.x, np.array([0.5, 0.5]), atol=1e-5)
        # KKT multiplier: lambda = 0.5 (NOT 1.0 — see FIX_PLAN.md T1.a math).
        np.testing.assert_allclose(
            np.asarray(result.multipliers, dtype=float),
            np.array([0.5]),
            atol=1e-5,
        )
        # Inequality multipliers must be non-negative.
        self.assertGreaterEqual(float(result.multipliers[0]), 0.0)
        # Constraint feasibility within tol.
        final_eval = evaluate_problem(result.x, result.multipliers, result.penalty)
        self.assertLessEqual(
            float(np.max(final_eval["constraint_values"])),
            settings.feasibility_tol + 1e-9,
        )


class AlmContinuationStepTests(unittest.TestCase):
    @staticmethod
    def _settings(module, **overrides):
        defaults = dict(
            max_outer_iterations=2,
            penalty_init=1.0,
            penalty_scale=10.0,
            feasibility_tol=1e-8,
            stationarity_tol=1e-8,
        )
        defaults.update(overrides)
        return module.ALMSettings(**defaults)

    @staticmethod
    def _run_state(module, x):
        return module.ALMRunState(
            x=np.asarray(x, dtype=float).copy(),
            total_inner_iterations=0,
            trust_radius=None,
            history=[],
            history_truncated_count=0,
            cap_binding_detected=False,
            cap_binding_indices=set(),
            penalty_cap_reached=False,
            penalty_cap_requested=None,
        )

    @staticmethod
    def _kwargs(module, run_state, evaluate_problem, **overrides):
        kwargs = dict(
            settings=overrides.pop("settings"),
            run_state=run_state,
            multipliers=overrides.pop("multipliers", np.array([0.0])),
            penalty=overrides.pop("penalty", 1.0),
            update_feasibility_tol=overrides.pop("update_feasibility_tol", 1e-2),
            update_stationarity_tol=overrides.pop("update_stationarity_tol", 1e-2),
            feasible_stall_count=overrides.pop("feasible_stall_count", 0),
            last_result=overrides.pop("last_result", None),
            final_eval=overrides.pop("final_eval", None),
            final_multipliers=overrides.pop("final_multipliers", np.array([0.0])),
            final_penalty=overrides.pop("final_penalty", 1.0),
            best_feasible=overrides.pop("best_feasible", None),
            inner_options=overrides.pop("inner_options", {"maxiter": 50}),
            outer_iteration=overrides.pop("outer_iteration", 1),
            continuation_iteration=overrides.pop("continuation_iteration", 0),
            is_final_outer=overrides.pop("is_final_outer", False),
            evaluate_problem=evaluate_problem,
            inner_callback=overrides.pop("inner_callback", None),
            accepted_callback=overrides.pop("accepted_callback", None),
            history_callback=overrides.pop("history_callback", None),
            snapshot_accepted_state_fn=overrides.pop("snapshot_accepted_state_fn", None),
            restore_incumbent_state_fn=overrides.pop("restore_incumbent_state_fn", None),
            constraint_names=overrides.pop("constraint_names", ["upper"]),
            constraint_names_tuple=overrides.pop("constraint_names_tuple", ("upper",)),
            constraint_blocks_tuple=overrides.pop("constraint_blocks_tuple", None),
        )
        if overrides:
            raise TypeError(f"unexpected kwargs: {sorted(overrides)}")
        return kwargs

    def test_continuation_step_short_circuits_when_iterate_is_already_converged(self):
        module = load_alm_utils_module()
        settings = self._settings(module)

        def evaluate_problem(x, multipliers, penalty):
            value = 0.5 * (x[0] - 2.0) ** 2
            grad = np.array([x[0] - 2.0])
            signed = np.array([x[0] - 1.0])
            return module.augmented_inequality_objective(
                value, grad, signed, [np.array([1.0])], multipliers, penalty
            )

        run_state = self._run_state(module, np.array([1.0]))
        result = module._run_alm_continuation_step(
            **self._kwargs(
                module,
                run_state,
                evaluate_problem,
                settings=settings,
                multipliers=np.array([1.0]),
                final_multipliers=np.array([1.0]),
                update_feasibility_tol=settings.feasibility_tol,
                update_stationarity_tol=settings.stationarity_tol,
            )
        )

        self.assertEqual(result.decision, module._ALMContinuationDecision.RETURN)
        self.assertIsNotNone(result.result)
        self.assertEqual(result.result.termination_reason, "converged")
        self.assertTrue(result.result.success)
        self.assertEqual(len(run_state.history), 1)
        self.assertEqual(run_state.history[0]["action"], "converged")

    def test_continuation_step_breaks_outer_after_infeasible_stall_penalty_increase(self):
        module = load_alm_utils_module()
        settings = self._settings(module)

        def evaluate_problem(x, multipliers, penalty):
            value = 0.5 * (x[0] - 2.0) ** 2
            grad = np.array([x[0] - 2.0])
            signed = np.array([x[0] - 1.0])
            return module.augmented_inequality_objective(
                value, grad, signed, [np.array([1.0])], multipliers, penalty
            )

        run_state = self._run_state(module, np.array([2.0]))
        with patch.object(module, "minimize", _make_fake_minimize()):
            result = module._run_alm_continuation_step(
                **self._kwargs(
                    module,
                    run_state,
                    evaluate_problem,
                    settings=settings,
                    inner_options={"maxiter": 1, "ftol": 1e-12, "gtol": 1e-12},
                )
            )

        self.assertEqual(result.decision, module._ALMContinuationDecision.BREAK_OUTER)
        self.assertIsNone(result.result)
        self.assertEqual(result.penalty, settings.penalty_init * settings.penalty_scale)
        self.assertGreater(len(run_state.history), 0)
        self.assertEqual(
            run_state.history[0]["action"],
            "infeasible_stall_penalty_increase",
        )
        self.assertGreaterEqual(run_state.total_inner_iterations, 1)


class AlmOuterIterationTests(unittest.TestCase):
    @staticmethod
    def _settings(module, **overrides):
        defaults = dict(
            max_outer_iterations=2,
            penalty_init=1.0,
            penalty_scale=10.0,
            feasibility_tol=1e-8,
            stationarity_tol=1e-8,
            max_subproblem_continuations=3,
        )
        defaults.update(overrides)
        return module.ALMSettings(**defaults)

    @staticmethod
    def _run_state(module):
        return module.ALMRunState(
            x=np.zeros(1),
            total_inner_iterations=0,
            trust_radius=None,
            history=[],
            history_truncated_count=0,
            cap_binding_detected=False,
            cap_binding_indices=set(),
            penalty_cap_reached=False,
            penalty_cap_requested=None,
        )

    @staticmethod
    def _outer_kwargs(module, run_state, **overrides):
        kwargs = dict(
            settings=overrides.pop("settings"),
            run_state=run_state,
            multipliers=overrides.pop("multipliers", np.zeros(1)),
            penalty=overrides.pop("penalty", 1.0),
            update_feasibility_tol=overrides.pop("update_feasibility_tol", 1e-2),
            update_stationarity_tol=overrides.pop("update_stationarity_tol", 1e-2),
            last_result=overrides.pop("last_result", None),
            final_eval=overrides.pop("final_eval", None),
            final_multipliers=overrides.pop("final_multipliers", np.zeros(1)),
            final_penalty=overrides.pop("final_penalty", 1.0),
            best_feasible=overrides.pop("best_feasible", None),
            inner_options=overrides.pop("inner_options", {"maxiter": 1}),
            outer_iteration=overrides.pop("outer_iteration", 1),
            is_final_outer=overrides.pop("is_final_outer", False),
            evaluate_problem=overrides.pop(
                "evaluate_problem", lambda x, m, p: {"total": 0.0, "grad": np.zeros(1)}
            ),
            inner_callback=overrides.pop("inner_callback", None),
            accepted_callback=overrides.pop("accepted_callback", None),
            outer_state_callback=overrides.pop("outer_state_callback", None),
            history_callback=overrides.pop("history_callback", None),
            snapshot_accepted_state_fn=overrides.pop("snapshot_accepted_state_fn", None),
            restore_incumbent_state_fn=overrides.pop("restore_incumbent_state_fn", None),
            constraint_names=overrides.pop("constraint_names", ["c0"]),
            constraint_names_tuple=overrides.pop("constraint_names_tuple", ("c0",)),
            constraint_blocks_tuple=overrides.pop("constraint_blocks_tuple", None),
        )
        if overrides:
            raise TypeError(f"unexpected kwargs: {sorted(overrides)}")
        return kwargs

    def _make_step_result(self, module, decision, result_payload=None, **overrides):
        return module._ALMContinuationStepResult(
            decision=decision,
            result=result_payload,
            multipliers=overrides.get("multipliers", np.zeros(1)),
            penalty=overrides.get("penalty", 1.0),
            update_feasibility_tol=overrides.get("update_feasibility_tol", 1e-2),
            update_stationarity_tol=overrides.get("update_stationarity_tol", 1e-2),
            feasible_stall_count=overrides.get("feasible_stall_count", 0),
            last_result=overrides.get("last_result", None),
            final_eval=overrides.get("final_eval", None),
            final_multipliers=overrides.get("final_multipliers", np.zeros(1)),
            final_penalty=overrides.get("final_penalty", 1.0),
            best_feasible=overrides.get("best_feasible", None),
            inner_options=overrides.get("inner_options", {"maxiter": 1}),
        )

    def test_outer_iteration_propagates_return_decision_from_continuation_step(self):
        module = load_alm_utils_module()
        settings = self._settings(module)
        run_state = self._run_state(module)
        sentinel_payload = SimpleNamespace(termination_reason="converged")
        step_result = self._make_step_result(
            module,
            module._ALMContinuationDecision.RETURN,
            result_payload=sentinel_payload,
        )

        with patch.object(module, "_run_alm_continuation_step", return_value=step_result) as mocked:
            outcome = module._run_alm_outer_iteration(
                **self._outer_kwargs(module, run_state, settings=settings)
            )

        self.assertEqual(outcome.decision, module._ALMOuterDecision.RETURN)
        self.assertIs(outcome.result, sentinel_payload)
        self.assertEqual(mocked.call_count, 1)

    def test_outer_iteration_signals_exhaust_when_final_outer_and_no_return(self):
        module = load_alm_utils_module()
        settings = self._settings(module)
        run_state = self._run_state(module)
        step_result = self._make_step_result(
            module, module._ALMContinuationDecision.BREAK_OUTER
        )

        with patch.object(module, "_run_alm_continuation_step", return_value=step_result):
            outcome = module._run_alm_outer_iteration(
                **self._outer_kwargs(module, run_state, settings=settings, is_final_outer=True)
            )

        self.assertEqual(outcome.decision, module._ALMOuterDecision.EXHAUST)
        self.assertIsNone(outcome.result)

    def test_outer_iteration_signals_next_outer_when_not_final_and_no_return(self):
        module = load_alm_utils_module()
        settings = self._settings(module)
        run_state = self._run_state(module)
        step_result = self._make_step_result(
            module, module._ALMContinuationDecision.BREAK_OUTER
        )

        with patch.object(module, "_run_alm_continuation_step", return_value=step_result):
            outcome = module._run_alm_outer_iteration(
                **self._outer_kwargs(module, run_state, settings=settings, is_final_outer=False)
            )

        self.assertEqual(outcome.decision, module._ALMOuterDecision.NEXT_OUTER)

    def test_outer_iteration_emits_outer_state_callback_exactly_once_before_continuation(self):
        module = load_alm_utils_module()
        settings = self._settings(module)
        run_state = self._run_state(module)
        events = []

        def outer_state_callback(iteration, multipliers, penalty):
            events.append(("outer_state", iteration))

        def fake_step(**kwargs):
            events.append(("continuation", kwargs["continuation_iteration"]))
            return self._make_step_result(
                module, module._ALMContinuationDecision.BREAK_OUTER
            )

        with patch.object(module, "_run_alm_continuation_step", side_effect=fake_step):
            module._run_alm_outer_iteration(
                **self._outer_kwargs(
                    module,
                    run_state,
                    settings=settings,
                    outer_state_callback=outer_state_callback,
                )
            )

        self.assertEqual(events[0], ("outer_state", 1))
        self.assertEqual(
            sum(1 for tag, _ in events if tag == "outer_state"),
            1,
            f"outer_state_callback must fire exactly once: {events}",
        )

    def test_outer_iteration_dispatches_until_continuation_budget_exhausted(self):
        module = load_alm_utils_module()
        settings = self._settings(module, max_subproblem_continuations=2)
        run_state = self._run_state(module)

        def fake_step(**kwargs):
            return self._make_step_result(
                module, module._ALMContinuationDecision.CONTINUE_CONTINUATION
            )

        with patch.object(
            module, "_run_alm_continuation_step", side_effect=fake_step
        ) as mocked:
            outcome = module._run_alm_outer_iteration(
                **self._outer_kwargs(module, run_state, settings=settings)
            )

        self.assertEqual(outcome.decision, module._ALMOuterDecision.NEXT_OUTER)
        self.assertEqual(mocked.call_count, settings.max_subproblem_continuations + 1)


class ClassifyInfeasibleInnerStallTests(unittest.TestCase):
    """Direct-helper tests for `_classify_infeasible_inner_stall`.

    The integration regression at
    `MinimizeAlmTests.test_minimize_alm_classifies_relative_reduction_false_success_as_infeasible_stall`
    exercises this through the full ALM stack. These tests pin the helper's
    SciPy-message-text contract directly so the four classification arms stay
    locked even if a future refactor decouples them from the outer driver.
    The repo only pins `scipy>=1.5.4`, so the message-text match is the only
    cross-version contract we have for the false-success arm.
    """

    @staticmethod
    def _evals(current_total: float, candidate_total: float, current_violation: float, candidate_violation: float):
        current_eval = _complete_alm_evaluation({
            "total": float(current_total),
            "constraint_values": np.array([float(current_violation)]),
        })
        candidate_eval = _complete_alm_evaluation({
            "total": float(candidate_total),
            "constraint_values": np.array([float(candidate_violation)]),
        })
        return current_eval, candidate_eval

    def test_relative_reduction_of_f_with_no_feasibility_gain_is_false_success(self):
        module = load_alm_utils_module()
        current_eval, candidate_eval = self._evals(1.0, 1.0 - 1.0e-12, 2.0e-2, 2.0e-2)
        result = SimpleNamespace(
            success=True,
            message="CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH",
            nit=12,
        )

        infeasible_stall, false_success, reason = module._classify_infeasible_inner_stall(
            current_eval,
            candidate_eval,
            result,
            0.0,
            1.0e-6,
            1.0e-3,
        )

        self.assertTrue(infeasible_stall)
        self.assertTrue(false_success)
        self.assertEqual(reason, "relative_objective_termination_without_feasibility_gain")

    def test_other_success_message_is_generic_false_success(self):
        module = load_alm_utils_module()
        current_eval, candidate_eval = self._evals(1.0, 1.0 - 1.0e-12, 2.0e-2, 2.0e-2)
        result = SimpleNamespace(
            success=True,
            message="CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL",
            nit=12,
        )

        infeasible_stall, false_success, reason = module._classify_infeasible_inner_stall(
            current_eval,
            candidate_eval,
            result,
            0.0,
            1.0e-6,
            1.0e-3,
        )

        self.assertTrue(infeasible_stall)
        self.assertTrue(false_success)
        self.assertEqual(reason, "successful_inner_solve_without_feasibility_gain")

    def test_failed_inner_without_feasibility_gain_is_failed_stall(self):
        module = load_alm_utils_module()
        current_eval, candidate_eval = self._evals(1.0, 1.0 - 1.0e-12, 2.0e-2, 2.0e-2)
        result = SimpleNamespace(success=False, message="ABNORMAL TERMINATION", nit=5)

        infeasible_stall, false_success, reason = module._classify_infeasible_inner_stall(
            current_eval,
            candidate_eval,
            result,
            0.0,
            1.0e-6,
            1.0e-3,
        )

        self.assertTrue(infeasible_stall)
        self.assertFalse(false_success)
        self.assertEqual(reason, "failed_inner_solve_without_feasibility_gain")

    def test_movement_above_tolerance_is_not_stall(self):
        module = load_alm_utils_module()
        current_eval, candidate_eval = self._evals(1.0, 0.9, 2.0e-2, 2.0e-2)
        result = SimpleNamespace(success=True, message="CONVERGENCE", nit=12)

        infeasible_stall, false_success, reason = module._classify_infeasible_inner_stall(
            current_eval,
            candidate_eval,
            result,
            1.0e-3,
            1.0e-6,
            1.0e-3,
        )

        self.assertFalse(infeasible_stall)
        self.assertFalse(false_success)
        self.assertIsNone(reason)

    def test_candidate_within_feasibility_gate_is_not_stall(self):
        module = load_alm_utils_module()
        current_eval, candidate_eval = self._evals(1.0, 1.0 - 1.0e-12, 2.0e-2, 1.0e-4)
        result = SimpleNamespace(
            success=True,
            message="CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH",
            nit=12,
        )

        infeasible_stall, false_success, reason = module._classify_infeasible_inner_stall(
            current_eval,
            candidate_eval,
            result,
            0.0,
            1.0e-6,
            1.0e-3,
        )

        self.assertFalse(infeasible_stall)
        self.assertFalse(false_success)
        self.assertIsNone(reason)


class SkippedInnerShortcutTests(unittest.TestCase):
    """Pin the non-Stage-2 skipped-inner-solve fast path at `alm_utils.py:3846`.

    When an evaluator without explicit Stage-2 signal fields returns an iterate
    that already meets the user feasibility/stationarity tolerances, the ALM
    driver must skip the L-BFGS-B inner solve and emit a single ``converged``
    history entry. Existing tests cover the history schema; this test pins the
    short-circuit behavior itself by asserting that SciPy's `minimize` is never
    invoked.
    """

    def test_minimize_alm_skips_inner_solve_when_already_converged(self):
        module = load_alm_utils_module()
        settings = module.ALMSettings(
            max_outer_iterations=1,
            max_subproblem_continuations=1,
            penalty_init=1.0,
            penalty_scale=10.0,
            feasibility_tol=1.0e-6,
            stationarity_tol=1.0e-6,
            history_max_entries=None,
        )

        evaluator_calls = []

        def evaluate_problem(x, multipliers, penalty):
            del x, multipliers
            evaluator_calls.append(float(penalty))
            return _complete_alm_evaluation({
                "total": 0.0,
                "grad": np.zeros(1),
                "constraint_values": np.array([-1.0]),
                "stationarity_norm": 0.0,
            })

        def fail_minimize(*args, **kwargs):
            del args, kwargs
            raise AssertionError(
                "inner L-BFGS-B must not be invoked when the iterate already"
                " satisfies the user feasibility/stationarity tolerances"
            )

        with patch.object(module, "minimize", side_effect=fail_minimize):
            result = module.minimize_alm(
                np.zeros(1),
                ["c0"],
                evaluate_problem,
                settings,
                {"maxiter": 10},
            )

        self.assertTrue(result.success)
        self.assertEqual(result.termination_reason, "converged")
        self.assertEqual(len(result.history), 1)
        self.assertEqual(result.history[0]["action"], "converged")
        self.assertEqual(result.history[0]["inner_iterations"], 0)
        self.assertEqual(result.history[0]["inner_attempts"], 0)
        self.assertEqual(len(evaluator_calls), 1)

    def test_minimize_alm_blocks_skipped_inner_shortcut_when_cap_was_bound(self):
        """H1: pin the M4 cap-binding gate on the start-of-outer skipped-inner
        shortcut. After a prior outer iteration's dual update binds at
        ``multiplier_max``, the next outer's pre-inner shortcut must NOT emit
        ``success=True, "converged"`` even if feasibility/stationarity hold at
        the cap-clamped multipliers — the cap distorts the Lagrangian, so KKT
        is not actually attained.

        Scenario:
        1. Outer 0: evaluator returns infeasible (constraint = 0.5) when
           multipliers are still zero, forcing the inner solve and a dual
           update that clamps the multiplier at ``multiplier_max=0.2`` — sets
           ``run_state.last_cap_binding_active = True``.
        2. Outer 1 start: evaluator returns feasible+stationary when
           multipliers > 0. Without the H1 gate, the start-of-outer
           skipped-inner shortcut in ``_run_alm_continuation_step`` (the
           third success arm, alongside the post-inner converged arm and
           the constraints-inactive arm) fires and falsely labels
           ``converged``. With the gate, the shortcut is blocked.
        """
        module = load_alm_utils_module()
        settings = module.ALMSettings(
            max_outer_iterations=2,
            max_subproblem_continuations=1,
            penalty_init=1.0,
            penalty_scale=10.0,
            feasibility_tol=1e-6,
            stationarity_tol=1e-6,
            relaxed_feasibility_gate_cap=1.0,
            multiplier_max=0.2,
            history_max_entries=None,
        )

        def evaluate_problem(x, multipliers, penalty):
            del x, penalty
            if float(multipliers[0]) <= 0.0:
                return _complete_alm_evaluation({
                    "total": 1.0,
                    "grad": np.array([1.0]),
                    "constraint_values": np.array([0.5]),
                    "stationarity_norm": 1.0,
                })
            return _complete_alm_evaluation({
                "total": 0.0,
                "grad": np.zeros(1),
                "constraint_values": np.array([0.0]),
                "stationarity_norm": 0.0,
            })

        def fake_minimize(fun, x, jac, method, bounds, callback, options):
            del fun, x, jac, method, bounds, callback, options
            return SimpleNamespace(
                x=np.array([0.0]),
                nit=1,
                success=True,
                message="CONVERGENCE",
            )

        with patch.object(module, "minimize", side_effect=fake_minimize):
            result = module.minimize_alm(
                np.array([0.1]),
                ["c0"],
                evaluate_problem,
                settings,
                {"maxiter": 5, "ftol": 1e-12, "gtol": 1e-12},
            )

        self.assertTrue(result.multiplier_cap_binding)
        self.assertFalse(result.success)
        self.assertNotEqual(result.termination_reason, "converged")


if __name__ == "__main__":
    unittest.main()
