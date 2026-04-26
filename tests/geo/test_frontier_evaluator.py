import importlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_ROOT = REPO_ROOT / "examples" / "single_stage_optimization"


def _ensure_examples_import_path():
    examples_root = str(EXAMPLES_ROOT)
    if examples_root not in sys.path:
        sys.path.insert(0, examples_root)


def _load_banana_opt_module(module_name: str):
    _ensure_examples_import_path()
    return importlib.import_module(f"banana_opt.{module_name}")


def load_frontier_evaluator_module():
    return _load_banana_opt_module("frontier_evaluator")


def load_frontier_engine_nsga3_module():
    return _load_banana_opt_module("frontier_engine_nsga3")


def _run_python_snippet(source: str, *args: str) -> str:
    result = subprocess.run(
        [sys.executable, "-c", source, *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


class _FakeAlgebraicObjective:
    def __init__(self, value, grad):
        self._value = float(value)
        self._grad = np.asarray(grad, dtype=float)

    def J(self):
        return self._value

    def dJ(self, partials=False):
        if partials:
            return lambda _objective: self._grad.copy()
        return self._grad.copy()

    def __add__(self, other):
        if other == 0:
            return self
        return _FakeAlgebraicObjective(
            self._value + other._value,
            self._grad + other._grad,
        )

    __radd__ = __add__

    def __mul__(self, scalar):
        return _FakeAlgebraicObjective(
            self._value * float(scalar),
            self._grad * float(scalar),
        )

    __rmul__ = __mul__


def _demo_evaluation(
    module,
    spec,
    *,
    candidate,
    cache_key,
    termination_message,
    diagnostics_source,
):
    return module.SingleStageFrontierEvaluation(
        schema_version=module.FRONTIER_EVALUATION_SCHEMA_VERSION,
        candidate_id=cache_key[:16],
        x=np.asarray(candidate, dtype=float).tolist(),
        valid=True,
        objective_metrics={
            "iota": float(candidate[0]),
            "volume": 0.11,
            "qa_error": 0.01,
            "boozer_residual": 0.007,
        },
        reference_metrics=dict(spec.reference_metrics),
        constraint_violations={
            bucket_name: 0.0 for bucket_name in module.FRONTIER_EVALUATOR_CV_BUCKETS
        },
        results_payload={"TERMINATION_MESSAGE": termination_message},
        diagnostics={"source": diagnostics_source},
        cache_key=cache_key,
    )


class FrontierEvaluatorTests(unittest.TestCase):
    def test_nsga3_population_checkpoint_uses_final_population_arrays(self):
        evaluator_module = load_frontier_evaluator_module()
        engine_module = load_frontier_engine_nsga3_module()
        spec = self._demo_spec(evaluator_module)

        def make_evaluation(candidate):
            return _demo_evaluation(
                evaluator_module,
                spec,
                candidate=candidate,
                cache_key=f"cache-{candidate[0]}",
                termination_message="valid",
                diagnostics_source="nsga3-population-test",
            )

        class _ResultPopulation:
            def __init__(self, x_values, f_values):
                self._payload = {
                    "X": np.asarray(x_values, dtype=float),
                    "F": np.asarray(f_values, dtype=float),
                }

            def get(self, name):
                return self._payload.get(name)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir) / "outputs"
            result_population = _ResultPopulation(
                [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
                [
                    [-0.18, -0.11, 0.0104, 0.0069],
                    [-0.17, -0.10, 0.0108, 0.0072],
                    [-0.16, -0.09, 0.0111, 0.0076],
                ],
            )

            evaluator_stub = SimpleNamespace(
                spec=spec,
                cache_hits=0,
                cache_misses=3,
                evaluate=lambda candidate: make_evaluation(candidate),
                evaluate_batch=lambda X: [make_evaluation(candidate) for candidate in X],
            )

            def fake_minimize(problem, algorithm, termination, seed, callback, verbose):
                algorithm.n_gen = 1
                algorithm.pop = _ResultPopulation(
                    [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
                    [
                        [-0.18, -0.11, 0.0104, 0.0069],
                        [-0.17, -0.10, 0.0108, 0.0072],
                        [-0.16, -0.09, 0.0111, 0.0076],
                    ],
                )
                callback.notify(algorithm)
                return SimpleNamespace(
                    X=np.asarray([[9.0, 9.0]], dtype=float),
                    F=np.asarray([[-9.0, -9.0, 9.0, 9.0]], dtype=float),
                    pop=result_population,
                )

            class _FakeProblem:
                def __init__(self, evaluator):
                    self.evaluator = evaluator

            with patch.object(
                engine_module,
                "_PYMOO_IMPORT_ERROR",
                None,
            ), patch.object(
                engine_module,
                "build_single_stage_frontier_evaluator_spec",
                return_value=spec,
            ), patch.object(
                engine_module.SingleStageFrontierEvaluator,
                "from_spec",
                return_value=evaluator_stub,
            ), patch.object(
                engine_module,
                "generate_frontier_reference_directions",
                return_value=[[1.0, 0.0, 0.0, 0.0]],
            ), patch.object(
                engine_module,
                "write_frontier_evaluator_spec",
            ), patch.object(
                engine_module,
                "build_archive_member_from_results",
                side_effect=lambda **kwargs: kwargs,
            ), patch.object(
                engine_module,
                "finalize_archive_member",
                side_effect=lambda member: member,
            ), patch.object(
                engine_module,
                "update_frontier_archive",
                side_effect=lambda members, final_member, pareto_objective_normalization=None: (
                    [*members, final_member],
                    {},
                ),
            ), patch.object(
                engine_module,
                "certified_archive_members",
                side_effect=lambda members: list(members),
            ), patch.object(
                engine_module,
                "frontier_archive_hypervolume",
                return_value=1.0e-4,
            ), patch.object(
                engine_module,
                "_FrontierNSGA3Problem",
                _FakeProblem,
            ), patch.object(
                engine_module,
                "NSGA3",
                side_effect=lambda **kwargs: SimpleNamespace(**kwargs),
            ), patch.object(
                engine_module,
                "minimize",
                side_effect=fake_minimize,
            ):
                artifacts = engine_module.run_nsga3_frontier_campaign(
                    SimpleNamespace(
                        frontier_reference_mode=engine_module.FRONTIER_REFERENCE_MODE_ACHIEVEMENT_FULL_SIMPLEX,
                        frontier_num_lanes=1,
                        frontier_full_simplex_partitions=1,
                        frontier_rng_seed=7,
                    ),
                    campaign_id="campaign",
                    output_root=output_root,
                    stage2_bs_path=Path("/tmp/demo-bs.json"),
                    stage2_results_path=None,
                    stage2_results={"PLASMA_SURF_FILENAME": "demo.nc"},
                    hypervolume_reference=None,
                    pareto_objective_normalization=None,
                    total_budget=3,
                )

            checkpoint_payload = json.loads(
                Path(artifacts.population_checkpoint_path).read_text(encoding="utf-8")
            )
            self.assertEqual(
                checkpoint_payload["X"],
                [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
            )
            self.assertEqual(
                checkpoint_payload["F"],
                [
                    [-0.18, -0.11, 0.0104, 0.0069],
                    [-0.17, -0.1, 0.0108, 0.0072],
                    [-0.16, -0.09, 0.0111, 0.0076],
                ],
            )

    def test_objective_vector_for_minimization_flips_max_metrics(self):
        module = load_frontier_evaluator_module()

        vector = module.objective_vector_for_minimization(
            {
                "iota": 0.18,
                "volume": 0.11,
                "qa_error": 0.0105,
                "boozer_residual": 0.0072,
            }
        )

        self.assertEqual(vector, [-0.18, -0.11, 0.0105, 0.0072])

    def test_evaluator_spec_round_trips_and_keeps_fingerprint(self):
        module = load_frontier_evaluator_module()
        spec = self._demo_spec(module)

        round_tripped = module.SingleStageFrontierEvaluatorSpec.from_json_dict(
            spec.to_json_dict()
        )

        self.assertEqual(round_tripped.to_json_dict(), spec.to_json_dict())
        self.assertEqual(round_tripped.fingerprint(), spec.fingerprint())

    def test_frontier_evaluation_total_cv_counts_only_positive_buckets(self):
        module = load_frontier_evaluator_module()

        evaluation = module.SingleStageFrontierEvaluation(
            schema_version=module.FRONTIER_EVALUATION_SCHEMA_VERSION,
            candidate_id="candidate",
            x=[0.0],
            valid=False,
            objective_metrics={
                "iota": 0.15,
                "volume": 0.10,
                "qa_error": 0.012,
                "boozer_residual": 0.008,
            },
            reference_metrics={
                "iota": 0.15,
                "volume": 0.10,
                "qa_error": 0.012,
                "boozer_residual": 0.008,
            },
            constraint_violations={
                "surface_solve_failed": 1.0,
                "geometry_state_unrestorable": 0.0,
                "missing_search_eval": 0.0,
                "nonfinite_evaluation": 2.0,
                "topology_broken": -1.0,
                "topology_deficit": 0.3,
                "hardware_violation_ratio": 0.0,
                "frontier_trust_excess_ratio": 0.1,
            },
            results_payload={"TERMINATION_MESSAGE": "surface_solve_failed"},
            diagnostics={},
            cache_key="cache",
        )

        self.assertAlmostEqual(evaluation.total_cv, 3.4)

    def test_frontier_evaluator_spec_file_round_trip_survives_fresh_process(self):
        module = load_frontier_evaluator_module()
        spec = self._demo_spec(module)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            spec_path = tmpdir_path / "spec.json"
            cache_dir = tmpdir_path / "cache"
            module.write_frontier_evaluator_spec(spec_path, spec)

            payload = json.loads(
                _run_python_snippet(
                    """
import json
import sys

sys.path.insert(0, sys.argv[1])
from banana_opt import frontier_evaluator as m

spec = m.load_frontier_evaluator_spec(sys.argv[2])
m.build_single_stage_frontier_runtime = lambda loaded_spec: type(
    "Runtime",
    (),
    {"spec": loaded_spec},
)()

def fake_eval(self, candidate, *, cache_key):
    return m.SingleStageFrontierEvaluation(
        schema_version=m.FRONTIER_EVALUATION_SCHEMA_VERSION,
        candidate_id=cache_key[:16],
        x=candidate.tolist(),
        valid=True,
        objective_metrics={
            "iota": float(candidate[0]),
            "volume": 0.11,
            "qa_error": 0.01,
            "boozer_residual": 0.007,
        },
        reference_metrics=dict(spec.reference_metrics),
        constraint_violations={name: 0.0 for name in spec.cv_bucket_names},
        results_payload={"TERMINATION_MESSAGE": "fresh_process_eval"},
        diagnostics={"source": "subprocess"},
        cache_key=cache_key,
    )

m.SingleStageFrontierEvaluator._evaluate_uncached = fake_eval
evaluator = m.SingleStageFrontierEvaluator.from_spec(spec, cache_dir=sys.argv[3])
print(
    json.dumps(
        {
            "fingerprint": evaluator.spec.fingerprint(),
            "evaluation": evaluator.evaluate([0.25]).to_json_dict(),
        }
    )
)
""",
                    str(EXAMPLES_ROOT),
                    str(spec_path),
                    str(cache_dir),
                )
            )

            self.assertEqual(payload["fingerprint"], spec.fingerprint())
            self.assertEqual(payload["evaluation"]["objective_metrics"]["iota"], 0.25)
            self.assertEqual(
                payload["evaluation"]["results_payload"]["TERMINATION_MESSAGE"],
                "fresh_process_eval",
            )

    def test_evaluator_file_cache_reuses_results_after_reinstantiation(self):
        module = load_frontier_evaluator_module()
        spec = self._demo_spec(module)
        evaluation = _demo_evaluation(
            module,
            spec,
            candidate=[0.2],
            cache_key="placeholder",
            termination_message="cached",
            diagnostics_source="unit-test",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_stub = SimpleNamespace(spec=spec)
            with patch.object(
                module,
                "build_single_stage_frontier_runtime",
                return_value=runtime_stub,
            ), patch.object(
                module.SingleStageFrontierEvaluator,
                "_evaluate_uncached",
                autospec=True,
                side_effect=lambda self, candidate, *, cache_key: module.SingleStageFrontierEvaluation(
                    **{
                        **evaluation.to_json_dict(),
                        "candidate_id": cache_key[:16],
                        "cache_key": cache_key,
                    }
                ),
            ) as evaluate_uncached:
                evaluator = module.SingleStageFrontierEvaluator.from_spec(
                    spec,
                    cache_dir=tmpdir,
                )
                first = evaluator.evaluate([0.2])
                self.assertEqual(evaluate_uncached.call_count, 1)
                self.assertEqual(evaluator.cache_misses, 1)
                cache_path = (
                    Path(tmpdir)
                    / first.cache_key[:2]
                    / f"{first.cache_key}.json"
                )
                self.assertTrue(cache_path.exists())

            with patch.object(
                module,
                "build_single_stage_frontier_runtime",
                return_value=runtime_stub,
            ), patch.object(
                module.SingleStageFrontierEvaluator,
                "_evaluate_uncached",
                autospec=True,
            ) as evaluate_uncached:
                evaluator = module.SingleStageFrontierEvaluator.from_spec(
                    spec,
                    cache_dir=tmpdir,
                )
                second = evaluator.evaluate([0.2])
                self.assertEqual(evaluate_uncached.call_count, 0)
                self.assertEqual(evaluator.cache_hits, 1)

            self.assertEqual(first.to_json_dict(), second.to_json_dict())

    def test_frontier_evaluator_memory_cache_evicts_lru_entries(self):
        module = load_frontier_evaluator_module()
        spec = self._demo_spec(module)
        runtime_stub = SimpleNamespace(spec=spec)

        def make_evaluation(self, candidate, *, cache_key):
            del self
            return _demo_evaluation(
                module,
                spec,
                candidate=candidate,
                cache_key=cache_key,
                termination_message="lru",
                diagnostics_source="unit-test",
            )

        with tempfile.TemporaryDirectory() as tmpdir, patch.object(
            module,
            "build_single_stage_frontier_runtime",
            return_value=runtime_stub,
        ), patch.object(
            module.SingleStageFrontierEvaluator,
            "_evaluate_uncached",
            autospec=True,
            side_effect=make_evaluation,
        ):
            evaluator = module.SingleStageFrontierEvaluator.from_spec(
                spec,
                cache_dir=tmpdir,
                cache_max_entries=2,
            )
            first = evaluator.evaluate([0.1])
            second = evaluator.evaluate([0.2])
            evaluator.evaluate([0.1])
            third = evaluator.evaluate([0.3])

            self.assertEqual(set(evaluator._cache), {first.cache_key, third.cache_key})
            self.assertNotIn(second.cache_key, evaluator._cache)
            self.assertEqual(len(evaluator._cache), 2)

    def test_frontier_runtime_rejects_independent_banana_current_mode(self):
        module = load_frontier_evaluator_module()
        spec = self._demo_spec(module)

        with patch.object(
            module.single_stage,
            "apply_default_stage2_seed_args",
            return_value=SimpleNamespace(
                single_stage_banana_current_mode="independent",
            ),
        ):
            with self.assertRaisesRegex(
                module.FrontierEvaluatorInitializationError,
                "does not support .*single-stage-banana-current-mode=independent",
            ):
                module.build_single_stage_frontier_runtime(spec)

    def test_evaluate_batch_reuses_in_batch_duplicates_and_preserves_order(self):
        module = load_frontier_evaluator_module()
        spec = self._demo_spec(module)

        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_stub = SimpleNamespace(spec=spec)

            def make_evaluation(self, candidate, *, cache_key):
                return _demo_evaluation(
                    module,
                    spec,
                    candidate=candidate,
                    cache_key=cache_key,
                    termination_message="batched",
                    diagnostics_source="unit-test",
                )

            with patch.object(
                module,
                "build_single_stage_frontier_runtime",
                return_value=runtime_stub,
            ), patch.object(
                module.SingleStageFrontierEvaluator,
                "_evaluate_uncached",
                autospec=True,
                side_effect=make_evaluation,
            ) as evaluate_uncached:
                evaluator = module.SingleStageFrontierEvaluator.from_spec(
                    spec,
                    cache_dir=tmpdir,
                )
                batch = evaluator.evaluate_batch([[0.2], [0.3], [0.2]])

                self.assertEqual(evaluate_uncached.call_count, 2)
                self.assertEqual(evaluator.cache_misses, 2)
                self.assertEqual(evaluator.cache_hits, 0)
                self.assertEqual(
                    [entry.objective_metrics["iota"] for entry in batch],
                    [0.2, 0.3, 0.2],
                )
                self.assertEqual(batch[0].cache_key, batch[2].cache_key)

                repeated = evaluator.evaluate_batch([[0.3], [0.2]])
                self.assertEqual(evaluate_uncached.call_count, 2)
                self.assertEqual(evaluator.cache_hits, 2)
                self.assertEqual(
                    [entry.objective_metrics["iota"] for entry in repeated],
                    [0.3, 0.2],
                )

    def test_frontier_search_objective_shadow_matches_single_stage_inline_path(self):
        evaluator_module = load_frontier_evaluator_module()
        single_stage = evaluator_module.single_stage
        frontier_goal_config = single_stage.build_frontier_goal_config(
            initial_iota=0.15,
            initial_volume=0.10,
            initial_qs_objective=0.012,
            initial_boozer_objective=0.008,
            res_weight=1000.0,
            iotas_weight=100.0,
            volume_weight=120.0,
            scalarization_type="achievement_chebyshev_sweep_v1",
            chebyshev_rho_override=0.02,
            chebyshev_weight_iota_override=2.0,
            chebyshev_weight_volume_override=1.5,
            chebyshev_weight_qa_override=1.0,
            chebyshev_weight_boozer_override=0.5,
        )
        surface_weights = np.array([1.0, 2.0], dtype=float)
        bundle = {
            "surface_iota_terms": [_FakeAlgebraicObjective(0.13, [1.0, 0.0])],
            "surface_volume_term": _FakeAlgebraicObjective(0.09, [0.0, 1.0]),
            "nonQSs": [
                _FakeAlgebraicObjective(1.2e-4, [0.5, 0.0]),
                _FakeAlgebraicObjective(1.8e-4, [0.1, 0.2]),
            ],
            "brs": [
                _FakeAlgebraicObjective(2.0e-6, [0.0, 0.4]),
                _FakeAlgebraicObjective(5.0e-6, [0.1, 0.1]),
            ],
            "curvelength": _FakeAlgebraicObjective(1.7, [0.0, 0.0]),
            "Jiota": _FakeAlgebraicObjective(-0.1, [-0.3, 0.0]),
            "JVolume": _FakeAlgebraicObjective(-0.2, [0.0, -0.2]),
            "JnonQSRatio": _FakeAlgebraicObjective(1.6e-4, [0.0, 0.0]),
            "JnonQSRatioObjective": _FakeAlgebraicObjective(1.2, [0.5, 0.0]),
            "JBoozerResidual": _FakeAlgebraicObjective(3.0e-6, [0.0, 0.0]),
            "JBoozerResidualObjective": _FakeAlgebraicObjective(2.0, [0.0, 0.4]),
            "effective_res_weight": frontier_goal_config.effective_boozer_weight,
            "effective_iotas_weight": frontier_goal_config.effective_iota_weight,
            "effective_volume_weight": frontier_goal_config.effective_volume_weight,
            "frontier_goal_config": frontier_goal_config,
            "JCurveLength": _FakeAlgebraicObjective(0.05, [0.1, 0.1]),
            "JCurveCurve": _FakeAlgebraicObjective(0.25, [0.3, 0.4]),
            "JCurveSurface": _FakeAlgebraicObjective(0.15, [0.2, -0.1]),
            "JSurfSurf": None,
            "JCurvature": _FakeAlgebraicObjective(0.35, [0.2, 0.3]),
            "JF": object(),
        }
        runtime = SimpleNamespace(
            constraint_method="penalty",
            surface_weights=surface_weights,
            objective_bundle=bundle,
            objective_optimizable=object(),
            trust_threshold=float(frontier_goal_config.boozer_trust_threshold),
            trust_penalty_scale=float(frontier_goal_config.boozer_trust_penalty_scale),
            args=SimpleNamespace(
                length_weight=0.5,
                cc_weight=2.0,
                cs_weight=3.0,
                curvature_weight=4.0,
                surf_dist_weight=0.0,
                alm_formulation="weighted_sum",
                single_stage_goal_mode="frontier",
            ),
        )

        shadow_eval = evaluator_module._evaluate_search_objective(runtime)

        with patch.multiple(
            single_stage,
            create=True,
            SINGLE_STAGE_GOAL_MODE="frontier",
            FRONTIER_GOAL_CONFIG=frontier_goal_config,
            surface_iota_terms=bundle["surface_iota_terms"],
            surface_volume_term=bundle["surface_volume_term"],
            nonQSs=bundle["nonQSs"],
            brs=bundle["brs"],
            Jiota=bundle["Jiota"],
            JVolume=bundle["JVolume"],
            JnonQSRatioObjective=bundle["JnonQSRatioObjective"],
            JBoozerResidualObjective=bundle["JBoozerResidualObjective"],
            EFFECTIVE_RES_WEIGHT=bundle["effective_res_weight"],
            EFFECTIVE_IOTAS_WEIGHT=bundle["effective_iotas_weight"],
            EFFECTIVE_VOLUME_WEIGHT=bundle["effective_volume_weight"],
            JCurveLength=bundle["JCurveLength"],
            JCurveCurve=bundle["JCurveCurve"],
            JCurveSurface=bundle["JCurveSurface"],
            JSurfSurf=bundle["JSurfSurf"],
            JCurvature=bundle["JCurvature"],
            JF=bundle["JF"],
            RES_WEIGHT=1000.0,
            IOTAS_WEIGHT=100.0,
            LENGTH_WEIGHT=0.5,
            CC_WEIGHT=2.0,
            CS_WEIGHT=3.0,
            CURVATURE_WEIGHT=4.0,
            SURF_DIST_WEIGHT=0.0,
        ):
            inline_eval = single_stage.evaluate_search_objective(surface_weights)

        for key in (
            "total",
            "frontier_goal_total",
            "frontier_scalarization_total",
            "J_QS",
            "J_Boozer",
            "J_iota",
            "J_volume",
            "frontier_trust_penalty",
            "frontier_boozer_trust_excess_ratio",
        ):
            self.assertAlmostEqual(shadow_eval[key], inline_eval[key])
        np.testing.assert_allclose(shadow_eval["grad"], inline_eval["grad"])
        np.testing.assert_allclose(
            shadow_eval["frontier_goal_grad"],
            inline_eval["frontier_goal_grad"],
        )
        np.testing.assert_allclose(
            shadow_eval["dJ_iota_metric"],
            inline_eval["dJ_iota_metric"],
        )
        np.testing.assert_allclose(
            shadow_eval["dJ_volume_metric"],
            inline_eval["dJ_volume_metric"],
        )

    def test_frontier_alm_surface_stack_uses_configured_gap_not_ramped_gate(self):
        evaluator_module = load_frontier_evaluator_module()
        captured = {}
        bundle = {
            "nonQSs": [_FakeAlgebraicObjective(0.0, [0.0])],
            "brs": [_FakeAlgebraicObjective(0.0, [0.0])],
            "effective_res_weight": 1.0,
            "effective_iotas_weight": 1.0,
            "effective_volume_weight": 1.0,
            "Jiota": _FakeAlgebraicObjective(0.0, [0.0]),
            "JVolume": _FakeAlgebraicObjective(0.0, [0.0]),
            "JCurveLength": _FakeAlgebraicObjective(0.0, [0.0]),
            "JCurveCurve": _FakeAlgebraicObjective(0.0, [0.0]),
            "JCurveSurface": _FakeAlgebraicObjective(0.0, [0.0]),
            "JCurvature": _FakeAlgebraicObjective(0.0, [0.0]),
            "JSurfSurf": None,
            "curvelength": _FakeAlgebraicObjective(0.0, [0.0]),
            "frontier_goal_config": None,
            "surface_iota_terms": [_FakeAlgebraicObjective(0.0, [0.0])],
            "surface_volume_term": _FakeAlgebraicObjective(0.0, [0.0]),
            "JnonQSRatioObjective": _FakeAlgebraicObjective(0.0, [0.0]),
            "JBoozerResidualObjective": _FakeAlgebraicObjective(0.0, [0.0]),
        }
        runtime = SimpleNamespace(
            constraint_method="alm",
            surface_weights=np.array([1.0, 1.0], dtype=float),
            objective_bundle=bundle,
            objective_optimizable=object(),
            trust_threshold=None,
            trust_penalty_scale=None,
            curves=[object()],
            surface_data=[
                {"boozer_surface": SimpleNamespace(surface=object())},
                {"boozer_surface": SimpleNamespace(surface=object())},
            ],
            search_gate={"surface_gap_threshold": 0.0},
            args=SimpleNamespace(
                alm_formulation="weighted_sum",
                alm_penalty_init=1.0,
                alm_distance_smoothing=0.005,
                alm_curvature_smoothing=0.05,
                alm_qs_threshold=None,
                alm_boozer_threshold=None,
                alm_iota_penalty_threshold=None,
                alm_length_penalty_threshold=None,
                surface_gap_threshold=0.02,
                single_stage_goal_mode="target",
                length_weight=1.0,
                cc_weight=1.0,
                cs_weight=1.0,
                curvature_weight=1.0,
                surf_dist_weight=0.0,
            ),
            curve_curve_distance_threshold=0.05,
            curve_surface_distance_threshold=0.015,
            curvature_threshold=100.0,
            vessel_surface=None,
            surface_vessel_distance_threshold=0.04,
            banana_coils=[SimpleNamespace(curve=object(), current=object())],
            banana_current_max_A=16000.0,
            length_target=1.7,
        )

        def fake_evaluate_alm_objective(*_args, **kwargs):
            captured.update(kwargs)
            return {"total": 0.0, "grad": np.zeros(1)}

        with patch.object(
            evaluator_module,
            "evaluate_alm_objective",
            side_effect=fake_evaluate_alm_objective,
        ):
            evaluator_module._evaluate_search_objective(runtime)

        self.assertIn("surface_surface_spacing", captured["constraint_names"])
        self.assertEqual(captured["surface_stack_min_distance"], 0.02)
        self.assertEqual(len(captured["surface_stack_surfaces"]), 2)

    def _demo_spec(self, module):
        return module.SingleStageFrontierEvaluatorSpec(
            schema_version=module.FRONTIER_EVALUATOR_SPEC_SCHEMA_VERSION,
            args_payload={"single_stage_goal_mode": "frontier"},
            stage2_bs_path="/tmp/biot_savart_opt.json",
            stage2_results_path="/tmp/results.json",
            stage2_results={"PLASMA_SURF_FILENAME": "demo.nc"},
            run_identity="demo-run",
            decision_variables=[
                module.SingleStageFrontierDecisionVariableSpec(
                    name="phic(1)",
                    semantic_role="phic",
                    harmonic_index=1,
                    lower_bound=-1.0,
                    upper_bound=1.0,
                )
            ],
            lower_bounds=[-1.0],
            upper_bounds=[1.0],
            seed_x=[0.2],
            reference_metrics={
                "iota": 0.15,
                "volume": 0.10,
                "qa_error": 0.012,
                "boozer_residual": 0.008,
            },
            cv_bucket_names=list(module.FRONTIER_EVALUATOR_CV_BUCKETS),
            surface_weight_schedule=[1.0],
            search_gate={"surface_gap_threshold": 0.0},
        )
