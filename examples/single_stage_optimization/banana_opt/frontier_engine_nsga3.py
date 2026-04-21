from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .frontier_archive import (
    FRONTIER_ARCHIVE_STATE_PROVISIONAL,
    build_archive_member_from_results,
    certified_archive_members,
    finalize_archive_member,
    frontier_archive_hypervolume,
    update_frontier_archive,
)
from .frontier_evaluator import (
    FrontierEvaluatorInitializationError,
    SingleStageFrontierEvaluation,
    SingleStageFrontierEvaluator,
    build_single_stage_frontier_evaluator_spec,
    load_frontier_evaluator_spec,
    objective_vector_for_minimization,
    write_frontier_evaluator_spec,
)
from .frontier_scalarization import (
    FRONTIER_REFERENCE_MODE_ACHIEVEMENT_FULL_SIMPLEX,
    generate_frontier_reference_directions,
)

try:
    from pymoo.algorithms.moo.nsga3 import NSGA3
    from pymoo.core.callback import Callback
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.optimize import minimize

    _PYMOO_IMPORT_ERROR = None
except ImportError as error:  # pragma: no cover - optional dependency
    NSGA3 = None
    Callback = object
    ElementwiseProblem = object
    minimize = None
    _PYMOO_IMPORT_ERROR = error


@dataclass(frozen=True)
class NSGA3EngineArtifacts:
    evaluator_spec: dict[str, object]
    evaluator_spec_path: str
    generation_history: list[dict[str, object]]
    archive_members: list
    provisional_archive_members: list
    population_checkpoint_path: str
    generation_history_path: str
    engine_stats: dict[str, object]


class _FrontierNSGA3Problem(ElementwiseProblem):
    def __init__(self, evaluator: SingleStageFrontierEvaluator) -> None:
        spec = evaluator.spec
        super().__init__(
            n_var=len(spec.lower_bounds),
            n_obj=4,
            n_ieq_constr=len(spec.cv_bucket_names),
            xl=np.asarray(spec.lower_bounds, dtype=float),
            xu=np.asarray(spec.upper_bounds, dtype=float),
        )
        self._evaluator = evaluator

    def _evaluate(self, x, out, *args, **kwargs) -> None:
        evaluation = self._evaluator.evaluate(x)
        out["F"] = np.asarray(
            objective_vector_for_minimization(evaluation.objective_metrics),
            dtype=float,
        )
        out["G"] = np.asarray(
            [
                float(evaluation.constraint_violations[bucket_name])
                for bucket_name in self._evaluator.spec.cv_bucket_names
            ],
            dtype=float,
        )


class _ArchiveTrackingCallback(Callback):
    def __init__(
        self,
        *,
        evaluator: SingleStageFrontierEvaluator,
        campaign_id: str,
        hypervolume_reference: Mapping[str, float] | None,
        pareto_objective_normalization: Mapping[str, object] | None,
        population_checkpoint_path: Path,
    ) -> None:
        super().__init__()
        self._evaluator = evaluator
        self._campaign_id = campaign_id
        self._hypervolume_reference = hypervolume_reference
        self._pareto_objective_normalization = pareto_objective_normalization
        self._population_checkpoint_path = population_checkpoint_path
        self.archive_members: list = []
        self.provisional_archive_members: list = []
        self.generation_history: list[dict[str, object]] = []

    def notify(self, algorithm) -> None:
        X = np.asarray(algorithm.pop.get("X"), dtype=float)
        evaluations = self._evaluator.evaluate_batch(X)
        failure_histogram = _failure_histogram(evaluations)
        certified_before = len(certified_archive_members(self.archive_members))
        for index, evaluation in enumerate(evaluations):
            payload = {
                "result_source": "nsga3_generation",
                "results_path": str(self._population_checkpoint_path),
                "results": dict(evaluation.results_payload),
            }
            rerun_contract = {
                "frontier_engine": "nsga3",
                "evaluator_spec_fingerprint": self._evaluator.spec.fingerprint(),
                "candidate_x": list(evaluation.x),
            }
            provisional_member = build_archive_member_from_results(
                campaign_id=self._campaign_id,
                lane_id=_candidate_lane_id(algorithm.n_gen, index),
                payload=payload,
                rerun_contract=rerun_contract,
                archive_state=FRONTIER_ARCHIVE_STATE_PROVISIONAL,
                pareto_objective_normalization=self._pareto_objective_normalization,
            )
            self.provisional_archive_members.append(provisional_member)
            final_member = finalize_archive_member(provisional_member)
            self.archive_members, _archive_update = update_frontier_archive(
                self.archive_members,
                final_member,
                pareto_objective_normalization=self._pareto_objective_normalization,
            )
        certified_after = certified_archive_members(self.archive_members)
        cv_values = np.asarray([evaluation.total_cv for evaluation in evaluations], dtype=float)
        self.generation_history.append(
            {
                "generation": int(algorithm.n_gen),
                "population_size": int(len(evaluations)),
                "feasible_count": int(sum(1 for evaluation in evaluations if evaluation.valid)),
                "archive_size": len(certified_after),
                "archive_growth": len(certified_after) - certified_before,
                "cv_min": float(np.min(cv_values)) if cv_values.size else 0.0,
                "cv_mean": float(np.mean(cv_values)) if cv_values.size else 0.0,
                "cv_max": float(np.max(cv_values)) if cv_values.size else 0.0,
                "failure_histogram": failure_histogram,
                "cache_hits": int(self._evaluator.cache_hits),
                "cache_misses": int(self._evaluator.cache_misses),
                "hypervolume": frontier_archive_hypervolume(
                    certified_after,
                    hypervolume_reference=self._hypervolume_reference,
                ),
            }
        )


def run_nsga3_frontier_campaign(
    args,
    *,
    campaign_id: str,
    output_root: Path,
    stage2_bs_path: Path,
    stage2_results_path: Path | None,
    stage2_results: Mapping[str, object],
    hypervolume_reference: Mapping[str, float] | None,
    pareto_objective_normalization: Mapping[str, object] | None,
    total_budget: int,
) -> NSGA3EngineArtifacts:
    if _PYMOO_IMPORT_ERROR is not None:  # pragma: no cover - depends on optional dep
        raise FrontierEvaluatorInitializationError(
            "frontier-engine=nsga3 requires pymoo to be installed."
        ) from _PYMOO_IMPORT_ERROR
    if (
        args.frontier_reference_mode
        != FRONTIER_REFERENCE_MODE_ACHIEVEMENT_FULL_SIMPLEX
    ):
        raise ValueError(
            "frontier-engine=nsga3 currently requires "
            "--frontier-reference-mode=achievement_chebyshev_full_simplex_v1."
        )

    engine_dir = output_root / "global_engine_nsga3"
    engine_dir.mkdir(parents=True, exist_ok=True)
    evaluator_spec = build_single_stage_frontier_evaluator_spec(
        args,
        stage2_bs_path=stage2_bs_path,
        stage2_results_path=stage2_results_path,
        stage2_results=stage2_results,
    )
    evaluator = SingleStageFrontierEvaluator.from_spec(
        evaluator_spec,
        cache_dir=engine_dir / "evaluator_cache",
    )
    reference_dirs = np.asarray(
        generate_frontier_reference_directions(
            requested_num_directions=max(int(args.frontier_num_lanes), 1),
            n_dim=4,
            partitions=args.frontier_full_simplex_partitions,
        ),
        dtype=float,
    )
    if reference_dirs.ndim != 2 or reference_dirs.shape[1] != 4:
        raise ValueError("NSGA-III reference directions must resolve to a 4-objective simplex.")
    population_size = int(reference_dirs.shape[0])
    generations = max(1, int(math.ceil(float(total_budget) / float(population_size))))
    evaluator_spec_path = engine_dir / "evaluator_spec.json"
    population_checkpoint_path = engine_dir / "population_checkpoint.json"
    generation_history_path = engine_dir / "generation_history.json"
    write_frontier_evaluator_spec(evaluator_spec_path, evaluator_spec)
    callback = _ArchiveTrackingCallback(
        evaluator=evaluator,
        campaign_id=campaign_id,
        hypervolume_reference=hypervolume_reference,
        pareto_objective_normalization=pareto_objective_normalization,
        population_checkpoint_path=population_checkpoint_path,
    )
    problem = _FrontierNSGA3Problem(evaluator)
    algorithm = NSGA3(
        pop_size=population_size,
        ref_dirs=reference_dirs,
    )
    result = minimize(
        problem,
        algorithm,
        termination=("n_gen", generations),
        seed=int(args.frontier_rng_seed),
        callback=callback,
        verbose=False,
    )
    population_payload = {
        "population_size": population_size,
        "generations": generations,
        "ref_dirs": reference_dirs.tolist(),
        "X": _population_field(result, "X"),
        "F": _population_field(result, "F"),
    }
    population_checkpoint_path.write_text(
        json.dumps(population_payload, indent=2),
        encoding="utf-8",
    )
    generation_history_path.write_text(
        json.dumps(callback.generation_history, indent=2),
        encoding="utf-8",
    )
    certified_members = certified_archive_members(callback.archive_members)
    engine_stats = {
        "population_size": population_size,
        "generations": generations,
        "archive_size": len(certified_members),
        "cache_hits": evaluator.cache_hits,
        "cache_misses": evaluator.cache_misses,
    }
    return NSGA3EngineArtifacts(
        evaluator_spec=evaluator_spec.to_json_dict(),
        evaluator_spec_path=str(evaluator_spec_path),
        generation_history=callback.generation_history,
        archive_members=callback.archive_members,
        provisional_archive_members=callback.provisional_archive_members,
        population_checkpoint_path=str(population_checkpoint_path),
        generation_history_path=str(generation_history_path),
        engine_stats=engine_stats,
    )


def load_nsga3_frontier_campaign_artifacts(
    *,
    output_root: Path,
    archive_members: Sequence,
    provisional_archive_members: Sequence,
) -> NSGA3EngineArtifacts | None:
    engine_dir = output_root / "global_engine_nsga3"
    evaluator_spec_path = engine_dir / "evaluator_spec.json"
    population_checkpoint_path = engine_dir / "population_checkpoint.json"
    generation_history_path = engine_dir / "generation_history.json"
    required_paths = (
        evaluator_spec_path,
        population_checkpoint_path,
        generation_history_path,
    )
    if not all(path.exists() for path in required_paths):
        return None
    evaluator_spec_payload = load_frontier_evaluator_spec(
        evaluator_spec_path
    ).to_json_dict()
    population_payload = json.loads(
        population_checkpoint_path.read_text(encoding="utf-8")
    )
    generation_history = json.loads(
        generation_history_path.read_text(encoding="utf-8")
    )
    if not isinstance(population_payload, dict):
        raise ValueError("nsga3 population checkpoint must be a JSON object")
    if not isinstance(generation_history, list):
        raise ValueError("nsga3 generation history checkpoint must be a JSON list")
    certified_members = certified_archive_members(list(archive_members))
    last_generation = (
        generation_history[-1]
        if generation_history and isinstance(generation_history[-1], Mapping)
        else {}
    )
    return NSGA3EngineArtifacts(
        evaluator_spec=evaluator_spec_payload,
        evaluator_spec_path=str(evaluator_spec_path),
        generation_history=[
            dict(entry) if isinstance(entry, Mapping) else {"value": entry}
            for entry in generation_history
        ],
        archive_members=list(archive_members),
        provisional_archive_members=list(provisional_archive_members),
        population_checkpoint_path=str(population_checkpoint_path),
        generation_history_path=str(generation_history_path),
        engine_stats={
            "population_size": int(population_payload.get("population_size", 0)),
            "generations": int(population_payload.get("generations", 0)),
            "archive_size": len(certified_members),
            "cache_hits": int(last_generation.get("cache_hits", 0)),
            "cache_misses": int(last_generation.get("cache_misses", 0)),
        },
    )


def build_nsga3_hypervolume_history(
    generation_history: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    history: list[dict[str, object]] = []
    for entry in generation_history:
        generation = int(entry["generation"])
        history.append(
            {
                "lane_id": f"generation_{generation:04d}",
                "status": "completed",
                "archive_size": int(entry["archive_size"]),
                "hypervolume": float(entry["hypervolume"]),
            }
        )
    return history


def _failure_histogram(
    evaluations: Sequence[SingleStageFrontierEvaluation],
) -> dict[str, int]:
    histogram: dict[str, int] = {}
    for evaluation in evaluations:
        reason = str(
            evaluation.results_payload.get(
                "TERMINATION_MESSAGE",
                "unknown",
            )
        )
        histogram[reason] = histogram.get(reason, 0) + 1
    return histogram


def _candidate_lane_id(generation: int, candidate_index: int) -> str:
    return f"gen_{generation:04d}_cand_{candidate_index:04d}"


def _jsonable_array(value) -> list[object]:
    return np.asarray(value, dtype=float).tolist()


def _population_field(result, name: str) -> list[object] | None:
    population = getattr(result, "pop", None)
    if population is None:
        value = getattr(result, name, None)
        return None if value is None else _jsonable_array(value)
    value = population.get(name)
    return None if value is None else _jsonable_array(value)
