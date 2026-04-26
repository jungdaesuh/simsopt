from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from simsopt.geo.surfaceobjectives import Iotas

from SINGLE_STAGE import single_stage_banana_example as single_stage

from .frontier_constraints import (
    annotate_frontier_search_eval,
    evaluate_frontier_hard_invalidation,
    hardware_violation_ratios,
)
from .frontier_dominance import PARETO_OBJECTIVE_SPECS
from .single_stage_banana_current_mode import BANANA_CURRENT_MODE_SHARED
from .single_stage_geometry import (
    build_surface_search_gate,
    build_surface_search_weights,
    evaluate_single_stage_hardware_snapshot,
    restore_surface_states,
    snapshot_surface_states,
    solve_surface_stack_at_dofs,
    topology_gate_deficit,
)
from .single_stage_objectives import (
    apply_frontier_scalarization_override,
    evaluate_alm_objective,
    evaluate_total_objective,
)

FRONTIER_EVALUATOR_SPEC_SCHEMA_VERSION = "single_stage_frontier_evaluator_spec_v1"
FRONTIER_EVALUATION_SCHEMA_VERSION = "single_stage_frontier_evaluation_v1"
FRONTIER_EVALUATOR_CACHE_SCHEMA_VERSION = "single_stage_frontier_evaluator_cache_v1"
FRONTIER_EVALUATOR_CV_BUCKETS = (
    "surface_solve_failed",
    "geometry_state_unrestorable",
    "missing_search_eval",
    "nonfinite_evaluation",
    "topology_broken",
    "topology_deficit",
    "hardware_violation_ratio",
    "frontier_trust_excess_ratio",
)
_DOF_NAME_PATTERN = re.compile(r"^(?P<family>[A-Za-z_]+)\((?P<index>\d+)\)$")


class FrontierEvaluatorInitializationError(RuntimeError):
    """Raised when the evaluator spec cannot be instantiated."""


@dataclass(frozen=True)
class SingleStageFrontierDecisionVariableSpec:
    name: str
    semantic_role: str
    harmonic_index: int | None
    lower_bound: float
    upper_bound: float

    def to_json_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_json_dict(
        cls,
        payload: Mapping[str, object],
    ) -> SingleStageFrontierDecisionVariableSpec:
        return cls(
            name=str(payload["name"]),
            semantic_role=str(payload["semantic_role"]),
            harmonic_index=None
            if payload.get("harmonic_index") is None
            else int(payload["harmonic_index"]),
            lower_bound=float(payload["lower_bound"]),
            upper_bound=float(payload["upper_bound"]),
        )


@dataclass(frozen=True)
class SingleStageFrontierEvaluatorSpec:
    schema_version: str
    args_payload: dict[str, object]
    stage2_bs_path: str
    stage2_results_path: str | None
    stage2_results: dict[str, object]
    run_identity: str
    decision_variables: list[SingleStageFrontierDecisionVariableSpec]
    lower_bounds: list[float]
    upper_bounds: list[float]
    seed_x: list[float]
    reference_metrics: dict[str, float | None]
    cv_bucket_names: list[str]
    surface_weight_schedule: list[float]
    search_gate: dict[str, object]

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "args_payload": dict(self.args_payload),
            "stage2_bs_path": self.stage2_bs_path,
            "stage2_results_path": self.stage2_results_path,
            "stage2_results": dict(self.stage2_results),
            "run_identity": self.run_identity,
            "decision_variables": [
                variable.to_json_dict() for variable in self.decision_variables
            ],
            "lower_bounds": list(self.lower_bounds),
            "upper_bounds": list(self.upper_bounds),
            "seed_x": list(self.seed_x),
            "reference_metrics": dict(self.reference_metrics),
            "cv_bucket_names": list(self.cv_bucket_names),
            "surface_weight_schedule": list(self.surface_weight_schedule),
            "search_gate": dict(self.search_gate),
        }

    @classmethod
    def from_json_dict(
        cls,
        payload: Mapping[str, object],
    ) -> SingleStageFrontierEvaluatorSpec:
        return cls(
            schema_version=str(payload["schema_version"]),
            args_payload=dict(payload["args_payload"]),
            stage2_bs_path=str(payload["stage2_bs_path"]),
            stage2_results_path=None
            if payload.get("stage2_results_path") is None
            else str(payload["stage2_results_path"]),
            stage2_results=dict(payload["stage2_results"]),
            run_identity=str(payload["run_identity"]),
            decision_variables=[
                SingleStageFrontierDecisionVariableSpec.from_json_dict(item)
                for item in payload["decision_variables"]
            ],
            lower_bounds=[float(value) for value in payload["lower_bounds"]],
            upper_bounds=[float(value) for value in payload["upper_bounds"]],
            seed_x=[float(value) for value in payload["seed_x"]],
            reference_metrics={
                str(key): None if value is None else float(value)
                for key, value in dict(payload["reference_metrics"]).items()
            },
            cv_bucket_names=[str(item) for item in payload["cv_bucket_names"]],
            surface_weight_schedule=[
                float(value) for value in payload["surface_weight_schedule"]
            ],
            search_gate=dict(payload["search_gate"]),
        )

    def fingerprint(self) -> str:
        payload = json.dumps(
            self.to_json_dict(),
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_frontier_evaluator_spec(
    path: str | Path,
    spec: SingleStageFrontierEvaluatorSpec,
) -> None:
    spec_path = Path(path)
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(
        json.dumps(spec.to_json_dict(), indent=2),
        encoding="utf-8",
    )


def load_frontier_evaluator_spec(
    path: str | Path,
) -> SingleStageFrontierEvaluatorSpec:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("frontier evaluator spec payload must be a JSON object")
    spec = SingleStageFrontierEvaluatorSpec.from_json_dict(payload)
    if spec.schema_version != FRONTIER_EVALUATOR_SPEC_SCHEMA_VERSION:
        raise ValueError(
            "frontier evaluator spec schema_version must be "
            f"{FRONTIER_EVALUATOR_SPEC_SCHEMA_VERSION!r}"
        )
    return spec


@dataclass(frozen=True)
class SingleStageFrontierEvaluation:
    schema_version: str
    candidate_id: str
    x: list[float]
    valid: bool
    objective_metrics: dict[str, float]
    reference_metrics: dict[str, float | None]
    constraint_violations: dict[str, float]
    results_payload: dict[str, object]
    diagnostics: dict[str, object]
    cache_key: str

    def to_json_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_json_dict(
        cls,
        payload: Mapping[str, object],
    ) -> SingleStageFrontierEvaluation:
        return cls(
            schema_version=str(payload["schema_version"]),
            candidate_id=str(payload["candidate_id"]),
            x=[float(value) for value in payload["x"]],
            valid=bool(payload["valid"]),
            objective_metrics={
                str(key): float(value)
                for key, value in dict(payload["objective_metrics"]).items()
            },
            reference_metrics={
                str(key): None if value is None else float(value)
                for key, value in dict(payload["reference_metrics"]).items()
            },
            constraint_violations={
                str(key): float(value)
                for key, value in dict(payload["constraint_violations"]).items()
            },
            results_payload=dict(payload["results_payload"]),
            diagnostics=dict(payload["diagnostics"]),
            cache_key=str(payload["cache_key"]),
        )

    @property
    def total_cv(self) -> float:
        return float(sum(max(value, 0.0) for value in self.constraint_violations.values()))


@dataclass
class SingleStageFrontierRuntime:
    args: argparse.Namespace
    spec: SingleStageFrontierEvaluatorSpec
    constraint_method: str
    curves: list[object]
    surface_data: list[dict[str, object]]
    vessel_surface: object | None
    bfield: object
    objective_bundle: dict[str, object]
    objective_optimizable: object
    initial_surface_state: dict[str, list[object]]
    surface_weights: np.ndarray
    search_gate: dict[str, object]
    surface_mode_contract: object
    stage2_tf_current_A: float
    banana_coils: list[object]
    banana_current_max_A: float
    length_target: float
    curvature_threshold: float
    curve_curve_distance_threshold: float
    curve_surface_distance_threshold: float
    surface_vessel_distance_threshold: float
    reference_metrics: dict[str, float | None]
    trust_threshold: float | None
    trust_penalty_scale: float | None


def objective_vector_for_minimization(
    objective_metrics: Mapping[str, float],
) -> list[float]:
    vector: list[float] = []
    for metric_name, direction, _ in PARETO_OBJECTIVE_SPECS:
        value = float(objective_metrics[metric_name])
        vector.append(-value if direction == "max" else value)
    return vector


class SingleStageFrontierEvaluator:
    def __init__(
        self,
        spec: SingleStageFrontierEvaluatorSpec,
        *,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.spec = spec
        self.cache_dir = None if cache_dir is None else Path(cache_dir)
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.runtime = build_single_stage_frontier_runtime(spec)
        self._cache: dict[str, SingleStageFrontierEvaluation] = {}
        self.cache_hits = 0
        self.cache_misses = 0

    @classmethod
    def from_spec(
        cls,
        spec: SingleStageFrontierEvaluatorSpec,
        *,
        cache_dir: str | Path | None = None,
    ) -> SingleStageFrontierEvaluator:
        return cls(spec, cache_dir=cache_dir)

    def evaluate(
        self,
        x: Sequence[float],
    ) -> SingleStageFrontierEvaluation:
        candidate, cache_key = self._prepare_candidate(x)
        cached = self._load_cached(cache_key)
        if cached is not None:
            self.cache_hits += 1
            return cached
        self.cache_misses += 1
        evaluation = self._evaluate_uncached(candidate, cache_key=cache_key)
        self._store_cached(cache_key, evaluation)
        return evaluation

    def evaluate_batch(
        self,
        X: Sequence[Sequence[float]],
    ) -> list[SingleStageFrontierEvaluation]:
        evaluations: list[SingleStageFrontierEvaluation | None] = [None] * len(X)
        pending_by_cache_key: dict[str, tuple[np.ndarray, list[int]]] = {}

        for index, x in enumerate(X):
            candidate, cache_key = self._prepare_candidate(x)
            cached = self._load_cached(cache_key)
            if cached is not None:
                self.cache_hits += 1
                evaluations[index] = cached
                continue
            pending = pending_by_cache_key.get(cache_key)
            if pending is None:
                pending_by_cache_key[cache_key] = (candidate, [index])
                continue
            pending[1].append(index)

        for cache_key, (candidate, positions) in pending_by_cache_key.items():
            self.cache_misses += 1
            evaluation = self._evaluate_uncached(candidate, cache_key=cache_key)
            self._store_cached(cache_key, evaluation)
            for index in positions:
                evaluations[index] = evaluation

        if any(evaluation is None for evaluation in evaluations):
            raise RuntimeError(
                "Batch evaluation failed to populate every candidate result."
            )

        return [
            evaluation
            for evaluation in evaluations
            if evaluation is not None
        ]

    def _prepare_candidate(
        self,
        x: Sequence[float],
    ) -> tuple[np.ndarray, str]:
        candidate = np.asarray(x, dtype=float)
        if candidate.shape != (len(self.spec.lower_bounds),):
            raise ValueError(
                "Evaluator candidate length does not match the explicit decision-variable contract."
            )
        return candidate, self._cache_key(candidate)

    def _cache_key(self, candidate: np.ndarray) -> str:
        digest = hashlib.sha256()
        digest.update(self.spec.fingerprint().encode("utf-8"))
        digest.update(np.asarray(candidate, dtype=np.float64).tobytes())
        return digest.hexdigest()

    def _cache_path(self, cache_key: str) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{cache_key}.json"

    def _load_cached(
        self,
        cache_key: str,
    ) -> SingleStageFrontierEvaluation | None:
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        cache_path = self._cache_path(cache_key)
        if cache_path is None or not cache_path.exists():
            return None
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if payload.get("cache_schema_version") != FRONTIER_EVALUATOR_CACHE_SCHEMA_VERSION:
            return None
        evaluation = SingleStageFrontierEvaluation.from_json_dict(
            payload["evaluation"]
        )
        self._cache[cache_key] = evaluation
        return evaluation

    def _store_cached(
        self,
        cache_key: str,
        evaluation: SingleStageFrontierEvaluation,
    ) -> None:
        self._cache[cache_key] = evaluation
        cache_path = self._cache_path(cache_key)
        if cache_path is None:
            return
        payload = {
            "cache_schema_version": FRONTIER_EVALUATOR_CACHE_SCHEMA_VERSION,
            "evaluation": evaluation.to_json_dict(),
        }
        cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _evaluate_uncached(
        self,
        candidate: np.ndarray,
        *,
        cache_key: str,
    ) -> SingleStageFrontierEvaluation:
        runtime = self.runtime
        try:
            restore_surface_states(runtime.surface_data, runtime.initial_surface_state)
            runtime.objective_optimizable.x = np.asarray(runtime.spec.seed_x, dtype=float)
            stack_status = solve_surface_stack_at_dofs(
                candidate,
                runtime.objective_optimizable,
                runtime.surface_data,
                runtime.initial_surface_state,
                vessel_surface=runtime.vessel_surface,
                surface_gap_threshold=float(
                    runtime.search_gate["surface_gap_threshold"]
                ),
                vessel_gap_threshold=float(
                    runtime.search_gate["vessel_gap_threshold"]
                ),
                enforce_nesting=bool(runtime.search_gate["enforce_nesting"]),
            )
            search_eval = None
            if stack_status["success"]:
                search_eval = _evaluate_search_objective(runtime)
            hard_invalidation = evaluate_frontier_hard_invalidation(
                search_eval=search_eval,
                surface_success=bool(stack_status["success"]),
                surface_status=stack_status,
            )
            topology_status = (
                _evaluate_topology_status(runtime)
                if stack_status["success"] and search_eval is not None
                else single_stage.skipped_topology_gate_status()
            )
            topology_broken = bool(
                topology_status is not None
                and topology_status.get("enabled")
                and topology_status.get("success") is False
            )
            hardware_status = (
                _evaluate_hardware_status(runtime, stack_status)
                if stack_status["success"]
                else _failed_hardware_status()
            )
        except Exception as error:
            return _invalid_evaluation(
                runtime,
                candidate,
                cache_key=cache_key,
                reason="geometry_state_unrestorable",
                diagnostics={
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                },
            )

        if search_eval is None:
            return _invalid_evaluation(
                runtime,
                candidate,
                cache_key=cache_key,
                reason=(
                    hard_invalidation["reason"]
                    if hard_invalidation["invalid"]
                    else "missing_search_eval"
                ),
                diagnostics={
                    "surface_status": _jsonable_value(stack_status),
                    "topology_status": _jsonable_value(topology_status),
                    "hardware_status": _jsonable_value(hardware_status),
                },
            )

        iota_metric = float(runtime.objective_bundle["surface_iota_terms"][-1].J())
        volume_metric = float(runtime.objective_bundle["surface_volume_term"].J())
        trust_ok = search_eval.get("frontier_trust_ok")
        topology_success = (
            None
            if topology_status is None or not topology_status.get("enabled")
            else bool(topology_status.get("success"))
        )
        hardware_ok = bool(hardware_status["success"])
        hard_ok = (
            not hard_invalidation["invalid"]
            and not topology_broken
            and hardware_ok
            and trust_ok is not False
        )
        objective_metrics = {
            "iota": iota_metric,
            "volume": volume_metric,
            "qa_error": float(search_eval["J_QS"]),
            "boozer_residual": float(search_eval["J_Boozer"]),
        }
        violation_ratios = hardware_violation_ratios(hardware_status)
        constraint_violations = {
            "surface_solve_failed": 1.0
            if hard_invalidation["reason"] == "surface_solve_failed"
            else 0.0,
            "geometry_state_unrestorable": 1.0
            if hard_invalidation["reason"] == "geometry_state_unrestorable"
            else 0.0,
            "missing_search_eval": 1.0
            if hard_invalidation["reason"] == "missing_search_eval"
            else 0.0,
            "nonfinite_evaluation": float(
                max(len(search_eval.get("nonfinite_fields", [])), 0)
            )
            if hard_invalidation["reason"] == "nonfinite_evaluation"
            else 0.0,
            "topology_broken": 1.0 if topology_broken else 0.0,
            "topology_deficit": float(
                0.0
                if topology_status is None
                else topology_gate_deficit(topology_status)
            ),
            "hardware_violation_ratio": float(
                max(violation_ratios.values(), default=0.0)
            ),
            "frontier_trust_excess_ratio": float(
                search_eval.get("frontier_boozer_trust_excess_ratio", 0.0) or 0.0
            ),
        }
        diagnostics = {
            "surface_status": _jsonable_value(stack_status),
            "topology_status": _jsonable_value(topology_status),
            "hardware_status": _jsonable_value(hardware_status),
            "hard_invalidation": dict(hard_invalidation),
            "objective_vector_minimize": objective_vector_for_minimization(
                objective_metrics
            ),
        }
        results_payload = {
            "FINAL_IOTA": objective_metrics["iota"],
            "FINAL_VOLUME": objective_metrics["volume"],
            "NONQS_RATIO": objective_metrics["qa_error"],
            "BOOZER_RESIDUAL": objective_metrics["boozer_residual"],
            "FRONTIER_REFERENCE_IOTA": runtime.reference_metrics["iota"],
            "FRONTIER_REFERENCE_VOLUME": runtime.reference_metrics["volume"],
            "FRONTIER_REFERENCE_QA": runtime.reference_metrics["qa_error"],
            "FRONTIER_REFERENCE_BOOZER": runtime.reference_metrics["boozer_residual"],
            "COIL_LENGTH": hardware_status.get("coil_length"),
            "CURVE_CURVE_MIN_DIST": hardware_status.get("curve_curve_min_dist"),
            "CURVE_SURFACE_MIN_DIST": hardware_status.get("curve_surface_min_dist"),
            "SURFACE_VESSEL_MIN_DIST": hardware_status.get("surface_vessel_min_dist"),
            "MAX_CURVATURE": hardware_status.get("max_curvature"),
            "HARDWARE_CONSTRAINTS_OK": hardware_ok,
            "FINAL_FEASIBILITY_OK": hard_ok,
            "FINAL_TOPOLOGY_GATE_SUCCESS": topology_success,
            "FRONTIER_TRUST_OK": trust_ok,
            "FRONTIER_RANK_OBJECTIVE_J": search_eval.get(
                "frontier_rank_total",
                search_eval.get("total"),
            ),
            "SEARCH_OBJECTIVE_J": search_eval.get("total"),
            "OPTIMIZER_SUCCESS": hard_ok,
            "TERMINATION_MESSAGE": "evaluator_candidate_valid"
            if hard_ok
            else "evaluator_candidate_invalid",
        }
        return SingleStageFrontierEvaluation(
            schema_version=FRONTIER_EVALUATION_SCHEMA_VERSION,
            candidate_id=cache_key[:16],
            x=candidate.tolist(),
            valid=hard_ok,
            objective_metrics=objective_metrics,
            reference_metrics=dict(runtime.reference_metrics),
            constraint_violations=constraint_violations,
            results_payload=results_payload,
            diagnostics=diagnostics,
            cache_key=cache_key,
        )


def build_single_stage_frontier_runtime(
    spec: SingleStageFrontierEvaluatorSpec,
) -> SingleStageFrontierRuntime:
    args = single_stage.apply_default_stage2_seed_args(
        argparse.Namespace(**dict(spec.args_payload))
    )
    if (
        getattr(args, "single_stage_banana_current_mode", BANANA_CURRENT_MODE_SHARED)
        != BANANA_CURRENT_MODE_SHARED
    ):
        raise FrontierEvaluatorInitializationError(
            "single-stage frontier evaluator does not support "
            "--single-stage-banana-current-mode=independent; the evaluator "
            "runtime still assumes a scalar banana-current contract."
        )
    stage2_results = dict(spec.stage2_results)
    stage2_bs_path = Path(spec.stage2_bs_path)

    try:
        single_stage.validate_stage2_seed_contract(stage2_results)
        R0 = single_stage.validate_major_radius(
            float(stage2_results["MAJOR_RADIUS"]),
            accept_offspec=bool(args.accept_offspec_r0_seed),
        )
        surface_mode_contract = single_stage.resolve_surface_mode_contract(args)
        single_stage.validate_surface_mode_runtime_support(surface_mode_contract)
        effective_num_surfaces = surface_mode_contract.num_surfaces
        effective_inner_surface_ratio = single_stage.resolve_surface_mode_inner_surface_ratio(
            surface_mode_contract,
            fallback_inner_surface_ratio=args.inner_surface_ratio,
        )
        finite_current_mode = single_stage.resolve_stage2_finite_current_mode(
            stage2_results,
            getattr(args, "finite_current_mode", single_stage.DEFAULT_FINITE_CURRENT_MODE),
        )
        plasma_current_settings = single_stage.resolve_plasma_current_settings(
            args,
            finite_current_mode=finite_current_mode,
            default_plasma_current_A=float(
                stage2_results.get("PROXY_PLASMA_CURRENT_A", 0.0)
            ),
            num_surfaces=effective_num_surfaces,
        )
        file_loc = single_stage.build_equilibrium_path(args)
        surface_configs = single_stage.build_surface_configs(
            file_loc,
            args.nphi,
            args.ntheta,
            float(stage2_results["TOROIDAL_FLUX"]),
            R0,
            args.vol_target,
            effective_num_surfaces,
            effective_inner_surface_ratio,
        )
        num_tf_coils = single_stage.resolve_stage2_num_tf_coils(
            stage2_results,
            args.num_tf_coils,
        )
        bs, coil_partitions = single_stage.load_stage2_seed_biot_savart(
            stage2_bs_path,
            stage2_results=stage2_results,
            num_tf_coils=num_tf_coils,
            seed_order_upgrade=getattr(args, "seed_order_upgrade", None),
        )
        tf_coils = list(coil_partitions.tf_coils)
        banana_coils = list(coil_partitions.banana_coils)
        proxy_coils = list(coil_partitions.proxy_coils)
        vf_coils = list(coil_partitions.vf_coils)
        del proxy_coils, vf_coils
        coils = bs.coils
        curves = [coil.curve for coil in coils]
        banana_curves = [coil.curve for coil in banana_coils]
        banana_curve = banana_curves[0]
        (
            vessel_surface,
            _lcfs_clearance_reference,
            _surf_coils,
        ) = single_stage.build_hbt_reference_surfaces(
            banana_curve.surf.nfp,
            single_stage.resolve_single_stage_banana_surf_radius(
                stage2_results,
                args.banana_surf_radius,
            ),
        )
        boozer_I = plasma_current_settings["boozer_I"]
        G0 = single_stage.compute_tf_G0(tf_coils)
        surface_data: list[dict[str, object]] = []
        for config in surface_configs:
            boozer_surface = single_stage.initialize_boozer_surface(
                config["initial_surface"],
                args.mpol,
                args.ntor,
                bs,
                config["target_volume"],
                None if args.constraint_weight < 0 else args.constraint_weight,
                args.iota_target,
                G0,
                boozer_I,
                nfp=banana_curve.surf.nfp,
            )
            surface_data.append(
                {
                    "name": config["name"],
                    "seed_label": config["seed_label"],
                    "target_volume": config["target_volume"],
                    "boozer_surface": boozer_surface,
                }
            )
        outer_surface_data = surface_data[-1]
        initial_iota = Iotas(outer_surface_data["boozer_surface"]).J()
        initial_volume = outer_surface_data["boozer_surface"].surface.volume()
        initial_qs_objective, initial_boozer_objective = (
            single_stage.measure_frontier_reference_metrics(
                args.boozer_stage,
                surface_data,
                coils,
            )
        )
        allow_offspec = bool(args.allow_offspec_engineering_constraints)
        length_target = (
            float(args.length_target)
            if allow_offspec
            else min(
                float(args.length_target),
                float(single_stage.COIL_LENGTH_HARD_LIMIT_M),
            )
        )
        curvature_threshold = (
            float(args.curvature_threshold)
            if allow_offspec
            else min(
                float(args.curvature_threshold),
                float(single_stage.MAX_CURVATURE_INV_M),
            )
        )
        frontier_goal_config = None
        if args.single_stage_goal_mode == "frontier":
            frontier_goal_config = single_stage.build_frontier_goal_config(
                initial_iota=initial_iota,
                initial_volume=initial_volume,
                initial_qs_objective=initial_qs_objective,
                initial_boozer_objective=initial_boozer_objective,
                res_weight=args.res_weight,
                iotas_weight=args.iotas_weight,
                volume_weight=args.frontier_volume_weight,
                iota_reference_override=args.frontier_reference_iota,
                iota_scale_override=args.frontier_reference_iota_scale,
                volume_reference_override=args.frontier_reference_volume,
                volume_scale_override=args.frontier_reference_volume_scale,
                qs_reference_override=args.frontier_reference_qa,
                boozer_reference_override=args.frontier_reference_boozer,
                boozer_trust_threshold_override=args.frontier_boozer_trust_threshold,
                boozer_trust_penalty_scale_override=args.frontier_boozer_trust_penalty_scale,
                scalarization_type=args.frontier_scalarization_type,
                chebyshev_rho_override=args.frontier_chebyshev_rho,
                chebyshev_sharpness_override=args.frontier_chebyshev_sharpness,
                chebyshev_weight_iota_override=args.frontier_chebyshev_weight_iota,
                chebyshev_weight_volume_override=args.frontier_chebyshev_weight_volume,
                chebyshev_weight_qa_override=args.frontier_chebyshev_weight_qa,
                chebyshev_weight_boozer_override=args.frontier_chebyshev_weight_boozer,
                epsilon_constraint_qa_max_override=args.epsilon_constraint_qa_max,
                epsilon_constraint_boozer_max_override=args.epsilon_constraint_boozer_max,
                epsilon_penalty_weight_override=args.frontier_epsilon_penalty_weight,
            )
        objective_bundle = single_stage.build_single_stage_objective_bundle(
            args.boozer_stage,
            surface_data,
            coils,
            banana_curves,
            banana_curves,
            args.iota_target,
            args.res_weight,
            args.iotas_weight,
            args.length_weight,
            args.cc_weight,
            max(args.cc_dist, single_stage.COIL_COIL_MIN_DIST_M),
            args.cs_weight,
            max(args.cs_dist, single_stage.COIL_PLASMA_MIN_DIST_M),
            args.curvature_weight,
            curvature_threshold,
            length_target=length_target,
            SURF_DIST_WEIGHT=args.surf_dist_weight,
            vessel_surface=vessel_surface,
            vessel_gap_threshold=max(
                args.ss_dist,
                single_stage.PLASMA_VESSEL_MIN_DIST_M,
            ),
            goal_mode=args.single_stage_goal_mode,
            frontier_goal_config=frontier_goal_config,
        )
        objective_optimizable = objective_bundle["JF"]
        surface_weights = build_surface_search_weights(
            len(surface_data),
            0,
            args.multisurface_ramp_iterations,
            args.inner_surface_initial_weight,
        )
        search_gate = build_surface_search_gate(
            len(surface_data),
            0,
            args.multisurface_ramp_iterations,
            args.inner_surface_initial_weight,
            max(args.surface_gap_threshold, 0.0),
            max(args.ss_dist, single_stage.PLASMA_VESSEL_MIN_DIST_M),
        )
        seed_state = snapshot_surface_states(surface_data)
        run_identity = single_stage.build_run_identity_config(
            single_stage.make_run_identity_config(
                args,
                str(stage2_bs_path),
                args.boozer_stage,
                None if args.constraint_weight < 0 else args.constraint_weight,
                args.constraint_method,
                args.vol_target,
                args.iota_target,
                plasma_current_settings["boozer_I"],
                plasma_current_settings["plasma_current_A"],
                single_stage.resolve_single_stage_banana_surf_radius(
                    stage2_results,
                    args.banana_surf_radius,
                ),
                args.nphi,
                args.ntheta,
                None,
                surface_mode_contract=surface_mode_contract,
                effective_num_surfaces=effective_num_surfaces,
                effective_inner_surface_ratio=effective_inner_surface_ratio,
            )
        )
        dof_names = [str(name) for name in objective_optimizable.dof_names]
        lower_bounds = np.asarray(objective_optimizable.lower_bounds, dtype=float)
        upper_bounds = np.asarray(objective_optimizable.upper_bounds, dtype=float)
        seed_x = np.asarray(objective_optimizable.x, dtype=float)
        reference_metrics = {
            "iota": float(initial_iota),
            "volume": float(initial_volume),
            "qa_error": float(initial_qs_objective),
            "boozer_residual": float(initial_boozer_objective),
        }
        spec_from_runtime = SingleStageFrontierEvaluatorSpec(
            schema_version=FRONTIER_EVALUATOR_SPEC_SCHEMA_VERSION,
            args_payload={
                key: _jsonable_value(value) for key, value in vars(args).items()
            },
            stage2_bs_path=str(stage2_bs_path),
            stage2_results_path=spec.stage2_results_path,
            stage2_results=stage2_results,
            run_identity=run_identity,
            decision_variables=_build_decision_variable_specs(
                dof_names,
                lower_bounds,
                upper_bounds,
            ),
            lower_bounds=lower_bounds.tolist(),
            upper_bounds=upper_bounds.tolist(),
            seed_x=seed_x.tolist(),
            reference_metrics=reference_metrics,
            cv_bucket_names=list(FRONTIER_EVALUATOR_CV_BUCKETS),
            surface_weight_schedule=surface_weights.tolist(),
            search_gate=_jsonable_value(search_gate),
        )
        resolved_spec = spec_from_runtime
        if spec.run_identity:
            if spec_from_runtime.to_json_dict() != spec.to_json_dict():
                raise FrontierEvaluatorInitializationError(
                    "Re-instantiated evaluator drifted from the serialized contract."
                )
            resolved_spec = spec
        return SingleStageFrontierRuntime(
            args=args,
            spec=resolved_spec,
            constraint_method=str(args.constraint_method),
            curves=curves,
            surface_data=surface_data,
            vessel_surface=vessel_surface,
            bfield=bs,
            objective_bundle=objective_bundle,
            objective_optimizable=objective_optimizable,
            initial_surface_state=seed_state,
            surface_weights=surface_weights,
            search_gate=search_gate,
            surface_mode_contract=surface_mode_contract,
            stage2_tf_current_A=float(
                single_stage.resolve_stage2_tf_current_A(stage2_results, tf_coils)
            ),
            banana_coils=banana_coils,
            banana_current_max_A=float(args.banana_current_max_A),
            length_target=length_target,
            curvature_threshold=curvature_threshold,
            curve_curve_distance_threshold=max(
                float(args.cc_dist),
                float(single_stage.COIL_COIL_MIN_DIST_M),
            ),
            curve_surface_distance_threshold=max(
                float(args.cs_dist),
                float(single_stage.COIL_PLASMA_MIN_DIST_M),
            ),
            surface_vessel_distance_threshold=max(
                float(args.ss_dist),
                float(single_stage.PLASMA_VESSEL_MIN_DIST_M),
            ),
            reference_metrics=reference_metrics,
            trust_threshold=None
            if frontier_goal_config is None
            else float(frontier_goal_config.boozer_trust_threshold),
            trust_penalty_scale=None
            if frontier_goal_config is None
            else float(frontier_goal_config.boozer_trust_penalty_scale),
        )
    except FrontierEvaluatorInitializationError:
        raise
    except Exception as error:
        raise FrontierEvaluatorInitializationError(
            f"{type(error).__name__}: {error}"
        ) from error


def build_single_stage_frontier_evaluator_spec(
    args: argparse.Namespace,
    *,
    stage2_bs_path: Path,
    stage2_results_path: Path | None,
    stage2_results: Mapping[str, object],
) -> SingleStageFrontierEvaluatorSpec:
    args_copy = argparse.Namespace(**vars(args))
    args_payload = {
        key: _jsonable_value(value)
        for key, value in vars(
            single_stage.apply_default_stage2_seed_args(args_copy)
        ).items()
    }
    placeholder = SingleStageFrontierEvaluatorSpec(
        schema_version=FRONTIER_EVALUATOR_SPEC_SCHEMA_VERSION,
        args_payload=args_payload,
        stage2_bs_path=str(stage2_bs_path),
        stage2_results_path=None
        if stage2_results_path is None
        else str(stage2_results_path),
        stage2_results=dict(stage2_results),
        run_identity="",
        decision_variables=[],
        lower_bounds=[],
        upper_bounds=[],
        seed_x=[],
        reference_metrics={},
        cv_bucket_names=list(FRONTIER_EVALUATOR_CV_BUCKETS),
        surface_weight_schedule=[],
        search_gate={},
    )
    runtime = build_single_stage_frontier_runtime(placeholder)
    return runtime.spec


def _evaluate_search_objective(
    runtime: SingleStageFrontierRuntime,
) -> dict[str, object]:
    bundle = runtime.objective_bundle
    alm_multipliers = None
    alm_penalty = None
    if runtime.constraint_method == "alm":
        alm_multipliers = np.zeros(
            len(
                single_stage.single_stage_alm_constraint_names(
                    alm_formulation=runtime.args.alm_formulation,
                    include_surface_surface=bundle["JSurfSurf"] is not None,
                )
            ),
            dtype=float,
        )
        alm_penalty = float(runtime.args.alm_penalty_init)
        raw_eval = evaluate_alm_objective(
            runtime.surface_weights,
            bundle["nonQSs"],
            bundle["brs"],
            bundle["effective_res_weight"],
            bundle["Jiota"],
            bundle["effective_iotas_weight"],
            bundle["JCurveLength"],
            runtime.args.length_weight,
            bundle["JCurveCurve"],
            bundle["JCurveSurface"],
            bundle["JCurvature"],
            alm_multipliers,
            alm_penalty,
            objective_optimizable=runtime.objective_optimizable,
            curves=runtime.curves,
            curve_curve_min_distance=runtime.curve_curve_distance_threshold,
            outer_surface=runtime.surface_data[-1]["boozer_surface"].surface,
            curve_surface_min_distance=runtime.curve_surface_distance_threshold,
            banana_curve=runtime.banana_coils[0].curve,
            curvature_threshold=runtime.curvature_threshold,
            distance_smoothing=runtime.args.alm_distance_smoothing,
            curvature_smoothing=runtime.args.alm_curvature_smoothing,
            constraint_names=single_stage.single_stage_alm_constraint_names(
                alm_formulation=runtime.args.alm_formulation,
                include_surface_surface=bundle["JSurfSurf"] is not None,
            ),
            curve_curve_constraint_fn=single_stage._smooth_min_curve_curve_signed_constraint,
            curve_surface_constraint_fn=single_stage._smooth_min_curve_surface_signed_constraint,
            curvature_constraint_fn=single_stage._smooth_max_curvature_signed_constraint,
            JSurfSurf=bundle["JSurfSurf"],
            vessel_surface=runtime.vessel_surface,
            surface_surface_min_distance=runtime.surface_vessel_distance_threshold,
            surface_surface_constraint_fn=single_stage._smooth_min_surface_surface_signed_constraint,
            alm_formulation=runtime.args.alm_formulation,
            qs_threshold=runtime.args.alm_qs_threshold,
            boozer_threshold=runtime.args.alm_boozer_threshold,
            iota_penalty_threshold=runtime.args.alm_iota_penalty_threshold,
            length_penalty_threshold=runtime.args.alm_length_penalty_threshold,
            coil_length_objective=bundle["curvelength"],
            coil_length_threshold=runtime.length_target,
            banana_current=runtime.banana_coils[0].current,
            banana_current_threshold=runtime.banana_current_max_A,
            JNonQSObjective=bundle["JnonQSRatioObjective"],
            JBoozerObjective=bundle["JBoozerResidualObjective"],
        )
    else:
        raw_eval = evaluate_total_objective(
            runtime.surface_weights,
            bundle["nonQSs"],
            bundle["brs"],
            bundle["effective_res_weight"],
            bundle["Jiota"],
            bundle["effective_iotas_weight"],
            bundle["JCurveLength"],
            runtime.args.length_weight,
            bundle["JCurveCurve"],
            runtime.args.cc_weight,
            bundle["JCurveSurface"],
            runtime.args.cs_weight,
            bundle["JCurvature"],
            runtime.args.curvature_weight,
            JSurfSurf=bundle["JSurfSurf"],
            SURF_DIST_WEIGHT=runtime.args.surf_dist_weight,
            JNonQSObjective=bundle["JnonQSRatioObjective"],
            JBoozerObjective=bundle["JBoozerResidualObjective"],
            JVolume=bundle["JVolume"],
            VOLUME_WEIGHT=bundle["effective_volume_weight"],
            objective_optimizable=runtime.objective_optimizable,
        )
    scalarized_eval = apply_frontier_scalarization_override(
        raw_eval,
        enabled=runtime.args.single_stage_goal_mode == "frontier",
        frontier_goal_config=bundle["frontier_goal_config"],
        surface_iota_term=bundle["surface_iota_terms"][-1],
        surface_volume_term=bundle["surface_volume_term"],
        effective_res_weight=bundle["effective_res_weight"],
        effective_iotas_weight=bundle["effective_iotas_weight"],
        effective_volume_weight=bundle["effective_volume_weight"],
        length_weight=runtime.args.length_weight,
        cc_weight=runtime.args.cc_weight,
        cs_weight=runtime.args.cs_weight,
        curvature_weight=runtime.args.curvature_weight,
        surf_dist_weight=runtime.args.surf_dist_weight,
        objective_optimizable=runtime.objective_optimizable,
        alm_formulation=(
            runtime.args.alm_formulation
            if runtime.constraint_method == "alm"
            else "weighted_sum"
        ),
        alm_multipliers=alm_multipliers,
        alm_penalty=alm_penalty,
    )
    return annotate_frontier_search_eval(
        scalarized_eval,
        enabled=runtime.trust_threshold is not None,
        threshold=runtime.trust_threshold,
        penalty_scale=runtime.trust_penalty_scale,
    )


def _evaluate_topology_status(
    runtime: SingleStageFrontierRuntime,
) -> dict[str, object] | None:
    if int(runtime.args.topology_gate_fieldlines) <= 0:
        return single_stage.disabled_topology_gate_status(
            runtime.args.topology_gate_tmax,
            runtime.args.topology_gate_tol,
            runtime.args.topology_gate_survival_threshold,
        )
    if not single_stage.surface_mode_supports_topology_gate(
        runtime.surface_mode_contract
    ):
        return single_stage.disabled_topology_gate_status(
            runtime.args.topology_gate_tmax,
            runtime.args.topology_gate_tol,
            runtime.args.topology_gate_survival_threshold,
        )
    return single_stage.safe_evaluate_topology_gate(
        runtime.surface_data[-1]["boozer_surface"].surface,
        runtime.bfield,
        nfieldlines=runtime.args.topology_gate_fieldlines,
        tmax=runtime.args.topology_gate_tmax,
        tol=runtime.args.topology_gate_tol,
        survival_threshold=runtime.args.topology_gate_survival_threshold,
    )


def _evaluate_hardware_status(
    runtime: SingleStageFrontierRuntime,
    stack_status: Mapping[str, object],
) -> dict[str, object]:
    bundle = runtime.objective_bundle
    banana_curve = runtime.banana_coils[0].curve
    banana_current_A = float(runtime.banana_coils[0].current.get_value())
    snapshot = evaluate_single_stage_hardware_snapshot(
        bundle["JCurveCurve"],
        runtime.curve_curve_distance_threshold,
        bundle["JCurveSurface"],
        runtime.curve_surface_distance_threshold,
        bundle["JSurfSurf"],
        stack_status,
        runtime.surface_vessel_distance_threshold,
        banana_curve,
        runtime.curvature_threshold,
        outer_surface=runtime.surface_data[-1]["boozer_surface"].surface,
        vessel_surface=runtime.vessel_surface,
        coil_length=float(bundle["curvelength"].J()),
        length_target=runtime.length_target,
        poloidal_extent_rad=single_stage.max_poloidal_extent_rad(
            banana_curve,
            single_stage.VACUUM_VESSEL_MAJOR_RADIUS_M,
        ),
        poloidal_extent_threshold_rad=single_stage.POLOIDAL_EXTENT_HALF_WIDTH_RAD,
        tf_current_A=runtime.stage2_tf_current_A,
        tf_current_limit_A=single_stage.TF_CURRENT_HARD_LIMIT_A,
        banana_current_A=banana_current_A,
        banana_current_max_A=runtime.banana_current_max_A,
    )
    return snapshot["search_hardware_status"]


def _failed_hardware_status() -> dict[str, object]:
    return {
        "success": False,
        "violations": ["surface_solve_failed"],
        "curve_curve_min_dist": None,
        "curve_surface_min_dist": None,
        "surface_vessel_min_dist": None,
        "max_curvature": None,
        "coil_length": None,
    }


def _invalid_evaluation(
    runtime: SingleStageFrontierRuntime,
    candidate: np.ndarray,
    *,
    cache_key: str,
    reason: str,
    diagnostics: Mapping[str, object],
) -> SingleStageFrontierEvaluation:
    constraint_violations = {
        bucket_name: 0.0 for bucket_name in FRONTIER_EVALUATOR_CV_BUCKETS
    }
    if reason in constraint_violations:
        constraint_violations[reason] = 1.0
    results_payload = {
        "FINAL_IOTA": runtime.reference_metrics["iota"],
        "FINAL_VOLUME": runtime.reference_metrics["volume"],
        "NONQS_RATIO": runtime.reference_metrics["qa_error"],
        "BOOZER_RESIDUAL": runtime.reference_metrics["boozer_residual"],
        "FRONTIER_REFERENCE_IOTA": runtime.reference_metrics["iota"],
        "FRONTIER_REFERENCE_VOLUME": runtime.reference_metrics["volume"],
        "FRONTIER_REFERENCE_QA": runtime.reference_metrics["qa_error"],
        "FRONTIER_REFERENCE_BOOZER": runtime.reference_metrics["boozer_residual"],
        "HARDWARE_CONSTRAINTS_OK": False,
        "FINAL_FEASIBILITY_OK": False,
        "FINAL_TOPOLOGY_GATE_SUCCESS": False if reason == "topology_broken" else None,
        "FRONTIER_TRUST_OK": False if reason == "frontier_trust_excess_ratio" else None,
        "OPTIMIZER_SUCCESS": False,
        "TERMINATION_MESSAGE": reason,
    }
    return SingleStageFrontierEvaluation(
        schema_version=FRONTIER_EVALUATION_SCHEMA_VERSION,
        candidate_id=cache_key[:16],
        x=candidate.tolist(),
        valid=False,
        objective_metrics={
            metric_name: float(runtime.reference_metrics[metric_name] or 0.0)
            for metric_name, _, _ in PARETO_OBJECTIVE_SPECS
        },
        reference_metrics=dict(runtime.reference_metrics),
        constraint_violations=constraint_violations,
        results_payload=results_payload,
        diagnostics=dict(diagnostics),
        cache_key=cache_key,
    )


def _build_decision_variable_specs(
    dof_names: Sequence[str],
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> list[SingleStageFrontierDecisionVariableSpec]:
    specs: list[SingleStageFrontierDecisionVariableSpec] = []
    for name, lower, upper in zip(
        dof_names,
        np.asarray(lower_bounds, dtype=float),
        np.asarray(upper_bounds, dtype=float),
        strict=True,
    ):
        match = _DOF_NAME_PATTERN.match(name)
        semantic_role = name if match is None else str(match.group("family"))
        harmonic_index = None if match is None else int(match.group("index"))
        specs.append(
            SingleStageFrontierDecisionVariableSpec(
                name=str(name),
                semantic_role=semantic_role,
                harmonic_index=harmonic_index,
                lower_bound=float(lower),
                upper_bound=float(upper),
            )
        )
    return specs


_jsonable_value = single_stage._jsonable_value
