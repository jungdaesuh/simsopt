from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from alm_utils import ALMSettings, augmented_inequality_objective, minimize_alm
from banana_opt.alm_benchmarking import (
    BENCHMARK_DIR_NAME,
    autoresearch_root_from_arg,
)


SCHEMA_VERSION = "alm_normalization_fixture_benchmark_v1"
FORMULATION_RAW = "raw_units"
FORMULATION_NORMALIZED = "normalized_units"
DEFAULT_FIXTURE_SEED = 0
SOLVER_CHECKOUT = Path(__file__).resolve().parents[3]
DEFAULT_INNER_OPTIONS = {
    "maxiter": 120,
    "ftol": 1.0e-12,
    "gtol": 1.0e-9,
    "maxls": 50,
}


@dataclass(frozen=True)
class ALMFixture:
    name: str
    description: str
    x0: tuple[float, ...]
    target: tuple[float, ...]
    upper_bounds: tuple[float, ...]
    objective_weights: tuple[float, ...]
    constraint_scales: tuple[float, ...]
    constraint_names: tuple[str, ...]
    constraint_blocks: tuple[str, ...]

    def scale_array(self) -> np.ndarray:
        return np.asarray(self.constraint_scales, dtype=float)

    def upper_bound_array(self) -> np.ndarray:
        return np.asarray(self.upper_bounds, dtype=float)

    def target_array(self) -> np.ndarray:
        return np.asarray(self.target, dtype=float)

    def objective_weight_array(self) -> np.ndarray:
        return np.asarray(self.objective_weights, dtype=float)

    def raw_signed_values(self, x: np.ndarray) -> np.ndarray:
        return self.scale_array() * (np.asarray(x, dtype=float) - self.upper_bound_array())

    def normalized_signed_values(self, x: np.ndarray) -> np.ndarray:
        return self.raw_signed_values(x) / self.scale_array()

    def raw_constraint_grads(self) -> list[np.ndarray]:
        dimension = len(self.x0)
        return [
            float(scale) * _unit_vector(index, dimension)
            for index, scale in enumerate(self.constraint_scales)
        ]

    def normalized_constraint_grads(self) -> list[np.ndarray]:
        dimension = len(self.x0)
        return [
            _unit_vector(index, dimension)
            for index in range(len(self.constraint_scales))
        ]


@dataclass(frozen=True)
class FixtureRun:
    row: dict[str, object]
    evaluation_count: int
    wall_time_s: float


def default_fixtures() -> tuple[ALMFixture, ...]:
    return (
        ALMFixture(
            name="two_scale_hardware_boundary",
            description="Two active upper-bound constraints with meter/current-like scale separation.",
            x0=(1.35, 1.35),
            target=(1.20, 1.20),
            upper_bounds=(1.0, 1.0),
            objective_weights=(1.0, 1.0),
            constraint_scales=(1.0, 16000.0),
            constraint_names=("length_m", "banana_current_a"),
            constraint_blocks=("geometry", "current"),
        ),
        ALMFixture(
            name="three_block_mixed_units",
            description="Geometry, curvature, and current constraints with heterogeneous raw units.",
            x0=(1.30, 1.30, 1.30),
            target=(1.18, 1.18, 1.18),
            upper_bounds=(1.0, 1.0, 1.0),
            objective_weights=(1.0, 1.0, 1.0),
            constraint_scales=(0.01, 40.0, 16000.0),
            constraint_names=("coil_surface_gap_m", "max_curvature_inv_m", "banana_current_a"),
            constraint_blocks=("geometry", "geometry", "current"),
        ),
    )


def benchmark_output_dir(autoresearch_root: Path) -> Path:
    return autoresearch_root / "artifact_exports" / BENCHMARK_DIR_NAME


def solver_commit(checkout: Path = SOLVER_CHECKOUT) -> str:
    return subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _unit_vector(index: int, dimension: int) -> np.ndarray:
    vector = np.zeros(dimension, dtype=float)
    vector[index] = 1.0
    return vector


def _base_objective(fixture: ALMFixture, x: np.ndarray) -> tuple[float, np.ndarray]:
    delta = np.asarray(x, dtype=float) - fixture.target_array()
    weights = fixture.objective_weight_array()
    return 0.5 * float(np.dot(weights, delta * delta)), weights * delta


def _evaluate_fixture(
    fixture: ALMFixture,
    x: np.ndarray,
    multipliers: np.ndarray,
    penalty: float,
    *,
    formulation: str,
) -> dict:
    base_value, base_grad = _base_objective(fixture, x)
    raw_signed = fixture.raw_signed_values(x)
    normalized_signed = raw_signed / fixture.scale_array()
    if formulation == FORMULATION_NORMALIZED:
        constraint_values = normalized_signed
        constraint_grads = fixture.normalized_constraint_grads()
        feasibility_values = np.maximum(normalized_signed, 0.0)
    elif formulation == FORMULATION_RAW:
        constraint_values = raw_signed
        constraint_grads = fixture.raw_constraint_grads()
        feasibility_values = np.maximum(raw_signed, 0.0)
    else:
        raise ValueError(f"unknown ALM fixture formulation: {formulation}")

    evaluation = augmented_inequality_objective(
        base_value,
        base_grad,
        constraint_values,
        constraint_grads,
        multipliers,
        penalty,
    )
    raw_violation = np.maximum(raw_signed, 0.0)
    normalized_violation = np.maximum(normalized_signed, 0.0)
    evaluation.update(
        {
            "feasibility_values": feasibility_values,
            "max_feasibility_violation": float(np.max(feasibility_values)),
            "max_violation": float(np.max(feasibility_values)),
            "raw_dual_update_values": raw_signed,
            "raw_feasibility_values": raw_violation,
            "raw_hard_signed_constraint_values": raw_signed,
            "raw_hard_violation_values": raw_violation,
            "raw_surrogate_signed_constraint_values": raw_signed,
            "normalized_signed_constraint_values": normalized_signed,
            "normalized_feasibility_values": normalized_violation,
            "constraint_scales": list(fixture.constraint_scales),
            "constraint_blocks": list(fixture.constraint_blocks),
            "constraint_scale_sources": ["fixture_scale"] * len(fixture.constraint_scales),
        }
    )
    return evaluation


def _max_positive(values: np.ndarray) -> float:
    return float(np.max(np.maximum(values, 0.0))) if values.size > 0 else 0.0


def _fixture_result_row(
    *,
    fixture: ALMFixture,
    formulation: str,
    result,
    evaluation_count: int,
    wall_time_s: float,
) -> dict[str, object]:
    raw_signed = fixture.raw_signed_values(np.asarray(result.x, dtype=float))
    normalized_signed = raw_signed / fixture.scale_array()
    return {
        "fixture": fixture.name,
        "formulation": formulation,
        "success": bool(result.success),
        "termination_reason": result.termination_reason,
        "message": result.message,
        "outer_iterations": int(result.outer_iterations),
        "inner_iterations": int(result.nit),
        "objective_eval_count": int(evaluation_count),
        "wall_time_s": float(wall_time_s),
        "final_base_objective": float(result.final_base_objective),
        "final_objective": float(result.final_objective),
        "final_raw_max_violation": _max_positive(raw_signed),
        "final_normalized_max_violation": _max_positive(normalized_signed),
        "final_stationarity_norm": float(result.final_stationarity_norm),
        "final_penalty": float(result.penalty),
        "multiplier_cap_binding": bool(result.multiplier_cap_binding),
        "x": [float(value) for value in result.x],
        "multipliers": [float(value) for value in result.multipliers],
    }


def run_fixture_case(
    fixture: ALMFixture,
    *,
    formulation: str,
    settings: ALMSettings,
    inner_options: Mapping[str, float | int],
) -> FixtureRun:
    evaluation_count = 0

    def evaluate_problem(x, multipliers, penalty):
        nonlocal evaluation_count
        evaluation_count += 1
        return _evaluate_fixture(
            fixture,
            np.asarray(x, dtype=float),
            np.asarray(multipliers, dtype=float),
            float(penalty),
            formulation=formulation,
        )

    start = time.perf_counter()
    result = minimize_alm(
        np.asarray(fixture.x0, dtype=float),
        fixture.constraint_names,
        evaluate_problem,
        settings,
        dict(inner_options),
    )
    wall_time_s = time.perf_counter() - start
    return FixtureRun(
        row=_fixture_result_row(
            fixture=fixture,
            formulation=formulation,
            result=result,
            evaluation_count=evaluation_count,
            wall_time_s=wall_time_s,
        ),
        evaluation_count=evaluation_count,
        wall_time_s=wall_time_s,
    )


def _comparison(raw_row: Mapping[str, object], normalized_row: Mapping[str, object]) -> dict[str, object]:
    return {
        "fixture": raw_row["fixture"],
        "raw_success": raw_row["success"],
        "normalized_success": normalized_row["success"],
        "raw_final_normalized_violation": raw_row["final_normalized_max_violation"],
        "normalized_final_normalized_violation": normalized_row["final_normalized_max_violation"],
        "raw_eval_count": raw_row["objective_eval_count"],
        "normalized_eval_count": normalized_row["objective_eval_count"],
        "eval_count_delta_raw_minus_normalized": (
            int(raw_row["objective_eval_count"])
            - int(normalized_row["objective_eval_count"])
        ),
        "raw_outer_iterations": raw_row["outer_iterations"],
        "normalized_outer_iterations": normalized_row["outer_iterations"],
        "raw_wall_time_s": raw_row["wall_time_s"],
        "normalized_wall_time_s": normalized_row["wall_time_s"],
        "raw_termination_reason": raw_row["termination_reason"],
        "normalized_termination_reason": normalized_row["termination_reason"],
    }


def run_fixture_benchmark(
    *,
    fixtures: Sequence[ALMFixture] | None = None,
    settings: ALMSettings | None = None,
    inner_options: Mapping[str, float | int] | None = None,
    seed: int = DEFAULT_FIXTURE_SEED,
) -> dict[str, object]:
    active_fixtures = tuple(default_fixtures() if fixtures is None else fixtures)
    run_solver_commit = solver_commit(SOLVER_CHECKOUT)
    alm_settings = (
        ALMSettings(
            max_outer_iterations=8,
            penalty_init=1.0,
            penalty_scale=10.0,
            penalty_max=1.0e8,
            feasibility_tol=1.0e-8,
            stationarity_tol=1.0e-8,
            multiplier_max=1.0e8,
        )
        if settings is None
        else settings
    )
    lbfgsb_options = (
        dict(DEFAULT_INNER_OPTIONS)
        if inner_options is None
        else dict(inner_options)
    )
    rows = []
    comparisons = []
    for fixture in active_fixtures:
        raw_run = run_fixture_case(
            fixture,
            formulation=FORMULATION_RAW,
            settings=alm_settings,
            inner_options=lbfgsb_options,
        )
        normalized_run = run_fixture_case(
            fixture,
            formulation=FORMULATION_NORMALIZED,
            settings=alm_settings,
            inner_options=lbfgsb_options,
        )
        rows.extend([raw_run.row, normalized_run.row])
        comparisons.append(_comparison(raw_run.row, normalized_run.row))

    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "solver_checkout": str(SOLVER_CHECKOUT),
        "solver_commit": run_solver_commit,
        "settings": {
            "seed": int(seed),
            "max_outer_iterations": alm_settings.max_outer_iterations,
            "penalty_init": alm_settings.penalty_init,
            "penalty_scale": alm_settings.penalty_scale,
            "penalty_max": alm_settings.penalty_max,
            "feasibility_tol": alm_settings.feasibility_tol,
            "stationarity_tol": alm_settings.stationarity_tol,
            "multiplier_max": alm_settings.multiplier_max,
        },
        "inner_options": lbfgsb_options,
        "fixture_rows": rows,
        "comparisons": comparisons,
    }


def write_fixture_benchmark(payload: Mapping[str, object], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run deterministic ALM raw-vs-normalized fixture benchmarks."
    )
    parser.add_argument(
        "--autoresearch-root",
        type=Path,
        default=None,
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=DEFAULT_FIXTURE_SEED)
    parser.add_argument("--stamp", default=utc_stamp())
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    payload = run_fixture_benchmark(seed=args.seed)
    if args.output is None:
        autoresearch_root = autoresearch_root_from_arg(args.autoresearch_root)
        output_path = (
            benchmark_output_dir(autoresearch_root)
            / f"fixture_benchmark_{args.stamp}.json"
        )
    else:
        output_path = args.output
    path = write_fixture_benchmark(payload, output_path)
    print(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "fixture_rows": len(payload["fixture_rows"]),
                "comparisons": len(payload["comparisons"]),
                "path": str(path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0
