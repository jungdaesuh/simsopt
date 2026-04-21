from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Mapping

import numpy as np

from .frontier_contracts import FRONTIER_SOLVER_CHECKPOINT_SCHEMA_VERSION
from .incumbents import (
    SingleStageIncumbentState,
    single_stage_incumbent_state_from_json_dict,
    single_stage_incumbent_state_to_json_dict,
)

DEFAULT_SOLVER_CHECKPOINT_JSON = "solver_state_checkpoint.json"


def solver_checkpoint_path(output_root: str | Path) -> Path:
    return Path(output_root) / DEFAULT_SOLVER_CHECKPOINT_JSON


def write_solver_checkpoint(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2)
        os.replace(tmp_path, str(path))
    except BaseException:
        os.unlink(tmp_path)
        raise


def load_solver_checkpoint(path: str | Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_solver_checkpoint_payload(payload)
    return payload


def build_solver_checkpoint_payload(
    *,
    goal_mode: str,
    constraint_method: str,
    stage2_bs_path: str,
    requested_maxiter: int,
    runtime_maxiter: int,
    accepted_iterations: int,
    accepted_boozer_stage: str,
    accepted_incumbent: SingleStageIncumbentState,
    best_accepted_incumbent: SingleStageIncumbentState | None,
    best_accepted_stage: str | None,
    best_accepted_metric: float | None,
    best_feasible_incumbent: SingleStageIncumbentState | None,
    best_feasible_stage: str | None,
    best_feasible_metric: float | None,
    out_dir_iter: str,
    run_counters: Mapping[str, object],
    alm_state: Mapping[str, object] | None = None,
    conditioning_seed_report: Mapping[str, object] | None = None,
    conditioning_first_accepted_report: Mapping[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": FRONTIER_SOLVER_CHECKPOINT_SCHEMA_VERSION,
        "goal_mode": goal_mode,
        "constraint_method": constraint_method,
        "stage2_bs_path": str(stage2_bs_path),
        "requested_maxiter": int(requested_maxiter),
        "runtime_maxiter": int(runtime_maxiter),
        "remaining_maxiter": max(int(runtime_maxiter) - int(accepted_iterations), 0),
        "accepted_iterations": int(accepted_iterations),
        "accepted_boozer_stage": str(accepted_boozer_stage),
        "accepted_incumbent": single_stage_incumbent_state_to_json_dict(
            accepted_incumbent
        ),
        "best_accepted_incumbent": None
        if best_accepted_incumbent is None
        else single_stage_incumbent_state_to_json_dict(best_accepted_incumbent),
        "best_accepted_stage": best_accepted_stage,
        "best_accepted_metric": _optional_float(best_accepted_metric),
        "best_feasible_incumbent": None
        if best_feasible_incumbent is None
        else single_stage_incumbent_state_to_json_dict(best_feasible_incumbent),
        "best_feasible_stage": best_feasible_stage,
        "best_feasible_metric": _optional_float(best_feasible_metric),
        "out_dir_iter": str(out_dir_iter),
        "run_counters": {
            str(key): _serialize_value(value)
            for key, value in run_counters.items()
        },
        "alm_state": None
        if alm_state is None
        else {
            str(key): _serialize_value(value)
            for key, value in alm_state.items()
        },
        "conditioning_seed_report": None
        if conditioning_seed_report is None
        else _serialize_value(dict(conditioning_seed_report)),
        "conditioning_first_accepted_report": None
        if conditioning_first_accepted_report is None
        else _serialize_value(dict(conditioning_first_accepted_report)),
    }
    validate_solver_checkpoint_payload(payload)
    return payload


def restore_incumbent_from_solver_checkpoint(
    payload: Mapping[str, object],
) -> SingleStageIncumbentState:
    return single_stage_incumbent_state_from_json_dict(
        dict(payload["accepted_incumbent"])
    )


def restore_optional_incumbent(
    payload: Mapping[str, object],
    field_name: str,
) -> SingleStageIncumbentState | None:
    incumbent_payload = payload.get(field_name)
    if incumbent_payload is None:
        return None
    return single_stage_incumbent_state_from_json_dict(dict(incumbent_payload))


def validate_solver_checkpoint_payload(payload: Mapping[str, object]) -> None:
    if payload.get("schema_version") != FRONTIER_SOLVER_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("Unexpected single-stage solver checkpoint schema_version")
    required_keys = (
        "goal_mode",
        "constraint_method",
        "stage2_bs_path",
        "requested_maxiter",
        "runtime_maxiter",
        "remaining_maxiter",
        "accepted_iterations",
        "accepted_boozer_stage",
        "accepted_incumbent",
        "run_counters",
        "out_dir_iter",
    )
    missing = [key for key in required_keys if key not in payload]
    if missing:
        raise ValueError(f"Missing single-stage solver checkpoint keys: {missing}")


def _optional_float(value) -> float | None:
    if value is None:
        return None
    return float(value)


def _serialize_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {
            str(key): _serialize_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_serialize_value(item) for item in value]
    if isinstance(value, tuple):
        return [_serialize_value(item) for item in value]
    return value
