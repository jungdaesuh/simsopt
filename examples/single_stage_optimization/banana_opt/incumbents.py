import copy
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SingleStageIncumbentState:
    x: np.ndarray
    surface_state: dict
    objective_total: float
    objective_grad: np.ndarray
    search_eval: dict
    surface_status: dict
    search_surface_status: dict
    accepted_hardware_status: dict | None
    topology_gate_status: dict


def single_stage_incumbent_state_to_json_dict(
    state: SingleStageIncumbentState,
) -> dict[str, object]:
    return {
        "x": _serialize_value(state.x),
        "surface_state": _serialize_value(state.surface_state),
        "objective_total": float(state.objective_total),
        "objective_grad": _serialize_value(state.objective_grad),
        "search_eval": _serialize_value(state.search_eval),
        "surface_status": _serialize_value(state.surface_status),
        "search_surface_status": _serialize_value(state.search_surface_status),
        "accepted_hardware_status": _serialize_value(state.accepted_hardware_status),
        "topology_gate_status": _serialize_value(state.topology_gate_status),
    }


def single_stage_incumbent_state_from_json_dict(
    payload: dict[str, object],
) -> SingleStageIncumbentState:
    return SingleStageIncumbentState(
        x=np.asarray(payload["x"], dtype=float),
        surface_state=_restore_value(payload["surface_state"]),
        objective_total=float(payload["objective_total"]),
        objective_grad=np.asarray(payload["objective_grad"], dtype=float),
        search_eval=_restore_value(payload["search_eval"]),
        surface_status=_restore_value(payload["surface_status"]),
        search_surface_status=_restore_value(payload["search_surface_status"]),
        accepted_hardware_status=_restore_value(payload["accepted_hardware_status"]),
        topology_gate_status=_restore_value(payload["topology_gate_status"]),
    )


def snapshot_single_stage_incumbent_state(run_dict) -> SingleStageIncumbentState:
    return SingleStageIncumbentState(
        x=np.asarray(run_dict["accepted_x"], dtype=float).copy(),
        surface_state=copy.deepcopy(run_dict["surface_state"]),
        objective_total=float(run_dict["J"]),
        objective_grad=np.asarray(run_dict["dJ"], dtype=float).copy(),
        search_eval=copy.deepcopy(run_dict["search_eval"]),
        surface_status=copy.deepcopy(run_dict["surface_status"]),
        search_surface_status=copy.deepcopy(run_dict["search_surface_status"]),
        accepted_hardware_status=copy.deepcopy(run_dict.get("accepted_hardware_status")),
        topology_gate_status=copy.deepcopy(run_dict["topology_gate_status"]),
    )


def restore_single_stage_incumbent_state(run_dict, incumbent: SingleStageIncumbentState) -> None:
    run_dict["accepted_x"] = np.asarray(incumbent.x, dtype=float).copy()
    run_dict["surface_state"] = copy.deepcopy(incumbent.surface_state)
    run_dict["J"] = float(incumbent.objective_total)
    run_dict["dJ"] = np.asarray(incumbent.objective_grad, dtype=float).copy()
    run_dict["search_eval"] = copy.deepcopy(incumbent.search_eval)
    run_dict["surface_status"] = copy.deepcopy(incumbent.surface_status)
    run_dict["search_surface_status"] = copy.deepcopy(incumbent.search_surface_status)
    run_dict["accepted_hardware_status"] = copy.deepcopy(incumbent.accepted_hardware_status)
    run_dict["topology_gate_status"] = copy.deepcopy(incumbent.topology_gate_status)
    run_dict.pop("last_successful_eval", None)
    run_dict.pop("last_successful_eval_weights", None)


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


def _restore_value(value):
    if isinstance(value, dict):
        return {
            str(key): _restore_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_restore_value(item) for item in value]
    return value
