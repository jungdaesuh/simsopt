from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Mapping

import numpy as np

from .incumbents import (
    SingleStageIncumbentState,
    single_stage_incumbent_state_from_json_dict,
    single_stage_incumbent_state_to_json_dict,
)

BANANA_CURRENT_REPLAY_CONTEXT_SCHEMA_VERSION = (
    "single_stage_banana_current_replay_context_v1"
)
BANANA_CURRENT_REPLAY_CONTEXT_FILENAME = "banana_current_replay_context.json"
BANANA_CURRENT_REJECTED_TRIAL_REPLAY_SCHEMA_VERSION = (
    "single_stage_banana_current_rejected_trial_replay_v1"
)
BANANA_CURRENT_REJECTED_TRIAL_REPLAY_FILENAME = (
    "banana_current_rejected_trial_replay.json"
)


def banana_current_replay_context_path(output_root: str | Path) -> Path:
    return Path(output_root) / BANANA_CURRENT_REPLAY_CONTEXT_FILENAME


def banana_current_rejected_trial_replay_path(output_root: str | Path) -> Path:
    return Path(output_root) / BANANA_CURRENT_REJECTED_TRIAL_REPLAY_FILENAME


def build_banana_current_replay_context_state() -> dict[str, object]:
    return {
        "schema_version": BANANA_CURRENT_REPLAY_CONTEXT_SCHEMA_VERSION,
        "replay_contract": None,
        "accepted_incumbents": {},
    }


def set_banana_current_replay_context_contract(
    context_state: dict[str, object],
    *,
    mode: str,
    num_control_currents: int,
    coordinate_dof_names,
    current_coordinate_scale_factors_A,
    seed_currents_A,
    configured_seed_currents_A=None,
) -> dict[str, object]:
    context_state["replay_contract"] = {
        "mode": str(mode),
        "num_control_currents": int(num_control_currents),
        "coordinate_dof_names": [str(name) for name in coordinate_dof_names],
        "current_coordinate_scale_factors_A": _float_list(
            current_coordinate_scale_factors_A
        ),
        "seed_currents_A": _float_list(seed_currents_A),
        "configured_seed_currents_A": (
            None
            if configured_seed_currents_A is None
            else _float_list(configured_seed_currents_A)
        ),
    }
    return context_state


def record_banana_current_replay_context_snapshot(
    context_state: dict[str, object],
    *,
    accepted_iteration: int,
    accepted_boozer_stage: str,
    incumbent: SingleStageIncumbentState,
) -> dict[str, object]:
    accepted_incumbents = context_state.setdefault("accepted_incumbents", {})
    if not isinstance(accepted_incumbents, dict):
        raise ValueError("banana-current replay context must store a dict of incumbents")
    accepted_incumbents[str(int(accepted_iteration))] = {
        "accepted_iteration": int(accepted_iteration),
        "accepted_boozer_stage": str(accepted_boozer_stage),
        "incumbent": single_stage_incumbent_state_to_json_dict(incumbent),
    }
    return context_state


def write_banana_current_replay_context_artifact(
    output_root: str | Path,
    context_state: Mapping[str, object],
) -> Path:
    artifact_path = banana_current_replay_context_path(output_root)
    _write_json_artifact(artifact_path, context_state)
    return artifact_path


def load_banana_current_replay_context(
    path: str | Path,
    *,
    require_replay_contract: bool = False,
) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_banana_current_replay_context(
        payload,
        require_replay_contract=require_replay_contract,
    )
    return payload


def validate_banana_current_replay_context(
    payload: Mapping[str, object],
    *,
    require_replay_contract: bool = False,
) -> None:
    if (
        payload.get("schema_version")
        != BANANA_CURRENT_REPLAY_CONTEXT_SCHEMA_VERSION
    ):
        raise ValueError("Unexpected banana-current replay context schema_version")
    accepted_incumbents = payload.get("accepted_incumbents")
    if not isinstance(accepted_incumbents, dict):
        raise ValueError("Missing banana-current replay accepted_incumbents")
    replay_contract = payload.get("replay_contract")
    if replay_contract is not None and not isinstance(replay_contract, Mapping):
        raise ValueError("Banana-current replay context replay_contract must be a mapping")
    if require_replay_contract and replay_contract is None:
        raise ValueError(
            "Banana-current replay context is missing replay_contract metadata."
        )


def validate_banana_current_replay_context_contract(
    context_state: Mapping[str, object],
    diagnostics_payload: Mapping[str, object],
) -> None:
    replay_contract = context_state.get("replay_contract")
    if replay_contract is None:
        return
    contract = dict(replay_contract)
    if str(contract["mode"]) != str(diagnostics_payload["mode"]):
        raise ValueError(
            "Banana-current replay context mode does not match diagnostics mode."
        )
    if int(contract["num_control_currents"]) != int(
        diagnostics_payload["num_control_currents"]
    ):
        raise ValueError(
            "Banana-current replay context control-count does not match diagnostics."
        )
    diagnostics_seed_report = dict(diagnostics_payload["seed_report"])
    diagnostics_dof_names = tuple(diagnostics_seed_report["coordinate_dof_names"])
    context_dof_names = tuple(contract["coordinate_dof_names"])
    if context_dof_names != diagnostics_dof_names:
        raise ValueError(
            "Banana-current replay context DOF names do not match diagnostics."
        )
    _require_matching_float_vectors(
        contract["current_coordinate_scale_factors_A"],
        diagnostics_seed_report["current_coordinate_scale_factors_A"],
        mismatch_message=(
            "Banana-current replay context scale factors do not match diagnostics."
        ),
    )
    _require_matching_float_vectors(
        contract["seed_currents_A"],
        diagnostics_payload["seed_currents_A"],
        mismatch_message=(
            "Banana-current replay context seed currents do not match diagnostics."
        ),
    )
    configured_seed_currents_A = contract.get("configured_seed_currents_A")
    diagnostics_configured_seed_currents_A = diagnostics_payload.get(
        "configured_seed_currents_A"
    )
    if (
        configured_seed_currents_A is not None
        and diagnostics_configured_seed_currents_A is not None
    ):
        _require_matching_float_vectors(
            configured_seed_currents_A,
            diagnostics_configured_seed_currents_A,
            mismatch_message=(
                "Banana-current replay context configured seed currents do not "
                "match diagnostics."
            ),
        )


def restore_banana_current_replay_incumbent(
    context_state: Mapping[str, object],
    accepted_iteration: int,
) -> tuple[str, SingleStageIncumbentState]:
    validate_banana_current_replay_context(context_state)
    accepted_incumbents = dict(context_state["accepted_incumbents"])
    entry = dict(accepted_incumbents[str(int(accepted_iteration))])
    return (
        str(entry["accepted_boozer_stage"]),
        single_stage_incumbent_state_from_json_dict(dict(entry["incumbent"])),
    )


def validate_banana_current_replay_coordinate_contract(
    seed_report: Mapping[str, object],
    *,
    live_dof_names,
    live_scale_factors_A,
) -> None:
    expected_dof_names = tuple(str(name) for name in seed_report["coordinate_dof_names"])
    resolved_live_dof_names = tuple(str(name) for name in live_dof_names)
    if expected_dof_names != resolved_live_dof_names:
        raise ValueError(
            "Banana-current replay diagnostics DOF names do not match the live "
            f"banana-current coordinate contract: {expected_dof_names!r} vs "
            f"{resolved_live_dof_names!r}."
        )
    _require_matching_float_vectors(
        seed_report["current_coordinate_scale_factors_A"],
        live_scale_factors_A,
        mismatch_message=(
            "Banana-current replay diagnostics scale factors do not match the "
            "live banana-current coordinate contract."
        ),
    )


def build_replayed_candidate_x(
    accepted_x,
    coordinate_indices,
    optimizer_coordinate_values,
) -> np.ndarray:
    candidate_x = np.asarray(accepted_x, dtype=float).copy()
    indices = np.asarray(coordinate_indices, dtype=int)
    replay_values = np.asarray(optimizer_coordinate_values, dtype=float)
    if indices.shape != replay_values.shape:
        raise ValueError(
            "Replay coordinate indices and optimizer-coordinate values must share a shape."
        )
    candidate_x[indices] = replay_values
    return candidate_x


def _write_json_artifact(path: str | Path, payload: Mapping[str, object]) -> None:
    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    temp_fd, temp_path = tempfile.mkstemp(
        dir=str(artifact_path.parent),
        prefix=f".{artifact_path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(temp_fd, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2)
        os.replace(temp_path, artifact_path)
    except BaseException:
        os.unlink(temp_path)
        raise


def _float_list(values) -> list[float]:
    return [float(value) for value in values]


def _require_matching_float_vectors(
    expected_values,
    observed_values,
    *,
    mismatch_message: str,
) -> None:
    expected = np.asarray(expected_values, dtype=float)
    observed = np.asarray(observed_values, dtype=float)
    if expected.shape != observed.shape or not np.allclose(
        expected,
        observed,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError(mismatch_message)
