"""Hash-bound serialization for one isolated parity lane observation."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

from examples.jax.parity.arbiter import LaneObservation
from examples.jax.parity.artifacts import (
    canonical_json_bytes,
    read_array,
    read_bytes,
    write_array,
    write_bytes_exclusive,
)
from examples.jax.parity.contracts import ArrayReference
from examples.jax.parity.provenance import (
    lane_provenance_from_payload,
    lane_provenance_payload,
)

_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "lane",
        "backend_mode",
        "platform",
        "precision",
        "input_fingerprint",
        "configuration_fingerprint",
        "effective_construction_fingerprint",
        "driver",
        "normalized_status",
        "raw_status",
        "success",
        "nit",
        "nfev",
        "njev",
        "completed_workflow_stages",
        "applicability",
        "provenance",
        "values",
    }
)


def _value_filename(key: str) -> str:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return f"values/{digest}.npy"


def write_lane_observation(root: Path, observation: LaneObservation) -> Path:
    """Write one lane receipt and its deterministic array sidecars."""
    if observation.provenance is None:
        raise ValueError("lane observation requires provenance before publication")
    references = {
        key: write_array(root, _value_filename(key), value)
        for key, value in sorted(observation.values.items())
    }
    payload = {
        "schema_version": 1,
        "lane": observation.lane,
        "backend_mode": observation.backend_mode,
        "platform": observation.platform,
        "precision": observation.precision,
        "input_fingerprint": observation.input_fingerprint,
        "configuration_fingerprint": observation.configuration_fingerprint,
        "effective_construction_fingerprint": (
            observation.effective_construction_fingerprint
        ),
        "driver": observation.driver,
        "normalized_status": observation.normalized_status,
        "raw_status": observation.raw_status,
        "success": observation.success,
        "nit": observation.nit,
        "nfev": observation.nfev,
        "njev": observation.njev,
        "completed_workflow_stages": list(observation.completed_workflow_stages),
        "applicability": dict(observation.applicability),
        "provenance": lane_provenance_payload(observation.provenance),
        "values": {
            key: dataclasses.asdict(reference)
            for key, reference in sorted(references.items())
        },
    }
    receipt = root / "lane_result.json"
    write_bytes_exclusive(root, receipt.name, canonical_json_bytes(payload))
    return receipt


def _required_string(document: dict[str, object], field: str) -> str:
    value = document[field]
    if not isinstance(value, str) or not value:
        raise ValueError(f"lane receipt field {field} must be a non-empty string")
    return value


def _array_reference(value: object, key: str) -> ArrayReference:
    if not isinstance(value, dict) or set(value) != {
        "path",
        "dtype",
        "shape",
        "order",
        "sha256",
    }:
        raise ValueError(f"invalid array reference for lane value {key}")
    path, dtype = value["path"], value["dtype"]
    shape, order, sha256 = value["shape"], value["order"], value["sha256"]
    if not all(isinstance(item, str) for item in (path, dtype, order, sha256)):
        raise ValueError(f"invalid array reference strings for lane value {key}")
    if not isinstance(shape, list) or not all(isinstance(item, int) for item in shape):
        raise ValueError(f"invalid array reference shape for lane value {key}")
    return ArrayReference(path, dtype, tuple(shape), order, sha256)


def _optional_counter(document: dict[str, object], field: str) -> int | None:
    value = document[field]
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"lane receipt field {field} must be an integer or null")
    return value


def _required_bool(document: dict[str, object], field: str) -> bool:
    value = document[field]
    if not isinstance(value, bool):
        raise TypeError(f"lane receipt field {field} must be boolean")
    return value


def _required_strings(document: dict[str, object], field: str) -> tuple[str, ...]:
    value = document[field]
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item for item in value)
        or len(value) != len(set(value))
    ):
        raise ValueError(
            f"lane receipt field {field} must be a non-empty unique string array"
        )
    return tuple(value)


def _applicability(document: dict[str, object]) -> dict[str, bool]:
    value = document["applicability"]
    if (
        not isinstance(value, dict)
        or "optimizer_outcome" not in value
        or not all(
            isinstance(key, str) and key and isinstance(item, bool)
            for key, item in value.items()
        )
    ):
        raise ValueError("lane receipt applicability must be a boolean-valued object")
    return {
        key: item
        for key, item in value.items()
        if isinstance(key, str) and isinstance(item, bool)
    }


def load_lane_observation(root: Path) -> LaneObservation:
    """Load a lane receipt after validating schema and every sidecar."""
    raw_document = json.loads(read_bytes(root, "lane_result.json"))
    if not isinstance(raw_document, dict) or not all(
        isinstance(key, str) for key in raw_document
    ):
        raise ValueError("lane receipt must be a JSON object")
    document = {
        key: value for key, value in raw_document.items() if isinstance(key, str)
    }
    if set(document) != _RECEIPT_FIELDS:
        raise ValueError("lane receipt has invalid fields")
    if document["schema_version"] != 1:
        raise ValueError("unsupported lane receipt schema")
    values = document["values"]
    if not isinstance(values, dict) or not all(isinstance(key, str) for key in values):
        raise ValueError("lane receipt values must be an object")
    arrays = {
        key: read_array(root, _array_reference(value, key))
        for key, value in values.items()
        if isinstance(key, str)
    }
    observation = LaneObservation(
        lane=_required_string(document, "lane"),
        backend_mode=_required_string(document, "backend_mode"),
        platform=_required_string(document, "platform"),
        precision=_required_string(document, "precision"),
        input_fingerprint=_required_string(document, "input_fingerprint"),
        configuration_fingerprint=_required_string(
            document, "configuration_fingerprint"
        ),
        effective_construction_fingerprint=_required_string(
            document, "effective_construction_fingerprint"
        ),
        driver=_required_string(document, "driver"),
        normalized_status=_required_string(document, "normalized_status"),
        raw_status=_required_string(document, "raw_status"),
        success=_required_bool(document, "success"),
        nit=_optional_counter(document, "nit"),
        nfev=_optional_counter(document, "nfev"),
        njev=_optional_counter(document, "njev"),
        completed_workflow_stages=_required_strings(
            document, "completed_workflow_stages"
        ),
        provenance=lane_provenance_from_payload(document["provenance"]),
        values=arrays,
        applicability=_applicability(document),
    )
    return observation
