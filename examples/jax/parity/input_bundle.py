"""Canonical parity inputs generated once and consumed by every lane."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np
from simsopt_jax.examples import ExecutionScale
from examples.jax.parity.artifacts import (
    canonical_json_bytes,
    read_array,
    read_bytes,
    write_array,
    write_bytes_exclusive,
)
from examples.jax.parity.contracts import ArrayReference

_ARRAY_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")


@dataclass(frozen=True)
class InputBundle:
    schema_version: int
    case_id: str
    scale: ExecutionScale
    random_seed: int
    configuration: Mapping[str, object]
    configuration_fingerprint: str
    arrays: Mapping[str, ArrayReference]
    input_fingerprint: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "configuration", MappingProxyType(dict(self.configuration))
        )
        object.__setattr__(self, "arrays", MappingProxyType(dict(self.arrays)))


def _digest(payload: object) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def effective_construction_fingerprint(
    bundle: InputBundle, applied_construction: Mapping[str, object]
) -> str:
    """Fingerprint the seed and values reconstructed from an instantiated case."""
    return _digest(
        {
            "case_id": bundle.case_id,
            "scale": bundle.scale,
            "random_seed": bundle.random_seed,
            "applied_construction": dict(applied_construction),
        }
    )


def _payload(bundle: InputBundle) -> dict[str, object]:
    return {
        "schema_version": bundle.schema_version,
        "case_id": bundle.case_id,
        "scale": bundle.scale,
        "random_seed": bundle.random_seed,
        "configuration": dict(bundle.configuration),
        "configuration_fingerprint": bundle.configuration_fingerprint,
        "arrays": {
            name: dataclasses.asdict(reference)
            for name, reference in sorted(bundle.arrays.items())
        },
        "input_fingerprint": bundle.input_fingerprint,
    }


def create_input_bundle(
    root: Path,
    *,
    case_id: str,
    random_seed: int,
    arrays: Mapping[str, np.ndarray],
    configuration: Mapping[str, object],
    scale: ExecutionScale = "bounded",
) -> InputBundle:
    """Persist canonical arrays and return their immutable bundle contract."""
    if not case_id:
        raise ValueError("case_id must not be empty")
    if not arrays:
        raise ValueError("input bundle requires at least one array")
    references: dict[str, ArrayReference] = {}
    for name, values in sorted(arrays.items()):
        if _ARRAY_NAME.fullmatch(name) is None:
            raise ValueError(f"invalid input array name: {name!r}")
        references[name] = write_array(root, f"inputs/{name}.npy", values)
    configuration_payload = dict(configuration)
    configuration_fingerprint = _digest(configuration_payload)
    fingerprint_payload = {
        "case_id": case_id,
        "scale": scale,
        "random_seed": random_seed,
        "configuration_fingerprint": configuration_fingerprint,
        "arrays": {
            name: dataclasses.asdict(reference)
            for name, reference in sorted(references.items())
        },
    }
    bundle = InputBundle(
        schema_version=2,
        case_id=case_id,
        scale=scale,
        random_seed=random_seed,
        configuration=configuration_payload,
        configuration_fingerprint=configuration_fingerprint,
        arrays=references,
        input_fingerprint=_digest(fingerprint_payload),
    )
    write_bytes_exclusive(
        root,
        "input_bundle.json",
        canonical_json_bytes(_payload(bundle)),
    )
    return bundle


def _reference(value: object, name: str) -> ArrayReference:
    if not isinstance(value, dict):
        raise TypeError(f"array reference must be an object: {name}")
    try:
        path = value["path"]
        dtype = value["dtype"]
        shape = value["shape"]
        order = value["order"]
        sha256 = value["sha256"]
    except KeyError as error:
        raise ValueError(
            f"array reference is missing {error.args[0]}: {name}"
        ) from error
    if not all(isinstance(item, str) for item in (path, dtype, order, sha256)):
        raise ValueError(f"array reference has invalid string fields: {name}")
    if not isinstance(shape, list) or not all(isinstance(item, int) for item in shape):
        raise ValueError(f"array reference has invalid shape: {name}")
    return ArrayReference(path, dtype, tuple(shape), order, sha256)


def read_input_bundle(root: Path) -> tuple[InputBundle, dict[str, np.ndarray]]:
    """Load a persisted bundle and recompute every declared fingerprint."""
    document = json.loads(read_bytes(root, "input_bundle.json"))
    if not isinstance(document, dict) or document.get("schema_version") not in (1, 2):
        raise ValueError("unsupported input bundle schema")
    schema_version = int(document["schema_version"])
    scale_value = document.get("scale", "bounded")
    if scale_value not in ("bounded", "native_default"):
        raise ValueError("invalid input bundle scale")
    scale: ExecutionScale = "bounded" if scale_value == "bounded" else "native_default"
    arrays_value = document.get("arrays")
    if not isinstance(arrays_value, dict):
        raise TypeError("input bundle arrays must be an object")
    references = {
        name: _reference(value, name)
        for name, value in arrays_value.items()
        if isinstance(name, str)
    }
    configuration = document.get("configuration")
    if not isinstance(configuration, dict):
        raise TypeError("input bundle configuration must be an object")
    loaded = InputBundle(
        schema_version=schema_version,
        case_id=str(document.get("case_id", "")),
        scale=scale,
        random_seed=int(document.get("random_seed", -1)),
        configuration=configuration,
        configuration_fingerprint=str(document.get("configuration_fingerprint", "")),
        arrays=references,
        input_fingerprint=str(document.get("input_fingerprint", "")),
    )
    if loaded.configuration_fingerprint != _digest(dict(loaded.configuration)):
        raise ValueError("persisted configuration fingerprint mismatch")
    fingerprint_payload = {
        "case_id": loaded.case_id,
        **({"scale": loaded.scale} if loaded.schema_version >= 2 else {}),
        "random_seed": loaded.random_seed,
        "configuration_fingerprint": loaded.configuration_fingerprint,
        "arrays": {
            name: dataclasses.asdict(reference)
            for name, reference in sorted(loaded.arrays.items())
        },
    }
    if loaded.input_fingerprint != _digest(fingerprint_payload):
        raise ValueError("persisted input fingerprint mismatch")
    arrays = {
        name: read_array(root, reference)
        for name, reference in sorted(references.items())
    }
    return loaded, arrays


def load_input_bundle(
    root: Path, expected: InputBundle
) -> tuple[InputBundle, dict[str, np.ndarray]]:
    """Load a persisted bundle and reject drift from an expected contract."""
    loaded, arrays = read_input_bundle(root)
    if loaded != expected:
        raise ValueError("persisted input bundle does not match expected fingerprint")
    return loaded, arrays
