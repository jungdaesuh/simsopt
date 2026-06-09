"""Explicit adapter-owned JSON IO for JAX specs and legacy object bridges."""

from __future__ import annotations

import json
import pathlib
from dataclasses import fields, is_dataclass
from typing import Any

import numpy as np

import simsopt_jax.core.specs as _jax_specs
from simsopt._core.json import GSONEncoder, SIMSON
from simsopt._core.optimizable import load
from simsopt_jax.core.field import coil_set_spec_from_dof_extraction_spec
from simsopt_jax.core.surface_rzfourier import surface_rz_fourier_spec_from_dofs
from simsopt_jax.runtime.host_boundary import host_tree
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX

__all__ = (
    "load_specs",
    "save_biot_savart_spec",
    "save_surface_rz_fourier_spec",
    "save_surface_xyz_fourier_spec",
    "save_surface_xyz_tensor_fourier_spec",
)

_JAX_SPECS_MODULE = "simsopt_jax.core.specs"
_LEGACY_BIOT_SAVART_PAYLOAD = ("simsopt.field.biotsavart", "BiotSavart")
_LEGACY_SURFACE_RZ_FOURIER_PAYLOAD = (
    "simsopt.geo.surfacerzfourier",
    "SurfaceRZFourier",
)
_SUPPORTED_LEGACY_PAYLOADS = frozenset(
    {
        _LEGACY_BIOT_SAVART_PAYLOAD,
        _LEGACY_SURFACE_RZ_FOURIER_PAYLOAD,
    }
)

_SPEC_CLASSES = {
    name: spec_cls
    for name in _jax_specs.__all__
    if isinstance((spec_cls := getattr(_jax_specs, name, None)), type)
    and is_dataclass(spec_cls)
    and spec_cls.__module__ == _JAX_SPECS_MODULE
}


def _load_raw_json(filename):
    with pathlib.Path(filename).open("rt", encoding="utf-8") as fp:
        return json.load(fp)


def _require_simson_wrapper(raw):
    if not isinstance(raw, dict):
        raise ValueError("load_specs expected GSON SIMSON wrapper")
    if raw.get("@module") != "simsopt._core.json" or raw.get("@class") != "SIMSON":
        raise ValueError("load_specs expected GSON SIMSON wrapper")


def _load_specs_graph_payload(raw):
    graph = raw.get("graph")
    if isinstance(graph, dict) and "$type" in graph:
        if graph["$type"] != "ref":
            raise NotImplementedError("Unsupported GSON value type")
        return raw.get("simsopt_objs", {})[graph["value"]]
    return graph


def _payload_type(payload):
    if not isinstance(payload, dict):
        raise NotImplementedError("Unsupported GSON value type")
    return payload.get("@module"), payload.get("@class")


def _decode_numpy_array(payload):
    dtype = np.dtype(payload["dtype"])
    data = payload["data"]
    if dtype.kind == "c":
        return np.asarray(data[0], dtype=dtype) + 1j * np.asarray(data[1], dtype=dtype)
    return np.asarray(data, dtype=dtype)


def _decode_spec_payload(payload):
    classname = payload.get("@class")
    spec_cls = _SPEC_CLASSES.get(classname)
    if spec_cls is None:
        raise NotImplementedError(f"{payload.get('@module')}.{classname}")
    kwargs: dict[str, Any] = {}
    for field in fields(spec_cls):
        if field.name in payload:
            kwargs[field.name] = _decode_spec_value(payload[field.name])
    return spec_cls(**kwargs)


def _decode_spec_value(value):
    if isinstance(value, list):
        return tuple(_decode_spec_value(item) for item in value)
    if isinstance(value, dict):
        module, classname = value.get("@module"), value.get("@class")
        if (module, classname) == ("numpy", "array"):
            return _decode_numpy_array(value)
        if module == _JAX_SPECS_MODULE:
            return _decode_spec_payload(value)
        return {key: _decode_spec_value(item) for key, item in value.items()}
    return value


def _surface_rz_fourier_spec_from_surface(surface):
    return surface_rz_fourier_spec_from_dofs(
        surface.get_dofs(),
        quadpoints_phi=surface.quadpoints_phi,
        quadpoints_theta=surface.quadpoints_theta,
        mpol=surface.mpol,
        ntor=surface.ntor,
        nfp=surface.nfp,
        stellsym=surface.stellsym,
    )


def _load_specs_from_object(obj):
    if isinstance(obj, _jax_specs.GroupedCoilSetSpec):
        return {"coil_set_spec": obj}
    if isinstance(obj, _jax_specs.BiotSavartSpec):
        return {
            "biot_savart_spec": obj,
            "coil_set_spec": coil_set_spec_from_dof_extraction_spec(
                obj.coil_dof_extraction,
                obj.coil_dofs,
            ),
        }
    if isinstance(
        obj,
        (
            _jax_specs.SurfaceRZFourierSpec,
            _jax_specs.SurfaceXYZFourierSpec,
            _jax_specs.SurfaceXYZTensorFourierSpec,
        ),
    ):
        return {"surface_spec": obj}
    if type(obj).__name__ == "BiotSavart":
        return {"coil_set_spec": BiotSavartJAX(obj.coils).coil_set_spec()}
    if type(obj).__name__ == "SurfaceRZFourier":
        return {"surface_spec": _surface_rz_fourier_spec_from_surface(obj)}
    raise NotImplementedError(f"{type(obj).__module__}.{type(obj).__name__}")


def load_specs(filename):
    raw = _load_raw_json(filename)
    _require_simson_wrapper(raw)
    payload = _load_specs_graph_payload(raw)
    module, classname = _payload_type(payload)
    if module == _JAX_SPECS_MODULE:
        return _load_specs_from_object(_decode_spec_payload(payload))
    if (module, classname) in _SUPPORTED_LEGACY_PAYLOADS:
        return _load_specs_from_object(load(filename))
    raise NotImplementedError(f"{module}.{classname}")


def _encode_spec_value(value):
    if is_dataclass(value):
        encoded = {
            "@module": str(value.__class__.__module__),
            "@class": str(value.__class__.__name__),
        }
        for field in fields(value):
            encoded[field.name] = _encode_spec_value(getattr(value, field.name))
        return encoded
    if isinstance(value, tuple):
        return [_encode_spec_value(item) for item in value]
    if isinstance(value, list):
        return [_encode_spec_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _encode_spec_value(item) for key, item in value.items()}
    return value


def _save_spec(filename, spec):
    with pathlib.Path(filename).open("wt", encoding="utf-8") as fp:
        json.dump(
            SIMSON(_encode_spec_value(host_tree(spec))),
            fp,
            cls=GSONEncoder,
            indent=2,
        )


def save_biot_savart_spec(filename, spec):
    _save_spec(filename, spec)


def save_surface_rz_fourier_spec(filename, spec):
    _save_spec(filename, spec)


def save_surface_xyz_fourier_spec(filename, spec):
    _save_spec(filename, spec)


def save_surface_xyz_tensor_fourier_spec(filename, spec):
    _save_spec(filename, spec)
