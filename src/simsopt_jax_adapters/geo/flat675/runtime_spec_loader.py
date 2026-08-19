"""Decoder for the ``single_stage_jax_runtime_spec.json`` wire format.

The runtime spec is a tagged tree: every node carries a ``kind`` of ``array``,
``scalar``, ``tuple``, ``list``, or ``dataclass``, and dataclass nodes name one
of the immutable spec types this repo publishes.  This module owns that wire
format and nothing else, so the flat-675 port can read a frozen seed without
importing the 36k-line single-stage example that wrote it.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Final, TypeVar

import numpy as np
from numpy.typing import NDArray
from simsopt_jax.core import specs as jax_specs
from simsopt_jax.core.specs import (
    SingleStageRuntimeSpec,
    make_single_stage_runtime_spec,
    make_single_stage_seed_spec,
    make_surface_xyz_tensor_fourier_spec,
)

from .formulation import Flat675ContractError

_SpecT = TypeVar("_SpecT")

FLAT675_RUNTIME_SPEC_FILENAME: Final[str] = "single_stage_jax_runtime_spec.json"
_RUNTIME_SPEC_SCHEMA: Final[str] = "simsopt.single_stage.jax_runtime_spec"
_SUPPORTED_SURFACE_CLASS: Final[str] = "SurfaceXYZTensorFourier"

# The closed set of immutable spec dataclasses a runtime seed may name.  A
# payload that names anything else is rejected rather than resolved, so the
# decoder can never be steered into constructing an arbitrary type.
_SPEC_DATACLASSES: Final[dict[str, type]] = {
    name: getattr(jax_specs, name)
    for name in (
        "CoilDofExtractionSpec",
        "CoilGroupSpec",
        "CoilSetDofExtractionSpec",
        "CoilSpec",
        "CoilSymmetrySpec",
        "CurrentValueSpec",
        "CurveCWSFourierRZSpec",
        "CurveFilamentSpec",
        "CurveHelicalSpec",
        "CurvePerturbedSpec",
        "CurvePlanarFourierSpec",
        "CurveRZFourierSpec",
        "CurveXYZFourierSpec",
        "FrameRotationSpec",
        "GroupedCoilSetSpec",
        "OptimizableDofMapSpec",
        "SurfaceRZFourierSpec",
        "SurfaceXYZFourierSpec",
        "SurfaceXYZTensorFourierSpec",
        "ZeroRotationSpec",
    )
}


def _mapping(value: object, where: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise Flat675ContractError(f"{where} must be a string-keyed JSON object.")
    return value


def _sequence(value: object, where: str) -> list[object]:
    if not isinstance(value, list):
        raise Flat675ContractError(f"{where} must be a JSON array.")
    return value


def _integer(value: object, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise Flat675ContractError(f"{where} must be an integer.")
    return int(value)


def _float(value: object, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Flat675ContractError(f"{where} must be a real scalar.")
    scalar = float(value)
    if not math.isfinite(scalar):
        raise Flat675ContractError(f"{where} must be finite.")
    return scalar


def _float64_array(value: object, where: str) -> NDArray[np.float64]:
    """Decode one ``{dtype, shape, data}`` float64 array payload."""
    record = _mapping(value, where)
    if record.get("dtype") != "float64":
        raise Flat675ContractError(f"{where} must be a float64 runtime-spec array.")
    shape = tuple(
        _integer(item, f"{where}.shape[{index}]")
        for index, item in enumerate(_sequence(record.get("shape"), f"{where}.shape"))
    )
    try:
        array = np.asarray(record["data"], dtype=np.float64).reshape(shape)
    except (KeyError, TypeError, ValueError) as error:
        raise Flat675ContractError(f"{where} is not a valid float64 array.") from error
    if not np.all(np.isfinite(array)):
        raise Flat675ContractError(f"{where} must be finite.")
    return array


def _decode_node(value: object, where: str) -> object:
    """Decode one tagged runtime-spec node into its immutable Python value."""
    node = _mapping(value, where)
    kind = node.get("kind")
    if kind == "array":
        return _float64_array(node, where)
    if kind == "scalar":
        if "value" not in node:
            raise Flat675ContractError(f"{where}.value is required.")
        return node["value"]
    if kind == "tuple":
        return tuple(
            _decode_node(item, f"{where}[{index}]")
            for index, item in enumerate(_sequence(node.get("items"), f"{where}.items"))
        )
    if kind == "list":
        return [
            _decode_node(item, f"{where}[{index}]")
            for index, item in enumerate(_sequence(node.get("items"), f"{where}.items"))
        ]
    if kind == "dataclass":
        class_name = node.get("class")
        spec_class = _SPEC_DATACLASSES.get(str(class_name))
        if spec_class is None:
            raise Flat675ContractError(
                f"{where} names unsupported runtime-spec class {class_name!r}."
            )
        fields = _mapping(node.get("fields"), f"{where}.fields")
        return spec_class(
            **{
                name: _decode_node(field_value, f"{where}.{name}")
                for name, field_value in fields.items()
            }
        )
    raise Flat675ContractError(f"{where} has unsupported node kind {kind!r}.")


def _decode_spec(value: object, expected_type: type[_SpecT], where: str) -> _SpecT:
    decoded = _decode_node(value, where)
    if not isinstance(decoded, expected_type):
        raise Flat675ContractError(f"{where} must decode to {expected_type.__name__}.")
    return decoded


def load_flat675_runtime_spec(
    bundle_root: Path | str,
) -> SingleStageRuntimeSpec:
    """Read one frozen bundle's runtime seed spec as an immutable spec tree."""
    path = Path(bundle_root) / FLAT675_RUNTIME_SPEC_FILENAME
    payload = _mapping(json.loads(path.read_bytes()), "runtime_spec")
    if payload.get("schema") != _RUNTIME_SPEC_SCHEMA:
        raise Flat675ContractError(
            f"{path} does not carry the {_RUNTIME_SPEC_SCHEMA} schema."
        )
    surface_payload = _mapping(payload.get("surface"), "runtime_spec.surface")
    if surface_payload.get("surface_class") != _SUPPORTED_SURFACE_CLASS:
        raise Flat675ContractError(
            f"flat-675 requires a canonical {_SUPPORTED_SURFACE_CLASS} runtime surface."
        )
    stellsym = surface_payload.get("stellsym")
    if not isinstance(stellsym, bool):
        raise Flat675ContractError("runtime_spec.surface.stellsym must be a boolean.")
    mpol = _integer(surface_payload.get("mpol"), "runtime_spec.surface.mpol")
    ntor = _integer(surface_payload.get("ntor"), "runtime_spec.surface.ntor")
    nfp = _integer(surface_payload.get("nfp"), "runtime_spec.surface.nfp")
    surface = make_surface_xyz_tensor_fourier_spec(
        dofs=_float64_array(surface_payload.get("dofs"), "runtime_spec.surface.dofs"),
        quadpoints_phi=_float64_array(
            surface_payload.get("quadpoints_phi"),
            "runtime_spec.surface.quadpoints_phi",
        ),
        quadpoints_theta=_float64_array(
            surface_payload.get("quadpoints_theta"),
            "runtime_spec.surface.quadpoints_theta",
        ),
        nfp=nfp,
        stellsym=stellsym,
        mpol=mpol,
        ntor=ntor,
    )

    field_payload = _mapping(payload.get("field"), "runtime_spec.field")
    boozer_payload = _mapping(payload.get("boozer_init"), "runtime_spec.boozer_init")
    quadrature = _mapping(payload.get("quadrature"), "runtime_spec.quadrature")
    hardware_constants = _mapping(
        payload.get("hardware_constants"), "runtime_spec.hardware_constants"
    )
    target_labels = payload.get("target_labels")
    if not isinstance(target_labels, list) or not all(
        isinstance(label, str) for label in target_labels
    ):
        raise Flat675ContractError(
            "runtime_spec.target_labels must be a list of strings."
        )
    self_intersection_mode = payload.get("self_intersection_mode")
    if not isinstance(self_intersection_mode, str):
        raise Flat675ContractError(
            "runtime_spec.self_intersection_mode must be a string."
        )
    seed = make_single_stage_seed_spec(
        surface=surface,
        coil_set=_decode_spec(
            field_payload.get("coil_set"),
            jax_specs.GroupedCoilSetSpec,
            "runtime_spec.field.coil_set",
        ),
        coil_dof_extraction=_decode_spec(
            field_payload.get("coil_dof_extraction"),
            jax_specs.CoilSetDofExtractionSpec,
            "runtime_spec.field.coil_dof_extraction",
        ),
        coil_dofs=_float64_array(
            field_payload.get("coil_dofs"), "runtime_spec.field.coil_dofs"
        ),
        boozer_iota=_float(boozer_payload.get("iota"), "runtime_spec.boozer_init.iota"),
        boozer_G=_float(boozer_payload.get("G"), "runtime_spec.boozer_init.G"),
        target_labels=target_labels,
        hardware_constants=tuple(hardware_constants.items()),
        self_intersection_mode=self_intersection_mode,
        schema_version=_integer(
            payload.get("schema_version"), "runtime_spec.schema_version"
        ),
        num_tf_coils=_integer(
            field_payload.get("num_tf_coils"), "runtime_spec.field.num_tf_coils"
        ),
        # The frozen wire format still spells the optimized coil the way the
        # source campaign did; the runtime spec renamed the same slot.
        optimized_coil_index=_integer(
            field_payload.get("banana_curve_index"),
            "runtime_spec.field.banana_curve_index",
        ),
        tf_current_A=_float(
            field_payload.get("tf_current_A"), "runtime_spec.field.tf_current_A"
        ),
        optimized_coil_current_A=_float(
            field_payload.get("banana_current_A"),
            "runtime_spec.field.banana_current_A",
        ),
    )
    return make_single_stage_runtime_spec(
        seed=seed,
        mpol=mpol,
        ntor=ntor,
        nfp=nfp,
        nphi=_integer(quadrature.get("nphi"), "runtime_spec.quadrature.nphi"),
        ntheta=_integer(quadrature.get("ntheta"), "runtime_spec.quadrature.ntheta"),
    )


__all__ = [
    "FLAT675_RUNTIME_SPEC_FILENAME",
    "load_flat675_runtime_spec",
]
