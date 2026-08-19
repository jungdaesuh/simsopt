"""Physics-only reader for a frozen flat-675 input bundle.

Two plain JSON records of the bundle are parsed here: ``manifest.json``, which
carries the start candidate and the two policies, and ``vessel_material.json``,
which carries the vessel surface template.  Only the physics is read — the
bundle's provenance, file-identity, and evidence machinery has no consumer in
this port and is deliberately left in the campaign tooling that mints it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray
from simsopt_jax.core.specs import SurfaceRZFourierSpec, make_surface_rzfourier_spec

from .formulation import (
    Flat675Candidate,
    Flat675ContractError,
)
from .formulation import (
    flat675_finite_float as _float,
)
from .policy import (
    Flat675BoozerLabelType,
    Flat675BoozerSystemPolicy,
    Flat675HardwareSoftPenaltyPolicy,
    Flat675ObjectivePolicy,
)

FLAT675_MANIFEST_FILENAME: Final[str] = "manifest.json"
FLAT675_VESSEL_MATERIAL_FILENAME: Final[str] = "vessel_material.json"

_OBJECTIVE_POLICY_SCHEMA: Final[str] = (
    "simsopt.single_stage.fixed_state_genuine_675.objective_policy.v2"
)
_BOOZER_POLICY_SCHEMA: Final[str] = (
    "simsopt.single_stage.experimental_fullspace_675.boozer_system_policy.v1"
)
_VESSEL_MATERIAL_SCHEMA: Final[str] = "simsopt.active_surface_vessel_material.v1"


def _mapping(value: object, where: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise Flat675ContractError(f"{where} must be a string-keyed JSON object.")
    return value


def _integer(value: object, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise Flat675ContractError(f"{where} must be an integer.")
    return int(value)


def _string(value: object, where: str) -> str:
    if not isinstance(value, str):
        raise Flat675ContractError(f"{where} must be a string.")
    return value


def _required_keys(
    value: dict[str, object],
    expected: frozenset[str],
    where: str,
) -> None:
    missing = sorted(expected - frozenset(value))
    if missing:
        raise Flat675ContractError(f"{where} is missing {missing!r}.")


def _objective_policy(payload: object) -> Flat675ObjectivePolicy:
    value = _mapping(payload, "objective_policy")
    if value.get("schema") != _OBJECTIVE_POLICY_SCHEMA:
        raise Flat675ContractError(
            f"objective_policy.schema must be {_OBJECTIVE_POLICY_SCHEMA}."
        )
    constraint = _mapping(
        value.get("boozer_label_constraint"),
        "objective_policy.boozer_label_constraint",
    )
    _required_keys(
        constraint,
        frozenset(("weight", "target", "type", "phi_index")),
        "objective_policy.boozer_label_constraint",
    )
    weights = _mapping(value.get("weights"), "objective_policy.weights")
    _required_keys(
        weights,
        frozenset(
            (
                "non_qs",
                "boozer_residual",
                "iota_penalty",
                "curve_length",
                "curve_curve_distance",
                "curve_surface_distance",
                "surface_vessel_distance",
                "curve_curvature",
            )
        ),
        "objective_policy.weights",
    )
    targets = _mapping(value.get("targets"), "objective_policy.targets")
    _required_keys(
        targets,
        frozenset(
            (
                "curve_length_m",
                "curve_curve_distance_m",
                "curve_surface_distance_m",
                "surface_vessel_distance_m",
                "curve_curvature_inverse_m",
            )
        ),
        "objective_policy.targets",
    )
    soft_penalty_payload = _mapping(
        value.get("hardware_soft_penalty_policy"),
        "objective_policy.hardware_soft_penalty_policy",
    )
    base_weights = _mapping(
        soft_penalty_payload.get("base_weights"),
        "objective_policy.hardware_soft_penalty_policy.base_weights",
    )
    thresholds = _mapping(
        soft_penalty_payload.get("thresholds"),
        "objective_policy.hardware_soft_penalty_policy.thresholds",
    )
    soft_penalty_where = "objective_policy.hardware_soft_penalty_policy"
    soft_penalty = Flat675HardwareSoftPenaltyPolicy(
        common_scale=_float(
            soft_penalty_payload.get("common_scale"),
            f"{soft_penalty_where}.common_scale",
        ),
        curve_curve_base_weight=_float(
            base_weights.get("coil_coil_distance"),
            f"{soft_penalty_where}.base_weights.coil_coil_distance",
        ),
        curvature_base_weight=_float(
            base_weights.get("curvature"),
            f"{soft_penalty_where}.base_weights.curvature",
        ),
        curve_curve_threshold_m=_float(
            thresholds.get("coil_coil_distance_m"),
            f"{soft_penalty_where}.thresholds.coil_coil_distance_m",
        ),
        curvature_threshold_inverse_m=_float(
            thresholds.get("curvature_inverse_m"),
            f"{soft_penalty_where}.thresholds.curvature_inverse_m",
        ),
        penalty_exponent=_float(
            soft_penalty_payload.get("penalty_exponent"),
            f"{soft_penalty_where}.penalty_exponent",
        ),
    )
    # The manifest publishes the curvature exponent twice.  The soft-penalty
    # policy is the owner, so the top-level copy is checked against it rather
    # than read: two disagreeing sources are not the certified objective.
    declared_p_norm = _float(
        value.get("curvature_p_norm"), "objective_policy.curvature_p_norm"
    )
    if declared_p_norm != soft_penalty.penalty_exponent:
        raise Flat675ContractError(
            "objective_policy.curvature_p_norm differs from the hardware "
            "soft-penalty exponent."
        )
    label_type_value = _string(
        constraint["type"], "objective_policy.boozer_label_constraint.type"
    )
    if label_type_value not in tuple(item.value for item in Flat675BoozerLabelType):
        raise Flat675ContractError(
            f"objective_policy Boozer label type {label_type_value!r} is unknown."
        )
    return Flat675ObjectivePolicy(
        iota_target=_float(value.get("iota_target"), "objective_policy.iota_target"),
        boozer_constraint_weight=_float(
            constraint["weight"],
            "objective_policy.boozer_label_constraint.weight",
        ),
        boozer_target_label=_float(
            constraint["target"],
            "objective_policy.boozer_label_constraint.target",
        ),
        boozer_label_type=Flat675BoozerLabelType(label_type_value),
        boozer_label_phi_index=_integer(
            constraint["phi_index"],
            "objective_policy.boozer_label_constraint.phi_index",
        ),
        non_qs_weight=_float(weights["non_qs"], "objective_policy.weights.non_qs"),
        non_qs_grid_size=_integer(
            value.get("non_qs_grid_size"), "objective_policy.non_qs_grid_size"
        ),
        residual_weight=_float(
            weights["boozer_residual"], "objective_policy.weights.boozer_residual"
        ),
        iota_weight=_float(
            weights["iota_penalty"], "objective_policy.weights.iota_penalty"
        ),
        length_weight=_float(
            weights["curve_length"], "objective_policy.weights.curve_length"
        ),
        length_target_m=_float(
            targets["curve_length_m"], "objective_policy.targets.curve_length_m"
        ),
        curve_curve_weight=_float(
            weights["curve_curve_distance"],
            "objective_policy.weights.curve_curve_distance",
        ),
        curve_curve_threshold_m=_float(
            targets["curve_curve_distance_m"],
            "objective_policy.targets.curve_curve_distance_m",
        ),
        curve_surface_weight=_float(
            weights["curve_surface_distance"],
            "objective_policy.weights.curve_surface_distance",
        ),
        curve_surface_threshold_m=_float(
            targets["curve_surface_distance_m"],
            "objective_policy.targets.curve_surface_distance_m",
        ),
        surface_vessel_weight=_float(
            weights["surface_vessel_distance"],
            "objective_policy.weights.surface_vessel_distance",
        ),
        surface_vessel_threshold_m=_float(
            targets["surface_vessel_distance_m"],
            "objective_policy.targets.surface_vessel_distance_m",
        ),
        curvature_weight=_float(
            weights["curve_curvature"], "objective_policy.weights.curve_curvature"
        ),
        curvature_threshold_inverse_m=_float(
            targets["curve_curvature_inverse_m"],
            "objective_policy.targets.curve_curvature_inverse_m",
        ),
        optimized_coil_index=_integer(
            value.get("banana_curve_index"), "objective_policy.banana_curve_index"
        ),
        hardware_soft_penalty_policy=soft_penalty,
    )


def _boozer_policy(payload: object) -> Flat675BoozerSystemPolicy:
    value = _mapping(payload, "boozer_construction_policy")
    if value.get("schema") != _BOOZER_POLICY_SCHEMA:
        raise Flat675ContractError(
            f"boozer_construction_policy.schema must be {_BOOZER_POLICY_SCHEMA}."
        )
    weight_by_inverse = value.get("weight_by_inverse_field_magnitude")
    if not isinstance(weight_by_inverse, bool):
        raise Flat675ContractError(
            "boozer_construction_policy.weight_by_inverse_field_magnitude "
            "must be a boolean."
        )
    return Flat675BoozerSystemPolicy(
        weight_by_inverse_field_magnitude=weight_by_inverse
    )


@dataclass(frozen=True, slots=True)
class Flat675InputManifest:
    """The three physics records a frozen flat-675 bundle publishes."""

    candidate: Flat675Candidate
    objective_policy: Flat675ObjectivePolicy
    boozer_construction_policy: Flat675BoozerSystemPolicy


def load_flat675_input_manifest(bundle_root: Path | str) -> Flat675InputManifest:
    """Read the start candidate and both policies from one frozen bundle."""
    path = Path(bundle_root) / FLAT675_MANIFEST_FILENAME
    payload = _mapping(json.loads(path.read_bytes()), "manifest")
    return Flat675InputManifest(
        candidate=Flat675Candidate.from_payload(
            _mapping(payload.get("candidate"), "candidate")
        ),
        objective_policy=_objective_policy(payload.get("objective_policy")),
        boozer_construction_policy=_boozer_policy(
            payload.get("boozer_construction_policy")
        ),
    )


def _vessel_array(arrays: dict[str, object], name: str) -> NDArray[np.float64]:
    where = f"vessel_material.arrays.{name}"
    record = _mapping(arrays.get(name), where)
    raw_shape = record.get("shape")
    if not isinstance(raw_shape, list):
        raise Flat675ContractError(f"{where}.shape must be a JSON array.")
    shape = tuple(
        _integer(item, f"{where}.shape[{index}]")
        for index, item in enumerate(raw_shape)
    )
    try:
        return np.asarray(record["values"], dtype=np.float64).reshape(shape)
    except (KeyError, TypeError, ValueError) as error:
        raise Flat675ContractError(
            f"vessel_material.arrays.{name} is not a valid float64 array."
        ) from error


def load_flat675_vessel_template(bundle_root: Path | str) -> SurfaceRZFourierSpec:
    """Read the frozen vessel surface the surface-to-vessel term measures against."""
    path = Path(bundle_root) / FLAT675_VESSEL_MATERIAL_FILENAME
    payload = _mapping(json.loads(path.read_bytes()), "vessel_material")
    if payload.get("schema") != _VESSEL_MATERIAL_SCHEMA:
        raise Flat675ContractError(
            f"vessel_material.schema must be {_VESSEL_MATERIAL_SCHEMA}."
        )
    arrays = _mapping(payload.get("arrays"), "vessel_material.arrays")
    surface = _mapping(payload.get("surface"), "vessel_material.surface")
    stellsym = surface.get("stellsym")
    if not isinstance(stellsym, bool):
        raise Flat675ContractError(
            "vessel_material.surface.stellsym must be a boolean."
        )
    return make_surface_rzfourier_spec(
        rc=_vessel_array(arrays, "rc"),
        zs=_vessel_array(arrays, "zs"),
        rs=_vessel_array(arrays, "rs"),
        zc=_vessel_array(arrays, "zc"),
        quadpoints_phi=_vessel_array(arrays, "quadpoints_phi"),
        quadpoints_theta=_vessel_array(arrays, "quadpoints_theta"),
        nfp=_integer(surface.get("nfp"), "vessel_material.surface.nfp"),
        stellsym=stellsym,
    )


__all__ = [
    "FLAT675_MANIFEST_FILENAME",
    "FLAT675_VESSEL_MATERIAL_FILENAME",
    "Flat675InputManifest",
    "load_flat675_input_manifest",
    "load_flat675_vessel_template",
]
