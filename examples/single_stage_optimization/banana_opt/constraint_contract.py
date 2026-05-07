"""Shared constraint contract for the HBT-EP single-stage and Stage 2 pipelines.

Single source of truth for the engineering and geometry values that every Stage 2
runner, single-stage donor, sweep orchestrator, and smoke harness has previously
redeclared. Three classes of field exist:

* Fixed geometry (vessel major/minor radius and the banana winding-surface major
  radius) always come from :mod:`banana_opt.hardware_contracts` and are not
  routed through the profile/spec/cli override ladder.
* Default engineering values (TF current, banana-coil current ceiling, length
  target, coil-coil spacing threshold, coil-plasma and plasma-vessel clearances,
  curvature ceiling, and the banana winding surface minor radius) ship with
  hardware defaults and may be replaced via ``profile < spec_json < cli``.
* Target plasma ceilings (the requested LCFS major/minor radius) live next to
  the engineering values but are validated against their own hardware caps and
  are **not** interchangeable with the vessel or winding-surface geometry.

The resolver returns an immutable contract mapping plus a provenance trace, a
stable SHA-256 hash, and a metadata builder used at artifact-write time.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from . import hardware_contracts as _hc

CONSTRAINT_SCHEMA_VERSION = 1

_KEY_VACUUM_VESSEL_MAJOR_RADIUS_M = "VACUUM_VESSEL_MAJOR_RADIUS_M"
_KEY_VACUUM_VESSEL_MINOR_RADIUS_M = "VACUUM_VESSEL_MINOR_RADIUS_M"
_KEY_BANANA_WINDING_SURFACE_MAJOR_RADIUS_M = "BANANA_WINDING_SURFACE_MAJOR_RADIUS_M"

_KEY_TF_CURRENT_A = "TF_CURRENT_A"
_KEY_BANANA_CURRENT_MAX_A = "BANANA_CURRENT_MAX_A"
_KEY_COIL_LENGTH_TARGET_M = "COIL_LENGTH_TARGET_M"
_KEY_CC_THRESHOLD = "CC_THRESHOLD"
_KEY_COIL_PLASMA_MIN_DIST_M = "COIL_PLASMA_MIN_DIST_M"
_KEY_PLASMA_VESSEL_MIN_DIST_M = "PLASMA_VESSEL_MIN_DIST_M"
_KEY_CURVATURE_THRESHOLD = "CURVATURE_THRESHOLD"
_KEY_BANANA_SURF_RADIUS = "banana_surf_radius"

_KEY_TARGET_LCFS_MAX_MAJOR_RADIUS_M = "TARGET_LCFS_MAX_MAJOR_RADIUS_M"
_KEY_TARGET_LCFS_MAX_MINOR_RADIUS_M = "TARGET_LCFS_MAX_MINOR_RADIUS_M"

FIXED_GEOMETRY_KEYS: frozenset[str] = frozenset({
    _KEY_VACUUM_VESSEL_MAJOR_RADIUS_M,
    _KEY_VACUUM_VESSEL_MINOR_RADIUS_M,
    _KEY_BANANA_WINDING_SURFACE_MAJOR_RADIUS_M,
})

ENGINEERING_DEFAULT_KEYS: frozenset[str] = frozenset({
    _KEY_TF_CURRENT_A,
    _KEY_BANANA_CURRENT_MAX_A,
    _KEY_COIL_LENGTH_TARGET_M,
    _KEY_CC_THRESHOLD,
    _KEY_COIL_PLASMA_MIN_DIST_M,
    _KEY_PLASMA_VESSEL_MIN_DIST_M,
    _KEY_CURVATURE_THRESHOLD,
    _KEY_BANANA_SURF_RADIUS,
})

TARGET_PLASMA_CEILING_KEYS: frozenset[str] = frozenset({
    _KEY_TARGET_LCFS_MAX_MAJOR_RADIUS_M,
    _KEY_TARGET_LCFS_MAX_MINOR_RADIUS_M,
})

CONSTRAINT_FIELD_TYPES: Mapping[str, type] = MappingProxyType({
    _KEY_VACUUM_VESSEL_MAJOR_RADIUS_M: float,
    _KEY_VACUUM_VESSEL_MINOR_RADIUS_M: float,
    _KEY_BANANA_WINDING_SURFACE_MAJOR_RADIUS_M: float,
    _KEY_TF_CURRENT_A: float,
    _KEY_BANANA_CURRENT_MAX_A: float,
    _KEY_COIL_LENGTH_TARGET_M: float,
    _KEY_CC_THRESHOLD: float,
    _KEY_COIL_PLASMA_MIN_DIST_M: float,
    _KEY_PLASMA_VESSEL_MIN_DIST_M: float,
    _KEY_CURVATURE_THRESHOLD: float,
    _KEY_BANANA_SURF_RADIUS: float,
    _KEY_TARGET_LCFS_MAX_MAJOR_RADIUS_M: float,
    _KEY_TARGET_LCFS_MAX_MINOR_RADIUS_M: float,
})

CONSTRAINT_SOURCE_HARDWARE = "hardware_contract"
CONSTRAINT_SOURCE_PROFILE = "profile"
CONSTRAINT_SOURCE_SPEC_JSON = "spec_json"
CONSTRAINT_SOURCE_CLI = "cli"
WIRE_NAME_ALIASES: Mapping[str, str] = MappingProxyType({
    _KEY_VACUUM_VESSEL_MAJOR_RADIUS_M: _KEY_VACUUM_VESSEL_MAJOR_RADIUS_M,
    _KEY_VACUUM_VESSEL_MINOR_RADIUS_M: _KEY_VACUUM_VESSEL_MINOR_RADIUS_M,
    _KEY_BANANA_WINDING_SURFACE_MAJOR_RADIUS_M: _KEY_BANANA_WINDING_SURFACE_MAJOR_RADIUS_M,
    "major_radius": _KEY_VACUUM_VESSEL_MAJOR_RADIUS_M,
    "tf_current_A": _KEY_TF_CURRENT_A,
    "TF_CURRENT_A": _KEY_TF_CURRENT_A,
    "banana_current_max_A": _KEY_BANANA_CURRENT_MAX_A,
    "BANANA_CURRENT_MAX_A": _KEY_BANANA_CURRENT_MAX_A,
    "length_target": _KEY_COIL_LENGTH_TARGET_M,
    "LENGTH_TARGET": _KEY_COIL_LENGTH_TARGET_M,
    "COIL_LENGTH_TARGET_M": _KEY_COIL_LENGTH_TARGET_M,
    "cc_threshold": _KEY_CC_THRESHOLD,
    "CC_THRESHOLD": _KEY_CC_THRESHOLD,
    "coil_plasma_min_dist_m": _KEY_COIL_PLASMA_MIN_DIST_M,
    "COIL_PLASMA_MIN_DIST_M": _KEY_COIL_PLASMA_MIN_DIST_M,
    "plasma_vessel_min_dist_m": _KEY_PLASMA_VESSEL_MIN_DIST_M,
    "PLASMA_VESSEL_MIN_DIST_M": _KEY_PLASMA_VESSEL_MIN_DIST_M,
    "curvature_threshold": _KEY_CURVATURE_THRESHOLD,
    "CURVATURE_THRESHOLD": _KEY_CURVATURE_THRESHOLD,
    _KEY_BANANA_SURF_RADIUS: _KEY_BANANA_SURF_RADIUS,
    "target_lcfs_max_major_radius_m": _KEY_TARGET_LCFS_MAX_MAJOR_RADIUS_M,
    "TARGET_LCFS_MAX_MAJOR_RADIUS_M": _KEY_TARGET_LCFS_MAX_MAJOR_RADIUS_M,
    "target_lcfs_max_minor_radius_m": _KEY_TARGET_LCFS_MAX_MINOR_RADIUS_M,
    "TARGET_LCFS_MAX_MINOR_RADIUS_M": _KEY_TARGET_LCFS_MAX_MINOR_RADIUS_M,
})


def _translate_layer(layer: Mapping[str, Any] | None) -> dict[str, Any]:
    if layer is None:
        return {}
    translated: dict[str, Any] = {}
    unknown: set[str] = set()
    for key, value in layer.items():
        canonical = WIRE_NAME_ALIASES.get(key)
        if canonical is None:
            unknown.add(str(key))
            continue
        translated[canonical] = value
    if unknown:
        raise ValueError(
            "Constraint wire-name layer has unknown fields: "
            f"{', '.join(sorted(unknown))}"
        )
    return translated

_LADDER_SOURCES: tuple[str, ...] = (
    CONSTRAINT_SOURCE_PROFILE,
    CONSTRAINT_SOURCE_SPEC_JSON,
    CONSTRAINT_SOURCE_CLI,
)


def hardware_default_contract() -> dict[str, float]:
    return {
        _KEY_VACUUM_VESSEL_MAJOR_RADIUS_M: float(_hc.VACUUM_VESSEL_MAJOR_RADIUS_M),
        _KEY_VACUUM_VESSEL_MINOR_RADIUS_M: float(_hc.VACUUM_VESSEL_MINOR_RADIUS_M),
        _KEY_BANANA_WINDING_SURFACE_MAJOR_RADIUS_M: float(
            _hc.BANANA_WINDING_SURFACE_MAJOR_RADIUS_M
        ),
        _KEY_TF_CURRENT_A: float(_hc.TF_CURRENT_CW_DEFAULT_A),
        _KEY_BANANA_CURRENT_MAX_A: float(_hc.BANANA_CURRENT_HARD_LIMIT_A),
        _KEY_COIL_LENGTH_TARGET_M: float(_hc.COIL_LENGTH_TARGET_M),
        _KEY_CC_THRESHOLD: float(_hc.COIL_COIL_MIN_DIST_M),
        _KEY_COIL_PLASMA_MIN_DIST_M: float(_hc.COIL_PLASMA_MIN_DIST_M),
        _KEY_PLASMA_VESSEL_MIN_DIST_M: float(_hc.PLASMA_VESSEL_MIN_DIST_M),
        _KEY_CURVATURE_THRESHOLD: float(_hc.MAX_CURVATURE_INV_M),
        _KEY_BANANA_SURF_RADIUS: float(_hc.BANANA_WINDING_MINOR_RADIUS_M),
        _KEY_TARGET_LCFS_MAX_MAJOR_RADIUS_M: float(_hc.TARGET_LCFS_MAX_MAJOR_RADIUS_M),
        _KEY_TARGET_LCFS_MAX_MINOR_RADIUS_M: float(_hc.TARGET_LCFS_MAX_MINOR_RADIUS_M),
    }


def _coerce(key: str, value: Any) -> float:
    if value is None:
        raise ValueError(f"Constraint field {key!r} cannot be None.")
    return float(value)


def _assert_known_keys(source_label: str, layer: Mapping[str, Any]) -> None:
    unknown = sorted(set(layer) - set(CONSTRAINT_FIELD_TYPES))
    if unknown:
        raise ValueError(
            f"Constraint {source_label} layer has unknown fields: "
            f"{', '.join(unknown)}"
        )


def _apply_layer(
    *,
    contract: dict[str, float],
    trace: dict[str, str],
    source_label: str,
    layer: Mapping[str, Any],
) -> None:
    _assert_known_keys(source_label, layer)
    for key, raw_value in layer.items():
        if raw_value is None:
            continue
        if key in FIXED_GEOMETRY_KEYS:
            raise ValueError(
                f"Fixed-geometry field {key!r} cannot be overridden via "
                f"{source_label}."
            )
        contract[key] = _coerce(key, raw_value)
        trace[key] = source_label


def _validate_engineering_values(contract: dict[str, float]) -> None:
    _hc.validate_tf_current_limit(contract[_KEY_TF_CURRENT_A])
    _hc.validate_banana_winding_surface_radius(contract[_KEY_BANANA_SURF_RADIUS])
    if contract[_KEY_BANANA_CURRENT_MAX_A] <= 0.0:
        raise ValueError(
            "BANANA_CURRENT_MAX_A must be positive; got "
            f"{contract[_KEY_BANANA_CURRENT_MAX_A]!r}."
        )
    if contract[_KEY_COIL_LENGTH_TARGET_M] <= 0.0:
        raise ValueError("COIL_LENGTH_TARGET_M must be positive.")
    if contract[_KEY_CC_THRESHOLD] <= 0.0:
        raise ValueError("CC_THRESHOLD must be positive.")
    if contract[_KEY_CC_THRESHOLD] < _hc.COIL_COIL_MIN_DIST_M:
        raise ValueError(
            "CC_THRESHOLD is below the hardware floor "
            f"{_hc.COIL_COIL_MIN_DIST_M:.3f} m."
        )
    if contract[_KEY_COIL_PLASMA_MIN_DIST_M] <= 0.0:
        raise ValueError("COIL_PLASMA_MIN_DIST_M must be positive.")
    if contract[_KEY_COIL_PLASMA_MIN_DIST_M] < _hc.COIL_PLASMA_MIN_DIST_M:
        raise ValueError(
            "COIL_PLASMA_MIN_DIST_M is below the hardware floor "
            f"{_hc.COIL_PLASMA_MIN_DIST_M:.3f} m."
        )
    if contract[_KEY_PLASMA_VESSEL_MIN_DIST_M] <= 0.0:
        raise ValueError("PLASMA_VESSEL_MIN_DIST_M must be positive.")
    if contract[_KEY_PLASMA_VESSEL_MIN_DIST_M] < _hc.PLASMA_VESSEL_MIN_DIST_M:
        raise ValueError(
            "PLASMA_VESSEL_MIN_DIST_M is below the hardware floor "
            f"{_hc.PLASMA_VESSEL_MIN_DIST_M:.3f} m."
        )
    if contract[_KEY_CURVATURE_THRESHOLD] <= 0.0:
        raise ValueError("CURVATURE_THRESHOLD must be positive.")


def _validate_target_plasma_ceiling(contract: dict[str, float]) -> None:
    _hc.validate_target_lcfs_major_radius(contract[_KEY_TARGET_LCFS_MAX_MAJOR_RADIUS_M])
    _hc.validate_target_lcfs_minor_radius(contract[_KEY_TARGET_LCFS_MAX_MINOR_RADIUS_M])


def resolve_constraint_contract(
    *,
    profile: Mapping[str, Any] | None = None,
    spec_json: Mapping[str, Any] | None = None,
    cli_overrides: Mapping[str, Any] | None = None,
) -> tuple[Mapping[str, float], Mapping[str, str]]:
    """Resolve the full constraint contract from layered inputs.

    Parameters
    ----------
    profile, spec_json, cli_overrides
        Partial mappings with engineering-default or target-plasma-ceiling
        field overrides. ``None`` values inside a layer are ignored. Fixed
        geometry keys are rejected at every layer.
    Returns
    -------
    contract, trace
        Immutable mappings. ``contract`` is keyed by the canonical field
        names above; ``trace`` reports the provenance of each field.
    """
    contract = hardware_default_contract()
    trace: dict[str, str] = {key: CONSTRAINT_SOURCE_HARDWARE for key in contract}

    for source_label, layer in zip(
        _LADDER_SOURCES,
        (profile, spec_json, cli_overrides),
    ):
        if layer is None:
            continue
        _apply_layer(
            contract=contract,
            trace=trace,
            source_label=source_label,
            layer=layer,
        )

    _validate_engineering_values(contract)
    if contract[_KEY_BANANA_CURRENT_MAX_A] > _hc.BANANA_CURRENT_HARD_LIMIT_A:
        raise ValueError(
            "BANANA_CURRENT_MAX_A exceeds the hardware limit "
            f"{_hc.BANANA_CURRENT_HARD_LIMIT_A:.0f} A."
        )
    if contract[_KEY_COIL_LENGTH_TARGET_M] > _hc.COIL_LENGTH_HARD_LIMIT_M:
        raise ValueError(
            "COIL_LENGTH_TARGET_M exceeds the hardware limit "
            f"{_hc.COIL_LENGTH_HARD_LIMIT_M:.3f} m."
        )
    if contract[_KEY_CURVATURE_THRESHOLD] > _hc.MAX_CURVATURE_INV_M:
        raise ValueError(
            "CURVATURE_THRESHOLD exceeds the hardware limit "
            f"{_hc.MAX_CURVATURE_INV_M:.0f} m^-1."
        )
    _validate_target_plasma_ceiling(contract)

    return MappingProxyType(contract), MappingProxyType(trace)


def resolve_constraint_contract_from_wire_names(
    *,
    profile: Mapping[str, Any] | None = None,
    spec_json: Mapping[str, Any] | None = None,
    cli_overrides: Mapping[str, Any] | None = None,
) -> tuple[Mapping[str, float], Mapping[str, str]]:
    """Like :func:`resolve_constraint_contract` but accepts legacy wire names.

    The historical ``major_radius`` mirror key maps to the fixed vacuum-vessel
    major-radius field, so the shared resolver rejects it like every other
    fixed-geometry override. All unknown wire-name keys also raise.
    """
    return resolve_constraint_contract(
        profile=_translate_layer(profile),
        spec_json=_translate_layer(spec_json),
        cli_overrides=_translate_layer(cli_overrides),
    )


def _canonical_payload(contract: Mapping[str, Any]) -> dict[str, float]:
    missing = sorted(set(CONSTRAINT_FIELD_TYPES) - set(contract))
    if missing:
        raise ValueError(
            f"Cannot hash partial constraint contract; missing: {', '.join(missing)}"
        )
    return {key: float(contract[key]) for key in sorted(contract)}


def _hash_payload(payload: Mapping[str, float]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def compute_constraint_contract_hash(contract: Mapping[str, Any]) -> str:
    return _hash_payload(_canonical_payload(contract))


def contract_is_all_hardware_defaults(trace: Mapping[str, str]) -> bool:
    return all(source == CONSTRAINT_SOURCE_HARDWARE for source in trace.values())


def build_constraint_metadata(
    contract: Mapping[str, Any],
    *,
    profile_name: str,
    override_reason: str | None = None,
    trace: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    effective_values = _canonical_payload(contract)
    metadata: dict[str, Any] = {
        "CONSTRAINT_PROFILE": profile_name,
        "EFFECTIVE_VALUES": effective_values,
        "OVERRIDE_REASON": override_reason,
        "CONTRACT_HASH": _hash_payload(effective_values),
        "CONTRACT_SCHEMA_VERSION": CONSTRAINT_SCHEMA_VERSION,
    }
    if trace is not None:
        metadata["CONSTRAINT_PROVENANCE"] = dict(trace)
    return metadata


__all__ = [
    "CONSTRAINT_FIELD_TYPES",
    "CONSTRAINT_SCHEMA_VERSION",
    "CONSTRAINT_SOURCE_CLI",
    "CONSTRAINT_SOURCE_HARDWARE",
    "CONSTRAINT_SOURCE_PROFILE",
    "CONSTRAINT_SOURCE_SPEC_JSON",
    "ENGINEERING_DEFAULT_KEYS",
    "FIXED_GEOMETRY_KEYS",
    "TARGET_PLASMA_CEILING_KEYS",
    "WIRE_NAME_ALIASES",
    "build_constraint_metadata",
    "compute_constraint_contract_hash",
    "contract_is_all_hardware_defaults",
    "hardware_default_contract",
    "resolve_constraint_contract",
    "resolve_constraint_contract_from_wire_names",
]
