from __future__ import annotations

from dataclasses import dataclass
from typing import Collection, Iterable, Literal, Mapping

from banana_opt.hardware_contracts import (
    BANANA_CURRENT_HARD_LIMIT_A,
    COIL_COIL_MIN_DIST_M,
    COIL_LENGTH_HARD_LIMIT_M,
    COIL_PLASMA_MIN_DIST_M,
    MAX_CURVATURE_INV_M,
    PLASMA_VESSEL_MIN_DIST_M,
    POLOIDAL_EXTENT_HALF_WIDTH_RAD,
    TF_CURRENT_HARD_LIMIT_A,
)

ConstraintKind = Literal["lower_bound", "upper_bound", "box_bound"]
ConstraintTarget = Literal["penalty", "alm", "artifact"]
# Traversal policy is search-role metadata, not a universal mode dispatcher.
# It buckets realized status reporting for every mode, while only the
# penalty/box_bound/forbidden subset becomes a hard search-time bound.
TraversalPolicy = Literal["allowed", "forbidden"]


@dataclass(frozen=True)
class HardwareConstraintSpec:
    name: str
    kind: ConstraintKind
    threshold: float
    applies_to: frozenset[ConstraintTarget]
    traversal_policy: TraversalPolicy


HARDWARE_CONSTRAINT_SCHEMA: tuple[HardwareConstraintSpec, ...] = (
    HardwareConstraintSpec(
        name="coil_coil_spacing",
        kind="lower_bound",
        threshold=COIL_COIL_MIN_DIST_M,
        applies_to=frozenset({"penalty", "alm", "artifact"}),
        traversal_policy="allowed",
    ),
    HardwareConstraintSpec(
        name="coil_surface_spacing",
        kind="lower_bound",
        threshold=COIL_PLASMA_MIN_DIST_M,
        applies_to=frozenset({"penalty", "alm", "artifact"}),
        traversal_policy="allowed",
    ),
    HardwareConstraintSpec(
        name="surface_vessel_spacing",
        kind="lower_bound",
        threshold=PLASMA_VESSEL_MIN_DIST_M,
        applies_to=frozenset({"penalty", "alm", "artifact"}),
        traversal_policy="allowed",
    ),
    HardwareConstraintSpec(
        name="max_curvature",
        kind="upper_bound",
        threshold=MAX_CURVATURE_INV_M,
        applies_to=frozenset({"penalty", "alm", "artifact"}),
        traversal_policy="allowed",
    ),
    HardwareConstraintSpec(
        name="coil_length",
        kind="upper_bound",
        threshold=COIL_LENGTH_HARD_LIMIT_M,
        applies_to=frozenset({"alm", "artifact"}),
        traversal_policy="allowed",
    ),
    HardwareConstraintSpec(
        name="poloidal_extent",
        kind="upper_bound",
        threshold=POLOIDAL_EXTENT_HALF_WIDTH_RAD,
        applies_to=frozenset({"penalty", "alm", "artifact"}),
        traversal_policy="allowed",
    ),
    HardwareConstraintSpec(
        name="banana_current",
        kind="box_bound",
        threshold=BANANA_CURRENT_HARD_LIMIT_A,
        applies_to=frozenset({"penalty", "alm", "artifact"}),
        traversal_policy="forbidden",
    ),
    HardwareConstraintSpec(
        name="tf_current",
        kind="box_bound",
        threshold=TF_CURRENT_HARD_LIMIT_A,
        applies_to=frozenset({"artifact"}),
        traversal_policy="forbidden",
    ),
)

_SCHEMA_BY_NAME = {spec.name: spec for spec in HARDWARE_CONSTRAINT_SCHEMA}
_ARTIFACT_VALUE_FIELD_BY_NAME = {
    "coil_coil_spacing": ("curve_curve_min_dist", "CURVE_CURVE_MIN_DIST"),
    "coil_surface_spacing": ("curve_surface_min_dist", "CURVE_SURFACE_MIN_DIST"),
    "surface_vessel_spacing": ("surface_vessel_min_dist", "SURFACE_VESSEL_MIN_DIST"),
    "max_curvature": ("max_curvature", "MAX_CURVATURE"),
    "coil_length": ("coil_length", "COIL_LENGTH"),
    "poloidal_extent": ("poloidal_extent_rad", "POLOIDAL_EXTENT_RAD"),
    "banana_current": ("banana_current_A", "BANANA_CURRENT_A"),
    "tf_current": ("tf_current_A", "TF_CURRENT_A"),
}
_ARTIFACT_THRESHOLD_FIELD_BY_NAME = {
    "coil_length": ("length_target", "LENGTH_TARGET"),
    "poloidal_extent": (
        "poloidal_extent_threshold_rad",
        "POLOIDAL_EXTENT_THRESHOLD_RAD",
    ),
    "banana_current": ("banana_current_max_A", "BANANA_CURRENT_MAX_A"),
    "tf_current": ("tf_current_limit_A", "TF_CURRENT_LIMIT_A"),
}
_BOOTABILITY_STATUS_FIELD_NAMES = (
    "BOOZER_BOOTABLE",
    "IOTA_FEASIBLE",
    "BOOTABILITY_REASON",
    "BOOTABILITY_STAGE",
    "BOOTABILITY_TARGET_IOTA",
    "BOOTABILITY_SOLVED_IOTA",
    "BOOTABILITY_SELF_INTERSECTING",
)
_BOOTABILITY_DIAGNOSTIC_FIELD_NAMES = (
    "BOOTABILITY_SOLVE_SUCCESS",
    "BOOTABILITY_ABS_IOTA_ERROR",
    "BOOTABILITY_ERROR_TYPE",
    "BOOTABILITY_ERROR_MESSAGE",
)
_BOOTABILITY_PAYLOAD_FIELD_NAMES = (
    _BOOTABILITY_STATUS_FIELD_NAMES + _BOOTABILITY_DIAGNOSTIC_FIELD_NAMES
)
_BOOTABILITY_PROVENANCE_FIELD_NAMES = (
    "STAGE2_BS_PATH",
    "STAGE2_RESULTS_PATH",
)
_RECOVERY_STATUS_FIELD_NAMES = (
    "RECOVERY_ATTEMPTED",
    "RECOVERY_SUCCEEDED",
    "RECOVERY_ITERS",
    "RECOVERY_TERMINATION_REASON",
)


def hardware_constraint_schema() -> tuple[HardwareConstraintSpec, ...]:
    return HARDWARE_CONSTRAINT_SCHEMA


def get_hardware_constraint_spec(name: str) -> HardwareConstraintSpec:
    try:
        return _SCHEMA_BY_NAME[name]
    except KeyError as exc:
        raise KeyError(f"Unknown hardware constraint {name!r}.") from exc


def hardware_constraint_specs(
    *,
    applies_to: ConstraintTarget | None = None,
    names: Collection[str] | None = None,
    traversal_policy: TraversalPolicy | None = None,
) -> tuple[HardwareConstraintSpec, ...]:
    allowed_names = None if names is None else set(names)
    filtered: list[HardwareConstraintSpec] = []
    for spec in HARDWARE_CONSTRAINT_SCHEMA:
        if applies_to is not None and applies_to not in spec.applies_to:
            continue
        if allowed_names is not None and spec.name not in allowed_names:
            continue
        if traversal_policy is not None and spec.traversal_policy != traversal_policy:
            continue
        filtered.append(spec)
    return tuple(filtered)


def hardware_constraint_artifact_field_names(
    *,
    names: Collection[str] | None = None,
) -> tuple[str, ...]:
    return tuple(
        spec.name for spec in hardware_constraint_specs(applies_to="artifact", names=names)
    )


def hardware_constraint_artifact_payload_field_names(
    *,
    prefix: str = "",
    names: Collection[str] | None = None,
    include_status: bool = True,
) -> tuple[str, ...]:
    field_names: list[str] = []
    for name in hardware_constraint_artifact_field_names(names=names):
        _, value_field = _ARTIFACT_VALUE_FIELD_BY_NAME[name]
        field_names.append(f"{prefix}{value_field}")
        threshold_entry = _ARTIFACT_THRESHOLD_FIELD_BY_NAME.get(name)
        if threshold_entry is not None:
            _, threshold_field = threshold_entry
            field_names.append(f"{prefix}{threshold_field}")
    if include_status:
        field_names.extend(
            (
                f"{prefix}HARDWARE_CONSTRAINTS_OK",
                f"{prefix}HARDWARE_CONSTRAINT_VIOLATIONS",
            )
        )
    return tuple(field_names)


def build_hardware_constraint_artifact_payload_fields(
    hardware_snapshot: Mapping[str, object] | None,
    *,
    prefix: str = "",
    names: Collection[str] | None = None,
) -> dict[str, object]:
    if hardware_snapshot is None:
        return {
            field_name: None
            for field_name in hardware_constraint_artifact_payload_field_names(
                prefix=prefix,
                names=names,
            )
        }

    payload_fields: dict[str, object] = {}
    for name in hardware_constraint_artifact_field_names(names=names):
        snapshot_key, value_field = _ARTIFACT_VALUE_FIELD_BY_NAME[name]
        value = hardware_snapshot.get(snapshot_key)
        payload_fields[f"{prefix}{value_field}"] = (
            None if value is None else float(value)
        )
        threshold_entry = _ARTIFACT_THRESHOLD_FIELD_BY_NAME.get(name)
        if threshold_entry is None:
            continue
        threshold_key, threshold_field = threshold_entry
        threshold_value = hardware_snapshot.get(threshold_key)
        payload_fields[f"{prefix}{threshold_field}"] = (
            None if threshold_value is None else float(threshold_value)
        )

    artifact_hardware_status = hardware_snapshot.get("artifact_hardware_status")
    payload_fields[f"{prefix}HARDWARE_CONSTRAINTS_OK"] = (
        None
        if artifact_hardware_status is None
        else bool(artifact_hardware_status["success"])
    )
    payload_fields[f"{prefix}HARDWARE_CONSTRAINT_VIOLATIONS"] = (
        None
        if artifact_hardware_status is None
        else list(artifact_hardware_status["violations"])
    )
    return payload_fields


def bootability_recovery_payload_field_names(
    *,
    prefix: str = "",
    include_provenance: bool = True,
    include_recovery: bool = True,
) -> tuple[str, ...]:
    field_names = [f"{prefix}{field}" for field in _BOOTABILITY_PAYLOAD_FIELD_NAMES]
    if include_provenance:
        field_names.extend(
            f"{prefix}{field}" for field in _BOOTABILITY_PROVENANCE_FIELD_NAMES
        )
    if include_recovery:
        field_names.extend(
            f"{prefix}{field}" for field in _RECOVERY_STATUS_FIELD_NAMES
        )
    return tuple(field_names)


def build_bootability_recovery_payload_fields(
    bootability_status: Mapping[str, object] | None,
    *,
    prefix: str = "",
    stage2_bs_path: str | None = None,
    stage2_results_path: str | None = None,
    include_recovery: bool = True,
    recovery_attempted: bool | None = None,
    recovery_succeeded: bool | None = None,
    recovery_iters: int | None = None,
    recovery_termination_reason: str | None = None,
) -> dict[str, object]:
    payload_fields = {
        field_name: None
        for field_name in bootability_recovery_payload_field_names(
            prefix=prefix,
            include_recovery=include_recovery,
        )
    }
    if bootability_status is not None:
        for field_name in _BOOTABILITY_PAYLOAD_FIELD_NAMES:
            payload_fields[f"{prefix}{field_name}"] = bootability_status.get(field_name)
    payload_fields[f"{prefix}STAGE2_BS_PATH"] = stage2_bs_path
    payload_fields[f"{prefix}STAGE2_RESULTS_PATH"] = stage2_results_path
    if include_recovery:
        payload_fields[f"{prefix}RECOVERY_ATTEMPTED"] = (
            None if recovery_attempted is None else bool(recovery_attempted)
        )
        payload_fields[f"{prefix}RECOVERY_SUCCEEDED"] = (
            None if recovery_succeeded is None else bool(recovery_succeeded)
        )
        payload_fields[f"{prefix}RECOVERY_ITERS"] = (
            None if recovery_iters is None else int(recovery_iters)
        )
        payload_fields[f"{prefix}RECOVERY_TERMINATION_REASON"] = (
            None
            if recovery_termination_reason is None
            else str(recovery_termination_reason)
        )
    return payload_fields


def alm_constraint_name(spec: HardwareConstraintSpec) -> str:
    if spec.kind == "lower_bound":
        return spec.name
    if spec.kind == "box_bound":
        return f"{spec.name}_upper_bound"
    if spec.name == "coil_length":
        return "coil_length_upper_bound"
    if spec.kind == "upper_bound":
        return spec.name
    raise ValueError(f"Unsupported hardware constraint kind {spec.kind!r}.")


def hardware_constraint_alm_names(
    *,
    names: Collection[str] | None = None,
) -> tuple[str, ...]:
    return tuple(
        alm_constraint_name(spec)
        for spec in hardware_constraint_specs(applies_to="alm", names=names)
    )


def hardware_constraint_penalty_box_bound_names(
    *,
    names: Collection[str] | None = None,
    traversal_policy: TraversalPolicy | None = None,
) -> tuple[str, ...]:
    """Return the schema subset that becomes a penalty-search runtime bound.

    This is intentionally narrower than the full hardware contract surface.
    ALM inclusion comes from ``hardware_constraint_alm_names``; artifact and
    final certification consume the schema through the status/payload builders.
    """
    return tuple(
        spec.name
        for spec in hardware_constraint_specs(
            applies_to="penalty",
            names=names,
            traversal_policy=traversal_policy,
        )
        if spec.kind == "box_bound"
    )


def resolve_penalty_box_bound_threshold(
    name: str,
    *,
    requested_threshold: float | None = None,
) -> float:
    spec = get_hardware_constraint_spec(name)
    if (
        "penalty" not in spec.applies_to
        or spec.kind != "box_bound"
        or spec.traversal_policy != "forbidden"
    ):
        raise ValueError(
            f"{name!r} is not a traversal-forbidden penalty box-bound hardware constraint."
        )
    if requested_threshold is None:
        return float(spec.threshold)
    return min(float(spec.threshold), float(requested_threshold))


def _resolved_threshold(
    spec: HardwareConstraintSpec,
    threshold_overrides: Mapping[str, float] | None,
) -> float:
    if threshold_overrides is None or spec.name not in threshold_overrides:
        return float(spec.threshold)
    return float(threshold_overrides[spec.name])


def hardware_constraint_signed_value(
    spec: HardwareConstraintSpec,
    value: float,
    *,
    threshold_overrides: Mapping[str, float] | None = None,
) -> float:
    threshold = _resolved_threshold(spec, threshold_overrides)
    scalar_value = float(value)
    if spec.kind == "lower_bound":
        return threshold - scalar_value
    if spec.kind == "upper_bound":
        return scalar_value - threshold
    if spec.kind == "box_bound":
        return abs(scalar_value) - threshold
    raise ValueError(f"Unsupported hardware constraint kind {spec.kind!r}.")


def hardware_constraint_violation(
    spec: HardwareConstraintSpec,
    value: float,
    *,
    threshold_overrides: Mapping[str, float] | None = None,
) -> float:
    return max(
        0.0,
        hardware_constraint_signed_value(
            spec,
            value,
            threshold_overrides=threshold_overrides,
        ),
    )


def format_hardware_constraint_violation(
    spec: HardwareConstraintSpec,
    value: float,
    *,
    threshold_overrides: Mapping[str, float] | None = None,
) -> str:
    threshold = _resolved_threshold(spec, threshold_overrides)
    scalar_value = float(value)
    if spec.kind == "lower_bound":
        return f"{spec.name} {scalar_value:.6f} below threshold {threshold:.6f}"
    if spec.kind == "upper_bound":
        return f"{spec.name} {scalar_value:.6f} exceeds threshold {threshold:.6f}"
    if spec.kind == "box_bound":
        return f"|{spec.name}| {abs(scalar_value):.6f} exceeds threshold {threshold:.6f}"
    raise ValueError(f"Unsupported hardware constraint kind {spec.kind!r}.")


def _empty_constraint_status() -> dict[str, object]:
    return {
        "success": True,
        "violations": [],
        "constraints": {},
    }


def _constraint_status_entry(
    spec: HardwareConstraintSpec,
    *,
    threshold: float,
    value: float,
    signed_value: float,
    violation: float,
    success: bool,
) -> dict[str, object]:
    return {
        "name": spec.name,
        "kind": spec.kind,
        "threshold": float(threshold),
        "value": float(value),
        "signed_value": float(signed_value),
        "violation": float(violation),
        "success": bool(success),
        "applies_to": tuple(sorted(spec.applies_to)),
        "traversal_policy": spec.traversal_policy,
    }


def build_hardware_constraint_status(
    measured_values: Mapping[str, float | None],
    *,
    applies_to: ConstraintTarget,
    names: Collection[str] | None = None,
    threshold_overrides: Mapping[str, float] | None = None,
) -> dict[str, object]:
    constraints: dict[str, dict[str, object]] = {}
    violations: list[str] = []
    # Keep the top-level success/violations view intact while also surfacing
    # schema-driven traversal semantics for callers that need to distinguish
    # search-time forbidden violations from allowed soft violations.
    traversal_statuses: dict[TraversalPolicy, dict[str, object]] = {
        "allowed": _empty_constraint_status(),
        "forbidden": _empty_constraint_status(),
    }
    for spec in hardware_constraint_specs(applies_to=applies_to, names=names):
        value = measured_values.get(spec.name)
        if value is None:
            continue
        threshold = _resolved_threshold(spec, threshold_overrides)
        signed_value = hardware_constraint_signed_value(
            spec,
            value,
            threshold_overrides=threshold_overrides,
        )
        violation = hardware_constraint_violation(
            spec,
            value,
            threshold_overrides=threshold_overrides,
        )
        success = violation == 0.0
        constraint_entry = _constraint_status_entry(
            spec,
            threshold=threshold,
            value=value,
            signed_value=signed_value,
            violation=violation,
            success=success,
        )
        constraints[spec.name] = constraint_entry
        traversal_status = traversal_statuses[spec.traversal_policy]
        traversal_status["constraints"][spec.name] = constraint_entry
        if not success:
            violation_message = format_hardware_constraint_violation(
                spec,
                value,
                threshold_overrides=threshold_overrides,
            )
            violations.append(violation_message)
            traversal_status["success"] = False
            traversal_status["violations"].append(violation_message)
    return {
        "success": not violations,
        "violations": violations,
        "constraints": constraints,
        "allowed_traversal_status": traversal_statuses["allowed"],
        "forbidden_traversal_status": traversal_statuses["forbidden"],
    }


def build_threshold_overrides(items: Iterable[tuple[str, float | None]]) -> dict[str, float]:
    return {
        str(name): float(value)
        for name, value in items
        if value is not None
    }
