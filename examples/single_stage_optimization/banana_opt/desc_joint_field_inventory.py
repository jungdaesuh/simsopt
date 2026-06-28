"""Static BiotSavart field inventory for DESC joint preflight contracts."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

_BIOT_SAVART_CLASS = "BiotSavart"
_COIL_CLASS = "Coil"
_CURRENT_CLASS = "Current"
_CURVE_CWS_CLASS = "CurveCWSFourierCPP"
_CURVE_XYZ_CLASS = "CurveXYZFourier"
_SCALED_CURRENT_CLASS = "ScaledCurrent"


@dataclass(frozen=True, slots=True)
class DescJointFieldInventory:
    field_path: Path
    biot_savart_name: str
    coil_count: int
    cws_curve_count: int
    xyz_curve_count: int
    current_values_A: tuple[float, ...]
    current_signs: tuple[str, ...]
    current_sign_counts: Mapping[str, int]

    def to_json_dict(self) -> dict[str, object]:
        return {
            "field_path": os.fspath(self.field_path),
            "biot_savart_name": self.biot_savart_name,
            "coil_count": self.coil_count,
            "cws_curve_count": self.cws_curve_count,
            "xyz_curve_count": self.xyz_curve_count,
            "current_values_A": list(self.current_values_A),
            "current_signs": list(self.current_signs),
            "current_sign_counts": dict(self.current_sign_counts),
        }


def load_desc_joint_field_inventory(path: str | Path) -> DescJointFieldInventory:
    field_path = Path(path).expanduser().resolve()
    payload = _read_json_mapping(field_path)
    objects = _simson_objects(payload)
    biot_savart_name, biot_savart = _single_biot_savart_object(payload, objects)
    coil_refs = _reference_list(biot_savart.get("coils"))
    field_class_counts = _referenced_object_class_counts(objects, coil_refs)
    current_values = tuple(
        _coil_current_value_A(objects, coil_ref)
        for coil_ref in coil_refs
    )
    current_signs = tuple(_current_sign(current_A) for current_A in current_values)
    return DescJointFieldInventory(
        field_path=field_path,
        biot_savart_name=biot_savart_name,
        coil_count=len(coil_refs),
        cws_curve_count=field_class_counts.get(_CURVE_CWS_CLASS, 0),
        xyz_curve_count=field_class_counts.get(_CURVE_XYZ_CLASS, 0),
        current_values_A=current_values,
        current_signs=current_signs,
        current_sign_counts=_sign_counts(current_signs),
    )


def _read_json_mapping(path: Path) -> Mapping[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"BiotSavart inventory input must be a JSON object: {path}.")
    return payload


def _simson_objects(payload: Mapping[str, object]) -> Mapping[str, object]:
    raw_objects = payload.get("simsopt_objs")
    if isinstance(raw_objects, Mapping):
        return raw_objects
    if payload.get("@class") == _BIOT_SAVART_CLASS:
        return {"BiotSavart": payload}
    raise ValueError("BiotSavart inventory input must contain simsopt_objs.")


def _single_biot_savart_object(
    payload: Mapping[str, object],
    objects: Mapping[str, object],
) -> tuple[str, Mapping[str, object]]:
    if payload.get("@class") == _BIOT_SAVART_CLASS:
        return "BiotSavart", payload
    matches = [
        (name, value)
        for name, value in objects.items()
        if isinstance(name, str)
        and isinstance(value, Mapping)
        and value.get("@class") == _BIOT_SAVART_CLASS
    ]
    if len(matches) != 1:
        raise ValueError(
            "BiotSavart inventory input must contain exactly one BiotSavart object; "
            f"found {len(matches)}."
        )
    return matches[0]


def _reference_list(value: object) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError("BiotSavart coils must be a list of object references.")
    return tuple(_reference_name(item) for item in value)


def _reference_name(value: object) -> str:
    if not isinstance(value, Mapping):
        raise ValueError("SIMSON reference must be an object.")
    if value.get("$type") != "ref":
        raise ValueError("SIMSON reference object must have $type='ref'.")
    name = value.get("value")
    if not isinstance(name, str) or name == "":
        raise ValueError("SIMSON reference value must be a nonempty string.")
    return name


def _object_by_name(
    objects: Mapping[str, object],
    name: str,
    *,
    expected_class: str,
) -> Mapping[str, object]:
    value = objects.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"SIMSON object {name!r} is missing.")
    if value.get("@class") != expected_class:
        raise ValueError(
            f"SIMSON object {name!r} must have @class={expected_class!r}; "
            f"got {value.get('@class')!r}."
        )
    return value


def _coil_current_value_A(objects: Mapping[str, object], coil_name: str) -> float:
    coil = _object_by_name(objects, coil_name, expected_class=_COIL_CLASS)
    current_name = _reference_name(coil.get("current"))
    return _current_value_A(objects, current_name, seen=frozenset())


def _current_value_A(
    objects: Mapping[str, object],
    current_name: str,
    *,
    seen: frozenset[str],
) -> float:
    if current_name in seen:
        raise ValueError(f"SIMSON current graph contains a cycle at {current_name!r}.")
    raw_current = objects.get(current_name)
    if not isinstance(raw_current, Mapping):
        raise ValueError(f"SIMSON current object {current_name!r} is missing.")
    current_class = raw_current.get("@class")
    if current_class == _CURRENT_CLASS:
        return _finite_float(raw_current.get("current"), field_name=current_name)
    if current_class == _SCALED_CURRENT_CLASS:
        base_name = _reference_name(raw_current.get("current_to_scale"))
        scale = _finite_float(raw_current.get("scale"), field_name=current_name)
        return scale * _current_value_A(
            objects,
            base_name,
            seen=frozenset([*seen, current_name]),
        )
    raise ValueError(
        f"SIMSON current object {current_name!r} must be Current or ScaledCurrent; "
        f"got {current_class!r}."
    )


def _finite_float(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"SIMSON numeric field {field_name!r} must be numeric.")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"SIMSON numeric field {field_name!r} must be finite.")
    return converted


def _current_sign(current_A: float) -> str:
    if current_A > 0.0:
        return "positive"
    if current_A < 0.0:
        return "negative"
    return "zero"


def _sign_counts(current_signs: Sequence[str]) -> dict[str, int]:
    counts = {"negative": 0, "zero": 0, "positive": 0}
    for current_sign in current_signs:
        counts[current_sign] += 1
    return counts


def _referenced_object_class_counts(
    objects: Mapping[str, object],
    root_names: Sequence[str],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    seen: set[str] = set()
    for root_name in root_names:
        _collect_referenced_object_class_counts(objects, root_name, seen, counts)
    return counts


def _collect_referenced_object_class_counts(
    objects: Mapping[str, object],
    object_name: str,
    seen: set[str],
    counts: dict[str, int],
) -> None:
    if object_name in seen:
        return
    seen.add(object_name)
    value = objects.get(object_name)
    if not isinstance(value, Mapping):
        raise ValueError(f"SIMSON object {object_name!r} is missing.")
    class_name = value.get("@class")
    if isinstance(class_name, str):
        counts[class_name] = counts.get(class_name, 0) + 1
    for referenced_name in _reference_names_in_value(value):
        _collect_referenced_object_class_counts(
            objects,
            referenced_name,
            seen,
            counts,
        )


def _reference_names_in_value(value: object) -> tuple[str, ...]:
    if isinstance(value, Mapping):
        if value.get("$type") == "ref":
            referenced_name = value.get("value")
            if isinstance(referenced_name, str) and referenced_name != "":
                return (referenced_name,)
            raise ValueError("SIMSON reference value must be a nonempty string.")
        names: list[str] = []
        for child in value.values():
            names.extend(_reference_names_in_value(child))
        return tuple(names)
    if not isinstance(value, str) and isinstance(value, Sequence):
        names: list[str] = []
        for child in value:
            names.extend(_reference_names_in_value(child))
        return tuple(names)
    return ()


__all__ = [
    "DescJointFieldInventory",
    "load_desc_joint_field_inventory",
]
