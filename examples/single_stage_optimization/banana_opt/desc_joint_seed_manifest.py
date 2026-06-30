"""Seed manifest contract for DESC joint banana runner tests and preflight."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeVar

from banana_opt.coil_groups import (
    infer_manifest_from_legacy_counts,
    read_manifest_from_results,
    validate_manifest_against_coils,
)
from banana_opt.desc_joint_field_inventory import (
    DescJointFieldInventory,
    load_desc_joint_field_inventory,
)
from banana_opt.desc_joint_io import (
    read_json_mapping,
    sha256_file as _sha256_file,
)

DESC_JOINT_SEED_MANIFEST_SCHEMA_VERSION = "desc_joint_seed_manifest_v1"

SeedSurfaceKind = Literal["boozer_surface", "bare_surface"]
CoilGroupSource = Literal["source_results", "manifest", "unavailable"]
_SURFACE_KINDS = frozenset({"boozer_surface", "bare_surface"})
_BOOZER_SURFACE_CLASSES = frozenset({"BoozerSurface", "BoozerSurfaceFiniteI"})
_BIOT_SAVART_CLASS = "BiotSavart"
_CURVE_CWS_CLASS = "CurveCWSFourierCPP"
_MATERIALIZED_CWS_GEOMETRY_MODE = "materialized_cws"
_SINGLE_STAGE_BANANA_GEOMETRY_MODE_KEY = "SINGLE_STAGE_BANANA_GEOMETRY_MODE"
MetadataValueT = TypeVar("MetadataValueT", int, bool)


@dataclass(frozen=True, slots=True)
class DescJointSeedCoilGroup:
    name: str
    count: int

    def to_json_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "count": self.count,
        }


@dataclass(frozen=True, slots=True)
class DescJointSeedCandidate:
    label: str
    group: str
    surface_path: Path
    field_path: Path
    surface_kind: SeedSurfaceKind
    coil_groups: tuple[DescJointSeedCoilGroup, ...] = ()
    coil_group_source: CoilGroupSource = "unavailable"
    source_nfp: int | None = None
    source_stellarator_symmetry: bool | None = None
    source_results_path: Path | None = None
    state_path: Path | None = None
    poincare_metrics_path: Path | None = None
    poincare_png_path: Path | None = None

    def source_checksums(self) -> dict[str, str]:
        paths = {
            "surface": self.surface_path,
            "field": self.field_path,
            "source_results": self.source_results_path,
            "state": self.state_path,
            "poincare_metrics": self.poincare_metrics_path,
            "poincare_png": self.poincare_png_path,
        }
        return {
            key: _sha256_file(path)
            for key, path in paths.items()
            if path is not None
        }

    def to_json_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "group": self.group,
            "surface": os.fspath(self.surface_path),
            "field": os.fspath(self.field_path),
            "surface_kind": self.surface_kind,
            "coil_groups": [
                coil_group.to_json_dict()
                for coil_group in self.coil_groups
            ],
            "coil_group_source": self.coil_group_source,
            "source_nfp": self.source_nfp,
            "source_stellarator_symmetry": self.source_stellarator_symmetry,
            "source_results": _optional_fspath(self.source_results_path),
            "state": _optional_fspath(self.state_path),
            "poincare_metrics": _optional_fspath(self.poincare_metrics_path),
            "poincare_png": _optional_fspath(self.poincare_png_path),
            "source_checksums": self.source_checksums(),
        }


@dataclass(frozen=True, slots=True)
class DescJointSeedManifest:
    manifest_path: Path
    candidates: tuple[DescJointSeedCandidate, ...]

    def candidate_by_label(self, label: str) -> DescJointSeedCandidate:
        for candidate in self.candidates:
            if candidate.label == label:
                return candidate
        raise KeyError(f"Unknown DESC joint seed candidate {label!r}.")

    def to_input_contract(self) -> dict[str, object]:
        return {
            "schema_version": DESC_JOINT_SEED_MANIFEST_SCHEMA_VERSION,
            "manifest_path": os.fspath(self.manifest_path),
            "candidates": [
                candidate.to_json_dict()
                for candidate in self.candidates
            ],
        }


def load_desc_joint_seed_manifest(path: str | Path) -> DescJointSeedManifest:
    manifest_path = Path(path).expanduser().resolve()
    payload = _read_json_mapping(manifest_path)
    if payload.get("schema_version") != DESC_JOINT_SEED_MANIFEST_SCHEMA_VERSION:
        raise ValueError("Unexpected DESC joint seed manifest schema_version.")
    raw_candidates = payload.get("candidates")
    if isinstance(raw_candidates, str) or not isinstance(raw_candidates, Sequence):
        raise ValueError("DESC joint seed manifest candidates must be a list.")
    candidates = tuple(
        _coerce_candidate(manifest_path, candidate)
        for candidate in raw_candidates
    )
    labels = [candidate.label for candidate in candidates]
    duplicates = sorted(label for label in set(labels) if labels.count(label) > 1)
    if duplicates:
        raise ValueError(
            "DESC joint seed manifest has duplicate candidate labels: "
            f"{', '.join(duplicates)}."
        )
    return DescJointSeedManifest(manifest_path=manifest_path, candidates=candidates)


def _coerce_candidate(
    manifest_path: Path,
    raw_candidate: object,
) -> DescJointSeedCandidate:
    if not isinstance(raw_candidate, Mapping):
        raise ValueError("DESC joint seed manifest candidate must be an object.")
    label = _require_nonempty_string(raw_candidate, "label")
    group = _require_nonempty_string(raw_candidate, "group")
    surface_kind = _coerce_surface_kind(label, raw_candidate.get("surface_kind"))
    surface_path = _require_existing_path(manifest_path, raw_candidate, "surface")
    field_path = _require_existing_path(manifest_path, raw_candidate, "field")
    _validate_surface_artifact_kind(label, surface_path, surface_kind)
    _validate_field_artifact_kind(label, field_path)
    source_results_path = _optional_existing_path(
        manifest_path,
        raw_candidate,
        "source_results",
    )
    field_inventory = load_desc_joint_field_inventory(field_path)
    _validate_materialized_cws_field_contract(
        label,
        field_path=field_path,
        source_results_path=source_results_path,
        field_inventory=field_inventory,
    )
    source_nfp, source_stellarator_symmetry = _coerce_candidate_source_metadata(
        raw_candidate,
        source_results_path=source_results_path,
        surface_path=surface_path,
    )
    coil_groups, coil_group_source = _coerce_candidate_coil_groups(
        raw_candidate,
        source_results_path=source_results_path,
        field_coil_count=field_inventory.coil_count,
    )
    return DescJointSeedCandidate(
        label=label,
        group=group,
        surface_path=surface_path,
        field_path=field_path,
        surface_kind=surface_kind,
        coil_groups=coil_groups,
        coil_group_source=coil_group_source,
        source_nfp=source_nfp,
        source_stellarator_symmetry=source_stellarator_symmetry,
        source_results_path=source_results_path,
        state_path=_optional_existing_path(manifest_path, raw_candidate, "state"),
        poincare_metrics_path=_optional_existing_path(
            manifest_path,
            raw_candidate,
            "poincare_metrics",
        ),
        poincare_png_path=_optional_existing_path(
            manifest_path,
            raw_candidate,
            "poincare_png",
        ),
    )


def _validate_materialized_cws_field_contract(
    label: str,
    *,
    field_path: Path,
    source_results_path: Path | None,
    field_inventory: DescJointFieldInventory,
) -> None:
    if source_results_path is None:
        return
    source_results = _read_json_mapping(source_results_path)
    raw_geometry_mode = source_results.get(_SINGLE_STAGE_BANANA_GEOMETRY_MODE_KEY)
    if raw_geometry_mode is None:
        return
    if not isinstance(raw_geometry_mode, str):
        raise ValueError(
            "source_results field "
            f"{_SINGLE_STAGE_BANANA_GEOMETRY_MODE_KEY!r} must be a string."
        )
    if (
        raw_geometry_mode == _MATERIALIZED_CWS_GEOMETRY_MODE
        and field_inventory.cws_curve_count == 0
    ):
        raise ValueError(
            f"candidate {label!r} source_results declares "
            f"{_SINGLE_STAGE_BANANA_GEOMETRY_MODE_KEY}="
            f"{_MATERIALIZED_CWS_GEOMETRY_MODE!r}, but {field_path} contains "
            f"0 {_CURVE_CWS_CLASS} curves. Use the CWS field artifact "
            "(for example slid_cws_field*.json) instead of flattened "
            "biot_savart_opt.json."
        )


def _coerce_candidate_coil_groups(
    raw_candidate: Mapping[str, object],
    *,
    source_results_path: Path | None,
    field_coil_count: int,
) -> tuple[tuple[DescJointSeedCoilGroup, ...], CoilGroupSource]:
    raw_coil_groups = raw_candidate.get("coil_groups")
    if source_results_path is not None:
        source_results_groups = _coil_groups_from_source_results(
            source_results_path,
            field_coil_count=field_coil_count,
        )
        manifest_groups = _coerce_coil_groups(raw_coil_groups)
        if manifest_groups and manifest_groups != source_results_groups:
            raise ValueError(
                "candidate field 'coil_groups' must match source_results "
                "COIL_GROUPS when both are provided."
            )
        return source_results_groups, "source_results"
    coil_groups = _coerce_coil_groups(raw_coil_groups)
    if coil_groups:
        _validate_coil_group_total(coil_groups, field_coil_count=field_coil_count)
        return coil_groups, "manifest"
    return (), "unavailable"


def _coerce_candidate_source_metadata(
    raw_candidate: Mapping[str, object],
    *,
    source_results_path: Path | None,
    surface_path: Path,
) -> tuple[int | None, bool | None]:
    nfp_sources: list[tuple[str, int]] = []
    symmetry_sources: list[tuple[str, bool]] = []
    candidate_nfp = _optional_positive_int(raw_candidate, "nfp")
    if candidate_nfp is not None:
        nfp_sources.append(("candidate.nfp", candidate_nfp))
    candidate_symmetry = _optional_bool(raw_candidate, "stellarator_symmetry")
    if candidate_symmetry is not None:
        symmetry_sources.append(
            ("candidate.stellarator_symmetry", candidate_symmetry)
        )
    if source_results_path is not None:
        source_results = _read_json_mapping(source_results_path)
        source_results_nfp = _optional_positive_int_from_keys(
            source_results,
            keys=("NFP", "nfp"),
            context="source_results",
        )
        if source_results_nfp is not None:
            nfp_sources.append(("source_results.NFP", source_results_nfp))
        source_results_symmetry = _optional_bool_from_keys(
            source_results,
            keys=(
                "STELLSYM",
                "stellsym",
                "STELLARATOR_SYMMETRY",
                "stellarator_symmetry",
            ),
            context="source_results",
        )
        if source_results_symmetry is not None:
            symmetry_sources.append(
                ("source_results.stellarator_symmetry", source_results_symmetry)
            )
    surface_nfp, surface_symmetry = _surface_source_metadata(surface_path)
    if surface_nfp is not None:
        nfp_sources.append(("surface.nfp", surface_nfp))
    if surface_symmetry is not None:
        symmetry_sources.append(("surface.stellarator_symmetry", surface_symmetry))
    return (
        _merge_optional_metadata(nfp_sources, field_name="NFP"),
        _merge_optional_metadata(
            symmetry_sources,
            field_name="stellarator_symmetry",
        ),
    )


def _surface_source_metadata(surface_path: Path) -> tuple[int | None, bool | None]:
    payload = json.loads(surface_path.read_text(encoding="utf-8"))
    surface_like_payloads = _collect_surface_like_mappings(payload)
    nfp_sources = [
        _optional_positive_int_from_keys(
            surface_payload,
            keys=("nfp", "NFP"),
            context="surface",
        )
        for surface_payload in surface_like_payloads
    ]
    symmetry_sources = [
        _optional_bool_from_keys(
            surface_payload,
            keys=("stellsym", "stellarator_symmetry", "STELLSYM"),
            context="surface",
        )
        for surface_payload in surface_like_payloads
    ]
    return (
        _merge_optional_metadata(
            [
                (f"surface[{index}].nfp", nfp)
                for index, nfp in enumerate(nfp_sources)
                if nfp is not None
            ],
            field_name="surface NFP",
        ),
        _merge_optional_metadata(
            [
                (f"surface[{index}].stellarator_symmetry", symmetry)
                for index, symmetry in enumerate(symmetry_sources)
                if symmetry is not None
            ],
            field_name="surface stellarator_symmetry",
        ),
    )


def _collect_surface_like_mappings(value: object) -> tuple[Mapping[str, object], ...]:
    surface_payloads: list[Mapping[str, object]] = []
    if isinstance(value, Mapping):
        class_name = value.get("@class")
        if isinstance(class_name, str) and (
            class_name.startswith("Surface")
            or class_name in _BOOZER_SURFACE_CLASSES
        ):
            surface_payloads.append(value)
        for child in value.values():
            surface_payloads.extend(_collect_surface_like_mappings(child))
    elif not isinstance(value, str) and isinstance(value, Sequence):
        for child in value:
            surface_payloads.extend(_collect_surface_like_mappings(child))
    return tuple(surface_payloads)


def _optional_positive_int_from_keys(
    payload: Mapping[str, object],
    *,
    keys: tuple[str, ...],
    context: str,
) -> int | None:
    values = [
        (key, _optional_positive_int(payload, key))
        for key in keys
        if payload.get(key) is not None
    ]
    if not values:
        return None
    return _merge_optional_metadata(
        [(f"{context}.{key}", value) for key, value in values],
        field_name=f"{context} NFP",
    )


def _optional_bool_from_keys(
    payload: Mapping[str, object],
    *,
    keys: tuple[str, ...],
    context: str,
) -> bool | None:
    values = [
        (key, _optional_bool(payload, key))
        for key in keys
        if payload.get(key) is not None
    ]
    if not values:
        return None
    return _merge_optional_metadata(
        [(f"{context}.{key}", value) for key, value in values],
        field_name=f"{context} stellarator_symmetry",
    )


def _merge_optional_metadata(
    values: list[tuple[str, MetadataValueT]],
    *,
    field_name: str,
) -> MetadataValueT | None:
    if not values:
        return None
    first_source, first_value = values[0]
    mismatches = [
        f"{source}={value!r}"
        for source, value in values[1:]
        if value != first_value
    ]
    if mismatches:
        raise ValueError(
            f"DESC joint seed source {field_name} metadata disagrees: "
            f"{first_source}={first_value!r}; {', '.join(mismatches)}."
        )
    return first_value


def _coil_groups_from_source_results(
    source_results_path: Path,
    *,
    field_coil_count: int,
) -> tuple[DescJointSeedCoilGroup, ...]:
    source_results = _read_json_mapping(source_results_path)
    manifest = read_manifest_from_results(source_results)
    if manifest is None:
        manifest = infer_manifest_from_legacy_counts(
            source_results,
            total_loaded_coils=field_coil_count,
        )
    else:
        validate_manifest_against_coils(
            manifest,
            total_loaded_coils=field_coil_count,
        )
    return tuple(
        DescJointSeedCoilGroup(name=group.role, count=group.count)
        for group in manifest.groups
    )


def _coerce_coil_groups(raw_coil_groups: object) -> tuple[DescJointSeedCoilGroup, ...]:
    if raw_coil_groups is None:
        return ()
    if isinstance(raw_coil_groups, str) or not isinstance(raw_coil_groups, Sequence):
        raise ValueError("candidate field 'coil_groups' must be a list of objects.")
    coil_groups = tuple(_coerce_coil_group(raw_group) for raw_group in raw_coil_groups)
    names = [coil_group.name for coil_group in coil_groups]
    duplicates = sorted(name for name in set(names) if names.count(name) > 1)
    if duplicates:
        raise ValueError(
            "candidate field 'coil_groups' has duplicate group names: "
            f"{', '.join(duplicates)}."
        )
    return coil_groups


def _validate_coil_group_total(
    coil_groups: tuple[DescJointSeedCoilGroup, ...],
    *,
    field_coil_count: int,
) -> None:
    total = sum(coil_group.count for coil_group in coil_groups)
    if total != field_coil_count:
        raise ValueError(
            "candidate field 'coil_groups' count does not match field coil count: "
            f"{total} vs {field_coil_count}."
        )


def _coerce_coil_group(raw_group: object) -> DescJointSeedCoilGroup:
    if not isinstance(raw_group, Mapping):
        raise ValueError("candidate coil group entry must be an object.")
    name = _require_nonempty_string(raw_group, "name")
    count = _nonnegative_int(raw_group, "count")
    return DescJointSeedCoilGroup(name=name, count=count)


def _coerce_surface_kind(label: str, raw_kind: object) -> SeedSurfaceKind:
    if raw_kind == "boozer_surface":
        return "boozer_surface"
    if raw_kind == "bare_surface":
        return "bare_surface"
    choices = ", ".join(sorted(_SURFACE_KINDS))
    raise ValueError(
        f"candidate {label!r} surface_kind must be one of {{{choices}}}; "
        f"got {raw_kind!r}."
    )


def _read_json_mapping(path: Path) -> Mapping[str, object]:
    return read_json_mapping(
        path,
        error_message=lambda payload: (
            "DESC joint seed manifest must be a JSON object; got "
            f"{type(payload).__name__}."
        ),
    )


def _require_nonempty_string(payload: Mapping[str, object], field_name: str) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str) or value == "":
        raise ValueError(f"candidate field {field_name!r} must be a nonempty string.")
    return value


def _nonnegative_int(payload: Mapping[str, object], field_name: str) -> int:
    value = payload.get(field_name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"candidate field {field_name!r} must be an integer.")
    if value < 0:
        raise ValueError(f"candidate field {field_name!r} must be nonnegative.")
    return value


def _optional_positive_int(
    payload: Mapping[str, object],
    field_name: str,
) -> int | None:
    value = payload.get(field_name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"candidate field {field_name!r} must be an integer.")
    if value <= 0:
        raise ValueError(f"candidate field {field_name!r} must be positive.")
    return value


def _optional_bool(
    payload: Mapping[str, object],
    field_name: str,
) -> bool | None:
    value = payload.get(field_name)
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError(f"candidate field {field_name!r} must be boolean.")
    return value


def _resolve_path(manifest_path: Path, raw_path: object, field_name: str) -> Path:
    if not isinstance(raw_path, str) or raw_path == "":
        raise ValueError(
            f"candidate field {field_name!r} must be a nonempty path string."
        )
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def _require_existing_path(
    manifest_path: Path,
    payload: Mapping[str, object],
    field_name: str,
) -> Path:
    path = _resolve_path(manifest_path, payload.get(field_name), field_name)
    if not path.exists():
        raise ValueError(f"candidate field {field_name!r} does not exist: {path}.")
    if not path.is_file():
        raise ValueError(f"candidate field {field_name!r} must be a file: {path}.")
    return path


def _validate_surface_artifact_kind(
    label: str,
    surface_path: Path,
    surface_kind: SeedSurfaceKind,
) -> None:
    class_names = _json_class_names(surface_path)
    has_boozer_surface = bool(class_names & _BOOZER_SURFACE_CLASSES)
    has_surface = any(class_name.startswith("Surface") for class_name in class_names)
    has_biot_savart = _BIOT_SAVART_CLASS in class_names
    if surface_kind == "boozer_surface":
        if not has_boozer_surface:
            raise ValueError(
                f"candidate {label!r} declares surface_kind='boozer_surface' but "
                f"{surface_path} does not contain a BoozerSurface object."
            )
        if not has_biot_savart:
            raise ValueError(
                f"candidate {label!r} BoozerSurface JSON must embed a BiotSavart "
                f"field: {surface_path}."
            )
        return
    if has_boozer_surface:
        raise ValueError(
            f"candidate {label!r} declares surface_kind='bare_surface' but "
            f"{surface_path} contains a BoozerSurface object."
        )
    if not has_surface:
        raise ValueError(
            f"candidate {label!r} bare surface JSON must contain a surface object: "
            f"{surface_path}."
        )


def _validate_field_artifact_kind(label: str, field_path: Path) -> None:
    class_names = _json_class_names(field_path)
    if _BIOT_SAVART_CLASS not in class_names:
        raise ValueError(
            f"candidate {label!r} field JSON must contain a BiotSavart object: "
            f"{field_path}."
        )


def _json_class_names(path: Path) -> frozenset[str]:
    return frozenset(
        _collect_json_class_names(json.loads(path.read_text(encoding="utf-8")))
    )


def _collect_json_class_names(value: object) -> set[str]:
    names: set[str] = set()
    if isinstance(value, Mapping):
        class_name = value.get("@class")
        if isinstance(class_name, str):
            names.add(class_name)
        for child in value.values():
            names.update(_collect_json_class_names(child))
    elif not isinstance(value, str) and isinstance(value, Sequence):
        for child in value:
            names.update(_collect_json_class_names(child))
    return names


def _optional_existing_path(
    manifest_path: Path,
    payload: Mapping[str, object],
    field_name: str,
) -> Path | None:
    if payload.get(field_name) is None:
        return None
    return _require_existing_path(manifest_path, payload, field_name)


def _optional_fspath(path: Path | None) -> str | None:
    return None if path is None else os.fspath(path)


__all__ = [
    "DESC_JOINT_SEED_MANIFEST_SCHEMA_VERSION",
    "CoilGroupSource",
    "DescJointSeedCandidate",
    "DescJointSeedCoilGroup",
    "DescJointSeedManifest",
    "SeedSurfaceKind",
    "load_desc_joint_seed_manifest",
]
