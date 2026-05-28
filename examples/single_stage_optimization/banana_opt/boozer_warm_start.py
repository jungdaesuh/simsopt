from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np

BOOZER_WARM_START_STATE_SCHEMA_VERSION = 1


def warm_start_boozer_state_path(surface_path: str | Path) -> Path:
    path = Path(surface_path)
    if path.name.endswith("_boozer_surface.json"):
        return path.with_name(
            path.name.removesuffix("_boozer_surface.json") + "_boozer_state.json"
        )
    return path.with_name(path.name + ".boozer_state.json")


def _finite_scalar(value: object) -> float | None:
    if value is None:
        return None
    scalar = float(value)
    if not np.isfinite(scalar):
        return None
    return scalar


def boozer_surface_solved_state(boozer_surface: object) -> tuple[float, float] | None:
    res = getattr(boozer_surface, "res", None)
    if not isinstance(res, Mapping):
        return None
    iota = _finite_scalar(res.get("iota"))
    G = _finite_scalar(res.get("G"))
    if iota is None or G is None:
        return None
    return iota, G


def load_boozer_solved_state_sidecar(
    surface_path: str | Path,
) -> tuple[float, float] | None:
    state_path = warm_start_boozer_state_path(surface_path)
    if not state_path.is_file():
        return None
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Boozer warm-start state must be a JSON object: {state_path}")
    iota = _finite_scalar(payload.get("iota"))
    G = _finite_scalar(payload.get("G"))
    if iota is None or G is None:
        raise ValueError(
            f"Boozer warm-start state must contain finite iota and G: {state_path}"
        )
    return iota, G


def write_boozer_solved_state_sidecar(
    surface_path: str | Path,
    boozer_surface: object,
    *,
    interchange_manifest: Mapping[str, object] | None = None,
) -> Path | None:
    solved_state = boozer_surface_solved_state(boozer_surface)
    if solved_state is None:
        return None
    iota, G = solved_state
    state_path = warm_start_boozer_state_path(surface_path)
    payload = {
        "schema_version": BOOZER_WARM_START_STATE_SCHEMA_VERSION,
        "surface_path": str(Path(surface_path).name),
        "iota": iota,
        "G": G,
    }
    if interchange_manifest is not None:
        payload["interchange_manifest"] = {
            **dict(interchange_manifest),
            "boozer_surface_module": type(boozer_surface).__module__,
            "boozer_surface_class": type(boozer_surface).__name__,
            "boozer_surface_has_i_field": hasattr(boozer_surface, "I"),
        }
    state_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return state_path


def save_boozer_surface_with_state(
    boozer_surface: object,
    surface_path: str | Path,
    *,
    interchange_manifest: Mapping[str, object] | None = None,
    surface_validator=None,
) -> Path | None:
    boozer_surface.save(str(surface_path))
    if surface_validator is not None:
        validation = surface_validator(surface_path)
        if not validation["passed"]:
            raise ValueError(
                "Baseline-replayable Boozer artifacts must use "
                "simsopt.geo.BoozerSurface without an I field: "
                f"{surface_path}"
            )
    return write_boozer_solved_state_sidecar(
        surface_path,
        boozer_surface,
        interchange_manifest=interchange_manifest,
    )
