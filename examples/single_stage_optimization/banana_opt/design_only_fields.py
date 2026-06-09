from __future__ import annotations

import json
from pathlib import Path
from typing import Final, Mapping

from simsopt.field import BiotSavart


DESIGN_ONLY_RESULTS_KEY: Final[str] = "DESIGN_ONLY_NO_TOPOLOGY_GATE"
DESIGN_ONLY_REASON_KEY: Final[str] = "DESIGN_ONLY_REASON"
DESIGN_ONLY_EVIDENCE_KEY: Final[str] = "DESIGN_ONLY_EVIDENCE"
DESIGN_ONLY_ALLOWED_USE_KEY: Final[str] = "DESIGN_ONLY_ALLOWED_USE"
DESIGN_ONLY_FORBIDDEN_USE_KEY: Final[str] = "DESIGN_ONLY_FORBIDDEN_USE"
_DESIGN_ONLY_FIELD_ATTR: Final[str] = "_design_only_no_topology_gate"


class DesignOnlyTopologyFieldError(RuntimeError):
    """Raised when a prescribed design field reaches a topology gate."""


def mark_design_only_field(bs: BiotSavart, *, reason: str) -> None:
    setattr(bs, _DESIGN_ONLY_FIELD_ATTR, True)
    setattr(bs, f"{_DESIGN_ONLY_FIELD_ATTR}_reason", str(reason))


def build_design_only_results_fields(*, reason: str) -> dict[str, object]:
    return {
        DESIGN_ONLY_RESULTS_KEY: True,
        DESIGN_ONLY_REASON_KEY: str(reason),
        DESIGN_ONLY_EVIDENCE_KEY: (
            "Prescribed proxy line-current field is a design/source model; "
            "BiotSavart JSON does not preserve the in-memory marker."
        ),
        DESIGN_ONLY_ALLOWED_USE_KEY: (
            "coil geometry inspection",
            "field visualization with explicit diagnostic override",
        ),
        DESIGN_ONLY_FORBIDDEN_USE_KEY: (
            "topology gate",
            "Poincare validation",
            "promotion or production handoff",
        ),
    }


def load_design_only_results_metadata(out_dir: str | Path) -> dict[str, object] | None:
    """Load the run-level sidecar that persists the design-only field marker."""
    results_path = Path(out_dir) / "results.json"
    if not results_path.exists():
        return None
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {results_path}.")
    return {str(key): value for key, value in payload.items()}


def field_is_design_only(
    bs: object,
    results: Mapping[str, object] | None,
) -> bool:
    in_memory_marker = getattr(bs, _DESIGN_ONLY_FIELD_ATTR, False) is True
    persisted_marker = (
        False
        if results is None
        else results.get(DESIGN_ONLY_RESULTS_KEY, False) is True
    )
    return in_memory_marker or persisted_marker


def assert_topology_field_allowed(
    bs: object,
    results: Mapping[str, object] | None,
    *,
    allow_design_only_field: bool,
    consumer: str,
) -> None:
    if allow_design_only_field or not field_is_design_only(bs, results):
        return
    reason = None if results is None else results.get(DESIGN_ONLY_REASON_KEY)
    detail = "" if reason is None or reason == "" else f" Reason: {reason}."
    raise DesignOnlyTopologyFieldError(
        f"{consumer} cannot use a design-only prescribed proxy field for "
        f"topology or Poincare validation.{detail}"
    )
