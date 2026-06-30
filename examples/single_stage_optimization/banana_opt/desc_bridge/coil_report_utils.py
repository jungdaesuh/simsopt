"""Shared report helpers for DESC bridge coil conversion."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from banana_opt.desc_bridge.artifact_metadata import (
    EMPTY_DESC_BRIDGE_ARTIFACT_METADATA,
    DescBridgeArtifactMetadata,
)


class CoilReportEntry(Protocol):
    group: str
    current_sign: str


COIL_CONVENTION_REPORT: Mapping[str, str] = {
    "coil_curve_orientation": "preserve_source_parameterization",
    "current_sign": "preserve_signed_current_A",
    "hbt_tf_cw_current_sign": "negative",
}


def coil_convention_report() -> dict[str, str]:
    return dict(COIL_CONVENTION_REPORT)


def current_sign(current_A: float) -> str:
    if current_A > 0.0:
        return "positive"
    if current_A < 0.0:
        return "negative"
    return "zero"


def group_counts(
    entries: Sequence[CoilReportEntry],
    *,
    group_order: Sequence[str],
) -> dict[str, int]:
    counts = {group: 0 for group in group_order}
    for entry in entries:
        counts[entry.group] = counts.get(entry.group, 0) + 1
    return counts


def current_sign_counts(entries: Sequence[CoilReportEntry]) -> dict[str, int]:
    counts = {"negative": 0, "zero": 0, "positive": 0}
    for entry in entries:
        counts[entry.current_sign] += 1
    return counts


def optional_delta(
    final_value: float | None,
    initial_value: float | None,
) -> float | None:
    if final_value is None or initial_value is None:
        return None
    return final_value - initial_value


def artifact_metadata_or_empty(
    artifact_metadata: DescBridgeArtifactMetadata | None,
) -> DescBridgeArtifactMetadata:
    if artifact_metadata is None:
        return EMPTY_DESC_BRIDGE_ARTIFACT_METADATA
    return artifact_metadata


def validate_group_names(coil_groups: Mapping[str, object], *, context: str) -> None:
    for group_name in coil_groups:
        if not isinstance(group_name, str) or group_name == "":
            raise ValueError(f"{context} coil group names must be nonempty strings.")


__all__ = [
    "CoilReportEntry",
    "artifact_metadata_or_empty",
    "coil_convention_report",
    "current_sign",
    "current_sign_counts",
    "group_counts",
    "optional_delta",
    "validate_group_names",
]
