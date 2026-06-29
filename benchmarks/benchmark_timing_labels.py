"""Shared timing-classification labels for benchmark artifacts and launchers."""

from __future__ import annotations

from typing import Final, Literal, TypedDict


TimingClassification = Literal[
    "gpu-target-only",
    "mixed-parity-reference",
    "reference-only",
]

GPU_TARGET_ONLY: Final[TimingClassification] = "gpu-target-only"
MIXED_PARITY_REFERENCE: Final[TimingClassification] = "mixed-parity-reference"
REFERENCE_ONLY: Final[TimingClassification] = "reference-only"
TIMING_CLASSIFICATION_VALUES: Final[tuple[TimingClassification, ...]] = (
    GPU_TARGET_ONLY,
    MIXED_PARITY_REFERENCE,
    REFERENCE_ONLY,
)


class BenchmarkTimingLabel(TypedDict):
    timing_classification: TimingClassification
    timing_classification_values: tuple[TimingClassification, ...]
    timing_includes_gpu_target: bool
    timing_includes_cpu_reference: bool
    supports_performance_headline: bool
    headline_timing_classification: TimingClassification | None
    timing_label_note: str


def benchmark_timing_label(
    classification: TimingClassification,
    *,
    includes_gpu_target: bool,
    includes_cpu_reference: bool,
    supports_performance_headline: bool,
    headline_timing_classification: TimingClassification | None,
    note: str,
) -> BenchmarkTimingLabel:
    return {
        "timing_classification": classification,
        "timing_classification_values": TIMING_CLASSIFICATION_VALUES,
        "timing_includes_gpu_target": includes_gpu_target,
        "timing_includes_cpu_reference": includes_cpu_reference,
        "supports_performance_headline": supports_performance_headline,
        "headline_timing_classification": headline_timing_classification,
        "timing_label_note": note,
    }
