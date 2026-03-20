"""Shared benchmark configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkConfig:
    label: str
    ncoils: int
    nphi: int
    ntheta: int
    mpol: int
    ntor: int
    nfp: int = 1
    nquad: int = 128


DEFAULT_CONFIGS = (
    BenchmarkConfig("Small (4 coils, 15x15)", 4, 15, 15, 2, 2, nquad=64),
    BenchmarkConfig("Medium (6 coils, 15x15)", 6, 15, 15, 4, 4, nquad=128),
    BenchmarkConfig("HBT-like (12 coils, 15x15)", 12, 15, 15, 4, 4, nquad=128),
    BenchmarkConfig("Prod-grid (12 coils, 64x64)", 12, 64, 64, 4, 4, nquad=128),
    BenchmarkConfig("Columbia (12 coils, 128x64)", 12, 128, 64, 8, 6, nquad=200),
    BenchmarkConfig("Full-HBT (22 coils, 128x64)", 22, 128, 64, 8, 6, nquad=200),
)


def available_config_labels() -> tuple[str, ...]:
    return tuple(config.label for config in DEFAULT_CONFIGS)


def resolve_configs(labels: list[str] | tuple[str, ...] | None) -> tuple[BenchmarkConfig, ...]:
    if not labels:
        return DEFAULT_CONFIGS
    configs_by_label = {config.label: config for config in DEFAULT_CONFIGS}
    unknown = [label for label in labels if label not in configs_by_label]
    if unknown:
        available = ", ".join(available_config_labels())
        unknown_display = ", ".join(unknown)
        raise ValueError(
            f"Unknown benchmark config(s): {unknown_display}. Available configs: {available}"
        )
    return tuple(configs_by_label[label] for label in labels)
