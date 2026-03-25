"""Shared benchmark helpers for single-stage inner Boozer backend routing."""

from __future__ import annotations


def resolve_boozer_optimizer_backend(
    optimizer_backend: str,
    boozer_optimizer_backend: str | None,
) -> str:
    if boozer_optimizer_backend is None:
        return optimizer_backend
    return boozer_optimizer_backend


def resolve_boozer_limited_memory(
    boozer_optimizer_backend: str,
    boozer_limited_memory_requested: bool,
) -> bool:
    return bool(
        boozer_optimizer_backend == "ondevice"
        and boozer_limited_memory_requested
    )


def resolve_boozer_optimizer_method(boozer_optimizer_backend: str) -> str:
    if boozer_optimizer_backend == "scipy":
        return "bfgs"
    if boozer_optimizer_backend == "hybrid":
        return "bfgs-hybrid"
    return "bfgs-ondevice"
