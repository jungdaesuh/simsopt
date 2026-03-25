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
    """Match the current benchmark/example plumbing for effective limited memory."""
    return bool(
        boozer_optimizer_backend == "ondevice"
        and boozer_limited_memory_requested
    )


def resolve_boozer_optimizer_method(
    boozer_optimizer_backend: str,
    *,
    limited_memory: bool = False,
) -> str:
    if boozer_optimizer_backend == "scipy":
        return "lbfgs" if limited_memory else "bfgs"
    if boozer_optimizer_backend == "hybrid":
        if limited_memory:
            raise ValueError(
                "optimizer_backend='hybrid' does not support limited_memory=True."
            )
        return "bfgs-hybrid"
    return "lbfgs-ondevice" if limited_memory else "bfgs-ondevice"
