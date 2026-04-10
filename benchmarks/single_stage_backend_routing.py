"""Shared benchmark helpers for single-stage inner Boozer backend routing."""

from __future__ import annotations


def resolve_boozer_optimizer_backend(
    optimizer_backend: str,
    boozer_optimizer_backend: str | None,
) -> str:
    effective_backend = (
        optimizer_backend
        if boozer_optimizer_backend is None
        else boozer_optimizer_backend
    )
    if effective_backend != "ondevice":
        raise ValueError(
            "Single-stage JAX benchmark probes require "
            "boozer_optimizer_backend='ondevice'."
        )
    return effective_backend


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
    least_squares_algorithm: str | None = None,
) -> str:
    if boozer_optimizer_backend != "ondevice":
        raise ValueError(
            "Single-stage JAX benchmark probes require "
            "boozer_optimizer_backend='ondevice'."
        )
    if least_squares_algorithm is None:
        least_squares_algorithm = resolve_boozer_least_squares_algorithm(
            boozer_optimizer_backend
        )
    if least_squares_algorithm == "lm":
        if limited_memory:
            raise ValueError(
                "least_squares_algorithm='lm' is incompatible with "
                "limited_memory=True."
            )
        return "lm-ondevice"
    return "lbfgs-ondevice" if limited_memory else "bfgs-ondevice"


def resolve_boozer_least_squares_algorithm(
    boozer_optimizer_backend: str,
    least_squares_algorithm: str | None = None,
) -> str:
    if boozer_optimizer_backend != "ondevice":
        raise ValueError(
            "Single-stage JAX benchmark probes require "
            "boozer_optimizer_backend='ondevice'."
        )
    if least_squares_algorithm is not None:
        return least_squares_algorithm
    return "lm"
