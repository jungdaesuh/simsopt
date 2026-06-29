"""Single-stage JAX optimizer routing policy.

This module is dependency-light so examples, benchmarks, and subprocess probes
can share the same backend resolution rules without importing the optimizer
runtime or benchmark package.
"""

from __future__ import annotations

from simsopt_jax.geo._optimizer_backend_choices import (
    TARGET_PUBLIC_LBFGS_OPTIMIZER_BACKENDS,
)

JAX_TARGET_PUBLIC_LBFGS_OPTIMIZER_BACKENDS = TARGET_PUBLIC_LBFGS_OPTIMIZER_BACKENDS
JAX_TARGET_LBFGS_OUTER_OPTIMIZER_BACKENDS = (
    frozenset({"ondevice"}) | JAX_TARGET_PUBLIC_LBFGS_OPTIMIZER_BACKENDS
)
JAX_TARGET_OUTER_OPTIMIZER_BACKENDS = (
    frozenset({"ondevice", "scipy-jax", "scipy-jax-decomposed"})
    | JAX_TARGET_PUBLIC_LBFGS_OPTIMIZER_BACKENDS
)
JAX_SCIPY_OUTER_OPTIMIZER_METHOD = "lbfgs-scipy-jax"
JAX_DECOMPOSED_SCIPY_OUTER_OPTIMIZER_METHOD = "lbfgs-scipy-jax-decomposed"
JAX_FULL_GRAPH_SCIPY_OUTER_OPTIMIZER_BACKEND = "scipy-jax-fullgraph"
JAX_FULL_GRAPH_SCIPY_OUTER_OPTIMIZER_METHOD = "lbfgs-scipy-jax-fullgraph"
JAX_HOST_CONTROL_OUTER_OPTIMIZER_BACKEND = "host-jax"
JAX_ONDEVICE_BOOZER_OUTER_OPTIMIZER_BACKENDS = JAX_TARGET_OUTER_OPTIMIZER_BACKENDS
JAX_TARGET_OUTER_MAXLS_BACKENDS = JAX_TARGET_OUTER_OPTIMIZER_BACKENDS | {
    JAX_FULL_GRAPH_SCIPY_OUTER_OPTIMIZER_BACKEND
}
JAX_BOOZER_INNER_OPTIMIZER_BACKENDS = frozenset({"host-jax", "ondevice", "scipy"})

__all__ = (
    "JAX_DECOMPOSED_SCIPY_OUTER_OPTIMIZER_METHOD",
    "JAX_FULL_GRAPH_SCIPY_OUTER_OPTIMIZER_BACKEND",
    "JAX_FULL_GRAPH_SCIPY_OUTER_OPTIMIZER_METHOD",
    "JAX_HOST_CONTROL_OUTER_OPTIMIZER_BACKEND",
    "JAX_ONDEVICE_BOOZER_OUTER_OPTIMIZER_BACKENDS",
    "JAX_SCIPY_OUTER_OPTIMIZER_METHOD",
    "JAX_TARGET_LBFGS_OUTER_OPTIMIZER_BACKENDS",
    "JAX_BOOZER_INNER_OPTIMIZER_BACKENDS",
    "JAX_TARGET_OUTER_MAXLS_BACKENDS",
    "JAX_TARGET_OUTER_OPTIMIZER_BACKENDS",
    "JAX_TARGET_PUBLIC_LBFGS_OPTIMIZER_BACKENDS",
    "resolve_boozer_least_squares_algorithm",
    "resolve_boozer_limited_memory",
    "resolve_boozer_optimizer_backend",
    "resolve_boozer_optimizer_method",
    "resolve_single_stage_jax_boozer_optimizer_backend",
)


def resolve_single_stage_jax_boozer_optimizer_backend(
    field_backend,
    optimizer_backend,
    boozer_optimizer_backend=None,
) -> str | None:
    """Resolve the inner Boozer LS backend for the single-stage JAX lane."""
    if field_backend != "jax":
        return None
    if optimizer_backend == "scipy":
        raise ValueError(
            "Single-stage JAX backend with optimizer_backend='scipy' is "
            "CPU/reference-only; use optimizer_backend='ondevice', "
            "optimizer_backend='scipy-jax-decomposed', "
            "optimizer_backend='host-jax', "
            "optimizer_backend='scipy-jax-fullgraph', "
            "optimizer_backend='optax-lbfgs', or "
            "optimizer_backend='optimistix-lbfgs'. For the JAX target lane, "
            "select boozer_optimizer_backend='ondevice'. For the host-control "
            "kernelized lane, select optimizer_backend='host-jax' or "
            "boozer_optimizer_backend='host-jax'."
        )
    return resolve_boozer_optimizer_backend(
        optimizer_backend,
        boozer_optimizer_backend,
    )


def resolve_boozer_optimizer_backend(
    optimizer_backend: str,
    boozer_optimizer_backend: str | None,
) -> str:
    if boozer_optimizer_backend is None:
        effective_backend = (
            "scipy"
            if optimizer_backend == JAX_FULL_GRAPH_SCIPY_OUTER_OPTIMIZER_BACKEND
            else (
                "ondevice"
                if optimizer_backend in JAX_ONDEVICE_BOOZER_OUTER_OPTIMIZER_BACKENDS
                else optimizer_backend
            )
        )
    else:
        effective_backend = boozer_optimizer_backend
    if effective_backend not in JAX_BOOZER_INNER_OPTIMIZER_BACKENDS:
        raise ValueError(
            "Single-stage JAX routing requires "
            "boozer_optimizer_backend='ondevice', 'host-jax', or 'scipy'."
        )
    return effective_backend


def resolve_boozer_limited_memory(
    boozer_optimizer_backend: str,
    boozer_limited_memory_requested: bool,
) -> bool:
    """Match the current benchmark/example plumbing for effective limited memory."""
    return bool(
        boozer_optimizer_backend in {"host-jax", "ondevice"}
        and boozer_limited_memory_requested
    )


def resolve_boozer_optimizer_method(
    boozer_optimizer_backend: str,
    *,
    limited_memory: bool = False,
    least_squares_algorithm: str | None = None,
) -> str:
    if boozer_optimizer_backend not in JAX_BOOZER_INNER_OPTIMIZER_BACKENDS:
        raise ValueError(
            "Single-stage JAX routing requires "
            "boozer_optimizer_backend='ondevice', 'host-jax', or 'scipy'."
        )
    if boozer_optimizer_backend == "scipy":
        return "scipy"
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
        return "lm" if boozer_optimizer_backend == "host-jax" else "lm-ondevice"
    if boozer_optimizer_backend == "host-jax":
        return "lbfgs" if limited_memory else "bfgs"
    return "lbfgs-ondevice" if limited_memory else "bfgs-ondevice"


def resolve_boozer_least_squares_algorithm(
    boozer_optimizer_backend: str,
    least_squares_algorithm: str | None = None,
) -> str:
    if boozer_optimizer_backend not in JAX_BOOZER_INNER_OPTIMIZER_BACKENDS:
        raise ValueError(
            "Single-stage JAX routing requires "
            "boozer_optimizer_backend='ondevice', 'host-jax', or 'scipy'."
        )
    if least_squares_algorithm is not None:
        return least_squares_algorithm
    if boozer_optimizer_backend == "host-jax":
        return "lm"
    return "quasi-newton"
