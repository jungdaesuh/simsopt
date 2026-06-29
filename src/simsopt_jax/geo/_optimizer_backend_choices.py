"""SSOT for the outer-loop optimizer-backend choice vocabulary.

Kept dependency-light (no JAX import) so jax-cold benchmark helpers can
share the same set/order/message as ``simsopt_jax.geo.optimizers.optimizer`` without
pulling JAX into the import graph.
"""

from __future__ import annotations


CONCRETE_OPTIMIZER_BACKENDS = frozenset({"scipy", "ondevice"})
TARGET_SCIPY_CONTROL_OPTIMIZER_BACKENDS = frozenset(
    {"scipy-jax", "scipy-jax-decomposed", "scipy-jax-fullgraph"}
)
SINGLE_STAGE_DEPRECATED_OPTIMIZER_BACKENDS = frozenset({"scipy-jax"})
_SINGLE_STAGE_OPTIMIZER_BACKEND_REPLACEMENTS = {
    "scipy-jax": "scipy-jax-decomposed",
}
HOST_JAX_OUTER_OPTIMIZER_BACKEND = "host-jax"
TARGET_PUBLIC_LBFGS_OPTIMIZER_BACKENDS = frozenset({"optax-lbfgs", "optimistix-lbfgs"})
TARGET_OUTER_OPTIMIZER_BACKENDS = (
    frozenset({"ondevice"})
    | TARGET_SCIPY_CONTROL_OPTIMIZER_BACKENDS
    | TARGET_PUBLIC_LBFGS_OPTIMIZER_BACKENDS
)
VALID_OPTIMIZER_BACKENDS = frozenset({"auto"}) | CONCRETE_OPTIMIZER_BACKENDS
VALID_OUTER_OPTIMIZER_BACKENDS = (
    TARGET_SCIPY_CONTROL_OPTIMIZER_BACKENDS
    | TARGET_PUBLIC_LBFGS_OPTIMIZER_BACKENDS
    | CONCRETE_OPTIMIZER_BACKENDS
    | frozenset({HOST_JAX_OUTER_OPTIMIZER_BACKEND})
)
_OUTER_OPTIMIZER_BACKEND_DISPLAY_ORDER = (
    "scipy",
    "ondevice",
    "scipy-jax",
    "scipy-jax-decomposed",
    HOST_JAX_OUTER_OPTIMIZER_BACKEND,
    "scipy-jax-fullgraph",
    "optax-lbfgs",
    "optimistix-lbfgs",
)
_RESOLVABLE_OPTIMIZER_BACKEND_DISPLAY_ORDER = (
    "auto",
    *_OUTER_OPTIMIZER_BACKEND_DISPLAY_ORDER,
)
assert (
    frozenset(_OUTER_OPTIMIZER_BACKEND_DISPLAY_ORDER) == VALID_OUTER_OPTIMIZER_BACKENDS
)
assert frozenset(_RESOLVABLE_OPTIMIZER_BACKEND_DISPLAY_ORDER) == (
    VALID_OUTER_OPTIMIZER_BACKENDS | {"auto"}
)


def render_invalid_optimizer_backend_message(scope: str) -> str:
    if scope == "outer":
        names = _OUTER_OPTIMIZER_BACKEND_DISPLAY_ORDER
    elif scope == "resolvable":
        names = _RESOLVABLE_OPTIMIZER_BACKEND_DISPLAY_ORDER
    else:
        raise ValueError(f"scope must be 'outer' or 'resolvable', got {scope!r}.")
    return f"optimizer_backend must be one of: {', '.join(names)}."


OUTER_OPTIMIZER_BACKEND_MESSAGE = render_invalid_optimizer_backend_message("outer")
RESOLVABLE_OPTIMIZER_BACKEND_MESSAGE = render_invalid_optimizer_backend_message(
    "resolvable"
)


def single_stage_optimizer_backend_replacement(optimizer_backend: str) -> str | None:
    return _SINGLE_STAGE_OPTIMIZER_BACKEND_REPLACEMENTS.get(optimizer_backend)
