"""Explicit JAXPR closure operands for strict device-resident programs."""

from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp
from jax.extend import core as jax_core
from jax.sharding import NamedSharding, PartitionSpec as P

from simsopt_jax.backend.dtypes import runtime_device_put_tree


def closure_converted_value_and_grad(value_and_grad, example_x):
    """Return a pure program and its explicit closed-over array operands."""

    def coerce_result(x):
        value, gradient = value_and_grad(x)
        value = jnp.asarray(value, dtype=x.dtype)
        gradient = jnp.asarray(gradient, dtype=x.dtype)
        if gradient.shape != x.shape:
            raise ValueError(
                "On-device explicit value-and-gradient objectives must return a "
                f"gradient matching x.shape={x.shape}, got {gradient.shape}."
            )
        return value, gradient

    converted = jax.make_jaxpr(coerce_result)(example_x)
    value_and_grad_jaxpr = converted.jaxpr
    value_and_grad_consts = tuple(converted.consts)

    def value_and_grad_from_jaxpr(x, consts):
        closed_jaxpr = jax_core.ClosedJaxpr(value_and_grad_jaxpr, consts)
        return jax_core.jaxpr_as_fun(closed_jaxpr)(x)

    return value_and_grad_from_jaxpr, value_and_grad_consts


def _const_leaf_placement(reference_placement, leaf):
    if not isinstance(reference_placement, NamedSharding):
        return reference_placement
    if len(reference_placement.spec) <= int(np.ndim(leaf)):
        return reference_placement
    return NamedSharding(reference_placement.mesh, P())


def device_put_closure_consts(value_and_grad_consts, example_x):
    """Place closure operands with shardings compatible with the decision vector."""
    reference_placement = (
        jax.typeof(example_x).sharding
        if isinstance(example_x, jax.core.Tracer)
        else example_x.sharding
    )
    return jax.tree.map(
        lambda leaf: runtime_device_put_tree(
            leaf,
            target=_const_leaf_placement(reference_placement, leaf),
        ),
        value_and_grad_consts,
    )
