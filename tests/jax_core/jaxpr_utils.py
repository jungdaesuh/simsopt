"""JAXPR inspection helpers for structural tests."""

from __future__ import annotations

from collections.abc import Mapping

from jax.extend import core as jax_core


def count_jaxpr_primitives(
    closed_jaxpr: jax_core.ClosedJaxpr, primitive_name: str
) -> int:
    return _count_nested_primitives(closed_jaxpr, primitive_name)


def _count_nested_primitives(param: object, primitive_name: str) -> int:
    if isinstance(param, jax_core.ClosedJaxpr):
        return _count_nested_primitives(param.jaxpr, primitive_name)
    if isinstance(param, jax_core.Jaxpr):
        total = 0
        for eqn in param.eqns:
            total += int(eqn.primitive.name == primitive_name)
            total += _count_nested_primitives(eqn.params, primitive_name)
        return total
    if isinstance(param, Mapping):
        return sum(
            _count_nested_primitives(value, primitive_name) for value in param.values()
        )
    if isinstance(param, (tuple, list)):
        return sum(_count_nested_primitives(value, primitive_name) for value in param)
    return 0
