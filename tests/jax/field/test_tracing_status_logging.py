"""Negative terminal statuses must be decoded for the route that produced them.

The core's documented rule is ``-1 - i`` for ``stopping_criteria[i]``
(``src/simsopt_jax/core/tracing.py`` module docstring, mirroring
``legacy native extension/tracing.cpp:434``
``res_phi_hits.push_back(join<2, RHS::Size>({t, -1-double(i)}, y))``). The two
Boozer drivers additionally emit ``-2`` when the lane leaves the axis
(``s <= 0``), which is branch-added and has no upstream counterpart, so on a
Boozer route ``-2`` must not be reported as "criterion index 1".
"""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import logging
from typing import NamedTuple

import jax
import jax.numpy as jnp
from simsopt_jax.core.tracing import TRACING_STATUS_BOOZER_AXIS
from simsopt_jax_adapters.field.tracing import _log_batched_jax_trace_statuses


class _BatchedTraceResult(NamedTuple):
    """The three fields ``_batched_trace_status_arrays`` reads, same shapes."""

    status: jax.Array
    t_final: jax.Array
    steps_taken: jax.Array


def _result(statuses: tuple[int, ...]) -> _BatchedTraceResult:
    return _BatchedTraceResult(
        status=jnp.asarray(statuses, dtype=jnp.int32),
        t_final=jnp.zeros((len(statuses),), dtype=jnp.float64),
        steps_taken=jnp.ones((len(statuses),), dtype=jnp.int32),
    )


def _debug_messages(caplog) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.DEBUG
    ]


def test_boozer_axis_status_is_not_reported_as_a_criterion_index(caplog) -> None:
    statuses = (TRACING_STATUS_BOOZER_AXIS, -1, -3)
    with caplog.at_level(logging.DEBUG, logger="simsopt_jax_adapters.field.tracing"):
        lost = _log_batched_jax_trace_statuses(
            _result(statuses),
            first=0,
            total=len(statuses),
            tmax=1.0,
            label="JAX Boozer guiding-centre",
            boozer_axis_status=True,
        )

    messages = _debug_messages(caplog)
    assert lost == len(statuses)
    axis_line = next(message for message in messages if message.startswith("  1/"))
    assert "left the Boozer axis" in axis_line
    # The unqualified decoding is what the adapter used to print for -2.
    assert "stopped by criterion index 1" not in axis_line
    # The genuine criterion stops on the same route are still decoded.
    assert "criterion index 0" in next(
        message for message in messages if message.startswith("  2/")
    )
    assert "criterion index 2" in next(
        message for message in messages if message.startswith("  3/")
    )


def test_cartesian_routes_decode_every_negative_status_as_a_criterion(caplog) -> None:
    """The Cartesian and full-orbit routes never emit the axis status."""
    statuses = (-1, TRACING_STATUS_BOOZER_AXIS)
    with caplog.at_level(logging.DEBUG, logger="simsopt_jax_adapters.field.tracing"):
        _log_batched_jax_trace_statuses(
            _result(statuses),
            first=0,
            total=len(statuses),
            tmax=1.0,
            label="JAX fieldline",
        )

    messages = _debug_messages(caplog)
    assert "criterion index 0" in messages[0]
    assert "stopped by criterion index 1" in messages[1]
    assert "axis" not in messages[1]
