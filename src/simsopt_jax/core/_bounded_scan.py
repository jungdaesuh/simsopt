"""Small control-flow helpers for fixed-capacity JAX scans."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import jax
import jax.numpy as jnp

Carry = TypeVar("Carry")


def bounded_scan_until_done(
    initial: Carry,
    *,
    max_steps: int,
    is_done: Callable[[Carry], jax.Array],
    step: Callable[[Carry, jax.Array], Carry],
) -> Carry:
    """Run a fixed-length scan, skipping ``step`` once ``is_done`` is true."""

    def _scan_body(current: Carry, iteration: jax.Array):
        next_state = jax.lax.cond(
            is_done(current),
            lambda done_state: done_state,
            lambda active_state: step(active_state, iteration),
            current,
        )
        return next_state, None

    final_state, _ = jax.lax.scan(
        _scan_body,
        initial,
        jnp.arange(int(max_steps), dtype=jnp.int32),
    )
    return final_state
