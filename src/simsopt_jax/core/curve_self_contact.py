"""Smooth self-contact penalty of one closed curve.

For a curve sampled at ``N`` nodes of an open, uniform periodic rule
(``t_k = k / N``), with node speed ``v_i = |gamma'_i|``,

    J = 1/2 N^-2 sum_{i,j} chi(e_ij) v_i v_j h(q_ij),

* ``q_ij = |gamma_i - gamma_j|^2`` and ``h(q) = max(d^2 - q, 0)^2 / (4 d^2)``:
  polynomial in the positions, so an exact coincidence ``q = 0`` is an
  ordinary smooth point; ``h'`` is Lipschitz and ``h''`` jumps only at the
  activation ``q = d^2``.  Near activation ``h = (d - r)^2 (1 + O(d - r))``;
  at coincidence ``h = d^2 / 4``.
* ``e_ij = (L/pi)^2 sin^2(pi (s_i - s_j) / L)``, a smooth squared periodic
  arclength separation: ``s_i`` is the periodic-trapezoid node arclength and
  ``L`` the curve length on the same rule.  No ``|.|`` or cyclic minimum.
* ``chi(e) = S((e - w0^2) / (w1^2 - w0^2))`` with the C2 quintic smoothstep
  ``S(u) = u^3 (10 - 15 u + 6 u^2)`` on ``[0, 1]`` (0 below, 1 above): the
  ramp acts on ``e``, the smooth periodic squared separation above, not on
  literal arclength -- pairs with ``e`` below ``w0^2`` are local and never
  penalized, pairs with ``e`` above ``w1^2`` carry full weight.

Domain.  On ``R = {every node speed > 0}`` the penalty is C1 in the node
positions and derivatives with a locally Lipschitz gradient, which reverse
mode returns everywhere on ``R`` (exact coincidences and every ramp state
included).  A curve with a zero-speed node is outside ``R``: the value is NaN
there (explicitly, not a limit), so an optimizer that rejects non-finite trial
values rejects the point; it is never given a conventional derivative.

For a transversal crossing of two straight strands at angle ``theta``, in the
full-weight regime and the continuum limit, ``J = pi d^4 / (12 |sin theta|)``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _smoothstep5(u: jax.Array) -> jax.Array:
    u = jnp.clip(u, 0.0, 1.0)
    return u * u * u * (10.0 - 15.0 * u + 6.0 * u * u)


def _node_arclength(speed: jax.Array) -> tuple[jax.Array, jax.Array]:
    """``(s, L)``: periodic-trapezoid arclength at each node (``s_0 = 0``) and the length."""
    count = speed.shape[0]
    increments = 0.5 * (speed + jnp.roll(speed, -1)) / count
    arclength = jnp.concatenate(
        [jnp.zeros((1,), dtype=speed.dtype), jnp.cumsum(increments)[:-1]]
    )
    return arclength, jnp.sum(increments)


def self_contact_separation_weight(
    speed: jax.Array, ramp_start: float, ramp_end: float
) -> jax.Array:
    """``chi(e_ij)`` for every node pair, from the node speeds (``(N, N)``)."""
    arclength, length = _node_arclength(speed)
    phase = jnp.pi * (arclength[:, None] - arclength[None, :]) / length
    squared = (length / jnp.pi) ** 2 * jnp.sin(phase) ** 2
    return _smoothstep5((squared - ramp_start**2) / (ramp_end**2 - ramp_start**2))


def self_contact_energy(
    squared_distance: jax.Array, minimum_distance: float
) -> jax.Array:
    """``h(q) = max(d^2 - q, 0)^2 / (4 d^2)``."""
    deficit = jnp.maximum(minimum_distance**2 - squared_distance, 0.0)
    return deficit * deficit / (4.0 * minimum_distance**2)


def curve_self_contact_penalty_pure(
    gamma: jax.Array,
    gammadash: jax.Array,
    minimum_distance: float,
    ramp_start: float,
    ramp_end: float,
) -> jax.Array:
    """``J`` of one curve from its ``(N, 3)`` nodes and derivatives; NaN off the domain."""
    speed = jnp.linalg.norm(gammadash, axis=1)
    weight = self_contact_separation_weight(speed, ramp_start, ramp_end)
    squared = jnp.sum((gamma[:, None, :] - gamma[None, :, :]) ** 2, axis=2)
    count = gamma.shape[0]
    penalty = (
        0.5
        * jnp.sum(
            weight
            * speed[:, None]
            * speed[None, :]
            * self_contact_energy(squared, minimum_distance)
        )
        / (count * count)
    )
    return jnp.where(jnp.min(speed) > 0.0, penalty, jnp.nan)


__all__ = [
    "curve_self_contact_penalty_pure",
    "self_contact_energy",
    "self_contact_separation_weight",
]
