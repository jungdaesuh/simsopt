"""Pure JAX kernels for wireframe magnetic fields (item 29).

This module ports the field-kernel portion of the C++ wireframe implementation:

* ``simsoptpp/wireframe_field_impl.h``
* ``simsoptpp/magneticfield_wireframe.cpp``

The upstream optimiser (``wireframe_optimization.cpp``, GSCO) is item 31 and
lives in ``src/simsopt/solve/wireframe_optimization_jax.py``; this module owns
only the item-29 field-kernel arithmetic.

Layout
------

* ``points`` has shape ``(N, 3)`` -- cartesian field-evaluation points in
  metres.
* ``nodes`` has shape ``(n_halfprds, n_nodes, 3)`` -- one node-position
  table per half-period (the upstream ``WireframeField`` keeps a copy of
  the node coordinates for every nfp-symmetric half-period).
* ``segments`` has shape ``(n_segments, 2)`` -- integer indices into the
  ``n_nodes`` axis of ``nodes``. Positive current flows from
  ``segments[i, 0]`` to ``segments[i, 1]``.
* ``seg_signs`` has shape ``(n_halfprds,)`` -- per-half-period sign that
  flips the current direction in the reflected half-periods.
* ``currents`` has shape ``(n_segments,)`` -- the scalar current carried
  by each segment.
* ``B`` has shape ``(N, 3)`` and ``dB[p, k, m] = d_m B_k(x_p)`` with axis
  1 = ``B`` component and axis 2 = derivative direction. This matches
  the *literal* storage in
  ``wireframe_field_impl.h``::

      dB_by_dX(p, k, 0) = fak * dB_dX_i[k].x;   // d_x B_k
      dB_by_dX(p, k, 1) = fak * dB_dX_i[k].y;   // d_y B_k
      dB_by_dX(p, k, 2) = fak * dB_dX_i[k].z;   // d_z B_k

  and what ``simsoptpp.WireframeField.dB_by_dX()`` returns. The same
  layout is used by ``simsoptpp.BiotSavart``; the abstract simsopt-jax
  convention quoted in ``CLAUDE.md`` (``dB[p, j, l] = d_j B_l``) names
  the same array with swapped axis labels, but the storage is
  component-first regardless.

Closed form
-----------

For a single segment from ``a`` to ``b`` carrying unit current, the
field at point ``r`` is

.. math::

   B(r) = \\frac{\\mu_0}{4\\pi}
          \\frac{|r_1| + |r_2|}{|r_1||r_2|(|r_1||r_2| + r_1\\cdot r_2)}
          \\,(r_1 \\times r_2)

with ``r_1 = r - a`` and ``r_2 = r - b``. This is mathematically
equivalent to the ``(\\cos\\theta_1 - \\cos\\theta_2)`` form, but
matches the upstream arithmetic literally so that ``direct_kernel``
parity is bit-identical.

The closed-form Jacobian is also ported literally from
``wireframe_field_impl.h``; no autodiff layer is used so that the
floating-point trace matches the C++ oracle.

Singular regimes
----------------

The closed form diverges in two regimes:

* ``r`` collinear with the segment, *between* ``a`` and ``b``: the wire
  itself is singular (``denom -> 0``). The upstream C++ does not guard
  this; it returns ``inf`` / ``nan``. The JAX kernel matches.
* ``r = a`` or ``r = b``: ``|r_1| = 0`` or ``|r_2| = 0``. Same upstream
  behaviour, same JAX behaviour.

No defensive floors are inserted; the kernel is a faithful port of the
upstream arithmetic.

Second derivatives are not provided because the C++ ``WireframeField``
path raises ``logic_error`` for ``d2B_by_dXdX`` (see
``magneticfield_wireframe.cpp`` line ~107).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ._math_utils import (
    as_jax_float64 as _as_jax_float64,
    as_jax_int32 as _as_jax_int32,
)

__all__ = [
    "wireframe_B",
    "wireframe_B_and_dB_by_dX",
    "wireframe_dB_by_dX",
    "wireframe_segment_B",
    "wireframe_segment_B_and_dB_by_dX",
    "wireframe_segment_B_contributions",
    "wireframe_segment_dB_by_dX",
    "wireframe_segment_dB_by_dX_contributions",
]

_MU0_OVER_4PI = 1e-7


def _component(array: jax.Array, index: int) -> jax.Array:
    return array[..., index]


def _gather_segment_nodes(
    nodes: object,
    segments: object,
) -> tuple[jax.Array, jax.Array]:
    nodes_jax = _as_jax_float64(nodes)
    segments_jax = _as_jax_int32(segments)
    node0 = jnp.take(nodes_jax, segments_jax[:, 0], axis=1)
    node1 = jnp.take(nodes_jax, segments_jax[:, 1], axis=1)
    return node0, node1


def wireframe_segment_B(
    points: object,
    node0: object,
    node1: object,
) -> jax.Array:
    """Magnetic field from one straight wire segment with unit current."""
    B, _ = wireframe_segment_B_and_dB_by_dX(points, node0, node1)
    return B


def wireframe_segment_dB_by_dX(
    points: object,
    node0: object,
    node1: object,
) -> jax.Array:
    """First spatial derivative from one straight wire segment with unit current."""
    _, dB = wireframe_segment_B_and_dB_by_dX(points, node0, node1)
    return dB


def wireframe_segment_B_and_dB_by_dX(
    points: object,
    node0: object,
    node1: object,
) -> tuple[jax.Array, jax.Array]:
    """Return ``(B, dB_by_dX)`` for one straight segment with unit current."""
    points_jax = _as_jax_float64(points)
    node0_jax = _as_jax_float64(node0)
    node1_jax = _as_jax_float64(node1)

    diff0 = points_jax - node0_jax
    diff1 = points_jax - node1_jax
    norm_diff0_sq = jnp.sum(diff0 * diff0, axis=-1)
    norm_diff1_sq = jnp.sum(diff1 * diff1, axis=-1)
    norm_diff0 = jnp.sqrt(norm_diff0_sq)
    norm_diff1 = jnp.sqrt(norm_diff1_sq)
    diff0_diff1 = norm_diff0 * norm_diff1
    denom = diff0_diff1 * (diff0_diff1 + jnp.sum(diff0 * diff1, axis=-1))
    factor = (norm_diff0 + norm_diff1) / denom
    diff0_cross_diff1 = jnp.cross(diff0, diff1)
    B = _MU0_OVER_4PI * factor[:, None] * diff0_cross_diff1

    p0 = diff0 * norm_diff1[:, None]
    p1 = diff1 * norm_diff0[:, None]
    factorsq = factor * factor
    grad_factor = (p0 + p1) * (-factorsq[:, None]) - (
        p0 / norm_diff0_sq[:, None] + p1 / norm_diff1_sq[:, None]
    ) / denom[:, None]

    d0x = _component(diff0, 0)
    d0y = _component(diff0, 1)
    d0z = _component(diff0, 2)
    d1x = _component(diff1, 0)
    d1y = _component(diff1, 1)
    d1z = _component(diff1, 2)
    cx = _component(diff0_cross_diff1, 0)
    cy = _component(diff0_cross_diff1, 1)
    cz = _component(diff0_cross_diff1, 2)
    gfx = _component(grad_factor, 0)
    gfy = _component(grad_factor, 1)
    gfz = _component(grad_factor, 2)

    dBdx = jnp.stack(
        (
            gfx * cx,
            gfy * cx + factor * (d1z - d0z),
            gfz * cx + factor * (d0y - d1y),
        ),
        axis=-1,
    )
    dBdy = jnp.stack(
        (
            gfx * cy + factor * (-d1z + d0z),
            gfy * cy,
            gfz * cy + factor * (-d0x + d1x),
        ),
        axis=-1,
    )
    dBdz = jnp.stack(
        (
            gfx * cz + factor * (d1y - d0y),
            gfy * cz + factor * (d0x - d1x),
            gfz * cz,
        ),
        axis=-1,
    )
    dB = _MU0_OVER_4PI * jnp.stack((dBdx, dBdy, dBdz), axis=-2)
    return B, dB


def wireframe_segment_B_contributions(
    points: object,
    nodes: object,
    segments: object,
    seg_signs: object,
) -> jax.Array:
    """Return unit-current ``B`` contributions as ``(n_segments, n_points, 3)``."""
    points_jax = _as_jax_float64(points)
    seg_signs_jax = _as_jax_float64(seg_signs).reshape((-1,))
    node0, node1 = _gather_segment_nodes(nodes, segments)

    def half_period_B(
        node0_by_half: jax.Array,
        node1_by_half: jax.Array,
        seg_sign: jax.Array,
    ) -> jax.Array:
        return seg_sign * wireframe_segment_B(
            points_jax,
            node0_by_half,
            node1_by_half,
        )

    def segment_B(node0_by_segment: jax.Array, node1_by_segment: jax.Array):
        return jnp.sum(
            jax.vmap(half_period_B)(
                node0_by_segment,
                node1_by_segment,
                seg_signs_jax,
            ),
            axis=0,
        )

    return jax.vmap(segment_B, in_axes=(1, 1), out_axes=0)(node0, node1)


def wireframe_segment_dB_by_dX_contributions(
    points: object,
    nodes: object,
    segments: object,
    seg_signs: object,
) -> jax.Array:
    """Return unit-current ``dB_by_dX`` as ``(n_segments, n_points, 3, 3)``."""
    points_jax = _as_jax_float64(points)
    seg_signs_jax = _as_jax_float64(seg_signs).reshape((-1,))
    node0, node1 = _gather_segment_nodes(nodes, segments)

    def half_period_dB(
        node0_by_half: jax.Array,
        node1_by_half: jax.Array,
        seg_sign: jax.Array,
    ) -> jax.Array:
        return seg_sign * wireframe_segment_dB_by_dX(
            points_jax,
            node0_by_half,
            node1_by_half,
        )

    def segment_dB(node0_by_segment: jax.Array, node1_by_segment: jax.Array):
        return jnp.sum(
            jax.vmap(half_period_dB)(
                node0_by_segment,
                node1_by_segment,
                seg_signs_jax,
            ),
            axis=0,
        )

    return jax.vmap(segment_dB, in_axes=(1, 1), out_axes=0)(node0, node1)


def _wireframe_segment_B_and_dB_by_dX_contributions(
    points: object,
    nodes: object,
    segments: object,
    seg_signs: object,
) -> tuple[jax.Array, jax.Array]:
    points_jax = _as_jax_float64(points)
    seg_signs_jax = _as_jax_float64(seg_signs).reshape((-1,))
    node0, node1 = _gather_segment_nodes(nodes, segments)

    def half_period_B_and_dB(
        node0_by_half: jax.Array,
        node1_by_half: jax.Array,
        seg_sign: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        B, dB = wireframe_segment_B_and_dB_by_dX(
            points_jax,
            node0_by_half,
            node1_by_half,
        )
        return seg_sign * B, seg_sign * dB

    def segment_B_and_dB(
        node0_by_segment: jax.Array,
        node1_by_segment: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        B_by_half, dB_by_half = jax.vmap(half_period_B_and_dB)(
            node0_by_segment,
            node1_by_segment,
            seg_signs_jax,
        )
        return jnp.sum(B_by_half, axis=0), jnp.sum(dB_by_half, axis=0)

    return jax.vmap(segment_B_and_dB, in_axes=(1, 1), out_axes=(0, 0))(
        node0,
        node1,
    )


@jax.jit
def _wireframe_B_jit(
    points: jax.Array,
    nodes: jax.Array,
    segments: jax.Array,
    seg_signs: jax.Array,
    currents: jax.Array,
) -> jax.Array:
    segment_B = wireframe_segment_B_contributions(points, nodes, segments, seg_signs)
    return jnp.sum(currents[:, None, None] * segment_B, axis=0)


@jax.jit
def _wireframe_dB_jit(
    points: jax.Array,
    nodes: jax.Array,
    segments: jax.Array,
    seg_signs: jax.Array,
    currents: jax.Array,
) -> jax.Array:
    segment_dB = wireframe_segment_dB_by_dX_contributions(
        points, nodes, segments, seg_signs
    )
    return jnp.sum(currents[:, None, None, None] * segment_dB, axis=0)


@jax.jit
def _wireframe_B_and_dB_jit(
    points: jax.Array,
    nodes: jax.Array,
    segments: jax.Array,
    seg_signs: jax.Array,
    currents: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    segment_B, segment_dB = _wireframe_segment_B_and_dB_by_dX_contributions(
        points, nodes, segments, seg_signs
    )
    B = jnp.sum(currents[:, None, None] * segment_B, axis=0)
    dB = jnp.sum(currents[:, None, None, None] * segment_dB, axis=0)
    return B, dB


def _coerce_inputs(
    points: object,
    nodes: object,
    segments: object,
    seg_signs: object,
    currents: object,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    return (
        _as_jax_float64(points),
        _as_jax_float64(nodes),
        _as_jax_int32(segments),
        _as_jax_float64(seg_signs).reshape((-1,)),
        _as_jax_float64(currents).reshape((-1,)),
    )


def wireframe_B(
    points: object,
    nodes: object,
    segments: object,
    seg_signs: object,
    currents: object,
) -> jax.Array:
    """Return the total wireframe magnetic field."""
    return _wireframe_B_jit(
        *_coerce_inputs(points, nodes, segments, seg_signs, currents)
    )


def wireframe_dB_by_dX(
    points: object,
    nodes: object,
    segments: object,
    seg_signs: object,
    currents: object,
) -> jax.Array:
    """Return the total first spatial derivative of the wireframe field."""
    return _wireframe_dB_jit(
        *_coerce_inputs(points, nodes, segments, seg_signs, currents)
    )


def wireframe_B_and_dB_by_dX(
    points: object,
    nodes: object,
    segments: object,
    seg_signs: object,
    currents: object,
) -> tuple[jax.Array, jax.Array]:
    """Return ``(B, dB_by_dX)`` for the total wireframe field."""
    return _wireframe_B_and_dB_jit(
        *_coerce_inputs(points, nodes, segments, seg_signs, currents)
    )
