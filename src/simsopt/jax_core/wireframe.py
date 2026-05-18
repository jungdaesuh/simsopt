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

from .._host_array_contracts import require_nonnegative_int32_indices
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


def _as_points(points: object) -> jax.Array:
    points_jax = _as_jax_float64(points)
    if points_jax.ndim != 2 or points_jax.shape[-1] != 3:
        raise ValueError(
            f"points must have shape (n_points, 3); got {points_jax.shape}."
        )
    return points_jax


def _as_segments(segments: object) -> jax.Array:
    if isinstance(segments, jax.Array) or hasattr(segments, "aval"):
        segments_jax = jnp.asarray(segments)
        if segments_jax.dtype != jnp.int32:
            raise TypeError(
                "device-resident segments must already be int32; pass host "
                "segments for range-checked staging."
            )
        return segments_jax
    return _as_jax_int32(require_nonnegative_int32_indices("segments", segments))


def _gather_segment_nodes(
    nodes: object,
    segments: object,
) -> tuple[jax.Array, jax.Array]:
    nodes_jax = _as_jax_float64(nodes)
    segments_jax = _as_segments(segments)
    node0 = jnp.take(nodes_jax, segments_jax[:, 0], axis=1)
    node1 = jnp.take(nodes_jax, segments_jax[:, 1], axis=1)
    return node0, node1


def _wireframe_segment_B_from_arrays(
    points: jax.Array,
    node0: jax.Array,
    node1: jax.Array,
) -> jax.Array:
    diff0 = points - node0
    diff1 = points - node1
    norm_diff0 = jnp.sqrt(jnp.sum(diff0 * diff0, axis=-1))
    norm_diff1 = jnp.sqrt(jnp.sum(diff1 * diff1, axis=-1))
    diff0_diff1 = norm_diff0 * norm_diff1
    denom = diff0_diff1 * (diff0_diff1 + jnp.sum(diff0 * diff1, axis=-1))
    factor = (norm_diff0 + norm_diff1) / denom
    return _MU0_OVER_4PI * factor[:, None] * jnp.cross(diff0, diff1)


def wireframe_segment_B(
    points: object,
    node0: object,
    node1: object,
) -> jax.Array:
    """Magnetic field from one straight wire segment with unit current."""
    return _wireframe_segment_B_from_arrays(
        _as_points(points),
        _as_jax_float64(node0),
        _as_jax_float64(node1),
    )


def wireframe_segment_dB_by_dX(
    points: object,
    node0: object,
    node1: object,
) -> jax.Array:
    """First spatial derivative from one straight wire segment with unit current.

    Returns
    -------
    dB : jax.Array
        Shape ``(n_points, 3, 3)``. Axis convention:
        ``dB[p, l, j] = ∂_j B_l(x_p)`` (component-first; matches the
        simsoptpp C++ storage order). Axis 1 is the B-field component;
        axis 2 is the spatial derivative direction. See the module
        head docstring for the C++ ``dB_by_dX(p, k, m)`` alignment.
    """
    _, dB = wireframe_segment_B_and_dB_by_dX(points, node0, node1)
    return dB


def wireframe_segment_B_and_dB_by_dX(
    points: object,
    node0: object,
    node1: object,
) -> tuple[jax.Array, jax.Array]:
    """Return ``(B, dB_by_dX)`` for one straight segment with unit current.

    Returns
    -------
    B : jax.Array
        Shape ``(n_points, 3)``.
    dB : jax.Array
        Shape ``(n_points, 3, 3)``. Axis convention:
        ``dB[p, l, j] = ∂_j B_l(x_p)`` (component-first; matches the
        simsoptpp C++ storage order). Axis 1 is the B-field component;
        axis 2 is the spatial derivative direction.
    """
    return _wireframe_segment_B_and_dB_by_dX_from_arrays(
        _as_points(points),
        _as_jax_float64(node0),
        _as_jax_float64(node1),
    )


def _wireframe_segment_B_and_dB_by_dX_from_arrays(
    points_jax: jax.Array,
    node0_jax: jax.Array,
    node1_jax: jax.Array,
) -> tuple[jax.Array, jax.Array]:

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
    points_jax = _as_points(points)
    seg_signs_jax = _as_jax_float64(seg_signs).reshape((-1,))
    node0, node1 = _gather_segment_nodes(nodes, segments)

    def segment_B(node0_by_segment: jax.Array, node1_by_segment: jax.Array):
        return _segment_total_B(
            points_jax, node0_by_segment, node1_by_segment, seg_signs_jax
        )

    return jax.vmap(segment_B, in_axes=(1, 1), out_axes=0)(node0, node1)


def wireframe_segment_dB_by_dX_contributions(
    points: object,
    nodes: object,
    segments: object,
    seg_signs: object,
) -> jax.Array:
    """Return unit-current ``dB_by_dX`` per segment.

    Returns
    -------
    dB : jax.Array
        Shape ``(n_segments, n_points, 3, 3)``. Axis convention on the
        trailing 3x3 block: ``dB[i, p, l, j] = ∂_j B_l(x_p)`` for
        segment ``i`` (component-first; matches the simsoptpp C++
        storage order). Axis 2 is the B-field component; axis 3 is
        the spatial derivative direction.
    """
    points_jax = _as_points(points)
    seg_signs_jax = _as_jax_float64(seg_signs).reshape((-1,))
    node0, node1 = _gather_segment_nodes(nodes, segments)

    def segment_dB(node0_by_segment: jax.Array, node1_by_segment: jax.Array):
        return _segment_total_dB(
            points_jax, node0_by_segment, node1_by_segment, seg_signs_jax
        )

    return jax.vmap(segment_dB, in_axes=(1, 1), out_axes=0)(node0, node1)


def _nodes_by_segment(
    nodes: jax.Array, segments: jax.Array
) -> tuple[jax.Array, jax.Array]:
    node0, node1 = _gather_segment_nodes(nodes, segments)
    return jnp.moveaxis(node0, 1, 0), jnp.moveaxis(node1, 1, 0)


def _zero_dB_like(points: jax.Array) -> jax.Array:
    return jnp.zeros(points.shape + (3,), dtype=points.dtype)


def _segment_total_B(
    points: jax.Array,
    node0_by_segment: jax.Array,
    node1_by_segment: jax.Array,
    seg_signs: jax.Array,
) -> jax.Array:
    def add_half_period(
        acc: jax.Array, half_period: tuple[jax.Array, jax.Array, jax.Array]
    ):
        node0, node1, seg_sign = half_period
        return acc + seg_sign * _wireframe_segment_B_from_arrays(
            points, node0, node1
        ), None

    B, _ = jax.lax.scan(
        add_half_period,
        jnp.zeros_like(points),
        (node0_by_segment, node1_by_segment, seg_signs),
    )
    return B


def _segment_total_dB(
    points: jax.Array,
    node0_by_segment: jax.Array,
    node1_by_segment: jax.Array,
    seg_signs: jax.Array,
) -> jax.Array:
    def add_half_period(
        acc: jax.Array, half_period: tuple[jax.Array, jax.Array, jax.Array]
    ):
        node0, node1, seg_sign = half_period
        _B, dB = _wireframe_segment_B_and_dB_by_dX_from_arrays(points, node0, node1)
        return acc + seg_sign * dB, None

    dB, _ = jax.lax.scan(
        add_half_period,
        _zero_dB_like(points),
        (node0_by_segment, node1_by_segment, seg_signs),
    )
    return dB


def _segment_total_B_and_dB(
    points: jax.Array,
    node0_by_segment: jax.Array,
    node1_by_segment: jax.Array,
    seg_signs: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    def add_half_period(
        acc: tuple[jax.Array, jax.Array],
        half_period: tuple[jax.Array, jax.Array, jax.Array],
    ):
        node0, node1, seg_sign = half_period
        B_acc, dB_acc = acc
        B, dB = _wireframe_segment_B_and_dB_by_dX_from_arrays(points, node0, node1)
        return (B_acc + seg_sign * B, dB_acc + seg_sign * dB), None

    return jax.lax.scan(
        add_half_period,
        (
            jnp.zeros_like(points),
            _zero_dB_like(points),
        ),
        (node0_by_segment, node1_by_segment, seg_signs),
    )[0]


@jax.jit
def _wireframe_B_jit(
    points: jax.Array,
    nodes: jax.Array,
    segments: jax.Array,
    seg_signs: jax.Array,
    currents: jax.Array,
) -> jax.Array:
    node0, node1 = _nodes_by_segment(nodes, segments)

    def add_segment(acc: jax.Array, segment: tuple[jax.Array, jax.Array, jax.Array]):
        node0_by_segment, node1_by_segment, current = segment
        B = _segment_total_B(points, node0_by_segment, node1_by_segment, seg_signs)
        return acc + current * B, None

    B, _ = jax.lax.scan(add_segment, jnp.zeros_like(points), (node0, node1, currents))
    return B


@jax.jit
def _wireframe_dB_jit(
    points: jax.Array,
    nodes: jax.Array,
    segments: jax.Array,
    seg_signs: jax.Array,
    currents: jax.Array,
) -> jax.Array:
    node0, node1 = _nodes_by_segment(nodes, segments)

    def add_segment(acc: jax.Array, segment: tuple[jax.Array, jax.Array, jax.Array]):
        node0_by_segment, node1_by_segment, current = segment
        dB = _segment_total_dB(points, node0_by_segment, node1_by_segment, seg_signs)
        return acc + current * dB, None

    dB, _ = jax.lax.scan(
        add_segment,
        _zero_dB_like(points),
        (node0, node1, currents),
    )
    return dB


@jax.jit
def _wireframe_B_and_dB_jit(
    points: jax.Array,
    nodes: jax.Array,
    segments: jax.Array,
    seg_signs: jax.Array,
    currents: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    node0, node1 = _nodes_by_segment(nodes, segments)

    def add_segment(
        acc: tuple[jax.Array, jax.Array],
        segment: tuple[jax.Array, jax.Array, jax.Array],
    ):
        node0_by_segment, node1_by_segment, current = segment
        B_acc, dB_acc = acc
        B, dB = _segment_total_B_and_dB(
            points, node0_by_segment, node1_by_segment, seg_signs
        )
        return (B_acc + current * B, dB_acc + current * dB), None

    return jax.lax.scan(
        add_segment,
        (
            jnp.zeros_like(points),
            _zero_dB_like(points),
        ),
        (node0, node1, currents),
    )[0]


def _coerce_inputs(
    points: object,
    nodes: object,
    segments: object,
    seg_signs: object,
    currents: object,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    return (
        _as_points(points),
        _as_jax_float64(nodes),
        _as_segments(segments),
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
    """Return the total first spatial derivative of the wireframe field.

    Returns
    -------
    dB : jax.Array
        Shape ``(n_points, 3, 3)``. Axis convention:
        ``dB[p, l, j] = ∂_j B_l(x_p)`` (component-first; matches the
        simsoptpp C++ storage order). Axis 1 is the B-field component;
        axis 2 is the spatial derivative direction.
    """
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
    """Return ``(B, dB_by_dX)`` for the total wireframe field.

    Returns
    -------
    B : jax.Array
        Shape ``(n_points, 3)``.
    dB : jax.Array
        Shape ``(n_points, 3, 3)``. Axis convention:
        ``dB[p, l, j] = ∂_j B_l(x_p)`` (component-first; matches the
        simsoptpp C++ storage order). Axis 1 is the B-field component;
        axis 2 is the spatial derivative direction.
    """
    return _wireframe_B_and_dB_jit(
        *_coerce_inputs(points, nodes, segments, seg_signs, currents)
    )
