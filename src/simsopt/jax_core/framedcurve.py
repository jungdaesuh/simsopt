"""Pure JAX kernels for framed-curve reference frames.

Wave R4 item 18: SSOT JAX port of the math kernels that build the Frenet
and coil-centroid orthonormal frames along a closed curve, optionally
rotated by an angle ``alpha`` along the curve parameter.

The upstream Python implementation in
``simsopt.geo.framedcurve.rotated_centroid_frame`` and
``rotated_frenet_frame`` is already a JAX-native gather-style kernel.
This module re-expresses the same arithmetic line-by-line as a pure
function package without the surrounding ``Optimizable`` graph, so the
kernels are reusable from immutable specs, compose cleanly under
``jax.jit`` / ``jax.vmap`` / ``jax.lax``, and can be parity-checked
directly against the upstream JAX evaluation at the
``direct_kernel`` tolerance lane.

Conventions
-----------

* ``gamma``, ``gammadash``, ``gammadashdash`` are arrays of shape
  ``(N, 3)`` where ``N`` is the quadrature-point count.
* ``alpha`` is a length-``N`` array of rotation angles in radians.
* Outputs ``t, n, b`` are each shape ``(N, 3)``. They form a right-handed
  orthonormal triple at every quadrature point.
* The unrotated ``frenet_frame`` / ``centroid_frame`` helpers fall out of
  the rotated variants at ``alpha = 0`` and are exposed as their own
  pure functions for callers that do not need rotation support.

The kernels operate on materialised JAX arrays (no SciPy / SciPy-like
fallback). They are JIT-friendly: every operation is expressed in terms
of ``jnp`` reductions, broadcasts, and ``jnp.cross``; no Python-level
control flow depends on dynamic input.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ._math_utils import as_jax_float64 as _as_jax_float64


__all__ = [
    "centroid_frame",
    "centroid_frame_dash",
    "frenet_frame",
    "frenet_frame_dash",
    "rotated_centroid_frame",
    "rotated_centroid_frame_dash",
    "rotated_frenet_frame",
    "rotated_frenet_frame_dash",
    "rotation_alpha",
    "rotation_alphadash",
    "rotation_dcoeff",
    "rotationdash_dcoeff",
]


def _row_norm(matrix: jax.Array) -> jax.Array:
    """Return ``sqrt(sum(matrix ** 2, axis=1))`` as a column vector ``(N, 1)``."""
    return jnp.linalg.norm(matrix, axis=1)[:, None]


def _row_dot(left: jax.Array, right: jax.Array) -> jax.Array:
    """Return ``sum(left * right, axis=1)`` as a column vector ``(N, 1)``."""
    return jnp.sum(left * right, axis=1)[:, None]


def _rotate_frame(
    n: jax.Array,
    b: jax.Array,
    alpha: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Apply a per-point rotation ``alpha`` in the ``(n, b)`` plane.

    Matches the upstream ``rotated_centroid_frame`` / ``rotated_frenet_frame``
    convention bit-for-bit: ``nn = cos(alpha) n - sin(alpha) b`` and
    ``bb = sin(alpha) n + cos(alpha) b``.
    """
    cos_alpha = jnp.cos(alpha)[:, None]
    sin_alpha = jnp.sin(alpha)[:, None]
    nn = cos_alpha * n - sin_alpha * b
    bb = sin_alpha * n + cos_alpha * b
    return nn, bb


@jax.jit
def centroid_frame(
    gamma: jax.Array,
    gammadash: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return the unrotated coil-centroid frame ``(t, n, b)``.

    Implements the Singh et al. (2020) coil-centroid construction:

    * ``t = gammadash / |gammadash|``
    * ``R = mean(gamma, axis=0)`` (centroid)
    * ``delta = gamma - R``
    * ``n = (delta - (delta . t) t) / |.|``
    * ``b = t x n``

    Equivalent to ``rotated_centroid_frame(gamma, gammadash, alpha=0)``.
    """
    gamma_jax = _as_jax_float64(gamma)
    gammadash_jax = _as_jax_float64(gammadash)
    arc_length = _row_norm(gammadash_jax)
    t = gammadash_jax / arc_length
    centroid = jnp.mean(gamma_jax, axis=0)
    delta = gamma_jax - centroid[None, :]
    projection = _row_dot(delta, t) * t
    n_unnormed = delta - projection
    n = n_unnormed / _row_norm(n_unnormed)
    b = jnp.cross(t, n, axis=1)
    return t, n, b


@jax.jit
def rotated_centroid_frame(
    gamma: jax.Array,
    gammadash: jax.Array,
    alpha: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return the coil-centroid frame ``(t, n, b)`` rotated by ``alpha``.

    Matches ``simsopt.geo.framedcurve.rotated_centroid_frame`` arithmetic
    line-for-line; suitable for direct-parity validation at the
    ``direct_kernel`` ladder lane.
    """
    t, n, b = centroid_frame(gamma, gammadash)
    alpha_jax = _as_jax_float64(alpha)
    nn, bb = _rotate_frame(n, b, alpha_jax)
    return t, nn, bb


@jax.jit
def frenet_frame(
    gamma: jax.Array,
    gammadash: jax.Array,
    gammadashdash: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return the unrotated Frenet frame ``(t, n, b)``.

    Implements

    * ``t = gammadash / |gammadash|``
    * ``tdash = (gammadashdash - ((gammadash . gammadashdash) / |gammadash|^2) gammadash) / |gammadash|``
    * ``n = tdash / |tdash|``
    * ``b = t x n``

    The ``gamma`` argument is accepted for API symmetry with the rotated
    variant and the centroid frame; it is unused because the Frenet frame
    depends only on ``gammadash`` and ``gammadashdash``.

    Equivalent to ``rotated_frenet_frame(gamma, gammadash, gammadashdash, alpha=0)``.
    """
    del gamma  # Retained for API symmetry; Frenet frame is curvature-only.
    gammadash_jax = _as_jax_float64(gammadash)
    gammadashdash_jax = _as_jax_float64(gammadashdash)
    arc_length = _row_norm(gammadash_jax)
    inv_arc_length_sq = 1.0 / (arc_length * arc_length)
    t = gammadash_jax / arc_length
    inner_gd_gdd = _row_dot(gammadash_jax, gammadashdash_jax)
    tdash = inv_arc_length_sq * (
        arc_length * gammadashdash_jax - (inner_gd_gdd / arc_length) * gammadash_jax
    )
    n = tdash / _row_norm(tdash)
    b = jnp.cross(t, n, axis=1)
    return t, n, b


@jax.jit
def rotated_frenet_frame(
    gamma: jax.Array,
    gammadash: jax.Array,
    gammadashdash: jax.Array,
    alpha: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return the Frenet frame ``(t, n, b)`` rotated by ``alpha``.

    Matches ``simsopt.geo.framedcurve.rotated_frenet_frame`` arithmetic
    line-for-line; suitable for direct-parity validation at the
    ``direct_kernel`` ladder lane.
    """
    t, n, b = frenet_frame(gamma, gammadash, gammadashdash)
    alpha_jax = _as_jax_float64(alpha)
    nn, bb = _rotate_frame(n, b, alpha_jax)
    return t, nn, bb


@jax.jit
def rotated_centroid_frame_dash(
    gamma: jax.Array,
    gammadash: jax.Array,
    gammadashdash: jax.Array,
    alpha: jax.Array,
    alphadash: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return the parameter derivative of ``rotated_centroid_frame``.

    The derivative is taken with respect to the curve parameter ``phi``.
    It is built as the JVP of ``rotated_centroid_frame`` against the
    tangent stack ``(gammadash, gammadashdash, alphadash)`` so it matches
    the upstream lambda definition line-for-line.
    """

    def _frame(gamma_in, gammadash_in, alpha_in):
        return rotated_centroid_frame(gamma_in, gammadash_in, alpha_in)

    primals = (
        _as_jax_float64(gamma),
        _as_jax_float64(gammadash),
        _as_jax_float64(alpha),
    )
    tangents = (
        _as_jax_float64(gammadash),
        _as_jax_float64(gammadashdash),
        _as_jax_float64(alphadash),
    )
    _, frame_dash = jax.jvp(_frame, primals, tangents)
    return frame_dash


@jax.jit
def centroid_frame_dash(
    gamma: jax.Array,
    gammadash: jax.Array,
    gammadashdash: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return the parameter derivative of the unrotated centroid frame.

    Equivalent to
    ``rotated_centroid_frame_dash(gamma, gammadash, gammadashdash, 0, 0)``.
    """
    zero_alpha = jnp.zeros(_as_jax_float64(gamma).shape[0], dtype=jnp.float64)
    return rotated_centroid_frame_dash(
        gamma,
        gammadash,
        gammadashdash,
        zero_alpha,
        zero_alpha,
    )


@jax.jit
def rotated_frenet_frame_dash(
    gamma: jax.Array,
    gammadash: jax.Array,
    gammadashdash: jax.Array,
    gammadashdashdash: jax.Array,
    alpha: jax.Array,
    alphadash: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return the parameter derivative of ``rotated_frenet_frame``.

    Built as the JVP of ``rotated_frenet_frame`` against the tangent
    stack ``(gammadash, gammadashdash, gammadashdashdash, alphadash)``.
    """

    def _frame(gamma_in, gammadash_in, gammadashdash_in, alpha_in):
        return rotated_frenet_frame(gamma_in, gammadash_in, gammadashdash_in, alpha_in)

    primals = (
        _as_jax_float64(gamma),
        _as_jax_float64(gammadash),
        _as_jax_float64(gammadashdash),
        _as_jax_float64(alpha),
    )
    tangents = (
        _as_jax_float64(gammadash),
        _as_jax_float64(gammadashdash),
        _as_jax_float64(gammadashdashdash),
        _as_jax_float64(alphadash),
    )
    _, frame_dash = jax.jvp(_frame, primals, tangents)
    return frame_dash


@jax.jit
def frenet_frame_dash(
    gamma: jax.Array,
    gammadash: jax.Array,
    gammadashdash: jax.Array,
    gammadashdashdash: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return the parameter derivative of the unrotated Frenet frame.

    Equivalent to
    ``rotated_frenet_frame_dash(gamma, gammadash, gammadashdash,
    gammadashdashdash, 0, 0)``.
    """
    zero_alpha = jnp.zeros(_as_jax_float64(gamma).shape[0], dtype=jnp.float64)
    return rotated_frenet_frame_dash(
        gamma,
        gammadash,
        gammadashdash,
        gammadashdashdash,
        zero_alpha,
        zero_alpha,
    )


def _rotation_alpha_impl(dofs: jax.Array, points: jax.Array, order: int) -> jax.Array:
    """Return ``alpha(points)`` for a Fourier rotation Optimizable."""
    dofs_jax = _as_jax_float64(dofs)
    points_jax = _as_jax_float64(points)
    rotation = jnp.broadcast_to(dofs_jax[0], points_jax.shape)
    two_pi = _as_jax_float64(2.0 * np.pi)
    for mode in range(1, int(order) + 1):
        angle = two_pi * mode * points_jax
        rotation = rotation + dofs_jax[2 * mode - 1] * jnp.sin(angle)
        rotation = rotation + dofs_jax[2 * mode] * jnp.cos(angle)
    return rotation


def _rotation_alphadash_impl(
    dofs: jax.Array, points: jax.Array, order: int
) -> jax.Array:
    """Return ``d alpha / d phi`` evaluated at ``points``."""
    dofs_jax = _as_jax_float64(dofs)
    points_jax = _as_jax_float64(points)
    rotation = jnp.zeros_like(points_jax)
    two_pi = _as_jax_float64(2.0 * np.pi)
    for mode in range(1, int(order) + 1):
        scale = two_pi * mode
        angle = scale * points_jax
        rotation = rotation + dofs_jax[2 * mode - 1] * scale * jnp.cos(angle)
        rotation = rotation - dofs_jax[2 * mode] * scale * jnp.sin(angle)
    return rotation


rotation_alpha = jax.jit(_rotation_alpha_impl, static_argnums=(2,))
rotation_alphadash = jax.jit(_rotation_alphadash_impl, static_argnums=(2,))


def rotation_dcoeff(points: object, order: int) -> np.ndarray:
    """Return the host-side Jacobian of ``rotation_alpha`` w.r.t. rotation DOFs."""
    points_array = np.asarray(points, dtype=np.float64)
    npoints = points_array.shape[0]
    int_order = int(order)
    jac = np.zeros((npoints, 2 * int_order + 1), dtype=np.float64)
    jac[:, 0] = 1.0
    for mode in range(1, int_order + 1):
        angle = 2.0 * np.pi * mode * points_array
        jac[:, 2 * mode - 1] = np.sin(angle)
        jac[:, 2 * mode] = np.cos(angle)
    return jac


def rotationdash_dcoeff(points: object, order: int) -> np.ndarray:
    """Return the host-side Jacobian of ``rotation_alphadash`` w.r.t. rotation DOFs."""
    points_array = np.asarray(points, dtype=np.float64)
    npoints = points_array.shape[0]
    int_order = int(order)
    jac = np.zeros((npoints, 2 * int_order + 1), dtype=np.float64)
    for mode in range(1, int_order + 1):
        scale = 2.0 * np.pi * mode
        angle = scale * points_array
        jac[:, 2 * mode - 1] = scale * np.cos(angle)
        jac[:, 2 * mode] = -scale * np.sin(angle)
    return jac
