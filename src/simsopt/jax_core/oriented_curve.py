"""Pure JAX kernels for the OrientedCurveXYZFourier geometry.

CPU oracle: ``simsopt.geo.orientedcurve.OrientedCurveXYZFourier``. These
helpers are pure JAX functions over the curve degrees of freedom and
quadrature points; the host adapter class in
``simsopt.geo.orientedcurve`` re-exports them so existing
``simsopt.geo.orientedcurve.centercurve_pure`` callers continue to work.
"""

from __future__ import annotations

from math import pi

import jax.numpy as jnp


def shift_pure(v, xyz):
    """Apply translation in cartesian coordinates.

    Args:
     - v: array to translate. Should have size Nx3.
     - xyz: translation vector. Should have size 3.

    Returns:
     - v+xyz: translated array, size Nx3
    """
    for ii in range(0, 3):
        v = v.at[:, ii].add(xyz[ii])
    return v


def rotate_pure(v, ypr):
    """Apply rotation around x, y, and z axis.

    Args:
     - v: set of points to rotate. Should have size Nx3.
     - ypr: rotation angles.
            ypr[0] describes the rotation around the z-axis.
            ypr[1] describes the rotation around the y-axis.
            ypr[2] describes the rotation around the x-axis.

    Returns:
    - v: Rotated set of points
    """
    yaw = ypr[0]
    pitch = ypr[1]
    roll = ypr[2]

    Myaw = jnp.asarray(
        [[jnp.cos(yaw), -jnp.sin(yaw), 0], [jnp.sin(yaw), jnp.cos(yaw), 0], [0, 0, 1]]
    )
    Mpitch = jnp.asarray(
        [
            [jnp.cos(pitch), 0, jnp.sin(pitch)],
            [0, 1, 0],
            [-jnp.sin(pitch), 0, jnp.cos(pitch)],
        ]
    )
    Mroll = jnp.asarray(
        [
            [1, 0, 0],
            [0, jnp.cos(roll), -jnp.sin(roll)],
            [0, jnp.sin(roll), jnp.cos(roll)],
        ]
    )

    return v @ Myaw @ Mpitch @ Mroll


def centercurve_pure(dofs, quadpoints, order):
    """Construct curve centered at the origin.

    Args:
     - dofs: Set of degrees of freedom
     - quadpoints: Quadrature points. Array of size N, with float values between 0 and 1.
     - order: Maximum Fourier mode number.

    Returns:
     - gamma: Curve that has been translated and rotated to the desired position.
    """
    xyz = dofs[0:3]
    ypr = dofs[3:6]
    fmn = dofs[6:]

    k = len(fmn) // 3
    coeffs = [fmn[:k], fmn[k : (2 * k)], fmn[(2 * k) :]]
    points = quadpoints
    gamma = jnp.zeros((len(points), 3))
    for i in range(0, 3):
        for j in range(0, order):
            gamma = gamma.at[:, i].add(
                coeffs[i][2 * j] * jnp.sin(2 * pi * (j + 1) * points)
            )
            gamma = gamma.at[:, i].add(
                coeffs[i][2 * j + 1] * jnp.cos(2 * pi * (j + 1) * points)
            )

    return shift_pure(rotate_pure(gamma, ypr), xyz)
