"""Optimizable-contract tests for the JAX-backed framed-curve wrappers.

Wave R4 item 18 wrapper closure: ``FrameRotationJAX``,
``ZeroRotationJAX``, ``FramedCurveFrenetJAX``, and
``FramedCurveCentroidJAX`` sit alongside the C++/host
``simsopt.geo.framedcurve`` classes and route hot paths through the
JAX kernels in ``simsopt.jax_core.framedcurve``.

These tests pin the **Optimizable contract** of the wrappers:

* ``FrameRotationJAX`` round-trips its DOF vector through
  :class:`simsopt._core.optimizable.Optimizable`.
* The wrappers compose into the upstream ``Optimizable`` dependency
  graph (curve + rotation appear in ``wrapper.parents``).
* DOFs of the parent rotation flow into the JAX kernel: a finite
  difference of a scalar functional ``J = sum(frame_torsion ** 2)``
  built from the wrapper output matches a ``jax.grad`` of the same
  functional taken directly on the rotation DOFs at the same base
  point.

Kernel-level parity (analytic closed form on a planar circle,
orthonormality, ``alpha = 0`` reduction, strict transfer-guard
behaviour) is owned by ``tests/geo/test_framedcurve_jax_item18.py``.
The redundant ``*_jax_matches_host`` comparisons that previously
lived in this file were JAX-vs-JAX assertions (the host
``rotated_*_frame`` symbols are re-exports of
``simsopt.jax_core.framedcurve``) and have been replaced with the
FD-on-Optimizable check below.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from simsopt.geo.curveplanarfourier import CurvePlanarFourier
from simsopt.geo.framedcurve_jax import (
    FrameRotationJAX,
    FramedCurveCentroidJAX,
    FramedCurveFrenetJAX,
    ZeroRotationJAX,
)
from simsopt.jax_core.framedcurve import (
    rotated_centroid_frame,
    rotated_centroid_frame_dash,
    rotation_alpha,
    rotation_alphadash,
)


_NQUADPOINTS = 48
_CURVE_ORDER = 2
_CURVE_DOFS = np.array(
    [1.05, 0.12, -0.08, 0.04, -0.02, 1.02, 0.18, -0.09, 0.27, 0.13, -0.18, 0.05],
    dtype=np.float64,
)
_ROTATION_ORDER = 2
_ROTATION_DOFS = np.array([0.1, 0.2, -0.3, 0.05, -0.05], dtype=np.float64)

# ``h = 1e-5`` is the central-difference step size; ``rtol = 1e-5`` /
# ``atol = 1e-7`` are the matching tolerances for a smooth scalar
# functional with magnitudes ~1e-2 (see ``test_boozer_derivatives_jax``
# for the same FD convention).
_FD_STEP = 1e-5
_FD_RTOL = 1e-5
_FD_ATOL = 1e-7


def _build_curve() -> CurvePlanarFourier:
    curve = CurvePlanarFourier(_NQUADPOINTS, order=_CURVE_ORDER)
    curve.set_dofs(_CURVE_DOFS)
    return curve


def test_frame_rotation_jax_dof_round_trip():
    quad = np.linspace(0.0, 1.0, _NQUADPOINTS, endpoint=False, dtype=np.float64)
    jax_wrapper = FrameRotationJAX(quad, _ROTATION_ORDER)
    jax_wrapper.x = _ROTATION_DOFS.copy()

    round_trip = np.asarray(jax_wrapper.x, dtype=np.float64)
    np.testing.assert_allclose(round_trip, _ROTATION_DOFS, rtol=0.0, atol=0.0)
    assert jax_wrapper.order == _ROTATION_ORDER
    assert jax_wrapper.local_dof_size == _ROTATION_DOFS.size


def test_framed_curve_jax_dependency_graph():
    """JAX wrappers compose into the upstream ``Optimizable`` dependency graph."""
    curve = _build_curve()
    rotation = FrameRotationJAX(curve.quadpoints, _ROTATION_ORDER)
    rotation.x = _ROTATION_DOFS.copy()
    frenet = FramedCurveFrenetJAX(curve, rotation)
    centroid = FramedCurveCentroidJAX(curve, ZeroRotationJAX(curve.quadpoints))

    # Direct curve + rotation dependencies must appear in the wrapper's tree.
    assert curve in frenet.parents
    assert rotation in frenet.parents
    assert curve in centroid.parents


def _centroid_torsion_from_rotation_dofs(
    rotation_dofs: jax.Array,
    quadpoints: jax.Array,
    order: int,
    gamma: jax.Array,
    gammadash: jax.Array,
    gammadashdash: jax.Array,
) -> jax.Array:
    """Reproduce ``FramedCurveCentroidJAX.frame_torsion()`` from raw inputs.

    Used as the analytic oracle for the Optimizable-FD test: matches the
    wrapper's internal call graph (``rotation_alpha`` /
    ``rotation_alphadash`` followed by ``rotated_centroid_frame*``) and is
    differentiable through ``rotation_dofs`` so ``jax.grad`` provides the
    closed-form gradient against which the FD slope is checked.
    """
    alpha = rotation_alpha(rotation_dofs, quadpoints, order)
    alphadash = rotation_alphadash(rotation_dofs, quadpoints, order)
    _, _, b = rotated_centroid_frame(gamma, gammadash, alpha)
    _, ndash, _ = rotated_centroid_frame_dash(
        gamma, gammadash, gammadashdash, alpha, alphadash
    )
    arc_length = jnp.linalg.norm(gammadash, axis=1)[:, None]
    return jnp.sum((ndash / arc_length) * b, axis=1)


def _scalar_objective(torsion: jax.Array) -> jax.Array:
    return jnp.sum(torsion * torsion)


def test_frame_rotation_jax_dofs_drive_wrapper_outputs_via_fd():
    """FD check on the JAX Optimizable graph for ``FrameRotationJAX``.

    Validates that perturbing a rotation DOF changes the
    ``FramedCurveCentroidJAX.frame_torsion()`` output consistently with
    the analytic ``jax.grad`` of the same scalar functional. This pins
    the Optimizable wiring (DOFs flow into the wrapper, the wrapper
    drives the kernel) without comparing JAX against another JAX-backed
    oracle.
    """
    curve = _build_curve()
    quad = curve.quadpoints
    rotation = FrameRotationJAX(quad, _ROTATION_ORDER)
    base_dofs = _ROTATION_DOFS.copy()
    rotation.x = base_dofs.copy()
    wrapper = FramedCurveCentroidJAX(curve, rotation)

    # Cache curve quadrature data once; the FD perturbs rotation DOFs only,
    # so the curve outputs are constant across the loop.
    gamma = jnp.asarray(curve.gamma(), dtype=jnp.float64)
    gammadash = jnp.asarray(curve.gammadash(), dtype=jnp.float64)
    gammadashdash = jnp.asarray(curve.gammadashdash(), dtype=jnp.float64)
    quad_jax = jnp.asarray(quad, dtype=jnp.float64)

    def _scalar(rotation_dofs: jax.Array) -> jax.Array:
        torsion = _centroid_torsion_from_rotation_dofs(
            rotation_dofs,
            quad_jax,
            _ROTATION_ORDER,
            gamma,
            gammadash,
            gammadashdash,
        )
        return _scalar_objective(torsion)

    dofs_jax = jnp.asarray(base_dofs, dtype=jnp.float64)
    analytic_grad = np.asarray(jax.grad(_scalar)(dofs_jax), dtype=np.float64)

    # Forward difference against the wrapper itself: each perturbation is
    # injected via ``rotation.x``, exercising the full Optimizable graph.
    fd_grad = np.empty_like(base_dofs)
    for i in range(base_dofs.size):
        plus = base_dofs.copy()
        plus[i] += _FD_STEP
        rotation.x = plus
        torsion_plus = wrapper.frame_torsion()
        j_plus = float(jnp.sum(torsion_plus * torsion_plus))

        minus = base_dofs.copy()
        minus[i] -= _FD_STEP
        rotation.x = minus
        torsion_minus = wrapper.frame_torsion()
        j_minus = float(jnp.sum(torsion_minus * torsion_minus))

        fd_grad[i] = (j_plus - j_minus) / (2.0 * _FD_STEP)

    # Restore base DOFs so any later assertions or fixtures see the same
    # state used to compute ``analytic_grad``.
    rotation.x = base_dofs.copy()

    np.testing.assert_allclose(
        fd_grad,
        analytic_grad,
        rtol=_FD_RTOL,
        atol=_FD_ATOL,
        err_msg=(
            "FramedCurveCentroidJAX.frame_torsion() FD gradient through "
            "FrameRotationJAX DOFs does not match jax.grad oracle. "
            "Indicates the Optimizable graph does not thread DOF "
            "perturbations into the JAX kernel correctly."
        ),
    )
