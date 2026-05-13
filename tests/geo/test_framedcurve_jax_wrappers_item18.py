"""Parity tests for the JAX-backed framed-curve Optimizable wrappers.

Wave R4 item 18 wrapper closure: ``FrameRotationJAX``,
``ZeroRotationJAX``, ``FramedCurveFrenetJAX``, and
``FramedCurveCentroidJAX`` mirror the public API of their host
counterparts (``simsopt.geo.framedcurve.FrameRotation``,
``ZeroRotation``, ``FramedCurveFrenet``, ``FramedCurveCentroid``) and
route hot paths through the JAX kernels in
``simsopt.jax_core.framedcurve``. These tests pin the wrapper outputs
to the upstream host classes at the ``direct_kernel`` parity-ladder
lane and confirm the DOF round-trip / dependency-graph contract.
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.validation_ladder_contract import parity_ladder_tolerances
from simsopt.geo.curveplanarfourier import CurvePlanarFourier
from simsopt.geo.framedcurve import (
    FrameRotation,
    FramedCurveCentroid,
    FramedCurveFrenet,
    ZeroRotation,
)
from simsopt.geo.framedcurve_jax import (
    FrameRotationJAX,
    FramedCurveCentroidJAX,
    FramedCurveFrenetJAX,
    ZeroRotationJAX,
)


_DIRECT_KERNEL = parity_ladder_tolerances("direct_kernel")
_RTOL = _DIRECT_KERNEL["rtol"]
_ATOL = _DIRECT_KERNEL["atol"]

_NQUADPOINTS = 48
_CURVE_ORDER = 2
_CURVE_DOFS = np.array(
    [1.05, 0.12, -0.08, 0.04, -0.02, 1.02, 0.18, -0.09, 0.27, 0.13, -0.18, 0.05],
    dtype=np.float64,
)
_ROTATION_ORDER = 2
_ROTATION_DOFS = np.array([0.1, 0.2, -0.3, 0.05, -0.05], dtype=np.float64)


def _build_curve() -> CurvePlanarFourier:
    curve = CurvePlanarFourier(_NQUADPOINTS, order=_CURVE_ORDER)
    curve.set_dofs(_CURVE_DOFS)
    return curve


def _assert_arrays_close(label: str, actual, expected) -> None:
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float64),
        np.asarray(expected, dtype=np.float64),
        rtol=_RTOL,
        atol=_ATOL,
        err_msg=label,
    )


def _assert_frame_close(label: str, actual, expected) -> None:
    for axis, exp, act in zip(("t", "n", "b"), expected, actual, strict=True):
        _assert_arrays_close(f"{label}: {axis}", act, exp)


def test_frame_rotation_jax_matches_host():
    quad = np.linspace(0.0, 1.0, _NQUADPOINTS, endpoint=False, dtype=np.float64)
    host = FrameRotation(quad, _ROTATION_ORDER)
    host.x = _ROTATION_DOFS.copy()

    jax_wrapper = FrameRotationJAX(quad, _ROTATION_ORDER)
    jax_wrapper.x = _ROTATION_DOFS.copy()

    _assert_arrays_close("alpha", jax_wrapper.alpha(quad), host.alpha(quad))
    _assert_arrays_close("alphadash", jax_wrapper.alphadash(quad), host.alphadash(quad))

    cotangent = np.linspace(-0.5, 0.5, _NQUADPOINTS, dtype=np.float64)
    host_d = host.dalpha_by_dcoeff_vjp(quad, cotangent)
    jax_d = jax_wrapper.dalpha_by_dcoeff_vjp(quad, cotangent)
    _assert_arrays_close("dalpha_by_dcoeff_vjp", jax_d(jax_wrapper), host_d(host))

    host_dd = host.dalphadash_by_dcoeff_vjp(quad, cotangent)
    jax_dd = jax_wrapper.dalphadash_by_dcoeff_vjp(quad, cotangent)
    _assert_arrays_close("dalphadash_by_dcoeff_vjp", jax_dd(jax_wrapper), host_dd(host))


def test_frame_rotation_jax_dof_round_trip():
    quad = np.linspace(0.0, 1.0, _NQUADPOINTS, endpoint=False, dtype=np.float64)
    jax_wrapper = FrameRotationJAX(quad, _ROTATION_ORDER)
    jax_wrapper.x = _ROTATION_DOFS.copy()

    round_trip = np.asarray(jax_wrapper.x, dtype=np.float64)
    _assert_arrays_close("FrameRotationJAX DOF round trip", round_trip, _ROTATION_DOFS)
    assert jax_wrapper.order == _ROTATION_ORDER
    assert jax_wrapper.local_dof_size == _ROTATION_DOFS.size


def test_zero_rotation_jax_matches_host():
    quad = np.linspace(0.0, 1.0, _NQUADPOINTS, endpoint=False, dtype=np.float64)
    host = ZeroRotation(quad)
    jax_wrapper = ZeroRotationJAX(quad)

    _assert_arrays_close("zero alpha", jax_wrapper.alpha(quad), host.alpha(quad))
    _assert_arrays_close(
        "zero alphadash", jax_wrapper.alphadash(quad), host.alphadash(quad)
    )

    v = np.linspace(-0.2, 0.4, _NQUADPOINTS, dtype=np.float64)
    host_d = host.dalpha_by_dcoeff_vjp(quad, v)
    jax_d = jax_wrapper.dalpha_by_dcoeff_vjp(quad, v)
    assert dict(host_d.data) == {}
    assert dict(jax_d.data) == {}

    host_dd = host.dalphadash_by_dcoeff_vjp(quad, v)
    jax_dd = jax_wrapper.dalphadash_by_dcoeff_vjp(quad, v)
    assert dict(host_dd.data) == {}
    assert dict(jax_dd.data) == {}


@pytest.mark.parametrize("rotation_kind", ("zero", "frame"))
def test_framed_curve_frenet_jax_matches_host(rotation_kind: str):
    curve = _build_curve()
    quad = curve.quadpoints
    if rotation_kind == "zero":
        host_rotation = ZeroRotation(quad)
        jax_rotation = ZeroRotationJAX(quad)
    else:
        host_rotation = FrameRotation(quad, _ROTATION_ORDER)
        host_rotation.x = _ROTATION_DOFS.copy()
        jax_rotation = FrameRotationJAX(quad, _ROTATION_ORDER)
        jax_rotation.x = _ROTATION_DOFS.copy()

    host_framed = FramedCurveFrenet(curve, host_rotation)
    jax_framed = FramedCurveFrenetJAX(curve, jax_rotation)

    _assert_frame_close(
        f"Frenet rotated_frame ({rotation_kind})",
        jax_framed.rotated_frame(),
        host_framed.rotated_frame(),
    )
    _assert_frame_close(
        f"Frenet rotated_frame_dash ({rotation_kind})",
        jax_framed.rotated_frame_dash(),
        host_framed.rotated_frame_dash(),
    )
    _assert_arrays_close(
        f"Frenet frame_torsion ({rotation_kind})",
        jax_framed.frame_torsion(),
        host_framed.frame_torsion(),
    )
    _assert_arrays_close(
        f"Frenet frame_binormal_curvature ({rotation_kind})",
        jax_framed.frame_binormal_curvature(),
        host_framed.frame_binormal_curvature(),
    )


@pytest.mark.parametrize("rotation_kind", ("zero", "frame"))
def test_framed_curve_centroid_jax_matches_host(rotation_kind: str):
    curve = _build_curve()
    quad = curve.quadpoints
    if rotation_kind == "zero":
        host_rotation = ZeroRotation(quad)
        jax_rotation = ZeroRotationJAX(quad)
    else:
        host_rotation = FrameRotation(quad, _ROTATION_ORDER)
        host_rotation.x = _ROTATION_DOFS.copy()
        jax_rotation = FrameRotationJAX(quad, _ROTATION_ORDER)
        jax_rotation.x = _ROTATION_DOFS.copy()

    host_framed = FramedCurveCentroid(curve, host_rotation)
    jax_framed = FramedCurveCentroidJAX(curve, jax_rotation)

    _assert_frame_close(
        f"Centroid rotated_frame ({rotation_kind})",
        jax_framed.rotated_frame(),
        host_framed.rotated_frame(),
    )
    _assert_frame_close(
        f"Centroid rotated_frame_dash ({rotation_kind})",
        jax_framed.rotated_frame_dash(),
        host_framed.rotated_frame_dash(),
    )
    _assert_arrays_close(
        f"Centroid frame_torsion ({rotation_kind})",
        jax_framed.frame_torsion(),
        host_framed.frame_torsion(),
    )
    _assert_arrays_close(
        f"Centroid frame_binormal_curvature ({rotation_kind})",
        jax_framed.frame_binormal_curvature(),
        host_framed.frame_binormal_curvature(),
    )


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
