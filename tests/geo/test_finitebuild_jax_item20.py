"""Wave R4 item 20 parity tests for finite-build CurveFilament JAX paths."""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks.validation_ladder_contract import parity_ladder_tolerances
from simsopt.geo import (
    CurveFilament,
    FrameRotation,
    FramedCurveCentroid,
    FramedCurveFrenet,
    ZeroRotation,
    create_multifilament_grid,
)
from simsopt.geo.curveplanarfourier import CurvePlanarFourier
from simsopt_jax.core import (
    CurveFilamentSpec,
    FrameRotationSpec,
    ZeroRotationSpec,
    curve_gamma_vjp_from_dofs,
    curve_gamma_and_dash_from_spec,
    curve_gamma_and_dash_from_dofs,
    curve_gammadash_vjp_from_dofs,
    curve_pullback_from_spec,
    make_curve_filament_spec,
    make_curve_planarfourier_spec,
    make_frame_rotation_spec,
    make_optimizable_dof_map_spec,
    make_zero_rotation_spec,
)


_DIRECT_KERNEL = parity_ladder_tolerances("direct_kernel")
_RTOL = _DIRECT_KERNEL["rtol"]
_ATOL = _DIRECT_KERNEL["atol"]

_NQUADPOINTS = 48
_CURVE_ORDER = 2
_CURVE_DOFS = np.array(
    [1.1, 0.14, -0.09, 0.05, -0.02, 1.0, 0.2, -0.1, 0.3, 0.15, -0.2, 0.05],
    dtype=np.float64,
)
_ROTATION_DOFS = np.array([0.07, -0.03, 0.02], dtype=np.float64)


def _build_base_curve() -> CurvePlanarFourier:
    curve = CurvePlanarFourier(_NQUADPOINTS, order=_CURVE_ORDER)
    curve.set_dofs(_CURVE_DOFS)
    return curve


def _build_rotation(curve: CurvePlanarFourier, rotation_kind: str):
    if rotation_kind == "zero":
        return ZeroRotation(curve.quadpoints)
    rotation = FrameRotation(curve.quadpoints, order=1)
    rotation.x = _ROTATION_DOFS
    return rotation


def _build_framed_curve(
    curve: CurvePlanarFourier,
    frame_kind: str,
    rotation_kind: str,
):
    rotation = _build_rotation(curve, rotation_kind)
    if frame_kind == "centroid":
        return FramedCurveCentroid(curve, rotation)
    return FramedCurveFrenet(curve, rotation)


def _local_dof_map(template_dofs: np.ndarray, owner_start: int):
    size = int(template_dofs.size)
    owner_segments = (
        ((int(owner_start), int(owner_start) + size, 0, size),) if size else ()
    )
    return make_optimizable_dof_map_spec(
        template_full_dofs=template_dofs,
        owner_segments=owner_segments,
        input_mode="local",
        input_start=0,
        input_end=size,
    )


def _rotation_spec_and_dofs(curve: CurvePlanarFourier, rotation):
    if isinstance(rotation, ZeroRotation):
        return make_zero_rotation_spec(quadpoints=curve.quadpoints), np.empty(0)
    if isinstance(rotation, FrameRotation):
        dofs = np.asarray(rotation.full_x, dtype=np.float64)
        return (
            make_frame_rotation_spec(
                dofs=dofs,
                quadpoints=curve.quadpoints,
                order=rotation.order,
                scale=rotation.scale,
            ),
            dofs,
        )
    raise TypeError(f"Unsupported finite-build rotation: {type(rotation).__name__}")


def _curvefilament_spec(
    *,
    curve: CurvePlanarFourier,
    rotation,
    frame_kind: str,
    dn: float,
    db: float,
) -> CurveFilamentSpec:
    base_dofs = np.asarray(curve.get_dofs(), dtype=np.float64)
    rotation_spec, rotation_dofs = _rotation_spec_and_dofs(curve, rotation)
    owner_dofs = (
        np.concatenate((base_dofs, rotation_dofs))
        if rotation_dofs.size
        else base_dofs.copy()
    )
    return make_curve_filament_spec(
        dofs=owner_dofs,
        quadpoints=curve.quadpoints,
        base_curve=make_curve_planarfourier_spec(
            dofs=base_dofs,
            quadpoints=curve.quadpoints,
            order=_CURVE_ORDER,
        ),
        base_curve_map=_local_dof_map(base_dofs, 0),
        rotation=rotation_spec,
        rotation_map=_local_dof_map(rotation_dofs, base_dofs.size),
        frame_kind=frame_kind,
        dn=dn,
        db=db,
    )


def _curvefilament_spec_from_live(
    filament: CurveFilament,
    frame_kind: str,
) -> CurveFilamentSpec:
    return _curvefilament_spec(
        curve=filament.curve,
        rotation=filament.rotation,
        frame_kind=frame_kind,
        dn=filament.dn,
        db=filament.db,
    )


def _assert_live_and_spec_geometry_match(
    curve: CurveFilament,
    spec: CurveFilamentSpec,
) -> None:
    assert isinstance(spec, CurveFilamentSpec)
    gamma, gammadash = jax.jit(curve_gamma_and_dash_from_spec)(spec)
    np.testing.assert_allclose(
        np.asarray(gamma),
        np.asarray(curve.gamma()),
        rtol=_RTOL,
        atol=_ATOL,
    )
    np.testing.assert_allclose(
        np.asarray(gammadash),
        np.asarray(curve.gammadash()),
        rtol=_RTOL,
        atol=_ATOL,
    )


@pytest.mark.parametrize("frame_kind", ("centroid", "frenet"))
@pytest.mark.parametrize("rotation_order", (None, 1))
def test_multifilament_grid_preserves_offsets_and_spec_geometry(
    frame_kind: str,
    rotation_order: Optional[int],
):
    """Grid construction keeps the expected offsets and JAX spec geometry."""
    curve = _build_base_curve()
    filaments = create_multifilament_grid(
        curve,
        numfilaments_n=2,
        numfilaments_b=3,
        gapsize_n=0.02,
        gapsize_b=0.03,
        rotation_order=rotation_order,
        frame=frame_kind,
    )

    expected_offsets = (
        (-0.01, -0.03),
        (-0.01, 0.0),
        (-0.01, 0.03),
        (0.01, -0.03),
        (0.01, 0.0),
        (0.01, 0.03),
    )
    assert len(filaments) == len(expected_offsets)
    shared_rotation = filaments[0].rotation

    for filament, (dn, db) in zip(filaments, expected_offsets):
        assert filament.curve is curve
        assert filament.rotation is shared_rotation
        assert filament.dn == pytest.approx(dn)
        assert filament.db == pytest.approx(db)
        spec = _curvefilament_spec_from_live(filament, frame_kind)
        assert spec.frame_kind == frame_kind
        assert spec.dn == pytest.approx(dn)
        assert spec.db == pytest.approx(db)
        if rotation_order is None:
            assert isinstance(spec.rotation, ZeroRotationSpec)
        else:
            assert isinstance(spec.rotation, FrameRotationSpec)
        _assert_live_and_spec_geometry_match(filament, spec)


_FD_STEP = 1e-5
_FD_RTOL = 1e-6
_FD_ATOL = 1e-8


def _central_fd_gradient(scalar_fn, dofs: np.ndarray, step: float) -> np.ndarray:
    """Central finite difference of a scalar function of a 1D dof vector."""
    grad = np.empty_like(dofs)
    for i in range(dofs.size):
        plus = dofs.copy()
        minus = dofs.copy()
        plus[i] += step
        minus[i] -= step
        grad[i] = (scalar_fn(plus) - scalar_fn(minus)) / (2.0 * step)
    return grad


@pytest.mark.parametrize("frame_kind", ("centroid", "frenet"))
@pytest.mark.parametrize("rotation_kind", ("zero", "frame"))
def test_curvefilament_jax_gamma_vjp_matches_central_fd(
    frame_kind: str,
    rotation_kind: str,
):
    """JAX VJP of gamma w.r.t. CurveFilament dofs matches centered FD."""
    curve = _build_base_curve()
    framed_curve = _build_framed_curve(curve, frame_kind, rotation_kind)
    filament = CurveFilament(framed_curve, dn=0.012, db=-0.009)
    spec = _curvefilament_spec_from_live(filament, frame_kind)

    gamma, _gammadash = curve_gamma_and_dash_from_spec(spec)
    gamma = np.asarray(gamma, dtype=np.float64)
    cotangent = np.reshape(
        np.linspace(0.2, 1.1, gamma.size, dtype=np.float64),
        gamma.shape,
    )
    dofs0 = np.asarray(spec.dofs, dtype=np.float64)
    cotangent_j = jnp.asarray(cotangent, dtype=jnp.float64)

    jax_grad = np.asarray(
        curve_gamma_vjp_from_dofs(spec, jnp.asarray(dofs0), cotangent_j)
    )

    def scalar(dofs: np.ndarray) -> float:
        g, _gd = curve_gamma_and_dash_from_dofs(
            spec,
            jnp.asarray(dofs, dtype=jnp.float64),
        )
        return float(jnp.sum(cotangent_j * g))

    fd_grad = _central_fd_gradient(scalar, dofs0, _FD_STEP)
    np.testing.assert_allclose(jax_grad, fd_grad, rtol=_FD_RTOL, atol=_FD_ATOL)


@pytest.mark.parametrize("frame_kind", ("centroid", "frenet"))
@pytest.mark.parametrize("rotation_kind", ("zero", "frame"))
def test_curvefilament_jax_gammadash_vjp_matches_central_fd(
    frame_kind: str,
    rotation_kind: str,
):
    """JAX VJP of gammadash w.r.t. CurveFilament dofs matches centered FD."""
    curve = _build_base_curve()
    framed_curve = _build_framed_curve(curve, frame_kind, rotation_kind)
    filament = CurveFilament(framed_curve, dn=0.012, db=-0.009)
    spec = _curvefilament_spec_from_live(filament, frame_kind)

    _gamma, gammadash = curve_gamma_and_dash_from_spec(spec)
    gammadash = np.asarray(gammadash, dtype=np.float64)
    cotangent = np.reshape(
        np.linspace(-0.7, 0.4, gammadash.size, dtype=np.float64),
        gammadash.shape,
    )
    dofs0 = np.asarray(spec.dofs, dtype=np.float64)
    cotangent_j = jnp.asarray(cotangent, dtype=jnp.float64)

    jax_grad = np.asarray(
        curve_gammadash_vjp_from_dofs(spec, jnp.asarray(dofs0), cotangent_j)
    )

    def scalar(dofs: np.ndarray) -> float:
        _g, gd = curve_gamma_and_dash_from_dofs(
            spec,
            jnp.asarray(dofs, dtype=jnp.float64),
        )
        return float(jnp.sum(cotangent_j * gd))

    fd_grad = _central_fd_gradient(scalar, dofs0, _FD_STEP)
    np.testing.assert_allclose(jax_grad, fd_grad, rtol=_FD_RTOL, atol=_FD_ATOL)


@pytest.mark.parametrize("frame_kind", ("centroid", "frenet"))
@pytest.mark.parametrize("rotation_kind", ("zero", "frame"))
def test_curvefilament_spec_pullback_matches_central_fd(
    frame_kind: str,
    rotation_kind: str,
):
    """Spec-pullback VJP w.r.t. CurveFilament dofs matches centered FD."""
    curve = _build_base_curve()
    framed_curve = _build_framed_curve(curve, frame_kind, rotation_kind)
    filament = CurveFilament(framed_curve, dn=0.012, db=-0.009)
    spec = _curvefilament_spec_from_live(filament, frame_kind)

    gamma, gammadash = curve_gamma_and_dash_from_spec(spec)
    gamma = np.asarray(gamma, dtype=np.float64)
    gammadash = np.asarray(gammadash, dtype=np.float64)
    dg = np.reshape(
        np.linspace(0.2, 1.1, gamma.size, dtype=np.float64),
        gamma.shape,
    )
    dgd = np.reshape(
        np.linspace(-0.7, 0.4, gammadash.size, dtype=np.float64),
        gammadash.shape,
    )
    dg_j = jnp.asarray(dg, dtype=jnp.float64)
    dgd_j = jnp.asarray(dgd, dtype=jnp.float64)

    spec_cotangent, surface_cotangent = curve_pullback_from_spec(
        spec,
        dg_j,
        dgd_j,
    )
    assert surface_cotangent is None
    jax_grad = np.asarray(spec_cotangent)

    dofs0 = np.asarray(spec.dofs, dtype=np.float64)

    def scalar(dofs: np.ndarray) -> float:
        dofs_j = jnp.asarray(dofs, dtype=jnp.float64)
        g, gd = curve_gamma_and_dash_from_dofs(spec, dofs_j)
        return float(jnp.sum(dg_j * g) + jnp.sum(dgd_j * gd))

    fd_grad = _central_fd_gradient(scalar, dofs0, _FD_STEP)
    np.testing.assert_allclose(jax_grad, fd_grad, rtol=_FD_RTOL, atol=_FD_ATOL)
