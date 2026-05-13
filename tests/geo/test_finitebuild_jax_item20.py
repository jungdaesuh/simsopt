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
from simsopt.jax_core import (
    CurveFilamentSpec,
    FrameRotationSpec,
    ZeroRotationSpec,
    curve_gamma_and_dash_from_spec,
    curve_pullback_from_spec,
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


def _assert_live_and_spec_geometry_match(curve: CurveFilament) -> None:
    spec = curve.to_spec()
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
        spec = filament.to_spec()
        assert spec.frame_kind == frame_kind
        assert spec.dn == pytest.approx(dn)
        assert spec.db == pytest.approx(db)
        if rotation_order is None:
            assert isinstance(spec.rotation, ZeroRotationSpec)
        else:
            assert isinstance(spec.rotation, FrameRotationSpec)
        _assert_live_and_spec_geometry_match(filament)


@pytest.mark.parametrize("frame_kind", ("centroid", "frenet"))
@pytest.mark.parametrize("rotation_kind", ("zero", "frame"))
def test_curvefilament_jax_vjps_match_public_derivative_methods(
    frame_kind: str,
    rotation_kind: str,
):
    """Finite-build JAX VJPs match the public Optimizable derivative contract."""
    curve = _build_base_curve()
    framed_curve = _build_framed_curve(curve, frame_kind, rotation_kind)
    filament = CurveFilament(framed_curve, dn=0.012, db=-0.009)

    gamma = np.asarray(filament.gamma(), dtype=np.float64)
    gammadash = np.asarray(filament.gammadash(), dtype=np.float64)
    dg = np.reshape(
        np.linspace(0.2, 1.1, gamma.size, dtype=np.float64),
        gamma.shape,
    )
    dgd = np.reshape(
        np.linspace(-0.7, 0.4, gammadash.size, dtype=np.float64),
        gammadash.shape,
    )

    expected_gamma_vjp = filament.dgamma_by_dcoeff_vjp(dg)(filament)
    expected_gammadash_vjp = filament.dgammadash_by_dcoeff_vjp(dgd)(filament)
    dofs = jnp.asarray(filament.full_x, dtype=jnp.float64)

    np.testing.assert_allclose(
        np.asarray(filament.dgamma_by_dcoeff_vjp_jax(dofs, dg)),
        expected_gamma_vjp,
        rtol=_RTOL,
        atol=_ATOL,
    )
    np.testing.assert_allclose(
        np.asarray(filament.dgammadash_by_dcoeff_vjp_jax(dofs, dgd)),
        expected_gammadash_vjp,
        rtol=_RTOL,
        atol=_ATOL,
    )

    spec_cotangent, surface_cotangent = curve_pullback_from_spec(
        filament.to_spec(),
        jnp.asarray(dg, dtype=jnp.float64),
        jnp.asarray(dgd, dtype=jnp.float64),
    )
    assert surface_cotangent is None
    np.testing.assert_allclose(
        np.asarray(spec_cotangent),
        expected_gamma_vjp + expected_gammadash_vjp,
        rtol=_RTOL,
        atol=_ATOL,
    )
