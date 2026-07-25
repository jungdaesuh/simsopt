"""Wave 4 closeout: pure JAX curve-geometry objective mirrors."""

from __future__ import annotations

from conftest import (
    skip_strict_gpu_collection,
)

skip_strict_gpu_collection(
    "curve objective adapter-boundary suite is not part of strict jax_gpu_parity"
)

import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks.validation_ladder_contract import parity_ladder_tolerances
from simsopt.geo.curve import centroid_pure
from simsopt.geo.curveobjectives import (
    ArclengthVariation,
    CurveCurveDistance,
    CurveSurfaceDistance,
    CurveLength,
    LpCurveCurvature,
    LinkingNumber,
    LpCurveTorsion,
    MeanSquaredCurvature,
    Lp_curvature_pure,
    Lp_torsion_pure,
    cc_distance_pure,
    curve_arclengthvariation_pure,
    curve_length_pure,
    curve_msc_pure,
)
from simsopt_jax_adapters.geo.curve_objectives import (
    CurveCurveDistanceBarrierJAX,
    CurveCurveDistanceJAX,
    CurveLengthJAX,
    CurveSurfaceDistanceJAX,
    LpCurveCurvatureBarrierJAX,
    LpCurveCurvatureJAX,
    LinkingNumberJAX,
    MeanSquaredCurvatureJAX,
    cc_distance_barrier_pure,
    curvature_barrier_pure,
)
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.geo.curvexyzfourier import CurveXYZFourier
from simsopt_jax.core import (
    curve_dkappa_by_dcoeff_from_dofs,
    curve_dtorsion_by_dcoeff_from_dofs,
    curve_geometry_from_dofs,
    curve_geometry_from_spec,
    curve_incremental_arclength_from_dofs,
    curve_incremental_arclength_from_spec,
    curve_kappa_from_dofs,
    curve_kappa_from_spec,
    curve_torsion_from_dofs,
    curve_torsion_from_spec,
)
from simsopt_jax.core.curve_geometry import _curve_geometry_with_third_derivative_from_dofs
from simsopt_jax.core.framedcurve import frenet_frame
from simsopt_jax_adapters.geo.curve_specs import curve_spec_from_adapter_curve


_DIRECT_KERNEL = parity_ladder_tolerances("direct_kernel")
_FD_GRADIENT = parity_ladder_tolerances("fd-gradient")
_RTOL = _DIRECT_KERNEL["rtol"]
_ATOL = _DIRECT_KERNEL["atol"]


def _build_nonplanar_curve(quadpoints=64):
    curve = CurveXYZFourier(quadpoints, order=3)
    curve.set("xc(1)", 1.0)
    curve.set("ys(1)", 1.0)
    curve.set("xs(2)", 0.04)
    curve.set("yc(2)", -0.03)
    curve.set("zs(2)", 0.12)
    curve.set("zc(3)", -0.02)
    return curve


def _curve_kappadash_from_dofs(spec, dofs):
    _gamma, gammadash, gammadashdash, gammadashdashdash = (
        _curve_geometry_with_third_derivative_from_dofs(spec, dofs)
    )
    del _gamma

    def norm(values):
        return jnp.linalg.norm(values, axis=1)

    def inner(left, right):
        return jnp.sum(left * right, axis=1)

    d1_cross_d2 = jnp.cross(gammadash, gammadashdash, axis=1)
    d1_cross_d3 = jnp.cross(gammadash, gammadashdashdash, axis=1)
    return inner(d1_cross_d2, d1_cross_d3) / (
        norm(d1_cross_d2) * norm(gammadash) ** 3
    ) - 3.0 * inner(gammadash, gammadashdash) * norm(d1_cross_d2) / (
        norm(gammadash) ** 5
    )


def _build_offset_nonplanar_curve(x_offset: float):
    curve = _build_nonplanar_curve()
    curve.set("xc(0)", x_offset)
    return curve


def test_curve_geometry_scalar_wrappers_match_cpu_curve_methods_and_jit():
    curve = _build_nonplanar_curve()
    spec = curve_spec_from_adapter_curve(curve)

    np.testing.assert_allclose(
        np.asarray(curve_incremental_arclength_from_spec(spec), dtype=np.float64),
        np.asarray(curve.incremental_arclength(), dtype=np.float64),
        rtol=_RTOL,
        atol=_ATOL,
    )
    np.testing.assert_allclose(
        np.asarray(curve_kappa_from_spec(spec), dtype=np.float64),
        np.asarray(curve.kappa(), dtype=np.float64),
        rtol=_RTOL,
        atol=_ATOL,
    )
    np.testing.assert_allclose(
        np.asarray(curve_torsion_from_spec(spec), dtype=np.float64),
        np.asarray(curve.torsion(), dtype=np.float64),
        rtol=_RTOL,
        atol=_ATOL,
    )

    compiled_scalars = jax.jit(
        lambda dofs: (
            curve_incremental_arclength_from_dofs(spec, dofs),
            curve_kappa_from_dofs(spec, dofs),
            curve_torsion_from_dofs(spec, dofs),
        )
    )
    inc_arc, kappa, torsion = compiled_scalars(spec.dofs)
    assert inc_arc.shape == (len(curve.quadpoints),)
    assert kappa.shape == (len(curve.quadpoints),)
    assert torsion.shape == (len(curve.quadpoints),)


def test_curve_parameter_derivatives_match_legacy_first_second_contract():
    epss = np.asarray([0.5**index for index in range(10, 15)], dtype=np.float64)
    quadpoints = np.asarray([0.6, *(0.6 + epss)], dtype=np.float64)
    curve = _build_nonplanar_curve(quadpoints)
    spec = curve_spec_from_adapter_curve(curve)
    gamma, gammadash, gammadashdash = curve_geometry_from_spec(spec)

    np.testing.assert_allclose(
        np.asarray(gamma, dtype=np.float64),
        np.asarray(curve.gamma(), dtype=np.float64),
        rtol=_RTOL,
        atol=_ATOL,
    )
    np.testing.assert_allclose(
        np.asarray(gammadash, dtype=np.float64),
        np.asarray(curve.gammadash(), dtype=np.float64),
        rtol=_RTOL,
        atol=_ATOL,
    )
    np.testing.assert_allclose(
        np.asarray(gammadashdash, dtype=np.float64),
        np.asarray(curve.gammadashdash(), dtype=np.float64),
        rtol=_RTOL,
        atol=_ATOL,
    )

    err_old = np.inf
    for offset, eps in enumerate(epss, start=1):
        deriv_est = (gamma[offset] - gamma[0]) / eps
        err = float(np.linalg.norm(np.asarray(deriv_est - gammadash[0])))
        assert err < 0.55 * err_old
        err_old = err

    err_old = np.inf
    for offset, eps in enumerate(epss, start=1):
        deriv_est = (gammadash[offset] - gammadash[0]) / eps
        err = float(np.linalg.norm(np.asarray(deriv_est - gammadashdash[0])))
        assert err < 0.55 * err_old
        err_old = err


def test_curve_coefficient_derivative_jacobians_match_cpu_curve_methods():
    curve = _build_nonplanar_curve()
    spec = curve_spec_from_adapter_curve(curve)

    derivative_cases = [
        (0, curve.dgamma_by_dcoeff()),
        (1, curve.dgammadash_by_dcoeff()),
        (2, curve.dgammadashdash_by_dcoeff()),
    ]
    for term_index, cpu_derivative in derivative_cases:
        jax_derivative = jax.jacfwd(
            lambda dofs: curve_geometry_from_dofs(spec, dofs)[term_index]
        )(spec.dofs)
        np.testing.assert_allclose(
            np.asarray(jax_derivative, dtype=np.float64),
            np.asarray(cpu_derivative, dtype=np.float64),
            rtol=1.0e-10,
            atol=1.0e-10,
        )

    jax_third_derivative = jax.jacfwd(
        lambda dofs: _curve_geometry_with_third_derivative_from_dofs(spec, dofs)[3]
    )(spec.dofs)
    np.testing.assert_allclose(
        np.asarray(jax_third_derivative, dtype=np.float64),
        np.asarray(curve.dgammadashdashdash_by_dcoeff(), dtype=np.float64),
        rtol=1.0e-10,
        atol=1.0e-10,
    )


def test_curve_curvature_torsion_and_kappadash_derivatives_match_cpu_methods():
    curve = _build_nonplanar_curve()
    spec = curve_spec_from_adapter_curve(curve)

    np.testing.assert_allclose(
        np.asarray(_curve_kappadash_from_dofs(spec, spec.dofs), dtype=np.float64),
        np.asarray(curve.kappadash(), dtype=np.float64),
        rtol=_RTOL,
        atol=_ATOL,
    )
    np.testing.assert_allclose(
        np.asarray(curve_dkappa_by_dcoeff_from_dofs(spec, spec.dofs), dtype=np.float64),
        np.asarray(curve.dkappa_by_dcoeff(), dtype=np.float64),
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        np.asarray(curve_dtorsion_by_dcoeff_from_dofs(spec, spec.dofs), dtype=np.float64),
        np.asarray(curve.dtorsion_by_dcoeff(), dtype=np.float64),
        rtol=1.0e-10,
        atol=1.0e-10,
    )

    jax_kappadash_derivative = jax.jacfwd(
        lambda dofs: _curve_kappadash_from_dofs(spec, dofs)
    )(spec.dofs)
    np.testing.assert_allclose(
        np.asarray(jax_kappadash_derivative, dtype=np.float64),
        np.asarray(curve.dkappadash_by_dcoeff(), dtype=np.float64),
        rtol=1.0e-10,
        atol=1.0e-10,
    )


def test_curve_frenet_frame_and_derivatives_match_cpu_methods():
    curve = _build_nonplanar_curve()
    spec = curve_spec_from_adapter_curve(curve)
    gamma, gammadash, gammadashdash = curve_geometry_from_spec(spec)
    jax_frame = frenet_frame(gamma, gammadash, gammadashdash)
    cpu_frame = curve.frenet_frame()

    for jax_component, cpu_component in zip(jax_frame, cpu_frame):
        np.testing.assert_allclose(
            np.asarray(jax_component, dtype=np.float64),
            np.asarray(cpu_component, dtype=np.float64),
            rtol=_RTOL,
            atol=_ATOL,
        )

    tangent, normal, binormal = jax_frame
    for left, right in [
        (tangent, normal),
        (tangent, binormal),
        (normal, binormal),
    ]:
        np.testing.assert_allclose(
            np.sum(np.asarray(left * right, dtype=np.float64), axis=1),
            0.0,
            atol=1.0e-13,
        )
    for component in jax_frame:
        np.testing.assert_allclose(
            np.sum(np.asarray(component * component, dtype=np.float64), axis=1),
            1.0,
            atol=1.0e-13,
        )

    cpu_frame_derivatives = curve.dfrenet_frame_by_dcoeff()
    for frame_index, cpu_derivative in enumerate(cpu_frame_derivatives):
        jax_derivative = jax.jacfwd(
            lambda dofs: frenet_frame(*curve_geometry_from_dofs(spec, dofs))[
                frame_index
            ]
        )(spec.dofs)
        np.testing.assert_allclose(
            np.asarray(jax_derivative, dtype=np.float64),
            np.asarray(cpu_derivative, dtype=np.float64),
            rtol=1.0e-10,
            atol=1.0e-10,
        )


def test_curve_centroid_matches_cpu_method():
    curve = _build_nonplanar_curve()
    spec = curve_spec_from_adapter_curve(curve)
    gamma, gammadash, _gammadashdash = curve_geometry_from_spec(spec)

    np.testing.assert_allclose(
        np.asarray(centroid_pure(gamma, gammadash), dtype=np.float64),
        np.asarray(curve.centroid(), dtype=np.float64),
        rtol=_RTOL,
        atol=_ATOL,
    )


def test_representative_curve_objective_mirrors_match_cpu_values():
    curve = _build_nonplanar_curve()
    spec = curve_spec_from_adapter_curve(curve)

    inc_arc = curve_incremental_arclength_from_spec(spec)
    kappa = curve_kappa_from_spec(spec)
    torsion = curve_torsion_from_spec(spec)
    _gamma, gammadash, _gammadashdash = curve_geometry_from_spec(spec)
    del _gamma, _gammadashdash

    curvature = LpCurveCurvature(curve, p=2, threshold=0.0)
    torsion_objective = LpCurveTorsion(curve, p=2, threshold=0.0)
    arclength_variation = ArclengthVariation(curve, nintervals=8)

    np.testing.assert_allclose(
        np.asarray(curve_length_pure(inc_arc), dtype=np.float64),
        np.asarray(CurveLength(curve).J(), dtype=np.float64),
        rtol=_RTOL,
        atol=_ATOL,
    )
    np.testing.assert_allclose(
        np.asarray(Lp_curvature_pure(kappa, gammadash, 2, 0.0), dtype=np.float64),
        np.asarray(curvature.J(), dtype=np.float64),
        rtol=_RTOL,
        atol=_ATOL,
    )
    np.testing.assert_allclose(
        np.asarray(Lp_torsion_pure(torsion, gammadash, 2, 0.0), dtype=np.float64),
        np.asarray(torsion_objective.J(), dtype=np.float64),
        rtol=_RTOL,
        atol=_ATOL,
    )
    np.testing.assert_allclose(
        np.asarray(
            curve_arclengthvariation_pure(inc_arc, arclength_variation.mat),
            dtype=np.float64,
        ),
        np.asarray(arclength_variation.J(), dtype=np.float64),
        rtol=_RTOL,
        atol=_ATOL,
    )
    np.testing.assert_allclose(
        np.asarray(curve_msc_pure(kappa, gammadash), dtype=np.float64),
        np.asarray(MeanSquaredCurvature(curve).J(), dtype=np.float64),
        rtol=_RTOL,
        atol=_ATOL,
    )


def test_remaining_curve_objective_mirrors_match_cpu_values():
    curve1 = _build_offset_nonplanar_curve(0.0)
    curve2 = _build_offset_nonplanar_curve(3.5)

    spec1 = curve_spec_from_adapter_curve(curve1)
    spec2 = curve_spec_from_adapter_curve(curve2)
    gamma1, gammadash1, _ = curve_geometry_from_spec(spec1)
    gamma2, gammadash2, _ = curve_geometry_from_spec(spec2)
    kappa1 = curve_kappa_from_spec(spec1)

    curvature_threshold = 2.0 * float(np.max(curve1.kappa()))
    curvature_barrier = LpCurveCurvatureBarrierJAX(curve1, curvature_threshold)
    distance = CurveCurveDistance([curve1, curve2], minimum_distance=10.0)
    sampled_min_distance = min(
        np.linalg.norm(first - second)
        for first in np.asarray(curve1.gamma(), dtype=np.float64)
        for second in np.asarray(curve2.gamma(), dtype=np.float64)
    )
    distance_barrier_threshold = 0.5 * sampled_min_distance
    distance_barrier = CurveCurveDistanceBarrierJAX(
        [curve1, curve2],
        minimum_distance=distance_barrier_threshold,
    )

    np.testing.assert_allclose(
        np.asarray(
            curvature_barrier_pure(kappa1, gammadash1, curvature_threshold),
            dtype=np.float64,
        ),
        np.asarray(curvature_barrier.J(), dtype=np.float64),
        rtol=_RTOL,
        atol=_ATOL,
    )
    np.testing.assert_allclose(
        np.asarray(
            cc_distance_pure(gamma2, gammadash2, gamma1, gammadash1, 10.0),
            dtype=np.float64,
        ),
        np.asarray(distance.J(), dtype=np.float64),
        rtol=_RTOL,
        atol=_ATOL,
    )
    np.testing.assert_allclose(
        np.asarray(
            cc_distance_barrier_pure(
                gamma2,
                gammadash2,
                gamma1,
                gammadash1,
                distance_barrier_threshold,
            ),
            dtype=np.float64,
        ),
        np.asarray(distance_barrier.J(), dtype=np.float64),
        rtol=_RTOL,
        atol=_ATOL,
    )


def _assert_objective_matches_cpu(cpu_objective, jax_objective):
    np.testing.assert_allclose(
        np.asarray(jax_objective.J(), dtype=np.float64),
        np.asarray(cpu_objective.J(), dtype=np.float64),
        rtol=_RTOL,
        atol=_ATOL,
    )
    np.testing.assert_allclose(
        np.asarray(jax_objective.dJ(), dtype=np.float64),
        np.asarray(cpu_objective.dJ(), dtype=np.float64),
        rtol=5e-9,
        atol=5e-10,
    )


def _assert_curve_objective_directional_fd(objective, curve):
    x0 = np.asarray(curve.x, dtype=np.float64).copy()
    direction = np.linspace(-1.0, 1.0, x0.size, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    gradient = np.asarray(objective.dJ(), dtype=np.float64)
    directional_gradient = float(np.dot(gradient, direction))

    step = 1.0e-6
    curve.x = x0 + step * direction
    value_plus = float(objective.J())
    curve.x = x0 - step * direction
    value_minus = float(objective.J())
    curve.x = x0

    directional_fd = (value_plus - value_minus) / (2.0 * step)
    np.testing.assert_allclose(
        directional_gradient,
        directional_fd,
        rtol=float(_FD_GRADIENT["directional_fd_rtol"]),
        atol=float(_FD_GRADIENT["directional_fd_atol"]),
    )


@pytest.mark.parametrize(
    "objective_factory",
    [
        lambda curve: CurveLengthJAX(curve),
        lambda curve: LpCurveCurvatureJAX(curve, p=2, threshold=0.0),
        lambda curve: MeanSquaredCurvatureJAX(curve),
        lambda curve: LpCurveCurvatureBarrierJAX(
            curve,
            2.0 * float(np.max(curve.kappa())),
        ),
    ],
)
def test_public_curve_objective_jax_gradients_match_directional_fd(
    objective_factory,
):
    curve = _build_nonplanar_curve()
    _assert_curve_objective_directional_fd(objective_factory(curve), curve)


def test_public_curve_objective_jax_wrappers_match_cpu_values_and_gradients():
    curve = _build_nonplanar_curve()

    _assert_objective_matches_cpu(CurveLength(curve), CurveLengthJAX(curve))
    _assert_objective_matches_cpu(
        LpCurveCurvature(curve, p=2, threshold=0.0),
        LpCurveCurvatureJAX(curve, p=2, threshold=0.0),
    )
    _assert_objective_matches_cpu(
        MeanSquaredCurvature(curve),
        MeanSquaredCurvatureJAX(curve),
    )


def test_lp_curve_curvature_jax_value_composes_at_the_host_boundary():
    curve = _build_nonplanar_curve()
    objective = LpCurveCurvatureJAX(curve, p=2, threshold=0.0)
    scaled_objective = 3.0 * objective

    value = objective.J()
    scaled_value = scaled_objective.J()

    assert isinstance(value, float)
    np.testing.assert_allclose(scaled_value, 3.0 * value, rtol=0.0, atol=0.0)


def test_public_curve_distance_jax_wrappers_match_cpu_values_and_gradients():
    curve1 = _build_offset_nonplanar_curve(0.0)
    curve2 = _build_offset_nonplanar_curve(0.3)

    distance_cpu = CurveCurveDistance(
        [curve1, curve2],
        minimum_distance=0.75,
        num_basecurves=2,
    )
    distance_jax = CurveCurveDistanceJAX(
        [curve1, curve2],
        minimum_distance=0.75,
        num_basecurves=2,
    )
    _assert_objective_matches_cpu(distance_cpu, distance_jax)


def test_curve_distance_jax_wrapper_signatures_match_cpu_contracts():
    assert inspect.signature(CurveCurveDistanceJAX.__init__) == inspect.signature(
        CurveCurveDistance.__init__
    )
    assert inspect.signature(CurveSurfaceDistanceJAX.__init__) == inspect.signature(
        CurveSurfaceDistance.__init__
    )
    assert inspect.signature(LinkingNumberJAX.__init__) == inspect.signature(
        LinkingNumber.__init__
    )


def test_public_linking_number_jax_wrapper_matches_cpu_value_and_gradient():
    curves = [_build_offset_nonplanar_curve(0.0), _build_offset_nonplanar_curve(0.3)]
    _assert_objective_matches_cpu(
        LinkingNumber(curves, downsample=2),
        LinkingNumberJAX(curves, downsample=2),
    )


def test_public_curve_surface_distance_jax_wrapper_matches_cpu_value_and_gradient():
    surface = SurfaceRZFourier(
        nfp=1,
        mpol=1,
        ntor=1,
        quadpoints_phi=np.linspace(0.0, 1.0, 10, endpoint=False),
        quadpoints_theta=np.linspace(0.0, 1.0, 10, endpoint=False),
    )
    surface.set("rc(0,0)", 1.0)
    surface.set("rc(1,0)", 0.2)
    surface.set("zs(1,0)", 0.2)
    curve = _build_nonplanar_curve()

    _assert_objective_matches_cpu(
        CurveSurfaceDistance([curve], surface, minimum_distance=0.8),
        CurveSurfaceDistanceJAX([curve], surface, minimum_distance=0.8),
    )


def test_public_distance_jax_values_compose_at_the_host_boundary():
    curve1 = _build_offset_nonplanar_curve(0.0)
    curve2 = _build_offset_nonplanar_curve(0.3)
    surface = SurfaceRZFourier(
        nfp=1,
        mpol=1,
        ntor=1,
        quadpoints_phi=np.linspace(0.0, 1.0, 10, endpoint=False),
        quadpoints_theta=np.linspace(0.0, 1.0, 10, endpoint=False),
    )
    surface.set("rc(0,0)", 1.0)
    surface.set("rc(1,0)", 0.2)
    surface.set("zs(1,0)", 0.2)
    objectives = (
        CurveCurveDistanceJAX(
            [curve1, curve2],
            minimum_distance=0.75,
            num_basecurves=2,
        ),
        CurveSurfaceDistanceJAX([curve1], surface, minimum_distance=0.8),
    )

    for objective in objectives:
        value = objective.J()
        scaled_value = (3.0 * objective).J()

        assert isinstance(value, float)
        np.testing.assert_allclose(scaled_value, 3.0 * value, rtol=0.0, atol=0.0)
