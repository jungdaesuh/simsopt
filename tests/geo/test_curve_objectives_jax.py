"""Wave 4 closeout: pure JAX curve-geometry objective mirrors."""

from __future__ import annotations

from jax_test_support import (
    fixture_jax_runtime_guard,  # noqa: F401
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

from simsopt_jax.parity_tolerances import parity_ladder_tolerances
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
    ArclengthVariationJAX,
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
from simsopt_jax.backend import (
    get_pairwise_penalty_chunk_size,
    invalidate_backend_cache,
)
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
from simsopt_jax.core.curve_geometry import (
    _curve_geometry_with_third_derivative_from_dofs,
)
from simsopt_jax.core import curve_kernels as curve_kernels_module
from simsopt_jax.core._pairwise_reductions import _use_dense_pairwise_path
from simsopt_jax.core.curve_kernels import curve_surface_distance_penalty_pure
from simsopt_jax.core.framedcurve import frenet_frame
from simsopt_jax_adapters.geo import curve_objectives as curve_objectives_module
from simsopt_jax_adapters.geo.curve_specs import curve_spec_from_adapter_curve

from .pairwise_test_helpers import set_pairwise_penalty_chunk_size


_DIRECT_KERNEL = parity_ladder_tolerances("direct_kernel")
_FD_GRADIENT = parity_ladder_tolerances("fd-gradient")
_RTOL = _DIRECT_KERNEL["rtol"]
_ATOL = _DIRECT_KERNEL["atol"]


def test_geo_pairwise_module_reexports_core_owner() -> None:
    from simsopt_jax.core import _pairwise_reductions as core_pairwise
    from simsopt_jax.geo import _pairwise_reductions as geo_pairwise

    assert (
        geo_pairwise.pairwise_thresholded_mean_square_distance_pure
        is core_pairwise.pairwise_thresholded_mean_square_distance_pure
    )


def test_use_dense_pairwise_path_pins_the_chunking_boundary() -> None:
    assert _use_dense_pairwise_path(64, 64, 0), (
        "chunk_size=0 disables chunking, so even large point clouds stay dense"
    )
    assert _use_dense_pairwise_path(4, 4, 4), (
        "counts equal to chunk_size fill exactly one block: the boundary is "
        "inclusive, so the dense path must still be selected"
    )
    assert _use_dense_pairwise_path(4, 3, 4), (
        "counts below chunk_size fit one block and must stay dense"
    )
    assert not _use_dense_pairwise_path(5, 4, 4), (
        "a row count one past chunk_size must select the chunked path"
    )
    assert not _use_dense_pairwise_path(4, 5, 4), (
        "a column count one past chunk_size must select the chunked path"
    )
    assert not _use_dense_pairwise_path(5, 5, 4), (
        "both counts past chunk_size must select the chunked path"
    )
    assert _use_dense_pairwise_path(1, 1, 1), (
        "the smallest enabled chunk_size still sweeps a single-point pair densely"
    )


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
        np.asarray(
            curve_dtorsion_by_dcoeff_from_dofs(spec, spec.dofs), dtype=np.float64
        ),
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
        lambda curve: ArclengthVariationJAX(curve),
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
    _assert_objective_matches_cpu(
        ArclengthVariation(curve),
        ArclengthVariationJAX(curve),
    )
    _assert_objective_matches_cpu(
        ArclengthVariation(curve, nintervals="partial"),
        ArclengthVariationJAX(curve, nintervals="partial"),
    )
    _assert_objective_matches_cpu(
        ArclengthVariation(curve, nintervals=2),
        ArclengthVariationJAX(curve, nintervals=2),
    )


def test_core_curve_surface_distance_dense_and_chunked_value_gradients_match(
    monkeypatch,
):
    curve_gamma = jnp.asarray(
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=jnp.float64,
    )
    curve_gammadash = jnp.asarray(
        [[1.0, 0.0, 0.0], [1.0, 0.2, 0.0], [1.0, 0.0, 0.1]],
        dtype=jnp.float64,
    )
    surface_gamma = jnp.asarray(
        [[0.0, 0.0, 0.02], [0.5, 0.0, 0.03], [1.0, 0.0, 0.04]],
        dtype=jnp.float64,
    )
    surface_normal = jnp.asarray(
        [[0.0, 0.0, 1.0], [0.0, 0.1, 1.0], [0.1, 0.0, 1.0]],
        dtype=jnp.float64,
    )
    minimum_distance = 0.1

    def evaluate(curve_points):
        return curve_surface_distance_penalty_pure(
            curve_points,
            curve_gammadash,
            surface_gamma,
            surface_normal,
            minimum_distance,
        )

    def fail_pairwise_distances(current_curve_gamma, current_surface_gamma):
        raise AssertionError(
            "chunked curve-surface penalty must not materialize the dense "
            "distance matrix"
        )

    try:
        set_pairwise_penalty_chunk_size(monkeypatch, 0)
        dense_value, dense_grad = jax.value_and_grad(evaluate)(curve_gamma)

        set_pairwise_penalty_chunk_size(monkeypatch, 2)
        assert get_pairwise_penalty_chunk_size() == 2
        monkeypatch.setattr(
            curve_kernels_module,
            "_pairwise_distances",
            fail_pairwise_distances,
        )
        chunked_value, chunked_grad = jax.value_and_grad(evaluate)(curve_gamma)
    finally:
        monkeypatch.delenv("SIMSOPT_JAX_PENALTY_POINT_CHUNK_SIZE", raising=False)
        invalidate_backend_cache()

    np.testing.assert_allclose(dense_value, chunked_value, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(dense_grad, chunked_grad, rtol=1e-12, atol=1e-12)


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


def _mixed_quadrature_curves():
    """Three overlapping and two far curves with two quadrature counts."""
    curves = []
    for x_offset, quadpoints in (
        (0.0, 24),
        (0.3, 16),
        (0.6, 24),
        (40.0, 16),
        (80.0, 24),
    ):
        curve = _build_nonplanar_curve(quadpoints)
        curve.set("xc(0)", x_offset)
        curves.append(curve)
    return curves


def _record_sliced_pair_sweeps(monkeypatch):
    """Record the ``batch_size`` of every ``lax.map`` traced from now on."""
    sweeps = []
    lax_map = jax.lax.map

    def recording_map(function, inputs, *, batch_size=None):
        sweeps.append(batch_size)
        return lax_map(function, inputs, batch_size=batch_size)

    monkeypatch.setattr(jax.lax, "map", recording_map)
    return sweeps


def _per_pair_curve_distance_reference(
    curves,
    minimum_distance,
    pair_kernel,
    num_basecurves=None,
    downsample=1,
):
    """Value and derivative of a per-pair loop over ``pair_kernel`` (the oracle)."""
    num_basecurves = num_basecurves or len(curves)
    pair_gradient = jax.jit(jax.grad(pair_kernel, argnums=(0, 1, 2, 3)))
    samples = [
        (
            jnp.asarray(curve.gamma()[::downsample], dtype=jnp.float64),
            jnp.asarray(curve.gammadash()[::downsample], dtype=jnp.float64),
        )
        for curve in curves
    ]
    dgammas = [np.zeros_like(curve.gamma()) for curve in curves]
    dgammadashes = [np.zeros_like(curve.gammadash()) for curve in curves]
    value = 0.0
    for i in range(len(curves)):
        for j in range(min(i, num_basecurves)):
            pair_arguments = (*samples[i], *samples[j], minimum_distance)
            value += float(pair_kernel(*pair_arguments))
            grads = [np.asarray(g) for g in pair_gradient(*pair_arguments)]
            dgammas[i][::downsample] += grads[0]
            dgammadashes[i][::downsample] += grads[1]
            dgammas[j][::downsample] += grads[2]
            dgammadashes[j][::downsample] += grads[3]
    derivative = sum(
        curve.dgamma_by_dcoeff_vjp(dgamma) + curve.dgammadash_by_dcoeff_vjp(dgammadash)
        for curve, dgamma, dgammadash in zip(curves, dgammas, dgammadashes)
    )
    return value, derivative


def _assert_batched_matches_per_pair(objective, reference):
    reference_value, reference_derivative = reference
    reference_gradient = np.asarray(reference_derivative(objective), dtype=np.float64)
    gradient = np.asarray(objective.dJ(), dtype=np.float64)
    np.testing.assert_allclose(
        float(objective.J()), reference_value, rtol=1e-12, atol=0.0
    )
    np.testing.assert_allclose(
        gradient,
        reference_gradient,
        rtol=1e-12,
        atol=1e-12 * float(np.max(np.abs(reference_gradient))),
    )


@pytest.mark.parametrize(
    ("num_basecurves", "downsample"),
    [(None, 1), (2, 1), (None, 2)],
)
def test_batched_curve_curve_distance_matches_per_pair_loop(
    num_basecurves,
    downsample,
):
    curves = _mixed_quadrature_curves()
    objective = CurveCurveDistanceJAX(
        curves,
        minimum_distance=0.75,
        num_basecurves=num_basecurves,
        downsample=downsample,
    )
    reference = _per_pair_curve_distance_reference(
        curves,
        0.75,
        cc_distance_pure,
        num_basecurves=num_basecurves,
        downsample=downsample,
    )

    assert reference[0] > 0.0
    _assert_batched_matches_per_pair(objective, reference)


def test_batched_curve_curve_distance_ignores_pairs_beyond_the_threshold():
    curves = _mixed_quadrature_curves()
    near_curves = curves[:3]
    near = CurveCurveDistanceJAX(near_curves, minimum_distance=0.75)
    everything = CurveCurveDistanceJAX(curves, minimum_distance=0.75)

    np.testing.assert_allclose(everything.J(), near.J(), rtol=1e-12, atol=0.0)
    near_gradient = everything.dJ(partials=True)
    for curve in near_curves:
        np.testing.assert_allclose(
            near_gradient(curve),
            near.dJ(partials=True)(curve),
            rtol=1e-12,
            atol=1e-12 * float(np.max(np.abs(near.dJ()))),
        )
    for curve in curves[3:]:
        assert not np.any(near_gradient(curve))


def test_batched_curve_curve_distance_barrier_matches_per_pair_loop():
    curves = _mixed_quadrature_curves()
    sampled_min_distance = min(
        float(np.min(np.linalg.norm(first[:, None, :] - second[None, :, :], axis=-1)))
        for index, first in enumerate(curve.gamma() for curve in curves)
        for second in (curve.gamma() for curve in curves[:index])
    )
    threshold = 0.5 * sampled_min_distance
    objective = CurveCurveDistanceBarrierJAX(curves, minimum_distance=threshold)

    _assert_batched_matches_per_pair(
        objective,
        _per_pair_curve_distance_reference(
            curves,
            threshold,
            cc_distance_barrier_pure,
        ),
    )


def test_batched_curve_curve_distance_retraces_when_the_chunk_size_changes(
    monkeypatch,
):
    curves = _mixed_quadrature_curves()[:3]
    objective = CurveCurveDistanceJAX(curves, minimum_distance=0.75)
    chunked_traces = []
    chunk_rows = curve_kernels_module._chunk_rows

    def counting_chunk_rows(*arguments, **keywords):
        chunked_traces.append(arguments[1])
        return chunk_rows(*arguments, **keywords)

    def fail_pairwise_distances(gamma1, gamma2):
        raise AssertionError(
            "a chunk size below the sample count must trace the chunked sweep"
        )

    def compiled_programs():
        return (
            curve_objectives_module._curve_pair_penalty._cache_size(),
            curve_objectives_module._curve_pair_penalty_value_and_grad._cache_size(),
        )

    try:
        monkeypatch.setenv("SIMSOPT_JAX_PENALTY_POINT_CHUNK_SIZE", "0")
        invalidate_backend_cache()
        dense_value = objective.J()
        dense_gradient = np.asarray(objective.dJ(), dtype=np.float64)
        dense_programs = compiled_programs()

        # No jax.clear_caches(): the jitted program must key on the chunk size.
        monkeypatch.setenv("SIMSOPT_JAX_PENALTY_POINT_CHUNK_SIZE", "5")
        invalidate_backend_cache()
        monkeypatch.setattr(
            curve_kernels_module,
            "_pairwise_distances",
            fail_pairwise_distances,
        )
        monkeypatch.setattr(curve_kernels_module, "_chunk_rows", counting_chunk_rows)
        chunked_value = objective.J()
        value_traces = len(chunked_traces)
        chunked_gradient = np.asarray(objective.dJ(), dtype=np.float64)
        chunked_programs = compiled_programs()
    finally:
        monkeypatch.delenv("SIMSOPT_JAX_PENALTY_POINT_CHUNK_SIZE", raising=False)
        invalidate_backend_cache()

    assert value_traces > 0, "J replayed a cached dense program"
    assert len(chunked_traces) > value_traces, "dJ replayed a cached dense program"
    assert set(chunked_traces) == {5}
    assert chunked_programs == tuple(count + 1 for count in dense_programs)
    np.testing.assert_allclose(chunked_value, dense_value, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(
        chunked_gradient,
        dense_gradient,
        rtol=1e-12,
        atol=1e-12 * float(np.max(np.abs(dense_gradient))),
    )


@pytest.mark.parametrize(
    ("objective_class", "minimum_distance"),
    [(CurveCurveDistanceJAX, 0.75), (CurveCurveDistanceBarrierJAX, 0.05)],
)
def test_batched_curve_curve_distance_memory_budget_slices_match_one_vmap(
    monkeypatch,
    objective_class,
    minimum_distance,
):
    curves = _mixed_quadrature_curves()
    objective = objective_class(curves, minimum_distance=minimum_distance)
    unbounded_value = float(objective.J())
    unbounded_gradient = np.asarray(objective.dJ(), dtype=np.float64)

    # Two 24x24-sample pairs per slice: the three-pair (24, 24) batch runs as a
    # full slice plus a remainder; the (16, 24), (24, 16) and (16, 16) batches
    # (three, three and one pairs) still fit one vmap.
    budget = 2 * 24 * 24 * 8 * curve_objectives_module._CURVE_PAIR_SCRATCH_ARRAYS
    monkeypatch.setattr(curve_objectives_module, "_CURVE_PAIR_BATCH_BYTES", budget)
    sliced_sweeps = _record_sliced_pair_sweeps(monkeypatch)
    plan = objective._pair_plan
    counts = tuple(len(members) for members in plan.class_members)
    samples = tuple(
        int(curves[members[0]].gamma().shape[0]) for members in plan.class_members
    )
    batch_sizes = [
        curve_objectives_module._curve_pair_batch_size(
            samples[batch.first_class],
            samples[batch.second_class],
            len(batch.first_rows),
            curve_objectives_module._resolve_pairwise_penalty_chunk_size(),
            8,
            budget,
        )
        for batch in plan.batches
    ]
    assert counts == (3, 2)
    assert any(
        size < len(batch.first_rows) for size, batch in zip(batch_sizes, plan.batches)
    )
    bounded_value = float(objective.J())
    bounded_gradient = np.asarray(objective.dJ(), dtype=np.float64)

    # The patched budget is a new static key, so J and dJ each trace once and
    # each sweeps the (24, 24) batch through lax.map in slices of two.
    assert sliced_sweeps == [2, 2]
    assert np.isfinite(unbounded_value) and unbounded_value > 0.0
    np.testing.assert_allclose(bounded_value, unbounded_value, rtol=1e-15, atol=0.0)
    np.testing.assert_allclose(
        bounded_gradient,
        unbounded_gradient,
        rtol=1e-15,
        atol=1e-15 * float(np.max(np.abs(unbounded_gradient))),
    )


def test_batched_curve_curve_distance_barrier_too_close_matches_per_pair_loop():
    curves = _mixed_quadrature_curves()
    objective = CurveCurveDistanceBarrierJAX(curves, minimum_distance=0.75)
    reference_value, reference_derivative = _per_pair_curve_distance_reference(
        curves,
        0.75,
        cc_distance_barrier_pure,
    )
    reference_gradient = np.asarray(reference_derivative(objective), dtype=np.float64)
    gradient = np.asarray(objective.dJ(), dtype=np.float64)

    assert reference_value == np.inf
    assert float(objective.J()) == np.inf
    np.testing.assert_array_equal(np.isnan(gradient), np.isnan(reference_gradient))
    np.testing.assert_array_equal(np.isinf(gradient), np.isinf(reference_gradient))
    np.testing.assert_array_equal(
        np.sign(gradient[np.isinf(gradient)]),
        np.sign(reference_gradient[np.isinf(reference_gradient)]),
    )
    finite = np.isfinite(reference_gradient)
    np.testing.assert_allclose(
        gradient[finite],
        reference_gradient[finite],
        rtol=1e-12,
        atol=1e-12 * float(np.max(np.abs(reference_gradient[finite]))),
    )


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
