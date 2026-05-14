"""Wave 4 closeout: pure JAX curve-geometry objective mirrors."""

from __future__ import annotations

import inspect

import jax
import numpy as np

from benchmarks.validation_ladder_contract import parity_ladder_tolerances
from simsopt.geo import curveobjectives as curveobjectives_module
from simsopt.geo.curveobjectives import (
    ArclengthVariation,
    CurveCurveDistance,
    CurveCurveDistanceBarrier,
    CurveSurfaceDistance,
    CurveLength,
    LpCurveCurvature,
    LpCurveCurvatureBarrier,
    LinkingNumber,
    LpCurveTorsion,
    MeanSquaredCurvature,
    Lp_curvature_pure,
    Lp_torsion_pure,
    cc_distance_barrier_pure,
    cc_distance_pure,
    curve_arclengthvariation_pure,
    curvature_barrier_pure,
    curve_length_pure,
    curve_msc_pure,
)
from simsopt.geo.curveobjectives_jax import (
    CurveCurveDistanceBarrierJAX,
    CurveCurveDistanceJAX,
    CurveLengthJAX,
    CurveSurfaceDistanceJAX,
    LpCurveCurvatureBarrierJAX,
    LpCurveCurvatureJAX,
    LinkingNumberJAX,
    MeanSquaredCurvatureJAX,
)
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.geo.curvexyzfourier import CurveXYZFourier
from simsopt.jax_core import (
    curve_geometry_from_spec,
    curve_incremental_arclength_from_dofs,
    curve_incremental_arclength_from_spec,
    curve_kappa_from_dofs,
    curve_kappa_from_spec,
    curve_spec_from_curve,
    curve_torsion_from_dofs,
    curve_torsion_from_spec,
)


_DIRECT_KERNEL = parity_ladder_tolerances("direct_kernel")
_RTOL = _DIRECT_KERNEL["rtol"]
_ATOL = _DIRECT_KERNEL["atol"]


def _build_nonplanar_curve():
    curve = CurveXYZFourier(64, order=3)
    curve.set("xc(1)", 1.0)
    curve.set("ys(1)", 1.0)
    curve.set("xs(2)", 0.04)
    curve.set("yc(2)", -0.03)
    curve.set("zs(2)", 0.12)
    curve.set("zc(3)", -0.02)
    return curve


def _build_offset_nonplanar_curve(x_offset: float):
    curve = _build_nonplanar_curve()
    curve.set("xc(0)", x_offset)
    return curve


def test_curve_geometry_scalar_wrappers_match_cpu_curve_methods_and_jit():
    curve = _build_nonplanar_curve()
    spec = curve_spec_from_curve(curve)

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


def test_representative_curve_objective_mirrors_match_cpu_values():
    curve = _build_nonplanar_curve()
    spec = curve_spec_from_curve(curve)

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


def test_remaining_curve_objective_mirrors_match_cpu_values(monkeypatch):
    monkeypatch.setattr(curveobjectives_module, "is_jax_backend", lambda: True)
    curve1 = _build_offset_nonplanar_curve(0.0)
    curve2 = _build_offset_nonplanar_curve(3.5)

    spec1 = curve_spec_from_curve(curve1)
    spec2 = curve_spec_from_curve(curve2)
    gamma1, gammadash1, _ = curve_geometry_from_spec(spec1)
    gamma2, gammadash2, _ = curve_geometry_from_spec(spec2)
    kappa1 = curve_kappa_from_spec(spec1)

    curvature_threshold = 2.0 * float(np.max(curve1.kappa()))
    curvature_barrier = LpCurveCurvatureBarrier(curve1, curvature_threshold)
    distance = CurveCurveDistance([curve1, curve2], minimum_distance=10.0)
    sampled_min_distance = min(
        np.linalg.norm(first - second)
        for first in np.asarray(curve1.gamma(), dtype=np.float64)
        for second in np.asarray(curve2.gamma(), dtype=np.float64)
    )
    distance_barrier_threshold = 0.5 * sampled_min_distance
    distance_barrier = CurveCurveDistanceBarrier(
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


def test_public_curve_objective_jax_wrappers_match_cpu_values_and_gradients():
    curve = _build_nonplanar_curve()
    curvature_threshold = 2.0 * float(np.max(curve.kappa()))

    _assert_objective_matches_cpu(CurveLength(curve), CurveLengthJAX(curve))
    _assert_objective_matches_cpu(
        LpCurveCurvature(curve, p=2, threshold=0.0),
        LpCurveCurvatureJAX(curve, p=2, threshold=0.0),
    )
    _assert_objective_matches_cpu(
        LpCurveCurvatureBarrier(curve, curvature_threshold),
        LpCurveCurvatureBarrierJAX(curve, curvature_threshold),
    )
    _assert_objective_matches_cpu(
        MeanSquaredCurvature(curve),
        MeanSquaredCurvatureJAX(curve),
    )


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

    barrier_threshold = 0.5 * distance_cpu.shortest_distance()
    _assert_objective_matches_cpu(
        CurveCurveDistanceBarrier([curve1, curve2], barrier_threshold),
        CurveCurveDistanceBarrierJAX([curve1, curve2], barrier_threshold),
    )


def test_curve_distance_jax_wrapper_signatures_match_cpu_contracts():
    assert inspect.signature(CurveCurveDistanceJAX.__init__) == inspect.signature(
        CurveCurveDistance.__init__
    )
    assert inspect.signature(
        CurveCurveDistanceBarrierJAX.__init__
    ) == inspect.signature(CurveCurveDistanceBarrier.__init__)
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
