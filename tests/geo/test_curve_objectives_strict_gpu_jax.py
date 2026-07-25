"""Strict-CUDA coverage for public JAX curve-objective boundaries."""

from __future__ import annotations

from conftest import enable_strict_parity_backend, parity_default_device

import jax
import numpy as np

from simsopt.field.coil import Coil, Current
from simsopt.geo.curvexyzfourier import CurveXYZFourier
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt_jax.core.specs import make_biot_savart_spec
from simsopt_jax_adapters.field.biotsavart_backend import (
    BiotSavartJAX,
    SpecBackedBiotSavartJAX,
)
from simsopt_jax_adapters.geo.curve_objectives import (
    CurveCurveDistanceJAX,
    CurveSurfaceDistanceJAX,
    LpCurveCurvatureJAX,
)


def _build_nonplanar_curve() -> CurveXYZFourier:
    curve = CurveXYZFourier(64, order=3)
    curve.set("xc(1)", 1.0)
    curve.set("ys(1)", 1.0)
    curve.set("xs(2)", 0.04)
    curve.set("yc(2)", -0.03)
    curve.set("zs(2)", 0.12)
    curve.set("zc(3)", -0.02)
    return curve


def _build_offset_nonplanar_curve(x_offset: float) -> CurveXYZFourier:
    curve = _build_nonplanar_curve()
    curve.set("xc(0)", x_offset)
    return curve


def test_public_curve_geometry_values_and_derivatives_obey_strict_gpu_guard(
    monkeypatch,
    request,
) -> None:
    enable_strict_parity_backend(monkeypatch, request, "gpu")
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
    objectives_and_owners = (
        (LpCurveCurvatureJAX(curve1, p=2, threshold=0.0), (curve1,)),
        (
            CurveCurveDistanceJAX(
                [curve1, curve2],
                minimum_distance=0.75,
                num_basecurves=2,
            ),
            (curve1, curve2),
        ),
        (
            CurveSurfaceDistanceJAX(
                [curve1],
                surface,
                minimum_distance=0.8,
            ),
            (curve1,),
        ),
    )

    with parity_default_device("gpu"):
        with jax.transfer_guard("disallow"):
            results = tuple(
                (
                    objective.J(),
                    tuple(
                        objective.dJ(partials=True)(owner) for owner in owners
                    ),
                )
                for objective, owners in objectives_and_owners
            )

    for value, gradients in results:
        assert np.isfinite(value)
        assert all(np.all(np.isfinite(gradient)) for gradient in gradients)


def test_spec_backed_curvature_objective_preserves_public_vjp_contract_on_gpu(
    monkeypatch,
    request,
) -> None:
    enable_strict_parity_backend(monkeypatch, request, "gpu")
    curve = _build_nonplanar_curve()
    field = BiotSavartJAX([Coil(curve, Current(1.0e5))])
    spec = make_biot_savart_spec(
        coil_dof_extraction=field.coil_dof_extraction_spec(),
        coil_dofs=np.asarray(field.x, dtype=np.float64),
    )
    spec_backed_field = SpecBackedBiotSavartJAX(spec)
    spec_backed_curve = spec_backed_field.coils[0].curve
    objective = LpCurveCurvatureJAX(
        spec_backed_curve,
        p=2,
        threshold=0.0,
    )

    with parity_default_device("gpu"):
        with jax.transfer_guard("disallow"):
            value = objective.J()
            gradient = objective.dJ(partials=True)(spec_backed_field)

    assert np.isfinite(value)
    assert np.all(np.isfinite(gradient))
