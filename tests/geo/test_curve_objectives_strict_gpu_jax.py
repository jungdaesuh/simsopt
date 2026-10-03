"""Strict-CUDA coverage for public JAX curve-objective boundaries."""

from __future__ import annotations

from jax_test_support import (
    fixture_jax_runtime_guard,  # noqa: F401
    enable_strict_parity_backend,
    parity_default_device,
)

import jax
import numpy as np
import pytest

from simsopt.field.coil import Coil, Current
from simsopt.geo.curvexyzfourier import CurveXYZFourier
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt_jax.core.specs import make_biot_savart_spec
from simsopt_jax_adapters.field.biotsavart_backend import (
    BiotSavartJAX,
    SpecBackedBiotSavartJAX,
)
from simsopt_jax_adapters.geo import curve_objectives as curve_objectives_module
from simsopt_jax_adapters.geo.curve_objectives import (
    ArclengthVariationJAX,
    CurveCurveDistanceBarrierJAX,
    CurveCurveDistanceJAX,
    CurveLengthJAX,
    CurveSurfaceDistanceJAX,
    LpCurveCurvatureJAX,
    MeanSquaredCurvatureJAX,
)


def _build_nonplanar_curve(quadpoints: int = 64) -> CurveXYZFourier:
    curve = CurveXYZFourier(quadpoints, order=3)
    curve.set("xc(1)", 1.0)
    curve.set("ys(1)", 1.0)
    curve.set("xs(2)", 0.04)
    curve.set("yc(2)", -0.03)
    curve.set("zs(2)", 0.12)
    curve.set("zc(3)", -0.02)
    return curve


def _build_offset_nonplanar_curve(
    x_offset: float, quadpoints: int = 64
) -> CurveXYZFourier:
    curve = _build_nonplanar_curve(quadpoints)
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
        (ArclengthVariationJAX(curve1), (curve1,)),
        (CurveLengthJAX(curve1), (curve1,)),
        (LpCurveCurvatureJAX(curve1, p=2, threshold=0.0), (curve1,)),
        (MeanSquaredCurvatureJAX(curve1), (curve1,)),
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
            (curve1, surface),
        ),
    )

    with parity_default_device("gpu"):
        with jax.transfer_guard("disallow"):
            results = tuple(
                (
                    objective.J(),
                    tuple(objective.dJ(partials=True)(owner) for owner in owners),
                )
                for objective, owners in objectives_and_owners
            )

    for value, gradients in results:
        assert np.isfinite(value)
        assert all(np.all(np.isfinite(gradient)) for gradient in gradients)


@pytest.mark.parametrize("pair_slices", [False, True], ids=["one_vmap", "sliced"])
def test_batched_curve_curve_distances_obey_strict_gpu_guard(
    monkeypatch,
    request,
    pair_slices,
) -> None:
    enable_strict_parity_backend(monkeypatch, request, "gpu")
    if pair_slices:
        # Budget of one 64x64-sample pair: the three (32, 64) pairs run through
        # lax.map in slices of two.
        monkeypatch.setattr(
            curve_objectives_module,
            "_CURVE_PAIR_BATCH_BYTES",
            64 * 64 * 8 * curve_objectives_module._CURVE_PAIR_SCRATCH_ARRAYS,
        )
    curves = [
        _build_offset_nonplanar_curve(x_offset, quadpoints)
        for x_offset, quadpoints in ((0.0, 64), (0.3, 32), (0.6, 64), (0.9, 32))
    ]
    objectives = (
        CurveCurveDistanceJAX(curves, minimum_distance=0.75),
        CurveCurveDistanceJAX(curves, minimum_distance=0.75, num_basecurves=2),
        CurveCurveDistanceJAX(curves, minimum_distance=0.75, downsample=2),
        CurveCurveDistanceBarrierJAX(curves, minimum_distance=0.01),
    )
    assert all(len(o._pair_plan.class_members) == 2 for o in objectives)
    sliced_batches = sum(
        curve_objectives_module._curve_pair_batch_size(
            64 if batch.first_class == 0 else 32,
            64 if batch.second_class == 0 else 32,
            len(batch.first_rows),
            curve_objectives_module._resolve_pairwise_penalty_chunk_size(),
            8,
            curve_objectives_module._CURVE_PAIR_BATCH_BYTES,
        )
        < len(batch.first_rows)
        for batch in objectives[0]._pair_plan.batches
    )
    assert (sliced_batches > 0) == pair_slices
    sliced_sweeps = []
    lax_map = jax.lax.map

    def recording_map(function, inputs, *, batch_size=None):
        sliced_sweeps.append(batch_size)
        return lax_map(function, inputs, batch_size=batch_size)

    monkeypatch.setattr(jax.lax, "map", recording_map)

    with parity_default_device("gpu"):
        with jax.transfer_guard("disallow"):
            results = tuple(
                (
                    objective.J(),
                    tuple(objective.dJ(partials=True)(curve) for curve in curves),
                )
                for objective in objectives
            )

    # Sliced: the all-pairs distance and the barrier each hold the three-pair
    # (32, 64) batch, swept in slices of two by J and by dJ; the
    # num_basecurves=2 (two such pairs) and downsample=2 plans fit one vmap.
    assert sliced_sweeps == ([2, 2, 2, 2] if pair_slices else [])
    for value, gradients in results:
        assert np.isfinite(float(value)) and float(value) > 0.0
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
