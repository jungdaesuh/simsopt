"""Focused N2/N7 coverage for Boozer interpolants and regular-grid contraction."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from simsopt.field.boozermagneticfield import BoozerAnalytic
from simsopt.field.boozermagneticfield_jax import InterpolatedBoozerFieldJAX
from simsopt.jax_core import interpolated_boozer_field as ibf
from simsopt.jax_core.regular_grid_interp import (
    UniformInterpolationRule,
    build_regular_grid_interpolant_3d,
    build_regular_grid_interpolant_3d_device_spec,
    evaluate_batch,
    evaluate_batch_device,
)


def _analytic_field() -> BoozerAnalytic:
    return BoozerAnalytic(etabar=0.5, B0=1.0, N=0, G0=1.5, psi0=0.3, iota0=0.4)


def _build_wrapper(*, scalars: tuple[str, ...]) -> InterpolatedBoozerFieldJAX:
    return InterpolatedBoozerFieldJAX(
        _analytic_field(),
        degree=2,
        srange=[0.3, 0.7, 2],
        thetarange=[0.0, np.pi, 2],
        zetarange=[0.0, 2.0 * np.pi, 2],
        extrapolate=True,
        nfp=1,
        stellsym=True,
        scalars=scalars,
    )


def test_interpolated_boozer_scalar_siblings_reuse_device_specs(monkeypatch) -> None:
    calls: list[int] = []
    original_builder = ibf.build_regular_grid_interpolant_3d_device_spec

    def counted_builder(spec):
        calls.append(id(spec))
        return original_builder(spec)

    monkeypatch.setattr(
        ibf,
        "build_regular_grid_interpolant_3d_device_spec",
        counted_builder,
    )
    wrapper = _build_wrapper(scalars=("modB", "K"))
    assert len(calls) == 2

    points = np.asarray(
        [
            [0.4, 0.5, 1.0],
            [0.5, 1.0, 2.0],
        ],
        dtype=np.float64,
    )
    wrapper.set_points(points)
    np.asarray(wrapper.modB())
    np.asarray(wrapper.K())
    wrapper.set_points(points + np.asarray([[0.01, 0.02, 0.03], [0.02, 0.01, 0.04]]))
    np.asarray(wrapper.modB())
    np.asarray(wrapper.K())

    assert len(calls) == 2


def test_interpolated_boozer_lazy_scalar_device_spec_is_cached_once(monkeypatch) -> None:
    calls: list[int] = []
    original_builder = ibf.build_regular_grid_interpolant_3d_device_spec

    def counted_builder(spec):
        calls.append(id(spec))
        return original_builder(spec)

    monkeypatch.setattr(
        ibf,
        "build_regular_grid_interpolant_3d_device_spec",
        counted_builder,
    )
    wrapper = _build_wrapper(scalars=("modB",))
    assert len(calls) == 1

    points = np.asarray([[0.4, 0.5, 1.0]], dtype=np.float64)
    wrapper.set_points(points)
    np.asarray(wrapper.K())
    wrapper.set_points(points + np.asarray([[0.01, 0.02, 0.03]], dtype=np.float64))
    np.asarray(wrapper.K())

    assert len(calls) == 2


def test_interpolated_boozer_cached_device_values_match_existing_path() -> None:
    wrapper = _build_wrapper(scalars=("modB",))
    points = np.asarray(
        [
            [0.4, 0.5, 1.0],
            [0.5, 1.0, 2.0],
        ],
        dtype=np.float64,
    )
    wrapper.set_points(points)

    folded, flipped = ibf.fold_points_for_symmetry(
        jnp.asarray(points, dtype=jnp.float64),
        period=jnp.asarray(wrapper.frozen_state.period, dtype=jnp.float64),
        stellsym=wrapper.frozen_state.stellsym,
    )
    assert not np.any(np.asarray(flipped))
    expected = np.asarray(evaluate_batch(wrapper.frozen_state.specs["modB"], folded))
    actual = np.asarray(wrapper.modB())

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_regular_grid_fused_nonparity_matches_strict_tensor_contract() -> None:
    def polynomial(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        values = np.column_stack(
            [
                1.0 + x + 2.0 * y - 0.5 * z + x * y,
                0.25 + x * z - y * z + x * y * z,
            ]
        )
        return np.ascontiguousarray(values).ravel()

    spec = build_regular_grid_interpolant_3d(
        rule=UniformInterpolationRule(2),
        xrange=(0.0, 1.0, 3),
        yrange=(0.0, 1.0, 2),
        zrange=(0.0, 1.0, 2),
        value_size=2,
        f=polynomial,
        out_of_bounds_ok=True,
    )
    device_spec = build_regular_grid_interpolant_3d_device_spec(spec)
    xyz = jnp.asarray(
        [
            [0.15, 0.25, 0.35],
            [0.45, 0.55, 0.65],
            [0.85, 0.75, 0.15],
        ],
        dtype=jnp.float64,
    )

    strict = evaluate_batch_device(device_spec, xyz, strict_cell_order=True)
    fused = evaluate_batch_device(device_spec, xyz, strict_cell_order=False)

    np.testing.assert_allclose(
        np.asarray(fused),
        np.asarray(strict),
        rtol=1e-12,
        atol=1e-12,
    )
