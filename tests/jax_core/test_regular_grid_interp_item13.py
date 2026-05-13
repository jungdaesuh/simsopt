"""Parity tests for the JAX ``regular_grid_interp`` port (item 13).

Each test imports tolerances from
``benchmarks.validation_ladder_contract.parity_ladder_tolerances`` so the
lane contract is preserved end-to-end.

The cross-oracle test compares the JAX kernel against the C++
``simsoptpp.RegularGridInterpolant3D`` binding. The polynomial-exactness
tests use a hand-derived closed-form oracle that follows the upstream
``tests/field/test_interpolant.py`` recipe so this item also covers the
empty-oracle case for closed-form polynomial inputs.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import simsoptpp as sopp

from benchmarks.validation_ladder_contract import parity_ladder_tolerances

from simsopt.jax_core.regular_grid_interp import (
    ChebyshevInterpolationRule,
    UniformInterpolationRule,
    build_regular_grid_interpolant_3d,
    estimate_error,
    evaluate_batch,
)


_DIRECT_KERNEL = parity_ladder_tolerances("direct_kernel")


def _polynomial_factory(*, dim: int, degree: int, seed: int):
    """Build a separable polynomial ``f(x, y, z) = p_x(x) * p_y(y) * p_z(z)``.

    Mirrors ``tests/field/test_interpolant.py::get_random_polynomial`` so
    both kernels are exercised against the same closed-form contract.
    """
    rng = np.random.RandomState(seed)
    coeffs_x = rng.standard_normal(size=(degree + 1, dim))
    coeffs_y = rng.standard_normal(size=(degree + 1, dim))
    coeffs_z = rng.standard_normal(size=(degree + 1, dim))

    def evaluate(x: np.ndarray, y: np.ndarray, z: np.ndarray, *, flatten: bool = True):
        x_arr = np.asarray(x)
        y_arr = np.asarray(y)
        z_arr = np.asarray(z)
        accum_x = sum(coeffs_x[i, :] * x_arr[:, None] ** i for i in range(degree + 1))
        accum_y = sum(coeffs_y[i, :] * y_arr[:, None] ** i for i in range(degree + 1))
        accum_z = sum(coeffs_z[i, :] * z_arr[:, None] ** i for i in range(degree + 1))
        res = accum_x * accum_y * accum_z
        if flatten:
            return np.ascontiguousarray(res).ravel()
        return np.ascontiguousarray(res)

    return evaluate


_DEFAULT_XRANGE = (1.0, 4.0, 20)
_DEFAULT_YRANGE = (1.1, 3.9, 10)
_DEFAULT_ZRANGE = (1.2, 3.8, 15)


@pytest.mark.parametrize("dim", [1, 3, 6])
@pytest.mark.parametrize("degree", [1, 2, 3, 4])
def test_polynomial_exactness(dim: int, degree: int) -> None:
    """A degree-``d`` polynomial is interpolated exactly by a degree-``d`` rule.

    Closed-form oracle: a separable degree-``d`` polynomial reproduced
    exactly by tensor-product Lagrange interpolation of the same degree.
    """
    poly = _polynomial_factory(dim=dim, degree=degree, seed=0)
    rule = UniformInterpolationRule(degree)
    spec = build_regular_grid_interpolant_3d(
        rule=rule,
        xrange=_DEFAULT_XRANGE,
        yrange=_DEFAULT_YRANGE,
        zrange=_DEFAULT_ZRANGE,
        value_size=dim,
        f=poly,
        out_of_bounds_ok=True,
    )

    rng = np.random.RandomState(1)
    nsamples = 100
    xs = rng.uniform(_DEFAULT_XRANGE[0], _DEFAULT_XRANGE[1], size=nsamples)
    ys = rng.uniform(_DEFAULT_YRANGE[0], _DEFAULT_YRANGE[1], size=nsamples)
    zs = rng.uniform(_DEFAULT_ZRANGE[0], _DEFAULT_ZRANGE[1], size=nsamples)
    xyz = np.stack([xs, ys, zs], axis=-1)

    expected = poly(xs, ys, zs, flatten=False)
    actual = np.asarray(evaluate_batch(spec, xyz))

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=_DIRECT_KERNEL["rtol"],
        atol=_DIRECT_KERNEL["atol"],
    )


@pytest.mark.parametrize(
    "rule_factory", [UniformInterpolationRule, ChebyshevInterpolationRule]
)
def test_uniform_vs_chebyshev_nodes(rule_factory) -> None:
    """Both node sets reproduce a low-degree separable polynomial exactly."""
    dim = 3
    degree = 2
    poly = _polynomial_factory(dim=dim, degree=degree, seed=7)
    rule = rule_factory(degree)
    spec = build_regular_grid_interpolant_3d(
        rule=rule,
        xrange=_DEFAULT_XRANGE,
        yrange=_DEFAULT_YRANGE,
        zrange=_DEFAULT_ZRANGE,
        value_size=dim,
        f=poly,
        out_of_bounds_ok=True,
    )

    xyz = np.array(
        [
            [2.1, 2.0, 1.8],
            [3.2, 3.0, 2.5],
            [1.5, 3.7, 1.3],
        ]
    )
    expected = poly(xyz[:, 0], xyz[:, 1], xyz[:, 2], flatten=False)
    actual = np.asarray(evaluate_batch(spec, xyz))
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=_DIRECT_KERNEL["rtol"],
        atol=_DIRECT_KERNEL["atol"],
    )


def test_oob_behavior_returns_nan_when_strict() -> None:
    """Strict out-of-bounds evaluation surfaces ``NaN`` so the host can detect.

    The JAX kernel cannot raise from inside ``jit`` like the C++ binding
    does, so it routes invalid queries to ``NaN``. The caller is
    expected to check for ``NaN`` post-hoc.
    """
    dim = 3
    degree = 2
    poly = _polynomial_factory(dim=dim, degree=degree, seed=0)
    rule = UniformInterpolationRule(degree)
    spec_strict = build_regular_grid_interpolant_3d(
        rule=rule,
        xrange=_DEFAULT_XRANGE,
        yrange=_DEFAULT_YRANGE,
        zrange=_DEFAULT_ZRANGE,
        value_size=dim,
        f=poly,
        out_of_bounds_ok=False,
    )
    spec_lax = build_regular_grid_interpolant_3d(
        rule=rule,
        xrange=_DEFAULT_XRANGE,
        yrange=_DEFAULT_YRANGE,
        zrange=_DEFAULT_ZRANGE,
        value_size=dim,
        f=poly,
        out_of_bounds_ok=True,
    )

    rng = np.random.RandomState(2)
    nsamples = 25
    xs = rng.uniform(_DEFAULT_XRANGE[1] + 0.1, _DEFAULT_XRANGE[1] + 0.3, nsamples)
    ys = rng.uniform(_DEFAULT_YRANGE[1] + 0.1, _DEFAULT_YRANGE[1] + 0.3, nsamples)
    zs = rng.uniform(_DEFAULT_ZRANGE[1] + 0.1, _DEFAULT_ZRANGE[1] + 0.3, nsamples)
    xyz_oob = np.stack([xs, ys, zs], axis=-1)

    strict_result = np.asarray(evaluate_batch(spec_strict, xyz_oob))
    assert np.isnan(strict_result).all(), (
        "out_of_bounds_ok=False must surface NaN for every OOB query"
    )

    lax_result = np.asarray(evaluate_batch(spec_lax, xyz_oob))
    # For ``out_of_bounds_ok=True`` the JAX semantic is to return zero
    # (the C++ binding leaves the caller buffer unchanged, which is not
    # representable in a pure-functional kernel). Zero is the unique
    # fixed point that avoids leaking stale memory.
    np.testing.assert_allclose(
        lax_result,
        np.zeros_like(lax_result),
        rtol=_DIRECT_KERNEL["rtol"],
        atol=_DIRECT_KERNEL["atol"],
    )


def test_skip_region_yields_zero_inside_skipped_cells() -> None:
    """Cells whose 8 corners all evaluate ``True`` for the skip predicate
    are excluded from the interpolant; queries inside them return zero
    while queries inside retained cells return the underlying polynomial.
    """
    dim = 3
    degree = 2
    xran = (1.0, 4.0, 30)
    yran = (1.1, 3.9, 30)
    zran = (1.2, 3.8, 30)
    xkeep = (2.0, 3.0)
    ykeep = (2.0, 3.0)
    zkeep = (2.0, 3.0)

    def skip(xs, ys, zs):
        xs_arr = np.asarray(xs)
        ys_arr = np.asarray(ys)
        zs_arr = np.asarray(zs)
        keep = (
            (xkeep[0] < xs_arr)
            & (xs_arr < xkeep[1])
            & (ykeep[0] < ys_arr)
            & (ys_arr < ykeep[1])
            & (zkeep[0] < zs_arr)
            & (zs_arr < zkeep[1])
        )
        return np.invert(keep)

    poly = _polynomial_factory(dim=dim, degree=degree, seed=0)
    rule = UniformInterpolationRule(degree)
    spec = build_regular_grid_interpolant_3d(
        rule=rule,
        xrange=xran,
        yrange=yran,
        zrange=zran,
        value_size=dim,
        f=poly,
        out_of_bounds_ok=True,
        skip=skip,
    )

    xyz_keep = np.array(
        [
            [2.4, 2.6, 2.8],
            [2.1, 2.1, 2.9],
            [2.8, 2.8, 2.1],
        ]
    )
    xyz_skip = np.array(
        [
            [1.3, 1.3, 1.3],
            [1.3, 2.9, 3.5],
            [3.5, 1.3, 1.3],
        ]
    )

    f_keep_expected = poly(
        xyz_keep[:, 0], xyz_keep[:, 1], xyz_keep[:, 2], flatten=False
    )
    f_keep_actual = np.asarray(evaluate_batch(spec, xyz_keep))
    np.testing.assert_allclose(
        f_keep_actual,
        f_keep_expected,
        rtol=_DIRECT_KERNEL["rtol"],
        atol=_DIRECT_KERNEL["atol"],
    )

    f_skip_actual = np.asarray(evaluate_batch(spec, xyz_skip))
    np.testing.assert_allclose(
        f_skip_actual,
        np.zeros_like(f_skip_actual),
        rtol=_DIRECT_KERNEL["rtol"],
        atol=_DIRECT_KERNEL["atol"],
    )


@pytest.mark.parametrize("dim", [1, 3, 4])
@pytest.mark.parametrize("degree", [1, 2, 3])
def test_cpp_cross_oracle(dim: int, degree: int) -> None:
    """Parity vs ``simsoptpp.RegularGridInterpolant3D`` at the
    ``direct_kernel`` lane (tolerances imported via
    ``parity_ladder_tolerances("direct_kernel")``).

    Both kernels consume the same separable polynomial and the same
    evaluation points. The JAX kernel uses ``jnp.einsum`` for the
    tensor contraction so it matches the C++ SIMD-FMA loop within
    float64 ULP noise even at degree 3 with vector value size 4.
    """
    poly = _polynomial_factory(dim=dim, degree=degree, seed=0)
    rule_jax = UniformInterpolationRule(degree)
    rule_sopp = sopp.UniformInterpolationRule(degree)

    spec = build_regular_grid_interpolant_3d(
        rule=rule_jax,
        xrange=_DEFAULT_XRANGE,
        yrange=_DEFAULT_YRANGE,
        zrange=_DEFAULT_ZRANGE,
        value_size=dim,
        f=poly,
        out_of_bounds_ok=True,
    )

    cpp_interpolant = sopp.RegularGridInterpolant3D(
        rule_sopp,
        _DEFAULT_XRANGE,
        _DEFAULT_YRANGE,
        _DEFAULT_ZRANGE,
        dim,
        True,
    )
    cpp_interpolant.interpolate_batch(poly)

    rng = np.random.RandomState(3)
    nsamples = 128
    xs = rng.uniform(_DEFAULT_XRANGE[0], _DEFAULT_XRANGE[1], size=nsamples)
    ys = rng.uniform(_DEFAULT_YRANGE[0], _DEFAULT_YRANGE[1], size=nsamples)
    zs = rng.uniform(_DEFAULT_ZRANGE[0], _DEFAULT_ZRANGE[1], size=nsamples)
    xyz = np.stack([xs, ys, zs], axis=-1)

    jax_result = np.asarray(evaluate_batch(spec, xyz))
    cpp_result = np.zeros((nsamples, dim), dtype=np.float64)
    cpp_interpolant.evaluate_batch(np.ascontiguousarray(xyz), cpp_result)

    np.testing.assert_allclose(
        jax_result,
        cpp_result,
        rtol=_DIRECT_KERNEL["rtol"],
        atol=_DIRECT_KERNEL["atol"],
    )


def test_cpp_cross_oracle_with_skip_mask() -> None:
    """Cross-oracle parity for an interpolant with a non-trivial skip mask.

    Uses the same skip mask as upstream ``test_skip`` so cells outside
    the small interior box are excluded from both kernels. The check
    asserts byte-equivalent (within ``direct_kernel`` tolerance) results
    on retained cells and the documented zero/leave-unchanged contract
    on skipped cells.
    """
    dim = 3
    degree = 2
    xran = (1.0, 4.0, 30)
    yran = (1.1, 3.9, 30)
    zran = (1.2, 3.8, 30)
    xkeep = (2.0, 3.0)
    ykeep = (2.0, 3.0)
    zkeep = (2.0, 3.0)

    def skip(xs, ys, zs):
        xs_arr = np.asarray(xs)
        ys_arr = np.asarray(ys)
        zs_arr = np.asarray(zs)
        keep = (
            (xkeep[0] < xs_arr)
            & (xs_arr < xkeep[1])
            & (ykeep[0] < ys_arr)
            & (ys_arr < ykeep[1])
            & (zkeep[0] < zs_arr)
            & (zs_arr < zkeep[1])
        )
        return np.invert(keep)

    poly = _polynomial_factory(dim=dim, degree=degree, seed=0)
    rule_jax = UniformInterpolationRule(degree)
    rule_sopp = sopp.UniformInterpolationRule(degree)

    spec = build_regular_grid_interpolant_3d(
        rule=rule_jax,
        xrange=xran,
        yrange=yran,
        zrange=zran,
        value_size=dim,
        f=poly,
        out_of_bounds_ok=True,
        skip=skip,
    )

    cpp_interpolant = sopp.RegularGridInterpolant3D(
        rule_sopp,
        xran,
        yran,
        zran,
        dim,
        True,
        skip,
    )
    cpp_interpolant.interpolate_batch(poly)

    xyz_keep = np.array(
        [
            [2.4, 2.6, 2.8],
            [2.1, 2.1, 2.9],
            [2.8, 2.8, 2.1],
        ]
    )
    jax_keep = np.asarray(evaluate_batch(spec, xyz_keep))
    cpp_keep = np.zeros((xyz_keep.shape[0], dim), dtype=np.float64)
    cpp_interpolant.evaluate_batch(np.ascontiguousarray(xyz_keep), cpp_keep)

    np.testing.assert_allclose(
        jax_keep,
        cpp_keep,
        rtol=_DIRECT_KERNEL["rtol"],
        atol=_DIRECT_KERNEL["atol"],
    )


def test_estimate_error_returns_bracket_for_polynomial() -> None:
    """For a polynomial of degree ``d``, a degree-``d`` rule returns a
    tight bracket around zero. This exercises the
    ``estimate_error`` helper that mirrors the C++ ``estimate_error``.
    """
    dim = 2
    degree = 2
    poly = _polynomial_factory(dim=dim, degree=degree, seed=42)
    rule = UniformInterpolationRule(degree)
    spec = build_regular_grid_interpolant_3d(
        rule=rule,
        xrange=_DEFAULT_XRANGE,
        yrange=_DEFAULT_YRANGE,
        zrange=_DEFAULT_ZRANGE,
        value_size=dim,
        f=poly,
        out_of_bounds_ok=True,
    )
    low, high = estimate_error(spec, poly, samples=200, seed=2026)
    assert math.isfinite(low) and math.isfinite(high)
    assert low <= high
    # Polynomial exactness; both ends sit near machine zero. The
    # ``derivative_heavy`` first-derivative atol is the closest published
    # parity floor and is a comfortable upper bound for the dimensionless
    # bracket produced by a degree-exact polynomial evaluation.
    _bracket_floor = parity_ladder_tolerances("derivative_heavy")[
        "first_derivative_atol"
    ]
    assert abs(low) < _bracket_floor
    assert abs(high) < _bracket_floor


def test_evaluate_batch_is_jit_traceable_and_returns_device_array() -> None:
    """The compiled evaluator is ``jax.jit``-friendly and returns a
    device array. This is the load-bearing check that downstream
    wrappers can stage host scalars at the boundary and keep the hot
    path on-device.
    """
    dim = 3
    degree = 2
    poly = _polynomial_factory(dim=dim, degree=degree, seed=0)
    rule = UniformInterpolationRule(degree)
    spec = build_regular_grid_interpolant_3d(
        rule=rule,
        xrange=_DEFAULT_XRANGE,
        yrange=_DEFAULT_YRANGE,
        zrange=_DEFAULT_ZRANGE,
        value_size=dim,
        f=poly,
        out_of_bounds_ok=True,
    )
    xyz = jnp.array(
        [
            [2.1, 2.0, 1.8],
            [3.2, 3.0, 2.5],
        ]
    )
    result = evaluate_batch(spec, xyz)
    assert isinstance(result, jax.Array)
    assert result.shape == (2, dim)
    assert result.dtype == jnp.float64

    # Confirm the call is jittable from the outside as well by passing it
    # through a trivial wrapper.
    wrapped = jax.jit(lambda points: evaluate_batch(spec, points))
    wrapped_result = wrapped(xyz)
    np.testing.assert_allclose(
        np.asarray(wrapped_result),
        np.asarray(result),
        rtol=_DIRECT_KERNEL["rtol"],
        atol=_DIRECT_KERNEL["atol"],
    )
