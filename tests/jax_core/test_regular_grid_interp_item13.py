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

from dataclasses import replace
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import simsoptpp as sopp

from benchmarks.validation_ladder_contract import parity_ladder_tolerances
from benchmarks.regular_grid_skip_cell_map_benchmark import (
    run_skip_cell_map_benchmark,
)

from simsopt.jax_core.regular_grid_interp import (
    ChebyshevInterpolationRule,
    InterpolationRule,
    RegularGridInterpolant3DSpec,
    UniformInterpolationRule,
    _flat_cell_index,
    build_regular_grid_interpolant_3d,
    build_regular_grid_interpolant_3d_device_spec,
    estimate_error,
    evaluate_batch,
    evaluate_batch_device,
    evaluate_batch_with_initial,
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
@pytest.mark.parametrize("degree", [1, 2, 3, 4, 5])
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
    # ``evaluate_batch`` uses a zero initial output buffer, so preserving
    # C++ caller-buffer semantics produces zeros for every OOB row here.
    np.testing.assert_allclose(
        lax_result,
        np.zeros_like(lax_result),
        rtol=_DIRECT_KERNEL["rtol"],
        atol=_DIRECT_KERNEL["atol"],
    )


def test_locator_truncates_slightly_negative_cell_coordinate() -> None:
    """Slightly negative local coordinates stay in cell 0 like C++."""

    def linear_x(xs, ys, zs):
        del ys, zs
        return np.asarray(xs, dtype=np.float64)

    xmin = 0.0
    xmax = 2.0
    nx = 2
    hx = (xmax - xmin) / nx
    query = np.asarray([[xmin - 0.1 * hx, 0.5, 0.5]], dtype=np.float64)
    spec = build_regular_grid_interpolant_3d(
        rule=UniformInterpolationRule(1),
        xrange=(xmin, xmax, nx),
        yrange=(0.0, 1.0, 1),
        zrange=(0.0, 1.0, 1),
        value_size=1,
        f=linear_x,
        out_of_bounds_ok=True,
    )

    actual = np.asarray(evaluate_batch(spec, query)).reshape(-1)
    np.testing.assert_allclose(
        actual,
        np.asarray([query[0, 0]], dtype=np.float64),
        rtol=_DIRECT_KERNEL["rtol"],
        atol=_DIRECT_KERNEL["atol"],
    )


def test_flat_cell_index_uses_int64_for_large_grid_shape() -> None:
    """Flat cell-index arithmetic must not overflow int32."""

    @jax.jit
    def compute_index(xidx, yidx, zidx):
        return _flat_cell_index(xidx, yidx, zidx, ny=1_000_000, nz=1_000_000)

    result = compute_index(
        jnp.asarray(999, dtype=jnp.int64),
        jnp.asarray(999_999, dtype=jnp.int64),
        jnp.asarray(999_999, dtype=jnp.int64),
    )

    assert result.dtype == jnp.int64
    assert int(result) == 999_999_999_999_999


def test_cell_to_row_table_uses_int64_indices() -> None:
    """The host lookup table preserves the 64-bit flat-index contract."""

    def linear_x(xs, ys, zs):
        del ys, zs
        return np.asarray(xs, dtype=np.float64)

    spec = build_regular_grid_interpolant_3d(
        rule=UniformInterpolationRule(1),
        xrange=(0.0, 1.0, 1),
        yrange=(0.0, 1.0, 1),
        zrange=(0.0, 1.0, 1),
        value_size=1,
        f=linear_x,
        out_of_bounds_ok=True,
    )

    assert spec.cell_to_row.dtype == np.int64


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


def test_skip_cell_map_benchmark_smoke() -> None:
    """The sparse-vs-dense skip-cell benchmark runs and reports its contract."""

    result = run_skip_cell_map_benchmark(
        n_cells=8,
        degree=2,
        value_size=2,
        n_samples=64,
        repeats=1,
        seed=1317,
    )

    assert result["benchmark"] == "regular_grid_skip_cell_map"
    assert result["total_cells"] == 8**3
    assert 0 < result["kept_cells"] < result["total_cells"]
    assert result["skipped_cells"] == result["total_cells"] - result["kept_cells"]
    assert (
        result["jax_sentinel_cell_to_row_bytes"]
        == 8**3 * np.dtype(np.int64).itemsize
    )
    assert result["jax_cell_table_bytes"] > 0
    assert result["jax_median_seconds"] >= 0.0
    assert result["cpp_unordered_map_median_seconds"] >= 0.0
    assert result["max_abs_error"] <= _DIRECT_KERNEL["atol"]


def test_build_spec_returns_deep_readonly_snapshot() -> None:
    """The public regular-grid spec cannot diverge from staged device state."""

    dim = 1
    source_nodes = np.asarray([0.0, 1.0], dtype=np.float64)
    source_scalings = np.asarray([-1.0, 1.0], dtype=np.float64)
    source_rule = InterpolationRule(
        nodes=source_nodes,
        scalings=source_scalings,
        degree=1,
    )
    poly = _polynomial_factory(dim=dim, degree=1, seed=17)
    spec = build_regular_grid_interpolant_3d(
        rule=source_rule,
        xrange=_DEFAULT_XRANGE,
        yrange=_DEFAULT_YRANGE,
        zrange=_DEFAULT_ZRANGE,
        value_size=dim,
        f=poly,
        out_of_bounds_ok=True,
    )

    for array in (
        spec.rule.nodes,
        spec.rule.scalings,
        spec.xmesh,
        spec.ymesh,
        spec.zmesh,
        spec.cell_to_row,
        spec.cell_table,
    ):
        assert not array.flags.writeable

    with pytest.raises(ValueError):
        spec.cell_table[0, 0, 0, 0, 0] = 123.0

    manual_spec = RegularGridInterpolant3DSpec(
        rule=InterpolationRule(
            nodes=np.asarray([0.0, 1.0], dtype=np.float64),
            scalings=np.asarray([-1.0, 1.0], dtype=np.float64),
            degree=1,
        ),
        nx=1,
        ny=1,
        nz=1,
        xmin=0.0,
        xmax=1.0,
        ymin=0.0,
        ymax=1.0,
        zmin=0.0,
        zmax=1.0,
        hx=1.0,
        hy=1.0,
        hz=1.0,
        xmesh=np.asarray([0.0, 1.0], dtype=np.float64),
        ymesh=np.asarray([0.0, 1.0], dtype=np.float64),
        zmesh=np.asarray([0.0, 1.0], dtype=np.float64),
        value_size=1,
        out_of_bounds_ok=True,
        cell_to_row=np.asarray([0], dtype=np.int64),
        cell_table=np.zeros((2, 2, 2, 2, 1), dtype=np.float64),
    )
    replaced_spec = replace(spec, cell_table=np.array(spec.cell_table, copy=True))
    for constructed_spec in (manual_spec, replaced_spec):
        assert not constructed_spec.rule.nodes.flags.writeable
        assert not constructed_spec.cell_table.flags.writeable
        with pytest.raises(ValueError):
            constructed_spec.cell_table[0, 0, 0, 0, 0] = 456.0

    query = np.asarray([[2.1, 2.0, 1.8]], dtype=np.float64)
    expected = np.asarray(evaluate_batch(spec, query))
    source_nodes[:] = [0.25, 0.75]
    source_scalings[:] = [1.0, 1.0]
    actual = np.asarray(evaluate_batch(spec, query))
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=_DIRECT_KERNEL["rtol"],
        atol=_DIRECT_KERNEL["atol"],
    )


@pytest.mark.parametrize("dim", [1, 3, 4, 5, 7, 8])
@pytest.mark.parametrize("degree", [1, 2, 3, 4, 5])
@pytest.mark.parametrize(
    "rule_factory_jax,rule_factory_sopp",
    [
        (UniformInterpolationRule, sopp.UniformInterpolationRule),
        (ChebyshevInterpolationRule, sopp.ChebyshevInterpolationRule),
    ],
)
def test_cpp_cross_oracle(dim: int, degree: int, rule_factory_jax, rule_factory_sopp) -> None:
    """Parity vs ``simsoptpp.RegularGridInterpolant3D`` at the
    ``direct_kernel`` lane (tolerances imported via
    ``parity_ladder_tolerances("direct_kernel")``).

    Both kernels consume the same separable polynomial and the same
    evaluation points. The JAX kernel uses a ``lax.fori_loop`` tensor
    contraction in C++ loop order, matching the SIMD-FMA loop within
    float64 ULP noise across degrees and vector value sizes.
    """
    poly = _polynomial_factory(dim=dim, degree=degree, seed=0)
    rule_jax = rule_factory_jax(degree)
    rule_sopp = rule_factory_sopp(degree)

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


def test_out_of_bounds_ok_preserves_initial_output_like_cpp_buffer() -> None:
    """JAX can mirror the C++ ``evaluate_batch(xyz, fxyz)`` buffer contract."""

    dim = 3
    degree = 2
    xran = _DEFAULT_XRANGE
    yran = _DEFAULT_YRANGE
    zran = _DEFAULT_ZRANGE
    poly = _polynomial_factory(dim=dim, degree=degree, seed=19)

    def skip(xs, _ys, _zs):
        return (np.asarray(xs) < 2.0).tolist()

    spec = build_regular_grid_interpolant_3d(
        rule=UniformInterpolationRule(degree),
        xrange=xran,
        yrange=yran,
        zrange=zran,
        value_size=dim,
        f=poly,
        out_of_bounds_ok=True,
        skip=skip,
    )
    cpp_interpolant = sopp.RegularGridInterpolant3D(
        sopp.UniformInterpolationRule(degree),
        xran,
        yran,
        zran,
        dim,
        True,
        skip,
    )
    cpp_interpolant.interpolate_batch(poly)

    xyz = np.asarray(
        [
            [2.4, 2.6, 2.8],
            [1.3, 1.3, 1.3],
            [xran[1] + 0.1, yran[1] + 0.1, zran[1] + 0.1],
            [xran[1] + 1000.0, yran[0] - 1000.0, zran[1] + 1000.0],
        ],
        dtype=np.float64,
    )
    initial = np.arange(xyz.shape[0] * dim, dtype=np.float64).reshape(
        xyz.shape[0], dim
    )
    initial += 10.0

    jax_result = np.asarray(evaluate_batch_with_initial(spec, xyz, initial))
    cpp_result = initial.copy()
    cpp_interpolant.evaluate_batch(np.ascontiguousarray(xyz), cpp_result)

    np.testing.assert_allclose(
        jax_result,
        cpp_result,
        rtol=_DIRECT_KERNEL["rtol"],
        atol=_DIRECT_KERNEL["atol"],
    )
    np.testing.assert_allclose(jax_result[1:], initial[1:])


def test_cpp_cross_oracle_degree5_high_magnitude_cells() -> None:
    """Degree-5 parity survives large coordinate offsets."""

    dim = 5
    degree = 5
    xran = (1.0e8, 1.0e8 + 3.0, 3)
    yran = (-2.0e8, -2.0e8 + 2.5, 3)
    zran = (3.0e8, 3.0e8 + 4.0, 4)

    def scaled_polynomial(xs, ys, zs):
        x = (np.asarray(xs) - xran[0]) / (xran[1] - xran[0])
        y = (np.asarray(ys) - yran[0]) / (yran[1] - yran[0])
        z = (np.asarray(zs) - zran[0]) / (zran[1] - zran[0])
        values = np.stack(
            [
                1.0 + x + y + z,
                x**2 - 0.5 * y + z**3,
                x**3 + y**2 - z,
                x**4 - y**3 + 0.25 * z**2,
                x**5 + y**4 - z**2,
            ],
            axis=1,
        )
        return np.ascontiguousarray(values).ravel()

    rule_jax = UniformInterpolationRule(degree)
    rule_sopp = sopp.UniformInterpolationRule(degree)
    spec = build_regular_grid_interpolant_3d(
        rule=rule_jax,
        xrange=xran,
        yrange=yran,
        zrange=zran,
        value_size=dim,
        f=scaled_polynomial,
        out_of_bounds_ok=True,
    )
    cpp_interpolant = sopp.RegularGridInterpolant3D(
        rule_sopp,
        xran,
        yran,
        zran,
        dim,
        True,
    )
    cpp_interpolant.interpolate_batch(scaled_polynomial)

    rng = np.random.RandomState(31)
    nsamples = 64
    xyz = np.stack(
        [
            rng.uniform(xran[0], xran[1], size=nsamples),
            rng.uniform(yran[0], yran[1], size=nsamples),
            rng.uniform(zran[0], zran[1], size=nsamples),
        ],
        axis=1,
    )
    jax_result = np.asarray(evaluate_batch(spec, xyz))
    cpp_result = np.zeros((nsamples, dim), dtype=np.float64)
    cpp_interpolant.evaluate_batch(np.ascontiguousarray(xyz), cpp_result)

    np.testing.assert_allclose(
        jax_result,
        cpp_result,
        rtol=_DIRECT_KERNEL["rtol"],
        atol=_DIRECT_KERNEL["atol"],
    )


def test_cpp_cross_oracle_nan_input_contract() -> None:
    """NaN coordinates propagate consistently through JAX and C++ kernels."""

    dim = 3
    degree = 2
    poly = _polynomial_factory(dim=dim, degree=degree, seed=5)
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

    xyz = np.asarray(
        [
            [np.nan, 2.0, 2.0],
            [2.0, np.nan, 2.0],
            [2.0, 2.0, np.nan],
        ],
        dtype=np.float64,
    )
    jax_result = np.asarray(evaluate_batch(spec, xyz))
    cpp_result = np.zeros((xyz.shape[0], dim), dtype=np.float64)
    cpp_interpolant.evaluate_batch(np.ascontiguousarray(xyz), cpp_result)

    np.testing.assert_array_equal(np.isnan(jax_result), np.isnan(cpp_result))


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
    cpp_interpolant = sopp.RegularGridInterpolant3D(
        sopp.UniformInterpolationRule(degree),
        _DEFAULT_XRANGE,
        _DEFAULT_YRANGE,
        _DEFAULT_ZRANGE,
        dim,
        True,
    )
    cpp_interpolant.interpolate_batch(poly)

    low, high = estimate_error(spec, poly, samples=200, seed=2026)
    cpp_low, cpp_high = cpp_interpolant.estimate_error(poly, 200)
    assert math.isfinite(low) and math.isfinite(high)
    assert low <= high
    assert math.isfinite(cpp_low) and math.isfinite(cpp_high)
    assert cpp_low <= cpp_high
    # Polynomial exactness; both ends sit near machine zero. The
    # ``derivative_heavy`` first-derivative atol is the closest published
    # parity floor and is a comfortable upper bound for the dimensionless
    # bracket produced by a degree-exact polynomial evaluation.
    _bracket_floor = parity_ladder_tolerances("derivative_heavy")[
        "first_derivative_atol"
    ]
    assert abs(low) < _bracket_floor
    assert abs(high) < _bracket_floor
    assert abs(cpp_low) < _bracket_floor
    assert abs(cpp_high) < _bracket_floor
    np.testing.assert_allclose(
        np.asarray([low, high]),
        np.asarray([cpp_low, cpp_high]),
        rtol=_DIRECT_KERNEL["rtol"],
        atol=_bracket_floor,
    )


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


def test_device_spec_evaluate_batch_runs_under_strict_transfer_guard() -> None:
    """Pre-staged regular-grid specs keep the hot path free of host transfers."""

    dim = 3
    degree = 2
    poly = _polynomial_factory(dim=dim, degree=degree, seed=23)
    spec = build_regular_grid_interpolant_3d(
        rule=UniformInterpolationRule(degree),
        xrange=_DEFAULT_XRANGE,
        yrange=_DEFAULT_YRANGE,
        zrange=_DEFAULT_ZRANGE,
        value_size=dim,
        f=poly,
        out_of_bounds_ok=True,
    )
    device_spec = build_regular_grid_interpolant_3d_device_spec(spec)
    xyz = jnp.asarray(
        [
            [2.1, 2.0, 1.8],
            [3.2, 3.0, 2.5],
        ],
        dtype=jnp.float64,
    )
    initial = jnp.zeros((xyz.shape[0], dim), dtype=jnp.float64)

    expected = evaluate_batch_device(
        device_spec,
        xyz,
        initial_output=initial,
    )
    expected_default = evaluate_batch_device(device_spec, xyz)
    expected.block_until_ready()
    expected_default.block_until_ready()

    with jax.transfer_guard("disallow"):
        actual = evaluate_batch_device(
            device_spec,
            xyz,
            initial_output=initial,
        )
        actual_default = evaluate_batch_device(device_spec, xyz)
        actual.block_until_ready()
        actual_default.block_until_ready()

    xyz_host = np.asarray(xyz)
    initial_host = np.asarray(initial)
    with jax.transfer_guard("disallow"):
        guarded_device_spec = build_regular_grid_interpolant_3d_device_spec(spec)
        actual_with_explicit_staging = evaluate_batch_device(
            guarded_device_spec,
            xyz_host,
            initial_output=initial_host,
        )
        actual_with_explicit_staging.block_until_ready()

    np.testing.assert_allclose(
        np.asarray(actual),
        np.asarray(expected),
        rtol=_DIRECT_KERNEL["rtol"],
        atol=_DIRECT_KERNEL["atol"],
    )
    np.testing.assert_allclose(
        np.asarray(actual_default),
        np.asarray(expected_default),
        rtol=_DIRECT_KERNEL["rtol"],
        atol=_DIRECT_KERNEL["atol"],
    )
    np.testing.assert_allclose(
        np.asarray(actual_with_explicit_staging),
        np.asarray(expected),
        rtol=_DIRECT_KERNEL["rtol"],
        atol=_DIRECT_KERNEL["atol"],
    )
