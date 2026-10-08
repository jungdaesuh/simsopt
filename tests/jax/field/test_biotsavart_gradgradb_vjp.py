"""gradgradB VJP against native Hessian finite differences, on CPU."""

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

from dataclasses import replace
from typing import Protocol, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from simsopt._core.derivative import Derivative
from simsopt._core.optimizable import Optimizable
from simsopt.configs import get_ncsx_data
from simsopt.field import BiotSavart, Coil, Current, coils_via_symmetries
from simsopt.geo import create_equally_spaced_curves
from simsopt_jax.backend import get_field_kernel_tuning, set_backend
from simsopt_jax.core import biotsavart as core
from simsopt_jax.core.field import grouped_biot_savart_d2B_by_dXdX_from_inputs
from simsopt_jax_adapters.field.biotsavart_backend import JaxBiotSavart

_DIRECT_RTOL, _DIRECT_ATOL = 1e-10, 1e-12
_FD_RTOL = 1e-7


class _CompiledCache(Protocol):
    """``jax.jit`` wrappers expose their compiled-program count as ``_cache_size``."""

    def _cache_size(self) -> int: ...


def _ncsx_case(case, fixed):
    curves, currents, _ = get_ncsx_data()
    coils = coils_via_symmetries(curves, currents, 3, True)
    if case == "one":
        coils, curves, currents = [coils[0]], curves[:1], currents[:1]
    if fixed == "partial":
        for curve in curves:
            curve.fix(curve.local_dof_names[0])
        currents[0].fix_all()
    elif fixed == "all":
        for owner in [*curves, *currents]:
            owner.fix_all()
    rng = np.random.default_rng(649)
    phi = rng.uniform(0, 2 * np.pi, 5)
    points = np.stack((1.5 * np.cos(phi), 1.5 * np.sin(phi), rng.uniform(-0.1, 0.1, 5)), axis=1)
    seeds = tuple(rng.standard_normal((len(points),) + (3,) * rank) for rank in (1, 2, 3))
    native = BiotSavart(coils).set_points(points)
    field = JaxBiotSavart(coils).set_points(points)
    return field, native, curves, currents, seeds


def _central_difference(native, owner: Optimizable, seed, step):
    original = cast(np.ndarray, owner.local_full_x).copy()
    points = native.get_points_cart().copy()
    gradient = np.empty_like(original)
    for index in range(len(original)):
        direction = np.zeros_like(original)
        direction[index] = step
        owner.local_full_x = original + direction
        # Native fields do not receive recompute notifications from all-fixed graphs.
        native.set_points(points)
        plus = np.sum(native.d2B_by_dXdX() * seed)
        owner.local_full_x = original - direction
        native.set_points(points)
        minus = np.sum(native.d2B_by_dXdX() * seed)
        gradient[index] = (plus - minus) / (2 * step)
    owner.local_full_x = original
    native.set_points(points)
    return gradient


@pytest.mark.parametrize("case", ["one", "ncsx"])
@pytest.mark.parametrize("fixed", ["free", "partial", "all"])
def test_gradgradb_vjp_matches_native_curve_and_current_central_differences(case, fixed):
    """Hessian VJPs match native finite differences for free and fixed coil DOFs."""
    set_backend("jax", device="cpu", intent="parity")
    field, native, curves, currents, seeds = _ncsx_case(case, fixed)
    assert not np.array_equal(seeds[2], seeds[2].swapaxes(1, 2))
    np.testing.assert_allclose(field.d2B_by_dXdX(), native.d2B_by_dXdX(), rtol=_DIRECT_RTOL, atol=_DIRECT_ATOL)
    derivatives = field.B_and_dB_and_d2B_vjp(*seeds)
    assert len(derivatives) == 3 and all(isinstance(value, Derivative) for value in derivatives)
    derivative = derivatives[2]
    partials = cast(Derivative, derivative(field, as_derivative=True))
    for owner in [*curves, *currents]:
        step = 0.1 if isinstance(owner, Current) else 2e-5
        coarse = _central_difference(native, owner, seeds[2], step)
        fine = _central_difference(native, owner, seeds[2], step / 2)
        # Extrapolate central differences to remove their leading O(step^2) error.
        expected = (4 * fine - coarse) / 3
        np.testing.assert_allclose(
            np.asarray(partials.data[owner]), expected, rtol=_FD_RTOL,
            atol=1e-12 * float(np.max(np.abs(expected))),
            err_msg=f"{case}/{fixed}: full {type(owner).__name__} cotangents",
        )
    pullback = field.d2B_by_dXdX_pullback_native(seeds[2])
    projected = field.coil_cotangents_to_dofs_gradient(pullback.d_coil_arrays, pullback.coil_indices)
    gradient = np.asarray(derivative(field))
    np.testing.assert_allclose(projected, gradient, rtol=1e-12, atol=1e-14)
    assert len(gradient) == len(native.x)


@pytest.mark.parametrize("case", ["one", "ncsx"])
def test_first_two_slots_equal_existing_B_and_dB_vjp(case):
    """Adding the Hessian slot preserves both existing field and gradient VJPs."""
    field, _, _, _, seeds = _ncsx_case(case, "partial")
    expected = field.B_and_dB_vjp(*seeds[:2])
    actual = field.B_and_dB_and_d2B_vjp(*seeds)
    for previous, added in zip(expected, actual[:2], strict=True):
        np.testing.assert_array_equal(previous(field), added(field))
        for owner, value in previous.data.items():
            np.testing.assert_array_equal(value, added.data[owner])


def _set_reverse_tile(monkeypatch, tile):
    tuning = replace(
        get_field_kernel_tuning(), coil_chunk_size=0, quadrature_block_size=0,
        hessian_vjp_point_chunk_size=tile,
    )
    monkeypatch.setattr(core, "get_field_kernel_tuning", lambda: tuning)
    core.invalidate_kernel_cache()


@pytest.mark.parametrize("point_count", [0, 1, 4, 5, 8, 12, 13, 37])
def test_tiled_hessian_vjp_matches_dense_tile_zero_with_multiple_groups(monkeypatch, point_count):
    """Tiled Hessian pullbacks match dense and direct autodiff across coil groups."""
    curves = create_equally_spaced_curves(
        2, 1, stellsym=False, R0=1.0, R1=0.3, order=2, numquadpoints=12,
    )
    curves[1] = create_equally_spaced_curves(
        2, 1, stellsym=False, R0=1.0, R1=0.3, order=2, numquadpoints=16,
    )[1]
    field = JaxBiotSavart([Coil(curves[0], Current(1e5)), Coil(curves[1], Current(-2e4))])
    rng = np.random.default_rng(649)
    points = rng.uniform(-0.2, 0.2, (point_count, 3)) + np.array([0.9, 0, 0])
    field.set_points(points)
    seed = jnp.asarray(rng.standard_normal((point_count, 3, 3, 3)))
    _set_reverse_tile(monkeypatch, 0)
    expected = field.d2B_by_dXdX_pullback_native(seed).d_coil_arrays
    direct = field._field_pullback_native(grouped_biot_savart_d2B_by_dXdX_from_inputs, seed).d_coil_arrays
    _set_reverse_tile(monkeypatch, 4)
    pullback = field.d2B_by_dXdX_pullback_native(seed)
    assert pullback.coil_indices == field.coil_set_spec().coil_index_lists()
    for actual, reference in zip(jax.tree.leaves(pullback.d_coil_arrays), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(actual, reference, rtol=1e-12, atol=1e-14)
    for actual, reference in zip(jax.tree.leaves(pullback.d_coil_arrays), jax.tree.leaves(direct), strict=True):
        np.testing.assert_allclose(actual, reference, rtol=1e-12, atol=1e-14)


def test_nonsymmetric_seed_contracts_native_ordered_hessian_current_gradient():
    """Mixed spatial partials commute; this oracle detects component-axis mixups."""
    set_backend("jax", device="cpu", intent="parity")
    field, native, _, currents, _ = _ncsx_case("one", "free")
    center = field.coils[0].curve.gamma()[0]
    # Far-field vacuum Hessians can be symmetric even in their component axis.
    # Near-coil finite quadrature distinguishes component/spatial ordering.
    points = center + np.array([[0.02, 0.01, -0.025], [-0.01, 0.025, 0.02], [0.02, -0.02, 0.015]])
    field.set_points(points)
    native.set_points(points)
    seed = np.random.default_rng(649).standard_normal((len(points), 3, 3, 3))
    assert not np.array_equal(seed, seed.swapaxes(1, 2))
    hessian = native.d2B_by_dXdX()
    # B is linear in the single coil's current, so this is an independent
    # exact current-gradient oracle with explicit [point, d1, d2, component] axes.
    expected = np.einsum("pijc,pijc->", seed, hessian) / currents[0].get_value()
    derivative = field.B_and_dB_and_d2B_vjp(np.zeros((len(points), 3)), np.zeros((len(points), 3, 3)), seed)[2]
    actual = derivative.data[currents[0]][0]
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12 * abs(expected))


def test_new_seeds_reuse_compiled_hessian_vjp(monkeypatch):
    """Changing the Hessian seed reuses compilation and scales the cotangents."""
    field, _, _, _, seeds = _ncsx_case("one", "free")
    _set_reverse_tile(monkeypatch, 4)
    kernel = cast(_CompiledCache, core._make_d2B_vjp_kernel(0, 0, 4))
    first = field.d2B_by_dXdX_pullback_native(seeds[2])
    jax.block_until_ready(first.d_coil_arrays)
    compiled_count = kernel._cache_size()
    assert compiled_count == 1
    second = field.d2B_by_dXdX_pullback_native(-2 * seeds[2])
    jax.block_until_ready(second.d_coil_arrays)
    assert kernel._cache_size() == compiled_count
    for actual, original in zip(jax.tree.leaves(second.d_coil_arrays), jax.tree.leaves(first.d_coil_arrays), strict=True):
        np.testing.assert_allclose(actual, -2 * original, rtol=1e-13, atol=1e-14)
