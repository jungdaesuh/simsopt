"""gradgradB VJP against native Hessian finite differences, on CPU."""

from __future__ import annotations

from unittest_jax_support import JAX_IMPORT_ERROR, JaxTestCase
from contextlib import ExitStack
from unittest import mock
from dataclasses import replace
from typing import Protocol, cast

import numpy as np

from simsopt._core.derivative import Derivative
from simsopt._core.optimizable import Optimizable
from simsopt.configs import get_ncsx_data
from simsopt.field import BiotSavart, Coil, Current, coils_via_symmetries
from simsopt.geo import create_equally_spaced_curves

try:
    import simsopt_jax  # noqa: F401
    import jax
    import jax.numpy as jnp
    from core.test_buffer_ownership import (
        _host_array as _host_input,
        make_execution_gate as _make_execution_gate,
    )
    from simsopt_jax.backend import get_field_kernel_tuning, set_backend
    from simsopt_jax.core import biotsavart as core
    from simsopt_jax.core.field import grouped_biot_savart_d2B_by_dXdX_from_inputs
    from simsopt_jax_adapters.field.biotsavart_backend import JaxBiotSavart
except ImportError:
    if JAX_IMPORT_ERROR is None:
        raise

_DIRECT_RTOL, _DIRECT_ATOL = (1e-10, 1e-12)
_FD_RTOL = 1e-07


class _CompiledCache(Protocol):
    """``jax.jit`` wrappers expose their compiled-program count as ``_cache_size``."""

    def _cache_size(self) -> int: ...


def _ncsx_case(case, fixed):
    curves, currents, _ = get_ncsx_data()
    coils = coils_via_symmetries(curves, currents, 3, True)
    if case == "one":
        coils, curves, currents = ([coils[0]], curves[:1], currents[:1])
    if fixed == "partial":
        for curve in curves:
            curve.fix(curve.local_dof_names[0])
        currents[0].fix_all()
    elif fixed == "all":
        for owner in [*curves, *currents]:
            owner.fix_all()
    rng = np.random.default_rng(649)
    phi = rng.uniform(0, 2 * np.pi, 5)
    points = np.stack(
        (1.5 * np.cos(phi), 1.5 * np.sin(phi), rng.uniform(-0.1, 0.1, 5)), axis=1
    )
    seeds = tuple(
        (rng.standard_normal((len(points),) + (3,) * rank) for rank in (1, 2, 3))
    )
    native = BiotSavart(coils).set_points(points)
    field = JaxBiotSavart(coils).set_points(points)
    return (field, native, curves, currents, seeds)


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


def _set_reverse_tile(patches: ExitStack, tile: int) -> None:
    tuning = replace(
        get_field_kernel_tuning(),
        coil_chunk_size=0,
        quadrature_block_size=0,
        hessian_vjp_point_chunk_size=tile,
    )
    patches.enter_context(
        mock.patch.object(core, "get_field_kernel_tuning", lambda: tuning)
    )
    core.invalidate_kernel_cache()


class TestBiotsavartGradgradbVjp(JaxTestCase):
    def test_gradgradb_vjp_contracted_hessian_shrinking_steps(self):
        """Contracted Hessian VJPs converge for curve DOFs and stay linear in current."""
        for fixed in ["free", "partial", "all"]:
            for case in ["one", "ncsx"]:
                with self.subTest(fixed=fixed, case=case), self.case() as patches:
                    self._case_gradgradb_vjp_contracted_hessian_shrinking_steps(
                        case, fixed, patches
                    )

    def _case_gradgradb_vjp_contracted_hessian_shrinking_steps(
        self, case, fixed, patches: ExitStack
    ):
        """Compare full owner partials against native contracted Hessians at five scales."""
        set_backend("jax", device="cpu", intent="parity")
        field, native, curves, currents, seeds = _ncsx_case(case, fixed)
        derivative = field.B_and_dB_and_d2B_vjp(*seeds)[2]
        partials = cast(Derivative, derivative(field, as_derivative=True))
        points = native.get_points_cart().copy()
        patches.callback(native.set_points, points)
        for owner in [*curves, *currents]:
            original = cast(np.ndarray, owner.local_full_x).copy()
            patches.callback(setattr, owner, "local_full_x", original)
            if isinstance(owner, Current):
                # Relative current perturbations avoid subtracting nearly equal
                # multi-coil fields when the current is of order 1e5 amperes.
                direction = 1e-2 * original
                analytic = float(np.asarray(partials.data[owner]) @ direction)
                # Current dependence is affine, so no truncation ratio is expected.
                for exponent in range(5, 10):
                    step = 0.5**exponent
                    owner.local_full_x = original + step * direction
                    native.set_points(points)
                    plus = np.sum(native.d2B_by_dXdX() * seeds[2])
                    owner.local_full_x = original - step * direction
                    native.set_points(points)
                    minus = np.sum(native.d2B_by_dXdX() * seeds[2])
                    np.testing.assert_allclose(
                        analytic,
                        (plus - minus) / (2 * step),
                        rtol=_FD_RTOL,
                        atol=1e-12 * abs(analytic),
                        err_msg=f"{case}/{fixed}: current Hessian contraction at step {step}",
                    )
            else:
                # Extend the native B/dB VJP forward stencil to contracted d2B.
                baseline = np.sum(native.d2B_by_dXdX() * seeds[2])
                direction = 1e-2 * np.random.RandomState(1).rand(original.size)
                analytic = np.asarray(partials.data[owner]) @ direction
                error = 1e6
                for exponent in range(5, 10):
                    step = 0.5**exponent
                    owner.local_full_x = original + step * direction
                    # All-fixed graphs need explicit native cache invalidation.
                    native.set_points(points)
                    value = np.sum(native.d2B_by_dXdX() * seeds[2])
                    next_error = np.linalg.norm((value - baseline) / step - analytic)
                    self.assertLess(
                        next_error,
                        0.55 * error,
                        f"{case}/{fixed}: {type(owner).__name__} Hessian contraction at step {step}",
                    )
                    error = next_error
            owner.local_full_x = original
            native.set_points(points)

    def test_gradgradb_vjp_matches_native_curve_and_current_central_differences(self):
        """Hessian VJPs match native finite differences for free and fixed coil DOFs."""
        for fixed in ["free", "partial", "all"]:
            for case in ["one", "ncsx"]:
                with self.subTest(fixed=fixed, case=case), self.case() as patches:
                    self._case_gradgradb_vjp_matches_native_curve_and_current_central_differences(
                        case, fixed, patches
                    )

    def _case_gradgradb_vjp_matches_native_curve_and_current_central_differences(
        self, case, fixed, patches: ExitStack
    ):
        """Hessian VJPs match native finite differences for free and fixed coil DOFs."""
        set_backend("jax", device="cpu", intent="parity")
        field, native, curves, currents, seeds = _ncsx_case(case, fixed)
        self.assertFalse(
            np.array_equal(seeds[2], seeds[2].swapaxes(1, 2)),
            "not np.array_equal(seeds[2], seeds[2].swapaxes(1, 2))",
        )
        np.testing.assert_allclose(
            field.d2B_by_dXdX(),
            native.d2B_by_dXdX(),
            rtol=_DIRECT_RTOL,
            atol=_DIRECT_ATOL,
        )
        derivatives = field.B_and_dB_and_d2B_vjp(*seeds)
        self.assertTrue(
            len(derivatives) == 3
            and all((isinstance(value, Derivative) for value in derivatives)),
            "len(derivatives) == 3 and all((isinstance(value, Derivative) for value in derivatives))",
        )
        derivative = derivatives[2]
        partials = cast(Derivative, derivative(field, as_derivative=True))
        for owner in [*curves, *currents]:
            step = 0.1 if isinstance(owner, Current) else 2e-05
            coarse = _central_difference(native, owner, seeds[2], step)
            fine = _central_difference(native, owner, seeds[2], step / 2)
            # Extrapolate central differences to remove their leading O(step^2) error.
            expected = (4 * fine - coarse) / 3
            np.testing.assert_allclose(
                np.asarray(partials.data[owner]),
                expected,
                rtol=_FD_RTOL,
                atol=1e-12 * float(np.max(np.abs(expected))),
                err_msg=f"{case}/{fixed}: full {type(owner).__name__} cotangents",
            )
        pullback = field.d2B_by_dXdX_pullback_native(seeds[2])
        projected = field.coil_cotangents_to_dofs_gradient(
            pullback.d_coil_arrays, pullback.coil_indices
        )
        gradient = np.asarray(derivative(field))
        np.testing.assert_allclose(projected, gradient, rtol=1e-12, atol=1e-14)
        self.assertEqual(len(gradient), len(native.x), "len(gradient) == len(native.x)")

    def test_first_two_slots_equal_existing_B_and_dB_vjp(self):
        """Adding the Hessian slot preserves both existing field and gradient VJPs."""
        for case in ["one", "ncsx"]:
            with self.subTest(case=case), self.case() as patches:
                self._case_first_two_slots_equal_existing_B_and_dB_vjp(case, patches)

    def _case_first_two_slots_equal_existing_B_and_dB_vjp(
        self, case, patches: ExitStack
    ):
        """Adding the Hessian slot preserves both existing field and gradient VJPs."""
        field, _, _, _, seeds = _ncsx_case(case, "partial")
        expected = field.B_and_dB_vjp(*seeds[:2])
        actual = field.B_and_dB_and_d2B_vjp(*seeds)
        for previous, added in zip(expected, actual[:2], strict=True):
            np.testing.assert_array_equal(previous(field), added(field))
            for owner, value in previous.data.items():
                np.testing.assert_array_equal(value, added.data[owner])

    def test_tiled_hessian_vjp_matches_dense_tile_zero_with_multiple_groups(self):
        """Tiled Hessian pullbacks match dense and direct autodiff across coil groups."""
        for point_count in [0, 1, 4, 5, 8, 12, 13, 37]:
            with self.subTest(point_count=point_count), self.case() as patches:
                self._case_tiled_hessian_vjp_matches_dense_tile_zero_with_multiple_groups(
                    point_count, patches
                )

    def _case_tiled_hessian_vjp_matches_dense_tile_zero_with_multiple_groups(
        self, point_count, patches: ExitStack
    ):
        """Tiled Hessian pullbacks match dense and direct autodiff across coil groups."""
        curves = create_equally_spaced_curves(
            2, 1, stellsym=False, R0=1.0, R1=0.3, order=2, numquadpoints=12
        )
        curves[1] = create_equally_spaced_curves(
            2, 1, stellsym=False, R0=1.0, R1=0.3, order=2, numquadpoints=16
        )[1]
        field = JaxBiotSavart(
            [Coil(curves[0], Current(100000.0)), Coil(curves[1], Current(-20000.0))]
        )
        rng = np.random.default_rng(649)
        points = rng.uniform(-0.2, 0.2, (point_count, 3)) + np.array([0.9, 0, 0])
        field.set_points(points)
        seed = jnp.asarray(rng.standard_normal((point_count, 3, 3, 3)))
        _set_reverse_tile(patches, 0)
        expected = field.d2B_by_dXdX_pullback_native(seed).d_coil_arrays
        direct = field._field_pullback_native(
            grouped_biot_savart_d2B_by_dXdX_from_inputs, seed
        ).d_coil_arrays
        _set_reverse_tile(patches, 4)
        pullback = field.d2B_by_dXdX_pullback_native(seed)
        self.assertEqual(
            pullback.coil_indices,
            field.coil_set_spec().coil_index_lists(),
            "pullback.coil_indices == field.coil_set_spec().coil_index_lists()",
        )
        for actual, reference in zip(
            jax.tree.leaves(pullback.d_coil_arrays),
            jax.tree.leaves(expected),
            strict=True,
        ):
            np.testing.assert_allclose(actual, reference, rtol=1e-12, atol=1e-14)
        for actual, reference in zip(
            jax.tree.leaves(pullback.d_coil_arrays),
            jax.tree.leaves(direct),
            strict=True,
        ):
            np.testing.assert_allclose(actual, reference, rtol=1e-12, atol=1e-14)

    def test_nonsymmetric_seed_contracts_native_ordered_hessian_current_gradient(self):
        """Mixed spatial partials commute; this oracle detects component-axis mixups."""
        set_backend("jax", device="cpu", intent="parity")
        field, native, _, currents, _ = _ncsx_case("one", "free")
        center = field.coils[0].curve.gamma()[0]
        # Far-field vacuum Hessians can be symmetric even in their component axis.
        # Near-coil finite quadrature distinguishes component/spatial ordering.
        points = center + np.array(
            [[0.02, 0.01, -0.025], [-0.01, 0.025, 0.02], [0.02, -0.02, 0.015]]
        )
        field.set_points(points)
        native.set_points(points)
        seed = np.random.default_rng(649).standard_normal((len(points), 3, 3, 3))
        self.assertFalse(
            np.array_equal(seed, seed.swapaxes(1, 2)),
            "not np.array_equal(seed, seed.swapaxes(1, 2))",
        )
        hessian = native.d2B_by_dXdX()
        # B is linear in the single coil's current, so this is an independent
        # exact current-gradient oracle with explicit [point, d1, d2, component] axes.
        expected = np.einsum("pijc,pijc->", seed, hessian) / currents[0].get_value()
        derivative = field.B_and_dB_and_d2B_vjp(
            np.zeros((len(points), 3)), np.zeros((len(points), 3, 3)), seed
        )[2]
        actual = derivative.data[currents[0]][0]
        np.testing.assert_allclose(
            actual, expected, rtol=1e-10, atol=1e-12 * abs(expected)
        )

    def test_new_seeds_reuse_compiled_hessian_vjp(self):
        """Changing the Hessian seed reuses compilation and scales the cotangents."""
        patches = self.patches
        field, _, _, _, seeds = _ncsx_case("one", "free")
        _set_reverse_tile(patches, 4)
        kernel = cast(_CompiledCache, core._make_d2B_vjp_kernel(0, 0, 4))
        first = field.d2B_by_dXdX_pullback_native(seeds[2])
        jax.block_until_ready(first.d_coil_arrays)
        compiled_count = kernel._cache_size()
        self.assertEqual(compiled_count, 1, "compiled_count == 1")
        second = field.d2B_by_dXdX_pullback_native(-2 * seeds[2])
        jax.block_until_ready(second.d_coil_arrays)
        self.assertEqual(
            kernel._cache_size(),
            compiled_count,
            "kernel._cache_size() == compiled_count",
        )
        for actual, original in zip(
            jax.tree.leaves(second.d_coil_arrays),
            jax.tree.leaves(first.d_coil_arrays),
            strict=True,
        ):
            np.testing.assert_allclose(actual, -2 * original, rtol=1e-13, atol=1e-14)

    def test_hessian_vjp_owns_numpy_inputs(self):
        """Caller mutation after dispatch cannot alter pending Hessian cotangents."""
        for pending_input in ["points", "seed"]:
            for misaligned in [False, True]:
                with self.subTest(
                    pending_input=pending_input,
                    misaligned=misaligned,
                    alignment="misaligned" if misaligned else "aligned",
                ), self.case() as patches:
                    self._case_hessian_vjp_owns_numpy_inputs(
                        misaligned, pending_input, patches
                    )

    def _case_hessian_vjp_owns_numpy_inputs(
        self, misaligned, pending_input, patches: ExitStack
    ):
        """Caller mutation after dispatch cannot alter pending Hessian cotangents."""
        _execution_gate = _make_execution_gate()
        set_backend("jax", device="cpu", intent="parity")
        _set_reverse_tile(patches, 4)
        sources = [
            _host_input(shape, misaligned=misaligned)
            for shape in ((13, 3), (13, 3, 3, 3), (1, 12, 3), (1, 12, 3), (1,))
        ]
        sources[0] *= 0.2
        expected = jax.tree.map(
            lambda value: np.array(value, copy=True),
            jax.block_until_ready(core.biot_savart_d2B_by_dXdX_vjp(*sources)),
        )
        index = 0 if pending_input == "points" else 1
        device_input = jnp.asarray(sources[index].copy())
        _execution_gate(device_input).block_until_ready()
        inputs = list(sources)
        inputs[index] = _execution_gate(device_input)
        result = core.biot_savart_d2B_by_dXdX_vjp(*inputs)
        self.assertTrue(
            any((not leaf.is_ready() for leaf in jax.tree.leaves(result))),
            "pullback completed before mutation",
        )
        for source in sources:
            source[:] = -7.0
        jax.block_until_ready(result)
        for observed, saved in zip(
            jax.tree.leaves(result), jax.tree.leaves(expected), strict=True
        ):
            np.testing.assert_array_equal(observed, saved)
