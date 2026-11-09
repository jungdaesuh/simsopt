"""Exact values and derivatives of the JAX helpers, and the pytree records they compute with.

1. The Biot-Savart helpers keep point-on-coil singularities loud, and the
   pairwise sum keeps IEEE semantics.
2. ``device_one`` adds no derivative path, and ``placement_zero`` is an exact
   placed +0.0 with exact zero tangents and cotangents, with JIT or without,
   batched and nested.
3. ``simsopt_jax.pytree`` registers frozen records whose leaves round-trip in
   the declared order, and rejects invalid partitions without registering.
4. A coil set without coils has no field, and coil specs rebuilt from DOFs
   keep current expressions, scaled currents and shared symmetric geometry.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import FrozenInstanceError, dataclass, field
from typing import cast

import numpy as np
from unittest_jax_support import JaxTestCase, optional_jax_imports

with optional_jax_imports():
    import simsopt_jax  # noqa: F401
    import jax
    import jax.numpy as jnp
    from simsopt_jax.backend.jit import jit
    from simsopt_jax.core._device_scalars import device_one, placement_zero
    from simsopt_jax.core._math_utils import pairwise_sum_axis
    from simsopt_jax.core.biotsavart import (
        _radius_squared,
        _separation,
        biot_savart_A,
        biot_savart_B,
        biot_savart_dB_by_dX,
    )
    from simsopt_jax.core.curve_xyz_fourier import _constant_row
    from simsopt_jax.core.field import (
        grouped_biot_savart_A_from_spec,
        grouped_biot_savart_B_and_dB_from_spec,
        grouped_biot_savart_B_from_spec,
        grouped_biot_savart_d2B_by_dXdX_from_spec,
        grouped_biot_savart_dB_by_dX_from_spec,
    )
    from simsopt_jax.core.specs import make_grouped_coil_set_spec
    from simsopt_jax import pytree
    from simsopt_jax.pytree import (
        pytree_dataclass,
        pytree_node,
        registered_pytree_classes,
    )
    from simsopt_jax_adapters.field.biotsavart_backend import JaxBiotSavart

    # Native simsopt.geo imports jax too.
    from simsopt.field import Coil, Current, coils_via_symmetries
    from simsopt.geo import create_equally_spaced_curves


# ── Kernel numeric helpers ────────────────────────────────────────────


class TestKernelNumericHelpers(JaxTestCase):
    """Small JAX kernel numeric helpers keep exact zeros, singularities and IEEE sums."""

    def test_radius_squared_preserves_zero_singularity(self):
        """Squared radius remains exactly zero at a coincident point."""
        diff = jnp.zeros((1, 3), dtype=jnp.float64)

        radius_squared = _radius_squared(diff)

        np.testing.assert_allclose(np.asarray(radius_squared), np.zeros((1,)))

    def test_separation_preserves_zero_singularity(self):
        """The component-major separation is exact: a coincident point gives zero
        components and an exactly zero squared radius."""
        x = jnp.asarray([0.25, -0.5, 1.0], dtype=jnp.float64)
        gammas = jnp.reshape(x, (1, 1, 3))

        components, radius_squared = _separation(x, gammas)

        for component in components:
            np.testing.assert_array_equal(np.asarray(component), np.zeros((1, 1)))
        np.testing.assert_array_equal(np.asarray(radius_squared), np.zeros((1, 1)))

    def test_biot_savart_public_kernels_preserve_point_singularity(self):
        """Public vector-potential, field and field-gradient kernels retain filament
        singularities."""
        points = jnp.asarray([[0.0, 0.0, 0.0]], dtype=jnp.float64)
        gammas = jnp.asarray([[[0.0, 0.0, 0.0]]], dtype=jnp.float64)
        gammadashs = jnp.asarray([[[1.0, 0.0, 0.0]]], dtype=jnp.float64)
        currents = jnp.asarray([1.0], dtype=jnp.float64)

        for kernel in (biot_savart_A, biot_savart_B, biot_savart_dB_by_dX):
            value = kernel(points, gammas, gammadashs, currents)

            self.assertTrue(
                not np.all(np.isfinite(np.asarray(value))),
                "not np.all(np.isfinite(np.asarray(value)))",
            )

    def test_constant_row_builds_its_row_from_a_traced_reference(self):
        """The row is selected statically but sourced from the reference's device."""

        @jit
        def rows_for(reference):
            return (
                _constant_row(3, is_one=True, reference=reference),
                _constant_row(3, is_one=False, reference=reference),
            )

        ones_row, zeros_row = rows_for(jnp.asarray(1.0, dtype=jnp.float64))

        np.testing.assert_allclose(np.asarray(ones_row), np.ones((1, 3)))
        np.testing.assert_allclose(np.asarray(zeros_row), np.zeros((1, 3)))

    def test_pairwise_sum_padding_reads_no_values(self):
        """Power-of-two padding is exact zeros, so a quadrature sum keeps IEEE
        semantics: an infinite term gives an infinite sum, and finite terms whose
        plain sum overflows do not turn the padding into NaN."""
        rows = [
            ((np.inf, 1.0, 1.0), np.inf),
            ((1.0e308, 1.0e308, -1.0e308), np.inf),
            ((1.0e308, -1.0e308, 1.0), 1.0),
        ]
        for values, expected in rows:
            with self.subTest(values=values):
                total = pairwise_sum_axis(jnp.asarray(values), axis=0)
                self.assertEqual(float(total), expected)


# ── device_one ────────────────────────────────────────────────────────

UNIT_ROUNDOFF = float(np.finfo(np.float64).eps) / 2.0


def _gamma(k: int) -> float:
    """Higham's ``gamma_k = k u / (1 - k u)``."""
    return k * UNIT_ROUNDOFF / (1.0 - k * UNIT_ROUNDOFF)


def _coils(quadrature: int):
    t = np.arange(quadrature) / quadrature
    phi = 2.0 * np.pi * t
    gammas, gammadashs = [], []
    for radius, height in ((0.5, 0.0), (0.4, 0.3)):
        gammas.append(
            np.stack(
                [1.0 + radius * np.cos(phi), radius * np.sin(phi), height + 0.0 * phi],
                axis=1,
            )
        )
        gammadashs.append(
            2.0
            * np.pi
            * np.stack([-radius * np.sin(phi), radius * np.cos(phi), 0.0 * phi], axis=1)
        )
    return np.stack(gammas), np.stack(gammadashs)


def _B_current_cotangent(points, gammas, gammadashs, currents, cotangent):
    """``d(cotangent . B)/dI`` in T/A per unit cotangent, shape (C,)."""
    _, pullback = jax.vjp(
        lambda i: biot_savart_B(
            jnp.asarray(points), jnp.asarray(gammas), jnp.asarray(gammadashs), i
        ),
        jnp.asarray(currents),
    )
    return np.asarray(pullback(jnp.asarray(cotangent))[0])


class TestDeviceOneGradientPath(JaxTestCase):
    """``device_one`` must add no derivative path.

    ``device_one(r)`` builds a 1.0 on ``r``'s device without reading ``r``.  Its
    derivative is zero, but when it was built as a differentiable ``exp(sum(r - r))``
    reverse mode sent the
    whole cotangent ``c`` of each product ``device_one(r) * y`` into ``r`` twice, as
    ``+c`` and ``-c``, beside ``r``'s own cotangent ``g``: the accumulated ``(g + c)
    - c`` then carries an error ``u |c|`` instead of ``u |g|``.  In Biot-Savart the
    product is ``mu0/4pi * (currents . integral)`` with the currents as reference,
    so ``c`` is the field's whole cotangent contraction and ``g = c / I``: the
    current gradient of a Stage-II flux lost five digits (relative error 1e-11 at
    ``I = 1e5``, planar-coils profile 2026-09-29).  Reading ``r`` also turned the
    1.0 into NaN for one infinite current, which then spread to every coil.
    """

    def test_device_one_has_a_zero_cotangent(self):
        """Multiplying by device_one leaves the reference gradient equal to that of the
        original objective."""
        reference = jnp.asarray([1.0e5, -3.0, 2.5e-7])
        cotangent = jax.grad(lambda r: device_one(r) * jnp.sum(r * r))(reference)
        np.testing.assert_array_equal(
            np.asarray(cotangent), 2.0 * np.asarray(reference)
        )

    def test_biot_savart_current_cotangent_is_the_unit_field_contraction(self):
        """``d(ct . B)/dI_c = ct . B_c(I_c = 1)`` to the rounding of that contraction.

        Both sides evaluate ``1e-7 / Q * sum_{p, j, q} ct[p, j] * k[c, q, p, j]``
        for the Biot-Savart integrand ``k`` over ``Q`` quadrature points, only in
        different orders, so they differ by at most ``2 gamma_K S`` with ``S`` the
        sum of the absolute products and ``K`` the additions (``3 P Q - 1``) plus
        the integrand's own roundings (at most 16: difference, squared radius,
        reciprocal square root and cube, cross product, scalings).  The currents are
        ``1e5``, the planar-coils scale, where the removed ``device_one`` path cost
        a relative 1e-11.
        """
        quadrature = 32
        gammas, gammadashs = _coils(quadrature)
        points = np.array(
            [[1.0, 0.0, 0.1], [1.2, 0.3, -0.2], [0.7, -0.4, 0.25], [1.1, 0.05, 0.6]]
        )
        currents = np.array([1.0e5, -1.0e5])
        cotangent = np.random.default_rng(20260929).standard_normal(points.shape)

        current_cotangent = _B_current_cotangent(
            points, gammas, gammadashs, currents, cotangent
        )

        difference = (
            points[None, None, :, :] - gammas[:, :, None, :]
        )  # [coil, q, point, 3]
        radius = np.linalg.norm(difference, axis=-1)
        integrand = (
            np.cross(gammadashs[:, :, None, :], difference) / radius[..., None] ** 3
        )
        scale = 1.0e-7 / quadrature
        expected = scale * np.einsum("pj,cqpj->c", cotangent, integrand)
        magnitude = scale * np.einsum(
            "pj,cqpj->c", np.abs(cotangent), np.abs(integrand)
        )
        additions = 3 * points.shape[0] * quadrature - 1
        bound = 2.0 * _gamma(additions + 16) * magnitude

        error = np.abs(current_cotangent - expected)
        self.assertTrue(
            np.all(error <= bound), (error, bound, error / np.abs(expected))
        )

    def test_an_infinite_current_leaves_the_other_coils_derivative_unchanged(self):
        """B is linear in each current, so ``d(ct . B)/dI_1`` is coil 1's unit field
        contraction whatever coil 0 carries, even an infinite current."""
        gammas, gammadashs = _coils(16)
        points = np.array([[1.0, 0.0, 0.1], [1.2, 0.3, -0.2]])
        cotangent = np.random.default_rng(20261029).standard_normal(points.shape)

        finite = _B_current_cotangent(
            points, gammas, gammadashs, np.array([1.0e5, -1.0e5]), cotangent
        )
        infinite = _B_current_cotangent(
            points, gammas, gammadashs, np.array([np.inf, -1.0e5]), cotangent
        )

        self.assertTrue(np.isfinite(finite[1]), finite)
        np.testing.assert_array_equal(infinite[1], finite[1])


# ── placement_zero ────────────────────────────────────────────────────

REFERENCES = (
    (1.0e308, 1.0e308),
    (-1.0e308, -1.0e308),
    (np.nan, 1.0),
    (np.inf, -np.inf),
    (1.0, 2.0),
)
SEEDS = (np.nan, np.inf, -np.inf, 1.0)


def _placed(values) -> jax.Array:
    return jnp.array(np.asarray(values, dtype=np.float64))


class TestPlacementZero(JaxTestCase):
    """An exact placed zero must have exact zero derivatives even for non-finite seeds.

    Fixed currents need a tangent on the reference device under strict transfer
    guards. Computing zero from reference values could overflow for finite inputs.
    """

    def test_placement_zero_is_an_exact_zero_on_the_reference_device(self):
        """placement_zero is bitwise positive zero with the reference dtype and device
        under a strict guard."""
        for reference in REFERENCES:
            with self.subTest(reference=reference), self.case():
                self._case_placement_zero_is_an_exact_zero_on_the_reference_device(
                    reference
                )

    def _case_placement_zero_is_an_exact_zero_on_the_reference_device(self, reference):
        """Run one row with case-local objects released before runtime cleanup."""
        placed = _placed(reference)
        with jax.transfer_guard("disallow"):
            zero = placement_zero(placed)
            jax.block_until_ready(zero)
        self.assertTrue(zero.dtype == jnp.float64, "zero.dtype == jnp.float64")
        self.assertTrue(
            zero.devices() == placed.devices(),
            "zero.devices() == placed.devices()",
        )
        self.assertTrue(
            np.asarray(zero).view(np.uint64) == np.float64(0.0).view(np.uint64),
            "np.asarray(zero).view(np.uint64) == np.float64(0.0).view(np.uint64)",
        )

    def test_placement_zero_tangent_is_exactly_zero(self):
        """placement_zero produces a bitwise zero tangent on the reference device for
        every tested seed."""
        for seed in SEEDS:
            for reference in REFERENCES:
                with self.subTest(seed=seed, reference=reference), self.case():
                    self._case_placement_zero_tangent_is_exactly_zero(seed, reference)

    def _case_placement_zero_tangent_is_exactly_zero(self, seed, reference):
        """Run one row with case-local objects released before runtime cleanup."""
        placed = _placed(reference)
        tangent = _placed((seed, 1.0))
        with jax.transfer_guard("disallow"):
            _, zero_tangent = jax.jvp(placement_zero, (placed,), (tangent,))
            jax.block_until_ready(zero_tangent)
        self.assertTrue(
            zero_tangent.devices() == placed.devices(),
            "zero_tangent.devices() == placed.devices()",
        )
        self.assertTrue(
            np.asarray(zero_tangent).view(np.uint64) == np.float64(0.0).view(np.uint64),
            "np.asarray(zero_tangent).view(np.uint64) == np.float64(0.0).view(np.uint64)",
        )

    def test_placement_zero_cotangent_is_exactly_zero(self):
        """placement_zero sends bitwise positive-zero cotangents to every reference
        entry."""
        for seed in SEEDS:
            for reference in REFERENCES:
                with self.subTest(seed=seed, reference=reference), self.case():
                    self._case_placement_zero_cotangent_is_exactly_zero(seed, reference)

    def _case_placement_zero_cotangent_is_exactly_zero(self, seed, reference):
        """Run one row with case-local objects released before runtime cleanup."""
        placed = _placed(reference)
        cotangent = _placed(seed)
        with jax.transfer_guard("disallow"):
            _, pullback = jax.vjp(placement_zero, placed)
            (reference_cotangent,) = pullback(cotangent)
            jax.block_until_ready(reference_cotangent)
        np.testing.assert_array_equal(
            np.asarray(reference_cotangent).view(np.uint64),
            np.zeros(2, dtype=np.float64).view(np.uint64),
        )

    def test_placement_zero_linearizes_a_fixed_value_on_device(self):
        """A fixed value plus the placed zero linearizes under the strict guard."""
        placed = _placed((1.0e308, 1.0e308))
        fixed = _placed(3.0)
        tangent = _placed((np.inf, np.nan))
        with jax.transfer_guard("disallow"):
            value, linearized = jax.linearize(
                lambda r: fixed + placement_zero(r), placed
            )
            fixed_tangent = linearized(tangent)
            jax.block_until_ready((value, fixed_tangent))
        self.assertTrue(float(value) == 3.0, "float(value) == 3.0")
        self.assertTrue(
            np.asarray(fixed_tangent).view(np.uint64)
            == np.float64(0.0).view(np.uint64),
            "np.asarray(fixed_tangent).view(np.uint64) == np.float64(0.0).view(np.uint64)",
        )


class TestPlacementZeroEager(JaxTestCase):
    """Placed exact zeros survive eager AD, batching and disabled JIT."""

    def test_primal_tangent_and_transpose_stay_placed_without_jit(self) -> None:
        """Eager value, JVP and VJP are +0.0 on the input's device and dtype for non-finite inputs and seeds."""
        references = (
            np.asarray(np.nan),
            np.asarray((1.0e308, 1.0e308)),
            np.asarray((np.nan, np.inf, -np.inf)),
            np.asarray(((np.inf, np.nan), (-np.inf, 1.0))),
            np.empty((0,)),
        )
        for reference in references:
            placed = jnp.array(reference)
            for seed in (np.nan, np.inf, -np.inf, 1.0):
                tangent = jnp.array(np.full(reference.shape, seed))
                cotangent = jnp.array(np.asarray(seed))
                with self.subTest(shape=reference.shape, seed=seed):
                    with jax.disable_jit(), jax.transfer_guard("disallow"):
                        zero, derivative = jax.jvp(
                            placement_zero, (placed,), (tangent,)
                        )
                        _, pullback = jax.vjp(placement_zero, placed)
                        (transpose,) = pullback(cotangent)
                        jax.block_until_ready((zero, derivative, transpose))
                    for actual in (zero, derivative, transpose):
                        self.assertEqual(actual.devices(), placed.devices())
                        self.assertEqual(actual.dtype, placed.dtype)
                        np.testing.assert_array_equal(actual, np.zeros(actual.shape))
                        self.assertFalse(np.any(np.signbit(actual)))

    def test_nested_jvp_and_hvp_are_exact_zeros_without_jit(self) -> None:
        """Second-order JVPs and Hessian-vector products are exact +0.0, with and without JIT."""
        reference = jnp.array(np.asarray((np.nan, np.inf, 1.0e308)))
        direction = jnp.array(np.asarray((np.inf, np.nan, -np.inf)))
        seed = jnp.array(np.asarray(np.inf))
        seed_tangent = jnp.array(np.asarray(np.nan))

        def first_derivative(values: jax.Array) -> jax.Array:
            return jax.jvp(placement_zero, (values,), (values,))[1]

        def gradient(values: jax.Array, weight: jax.Array) -> jax.Array:
            _, pullback = jax.vjp(placement_zero, values)
            return pullback(weight)[0]

        for disable_jit in (False, True):
            with self.subTest(disable_jit=disable_jit):
                with jax.disable_jit(disable_jit), jax.transfer_guard("disallow"):
                    first, second = jax.jvp(
                        first_derivative, (reference,), (direction,)
                    )
                    derivative, hvp = jax.jvp(
                        gradient,
                        (reference, seed),
                        (direction, seed_tangent),
                    )
                    jax.block_until_ready((first, second, derivative, hvp))
                for actual in (first, second, derivative, hvp):
                    np.testing.assert_array_equal(actual, np.zeros(actual.shape))
                    self.assertFalse(np.any(np.signbit(actual)))

    def test_batched_zero_has_exact_nonfinite_jvp_and_transpose(self) -> None:
        """Under vmap, the value, JVP and VJP stay exact +0.0 for non-finite batches."""
        references = jnp.array(np.asarray(((np.nan, np.inf), (1.0e308, 1.0e308))))
        direction = jnp.array(np.full(references.shape, np.nan))
        seed = jnp.array(np.full((references.shape[0],), np.inf))
        with jax.disable_jit(), jax.transfer_guard("disallow"):
            values, tangent = jax.jvp(
                jax.vmap(placement_zero), (references,), (direction,)
            )
            _, pullback = jax.vjp(jax.vmap(placement_zero), references)
            (transpose,) = pullback(seed)
            jax.block_until_ready((values, tangent, transpose))
        for actual in (values, tangent, transpose):
            np.testing.assert_array_equal(actual, np.zeros(actual.shape))
            self.assertFalse(np.any(np.signbit(actual)))

    def test_finite_differences_agree_with_exact_zero_directional_derivative(
        self,
    ) -> None:
        """Central differences of placement_zero are exactly zero at every step size."""
        reference = np.asarray((1.0, -2.0, 3.0))
        direction = np.asarray((0.5, 0.2, -0.3))
        for step in (1.0e-2, 1.0e-4, 1.0e-6):
            plus = jnp.array(reference + step * direction)
            minus = jnp.array(reference - step * direction)
            with self.subTest(step=step):
                with jax.disable_jit(), jax.transfer_guard("disallow"):
                    values = (placement_zero(plus), placement_zero(minus))
                    jax.block_until_ready(values)
                finite_difference = (np.asarray(values[0]) - np.asarray(values[1])) / (
                    2.0 * step
                )
                np.testing.assert_array_equal(finite_difference, 0.0)


# ── Pytree records ────────────────────────────────────────────────────


class TestPytreeRecords(JaxTestCase):
    """``pytree_dataclass`` and ``pytree_node`` register frozen records with ordered leaves."""

    def test_plain_class_is_frozen_and_round_trips_in_data_order(self):
        """The dataclass helper freezes plain classes and reconstructs fields in declared
        leaf order."""

        @pytree.pytree_dataclass(data=("second", "first"), meta=("mode",))
        class Payload:
            first: int
            second: int
            mode: str

        payload = Payload(first=1, second=2, mode="test")
        with self.assertRaises(FrozenInstanceError):
            setattr(payload, "first", 3)
        leaves, treedef = jax.tree_util.tree_flatten(payload)
        self.assertTrue(leaves == [2, 1], "leaves == [2, 1]")
        self.assertTrue(
            jax.tree_util.tree_unflatten(treedef, leaves) == payload,
            "jax.tree_util.tree_unflatten(treedef, leaves) == payload",
        )
        self.assertTrue(
            jax.tree_util.tree_unflatten(treedef, [20, 10]) == Payload(10, 20, "test"),
            'jax.tree_util.tree_unflatten(treedef, [20, 10]) == Payload(10, 20, "test")',
        )

    def test_mutable_dataclass_is_rejected(self):
        """The dataclass helper rejects an existing mutable dataclass."""

        @dataclass
        class Mutable:
            value: int

        with self.assertRaisesRegex(TypeError, "Mutable must be a frozen dataclass"):
            pytree_dataclass(data=("value",))(Mutable)

    def test_invalid_partition_is_rejected(self):
        """Invalid data/meta partitions fail without changing the pytree registry."""
        for data, meta, message in (
            (("value",), ("value", "mode"), "data and meta overlap"),
            (("value",), (), "missing=\\['mode'\\]"),
            (("value", "extra"), ("mode",), "unknown=\\['extra'\\]"),
        ):
            with self.subTest(data=data, meta=meta, message=message), self.case():
                self._case_invalid_partition_is_rejected(data, meta, message)

    def _case_invalid_partition_is_rejected(self, data, meta, message):
        """Run one row with case-local objects released before runtime cleanup."""

        class Payload:
            value: int
            mode: str

        before = registered_pytree_classes()
        with self.assertRaisesRegex(ValueError, message):
            pytree_dataclass(data=data, meta=meta)(Payload)
        self.assertTrue(
            registered_pytree_classes() == before,
            "registered_pytree_classes() == before",
        )

    def test_existing_dataclass_options_and_constructor_are_preserved(self):
        """Registration retains frozen-dataclass options and reconstructs through its
        custom constructor."""

        @dataclass(frozen=True, init=False, eq=False)
        class Payload:
            value: int
            derived: int = field(init=False)

            def __init__(self, value):
                object.__setattr__(self, "value", value)
                object.__setattr__(self, "derived", 2 * value)

        self.assertTrue(
            pytree_dataclass(data=("value",))(Payload) is Payload,
            'pytree_dataclass(data=("value",))(Payload) is Payload',
        )
        self.assertTrue(Payload(3).derived == 6, "Payload(3).derived == 6")
        self.assertTrue(Payload(3) != Payload(3), "Payload(3) != Payload(3)")
        leaves, treedef = jax.tree_util.tree_flatten(Payload(3))
        self.assertTrue(
            jax.tree_util.tree_unflatten(treedef, leaves).derived == 6,
            "jax.tree_util.tree_unflatten(treedef, leaves).derived == 6",
        )

    def test_registry_returns_immutable_snapshot(self):
        """The registry returns immutable snapshots that do not change after later
        registration."""
        before = registered_pytree_classes()

        @pytree.pytree_dataclass(data=("value",))
        class Payload:
            value: int

        self.assertTrue(isinstance(before, tuple), "isinstance(before, tuple)")
        self.assertTrue(
            registered_pytree_classes() == (*before, Payload),
            "registered_pytree_classes() == (*before, Payload)",
        )
        self.assertTrue(Payload not in before, "Payload not in before")

    def test_duplicate_partition_names_are_rejected_without_registering(self):
        """Duplicate partition names fail before registration and leave the payload
        opaque to JAX."""
        for data, meta, partition in (
            (("value", "value"), ("mode",), "data"),
            (("value",), ("mode", "mode"), "meta"),
        ):
            with self.subTest(data=data, meta=meta, partition=partition), self.case():
                self._case_duplicate_partition_names_are_rejected_without_registering(
                    data, meta, partition
                )

    def _case_duplicate_partition_names_are_rejected_without_registering(
        self, data, meta, partition
    ):
        """Run one row with case-local objects released before runtime cleanup."""

        class Payload:
            value: int
            mode: str

        before = registered_pytree_classes()
        with self.assertRaisesRegex(ValueError, f"duplicate names in {partition}"):
            pytree_dataclass(data=data, meta=meta)(Payload)
        self.assertTrue(
            registered_pytree_classes() == before,
            "registered_pytree_classes() == before",
        )
        payload = cast(Callable[..., Payload], Payload)(value=1, mode="test")
        self.assertTrue(
            jax.tree_util.tree_leaves(payload) == [payload],
            "jax.tree_util.tree_leaves(payload) == [payload]",
        )

    def test_undecorated_subclass_is_frozen_and_all_fields_round_trip(self):
        """The dataclass helper freezes inherited and new fields and reconstructs in
        declared data order."""

        @dataclass(frozen=True)
        class Base:
            value: int

        @pytree.pytree_dataclass(data=("extra", "value"))
        class Child(Base):
            extra: int = 2

        child = Child(value=1, extra=3)
        with self.assertRaises(FrozenInstanceError):
            setattr(child, "extra", 9)
        with self.assertRaises(FrozenInstanceError):
            setattr(child, "unregistered", 9)
        leaves, treedef = jax.tree_util.tree_flatten(child)
        self.assertTrue(leaves == [3, 1], "leaves == [3, 1]")
        self.assertTrue(
            jax.tree_util.tree_unflatten(treedef, [5, 4]) == Child(value=4, extra=5),
            "jax.tree_util.tree_unflatten(treedef, [5, 4]) == Child(value=4, extra=5)",
        )
        self.assertTrue(
            Child in registered_pytree_classes(), "Child in registered_pytree_classes()"
        )

    def test_undecorated_subclass_cannot_omit_its_new_fields(self):
        """Subclass registration rejects a partition that omits newly declared fields."""

        @dataclass(frozen=True)
        class Base:
            value: int

        class Child(Base):
            extra: int = 2

        before = registered_pytree_classes()
        with self.assertRaisesRegex(ValueError, "missing=\\['extra'\\]"):
            pytree_dataclass(data=("value",))(Child)
        self.assertTrue(
            registered_pytree_classes() == before,
            "registered_pytree_classes() == before",
        )

    def test_custom_node_preserves_sealed_constructor_and_ordered_reconstruction(self):
        """Custom-node registration preserves sealed construction and leaf order through
        reconstruction, JIT and gradients."""
        before = registered_pytree_classes()

        @dataclass(frozen=True, slots=True, init=False)
        class Payload:
            first: object
            second: object
            mode: str

            def __init__(self, *args, **kwargs):
                raise RuntimeError("producer only")

            def tree_flatten(self):
                return (self.second, self.first), self.mode

            @classmethod
            def tree_unflatten(cls, mode, children):
                second, first = children
                instance = object.__new__(cls)
                object.__setattr__(instance, "first", first)
                object.__setattr__(instance, "second", second)
                object.__setattr__(instance, "mode", mode)
                return instance

        self.assertTrue(
            pytree_node(Payload) is Payload, "pytree_node(Payload) is Payload"
        )
        self.assertTrue(
            registered_pytree_classes() == (*before, Payload),
            "registered_pytree_classes() == (*before, Payload)",
        )
        with self.assertRaisesRegex(RuntimeError, "producer only"):
            Payload(1, 2, "sum")
        payload = Payload.tree_unflatten("sum", (jnp.asarray(2.0), jnp.asarray(1.0)))
        leaves, treedef = jax.tree_util.tree_flatten(payload)
        self.assertTrue(
            [float(value) for value in leaves] == [2.0, 1.0],
            "[float(value) for value in leaves] == [2.0, 1.0]",
        )
        restored = jax.tree_util.tree_unflatten(treedef, leaves)
        self.assertTrue(restored.mode == "sum", 'restored.mode == "sum"')
        with self.assertRaises(FrozenInstanceError):
            restored.first = 3
        self.assertTrue(
            not hasattr(restored, "__dict__"), 'not hasattr(restored, "__dict__")'
        )

        def weighted_sum(value):
            return value.first + 2 * value.second

        self.assertTrue(
            float(jit(weighted_sum)(restored)) == 5.0,
            "float(jit(weighted_sum)(restored)) == 5.0",
        )
        gradient = jax.grad(weighted_sum)(restored)
        self.assertTrue(gradient.mode == "sum", 'gradient.mode == "sum"')
        self.assertTrue(
            [float(value) for value in jax.tree_util.tree_leaves(gradient)]
            == [2.0, 1.0],
            "[float(value) for value in jax.tree_util.tree_leaves(gradient)] == [2.0, 1.0]",
        )

    def test_custom_node_rejects_mutable_dataclass_without_registering(self):
        """Custom-node registration rejects mutable dataclasses without altering the
        registry."""

        @dataclass
        class Mutable:
            value: int

            def tree_flatten(self):
                return (self.value,), None

            @classmethod
            def tree_unflatten(cls, _meta, children):
                return cls(*children)

        before = registered_pytree_classes()
        with self.assertRaisesRegex(TypeError, "Mutable must be a frozen dataclass"):
            pytree_node(Mutable)
        self.assertTrue(
            registered_pytree_classes() == before,
            "registered_pytree_classes() == before",
        )
        payload = Mutable(1)
        self.assertTrue(
            jax.tree_util.tree_leaves(payload) == [payload],
            "jax.tree_util.tree_leaves(payload) == [payload]",
        )

    def test_custom_node_freezes_undecorated_subclass_and_keeps_custom_leaf_order(self):
        """Custom-node registration freezes subclasses while preserving their custom
        flatten/unflatten order."""

        @dataclass(frozen=True)
        class Base:
            first: int

        @pytree.pytree_node
        class Payload(Base):
            second: int

            def tree_flatten(self):
                return (self.second, self.first), None

            @classmethod
            def tree_unflatten(cls, _meta, children):
                second, first = children
                return cls(first=first, second=second)

        payload = Payload(first=1, second=2)
        with self.assertRaises(FrozenInstanceError):
            setattr(payload, "second", 3)
        leaves, treedef = jax.tree_util.tree_flatten(payload)
        self.assertTrue(leaves == [2, 1], "leaves == [2, 1]")
        self.assertTrue(
            jax.tree_util.tree_unflatten(treedef, [4, 3]) == Payload(first=3, second=4),
            "jax.tree_util.tree_unflatten(treedef, [4, 3]) == Payload(first=3, second=4)",
        )

    def test_dataclass_reconstruction_calls_post_init(self):
        """Dataclass pytree reconstruction invokes post-init for the reconstructed field
        values."""
        constructed = []

        @pytree.pytree_dataclass(data=("value",))
        class Payload:
            value: int

            def __post_init__(self):
                constructed.append(self.value)

        leaves, treedef = jax.tree_util.tree_flatten(Payload(1))
        self.assertTrue(leaves == [1], "leaves == [1]")
        self.assertTrue(
            jax.tree_util.tree_unflatten(treedef, [2]).value == 2,
            "jax.tree_util.tree_unflatten(treedef, [2]).value == 2",
        )
        self.assertTrue(constructed == [1, 2], "constructed == [1, 2]")


# ── Coil sets ─────────────────────────────────────────────────────────


class TestGroupedFieldWithoutCoils(JaxTestCase):
    """A coil set without coils has no field, whatever form the points take."""

    def test_empty_coil_set_gives_zero_field_for_list_points(self):
        """B, A and their derivatives vanish, shaped (P, 3, ...), for list points."""
        points = [[1.0, 0.0, 0.1], [0.9, 0.2, -0.1]]
        spec = make_grouped_coil_set_spec(())
        rows = [
            ("B", grouped_biot_savart_B_from_spec, (2, 3)),
            ("A", grouped_biot_savart_A_from_spec, (2, 3)),
            ("dB", grouped_biot_savart_dB_by_dX_from_spec, (2, 3, 3)),
            ("d2B", grouped_biot_savart_d2B_by_dXdX_from_spec, (2, 3, 3, 3)),
        ]
        for name, kernel, shape in rows:
            with self.subTest(name):
                np.testing.assert_array_equal(kernel(points, spec), np.zeros(shape))
        B, dB = grouped_biot_savart_B_and_dB_from_spec(points, spec)
        np.testing.assert_array_equal(B, np.zeros((2, 3)))
        np.testing.assert_array_equal(dB, np.zeros((2, 3, 3)))


class TestCurrentExpressionExtraction(JaxTestCase):
    """Coil specs rebuilt from DOFs keep current expressions, scaled currents and
    shared symmetric geometry."""

    def test_coil_specs_preserve_fixed_total_current_expression(self):
        """Reconstruction preserves a fixed total-current constraint and opposite term
        derivatives."""
        curves = create_equally_spaced_curves(
            2,
            1,
            stellsym=False,
            R0=1.0,
            R1=0.25,
            order=2,
            numquadpoints=12,
        )
        for curve in curves:
            curve.fix_all()
        first = Current(3.0)
        total = Current(10.0)
        total.fix_all()
        field = JaxBiotSavart(
            [
                Coil(curves[0], first),
                Coil(curves[1], total - first),
            ]
        )

        def currents(owner_dofs: jax.Array) -> jax.Array:
            return jnp.stack(
                tuple(
                    coil.current.value[0]
                    for coil in field.coil_specs_from_dofs(owner_dofs)
                )
            )

        owner_dofs = jnp.asarray([4.0], dtype=jnp.float64)
        np.testing.assert_allclose(currents(owner_dofs), np.asarray([4.0, 6.0]))
        np.testing.assert_allclose(
            jax.jacrev(currents)(owner_dofs),
            np.asarray([[1.0], [-1.0]]),
        )

    def test_coil_set_spec_preserves_scaled_current_optimizer_coordinates(self):
        """Scaled physical currents retain optimizer coordinates and the corresponding
        Jacobian scale."""
        curve = create_equally_spaced_curves(
            1,
            1,
            stellsym=False,
            R0=1.0,
            R1=0.25,
            order=2,
            numquadpoints=12,
        )[0]
        curve.fix_all()
        field = JaxBiotSavart([Coil(curve, Current(1.0) * 1.0e5)])

        def current(owner_dofs: jax.Array) -> jax.Array:
            return field.coil_set_spec_from_dofs(owner_dofs).groups[0].currents

        owner_dofs = jnp.asarray(field.x, dtype=jnp.float64)
        np.testing.assert_array_equal(owner_dofs, np.asarray([1.0]))
        np.testing.assert_allclose(current(owner_dofs), np.asarray([1.0e5]))
        np.testing.assert_allclose(
            jax.jacrev(current)(owner_dofs),
            np.asarray([[1.0e5]]),
        )

    def test_coil_specs_reuse_geometry_for_symmetry_replicas(self):
        """Symmetry replicas share one reconstructed immutable curve payload."""
        curve = create_equally_spaced_curves(
            1,
            2,
            stellsym=True,
            R0=1.0,
            R1=0.25,
            order=2,
            numquadpoints=12,
        )[0]
        current = Current(3.0)
        coils = coils_via_symmetries([curve], [current], 2, True)
        field = JaxBiotSavart(coils)

        coil_specs = field.coil_specs_from_dofs(jnp.asarray(field.x))

        self.assertTrue(len(coil_specs) == 4, "len(coil_specs) == 4")
        self.assertTrue(
            len({id(coil_spec.curve) for coil_spec in coil_specs}) == 1,
            "len({id(coil_spec.curve) for coil_spec in coil_specs}) == 1",
        )
