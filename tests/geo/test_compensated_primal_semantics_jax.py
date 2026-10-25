"""IEEE special-value and finite-difference checks for the compensated JAX surface primal."""

from __future__ import annotations

from collections.abc import Callable
from unittest import SkipTest

from unittest_jax_support import JAX_IMPORT_ERROR, JaxTestCase, host_array

if JAX_IMPORT_ERROR is not None:
    raise SkipTest(JAX_IMPORT_ERROR)

import jax
import jax.numpy as jnp
import numpy as np
from simsopt_jax.core.specs import SurfaceXYZTensorFourierSpec
from simsopt_jax.core.surface_fourier_series import _compensated_matmul
from simsopt_jax.core.surface_geometry import surface_gamma

LARGEST = float.fromhex("0x1.fffffffffffffp+1023")

_LITERAL_CASES = (
    ("positive-sum-overflow", [[LARGEST, LARGEST]], [[1.0], [1.0]], np.inf),
    ("negative-sum-overflow", [[-LARGEST, -LARGEST]], [[1.0], [1.0]], -np.inf),
    ("positive-product-overflow", [[LARGEST]], [[2.0]], np.inf),
    ("negative-product-overflow", [[-LARGEST]], [[2.0]], -np.inf),
    ("nan-input", [[np.nan]], [[1.0]], np.nan),
    ("opposite-infinities", [[np.inf, -np.inf]], [[1.0], [1.0]], np.nan),
    ("positive-infinity", [[np.inf]], [[1.0]], np.inf),
    ("negative-infinity", [[-np.inf]], [[1.0]], -np.inf),
    (
        "finite-cancellation",
        [[float(2**54), 1.0, -float(2**54)]],
        [[1.0], [1.0], [1.0]],
        1.0,
    ),
)
_PUBLIC_CASES = (
    ("positive-overflow", [LARGEST, LARGEST, 0.0, 0.0, 0.0], np.inf),
    ("negative-overflow", [-LARGEST, -LARGEST, 0.0, 0.0, 0.0], -np.inf),
    ("cancellation", [float(2**54), 1.0, -float(2**54), 0.0, 0.0], 1.0),
    ("finite-control", [2.0, 3.0, 4.0, 0.0, 0.0], 9.0),
)


def _cancellation_terms(value: jax.Array, pad: int = 0) -> jax.Array:
    """Row ``[2**54, value**2, -2**54]`` followed by ``pad`` zeros."""
    return jnp.stack(
        (
            jnp.asarray(2**54, dtype=jnp.float64),
            value * value,
            jnp.asarray(-(2**54), dtype=jnp.float64),
            *(jnp.zeros_like(value) for _ in range(pad)),
        )
    )[None, :]


def _polynomial(value: jax.Array) -> jax.Array:
    ones = jnp.ones((3, 1), dtype=jnp.float64)
    return _compensated_matmul(_cancellation_terms(value), ones)[0, 0]


def _tensor_spec(zcs: jax.Array) -> SurfaceXYZTensorFourierSpec:
    # At phi=theta=0 all three cosine modes equal one; sine modes vanish.
    zeros = jnp.zeros((1, 5), dtype=jnp.float64)
    points = jnp.zeros((1,), dtype=jnp.float64)
    return SurfaceXYZTensorFourierSpec(
        xcs=zeros,
        ycs=zeros,
        zcs=zcs,
        quadpoints_phi=points,
        quadpoints_theta=points,
        nfp=1,
        stellsym=False,
        clamped_dims=(False, False, False),
    )


def _public_polynomial(value: jax.Array) -> jax.Array:
    return surface_gamma(_tensor_spec(_cancellation_terms(value, pad=2)))[0, 0, 2]


class CompensatedPrimalSemanticsTests(JaxTestCase):
    def _assert_fp64(self, *arrays: jax.Array) -> None:
        self.assertTrue(jax.config.x64_enabled)
        for array in arrays:
            self.assertEqual(array.dtype, np.dtype(np.float64))

    def test_literal_product_semantics(self) -> None:
        for name, left, right, expected in _LITERAL_CASES:
            with self.subTest(case=name), self.case():
                self._check_literal_product(left, right, expected)

    def _check_literal_product(
        self, left: list[list[float]], right: list[list[float]], expected: float
    ) -> None:
        # Oracles are literal IEEE classes and exact integer cancellation.
        left_array = jnp.asarray(left, dtype=jnp.float64)
        right_array = jnp.asarray(right, dtype=jnp.float64)
        self._assert_fp64(left_array, right_array)
        actual = jax.jit(_compensated_matmul)(left_array, right_array)
        self._assert_fp64(actual)
        np.testing.assert_array_equal(host_array(actual), [[expected]])

    def test_nonzero_helper_jvp_reverse_and_hvp(self) -> None:
        self._check_polynomial(_polynomial)

    def _check_polynomial(self, polynomial: Callable[[jax.Array], jax.Array]) -> None:
        # Exact real polynomial v**2 gives value/JVP/gradient/HVP 9,6,6,2 at 3.
        value = jnp.asarray(3.0, dtype=jnp.float64)
        self._assert_fp64(value)
        primal, tangent = jax.jit(
            lambda v: jax.jvp(polynomial, (v,), (jnp.ones_like(v),))
        )(value)
        gradient, hvp = jax.jit(
            lambda v: jax.jvp(jax.grad(polynomial), (v,), (jnp.ones_like(v),))
        )(value)
        self._assert_fp64(primal, tangent, gradient, hvp)
        np.testing.assert_array_equal(
            [host_array(x) for x in (primal, tangent, gradient, hvp)],
            [9.0, 6.0, 6.0, 2.0],
        )

    def test_public_tensor_gamma_semantics(self) -> None:
        for name, coefficients, expected_z in _PUBLIC_CASES:
            with self.subTest(case=name), self.case():
                self._check_public_tensor(coefficients, expected_z)

    def _check_public_tensor(
        self, coefficients: list[float], expected_z: float
    ) -> None:
        values = jnp.asarray([coefficients], dtype=jnp.float64)
        self._assert_fp64(values)
        actual = surface_gamma(_tensor_spec(values))
        self._assert_fp64(actual)
        np.testing.assert_array_equal(host_array(actual), [[[0.0, 0.0, expected_z]]])

    def test_nonzero_public_tensor_jvp_reverse_and_hvp(self) -> None:
        self._check_polynomial(_public_polynomial)
