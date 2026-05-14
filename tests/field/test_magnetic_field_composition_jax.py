"""Parity and fail-fast tests for N09 — magnetic field composition closure.

Covers:

* ``MagneticFieldSum([JAX, JAX])`` parity vs the CPU ``MagneticFieldSum``
  oracle at the ``direct_kernel`` parity-ladder lane.
* ``MagneticFieldMultiply(scalar, JAX)`` parity vs the CPU
  ``MagneticFieldMultiply`` oracle at the same lane.
* dB_by_dX / A / dA_by_dX composition coverage where supported by the
  underlying child class.
* Strict-mode rejection of any composition that contains a CPU-only
  child (``ToroidalField``, ``PoloidalField``, etc.) when the JAX
  backend runtime is configured with ``strict=True``.
* ``simsopt.jax_core.magneticfield_composition`` pure-JAX primitives
  (``compose_B_sum``, ``compose_B_scaled``, …) produce the same on-device
  result as the per-child :func:`toroidal_B` / :func:`poloidal_B` /
  :func:`mirror_B` kernels invoked individually.
* JAX transfer-guard ``disallow`` compatibility: composing JAX fields
  under ``jax.transfer_guard("disallow")`` does not trigger implicit
  host-to-device transfers beyond the explicit ``jax.device_put`` in
  the per-child kernels.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

from benchmarks.validation_ladder_contract import parity_ladder_tolerances
from simsopt.backend import (
    get_backend_config,
    invalidate_backend_cache,
    is_backend_strict,
    is_jax_backend,
)
from simsopt.field import (
    MirrorModel,
    PoloidalField,
    ToroidalField,
)
from simsopt.field.magneticfield import (
    MagneticFieldMultiply,
    MagneticFieldSum,
)
from simsopt.field.toroidal_field_jax import ToroidalFieldJAX
from simsopt.field.poloidal_field_jax import PoloidalFieldJAX
from simsopt.field.mirror_model_jax import MirrorModelJAX
from simsopt.jax_core.analytic_pure_fields import (
    ToroidalFieldSpec,
    toroidal_A,
    toroidal_B,
    toroidal_dA,
    toroidal_dB,
    toroidal_d2B,
)
from simsopt.jax_core.magneticfield_composition import (
    compose_A_scaled,
    compose_A_sum,
    compose_B_scaled,
    compose_B_sum,
    compose_dA_scaled,
    compose_dA_sum,
    compose_dB_scaled,
    compose_dB_sum,
    compose_d2A_scaled,
    compose_d2A_sum,
    compose_d2B_scaled,
    compose_d2B_sum,
)


_DIRECT_KERNEL = parity_ladder_tolerances("direct_kernel")
_RTOL = _DIRECT_KERNEL["rtol"]
_ATOL = _DIRECT_KERNEL["atol"]


@pytest.fixture
def relax_backend_strict_mode(monkeypatch):
    """Temporarily clear strict-JAX backend env vars for CPU oracle construction.

    The parity tests must build CPU ``MagneticFieldSum`` /
    ``MagneticFieldMultiply`` instances to compare against the JAX
    composition output. Under the production strict-JAX gate
    (``SIMSOPT_BACKEND_STRICT=1`` + ``SIMSOPT_BACKEND_MODE=jax_*_parity``),
    the JAX strict guard installed in ``magneticfield.py`` blocks any
    CPU child at composition construction, including the CPU oracle
    objects these tests need.

    This fixture clears the strict env so the CPU oracle objects can be
    constructed. The strict-mode fail-fast contract is exercised by the
    dedicated ``TestStrictJAXModeFailFast`` class in this file, not by
    the parity tests.
    """
    monkeypatch.delenv("SIMSOPT_BACKEND_STRICT", raising=False)
    monkeypatch.delenv("SIMSOPT_BACKEND_MODE", raising=False)
    monkeypatch.delenv("SIMSOPT_BACKEND", raising=False)
    monkeypatch.delenv("STAGE2_BACKEND", raising=False)
    invalidate_backend_cache()
    yield
    invalidate_backend_cache()


def _production_points(seed: int, count: int = 60) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    points = np.zeros((count, 3), dtype=np.float64)
    points[:, 0] = rng.uniform(0.4, 1.8, size=count)
    points[:, 1] = rng.uniform(0.4, 1.8, size=count)
    points[:, 2] = rng.uniform(-0.5, 0.5, size=count)
    return np.ascontiguousarray(points)


class _D2AJAXTestField(ToroidalFieldJAX):
    """JAX-native test field for the public d2A composition surface."""

    def __init__(self, scale: float):
        super().__init__(R0=1.0, B0=1.0)
        self.scale = float(scale)

    def _d2A_by_dXdX_impl(self, ddA):
        points = self.get_points_cart_ref()
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        ddA[:] = self.scale * np.stack(
            (
                np.stack(
                    (
                        np.stack((x + y, x - y, z), axis=1),
                        np.stack((2.0 * x, y + z, x * z), axis=1),
                        np.stack((z - y, x + z, y * z), axis=1),
                    ),
                    axis=1,
                ),
                np.stack(
                    (
                        np.stack((x * y, y - z, x + 1.0), axis=1),
                        np.stack((y + 2.0, z + 3.0, x - 4.0), axis=1),
                        np.stack((x * x, y * y, z * z), axis=1),
                    ),
                    axis=1,
                ),
                np.stack(
                    (
                        np.stack((x - z, y + z, x + y + z), axis=1),
                        np.stack((x * z, y * z, x * y), axis=1),
                        np.stack((x + 5.0, y + 6.0, z + 7.0), axis=1),
                    ),
                    axis=1,
                ),
            ),
            axis=1,
        )


# ── MagneticFieldSum parity ──────────────────────────────────────────


@pytest.mark.usefixtures("relax_backend_strict_mode")
class TestMagneticFieldSumJAXParity:
    def test_sum_two_jax_toroidal_fields_matches_cpu_oracle(self):
        """``Sum([ToroidalFieldJAX, ToroidalFieldJAX])`` matches CPU
        ``Sum([ToroidalField, ToroidalField])`` at the ``direct_kernel``
        parity-ladder lane.
        """
        points = _production_points(seed=901, count=60)
        jax1 = ToroidalFieldJAX(R0=1.3, B0=0.8)
        jax2 = ToroidalFieldJAX(R0=1.5, B0=0.4)
        cpu1 = ToroidalField(R0=1.3, B0=0.8)
        cpu2 = ToroidalField(R0=1.5, B0=0.4)
        sum_jax = MagneticFieldSum([jax1, jax2])
        sum_cpu = MagneticFieldSum([cpu1, cpu2])
        sum_jax.set_points_cart(points)
        sum_cpu.set_points_cart(points)
        np.testing.assert_allclose(
            np.asarray(sum_jax.B()),
            np.asarray(sum_cpu.B()),
            rtol=_RTOL,
            atol=_ATOL,
        )

    def test_sum_two_jax_toroidal_dB_matches_cpu_oracle(self):
        """Composition dB_by_dX matches the CPU oracle."""
        points = _production_points(seed=902, count=60)
        jax1 = ToroidalFieldJAX(R0=1.3, B0=0.8)
        jax2 = ToroidalFieldJAX(R0=1.5, B0=0.4)
        cpu1 = ToroidalField(R0=1.3, B0=0.8)
        cpu2 = ToroidalField(R0=1.5, B0=0.4)
        sum_jax = MagneticFieldSum([jax1, jax2])
        sum_cpu = MagneticFieldSum([cpu1, cpu2])
        sum_jax.set_points_cart(points)
        sum_cpu.set_points_cart(points)
        np.testing.assert_allclose(
            np.asarray(sum_jax.dB_by_dX()),
            np.asarray(sum_cpu.dB_by_dX()),
            rtol=_RTOL,
            atol=_ATOL,
        )

    def test_sum_two_jax_toroidal_d2B_matches_cpu_oracle(self):
        """Composition d2B_by_dXdX matches the CPU oracle."""
        points = _production_points(seed=903, count=60)
        jax1 = ToroidalFieldJAX(R0=1.3, B0=0.8)
        jax2 = ToroidalFieldJAX(R0=1.5, B0=0.4)
        cpu1 = ToroidalField(R0=1.3, B0=0.8)
        cpu2 = ToroidalField(R0=1.5, B0=0.4)
        sum_jax = MagneticFieldSum([jax1, jax2])
        sum_cpu = MagneticFieldSum([cpu1, cpu2])
        sum_jax.set_points_cart(points)
        sum_cpu.set_points_cart(points)
        np.testing.assert_allclose(
            np.asarray(sum_jax.d2B_by_dXdX()),
            np.asarray(sum_cpu.d2B_by_dXdX()),
            rtol=_RTOL,
            atol=_ATOL,
        )

    def test_sum_two_jax_toroidal_A_dA_matches_cpu_oracle(self):
        """Composition A and dA_by_dX match the CPU oracle."""
        points = _production_points(seed=904, count=60)
        jax1 = ToroidalFieldJAX(R0=1.3, B0=0.8)
        jax2 = ToroidalFieldJAX(R0=1.5, B0=0.4)
        cpu1 = ToroidalField(R0=1.3, B0=0.8)
        cpu2 = ToroidalField(R0=1.5, B0=0.4)
        sum_jax = MagneticFieldSum([jax1, jax2])
        sum_cpu = MagneticFieldSum([cpu1, cpu2])
        sum_jax.set_points_cart(points)
        sum_cpu.set_points_cart(points)
        np.testing.assert_allclose(
            np.asarray(sum_jax.A()),
            np.asarray(sum_cpu.A()),
            rtol=_RTOL,
            atol=_ATOL,
        )
        np.testing.assert_allclose(
            np.asarray(sum_jax.dA_by_dX()),
            np.asarray(sum_cpu.dA_by_dX()),
            rtol=_RTOL,
            atol=_ATOL,
        )

    def test_sum_two_jax_native_d2A_matches_child_sum(self):
        """Public ``MagneticFieldSum.d2A_by_dXdX`` sums JAX-native children."""
        points = _production_points(seed=907, count=60)
        child1 = _D2AJAXTestField(scale=1.25)
        child2 = _D2AJAXTestField(scale=-0.5)
        sum_jax = MagneticFieldSum([child1, child2])
        sum_jax.set_points_cart(points)
        child1.set_points_cart(points)
        child2.set_points_cart(points)
        expected = child1.d2A_by_dXdX() + child2.d2A_by_dXdX()
        np.testing.assert_allclose(
            np.asarray(sum_jax.d2A_by_dXdX()),
            np.asarray(expected),
            rtol=_RTOL,
            atol=_ATOL,
        )

    def test_sum_three_jax_children_matches_cpu_oracle(self):
        """Sum of three native JAX children (mixed kinds where possible)."""
        R0_axis = 1.4
        # Use points away from R = R0 to avoid PoloidalField singularity.
        candidates = _production_points(seed=905, count=240)
        R_xy = np.sqrt(candidates[:, 0] ** 2 + candidates[:, 1] ** 2)
        points = np.ascontiguousarray(candidates[np.abs(R_xy - R0_axis) > 0.2])
        assert points.shape[0] >= 50, "production-scale floor"
        jax_children = [
            ToroidalFieldJAX(R0=1.3, B0=0.8),
            PoloidalFieldJAX(R0=R0_axis, B0=0.5, q=1.2),
            ToroidalFieldJAX(R0=1.5, B0=0.4),
        ]
        cpu_children = [
            ToroidalField(R0=1.3, B0=0.8),
            PoloidalField(R0=R0_axis, B0=0.5, q=1.2),
            ToroidalField(R0=1.5, B0=0.4),
        ]
        sum_jax = MagneticFieldSum(jax_children)
        sum_cpu = MagneticFieldSum(cpu_children)
        sum_jax.set_points_cart(points)
        sum_cpu.set_points_cart(points)
        np.testing.assert_allclose(
            np.asarray(sum_jax.B()),
            np.asarray(sum_cpu.B()),
            rtol=_RTOL,
            atol=_ATOL,
        )
        np.testing.assert_allclose(
            np.asarray(sum_jax.dB_by_dX()),
            np.asarray(sum_cpu.dB_by_dX()),
            rtol=_RTOL,
            atol=_ATOL,
        )

    def test_sum_operator_overload_matches_explicit_sum(self):
        """The ``+`` operator on JAX fields routes through MagneticFieldSum."""
        points = _production_points(seed=906, count=60)
        jax1 = ToroidalFieldJAX(R0=1.3, B0=0.8)
        jax2 = ToroidalFieldJAX(R0=1.5, B0=0.4)
        explicit = MagneticFieldSum([jax1, jax2])
        chained = jax1 + jax2
        explicit.set_points_cart(points)
        chained.set_points_cart(points)
        np.testing.assert_allclose(
            np.asarray(chained.B()),
            np.asarray(explicit.B()),
            rtol=_RTOL,
            atol=_ATOL,
        )


# ── MagneticFieldMultiply parity ─────────────────────────────────────


@pytest.mark.usefixtures("relax_backend_strict_mode")
class TestMagneticFieldMultiplyJAXParity:
    def test_multiply_scalar_jax_toroidal_matches_cpu_oracle(self):
        """``Multiply(scalar, ToroidalFieldJAX)`` matches CPU."""
        points = _production_points(seed=910, count=60)
        jax_ = ToroidalFieldJAX(R0=1.3, B0=0.8)
        cpu = ToroidalField(R0=1.3, B0=0.8)
        mul_jax = MagneticFieldMultiply(2.5, jax_)
        mul_cpu = MagneticFieldMultiply(2.5, cpu)
        mul_jax.set_points_cart(points)
        mul_cpu.set_points_cart(points)
        np.testing.assert_allclose(
            np.asarray(mul_jax.B()),
            np.asarray(mul_cpu.B()),
            rtol=_RTOL,
            atol=_ATOL,
        )
        np.testing.assert_allclose(
            np.asarray(mul_jax.dB_by_dX()),
            np.asarray(mul_cpu.dB_by_dX()),
            rtol=_RTOL,
            atol=_ATOL,
        )
        np.testing.assert_allclose(
            np.asarray(mul_jax.d2B_by_dXdX()),
            np.asarray(mul_cpu.d2B_by_dXdX()),
            rtol=_RTOL,
            atol=_ATOL,
        )
        np.testing.assert_allclose(
            np.asarray(mul_jax.A()),
            np.asarray(mul_cpu.A()),
            rtol=_RTOL,
            atol=_ATOL,
        )
        np.testing.assert_allclose(
            np.asarray(mul_jax.dA_by_dX()),
            np.asarray(mul_cpu.dA_by_dX()),
            rtol=_RTOL,
            atol=_ATOL,
        )

    def test_multiply_jax_native_d2A_matches_child_scaled(self):
        """Public ``MagneticFieldMultiply.d2A_by_dXdX`` scales a JAX child."""
        points = _production_points(seed=912, count=60)
        child = _D2AJAXTestField(scale=1.25)
        mul_jax = MagneticFieldMultiply(2.5, child)
        mul_jax.set_points_cart(points)
        child.set_points_cart(points)
        expected = 2.5 * child.d2A_by_dXdX()
        np.testing.assert_allclose(
            np.asarray(mul_jax.d2A_by_dXdX()),
            np.asarray(expected),
            rtol=_RTOL,
            atol=_ATOL,
        )

    def test_multiply_rmul_operator_matches_explicit_multiply(self):
        """``scalar * jax_field`` and ``jax_field * scalar`` both route through
        ``MagneticFieldMultiply`` and reproduce the explicit form."""
        points = _production_points(seed=911, count=60)
        jax_ = ToroidalFieldJAX(R0=1.3, B0=0.8)
        explicit = MagneticFieldMultiply(2.5, jax_)
        rmul = 2.5 * jax_
        lmul = jax_ * 2.5
        explicit.set_points_cart(points)
        rmul.set_points_cart(points)
        lmul.set_points_cart(points)
        np.testing.assert_allclose(
            np.asarray(rmul.B()),
            np.asarray(explicit.B()),
            rtol=_RTOL,
            atol=_ATOL,
        )
        np.testing.assert_allclose(
            np.asarray(lmul.B()),
            np.asarray(explicit.B()),
            rtol=_RTOL,
            atol=_ATOL,
        )


# ── Strict-mode fail-fast guard ──────────────────────────────────────


def _strict_jax_env_active() -> bool:
    get_backend_config()  # warm cache from current env
    return is_jax_backend() and is_backend_strict()


def _evaluate_B(field, *, seed: int, count: int = 10) -> np.ndarray:
    field.set_points_cart(_production_points(seed=seed, count=count))
    return np.asarray(field.B())


@pytest.fixture
def enforce_strict_jax_mode(monkeypatch):
    """Force strict-JAX backend env vars and refresh the runtime cache.

    The strict-mode fail-fast contract is the load-bearing N09 guarantee;
    historically this class was ``skipif``-gated on the ambient env vars
    and silently skipped under the default test invocation, so the
    contract was unverified by ``pytest`` in CI. Using ``monkeypatch.setenv``
    plus :func:`invalidate_backend_cache` makes the guard fire
    deterministically regardless of how the test runner was invoked.
    """
    monkeypatch.setenv("SIMSOPT_BACKEND", "jax")
    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_cpu_parity")
    monkeypatch.setenv("SIMSOPT_BACKEND_STRICT", "1")
    monkeypatch.delenv("STAGE2_BACKEND", raising=False)
    invalidate_backend_cache()
    assert _strict_jax_env_active(), (
        "strict-JAX backend should be active after monkeypatch.setenv"
    )
    yield
    invalidate_backend_cache()


@pytest.mark.usefixtures("enforce_strict_jax_mode")
class TestStrictJAXModeFailFast:
    def test_sum_mixed_cpu_jax_raises_in_strict_jax_mode(self):
        """``MagneticFieldSum([JAX, CPU])`` must raise in strict JAX mode."""
        with pytest.raises(RuntimeError, match=r"MagneticFieldSum.*CPU-only"):
            MagneticFieldSum(
                [
                    ToroidalFieldJAX(R0=1.3, B0=0.8),
                    ToroidalField(R0=1.5, B0=0.4),
                ]
            )

    def test_sum_all_cpu_raises_in_strict_jax_mode(self):
        """All-CPU sum in strict JAX mode must also fail-fast."""
        with pytest.raises(RuntimeError, match=r"MagneticFieldSum.*CPU-only"):
            MagneticFieldSum(
                [
                    ToroidalField(R0=1.3, B0=0.8),
                    ToroidalField(R0=1.5, B0=0.4),
                ]
            )

    def test_multiply_cpu_raises_in_strict_jax_mode(self):
        """``Multiply(scalar, CPU)`` must raise in strict JAX mode."""
        with pytest.raises(RuntimeError, match=r"MagneticFieldMultiply.*CPU-only"):
            MagneticFieldMultiply(2.5, ToroidalField(R0=1.3, B0=0.8))

    def test_sum_all_jax_succeeds_in_strict_jax_mode(self):
        """Sentinel: all-JAX composition does NOT trip the guard."""
        sum_jax = MagneticFieldSum(
            [
                ToroidalFieldJAX(R0=1.3, B0=0.8),
                ToroidalFieldJAX(R0=1.5, B0=0.4),
            ]
        )
        _evaluate_B(sum_jax, seed=920)

    def test_nested_all_jax_sum_succeeds_in_strict_jax_mode(self):
        """An all-JAX composite remains JAX-native when nested."""
        chained = (
            ToroidalFieldJAX(R0=1.3, B0=0.8)
            + ToroidalFieldJAX(R0=1.5, B0=0.4)
            + ToroidalFieldJAX(R0=1.7, B0=0.2)
        )
        _evaluate_B(chained, seed=922)

    def test_multiply_jax_succeeds_in_strict_jax_mode(self):
        """Sentinel: pure-JAX multiply does NOT trip the guard."""
        mul = MagneticFieldMultiply(2.5, ToroidalFieldJAX(R0=1.3, B0=0.8))
        _evaluate_B(mul, seed=921)

    def test_nested_all_jax_multiply_succeeds_in_strict_jax_mode(self):
        """Multiplying an all-JAX composite remains in the JAX-native set."""
        inner = MagneticFieldSum(
            [
                ToroidalFieldJAX(R0=1.3, B0=0.8),
                ToroidalFieldJAX(R0=1.5, B0=0.4),
            ]
        )
        mul = MagneticFieldMultiply(2.5, inner)
        _evaluate_B(mul, seed=923)


# ── Pure JAX composition primitives ──────────────────────────────────


@pytest.mark.usefixtures("relax_backend_strict_mode")
class TestPureJAXCompositionPrimitives:
    def test_compose_B_sum_matches_individual_kernel_summation(self):
        """``compose_B_sum`` matches per-child kernel sum on device."""
        points_np = _production_points(seed=950, count=60)
        points = jax.device_put(points_np)
        spec1 = ToroidalFieldSpec(R0=1.3, B0=0.8)
        spec2 = ToroidalFieldSpec(R0=1.5, B0=0.4)
        children = (
            lambda pts, s=spec1: toroidal_B(s, pts),
            lambda pts, s=spec2: toroidal_B(s, pts),
        )
        composed = compose_B_sum(children, points)
        manual = toroidal_B(spec1, points) + toroidal_B(spec2, points)
        np.testing.assert_allclose(
            np.asarray(composed),
            np.asarray(manual),
            rtol=_RTOL,
            atol=_ATOL,
        )

    def test_compose_B_sum_matches_cpu_oracle(self):
        """``compose_B_sum`` matches the CPU ``MagneticFieldSum`` oracle."""
        points_np = _production_points(seed=951, count=60)
        points = jax.device_put(points_np)
        spec1 = ToroidalFieldSpec(R0=1.3, B0=0.8)
        spec2 = ToroidalFieldSpec(R0=1.5, B0=0.4)
        children = (
            lambda pts, s=spec1: toroidal_B(s, pts),
            lambda pts, s=spec2: toroidal_B(s, pts),
        )
        composed = compose_B_sum(children, points)
        sum_cpu = MagneticFieldSum(
            [ToroidalField(R0=1.3, B0=0.8), ToroidalField(R0=1.5, B0=0.4)]
        )
        sum_cpu.set_points_cart(points_np)
        np.testing.assert_allclose(
            np.asarray(composed),
            np.asarray(sum_cpu.B()),
            rtol=_RTOL,
            atol=_ATOL,
        )

    def test_compose_dB_sum_matches_cpu_oracle(self):
        """``compose_dB_sum`` matches the CPU ``MagneticFieldSum`` oracle."""
        points_np = _production_points(seed=952, count=60)
        points = jax.device_put(points_np)
        spec1 = ToroidalFieldSpec(R0=1.3, B0=0.8)
        spec2 = ToroidalFieldSpec(R0=1.5, B0=0.4)
        children = (
            lambda pts, s=spec1: toroidal_dB(s, pts),
            lambda pts, s=spec2: toroidal_dB(s, pts),
        )
        composed = compose_dB_sum(children, points)
        sum_cpu = MagneticFieldSum(
            [ToroidalField(R0=1.3, B0=0.8), ToroidalField(R0=1.5, B0=0.4)]
        )
        sum_cpu.set_points_cart(points_np)
        np.testing.assert_allclose(
            np.asarray(composed),
            np.asarray(sum_cpu.dB_by_dX()),
            rtol=_RTOL,
            atol=_ATOL,
        )

    def test_compose_d2B_sum_matches_cpu_oracle(self):
        """``compose_d2B_sum`` matches the CPU ``MagneticFieldSum`` oracle."""
        points_np = _production_points(seed=957, count=60)
        points = jax.device_put(points_np)
        spec1 = ToroidalFieldSpec(R0=1.3, B0=0.8)
        spec2 = ToroidalFieldSpec(R0=1.5, B0=0.4)
        children = (
            lambda pts, s=spec1: toroidal_d2B(s, pts),
            lambda pts, s=spec2: toroidal_d2B(s, pts),
        )
        composed = compose_d2B_sum(children, points)
        sum_cpu = MagneticFieldSum(
            [ToroidalField(R0=1.3, B0=0.8), ToroidalField(R0=1.5, B0=0.4)]
        )
        sum_cpu.set_points_cart(points_np)
        np.testing.assert_allclose(
            np.asarray(composed),
            np.asarray(sum_cpu.d2B_by_dXdX()),
            rtol=_RTOL,
            atol=_ATOL,
        )

    def test_compose_A_and_dA_sum_match_cpu_oracle(self):
        """``compose_A_sum`` and ``compose_dA_sum`` match CPU."""
        points_np = _production_points(seed=953, count=60)
        points = jax.device_put(points_np)
        spec1 = ToroidalFieldSpec(R0=1.3, B0=0.8)
        spec2 = ToroidalFieldSpec(R0=1.5, B0=0.4)
        A_children = (
            lambda pts, s=spec1: toroidal_A(s, pts),
            lambda pts, s=spec2: toroidal_A(s, pts),
        )
        dA_children = (
            lambda pts, s=spec1: toroidal_dA(s, pts),
            lambda pts, s=spec2: toroidal_dA(s, pts),
        )
        composed_A = compose_A_sum(A_children, points)
        composed_dA = compose_dA_sum(dA_children, points)
        sum_cpu = MagneticFieldSum(
            [ToroidalField(R0=1.3, B0=0.8), ToroidalField(R0=1.5, B0=0.4)]
        )
        sum_cpu.set_points_cart(points_np)
        np.testing.assert_allclose(
            np.asarray(composed_A),
            np.asarray(sum_cpu.A()),
            rtol=_RTOL,
            atol=_ATOL,
        )
        np.testing.assert_allclose(
            np.asarray(composed_dA),
            np.asarray(sum_cpu.dA_by_dX()),
            rtol=_RTOL,
            atol=_ATOL,
        )

    def test_compose_B_scaled_matches_cpu_oracle(self):
        """``compose_B_scaled`` matches the CPU ``MagneticFieldMultiply``."""
        points_np = _production_points(seed=954, count=60)
        points = jax.device_put(points_np)
        spec = ToroidalFieldSpec(R0=1.3, B0=0.8)
        child = lambda pts, s=spec: toroidal_B(s, pts)
        scaled = compose_B_scaled(child, 2.5, points)
        mul_cpu = MagneticFieldMultiply(2.5, ToroidalField(R0=1.3, B0=0.8))
        mul_cpu.set_points_cart(points_np)
        np.testing.assert_allclose(
            np.asarray(scaled),
            np.asarray(mul_cpu.B()),
            rtol=_RTOL,
            atol=_ATOL,
        )

    def test_compose_dB_dA_scaled_match_cpu_oracle(self):
        """``compose_dB_scaled``, ``compose_A_scaled``, ``compose_dA_scaled``
        all match the CPU ``MagneticFieldMultiply`` oracle.
        """
        points_np = _production_points(seed=955, count=60)
        points = jax.device_put(points_np)
        spec = ToroidalFieldSpec(R0=1.3, B0=0.8)
        dB_child = lambda pts, s=spec: toroidal_dB(s, pts)
        A_child = lambda pts, s=spec: toroidal_A(s, pts)
        dA_child = lambda pts, s=spec: toroidal_dA(s, pts)
        scaled_dB = compose_dB_scaled(dB_child, 2.5, points)
        scaled_A = compose_A_scaled(A_child, 2.5, points)
        scaled_dA = compose_dA_scaled(dA_child, 2.5, points)
        mul_cpu = MagneticFieldMultiply(2.5, ToroidalField(R0=1.3, B0=0.8))
        mul_cpu.set_points_cart(points_np)
        np.testing.assert_allclose(
            np.asarray(scaled_dB),
            np.asarray(mul_cpu.dB_by_dX()),
            rtol=_RTOL,
            atol=_ATOL,
        )
        np.testing.assert_allclose(
            np.asarray(scaled_A),
            np.asarray(mul_cpu.A()),
            rtol=_RTOL,
            atol=_ATOL,
        )
        np.testing.assert_allclose(
            np.asarray(scaled_dA),
            np.asarray(mul_cpu.dA_by_dX()),
            rtol=_RTOL,
            atol=_ATOL,
        )

    def test_compose_d2B_scaled_matches_cpu_oracle(self):
        """``compose_d2B_scaled`` matches the CPU ``MagneticFieldMultiply``."""
        points_np = _production_points(seed=958, count=60)
        points = jax.device_put(points_np)
        spec = ToroidalFieldSpec(R0=1.3, B0=0.8)
        child = lambda pts, s=spec: toroidal_d2B(s, pts)
        scaled = compose_d2B_scaled(child, 2.5, points)
        mul_cpu = MagneticFieldMultiply(2.5, ToroidalField(R0=1.3, B0=0.8))
        mul_cpu.set_points_cart(points_np)
        np.testing.assert_allclose(
            np.asarray(scaled),
            np.asarray(mul_cpu.d2B_by_dXdX()),
            rtol=_RTOL,
            atol=_ATOL,
        )

    def test_compose_d2A_sum_and_scaled_match_manual_device_result(self):
        """``compose_d2A_*`` sum and scale opaque child d2A tensors."""
        points = jax.device_put(_production_points(seed=959, count=60))

        def child(scale: float):
            return lambda pts: (
                scale
                * (
                    pts[:, :, None, None]
                    + 2.0 * pts[:, None, :, None]
                    - pts[:, None, None, :]
                )
            )

        child1 = child(1.25)
        child2 = child(-0.5)
        summed = compose_d2A_sum((child1, child2), points)
        scaled = compose_d2A_scaled(child1, 2.5, points)
        np.testing.assert_allclose(
            np.asarray(summed),
            np.asarray(child1(points) + child2(points)),
            rtol=_RTOL,
            atol=_ATOL,
        )
        np.testing.assert_allclose(
            np.asarray(scaled),
            np.asarray(2.5 * child1(points)),
            rtol=_RTOL,
            atol=_ATOL,
        )

    def test_compose_B_sum_rejects_zero_children(self):
        """The composition primitive must reject an empty child tuple."""
        # Pre-stage points to device with explicit jax.device_put so this
        # also runs under transfer_guard("disallow").
        points = jax.device_put(_production_points(seed=956, count=4))
        with pytest.raises(ValueError, match=r"at least one child"):
            compose_B_sum((), points)

    def test_compose_B_sum_rejects_malformed_points(self):
        """Points must have shape ``(N, 3)``."""
        spec = ToroidalFieldSpec(R0=1.3, B0=0.8)
        child = lambda pts, s=spec: toroidal_B(s, pts)
        bad_points = jax.device_put(np.zeros((4, 5), dtype=np.float64))
        with pytest.raises(ValueError, match=r"shape \(N, 3\)"):
            compose_B_sum((child,), bad_points)


# ── Transfer-guard compatibility ─────────────────────────────────────


class TestTransferGuardCompatibility:
    def test_compose_B_sum_under_disallow_transfer_guard(self):
        """The pure JAX composition primitive runs under
        ``jax.transfer_guard("disallow")`` without implicit transfers.
        """
        points_np = _production_points(seed=960, count=20)
        # Pre-stage points to device explicitly.
        points = jax.device_put(points_np)
        spec1 = ToroidalFieldSpec(R0=1.3, B0=0.8)
        spec2 = ToroidalFieldSpec(R0=1.5, B0=0.4)
        children = (
            lambda pts, s=spec1: toroidal_B(s, pts),
            lambda pts, s=spec2: toroidal_B(s, pts),
        )
        with jax.transfer_guard("disallow"):
            composed = compose_B_sum(children, points)
            # Forcing materialization in this scope is also allowed
            # because JAX device arrays remain on device.
            _ = composed.block_until_ready()


# ── Mixed-class JAX composition ──────────────────────────────────────


@pytest.mark.usefixtures("relax_backend_strict_mode")
class TestMixedJAXClassComposition:
    def test_sum_jax_toroidal_plus_mirror_matches_cpu_oracle(self):
        """``Sum([ToroidalFieldJAX, MirrorModelJAX])`` matches CPU."""
        points = _production_points(seed=970, count=60)
        # Points away from R=0 to avoid MirrorModel singularity.
        R_xy = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
        points = np.ascontiguousarray(points[R_xy > 0.3])
        assert points.shape[0] >= 50, "production-scale floor"
        sum_jax = MagneticFieldSum(
            [
                ToroidalFieldJAX(R0=1.3, B0=0.8),
                MirrorModelJAX(B0=2.5, gamma=0.4, Z_m=0.6),
            ]
        )
        sum_cpu = MagneticFieldSum(
            [
                ToroidalField(R0=1.3, B0=0.8),
                MirrorModel(B0=2.5, gamma=0.4, Z_m=0.6),
            ]
        )
        sum_jax.set_points_cart(points)
        sum_cpu.set_points_cart(points)
        np.testing.assert_allclose(
            np.asarray(sum_jax.B()),
            np.asarray(sum_cpu.B()),
            rtol=_RTOL,
            atol=_ATOL,
        )
