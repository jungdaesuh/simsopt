"""Item 21 parity tests for the JAX port of ``compute_on_axis_iota``.

These tests assert that:

1. The JAX kernel :func:`simsopt.jax_core.magnetic_axis_helpers.on_axis_iota_rk`
   reproduces the upstream CPU oracle
   :func:`simsopt.field.magnetic_axis_helpers.compute_on_axis_iota` on the
   three production-scale zoo configurations (``hsx``, ``ncsx``,
   ``giuliani``) at the ``derivative_heavy`` parity-ladder lane.

   The lane choice reflects the upstream contract: the CPU oracle runs an
   adaptive RK45 at ``rtol = atol = 1e-12``, and the JAX port runs a
   self-contained Dormand-Prince RK4(5) at the same tolerances. The two
   integrators agree on the scalar value but their floating-point traces
   diverge step-by-step, so a strict ``direct_kernel`` lane would not
   apply. ``derivative_heavy`` matches the lane intent of "smooth scalar
   produced by a strict-tolerance derivative-bearing kernel".

2. The pure axis kernel :func:`axis_position` reproduces the
   ``CurveRZFourier.gamma_impl`` evaluation at arbitrary ``phi`` values
   (production-scale grid of ``[0, 1/nfp]``) at ``direct_kernel`` lane.

3. The JAX kernel is JIT-able: a second invocation under
   :func:`jax.jit` produces the same scalar.

All tolerances are imported from
``benchmarks.validation_ladder_contract.parity_ladder_tolerances``; no
``rtol`` / ``atol`` literals appear inline in the test body.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks.validation_ladder_contract import parity_ladder_tolerances
from simsopt.configs.zoo import get_data
from simsopt.field.biotsavart_jax_backend import BiotSavartJAX
from simsopt.field.magnetic_axis_helpers import compute_on_axis_iota
from simsopt.jax_core.field import grouped_biot_savart_B_and_dB_from_spec
from simsopt.jax_core.analytic_pure_fields import (
    ToroidalFieldSpec,
    toroidal_B,
    toroidal_dB,
)
from simsopt.jax_core.magnetic_axis_helpers import (
    _first_eigenvalue_angle_2x2,
    axis_position,
    axis_position_and_tangent,
    on_axis_iota_rk,
    tangent_map_state_dim,
    tangent_map_y0,
)


_DIRECT_KERNEL = parity_ladder_tolerances("direct_kernel")
_DERIVATIVE_HEAVY = parity_ladder_tolerances("derivative_heavy")

_DIRECT_RTOL = _DIRECT_KERNEL["rtol"]
_DIRECT_ATOL = _DIRECT_KERNEL["atol"]
_HEAVY_SCALAR_RTOL = _DERIVATIVE_HEAVY["scalar_value_rtol"]
_HEAVY_SCALAR_ATOL = _DERIVATIVE_HEAVY["scalar_value_atol"]
_HEAVY_FIRST_DERIVATIVE_RTOL = _DERIVATIVE_HEAVY["first_derivative_rtol"]
_HEAVY_FIRST_DERIVATIVE_ATOL = _DERIVATIVE_HEAVY["first_derivative_atol"]


_ZOO_TARGETS = [
    ("hsx", 1.0418687161633922),
    ("ncsx", 0.39549339846119463),
    ("giuliani", 0.42297724084249616),
]


def _make_jax_field_eval(coils):
    coil_spec = BiotSavartJAX(coils).coil_set_spec()

    def field_eval_fn(points):
        return grouped_biot_savart_B_and_dB_from_spec(
            jnp.asarray(points, dtype=jnp.float64),
            coil_spec,
        )

    return field_eval_fn


# ── State convention ──────────────────────────────────────────────────


class TestTangentMapStateConvention:
    def test_state_dim_is_four(self):
        assert tangent_map_state_dim() == 4

    def test_y0_is_identity_flattened(self):
        y0 = np.asarray(tangent_map_y0())
        np.testing.assert_array_equal(y0, np.array([1.0, 0.0, 0.0, 1.0]))

    @pytest.mark.parametrize(
        "matrix",
        [
            np.array([[0.25, -1.5], [0.75, 0.25]], dtype=np.float64),
            np.array([[0.8, 0.2], [-0.4, 0.8]], dtype=np.float64),
            np.array([[2.0, 0.25], [0.0, 0.5]], dtype=np.float64),
        ],
    )
    def test_first_eigenvalue_angle_2x2_matches_jax_eig(self, matrix):
        matrix_jax = jnp.asarray(matrix, dtype=jnp.float64)
        evals, _ = jnp.linalg.eig(matrix_jax)
        expected = jnp.arctan2(jnp.imag(evals[0]), jnp.real(evals[0]))

        actual = _first_eigenvalue_angle_2x2(matrix_jax)

        np.testing.assert_allclose(
            np.asarray(actual),
            np.asarray(expected),
            rtol=_DIRECT_RTOL,
            atol=_DIRECT_ATOL,
        )

    def test_first_eigenvalue_angle_2x2_uses_closed_form_primitives(self):
        matrix = jnp.asarray(
            [[0.25, -1.5], [0.75, 0.25]],
            dtype=jnp.float64,
        )

        jaxpr = jax.make_jaxpr(_first_eigenvalue_angle_2x2)(matrix).jaxpr

        primitive_names = {eqn.primitive.name for eqn in jaxpr.eqns}
        assert "eig" not in primitive_names
        assert "atan2" in primitive_names


# ── axis_position parity against CurveRZFourier.gamma ─────────────────


class TestAxisPositionParity:
    @pytest.mark.parametrize("config_name", [name for name, _ in _ZOO_TARGETS])
    def test_axis_position_matches_curve_gamma(self, config_name):
        """``axis_position`` agrees with ``CurveRZFourier.gamma_impl``."""
        _, _, ma, _, _ = get_data(config_name)
        # Production-scale grid: 64 points over one field period.
        nfp = int(ma.nfp)
        phi_grid = np.linspace(0.0, 1.0 / nfp, 64, endpoint=False, dtype=np.float64)
        expected = np.zeros((phi_grid.size, 3), dtype=np.float64)
        ma.gamma_impl(expected, phi_grid)

        actual = np.asarray(axis_position(ma.to_spec(), phi_grid))
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=_DIRECT_RTOL,
            atol=_DIRECT_ATOL,
        )

    @pytest.mark.parametrize("config_name", [name for name, _ in _ZOO_TARGETS])
    def test_axis_position_and_tangent_finite_difference(self, config_name):
        """``dgamma/dphi`` from JVP matches a central FD against ``gamma_impl``."""
        _, _, ma, _, _ = get_data(config_name)
        nfp = int(ma.nfp)
        phi_grid = np.linspace(0.0, 1.0 / nfp, 16, endpoint=False, dtype=np.float64)
        eps = 1.0e-6

        gamma_p = np.zeros((phi_grid.size, 3), dtype=np.float64)
        gamma_m = np.zeros((phi_grid.size, 3), dtype=np.float64)
        ma.gamma_impl(gamma_p, phi_grid + eps)
        ma.gamma_impl(gamma_m, phi_grid - eps)
        fd_tangent = (gamma_p - gamma_m) / (2.0 * eps)

        _, jvp_tangent = axis_position_and_tangent(ma.to_spec(), phi_grid)
        np.testing.assert_allclose(
            np.asarray(jvp_tangent),
            fd_tangent,
            rtol=_HEAVY_FIRST_DERIVATIVE_RTOL,
            atol=_HEAVY_FIRST_DERIVATIVE_ATOL,
        )


# ── on_axis_iota_rk parity against the CPU oracle ────────────────────


class TestOnAxisIotaParity:
    @pytest.mark.parametrize("config_name,target_iota", _ZOO_TARGETS)
    def test_jax_kernel_matches_cpu_oracle(self, config_name, target_iota):
        """``on_axis_iota_rk`` matches ``compute_on_axis_iota`` at ``derivative_heavy``."""
        _, _, ma, _, bs = get_data(config_name)
        cpu_iota = compute_on_axis_iota(ma, bs)
        field_eval_fn = _make_jax_field_eval(bs.coils)
        jax_iota, steps, succeeded = on_axis_iota_rk(
            ma.to_spec(),
            field_eval_fn,
            rtol=1.0e-12,
            atol=1.0e-12,
            max_steps=10000,
        )
        assert bool(succeeded), "JAX RK integrator did not reach phi_end"
        assert int(steps) >= 1, "JAX RK integrator took no steps"
        np.testing.assert_allclose(
            float(jax_iota),
            cpu_iota,
            rtol=_HEAVY_SCALAR_RTOL,
            atol=_HEAVY_SCALAR_ATOL,
        )
        # Sanity: also agree with the published zoo target.
        np.testing.assert_allclose(
            float(jax_iota),
            target_iota,
            rtol=_HEAVY_SCALAR_RTOL,
            atol=_HEAVY_SCALAR_ATOL,
        )


# ── JIT-ability + repeatability ───────────────────────────────────────


class TestJitAndRepeatability:
    def test_kernel_is_jit_repeatable(self):
        """JIT-compiling the kernel and running twice yields the same scalar.

        Repeatability is the floor of "JIT-compatible" — if the kernel
        captures host state or branches on a Python value, the second
        traced call would diverge or recompile.
        """
        config_name = "hsx"
        _, _, ma, _, bs = get_data(config_name)
        spec = ma.to_spec()
        field_eval_fn = _make_jax_field_eval(bs.coils)

        @jax.jit
        def run(rtol_arr, atol_arr):
            iota, steps, succ = on_axis_iota_rk(
                spec,
                field_eval_fn,
                rtol=rtol_arr,
                atol=atol_arr,
                max_steps=10000,
            )
            return iota, steps, succ

        rtol_arr = jnp.asarray(1.0e-12, dtype=jnp.float64)
        atol_arr = jnp.asarray(1.0e-12, dtype=jnp.float64)
        iota1, steps1, succ1 = run(rtol_arr, atol_arr)
        iota2, steps2, succ2 = run(rtol_arr, atol_arr)
        np.testing.assert_array_equal(np.asarray(iota1), np.asarray(iota2))
        np.testing.assert_array_equal(np.asarray(steps1), np.asarray(steps2))
        np.testing.assert_array_equal(np.asarray(succ1), np.asarray(succ2))
        assert bool(succ1)

    def test_compiled_kernel_with_jax_field_runs_under_strict_transfer_guard(self):
        """Compiled kernel execution has no implicit host/device transfers."""
        config_name = "hsx"
        _, _, ma, _, _ = get_data(config_name)
        spec = jax.device_put(ma.to_spec())
        field_spec = ToroidalFieldSpec(R0=1.0, B0=1.0)

        def field_eval_fn(points):
            points_jax = jnp.asarray(points, dtype=jnp.float64)
            return toroidal_B(field_spec, points_jax), toroidal_dB(
                field_spec,
                points_jax,
            )

        @jax.jit
        def run(spec_arg, rtol_arr, atol_arr):
            return on_axis_iota_rk(
                spec_arg,
                field_eval_fn,
                rtol=rtol_arr,
                atol=atol_arr,
                max_steps=20000,
            )

        rtol_arr = jax.device_put(jnp.asarray(1.0e-8, dtype=jnp.float64))
        atol_arr = jax.device_put(jnp.asarray(1.0e-8, dtype=jnp.float64))
        compiled = run.lower(spec, rtol_arr, atol_arr).compile()

        with jax.transfer_guard("disallow"):
            iota, steps, succeeded = compiled(spec, rtol_arr, atol_arr)

        assert bool(succeeded)
        assert int(steps) >= 1
        assert np.isfinite(float(iota))

    def test_kernel_rejects_non_curve_rz_fourier_spec(self):
        with pytest.raises(TypeError, match="CurveRZFourierSpec"):
            on_axis_iota_rk(object(), lambda p: (p, p), rtol=1.0e-8, atol=1.0e-8)

    def test_kernel_rejects_non_positive_max_steps(self):
        config_name = "hsx"
        _, _, ma, _, bs = get_data(config_name)
        spec = ma.to_spec()
        field_eval_fn = _make_jax_field_eval(bs.coils)
        with pytest.raises(ValueError, match="max_steps"):
            on_axis_iota_rk(spec, field_eval_fn, max_steps=0)
