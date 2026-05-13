import jax
import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks.validation_ladder_contract import parity_ladder_tolerances
from simsopt.field.coil import Current
from simsopt.field.coilobjective import CurrentPenalty, current_penalty_pure

_DIRECT_KERNEL = parity_ladder_tolerances("direct_kernel")
_RTOL = _DIRECT_KERNEL["rtol"]
_ATOL = _DIRECT_KERNEL["atol"]


@pytest.mark.parametrize(
    ("current_value", "threshold", "expected_value", "expected_gradient"),
    (
        (3.0, 5.0, 0.0, 0.0),
        (8.0, 5.0, 9.0, 6.0),
        (-8.0, 5.0, 9.0, -6.0),
    ),
)
def test_current_penalty_matches_scalar_cpu_oracle(
    current_value,
    threshold,
    expected_value,
    expected_gradient,
):
    current = Current(current_value)
    objective = CurrentPenalty(current, threshold)

    assert float(objective.J()) == pytest.approx(expected_value, rel=_RTOL, abs=_ATOL)
    np.testing.assert_allclose(
        objective.dJ(),
        np.asarray([expected_gradient], dtype=np.float64),
        rtol=_RTOL,
        atol=_ATOL,
    )


def test_current_penalty_projects_scaled_current_gradient_to_base_current():
    base_current = Current(4.0)
    objective = CurrentPenalty(2.0 * base_current, threshold=5.0)

    assert float(objective.J()) == pytest.approx(9.0, rel=_RTOL, abs=_ATOL)
    np.testing.assert_allclose(
        objective.dJ(),
        np.asarray([12.0], dtype=np.float64),
        rtol=_RTOL,
        atol=_ATOL,
    )


def test_current_penalty_projects_current_sum_gradient_to_each_base_current():
    current_a = Current(2.0)
    current_b = Current(4.0)
    objective = CurrentPenalty(current_a + current_b, threshold=5.0)

    assert float(objective.J()) == pytest.approx(1.0, rel=_RTOL, abs=_ATOL)
    np.testing.assert_allclose(
        objective.dJ(),
        np.asarray([2.0, 2.0], dtype=np.float64),
        rtol=_RTOL,
        atol=_ATOL,
    )


def test_current_penalty_pure_jit_vmap_accepts_device_inputs_under_strict_guard():
    current = jax.device_put(np.asarray(8.0, dtype=np.float64))
    threshold = jax.device_put(np.asarray(5.0, dtype=np.float64))
    currents = jax.device_put(np.asarray([8.0, -8.0], dtype=np.float64))
    thresholds = jax.device_put(np.asarray([5.0, 5.0], dtype=np.float64))

    with jax.transfer_guard("disallow"):
        value = jax.jit(current_penalty_pure)(current, threshold)
        values = jax.vmap(current_penalty_pure)(currents, thresholds)

    assert float(value) == pytest.approx(9.0, rel=_RTOL, abs=_ATOL)
    np.testing.assert_allclose(
        np.asarray(values),
        np.asarray([9.0, 9.0], dtype=np.float64),
        rtol=_RTOL,
        atol=_ATOL,
    )


def test_current_penalty_pure_preserves_infinite_penalty():
    assert jnp.isinf(current_penalty_pure(jnp.asarray(np.inf), 5.0))


def test_current_penalty_pure_gradient_at_infinity_preserves_sign():
    """At I=±inf the analytic derivative is 2*(|I|-t)*sign(I) -> ±inf, not NaN.

    Regression guard for the prior `zero_source - zero_source` strict-
    transfer trick which forced 0 * inf through the chain rule and
    silently turned the autodiff gradient into NaN at the boundary.
    L-BFGS line-search overshoots that produce I=±inf rely on a
    finite-or-infinite (not NaN) gradient to backtrack cleanly.
    """
    grad_pos = jax.grad(current_penalty_pure)(jnp.asarray(jnp.inf), 5.0)
    grad_neg = jax.grad(current_penalty_pure)(jnp.asarray(-jnp.inf), 5.0)
    assert jnp.isinf(grad_pos) and grad_pos > 0
    assert jnp.isinf(grad_neg) and grad_neg < 0


def test_current_penalty_production_scale_multi_coil_sum_parity():
    """Production-scale floor: sum-over-ncoils CurrentPenalty matches a
    NumPy reference at the Stage-2 banana TF coil count (16 coils).
    """
    ncoils = 16
    rng = np.random.default_rng(seed=1729)
    current_values = rng.normal(loc=0.0, scale=8.0, size=ncoils).astype(np.float64)
    threshold = 5.0

    objectives = [CurrentPenalty(Current(float(v)), threshold) for v in current_values]
    jax_total = sum(float(obj.J()) for obj in objectives)
    numpy_total = float(np.sum(np.maximum(np.abs(current_values) - threshold, 0.0) ** 2))

    assert jax_total == pytest.approx(numpy_total, rel=_RTOL, abs=_ATOL)

    currents_device = jax.device_put(current_values)
    thresholds_device = jax.device_put(np.full(ncoils, threshold, dtype=np.float64))

    with jax.transfer_guard("disallow"):
        vmap_values = jax.vmap(current_penalty_pure)(currents_device, thresholds_device)

    np.testing.assert_allclose(
        np.asarray(vmap_values),
        np.maximum(np.abs(current_values) - threshold, 0.0) ** 2,
        rtol=_RTOL,
        atol=_ATOL,
    )


def test_current_penalty_wrapper_uses_explicit_transfer_boundary():
    objective = CurrentPenalty(Current(8.0), threshold=5.0)

    with jax.transfer_guard("disallow"):
        value = objective.J()
        gradient = objective.dJ()

    assert float(value) == pytest.approx(9.0, rel=_RTOL, abs=_ATOL)
    np.testing.assert_allclose(
        gradient,
        np.asarray([6.0], dtype=np.float64),
        rtol=_RTOL,
        atol=_ATOL,
    )
