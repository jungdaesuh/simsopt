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
