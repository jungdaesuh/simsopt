"""Tests for the JAX-free numerical-policy owner."""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import math
import sys

import pytest
from simsopt_jax.numerical_policy import (
    MIXED_DENSE_IR_ACCURACY_POLICY,
    NEWTON_ARMIJO_C1,
    MixedDenseIrAccuracyPolicy,
)


def test_newton_armijo_constant_is_policy_owned() -> None:
    assert NEWTON_ARMIJO_C1 == 1.0e-4


def test_mixed_dense_ir_forward_error_limit_uses_the_policy_floor():
    policy = MIXED_DENSE_IR_ACCURACY_POLICY

    assert policy.forward_error_tolerance(1.0e-14) == pytest.approx(
        math.sqrt(sys.float_info.epsilon)
    )
    assert policy.forward_error_tolerance(1.0e-10) == pytest.approx(
        math.sqrt(sys.float_info.epsilon)
    )


@pytest.mark.parametrize("tolerance", (-1.0, 0.0, 1.0e-15, 1.0e-9, math.inf))
def test_mixed_dense_ir_forward_error_limit_rejects_out_of_policy_tolerance(
    tolerance: float,
):
    with pytest.raises(ValueError, match="outside the FP64 policy"):
        MIXED_DENSE_IR_ACCURACY_POLICY.forward_error_tolerance(tolerance)


@pytest.mark.parametrize(
    "overrides",
    (
        {"linear_solve_tolerance_floor": 0.0},
        {"linear_solve_tolerance_floor": math.nan},
        {"linear_solve_tolerance_cap": math.inf},
        {
            "linear_solve_tolerance_floor": 1.0e-9,
            "linear_solve_tolerance_cap": 1.0e-10,
        },
        {"forward_error_tolerance_multiplier": 0.0},
        {"forward_error_tolerance_multiplier": math.nan},
    ),
)
def test_mixed_dense_ir_accuracy_policy_rejects_invalid_values_before_jit(
    overrides: dict[str, object],
) -> None:
    values: dict[str, object] = {
        "linear_solve_tolerance_floor": 1.0e-14,
        "linear_solve_tolerance_cap": 1.0e-10,
        "forward_error_tolerance_multiplier": 10.0,
    }
    values.update(overrides)

    with pytest.raises(ValueError, match="Mixed dense-IR accuracy policy"):
        MixedDenseIrAccuracyPolicy(**values)
