"""Tests for the JAX-free numerical-policy owner."""

from __future__ import annotations

import math
import sys

import pytest
from simsopt_jax.numerical_policy import (
    MIXED_DENSE_IR_ACCURACY_POLICY,
    dense_ir_factorization_precision_evidence_is_complete,
)


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
    ("event_count", "factor_bits", "apply_bits", "expected"),
    (
        (0, None, None, True),
        (0, 32, 32, False),
        (1, 32, 32, True),
        (1, 32, None, False),
        (1, None, 32, False),
    ),
)
def test_dense_ir_factorization_precision_evidence_completeness(
    event_count: int,
    factor_bits: int | None,
    apply_bits: int | None,
    expected: bool,
):
    assert (
        dense_ir_factorization_precision_evidence_is_complete(
            factorization_event_count=event_count,
            last_factorization_dtype_bits=factor_bits,
            factor_application_dtype_bits=apply_bits,
        )
        is expected
    )
