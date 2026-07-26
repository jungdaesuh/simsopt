"""Tests for the JAX-free numerical-policy owner."""

from __future__ import annotations

import math
import sys

import pytest
from simsopt_jax.numerical_policy import (
    DENSE_IR_HISTORY_CONTRACTION_RATIO_CAPACITY,
    DENSE_IR_HISTORY_RESIDUAL_RELATIVE_CAPACITY,
    MIXED_DENSE_IR_ACCURACY_POLICY,
    MIXED_DENSE_IR_CONTRACTION_IDEAL_GAUSSIAN_FAILURE_PROBABILITY_BOUND,
    MIXED_DENSE_IR_CONTRACTION_NORM_UPPER_LIMIT,
    MIXED_DENSE_IR_CONTRACTION_PROBE_ALPHA,
    MIXED_DENSE_IR_CONTRACTION_PROBE_COUNT,
    MIXED_DENSE_IR_MAX_REFINEMENT_CORRECTIONS,
    NEWTON_ARMIJO_C1,
    DenseIrHistorySource,
    MixedDenseIrAccuracyPolicy,
    dense_ir_factorization_precision_evidence_is_complete,
    mixed_dense_ir_certificate_dtype_name,
)


def test_dense_ir_history_capacity_and_source_codes_share_one_policy() -> None:
    assert DENSE_IR_HISTORY_RESIDUAL_RELATIVE_CAPACITY == (
        MIXED_DENSE_IR_MAX_REFINEMENT_CORRECTIONS + 1
    )
    assert DENSE_IR_HISTORY_CONTRACTION_RATIO_CAPACITY == (
        MIXED_DENSE_IR_MAX_REFINEMENT_CORRECTIONS
    )
    assert tuple(int(source) for source in DenseIrHistorySource) == (0, 1, 2, 3)
    assert NEWTON_ARMIJO_C1 == 1.0e-4


def test_mixed_dense_ir_hmt_and_certificate_dtype_share_one_policy() -> None:
    assert MIXED_DENSE_IR_CONTRACTION_PROBE_COUNT == 64
    assert MIXED_DENSE_IR_CONTRACTION_PROBE_ALPHA == 2.0
    assert MIXED_DENSE_IR_CONTRACTION_NORM_UPPER_LIMIT == 0.9
    assert MIXED_DENSE_IR_CONTRACTION_IDEAL_GAUSSIAN_FAILURE_PROBABILITY_BOUND == (
        MIXED_DENSE_IR_CONTRACTION_PROBE_ALPHA**-MIXED_DENSE_IR_CONTRACTION_PROBE_COUNT
    )
    assert mixed_dense_ir_certificate_dtype_name() == (
        MIXED_DENSE_IR_ACCURACY_POLICY.certificate_dtype
    )
    assert MIXED_DENSE_IR_ACCURACY_POLICY.certificate_dtype == "float64"


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
        {"certificate_dtype": "float32"},
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
        "certificate_dtype": "float64",
        "linear_solve_tolerance_floor": 1.0e-14,
        "linear_solve_tolerance_cap": 1.0e-10,
        "forward_error_tolerance_multiplier": 10.0,
    }
    values.update(overrides)

    with pytest.raises(ValueError, match="Mixed dense-IR accuracy policy"):
        MixedDenseIrAccuracyPolicy(**values)


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
