"""The RCLS constraint-roundoff tolerance bucket is a derived bound, not a fit."""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

from simsopt_jax.parity_tolerances import PARITY_LADDER_TOLERANCES

ROUNDOFF_BUCKET = PARITY_LADDER_TOLERANCES["rcls_constraint_roundoff"]


def test_roundoff_bucket_is_the_a_priori_bound_and_is_not_tuned() -> None:
    # The bucket compares residuals already divided by a two-evaluation FP64
    # rounding bound, so `atol 1.0` means "one a-priori pair bound". Retuning it
    # toward an observed drift would turn a derived bound into a measured one;
    # this test exists to make that edit fail.
    assert ROUNDOFF_BUCKET["rtol"] == 0.0
    assert ROUNDOFF_BUCKET["atol"] == 1.0
    assert ROUNDOFF_BUCKET["requires_same_state"] is True
