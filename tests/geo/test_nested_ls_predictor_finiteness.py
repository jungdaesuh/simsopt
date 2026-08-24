"""Finite-surface guarantees for the pure nested-LS predictor policy."""

from __future__ import annotations

import numpy as np
import pytest

from simsopt_jax_adapters.geo.nested_ls_contract import (
    NESTED_LS_PREDICTOR_ARM_BARE,
    nested_ls_predictor_arm,
    nested_ls_predictor_trust_region,
)


@pytest.mark.parametrize(
    "raw_delta",
    [
        np.array([np.nan, 0.0]),
        np.array([np.inf, 0.0]),
        np.array([-np.inf, 0.0]),
    ],
)
def test_nonfinite_predictor_delta_falls_back_to_the_bare_anchor(
    raw_delta: np.ndarray,
) -> None:
    """A nonfinite tangent may not create nonfinite predicted surface bytes."""

    anchor = np.array([3.0, 4.0])
    applied, _raw_l2, applied_l2, _cap, scaled = nested_ls_predictor_trust_region(
        delta_surface=raw_delta,
        anchor_surface_dofs=anchor,
    )

    assert np.array_equal(applied, np.zeros_like(anchor))
    assert applied_l2 == 0.0
    assert scaled is False
    assert np.all(np.isfinite(anchor + applied))


@pytest.mark.parametrize("predicted_gradient_l2", [np.nan, np.inf])
def test_nonfinite_predicted_gradient_selects_the_bare_anchor(
    predicted_gradient_l2: float,
) -> None:
    """An invalid predictor diagnostic cannot win its envelope comparison."""

    assert (
        nested_ls_predictor_arm(
            bare_gradient_l2=1.0,
            predicted_gradient_l2=predicted_gradient_l2,
        )
        == NESTED_LS_PREDICTOR_ARM_BARE
    )
