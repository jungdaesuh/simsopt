"""Immutable-spec tests for OrientedCurveXYZFourier JAX geometry."""

from __future__ import annotations

import jax
import numpy as np

from benchmarks.validation_ladder_contract import parity_ladder_tolerances
from simsopt.geo.orientedcurve import OrientedCurveXYZFourier
from simsopt.jax_core import (
    OrientedCurveXYZFourierSpec,
    curve_gamma_and_dash_from_spec,
)


_DIRECT_KERNEL = parity_ladder_tolerances("direct_kernel")
_RTOL = _DIRECT_KERNEL["rtol"]
_ATOL = _DIRECT_KERNEL["atol"]


def test_oriented_curve_to_spec_preserves_live_geometry():
    curve = OrientedCurveXYZFourier(16, order=2)
    curve.x = np.array(
        [
            1.2,
            -0.1,
            0.3,
            0.2,
            -0.15,
            0.05,
            0.1,
            -0.04,
            0.03,
            0.2,
            -0.05,
            0.07,
            0.09,
            -0.11,
            0.08,
            0.13,
            -0.02,
            0.06,
        ],
        dtype=np.float64,
    )

    spec = curve.to_spec()
    assert isinstance(spec, OrientedCurveXYZFourierSpec)
    gamma, gammadash = jax.jit(curve_gamma_and_dash_from_spec)(spec)

    np.testing.assert_allclose(
        np.asarray(gamma),
        np.asarray(curve.gamma()),
        rtol=_RTOL,
        atol=_ATOL,
    )
    np.testing.assert_allclose(
        np.asarray(gammadash),
        np.asarray(curve.gammadash()),
        rtol=_RTOL,
        atol=_ATOL,
    )
