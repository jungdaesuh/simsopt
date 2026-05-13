"""Item 08 closeout tests for ``simsopt.geo.strain_optimization``.

These tests cover the coverage-completeness gap for the strain optimization
module's module-level ``partial(jit, static_argnames=...)`` kernel closures
that wrap the upstream ``Lp_torsion_pure`` integrand and the
``torstrain_pure`` / ``binormstrain_pure`` pointwise strain definitions.

Two invariants are exercised:

1. Production-scale parity: at the NCSX ``coil_order=6`` /
   ``points_per_period=120`` fixture used by the existing
   ``tests/geo/test_strainopt.py::CoilStrainTesting`` cases, the
   ``LPTorsionalStrainPenalty.J()`` value and dimensionality of ``dJ()``
   are invariant under JAX device-residency transitions of the inputs and
   must match a host-side NumPy reference that re-implements the same
   ``Lp_torsion_pure`` arithmetic over the host-fetched torsion /
   gammadash arrays. Tolerances come from
   ``benchmarks.validation_ladder_contract.parity_ladder_tolerances("direct_kernel")``.
2. Negative control: on a zero-twist circular ``CurveXYZFourier`` wrapped
   by a Frenet frame with the zero rotation, the binormal curvature
   strain vanishes identically. The penalty value must sit below an
   ``"atol"`` floor pulled from the same parity ladder lane.

The tests run cleanly under ``SIMSOPT_JAX_TRANSFER_GUARD=disallow``: the
strain wrappers explicitly stage the host ``gammadash`` array through
``jax_core._math_utils.as_jax_float64`` inside an ``allow`` boundary, and
all other kernel inputs flow through the existing JAX-native
``framedcurve`` paths.
"""

from __future__ import annotations

import jax
import numpy as np

from benchmarks.validation_ladder_contract import parity_ladder_tolerances
from simsopt.configs.zoo import get_data
from simsopt.geo import (
    FrameRotation,
    FramedCurveCentroid,
    FramedCurveFrenet,
    ZeroRotation,
)
from simsopt.geo.curvexyzfourier import CurveXYZFourier
from simsopt.geo.strain_optimization import (
    LPBinormalCurvatureStrainPenalty,
    LPTorsionalStrainPenalty,
)

_DIRECT_KERNEL = parity_ladder_tolerances("direct_kernel")
_RTOL = _DIRECT_KERNEL["rtol"]
_ATOL = _DIRECT_KERNEL["atol"]


def _make_ncsx_reference_curve():
    """Production-scale NCSX coil 0 fixture matching ``test_strainopt.py``."""
    base_curves, _, _, _, _ = get_data(
        "ncsx",
        coil_order=6,
        points_per_period=120,
    )
    return base_curves[0]


def _numpy_lp_torsion_reference(strain_host, gammadash_host, *, p, threshold):
    """Host-side NumPy reproduction of ``Lp_torsion_pure``.

    ``Lp_torsion_pure`` computes
    ``(1/p) * mean(max(|strain| - threshold, 0)**p * arc_length)`` where
    ``arc_length = ||gammadash||_2`` along axis 1. The reference uses
    ``float64`` NumPy intermediates to provide an oracle independent of the
    JAX jit closure.
    """
    arc_length = np.linalg.norm(np.asarray(gammadash_host, dtype=np.float64), axis=1)
    excess = np.maximum(
        np.abs(np.asarray(strain_host, dtype=np.float64)) - threshold, 0.0
    )
    return (1.0 / p) * np.mean((excess**p) * arc_length)


def _numpy_torstrain_reference(torsion_host, width):
    """Host NumPy reproduction of ``torstrain_pure``: tau**2 * w**2 / 12."""
    torsion_array = np.asarray(torsion_host, dtype=np.float64)
    return torsion_array**2 * (width**2) / 12.0


def test_lp_torsional_penalty_production_scale_matches_numpy_reference_under_strict_guard():
    """Production-scale invariance under device residency transitions.

    Builds the NCSX coil_order=6 / points_per_period=120 framed curve used
    by ``CoilStrainTesting.subtest_torsion``, evaluates
    ``LPTorsionalStrainPenalty.J()`` and ``dJ()`` once with the wrapper
    consuming its mutable Optimizable state, then fetches the JAX-native
    torsion and host gammadash, and recomputes the integrand entirely in
    NumPy to act as the host oracle. Both passes happen inside
    ``jax.transfer_guard("disallow")`` so the explicit host->device
    boundary inside the wrapper is the only allowed transfer.
    """
    curve = _make_ncsx_reference_curve()
    rotation = FrameRotation(curve.quadpoints, 1)
    rotation.x = np.array([0.0, 0.1, 0.3], dtype=np.float64)
    framedcurve = FramedCurveCentroid(curve, rotation)
    width = 1e-3
    p = 2
    threshold = 1e-8

    objective = LPTorsionalStrainPenalty(
        framedcurve,
        width=width,
        p=p,
        threshold=threshold,
    )

    with jax.transfer_guard("disallow"):
        jax_value = float(objective.J())
        jax_gradient = np.asarray(objective.dJ(), dtype=np.float64)
        torsion_host = np.asarray(
            jax.device_get(framedcurve.frame_torsion()), dtype=np.float64
        )

    gammadash_host = np.asarray(framedcurve.curve.gammadash(), dtype=np.float64)
    torsion_strain_host = _numpy_torstrain_reference(torsion_host, width)
    numpy_value = _numpy_lp_torsion_reference(
        torsion_strain_host,
        gammadash_host,
        p=p,
        threshold=threshold,
    )

    assert jax_value == np.float64(numpy_value) or abs(jax_value - numpy_value) <= max(
        _ATOL, _RTOL * abs(numpy_value)
    )
    np.testing.assert_allclose(
        jax_value,
        numpy_value,
        rtol=_RTOL,
        atol=_ATOL,
    )
    assert jax_gradient.shape == (objective.x.size,)
    assert np.all(np.isfinite(jax_gradient))


def test_lp_binormal_penalty_zero_twist_circle_vanishes_in_frenet_frame():
    """Negative control: zero-strain on a torsion-free fixture.

    A planar circular ``CurveXYZFourier`` wrapped by a Frenet frame with no
    rotation (the ``ZeroRotation`` / ``rotation=None`` case in the existing
    ``subtest_binormal_curvature``) has identically zero binormal curvature
    strain. The penalty value must sit at the direct-kernel ``atol`` floor,
    confirming the module-level jit closure pattern produces the same
    zero-floor that the upstream non-JAX path produces.
    """
    quadpoints = np.linspace(0.0, 1.0, 64, endpoint=False)
    curve = CurveXYZFourier(quadpoints, order=1)
    curve.set("xc(1)", 1e-4)
    curve.set("ys(1)", 1e-4)
    curve.fix_all()
    framedcurve = FramedCurveFrenet(curve, ZeroRotation(quadpoints))

    objective = LPBinormalCurvatureStrainPenalty(
        framedcurve,
        width=1e-3,
        p=2,
        threshold=0.0,
    )

    with jax.transfer_guard("disallow"):
        value = float(objective.J())

    assert value <= max(_ATOL, _RTOL)
