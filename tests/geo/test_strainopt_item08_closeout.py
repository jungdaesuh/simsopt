"""Item 08 closeout tests for ``simsopt.geo.strain_optimization``.

These tests cover the coverage-completeness gap for the strain optimization
module's module-level ``partial(jit, static_argnames=...)`` kernel closures
that wrap the upstream ``Lp_torsion_pure`` integrand and the
``torstrain_pure`` / ``binormstrain_pure`` pointwise strain definitions.

Two invariants are exercised, both anchored to closed-form analytic
oracles (no host-side reproduction of the formula under test):

1. ``Lp_torsion_pure`` integrand identity on constant-strain inputs.
   With a constant strain array ``s`` and a constant tangent-magnitude
   array ``|gammadash| = v``, the integrand reduces to a closed-form
   value
   ``J = (1/p) * max(|s| - threshold, 0)**p * v``,
   independent of the number of quadrature points. The JIT closure
   ``_lp_strain_penalty_value`` is invoked directly on the synthetic
   arrays so the oracle is independent of ``frame_torsion`` / ``Lp_torsion_pure``
   internals. End-to-end wrapper behaviour (mutable Optimizable state,
   gradient dimensionality, and the strict ``jax.transfer_guard``
   boundary) is exercised separately on the NCSX production-scale
   fixture, where only the dimensionality, finiteness, and
   transfer-guard surface is asserted.

2. Negative control on a torsion-free planar circle. A planar
   ``CurveXYZFourier`` (``z`` identically zero) has Frenet and centroid
   binormals identically equal to ``(0, 0, ±1)`` along the curve, so
   ``n'(s) . b = 0`` and ``t'(s) . b = 0`` are *symbolic* zeros — both
   ``frame_torsion`` and ``frame_binormal_curvature`` evaluate to
   bit-exact zero arrays under JAX float64, and the strain wrappers
   therefore return ``0.0`` exactly. The assertion is strict
   ``value == 0.0`` (verified bit-exact in the JAX 0.10.0 runtime),
   replacing the prior ``<= 1e-10`` floor that admitted a buggy kernel
   returning small non-zero arithmetic.

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
    _lp_strain_penalty_value,
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


def test_lp_strain_penalty_value_matches_closed_form_constant_strain_oracle():
    """Closed-form identity for ``_lp_strain_penalty_value`` on constant inputs.

    Construct a synthetic strain array ``s_i = s_0`` (constant) and a
    tangent array ``gammadash_i`` whose Euclidean magnitude is constant
    ``|gammadash_i| = v_0``. The integrand kernel
    ``Lp_torsion_pure`` then computes

    .. math::

       J = (1/p) \\cdot \\mathrm{mean}\\left[
           \\max(|s_0| - \\tau_0, 0)^p \\cdot v_0
       \\right] = (1/p) \\cdot \\max(|s_0| - \\tau_0, 0)^p \\cdot v_0

    independent of the number of quadrature points ``N``. Two regimes
    are checked:

    * Active strain (``|s_0| > threshold``): non-zero closed-form value.
    * Below threshold (``|s_0| <= threshold``): exact zero from the
      ``max`` clamp.

    The synthetic-input strategy makes the oracle independent of the
    ``frame_torsion`` / ``torstrain_pure`` upstream pipeline and of
    ``Lp_torsion_pure``'s own arithmetic — only the contract
    ``mean(max(|s| - threshold, 0)**p * |gammadash|) / p`` is tested.
    """
    rng = np.random.default_rng(seed=20260513)
    for n_quad in (32, 65, 128):
        s0 = 3.0
        v0 = 4.0
        p = 3
        threshold = 0.5

        strain = np.full(n_quad, s0, dtype=np.float64)
        # Build gammadash with constant magnitude v0 along a random
        # direction at each point: orientation does not change the
        # integral because the kernel consumes only ||gammadash||.
        directions = rng.standard_normal(size=(n_quad, 3))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        gammadash = v0 * directions

        jax_value = float(
            _lp_strain_penalty_value(strain, gammadash, p=p, threshold=threshold)
        )
        excess = max(abs(s0) - threshold, 0.0)
        expected = (1.0 / p) * (excess**p) * v0
        np.testing.assert_allclose(jax_value, expected, rtol=_RTOL, atol=_ATOL)

        # Below-threshold regime collapses to exact zero from the
        # max-clamp inside Lp_torsion_pure.
        zeroed = float(
            _lp_strain_penalty_value(strain, gammadash, p=p, threshold=abs(s0))
        )
        assert zeroed == 0.0


def test_lp_torsional_penalty_production_scale_wrapper_runs_under_strict_guard():
    """End-to-end smoke for the LPTorsionalStrainPenalty wrapper.

    Builds the NCSX coil_order=6 / points_per_period=120 framed curve
    used by ``CoilStrainTesting.subtest_torsion`` and evaluates the
    wrapper's value and gradient inside ``jax.transfer_guard("disallow")``.
    The closed-form identity for ``_lp_strain_penalty_value`` is covered
    by the constant-strain test above; this test asserts only the
    surface a wrong-formula bug would not perturb: gradient
    dimensionality, finiteness, and that the explicit host->device
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

    assert np.isfinite(jax_value)
    assert jax_value >= 0.0
    assert jax_gradient.shape == (objective.x.size,)
    assert np.all(np.isfinite(jax_gradient))


def test_lp_binormal_penalty_zero_twist_circle_vanishes_in_frenet_frame():
    """Negative control: bit-exact zero on a torsion-free planar circle.

    A planar circular ``CurveXYZFourier`` with ``z`` identically zero
    has Frenet binormal ``b(s) = (0, 0, +-1)`` constant along the curve
    by construction (the binormal of any planar curve is the unit
    normal to the plane). Therefore ``t'(s) . b = 0`` is a *symbolic*
    zero (cross-product of an in-plane curvature vector with the
    out-of-plane binormal), and JAX float64 evaluates the resulting
    expression to bit-exact ``0.0`` — verified at audit time on the
    JAX 0.10.0 runtime (see audit TODO #15). The same holds for
    torsion ``n'(s) . b``. The strict equality assertion replaces the
    prior ``<= 1e-10`` floor, which admitted any non-vanishing
    arithmetic below that threshold.
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

    assert value == 0.0


def test_lp_torsional_penalty_zero_twist_circle_vanishes_in_centroid_frame():
    """Negative control: bit-exact zero on a planar circle in the centroid frame.

    The centroid-frame binormal of a planar curve is the unit normal
    to the plane (constant), and the centroid-frame normal-dash
    inner-product with this binormal therefore vanishes symbolically.
    Under JAX float64 the kernel returns bit-exact ``0.0`` for both
    ``frame_torsion`` and the composed ``LPTorsionalStrainPenalty.J()``
    value — verified at audit time (see audit TODO #15). This is the
    centroid-frame companion to the Frenet case above, closing the
    coverage row that both wrappers vanish on the planar-curve
    fixture.
    """
    quadpoints = np.linspace(0.0, 1.0, 64, endpoint=False)
    curve = CurveXYZFourier(quadpoints, order=1)
    curve.set("xc(1)", 1e-4)
    curve.set("ys(1)", 1e-4)
    curve.fix_all()
    framedcurve = FramedCurveCentroid(curve, ZeroRotation(quadpoints))

    objective = LPTorsionalStrainPenalty(
        framedcurve,
        width=1e-3,
        p=2,
        threshold=0.0,
    )

    with jax.transfer_guard("disallow"):
        value = float(objective.J())

    assert value == 0.0
