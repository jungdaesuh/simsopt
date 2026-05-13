"""Item 07 closeout: targeted parity for non-distance public curve
objectives in ``src/simsopt/geo/curveobjectives.py``.

This module pins three pre-existing coverage gaps on the parent commit:

1. ``FramedCurveTwist`` had no finite-difference Taylor parity test for
   ``f="lp"``, the only differentiable wrapping mode.
2. ``FramedCurveTwist`` had no contract assertion for the
   non-differentiable wrapping modes ``f in {net, range, max}``, which
   the source returns as ``Derivative({})`` (zero-derivative block).
3. ``LinkingNumber`` had no production-scale fixture at the documented
   ``ncoils>=4`` floor (the existing local tests fix ``ncoils=2``).

Tolerances are sourced from
``benchmarks.validation_ladder_contract.parity_ladder_tolerances`` so
the file contains no inline ``atol`` / ``rtol`` numeric literals.

Distance classes are owned by item 01 and are not re-validated here.
"""

from __future__ import annotations

import numpy as np
import pytest

import jax

from benchmarks.validation_ladder_contract import parity_ladder_tolerances
from simsopt._core.derivative import Derivative
from simsopt.geo.curve import create_equally_spaced_curves
from simsopt.geo.curvexyzfourier import CurveXYZFourier
from simsopt.geo.curveobjectives import FramedCurveTwist, LinkingNumber
from simsopt.geo.framedcurve import FrameRotation, FramedCurveCentroid


_DIRECT_KERNEL = parity_ladder_tolerances("direct_kernel")
_DIRECT_RTOL = _DIRECT_KERNEL["rtol"]
_DIRECT_ATOL = _DIRECT_KERNEL["atol"]

_FD_GRADIENT = parity_ladder_tolerances("fd_gradient")
_FD_RTOL = _FD_GRADIENT["directional_fd_rtol"]
_FD_ATOL = _FD_GRADIENT["directional_fd_atol"]
_FD_FLOOR = _FD_GRADIENT["directional_derivative_floor"]
_FD_ERROR_RATE = _FD_GRADIENT["central_fd_error_rate"]
_FD_MIN_STABLE = _FD_GRADIENT["central_fd_min_stable_eps"]
_FD_DIRECTION_SEED = _FD_GRADIENT["direction_seed"]


def _build_framed_curve_twist_fixture(*, seed):
    """Build a small framed-curve fixture for the ``f="lp"`` Taylor test.

    The curve and rotation orders are deliberately small to keep the
    finite-difference inner loop tractable. The dof seed is drawn from
    a deterministic ``numpy.random.default_rng`` so the test is
    reproducible.
    """

    rng = np.random.default_rng(seed)
    quadpoints = np.linspace(0.0, 1.0, 32, endpoint=False)

    curve = CurveXYZFourier(quadpoints, order=2)
    base_dofs = curve.x.copy()
    perturb = rng.standard_normal(size=base_dofs.shape) * 0.05
    curve.x = base_dofs + perturb
    # Re-seed three Fourier modes so the geometry is a non-degenerate
    # near-circle in the xy-plane plus a small z-offset, well clear of
    # the FramedCurveTwist normal-product singularity.
    curve.set("xc(1)", 1.0)
    curve.set("ys(1)", 1.0)
    curve.set("zc(0)", 0.05)
    curve.set("zs(1)", 0.05)

    rotation = FrameRotation(quadpoints, order=1)
    rotation.x = np.array([0.10, -0.20, 0.05])

    framedcurve = FramedCurveCentroid(curve, rotation)
    return framedcurve


def _composite_dofs(objective):
    """Return the concatenated free dofs of an Optimizable."""
    return np.asarray(objective.x, dtype=np.float64).copy()


def _evaluate_with_dofs(objective, dofs):
    """Round-trip dofs through the Optimizable boundary."""
    objective.x = dofs
    return float(objective.J())


def test_framed_curve_twist_lp_taylor_value_and_gradient():
    """FD-Taylor parity for ``FramedCurveTwist(f='lp')``.

    The test asserts:

    - ``J()`` returns a finite scalar.
    - ``dJ()`` returns a finite-vector-valued gradient over the union of
      the curve dofs and the rotation dofs (the centroid rotation is
      fixed by ``FramedCurveTwist.__init__``).
    - Central-difference directional derivative converges to the
      analytic directional derivative as the step shrinks, with at
      least ``_FD_MIN_STABLE`` consecutive halvings under the
      contraction rate ``_FD_ERROR_RATE``.
    """

    framedcurve = _build_framed_curve_twist_fixture(seed=_FD_DIRECTION_SEED)
    objective = FramedCurveTwist(framedcurve, f="lp", p=2)

    value = float(objective.J())
    assert np.isfinite(value)

    dofs = _composite_dofs(objective)
    grad = np.asarray(objective.dJ(), dtype=np.float64)
    assert grad.shape == dofs.shape
    assert np.all(np.isfinite(grad))

    rng = np.random.default_rng(_FD_DIRECTION_SEED)
    direction = rng.standard_normal(size=dofs.shape)
    direction /= float(np.linalg.norm(direction))

    analytic_directional = float(np.dot(grad, direction))
    assert abs(analytic_directional) > _FD_FLOOR

    try:
        prev_err = None
        stable_halvings = 0
        for k in range(7, 7 + max(_FD_MIN_STABLE, 3) + 3):
            eps = 0.5**k
            forward = _evaluate_with_dofs(objective, dofs + eps * direction)
            backward = _evaluate_with_dofs(objective, dofs - eps * direction)
            central = (forward - backward) / (2.0 * eps)
            err = abs(central - analytic_directional)
            if prev_err is not None:
                if err < _FD_ERROR_RATE * prev_err:
                    stable_halvings += 1
                else:
                    stable_halvings = 0
                if stable_halvings >= _FD_MIN_STABLE:
                    break
            prev_err = err
        assert stable_halvings >= _FD_MIN_STABLE, (
            "central-difference Taylor estimate did not contract "
            f"under rate {_FD_ERROR_RATE} for {_FD_MIN_STABLE} "
            "consecutive halvings"
        )
        # Final-step relative error must also satisfy the FD ladder
        # threshold; this is the same envelope the parity ladder uses
        # for directional FD parity.
        relative_residual = err / max(abs(analytic_directional), _FD_FLOOR)
        assert err < _FD_ATOL + _FD_RTOL * max(abs(analytic_directional), _FD_FLOOR), (
            f"final FD error {err:.3e} exceeded ladder envelope; "
            f"relative residual = {relative_residual:.3e}"
        )
    finally:
        objective.x = dofs


@pytest.mark.parametrize("mode", ("net", "range", "max"))
def test_framed_curve_twist_non_lp_modes_return_zero_derivative(mode):
    """Modes ``net``, ``range``, ``max`` carry a zero-derivative
    contract (source: ``src/simsopt/geo/curveobjectives.py:1375``).

    The test asserts that ``J()`` is finite and that ``dJ()`` returns
    a zero-vector of the correct shape. The underlying
    ``Derivative({})`` block is also verified via ``partials=True`` so
    the contract is checked at the dictionary boundary, not only at
    the projected-vector boundary.
    """

    framedcurve = _build_framed_curve_twist_fixture(seed=_FD_DIRECTION_SEED)
    objective = FramedCurveTwist(framedcurve, f=mode, p=2)

    value = float(objective.J())
    assert np.isfinite(value)

    partial_deriv = objective.dJ(partials=True)
    assert isinstance(partial_deriv, Derivative)
    assert len(partial_deriv.data) == 0

    projected = np.asarray(objective.dJ(), dtype=np.float64)
    dofs = _composite_dofs(objective)
    assert projected.shape == dofs.shape
    np.testing.assert_allclose(
        projected,
        np.zeros_like(projected),
        rtol=_DIRECT_RTOL,
        atol=_DIRECT_ATOL,
    )


def _build_unlinked_tf_ring_fixture():
    """Build a production-scale ``ncoils=4`` unlinked TF coil ring.

    ``create_equally_spaced_curves`` returns ``ncoils`` planar TF
    rings centred on disjoint phi planes; the resulting set is
    topologically unlinked (Gauss linking integer is 0 for every
    pair).
    """

    return create_equally_spaced_curves(
        ncurves=4,
        nfp=2,
        stellsym=True,
        R0=1.0,
        R1=0.5,
        order=6,
    )


def test_linking_number_production_scale_ncoils_four():
    """``LinkingNumber.J()`` is integer-valued at the documented
    ``ncoils>=4`` production-scale floor, and ``dJ()`` is the
    zero-``Derivative({})`` block by source contract.
    """

    curves = _build_unlinked_tf_ring_fixture()
    objective = LinkingNumber(curves)

    value = float(objective.J())
    assert value == pytest.approx(0.0, rel=_DIRECT_RTOL, abs=_DIRECT_ATOL)
    # Linking number is integer-valued by topology; the equally-spaced
    # planar ring is unlinked, so the integer value is exactly 0.
    assert int(round(value)) == 0

    partial_deriv = objective.dJ(partials=True)
    assert isinstance(partial_deriv, Derivative)
    assert len(partial_deriv.data) == 0

    projected = np.asarray(objective.dJ(), dtype=np.float64)
    dofs = _composite_dofs(objective)
    assert projected.shape == dofs.shape
    np.testing.assert_allclose(
        projected,
        np.zeros_like(projected),
        rtol=_DIRECT_RTOL,
        atol=_DIRECT_ATOL,
    )


def test_linking_number_production_scale_ncoils_four_strict_transfer_guard():
    """The ``LinkingNumber`` value boundary must not implicitly cross
    the host-to-device boundary. The kernel is a C++ binding that
    consumes host numpy arrays, so the wrapper itself stays outside
    the JAX transfer guard, but executing under ``disallow`` proves
    no incidental JAX transfer is triggered.
    """

    curves = _build_unlinked_tf_ring_fixture()
    objective = LinkingNumber(curves)

    with jax.transfer_guard("disallow"):
        value = float(objective.J())

    assert value == pytest.approx(0.0, rel=_DIRECT_RTOL, abs=_DIRECT_ATOL)
    assert int(round(value)) == 0
