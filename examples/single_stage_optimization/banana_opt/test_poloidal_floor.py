"""Unit + FD/Taylor tests for the poloidal-extent FLOOR (spread) objective.

This pins the *contract* of ``PoloidalExtentFloor`` and its wiring into the
single-stage total objective, not the source text:

  J = (1/p) * max(theta_floor - smoothmax|theta_in|, 0)^p   [rad^p]

a one-sided squared hinge that is zero once the peak inboard poloidal extent
reaches ``theta_floor`` and positive-and-decreasing-in-extent below it, so
descent spreads the banana-tip U-turn further poloidally. It is the complement
of ``PoloidalExtent`` (which caps the peak from above).

Each assertion is tied to an externally checkable consequence:
  * the analytic hinge value on a controlled-angle point cloud (constant profile
    => the normalized log-mean-exp smooth max is *exact*, so J is exact),
  * monotone decrease of J as the realized extent rises toward the floor,
  * the smooth max tracking the hard max within the temperature band (the
    property that makes a "spread to 86 deg" target meaningful),
  * the *additive, default-OFF byte-identical* contract verified through the
    real ``build_total_objective`` (floor None => identical objective; floor
    present => objective rises by exactly weight * floor.J()),
  * a JAX-gradient Taylor test of dJ on a real ``CurveXYZFourier``.

No test mirrors the implementation.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

import jax

# Production runs the coil objectives in double precision; the gradient Taylor
# test needs it for a clean second-order slope.
jax.config.update("jax_enable_x64", True)

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXAMPLE_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _EXAMPLE_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from simsopt.geo import CurveXYZFourier  # noqa: E402

from banana_opt.poloidal_extent import (  # noqa: E402
    PoloidalExtentFloor,
    _poloidal_floor_pure,
    max_poloidal_extent_rad,
    POLOIDAL_FLOOR_SMOOTH_MAX_TEMPERATURE_RAD as _TEMP,
)
import single_stage_objectives as sso  # noqa: E402


_R_WINDING = 1.0
_RHO = 0.1


def _gamma_from_inboard_angles(angles):
    """Point cloud whose inboard poloidal angle about ``_R_WINDING`` IS ``angles``.

    With y=0 and x=R, the term computes theta_in = arctan2(Z, -(R - R_winding));
    placing x = R_winding - rho*cos(theta), z = rho*sin(theta) makes theta_in
    reproduce ``theta`` exactly (rho cancels), so we control the angle profile.
    """
    angles = np.asarray(angles, dtype=float)
    x = _R_WINDING - _RHO * np.cos(angles)
    y = np.zeros_like(angles)
    z = _RHO * np.sin(angles)
    return np.stack([x, y, z], axis=-1)


def _floor_J_on_angles(angles, theta_floor, p=2):
    gamma = _gamma_from_inboard_angles(angles)
    return float(
        _poloidal_floor_pure(gamma, _R_WINDING, 0.0, theta_floor, _TEMP, p)
    )


# --------------------------------------------------------------------------- #
# A. The hinge value on a controlled-angle point cloud
# --------------------------------------------------------------------------- #
def test_floor_inactive_once_peak_reaches_target_is_exactly_zero():
    """Constant extent at/above the floor => smooth max == extent => J == 0."""
    # Constant profile at 1.2 rad, floor 1.0 rad: peak exceeds floor.
    assert _floor_J_on_angles([1.2] * 48, theta_floor=1.0) == 0.0
    # Exactly at the floor is also the join value 0 (one-sided hinge).
    assert _floor_J_on_angles([1.0] * 48, theta_floor=1.0) == pytest.approx(
        0.0, abs=1e-12
    )


def test_floor_active_below_target_matches_squared_hinge():
    """Constant extent below the floor => exact 0.5*(floor - extent)^2.

    For a constant |theta| profile the normalized log-mean-exp smooth max equals
    the value exactly, so the hinge value is analytic and independent of the
    smoothing temperature -- the contract a squared floor must satisfy.
    """
    floor = 1.0
    for extent in (0.3, 0.5, 0.9):
        expected = 0.5 * (floor - extent) ** 2
        assert _floor_J_on_angles([extent] * 48, theta_floor=floor) == pytest.approx(
            expected, rel=1e-9, abs=1e-12
        )


def test_floor_value_decreases_as_extent_rises_toward_target():
    """The spread incentive weakens monotonically as the coil reaches outward."""
    floor = 1.0
    vals = [_floor_J_on_angles([c] * 48, theta_floor=floor) for c in (0.3, 0.6, 0.9)]
    assert vals[0] > vals[1] > vals[2] > 0.0


def test_smooth_max_tracks_hard_max_within_temperature_band():
    """Floor target is meaningful only if smoothmax ~ hard max for a peaked profile.

    Recover the smooth max used inside J (J = 0.5*(floor - smoothmax)^2 with the
    floor set just above the true peak) and confirm it sits just below the hard
    max -- never far below it -- so pushing toward an 86 deg floor really pushes
    the peak, not some mean-biased proxy.
    """
    angles = np.linspace(-0.9, 1.1, 200)  # asymmetric, hard max |theta| = 1.1
    hard_max = float(np.max(np.abs(angles)))
    floor = hard_max + 0.2
    J = _floor_J_on_angles(angles, theta_floor=floor)
    smooth_max = floor - np.sqrt(2.0 * J)
    assert smooth_max <= hard_max + 1e-9
    assert smooth_max >= hard_max - 5.0 * _TEMP  # within the smoothing band


# --------------------------------------------------------------------------- #
# B. A real curve: J wraps the pure term; dJ passes a Taylor test
# --------------------------------------------------------------------------- #
def _make_curve(seed=0):
    """A closed CurveXYZFourier loop with a finite, nonzero inboard extent."""
    order = 3
    curve = CurveXYZFourier(quadpoints=96, order=order)
    x = np.zeros(curve.dof_size)
    # Base loop in the x-z plane, offset outboard of the winding axis with a
    # mild perturbation so dJ is exercised on a generic (non-degenerate) curve.
    # CurveXYZFourier dof layout per coordinate: [c0, (s_k, c_k) for k=1..order].
    block = 2 * order + 1
    x[0] = _R_WINDING + 0.20          # x: c0 (loop center outboard of axis)
    x[2] = 0.08                        # x: c1 (cos) -> R modulation
    x[block + 1] = 0.08                # y: s1 (sin) -> small out-of-plane lobe
    x[2 * block + 1] = 0.08            # z: s1 (sin) -> Z modulation
    rng = np.random.default_rng(seed)
    x = x + 1e-3 * rng.standard_normal(curve.dof_size)
    curve.x = x
    return curve


def test_curve_J_matches_pure_term_and_floor_activation():
    curve = _make_curve()
    extent = max_poloidal_extent_rad(curve, _R_WINDING)
    floor = extent + 0.3  # active: peak below floor
    term = PoloidalExtentFloor(curve, _R_WINDING, floor)
    # J wraps the same pure computation evaluated on the curve geometry.
    expected = float(
        _poloidal_floor_pure(curve.gamma(), _R_WINDING, 0.0, floor, _TEMP, 2)
    )
    assert term.J() == pytest.approx(expected, rel=1e-12, abs=1e-14)
    assert term.J() > 0.0
    # A floor below the realized extent is inactive.
    inactive = PoloidalExtentFloor(curve, _R_WINDING, extent - 0.3)
    assert inactive.J() == pytest.approx(0.0, abs=1e-12)


def test_curve_gradient_centered_taylor_is_second_order():
    """Centered FD of J along curve dofs matches dJ at second order."""
    curve = _make_curve()
    extent = max_poloidal_extent_rad(curve, _R_WINDING)
    term = PoloidalExtentFloor(curve, _R_WINDING, extent + 0.3)
    x0 = curve.x.copy()

    def build_J(x):
        curve.x = x
        return term.J()

    grad0 = np.asarray(term.dJ(), dtype=float)
    assert grad0.shape == (x0.size,)
    rng = np.random.default_rng(1)
    direction = rng.standard_normal(x0.size)
    direction /= np.linalg.norm(direction)
    g_dot_h = float(grad0 @ direction)

    # Small h: the normalized log-mean-exp smooth max has high curvature at the
    # production temperature, so the asymptotic second-order regime is entered
    # only once h is well below the smoothing scale (verified: err ~1e-5 at
    # h=4e-4, falling ~4x per halving). Larger h sees the stiffness, not a bug.
    schedule = (4e-4, 2e-4, 1e-4, 5e-5)
    errors = []
    for h in schedule:
        plus = build_J(x0 + h * direction)
        minus = build_J(x0 - h * direction)
        approx = (plus - minus) / (2.0 * h)
        errors.append(abs(approx - g_dot_h))
    curve.x = x0  # restore
    # Halving h shrinks the centered-FD error by ~4x (second order); require >3x
    # while comfortably above the round-off floor.
    for k in range(1, len(errors)):
        if errors[k - 1] > 1e-10:
            assert errors[k] < errors[k - 1] / 3.0


# --------------------------------------------------------------------------- #
# C. Additive, default-OFF byte-identical contract via build_total_objective
# --------------------------------------------------------------------------- #
def _base_total_objective(curve, **floor_kwargs):
    """build_total_objective with one shared cheap term in every base slot.

    JnonQSRatio enters with coefficient 1 and all weighted slots use weight 0,
    so the base objective J equals that shared term's J -- a clean, real anchor
    against which the floor's additive contribution is isolated.
    """
    from simsopt.geo import CurveLength

    anchor = CurveLength(curve)
    return sso.build_total_objective(
        anchor,   # JnonQSRatio (coeff 1)
        0.0, anchor,    # RES_WEIGHT, JBoozerResidual
        0.0, anchor,    # IOTAS_WEIGHT, Jiota
        0.0, None,      # VOLUME_WEIGHT, JVolume
        0.0, anchor,    # LENGTH_WEIGHT, JCurveLength
        0.0, anchor,    # CC_WEIGHT, JCurveCurve
        0.0, anchor,    # CS_WEIGHT, JCurveSurface
        0.0, anchor,    # CURVATURE_WEIGHT, JCurvature
        **floor_kwargs,
    )


def test_total_objective_floor_off_is_byte_identical():
    curve = _make_curve()
    j_omitted = _base_total_objective(curve).J()
    j_none = _base_total_objective(
        curve, JPoloidalExtentFloor=None, POLOIDAL_FLOOR_WEIGHT=5.0
    ).J()
    assert j_none == j_omitted  # None floor term => never added


def test_total_objective_floor_adds_exactly_weight_times_term():
    curve = _make_curve()
    extent = max_poloidal_extent_rad(curve, _R_WINDING)
    floor = PoloidalExtentFloor(curve, _R_WINDING, extent + 0.3)
    weight = 7.5
    base = _base_total_objective(curve).J()
    with_floor = _base_total_objective(
        curve, JPoloidalExtentFloor=floor, POLOIDAL_FLOOR_WEIGHT=weight
    ).J()
    assert floor.J() > 0.0
    assert with_floor - base == pytest.approx(weight * floor.J(), rel=1e-10, abs=1e-12)
