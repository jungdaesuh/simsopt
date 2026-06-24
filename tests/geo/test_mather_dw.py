r"""Tests for the Mather :math:`\Delta W` last-torus-flux certification.

Two analytic gates carry this suite. **Gauge invariance**: the closed-loop action
:math:`\oint\mathbf{A}\cdot d\boldsymbol{\ell}` must be unchanged by
:math:`\mathbf{A}\to\mathbf{A}+\nabla f`; the trapezoidal quadrature makes this
exact (round-off) on a closed loop, so the certificate can never be perturbed by a
gauge choice. **Stokes/flux consistency**: in the symmetric gauge the action of a
circle must equal the enclosed flux :math:`B_0\pi R^2`.

The physics gate is the **standard-map calibration**: a generating-function action
(hand-checked against the integrable ``K=0`` orbit and the exact ``(0, 1/2)``
period-2 orbit) whose golden-mean-convergent :math:`|\Delta W|` lifts from
``~1e-9`` (intact torus) to ``~1e-4`` (broken) across :math:`K_c\approx0.971635`.
A full multi-convergent breakup *sweep* (mapping :math:`K_c(\omega^\ast)` from the
convergent chain) is out of scope here; see ``test_standard_map_*`` and the
``SCOPE`` note in :func:`_standard_map_symmetric_orbit`.
"""

from __future__ import annotations

import numpy as np
import pytest

from examples.single_stage_optimization.banana_opt.topology.mather_dw import (
    GOLDEN_MEAN,
    MatherCertificate,
    closed_loop_action,
    golden_convergents,
    mather_delta_w,
    mather_delta_w_from_orbits,
    noble_convergents,
)


# --------------------------------------------------------------------------- #
# Helpers: closed loops and an analytic symmetric-gauge vector potential.
# --------------------------------------------------------------------------- #
def _unit_circle(radius: float, n_samples: int) -> np.ndarray:
    """``(N, 3)`` Cartesian samples of a radius-``radius`` circle in the z=0 plane.

    Sampled at ``N`` points over ``[0, 2*pi)`` with ``endpoint=False`` so the loop
    is closed by the final segment ``N-1 -> 0`` (the first point is not repeated),
    matching the closed-loop convention of :func:`closed_loop_action`.
    """
    angles = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)
    return np.stack(
        [radius * np.cos(angles), radius * np.sin(angles), np.zeros_like(angles)],
        axis=1,
    )


def _symmetric_gauge_a(points: np.ndarray, b0: float) -> np.ndarray:
    """Symmetric-gauge ``A = 0.5 * B0 * (-y, x, 0)`` sampled at ``points``.

    Its curl is the uniform axial field ``B = B0 z_hat``, so by Stokes a planar loop
    encloses flux ``B0 * area``.
    """
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * b0 * np.stack([-y, x, np.zeros_like(x)], axis=1)


# --------------------------------------------------------------------------- #
# Gauge invariance: the headline analytic gate.
# --------------------------------------------------------------------------- #
def test_closed_loop_action_is_gauge_invariant():
    # A -> A + grad f must not change the closed-loop action. The trapezoid makes
    # this exact (round-off) on a closed loop, so the tolerance is far tighter than
    # the 1e-9 the design asks for. f = sin(x) cos(y) + x y z, grad f analytic.
    loop = _unit_circle(1.0, 200)
    base_a = _symmetric_gauge_a(loop, b0=1.3)
    x, y, z = loop[:, 0], loop[:, 1], loop[:, 2]
    grad_f = np.stack(
        [
            np.cos(x) * np.cos(y) + y * z,  # d/dx [sin x cos y + x y z]
            -np.sin(x) * np.sin(y) + x * z,  # d/dy
            x * y,  # d/dz
        ],
        axis=1,
    )
    action_plain = closed_loop_action(loop, base_a)
    action_gauged = closed_loop_action(loop, base_a + grad_f)
    assert action_gauged == pytest.approx(action_plain, abs=1e-9)
    # The pure gradient circulates to (machine) zero around the closed loop.
    assert closed_loop_action(loop, grad_f) == pytest.approx(0.0, abs=1e-9)


def test_gauge_invariance_holds_on_a_nonplanar_wiggly_loop():
    # The gauge cancellation is structural (closed-loop telescoping), not an
    # artifact of the circle's symmetry: an off-plane wiggly closed loop cancels too.
    t = np.linspace(0.0, 2.0 * np.pi, 173, endpoint=False)
    loop = np.stack(
        [np.cos(t) + 0.3 * np.cos(3.0 * t), 1.7 * np.sin(t), 0.4 * np.sin(2.0 * t)],
        axis=1,
    )
    base_a = _symmetric_gauge_a(loop, b0=0.7)
    x, y, z = loop[:, 0], loop[:, 1], loop[:, 2]
    grad_f = np.stack(
        [np.cos(x) * np.cos(y) + y * z, -np.sin(x) * np.sin(y) + x * z, x * y],
        axis=1,
    )
    assert closed_loop_action(loop, base_a + grad_f) == pytest.approx(
        closed_loop_action(loop, base_a), abs=1e-9
    )


# --------------------------------------------------------------------------- #
# Stokes / flux consistency: action of a circle = enclosed flux B0 pi R^2.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(("b0", "radius"), [(1.0, 1.0), (2.5, 0.6), (0.4, 1.8)])
def test_closed_loop_action_matches_enclosed_flux(b0, radius):
    # Symmetric gauge => B = B0 z_hat => circle of radius R encloses flux B0 pi R^2.
    # Trapezoid on 256 samples reaches the ~1e-3 the design specifies.
    loop = _unit_circle(radius, 256)
    action = closed_loop_action(loop, _symmetric_gauge_a(loop, b0=b0))
    assert action == pytest.approx(b0 * np.pi * radius**2, abs=1e-3)


def test_flux_consistency_refines_with_more_samples():
    # The trapezoid error on the smooth circle shrinks as the sampling is refined.
    loop_coarse = _unit_circle(1.0, 64)
    loop_fine = _unit_circle(1.0, 512)
    target = 1.0 * np.pi * 1.0**2
    err_coarse = abs(closed_loop_action(loop_coarse, _symmetric_gauge_a(loop_coarse, 1.0)) - target)
    err_fine = abs(closed_loop_action(loop_fine, _symmetric_gauge_a(loop_fine, 1.0)) - target)
    assert err_fine < err_coarse


# --------------------------------------------------------------------------- #
# Delta W sign / zero.
# --------------------------------------------------------------------------- #
def test_delta_w_is_zero_for_equal_actions():
    assert mather_delta_w(1.5, 1.5) == 0.0


def test_delta_w_sign_follows_minimax_minus_minimizer():
    # X (minimax) above O (minimizer) => positive; swapped inputs flip the sign.
    assert mather_delta_w(2.0, 1.0) == pytest.approx(1.0)
    assert mather_delta_w(1.0, 2.0) == pytest.approx(-1.0)


def test_delta_w_rejects_non_finite_actions():
    with pytest.raises(ValueError, match="finite"):
        mather_delta_w(float("nan"), 1.0)
    with pytest.raises(ValueError, match="finite"):
        mather_delta_w(1.0, float("inf"))


# --------------------------------------------------------------------------- #
# Closed-loop action input validation (never certify a degenerate orbit).
# --------------------------------------------------------------------------- #
def test_closed_loop_action_rejects_bad_shapes_and_non_finite():
    loop = _unit_circle(1.0, 16)
    good_a = _symmetric_gauge_a(loop, 1.0)
    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        closed_loop_action(loop[:, :2], good_a[:, :2])  # (N, 2) points
    with pytest.raises(ValueError, match="match points shape"):
        closed_loop_action(loop, good_a[:-1])  # mismatched A length
    with pytest.raises(ValueError, match="at least 3 samples"):
        closed_loop_action(loop[:2], good_a[:2])  # too few to close a loop
    poisoned = good_a.copy()
    poisoned[3, 1] = np.inf
    with pytest.raises(ValueError, match="finite"):
        closed_loop_action(loop, poisoned)


# --------------------------------------------------------------------------- #
# Noble convergents -> golden mean (consecutive-Fibonacci ratios).
# --------------------------------------------------------------------------- #
def test_golden_convergents_are_consecutive_fibonacci_ratios():
    # F_i/F_{i+1}: 1/1, 1/2, 2/3, 3/5, 5/8, 8/13, 13/21, 21/34.
    expected = ((1, 1), (1, 2), (2, 3), (3, 5), (5, 8), (8, 13), (13, 21), (21, 34))
    assert golden_convergents(8) == expected


def test_golden_convergents_converge_to_golden_mean_monotonically():
    convergents = golden_convergents(12)
    errors = [abs(p / q - GOLDEN_MEAN) for p, q in convergents]
    assert errors[-1] < 1e-5
    # Continued-fraction convergents bracket the limit, so the error strictly
    # shrinks at every step.
    for earlier, later in zip(errors, errors[1:]):
        assert later < earlier


def test_noble_convergents_match_golden_helper_and_handle_other_tails():
    # golden_convergents is the all-ones tail of noble_convergents.
    assert noble_convergents((1, 1, 1, 1, 1), 5) == golden_convergents(5)
    # Silver mean [0; 2, 2, 2, ...] gives the Pell-number ratios 1/2, 2/5, 5/12, 12/29.
    assert noble_convergents((2, 2, 2, 2), 4) == ((1, 2), (2, 5), (5, 12), (12, 29))
    # depth is truncated to the available tail length.
    assert noble_convergents((1, 1), 10) == ((1, 1), (1, 2))


def test_noble_convergents_reject_bad_input():
    with pytest.raises(ValueError, match="depth must be positive"):
        noble_convergents((1, 1), 0)
    with pytest.raises(ValueError, match="at least one"):
        noble_convergents((), 3)
    with pytest.raises(ValueError, match="positive"):
        noble_convergents((1, 0, 1), 3)


# --------------------------------------------------------------------------- #
# Standard-map generating-function action and Aubry-Mather orbits.
#
# Chirikov-Taylor generating function and its stationarity (the standard map):
#   F(x, x') = 0.5*(x' - x)^2 - (K/(2pi)^2) * cos(2pi x),
#   period-q action  W = sum_{n=0}^{q-1} F(x_n, x_{n+1})   with  x_q = x_0 + p,
#   dW/dx_n = 0  <=>  x_{n+1} - 2 x_n + x_{n-1} = (K/2pi) * sin(2pi x_n).
# The two dominant SYMMETRIC period-q orbits are the minimizing (O, elliptic for
# small K) one on the x=0 symmetry line and the minimax (X, hyperbolic) one on the
# half-spacing-shifted line; the O point is the action minimizer so W_X >= W_O.
# --------------------------------------------------------------------------- #
STANDARD_MAP_K_CRITICAL = 0.971635  # golden-mean last-torus breakup of the standard map.


def _standard_map_action(orbit: np.ndarray, p: int, k_strength: float) -> float:
    """Generating-function action of a period-``q`` ``p``-winding standard-map orbit.

    ``orbit`` is the length-``q`` sequence ``x_0..x_{q-1}`` on the universal cover;
    the chain closes with ``x_q = x_0 + p``. Returns
    ``sum_n 0.5*(x_{n+1}-x_n)^2 - (K/(2pi)^2) cos(2pi x_n)``.
    """
    extended = np.append(orbit, orbit[0] + p)
    steps = extended[1:] - extended[:-1]
    return float(
        np.sum(0.5 * steps**2 - (k_strength / (2.0 * np.pi) ** 2) * np.cos(2.0 * np.pi * orbit))
    )


def _standard_map_residual(orbit: np.ndarray, p: int, k_strength: float) -> np.ndarray:
    """Stationarity residual ``x_{n+1} - 2 x_n + x_{n-1} - (K/2pi) sin(2pi x_n)``.

    Periodic with ``x_{n+q} = x_n + p`` (the wrap carries the winding ``p``); the
    residual vanishes exactly on a period-``q`` orbit.
    """
    prev = np.roll(orbit, 1)
    nxt = np.roll(orbit, -1)
    prev[0] = orbit[-1] - p
    nxt[-1] = orbit[0] + p
    return (nxt - 2.0 * orbit + prev) - (k_strength / (2.0 * np.pi)) * np.sin(2.0 * np.pi * orbit)


def _standard_map_symmetric_orbit(
    p: int,
    q: int,
    k_strength: float,
    line_offset: float,
) -> np.ndarray:
    r"""Newton-solve a symmetric period-``q`` standard-map orbit from a uniform seed.

    Seeded with the uniform rotation ``x_n = (p/q) n + line_offset`` (``line_offset``
    selects the symmetry line: ``0`` -> minimizing/O orbit on the ``x=0`` line,
    ``0.5/q`` -> minimax/X orbit on the half-spacing-shifted line). A damped Newton
    on the tridiagonal (periodic) stationarity Jacobian drives the residual to
    ``< 1e-12``; the step is capped so it stays in the seed's basin below ``K_c``.

    SCOPE: this focused solver converges for ``K`` up to ~1.3 on the golden-mean
    convergents tested. A full breakup *sweep* -- continuing each O/X pair across
    ``K_c`` for the whole convergent chain to map ``K_c(omega_star)`` -- is a larger
    undertaking left as a documented TODO; the tests below pin the action against
    hand-checked orbits and the ``|Delta W|`` lift across ``K_c`` for fixed
    convergents, which is what calibrates the diagnostic.
    """
    indices = np.arange(q)
    orbit = (p / q) * indices + line_offset
    for _ in range(500):
        residual = _standard_map_residual(orbit, p, k_strength)
        if np.max(np.abs(residual)) < 1e-12:
            break
        diagonal = -2.0 - k_strength * np.cos(2.0 * np.pi * orbit)
        jacobian = (
            np.diag(diagonal)
            + np.diag(np.ones(q - 1), 1)
            + np.diag(np.ones(q - 1), -1)
        )
        jacobian[0, -1] += 1.0  # periodic wrap couples x_0 and x_{q-1}
        jacobian[-1, 0] += 1.0
        step = np.linalg.solve(jacobian, -residual)
        step_norm = float(np.max(np.abs(step)))
        if step_norm > 0.3:
            step *= 0.3 / step_norm  # trust cap: keep the seed's symmetry basin
        orbit = orbit + step
    return orbit


def _standard_map_delta_w(p: int, q: int, k_strength: float) -> float:
    """``Delta W = W_X - W_O`` for the ``p/q`` chain, O = minimizer, X = minimax.

    Solves both symmetric orbits, labels the lower-action one O and the higher-action
    one X (so the result is non-negative), and returns ``mather_delta_w`` of the two.
    """
    orbit_zero_line = _standard_map_symmetric_orbit(p, q, k_strength, line_offset=0.0)
    orbit_half_line = _standard_map_symmetric_orbit(p, q, k_strength, line_offset=0.5 / q)
    action_zero = _standard_map_action(orbit_zero_line, p, k_strength)
    action_half = _standard_map_action(orbit_half_line, p, k_strength)
    action_o = min(action_zero, action_half)  # minimizing (O) orbit
    action_x = max(action_zero, action_half)  # minimax (X) orbit
    return mather_delta_w(action_x, action_o)


def test_standard_map_action_matches_integrable_closed_form():
    # K=0: the action is purely kinetic and a uniform-rotation period-q orbit has
    # every step exactly p/q, so W = q * 0.5 * (p/q)^2 = p^2/(2q). Hand-checkable.
    for p, q in [(1, 2), (2, 3), (3, 5), (5, 8)]:
        orbit = (p / q) * np.arange(q)
        assert _standard_map_action(orbit, p, 0.0) == pytest.approx(p**2 / (2 * q))


def test_standard_map_action_on_exact_period_two_orbit_is_hand_checked():
    # (x0, x1) = (0, 1/2) is an EXACT p=1 period-2 orbit for ALL K: the stationarity
    # residual vanishes since sin(0) = sin(pi) = 0. Its action is 0.25 for every K --
    # the two cosine terms cos(0)=+1 and cos(pi)=-1 cancel, leaving only the kinetic
    # 2 * 0.5 * (1/2)^2 = 0.25. This pins both the residual and the action by hand.
    orbit = np.array([0.0, 0.5])
    for k_strength in [0.0, 0.5, 1.0, 1.3]:
        assert np.max(np.abs(_standard_map_residual(orbit, 1, k_strength))) == pytest.approx(
            0.0, abs=1e-14
        )
        assert _standard_map_action(orbit, 1, k_strength) == pytest.approx(0.25)


def test_standard_map_orbits_converge_to_machine_residual():
    # The focused Newton solver must actually find genuine orbits (residual -> 0),
    # not just return the seed -- otherwise the Delta W comparison is meaningless.
    for p, q in [(5, 8), (8, 13)]:
        for k_strength in [0.3, 0.9, 1.2]:
            for offset in (0.0, 0.5 / q):
                orbit = _standard_map_symmetric_orbit(p, q, k_strength, offset)
                residual = np.max(np.abs(_standard_map_residual(orbit, p, k_strength)))
                assert residual < 1e-10


def test_standard_map_delta_w_is_nonnegative_with_minimizer_as_o():
    # Labeling O = action minimizer forces W_X >= W_O, i.e. Delta W >= 0, at every K.
    for k_strength in [0.3, 0.6, 0.9, STANDARD_MAP_K_CRITICAL, 1.1, 1.3]:
        assert _standard_map_delta_w(5, 8, k_strength) >= -1e-12


def test_standard_map_delta_w_lifts_across_k_critical():
    # THE physics gate for one convergent: deep below K_c the 5/8 torus is intact and
    # |Delta W| ~ 1e-9; across K_c it lifts by orders of magnitude. Verified values
    # (5/8): K=0.3 -> 7.9e-9, K=0.9 -> 5.0e-5, K=1.3 -> 7.6e-4 (monotone increasing).
    p, q = 5, 8
    dw_deep_intact = _standard_map_delta_w(p, q, 0.3)
    dw_near_below = _standard_map_delta_w(p, q, 0.9)
    dw_above = _standard_map_delta_w(p, q, 1.3)
    # Deep below K_c the convergent action gap is tiny (intact torus).
    assert dw_deep_intact < 1e-7
    # Crossing toward and past K_c lifts |Delta W| monotonically by orders of magnitude.
    assert dw_near_below > 10.0 * dw_deep_intact
    assert dw_above > 10.0 * dw_near_below
    assert dw_above > 1e-4


def test_standard_map_delta_w_grows_monotonically_in_k():
    # The lift is monotone in K across the breakup, not a single lucky pair of points.
    k_values = [0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 1.2, 1.3]
    delta_w = [_standard_map_delta_w(5, 8, k) for k in k_values]
    for earlier, later in zip(delta_w, delta_w[1:]):
        assert later > earlier


def test_higher_convergent_has_smaller_delta_w_below_k_critical():
    # Aubry-Mather convergence: at fixed K below K_c the deeper convergent (8/13,
    # closer to the golden mean) tracks the true torus better, so its |Delta W| is
    # smaller than the shallower 5/8 convergent's.
    k_below = 0.6
    dw_shallow = _standard_map_delta_w(5, 8, k_below)
    dw_deep = _standard_map_delta_w(8, 13, k_below)
    assert dw_deep < dw_shallow


# --------------------------------------------------------------------------- #
# MatherCertificate dataclass.
# --------------------------------------------------------------------------- #
def test_mather_certificate_carries_chain_and_verdict():
    convergents = ((3, 5, 1e-9), (5, 8, 8e-9), (8, 13, 5e-12))
    certificate = MatherCertificate(
        omega_star=GOLDEN_MEAN,
        convergents=convergents,
        terminal_abs_delta_w=5e-12,
        torus_intact=True,
    )
    assert certificate.omega_star == pytest.approx(GOLDEN_MEAN)
    assert certificate.convergents == convergents
    assert certificate.terminal_abs_delta_w == pytest.approx(5e-12)
    assert certificate.torus_intact is True
    assert certificate.heuristic is False  # default: twist regime, not flagged


def test_mather_certificate_heuristic_flag_marks_nontwist():
    certificate = MatherCertificate(
        omega_star=0.5,
        convergents=((1, 2, 0.3),),
        terminal_abs_delta_w=0.3,
        torus_intact=False,
        heuristic=True,
    )
    assert certificate.heuristic is True
    assert certificate.torus_intact is False


def test_mather_certificate_rejects_invalid_fields():
    with pytest.raises(ValueError, match="omega_star must be finite"):
        MatherCertificate(
            omega_star=float("nan"),
            convergents=(),
            terminal_abs_delta_w=0.0,
            torus_intact=False,
        )
    with pytest.raises(ValueError, match="non-negative"):
        MatherCertificate(
            omega_star=0.5,
            convergents=(),
            terminal_abs_delta_w=-1.0,
            torus_intact=False,
        )
    with pytest.raises(ValueError, match="q must be positive"):
        MatherCertificate(
            omega_star=0.5,
            convergents=((1, 0, 0.1),),
            terminal_abs_delta_w=0.1,
            torus_intact=False,
        )


# --------------------------------------------------------------------------- #
# Field-extraction adapter: Delta W from O/X orbits sampled in a field.
# --------------------------------------------------------------------------- #
class _StubSymmetricGaugeField:
    """Minimal ``BiotSavart``-like stub: ``A = 0.5 * B0 * (-y, x, 0)`` at set points.

    Implements only the ``set_points`` / ``A`` surface
    :func:`mather_delta_w_from_orbits` uses, so the adapter can be exercised without
    constructing real coils. The analytic gauge makes the action of a planar circle
    equal to the enclosed flux ``B0 pi R^2``.
    """

    def __init__(self, b0: float) -> None:
        self._b0 = float(b0)
        self._points = np.zeros((0, 3))

    def set_points(self, points: np.ndarray) -> None:
        self._points = np.asarray(points, dtype=float)

    def A(self) -> np.ndarray:
        return _symmetric_gauge_a(self._points, self._b0)


def test_delta_w_from_orbits_scores_two_loops_via_field_flux():
    # Two concentric circles in the symmetric gauge: their actions are the enclosed
    # fluxes B0 pi R^2, so Delta W = W_X - W_O = B0 pi (R_x^2 - R_o^2). With the outer
    # circle as X this is a clean analytic check of the field adapter wiring.
    field = _StubSymmetricGaugeField(b0=1.5)
    o_loop = _unit_circle(0.8, 256)
    x_loop = _unit_circle(1.2, 256)
    delta_w = mather_delta_w_from_orbits(field, o_loop, x_loop)
    expected = 1.5 * np.pi * (1.2**2 - 0.8**2)
    assert delta_w == pytest.approx(expected, abs=1e-3)


def test_delta_w_from_orbits_is_zero_for_identical_orbits():
    # Same loop for O and X => identical sampled action => Delta W exactly 0.
    field = _StubSymmetricGaugeField(b0=2.0)
    loop = _unit_circle(1.0, 200)
    assert mather_delta_w_from_orbits(field, loop, loop) == pytest.approx(0.0, abs=1e-12)


def test_action_field_passes_c_contiguous_points_to_set_points():
    """A non-C-contiguous orbit must reach ``set_points`` as a C-contiguous array.

    The pybind ``simsoptpp.BiotSavart.set_points`` overload rejects a non-C-contiguous
    (N, 3) array; a Cartesian orbit assembled by column-stacking is column-strided, and
    a bare ``reshape(-1, 3)`` preserves those strides. Regression: feed a Fortran-ordered
    (non-C-contiguous) trajectory through a stub whose ``set_points`` asserts contiguity.
    This FAILS under the old bare-reshape and passes after ``np.ascontiguousarray``.
    """

    class _ContiguityCheckingField(_StubSymmetricGaugeField):
        def set_points(self, points: np.ndarray) -> None:
            assert points.flags["C_CONTIGUOUS"], "set_points received non-contiguous points"
            super().set_points(points)

    field = _ContiguityCheckingField(b0=1.0)
    strided = np.asfortranarray(_unit_circle(1.0, 64))  # same orbit, column-major
    assert not strided.flags["C_CONTIGUOUS"]
    assert mather_delta_w_from_orbits(field, strided, strided) == pytest.approx(0.0, abs=1e-12)


def test_delta_w_from_orbits_rejects_bad_trajectory_shape():
    field = _StubSymmetricGaugeField(b0=1.0)
    good_loop = _unit_circle(1.0, 32)
    bad_loop = good_loop[:, :2]  # (N, 2), not Cartesian (N, 3)
    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        mather_delta_w_from_orbits(field, bad_loop, good_loop)
