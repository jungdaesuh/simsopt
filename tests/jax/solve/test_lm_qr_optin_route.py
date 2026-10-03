"""Contract tests for the dense-QR Levenberg-Marquardt route.

Background
----------
``Driver.SIMSOPT_LM_QR`` (``least_squares_algorithm="lm-minpack"``, method
``"lm-minpack-ondevice"``) is the one Levenberg-Marquardt lane. Its inner step
factorizes the dense damped Jacobian with column-pivoted QR (MINPACK-style,
factorize-once), mirroring upstream ``least_squares(method="lm")``.

These tests pin the properties the lane ships with:

1. routing: ``lm-minpack`` is the ``host-jax`` Boozer default and the only
   spelling that reaches ``SIMSOPT_LM_QR``; the other backends default to the
   quasi-Newton route,
2. the executed route is readable off the result,
3. it reaches the closed-form / recoverable optimum on reference fixtures and
   follows MINPACK ``lmder`` step for step (``scipy.optimize.least_squares``
   with ``method="lm"``, ``x_scale=1.0``),
4. its dense materialization is refused up front against a declared byte
   budget, and
5. repeated solves reuse one compiled executable instead of retracing per call,
   while problems with different embedded constants keep their own.
"""

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import jax
import numpy as np
import pytest
import jax.numpy as jnp
import scipy.linalg
import scipy.optimize

from simsopt_jax.geo.optimizers.single_stage_routing import (
    resolve_boozer_least_squares_algorithm,
)
import simsopt_jax.geo.optimizers.optimizer as _opt
from simsopt_jax.solve.dispatch import least_squares
from simsopt_jax.solve.driver import legacy_target_least_squares_method
from simsopt_jax.solve import (
    Driver,
    SimsoptLMQROptions,
)


# --------------------------------------------------------------------------
# Reference least-squares fixtures
# --------------------------------------------------------------------------

_FIXTURE_SEED = 20260816


def _linear_fixture():
    """Overdetermined linear least squares with a closed-form optimum."""
    rng = np.random.default_rng(_FIXTURE_SEED)
    matrix = jnp.asarray(rng.standard_normal((40, 8)))
    rhs = jnp.asarray(rng.standard_normal(40))

    def residual(x):
        return matrix @ x - rhs

    optimum = np.linalg.lstsq(np.asarray(matrix), np.asarray(rhs), rcond=None)[0]
    return residual, jnp.zeros(8), optimum


def _nonlinear_fixture():
    """Well-conditioned exponential fit with an exactly recoverable optimum."""
    grid = jnp.linspace(0.0, 3.0, 60)
    truth = jnp.array([2.5, -0.8, 0.4])
    data = truth[0] * jnp.exp(truth[1] * grid) + truth[2]

    def residual(x):
        return x[0] * jnp.exp(x[1] * grid) + x[2] - data

    return residual, jnp.array([1.0, -0.3, 0.0]), np.asarray(truth)


def _solve_qr_lane(residual, x0, *, maxiter=400):
    return least_squares(
        residual,
        x0,
        driver=Driver.SIMSOPT_LM_QR,
        options=SimsoptLMQROptions(maxiter=maxiter),
    )


# --------------------------------------------------------------------------
# 1. Routing: the host-jax default is the QR lane; the others stay quasi-Newton
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("boozer_optimizer_backend", "expected_algorithm"),
    [
        ("ondevice", "quasi-newton"),
        ("host-jax", "lm-minpack"),
        ("scipy", "quasi-newton"),
    ],
)
def test_default_least_squares_algorithm_per_backend(
    boozer_optimizer_backend,
    expected_algorithm,
):
    """With no explicit request, only ``host-jax`` defaults to ``lm-minpack``."""
    resolved = resolve_boozer_least_squares_algorithm(boozer_optimizer_backend)

    assert resolved == expected_algorithm


def test_qr_lane_requires_an_explicit_algorithm_string():
    """``lm-minpack`` is the only spelling that reaches ``SIMSOPT_LM_QR``."""
    assert (
        resolve_boozer_least_squares_algorithm(
            "ondevice",
            least_squares_algorithm="lm-minpack",
        )
        == "lm-minpack"
    )
    assert (
        _opt.resolve_boozer_inner_driver(
            "ondevice",
            limited_memory=False,
            least_squares_algorithm="lm-minpack",
        )
        is Driver.SIMSOPT_LM_QR
    )


def test_lm_minpack_algorithm_string_binds_only_the_qr_driver():
    """``lm-minpack`` is a valid choice owned by the QR driver alone."""
    assert "lm-minpack" in _opt.VALID_LEAST_SQUARES_ALGORITHMS

    # Every Boozer inner driver whose options carry ``lm-minpack`` must be the
    # QR driver, so the algorithm string cannot leak into another lane.
    for driver, options in _opt._BOOZER_INNER_DRIVER_OPTIONS.items():
        if options.least_squares_algorithm == "lm-minpack":
            assert driver is Driver.SIMSOPT_LM_QR


# --------------------------------------------------------------------------
# 2. Route-string visibility: receipts can pin the lane
# --------------------------------------------------------------------------


def test_result_records_the_qr_route_explicitly():
    """A receipt must be able to read the executed route off the result."""
    residual, x0, _ = _linear_fixture()

    result = least_squares(
        residual,
        x0,
        driver=Driver.SIMSOPT_LM_QR,
        options=SimsoptLMQROptions(maxiter=50),
    )

    assert result.driver is Driver.SIMSOPT_LM_QR
    assert result.driver.value == "simsopt_lm_qr"
    assert legacy_target_least_squares_method(Driver.SIMSOPT_LM_QR) == (
        "lm-minpack-ondevice"
    )


# --------------------------------------------------------------------------
# 3. Correctness: the QR lane reaches the reference optimum
# --------------------------------------------------------------------------


def test_qr_lane_reaches_the_linear_least_squares_optimum():
    residual, x0, optimum = _linear_fixture()
    qr = _solve_qr_lane(residual, x0)

    assert qr.success

    # The lane must land on the closed-form ``lstsq`` optimum.
    np.testing.assert_allclose(np.asarray(qr.x), optimum, rtol=0, atol=1e-7)


def test_qr_lane_reaches_the_nonlinear_least_squares_optimum():
    residual, x0, optimum = _nonlinear_fixture()
    qr = _solve_qr_lane(residual, x0)

    assert qr.success

    np.testing.assert_allclose(np.asarray(qr.x), optimum, rtol=0, atol=1e-7)

    # The lane drives the residual to (numerical) zero on this fixture.
    assert float(qr.fun) < 1e-14


# Rosenbrock chains: two coupled banana valleys in three variables. From these
# starts an LM whose step bound is bookkeeping, detached from the step it takes,
# collapses the bound and stops on xtol far from the minimum (cost 2.1 to 1e2);
# MINPACK's lmder, where the bound sets the Marquardt parameter, reaches zero.
_ROSENBROCK_CHAIN_CASES = (
    (10.0, (-1.2, 1.0, 1.0)),
    (10.0, (-1.2, 1.0, -1.2)),
    (10.0, (-1.5, 2.0, 0.5)),
    (100.0, (-1.2, 1.0, -1.2)),
    (100.0, (0.0, 0.0, 0.0)),
)
# MINPACK ``info`` -> ``least_squares`` ``status`` (SciPy's
# ``FROM_MINPACK_TO_COMMON``).
_MINPACK_INFO_TO_SCIPY_STATUS = {1: 2, 2: 3, 3: 4, 4: 1, 5: 0}
# SciPy's own defaults for method="lm", passed explicitly to both sides.
_MINPACK_TOLERANCES = {"ftol": 1e-8, "xtol": 1e-8, "gtol": 1e-8}
_MINPACK_MAX_NFEV = 300


def _rosenbrock_chain(scale, *, anchor_last=False):
    def residual(x):
        terms = [
            scale * (x[1] - x[0] ** 2),
            1.0 - x[0],
            scale * (x[2] - x[1] ** 2),
            1.0 - x[1],
        ]
        if anchor_last:
            terms.append(1.0 - x[2])
        return jnp.stack(terms)

    return residual


def _solve_both_minpack_lanes(residual, x0, *, max_nfev=_MINPACK_MAX_NFEV):
    """Run the QR lane and SciPy's MINPACK lmder on one problem."""
    x0 = np.asarray(x0, dtype=np.float64)
    fun = jax.jit(residual)
    jac = jax.jit(jax.jacfwd(residual))
    reference = scipy.optimize.least_squares(
        lambda x: np.asarray(fun(x)),
        x0,
        jac=lambda x: np.asarray(jac(x)),
        method="lm",
        x_scale=1.0,
        max_nfev=max_nfev,
        **_MINPACK_TOLERANCES,
    )
    lane = _opt.target_least_squares(
        residual,
        jnp.asarray(x0),
        method="lm-minpack-ondevice",
        maxiter=max_nfev,
        options=dict(_MINPACK_TOLERANCES),
    )
    return lane, reference


@pytest.mark.parametrize(("scale", "x0"), _ROSENBROCK_CHAIN_CASES)
def test_qr_lane_converges_on_rosenbrock_chains_like_minpack(scale, x0):
    """The lane reaches the zero-residual minimum on the same MINPACK path."""
    lane, reference = _solve_both_minpack_lanes(_rosenbrock_chain(scale), x0)

    assert reference.cost <= 1e-20
    assert lane.success, lane.message
    assert float(lane.fun) <= 1e-20
    np.testing.assert_allclose(np.asarray(lane.x), np.ones(3), rtol=0, atol=1e-10)
    # Same trial count, Jacobian count and MINPACK stop: the lane walks
    # lmder's path.
    assert lane.nfev == reference.nfev
    assert lane.njev == reference.njev
    assert _MINPACK_INFO_TO_SCIPY_STATUS[lane.info] == reference.status


def test_qr_lane_matches_minpack_at_a_nonzero_residual_minimum():
    """Anchoring x[2] leaves a local minimum with cost 1.85; both stop on ftol."""
    lane, reference = _solve_both_minpack_lanes(
        _rosenbrock_chain(10.0, anchor_last=True),
        (-1.2, 1.0, 1.0),
    )

    assert reference.cost > 1.0
    assert lane.success, lane.message
    np.testing.assert_allclose(float(lane.fun), reference.cost, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(lane.x), reference.x, rtol=0, atol=1e-8)
    assert lane.nfev == reference.nfev
    assert _MINPACK_INFO_TO_SCIPY_STATUS[lane.info] == reference.status


def test_qr_lane_never_reports_success_with_a_nonfinite_jacobian():
    """An accepted step can land where the residual is finite but J is not.

    From t = 1 the Gauss-Newton step reaches t = 0, where sqrt|t| has an
    infinite slope, and that same step meets ftol. The solve must fail
    rather than return success with a nonfinite Jacobian and gradient.
    """

    def residual(x):
        t = x[0]
        return jnp.stack((t + (t - 1.0) ** 2 * jnp.sqrt(jnp.abs(t)), 1.0e5 + 0.0 * t))

    result = _opt.target_least_squares(
        residual, jnp.ones(1), method="lm-minpack-ondevice", maxiter=50
    )
    finite_derivatives = bool(
        np.all(np.isfinite(np.asarray(result.residual_jacobian)))
        and np.all(np.isfinite(np.asarray(result.jac)))
    )

    assert not (result.success and not finite_derivatives)
    assert not result.success
    assert result.status == 2
    assert result.message.startswith("non-finite")


@pytest.mark.parametrize("start", [np.inf, -np.inf, np.nan])
def test_qr_lane_fails_on_a_nonfinite_start(start):
    """A constant, finite residual has a zero Jacobian, which meets gtol at
    once; the solve must still fail on a non-finite x rather than return it
    as converged."""
    result = _opt.target_least_squares(
        lambda x: jnp.ones(2),
        jnp.full(1, start),
        method="lm-minpack-ondevice",
        maxiter=10,
    )

    assert not result.success
    assert result.status == 2
    assert result.message.startswith("non-finite")


def test_qr_lane_fails_when_the_returned_gradient_overflows():
    """r = [1e109 + 1e200 x, 1e154] from x = 0: r and J are finite.

    J^T r is 1e309 and J^T J 1e400, both inf in float64. The solve stops on
    gtol at once, but a result carrying an infinite gradient or Hessian must
    not report success.
    """
    result = _opt.target_least_squares(
        lambda x: jnp.stack((1.0e109 + 1.0e200 * x[0], 1.0e154 + 0.0 * x[0])),
        jnp.zeros(1),
        method="lm-minpack-ondevice",
        maxiter=50,
    )

    assert np.all(np.isfinite(np.asarray(result.residual)))
    assert np.all(np.isfinite(np.asarray(result.residual_jacobian)))
    assert not np.all(np.isfinite(np.asarray(result.jac)))
    assert not result.success
    assert result.status == 2
    assert result.message.startswith("non-finite")


def test_qr_lane_norms_do_not_overflow_on_large_finite_residuals():
    """r = 1e200 (x - 1) from x = 0: every norm is finite (1e200).

    A plain sum of squares overflows to inf, which zeroes the gradient cosine
    and stops on gtol at the start. MINPACK's enorm scales the sum, so the
    solve moves to x = 1. J^T J = 1e400 still overflows, so the result, which
    carries it, is a non-finite failure rather than a success.
    """
    result = _opt.target_least_squares(
        lambda x: 1.0e200 * (x - 1.0),
        jnp.zeros(1),
        method="lm-minpack-ondevice",
        maxiter=50,
    )

    assert result.nfev > 1, result.message
    np.testing.assert_allclose(np.asarray(result.x), [1.0], rtol=1e-12, atol=0)
    assert not result.success
    assert result.status == 2


def test_qr_lane_converges_from_a_residual_whose_square_overflows():
    """r = x - 1e200 from x = 1e199: ||r||^2 overflows, J = 1 does not.

    With an unscaled norm the start's gradient cosine is 0 and the solve
    stops there; with enorm it reaches x = 1e200, where every returned value
    is finite.
    """
    result = _opt.target_least_squares(
        lambda x: x - 1.0e200,
        jnp.full(1, 1.0e199),
        method="lm-minpack-ondevice",
        maxiter=50,
    )

    assert result.nfev > 1, result.message
    assert result.success, result.message
    np.testing.assert_allclose(np.asarray(result.x), [1.0e200], rtol=1e-12, atol=0)


def test_public_route_cost_stays_finite_when_the_residual_square_overflows():
    """r = [1.5e154], constant: r.r = 2.25e308 overflows, the cost does not.

    J = 0 meets gtol at once. The true cost 0.5 r.r = 1.125e308 is
    representable, so the cost is formed from MINPACK's enorm, which rounds
    within two ulps of it, and the typed result is finite and successful
    rather than a non-finite failure after convergence.
    """
    result = _solve_qr_lane(lambda x: jnp.full(1, 1.5e154) + 0.0 * x, jnp.zeros(1))

    assert result.success, result.message
    assert result.nonfinite_fields == ()
    assert abs(result.fun - 1.125e308) <= 2 * np.spacing(1.125e308)


def test_public_route_cost_is_the_plain_half_sum_of_squares_on_normal_residuals():
    """Away from overflow the cost is the plain 0.5 * vdot(r, r), bit for bit.

    SciPy reports 0.5 * np.dot(f, f); the device dot and NumPy's dot sum in
    different orders, so the two agree to rounding, not bitwise.
    """
    residual, x0, _ = _linear_fixture()
    result = _solve_qr_lane(residual, x0)
    final_residual = jnp.asarray(result.residual)

    assert result.success, result.message
    assert result.fun == float(0.5 * jnp.vdot(final_residual, final_residual).real)
    np.testing.assert_allclose(
        result.fun,
        0.5 * np.dot(result.residual, result.residual),
        rtol=1.0e-15,
        atol=0,
    )


def _enorm_reference(vector):
    """netlib MINPACK enorm, term by term (More, Garbow and Hillstrom 1980)."""
    rdwarf, rgiant = 3.834e-20, 1.304e19
    s1 = s2 = s3 = x1max = x3max = 0.0
    agiant = rgiant / len(vector)
    for value in vector:
        xabs = abs(float(value))
        if rdwarf < xabs < agiant:
            s2 += xabs**2
        elif xabs <= rdwarf:
            if xabs > x3max:
                s3 = 1.0 + s3 * (x3max / xabs) ** 2
                x3max = xabs
            elif xabs != 0.0:
                s3 += (xabs / x3max) ** 2
        elif xabs > x1max:
            s1 = 1.0 + s1 * (x1max / xabs) ** 2
            x1max = xabs
        else:
            s1 += (xabs / x1max) ** 2
    if s1 != 0.0:
        return x1max * np.sqrt(s1 + (s2 / x1max) / x1max)
    if s2 != 0.0:
        if s2 >= x3max:
            return np.sqrt(s2 * (1.0 + (x3max / s2) * (x3max * s3)))
        return np.sqrt(x3max * ((s2 / x3max) + (x3max * s3)))
    return x3max * np.sqrt(s3)


@pytest.mark.parametrize(
    "vector",
    [
        [3.0, 4.0],
        [1.0e200, -2.0e200, 3.0],
        [1.0e-200, 2.0e-200],
        [1.0e-25, 3.0e-30, 0.0],
        [2.0e19, 1.0, 1.0e-30],
        [0.0, 0.0],
    ],
)
def test_minpack_enorm_matches_the_reference_algorithm(vector):
    np.testing.assert_allclose(
        float(_opt._minpack_enorm(jnp.asarray(vector))),
        _enorm_reference(vector),
        rtol=1e-15,
        atol=0,
    )


def test_njev_counts_minpack_jacobians_not_the_terminal_one():
    """r = x - 1, max_nfev = 2: lmder factors one Jacobian, then stops on budget.

    The lane also evaluates J at the accepted end point for its result, as
    SciPy's wrapper does; that one is reported separately, not in njev.
    """
    lane, reference = _solve_both_minpack_lanes(lambda x: x - 1.0, (0.0,), max_nfev=2)

    assert reference.njev == 1
    assert lane.njev == reference.njev
    assert lane.jacobian_evaluations == 2


def test_typed_result_carries_both_jacobian_counts():
    """The typed driver result keeps njev (SciPy's meaning) and the true count.

    r = x - 1 from 0 with a budget of 2 evaluations: lmder factors one
    Jacobian (njev = 1); the lane also evaluates J at the accepted end point
    for the result (jacobian_evaluations = 2).
    """
    result = least_squares(
        lambda x: x - 1.0,
        jnp.zeros(1),
        driver=Driver.SIMSOPT_LM_QR,
        options=SimsoptLMQROptions(maxiter=2),
    )

    assert result.nfev == 2
    assert result.njev == 1
    assert result.jacobian_evaluations == 2


def _pivoted_qr_problem(seed):
    rng = np.random.default_rng(seed)
    jacobian = rng.standard_normal((12, 5)) @ np.diag([1.0, 3.0, 1e-2, 0.5, 10.0])
    residual = rng.standard_normal(12)
    q_matrix, r_matrix, pivots = scipy.linalg.qr(
        jacobian, pivoting=True, mode="economic"
    )
    return jacobian, residual, r_matrix, pivots, q_matrix.T @ residual


def test_lmpar_solves_the_secular_equation_inside_a_short_bound():
    """A bound shorter than the Gauss-Newton step yields par > 0 on ||x|| ~ delta."""
    jacobian, residual, r_matrix, pivots, qtb = _pivoted_qr_problem(_FIXTURE_SEED)
    gauss_newton = np.linalg.lstsq(jacobian, residual, rcond=None)[0]
    delta = 0.05 * np.linalg.norm(gauss_newton)

    x, par = _opt._minpack_lmpar(
        jnp.asarray(r_matrix),
        jnp.asarray(pivots),
        jnp.ones(5),
        jnp.asarray(qtb),
        jnp.asarray(delta),
        jnp.asarray(0.0),
    )
    x = np.asarray(x)
    par = float(par)

    assert par > 0.0
    assert abs(np.linalg.norm(x) - delta) <= 0.1 * delta
    np.testing.assert_allclose(
        (jacobian.T @ jacobian + par * np.eye(5)) @ x,
        jacobian.T @ residual,
        rtol=1e-10,
        atol=1e-12,
    )


def test_lmpar_returns_the_gauss_newton_step_inside_a_long_bound():
    """A bound past 1.1 ||Gauss-Newton step|| gives par = 0 and that step."""
    jacobian, residual, r_matrix, pivots, qtb = _pivoted_qr_problem(_FIXTURE_SEED + 5)
    gauss_newton = np.linalg.lstsq(jacobian, residual, rcond=None)[0]

    x, par = _opt._minpack_lmpar(
        jnp.asarray(r_matrix),
        jnp.asarray(pivots),
        jnp.ones(5),
        jnp.asarray(qtb),
        jnp.asarray(2.0 * np.linalg.norm(gauss_newton)),
        jnp.asarray(0.0),
    )

    assert float(par) == 0.0
    np.testing.assert_allclose(np.asarray(x), gauss_newton, rtol=1e-10, atol=1e-12)


def _spy_minpack_tolerances(monkeypatch):
    """Record the MINPACK tolerances ``target_least_squares`` hands the lane."""
    captured = {}
    solve = _opt.levenberg_marquardt_minpack_traceable

    def spy(residual_fn, x0, **kwargs):
        captured.update({key: kwargs[key] for key in ("ftol", "xtol", "gtol")})
        return solve(residual_fn, x0, **kwargs)

    monkeypatch.setattr(_opt, "levenberg_marquardt_minpack_traceable", spy)
    return captured


def test_a_single_tol_gates_ftol_xtol_and_gtol_like_upstream(monkeypatch):
    """Upstream's Boozer LS calls least_squares(ftol=tol, xtol=tol, gtol=tol)."""
    captured = _spy_minpack_tolerances(monkeypatch)
    residual, x0, _ = _linear_fixture()

    _opt.target_least_squares(
        residual, x0, method="lm-minpack-ondevice", tol=1e-11, maxiter=100
    )

    assert captured == {"ftol": 1e-11, "xtol": 1e-11, "gtol": 1e-11}


def test_explicit_minpack_tolerances_override_the_single_tol(monkeypatch):
    captured = _spy_minpack_tolerances(monkeypatch)
    residual, x0, _ = _linear_fixture()

    _opt.target_least_squares(
        residual,
        x0,
        method="lm-minpack-ondevice",
        tol=1e-9,
        maxiter=100,
        options={"ftol": 1e-6, "gtol": None},
    )

    assert captured == {"ftol": 1e-6, "xtol": 1e-9, "gtol": 1e-9}


def test_default_tol_does_not_accept_a_start_with_a_small_gradient():
    """r(x) = x - 5e-9 at x = 0: ||J^T r|| is 5e-9, the residual is not zero."""
    result = _opt.target_least_squares(
        lambda x: x - 5e-9,
        jnp.zeros(1),
        method="lm-minpack-ondevice",
        tol=1e-10,
        maxiter=100,
    )

    assert result.success, result.message
    assert result.nfev > 1
    np.testing.assert_allclose(np.asarray(result.x), [5e-9], rtol=1e-12, atol=0)
    assert float(result.fun) == 0.0


# --------------------------------------------------------------------------
# 4. Dense materialization is capped up front
# --------------------------------------------------------------------------


def test_qr_lane_refuses_dense_jacobian_over_the_declared_budget():
    """The cap must fail closed with a clear, quantified message."""
    residual, x0, _ = _linear_fixture()

    with pytest.raises(MemoryError, match="max_dense_linearization_bytes"):
        least_squares(
            residual,
            x0,
            driver=Driver.SIMSOPT_LM_QR,
            options=SimsoptLMQROptions(
                maxiter=400,
                max_dense_linearization_bytes=1,
            ),
        )


def test_qr_lane_cap_message_reports_the_required_and_allowed_bytes():
    residual, x0, _ = _linear_fixture()

    with pytest.raises(MemoryError) as excinfo:
        least_squares(
            residual,
            x0,
            driver=Driver.SIMSOPT_LM_QR,
            options=SimsoptLMQROptions(maxiter=1, max_dense_linearization_bytes=1),
        )

    message = str(excinfo.value)
    # 40x8 Jacobian + 8x8 Hessian in float64 == (320 + 64) * 8 == 3072 bytes.
    assert "3072 bytes" in message
    assert "float64" in message
    assert "max_dense_linearization_bytes=1" in message


def test_qr_lane_runs_when_the_dense_jacobian_fits_the_budget():
    residual, x0, optimum = _linear_fixture()

    result = least_squares(
        residual,
        x0,
        driver=Driver.SIMSOPT_LM_QR,
        options=SimsoptLMQROptions(
            maxiter=400,
            # Comfortably above the 3072 bytes this fixture needs.
            max_dense_linearization_bytes=1 << 20,
        ),
    )

    assert result.success
    np.testing.assert_allclose(np.asarray(result.x), optimum, rtol=0, atol=1e-7)


def test_qr_lane_refuses_materialize_false():
    """The matrix-free LM is gone; asking for it must fail, not materialize."""
    residual, x0, _ = _linear_fixture()

    with pytest.raises(
        ValueError, match="cannot honour materialize_dense_linearization=False"
    ):
        _opt.target_least_squares(
            residual,
            x0,
            method="lm-minpack-ondevice",
            maxiter=50,
            options={"materialize_dense_linearization": False},
        )


def test_qr_lane_budget_is_the_shared_dense_materialization_convention():
    """The cap reuses the repo-wide ``max_dense_linearization_bytes`` name."""
    fields = SimsoptLMQROptions().__dataclass_fields__

    assert "max_dense_linearization_bytes" in fields
    # Unset by default: callers declare the budget they are willing to spend.
    assert SimsoptLMQROptions().max_dense_linearization_bytes is None


# --------------------------------------------------------------------------
# 5. Warm solves reuse one compiled executable
# --------------------------------------------------------------------------
#
# The QR lane routes through the memoized ``_cached_traceable_runner`` seam: the
# residual callable owns the cache entry, and the
# runner's build-time constant set is the cache key. Before that wiring, every
# call rebuilt the closure and re-entered ``jax.jit`` on a fresh function
# object, so no call ever hit the JIT cache.
#
# Every QR solve calls the residual exactly once *outside* the compiled runner:
# the dense-materialization cap of section 4 is probed with ``jax.eval_shape``
# before anything is materialized. So a warm solve costs exactly that one
# residual trace, and a cold solve costs it plus the runner's own trace.
_PREFLIGHT_RESIDUAL_TRACES = 1


def _trace_counting(residual):
    """Wrap ``residual`` so each Python (i.e. tracing) call bumps ``.traces``.

    Under ``jax.jit`` the Python body runs only while tracing, so the counter
    measures retraces directly rather than wall-clock, which would be flaky.
    """

    def counted(x, *args):
        counted.traces += 1
        return residual(x, *args)

    counted.traces = 0
    return counted


def _solve_qr(residual, x0, *, maxiter=400, residual_args=()):
    return least_squares(
        residual,
        x0,
        driver=Driver.SIMSOPT_LM_QR,
        options=SimsoptLMQROptions(maxiter=maxiter),
        residual_args=residual_args,
    )


def test_warm_qr_solve_reuses_the_compiled_executable():
    """A second identical solve must not retrace the solver body."""
    residual, x0, optimum = _linear_fixture()
    counted = _trace_counting(residual)

    _solve_qr(counted, x0)
    cold_traces = counted.traces

    warm = _solve_qr(counted, x0)
    warm_traces = counted.traces - cold_traces

    assert cold_traces > _PREFLIGHT_RESIDUAL_TRACES, (
        "the first solve must trace the solver body; it traced the residual "
        f"only {cold_traces} time(s), i.e. no more than the cap preflight"
    )
    assert warm_traces == _PREFLIGHT_RESIDUAL_TRACES, (
        "the warm solve retraced the solver body "
        f"({warm_traces} residual traces, expected only the cap preflight); "
        "the LM_QR lane is not hitting the memoized runner cache"
    )
    np.testing.assert_allclose(np.asarray(warm.x), optimum, rtol=0, atol=1e-7)


def test_qr_runner_is_memoized_per_residual_callable_and_constant_set():
    """The seam returns one runner object, and JAX compiles it once."""
    residual, x0, _optimum = _linear_fixture()

    first = _opt._make_traceable_levenberg_marquardt_minpack_runner(
        residual, 400, 1e-8, 1e-8, 1e-8, False, False
    )
    second = _opt._make_traceable_levenberg_marquardt_minpack_runner(
        residual, 400, 1e-8, 1e-8, 1e-8, False, False
    )
    assert first is second

    # Second, independent evidence: JAX's own JIT cache for that runner holds a
    # single entry however often the runner is invoked.
    first(x0, ())
    assert first._cache_size() == 1
    first(x0, ())
    assert first._cache_size() == 1

    # A different build-time constant is a different executable, but both stay
    # reachable under the same residual callable.
    other_maxiter = _opt._make_traceable_levenberg_marquardt_minpack_runner(
        residual, 7, 1e-8, 1e-8, 1e-8, False, False
    )
    assert other_maxiter is not first
    assert (
        _opt._make_traceable_levenberg_marquardt_minpack_runner(
            residual, 7, 1e-8, 1e-8, 1e-8, False, False
        )
        is other_maxiter
    )

    # ...and a callback-instrumented build is a third, separate executable,
    # because its runner carries the debug-callback effects.
    assert (
        _opt._make_traceable_levenberg_marquardt_minpack_runner(
            residual, 400, 1e-8, 1e-8, 1e-8, True, False
        )
        is not first
    )


def test_qr_solves_with_different_problem_constants_keep_separate_executables():
    """Two residuals with different embedded data must not share a program."""
    linear_residual, linear_x0, linear_optimum = _linear_fixture()
    nonlinear_residual, nonlinear_x0, nonlinear_optimum = _nonlinear_fixture()
    counted_linear = _trace_counting(linear_residual)
    counted_nonlinear = _trace_counting(nonlinear_residual)

    linear = _solve_qr(counted_linear, linear_x0)
    nonlinear = _solve_qr(counted_nonlinear, nonlinear_x0)

    # The second problem traced its own body rather than reusing the first's.
    assert counted_nonlinear.traces > _PREFLIGHT_RESIDUAL_TRACES
    np.testing.assert_allclose(np.asarray(linear.x), linear_optimum, rtol=0, atol=1e-7)
    np.testing.assert_allclose(
        np.asarray(nonlinear.x), nonlinear_optimum, rtol=0, atol=1e-7
    )

    # Re-solving the first problem is still warm: the cache is keyed per
    # residual callable, not clobbered by the intervening solve.
    traces_before = counted_linear.traces
    again = _solve_qr(counted_linear, linear_x0)
    assert counted_linear.traces - traces_before == _PREFLIGHT_RESIDUAL_TRACES
    assert np.array_equal(np.asarray(again.x), np.asarray(linear.x))


def test_qr_lane_reuses_one_executable_across_residual_arg_values():
    """``residual_args`` are runtime arguments, never baked into the program.

    This is the staleness guard: if the args were captured at build time, the
    second solve would silently replay the first problem's answer.
    """
    rng = np.random.default_rng(_FIXTURE_SEED + 1)
    matrix = jnp.asarray(rng.standard_normal((30, 5)))
    first_rhs = jnp.asarray(rng.standard_normal(30))
    second_rhs = jnp.asarray(rng.standard_normal(30))

    def residual(x, rhs):
        return matrix @ x - rhs

    counted = _trace_counting(residual)
    x0 = jnp.zeros(5)

    first = _solve_qr(counted, x0, residual_args=(first_rhs,))
    traces_after_first = counted.traces

    second = _solve_qr(counted, x0, residual_args=(second_rhs,))
    assert counted.traces - traces_after_first == _PREFLIGHT_RESIDUAL_TRACES

    host_matrix = np.asarray(matrix)
    for solved, rhs in ((first, first_rhs), (second, second_rhs)):
        expected = np.linalg.lstsq(host_matrix, np.asarray(rhs), rcond=None)[0]
        np.testing.assert_allclose(np.asarray(solved.x), expected, rtol=0, atol=1e-7)
    assert not np.array_equal(np.asarray(first.x), np.asarray(second.x))


@pytest.mark.parametrize("strict_target_lane", ["0", "1"])
def test_qr_warm_reuse_is_independent_of_the_strict_target_lane_flag(
    monkeypatch,
    strict_target_lane,
):
    """The residual lane never passes through the strict purity wrapper.

    ``wrap_strict_target_lane_value_and_grad`` guards the scalar ``minimize``
    entrypoints only, so ``SIMSOPT_TARGET_LANE_STRICT`` must not interpose a
    per-solve wrapper object between the caller's residual and the runner
    cache. Pinned in both modes because a strict-mode-only wrapper is exactly
    what silently defeats callable-identity cache keys.
    """
    monkeypatch.setenv("SIMSOPT_TARGET_LANE_STRICT", strict_target_lane)
    residual, x0, optimum = _linear_fixture()
    counted = _trace_counting(residual)

    _solve_qr(counted, x0)
    traces_after_cold = counted.traces
    warm = _solve_qr(counted, x0)

    assert counted.traces - traces_after_cold == _PREFLIGHT_RESIDUAL_TRACES
    np.testing.assert_allclose(np.asarray(warm.x), optimum, rtol=0, atol=1e-7)


def test_qr_lane_caches_pytree_decision_vectors_and_returns_the_pytree():
    """A structured ``x0`` flattens inside the trace and still warm-reuses."""
    rng = np.random.default_rng(_FIXTURE_SEED + 2)
    matrix = jnp.asarray(rng.standard_normal((24, 6)))
    rhs = jnp.asarray(rng.standard_normal(24))

    def residual(tree):
        return matrix @ jnp.concatenate((tree["head"], tree["tail"])) - rhs

    counted = _trace_counting(residual)
    x0 = {"head": jnp.zeros(4), "tail": jnp.zeros(2)}

    first = _opt.target_least_squares(
        counted, x0, method="lm-minpack-ondevice", maxiter=400
    )
    traces_after_cold = counted.traces
    second = _opt.target_least_squares(
        counted, x0, method="lm-minpack-ondevice", maxiter=400
    )

    assert counted.traces - traces_after_cold == _PREFLIGHT_RESIDUAL_TRACES
    assert sorted(first.x.keys()) == ["head", "tail"]
    optimum = np.linalg.lstsq(np.asarray(matrix), np.asarray(rhs), rcond=None)[0]
    np.testing.assert_allclose(
        np.concatenate((first.x["head"], first.x["tail"])),
        optimum,
        rtol=0,
        atol=1e-7,
    )
    for key in ("head", "tail"):
        assert np.array_equal(np.asarray(first.x[key]), np.asarray(second.x[key]))


def test_one_runner_keeps_two_decision_structures_apart():
    """Structure is discriminated by JAX, which is why it is not a cache key.

    ``unravel`` is rebuilt inside the trace, so one memoized runner serves every
    decision-vector structure and ``jax.jit``'s own argument signature keeps the
    programs apart. If the in-trace ravel were ever hoisted back out, this test
    would see one program answering for both structures.
    """
    rng = np.random.default_rng(_FIXTURE_SEED + 3)
    matrix = jnp.asarray(rng.standard_normal((12, 3)))
    rhs = jnp.asarray(rng.standard_normal(12))

    def residual(x):
        flat = x if isinstance(x, jnp.ndarray) else jnp.concatenate((x["a"], x["b"]))
        return matrix @ flat - rhs

    flat_solved = _opt.target_least_squares(
        residual, jnp.zeros(3), method="lm-minpack-ondevice", maxiter=100
    )
    tree_solved = _opt.target_least_squares(
        residual,
        {"a": jnp.zeros(2), "b": jnp.zeros(1)},
        method="lm-minpack-ondevice",
        maxiter=100,
    )
    # ``target_least_squares``'s default tol=1e-10 sets all three tolerances.
    runner = _opt._make_traceable_levenberg_marquardt_minpack_runner(
        residual, 100, 1e-10, 1e-10, 1e-10, False, False
    )

    assert runner._cache_size() == 2, (
        "one compiled program is answering for both decision structures"
    )
    optimum = np.linalg.lstsq(np.asarray(matrix), np.asarray(rhs), rcond=None)[0]
    np.testing.assert_allclose(np.asarray(flat_solved.x), optimum, rtol=0, atol=1e-7)
    np.testing.assert_allclose(
        np.concatenate((tree_solved.x["a"], tree_solved.x["b"])),
        optimum,
        rtol=0,
        atol=1e-7,
    )


_INSTRUMENTED_SOLVE_COUNT = 12


def _compiled_executables_for(runner_cache, residual):
    """Total compiled programs the lane retains for one residual callable.

    Sums ``_cache_size()`` over every memoized runner filed under ``residual``,
    so it catches both a runner that forks its own JIT cache and a lane that
    hands out extra runner objects.
    """
    total = 0
    for callable_cell, runners_by_key in runner_cache.values():
        if callable_cell() is not residual:
            continue
        total += sum(runner._cache_size() for runner in runners_by_key.values())
    return total


@pytest.mark.parametrize(
    ("method", "runner_cache_name"),
    [
        ("lm-minpack-ondevice", "_TRACEABLE_LM_QR_RUNNER_CACHE"),
    ],
    ids=["lm_qr"],
)
def test_instrumented_solves_share_one_compiled_executable(method, runner_cache_name):
    """Callback tokens are traced operands, so they must not fork the cache.

    Tokens are minted per call. Declared ``static_argnums`` they compile a
    fresh executable per instrumented solve, and because the runner itself is
    memoized the lane then *retains* every one of them — an unbounded per-solve
    leak that a single-solve test cannot see.
    """
    residual, x0, _optimum = _nonlinear_fixture()
    runner_cache = getattr(_opt, runner_cache_name)

    steps_per_solve = []
    for _ in range(_INSTRUMENTED_SOLVE_COUNT):
        steps = []
        _opt.target_least_squares(
            residual,
            x0,
            method=method,
            maxiter=400,
            callback=steps.append,
            progress_callback=lambda nit, fun, grad_norm: None,
        )
        steps_per_solve.append(len(steps))

    retained = _compiled_executables_for(runner_cache, residual)
    assert retained == 1, (
        f"{_INSTRUMENTED_SOLVE_COUNT} instrumented {method} solves retain "
        f"{retained} compiled programs; per-call callback tokens are forking "
        "the JIT cache and the memoized runner is holding every entry"
    )
    assert steps_per_solve[0] > 0, "no accepted step reached the callback"
    assert len(set(steps_per_solve)) == 1, (
        f"instrumented solves delivered varying step counts {steps_per_solve}; "
        "the shared executable is not reproducing the callback stream"
    )


def test_qr_lane_step_callback_still_delivers_the_decision_pytree():
    """Routing steps through the traceable-callback registry keeps the shape.

    The compiled runner carries the flat iterate, so the registered adapter has
    to restore the caller's pytree before the user callback sees it.
    """
    rng = np.random.default_rng(_FIXTURE_SEED + 2)
    matrix = jnp.asarray(rng.standard_normal((24, 6)))
    rhs = jnp.asarray(rng.standard_normal(24))

    def residual(tree):
        return matrix @ jnp.concatenate((tree["head"], tree["tail"])) - rhs

    steps = []
    progress = []
    x0 = {"head": jnp.zeros(4), "tail": jnp.zeros(2)}

    result = _opt.target_least_squares(
        residual,
        x0,
        method="lm-minpack-ondevice",
        maxiter=400,
        callback=steps.append,
        progress_callback=lambda nit, fun, grad: progress.append(int(nit)),
    )

    assert steps, "no accepted step was delivered to the callback"
    assert len(progress) == len(steps)
    for step in steps:
        assert sorted(step.keys()) == ["head", "tail"]
        assert np.shape(step["head"]) == (4,)
        assert np.shape(step["tail"]) == (2,)
    for key in ("head", "tail"):
        np.testing.assert_allclose(
            np.asarray(steps[-1][key]), np.asarray(result.x[key]), rtol=0, atol=1e-12
        )
