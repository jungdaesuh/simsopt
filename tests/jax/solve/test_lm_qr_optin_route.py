"""Contract tests for the explicit opt-in dense-QR Levenberg-Marquardt route.

Background
----------
The on-device LM family has two inner linear solves:

* ``Driver.SIMSOPT_LM_GMRES`` (``least_squares_algorithm="lm"``) is the
  **default** target lane. Its inner step is matrix-free GMRES against the
  regularized Gauss-Newton operator ``J^T J + lambda I``. It is the byte-equality
  oracle partner of the host-driven ``Driver.SIMSOPT_LM_GMRES_HOST``
  (``least_squares_algorithm="lm"``, ``optimizer_backend="scipy"``), and the
  SHA-bound receipts in this repo pin ``simsopt_lm_gmres`` by name.
* ``Driver.SIMSOPT_LM_QR`` (``least_squares_algorithm="lm-minpack"``) is an
  **opt-in only** lane. Its inner step factorizes the dense damped Jacobian with
  column-pivoted QR (DESC-style factorize-once), never GMRES.

These tests pin the three properties that make the QR lane safe to ship
alongside the default without disturbing it:

1. it is never reachable by default (routing pins),
2. it agrees with the GMRES lane on the *optimum* but is deliberately **not**
   its byte-equality oracle (different linear algebra => different roundoff and
   different iterate trajectories), and
3. its dense materialization is refused up front against a declared byte budget.
"""

import numpy as np
import pytest
import jax.numpy as jnp

from simsopt_jax.geo.optimizers.single_stage_routing import (
    resolve_boozer_least_squares_algorithm,
)
import simsopt_jax.geo.optimizers.optimizer as _opt
from simsopt_jax.solve.dispatch import least_squares
from simsopt_jax.solve.driver import legacy_target_least_squares_method
from simsopt_jax.solve import (
    Driver,
    SimsoptLMGMRESOptions,
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


def _solve_both(residual, x0, *, maxiter=400):
    qr = least_squares(
        residual,
        x0,
        driver=Driver.SIMSOPT_LM_QR,
        options=SimsoptLMQROptions(maxiter=maxiter),
    )
    gmres = least_squares(
        residual,
        x0,
        driver=Driver.SIMSOPT_LM_GMRES,
        options=SimsoptLMGMRESOptions(maxiter=maxiter),
    )
    return qr, gmres


# --------------------------------------------------------------------------
# 1. The QR lane is opt-in only: no default reaches it
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "boozer_optimizer_backend",
    ["ondevice", "host-jax", "scipy"],
)
def test_default_least_squares_algorithm_never_resolves_to_the_qr_lane(
    boozer_optimizer_backend,
):
    """With no explicit request, routing must never pick ``lm-minpack``."""
    resolved = resolve_boozer_least_squares_algorithm(boozer_optimizer_backend)

    assert resolved != "lm-minpack"
    assert resolved in {"lm", "quasi-newton"}


def test_qr_lane_requires_an_explicit_algorithm_string():
    """``lm-minpack`` is the only spelling that reaches ``SIMSOPT_LM_QR``."""
    assert (
        resolve_boozer_least_squares_algorithm(
            "ondevice",
            least_squares_algorithm="lm-minpack",
        )
        == "lm-minpack"
    )

    # The default LM spelling keeps routing to the GMRES lane that the
    # byte-equality oracle and the SHA-bound receipts pin.
    assert (
        _opt.resolve_boozer_inner_driver(
            "ondevice",
            limited_memory=False,
            least_squares_algorithm="lm",
        )
        is Driver.SIMSOPT_LM_GMRES
    )
    assert (
        _opt.resolve_boozer_inner_driver(
            "ondevice",
            limited_memory=False,
            least_squares_algorithm="lm-minpack",
        )
        is Driver.SIMSOPT_LM_QR
    )


def test_qr_lane_is_not_in_the_default_algorithm_position():
    """``lm-minpack`` is a valid *choice*, never a default value."""
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


def test_gmres_route_string_is_unchanged():
    """The default lane keeps the exact string the existing receipts pin."""
    residual, x0, _ = _linear_fixture()

    result = least_squares(
        residual,
        x0,
        driver=Driver.SIMSOPT_LM_GMRES,
        options=SimsoptLMGMRESOptions(maxiter=50),
    )

    assert result.driver is Driver.SIMSOPT_LM_GMRES
    assert result.driver.value == "simsopt_lm_gmres"
    assert legacy_target_least_squares_method(Driver.SIMSOPT_LM_GMRES) == "lm-ondevice"


# --------------------------------------------------------------------------
# 3. Correctness: same optimum as the GMRES lane, deliberately not byte-equal
# --------------------------------------------------------------------------


def test_qr_lane_reaches_the_linear_least_squares_optimum():
    residual, x0, optimum = _linear_fixture()
    qr, gmres = _solve_both(residual, x0)

    assert qr.success
    assert gmres.success

    # Both lanes must land on the closed-form ``lstsq`` optimum.
    np.testing.assert_allclose(np.asarray(qr.x), optimum, rtol=0, atol=1e-7)
    np.testing.assert_allclose(np.asarray(gmres.x), optimum, rtol=0, atol=1e-7)

    # ...and therefore on each other, at the same tolerance level.
    np.testing.assert_allclose(np.asarray(qr.x), np.asarray(gmres.x), rtol=0, atol=1e-7)


def test_qr_lane_reaches_the_nonlinear_least_squares_optimum():
    residual, x0, optimum = _nonlinear_fixture()
    qr, gmres = _solve_both(residual, x0)

    assert qr.success
    assert gmres.success

    np.testing.assert_allclose(np.asarray(qr.x), optimum, rtol=0, atol=1e-7)
    np.testing.assert_allclose(np.asarray(gmres.x), optimum, rtol=0, atol=1e-7)
    np.testing.assert_allclose(np.asarray(qr.x), np.asarray(gmres.x), rtol=0, atol=1e-7)

    # Both drive the residual to (numerical) zero on this fixture.
    assert float(qr.fun) < 1e-14
    assert float(gmres.fun) < 1e-14


def test_qr_lane_is_tolerance_equivalent_but_not_a_byte_oracle():
    """The QR lane is NOT the GMRES lane's byte-equality partner.

    This is by construction, not by defect. The GMRES lane solves
    ``(J^T J + lambda I) delta = -J^T r`` matrix-free; the QR lane factorizes the
    augmented matrix ``[J; sqrt(lambda) I]`` against ``[-r; 0]`` with
    column-pivoted QR. Different algebra produces different rounding and a
    different accepted-step trajectory, so the lanes agree on the *optimum* but
    not bit-for-bit. Byte equality on this repo's LM family remains the
    ``lm`` <-> ``lm-ondevice`` (host GMRES <-> on-device GMRES) pair, which this
    opt-in lane leaves untouched.
    """
    residual, x0, _ = _nonlinear_fixture()
    qr, gmres = _solve_both(residual, x0)

    qr_x = np.asarray(qr.x)
    gmres_x = np.asarray(gmres.x)

    # Same optimum...
    np.testing.assert_allclose(qr_x, gmres_x, rtol=0, atol=1e-7)

    # ...but genuinely different arithmetic: the lanes are not bit-identical.
    assert not np.array_equal(qr_x, gmres_x), (
        "QR and GMRES lanes are bit-identical; the opt-in lane is not actually "
        "running a different inner solve."
    )


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


def test_qr_lane_budget_is_the_shared_dense_materialization_convention():
    """The cap reuses the repo-wide ``max_dense_linearization_bytes`` name."""
    fields = SimsoptLMQROptions().__dataclass_fields__

    assert "max_dense_linearization_bytes" in fields
    # Unset by default: callers declare the budget they are willing to spend,
    # matching SimsoptLMGMRESOptions and OptimistixLMOptions.
    assert SimsoptLMQROptions().max_dense_linearization_bytes is None
    assert SimsoptLMGMRESOptions().max_dense_linearization_bytes is None
