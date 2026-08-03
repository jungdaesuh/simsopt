"""RED contracts for the fused on-device L-BFGS run mode.

``fused_stepwise`` moves the stepwise reverse-communication driver on device:
one bounded ``lax.while_loop`` whose body dispatches over the *same* macro-step
transitions the host driver alternates between.  These tests pin the routing,
the fail-closed callback contract, and accepted-state parity against the
host-driven ``stepwise`` route.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from simsopt_jax.geo.optimizers.private import _lbfgs, prepare_lbfgs_private
from simsopt_jax.geo.optimizers.private._common import (
    private_optimizer_runtime_is_supported,
)
from simsopt_jax.geo.optimizers.private._lbfgs import (
    _LBFGS_RUN_MODE_FUSED_STEPWISE,
    _LBFGS_RUN_MODE_MONOLITHIC_DEBUG,
    _LBFGS_RUN_MODE_STEPWISE,
    _check_lbfgsb_run_mode,
    _minimize_lbfgs_private,
)

pytestmark = [
    pytest.mark.private_optimizer_runtime,
    pytest.mark.skipif(
        not private_optimizer_runtime_is_supported(jax.__version__),
        reason="Fused L-BFGS-B is validated on the pinned JAX runtime.",
    ),
]

# Endpoint tolerance: the driver is moved on device, not re-derived, so the
# only admissible drift is reassociation inside XLA's fused loop.  Measured
# drift on coil47/GPU was max|dx| ~1.6e-14 with |df| == 0; keep a principled
# margin above that rather than demanding exact bits.
_ENDPOINT_ATOL = 1.0e-11
_OBJECTIVE_RTOL = 1.0e-12


def _rosenbrock(x: jax.Array) -> jax.Array:
    return 100.0 * (x[1] - x[0] ** 2) ** 2 + (1.0 - x[0]) ** 2


def _quadratic(x: jax.Array) -> jax.Array:
    weights = jnp.arange(1, x.size + 1, dtype=x.dtype)
    return jnp.sum(weights * (x - 1.5) ** 2)


def _solve(objective, x0, *, run_mode, **kwargs):
    return _minimize_lbfgs_private(
        objective,
        jnp.asarray(x0, dtype=jnp.float64),
        run_mode=run_mode,
        **kwargs,
    )


def _endpoint(result):
    return (
        np.asarray(result.x_k, dtype=np.float64),
        float(np.asarray(result.f_k)),
        int(np.asarray(result.status)),
        int(np.asarray(result.k)),
        int(np.asarray(result.nfev)),
    )


# --------------------------------------------------------------------------
# routing / validation
# --------------------------------------------------------------------------


def test_fused_stepwise_is_a_valid_run_mode() -> None:
    assert _check_lbfgsb_run_mode("fused_stepwise") == _LBFGS_RUN_MODE_FUSED_STEPWISE


def test_unknown_run_mode_is_still_rejected() -> None:
    with pytest.raises(ValueError, match="lbfgs_run_mode must be"):
        _check_lbfgsb_run_mode("on_device_please")


def test_unknown_run_mode_error_names_every_supported_mode() -> None:
    with pytest.raises(ValueError) as excinfo:
        _check_lbfgsb_run_mode("junk")
    message = str(excinfo.value)
    for mode in (
        _LBFGS_RUN_MODE_STEPWISE,
        _LBFGS_RUN_MODE_FUSED_STEPWISE,
        _LBFGS_RUN_MODE_MONOLITHIC_DEBUG,
    ):
        assert mode in message


def test_prepare_routes_fused_stepwise_to_the_fused_program() -> None:
    x0 = jnp.asarray([-1.2, 1.0], dtype=jnp.float64)
    fused = prepare_lbfgs_private(
        _rosenbrock, x0, run_mode=_LBFGS_RUN_MODE_FUSED_STEPWISE
    )
    assert fused.run_mode == _LBFGS_RUN_MODE_FUSED_STEPWISE
    assert fused.fused_solve is not None


def test_prepare_default_and_stepwise_keep_the_host_driven_path() -> None:
    x0 = jnp.asarray([-1.2, 1.0], dtype=jnp.float64)
    default = prepare_lbfgs_private(_rosenbrock, x0)
    stepwise = prepare_lbfgs_private(_rosenbrock, x0, run_mode=_LBFGS_RUN_MODE_STEPWISE)
    for prepared in (default, stepwise):
        assert prepared.run_mode == _LBFGS_RUN_MODE_STEPWISE
        assert prepared.fused_solve is None
        assert prepared.advance_from_start is not None
        assert prepared.advance_from_search is not None
        assert prepared.reenter_new_x is not None


def test_prepare_rejects_monolithic_debug_for_prepared_programs() -> None:
    x0 = jnp.asarray([-1.2, 1.0], dtype=jnp.float64)
    with pytest.raises(ValueError, match="monolithic_debug"):
        prepare_lbfgs_private(
            _rosenbrock, x0, run_mode=_LBFGS_RUN_MODE_MONOLITHIC_DEBUG
        )


def _cacheable_owner():
    class Owner:
        pass

    owner = Owner()
    # The private solver cache only engages for owners that opted in.
    object.__setattr__(owner, "_simsopt_cache_jit_value_and_grad", True)
    return owner


def _stepwise_kernel(owner, prefix):
    return _lbfgs._lbfgsb_advance_to_next_observable_kernel(
        lambda x, consts: (x[0], x),
        cache_owner=owner,
        cache_key_prefix=prefix,
        accepted_step_callback=None,
        entry_kind=_lbfgs._LBFGS_STEP_ENTRY_SEARCH,
        unconstrained_fast_path=True,
        dynamic_limits=True,
    )


def _fused_kernel(owner, prefix):
    return _lbfgs._lbfgsb_fused_stepwise_kernel(
        lambda x, consts: (x[0], x),
        cache_owner=owner,
        cache_key_prefix=prefix,
        entry_kind=_lbfgs._LBFGS_STEP_ENTRY_START,
        unconstrained_fast_path=True,
    )


def test_fused_kernel_build_does_not_disturb_stepwise_cache_identity() -> None:
    owner = _cacheable_owner()
    prefix = ("fused-mode-test",)
    first = _stepwise_kernel(owner, prefix)
    _fused_kernel(owner, prefix)
    assert _stepwise_kernel(owner, prefix) is first


def test_fused_kernel_is_cached_per_owner() -> None:
    owner = _cacheable_owner()
    prefix = ("fused-mode-test",)
    first = _fused_kernel(owner, prefix)
    assert _fused_kernel(owner, prefix) is first


def test_fused_and_stepwise_kernels_do_not_share_a_cache_slot() -> None:
    owner = _cacheable_owner()
    prefix = ("fused-mode-test",)
    assert _fused_kernel(owner, prefix) is not _stepwise_kernel(owner, prefix)


def test_prepared_fused_and_stepwise_programs_are_distinct_routes() -> None:
    owner = _cacheable_owner()
    x0 = jnp.asarray([-1.2, 1.0], dtype=jnp.float64)
    stepwise = prepare_lbfgs_private(_rosenbrock, x0, cache_owner=owner)
    fused = prepare_lbfgs_private(
        _rosenbrock, x0, cache_owner=owner, run_mode=_LBFGS_RUN_MODE_FUSED_STEPWISE
    )
    assert stepwise.fused_solve is None
    assert fused.fused_solve is not None
    assert fused.advance_from_start is None


# --------------------------------------------------------------------------
# fail-closed callback contract
# --------------------------------------------------------------------------


def test_fused_mode_rejects_accepted_step_callback() -> None:
    with pytest.raises(ValueError, match="stepwise"):
        _solve(
            _rosenbrock,
            [-1.2, 1.0],
            run_mode=_LBFGS_RUN_MODE_FUSED_STEPWISE,
            callback=lambda x: None,
        )


def test_fused_mode_rejects_progress_callback() -> None:
    with pytest.raises(ValueError, match="stepwise"):
        _solve(
            _rosenbrock,
            [-1.2, 1.0],
            run_mode=_LBFGS_RUN_MODE_FUSED_STEPWISE,
            progress_callback=lambda k, f, g: None,
        )


def test_fused_mode_rejects_optimizer_state_trace() -> None:
    with pytest.raises(ValueError, match="stepwise"):
        _solve(
            _rosenbrock,
            [-1.2, 1.0],
            run_mode=_LBFGS_RUN_MODE_FUSED_STEPWISE,
            record_optimizer_state_trace=True,
        )


def test_stepwise_still_accepts_callbacks() -> None:
    seen: list[np.ndarray] = []
    result = _solve(
        _rosenbrock,
        [-1.2, 1.0],
        run_mode=_LBFGS_RUN_MODE_STEPWISE,
        maxiter=200,
        callback=lambda x: seen.append(np.asarray(x)),
    )
    assert seen
    assert int(np.asarray(result.status)) == 0


# --------------------------------------------------------------------------
# endpoint / counter parity against the stepwise route
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("objective", "x0"),
    [
        (_rosenbrock, [-1.2, 1.0]),
        (_quadratic, [3.0, -2.0, 0.5, 4.25, -1.75]),
    ],
)
def test_fused_matches_stepwise_endpoint(objective, x0) -> None:
    stepwise = _solve(objective, x0, run_mode=_LBFGS_RUN_MODE_STEPWISE, maxiter=60)
    fused = _solve(objective, x0, run_mode=_LBFGS_RUN_MODE_FUSED_STEPWISE, maxiter=60)
    sx, sf, sstatus, sk, snfev = _endpoint(stepwise)
    fx, ff, fstatus, fk, fnfev = _endpoint(fused)
    assert fstatus == sstatus
    assert fk == sk
    assert fnfev == snfev
    assert ff == pytest.approx(sf, rel=_OBJECTIVE_RTOL, abs=0.0)
    np.testing.assert_allclose(fx, sx, rtol=0.0, atol=_ENDPOINT_ATOL)


def test_fused_matches_stepwise_gradient_and_convergence_flags() -> None:
    stepwise = _solve(
        _rosenbrock, [-1.2, 1.0], run_mode=_LBFGS_RUN_MODE_STEPWISE, maxiter=60
    )
    fused = _solve(
        _rosenbrock, [-1.2, 1.0], run_mode=_LBFGS_RUN_MODE_FUSED_STEPWISE, maxiter=60
    )
    np.testing.assert_allclose(
        np.asarray(fused.g_k), np.asarray(stepwise.g_k), rtol=0.0, atol=_ENDPOINT_ATOL
    )
    assert bool(np.asarray(fused.converged)) is bool(np.asarray(stepwise.converged))
    assert bool(np.asarray(fused.failed)) is bool(np.asarray(stepwise.failed))
    assert int(np.asarray(fused.ngev)) == int(np.asarray(stepwise.ngev))


def test_fused_matches_stepwise_inverse_hessian_history() -> None:
    stepwise = _solve(
        _rosenbrock, [-1.2, 1.0], run_mode=_LBFGS_RUN_MODE_STEPWISE, maxiter=60
    )
    fused = _solve(
        _rosenbrock, [-1.2, 1.0], run_mode=_LBFGS_RUN_MODE_FUSED_STEPWISE, maxiter=60
    )
    assert int(np.asarray(fused.hess_inv_n_corrs)) == int(
        np.asarray(stepwise.hess_inv_n_corrs)
    )
    np.testing.assert_allclose(
        np.asarray(fused.hess_inv_s),
        np.asarray(stepwise.hess_inv_s),
        rtol=0.0,
        atol=_ENDPOINT_ATOL,
    )
    np.testing.assert_allclose(
        np.asarray(fused.hess_inv_y),
        np.asarray(stepwise.hess_inv_y),
        rtol=0.0,
        atol=_ENDPOINT_ATOL,
    )


@pytest.mark.parametrize("maxiter", [0, 1, 3])
def test_fused_matches_stepwise_under_small_iteration_budgets(maxiter) -> None:
    stepwise = _solve(
        _rosenbrock, [-1.2, 1.0], run_mode=_LBFGS_RUN_MODE_STEPWISE, maxiter=maxiter
    )
    fused = _solve(
        _rosenbrock,
        [-1.2, 1.0],
        run_mode=_LBFGS_RUN_MODE_FUSED_STEPWISE,
        maxiter=maxiter,
    )
    sx, _sf, sstatus, sk, snfev = _endpoint(stepwise)
    fx, _ff, fstatus, fk, fnfev = _endpoint(fused)
    assert (fstatus, fk, fnfev) == (sstatus, sk, snfev)
    np.testing.assert_allclose(fx, sx, rtol=0.0, atol=_ENDPOINT_ATOL)


@pytest.mark.parametrize("maxfun", [1, 4])
def test_fused_matches_stepwise_under_evaluation_budgets(maxfun) -> None:
    stepwise = _solve(
        _rosenbrock,
        [-1.2, 1.0],
        run_mode=_LBFGS_RUN_MODE_STEPWISE,
        maxiter=60,
        maxfun=maxfun,
    )
    fused = _solve(
        _rosenbrock,
        [-1.2, 1.0],
        run_mode=_LBFGS_RUN_MODE_FUSED_STEPWISE,
        maxiter=60,
        maxfun=maxfun,
    )
    sx, _sf, sstatus, sk, snfev = _endpoint(stepwise)
    fx, _ff, fstatus, fk, fnfev = _endpoint(fused)
    assert (fstatus, fk, fnfev) == (sstatus, sk, snfev)
    np.testing.assert_allclose(fx, sx, rtol=0.0, atol=_ENDPOINT_ATOL)


def test_prepared_fused_run_matches_prepared_stepwise_run() -> None:
    x0 = jnp.asarray([-1.2, 1.0], dtype=jnp.float64)
    stepwise = prepare_lbfgs_private(_rosenbrock, x0, gtol=1e-10)
    fused = prepare_lbfgs_private(
        _rosenbrock, x0, gtol=1e-10, run_mode=_LBFGS_RUN_MODE_FUSED_STEPWISE
    )
    step_result = stepwise.run(x0, maxiter=60)
    fused_result = fused.run(x0, maxiter=60)
    sx, sf, sstatus, sk, snfev = _endpoint(step_result)
    fx, ff, fstatus, fk, fnfev = _endpoint(fused_result)
    assert (fstatus, fk, fnfev) == (sstatus, sk, snfev)
    assert ff == pytest.approx(sf, rel=_OBJECTIVE_RTOL, abs=0.0)
    np.testing.assert_allclose(fx, sx, rtol=0.0, atol=_ENDPOINT_ATOL)


# --------------------------------------------------------------------------
# traceability: the whole point of moving the driver on device
# --------------------------------------------------------------------------


def test_fused_mode_runs_inside_jit_while_stepwise_refuses() -> None:
    def solve_under_trace(x0):
        return _minimize_lbfgs_private(
            _rosenbrock,
            x0,
            maxiter=60,
            run_mode=_LBFGS_RUN_MODE_FUSED_STEPWISE,
        ).x_k

    x0 = jnp.asarray([-1.2, 1.0], dtype=jnp.float64)
    traced = jax.jit(solve_under_trace)(x0)
    eager = _solve(_rosenbrock, x0, run_mode=_LBFGS_RUN_MODE_FUSED_STEPWISE, maxiter=60)
    np.testing.assert_allclose(
        np.asarray(traced), np.asarray(eager.x_k), rtol=0.0, atol=_ENDPOINT_ATOL
    )

    with pytest.raises(ValueError, match="cannot run while tracing"):
        jax.jit(
            lambda x: _minimize_lbfgs_private(
                _rosenbrock, x, run_mode=_LBFGS_RUN_MODE_STEPWISE
            ).x_k
        )(x0)


def test_fused_mode_emits_no_host_observation_transfers() -> None:
    from simsopt_jax.runtime.host_boundary import host_transfer_audit

    x0 = jnp.asarray([-1.2, 1.0], dtype=jnp.float64)
    with host_transfer_audit() as audit:
        _solve(
            _rosenbrock, x0, run_mode=_LBFGS_RUN_MODE_FUSED_STEPWISE, maxiter=60
        )
    advance_calls = sum(
        entry.calls for entry in audit.summary() if entry.phase == "advance"
    )
    assert advance_calls == 0

    with host_transfer_audit() as stepwise_audit:
        _solve(_rosenbrock, x0, run_mode=_LBFGS_RUN_MODE_STEPWISE, maxiter=60)
    stepwise_calls = sum(
        entry.calls for entry in stepwise_audit.summary() if entry.phase == "advance"
    )
    assert stepwise_calls > 0


def test_fused_kernel_uses_the_unconstrained_fast_path() -> None:
    assert _lbfgs._lbfgsb_unconstrained_fast_path_enabled(_lbfgs._LBFGSB_PRIVATE_BOUNDS)
