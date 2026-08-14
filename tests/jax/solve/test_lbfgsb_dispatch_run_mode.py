"""The dispatch L-BFGS-B lane picks its own run mode; no caller knob decides it.

``simsopt_jax.solve.dispatch`` owns the choice between the fused on-device
driver and the callback-capable host driver: the fused route has no host
boundary left to deliver per-iteration events on, so it is correct exactly when
no observer is attached.  These tests pin the emitted mode on both sides of that
decision and prove the fused default lands on the same numbers, bit for bit, as
the stepwise route a callback caller still gets.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import simsopt_jax.geo.optimizers.optimizer as legacy_optimizer
from scipy.optimize import OptimizeResult
from simsopt_jax.geo.optimizers.private._common import (
    private_optimizer_runtime_is_supported,
)
from simsopt_jax.runtime.host_boundary import host_transfer_audit
from simsopt_jax.solve import Driver, SimsoptLBFGSBOptions
from simsopt_jax.solve.dispatch import _legacy_lbfgsb_options, minimize
from simsopt_jax.solve.driver import legacy_target_minimize_method

_ROSENBROCK_X0 = np.array([-1.2, 1.0, -0.5, 2.0, 0.3], dtype=np.float64)
_SOLVE_OPTIONS = SimsoptLBFGSBOptions(
    maxiter=200,
    maxfun=4000,
    gtol=1.0e-10,
    ftol=1.0e-14,
    maxcor=10,
)


def _fake_result():
    return OptimizeResult(
        x=np.zeros(2),
        fun=0.0,
        jac=np.zeros(2),
        nit=0,
        nfev=1,
        njev=1,
        status=0,
        success=True,
        message="ok",
    )


def _flat_value_and_grad(x):
    return 0.0, np.zeros_like(np.asarray(x, dtype=float))


def _rosenbrock_value_and_grad(x):
    def objective(current):
        return jnp.sum(
            100.0 * (current[1:] - current[:-1] ** 2) ** 2 + (1.0 - current[:-1]) ** 2
        )

    return jax.value_and_grad(objective)(jnp.asarray(x, dtype=jnp.float64))


def _bits(values) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(values, dtype=np.float64)).view(np.uint64)


def _captured_legacy_options(*, emit_one_accepted_step: bool = False):
    """Stand in for the legacy target lane and record the kwargs it was handed."""
    captured: dict[str, object] = {}

    def target_minimize(*_args, **kwargs):
        captured.update(kwargs)
        if emit_one_accepted_step:
            kwargs["callback"](np.zeros(2, dtype=np.float64))
            kwargs["progress_callback"](1, 0.0, 0.0)
        return _fake_result()

    return captured, target_minimize


# --------------------------------------------------------------------------
# the emitted legacy option payload
# --------------------------------------------------------------------------


def test_unobserved_lbfgsb_options_select_the_fused_device_route() -> None:
    payload = _legacy_lbfgsb_options(
        SimsoptLBFGSBOptions(),
        observes_accepted_steps=False,
    )
    assert payload["lbfgs_run_mode"] == "fused_stepwise"


def test_observed_lbfgsb_options_keep_the_callback_capable_route() -> None:
    payload = _legacy_lbfgsb_options(
        SimsoptLBFGSBOptions(),
        observes_accepted_steps=True,
    )
    assert payload["lbfgs_run_mode"] == "stepwise"


def test_run_mode_does_not_disturb_the_forwarded_tuning_options() -> None:
    options = SimsoptLBFGSBOptions(maxcor=7, maxfun=123, maxls=9, ftol=1.5e-13)
    payload = _legacy_lbfgsb_options(options, observes_accepted_steps=False)
    assert set(payload) == {"ftol", "maxcor", "maxfun", "maxls", "lbfgs_run_mode"}, (
        "the legacy L-BFGS-B payload gained or lost a key the target lane must "
        f"interpret: {sorted(payload)}"
    )
    assert payload["maxcor"] == 7
    assert payload["maxfun"] == 123
    assert payload["maxls"] == 9
    assert payload["ftol"] == 1.5e-13


# --------------------------------------------------------------------------
# what the dispatch entry point actually hands the legacy target lane
# --------------------------------------------------------------------------


def test_dispatch_lbfgsb_solve_requests_the_fused_run_mode(monkeypatch) -> None:
    captured, target_minimize = _captured_legacy_options()
    monkeypatch.setattr(legacy_optimizer, "target_minimize", target_minimize)

    minimize(_flat_value_and_grad, np.zeros(2), driver=Driver.SIMSOPT_LBFGSB)

    assert captured["method"] == "lbfgs-ondevice"
    assert captured["options"]["lbfgs_run_mode"] == "fused_stepwise"
    assert captured["callback"] is None
    assert captured["progress_callback"] is None


def test_dispatch_lbfgsb_solve_with_a_callback_requests_the_stepwise_run_mode(
    monkeypatch,
) -> None:
    events: list[object] = []
    captured, target_minimize = _captured_legacy_options(emit_one_accepted_step=True)
    monkeypatch.setattr(legacy_optimizer, "target_minimize", target_minimize)

    minimize(
        _flat_value_and_grad,
        np.zeros(2),
        driver=Driver.SIMSOPT_LBFGSB,
        callback=events.append,
    )

    assert captured["options"]["lbfgs_run_mode"] == "stepwise"
    assert captured["callback"] is not None
    assert len(events) == 1


# --------------------------------------------------------------------------
# the fused default must not move the answer
# --------------------------------------------------------------------------


@pytest.mark.private_optimizer_runtime
@pytest.mark.skipif(
    not private_optimizer_runtime_is_supported(jax.__version__),
    reason="Fused L-BFGS-B is validated on the pinned JAX runtime.",
)
def test_dispatch_lbfgsb_endpoint_is_bitwise_identical_to_the_stepwise_route() -> None:
    """The fused dispatch default reproduces the stepwise endpoint exactly.

    The reference leg is spelled out here rather than reused from the dispatch
    helper so the comparison stays an independent statement of the pre-change
    behavior: the same legacy target lane, the same tuning options, only
    ``lbfgs_run_mode`` forced back to the host-driven route.
    """
    stepwise = legacy_optimizer.target_minimize(
        _rosenbrock_value_and_grad,
        jnp.asarray(_ROSENBROCK_X0, dtype=jnp.float64),
        method=legacy_target_minimize_method(Driver.SIMSOPT_LBFGSB),
        tol=_SOLVE_OPTIONS.gtol,
        maxiter=_SOLVE_OPTIONS.maxiter,
        options={
            "ftol": _SOLVE_OPTIONS.ftol,
            "maxcor": _SOLVE_OPTIONS.maxcor,
            "maxfun": _SOLVE_OPTIONS.maxfun,
            "maxls": _SOLVE_OPTIONS.maxls,
            "lbfgs_run_mode": "stepwise",
        },
        value_and_grad=True,
        callback=None,
        progress_callback=None,
    )
    fused = minimize(
        _rosenbrock_value_and_grad,
        jnp.asarray(_ROSENBROCK_X0, dtype=jnp.float64),
        driver=Driver.SIMSOPT_LBFGSB,
        options=_SOLVE_OPTIONS,
    )

    assert fused.nit == int(stepwise.nit) > 0, (
        "fused dispatch solve took a different number of accepted iterations "
        f"({fused.nit}) than the stepwise route ({int(stepwise.nit)})"
    )
    assert (fused.nfev, fused.njev) == (int(stepwise.nfev), int(stepwise.njev))
    assert fused.success is bool(stepwise.success)
    assert fused.fun.hex() == float(stepwise.fun).hex(), (
        f"objective diverged: fused={fused.fun!r} stepwise={float(stepwise.fun)!r}"
    )
    assert np.array_equal(_bits(fused.x), _bits(stepwise.x)), (
        "solution diverged from the stepwise route by "
        f"max|dx|={np.max(np.abs(np.asarray(fused.x) - np.asarray(stepwise.x))):.3e}"
    )
    assert np.array_equal(_bits(fused.jac), _bits(stepwise.jac))


@pytest.mark.private_optimizer_runtime
@pytest.mark.skipif(
    not private_optimizer_runtime_is_supported(jax.__version__),
    reason="Fused L-BFGS-B is validated on the pinned JAX runtime.",
)
def test_dispatch_lbfgsb_solve_leaves_no_per_step_host_observations() -> None:
    """Fusion signature: the default dispatch solve never observes a step on host."""
    with host_transfer_audit() as audit:
        minimize(
            _rosenbrock_value_and_grad,
            jnp.asarray(_ROSENBROCK_X0, dtype=jnp.float64),
            driver=Driver.SIMSOPT_LBFGSB,
            options=_SOLVE_OPTIONS,
        )
    observed_calls = {entry.phase: entry.calls for entry in audit.summary()}
    # Positive control: the audit must have seen the solve at all, or renaming a
    # phase label would silently turn the fusion assertion into a tautology.
    assert observed_calls.get("final_result", 0) > 0, (
        "the transfer audit recorded no endpoint read-back, so it observed "
        f"nothing to disprove: {observed_calls}"
    )
    assert observed_calls.get("advance", 0) == 0, (
        "dispatch L-BFGS-B still round-trips accepted steps through the host; "
        "the fused run mode did not reach the legacy target lane"
    )
