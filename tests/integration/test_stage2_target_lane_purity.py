"""Stage 2 target-lane purity guard tests."""

from __future__ import annotations

import types

import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip(
    "simsoptpp",
    reason="Stage 2 target-lane purity tests require simsoptpp.",
)

from simsopt_jax.backend.runtime import (
    raise_if_target_lane_bypass,
    strict_target_lane_purity,
)
from simsopt_jax.geo.optimizers import optimizer as optimizer_jax
from simsopt_jax_adapters.objectives.stage2_target import (
    stage2_target_optimizer_state_to_dofs,
)
STRICT_TARGET_LANE_ENV = "SIMSOPT_TARGET_LANE_STRICT"

from examples.single_stage_optimization.STAGE_2 import (  # noqa: E402
    banana_coil_solver as stage2_solver_module,
)


def _load_stage2_script_module():
    return stage2_solver_module


def test_target_lane_purity_guard_is_env_and_stack_scoped(monkeypatch):
    monkeypatch.setenv(STRICT_TARGET_LANE_ENV, "1")

    raise_if_target_lane_bypass("snapshot-metric")

    with pytest.raises(RuntimeError, match="target-lane bypass: snapshot-metric"):
        with strict_target_lane_purity():
            raise_if_target_lane_bypass("snapshot-metric")


def test_target_minimize_wraps_explicit_value_and_grad_in_strict_context(
    monkeypatch,
):
    monkeypatch.setenv(STRICT_TARGET_LANE_ENV, "1")
    monkeypatch.setattr(
        optimizer_jax, "require_target_backend_x64", lambda _backend: None
    )

    def fake_minimize(fun, x0, **_kwargs):
        return fun(x0)

    monkeypatch.setattr(
        optimizer_jax,
        "_minimize_lbfgs_private_value_and_grad",
        fake_minimize,
    )

    def value_and_grad(x):
        raise_if_target_lane_bypass("synthetic-value-and-grad")
        return jnp.sum(jnp.square(x)), 2.0 * x

    with pytest.raises(
        RuntimeError,
        match="target-lane bypass: synthetic-value-and-grad",
    ):
        optimizer_jax.target_minimize(
            value_and_grad,
            jnp.asarray([1.0, -2.0], dtype=jnp.float64),
            method="lbfgs-ondevice",
            value_and_grad=True,
            maxiter=1,
        )


def test_stage2_optimizer_wraps_target_value_and_grad_before_dispatch(monkeypatch):
    monkeypatch.setenv(STRICT_TARGET_LANE_ENV, "1")
    stage2_script = _load_stage2_script_module()
    import simsopt_jax.geo.optimizers.optimizer as optimizer_jax_module

    def fake_target_minimize(
        fun,
        x0,
        *,
        method,
        tol,
        maxiter,
        options,
        value_and_grad,
        callback=None,
        progress_callback=None,
        failure_callback=None,
    ):
        del method, tol, maxiter, options, value_and_grad
        del callback, progress_callback, failure_callback
        fun(x0)
        return types.SimpleNamespace(
            x=x0,
            nit=0,
            success=True,
            message="ok",
        )

    monkeypatch.setattr(
        optimizer_jax_module,
        "target_minimize",
        fake_target_minimize,
    )

    contract = stage2_script.resolve_stage2_optimizer_contract("jax", "ondevice")
    dofs = np.asarray([0.25, -0.5], dtype=np.float64)
    optimizer_state = stage2_script.build_stage2_target_optimizer_state(
        types.SimpleNamespace(expected_dof_count=2),
        dofs,
    )

    def target_value_and_grad(x):
        raise_if_target_lane_bypass("stage2-target-value-and-grad")
        flat_x = jnp.asarray(
            stage2_target_optimizer_state_to_dofs(x),
            dtype=jnp.float64,
        )
        return jnp.sum(jnp.square(flat_x)), 2.0 * flat_x

    with pytest.raises(
        RuntimeError,
        match="target-lane bypass: stage2-target-value-and-grad",
    ):
        stage2_script.run_stage2_optimizer(
            value_and_grad_fun=target_value_and_grad,
            dofs=optimizer_state,
            contract=contract,
            maxiter=1,
            ftol=0.0,
            gtol=1e-12,
            scalar_fun=lambda x: jnp.sum(
                jnp.square(stage2_target_optimizer_state_to_dofs(x))
            ),
        )
