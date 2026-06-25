"""Target-lane purity guard tests."""

from __future__ import annotations

import jax.numpy as jnp
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
STRICT_TARGET_LANE_ENV = "SIMSOPT_TARGET_LANE_STRICT"


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
