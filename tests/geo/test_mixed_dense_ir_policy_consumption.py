"""Behavioral coverage for live mixed dense-IR accuracy-policy consumers."""

from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from simsopt_jax import numerical_policy
from simsopt_jax.geo.optimizers import dense_ir, linear_solve, optimizer


def _install_accuracy_policy(monkeypatch, **changes) -> None:
    policy = replace(
        numerical_policy.MIXED_DENSE_IR_ACCURACY_POLICY,
        **changes,
    )
    monkeypatch.setattr(
        numerical_policy,
        "MIXED_DENSE_IR_ACCURACY_POLICY",
        policy,
    )


def test_certificate_dtype_controls_live_dense_ir_requirement() -> None:
    assert dense_ir._require_policy_certificate_dtype(
        np.dtype(np.float64),
        detail="test",
    ) == np.dtype(np.float64)
    with pytest.raises(ValueError, match="requires policy certificate dtype float64"):
        dense_ir._require_policy_certificate_dtype(
            np.dtype(np.float32),
            detail="test",
        )


def test_tolerance_floor_controls_live_newton_gate(monkeypatch) -> None:
    _install_accuracy_policy(
        monkeypatch,
        linear_solve_tolerance_floor=4.0e-12,
    )

    actual = optimizer._eisenstat_walker_strict_cap(
        jnp.asarray(1.0e-11, dtype=jnp.float64),
        dtype=np.dtype(np.float64),
    )

    assert float(actual) == pytest.approx(4.0e-12)


def test_tolerance_cap_controls_live_newton_gate(monkeypatch) -> None:
    _install_accuracy_policy(
        monkeypatch,
        linear_solve_tolerance_cap=3.0e-12,
    )

    actual = optimizer._eisenstat_walker_strict_cap(
        jnp.asarray(1.0e-8, dtype=jnp.float64),
        dtype=np.dtype(np.float64),
    )

    assert float(actual) == pytest.approx(3.0e-12)


def test_forward_error_multiplier_controls_live_dense_gate(monkeypatch) -> None:
    _install_accuracy_policy(
        monkeypatch,
        forward_error_tolerance_multiplier=1.0e4,
    )

    actual = linear_solve._forward_error_tolerance(
        tol=jnp.asarray(1.0e-10, dtype=jnp.float64),
        dtype=np.dtype(np.float64),
    )

    assert float(actual) == pytest.approx(1.0e-6)
