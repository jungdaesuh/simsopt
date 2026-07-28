"""Shared JAX example execution-policy tests."""

from __future__ import annotations

import pytest

import simsopt_jax.examples.execution as execution
from simsopt_jax.solve.driver import Driver


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("jax_cpu_fast", Driver.SIMSOPT_LBFGSB),
        ("jax_gpu_fast", Driver.SIMSOPT_LBFGSB),
        ("jax_cpu_parity", Driver.SIMSOPT_BFGS),
        ("jax_gpu_parity", Driver.SIMSOPT_BFGS),
    ],
)
def test_scalar_example_driver_separates_fast_and_parity_modes(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    expected: Driver,
) -> None:
    monkeypatch.setattr(execution, "get_backend_mode", lambda: mode)

    assert execution.scalar_example_driver() == expected
