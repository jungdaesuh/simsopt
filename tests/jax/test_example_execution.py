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


@pytest.mark.parametrize(
    "mode",
    ("jax_cpu_fast", "jax_gpu_fast", "jax_cpu_parity", "jax_gpu_parity"),
)
def test_scalar_example_driver_preserves_declared_native_lbfgsb(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    monkeypatch.setattr(execution, "get_backend_mode", lambda: mode)

    assert (
        execution.scalar_example_driver(native_driver=Driver.SIMSOPT_LBFGSB)
        == Driver.SIMSOPT_LBFGSB
    )
