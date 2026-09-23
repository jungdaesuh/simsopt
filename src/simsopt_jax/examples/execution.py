"""Shared execution contract for SIMSOPT JAX examples."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Final

from simsopt_contracts import examples_runtime as _execution

from simsopt_jax.backend.runtime import (
    get_backend_mode,
    get_resolved_precision,
    get_runtime_jax_device,
)
from simsopt_jax.solve.driver import Driver

EXECUTION_SCALES: Final = _execution.EXECUTION_SCALES
ExecutionScale = _execution.ExecutionScale


def example_runtime_metadata(scale: ExecutionScale) -> dict[str, str]:
    """Return the runtime identity every executable example must publish."""
    device = get_runtime_jax_device()
    return {
        "backend_mode": get_backend_mode(),
        "platform": "cpu" if device is None else device.platform,
        "precision": get_resolved_precision(),
        "scale": scale,
    }


@dataclass(frozen=True)
class ExampleResult(_execution.ExampleResult):
    """Immutable scientific result published by an executable example."""

    def json_object(self, scale: ExecutionScale) -> dict[str, object]:
        """Preserve the JAX example result API with JAX runtime metadata."""

        return _execution.serialize_example_result(
            self,
            scale,
            example_runtime_metadata(scale),
        )


ExampleSolve = Callable[[Path, int, ExecutionScale], ExampleResult]


def scalar_example_driver() -> Driver:
    """Select limited-memory fast solves and dense parity-compatible solves."""
    if get_backend_mode().endswith("_fast"):
        return Driver.SIMSOPT_LBFGSB
    return Driver.SIMSOPT_BFGS


def run_example(
    arguments: list[str] | None,
    *,
    description: str | None,
    temporary_prefix: str,
    bounded_steps: int,
    native_default_steps: int,
    solve: ExampleSolve,
    result_filename: str | None = None,
) -> int:
    """Parse the common CLI, execute one example, and publish its result."""

    return _execution.run_example(
        arguments,
        description=description,
        temporary_prefix=temporary_prefix,
        bounded_steps=bounded_steps,
        native_default_steps=native_default_steps,
        solve=solve,
        runtime_metadata=example_runtime_metadata,
        result_filename=result_filename,
    )
