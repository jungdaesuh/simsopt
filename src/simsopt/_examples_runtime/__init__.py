"""Dependency-neutral execution helpers for the SIMSOPT example scripts.

This package is internal: the example scripts and the JAX example contract are
its only callers, and it is deliberately not part of the public SIMSOPT API.
"""

from .execution import (
    EXECUTION_SCALES,
    ExampleResult,
    ExampleRuntimeMetadata,
    ExampleSolve,
    ExecutionScale,
    run_example,
    serialize_example_result,
)

__all__ = (
    "EXECUTION_SCALES",
    "ExampleResult",
    "ExampleRuntimeMetadata",
    "ExampleSolve",
    "ExecutionScale",
    "run_example",
    "serialize_example_result",
)
