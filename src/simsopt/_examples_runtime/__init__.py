"""Internal native import location for shared example execution helpers.

The implementation lives in ``simsopt_contracts.examples_runtime`` so JAX
callers can use it without importing the native package.
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
