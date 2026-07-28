"""Public execution types for SIMSOPT JAX examples."""

from .execution import (
    EXECUTION_SCALES,
    ExampleResult,
    ExampleSolve,
    ExecutionScale,
    example_runtime_metadata,
    run_example,
    scalar_example_driver,
)
from .stochastic_samples import (
    GaussianPerturbationSampler,
    StochasticPerturbationBundle,
    materialize_stochastic_coil_perturbations,
)
from .weighted_quadratic import (
    WeightedQuadraticDeviceResult,
    solve_weighted_quadratic,
    weighted_quadratic_gradient,
    weighted_quadratic_residuals,
)

__all__ = (
    "EXECUTION_SCALES",
    "ExampleResult",
    "ExampleSolve",
    "ExecutionScale",
    "GaussianPerturbationSampler",
    "StochasticPerturbationBundle",
    "WeightedQuadraticDeviceResult",
    "example_runtime_metadata",
    "materialize_stochastic_coil_perturbations",
    "run_example",
    "scalar_example_driver",
    "solve_weighted_quadratic",
    "weighted_quadratic_gradient",
    "weighted_quadratic_residuals",
)
