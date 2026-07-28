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
from .rz_curve_length import RZCurveLengthDeviceResult, solve_rz_curve_length
from .rz_surface_area_volume import (
    SurfaceAreaVolumeSequenceDeviceResult,
    SurfaceAreaVolumeStageDeviceResult,
    solve_rz_surface_area_volume_sequence,
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
    "RZCurveLengthDeviceResult",
    "StochasticPerturbationBundle",
    "SurfaceAreaVolumeSequenceDeviceResult",
    "SurfaceAreaVolumeStageDeviceResult",
    "WeightedQuadraticDeviceResult",
    "example_runtime_metadata",
    "materialize_stochastic_coil_perturbations",
    "run_example",
    "scalar_example_driver",
    "solve_rz_curve_length",
    "solve_rz_surface_area_volume_sequence",
    "solve_weighted_quadratic",
    "weighted_quadratic_gradient",
    "weighted_quadratic_residuals",
)
