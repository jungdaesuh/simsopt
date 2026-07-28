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
from .qfm_sequence import (
    QfmDeviceState,
    QfmSequenceDeviceResult,
    QfmStageDeviceResult,
    solve_qfm_sequence,
)
from .rz_curve_length import RZCurveLengthDeviceResult, solve_rz_curve_length
from .rz_surface_area_volume import (
    SurfaceAreaVolumeSequenceDeviceResult,
    SurfaceAreaVolumeStageDeviceResult,
    solve_rz_surface_area_volume_sequence,
)
from .stage_two_minimal import (
    MinimalStageTwoDeviceResult,
    MinimalStageTwoState,
    solve_minimal_stage_two,
)
from .stage_two_standard import (
    StandardStageTwoDeviceResult,
    StandardStageTwoState,
    solve_standard_stage_two,
)
from .strain_optimization import (
    StrainOptimizationDeviceResult,
    StrainState,
    solve_strain_rotation,
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
from .wireframe_rcls import (
    WireframeRCLSDeviceResult,
    WireframeRCLSState,
    solve_wireframe_rcls,
)

__all__ = (
    "EXECUTION_SCALES",
    "ExampleResult",
    "ExampleSolve",
    "ExecutionScale",
    "GaussianPerturbationSampler",
    "MinimalStageTwoDeviceResult",
    "MinimalStageTwoState",
    "QfmDeviceState",
    "QfmSequenceDeviceResult",
    "QfmStageDeviceResult",
    "RZCurveLengthDeviceResult",
    "StochasticPerturbationBundle",
    "StandardStageTwoDeviceResult",
    "StandardStageTwoState",
    "StrainOptimizationDeviceResult",
    "StrainState",
    "SurfaceAreaVolumeSequenceDeviceResult",
    "SurfaceAreaVolumeStageDeviceResult",
    "WeightedQuadraticDeviceResult",
    "WireframeRCLSDeviceResult",
    "WireframeRCLSState",
    "example_runtime_metadata",
    "materialize_stochastic_coil_perturbations",
    "run_example",
    "scalar_example_driver",
    "solve_minimal_stage_two",
    "solve_standard_stage_two",
    "solve_qfm_sequence",
    "solve_rz_curve_length",
    "solve_rz_surface_area_volume_sequence",
    "solve_strain_rotation",
    "solve_weighted_quadratic",
    "solve_wireframe_rcls",
    "weighted_quadratic_gradient",
    "weighted_quadratic_residuals",
)
