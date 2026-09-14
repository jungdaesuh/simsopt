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
from .scalar_stage import (
    STAGE_OPTIMIZER_OBSERVABLES,
    TWO_STAGE_OPTIMIZER_OBSERVABLES,
    solve_scalar_stage,
    stage_optimizer_observables,
    two_stage_optimizer_observables,
)
from .stage_two_minimal import (
    MinimalStageTwoDeviceResult,
    MinimalStageTwoState,
    solve_minimal_stage_two,
)
from .stage_two_standard import (
    STANDARD_STAGE_TWO_OPTIMIZER_OBSERVABLES,
    StandardStageTwoDeviceResult,
    StandardStageTwoState,
    solve_standard_stage_two,
    standard_stage_two_optimizer_observables,
    standard_stage_two_state,
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
from .stochastic_stage_two import (
    STOCHASTIC_STAGE_TWO_OPTIMIZER_OBSERVABLES,
    StochasticStageTwoConfiguration,
    solve_stochastic_stage_two,
    stochastic_stage_two_configuration,
    stochastic_stage_two_optimizer_observables,
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
    "STAGE_OPTIMIZER_OBSERVABLES",
    "STANDARD_STAGE_TWO_OPTIMIZER_OBSERVABLES",
    "STOCHASTIC_STAGE_TWO_OPTIMIZER_OBSERVABLES",
    "TWO_STAGE_OPTIMIZER_OBSERVABLES",
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
    "StochasticStageTwoConfiguration",
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
    "solve_stochastic_stage_two",
    "solve_qfm_sequence",
    "solve_rz_curve_length",
    "solve_rz_surface_area_volume_sequence",
    "solve_scalar_stage",
    "solve_strain_rotation",
    "solve_weighted_quadratic",
    "solve_wireframe_rcls",
    "stage_optimizer_observables",
    "standard_stage_two_optimizer_observables",
    "standard_stage_two_state",
    "stochastic_stage_two_configuration",
    "stochastic_stage_two_optimizer_observables",
    "two_stage_optimizer_observables",
    "weighted_quadratic_gradient",
    "weighted_quadratic_residuals",
)
