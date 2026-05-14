"""VMEC true single-stage objective for HBT (paper arXiv:2302.10622v2).

Public surface for the HBT VMEC true single-stage driver. Caller imports:

.. code-block:: python

    from examples.single_stage_optimization.VMEC_SINGLE_STAGE import (
        VmecSingleStageConfig,
        build_objective,
        check_seed_zero_gradient_trap,
        iota_at_promotion_surface,
        assemble_dof_schema,
        IotaSurfaceTargetMiss,
        SeedZeroGradientTrap,
        VmecCallFailure,
        StepClampBarrierTriggered,
        taylor_J2_alone_centered_eq28,
        taylor_J_combined_forward_eq29_xcoils,
        taylor_J_combined_centered_eq29_xsurface,
        assert_dJ1_dxcoils_zero,
        GRADIENT_NORM_FLOOR_BOUNDARY,
        GRADIENT_NORM_FLOOR_COILS,
    )

See ``program_cw_poloidal_legacy_vmec_true_single_stage_plan.md`` for the
full design contract and ``examples/3_Advanced/single_stage_optimization.py``
for the SIMSOPT single-stage reference example this module adapts.
"""

from .vmec_single_stage_banana import (
    VmecSingleStageObjective,
    assemble_dof_schema,
    build_objective,
    check_seed_zero_gradient_trap,
    iota_at_promotion_surface,
)
from .vmec_single_stage_config import (
    GRADIENT_NORM_FLOOR_BOUNDARY,
    GRADIENT_NORM_FLOOR_COILS,
    VmecSingleStageConfig,
)
from .vmec_single_stage_exceptions import (
    IotaSurfaceTargetMiss,
    SeedZeroGradientTrap,
    StepClampBarrierTriggered,
    VmecCallFailure,
)
from .vmec_single_stage_taylor_tests import (
    DEFAULT_TAYLOR_SCHEDULE,
    DEFAULT_TAYLOR_RNG_SEED,
    FIRST_ORDER_SLOPE_BAND,
    SECOND_ORDER_SLOPE_BAND,
    assert_dJ1_dxcoils_zero,
    taylor_J2_alone_centered_eq28,
    taylor_J_combined_centered_eq29_xsurface,
    taylor_J_combined_forward_eq29_xcoils,
)

__all__ = [
    "GRADIENT_NORM_FLOOR_BOUNDARY",
    "GRADIENT_NORM_FLOOR_COILS",
    "DEFAULT_TAYLOR_SCHEDULE",
    "DEFAULT_TAYLOR_RNG_SEED",
    "FIRST_ORDER_SLOPE_BAND",
    "SECOND_ORDER_SLOPE_BAND",
    "IotaSurfaceTargetMiss",
    "SeedZeroGradientTrap",
    "StepClampBarrierTriggered",
    "VmecCallFailure",
    "VmecSingleStageConfig",
    "VmecSingleStageObjective",
    "assemble_dof_schema",
    "assert_dJ1_dxcoils_zero",
    "build_objective",
    "check_seed_zero_gradient_trap",
    "iota_at_promotion_surface",
    "taylor_J2_alone_centered_eq28",
    "taylor_J_combined_centered_eq29_xsurface",
    "taylor_J_combined_forward_eq29_xcoils",
]
