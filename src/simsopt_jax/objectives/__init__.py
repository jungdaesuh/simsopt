"""Composable JAX objectives."""

from .stage_two import (
    CoilDofExtractionProvider,
    StageTwoObjectiveConfig,
    make_stage_two_objective,
    stage_two_geometric_penalty,
)

__all__ = (
    "CoilDofExtractionProvider",
    "StageTwoObjectiveConfig",
    "make_stage_two_objective",
    "stage_two_geometric_penalty",
)
