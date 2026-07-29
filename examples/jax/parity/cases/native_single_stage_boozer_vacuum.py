"""Matched VMEC-free implicit-Boozer single-stage workflow."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from examples.jax.parity.arbiter import LaneObservation
from examples.jax.parity.cases.native_boozerqa import (
    BoozerSingleStageSpec,
    create_variant_input,
    execute_variant,
)
from examples.jax.parity.input_bundle import InputBundle
from examples.jax.parity.runtime import ParityLane
from simsopt_jax.examples import ExecutionScale

WORKFLOW_STAGES = (
    "construct_ncsx_coils_and_volume_labelled_surface",
    "solve_initial_boozer_surface",
    "assemble_nonqs_residual_iota_radius_and_length_objective",
    "evaluate_initial_objective_and_gradient",
    "optimize_coils_and_currents_with_bfgs",
    "record_final_objective_gradient_and_implicit_physics_state",
)

SPEC = BoozerSingleStageSpec(
    case_id="native-single-stage-boozer-vacuum-optimization",
    workflow_stages=WORKFLOW_STAGES,
    bounded_resolution=1,
    native_resolution=6,
    inner_tolerance=1.0e-13,
    bounded_outer_maxiter=2,
    native_outer_maxiter=1_000,
    bounded_non_qs_sdim=4,
    native_non_qs_sdim=20,
    residual_weight=1.0,
    report_residual=True,
)


def create_input(root: Path, scale: ExecutionScale) -> InputBundle:
    """Freeze the exact native/JAX single-stage construction."""
    return create_variant_input(root, scale, SPEC)


def execute(
    lane: ParityLane,
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
) -> LaneObservation:
    """Execute the VMEC-free single-stage workflow in one isolated lane."""
    return execute_variant(lane, bundle, arrays, SPEC)
