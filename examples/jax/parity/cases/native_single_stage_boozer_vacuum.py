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
from examples.jax.parity.contracts import QualityBand
from examples.jax.parity.input_bundle import InputBundle
from examples.jax.parity.measurement import MeasurementExecution
from examples.jax.parity.runtime import ParityLane
from simsopt.single_stage_boozer_vacuum import NATIVE_ITERATIONS
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
    native_outer_maxiter=NATIVE_ITERATIONS,
    bounded_non_qs_sdim=4,
    native_non_qs_sdim=20,
    residual_weight=1.0,
    report_residual=True,
    enforce_endpoint_certificate=True,
)

# Rule 3 of the 2026-08-15 native_default certification-gate ruling: this
# continuous optimizer forks by rejection sentinel and line search, so at
# native_default it is certifiable only as an endpoint quality band, never as
# final-value equivalence. Bounded scale is untouched -- no band applies there.
#
# Derivation (2026-08-14 three-lane native_default run, durable archive
# ~/simsopt-campaigns/ndparity-boozer-vacuum-20260814/, copied from
# .artifacts/jax-example-parity/20260814T010929Z-10f0606d.partial/): every lane
# ended budget_exhausted at the matched 1000-iteration budget, reducing
# final:objective from 8.4442e-05 to 4.3972e-08 (native-cpu), 4.5074e-08
# (jax-cpu), and 4.5614e-08 (jax-gpu) -- lane forks of 2.5e-2 and 3.7e-2
# relative, far outside the mirror_single_stage_final_value equality bucket but
# indistinguishable in delivered endpoint quality. The band is the next decade
# above the worst measured endpoint: 4.5614e-08 -> 1.0e-07, a 2.19x margin over
# the worst lane and a 2.27x margin over the native reference. A lane that
# lands one decade worse than measured (>= 4.6e-07) therefore fails closed,
# while ordinary run-to-run fork of the measured size passes.
NATIVE_DEFAULT_QUALITY_BAND = QualityBand(
    observable="final:objective",
    max_value=1.0e-07,
    derivation=(
        "2026-08-14 native_default three-lane run "
        "(ndparity-boozer-vacuum-20260814): all lanes budget_exhausted at 1000 "
        "iterations with final:objective 4.3972e-08 (native-cpu), 4.5074e-08 "
        "(jax-cpu), 4.5614e-08 (jax-gpu) from 8.4442e-05; band set one decade "
        "above the worst measured endpoint"
    ),
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


def execute_measurement(
    lane: ParityLane,
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
    measurement: MeasurementExecution,
) -> LaneObservation:
    """Execute one instrumented single-stage measurement lane."""
    return execute_variant(lane, bundle, arrays, SPEC, measurement=measurement)
