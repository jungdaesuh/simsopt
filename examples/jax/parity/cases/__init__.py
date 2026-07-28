"""Static registry of typed native/JAX parity cases."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from simsopt_jax.examples import ExecutionScale
from examples.jax.parity.arbiter import LaneObservation
from examples.jax.parity.cases.coil_flux import create_input as create_coil_flux_input
from examples.jax.parity.cases.coil_flux import execute as execute_coil_flux
from examples.jax.parity.cases.coil_force import create_input as create_coil_force_input
from examples.jax.parity.cases.coil_force import execute as execute_coil_force
from examples.jax.parity.cases.curve_length import create_input as create_curve_input
from examples.jax.parity.cases.curve_length import execute as execute_curve
from examples.jax.parity.cases.permanent_magnet import (
    create_input as create_permanent_magnet_input,
)
from examples.jax.parity.cases.permanent_magnet import (
    execute as execute_permanent_magnet,
)
from examples.jax.parity.cases.qfm_surface import create_input as create_qfm_input
from examples.jax.parity.cases.qfm_surface import execute as execute_qfm
from examples.jax.parity.cases.surface_geometry import (
    create_input as create_surface_input,
)
from examples.jax.parity.cases.surface_geometry import execute as execute_surface
from examples.jax.parity.cases.traceable_least_squares import (
    create_input as create_traceable_least_squares_input,
)
from examples.jax.parity.cases.traceable_least_squares import (
    execute as execute_traceable_least_squares,
)
from examples.jax.parity.cases.wireframe import create_input as create_wireframe_input
from examples.jax.parity.cases.wireframe import execute as execute_wireframe
from examples.jax.parity.input_bundle import InputBundle
from examples.jax.parity.runtime import ParityLane


@dataclass(frozen=True)
class CaseDefinition:
    case_id: str
    create_input: Callable[[Path, ExecutionScale], InputBundle]
    execute: Callable[[ParityLane, InputBundle, dict[str, np.ndarray]], LaneObservation]


_CASES = {
    "traceable-least-squares": CaseDefinition(
        case_id="traceable-least-squares",
        create_input=create_traceable_least_squares_input,
        execute=execute_traceable_least_squares,
    ),
    "curve-length-optimization": CaseDefinition(
        case_id="curve-length-optimization",
        create_input=create_curve_input,
        execute=execute_curve,
    ),
    "surface-geometry-optimization": CaseDefinition(
        case_id="surface-geometry-optimization",
        create_input=create_surface_input,
        execute=execute_surface,
    ),
    "coil-flux-optimization": CaseDefinition(
        case_id="coil-flux-optimization",
        create_input=create_coil_flux_input,
        execute=execute_coil_flux,
    ),
    "permanent-magnet-optimization": CaseDefinition(
        case_id="permanent-magnet-optimization",
        create_input=create_permanent_magnet_input,
        execute=execute_permanent_magnet,
    ),
    "wireframe-optimization": CaseDefinition(
        case_id="wireframe-optimization",
        create_input=create_wireframe_input,
        execute=execute_wireframe,
    ),
    "coil-force-and-finite-build": CaseDefinition(
        case_id="coil-force-and-finite-build",
        create_input=create_coil_force_input,
        execute=execute_coil_force,
    ),
    "qfm-surface-optimization": CaseDefinition(
        case_id="qfm-surface-optimization",
        create_input=create_qfm_input,
        execute=execute_qfm,
    ),
}


def get_case(case_id: str) -> CaseDefinition:
    """Return one statically registered parity case."""
    try:
        return _CASES[case_id]
    except KeyError as error:
        raise ValueError(f"unknown or unimplemented parity case: {case_id}") from error


def implemented_case_ids() -> tuple[str, ...]:
    """Return implemented case IDs in deterministic registry order."""
    return tuple(_CASES)
