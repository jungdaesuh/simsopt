"""Static registry of typed native/JAX parity cases."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from examples.jax.parity.arbiter import LaneObservation
from examples.jax.parity.cases.coil_flux import create_input as create_coil_flux_input
from examples.jax.parity.cases.coil_flux import execute as execute_coil_flux
from examples.jax.parity.cases.coil_force import create_input as create_coil_force_input
from examples.jax.parity.cases.coil_force import execute as execute_coil_force
from examples.jax.parity.cases.curve_length import create_input as create_curve_input
from examples.jax.parity.cases.curve_length import execute as execute_curve
from examples.jax.parity.cases.native_just_a_quadratic import (
    create_input as create_native_just_a_quadratic_input,
)
from examples.jax.parity.cases.native_just_a_quadratic import (
    execute as execute_native_just_a_quadratic,
)
from examples.jax.parity.cases.native_minimize_curve_length import (
    create_input as create_native_minimize_curve_length_input,
)
from examples.jax.parity.cases.native_minimize_curve_length import (
    execute as execute_native_minimize_curve_length,
)
from examples.jax.parity.cases.native_permanent_magnet_simple import (
    create_input as create_native_permanent_magnet_simple_input,
)
from examples.jax.parity.cases.native_permanent_magnet_simple import (
    execute as execute_native_permanent_magnet_simple,
)
from examples.jax.parity.cases.native_qfm import (
    create_input as create_native_qfm_input,
)
from examples.jax.parity.cases.native_qfm import execute as execute_native_qfm
from examples.jax.parity.cases.native_stage_two_optimization_minimal import (
    create_input as create_native_stage_two_optimization_minimal_input,
)
from examples.jax.parity.cases.native_stage_two_optimization_minimal import (
    execute as execute_native_stage_two_optimization_minimal,
)
from examples.jax.parity.cases.native_strain_optimization import (
    create_input as create_native_strain_optimization_input,
)
from examples.jax.parity.cases.native_strain_optimization import (
    execute as execute_native_strain_optimization,
)
from examples.jax.parity.cases.native_surf_vol_area import (
    create_input as create_native_surf_vol_area_input,
)
from examples.jax.parity.cases.native_surf_vol_area import (
    execute as execute_native_surf_vol_area,
)
from examples.jax.parity.cases.native_wireframe_rcls_basic import (
    create_input as create_native_wireframe_rcls_basic_input,
)
from examples.jax.parity.cases.native_wireframe_rcls_basic import (
    execute as execute_native_wireframe_rcls_basic,
)
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
from simsopt_jax.examples import ExecutionScale


@dataclass(frozen=True)
class CaseDefinition:
    case_id: str
    create_input: Callable[[Path, ExecutionScale], InputBundle]
    execute: Callable[[ParityLane, InputBundle, dict[str, np.ndarray]], LaneObservation]


_CASES = {
    "native-just-a-quadratic": CaseDefinition(
        case_id="native-just-a-quadratic",
        create_input=create_native_just_a_quadratic_input,
        execute=execute_native_just_a_quadratic,
    ),
    "native-minimize-curve-length": CaseDefinition(
        case_id="native-minimize-curve-length",
        create_input=create_native_minimize_curve_length_input,
        execute=execute_native_minimize_curve_length,
    ),
    "native-permanent-magnet-simple": CaseDefinition(
        case_id="native-permanent-magnet-simple",
        create_input=create_native_permanent_magnet_simple_input,
        execute=execute_native_permanent_magnet_simple,
    ),
    "native-qfm": CaseDefinition(
        case_id="native-qfm",
        create_input=create_native_qfm_input,
        execute=execute_native_qfm,
    ),
    "native-surf-vol-area": CaseDefinition(
        case_id="native-surf-vol-area",
        create_input=create_native_surf_vol_area_input,
        execute=execute_native_surf_vol_area,
    ),
    "native-stage-two-optimization-minimal": CaseDefinition(
        case_id="native-stage-two-optimization-minimal",
        create_input=create_native_stage_two_optimization_minimal_input,
        execute=execute_native_stage_two_optimization_minimal,
    ),
    "native-strain-optimization": CaseDefinition(
        case_id="native-strain-optimization",
        create_input=create_native_strain_optimization_input,
        execute=execute_native_strain_optimization,
    ),
    "native-wireframe-rcls-basic": CaseDefinition(
        case_id="native-wireframe-rcls-basic",
        create_input=create_native_wireframe_rcls_basic_input,
        execute=execute_native_wireframe_rcls_basic,
    ),
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
