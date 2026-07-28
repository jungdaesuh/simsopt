"""Exact matched workflow for ``2_Intermediate/wireframe_gsco_sector_saddle.py``."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from examples.jax.parity.arbiter import LaneObservation
from examples.jax.parity.cases._wireframe_gsco import (
    create_gsco_input,
    execute_gsco,
)
from examples.jax.parity.input_bundle import InputBundle
from examples.jax.parity.runtime import ParityLane
from simsopt.geo import SurfaceRZFourier, ToroidalWireframe
from simsopt_jax.examples import ExecutionScale

TEST_DATA = Path(__file__).resolve().parents[4] / "tests" / "test_files"
SURFACE_INPUT = TEST_DATA / "input.LandremanPaul2021_QA"
WIREFRAME_INPUT = TEST_DATA / "nescin.LandremanPaul2021_QA"


def _configuration(scale: ExecutionScale) -> dict[str, object]:
    native = scale == "native_default"
    return {
        "plasma_resolution": 32 if native else 4,
        "wireframe_nphi": 48 if native else 18,
        "wireframe_ntheta": 50 if native else 8,
        "max_iterations": 2_000 if native else 40,
        "field_on_axis": 1.0,
        "number_of_tf_coils": 3,
        "break_width": 2,
        "gsco_current_fraction": 0.05,
        "lambda_s": 10.0**-6.5,
        "no_crossing": True,
        "match_current": False,
        "no_new_coils": False,
        "max_loop_count": 0,
        "surface_input_sha256": hashlib.sha256(SURFACE_INPUT.read_bytes()).hexdigest(),
        "wireframe_input_sha256": hashlib.sha256(
            WIREFRAME_INPUT.read_bytes()
        ).hexdigest(),
    }


def _build_geometry(
    configuration: Mapping[str, object],
) -> tuple[ToroidalWireframe, SurfaceRZFourier, float]:
    resolution = configuration["plasma_resolution"]
    nphi = configuration["wireframe_nphi"]
    ntheta = configuration["wireframe_ntheta"]
    number_of_tf_coils = configuration["number_of_tf_coils"]
    break_width = configuration["break_width"]
    assert isinstance(resolution, int)
    assert isinstance(nphi, int)
    assert isinstance(ntheta, int)
    assert isinstance(number_of_tf_coils, int)
    assert isinstance(break_width, int)
    plasma = SurfaceRZFourier.from_vmec_input(
        SURFACE_INPUT,
        nphi=resolution,
        ntheta=resolution,
        range="half period",
    )
    wireframe_surface = SurfaceRZFourier.from_nescoil_input(
        WIREFRAME_INPUT,
        "current",
    )
    wireframe = ToroidalWireframe(wireframe_surface, nphi, ntheta)
    mu0 = 4.0 * np.pi * 1.0e-7
    poloidal_current = (
        -2.0 * np.pi * plasma.get_rc(0, 0) * float(configuration["field_on_axis"]) / mu0
    )
    tf_current = poloidal_current / (2 * wireframe.nfp * number_of_tf_coils)
    wireframe.add_tfcoil_currents(number_of_tf_coils, tf_current)
    wireframe.set_toroidal_breaks(
        number_of_tf_coils,
        break_width,
        allow_pol_current=True,
    )
    wireframe.set_poloidal_current(poloidal_current)
    return wireframe, plasma, poloidal_current


def create_input(root: Path, scale: ExecutionScale) -> InputBundle:
    """Create the exact sector-saddle GSCO input bundle."""
    configuration = _configuration(scale)
    _, _, poloidal_current = _build_geometry(configuration)
    current = abs(float(configuration["gsco_current_fraction"]) * poloidal_current)
    configuration["default_current"] = current
    configuration["max_current"] = 1.1 * current
    return create_gsco_input(
        root,
        scale,
        case_id="native-wireframe-gsco-sector-saddle",
        configuration=configuration,
        build_geometry=_build_geometry,
    )


def execute(
    lane: ParityLane,
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
) -> LaneObservation:
    """Execute the sector-saddle GSCO source workflow."""
    return execute_gsco(lane, bundle, arrays, _build_geometry)
