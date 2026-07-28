"""Exact matched workflow for ``2_Intermediate/permanent_magnet_MUSE.py``."""

from __future__ import annotations

import hashlib
import io
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
from examples.jax.parity.arbiter import LaneObservation
from examples.jax.parity.cases._permanent_magnet_arbvec import (
    execute_arbvec_case,
    frozen_grid_arrays,
)
from examples.jax.parity.input_bundle import InputBundle, create_input_bundle
from examples.jax.parity.runtime import ParityLane
from simsopt_jax.examples import ExecutionScale

TEST_DATA = Path(__file__).resolve().parents[4] / "tests" / "test_files"
SURFACE_INPUT = TEST_DATA / "input.muse"
FAMUS_INPUT = TEST_DATA / "zot80.focus"

WORKFLOW_STAGES = (
    "construct_muse_boundary_and_tf_coil_field",
    "construct_downsampled_famus_grid_and_face_polarizations",
    "evaluate_initial_normal_field_residual",
    "run_arbitrary_vector_backtracking_gpmo",
    "evaluate_final_moments_and_normal_field_residual",
)


def _scale_configuration(scale: ExecutionScale) -> dict[str, object]:
    native_scale = scale == "native_default"
    return {
        "nphi": 16 if native_scale else 2,
        "ntheta": 16 if native_scale else 2,
        "downsample": 10 if native_scale else 100,
        "iterations": 10_000 if native_scale else 20,
        "backtracking": 200 if native_scale else 50,
        "max_magnets": 5_000 if native_scale else 20,
        "history_count": 20,
        "adjacent_count": 1,
        "threshold_angle": float(np.pi),
        "regularization_l2": 0.0,
        "radial_extent": 0.01,
        "coordinate_flag": "cartesian",
        "surface_input_sha256": hashlib.sha256(SURFACE_INPUT.read_bytes()).hexdigest(),
        "famus_input_sha256": hashlib.sha256(FAMUS_INPUT.read_bytes()).hexdigest(),
    }


def _build_cpu_grid(configuration: dict[str, object]):
    from simsopt.field import BiotSavart
    from simsopt.geo import PermanentMagnetGrid, SurfaceRZFourier
    from simsopt.util import FocusData, discretize_polarizations, polarization_axes
    from simsopt.util.permanent_magnet_helper_functions import (
        initialize_coils_for_pm_optimization,
    )

    nphi = configuration["nphi"]
    ntheta = configuration["ntheta"]
    downsample = configuration["downsample"]
    radial_extent = configuration["radial_extent"]
    assert isinstance(nphi, int)
    assert isinstance(ntheta, int)
    assert isinstance(downsample, int)
    assert isinstance(radial_extent, float)
    surface = SurfaceRZFourier.from_focus(
        SURFACE_INPUT,
        range="half period",
        nphi=nphi,
        ntheta=ntheta,
    )
    _, _, coils = initialize_coils_for_pm_optimization(
        "muse_famus",
        TEST_DATA,
        surface,
    )
    field = BiotSavart(coils)
    field.set_points(surface.gamma().reshape((-1, 3)))
    normal_field = np.sum(
        field.B().reshape((nphi, ntheta, 3)) * surface.unitnormal(),
        axis=2,
    )
    magnet_data = FocusData(FAMUS_INPUT, downsample=downsample)
    axes, polarization_types = polarization_axes(["face"])
    positive_count = len(polarization_types) // 2
    orientation = np.arctan2(magnet_data.oy, magnet_data.ox)
    discretize_polarizations(
        magnet_data,
        orientation,
        axes[:positive_count],
        polarization_types[:positive_count],
    )
    polarizations = np.stack(
        (magnet_data.pol_x, magnet_data.pol_y, magnet_data.pol_z),
        axis=-1,
    )
    with redirect_stdout(io.StringIO()):
        return PermanentMagnetGrid.geo_setup_from_famus(
            surface,
            normal_field,
            FAMUS_INPUT,
            pol_vectors=polarizations,
            downsample=downsample,
            dr=radial_extent,
        )


def create_input(root: Path, scale: ExecutionScale) -> InputBundle:
    """Freeze the MUSE fixed-state grid consumed by every solver lane."""
    configuration = _scale_configuration(scale)
    grid = _build_cpu_grid(configuration)
    return create_input_bundle(
        root,
        case_id="native-permanent-magnet-muse",
        random_seed=0,
        arrays=frozen_grid_arrays(grid),
        configuration={
            **configuration,
            "R0": float(grid.R0),
            "nfp": int(grid.plasma_boundary.nfp),
            "stellsym": bool(grid.plasma_boundary.stellsym),
            "ndipoles": int(grid.ndipoles),
        },
        scale=scale,
    )


def execute(
    lane: ParityLane,
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
) -> LaneObservation:
    """Execute the exact MUSE permanent-magnet workflow in one solver lane."""
    return execute_arbvec_case(lane, bundle, arrays, WORKFLOW_STAGES)
