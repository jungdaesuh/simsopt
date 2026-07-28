"""Exact matched workflow for ``2_Intermediate/permanent_magnet_PM4Stell.py``."""

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
PLASMA_INPUT = TEST_DATA / "c09r00_B_axis_half_tesla_PM4Stell.plasma"
COIL_INPUT = TEST_DATA / "tf_only_half_tesla_symmetry_baxis_PM4Stell.focus"
FAMUS_INPUT = TEST_DATA / "magpie_trial104b_PM4Stell.focus"
CORNER_INPUT = TEST_DATA / "magpie_trial104b_corners_PM4Stell.csv"

WORKFLOW_STAGES = (
    "construct_ncsx_boundary_plasma_field_and_tf_coil_field",
    "construct_downsampled_pm4stell_grid_and_face_triplet_polarizations",
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
        "iterations": 2_000 if native_scale else 20,
        "backtracking": 200 if native_scale else 20,
        "max_magnets": 1_000 if native_scale else 20,
        "history_count": 10,
        "adjacent_count": 10,
        "threshold_angle": float(np.pi),
        "regularization_l2": 0.0,
        "coordinate_flag": "cartesian",
        "magnetization_maximum": 5.0 / (4.0 * np.pi * 1.0e-7),
        "plasma_input_sha256": hashlib.sha256(PLASMA_INPUT.read_bytes()).hexdigest(),
        "coil_input_sha256": hashlib.sha256(COIL_INPUT.read_bytes()).hexdigest(),
        "famus_input_sha256": hashlib.sha256(FAMUS_INPUT.read_bytes()).hexdigest(),
        "corner_input_sha256": hashlib.sha256(CORNER_INPUT.read_bytes()).hexdigest(),
    }


def _positive_polarization_family(
    name: str,
    family: int,
) -> tuple[np.ndarray, np.ndarray]:
    from simsopt.util import polarization_axes

    axes, types = polarization_axes([name])
    positive_count = len(types) // 2
    return axes[:positive_count], types[:positive_count] + family


def _build_cpu_grid(configuration: dict[str, object]):
    from simsopt.field import BiotSavart, Coil
    from simsopt.geo import PermanentMagnetGrid, SurfaceRZFourier
    from simsopt.util import (
        FocusData,
        FocusPlasmaBnormal,
        discretize_polarizations,
        orientation_phi,
        read_focus_coils,
    )

    nphi = configuration["nphi"]
    ntheta = configuration["ntheta"]
    downsample = configuration["downsample"]
    maximum = configuration["magnetization_maximum"]
    assert isinstance(nphi, int)
    assert isinstance(ntheta, int)
    assert isinstance(downsample, int)
    assert isinstance(maximum, float)
    surface = SurfaceRZFourier.from_focus(
        PLASMA_INPUT,
        range="half period",
        nphi=nphi,
        ntheta=ntheta,
    )
    plasma_normal = FocusPlasmaBnormal(PLASMA_INPUT).bnormal_grid(
        nphi,
        ntheta,
        "half period",
    )
    base_curves, base_currents, coil_count = read_focus_coils(COIL_INPUT)
    coils = [
        Coil(base_curves[index], base_currents[index]) for index in range(coil_count)
    ]
    field = BiotSavart(coils)
    field.set_points(surface.gamma().reshape((-1, 3)))
    coil_normal = np.sum(
        field.B().reshape((nphi, ntheta, 3)) * surface.unitnormal(),
        axis=2,
    )
    magnet_data = FocusData(FAMUS_INPUT, downsample=downsample)
    families = (
        _positive_polarization_family("face", 0),
        _positive_polarization_family("fe_ftri", 1),
        _positive_polarization_family("fc_ftri", 2),
    )
    axes = np.concatenate(tuple(family[0] for family in families), axis=0)
    types = np.concatenate(tuple(family[1] for family in families), axis=0)
    orientations = orientation_phi(CORNER_INPUT)[: magnet_data.nMagnets]
    discretize_polarizations(magnet_data, orientations, axes, types)
    polarizations = np.stack(
        (magnet_data.pol_x, magnet_data.pol_y, magnet_data.pol_z),
        axis=-1,
    )
    with redirect_stdout(io.StringIO()):
        grid = PermanentMagnetGrid.geo_setup_from_famus(
            surface,
            plasma_normal + coil_normal,
            FAMUS_INPUT,
            pol_vectors=polarizations,
            m_maxima=maximum,
            downsample=downsample,
        )
    return grid


def create_input(root: Path, scale: ExecutionScale) -> InputBundle:
    """Freeze the PM4Stell fixed-state grid consumed by every solver lane."""
    configuration = _scale_configuration(scale)
    grid = _build_cpu_grid(configuration)
    return create_input_bundle(
        root,
        case_id="native-permanent-magnet-pm4stell",
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
    """Execute the exact PM4Stell permanent-magnet workflow in one lane."""
    return execute_arbvec_case(lane, bundle, arrays, WORKFLOW_STAGES)
