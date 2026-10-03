"""Strict-transfer coverage for the exact permanent-magnet workflow."""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import io
from contextlib import redirect_stdout
from pathlib import Path

import jax
import numpy as np
from simsopt.field import ToroidalField
from simsopt.geo import PermanentMagnetGrid, SurfaceRZFourier
from simsopt_jax.geo.permanent_magnet_grid import PermanentMagnetGridJAX
from simsopt_jax.solve.permanent_magnet import GPMO_baseline_jax

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TEST_DATA = REPOSITORY_ROOT / "tests" / "test_files"
#: ``1_Simple/permanent_magnet_simple.py`` at upstream's ``in_github_actions``
#: resolution: a 2x2 half-period boundary and every hundredth FAMUS dipole.
BOUNDED_RESOLUTION = 2
BOUNDED_DOWNSAMPLE = 100
NET_POLOIDAL_CURRENT_AMPERES = 3.7713e6
VACUUM_PERMEABILITY = 4.0 * np.pi * 1.0e-7


def _bounded_simple_grid() -> PermanentMagnetGrid:
    """The cylindrical FAMUS grid of the simple example at its CI resolution."""
    surface = SurfaceRZFourier.from_wout(
        str(TEST_DATA / "wout_c09r00_fixedBoundary_0.5T_vacuum_ns201.nc"),
        range="half period",
        nphi=BOUNDED_RESOLUTION,
        ntheta=BOUNDED_RESOLUTION,
    )
    field = ToroidalField(
        R0=1.0,
        B0=VACUUM_PERMEABILITY * NET_POLOIDAL_CURRENT_AMPERES / (2.0 * np.pi),
    )
    field.set_points(surface.gamma().reshape((-1, 3)))
    normal_field = np.sum(
        field.B().reshape((BOUNDED_RESOLUTION, BOUNDED_RESOLUTION, 3))
        * surface.unitnormal(),
        axis=2,
    )
    with redirect_stdout(io.StringIO()):
        return PermanentMagnetGrid.geo_setup_from_famus(
            surface,
            normal_field,
            TEST_DATA / "init_orient_pm_nonorm_5E4_q4_dp.focus",
            coordinate_flag="cylindrical",
            downsample=BOUNDED_DOWNSAMPLE,
        )


def test_permanent_magnet_example_has_one_batched_numerical_host_boundary() -> None:
    source = (
        REPOSITORY_ROOT / "examples/jax/1_Simple/permanent_magnet_simple.py"
    ).read_text()

    assert source.count("jax.device_get(") == 1


def test_permanent_magnet_gpmo_keeps_numerical_workflow_on_device() -> None:
    grid = PermanentMagnetGridJAX.from_cpu(_bounded_simple_grid())

    with jax.transfer_guard("disallow"):
        result = GPMO_baseline_jax(grid, K=40)

    moments, residual, selected, target = jax.device_get(
        (result.m, result.residual, result.selected_dipoles, grid.b_obj)
    )
    initial_residual = np.asarray(target, dtype=np.float64)
    assert selected.shape == (40,)
    assert int(np.count_nonzero(np.linalg.norm(moments, axis=1))) == 40
    assert float(np.vdot(residual, residual)) < float(
        np.vdot(initial_residual, initial_residual)
    )
