"""Memory-bounded result coverage for the baseline JAX GPMO solver."""

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

TEST_DATA = Path(__file__).resolve().parents[1] / "test_files"
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


def test_permanent_magnet_example_can_omit_iteration_state_history() -> None:
    grid = PermanentMagnetGridJAX.from_cpu(_bounded_simple_grid())

    result = GPMO_baseline_jax(grid, K=40, retain_history=False)
    moments, selected = jax.device_get((result.m, result.selected_dipoles))

    assert result.x_history.shape == (0, grid.ndipoles, 3)
    assert result.m_history.shape == (0, grid.ndipoles, 3)
    assert result.residual_history.shape == (0,)
    assert selected.shape == (40,)
    assert int(np.count_nonzero(np.linalg.norm(moments, axis=1))) == 40
