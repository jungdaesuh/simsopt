"""The GPMO carries hold no host buffer.

Coordinator flag from unit ``rcls-gsco``: ``runtime_init_array`` builds a HOST
``np.full(...)`` and places it with ``jax.device_put``; inside a traced loop
that becomes a staged host-to-device copy, which a
``transfer_guard_host_to_device("disallow")`` region rejects -- the GPU-only
"Disallowed host-to-device transfer" that killed the GSCO multistep lane.

The GPMO lanes do not share the pattern. The shipped kernels
(``simsopt_jax.solve.permanent_magnet`` -> ``simsopt_jax/core/pm_optimization.py``)
build every carry buffer with ``jnp.zeros``, which is device-native. This test
holds that negative: it fails on CPU if any GPMO carry gains a host buffer.
"""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import io
from contextlib import redirect_stdout
from pathlib import Path

import jax
import numpy as np
import pytest
from simsopt.geo import PermanentMagnetGrid
from simsopt.util import FocusData, discretize_polarizations, polarization_axes
from simsopt_jax.geo.permanent_magnet_grid import PermanentMagnetGridJAX
from simsopt_jax.solve.permanent_magnet import (
    GPMO_ArbVec_backtracking_jax,
    GPMO_baseline_jax,
)
from simsopt_jax_adapters.examples.muse import build_muse_post_geometry

TEST_DATA = Path(__file__).resolve().parents[2] / "test_files"
#: Upstream's ``in_github_actions`` configuration of
#: ``examples/2_Intermediate/permanent_magnet_MUSE.py``.
MUSE_RESOLUTION = 2
MUSE_DOWNSAMPLE = 100
MUSE_RADIAL_EXTENT = 0.01
MUSE_ITERATIONS = 100
MUSE_BACKTRACKING = 50
MUSE_MAGNET_CAP = 20
MUSE_HISTORY_COUNT = 20
MUSE_ADJACENT_COUNT = 1
MUSE_THRESHOLD_ANGLE = np.pi
BASELINE_STEPS = 10


@pytest.fixture(scope="module")
def bounded_muse_grid() -> PermanentMagnetGrid:
    """The face-polarized MUSE FAMUS grid at upstream's CI resolution, built once."""
    geometry = build_muse_post_geometry(
        TEST_DATA / "input.muse",
        TEST_DATA,
        nphi=MUSE_RESOLUTION,
        ntheta=MUSE_RESOLUTION,
    )
    surface = geometry.optimization_surface
    field = geometry.coil_field
    field.set_points(surface.gamma().reshape((-1, 3)))
    normal_field = np.sum(
        field.B().reshape((MUSE_RESOLUTION, MUSE_RESOLUTION, 3)) * surface.unitnormal(),
        axis=2,
    )
    famus = TEST_DATA / "zot80.focus"
    magnet_data = FocusData(famus, downsample=MUSE_DOWNSAMPLE)
    axes, polarization_types = polarization_axes(["face"])
    positive_count = len(polarization_types) // 2
    discretize_polarizations(
        magnet_data,
        np.arctan2(magnet_data.oy, magnet_data.ox),
        axes[:positive_count],
        polarization_types[:positive_count],
    )
    with redirect_stdout(io.StringIO()):
        return PermanentMagnetGrid.geo_setup_from_famus(
            surface,
            normal_field,
            famus,
            pol_vectors=np.stack(
                (magnet_data.pol_x, magnet_data.pol_y, magnet_data.pol_z), axis=-1
            ),
            downsample=MUSE_DOWNSAMPLE,
            dr=MUSE_RADIAL_EXTENT,
        )


def _device_grid(cpu_grid: PermanentMagnetGrid) -> PermanentMagnetGridJAX:
    """The bounded MUSE grid, placed before the guard is armed."""
    return jax.device_put(PermanentMagnetGridJAX.from_cpu(cpu_grid))


def test_arbvec_backtracking_runs_without_a_host_to_device_transfer(
    bounded_muse_grid: PermanentMagnetGrid,
) -> None:
    grid = _device_grid(bounded_muse_grid)

    with jax.transfer_guard_host_to_device("disallow"):
        result = GPMO_ArbVec_backtracking_jax(
            grid,
            K=MUSE_ITERATIONS,
            Nadjacent=MUSE_ADJACENT_COUNT,
            backtracking=MUSE_BACKTRACKING,
            thresh_angle=MUSE_THRESHOLD_ANGLE,
            max_nMagnets=MUSE_MAGNET_CAP,
            record_every=MUSE_ITERATIONS // MUSE_HISTORY_COUNT,
        )
        jax.block_until_ready(result.m)

    moments = np.asarray(jax.device_get(result.m), dtype=np.float64)
    assert int(np.count_nonzero(np.linalg.norm(moments, axis=1))) == MUSE_MAGNET_CAP


def test_baseline_gpmo_runs_without_a_host_to_device_transfer(
    bounded_muse_grid: PermanentMagnetGrid,
) -> None:
    grid = _device_grid(bounded_muse_grid)

    with jax.transfer_guard_host_to_device("disallow"):
        result = GPMO_baseline_jax(grid, K=BASELINE_STEPS)
        jax.block_until_ready(result.m)

    moments = np.asarray(jax.device_get(result.m), dtype=np.float64)
    assert int(np.count_nonzero(np.linalg.norm(moments, axis=1))) == BASELINE_STEPS
