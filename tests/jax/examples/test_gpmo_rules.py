"""The official GPMO history rules, pinned against live native GPMO runs.

Authority: ``simsopt/solve/permanent_magnet_optimization.py`` and
``examples/2_Intermediate/permanent_magnet_MUSE.py`` at upstream
``9e027eac38028d57aa23777be52a781aa860e347``. The recorded histories the rules
are applied to are produced here, by native ``simsopt.solve.GPMO`` (upstream's
own algorithm) at upstream's ``in_github_actions`` configurations of the MUSE
and PM4Stell scripts, so this file stores no reference data.
"""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import io
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from simsopt.field import BiotSavart, Coil
from simsopt.geo import PermanentMagnetGrid, SurfaceRZFourier
from simsopt.solve import GPMO
from simsopt.util import (
    FocusData,
    FocusPlasmaBnormal,
    discretize_polarizations,
    orientation_phi,
    polarization_axes,
    read_focus_coils,
)
from simsopt_jax_adapters.examples.gpmo_rules import (
    gpmo_backtracking_outputs_usable,
    gpmo_history_period,
    recorded_history_length,
    select_minimum_objective_snapshot,
)
from simsopt_jax_adapters.examples.muse import build_muse_post_geometry

TEST_DATA = Path(__file__).resolve().parents[2] / "test_files"
VACUUM_PERMEABILITY = 4.0 * np.pi * 1.0e-7
#: Upstream's ``in_github_actions`` resolution of both scripts.
CI_RESOLUTION = 2
CI_DOWNSAMPLE = 100
#: ``permanent_magnet_MUSE.py`` under ``in_github_actions``.
MUSE_CI_OPTIONS = {
    "K": 100,
    "nhistory": 20,
    "backtracking": 50,
    "Nadjacent": 1,
    "thresh_angle": np.pi,
    "max_nMagnets": 20,
}
#: ``permanent_magnet_PM4Stell.py`` under ``in_github_actions``.
PM4STELL_CI_OPTIONS = {
    "K": 100,
    "nhistory": 10,
    "backtracking": 200,
    "Nadjacent": 10,
    "thresh_angle": np.pi,
    "max_nMagnets": 20,
}


def _muse_ci_grid() -> PermanentMagnetGrid:
    geometry = build_muse_post_geometry(
        TEST_DATA / "input.muse",
        TEST_DATA,
        nphi=CI_RESOLUTION,
        ntheta=CI_RESOLUTION,
    )
    surface = geometry.optimization_surface
    field = geometry.coil_field
    field.set_points(surface.gamma().reshape((-1, 3)))
    normal_field = np.sum(
        field.B().reshape((CI_RESOLUTION, CI_RESOLUTION, 3)) * surface.unitnormal(),
        axis=2,
    )
    famus = TEST_DATA / "zot80.focus"
    magnet_data = FocusData(famus, downsample=CI_DOWNSAMPLE)
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
            downsample=CI_DOWNSAMPLE,
            dr=0.01,
        )


def _pm4stell_ci_grid() -> PermanentMagnetGrid:
    plasma = TEST_DATA / "c09r00_B_axis_half_tesla_PM4Stell.plasma"
    famus = TEST_DATA / "magpie_trial104b_PM4Stell.focus"
    surface = SurfaceRZFourier.from_focus(
        plasma, range="half period", nphi=CI_RESOLUTION, ntheta=CI_RESOLUTION
    )
    curves, currents, count = read_focus_coils(
        TEST_DATA / "tf_only_half_tesla_symmetry_baxis_PM4Stell.focus"
    )
    field = BiotSavart([Coil(curves[index], currents[index]) for index in range(count)])
    field.set_points(surface.gamma().reshape((-1, 3)))
    coil_normal = np.sum(
        field.B().reshape((CI_RESOLUTION, CI_RESOLUTION, 3)) * surface.unitnormal(),
        axis=2,
    )
    plasma_normal = FocusPlasmaBnormal(plasma).bnormal_grid(
        CI_RESOLUTION, CI_RESOLUTION, "half period"
    )
    magnet_data = FocusData(famus, downsample=CI_DOWNSAMPLE)
    # The positive half of each official polarization family.
    families = [polarization_axes([name]) for name in ("face", "fe_ftri", "fc_ftri")]
    axes = np.concatenate([axes[: len(types) // 2] for axes, types in families])
    types = np.concatenate(
        [types[: len(types) // 2] + index for index, (_, types) in enumerate(families)]
    )
    orientation = orientation_phi(TEST_DATA / "magpie_trial104b_corners_PM4Stell.csv")
    discretize_polarizations(
        magnet_data, orientation[: magnet_data.nMagnets], axes, types
    )
    with redirect_stdout(io.StringIO()):
        return PermanentMagnetGrid.geo_setup_from_famus(
            surface,
            plasma_normal + coil_normal,
            famus,
            pol_vectors=np.stack(
                (magnet_data.pol_x, magnet_data.pol_y, magnet_data.pol_z), axis=-1
            ),
            m_maxima=5.0 / VACUUM_PERMEABILITY,
            downsample=CI_DOWNSAMPLE,
        )


def _native_history(
    grid: PermanentMagnetGrid, options: dict[str, float]
) -> NDArray[np.float64]:
    """Upstream's ``R2`` history: ``errors = algorithm_history[!= 0]``."""
    with redirect_stdout(io.StringIO()):
        errors, _, _ = GPMO(
            grid,
            "ArbVec_backtracking",
            dipole_grid_xyz=grid.dipole_grid_xyz,
            verbose=True,
            **options,
        )
    return np.asarray(errors, dtype=np.float64)


@pytest.fixture(scope="module")
def muse_ci_r2() -> NDArray[np.float64]:
    """``R2`` of native MUSE at its CI configuration (K=100, nhistory=20, cap 20)."""
    return _native_history(_muse_ci_grid(), MUSE_CI_OPTIONS)


@pytest.fixture(scope="module")
def pm4stell_ci_r2() -> NDArray[np.float64]:
    """``R2`` of native PM4Stell at its CI configuration (K=100, nhistory=10, cap 20).

    Non-monotone: the minimum is row 1, the endpoint is row 3.
    """
    return _native_history(_pm4stell_ci_grid(), PM4STELL_CI_OPTIONS)


def _moment_history(objective: np.ndarray) -> np.ndarray:
    """One distinguishable ``(ndipoles, 3, rows)`` snapshot per recorded row."""

    rows = objective.size
    history = np.zeros((4, 3, rows), dtype=np.float64)
    for row in range(rows):
        history[:, :, row] = float(row + 1)
    return history


def test_the_official_period_is_upstream_int_division() -> None:
    """``k % int(K / nhistory)`` (``permanent_magnet_optimization.cpp:942``)."""

    # The shipped MUSE run (K=10000, nhistory=20) and the PM4Stell CI run.
    assert gpmo_history_period(iterations=10_000, history_count=20) == 500
    assert (
        gpmo_history_period(
            iterations=PM4STELL_CI_OPTIONS["K"],
            history_count=PM4STELL_CI_OPTIONS["nhistory"],
        )
        == 10
    )
    # ``nhistory == K`` is the boundary upstream allows: one row per iteration.
    assert gpmo_history_period(iterations=20, history_count=20) == 1


def test_nhistory_above_k_is_refused_as_upstream_refuses_it() -> None:
    """``GPMO`` raises for ``nhistory > K``; the C++ would compute ``k % 0``."""

    with pytest.raises(ValueError, match="nhistory"):
        gpmo_history_period(iterations=20, history_count=21)


def test_the_recorded_length_is_the_filled_prefix(
    muse_ci_r2: NDArray[np.float64],
) -> None:
    """``errors = algorithm_history[algorithm_history != 0]``.

    The C++ allocates ``nhistory + 2`` slots and fills a prefix; the native MUSE
    CI run fills 6 of 22.
    """

    slots = int(MUSE_CI_OPTIONS["nhistory"]) + 2
    padded = np.zeros(slots, dtype=np.float64)
    padded[: muse_ci_r2.size] = muse_ci_r2
    assert recorded_history_length(padded) == muse_ci_r2.size


def test_a_hole_in_the_recorded_history_is_refused() -> None:
    holed = np.asarray([1.0, 0.0, 3.0, 0.0], dtype=np.float64)
    with pytest.raises(ValueError, match="filled prefix"):
        recorded_history_length(holed)


def test_official_muse_history_selects_its_endpoint(
    muse_ci_r2: NDArray[np.float64],
) -> None:
    """Monotone ``R2`` -> the last recorded row, which is the run endpoint."""

    selection = select_minimum_objective_snapshot(
        muse_ci_r2,
        _moment_history(muse_ci_r2),
    )
    recorded_length = muse_ci_r2.size
    assert selection.length == recorded_length
    assert selection.index == recorded_length - 1
    assert np.array_equal(
        selection.moments,
        np.full((4, 3), float(recorded_length)),
    )


def test_a_non_monotone_history_does_not_select_the_endpoint(
    pm4stell_ci_r2: NDArray[np.float64],
) -> None:
    """PM4Stell CI rises after row 1, so argmin is not the endpoint.

    This is the shape that separates "post metrics from the selected snapshot"
    from "post metrics from the GPMO endpoint": the two rows differ.
    """

    selection = select_minimum_objective_snapshot(
        pm4stell_ci_r2,
        _moment_history(pm4stell_ci_r2),
    )
    recorded_length = pm4stell_ci_r2.size
    assert selection.length == recorded_length
    assert selection.index == 1
    assert selection.index != recorded_length - 1
    assert np.array_equal(selection.moments, np.full((4, 3), 2.0))


def test_the_selection_is_invariant_under_the_half_factor(
    pm4stell_ci_r2: NDArray[np.float64],
) -> None:
    """The JAX kernel records ``sum(r * r)``; upstream records half of it."""

    history = _moment_history(pm4stell_ci_r2)
    assert (
        select_minimum_objective_snapshot(2.0 * pm4stell_ci_r2, history).index
        == select_minimum_objective_snapshot(pm4stell_ci_r2, history).index
    )


def test_a_short_moment_history_is_refused(
    pm4stell_ci_r2: NDArray[np.float64],
) -> None:
    with pytest.raises(ValueError, match="ndipoles"):
        select_minimum_objective_snapshot(pm4stell_ci_r2, np.zeros((4, 3, 2)))


def test_a_nan_moment_row_is_invisible_to_the_nonzero_selection_filter() -> None:
    """The mechanism the MUSE example's success gate has to survive.

    ``selected = np.flatnonzero(np.linalg.norm(moments, axis=1) > 0.0)`` drops a
    NaN row, because ``norm([nan, 0, 0])`` is ``nan`` and ``nan > 0.0`` is
    False. A finiteness check applied to ``moments[selected]`` therefore never
    sees the NaN, and the example published ``status='ok'`` with a NaN in the
    solution. The rule checks the FULL array instead.
    """

    moments = np.zeros((3, 3), dtype=np.float64)
    moments[0] = (np.nan, 0.0, 0.0)
    moments[1] = (1.0, 0.0, 0.0)
    selected = np.flatnonzero(np.linalg.norm(moments, axis=1) > 0.0)
    assert selected.tolist() == [1]
    assert bool(np.all(np.isfinite(moments[selected]))) is True
    assert (
        gpmo_backtracking_outputs_usable(
            moments=moments,
            nonzero_count=int(selected.size),
            grid_size=3,
            magnet_cap=20,
        )
        is False
    )
