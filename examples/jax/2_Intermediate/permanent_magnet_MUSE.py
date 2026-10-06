"""JAX port of ``examples/2_Intermediate/permanent_magnet_MUSE.py``.

The host reads the native MUSE boundary, coils, and FAMUS magnet inventory and
constructs the fixed permanent-magnet response matrix.  The immutable
``PermanentMagnetGridJAX`` snapshot is the host/device boundary; arbitrary-vector
GPMO with backtracking then executes on the selected JAX device. The mandatory
post-optimization metrics and TF-coil fieldline trace follow the official
workflow; the native example's disabled VMEC post-check remains excluded.
"""

from __future__ import annotations

import hashlib
import io
from contextlib import redirect_stdout
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from simsopt.geo import PermanentMagnetGrid
from simsopt.util import FocusData, discretize_polarizations, polarization_axes
from simsopt_jax.examples import ExampleResult, ExecutionScale, run_example
from simsopt_jax.geo.permanent_magnet_grid import PermanentMagnetGridJAX
from simsopt_jax.solve.permanent_magnet import GPMO_ArbVec_backtracking_jax
from simsopt_jax_adapters.examples.gpmo_rules import (
    gpmo_backtracking_outputs_usable,
    gpmo_history_period,
    select_minimum_objective_snapshot,
)
from simsopt_jax_adapters.examples.muse import (
    MusePostGeometry,
    build_muse_post_geometry,
    muse_post_workflow_policy,
    run_jax_muse_post_workflow,
)

EXAMPLE_ID = "native-permanent-magnet-muse"
NATIVE_ITERATIONS = 10_000
#: Bounded scale is upstream's own ``in_github_actions`` configuration of
#: ``examples/2_Intermediate/permanent_magnet_MUSE.py`` (K=100,
#: nBacktracking=50, max_nMagnets=20, downsample=100, nphi=ntheta=2,
#: nHistory=20) -- the configuration the official CI run measured.
BOUNDED_ITERATIONS = 100
#: Official ``nHistory`` (``permanent_magnet_MUSE.py:153``), the same at both
#: scales; ``int(K / nHistory)`` is upstream's record period.
HISTORY_COUNT = 20
NATIVE_BACKTRACKING = 200
BOUNDED_BACKTRACKING = 50
NATIVE_MAGNET_CAP = 5_000
BOUNDED_MAGNET_CAP = 20
#: Official ``nAdjacent`` and ``dr``; neither depends on the scale.
ADJACENT_COUNT = 1
RADIAL_EXTENT = 0.01
NATIVE_NPHI = 16
BOUNDED_NPHI = 2
NATIVE_DOWNSAMPLE = 10
BOUNDED_DOWNSAMPLE = 100
TEST_DATA = Path(__file__).resolve().parents[3] / "tests" / "test_files"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _face_polarizations(magnet_data: FocusData) -> np.ndarray:
    axes, types = polarization_axes(["face"])
    positive_count = len(types) // 2
    positive_axes = axes[:positive_count]
    positive_types = types[:positive_count]
    orientation = np.arctan2(magnet_data.oy, magnet_data.ox)
    discretize_polarizations(
        magnet_data,
        orientation,
        positive_axes,
        positive_types,
    )
    return np.stack(
        (magnet_data.pol_x, magnet_data.pol_y, magnet_data.pol_z),
        axis=-1,
    )


def _build_grid(
    scale: ExecutionScale,
) -> tuple[PermanentMagnetGridJAX, dict[str, str], MusePostGeometry]:
    native_scale = scale == "native_default"
    nphi = NATIVE_NPHI if native_scale else BOUNDED_NPHI
    downsample = NATIVE_DOWNSAMPLE if native_scale else BOUNDED_DOWNSAMPLE
    surface_path = TEST_DATA / "input.muse"
    magnet_path = TEST_DATA / "zot80.focus"
    post_geometry = build_muse_post_geometry(
        surface_path,
        TEST_DATA,
        nphi=nphi,
        ntheta=nphi,
    )
    surface = post_geometry.optimization_surface
    field = post_geometry.coil_field
    field.set_points(surface.gamma().reshape((-1, 3)))
    normal_field = np.sum(
        field.B().reshape((nphi, nphi, 3)) * surface.unitnormal(),
        axis=2,
    )
    magnet_data = FocusData(magnet_path, downsample=downsample)
    polarizations = _face_polarizations(magnet_data)
    with redirect_stdout(io.StringIO()):
        cpu_grid = PermanentMagnetGrid.geo_setup_from_famus(
            surface,
            normal_field,
            magnet_path,
            pol_vectors=polarizations,
            downsample=downsample,
            dr=RADIAL_EXTENT,
        )
    return (
        PermanentMagnetGridJAX.from_cpu(cpu_grid),
        {
            surface_path.name: _sha256(surface_path),
            magnet_path.name: _sha256(magnet_path),
        },
        post_geometry,
    )


def solve(
    _output_directory: Path, max_steps: int, scale: ExecutionScale
) -> ExampleResult:
    native_scale = scale == "native_default"
    grid, input_sha256, post_geometry = _build_grid(scale)
    initial_error_device = jnp.linalg.norm(grid.b_obj)
    magnet_cap = NATIVE_MAGNET_CAP if native_scale else BOUNDED_MAGNET_CAP
    result = GPMO_ArbVec_backtracking_jax(
        grid,
        K=max_steps,
        Nadjacent=ADJACENT_COUNT,
        backtracking=NATIVE_BACKTRACKING if native_scale else BOUNDED_BACKTRACKING,
        thresh_angle=np.pi,
        max_nMagnets=magnet_cap,
        # Official kwargs record the history every ``int(K / nHistory)``
        # iterations (``initialize_default_kwargs`` sets ``verbose=True``);
        # ``record_every`` is the JAX kernel's mirror of that period. Upstream
        # refuses ``nhistory > K`` (the C++ would compute ``k % 0``), so a
        # budget below ``nHistory`` records one row per iteration instead of
        # aborting the run: ``--max-steps`` accepts any budget >= 1, while the
        # clamp is inert at both official budgets (100 and 10000).
        record_every=gpmo_history_period(
            iterations=max_steps,
            history_count=min(HISTORY_COUNT, max_steps),
        ),
    )
    final_error_device = jnp.linalg.norm(result.residual)
    moments, moment_history, objective_history = (
        np.asarray(value, dtype=np.float64)
        for value in jax.device_get(
            (result.m, result.m_history, result.residual_history)
        )
    )
    dipole_grid_xyz, moment_maxima = (
        np.asarray(value, dtype=np.float64)
        for value in jax.device_get((grid.dipole_grid_xyz, grid.m_maxima))
    )
    errors = np.asarray(
        jax.device_get(jnp.stack((initial_error_device, final_error_device))),
        dtype=np.float64,
    )
    # Official ``permanent_magnet_MUSE.py:184-185`` replaces ``pm_opt.m`` with
    # the recorded snapshot of smallest ``R2`` BEFORE the printed volume, the
    # dipole squared flux on the plotting surface and the total magnet volume.
    # ``residual_history`` is ``sum(r * r)`` and upstream's ``R2`` is half of
    # it, which does not move the argmin; ``m_history`` arrives as
    # ``(rows, ndipoles, 3)`` and the rule wants the C++ ``(ndipoles, 3, rows)``.
    selection = select_minimum_objective_snapshot(
        0.5 * objective_history,
        np.transpose(moment_history, (1, 2, 0)),
    )
    snapshot_moments = selection.moments
    selected = np.flatnonzero(np.linalg.norm(snapshot_moments, axis=1) > 0.0)
    initial_error, final_error = (float(value) for value in errors)
    # Upstream GPMO stops at the magnet cap, at a full grid or at K, and makes
    # no claim about the objective: official PM4Stell itself rises from 0.16177
    # to 0.83456. "The error decreased" and "every traced line reached a
    # terminal state" are therefore published diagnostics, not completion
    # gates; the example is complete when it returned finite moments under the
    # cap. Finiteness is checked on the FULL moment array and on both errors
    # before any selection: ``norm([nan, 0, 0]) > 0.0`` is False, so filtering
    # first would drop exactly the rows a finiteness check must see.
    solver_success = bool(
        gpmo_backtracking_outputs_usable(
            moments=snapshot_moments,
            nonzero_count=int(selected.size),
            grid_size=grid.ndipoles,
            magnet_cap=magnet_cap,
        )
        and np.all(np.isfinite(moments))
        and np.isfinite(initial_error)
        and np.isfinite(final_error)
    )
    post_diagnostics = run_jax_muse_post_workflow(
        post_geometry,
        muse_post_workflow_policy(scale),
        snapshot_moments,
        dipole_grid_xyz,
        moment_maxima,
        coordinate_flag=grid.coordinate_flag,
    )
    return ExampleResult(
        example_id=EXAMPLE_ID,
        observables={
            "initial_normal_error": initial_error,
            "final_normal_error": final_error,
            "selected_dipoles": tuple(int(index) for index in selected),
            "moments": tuple(
                tuple(float(component) for component in moment)
                for moment in snapshot_moments[selected]
            ),
            # Which recorded snapshot the official rule selected.
            "selected_history_index": selection.index,
            "selected_history_length": selection.length,
            "selected_history_equals_endpoint": bool(
                np.array_equal(snapshot_moments, moments)
            ),
            "input_sha256": input_sha256,
            "solver_success": solver_success,
            "normal_error_decreased": bool(final_error < initial_error),
            "post_coil_only_trace_terminal": post_diagnostics.trace_success,
            "post_dipole_squared_flux": post_diagnostics.dipole_squared_flux,
            "post_total_magnet_volume": post_diagnostics.total_magnet_volume,
            "post_coil_only_trace_status": post_diagnostics.trace_statuses,
            "post_coil_only_trace_final_times": post_diagnostics.trace_final_times,
            "post_coil_only_trace_hit_counts": post_diagnostics.trace_hit_counts,
        },
        status="ok" if solver_success else "failed",
    )


def main(arguments: list[str] | None = None) -> int:
    return run_example(
        arguments,
        description=__doc__,
        temporary_prefix="simsopt-jax-permanent-magnet-muse-",
        bounded_steps=BOUNDED_ITERATIONS,
        native_default_steps=NATIVE_ITERATIONS,
        solve=solve,
    )


if __name__ == "__main__":
    raise SystemExit(main())
