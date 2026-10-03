"""JAX port of ``examples/2_Intermediate/permanent_magnet_PM4Stell.py``.

The host reads the native NCSX plasma, TF coils, PM4Stell magnet arrangement,
and corner orientations, then constructs one immutable
``PermanentMagnetGridJAX`` snapshot.  The face-triplet arbitrary-vector GPMO
backtracking solve executes on the selected JAX device; VTK output remains an
optional reporting concern outside the numerical region.
"""

from __future__ import annotations

import hashlib
import io
from contextlib import redirect_stdout
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from simsopt.field import BiotSavart, Coil
from simsopt.geo import PermanentMagnetGrid, SurfaceRZFourier
from simsopt.util import (
    FocusData,
    FocusPlasmaBnormal,
    discretize_polarizations,
    orientation_phi,
    polarization_axes,
    read_focus_coils,
)
from simsopt_jax.examples import ExampleResult, ExecutionScale, run_example
from simsopt_jax.geo.permanent_magnet_grid import PermanentMagnetGridJAX
from simsopt_jax.solve.permanent_magnet import GPMO_ArbVec_backtracking_jax
from simsopt_jax_adapters.examples.gpmo_rules import gpmo_history_period

EXAMPLE_ID = "native-permanent-magnet-pm4stell"
NATIVE_ITERATIONS = 2_000
#: Bounded scale is upstream's own ``in_github_actions`` configuration of
#: ``examples/2_Intermediate/permanent_magnet_PM4Stell.py`` (K=100,
#: max_nMagnets=20, nBacktracking=200, nAdjacent=10, nHistory=10,
#: downsample=100, N=2) -- the configuration the official CI run measured.
BOUNDED_ITERATIONS = 100
#: Upstream's ``nBacktracking``, ``nHistory`` and ``nAdjacent`` do not depend on
#: the scale; ``int(K / nHistory)`` is the record period.
BACKTRACKING = 200
HISTORY_COUNT = 10
ADJACENT_COUNT = 10
NATIVE_MAGNET_CAP = 1_000
BOUNDED_MAGNET_CAP = 20
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


def _positive_polarization_family(
    name: str, family: int
) -> tuple[np.ndarray, np.ndarray]:
    axes, types = polarization_axes([name])
    positive_count = len(types) // 2
    return axes[:positive_count], types[:positive_count] + family


def _face_triplet_polarizations(
    magnet_data: FocusData,
    corner_path: Path,
) -> np.ndarray:
    families = (
        _positive_polarization_family("face", 0),
        _positive_polarization_family("fe_ftri", 1),
        _positive_polarization_family("fc_ftri", 2),
    )
    axes = np.concatenate(tuple(family[0] for family in families), axis=0)
    types = np.concatenate(tuple(family[1] for family in families), axis=0)
    orientations = orientation_phi(corner_path)[: magnet_data.nMagnets]
    discretize_polarizations(magnet_data, orientations, axes, types)
    return np.stack(
        (magnet_data.pol_x, magnet_data.pol_y, magnet_data.pol_z),
        axis=-1,
    )


def _build_grid(
    scale: ExecutionScale,
) -> tuple[PermanentMagnetGridJAX, dict[str, str]]:
    native_scale = scale == "native_default"
    resolution = NATIVE_NPHI if native_scale else BOUNDED_NPHI
    downsample = NATIVE_DOWNSAMPLE if native_scale else BOUNDED_DOWNSAMPLE
    plasma_path = TEST_DATA / "c09r00_B_axis_half_tesla_PM4Stell.plasma"
    coil_path = TEST_DATA / "tf_only_half_tesla_symmetry_baxis_PM4Stell.focus"
    magnet_path = TEST_DATA / "magpie_trial104b_PM4Stell.focus"
    corner_path = TEST_DATA / "magpie_trial104b_corners_PM4Stell.csv"
    surface = SurfaceRZFourier.from_focus(
        plasma_path,
        range="half period",
        nphi=resolution,
        ntheta=resolution,
    )
    plasma_normal = FocusPlasmaBnormal(plasma_path).bnormal_grid(
        resolution,
        resolution,
        "half period",
    )
    base_curves, base_currents, number_of_coils = read_focus_coils(coil_path)
    coils = [
        Coil(base_curves[index], base_currents[index])
        for index in range(number_of_coils)
    ]
    field = BiotSavart(coils)
    field.set_points(surface.gamma().reshape((-1, 3)))
    coil_normal = np.sum(
        field.B().reshape((resolution, resolution, 3)) * surface.unitnormal(),
        axis=2,
    )
    magnet_data = FocusData(magnet_path, downsample=downsample)
    polarizations = _face_triplet_polarizations(magnet_data, corner_path)
    with redirect_stdout(io.StringIO()):
        cpu_grid = PermanentMagnetGrid.geo_setup_from_famus(
            surface,
            plasma_normal + coil_normal,
            magnet_path,
            pol_vectors=polarizations,
            m_maxima=5.0 / (4.0 * np.pi * 1.0e-7),
            downsample=downsample,
        )
    inputs = (plasma_path, coil_path, magnet_path, corner_path)
    return PermanentMagnetGridJAX.from_cpu(cpu_grid), {
        path.name: _sha256(path) for path in inputs
    }


def solve(
    _output_directory: Path, max_steps: int, scale: ExecutionScale
) -> ExampleResult:
    native_scale = scale == "native_default"
    grid, input_sha256 = _build_grid(scale)
    initial_error_device = jnp.linalg.norm(grid.b_obj)
    result = GPMO_ArbVec_backtracking_jax(
        grid,
        K=max_steps,
        Nadjacent=ADJACENT_COUNT,
        backtracking=BACKTRACKING,
        thresh_angle=np.pi,
        max_nMagnets=NATIVE_MAGNET_CAP if native_scale else BOUNDED_MAGNET_CAP,
        # Official kwargs record the history every ``int(K / nHistory)``
        # iterations (``initialize_default_kwargs`` sets ``verbose=True``), the
        # period upstream gives the native provider. Upstream
        # refuses ``nhistory > K``, so a budget below ``nHistory`` records one
        # row per iteration instead of aborting the run; the clamp is inert at
        # both official budgets (100 and 2000).
        record_every=gpmo_history_period(
            iterations=max_steps,
            history_count=min(HISTORY_COUNT, max_steps),
        ),
    )
    final_error_device = jnp.linalg.norm(result.residual)
    moments = np.asarray(jax.device_get(result.m), dtype=np.float64)
    errors = np.asarray(
        jax.device_get(jnp.stack((initial_error_device, final_error_device))),
        dtype=np.float64,
    )
    selected = np.flatnonzero(np.linalg.norm(moments, axis=1) > 0.0)
    selected_moments = moments[selected]
    initial_error, final_error = (float(value) for value in errors)
    solver_success = bool(
        np.all(np.isfinite(selected_moments))
        and selected.size > 0
        and final_error < initial_error
    )
    return ExampleResult(
        example_id=EXAMPLE_ID,
        observables={
            "initial_normal_error": initial_error,
            "final_normal_error": final_error,
            "selected_dipoles": tuple(int(index) for index in selected),
            "moments": tuple(
                tuple(float(component) for component in moment)
                for moment in selected_moments
            ),
            "input_sha256": input_sha256,
            "solver_success": solver_success,
        },
        status="ok" if solver_success else "failed",
    )


def main(arguments: list[str] | None = None) -> int:
    return run_example(
        arguments,
        description=__doc__,
        temporary_prefix="simsopt-jax-permanent-magnet-pm4stell-",
        bounded_steps=BOUNDED_ITERATIONS,
        native_default_steps=NATIVE_ITERATIONS,
        solve=solve,
    )


if __name__ == "__main__":
    raise SystemExit(main())
