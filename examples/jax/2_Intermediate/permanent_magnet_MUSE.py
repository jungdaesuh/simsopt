"""JAX port of ``examples/2_Intermediate/permanent_magnet_MUSE.py``.

The host reads the native MUSE boundary, coils, and FAMUS magnet inventory and
constructs the fixed permanent-magnet response matrix.  The immutable
``PermanentMagnetGridJAX`` snapshot is the host/device boundary; arbitrary-vector
GPMO with backtracking then executes on the selected JAX device.  Plotting and
the native example's optional VMEC post-check are outside this parity claim.
"""

from __future__ import annotations

import hashlib
import io
from contextlib import redirect_stdout
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from simsopt.field import BiotSavart
from simsopt.geo import PermanentMagnetGrid, SurfaceRZFourier
from simsopt.util import FocusData, discretize_polarizations, polarization_axes
from simsopt.util.permanent_magnet_helper_functions import (
    initialize_coils_for_pm_optimization,
)
from simsopt_jax.examples import ExampleResult, run_example
from simsopt_jax.geo.permanent_magnet_grid import PermanentMagnetGridJAX
from simsopt_jax.solve.permanent_magnet import GPMO_ArbVec_backtracking_jax

EXAMPLE_ID = "native-permanent-magnet-muse"
NATIVE_ITERATIONS = 10_000
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


def _build_grid(max_steps: int) -> tuple[PermanentMagnetGridJAX, dict[str, str]]:
    native_scale = max_steps >= NATIVE_ITERATIONS
    nphi = 16 if native_scale else 2
    downsample = 10 if native_scale else 100
    surface_path = TEST_DATA / "input.muse"
    magnet_path = TEST_DATA / "zot80.focus"
    surface = SurfaceRZFourier.from_focus(
        surface_path,
        range="half period",
        nphi=nphi,
        ntheta=nphi,
    )
    _, _, coils = initialize_coils_for_pm_optimization(
        "muse_famus",
        TEST_DATA,
        surface,
    )
    field = BiotSavart(coils)
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
            dr=0.01,
        )
    return PermanentMagnetGridJAX.from_cpu(cpu_grid), {
        surface_path.name: _sha256(surface_path),
        magnet_path.name: _sha256(magnet_path),
    }


def solve(_output_directory: Path, max_steps: int) -> ExampleResult:
    grid, input_sha256 = _build_grid(max_steps)
    initial_error_device = jnp.linalg.norm(grid.b_obj)
    result = GPMO_ArbVec_backtracking_jax(
        grid,
        K=max_steps,
        Nadjacent=1,
        backtracking=200 if max_steps >= NATIVE_ITERATIONS else 50,
        thresh_angle=np.pi,
        max_nMagnets=5000 if max_steps >= NATIVE_ITERATIONS else 20,
        record_every=max_steps,
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
        temporary_prefix="simsopt-jax-permanent-magnet-muse-",
        bounded_steps=20,
        native_default_steps=NATIVE_ITERATIONS,
        solve=solve,
    )


if __name__ == "__main__":
    raise SystemExit(main())
