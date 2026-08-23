"""JAX mirror of ``examples/1_Simple/permanent_magnet_simple.py``.

Host construction reads the same NCSX boundary and FAMUS magnet grid, computes
the toroidal-field normal target, and materializes the fixed optimization
matrices.  ``PermanentMagnetGridJAX.from_cpu`` is the explicit immutable
boundary; the greedy GPMO iterations and residual updates then execute on the
selected JAX device.  Optional VTK/FAMUS reporting is intentionally outside the
numerical JAX region.

GPMO places one dipole per iteration, so the published moment array is zero on
all but ``K`` of its rows.  The result therefore publishes the selected rows and
their dipole indices — the shape its permanent-magnet siblings publish — plus
the row count and a SHA-256 digest of the full moment array, so a bitwise
cross-check against another lane remains possible without shipping the zeros.
"""

from __future__ import annotations

import hashlib
import io
from contextlib import redirect_stdout
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from simsopt.field import ToroidalField
from simsopt.geo import PermanentMagnetGrid, SurfaceRZFourier
from simsopt_jax.examples import ExampleResult, ExecutionScale, run_example
from simsopt_jax.geo.permanent_magnet_grid import PermanentMagnetGridJAX
from simsopt_jax.solve.permanent_magnet import GPMO_baseline_jax

EXAMPLE_ID = "native-permanent-magnet-simple"
BOUNDED_ITERATIONS = 40
NATIVE_ITERATIONS = 500
TEST_DATA = Path(__file__).resolve().parents[3] / "tests" / "test_files"


@jax.jit
def _device_publication(
    moments: jax.Array,
    residual: jax.Array,
    selected_dipoles: jax.Array,
    target: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Fuse final diagnostics without retaining additional solver history."""
    return (
        jnp.linalg.norm(target),
        jnp.linalg.norm(residual),
        selected_dipoles,
        jnp.count_nonzero(jnp.linalg.norm(moments, axis=1)) / moments.shape[0],
        jnp.all(jnp.isfinite(moments)),
        moments,
    )


def _build_grid(scale: ExecutionScale) -> PermanentMagnetGridJAX:
    native_scale = scale == "native_default"
    nphi = 16 if native_scale else 2
    ntheta = 16 if native_scale else 2
    downsample = 4 if native_scale else 100
    surface = SurfaceRZFourier.from_wout(
        str(TEST_DATA / "wout_c09r00_fixedBoundary_0.5T_vacuum_ns201.nc"),
        range="half period",
        nphi=nphi,
        ntheta=ntheta,
    )
    net_poloidal_current = 3.7713e6
    mu0 = 4.0 * np.pi * 1.0e-7
    toroidal_field = ToroidalField(
        R0=1.0,
        B0=mu0 * net_poloidal_current / (2.0 * np.pi),
    )
    toroidal_field.set_points(surface.gamma().reshape((-1, 3)))
    normal_field = np.sum(
        toroidal_field.B().reshape((nphi, ntheta, 3)) * surface.unitnormal(),
        axis=2,
    )
    with redirect_stdout(io.StringIO()):
        cpu_grid = PermanentMagnetGrid.geo_setup_from_famus(
            surface,
            normal_field,
            TEST_DATA / "init_orient_pm_nonorm_5E4_q4_dp.focus",
            coordinate_flag="cylindrical",
            downsample=downsample,
        )
    return PermanentMagnetGridJAX.from_cpu(cpu_grid)


def solve(
    _output_directory: Path, max_steps: int, scale: ExecutionScale
) -> ExampleResult:
    grid = _build_grid(scale)
    result = GPMO_baseline_jax(grid, K=max_steps, retain_history=False)
    device_publication = _device_publication(
        result.m,
        result.residual,
        result.selected_dipoles,
        grid.b_obj,
    )
    (
        initial_error,
        final_error,
        selected,
        nonzero_fraction,
        finite_moments,
        final_moments,
    ) = jax.device_get(device_publication)
    moments = np.ascontiguousarray(final_moments, dtype=np.float64)
    placed = np.flatnonzero(np.linalg.norm(moments, axis=1) > 0.0)
    selected_moments = moments[placed]
    scientific_success = bool(
        selected.size == max_steps
        and finite_moments
        and final_error < initial_error
        and nonzero_fraction > 0.0
    )
    return ExampleResult(
        example_id=EXAMPLE_ID,
        observables={
            "initial_normal_error": float(initial_error),
            "final_normal_error": float(final_error),
            "selected_dipoles": tuple(int(value) for value in selected),
            "nonzero_fraction": float(nonzero_fraction),
            "solver_success": scientific_success,
            "ndipoles": int(moments.shape[0]),
            "selected_moment_count": int(placed.size),
            "selected_moment_indices": tuple(int(index) for index in placed),
            "selected_moments": tuple(
                tuple(float(component) for component in row) for row in selected_moments
            ),
            "moments_sha256": hashlib.sha256(moments.tobytes()).hexdigest(),
        },
        status="ok" if scientific_success else "failed",
    )


def main(arguments: list[str] | None = None) -> int:
    return run_example(
        arguments,
        description=__doc__,
        temporary_prefix="simsopt-jax-permanent-magnet-simple-",
        bounded_steps=BOUNDED_ITERATIONS,
        native_default_steps=NATIVE_ITERATIONS,
        solve=solve,
    )


if __name__ == "__main__":
    raise SystemExit(main())
