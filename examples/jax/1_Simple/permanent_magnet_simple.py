"""JAX mirror of ``examples/1_Simple/permanent_magnet_simple.py``.

Host construction reads the same NCSX boundary and FAMUS magnet grid, computes
the toroidal-field normal target, and materializes the fixed optimization
matrices.  ``PermanentMagnetGridJAX.from_cpu`` is the explicit immutable
boundary; the greedy GPMO iterations and residual updates then execute on the
selected JAX device.  Optional VTK/FAMUS reporting is intentionally outside the
numerical JAX region.
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from simsopt.field import ToroidalField
from simsopt.geo import PermanentMagnetGrid, SurfaceRZFourier
from simsopt_jax.examples import ExampleResult, run_example
from simsopt_jax.geo.permanent_magnet_grid import PermanentMagnetGridJAX
from simsopt_jax.solve.permanent_magnet import GPMO_baseline_jax

EXAMPLE_ID = "native-permanent-magnet-simple"
TEST_DATA = Path(__file__).resolve().parents[3] / "tests" / "test_files"


def _build_grid() -> PermanentMagnetGridJAX:
    nphi = 2
    ntheta = 2
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
            downsample=100,
        )
    return PermanentMagnetGridJAX.from_cpu(cpu_grid)


def solve(_output_directory: Path, max_steps: int) -> ExampleResult:
    grid = _build_grid()
    result = GPMO_baseline_jax(grid, K=max_steps)
    device_publication = (
        jnp.linalg.norm(grid.b_obj),
        jnp.linalg.norm(result.residual),
        result.selected_dipoles,
        jnp.count_nonzero(jnp.linalg.norm(result.m, axis=1)) / grid.ndipoles,
        jnp.all(jnp.isfinite(result.m)),
        result.m,
    )
    (
        initial_error,
        final_error,
        selected,
        nonzero_fraction,
        finite_moments,
        final_moments,
    ) = jax.device_get(device_publication)
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
            "moments": tuple(
                tuple(float(component) for component in row) for row in final_moments
            ),
        },
        status="ok" if scientific_success else "failed",
    )


def main(arguments: list[str] | None = None) -> int:
    return run_example(
        arguments,
        description=__doc__,
        temporary_prefix="simsopt-jax-permanent-magnet-simple-",
        bounded_steps=40,
        native_default_steps=500,
        solve=solve,
    )


if __name__ == "__main__":
    raise SystemExit(main())
