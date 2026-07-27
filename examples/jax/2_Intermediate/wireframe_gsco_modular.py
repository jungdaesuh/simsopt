"""JAX port of ``examples/2_Intermediate/wireframe_gsco_modular.py``.

The host constructs the native QA plasma, NESCOIL wireframe, modular-TF seed,
and poloidal-current constraint.  The fixed normal-field response and topology
are then consumed by the sampled-history JAX GSCO kernel on the selected
device.  Only the accepted currents and scalar diagnostics cross back to the
host; plotting is deliberately outside the numerical region.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from simsopt.geo import SurfaceRZFourier, ToroidalWireframe
from simsopt_jax.examples import ExampleResult, run_example
from simsopt_jax_adapters.solve.wireframe import (
    bnorm_obj_matrices_jax,
    gsco_wireframe_jax,
)

EXAMPLE_ID = "native-wireframe-gsco-modular"
NATIVE_ITERATIONS = 2_000
TEST_DATA = Path(__file__).resolve().parents[3] / "tests" / "test_files"


def _build_problem(
    max_steps: int,
) -> tuple[ToroidalWireframe, SurfaceRZFourier, float]:
    native_scale = max_steps >= NATIVE_ITERATIONS
    plasma_resolution = 32 if native_scale else 4
    wireframe_nphi = 48 if native_scale else 18
    wireframe_ntheta = 50 if native_scale else 8
    plasma = SurfaceRZFourier.from_vmec_input(
        TEST_DATA / "input.LandremanPaul2021_QA",
        nphi=plasma_resolution,
        ntheta=plasma_resolution,
        range="half period",
    )
    wireframe_surface = SurfaceRZFourier.from_nescoil_input(
        TEST_DATA / "nescin.LandremanPaul2021_QA",
        "current",
    )
    wireframe = ToroidalWireframe(
        wireframe_surface,
        wireframe_nphi,
        wireframe_ntheta,
    )
    mu0 = 4.0 * np.pi * 1.0e-7
    poloidal_current = -2.0 * np.pi * plasma.get_rc(0, 0) / mu0
    number_of_coils = 6
    coil_current = poloidal_current / (2 * wireframe.nfp * number_of_coils)
    wireframe.add_tfcoil_currents(number_of_coils, coil_current)
    wireframe.set_poloidal_current(poloidal_current)
    return wireframe, plasma, poloidal_current


def solve(_output_directory: Path, max_steps: int) -> ExampleResult:
    wireframe, plasma, poloidal_current = _build_problem(max_steps)
    response, target = bnorm_obj_matrices_jax(
        wireframe,
        plasma,
        area_weighted=True,
        verbose=False,
    )
    initial_currents = np.asarray(wireframe.currents, dtype=np.float64).reshape((-1, 1))
    result = gsco_wireframe_jax(
        wireframe,
        response,
        target,
        lambda_S=1.0e-6,
        no_crossing=True,
        match_current=False,
        default_current=abs(poloidal_current / (2 * wireframe.nfp * 6)),
        max_current=1.1 * abs(poloidal_current / (2 * wireframe.nfp * 6)),
        max_iter=max_steps,
        print_interval=max_steps,
        record_every=max_steps,
        verbose=False,
    )
    response_device = jnp.asarray(response, dtype=jnp.float64)
    target_device = jnp.asarray(target, dtype=jnp.float64)
    initial_device = jnp.asarray(initial_currents, dtype=jnp.float64)
    initial_error_device = jnp.linalg.norm(
        response_device @ initial_device - target_device
    )
    final_error_device = jnp.linalg.norm(response_device @ result.x - target_device)
    diagnostics = np.asarray(
        jax.device_get(
            jnp.stack(
                (
                    initial_error_device,
                    final_error_device,
                    jnp.max(jnp.abs(result.x)),
                )
            )
        ),
        dtype=np.float64,
    )
    solution = np.asarray(jax.device_get(result.x), dtype=np.float64).ravel()
    wireframe.currents[:] = solution
    history_length = int(jax.device_get(result.history_length))
    iterations = int(
        np.asarray(jax.device_get(result.iter_history), dtype=np.int64)[
            history_length - 1
        ]
    )
    initial_error, final_error, maximum_current = (
        float(value) for value in diagnostics
    )
    solver_success = bool(
        np.all(np.isfinite(solution))
        and final_error < initial_error
        and maximum_current > 0.0
        and wireframe.check_constraints()
    )
    return ExampleResult(
        example_id=EXAMPLE_ID,
        observables={
            "initial_normal_error": initial_error,
            "final_normal_error": final_error,
            "maximum_current": maximum_current,
            "iterations": iterations,
            "solver_success": solver_success,
        },
        status="ok" if solver_success else "failed",
    )


def main(arguments: list[str] | None = None) -> int:
    return run_example(
        arguments,
        description=__doc__,
        temporary_prefix="simsopt-jax-wireframe-gsco-modular-",
        bounded_steps=40,
        native_default_steps=NATIVE_ITERATIONS,
        solve=solve,
    )


if __name__ == "__main__":
    raise SystemExit(main())
