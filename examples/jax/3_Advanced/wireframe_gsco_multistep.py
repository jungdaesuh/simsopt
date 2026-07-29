"""JAX port of ``examples/3_Advanced/wireframe_gsco_multistep.py``.

The host constructs the native QA plasma, sector-constrained NESCOIL
wireframe, and external planar TF coils.  Their fixed normal-field response and
wireframe topology form one immutable payload for the complete multistep GSCO
state machine on the selected JAX device.  Stage objectives remain device-side
until the final report; accepted currents are published once for validation.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from simsopt.field import Current, coils_via_symmetries
from simsopt.geo import (
    SurfaceRZFourier,
    ToroidalWireframe,
    create_equally_spaced_curves,
)
from simsopt_jax.core.wireframe_workflow import (
    WireframeGSCOLiveParams,
    wireframe_gsco_multistep_loop_jax,
)
from simsopt_jax.examples import ExampleResult, ExecutionScale, run_example
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt_jax_adapters.solve.wireframe import bnorm_obj_matrices_jax

EXAMPLE_ID = "native-wireframe-gsco-multistep"
NATIVE_ITERATIONS = 2_500
TEST_DATA = Path(__file__).resolve().parents[3] / "tests" / "test_files"


def _build_problem(
    scale: ExecutionScale,
) -> tuple[ToroidalWireframe, SurfaceRZFourier, BiotSavartJAX, float]:
    native_scale = scale == "native_default"
    plasma_resolution = 32 if native_scale else 4
    wireframe_nphi = 96 if native_scale else 24
    wireframe_ntheta = 100 if native_scale else 8
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
    number_of_tf_coils = 3
    wireframe.set_toroidal_breaks(
        number_of_tf_coils,
        4,
        allow_pol_current=True,
    )
    wireframe.set_poloidal_current(0.0)
    tf_curves = create_equally_spaced_curves(
        number_of_tf_coils,
        plasma.nfp,
        True,
        R0=1.0,
        R1=0.85,
    )
    tf_currents = [
        Current(-poloidal_current / (2 * number_of_tf_coils * plasma.nfp))
        for _ in range(number_of_tf_coils)
    ]
    tf_coils = coils_via_symmetries(
        tf_curves,
        tf_currents,
        plasma.nfp,
        True,
    )
    return wireframe, plasma, BiotSavartJAX(tf_coils), poloidal_current


def solve(
    _output_directory: Path, max_steps: int, scale: ExecutionScale
) -> ExampleResult:
    wireframe, plasma, external_field, poloidal_current = _build_problem(scale)
    response, target = bnorm_obj_matrices_jax(
        wireframe,
        plasma,
        ext_field=external_field,
        area_weighted=True,
        verbose=False,
    )
    loops = np.asarray(wireframe.get_cell_key(), dtype=np.int32)
    free_loops = np.asarray(
        wireframe.get_free_cells(form="logical"),
        dtype=np.int32,
    )
    segments = np.asarray(wireframe.segments, dtype=np.int32)
    connections = np.asarray(wireframe.connected_segments, dtype=np.int32)
    base_constrained = np.zeros(wireframe.n_segments, dtype=np.bool_)
    base_constrained[np.asarray(wireframe.constrained_segments(), dtype=np.intp)] = True
    initial_fraction = 0.2
    current_scale = abs(poloidal_current)
    initial_default_current = initial_fraction * current_scale
    response_device = jax.device_put(np.asarray(response, dtype=np.float64))
    target_device = jax.device_put(np.asarray(target, dtype=np.float64))
    loops_device = jax.device_put(loops)
    neighbors_device = jax.device_put(
        np.asarray(wireframe.get_cell_neighbors(), dtype=np.int32)
    )
    params = WireframeGSCOLiveParams(
        A=response_device,
        loops=loops_device,
        free_loops=jax.device_put(free_loops),
        segments=jax.device_put(segments),
        connections=jax.device_put(connections),
        default_current=jax.device_put(
            np.asarray(initial_default_current, dtype=np.float64)
        ),
        max_current=jax.device_put(
            np.asarray(1.1 * initial_default_current, dtype=np.float64)
        ),
        lambda_s=jax.device_put(np.asarray(1.0e-7, dtype=np.float64)),
        tol=jax.device_put(
            np.asarray(0.001 * initial_default_current, dtype=np.float64)
        ),
        max_loop_count=1,
        no_crossing=True,
        no_new_coils=False,
        match_current=False,
    )
    initial_currents = jax.device_put(
        np.zeros((wireframe.n_segments,), dtype=np.float64)
    )
    result = wireframe_gsco_multistep_loop_jax(
        params,
        target_device,
        initial_currents,
        jax.device_put(np.zeros((loops.shape[0],), dtype=np.int32)),
        loops_device,
        neighbors_device,
        jax.device_put(base_constrained),
        max_iter_per_step=max_steps,
        max_outer_steps=12 if scale == "native_default" else 4,
        initial_current_fraction=initial_fraction,
        current_scale=current_scale,
        min_coil_size=20 if scale == "native_default" else 2,
        final_max_current=1.1 * initial_default_current,
    )
    stage_count = int(jax.device_get(result.stage_count))
    all_stage_objectives = np.asarray(
        jax.device_get(result.stage_objectives),
        dtype=np.float64,
    )
    stage_objectives = all_stage_objectives[:stage_count]
    solution = np.asarray(jax.device_get(result.x), dtype=np.float64).ravel()
    initial_normal_error = float(jax.device_get(jnp.linalg.norm(target_device)))
    final_normal_error = float(np.sqrt(2.0 * stage_objectives[-1]))
    maximum_current = float(np.max(np.abs(solution)))
    nonfinal_steps = int(jax.device_get(result.nonfinal_steps))
    wireframe.currents[:] = solution
    solver_success = bool(
        np.all(np.isfinite(solution))
        and np.all(np.isfinite(stage_objectives))
        and stage_count > 0
        and final_normal_error < initial_normal_error
        and maximum_current > 0.0
        and wireframe.check_constraints()
    )
    return ExampleResult(
        example_id=EXAMPLE_ID,
        observables={
            "stage_objectives": tuple(float(value) for value in stage_objectives),
            "final_normal_error": final_normal_error,
            "maximum_current": maximum_current,
            "iterations": nonfinal_steps * max_steps,
            "solver_success": solver_success,
        },
        status="ok" if solver_success else "failed",
    )


def main(arguments: list[str] | None = None) -> int:
    return run_example(
        arguments,
        description=__doc__,
        temporary_prefix="simsopt-jax-wireframe-gsco-multistep-",
        bounded_steps=40,
        native_default_steps=NATIVE_ITERATIONS,
        solve=solve,
    )


if __name__ == "__main__":
    raise SystemExit(main())
