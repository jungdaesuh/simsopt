"""JAX port of ``examples/2_Intermediate/wireframe_rcls_with_ports.py``.

The host constructs the native QA plasma, offset wireframe, symmetric circular
ports, collision constraints, and poloidal-current constraint.  The immutable
normal-field response enters the JAX constrained least-squares solve on the
selected device.  Accepted currents are published once for constraint and port
clearance validation; plotting remains outside the numerical region.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from simsopt.geo import CircularPort, PortSet, SurfaceRZFourier, ToroidalWireframe
from simsopt_jax.examples import ExampleResult, run_example
from simsopt_jax_adapters.solve.wireframe import (
    bnorm_obj_matrices_jax,
    rcls_wireframe_jax,
)

EXAMPLE_ID = "native-wireframe-rcls-with-ports"
TEST_DATA = Path(__file__).resolve().parents[3] / "tests" / "test_files"
REGULARIZATION_WEIGHT = 1.0e-10
PORT_GAP = 0.04


def _ports_on_surface(surface: SurfaceRZFourier) -> PortSet:
    ports = PortSet()
    gamma = surface.gamma()
    normal = surface.normal()
    for phi in (np.pi / 8.0, 3.0 * np.pi / 8.0):
        phi_index = int(np.argmin(np.abs((0.5 / np.pi) * phi - surface.quadpoints_phi)))
        for theta in (np.pi / 4.0, 7.0 * np.pi / 4.0):
            theta_index = int(
                np.argmin(np.abs((0.5 / np.pi) * theta - surface.quadpoints_theta))
            )
            origin = gamma[phi_index, theta_index]
            axis = normal[phi_index, theta_index]
            ports.add_ports(
                [
                    CircularPort(
                        ox=origin[0],
                        oy=origin[1],
                        oz=origin[2],
                        ax=axis[0],
                        ay=axis[1],
                        az=axis[2],
                        ir=0.1,
                        thick=0.005,
                        l0=-0.15,
                        l1=0.15,
                    )
                ]
            )
    return ports.repeat_via_symmetries(surface.nfp, True)


def _active_segments_clear_ports(
    wireframe: ToroidalWireframe,
    ports: PortSet,
    currents: np.ndarray,
) -> bool:
    active = np.flatnonzero(np.abs(currents) > 0.0)
    positions = np.linspace(0.0, 1.0, 32).reshape((-1, 1, 1))
    point0 = wireframe.nodes[0][wireframe.segments[active, 0]]
    point1 = wireframe.nodes[0][wireframe.segments[active, 1]]
    samples = point0 + positions * (point1 - point0)
    collisions = ports.collides(
        samples[:, :, 0],
        samples[:, :, 1],
        samples[:, :, 2],
        gap=PORT_GAP,
    )
    return not bool(np.any(collisions))


def solve(_output_directory: Path, resolution: int) -> ExampleResult:
    plasma = SurfaceRZFourier.from_vmec_input(
        TEST_DATA / "input.LandremanPaul2021_QA",
        nphi=resolution,
        ntheta=resolution,
        range="half period",
    )
    wireframe_surface = SurfaceRZFourier.from_vmec_input(
        TEST_DATA / "input.LandremanPaul2021_QA"
    )
    wireframe_surface.extend_via_projected_normal(0.3)
    wireframe = ToroidalWireframe(wireframe_surface, 12, 22)
    ports = _ports_on_surface(wireframe_surface)
    wireframe.constrain_colliding_segments(ports.collides, gap=PORT_GAP)
    mu0 = 4.0 * np.pi * 1.0e-7
    poloidal_current = -2.0 * np.pi * plasma.get_rc(0, 0) / mu0
    wireframe.set_poloidal_current(poloidal_current)

    response, target = bnorm_obj_matrices_jax(
        wireframe,
        plasma,
        area_weighted=True,
        verbose=False,
    )
    constraint, constraint_target = wireframe.constraint_matrices(
        assume_no_crossings=False,
        remove_constrained_segments=True,
    )
    constraint_array = np.asarray(constraint, dtype=np.float64)
    constraint_target_array = np.asarray(
        constraint_target,
        dtype=np.float64,
    ).reshape((-1, 1))
    free_segments = np.asarray(wireframe.unconstrained_segments(), dtype=np.intp)
    feasible_free_currents = constraint_array.T @ np.linalg.solve(
        constraint_array @ constraint_array.T,
        constraint_target_array,
    )
    feasible_currents = np.zeros((wireframe.n_segments, 1), dtype=np.float64)
    feasible_currents[free_segments] = feasible_free_currents
    result = rcls_wireframe_jax(
        wireframe,
        response,
        target,
        REGULARIZATION_WEIGHT,
        assume_no_crossings=False,
    )
    solution = np.asarray(jax.device_get(result.x), dtype=np.float64).ravel()
    response_device = jnp.asarray(response, dtype=jnp.float64)
    target_device = jnp.asarray(target, dtype=jnp.float64)
    errors = np.asarray(
        jax.device_get(
            jnp.stack(
                (
                    jnp.linalg.norm(
                        response_device
                        @ jnp.asarray(feasible_currents, dtype=jnp.float64)
                        - target_device
                    ),
                    jnp.linalg.norm(response_device @ result.x - target_device),
                )
            )
        ),
        dtype=np.float64,
    )
    initial_error, final_error = (float(value) for value in errors)
    constraint_error = float(
        np.linalg.norm(
            constraint_array @ solution[free_segments, None] - constraint_target_array,
            ord=np.inf,
        )
    )
    maximum_current = float(np.max(np.abs(solution)))
    wireframe.currents[:] = solution
    ports_clear = _active_segments_clear_ports(wireframe, ports, solution)
    constraint_scale = max(
        1.0,
        float(np.linalg.norm(constraint_target, ord=np.inf)),
    )
    solver_success = bool(
        np.all(np.isfinite(solution))
        and np.isfinite(final_error)
        and final_error < initial_error
        and maximum_current > 0.0
        and constraint_error <= 1.0e-11 * constraint_scale
        and ports_clear
        and wireframe.check_constraints()
    )
    return ExampleResult(
        example_id=EXAMPLE_ID,
        observables={
            "initial_normal_error": initial_error,
            "final_normal_error": final_error,
            "constraint_residual": constraint_error,
            "port_clearance": PORT_GAP,
            "maximum_current": maximum_current,
            "solver_success": solver_success,
        },
        status="ok" if solver_success else "failed",
    )


def main(arguments: list[str] | None = None) -> int:
    return run_example(
        arguments,
        description=__doc__,
        temporary_prefix="simsopt-jax-wireframe-rcls-with-ports-",
        bounded_steps=4,
        native_default_steps=32,
        solve=solve,
    )


if __name__ == "__main__":
    raise SystemExit(main())
