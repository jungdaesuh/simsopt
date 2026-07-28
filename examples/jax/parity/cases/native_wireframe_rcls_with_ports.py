"""Exact matched workflow for ``2_Intermediate/wireframe_rcls_with_ports.py``."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import numpy as np
from examples.jax.parity.arbiter import LaneObservation
from examples.jax.parity.cases.native_wireframe_rcls_basic import (
    _effective_fingerprint,
    _minimum_norm_feasible_currents,
    execute_problem,
)
from examples.jax.parity.input_bundle import InputBundle, create_input_bundle
from examples.jax.parity.runtime import ParityLane
from simsopt_jax.examples import ExecutionScale

TEST_DATA = Path(__file__).resolve().parents[4] / "tests" / "test_files"
SURFACE_INPUT = TEST_DATA / "input.LandremanPaul2021_QA"
PORT_GAP = 0.04


def _scale_configuration(scale: ExecutionScale) -> dict[str, object]:
    return {
        "plasma_nphi": 32 if scale == "native_default" else 16,
        "plasma_ntheta": 32 if scale == "native_default" else 16,
        "wireframe_nphi": 12,
        "wireframe_ntheta": 22,
        "wireframe_surface_distance": 0.3,
        "field_on_axis": 1.0,
        "regularization_weight": 1.0e-10,
        "assume_no_crossings": False,
        "port_gap": PORT_GAP,
        "surface_input_sha256": hashlib.sha256(SURFACE_INPUT.read_bytes()).hexdigest(),
    }


def _ports_on_surface(surface):
    from simsopt.geo import CircularPort, PortSet

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


def _build_geometry(configuration: Mapping[str, object]):
    from simsopt.geo import SurfaceRZFourier, ToroidalWireframe

    plasma_nphi = configuration["plasma_nphi"]
    plasma_ntheta = configuration["plasma_ntheta"]
    wireframe_nphi = configuration["wireframe_nphi"]
    wireframe_ntheta = configuration["wireframe_ntheta"]
    assert isinstance(plasma_nphi, int)
    assert isinstance(plasma_ntheta, int)
    assert isinstance(wireframe_nphi, int)
    assert isinstance(wireframe_ntheta, int)
    plasma = SurfaceRZFourier.from_vmec_input(
        str(SURFACE_INPUT),
        nphi=plasma_nphi,
        ntheta=plasma_ntheta,
        range="half period",
    )
    wireframe_surface = SurfaceRZFourier.from_vmec_input(str(SURFACE_INPUT))
    wireframe_surface.extend_via_projected_normal(
        float(configuration["wireframe_surface_distance"])
    )
    wireframe = ToroidalWireframe(
        wireframe_surface,
        wireframe_nphi,
        wireframe_ntheta,
    )
    ports = _ports_on_surface(wireframe_surface)
    wireframe.constrain_colliding_segments(
        ports.collides,
        gap=float(configuration["port_gap"]),
    )
    mu0 = 4.0 * np.pi * 1.0e-7
    poloidal_current = (
        -2.0
        * np.pi
        * plasma.get_rc(0, 0)
        * float(configuration["field_on_axis"])
        / mu0
    )
    wireframe.set_poloidal_current(poloidal_current)
    return plasma, wireframe


def create_input(root: Path, scale: ExecutionScale) -> InputBundle:
    """Freeze the port geometry, constrained response, and feasible state."""
    from simsopt.solve.wireframe_optimization import bnorm_obj_matrices

    configuration = _scale_configuration(scale)
    plasma, wireframe = _build_geometry(configuration)
    response, target = bnorm_obj_matrices(
        wireframe,
        plasma,
        area_weighted=True,
        verbose=False,
    )
    initial_currents, constraint, constraint_target, free_segments = (
        _minimum_norm_feasible_currents(wireframe)
    )
    normal = np.asarray(plasma.normal(), dtype=np.float64)
    constrained_segments = np.asarray(
        wireframe.constrained_segments(),
        dtype=np.int64,
    )
    arrays = {
        "response_matrix": np.array(response, dtype=np.float64, copy=True),
        "target": np.array(target, dtype=np.float64, copy=True),
        "constraint_matrix": np.array(constraint, dtype=np.float64, copy=True),
        "constraint_target": np.array(
            constraint_target,
            dtype=np.float64,
            copy=True,
        ),
        "free_segments": np.array(free_segments, dtype=np.int64, copy=True),
        "constrained_segments": constrained_segments,
        "initial_currents": np.array(initial_currents, dtype=np.float64, copy=True),
        "plasma_points": np.array(
            plasma.gamma().reshape((-1, 3)),
            dtype=np.float64,
            copy=True,
        ),
        "plasma_unit_normal": np.array(
            plasma.unitnormal().reshape((-1, 3)),
            dtype=np.float64,
            copy=True,
        ),
        "plasma_area_weights": np.array(
            np.linalg.norm(normal, axis=2).reshape(-1)
            / normal.shape[0]
            / normal.shape[1],
            dtype=np.float64,
            copy=True,
        ),
        "wireframe_nodes": np.array(
            np.stack(wireframe.nodes),
            dtype=np.float64,
            copy=True,
        ),
        "wireframe_segments": np.array(
            wireframe.segments,
            dtype=np.int32,
            copy=True,
        ),
        "wireframe_segment_signs": np.array(
            wireframe.seg_signs,
            dtype=np.float64,
            copy=True,
        ),
    }
    return create_input_bundle(
        root,
        case_id="native-wireframe-rcls-with-ports",
        random_seed=0,
        arrays=arrays,
        configuration={
            **configuration,
            "n_segments": int(wireframe.n_segments),
            "n_constrained_segments": int(constrained_segments.size),
            "degrees_of_freedom": int(wireframe.n_segments - constraint.shape[0]),
        },
        scale=scale,
    )


def execute(
    lane: ParityLane,
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
) -> LaneObservation:
    """Execute the exact port-constrained RCLS workflow in one lane."""
    observation = execute_problem(lane, bundle, arrays, _build_geometry)
    final_currents = np.asarray(
        observation.values["final:currents"],
        dtype=np.float64,
    ).reshape(-1)
    constrained_segments = arrays["constrained_segments"]
    port_constraints_satisfied = bool(
        np.all(final_currents[constrained_segments] == 0.0)
    )
    values = {
        **observation.values,
        "construction:constrained_segments": constrained_segments,
        "final:port_constraints_satisfied": np.asarray(
            port_constraints_satisfied
        ),
    }
    success = bool(observation.success and port_constraints_satisfied)
    return replace(
        observation,
        effective_construction_fingerprint=_effective_fingerprint(bundle, arrays),
        normalized_status="converged" if success else "failed",
        success=success,
        values=values,
    )
