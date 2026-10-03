"""``solve_wireframe_rcls`` reproduces native SIMSOPT RCLS on the example workflows.

Both lanes start from the same live inputs: the plasma surface, the offset
wireframe, its area-weighted normal-field response (``bnorm_obj_matrices``) and
the minimum-norm feasible initial currents. The native lane is
``simsopt.solve.wireframe_optimization.rcls_wireframe`` followed by
``WireframeField``; the JAX lane is ``simsopt_jax.examples.solve_wireframe_rcls``
on the same arrays. One test covers ``2_Intermediate/wireframe_rcls_basic.py``,
the other ``2_Intermediate/wireframe_rcls_with_ports.py``, each at its reduced
plasma resolution.
"""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

from dataclasses import dataclass
from pathlib import Path

import jax
import numpy as np
import pytest
from simsopt.field import WireframeField
from simsopt.geo import CircularPort, PortSet, SurfaceRZFourier, ToroidalWireframe
from simsopt.solve.wireframe_optimization import bnorm_obj_matrices, rcls_wireframe
from simsopt_jax.backend.runtime import get_runtime_jax_device
from simsopt_jax.examples import solve_wireframe_rcls

_SURFACE_INPUT = (
    Path(__file__).resolve().parents[1] / "test_files" / "input.LandremanPaul2021_QA"
)
_PLASMA_RESOLUTION = 16
_WIREFRAME_SURFACE_DISTANCE = 0.3
_REGULARIZATION_WEIGHT = 1.0e-10
_ASSUME_NO_CROSSINGS = False
_PORT_GAP = 0.04

#: Relative floor for physical feasibility of ``C x - b`` in either phase, in
#: units of ``max(1, ||b||_inf)``: the success predicate the workflow applies
#: to the initial and the final state of each lane.
_CONSTRAINT_FEASIBILITY_RELATIVE_LIMIT = 1.0e-11

_STATE_OBSERVABLES = (
    "currents",
    "normal_field_residual",
    "normal_objective",
    "regularization_objective",
    "total_objective",
    "constraint_residual",
)


@dataclass(frozen=True)
class _Inputs:
    plasma: SurfaceRZFourier
    response: np.ndarray
    target: np.ndarray
    constraint_matrix: np.ndarray
    constraint_target: np.ndarray
    free_segments: np.ndarray
    initial_currents: np.ndarray
    plasma_points: np.ndarray
    plasma_unit_normal: np.ndarray
    plasma_area_weights: np.ndarray


def _wireframe_surface() -> SurfaceRZFourier:
    surface = SurfaceRZFourier.from_vmec_input(str(_SURFACE_INPUT))
    surface.extend_via_projected_normal(_WIREFRAME_SURFACE_DISTANCE)
    return surface


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


def _build_geometry(
    *, nphi: int, ntheta: int, with_ports: bool
) -> tuple[SurfaceRZFourier, ToroidalWireframe]:
    plasma = SurfaceRZFourier.from_vmec_input(
        str(_SURFACE_INPUT),
        nphi=_PLASMA_RESOLUTION,
        ntheta=_PLASMA_RESOLUTION,
        range="half period",
    )
    wireframe_surface = _wireframe_surface()
    wireframe = ToroidalWireframe(wireframe_surface, nphi, ntheta)
    if with_ports:
        wireframe.constrain_colliding_segments(
            _ports_on_surface(wireframe_surface).collides,
            gap=_PORT_GAP,
        )
    mu0 = 4.0 * np.pi * 1.0e-7
    wireframe.set_poloidal_current(-2.0 * np.pi * plasma.get_rc(0, 0) / mu0)
    return plasma, wireframe


def _inputs(plasma: SurfaceRZFourier, wireframe: ToroidalWireframe) -> _Inputs:
    """Build the shared operands, then freeze each one in C order.

    The workflow derives the minimum-norm initial currents from the matrices
    exactly as ``constraint_matrices`` returns them (Fortran order) and only
    then hands both lanes C-ordered copies. The layout fixes the summation
    order of the native lane's dense products, and the initial-state
    comparison below is tighter than a change of that order.
    """
    response, target = bnorm_obj_matrices(
        wireframe, plasma, area_weighted=True, verbose=False
    )
    constraint, constraint_target = wireframe.constraint_matrices(
        assume_no_crossings=_ASSUME_NO_CROSSINGS,
        remove_constrained_segments=True,
    )
    constraint_matrix = np.asarray(constraint, dtype=np.float64)
    constraint_target_array = np.asarray(constraint_target, dtype=np.float64).reshape(
        (-1, 1)
    )
    free_segments = np.asarray(wireframe.unconstrained_segments(), dtype=np.int64)
    initial_currents = np.zeros((wireframe.n_segments, 1), dtype=np.float64)
    initial_currents[free_segments] = constraint_matrix.T @ np.linalg.solve(
        constraint_matrix @ constraint_matrix.T, constraint_target_array
    )
    normal = np.asarray(plasma.normal(), dtype=np.float64)
    return _Inputs(
        plasma=plasma,
        response=np.ascontiguousarray(response, dtype=np.float64),
        target=np.ascontiguousarray(target, dtype=np.float64),
        constraint_matrix=np.ascontiguousarray(constraint_matrix),
        constraint_target=np.ascontiguousarray(constraint_target_array),
        free_segments=np.ascontiguousarray(free_segments),
        initial_currents=np.ascontiguousarray(initial_currents),
        plasma_points=np.ascontiguousarray(plasma.gamma().reshape((-1, 3))),
        plasma_unit_normal=np.ascontiguousarray(plasma.unitnormal().reshape((-1, 3))),
        plasma_area_weights=np.ascontiguousarray(
            np.linalg.norm(normal, axis=2).reshape(-1)
            / normal.shape[0]
            / normal.shape[1]
        ),
    )


def _constraint_feasible(residual: np.ndarray, target: np.ndarray) -> bool:
    limit = _CONSTRAINT_FEASIBILITY_RELATIVE_LIMIT * max(
        1.0, float(np.linalg.norm(target, ord=np.inf))
    )
    return bool(
        np.all(np.isfinite(residual)) and np.linalg.norm(residual, ord=np.inf) <= limit
    )


def _native_state(
    prefix: str, inputs: _Inputs, currents: np.ndarray
) -> dict[str, np.ndarray]:
    residual = inputs.response @ currents - inputs.target
    normal_objective = 0.5 * np.vdot(residual, residual)
    regularization_objective = (
        0.5 * _REGULARIZATION_WEIGHT**2 * np.vdot(currents, currents)
    )
    return {
        f"{prefix}:currents": currents,
        f"{prefix}:normal_field_residual": residual,
        f"{prefix}:normal_objective": np.asarray(normal_objective),
        f"{prefix}:regularization_objective": np.asarray(regularization_objective),
        f"{prefix}:total_objective": np.asarray(
            normal_objective + regularization_objective
        ),
        f"{prefix}:constraint_residual": (
            inputs.constraint_matrix @ currents[inputs.free_segments]
            - inputs.constraint_target
        ),
    }


def _native_lane(
    wireframe: ToroidalWireframe, inputs: _Inputs
) -> tuple[dict[str, np.ndarray], bool]:
    final, _, _, _ = rcls_wireframe(
        wireframe,
        inputs.response,
        inputs.target,
        _REGULARIZATION_WEIGHT,
        _ASSUME_NO_CROSSINGS,
        False,
    )
    final_currents = np.asarray(final, dtype=np.float64)
    wireframe.currents[:] = final_currents.reshape(-1)
    field = WireframeField(wireframe)
    field.set_points(inputs.plasma_points)
    magnetic_field = np.asarray(field.B(), dtype=np.float64)
    normal_field = np.sum(magnetic_field * inputs.plasma_unit_normal, axis=1)
    area = inputs.plasma_area_weights
    values = {
        **_native_state("initial", inputs, inputs.initial_currents),
        **_native_state("final", inputs, final_currents),
        "final:magnetic_field": magnetic_field,
        "final:normal_field": normal_field,
        "final:mean_relative_normal_field": np.asarray(
            np.sum(np.abs(normal_field / np.linalg.norm(magnetic_field, axis=1)) * area)
            / np.sum(area)
        ),
        "final:maximum_current": np.asarray(np.max(np.abs(final_currents))),
    }
    success = bool(
        np.all(np.isfinite(final_currents))
        and _constraint_feasible(
            values["initial:constraint_residual"], inputs.constraint_target
        )
        and _constraint_feasible(
            values["final:constraint_residual"], inputs.constraint_target
        )
        and values["final:normal_objective"] < values["initial:normal_objective"]
        and wireframe.check_constraints()
    )
    return values, success


def _jax_lane(
    wireframe: ToroidalWireframe, inputs: _Inputs
) -> tuple[dict[str, np.ndarray], bool]:
    device = get_runtime_jax_device()
    result = jax.device_get(
        solve_wireframe_rcls(
            wireframe=wireframe,
            response=jax.device_put(inputs.response, device),
            target=jax.device_put(inputs.target, device),
            regularization=_REGULARIZATION_WEIGHT,
            initial_currents=jax.device_put(inputs.initial_currents, device),
            plasma_points=jax.device_put(inputs.plasma_points, device),
            plasma_unit_normal=jax.device_put(inputs.plasma_unit_normal, device),
            plasma_area_weights=jax.device_put(inputs.plasma_area_weights, device),
            wireframe_nodes=jax.device_put(np.stack(wireframe.nodes), device),
            wireframe_segments=jax.device_put(
                np.asarray(wireframe.segments, dtype=np.int32), device
            ),
            wireframe_segment_signs=jax.device_put(
                np.asarray(wireframe.seg_signs, dtype=np.float64), device
            ),
            assume_no_crossings=_ASSUME_NO_CROSSINGS,
        )
    )
    values = {
        f"{prefix}:{name}": np.asarray(getattr(state, name), dtype=np.float64)
        for prefix, state in (("initial", result.initial), ("final", result.final))
        for name in _STATE_OBSERVABLES
    }
    values.update(
        {
            "final:magnetic_field": np.asarray(result.magnetic_field),
            "final:normal_field": np.asarray(result.normal_field),
            "final:mean_relative_normal_field": np.asarray(
                result.mean_relative_normal_field
            ),
            "final:maximum_current": np.asarray(result.maximum_current),
        }
    )
    success = bool(
        result.finite_currents
        and _constraint_feasible(
            values["initial:constraint_residual"], inputs.constraint_target
        )
        and _constraint_feasible(
            values["final:constraint_residual"], inputs.constraint_target
        )
        and values["final:normal_objective"] < values["initial:normal_objective"]
    )
    return values, success


def _enable_jax_cpu_parity_lane(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_cpu_parity")
    monkeypatch.setenv("SIMSOPT_PRECISION", "fp64")
    monkeypatch.setenv("JAX_ENABLE_X64", "1")


def test_wireframe_rcls_basic_matches_native(monkeypatch: pytest.MonkeyPatch) -> None:
    plasma, native_wireframe = _build_geometry(nphi=4, ntheta=6, with_ports=False)
    inputs = _inputs(plasma, native_wireframe)
    native, native_success = _native_lane(native_wireframe, inputs)

    _enable_jax_cpu_parity_lane(monkeypatch)
    _, jax_wireframe = _build_geometry(nphi=4, ntheta=6, with_ports=False)
    jax_values, jax_success = _jax_lane(jax_wireframe, inputs)

    assert native_success is True
    assert jax_success is True
    assert set(native) == set(jax_values)

    for phase in ("initial", "final"):
        for observable in _STATE_OBSERVABLES:
            np.testing.assert_allclose(
                jax_values[f"{phase}:{observable}"],
                native[f"{phase}:{observable}"],
                rtol=1.0e-11,
                atol=1.0e-7,
            )

    for observable in (
        "final:magnetic_field",
        "final:normal_field",
        "final:mean_relative_normal_field",
        "final:maximum_current",
    ):
        np.testing.assert_allclose(
            jax_values[observable],
            native[observable],
            rtol=1.0e-11,
            atol=1.0e-12,
        )

    assert inputs.response.shape == (256, 48)
    assert native["final:magnetic_field"].shape == (256, 3)
    assert float(native["final:mean_relative_normal_field"]) < 0.04
    assert float(native["final:normal_objective"]) < float(
        native["initial:normal_objective"]
    )


def test_wireframe_rcls_with_ports_matches_native(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plasma, native_wireframe = _build_geometry(nphi=12, ntheta=22, with_ports=True)
    inputs = _inputs(plasma, native_wireframe)
    constrained = np.asarray(native_wireframe.constrained_segments(), dtype=np.int64)
    native, native_success = _native_lane(native_wireframe, inputs)
    native_ports_satisfied = bool(
        np.all(native["final:currents"].reshape(-1)[constrained] == 0.0)
    )

    _enable_jax_cpu_parity_lane(monkeypatch)
    _, jax_wireframe = _build_geometry(nphi=12, ntheta=22, with_ports=True)
    jax_values, jax_success = _jax_lane(jax_wireframe, inputs)
    jax_final_currents = jax_values["final:currents"].reshape(-1)
    jax_ports_satisfied = bool(np.all(jax_final_currents[constrained] == 0.0))

    assert native_success and native_ports_satisfied
    assert jax_success and jax_ports_satisfied
    assert set(native) == set(jax_values)

    for observable in (f"initial:{name}" for name in _STATE_OBSERVABLES):
        np.testing.assert_allclose(
            jax_values[observable],
            native[observable],
            rtol=1.0e-13,
            atol=1.0e-12,
        )

    # The constrained system is underdetermined. Its two direct solvers may
    # choose slightly different current vectors while agreeing much more
    # tightly on the physical field and minimized objective.
    np.testing.assert_allclose(
        jax_values["final:currents"],
        native["final:currents"],
        rtol=5.0e-7,
        atol=1.0e-2,
    )
    for observable in (
        "final:normal_field_residual",
        "final:normal_objective",
        "final:regularization_objective",
        "final:total_objective",
        "final:magnetic_field",
        "final:normal_field",
        "final:mean_relative_normal_field",
        "final:maximum_current",
    ):
        np.testing.assert_allclose(
            jax_values[observable],
            native[observable],
            rtol=1.0e-9,
            atol=1.0e-11,
        )
    np.testing.assert_allclose(
        jax_values["final:constraint_residual"],
        native["final:constraint_residual"],
        rtol=0.0,
        atol=1.0e-8,
    )
    assert jax_ports_satisfied == native_ports_satisfied

    np.testing.assert_array_equal(jax_final_currents[constrained], 0.0)
    assert constrained.size > 0
    assert float(jax_values["final:normal_objective"]) < float(
        jax_values["initial:normal_objective"]
    )
