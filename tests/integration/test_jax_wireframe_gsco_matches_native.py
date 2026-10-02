"""The JAX GSCO solvers reproduce native SIMSOPT GSCO on the example workflows.

Three workflows, each at the examples' reduced (``--smoke``) size:

* ``2_Intermediate/wireframe_gsco_modular.py`` and
  ``2_Intermediate/wireframe_gsco_sector_saddle.py``: one
  ``simsopt.solve.wireframe_optimization.gsco_wireframe`` call against
  ``simsopt_jax_adapters.solve.wireframe.gsco_wireframe_jax``;
* ``3_Advanced/wireframe_gsco_multistep.py``: the source script's staged loop
  (GSCO stage, prune small coils, constrain the enclosed segments, halve the
  current, until the currents stop changing, then one matched-current
  adjustment), written out here with native ``gsco_wireframe`` calls, against
  ``simsopt_jax.core.wireframe_workflow.wireframe_gsco_multistep_loop_jax``.

Both lanes of a workflow read the same live operands, and each lane's terminal
label is derived from its own returned arrays by
``simsopt_jax.examples.solver_terminal_status``.
"""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import jax
import numpy as np
import pytest
from simsopt.field import BiotSavart, Current, coils_via_symmetries
from simsopt.geo import (
    SurfaceRZFourier,
    ToroidalWireframe,
    create_equally_spaced_curves,
)
from simsopt.solve.wireframe_optimization import bnorm_obj_matrices, gsco_wireframe
from simsopt_jax.backend.runtime import get_runtime_jax_device
from simsopt_jax.core.wireframe_workflow import (
    WIREFRAME_GSCO_MULTISTEP_STAGE_CAPACITY,
    WireframeGSCOLiveParams,
    wireframe_gsco_multistep_loop_jax,
)
from simsopt_jax.examples.solver_terminal_status import (
    LaneTerminalLabel,
    gsco_multistep_terminal_label,
    gsco_terminal_label,
)
from simsopt_jax_adapters.solve.wireframe import gsco_wireframe_jax

_TEST_FILES = Path(__file__).resolve().parents[1] / "test_files"
_SURFACE_INPUT = _TEST_FILES / "input.LandremanPaul2021_QA"
_WIREFRAME_INPUT = _TEST_FILES / "nescin.LandremanPaul2021_QA"
_MU0 = 4.0 * np.pi * 1.0e-7

#: Reduced sizes of the examples' ``--smoke`` run.
_PLASMA_RESOLUTION = 4
_SINGLE_STAGE_NPHI = 18
_SINGLE_STAGE_NTHETA = 8
_SINGLE_STAGE_MAX_ITERATIONS = 40
_MULTISTEP_NPHI = 24
_MULTISTEP_NTHETA = 8
_MULTISTEP_MAX_ITERATIONS_PER_STEP = 40
_MULTISTEP_MAX_OUTER_STEPS = 4
_MULTISTEP_MINIMUM_COIL_SIZE = 2
_MULTISTEP_TF_COILS = 3
_MULTISTEP_BREAK_WIDTH = 4
_MULTISTEP_INITIAL_CURRENT_FRACTION = 0.2
_MULTISTEP_LAMBDA_S = 1.0e-7
_MULTISTEP_GUARD_RAW_STATUS = "outer_step_guard_exhausted_without_final_adjustment"


@dataclass(frozen=True)
class _SingleStageProblem:
    build_wireframe: Callable[[], ToroidalWireframe]
    plasma: SurfaceRZFourier
    lambda_s: float
    default_current: float
    max_current: float


@dataclass(frozen=True)
class _Lane:
    label: LaneTerminalLabel
    values: dict[str, np.ndarray]


def _plasma() -> SurfaceRZFourier:
    return SurfaceRZFourier.from_vmec_input(
        _SURFACE_INPUT,
        nphi=_PLASMA_RESOLUTION,
        ntheta=_PLASMA_RESOLUTION,
        range="half period",
    )


def _nescoil_wireframe(nphi: int, ntheta: int) -> ToroidalWireframe:
    surface = SurfaceRZFourier.from_nescoil_input(_WIREFRAME_INPUT, "current")
    return ToroidalWireframe(surface, nphi, ntheta)


def _poloidal_current(plasma: SurfaceRZFourier) -> float:
    return float(-2.0 * np.pi * plasma.get_rc(0, 0) / _MU0)


def _modular_problem() -> _SingleStageProblem:
    plasma = _plasma()
    poloidal_current = _poloidal_current(plasma)
    number_of_coils = 6

    def build_wireframe() -> ToroidalWireframe:
        wireframe = _nescoil_wireframe(_SINGLE_STAGE_NPHI, _SINGLE_STAGE_NTHETA)
        coil_current = poloidal_current / (2 * wireframe.nfp * number_of_coils)
        wireframe.add_tfcoil_currents(number_of_coils, coil_current)
        wireframe.set_poloidal_current(poloidal_current)
        return wireframe

    coil_current = abs(poloidal_current / (2 * plasma.nfp * number_of_coils))
    return _SingleStageProblem(
        build_wireframe=build_wireframe,
        plasma=plasma,
        lambda_s=1.0e-6,
        default_current=coil_current,
        max_current=1.1 * coil_current,
    )


def _sector_saddle_problem() -> _SingleStageProblem:
    plasma = _plasma()
    poloidal_current = _poloidal_current(plasma)
    number_of_tf_coils = 3

    def build_wireframe() -> ToroidalWireframe:
        wireframe = _nescoil_wireframe(_SINGLE_STAGE_NPHI, _SINGLE_STAGE_NTHETA)
        tf_current = poloidal_current / (2 * wireframe.nfp * number_of_tf_coils)
        wireframe.add_tfcoil_currents(number_of_tf_coils, tf_current)
        wireframe.set_toroidal_breaks(number_of_tf_coils, 2, allow_pol_current=True)
        wireframe.set_poloidal_current(poloidal_current)
        return wireframe

    current = abs(0.05 * poloidal_current)
    return _SingleStageProblem(
        build_wireframe=build_wireframe,
        plasma=plasma,
        lambda_s=10.0**-6.5,
        default_current=current,
        max_current=1.1 * current,
    )


def _single_stage_values(
    response: np.ndarray,
    target: np.ndarray,
    initial_currents: np.ndarray,
    constrained: np.ndarray,
    *,
    x: np.ndarray,
    loop_count: np.ndarray,
    iter_history: np.ndarray,
    current_history: np.ndarray,
    loop_history: np.ndarray,
    normal_history: np.ndarray,
    sparsity_history: np.ndarray,
    total_history: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        "initial:currents": initial_currents,
        "initial:normal_field_residual": response @ initial_currents - target,
        "initial:normal_objective": np.asarray(normal_history[0]),
        "initial:sparsity_objective": np.asarray(sparsity_history[0]),
        "initial:total_objective": np.asarray(total_history[0]),
        "final:currents": x,
        "final:loop_count": loop_count,
        "final:normal_field_residual": response @ x - target,
        "final:normal_objective": np.asarray(normal_history[-1]),
        "final:sparsity_objective": np.asarray(sparsity_history[-1]),
        "final:total_objective": np.asarray(total_history[-1]),
        "final:iterations": np.asarray(iter_history[-1], dtype=np.int64),
        "final:maximum_current": np.asarray(np.max(np.abs(x))),
        "final:active_segments": np.asarray(np.count_nonzero(x), dtype=np.int64),
        "final:constraints_satisfied": np.asarray(
            np.all(x.reshape(-1)[constrained] == 0.0)
        ),
        "history:iterations": iter_history,
        "history:currents": current_history,
        "history:loops": loop_history,
        "history:normal_objective": normal_history,
        "history:sparsity_objective": sparsity_history,
        "history:total_objective": total_history,
    }


def _single_stage_label(values: dict[str, np.ndarray]) -> LaneTerminalLabel:
    return gsco_terminal_label(
        accepted_updates=int(values["final:iterations"]),
        max_iterations=_SINGLE_STAGE_MAX_ITERATIONS,
        loop_history=values["history:loops"],
        current_history=values["history:currents"],
        endpoint_usable=bool(
            np.all(np.isfinite(values["final:currents"]))
            and bool(values["final:constraints_satisfied"])
        ),
    )


def _single_stage_lanes(
    problem: _SingleStageProblem, monkeypatch: pytest.MonkeyPatch
) -> tuple[_Lane, _Lane]:
    wireframe = problem.build_wireframe()
    response, target = bnorm_obj_matrices(
        wireframe, problem.plasma, area_weighted=True, verbose=False
    )
    # Operands frozen in C order, as the example hands them to both lanes.
    response = np.ascontiguousarray(response, dtype=np.float64)
    target = np.ascontiguousarray(target, dtype=np.float64)
    initial_currents = np.ascontiguousarray(
        np.asarray(wireframe.currents, dtype=np.float64).reshape((-1, 1))
    )
    initial_loop_count = np.zeros(
        len(wireframe.get_free_cells(form="logical")), dtype=np.int64
    )
    constrained = np.asarray(wireframe.constrained_segments(), dtype=np.int64)
    arguments = (
        problem.lambda_s,
        True,
        False,
        problem.default_current,
        problem.max_current,
        _SINGLE_STAGE_MAX_ITERATIONS,
        _SINGLE_STAGE_MAX_ITERATIONS,
    )

    x, loop_count, iterations, currents, loops, normal, sparsity, total, _ = (
        gsco_wireframe(
            problem.build_wireframe(),
            response,
            target,
            *arguments,
            no_new_coils=False,
            max_loop_count=0,
            x_init=initial_currents,
            loop_count_init=initial_loop_count,
            verbose=False,
        )
    )
    native_values = _single_stage_values(
        response,
        target,
        initial_currents,
        constrained,
        x=np.asarray(x, dtype=np.float64),
        loop_count=np.asarray(loop_count, dtype=np.int64),
        iter_history=np.asarray(iterations, dtype=np.int64),
        current_history=np.asarray(currents, dtype=np.float64),
        loop_history=np.asarray(loops, dtype=np.int64),
        normal_history=np.asarray(normal, dtype=np.float64),
        sparsity_history=np.asarray(sparsity, dtype=np.float64),
        total_history=np.asarray(total, dtype=np.float64),
    )

    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_cpu_parity")
    monkeypatch.setenv("SIMSOPT_PRECISION", "fp64")
    monkeypatch.setenv("JAX_ENABLE_X64", "1")
    device = get_runtime_jax_device()
    result = jax.device_get(
        gsco_wireframe_jax(
            problem.build_wireframe(),
            jax.device_put(response, device),
            jax.device_put(target, device),
            *arguments,
            no_new_coils=False,
            max_loop_count=0,
            x_init=jax.device_put(initial_currents, device),
            loop_count_init=jax.device_put(initial_loop_count, device),
            record_every=1,
            verbose=False,
        )
    )
    history = slice(0, int(result.history_length))
    jax_values = _single_stage_values(
        response,
        target,
        initial_currents,
        constrained,
        x=np.asarray(result.x, dtype=np.float64),
        loop_count=np.asarray(result.loop_count, dtype=np.int64),
        iter_history=np.asarray(result.iter_history[history], dtype=np.int64),
        current_history=np.asarray(result.curr_history[history], dtype=np.float64),
        loop_history=np.asarray(result.loop_history[history], dtype=np.int64),
        normal_history=np.asarray(result.f_B_history[history], dtype=np.float64),
        sparsity_history=np.asarray(result.f_S_history[history], dtype=np.float64),
        total_history=np.asarray(result.f_history[history], dtype=np.float64),
    )
    return (
        _Lane(_single_stage_label(native_values), native_values),
        _Lane(_single_stage_label(jax_values), jax_values),
    )


def _multistep_wireframe() -> ToroidalWireframe:
    wireframe = _nescoil_wireframe(_MULTISTEP_NPHI, _MULTISTEP_NTHETA)
    wireframe.set_toroidal_breaks(
        _MULTISTEP_TF_COILS, _MULTISTEP_BREAK_WIDTH, allow_pol_current=True
    )
    wireframe.set_poloidal_current(0.0)
    return wireframe


def _multistep_external_field(
    plasma: SurfaceRZFourier, poloidal_current: float
) -> BiotSavart:
    curves = create_equally_spaced_curves(
        _MULTISTEP_TF_COILS, plasma.nfp, True, R0=1.0, R1=0.85
    )
    currents = [
        Current(-poloidal_current / (2 * _MULTISTEP_TF_COILS * plasma.nfp))
        for _ in range(_MULTISTEP_TF_COILS)
    ]
    return BiotSavart(coils_via_symmetries(curves, currents, plasma.nfp, True))


def _coil_sizes(loop_count: np.ndarray, neighbors: np.ndarray) -> np.ndarray:
    coil_ids = np.full(loop_count.shape[0], -1, dtype=np.int64)
    coil_sizes = np.zeros(loop_count.shape[0], dtype=np.int64)
    coil_id = -1

    def count_connected(index: int, current_id: int) -> int:
        if loop_count[index] == 0 or coil_ids[index] >= 0:
            return 0
        coil_ids[index] = current_id
        return 1 + sum(
            count_connected(int(neighbor), current_id) for neighbor in neighbors[index]
        )

    for index in range(loop_count.shape[0]):
        if loop_count[index] != 0:
            coil_id += 1
            count = count_connected(index, coil_id)
            coil_sizes[coil_ids == coil_id] = count
    return coil_sizes


def _prune_small_coils(
    x: np.ndarray,
    loop_count: np.ndarray,
    loops: np.ndarray,
    neighbors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    sizes = _coil_sizes(loop_count, neighbors)
    small = np.logical_and(sizes > 0, sizes < _MULTISTEP_MINIMUM_COIL_SIZE)
    segments_to_zero = np.unique(loops[small].reshape(-1))
    pruned_x = x.copy()
    pruned_loop_count = loop_count.copy()
    pruned_x[segments_to_zero] = 0.0
    pruned_loop_count[small] = 0
    return pruned_x, pruned_loop_count


def _enclosed_segment_mask(
    x: np.ndarray, loop_count: np.ndarray, loops: np.ndarray
) -> np.ndarray:
    enclosed = np.zeros(x.shape[0], dtype=np.bool_)
    enclosed[np.unique(loops[loop_count != 0].reshape(-1))] = True
    enclosed[x != 0.0] = False
    return enclosed


@dataclass(frozen=True)
class _MultistepInputs:
    poloidal_current: float
    response: np.ndarray
    target: np.ndarray
    initial_currents: np.ndarray
    initial_loop_count: np.ndarray
    loops: np.ndarray
    free_loops: np.ndarray
    segments: np.ndarray
    connections: np.ndarray
    neighbors: np.ndarray
    base_constrained_segments: np.ndarray


def _multistep_inputs() -> _MultistepInputs:
    plasma = _plasma()
    poloidal_current = _poloidal_current(plasma)
    wireframe = _multistep_wireframe()
    response, target = bnorm_obj_matrices(
        wireframe,
        plasma,
        ext_field=_multistep_external_field(plasma, poloidal_current),
        area_weighted=True,
        verbose=False,
    )
    loops = np.asarray(wireframe.get_cell_key(), dtype=np.int32)
    base_constrained = np.zeros(wireframe.n_segments, dtype=np.bool_)
    base_constrained[np.asarray(wireframe.constrained_segments(), dtype=np.int64)] = (
        True
    )
    return _MultistepInputs(
        poloidal_current=poloidal_current,
        response=np.ascontiguousarray(response, dtype=np.float64),
        target=np.ascontiguousarray(target, dtype=np.float64),
        initial_currents=np.zeros((wireframe.n_segments,), dtype=np.float64),
        initial_loop_count=np.zeros(loops.shape[0], dtype=np.int64),
        loops=loops,
        free_loops=np.asarray(wireframe.get_free_cells(form="logical"), dtype=np.int32),
        segments=np.asarray(wireframe.segments, dtype=np.int32),
        connections=np.asarray(wireframe.connected_segments, dtype=np.int32),
        neighbors=np.asarray(wireframe.get_cell_neighbors(), dtype=np.int32),
        base_constrained_segments=base_constrained,
    )


def _multistep_values(
    inputs: _MultistepInputs,
    *,
    x: np.ndarray,
    loop_count: np.ndarray,
    enclosed: np.ndarray,
    enclosed_before_final_adjustment: np.ndarray,
    stage_objectives: np.ndarray,
    stage_iterations: np.ndarray,
    final_objective: float,
    nonfinal_steps: int,
    final_adjustment_run: bool,
) -> dict[str, np.ndarray]:
    initial_residual = -inputs.target.reshape(-1)
    final_residual = (inputs.response @ x.reshape((-1, 1)) - inputs.target).reshape(-1)
    assert stage_iterations.shape == stage_objectives.shape
    return {
        "initial:currents": inputs.initial_currents,
        "initial:normal_field_residual": initial_residual,
        "initial:normal_objective": np.asarray(
            0.5 * np.vdot(initial_residual, initial_residual)
        ),
        "final:currents": x,
        "final:loop_count": loop_count,
        "final:enclosed_segment_mask": enclosed,
        "final:enclosed_segment_mask_before_final_adjustment": (
            enclosed_before_final_adjustment
        ),
        "final:normal_field_residual": final_residual,
        "final:normal_objective": np.asarray(final_objective),
        "final:maximum_current": np.asarray(np.max(np.abs(x))),
        "final:active_segments": np.asarray(np.count_nonzero(x), dtype=np.int64),
        "final:allocated_iteration_budget": np.asarray(
            len(stage_objectives) * _MULTISTEP_MAX_ITERATIONS_PER_STEP,
            dtype=np.int64,
        ),
        "final:nonfinal_steps": np.asarray(nonfinal_steps, dtype=np.int64),
        "final:adjustment_run": np.asarray(final_adjustment_run),
        "final:constraints_satisfied": np.asarray(
            _multistep_wireframe().check_constraints(currents=x)
        ),
        "history:stage_normal_objective": stage_objectives,
        "history:stage_iterations": np.asarray(stage_iterations, dtype=np.int64),
        "history:stage_allocated_iteration_budget": np.full(
            stage_iterations.shape,
            _MULTISTEP_MAX_ITERATIONS_PER_STEP,
            dtype=np.int64,
        ),
    }


def _multistep_label(values: dict[str, np.ndarray]) -> LaneTerminalLabel:
    return gsco_multistep_terminal_label(
        stage_iterations=values["history:stage_iterations"],
        stage_budget=values["history:stage_allocated_iteration_budget"],
        final_adjustment_run=bool(values["final:adjustment_run"]),
        endpoint_usable=bool(
            np.all(np.isfinite(values["final:currents"]))
            and np.all(np.isfinite(values["history:stage_normal_objective"]))
            and bool(values["final:constraints_satisfied"])
        ),
        limit_raw_status=_MULTISTEP_GUARD_RAW_STATUS,
    )


def _multistep_native(inputs: _MultistepInputs) -> dict[str, np.ndarray]:
    current_scale = abs(inputs.poloidal_current)
    fraction = _MULTISTEP_INITIAL_CURRENT_FRACTION
    final_max_current = 1.1 * _MULTISTEP_INITIAL_CURRENT_FRACTION * current_scale
    x = inputs.initial_currents.copy()
    previous_x = np.zeros_like(x)
    loop_count = inputs.initial_loop_count.copy()
    enclosed = np.zeros_like(inputs.base_constrained_segments)
    enclosed_before_final_adjustment = np.zeros_like(enclosed)
    has_previous = False
    final_adjustment_run = False
    nonfinal_steps = 0
    stage_objectives: list[float] = []
    stage_iterations: list[int] = []

    while True:
        final_step = has_previous and np.array_equal(previous_x, x)
        wireframe = _multistep_wireframe()
        if not final_step:
            wireframe.set_segments_constrained(np.flatnonzero(enclosed))
        result = gsco_wireframe(
            wireframe,
            inputs.response,
            inputs.target,
            _MULTISTEP_LAMBDA_S,
            True,
            final_step,
            0.0 if final_step else fraction * current_scale,
            final_max_current if final_step else 1.1 * fraction * current_scale,
            _MULTISTEP_MAX_ITERATIONS_PER_STEP,
            _MULTISTEP_MAX_ITERATIONS_PER_STEP,
            no_new_coils=final_step,
            max_loop_count=1,
            x_init=x,
            loop_count_init=loop_count,
            verbose=False,
        )
        next_x = np.asarray(result[0], dtype=np.float64).reshape(-1)
        next_loop_count = np.asarray(result[1], dtype=np.int64)
        # One history entry per accepted update, holding its own index.
        stage_iterations.append(int(np.asarray(result[2]).reshape(-1)[-1]))
        if final_step:
            x = next_x
            loop_count = next_loop_count
            enclosed_before_final_adjustment = enclosed
            enclosed = np.zeros_like(enclosed)
            final_adjustment_run = True
        else:
            pruned_x, pruned_loop_count = _prune_small_coils(
                next_x, next_loop_count, inputs.loops, inputs.neighbors
            )
            previous_x = x
            x = pruned_x
            loop_count = pruned_loop_count
            enclosed = _enclosed_segment_mask(x, loop_count, inputs.loops)
            fraction *= 0.5
            has_previous = True
            nonfinal_steps += 1
        residual = inputs.response @ x.reshape((-1, 1)) - inputs.target
        stage_objectives.append(float(0.5 * np.vdot(residual, residual)))
        if final_step or len(stage_objectives) >= _MULTISTEP_MAX_OUTER_STEPS:
            break

    return _multistep_values(
        inputs,
        x=x,
        loop_count=loop_count,
        enclosed=enclosed,
        enclosed_before_final_adjustment=enclosed_before_final_adjustment,
        stage_objectives=np.asarray(stage_objectives, dtype=np.float64),
        stage_iterations=np.asarray(stage_iterations, dtype=np.int64),
        final_objective=stage_objectives[-1],
        nonfinal_steps=nonfinal_steps,
        final_adjustment_run=final_adjustment_run,
    )


def _multistep_jax(inputs: _MultistepInputs) -> dict[str, np.ndarray]:
    device = get_runtime_jax_device()
    initial_default_current = _MULTISTEP_INITIAL_CURRENT_FRACTION * abs(
        inputs.poloidal_current
    )
    result = jax.device_get(
        wireframe_gsco_multistep_loop_jax(
            WireframeGSCOLiveParams(
                A=jax.device_put(inputs.response, device),
                loops=jax.device_put(inputs.loops, device),
                free_loops=jax.device_put(inputs.free_loops, device),
                segments=jax.device_put(inputs.segments, device),
                connections=jax.device_put(inputs.connections, device),
                default_current=jax.device_put(initial_default_current, device),
                max_current=jax.device_put(1.1 * initial_default_current, device),
                lambda_s=jax.device_put(_MULTISTEP_LAMBDA_S, device),
                tol=jax.device_put(0.001 * initial_default_current, device),
                max_loop_count=1,
                no_crossing=True,
                no_new_coils=False,
                match_current=False,
            ),
            jax.device_put(inputs.target, device),
            jax.device_put(inputs.initial_currents, device),
            jax.device_put(inputs.initial_loop_count, device),
            jax.device_put(inputs.loops, device),
            jax.device_put(inputs.neighbors, device),
            jax.device_put(inputs.base_constrained_segments, device),
            max_iter_per_step=_MULTISTEP_MAX_ITERATIONS_PER_STEP,
            max_outer_steps=_MULTISTEP_MAX_OUTER_STEPS,
            initial_current_fraction=_MULTISTEP_INITIAL_CURRENT_FRACTION,
            current_scale=abs(inputs.poloidal_current),
            min_coil_size=_MULTISTEP_MINIMUM_COIL_SIZE,
            final_max_current=1.1 * initial_default_current,
            stage_history_capacity=WIREFRAME_GSCO_MULTISTEP_STAGE_CAPACITY,
        )
    )
    stage_count = int(result.stage_count)
    return _multistep_values(
        inputs,
        x=np.asarray(result.x, dtype=np.float64).reshape(-1),
        loop_count=np.asarray(result.loop_count, dtype=np.int64),
        enclosed=np.asarray(result.enclosed_segment_mask, dtype=np.bool_),
        enclosed_before_final_adjustment=np.asarray(
            result.enclosed_segment_mask_before_final_adjustment, dtype=np.bool_
        ),
        stage_objectives=np.asarray(
            result.stage_objectives[:stage_count], dtype=np.float64
        ),
        stage_iterations=np.asarray(
            result.stage_iterations[:stage_count], dtype=np.int64
        ),
        final_objective=float(result.final_objective),
        nonfinal_steps=int(result.nonfinal_steps),
        final_adjustment_run=bool(result.final_adjustment_run),
    )


def _multistep_lanes(monkeypatch: pytest.MonkeyPatch) -> tuple[_Lane, _Lane]:
    inputs = _multistep_inputs()
    native_values = _multistep_native(inputs)
    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_cpu_parity")
    monkeypatch.setenv("SIMSOPT_PRECISION", "fp64")
    monkeypatch.setenv("JAX_ENABLE_X64", "1")
    jax_values = _multistep_jax(inputs)
    return (
        _Lane(_multistep_label(native_values), native_values),
        _Lane(_multistep_label(jax_values), jax_values),
    )


@pytest.mark.parametrize("workflow", ("modular", "sector-saddle", "multistep"))
def test_exact_wireframe_gsco_matches_native_and_jax_cpu(
    workflow: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if workflow == "multistep":
        native, jax_lane = _multistep_lanes(monkeypatch)
    else:
        problem = (
            _modular_problem() if workflow == "modular" else _sector_saddle_problem()
        )
        native, jax_lane = _single_stage_lanes(problem, monkeypatch)

    assert native.label.normalized_status == jax_lane.label.normalized_status
    assert native.label.raw_status == jax_lane.label.raw_status
    assert native.label.success == jax_lane.label.success

    for observable in native.values:
        np.testing.assert_allclose(
            jax_lane.values[observable],
            native.values[observable],
            rtol=1.0e-12,
            atol=1.0e-12,
        )

    if workflow == "multistep":
        stages = int(native.values["final:nonfinal_steps"]) + int(
            native.values["final:adjustment_run"]
        )
        budget = _MULTISTEP_MAX_ITERATIONS_PER_STEP
        assert stages > 0
        assert int(jax_lane.values["final:allocated_iteration_budget"]) == (
            stages * budget
        )
        assert len(jax_lane.values["history:stage_normal_objective"]) == stages
        # One iteration count and one allocated budget per stage that ran, as
        # one array each in the history phase, in the objective series' order.
        iterations = jax_lane.values["history:stage_iterations"]
        allocated = jax_lane.values["history:stage_allocated_iteration_budget"]
        assert iterations.shape == (stages,)
        np.testing.assert_array_equal(allocated, np.full(stages, budget))
        assert bool(np.all((iterations >= 0) & (iterations <= budget)))
    else:
        # At this size the reduced 40-iteration cap is reached, which is
        # upstream's stop_last_iter (wireframe_optimization.cpp) and a budget
        # stop, not convergence.
        assert int(jax_lane.values["final:iterations"]) == (
            _SINGLE_STAGE_MAX_ITERATIONS
        )
        assert native.label.raw_status == "gsco_maximum_iteration_reached"
        assert native.label.normalized_status == "budget_exhausted"
        assert native.label.success is False
    # Reported diagnostic, never a success gate.
    assert float(jax_lane.values["final:normal_objective"]) < float(
        jax_lane.values["initial:normal_objective"]
    )
