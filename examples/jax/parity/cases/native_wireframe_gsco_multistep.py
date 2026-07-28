"""Exact matched workflow for ``3_Advanced/wireframe_gsco_multistep.py``."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from examples.jax.parity.arbiter import LaneObservation
from examples.jax.parity.cases.native_wireframe_rcls_basic import (
    _effective_fingerprint,
)
from examples.jax.parity.input_bundle import InputBundle, create_input_bundle
from examples.jax.parity.runtime import ParityLane
from simsopt.geo import SurfaceRZFourier, ToroidalWireframe
from simsopt_jax.examples import ExecutionScale

TEST_DATA = Path(__file__).resolve().parents[4] / "tests" / "test_files"
SURFACE_INPUT = TEST_DATA / "input.LandremanPaul2021_QA"
WIREFRAME_INPUT = TEST_DATA / "nescin.LandremanPaul2021_QA"

WORKFLOW_STAGES = (
    "construct_qa_plasma_sector_wireframe_and_external_tf_coils",
    "construct_area_weighted_combined_normal_field_response",
    "initialize_multistep_gsco_state",
    "run_fixed_budget_gsco_stages",
    "prune_small_coils_and_constrain_enclosed_segments",
    "evaluate_final_currents_objective_and_constraints",
)


def _configuration(scale: ExecutionScale) -> dict[str, object]:
    native = scale == "native_default"
    return {
        "plasma_resolution": 32 if native else 4,
        "wireframe_nphi": 96 if native else 24,
        "wireframe_ntheta": 100 if native else 8,
        "max_iterations_per_step": 2_500 if native else 40,
        "max_outer_steps": 12 if native else 4,
        "number_of_tf_coils": 3,
        "break_width": 4,
        "initial_current_fraction": 0.2,
        "minimum_coil_size": 20 if native else 2,
        "field_on_axis": 1.0,
        "lambda_s": 1.0e-7,
        "surface_input_sha256": hashlib.sha256(SURFACE_INPUT.read_bytes()).hexdigest(),
        "wireframe_input_sha256": hashlib.sha256(
            WIREFRAME_INPUT.read_bytes()
        ).hexdigest(),
    }


def _configuration_int(bundle: InputBundle, name: str) -> int:
    value = bundle.configuration[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"configuration {name} must be an integer")
    return value


def _configuration_float(bundle: InputBundle, name: str) -> float:
    value = bundle.configuration[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"configuration {name} must be numeric")
    return float(value)


def _build_wireframe(
    configuration: Mapping[str, object],
) -> tuple[ToroidalWireframe, SurfaceRZFourier, float]:
    resolution = configuration["plasma_resolution"]
    nphi = configuration["wireframe_nphi"]
    ntheta = configuration["wireframe_ntheta"]
    number_of_tf_coils = configuration["number_of_tf_coils"]
    break_width = configuration["break_width"]
    assert isinstance(resolution, int)
    assert isinstance(nphi, int)
    assert isinstance(ntheta, int)
    assert isinstance(number_of_tf_coils, int)
    assert isinstance(break_width, int)
    plasma = SurfaceRZFourier.from_vmec_input(
        SURFACE_INPUT,
        nphi=resolution,
        ntheta=resolution,
        range="half period",
    )
    wireframe_surface = SurfaceRZFourier.from_nescoil_input(
        WIREFRAME_INPUT,
        "current",
    )
    wireframe = ToroidalWireframe(wireframe_surface, nphi, ntheta)
    mu0 = 4.0 * np.pi * 1.0e-7
    poloidal_current = (
        -2.0 * np.pi * plasma.get_rc(0, 0) * float(configuration["field_on_axis"]) / mu0
    )
    wireframe.set_toroidal_breaks(
        number_of_tf_coils,
        break_width,
        allow_pol_current=True,
    )
    wireframe.set_poloidal_current(0.0)
    return wireframe, plasma, poloidal_current


def _native_external_field(
    plasma: SurfaceRZFourier,
    poloidal_current: float,
    number_of_tf_coils: int,
):
    from simsopt.field import BiotSavart, Current, coils_via_symmetries
    from simsopt.geo import create_equally_spaced_curves

    curves = create_equally_spaced_curves(
        number_of_tf_coils,
        plasma.nfp,
        True,
        R0=1.0,
        R1=0.85,
    )
    currents = [
        Current(-poloidal_current / (2 * number_of_tf_coils * plasma.nfp))
        for _ in range(number_of_tf_coils)
    ]
    return BiotSavart(
        coils_via_symmetries(
            curves,
            currents,
            plasma.nfp,
            True,
        )
    )


def create_input(root: Path, scale: ExecutionScale) -> InputBundle:
    """Freeze the exact multistep response, topology, and initial state."""
    from simsopt.solve.wireframe_optimization import bnorm_obj_matrices

    configuration = _configuration(scale)
    wireframe, plasma, poloidal_current = _build_wireframe(configuration)
    number_of_tf_coils = configuration["number_of_tf_coils"]
    assert isinstance(number_of_tf_coils, int)
    external_field = _native_external_field(
        plasma,
        poloidal_current,
        number_of_tf_coils,
    )
    response, target = bnorm_obj_matrices(
        wireframe,
        plasma,
        ext_field=external_field,
        area_weighted=True,
        verbose=False,
    )
    loops = np.asarray(wireframe.get_cell_key(), dtype=np.int32)
    base_constrained = np.zeros(wireframe.n_segments, dtype=np.bool_)
    base_constrained[np.asarray(wireframe.constrained_segments(), dtype=np.int64)] = (
        True
    )
    initial_current = float(configuration["initial_current_fraction"]) * abs(
        poloidal_current
    )
    return create_input_bundle(
        root,
        case_id="native-wireframe-gsco-multistep",
        random_seed=0,
        arrays={
            "response_matrix": np.array(response, dtype=np.float64, copy=True),
            "target": np.array(target, dtype=np.float64, copy=True),
            "initial_currents": np.zeros(
                (wireframe.n_segments,),
                dtype=np.float64,
            ),
            "initial_loop_count": np.zeros(
                loops.shape[0],
                dtype=np.int64,
            ),
            "loops": loops,
            "free_loops": np.asarray(
                wireframe.get_free_cells(form="logical"),
                dtype=np.int32,
            ),
            "segments": np.asarray(wireframe.segments, dtype=np.int32),
            "connections": np.asarray(
                wireframe.connected_segments,
                dtype=np.int32,
            ),
            "neighbors": np.asarray(
                wireframe.get_cell_neighbors(),
                dtype=np.int32,
            ),
            "base_constrained_segments": base_constrained,
        },
        configuration={
            **configuration,
            "poloidal_current": poloidal_current,
            "initial_default_current": initial_current,
            "final_max_current": 1.1 * initial_current,
            "n_segments": int(wireframe.n_segments),
        },
        scale=scale,
    )


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
    *,
    minimum_coil_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    sizes = _coil_sizes(loop_count, neighbors)
    small = np.logical_and(sizes > 0, sizes < minimum_coil_size)
    segments_to_zero = np.unique(loops[small].reshape(-1))
    pruned_x = x.copy()
    pruned_loop_count = loop_count.copy()
    pruned_x[segments_to_zero] = 0.0
    pruned_loop_count[small] = 0
    return pruned_x, pruned_loop_count


def _enclosed_segment_mask(
    x: np.ndarray,
    loop_count: np.ndarray,
    loops: np.ndarray,
) -> np.ndarray:
    enclosed = np.zeros(x.shape[0], dtype=np.bool_)
    enclosed[np.unique(loops[loop_count != 0].reshape(-1))] = True
    enclosed[x != 0.0] = False
    return enclosed


def _values(
    arrays: dict[str, np.ndarray],
    *,
    x: np.ndarray,
    loop_count: np.ndarray,
    enclosed: np.ndarray,
    stage_objectives: np.ndarray,
    nonfinal_steps: int,
    final_adjustment_run: bool,
    max_iterations_per_step: int,
) -> dict[str, np.ndarray]:
    initial_residual = -arrays["target"].reshape(-1)
    final_residual = (
        arrays["response_matrix"] @ x.reshape((-1, 1)) - arrays["target"]
    ).reshape(-1)
    base_constrained = arrays["base_constrained_segments"]
    constraints_satisfied = np.all(x[base_constrained] == 0.0)
    return {
        **{
            f"construction:{name}": value
            for name, value in arrays.items()
            if name not in {"initial_currents", "initial_loop_count"}
        },
        "initial:currents": arrays["initial_currents"],
        "initial:normal_field_residual": initial_residual,
        "initial:normal_objective": np.asarray(
            0.5 * np.vdot(initial_residual, initial_residual),
            dtype=np.float64,
        ),
        "final:currents": x,
        "final:loop_count": loop_count,
        "final:enclosed_segment_mask": enclosed,
        "final:normal_field_residual": final_residual,
        "final:normal_objective": np.asarray(
            stage_objectives[-1],
            dtype=np.float64,
        ),
        "final:maximum_current": np.asarray(
            np.max(np.abs(x)),
            dtype=np.float64,
        ),
        "final:active_segments": np.asarray(
            np.count_nonzero(x),
            dtype=np.int64,
        ),
        "final:iterations": np.asarray(
            nonfinal_steps * max_iterations_per_step,
            dtype=np.int64,
        ),
        "final:nonfinal_steps": np.asarray(nonfinal_steps, dtype=np.int64),
        "final:adjustment_run": np.asarray(final_adjustment_run),
        "final:constraints_satisfied": np.asarray(constraints_satisfied),
        "history:stage_normal_objective": stage_objectives,
    }


def _observation(
    lane: str,
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
    *,
    backend_mode: str,
    platform: str,
    driver: str,
    values: dict[str, np.ndarray],
) -> LaneObservation:
    success = bool(
        np.all(np.isfinite(values["final:currents"]))
        and np.all(np.isfinite(values["history:stage_normal_objective"]))
        and values["final:normal_objective"] < values["initial:normal_objective"]
        and bool(values["final:constraints_satisfied"])
    )
    return LaneObservation(
        lane=lane,
        backend_mode=backend_mode,
        platform=platform,
        precision="fp64",
        scale=bundle.scale,
        input_fingerprint=bundle.input_fingerprint,
        configuration_fingerprint=bundle.configuration_fingerprint,
        effective_construction_fingerprint=_effective_fingerprint(bundle, arrays),
        driver=driver,
        normalized_status="converged" if success else "failed",
        raw_status="bounded_multistep_gsco_complete",
        success=success,
        nit=int(values["final:iterations"]),
        nfev=int(values["final:iterations"]),
        njev=None,
        completed_workflow_stages=WORKFLOW_STAGES,
        provenance=None,
        values=values,
    )


def _native(
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
) -> LaneObservation:
    from simsopt.solve.wireframe_optimization import gsco_wireframe

    max_iterations = _configuration_int(bundle, "max_iterations_per_step")
    maximum_outer_steps = _configuration_int(bundle, "max_outer_steps")
    minimum_coil_size = _configuration_int(bundle, "minimum_coil_size")
    current_scale = abs(_configuration_float(bundle, "poloidal_current"))
    fraction = _configuration_float(bundle, "initial_current_fraction")
    x = arrays["initial_currents"].copy()
    previous_x = np.zeros_like(x)
    loop_count = arrays["initial_loop_count"].copy()
    enclosed = np.zeros_like(arrays["base_constrained_segments"])
    has_previous = False
    final_adjustment_run = False
    nonfinal_steps = 0
    stage_objectives: list[float] = []

    for _ in range(maximum_outer_steps):
        final_step = has_previous and np.array_equal(previous_x, x)
        wireframe, _, _ = _build_wireframe(bundle.configuration)
        if not final_step:
            wireframe.set_segments_constrained(np.flatnonzero(enclosed))
        result = gsco_wireframe(
            wireframe,
            arrays["response_matrix"],
            arrays["target"],
            _configuration_float(bundle, "lambda_s"),
            True,
            final_step,
            0.0 if final_step else fraction * current_scale,
            (
                _configuration_float(bundle, "final_max_current")
                if final_step
                else 1.1 * fraction * current_scale
            ),
            max_iterations,
            max_iterations,
            no_new_coils=final_step,
            max_loop_count=1,
            x_init=x,
            loop_count_init=loop_count,
            verbose=False,
        )
        next_x = np.asarray(result[0], dtype=np.float64).reshape(-1)
        next_loop_count = np.asarray(result[1], dtype=np.int64)
        if final_step:
            x = next_x
            loop_count = next_loop_count
            enclosed = np.zeros_like(enclosed)
            final_adjustment_run = True
        else:
            pruned_x, pruned_loop_count = _prune_small_coils(
                next_x,
                next_loop_count,
                arrays["loops"],
                arrays["neighbors"],
                minimum_coil_size=minimum_coil_size,
            )
            previous_x = x
            x = pruned_x
            loop_count = pruned_loop_count
            enclosed = _enclosed_segment_mask(
                x,
                loop_count,
                arrays["loops"],
            )
            fraction *= 0.5
            has_previous = True
            nonfinal_steps += 1
        residual = arrays["response_matrix"] @ x.reshape((-1, 1)) - arrays["target"]
        stage_objectives.append(float(0.5 * np.vdot(residual, residual)))
        if final_step:
            break

    values = _values(
        arrays,
        x=x,
        loop_count=loop_count,
        enclosed=enclosed,
        stage_objectives=np.asarray(stage_objectives, dtype=np.float64),
        nonfinal_steps=nonfinal_steps,
        final_adjustment_run=final_adjustment_run,
        max_iterations_per_step=max_iterations,
    )
    return _observation(
        "native-cpu",
        bundle,
        arrays,
        backend_mode="native_cpu",
        platform="cpu",
        driver="simsopt_cpp_multistep_gsco",
        values=values,
    )


def _jax(
    lane: ParityLane,
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
) -> LaneObservation:
    import jax

    from simsopt_jax.backend.runtime import get_runtime_jax_device
    from simsopt_jax.core.wireframe_workflow import (
        WireframeGSCOLiveParams,
        wireframe_gsco_multistep_loop_jax,
    )

    device = get_runtime_jax_device()

    def put(name: str):
        return jax.device_put(arrays[name], device)

    initial_default_current = _configuration_float(
        bundle,
        "initial_default_current",
    )
    result = jax.device_get(
        wireframe_gsco_multistep_loop_jax(
            WireframeGSCOLiveParams(
                A=put("response_matrix"),
                loops=put("loops"),
                free_loops=put("free_loops"),
                segments=put("segments"),
                connections=put("connections"),
                default_current=jax.device_put(initial_default_current, device),
                max_current=jax.device_put(1.1 * initial_default_current, device),
                lambda_s=jax.device_put(
                    _configuration_float(bundle, "lambda_s"),
                    device,
                ),
                tol=jax.device_put(0.001 * initial_default_current, device),
                max_loop_count=1,
                no_crossing=True,
                no_new_coils=False,
                match_current=False,
            ),
            put("target"),
            put("initial_currents"),
            put("initial_loop_count"),
            put("loops"),
            put("neighbors"),
            put("base_constrained_segments"),
            max_iter_per_step=_configuration_int(
                bundle,
                "max_iterations_per_step",
            ),
            max_outer_steps=_configuration_int(bundle, "max_outer_steps"),
            initial_current_fraction=_configuration_float(
                bundle,
                "initial_current_fraction",
            ),
            current_scale=abs(_configuration_float(bundle, "poloidal_current")),
            min_coil_size=_configuration_int(bundle, "minimum_coil_size"),
            final_max_current=_configuration_float(bundle, "final_max_current"),
        )
    )
    stage_count = int(result.stage_count)
    values = _values(
        arrays,
        x=np.asarray(result.x, dtype=np.float64).reshape(-1),
        loop_count=np.asarray(result.loop_count, dtype=np.int64),
        enclosed=np.asarray(result.enclosed_segment_mask, dtype=np.bool_),
        stage_objectives=np.asarray(
            result.stage_objectives[:stage_count],
            dtype=np.float64,
        ),
        nonfinal_steps=int(result.nonfinal_steps),
        final_adjustment_run=bool(result.final_adjustment_run),
        max_iterations_per_step=_configuration_int(
            bundle,
            "max_iterations_per_step",
        ),
    )
    platform = "cpu" if device is None else device.platform
    return _observation(
        lane,
        bundle,
        arrays,
        backend_mode=os.environ["SIMSOPT_BACKEND_MODE"],
        platform="gpu" if platform in {"cuda", "gpu"} else platform,
        driver="simsopt_jax_multistep_gsco",
        values=values,
    )


def execute(
    lane: ParityLane,
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
) -> LaneObservation:
    """Execute the exact bounded multistep GSCO source workflow."""
    if lane == "native-cpu":
        return _native(bundle, arrays)
    return _jax(lane, bundle, arrays)
