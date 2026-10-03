"""Every GSCO state buffer must be built on the device, not placed from the host.

``jax.device_put`` of a non-scalar HOST array inside a traced region is staged
into the computation and becomes an implicit host-to-device copy when the
computation runs. The strict GPU lanes execute with
``jax_transfer_guard_host_to_device = disallow``, so such a buffer aborts the run:
the multistep GSCO workflow died that way on the jax-gpu lane, inside
``jax.lax.while_loop`` in ``wireframe_gsco_multistep_loop_jax``, with
``aval=ShapedArray(int32[2501])`` -- the ``max_iter_per_step + 1`` history buffer
that ``wireframe_gsco_initial_state`` allocates for each staged solve. The guard
fires identically on the CPU backend, so this file reproduces the defect without
a GPU. A 0-d value is exempt from the guard, which is why the single-stage
lanes, whose state is built eagerly, survived.
"""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import jax
import jax.numpy as jnp
import numpy as np
from simsopt_jax.core.wireframe_workflow import (
    WireframeGSCOLiveParams,
    greedy_stellarator_coil_optimization_jax,
    gsco_live_loop_jax,
    wireframe_gsco_initial_state,
    wireframe_gsco_multistep_loop_jax,
)

DEVICE = jax.devices()[0]
MAX_ITER_PER_STEP = 5


def _place(value, dtype: np.dtype) -> jax.Array:
    """Put one operand on the device BEFORE the guard is armed.

    ``np.asarray`` keeps a scalar 0-d: ``np.ascontiguousarray`` would make it
    ``(1,)`` and every scalar of the GSCO state would inherit that shape.
    """
    return jax.device_put(np.asarray(value, dtype=dtype), DEVICE)


def _problem() -> dict[str, jax.Array]:
    """A two-loop, six-segment GSCO problem, entirely device resident."""
    generator = np.random.default_rng(20260920)
    return {
        "A": _place(generator.standard_normal((5, 6)), np.float64),
        "b": _place(generator.standard_normal((5, 1)), np.float64),
        "loops": _place([[0, 1, 2, 3], [2, 3, 4, 5]], np.int32),
        "free_loops": _place([1, 1], np.int32),
        "segments": _place([[0, 1], [1, 2], [2, 3], [3, 0], [0, 2], [1, 3]], np.int32),
        "connections": _place(
            [[0, 3, 4, 0], [0, 1, 5, 0], [1, 2, 4, 0], [2, 3, 5, 0]], np.int32
        ),
        "neighbors": _place([[1, 1, 1, 1], [0, 0, 0, 0]], np.int32),
        "x_init": _place(np.zeros((6, 1)), np.float64),
        "loop_count_init": _place(np.zeros(2), np.int32),
        "constrained": jax.device_put(np.zeros((6,), dtype=np.bool_), DEVICE),
        "default_current": _place(0.2, np.float64),
        "max_current": _place(1.0, np.float64),
        "lambda_s": _place(0.15, np.float64),
        "tol": _place(2.0e-4, np.float64),
    }


def _params(problem: dict[str, jax.Array]) -> WireframeGSCOLiveParams:
    return WireframeGSCOLiveParams(
        A=problem["A"],
        loops=problem["loops"],
        free_loops=problem["free_loops"],
        segments=problem["segments"],
        connections=problem["connections"],
        default_current=problem["default_current"],
        max_current=problem["max_current"],
        lambda_s=problem["lambda_s"],
        tol=problem["tol"],
        max_loop_count=1,
        no_crossing=False,
        no_new_coils=False,
        match_current=False,
    )


def test_multistep_loop_runs_under_a_strict_host_to_device_guard() -> None:
    # The staged workflow builds one single-stage state per outer iteration,
    # INSIDE jax.lax.while_loop. Placing those buffers from the host is the
    # jax-gpu crash; this call raises JaxRuntimeError on the pre-fix module.
    problem = _problem()
    with jax.transfer_guard_host_to_device("disallow"):
        result = wireframe_gsco_multistep_loop_jax(
            _params(problem),
            problem["b"],
            problem["x_init"],
            problem["loop_count_init"],
            problem["loops"],
            problem["neighbors"],
            problem["constrained"],
            max_iter_per_step=MAX_ITER_PER_STEP,
            max_outer_steps=4,
            initial_current_fraction=0.2,
            current_scale=1.0,
            min_coil_size=3,
            final_max_current=0.22,
            stage_history_capacity=4,
        )
        jax.block_until_ready(result.x)
    assert result.stage_iterations.shape == (4,)
    assert bool(np.all(np.isfinite(np.asarray(jax.device_get(result.x)))))


def test_single_stage_loop_and_state_run_under_the_same_guard() -> None:
    # The same construction, in the eagerly built single-stage state and its
    # scan. It passes today; it must keep passing after the buffers become
    # device-native, and it fails if a host array is introduced anywhere in the
    # scan carry.
    problem = _problem()
    params = _params(problem)
    with jax.transfer_guard_host_to_device("disallow"):
        state = wireframe_gsco_initial_state(
            params,
            problem["b"],
            problem["x_init"],
            problem["loop_count_init"],
            history_capacity=MAX_ITER_PER_STEP + 1,
        )
        final = gsco_live_loop_jax(
            state,
            max_steps=MAX_ITER_PER_STEP,
            params=params,
        )
        jax.block_until_ready(final.x)
    assert state.iter_history.shape == (MAX_ITER_PER_STEP + 1,)
    assert int(jax.device_get(final.history_length)) >= 1


def test_state_buffers_are_zero_and_keep_the_operand_placement() -> None:
    # The device-native construction must produce exactly the buffers the host
    # placement produced: all zeros, the operands' dtypes, on the operands'
    # device.
    problem = _problem()
    state = wireframe_gsco_initial_state(
        _params(problem),
        problem["b"],
        problem["x_init"],
        problem["loop_count_init"],
        history_capacity=MAX_ITER_PER_STEP + 1,
    )
    for buffer, dtype in (
        (state.iter_history, jnp.int32),
        (state.loop_history, jnp.int32),
        (state.curr_history, problem["A"].dtype),
    ):
        assert buffer.dtype == dtype
        assert buffer.devices() == problem["A"].devices()
        np.testing.assert_array_equal(
            np.asarray(jax.device_get(buffer)),
            np.zeros((MAX_ITER_PER_STEP + 1,), dtype=np.asarray(buffer).dtype),
        )


def test_single_stage_lane_entry_point_runs_under_the_same_guard() -> None:
    # The path the single-stage GSCO lanes really execute
    # (``gsco_wireframe_jax`` forwards to this function). It builds its state
    # and its sampled history the same way, so it carries the same risk.
    problem = _problem()
    with jax.transfer_guard_host_to_device("disallow"):
        result = greedy_stellarator_coil_optimization_jax(
            False,
            False,
            False,
            problem["A"],
            problem["b"],
            problem["default_current"],
            problem["max_current"],
            1,
            problem["loops"],
            problem["free_loops"],
            problem["segments"],
            problem["connections"],
            problem["lambda_s"],
            MAX_ITER_PER_STEP,
            problem["x_init"],
            problem["loop_count_init"],
            record_every=1,
        )
        jax.block_until_ready(result.x)
    assert int(jax.device_get(result.history_length)) >= 1


def test_the_traced_multistep_program_carries_no_host_constant() -> None:
    # The structural form of the same defect, independent of which device the
    # test runs on: a host buffer built inside the traced region becomes a
    # CONSTANT of the staged program, and every such constant is copied to the
    # device when the program runs. Before the fix this jaxpr carried ten host
    # numpy constants (the per-stage history buffers and the stage carries);
    # after it, none.
    problem = _problem()
    params = _params(problem)

    def _run(
        run_params: WireframeGSCOLiveParams,
        b: jax.Array,
        x_init: jax.Array,
        loop_count_init: jax.Array,
        cell_key: jax.Array,
        neighbors: jax.Array,
        constrained: jax.Array,
    ):
        return wireframe_gsco_multistep_loop_jax(
            run_params,
            b,
            x_init,
            loop_count_init,
            cell_key,
            neighbors,
            constrained,
            max_iter_per_step=MAX_ITER_PER_STEP,
            max_outer_steps=4,
            initial_current_fraction=0.2,
            current_scale=1.0,
            min_coil_size=3,
            final_max_current=0.22,
            stage_history_capacity=4,
        )

    jaxpr = jax.make_jaxpr(_run)(
        params,
        problem["b"],
        problem["x_init"],
        problem["loop_count_init"],
        problem["loops"],
        problem["neighbors"],
        problem["constrained"],
    )
    host_constants = [
        (np.shape(constant), np.asarray(constant).dtype)
        for constant in jaxpr.consts
        if isinstance(constant, np.ndarray)
    ]
    assert host_constants == []
    assert jaxpr.consts == []
