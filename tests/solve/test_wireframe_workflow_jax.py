"""N7 live-loop tests for wireframe GSCO JAX workflow state.

Lane key: reporting_contract for host-boundary live-loop invariants.
"""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import simsoptpp as sopp

from simsopt.jax_core.wireframe_workflow import (
    WireframeGSCOLiveParams,
    _wireframe_gsco_multistep_initial_state,
    find_wireframe_coil_sizes_jax,
    greedy_stellarator_coil_optimization_jax,
    gsco_live_loop_jax,
    wireframe_gsco_initial_state,
    wireframe_gsco_multistep_loop_jax,
    wireframe_gsco_never_stop,
)


def _gsco_problem():
    rng = np.random.default_rng(3104)
    A = np.ascontiguousarray(rng.standard_normal(size=(5, 6)))
    b = np.ascontiguousarray(rng.standard_normal(size=(5, 1)))
    loops = np.ascontiguousarray(np.array([[0, 1, 2, 3], [2, 3, 4, 5]], dtype=np.int64))
    free_loops = np.ascontiguousarray(np.ones(2, dtype=np.int64))
    segments = np.ascontiguousarray(
        np.array(
            [[0, 1], [1, 2], [2, 3], [3, 0], [0, 2], [1, 3]],
            dtype=np.int64,
        )
    )
    connections = np.ascontiguousarray(
        np.array(
            [[0, 3, 4, 0], [0, 1, 5, 0], [1, 2, 4, 0], [2, 3, 5, 0]],
            dtype=np.int64,
        )
    )
    x_init = np.ascontiguousarray(np.zeros((6, 1), dtype=np.float64))
    loop_count_init = np.ascontiguousarray(np.zeros(2, dtype=np.int64))
    return A, b, loops, free_loops, segments, connections, x_init, loop_count_init


def _synthetic_gsco_problem(*, n_grid: int, n_loops: int, seed: int):
    loops = np.arange(4 * n_loops, dtype=np.int64).reshape(n_loops, 4)
    free_loops = np.ones((n_loops,), dtype=np.int64)

    rng = np.random.default_rng(seed)
    A = rng.standard_normal(size=(n_grid, 4 * n_loops))
    b = rng.standard_normal(size=(n_grid, 1))

    nodes = np.arange(4 * n_loops, dtype=np.int64)
    segments = np.stack([nodes, np.roll(nodes, -1)], axis=1)
    connections = np.zeros((4 * n_loops, 4), dtype=np.int64)
    connections[:, 0] = nodes
    x_init = np.zeros((4 * n_loops, 1), dtype=np.float64)
    loop_count_init = np.zeros((n_loops,), dtype=np.int64)
    return (
        np.ascontiguousarray(A),
        np.ascontiguousarray(b),
        np.ascontiguousarray(loops),
        np.ascontiguousarray(free_loops),
        np.ascontiguousarray(segments),
        np.ascontiguousarray(connections),
        np.ascontiguousarray(x_init),
        np.ascontiguousarray(loop_count_init),
    )


def _params(
    A,
    loops,
    free_loops,
    segments,
    connections,
    *,
    default_current: float,
    max_current: float,
    max_loop_count: int,
    lambda_s: float,
) -> WireframeGSCOLiveParams:
    device = jax.devices()[0]
    A_arr = jax.device_put(np.asarray(A, dtype=np.float64), device=device)
    default_current_abs = jnp.abs(
        jax.device_put(np.asarray(default_current, dtype=A_arr.dtype), device=device)
    )
    tolerance_fraction = jax.device_put(
        np.asarray(0.001, dtype=A_arr.dtype), device=device
    )
    return WireframeGSCOLiveParams(
        A=A_arr,
        loops=jax.device_put(np.asarray(loops, dtype=np.int32), device=device),
        free_loops=jax.device_put(
            np.asarray(free_loops, dtype=np.int32), device=device
        ),
        segments=jax.device_put(np.asarray(segments, dtype=np.int32), device=device),
        connections=jax.device_put(
            np.asarray(connections, dtype=np.int32), device=device
        ),
        default_current=default_current_abs,
        max_current=jnp.abs(
            jax.device_put(np.asarray(max_current, dtype=A_arr.dtype), device=device)
        ),
        lambda_s=jax.device_put(np.asarray(lambda_s, dtype=A_arr.dtype), device=device),
        tol=tolerance_fraction * default_current_abs,
        max_loop_count=abs(max_loop_count),
        no_crossing=False,
        no_new_coils=False,
        match_current=False,
    )


def test_wireframe_initial_states_allow_strict_host_to_device_transfer_guard():
    A, b, loops, free_loops, segments, connections, x_init, loop_count_init = (
        _gsco_problem()
    )
    params = _params(
        A,
        loops,
        free_loops,
        segments,
        connections,
        default_current=0.2,
        max_current=1.0,
        max_loop_count=3,
        lambda_s=0.01,
    )

    with jax.transfer_guard_host_to_device("disallow"):
        state = wireframe_gsco_initial_state(
            params,
            b,
            x_init,
            loop_count_init,
            history_capacity=4,
        )
        multistep_state = _wireframe_gsco_multistep_initial_state(
            x_init,
            loop_count_init,
            current_fraction=0.5,
        )
        assert state.iter_history.shape == (4,)
        assert multistep_state.current_fraction.shape == ()
        result = greedy_stellarator_coil_optimization_jax(
            False,
            False,
            False,
            A,
            b,
            0.2,
            1.0,
            3,
            loops,
            free_loops,
            segments,
            connections,
            0.01,
            3,
            x_init,
            loop_count_init,
        )
        sampled = greedy_stellarator_coil_optimization_jax(
            False,
            False,
            False,
            A,
            b,
            0.2,
            1.0,
            3,
            loops,
            free_loops,
            segments,
            connections,
            0.01,
            3,
            x_init,
            loop_count_init,
            record_every=1,
        )
        no_crossing = greedy_stellarator_coil_optimization_jax(
            True,
            False,
            False,
            A,
            b,
            0.2,
            1.0,
            3,
            loops,
            free_loops,
            segments,
            connections,
            0.01,
            3,
            x_init,
            loop_count_init,
        )
        assert result.iter_history.shape == (4,)
        assert sampled.iter_history.shape == (5,)
        assert no_crossing.iter_history.shape == (4,)
    assert bool(jax.device_get(wireframe_gsco_never_stop(state))) is False
    assert bool(jax.device_get(multistep_state.done)) is False


def _run_cpp_gsco(
    A,
    b,
    loops,
    free_loops,
    segments,
    connections,
    x_init,
    loop_count_init,
    *,
    default_current: float,
    max_current: float,
    max_loop_count: int,
    lambda_s: float,
    max_steps: int,
    no_new_coils: bool = False,
    match_current: bool = False,
):
    return sopp.GSCO(
        False,
        no_new_coils,
        match_current,
        A,
        b,
        abs(default_current),
        abs(max_current),
        abs(max_loop_count),
        loops,
        free_loops,
        segments,
        connections,
        lambda_s,
        max_steps,
        x_init,
        loop_count_init,
        1,
    )


def _free_loops_from_constraints(loops, constrained_segment_mask):
    return ~np.any(constrained_segment_mask[np.asarray(loops, dtype=np.int64)], axis=1)


def _host_coil_sizes(loop_count, neighbors):
    active = np.asarray(loop_count) != 0
    neighbors_arr = np.asarray(neighbors, dtype=np.int64)
    coil_ids = np.full(active.shape, -1, dtype=np.int64)
    sizes = np.zeros(active.shape, dtype=np.int64)
    coil_id = 0
    for start in range(active.size):
        if not active[start] or coil_ids[start] >= 0:
            continue
        stack = [start]
        members = []
        coil_ids[start] = coil_id
        while stack:
            cell = stack.pop()
            members.append(cell)
            for neighbor in neighbors_arr[cell]:
                if active[neighbor] and coil_ids[neighbor] < 0:
                    coil_ids[neighbor] = coil_id
                    stack.append(int(neighbor))
        sizes[members] = len(members)
        coil_id += 1
    return sizes


def _host_multistep_reference(
    A,
    b,
    loops,
    segments,
    connections,
    neighbors,
    x_init,
    loop_count_init,
    *,
    max_iter_per_step: int,
    max_outer_steps: int,
    initial_current_fraction: float,
    current_scale: float,
    min_coil_size: int,
    final_max_current: float,
    lambda_s: float,
):
    x = np.asarray(x_init, dtype=np.float64).reshape((-1, 1))
    loop_count = np.asarray(loop_count_init, dtype=np.int64).copy()
    previous_x = None
    enclosed = np.zeros((x.size,), dtype=bool)
    base_constrained = np.zeros((x.size,), dtype=bool)
    current_fraction = initial_current_fraction
    nonfinal_steps = 0
    for _outer in range(max_outer_steps):
        final_step = previous_x is not None and np.array_equal(previous_x, x)
        x_start = x.copy()
        constrained = base_constrained if final_step else base_constrained | enclosed
        free_loops = np.ascontiguousarray(
            _free_loops_from_constraints(loops, constrained).astype(np.int64)
        )
        default_current = 0.0 if final_step else abs(current_fraction * current_scale)
        max_current = (
            abs(final_max_current)
            if final_step
            else 1.1 * abs(current_fraction * current_scale)
        )
        x_next, loop_count_next, *_history = _run_cpp_gsco(
            A,
            b,
            loops,
            free_loops,
            segments,
            connections,
            np.ascontiguousarray(x),
            loop_count,
            default_current=default_current,
            max_current=max_current,
            max_loop_count=1,
            lambda_s=lambda_s,
            max_steps=max_iter_per_step,
            no_new_coils=final_step,
            match_current=final_step,
        )
        x = np.asarray(x_next, dtype=np.float64).reshape((-1, 1))
        loop_count = np.asarray(loop_count_next, dtype=np.int64)
        if final_step:
            return x, loop_count, np.zeros_like(enclosed), nonfinal_steps, True

        coil_sizes = _host_coil_sizes(loop_count, neighbors)
        small_cells = np.logical_and(coil_sizes > 0, coil_sizes < min_coil_size)
        segment_prune_mask = np.zeros((x.size,), dtype=bool)
        segment_prune_mask[np.unique(np.asarray(loops)[small_cells].reshape((-1)))] = (
            True
        )
        x[segment_prune_mask, :] = 0.0
        loop_count[small_cells] = 0

        active_cells = loop_count != 0
        enclosed = np.zeros((x.size,), dtype=bool)
        enclosed[np.unique(np.asarray(loops)[active_cells].reshape((-1)))] = True
        enclosed[x.reshape((-1)) != 0.0] = False
        previous_x = x_start
        nonfinal_steps += 1
        current_fraction *= 0.5
    return x, loop_count, enclosed, nonfinal_steps, False


def _assert_live_state_matches_cpp(state, expected) -> None:
    (
        x_expected,
        loop_count_expected,
        iter_hist_expected,
        curr_hist_expected,
        loop_hist_expected,
        f_B_hist_expected,
        f_S_hist_expected,
        f_hist_expected,
    ) = expected
    history_slice = slice(0, int(np.asarray(state.history_length)))

    np.testing.assert_allclose(np.asarray(state.x).reshape((-1, 1)), x_expected)
    np.testing.assert_array_equal(np.asarray(state.loop_count), loop_count_expected)
    np.testing.assert_array_equal(
        np.asarray(state.iter_history)[history_slice], iter_hist_expected
    )
    np.testing.assert_allclose(
        np.asarray(state.curr_history)[history_slice], curr_hist_expected
    )
    np.testing.assert_array_equal(
        np.asarray(state.loop_history)[history_slice], loop_hist_expected
    )
    np.testing.assert_allclose(
        np.asarray(state.f_B_history)[history_slice], f_B_hist_expected
    )
    np.testing.assert_allclose(
        np.asarray(state.f_S_history)[history_slice], f_S_hist_expected
    )
    np.testing.assert_allclose(
        np.asarray(state.f_history)[history_slice], f_hist_expected
    )


def test_gsco_live_loop_matches_cpp_host_loop_for_five_steps() -> None:
    A, b, loops, free_loops, segments, connections, x_init, loop_count_init = (
        _gsco_problem()
    )
    max_steps = 5
    params = _params(
        A,
        loops,
        free_loops,
        segments,
        connections,
        default_current=0.2,
        max_current=np.inf,
        max_loop_count=0,
        lambda_s=0.15,
    )
    initial = wireframe_gsco_initial_state(
        params,
        jnp.asarray(b),
        jnp.asarray(x_init),
        jnp.asarray(loop_count_init),
        history_capacity=max_steps + 1,
    )

    actual = gsco_live_loop_jax(initial, max_steps=max_steps, params=params)
    expected = _run_cpp_gsco(
        A,
        b,
        loops,
        free_loops,
        segments,
        connections,
        x_init,
        loop_count_init,
        default_current=0.2,
        max_current=np.inf,
        max_loop_count=0,
        lambda_s=0.15,
        max_steps=max_steps,
    )

    _assert_live_state_matches_cpp(actual, expected)


def test_gsco_live_loop_matches_cpp_host_loop_for_fifty_steps() -> None:
    A, b, loops, free_loops, segments, connections, x_init, loop_count_init = (
        _synthetic_gsco_problem(n_grid=200, n_loops=50, seed=3110)
    )
    max_steps = 50
    params = _params(
        A,
        loops,
        free_loops,
        segments,
        connections,
        default_current=0.2,
        max_current=1.0,
        max_loop_count=3,
        lambda_s=0.05,
    )
    initial = wireframe_gsco_initial_state(
        params,
        jnp.asarray(b),
        jnp.asarray(x_init),
        jnp.asarray(loop_count_init),
        history_capacity=max_steps + 1,
    )

    actual = gsco_live_loop_jax(initial, max_steps=max_steps, params=params)
    expected = _run_cpp_gsco(
        A,
        b,
        loops,
        free_loops,
        segments,
        connections,
        x_init,
        loop_count_init,
        default_current=0.2,
        max_current=1.0,
        max_loop_count=3,
        lambda_s=0.05,
        max_steps=max_steps,
    )

    _assert_live_state_matches_cpp(actual, expected)


def test_gsco_live_loop_restart_continuation_is_exact() -> None:
    A, b, loops, free_loops, segments, connections, x_init, loop_count_init = (
        _synthetic_gsco_problem(n_grid=80, n_loops=20, seed=3112)
    )
    params = _params(
        A,
        loops,
        free_loops,
        segments,
        connections,
        default_current=0.2,
        max_current=1.0,
        max_loop_count=3,
        lambda_s=0.05,
    )
    initial = wireframe_gsco_initial_state(
        params,
        jnp.asarray(b),
        jnp.asarray(x_init),
        jnp.asarray(loop_count_init),
        history_capacity=11,
    )

    full = gsco_live_loop_jax(initial, max_steps=10, params=params)
    partial = gsco_live_loop_jax(initial, max_steps=4, params=params)
    continued = gsco_live_loop_jax(partial, max_steps=6, params=params)

    np.testing.assert_allclose(np.asarray(continued.x), np.asarray(full.x))
    np.testing.assert_array_equal(
        np.asarray(continued.loop_count), np.asarray(full.loop_count)
    )
    np.testing.assert_array_equal(
        np.asarray(continued.iter_history), np.asarray(full.iter_history)
    )
    np.testing.assert_allclose(
        np.asarray(continued.curr_history), np.asarray(full.curr_history)
    )
    np.testing.assert_array_equal(
        np.asarray(continued.loop_history), np.asarray(full.loop_history)
    )
    np.testing.assert_allclose(
        np.asarray(continued.f_history), np.asarray(full.f_history)
    )


def test_find_wireframe_coil_sizes_jax_matches_host_components() -> None:
    loop_count = jnp.asarray([1, 1, 0, -1, -1, -1], dtype=jnp.int32)
    neighbors = jnp.asarray(
        [
            [1, 1, 1, 1],
            [0, 0, 0, 0],
            [2, 2, 2, 2],
            [4, 4, 4, 4],
            [3, 5, 3, 5],
            [4, 4, 4, 4],
        ],
        dtype=jnp.int32,
    )

    actual = find_wireframe_coil_sizes_jax(loop_count, neighbors)

    np.testing.assert_array_equal(
        np.asarray(actual),
        _host_coil_sizes(np.asarray(loop_count), np.asarray(neighbors)),
    )


def test_gsco_multistep_loop_matches_cpp_host_orchestration_with_final_adjustment() -> (
    None
):
    A, b, loops, free_loops, segments, connections, x_init, loop_count_init = (
        _gsco_problem()
    )
    neighbors = np.ascontiguousarray(
        np.array([[1, 1, 1, 1], [0, 0, 0, 0]], dtype=np.int64)
    )
    params = _params(
        A,
        loops,
        free_loops,
        segments,
        connections,
        default_current=0.2,
        max_current=1.0,
        max_loop_count=1,
        lambda_s=0.15,
    )

    actual = wireframe_gsco_multistep_loop_jax(
        params,
        jnp.asarray(b),
        jnp.asarray(x_init),
        jnp.asarray(loop_count_init),
        jnp.asarray(loops),
        jnp.asarray(neighbors),
        jnp.zeros((x_init.size,), dtype=bool),
        max_iter_per_step=5,
        max_outer_steps=4,
        initial_current_fraction=0.2,
        current_scale=1.0,
        min_coil_size=3,
        final_max_current=0.22,
    )
    expected = _host_multistep_reference(
        A,
        b,
        loops,
        segments,
        connections,
        neighbors,
        x_init,
        loop_count_init,
        max_iter_per_step=5,
        max_outer_steps=4,
        initial_current_fraction=0.2,
        current_scale=1.0,
        min_coil_size=3,
        final_max_current=0.22,
        lambda_s=0.15,
    )
    (
        expected_x,
        expected_loop_count,
        expected_enclosed,
        expected_nonfinal_steps,
        expected_final_adjustment_run,
    ) = expected

    np.testing.assert_allclose(np.asarray(actual.x), expected_x)
    np.testing.assert_array_equal(np.asarray(actual.loop_count), expected_loop_count)
    np.testing.assert_array_equal(
        np.asarray(actual.enclosed_segment_mask), expected_enclosed
    )
    assert int(np.asarray(actual.nonfinal_steps)) == expected_nonfinal_steps
    assert (
        bool(np.asarray(actual.final_adjustment_run)) is expected_final_adjustment_run
    )

    def _run_impl(A_data):
        return wireframe_gsco_multistep_loop_jax(
            replace(params, A=A_data),
            jnp.asarray(b),
            jnp.asarray(x_init),
            jnp.asarray(loop_count_init),
            jnp.asarray(loops),
            jnp.asarray(neighbors),
            jnp.zeros((x_init.size,), dtype=bool),
            max_iter_per_step=5,
            max_outer_steps=4,
            initial_current_fraction=0.2,
            current_scale=1.0,
            min_coil_size=3,
            final_max_current=0.22,
        )

    A_device = jnp.asarray(A, dtype=jnp.float64)
    assert "scan[" in str(jax.make_jaxpr(_run_impl)(A_device))
    _run = jax.jit(_run_impl)
    compiled = _run(A_device)
    compiled.x.block_until_ready()

    with jax.transfer_guard("disallow"):
        guarded = _run(A_device)
        guarded.x.block_until_ready()


def test_gsco_final_adjustment_oracle_flags_change_cpp_result() -> None:
    A, b, loops, free_loops, segments, connections, _x_init, _loop_count_init = (
        _gsco_problem()
    )
    x_init = np.ascontiguousarray(
        np.array([0.2, 0.2, -0.2, -0.2, 0.0, 0.0], dtype=np.float64).reshape((-1, 1))
    )
    loop_count_init = np.ascontiguousarray(np.array([1, 0], dtype=np.int64))

    strict_final = _run_cpp_gsco(
        A,
        b,
        loops,
        free_loops,
        segments,
        connections,
        x_init,
        loop_count_init,
        default_current=0.0,
        max_current=0.22,
        max_loop_count=1,
        lambda_s=0.15,
        max_steps=5,
        no_new_coils=True,
        match_current=True,
    )
    loose_final = _run_cpp_gsco(
        A,
        b,
        loops,
        free_loops,
        segments,
        connections,
        x_init,
        loop_count_init,
        default_current=0.0,
        max_current=0.22,
        max_loop_count=1,
        lambda_s=0.15,
        max_steps=5,
        no_new_coils=True,
        match_current=False,
    )

    assert not np.array_equal(strict_final[0], loose_final[0])
    assert not np.array_equal(strict_final[1], loose_final[1])


def test_gsco_multistep_loop_marks_enclosed_zero_current_segments() -> None:
    A, b, loops, free_loops, segments, connections, x_init, _loop_count_init = (
        _gsco_problem()
    )
    neighbors = np.ascontiguousarray(
        np.array([[1, 1, 1, 1], [0, 0, 0, 0]], dtype=np.int64)
    )
    params = _params(
        A,
        loops,
        free_loops,
        segments,
        connections,
        default_current=0.2,
        max_current=1.0,
        max_loop_count=1,
        lambda_s=0.15,
    )
    loop_count_init = np.ascontiguousarray(np.array([1, 0], dtype=np.int64))

    actual = wireframe_gsco_multistep_loop_jax(
        params,
        jnp.asarray(b),
        jnp.asarray(x_init),
        jnp.asarray(loop_count_init),
        jnp.asarray(loops),
        jnp.asarray(neighbors),
        jnp.zeros((x_init.size,), dtype=bool),
        max_iter_per_step=0,
        max_outer_steps=1,
        initial_current_fraction=0.2,
        current_scale=1.0,
        min_coil_size=1,
        final_max_current=0.22,
    )
    expected_enclosed = np.zeros((x_init.size,), dtype=bool)
    expected_enclosed[loops[0]] = True

    np.testing.assert_array_equal(
        np.asarray(actual.enclosed_segment_mask),
        expected_enclosed,
    )
    assert int(np.asarray(actual.nonfinal_steps)) == 1
    assert bool(np.asarray(actual.final_adjustment_run)) is False


def test_gsco_live_loop_rejects_eager_capacity_overrun() -> None:
    A, b, loops, free_loops, segments, connections, x_init, loop_count_init = (
        _gsco_problem()
    )
    params = _params(
        A,
        loops,
        free_loops,
        segments,
        connections,
        default_current=0.2,
        max_current=np.inf,
        max_loop_count=0,
        lambda_s=0.15,
    )
    initial = wireframe_gsco_initial_state(
        params,
        jnp.asarray(b),
        jnp.asarray(x_init),
        jnp.asarray(loop_count_init),
        history_capacity=3,
    )

    with pytest.raises(ValueError, match="history capacity"):
        gsco_live_loop_jax(initial, max_steps=3, params=params)


def test_gsco_live_loop_accepts_staged_restart_state_under_jit() -> None:
    A, b, loops, free_loops, segments, connections, x_init, loop_count_init = (
        _gsco_problem()
    )
    params = _params(
        A,
        loops,
        free_loops,
        segments,
        connections,
        default_current=0.2,
        max_current=np.inf,
        max_loop_count=0,
        lambda_s=0.15,
    )
    initial = wireframe_gsco_initial_state(
        params,
        jnp.asarray(b),
        jnp.asarray(x_init),
        jnp.asarray(loop_count_init),
        history_capacity=6,
    )

    @jax.jit
    def _run(state):
        return gsco_live_loop_jax(state, max_steps=5, params=params)

    actual = _run(initial)
    expected = gsco_live_loop_jax(initial, max_steps=5, params=params)

    np.testing.assert_allclose(np.asarray(actual.x), np.asarray(expected.x))
    np.testing.assert_array_equal(
        np.asarray(actual.loop_count), np.asarray(expected.loop_count)
    )
    np.testing.assert_allclose(
        np.asarray(actual.f_history), np.asarray(expected.f_history)
    )


def test_gsco_live_loop_rejects_staged_capacity_overrun_under_jit() -> None:
    A, b, loops, free_loops, segments, connections, x_init, loop_count_init = (
        _gsco_problem()
    )
    params = _params(
        A,
        loops,
        free_loops,
        segments,
        connections,
        default_current=0.2,
        max_current=np.inf,
        max_loop_count=0,
        lambda_s=0.15,
    )
    initial = wireframe_gsco_initial_state(
        params,
        jnp.asarray(b),
        jnp.asarray(x_init),
        jnp.asarray(loop_count_init),
        history_capacity=3,
    )
    over_capacity = replace(
        initial,
        history_length=jnp.asarray(3, dtype=initial.history_length.dtype),
    )

    @jax.jit
    def _run(state):
        return gsco_live_loop_jax(state, max_steps=1, params=params)

    with pytest.raises(jax.errors.JaxRuntimeError, match="history capacity"):
        jax.block_until_ready(_run(over_capacity))


def test_gsco_live_loop_jits_under_transfer_guard() -> None:
    A, b, loops, free_loops, segments, connections, x_init, loop_count_init = (
        _gsco_problem()
    )
    params = _params(
        A,
        loops,
        free_loops,
        segments,
        connections,
        default_current=0.2,
        max_current=np.inf,
        max_loop_count=0,
        lambda_s=0.15,
    )
    initial = wireframe_gsco_initial_state(
        params,
        jnp.asarray(b),
        jnp.asarray(x_init),
        jnp.asarray(loop_count_init),
        history_capacity=6,
    )

    def _run_impl(state, A_data):
        runtime_params = WireframeGSCOLiveParams(
            A=A_data,
            loops=params.loops,
            free_loops=params.free_loops,
            segments=params.segments,
            connections=params.connections,
            default_current=params.default_current,
            max_current=params.max_current,
            lambda_s=params.lambda_s,
            tol=params.tol,
            max_loop_count=params.max_loop_count,
            no_crossing=params.no_crossing,
            no_new_coils=params.no_new_coils,
            match_current=params.match_current,
        )
        return gsco_live_loop_jax(state, max_steps=5, params=runtime_params)

    run = jax.jit(_run_impl)
    compiled = run(initial, params.A)
    compiled.x.block_until_ready()

    with jax.transfer_guard("disallow"):
        result = run(initial, params.A)
        result.x.block_until_ready()

    assert int(np.asarray(result.history_length)) == 5
    assert "scan[" in str(jax.make_jaxpr(_run_impl)(initial, params.A))
