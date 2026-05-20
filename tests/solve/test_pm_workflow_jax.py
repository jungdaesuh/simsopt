"""N7 live-loop tests for permanent-magnet JAX workflow state.

Lane keys: reporting_contract for live-loop host-boundary state, direct_kernel
and pm_mwpgp_fixed_step for fixed-state PM math covered by wrapper tests.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from simsopt.jax_core.pm_optimization import (
    GPMOArbVecBacktrackingSpec,
    GPMOArbVecSpec,
    GPMOBacktrackingSpec,
    GPMOBaselineSpec,
    GPMOMultiSpec,
    _gpmo_arbvec_contributions,
    gpmo_arbvec_backtracking_step,
    gpmo_arbvec_step,
    gpmo_backtracking_step,
    gpmo_baseline_step,
    gpmo_connectivity_matrix,
    gpmo_multi_step,
)
from simsopt.jax_core.pm_workflow import (
    PMGPMOArbVecBacktrackingLiveState,
    PMGPMOArbVecLiveState,
    PMGPMOBacktrackingLiveState,
    PMGPMOLiveState,
    PMGPMOMultiLiveState,
    pm_gpmo_arbvec_backtracking_initial_state,
    pm_gpmo_arbvec_backtracking_live_loop_jax,
    pm_gpmo_arbvec_initial_state,
    pm_gpmo_arbvec_live_loop_jax,
    pm_gpmo_arbvec_never_stop,
    pm_gpmo_baseline_initial_state,
    pm_gpmo_backtracking_initial_state,
    pm_gpmo_backtracking_live_loop_jax,
    pm_gpmo_live_loop_jax,
    pm_gpmo_multi_initial_state,
    pm_gpmo_multi_live_loop_jax,
    pm_gpmo_multi_never_stop,
    pm_gpmo_never_stop,
)


def _baseline_problem() -> tuple[jax.Array, jax.Array, GPMOBaselineSpec]:
    ndipoles = 6
    A = jnp.eye(3 * ndipoles, dtype=jnp.float64)
    b = jnp.asarray(
        [
            1.7,
            -0.2,
            0.1,
            -1.4,
            0.3,
            0.2,
            0.1,
            1.2,
            -0.4,
            0.2,
            0.1,
            -1.1,
            0.9,
            0.1,
            0.2,
            -0.8,
            0.3,
            0.1,
        ],
        dtype=jnp.float64,
    )
    spec = GPMOBaselineSpec(
        m_maxima=jnp.ones((ndipoles,), dtype=jnp.float64),
        reg_l2=jnp.asarray(0.0, dtype=jnp.float64),
        single_direction=-1,
    )
    return A, b, spec


def _multi_problem() -> tuple[jax.Array, jax.Array, GPMOMultiSpec]:
    ndipoles = 6
    A = jnp.eye(3 * ndipoles, dtype=jnp.float64)
    b = jnp.asarray(
        [
            1.8,
            -0.3,
            0.1,
            1.3,
            -0.2,
            0.2,
            -1.1,
            0.4,
            -0.2,
            -0.9,
            0.3,
            0.1,
            0.6,
            -0.5,
            0.2,
            -0.4,
            0.3,
            -0.1,
        ],
        dtype=jnp.float64,
    )
    spec = GPMOMultiSpec(
        m_maxima=jnp.ones((ndipoles,), dtype=jnp.float64),
        reg_l2=jnp.asarray(0.0, dtype=jnp.float64),
        dipole_grid_xyz=jnp.arange(3 * ndipoles, dtype=jnp.float64).reshape(
            ndipoles,
            3,
        ),
        single_direction=-1,
        Nadjacent=2,
    )
    return A, b, spec


def _backtracking_problem() -> tuple[jax.Array, jax.Array, GPMOBacktrackingSpec]:
    ndipoles = 6
    A = jnp.eye(3 * ndipoles, dtype=jnp.float64)
    b = jnp.asarray(
        [
            1.7,
            -0.2,
            0.1,
            1.2,
            -0.1,
            0.3,
            -1.1,
            0.4,
            -0.2,
            -0.9,
            0.2,
            0.1,
            0.8,
            -0.5,
            0.2,
            -0.6,
            0.3,
            -0.1,
        ],
        dtype=jnp.float64,
    )
    spec = GPMOBacktrackingSpec(
        m_maxima=jnp.ones((ndipoles,), dtype=jnp.float64),
        reg_l2=jnp.asarray(0.0, dtype=jnp.float64),
        dipole_grid_xyz=jnp.arange(3 * ndipoles, dtype=jnp.float64).reshape(
            ndipoles,
            3,
        ),
        single_direction=-1,
        Nadjacent=2,
        backtracking=2,
        max_nMagnets=ndipoles,
    )
    return A, b, spec


def _arbvec_problem() -> tuple[jax.Array, jax.Array, GPMOArbVecSpec]:
    ndipoles = 6
    A = jnp.eye(3 * ndipoles, dtype=jnp.float64)
    b = jnp.asarray(
        [
            1.6,
            -0.2,
            0.3,
            -1.4,
            0.1,
            0.4,
            0.9,
            -1.2,
            0.2,
            0.5,
            0.8,
            -1.0,
            -0.7,
            0.4,
            0.6,
            0.3,
            -0.5,
            0.2,
        ],
        dtype=jnp.float64,
    )
    pol_vectors = jnp.asarray(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        ],
        dtype=jnp.float64,
    )
    spec = GPMOArbVecSpec(
        m_maxima=jnp.ones((ndipoles,), dtype=jnp.float64),
        reg_l2=jnp.asarray(0.0, dtype=jnp.float64),
        pol_vectors=pol_vectors,
    )
    return A, b, spec


def _arbvec_backtracking_problem() -> tuple[
    jax.Array, jax.Array, GPMOArbVecBacktrackingSpec, jax.Array
]:
    A, b, arbvec_spec = _arbvec_problem()
    spec = GPMOArbVecBacktrackingSpec(
        m_maxima=arbvec_spec.m_maxima,
        reg_l2=arbvec_spec.reg_l2,
        dipole_grid_xyz=jnp.arange(18, dtype=jnp.float64).reshape(6, 3),
        pol_vectors=arbvec_spec.pol_vectors,
        Nadjacent=2,
        backtracking=2,
        thresh_angle=np.pi,
        max_nMagnets=6,
    )
    x_init = (
        jnp.zeros((6, 3), dtype=jnp.float64).at[0, :].set(spec.pol_vectors[0, 0, :])
    )
    return A, b, spec, x_init


def test_pm_initial_states_allow_strict_host_to_device_transfer_guard():
    ndipoles = 2
    A = np.eye(3 * ndipoles, dtype=np.float64)
    b = np.arange(3 * ndipoles, dtype=np.float64)
    m_maxima = np.ones((ndipoles,), dtype=np.float64)

    multi_spec = GPMOMultiSpec(
        m_maxima=m_maxima,
        reg_l2=np.asarray(0.0, dtype=np.float64),
        dipole_grid_xyz=np.arange(3 * ndipoles, dtype=np.float64).reshape(ndipoles, 3),
        single_direction=-1,
        Nadjacent=1,
    )
    arbvec_spec = GPMOArbVecSpec(
        m_maxima=m_maxima,
        reg_l2=np.asarray(0.0, dtype=np.float64),
        pol_vectors=np.ones((ndipoles, 1, 3), dtype=np.float64),
    )

    with jax.transfer_guard_host_to_device("disallow"):
        baseline_state = pm_gpmo_baseline_initial_state(
            A,
            b,
            ndipoles=ndipoles,
            history_capacity=3,
        )
        multi_state = pm_gpmo_multi_initial_state(
            A,
            b,
            multi_spec,
            history_capacity=3,
        )
        arbvec_state = pm_gpmo_arbvec_initial_state(
            A,
            b,
            arbvec_spec,
            history_capacity=3,
        )
        assert np.asarray(pm_gpmo_never_stop(baseline_state)).item() is False
        assert np.asarray(pm_gpmo_multi_never_stop(multi_state)).item() is False
        assert np.asarray(pm_gpmo_arbvec_never_stop(arbvec_state)).item() is False
        assert baseline_state.x.shape == (ndipoles, 3)
        assert multi_state.selected_groups.shape == (3, 1)
        assert arbvec_state.available.shape == (ndipoles,)


def _run_step_by_step(
    state: PMGPMOLiveState,
    spec: GPMOBaselineSpec,
    A: jax.Array,
    n_steps: int,
) -> PMGPMOLiveState:
    current = state
    for _ in range(n_steps):
        (x, residual, available), trace = gpmo_baseline_step(
            spec,
            (current.x, current.residual, current.available),
            A,
        )
        dipole, component, sign, residual_sq = trace
        index = int(np.asarray(current.steps_taken))
        current = PMGPMOLiveState(
            x=x,
            residual=residual,
            available=available,
            steps_taken=current.steps_taken + jnp.asarray(1, dtype=jnp.int32),
            done=jnp.asarray(False),
            selected_dipoles=current.selected_dipoles.at[index].set(dipole),
            selected_components=current.selected_components.at[index].set(component),
            selected_signs=current.selected_signs.at[index].set(sign),
            residual_history=current.residual_history.at[index].set(residual_sq),
        )
    return current


def _run_arbvec_step_by_step(
    state: PMGPMOArbVecLiveState,
    spec: GPMOArbVecSpec,
    A: jax.Array,
    n_steps: int,
) -> PMGPMOArbVecLiveState:
    current = state
    contributions = _gpmo_arbvec_contributions(A, spec.pol_vectors)
    for _ in range(n_steps):
        (x, residual, available), trace = gpmo_arbvec_step(
            spec,
            (current.x, current.residual, current.available),
            A,
            contributions,
        )
        dipole, vector_index, sign, residual_sq = trace
        index = int(np.asarray(current.steps_taken))
        current = PMGPMOArbVecLiveState(
            x=x,
            residual=residual,
            available=available,
            steps_taken=current.steps_taken + jnp.asarray(1, dtype=jnp.int32),
            done=jnp.asarray(False),
            selected_dipoles=current.selected_dipoles.at[index].set(dipole),
            selected_vector_indices=current.selected_vector_indices.at[index].set(
                vector_index
            ),
            selected_signs=current.selected_signs.at[index].set(sign),
            residual_history=current.residual_history.at[index].set(residual_sq),
        )
    return current


def _run_backtracking_step_by_step(
    state: PMGPMOBacktrackingLiveState,
    spec: GPMOBacktrackingSpec,
    A: jax.Array,
    n_steps: int,
) -> PMGPMOBacktrackingLiveState:
    current = state
    connectivity = gpmo_connectivity_matrix(spec.dipole_grid_xyz)
    history_capacity = int(state.selected_dipoles.shape[0])
    for _ in range(n_steps):
        next_core, trace = gpmo_backtracking_step(
            spec,
            (
                current.x,
                current.residual,
                current.available,
                current.current_signs,
                current.current_components,
                current.selected_dipoles,
                current.selected_components,
                current.selected_signs,
                current.done,
            ),
            A,
            connectivity,
            current.steps_taken,
            K=history_capacity,
        )
        (
            dipole,
            component,
            sign,
            residual_sq,
            x_snapshot,
            num_nonzeros,
            removed_pair_count,
            done_snapshot,
        ) = trace
        (
            x,
            residual,
            available,
            current_signs,
            current_components,
            selected_dipoles,
            selected_components,
            selected_signs,
            done,
        ) = next_core
        index = int(np.asarray(current.steps_taken))
        current = PMGPMOBacktrackingLiveState(
            x=x,
            residual=residual,
            available=available,
            current_signs=current_signs,
            current_components=current_components,
            steps_taken=current.steps_taken + jnp.asarray(1, dtype=jnp.int32),
            done=done,
            selected_dipoles=selected_dipoles,
            selected_components=selected_components,
            selected_signs=selected_signs,
            residual_history=current.residual_history.at[index].set(residual_sq),
            x_history=current.x_history.at[index].set(x_snapshot),
            num_nonzeros_history=current.num_nonzeros_history.at[index].set(
                num_nonzeros
            ),
            removed_pair_count_history=current.removed_pair_count_history.at[index].set(
                removed_pair_count
            ),
            done_history=current.done_history.at[index].set(done_snapshot),
        )
    return current


def _run_arbvec_backtracking_step_by_step(
    state: PMGPMOArbVecBacktrackingLiveState,
    spec: GPMOArbVecBacktrackingSpec,
    A: jax.Array,
    n_steps: int,
) -> PMGPMOArbVecBacktrackingLiveState:
    current = state
    connectivity = gpmo_connectivity_matrix(spec.dipole_grid_xyz)
    cos_thresh_angle = jnp.cos(jnp.asarray(spec.thresh_angle, dtype=A.dtype))
    contributions = _gpmo_arbvec_contributions(A, spec.pol_vectors)
    for _ in range(n_steps):
        next_core, trace = gpmo_arbvec_backtracking_step(
            spec,
            (
                current.x,
                current.residual,
                current.available,
                current.current_vector_indices,
                current.current_signs,
                current.selected_dipoles,
                current.selected_vector_indices,
                current.selected_signs,
                current.done,
            ),
            A,
            connectivity,
            cos_thresh_angle,
            current.steps_taken,
            contributions,
        )
        (
            dipole,
            vector_index,
            sign,
            residual_sq,
            x_snapshot,
            num_nonzeros,
            removed_pair_count,
            done_snapshot,
        ) = trace
        (
            x,
            residual,
            available,
            current_vector_indices,
            current_signs,
            selected_dipoles,
            selected_vector_indices,
            selected_signs,
            done,
        ) = next_core
        index = int(np.asarray(current.steps_taken))
        current = PMGPMOArbVecBacktrackingLiveState(
            x=x,
            residual=residual,
            available=available,
            current_vector_indices=current_vector_indices,
            current_signs=current_signs,
            steps_taken=current.steps_taken + jnp.asarray(1, dtype=jnp.int32),
            done=done,
            selected_dipoles=selected_dipoles,
            selected_vector_indices=selected_vector_indices,
            selected_signs=selected_signs,
            residual_history=current.residual_history.at[index].set(residual_sq),
            x_history=current.x_history.at[index].set(x_snapshot),
            num_nonzeros_history=current.num_nonzeros_history.at[index].set(
                num_nonzeros
            ),
            removed_pair_count_history=current.removed_pair_count_history.at[index].set(
                removed_pair_count
            ),
            done_history=current.done_history.at[index].set(done_snapshot),
            initial_x=current.initial_x,
            initial_residual=current.initial_residual,
            initial_num_nonzero=current.initial_num_nonzero,
        )
    return current


def _run_multi_step_by_step(
    state: PMGPMOMultiLiveState,
    spec: GPMOMultiSpec,
    A: jax.Array,
    n_steps: int,
) -> PMGPMOMultiLiveState:
    current = state
    connectivity = gpmo_connectivity_matrix(spec.dipole_grid_xyz)
    for _ in range(n_steps):
        (x, residual, available), trace = gpmo_multi_step(
            spec,
            (current.x, current.residual, current.available),
            A,
            connectivity,
        )
        seed_dipole, component, sign, residual_sq, selected_group = trace
        index = int(np.asarray(current.steps_taken))
        current = PMGPMOMultiLiveState(
            x=x,
            residual=residual,
            available=available,
            steps_taken=current.steps_taken + jnp.asarray(1, dtype=jnp.int32),
            done=jnp.asarray(False),
            selected_seed_dipoles=current.selected_seed_dipoles.at[index].set(
                seed_dipole
            ),
            selected_components=current.selected_components.at[index].set(component),
            selected_signs=current.selected_signs.at[index].set(sign),
            residual_history=current.residual_history.at[index].set(residual_sq),
            selected_groups=current.selected_groups.at[index].set(selected_group),
        )
    return current


def test_pm_gpmo_live_loop_matches_step_by_step_host_loop() -> None:
    A, b, spec = _baseline_problem()
    initial = pm_gpmo_baseline_initial_state(
        A,
        b,
        ndipoles=spec.m_maxima.shape[0],
        history_capacity=5,
    )

    actual = pm_gpmo_live_loop_jax(initial, spec, A, max_steps=5)
    expected = _run_step_by_step(initial, spec, A, 5)

    np.testing.assert_allclose(np.asarray(actual.x), np.asarray(expected.x))
    np.testing.assert_allclose(
        np.asarray(actual.residual), np.asarray(expected.residual)
    )
    np.testing.assert_array_equal(
        np.asarray(actual.available), np.asarray(expected.available)
    )
    np.testing.assert_array_equal(
        np.asarray(actual.selected_dipoles), np.asarray(expected.selected_dipoles)
    )
    np.testing.assert_array_equal(
        np.asarray(actual.selected_components),
        np.asarray(expected.selected_components),
    )
    np.testing.assert_allclose(
        np.asarray(actual.selected_signs), np.asarray(expected.selected_signs)
    )
    np.testing.assert_allclose(
        np.asarray(actual.residual_history), np.asarray(expected.residual_history)
    )


def test_pm_gpmo_live_loop_restart_continuation_is_exact() -> None:
    A, b, spec = _baseline_problem()
    initial = pm_gpmo_baseline_initial_state(
        A,
        b,
        ndipoles=spec.m_maxima.shape[0],
        history_capacity=5,
    )

    full = pm_gpmo_live_loop_jax(initial, spec, A, max_steps=5)
    partial = pm_gpmo_live_loop_jax(initial, spec, A, max_steps=2)
    continued = pm_gpmo_live_loop_jax(partial, spec, A, max_steps=3)

    np.testing.assert_allclose(np.asarray(continued.x), np.asarray(full.x))
    np.testing.assert_allclose(
        np.asarray(continued.residual), np.asarray(full.residual)
    )
    np.testing.assert_array_equal(
        np.asarray(continued.selected_dipoles), np.asarray(full.selected_dipoles)
    )
    np.testing.assert_array_equal(
        np.asarray(continued.selected_components),
        np.asarray(full.selected_components),
    )
    np.testing.assert_allclose(
        np.asarray(continued.selected_signs), np.asarray(full.selected_signs)
    )
    np.testing.assert_allclose(
        np.asarray(continued.residual_history), np.asarray(full.residual_history)
    )


def test_pm_gpmo_live_loop_applies_pure_prune_rule_before_selection() -> None:
    A, b, spec = _baseline_problem()
    initial = pm_gpmo_baseline_initial_state(
        A,
        b,
        ndipoles=spec.m_maxima.shape[0],
        history_capacity=1,
    )

    def _prune_first_dipole(state: PMGPMOLiveState):
        prune_mask = jnp.zeros_like(state.available).at[0, :].set(True)
        return state, prune_mask

    actual = pm_gpmo_live_loop_jax(
        initial,
        spec,
        A,
        max_steps=1,
        prune_rule=_prune_first_dipole,
    )

    assert int(np.asarray(actual.selected_dipoles[0])) == 1
    assert not bool(np.asarray(actual.available[0]).any())


def test_pm_gpmo_live_loop_rejects_capacity_overrun_before_scan() -> None:
    A, b, spec = _baseline_problem()
    initial = pm_gpmo_baseline_initial_state(
        A,
        b,
        ndipoles=spec.m_maxima.shape[0],
        history_capacity=2,
    )

    with pytest.raises(ValueError, match="history capacity"):
        pm_gpmo_live_loop_jax(initial, spec, A, max_steps=4)


def test_pm_gpmo_live_loop_rejects_malformed_history_capacity() -> None:
    A, b, spec = _baseline_problem()
    initial = pm_gpmo_baseline_initial_state(
        A,
        b,
        ndipoles=spec.m_maxima.shape[0],
        history_capacity=2,
    )
    malformed = PMGPMOLiveState(
        x=initial.x,
        residual=initial.residual,
        available=initial.available,
        steps_taken=initial.steps_taken,
        done=initial.done,
        selected_dipoles=initial.selected_dipoles,
        selected_components=initial.selected_components[:1],
        selected_signs=initial.selected_signs,
        residual_history=initial.residual_history[:1],
    )

    with pytest.raises(ValueError, match="history arrays must share one capacity"):
        pm_gpmo_live_loop_jax(malformed, spec, A, max_steps=2)


def test_pm_gpmo_live_loop_rejects_traced_restart_counter() -> None:
    A, b, spec = _baseline_problem()
    initial = pm_gpmo_baseline_initial_state(
        A,
        b,
        ndipoles=spec.m_maxima.shape[0],
        history_capacity=5,
    )

    @jax.jit
    def _run(state: PMGPMOLiveState) -> PMGPMOLiveState:
        return pm_gpmo_live_loop_jax(state, spec, A, max_steps=5)

    with pytest.raises(ValueError, match="state.steps_taken must be concrete"):
        _run(initial)


def test_pm_gpmo_live_loop_jits_under_transfer_guard() -> None:
    A, b, spec = _baseline_problem()
    initial = pm_gpmo_baseline_initial_state(
        A,
        b,
        ndipoles=spec.m_maxima.shape[0],
        history_capacity=5,
    )

    def _run_impl(A_data: jax.Array) -> PMGPMOLiveState:
        return pm_gpmo_live_loop_jax(initial, spec, A_data, max_steps=5)

    _run = jax.jit(_run_impl)
    compiled = _run(A)
    compiled.x.block_until_ready()

    with jax.transfer_guard("disallow"):
        result = _run(A)
        result.x.block_until_ready()

    assert int(np.asarray(result.steps_taken)) == 5
    assert "scan[" in str(jax.make_jaxpr(_run_impl)(A))


def test_pm_gpmo_multi_live_loop_matches_step_by_step_host_loop() -> None:
    A, b, spec = _multi_problem()
    initial = pm_gpmo_multi_initial_state(A, b, spec, history_capacity=3)

    actual = pm_gpmo_multi_live_loop_jax(initial, spec, A, max_steps=3)
    expected = _run_multi_step_by_step(initial, spec, A, 3)

    np.testing.assert_allclose(np.asarray(actual.x), np.asarray(expected.x))
    np.testing.assert_allclose(
        np.asarray(actual.residual), np.asarray(expected.residual)
    )
    np.testing.assert_array_equal(
        np.asarray(actual.available), np.asarray(expected.available)
    )
    np.testing.assert_array_equal(
        np.asarray(actual.selected_seed_dipoles),
        np.asarray(expected.selected_seed_dipoles),
    )
    np.testing.assert_array_equal(
        np.asarray(actual.selected_components),
        np.asarray(expected.selected_components),
    )
    np.testing.assert_allclose(
        np.asarray(actual.selected_signs), np.asarray(expected.selected_signs)
    )
    np.testing.assert_allclose(
        np.asarray(actual.residual_history), np.asarray(expected.residual_history)
    )
    np.testing.assert_array_equal(
        np.asarray(actual.selected_groups), np.asarray(expected.selected_groups)
    )


def test_pm_gpmo_multi_live_loop_restart_continuation_is_exact() -> None:
    A, b, spec = _multi_problem()
    initial = pm_gpmo_multi_initial_state(A, b, spec, history_capacity=3)

    full = pm_gpmo_multi_live_loop_jax(initial, spec, A, max_steps=3)
    partial = pm_gpmo_multi_live_loop_jax(initial, spec, A, max_steps=1)
    continued = pm_gpmo_multi_live_loop_jax(partial, spec, A, max_steps=2)

    np.testing.assert_allclose(np.asarray(continued.x), np.asarray(full.x))
    np.testing.assert_allclose(
        np.asarray(continued.residual), np.asarray(full.residual)
    )
    np.testing.assert_array_equal(
        np.asarray(continued.selected_seed_dipoles),
        np.asarray(full.selected_seed_dipoles),
    )
    np.testing.assert_array_equal(
        np.asarray(continued.selected_components),
        np.asarray(full.selected_components),
    )
    np.testing.assert_allclose(
        np.asarray(continued.selected_signs), np.asarray(full.selected_signs)
    )
    np.testing.assert_array_equal(
        np.asarray(continued.selected_groups), np.asarray(full.selected_groups)
    )


def test_pm_gpmo_multi_live_loop_rejects_capacity_overrun_before_scan() -> None:
    A, b, spec = _multi_problem()
    initial = pm_gpmo_multi_initial_state(A, b, spec, history_capacity=4)

    with pytest.raises(ValueError, match="K \\* Nadjacent"):
        pm_gpmo_multi_live_loop_jax(initial, spec, A, max_steps=4)


def test_pm_gpmo_multi_live_loop_preserves_core_static_arg_guards() -> None:
    A, b, spec = _multi_problem()
    invalid_nadjacent = GPMOMultiSpec(
        m_maxima=spec.m_maxima,
        reg_l2=spec.reg_l2,
        dipole_grid_xyz=spec.dipole_grid_xyz,
        single_direction=spec.single_direction,
        Nadjacent=0,
    )
    invalid_direction = GPMOMultiSpec(
        m_maxima=spec.m_maxima,
        reg_l2=spec.reg_l2,
        dipole_grid_xyz=spec.dipole_grid_xyz,
        single_direction=99,
        Nadjacent=spec.Nadjacent,
    )

    zero_group_state = pm_gpmo_multi_initial_state(
        A,
        b,
        invalid_nadjacent,
        history_capacity=1,
    )
    with pytest.raises(ValueError, match="Nadjacent must be positive"):
        pm_gpmo_multi_live_loop_jax(
            zero_group_state,
            invalid_nadjacent,
            A,
            max_steps=1,
        )

    invalid_direction_state = pm_gpmo_multi_initial_state(
        A,
        b,
        invalid_direction,
        history_capacity=1,
    )
    with pytest.raises(ValueError, match="single_direction must be"):
        pm_gpmo_multi_live_loop_jax(
            invalid_direction_state,
            invalid_direction,
            A,
            max_steps=1,
        )


def test_pm_gpmo_multi_live_loop_rejects_malformed_group_capacity() -> None:
    A, b, spec = _multi_problem()
    initial = pm_gpmo_multi_initial_state(A, b, spec, history_capacity=3)
    malformed = PMGPMOMultiLiveState(
        x=initial.x,
        residual=initial.residual,
        available=initial.available,
        steps_taken=initial.steps_taken,
        done=initial.done,
        selected_seed_dipoles=initial.selected_seed_dipoles,
        selected_components=initial.selected_components,
        selected_signs=initial.selected_signs,
        residual_history=initial.residual_history,
        selected_groups=initial.selected_groups[:, :1],
    )

    with pytest.raises(ValueError, match="selected_groups must have shape"):
        pm_gpmo_multi_live_loop_jax(malformed, spec, A, max_steps=3)


def test_pm_gpmo_multi_live_loop_rejects_traced_restart_counter() -> None:
    A, b, spec = _multi_problem()
    initial = pm_gpmo_multi_initial_state(A, b, spec, history_capacity=3)

    @jax.jit
    def _run(state: PMGPMOMultiLiveState) -> PMGPMOMultiLiveState:
        return pm_gpmo_multi_live_loop_jax(state, spec, A, max_steps=3)

    with pytest.raises(ValueError, match="state.steps_taken must be concrete"):
        _run(initial)


def test_pm_gpmo_multi_live_loop_jits_under_transfer_guard() -> None:
    A, b, spec = _multi_problem()
    initial = pm_gpmo_multi_initial_state(A, b, spec, history_capacity=3)

    def _run_impl(A_data: jax.Array) -> PMGPMOMultiLiveState:
        return pm_gpmo_multi_live_loop_jax(initial, spec, A_data, max_steps=3)

    _run = jax.jit(_run_impl)
    compiled = _run(A)
    compiled.x.block_until_ready()

    with jax.transfer_guard("disallow"):
        result = _run(A)
        result.x.block_until_ready()

    assert int(np.asarray(result.steps_taken)) == 3
    assert "scan[" in str(jax.make_jaxpr(_run_impl)(A))


def test_pm_gpmo_arbvec_live_loop_matches_step_by_step_host_loop() -> None:
    A, b, spec = _arbvec_problem()
    initial = pm_gpmo_arbvec_initial_state(A, b, spec, history_capacity=4)

    actual = pm_gpmo_arbvec_live_loop_jax(initial, spec, A, max_steps=4)
    expected = _run_arbvec_step_by_step(initial, spec, A, 4)

    np.testing.assert_allclose(np.asarray(actual.x), np.asarray(expected.x))
    np.testing.assert_allclose(
        np.asarray(actual.residual), np.asarray(expected.residual)
    )
    np.testing.assert_array_equal(
        np.asarray(actual.available), np.asarray(expected.available)
    )
    np.testing.assert_array_equal(
        np.asarray(actual.selected_dipoles), np.asarray(expected.selected_dipoles)
    )
    np.testing.assert_array_equal(
        np.asarray(actual.selected_vector_indices),
        np.asarray(expected.selected_vector_indices),
    )
    np.testing.assert_allclose(
        np.asarray(actual.selected_signs), np.asarray(expected.selected_signs)
    )
    np.testing.assert_allclose(
        np.asarray(actual.residual_history), np.asarray(expected.residual_history)
    )


def test_pm_gpmo_arbvec_live_loop_restart_continuation_is_exact() -> None:
    A, b, spec = _arbvec_problem()
    initial = pm_gpmo_arbvec_initial_state(A, b, spec, history_capacity=4)

    full = pm_gpmo_arbvec_live_loop_jax(initial, spec, A, max_steps=4)
    partial = pm_gpmo_arbvec_live_loop_jax(initial, spec, A, max_steps=2)
    continued = pm_gpmo_arbvec_live_loop_jax(partial, spec, A, max_steps=2)

    np.testing.assert_allclose(np.asarray(continued.x), np.asarray(full.x))
    np.testing.assert_allclose(
        np.asarray(continued.residual), np.asarray(full.residual)
    )
    np.testing.assert_array_equal(
        np.asarray(continued.selected_dipoles), np.asarray(full.selected_dipoles)
    )
    np.testing.assert_array_equal(
        np.asarray(continued.selected_vector_indices),
        np.asarray(full.selected_vector_indices),
    )
    np.testing.assert_allclose(
        np.asarray(continued.selected_signs), np.asarray(full.selected_signs)
    )


def test_pm_gpmo_arbvec_live_loop_rejects_capacity_overrun_before_scan() -> None:
    A, b, spec = _arbvec_problem()
    initial = pm_gpmo_arbvec_initial_state(A, b, spec, history_capacity=7)

    with pytest.raises(ValueError, match="K must be <= ndipoles"):
        pm_gpmo_arbvec_live_loop_jax(initial, spec, A, max_steps=7)


def test_pm_gpmo_arbvec_live_loop_preserves_core_static_arg_guards() -> None:
    A, b, spec = _arbvec_problem()
    invalid_vectors = GPMOArbVecSpec(
        m_maxima=spec.m_maxima,
        reg_l2=spec.reg_l2,
        pol_vectors=spec.pol_vectors[:, :, :2],
    )
    initial = pm_gpmo_arbvec_initial_state(A, b, invalid_vectors, history_capacity=1)

    with pytest.raises(ValueError, match="pol_vectors third dimension must be 3"):
        pm_gpmo_arbvec_live_loop_jax(initial, invalid_vectors, A, max_steps=1)


def test_pm_gpmo_arbvec_live_loop_rejects_malformed_history_capacity() -> None:
    A, b, spec = _arbvec_problem()
    initial = pm_gpmo_arbvec_initial_state(A, b, spec, history_capacity=4)
    malformed = PMGPMOArbVecLiveState(
        x=initial.x,
        residual=initial.residual,
        available=initial.available,
        steps_taken=initial.steps_taken,
        done=initial.done,
        selected_dipoles=initial.selected_dipoles,
        selected_vector_indices=initial.selected_vector_indices[:1],
        selected_signs=initial.selected_signs,
        residual_history=initial.residual_history[:1],
    )

    with pytest.raises(ValueError, match="history arrays must share one capacity"):
        pm_gpmo_arbvec_live_loop_jax(malformed, spec, A, max_steps=4)


def test_pm_gpmo_arbvec_live_loop_rejects_traced_restart_counter() -> None:
    A, b, spec = _arbvec_problem()
    initial = pm_gpmo_arbvec_initial_state(A, b, spec, history_capacity=4)

    @jax.jit
    def _run(state: PMGPMOArbVecLiveState) -> PMGPMOArbVecLiveState:
        return pm_gpmo_arbvec_live_loop_jax(state, spec, A, max_steps=4)

    with pytest.raises(ValueError, match="state.steps_taken must be concrete"):
        _run(initial)


def test_pm_gpmo_arbvec_live_loop_jits_under_transfer_guard() -> None:
    A, b, spec = _arbvec_problem()
    initial = pm_gpmo_arbvec_initial_state(A, b, spec, history_capacity=4)

    def _run_impl(A_data: jax.Array) -> PMGPMOArbVecLiveState:
        return pm_gpmo_arbvec_live_loop_jax(initial, spec, A_data, max_steps=4)

    _run = jax.jit(_run_impl)
    compiled = _run(A)
    compiled.x.block_until_ready()

    with jax.transfer_guard("disallow"):
        result = _run(A)
        result.x.block_until_ready()

    assert int(np.asarray(result.steps_taken)) == 4
    assert "scan[" in str(jax.make_jaxpr(_run_impl)(A))


def test_pm_gpmo_backtracking_live_loop_matches_step_by_step_host_loop() -> None:
    A, b, spec = _backtracking_problem()
    initial = pm_gpmo_backtracking_initial_state(A, b, spec, history_capacity=5)

    actual = pm_gpmo_backtracking_live_loop_jax(initial, spec, A, max_steps=5)
    expected = _run_backtracking_step_by_step(initial, spec, A, 5)

    np.testing.assert_allclose(np.asarray(actual.x), np.asarray(expected.x))
    np.testing.assert_allclose(
        np.asarray(actual.residual), np.asarray(expected.residual)
    )
    np.testing.assert_array_equal(
        np.asarray(actual.available), np.asarray(expected.available)
    )
    np.testing.assert_array_equal(
        np.asarray(actual.current_components), np.asarray(expected.current_components)
    )
    np.testing.assert_allclose(
        np.asarray(actual.current_signs), np.asarray(expected.current_signs)
    )
    np.testing.assert_array_equal(
        np.asarray(actual.selected_dipoles), np.asarray(expected.selected_dipoles)
    )
    np.testing.assert_array_equal(
        np.asarray(actual.selected_components),
        np.asarray(expected.selected_components),
    )
    np.testing.assert_allclose(
        np.asarray(actual.selected_signs), np.asarray(expected.selected_signs)
    )
    np.testing.assert_allclose(
        np.asarray(actual.residual_history), np.asarray(expected.residual_history)
    )
    np.testing.assert_allclose(
        np.asarray(actual.x_history), np.asarray(expected.x_history)
    )
    np.testing.assert_array_equal(
        np.asarray(actual.num_nonzeros_history),
        np.asarray(expected.num_nonzeros_history),
    )
    np.testing.assert_array_equal(
        np.asarray(actual.removed_pair_count_history),
        np.asarray(expected.removed_pair_count_history),
    )
    np.testing.assert_array_equal(
        np.asarray(actual.done_history), np.asarray(expected.done_history)
    )


def test_pm_gpmo_backtracking_live_loop_restart_continuation_is_exact() -> None:
    A, b, spec = _backtracking_problem()
    initial = pm_gpmo_backtracking_initial_state(A, b, spec, history_capacity=5)

    full = pm_gpmo_backtracking_live_loop_jax(initial, spec, A, max_steps=5)
    partial = pm_gpmo_backtracking_live_loop_jax(initial, spec, A, max_steps=2)
    continued = pm_gpmo_backtracking_live_loop_jax(partial, spec, A, max_steps=3)

    np.testing.assert_allclose(np.asarray(continued.x), np.asarray(full.x))
    np.testing.assert_allclose(
        np.asarray(continued.residual), np.asarray(full.residual)
    )
    np.testing.assert_array_equal(
        np.asarray(continued.selected_dipoles), np.asarray(full.selected_dipoles)
    )
    np.testing.assert_array_equal(
        np.asarray(continued.selected_components),
        np.asarray(full.selected_components),
    )
    np.testing.assert_allclose(
        np.asarray(continued.selected_signs), np.asarray(full.selected_signs)
    )
    np.testing.assert_allclose(
        np.asarray(continued.x_history), np.asarray(full.x_history)
    )
    np.testing.assert_array_equal(
        np.asarray(continued.removed_pair_count_history),
        np.asarray(full.removed_pair_count_history),
    )


def test_pm_gpmo_backtracking_live_loop_rejects_capacity_overrun_before_scan() -> None:
    A, b, spec = _backtracking_problem()
    initial = pm_gpmo_backtracking_initial_state(A, b, spec, history_capacity=2)

    with pytest.raises(ValueError, match="history capacity"):
        pm_gpmo_backtracking_live_loop_jax(initial, spec, A, max_steps=3)


def test_pm_gpmo_backtracking_live_loop_preserves_core_static_arg_guards() -> None:
    A, b, spec = _backtracking_problem()
    invalid_backtracking = GPMOBacktrackingSpec(
        m_maxima=spec.m_maxima,
        reg_l2=spec.reg_l2,
        dipole_grid_xyz=spec.dipole_grid_xyz,
        single_direction=spec.single_direction,
        Nadjacent=spec.Nadjacent,
        backtracking=0,
        max_nMagnets=spec.max_nMagnets,
    )
    invalid_direction = GPMOBacktrackingSpec(
        m_maxima=spec.m_maxima,
        reg_l2=spec.reg_l2,
        dipole_grid_xyz=spec.dipole_grid_xyz,
        single_direction=99,
        Nadjacent=spec.Nadjacent,
        backtracking=spec.backtracking,
        max_nMagnets=spec.max_nMagnets,
    )

    invalid_backtracking_state = pm_gpmo_backtracking_initial_state(
        A,
        b,
        invalid_backtracking,
        history_capacity=1,
    )
    with pytest.raises(ValueError, match="backtracking must be positive"):
        pm_gpmo_backtracking_live_loop_jax(
            invalid_backtracking_state,
            invalid_backtracking,
            A,
            max_steps=1,
        )

    invalid_direction_state = pm_gpmo_backtracking_initial_state(
        A,
        b,
        invalid_direction,
        history_capacity=1,
    )
    with pytest.raises(ValueError, match="single_direction must be"):
        pm_gpmo_backtracking_live_loop_jax(
            invalid_direction_state,
            invalid_direction,
            A,
            max_steps=1,
        )


def test_pm_gpmo_backtracking_live_loop_rejects_malformed_history_capacity() -> None:
    A, b, spec = _backtracking_problem()
    initial = pm_gpmo_backtracking_initial_state(A, b, spec, history_capacity=5)
    malformed = PMGPMOBacktrackingLiveState(
        x=initial.x,
        residual=initial.residual,
        available=initial.available,
        current_signs=initial.current_signs,
        current_components=initial.current_components,
        steps_taken=initial.steps_taken,
        done=initial.done,
        selected_dipoles=initial.selected_dipoles,
        selected_components=initial.selected_components[:1],
        selected_signs=initial.selected_signs,
        residual_history=initial.residual_history[:1],
        x_history=initial.x_history,
        num_nonzeros_history=initial.num_nonzeros_history,
        removed_pair_count_history=initial.removed_pair_count_history,
        done_history=initial.done_history,
    )

    with pytest.raises(ValueError, match="history arrays must share one capacity"):
        pm_gpmo_backtracking_live_loop_jax(malformed, spec, A, max_steps=5)


def test_pm_gpmo_backtracking_live_loop_rejects_traced_restart_counter() -> None:
    A, b, spec = _backtracking_problem()
    initial = pm_gpmo_backtracking_initial_state(A, b, spec, history_capacity=5)

    @jax.jit
    def _run(state: PMGPMOBacktrackingLiveState) -> PMGPMOBacktrackingLiveState:
        return pm_gpmo_backtracking_live_loop_jax(state, spec, A, max_steps=5)

    with pytest.raises(ValueError, match="state.steps_taken must be concrete"):
        _run(initial)


def test_pm_gpmo_backtracking_live_loop_jits_under_transfer_guard() -> None:
    A, b, spec = _backtracking_problem()
    initial = pm_gpmo_backtracking_initial_state(A, b, spec, history_capacity=5)

    def _run_impl(A_data: jax.Array) -> PMGPMOBacktrackingLiveState:
        return pm_gpmo_backtracking_live_loop_jax(initial, spec, A_data, max_steps=5)

    _run = jax.jit(_run_impl)
    compiled = _run(A)
    compiled.x.block_until_ready()

    with jax.transfer_guard("disallow"):
        result = _run(A)
        result.x.block_until_ready()

    assert int(np.asarray(result.steps_taken)) == 5
    assert "scan[" in str(jax.make_jaxpr(_run_impl)(A))


def test_pm_gpmo_arbvec_backtracking_live_loop_matches_step_by_step_host_loop() -> None:
    A, b, spec, x_init = _arbvec_backtracking_problem()
    initial = pm_gpmo_arbvec_backtracking_initial_state(
        A,
        b,
        spec,
        history_capacity=5,
        x_init=x_init,
    )

    actual = pm_gpmo_arbvec_backtracking_live_loop_jax(
        initial,
        spec,
        A,
        max_steps=5,
    )
    expected = _run_arbvec_backtracking_step_by_step(initial, spec, A, 5)

    np.testing.assert_allclose(np.asarray(actual.initial_x), np.asarray(x_init))
    np.testing.assert_allclose(np.asarray(actual.x), np.asarray(expected.x))
    np.testing.assert_allclose(
        np.asarray(actual.residual), np.asarray(expected.residual)
    )
    np.testing.assert_array_equal(
        np.asarray(actual.available), np.asarray(expected.available)
    )
    np.testing.assert_array_equal(
        np.asarray(actual.current_vector_indices),
        np.asarray(expected.current_vector_indices),
    )
    np.testing.assert_allclose(
        np.asarray(actual.current_signs), np.asarray(expected.current_signs)
    )
    np.testing.assert_array_equal(
        np.asarray(actual.selected_dipoles), np.asarray(expected.selected_dipoles)
    )
    np.testing.assert_array_equal(
        np.asarray(actual.selected_vector_indices),
        np.asarray(expected.selected_vector_indices),
    )
    np.testing.assert_allclose(
        np.asarray(actual.selected_signs), np.asarray(expected.selected_signs)
    )
    np.testing.assert_allclose(
        np.asarray(actual.residual_history), np.asarray(expected.residual_history)
    )
    np.testing.assert_allclose(
        np.asarray(actual.x_history), np.asarray(expected.x_history)
    )
    np.testing.assert_array_equal(
        np.asarray(actual.num_nonzeros_history),
        np.asarray(expected.num_nonzeros_history),
    )
    np.testing.assert_array_equal(
        np.asarray(actual.removed_pair_count_history),
        np.asarray(expected.removed_pair_count_history),
    )
    np.testing.assert_array_equal(
        np.asarray(actual.done_history), np.asarray(expected.done_history)
    )


def test_pm_gpmo_arbvec_backtracking_live_loop_restart_continuation_is_exact() -> None:
    A, b, spec, x_init = _arbvec_backtracking_problem()
    initial = pm_gpmo_arbvec_backtracking_initial_state(
        A,
        b,
        spec,
        history_capacity=5,
        x_init=x_init,
    )

    full = pm_gpmo_arbvec_backtracking_live_loop_jax(initial, spec, A, max_steps=5)
    partial = pm_gpmo_arbvec_backtracking_live_loop_jax(initial, spec, A, max_steps=2)
    continued = pm_gpmo_arbvec_backtracking_live_loop_jax(
        partial,
        spec,
        A,
        max_steps=3,
    )

    np.testing.assert_allclose(np.asarray(continued.x), np.asarray(full.x))
    np.testing.assert_allclose(
        np.asarray(continued.residual), np.asarray(full.residual)
    )
    np.testing.assert_array_equal(
        np.asarray(continued.selected_dipoles), np.asarray(full.selected_dipoles)
    )
    np.testing.assert_array_equal(
        np.asarray(continued.selected_vector_indices),
        np.asarray(full.selected_vector_indices),
    )
    np.testing.assert_allclose(
        np.asarray(continued.selected_signs), np.asarray(full.selected_signs)
    )
    np.testing.assert_allclose(
        np.asarray(continued.x_history), np.asarray(full.x_history)
    )
    np.testing.assert_array_equal(
        np.asarray(continued.removed_pair_count_history),
        np.asarray(full.removed_pair_count_history),
    )


def test_pm_gpmo_arbvec_backtracking_live_loop_rejects_capacity_overrun() -> None:
    A, b, spec, _x_init = _arbvec_backtracking_problem()
    initial = pm_gpmo_arbvec_backtracking_initial_state(
        A,
        b,
        spec,
        history_capacity=2,
    )

    with pytest.raises(ValueError, match="history capacity"):
        pm_gpmo_arbvec_backtracking_live_loop_jax(initial, spec, A, max_steps=3)


def test_pm_gpmo_arbvec_backtracking_live_loop_preserves_core_static_arg_guards() -> (
    None
):
    A, b, spec, _x_init = _arbvec_backtracking_problem()
    invalid_vectors = GPMOArbVecBacktrackingSpec(
        m_maxima=spec.m_maxima,
        reg_l2=spec.reg_l2,
        dipole_grid_xyz=spec.dipole_grid_xyz,
        pol_vectors=spec.pol_vectors[:, :, :2],
        Nadjacent=spec.Nadjacent,
        backtracking=spec.backtracking,
        thresh_angle=spec.thresh_angle,
        max_nMagnets=spec.max_nMagnets,
    )
    invalid_threshold = GPMOArbVecBacktrackingSpec(
        m_maxima=spec.m_maxima,
        reg_l2=spec.reg_l2,
        dipole_grid_xyz=spec.dipole_grid_xyz,
        pol_vectors=spec.pol_vectors,
        Nadjacent=spec.Nadjacent,
        backtracking=spec.backtracking,
        thresh_angle=jnp.asarray(spec.thresh_angle),
        max_nMagnets=spec.max_nMagnets,
    )

    with pytest.raises(ValueError, match="pol_vectors third dimension must be 3"):
        pm_gpmo_arbvec_backtracking_initial_state(
            A,
            b,
            invalid_vectors,
            history_capacity=1,
        )

    with pytest.raises(TypeError, match="thresh_angle must be a Python float"):
        pm_gpmo_arbvec_backtracking_initial_state(
            A,
            b,
            invalid_threshold,
            history_capacity=1,
        )


def test_pm_gpmo_arbvec_backtracking_live_loop_rejects_malformed_history() -> None:
    A, b, spec, _x_init = _arbvec_backtracking_problem()
    initial = pm_gpmo_arbvec_backtracking_initial_state(A, b, spec, history_capacity=5)
    malformed = PMGPMOArbVecBacktrackingLiveState(
        x=initial.x,
        residual=initial.residual,
        available=initial.available,
        current_vector_indices=initial.current_vector_indices,
        current_signs=initial.current_signs,
        steps_taken=initial.steps_taken,
        done=initial.done,
        selected_dipoles=initial.selected_dipoles,
        selected_vector_indices=initial.selected_vector_indices[:1],
        selected_signs=initial.selected_signs,
        residual_history=initial.residual_history[:1],
        x_history=initial.x_history,
        num_nonzeros_history=initial.num_nonzeros_history,
        removed_pair_count_history=initial.removed_pair_count_history,
        done_history=initial.done_history,
        initial_x=initial.initial_x,
        initial_residual=initial.initial_residual,
        initial_num_nonzero=initial.initial_num_nonzero,
    )

    with pytest.raises(ValueError, match="history arrays must share one capacity"):
        pm_gpmo_arbvec_backtracking_live_loop_jax(malformed, spec, A, max_steps=5)


def test_pm_gpmo_arbvec_backtracking_live_loop_rejects_traced_restart_counter() -> None:
    A, b, spec, _x_init = _arbvec_backtracking_problem()
    initial = pm_gpmo_arbvec_backtracking_initial_state(A, b, spec, history_capacity=5)

    @jax.jit
    def _run(
        state: PMGPMOArbVecBacktrackingLiveState,
    ) -> PMGPMOArbVecBacktrackingLiveState:
        return pm_gpmo_arbvec_backtracking_live_loop_jax(
            state,
            spec,
            A,
            max_steps=5,
        )

    with pytest.raises(ValueError, match="state.steps_taken must be concrete"):
        _run(initial)


def test_pm_gpmo_arbvec_backtracking_live_loop_jits_under_transfer_guard() -> None:
    A, b, spec, _x_init = _arbvec_backtracking_problem()
    initial = pm_gpmo_arbvec_backtracking_initial_state(A, b, spec, history_capacity=5)

    def _run_impl(A_data: jax.Array) -> PMGPMOArbVecBacktrackingLiveState:
        return pm_gpmo_arbvec_backtracking_live_loop_jax(
            initial,
            spec,
            A_data,
            max_steps=5,
        )

    _run = jax.jit(_run_impl)
    compiled = _run(A)
    compiled.x.block_until_ready()

    with jax.transfer_guard("disallow"):
        result = _run(A)
        result.x.block_until_ready()

    assert int(np.asarray(result.steps_taken)) == 5
    assert "scan[" in str(jax.make_jaxpr(_run_impl)(A))
