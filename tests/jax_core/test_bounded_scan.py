from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from simsopt.jax_core._bounded_scan import bounded_scan_until_done


@dataclass(frozen=True)
class CounterState:
    count: jax.Array
    done: jax.Array
    status: jax.Array


jax.tree_util.register_dataclass(
    CounterState,
    data_fields=["count", "done", "status"],
    meta_fields=[],
)


def _state(*, count: int = 0, done: bool = False, status: int = 0) -> CounterState:
    return CounterState(
        count=jnp.asarray(count, dtype=jnp.int32),
        done=jnp.asarray(done, dtype=jnp.bool_),
        status=jnp.asarray(status, dtype=jnp.int32),
    )


def _count_until_three(current: CounterState, _iteration: jax.Array) -> CounterState:
    next_count = current.count + jnp.asarray(1, dtype=jnp.int32)
    done = next_count >= jnp.asarray(3, dtype=jnp.int32)
    return CounterState(
        count=next_count,
        done=done,
        status=jnp.where(done, jnp.asarray(7, dtype=jnp.int32), current.status),
    )


def test_bounded_scan_until_done_zero_steps_preserves_initial_state():
    result = bounded_scan_until_done(
        _state(count=2, status=5),
        max_steps=0,
        is_done=lambda current: current.done,
        step=_count_until_three,
    )

    assert int(np.asarray(result.count)) == 2
    assert bool(np.asarray(result.done)) is False
    assert int(np.asarray(result.status)) == 5


def test_bounded_scan_until_done_skips_after_early_completion():
    result = bounded_scan_until_done(
        _state(),
        max_steps=10,
        is_done=lambda current: current.done,
        step=_count_until_three,
    )

    assert int(np.asarray(result.count)) == 3
    assert bool(np.asarray(result.done)) is True
    assert int(np.asarray(result.status)) == 7


def test_bounded_scan_until_done_preserves_never_completed_status():
    def step(current: CounterState, _iteration: jax.Array) -> CounterState:
        return CounterState(
            count=current.count + jnp.asarray(1, dtype=jnp.int32),
            done=jnp.asarray(False, dtype=jnp.bool_),
            status=current.status,
        )

    result = bounded_scan_until_done(
        _state(status=-1),
        max_steps=4,
        is_done=lambda current: current.done,
        step=step,
    )

    assert int(np.asarray(result.count)) == 4
    assert bool(np.asarray(result.done)) is False
    assert int(np.asarray(result.status)) == -1


def test_bounded_scan_until_done_jits_under_transfer_guard():
    @jax.jit
    def run(initial: CounterState) -> CounterState:
        return bounded_scan_until_done(
            initial,
            max_steps=10,
            is_done=lambda current: current.done,
            step=_count_until_three,
        )

    initial = _state()
    compiled = run(initial)
    compiled.count.block_until_ready()

    with jax.transfer_guard("disallow"):
        result = run(initial)
        result.count.block_until_ready()

    assert int(np.asarray(result.count)) == 3
    assert "scan[" in str(jax.make_jaxpr(run)(initial))
