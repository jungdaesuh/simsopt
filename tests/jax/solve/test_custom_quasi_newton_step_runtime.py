"""RED contracts for the custom quasi-Newton eager step runtime."""

from __future__ import annotations

import gc
import weakref
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import Event, Thread
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from benchmarks.fixtures.custom_quasi_newton import fixture
from jax.flatten_util import ravel_pytree
from simsopt_jax.geo.optimizers._shared import (
    _STRUCTURED_SOLVER_CACHE_TOKEN_ATTR,
    _OptimizerPytreeAdapter,
)
from simsopt_jax.geo.optimizers.private import (
    _bfgs,
    _common,
    _lbfgs,
    _result_converters,
)
from simsopt_jax.geo.optimizers.private._step_runtime import (
    ContinueDecision,
    RuntimeLimits,
    StepOps,
    run_eager,
)
from simsopt_jax.runtime.host_boundary import host_transfer_audit
from simsopt_jax.solve.dispatch import minimize
from simsopt_jax.solve.driver import Driver
from simsopt_jax.solve.simsopt import SimsoptLBFGSBOptions


@dataclass(frozen=True)
class _State:
    value: int


@dataclass(frozen=True)
class _Transition:
    state: _State
    accepted: bool
    terminal: bool
    iteration: int


def test_eager_runtime_materializes_one_observation_per_advance() -> None:
    observations: list[int] = []
    advances = 0

    def advance(state: _State, limits: RuntimeLimits) -> _Transition:
        nonlocal advances
        advances += 1
        next_value = state.value + 1
        return _Transition(
            state=_State(next_value),
            accepted=True,
            terminal=next_value >= limits.maxiter,
            iteration=next_value,
        )

    def sink(observation: _Transition, payload: None) -> ContinueDecision:
        observations.append(observation.iteration)
        return ContinueDecision.CONTINUE

    ops = StepOps(
        initialize=lambda value, _limits: _State(value),
        advance=advance,
        state_of=lambda transition: transition.state,
        observe=lambda transition: transition,
        host_observation=lambda observation: observation,
        payload=lambda _transition: None,
        terminal=lambda observation: observation.terminal,
        initial_terminal=lambda _state, _limits: False,
    )

    final_state = run_eager(
        ops,
        0,
        RuntimeLimits(maxiter=3),
        sink=sink,
    )

    assert final_state == _State(3)
    assert advances == 3
    assert observations == [1, 2, 3]


def test_eager_runtime_preserves_prefix_when_sink_stops() -> None:
    observations: list[int] = []

    def advance(state: _State, _limits: RuntimeLimits) -> _Transition:
        value = state.value + 1
        return _Transition(
            state=_State(value), accepted=True, terminal=False, iteration=value
        )

    def sink(observation: _Transition, payload: None) -> ContinueDecision:
        observations.append(observation.iteration)
        return ContinueDecision.STOP

    ops = StepOps(
        initialize=lambda value, _limits: _State(value),
        advance=advance,
        state_of=lambda transition: transition.state,
        observe=lambda transition: transition,
        host_observation=lambda observation: observation,
        payload=lambda _transition: None,
        terminal=lambda observation: observation.terminal,
        initial_terminal=lambda _state, _limits: False,
    )

    assert run_eager(ops, 0, RuntimeLimits(maxiter=100), sink=sink) == _State(1)
    assert observations == [1]


def test_runtime_limits_are_immutable() -> None:
    limits = RuntimeLimits(maxiter=4, maxfun=9)
    with pytest.raises(AttributeError):
        limits.maxiter = 5  # type: ignore[misc]


def test_custom_solver_entrypoints_expose_eager_step_kernels() -> None:
    assert callable(_bfgs._bfgs_accepted_step)
    assert callable(_lbfgs._lbfgsb_advance_to_next_observable_kernel)
    assert callable(_lbfgs._lbfgsb_result_payload_kernel)


def test_private_solver_cache_is_bounded() -> None:
    assert _common._PRIVATE_SOLVER_CACHE_CAPACITY == 8
    assert callable(_common._private_solver_cache_size)

    owner = lambda x: x
    setattr(owner, _common._CACHEABLE_VALUE_AND_GRAD_ATTR, True)
    first = _common._cached_private_solver(
        owner,
        cache_key=("same",),
        builder=lambda: object(),
    )
    assert (
        _common._cached_private_solver(
            owner,
            cache_key=("same",),
            builder=lambda: object(),
        )
        is first
    )
    for index in range(_common._PRIVATE_SOLVER_CACHE_CAPACITY + 3):
        _common._cached_private_solver(
            owner,
            cache_key=("churn", index),
            builder=lambda: object(),
        )
    assert (
        _common._private_solver_cache_size(owner)
        <= _common._PRIVATE_SOLVER_CACHE_CAPACITY
    )


def test_private_solver_cache_keeps_an_in_use_wrapper_alive_during_eviction() -> None:
    owner = lambda x: x
    setattr(owner, _common._CACHEABLE_VALUE_AND_GRAD_ATTR, True)
    held = _common._cached_private_solver(
        owner,
        cache_key=("held",),
        builder=object,
    )
    assert (
        _common._cached_private_solver(
            owner,
            cache_key=("held",),
            builder=object,
        )
        is held
    )

    entries = [
        _common._cached_private_solver(
            owner,
            cache_key=("churn", index),
            builder=object,
        )
        for index in range(_common._PRIVATE_SOLVER_CACHE_CAPACITY - 1)
    ]

    assert (
        _common._cached_private_solver(
            owner,
            cache_key=("held",),
            builder=object,
        )
        is held
    )
    _common._cached_private_solver(
        owner,
        cache_key=("new",),
        builder=object,
    )
    replacement = _common._cached_private_solver(
        owner,
        cache_key=("churn", 0),
        builder=object,
    )
    assert replacement is not entries[0]
    assert held is not None


def test_private_solver_cache_does_not_keep_owner_alive() -> None:
    owner = lambda x: x
    setattr(owner, _common._CACHEABLE_VALUE_AND_GRAD_ATTR, True)
    owner_reference = weakref.ref(owner)
    _common._cached_private_solver(owner, cache_key=("owner",), builder=object)

    del owner
    gc.collect()

    assert owner_reference() is None


def test_private_solver_cache_distinguishes_compile_identity_fields() -> None:
    owner = lambda x: x
    setattr(owner, _common._CACHEABLE_VALUE_AND_GRAD_ATTR, True)
    keys = (
        ("bfgs", "<f8", (2,), 10, 1.0e-10, 0.0),
        ("lbfgs", "<f8", (2,), 10, 1.0e-10, 0.0),
        ("bfgs", "<f4", (2,), 10, 1.0e-10, 0.0),
        ("bfgs", "<f8", (3,), 10, 1.0e-10, 0.0),
        ("bfgs", "<f8", (2,), 20, 1.0e-10, 0.0),
        ("bfgs", "<f8", (2,), 10, 1.0e-8, 0.0),
        ("bfgs", "<f8", (2,), 10, 1.0e-10, 1.0e-12),
    )
    values = [
        _common._cached_private_solver(
            owner,
            cache_key=key,
            builder=object,
        )
        for key in keys
    ]

    assert len({id(value) for value in values}) == len(keys)
    assert _common._private_solver_cache_size(owner) <= 8


def test_structured_adapter_cache_keys_distinguish_tree_definitions() -> None:
    def make_adapter(tree):
        leaves, tree_def = jax.tree.flatten(tree)
        _flat, unravel = ravel_pytree(tree)
        return _OptimizerPytreeAdapter(
            flat_dtype=np.dtype(np.float64),
            unravel=unravel,
            tree_def=tree_def,
            leaf_signature=tuple(
                (tuple(leaf.shape), np.dtype(leaf.dtype).str) for leaf in leaves
            ),
        )

    tuple_adapter = make_adapter((jnp.ones(2), jnp.ones(1)))
    list_adapter = make_adapter([jnp.ones(2), jnp.ones(1)])

    tuple_prefix = _lbfgs._lbfgsb_cache_context(
        None,
        tuple_adapter,
        "value_and_grad",
        jnp.float64,
        (3,),
    )[1]
    list_prefix = _lbfgs._lbfgsb_cache_context(
        None,
        list_adapter,
        "value_and_grad",
        jnp.float64,
        (3,),
    )[1]

    assert tuple_adapter.solver_cache_key() != list_adapter.solver_cache_key()
    assert tuple_prefix != list_prefix


def test_structured_adapter_cache_keys_remain_execution_distinct() -> None:
    def make_adapter(tree):
        leaves, tree_def = jax.tree.flatten(tree)
        _flat, unravel = ravel_pytree(tree)
        return _OptimizerPytreeAdapter(
            flat_dtype=np.dtype(np.float64),
            unravel=unravel,
            tree_def=tree_def,
            leaf_signature=tuple(
                (tuple(leaf.shape), np.dtype(leaf.dtype).str) for leaf in leaves
            ),
        )

    def value_and_grad(tree):
        return (
            sum(jnp.sum(leaf * leaf) for leaf in jax.tree.leaves(tree)),
            jax.tree.map(lambda leaf: 2.0 * leaf, tree),
        )

    owner = lambda x: x
    setattr(owner, _common._CACHEABLE_VALUE_AND_GRAD_ATTR, True)
    setattr(owner, _STRUCTURED_SOLVER_CACHE_TOKEN_ATTR, "initial")
    x0 = jnp.ones(3, dtype=jnp.float64)
    kernels = []
    outputs = []
    for tree in (
        (jnp.ones(2, dtype=jnp.float64), jnp.ones(1, dtype=jnp.float64)),
        [jnp.ones(2, dtype=jnp.float64), jnp.ones(1, dtype=jnp.float64)],
    ):
        adapter = make_adapter(tree)
        wrapped = adapter.wrap_fun(value_and_grad, value_and_grad=True)
        kernel, consts = _lbfgs._cached_lbfgs_value_and_grad_kernel(
            wrapped,
            cache_owner=owner,
            adapter=adapter,
            objective_mode="value_and_grad",
            dtype=jnp.float64,
            example_x=x0,
        )
        kernels.append(kernel)
        outputs.append(kernel(x0, consts))

    assert kernels[0] is not kernels[1]
    for value, gradient in outputs:
        np.testing.assert_allclose(np.asarray(value), 3.0, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(np.asarray(gradient), 2.0, rtol=0.0, atol=0.0)


def test_structured_solver_cache_token_changes_invalidate_execution_kernel() -> None:
    tree = (jnp.ones(2, dtype=jnp.float64),)
    leaves, tree_def = jax.tree.flatten(tree)
    _flat, unravel = ravel_pytree(tree)
    adapter = _OptimizerPytreeAdapter(
        flat_dtype=np.dtype(np.float64),
        unravel=unravel,
        tree_def=tree_def,
        leaf_signature=tuple(
            (tuple(leaf.shape), np.dtype(leaf.dtype).str) for leaf in leaves
        ),
    )

    def value_and_grad(value):
        return jnp.sum(value[0] * value[0]), (2.0 * value[0],)

    wrapped = adapter.wrap_fun(value_and_grad, value_and_grad=True)

    owner = lambda x: x
    setattr(owner, _common._CACHEABLE_VALUE_AND_GRAD_ATTR, True)
    setattr(owner, _STRUCTURED_SOLVER_CACHE_TOKEN_ATTR, "version-a")
    x0 = jnp.ones(2, dtype=jnp.float64)
    first = _lbfgs._cached_lbfgs_value_and_grad_kernel(
        wrapped,
        cache_owner=owner,
        adapter=adapter,
        objective_mode="value_and_grad",
        dtype=jnp.float64,
        example_x=x0,
    )[0]
    setattr(owner, _STRUCTURED_SOLVER_CACHE_TOKEN_ATTR, "version-b")
    second = _lbfgs._cached_lbfgs_value_and_grad_kernel(
        wrapped,
        cache_owner=owner,
        adapter=adapter,
        objective_mode="value_and_grad",
        dtype=jnp.float64,
        example_x=x0,
    )[0]

    assert first is not second


def test_lbfgsb_eager_budget_changes_reuse_the_same_step_wrapper() -> None:
    def value_and_grad(x):
        return 0.5 * (x[0] ** 2), x

    owner = lambda x: x
    setattr(owner, _common._CACHEABLE_VALUE_AND_GRAD_ATTR, True)
    first = _lbfgs._lbfgsb_advance_to_next_observable_kernel(
        value_and_grad,
        cache_owner=owner,
        cache_key_prefix=("test",),
        maxiter=3,
        maxfun=8,
        entry_kind=_lbfgs._LBFGS_STEP_ENTRY_SEARCH,
        dynamic_limits=True,
    )
    second = _lbfgs._lbfgsb_advance_to_next_observable_kernel(
        value_and_grad,
        cache_owner=owner,
        cache_key_prefix=("test",),
        maxiter=300,
        maxfun=800,
        entry_kind=_lbfgs._LBFGS_STEP_ENTRY_SEARCH,
        dynamic_limits=True,
    )
    assert first is second


def test_lbfgsb_dynamic_budget_changes_compile_once(monkeypatch) -> None:
    def value_and_grad(x):
        return 0.5 * (x[0] ** 2), x

    owner = lambda x: x
    setattr(owner, _common._CACHEABLE_VALUE_AND_GRAD_ATTR, True)
    jit_calls = 0
    original_jit = _lbfgs.jax.jit

    def counted_jit(fun, *args, **kwargs):
        nonlocal jit_calls
        jit_calls += 1
        return original_jit(fun, *args, **kwargs)

    monkeypatch.setattr(_lbfgs.jax, "jit", counted_jit)
    _lbfgs._lbfgsb_advance_to_next_observable_kernel(
        value_and_grad,
        cache_owner=owner,
        cache_key_prefix=("compile-count",),
        maxiter=3,
        maxfun=8,
        entry_kind=_lbfgs._LBFGS_STEP_ENTRY_SEARCH,
        dynamic_limits=True,
    )
    _lbfgs._lbfgsb_advance_to_next_observable_kernel(
        value_and_grad,
        cache_owner=owner,
        cache_key_prefix=("compile-count",),
        maxiter=300,
        maxfun=800,
        entry_kind=_lbfgs._LBFGS_STEP_ENTRY_SEARCH,
        dynamic_limits=True,
    )

    assert jit_calls == 1


def test_lbfgsb_cache_keys_include_initial_state_semantics() -> None:
    owner = lambda x: x
    setattr(owner, _common._CACHEABLE_VALUE_AND_GRAD_ATTR, True)
    common = {
        "cache_owner": owner,
        "cache_key_prefix": ("initial-state-semantics",),
    }
    variants = (
        _lbfgs._lbfgsb_initial_state_kernel(
            m=2, ftol=0.0, gtol=1.0e-5, maxls=20, **common
        ),
        _lbfgs._lbfgsb_initial_state_kernel(
            m=3, ftol=0.0, gtol=1.0e-5, maxls=20, **common
        ),
        _lbfgs._lbfgsb_initial_state_kernel(
            m=2, ftol=1.0e-9, gtol=1.0e-5, maxls=20, **common
        ),
        _lbfgs._lbfgsb_initial_state_kernel(
            m=2, ftol=0.0, gtol=1.0e-6, maxls=20, **common
        ),
        _lbfgs._lbfgsb_initial_state_kernel(
            m=2, ftol=0.0, gtol=1.0e-5, maxls=19, **common
        ),
    )

    assert len({id(kernel) for kernel in variants}) == len(variants)


def test_bfgs_dynamic_maxiter_reuses_eager_executables(monkeypatch) -> None:
    def objective(x):
        return 0.5 * jnp.dot(x, x)

    setattr(objective, _common._CACHEABLE_VALUE_AND_GRAD_ATTR, True)
    jit_calls = 0
    original_jit = _bfgs.jax.jit

    def counted_jit(fun, *args, **kwargs):
        nonlocal jit_calls
        jit_calls += 1
        return original_jit(fun, *args, **kwargs)

    monkeypatch.setattr(_bfgs.jax, "jit", counted_jit)
    x0 = jnp.asarray([1.0, -2.0], dtype=jax.numpy.float64)
    # ``memory_analysis_callback`` pins the eager route, whose step and finalize
    # executables this test claims are reused across iteration budgets.
    _bfgs._minimize_bfgs_private(
        objective,
        x0,
        maxiter=2,
        memory_analysis_callback=lambda report: None,
    )
    first_call_jits = jit_calls
    _bfgs._minimize_bfgs_private(
        objective,
        x0,
        maxiter=7,
        memory_analysis_callback=lambda report: None,
    )

    assert first_call_jits > 0
    assert jit_calls == first_call_jits


def test_private_solver_cache_compilation_is_single_flight() -> None:
    owner = lambda x: x
    setattr(owner, _common._CACHEABLE_VALUE_AND_GRAD_ATTR, True)
    started = Event()
    release = Event()
    calls: list[int] = []
    results: list[object] = []

    def builder() -> object:
        calls.append(1)
        started.set()
        assert release.wait(timeout=5.0)
        return object()

    first_thread = Thread(
        target=lambda: results.append(
            _common._cached_private_solver(
                owner,
                cache_key=("single-flight",),
                builder=builder,
            )
        )
    )
    second_thread = Thread(
        target=lambda: results.append(
            _common._cached_private_solver(
                owner,
                cache_key=("single-flight",),
                builder=builder,
            )
        )
    )
    first_thread.start()
    assert started.wait(timeout=5.0)
    second_thread.start()
    release.set()
    first_thread.join(timeout=5.0)
    second_thread.join(timeout=5.0)
    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert len(calls) == 1
    assert len(results) == 2
    assert results[0] is results[1]


def test_private_solver_cache_removes_failed_compilation() -> None:
    owner = lambda x: x
    setattr(owner, _common._CACHEABLE_VALUE_AND_GRAD_ATTR, True)

    def failing_builder() -> object:
        raise RuntimeError("compile failed")

    with pytest.raises(RuntimeError, match="compile failed"):
        _common._cached_private_solver(
            owner,
            cache_key=("failed",),
            builder=failing_builder,
        )

    recovered = _common._cached_private_solver(
        owner,
        cache_key=("failed",),
        builder=object,
    )
    assert recovered is not None


def test_independent_bfgs_solves_do_not_cross_callbacks() -> None:
    def worker(index: int, center: np.ndarray) -> tuple[int, np.ndarray, list[int]]:
        progress: list[int] = []
        center_device = jnp.asarray(center, dtype=jnp.float64)

        def objective(x):
            delta = x - center_device
            return 0.5 * jnp.dot(delta, delta)

        state = _bfgs._minimize_bfgs_private(
            objective,
            jnp.zeros(2, dtype=jnp.float64),
            maxiter=5,
            progress_callback=lambda iteration, _value, _gradient: progress.append(
                iteration
            ),
        )
        return index, np.asarray(state.x_k), progress

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = (
            executor.submit(worker, 0, np.asarray([1.0, 2.0])),
            executor.submit(worker, 1, np.asarray([-2.0, 3.0])),
        )
        results = [future.result(timeout=30.0) for future in futures]

    for index, actual, progress in results:
        expected = np.asarray([1.0, 2.0]) if index == 0 else np.asarray([-2.0, 3.0])
        np.testing.assert_allclose(actual, expected, atol=1.0e-12)
        assert progress
        assert progress == list(range(1, len(progress) + 1))


def test_independent_lbfgs_solves_do_not_cross_callbacks() -> None:
    def worker(index: int, center: np.ndarray) -> tuple[int, np.ndarray, list[int]]:
        progress: list[int] = []
        center_device = jnp.asarray(center, dtype=jnp.float64)

        def objective(x):
            delta = x - center_device
            return 0.5 * jnp.dot(delta, delta)

        state = _lbfgs._minimize_lbfgs_private(
            objective,
            jnp.zeros(2, dtype=jnp.float64),
            maxiter=5,
            maxcor=3,
            gtol=1.0e-8,
            progress_callback=lambda iteration, _value, _gradient: progress.append(
                iteration
            ),
        )
        return index, np.asarray(state.x_k), progress

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = (
            executor.submit(worker, 0, np.asarray([1.0, 2.0])),
            executor.submit(worker, 1, np.asarray([-2.0, 3.0])),
        )
        results = [future.result(timeout=30.0) for future in futures]

    for index, actual, progress in results:
        expected = np.asarray([1.0, 2.0]) if index == 0 else np.asarray([-2.0, 3.0])
        np.testing.assert_allclose(actual, expected, atol=1.0e-12)
        assert progress
        assert progress == list(range(1, len(progress) + 1))


def test_typed_bfgs_callback_stop_is_returned_as_unsuccessful_result() -> None:
    def value_and_grad(x):
        return 0.5 * jnp.dot(x, x), x

    callback_points: list[np.ndarray] = []

    def callback(event):
        callback_points.append(np.asarray(event.x, dtype=np.float64))
        raise StopIteration

    result = minimize(
        value_and_grad,
        jnp.asarray([1.0, -2.0], dtype=jax.numpy.float64),
        driver=Driver.SIMSOPT_BFGS,
        callback=callback,
    )

    assert len(callback_points) == 1
    np.testing.assert_allclose(result.x, callback_points[0], rtol=0.0, atol=0.0)
    assert result.nit == 1
    assert result.status == 99
    assert result.success is False


def test_typed_lbfgs_callback_stop_is_returned_as_unsuccessful_result() -> None:
    def value_and_grad(x):
        return 0.5 * jnp.dot(x, x), x

    callback_points: list[np.ndarray] = []

    def callback(event):
        callback_points.append(np.asarray(event.x, dtype=np.float64))
        raise StopIteration

    result = minimize(
        value_and_grad,
        jnp.asarray([1.0, -2.0], dtype=jax.numpy.float64),
        driver=Driver.SIMSOPT_LBFGSB,
        options=SimsoptLBFGSBOptions(maxiter=5),
        callback=callback,
    )

    assert len(callback_points) == 1
    np.testing.assert_allclose(result.x, callback_points[0], rtol=0.0, atol=0.0)
    assert result.nit == 1
    assert result.status == 99
    assert result.success is False


def test_bfgs_eager_initial_terminal_state_avoids_a_step() -> None:
    def nonfinite_objective(x):
        del x
        return jnp.asarray(jnp.inf, dtype=jnp.float64)

    result = _bfgs._minimize_bfgs_private(
        nonfinite_objective,
        jnp.asarray([1.0], dtype=jnp.float64),
        maxiter=5,
        gtol=1.0e-8,
    )

    assert int(result.k) == 0
    assert int(result.nfev) == 2
    assert bool(result.failed) is False
    assert (
        _result_converters._private_bfgs_result_to_optimize_result(result).success
        is False
    )


def test_zero_budget_preserves_bfgs_and_lbfgs_limit_timing() -> None:
    objective = lambda x: 0.5 * jnp.dot(x, x)
    value_and_grad = lambda x: (objective(x), x)
    x0 = jnp.asarray([1.0, -2.0], dtype=jax.numpy.float64)

    bfgs_result = _bfgs._minimize_bfgs_private(objective, x0, maxiter=0)
    lbfgs_result = _lbfgs._minimize_lbfgs_private_value_and_grad(
        value_and_grad,
        x0,
        maxiter=0,
    )

    assert int(bfgs_result.k) == 0
    assert int(bfgs_result.status) == 1
    assert bool(bfgs_result.failed) is False
    assert int(lbfgs_result.k) == 1
    assert int(lbfgs_result.status) == 1
    assert bool(lbfgs_result.failed) is True


def test_initial_convergence_avoids_bfgs_and_lbfgs_steps() -> None:
    objective = lambda x: 0.5 * jnp.dot(x, x)
    value_and_grad = lambda x: (objective(x), x)
    x0 = jnp.zeros(2, dtype=jax.numpy.float64)

    bfgs_result = _bfgs._minimize_bfgs_private(objective, x0, maxiter=5)
    lbfgs_result = _lbfgs._minimize_lbfgs_private_value_and_grad(
        value_and_grad,
        x0,
        maxiter=5,
    )

    assert int(bfgs_result.k) == 0
    assert int(bfgs_result.status) == 0
    assert int(bfgs_result.nfev) == 2
    assert bool(bfgs_result.converged) is True
    assert int(lbfgs_result.k) == 0
    assert int(lbfgs_result.status) == 0
    assert int(lbfgs_result.nfev) == 1
    assert bool(lbfgs_result.converged) is True


def test_no_observer_paths_do_not_compute_host_numpy_norm(monkeypatch) -> None:
    def forbidden_numpy_norm(*_args, **_kwargs):
        raise AssertionError("no-observer solver path performed NumPy norm work")

    monkeypatch.setattr(np.linalg, "norm", forbidden_numpy_norm)
    objective = lambda x: 0.5 * jnp.dot(x, x)
    value_and_grad = lambda x: (objective(x), x)
    x0 = jnp.asarray([1.0, -2.0], dtype=jnp.float64)

    bfgs_result = _bfgs._minimize_bfgs_private(objective, x0, maxiter=2)
    lbfgs_result = _lbfgs._minimize_lbfgs_private_value_and_grad(
        value_and_grad,
        x0,
        maxiter=2,
    )

    assert int(bfgs_result.k) > 0
    assert int(lbfgs_result.k) > 0


def test_bfgs_eager_scalar_observation_has_one_host_boundary_per_step(
    monkeypatch,
) -> None:
    calls = 0
    original_host_value = _bfgs.host_value

    def counted_host_value(value):
        nonlocal calls
        calls += 1
        return original_host_value(value)

    monkeypatch.setattr(_bfgs, "host_value", counted_host_value)
    # ``memory_analysis_callback`` routes to the eager driver without installing
    # an observation sink, so the driver materializes the scalar (no iterate)
    # observation payload this test is about.
    result = _bfgs._minimize_bfgs_private(
        lambda x: 0.5 * jnp.dot(x, x),
        jnp.asarray([1.0, -2.0], dtype=jnp.float64),
        maxiter=1,
        memory_analysis_callback=lambda report: None,
    )

    assert int(result.k) == 1
    assert calls == 1


@pytest.mark.parametrize("include_observation", (False, True))
@pytest.mark.parametrize(
    ("terminal", "accepted_new_x"),
    ((False, False), (False, True), (True, False), (True, True)),
)
def test_lbfgsb_status_packs_one_host_transfer_per_step(
    monkeypatch,
    include_observation: bool,
    terminal: bool,
    accepted_new_x: bool,
) -> None:
    calls = 0
    original_host_value = _lbfgs.host_value

    def counted_host_value(value):
        nonlocal calls
        calls += 1
        return original_host_value(value)

    monkeypatch.setattr(_lbfgs, "host_value", counted_host_value)
    state = SimpleNamespace(
        x=jnp.asarray([1.0, -2.0], dtype=jnp.float64),
        f=jnp.asarray(2.5, dtype=jnp.float64),
        g=jnp.asarray([1.0, -2.0], dtype=jnp.float64),
        n_iterations=jnp.asarray(1, dtype=jnp.int32),
        nfev=jnp.asarray(2, dtype=jnp.int32),
        njev=jnp.asarray(2, dtype=jnp.int32),
    )
    step_result = SimpleNamespace(
        state=state,
        terminal=jnp.asarray(terminal),
        entry_kind=jnp.asarray(_lbfgs.lbfgsb.LBFGSB_STEP_ENTRY_SEARCH),
        accepted_new_x=jnp.asarray(accepted_new_x),
    )

    status = _lbfgs._lbfgsb_stepwise_host_status(
        step_result,
        include_observation=include_observation,
    )

    assert calls == 1
    assert status.terminal is terminal
    assert status.accepted_new_x is accepted_new_x


@pytest.mark.parametrize("with_callback", (False, True))
def test_lbfgsb_eager_driver_uses_one_status_boundary_per_transition(
    monkeypatch,
    with_callback: bool,
) -> None:
    calls = 0
    original_host_value = _lbfgs.host_value

    def counted_host_value(value):
        nonlocal calls
        calls += 1
        return original_host_value(value)

    monkeypatch.setattr(_lbfgs, "host_value", counted_host_value)
    callback = (lambda _event: None) if with_callback else None
    result = _lbfgs._minimize_lbfgs_private_value_and_grad(
        lambda x: (0.5 * jnp.dot(x, x), x),
        jnp.asarray([1.0, -2.0], dtype=jnp.float64),
        maxiter=1,
        callback=callback,
    )

    assert int(result.k) == 1
    assert calls == 1


def test_eager_bfgs_transfer_audit_separates_initial_advance_and_final_phases() -> None:
    objective = lambda x: 0.5 * jnp.dot(x, x)
    x0 = jnp.asarray([1.0, -2.0], dtype=jnp.float64)

    with host_transfer_audit() as audit:
        # The eager driver is required for the "advance" phase to exist at all;
        # ``memory_analysis_callback`` selects it without adding a sink, which
        # would relabel the phase "callback".
        state = _bfgs._minimize_bfgs_private(
            objective,
            x0,
            maxiter=2,
            memory_analysis_callback=lambda report: None,
        )
        _result_converters._private_bfgs_result_to_optimize_result(state)

    summary = {entry.phase: entry for entry in audit.summary()}
    assert summary["initialization"].calls >= 1
    assert summary["advance"].calls >= 1
    assert summary["final_result"].calls >= 1
    assert all(entry.leaves > 0 and entry.bytes > 0 for entry in summary.values())


def test_eager_lbfgs_transfer_audit_separates_advance_and_final_phases() -> None:
    objective = lambda x: 0.5 * jnp.dot(x, x)
    x0 = jnp.asarray([1.0, -2.0], dtype=jnp.float64)

    with host_transfer_audit() as audit:
        state = _lbfgs._minimize_lbfgs_private(objective, x0, maxiter=2)
        _result_converters._private_lbfgs_result_to_optimize_result(state)

    summary = {entry.phase: entry for entry in audit.summary()}
    assert summary["advance"].calls >= 1
    assert summary["final_result"].calls >= 1
    assert summary["initialization"].calls == 0, (
        "the eager L-BFGS-B driver seeds its state on device; reading an "
        "initial packet back would be a new host round trip"
    )
    assert all(
        entry.leaves > 0 and entry.bytes > 0
        for entry in summary.values()
        if entry.calls > 0
    )


def test_eager_transfer_audit_marks_observer_payload_as_callback_phase() -> None:
    objective = lambda x: 0.5 * jnp.dot(x, x)
    x0 = jnp.asarray([1.0, -2.0], dtype=jnp.float64)
    progress: list[tuple[int, float, float]] = []

    with host_transfer_audit() as audit:
        _bfgs._minimize_bfgs_private(
            objective,
            x0,
            maxiter=1,
            progress_callback=lambda iteration, value, gradient_inf: progress.append(
                (iteration, value, gradient_inf)
            ),
        )

    summary = {entry.phase: entry for entry in audit.summary()}
    assert progress
    assert summary["callback"].calls == len(progress)
    assert "advance" not in summary


def test_bfgs_nonstopping_progress_observer_preserves_solution() -> None:
    objective = lambda x: 0.5 * jnp.dot(x, x)
    x0 = jnp.asarray([1.0, -2.0], dtype=jnp.float64)
    plain = _bfgs._minimize_bfgs_private(objective, x0, maxiter=3)
    progress: list[tuple[int, float, float]] = []
    observed = _bfgs._minimize_bfgs_private(
        objective,
        x0,
        maxiter=3,
        progress_callback=lambda iteration, value, gradient_inf_norm: progress.append(
            (iteration, value, gradient_inf_norm)
        ),
    )

    np.testing.assert_allclose(observed.x_k, plain.x_k, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(observed.f_k, plain.f_k, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(observed.g_k, plain.g_k, rtol=0.0, atol=0.0)
    assert int(observed.k) == int(plain.k)
    assert int(observed.nfev) == int(plain.nfev)
    assert [entry[0] for entry in progress] == list(range(1, int(observed.k) + 1))


def test_bfgs_fp64_quadratic_freezes_accepted_steps() -> None:
    objective = lambda x: (
        0.5
        * jnp.dot(
            jnp.asarray([1.0, 2.0], dtype=jnp.float64) * x,
            x,
        )
    )
    x0 = jnp.asarray([1.0, -2.0], dtype=jnp.float64)
    accepted_x: list[np.ndarray] = []
    result = _bfgs._minimize_bfgs_private(
        objective,
        x0,
        maxiter=3,
        callback=lambda x: accepted_x.append(np.asarray(x, dtype=np.float64)),
    )

    expected_x = np.asarray(
        [
            [0.7550390187133036, -1.0201560748532146],
            [-0.014692378328741929, -0.0018365472910928382],
            [0.0003424350943112154, 0.00012942059756598226],
        ],
        dtype=np.float64,
    )
    expected_step_norms = np.asarray(
        [1.0100000000000002, 1.2765034601624243, 0.015162804635200464],
        dtype=np.float64,
    )
    np.testing.assert_allclose(np.asarray(accepted_x), expected_x, rtol=0.0, atol=2e-14)
    np.testing.assert_allclose(
        np.linalg.norm(
            np.diff(np.vstack((np.asarray(x0), expected_x)), axis=0), axis=1
        ),
        expected_step_norms,
        rtol=0.0,
        atol=2e-14,
    )
    np.testing.assert_allclose(result.f_k, 7.538058798230142e-08, rtol=0.0, atol=2e-20)
    np.testing.assert_allclose(
        result.g_k,
        np.asarray([0.0003424350943112154, 0.0002588411951319645]),
        rtol=0.0,
        atol=2e-16,
    )
    assert int(result.k) == 3
    assert int(result.nfev) == 5
    assert int(result.ngev) == 5


def test_bfgs_rosenbrock_accepted_steps_match_pre_refactor_contract() -> None:
    x0 = jnp.asarray([-1.2, 1.0], dtype=jnp.float64)
    accepted: list[np.ndarray] = []
    result = _bfgs._minimize_bfgs_private(
        lambda x: 100.0 * (x[1] - x[0] ** 2) ** 2 + (1.0 - x[0]) ** 2,
        x0,
        maxiter=3,
        callback=lambda x: accepted.append(np.asarray(x, dtype=np.float64).copy()),
    )

    np.testing.assert_allclose(
        np.asarray(accepted),
        np.asarray(
            [
                [-0.9140760015803712, 1.1167036728243382],
                [-0.5827730883522614, 0.37423666302094827],
                [-0.5488972623005329, 0.3059645719760432],
            ],
            dtype=np.float64,
        ),
        rtol=0.0,
        atol=2.0e-14,
    )
    assert int(result.k) == 3
    assert int(result.nfev) == 10
    assert int(result.status) == 1
    assert int(result.status) == 1
    assert int(result.line_search_status) == 0


def test_bfgs_traceable_whole_solve_remains_reachable() -> None:
    def objective(x):
        return 0.5 * jnp.dot(x, x)

    solve = jax.jit(lambda x: _bfgs._minimize_bfgs_private(objective, x, maxiter=2))
    state = solve(jnp.asarray([1.0, -2.0], dtype=jax.numpy.float64))

    assert int(state.k) == 2
    assert bool(state.converged) is True


def test_lbfgs_budget_scalars_use_explicit_device_placement() -> None:
    def value_and_grad(x):
        return 0.5 * jnp.dot(x, x), x

    x0 = jnp.asarray([1.0, -2.0], dtype=jax.numpy.float64)
    with jax.transfer_guard("disallow"):
        state = _lbfgs._minimize_lbfgs_private_value_and_grad(
            value_and_grad,
            x0,
            maxiter=5,
            gtol=1.0e-10,
        )

    assert int(state.k) > 0
    assert bool(state.failed) is False


def test_lbfgs_stepwise_observer_uses_one_packed_host_boundary(monkeypatch) -> None:
    calls = 0
    original_host_value = _lbfgs.host_value

    def counted_host_value(value):
        nonlocal calls
        calls += 1
        return original_host_value(value)

    monkeypatch.setattr(_lbfgs, "host_value", counted_host_value)
    state = SimpleNamespace(
        x=jnp.asarray([1.0, -2.0], dtype=jax.numpy.float64),
        f=jnp.asarray(2.5, dtype=jax.numpy.float64),
        g=jnp.asarray([1.0, -2.0], dtype=jax.numpy.float64),
        n_iterations=jnp.asarray(1, dtype=jnp.int32),
        nfev=jnp.asarray(2, dtype=jnp.int32),
        njev=jnp.asarray(2, dtype=jnp.int32),
    )
    step_result = SimpleNamespace(
        state=state,
        terminal=jnp.asarray(False),
        entry_kind=jnp.asarray(_lbfgs.lbfgsb.LBFGSB_STEP_ENTRY_SEARCH),
        accepted_new_x=jnp.asarray(True),
    )

    observation = _lbfgs._lbfgsb_stepwise_host_status(
        step_result,
        include_observation=True,
    )

    assert calls == 1
    assert observation.accepted_new_x is True
    assert observation.n_iterations == 1
    assert observation.x is not None


@pytest.mark.slow
def test_boozer_fixture_exercises_traceable_route() -> None:
    fixture_case = fixture("boozer")
    assert fixture_case.source == "source_owned_boozer_vacuum"
    assert fixture_case.expected_dimension == 65
    metadata = dict(fixture_case.metadata)
    assert metadata["traceable_run_code_success"] is True
    assert fixture_case.value_and_grad is not None
    value, gradient = fixture_case.value_and_grad(
        jnp.asarray(fixture_case.initial, dtype=jax.numpy.float64)
    )
    jax.block_until_ready((value, gradient))
    assert bool(jnp.isfinite(value))
    assert bool(jnp.all(jnp.isfinite(gradient)))
    assert gradient.shape == (65,)


@pytest.mark.slow
def test_coil47_fixture_is_source_owned_and_has_47_dofs() -> None:
    fixture_case = fixture("coil47")
    assert fixture_case.source == "source_owned_fixed_surface_coil_flux"
    assert fixture_case.expected_dimension == 47
    value, gradient = jax.value_and_grad(fixture_case.objective)(
        jnp.asarray(fixture_case.initial, dtype=jax.numpy.float64)
    )
    jax.block_until_ready((value, gradient))
    assert bool(jnp.isfinite(value))
    assert bool(jnp.all(jnp.isfinite(gradient)))
    assert gradient.shape == (47,)
