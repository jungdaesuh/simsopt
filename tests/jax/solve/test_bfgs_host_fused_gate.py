"""Concrete BFGS callers take the fused loop unless a host observer needs them.

The on-device BFGS solver owns two drivers over the same transitions: an eager
host loop that materializes one observation per iteration, and a fused
``lax.while_loop`` program that syncs once per solve.  Both deliver
``callback``/``progress_callback`` — the fused loop through
``jax.debug.callback`` — but only the eager driver can stop a solve from a
callback that raises ``StopIteration`` and only it reports
``memory_analysis_callback``, so a concrete observed solve routes there.  These
tests pin that gate and the bitwise equality of the two drivers' iterates.

The last test pins the neighbouring strict-target-lane invariant: under
``SIMSOPT_TARGET_LANE_STRICT`` the solvers cache their compiled executables on
the purity guard wrapper, so concurrent solves of one marked problem must agree
on a single wrapper.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from threading import Barrier

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from simsopt_jax.geo.optimizers import optimizer
from simsopt_jax.geo.optimizers._shared import mark_cacheable_jit_value_and_grad
from simsopt_jax.geo.optimizers.optimizer import (
    _STRICT_TARGET_LANE_WRAPPER_ATTR,
    wrap_strict_target_lane_value_and_grad,
)
from simsopt_jax.geo.optimizers.private._bfgs import _minimize_bfgs_private
from simsopt_jax.geo.optimizers.private._common import (
    _PRIVATE_SOLVER_CACHE_ATTR,
    private_optimizer_runtime_is_supported,
)
from simsopt_jax.solve.driver import Driver
from simsopt_jax.solve.serial import TraceableScalarProblem, serial_solve_jax

pytestmark = [
    pytest.mark.private_optimizer_runtime,
    pytest.mark.skipif(
        not private_optimizer_runtime_is_supported(jax.__version__),
        reason="The fused BFGS route is validated on the pinned JAX runtime.",
    ),
]


def _rosenbrock(x: jax.Array) -> jax.Array:
    return jnp.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)


def _start() -> jax.Array:
    return jnp.asarray(
        [-1.2, 1.0, -0.7, 1.3, 0.5, -1.1, 0.9, 1.4, -0.3, 0.8],
        dtype=jnp.float64,
    )


def _cache_key_names(objective) -> set[str]:
    cached = getattr(objective, _PRIVATE_SOLVER_CACHE_ATTR, {})
    return {key[0] for key in cached}


def test_unobserved_solve_matches_the_observed_eager_solve_bitwise() -> None:
    x0 = _start()
    fused = _minimize_bfgs_private(_rosenbrock, x0, maxiter=200, gtol=1.0e-10)
    eager = _minimize_bfgs_private(
        _rosenbrock,
        x0,
        maxiter=200,
        gtol=1.0e-10,
        callback=lambda _x: None,
    )

    assert int(fused.k) >= 50, "the parity case must exercise a long iterate chain"
    assert bool(jnp.all(fused.x_k == eager.x_k)), (
        "the fused loop and the eager host loop produced different iterates"
    )
    assert bool(fused.f_k == eager.f_k)
    assert bool(jnp.all(fused.g_k == eager.g_k))
    assert bool(jnp.all(fused.H_k == eager.H_k))
    assert int(fused.k) == int(eager.k)
    assert int(fused.nfev) == int(eager.nfev)
    assert int(fused.ngev) == int(eager.ngev)
    assert int(fused.status) == int(eager.status)
    assert bool(fused.converged) is bool(eager.converged)


def test_unobserved_solve_compiles_the_fused_program() -> None:
    objective = mark_cacheable_jit_value_and_grad(
        lambda x: jnp.sum(jnp.cosh(x) + 0.25 * x**4)
    )
    x0 = jnp.asarray([1.0, -2.0, 0.5], dtype=jnp.float64)

    _minimize_bfgs_private(objective, x0, maxiter=8)

    assert "bfgs" in _cache_key_names(objective)
    assert "bfgs-eager-runtime" not in _cache_key_names(objective)


def test_host_observed_solve_keeps_the_eager_driver() -> None:
    objective = mark_cacheable_jit_value_and_grad(
        lambda x: jnp.sum(jnp.cosh(x) + 0.25 * x**4)
    )
    x0 = jnp.asarray([1.0, -2.0, 0.5], dtype=jnp.float64)
    iterations: list[int] = []

    state = _minimize_bfgs_private(
        objective,
        x0,
        maxiter=8,
        progress_callback=lambda iteration, _value, _grad_inf: iterations.append(
            iteration
        ),
    )

    assert iterations == list(range(1, int(state.k) + 1))
    assert "bfgs-eager-runtime" in _cache_key_names(objective)


def test_memory_analysis_callback_keeps_the_eager_driver() -> None:
    objective = mark_cacheable_jit_value_and_grad(lambda x: jnp.sum(x * x))
    reports: list[dict[str, int]] = []

    _minimize_bfgs_private(
        objective,
        jnp.asarray([1.0, -2.0], dtype=jnp.float64),
        maxiter=3,
        memory_analysis_callback=reports.append,
    )

    assert len(reports) == 1
    assert reports[0]["compiled_step_peak_live_bytes"] > 0
    assert "bfgs-eager-runtime" in _cache_key_names(objective)
    assert "bfgs" not in _cache_key_names(objective)


def test_fused_program_is_reused_across_iteration_budgets() -> None:
    objective = mark_cacheable_jit_value_and_grad(
        lambda x: jnp.sum(jnp.cosh(x) + 0.25 * x**4)
    )
    x0 = jnp.asarray([1.5, -2.5, 0.75], dtype=jnp.float64)

    short = _minimize_bfgs_private(objective, x0, maxiter=2, gtol=1.0e-14)
    after_short = dict(getattr(objective, _PRIVATE_SOLVER_CACHE_ATTR))
    long = _minimize_bfgs_private(objective, x0, maxiter=9, gtol=1.0e-14)
    after_long = dict(getattr(objective, _PRIVATE_SOLVER_CACHE_ATTR))

    assert int(short.k) == 2
    assert int(long.k) > int(short.k)
    assert after_long.keys() == after_short.keys(), (
        "the iteration budget must be a solver operand, not part of the cache key"
    )
    for cache_key, solver in after_short.items():
        assert after_long[cache_key] is solver


def test_traced_fused_solve_matches_the_concrete_eager_solve_bitwise() -> None:
    """A traced solve stays fused and still lands on the eager driver's iterates.

    The reference leg is forced onto the eager host loop with
    ``memory_analysis_callback``; comparing the traced solve against an
    unobserved concrete solve would compare the fused program with itself.
    """
    x0 = _start()
    eager = _minimize_bfgs_private(
        _rosenbrock,
        x0,
        maxiter=200,
        gtol=1.0e-10,
        memory_analysis_callback=lambda _report: None,
    )
    traced = jax.jit(
        lambda start: _minimize_bfgs_private(
            _rosenbrock,
            start,
            maxiter=200,
            gtol=1.0e-10,
        )
    )(x0)

    assert int(eager.k) >= 50, "the parity case must exercise a long iterate chain"
    np.testing.assert_array_equal(np.asarray(traced.x_k), np.asarray(eager.x_k))
    assert int(traced.k) == int(eager.k)
    assert int(traced.status) == int(eager.status)


def test_traced_solve_delivers_nonstopping_callbacks_without_changing_iterates() -> (
    None
):
    """The fused program emits traced callbacks and matches the unobserved solve.

    ``_bfgs_accepted_step``'s callback hook is reachable only on the
    traced-observed path (``jax.debug.callback`` inside the fused loop); this
    pins that the emissions fire once per accepted step, in order, and leave
    the iterates byte-identical to the unobserved fused solve.
    """
    x0 = _start()
    unobserved = jax.jit(
        lambda start: _minimize_bfgs_private(
            _rosenbrock,
            start,
            maxiter=8,
            gtol=1.0e-14,
        )
    )(x0)
    accepted_iterations: list[int] = []
    observed = jax.jit(
        lambda start: _minimize_bfgs_private(
            _rosenbrock,
            start,
            maxiter=8,
            gtol=1.0e-14,
            progress_callback=lambda iteration, value, gradient_inf: (
                accepted_iterations.append(int(iteration))
            ),
        )
    )(x0)
    jax.block_until_ready(observed.x_k)

    assert accepted_iterations == list(range(1, int(observed.k) + 1))
    np.testing.assert_array_equal(np.asarray(observed.x_k), np.asarray(unobserved.x_k))
    assert int(observed.k) == int(unobserved.k)
    assert int(observed.status) == int(unobserved.status)


def _strict_lane_quartic(x: jax.Array) -> jax.Array:
    shifted = x - jnp.asarray([0.5, -1.5, 2.0], dtype=x.dtype)
    return jnp.sum(shifted**2) + 0.25 * jnp.sum(shifted**4)


def _strict_lane_start() -> jax.Array:
    return jnp.asarray([3.0, 1.0, 4.0], dtype=jnp.float64)


def test_strict_lane_rejects_a_guard_wrapper_inherited_from_another_callable(
    monkeypatch,
) -> None:
    """An inherited memo entry must never answer for a different objective.

    ``functools.wraps`` copies the whole attribute dictionary, so a callable
    built from a marked one carries that one's memoized wrapper; reusing it
    would evaluate the wrong objective under the guard.
    """
    monkeypatch.setenv("SIMSOPT_TARGET_LANE_STRICT", "1")
    owner = mark_cacheable_jit_value_and_grad(lambda x: jnp.sum(x * x))
    owner_guard = wrap_strict_target_lane_value_and_grad(owner)

    @wraps(owner)
    def impostor(x):
        return jnp.sum(x**4)

    assert getattr(impostor, _STRICT_TARGET_LANE_WRAPPER_ATTR) is owner_guard
    impostor_guard = wrap_strict_target_lane_value_and_grad(impostor)

    assert impostor_guard is not owner_guard
    probe = jnp.asarray([2.0], dtype=jnp.float64)
    assert float(impostor_guard(probe)) == float(impostor(probe))


def test_concurrent_strict_lane_solves_share_one_guard_wrapper(
    tmp_path,
    monkeypatch,
) -> None:
    """Two threads entering the strict lane at once share exactly one wrapper.

    Under ``SIMSOPT_TARGET_LANE_STRICT`` the guard wrapper — not the problem's
    own callable — owns the compiled private solvers, so a second wrapper
    reaching the solvers would strand one thread's executables and force a
    recompile on the next solve.  The barrier in ``functools.wraps`` only holds
    both threads inside wrapper *construction*, which is upstream of the
    check-then-install window under ``_STRICT_TARGET_LANE_WRAPPER_LOCK``; it
    cannot schedule that window, so this test pins the observable contract —
    one memoized wrapper answers both threads and owns the compiled solvers —
    rather than proving the lock.  The lock itself is correct by inspection
    against its sibling, ``_cached_private_solver``'s double-checked install.
    Sharing one problem across the two threads deliberately exceeds the
    class's one-solve-at-a-time contract: every assertion here is
    race-insensitive (wrapper identity and cache-key identity; shapes and
    dtypes never vary), and no solve result is read.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SIMSOPT_TARGET_LANE_STRICT", "1")
    problem = TraceableScalarProblem(
        objective_fn=_strict_lane_quartic,
        x=_strict_lane_start(),
    )
    objective = problem._solver_value_and_grad_fn
    inside_construction = Barrier(2, timeout=60.0)
    unsynchronized_wraps = optimizer.wraps

    def synchronized_wraps(target):
        if target is objective:
            inside_construction.wait()
        return unsynchronized_wraps(target)

    monkeypatch.setattr(optimizer, "wraps", synchronized_wraps)

    problem.x = _strict_lane_start()

    def solve_once():
        guard = wrap_strict_target_lane_value_and_grad(objective)
        serial_solve_jax(
            problem,
            driver=Driver.SIMSOPT_BFGS,
            max_steps=32,
            require_success=False,
        )
        return guard

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = (executor.submit(solve_once), executor.submit(solve_once))
        first_guard, second_guard = (future.result(timeout=120.0) for future in futures)

    installed = getattr(objective, _STRICT_TARGET_LANE_WRAPPER_ATTR)
    assert first_guard is second_guard, (
        "concurrent strict-lane entries built two guard wrappers, so the "
        "compiled solvers of one thread are unreachable from the other"
    )
    assert installed is first_guard
    assert installed.__wrapped__ is objective
    assert not hasattr(objective, _PRIVATE_SOLVER_CACHE_ATTR), (
        "the compiled private solvers must live on the guard wrapper the "
        "solvers were handed, not on the problem's own callable"
    )

    compiled = dict(getattr(installed, _PRIVATE_SOLVER_CACHE_ATTR))
    assert compiled, "the concurrent solves left no compiled solver behind"

    problem.x = _strict_lane_start()
    serial_solve_jax(
        problem,
        driver=Driver.SIMSOPT_BFGS,
        max_steps=32,
        require_success=False,
    )
    after_third = dict(getattr(installed, _PRIVATE_SOLVER_CACHE_ATTR))
    assert after_third.keys() == compiled.keys()
    for cache_key, solver in compiled.items():
        assert after_third[cache_key] is solver, (
            f"solver for {cache_key!r} was rebuilt after the concurrent solves"
        )
