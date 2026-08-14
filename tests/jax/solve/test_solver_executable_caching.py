"""Serial JAX problems own their compiled private-solver executables.

``TraceableScalarProblem`` and ``TraceableParametricScalarProblem`` build one
stable device callable in ``__post_init__``.  Marking that callable as cacheable
lets :func:`_cached_private_solver` keep the compiled solver alive across
solves.  These tests pin the reuse, the numerical equivalence with an unmarked
callable, per-problem isolation, and the parametric operand contract that makes
reuse safe when ``set_objective_parameter`` changes the parameter.

Every case runs with ``SIMSOPT_TARGET_LANE_STRICT`` off and on.  Under strict
purity the target lane hands the solver a guard wrapper instead of the problem's
own callable, so the wrapper — memoized on that callable — is what must own the
compiled solvers; a per-solve wrapper would silently discard them.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from simsopt_jax.geo.optimizers._shared import _CACHEABLE_VALUE_AND_GRAD_ATTR
from simsopt_jax.geo.optimizers.optimizer import _STRICT_TARGET_LANE_WRAPPER_ATTR
from simsopt_jax.geo.optimizers.private._common import (
    _PRIVATE_SOLVER_CACHE_ATTR,
    private_optimizer_runtime_is_supported,
)
from simsopt_jax.solve.driver import Driver
from simsopt_jax.solve.serial import (
    TraceableParametricScalarProblem,
    TraceableScalarProblem,
    serial_solve_jax,
)

pytestmark = [
    pytest.mark.private_optimizer_runtime,
    pytest.mark.skipif(
        not private_optimizer_runtime_is_supported(jax.__version__),
        reason="Private solver caching is validated on the pinned JAX runtime.",
    ),
]

_SCALAR_DRIVERS = (Driver.SIMSOPT_BFGS, Driver.SIMSOPT_LBFGSB)
_STRICT_TARGET_LANE_ENV = "SIMSOPT_TARGET_LANE_STRICT"


@pytest.fixture(params=[False, True], ids=["lane_default", "lane_strict"])
def strict_target_lane(request, monkeypatch) -> bool:
    """Run one case with strict target-lane purity requested and one without."""

    monkeypatch.setenv(_STRICT_TARGET_LANE_ENV, "1" if request.param else "0")
    return bool(request.param)


def _weighted_quartic(x: jax.Array) -> jax.Array:
    weights = jnp.arange(1, x.size + 1, dtype=x.dtype)
    shifted = x - jnp.asarray([0.5, -1.5, 2.0], dtype=x.dtype)
    return jnp.sum(weights * shifted**2) + 0.25 * jnp.sum(shifted**4)


def _shifted_quadratic(x: jax.Array) -> jax.Array:
    return jnp.sum((x + jnp.asarray([3.0, -4.0, 5.0], dtype=x.dtype)) ** 2)


def _parametric_objective(x: jax.Array, target: jax.Array) -> jax.Array:
    residual = x - target
    return jnp.vdot(residual, residual).real + jnp.sum(jnp.cos(x * target))


def _start() -> jax.Array:
    return jnp.asarray([3.0, 1.0, 4.0], dtype=jnp.float64)


def _solve(problem, driver: Driver):
    problem.x = _start()
    return serial_solve_jax(problem, driver=driver, max_steps=64)


def _solver_cache_owner(problem, *, strict_target_lane: bool):
    """Return the callable the solver used as its private-solver cache owner."""

    objective = problem._solver_value_and_grad_fn
    guard_wrapper = getattr(objective, _STRICT_TARGET_LANE_WRAPPER_ATTR, None)
    cacheable = bool(getattr(objective, _CACHEABLE_VALUE_AND_GRAD_ATTR, False))
    assert (guard_wrapper is not None) is (strict_target_lane and cacheable), (
        "the strict target-lane guard wrapper must be memoized exactly when "
        f"{_STRICT_TARGET_LANE_ENV} is set and the objective is cacheable, so "
        "both parameters exercise a different cache owner"
    )
    return objective if guard_wrapper is None else guard_wrapper


def _compiled_solvers(problem, *, strict_target_lane: bool) -> dict[object, object]:
    """Return the compiled executables the solver's cache owner currently holds."""

    owner = _solver_cache_owner(problem, strict_target_lane=strict_target_lane)
    return dict(getattr(owner, _PRIVATE_SOLVER_CACHE_ATTR))


@pytest.mark.parametrize("driver", _SCALAR_DRIVERS)
def test_second_scalar_solve_reuses_the_same_compiled_solvers(
    tmp_path,
    monkeypatch,
    driver: Driver,
    strict_target_lane: bool,
) -> None:
    monkeypatch.chdir(tmp_path)
    problem = TraceableScalarProblem(objective_fn=_weighted_quartic, x=_start())

    _solve(problem, driver)
    after_first = _compiled_solvers(problem, strict_target_lane=strict_target_lane)
    _solve(problem, driver)
    after_second = _compiled_solvers(problem, strict_target_lane=strict_target_lane)

    assert after_first, (
        "the problem-owned objective must retain compiled private solvers so a "
        "repeated solve does not recompile them"
    )
    assert after_second.keys() == after_first.keys()
    for cache_key, solver in after_first.items():
        assert after_second[cache_key] is solver, (
            f"solver for {cache_key!r} was rebuilt instead of reused"
        )


@pytest.mark.parametrize("driver", _SCALAR_DRIVERS)
def test_cached_scalar_solve_matches_the_uncached_result_exactly(
    tmp_path,
    monkeypatch,
    driver: Driver,
    strict_target_lane: bool,
) -> None:
    monkeypatch.chdir(tmp_path)
    cached_problem = TraceableScalarProblem(objective_fn=_weighted_quartic, x=_start())
    baseline_problem = TraceableScalarProblem(
        objective_fn=_weighted_quartic,
        x=_start(),
    )
    monkeypatch.delattr(
        baseline_problem._solver_value_and_grad_fn,
        _CACHEABLE_VALUE_AND_GRAD_ATTR,
    )

    _solve(cached_problem, driver)
    cached = _solve(cached_problem, driver)
    baseline = _solve(baseline_problem, driver)

    assert not hasattr(
        _solver_cache_owner(baseline_problem, strict_target_lane=strict_target_lane),
        _PRIVATE_SOLVER_CACHE_ATTR,
    )
    assert bool(jnp.all(cached.x == baseline.x)), (
        "reusing a compiled solver changed the reported minimizer"
    )
    assert cached.fun == baseline.fun
    assert cached.nit == baseline.nit


@pytest.mark.parametrize("driver", _SCALAR_DRIVERS)
def test_distinct_problems_keep_distinct_solutions(
    tmp_path,
    monkeypatch,
    driver: Driver,
    strict_target_lane: bool,
) -> None:
    monkeypatch.chdir(tmp_path)
    quartic_problem = TraceableScalarProblem(objective_fn=_weighted_quartic, x=_start())
    quadratic_problem = TraceableScalarProblem(
        objective_fn=_shifted_quadratic,
        x=_start(),
    )
    isolated_quadratic = TraceableScalarProblem(
        objective_fn=_shifted_quadratic,
        x=_start(),
    )

    _solve(quartic_problem, driver)
    shared_process = _solve(quadratic_problem, driver)
    _solve(quartic_problem, driver)
    isolated = _solve(isolated_quadratic, driver)

    assert (
        _compiled_solvers(quartic_problem, strict_target_lane=strict_target_lane).keys()
        == _compiled_solvers(
            quadratic_problem,
            strict_target_lane=strict_target_lane,
        ).keys()
    )
    assert bool(jnp.all(shared_process.x == isolated.x)), (
        "a second problem must not be answered from the first problem's solver"
    )
    assert bool(
        jnp.allclose(shared_process.x, jnp.asarray([-3.0, 4.0, -5.0]), atol=1.0e-8)
    )


@pytest.mark.parametrize("driver", _SCALAR_DRIVERS)
def test_changed_objective_parameter_reaches_the_cached_solver(
    tmp_path,
    monkeypatch,
    driver: Driver,
    strict_target_lane: bool,
) -> None:
    monkeypatch.chdir(tmp_path)
    first_target = jnp.asarray([1.0, -2.0], dtype=jnp.float64)
    second_target = jnp.asarray([-3.0, 4.0], dtype=jnp.float64)
    start = jnp.asarray([0.3, 0.2], dtype=jnp.float64)

    problem = TraceableParametricScalarProblem(
        objective_fn=_parametric_objective,
        objective_parameter=first_target,
        x=start,
    )
    serial_solve_jax(problem, driver=driver, max_steps=64, require_success=False)
    after_first = _compiled_solvers(problem, strict_target_lane=strict_target_lane)

    problem.x = start
    problem.set_objective_parameter(second_target)
    reused = serial_solve_jax(
        problem,
        driver=driver,
        max_steps=64,
        require_success=False,
    )
    after_second = _compiled_solvers(problem, strict_target_lane=strict_target_lane)

    fresh_problem = TraceableParametricScalarProblem(
        objective_fn=_parametric_objective,
        objective_parameter=second_target,
        x=start,
    )
    fresh = serial_solve_jax(
        fresh_problem,
        driver=driver,
        max_steps=64,
        require_success=False,
    )

    assert after_second.keys() == after_first.keys()
    for cache_key, solver in after_first.items():
        assert after_second[cache_key] is solver
    assert bool(jnp.all(reused.x == fresh.x)), (
        "the cached solver answered with the previous objective_parameter"
    )
    assert reused.fun == fresh.fun
    assert reused.nit == fresh.nit
