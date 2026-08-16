"""Contract tests for the explicit opt-in dense-QR Levenberg-Marquardt route.

Background
----------
The on-device LM family has two inner linear solves:

* ``Driver.SIMSOPT_LM_GMRES`` (``least_squares_algorithm="lm"``) is the
  **default** target lane. Its inner step is matrix-free GMRES against the
  regularized Gauss-Newton operator ``J^T J + lambda I``. It is the byte-equality
  oracle partner of the host-driven ``Driver.SIMSOPT_LM_GMRES_HOST``
  (``least_squares_algorithm="lm"``, ``optimizer_backend="scipy"``), and the
  SHA-bound receipts in this repo pin ``simsopt_lm_gmres`` by name.
* ``Driver.SIMSOPT_LM_QR`` (``least_squares_algorithm="lm-minpack"``) is an
  **opt-in only** lane. Its inner step factorizes the dense damped Jacobian with
  column-pivoted QR (DESC-style factorize-once), never GMRES.

These tests pin the four properties that make the QR lane safe to ship
alongside the default without disturbing it:

1. it is never reachable by default (routing pins),
2. it agrees with the GMRES lane on the *optimum* but is deliberately **not**
   its byte-equality oracle (different linear algebra => different roundoff and
   different iterate trajectories),
3. its dense materialization is refused up front against a declared byte budget,
   and
4. repeated solves reuse one compiled executable instead of retracing per call,
   while problems with different embedded constants keep their own.
"""

import numpy as np
import pytest
import jax.numpy as jnp

from simsopt_jax.geo.optimizers.single_stage_routing import (
    resolve_boozer_least_squares_algorithm,
)
import simsopt_jax.geo.optimizers.optimizer as _opt
from simsopt_jax.solve.dispatch import least_squares
from simsopt_jax.solve.driver import legacy_target_least_squares_method
from simsopt_jax.solve import (
    Driver,
    SimsoptLMGMRESOptions,
    SimsoptLMQROptions,
)


# --------------------------------------------------------------------------
# Reference least-squares fixtures
# --------------------------------------------------------------------------

_FIXTURE_SEED = 20260816


def _linear_fixture():
    """Overdetermined linear least squares with a closed-form optimum."""
    rng = np.random.default_rng(_FIXTURE_SEED)
    matrix = jnp.asarray(rng.standard_normal((40, 8)))
    rhs = jnp.asarray(rng.standard_normal(40))

    def residual(x):
        return matrix @ x - rhs

    optimum = np.linalg.lstsq(np.asarray(matrix), np.asarray(rhs), rcond=None)[0]
    return residual, jnp.zeros(8), optimum


def _nonlinear_fixture():
    """Well-conditioned exponential fit with an exactly recoverable optimum."""
    grid = jnp.linspace(0.0, 3.0, 60)
    truth = jnp.array([2.5, -0.8, 0.4])
    data = truth[0] * jnp.exp(truth[1] * grid) + truth[2]

    def residual(x):
        return x[0] * jnp.exp(x[1] * grid) + x[2] - data

    return residual, jnp.array([1.0, -0.3, 0.0]), np.asarray(truth)


def _solve_both(residual, x0, *, maxiter=400):
    qr = least_squares(
        residual,
        x0,
        driver=Driver.SIMSOPT_LM_QR,
        options=SimsoptLMQROptions(maxiter=maxiter),
    )
    gmres = least_squares(
        residual,
        x0,
        driver=Driver.SIMSOPT_LM_GMRES,
        options=SimsoptLMGMRESOptions(maxiter=maxiter),
    )
    return qr, gmres


# --------------------------------------------------------------------------
# 1. The QR lane is opt-in only: no default reaches it
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "boozer_optimizer_backend",
    ["ondevice", "host-jax", "scipy"],
)
def test_default_least_squares_algorithm_never_resolves_to_the_qr_lane(
    boozer_optimizer_backend,
):
    """With no explicit request, routing must never pick ``lm-minpack``."""
    resolved = resolve_boozer_least_squares_algorithm(boozer_optimizer_backend)

    assert resolved != "lm-minpack"
    assert resolved in {"lm", "quasi-newton"}


def test_qr_lane_requires_an_explicit_algorithm_string():
    """``lm-minpack`` is the only spelling that reaches ``SIMSOPT_LM_QR``."""
    assert (
        resolve_boozer_least_squares_algorithm(
            "ondevice",
            least_squares_algorithm="lm-minpack",
        )
        == "lm-minpack"
    )

    # The default LM spelling keeps routing to the GMRES lane that the
    # byte-equality oracle and the SHA-bound receipts pin.
    assert (
        _opt.resolve_boozer_inner_driver(
            "ondevice",
            limited_memory=False,
            least_squares_algorithm="lm",
        )
        is Driver.SIMSOPT_LM_GMRES
    )
    assert (
        _opt.resolve_boozer_inner_driver(
            "ondevice",
            limited_memory=False,
            least_squares_algorithm="lm-minpack",
        )
        is Driver.SIMSOPT_LM_QR
    )


def test_qr_lane_is_not_in_the_default_algorithm_position():
    """``lm-minpack`` is a valid *choice*, never a default value."""
    assert "lm-minpack" in _opt.VALID_LEAST_SQUARES_ALGORITHMS

    # Every Boozer inner driver whose options carry ``lm-minpack`` must be the
    # QR driver, so the algorithm string cannot leak into another lane.
    for driver, options in _opt._BOOZER_INNER_DRIVER_OPTIONS.items():
        if options.least_squares_algorithm == "lm-minpack":
            assert driver is Driver.SIMSOPT_LM_QR


# --------------------------------------------------------------------------
# 2. Route-string visibility: receipts can pin the lane
# --------------------------------------------------------------------------


def test_result_records_the_qr_route_explicitly():
    """A receipt must be able to read the executed route off the result."""
    residual, x0, _ = _linear_fixture()

    result = least_squares(
        residual,
        x0,
        driver=Driver.SIMSOPT_LM_QR,
        options=SimsoptLMQROptions(maxiter=50),
    )

    assert result.driver is Driver.SIMSOPT_LM_QR
    assert result.driver.value == "simsopt_lm_qr"
    assert legacy_target_least_squares_method(Driver.SIMSOPT_LM_QR) == (
        "lm-minpack-ondevice"
    )


def test_gmres_route_string_is_unchanged():
    """The default lane keeps the exact string the existing receipts pin."""
    residual, x0, _ = _linear_fixture()

    result = least_squares(
        residual,
        x0,
        driver=Driver.SIMSOPT_LM_GMRES,
        options=SimsoptLMGMRESOptions(maxiter=50),
    )

    assert result.driver is Driver.SIMSOPT_LM_GMRES
    assert result.driver.value == "simsopt_lm_gmres"
    assert legacy_target_least_squares_method(Driver.SIMSOPT_LM_GMRES) == "lm-ondevice"


# --------------------------------------------------------------------------
# 3. Correctness: same optimum as the GMRES lane, deliberately not byte-equal
# --------------------------------------------------------------------------


def test_qr_lane_reaches_the_linear_least_squares_optimum():
    residual, x0, optimum = _linear_fixture()
    qr, gmres = _solve_both(residual, x0)

    assert qr.success
    assert gmres.success

    # Both lanes must land on the closed-form ``lstsq`` optimum.
    np.testing.assert_allclose(np.asarray(qr.x), optimum, rtol=0, atol=1e-7)
    np.testing.assert_allclose(np.asarray(gmres.x), optimum, rtol=0, atol=1e-7)

    # ...and therefore on each other, at the same tolerance level.
    np.testing.assert_allclose(np.asarray(qr.x), np.asarray(gmres.x), rtol=0, atol=1e-7)


def test_qr_lane_reaches_the_nonlinear_least_squares_optimum():
    residual, x0, optimum = _nonlinear_fixture()
    qr, gmres = _solve_both(residual, x0)

    assert qr.success
    assert gmres.success

    np.testing.assert_allclose(np.asarray(qr.x), optimum, rtol=0, atol=1e-7)
    np.testing.assert_allclose(np.asarray(gmres.x), optimum, rtol=0, atol=1e-7)
    np.testing.assert_allclose(np.asarray(qr.x), np.asarray(gmres.x), rtol=0, atol=1e-7)

    # Both drive the residual to (numerical) zero on this fixture.
    assert float(qr.fun) < 1e-14
    assert float(gmres.fun) < 1e-14


def test_qr_lane_is_tolerance_equivalent_but_not_a_byte_oracle():
    """The QR lane is NOT the GMRES lane's byte-equality partner.

    This is by construction, not by defect. The GMRES lane solves
    ``(J^T J + lambda I) delta = -J^T r`` matrix-free; the QR lane factorizes the
    augmented matrix ``[J; sqrt(lambda) I]`` against ``[-r; 0]`` with
    column-pivoted QR. Different algebra produces different rounding and a
    different accepted-step trajectory, so the lanes agree on the *optimum* but
    not bit-for-bit. Byte equality on this repo's LM family remains the
    ``lm`` <-> ``lm-ondevice`` (host GMRES <-> on-device GMRES) pair, which this
    opt-in lane leaves untouched.
    """
    residual, x0, _ = _nonlinear_fixture()
    qr, gmres = _solve_both(residual, x0)

    qr_x = np.asarray(qr.x)
    gmres_x = np.asarray(gmres.x)

    # Same optimum...
    np.testing.assert_allclose(qr_x, gmres_x, rtol=0, atol=1e-7)

    # ...but genuinely different arithmetic: the lanes are not bit-identical.
    assert not np.array_equal(qr_x, gmres_x), (
        "QR and GMRES lanes are bit-identical; the opt-in lane is not actually "
        "running a different inner solve."
    )


# --------------------------------------------------------------------------
# 4. Dense materialization is capped up front
# --------------------------------------------------------------------------


def test_qr_lane_refuses_dense_jacobian_over_the_declared_budget():
    """The cap must fail closed with a clear, quantified message."""
    residual, x0, _ = _linear_fixture()

    with pytest.raises(MemoryError, match="max_dense_linearization_bytes"):
        least_squares(
            residual,
            x0,
            driver=Driver.SIMSOPT_LM_QR,
            options=SimsoptLMQROptions(
                maxiter=400,
                max_dense_linearization_bytes=1,
            ),
        )


def test_qr_lane_cap_message_reports_the_required_and_allowed_bytes():
    residual, x0, _ = _linear_fixture()

    with pytest.raises(MemoryError) as excinfo:
        least_squares(
            residual,
            x0,
            driver=Driver.SIMSOPT_LM_QR,
            options=SimsoptLMQROptions(maxiter=1, max_dense_linearization_bytes=1),
        )

    message = str(excinfo.value)
    # 40x8 Jacobian + 8x8 Hessian in float64 == (320 + 64) * 8 == 3072 bytes.
    assert "3072 bytes" in message
    assert "float64" in message
    assert "max_dense_linearization_bytes=1" in message


def test_qr_lane_runs_when_the_dense_jacobian_fits_the_budget():
    residual, x0, optimum = _linear_fixture()

    result = least_squares(
        residual,
        x0,
        driver=Driver.SIMSOPT_LM_QR,
        options=SimsoptLMQROptions(
            maxiter=400,
            # Comfortably above the 3072 bytes this fixture needs.
            max_dense_linearization_bytes=1 << 20,
        ),
    )

    assert result.success
    np.testing.assert_allclose(np.asarray(result.x), optimum, rtol=0, atol=1e-7)


def test_qr_lane_budget_is_the_shared_dense_materialization_convention():
    """The cap reuses the repo-wide ``max_dense_linearization_bytes`` name."""
    fields = SimsoptLMQROptions().__dataclass_fields__

    assert "max_dense_linearization_bytes" in fields
    # Unset by default: callers declare the budget they are willing to spend,
    # matching SimsoptLMGMRESOptions and OptimistixLMOptions.
    assert SimsoptLMQROptions().max_dense_linearization_bytes is None
    assert SimsoptLMGMRESOptions().max_dense_linearization_bytes is None


# --------------------------------------------------------------------------
# 5. Warm solves reuse one compiled executable
# --------------------------------------------------------------------------
#
# The QR lane routes through the memoized ``_cached_traceable_runner`` seam the
# GMRES lane already uses: the residual callable owns the cache entry, and the
# runner's build-time constant set is the cache key. Before that wiring, every
# call rebuilt the closure and re-entered ``jax.jit`` on a fresh function
# object, so no call ever hit the JIT cache.
#
# Every QR solve calls the residual exactly once *outside* the compiled runner:
# the dense-materialization cap of section 4 is probed with ``jax.eval_shape``
# before anything is materialized. So a warm solve costs exactly that one
# residual trace, and a cold solve costs it plus the runner's own trace.
_PREFLIGHT_RESIDUAL_TRACES = 1


def _trace_counting(residual):
    """Wrap ``residual`` so each Python (i.e. tracing) call bumps ``.traces``.

    Under ``jax.jit`` the Python body runs only while tracing, so the counter
    measures retraces directly rather than wall-clock, which would be flaky.
    """

    def counted(x, *args):
        counted.traces += 1
        return residual(x, *args)

    counted.traces = 0
    return counted


def _solve_qr(residual, x0, *, maxiter=400, residual_args=()):
    return least_squares(
        residual,
        x0,
        driver=Driver.SIMSOPT_LM_QR,
        options=SimsoptLMQROptions(maxiter=maxiter),
        residual_args=residual_args,
    )


def test_warm_qr_solve_reuses_the_compiled_executable():
    """A second identical solve must not retrace the solver body."""
    residual, x0, optimum = _linear_fixture()
    counted = _trace_counting(residual)

    _solve_qr(counted, x0)
    cold_traces = counted.traces

    warm = _solve_qr(counted, x0)
    warm_traces = counted.traces - cold_traces

    assert cold_traces > _PREFLIGHT_RESIDUAL_TRACES, (
        "the first solve must trace the solver body; it traced the residual "
        f"only {cold_traces} time(s), i.e. no more than the cap preflight"
    )
    assert warm_traces == _PREFLIGHT_RESIDUAL_TRACES, (
        "the warm solve retraced the solver body "
        f"({warm_traces} residual traces, expected only the cap preflight); "
        "the LM_QR lane is not hitting the memoized runner cache"
    )
    np.testing.assert_allclose(np.asarray(warm.x), optimum, rtol=0, atol=1e-7)


def test_qr_runner_is_memoized_per_residual_callable_and_constant_set():
    """The seam returns one runner object, and JAX compiles it once."""
    residual, x0, _optimum = _linear_fixture()

    first = _opt._make_traceable_levenberg_marquardt_minpack_runner(
        residual, 400, 1e-10, 1e-8, 1e-8, 1e-8, False, False
    )
    second = _opt._make_traceable_levenberg_marquardt_minpack_runner(
        residual, 400, 1e-10, 1e-8, 1e-8, 1e-8, False, False
    )
    assert first is second

    # Second, independent evidence: JAX's own JIT cache for that runner holds a
    # single entry however often the runner is invoked.
    first(x0, ())
    assert first._cache_size() == 1
    first(x0, ())
    assert first._cache_size() == 1

    # A different build-time constant is a different executable, but both stay
    # reachable under the same residual callable.
    other_maxiter = _opt._make_traceable_levenberg_marquardt_minpack_runner(
        residual, 7, 1e-10, 1e-8, 1e-8, 1e-8, False, False
    )
    assert other_maxiter is not first
    assert (
        _opt._make_traceable_levenberg_marquardt_minpack_runner(
            residual, 7, 1e-10, 1e-8, 1e-8, 1e-8, False, False
        )
        is other_maxiter
    )

    # ...and a callback-instrumented build is a third, separate executable,
    # because its runner carries the debug-callback effects.
    assert (
        _opt._make_traceable_levenberg_marquardt_minpack_runner(
            residual, 400, 1e-10, 1e-8, 1e-8, 1e-8, True, False
        )
        is not first
    )


def test_qr_solves_with_different_problem_constants_keep_separate_executables():
    """Two residuals with different embedded data must not share a program."""
    linear_residual, linear_x0, linear_optimum = _linear_fixture()
    nonlinear_residual, nonlinear_x0, nonlinear_optimum = _nonlinear_fixture()
    counted_linear = _trace_counting(linear_residual)
    counted_nonlinear = _trace_counting(nonlinear_residual)

    linear = _solve_qr(counted_linear, linear_x0)
    nonlinear = _solve_qr(counted_nonlinear, nonlinear_x0)

    # The second problem traced its own body rather than reusing the first's.
    assert counted_nonlinear.traces > _PREFLIGHT_RESIDUAL_TRACES
    np.testing.assert_allclose(np.asarray(linear.x), linear_optimum, rtol=0, atol=1e-7)
    np.testing.assert_allclose(
        np.asarray(nonlinear.x), nonlinear_optimum, rtol=0, atol=1e-7
    )

    # Re-solving the first problem is still warm: the cache is keyed per
    # residual callable, not clobbered by the intervening solve.
    traces_before = counted_linear.traces
    again = _solve_qr(counted_linear, linear_x0)
    assert counted_linear.traces - traces_before == _PREFLIGHT_RESIDUAL_TRACES
    assert np.array_equal(np.asarray(again.x), np.asarray(linear.x))


def test_qr_lane_reuses_one_executable_across_residual_arg_values():
    """``residual_args`` are runtime arguments, never baked into the program.

    This is the staleness guard: if the args were captured at build time, the
    second solve would silently replay the first problem's answer.
    """
    rng = np.random.default_rng(_FIXTURE_SEED + 1)
    matrix = jnp.asarray(rng.standard_normal((30, 5)))
    first_rhs = jnp.asarray(rng.standard_normal(30))
    second_rhs = jnp.asarray(rng.standard_normal(30))

    def residual(x, rhs):
        return matrix @ x - rhs

    counted = _trace_counting(residual)
    x0 = jnp.zeros(5)

    first = _solve_qr(counted, x0, residual_args=(first_rhs,))
    traces_after_first = counted.traces

    second = _solve_qr(counted, x0, residual_args=(second_rhs,))
    assert counted.traces - traces_after_first == _PREFLIGHT_RESIDUAL_TRACES

    host_matrix = np.asarray(matrix)
    for solved, rhs in ((first, first_rhs), (second, second_rhs)):
        expected = np.linalg.lstsq(host_matrix, np.asarray(rhs), rcond=None)[0]
        np.testing.assert_allclose(np.asarray(solved.x), expected, rtol=0, atol=1e-7)
    assert not np.array_equal(np.asarray(first.x), np.asarray(second.x))


@pytest.mark.parametrize("strict_target_lane", ["0", "1"])
def test_qr_warm_reuse_is_independent_of_the_strict_target_lane_flag(
    monkeypatch,
    strict_target_lane,
):
    """The residual lane never passes through the strict purity wrapper.

    ``wrap_strict_target_lane_value_and_grad`` guards the scalar ``minimize``
    entrypoints only, so ``SIMSOPT_TARGET_LANE_STRICT`` must not interpose a
    per-solve wrapper object between the caller's residual and the runner
    cache. Pinned in both modes because a strict-mode-only wrapper is exactly
    what silently defeats callable-identity cache keys.
    """
    monkeypatch.setenv("SIMSOPT_TARGET_LANE_STRICT", strict_target_lane)
    residual, x0, optimum = _linear_fixture()
    counted = _trace_counting(residual)

    _solve_qr(counted, x0)
    traces_after_cold = counted.traces
    warm = _solve_qr(counted, x0)

    assert counted.traces - traces_after_cold == _PREFLIGHT_RESIDUAL_TRACES
    np.testing.assert_allclose(np.asarray(warm.x), optimum, rtol=0, atol=1e-7)


def test_qr_lane_caches_pytree_decision_vectors_and_returns_the_pytree():
    """A structured ``x0`` flattens inside the trace and still warm-reuses."""
    rng = np.random.default_rng(_FIXTURE_SEED + 2)
    matrix = jnp.asarray(rng.standard_normal((24, 6)))
    rhs = jnp.asarray(rng.standard_normal(24))

    def residual(tree):
        return matrix @ jnp.concatenate((tree["head"], tree["tail"])) - rhs

    counted = _trace_counting(residual)
    x0 = {"head": jnp.zeros(4), "tail": jnp.zeros(2)}

    first = _opt.target_least_squares(
        counted, x0, method="lm-minpack-ondevice", maxiter=400
    )
    traces_after_cold = counted.traces
    second = _opt.target_least_squares(
        counted, x0, method="lm-minpack-ondevice", maxiter=400
    )

    assert counted.traces - traces_after_cold == _PREFLIGHT_RESIDUAL_TRACES
    assert sorted(first.x.keys()) == ["head", "tail"]
    optimum = np.linalg.lstsq(np.asarray(matrix), np.asarray(rhs), rcond=None)[0]
    np.testing.assert_allclose(
        np.concatenate((first.x["head"], first.x["tail"])),
        optimum,
        rtol=0,
        atol=1e-7,
    )
    for key in ("head", "tail"):
        assert np.array_equal(np.asarray(first.x[key]), np.asarray(second.x[key]))


def test_one_runner_keeps_two_decision_structures_apart():
    """Structure is discriminated by JAX, which is why it is not a cache key.

    ``unravel`` is rebuilt inside the trace, so one memoized runner serves every
    decision-vector structure and ``jax.jit``'s own argument signature keeps the
    programs apart. If the in-trace ravel were ever hoisted back out, this test
    would see one program answering for both structures.
    """
    rng = np.random.default_rng(_FIXTURE_SEED + 3)
    matrix = jnp.asarray(rng.standard_normal((12, 3)))
    rhs = jnp.asarray(rng.standard_normal(12))

    def residual(x):
        flat = x if isinstance(x, jnp.ndarray) else jnp.concatenate((x["a"], x["b"]))
        return matrix @ flat - rhs

    flat_solved = _opt.target_least_squares(
        residual, jnp.zeros(3), method="lm-minpack-ondevice", maxiter=100
    )
    tree_solved = _opt.target_least_squares(
        residual,
        {"a": jnp.zeros(2), "b": jnp.zeros(1)},
        method="lm-minpack-ondevice",
        maxiter=100,
    )
    runner = _opt._make_traceable_levenberg_marquardt_minpack_runner(
        residual, 100, 1e-10, 1e-8, 1e-8, 1e-8, False, False
    )

    assert runner._cache_size() == 2, (
        "one compiled program is answering for both decision structures"
    )
    optimum = np.linalg.lstsq(np.asarray(matrix), np.asarray(rhs), rcond=None)[0]
    np.testing.assert_allclose(np.asarray(flat_solved.x), optimum, rtol=0, atol=1e-7)
    np.testing.assert_allclose(
        np.concatenate((tree_solved.x["a"], tree_solved.x["b"])),
        optimum,
        rtol=0,
        atol=1e-7,
    )


_INSTRUMENTED_SOLVE_COUNT = 12


def _compiled_executables_for(runner_cache, residual):
    """Total compiled programs the lane retains for one residual callable.

    Sums ``_cache_size()`` over every memoized runner filed under ``residual``,
    so it catches both a runner that forks its own JIT cache and a lane that
    hands out extra runner objects.
    """
    total = 0
    for callable_cell, runners_by_key in runner_cache.values():
        if callable_cell() is not residual:
            continue
        total += sum(runner._cache_size() for runner in runners_by_key.values())
    return total


@pytest.mark.parametrize(
    ("method", "runner_cache_name"),
    [
        ("lm-minpack-ondevice", "_TRACEABLE_LM_QR_RUNNER_CACHE"),
        ("lm-ondevice", "_TRACEABLE_LM_RUNNER_CACHE"),
    ],
    ids=["lm_qr", "lm_gmres"],
)
def test_instrumented_solves_share_one_compiled_executable(method, runner_cache_name):
    """Callback tokens are traced operands, so they must not fork the cache.

    Tokens are minted per call. Declared ``static_argnums`` they compile a
    fresh executable per instrumented solve, and because the runner itself is
    memoized the lane then *retains* every one of them — an unbounded per-solve
    leak that a single-solve test cannot see. Pinned on both LM lanes.
    """
    residual, x0, _optimum = _nonlinear_fixture()
    runner_cache = getattr(_opt, runner_cache_name)

    steps_per_solve = []
    for _ in range(_INSTRUMENTED_SOLVE_COUNT):
        steps = []
        _opt.target_least_squares(
            residual,
            x0,
            method=method,
            maxiter=400,
            callback=steps.append,
            progress_callback=lambda nit, fun, grad_norm: None,
        )
        steps_per_solve.append(len(steps))

    retained = _compiled_executables_for(runner_cache, residual)
    assert retained == 1, (
        f"{_INSTRUMENTED_SOLVE_COUNT} instrumented {method} solves retain "
        f"{retained} compiled programs; per-call callback tokens are forking "
        "the JIT cache and the memoized runner is holding every entry"
    )
    assert steps_per_solve[0] > 0, "no accepted step reached the callback"
    assert len(set(steps_per_solve)) == 1, (
        f"instrumented solves delivered varying step counts {steps_per_solve}; "
        "the shared executable is not reproducing the callback stream"
    )


def test_qr_lane_step_callback_still_delivers_the_decision_pytree():
    """Routing steps through the traceable-callback registry keeps the shape.

    The compiled runner carries the flat iterate, so the registered adapter has
    to restore the caller's pytree before the user callback sees it.
    """
    rng = np.random.default_rng(_FIXTURE_SEED + 2)
    matrix = jnp.asarray(rng.standard_normal((24, 6)))
    rhs = jnp.asarray(rng.standard_normal(24))

    def residual(tree):
        return matrix @ jnp.concatenate((tree["head"], tree["tail"])) - rhs

    steps = []
    progress = []
    x0 = {"head": jnp.zeros(4), "tail": jnp.zeros(2)}

    result = _opt.target_least_squares(
        residual,
        x0,
        method="lm-minpack-ondevice",
        maxiter=400,
        callback=steps.append,
        progress_callback=lambda nit, fun, grad: progress.append(int(nit)),
    )

    assert steps, "no accepted step was delivered to the callback"
    assert len(progress) == len(steps)
    for step in steps:
        assert sorted(step.keys()) == ["head", "tail"]
        assert np.shape(step["head"]) == (4,)
        assert np.shape(step["tail"]) == (2,)
    for key in ("head", "tail"):
        np.testing.assert_allclose(
            np.asarray(steps[-1][key]), np.asarray(result.x[key]), rtol=0, atol=1e-12
        )
