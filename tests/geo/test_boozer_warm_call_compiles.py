"""Warm calls of the Boozer inner-solve kernels compile nothing.

A new coil state, a new matrix or a new adjoint solution of an already-seen
shape must reuse the compiled program: the ``bfgs-ondevice`` runner takes the
coils as an argument, the Hager-Higham condition estimate is jitted whole, and
the adjoint NaN-on-failure ``cond`` has module-level branches.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import jax
import jax.scipy.linalg as jsp_linalg
import numpy as np
from simsopt_jax.core.field import grouped_coil_set_spec_from_lists
from simsopt_jax.geo.optimizers import linear_solve, optimizer

from .boozersurface_jax_test_helpers import _bsj, _make_mock_boozer_surface

_COMPILE_EVENTS = (
    "/jax/core/compile/jaxpr_trace_duration",
    "/jax/core/compile/backend_compile_duration",
)


@contextmanager
def _compile_events() -> Iterator[list[str]]:
    """Collect the trace and backend-compile events raised inside the block."""
    events: list[str] = []

    def listener(event: str, _duration: float, **_: object) -> None:
        if event in _COMPILE_EVENTS:
            events.append(event)

    jax.monitoring.register_event_duration_secs_listener(listener)
    try:
        yield events
    finally:
        jax.monitoring.unregister_event_duration_listener(listener)


def _move_coils(booz, scale: float):
    """Give the field a new coil state (every coil scaled) and refresh the solver's copy.

    The new state enters through ``coil_set_spec()`` and ``_refresh_coil_data``,
    the path a moved coil takes in ``run_code``.
    """
    coils = booz.biotsavart.coils
    booz.biotsavart._coil_spec = grouped_coil_set_spec_from_lists(
        [np.asarray(coil.curve.gamma()) * scale for coil in coils],
        [np.asarray(coil.curve.gammadash()) * scale for coil in coils],
        [coil.current.get_value() for coil in coils],
    )
    booz._refresh_coil_data()
    return booz.coil_set_spec


def test_device_bfgs_runner_reuses_one_program_across_coil_states():
    booz = _make_mock_boozer_surface()
    runner = booz._get_device_bfgs_runner(
        True, True, 1.0, maxiter=40, gtol=1e-10, line_search_maxiter=10
    )
    assert (
        booz._get_device_bfgs_runner(
            True, True, 1.0, maxiter=40, gtol=1e-10, line_search_maxiter=10
        )
        is runner
    )
    x0 = booz._pack_decision_vector(0.3, 0.05)
    first = runner.minimize(x0, booz.coil_set_spec)
    assert np.all(np.isfinite(first.x))

    for scale in (1.01, 0.99):
        coil_set_spec = _move_coils(booz, scale)
        with _compile_events() as events:
            result = runner.minimize(x0, coil_set_spec)
        assert events == [], f"coil scale {scale} recompiled: {events}"
        assert np.all(np.isfinite(result.x))
        assert not np.array_equal(result.x, first.x), (
            "the coil state must reach the solve"
        )


def test_public_bfgs_ondevice_stage_compiles_nothing_for_moved_coils():
    """The public LS BFGS stage: moved coils reuse the compiled solve."""
    booz = _make_mock_boozer_surface()
    booz.options["optimizer_backend"] = "ondevice"
    booz.options["limited_memory"] = False
    solved_iota = []
    for scale in (1.0, 1.01, 0.99):
        _move_coils(booz, scale)
        booz.need_to_run_code = True
        with _compile_events() as events:
            res = booz.minimize_boozer_penalty_constraints_LBFGS(
                iota=0.3, G=0.05, tol=1e-10, maxiter=40
            )
        assert res["optimizer_method"] == "bfgs-ondevice"
        if scale != 1.0:
            assert events == [], f"coil scale {scale} recompiled: {events}"
        solved_iota.append(float(jax.device_get(res["iota"])))
    assert len(set(solved_iota)) == 3, "each coil state must reach the solve"


def test_device_bfgs_runner_cache_clears_with_the_kernel_bundles():
    booz = _make_mock_boozer_surface()
    booz._get_device_bfgs_runner(
        True, True, 1.0, maxiter=40, gtol=1e-10, line_search_maxiter=10
    )
    assert booz._device_bfgs_runner_cache
    booz._coil_set_static_signature = ("another coil layout",)
    booz._refresh_coil_data()
    assert not booz._device_bfgs_runner_cache
    assert not booz._kernel_bundle_cache


def test_lbfgs_stage_routes_bfgs_ondevice_through_the_runner(monkeypatch):
    booz = _make_mock_boozer_surface()
    booz.options["optimizer_backend"] = "ondevice"
    booz.options["limited_memory"] = False
    booz.options["bfgs_maxiter"] = 40
    calls = []
    minimize = _bsj._DeviceBfgsRunner.minimize

    def recorded(self, x0, coil_set_spec):
        calls.append((self.maxiter, self.gtol, self.line_search_maxiter, coil_set_spec))
        return minimize(self, x0, coil_set_spec)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("bfgs-ondevice must not rebuild a coil-bound closure")

    monkeypatch.setattr(_bsj._DeviceBfgsRunner, "minimize", recorded)
    monkeypatch.setattr(_bsj, "target_minimize", forbidden)

    res = booz.minimize_boozer_penalty_constraints_LBFGS(iota=0.3, G=0.05)

    assert res["optimizer_method"] == "bfgs-ondevice"
    assert calls == [(40, booz.options["bfgs_tol"], 10, booz.coil_set_spec)]


def test_condition_estimate_reuses_its_program_for_a_new_matrix():
    rng = np.random.default_rng(0)
    second_host = np.eye(12) * 5.0 + rng.normal(size=(12, 12))
    first = jax.device_put(np.eye(12) * 4.0 + rng.normal(size=(12, 12)))
    second = jax.device_put(second_host)
    for transpose_operator in (False, True):
        linear_solve._dense_matrix_condition_estimate(
            first,
            lu_piv=jsp_linalg.lu_factor(first),
            transpose_operator=transpose_operator,
        )
        lu_piv = jsp_linalg.lu_factor(second)
        with _compile_events() as events:
            estimate = linear_solve._dense_matrix_condition_estimate(
                second, lu_piv=lu_piv, transpose_operator=transpose_operator
            )
        assert events == []
        exact = np.linalg.cond(second_host.T if transpose_operator else second_host, 1)
        assert 0.0 < float(jax.device_get(estimate)) <= exact * (1.0 + 1e-12)
    assert optimizer._dense_matrix_condition_estimate is (
        linear_solve._dense_matrix_condition_estimate
    )


def test_adjoint_nan_on_failure_cond_reuses_its_program():
    solutions = [jax.device_put(np.arange(5.0) + shift) for shift in range(3)]
    flags = [jax.device_put(np.bool_(flag)) for flag in (True, True, False)]
    _bsj._solve_with_nan_on_failure(solutions[0], flags[0])
    with _compile_events() as events:
        kept = _bsj._solve_with_nan_on_failure(solutions[1], flags[1])
        failed = _bsj._solve_with_nan_on_failure(solutions[2], flags[2])
    assert events == []
    np.testing.assert_array_equal(jax.device_get(kept), np.arange(5.0) + 1.0)
    assert np.all(np.isnan(jax.device_get(failed)))
