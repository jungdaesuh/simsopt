"""Warm calls of the Boozer inner-solve kernels act on their new inputs.

A new coil state, a new matrix or a new adjoint solution of an already-seen
shape must reach the result of a kernel that has already run once: the
``bfgs-ondevice`` runner takes the coils as an argument, the Hager-Higham
condition estimate bounds the new matrix, and the adjoint NaN-on-failure
``cond`` follows the new flag.
"""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import jax
import jax.scipy.linalg as jsp_linalg
import numpy as np
from simsopt_jax.core.field import grouped_coil_set_spec_from_lists
from simsopt_jax.geo.optimizers import linear_solve

from .boozersurface_jax_test_helpers import _bsj, _make_mock_boozer_surface


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


def test_device_bfgs_runner_solves_each_new_coil_state():
    booz = _make_mock_boozer_surface()
    runner = booz._get_device_bfgs_runner(
        True, True, 1.0, maxiter=40, gtol=1e-10, line_search_maxiter=10
    )
    x0 = booz._pack_decision_vector(0.3, 0.05)
    first = runner.minimize(x0, booz.coil_set_spec)
    assert np.all(np.isfinite(first.x))

    for scale in (1.01, 0.99):
        result = runner.minimize(x0, _move_coils(booz, scale))
        assert np.all(np.isfinite(result.x))
        assert not np.array_equal(result.x, first.x), (
            "the coil state must reach the solve"
        )


def test_public_bfgs_ondevice_stage_solves_each_new_coil_state():
    """The public LS BFGS stage: each moved coil state reaches the solve."""
    booz = _make_mock_boozer_surface()
    booz.options["optimizer_backend"] = "ondevice"
    booz.options["limited_memory"] = False
    solved_iota = []
    for scale in (1.0, 1.01, 0.99):
        _move_coils(booz, scale)
        booz.need_to_run_code = True
        res = booz.minimize_boozer_penalty_constraints_LBFGS(
            iota=0.3, G=0.05, tol=1e-10, maxiter=40
        )
        solved_iota.append(float(jax.device_get(res["iota"])))
    assert len(set(solved_iota)) == 3, "each coil state must reach the solve"


def test_condition_estimate_bounds_a_new_matrix_after_a_warm_call():
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
        estimate = linear_solve._dense_matrix_condition_estimate(
            second,
            lu_piv=jsp_linalg.lu_factor(second),
            transpose_operator=transpose_operator,
        )
        exact = np.linalg.cond(second_host.T if transpose_operator else second_host, 1)
        assert 0.0 < float(jax.device_get(estimate)) <= exact * (1.0 + 1e-12)


def test_adjoint_nan_on_failure_cond_follows_each_new_flag():
    solutions = [jax.device_put(np.arange(5.0) + shift) for shift in range(3)]
    flags = [jax.device_put(np.bool_(flag)) for flag in (True, True, False)]
    _bsj._solve_with_nan_on_failure(solutions[0], flags[0])
    kept = _bsj._solve_with_nan_on_failure(solutions[1], flags[1])
    failed = _bsj._solve_with_nan_on_failure(solutions[2], flags[2])
    np.testing.assert_array_equal(jax.device_get(kept), np.arange(5.0) + 1.0)
    assert np.all(np.isnan(jax.device_get(failed)))
