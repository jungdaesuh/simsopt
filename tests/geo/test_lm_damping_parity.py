import numpy as np
import jax
import jax.numpy as jnp
import pytest
from scipy.optimize import least_squares

from simsopt.geo import optimizer_jax as _opt


def _initial_state(residual_fn, x0, *, damping):
    residual, cost, grad, grad_norm_inf, _ = _opt._least_squares_gradient_state(
        residual_fn,
        x0,
    )
    return {
        "x": x0,
        "residual": residual,
        "cost": cost,
        "grad": grad,
        "grad_norm_inf": grad_norm_inf,
        "damping": jnp.asarray(damping, dtype=x0.dtype),
        "delta": _opt._lm_initial_delta(x0, dtype=x0.dtype),
        "nit": jnp.asarray(0, dtype=jnp.int32),
        "status": jnp.asarray(0, dtype=jnp.int32),
        "info": jnp.asarray(0, dtype=jnp.int32),
        "accepted": jnp.asarray(False),
        "success": jnp.asarray(False),
    }


def _single_lm_iteration(residual_fn, state):
    zero = jnp.asarray(0.0, dtype=state["damping"].dtype)
    return _opt._lm_iteration(
        residual_fn,
        state,
        tol=zero,
        gradient_tol=zero,
        ftol=zero,
        xtol=zero,
        maxiter=10,
    )


def test_lm_damping_decreases_by_half_for_good_step():
    def residual_fn(x):
        return x - jnp.asarray([1.0], dtype=jnp.float64)

    state = _initial_state(
        residual_fn,
        jnp.asarray([0.0], dtype=jnp.float64),
        damping=1.0e-3,
    )
    next_state = _single_lm_iteration(residual_fn, state)

    assert bool(next_state["accepted"])
    np.testing.assert_allclose(next_state["damping"], 5.0e-4)


def test_lm_damping_increases_by_two_for_rejected_step(monkeypatch):
    def residual_fn(x):
        return x - jnp.asarray([1.0], dtype=jnp.float64)

    def fake_gmres(_flat_residual_fn, x, grad, _pullback, *, damping, tol):
        del _flat_residual_fn, grad, _pullback, damping, tol
        step = jnp.asarray([-1.0e6], dtype=x.dtype)
        return step, jnp.zeros_like(x), None

    monkeypatch.setattr(_opt, "_gmres_solve_least_squares_system", fake_gmres)
    state = _initial_state(
        residual_fn,
        jnp.asarray([0.0], dtype=jnp.float64),
        damping=1.0e-3,
    )
    next_state = _single_lm_iteration(residual_fn, state)

    assert not bool(next_state["accepted"])
    np.testing.assert_allclose(next_state["damping"], 2.0e-3)


def test_lm_rejects_positive_step_below_minpack_ratio_threshold(monkeypatch):
    def residual_fn(x):
        return x - jnp.asarray([1.0], dtype=jnp.float64)

    def fake_gmres(_flat_residual_fn, x, grad, _pullback, *, damping, tol):
        del _flat_residual_fn, grad, _pullback, damping, tol
        step = jnp.asarray([-1.99995], dtype=x.dtype)
        return step, jnp.zeros_like(x), None

    monkeypatch.setattr(_opt, "_gmres_solve_least_squares_system", fake_gmres)
    x0 = jnp.asarray([0.0], dtype=jnp.float64)
    state = _initial_state(residual_fn, x0, damping=1.0e-3)

    next_state = _single_lm_iteration(residual_fn, state)

    assert not bool(next_state["accepted"])
    np.testing.assert_allclose(next_state["x"], x0)
    np.testing.assert_allclose(next_state["damping"], 2.0e-3)


def test_matrix_free_lm_iteration_count_stays_close_to_scipy_lm():
    A_np = np.asarray([[3.0, 1.0], [1.0, 4.0], [2.0, -1.0]], dtype=float)
    b_np = np.asarray([5.0, 7.0, 0.5], dtype=float)
    scipy_result = least_squares(
        lambda x: A_np @ x - b_np,
        np.zeros(2, dtype=float),
        jac=lambda _x: A_np,
        method="lm",
        ftol=1.0e-8,
        xtol=1.0e-8,
        gtol=1.0e-8,
        max_nfev=200,
    )

    A = jnp.asarray(A_np, dtype=jnp.float64)
    b = jnp.asarray(b_np, dtype=jnp.float64)
    jax_result = _opt.levenberg_marquardt(
        lambda x: A @ x - b,
        jnp.zeros(2, dtype=jnp.float64),
        maxiter=200,
        tol=0.0,
        ftol=1.0e-8,
        xtol=1.0e-8,
    )

    assert scipy_result.success
    assert jax_result["success"]
    assert jax_result["nit"] <= int(np.ceil(1.5 * scipy_result.nfev))
    np.testing.assert_allclose(jax_result["x"], scipy_result.x, atol=1.0e-7)


@pytest.mark.slow
def test_matrix_free_lm_iteration_count_stays_close_on_oversampled_boozer_fixture():
    from benchmarks.benchmark_problem import build_ls_parity_problem
    from simsopt.field.biotsavart_jax_backend import BiotSavartJAX
    from simsopt.geo import Volume
    from simsopt.geo.boozersurface_jax import BoozerSurfaceJAX

    problem = build_ls_parity_problem(ncoils=4, nphi=16, ntheta=8)
    booz = BoozerSurfaceJAX(
        BiotSavartJAX(problem.coils),
        problem.surface,
        Volume(problem.surface),
        problem.vol_target,
        constraint_weight=1.0,
        options={
            "optimizer_backend": "ondevice",
            "least_squares_algorithm": "lm",
            "verbose": False,
        },
    )
    x0 = booz._pack_decision_vector(problem.iota0, problem.G0)
    residual_fn = booz._make_penalty_residual_with(
        True,
        booz.options["weight_inv_modB"],
        booz.constraint_weight,
        hostify_inputs=False,
    )
    jacobian_fn = jax.jit(jax.jacobian(residual_fn))

    def residual_np(x):
        return np.asarray(
            residual_fn(jnp.asarray(x, dtype=jnp.float64)),
            dtype=float,
        )

    def jacobian_np(x):
        return np.asarray(
            jacobian_fn(jnp.asarray(x, dtype=jnp.float64)),
            dtype=float,
        )

    scipy_result = least_squares(
        residual_np,
        np.asarray(x0, dtype=float),
        jac=jacobian_np,
        method="lm",
        ftol=1.0e-8,
        xtol=1.0e-8,
        gtol=1.0e-8,
        max_nfev=1500,
    )
    jax_result = _opt.levenberg_marquardt(
        residual_fn,
        jnp.asarray(x0, dtype=jnp.float64),
        maxiter=1500,
        tol=1.0e-8,
        ftol=1.0e-8,
        xtol=1.0e-8,
        gtol=1.0e-8,
        materialize_dense_linearization=False,
    )

    assert scipy_result.success
    assert jax_result["success"]
    assert jax_result["nit"] <= int(np.ceil(1.5 * scipy_result.nfev))
