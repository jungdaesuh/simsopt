import numpy as np
import pytest
import jax
import jax.numpy as jnp

from benchmarks.validation_ladder_contract import parity_ladder_tolerances
from scipy.optimize import least_squares

from simsopt.geo import optimizer_jax as _opt


_LM_MINPACK_DIRECT_RTOL = 1.0e-10
_LM_MINPACK_DIRECT_ATOL = 1.0e-10
_LM_MINPACK_SOLVER_TOL = 1.0e-12
_BOOZER_BRANCH_STABLE_STATE_ATOL = float(
    parity_ladder_tolerances("branch-stable-resolve")["core_value_rtol"]
)
_BOOZER_BRANCH_STABLE_RESIDUAL_ATOL = 1.0e-9
_BOOZER_BRANCH_STABLE_COST_RTOL = 1.0e-9
_SINGULAR_COST_ATOL = 1.0e-16
_SINGULAR_GRAD_INF_ATOL = 1.0e-10


def _rosenbrock_residual_np(x):
    return np.asarray([10.0 * (x[1] - x[0] ** 2), 1.0 - x[0]])


def _rosenbrock_jacobian_np(x):
    return np.asarray([[-20.0 * x[0], 10.0], [-1.0, 0.0]])


def _rosenbrock_residual_jax(x):
    return jnp.asarray([10.0 * (x[1] - x[0] ** 2), 1.0 - x[0]])


def _helical_theta_np(x0, x1):
    theta = np.arctan(x1 / x0) / (2.0 * np.pi)
    if x0 <= 0.0:
        theta += 0.5
    return theta


def _helical_valley_residual_np(x):
    return np.asarray(
        [
            10.0 * (x[2] - 10.0 * _helical_theta_np(x[0], x[1])),
            10.0 * (np.sqrt(x[0] * x[0] + x[1] * x[1]) - 1.0),
            x[2],
        ]
    )


def _helical_valley_jacobian_np(x):
    r2 = x[0] * x[0] + x[1] * x[1]
    r = np.sqrt(r2)
    return np.asarray(
        [
            [50.0 * x[1] / (np.pi * r2), -50.0 * x[0] / (np.pi * r2), 10.0],
            [10.0 * x[0] / r, 10.0 * x[1] / r, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def _helical_valley_residual_jax(x):
    theta = jnp.arctan(x[1] / x[0]) / (2.0 * jnp.pi)
    theta = jnp.where(x[0] <= 0.0, theta + 0.5, theta)
    return jnp.asarray(
        [
            10.0 * (x[2] - 10.0 * theta),
            10.0 * (jnp.sqrt(x[0] * x[0] + x[1] * x[1]) - 1.0),
            x[2],
        ]
    )


def _powell_singular_residual_np(x):
    return np.asarray(
        [
            x[0] + 10.0 * x[1],
            np.sqrt(5.0) * (x[2] - x[3]),
            (x[1] - 2.0 * x[2]) ** 2,
            np.sqrt(10.0) * (x[0] - x[3]) ** 2,
        ]
    )


def _powell_singular_jacobian_np(x):
    return np.asarray(
        [
            [1.0, 10.0, 0.0, 0.0],
            [0.0, 0.0, np.sqrt(5.0), -np.sqrt(5.0)],
            [0.0, 2.0 * (x[1] - 2.0 * x[2]), -4.0 * (x[1] - 2.0 * x[2]), 0.0],
            [
                2.0 * np.sqrt(10.0) * (x[0] - x[3]),
                0.0,
                0.0,
                -2.0 * np.sqrt(10.0) * (x[0] - x[3]),
            ],
        ]
    )


def _powell_singular_residual_jax(x):
    return jnp.asarray(
        [
            x[0] + 10.0 * x[1],
            jnp.sqrt(5.0) * (x[2] - x[3]),
            (x[1] - 2.0 * x[2]) ** 2,
            jnp.sqrt(10.0) * (x[0] - x[3]) ** 2,
        ]
    )


def _brown_almost_linear_residual_np(x):
    total = np.sum(x)
    residual = np.empty_like(x)
    residual[:-1] = x[:-1] + total - (x.size + 1.0)
    residual[-1] = np.prod(x) - 1.0
    return residual


def _brown_almost_linear_jacobian_np(x):
    n = x.size
    jacobian = np.ones((n, n), dtype=np.float64)
    jacobian[:-1, :-1] += np.eye(n - 1)
    for column in range(n):
        jacobian[-1, column] = np.prod(np.delete(x, column))
    return jacobian


def _brown_almost_linear_residual_jax(x):
    total = jnp.sum(x)
    residual = x[:-1] + total - (x.size + 1.0)
    return jnp.concatenate([residual, jnp.asarray([jnp.prod(x) - 1.0])])


def _beale_residual_np(x):
    return np.asarray(
        [
            1.5 - x[0] * (1.0 - x[1]),
            2.25 - x[0] * (1.0 - x[1] ** 2),
            2.625 - x[0] * (1.0 - x[1] ** 3),
        ]
    )


def _beale_jacobian_np(x):
    return np.asarray(
        [
            [-(1.0 - x[1]), x[0]],
            [-(1.0 - x[1] ** 2), 2.0 * x[0] * x[1]],
            [-(1.0 - x[1] ** 3), 3.0 * x[0] * x[1] ** 2],
        ]
    )


def _beale_residual_jax(x):
    return jnp.asarray(
        [
            1.5 - x[0] * (1.0 - x[1]),
            2.25 - x[0] * (1.0 - x[1] ** 2),
            2.625 - x[0] * (1.0 - x[1] ** 3),
        ]
    )


def _linear_least_squares_problem():
    matrix = jnp.asarray(
        [[1.0, 2.0], [3.0, -1.0], [2.0, 0.5]],
        dtype=jnp.float64,
    )
    rhs = jnp.asarray([1.0, -2.0, 0.5], dtype=jnp.float64)
    return matrix, rhs


def _least_squares_cost(residual):
    residual = np.asarray(residual, dtype=np.float64)
    return 0.5 * float(residual @ residual)


def _gradient_inf_norm(gradient):
    return float(np.max(np.abs(np.asarray(gradient, dtype=np.float64))))


def _max_abs_difference(left, right):
    return float(
        np.max(
            np.abs(
                np.asarray(left, dtype=np.float64)
                - np.asarray(right, dtype=np.float64)
            )
        )
    )


def _lm_minpack_options(**overrides):
    return {
        "ftol": _LM_MINPACK_SOLVER_TOL,
        "xtol": _LM_MINPACK_SOLVER_TOL,
        "gtol": _LM_MINPACK_SOLVER_TOL,
        **overrides,
    }


def _assert_residual_oracle_matches_jax(residual_np, jacobian_np, residual_jax, x):
    x_jax = jnp.asarray(x, dtype=jnp.float64)
    np.testing.assert_allclose(
        residual_jax(x_jax),
        residual_np(x),
        rtol=_LM_MINPACK_DIRECT_RTOL,
        atol=_LM_MINPACK_DIRECT_ATOL,
    )
    np.testing.assert_allclose(
        jax.jacobian(residual_jax)(x_jax),
        jacobian_np(x),
        rtol=_LM_MINPACK_DIRECT_RTOL,
        atol=_LM_MINPACK_DIRECT_ATOL,
    )


def _run_scipy_lm(residual_np, jacobian_np, x0, *, max_nfev):
    return least_squares(
        residual_np,
        np.asarray(x0, dtype=np.float64),
        jac=jacobian_np,
        method="lm",
        ftol=_LM_MINPACK_SOLVER_TOL,
        xtol=_LM_MINPACK_SOLVER_TOL,
        gtol=_LM_MINPACK_SOLVER_TOL,
        max_nfev=max_nfev,
    )


def _run_jax_lm_minpack(residual_jax, x0, *, maxiter):
    return _opt.target_least_squares(
        residual_jax,
        jnp.asarray(x0, dtype=jnp.float64),
        method="lm-minpack-ondevice",
        tol=_LM_MINPACK_SOLVER_TOL,
        maxiter=maxiter,
        options=_lm_minpack_options(),
    )


_MGH_REGULAR_CASES = (
    pytest.param(
        np.asarray([-1.2, 1.0], dtype=np.float64),
        _rosenbrock_residual_np,
        _rosenbrock_jacobian_np,
        _rosenbrock_residual_jax,
        id="rosenbrock",
    ),
    pytest.param(
        np.asarray([-1.0, 0.0, 0.0], dtype=np.float64),
        _helical_valley_residual_np,
        _helical_valley_jacobian_np,
        _helical_valley_residual_jax,
        id="helical_valley",
    ),
    pytest.param(
        np.full(10, 0.5, dtype=np.float64),
        _brown_almost_linear_residual_np,
        _brown_almost_linear_jacobian_np,
        _brown_almost_linear_residual_jax,
        id="brown_almost_linear_10",
    ),
    pytest.param(
        np.asarray([1.0, 1.0], dtype=np.float64),
        _beale_residual_np,
        _beale_jacobian_np,
        _beale_residual_jax,
        id="beale",
    ),
)


def _build_boozer_lm_minpack_case(**builder_kwargs):
    from benchmarks.benchmark_problem import build_ls_parity_problem
    from simsopt.field.biotsavart_jax_backend import BiotSavartJAX
    from simsopt.geo import Volume
    from simsopt.geo.boozersurface_jax import BoozerSurfaceJAX

    problem = build_ls_parity_problem(**builder_kwargs)
    booz = BoozerSurfaceJAX(
        BiotSavartJAX(problem.coils),
        problem.surface,
        Volume(problem.surface),
        problem.vol_target,
        constraint_weight=1.0,
        options={
            "optimizer_backend": "ondevice",
            "least_squares_algorithm": "lm-minpack",
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
    residual_eval_fn = jax.jit(residual_fn)
    jacobian_fn = jax.jit(jax.jacobian(residual_fn))

    def residual_np(x):
        return np.asarray(
            residual_eval_fn(jnp.asarray(x, dtype=jnp.float64)),
            dtype=np.float64,
        )

    def jacobian_np(x):
        return np.asarray(
            jacobian_fn(jnp.asarray(x, dtype=jnp.float64)),
            dtype=np.float64,
        )

    x0_np = np.asarray(x0, dtype=np.float64)
    jacobian0 = jacobian_np(x0_np)
    scipy_result = _run_scipy_lm(
        residual_np,
        jacobian_np,
        x0_np,
        max_nfev=1500,
    )
    jax_result = _run_jax_lm_minpack(
        residual_fn,
        x0_np,
        maxiter=1500,
    )
    return {
        "jacobian0": jacobian0,
        "scipy_result": scipy_result,
        "jax_result": jax_result,
        "state_inf": _max_abs_difference(jax_result.x, scipy_result.x),
        "residual_inf": _max_abs_difference(jax_result.residual, scipy_result.fun),
        "scipy_cost": _least_squares_cost(scipy_result.fun),
        "jax_cost": _least_squares_cost(jax_result.residual),
    }


def test_target_lm_minpack_qr_lane_matches_scipy_lm_solution():
    x0 = np.asarray([-2.0, 1.0], dtype=np.float64)
    scipy_result = _run_scipy_lm(
        _rosenbrock_residual_np,
        _rosenbrock_jacobian_np,
        x0,
        max_nfev=200,
    )

    jax_result = _run_jax_lm_minpack(
        _rosenbrock_residual_jax,
        x0,
        maxiter=200,
    )

    assert jax_result.success
    np.testing.assert_allclose(
        jax_result.x,
        scipy_result.x,
        rtol=_LM_MINPACK_DIRECT_RTOL,
        atol=_LM_MINPACK_DIRECT_ATOL,
    )
    np.testing.assert_allclose(
        jax_result.residual,
        scipy_result.fun,
        rtol=_LM_MINPACK_DIRECT_RTOL,
        atol=_LM_MINPACK_DIRECT_ATOL,
    )
    assert jax_result.residual_jacobian.shape == (2, 2)
    assert jax_result.hessian.shape == (2, 2)


@pytest.mark.parametrize(
    ("x0", "residual_np", "jacobian_np", "residual_jax"),
    _MGH_REGULAR_CASES,
)
def test_target_lm_minpack_qr_lane_matches_regular_mgh_final_state(
    x0,
    residual_np,
    jacobian_np,
    residual_jax,
):
    _assert_residual_oracle_matches_jax(
        residual_np,
        jacobian_np,
        residual_jax,
        x0,
    )
    scipy_result = _run_scipy_lm(
        residual_np,
        jacobian_np,
        x0,
        max_nfev=1000,
    )
    jax_result = _run_jax_lm_minpack(
        residual_jax,
        x0,
        maxiter=1000,
    )

    assert scipy_result.success
    assert jax_result.success
    np.testing.assert_allclose(
        jax_result.x,
        scipy_result.x,
        rtol=_LM_MINPACK_DIRECT_RTOL,
        atol=_LM_MINPACK_DIRECT_ATOL,
    )
    np.testing.assert_allclose(
        jax_result.residual,
        scipy_result.fun,
        rtol=_LM_MINPACK_DIRECT_RTOL,
        atol=_LM_MINPACK_DIRECT_ATOL,
    )


def test_target_lm_minpack_qr_lane_classifies_powell_singular_by_optimality():
    x0 = np.asarray([3.0, -1.0, 0.0, 1.0], dtype=np.float64)
    _assert_residual_oracle_matches_jax(
        _powell_singular_residual_np,
        _powell_singular_jacobian_np,
        _powell_singular_residual_jax,
        x0,
    )
    scipy_result = _run_scipy_lm(
        _powell_singular_residual_np,
        _powell_singular_jacobian_np,
        x0,
        max_nfev=1000,
    )
    jax_result = _run_jax_lm_minpack(
        _powell_singular_residual_jax,
        x0,
        maxiter=1000,
    )

    assert scipy_result.success
    assert jax_result.success
    np.testing.assert_allclose(
        jax_result.residual,
        scipy_result.fun,
        rtol=_LM_MINPACK_DIRECT_RTOL,
        atol=1.0e-8,
    )
    np.testing.assert_allclose(
        _least_squares_cost(jax_result.residual),
        _least_squares_cost(scipy_result.fun),
        rtol=_LM_MINPACK_DIRECT_RTOL,
        atol=_SINGULAR_COST_ATOL,
    )
    assert _gradient_inf_norm(jax_result.jac) <= _SINGULAR_GRAD_INF_ATOL
    assert scipy_result.optimality <= _SINGULAR_GRAD_INF_ATOL


def test_target_lm_minpack_qr_lane_uses_pivoted_qr_on_overdetermined_system():
    matrix = jnp.asarray(
        [[1.0, 2.0], [3.0, -1.0], [2.0, 0.5], [-1.0, 4.0]],
        dtype=jnp.float64,
    )
    rhs = jnp.asarray([1.0, -2.0, 0.5, 3.0], dtype=jnp.float64)
    expected_x, *_ = np.linalg.lstsq(np.asarray(matrix), np.asarray(rhs), rcond=None)

    result = _opt.target_least_squares(
        lambda x: matrix @ x - rhs,
        jnp.zeros(2, dtype=jnp.float64),
        method="lm-minpack-ondevice",
        tol=_LM_MINPACK_SOLVER_TOL,
        maxiter=50,
        options=_lm_minpack_options(),
    )

    assert result.success
    np.testing.assert_allclose(
        result.x,
        expected_x,
        rtol=_LM_MINPACK_DIRECT_RTOL,
        atol=_LM_MINPACK_DIRECT_ATOL,
    )
    assert result.info in {0, 1, 2, 3, 4}


def test_target_lm_minpack_qr_lane_always_reports_required_dense_artifacts():
    matrix, rhs = _linear_least_squares_problem()

    result = _opt.target_least_squares(
        lambda x: matrix @ x - rhs,
        jnp.zeros(2, dtype=jnp.float64),
        method="lm-minpack-ondevice",
        tol=_LM_MINPACK_SOLVER_TOL,
        maxiter=50,
        options=_lm_minpack_options(materialize_dense_linearization=False),
    )

    assert result.dense_linearization_materialized is True
    np.testing.assert_allclose(result.residual_jacobian, matrix)
    np.testing.assert_allclose(result.hessian, matrix.T @ matrix)


def test_target_lm_minpack_qr_lane_enforces_dense_linearization_byte_cap():
    matrix, rhs = _linear_least_squares_problem()

    with pytest.raises(
        MemoryError,
        match="dense QR solve requires.*max_dense_linearization_bytes=1",
    ):
        _opt.target_least_squares(
            lambda x: matrix @ x - rhs,
            jnp.zeros(2, dtype=jnp.float64),
            method="lm-minpack-ondevice",
            tol=_LM_MINPACK_SOLVER_TOL,
            maxiter=50,
            options={"max_dense_linearization_bytes": 1},
        )


@pytest.mark.slow
def test_target_lm_minpack_qr_lane_boozer_oversampled_branch_stable_parity():
    case = _build_boozer_lm_minpack_case(ncoils=4, nphi=16, ntheta=8)

    assert float(np.linalg.cond(case["jacobian0"])) < 1.0e3
    assert case["scipy_result"].success
    assert case["jax_result"].success
    assert case["state_inf"] <= _BOOZER_BRANCH_STABLE_STATE_ATOL
    assert case["residual_inf"] <= _BOOZER_BRANCH_STABLE_RESIDUAL_ATOL
    np.testing.assert_allclose(
        case["jax_cost"],
        case["scipy_cost"],
        rtol=_BOOZER_BRANCH_STABLE_COST_RTOL,
        atol=_SINGULAR_COST_ATOL,
    )


@pytest.mark.slow
def test_target_lm_minpack_qr_lane_boozer_default_gates_physics_not_raw_state():
    case = _build_boozer_lm_minpack_case()

    assert case["scipy_result"].success
    assert case["jax_result"].success
    assert np.isfinite(case["state_inf"])
    assert case["residual_inf"] <= _BOOZER_BRANCH_STABLE_RESIDUAL_ATOL
    np.testing.assert_allclose(
        case["jax_cost"],
        case["scipy_cost"],
        rtol=_BOOZER_BRANCH_STABLE_COST_RTOL,
        atol=_SINGULAR_COST_ATOL,
    )
