import numpy as np
import pytest
import jax
import jax.numpy as jnp

pytest.importorskip("optimistix", minversion="0.1.0")
pytest.importorskip("lineax", minversion="0.1.1")
pytest.importorskip("equinox", minversion="0.11.11")

from benchmarks.validation_ladder_contract import parity_ladder_tolerances
from simsopt.geo import optimizer_jax as _opt


_OPTIMISTIX_SOLVER_TOL = 1.0e-10
_DIRECT_ATOL = 1.0e-8
_DIRECT_RTOL = 1.0e-8
_BOOZER_COST_ATOL = 1.0e-10
_BOOZER_RESIDUAL_NORM_ATOL = 1.0e-6
_BRANCH_STABLE_RESOLVE_TOLS = parity_ladder_tolerances("branch-stable-resolve")
_BOOZER_RESIDUAL_VECTOR_ATOL = 1.0e-9


def _rosenbrock_residual(x):
    return jnp.asarray([10.0 * (x[1] - x[0] ** 2), 1.0 - x[0]])


def _linear_least_squares_problem():
    matrix = jnp.asarray(
        [[1.0, 2.0], [3.0, -1.0], [2.0, 0.5], [-1.0, 4.0]],
        dtype=jnp.float64,
    )
    rhs = jnp.asarray([1.0, -2.0, 0.5, 3.0], dtype=jnp.float64)
    return matrix, rhs


def _run_optimistix_lm(residual_fn, x0, *, maxiter=200, materialize=True):
    return _opt.target_least_squares(
        residual_fn,
        jax.tree_util.tree_map(lambda leaf: jnp.asarray(leaf, dtype=jnp.float64), x0),
        method="optimistix-lm-ondevice",
        tol=_OPTIMISTIX_SOLVER_TOL,
        maxiter=maxiter,
        options={"materialize_dense_linearization": materialize},
    )


def _least_squares_cost(residual):
    residual = np.asarray(residual, dtype=np.float64)
    return 0.5 * float(residual @ residual)


def _max_abs_difference(left, right):
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    return float(np.max(np.abs(left_array - right_array)))


def _branch_stable_endpoint_threshold(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    scale = max(float(np.max(np.abs(left))), float(np.max(np.abs(right))))
    return (
        float(_BRANCH_STABLE_RESOLVE_TOLS["core_value_atol"])
        + float(_BRANCH_STABLE_RESOLVE_TOLS["core_value_rtol"]) * scale
    )


def _build_boozer_residual_case(**builder_kwargs):
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
            "least_squares_algorithm": "optimistix-lm",
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
    return residual_fn, x0


def test_target_optimistix_lm_lane_matches_linear_lstsq_solution():
    matrix, rhs = _linear_least_squares_problem()
    expected_x, *_ = np.linalg.lstsq(np.asarray(matrix), np.asarray(rhs), rcond=None)

    result = _run_optimistix_lm(lambda x: matrix @ x - rhs, jnp.zeros(2))

    assert result.success
    np.testing.assert_allclose(
        result.x,
        expected_x,
        rtol=_DIRECT_RTOL,
        atol=_DIRECT_ATOL,
    )
    np.testing.assert_allclose(result.residual_jacobian, matrix)
    np.testing.assert_allclose(result.hessian, matrix.T @ matrix)
    assert result.dense_linearization_kind == "post_hoc"


def test_target_optimistix_lm_lane_solves_rosenbrock_residual():
    result = _run_optimistix_lm(
        _rosenbrock_residual,
        jnp.asarray([-2.0, 1.0], dtype=jnp.float64),
    )

    assert result.success
    np.testing.assert_allclose(
        result.x,
        np.ones(2, dtype=np.float64),
        rtol=_DIRECT_RTOL,
        atol=_DIRECT_ATOL,
    )
    assert _least_squares_cost(result.residual) <= 1.0e-18


@pytest.mark.parametrize("callback_name", ["callback", "progress_callback"])
def test_target_optimistix_lm_lane_rejects_callbacks(callback_name):
    kwargs = {callback_name: lambda *_args: None}
    with pytest.raises(ValueError, match="does not support solver callbacks"):
        _opt.target_least_squares(
            _rosenbrock_residual,
            jnp.asarray([-2.0, 1.0], dtype=jnp.float64),
            method="optimistix-lm-ondevice",
            tol=_OPTIMISTIX_SOLVER_TOL,
            maxiter=200,
            **kwargs,
        )


@pytest.mark.parametrize(
    ("options", "expected_option"),
    [
        ({"ftol": 1.0e-6}, "ftol"),
        ({"xtol": 1.0e-6}, "xtol"),
        ({"gtol": 1.0e-8}, "gtol"),
    ],
)
def test_target_optimistix_lm_lane_rejects_nondefault_lm_tuning(
    options,
    expected_option,
):
    with pytest.raises(ValueError, match=expected_option):
        _opt.target_least_squares(
            _rosenbrock_residual,
            jnp.asarray([-2.0, 1.0], dtype=jnp.float64),
            method="optimistix-lm-ondevice",
            tol=_OPTIMISTIX_SOLVER_TOL,
            maxiter=200,
            options=options,
        )


def test_target_optimistix_lm_lane_maps_outer_max_steps_to_info_5():
    result = _run_optimistix_lm(
        _rosenbrock_residual,
        jnp.asarray([-2.0, 1.0], dtype=jnp.float64),
        maxiter=2,
        materialize=False,
    )

    assert not result.success
    assert result.info == 5
    assert result.message == "maximum iterations reached"
    assert result.dense_linearization_kind is None
    assert result.optimistix_result
    assert result.optimistix_result_message


def test_target_optimistix_lm_lane_maps_nonfinite_state_to_status_2():
    result = _run_optimistix_lm(
        lambda x: jnp.asarray([jnp.nan, x[0]], dtype=jnp.float64),
        jnp.asarray([1.0], dtype=jnp.float64),
        maxiter=2,
        materialize=False,
    )

    assert not result.success
    assert result.status == 2
    assert (
        result.message == "non-finite residual, gradient, or linear solve encountered"
    )


@pytest.mark.slow
def test_target_optimistix_lm_boozer_oversampled_matches_objective_not_state():
    residual_fn, x0 = _build_boozer_residual_case(ncoils=4, nphi=16, ntheta=8)

    matrix_free = _opt.target_least_squares(
        residual_fn,
        x0,
        method="lm-ondevice",
        tol=_OPTIMISTIX_SOLVER_TOL,
        maxiter=300,
        options={"materialize_dense_linearization": False},
    )
    optimistix = _run_optimistix_lm(residual_fn, x0, maxiter=300, materialize=False)

    assert matrix_free.success
    assert optimistix.success
    optimistix_residual_norm = float(np.max(np.abs(optimistix.residual)))
    matrix_free_residual_norm = float(np.max(np.abs(matrix_free.residual)))
    optimistix_cost = _least_squares_cost(optimistix.residual)
    matrix_free_cost = _least_squares_cost(matrix_free.residual)

    assert (
        abs(optimistix_residual_norm - matrix_free_residual_norm)
        <= _BOOZER_RESIDUAL_NORM_ATOL
    )
    np.testing.assert_allclose(
        optimistix_cost,
        matrix_free_cost,
        rtol=1.0e-5,
        atol=_BOOZER_COST_ATOL,
    )
    assert _max_abs_difference(
        optimistix.x, matrix_free.x
    ) > _branch_stable_endpoint_threshold(optimistix.x, matrix_free.x)
    assert (
        _max_abs_difference(optimistix.residual, matrix_free.residual)
        > _BOOZER_RESIDUAL_VECTOR_ATOL
    )


@pytest.mark.slow
def test_target_optimistix_lm_boozer_default_remains_experimental_not_robustness_gate():
    residual_fn, x0 = _build_boozer_residual_case()

    matrix_free = _opt.target_least_squares(
        residual_fn,
        x0,
        method="lm-ondevice",
        tol=_OPTIMISTIX_SOLVER_TOL,
        maxiter=300,
        options={"materialize_dense_linearization": False},
    )
    optimistix = _run_optimistix_lm(residual_fn, x0, maxiter=300, materialize=False)

    assert matrix_free.success
    assert optimistix.success
    optimistix_cost = _least_squares_cost(optimistix.residual)
    matrix_free_cost = _least_squares_cost(matrix_free.residual)

    assert np.isfinite(optimistix_cost)
    assert optimistix_cost <= 1.0e-12
    assert matrix_free_cost <= optimistix_cost
