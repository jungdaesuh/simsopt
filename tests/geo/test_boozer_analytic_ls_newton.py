from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from simsopt_jax.geo.optimizers.native_ls_newton import newton_ls_native_dense

jax.config.update("jax_enable_x64", True)


def _polynomial_ls_numpy(
    values: np.ndarray,
) -> tuple[np.float64, np.ndarray, np.ndarray]:
    x0, x1 = values
    residual = np.asarray(
        [x0**2 + x1 - 1.0, x0 + x1**2 - 1.0, 0.35 * x0 - 0.2 * x1 + 0.15]
    )
    jacobian = np.asarray([[2.0 * x0, 1.0], [1.0, 2.0 * x1], [0.35, -0.2]])
    gradient = jacobian.T @ residual
    hessian = jacobian.T @ jacobian
    hessian += residual[0] * np.asarray([[2.0, 0.0], [0.0, 0.0]])
    hessian += residual[1] * np.asarray([[0.0, 0.0], [0.0, 2.0]])
    return np.float64(0.5 * residual @ residual), gradient, hessian


def _polynomial_ls_jax(
    values: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    x0, x1 = values
    residual = jnp.stack(
        (
            x0**2 + x1 - 1.0,
            x0 + x1**2 - 1.0,
            0.35 * x0 - 0.2 * x1 + 0.15,
        )
    )
    jacobian = jnp.stack(
        (
            jnp.stack((2.0 * x0, jnp.asarray(1.0, dtype=values.dtype))),
            jnp.stack((jnp.asarray(1.0, dtype=values.dtype), 2.0 * x1)),
            jnp.asarray([0.35, -0.2], dtype=values.dtype),
        )
    )
    gradient = jacobian.T @ residual
    hessian = jacobian.T @ jacobian
    hessian += residual[0] * jnp.asarray([[2.0, 0.0], [0.0, 0.0]], dtype=values.dtype)
    hessian += residual[1] * jnp.asarray([[0.0, 0.0], [0.0, 2.0]], dtype=values.dtype)
    return 0.5 * residual @ residual, gradient, hessian


def _literal_native_numpy_oracle(
    value_grad_hessian_fn,
    initial: np.ndarray,
    *,
    maxiter: int,
    tol: float,
    stab: float,
):
    initial_x = np.array(initial, copy=True)
    x = np.array(initial, copy=True)
    fun, grad, hessian = value_grad_hessian_fn(x)
    norm = np.linalg.norm(grad)
    initial_norm = norm
    nit = 0
    while nit < maxiter and norm > tol:
        stabilized_hessian = hessian + stab * np.identity(hessian.shape[0])
        direction = np.linalg.solve(stabilized_hessian, grad)
        if norm < 1.0e-9:
            direction += np.linalg.solve(
                stabilized_hessian,
                grad - stabilized_hessian @ direction,
            )
        x = x - direction
        fun, grad, hessian = value_grad_hessian_fn(x)
        norm = np.linalg.norm(grad)
        nit += 1
    success = norm <= tol
    persist_solved_state = success or (np.isfinite(norm) and norm <= initial_norm)
    if not persist_solved_state:
        x = initial_x
        fun, grad, hessian = value_grad_hessian_fn(x)
    return {
        "x": x,
        "fun": fun,
        "grad": grad,
        "hessian": hessian,
        "nit": nit,
        "success": success,
        "initial_norm": initial_norm,
        "final_norm": norm,
        "persist_solved_state": persist_solved_state,
    }


def _scalar_residual_ls(
    values: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    x = values[0]
    residual = x**2 - 1.0
    return (
        0.5 * residual**2,
        jnp.asarray([2.0 * x * residual], dtype=values.dtype),
        jnp.asarray([[6.0 * x**2 - 2.0]], dtype=values.dtype),
    )


def test_native_ls_newton_matches_literal_numpy_full_hessian_policy() -> None:
    initial = np.asarray([0.75, 0.45], dtype=np.float64)
    expected = _literal_native_numpy_oracle(
        _polynomial_ls_numpy,
        initial,
        maxiter=6,
        tol=1.0e-13,
        stab=0.03,
    )
    actual = jax.jit(
        lambda values: newton_ls_native_dense(
            _polynomial_ls_jax,
            values,
            maxiter=6,
            tol=1.0e-13,
            stab=0.03,
        )
    )(jnp.asarray(initial))

    for key in ("x", "fun", "grad", "hessian", "initial_norm", "final_norm"):
        np.testing.assert_allclose(
            np.asarray(actual[key]),
            np.asarray(expected[key]),
            rtol=2.0e-12,
            atol=2.0e-12,
        )
    assert int(actual["nit"]) == expected["nit"]
    assert bool(actual["success"]) == bool(expected["success"])
    assert bool(actual["persist_solved_state"]) == bool(
        expected["persist_solved_state"]
    )
    x0, x1 = np.asarray(actual["x"])
    endpoint_residual = np.asarray(
        [x0**2 + x1 - 1.0, x0 + x1**2 - 1.0, 0.35 * x0 - 0.2 * x1 + 0.15]
    )
    assert np.linalg.norm(endpoint_residual) > 1.0e-3


def test_native_ls_newton_stabilizes_step_but_returns_unstabilized_hessian() -> None:
    initial = jnp.asarray([2.0], dtype=jnp.float64)
    actual = newton_ls_native_dense(
        _scalar_residual_ls,
        initial,
        maxiter=1,
        tol=0.0,
        stab=5.0,
    )

    expected_x = 2.0 - 12.0 / (22.0 + 5.0)
    np.testing.assert_allclose(
        np.asarray(actual["x"]), [expected_x], rtol=0.0, atol=1e-15
    )
    np.testing.assert_allclose(
        np.asarray(actual["hessian"]),
        [[6.0 * expected_x**2 - 2.0]],
        rtol=0.0,
        atol=2.0e-15,
    )


def test_native_ls_newton_applies_single_refinement_below_native_threshold() -> None:
    hessian = np.asarray(
        [
            [0.46951349655358615, 0.3600822170797091, 0.34554827091087414],
            [0.3600822170797091, 0.2761621357568279, 0.26498507145917916],
            [0.34554827091087414, 0.26498507145917916, 0.25442437768958576],
        ],
        dtype=np.float64,
    )
    initial_gradient = np.asarray(
        [-1.2609278158585695e-10, 1.3887418654282017e-10, -2.845537097802725e-12],
        dtype=np.float64,
    )

    def quadratic(values: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        matrix = jnp.asarray(hessian)
        linear = jnp.asarray(initial_gradient)
        return (
            0.5 * values @ matrix @ values + linear @ values,
            matrix @ values + linear,
            matrix,
        )

    first_direction = np.linalg.solve(hessian, initial_gradient)
    refined_direction = first_direction + np.linalg.solve(
        hessian,
        initial_gradient - hessian @ first_direction,
    )
    actual = newton_ls_native_dense(
        quadratic,
        jnp.zeros(3, dtype=jnp.float64),
        maxiter=1,
        tol=0.0,
    )

    np.testing.assert_allclose(
        np.asarray(actual["x"]),
        -refined_direction,
        rtol=2.0e-8,
        atol=2.0e-12,
    )
    assert np.linalg.norm(np.asarray(actual["x"]) + first_direction) > 1.0e-12

    above_threshold_gradient = 10.0 * initial_gradient

    def above_threshold_quadratic(
        values: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        matrix = jnp.asarray(hessian)
        linear = jnp.asarray(above_threshold_gradient)
        return (
            0.5 * values @ matrix @ values + linear @ values,
            matrix @ values + linear,
            matrix,
        )

    above_threshold_direction = np.linalg.solve(hessian, above_threshold_gradient)
    above_threshold_actual = newton_ls_native_dense(
        above_threshold_quadratic,
        jnp.zeros(3, dtype=jnp.float64),
        maxiter=1,
        tol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(above_threshold_actual["x"]),
        -above_threshold_direction,
        rtol=2.0e-8,
        atol=2.0e-12,
    )


def test_native_ls_newton_maxiter_zero_returns_initial_derivatives() -> None:
    initial = jnp.asarray([0.6], dtype=jnp.float64)
    actual = newton_ls_native_dense(
        _scalar_residual_ls,
        initial,
        maxiter=0,
        tol=1.0e-14,
    )
    expected_fun, expected_grad, expected_hessian = _scalar_residual_ls(initial)

    assert int(actual["nit"]) == 0
    assert not bool(actual["success"])
    assert bool(actual["persist_solved_state"])
    np.testing.assert_array_equal(np.asarray(actual["x"]), np.asarray(initial))
    np.testing.assert_array_equal(np.asarray(actual["fun"]), np.asarray(expected_fun))
    np.testing.assert_array_equal(np.asarray(actual["grad"]), np.asarray(expected_grad))
    np.testing.assert_allclose(
        np.asarray(actual["hessian"]),
        np.asarray(expected_hessian),
        rtol=0.0,
        atol=3.0e-16,
    )


def test_native_ls_newton_worsening_failure_rolls_back_and_recomputes() -> None:
    initial = jnp.asarray([0.6], dtype=jnp.float64)
    actual = newton_ls_native_dense(
        _scalar_residual_ls,
        initial,
        maxiter=1,
        tol=1.0e-14,
    )
    expected_fun, expected_grad, expected_hessian = _scalar_residual_ls(initial)

    assert int(actual["nit"]) == 1
    assert not bool(actual["success"])
    assert not bool(actual["persist_solved_state"])
    assert float(actual["attempted_norm"]) > float(actual["initial_norm"])
    np.testing.assert_array_equal(
        np.asarray(actual["final_norm"]), np.asarray(actual["initial_norm"])
    )
    np.testing.assert_array_equal(np.asarray(actual["x"]), np.asarray(initial))
    np.testing.assert_array_equal(np.asarray(actual["fun"]), np.asarray(expected_fun))
    np.testing.assert_array_equal(np.asarray(actual["grad"]), np.asarray(expected_grad))
    np.testing.assert_allclose(
        np.asarray(actual["hessian"]),
        np.asarray(expected_hessian),
        rtol=0.0,
        atol=3.0e-16,
    )
