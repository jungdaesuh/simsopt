from __future__ import annotations

from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from simsopt._core import load
from simsopt.field import BiotSavart
from simsopt.geo import BoozerSurface, Volume
from simsopt_jax.geo.optimizers.native_ls_newton import newton_ls_native_dense
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt_jax_adapters.geo.boozer_surface import BoozerSurfaceJAX
from simsopt_jax_adapters.geo.nested_ls_ncsx import (
    NCSX_EXAMPLE_JSON,
    upsample_surface_xyz_tensor_fourier,
)

from .surface_test_helpers import get_boozer_surface

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
    divergence_factor=1e3,
):
    initial_x = np.array(initial, copy=True)
    x = np.array(initial, copy=True)
    fun, grad, hessian = value_grad_hessian_fn(x)
    norm = np.linalg.norm(grad)
    initial_norm = norm
    nit = 0
    while (
        nit < maxiter
        and norm > tol
        and (
            divergence_factor is None
            or divergence_factor == 0
            or norm <= divergence_factor * initial_norm
        )
    ):
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
        norm = np.linalg.norm(grad)
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
        divergence_factor=None,
    )
    actual = jax.jit(
        lambda values: newton_ls_native_dense(
            _polynomial_ls_jax,
            values,
            maxiter=6,
            tol=1.0e-13,
            stab=0.03,
            divergence_factor=None,
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


def _runaway_ls_numpy(values: np.ndarray):
    packed = np.asarray(values, dtype=np.float64)
    hessian = 0.05 * np.eye(packed.size, dtype=packed.dtype)
    return np.float64(0.5 * packed @ packed), packed, hessian


def _runaway_ls_jax(
    values: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    return (
        0.5 * values @ values,
        values,
        jnp.asarray(0.05, dtype=values.dtype)
        * jnp.eye(values.shape[0], dtype=values.dtype),
    )


def _ncsx_18_ls_pair():
    packed = load(str(NCSX_EXAMPLE_JSON))
    _, _, coils, _, surfaces, boozer_surfaces, ress = packed
    native_surface = upsample_surface_xyz_tensor_fourier(
        surfaces[0], mpol=6, ntor=6, nphi=18, ntheta=18
    )
    device_surface = upsample_surface_xyz_tensor_fourier(
        surfaces[0], mpol=6, ntor=6, nphi=18, ntheta=18
    )
    constraint_weight = float(boozer_surfaces[0].constraint_weight)
    native_label = Volume(native_surface)
    target = float(native_label.J())
    options = {"verbose": False, "weight_inv_modB": True}
    native = BoozerSurface(
        BiotSavart(coils),
        native_surface,
        native_label,
        target,
        constraint_weight=constraint_weight,
        options=options,
    )
    device = BoozerSurfaceJAX(
        BiotSavartJAX(coils),
        device_surface,
        Volume(device_surface),
        target,
        constraint_weight=constraint_weight,
        options=options,
    )
    return (
        native,
        device,
        float(ress[0]["iota"]),
        float(ress[0]["G"]),
        constraint_weight,
    )


def test_native_ls_newton_divergence_guard_stops_exploding_walk() -> None:
    initial = jnp.asarray([1.0], dtype=jnp.float64)
    guarded = newton_ls_native_dense(
        _runaway_ls_jax,
        initial,
        maxiter=40,
        tol=1.0e-14,
    )
    disabled = newton_ls_native_dense(
        _runaway_ls_jax,
        initial,
        maxiter=40,
        tol=1.0e-14,
        divergence_factor=None,
    )
    disabled_zero = newton_ls_native_dense(
        _runaway_ls_jax,
        initial,
        maxiter=40,
        tol=1.0e-14,
        divergence_factor=0.0,
    )

    assert not bool(guarded["success"])
    assert not bool(guarded["persist_solved_state"])
    assert int(guarded["nit"]) <= 5
    assert int(guarded["nit"]) > 0
    assert int(disabled["nit"]) == 40
    assert int(disabled_zero["nit"]) == 40
    assert not bool(disabled["success"])
    np.testing.assert_array_equal(np.asarray(guarded["x"]), np.asarray(initial))


def _convex_quadratic_jax(
    values: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    target = jnp.asarray([1.0, -0.5], dtype=values.dtype)
    residual = values - target
    return (
        0.5 * residual @ residual,
        residual,
        jnp.eye(values.shape[0], dtype=values.dtype),
    )


def test_native_ls_newton_accepted_walk_bitwise_identical_with_guard_on_and_off() -> (
    None
):
    initial = np.asarray([0.75, 0.45], dtype=np.float64)
    guarded = newton_ls_native_dense(
        _convex_quadratic_jax,
        jnp.asarray(initial),
        maxiter=6,
        tol=1.0e-13,
    )
    disabled = newton_ls_native_dense(
        _convex_quadratic_jax,
        jnp.asarray(initial),
        maxiter=6,
        tol=1.0e-13,
        divergence_factor=None,
    )
    assert bool(guarded["success"])
    assert int(guarded["nit"]) == int(disabled["nit"])
    for key in ("x", "fun", "grad", "hessian", "final_norm"):
        np.testing.assert_array_equal(
            np.asarray(guarded[key]), np.asarray(disabled[key])
        )


def test_analytic_ls_newton_18_accepted_bitwise_guard_on_and_off() -> None:
    native, device, iota0, g_value, constraint_weight = _ncsx_18_ls_pair()
    seed = np.concatenate((native.surface.get_dofs(), [iota0, g_value]))
    _, derivatives = device._make_analytic_penalty_derivatives(
        True, True, constraint_weight
    )
    jax_guarded = newton_ls_native_dense(
        derivatives,
        jnp.asarray(seed),
        maxiter=40,
        tol=1e-11,
        args=(device.coil_set_spec,),
    )
    jax_disabled = newton_ls_native_dense(
        derivatives,
        jnp.asarray(seed),
        maxiter=40,
        tol=1e-11,
        divergence_factor=None,
        args=(device.coil_set_spec,),
    )
    assert bool(jax_guarded["success"])
    assert bool(jax_disabled["success"])
    assert int(jax_guarded["nit"]) == int(jax_disabled["nit"])
    np.testing.assert_array_equal(
        np.asarray(jax_guarded["x"]), np.asarray(jax_disabled["x"])
    )
    np.testing.assert_array_equal(
        np.asarray(jax_guarded["fun"]), np.asarray(jax_disabled["fun"])
    )


def _nonmonotone_then_converge_jax(
    values: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    x = values[0]
    grad = jnp.where(
        x < 0.5,
        jnp.asarray(-1.0, dtype=values.dtype),
        jnp.where(
            x < 1.0005,
            jnp.asarray(-0.001, dtype=values.dtype),
            jnp.where(
                x < 1.5,
                -(jnp.asarray(2.0, dtype=values.dtype) - x),
                jnp.asarray(0.0, dtype=values.dtype),
            ),
        ),
    )
    hessian = jnp.ones((1, 1), dtype=values.dtype)
    return 0.5 * grad * grad, jnp.stack((grad,)), hessian


def test_native_ls_newton_nonmonotone_convergent_walk_is_not_aborted() -> None:
    actual = newton_ls_native_dense(
        _nonmonotone_then_converge_jax,
        jnp.asarray([0.0], dtype=jnp.float64),
        maxiter=10,
        tol=1.0e-12,
    )
    assert bool(actual["success"])
    assert int(actual["nit"]) == 3
    np.testing.assert_allclose(np.asarray(actual["x"]), [2.0], atol=1e-12)


def test_jax_and_native_ls_newton_stop_at_same_iteration_on_diverging_input() -> None:
    _, boozer_surface = get_boozer_surface(boozer_type="ls", converge=False)
    dofs = np.array(boozer_surface.surface.get_dofs(), copy=True)
    dofs[0] = 0.6
    boozer_surface.surface.set_dofs(dofs)
    iota = -0.406
    g_value = -2.0
    initial = np.concatenate((dofs, [iota, g_value]))

    def exploding_penalty(
        state,
        derivatives=2,
        constraint_weight=1.0,
        optimize_G=False,
        weight_inv_modB=True,
    ):
        assert derivatives == 2
        assert optimize_G
        return _runaway_ls_numpy(state)

    boozer_surface.need_to_run_code = True
    with mock.patch.object(
        boozer_surface,
        "boozer_penalty_constraints_vectorized",
        side_effect=exploding_penalty,
    ):
        native = boozer_surface.minimize_boozer_penalty_constraints_newton(
            tol=1e-14,
            maxiter=40,
            constraint_weight=1.0,
            iota=iota,
            G=g_value,
            verbose=False,
        )
    jax_result = newton_ls_native_dense(
        _runaway_ls_jax,
        jnp.asarray(initial),
        maxiter=40,
        tol=1e-14,
    )
    assert not bool(native["success"])
    assert not bool(jax_result["success"])
    assert native["iter"] == int(jax_result["nit"])
    assert native["iter"] <= 5
    assert native["iter"] > 0
