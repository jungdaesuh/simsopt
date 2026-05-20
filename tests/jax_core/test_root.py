import jax
import jax.numpy as jnp
import numpy as np

from simsopt.jax_core._root import newton_scan_fixed_iters, newton_with_implicit_vjp


def test_newton_scan_fixed_iters_solves_vector_system_and_is_jittable():
    """Oracle: closed-form root of a decoupled quadratic vector system."""

    def residual(x):
        return x * x - jnp.array([4.0, 9.0], dtype=x.dtype)

    root = jax.jit(lambda x0: newton_scan_fixed_iters(residual, x0, max_iter=8))(
        jnp.array([1.0, 1.0], dtype=jnp.float64)
    )

    np.testing.assert_allclose(np.asarray(root), np.array([2.0, 3.0]), rtol=1e-12)


def test_newton_scan_fixed_iters_can_use_explicit_jacobian():
    """Oracle: same closed-form root with caller-supplied Jacobian."""

    def residual(x):
        return jnp.array(
            (
                x[0] + 2.0 * x[1] - 5.0,
                3.0 * x[0] - x[1] - 4.0,
            ),
            dtype=x.dtype,
        )

    def jacobian(x):
        return jnp.array(((1.0, 2.0), (3.0, -1.0)), dtype=x.dtype)

    root = newton_scan_fixed_iters(
        residual,
        jnp.zeros((2,), dtype=jnp.float64),
        max_iter=1,
        jac=jacobian,
    )

    np.testing.assert_allclose(np.asarray(root), np.array([13.0 / 7.0, 11.0 / 7.0]))


def test_newton_with_implicit_vjp_matches_closed_form_parameter_gradient():
    """Oracle: implicit derivative of ``x**2 - p = 0`` is ``1 / (2 sqrt(p))``."""

    def residual(x, param):
        return x * x - param

    def solve(param):
        return newton_with_implicit_vjp(
            residual,
            jnp.asarray(1.0, dtype=jnp.float64),
            param,
            max_iter=20,
            tol=1e-13,
        )

    param = jnp.asarray(4.0, dtype=jnp.float64)
    root = solve(param)
    gradient = jax.grad(solve)(param)

    np.testing.assert_allclose(np.asarray(root), 2.0, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(gradient), 0.25, rtol=1e-12)


def test_newton_with_implicit_vjp_matches_vector_parameter_gradient():
    """Oracle: root of ``x - p**2 = 0`` has gradient ``2 * p``."""

    def residual(x, params):
        return x - params["scale"] * params["scale"]

    params = {"scale": jnp.array([1.5, 2.5], dtype=jnp.float64)}

    def objective(scale):
        return jnp.sum(
            newton_with_implicit_vjp(
                residual,
                jnp.zeros((2,), dtype=jnp.float64),
                {"scale": scale},
                max_iter=4,
                tol=1e-13,
            )
        )

    gradient = jax.grad(objective)(params["scale"])

    np.testing.assert_allclose(
        np.asarray(gradient),
        np.array([3.0, 5.0]),
        rtol=1e-12,
    )


def test_newton_with_implicit_vjp_uses_explicit_jacobian_for_coupled_system():
    """Oracle: analytic IFT derivative for a coupled linear 2-D system."""

    matrix = jnp.array(((2.0, -1.0), (0.5, 3.0)), dtype=jnp.float64)

    def residual(x, params):
        return matrix @ x - params["rhs"]

    def jacobian(_x, _params):
        return matrix

    def solve(rhs):
        return newton_with_implicit_vjp(
            residual,
            jnp.zeros((2,), dtype=jnp.float64),
            {"rhs": rhs},
            max_iter=1,
            tol=1e-13,
            jac=jacobian,
        )

    rhs = jnp.array([1.0, -2.0], dtype=jnp.float64)
    weights = jnp.array([0.4, -1.3], dtype=jnp.float64)
    root = solve(rhs)
    gradient = jax.grad(lambda rhs_arg: jnp.vdot(weights, solve(rhs_arg)))(rhs)

    expected_root = np.linalg.solve(np.asarray(matrix), np.asarray(rhs))
    expected_gradient = np.linalg.solve(np.asarray(matrix).T, np.asarray(weights))
    np.testing.assert_allclose(np.asarray(root), expected_root, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        np.asarray(gradient),
        expected_gradient,
        rtol=1e-12,
        atol=1e-12,
    )


def test_newton_with_implicit_vjp_transfer_guard_clean():
    """Oracle: compiled implicit solve runs without implicit host transfers."""

    def residual(x, param):
        return x * x - param

    compiled = jax.jit(
        lambda param: newton_with_implicit_vjp(
            residual,
            jnp.asarray(1.0, dtype=jnp.float64),
            param,
            max_iter=20,
            tol=1e-13,
        )
    )
    param = jnp.asarray(4.0, dtype=jnp.float64)
    compiled(param).block_until_ready()

    with jax.transfer_guard("disallow"):
        root = compiled(param)
        root.block_until_ready()

    np.testing.assert_allclose(np.asarray(root), 2.0, rtol=1e-12)
