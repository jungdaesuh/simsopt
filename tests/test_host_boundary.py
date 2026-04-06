import jax
import jax.numpy as jnp
import numpy as np

from simsopt._core.jax_host_boundary import (
    explicit_cotangent_basis,
    host_array,
    host_scalar,
    host_tree,
    strict_scalar_grad,
    strict_scalar_value_and_grad,
)


def test_host_tree_materializes_jax_leaves():
    tree = {
        "x": jnp.asarray([1.0, 2.0], dtype=jnp.float64),
        "nested": (jnp.asarray(3.5, dtype=jnp.float64), np.asarray([4.0])),
    }

    result = host_tree(tree)

    assert isinstance(result["x"], np.ndarray)
    assert isinstance(result["nested"][0], np.ndarray)
    np.testing.assert_allclose(result["x"], np.array([1.0, 2.0]))
    assert host_scalar(result["nested"][0]) == 3.5


def test_host_tree_dtype_coerces_numeric_leaves():
    tree = {
        "x": jnp.asarray([1, 2], dtype=jnp.int32),
        "nested": (np.asarray([3], dtype=np.int16), 4),
    }

    result = host_tree(tree, dtype=np.float64)

    assert result["x"].dtype == np.float64
    assert result["nested"][0].dtype == np.float64
    assert result["nested"][1].dtype == np.float64
    np.testing.assert_allclose(result["x"], np.array([1.0, 2.0]))


def test_strict_scalar_grad_helpers_use_explicit_scalar_seed():
    def objective(x, scale):
        return scale * jnp.sum(x * x)

    x = jnp.asarray([1.0, -2.0], dtype=jnp.float64)
    value, grad = strict_scalar_value_and_grad(objective, x, 0.5)

    np.testing.assert_allclose(host_array(value), np.array(2.5))
    np.testing.assert_allclose(host_array(grad), np.array([1.0, -2.0]))
    np.testing.assert_allclose(
        host_array(strict_scalar_grad(lambda arg: objective(arg, 0.5), x)),
        np.array([1.0, -2.0]),
    )


def test_explicit_cotangent_basis_returns_runtime_unit_vector():
    basis = explicit_cotangent_basis(4, 2, dtype=jnp.float64)

    assert isinstance(basis, jax.Array)
    np.testing.assert_allclose(host_array(basis), np.array([0.0, 0.0, 1.0, 0.0]))
