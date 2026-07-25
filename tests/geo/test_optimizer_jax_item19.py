"""Historical item-19 optimizer routing/import regression tests.

The filename tracks the original review item for optimizer dispatch cleanup;
the Boozer M1 Hessian split regression lives in ``test_boozer_residual_jax.py``.
"""

import inspect
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import simsopt_jax.geo.optimizers.optimizer as _opt


def test_item19_optimizer_jax_product_code_has_no_dynamic_private_import():
    source = Path(_opt.__file__).read_text()

    assert "import importlib" not in source
    assert "importlib.import_module" not in source
    assert "__import__(" not in source


def test_item19_optimizer_jax_exposes_no_dynamic_private_loader():
    assert not hasattr(_opt, "_private_pkg")
    assert not hasattr(_opt, "_load_private_pkg")
    assert not hasattr(_opt, "_load_reference_optimizer_module")


def test_item19_host_dense_hessian_is_independent_from_device_materializer(
    monkeypatch,
):
    def forbidden_device_materializer(*_args, **_kwargs):
        raise AssertionError("host materialization called the device materializer")

    monkeypatch.setattr(
        _opt,
        "_materialize_dense_hessian",
        forbidden_device_materializer,
    )

    operator = jnp.asarray([[2.0, 0.5], [0.5, 3.0]], dtype=jnp.float64)

    def hvp_fn(_x, vector):
        return operator @ vector

    actual = _opt._materialize_dense_hessian_host(
        hvp_fn,
        jnp.zeros(2, dtype=jnp.float64),
    )

    np.testing.assert_array_equal(np.asarray(actual), np.asarray(operator))


def test_item19_host_dense_hessian_agrees_with_device_materializer():
    operator = jnp.asarray(
        [
            [2.0, 0.5 + 1.0e-8, -0.25],
            [0.5 - 2.0e-8, 3.0, 0.75 + 3.0e-8],
            [-0.25, 0.75 - 1.0e-8, 4.0],
        ],
        dtype=jnp.float64,
    )

    def hvp_fn(_x, v):
        return operator @ v

    x = jnp.zeros(3, dtype=jnp.float64)
    host_hessian = _opt._materialize_dense_hessian_host(hvp_fn, x)
    device_hessian = _opt._materialize_dense_hessian(hvp_fn, x)

    np.testing.assert_allclose(
        np.asarray(host_hessian),
        np.asarray(device_hessian),
        rtol=1e-12,
        atol=1e-12,
    )


def test_item19_dense_lm_damping_avoids_identity_allocation():
    propose_step_source = inspect.getsource(_opt._dense_lm_propose_step)
    stabilize_source = inspect.getsource(_opt._stabilize_dense_hessian)

    assert "jnp.diag_indices" in propose_step_source
    assert "jnp.diag_indices" in stabilize_source
    assert "jnp.eye" not in propose_step_source
    assert "jnp.eye" not in stabilize_source


def test_item19_target_outer_loop_contract_defaults_to_ondevice_lbfgs():
    contract = _opt.resolve_target_outer_loop_optimizer_contract(
        "jax",
        "ondevice",
        component_label="item19 target optimizer",
    )

    assert contract == _opt.TargetOptimizerContract(
        driver=_opt.Driver.SIMSOPT_LBFGSB,
        use_least_squares_objective=False,
    )


@pytest.mark.parametrize(
    ("optimizer_backend", "limited_memory", "least_squares_algorithm", "driver"),
    [
        ("scipy", False, "quasi-newton", _opt.Driver.SCIPY_BFGS),
        ("scipy", True, "quasi-newton", _opt.Driver.SCIPY_LBFGSB),
        ("host-jax", False, "quasi-newton", _opt.Driver.SCIPY_BFGS),
        ("host-jax", True, "quasi-newton", _opt.Driver.SCIPY_LBFGSB),
        ("ondevice", False, "quasi-newton", _opt.Driver.SIMSOPT_BFGS),
        ("ondevice", True, "quasi-newton", _opt.Driver.SIMSOPT_LBFGSB),
        ("scipy", False, "lm", _opt.Driver.SIMSOPT_LM_GMRES_HOST),
        ("host-jax", False, "lm", _opt.Driver.SIMSOPT_LM_GMRES_HOST),
        ("ondevice", False, "lm", _opt.Driver.SIMSOPT_LM_GMRES),
        ("ondevice", False, "lm-minpack", _opt.Driver.SIMSOPT_LM_QR),
        ("ondevice", False, "optimistix-lm", _opt.Driver.OPTIMISTIX_LM),
    ],
)
def test_item19_boozer_inner_driver_contract_stays_typed(
    optimizer_backend,
    limited_memory,
    least_squares_algorithm,
    driver,
):
    assert (
        _opt.resolve_boozer_inner_driver(
            optimizer_backend,
            limited_memory=limited_memory,
            least_squares_algorithm=least_squares_algorithm,
        )
        == driver
    )


def test_item19_host_jax_least_squares_uses_explicit_dense_state(monkeypatch):
    monkeypatch.setattr(
        _opt,
        "require_boozer_inner_backend_x64",
        lambda optimizer_backend: None,
    )

    def residual_fn(_x):
        raise AssertionError("state_fn path must not call residual_fn directly")

    @jax.jit
    def dense_state_fn(x, dynamic_state):
        residual = dynamic_state["matrix"] @ x - dynamic_state["rhs"]
        return _opt._dense_lm_state_from_residual_jacobian(
            residual,
            dynamic_state["matrix"],
        )

    dynamic_state = {
        "matrix": jnp.asarray([[2.0, 0.0], [0.0, 4.0]], dtype=jnp.float64),
        "rhs": jnp.asarray([2.0, 8.0], dtype=jnp.float64),
    }

    result = _opt.host_jax_least_squares(
        residual_fn,
        jnp.asarray([0.0, 0.0], dtype=jnp.float64),
        method="lm",
        tol=1e-10,
        maxiter=20,
        options={"gtol": 1e-10, "materialize_dense_linearization": True},
        args=(dynamic_state,),
        state_fn=dense_state_fn,
    )

    assert result.success
    np.testing.assert_allclose(np.asarray(result.x), np.asarray([1.0, 2.0]), rtol=1e-8)
    np.testing.assert_allclose(np.asarray(result.residual), np.zeros(2), atol=1e-8)
    assert result.dense_linearization_kind == "in_loop"


def test_item19_host_jax_least_squares_uses_jacobian_blocks(monkeypatch):
    monkeypatch.setattr(
        _opt,
        "require_boozer_inner_backend_x64",
        lambda optimizer_backend: None,
    )

    @jax.jit
    def residual_fn(x, dynamic_state):
        return dynamic_state["matrix"] @ x - dynamic_state["rhs"]

    @jax.jit
    def jacobian_block_fn(_x, tangent_block, dynamic_state):
        return dynamic_state["matrix"] @ tangent_block.T

    dynamic_state = {
        "matrix": jnp.asarray([[3.0, 0.0], [0.0, 5.0]], dtype=jnp.float64),
        "rhs": jnp.asarray([6.0, 5.0], dtype=jnp.float64),
    }

    result = _opt.host_jax_least_squares(
        residual_fn,
        jnp.asarray([0.0, 0.0], dtype=jnp.float64),
        method="lm",
        tol=1e-10,
        maxiter=20,
        options={"gtol": 1e-10, "materialize_dense_linearization": True},
        args=(dynamic_state,),
        jacobian_block_fn=jacobian_block_fn,
        jacobian_chunk_size=1,
    )

    assert result.success
    np.testing.assert_allclose(np.asarray(result.x), np.asarray([2.0, 1.0]), rtol=1e-8)
    np.testing.assert_allclose(np.asarray(result.residual), np.zeros(2), atol=1e-8)
    assert result.dense_residual_jacobian_shape == (2, 2)


def test_item19_reference_and_target_optimizer_lanes_stay_explicit():
    reference_contract = _opt.resolve_reference_outer_loop_optimizer_contract(
        "cpu",
        "scipy",
        component_label="item19 reference optimizer",
    )
    assert reference_contract == _opt.ReferenceOptimizerContract(
        driver=_opt.Driver.SCIPY_LBFGSB,
    )

    with pytest.raises(ValueError, match="requires optimizer_backend='ondevice'"):
        _opt.resolve_target_outer_loop_optimizer_contract(
            "jax",
            "scipy",
            component_label="item19 target optimizer",
        )

    with pytest.raises(ValueError, match="SciPy/reference optimizer lane"):
        _opt.resolve_reference_outer_loop_optimizer_contract(
            "jax",
            "ondevice",
            component_label="item19 reference optimizer",
        )


def test_item19_reference_minimize_host_control_permission_is_explicit(monkeypatch):
    class _StrictJaxConfig:
        backend = "jax"
        mode = "jax_gpu_parity"

    monkeypatch.setattr(
        _opt,
        "get_backend_config",
        lambda: _StrictJaxConfig(),
    )

    def value_and_grad(x):
        return jnp.sum(x * x), 2.0 * x

    x0 = jnp.asarray([1.0, -2.0], dtype=jnp.float64)

    with pytest.raises(RuntimeError, match="requires an ondevice optimizer method"):
        _opt.reference_minimize(
            value_and_grad,
            x0,
            method="lbfgs",
            value_and_grad=True,
            maxiter=10,
        )

    result = _opt.reference_minimize(
        value_and_grad,
        x0,
        method="lbfgs",
        value_and_grad=True,
        maxiter=10,
        allow_jax_host_control=True,
    )

    assert result.success
    np.testing.assert_allclose(np.asarray(result.x), np.zeros(2), atol=1e-8)
