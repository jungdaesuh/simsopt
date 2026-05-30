from __future__ import annotations

from dataclasses import dataclass, replace
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from simsopt.jax_core.mps_boozer_kernel_contract import (
    DEFAULT_CONTRACT_ARTIFACT_DIR,
    SCHEMA_VERSION,
    SIMSOPT_MPS_BOOZER_VALUE_GRAD_CUSTOM_CALL_API_VERSION,
    SIMSOPT_MPS_BOOZER_VALUE_GRAD_TARGET,
    UNKNOWN_ITERATION_COUNT,
    _fused_custom_call_backend_config,
    build_mps_boozer_direct_kernel_contract,
    evaluate_mps_boozer_direct_cpu_oracle,
    evaluate_mps_boozer_fused_solve_custom_call,
    evaluate_mps_boozer_fused_solve_cpu_oracle,
    mps_boozer_kernel_contract_artifact,
    write_mps_boozer_kernel_contract_artifact,
)
from simsopt.jax_core.specs import CoilGroupSpec, GroupedCoilSetSpec


@dataclass(frozen=True)
class _SolvedState:
    iota: jax.Array
    G: jax.Array | None
    sdofs: jax.Array
    weight_inv_modB: bool


class _BoozerSurfaceStub:
    mpol = 2
    ntor = 2
    nfp = 2
    stellsym = True
    _surface_geometry_kind = "xyztensorfourier"
    label_mpol = 2
    label_ntor = 2
    label_nfp = 2
    label_stellsym = True
    _label_surface_geometry_kind = "xyztensorfourier"
    label_type = "volume"
    phi_idx = None
    targetlabel = 1.25
    options = {
        "optimizer_backend": "ondevice",
        "newton_maxiter": 20,
        "newton_tol": 1e-6,
    }

    def __init__(self):
        self.quadpoints_phi = jnp.asarray([0.0, 0.25], dtype=jnp.float32)
        self.quadpoints_theta = jnp.asarray([0.0, 0.5, 0.75], dtype=jnp.float32)
        self.label_quadpoints_phi = jnp.asarray([0.0], dtype=jnp.float32)
        self.label_quadpoints_theta = jnp.asarray([0.0, 0.5], dtype=jnp.float32)
        self.scatter_indices = jnp.asarray(
            np.eye(75, 37, dtype=np.float32),
            dtype=jnp.float32,
        )
        self.label_scatter_indices = jnp.asarray(
            np.eye(75, 37, dtype=np.float32),
            dtype=jnp.float32,
        )
        self.res = None
        self._solved_runtime_state = None

    def _pack_decision_vector(self, iota, G, *, sdofs):
        tail = jnp.asarray([iota], dtype=sdofs.dtype)
        if G is not None:
            tail = jnp.concatenate((tail, jnp.asarray([G], dtype=sdofs.dtype)))
        return jnp.concatenate((sdofs, tail))

    def get_solved_runtime_state(self):
        return self._solved_runtime_state


class _BiotSavartStub:
    def __init__(self):
        self.x = jnp.asarray([0.5, -0.25, 0.75], dtype=jnp.float32)
        self.pullback_operator = jnp.asarray(
            np.linspace(-0.3, 0.4, 150, dtype=np.float32).reshape(3, 50),
            dtype=jnp.float32,
        )

    def coil_set_spec_from_dofs(self, coil_dofs):
        offset = jnp.sum(coil_dofs) * jnp.asarray(0.0, dtype=coil_dofs.dtype)
        gammas = jnp.arange(24, dtype=coil_dofs.dtype).reshape(2, 4, 3) + offset
        gammadashs = gammas + jnp.asarray(0.5, dtype=coil_dofs.dtype)
        currents = jnp.asarray([1.0, -2.0], dtype=coil_dofs.dtype)
        return GroupedCoilSetSpec(
            groups=(
                CoilGroupSpec(
                    gammas=gammas,
                    gammadashs=gammadashs,
                    currents=currents,
                    coil_indices=(4, 7),
                ),
            ),
        )

    def coil_cotangents_to_dofs_gradient(
        self,
        d_coil_arrays,
        coil_indices,
        *,
        coil_dofs=None,
    ):
        del coil_indices, coil_dofs
        ((d_gammas, d_gammadashs, d_currents),) = d_coil_arrays
        flat_cotangent = jnp.concatenate(
            (
                d_gammas.reshape(-1),
                d_gammadashs.reshape(-1),
                d_currents.reshape(-1),
            )
        )
        return self.pullback_operator @ flat_cotangent


class _BoozerResidualStub:
    constraint_weight = 3.5

    def __init__(self):
        self.boozer_surface = _BoozerSurfaceStub()
        self.biotsavart = _BiotSavartStub()
        self._direct_objective_value_and_grad = self._value_and_grad

    def _inner_objective_state(self, iota, G, *, sdofs):
        return self.boozer_surface._pack_decision_vector(
            iota,
            G,
            sdofs=sdofs,
        ), G is not None

    def _value_and_grad(self, coil_dofs, x_inner, optimize_G, weight_inv_modB):
        objective_value = (
            jnp.sum(coil_dofs)
            + jnp.sum(x_inner)
            + jnp.asarray(float(optimize_G), dtype=coil_dofs.dtype)
        )
        gradient_scale = 2.0 if weight_inv_modB else -2.0
        return objective_value, coil_dofs * jnp.asarray(
            gradient_scale,
            dtype=coil_dofs.dtype,
        )

    def _value_and_dJ_by_dcoil_dofs(
        self, solved_state, current_coil_dofs, coil_set_spec
    ):
        del solved_state, coil_set_spec
        value = jnp.sum(current_coil_dofs) + jnp.asarray(
            7.0, dtype=current_coil_dofs.dtype
        )
        return value, current_coil_dofs * jnp.asarray(
            3.0, dtype=current_coil_dofs.dtype
        )


def _contract_fixture():
    solved_state = _SolvedState(
        iota=jnp.asarray(0.3, dtype=jnp.float32),
        G=jnp.asarray(0.05, dtype=jnp.float32),
        sdofs=jnp.asarray(
            np.linspace(-0.4, 0.8, 37, dtype=np.float32),
            dtype=jnp.float32,
        ),
        weight_inv_modB=True,
    )
    owner = _BoozerResidualStub()
    owner.boozer_surface._solved_runtime_state = solved_state
    owner.boozer_surface.res = {
        "success": True,
        "primal_success": True,
        "residual": jnp.asarray([0.25, -0.5], dtype=jnp.float32),
        "final_gradient_norm": jnp.asarray(3.0, dtype=jnp.float32),
        "iter": jnp.asarray(4, dtype=jnp.int32),
        "gmres_iteration_count": jnp.asarray(17, dtype=jnp.int32),
    }
    return owner, build_mps_boozer_direct_kernel_contract(
        owner,
        solved_state=solved_state,
    )


def _fused_custom_call_with_arrays(contract, *arrays):
    (
        coil_dofs,
        x_inner,
        surface_dofs,
        quadpoints_phi,
        quadpoints_theta,
        label_quadpoints_phi,
        label_quadpoints_theta,
        surface_scatter_indices,
        label_scatter_indices,
        gammas,
        gammadashs,
        currents,
        coil_pullback_operator,
    ) = arrays
    traced_contract = replace(
        contract,
        coil_dofs=coil_dofs,
        x_inner=x_inner,
        surface_dofs=surface_dofs,
        quadpoints_phi=quadpoints_phi,
        quadpoints_theta=quadpoints_theta,
        label_quadpoints_phi=label_quadpoints_phi,
        label_quadpoints_theta=label_quadpoints_theta,
        surface_scatter_indices=surface_scatter_indices,
        label_scatter_indices=label_scatter_indices,
        coil_group_gammas=(gammas,),
        coil_group_gammadashs=(gammadashs,),
        coil_group_currents=(currents,),
        coil_pullback_operator=coil_pullback_operator,
    )
    return evaluate_mps_boozer_fused_solve_custom_call(traced_contract)


def _fused_custom_call_args(contract):
    return (
        contract.coil_dofs,
        contract.x_inner,
        contract.surface_dofs,
        contract.quadpoints_phi,
        contract.quadpoints_theta,
        contract.label_quadpoints_phi,
        contract.label_quadpoints_theta,
        contract.surface_scatter_indices,
        contract.label_scatter_indices,
        contract.coil_group_gammas[0],
        contract.coil_group_gammadashs[0],
        contract.coil_group_currents[0],
        contract.coil_pullback_operator,
    )


def test_mps_boozer_contract_artifact_records_flattened_schema(tmp_path):
    owner, contract = _contract_fixture()
    oracle_result = evaluate_mps_boozer_direct_cpu_oracle(owner, contract)
    fused_oracle_result = evaluate_mps_boozer_fused_solve_cpu_oracle(owner, contract)

    artifact = mps_boozer_kernel_contract_artifact(
        contract,
        oracle_result=oracle_result,
        fused_oracle_result=fused_oracle_result,
    )

    assert DEFAULT_CONTRACT_ARTIFACT_DIR.name == "mps_custom_kernel_contract"
    assert artifact["schema_version"] == SCHEMA_VERSION
    assert artifact["target_name"] == "mps.simsopt_boozer_value_grad"
    assert artifact["output_contract"] == {
        "unknown_iteration_count": UNKNOWN_ITERATION_COUNT,
        "fused_solve_fields": [
            "value",
            "coil_gradient",
            "final_x_inner",
            "residual_norm",
            "gradient_norm",
            "newton_iteration_count",
            "gmres_iteration_count",
            "converged",
            "finite",
        ],
    }
    assert artifact["runtime_arrays"]["coil_dofs"] == {
        "shape": [3],
        "dtype": "float32",
        "ndim": 1,
        "size": 3,
    }
    assert artifact["runtime_arrays"]["x_inner"]["shape"] == [39]
    assert artifact["runtime_arrays"]["surface_dofs"]["shape"] == [37]
    assert artifact["runtime_arrays"]["surface_scatter_indices"] == {
        "shape": [75, 37],
        "dtype": "float32",
        "ndim": 2,
        "size": 2775,
    }
    assert artifact["runtime_arrays"]["label_scatter_indices"] == {
        "shape": [75, 37],
        "dtype": "float32",
        "ndim": 2,
        "size": 2775,
    }
    assert artifact["runtime_arrays"]["coil_pullback_operator"] == {
        "shape": [3, 50],
        "dtype": "float32",
        "ndim": 2,
        "size": 150,
    }
    assert artifact["runtime_arrays"]["coil_groups"] == [
        {
            "group_index": 0,
            "coil_indices": [4, 7],
            "gammas": {"shape": [2, 4, 3], "dtype": "float32", "ndim": 3, "size": 24},
            "gammadashs": {
                "shape": [2, 4, 3],
                "dtype": "float32",
                "ndim": 3,
                "size": 24,
            },
            "currents": {"shape": [2], "dtype": "float32", "ndim": 1, "size": 2},
        }
    ]
    assert artifact["static_metadata"]["mpol"] == owner.boozer_surface.mpol
    assert artifact["static_metadata"]["solver_options"] == [
        ["newton_maxiter", 20],
        ["newton_tol", 1e-06],
        ["optimizer_backend", "ondevice"],
    ]
    assert artifact["oracle_outputs"] == {
        "value": {"shape": [], "dtype": "float32", "ndim": 0, "size": 1},
        "gradient": {"shape": [3], "dtype": "float32", "ndim": 1, "size": 3},
    }
    assert artifact["fused_oracle_outputs"] == {
        "value": {"shape": [], "dtype": "float32", "ndim": 0, "size": 1},
        "coil_gradient": {
            "shape": [3],
            "dtype": "float32",
            "ndim": 1,
            "size": 3,
        },
        "final_x_inner": {"shape": [39], "dtype": "float32", "ndim": 1, "size": 39},
        "residual_norm": {"shape": [], "dtype": "float32", "ndim": 0, "size": 1},
        "gradient_norm": {"shape": [], "dtype": "float32", "ndim": 0, "size": 1},
        "newton_iteration_count": {
            "shape": [],
            "dtype": "int32",
            "ndim": 0,
            "size": 1,
        },
        "gmres_iteration_count": {
            "shape": [],
            "dtype": "int32",
            "ndim": 0,
            "size": 1,
        },
        "converged": {"shape": [], "dtype": "bool", "ndim": 0, "size": 1},
        "finite": {"shape": [], "dtype": "bool", "ndim": 0, "size": 1},
    }

    output_path = tmp_path / "contract.json"
    write_mps_boozer_kernel_contract_artifact(
        contract,
        output_path,
        oracle_result=oracle_result,
        fused_oracle_result=fused_oracle_result,
    )

    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact


def test_mps_boozer_direct_cpu_oracle_delegates_to_existing_value_grad():
    owner, contract = _contract_fixture()

    observed = evaluate_mps_boozer_direct_cpu_oracle(owner, contract)
    expected_value, expected_gradient = owner._direct_objective_value_and_grad(
        contract.coil_dofs,
        contract.x_inner,
        contract.static_metadata.optimize_G,
        contract.static_metadata.weight_inv_modB,
    )

    np.testing.assert_allclose(np.asarray(observed.value), np.asarray(expected_value))
    np.testing.assert_allclose(
        np.asarray(observed.gradient),
        np.asarray(expected_gradient),
    )


def test_mps_boozer_fused_solve_cpu_oracle_records_full_solved_outputs():
    owner, contract = _contract_fixture()

    observed = evaluate_mps_boozer_fused_solve_cpu_oracle(owner, contract)

    np.testing.assert_allclose(np.asarray(observed.value), np.asarray(7.0 + 1.0))
    np.testing.assert_allclose(
        np.asarray(observed.coil_gradient),
        np.asarray(contract.coil_dofs) * 3.0,
    )
    np.testing.assert_allclose(
        np.asarray(observed.final_x_inner),
        np.asarray(contract.x_inner),
    )
    np.testing.assert_allclose(
        np.asarray(observed.residual_norm),
        np.linalg.norm(np.asarray(owner.boozer_surface.res["residual"])),
    )
    np.testing.assert_allclose(
        np.asarray(observed.gradient_norm),
        np.asarray(owner.boozer_surface.res["final_gradient_norm"]),
    )
    assert int(np.asarray(observed.newton_iteration_count)) == 4
    assert int(np.asarray(observed.gmres_iteration_count)) == 17
    assert bool(np.asarray(observed.converged)) is True
    assert bool(np.asarray(observed.finite)) is True


def test_mps_boozer_fused_solve_cpu_oracle_rejects_stale_coil_dofs():
    owner, contract = _contract_fixture()
    stale_contract = replace(
        contract,
        coil_dofs=contract.coil_dofs + jnp.asarray(1.0, dtype=contract.coil_dofs.dtype),
    )

    with pytest.raises(ValueError, match="requires contract.coil_dofs to match"):
        evaluate_mps_boozer_fused_solve_cpu_oracle(owner, stale_contract)


def test_mps_boozer_fused_custom_call_lowers_to_named_stablehlo_target():
    _, contract = _contract_fixture()
    args = _fused_custom_call_args(contract)

    lowered = (
        jax.jit(lambda *leaves: _fused_custom_call_with_arrays(contract, *leaves))
        .lower(*args)
        .as_text()
    )
    result_shape = jax.eval_shape(
        lambda *leaves: _fused_custom_call_with_arrays(contract, *leaves),
        *args,
    )

    assert "stablehlo.custom_call" in lowered
    assert f"@{SIMSOPT_MPS_BOOZER_VALUE_GRAD_TARGET}" in lowered
    assert (
        f"api_version = {SIMSOPT_MPS_BOOZER_VALUE_GRAD_CUSTOM_CALL_API_VERSION}"
        in lowered
    )
    assert "schema_version" in lowered
    backend_config = json.loads(_fused_custom_call_backend_config(contract))
    assert backend_config["schema_version"] == SCHEMA_VERSION
    assert backend_config["target_name"] == SIMSOPT_MPS_BOOZER_VALUE_GRAD_TARGET
    assert backend_config["static_metadata"]["mpol"] == contract.static_metadata.mpol
    assert backend_config["static_metadata"]["weight_inv_modB"] == (
        contract.static_metadata.weight_inv_modB
    )
    assert result_shape.value.shape == ()
    assert result_shape.value.dtype == jnp.float32
    assert result_shape.coil_gradient.shape == contract.coil_dofs.shape
    assert result_shape.final_x_inner.shape == contract.x_inner.shape
    assert result_shape.residual_norm.shape == ()
    assert result_shape.gradient_norm.shape == ()
    assert result_shape.newton_iteration_count.dtype == jnp.int32
    assert result_shape.gmres_iteration_count.dtype == jnp.int32
    assert result_shape.converged.dtype == jnp.bool_
    assert result_shape.finite.dtype == jnp.bool_


def test_mps_boozer_fused_custom_call_rejects_unsupported_contracts():
    _, contract = _contract_fixture()
    with pytest.raises(ValueError, match="coil_dofs must be float32"):
        evaluate_mps_boozer_fused_solve_custom_call(
            replace(contract, coil_dofs=contract.coil_dofs.astype(jnp.float16))
        )

    with pytest.raises(ValueError, match="2D float32 scatter operator"):
        evaluate_mps_boozer_fused_solve_custom_call(
            replace(
                contract,
                surface_scatter_indices=contract.surface_scatter_indices.reshape(-1),
            )
        )

    with pytest.raises(ValueError, match="exactly one coil group"):
        evaluate_mps_boozer_fused_solve_custom_call(
            replace(
                contract,
                coil_group_gammas=(
                    contract.coil_group_gammas[0],
                    contract.coil_group_gammas[0],
                ),
            )
        )

    with pytest.raises(ValueError, match="coil_pullback_operator must have shape"):
        evaluate_mps_boozer_fused_solve_custom_call(
            replace(
                contract,
                coil_pullback_operator=contract.coil_pullback_operator[:, :-1],
            )
        )


@pytest.mark.mps
def test_mps_boozer_fused_custom_call_returns_structured_status_on_real_mps_backend():
    mps_devices = tuple(
        device for device in jax.devices() if device.platform.lower() == "mps"
    )
    if not mps_devices:
        pytest.skip("requires a JAX MPS device")

    _, contract = _contract_fixture()
    host_args = _fused_custom_call_args(contract)
    mps_args = tuple(jax.device_put(value, mps_devices[0]) for value in host_args)

    actual = jax.jit(lambda *leaves: _fused_custom_call_with_arrays(contract, *leaves))(
        *mps_args
    )
    jax.block_until_ready(actual)

    for leaf in actual:
        assert leaf.device.platform.lower() == "mps"
    assert bool(np.isnan(np.asarray(actual.value))) is True
    assert bool(np.all(np.isnan(np.asarray(actual.coil_gradient)))) is True
    assert bool(np.all(np.isnan(np.asarray(actual.final_x_inner)))) is True
    assert bool(np.isnan(np.asarray(actual.residual_norm))) is True
    assert bool(np.isnan(np.asarray(actual.gradient_norm))) is True
    assert int(np.asarray(actual.newton_iteration_count)) == UNKNOWN_ITERATION_COUNT
    assert int(np.asarray(actual.gmres_iteration_count)) == UNKNOWN_ITERATION_COUNT
    assert bool(np.asarray(actual.converged)) is False
    assert bool(np.asarray(actual.finite)) is False
