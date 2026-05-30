from __future__ import annotations

from dataclasses import dataclass
import json

import jax
import jax.numpy as jnp
import numpy as np

from simsopt.jax_core.mps_boozer_kernel_contract import (
    DEFAULT_CONTRACT_ARTIFACT_DIR,
    SCHEMA_VERSION,
    build_mps_boozer_direct_kernel_contract,
    evaluate_mps_boozer_direct_cpu_oracle,
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
    _surface_geometry_kind = "xyz_tensor_fourier"
    label_mpol = 2
    label_ntor = 2
    label_nfp = 2
    label_stellsym = True
    _label_surface_geometry_kind = "xyz_tensor_fourier"
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

    def _pack_decision_vector(self, iota, G, *, sdofs):
        tail = jnp.asarray([iota], dtype=sdofs.dtype)
        if G is not None:
            tail = jnp.concatenate((tail, jnp.asarray([G], dtype=sdofs.dtype)))
        return jnp.concatenate((sdofs, tail))


class _BiotSavartStub:
    def __init__(self):
        self.x = jnp.asarray([0.5, -0.25, 0.75], dtype=jnp.float32)

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


def _contract_fixture():
    solved_state = _SolvedState(
        iota=jnp.asarray(0.3, dtype=jnp.float32),
        G=jnp.asarray(0.05, dtype=jnp.float32),
        sdofs=jnp.asarray([0.1, 0.2, -0.4, 0.8], dtype=jnp.float32),
        weight_inv_modB=True,
    )
    owner = _BoozerResidualStub()
    return owner, build_mps_boozer_direct_kernel_contract(
        owner,
        solved_state=solved_state,
    )


def test_mps_boozer_contract_artifact_records_flattened_schema(tmp_path):
    owner, contract = _contract_fixture()
    oracle_result = evaluate_mps_boozer_direct_cpu_oracle(owner, contract)

    artifact = mps_boozer_kernel_contract_artifact(
        contract,
        oracle_result=oracle_result,
    )

    assert DEFAULT_CONTRACT_ARTIFACT_DIR.name == "mps_custom_kernel_contract"
    assert artifact["schema_version"] == SCHEMA_VERSION
    assert artifact["target_name"] == "mps.simsopt_boozer_value_grad"
    assert artifact["runtime_arrays"]["coil_dofs"] == {
        "shape": [3],
        "dtype": "float32",
        "ndim": 1,
        "size": 3,
    }
    assert artifact["runtime_arrays"]["x_inner"]["shape"] == [6]
    assert artifact["runtime_arrays"]["surface_dofs"]["shape"] == [4]
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

    output_path = tmp_path / "contract.json"
    write_mps_boozer_kernel_contract_artifact(
        contract,
        output_path,
        oracle_result=oracle_result,
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
