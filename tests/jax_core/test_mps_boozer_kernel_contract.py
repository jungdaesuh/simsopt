from __future__ import annotations

from dataclasses import dataclass, replace
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import simsopt.jax_core.mps_boozer_kernel_contract as mps_boozer_kernel_contract
from simsopt.jax_core.mps_boozer_kernel_contract import (
    DEFAULT_CONTRACT_ARTIFACT_DIR,
    MAX_EXPERIMENTAL_MPS_BOOZER_FIXED_SURFACE_NEWTON_MAXITER,
    MPS_BOOZER_FIXED_SURFACE_G_IOTA_MODE,
    MpsBoozerFusedCustomCallResult,
    MpsBoozerKernelContract,
    MpsBoozerKernelStaticMetadata,
    SCHEMA_VERSION,
    SIMSOPT_MPS_BOOZER_VALUE_GRAD_CUSTOM_CALL_API_VERSION,
    SIMSOPT_MPS_BOOZER_VALUE_GRAD_TARGET,
    UNKNOWN_ITERATION_COUNT,
    _fused_custom_call_backend_config,
    build_mps_boozer_direct_kernel_contract,
    build_mps_boozer_fused_solve_state_payload,
    build_mps_boozer_fused_solve_value_and_grad,
    evaluate_mps_boozer_direct_cpu_oracle,
    evaluate_mps_boozer_fused_solve_custom_call,
    evaluate_mps_boozer_fused_solve_cpu_oracle,
    mps_boozer_fused_solve_state_payload,
    mps_boozer_kernel_contract_artifact,
    mps_boozer_fixed_surface_g_iota_supported,
    require_mps_boozer_fixed_surface_g_iota_supported,
    write_mps_boozer_kernel_contract_artifact,
)
from simsopt.jax_core.specs import CoilGroupSpec, GroupedCoilSetSpec

_TWO_PI = np.asarray(2.0 * np.pi, dtype=np.float32)


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


def _supported_fixed_surface_owner_fixture():
    owner, _ = _contract_fixture()
    owner.constraint_weight = 0.0
    owner.boozer_surface.options = {
        "gmres_maxiter": 2,
        "mps_solver_mode": MPS_BOOZER_FIXED_SURFACE_G_IOTA_MODE,
        "newton_maxiter": 1,
        "newton_tol": 1.0e-5,
    }
    solved_state = owner.boozer_surface.get_solved_runtime_state()
    return owner, build_mps_boozer_direct_kernel_contract(
        owner,
        solved_state=solved_state,
    )


def _fixed_surface_backend_contract_fixture():
    dtype = jnp.float32
    surface_dofs = jnp.asarray(
        np.linspace(-0.4, 0.8, 37, dtype=np.float32),
        dtype=dtype,
    )
    metadata = MpsBoozerKernelStaticMetadata(
        target_name=SIMSOPT_MPS_BOOZER_VALUE_GRAD_TARGET,
        mpol=2,
        ntor=2,
        nfp=2,
        stellsym=True,
        surface_kind="xyztensorfourier",
        label_mpol=2,
        label_ntor=2,
        label_nfp=2,
        label_stellsym=True,
        label_surface_kind="xyztensorfourier",
        label_type="theta",
        phi_idx=None,
        targetlabel=0.0,
        constraint_weight=0.0,
        optimize_G=True,
        weight_inv_modB=False,
        solver_options=(
            ("gmres_maxiter", 2),
            ("mps_solver_mode", MPS_BOOZER_FIXED_SURFACE_G_IOTA_MODE),
            ("newton_maxiter", 1),
            ("newton_tol", 1.0e-5),
        ),
        coil_group_indices=((0, 1),),
    )
    gammas = jnp.asarray(
        np.linspace(0.4, 3.1, 24, dtype=np.float32).reshape(2, 4, 3),
        dtype=dtype,
    )
    gammadashs = jnp.asarray(
        np.linspace(-0.3, 0.9, 24, dtype=np.float32).reshape(2, 4, 3),
        dtype=dtype,
    )
    return MpsBoozerKernelContract(
        coil_dofs=jnp.asarray([0.5, -0.25, 0.75], dtype=dtype),
        x_inner=jnp.concatenate((surface_dofs, jnp.asarray([0.3, 0.05], dtype=dtype))),
        surface_dofs=surface_dofs,
        quadpoints_phi=jnp.asarray([0.0, 0.25], dtype=dtype),
        quadpoints_theta=jnp.asarray([0.0, 0.5, 0.75], dtype=dtype),
        label_quadpoints_phi=jnp.asarray([0.0], dtype=dtype),
        label_quadpoints_theta=jnp.asarray([0.0, 0.5], dtype=dtype),
        surface_scatter_indices=jnp.asarray(
            np.eye(75, 37, dtype=np.float32),
            dtype=dtype,
        ),
        label_scatter_indices=jnp.asarray(
            np.eye(75, 37, dtype=np.float32),
            dtype=dtype,
        ),
        coil_group_gammas=(gammas,),
        coil_group_gammadashs=(gammadashs,),
        coil_group_currents=(jnp.asarray([1.0e8, -2.0e8], dtype=dtype),),
        coil_pullback_operator=jnp.asarray(
            np.linspace(-0.3, 0.4, 150, dtype=np.float32).reshape(3, 50),
            dtype=dtype,
        ),
        static_metadata=metadata,
    )


def _two_group_backend_contract_fixture():
    contract = _fixed_surface_backend_contract_fixture()
    dtype = contract.coil_dofs.dtype
    second_gammas = jnp.asarray(
        np.linspace(-0.8, 0.4, 18, dtype=np.float32).reshape(1, 6, 3),
        dtype=dtype,
    )
    second_gammadashs = jnp.asarray(
        np.linspace(0.9, -0.6, 18, dtype=np.float32).reshape(1, 6, 3),
        dtype=dtype,
    )
    second_currents = jnp.asarray([3.0e8], dtype=dtype)
    second_width = int(
        second_gammas.size + second_gammadashs.size + second_currents.size
    )
    return replace(
        contract,
        static_metadata=replace(
            contract.static_metadata,
            coil_group_indices=(
                contract.static_metadata.coil_group_indices[0],
                (2,),
            ),
        ),
        coil_group_gammas=(contract.coil_group_gammas[0], second_gammas),
        coil_group_gammadashs=(
            contract.coil_group_gammadashs[0],
            second_gammadashs,
        ),
        coil_group_currents=(contract.coil_group_currents[0], second_currents),
        coil_pullback_operator=jnp.concatenate(
            (
                contract.coil_pullback_operator,
                jnp.zeros((contract.coil_dofs.shape[0], second_width), dtype=dtype),
            ),
            axis=1,
        ),
    )


def _theta_basis(theta_grid, mpol):
    theta = _TWO_PI * theta_grid[:, None]
    m_cos = np.arange(0, mpol + 1, dtype=np.float32)[None, :]
    m_sin = np.arange(1, mpol + 1, dtype=np.float32)[None, :]
    cos_arg = theta * m_cos
    sin_arg = theta * m_sin
    basis = np.concatenate((np.cos(cos_arg), np.sin(sin_arg)), axis=1)
    derivative = np.concatenate(
        (-_TWO_PI * m_cos * np.sin(cos_arg), _TWO_PI * m_sin * np.cos(sin_arg)),
        axis=1,
    )
    return basis, derivative


def _phi_basis(phi_grid, ntor, nfp):
    phi = _TWO_PI * phi_grid[:, None]
    n_cos = (np.arange(0, ntor + 1, dtype=np.float32) * np.float32(nfp))[None, :]
    n_sin = (np.arange(1, ntor + 1, dtype=np.float32) * np.float32(nfp))[None, :]
    cos_arg = phi * n_cos
    sin_arg = phi * n_sin
    basis = np.concatenate((np.cos(cos_arg), np.sin(sin_arg)), axis=1)
    derivative = np.concatenate(
        (-_TWO_PI * n_cos * np.sin(cos_arg), _TWO_PI * n_sin * np.cos(sin_arg)),
        axis=1,
    )
    return basis, derivative


def _eval_hat(phi_basis_value, theta_basis_value, coeffs):
    return phi_basis_value @ coeffs.T @ theta_basis_value.T


def _surface_geometry(contract):
    metadata = contract.static_metadata
    surface_dofs = np.asarray(contract.x_inner[:-2], dtype=np.float32)
    phi_grid = np.asarray(contract.quadpoints_phi, dtype=np.float32)
    theta_grid = np.asarray(contract.quadpoints_theta, dtype=np.float32)
    scatter_operator = np.asarray(contract.surface_scatter_indices, dtype=np.float32)
    theta_count = 2 * metadata.mpol + 1
    phi_count = 2 * metadata.ntor + 1
    coeff_count = theta_count * phi_count
    flat_coeffs = scatter_operator @ surface_dofs
    xc = flat_coeffs[:coeff_count].reshape(theta_count, phi_count)
    yc = flat_coeffs[coeff_count : 2 * coeff_count].reshape(theta_count, phi_count)
    zc = flat_coeffs[2 * coeff_count : 3 * coeff_count].reshape(
        theta_count,
        phi_count,
    )
    theta_b, theta_d = _theta_basis(theta_grid, metadata.mpol)
    phi_b, phi_d = _phi_basis(phi_grid, metadata.ntor, metadata.nfp)
    xhat = _eval_hat(phi_b, theta_b, xc)
    yhat = _eval_hat(phi_b, theta_b, yc)
    z = _eval_hat(phi_b, theta_b, zc)
    dxhat_dphi = _eval_hat(phi_d, theta_b, xc)
    dyhat_dphi = _eval_hat(phi_d, theta_b, yc)
    dz_dphi = _eval_hat(phi_d, theta_b, zc)
    dxhat_dtheta = _eval_hat(phi_b, theta_d, xc)
    dyhat_dtheta = _eval_hat(phi_b, theta_d, yc)
    dz_dtheta = _eval_hat(phi_b, theta_d, zc)
    phi_angle = _TWO_PI * phi_grid[:, None]
    cos_phi = np.cos(phi_angle)
    sin_phi = np.sin(phi_angle)
    x = xhat * cos_phi - yhat * sin_phi
    y = xhat * sin_phi + yhat * cos_phi
    dx_dphi = (
        dxhat_dphi * cos_phi
        - xhat * _TWO_PI * sin_phi
        - (dyhat_dphi * sin_phi + yhat * _TWO_PI * cos_phi)
    )
    dy_dphi = (
        dxhat_dphi * sin_phi
        + xhat * _TWO_PI * cos_phi
        + (dyhat_dphi * cos_phi - yhat * _TWO_PI * sin_phi)
    )
    dx_dtheta = dxhat_dtheta * cos_phi - dyhat_dtheta * sin_phi
    dy_dtheta = dxhat_dtheta * sin_phi + dyhat_dtheta * cos_phi
    return (
        np.stack((x, y, z), axis=2),
        np.stack((dx_dphi, dy_dphi, dz_dphi), axis=2),
        np.stack((dx_dtheta, dy_dtheta, dz_dtheta), axis=2),
    )


def _biot_savart_b(points, gammas, gammadashs, currents):
    diff = gammas[None, :, :, :] - points[:, None, None, :]
    radius_squared = np.sum(diff * diff, axis=-1)
    cross = np.cross(diff, gammadashs[None, :, :, :])
    inv_radius_cubed = (1.0 / np.sqrt(radius_squared)) / radius_squared
    weighted = cross * inv_radius_cubed[..., None]
    weighted = weighted * currents[None, :, None, None]
    return np.float32(1.0e-7) * np.sum(weighted, axis=(1, 2)) / gammas.shape[1]


def _biot_savart_vjp(points, field_cotangent, gammas, gammadashs, currents):
    diff = gammas[None, :, :, :] - points[:, None, None, :]
    radius_squared = np.sum(diff * diff, axis=-1)
    inv_radius_cubed = (1.0 / np.sqrt(radius_squared)) / radius_squared
    cross = np.cross(diff, gammadashs[None, :, :, :])
    dot_cotangent_cross = np.sum(field_cotangent[:, None, None, :] * cross, axis=-1)
    scale = np.float32(1.0e-7) / gammas.shape[1]
    grad_gammas = scale * np.sum(
        currents[None, :, None, None]
        * (
            inv_radius_cubed[..., None]
            * np.cross(gammadashs[None, :, :, :], field_cotangent[:, None, None, :])
            - 3.0
            * (dot_cotangent_cross * inv_radius_cubed / radius_squared)[..., None]
            * diff
        ),
        axis=0,
    )
    grad_gammadashs = scale * np.sum(
        currents[None, :, None, None]
        * inv_radius_cubed[..., None]
        * np.cross(field_cotangent[:, None, None, :], diff),
        axis=0,
    )
    grad_currents = scale * np.sum(dot_cotangent_cross * inv_radius_cubed, axis=(0, 2))
    return grad_gammas, grad_gammadashs, grad_currents


def _fixed_surface_direct_oracle(contract):
    x_inner = np.asarray(contract.x_inner, dtype=np.float32)
    gammas = np.asarray(contract.coil_group_gammas[0], dtype=np.float32)
    gammadashs = np.asarray(contract.coil_group_gammadashs[0], dtype=np.float32)
    currents = np.asarray(contract.coil_group_currents[0], dtype=np.float32)
    coil_pullback_operator = np.asarray(
        contract.coil_pullback_operator,
        dtype=np.float32,
    )
    surface_dofs = x_inner[:-2]
    iota = x_inner[-2]
    G = x_inner[-1]
    gamma, xphi, xtheta = _surface_geometry(contract)
    points = gamma.reshape(-1, 3)
    B = _biot_savart_b(points, gammas, gammadashs, currents).reshape(gamma.shape)
    B2 = np.sum(B * B, axis=2)
    residual = G * B - B2[..., None] * (xphi + iota * xtheta)
    jacobian_G = B.reshape(-1)
    jacobian_iota = (-B2[..., None] * xtheta).reshape(-1)
    residual_flat = residual.reshape(-1)
    grad_G = np.sum(jacobian_G * residual_flat)
    grad_iota = np.sum(jacobian_iota * residual_flat)
    h_GG = np.sum(jacobian_G * jacobian_G)
    h_GI = np.sum(jacobian_G * jacobian_iota)
    h_II = np.sum(jacobian_iota * jacobian_iota)
    rhs_G = -grad_G
    rhs_iota = -grad_iota
    determinant = h_GG * h_II - h_GI * h_GI
    G = G + (rhs_G * h_II - h_GI * rhs_iota) / determinant
    iota = iota + (h_GG * rhs_iota - h_GI * rhs_G) / determinant
    final_x_inner = np.concatenate(
        (surface_dofs, np.asarray([iota, G], dtype=np.float32))
    )

    tangent = xphi + iota * xtheta
    residual = G * B - B2[..., None] * tangent
    value = np.asarray(
        0.5 * np.sum(residual * residual) / residual.size, dtype=np.float32
    )
    residual_cotangent = residual / np.float32(residual.size)
    grad_B = (
        G * residual_cotangent
        - 2.0
        * np.sum(
            residual_cotangent * tangent,
            axis=2,
        )[..., None]
        * B
    )
    grad_gammas, grad_gammadashs, grad_currents = _biot_savart_vjp(
        points,
        grad_B.reshape(-1, 3),
        gammas,
        gammadashs,
        currents,
    )
    flat_cotangent = np.concatenate(
        (
            grad_gammas.reshape(-1),
            grad_gammadashs.reshape(-1),
            grad_currents.reshape(-1),
        )
    )
    coil_gradient = coil_pullback_operator @ flat_cotangent
    residual_norm = np.asarray(np.sqrt(np.sum(residual * residual)), dtype=np.float32)
    solver_grad_G = np.sum(jacobian_G * residual.reshape(-1))
    solver_grad_iota = np.sum(jacobian_iota * residual.reshape(-1))
    gradient_norm = np.asarray(
        np.sqrt(solver_grad_G * solver_grad_G + solver_grad_iota * solver_grad_iota),
        dtype=np.float32,
    )
    return value, coil_gradient, final_x_inner, residual_norm, gradient_norm


def _fused_custom_call_with_arrays(contract, *arrays):
    common_arrays = arrays[:9]
    group_arrays = arrays[9:-1]
    coil_pullback_operator = arrays[-1]
    if len(group_arrays) % 3 != 0:
        raise AssertionError("group arrays must be triples")
    coil_group_gammas = tuple(group_arrays[0::3])
    coil_group_gammadashs = tuple(group_arrays[1::3])
    coil_group_currents = tuple(group_arrays[2::3])
    traced_contract = replace(
        contract,
        coil_dofs=common_arrays[0],
        x_inner=common_arrays[1],
        surface_dofs=common_arrays[2],
        quadpoints_phi=common_arrays[3],
        quadpoints_theta=common_arrays[4],
        label_quadpoints_phi=common_arrays[5],
        label_quadpoints_theta=common_arrays[6],
        surface_scatter_indices=common_arrays[7],
        label_scatter_indices=common_arrays[8],
        coil_group_gammas=coil_group_gammas,
        coil_group_gammadashs=coil_group_gammadashs,
        coil_group_currents=coil_group_currents,
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
        *[
            leaf
            for group_leaves in zip(
                contract.coil_group_gammas,
                contract.coil_group_gammadashs,
                contract.coil_group_currents,
                strict=True,
            )
            for leaf in group_leaves
        ],
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


def test_mps_boozer_fused_custom_call_lowers_two_group_fixture():
    contract = _two_group_backend_contract_fixture()
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

    assert mps_boozer_fixed_surface_g_iota_supported(contract)
    assert "stablehlo.custom_call" in lowered
    assert f"@{SIMSOPT_MPS_BOOZER_VALUE_GRAD_TARGET}" in lowered
    assert result_shape.coil_gradient.shape == contract.coil_dofs.shape


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

    with pytest.raises(ValueError, match="one or two coil groups"):
        evaluate_mps_boozer_fused_solve_custom_call(
            replace(
                contract,
                coil_group_gammas=(
                    contract.coil_group_gammas[0],
                    contract.coil_group_gammas[0],
                    contract.coil_group_gammas[0],
                ),
                coil_group_gammadashs=(
                    contract.coil_group_gammadashs[0],
                    contract.coil_group_gammadashs[0],
                    contract.coil_group_gammadashs[0],
                ),
                coil_group_currents=(
                    contract.coil_group_currents[0],
                    contract.coil_group_currents[0],
                    contract.coil_group_currents[0],
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


def test_mps_boozer_fixed_surface_support_predicate_matches_backend_contract():
    _, unsupported_contract = _contract_fixture()
    supported_contract = _fixed_surface_backend_contract_fixture()
    two_group_contract = _two_group_backend_contract_fixture()

    assert not mps_boozer_fixed_surface_g_iota_supported(unsupported_contract)
    assert mps_boozer_fixed_surface_g_iota_supported(supported_contract)
    assert mps_boozer_fixed_surface_g_iota_supported(two_group_contract)
    assert mps_boozer_fixed_surface_g_iota_supported(
        replace(
            supported_contract,
            static_metadata=replace(
                supported_contract.static_metadata,
                solver_options=(
                    ("gmres_maxiter", 2),
                    ("mps_solver_mode", MPS_BOOZER_FIXED_SURFACE_G_IOTA_MODE),
                    ("newton_maxiter", 3),
                    ("newton_tol", 1.0e-5),
                ),
            ),
        )
    )
    assert not mps_boozer_fixed_surface_g_iota_supported(
        replace(
            supported_contract,
            static_metadata=replace(
                supported_contract.static_metadata,
                solver_options=(
                    ("gmres_maxiter", 2),
                    ("mps_solver_mode", MPS_BOOZER_FIXED_SURFACE_G_IOTA_MODE),
                    (
                        "newton_maxiter",
                        MAX_EXPERIMENTAL_MPS_BOOZER_FIXED_SURFACE_NEWTON_MAXITER + 1,
                    ),
                    ("newton_tol", 1.0e-5),
                ),
            ),
        )
    )
    assert not mps_boozer_fixed_surface_g_iota_supported(
        replace(
            supported_contract,
            coil_pullback_operator=supported_contract.coil_pullback_operator[:, :-1],
        )
    )


def test_mps_boozer_fixed_surface_require_raises_for_unsupported_contract():
    _, unsupported_contract = _contract_fixture()

    with pytest.raises(ValueError, match="only supports the fixed-surface G/iota"):
        require_mps_boozer_fixed_surface_g_iota_supported(unsupported_contract)


def test_mps_boozer_value_and_grad_builder_requires_mps_backend(monkeypatch):
    owner, _ = _supported_fixed_surface_owner_fixture()
    monkeypatch.setattr(
        mps_boozer_kernel_contract,
        "mps_boozer_jax_mps_backend_available",
        lambda: False,
    )

    with pytest.raises(RuntimeError, match="requires an active jax-mps backend"):
        build_mps_boozer_fused_solve_value_and_grad(owner)


def test_mps_boozer_value_and_grad_builder_rejects_unsupported_fixture(
    monkeypatch,
):
    owner, _ = _contract_fixture()
    monkeypatch.setattr(
        mps_boozer_kernel_contract,
        "mps_boozer_jax_mps_backend_available",
        lambda: True,
    )

    with pytest.raises(ValueError, match="only supports the fixed-surface G/iota"):
        build_mps_boozer_fused_solve_value_and_grad(owner)


def test_mps_boozer_value_and_grad_builder_reuses_value_grad_marker(
    monkeypatch,
):
    owner, contract = _supported_fixed_surface_owner_fixture()
    monkeypatch.setattr(
        mps_boozer_kernel_contract,
        "mps_boozer_jax_mps_backend_available",
        lambda: True,
    )

    value_and_grad = build_mps_boozer_fused_solve_value_and_grad(owner)
    lowered = jax.jit(value_and_grad).lower(contract.coil_dofs).as_text()
    result_shape = jax.eval_shape(value_and_grad, contract.coil_dofs)

    assert getattr(value_and_grad, "_simsopt_value_and_grad") is True
    assert getattr(value_and_grad, "_simsopt_mps_boozer_custom_kernel") is True
    assert "stablehlo.custom_call" in lowered
    assert f"@{SIMSOPT_MPS_BOOZER_VALUE_GRAD_TARGET}" in lowered
    assert result_shape[0].shape == ()
    assert result_shape[1].shape == contract.coil_dofs.shape


def test_mps_boozer_fused_solve_state_payload_splits_final_state():
    _owner, contract = _supported_fixed_surface_owner_fixture()
    perturbation = jnp.asarray(0.125, dtype=contract.x_inner.dtype)
    final_x_inner = contract.x_inner.at[0].set(contract.x_inner[0] + perturbation)
    result = MpsBoozerFusedCustomCallResult(
        value=jnp.asarray(3.0, dtype=contract.coil_dofs.dtype),
        coil_gradient=jnp.ones_like(contract.coil_dofs),
        final_x_inner=final_x_inner,
        residual_norm=jnp.asarray(0.0, dtype=contract.coil_dofs.dtype),
        gradient_norm=jnp.asarray(2.0, dtype=contract.coil_dofs.dtype),
        newton_iteration_count=jnp.asarray(1, dtype=jnp.int32),
        gmres_iteration_count=jnp.asarray(2, dtype=jnp.int32),
        converged=jnp.asarray(False),
        finite=jnp.asarray(True),
    )

    payload = mps_boozer_fused_solve_state_payload(contract, result)

    surface_dof_count = final_x_inner.shape[0] - 2
    np.testing.assert_allclose(
        np.asarray(payload.sdofs), np.asarray(final_x_inner[:surface_dof_count])
    )
    np.testing.assert_allclose(
        np.asarray(payload.iota), np.asarray(final_x_inner[surface_dof_count])
    )
    np.testing.assert_allclose(
        np.asarray(payload.G), np.asarray(final_x_inner[surface_dof_count + 1])
    )
    np.testing.assert_allclose(np.asarray(payload.value), np.asarray(result.value))
    np.testing.assert_allclose(
        np.asarray(payload.coil_gradient), np.asarray(result.coil_gradient)
    )
    np.testing.assert_allclose(np.asarray(payload.x), np.asarray(final_x_inner))
    assert not bool(payload.success)
    assert not bool(payload.primal_success)
    assert not bool(payload.converged)
    assert bool(payload.finite)


def test_mps_boozer_fused_solve_state_builder_reuses_custom_call_result(
    monkeypatch,
):
    owner, contract = _supported_fixed_surface_owner_fixture()
    monkeypatch.setattr(
        mps_boozer_kernel_contract,
        "mps_boozer_jax_mps_backend_available",
        lambda: True,
    )

    def evaluate_status(_contract):
        return MpsBoozerFusedCustomCallResult(
            value=jnp.asarray(3.0, dtype=contract.coil_dofs.dtype),
            coil_gradient=jnp.ones_like(contract.coil_dofs),
            final_x_inner=contract.x_inner,
            residual_norm=jnp.asarray(0.0, dtype=contract.coil_dofs.dtype),
            gradient_norm=jnp.asarray(2.0, dtype=contract.coil_dofs.dtype),
            newton_iteration_count=jnp.asarray(1, dtype=jnp.int32),
            gmres_iteration_count=jnp.asarray(2, dtype=jnp.int32),
            converged=jnp.asarray(False),
            finite=jnp.asarray(True),
        )

    monkeypatch.setattr(
        mps_boozer_kernel_contract,
        "evaluate_mps_boozer_fused_solve_custom_call",
        evaluate_status,
    )

    payload = build_mps_boozer_fused_solve_state_payload(owner)(contract.coil_dofs)

    np.testing.assert_allclose(np.asarray(payload.value), np.asarray(3.0))
    np.testing.assert_allclose(
        np.asarray(payload.coil_gradient), np.ones_like(contract.coil_dofs)
    )
    np.testing.assert_allclose(np.asarray(payload.x), np.asarray(contract.x_inner))
    assert not bool(payload.success)
    assert not bool(payload.primal_success)
    assert not bool(payload.converged)
    assert bool(payload.finite)


def test_mps_boozer_fused_solve_state_builder_exposes_converged_result(
    monkeypatch,
):
    owner, contract = _supported_fixed_surface_owner_fixture()
    monkeypatch.setattr(
        mps_boozer_kernel_contract,
        "mps_boozer_jax_mps_backend_available",
        lambda: True,
    )

    def evaluate_success_status(_contract):
        return MpsBoozerFusedCustomCallResult(
            value=jnp.asarray(3.0, dtype=contract.coil_dofs.dtype),
            coil_gradient=jnp.ones_like(contract.coil_dofs),
            final_x_inner=contract.x_inner,
            residual_norm=jnp.asarray(0.0, dtype=contract.coil_dofs.dtype),
            gradient_norm=jnp.asarray(2.0, dtype=contract.coil_dofs.dtype),
            newton_iteration_count=jnp.asarray(1, dtype=jnp.int32),
            gmres_iteration_count=jnp.asarray(2, dtype=jnp.int32),
            converged=jnp.asarray(True),
            finite=jnp.asarray(True),
        )

    monkeypatch.setattr(
        mps_boozer_kernel_contract,
        "evaluate_mps_boozer_fused_solve_custom_call",
        evaluate_success_status,
    )

    payload = build_mps_boozer_fused_solve_state_payload(owner)(contract.coil_dofs)

    np.testing.assert_allclose(np.asarray(payload.value), np.asarray(3.0))
    np.testing.assert_allclose(
        np.asarray(payload.coil_gradient), np.ones_like(contract.coil_dofs)
    )
    np.testing.assert_allclose(np.asarray(payload.x), np.asarray(contract.x_inner))
    assert bool(payload.success)
    assert bool(payload.primal_success)
    assert bool(payload.converged)
    assert bool(payload.finite)


def test_mps_boozer_value_and_grad_builder_returns_converged_result(
    monkeypatch,
):
    owner, contract = _supported_fixed_surface_owner_fixture()
    monkeypatch.setattr(
        mps_boozer_kernel_contract,
        "mps_boozer_jax_mps_backend_available",
        lambda: True,
    )

    def evaluate_success_status(_contract):
        return MpsBoozerFusedCustomCallResult(
            value=jnp.asarray(3.0, dtype=contract.coil_dofs.dtype),
            coil_gradient=jnp.ones_like(contract.coil_dofs),
            final_x_inner=contract.x_inner,
            residual_norm=jnp.asarray(0.0, dtype=contract.coil_dofs.dtype),
            gradient_norm=jnp.asarray(2.0, dtype=contract.coil_dofs.dtype),
            newton_iteration_count=jnp.asarray(1, dtype=jnp.int32),
            gmres_iteration_count=jnp.asarray(2, dtype=jnp.int32),
            converged=jnp.asarray(True),
            finite=jnp.asarray(True),
        )

    monkeypatch.setattr(
        mps_boozer_kernel_contract,
        "evaluate_mps_boozer_fused_solve_custom_call",
        evaluate_success_status,
    )

    value, gradient = build_mps_boozer_fused_solve_value_and_grad(owner)(
        contract.coil_dofs,
    )

    assert float(value) == pytest.approx(3.0)
    np.testing.assert_allclose(np.asarray(gradient), np.ones(contract.coil_dofs.shape))


def test_mps_boozer_value_and_grad_builder_masks_finite_nonconverged_status(
    monkeypatch,
):
    owner, contract = _supported_fixed_surface_owner_fixture()
    monkeypatch.setattr(
        mps_boozer_kernel_contract,
        "mps_boozer_jax_mps_backend_available",
        lambda: True,
    )

    def evaluate_failed_status(_contract):
        return MpsBoozerFusedCustomCallResult(
            value=jnp.asarray(3.0, dtype=contract.coil_dofs.dtype),
            coil_gradient=jnp.ones_like(contract.coil_dofs),
            final_x_inner=contract.x_inner,
            residual_norm=jnp.asarray(0.0, dtype=contract.coil_dofs.dtype),
            gradient_norm=jnp.asarray(2.0, dtype=contract.coil_dofs.dtype),
            newton_iteration_count=jnp.asarray(1, dtype=jnp.int32),
            gmres_iteration_count=jnp.asarray(2, dtype=jnp.int32),
            converged=jnp.asarray(False),
            finite=jnp.asarray(True),
        )

    monkeypatch.setattr(
        mps_boozer_kernel_contract,
        "evaluate_mps_boozer_fused_solve_custom_call",
        evaluate_failed_status,
    )

    value, gradient = build_mps_boozer_fused_solve_value_and_grad(owner)(
        contract.coil_dofs,
    )

    assert bool(jnp.isnan(value))
    assert bool(jnp.all(jnp.isnan(gradient)))


def test_mps_boozer_value_and_grad_builder_masks_nonfinite_status(
    monkeypatch,
):
    owner, contract = _supported_fixed_surface_owner_fixture()
    monkeypatch.setattr(
        mps_boozer_kernel_contract,
        "mps_boozer_jax_mps_backend_available",
        lambda: True,
    )

    def evaluate_nonfinite_status(_contract):
        return MpsBoozerFusedCustomCallResult(
            value=jnp.asarray(3.0, dtype=contract.coil_dofs.dtype),
            coil_gradient=jnp.ones_like(contract.coil_dofs),
            final_x_inner=contract.x_inner,
            residual_norm=jnp.asarray(0.0, dtype=contract.coil_dofs.dtype),
            gradient_norm=jnp.asarray(2.0, dtype=contract.coil_dofs.dtype),
            newton_iteration_count=jnp.asarray(1, dtype=jnp.int32),
            gmres_iteration_count=jnp.asarray(2, dtype=jnp.int32),
            converged=jnp.asarray(True),
            finite=jnp.asarray(False),
        )

    monkeypatch.setattr(
        mps_boozer_kernel_contract,
        "evaluate_mps_boozer_fused_solve_custom_call",
        evaluate_nonfinite_status,
    )

    value, gradient = build_mps_boozer_fused_solve_value_and_grad(owner)(
        contract.coil_dofs,
    )

    assert bool(jnp.isnan(value))
    assert bool(jnp.all(jnp.isnan(gradient)))


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


@pytest.mark.mps
def test_mps_boozer_fixed_surface_custom_call_matches_oracle_on_real_mps_backend():
    mps_devices = tuple(
        device for device in jax.devices() if device.platform.lower() == "mps"
    )
    if not mps_devices:
        pytest.skip("requires a JAX MPS device")

    contract = _fixed_surface_backend_contract_fixture()
    assert mps_boozer_fixed_surface_g_iota_supported(contract)
    host_args = _fused_custom_call_args(contract)
    mps_args = tuple(jax.device_put(value, mps_devices[0]) for value in host_args)

    actual = jax.jit(lambda *leaves: _fused_custom_call_with_arrays(contract, *leaves))(
        *mps_args
    )
    jax.block_until_ready(actual)
    expected = _fixed_surface_direct_oracle(contract)

    for leaf in actual:
        assert leaf.device.platform.lower() == "mps"
    for actual_leaf, expected_leaf in (
        (actual.value, expected[0]),
        (actual.coil_gradient, expected[1]),
        (actual.residual_norm, expected[3]),
    ):
        np.testing.assert_allclose(
            np.asarray(actual_leaf),
            expected_leaf,
            rtol=1e-5,
            atol=1e-7,
        )
    np.testing.assert_allclose(
        np.asarray(actual.final_x_inner),
        expected[2],
        rtol=1e-5,
        atol=5e-6,
    )
    np.testing.assert_allclose(
        np.asarray(actual.gradient_norm),
        expected[4],
        rtol=1e-5,
        atol=5e-6,
    )
    assert int(np.asarray(actual.newton_iteration_count)) == 1
    assert int(np.asarray(actual.gmres_iteration_count)) == 2
    assert bool(np.asarray(actual.converged)) is True
    assert bool(np.asarray(actual.finite)) is True
