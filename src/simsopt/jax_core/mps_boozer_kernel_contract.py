"""Flattened SIMSOPT Boozer custom-kernel contract helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

SCHEMA_VERSION = 4
DEFAULT_CONTRACT_ARTIFACT_DIR = Path(".artifacts/mps_custom_kernel_contract")
SIMSOPT_MPS_BOOZER_VALUE_GRAD_TARGET = "mps.simsopt_boozer_value_grad"
SIMSOPT_MPS_BOOZER_VALUE_GRAD_CUSTOM_CALL_API_VERSION = 3
MPS_BOOZER_FIXED_SURFACE_G_IOTA_MODE = "fixed_surface_g_iota"
UNKNOWN_ITERATION_COUNT = -1
DEFAULT_EXPERIMENTAL_MPS_BOOZER_FIXED_SURFACE_NEWTON_MAXITER = 1
MAX_EXPERIMENTAL_MPS_BOOZER_FIXED_SURFACE_NEWTON_MAXITER = 20
EXPERIMENTAL_MPS_BOOZER_FIXED_SURFACE_GMRES_MAXITER = 2


@dataclass(frozen=True)
class MpsBoozerKernelStaticMetadata:
    """Static policy and shape metadata for the first Boozer MPS custom call."""

    target_name: str
    mpol: int
    ntor: int
    nfp: int
    stellsym: bool
    surface_kind: str
    label_mpol: int
    label_ntor: int
    label_nfp: int
    label_stellsym: bool
    label_surface_kind: str
    label_type: str
    phi_idx: int | None
    targetlabel: float
    constraint_weight: float
    optimize_G: bool
    weight_inv_modB: bool
    solver_options: tuple[tuple[str, object], ...]
    coil_group_indices: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class MpsBoozerKernelContract:
    """Runtime array leaves plus static metadata for the direct Boozer op."""

    coil_dofs: jax.Array
    x_inner: jax.Array
    surface_dofs: jax.Array
    quadpoints_phi: jax.Array
    quadpoints_theta: jax.Array
    label_quadpoints_phi: jax.Array
    label_quadpoints_theta: jax.Array
    surface_scatter_indices: jax.Array
    label_scatter_indices: jax.Array
    coil_group_gammas: tuple[jax.Array, ...]
    coil_group_gammadashs: tuple[jax.Array, ...]
    coil_group_currents: tuple[jax.Array, ...]
    coil_pullback_operator: jax.Array
    static_metadata: MpsBoozerKernelStaticMetadata


@dataclass(frozen=True)
class MpsBoozerDirectOracleResult:
    """CPU/JAX direct-objective oracle output for custom-call parity tests."""

    value: jax.Array
    gradient: jax.Array


@dataclass(frozen=True)
class MpsBoozerFusedSolveOracleResult:
    """CPU/JAX solved-state oracle output for the planned fused Boozer op."""

    value: jax.Array
    coil_gradient: jax.Array
    final_x_inner: jax.Array
    residual_norm: jax.Array
    gradient_norm: jax.Array
    newton_iteration_count: jax.Array
    gmres_iteration_count: jax.Array
    converged: jax.Array
    finite: jax.Array


class MpsBoozerFusedCustomCallResult(NamedTuple):
    """JAX-pytree output layout for the fused Boozer MPS custom call."""

    value: jax.Array
    coil_gradient: jax.Array
    final_x_inner: jax.Array
    residual_norm: jax.Array
    gradient_norm: jax.Array
    newton_iteration_count: jax.Array
    gmres_iteration_count: jax.Array
    converged: jax.Array
    finite: jax.Array


def _json_scalar(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_json_scalar(item) for item in value]
    if isinstance(value, list):
        return [_json_scalar(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_scalar(item) for key, item in value.items()}
    return value


def _array_schema(value: object) -> dict[str, object]:
    shape = tuple(int(dim) for dim in value.shape)
    return {
        "shape": list(shape),
        "dtype": str(value.dtype),
        "ndim": int(value.ndim),
        "size": int(np.prod(shape, dtype=np.int64)) if shape else 1,
    }


def _solver_option_items(options: object) -> tuple[tuple[str, object], ...]:
    return tuple(
        (str(key), _json_scalar(value))
        for key, value in sorted(dict(options).items(), key=lambda item: str(item[0]))
    )


def _solver_options_mapping(
    options: tuple[tuple[str, object], ...],
) -> dict[str, object]:
    return {str(key): value for key, value in options}


def _is_float32_array(value: jax.Array) -> bool:
    return value.dtype == np.dtype(np.float32)


def _is_nonempty_vector(value: jax.Array) -> bool:
    return value.ndim == 1 and value.shape[0] != 0


def _is_scatter_operand(value: jax.Array) -> bool:
    if value.dtype == np.dtype(np.int32):
        return _is_nonempty_vector(value)
    return bool(
        value.dtype == np.dtype(np.float32)
        and value.ndim == 2
        and value.shape[0] != 0
        and value.shape[1] != 0
    )


def _fused_custom_call_payload_supported(
    contract: MpsBoozerKernelContract,
) -> bool:
    runtime_arrays = (
        contract.coil_dofs,
        contract.x_inner,
        contract.surface_dofs,
        contract.quadpoints_phi,
        contract.quadpoints_theta,
        contract.label_quadpoints_phi,
        contract.label_quadpoints_theta,
    )
    if not all(
        _is_float32_array(value) and _is_nonempty_vector(value)
        for value in runtime_arrays
    ):
        return False
    if not _is_scatter_operand(contract.surface_scatter_indices):
        return False
    if not _is_scatter_operand(contract.label_scatter_indices):
        return False
    if (
        len(contract.coil_group_gammas) != 1
        or len(contract.coil_group_gammadashs) != 1
        or len(contract.coil_group_currents) != 1
    ):
        return False

    gammas = contract.coil_group_gammas[0]
    gammadashs = contract.coil_group_gammadashs[0]
    currents = contract.coil_group_currents[0]
    if not all(_is_float32_array(value) for value in (gammas, gammadashs, currents)):
        return False
    if gammas.ndim != 3 or gammas.shape[2] != 3:
        return False
    if gammas.shape[0] == 0 or gammas.shape[1] == 0:
        return False
    if gammadashs.shape != gammas.shape:
        return False
    if currents.ndim != 1 or currents.shape[0] != gammas.shape[0]:
        return False

    flat_cotangent_size = int(gammas.size + gammadashs.size + currents.size)
    return bool(
        _is_float32_array(contract.coil_pullback_operator)
        and contract.coil_pullback_operator.ndim == 2
        and contract.coil_pullback_operator.shape
        == (contract.coil_dofs.shape[0], flat_cotangent_size)
    )


def mps_boozer_fixed_surface_g_iota_supported(
    contract: MpsBoozerKernelContract,
) -> bool:
    """Return whether this contract can use the fixed-surface G/iota backend."""

    if not _fused_custom_call_payload_supported(contract):
        return False

    metadata = contract.static_metadata
    solver_options = _solver_options_mapping(metadata.solver_options)
    newton_maxiter = int(
        solver_options.get(
            "newton_maxiter",
            DEFAULT_EXPERIMENTAL_MPS_BOOZER_FIXED_SURFACE_NEWTON_MAXITER,
        )
    )
    gmres_maxiter = int(
        solver_options.get(
            "gmres_maxiter",
            EXPERIMENTAL_MPS_BOOZER_FIXED_SURFACE_GMRES_MAXITER,
        )
    )
    coeff_count = (2 * int(metadata.mpol) + 1) * (2 * int(metadata.ntor) + 1)
    surface_dof_count = int(contract.x_inner.shape[0]) - 2
    return bool(
        metadata.target_name == SIMSOPT_MPS_BOOZER_VALUE_GRAD_TARGET
        and bool(metadata.optimize_G)
        and bool(metadata.stellsym)
        and int(metadata.mpol) > 0
        and int(metadata.ntor) > 0
        and int(metadata.nfp) > 0
        and metadata.surface_kind == "xyztensorfourier"
        and float(metadata.constraint_weight) == 0.0
        and 1
        <= newton_maxiter
        <= MAX_EXPERIMENTAL_MPS_BOOZER_FIXED_SURFACE_NEWTON_MAXITER
        and gmres_maxiter == EXPERIMENTAL_MPS_BOOZER_FIXED_SURFACE_GMRES_MAXITER
        and solver_options.get("mps_solver_mode")
        == MPS_BOOZER_FIXED_SURFACE_G_IOTA_MODE
        and surface_dof_count > 0
        and contract.surface_scatter_indices.dtype == np.dtype(np.float32)
        and contract.surface_scatter_indices.ndim == 2
        and int(contract.surface_scatter_indices.shape[0]) == 3 * coeff_count
        and int(contract.surface_scatter_indices.shape[1]) == surface_dof_count
    )


def mps_boozer_jax_mps_backend_available() -> bool:
    """Return whether JAX currently exposes a local jax-mps device."""

    return any(device.platform == "mps" for device in jax.local_devices())


def require_mps_boozer_fixed_surface_g_iota_supported(
    contract: MpsBoozerKernelContract,
) -> None:
    """Raise if a contract cannot use the first fused Boozer MPS backend."""

    _validate_fused_custom_call_contract(contract)
    if mps_boozer_fixed_surface_g_iota_supported(contract):
        return
    raise ValueError(
        "The experimental MPS Boozer custom kernel only supports the "
        "fixed-surface G/iota fixture: float32 arrays, one grouped coil set, "
        "stellsym xyztensorfourier surfaces, zero BoozerResidual constraint "
        "weight, optimize_G=True, mps_solver_mode='fixed_surface_g_iota', "
        "1 <= newton_maxiter <= "
        f"{MAX_EXPERIMENTAL_MPS_BOOZER_FIXED_SURFACE_NEWTON_MAXITER}, and "
        f"gmres_maxiter={EXPERIMENTAL_MPS_BOOZER_FIXED_SURFACE_GMRES_MAXITER}."
    )


def _coil_group_arrays(coil_set_spec: object):
    gammas = []
    gammadashs = []
    currents = []
    indices = []
    for group in coil_set_spec.groups:
        gammas.append(group.gammas)
        gammadashs.append(group.gammadashs)
        currents.append(group.currents)
        indices.append(tuple(int(index) for index in group.coil_indices))
    return tuple(gammas), tuple(gammadashs), tuple(currents), tuple(indices)


def _split_flat_coil_cotangent(
    flat_cotangent: jax.Array,
    *,
    gammas_shape: tuple[int, ...],
    gammadashs_shape: tuple[int, ...],
    currents_shape: tuple[int, ...],
) -> tuple[jax.Array, jax.Array, jax.Array]:
    gammas_size = int(np.prod(gammas_shape, dtype=np.int64))
    gammadashs_size = int(np.prod(gammadashs_shape, dtype=np.int64))
    currents_size = int(np.prod(currents_shape, dtype=np.int64))
    gammas_stop = gammas_size
    gammadashs_stop = gammas_stop + gammadashs_size
    currents_stop = gammadashs_stop + currents_size
    return (
        flat_cotangent[:gammas_stop].reshape(gammas_shape),
        flat_cotangent[gammas_stop:gammadashs_stop].reshape(gammadashs_shape),
        flat_cotangent[gammadashs_stop:currents_stop].reshape(currents_shape),
    )


def _coil_pullback_operator(
    biotsavart: object,
    *,
    coil_dofs: jax.Array,
    gammas: jax.Array,
    gammadashs: jax.Array,
    currents: jax.Array,
    coil_indices: tuple[int, ...],
) -> jax.Array:
    """Dense cotangent projection from grouped coil arrays to flat coil DOFs."""

    flat_cotangent_size = int(gammas.size + gammadashs.size + currents.size)
    basis = jnp.eye(flat_cotangent_size, dtype=coil_dofs.dtype)

    def project(flat_cotangent: jax.Array) -> jax.Array:
        group_cotangent = _split_flat_coil_cotangent(
            flat_cotangent,
            gammas_shape=gammas.shape,
            gammadashs_shape=gammadashs.shape,
            currents_shape=currents.shape,
        )
        return biotsavart.coil_cotangents_to_dofs_gradient(
            (group_cotangent,),
            (coil_indices,),
            coil_dofs=coil_dofs,
        )

    return jnp.asarray(jax.vmap(project)(basis).T, dtype=jnp.float32)


def _require_current_coil_contract(
    boozer_residual: object,
    contract: MpsBoozerKernelContract,
) -> None:
    current_coil_dofs = jnp.asarray(boozer_residual.biotsavart.x)
    if current_coil_dofs.shape != contract.coil_dofs.shape or not np.array_equal(
        np.asarray(current_coil_dofs),
        np.asarray(contract.coil_dofs),
    ):
        raise ValueError(
            "Fused Boozer oracle requires contract.coil_dofs to match the "
            "BoozerSurfaceJAX solved-state coil DOFs. Re-solve before building "
            "a solved-state custom-kernel oracle for different coil DOFs."
        )


def _require_current_inner_contract(
    boozer_residual: object,
    contract: MpsBoozerKernelContract,
    solved_state: object,
) -> None:
    expected_x_inner, expected_optimize_G = boozer_residual._inner_objective_state(
        solved_state.iota,
        solved_state.G,
        sdofs=solved_state.sdofs,
    )
    if bool(expected_optimize_G) != bool(contract.static_metadata.optimize_G):
        raise ValueError(
            "Fused Boozer oracle contract optimize_G flag does not match the "
            "current solved state."
        )
    if expected_x_inner.shape != contract.x_inner.shape or not np.array_equal(
        np.asarray(expected_x_inner),
        np.asarray(contract.x_inner),
    ):
        raise ValueError(
            "Fused Boozer oracle contract x_inner does not match the current "
            "BoozerSurfaceJAX solved state."
        )


def _require_float32_array(name: str, value: jax.Array) -> None:
    if value.dtype != np.dtype(np.float32):
        raise ValueError(f"{name} must be float32 for the MPS custom call")


def _require_vector(name: str, value: jax.Array) -> None:
    if value.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    if value.shape[0] == 0:
        raise ValueError(f"{name} must not be empty")


def _require_scatter_operand(name: str, value: jax.Array) -> None:
    if value.dtype == np.dtype(np.int32):
        _require_vector(name, value)
        return
    if value.dtype == np.dtype(np.float32):
        if value.ndim != 2:
            raise ValueError(
                f"{name} must be a 1D int32 index vector or a 2D float32 "
                "scatter operator"
            )
        if value.shape[0] == 0 or value.shape[1] == 0:
            raise ValueError(f"{name} must not be empty")
        return
    raise ValueError(
        f"{name} must be a 1D int32 index vector or a 2D float32 scatter operator"
    )


def _validate_fused_custom_call_contract(contract: MpsBoozerKernelContract) -> None:
    if contract.static_metadata.target_name != SIMSOPT_MPS_BOOZER_VALUE_GRAD_TARGET:
        raise ValueError(
            f"contract target_name must be {SIMSOPT_MPS_BOOZER_VALUE_GRAD_TARGET!r}"
        )

    runtime_arrays = (
        ("coil_dofs", contract.coil_dofs),
        ("x_inner", contract.x_inner),
        ("surface_dofs", contract.surface_dofs),
        ("quadpoints_phi", contract.quadpoints_phi),
        ("quadpoints_theta", contract.quadpoints_theta),
        ("label_quadpoints_phi", contract.label_quadpoints_phi),
        ("label_quadpoints_theta", contract.label_quadpoints_theta),
    )
    for name, value in runtime_arrays:
        _require_float32_array(name, value)
        _require_vector(name, value)
    _require_scatter_operand(
        "surface_scatter_indices",
        contract.surface_scatter_indices,
    )
    _require_scatter_operand(
        "label_scatter_indices",
        contract.label_scatter_indices,
    )

    if (
        len(contract.coil_group_gammas) != 1
        or len(contract.coil_group_gammadashs) != 1
        or len(contract.coil_group_currents) != 1
    ):
        raise ValueError(
            "the first fused MPS custom call supports exactly one coil group"
        )

    gammas = contract.coil_group_gammas[0]
    gammadashs = contract.coil_group_gammadashs[0]
    currents = contract.coil_group_currents[0]
    for name, value in (
        ("coil_group_gammas[0]", gammas),
        ("coil_group_gammadashs[0]", gammadashs),
        ("coil_group_currents[0]", currents),
    ):
        _require_float32_array(name, value)
    if gammas.ndim != 3 or gammas.shape[2] != 3:
        raise ValueError("coil_group_gammas[0] must have shape (ncoils, nquad, 3)")
    if gammas.shape[0] == 0 or gammas.shape[1] == 0:
        raise ValueError(
            "coil_group_gammas[0] must include coils and quadrature points"
        )
    if gammadashs.shape != gammas.shape:
        raise ValueError("coil_group_gammadashs[0] must match coil_group_gammas[0]")
    if currents.ndim != 1 or currents.shape[0] != gammas.shape[0]:
        raise ValueError("coil_group_currents[0] must have shape (ncoils,)")
    _require_float32_array("coil_pullback_operator", contract.coil_pullback_operator)
    if contract.coil_pullback_operator.ndim != 2:
        raise ValueError("coil_pullback_operator must be a 2D array")
    flat_cotangent_size = int(gammas.size + gammadashs.size + currents.size)
    if contract.coil_pullback_operator.shape != (
        contract.coil_dofs.shape[0],
        flat_cotangent_size,
    ):
        raise ValueError(
            "coil_pullback_operator must have shape "
            "(coil_dofs.size, gammas.size + gammadashs.size + currents.size)"
        )


def _fused_custom_call_backend_config(contract: MpsBoozerKernelContract) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "target_name": SIMSOPT_MPS_BOOZER_VALUE_GRAD_TARGET,
        "static_metadata": _json_scalar(asdict(contract.static_metadata)),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _result_vector_norm(
    solve_result: Mapping[str, object],
    keys: tuple[str, ...],
    *,
    dtype: object,
) -> jax.Array:
    for key in keys:
        value = solve_result.get(key)
        if value is not None:
            return jnp.linalg.norm(jnp.asarray(value, dtype=dtype))
    return jnp.asarray(jnp.nan, dtype=dtype)


def _result_scalar(
    solve_result: Mapping[str, object],
    keys: tuple[str, ...],
    *,
    dtype: object,
) -> jax.Array | None:
    for key in keys:
        value = solve_result.get(key)
        if value is not None:
            return jnp.asarray(value, dtype=dtype)
    return None


def _iteration_count_or_unknown(
    solve_result: Mapping[str, object],
    keys: tuple[str, ...],
) -> jax.Array:
    value = _result_scalar(solve_result, keys, dtype=jnp.int32)
    if value is None:
        return jnp.asarray(UNKNOWN_ITERATION_COUNT, dtype=jnp.int32)
    return value


def build_mps_boozer_direct_kernel_contract(
    boozer_residual: object,
    *,
    solved_state: object,
    coil_dofs: object | None = None,
) -> MpsBoozerKernelContract:
    """Build the flattened direct-objective contract from solved Boozer state."""

    booz_surf = boozer_residual.boozer_surface
    biotsavart = boozer_residual.biotsavart
    current_coil_dofs = biotsavart.x if coil_dofs is None else coil_dofs
    current_coil_dofs = jax.numpy.asarray(current_coil_dofs)
    x_inner, optimize_G = boozer_residual._inner_objective_state(
        solved_state.iota,
        solved_state.G,
        sdofs=solved_state.sdofs,
    )
    coil_set_spec = biotsavart.coil_set_spec_from_dofs(current_coil_dofs)
    gammas, gammadashs, currents, coil_group_indices = _coil_group_arrays(coil_set_spec)
    if len(gammas) != 1:
        raise ValueError(
            "the first fused MPS custom call supports exactly one coil group"
        )
    coil_pullback_operator = _coil_pullback_operator(
        biotsavart,
        coil_dofs=current_coil_dofs,
        gammas=gammas[0],
        gammadashs=gammadashs[0],
        currents=currents[0],
        coil_indices=coil_group_indices[0],
    )
    metadata = MpsBoozerKernelStaticMetadata(
        target_name="mps.simsopt_boozer_value_grad",
        mpol=int(booz_surf.mpol),
        ntor=int(booz_surf.ntor),
        nfp=int(booz_surf.nfp),
        stellsym=bool(booz_surf.stellsym),
        surface_kind=str(booz_surf._surface_geometry_kind),
        label_mpol=int(booz_surf.label_mpol),
        label_ntor=int(booz_surf.label_ntor),
        label_nfp=int(booz_surf.label_nfp),
        label_stellsym=bool(booz_surf.label_stellsym),
        label_surface_kind=str(booz_surf._label_surface_geometry_kind),
        label_type=str(booz_surf.label_type),
        phi_idx=None if booz_surf.phi_idx is None else int(booz_surf.phi_idx),
        targetlabel=float(booz_surf.targetlabel),
        constraint_weight=float(boozer_residual.constraint_weight),
        optimize_G=bool(optimize_G),
        weight_inv_modB=bool(solved_state.weight_inv_modB),
        solver_options=_solver_option_items(booz_surf.options),
        coil_group_indices=coil_group_indices,
    )
    return MpsBoozerKernelContract(
        coil_dofs=current_coil_dofs,
        x_inner=x_inner,
        surface_dofs=solved_state.sdofs,
        quadpoints_phi=booz_surf.quadpoints_phi,
        quadpoints_theta=booz_surf.quadpoints_theta,
        label_quadpoints_phi=booz_surf.label_quadpoints_phi,
        label_quadpoints_theta=booz_surf.label_quadpoints_theta,
        surface_scatter_indices=booz_surf.scatter_indices,
        label_scatter_indices=booz_surf.label_scatter_indices,
        coil_group_gammas=gammas,
        coil_group_gammadashs=gammadashs,
        coil_group_currents=currents,
        coil_pullback_operator=coil_pullback_operator,
        static_metadata=metadata,
    )


def evaluate_mps_boozer_direct_cpu_oracle(
    boozer_residual: object,
    contract: MpsBoozerKernelContract,
) -> MpsBoozerDirectOracleResult:
    """Evaluate the direct CPU/JAX oracle through the existing value/grad path."""

    value, gradient = boozer_residual._direct_objective_value_and_grad(
        contract.coil_dofs,
        contract.x_inner,
        contract.static_metadata.optimize_G,
        contract.static_metadata.weight_inv_modB,
    )
    return MpsBoozerDirectOracleResult(value=value, gradient=gradient)


def evaluate_mps_boozer_fused_solve_cpu_oracle(
    boozer_residual: object,
    contract: MpsBoozerKernelContract,
) -> MpsBoozerFusedSolveOracleResult:
    """Evaluate the full solved-state CPU/JAX oracle for the fused op contract."""

    _require_current_coil_contract(boozer_residual, contract)
    booz_surf = boozer_residual.boozer_surface
    solved_state = booz_surf.get_solved_runtime_state()
    _require_current_inner_contract(boozer_residual, contract, solved_state)
    coil_set_spec = boozer_residual.biotsavart.coil_set_spec_from_dofs(
        contract.coil_dofs,
    )
    value, coil_gradient = boozer_residual._value_and_dJ_by_dcoil_dofs(
        solved_state,
        contract.coil_dofs,
        coil_set_spec,
    )

    solve_result = booz_surf.res
    if not isinstance(solve_result, Mapping):
        raise RuntimeError("BoozerSurfaceJAX solved result is unavailable.")
    output_dtype = jnp.asarray(value).dtype
    residual_norm = _result_vector_norm(solve_result, ("residual",), dtype=output_dtype)
    gradient_norm = _result_scalar(
        solve_result,
        ("final_gradient_norm",),
        dtype=output_dtype,
    )
    if gradient_norm is None:
        gradient_norm = _result_vector_norm(
            solve_result,
            ("gradient", "jacobian", "grad"),
            dtype=output_dtype,
        )
    finite = (
        jnp.all(jnp.isfinite(value))
        & jnp.all(jnp.isfinite(coil_gradient))
        & jnp.all(jnp.isfinite(contract.x_inner))
        & jnp.all(jnp.isfinite(residual_norm))
        & jnp.all(jnp.isfinite(gradient_norm))
    )
    return MpsBoozerFusedSolveOracleResult(
        value=value,
        coil_gradient=coil_gradient,
        final_x_inner=contract.x_inner,
        residual_norm=residual_norm,
        gradient_norm=gradient_norm,
        newton_iteration_count=_iteration_count_or_unknown(
            solve_result,
            ("newton_iter", "iter", "nit"),
        ),
        gmres_iteration_count=_iteration_count_or_unknown(
            solve_result,
            ("gmres_iteration_count", "linear_iteration_count"),
        ),
        converged=jnp.asarray(
            bool(
                solve_result.get("primal_success", solve_result.get("success", False))
            ),
        ),
        finite=jnp.asarray(finite),
    )


def evaluate_mps_boozer_fused_solve_custom_call(
    contract: MpsBoozerKernelContract,
) -> MpsBoozerFusedCustomCallResult:
    """Emit the Stage 3 fused Boozer value-gradient custom-call boundary."""

    _validate_fused_custom_call_contract(contract)
    dtype = contract.coil_dofs.dtype
    output_type = (
        jax.ShapeDtypeStruct((), dtype),
        jax.ShapeDtypeStruct(contract.coil_dofs.shape, dtype),
        jax.ShapeDtypeStruct(contract.x_inner.shape, dtype),
        jax.ShapeDtypeStruct((), dtype),
        jax.ShapeDtypeStruct((), dtype),
        jax.ShapeDtypeStruct((), jnp.int32),
        jax.ShapeDtypeStruct((), jnp.int32),
        jax.ShapeDtypeStruct((), jnp.bool_),
        jax.ShapeDtypeStruct((), jnp.bool_),
    )
    gammas = contract.coil_group_gammas[0]
    gammadashs = contract.coil_group_gammadashs[0]
    currents = contract.coil_group_currents[0]
    return MpsBoozerFusedCustomCallResult(
        *jax.ffi.ffi_call(
            SIMSOPT_MPS_BOOZER_VALUE_GRAD_TARGET,
            output_type,
            vmap_method="broadcast_all",
            custom_call_api_version=(
                SIMSOPT_MPS_BOOZER_VALUE_GRAD_CUSTOM_CALL_API_VERSION
            ),
            legacy_backend_config=_fused_custom_call_backend_config(contract),
        )(
            contract.coil_dofs,
            contract.x_inner,
            contract.surface_dofs,
            contract.quadpoints_phi,
            contract.quadpoints_theta,
            contract.label_quadpoints_phi,
            contract.label_quadpoints_theta,
            contract.surface_scatter_indices,
            contract.label_scatter_indices,
            gammas,
            gammadashs,
            currents,
            contract.coil_pullback_operator,
        )
    )


def build_mps_boozer_fused_solve_value_and_grad(
    boozer_residual: object,
    *,
    solved_state: object | None = None,
):
    """Build the opt-in fixed-surface MPS Boozer value/grad callable.

    The returned callable has the existing ``_simsopt_value_and_grad`` marker,
    so callers can reuse the standard value/grad boundary without another
    dispatch concept. Unsupported platforms or fixture shapes fail before the
    callable is returned.
    """

    if not mps_boozer_jax_mps_backend_available():
        raise RuntimeError(
            "The experimental MPS Boozer custom kernel requires an active "
            "jax-mps backend; no local JAX device reports platform='mps'."
        )
    effective_solved_state = (
        boozer_residual.boozer_surface.get_solved_runtime_state()
        if solved_state is None
        else solved_state
    )
    if effective_solved_state is None:
        raise RuntimeError(
            "The experimental MPS Boozer custom kernel requires a solved "
            "BoozerSurfaceJAX runtime state."
        )
    probe_contract = build_mps_boozer_direct_kernel_contract(
        boozer_residual,
        solved_state=effective_solved_state,
    )
    require_mps_boozer_fixed_surface_g_iota_supported(probe_contract)

    def value_and_grad(coil_dofs):
        contract = build_mps_boozer_direct_kernel_contract(
            boozer_residual,
            solved_state=effective_solved_state,
            coil_dofs=coil_dofs,
        )
        result = evaluate_mps_boozer_fused_solve_custom_call(contract)
        status_ok = jnp.logical_and(result.converged, result.finite)
        failed_value = jnp.zeros_like(result.value) / jnp.zeros_like(result.value)
        failed_gradient = jnp.zeros_like(result.coil_gradient) / jnp.zeros_like(
            result.coil_gradient,
        )
        return (
            jnp.where(status_ok, result.value, failed_value),
            jnp.where(status_ok, result.coil_gradient, failed_gradient),
        )

    value_and_grad._simsopt_value_and_grad = True
    value_and_grad._simsopt_mps_boozer_custom_kernel = True
    return value_and_grad


def mps_boozer_kernel_contract_artifact(
    contract: MpsBoozerKernelContract,
    oracle_result: MpsBoozerDirectOracleResult | None = None,
    fused_oracle_result: MpsBoozerFusedSolveOracleResult | None = None,
) -> dict[str, object]:
    """Return a JSON-serializable shape/dtype contract artifact."""

    artifact = {
        "schema_version": SCHEMA_VERSION,
        "target_name": contract.static_metadata.target_name,
        "output_contract": {
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
        },
        "runtime_arrays": {
            "coil_dofs": _array_schema(contract.coil_dofs),
            "x_inner": _array_schema(contract.x_inner),
            "surface_dofs": _array_schema(contract.surface_dofs),
            "quadpoints_phi": _array_schema(contract.quadpoints_phi),
            "quadpoints_theta": _array_schema(contract.quadpoints_theta),
            "label_quadpoints_phi": _array_schema(contract.label_quadpoints_phi),
            "label_quadpoints_theta": _array_schema(contract.label_quadpoints_theta),
            "surface_scatter_indices": _array_schema(
                contract.surface_scatter_indices,
            ),
            "label_scatter_indices": _array_schema(
                contract.label_scatter_indices,
            ),
            "coil_pullback_operator": _array_schema(
                contract.coil_pullback_operator,
            ),
            "coil_groups": [
                {
                    "group_index": group_index,
                    "coil_indices": list(coil_indices),
                    "gammas": _array_schema(gammas),
                    "gammadashs": _array_schema(gammadashs),
                    "currents": _array_schema(currents),
                }
                for group_index, (
                    coil_indices,
                    gammas,
                    gammadashs,
                    currents,
                ) in enumerate(
                    zip(
                        contract.static_metadata.coil_group_indices,
                        contract.coil_group_gammas,
                        contract.coil_group_gammadashs,
                        contract.coil_group_currents,
                        strict=True,
                    )
                )
            ],
        },
        "static_metadata": _json_scalar(asdict(contract.static_metadata)),
    }
    if oracle_result is not None:
        artifact["oracle_outputs"] = {
            "value": _array_schema(oracle_result.value),
            "gradient": _array_schema(oracle_result.gradient),
        }
    if fused_oracle_result is not None:
        artifact["fused_oracle_outputs"] = {
            "value": _array_schema(fused_oracle_result.value),
            "coil_gradient": _array_schema(fused_oracle_result.coil_gradient),
            "final_x_inner": _array_schema(fused_oracle_result.final_x_inner),
            "residual_norm": _array_schema(fused_oracle_result.residual_norm),
            "gradient_norm": _array_schema(fused_oracle_result.gradient_norm),
            "newton_iteration_count": _array_schema(
                fused_oracle_result.newton_iteration_count,
            ),
            "gmres_iteration_count": _array_schema(
                fused_oracle_result.gmres_iteration_count,
            ),
            "converged": _array_schema(fused_oracle_result.converged),
            "finite": _array_schema(fused_oracle_result.finite),
        }
    return artifact


def write_mps_boozer_kernel_contract_artifact(
    contract: MpsBoozerKernelContract,
    path: str | Path,
    *,
    oracle_result: MpsBoozerDirectOracleResult | None = None,
    fused_oracle_result: MpsBoozerFusedSolveOracleResult | None = None,
) -> None:
    """Write the shape/dtype custom-kernel contract artifact."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            mps_boozer_kernel_contract_artifact(
                contract,
                oracle_result=oracle_result,
                fused_oracle_result=fused_oracle_result,
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
