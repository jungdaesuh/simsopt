"""Flattened SIMSOPT Boozer custom-kernel contract helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

SCHEMA_VERSION = 2
DEFAULT_CONTRACT_ARTIFACT_DIR = Path(".artifacts/mps_custom_kernel_contract")
UNKNOWN_ITERATION_COUNT = -1


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
    coil_group_gammas: tuple[jax.Array, ...]
    coil_group_gammadashs: tuple[jax.Array, ...]
    coil_group_currents: tuple[jax.Array, ...]
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
        coil_group_gammas=gammas,
        coil_group_gammadashs=gammadashs,
        coil_group_currents=currents,
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
