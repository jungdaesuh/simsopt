"""Flattened SIMSOPT Boozer custom-kernel contract helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import jax
import numpy as np

SCHEMA_VERSION = 1
DEFAULT_CONTRACT_ARTIFACT_DIR = Path(".artifacts/mps_custom_kernel_contract")


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


def mps_boozer_kernel_contract_artifact(
    contract: MpsBoozerKernelContract,
    oracle_result: MpsBoozerDirectOracleResult | None = None,
) -> dict[str, object]:
    """Return a JSON-serializable shape/dtype contract artifact."""

    artifact = {
        "schema_version": SCHEMA_VERSION,
        "target_name": contract.static_metadata.target_name,
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
    return artifact


def write_mps_boozer_kernel_contract_artifact(
    contract: MpsBoozerKernelContract,
    path: str | Path,
    *,
    oracle_result: MpsBoozerDirectOracleResult | None = None,
) -> None:
    """Write the shape/dtype custom-kernel contract artifact."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            mps_boozer_kernel_contract_artifact(
                contract,
                oracle_result=oracle_result,
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
