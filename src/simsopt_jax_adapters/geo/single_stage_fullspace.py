"""Host bootstrap for the GPU-native NCSX single-stage full-space problem."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
from simsopt.configs import get_data
from simsopt.geo import CurveLength, SurfaceXYZTensorFourier, Volume
from simsopt.geo.curve import Curve
from simsopt_jax.backend.dtypes import explicit_device_array
from simsopt_jax.core.specs import make_surface_xyz_tensor_fourier_spec
from simsopt_jax.geo._surface_stellsym import stellsym_mask_indices_for_grid_host
from simsopt_jax.objectives.dynamic_surface_stage_two import (
    freeze_coil_dof_extraction_spec,
)
from simsopt_jax.objectives.single_stage_fullspace import (
    FROZEN_LAYOUT,
    FullSpaceObjectiveConfig,
    FullSpaceProblem,
    FullSpaceState,
)
from simsopt_jax.runtime.host_boundary import host_array, host_bool, host_float

from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt_jax_adapters.geo.boozer_surface import BoozerSurfaceJAX


@dataclass(frozen=True, slots=True)
class Float64Fingerprint:
    name: str
    value: float
    hexadecimal: str
    little_endian_sha256: str


@dataclass(frozen=True, slots=True)
class SingleStageFullSpaceBootstrap:
    """One exact feasible initial state and its immutable pure-JAX problem."""

    problem: FullSpaceProblem
    z0: jax.Array
    targets: tuple[Float64Fingerprint, ...]
    initial_boozer_residual_norm: float
    first_base_current: Float64Fingerprint


def _float64_fingerprint(name: str, value: float) -> Float64Fingerprint:
    scalar = np.asarray(value, dtype="<f8")
    return Float64Fingerprint(
        name=name,
        value=float(scalar),
        hexadecimal=float(scalar).hex(),
        little_endian_sha256=hashlib.sha256(scalar.tobytes()).hexdigest(),
    )


def _surface_spec(
    surface_dofs: np.ndarray,
    quadpoints_phi: np.ndarray,
    quadpoints_theta: np.ndarray,
):
    return make_surface_xyz_tensor_fourier_spec(
        dofs=surface_dofs,
        quadpoints_phi=quadpoints_phi,
        quadpoints_theta=quadpoints_theta,
        nfp=3,
        stellsym=True,
        mpol=6,
        ntor=6,
        clamped_dims=(False, False, False),
    )


def build_single_stage_fullspace_bootstrap() -> SingleStageFullSpaceBootstrap:
    """Construct the authoritative native-scale NCSX bootstrap exactly once."""

    base_curves, base_currents, magnetic_axis, nfp, native_field = get_data("ncsx")
    if nfp != 3:
        raise ValueError("NCSX bootstrap must have nfp=3")
    magnetic_axis = cast(Curve, magnetic_axis)
    base_currents[0].fix_all()
    field = BiotSavartJAX(native_field.coils)
    current_sum = nfp * sum(abs(current.get_value()) for current in base_currents)
    G0 = 2.0 * np.pi * current_sum * (4.0 * np.pi * 1.0e-7 / (2.0 * np.pi))

    exact_phi = np.linspace(0.0, 1.0 / nfp, 13, endpoint=False)
    exact_theta = np.linspace(0.0, 1.0, 13, endpoint=False)
    surface = SurfaceXYZTensorFourier(
        mpol=6,
        ntor=6,
        stellsym=True,
        nfp=nfp,
        quadpoints_phi=exact_phi,
        quadpoints_theta=exact_theta,
    )
    surface.fit_to_curve(magnetic_axis, 0.1, flip_theta=True)
    volume_label = Volume(surface)
    volume_target = float(volume_label.J())
    boozer_surface = BoozerSurfaceJAX(
        field,
        surface,
        volume_label,
        volume_target,
        options={
            "newton_maxiter": 20,
            "newton_tol": 1.0e-13,
            "verbose": False,
        },
    )
    solve_result = cast(
        Mapping[str, object],
        boozer_surface.run_code_traceable(
            field.coil_set_spec(),
            explicit_device_array(surface.get_dofs(), dtype=jnp.float64),
            explicit_device_array(-0.406, dtype=jnp.float64),
            explicit_device_array(G0, dtype=jnp.float64),
        ),
    )
    if not host_bool(solve_result["success"]):
        raise RuntimeError("authoritative exact Boozer bootstrap failed")
    boozer_surface.install_traceable_solved_runtime_state(solve_result)

    coil_dofs = np.asarray(field.x, dtype=np.float64)
    surface_dofs = host_array(solve_result["sdofs"], dtype=np.float64)
    iota_target = host_float(solve_result["iota"])
    solved_G = host_float(solve_result["G"])
    initial_residual_norm = float(
        np.linalg.norm(host_array(solve_result["residual"], dtype=np.float64))
    )
    major_radius_target = float(surface.major_radius())
    length_target = float(sum(CurveLength(curve).J() for curve in base_curves))
    if coil_dofs.shape != (461,) or surface_dofs.shape != (253,):
        raise ValueError("NCSX bootstrap DOF counts do not match the frozen layout")

    non_qs_phi = np.linspace(0.0, 1.0 / nfp, 40, endpoint=False)
    non_qs_theta = np.linspace(0.0, 1.0, 40, endpoint=False)
    mask_indices = stellsym_mask_indices_for_grid_host(
        mpol=6,
        ntor=6,
        nfp=nfp,
        stellsym=True,
        quadpoints_phi=exact_phi,
        quadpoints_theta=exact_theta,
    )
    if mask_indices.shape != (254,):
        raise ValueError("NCSX exact Boozer mask must have 254 components")

    config = FullSpaceObjectiveConfig(
        iota_target=explicit_device_array(iota_target, dtype=jnp.float64),
        major_radius_target=explicit_device_array(
            major_radius_target, dtype=jnp.float64
        ),
        length_target=explicit_device_array(length_target, dtype=jnp.float64),
        volume_target=explicit_device_array(volume_target, dtype=jnp.float64),
        non_qs_weight=explicit_device_array(1.0, dtype=jnp.float64),
        residual_weight=explicit_device_array(1.0, dtype=jnp.float64),
        iota_weight=explicit_device_array(1.0, dtype=jnp.float64),
        major_radius_weight=explicit_device_array(1.0, dtype=jnp.float64),
        length_weight=explicit_device_array(1.0, dtype=jnp.float64),
        non_qs_axis=0,
        weight_inv_modB=False,
        length_coil_indices=(0, 1, 2),
    )
    problem = FullSpaceProblem(
        coil_dof_extraction=freeze_coil_dof_extraction_spec(field),
        exact_surface_template=_surface_spec(surface_dofs, exact_phi, exact_theta),
        label_surface_template=_surface_spec(surface_dofs, exact_phi, exact_theta),
        non_qs_surface_template=_surface_spec(
            surface_dofs,
            non_qs_phi,
            non_qs_theta,
        ),
        exact_mask_indices=explicit_device_array(mask_indices, dtype=jnp.int32),
        config=config,
        layout=FROZEN_LAYOUT,
    )
    z0 = FROZEN_LAYOUT.pack(
        FullSpaceState(
            coil_dofs=explicit_device_array(coil_dofs, dtype=jnp.float64),
            surface_dofs=explicit_device_array(surface_dofs, dtype=jnp.float64),
            iota=explicit_device_array(iota_target, dtype=jnp.float64),
            G=explicit_device_array(solved_G, dtype=jnp.float64),
        )
    )
    targets = (
        _float64_fingerprint("volume_target", volume_target),
        _float64_fingerprint("iota_target", iota_target),
        _float64_fingerprint("major_radius_target", major_radius_target),
        _float64_fingerprint("length_target", length_target),
    )
    return SingleStageFullSpaceBootstrap(
        problem=problem,
        z0=z0,
        targets=targets,
        initial_boozer_residual_norm=initial_residual_norm,
        first_base_current=_float64_fingerprint(
            "first_base_current",
            float(base_currents[0].get_value()),
        ),
    )


def bootstrap_target_payload(
    bootstrap: SingleStageFullSpaceBootstrap,
) -> dict[str, object]:
    """Return exact float64 target identities for canonical receipt encoding."""

    return {
        "schema_version": "single-stage-fullspace-bootstrap-targets-v1",
        "targets": [
            {
                "name": target.name,
                "value": target.value,
                "hexadecimal": target.hexadecimal,
                "little_endian_sha256": target.little_endian_sha256,
            }
            for target in bootstrap.targets
        ],
        "first_base_current": {
            "name": bootstrap.first_base_current.name,
            "value": bootstrap.first_base_current.value,
            "hexadecimal": bootstrap.first_base_current.hexadecimal,
            "little_endian_sha256": bootstrap.first_base_current.little_endian_sha256,
        },
        "initial_boozer_residual_norm": bootstrap.initial_boozer_residual_norm,
        "joint_dof_count": bootstrap.z0.shape[0],
        "equality_count": bootstrap.problem.exact_mask_indices.shape[0] + 1,
    }
