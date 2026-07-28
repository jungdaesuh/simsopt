"""Shared exact-lane execution for fixed-grid arbitrary-vector GPMO mirrors."""

from __future__ import annotations

import hashlib
import os

import numpy as np
from examples.jax.parity.arbiter import LaneObservation
from examples.jax.parity.input_bundle import (
    InputBundle,
    effective_construction_fingerprint,
)
from examples.jax.parity.runtime import ParityLane


def configuration_int(bundle: InputBundle, name: str) -> int:
    """Read one integer from a parity input configuration."""
    value = bundle.configuration[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"configuration {name} must be an integer")
    return value


def configuration_float(bundle: InputBundle, name: str) -> float:
    """Read one floating-point value from a parity input configuration."""
    value = bundle.configuration[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"configuration {name} must be numeric")
    return float(value)


def frozen_grid_arrays(grid) -> dict[str, np.ndarray]:
    """Copy the complete fixed optimization state from one native PM grid."""
    moments = np.array(grid.m, dtype=np.float64, copy=True).reshape((grid.ndipoles, 3))
    proxy_source = grid.m_proxy if hasattr(grid, "m_proxy") else grid.m
    return {
        "response_matrix": np.array(grid.A_obj, dtype=np.float64, copy=True),
        "target": np.array(grid.b_obj, dtype=np.float64, copy=True),
        "normal_norms": np.array(
            np.linalg.norm(grid.plasma_boundary.normal(), axis=-1).reshape(-1),
            dtype=np.float64,
            copy=True,
        ),
        "atb": np.array(grid.ATb, dtype=np.float64, copy=True).reshape(
            (grid.ndipoles, 3)
        ),
        "ata_scale": np.array(grid.ATA_scale, dtype=np.float64, copy=True),
        "initial_moments": np.array(
            grid.m0,
            dtype=np.float64,
            copy=True,
        ).reshape((grid.ndipoles, 3)),
        "moments": moments,
        "proxy_moments": np.array(
            proxy_source,
            dtype=np.float64,
            copy=True,
        ).reshape((grid.ndipoles, 3)),
        "moment_maxima": np.array(
            grid.m_maxima,
            dtype=np.float64,
            copy=True,
        ).reshape((grid.ndipoles,)),
        "dipole_grid_xyz": np.array(
            grid.dipole_grid_xyz,
            dtype=np.float64,
            copy=True,
        ),
        "polarization_vectors": np.array(
            grid.pol_vectors,
            dtype=np.float64,
            copy=True,
        ),
    }


def _array_digest(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def effective_grid_fingerprint(
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
) -> str:
    """Fingerprint the complete frozen grid without derived recomputation."""
    return effective_construction_fingerprint(
        bundle,
        {
            "arrays": {
                name: {
                    "dtype": str(value.dtype),
                    "shape": list(value.shape),
                    "sha256": _array_digest(value),
                }
                for name, value in sorted(arrays.items())
            },
            **bundle.configuration,
        },
    )


def _values(
    arrays: dict[str, np.ndarray],
    final_moments: np.ndarray,
    final_residual: np.ndarray,
) -> dict[str, np.ndarray]:
    response = arrays["response_matrix"]
    target = arrays["target"]
    initial_moments = arrays["initial_moments"]
    initial_residual = response @ initial_moments.reshape(-1) - target
    nonzero_mask = np.linalg.norm(final_moments, axis=1) != 0.0
    return {
        "construction:response_matrix": response,
        "construction:target": target,
        "construction:moment_maxima": arrays["moment_maxima"],
        "construction:dipole_grid_xyz": arrays["dipole_grid_xyz"],
        "construction:polarization_vectors": arrays["polarization_vectors"],
        "initial:moments": initial_moments,
        "initial:residual": initial_residual,
        "initial:objective_sum_squares": np.asarray(
            np.vdot(initial_residual, initial_residual),
            dtype=np.float64,
        ),
        "final:moments": final_moments,
        "final:residual": final_residual,
        "final:objective_sum_squares": np.asarray(
            np.vdot(final_residual, final_residual),
            dtype=np.float64,
        ),
        "final:nonzero_mask": nonzero_mask,
    }


def _observation(
    lane: ParityLane,
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
    values: dict[str, np.ndarray],
    workflow_stages: tuple[str, ...],
    *,
    driver: str,
    platform: str,
    precision: str,
) -> LaneObservation:
    success = bool(
        np.count_nonzero(values["final:nonzero_mask"])
        == configuration_int(bundle, "max_magnets")
        and np.all(np.isfinite(values["final:moments"]))
        and values["final:objective_sum_squares"]
        < values["initial:objective_sum_squares"]
    )
    return LaneObservation(
        lane=lane,
        backend_mode=(
            "native_cpu" if lane == "native-cpu" else os.environ["SIMSOPT_BACKEND_MODE"]
        ),
        platform=platform,
        precision=precision,
        scale=bundle.scale,
        input_fingerprint=bundle.input_fingerprint,
        configuration_fingerprint=bundle.configuration_fingerprint,
        effective_construction_fingerprint=effective_grid_fingerprint(
            bundle,
            arrays,
        ),
        driver=driver,
        normalized_status="converged" if success else "failed",
        raw_status="fixed_iteration_budget_complete",
        success=success,
        nit=configuration_int(bundle, "iterations"),
        nfev=configuration_int(bundle, "iterations"),
        njev=None,
        completed_workflow_stages=workflow_stages,
        provenance=None,
        values=values,
    )


def _execute_native(
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
    workflow_stages: tuple[str, ...],
) -> LaneObservation:
    import simsoptpp

    maxima = arrays["moment_maxima"]
    maxima_vector = np.repeat(maxima, 3)
    scaled_response = arrays["response_matrix"] * maxima_vector[None, :]
    _, _, _, _, normalized_moments = simsoptpp.GPMO_ArbVec_backtracking(
        np.ascontiguousarray(scaled_response.T),
        np.ascontiguousarray(arrays["target"]),
        np.sqrt(configuration_float(bundle, "regularization_l2")) * maxima_vector,
        np.ascontiguousarray(arrays["normal_norms"]),
        np.ascontiguousarray(arrays["polarization_vectors"]),
        K=configuration_int(bundle, "iterations"),
        verbose=False,
        nhistory=configuration_int(bundle, "history_count"),
        backtracking=configuration_int(bundle, "backtracking"),
        dipole_grid_xyz=np.ascontiguousarray(arrays["dipole_grid_xyz"]),
        Nadjacent=configuration_int(bundle, "adjacent_count"),
        thresh_angle=configuration_float(bundle, "threshold_angle"),
        max_nMagnets=configuration_int(bundle, "max_magnets"),
        x_init=np.zeros(
            (configuration_int(bundle, "ndipoles"), 3),
            dtype=np.float64,
        ),
    )
    final_moments = np.asarray(normalized_moments, dtype=np.float64) * maxima[:, None]
    final_residual = (
        arrays["response_matrix"] @ final_moments.reshape(-1) - arrays["target"]
    )
    return _observation(
        "native-cpu",
        bundle,
        arrays,
        _values(arrays, final_moments, final_residual),
        workflow_stages,
        driver="simsoptpp_gpmo_arbvec_backtracking",
        platform="cpu",
        precision="fp64",
    )


def _execute_jax(
    lane: ParityLane,
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
    workflow_stages: tuple[str, ...],
) -> LaneObservation:
    from simsopt_jax.backend.runtime import get_runtime_jax_device
    from simsopt_jax.geo.permanent_magnet_grid import PermanentMagnetGridJAX
    from simsopt_jax.solve.permanent_magnet import GPMO_ArbVec_backtracking_jax

    import jax

    device = get_runtime_jax_device()

    def put(name: str):
        return jax.device_put(arrays[name], device)

    grid = PermanentMagnetGridJAX(
        A_obj=put("response_matrix"),
        b_obj=put("target"),
        ATb=put("atb"),
        ATA_scale=put("ata_scale"),
        m0=put("initial_moments"),
        m=put("moments"),
        m_proxy=put("proxy_moments"),
        m_maxima=put("moment_maxima"),
        dipole_grid_xyz=put("dipole_grid_xyz"),
        coordinate_flag=str(bundle.configuration["coordinate_flag"]),
        R0=configuration_float(bundle, "R0"),
        nfp=configuration_int(bundle, "nfp"),
        stellsym=bool(bundle.configuration["stellsym"]),
        nphi=configuration_int(bundle, "nphi"),
        ntheta=configuration_int(bundle, "ntheta"),
        ndipoles=configuration_int(bundle, "ndipoles"),
        pol_vectors=put("polarization_vectors"),
    )
    device_result = GPMO_ArbVec_backtracking_jax(
        grid,
        K=configuration_int(bundle, "iterations"),
        reg_l2=configuration_float(bundle, "regularization_l2"),
        Nadjacent=configuration_int(bundle, "adjacent_count"),
        backtracking=configuration_int(bundle, "backtracking"),
        thresh_angle=configuration_float(bundle, "threshold_angle"),
        max_nMagnets=configuration_int(bundle, "max_magnets"),
        record_every=configuration_int(bundle, "iterations"),
    )
    host_result, construction_arrays = jax.device_get(
        (
            device_result,
            {
                "response_matrix": grid.A_obj,
                "target": grid.b_obj,
                "atb": grid.ATb,
                "ata_scale": grid.ATA_scale,
                "initial_moments": grid.m0,
                "moments": grid.m,
                "proxy_moments": grid.m_proxy,
                "moment_maxima": grid.m_maxima,
                "dipole_grid_xyz": grid.dipole_grid_xyz,
                "polarization_vectors": grid.pol_vectors,
            },
        )
    )
    host_arrays = {
        **arrays,
        **{
            name: np.asarray(value, dtype=arrays[name].dtype)
            for name, value in construction_arrays.items()
        },
    }
    values = _values(
        host_arrays,
        np.asarray(host_result.m, dtype=np.float64),
        np.asarray(host_result.residual, dtype=np.float64),
    )
    platform = "cpu" if device is None else device.platform
    return _observation(
        lane,
        bundle,
        host_arrays,
        values,
        workflow_stages,
        driver="simsopt_jax_gpmo_arbvec_backtracking",
        platform="gpu" if platform in {"cuda", "gpu"} else platform,
        precision="fp64" if bool(jax.config.read("jax_enable_x64")) else "fp32",
    )


def execute_arbvec_case(
    lane: ParityLane,
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
    workflow_stages: tuple[str, ...],
) -> LaneObservation:
    """Run one frozen arbitrary-vector GPMO case in its requested lane."""
    if lane == "native-cpu":
        return _execute_native(bundle, arrays, workflow_stages)
    return _execute_jax(lane, bundle, arrays, workflow_stages)
