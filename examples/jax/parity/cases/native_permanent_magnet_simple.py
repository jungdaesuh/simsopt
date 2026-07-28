"""Exact matched workflow for ``1_Simple/permanent_magnet_simple.py``."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
from examples.jax.parity.arbiter import LaneObservation
from examples.jax.parity.input_bundle import (
    InputBundle,
    create_input_bundle,
    effective_construction_fingerprint,
)
from examples.jax.parity.runtime import ParityLane
from simsopt_jax.examples import ExecutionScale

TEST_DATA = Path(__file__).resolve().parents[4] / "tests" / "test_files"
SURFACE_INPUT = TEST_DATA / "wout_c09r00_fixedBoundary_0.5T_vacuum_ns201.nc"
FAMUS_INPUT = TEST_DATA / "init_orient_pm_nonorm_5E4_q4_dp.focus"

WORKFLOW_STAGES = (
    "construct_ncsx_boundary_and_toroidal_background_field",
    "construct_cylindrical_famus_permanent_magnet_grid",
    "evaluate_initial_normal_field_residual",
    "run_baseline_gpmo",
    "evaluate_final_moments_and_normal_field_residual",
)


def _scale_configuration(scale: ExecutionScale) -> dict[str, object]:
    native_scale = scale == "native_default"
    return {
        "nphi": 16 if native_scale else 2,
        "ntheta": 16 if native_scale else 2,
        "downsample": 4 if native_scale else 100,
        "iterations": 500 if native_scale else 40,
        "history_count": 10,
        "regularization_l2": 0.0,
        "single_direction": -1,
        "coordinate_flag": "cylindrical",
        "net_poloidal_current_amperes": 3.7713e6,
        "surface_input_sha256": hashlib.sha256(SURFACE_INPUT.read_bytes()).hexdigest(),
        "famus_input_sha256": hashlib.sha256(FAMUS_INPUT.read_bytes()).hexdigest(),
    }


def _configuration_int(bundle: InputBundle, name: str) -> int:
    value = bundle.configuration[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"configuration {name} must be an integer")
    return value


def _configuration_float(bundle: InputBundle, name: str) -> float:
    value = bundle.configuration[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"configuration {name} must be numeric")
    return float(value)


def _build_cpu_grid(configuration: dict[str, object]):
    import io
    from contextlib import redirect_stdout

    from simsopt.field import ToroidalField
    from simsopt.geo import PermanentMagnetGrid, SurfaceRZFourier

    nphi = configuration["nphi"]
    ntheta = configuration["ntheta"]
    downsample = configuration["downsample"]
    coordinate_flag = configuration["coordinate_flag"]
    assert isinstance(nphi, int)
    assert isinstance(ntheta, int)
    assert isinstance(downsample, int)
    assert isinstance(coordinate_flag, str)
    surface = SurfaceRZFourier.from_wout(
        str(SURFACE_INPUT),
        range="half period",
        nphi=nphi,
        ntheta=ntheta,
    )
    mu0 = 4.0 * np.pi * 1.0e-7
    background = ToroidalField(
        R0=1.0,
        B0=(mu0 * float(configuration["net_poloidal_current_amperes"]) / (2.0 * np.pi)),
    )
    background.set_points(surface.gamma().reshape((-1, 3)))
    normal_field = np.sum(
        background.B().reshape((nphi, ntheta, 3)) * surface.unitnormal(),
        axis=2,
    )
    with redirect_stdout(io.StringIO()):
        grid = PermanentMagnetGrid.geo_setup_from_famus(
            surface,
            normal_field,
            FAMUS_INPUT,
            coordinate_flag=coordinate_flag,
            downsample=downsample,
        )
    return grid


def create_input(root: Path, scale: ExecutionScale) -> InputBundle:
    """Freeze the exact native fixed-state grid consumed by all three lanes."""
    configuration = _scale_configuration(scale)
    grid = _build_cpu_grid(configuration)
    moments = np.array(grid.m, dtype=np.float64, copy=True).reshape((grid.ndipoles, 3))
    m_proxy_source = grid.m_proxy if hasattr(grid, "m_proxy") else grid.m
    return create_input_bundle(
        root,
        case_id="native-permanent-magnet-simple",
        random_seed=0,
        arrays={
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
            "initial_moments": np.array(grid.m0, dtype=np.float64, copy=True).reshape(
                (grid.ndipoles, 3)
            ),
            "moments": moments,
            "proxy_moments": np.array(
                m_proxy_source, dtype=np.float64, copy=True
            ).reshape((grid.ndipoles, 3)),
            "moment_maxima": np.array(
                grid.m_maxima, dtype=np.float64, copy=True
            ).reshape((grid.ndipoles,)),
            "dipole_grid_xyz": np.array(
                grid.dipole_grid_xyz, dtype=np.float64, copy=True
            ),
        },
        configuration={
            **configuration,
            "R0": float(grid.R0),
            "nfp": int(grid.plasma_boundary.nfp),
            "stellsym": bool(grid.plasma_boundary.stellsym),
            "ndipoles": int(grid.ndipoles),
        },
        scale=scale,
    )


def _array_digest(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def _effective_fingerprint(
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
) -> str:
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
    *,
    response: np.ndarray,
    target: np.ndarray,
    moment_maxima: np.ndarray,
    dipole_grid_xyz: np.ndarray,
    initial_moments: np.ndarray,
    final_moments: np.ndarray,
    final_residual: np.ndarray,
) -> dict[str, np.ndarray]:
    initial_residual = response @ initial_moments.reshape(-1) - target
    nonzero_mask = np.linalg.norm(final_moments, axis=1) != 0.0
    return {
        "construction:response_matrix": response,
        "construction:target": target,
        "construction:moment_maxima": moment_maxima,
        "construction:dipole_grid_xyz": dipole_grid_xyz,
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
        "final:nonzero_fraction": np.asarray(
            np.count_nonzero(nonzero_mask) / nonzero_mask.size,
            dtype=np.float64,
        ),
    }


def _native(bundle: InputBundle, arrays: dict[str, np.ndarray]) -> LaneObservation:
    import simsoptpp

    response = arrays["response_matrix"]
    target = arrays["target"]
    maxima = arrays["moment_maxima"]
    maxima_vector = np.repeat(maxima, 3)
    _, _, _, normalized_moments = simsoptpp.GPMO_baseline(
        np.ascontiguousarray((response * maxima_vector).T),
        np.ascontiguousarray(target),
        np.sqrt(_configuration_float(bundle, "regularization_l2")) * maxima_vector,
        np.ascontiguousarray(arrays["normal_norms"]),
        K=_configuration_int(bundle, "iterations"),
        verbose=False,
        nhistory=_configuration_int(bundle, "history_count"),
        single_direction=_configuration_int(bundle, "single_direction"),
    )
    final_moments = np.asarray(normalized_moments, dtype=np.float64) * maxima[:, None]
    final_residual = response @ final_moments.reshape(-1) - target
    values = _values(
        response=response,
        target=target,
        moment_maxima=maxima,
        dipole_grid_xyz=arrays["dipole_grid_xyz"],
        initial_moments=arrays["initial_moments"],
        final_moments=final_moments,
        final_residual=final_residual,
    )
    selected = int(np.count_nonzero(values["final:nonzero_mask"]))
    success = bool(
        selected == _configuration_int(bundle, "iterations")
        and np.all(np.isfinite(final_moments))
        and values["final:objective_sum_squares"]
        < values["initial:objective_sum_squares"]
    )
    return LaneObservation(
        lane="native-cpu",
        backend_mode="native_cpu",
        platform="cpu",
        precision="fp64",
        scale=bundle.scale,
        input_fingerprint=bundle.input_fingerprint,
        configuration_fingerprint=bundle.configuration_fingerprint,
        effective_construction_fingerprint=_effective_fingerprint(bundle, arrays),
        driver="simsoptpp_gpmo_baseline",
        normalized_status="converged" if success else "failed",
        raw_status="fixed_iteration_budget_complete",
        success=success,
        nit=_configuration_int(bundle, "iterations"),
        nfev=_configuration_int(bundle, "iterations"),
        njev=None,
        completed_workflow_stages=WORKFLOW_STAGES,
        provenance=None,
        values=values,
    )


def _jax(
    lane: ParityLane,
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
) -> LaneObservation:
    from simsopt_jax.backend.runtime import get_runtime_jax_device
    from simsopt_jax.geo.permanent_magnet_grid import PermanentMagnetGridJAX
    from simsopt_jax.solve.permanent_magnet import GPMO_baseline_jax

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
        R0=_configuration_float(bundle, "R0"),
        nfp=_configuration_int(bundle, "nfp"),
        stellsym=bool(bundle.configuration["stellsym"]),
        nphi=_configuration_int(bundle, "nphi"),
        ntheta=_configuration_int(bundle, "ntheta"),
        ndipoles=_configuration_int(bundle, "ndipoles"),
    )
    device_result = GPMO_baseline_jax(
        grid,
        K=_configuration_int(bundle, "iterations"),
        reg_l2=_configuration_float(bundle, "regularization_l2"),
        single_direction=_configuration_int(bundle, "single_direction"),
    )
    host_result, response, target, maxima, dipole_grid_xyz, initial_moments = (
        jax.device_get(
            (
                device_result,
                grid.A_obj,
                grid.b_obj,
                grid.m_maxima,
                grid.dipole_grid_xyz,
                grid.m0,
            )
        )
    )
    construction_arrays = {
        **arrays,
        "response_matrix": np.asarray(response, dtype=np.float64),
        "target": np.asarray(target, dtype=np.float64),
        "moment_maxima": np.asarray(maxima, dtype=np.float64),
        "dipole_grid_xyz": np.asarray(dipole_grid_xyz, dtype=np.float64),
        "initial_moments": np.asarray(initial_moments, dtype=np.float64),
    }
    values = _values(
        response=construction_arrays["response_matrix"],
        target=construction_arrays["target"],
        moment_maxima=construction_arrays["moment_maxima"],
        dipole_grid_xyz=construction_arrays["dipole_grid_xyz"],
        initial_moments=construction_arrays["initial_moments"],
        final_moments=np.asarray(host_result.m, dtype=np.float64),
        final_residual=np.asarray(host_result.residual, dtype=np.float64),
    )
    selected = int(np.count_nonzero(values["final:nonzero_mask"]))
    success = bool(
        selected == _configuration_int(bundle, "iterations")
        and np.all(np.isfinite(values["final:moments"]))
        and values["final:objective_sum_squares"]
        < values["initial:objective_sum_squares"]
    )
    platform = device.platform
    return LaneObservation(
        lane=lane,
        backend_mode=os.environ["SIMSOPT_BACKEND_MODE"],
        platform="gpu" if platform in {"cuda", "gpu"} else platform,
        precision="fp64" if bool(jax.config.read("jax_enable_x64")) else "fp32",
        scale=bundle.scale,
        input_fingerprint=bundle.input_fingerprint,
        configuration_fingerprint=bundle.configuration_fingerprint,
        effective_construction_fingerprint=_effective_fingerprint(
            bundle, construction_arrays
        ),
        driver="simsopt_jax_gpmo_baseline",
        normalized_status="converged" if success else "failed",
        raw_status="fixed_iteration_budget_complete",
        success=success,
        nit=_configuration_int(bundle, "iterations"),
        nfev=_configuration_int(bundle, "iterations"),
        njev=None,
        completed_workflow_stages=WORKFLOW_STAGES,
        provenance=None,
        values=values,
    )


def execute(
    lane: ParityLane,
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
) -> LaneObservation:
    """Execute the exact permanent-magnet workflow in the selected lane."""
    if lane == "native-cpu":
        return _native(bundle, arrays)
    return _jax(lane, bundle, arrays)
