"""Exact matched workflow for ``2_Intermediate/permanent_magnet_MUSE.py``."""

from __future__ import annotations

import hashlib
import io
import os
from contextlib import redirect_stdout
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
SURFACE_INPUT = TEST_DATA / "input.muse"
FAMUS_INPUT = TEST_DATA / "zot80.focus"

WORKFLOW_STAGES = (
    "construct_muse_boundary_and_tf_coil_field",
    "construct_downsampled_famus_grid_and_face_polarizations",
    "evaluate_initial_normal_field_residual",
    "run_arbitrary_vector_backtracking_gpmo",
    "evaluate_final_moments_and_normal_field_residual",
)


def _scale_configuration(scale: ExecutionScale) -> dict[str, object]:
    native_scale = scale == "native_default"
    return {
        "nphi": 16 if native_scale else 2,
        "ntheta": 16 if native_scale else 2,
        "downsample": 10 if native_scale else 100,
        "iterations": 10_000 if native_scale else 20,
        "backtracking": 200 if native_scale else 50,
        "max_magnets": 5_000 if native_scale else 20,
        "history_count": 20,
        "adjacent_count": 1,
        "threshold_angle": float(np.pi),
        "regularization_l2": 0.0,
        "radial_extent": 0.01,
        "coordinate_flag": "cartesian",
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
    from simsopt.field import BiotSavart
    from simsopt.geo import PermanentMagnetGrid, SurfaceRZFourier
    from simsopt.util import FocusData, discretize_polarizations, polarization_axes
    from simsopt.util.permanent_magnet_helper_functions import (
        initialize_coils_for_pm_optimization,
    )

    nphi = configuration["nphi"]
    ntheta = configuration["ntheta"]
    downsample = configuration["downsample"]
    radial_extent = configuration["radial_extent"]
    assert isinstance(nphi, int)
    assert isinstance(ntheta, int)
    assert isinstance(downsample, int)
    assert isinstance(radial_extent, float)
    surface = SurfaceRZFourier.from_focus(
        SURFACE_INPUT,
        range="half period",
        nphi=nphi,
        ntheta=ntheta,
    )
    _, _, coils = initialize_coils_for_pm_optimization(
        "muse_famus",
        TEST_DATA,
        surface,
    )
    field = BiotSavart(coils)
    field.set_points(surface.gamma().reshape((-1, 3)))
    normal_field = np.sum(
        field.B().reshape((nphi, ntheta, 3)) * surface.unitnormal(),
        axis=2,
    )
    magnet_data = FocusData(FAMUS_INPUT, downsample=downsample)
    axes, polarization_types = polarization_axes(["face"])
    positive_count = len(polarization_types) // 2
    orientation = np.arctan2(magnet_data.oy, magnet_data.ox)
    discretize_polarizations(
        magnet_data,
        orientation,
        axes[:positive_count],
        polarization_types[:positive_count],
    )
    polarizations = np.stack(
        (magnet_data.pol_x, magnet_data.pol_y, magnet_data.pol_z),
        axis=-1,
    )
    with redirect_stdout(io.StringIO()):
        grid = PermanentMagnetGrid.geo_setup_from_famus(
            surface,
            normal_field,
            FAMUS_INPUT,
            pol_vectors=polarizations,
            downsample=downsample,
            dr=radial_extent,
        )
    return grid


def create_input(root: Path, scale: ExecutionScale) -> InputBundle:
    """Freeze the MUSE fixed-state grid consumed by every solver lane."""
    configuration = _scale_configuration(scale)
    grid = _build_cpu_grid(configuration)
    moments = np.array(grid.m, dtype=np.float64, copy=True).reshape((grid.ndipoles, 3))
    proxy_source = grid.m_proxy if hasattr(grid, "m_proxy") else grid.m
    return create_input_bundle(
        root,
        case_id="native-permanent-magnet-muse",
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


def _success(bundle: InputBundle, values: dict[str, np.ndarray]) -> bool:
    return bool(
        np.count_nonzero(values["final:nonzero_mask"])
        == _configuration_int(bundle, "max_magnets")
        and np.all(np.isfinite(values["final:moments"]))
        and values["final:objective_sum_squares"]
        < values["initial:objective_sum_squares"]
    )


def _observation(
    lane: ParityLane,
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
    values: dict[str, np.ndarray],
    *,
    driver: str,
    platform: str,
    precision: str,
) -> LaneObservation:
    success = _success(bundle, values)
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
        effective_construction_fingerprint=_effective_fingerprint(bundle, arrays),
        driver=driver,
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


def _native(bundle: InputBundle, arrays: dict[str, np.ndarray]) -> LaneObservation:
    import simsoptpp

    maxima = arrays["moment_maxima"]
    maxima_vector = np.repeat(maxima, 3)
    scaled_response = arrays["response_matrix"] * maxima_vector[None, :]
    _, _, _, _, normalized_moments = simsoptpp.GPMO_ArbVec_backtracking(
        np.ascontiguousarray(scaled_response.T),
        np.ascontiguousarray(arrays["target"]),
        np.sqrt(_configuration_float(bundle, "regularization_l2")) * maxima_vector,
        np.ascontiguousarray(arrays["normal_norms"]),
        np.ascontiguousarray(arrays["polarization_vectors"]),
        K=_configuration_int(bundle, "iterations"),
        verbose=False,
        nhistory=_configuration_int(bundle, "history_count"),
        backtracking=_configuration_int(bundle, "backtracking"),
        dipole_grid_xyz=np.ascontiguousarray(arrays["dipole_grid_xyz"]),
        Nadjacent=_configuration_int(bundle, "adjacent_count"),
        thresh_angle=_configuration_float(bundle, "threshold_angle"),
        max_nMagnets=_configuration_int(bundle, "max_magnets"),
        x_init=np.zeros(
            (_configuration_int(bundle, "ndipoles"), 3),
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
        _values(
            arrays=arrays,
            final_moments=final_moments,
            final_residual=final_residual,
        ),
        driver="simsoptpp_gpmo_arbvec_backtracking",
        platform="cpu",
        precision="fp64",
    )


def _jax(
    lane: ParityLane,
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
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
        R0=_configuration_float(bundle, "R0"),
        nfp=_configuration_int(bundle, "nfp"),
        stellsym=bool(bundle.configuration["stellsym"]),
        nphi=_configuration_int(bundle, "nphi"),
        ntheta=_configuration_int(bundle, "ntheta"),
        ndipoles=_configuration_int(bundle, "ndipoles"),
        pol_vectors=put("polarization_vectors"),
    )
    device_result = GPMO_ArbVec_backtracking_jax(
        grid,
        K=_configuration_int(bundle, "iterations"),
        reg_l2=_configuration_float(bundle, "regularization_l2"),
        Nadjacent=_configuration_int(bundle, "adjacent_count"),
        backtracking=_configuration_int(bundle, "backtracking"),
        thresh_angle=_configuration_float(bundle, "threshold_angle"),
        max_nMagnets=_configuration_int(bundle, "max_magnets"),
        record_every=_configuration_int(bundle, "iterations"),
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
        arrays=host_arrays,
        final_moments=np.asarray(host_result.m, dtype=np.float64),
        final_residual=np.asarray(host_result.residual, dtype=np.float64),
    )
    platform = "cpu" if device is None else device.platform
    return _observation(
        lane,
        bundle,
        host_arrays,
        values,
        driver="simsopt_jax_gpmo_arbvec_backtracking",
        platform="gpu" if platform in {"cuda", "gpu"} else platform,
        precision="fp64" if bool(jax.config.read("jax_enable_x64")) else "fp32",
    )


def execute(
    lane: ParityLane,
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
) -> LaneObservation:
    """Execute the exact MUSE permanent-magnet workflow in one solver lane."""
    if lane == "native-cpu":
        return _native(bundle, arrays)
    return _jax(lane, bundle, arrays)
