"""Exact matched workflow for ``2_Intermediate/permanent_magnet_QA.py``."""

from __future__ import annotations

import hashlib
import io
import os
from contextlib import redirect_stdout
from pathlib import Path
from typing import Mapping

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
SURFACE_INPUT = TEST_DATA / "input.LandremanPaul2021_QA_lowres"

WORKFLOW_STAGES = (
    "construct_qa_boundary_and_tf_coils",
    "optimize_tf_coil_currents_and_geometry",
    "construct_cylindrical_permanent_magnet_grid",
    "evaluate_initial_normal_field_residual",
    "run_two_stage_relax_and_split_continuation",
    "evaluate_final_sparse_moments_and_normal_field_residual",
)


def _scale_configuration(scale: ExecutionScale) -> dict[str, object]:
    native_scale = scale == "native_default"
    return {
        "nphi": 16 if native_scale else 4,
        "ntheta": 16 if native_scale else 4,
        "radial_extent": 0.02 if native_scale else 0.10,
        "coil_iterations": 500 if native_scale else 3,
        "inner_iterations": 10,
        "outer_iterations": 10 if native_scale else 1,
        "continuation_stages": 2,
        "nu": 1.0e10,
        "unscaled_reg_l0": 0.05,
        "coordinate_flag": "cylindrical",
        "surface_input_sha256": hashlib.sha256(SURFACE_INPUT.read_bytes()).hexdigest(),
    }


def _configuration_int(configuration: Mapping[str, object], name: str) -> int:
    value = configuration[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"configuration {name} must be an integer")
    return value


def _configuration_float(configuration: Mapping[str, object], name: str) -> float:
    value = configuration[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"configuration {name} must be numeric")
    return float(value)


def _build_source_grid(configuration: Mapping[str, object], output_root: Path):
    from simsopt.field import BiotSavart
    from simsopt.geo import PermanentMagnetGrid, SurfaceRZFourier
    from simsopt.util.coil_optimization_helper_functions import coil_optimization
    from simsopt.util.permanent_magnet_helper_functions import (
        initialize_coils_for_pm_optimization,
    )

    nphi = _configuration_int(configuration, "nphi")
    ntheta = _configuration_int(configuration, "ntheta")
    output_root.mkdir(parents=True, exist_ok=True)
    surface = SurfaceRZFourier.from_vmec_input(
        SURFACE_INPUT,
        range="half period",
        nphi=nphi,
        ntheta=ntheta,
    )
    inner_surface = SurfaceRZFourier.from_vmec_input(
        SURFACE_INPUT,
        range="half period",
        nphi=nphi,
        ntheta=ntheta,
    )
    outer_surface = SurfaceRZFourier.from_vmec_input(
        SURFACE_INPUT,
        range="half period",
        nphi=nphi,
        ntheta=ntheta,
    )
    inner_surface.extend_via_projected_normal(0.05)
    outer_surface.extend_via_projected_normal(0.15)
    with redirect_stdout(io.StringIO()):
        base_curves, curves, coils = initialize_coils_for_pm_optimization(
            "qa",
            TEST_DATA,
            surface,
            output_root,
        )
        field = coil_optimization(
            surface,
            BiotSavart(coils),
            base_curves,
            curves,
            MAXITER=_configuration_int(configuration, "coil_iterations"),
        )
    field.set_points(surface.gamma().reshape((-1, 3)))
    normal_field = np.sum(
        field.B().reshape((nphi, ntheta, 3)) * surface.unitnormal(),
        axis=2,
    )
    with redirect_stdout(io.StringIO()):
        grid = PermanentMagnetGrid.geo_setup_between_toroidal_surfaces(
            surface,
            normal_field,
            inner_surface,
            outer_surface,
            dr=_configuration_float(configuration, "radial_extent"),
            coordinate_flag=str(configuration["coordinate_flag"]),
        )
    return grid, np.asarray(field.x, dtype=np.float64)


def create_input(root: Path, scale: ExecutionScale) -> InputBundle:
    """Freeze the source-owned post-coil permanent-magnet problem."""
    configuration = _scale_configuration(scale)
    grid, optimized_coil_dofs = _build_source_grid(configuration, root)
    moments = np.asarray(grid.m, dtype=np.float64).reshape((grid.ndipoles, 3))
    return create_input_bundle(
        root,
        case_id="native-permanent-magnet-qa",
        random_seed=0,
        arrays={
            "response_matrix": np.asarray(grid.A_obj, dtype=np.float64),
            "target": np.asarray(grid.b_obj, dtype=np.float64),
            "atb": np.asarray(grid.ATb, dtype=np.float64).reshape(
                (grid.ndipoles, 3)
            ),
            "ata_scale_unregularized": np.asarray(
                grid.ATA_scale,
                dtype=np.float64,
            ),
            "initial_moments": np.asarray(grid.m0, dtype=np.float64).reshape(
                (grid.ndipoles, 3)
            ),
            "moments": moments,
            "proxy_moments": moments,
            "moment_maxima": np.asarray(
                grid.m_maxima,
                dtype=np.float64,
            ).reshape((grid.ndipoles,)),
            "dipole_grid_xyz": np.asarray(
                grid.dipole_grid_xyz,
                dtype=np.float64,
            ),
            "optimized_coil_dofs": optimized_coil_dofs,
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


class _NativeGrid:
    def __init__(
        self,
        bundle: InputBundle,
        arrays: dict[str, np.ndarray],
    ) -> None:
        self.A_obj = np.array(arrays["response_matrix"], copy=True)
        self.b_obj = np.array(arrays["target"], copy=True)
        self.ATb = np.array(arrays["atb"], copy=True)
        self.ATA_scale = arrays["ata_scale_unregularized"].item() + 1.0 / (
            _configuration_float(bundle.configuration, "nu")
        )
        self.m0 = np.array(arrays["initial_moments"], copy=True).reshape(-1)
        self.m = np.array(arrays["moments"], copy=True).reshape(-1)
        self.m_proxy = np.array(arrays["proxy_moments"], copy=True).reshape(-1)
        self.m_maxima = np.array(arrays["moment_maxima"], copy=True)
        self.ndipoles = _configuration_int(bundle.configuration, "ndipoles")

    def _print_initial_opt(self) -> None:
        return None


def _values(
    arrays: dict[str, np.ndarray],
    final_moments: np.ndarray,
    final_proxy: np.ndarray,
) -> dict[str, np.ndarray]:
    response = arrays["response_matrix"]
    target = arrays["target"]
    initial = arrays["initial_moments"]
    initial_residual = response @ initial.reshape(-1) - target
    final_residual = response @ final_proxy.reshape(-1) - target
    nonzero_mask = np.linalg.norm(final_proxy, axis=1) != 0.0
    return {
        "construction:response_matrix": response,
        "construction:target": target,
        "construction:moment_maxima": arrays["moment_maxima"],
        "construction:dipole_grid_xyz": arrays["dipole_grid_xyz"],
        "construction:optimized_coil_dofs": arrays["optimized_coil_dofs"],
        "initial:moments": initial,
        "initial:residual": initial_residual,
        "initial:objective_sum_squares": np.asarray(
            np.vdot(initial_residual, initial_residual),
            dtype=np.float64,
        ),
        "final:moments": final_moments,
        "final:proxy_moments": final_proxy,
        "final:residual": final_residual,
        "final:objective_sum_squares": np.asarray(
            np.vdot(final_residual, final_residual),
            dtype=np.float64,
        ),
        "final:residual_norm": np.asarray(
            np.linalg.norm(final_residual),
            dtype=np.float64,
        ),
        "final:moment_l2_norm": np.asarray(
            np.linalg.norm(final_moments),
            dtype=np.float64,
        ),
        "final:proxy_moment_l2_norm": np.asarray(
            np.linalg.norm(final_proxy),
            dtype=np.float64,
        ),
        "final:nonzero_mask": nonzero_mask,
        "final:nonzero_count": np.asarray(
            np.count_nonzero(nonzero_mask),
            dtype=np.int64,
        ),
    }


def _observation(
    lane: ParityLane,
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
    values: dict[str, np.ndarray],
    *,
    platform: str,
    precision: str,
    driver: str,
) -> LaneObservation:
    success = bool(
        np.all(np.isfinite(values["final:moments"]))
        and np.all(np.isfinite(values["final:proxy_moments"]))
        and np.count_nonzero(values["final:nonzero_mask"]) > 0
        and values["final:objective_sum_squares"]
        < values["initial:objective_sum_squares"]
    )
    iterations = (
        _configuration_int(bundle.configuration, "continuation_stages")
        * _configuration_int(bundle.configuration, "outer_iterations")
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
        effective_construction_fingerprint=_effective_fingerprint(bundle, arrays),
        driver=driver,
        normalized_status="converged" if success else "failed",
        raw_status="fixed_iteration_budget_complete",
        success=success,
        nit=iterations,
        nfev=iterations
        * _configuration_int(bundle.configuration, "inner_iterations"),
        njev=None,
        completed_workflow_stages=WORKFLOW_STAGES,
        provenance=None,
        values=values,
    )


def _native(
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
) -> LaneObservation:
    import simsoptpp
    from simsopt.solve.permanent_magnet_optimization import prox_l0

    grid = _NativeGrid(bundle, arrays)
    nu = _configuration_float(bundle.configuration, "nu")
    base_reg_l0 = (
        _configuration_float(bundle.configuration, "unscaled_reg_l0")
        / (2.0 * nu)
    )
    alpha = 2.0 * (1.0 - 1.0e-5) / grid.ATA_scale
    initial = np.array(grid.m0, copy=True)
    for stage in range(
        _configuration_int(bundle.configuration, "continuation_stages")
    ):
        reg_l0 = base_reg_l0 * (stage + 1) / 2.0
        moments = np.array(initial, copy=True)
        proxy = prox_l0(moments, grid.m_maxima, reg_l0, nu)
        for _outer in range(
            _configuration_int(bundle.configuration, "outer_iterations")
        ):
            _, _, _, moment_matrix = simsoptpp.MwPGP_algorithm(
                grid.A_obj,
                grid.b_obj,
                np.ascontiguousarray(grid.ATb),
                np.ascontiguousarray(proxy.reshape((grid.ndipoles, 3))),
                np.ascontiguousarray(moments.reshape((grid.ndipoles, 3))),
                grid.m_maxima,
                alpha,
                nu,
                0.0,
                reg_l0,
                0.0,
                0.0,
                _configuration_int(
                    bundle.configuration,
                    "inner_iterations",
                ),
                0.0,
                False,
            )
            moments = np.asarray(moment_matrix, dtype=np.float64).reshape(-1)
            proxy = prox_l0(moments, grid.m_maxima, reg_l0, nu)
        initial = moments
    grid.m = moments
    grid.m_proxy = proxy
    values = _values(
        arrays,
        np.asarray(grid.m, dtype=np.float64).reshape((-1, 3)),
        np.asarray(grid.m_proxy, dtype=np.float64).reshape((-1, 3)),
    )
    return _observation(
        "native-cpu",
        bundle,
        arrays,
        values,
        platform="cpu",
        precision="fp64",
        driver="simsoptpp_mwpgp_relax_and_split",
    )


def _jax(
    lane: ParityLane,
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
) -> LaneObservation:
    import jax
    from simsopt_jax.backend.runtime import get_runtime_jax_device
    from simsopt_jax.geo.permanent_magnet_grid import PermanentMagnetGridJAX
    from simsopt_jax.solve.permanent_magnet import relax_and_split_jax

    device = get_runtime_jax_device()

    def put(name: str):
        return jax.device_put(arrays[name], device)

    grid = PermanentMagnetGridJAX(
        A_obj=put("response_matrix"),
        b_obj=put("target"),
        ATb=put("atb"),
        ATA_scale=jax.device_put(
            np.asarray(
                arrays["ata_scale_unregularized"].item(),
                dtype=np.float64,
            ),
            device,
        ),
        m0=put("initial_moments"),
        m=put("moments"),
        m_proxy=put("proxy_moments"),
        m_maxima=put("moment_maxima"),
        dipole_grid_xyz=put("dipole_grid_xyz"),
        coordinate_flag=str(bundle.configuration["coordinate_flag"]),
        R0=_configuration_float(bundle.configuration, "R0"),
        nfp=_configuration_int(bundle.configuration, "nfp"),
        stellsym=bool(bundle.configuration["stellsym"]),
        nphi=_configuration_int(bundle.configuration, "nphi"),
        ntheta=_configuration_int(bundle.configuration, "ntheta"),
        ndipoles=_configuration_int(bundle.configuration, "ndipoles"),
    )
    nu = _configuration_float(bundle.configuration, "nu")
    base_reg_l0 = (
        _configuration_float(bundle.configuration, "unscaled_reg_l0")
        / (2.0 * nu)
    )
    alpha = (
        2.0
        * (1.0 - 1.0e-5)
        / (arrays["ata_scale_unregularized"].item() + 1.0 / nu)
    )
    initial = grid.m0
    result = None
    for stage in range(
        _configuration_int(bundle.configuration, "continuation_stages")
    ):
        result = relax_and_split_jax(
            grid,
            m0=initial,
            alpha=alpha,
            nu=nu,
            max_iter=_configuration_int(
                bundle.configuration,
                "inner_iterations",
            ),
            max_iter_RS=_configuration_int(
                bundle.configuration,
                "outer_iterations",
            ),
            reg_l0=base_reg_l0 * (stage + 1) / 2.0,
        )
        initial = result.m
    if result is None:
        raise RuntimeError("continuation requires at least one stage")
    final_moments, final_proxy = jax.device_get((result.m, result.m_proxy))
    values = _values(
        arrays,
        np.asarray(final_moments, dtype=np.float64),
        np.asarray(final_proxy, dtype=np.float64),
    )
    platform = "cpu" if device is None else device.platform
    return _observation(
        lane,
        bundle,
        arrays,
        values,
        platform="gpu" if platform in {"cuda", "gpu"} else platform,
        precision="fp64" if bool(jax.config.read("jax_enable_x64")) else "fp32",
        driver="simsopt_jax_mwpgp_relax_and_split",
    )


def execute(
    lane: ParityLane,
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
) -> LaneObservation:
    """Execute the matched two-stage permanent-magnet continuation."""
    if lane == "native-cpu":
        return _native(bundle, arrays)
    return _jax(lane, bundle, arrays)
