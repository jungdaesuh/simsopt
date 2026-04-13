from __future__ import annotations

import argparse
import importlib.util
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

import simsopt.config as simsopt_config  # type: ignore[import-untyped]
from simsopt.field import Coil, Current, coils_via_symmetries  # type: ignore[import-untyped]
from simsopt.field.coil import ScaledCurrent  # type: ignore[import-untyped]
from simsopt.geo import (  # type: ignore[import-untyped]
    FrameRotation,
    FramedCurveCentroid,
    SurfaceRZFourier,
    SurfaceXYZTensorFourier,
    create_equally_spaced_curves,
)
from simsopt.geo.boozersurface_jax import _surface_sample_z  # type: ignore[import-untyped]
from simsopt.geo.curve import gamma_2d  # type: ignore[import-untyped]
from simsopt.geo.curveperturbed import (  # type: ignore[import-untyped]
    CurvePerturbed,
    GaussianSampler,
    PerturbationSample,
)
from simsopt.geo.curveobjectives import pairwise_min_distance_pure  # type: ignore[import-untyped]
from simsopt.geo.curveobjectives import (  # type: ignore[import-untyped]
    cc_distance_pure,
    cs_distance_pure,
    CurveCurveDistance,
    CurveLength,
    CurveSurfaceDistance,
    FramedCurveTwist,
    LpCurveCurvature,
    LpCurveCurvatureBarrier,
    LpCurveTorsion,
)
from simsopt.geo.curvecwsfourier import CurveCWSFourierCPP  # type: ignore[import-untyped]
from simsopt.geo.curvexyzfourier import JaxCurveXYZFourier  # type: ignore[import-untyped]
from simsopt.geo.optimizer_jax import (  # type: ignore[import-untyped]
    _mark_cacheable_jit_value_and_grad,
    jax_minimize,
    private_optimizer_runtime_is_supported,
    target_minimize,
)
from simsopt.jax_core.biotsavart import (  # type: ignore[import-untyped]
    biot_savart_A,
    biot_savart_B,
)
from simsopt.jax_core.curve_geometry import (  # type: ignore[import-untyped]
    closed_curve_self_intersection_summary,
)
from simsopt.jax_core.field import (  # type: ignore[import-untyped]
    grouped_biot_savart_B_from_spec,
    grouped_coil_set_spec_from_lists,
)
from simsopt.jax_core.surface_rzfourier import (  # type: ignore[import-untyped]
    surface_rz_fourier_gamma_from_spec,
    surface_rz_fourier_normal_from_spec,
)
from simsopt.jax_core.specs import make_coil_symmetry_spec  # type: ignore[import-untyped]
from simsopt.objectives.stage2_target_objective_jax import (  # type: ignore[import-untyped]
    Stage2PenaltyConfig,
    build_stage2_target_objective,
)
from simsopt.geo.surface_fourier_jax import (  # type: ignore[import-untyped]
    stellsym_scatter_indices,
    surface_gamma_from_dofs,
)

OptimizerMethod = Literal["lbfgs-ondevice", "bfgs-ondevice"]
LegacyCurveObjectiveValueCase = Literal[
    "curve-length",
    "lp-curve-curvature",
    "curve-curve-distance",
    "curve-surface-distance",
    "lp-curve-curvature-barrier",
    "lp-curve-torsion",
    "framed-curve-twist",
]
LegacyCurveObjectiveGradientCase = Literal[
    "lp-curve-curvature-barrier",
    "lp-curve-curvature",
    "curve-curve-distance",
    "curve-surface-distance",
    "lp-curve-torsion",
    "framed-curve-twist",
]
_REPO_ROOT = Path(__file__).resolve().parents[2]
_STAGE2_SCRIPT_PATH = (
    _REPO_ROOT
    / "examples"
    / "single_stage_optimization"
    / "STAGE_2"
    / "banana_coil_solver.py"
)


def _configure_strict_cpu_parity_backend() -> bool:
    simsopt_config.set_backend(
        "jax_cpu_parity",
        strict=True,
        transfer_guard="disallow",
    )
    return private_optimizer_runtime_is_supported(jax.__version__)


def _configure_transfer_guard_cpu_parity_backend() -> None:
    """Non-strict backend with transfer_guard=disallow only.

    Used by cases that exercise C++ geometry objects (CurveCWSFourierCPP,
    CurvePerturbed) where CPU fallback is expected behaviour.
    """
    simsopt_config.set_backend(
        "jax_cpu_parity",
        transfer_guard="disallow",
    )


class _CompileCounter(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.count = 0

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if "Compiling jit(" in message and "_run_solver)" in message:
            self.count += 1


def _assert_run_solver_compiles_once(run_once) -> None:
    logger = logging.getLogger("jax")
    old_level = logger.level
    handler = _CompileCounter()
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    try:
        jax.clear_caches()
        with jax.log_compiles(True):
            for _ in range(3):
                run_once()
        assert handler.count == 1, handler.count
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)


def _run_compile_count_case(method: OptimizerMethod) -> None:
    if not _configure_strict_cpu_parity_backend():
        return

    half = jax.device_put(np.asarray(0.5, dtype=np.float64))

    def quad(x: jax.Array) -> jax.Array:
        vector = jnp.asarray(x, dtype=jnp.float64)
        return half * jnp.dot(vector, vector)

    cacheable_quad = _mark_cacheable_jit_value_and_grad(quad)
    x0 = jnp.asarray(np.array([1.0, -2.0], dtype=np.float64))

    def run_once() -> None:
        result = jax_minimize(cacheable_quad, x0, method=method, maxiter=5)
        assert result.success is True

    _assert_run_solver_compiles_once(run_once)


def _run_biot_savart_point_chunking_case() -> None:
    _configure_strict_cpu_parity_backend()

    points = jax.device_put(
        np.arange(257 * 3, dtype=np.float64).reshape(257, 3) * 1e-3
    )
    gammas = jax.device_put(
        np.linspace(0.0, 1.0, 2 * 8 * 3, dtype=np.float64).reshape(2, 8, 3)
    )
    gammadashs = jax.device_put(np.full((2, 8, 3), 0.25, dtype=np.float64))
    currents = jax.device_put(np.array([1.0, -0.5], dtype=np.float64))

    magnetic_field = biot_savart_B(points, gammas, gammadashs, currents)
    magnetic_vector_potential = biot_savart_A(points, gammas, gammadashs, currents)

    assert magnetic_field.shape == (257, 3)
    assert magnetic_vector_potential.shape == (257, 3)
    assert np.all(np.isfinite(np.asarray(magnetic_field)))
    assert np.all(np.isfinite(np.asarray(magnetic_vector_potential)))


def _run_target_compile_count_case() -> None:
    if not _configure_strict_cpu_parity_backend():
        return

    half = jax.device_put(np.asarray(0.5, dtype=np.float64))

    def quad_value_and_grad(x: jax.Array) -> tuple[jax.Array, jax.Array]:
        vector = jnp.asarray(x, dtype=jnp.float64)
        value = half * jnp.dot(vector, vector)
        grad = vector
        return value, grad

    cacheable_quad_value_and_grad = _mark_cacheable_jit_value_and_grad(
        jax.jit(quad_value_and_grad)
    )
    x0 = jnp.asarray(np.array([1.0, -2.0], dtype=np.float64))

    def run_once() -> None:
        result = target_minimize(
            cacheable_quad_value_and_grad,
            x0,
            method="lbfgs-ondevice",
            value_and_grad=True,
            maxiter=5,
        )
        assert result.success is True

    _assert_run_solver_compiles_once(run_once)


def _load_stage2_script_module():
    spec = importlib.util.spec_from_file_location(
        "stage2_banana_coil_solver_subprocess",
        _STAGE2_SCRIPT_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Failed to load Stage 2 script module from {_STAGE2_SCRIPT_PATH}"
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_stage2_target_compile_count_case() -> None:
    if not _configure_strict_cpu_parity_backend():
        return

    stage2_script = _load_stage2_script_module()
    cpu = jax.devices("cpu")[0]
    bundle, dofs_device = _build_stage2_target_objective_test_bundle(cpu)
    contract = stage2_script.resolve_stage2_optimizer_contract("jax", "ondevice")
    value_and_grad = bundle.value_and_grad
    if value_and_grad is None:
        raise RuntimeError("Stage 2 target bundle must expose value_and_grad.")
    initial_dofs = np.asarray(jax.device_get(dofs_device), dtype=np.float64)

    def run_once() -> None:
        optimizer_state = stage2_script.build_stage2_target_optimizer_state(
            bundle,
            initial_dofs,
        )
        result = stage2_script.run_stage2_optimizer(
            value_and_grad_fun=value_and_grad,
            dofs=optimizer_state,
            contract=contract,
            maxiter=5,
            ftol=0.0,
            gtol=1e-12,
            maxcor=7,
            scalar_fun=bundle.objective,
        )
        final_dofs = stage2_script.flatten_stage2_target_optimizer_state(result.x)
        assert final_dofs.shape == initial_dofs.shape
        assert np.all(np.isfinite(np.asarray(final_dofs, dtype=np.float64)))

    _assert_run_solver_compiles_once(run_once)


class _ShiftedQuadratic:
    def __init__(self, target: Sequence[float]) -> None:
        self.target = np.asarray(tuple(target), dtype=np.float64)
        self.half = jax.device_put(np.asarray(0.5, dtype=np.float64))

    def __call__(self, x: jax.Array) -> jax.Array:
        vector = jnp.asarray(x, dtype=jnp.float64)
        target = jnp.asarray(self.target, dtype=jnp.float64)
        diff = vector - target
        return self.half * jnp.dot(diff, diff)


def _find_gpu_device() -> jax.Device | None:
    for device in jax.devices():
        if device.platform == "gpu":
            return device
    return None


def _configure_strict_gpu_fast_backend() -> jax.Device | None:
    gpu = _find_gpu_device()
    if gpu is None:
        return None

    simsopt_config.set_backend(
        "jax_gpu_fast",
        strict=True,
        transfer_guard="disallow",
    )
    return gpu


def _build_grouped_biot_savart_device_geometry(
    gpu: jax.Device,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    points = jax.device_put(
        np.linspace(0.0, 1.0, 4 * 3, dtype=np.float64).reshape(4, 3),
        device=gpu,
    )
    gamma0 = jax.device_put(
        np.linspace(0.2, 0.8, 8 * 3, dtype=np.float64).reshape(8, 3),
        device=gpu,
    )
    gamma1 = jax.device_put(
        np.linspace(0.3, 0.9, 8 * 3, dtype=np.float64).reshape(8, 3),
        device=gpu,
    )
    gammadash0 = jax.device_put(
        np.full((8, 3), 0.1, dtype=np.float64),
        device=gpu,
    )
    gammadash1 = jax.device_put(
        np.full((8, 3), 0.15, dtype=np.float64),
        device=gpu,
    )
    return points, gamma0, gamma1, gammadash0, gammadash1


def _build_single_grouped_biot_savart_device_geometry(
    gpu: jax.Device,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    points = jax.device_put(
        np.linspace(0.0, 1.0, 4 * 3, dtype=np.float64).reshape(4, 3),
        device=gpu,
    )
    gamma = jax.device_put(
        np.linspace(0.2, 0.8, 8 * 3, dtype=np.float64).reshape(8, 3),
        device=gpu,
    )
    gammadash = jax.device_put(
        np.full((8, 3), 0.1, dtype=np.float64),
        device=gpu,
    )
    return points, gamma, gammadash


def _build_point_sharding_mesh() -> Mesh:
    devices = np.asarray(jax.devices(), dtype=object)
    return Mesh(devices, ("d",))


def _build_curvecws_surface() -> SurfaceRZFourier:
    return SurfaceRZFourier(
        nfp=5,
        stellsym=True,
        mpol=1,
        ntor=0,
        quadpoints_phi=np.arange(64) / 64,
        quadpoints_theta=np.arange(64) / 64,
    )


def _run_grouped_biot_savart_gpu_spec_eval_case() -> None:
    gpu = _configure_strict_gpu_fast_backend()
    if gpu is None:
        return

    points, gamma, gammadash = _build_single_grouped_biot_savart_device_geometry(gpu)
    current = jax.device_put(np.float64(1.25), device=gpu)
    coil_spec = grouped_coil_set_spec_from_lists([gamma], [gammadash], [current])
    magnetic_field = grouped_biot_savart_B_from_spec(points, coil_spec)

    assert magnetic_field.shape == (4, 3)
    assert np.all(np.isfinite(np.asarray(jax.device_get(magnetic_field))))


def _run_grouped_biot_savart_explicit_point_sharding_case() -> None:
    mesh = _build_point_sharding_mesh()
    points = jax.device_put(
        np.linspace(0.0, 1.0, 4 * 3, dtype=np.float64).reshape(4, 3),
        NamedSharding(mesh, P("d", None)),
    )
    gamma = jax.device_put(
        np.linspace(0.2, 0.8, 8 * 3, dtype=np.float64).reshape(8, 3),
    )
    gammadash = jax.device_put(
        np.full((8, 3), 0.1, dtype=np.float64),
    )
    current = jax.device_put(np.float64(1.25))
    coil_spec = grouped_coil_set_spec_from_lists([gamma], [gammadash], [current])
    magnetic_field = grouped_biot_savart_B_from_spec(points, coil_spec)

    assert magnetic_field.shape == (4, 3)
    assert isinstance(magnetic_field.sharding, NamedSharding)
    assert bool(jax.device_get(jnp.all(jnp.isfinite(magnetic_field))))


def _run_pairwise_penalty_explicit_row_sharding_case() -> None:
    mesh = _build_point_sharding_mesh()
    points_a = jax.device_put(
        np.linspace(0.0, 0.9, 4 * 3, dtype=np.float64).reshape(4, 3),
        NamedSharding(mesh, P("d", None)),
    )
    points_b = np.linspace(0.1, 0.7, 3 * 3, dtype=np.float64).reshape(3, 3)
    value = pairwise_min_distance_pure(points_a, points_b, chunk_size=2)

    assert np.isfinite(float(value))
    assert float(value) >= 0.0


def _run_shifted_grid_axis_sample_case() -> None:
    _configure_strict_cpu_parity_backend()

    rz = SurfaceRZFourier(
        nfp=5,
        stellsym=True,
        mpol=1,
        ntor=0,
        quadpoints_phi=np.array([0.17]),
        quadpoints_theta=np.array([0.31]),
    )
    rz.set_zs(1, 0, 1.0)
    rz_gamma = np.asarray(rz.gamma(), dtype=np.float64)
    rz_sample = float(jax.device_get(_surface_sample_z(jax.device_put(rz_gamma))))
    assert np.isclose(rz_sample, float(rz_gamma[0, 0, 2]))

    xyz = SurfaceXYZTensorFourier(
        mpol=1,
        ntor=1,
        stellsym=True,
        nfp=1,
        quadpoints_phi=np.array([0.23]),
        quadpoints_theta=np.array([0.37]),
    )
    xyz_dofs = xyz.get_dofs().copy()
    for index in range(min(6, xyz_dofs.size)):
        xyz_dofs[-(index + 1)] += 0.01 * (index + 1)
    xyz.set_dofs(xyz_dofs)
    xyz_gamma = np.asarray(xyz.gamma(), dtype=np.float64)
    xyz_sample = float(jax.device_get(_surface_sample_z(jax.device_put(xyz_gamma))))
    assert np.isclose(xyz_sample, float(xyz_gamma[0, 0, 2]))


def _run_gamma_2d_eager_host_constants_case() -> None:
    if _configure_strict_gpu_fast_backend() is None:
        return

    modes = np.zeros(10, dtype=np.float64)
    qpts = np.linspace(0.0, 1.0, 8, endpoint=False)
    phi, theta = gamma_2d(modes, qpts, 2, G=1, H=0)

    assert phi.shape == (8,)
    assert theta.shape == (8,)


def _run_closed_curve_self_intersection_summary_case() -> None:
    gpu = _configure_strict_gpu_fast_backend()
    if gpu is None:
        return

    gamma = jax.device_put(
        np.asarray(
            (
                (0.0, 0.0, 0.0),
                (1.0, 1.0, 0.0),
                (0.0, 1.0, 0.0),
                (1.0, 0.0, 0.0),
            ),
            dtype=np.float64,
        ),
        device=gpu,
    )
    summary = closed_curve_self_intersection_summary(gamma, neighbor_skip=1)
    min_distance = jax.device_get(summary[0])
    penalty = jax.device_get(summary[2])
    violation = jax.device_get(summary[3])

    assert np.isfinite(float(min_distance))
    assert np.isfinite(float(penalty))
    assert bool(violation)


def _run_surface_xyztensorfourier_gamma_from_dofs_case() -> None:
    gpu = _configure_strict_gpu_fast_backend()
    if gpu is None:
        return

    surf = SurfaceXYZTensorFourier(
        mpol=1,
        ntor=1,
        stellsym=True,
        nfp=1,
        quadpoints_phi=np.array([0.23, 0.41]),
        quadpoints_theta=np.array([0.37, 0.59]),
    )
    dofs = np.asarray(surf.get_dofs(), dtype=np.float64)
    scatter = stellsym_scatter_indices(surf.mpol, surf.ntor)

    def gamma_from_dofs(dofs_arg: jax.Array) -> jax.Array:
        return surface_gamma_from_dofs(
            dofs_arg,
            surf.quadpoints_phi,
            surf.quadpoints_theta,
            surf.mpol,
            surf.ntor,
            surf.nfp,
            surf.stellsym,
            scatter,
        )

    gamma = jax.jit(gamma_from_dofs)(jax.device_put(dofs, device=gpu))

    assert gamma.shape == (2, 2, 3)
    assert bool(jax.device_get(jnp.all(jnp.isfinite(gamma))))


def _run_coil_symmetry_spec_identity_default_case() -> None:
    _configure_transfer_guard_cpu_parity_backend()

    symmetry = make_coil_symmetry_spec(scale=2.5)

    assert symmetry.rotmat.shape == (3, 3)
    assert np.allclose(np.asarray(symmetry.rotmat), np.eye(3))
    assert symmetry.has_rotation is False


def _build_surface_rzfourier_transfer_guard_surface() -> SurfaceRZFourier:
    _configure_transfer_guard_cpu_parity_backend()
    return SurfaceRZFourier(
        nfp=5,
        stellsym=True,
        mpol=1,
        ntor=0,
        quadpoints_phi=np.arange(16) / 16,
        quadpoints_theta=np.arange(16) / 16,
    )


def _run_pairwise_curve_penalty_pure_functions_case() -> None:
    _configure_transfer_guard_cpu_parity_backend()

    gamma_curve = jax.device_put(
        np.array(
            [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]],
            dtype=np.float64,
        )
    )
    gammadash_curve = jax.device_put(
        np.array(
            [[0.1, 0.0, 0.0], [0.1, 0.0, 0.0]],
            dtype=np.float64,
        )
    )
    gamma_other = jax.device_put(
        np.array(
            [[1.0, 0.0, 0.0], [1.1, 0.0, 0.0]],
            dtype=np.float64,
        )
    )
    surface_normals = jax.device_put(
        np.array(
            [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float64,
        )
    )

    cc_value = cc_distance_pure(
        gamma_curve,
        gammadash_curve,
        gamma_other,
        gammadash_curve,
        0.05,
    )
    cs_value = cs_distance_pure(
        gamma_curve,
        gammadash_curve,
        gamma_other,
        surface_normals,
        0.05,
    )

    assert float(jax.device_get(cc_value)) == 0.0
    assert float(jax.device_get(cs_value)) == 0.0


def _run_surfacerzfourier_spec_defaults_case() -> None:
    surf = _build_surface_rzfourier_transfer_guard_surface()
    spec = surf.surface_spec()

    assert spec.rs.shape == spec.rc.shape
    assert spec.zc.shape == spec.rc.shape


def _run_surface_rzfourier_gamma_from_spec_case() -> None:
    surf = _build_surface_rzfourier_transfer_guard_surface()
    gamma = surface_rz_fourier_gamma_from_spec(surf.surface_spec())

    assert gamma.shape == (16, 16, 3)
    assert bool(jax.device_get(jnp.all(jnp.isfinite(gamma))))


def _run_surface_rzfourier_normal_from_spec_case() -> None:
    surf = _build_surface_rzfourier_transfer_guard_surface()
    normal = surface_rz_fourier_normal_from_spec(surf.surface_spec())

    assert normal.shape == (16, 16, 3)
    assert bool(jax.device_get(jnp.all(jnp.isfinite(normal))))


def _build_legacy_curve_objective_common_fixture() -> (
    tuple[Sequence[object], SurfaceRZFourier, FrameRotation]
):
    _configure_transfer_guard_cpu_parity_backend()

    curves = create_equally_spaced_curves(
        2,
        1,
        stellsym=False,
        R0=1.0,
        R1=0.2,
        order=3,
        numquadpoints=33,
    )
    surface = SurfaceRZFourier(
        nfp=1,
        stellsym=False,
        mpol=1,
        ntor=0,
        quadpoints_phi=np.arange(16) / 16,
        quadpoints_theta=np.arange(16) / 16,
    )
    rotation = FrameRotation(curves[0].quadpoints, order=1)
    rotation.x = np.array([0.1, -0.2, 0.05])
    return curves, surface, rotation


def _build_legacy_curve_objective_value_fixture() -> (
    tuple[Sequence[object], SurfaceRZFourier, FrameRotation]
):
    return _build_legacy_curve_objective_common_fixture()


def _build_legacy_curve_objective_gradient_fixture() -> (
    tuple[Sequence[object], SurfaceRZFourier, FrameRotation]
):
    curves, surface, rotation = _build_legacy_curve_objective_common_fixture()
    surface.set_rc(0, 0, 1.2)
    surface.set_rc(1, 0, 0.15)
    surface.set_zs(1, 0, 0.15)
    return curves, surface, rotation


def _run_legacy_curve_objective_value_case(
    case: LegacyCurveObjectiveValueCase,
) -> None:
    curves, surface, rotation = _build_legacy_curve_objective_value_fixture()

    if case == "curve-length":
        value = CurveLength(curves[0]).J()
        assert np.isfinite(float(value))
        return
    if case == "lp-curve-curvature":
        value = LpCurveCurvature(curves[0], p=4, threshold=10.0).J()
        assert np.isfinite(float(value))
        return
    if case == "curve-curve-distance":
        value = CurveCurveDistance(curves, 0.05).J()
        assert np.isfinite(float(value))
        return
    if case == "curve-surface-distance":
        value = CurveSurfaceDistance(curves, surface, 0.02).J()
        assert np.isfinite(float(value))
        return
    if case == "lp-curve-curvature-barrier":
        value = LpCurveCurvatureBarrier(curves[0], threshold=10.0).J()
        assert np.isfinite(float(value))
        return
    if case == "lp-curve-torsion":
        value = LpCurveTorsion(curves[0], p=4, threshold=10.0).J()
        assert np.isfinite(float(value))
        return
    if case == "framed-curve-twist":
        value = FramedCurveTwist(
            FramedCurveCentroid(curves[0], rotation),
            f="lp",
            p=2,
        ).J()
        assert np.isfinite(float(value))
        return
    raise ValueError(f"unsupported legacy curve objective value case {case!r}")


def _run_legacy_curve_objective_gradient_case(
    case: LegacyCurveObjectiveGradientCase,
) -> None:
    curves, surface, rotation = _build_legacy_curve_objective_gradient_fixture()

    if case == "lp-curve-curvature-barrier":
        grad = np.asarray(
            LpCurveCurvatureBarrier(curves[0], threshold=10.0).dJ(),
            dtype=float,
        )
        assert grad.size > 0
        assert np.all(np.isfinite(grad))
        return
    if case == "lp-curve-curvature":
        grad = np.asarray(
            LpCurveCurvature(curves[0], p=4, threshold=10.0).dJ(),
            dtype=float,
        )
        assert grad.size > 0
        assert np.all(np.isfinite(grad))
        return
    if case == "curve-curve-distance":
        grad = np.asarray(CurveCurveDistance(curves, 0.05).dJ(), dtype=float)
        assert grad.size > 0
        assert np.all(np.isfinite(grad))
        return
    if case == "curve-surface-distance":
        grad = np.asarray(
            CurveSurfaceDistance(curves, surface, 0.02).dJ(),
            dtype=float,
        )
        assert grad.size > 0
        assert np.all(np.isfinite(grad))
        return
    if case == "lp-curve-torsion":
        grad = np.asarray(
            LpCurveTorsion(curves[0], p=4, threshold=10.0).dJ(),
            dtype=float,
        )
        assert grad.size > 0
        assert np.all(np.isfinite(grad))
        return
    if case == "framed-curve-twist":
        grad = np.asarray(
            FramedCurveTwist(
                FramedCurveCentroid(curves[0], rotation),
                f="lp",
                p=2,
            ).dJ(),
            dtype=float,
        )
        assert grad.size > 0
        assert np.all(np.isfinite(grad))
        return
    raise ValueError(f"unsupported legacy curve objective gradient case {case!r}")


def _run_curvecwsfouriercpp_init_case() -> None:
    _configure_transfer_guard_cpu_parity_backend()

    quadpoints = np.linspace(0.0, 1.0, 33, endpoint=False)
    surf = _build_curvecws_surface()
    curve = CurveCWSFourierCPP(quadpoints, 3, surf, G=0, H=0)
    assert curve.numquadpoints == 33


def _run_curvecwsfouriercpp_curve_length_gradient_case() -> None:
    _configure_transfer_guard_cpu_parity_backend()

    quadpoints = np.linspace(0.0, 1.0, 33, endpoint=False)
    surf = _build_curvecws_surface()
    curve = CurveCWSFourierCPP(quadpoints, 3, surf, G=0, H=0)
    curve.set("thetas(1)", 0.1)
    curve.set("phic(1)", 0.05)
    objective = CurveLength(curve)
    value = objective.J()
    grad = objective.dJ(partials=True)(curve)

    assert np.isfinite(float(value))
    assert grad.shape == (curve.dof_size,)
    assert np.all(np.isfinite(grad))


def _run_curvecwsfouriercpp_curve_distance_gradient_case() -> None:
    _configure_transfer_guard_cpu_parity_backend()

    quadpoints = np.linspace(0.0, 1.0, 33, endpoint=False)
    surf = _build_curvecws_surface()
    banana_curve = CurveCWSFourierCPP(quadpoints, 3, surf, G=0, H=0)
    banana_curve.set("phic(0)", 0.05)
    banana_curve.set("thetas(1)", 0.1)
    tf_curves = create_equally_spaced_curves(
        2,
        5,
        stellsym=False,
        R0=1.0,
        R1=0.35,
        order=1,
        numquadpoints=33,
    )
    objective = CurveCurveDistance([banana_curve, *tf_curves], 0.05)
    value = objective.J()
    grad = objective.dJ(partials=True)(banana_curve)

    assert np.isfinite(float(value))
    assert grad.shape == (banana_curve.dof_size,)
    assert np.all(np.isfinite(grad))


def _run_curveperturbed_init_case() -> None:
    quadpoints = np.linspace(0.0, 1.0, 33, endpoint=False)
    sampler = GaussianSampler(quadpoints, sigma=0.1, length_scale=0.2, n_derivs=1)
    sample = PerturbationSample(sampler)
    curve = JaxCurveXYZFourier(33, 1)
    curve.x = np.array([1.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1])
    _configure_transfer_guard_cpu_parity_backend()
    perturbed = CurvePerturbed(curve, sample)
    gamma = perturbed.gamma_jax(jax.device_put(np.asarray(perturbed.full_x)))

    assert gamma.shape == (33, 3)


def _build_stage2_target_objective_test_bundle(gpu: jax.Device):
    eval_surf = SurfaceRZFourier.from_nphi_ntheta(
        nfp=1,
        stellsym=True,
        mpol=1,
        ntor=0,
        nphi=16,
        ntheta=16,
    )
    eval_dofs = eval_surf.get_dofs()
    eval_dofs[0] = 1.0
    eval_dofs[1] = 0.15
    eval_surf.set_dofs(eval_dofs)

    coil_surf = SurfaceRZFourier.from_nphi_ntheta(
        nfp=1,
        stellsym=True,
        mpol=1,
        ntor=0,
        nphi=16,
        ntheta=16,
    )
    coil_dofs = coil_surf.get_dofs()
    coil_dofs[0] = 1.15
    coil_dofs[1] = 0.18
    coil_surf.set_dofs(coil_dofs)

    tf_curves = create_equally_spaced_curves(
        2,
        1,
        stellsym=False,
        R0=1.0,
        R1=0.25,
        order=1,
        numquadpoints=33,
    )
    tf_currents = [Current(1.0) * 1e5 for _ in tf_curves]
    for tf_curve in tf_curves:
        tf_curve.fix_all()
    for tf_current in tf_currents:
        tf_current.fix_all()
    tf_coils = [
        Coil(curve, current) for curve, current in zip(tf_curves, tf_currents)
    ]

    quadpoints = np.linspace(0.0, 1.0, 33, endpoint=False)
    banana_curve = CurveCWSFourierCPP(quadpoints, 2, coil_surf, G=0, H=0)
    banana_curve.set("phic(0)", 0.05)
    banana_curve.set("thetac(0)", 0.45)
    banana_curve.set("phic(1)", 0.03)
    banana_curve.set("thetas(1)", 0.08)
    banana_current = Current(1.0)
    banana_coils = coils_via_symmetries(
        [banana_curve],
        [ScaledCurrent(banana_current, 1e4)],
        coil_surf.nfp,
        coil_surf.stellsym,
    )

    bundle = build_stage2_target_objective(
        surface=eval_surf,
        tf_coils=tf_coils,
        banana_coils=banana_coils,
        banana_curve=banana_curve,
        penalty_config=Stage2PenaltyConfig(
            squared_flux_weight=1.0,
            length_weight=0.0005,
            length_target=1.75,
            cc_weight=100.0,
            cc_threshold=0.05,
            curvature_weight=0.0001,
            curvature_threshold=40.0,
            curvature_p_norm=4,
        ),
    )

    dofs = np.concatenate(
        (
            np.array([1.0], dtype=np.float64),
            np.asarray(banana_curve.get_dofs(), dtype=np.float64),
        )
    )
    dofs_jax = jax.device_put(dofs, device=gpu)
    return bundle, dofs_jax


def _run_stage2_target_objective_host_closure_constants_case() -> None:
    gpu = _configure_strict_gpu_fast_backend()
    if gpu is None:
        return

    bundle, dofs_jax = _build_stage2_target_objective_test_bundle(gpu)
    value = bundle.objective(dofs_jax)

    assert np.isfinite(float(jax.device_get(value)))


def _run_stage2_target_objective_ondevice_entry_case() -> None:
    gpu = _configure_strict_gpu_fast_backend()
    if gpu is None:
        return

    bundle, dofs_jax = _build_stage2_target_objective_test_bundle(gpu)
    _, pullback = jax.vjp(bundle.objective, dofs_jax)
    grad = pullback(jax.device_put(np.array(1.0, dtype=np.float64), device=gpu))[0]
    result = jax_minimize(bundle.objective, dofs_jax, method="lbfgs-ondevice", maxiter=1)

    assert np.isfinite(float(jax.device_get(bundle.objective(dofs_jax))))
    assert np.all(np.isfinite(np.asarray(jax.device_get(grad), dtype=np.float64)))
    assert hasattr(result, "success")


def _run_grouped_biot_savart_gpu_current_arrays_case() -> None:
    gpu = _configure_strict_gpu_fast_backend()
    if gpu is None:
        return

    points, gamma0, gamma1, gammadash0, gammadash1 = (
        _build_grouped_biot_savart_device_geometry(gpu)
    )
    currents = jax.device_put(
        np.asarray([1.25, -0.75], dtype=np.float64),
        device=gpu,
    )
    coil_spec = grouped_coil_set_spec_from_lists(
        (gamma0, gamma1),
        (gammadash0, gammadash1),
        currents,
    )
    magnetic_field = grouped_biot_savart_B_from_spec(points, coil_spec)

    assert magnetic_field.shape == (4, 3)
    assert np.all(np.isfinite(np.asarray(jax.device_get(magnetic_field))))


def _run_grouped_biot_savart_host_scalar_currents_case() -> None:
    gpu = _configure_strict_gpu_fast_backend()
    if gpu is None:
        return

    points, gamma0, gamma1, gammadash0, gammadash1 = (
        _build_grouped_biot_savart_device_geometry(gpu)
    )
    host_scalar_currents: tuple[np.float64, np.float64] = (
        np.float64(1.25),
        np.float64(-0.75),
    )

    coil_spec = grouped_coil_set_spec_from_lists(
        (gamma0, gamma1),
        (gammadash0, gammadash1),
        host_scalar_currents,
    )
    magnetic_field = grouped_biot_savart_B_from_spec(points, coil_spec)

    assert magnetic_field.shape == (4, 3)
    assert np.all(np.isfinite(np.asarray(jax.device_get(magnetic_field))))


def _run_grouped_biot_savart_host_spec_vjp_case() -> None:
    gpu = _configure_strict_gpu_fast_backend()
    if gpu is None:
        return

    points = jax.device_put(
        np.linspace(0.0, 1.0, 4 * 3, dtype=np.float64).reshape(4, 3),
        device=gpu,
    )
    gamma = np.linspace(0.2, 0.8, 8 * 3, dtype=np.float64).reshape(8, 3)
    gammadash = np.full((8, 3), 0.1, dtype=np.float64)
    current = np.float64(1.25)
    coil_spec = grouped_coil_set_spec_from_lists([gamma], [gammadash], [current])

    def objective(eval_points: jax.Array) -> jax.Array:
        return jnp.sum(grouped_biot_savart_B_from_spec(eval_points, coil_spec))

    value, pullback = jax.vjp(objective, points)
    grad = pullback(jax.device_put(np.float64(1.0), device=gpu))[0]

    assert np.isfinite(float(jax.device_get(value)))
    assert grad.shape == points.shape
    assert bool(jax.device_get(jnp.all(jnp.isfinite(grad))))


def _run_mutable_objective_state_case() -> None:
    if not _configure_strict_cpu_parity_backend():
        return

    objective = _ShiftedQuadratic([0.0, 0.0])
    x0 = jnp.asarray(np.array([2.0, -1.0], dtype=np.float64))

    first = jax_minimize(objective, x0, method="bfgs-ondevice", maxiter=20)
    objective.target = np.asarray([1.5, -0.5], dtype=np.float64)
    second = jax_minimize(objective, x0, method="bfgs-ondevice", maxiter=20)

    np.testing.assert_allclose(
        np.asarray(first.x),
        np.asarray([0.0, 0.0]),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(second.x),
        np.asarray([1.5, -0.5]),
        atol=1e-6,
    )


def _parse_optimizer_method(method: str) -> OptimizerMethod:
    if method == "lbfgs-ondevice":
        return cast(OptimizerMethod, method)
    if method == "bfgs-ondevice":
        return cast(OptimizerMethod, method)
    raise ValueError(f"unsupported optimizer method {method!r}")


def _parse_legacy_curve_objective_value_case(
    case: str,
) -> LegacyCurveObjectiveValueCase:
    if case == "curve-length":
        return cast(LegacyCurveObjectiveValueCase, case)
    if case == "lp-curve-curvature":
        return cast(LegacyCurveObjectiveValueCase, case)
    if case == "curve-curve-distance":
        return cast(LegacyCurveObjectiveValueCase, case)
    if case == "curve-surface-distance":
        return cast(LegacyCurveObjectiveValueCase, case)
    if case == "lp-curve-curvature-barrier":
        return cast(LegacyCurveObjectiveValueCase, case)
    if case == "lp-curve-torsion":
        return cast(LegacyCurveObjectiveValueCase, case)
    if case == "framed-curve-twist":
        return cast(LegacyCurveObjectiveValueCase, case)
    raise ValueError(f"unsupported legacy curve objective value case {case!r}")


def _parse_legacy_curve_objective_gradient_case(
    case: str,
) -> LegacyCurveObjectiveGradientCase:
    if case == "lp-curve-curvature-barrier":
        return cast(LegacyCurveObjectiveGradientCase, case)
    if case == "lp-curve-curvature":
        return cast(LegacyCurveObjectiveGradientCase, case)
    if case == "curve-curve-distance":
        return cast(LegacyCurveObjectiveGradientCase, case)
    if case == "curve-surface-distance":
        return cast(LegacyCurveObjectiveGradientCase, case)
    if case == "lp-curve-torsion":
        return cast(LegacyCurveObjectiveGradientCase, case)
    if case == "framed-curve-twist":
        return cast(LegacyCurveObjectiveGradientCase, case)
    raise ValueError(f"unsupported legacy curve objective gradient case {case!r}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Process-isolated JAX runtime regression cases.",
    )
    subparsers = parser.add_subparsers(dest="case", required=True)

    compile_count = subparsers.add_parser("compile-count")
    compile_count.add_argument(
        "method",
        choices=("lbfgs-ondevice", "bfgs-ondevice"),
    )
    subparsers.add_parser("biot-savart-point-chunking")
    subparsers.add_parser("target-compile-count")
    subparsers.add_parser("stage2-target-compile-count")
    subparsers.add_parser("mutable-objective-state")
    subparsers.add_parser("grouped-gpu-spec-eval")
    subparsers.add_parser("grouped-explicit-point-sharding")
    subparsers.add_parser("pairwise-penalty-explicit-row-sharding")
    subparsers.add_parser("shifted-grid-axis-sample")
    subparsers.add_parser("gamma-2d-eager-host-constants")
    subparsers.add_parser("closed-curve-self-intersection-summary")
    subparsers.add_parser("surface-xyztensorfourier-gamma-from-dofs")
    subparsers.add_parser("coil-symmetry-spec-identity-default")
    subparsers.add_parser("pairwise-curve-penalty-pure-functions")
    subparsers.add_parser("surfacerzfourier-spec-defaults")
    subparsers.add_parser("surface-rzfourier-gamma-from-spec")
    subparsers.add_parser("surface-rzfourier-normal-from-spec")
    legacy_curve_objective_value = subparsers.add_parser(
        "legacy-curve-objective-value"
    )
    legacy_curve_objective_value.add_argument(
        "objective",
        choices=(
            "curve-length",
            "lp-curve-curvature",
            "curve-curve-distance",
            "curve-surface-distance",
            "lp-curve-curvature-barrier",
            "lp-curve-torsion",
            "framed-curve-twist",
        ),
    )
    legacy_curve_objective_gradient = subparsers.add_parser(
        "legacy-curve-objective-gradient"
    )
    legacy_curve_objective_gradient.add_argument(
        "objective",
        choices=(
            "lp-curve-curvature-barrier",
            "lp-curve-curvature",
            "curve-curve-distance",
            "curve-surface-distance",
            "lp-curve-torsion",
            "framed-curve-twist",
        ),
    )
    subparsers.add_parser("curvecwsfouriercpp-init")
    subparsers.add_parser("curvecwsfouriercpp-curve-length-gradient")
    subparsers.add_parser("curvecwsfouriercpp-curve-distance-gradient")
    subparsers.add_parser("curveperturbed-init")
    subparsers.add_parser("stage2-target-objective-host-closure-constants")
    subparsers.add_parser("stage2-target-objective-ondevice-entry")
    subparsers.add_parser("grouped-gpu-current-arrays")
    subparsers.add_parser("grouped-host-scalar-currents")
    subparsers.add_parser("grouped-host-spec-vjp")

    args = parser.parse_args(argv)

    if args.case == "compile-count":
        _run_compile_count_case(_parse_optimizer_method(args.method))
        return 0
    if args.case == "biot-savart-point-chunking":
        _run_biot_savart_point_chunking_case()
        return 0
    if args.case == "target-compile-count":
        _run_target_compile_count_case()
        return 0
    if args.case == "stage2-target-compile-count":
        _run_stage2_target_compile_count_case()
        return 0
    if args.case == "mutable-objective-state":
        _run_mutable_objective_state_case()
        return 0
    if args.case == "grouped-gpu-spec-eval":
        _run_grouped_biot_savart_gpu_spec_eval_case()
        return 0
    if args.case == "grouped-explicit-point-sharding":
        _run_grouped_biot_savart_explicit_point_sharding_case()
        return 0
    if args.case == "pairwise-penalty-explicit-row-sharding":
        _run_pairwise_penalty_explicit_row_sharding_case()
        return 0
    if args.case == "shifted-grid-axis-sample":
        _run_shifted_grid_axis_sample_case()
        return 0
    if args.case == "gamma-2d-eager-host-constants":
        _run_gamma_2d_eager_host_constants_case()
        return 0
    if args.case == "closed-curve-self-intersection-summary":
        _run_closed_curve_self_intersection_summary_case()
        return 0
    if args.case == "surface-xyztensorfourier-gamma-from-dofs":
        _run_surface_xyztensorfourier_gamma_from_dofs_case()
        return 0
    if args.case == "coil-symmetry-spec-identity-default":
        _run_coil_symmetry_spec_identity_default_case()
        return 0
    if args.case == "pairwise-curve-penalty-pure-functions":
        _run_pairwise_curve_penalty_pure_functions_case()
        return 0
    if args.case == "surfacerzfourier-spec-defaults":
        _run_surfacerzfourier_spec_defaults_case()
        return 0
    if args.case == "surface-rzfourier-gamma-from-spec":
        _run_surface_rzfourier_gamma_from_spec_case()
        return 0
    if args.case == "surface-rzfourier-normal-from-spec":
        _run_surface_rzfourier_normal_from_spec_case()
        return 0
    if args.case == "legacy-curve-objective-value":
        _run_legacy_curve_objective_value_case(
            _parse_legacy_curve_objective_value_case(args.objective)
        )
        return 0
    if args.case == "legacy-curve-objective-gradient":
        _run_legacy_curve_objective_gradient_case(
            _parse_legacy_curve_objective_gradient_case(args.objective)
        )
        return 0
    if args.case == "curvecwsfouriercpp-init":
        _run_curvecwsfouriercpp_init_case()
        return 0
    if args.case == "curvecwsfouriercpp-curve-length-gradient":
        _run_curvecwsfouriercpp_curve_length_gradient_case()
        return 0
    if args.case == "curvecwsfouriercpp-curve-distance-gradient":
        _run_curvecwsfouriercpp_curve_distance_gradient_case()
        return 0
    if args.case == "curveperturbed-init":
        _run_curveperturbed_init_case()
        return 0
    if args.case == "stage2-target-objective-host-closure-constants":
        _run_stage2_target_objective_host_closure_constants_case()
        return 0
    if args.case == "stage2-target-objective-ondevice-entry":
        _run_stage2_target_objective_ondevice_entry_case()
        return 0
    if args.case == "grouped-gpu-current-arrays":
        _run_grouped_biot_savart_gpu_current_arrays_case()
        return 0
    if args.case == "grouped-host-scalar-currents":
        _run_grouped_biot_savart_host_scalar_currents_case()
        return 0
    if args.case == "grouped-host-spec-vjp":
        _run_grouped_biot_savart_host_spec_vjp_case()
        return 0
    raise ValueError(f"unsupported subprocess case {args.case!r}")


if __name__ == "__main__":
    raise SystemExit(main())
