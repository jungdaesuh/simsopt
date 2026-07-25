from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Callable, Sequence
from pathlib import Path
import sys
from typing import Literal, TypedDict, cast

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_DIR = _REPO_ROOT / "src"
_LOCAL_SIMSOPT_IMPORT_PATHS = (_REPO_ROOT, _SRC_DIR)
_STRICT_CPU_PARITY_SKIP_REASON = (
    "strict CPU parity backend unavailable: private optimizer runtime not supported"
)
_STRICT_GPU_FAST_SKIP_REASON = (
    "strict GPU fast backend unavailable: no GPU device detected"
)
_MUTABLE_OBJECTIVE_SMOKE_MAXITER = 5


class SkippedCase(RuntimeError):
    pass


class _CompileCountPayload(TypedDict, total=False):
    compile_count: int
    stepwise_compile_count: int
    monolithic_compile_count: int
    counts_by_fragment: dict[str, int]


def _skip_case(reason: str) -> None:
    raise SkippedCase(reason)


def _prepend_sys_path(path: Path) -> None:
    """Move one path to the front of ``sys.path`` without duplicates."""
    path_str = str(path)
    sys.path[:] = [entry for entry in sys.path if entry != path_str]
    sys.path.insert(0, path_str)


def _prefer_local_simsopt_source_tree() -> None:
    """Bootstrap the local checkout before importing ``simsopt`` modules."""
    for path in _LOCAL_SIMSOPT_IMPORT_PATHS:
        _prepend_sys_path(path)
    from repo_bootstrap import bootstrap_local_simsopt

    bootstrap_local_simsopt(_SRC_DIR)


_prefer_local_simsopt_source_tree()

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

import simsopt_jax.config as simsopt_config  # type: ignore[import-untyped]
from simsopt.field import BiotSavart, Coil, Current  # type: ignore[import-untyped]
from simsopt_jax_adapters.field.biotsavart_backend import (  # type: ignore[import-untyped]
    BiotSavartFieldPullback,
    BiotSavartJAX,
)
from simsopt.geo import SurfaceRZFourier, SurfaceXYZTensorFourier  # type: ignore[import-untyped]
from simsopt_jax_adapters.geo.boozer_surface import (  # type: ignore[import-untyped]
    _boozer_penalty_objective,
    _surface_sample_z,
)
from simsopt_jax.core.curve_kernels import gamma_2d  # type: ignore[import-untyped]
from simsopt_jax.geo._pairwise_reductions import (  # type: ignore[import-untyped]
    pairwise_min_distance_pure,
)
from simsopt_jax_adapters.geo.curve_objectives import (  # type: ignore[import-untyped]
    cc_distance_pure,
    cs_distance_pure,
)
from simsopt.geo.curvexyzfourier import CurveXYZFourier  # type: ignore[import-untyped]
from simsopt_jax.geo.optimizers import optimizer as _optimizer_module  # type: ignore[import-untyped]
from simsopt_jax.geo.optimizers.optimizer import (  # type: ignore[import-untyped]
    _mark_cacheable_jit_value_and_grad,
    private_optimizer_runtime_is_supported,
    target_minimize,
)
from simsopt_jax.solve import (  # type: ignore[import-untyped]
    Driver,
    SimsoptBFGSOptions,
    SimsoptLBFGSBOptions,
)
from simsopt_jax.solve.dispatch import minimize  # type: ignore[import-untyped]
from simsopt_jax.core.biotsavart import (  # type: ignore[import-untyped]
    biot_savart_A,
    biot_savart_B,
    biot_savart_B_and_dB,
    biot_savart_d2A_by_dXdX,
    biot_savart_dA_by_dX,
    biot_savart_dB_by_dX,
)
from simsopt_jax.core.curve_geometry import (  # type: ignore[import-untyped]
    closed_curve_self_intersection_summary,
)
from simsopt_jax.core.field import (  # type: ignore[import-untyped]
    biot_savart_B_vjp_maybe_collective,
    grouped_field_inputs_from_spec,
    grouped_biot_savart_A_from_spec,
    grouped_biot_savart_A_from_inputs,
    grouped_biot_savart_B_and_dB_from_spec,
    grouped_biot_savart_B_from_spec,
    grouped_biot_savart_d2A_by_dXdX_from_spec,
    grouped_biot_savart_dA_by_dX_from_spec,
    grouped_biot_savart_dA_by_dX_from_inputs,
    grouped_biot_savart_dB_by_dX_from_spec,
    grouped_biot_savart_dB_by_dX_from_inputs,
    grouped_coil_set_spec_from_lists,
    grouped_field_sharding_summary,
)
from simsopt_jax.core.surface_rzfourier import (  # type: ignore[import-untyped]
    surface_rz_fourier_gamma_from_spec,
    surface_rz_fourier_normal_from_spec,
    surface_rz_fourier_spec_from_dofs,
)
from simsopt_jax.core.specs import (  # type: ignore[import-untyped]
    GroupedCoilSetSpec,
    make_coil_symmetry_spec,
)
from simsopt_jax.objectives.integral_bdotn import (  # type: ignore[import-untyped]
    integral_BdotN,
    integral_BdotN_surface_sharded,
    integral_BdotN_sharding_summary,
)
from simsopt_jax.core.sharding import (  # type: ignore[import-untyped]
    maybe_shard_seed_batch_inputs,
    maybe_shard_surface_quadrature_inputs,
    seed_batch_sharding_config,
    seed_batch_sharding_summary,
    surface_quadrature_sharding_config,
)
from simsopt_jax.geo.surface_fourier import (  # type: ignore[import-untyped]
    stellsym_scatter_indices,
    surface_gamma_from_dofs,
)
from simsopt_jax.numerical_policy import (  # type: ignore[import-untyped]
    MIXED_DENSE_IR_CERTIFICATE_KEY_WORD_MAX,
    CertificateProbeAuthority,
    CertificateProbeKeyData,
)
from simsopt_jax.runtime.host_boundary import (  # type: ignore[import-untyped]
    runtime_certificate_probe_key,
)

OptimizerMethod = Literal["lbfgs-ondevice", "bfgs-ondevice"]


def _run_certificate_probe_key_x64_disabled_case() -> None:
    assert jax.config.jax_enable_x64 is False
    authority = CertificateProbeAuthority(
        source="supplied_replay",
        key_data=CertificateProbeKeyData(
            MIXED_DENSE_IR_CERTIFICATE_KEY_WORD_MAX,
            MIXED_DENSE_IR_CERTIFICATE_KEY_WORD_MAX - 1,
        ),
    )
    restored = CertificateProbeAuthority.from_json(authority.as_json())
    key = runtime_certificate_probe_key(restored.key_data)
    key_data = jax.random.key_data(key)
    assert key_data.dtype == jnp.dtype(jnp.uint32)
    np.testing.assert_array_equal(
        np.asarray(jax.device_get(key_data)),
        np.asarray(restored.key_data.words, dtype=np.uint32),
    )


def _solve_jax_driver_for_optimizer_method(method: OptimizerMethod) -> Driver:
    if method == "lbfgs-ondevice":
        return Driver.SIMSOPT_LBFGSB
    if method == "bfgs-ondevice":
        return Driver.SIMSOPT_BFGS
    raise ValueError(f"Unknown optimizer method {method!r}.")


def _solve_jax_options_for_optimizer_method(
    method: OptimizerMethod,
    *,
    maxiter: int,
) -> SimsoptLBFGSBOptions | SimsoptBFGSOptions:
    if method == "lbfgs-ondevice":
        return SimsoptLBFGSBOptions(maxiter=maxiter)
    if method == "bfgs-ondevice":
        return SimsoptBFGSOptions(maxiter=maxiter)
    raise ValueError(f"Unknown optimizer method {method!r}.")


def _configure_strict_cpu_parity_backend() -> bool:
    simsopt_config.set_backend(
        "jax_cpu_parity",
        strict=True,
        transfer_guard="disallow",
    )
    return private_optimizer_runtime_is_supported(jax.__version__)


def _configure_transfer_guard_cpu_parity_backend() -> None:
    """Non-strict backend with transfer_guard=disallow only.

    Used by cases that exercise CPU geometry objects where fallback is expected
    behaviour.
    """
    simsopt_config.set_backend(
        "jax_cpu_parity",
        transfer_guard="disallow",
    )


class _CompileCounter(logging.Handler):
    def __init__(self, fragments: tuple[str, ...] = ("_run_solver)",)) -> None:
        super().__init__()
        self.fragments = fragments
        self.count = 0
        self.counts_by_fragment = {fragment: 0 for fragment in fragments}

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if "Compiling jit(" not in message:
            return
        for fragment in self.fragments:
            if fragment in message:
                self.count += 1
                self.counts_by_fragment[fragment] += 1
                return


def _assert_solver_compile_count(
    run_once,
    *,
    fragments: tuple[str, ...],
    expected_compile_count: int,
) -> dict[str, int]:
    logger = logging.getLogger("jax")
    old_level = logger.level
    handler = _CompileCounter(fragments)
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    try:
        jax.clear_caches()
        with jax.log_compiles(True):
            for _ in range(3):
                run_once()
        assert handler.count == expected_compile_count, handler.count
        return dict(handler.counts_by_fragment)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)


def _compile_count_payload(
    counts_by_fragment: dict[str, int],
) -> _CompileCountPayload:
    return {
        "compile_count": sum(counts_by_fragment.values()),
        "stepwise_compile_count": sum(
            counts_by_fragment[fragment]
            for fragment in (
                "lbfgs_private_value_and_grad)",
                "lbfgs_private_initial_state_solver)",
                "lbfgs_private_macro_step_solver)",
                "lbfgs_private_result_payload_solver)",
            )
            if fragment in counts_by_fragment
        ),
        "monolithic_compile_count": counts_by_fragment.get(
            "lbfgs_private_monolithic_mainlb_solver)", 0
        ),
    }


def _assert_run_solver_compiles_once(run_once) -> _CompileCountPayload:
    counts_by_fragment = _assert_solver_compile_count(
        run_once,
        fragments=("_run_solver)",),
        expected_compile_count=1,
    )
    return {
        "compile_count": counts_by_fragment["_run_solver)"],
    }


def _assert_lbfgs_private_step_kernels_compile_once_each(
    run_once,
) -> _CompileCountPayload:
    counts_by_fragment = _assert_solver_compile_count(
        run_once,
        fragments=(
            "lbfgs_private_value_and_grad)",
            "lbfgs_private_initial_state_solver)",
            "lbfgs_private_macro_step_solver)",
            "lbfgs_private_result_payload_solver)",
            "lbfgs_private_monolithic_mainlb_solver)",
        ),
        expected_compile_count=5,
    )
    payload = _compile_count_payload(counts_by_fragment)
    assert payload["monolithic_compile_count"] == 0, payload
    return {
        **payload,
        "counts_by_fragment": counts_by_fragment,
    }


def _run_compile_count_case(method: OptimizerMethod) -> None:
    if not _configure_strict_cpu_parity_backend():
        _skip_case(_STRICT_CPU_PARITY_SKIP_REASON)
        return

    half = jax.device_put(np.asarray(0.5, dtype=np.float64))

    def quad(x: jax.Array) -> jax.Array:
        vector = jnp.asarray(x, dtype=jnp.float64)
        return half * jnp.dot(vector, vector)

    cacheable_quad = _mark_cacheable_jit_value_and_grad(jax.value_and_grad(quad))
    x0 = jnp.asarray(np.array([1.0, -2.0], dtype=np.float64))

    def run_once() -> None:
        result = minimize(
            cacheable_quad,
            x0,
            driver=_solve_jax_driver_for_optimizer_method(method),
            options=_solve_jax_options_for_optimizer_method(method, maxiter=5),
        )
        assert result.success is True

    if method == "lbfgs-ondevice":
        compile_payload = _assert_lbfgs_private_step_kernels_compile_once_each(run_once)
    else:
        compile_payload = _assert_run_solver_compiles_once(run_once)
    print(
        json.dumps(
            {
                "case": "compile-count",
                "method": str(method),
                "run_count": 3,
                **compile_payload,
            },
            sort_keys=True,
        )
    )


def _run_biot_savart_point_chunking_case() -> None:
    _configure_strict_cpu_parity_backend()

    points = jax.device_put(np.arange(257 * 3, dtype=np.float64).reshape(257, 3) * 1e-3)
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
        _skip_case(_STRICT_CPU_PARITY_SKIP_REASON)
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

    compile_payload = _assert_lbfgs_private_step_kernels_compile_once_each(run_once)
    print(
        json.dumps(
            {
                "case": "target-compile-count",
                "method": "lbfgs-ondevice",
                "run_count": 3,
                "value_and_grad": True,
                **compile_payload,
            },
            sort_keys=True,
        )
    )


def _assert_implicit_host_transfer_rejected(
    fn,
    *args,
    case_name: str,
    **kwargs,
) -> None:
    try:
        fn(*args, **kwargs)
    except RuntimeError as exc:
        message = str(exc)
        normalized_message = message.lower()
        assert "transfer" in normalized_message and (
            "guard" in normalized_message or "disallow" in normalized_message
        ), f"{case_name} raised the wrong transfer-guard error: {message}"
    else:
        raise AssertionError(
            f"{case_name} should reject implicit host input under transfer guard"
        )


def _host_scalar_float64(value) -> float:
    return float(jax.device_get(value))


def _host_array_float64(value) -> np.ndarray:
    return np.asarray(jax.device_get(value), dtype=np.float64)


def _assert_finite_scalar(value, *, label: str | None = None) -> float:
    scalar = _host_scalar_float64(value)
    if label is None:
        assert np.isfinite(scalar)
    else:
        assert np.isfinite(scalar), label
    return scalar


def _assert_finite_scalar_matches_host(device_value, host_value) -> None:
    device_scalar = _assert_finite_scalar(device_value)
    host_scalar = _host_scalar_float64(host_value)
    np.testing.assert_allclose(host_scalar, device_scalar)


def _assert_finite_array(value, *, expected_shape=None) -> np.ndarray:
    array = _host_array_float64(value)
    if expected_shape is not None:
        assert array.shape == expected_shape
    assert np.all(np.isfinite(array))
    return array


class _ShiftedQuadratic:
    def __init__(self, target: Sequence[float]) -> None:
        self.target = np.asarray(tuple(target), dtype=np.float64)
        self.half = jax.device_put(np.asarray(0.5, dtype=np.float64))

    def __call__(self, x: jax.Array) -> jax.Array:
        vector = jnp.asarray(x, dtype=jnp.float64)
        target = jnp.asarray(self.target, dtype=jnp.float64)
        diff = vector - target
        return self.half * jnp.dot(diff, diff)


class _StructuredShiftedQuadraticValueAndGrad:
    def __init__(self, target: Sequence[float]) -> None:
        self.target = np.asarray(tuple(target), dtype=np.float64)
        self.half = jax.device_put(np.asarray(0.5, dtype=np.float64))

    def __call__(
        self, state: dict[str, jax.Array]
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        vector = jnp.asarray(state["x"], dtype=jnp.float64)
        target = jnp.asarray(self.target, dtype=jnp.float64)
        diff = vector - target
        value = self.half * jnp.dot(diff, diff)
        return value, {"x": diff}


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


def _run_grouped_biot_savart_gpu_spec_eval_case() -> None:
    gpu = _configure_strict_gpu_fast_backend()
    if gpu is None:
        _skip_case(_STRICT_GPU_FAST_SKIP_REASON)
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


def _collective_circular_coil(
    index: int,
    *,
    nquad: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    phi = np.linspace(0.0, 2.0 * np.pi, nquad, endpoint=False)
    radius = 1.0 + 0.05 * index
    gamma = np.stack(
        (
            radius * np.cos(phi),
            radius * np.sin(phi),
            np.full_like(phi, 0.03 * index),
        ),
        axis=1,
    )
    gammadash = np.stack(
        (
            -2.0 * np.pi * radius * np.sin(phi),
            2.0 * np.pi * radius * np.cos(phi),
            np.zeros_like(phi),
        ),
        axis=1,
    )
    return (
        jnp.asarray(gamma, dtype=jnp.float64),
        jnp.asarray(gammadash, dtype=jnp.float64),
        jnp.asarray(1.0 + 0.2 * index, dtype=jnp.float64),
    )


def _build_collective_circular_coils() -> tuple[
    list[jax.Array],
    list[jax.Array],
    list[jax.Array],
]:
    coils = [_collective_circular_coil(index, nquad=16) for index in range(5)]
    gammas, gammadashs, currents = zip(*coils)
    return list(gammas), list(gammadashs), list(currents)


def _collective_curve_coil(index: int) -> Coil:
    curve = CurveXYZFourier(16, 1)
    curve.x = np.array(
        [
            0.2 + 0.01 * index,
            0.0,
            0.0,
            1.0 + 0.03 * index,
            0.0,
            0.0,
            0.0,
            1.0 + 0.02 * index,
            0.0,
        ],
        dtype=np.float64,
    )
    return Coil(curve, Current(1.0 + 0.1 * index))


def _build_collective_curve_coils() -> list[Coil]:
    return [_collective_curve_coil(index) for index in range(5)]


def _assert_allclose_to_reference(value, reference) -> None:
    np.testing.assert_allclose(
        np.asarray(value),
        np.asarray(reference),
        rtol=1e-12,
        atol=1e-14,
    )


def _block_until_ready_tree(tree) -> None:
    leaves = jax.tree.leaves(tree)
    for leaf in leaves:
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def _assert_mixed_quadrature_collective_parity(points: jax.Array) -> None:
    group_16 = [_collective_circular_coil(index, nquad=16) for index in range(3)]
    group_12 = [_collective_circular_coil(index + 3, nquad=12) for index in range(2)]
    gammas_16, gammadashs_16, currents_16 = zip(*group_16)
    gammas_12, gammadashs_12, currents_12 = zip(*group_12)
    coil_spec = grouped_coil_set_spec_from_lists(
        list(gammas_16 + gammas_12),
        list(gammadashs_16 + gammadashs_12),
        list(currents_16 + currents_12),
    )
    dense_reference = biot_savart_B(
        points,
        jnp.stack(gammas_16),
        jnp.stack(gammadashs_16),
        jnp.stack(currents_16),
    ) + biot_savart_B(
        points,
        jnp.stack(gammas_12),
        jnp.stack(gammadashs_12),
        jnp.stack(currents_12),
    )

    _assert_allclose_to_reference(
        grouped_biot_savart_B_from_spec(points, coil_spec),
        dense_reference,
    )
    assert grouped_field_sharding_summary(points, coil_spec)["field_collective"] is True


def _assert_pullback_payloads_allclose(
    observed: BiotSavartFieldPullback,
    expected: BiotSavartFieldPullback,
) -> None:
    assert len(observed.d_coil_arrays) == len(expected.d_coil_arrays)
    assert observed.coil_indices == expected.coil_indices
    for observed_group, expected_group in zip(
        observed.d_coil_arrays,
        expected.d_coil_arrays,
    ):
        assert len(observed_group) == len(expected_group)
        for observed_leaf, expected_leaf in zip(observed_group, expected_group):
            _assert_allclose_to_reference(observed_leaf, expected_leaf)


def _assert_native_pullback_matches_grouped_forward(
    points: jax.Array,
    coil_spec: GroupedCoilSetSpec,
    native_pullback: BiotSavartFieldPullback,
    cotangent: jax.Array,
    grouped_forward: Callable[[jax.Array, object], jax.Array],
) -> None:
    grouped_inputs = grouped_field_inputs_from_spec(coil_spec)
    assert native_pullback.coil_indices == coil_spec.coil_index_lists()
    _assert_native_pullback_matches_grouped_inputs(
        points,
        native_pullback,
        cotangent,
        grouped_forward,
        grouped_inputs,
    )


def _assert_native_pullback_matches_grouped_inputs(
    points: jax.Array,
    native_pullback: BiotSavartFieldPullback,
    cotangent: jax.Array,
    grouped_forward: Callable[[jax.Array, object], jax.Array],
    grouped_inputs: tuple[tuple[jax.Array, jax.Array, jax.Array], ...],
) -> None:
    assert len(native_pullback.d_coil_arrays) == len(grouped_inputs)

    _, direct_pullback = jax.vjp(
        lambda grouped_inputs_arg: grouped_forward(points, grouped_inputs_arg),
        grouped_inputs,
    )
    direct_groups = direct_pullback(cotangent)[0]
    assert len(native_pullback.d_coil_arrays) == len(direct_groups)
    for native_group, direct_group in zip(
        native_pullback.d_coil_arrays,
        direct_groups,
    ):
        assert len(native_group) == len(direct_group)
        for native_leaf, direct_leaf in zip(native_group, direct_group):
            _assert_allclose_to_reference(native_leaf, direct_leaf)


def _assert_native_pullback_matches_expected_inputs(
    points: jax.Array,
    native_pullback: BiotSavartFieldPullback,
    expected_indices: tuple[tuple[int, ...], ...],
    cotangent: jax.Array,
    grouped_forward: Callable[[jax.Array, object], jax.Array],
    grouped_inputs: tuple[tuple[jax.Array, jax.Array, jax.Array], ...],
) -> None:
    assert native_pullback.coil_indices == expected_indices
    _assert_native_pullback_matches_grouped_inputs(
        points,
        native_pullback,
        cotangent,
        grouped_forward,
        grouped_inputs,
    )


def _assert_native_pullback_projects_to_public(
    bs_jax: BiotSavartJAX,
    pullback: BiotSavartFieldPullback,
    public: Callable[[object], object],
    coils: list[Coil],
) -> None:
    projected = bs_jax.coil_cotangents_to_derivative(
        pullback.d_coil_arrays,
        pullback.coil_indices,
    )
    for coil in coils:
        _assert_allclose_to_reference(projected(coil), public(coil))


def _compiled_pullback_hlo(
    pullback: Callable[..., BiotSavartFieldPullback],
    *cotangents: jax.Array,
) -> str:
    return jax.jit(pullback).lower(*cotangents).compile().as_text().lower()


def _assert_biot_savart_jax_native_pullback_collective_path(
    points: jax.Array,
) -> None:
    coils = _build_collective_curve_coils()
    bs_jax = BiotSavartJAX(coils)
    bs_jax.set_points(np.asarray(points, dtype=np.float64))
    cotangent = jnp.asarray(bs_jax.B())
    coil_spec = bs_jax.coil_set_spec()
    summary = grouped_field_sharding_summary(points, coil_spec)

    assert summary["field_collective"] is True
    assert summary["strategy"] == "coil_groups"
    assert summary["collective_axis"] == "coil"

    native_B = bs_jax.B_pullback_native(cotangent)
    assert native_B.coil_indices == ((0, 1, 2, 3, 4),)
    assert len(native_B.d_coil_arrays) == 1
    native_group = native_B.d_coil_arrays[0]
    grouped_inputs = grouped_field_inputs_from_spec(coil_spec)
    assert len(grouped_inputs) == 1
    direct_collective_pullback = biot_savart_B_vjp_maybe_collective(
        points,
        cotangent,
        *grouped_inputs[0],
    )
    assert len(native_group) == len(direct_collective_pullback)
    for native_leaf, direct_leaf in zip(
        native_group,
        direct_collective_pullback,
    ):
        _assert_allclose_to_reference(native_leaf, direct_leaf)

    public_B = bs_jax.B_vjp(cotangent)
    _assert_native_pullback_projects_to_public(bs_jax, native_B, public_B, coils)
    dofs_gradient = bs_jax.coil_cotangents_to_dofs_gradient(
        native_B.d_coil_arrays,
        native_B.coil_indices,
    )
    _assert_allclose_to_reference(dofs_gradient, public_B(bs_jax))

    A_cotangent = jnp.asarray(bs_jax.A())
    native_A = bs_jax.A_pullback_native(A_cotangent)
    _assert_native_pullback_matches_grouped_forward(
        points,
        coil_spec,
        native_A,
        A_cotangent,
        grouped_biot_savart_A_from_inputs,
    )
    _assert_native_pullback_projects_to_public(
        bs_jax,
        native_A,
        bs_jax.A_vjp(A_cotangent),
        coils,
    )

    dA_cotangent = jnp.asarray(bs_jax.dA_by_dX())
    native_dA = bs_jax.dA_by_dX_pullback_native(dA_cotangent)
    _assert_native_pullback_matches_grouped_forward(
        points,
        coil_spec,
        native_dA,
        dA_cotangent,
        grouped_biot_savart_dA_by_dX_from_inputs,
    )
    _, public_dA = bs_jax.A_and_dA_vjp(jnp.zeros_like(A_cotangent), dA_cotangent)
    _assert_native_pullback_projects_to_public(bs_jax, native_dA, public_dA, coils)

    dB_cotangent = jnp.asarray(bs_jax.dB_by_dX())
    native_dB = bs_jax.dB_by_dX_pullback_native(dB_cotangent)
    _assert_native_pullback_matches_grouped_forward(
        points,
        coil_spec,
        native_dB,
        dB_cotangent,
        grouped_biot_savart_dB_by_dX_from_inputs,
    )
    _, public_dB = bs_jax.B_and_dB_vjp(jnp.zeros_like(cotangent), dB_cotangent)
    _assert_native_pullback_projects_to_public(bs_jax, native_dB, public_dB, coils)

    pair_A, pair_dA = bs_jax.A_and_dA_pullback_native(A_cotangent, dA_cotangent)
    _assert_pullback_payloads_allclose(pair_A, native_A)
    _assert_pullback_payloads_allclose(pair_dA, native_dA)
    pair_B, pair_dB = bs_jax.B_and_dB_pullback_native(cotangent, dB_cotangent)
    _assert_pullback_payloads_allclose(pair_B, native_B)
    _assert_pullback_payloads_allclose(pair_dB, native_dB)

    native_pullback_hlos = (
        _compiled_pullback_hlo(lambda v: bs_jax.B_pullback_native(v), cotangent),
        _compiled_pullback_hlo(lambda v: bs_jax.A_pullback_native(v), A_cotangent),
        _compiled_pullback_hlo(
            lambda v: bs_jax.dA_by_dX_pullback_native(v),
            dA_cotangent,
        ),
        _compiled_pullback_hlo(
            lambda v: bs_jax.dB_by_dX_pullback_native(v),
            dB_cotangent,
        ),
        _compiled_pullback_hlo(
            lambda v, vgrad: bs_jax.A_and_dA_pullback_native(v, vgrad),
            A_cotangent,
            dA_cotangent,
        ),
        _compiled_pullback_hlo(
            lambda v, vgrad: bs_jax.B_and_dB_pullback_native(v, vgrad),
            cotangent,
            dB_cotangent,
        ),
    )
    for compiled_hlo in native_pullback_hlos:
        assert "all-gather" in compiled_hlo


def _assert_biot_savart_jax_native_pullbacks_skip_fixed_coils(
    points: jax.Array,
) -> None:
    coils = _build_collective_curve_coils()
    coils[0].curve.fix_all()
    coils[0].current.fix_all()
    bs_jax = BiotSavartJAX(coils)
    bs_jax.set_points(np.asarray(points, dtype=np.float64))

    coil_group = bs_jax.coil_set_spec().groups[0]
    free_grouped_inputs = (
        (
            coil_group.gammas[1:],
            coil_group.gammadashs[1:],
            coil_group.currents[1:],
        ),
    )
    free_coils = coils[1:]
    expected_indices = ((1, 2, 3, 4),)

    B_cotangent = jnp.asarray(bs_jax.B())
    native_B = bs_jax.B_pullback_native(B_cotangent)
    assert native_B.coil_indices == expected_indices
    _assert_native_pullback_projects_to_public(
        bs_jax,
        native_B,
        bs_jax.B_vjp(B_cotangent),
        free_coils,
    )

    A_cotangent = jnp.asarray(bs_jax.A())
    native_A = bs_jax.A_pullback_native(A_cotangent)
    _assert_native_pullback_matches_expected_inputs(
        points,
        native_A,
        expected_indices,
        A_cotangent,
        grouped_biot_savart_A_from_inputs,
        free_grouped_inputs,
    )
    _assert_native_pullback_projects_to_public(
        bs_jax,
        native_A,
        bs_jax.A_vjp(A_cotangent),
        free_coils,
    )

    dA_cotangent = jnp.asarray(bs_jax.dA_by_dX())
    native_dA = bs_jax.dA_by_dX_pullback_native(dA_cotangent)
    _assert_native_pullback_matches_expected_inputs(
        points,
        native_dA,
        expected_indices,
        dA_cotangent,
        grouped_biot_savart_dA_by_dX_from_inputs,
        free_grouped_inputs,
    )

    dB_cotangent = jnp.asarray(bs_jax.dB_by_dX())
    native_dB = bs_jax.dB_by_dX_pullback_native(dB_cotangent)
    _assert_native_pullback_matches_expected_inputs(
        points,
        native_dB,
        expected_indices,
        dB_cotangent,
        grouped_biot_savart_dB_by_dX_from_inputs,
        free_grouped_inputs,
    )


def _assert_grouped_pullback_matches_dense(
    points: jax.Array,
    stacked_gammas: jax.Array,
    stacked_gammadashs: jax.Array,
    stacked_currents: jax.Array,
    cotangent: jax.Array,
    forward: Callable[[jax.Array, jax.Array, jax.Array, jax.Array], jax.Array],
    grouped_forward: Callable[[jax.Array, object], jax.Array],
) -> None:
    coil_arrays = ((stacked_gammas, stacked_gammadashs, stacked_currents),)
    _, dense_pullback = jax.vjp(
        lambda group_gammas, group_gammadashs, group_currents: forward(
            points,
            group_gammas,
            group_gammadashs,
            group_currents,
        ),
        stacked_gammas,
        stacked_gammadashs,
        stacked_currents,
    )
    _, collective_pullback = jax.vjp(
        lambda grouped_inputs: grouped_forward(points, grouped_inputs),
        coil_arrays,
    )
    collective_vjp = collective_pullback(cotangent)[0][0]
    for collective_leaf, dense_leaf in zip(
        collective_vjp,
        dense_pullback(cotangent),
    ):
        _assert_allclose_to_reference(collective_leaf, dense_leaf)


def _assert_collective_field_pullback_parity(
    points: jax.Array,
    stacked_gammas: jax.Array,
    stacked_gammadashs: jax.Array,
    stacked_currents: jax.Array,
) -> None:
    _assert_grouped_pullback_matches_dense(
        points,
        stacked_gammas,
        stacked_gammadashs,
        stacked_currents,
        jnp.linspace(0.3, 1.2, points.size, dtype=jnp.float64).reshape(points.shape),
        biot_savart_A,
        grouped_biot_savart_A_from_inputs,
    )
    _assert_grouped_pullback_matches_dense(
        points,
        stacked_gammas,
        stacked_gammadashs,
        stacked_currents,
        jnp.linspace(
            -0.2,
            0.7,
            points.shape[0] * 9,
            dtype=jnp.float64,
        ).reshape(points.shape[0], 3, 3),
        biot_savart_dA_by_dX,
        grouped_biot_savart_dA_by_dX_from_inputs,
    )


def _run_grouped_biot_savart_coil_collective_case() -> None:
    points = jnp.linspace(0.1, 0.9, 18, dtype=jnp.float64).reshape(6, 3)
    gammas, gammadashs, currents = _build_collective_circular_coils()
    coil_spec = grouped_coil_set_spec_from_lists(gammas, gammadashs, currents)
    stacked_gammas = jnp.stack(gammas)
    stacked_gammadashs = jnp.stack(gammadashs)
    stacked_currents = jnp.stack(currents)

    _assert_allclose_to_reference(
        grouped_biot_savart_B_from_spec(points, coil_spec),
        biot_savart_B(points, stacked_gammas, stacked_gammadashs, stacked_currents),
    )
    _assert_allclose_to_reference(
        grouped_biot_savart_A_from_spec(points, coil_spec),
        biot_savart_A(points, stacked_gammas, stacked_gammadashs, stacked_currents),
    )
    _assert_allclose_to_reference(
        grouped_biot_savart_dA_by_dX_from_spec(points, coil_spec),
        biot_savart_dA_by_dX(
            points,
            stacked_gammas,
            stacked_gammadashs,
            stacked_currents,
        ),
    )
    _assert_allclose_to_reference(
        grouped_biot_savart_d2A_by_dXdX_from_spec(points, coil_spec),
        biot_savart_d2A_by_dXdX(
            points,
            stacked_gammas,
            stacked_gammadashs,
            stacked_currents,
        ),
    )
    _assert_allclose_to_reference(
        grouped_biot_savart_dB_by_dX_from_spec(points, coil_spec),
        biot_savart_dB_by_dX(
            points,
            stacked_gammas,
            stacked_gammadashs,
            stacked_currents,
        ),
    )

    dense_B, dense_dB = biot_savart_B_and_dB(
        points,
        stacked_gammas,
        stacked_gammadashs,
        stacked_currents,
    )
    collective_B, collective_dB = grouped_biot_savart_B_and_dB_from_spec(
        points,
        coil_spec,
    )
    _assert_allclose_to_reference(collective_B, dense_B)
    _assert_allclose_to_reference(collective_dB, dense_dB)

    v = jnp.linspace(0.2, 1.1, points.size, dtype=jnp.float64).reshape(points.shape)
    _, dense_pullback = jax.vjp(
        lambda group_gammas, group_gammadashs, group_currents: biot_savart_B(
            points,
            group_gammas,
            group_gammadashs,
            group_currents,
        ),
        stacked_gammas,
        stacked_gammadashs,
        stacked_currents,
    )
    collective_vjp = biot_savart_B_vjp_maybe_collective(
        points,
        v,
        stacked_gammas,
        stacked_gammadashs,
        stacked_currents,
    )
    for collective_leaf, dense_leaf in zip(collective_vjp, dense_pullback(v)):
        _assert_allclose_to_reference(collective_leaf, dense_leaf)

    _assert_collective_field_pullback_parity(
        points,
        stacked_gammas,
        stacked_gammadashs,
        stacked_currents,
    )
    _assert_mixed_quadrature_collective_parity(points)
    _assert_biot_savart_jax_native_pullback_collective_path(points)
    _assert_biot_savart_jax_native_pullbacks_skip_fixed_coils(points)

    summary = grouped_field_sharding_summary(points, coil_spec)
    assert summary["field_collective"] is True
    assert summary["strategy"] == "coil_groups"
    assert summary["collective_axis"] == "coil"
    assert summary["collective_device_count"] == 4

    lowered = (
        jax.jit(grouped_biot_savart_B_from_spec)
        .lower(
            points,
            coil_spec,
        )
        .as_text()
    )
    assert "all_reduce" in lowered.lower()


def _run_grouped_biot_savart_points_coils_collective_case() -> None:
    """Verify ``points_coils`` 2D sharding lowers to a collective reduction.

    Asserts:
    - field result matches the dense reference,
    - fused sharded ``B_and_dB`` matches the simsoptpp-backed CPU ``B`` and
      ``dB_by_dX`` oracle,
    - sharding summary advertises the 2D mesh axes and active collective,
    - compiled HLO contains an ``all_reduce`` (reduction across coil axis),
    - non-divisible coil counts and mixed quadrature groups still match the
      dense reference.
    """
    with jax.transfer_guard_host_to_device("allow"):
        points_np = np.linspace(0.1, 0.9, 24, dtype=np.float64).reshape(8, 3)
        curve_coils = _build_collective_curve_coils()
        bs_cpp = BiotSavart(curve_coils)
        bs_cpp.set_points(points_np)
        cpp_B = bs_cpp.B()
        cpp_dB = bs_cpp.dB_by_dX()
        points = jax.device_put(
            points_np,
        )
        gammas = [
            jnp.asarray(coil.curve.gamma(), dtype=jnp.float64) for coil in curve_coils
        ]
        gammadashs = [
            jnp.asarray(coil.curve.gammadash(), dtype=jnp.float64)
            for coil in curve_coils
        ]
        currents = [
            jnp.asarray(coil.current.get_value(), dtype=jnp.float64)
            for coil in curve_coils
        ]
        coil_spec = grouped_coil_set_spec_from_lists(gammas, gammadashs, currents)
        stacked_gammas = jnp.stack(gammas)
        stacked_gammadashs = jnp.stack(gammadashs)
        stacked_currents = jnp.stack(currents)

    with jax.transfer_guard("disallow"):
        collective_B = grouped_biot_savart_B_from_spec(points, coil_spec)
        dense_B_direct = biot_savart_B(
            points, stacked_gammas, stacked_gammadashs, stacked_currents
        )
        collective_A = grouped_biot_savart_A_from_spec(points, coil_spec)
        dense_A = biot_savart_A(
            points, stacked_gammas, stacked_gammadashs, stacked_currents
        )
        collective_dB = grouped_biot_savart_dB_by_dX_from_spec(points, coil_spec)
        dense_dB_direct = biot_savart_dB_by_dX(
            points,
            stacked_gammas,
            stacked_gammadashs,
            stacked_currents,
        )
        dense_B, dense_dB = biot_savart_B_and_dB(
            points,
            stacked_gammas,
            stacked_gammadashs,
            stacked_currents,
        )
        collective_B_and_dB = grouped_biot_savart_B_and_dB_from_spec(
            points,
            coil_spec,
        )
        summary = grouped_field_sharding_summary(points, coil_spec)
        lowered = (
            jax.jit(grouped_biot_savart_B_from_spec)
            .lower(
                points,
                coil_spec,
            )
            .as_text()
        )
        _block_until_ready_tree(
            (
                collective_B,
                dense_B_direct,
                collective_A,
                dense_A,
                collective_dB,
                dense_dB_direct,
                dense_B,
                dense_dB,
                collective_B_and_dB,
            )
        )

    _assert_allclose_to_reference(collective_B, dense_B_direct)
    _assert_allclose_to_reference(collective_A, dense_A)
    _assert_allclose_to_reference(collective_dB, dense_dB_direct)
    collective_B, collective_dB = collective_B_and_dB
    _assert_allclose_to_reference(collective_B, dense_B)
    _assert_allclose_to_reference(collective_dB, dense_dB)
    _assert_allclose_to_reference(collective_B, cpp_B)
    _assert_allclose_to_reference(collective_dB, cpp_dB)

    assert summary["field_collective"] is True, summary
    assert summary["strategy"] == "points_coils", summary
    assert summary["coil_axis"] == "coil", summary
    assert summary["point_axis"] is not None, summary
    assert summary["reduced_axis"] == "coil", summary
    assert summary["collective_device_count"] >= 1
    assert summary["point_device_count"] >= 1
    assert (summary["point_device_count"] * summary["collective_device_count"]) == 4, (
        summary
    )
    assert len(summary["mesh_axes"]) == 2, summary

    assert "all_reduce" in lowered.lower(), lowered[:2000]

    with jax.transfer_guard_host_to_device("allow"):
        with jax.transfer_guard_device_to_host("allow"):
            _assert_mixed_quadrature_collective_parity(points)


def _run_grouped_biot_savart_points_coils_non_divisible_case() -> None:
    """Verify ``points_coils`` handles non-divisible point counts via padding.

    Forces ``point_device_count > 1`` (via a 2x2 mesh) and uses 7 points so
    the point axis is not divisible. Asserts the result still matches the
    dense reference (i.e. padding-trim is correct).
    """
    with jax.transfer_guard_host_to_device("allow"):
        points = jax.device_put(
            np.linspace(0.1, 0.9, 21, dtype=np.float64).reshape(7, 3)
        )
        gammas, gammadashs, currents = _build_collective_circular_coils()
        coil_spec = grouped_coil_set_spec_from_lists(gammas, gammadashs, currents)
        stacked_gammas = jnp.stack(gammas)
        stacked_gammadashs = jnp.stack(gammadashs)
        stacked_currents = jnp.stack(currents)

    with jax.transfer_guard("disallow"):
        dense_B = biot_savart_B(
            points, stacked_gammas, stacked_gammadashs, stacked_currents
        )
        collective_B = grouped_biot_savart_B_from_spec(points, coil_spec)
        summary = grouped_field_sharding_summary(points, coil_spec)
        _block_until_ready_tree((dense_B, collective_B))

    _assert_allclose_to_reference(collective_B, dense_B)

    assert summary["field_collective"] is True
    assert summary["strategy"] == "points_coils"
    assert summary["point_device_count"] > 1, summary


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


def _run_surface_quadrature_sharding_case() -> None:
    nphi = 8
    ntheta = 5
    Bcoil_host = np.linspace(0.2, 1.4, nphi * ntheta * 3, dtype=np.float64).reshape(
        nphi, ntheta, 3
    )
    target_host = np.linspace(-0.01, 0.02, nphi * ntheta, dtype=np.float64).reshape(
        nphi, ntheta
    )
    normal_host = np.linspace(0.3, 1.7, nphi * ntheta * 3, dtype=np.float64).reshape(
        nphi, ntheta, 3
    )

    config = surface_quadrature_sharding_config(Bcoil_host)
    with jax.transfer_guard_host_to_device("allow"):
        with jax.transfer_guard_device_to_device("allow"):
            Bcoil, target, normal = maybe_shard_surface_quadrature_inputs(
                Bcoil_host,
                target_host,
                normal_host,
                config=config,
            )
    reference_Bcoil = jnp.asarray(Bcoil_host)
    reference_target = jnp.asarray(target_host)
    reference_normal = jnp.asarray(normal_host)

    for definition in ("quadratic flux", "normalized", "local"):
        sharded_value = integral_BdotN_surface_sharded(
            Bcoil,
            target,
            normal,
            definition,
        )
        reference_value = integral_BdotN(
            reference_Bcoil,
            reference_target,
            reference_normal,
            definition,
        )
        np.testing.assert_allclose(
            np.asarray(sharded_value),
            np.asarray(reference_value),
            rtol=1e-12,
            atol=1e-12,
        )

    summary = integral_BdotN_sharding_summary(
        Bcoil,
        target,
        normal,
        "quadratic flux",
    )
    assert summary["surface_quadrature_sharded"] is True, summary
    assert summary["kind"] == "NamedSharding", summary
    assert summary["surface_quadrature_device_count"] == 4, summary

    lowered = (
        jax.jit(integral_BdotN_surface_sharded, static_argnames=("definition",))
        .lower(Bcoil, target, normal, "quadratic flux")
        .as_text()
    )
    assert "all_reduce" in lowered.lower(), lowered[:2000]


def _run_seed_batch_value_grad_sharding_case() -> None:
    import simsopt_jax_adapters.geo.surface_objectives_traceable as surfaceobjectives_traceable_jax_module

    compiled_value_and_grad_for = jax.jit(
        lambda coil_dofs: (
            jnp.sum(coil_dofs * coil_dofs),
            2.0 * coil_dofs,
        )
    )
    batched_value_and_grad = surfaceobjectives_traceable_jax_module._make_traceable_batched_value_and_grad_pipeline(
        compiled_value_and_grad_for
    )
    coil_dofs_batch_host = np.linspace(-1.0, 1.0, 8 * 3, dtype=np.float64).reshape(8, 3)
    config = seed_batch_sharding_config(coil_dofs_batch_host)
    with jax.transfer_guard_host_to_device("allow"):
        with jax.transfer_guard_device_to_device("allow"):
            (coil_dofs_batch,) = maybe_shard_seed_batch_inputs(
                coil_dofs_batch_host,
                config=config,
            )

    values, grads = batched_value_and_grad(coil_dofs_batch)
    reference_values, reference_grads = jax.vmap(compiled_value_and_grad_for)(
        coil_dofs_batch
    )

    np.testing.assert_allclose(
        np.asarray(values),
        np.asarray(reference_values),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(grads),
        np.asarray(reference_grads),
        rtol=1e-12,
        atol=1e-12,
    )
    config = seed_batch_sharding_config(coil_dofs_batch)
    summary = seed_batch_sharding_summary(values, config=config)
    assert summary["seed_batch_sharded"] is True, summary
    assert summary["kind"] == "NamedSharding", summary
    assert summary["seed_batch_device_count"] == 4, summary


def _run_target_minimize_replicated_sharding_vjp_case() -> None:
    if not _configure_strict_cpu_parity_backend():
        _skip_case(_STRICT_CPU_PARITY_SKIP_REASON)
        return

    local_devices = jax.local_devices()
    mesh = Mesh(np.asarray(local_devices, dtype=object), ("d",))
    replicated = NamedSharding(mesh, P())
    with jax.transfer_guard_host_to_device("allow"):
        matrix = jax.device_put(np.eye(2, dtype=np.float64), replicated)
        target = jax.device_put(
            np.asarray([0.25, -0.75], dtype=np.float64),
            replicated,
        )
        x0 = jax.device_put(
            np.asarray([2.0, -1.0], dtype=np.float64),
            local_devices[0],
        )

    def objective(x):
        sharded_residual = matrix @ x - target
        local_residual = x
        return jnp.sum(sharded_residual * sharded_residual) + jnp.sum(
            local_residual * local_residual
        )

    result = target_minimize(
        objective,
        x0,
        method="bfgs-ondevice",
        maxiter=4,
    )

    assert result.success is True
    np.testing.assert_allclose(np.asarray(result.x), [0.125, -0.375], atol=1e-8)


def _run_boozer_penalty_replicated_sharding_vjp_case() -> None:
    if not _configure_strict_cpu_parity_backend():
        _skip_case(_STRICT_CPU_PARITY_SKIP_REASON)
        return

    mesh = Mesh(np.asarray(jax.devices(), dtype=object), ("d",))
    replicated = NamedSharding(mesh, P())
    mpol = 1
    ntor = 1
    nfp = 1
    shape = (2 * mpol + 1, 2 * ntor + 1)
    xc = np.zeros(shape, dtype=np.float64)
    yc = np.zeros(shape, dtype=np.float64)
    zc = np.zeros(shape, dtype=np.float64)
    xc[0, 0] = 1.0
    xc[1, 0] = 0.1
    zc[mpol + 1, 0] = 0.1
    surface_dofs = np.concatenate((xc.ravel(), yc.ravel(), zc.ravel()))
    with jax.transfer_guard_host_to_device("allow"):
        qphi = jnp.linspace(0.0, 1.0, 4, endpoint=False)
        qtheta = jnp.linspace(0.0, 1.0, 4, endpoint=False)

    phi = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    coil_gamma = np.stack(
        (np.cos(phi), np.sin(phi), 0.3 * np.ones_like(phi)),
        axis=-1,
    )
    coil_gammadash = np.stack(
        (-2.0 * np.pi * np.sin(phi), 2.0 * np.pi * np.cos(phi), np.zeros_like(phi)),
        axis=-1,
    )
    with jax.transfer_guard_host_to_device("allow"):
        x0 = jax.device_put(
            np.concatenate((surface_dofs, np.asarray([0.3, 1.0]))),
            replicated,
        )
        coil_arrays = [
            (
                jax.device_put(coil_gamma[None, :, :], replicated),
                jax.device_put(coil_gammadash[None, :, :], replicated),
                jax.device_put(np.asarray([1.0e5], dtype=np.float64), replicated),
            )
        ]

    def objective(x):
        return _boozer_penalty_objective(
            x,
            coil_arrays=coil_arrays,
            quadpoints_phi=qphi,
            quadpoints_theta=qtheta,
            mpol=mpol,
            ntor=ntor,
            nfp=nfp,
            stellsym=False,
            scatter_indices=None,
            surface_kind="generic",
            label_quadpoints_phi=qphi,
            label_quadpoints_theta=qtheta,
            label_mpol=mpol,
            label_ntor=ntor,
            label_nfp=nfp,
            label_stellsym=False,
            label_scatter_indices=None,
            label_surface_kind="generic",
            targetlabel=2.0 * np.pi**2 * 0.1**2,
            constraint_weight=1.0,
            label_type="volume",
            phi_idx=0,
            optimize_G=True,
            weight_inv_modB=True,
        )

    value, gradient = jax.jit(jax.value_and_grad(objective))(x0)
    with jax.transfer_guard_device_to_host("allow"):
        assert bool(jnp.isfinite(value))
        assert bool(jnp.all(jnp.isfinite(gradient)))
    assert getattr(gradient, "sharding").spec == P()
    assert len(getattr(gradient, "sharding").device_set) == 4


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
        _skip_case(_STRICT_GPU_FAST_SKIP_REASON)
        return

    modes = np.zeros(10, dtype=np.float64)
    qpts = np.linspace(0.0, 1.0, 8, endpoint=False)
    phi, theta = gamma_2d(modes, qpts, 2, G=1, H=0)

    assert phi.shape == (8,)
    assert theta.shape == (8,)


def _run_closed_curve_self_intersection_summary_case() -> None:
    gpu = _configure_strict_gpu_fast_backend()
    if gpu is None:
        _skip_case(_STRICT_GPU_FAST_SKIP_REASON)
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
        _skip_case(_STRICT_GPU_FAST_SKIP_REASON)
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

    eager_gamma = gamma_from_dofs(jax.device_put(dofs, device=gpu))
    jitted_gamma = jax.jit(gamma_from_dofs)(jax.device_put(dofs, device=gpu))

    _assert_finite_array(eager_gamma, expected_shape=(2, 2, 3))
    _assert_finite_array(jitted_gamma, expected_shape=(2, 2, 3))


def _run_coil_symmetry_spec_identity_default_case() -> None:
    _configure_transfer_guard_cpu_parity_backend()

    symmetry = make_coil_symmetry_spec(scale=2.5)

    assert symmetry.rotmat.shape == (3, 3)
    assert np.allclose(np.asarray(symmetry.rotmat), np.eye(3))
    assert symmetry.has_rotation is False


def _surface_rzfourier_transfer_guard_spec():
    spec = surface_rz_fourier_spec_from_dofs(
        jnp.asarray([1.0, 0.1, 0.1], dtype=jnp.float64),
        quadpoints_phi=jnp.asarray(np.arange(16) / 16, dtype=jnp.float64),
        quadpoints_theta=jnp.asarray(np.arange(16) / 16, dtype=jnp.float64),
        mpol=1,
        ntor=0,
        nfp=5,
        stellsym=True,
    )
    _configure_transfer_guard_cpu_parity_backend()
    return spec


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
    spec = _surface_rzfourier_transfer_guard_spec()

    assert spec.rs.shape == spec.rc.shape
    assert spec.zc.shape == spec.rc.shape


def _run_surface_rzfourier_gamma_from_spec_case() -> None:
    gamma = surface_rz_fourier_gamma_from_spec(_surface_rzfourier_transfer_guard_spec())

    _assert_finite_array(gamma, expected_shape=(16, 16, 3))


def _run_surface_rzfourier_normal_from_spec_case() -> None:
    normal = surface_rz_fourier_normal_from_spec(
        _surface_rzfourier_transfer_guard_spec()
    )

    _assert_finite_array(normal, expected_shape=(16, 16, 3))


def _run_grouped_biot_savart_gpu_current_arrays_case() -> None:
    gpu = _configure_strict_gpu_fast_backend()
    if gpu is None:
        _skip_case(_STRICT_GPU_FAST_SKIP_REASON)
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
        _skip_case(_STRICT_GPU_FAST_SKIP_REASON)
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
        _skip_case(_STRICT_GPU_FAST_SKIP_REASON)
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
        _skip_case(_STRICT_CPU_PARITY_SKIP_REASON)
        return

    from simsopt_jax.geo.optimizers.optimizer import jax_minimize  # type: ignore[import-untyped]

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


def _run_structured_mutable_objective_state_case() -> None:
    if not _configure_strict_cpu_parity_backend():
        _skip_case(_STRICT_CPU_PARITY_SKIP_REASON)
        return

    objective = _StructuredShiftedQuadraticValueAndGrad([0.0, 0.0])
    _mark_cacheable_jit_value_and_grad(objective)
    x0 = {
        "x": jnp.asarray(np.array([2.0, -1.0], dtype=np.float64)),
    }

    first = target_minimize(
        objective,
        x0,
        method="lbfgs-ondevice",
        value_and_grad=True,
        maxiter=_MUTABLE_OBJECTIVE_SMOKE_MAXITER,
    )
    objective.target = np.asarray([1.5, -0.5], dtype=np.float64)
    second = target_minimize(
        objective,
        x0,
        method="lbfgs-ondevice",
        value_and_grad=True,
        maxiter=_MUTABLE_OBJECTIVE_SMOKE_MAXITER,
    )

    assert first.success is True
    assert second.success is True
    np.testing.assert_allclose(
        np.asarray(first.x["x"]),
        np.asarray([0.0, 0.0]),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(second.x["x"]),
        np.asarray([1.5, -0.5]),
        atol=1e-6,
    )


def _parse_optimizer_method(method: str) -> OptimizerMethod:
    if method == "lbfgs-ondevice":
        return cast(OptimizerMethod, method)
    if method == "bfgs-ondevice":
        return cast(OptimizerMethod, method)
    raise ValueError(f"unsupported optimizer method {method!r}")


def _require_two_cpu_devices_for_dense_condition_case():
    devices = jax.devices("cpu")
    if len(devices) < 2:
        _skip_case(
            "dense-condition placement regression needs at least two CPU devices"
        )
    return devices[0], devices[1]


def _run_dense_condition_estimate_cross_device_factors_case() -> None:
    matrix_device, factor_device = _require_two_cpu_devices_for_dense_condition_case()
    diagonal = np.geomspace(1.0, 1.0e-5, 48)
    expected_condition = float(diagonal.max() / diagonal.min())
    matrix = jax.device_put(
        jnp.diag(jnp.asarray(diagonal, dtype=jnp.float64)),
        device=matrix_device,
    )
    lu, piv = jax.scipy.linalg.lu_factor(matrix)
    factors_on_other_device = (
        jax.device_put(lu, device=factor_device),
        jax.device_put(piv, device=factor_device),
    )

    estimate = _optimizer_module._dense_matrix_condition_estimate(
        matrix,
        lu_piv=factors_on_other_device,
    )
    estimate_value = float(jax.device_get(estimate))

    assert np.isfinite(estimate_value)
    relative_error = abs(estimate_value - expected_condition) / expected_condition
    assert relative_error < 1.0e-6, (estimate_value, expected_condition)


def _run_dense_condition_threshold_nondefault_device_case() -> None:
    _, certificate_device = _require_two_cpu_devices_for_dense_condition_case()
    certificate = jax.device_put(
        np.asarray(1.0e5, dtype=np.float64),
        device=certificate_device,
    )

    with jax.transfer_guard("disallow"):
        safe = _optimizer_module._dense_matrix_condition_estimate_numerically_safe(
            certificate,
            size=48,
            dtype=np.dtype(np.float64),
        )

    assert safe.device == certificate_device
    assert bool(jax.device_get(safe))


def _dispatch_case(args: argparse.Namespace) -> None:
    if args.case == "compile-count":
        _run_compile_count_case(_parse_optimizer_method(args.method))
        return
    if args.case == "biot-savart-point-chunking":
        _run_biot_savart_point_chunking_case()
        return
    if args.case == "target-compile-count":
        _run_target_compile_count_case()
        return
    if args.case == "mutable-objective-state":
        _run_mutable_objective_state_case()
        return
    if args.case == "structured-mutable-objective-state":
        _run_structured_mutable_objective_state_case()
        return
    if args.case == "grouped-gpu-spec-eval":
        _run_grouped_biot_savart_gpu_spec_eval_case()
        return
    if args.case == "grouped-explicit-point-sharding":
        _run_grouped_biot_savart_explicit_point_sharding_case()
        return
    if args.case == "grouped-coil-collective":
        _run_grouped_biot_savart_coil_collective_case()
        return
    if args.case == "grouped-points-coils-collective":
        _run_grouped_biot_savart_points_coils_collective_case()
        return
    if args.case == "grouped-points-coils-non-divisible":
        _run_grouped_biot_savart_points_coils_non_divisible_case()
        return
    if args.case == "pairwise-penalty-explicit-row-sharding":
        _run_pairwise_penalty_explicit_row_sharding_case()
        return
    if args.case == "surface-quadrature-sharding":
        _run_surface_quadrature_sharding_case()
        return
    if args.case == "seed-batch-value-grad-sharding":
        _run_seed_batch_value_grad_sharding_case()
        return
    if args.case == "target-minimize-replicated-sharding-vjp":
        _run_target_minimize_replicated_sharding_vjp_case()
        return
    if args.case == "boozer-penalty-replicated-sharding-vjp":
        _run_boozer_penalty_replicated_sharding_vjp_case()
        return
    if args.case == "shifted-grid-axis-sample":
        _run_shifted_grid_axis_sample_case()
        return
    if args.case == "gamma-2d-eager-host-constants":
        _run_gamma_2d_eager_host_constants_case()
        return
    if args.case == "closed-curve-self-intersection-summary":
        _run_closed_curve_self_intersection_summary_case()
        return
    if args.case == "surface-xyztensorfourier-gamma-from-dofs":
        _run_surface_xyztensorfourier_gamma_from_dofs_case()
        return
    if args.case == "coil-symmetry-spec-identity-default":
        _run_coil_symmetry_spec_identity_default_case()
        return
    if args.case == "pairwise-curve-penalty-pure-functions":
        _run_pairwise_curve_penalty_pure_functions_case()
        return
    if args.case == "surfacerzfourier-spec-defaults":
        _run_surfacerzfourier_spec_defaults_case()
        return
    if args.case == "surface-rzfourier-gamma-from-spec":
        _run_surface_rzfourier_gamma_from_spec_case()
        return
    if args.case == "surface-rzfourier-normal-from-spec":
        _run_surface_rzfourier_normal_from_spec_case()
        return
    if args.case == "grouped-gpu-current-arrays":
        _run_grouped_biot_savart_gpu_current_arrays_case()
        return
    if args.case == "grouped-host-scalar-currents":
        _run_grouped_biot_savart_host_scalar_currents_case()
        return
    if args.case == "grouped-host-spec-vjp":
        _run_grouped_biot_savart_host_spec_vjp_case()
        return
    if args.case == "dense-condition-estimate-cross-device-factors":
        _run_dense_condition_estimate_cross_device_factors_case()
        return
    if args.case == "dense-condition-threshold-nondefault-device":
        _run_dense_condition_threshold_nondefault_device_case()
        return
    if args.case == "certificate-probe-key-x64-disabled":
        _run_certificate_probe_key_x64_disabled_case()
        return
    raise ValueError(f"unsupported subprocess case {args.case!r}")


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
    subparsers.add_parser("mutable-objective-state")
    subparsers.add_parser("structured-mutable-objective-state")
    subparsers.add_parser("grouped-gpu-spec-eval")
    subparsers.add_parser("grouped-explicit-point-sharding")
    subparsers.add_parser("grouped-coil-collective")
    subparsers.add_parser("grouped-points-coils-collective")
    subparsers.add_parser("grouped-points-coils-non-divisible")
    subparsers.add_parser("pairwise-penalty-explicit-row-sharding")
    subparsers.add_parser("surface-quadrature-sharding")
    subparsers.add_parser("seed-batch-value-grad-sharding")
    subparsers.add_parser("target-minimize-replicated-sharding-vjp")
    subparsers.add_parser("boozer-penalty-replicated-sharding-vjp")
    subparsers.add_parser("shifted-grid-axis-sample")
    subparsers.add_parser("gamma-2d-eager-host-constants")
    subparsers.add_parser("closed-curve-self-intersection-summary")
    subparsers.add_parser("surface-xyztensorfourier-gamma-from-dofs")
    subparsers.add_parser("coil-symmetry-spec-identity-default")
    subparsers.add_parser("pairwise-curve-penalty-pure-functions")
    subparsers.add_parser("surfacerzfourier-spec-defaults")
    subparsers.add_parser("surface-rzfourier-gamma-from-spec")
    subparsers.add_parser("surface-rzfourier-normal-from-spec")
    subparsers.add_parser("grouped-gpu-current-arrays")
    subparsers.add_parser("grouped-host-scalar-currents")
    subparsers.add_parser("grouped-host-spec-vjp")
    subparsers.add_parser("dense-condition-estimate-cross-device-factors")
    subparsers.add_parser("dense-condition-threshold-nondefault-device")
    subparsers.add_parser("certificate-probe-key-x64-disabled")

    args = parser.parse_args(argv)

    try:
        _dispatch_case(args)
    except SkippedCase as exc:
        print(
            json.dumps(
                {
                    "case": args.case,
                    "checked": False,
                    "skipped": True,
                    "skip_reason": str(exc),
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
