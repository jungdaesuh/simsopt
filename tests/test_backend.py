from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys
import types
import warnings

import numpy as np
import pytest

from conftest import (
    _BACKEND_RUNTIME_ENV_VARS as _BACKEND_ENV_VARS,
    _force_x64,
    _require_jax,
    _restore_backend_runtime_env,
    _restore_loaded_jax_runtime_config,
    ensure_gpu_determinism_xla_flag,
)
_BACKEND_MODULE_NAMES = (
    "simsopt",
    "simsopt.backend",
    "simsopt.backend.runtime",
)
_MISSING_MODULE = object()
def _snapshot_backend_modules() -> dict[str, object]:
    return {
        name: sys.modules.get(name, _MISSING_MODULE) for name in _BACKEND_MODULE_NAMES
    }


def _restore_backend_module_snapshot(snapshot: dict[str, object]) -> None:
    for name, module in snapshot.items():
        if module is _MISSING_MODULE:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def _clear_backend_modules() -> None:
    for name in reversed(_BACKEND_MODULE_NAMES):
        sys.modules.pop(name, None)


@pytest.fixture(autouse=True)
def _restore_backend_modules():
    snapshot = _snapshot_backend_modules()
    try:
        yield
    finally:
        _restore_backend_module_snapshot(snapshot)


def _fresh_backend():
    package_root = Path(__file__).resolve().parents[1] / "src" / "simsopt"
    _clear_backend_modules()
    package = types.ModuleType("simsopt")
    package.__path__ = [str(package_root)]
    sys.modules["simsopt"] = package
    spec = importlib.util.spec_from_file_location(
        "simsopt.backend",
        package_root / "backend.py",
    )
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["simsopt.backend"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    mod.invalidate_backend_cache()
    return mod


def _fresh_backend_with_fake_runtime_home(monkeypatch):
    backend = _fresh_backend()
    _patch_fake_runtime_home(monkeypatch)
    return backend


def _fresh_sharding_module():
    import simsopt.jax_core.sharding as sharding

    return importlib.reload(sharding)


def _clear_backend_env(monkeypatch) -> None:
    for name in _BACKEND_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def _disable_chunk_autotune(monkeypatch) -> None:
    monkeypatch.setenv("SIMSOPT_JAX_CHUNK_AUTOTUNE", "0")


def _patch_fake_runtime_home(monkeypatch) -> None:
    runtime_module = sys.modules["simsopt.backend.runtime"]
    fake_home = Path("/tmp/simsopt-jax-cache-home")
    monkeypatch.setattr(runtime_module.Path, "home", lambda: fake_home)


def _install_fake_jax(monkeypatch, *, calls=None, default_backend=None) -> None:
    update = (
        (lambda name, value: None)
        if calls is None
        else (lambda name, value: calls.append((name, value)))
    )
    fake_jax = types.SimpleNamespace(
        config=types.SimpleNamespace(update=update),
    )
    if default_backend is not None:
        fake_jax.default_backend = lambda: default_backend
    monkeypatch.setitem(sys.modules, "jax", fake_jax)


def _set_distributed_init_env(
    monkeypatch,
    *,
    coordinator_address: str = "127.0.0.1:12345",
    num_processes: str = "4",
    process_id: str = "1",
    local_device_ids: str | None = None,
    visible_devices: str | None = None,
) -> None:
    monkeypatch.setenv("SIMSOPT_JAX_DISTRIBUTED_INIT", "1")
    monkeypatch.setenv("SIMSOPT_JAX_COORDINATOR_ADDRESS", coordinator_address)
    monkeypatch.setenv("SIMSOPT_JAX_NUM_PROCESSES", num_processes)
    monkeypatch.setenv("SIMSOPT_JAX_PROCESS_ID", process_id)
    if local_device_ids is None:
        monkeypatch.delenv("SIMSOPT_JAX_LOCAL_DEVICE_IDS", raising=False)
    else:
        monkeypatch.setenv("SIMSOPT_JAX_LOCAL_DEVICE_IDS", local_device_ids)
    if visible_devices is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible_devices)


def _assert_synced_runtime_env(
    backend, *, mode: str, backend_name: str, platform: str, strict: bool
) -> None:
    runtime_platform = "METAL" if platform == "metal" else platform
    runtime_platforms = "cuda,cpu" if platform == "cuda" else runtime_platform
    assert os.environ["SIMSOPT_BACKEND_MODE"] == mode
    assert os.environ["SIMSOPT_BACKEND_STRICT"] == ("1" if strict else "0")
    assert os.environ["SIMSOPT_BACKEND"] == backend_name
    assert os.environ["STAGE2_BACKEND"] == backend_name
    assert os.environ["SIMSOPT_JAX_PLATFORM"] == runtime_platform
    assert os.environ["SIMSOPT_JAX_BACKEND"] == runtime_platform
    assert os.environ["JAX_PLATFORMS"] == runtime_platforms


def _assert_backend_policy(
    policy,
    *,
    mode: str,
    backend_name: str,
    platform: str,
    strict: bool,
    parity_mode: bool,
    requires_x64: bool,
    chunk_policy: str,
    tolerance_tier: str,
    compilation_cache_policy: str,
    provenance_label: str,
) -> None:
    assert policy.mode == mode
    assert policy.backend == backend_name
    assert policy.jax_platform == platform
    assert policy.strict is strict
    assert policy.parity_mode is parity_mode
    assert policy.requires_x64 is requires_x64
    assert policy.chunk_policy == chunk_policy
    assert policy.tolerance_tier == tolerance_tier
    assert policy.compilation_cache_policy == compilation_cache_policy
    assert policy.provenance_label == provenance_label


def _assert_transfer_guard_resolution(backend, *, mode: str, expected: str | None):
    config = backend.set_backend(mode, configure_runtime=False)
    policy = backend.get_backend_policy(mode)

    assert config.transfer_guard == expected
    assert policy.transfer_guard == expected


def _capture_fallback_warnings(callback):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        callback()
    return caught


def test_backend_defaults_to_native_cpu(monkeypatch):
    _clear_backend_env(monkeypatch)
    backend = _fresh_backend()

    config = backend.get_backend_config()
    policy = backend.get_backend_policy()

    assert config.mode == "native_cpu"
    assert config.backend == "cpu"
    assert config.jax_platform == "cpu"
    assert config.strict is False
    assert backend.is_jax_backend() is False
    assert backend.is_backend_strict() is False
    _assert_backend_policy(
        policy,
        mode="native_cpu",
        backend_name="cpu",
        platform="cpu",
        strict=False,
        parity_mode=False,
        requires_x64=True,
        chunk_policy="host_reference",
        tolerance_tier="cpu_reference",
        compilation_cache_policy="not_applicable",
        provenance_label="native_cpu",
    )


def test_backend_resolves_legacy_env_pair(monkeypatch):
    _clear_backend_env(monkeypatch)
    monkeypatch.setenv("SIMSOPT_BACKEND", "jax")
    backend = _fresh_backend()

    config = backend.get_backend_config()

    assert config.mode == "jax_gpu_parity"
    assert config.backend == "jax"
    assert config.jax_platform == "cuda"


def test_backend_resolves_stage2_backend_env_alias(monkeypatch):
    _clear_backend_env(monkeypatch)
    monkeypatch.setenv("STAGE2_BACKEND", "jax")
    monkeypatch.setenv("SIMSOPT_JAX_BACKEND", "cuda")
    backend = _fresh_backend()

    config = backend.get_backend_config()

    assert config.mode == "jax_gpu_parity"
    assert config.backend == "jax"
    assert config.jax_platform == "cuda"


def test_backend_resolves_explicit_metal_legacy_env_pair(monkeypatch):
    _clear_backend_env(monkeypatch)
    monkeypatch.setenv("SIMSOPT_BACKEND", "jax")
    monkeypatch.setenv("SIMSOPT_JAX_PLATFORM", "metal")
    backend = _fresh_backend()

    config = backend.get_backend_config()
    policy = backend.get_backend_policy()

    assert config.mode == "jax_metal_smoke"
    assert config.backend == "jax"
    assert config.jax_platform == "metal"
    _assert_backend_policy(
        policy,
        mode="jax_metal_smoke",
        backend_name="jax",
        platform="metal",
        strict=False,
        parity_mode=False,
        requires_x64=False,
        chunk_policy="stable_default",
        tolerance_tier="smoke",
        compilation_cache_policy="optional_persistent",
        provenance_label="jax_metal_smoke",
    )
    assert backend.requires_x64() is False


def test_set_backend_updates_mode_and_legacy_envs(monkeypatch):
    _clear_backend_env(monkeypatch)
    backend = _fresh_backend()

    config = backend.set_backend(
        "jax_gpu_fast",
        strict=True,
        configure_runtime=False,
    )

    assert config.mode == "jax_gpu_fast"
    assert config.backend == "jax"
    assert config.jax_platform == "cuda"
    assert config.strict is True
    assert backend.get_backend_mode() == "jax_gpu_fast"
    assert backend.get_backend() == "jax"
    assert backend.get_jax_platform() == "cuda"
    assert backend.is_backend_strict() is True

    _assert_synced_runtime_env(
        backend,
        mode="jax_gpu_fast",
        backend_name="jax",
        platform="cuda",
        strict=True,
    )


def test_set_backend_updates_envs_for_jax_metal_smoke(monkeypatch):
    _clear_backend_env(monkeypatch)
    backend = _fresh_backend()

    config = backend.set_backend("jax_metal_smoke", configure_runtime=False)

    assert config.mode == "jax_metal_smoke"
    assert config.backend == "jax"
    assert config.jax_platform == "metal"
    assert backend.get_jax_platform() == "metal"
    assert backend.requires_x64() is False
    _assert_synced_runtime_env(
        backend,
        mode="jax_metal_smoke",
        backend_name="jax",
        platform="metal",
        strict=False,
    )


def test_backend_mode_policy_helpers(monkeypatch):
    _clear_backend_env(monkeypatch)
    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_gpu_parity")
    monkeypatch.setenv("SIMSOPT_BACKEND_STRICT", "1")
    backend = _fresh_backend()

    policy = backend.get_backend_policy()

    _assert_backend_policy(
        policy,
        mode="jax_gpu_parity",
        backend_name="jax",
        platform="cuda",
        strict=True,
        parity_mode=True,
        requires_x64=True,
        chunk_policy="stable_default",
        tolerance_tier="parity",
        compilation_cache_policy="optional_persistent",
        provenance_label="jax_gpu_parity",
    )
    assert backend.is_parity_mode() is True
    assert backend.requires_x64() is True
    assert backend.get_chunk_policy() == "stable_default"
    assert backend.get_tolerance_tier() == "parity"
    assert backend.get_compilation_cache_policy() == "optional_persistent"
    assert backend.get_provenance_label() == "jax_gpu_parity"
    assert backend.get_transfer_guard() == "log"


def test_fast_mode_policy_helpers(monkeypatch):
    _clear_backend_env(monkeypatch)
    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_gpu_fast")
    backend = _fresh_backend()

    policy = backend.get_backend_policy()

    _assert_backend_policy(
        policy,
        mode="jax_gpu_fast",
        backend_name="jax",
        platform="cuda",
        strict=False,
        parity_mode=False,
        requires_x64=True,
        chunk_policy="performance_tuned",
        tolerance_tier="fast",
        compilation_cache_policy="optional_persistent",
        provenance_label="jax_gpu_fast",
    )
    assert backend.is_parity_mode() is False
    _assert_transfer_guard_resolution(
        backend,
        mode="jax_gpu_fast",
        expected="log",
    )


def test_metal_smoke_mode_policy_helpers(monkeypatch):
    _clear_backend_env(monkeypatch)
    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_metal_smoke")
    backend = _fresh_backend()

    policy = backend.get_backend_policy()

    _assert_backend_policy(
        policy,
        mode="jax_metal_smoke",
        backend_name="jax",
        platform="metal",
        strict=False,
        parity_mode=False,
        requires_x64=False,
        chunk_policy="stable_default",
        tolerance_tier="smoke",
        compilation_cache_policy="optional_persistent",
        provenance_label="jax_metal_smoke",
    )
    assert backend.is_parity_mode() is False
    assert backend.requires_x64() is False
    _assert_transfer_guard_resolution(
        backend,
        mode="jax_metal_smoke",
        expected="log",
    )


def test_gpu_parity_mode_exposes_ci_contract_defaults(monkeypatch):
    _clear_backend_env(monkeypatch)
    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_gpu_parity")
    backend = _fresh_backend()

    policy = backend.get_backend_policy()

    assert policy.gpu_reduction_order_max_ulp == 10
    assert policy.gpu_reduction_order_rel_tol == pytest.approx(1e-12)
    assert policy.gpu_reproducibility_seed == 1729
    assert policy.gpu_reproducibility_sample_size == 1000
    assert policy.tolerance_ratchet_factor == pytest.approx(10.0)


def test_backend_cache_clear_callbacks_replace_reloaded_registrations(monkeypatch):
    _clear_backend_env(monkeypatch)
    backend = _fresh_backend()
    runtime = sys.modules["simsopt.backend.runtime"]
    calls: list[str] = []

    def callback_first():
        calls.append("first")

    def callback_second():
        calls.append("second")

    callback_first.__module__ = "simsopt.geo.boozersurface_jax"
    callback_first.__qualname__ = "_clear_backend_warning_cache"
    callback_second.__module__ = "simsopt.geo.boozersurface_jax"
    callback_second.__qualname__ = "_clear_backend_warning_cache"

    runtime.register_backend_cache_clear(callback_first)
    runtime.register_backend_cache_clear(callback_second)

    assert len(runtime._backend_cache_clear_callbacks) == 1

    backend.invalidate_backend_cache()

    assert calls == ["second"]


@pytest.mark.parametrize("mode", ["jax_cpu_parity", "jax_gpu_parity"])
def test_parity_modes_default_transfer_guard_to_log(monkeypatch, mode):
    _clear_backend_env(monkeypatch)
    backend = _fresh_backend()

    _assert_transfer_guard_resolution(backend, mode=mode, expected="log")


def test_point_chunk_size_defaults_follow_mode(monkeypatch):
    _clear_backend_env(monkeypatch)
    _disable_chunk_autotune(monkeypatch)
    backend = _fresh_backend()

    assert backend.get_point_chunk_size("native_cpu") == 0
    assert backend.get_point_chunk_size("jax_cpu_parity") == 256
    assert backend.get_point_chunk_size("jax_gpu_fast") == 1024


def test_pairwise_penalty_chunk_size_defaults_follow_mode(monkeypatch):
    _clear_backend_env(monkeypatch)
    _disable_chunk_autotune(monkeypatch)
    backend = _fresh_backend()

    assert backend.get_pairwise_penalty_chunk_size("native_cpu") == 0
    assert backend.get_pairwise_penalty_chunk_size("jax_cpu_parity") == 256
    assert backend.get_pairwise_penalty_chunk_size("jax_gpu_fast") == 1024


def test_field_kernel_tuning_defaults_follow_mode(monkeypatch):
    _clear_backend_env(monkeypatch)
    _disable_chunk_autotune(monkeypatch)
    backend = _fresh_backend()

    tuning = backend.get_field_kernel_tuning("jax_gpu_fast")

    assert tuning.mode == "jax_gpu_fast"
    assert tuning.chunk_policy == "performance_tuned"
    assert tuning.coil_chunk_size == 64
    assert tuning.quadrature_block_size == 64
    assert backend.get_coil_chunk_size("jax_cpu_parity") == 16
    assert backend.get_quadrature_block_size("jax_cpu_parity") == 0


def test_field_kernel_tuning_allows_explicit_env_overrides(monkeypatch):
    _clear_backend_env(monkeypatch)
    monkeypatch.setenv("SIMSOPT_JAX_COIL_CHUNK_SIZE", "23")
    monkeypatch.setenv("SIMSOPT_JAX_QUADRATURE_BLOCK_SIZE", "11")
    backend = _fresh_backend()

    tuning = backend.get_field_kernel_tuning("jax_gpu_fast")

    assert tuning.coil_chunk_size == 23
    assert tuning.quadrature_block_size == 11


def test_pairwise_penalty_chunk_size_allows_explicit_env_override(monkeypatch):
    _clear_backend_env(monkeypatch)
    monkeypatch.setenv("SIMSOPT_JAX_PENALTY_POINT_CHUNK_SIZE", "37")
    backend = _fresh_backend()

    assert backend.get_pairwise_penalty_chunk_size("jax_gpu_fast") == 37


def test_transfer_guard_disallow_forces_dense_audit_field_tuning(monkeypatch):
    _clear_backend_env(monkeypatch)
    _disable_chunk_autotune(monkeypatch)
    backend = _fresh_backend()

    backend.set_backend(
        "jax_cpu_parity",
        strict=True,
        transfer_guard="disallow",
        configure_runtime=False,
    )
    tuning = backend.get_field_kernel_tuning("jax_cpu_parity")

    assert backend.get_point_chunk_size("jax_cpu_parity") == 0
    assert tuning.chunk_policy == "stable_default_dense_audit"
    assert tuning.coil_chunk_size == 0
    assert tuning.quadrature_block_size == 0
    assert backend.get_pairwise_penalty_chunk_size("jax_cpu_parity") == 256


def test_chunk_tuning_autotunes_gpu_fast_from_memory_budget_env(monkeypatch):
    _clear_backend_env(monkeypatch)
    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_gpu_fast")
    monkeypatch.setenv("SIMSOPT_JAX_GPU_MEMORY_TOTAL_MB", "24576")
    backend = _fresh_backend()

    tuning = backend.get_chunk_tuning()

    assert tuning.mode == "jax_gpu_fast"
    assert tuning.chunk_policy == "performance_tuned"
    assert tuning.autotuned is True
    assert tuning.autotune_source == "SIMSOPT_JAX_GPU_MEMORY_TOTAL_MB"
    assert tuning.gpu_total_memory_mb == 24576
    assert tuning.coil_chunk_size == 128
    assert tuning.quadrature_block_size == 128
    assert tuning.point_chunk_size == 2048
    assert tuning.pairwise_penalty_chunk_size == 2048
    assert backend.get_field_kernel_tuning().coil_chunk_size == 128
    assert backend.get_point_chunk_size() == 2048
    assert backend.get_pairwise_penalty_chunk_size() == 2048


def test_chunk_tuning_autotune_respects_explicit_env_overrides(monkeypatch):
    _clear_backend_env(monkeypatch)
    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_gpu_fast")
    monkeypatch.setenv("SIMSOPT_JAX_GPU_MEMORY_TOTAL_MB", "24576")
    monkeypatch.setenv("SIMSOPT_JAX_COIL_CHUNK_SIZE", "19")
    monkeypatch.setenv("SIMSOPT_JAX_QUADRATURE_BLOCK_SIZE", "7")
    monkeypatch.setenv("SIMSOPT_JAX_PENALTY_POINT_CHUNK_SIZE", "11")
    backend = _fresh_backend()

    tuning = backend.get_chunk_tuning()

    assert tuning.autotuned is True
    assert tuning.coil_chunk_size == 19
    assert tuning.quadrature_block_size == 7
    assert tuning.point_chunk_size == 2048
    assert tuning.pairwise_penalty_chunk_size == 11


def _assert_gpu_chunk_autotune_probe(
    monkeypatch,
    *,
    env_value,
    expected_selector,
    stdout,
    fake_jax=None,
):
    _clear_backend_env(monkeypatch)
    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_gpu_fast")
    if env_value is not None:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", env_value)
    if fake_jax is not None:
        monkeypatch.setitem(sys.modules, "jax", fake_jax)
    else:
        monkeypatch.delitem(sys.modules, "jax", raising=False)
    calls = _install_fake_nvidia_smi(monkeypatch, stdout)
    backend = _fresh_backend()
    tuning = backend.get_chunk_tuning()

    assert tuning.autotuned is True
    assert tuning.autotune_source == f"nvidia-smi[{expected_selector}]"
    assert tuning.gpu_total_memory_mb == int(stdout.split(",", 1)[1].strip())
    assert calls
    assert calls[0][-2:] == ["-i", str(expected_selector)]


def _install_fake_nvidia_smi(monkeypatch, stdout):
    calls: list[list[str]] = []

    def fake_run(cmd, *, check, capture_output, text):
        del check, capture_output, text
        calls.append(cmd)
        return types.SimpleNamespace(stdout=stdout)

    monkeypatch.setattr("subprocess.run", fake_run)
    return calls


def _make_recording_local_devices(*, result=None, error: Exception | None = None):
    calls: list[str | None] = []

    def _local_devices(*, backend=None):
        calls.append(backend)
        if error is not None:
            raise error
        return [] if result is None else result

    return _local_devices, calls


def _make_fake_cuda_device(local_hardware_id: int):
    return types.SimpleNamespace(local_hardware_id=local_hardware_id)


def _make_recording_cuda_device_probe(local_hardware_id: int):
    return _make_recording_local_devices(
        result=[_make_fake_cuda_device(local_hardware_id)]
    )


def _assert_public_gpu_local_device_calls(calls: list[str | None]):
    assert len(calls) >= 1
    assert calls == ["gpu"] * len(calls)


def test_chunk_tuning_autotunes_from_visible_cuda_device(monkeypatch):
    _assert_gpu_chunk_autotune_probe(
        monkeypatch,
        env_value="3,1",
        expected_selector=3,
        stdout="3, 16384\n",
    )


def test_chunk_tuning_autotunes_from_active_jax_cuda_device(monkeypatch):
    local_devices, local_device_calls = _make_recording_cuda_device_probe(2)
    _assert_gpu_chunk_autotune_probe(
        monkeypatch,
        env_value=None,
        expected_selector=2,
        stdout="2, 24576\n",
        fake_jax=types.SimpleNamespace(local_devices=local_devices),
    )
    _assert_public_gpu_local_device_calls(local_device_calls)


def test_chunk_tuning_prefers_imported_jax_cuda_device_over_visible_env(monkeypatch):
    local_devices, local_device_calls = _make_recording_cuda_device_probe(1)
    _assert_gpu_chunk_autotune_probe(
        monkeypatch,
        env_value="3,1",
        expected_selector=1,
        stdout="1, 24576\n",
        fake_jax=types.SimpleNamespace(local_devices=local_devices),
    )
    _assert_public_gpu_local_device_calls(local_device_calls)


def test_chunk_tuning_autotunes_from_visible_cuda_uuid_selector(monkeypatch):
    uuid_selector = "GPU-8932f937-d72c-4106-c12f-20bd9faed9f6"
    _assert_gpu_chunk_autotune_probe(
        monkeypatch,
        env_value=uuid_selector,
        expected_selector=uuid_selector,
        stdout="7, 24576\n",
    )
    backend = _fresh_backend()

    assert backend.get_active_cuda_device_index() is None


def test_query_active_gpu_memory_mb_uses_visible_cuda_uuid_selector(monkeypatch):
    _clear_backend_env(monkeypatch)
    uuid_selector = "GPU-8932f937-d72c-4106-c12f-20bd9faed9f6"
    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_gpu_fast")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", uuid_selector)
    monkeypatch.delitem(sys.modules, "jax", raising=False)
    calls = _install_fake_nvidia_smi(monkeypatch, "7, 512\n")
    backend = _fresh_backend()

    assert backend.get_active_cuda_device_index() is None
    assert backend.query_active_gpu_memory_mb() == pytest.approx(512.0)
    assert calls
    assert calls[0][-2:] == ["-i", uuid_selector]


def test_query_active_gpu_memory_mb_uses_visible_cuda_mig_selector(monkeypatch):
    _clear_backend_env(monkeypatch)
    mig_selector = "MIG-GPU-8932f937-d72c-4106-c12f-20bd9faed9f6/1/2"
    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_gpu_fast")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mig_selector)
    monkeypatch.delitem(sys.modules, "jax", raising=False)
    calls = _install_fake_nvidia_smi(monkeypatch, "7, 768\n")
    backend = _fresh_backend()

    assert backend.get_active_cuda_device_index() is None
    assert backend.query_active_gpu_memory_mb() == pytest.approx(768.0)
    assert calls
    assert calls[0][-2:] == ["-i", mig_selector]


def test_query_active_gpu_memory_mb_uses_visible_cuda_device(monkeypatch):
    _clear_backend_env(monkeypatch)
    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_gpu_fast")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,1")
    monkeypatch.delitem(sys.modules, "jax", raising=False)
    calls = _install_fake_nvidia_smi(monkeypatch, "3, 512\n")
    backend = _fresh_backend()

    assert backend.get_active_cuda_device_index() == 3
    assert backend.query_active_gpu_memory_mb() == pytest.approx(512.0)
    assert calls
    assert calls[0][-2:] == ["-i", "3"]


def test_query_active_gpu_memory_mb_falls_back_to_visible_env_when_jax_device_query_fails(
    monkeypatch,
):
    local_devices, local_device_calls = _make_recording_local_devices(
        error=RuntimeError("cannot resolve devices for backend='gpu' yet")
    )

    _clear_backend_env(monkeypatch)
    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_gpu_fast")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,1")
    monkeypatch.setitem(
        sys.modules,
        "jax",
        types.SimpleNamespace(local_devices=local_devices),
    )
    calls = _install_fake_nvidia_smi(monkeypatch, "3, 640\n")
    backend = _fresh_backend()

    assert backend.get_active_cuda_device_index() == 3
    assert backend.query_active_gpu_memory_mb() == pytest.approx(640.0)
    _assert_public_gpu_local_device_calls(local_device_calls)
    assert calls
    assert calls[0][-2:] == ["-i", "3"]


def test_query_active_gpu_memory_mb_falls_back_to_visible_env_before_distributed_init(
    monkeypatch,
):
    local_device_calls: list[str | None] = []

    def _unexpected_local_devices(*, backend=None):
        local_device_calls.append(backend)
        raise AssertionError(
            f"local_devices should not run before distributed init for backend={backend!r}"
        )

    _clear_backend_env(monkeypatch)
    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_gpu_fast")
    _set_distributed_init_env(monkeypatch, visible_devices="3,1")
    monkeypatch.setitem(
        sys.modules,
        "jax",
        types.SimpleNamespace(
            distributed=types.SimpleNamespace(is_initialized=lambda: False),
            local_devices=_unexpected_local_devices,
        ),
    )
    calls = _install_fake_nvidia_smi(monkeypatch, "3, 640\n")
    backend = _fresh_backend()

    assert backend.get_active_cuda_device_index() == 3
    assert backend.query_active_gpu_memory_mb() == pytest.approx(640.0)
    assert local_device_calls == []
    assert calls
    assert calls[0][-2:] == ["-i", "3"]


def test_query_active_gpu_memory_mb_uses_active_jax_device_when_env_is_unset(
    monkeypatch,
):
    local_devices, local_device_calls = _make_recording_cuda_device_probe(2)
    _clear_backend_env(monkeypatch)
    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_gpu_fast")
    monkeypatch.setitem(
        sys.modules,
        "jax",
        types.SimpleNamespace(local_devices=local_devices),
    )
    calls = _install_fake_nvidia_smi(monkeypatch, "2, 768\n")
    backend = _fresh_backend()

    assert backend.get_active_cuda_device_index() == 2
    assert backend.query_active_gpu_memory_mb() == pytest.approx(768.0)
    _assert_public_gpu_local_device_calls(local_device_calls)
    assert calls
    assert calls[0][-2:] == ["-i", "2"]


def test_query_active_gpu_memory_mb_prefers_imported_jax_device_over_visible_env(
    monkeypatch,
):
    local_devices, local_device_calls = _make_recording_cuda_device_probe(1)
    _clear_backend_env(monkeypatch)
    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_gpu_fast")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,1")
    monkeypatch.setitem(
        sys.modules,
        "jax",
        types.SimpleNamespace(local_devices=local_devices),
    )
    calls = _install_fake_nvidia_smi(monkeypatch, "1, 256\n")
    backend = _fresh_backend()

    assert backend.get_active_cuda_device_index() == 1
    assert backend.query_active_gpu_memory_mb() == pytest.approx(256.0)
    _assert_public_gpu_local_device_calls(local_device_calls)
    assert calls
    assert calls[0][-2:] == ["-i", "1"]


def test_chunk_tuning_autotunes_from_visible_cuda_device_before_distributed_init(
    monkeypatch,
):
    local_device_calls: list[str | None] = []

    def _unexpected_local_devices(*, backend=None):
        local_device_calls.append(backend)
        raise AssertionError(
            f"local_devices should not run before distributed init for backend={backend!r}"
        )

    _clear_backend_env(monkeypatch)
    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_gpu_fast")
    _set_distributed_init_env(monkeypatch, visible_devices="3,1")
    monkeypatch.setitem(
        sys.modules,
        "jax",
        types.SimpleNamespace(
            distributed=types.SimpleNamespace(is_initialized=lambda: False),
            local_devices=_unexpected_local_devices,
        ),
    )
    calls = _install_fake_nvidia_smi(monkeypatch, "3, 24576\n")
    backend = _fresh_backend()

    tuning = backend.get_chunk_tuning()

    assert tuning.autotuned is True
    assert tuning.autotune_source == "nvidia-smi[3]"
    assert tuning.gpu_total_memory_mb == 24576
    assert local_device_calls == []
    assert calls
    assert calls[0][-2:] == ["-i", "3"]


def test_chunk_tuning_rejects_nonpositive_gpu_memory_override(monkeypatch):
    _clear_backend_env(monkeypatch)
    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_gpu_fast")
    monkeypatch.setenv("SIMSOPT_JAX_GPU_MEMORY_TOTAL_MB", "0")
    backend = _fresh_backend()

    with pytest.raises(
        ValueError,
        match="SIMSOPT_JAX_GPU_MEMORY_TOTAL_MB must be > 0 when set",
    ):
        backend.get_chunk_tuning()


def test_transfer_guard_dense_audit_keeps_pairwise_chunk_autotuning(monkeypatch):
    _clear_backend_env(monkeypatch)
    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_gpu_fast")
    monkeypatch.setenv("SIMSOPT_JAX_GPU_MEMORY_TOTAL_MB", "24576")
    backend = _fresh_backend()

    backend.set_backend(
        "jax_gpu_fast",
        strict=True,
        transfer_guard="disallow",
        configure_runtime=False,
    )
    tuning = backend.get_chunk_tuning()

    assert tuning.chunk_policy == "performance_tuned_dense_audit"
    assert tuning.autotuned is True
    assert tuning.coil_chunk_size == 0
    assert tuning.quadrature_block_size == 0
    assert tuning.point_chunk_size == 0
    assert tuning.pairwise_penalty_chunk_size == 2048


def test_sharding_tuning_defaults_fast_mode_to_hybrid_strategy(monkeypatch):
    _clear_backend_env(monkeypatch)
    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_gpu_fast")
    backend = _fresh_backend()
    runtime = sys.modules["simsopt.backend.runtime"]
    monkeypatch.setattr(runtime, "_detect_local_jax_device_count", lambda policy: 4)

    tuning = backend.get_sharding_tuning()

    assert tuning.mode == "jax_gpu_fast"
    assert tuning.strategy == "hybrid"
    assert tuning.mesh_axis_name == "d"
    assert tuning.coil_axis_name == "coil"
    assert tuning.min_points_to_shard == 2048
    assert tuning.min_pairwise_rows_to_shard == 32
    assert tuning.min_coils_to_shard == 4
    assert tuning.device_count == 4
    assert tuning.active is True
    assert backend.get_sharding_strategy() == "hybrid"
    assert backend.should_shard_points() is True
    assert backend.should_shard_pairwise_rows() is True
    assert backend.should_shard_coil_groups() is False


def test_sharding_tuning_defaults_parity_modes_to_none(monkeypatch):
    _clear_backend_env(monkeypatch)
    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_gpu_parity")
    backend = _fresh_backend()
    runtime = sys.modules["simsopt.backend.runtime"]
    monkeypatch.setattr(runtime, "_detect_local_jax_device_count", lambda policy: 8)

    tuning = backend.get_sharding_tuning()

    assert tuning.strategy == "none"
    assert tuning.active is False
    assert backend.should_shard_points() is False


def test_sharding_tuning_respects_env_overrides(monkeypatch):
    _clear_backend_env(monkeypatch)
    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_gpu_fast")
    monkeypatch.setenv("SIMSOPT_JAX_SHARDING", "none")
    monkeypatch.setenv("SIMSOPT_JAX_SHARDING_AXIS", "pts")
    monkeypatch.setenv("SIMSOPT_JAX_COIL_SHARDING_AXIS", "coils")
    monkeypatch.setenv("SIMSOPT_JAX_MIN_POINTS_TO_SHARD", "123")
    monkeypatch.setenv("SIMSOPT_JAX_MIN_PAIRWISE_ROWS_TO_SHARD", "7")
    monkeypatch.setenv("SIMSOPT_JAX_MIN_COILS_TO_SHARD", "3")
    backend = _fresh_backend()
    runtime = sys.modules["simsopt.backend.runtime"]
    monkeypatch.setattr(runtime, "_detect_local_jax_device_count", lambda policy: 8)

    tuning = backend.get_sharding_tuning()

    assert tuning.strategy == "none"
    assert tuning.mesh_axis_name == "pts"
    assert tuning.coil_axis_name == "coils"
    assert tuning.min_points_to_shard == 123
    assert tuning.min_pairwise_rows_to_shard == 7
    assert tuning.min_coils_to_shard == 3
    assert tuning.active is False


def test_sharding_tuning_stays_separate_from_dense_audit_transfer_guard(monkeypatch):
    _clear_backend_env(monkeypatch)
    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_gpu_fast")
    backend = _fresh_backend()
    runtime = sys.modules["simsopt.backend.runtime"]
    monkeypatch.setattr(runtime, "_detect_local_jax_device_count", lambda policy: 2)

    backend.set_backend(
        "jax_gpu_fast",
        strict=True,
        transfer_guard="disallow",
        configure_runtime=False,
    )
    tuning = backend.get_sharding_tuning()

    assert tuning.strategy == "hybrid"
    assert tuning.active is True
    assert tuning.device_count == 2
    assert backend.should_shard_pairwise_rows() is True


def test_sharding_tuning_accepts_pairwise_rows_strategy(monkeypatch):
    _clear_backend_env(monkeypatch)
    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_gpu_fast")
    monkeypatch.setenv("SIMSOPT_JAX_SHARDING", "pairwise_rows")
    monkeypatch.setenv("SIMSOPT_JAX_MIN_PAIRWISE_ROWS_TO_SHARD", "9")
    backend = _fresh_backend()
    runtime = sys.modules["simsopt.backend.runtime"]
    monkeypatch.setattr(runtime, "_detect_local_jax_device_count", lambda policy: 4)

    tuning = backend.get_sharding_tuning()

    assert tuning.strategy == "pairwise_rows"
    assert tuning.min_pairwise_rows_to_shard == 9
    assert backend.should_shard_points() is False
    assert backend.should_shard_pairwise_rows() is True


def test_sharding_tuning_accepts_coil_groups_strategy(monkeypatch):
    _clear_backend_env(monkeypatch)
    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_gpu_fast")
    monkeypatch.setenv("SIMSOPT_JAX_SHARDING", "coil_groups")
    monkeypatch.setenv("SIMSOPT_JAX_COIL_SHARDING_AXIS", "coil_batch")
    monkeypatch.setenv("SIMSOPT_JAX_MIN_COILS_TO_SHARD", "5")
    backend = _fresh_backend()
    runtime = sys.modules["simsopt.backend.runtime"]
    monkeypatch.setattr(runtime, "_detect_local_jax_device_count", lambda policy: 4)

    tuning = backend.get_sharding_tuning()

    assert tuning.strategy == "coil_groups"
    assert tuning.coil_axis_name == "coil_batch"
    assert tuning.min_coils_to_shard == 5
    assert backend.should_shard_points() is False
    assert backend.should_shard_pairwise_rows() is False
    assert backend.should_shard_coil_groups() is True


def test_sharding_tuning_rejects_points_coils_strategy(monkeypatch):
    _clear_backend_env(monkeypatch)
    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_gpu_fast")
    monkeypatch.setenv("SIMSOPT_JAX_SHARDING", "points_coils")
    backend = _fresh_backend()
    with pytest.raises(ValueError, match="SIMSOPT_JAX_SHARDING='points_coils'"):
        backend.get_sharding_tuning()


def test_distributed_runtime_config_reads_env(monkeypatch):
    _clear_backend_env(monkeypatch)
    _set_distributed_init_env(
        monkeypatch,
        process_id="2",
        local_device_ids="3,5",
    )
    backend = _fresh_backend()

    config = backend.get_distributed_runtime_config()

    assert config.enabled is True
    assert config.initialized is False
    assert config.coordinator_address == "127.0.0.1:12345"
    assert config.num_processes == 4
    assert config.process_id == 2
    assert config.local_device_ids == (3, 5)


def test_distributed_runtime_config_rejects_invalid_process_ids(monkeypatch):
    _clear_backend_env(monkeypatch)
    _set_distributed_init_env(monkeypatch, num_processes="2", process_id="2")
    backend = _fresh_backend()

    with pytest.raises(
        ValueError,
        match="SIMSOPT_JAX_PROCESS_ID=2 must be smaller than SIMSOPT_JAX_NUM_PROCESSES=2",
    ):
        backend.get_distributed_runtime_config()


def test_distributed_runtime_config_self_heals_after_external_init(monkeypatch):
    _clear_backend_env(monkeypatch)
    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_gpu_fast")
    _set_distributed_init_env(monkeypatch)
    distributed_state = {"initialized": False}
    monkeypatch.setitem(
        sys.modules,
        "jax",
        types.SimpleNamespace(
            distributed=types.SimpleNamespace(
                is_initialized=lambda: distributed_state["initialized"]
            )
        ),
    )

    backend = _fresh_backend()
    runtime = sys.modules["simsopt.backend.runtime"]
    monkeypatch.setattr(runtime, "_detect_local_jax_device_count", lambda policy: 2)
    monkeypatch.setattr(runtime, "_detect_global_jax_device_count", lambda policy: 8)

    pre_config = backend.get_distributed_runtime_config()
    pre_tuning = backend.get_sharding_tuning()

    assert pre_config.initialized is False
    assert pre_tuning.distributed_initialized is False
    assert pre_tuning.device_count == 2

    distributed_state["initialized"] = True

    post_config = backend.get_distributed_runtime_config()
    post_tuning = backend.get_sharding_tuning()

    assert post_config.initialized is True
    assert post_tuning.distributed_initialized is True
    assert post_tuning.device_count == 8


def test_maybe_initialize_distributed_jax_updates_sharding_device_counts(monkeypatch):
    _clear_backend_env(monkeypatch)
    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_gpu_fast")
    _set_distributed_init_env(monkeypatch)

    calls: list[dict[str, object]] = []
    fake_distributed = types.SimpleNamespace(
        is_initialized=lambda: False,
        initialize=lambda **kwargs: calls.append(kwargs),
    )
    monkeypatch.setitem(
        sys.modules,
        "jax",
        types.SimpleNamespace(distributed=fake_distributed),
    )

    backend = _fresh_backend()
    runtime = sys.modules["simsopt.backend.runtime"]
    monkeypatch.setattr(runtime, "_detect_local_jax_device_count", lambda policy: 2)
    monkeypatch.setattr(runtime, "_detect_global_jax_device_count", lambda policy: 8)

    config = backend.maybe_initialize_distributed_jax()
    tuning = backend.get_sharding_tuning()

    assert calls == [
        {
            "coordinator_address": "127.0.0.1:12345",
            "num_processes": 4,
            "process_id": 1,
            "local_device_ids": None,
        }
    ]
    assert config.initialized is True
    assert tuning.distributed_initialized is True
    assert tuning.local_device_count == 2
    assert tuning.device_count == 8


def test_maybe_initialize_distributed_jax_invalidates_preinit_chunk_caches(
    monkeypatch,
):
    def _fake_run(cmd, *, check, capture_output, text):
        del check, capture_output, text
        calls.append(cmd)
        index = cmd[-1]
        stdout_by_index = {
            "3": "3, 24576\n",
            "1": "1, 8192\n",
        }
        return types.SimpleNamespace(stdout=stdout_by_index[index])

    def _is_initialized():
        return distributed_state["initialized"]

    def _initialize(**kwargs):
        del kwargs
        distributed_state["initialized"] = True

    def _local_devices(*, backend=None):
        if not distributed_state["initialized"]:
            raise AssertionError(
                f"local_devices should not run before distributed init for backend={backend!r}"
            )
        return [_make_fake_cuda_device(1)]

    _clear_backend_env(monkeypatch)
    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_gpu_fast")
    _set_distributed_init_env(monkeypatch, visible_devices="3,1")
    distributed_state = {"initialized": False}
    monkeypatch.setitem(
        sys.modules,
        "jax",
        types.SimpleNamespace(
            distributed=types.SimpleNamespace(
                is_initialized=_is_initialized,
                initialize=_initialize,
            ),
            local_devices=_local_devices,
        ),
    )
    calls: list[list[str]] = []
    monkeypatch.setattr("subprocess.run", _fake_run)
    backend = _fresh_backend()

    pre_chunk = backend.get_chunk_tuning()
    pre_field = backend.get_field_kernel_tuning()
    config = backend.maybe_initialize_distributed_jax()
    post_chunk = backend.get_chunk_tuning()
    post_field = backend.get_field_kernel_tuning()

    assert pre_chunk.gpu_total_memory_mb == 24576
    assert pre_chunk.autotune_source == "nvidia-smi[3]"
    assert pre_field.coil_chunk_size == 128
    assert config.initialized is True
    assert backend.get_active_cuda_device_index() == 1
    assert backend.query_active_gpu_memory_mb() == pytest.approx(8192.0)
    assert post_chunk.gpu_total_memory_mb == 8192
    assert post_chunk.autotune_source == "nvidia-smi[1]"
    assert post_field.coil_chunk_size == 32
    assert [cmd[-2:] for cmd in calls] == [["-i", "3"], ["-i", "1"], ["-i", "1"]]


def _assert_distributed_runtime_initializes_before_device_put(
    monkeypatch,
    invoke,
):
    _clear_backend_env(monkeypatch)
    backend = _fresh_backend()
    math_utils = importlib.import_module("simsopt.jax_core._math_utils")

    backend.set_backend("jax_gpu_fast", configure_runtime=False)
    _set_distributed_init_env(monkeypatch)
    backend.invalidate_backend_cache()

    distributed_state = {"initialized": False}
    initialize_calls: list[dict[str, object]] = []
    expected_initialize_call = {
        "coordinator_address": "127.0.0.1:12345",
        "num_processes": 4,
        "process_id": 1,
        "local_device_ids": None,
    }

    def _initialize(
        *,
        coordinator_address,
        num_processes,
        process_id,
        local_device_ids,
    ):
        initialize_calls.append(
            {
                "coordinator_address": coordinator_address,
                "num_processes": num_processes,
                "process_id": process_id,
                "local_device_ids": local_device_ids,
            }
        )
        distributed_state["initialized"] = True

    def _device_put(value):
        assert distributed_state["initialized"] is True
        return np.asarray(value)

    monkeypatch.setattr(
        math_utils.jax,
        "distributed",
        types.SimpleNamespace(
            is_initialized=lambda: distributed_state["initialized"],
            initialize=_initialize,
        ),
        raising=False,
    )
    monkeypatch.setattr(math_utils.jax, "device_put", _device_put)

    value = invoke(math_utils)

    np.testing.assert_allclose(value, np.array([1.0, 2.0], dtype=np.float64))
    assert initialize_calls == [expected_initialize_call]


def test_as_jax_array_initializes_distributed_runtime_before_device_put(monkeypatch):
    _assert_distributed_runtime_initializes_before_device_put(
        monkeypatch,
        lambda math_utils: math_utils.as_jax_float64([1.0, 2.0]),
    )


def test_as_runtime_array_initializes_distributed_runtime_before_device_put(monkeypatch):
    _assert_distributed_runtime_initializes_before_device_put(
        monkeypatch,
        lambda math_utils: math_utils.as_runtime_float64(
            [1.0, 2.0],
            reference=np.asarray([0.0], dtype=np.float64),
        ),
    )


def test_curveobjectives_as_jax_float64_initializes_distributed_runtime_before_device_put(
    monkeypatch,
):
    curveobjectives = importlib.import_module("simsopt.geo.curveobjectives")

    _assert_distributed_runtime_initializes_before_device_put(
        monkeypatch,
        lambda _math_utils: curveobjectives._as_jax_float64(
            np.array([1.0, 2.0], dtype=np.float64)
        ),
    )


def test_framedcurve_as_jax_float64_array_initializes_distributed_runtime_before_device_put(
    monkeypatch,
):
    framedcurve = importlib.import_module("simsopt.geo.framedcurve")

    _assert_distributed_runtime_initializes_before_device_put(
        monkeypatch,
        lambda _math_utils: framedcurve._as_jax_float64_array(
            np.array([1.0, 2.0], dtype=np.float64)
        ),
    )


@pytest.mark.parametrize(
    ("module_name", "helper_name"),
    (
        ("simsopt.jax_core._math_utils", "as_runtime_float64"),
        ("simsopt.geo.curve", "_as_runtime_float64_ref"),
        ("simsopt.geo.curvexyzfourier", "_as_runtime_float64"),
    ),
)
def test_runtime_float64_helpers_keep_traced_references_on_device(
    module_name,
    helper_name,
):
    jax_module = _require_jax()
    jnp_module = importlib.import_module("jax.numpy")
    module = importlib.import_module(module_name)
    helper = getattr(module, helper_name)
    captured: dict[str, bool] = {}

    def traced(reference):
        value = helper(
            np.array([1.0, 2.0], dtype=np.float64),
            reference=reference,
        )
        captured["is_ndarray"] = isinstance(value, np.ndarray)
        captured["has_aval"] = hasattr(value, "aval")
        return jnp_module.sum(value) + reference

    jax_module.eval_shape(
        traced,
        jnp_module.asarray(3.0, dtype=jnp_module.float64),
    )

    assert captured == {
        "is_ndarray": False,
        "has_aval": True,
    }


def test_explicit_current_mode_policy_preserves_strict_state(monkeypatch):
    _clear_backend_env(monkeypatch)
    backend = _fresh_backend()

    backend.set_backend(
        "jax_gpu_parity",
        strict=True,
        configure_runtime=False,
    )

    implicit_policy = backend.get_backend_policy()
    explicit_policy = backend.get_backend_policy("jax_gpu_parity")

    assert implicit_policy.strict is True
    assert explicit_policy.strict is True


def test_native_cpu_mode_is_not_eager_jax_import(monkeypatch):
    _clear_backend_env(monkeypatch)
    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "native_cpu")
    backend = _fresh_backend()

    assert backend.should_eagerly_configure_jax() is False


def test_jax_mode_requests_eager_jax_import(monkeypatch):
    _clear_backend_env(monkeypatch)
    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_cpu_parity")
    backend = _fresh_backend()

    assert backend.should_eagerly_configure_jax() is True


def test_native_cpu_guardrail_env_does_not_trigger_eager_jax_import(monkeypatch):
    _clear_backend_env(monkeypatch)
    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "native_cpu")
    monkeypatch.setenv("SIMSOPT_JAX_DEBUG_NANS", "1")
    backend = _fresh_backend()

    assert backend.should_eagerly_configure_jax() is False


def test_jax_modes_default_compilation_cache_dir(monkeypatch):
    _clear_backend_env(monkeypatch)
    backend = _fresh_backend()
    runtime_module = sys.modules["simsopt.backend.runtime"]
    fake_home = Path("/tmp/simsopt-jax-cache-home")
    monkeypatch.setattr(runtime_module.Path, "home", lambda: fake_home)

    config = backend.set_backend("jax_cpu_parity", configure_runtime=False)
    policy = backend.get_backend_policy("jax_cpu_parity")
    expected = str(fake_home / ".cache" / "simsopt-jax-xla")

    assert config.compilation_cache_dir == expected
    assert backend.get_compilation_cache_dir() == expected
    assert policy.compilation_cache_dir == expected


def test_native_cpu_mode_keeps_compilation_cache_disabled(monkeypatch):
    _clear_backend_env(monkeypatch)
    backend = _fresh_backend()
    runtime_module = sys.modules["simsopt.backend.runtime"]
    fake_home = Path("/tmp/simsopt-jax-cache-home")
    monkeypatch.setattr(runtime_module.Path, "home", lambda: fake_home)

    config = backend.set_backend("native_cpu", configure_runtime=False)
    policy = backend.get_backend_policy("native_cpu")

    assert config.compilation_cache_dir is None
    assert backend.get_compilation_cache_dir() is None
    assert policy.compilation_cache_dir is None


def test_explicit_compilation_cache_env_overrides_default(monkeypatch):
    _clear_backend_env(monkeypatch)
    monkeypatch.setenv(
        "SIMSOPT_JAX_COMPILATION_CACHE_DIR",
        "/custom/cache/path",
    )
    backend = _fresh_backend()
    runtime_module = sys.modules["simsopt.backend.runtime"]
    fake_home = Path("/tmp/simsopt-jax-cache-home")
    monkeypatch.setattr(runtime_module.Path, "home", lambda: fake_home)

    config = backend.set_backend("jax_gpu_parity", configure_runtime=False)
    policy = backend.get_backend_policy("jax_gpu_parity")

    assert config.compilation_cache_dir == "/custom/cache/path"
    assert backend.get_compilation_cache_dir() == "/custom/cache/path"
    assert policy.compilation_cache_dir == "/custom/cache/path"


def test_apply_jax_runtime_config_applies_fast_mode_transfer_guard(monkeypatch):
    _clear_backend_env(monkeypatch)
    backend = _fresh_backend()
    runtime_module = sys.modules["simsopt.backend.runtime"]
    fake_home = Path("/tmp/simsopt-jax-cache-home")
    monkeypatch.setattr(runtime_module.Path, "home", lambda: fake_home)
    calls: list[tuple[str, object]] = []
    fake_jax = types.SimpleNamespace(
        config=types.SimpleNamespace(
            update=lambda name, value: calls.append((name, value))
        )
    )
    monkeypatch.setitem(sys.modules, "jax", fake_jax)

    backend.set_backend("jax_gpu_fast", configure_runtime=False)
    backend.apply_jax_runtime_config()

    assert ("jax_platforms", "cuda,cpu") in calls
    assert ("jax_enable_x64", True) in calls
    assert ("jax_debug_nans", False) in calls
    assert ("jax_transfer_guard", "log") in calls
    assert (
        "jax_compilation_cache_dir",
        str(fake_home / ".cache" / "simsopt-jax-xla"),
    ) in calls


def test_apply_jax_runtime_config_applies_metal_smoke_mode(monkeypatch):
    _clear_backend_env(monkeypatch)
    backend = _fresh_backend()
    runtime_module = sys.modules["simsopt.backend.runtime"]
    fake_home = Path("/tmp/simsopt-jax-cache-home")
    monkeypatch.setattr(runtime_module.Path, "home", lambda: fake_home)
    calls: list[tuple[str, object]] = []
    fake_jax = types.SimpleNamespace(
        config=types.SimpleNamespace(
            update=lambda name, value: calls.append((name, value))
        )
    )
    monkeypatch.setitem(sys.modules, "jax", fake_jax)

    backend.set_backend("jax_metal_smoke", configure_runtime=False)
    backend.apply_jax_runtime_config()

    assert ("jax_platforms", "METAL") in calls
    assert ("jax_enable_x64", False) in calls
    assert ("jax_debug_nans", False) in calls
    assert ("jax_transfer_guard", "log") in calls
    assert (
        "jax_compilation_cache_dir",
        str(fake_home / ".cache" / "simsopt-jax-xla"),
    ) in calls


def test_apply_jax_runtime_config_warns_without_cuda_determinism_flag(
    monkeypatch,
):
    _clear_backend_env(monkeypatch)
    backend = _fresh_backend_with_fake_runtime_home(monkeypatch)
    monkeypatch.delenv("XLA_FLAGS", raising=False)
    calls: list[tuple[str, object]] = []
    _install_fake_jax(monkeypatch, calls=calls)

    backend.set_backend("jax_gpu_parity", configure_runtime=False)
    with pytest.warns(RuntimeWarning, match="XLA_FLAGS does not enable"):
        backend.apply_jax_runtime_config()

    assert ("jax_platforms", "cuda,cpu") in calls


@pytest.mark.parametrize(
    "xla_flag",
    (
        "--xla_gpu_deterministic_ops",
        "--xla_gpu_deterministic_ops=true",
        "--xla_gpu_exclude_nondeterministic_ops",
        "--xla_gpu_exclude_nondeterministic_ops=true",
    ),
)
def test_apply_jax_runtime_config_accepts_supported_cuda_determinism_flags(
    monkeypatch,
    xla_flag,
):
    _clear_backend_env(monkeypatch)
    backend = _fresh_backend_with_fake_runtime_home(monkeypatch)
    monkeypatch.setenv("XLA_FLAGS", xla_flag)
    calls: list[tuple[str, object]] = []
    _install_fake_jax(monkeypatch, calls=calls)

    backend.set_backend("jax_gpu_parity", configure_runtime=False)
    backend.apply_jax_runtime_config()

    assert ("jax_platforms", "cuda,cpu") in calls


def test_apply_jax_runtime_config_warns_on_initialized_backend_mismatch(monkeypatch):
    _clear_backend_env(monkeypatch)
    backend = _fresh_backend()
    runtime_module = sys.modules["simsopt.backend.runtime"]
    fake_home = Path("/tmp/simsopt-jax-cache-home")
    monkeypatch.setattr(runtime_module.Path, "home", lambda: fake_home)
    calls: list[tuple[str, object]] = []
    fake_jax = types.SimpleNamespace(
        config=types.SimpleNamespace(
            update=lambda name, value: calls.append((name, value))
        ),
        default_backend=lambda: "cpu",
    )
    monkeypatch.setitem(sys.modules, "jax", fake_jax)

    backend.set_backend("jax_gpu_fast", configure_runtime=False)
    with pytest.warns(RuntimeWarning, match="active JAX default backend is 'cpu'"):
        backend.apply_jax_runtime_config()

    assert ("jax_platforms", "cuda,cpu") in calls


def test_restore_loaded_jax_runtime_config_restores_nullable_fields_to_none():
    jax_module = _require_jax()
    snapshot = {
        "jax_transfer_guard": None,
        "jax_platforms": None,
        "jax_compilation_cache_dir": None,
    }
    mutated = {
        "jax_transfer_guard": "disallow",
        "jax_platforms": "cpu",
        "jax_compilation_cache_dir": "/tmp/simsopt-jax-restore-cache",
    }
    original = {
        name: getattr(jax_module.config, name)
        for name in snapshot
    }
    try:
        for name, value in mutated.items():
            jax_module.config.update(name, value)

        _restore_loaded_jax_runtime_config(snapshot)

        for name in snapshot:
            assert getattr(jax_module.config, name) is None
    finally:
        _restore_loaded_jax_runtime_config(original)


def test_apply_jax_runtime_config_raises_on_initialized_backend_mismatch_in_strict_mode(
    monkeypatch,
):
    _clear_backend_env(monkeypatch)
    backend = _fresh_backend()
    runtime_module = sys.modules["simsopt.backend.runtime"]
    fake_home = Path("/tmp/simsopt-jax-cache-home")
    monkeypatch.setattr(runtime_module.Path, "home", lambda: fake_home)
    fake_jax = types.SimpleNamespace(
        config=types.SimpleNamespace(update=lambda name, value: None),
        default_backend=lambda: "cpu",
    )
    monkeypatch.setitem(sys.modules, "jax", fake_jax)

    backend.set_backend("jax_gpu_fast", strict=True, configure_runtime=False)
    with pytest.raises(RuntimeError, match="active JAX default backend is 'cpu'"):
        backend.apply_jax_runtime_config()


def test_apply_jax_runtime_config_raises_without_cuda_determinism_flag_in_strict_mode(
    monkeypatch,
):
    _clear_backend_env(monkeypatch)
    backend = _fresh_backend_with_fake_runtime_home(monkeypatch)
    monkeypatch.delenv("XLA_FLAGS", raising=False)
    _install_fake_jax(monkeypatch)

    backend.set_backend("jax_gpu_parity", strict=True, configure_runtime=False)
    with pytest.raises(RuntimeError, match="XLA_FLAGS does not enable"):
        backend.apply_jax_runtime_config()


def test_apply_jax_runtime_config_rejects_last_override_to_disabled_flag(
    monkeypatch,
):
    _clear_backend_env(monkeypatch)
    backend = _fresh_backend_with_fake_runtime_home(monkeypatch)
    monkeypatch.setenv(
        "XLA_FLAGS",
        "--xla_gpu_deterministic_ops=true --xla_gpu_deterministic_ops=false",
    )
    _install_fake_jax(monkeypatch)

    backend.set_backend("jax_gpu_parity", strict=True, configure_runtime=False)
    with pytest.raises(RuntimeError, match="XLA_FLAGS does not enable"):
        backend.apply_jax_runtime_config()


def test_apply_jax_runtime_config_accepts_last_override_to_enabled_flag(
    monkeypatch,
):
    _clear_backend_env(monkeypatch)
    backend = _fresh_backend_with_fake_runtime_home(monkeypatch)
    monkeypatch.setenv(
        "XLA_FLAGS",
        "--xla_gpu_deterministic_ops=false --xla_gpu_deterministic_ops=true",
    )
    calls: list[tuple[str, object]] = []
    _install_fake_jax(monkeypatch, calls=calls)

    backend.set_backend("jax_gpu_parity", strict=True, configure_runtime=False)
    backend.apply_jax_runtime_config()

    assert ("jax_platforms", "cuda,cpu") in calls


def test_ensure_gpu_determinism_xla_flag_preserves_unrelated_xla_flags():
    env = {
        "XLA_FLAGS": "--xla_gpu_cuda_data_dir=/tmp/cuda --other-flag=1",
    }

    ensure_gpu_determinism_xla_flag(env)

    assert env["XLA_FLAGS"].split() == [
        "--xla_gpu_cuda_data_dir=/tmp/cuda",
        "--other-flag=1",
        "--xla_gpu_deterministic_ops=true",
    ]


def test_ensure_gpu_determinism_xla_flag_replaces_disabled_value():
    env = {
        "XLA_FLAGS": "--xla_gpu_deterministic_ops=false --xla_gpu_cuda_data_dir=/tmp/cuda",
    }

    ensure_gpu_determinism_xla_flag(env)

    assert env["XLA_FLAGS"].split() == [
        "--xla_gpu_cuda_data_dir=/tmp/cuda",
        "--xla_gpu_deterministic_ops=true",
    ]


def test_ensure_gpu_determinism_xla_flag_preserves_enabled_official_flag():
    env = {
        "XLA_FLAGS": "--xla_gpu_exclude_nondeterministic_ops=true --other-flag=1",
    }

    ensure_gpu_determinism_xla_flag(env)

    assert env["XLA_FLAGS"].split() == [
        "--xla_gpu_exclude_nondeterministic_ops=true",
        "--other-flag=1",
    ]


def test_ensure_gpu_determinism_xla_flag_rewrites_disabled_last_override():
    env = {
        "XLA_FLAGS": "--xla_gpu_deterministic_ops=true --xla_gpu_deterministic_ops=false --other-flag=1",
    }

    ensure_gpu_determinism_xla_flag(env)

    assert env["XLA_FLAGS"].split() == [
        "--other-flag=1",
        "--xla_gpu_deterministic_ops=true",
    ]


def test_strict_fallback_helper_ignores_native_cpu(monkeypatch):
    _clear_backend_env(monkeypatch)
    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "native_cpu")
    monkeypatch.setenv("SIMSOPT_BACKEND_STRICT", "1")
    backend = _fresh_backend()

    backend.raise_if_strict_jax_fallback(
        component="test-component",
        detail="a fallback path",
    )


def test_strict_fallback_helper_rejects_jax_mode(monkeypatch):
    _clear_backend_env(monkeypatch)
    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_gpu_parity")
    monkeypatch.setenv("SIMSOPT_BACKEND_STRICT", "1")
    backend = _fresh_backend()

    with pytest.raises(RuntimeError, match="strict=True"):
        backend.raise_if_strict_jax_fallback(
            component="test-component",
            detail="a fallback path",
        )


def test_warn_fallback_helper_ignores_native_cpu(monkeypatch):
    _clear_backend_env(monkeypatch)
    backend = _fresh_backend()

    caught = _capture_fallback_warnings(
        lambda: backend.warn_if_jax_fallback(
            component="test-component",
            detail="a fallback path",
        )
    )

    assert caught == []


def test_warn_fallback_helper_emits_once_in_non_strict_jax_mode(monkeypatch):
    _clear_backend_env(monkeypatch)
    backend = _fresh_backend()
    backend.set_backend("jax_gpu_parity", configure_runtime=False)

    caught = _capture_fallback_warnings(
        lambda: (
            backend.warn_if_jax_fallback(
                component="test-component",
                detail="a fallback path",
            ),
            backend.warn_if_jax_fallback(
                component="test-component",
                detail="a fallback path",
            ),
        )
    )

    assert len(caught) == 1
    assert issubclass(caught[0].category, RuntimeWarning)
    assert "legacy adapter seam" in str(caught[0].message)


def test_warn_fallback_helper_ignores_strict_jax_mode(monkeypatch):
    _clear_backend_env(monkeypatch)
    backend = _fresh_backend()
    backend.set_backend("jax_gpu_parity", strict=True, configure_runtime=False)

    caught = _capture_fallback_warnings(
        lambda: backend.warn_if_jax_fallback(
            component="test-component",
            detail="a fallback path",
        )
    )

    assert caught == []


def test_backend_state_restoration_is_explicit_without_test_order(monkeypatch):
    env_snapshot = {name: os.environ.get(name) for name in _BACKEND_ENV_VARS}
    try:
        _clear_backend_env(monkeypatch)
        backend = _fresh_backend()
        backend.set_backend("jax_gpu_parity", strict=True, configure_runtime=False)

        assert os.environ["SIMSOPT_BACKEND_MODE"] == "jax_gpu_parity"
        assert os.environ["SIMSOPT_BACKEND_STRICT"] == "1"
        assert backend.get_backend_config().strict is True
    finally:
        _restore_backend_runtime_env(env_snapshot)

    backend.invalidate_backend_cache()
    config = backend.get_backend_config()

    assert config.mode == "native_cpu"
    assert config.strict is False


def test_distributed_backend_state_restoration_is_explicit_without_test_order(
    monkeypatch,
):
    env_snapshot = {name: os.environ.get(name) for name in _BACKEND_ENV_VARS}
    try:
        _clear_backend_env(monkeypatch)
        backend = _fresh_backend()
        backend.set_backend("jax_gpu_fast", configure_runtime=False)
        monkeypatch.setenv("SIMSOPT_JAX_DISTRIBUTED_INIT", "1")
        monkeypatch.setenv("SIMSOPT_JAX_COORDINATOR_ADDRESS", "127.0.0.1:12345")
        monkeypatch.setenv("SIMSOPT_JAX_NUM_PROCESSES", "4")
        monkeypatch.setenv("SIMSOPT_JAX_PROCESS_ID", "1")
        backend.invalidate_backend_cache()

        config = backend.get_distributed_runtime_config()

        assert backend.get_backend_config().mode == "jax_gpu_fast"
        assert config.enabled is True
        assert config.initialized is False
    finally:
        _restore_backend_runtime_env(env_snapshot)

    backend.invalidate_backend_cache()
    restored_config = backend.get_backend_config()
    tuning = backend.get_sharding_tuning()

    assert restored_config.mode == "native_cpu"
    assert restored_config.strict is False
    assert tuning.mode == "native_cpu"
    assert tuning.distributed_initialized is False


def test_backend_module_guard_restores_original_backend_modules():
    snapshot = _snapshot_backend_modules()
    runtime_module = sys.modules.get("simsopt.backend.runtime")

    reloaded_backend = _fresh_backend()

    assert sys.modules["simsopt.backend"] is reloaded_backend
    if runtime_module is not None:
        assert sys.modules["simsopt.backend.runtime"] is not runtime_module

    _restore_backend_module_snapshot(snapshot)
    for name in _BACKEND_MODULE_NAMES:
        expected = snapshot[name]
        if expected is _MISSING_MODULE:
            assert name not in sys.modules
        else:
            assert sys.modules[name] is expected


def test_force_x64_rejects_silent_config_update_failure():
    calls = []
    fake_jax = types.SimpleNamespace(
        config=types.SimpleNamespace(
            jax_enable_x64=False,
            update=lambda name, value: calls.append((name, value)),
        )
    )

    with pytest.raises(RuntimeError, match="requires jax_enable_x64=True"):
        _force_x64(fake_jax)

    assert calls == [("jax_enable_x64", True)]
