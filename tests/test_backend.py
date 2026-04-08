from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys
import types
import warnings

import pytest

_BACKEND_ENV_VARS = (
    "SIMSOPT_BACKEND_MODE",
    "SIMSOPT_BACKEND_STRICT",
    "SIMSOPT_JAX_DEBUG_NANS",
    "SIMSOPT_JAX_TRANSFER_GUARD",
    "SIMSOPT_JAX_COMPILATION_CACHE_DIR",
    "SIMSOPT_JAX_COIL_CHUNK_SIZE",
    "SIMSOPT_JAX_QUADRATURE_BLOCK_SIZE",
    "SIMSOPT_JAX_PENALTY_POINT_CHUNK_SIZE",
    "SIMSOPT_JAX_CHUNK_AUTOTUNE",
    "SIMSOPT_JAX_GPU_MEMORY_TOTAL_MB",
    "SIMSOPT_JAX_SHARDING",
    "SIMSOPT_JAX_SHARDING_AXIS",
    "SIMSOPT_JAX_MIN_POINTS_TO_SHARD",
    "SIMSOPT_JAX_MIN_PAIRWISE_ROWS_TO_SHARD",
    "SIMSOPT_JAX_DISTRIBUTED_INIT",
    "SIMSOPT_JAX_COORDINATOR_ADDRESS",
    "SIMSOPT_JAX_NUM_PROCESSES",
    "SIMSOPT_JAX_PROCESS_ID",
    "SIMSOPT_JAX_LOCAL_DEVICE_IDS",
    "SIMSOPT_BACKEND",
    "STAGE2_BACKEND",
    "SIMSOPT_JAX_PLATFORM",
    "SIMSOPT_JAX_BACKEND",
    "JAX_PLATFORMS",
    "CUDA_VISIBLE_DEVICES",
)


def _fresh_backend():
    package_root = Path(__file__).resolve().parents[1] / "src" / "simsopt"
    sys.modules.pop("simsopt.backend.runtime", None)
    sys.modules.pop("simsopt.backend", None)
    sys.modules.pop("simsopt", None)
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


def _clear_backend_env(monkeypatch) -> None:
    for name in _BACKEND_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def _disable_chunk_autotune(monkeypatch) -> None:
    monkeypatch.setenv("SIMSOPT_JAX_CHUNK_AUTOTUNE", "0")


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
    assert os.environ["SIMSOPT_BACKEND_MODE"] == mode
    assert os.environ["SIMSOPT_BACKEND_STRICT"] == ("1" if strict else "0")
    assert os.environ["SIMSOPT_BACKEND"] == backend_name
    assert os.environ["STAGE2_BACKEND"] == backend_name
    assert os.environ["SIMSOPT_JAX_PLATFORM"] == platform
    assert os.environ["SIMSOPT_JAX_BACKEND"] == platform
    assert os.environ["JAX_PLATFORMS"] == platform


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
    monkeypatch.setenv("SIMSOPT_JAX_PLATFORM", "cuda")
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
    callback_first.__qualname__ = "_clear_hidden_grouped_fallback_warning_cache"
    callback_second.__module__ = "simsopt.geo.boozersurface_jax"
    callback_second.__qualname__ = "_clear_hidden_grouped_fallback_warning_cache"

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


def _assert_gpu_chunk_autotune_probe(monkeypatch, *, env_value, expected_index, stdout, fake_jax=None):
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
    assert tuning.autotune_source == f"nvidia-smi[{expected_index}]"
    assert tuning.gpu_total_memory_mb == int(stdout.split(",", 1)[1].strip())
    assert calls
    assert calls[0][-2:] == ["-i", str(expected_index)]


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
        expected_index=3,
        stdout="3, 16384\n",
    )


def test_chunk_tuning_autotunes_from_active_jax_cuda_device(monkeypatch):
    local_devices, local_device_calls = _make_recording_cuda_device_probe(2)
    _assert_gpu_chunk_autotune_probe(
        monkeypatch,
        env_value=None,
        expected_index=2,
        stdout="2, 24576\n",
        fake_jax=types.SimpleNamespace(local_devices=local_devices),
    )
    _assert_public_gpu_local_device_calls(local_device_calls)


def test_chunk_tuning_prefers_imported_jax_cuda_device_over_visible_env(monkeypatch):
    local_devices, local_device_calls = _make_recording_cuda_device_probe(1)
    _assert_gpu_chunk_autotune_probe(
        monkeypatch,
        env_value="3,1",
        expected_index=1,
        stdout="1, 24576\n",
        fake_jax=types.SimpleNamespace(local_devices=local_devices),
    )
    _assert_public_gpu_local_device_calls(local_device_calls)


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


def test_query_active_gpu_memory_mb_uses_active_jax_device_when_env_is_unset(monkeypatch):
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
    assert tuning.min_points_to_shard == 2048
    assert tuning.min_pairwise_rows_to_shard == 32
    assert tuning.device_count == 4
    assert tuning.active is True
    assert backend.get_sharding_strategy() == "hybrid"
    assert backend.should_shard_points() is True
    assert backend.should_shard_pairwise_rows() is True


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
    monkeypatch.setenv("SIMSOPT_JAX_MIN_POINTS_TO_SHARD", "123")
    monkeypatch.setenv("SIMSOPT_JAX_MIN_PAIRWISE_ROWS_TO_SHARD", "7")
    backend = _fresh_backend()
    runtime = sys.modules["simsopt.backend.runtime"]
    monkeypatch.setattr(runtime, "_detect_local_jax_device_count", lambda policy: 8)

    tuning = backend.get_sharding_tuning()

    assert tuning.strategy == "none"
    assert tuning.mesh_axis_name == "pts"
    assert tuning.min_points_to_shard == 123
    assert tuning.min_pairwise_rows_to_shard == 7
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


def test_jax_modes_do_not_enable_compilation_cache_without_opt_in(monkeypatch):
    _clear_backend_env(monkeypatch)
    backend = _fresh_backend()

    config = backend.set_backend("jax_cpu_parity", configure_runtime=False)
    policy = backend.get_backend_policy("jax_cpu_parity")

    assert config.compilation_cache_dir is None
    assert backend.get_compilation_cache_dir() is None
    assert policy.compilation_cache_dir is None


def test_apply_jax_runtime_config_applies_fast_mode_transfer_guard(monkeypatch):
    _clear_backend_env(monkeypatch)
    backend = _fresh_backend()
    calls: list[tuple[str, object]] = []
    fake_jax = types.SimpleNamespace(
        config=types.SimpleNamespace(
            update=lambda name, value: calls.append((name, value))
        )
    )
    monkeypatch.setitem(sys.modules, "jax", fake_jax)

    backend.set_backend("jax_gpu_fast", configure_runtime=False)
    backend.apply_jax_runtime_config()

    assert ("jax_platforms", "cuda") in calls
    assert ("jax_enable_x64", True) in calls
    assert ("jax_debug_nans", False) in calls
    assert ("jax_transfer_guard", "log") in calls


def test_apply_jax_runtime_config_warns_on_initialized_backend_mismatch(monkeypatch):
    _clear_backend_env(monkeypatch)
    backend = _fresh_backend()
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

    assert ("jax_platforms", "cuda") in calls


def test_apply_jax_runtime_config_raises_on_initialized_backend_mismatch_in_strict_mode(
    monkeypatch,
):
    _clear_backend_env(monkeypatch)
    backend = _fresh_backend()
    fake_jax = types.SimpleNamespace(
        config=types.SimpleNamespace(update=lambda name, value: None),
        default_backend=lambda: "cpu",
    )
    monkeypatch.setitem(sys.modules, "jax", fake_jax)

    backend.set_backend("jax_gpu_fast", strict=True, configure_runtime=False)
    with pytest.raises(RuntimeError, match="active JAX default backend is 'cpu'"):
        backend.apply_jax_runtime_config()


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
