from __future__ import annotations

import importlib
import os

import pytest

_BACKEND_ENV_VARS = (
    "SIMSOPT_BACKEND_MODE",
    "SIMSOPT_BACKEND_STRICT",
    "SIMSOPT_JAX_DEBUG_NANS",
    "SIMSOPT_JAX_TRANSFER_GUARD",
    "SIMSOPT_JAX_COMPILATION_CACHE_DIR",
    "SIMSOPT_JAX_COIL_CHUNK_SIZE",
    "SIMSOPT_JAX_QUADRATURE_BLOCK_SIZE",
    "SIMSOPT_BACKEND",
    "STAGE2_BACKEND",
    "SIMSOPT_JAX_PLATFORM",
    "SIMSOPT_JAX_BACKEND",
    "JAX_PLATFORMS",
)


def _fresh_backend():
    import simsopt.backend as backend

    mod = importlib.reload(backend)
    mod.invalidate_backend_cache()
    return mod


def _clear_backend_env(monkeypatch) -> None:
    for name in _BACKEND_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


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
        expected=None,
    )


@pytest.mark.parametrize("mode", ["jax_cpu_parity", "jax_gpu_parity"])
def test_parity_modes_default_transfer_guard_to_log(monkeypatch, mode):
    _clear_backend_env(monkeypatch)
    backend = _fresh_backend()

    _assert_transfer_guard_resolution(backend, mode=mode, expected="log")


def test_point_chunk_size_defaults_follow_mode(monkeypatch):
    _clear_backend_env(monkeypatch)
    backend = _fresh_backend()

    assert backend.get_point_chunk_size("native_cpu") == 0
    assert backend.get_point_chunk_size("jax_cpu_parity") == 256
    assert backend.get_point_chunk_size("jax_gpu_fast") == 1024


def test_field_kernel_tuning_defaults_follow_mode(monkeypatch):
    _clear_backend_env(monkeypatch)
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
