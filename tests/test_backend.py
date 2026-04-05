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
    "SIMSOPT_BACKEND",
    "STAGE2_BACKEND",
    "SIMSOPT_JAX_PLATFORM",
    "SIMSOPT_JAX_BACKEND",
    "JAX_PLATFORMS",
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
