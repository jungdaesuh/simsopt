from __future__ import annotations

import importlib


def _fresh_backend():
    import simsopt.backend as backend

    return importlib.reload(backend)


def _clear_backend_env(monkeypatch) -> None:
    for name in (
        "SIMSOPT_BACKEND_MODE",
        "SIMSOPT_BACKEND_STRICT",
        "SIMSOPT_BACKEND",
        "STAGE2_BACKEND",
        "SIMSOPT_JAX_PLATFORM",
        "SIMSOPT_JAX_BACKEND",
        "JAX_PLATFORMS",
    ):
        monkeypatch.delenv(name, raising=False)


def _assert_synced_runtime_env(
    backend, *, mode: str, backend_name: str, platform: str, strict: bool
) -> None:
    assert backend.os.environ["SIMSOPT_BACKEND_MODE"] == mode
    assert backend.os.environ["SIMSOPT_BACKEND_STRICT"] == ("1" if strict else "0")
    assert backend.os.environ["SIMSOPT_BACKEND"] == backend_name
    assert backend.os.environ["STAGE2_BACKEND"] == backend_name
    assert backend.os.environ["SIMSOPT_JAX_PLATFORM"] == platform
    assert backend.os.environ["SIMSOPT_JAX_BACKEND"] == platform
    assert backend.os.environ["JAX_PLATFORMS"] == platform


def test_backend_defaults_to_native_cpu(monkeypatch):
    _clear_backend_env(monkeypatch)
    backend = _fresh_backend()

    config = backend.get_backend_config()

    assert config.mode == "native_cpu"
    assert config.backend == "cpu"
    assert config.jax_platform == "cpu"
    assert config.strict is False
    assert backend.is_jax_backend() is False
    assert backend.is_backend_strict() is False


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
