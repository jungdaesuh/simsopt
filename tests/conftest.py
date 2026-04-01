"""Root-level test fixtures shared across the JAX test suite."""

from __future__ import annotations


def _activate_backend_mode(monkeypatch, request, *, mode, strict):
    from simsopt.backend import invalidate_backend_cache

    backend_env_ctx = monkeypatch.context()
    backend_env = backend_env_ctx.__enter__()
    backend_env.setenv("SIMSOPT_BACKEND_MODE", mode)
    if strict:
        backend_env.setenv("SIMSOPT_BACKEND_STRICT", "1")
    else:
        backend_env.delenv("SIMSOPT_BACKEND_STRICT", raising=False)
    invalidate_backend_cache()

    def _restore_backend_mode():
        backend_env_ctx.__exit__(None, None, None)
        invalidate_backend_cache()

    request.addfinalizer(_restore_backend_mode)


def enable_strict_jax_backend(monkeypatch, request, mode="jax_gpu_parity"):
    """Activate strict JAX backend mode for a single test.

    Sets the backend mode and strict env vars, invalidates the config cache,
    and registers a finalizer to restore the env and re-invalidate after
    test-local backend mode changes.
    """
    _activate_backend_mode(monkeypatch, request, mode=mode, strict=True)


def enable_non_strict_jax_backend(monkeypatch, request, mode):
    """Activate non-strict JAX backend mode for a single test.

    Sets the backend mode, removes the strict env var, and invalidates the
    config cache.  ``mode`` is required — callers must be explicit about which
    backend mode they intend (e.g. ``"jax_cpu_parity"`` or ``"jax_gpu_parity"``).
    """
    _activate_backend_mode(monkeypatch, request, mode=mode, strict=False)


def relative_error(actual, reference):
    """Return ``|actual - reference| / (|reference| + 1e-30)``."""
    return abs(actual - reference) / (abs(reference) + 1e-30)
