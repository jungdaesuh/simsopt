"""Root-level test fixtures shared across the JAX test suite."""

from __future__ import annotations


def enable_strict_jax_backend(monkeypatch, mode="jax_gpu_parity"):
    """Activate strict JAX backend mode for a single test.

    Sets the backend mode and strict env vars, invalidates the config cache,
    and registers a teardown callback to re-invalidate after monkeypatch
    restores the original env.
    """
    from simsopt.backend import invalidate_backend_cache

    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", mode)
    monkeypatch.setenv("SIMSOPT_BACKEND_STRICT", "1")
    invalidate_backend_cache()
    monkeypatch.callback(invalidate_backend_cache)
