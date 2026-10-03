"""Shared pairwise-chunking helpers for the ``tests/geo`` suite."""

from __future__ import annotations

import jax

from simsopt_jax.backend import invalidate_backend_cache


def set_pairwise_penalty_chunk_size(monkeypatch, chunk_size: int) -> None:
    """Repoint every pairwise penalty reduction at ``chunk_size``.

    Shared by ``test_curve_objectives_jax.py`` and
    ``test_surface_objectives_jax.py``: both compare a dense sweep against a
    chunked sweep of the same reduction, and keeping one copy keeps the two
    from drifting apart.
    """
    monkeypatch.setenv("SIMSOPT_JAX_PENALTY_POINT_CHUNK_SIZE", str(chunk_size))
    invalidate_backend_cache()
    # ``pairwise_thresholded_mean_square_distance_pure`` (the surface-module
    # kernel) is jitted with ``chunk_size`` static, and its callers leave it at
    # ``None`` so the size is resolved from the environment *inside* the trace.
    # Clearing only simsopt_jax's backend cache therefore leaves JAX's
    # executable cache keyed on ``(chunk_size=None, avals)``, and a second call
    # at a new chunk size silently replays the first trace -- which would make
    # every dense-vs-chunked comparison in that module compare a cached path
    # against itself.  The curve-module kernel
    # (``curve_surface_distance_penalty_pure``) is not jitted today and reads
    # the environment eagerly, so for it the cache clear is prophylactic: it
    # keeps this helper correct if that kernel ever adopts the same static-jit
    # pattern.
    jax.clear_caches()
