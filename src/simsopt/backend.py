"""
Backend selection for simsopt JAX lane.

Provides a single source of truth for whether the JAX or CPU (simsoptpp)
code path is active, and which JAX device platform is targeted.

Two orthogonal settings:

1. **Code-path backend** (``SIMSOPT_BACKEND``):
   ``"cpu"`` → use simsoptpp C++ kernels (default).
   ``"jax"`` → use the JAX-native field / Boozer / objective modules.

2. **JAX device platform** (``SIMSOPT_JAX_PLATFORM``):
   ``"cpu"`` → JAX runs on CPU (default).
   ``"cuda"`` → JAX runs on a CUDA GPU.
   Only relevant when the code-path backend is ``"jax"``.

Legacy env var ``STAGE2_BACKEND`` is still read as a fallback for (1).
Legacy env var ``SIMSOPT_JAX_BACKEND`` is still read as a fallback for (2).
"""

import os

_VALID_BACKENDS = ("cpu", "jax")
_VALID_PLATFORMS = ("cpu", "cuda")

_BACKEND_ENV = "SIMSOPT_BACKEND"
_BACKEND_LEGACY_ENV = "STAGE2_BACKEND"
_PLATFORM_ENV = "SIMSOPT_JAX_PLATFORM"
_PLATFORM_LEGACY_ENV = "SIMSOPT_JAX_BACKEND"


def get_backend() -> str:
    """Return the active compute backend: ``'cpu'`` or ``'jax'``."""
    val = os.environ.get(_BACKEND_ENV)
    source = _BACKEND_ENV
    if val is None:
        val = os.environ.get(_BACKEND_LEGACY_ENV, "cpu")
        source = (
            _BACKEND_LEGACY_ENV if _BACKEND_LEGACY_ENV in os.environ else "(default)"
        )
    if val not in _VALID_BACKENDS:
        raise ValueError(f"{source}={val!r} is not valid. Accepted: {_VALID_BACKENDS}")
    return val


def is_jax_backend() -> bool:
    """``True`` when the JAX code path is selected."""
    return get_backend() == "jax"


def get_jax_platform() -> str:
    """Return the JAX device platform: ``'cpu'`` or ``'cuda'``."""
    val = os.environ.get(_PLATFORM_ENV)
    source = _PLATFORM_ENV
    if val is None:
        val = os.environ.get(_PLATFORM_LEGACY_ENV, "cpu")
        source = (
            _PLATFORM_LEGACY_ENV if _PLATFORM_LEGACY_ENV in os.environ else "(default)"
        )
    if val not in _VALID_PLATFORMS:
        raise ValueError(f"{source}={val!r} is not valid. Accepted: {_VALID_PLATFORMS}")
    return val
