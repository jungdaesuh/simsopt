"""Session-scoped cleanup for sys.modules stubs injected by JAX test modules.

Several test modules in this directory inject simsopt package stubs into
``sys.modules`` at import time so that the pure-JAX modules can resolve
intra-package imports without requiring ``simsoptpp``.  These stubs
persist across the test session and can break unrelated test modules
(e.g. ``test_jax_import_smoke.py``) that expect the real package layout.

This conftest captures the ``sys.modules`` state at conftest import time
(before any test modules inject stubs) and restores it after all tests
in this directory complete.
"""

import sys

import pytest

# Captured BEFORE test modules inject stubs (conftest.py is imported
# first by pytest's collection machinery).
_clean_simsopt_state = {
    k: v for k, v in sys.modules.items() if k.startswith("simsopt")
}


@pytest.fixture(autouse=True, scope="session")
def _restore_simsopt_modules():
    """Restore sys.modules after all geo tests finish."""
    yield
    for key in list(sys.modules):
        if key.startswith("simsopt") and key not in _clean_simsopt_state:
            del sys.modules[key]
    for key, mod in _clean_simsopt_state.items():
        if sys.modules.get(key) is not mod:
            sys.modules[key] = mod
