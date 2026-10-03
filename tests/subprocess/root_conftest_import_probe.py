"""Run as ``python -m root_conftest_import_probe`` with ``tests`` on ``PYTHONPATH``.

Importing the root ``tests/conftest.py`` must load no JAX or simsopt module and
leave ``XLA_FLAGS`` unchanged, so upstream's native tests run as on master.
"""

from __future__ import annotations

import os
import sys

_JAX_RUNTIME_PACKAGES = frozenset(
    {"jax", "simsopt", "simsopt_jax", "simsopt_jax_adapters"}
)


def _loaded_jax_runtime_packages() -> list[str]:
    return sorted(
        name for name in sys.modules if name.partition(".")[0] in _JAX_RUNTIME_PACKAGES
    )


_LOADED_BEFORE = _loaded_jax_runtime_packages()
_XLA_FLAGS_BEFORE = os.environ.get("XLA_FLAGS")

import conftest  # noqa: F401 - the import under test

assert _LOADED_BEFORE == [], _LOADED_BEFORE
assert _loaded_jax_runtime_packages() == [], _loaded_jax_runtime_packages()
assert os.environ.get("XLA_FLAGS") == _XLA_FLAGS_BEFORE
print("root conftest import leaves the JAX runtime alone")
