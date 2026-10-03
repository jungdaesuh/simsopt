"""Import ``simsopt.geo.jit`` against a recording fake ``jax``; print its updates.

``src/simsopt/geo/jit.py`` configures JAX at import. This child records the
``jax.config.update`` calls it makes, as a JSON list on stdout, without real JAX
and without ``simsopt.geo``'s package import: the fakes are installed in
``sys.modules`` first and the module is then imported by a static statement.
The parent selects the environment (``JAX_PLATFORMS`` / ``JAX_PLATFORM_NAME``).
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

_GEO_DIRECTORY = Path(__file__).resolve().parents[2] / "src" / "simsopt" / "geo"
UPDATES: list[tuple[str, object]] = []


class _FakeJaxConfig:
    def update(self, name: str, value: object) -> None:
        UPDATES.append((name, value))


_fake_jax = types.ModuleType("jax")
_fake_jax.config = _FakeJaxConfig()
_fake_jax.jit = lambda fun, **args: fun
_fake_simsopt = types.ModuleType("simsopt")
_fake_simsopt.__path__ = []
_fake_geo = types.ModuleType("simsopt.geo")
_fake_geo.__path__ = [str(_GEO_DIRECTORY)]
_fake_config = types.ModuleType("simsopt.geo.config")
_fake_config.parameters = {"jit": False}
sys.modules.update(
    {
        "jax": _fake_jax,
        "simsopt": _fake_simsopt,
        "simsopt.geo": _fake_geo,
        "simsopt.geo.config": _fake_config,
    }
)

import simsopt.geo.jit as legacy_jit  # noqa: E402  (after the fakes above)

if __name__ == "__main__":
    assert Path(legacy_jit.__file__).resolve() == _GEO_DIRECTORY / "jit.py"
    assert callable(legacy_jit.jit)
    print(json.dumps(UPDATES))
