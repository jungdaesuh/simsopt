from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

_BOOTSTRAP_VERSION = "0.0.dev0+source"
_BOOTSTRAP_VERSION_TUPLE = (0, 0, "dev0", "source")


def _install_bootstrap_version_stub(package_root: Path) -> None:
    """Provide ``simsopt._version`` when bootstrapping a clean source tree."""
    version_file = package_root / "_version.py"
    if version_file.exists():
        return

    module = types.ModuleType("simsopt._version")
    module.version = _BOOTSTRAP_VERSION
    module.__version__ = _BOOTSTRAP_VERSION
    module.version_tuple = _BOOTSTRAP_VERSION_TUPLE
    module.__version_tuple__ = _BOOTSTRAP_VERSION_TUPLE
    module.__all__ = (
        "__version__",
        "__version_tuple__",
        "version",
        "version_tuple",
    )
    sys.modules["simsopt._version"] = module


def bootstrap_local_simsopt(src_root: str | Path) -> None:
    """Force imports to resolve against this repo's local simsopt source tree."""
    package_root = Path(src_root) / "simsopt"
    sys.meta_path = [
        finder
        for finder in sys.meta_path
        if not (
            type(finder).__name__ == "ScikitBuildRedirectingFinder"
            and type(finder).__module__ == "_simsopt_editable"
        )
    ]
    for name in list(sys.modules):
        if name == "simsopt" or name.startswith("simsopt."):
            del sys.modules[name]
    _install_bootstrap_version_stub(package_root)
    spec = importlib.util.spec_from_file_location(
        "simsopt",
        package_root / "__init__.py",
        submodule_search_locations=[str(package_root)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to bootstrap local simsopt package from {package_root}")
    module = importlib.util.module_from_spec(spec)
    module.__path__ = [str(package_root)]
    sys.modules["simsopt"] = module
    spec.loader.exec_module(module)
