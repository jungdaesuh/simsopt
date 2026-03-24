from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


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
