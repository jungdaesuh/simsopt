from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

try:
    import _simsopt_editable
except ModuleNotFoundError:
    _simsopt_editable = None

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
PACKAGE_ROOT = SRC_ROOT / "simsopt"

sys.path.insert(0, str(SRC_ROOT))
sys.meta_path = [
    finder
    for finder in sys.meta_path
    if not (
        _simsopt_editable is not None
        and isinstance(finder, _simsopt_editable.ScikitBuildRedirectingFinder)
    )
]

for module_name in tuple(sys.modules):
    if module_name == "simsopt" or module_name.startswith("simsopt."):
        del sys.modules[module_name]

spec = importlib.util.spec_from_file_location(
    "simsopt",
    PACKAGE_ROOT / "__init__.py",
    submodule_search_locations=[str(PACKAGE_ROOT)],
)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules["simsopt"] = module
spec.loader.exec_module(module)
