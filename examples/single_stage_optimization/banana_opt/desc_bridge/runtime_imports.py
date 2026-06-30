"""Runtime import helpers for explicit DESC source roots."""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def activate_desc_source_root(path: Path | None) -> Iterator[None]:
    """Make an explicit DESC source root authoritative for runtime imports."""
    if path is None:
        yield
        return
    resolved_root = path.resolve()
    _drop_cached_desc_modules_outside_root(resolved_root)
    inserted = os.fspath(resolved_root)
    sys.path.insert(0, inserted)
    try:
        yield
    finally:
        # Keep loaded DESC modules cached so staged runtime calls share class identity.
        sys.path.remove(inserted)


def _drop_cached_desc_modules_outside_root(desc_source_root: Path) -> None:
    for module_name, module in tuple(sys.modules.items()):
        if module_name == "desc" or module_name.startswith("desc."):
            if not _module_loaded_from_root(module, desc_source_root):
                del sys.modules[module_name]


def _module_loaded_from_root(module: object, desc_source_root: Path) -> bool:
    module_file = getattr(module, "__file__", None)
    if not isinstance(module_file, str) or module_file == "":
        return False
    try:
        Path(module_file).resolve().relative_to(desc_source_root)
    except (OSError, ValueError):
        return False
    return True
