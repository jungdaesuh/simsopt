from __future__ import annotations

import os
import sys


def _prepend_sys_path(path: str) -> None:
    normalized_path = os.path.abspath(path)
    sys.path[:] = [
        existing_path
        for existing_path in sys.path
        if os.path.abspath(existing_path) != normalized_path
    ]
    sys.path.insert(0, normalized_path)


def _resolved_module_file(module_name: str) -> str | None:
    module = sys.modules.get(module_name)
    if module is None:
        return None
    module_file = getattr(module, "__file__", None)
    if module_file is None:
        return None
    return os.path.abspath(module_file)


def _clear_simsopt_modules() -> None:
    for module_name in tuple(sys.modules):
        if module_name == "simsopt" or module_name.startswith("simsopt."):
            del sys.modules[module_name]


def configure_local_simsopt_imports(script_file: str) -> tuple[str, str, str]:
    script_dir = os.path.dirname(os.path.abspath(script_file))
    example_root = os.path.abspath(os.path.join(script_dir, ".."))
    simsopt_root = os.path.abspath(os.path.join(example_root, "..", ".."))
    src_root = os.path.join(simsopt_root, "src")
    local_simsopt_init = os.path.join(src_root, "simsopt", "__init__.py")

    _prepend_sys_path(src_root)
    _prepend_sys_path(example_root)

    sys.meta_path = [
        finder
        for finder in sys.meta_path
        if not (
            type(finder).__module__ == "_simsopt_editable"
            and type(finder).__name__ == "ScikitBuildRedirectingFinder"
        )
    ]

    loaded_simsopt_init = _resolved_module_file("simsopt")
    if loaded_simsopt_init != local_simsopt_init:
        _clear_simsopt_modules()

    return example_root, simsopt_root, src_root
