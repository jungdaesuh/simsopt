"""Give isolated children the compiled ``simsoptpp`` the parent already loaded.

Checkout trees keep the extension under ``build/``; installed wheels keep it in
site-packages. ``src/simsoptpp/`` is C++ sources with no ``__init__.py``: if it
precedes the extension on ``sys.path``, Python imports a namespace package that
has no ``Curve``. ``python -I`` also ignores ``PYTHONPATH``, so isolated children
must receive the kernel directory explicitly.
"""

from __future__ import annotations

import os
import sys
import sysconfig
from pathlib import Path

import simsoptpp

_ISOLATED_CHILD_WRAPPER = Path(__file__).with_name("isolated_kernel_child.py")


def loaded_kernel_directory() -> Path:
    """Directory of the compiled ``simsoptpp`` this process already imported."""
    kernel_file = simsoptpp.__file__
    if kernel_file is None:
        raise RuntimeError(
            "compiled simsoptpp extension is not loaded "
            "(namespace package has no __file__)"
        )
    return Path(kernel_file).resolve().parent


def pythonpath_with_loaded_kernel(*entries: str) -> str:
    """Return PYTHONPATH with the loaded kernel directory first."""
    ordered = (str(loaded_kernel_directory()), *entries)
    return os.pathsep.join(dict.fromkeys(entry for entry in ordered if entry))


def repo_child_pythonpath(repo_root: Path, inherited_pythonpath: str | None) -> str:
    """PYTHONPATH for a repo child that must import both sources and the kernel."""
    extra = (
        ()
        if not inherited_pythonpath
        else tuple(part for part in inherited_pythonpath.split(os.pathsep) if part)
    )
    return pythonpath_with_loaded_kernel(
        str(repo_root / "src"),
        *extra,
        sysconfig.get_paths()["purelib"],
    )


def isolated_child_command(
    payload: tuple[str, ...],
    *,
    repo_root: Path | None = None,
    extra_path_entries: tuple[str, ...] = (),
    executable: str | None = None,
) -> tuple[str, ...]:
    """Build ``python -I`` argv that still sees the compiled kernel.

    Isolated mode ignores PYTHONPATH. The stdlib wrapper inserts the
    parent-loaded kernel directory, then optional repo ``src`` and root, then
    ``payload``.
    """
    python = sys.executable if executable is None else executable
    repo_entries = () if repo_root is None else (str(repo_root / "src"), str(repo_root))
    return (
        python,
        "-I",
        str(_ISOLATED_CHILD_WRAPPER),
        str(loaded_kernel_directory()),
        *repo_entries,
        *extra_path_entries,
        "--",
        *payload,
    )
