"""Isolated-mode child entry that exposes the parent-loaded simsoptpp kernel.

``python -I`` ignores PYTHONPATH. The parent passes the kernel directory from
its already-loaded ``simsoptpp.__file__`` as the first argument; this module
keeps that directory first on ``sys.path`` even if the payload later prepends
``src/``, then runs the original payload. Stdlib only: the wrapper must run
before any distribution import.
"""

from __future__ import annotations

import sys
import types
from collections.abc import Iterable
from pathlib import Path

_SEPARATOR = "--"


class _KernelFirstPath(list):
    """``sys.path`` that keeps the compiled kernel directory at index 0."""

    def __init__(self, kernel: str, entries: list[str]) -> None:
        self._kernel = kernel
        super().__init__(entries)
        self._keep_kernel_first()

    def insert(self, index: int, value: object) -> None:
        super().insert(index, value)
        self._keep_kernel_first()

    def append(self, value: object) -> None:
        super().append(value)
        self._keep_kernel_first()

    def extend(self, values: Iterable[object]) -> None:
        super().extend(values)
        self._keep_kernel_first()

    def __setitem__(self, index: int | slice, value: object) -> None:
        super().__setitem__(index, value)
        self._keep_kernel_first()

    def __iadd__(self, values: Iterable[object]) -> _KernelFirstPath:
        self.extend(values)
        return self

    def _keep_kernel_first(self) -> None:
        kernel = self._kernel
        super().__init__([kernel, *(entry for entry in self if entry != kernel)])


if __name__ == "__main__":
    separator_at = sys.argv.index(_SEPARATOR)
    path_entries = sys.argv[1:separator_at]
    payload = sys.argv[separator_at + 1 :]
    kernel = path_entries[0]
    wrapper_directory = str(Path(__file__).resolve().parent)
    inherited = [entry for entry in sys.path if entry != wrapper_directory]
    sys.path = _KernelFirstPath(kernel, [*path_entries, *inherited])
    main = types.ModuleType("__main__")
    sys.modules["__main__"] = main
    if payload[0] == "-c":
        script = payload[1]
        sys.argv = ["-c", *payload[2:]]
        sys.path.insert(1, "")
        main.__dict__["__name__"] = "__main__"
        exec(compile(script, "<string>", "exec"), main.__dict__)
    else:
        script_path = payload[0]
        sys.argv = payload
        sys.path.insert(1, str(Path(script_path).resolve().parent))
        main.__file__ = script_path
        main.__dict__["__name__"] = "__main__"
        main.__dict__["__file__"] = script_path
        exec(
            compile(Path(script_path).read_text(encoding="utf-8"), script_path, "exec"),
            main.__dict__,
        )
