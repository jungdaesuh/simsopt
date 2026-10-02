"""Isolated-mode child entry that exposes the parent-loaded simsoptpp kernel.

``python -I`` ignores PYTHONPATH. The parent passes the kernel directory from
its already-loaded ``simsoptpp.__file__`` as the first argument; this script
keeps that directory first on ``sys.path`` even if anything later prepends
``src/``, then runs one payload chosen by name from the fixed table
``PAYLOADS``. No source text is executed: every payload is a function of this
module, and the kernel it reaches is imported statically once the path is set.

The path setup runs at import, before the kernel import below it, so this
module is a script entry only and is never imported by the package.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from types import MappingProxyType

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


_SEPARATOR_AT = sys.argv.index(_SEPARATOR)
_PATH_ENTRIES = sys.argv[1:_SEPARATOR_AT]
_PAYLOAD = sys.argv[_SEPARATOR_AT + 1 :]
_WRAPPER_DIRECTORY = str(Path(__file__).resolve().parent)
sys.path = _KernelFirstPath(
    _PATH_ENTRIES[0],
    [
        *_PATH_ENTRIES,
        *(entry for entry in sys.path if entry != _WRAPPER_DIRECTORY),
    ],
)

from simsoptpp import Curve  # noqa: E402  (resolved through the path set above)


def _print_kernel_curve_name(arguments: list[str]) -> int:
    """Print the class name of the kernel's ``Curve``: proof the kernel loaded."""
    if arguments:
        raise SystemExit(f"simsoptpp-curve-name takes no arguments: {arguments}")
    print(Curve.__name__)
    return 0


#: Every payload an isolated child can run, by the name the parent passes.
#: ``simsopt_jax_adapters.isolated_kernel.IsolatedChildPayload`` names the same keys.
PAYLOADS: Mapping[str, Callable[[list[str]], int]] = MappingProxyType(
    {"simsoptpp-curve-name": _print_kernel_curve_name}
)


if __name__ == "__main__":
    name, *payload_arguments = _PAYLOAD
    if name not in PAYLOADS:
        raise SystemExit(f"unknown isolated-child payload {name!r}")
    raise SystemExit(PAYLOADS[name](payload_arguments))
