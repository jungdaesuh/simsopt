"""Importable implementations behind the shipped flat-675 example scripts.

The example tier directories are not packages, so the scripts under
``examples/`` are entry points only: they own the lesson prose and delegate to
a module here, which is what benchmarks and contract tests import.
"""

from pathlib import Path
from typing import Final

REPOSITORY_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
"""Checkout root of this source tree.

The shipped configurations are built from repository test-file geometry, so
these modules are runnable only from a checkout, exactly as the scripts are.
"""

__all__ = ("REPOSITORY_ROOT",)
