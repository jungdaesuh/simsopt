"""Public, import-light optimizer policy types."""

from __future__ import annotations

from typing import Literal

TraceableNewtonLinearSolver = Literal[
    "operator_gmres",
    "dense_lu",
    "hybrid_final_dense_lu",
    "hybrid_final_dense_ir",
]

__all__ = ("TraceableNewtonLinearSolver",)
