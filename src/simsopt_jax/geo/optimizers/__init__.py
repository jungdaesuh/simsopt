"""JAX-native geometry optimizers.

Executable optimizer APIs live in named public submodules so importing policy
types from this package remains independent of the JAX runtime.
"""

from .policy import TraceableNewtonLinearSolver

__all__ = ("TraceableNewtonLinearSolver",)
