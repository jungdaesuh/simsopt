"""Import the remaining JAX surfaces with ``simsoptpp`` blocked; print the mode.

Run as a script by ``tests/integration/test_remaining_jax_surfaces_mode_matrix.py``
in one fresh child per backend mode, with ``PYTHONPATH`` set to ``src``.
"""

import importlib.abc
import json
import sys


class BlockSimsoptpp(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        del path, target
        if fullname == "simsoptpp" or fullname.startswith("simsoptpp."):
            raise ModuleNotFoundError("blocked simsoptpp for mode-matrix smoke")
        return None


sys.meta_path.insert(0, BlockSimsoptpp())

from simsopt_jax.backend import get_backend_mode, get_jax_platform
from simsopt_jax.solve.serial import (
    TraceableLeastSquaresProblem,
    least_squares_serial_solve_jax,
)
import simsopt_jax.core._finite_difference as finite_difference_jax
import simsopt_jax.solve.permanent_magnet as pm_optimization_jax
import simsopt_jax.solve.serial as solve_serial_jax

direct_modules = {
    solve_serial_jax: (
        "TraceableEqualityConstrainedProblem",
        "TraceableLeastSquaresProblem",
        "TraceableScalarProblem",
        "constrained_serial_solve_jax",
        "least_squares_serial_solve_jax",
        "serial_solve_jax",
        "traceable_least_squares_jacobian",
    ),
    pm_optimization_jax: (
        "GPMO_ArbVec_backtracking_jax",
        "GPMO_ArbVec_jax",
        "GPMO_backtracking_jax",
        "GPMO_baseline_jax",
        "GPMO_multi_jax",
        "relax_and_split_jax",
    ),
    finite_difference_jax: (
        "forward_jacobian_shard_map",
        "forward_jacobian_shard_map_columns",
        "forward_jacobian_vmap",
    ),
}
for module, symbols in direct_modules.items():
    for symbol in symbols:
        getattr(module, symbol)

public_symbols = (
    TraceableLeastSquaresProblem,
    least_squares_serial_solve_jax,
)
assert all(symbol is not None for symbol in public_symbols)
assert "simsoptpp" not in sys.modules
print(
    json.dumps(
        {
            "mode": get_backend_mode(),
            "platform": get_jax_platform(),
            "symbols": len(public_symbols),
        },
        sort_keys=True,
    )
)
