"""Private optimizer internals for the JAX Boozer inner solve.

This package contains the private BFGS and L-BFGS implementations maintained
against the checked local JAX 0.10.0 runtime after the initial port from the
upstream JAX optimizer sources.

The public API is in ``simsopt_jax.geo.optimizers.optimizer``. The historical
``private._dense_ir`` path is a compatibility import for the acyclic
``simsopt_jax.geo.optimizers.dense_ir`` owner.
"""

from . import _line_search as _line_search_module
from ._types import (
    _BFGSResults,
    _LBFGSInvalidStepLog,
    _LBFGSResults,
    _LineSearchResults,
    _LineSearchState,
    _ZoomState,
)
from ._line_search import (
    _binary_replace,
    _cubicmin,
    _line_search,
    _line_search_value_and_grad,
    _quadmin,
    _zoom,
)
from ._bfgs import (
    _minimize_bfgs_private,
)
from ._lbfgs import (
    _minimize_lbfgs_private,
    _minimize_lbfgs_private_value_and_grad,
)
from ._result_converters import (
    _private_bfgs_result_to_optimize_result,
    _private_lbfgs_result_to_optimize_result,
    _status_message_bfgs,
    _status_message_lbfgs,
)
