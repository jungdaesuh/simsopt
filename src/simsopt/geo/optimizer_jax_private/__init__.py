"""Private optimizer internals for the JAX Boozer inner solve.

This package contains the private BFGS and L-BFGS implementations that use
public JAX APIs with a minimum supported JAX floor of 0.9.2.

The public API is in ``simsopt.geo.optimizer_jax``.
"""

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
    _scipy_result_is_continuable,
    _status_message_bfgs,
    _status_message_lbfgs,
)
