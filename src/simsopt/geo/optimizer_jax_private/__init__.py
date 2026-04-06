"""Private optimizer internals for the JAX Boozer inner solve.

This package contains the on-device BFGS/L-BFGS implementations that
use public JAX APIs with a minimum supported JAX floor of 0.9.2.

The public API is in ``simsopt.geo.optimizer_jax``.
"""

from ._types import (
    _BFGSResults,
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
    _make_bfgs_continuation_state,
    _minimize_bfgs_private,
)
from ._lbfgs import (
    _minimize_lbfgs_private,
    _minimize_lbfgs_private_value_and_grad,
    _two_loop_recursion,
    _update_history_scalars,
    _update_history_vectors,
)
from ._result_converters import (
    _coerce_dense_hess_inv,
    _private_bfgs_result_to_optimize_result,
    _private_lbfgs_result_to_optimize_result,
    _scipy_result_is_continuable,
    _status_message_bfgs,
    _status_message_lbfgs,
)
