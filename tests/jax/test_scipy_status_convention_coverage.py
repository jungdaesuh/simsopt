"""Every status the installed SciPy can return terminally must be classified.

``simsopt_contracts.optimization_endpoint`` transcribes one emitter's status
vocabulary per ``StatusConvention``. A status that neither counts as success
nor appears in that convention's failure table falls through
``_stopping_reason`` to a generic ``iteration-limit``/``failed`` guess, which is
exactly the "a status is never read through a guessed table" rule the module
states. These tests therefore do not restate the tables: they read the two
SciPy emitters' own vocabularies out of the installed SciPy and require the
contract to cover them.

* ``scipy.optimize.least_squares``: ``TERMINATION_MESSAGES`` is module level,
  so it is imported and used directly.
* ``scipy.optimize.minimize(method="SLSQP")``: ``exit_modes`` is a local of
  ``_minimize_slsqp``, so its dict literal is parsed out of that function's
  source with :mod:`ast`. Modes ``-1`` and ``1`` are the transient
  "evaluation required" requests the solver loop consumes
  (``if abs(state_dict['mode']) != 1: break``), so SciPy cannot return them
  terminally; they are still required to be covered, because the contract
  transcribes the emitter's whole vocabulary.
"""

from __future__ import annotations

import ast
import inspect

import pytest
import scipy.optimize._slsqp_py as slsqp_module
from scipy.optimize._lsq.least_squares import TERMINATION_MESSAGES
from simsopt_contracts.optimization_endpoint import (
    StatusConvention,
    certify_optimization_endpoint,
)
from simsopt_contracts.optimization_endpoint import (
    _FAILURE_REASON_BY_STATUS,
    _SUCCESS_STATUSES,
)

#: Modes SLSQP's own loop consumes before returning; see the module docstring.
SLSQP_TRANSIENT_MODES = frozenset({-1, 1})
#: SLSQP reports success for exactly this mode
#: (``scipy/optimize/_slsqp_py.py``, ``success=(state_dict['mode'] == 0)``).
SLSQP_SUCCESS_MODE = 0


def _slsqp_exit_modes() -> frozenset[int]:
    """The keys of ``exit_modes`` in the installed ``_minimize_slsqp``."""
    tree = ast.parse(inspect.getsource(slsqp_module._minimize_slsqp).lstrip())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        targets = [target.id for target in node.targets if isinstance(target, ast.Name)]
        if "exit_modes" in targets and isinstance(node.value, ast.Dict):
            return frozenset(
                ast.literal_eval(key) for key in node.value.keys if key is not None
            )
    raise AssertionError(
        "the installed scipy.optimize._slsqp_py._minimize_slsqp no longer "
        "assigns an `exit_modes` dict literal; the scipy-slsqp transcription "
        "cannot be checked against its source"
    )


def _covered(convention: StatusConvention) -> frozenset[int]:
    return frozenset(_SUCCESS_STATUSES[convention]) | frozenset(
        _FAILURE_REASON_BY_STATUS[convention]
    )


def test_scipy_slsqp_covers_every_exit_mode_the_solver_defines() -> None:
    exit_modes = _slsqp_exit_modes()

    assert SLSQP_SUCCESS_MODE in exit_modes
    missing = exit_modes - _covered("scipy-slsqp")
    assert not missing, (
        f"scipy-slsqp does not classify SLSQP exit mode(s) {sorted(missing)}; "
        "an unlisted status falls through to a guessed stopping reason"
    )
    assert _SUCCESS_STATUSES["scipy-slsqp"] == frozenset({SLSQP_SUCCESS_MODE})
    # Every non-success mode is a failure the table names, and no success mode
    # is also listed as a failure.
    assert frozenset(_FAILURE_REASON_BY_STATUS["scipy-slsqp"]) == (
        exit_modes - frozenset({SLSQP_SUCCESS_MODE})
    )
    assert SLSQP_TRANSIENT_MODES < exit_modes


def test_scipy_trf_covers_every_least_squares_termination_status() -> None:
    statuses = frozenset(TERMINATION_MESSAGES)

    missing = statuses - _covered("scipy-trf")
    assert not missing, (
        f"scipy-trf does not classify least_squares status(es) {sorted(missing)}; "
        "an unlisted status falls through to a guessed stopping reason"
    )
    # ``least_squares`` sets ``success = status > 0``
    # (scipy/optimize/_lsq/least_squares.py, ``OptimizeResult(... success=...)``).
    assert _SUCCESS_STATUSES["scipy-trf"] == frozenset(
        status for status in statuses if status > 0
    )
    assert frozenset(_FAILURE_REASON_BY_STATUS["scipy-trf"]) == frozenset(
        status for status in statuses if status <= 0
    )


@pytest.mark.parametrize(
    ("convention", "status", "expected"),
    (
        ("scipy-slsqp", 9, "iteration-limit"),
        ("scipy-slsqp", 8, "line-search-failed"),
        ("scipy-slsqp", 5, "failed"),
        ("scipy-trf", 0, "evaluation-limit"),
        ("scipy-trf", -1, "failed"),
        ("scipy-trf", -2, "callback-stopped"),
    ),
)
def test_the_new_conventions_classify_their_own_failures(
    convention: StatusConvention, status: int, expected: str
) -> None:
    """A failing call is classified by its own emitter, not by a neighbour's.

    ``scipy-lbfgsb`` reads status 1 as a budget; SLSQP's 1 is a transient
    request and its budget is 9. Reading either through the other's table is
    the defect these entries exist to prevent.
    """
    certificate = certify_optimization_endpoint(
        status_convention=convention,
        provider_success=False,
        provider_status=status,
        iterations=3,
        max_iterations=10,
        initial_gradient_inf_norm=1.0,
        final_gradient_inf_norm=1.0,
        parameters_finite=True,
        observables_finite=True,
        inner_success=True,
    )

    assert certificate.stopping_reason == expected
    assert certificate.success is False


@pytest.mark.parametrize(
    ("convention", "status"),
    (("scipy-slsqp", 0), ("scipy-trf", 1), ("scipy-trf", 4)),
)
def test_the_new_conventions_accept_their_own_success(
    convention: StatusConvention, status: int
) -> None:
    certificate = certify_optimization_endpoint(
        status_convention=convention,
        provider_success=True,
        provider_status=status,
        iterations=3,
        max_iterations=10,
        initial_gradient_inf_norm=1.0,
        final_gradient_inf_norm=0.0,
        parameters_finite=True,
        observables_finite=True,
        inner_success=True,
    )

    assert certificate.stopping_reason == "converged"
    assert certificate.success is True
