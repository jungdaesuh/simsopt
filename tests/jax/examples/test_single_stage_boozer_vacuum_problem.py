"""The shipped mirror's problem module carries the native example's policy.

The value/gradient parity of the underlying evaluator against native is proven
in ``tests/geo/test_single_stage_exact_analytic.py``; what this file holds is
that the example's own problem -- the bounded scale its ``--smoke`` lane runs --
is wired to that policy: the frozen seed targets, the published endpoint
physics, and the failed-inner-solve rule (native's 1.0e3 sentinel with the
pre-evaluation warm start restored).
"""

from __future__ import annotations

import logging
import re
from collections import Counter

import jax
import numpy as np
import pytest
from conftest import enable_non_strict_jax_backend, parity_device
from simsopt_jax_adapters.geo.single_stage_boozer_vacuum_problem import (
    BOUNDED_SCALE,
    NCSX_INITIAL_IOTA,
    SingleStageVacuumProblem,
)
from simsopt_jax_adapters.geo.single_stage_exact_analytic import INNER_FAILURE_VALUE

_JIT_NAME = re.compile(r"jit\(([^)]+)\)")
_ALLOWED_CONSTRUCTION_COMPILES = frozenset(
    {
        "solve",
        "evaluate",
        "reporting_metrics_from_solution",
        "reporting_metrics",
        "run_code_traceable_exact_C2_array_kernel",
    }
)


@pytest.fixture(params=("cpu", "gpu"), autouse=True)
def problem_backend(monkeypatch, request):
    device = parity_device(request.param)
    enable_non_strict_jax_backend(monkeypatch, request, f"jax_{request.param}_parity")
    with jax.default_device(device):
        yield device


@pytest.fixture
def bounded_problem():
    return SingleStageVacuumProblem(BOUNDED_SCALE)


def test_seed_targets_are_frozen_at_the_solved_surface(bounded_problem) -> None:
    """Native freezes the iota target at the seed solve, not at the seed guess."""
    endpoint = bounded_problem.endpoint(bounded_problem.initial_coil_dofs)

    assert endpoint.inner_success
    assert bounded_problem.iota_target != NCSX_INITIAL_IOTA
    # At the seed coils the solve reproduces the state the target was read from.
    assert endpoint.iota == pytest.approx(bounded_problem.iota_target, rel=1.0e-12)
    # The iota, major-radius and length penalties all sit at their targets
    # there, so the seed objective is the non-quasisymmetry ratio alone.
    assert endpoint.value == pytest.approx(endpoint.non_qs_ratio, rel=1.0e-12)
    assert endpoint.volume < 0.0
    assert endpoint.boozer_residual_rms < 1.0e-13


def test_evaluation_and_reporting_are_clean_under_the_strict_transfer_guard(
    bounded_problem,
) -> None:
    """The optimizer's per-step path may not cross the host boundary at all.

    This is the hot-path pin, and it is the full guard in both directions: the
    steady-state outer callable and endpoint report may neither push to nor
    read from the device implicitly.  The first call of each is made outside
    the guard so that what is measured here is steady state rather than the
    compile; the compile is pinned separately by the construction test below.
    The seed coils are a fixed point of the inner solve, so warming there does
    not move the state the guarded calls then observe.
    """
    seed_coils = bounded_problem.initial_coil_dofs
    bounded_problem.value_and_gradient(seed_coils)
    bounded_problem.endpoint(seed_coils)

    with jax.transfer_guard("disallow"):
        value, gradient = bounded_problem.value_and_gradient(seed_coils)
        endpoint = bounded_problem.endpoint(seed_coils)

    assert np.isfinite(value)
    assert np.all(np.isfinite(gradient))
    assert endpoint.inner_success


def test_construction_is_clean_under_the_strict_transfer_guard() -> None:
    """Construction crosses the host boundary only with explicit transfers.

    Both directions, which is what the certification-eligible ``gpu-strict``
    lane runs with (``JAX_TRANSFER_GUARD=disallow`` is global there, so it
    covers compilation too).  Host-to-device: the constructor used to run under
    ``transfer_guard("allow")`` because the analytic route's geometry setup
    transferred implicitly.  Device-to-host: construction also compiles, and
    XLA reads back every device array the program captures as a constant, so
    this fails unless the coil graph and the frozen surface grids are host
    arrays -- which is what ``make_coil_dof_extraction_spec`` and
    ``build_boozer_surface_runtime_state`` now guarantee.
    """
    with jax.transfer_guard("disallow"):
        problem = SingleStageVacuumProblem(BOUNDED_SCALE)

    assert problem.initial_coil_dofs.size > 0
    assert problem.iota_target != NCSX_INITIAL_IOTA


def test_persisted_inner_failure_reports_the_sentinel_and_keeps_the_warm_start(
    bounded_problem,
) -> None:
    """A failed solve must report 1.0e3 and never poison the next warm start."""
    seed_coils = bounded_problem.initial_coil_dofs
    # The seed state is a fixed point of the inner solve, so two evaluations
    # there agree bit for bit; that identity is what detects a moved warm start.
    reference_value, reference_gradient = bounded_problem.value_and_gradient(seed_coils)
    repeated_value, repeated_gradient = bounded_problem.value_and_gradient(seed_coils)
    assert repeated_value == reference_value
    np.testing.assert_array_equal(repeated_gradient, reference_gradient)

    # A coil move far enough that the exact Newton exhausts its 20-iteration
    # cap on a finite, improved iterate -- the case native keeps internally and
    # then rolls back.  ``tests/geo/test_single_stage_exact_analytic.py`` pins
    # the same draw against the native lane.
    direction = np.random.default_rng(2).standard_normal(seed_coils.size)
    failed_value, failed_gradient = bounded_problem.value_and_gradient(
        seed_coils * (1.0 + 0.1 * direction)
    )

    assert failed_value == INNER_FAILURE_VALUE
    assert np.all(np.isfinite(failed_gradient))

    restored_value, restored_gradient = bounded_problem.value_and_gradient(seed_coils)
    assert restored_value == reference_value
    np.testing.assert_array_equal(restored_gradient, reference_gradient)


def test_construction_compiles_only_the_seed_solve_and_not_tiny_primitives() -> None:
    """Host NumPy bake must not dispatch one-off primitive jits at construction.

    The seed inner Newton still compiles ``solve``. ``evaluate`` and reporting
    are allowed if they compile here; tiny primitives are not.
    """
    jax.config.update("jax_log_compiles", True)
    names: list[str] = []

    class _Tap(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            message = record.getMessage()
            if "Compiling jit(" not in message:
                return
            match = _JIT_NAME.search(message)
            if match is not None:
                names.append(match.group(1))

    tap = _Tap()
    tap.setLevel(logging.DEBUG)
    logger = logging.getLogger("jax._src.interpreters.pxla")
    logger.addHandler(tap)
    logger.setLevel(logging.INFO)
    problem = SingleStageVacuumProblem(BOUNDED_SCALE)
    logger.removeHandler(tap)
    unexpected = [name for name in names if name not in _ALLOWED_CONSTRUCTION_COMPILES]
    assert unexpected == [], Counter(names)
    assert problem.initial_coil_dofs.size > 0
