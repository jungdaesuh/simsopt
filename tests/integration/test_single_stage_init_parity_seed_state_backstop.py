"""Contract: a convergence-independent seed-state parity backstop is always armed.

The final-state metric gate compares two optimizer END-states and is legitimately
SKIPPED when the CPU reference aborts abnormally (see the convergence-gate tests),
which would otherwise leave a green verdict with no positive port-correctness
signal. ``evaluate_single_stage_init_parity`` therefore also compares the INITIAL
(seed) surface metrics, which both lanes evaluate at the IDENTICAL seed DOFs before
any outer optimizer runs. In the real production artifact these agree to machine
precision (INITIAL_IOTA/VOLUME exactly, INITIAL_FIELD_ERROR ~2e-15), so a real port
defect that perturbed the seed surface would trip this gate.

These tests pin that the seed-state gate:
- fails on seed-state drift even when both lanes converged; and
- stays armed (still fails) when the final-state gate is skipped for reference
  non-convergence -- the case the backstop exists for.
"""

from __future__ import annotations

from benchmarks.single_stage_init_parity import (
    _OUTER_LOOP_REQUIRED_RESULT_KEYS,
    _TARGET_OUTER_OPTIMIZER_METHOD,
    evaluate_single_stage_init_parity,
)

_CONVERGED = 0
_ABNORMAL = 2


def _result(*, final_iota: float, initial_iota: float, status: int) -> dict[str, object]:
    result: dict[str, object] = {key: 1.0 for key in _OUTER_LOOP_REQUIRED_RESULT_KEYS}
    result.update(
        {
            "FINAL_IOTA": final_iota,
            "FINAL_VOLUME": 0.05,
            "FIELD_ERROR": 1.0e-3,
            "MAX_CURVATURE": 1.0,
            "INITIAL_IOTA": initial_iota,
            "INITIAL_VOLUME": 0.05,
            "INITIAL_FIELD_ERROR": 1.0e-3,
            "SELF_INTERSECTING": False,
            "SELF_INTERSECTION_CHECK_AVAILABLE": True,
            "iterations": 5,
            "outer_optimizer_method": _TARGET_OUTER_OPTIMIZER_METHOD,
            "OPTIMIZER_SUCCESS": status == _CONVERGED,
            "OPTIMIZER_STATUS": status,
        }
    )
    return result


def _evaluate(cpu: dict[str, object], jax_lane: dict[str, object]):
    return evaluate_single_stage_init_parity(
        cpu,
        jax_lane,
        max_surface_geometry_abs=0.0,
        max_surface_geometry_rel=0.0,
        maxiter=1500,
        expected_jax_outer_optimizer_method=_TARGET_OUTER_OPTIMIZER_METHOD,
        require_final_metric_parity=True,
    )


def _has_seed_failure(failures) -> bool:
    return any("Initial seed-state iota disagreement" in f for f in failures)


def test_seed_state_agreement_passes():
    # Identical seed iota on both lanes (final iotas may differ): no seed failure.
    comparison, failures = _evaluate(
        _result(final_iota=0.10, initial_iota=0.046, status=_CONVERGED),
        _result(final_iota=0.10, initial_iota=0.046, status=_CONVERGED),
    )
    assert not _has_seed_failure(failures), failures
    assert comparison["initial_metric_parity_failures"] == []


def test_seed_state_drift_fails_even_when_both_converged():
    comparison, failures = _evaluate(
        _result(final_iota=0.10, initial_iota=0.046, status=_CONVERGED),
        _result(final_iota=0.10, initial_iota=0.030, status=_CONVERGED),  # seed drift
    )
    assert _has_seed_failure(failures), failures
    assert comparison["initial_metric_parity_failures"]


def test_seed_state_backstop_armed_when_final_state_is_skipped():
    # The key contract: CPU reference aborts (status 2) so the FINAL-state gate is
    # skipped -- but a seed-state (port) drift must STILL fail, because the seed
    # comparison is convergence-independent. A real port bug cannot hide behind
    # the reference-non-convergence skip.
    comparison, failures = _evaluate(
        _result(final_iota=0.0035, initial_iota=0.046, status=_ABNORMAL),
        _result(final_iota=0.143, initial_iota=0.030, status=_CONVERGED),  # seed drift
    )
    # final-state gate skipped for reference non-convergence ...
    assert comparison["final_metric_parity_skipped_for_nonconvergence"] is True
    assert not any("Final iota disagreement" in f for f in failures), failures
    # ... but the seed-state backstop still fires.
    assert _has_seed_failure(failures), failures
