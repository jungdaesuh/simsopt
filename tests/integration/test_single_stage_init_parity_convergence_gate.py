"""Contract: end-state metric parity is gated on optimizer convergence.

``evaluate_single_stage_init_parity`` compares two optimizer END-STATES
(``FINAL_IOTA``/``FINAL_VOLUME``/``FIELD_ERROR``) at machine-precision tolerances.
That comparison is only a port-correctness signal when BOTH outer optimizers
converged: if a lane's L-BFGS-B terminates abnormally (``OPTIMIZER_STATUS=2``,
e.g. the CPU reference stalling at iota~0.0035 with 39 nfev) it lands on a
non-optimum, so end-state drift then reflects reference non-convergence -- not a
port defect. These tests pin that the gate:

- still FAILS on end-state drift when both lanes converged (strict path intact);
- does NOT fail on the same drift when a lane did not converge, recording a
  ``final_metric_parity_skipped_for_nonconvergence`` diagnostic instead.

Real port regressions remain caught by the surface-geometry, self-intersection,
finite-result, and same-candidate fixed-state channels, none of which depend on
optimizer convergence.
"""

from __future__ import annotations

from benchmarks.single_stage_init_parity import (
    _OUTER_LOOP_REQUIRED_RESULT_KEYS,
    _TARGET_OUTER_OPTIMIZER_METHOD,
    evaluate_single_stage_init_parity,
)


def _result(*, iota: float, success: bool, status: int) -> dict[str, object]:
    """A complete, finite single-stage lane result with the given convergence."""
    result: dict[str, object] = {key: 1.0 for key in _OUTER_LOOP_REQUIRED_RESULT_KEYS}
    result.update(
        {
            "FINAL_IOTA": iota,
            "FINAL_VOLUME": 0.05,
            "FIELD_ERROR": 1.0e-3,
            "MAX_CURVATURE": 1.0,
            "SELF_INTERSECTING": False,
            "SELF_INTERSECTION_CHECK_AVAILABLE": True,
            "iterations": 5,
            "outer_optimizer_method": _TARGET_OUTER_OPTIMIZER_METHOD,
            "OPTIMIZER_SUCCESS": success,
            "OPTIMIZER_STATUS": status,
        }
    )
    return result


def _evaluate(cpu: dict[str, object], jax_lane: dict[str, object], *, maxiter: int = 1500):
    return evaluate_single_stage_init_parity(
        cpu,
        jax_lane,
        max_surface_geometry_abs=0.0,
        max_surface_geometry_rel=0.0,
        maxiter=maxiter,
        expected_jax_outer_optimizer_method=_TARGET_OUTER_OPTIMIZER_METHOD,
        require_final_metric_parity=True,
    )


def test_end_state_drift_fails_when_both_lanes_converged():
    cpu = _result(iota=0.10, success=True, status=0)
    jax_lane = _result(iota=0.20, success=True, status=0)  # 0.10 drift >> 1e-10
    comparison, failures = _evaluate(cpu, jax_lane)

    assert any("Final iota disagreement" in f for f in failures), failures
    assert not comparison.get("final_metric_parity_skipped_for_nonconvergence")


def test_end_state_drift_skipped_when_reference_did_not_converge():
    # Same large drift, but the CPU reference aborted abnormally (STATUS=2).
    cpu = _result(iota=0.0035, success=False, status=2)
    jax_lane = _result(iota=0.143, success=True, status=0)
    comparison, failures = _evaluate(cpu, jax_lane)

    assert not any("Final iota disagreement" in f for f in failures), failures
    assert comparison["final_metric_parity_skipped_for_nonconvergence"] is True
    assert comparison["skipped_final_metric_parity_failures"]
    assert comparison["cpu_optimizer_success"] is False
    assert comparison["jax_optimizer_success"] is True


def test_end_state_drift_skipped_when_target_did_not_converge():
    # Symmetry: a non-converged JAX target also suspends the end-state gate
    # (its non-convergence is surfaced, not silently passed).
    cpu = _result(iota=0.10, success=True, status=0)
    jax_lane = _result(iota=0.20, success=False, status=2)
    comparison, failures = _evaluate(cpu, jax_lane)

    assert not any("Final iota disagreement" in f for f in failures), failures
    assert comparison["final_metric_parity_skipped_for_nonconvergence"] is True


def test_end_state_drift_skipped_when_both_lanes_did_not_converge():
    cpu = _result(iota=0.0035, success=False, status=2)
    jax_lane = _result(iota=0.20, success=False, status=2)
    comparison, failures = _evaluate(cpu, jax_lane)

    assert not any("Final iota disagreement" in f for f in failures), failures
    assert comparison["final_metric_parity_skipped_for_nonconvergence"] is True


def test_end_state_drift_strict_when_no_outer_optimizer_runs():
    # maxiter<=0 runs no outer optimizer, so there is no convergence concept and
    # the strict end-state gate is preserved (the relaxation must not leak here).
    cpu = _result(iota=0.10, success=True, status=0)
    jax_lane = _result(iota=0.20, success=True, status=0)
    comparison, failures = _evaluate(cpu, jax_lane, maxiter=0)

    assert any("Final iota disagreement" in f for f in failures), failures
    assert not comparison.get("final_metric_parity_skipped_for_nonconvergence")
