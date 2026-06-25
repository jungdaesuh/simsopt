"""Contract: end-state metric parity is gated on optimizer convergence.

``evaluate_single_stage_init_parity`` compares two optimizer END-STATES
(``FINAL_IOTA``/``FINAL_VOLUME``/``FIELD_ERROR``) at machine-precision tolerances.
That comparison is only a port-correctness signal when each lane reached a usable
optimum. A lane whose L-BFGS-B terminated abnormally (``status=2``, line-search
failure / no progress -- the CPU reference stalling at iota~0.0035 in 39 nfev)
lands on a non-optimum.

The gate is ASYMMETRIC: end-state drift is suspended only when the CPU
*reference* aborted abnormally while the JAX *target* did not -- there is then no
converged reference to compare against. A JAX-*target* abnormal stop stays a hard
failure (a broken target must not hide behind the skip), and ``status=1`` (hit the
iteration budget -- a normal full-budget stop) stays strictly compared. These
tests pin that contract.
"""

from __future__ import annotations

from benchmarks.single_stage_init_parity import (
    _OUTER_LOOP_REQUIRED_RESULT_KEYS,
    _TARGET_OUTER_OPTIMIZER_METHOD,
    evaluate_single_stage_init_parity,
)

_CONVERGED = 0
_HIT_MAXITER = 1
_ABNORMAL = 2


def _result(
    *,
    iota: float,
    status: int,
    iterations: int = 5,
    jac_inf_norm: float = 1.0,
) -> dict[str, object]:
    """A complete, finite single-stage lane result with the given L-BFGS-B status."""
    result: dict[str, object] = {key: 1.0 for key in _OUTER_LOOP_REQUIRED_RESULT_KEYS}
    result.update(
        {
            "FINAL_IOTA": iota,
            "FINAL_VOLUME": 0.05,
            "FIELD_ERROR": 1.0e-3,
            "MAX_CURVATURE": 1.0,
            "SELF_INTERSECTING": False,
            "SELF_INTERSECTION_CHECK_AVAILABLE": True,
            "iterations": iterations,
            "outer_optimizer_method": _TARGET_OUTER_OPTIMIZER_METHOD,
            "OPTIMIZER_SUCCESS": status == _CONVERGED,
            "OPTIMIZER_STATUS": status,
            "OPTIMIZER_JAC_INF_NORM": jac_inf_norm,
            # Identical seed metrics on both lanes -> the seed-state backstop
            # passes, isolating these tests to the final-state gate under test.
            "INITIAL_IOTA": 0.0,
            "INITIAL_VOLUME": 0.05,
            "INITIAL_FIELD_ERROR": 1.0e-3,
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


def _has_iota_failure(failures) -> bool:
    return any("Final iota disagreement" in f for f in failures)


def _has_no_step_failure(failures) -> bool:
    return any("did not accept an optimizer step" in f for f in failures)


def test_end_state_drift_fails_when_both_lanes_converged():
    comparison, failures = _evaluate(
        _result(iota=0.10, status=_CONVERGED),
        _result(iota=0.20, status=_CONVERGED),  # 0.10 drift >> 1e-10
    )
    assert _has_iota_failure(failures), failures
    assert not comparison.get("final_metric_parity_skipped_for_nonconvergence")


def test_no_step_stationary_convergence_satisfies_outer_loop_probe():
    comparison, failures = _evaluate(
        _result(iota=0.10, status=_CONVERGED, iterations=0, jac_inf_norm=0.0),
        _result(iota=0.10, status=_CONVERGED, iterations=0, jac_inf_norm=0.0),
    )

    assert not _has_no_step_failure(failures), failures
    assert comparison["cpu_stationary_no_step"] is True
    assert comparison["jax_stationary_no_step"] is True


def test_no_step_nonstationary_convergence_fails_outer_loop_probe():
    comparison, failures = _evaluate(
        _result(iota=0.10, status=_CONVERGED, iterations=0, jac_inf_norm=1.0e-3),
        _result(iota=0.10, status=_CONVERGED, iterations=0, jac_inf_norm=1.0e-3),
    )

    assert _has_no_step_failure(failures), failures
    assert comparison["cpu_stationary_no_step"] is False
    assert comparison["jax_stationary_no_step"] is False


def test_end_state_drift_skipped_when_reference_aborts_abnormally():
    # The motivating incident: CPU reference aborted (status 2), JAX target fine.
    comparison, failures = _evaluate(
        _result(iota=0.0035, status=_ABNORMAL),
        _result(iota=0.143, status=_CONVERGED),
    )
    assert not _has_iota_failure(failures), failures
    assert comparison["final_metric_parity_skipped_for_nonconvergence"] is True
    assert comparison["skipped_final_metric_parity_failures"]
    assert comparison["cpu_optimizer_status"] == _ABNORMAL
    assert comparison["jax_optimizer_status"] == _CONVERGED


def test_end_state_drift_kept_when_target_aborts_abnormally():
    # Escape-hatch guard: a JAX-target abnormal stop must NOT be skipped -- a
    # broken target that stalls the optimizer is a port-relevant signal.
    comparison, failures = _evaluate(
        _result(iota=0.10, status=_CONVERGED),
        _result(iota=0.20, status=_ABNORMAL),
    )
    assert _has_iota_failure(failures), failures
    assert not comparison.get("final_metric_parity_skipped_for_nonconvergence")


def test_end_state_drift_kept_when_both_lanes_abort():
    # Both aborted -> the target also aborted, so the failure is kept (not skipped).
    comparison, failures = _evaluate(
        _result(iota=0.0035, status=_ABNORMAL),
        _result(iota=0.20, status=_ABNORMAL),
    )
    assert _has_iota_failure(failures), failures
    assert not comparison.get("final_metric_parity_skipped_for_nonconvergence")


def test_end_state_drift_strict_when_both_hit_maxiter():
    # status 1 (hit the iteration budget) is a normal full-budget stop -- both
    # lanes ran the full budget, so the end-state comparison is meaningful and
    # the strict gate is preserved (not relaxed like an abnormal abort).
    comparison, failures = _evaluate(
        _result(iota=0.10, status=_HIT_MAXITER),
        _result(iota=0.20, status=_HIT_MAXITER),
    )
    assert _has_iota_failure(failures), failures
    assert not comparison.get("final_metric_parity_skipped_for_nonconvergence")


def test_end_state_drift_strict_when_no_outer_optimizer_runs():
    # maxiter<=0 runs no outer optimizer, so there is no convergence concept and
    # the strict end-state gate is preserved (the relaxation must not leak here).
    comparison, failures = _evaluate(
        _result(iota=0.10, status=_CONVERGED),
        _result(iota=0.20, status=_CONVERGED),
        maxiter=0,
    )
    assert _has_iota_failure(failures), failures
    assert not comparison.get("final_metric_parity_skipped_for_nonconvergence")
