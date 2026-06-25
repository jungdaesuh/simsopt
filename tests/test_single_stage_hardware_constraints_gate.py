"""Coverage for the certification-mode hardware gate in the accepted-artifact check.

``accepted_result_rejection_reasons`` is shared by the single-stage and Stage-2
lanes.  The single-stage lane opts into a direct hardware check
(``require_hardware=not init_only``) so a real certification run cannot emit an
accepted ``results.json`` when its hardware verdict explicitly failed -- while a
benchmark run (verdict skipped -> ``hardware_constraints_ok=None``) and an
init-only run still pass the gate, and Stage-2 callers (which omit the new kwargs)
are unaffected.  This pins that truth table.
"""

import pytest

from examples.single_stage_optimization.hardware_constraints import (
    accepted_result_rejection_reasons,
)

# A payload+optimizer state that passes every other acceptance gate, so the only
# possible rejection reason under test is the hardware one.
_PASSING_STATE = dict(
    optimizer_success=True,
    optimizer_status=0,
    final_objective=1.0,
    final_dofs=[1.0, 2.0],
    final_gradient_finite=True,
    require_gradient=True,
)


@pytest.mark.parametrize(
    ("require_hardware", "hardware_constraints_ok", "expect_hardware_failed"),
    [
        pytest.param(True, False, True, id="certification_hardware_failed_rejects"),
        pytest.param(True, True, False, id="certification_hardware_passed_accepts"),
        pytest.param(True, None, False, id="benchmark_skipped_verdict_accepts"),
        pytest.param(False, False, False, id="init_only_failed_hardware_accepts"),
    ],
)
def test_hardware_gate_modes(
    require_hardware, hardware_constraints_ok, expect_hardware_failed
):
    reasons = accepted_result_rejection_reasons(
        {},
        require_hardware=require_hardware,
        hardware_constraints_ok=hardware_constraints_ok,
        **_PASSING_STATE,
    )
    assert ("hardware_constraints_failed" in reasons) is expect_hardware_failed
    # The fixture passes all other gates, so hardware is the only possible reason.
    assert [r for r in reasons if r != "hardware_constraints_failed"] == []


def test_stage2_default_callers_unaffected():
    # Stage-2 calls the shared gate without the new kwargs; the defaults
    # (require_hardware=False) must never add the hardware reason.
    reasons = accepted_result_rejection_reasons({}, **_PASSING_STATE)
    assert reasons == []
