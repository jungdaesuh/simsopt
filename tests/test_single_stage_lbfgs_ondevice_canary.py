from __future__ import annotations

import json

import pytest

from benchmarks.single_stage_lbfgs_ondevice_canary import (
    LbfgsOnDeviceCanaryLimits,
    evaluate_lbfgs_ondevice_canary,
    main,
)


def _progress_payload(
    *,
    initial_span_s: float = 10.0,
    main_span_s: float = 20.0,
    repeat_failure: bool = False,
) -> dict[str, object]:
    events: list[dict[str, object]] = [
        {
            "label": "lbfgs_initial_value_and_grad_seed_started",
            "event_elapsed_s": 100.0,
        },
        {
            "label": "lbfgs_initial_value_and_grad_seed_returned",
            "event_elapsed_s": 100.0 + initial_span_s,
        },
        {"label": "lbfgs_main_kernel_started", "event_elapsed_s": 200.0},
        {
            "label": "lbfgs_main_kernel_returned",
            "event_elapsed_s": 200.0 + main_span_s,
        },
    ]
    if repeat_failure:
        invalid_step = {
            "iteration": 0,
            "requested_initial_step": 1.0e-6,
            "first_tested_alpha": 1.0e-6,
            "line_search_failed": True,
            "ls_status": -9,
            "failure_reason": "line_search_failed",
        }
        events.extend(
            [
                {
                    "label": "phase2_attempt_0_returned",
                    "event_elapsed_s": 250.0,
                    "result": {"invalid_step_log": [dict(invalid_step)]},
                },
                {
                    "label": "phase2_retry_1_returned",
                    "event_elapsed_s": 300.0,
                    "result": {"invalid_step_log": [dict(invalid_step)]},
                },
            ]
        )
    return {"events": events}


def _limits() -> LbfgsOnDeviceCanaryLimits:
    return LbfgsOnDeviceCanaryLimits(
        max_initial_value_grad_s=100.0,
        max_main_optimizer_s=200.0,
        repeated_line_search_failure_limit=1,
        line_search_status=-9,
    )


def test_canary_accepts_budgeted_single_seed_artifact():
    gate_payload = evaluate_lbfgs_ondevice_canary(
        progress_payload=_progress_payload(),
        limits=_limits(),
    )

    assert gate_payload["passed"] is True
    assert gate_payload["failures"] == []
    assert gate_payload["measurements"]["initial_value_grad_s"] == pytest.approx(
        10.0
    )
    assert gate_payload["measurements"]["main_optimizer_s"] == pytest.approx(20.0)


def test_canary_rejects_initial_value_grad_walltime_growth():
    gate_payload = evaluate_lbfgs_ondevice_canary(
        progress_payload=_progress_payload(initial_span_s=101.0),
        limits=_limits(),
    )

    assert gate_payload["passed"] is False
    assert gate_payload["failures"] == [
        "initial value/grad exceeded canary budget: 101.000s > 100.000s"
    ]


def test_canary_rejects_main_optimizer_walltime_growth():
    gate_payload = evaluate_lbfgs_ondevice_canary(
        progress_payload=_progress_payload(main_span_s=201.0),
        limits=_limits(),
    )

    assert gate_payload["passed"] is False
    assert gate_payload["failures"] == [
        "L-BFGS main optimizer exceeded canary budget: 201.000s > 200.000s"
    ]


def test_canary_rejects_negative_initial_value_grad_span():
    gate_payload = evaluate_lbfgs_ondevice_canary(
        progress_payload=_progress_payload(initial_span_s=-1.0),
        limits=_limits(),
    )

    assert gate_payload["passed"] is False
    assert gate_payload["failures"] == [
        "invalid initial value/grad canary span: -1.000s"
    ]
    assert gate_payload["measurements"]["initial_value_grad_s"] == pytest.approx(-1.0)


def test_canary_rejects_negative_main_optimizer_span():
    gate_payload = evaluate_lbfgs_ondevice_canary(
        progress_payload=_progress_payload(main_span_s=-1.0),
        limits=_limits(),
    )

    assert gate_payload["passed"] is False
    assert gate_payload["failures"] == [
        "invalid L-BFGS main-optimizer canary span: -1.000s"
    ]
    assert gate_payload["measurements"]["main_optimizer_s"] == pytest.approx(-1.0)


def test_canary_rejects_repeated_line_search_failure_anchor():
    gate_payload = evaluate_lbfgs_ondevice_canary(
        progress_payload=_progress_payload(repeat_failure=True),
        limits=_limits(),
    )

    assert gate_payload["passed"] is False
    assert len(gate_payload["failures"]) == 1
    assert gate_payload["failures"][0].startswith(
        "line-search failure anchor repeated too often:"
    )
    assert gate_payload["measurements"]["line_search_failure_anchor_count"] == 2


def test_canary_counts_one_line_search_anchor_per_optimizer_event():
    invalid_step = {
        "iteration": 0,
        "requested_initial_step": 1.0e-6,
        "first_tested_alpha": 1.0e-6,
        "line_search_failed": True,
        "ls_status": -9,
        "failure_reason": "line_search_failed",
    }
    progress_payload = _progress_payload()
    progress_payload["events"].append(
        {
            "label": "phase2_attempt_0_returned",
            "event_elapsed_s": 250.0,
            "result": {
                "invalid_step_log": [
                    dict(invalid_step),
                    dict(invalid_step),
                ]
            },
        }
    )

    gate_payload = evaluate_lbfgs_ondevice_canary(
        progress_payload=progress_payload,
        limits=_limits(),
    )

    assert gate_payload["passed"] is True
    assert gate_payload["measurements"]["line_search_failure_anchor_count"] == 1


def test_canary_ignores_nonzero_iteration_line_search_failures():
    progress_payload = _progress_payload()
    progress_payload["events"].extend(
        [
            {
                "label": "phase2_attempt_0_returned",
                "event_elapsed_s": 250.0,
                "result": {
                    "invalid_step_log": [
                        {
                            "iteration": 1,
                            "requested_initial_step": 1.0e-6,
                            "first_tested_alpha": 1.0e-6,
                            "line_search_failed": True,
                            "ls_status": -9,
                            "failure_reason": "line_search_failed",
                        }
                    ]
                },
            },
            {
                "label": "phase2_retry_1_returned",
                "event_elapsed_s": 300.0,
                "result": {
                    "invalid_step_log": [
                        {
                            "iteration": 1,
                            "requested_initial_step": 1.0e-6,
                            "first_tested_alpha": 1.0e-6,
                            "line_search_failed": True,
                            "ls_status": -9,
                            "failure_reason": "line_search_failed",
                        }
                    ]
                },
            },
        ]
    )

    gate_payload = evaluate_lbfgs_ondevice_canary(
        progress_payload=progress_payload,
        limits=_limits(),
    )

    assert gate_payload["passed"] is True
    assert gate_payload["measurements"]["line_search_failure_anchor_count"] == 0


def test_canary_line_search_anchor_includes_failure_reason():
    first_invalid_step = {
        "iteration": 0,
        "requested_initial_step": 1.0e-6,
        "first_tested_alpha": 1.0e-6,
        "line_search_failed": True,
        "ls_status": -9,
        "failure_reason": "line_search_failed",
    }
    second_invalid_step = dict(first_invalid_step)
    second_invalid_step["failure_reason"] = "nonfinite_trial"
    progress_payload = _progress_payload()
    progress_payload["events"].extend(
        [
            {
                "label": "phase2_attempt_0_returned",
                "event_elapsed_s": 250.0,
                "result": {"invalid_step_log": [first_invalid_step]},
            },
            {
                "label": "phase2_retry_1_returned",
                "event_elapsed_s": 300.0,
                "result": {"invalid_step_log": [second_invalid_step]},
            },
        ]
    )

    gate_payload = evaluate_lbfgs_ondevice_canary(
        progress_payload=progress_payload,
        limits=_limits(),
    )

    assert gate_payload["passed"] is True
    assert gate_payload["measurements"]["line_search_failure_anchor_count"] == 2
    assert len(set(gate_payload["measurements"]["line_search_failure_anchors"])) == 2


def test_canary_cli_writes_output_json(tmp_path, monkeypatch):
    progress_json = tmp_path / "outer_optimizer_progress.json"
    output_json = tmp_path / "canary.json"
    progress_json.write_text(json.dumps(_progress_payload()), encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "single_stage_lbfgs_ondevice_canary.py",
            "--progress-json",
            str(progress_json),
            "--output-json",
            str(output_json),
            "--max-initial-value-grad-s",
            "100",
            "--max-main-optimizer-s",
            "200",
        ],
    )

    main()

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["passed"] is True
