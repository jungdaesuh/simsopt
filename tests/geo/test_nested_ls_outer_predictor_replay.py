"""CPU tests for the nested-LS outer predictor replay probe.

Everything here is pure: ledger parsing and fingerprint validation, the
trust-region rule, the arm-selection rule, the branch discriminator, the
log-log fit, the budget threshold lookup, and the strict-JSON round trip.
No device, no solver, no mocking of the solver -- each test names one
observable behaviour and asserts a value.
"""

from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# The probe pins JAX to CPU for itself only when ``--dry-run`` is on argv.
# Importing it from pytest must never reach for CUDA, so pin it here.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

from benchmarks import nested_ls_outer_predictor_replay as probe
from simsopt_jax_adapters.geo.nested_ls_reduced_scale import DEFAULT_F3_B37_GPU_LANE


@pytest.fixture(scope="module")
def ledger_payload() -> dict[str, Any]:
    return json.loads(probe.LEDGER_PATH.read_text())


def _write_ledger(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "ledger.json"
    path.write_text(json.dumps(payload))
    return path


# --------------------------------------------------------------------------
# Ledger parsing and fingerprint validation
# --------------------------------------------------------------------------


def test_committed_ledger_loads_and_exposes_the_recorded_anchor() -> None:
    """The shipped ledger passes every fingerprint and yields x38/s38/x39."""

    ledger = probe.load_replay_ledger(probe.LEDGER_PATH)

    assert ledger.sha256 == probe.LEDGER_SHA256
    assert ledger.anchor_coil_dofs.shape == (probe.COIL_DOF_COUNT,)
    assert ledger.anchor_surface_dofs.shape == (probe.SURFACE_DOF_COUNT,)
    assert ledger.anchor_iota == probe.ANCHOR_IOTA
    assert ledger.anchor_G == probe.ANCHOR_G
    assert ledger.anchor_j == probe.ANCHOR_J
    assert ledger.coil_step_l2 == probe.COIL_STEP_L2
    assert (
        probe.sha256_float64(ledger.anchor_surface_dofs) == probe.ANCHOR_SURFACE_SHA256
    )
    assert probe.sha256_float64(ledger.anchor_coil_dofs) == probe.ANCHOR_COIL_SHA256


def test_ledger_records_the_two_states_leg3_and_leg4_replay(
    ledger_payload: dict[str, Any],
) -> None:
    """Eval 39 is the recorded capture and eval 43 is bitwise x38."""

    evals = ledger_payload["outer_evals"]
    trial = evals[probe.TRIAL_EVAL_INDEX]
    post = evals[probe.POST_POISON_EVAL_INDEX]

    assert trial["inner_iota"] == probe.RECORDED_TRIAL_IOTA
    assert trial["j"] == probe.RECORDED_TRIAL_J
    assert trial["inner_iterations"] == probe.RECORDED_TRIAL_INNER_ITERATIONS
    assert trial["inner_surface_sha256"] == probe.POISONED_SURFACE_SHA256
    assert post["rejection_reason"] == "inner_solve_failed"
    assert probe.sha256_float64(post["coil_dofs"]) == probe.ANCHOR_COIL_SHA256


def test_corrupted_ledger_sha256_fails_closed_naming_the_field(
    tmp_path: Path, ledger_payload: dict[str, Any]
) -> None:
    """A byte-level edit is caught before any field is read."""

    corrupted = dict(ledger_payload)
    corrupted["endpoint_j"] = 0.0
    path = _write_ledger(tmp_path, corrupted)

    with pytest.raises(SystemExit, match="ledger_sha256"):
        probe.load_replay_ledger(path)


def test_missing_ledger_path_fails_closed_naming_the_field(tmp_path: Path) -> None:
    """A path that does not exist is named, not swallowed."""

    with pytest.raises(SystemExit, match="ledger_path"):
        probe.load_replay_ledger(tmp_path / "absent.json")


def test_corrupted_anchor_iota_fingerprint_fails_closed_naming_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Moving the expected anchor iota is refused, and the field is named."""

    monkeypatch.setattr(probe, "ANCHOR_IOTA", 0.5)

    with pytest.raises(SystemExit, match="endpoint_iota"):
        probe.load_replay_ledger(probe.LEDGER_PATH)


def test_corrupted_poisoned_surface_fingerprint_fails_closed_naming_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The eval-39 anchor hash is a checked fingerprint, not a comment."""

    monkeypatch.setattr(probe, "POISONED_SURFACE_SHA256", "00" * 32)

    with pytest.raises(SystemExit, match=r"outer_evals\[39\]\.inner_surface_sha256"):
        probe.load_replay_ledger(probe.LEDGER_PATH)


def test_corrupted_coil_step_fingerprint_fails_closed_naming_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The x39-x38 displacement is asserted exactly."""

    monkeypatch.setattr(probe, "COIL_STEP_L2", 0.001)

    with pytest.raises(SystemExit, match=r"\|\|x39 - x38\|\|_2"):
        probe.load_replay_ledger(probe.LEDGER_PATH)


@pytest.mark.skipif(
    not DEFAULT_F3_B37_GPU_LANE.is_file(),
    reason="host-local archived F3 B37 fused lane bundle is not present",
)
def test_lane_blocks_on_disk_bind_to_the_ledger() -> None:
    """The archived lane is the world the recorded run used."""

    ledger = probe.load_replay_ledger(probe.LEDGER_PATH)
    binding = probe.check_lane_binds_to_ledger(ledger)

    assert binding["lane_meta_equals_ledger_lane"] is True
    assert binding["lane_coil_sha256"] == ledger.fingerprints["start_coil_sha256"]
    assert binding["lane_surface_sha256"] == ledger.fingerprints["start_surface_sha256"]


# --------------------------------------------------------------------------
# Trust-region rule (DESC tr_ratio: scale, never reject)
# --------------------------------------------------------------------------


def test_trust_region_leaves_a_step_under_the_cap_untouched() -> None:
    """Under the bound, the predicted step passes through unchanged."""

    anchor = np.array([3.0, 4.0], dtype=np.float64)  # ||anchor|| = 5, cap = 0.5
    delta = np.array([0.3, 0.0], dtype=np.float64)

    applied, raw, applied_norm, cap, scaled = probe.apply_trust_region(
        delta, anchor, 0.1
    )

    assert cap == pytest.approx(0.5)
    assert raw == pytest.approx(0.3)
    assert applied_norm == pytest.approx(0.3)
    assert scaled is False
    np.testing.assert_array_equal(applied, delta)


def test_trust_region_scales_an_oversized_step_to_the_cap() -> None:
    """Over the bound, the step is scaled to the bound and not rejected."""

    anchor = np.array([3.0, 4.0], dtype=np.float64)  # cap = 0.5
    delta = np.array([2.0, 0.0], dtype=np.float64)

    applied, raw, applied_norm, cap, scaled = probe.apply_trust_region(
        delta, anchor, 0.1
    )

    assert raw == pytest.approx(2.0)
    assert scaled is True
    assert applied_norm == pytest.approx(cap)
    # Direction preserved, magnitude clipped: DESC scales, it does not reject.
    np.testing.assert_allclose(applied, np.array([0.5, 0.0]))


def test_trust_region_boundary_step_is_not_scaled() -> None:
    """A step exactly at the cap is left alone (strict > triggers scaling)."""

    anchor = np.array([3.0, 4.0], dtype=np.float64)  # cap = 0.5
    delta = np.array([0.5, 0.0], dtype=np.float64)

    applied, raw, applied_norm, cap, scaled = probe.apply_trust_region(
        delta, anchor, 0.1
    )

    assert scaled is False
    assert raw == pytest.approx(cap)
    assert applied_norm == pytest.approx(cap)
    np.testing.assert_array_equal(applied, delta)


def test_zero_coil_step_gives_a_zero_predicted_step() -> None:
    """The delta_c = 0 invariant: no motion, no prediction, no scaling."""

    anchor = np.array([3.0, 4.0], dtype=np.float64)
    delta = np.zeros(2, dtype=np.float64)

    applied, raw, applied_norm, _cap, scaled = probe.apply_trust_region(
        delta, anchor, 0.1
    )

    assert raw == 0.0
    assert applied_norm == 0.0
    assert scaled is False
    np.testing.assert_array_equal(applied, np.zeros(2))


# --------------------------------------------------------------------------
# Envelope-gradient arm selection
# --------------------------------------------------------------------------


def test_arm_selection_keeps_the_prediction_when_it_improves_the_gradient() -> None:
    assert probe.select_arm(1.0, 0.25) == "predicted"


def test_arm_selection_falls_back_when_the_prediction_is_worse() -> None:
    assert probe.select_arm(0.25, 1.0) == "bare_anchor"


def test_arm_selection_keeps_the_prediction_on_a_tie() -> None:
    """A tie must keep the prediction so delta_c = 0 reproduces predictor-OFF."""

    assert probe.select_arm(1.0, 1.0) == "predicted"


# --------------------------------------------------------------------------
# Branch discriminator
# --------------------------------------------------------------------------


def test_recorded_capture_iota_is_labelled_the_wrong_branch() -> None:
    assert probe.classify_branch(probe.RECORDED_TRIAL_IOTA) == "recorded_wrong_branch"


def test_anchor_iota_is_labelled_the_anchor_branch() -> None:
    assert probe.classify_branch(probe.ANCHOR_IOTA) == "anchor_branch"


def test_the_recorded_capture_sits_inside_the_iota_branch_guard() -> None:
    """The 0.05 guard cannot separate the two branches -- hence proximity."""

    delta = abs(probe.RECORDED_TRIAL_IOTA - probe.ANCHOR_IOTA)

    assert delta == pytest.approx(0.0081058863956, abs=1.0e-12)
    assert delta < 0.05


# --------------------------------------------------------------------------
# Log-log fit (hand-computed, not self-referential)
# --------------------------------------------------------------------------


def test_loglog_slope_recovers_an_exact_power_law() -> None:
    """y = 3 * x**2 has slope 2, intercept log10(3), zero residual."""

    x = (1.0e-8, 1.0e-6, 1.0e-4)
    y = tuple(3.0 * value**2 for value in x)

    slope, intercept, rms = probe.loglog_slope(x, y)

    assert slope == pytest.approx(2.0, rel=1.0e-12)
    assert intercept == pytest.approx(math.log10(3.0), rel=1.0e-12)
    assert rms == pytest.approx(0.0, abs=1.0e-12)


def test_loglog_slope_recovers_a_unit_slope_from_two_points() -> None:
    """Two points on y = x give slope 1 and intercept 0 exactly."""

    slope, intercept, rms = probe.loglog_slope((1.0e-10, 1.0e-4), (1.0e-10, 1.0e-4))

    assert slope == pytest.approx(1.0, rel=1.0e-12)
    assert intercept == pytest.approx(0.0, abs=1.0e-12)
    assert rms == pytest.approx(0.0, abs=1.0e-12)


def test_loglog_slope_reports_the_scatter_of_an_imperfect_fit() -> None:
    """Hand-computed: log10 y = (0, 1, 1) against log10 x = (0, 1, 2)."""

    slope, intercept, rms = probe.loglog_slope((1.0, 10.0, 100.0), (1.0, 10.0, 10.0))

    # mean_x = 1, mean_y = 2/3; slope = sum((x-1)(y-2/3)) / sum((x-1)^2)
    #        = ((-1)(-2/3) + 0 + (1)(1/3)) / 2 = 0.5; intercept = 2/3 - 1/2 = 1/6
    assert slope == pytest.approx(0.5, rel=1.0e-12)
    assert intercept == pytest.approx(1.0 / 6.0, rel=1.0e-12)
    # fitted = (1/6, 2/3, 7/6); residuals = (-1/6, 1/3, -1/6)
    # sum of squares = 1/36 + 4/36 + 1/36 = 1/6 -> rms = sqrt(1/18)
    assert rms == pytest.approx(math.sqrt(1.0 / 18.0), rel=1.0e-12)


def test_loglog_slope_fails_closed_on_a_non_positive_ordinate() -> None:
    """A zero relative error has no log10 and must be excluded, not fitted."""

    with pytest.raises(SystemExit, match="loglog_fit_ordinate_positive"):
        probe.loglog_slope((1.0e-8, 1.0e-6), (0.0, 1.0e-6))


def test_loglog_slope_fails_closed_on_a_single_point() -> None:
    with pytest.raises(SystemExit, match="loglog_fit_point_count"):
        probe.loglog_slope((1.0e-8,), (1.0e-8,))


def test_loglog_slope_fails_closed_on_mismatched_lengths() -> None:
    with pytest.raises(SystemExit, match="loglog_fit_point_counts"):
        probe.loglog_slope((1.0e-8, 1.0e-6), (1.0e-8,))


# --------------------------------------------------------------------------
# Budget threshold lookup
# --------------------------------------------------------------------------


def test_budget_lookup_returns_the_loosest_qualifying_residual() -> None:
    """Hand-computed: only the 1e-13 and 1e-9 rungs clear a 1e-8 bound."""

    rows = (
        (1.0e-15, 0.0),
        (1.0e-9, 5.0e-9),
        (1.0e-6, 2.0e-5),
    )

    assert probe.largest_residual_under_threshold(rows, 1.0e-8) == 1.0e-9
    assert probe.largest_residual_under_threshold(rows, 1.0e-4) == 1.0e-6
    assert probe.largest_residual_under_threshold(rows, 1.0e-10) == 1.0e-15


def test_budget_lookup_returns_none_when_no_rung_qualifies() -> None:
    rows = ((1.0e-9, 5.0e-9), (1.0e-6, 2.0e-5))

    assert probe.largest_residual_under_threshold(rows, 1.0e-12) is None


def test_budget_lookup_includes_a_rung_exactly_on_the_threshold() -> None:
    rows = ((1.0e-9, 1.0e-8),)

    assert probe.largest_residual_under_threshold(rows, 1.0e-8) == 1.0e-9


# --------------------------------------------------------------------------
# Evidence document
# --------------------------------------------------------------------------


def test_claim_boundary_disclaims_every_speed_and_timing_claim() -> None:
    boundary = probe.claim_boundary()

    assert boundary["nested_speed_claim"] is False
    assert boundary["timing_content"] is False
    assert boundary["inherits_f3_7_70x"] is False
    assert boundary["predictor_wired_into_children"] is False
    assert boundary["single_state_measurement"] is True


def test_evidence_document_round_trips_through_strict_json() -> None:
    """dump_strict_json is the writer, so the payload must survive it."""

    ledger = probe.load_replay_ledger(probe.LEDGER_PATH)
    payload = {
        "claim_boundary": probe.claim_boundary(),
        "ledger": {
            "path": str(ledger.path),
            "sha256": ledger.sha256,
            "fingerprints": ledger.fingerprints,
        },
        "schema": probe.SCHEMA,
        "solve_plan": probe.solve_plan("all"),
    }

    text = probe.dump_strict_json(payload)
    restored = json.loads(text)

    assert restored == payload
    assert restored["ledger"]["sha256"] == probe.LEDGER_SHA256
    assert text.endswith("\n")


def test_strict_json_refuses_a_non_finite_number() -> None:
    """A NaN cannot reach the receipt: allow_nan=False is the writer's gate."""

    with pytest.raises(ValueError, match="Out of range float"):
        probe.dump_strict_json({"broken": float("nan")})


def test_solve_plan_counts_match_the_declared_budget() -> None:
    """4 predictor solves, 6 tolerance solves, 10 together."""

    assert len(probe.solve_plan("predictor")) == 4
    assert len(probe.solve_plan("tolerance-budget")) == len(probe.TOLERANCE_RUNGS) + 1
    assert len(probe.solve_plan("all")) == 10


def test_default_out_path_lands_under_the_evidence_directory() -> None:
    assert probe.default_out_path("all").parent == probe.EVIDENCE
    assert probe.default_out_path("predictor").name.endswith(".predictor.json")
    assert probe.default_out_path("all").name.endswith(
        f"{datetime.now(timezone.utc).strftime('%Y%m%d')}.json"
    )


# --------------------------------------------------------------------------
# Leg 4's regenerated-anchor physics fingerprint
# --------------------------------------------------------------------------


def _fingerprint_check(
    verdict: probe.AnchorPhysicsVerdict, name: str
) -> dict[str, object]:
    return next(check for check in verdict.checks if check["name"] == name)


def test_the_recorded_anchor_passes_its_own_fingerprint() -> None:
    """The gate accepts the state it was derived from."""

    verdict = probe.check_regenerated_anchor_physics(
        iota=probe.RECORDED_TRIAL_IOTA,
        iteration_count=probe.RECORDED_TRIAL_INNER_ITERATIONS,
        objective_j=probe.RECORDED_TRIAL_J,
    )

    assert verdict.passed is True
    assert verdict.failures == ()
    assert len(verdict.checks) == 4


def test_the_committed_replay_regeneration_passes_the_physics_fingerprint() -> None:
    """The regeneration a bitwise gate rejects is accepted on physics.

    These are the exact values the committed transaction replay printed at
    A2/B2 -- the run whose surface hash is 7daf6c32..., not the ledger's
    052923e7... Physically it is the same poisoned anchor.
    """

    verdict = probe.check_regenerated_anchor_physics(
        iota=probe.COMMITTED_REPLAY_TRIAL_IOTA,
        iteration_count=9,
        objective_j=0.07471552895095307,
    )

    assert verdict.passed is True
    iota_check = _fingerprint_check(verdict, "inner_iota_within_tolerance")
    assert iota_check["measure"] == probe.COMMITTED_REPLAY_TRIAL_IOTA_ABS_DRIFT
    branch = _fingerprint_check(verdict, "inner_iota_on_the_recorded_wrong_branch")
    assert branch["passed"] is True
    j_check = _fingerprint_check(verdict, "objective_j_within_relative_band")
    assert j_check["measure"] == probe.COMMITTED_REPLAY_TRIAL_J_REL_DRIFT == 0.0


def test_an_anchor_on_the_healthy_branch_fails_closed() -> None:
    """A regeneration that is not poisoned cannot serve leg 4."""

    verdict = probe.check_regenerated_anchor_physics(
        iota=probe.ANCHOR_IOTA,
        iteration_count=probe.RECORDED_TRIAL_INNER_ITERATIONS,
        objective_j=probe.RECORDED_TRIAL_J,
    )

    assert verdict.passed is False
    assert "inner_iota_within_tolerance" in verdict.failures
    assert "inner_iota_on_the_recorded_wrong_branch" in verdict.failures


def test_a_drifted_iota_beyond_the_tolerance_fails_closed() -> None:
    """Twice the band fails; the branch-side check still passes."""

    verdict = probe.check_regenerated_anchor_physics(
        iota=probe.RECORDED_TRIAL_IOTA + 2.0 * probe.REGEN_IOTA_ABS_TOL,
        iteration_count=probe.RECORDED_TRIAL_INNER_ITERATIONS,
        objective_j=probe.RECORDED_TRIAL_J,
    )

    assert verdict.passed is False
    assert verdict.failures == ("inner_iota_within_tolerance",)


def test_a_different_inner_iteration_count_fails_closed() -> None:
    """The iteration count takes no tolerance."""

    verdict = probe.check_regenerated_anchor_physics(
        iota=probe.RECORDED_TRIAL_IOTA,
        iteration_count=probe.RECORDED_TRIAL_INNER_ITERATIONS + 1,
        objective_j=probe.RECORDED_TRIAL_J,
    )

    assert verdict.passed is False
    assert verdict.failures == ("inner_iterations_exact",)
    assert _fingerprint_check(verdict, "inner_iterations_exact")["measure"] == 1


def test_a_drifted_objective_beyond_the_band_fails_closed() -> None:
    """1e-6 relative is a thousand times the band and is refused."""

    verdict = probe.check_regenerated_anchor_physics(
        iota=probe.RECORDED_TRIAL_IOTA,
        iteration_count=probe.RECORDED_TRIAL_INNER_ITERATIONS,
        objective_j=probe.RECORDED_TRIAL_J * (1.0 + 1.0e-6),
    )

    assert verdict.passed is False
    assert verdict.failures == ("objective_j_within_relative_band",)


def test_a_measure_exactly_on_the_iota_tolerance_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The comparison is <=, not <, so the boundary is inside the band."""

    iota = probe.RECORDED_TRIAL_IOTA + 1.0e-9
    monkeypatch.setattr(
        probe, "REGEN_IOTA_ABS_TOL", abs(iota - probe.RECORDED_TRIAL_IOTA)
    )
    verdict = probe.check_regenerated_anchor_physics(
        iota=iota,
        iteration_count=probe.RECORDED_TRIAL_INNER_ITERATIONS,
        objective_j=probe.RECORDED_TRIAL_J,
    )

    check = _fingerprint_check(verdict, "inner_iota_within_tolerance")
    assert check["measure"] == check["tolerance"]
    assert check["passed"] is True


def test_a_measure_exactly_on_the_objective_band_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same boundary semantics on the relative J band."""

    objective = probe.RECORDED_TRIAL_J * (1.0 + 1.0e-9)
    relative = abs(objective - probe.RECORDED_TRIAL_J) / abs(probe.RECORDED_TRIAL_J)
    monkeypatch.setattr(probe, "REGEN_J_REL_TOL", relative)
    verdict = probe.check_regenerated_anchor_physics(
        iota=probe.RECORDED_TRIAL_IOTA,
        iteration_count=probe.RECORDED_TRIAL_INNER_ITERATIONS,
        objective_j=objective,
    )

    check = _fingerprint_check(verdict, "objective_j_within_relative_band")
    assert check["measure"] == check["tolerance"]
    assert check["passed"] is True


def test_every_fingerprint_check_states_where_its_tolerance_came_from() -> None:
    """A tolerance without a cited measurement is a number someone made up."""

    verdict = probe.check_regenerated_anchor_physics(
        iota=probe.RECORDED_TRIAL_IOTA,
        iteration_count=probe.RECORDED_TRIAL_INNER_ITERATIONS,
        objective_j=probe.RECORDED_TRIAL_J,
    )

    for check in verdict.checks:
        source = check["tolerance_source"]
        assert isinstance(source, str)
        assert len(source) > 20


def test_the_iota_band_resolves_the_branch_separation_it_must_discriminate() -> None:
    """Far above the measured drift, far below the gap it has to see."""

    assert (
        probe.REGEN_IOTA_ABS_TOL > 1.0e6 * probe.COMMITTED_REPLAY_TRIAL_IOTA_ABS_DRIFT
    )
    assert probe.REGEN_IOTA_ABS_TOL < 1.0e-5 * probe.RECORDED_BRANCH_SEPARATION
    assert probe.RECORDED_BRANCH_SEPARATION == pytest.approx(
        abs(probe.RECORDED_TRIAL_IOTA - probe.ANCHOR_IOTA), rel=1.0e-15
    )


def test_the_objective_band_is_far_above_the_measured_converged_drift() -> None:
    assert probe.REGEN_J_REL_TOL > 1.0e6 * probe.COMMITTED_REPLAY_ANCHOR_J_REL_DRIFT
    assert probe.COMMITTED_REPLAY_TRIAL_J_REL_DRIFT == 0.0


# --------------------------------------------------------------------------
# The published trajectory-drift finding
# --------------------------------------------------------------------------


def test_the_drift_finding_names_three_discrepancies_each_with_a_source() -> None:
    """The finding must be citable without reading the transcript."""

    rows = probe.trajectory_drift_discrepancies("deadbeef")

    assert len(rows) == 3
    assert [row["quantity"] for row in rows] == [
        "regenerated s39 surface sha256",
        "eight-term J at the eval-38 anchor",
        "inner residual of the failed x38 solve",
    ]
    for row in rows:
        assert row["ledger_source"]
        assert "replay log" in str(row["committed_replay_source"])
    assert rows[0]["this_run"] == "deadbeef"


def test_the_drift_finding_separates_converged_from_non_converged_drift() -> None:
    """2 ULP on converged quantities, 1.1e-4 on the non-converged one."""

    finding = probe.trajectory_drift_finding(None)

    assert finding["converged_quantity_drift_ulps"] == 2.0
    assert finding["non_converged_quantity_relative_drift"] == pytest.approx(
        1.1398182189350828e-4, rel=1.0e-12
    )
    assert (
        probe.COMMITTED_REPLAY_FAILED_RESIDUAL_REL_DRIFT
        > 1.0e11 * probe.COMMITTED_REPLAY_ANCHOR_J_REL_DRIFT
    )


def test_the_drift_finding_round_trips_through_strict_json() -> None:
    finding = probe.trajectory_drift_finding(None)

    assert json.loads(probe.dump_strict_json(finding)) == finding


def test_the_recorded_and_regenerated_anchor_hashes_are_disclosed_not_gated() -> None:
    """The hash is published; a bitwise match is not attainable, so not gated."""

    assert probe.POISONED_SURFACE_SHA256.startswith("052923e7")
    assert probe.COMMITTED_REPLAY_REGENERATED_SURFACE_SHA256_PREFIX == (
        "7daf6c3223b66041"
    )
    assert not probe.POISONED_SURFACE_SHA256.startswith(
        probe.COMMITTED_REPLAY_REGENERATED_SURFACE_SHA256_PREFIX
    )
    # The physics gate does not consult a hash at all.
    verdict = probe.check_regenerated_anchor_physics(
        iota=probe.COMMITTED_REPLAY_TRIAL_IOTA,
        iteration_count=9,
        objective_j=0.07471552895095307,
    )
    assert verdict.passed is True
