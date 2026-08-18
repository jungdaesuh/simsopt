"""Reduction-contract tests for the finite-build native/GPU benchmark.

Synthetic temporary raw rows exercise the WIN, CLOSED_BOUNDED_NEGATIVE, and
NOT_PRODUCED verdicts of the validator.  They make no timing claim and do not
re-implement the benchmark.  Every synthetic row carries the quality contract
shape: a converged reference endpoint that sets the target, and a truncated
anchor endpoint -- captured from the converged run's own trajectory -- that
every truncated lane is measured against.  Native timed legs follow the
time-to-quality protocol: they stop at the frozen rung through their solver
callback and publish ``stopped_at_target``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from benchmarks.stage_two_finitebuild_native_gpu import (
    _DISCLOSED_SOURCE_PINS,
    _PHYSICS_SOURCE_PINS,
    FINAL_PAIR_COUNT,
    FINAL_PAIR_WALL_NATIVE_ROLE,
    FINAL_PAIR_WARM_NATIVE_ROLE,
    FINAL_WARM_REPETITIONS,
    GATE_ENDPOINT_ATOL,
    GATE_ENDPOINT_RTOL,
    GATE_GRADIENT_NORM_FLOOR,
    GATE_GRADIENT_NORM_MARGIN,
    GATE_OBJECTIVE_MARGIN,
    GATE_QUALITY_CAP_MARGIN,
    GATE_REFERENCE_STEPS,
    GPU_BUDGET_SWEEP,
    GPU_HOST_OMP_THREADS,
    JAX_BISECT_ROLE,
    JAX_CROSSING_ROLE,
    JAX_HISTORY_SWEEP,
    KERNEL_CANARY_REPETITIONS,
    NATIVE_HISTORY_SWEEP,
    NATIVE_MATRIX_TIMED_ROLE,
    NATIVE_OMP_SWEEP,
    SELECTION_REPETITIONS,
    SHIPPED_DEFAULT_DISCLOSURE_ROLE,
    VERDICT_CLOSED,
    VERDICT_NOT_PRODUCED,
    VERDICT_WIN,
    BenchmarkError,
    _canonical_json_bytes,
    _derive_quality_contract,
    _fp64_conformance_failure,
    _gate_scaled_target,
    _gate_source_conformance,
    _leg_environment,
    _sha256_bytes,
    _source_fingerprints,
    _threading_conformance_failure,
    _write_json_exclusive,
    evaluate_quality_gate,
    first_qualifying_iteration,
    hlo_operation_census,
    reduce_baseline,
    reduce_final_pairs,
    reduce_hlo_diff,
    reduce_jax_sweep,
    reduce_kernel_canary,
    reduce_native_matrix,
    validate_run,
)

# The converged reference stops at 9.995, so the frozen target is 10.004995
# and the truncated anchor -- the first iteration to clear it -- sits at
# 10.0, exactly as the real gate derivation produces.
_CONVERGED_OBJECTIVE = 9.995
# The one frozen objective scale: the contract publishes it inside its
# converged reference solver record, and every scaled target derives from
# there rather than being restated per row.
_OBJECTIVE_SCALE = 2.0


def _truncated_reference_endpoint() -> dict[str, object]:
    return {
        "solution": [0.1, 0.2, 0.3],
        "objective": 10.0,
        "gradient": [1.0e-3, -5.0e-4, 2.0e-4],
        "gradient_inf_norm": 1.0e-3,
        "squared_flux": 4.0,
        "length_penalty": 3.0,
        "distance_penalty": 3.0,
        "minimum_clearance": 0.12,
        "coil_lengths": [5.0, 5.1, 5.2, 5.3],
    }


def _converged_endpoint() -> dict[str, object]:
    endpoint = _truncated_reference_endpoint()
    endpoint["objective"] = _CONVERGED_OBJECTIVE
    endpoint["squared_flux"] = 3.995
    endpoint["gradient"] = [1.0e-5, -5.0e-6, 2.0e-6]
    endpoint["gradient_inf_norm"] = 1.0e-5
    return endpoint


def _gate() -> dict[str, object]:
    return {
        "schema": "stage-two-finitebuild-quality-contract-v2",
        "case_id": "native-stage-two-optimization-finitebuild",
        "initial_objective": 100.0,
        "initial_parameters": [0.0, 0.0, 0.0],
        "solver": {"converged_reference": {"objective_scale": _OBJECTIVE_SCALE}},
        "converged_endpoint": _converged_endpoint(),
        "reference_endpoint": _truncated_reference_endpoint(),
        "reference_budget": 21,
        "target_objective": GATE_OBJECTIVE_MARGIN * _CONVERGED_OBJECTIVE,
        "tolerances": {
            "objective_margin": GATE_OBJECTIVE_MARGIN,
            "endpoint_rtol": GATE_ENDPOINT_RTOL,
            "endpoint_atol": GATE_ENDPOINT_ATOL,
            "gradient_norm_margin": GATE_GRADIENT_NORM_MARGIN,
            "gradient_norm_floor": GATE_GRADIENT_NORM_FLOOR,
            "quality_cap_margin": GATE_QUALITY_CAP_MARGIN,
        },
    }


def _gate_sha256() -> str:
    return _sha256_bytes(_canonical_json_bytes(_gate()))


def _fired(verdict: dict[str, object], fragment: str) -> bool:
    return any(fragment in failure for failure in verdict["failures"])


# ---------------------------------------------------------------------------
# Quality gate clauses
# ---------------------------------------------------------------------------


def test_quality_gate_accepts_the_truncated_reference_endpoint() -> None:
    verdict = evaluate_quality_gate(_gate(), _truncated_reference_endpoint())
    assert verdict["eligible"] is True
    assert verdict["failures"] == []


def test_quality_gate_accepts_an_endpoint_better_than_the_reference() -> None:
    """Converging past the truncated reference is the win, never a failure."""
    better = _truncated_reference_endpoint()
    better["objective"] = 9.5
    better["squared_flux"] = 3.5
    better["length_penalty"] = 2.8
    better["distance_penalty"] = 2.9
    better["gradient_inf_norm"] = 5.0e-4
    verdict = evaluate_quality_gate(_gate(), better)
    assert verdict["eligible"] is True, verdict["failures"]


def test_quality_gate_rejects_a_gradient_norm_three_times_the_reference() -> None:
    steep = _truncated_reference_endpoint()
    steep["gradient_inf_norm"] = 3.0e-3
    verdict = evaluate_quality_gate(_gate(), steep)
    assert verdict["eligible"] is False
    assert _fired(verdict, "gradient infinity norm ratio")


def test_quality_gate_names_the_clause_that_fired() -> None:
    gate = _gate()

    slow = _truncated_reference_endpoint()
    slow["objective"] = 10.2
    assert _fired(evaluate_quality_gate(gate, slow), "misses target")

    touching = _truncated_reference_endpoint()
    touching["minimum_clearance"] = -0.01
    assert _fired(
        evaluate_quality_gate(gate, touching), "minimum clearance is not positive"
    )

    flared = _truncated_reference_endpoint()
    flared["squared_flux"] = 10.0
    assert _fired(evaluate_quality_gate(gate, flared), "squared_flux 10.0 exceeds cap")

    stretched = _truncated_reference_endpoint()
    stretched["coil_lengths"] = [5.0, 5.1, 5.2, 9.0]
    assert _fired(evaluate_quality_gate(gate, stretched), "coil length 3")

    incomplete = _truncated_reference_endpoint()
    del incomplete["coil_lengths"]
    assert _fired(
        evaluate_quality_gate(gate, incomplete), "missing endpoint field coil_lengths"
    )

    nonfinite = _truncated_reference_endpoint()
    nonfinite["gradient"] = [float("nan"), 0.0, 0.0]
    assert _fired(evaluate_quality_gate(gate, nonfinite), "nonfinite endpoint field")


def test_quality_gate_reads_the_caps_from_their_own_tolerance_key() -> None:
    """The one-sided caps are their own clause, not the gradient-norm slack."""
    gate = _gate()
    gate["tolerances"]["quality_cap_margin"] = 1.0
    gate["tolerances"]["gradient_norm_margin"] = 5.0
    flared = _truncated_reference_endpoint()
    flared["squared_flux"] = 4.2
    verdict = evaluate_quality_gate(gate, flared)
    assert verdict["eligible"] is False
    assert _fired(verdict, "squared_flux 4.2 exceeds cap")

    steep = _truncated_reference_endpoint()
    steep["gradient_inf_norm"] = 4.0e-3
    assert evaluate_quality_gate(gate, steep)["eligible"] is True


def test_source_fingerprints_bind_the_preregistered_plan() -> None:
    fingerprints = _source_fingerprints("deadbeef")
    assert set(fingerprints) == {
        "git_commit",
        "objective_module_sha256",
        "parity_case_sha256",
        "benchmark_sha256",
        "plan_sha256",
        "successor_plan_sha256",
    }
    assert len(str(fingerprints["plan_sha256"])) == 64
    assert len(str(fingerprints["successor_plan_sha256"])) == 64


def test_quality_gate_does_not_reject_a_lower_penalty_than_the_reference() -> None:
    """The one-sided caps must not read as two-sided bands."""
    relaxed = _truncated_reference_endpoint()
    relaxed["length_penalty"] = 0.0
    relaxed["distance_penalty"] = 0.0
    relaxed["objective"] = 4.0
    assert evaluate_quality_gate(_gate(), relaxed)["eligible"] is True


# ---------------------------------------------------------------------------
# Gate derivation from the single reference run
# ---------------------------------------------------------------------------


def _gate_reference_row() -> dict[str, object]:
    """The converged reference leg, carrying its own captured anchor."""
    return {
        "leg_id": "native-reference",
        "fingerprints": {"bundle": "0" * 8},
        "identity": {"git": {"commit": "deadbeef"}},
        "solver": {
            "driver": "scipy_lbfgsb_finite_build",
            "max_steps": GATE_REFERENCE_STEPS,
            "history": 400,
            "objective_scale": _OBJECTIVE_SCALE,
            "status": 1,
            "success": False,
            "nit": GATE_REFERENCE_STEPS,
            "nfev": 500,
            "stopped_at_target": False,
        },
        "initial_parameters": [0.0, 0.0, 0.0],
        "initial_state": {"objective": 100.0},
        "endpoint": _converged_endpoint(),
        "anchor_endpoint": _truncated_reference_endpoint(),
        "anchor_budget": 21,
        # |g|inf over the last 21 accepted iterates up to the anchor; the
        # median (1.3e-3) sits above the anchor draw (1.0e-3), matching the
        # measured oscillation structure the successor clause exists for.
        "anchor_gradient_window": [1.0e-3, 1.3e-3, 1.6e-3] * 7,
    }


def test_gate_contract_anchors_on_the_reference_run_own_trajectory() -> None:
    """One leg freezes the contract: no replay, so no cross-process fork."""
    gate = _derive_quality_contract(_gate_reference_row())
    assert gate["converged_endpoint"] == _converged_endpoint()
    assert gate["reference_endpoint"] == _truncated_reference_endpoint()
    assert gate["reference_budget"] == 21
    assert gate["target_objective"] == GATE_OBJECTIVE_MARGIN * _CONVERGED_OBJECTIVE
    assert set(gate["solver"]) == {"converged_reference"}
    # Every timed native solve stops at this rung, derived from the contract.
    assert _gate_scaled_target(gate) == _OBJECTIVE_SCALE * float(
        gate["target_objective"]
    )
    # The successor gradient clause: window-median scale, published with its
    # audit inputs.
    assert gate["gradient_reference_scale"] == pytest.approx(1.3e-3)
    assert gate["anchor_gradient_window_count"] == 21
    assert gate["gradient_scale_to_anchor_ratio"] == pytest.approx(1.3)


def test_gate_halts_when_the_cap_would_admit_less_than_archived_landings() -> None:
    """The freeze-time audit is a halt, not a disclosure."""
    row = _gate_reference_row()
    # Window median 0.9e-3 against anchor 1.0e-3: 2.3 x 0.9e-3 < 2.41 x 1e-3.
    row["anchor_gradient_window"] = [0.8e-3, 0.9e-3, 1.0e-3] * 7
    with pytest.raises(BenchmarkError, match="freeze audit failed"):
        _derive_quality_contract(row)


def test_gradient_clause_uses_the_window_scale_when_published() -> None:
    """v4 contracts divide by the window median; v3 contracts keep the anchor."""
    gate = _gate()
    steep = _truncated_reference_endpoint()
    steep["gradient_inf_norm"] = 2.5e-3
    # v3 shape (no scale): ratio 2.5 exceeds the 2.3 margin.
    assert _fired(evaluate_quality_gate(gate, steep), "gradient infinity norm")
    # v4 shape: scale 1.3e-3 makes the same endpoint ratio 1.92 -- eligible.
    v4 = dict(gate, gradient_reference_scale=1.3e-3)
    assert evaluate_quality_gate(v4, steep)["eligible"] is True


def test_gate_refuses_to_freeze_on_a_short_reference_run() -> None:
    """The reference carries no stop target, so it must reach its cap."""
    row = _gate_reference_row()
    row["solver"]["nit"] = 399
    with pytest.raises(BenchmarkError, match="stopped at 399"):
        _derive_quality_contract(row)


def test_gate_refuses_to_freeze_when_the_anchor_misses_its_own_target() -> None:
    """Same-trajectory construction guarantees it; a miss is an internal error."""
    row = _gate_reference_row()
    row["anchor_endpoint"]["objective"] = 10.2
    with pytest.raises(BenchmarkError, match="internally inconsistent"):
        _derive_quality_contract(row)


# ---------------------------------------------------------------------------
# Final pairs
# ---------------------------------------------------------------------------


def _selection() -> dict[str, object]:
    # The native half freezes no budget: its legs stop at the frozen rung and
    # publish the iteration count they took, informational only.
    return {
        "native": {"omp": 8, "history": 20, "median_nit": 21},
        "jax": {"history": 10, "budget": 80},
    }


def _uniform(value: float) -> list[float]:
    return [value] * FINAL_PAIR_COUNT


# A warm leg's own subprocess wall carries its discarded warm-up solve and
# every repetition, so it is deliberately enormous here: a reducer that took
# the wall numerator from the native warm leg would publish a 3x wall ratio
# out of protocol alone.  Only the single-solve wall legs may be numerators.
_WARM_LEG_WALL_SECONDS = 300.0
_PAIR_WALL_BASE_SECONDS = 100.0


def _pair_rows(
    *,
    warm_ratios: list[float],
    wall_ratios: list[float],
    gpu_endpoint: dict[str, object] | None = None,
    oracle_endpoint: dict[str, object] | None = None,
    native_warm_repetitions: int = FINAL_WARM_REPETITIONS,
    native_stopped_at_target: bool = True,
) -> list[dict[str, object]]:
    """Five pairs of four timed legs: warm and wall, in both lanes."""
    gate_sha256 = _gate_sha256()
    selection = _selection()
    rows: list[dict[str, object]] = []
    for pair_index in range(FINAL_PAIR_COUNT):
        gpu_seconds = 10.0
        native_seconds = gpu_seconds * warm_ratios[pair_index]
        for lane, role, leg_id, endpoint, timings, wall, warm_protocol, reps in (
            (
                "native",
                FINAL_PAIR_WARM_NATIVE_ROLE,
                f"native-warm-pair{pair_index}",
                _truncated_reference_endpoint(),
                {"warm_solve_seconds": [native_seconds] * native_warm_repetitions},
                _WARM_LEG_WALL_SECONDS,
                True,
                native_warm_repetitions,
            ),
            (
                "native",
                FINAL_PAIR_WALL_NATIVE_ROLE,
                f"native-wall-pair{pair_index}",
                _truncated_reference_endpoint(),
                {"warm_solve_seconds": [native_seconds]},
                _PAIR_WALL_BASE_SECONDS * wall_ratios[pair_index],
                False,
                1,
            ),
            (
                "jax",
                "warm",
                f"jax-warm-pair{pair_index}",
                gpu_endpoint or _truncated_reference_endpoint(),
                {"warm_solve_seconds": [gpu_seconds] * FINAL_WARM_REPETITIONS},
                _WARM_LEG_WALL_SECONDS,
                None,
                FINAL_WARM_REPETITIONS,
            ),
            (
                "jax",
                "wall",
                f"jax-wall-pair{pair_index}",
                gpu_endpoint or _truncated_reference_endpoint(),
                {"warm_solve_seconds": []},
                _PAIR_WALL_BASE_SECONDS,
                None,
                0,
            ),
        ):
            # The native lane carries the 400-iteration cap and stops at the
            # frozen rung; only the GPU lane still runs a frozen budget.
            native = lane == "native"
            max_steps = (
                GATE_REFERENCE_STEPS if native else int(selection["jax"]["budget"])
            )
            specification: dict[str, object] = {
                "pair_index": pair_index,
                "role": role,
                "max_steps": max_steps,
                "history": int(selection[lane]["history"]),
                "warm_repetitions": reps,
                "omp_threads": (
                    int(selection["native"]["omp"]) if native else GPU_HOST_OMP_THREADS
                ),
            }
            if warm_protocol is not None:
                specification["warm_protocol"] = warm_protocol
            solver: dict[str, object] = {
                "nit": max_steps,
                "nfev": max_steps + 4,
                "status": 1,
            }
            if native:
                solver["nit"] = (
                    _MATRIX_NIT if native_stopped_at_target else GATE_REFERENCE_STEPS
                )
                solver["nfev"] = int(solver["nit"]) + 4
                solver["status"] = 99 if native_stopped_at_target else 1
                solver["stopped_at_target"] = native_stopped_at_target
            rows.append(
                {
                    "leg_id": leg_id,
                    "lane": lane,
                    "kind": "native-solve" if native else "jax-solve",
                    "gate_sha256": gate_sha256,
                    "specification": specification,
                    "endpoint": endpoint,
                    "timings": timings,
                    "solver": solver,
                }
            )
            rows.append(
                {
                    "record": "launch",
                    "leg_id": leg_id,
                    "process_wall_seconds": wall,
                }
            )
            if lane == "jax":
                rows.append(
                    {
                        "leg_id": f"native-endpoint-{leg_id}",
                        "lane": "native",
                        "kind": "native-endpoint-eval",
                        "gate_sha256": gate_sha256,
                        "subject_leg_id": leg_id,
                        "specification": {
                            "role": "endpoint-eval",
                            "subject_leg_id": leg_id,
                        },
                        "endpoint": oracle_endpoint
                        or gpu_endpoint
                        or _truncated_reference_endpoint(),
                    }
                )
    return rows


def _mutate_leg(
    rows: list[dict[str, object]], leg_id: str, field: str, **updates: object
) -> list[dict[str, object]]:
    """Apply ``updates`` to one data row's ``field`` mapping, in place."""
    for row in rows:
        if row.get("leg_id") == leg_id and field in row:
            row[field].update(updates)
    return rows


def test_final_pairs_win_requires_both_medians_and_every_pair() -> None:
    verdict = reduce_final_pairs(
        _gate(),
        _selection(),
        _pair_rows(warm_ratios=_uniform(1.3), wall_ratios=_uniform(1.2)),
    )
    assert verdict["verdict"] == VERDICT_WIN, verdict
    assert verdict["warm_solve_median_ratio"] > 1.1


def test_final_pairs_reject_an_off_contract_native_warm_repetition_count() -> None:
    """The finding-1 scenario is now rejected outright, not merely survived.

    A native warm leg that ran two timed solves instead of the contract's
    three is an asymmetric-protocol row set; the validator refuses to compute
    any ratio from it rather than trusting the orchestrator's split.
    """
    verdict = reduce_final_pairs(
        _gate(),
        _selection(),
        _pair_rows(
            warm_ratios=_uniform(1.3),
            wall_ratios=_uniform(1.2),
            native_warm_repetitions=2,
        ),
    )
    assert verdict["verdict"] == VERDICT_NOT_PRODUCED, verdict
    assert "warm_repetitions=2" in verdict["reason"]


def test_final_pairs_not_produced_when_a_native_leg_never_reaches_the_rung() -> None:
    """A native leg that exhausted its cap timed no quality at all."""
    rows = _mutate_leg(
        _pair_rows(warm_ratios=_uniform(1.3), wall_ratios=_uniform(1.3)),
        "native-wall-pair1",
        "solver",
        stopped_at_target=False,
        nit=GATE_REFERENCE_STEPS,
    )
    verdict = reduce_final_pairs(_gate(), _selection(), rows)
    assert verdict["verdict"] == VERDICT_NOT_PRODUCED
    assert "rung-unreachability" in verdict["reason"]
    assert f"exhausted its {GATE_REFERENCE_STEPS}-iteration cap" in verdict["reason"]

    every_leg_short = _pair_rows(
        warm_ratios=_uniform(1.3),
        wall_ratios=_uniform(1.3),
        native_stopped_at_target=False,
    )
    assert reduce_final_pairs(_gate(), _selection(), every_leg_short)[
        "reason"
    ].startswith("pair 0 warm-native exhausted its")

    below_cap = _mutate_leg(
        _pair_rows(warm_ratios=_uniform(1.3), wall_ratios=_uniform(1.3)),
        "native-wall-pair1",
        "solver",
        stopped_at_target=False,
        nit=GATE_REFERENCE_STEPS // 2,
    )
    verdict = reduce_final_pairs(_gate(), _selection(), below_cap)
    assert verdict["verdict"] == VERDICT_NOT_PRODUCED
    assert (
        f"stopped at nit {GATE_REFERENCE_STEPS // 2} of its "
        f"{GATE_REFERENCE_STEPS}-iteration cap"
    ) in verdict["reason"]


def test_final_pairs_not_produced_when_a_leg_carries_the_wrong_protocol() -> None:
    """Wall legs must not be warm-protocol; warm legs must be."""
    warm_wall = _mutate_leg(
        _pair_rows(warm_ratios=_uniform(1.3), wall_ratios=_uniform(1.3)),
        "native-wall-pair0",
        "specification",
        warm_protocol=True,
    )
    verdict = reduce_final_pairs(_gate(), _selection(), warm_wall)
    assert verdict["verdict"] == VERDICT_NOT_PRODUCED
    assert "wall-native carries warm_protocol=True" in verdict["reason"]

    cold_warm = _mutate_leg(
        _pair_rows(warm_ratios=_uniform(1.3), wall_ratios=_uniform(1.3)),
        "native-warm-pair0",
        "specification",
        warm_protocol=False,
    )
    verdict = reduce_final_pairs(_gate(), _selection(), cold_warm)
    assert verdict["verdict"] == VERDICT_NOT_PRODUCED
    assert "warm-native carries warm_protocol=False" in verdict["reason"]


def test_final_pairs_not_produced_when_a_native_leg_is_missing() -> None:
    for missing in ("native-warm-pair3", "native-wall-pair3"):
        rows = [
            row
            for row in _pair_rows(warm_ratios=_uniform(1.3), wall_ratios=_uniform(1.3))
            if row["leg_id"] != missing
        ]
        verdict = reduce_final_pairs(_gate(), _selection(), rows)
        assert verdict["verdict"] == VERDICT_NOT_PRODUCED, missing


def test_final_pairs_close_on_a_speed_miss() -> None:
    verdict = reduce_final_pairs(
        _gate(),
        _selection(),
        _pair_rows(warm_ratios=_uniform(1.05), wall_ratios=_uniform(1.3)),
    )
    assert verdict["verdict"] == VERDICT_CLOSED


def test_final_pairs_close_when_the_wall_median_misses_but_warm_passes() -> None:
    verdict = reduce_final_pairs(
        _gate(),
        _selection(),
        _pair_rows(warm_ratios=_uniform(1.4), wall_ratios=_uniform(1.05)),
    )
    assert verdict["verdict"] == VERDICT_CLOSED
    assert verdict["warm_solve_median_ratio"] >= 1.1
    assert verdict["process_wall_median_ratio"] < 1.1


def test_final_pairs_close_when_one_pair_loses_but_the_median_wins() -> None:
    warm_ratios = _uniform(1.3)
    warm_ratios[-1] = 0.99
    verdict = reduce_final_pairs(
        _gate(),
        _selection(),
        _pair_rows(warm_ratios=warm_ratios, wall_ratios=_uniform(1.3)),
    )
    assert verdict["verdict"] == VERDICT_CLOSED
    assert verdict["warm_solve_median_ratio"] >= 1.1
    assert min(verdict["warm_solve_ratios"]) < 1.0


def test_final_pairs_close_on_an_endpoint_gate_failure() -> None:
    failing = _truncated_reference_endpoint()
    failing["minimum_clearance"] = -1.0
    verdict = reduce_final_pairs(
        _gate(),
        _selection(),
        _pair_rows(
            warm_ratios=_uniform(1.5),
            wall_ratios=_uniform(1.5),
            gpu_endpoint=failing,
        ),
    )
    assert verdict["verdict"] == VERDICT_CLOSED
    assert verdict["gate_failures"]


def test_final_pairs_not_produced_when_the_lanes_disagree() -> None:
    """A GPU self-report far from the native oracle is a fork, not a speed."""
    forked = _truncated_reference_endpoint()
    forked["objective"] = 30.0
    forked["squared_flux"] = 24.0
    verdict = reduce_final_pairs(
        _gate(),
        _selection(),
        _pair_rows(
            warm_ratios=_uniform(1.5),
            wall_ratios=_uniform(1.5),
            gpu_endpoint=forked,
            oracle_endpoint=_truncated_reference_endpoint(),
        ),
    )
    assert verdict["verdict"] == VERDICT_NOT_PRODUCED
    assert "diverged from its native re-evaluation" in verdict["reason"]


def test_final_pairs_not_produced_without_a_native_re_evaluation() -> None:
    rows = [
        row
        for row in _pair_rows(warm_ratios=_uniform(1.3), wall_ratios=_uniform(1.3))
        if row.get("kind") != "native-endpoint-eval"
    ]
    verdict = reduce_final_pairs(_gate(), _selection(), rows)
    assert verdict["verdict"] == VERDICT_NOT_PRODUCED
    assert "native endpoint re-evaluation" in verdict["reason"]


def test_final_pairs_not_produced_on_a_missing_pair() -> None:
    rows = [
        row
        for row in _pair_rows(warm_ratios=_uniform(1.3), wall_ratios=_uniform(1.3))
        if "pair4" not in str(row["leg_id"])
    ]
    verdict = reduce_final_pairs(_gate(), _selection(), rows)
    assert verdict["verdict"] == VERDICT_NOT_PRODUCED


def test_final_pairs_not_produced_when_the_wall_leg_is_missing() -> None:
    rows = [
        row
        for row in _pair_rows(warm_ratios=_uniform(1.3), wall_ratios=_uniform(1.3))
        if not str(row["leg_id"]).startswith("jax-wall-pair2")
    ]
    verdict = reduce_final_pairs(_gate(), _selection(), rows)
    assert verdict["verdict"] == VERDICT_NOT_PRODUCED


def test_final_pairs_not_produced_on_a_foreign_gate_binding() -> None:
    rows = _pair_rows(warm_ratios=_uniform(1.3), wall_ratios=_uniform(1.3))
    rows[0]["gate_sha256"] = "0" * 64
    verdict = reduce_final_pairs(_gate(), _selection(), rows)
    assert verdict["verdict"] == VERDICT_NOT_PRODUCED


def test_final_pairs_not_produced_when_a_leg_ignores_the_frozen_selection() -> None:
    rows = _pair_rows(warm_ratios=_uniform(1.3), wall_ratios=_uniform(1.3))
    rows[0]["specification"]["history"] = 400
    verdict = reduce_final_pairs(_gate(), _selection(), rows)
    assert verdict["verdict"] == VERDICT_NOT_PRODUCED
    assert "frozen" in verdict["reason"]


def test_final_pairs_not_produced_on_a_zero_gpu_denominator() -> None:
    rows = _pair_rows(warm_ratios=_uniform(1.3), wall_ratios=_uniform(1.3))
    for row in rows:
        if row.get("leg_id") == "jax-warm-pair0" and "timings" in row:
            row["timings"]["warm_solve_seconds"] = [0.0, 0.0, 0.0]
    verdict = reduce_final_pairs(_gate(), _selection(), rows)
    assert verdict["verdict"] == VERDICT_NOT_PRODUCED
    assert "nonpositive" in verdict["reason"]


# ---------------------------------------------------------------------------
# Kernel canary
# ---------------------------------------------------------------------------


def _kernel_rows(
    *, gpu_median: float, repetitions: int = KERNEL_CANARY_REPETITIONS
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for repetition in range(repetitions):
        rows.append(
            {
                "leg_id": f"jax-value-grad-rep{repetition}",
                "kind": "jax-value-grad",
                "specification": {"omp_threads": GPU_HOST_OMP_THREADS},
                "timings": {"warm_value_grad_seconds": [gpu_median] * 5},
            }
        )
        for omp in NATIVE_OMP_SWEEP:
            rows.append(
                {
                    "leg_id": f"native-value-grad-omp{omp}-rep{repetition}",
                    "kind": "native-value-grad",
                    "specification": {"omp_threads": omp},
                    "timings": {"warm_value_grad_seconds": [1.0 + omp / 100.0] * 5},
                }
            )
    return rows


def test_kernel_canary_proceeds_at_or_above_the_frozen_ratio() -> None:
    verdict = reduce_kernel_canary(_kernel_rows(gpu_median=0.5))
    assert verdict["verdict"] == "PROCEED"
    assert verdict["best_native"]["omp"] == min(NATIVE_OMP_SWEEP)


def test_kernel_canary_closes_below_the_frozen_ratio() -> None:
    verdict = reduce_kernel_canary(_kernel_rows(gpu_median=1.0))
    assert verdict["verdict"] == VERDICT_CLOSED


def test_kernel_canary_not_produced_on_an_incomplete_omp_sweep() -> None:
    rows = [
        row
        for row in _kernel_rows(gpu_median=0.5)
        if f"omp{NATIVE_OMP_SWEEP[-1]}-" not in str(row["leg_id"])
    ]
    verdict = reduce_kernel_canary(rows)
    assert verdict["verdict"] == VERDICT_NOT_PRODUCED


def test_kernel_canary_not_produced_on_a_short_repetition_count() -> None:
    """A lane measured fewer times than the others moves its own median."""
    rows = [
        row
        for row in _kernel_rows(gpu_median=0.5)
        if str(row["leg_id"]) != f"native-value-grad-omp{NATIVE_OMP_SWEEP[0]}-rep2"
    ]
    verdict = reduce_kernel_canary(rows)
    assert verdict["verdict"] == VERDICT_NOT_PRODUCED
    assert "repetitions per lane" in verdict["reason"]


def test_kernel_canary_not_produced_on_a_duplicated_gpu_repetition() -> None:
    rows = _kernel_rows(gpu_median=0.5)
    rows.append(dict(rows[0], leg_id="jax-value-grad-rep0-copy"))
    verdict = reduce_kernel_canary(rows)
    assert verdict["verdict"] == VERDICT_NOT_PRODUCED


# ---------------------------------------------------------------------------
# Baseline identity
# ---------------------------------------------------------------------------


def _evaluator_row(
    kind: str, objective: float, *, gradient: list[float] | None = None
) -> dict[str, object]:
    state = {
        "objective": objective,
        "gradient": gradient if gradient is not None else [1.0, 2.0],
        "squared_flux": objective / 2.0,
        "length_penalty": objective / 4.0,
        "distance_penalty": objective / 4.0,
        "minimum_clearance": 0.1,
        "coil_lengths": [5.0],
    }
    return {
        "kind": kind,
        "fingerprints": {"input_fingerprint": "a", "configuration_fingerprint": "b"},
        "states": {
            "initial": dict(state),
            "perturbed_a": dict(state),
            "perturbed_b": dict(state),
        },
    }


def test_baseline_identity_passes_on_equal_states_and_fails_on_drift() -> None:
    rows = [_evaluator_row("native-eval", 10.0), _evaluator_row("jax-eval", 10.0)]
    assert reduce_baseline(rows)["verdict"] == "IDENTITY_OK"

    drifted = [_evaluator_row("native-eval", 10.0), _evaluator_row("jax-eval", 10.1)]
    assert reduce_baseline(drifted)["verdict"] == VERDICT_NOT_PRODUCED

    assert (
        reduce_baseline([_evaluator_row("native-eval", 10.0)])["verdict"]
        == VERDICT_NOT_PRODUCED
    )


def test_baseline_not_produced_on_a_nonfinite_gradient() -> None:
    rows = [
        _evaluator_row("native-eval", 10.0),
        _evaluator_row("jax-eval", 10.0, gradient=[float("nan"), 2.0]),
    ]
    verdict = reduce_baseline(rows)
    assert verdict["verdict"] == VERDICT_NOT_PRODUCED
    assert "nonfinite" in verdict["reason"]


def test_baseline_not_produced_on_a_duplicate_evaluator_row() -> None:
    rows = [
        _evaluator_row("native-eval", 10.0),
        _evaluator_row("native-eval", 10.0),
        _evaluator_row("jax-eval", 10.0),
    ]
    verdict = reduce_baseline(rows)
    assert verdict["verdict"] == VERDICT_NOT_PRODUCED
    assert "duplicate" in verdict["reason"]


def test_baseline_reports_many_gradient_mismatches_without_stopping_at_one() -> None:
    native = _evaluator_row("native-eval", 10.0, gradient=[1.0, 2.0, 3.0])
    jax_row = _evaluator_row("jax-eval", 10.0, gradient=[9.0, 8.0, 7.0])
    verdict = reduce_baseline([native, jax_row])
    assert verdict["verdict"] == VERDICT_NOT_PRODUCED
    assert "initial:gradient[0]" in verdict["mismatches"]
    assert "initial:gradient[2]" in verdict["mismatches"]


# ---------------------------------------------------------------------------
# Native matrix
# ---------------------------------------------------------------------------

# The disclosure lane is the only traced leg left: decreasing by one scaled
# unit per accepted step, the first value at or below the scaled target
# (20.00999) is trace[20] = 20.0, so its reported crossing is iteration 21.
_MATRIX_TRACE = [_OBJECTIVE_SCALE * (30.0 - index) for index in range(40)]
_MATRIX_BUDGET = 21
# What a crossing repetition records: it stopped well inside the 400-iteration
# cap, so ``nit`` is a measured crossing, never the cap.
_MATRIX_NIT = 21


def _solve_seconds(omp: int, history: int) -> float:
    return 5.0 if (omp, history) == (8, 20) else 9.0


def _matrix_rows(
    *,
    endpoint_objective: float = 10.0,
    non_crossing: tuple[int, int] | None = None,
    every_repetition_crosses: bool = True,
    include_disclosure: bool = True,
) -> list[dict[str, object]]:
    """Time-to-quality repetitions: SELECTION_REPETITIONS per configuration.

    ``non_crossing`` makes one repetition of one configuration exhaust its cap
    without reaching the rung; ``every_repetition_crosses=False`` does that to
    every repetition of every configuration.
    """
    gate_sha256 = _gate_sha256()
    rows: list[dict[str, object]] = []
    for omp in NATIVE_OMP_SWEEP:
        for history in NATIVE_HISTORY_SWEEP:
            endpoint = _truncated_reference_endpoint()
            endpoint["objective"] = endpoint_objective
            for repetition in range(SELECTION_REPETITIONS):
                crossed = every_repetition_crosses and not (
                    non_crossing == (omp, history) and repetition == 0
                )
                leg_id = f"native-time-to-quality-omp{omp}-h{history}-rep{repetition}"
                rows.append(
                    {
                        "leg_id": leg_id,
                        "kind": "native-solve",
                        "lane": "native",
                        "gate_sha256": gate_sha256,
                        "specification": {
                            "role": NATIVE_MATRIX_TIMED_ROLE,
                            "omp_threads": omp,
                            "history": history,
                            "max_steps": GATE_REFERENCE_STEPS,
                            "stop_at_scaled_target": _OBJECTIVE_SCALE
                            * GATE_OBJECTIVE_MARGIN
                            * _CONVERGED_OBJECTIVE,
                        },
                        "solver": {
                            "objective_scale": _OBJECTIVE_SCALE,
                            "nit": _MATRIX_NIT if crossed else GATE_REFERENCE_STEPS,
                            "nfev": 25,
                            "status": 99 if crossed else 1,
                            "stopped_at_target": crossed,
                        },
                        "endpoint": dict(endpoint),
                        "timings": {
                            "warm_solve_seconds": [_solve_seconds(omp, history)]
                        },
                    }
                )
    if include_disclosure:
        rows.append(
            {
                "leg_id": "native-shipped-default-disclosure",
                "kind": "native-solve",
                "lane": "native",
                "gate_sha256": gate_sha256,
                "specification": {
                    "role": SHIPPED_DEFAULT_DISCLOSURE_ROLE,
                    "history": 400,
                    "max_steps": GATE_REFERENCE_STEPS,
                },
                "solver": {
                    "objective_scale": _OBJECTIVE_SCALE,
                    "nit": _MATRIX_BUDGET,
                    "nfev": 25,
                    "status": 99,
                    "stopped_at_target": True,
                },
                "scaled_objective_trace": list(_MATRIX_TRACE),
                "endpoint": _truncated_reference_endpoint(),
                # Deliberately faster than every pinned lane: it must still
                # never win the selection.
                "timings": {"warm_solve_seconds": [1.0]},
            }
        )
        rows.append(
            {
                "record": "launch",
                "leg_id": "native-shipped-default-disclosure",
                "process_wall_seconds": 42.0,
            }
        )
    return rows


def test_native_matrix_selects_the_fastest_qualifying_configuration() -> None:
    verdict = reduce_native_matrix(_gate(), _matrix_rows())
    assert verdict["verdict"] == "NATIVE_SELECTED", verdict
    assert verdict["selected"] == {
        "omp": 8,
        "history": 20,
        "median_nit": _MATRIX_NIT,
        "median_fresh_process_solve_seconds": 5.0,
    }
    # No frozen iteration budget travels with the native selection any more:
    # the lane stops at the rung it measures.
    assert "budget" not in verdict["selected"]
    # The statistic must say what it is: single-solve fresh processes, whose
    # cold-start delta cannot reorder configurations at the frozen margin.
    assert "fresh-process solve time" in verdict["selection_metric"]
    assert "cannot reorder configurations" in verdict["selection_metric"]
    entry = verdict["table"]["omp8-h20"]
    assert entry["median_fresh_process_solve_seconds"] == 5.0
    assert entry["qualifying_repetitions"] == SELECTION_REPETITIONS
    assert entry["nit"] == [_MATRIX_NIT] * SELECTION_REPETITIONS


def test_native_matrix_reports_the_disclosure_lane_without_selecting_it() -> None:
    verdict = reduce_native_matrix(_gate(), _matrix_rows())
    disclosure = verdict["shipped_default_disclosure"]
    assert disclosure["leg_id"] == "native-shipped-default-disclosure"
    assert disclosure["budget"] == _MATRIX_BUDGET
    assert disclosure["solve_seconds"] == 1.0
    assert disclosure["process_wall_seconds"] == 42.0
    assert verdict["selected"]["median_fresh_process_solve_seconds"] == 5.0


def test_native_matrix_rejects_a_configuration_with_one_non_crossing_repetition() -> (
    None
):
    """All repetitions must cross: a sometimes-reaching lane is no denominator."""
    verdict = reduce_native_matrix(_gate(), _matrix_rows(non_crossing=(8, 20)))
    assert verdict["verdict"] == "NATIVE_SELECTED", verdict
    # The fastest configuration is disqualified outright, so a slower one wins.
    assert (verdict["selected"]["omp"], verdict["selected"]["history"]) != (8, 20)
    assert verdict["selected"]["median_fresh_process_solve_seconds"] == 9.0
    entry = verdict["table"]["omp8-h20"]
    assert entry["eligible"] is False
    assert entry["qualifying_repetitions"] == SELECTION_REPETITIONS - 1
    assert entry["nit"][0] == GATE_REFERENCE_STEPS
    assert "without reaching the frozen gate rung" in entry["failures"][0]
    assert f"exhausted its {GATE_REFERENCE_STEPS}-iteration cap" in entry["failures"][0]

    # A solver that terminated below its cap is named truthfully: it stopped
    # on its own criteria, it did not exhaust the budget.
    below_cap_rows = _matrix_rows(non_crossing=(8, 20))
    for row in below_cap_rows:
        if row["leg_id"] == "native-time-to-quality-omp8-h20-rep0":
            row["solver"]["nit"] = 130
    entry = reduce_native_matrix(_gate(), below_cap_rows)["table"]["omp8-h20"]
    assert (
        f"stopped at nit 130 of its {GATE_REFERENCE_STEPS}-iteration cap"
        in entry["failures"][0]
    )
    assert "without reaching the frozen gate rung" in entry["failures"][0]


def test_native_matrix_not_produced_when_no_configuration_reaches_the_rung() -> None:
    verdict = reduce_native_matrix(
        _gate(), _matrix_rows(every_repetition_crosses=False)
    )
    assert verdict["verdict"] == VERDICT_NOT_PRODUCED
    assert "frozen gate rung" in verdict["reason"]
    # The systematic unreachability must be visible per configuration.
    for entry in verdict["table"].values():
        assert entry["qualifying_repetitions"] == 0
        assert entry["nit"] == [GATE_REFERENCE_STEPS] * SELECTION_REPETITIONS


def test_native_matrix_not_produced_when_every_stop_endpoint_fails_the_gate() -> None:
    """Crossing the rung is not enough: the stop endpoint is gated too."""
    verdict = reduce_native_matrix(_gate(), _matrix_rows(endpoint_objective=10.2))
    assert verdict["verdict"] == VERDICT_NOT_PRODUCED
    assert "frozen gate rung" in verdict["reason"]
    entry = verdict["table"]["omp8-h20"]
    assert entry["qualifying_repetitions"] == 0
    assert "misses target" in entry["failures"][0]


def test_native_matrix_not_produced_on_a_foreign_gate_binding() -> None:
    rows = _matrix_rows()
    rows[0]["gate_sha256"] = "0" * 64
    verdict = reduce_native_matrix(_gate(), rows)
    assert verdict["verdict"] == VERDICT_NOT_PRODUCED
    assert "different gate" in verdict["reason"]


def test_native_matrix_not_produced_on_a_retired_replay_role() -> None:
    """The calibrate/replay protocol is gone; its rows are not accepted."""
    rows = _matrix_rows()
    rows[0]["specification"]["role"] = "replay"
    verdict = reduce_native_matrix(_gate(), rows)
    assert verdict["verdict"] == VERDICT_NOT_PRODUCED
    assert "unexpected role replay" in verdict["reason"]


def test_native_matrix_not_produced_on_a_miscounted_repetition_set() -> None:
    for rows in (
        _matrix_rows()
        + [
            dict(
                next(
                    row
                    for row in _matrix_rows()
                    if str(row["leg_id"]).startswith("native-time-to-quality-omp8-h20")
                ),
                leg_id="native-time-to-quality-omp8-h20-rep9",
            )
        ],
        [
            row
            for row in _matrix_rows()
            if row["leg_id"] != "native-time-to-quality-omp8-h20-rep0"
        ],
    ):
        verdict = reduce_native_matrix(_gate(), rows)
        assert verdict["verdict"] == VERDICT_NOT_PRODUCED
        assert f"exactly {SELECTION_REPETITIONS} time-to-quality" in verdict["reason"]


# ---------------------------------------------------------------------------
# JAX sweep
# ---------------------------------------------------------------------------


def _sweep_rows(
    history: int, budget: int, endpoint: dict[str, object]
) -> list[dict[str, object]]:
    leg_id = f"jax-sweep-h{history}-b{budget}"
    gate_sha256 = _gate_sha256()
    return [
        {
            "leg_id": leg_id,
            "lane": "jax",
            "kind": "jax-solve",
            "gate_sha256": gate_sha256,
            "specification": {
                "history": history,
                "max_steps": budget,
                "role": "sweep",
                "omp_threads": GPU_HOST_OMP_THREADS,
            },
            "endpoint": endpoint,
            "timings": {"warm_solve_seconds": [1.0, 1.1, 1.2]},
        },
        {
            "leg_id": f"native-endpoint-{leg_id}",
            "lane": "native",
            "kind": "native-endpoint-eval",
            "gate_sha256": gate_sha256,
            "subject_leg_id": leg_id,
            "specification": {"role": "endpoint-eval", "subject_leg_id": leg_id},
            "endpoint": dict(endpoint),
        },
    ]


def test_jax_sweep_closes_when_no_history_reaches_the_contract() -> None:
    failing = _truncated_reference_endpoint()
    failing["objective"] = 99.0
    rows = [
        row
        for history in JAX_HISTORY_SWEEP
        for budget in GPU_BUDGET_SWEEP
        for row in _sweep_rows(history, budget, dict(failing))
    ]
    verdict = reduce_jax_sweep(_gate(), rows)
    assert verdict["verdict"] == VERDICT_CLOSED


def test_jax_sweep_selects_the_fastest_qualifying_history() -> None:
    failing = _truncated_reference_endpoint()
    failing["objective"] = 99.0
    rows: list[dict[str, object]] = []
    for history in JAX_HISTORY_SWEEP:
        rows.extend(_sweep_rows(history, GPU_BUDGET_SWEEP[0], dict(failing)))
        qualifying = _sweep_rows(
            history, GPU_BUDGET_SWEEP[1], _truncated_reference_endpoint()
        )
        qualifying[0]["timings"]["warm_solve_seconds"] = [float(history)] * (
            SELECTION_REPETITIONS
        )
        rows.extend(qualifying)
    verdict = reduce_jax_sweep(_gate(), rows)
    assert verdict["verdict"] == "JAX_SELECTED"
    assert verdict["selected"]["history"] == min(JAX_HISTORY_SWEEP)
    assert verdict["selected"]["budget"] == GPU_BUDGET_SWEEP[1]


def test_jax_sweep_not_produced_on_a_short_warm_sample_count() -> None:
    failing = _truncated_reference_endpoint()
    failing["objective"] = 99.0
    rows: list[dict[str, object]] = []
    for history in JAX_HISTORY_SWEEP:
        rows.extend(_sweep_rows(history, GPU_BUDGET_SWEEP[0], dict(failing)))
        qualifying = _sweep_rows(
            history, GPU_BUDGET_SWEEP[1], _truncated_reference_endpoint()
        )
        qualifying[0]["timings"]["warm_solve_seconds"] = [1.0]
        rows.extend(qualifying)
    verdict = reduce_jax_sweep(_gate(), rows)
    assert verdict["verdict"] == VERDICT_NOT_PRODUCED
    assert "warm samples" in verdict["reason"]


def test_jax_sweep_not_produced_without_the_native_oracle_row() -> None:
    rows = [
        row
        for history in JAX_HISTORY_SWEEP
        for budget in GPU_BUDGET_SWEEP
        for row in _sweep_rows(history, budget, _truncated_reference_endpoint())
        if row["kind"] != "native-endpoint-eval"
    ]
    verdict = reduce_jax_sweep(_gate(), rows)
    assert verdict["verdict"] == VERDICT_NOT_PRODUCED


# ---------------------------------------------------------------------------
# Validator entry points and small pure helpers
# ---------------------------------------------------------------------------


def test_validate_run_not_produced_without_manifest(tmp_path: Path) -> None:
    verdict = validate_run(tmp_path)
    assert verdict["verdict"] == VERDICT_NOT_PRODUCED


def test_threading_conformance_rejects_a_mislabeled_leg() -> None:
    """A child that observed different threading than its spec declared."""
    row = {
        "leg_id": "native-time-to-quality-omp8-h20-rep0",
        "specification": {"omp_threads": 8, "cpu_affinity": [0, 1, 2, 3]},
        "identity": {
            "threading": {
                "environment": {"OMP_NUM_THREADS": "16"},
                "cpu_affinity": [0, 1, 2, 3],
            }
        },
    }
    failure = _threading_conformance_failure([row])
    assert failure is not None and "OMP_NUM_THREADS=16" in failure

    row["identity"]["threading"]["environment"]["OMP_NUM_THREADS"] = "8"
    assert _threading_conformance_failure([row]) is None

    row["identity"]["threading"]["cpu_affinity"] = [0, 1, 2, 5]
    failure = _threading_conformance_failure([row])
    assert failure is not None and "cpu_affinity" in failure


def test_fp64_conformance_rejects_a_leg_whose_jax_ran_float32() -> None:
    """Only the child's observed x64 state counts, never the declared pin.

    The 2026-08-17 taint: a native leg's transitive JAX ran float32 and
    published gradients that were not the declared physics. A row missing
    the observation entirely (the pre-fix row shape) must also fail closed.
    """
    row = {
        "leg_id": "native-time-to-quality-omp2-h10-rep0",
        "identity": {"jax_imported": True, "jax_enable_x64": True},
    }
    assert _fp64_conformance_failure([row]) is None

    row["identity"]["jax_enable_x64"] = False
    failure = _fp64_conformance_failure([row])
    assert failure is not None and "jax_enable_x64=False" in failure

    del row["identity"]["jax_enable_x64"]
    failure = _fp64_conformance_failure([row])
    assert failure is not None and "jax_enable_x64=None" in failure

    row["identity"] = {"jax_imported": False}
    assert _fp64_conformance_failure([row]) is None


def test_validate_run_fails_closed_on_an_fp32_leg(tmp_path: Path) -> None:
    """The wiring, not just the clause: validation itself must consult it."""
    row = {
        "leg_id": "native-eval",
        "kind": "native-eval",
        "fingerprints": {"input_fingerprint": "a"},
        "identity": {"jax_imported": True, "jax_enable_x64": False},
    }
    row_sha256 = _write_json_exclusive(tmp_path / "rows" / "native-eval.json", row)
    _write_json_exclusive(
        tmp_path / "manifest.json",
        {
            "schema": "stage-two-finitebuild-native-gpu-manifest-v1",
            "phase": "baseline",
            "rows": {"rows/native-eval.json": row_sha256},
        },
    )
    verdict = validate_run(tmp_path)
    assert verdict["verdict"] == VERDICT_NOT_PRODUCED
    assert "jax_enable_x64=False" in str(verdict["reason"])


def test_validate_run_not_produced_on_a_row_hash_mismatch(tmp_path: Path) -> None:
    row_path = tmp_path / "rows" / "leg.json"
    _write_json_exclusive(row_path, {"leg_id": "leg"})
    _write_json_exclusive(
        tmp_path / "manifest.json",
        {
            "schema": "stage-two-finitebuild-native-gpu-manifest-v1",
            "phase": "baseline",
            "rows": {"rows/leg.json": "0" * 64},
        },
    )
    verdict = validate_run(tmp_path)
    assert verdict["verdict"] == VERDICT_NOT_PRODUCED


def test_validate_run_not_produced_on_an_absent_expected_row(tmp_path: Path) -> None:
    _write_json_exclusive(
        tmp_path / "manifest.json",
        {
            "schema": "stage-two-finitebuild-native-gpu-manifest-v1",
            "phase": "kernel-canary",
            "rows": {"rows/missing.json": "absent"},
        },
    )
    verdict = validate_run(tmp_path)
    assert verdict["verdict"] == VERDICT_NOT_PRODUCED


def test_validate_run_not_produced_on_a_row_without_fingerprints(
    tmp_path: Path,
) -> None:
    row_path = tmp_path / "rows" / "leg.json"
    row_sha256 = _write_json_exclusive(row_path, {"leg_id": "leg", "kind": "jax-eval"})
    _write_json_exclusive(
        tmp_path / "manifest.json",
        {
            "schema": "stage-two-finitebuild-native-gpu-manifest-v1",
            "phase": "baseline",
            "rows": {"rows/leg.json": row_sha256},
        },
    )
    verdict = validate_run(tmp_path)
    assert verdict["verdict"] == VERDICT_NOT_PRODUCED
    assert "fingerprints" in verdict["reason"]


def test_canonical_json_publishes_nonfinite_floats_by_name() -> None:
    payload = _canonical_json_bytes({"objective": float("nan"), "wall": float("inf")})
    assert b'"objective":"NaN"' in payload
    assert b'"wall":"Infinity"' in payload


def test_first_qualifying_iteration_is_one_indexed_and_none_when_missed() -> None:
    assert first_qualifying_iteration([5.0, 3.0, 1.0], 3.0) == 2
    assert first_qualifying_iteration([5.0, 4.0], 3.0) is None


def test_hlo_diff_classifies_dce_null_and_changed() -> None:
    census = hlo_operation_census(
        "  %fusion.1 = f64[24,75]{1,0} fusion(%p.0), kind=kLoop\n"
        "  %add.2 = f64[3]{0} add(%a, %b)\n"
        "  %fusion.9 = (f64[2]{0}, f64[3]{0}) fusion(%c), kind=kInput\n"
    )
    assert census == {"fusion": 2, "add": 1}

    fingerprints = {"input_fingerprint": "a", "configuration_fingerprint": "b"}
    before = {
        "fingerprints": dict(fingerprints),
        "census": census,
        "cost_analysis": {"flops": 100.0},
        "artifacts": {"stablehlo_sha256": "a" * 64, "optimized_hlo_sha256": "c" * 64},
    }
    same = {
        "fingerprints": dict(fingerprints),
        "census": dict(census),
        "cost_analysis": {"flops": 100.0},
        "artifacts": {"stablehlo_sha256": "b" * 64, "optimized_hlo_sha256": "d" * 64},
    }
    diff = reduce_hlo_diff(before, same)
    assert diff["classification"] == "DCE_NULL"
    assert diff["stablehlo_changed"] is True
    assert diff["optimized_hlo_sha256"] == {"before": "c" * 64, "after": "d" * 64}

    smaller = {
        "fingerprints": dict(fingerprints),
        "census": {"fusion": 1, "add": 1},
        "cost_analysis": {"flops": 90.0},
        "artifacts": {"stablehlo_sha256": "b" * 64, "optimized_hlo_sha256": "d" * 64},
    }
    assert reduce_hlo_diff(before, smaller)["classification"] == "CHANGED"


def test_hlo_diff_not_produced_on_an_empty_capture_or_forked_inputs() -> None:
    fingerprints = {"input_fingerprint": "a", "configuration_fingerprint": "b"}
    complete = {
        "fingerprints": dict(fingerprints),
        "census": {"fusion": 2},
        "cost_analysis": {"flops": 100.0},
        "artifacts": {"stablehlo_sha256": "a" * 64, "optimized_hlo_sha256": "c" * 64},
    }
    empty = dict(complete, census={})
    assert reduce_hlo_diff(complete, empty)["classification"] == VERDICT_NOT_PRODUCED

    forked = dict(complete, fingerprints={"input_fingerprint": "z"})
    assert reduce_hlo_diff(complete, forked)["classification"] == VERDICT_NOT_PRODUCED


def test_leg_environment_pins_fp64_jax_in_both_lanes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The scrub removes inherited JAX settings, so each lane must re-pin the
    load-bearing ones.  The native lane imports JAX transitively through
    ``simsopt.geo``; without the x64 pin those pieces evaluate in fp32 and the
    native gradient forks from the GPU lane's self-report (measured
    2026-08-17: 2.6e-6 on a 2.9e-5 component at a sweep endpoint, with FD
    arbitration convicting the native value).
    """
    monkeypatch.setenv("JAX_ENABLE_X64", "0")
    monkeypatch.setenv("JAX_PLATFORMS", "cuda")
    monkeypatch.setenv("OMP_NUM_THREADS", "77")
    # Hostile inherited variables that no lane re-pins for a native leg:
    # they must vanish through the scrub, not survive into the child.
    monkeypatch.setenv("JAX_COMPILATION_CACHE_DIR", "/inherited/cache")
    monkeypatch.setenv("XLA_FLAGS", "--xla_force_host_platform_device_count=7")
    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "inherited")

    native = _leg_environment("native", omp_threads=4, cache_root=None)
    assert native["JAX_PLATFORMS"] == "cpu"
    assert native["JAX_ENABLE_X64"] == "1"
    assert native["OMP_NUM_THREADS"] == "4"
    assert "JAX_COMPILATION_CACHE_DIR" not in native
    assert "XLA_FLAGS" not in native
    assert "SIMSOPT_BACKEND_MODE" not in native

    jax_env = _leg_environment("jax", omp_threads=2, cache_root=tmp_path)
    assert jax_env["JAX_PLATFORMS"] == "cuda"
    assert jax_env["JAX_ENABLE_X64"] == "1"
    assert jax_env["JAX_COMPILATION_CACHE_DIR"] == str(tmp_path)
    assert jax_env["XLA_FLAGS"] == "--xla_gpu_exclude_nondeterministic_ops=true"
    assert jax_env["SIMSOPT_BACKEND_MODE"] == "jax_gpu_fast"


def test_gate_source_conformance_binds_physics_and_discloses_harness_drift() -> None:
    """Physics pins are fail-closed; harness/plan pins are disclosed drift.

    The dirty-file branch is exercised directly; the commit branch is
    exercised end-to-end by validating the real gate-consuming runs.
    """
    pins = dict(_PHYSICS_SOURCE_PINS) | dict(_DISCLOSED_SOURCE_PINS)
    shas = {key: format(index, "x") * 64 for index, key in enumerate(pins, 1)}
    shas = {key: value[:64] for key, value in shas.items()}
    identity_git = {
        "commit": "0" * 40,
        "changed_file_sha256": {pins[key]: shas[key] for key in pins},
    }
    row = {"leg_id": "leg", "identity": {"git": identity_git}}
    gate = {"source_fingerprints": dict(shas)}

    failure, drift = _gate_source_conformance(gate, [row])
    assert failure is None
    assert all(entry["identical"] for entry in drift.values())

    forked_physics = {
        "source_fingerprints": dict(shas, objective_module_sha256="f" * 64)
    }
    failure, drift = _gate_source_conformance(forked_physics, [row])
    assert failure is not None and "different physics" in failure
    assert drift == {}

    amended_harness = {"source_fingerprints": dict(shas, benchmark_sha256="e" * 64)}
    failure, drift = _gate_source_conformance(amended_harness, [row])
    assert failure is None
    assert drift["benchmark_sha256"]["identical"] is False
    assert drift["plan_sha256"]["identical"] is True


def test_validate_run_fails_closed_on_a_physics_forked_gate_consumer(
    tmp_path: Path,
) -> None:
    """The wiring: gate-consuming validation must run the conformance check."""
    pins = dict(_PHYSICS_SOURCE_PINS) | dict(_DISCLOSED_SOURCE_PINS)
    run_shas = {key: "a" * 64 for key in pins}
    gate = {"source_fingerprints": dict(run_shas, objective_module_sha256="f" * 64)}
    _write_json_exclusive(tmp_path / "gate" / "quality_contract.json", gate)
    row = {
        "leg_id": "jax-sweep-h10-b40",
        "kind": "jax-solve",
        "fingerprints": {"input_fingerprint": "a"},
        "identity": {
            "git": {
                "commit": "0" * 40,
                "changed_file_sha256": {pins[key]: run_shas[key] for key in pins},
            }
        },
    }
    row_sha256 = _write_json_exclusive(
        tmp_path / "rows" / "jax-sweep-h10-b40.json", row
    )
    _write_json_exclusive(
        tmp_path / "manifest.json",
        {
            "schema": "stage-two-finitebuild-native-gpu-manifest-v1",
            "phase": "jax-sweep",
            "rows": {"rows/jax-sweep-h10-b40.json": row_sha256},
        },
    )
    verdict = validate_run(tmp_path)
    assert verdict["verdict"] == VERDICT_NOT_PRODUCED
    assert "different physics" in str(verdict["reason"])


# ---------------------------------------------------------------------------
# Successor bisection sweep
# ---------------------------------------------------------------------------


def _bisect_probe(
    history: int,
    budget: int,
    *,
    self_objective: float,
    oracle_objective: float,
    solution: list[float] | None = None,
) -> list[dict[str, object]]:
    """One untimed probe leg plus its native re-evaluation."""
    leg_id = f"jax-bisect-h{history}-b{budget}"
    endpoint = _truncated_reference_endpoint()
    endpoint["objective"] = self_objective
    if solution is not None:
        endpoint["solution"] = solution
    oracle_endpoint = _truncated_reference_endpoint()
    oracle_endpoint["objective"] = oracle_objective
    return [
        {
            "leg_id": leg_id,
            "lane": "jax",
            "kind": "jax-solve",
            "gate_sha256": _gate_sha256(),
            "specification": {
                "role": JAX_BISECT_ROLE,
                "history": history,
                "max_steps": budget,
                "warm_repetitions": 0,
            },
            "endpoint": endpoint,
            "timings": {"warm_solve_seconds": []},
            "solver": {"nit": budget, "nfev": budget + 4, "status": 1},
        },
        {
            "leg_id": f"native-endpoint-{leg_id}",
            "lane": "native",
            "kind": "native-endpoint-eval",
            "gate_sha256": _gate_sha256(),
            "subject_leg_id": leg_id,
            "endpoint": oracle_endpoint,
        },
    ]


def _bisect_crossing(
    history: int,
    budget: int,
    *,
    warm: list[float],
    solution: list[float] | None = None,
) -> list[dict[str, object]]:
    """The timed crossing leg plus its native re-evaluation."""
    leg_id = f"jax-sweep-h{history}-b{budget}"
    endpoint = _truncated_reference_endpoint()
    if solution is not None:
        endpoint["solution"] = solution
    return [
        {
            "leg_id": leg_id,
            "lane": "jax",
            "kind": "jax-solve",
            "gate_sha256": _gate_sha256(),
            "specification": {
                "role": JAX_CROSSING_ROLE,
                "history": history,
                "max_steps": budget,
                "warm_repetitions": SELECTION_REPETITIONS,
            },
            "endpoint": endpoint,
            "timings": {"warm_solve_seconds": warm},
            "solver": {"nit": budget, "nfev": budget + 4, "status": 1},
        },
        {
            "leg_id": f"native-endpoint-{leg_id}",
            "lane": "native",
            "kind": "native-endpoint-eval",
            "gate_sha256": _gate_sha256(),
            "subject_leg_id": leg_id,
            "endpoint": _truncated_reference_endpoint(),
        },
    ]


def _bisect_rows(**h10_overrides: object) -> list[dict[str, object]]:
    """Minimal legal successor sweep: h10 crosses at the cap, h20/h40 never.

    The reducer requires the cap probe, the crossing probe, and the
    minimality probe at ``k*-1``; it does not require the full bisection
    path, so the fixture carries exactly those.
    """
    cap = 800
    rows: list[dict[str, object]] = []
    rows += _bisect_probe(10, cap - 1, self_objective=10.2, oracle_objective=10.2)
    rows += _bisect_probe(10, cap, self_objective=10.0, oracle_objective=10.0)
    crossing_kwargs = {"warm": [1.0, 1.1, 1.2]}
    crossing_kwargs.update(h10_overrides)
    rows += _bisect_crossing(10, cap, **crossing_kwargs)
    for history in (20, 40):
        rows += _bisect_probe(history, cap, self_objective=10.2, oracle_objective=10.2)
    return rows


def test_bisection_sweep_selects_the_crossing_history() -> None:
    verdict = reduce_jax_sweep(_gate(), _bisect_rows())
    assert verdict["verdict"] == "JAX_SELECTED", verdict
    selected = verdict["selected"]
    assert selected["history"] == 10
    assert selected["budget"] == 800
    assert selected["median_solve_seconds"] == 1.1
    assert selected["crossing_solution"] == [0.1, 0.2, 0.3]
    assert len(selected["crossing_solution_sha256"]) == 64
    assert verdict["table"]["h20"]["reached_rung"] is False
    assert verdict["table"]["h40"]["reached_rung"] is False


def test_bisection_sweep_closes_when_no_history_crosses() -> None:
    rows: list[dict[str, object]] = []
    for history in JAX_HISTORY_SWEEP:
        rows += _bisect_probe(history, 800, self_objective=10.2, oracle_objective=10.2)
    verdict = reduce_jax_sweep(_gate(), rows)
    assert verdict["verdict"] == VERDICT_CLOSED
    assert "no JAX history reached" in verdict["reason"]


def test_bisection_sweep_fails_closed_on_a_monotonicity_violation() -> None:
    """A self-reported objective that rises with budget breaks the instrument."""
    rows = _bisect_rows()
    for row in rows:
        if row.get("leg_id") == "jax-bisect-h10-b799":
            row["endpoint"]["objective"] = 9.9
    verdict = reduce_jax_sweep(_gate(), rows)
    assert verdict["verdict"] == VERDICT_NOT_PRODUCED
    assert "monotonicity" in verdict["reason"]


def test_bisection_sweep_requires_the_minimality_probe() -> None:
    rows = [
        row for row in _bisect_rows() if "h10-b799" not in str(row.get("leg_id", ""))
    ]
    verdict = reduce_jax_sweep(_gate(), rows)
    assert verdict["verdict"] == VERDICT_NOT_PRODUCED
    assert "unproved minimal" in verdict["reason"]


def test_bisection_sweep_rejects_a_crossing_below_the_published_budget() -> None:
    rows = _bisect_rows()
    for row in rows:
        if row.get("leg_id") == "native-endpoint-jax-bisect-h10-b799":
            row["endpoint"]["objective"] = 10.0
    verdict = reduce_jax_sweep(_gate(), rows)
    assert verdict["verdict"] == VERDICT_NOT_PRODUCED
    assert "below its published crossing budget" in verdict["reason"]


def test_bisection_sweep_rejects_a_probe_crossing_fork() -> None:
    """The timed crossing leg must be bitwise the probe at the same budget."""
    verdict = reduce_jax_sweep(
        _gate(), _bisect_rows(solution=[0.1, 0.2, 0.30000000001])
    )
    assert verdict["verdict"] == VERDICT_NOT_PRODUCED
    assert "determinism is broken" in verdict["reason"]


def test_final_pairs_verify_every_gpu_endpoint_against_the_frozen_solution() -> None:
    selection = _selection()
    selection["jax"] = dict(selection["jax"], crossing_solution=[0.1, 0.2, 0.3])
    rows = _pair_rows(warm_ratios=_uniform(1.3), wall_ratios=_uniform(1.3))
    verdict = reduce_final_pairs(_gate(), selection, rows)
    assert verdict["verdict"] == VERDICT_WIN, verdict

    forked = _pair_rows(warm_ratios=_uniform(1.3), wall_ratios=_uniform(1.3))
    for row in forked:
        if row.get("leg_id") == "jax-warm-pair3" and "endpoint" in row:
            row["endpoint"] = dict(row["endpoint"], solution=[0.1, 0.2, 0.31])
    verdict = reduce_final_pairs(_gate(), selection, forked)
    assert verdict["verdict"] == VERDICT_NOT_PRODUCED
    assert "deviates" in verdict["reason"] and "pair 3" in verdict["reason"]


def test_bisection_sweep_requires_the_cap_probe() -> None:
    """The final close at budget parity is evidenced by the cap probe."""
    rows = [
        row for row in _bisect_rows() if "h20-b800" not in str(row.get("leg_id", ""))
    ]
    rows += _bisect_probe(20, 400, self_objective=10.2, oracle_objective=10.2)
    verdict = reduce_jax_sweep(_gate(), rows)
    assert verdict["verdict"] == VERDICT_NOT_PRODUCED
    assert "no cap probe" in verdict["reason"]
