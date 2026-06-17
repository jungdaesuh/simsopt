import json
from types import SimpleNamespace

from geo._frontier_test_helpers import ensure_examples_import_path

ensure_examples_import_path()

from banana_opt.design_only_fields import build_design_only_results_fields
from banana_opt.topology.kam_birkhoff import (
    DEFAULT_WBA_MIN_RETURNS,
    KAM_CLASS_INVALID_POLAR_REFERENCE,
    KAM_CLASS_INVARIANT_TORUS,
    KAM_CLASS_ISLAND_CHAIN,
)
from banana_opt.topology_fidelity_ladder import WBA_SEED_CLASSIFICATIONS_KEY
import run_topology_fidelity_ladder as ladder


def _classified_result(classifications):
    return {
        "survival_fraction": 1.0,
        "confinement_score": 1.0,
        "broken": False,
        "evaluation_state": "evaluated",
        WBA_SEED_CLASSIFICATIONS_KEY: classifications,
    }


def _seed(seed_index, classification, return_count=DEFAULT_WBA_MIN_RETURNS):
    return {
        "seed_index": seed_index,
        "classification": classification,
        "return_count": return_count,
    }


def test_build_tier_case_record_enables_island_and_classifiability_gates():
    record = ladder.build_tier_case_record(
        _classified_result(
            [
                _seed(0, KAM_CLASS_INVARIANT_TORUS),
                _seed(1, KAM_CLASS_ISLAND_CHAIN),
            ]
        ),
        survival_threshold=1.0,
    )

    assert record["passed"] is False
    verdict = record["topology_verdict"]
    assert verdict["survival_gate_passed"] is True
    assert verdict["island_gate_enabled"] is True
    assert verdict["island_gate_passed"] is False
    assert verdict["island_chain_count"] == 1
    assert verdict["island_chain_seed_indices"] == [1]
    assert verdict["classifiability_floor_enabled"] is True


def test_build_tier_case_record_rejects_unclassifiable_survivors():
    record = ladder.build_tier_case_record(
        _classified_result(
            [
                _seed(0, KAM_CLASS_INVARIANT_TORUS, return_count=1),
                _seed(1, KAM_CLASS_INVARIANT_TORUS, return_count=1),
            ]
        ),
        survival_threshold=1.0,
    )

    assert record["passed"] is False
    verdict = record["topology_verdict"]
    assert verdict["survival_gate_passed"] is True
    assert verdict["island_gate_passed"] is True
    assert verdict["classifiability_floor_passed"] is False
    assert verdict["classifiable_fraction"] == 0.0


def test_build_tier_case_record_rejects_invalid_reference_survivor():
    record = ladder.build_tier_case_record(
        _classified_result(
            [
                _seed(0, KAM_CLASS_INVALID_POLAR_REFERENCE),
            ]
        ),
        survival_threshold=1.0,
    )

    assert record["passed"] is False
    verdict = record["topology_verdict"]
    assert verdict["survival_gate_passed"] is True
    assert verdict["island_gate_passed"] is True
    assert verdict["classifiability_floor_passed"] is False
    assert verdict["classifiable_fraction"] == 0.0


def test_build_tier_case_record_passes_clean_classified_result():
    record = ladder.build_tier_case_record(
        _classified_result(
            [
                _seed(0, KAM_CLASS_INVARIANT_TORUS),
                _seed(1, KAM_CLASS_INVARIANT_TORUS),
            ]
        ),
        survival_threshold=1.0,
    )

    assert record["passed"] is True
    verdict = record["topology_verdict"]
    assert verdict["survival_gate_passed"] is True
    assert verdict["island_gate_passed"] is True
    assert verdict["classifiability_floor_passed"] is True


def test_topology_fidelity_ladder_rejects_design_only_results_sidecar(
    tmp_path,
    monkeypatch,
):
    output_dir = tmp_path / "proxy_run"
    output_dir.mkdir()
    (output_dir / "biot_savart_opt.json").write_text("{}", encoding="utf-8")
    (output_dir / "surf_opt.json").write_text("{}", encoding="utf-8")
    (output_dir / "results.json").write_text(
        json.dumps(
            build_design_only_results_fields(
                reason="finite_current_proxy_line_current: wataru_proxy_field"
            )
        ),
        encoding="utf-8",
    )

    def fake_load(path):
        return SimpleNamespace(path_name=path.name)

    monkeypatch.setattr(ladder, "load", fake_load)

    record = ladder.evaluate_case(output_dir)

    assert record["field_label"] == "opt"
    for tier_name in ladder.DEFAULT_TOPOLOGY_TIER_SPECS:
        tier_record = record[tier_name]
        assert tier_record["broken"] is True
        assert tier_record["passed"] is False
        assert tier_record["evaluation_error_type"] == "DesignOnlyTopologyFieldError"
        assert tier_record["topology_verdict"]["island_gate_enabled"] is True
        assert (
            tier_record["topology_verdict"]["classifiability_floor_enabled"] is True
        )
