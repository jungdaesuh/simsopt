import json

from geo._frontier_test_helpers import EXAMPLE_ROOT, load_module


FRONTIER_KAM_CALIBRATION_SCRIPT = EXAMPLE_ROOT / "frontier_kam_calibration.py"
DEFAULT_INVARIANT_TORUS_FRACTION = object()


def load_calibration_module():
    return load_module(FRONTIER_KAM_CALIBRATION_SCRIPT, "frontier_kam_calibration")


def test_loading_calibration_module_preserves_simsopt_class_identity():
    from simsopt._core.optimizable import Optimizable as optimizable_before
    from simsopt.field.biotsavart import BiotSavart as biot_savart_before

    load_calibration_module()

    from simsopt._core.optimizable import Optimizable as optimizable_after
    from simsopt.field.biotsavart import BiotSavart as biot_savart_after

    assert optimizable_after is optimizable_before
    assert biot_savart_after is biot_savart_before
    assert issubclass(biot_savart_after, optimizable_after)


def make_row(
    module,
    label,
    *,
    kam_fraction,
    invariant_torus_fraction=DEFAULT_INVARIANT_TORUS_FRACTION,
    hardware_ok=True,
    topology_broken=False,
):
    if invariant_torus_fraction is DEFAULT_INVARIANT_TORUS_FRACTION:
        invariant_torus_fraction = kam_fraction
    return module.CalibrationRow(
        donor_label=label,
        run_dir=f"/tmp/{label}",
        biot_savart_path=f"/tmp/{label}/biot_savart_opt.json",
        surface_path=f"/tmp/{label}/surf_opt.json",
        results_path=f"/tmp/{label}/results.json",
        hardware_constraints_ok=hardware_ok,
        alm_hard_constraints_feasible=None,
        objective_j=None,
        nonqs_ratio=None,
        boozer_residual=None,
        final_iota=None,
        final_volume=None,
        topology_broken=topology_broken,
        survival_fraction=1.0,
        survived_lines=12,
        nfieldlines=12,
        tmax=50.0,
        nphis=4,
        invariant_torus_fraction=invariant_torus_fraction,
        kam_fraction=kam_fraction,
        kam_median_width=0.01,
        kam_width_ratio=0.25,
        cross_section_span=0.1,
        seed_contract_json='{"nfieldlines":12}',
        stop_reason_counts_json="{}",
        field_model_json='{"selected_mode":"native"}',
    )


def test_calibration_summary_recommends_floor_from_hw_clean_donors():
    module = load_calibration_module()
    rows = [
        make_row(module, "low", kam_fraction=0.25),
        make_row(module, "high", kam_fraction=0.75),
        make_row(module, "hw_failed", kam_fraction=0.0, hardware_ok=False),
        make_row(module, "broken", kam_fraction=1.0, topology_broken=True),
    ]

    summary = module.calibration_summary(
        rows,
        frontier_invariant_torus_min=0.30,
        selection_margin=1.0 / 12.0,
    )

    assert summary["hw_clean_donor_count"] == 2
    assert summary["hw_clean_evaluated_donor_count"] == 2
    assert summary["invariant_torus_fraction_min"] == 0.25
    assert summary["kam_fraction_min"] == 0.25
    assert summary["configured_threshold_rejecting_hw_clean_donors"] == ["low"]
    assert summary["configured_threshold_not_evaluated_hw_clean_donors"] == []
    assert summary["configured_threshold_accepts_all_hw_clean_donors"] is False
    assert summary["recommended_frontier_invariant_torus_min"] == 0.25 - 1.0 / 12.0
    assert summary["recommended_frontier_kam_min"] == 0.25 - 1.0 / 12.0


def test_calibration_summary_uses_invariant_torus_fraction_over_legacy_kam():
    module = load_calibration_module()
    rows = [
        make_row(
            module,
            "low_invariant_high_legacy",
            invariant_torus_fraction=0.20,
            kam_fraction=0.80,
        ),
        make_row(
            module,
            "high_invariant_low_legacy",
            invariant_torus_fraction=0.60,
            kam_fraction=0.10,
        ),
    ]

    summary = module.calibration_summary(
        rows,
        frontier_invariant_torus_min=0.30,
        selection_margin=0.05,
    )

    assert summary["invariant_torus_fraction_min"] == 0.20
    assert summary["kam_fraction_min"] == 0.20
    assert summary["configured_threshold_rejecting_hw_clean_donors"] == [
        "low_invariant_high_legacy"
    ]
    assert summary["recommended_frontier_invariant_torus_min"] == 0.20 - 0.05
    assert summary["recommended_frontier_kam_min"] == 0.20 - 0.05


def test_calibration_summary_allows_zero_floor_when_known_good_donor_has_zero_kam():
    module = load_calibration_module()
    rows = [
        make_row(module, "current_champion", kam_fraction=0.0),
        make_row(module, "other", kam_fraction=1.0 / 12.0),
    ]

    summary = module.calibration_summary(
        rows,
        frontier_invariant_torus_min=0.30,
        selection_margin=1.0 / 12.0,
    )

    assert summary["configured_threshold_rejecting_hw_clean_donors"] == [
        "current_champion",
        "other",
    ]
    assert summary["recommended_frontier_invariant_torus_min"] == 0.0
    assert summary["recommended_frontier_kam_min"] == 0.0


def test_calibration_summary_does_not_recommend_floor_from_not_evaluated_wba():
    module = load_calibration_module()
    rows = [
        make_row(
            module,
            "short_trace",
            kam_fraction=None,
            invariant_torus_fraction=None,
        ),
    ]

    summary = module.calibration_summary(
        rows,
        frontier_invariant_torus_min=0.0,
        selection_margin=1.0 / 12.0,
    )

    assert summary["hw_clean_donor_count"] == 1
    assert summary["hw_clean_evaluated_donor_count"] == 0
    assert summary["invariant_torus_fraction_min"] is None
    assert summary["configured_threshold_rejecting_hw_clean_donors"] == []
    assert summary["configured_threshold_not_evaluated_hw_clean_donors"] == [
        "short_trace"
    ]
    assert summary["configured_threshold_accepts_all_hw_clean_donors"] is False
    assert summary["recommended_frontier_invariant_torus_min"] is None
    assert summary["recommended_frontier_kam_min"] is None


def test_donor_label_keeps_parent_for_nested_run_dirs(tmp_path):
    module = load_calibration_module()
    run_dir = tmp_path / "campaign" / "mpol=10-ntor=10-abcd"
    run_dir.mkdir(parents=True)

    assert module.donor_label(run_dir) == "campaign/mpol=10-ntor=10-abcd"


def test_score_donor_run_uses_root_artifacts_and_scorer_contract(
    tmp_path,
    monkeypatch,
):
    module = load_calibration_module()
    run_dir = tmp_path / "campaign" / "donor"
    run_dir.mkdir(parents=True)
    (run_dir / "biot_savart_opt.json").write_text("{}", encoding="utf-8")
    (run_dir / "surf_opt_boozer_surface.json").write_text("{}", encoding="utf-8")
    (run_dir / "results.json").write_text(
        json.dumps(
            {
                "HARDWARE_CONSTRAINTS_OK": True,
                "ALM_HARD_CONSTRAINTS_FEASIBLE": True,
                "SEARCH_OBJECTIVE_J": 1.25,
                "NONQS_RATIO": 2.5e-4,
                "BOOZER_RESIDUAL": 3.5e-6,
                "FINAL_IOTA": 0.18,
                "FINAL_VOLUME": 0.105,
            }
        ),
        encoding="utf-8",
    )

    class BoozerWrapper:
        surface = "unwrapped-surface"

    def fake_load(path):
        if path.name == "surf_opt_boozer_surface.json":
            return BoozerWrapper()
        if path.name == "biot_savart_opt.json":
            return "biot-savart"
        raise AssertionError(f"unexpected load path {path}")

    captured = {}

    def fake_score_topology(surface, bfield, **kwargs):
        captured["surface"] = surface
        captured["bfield"] = bfield
        captured["kwargs"] = dict(kwargs)
        return {
            "survival_fraction": 0.75,
            "survived_lines": 9,
            "nfieldlines": 12,
            "tmax": 50.0,
            "mean_exit_time": None,
            "confinement_score": 0.91,
            "mean_line_loss": 0.1,
            "worst_k_line_loss": 0.2,
            "early_exit_fraction": 0.25,
            "confinement_loss": 0.12,
            "confinement_surrogate_k": 3,
            "confinement_early_exit_threshold": 0.2,
            "invariant_torus_fraction": None,
            "kam_fraction": None,
            "kam_median_width": 0.08,
            "kam_width_ratio": 0.25,
            "cross_section_span": 0.19,
            "stop_reason_counts": {"surface_exit": 3},
            "first_exit": None,
            "per_phi_hit_counts": [10, 10, 9, 8],
            "line_metrics": [],
            "line_lifetimes": [],
            "line_losses": [],
            "seed_contract": {"nfieldlines": 12},
            "field_model": {"selected_mode": "native"},
            "transport_diagnostics": {"status": "skipped_by_caller"},
        }

    monkeypatch.setattr(module, "load", fake_load)
    monkeypatch.setitem(
        module.topology_surface_for_scoring.__globals__, "load", fake_load
    )
    monkeypatch.setattr(module, "score_topology", fake_score_topology)

    row = module.score_donor_run(
        run_dir,
        nfieldlines=12,
        tmax=50.0,
        nphis=4,
        kam_width_ratio=0.25,
        inset_fraction=0.05,
    )

    assert captured["surface"] == "unwrapped-surface"
    assert captured["bfield"] == "biot-savart"
    assert captured["kwargs"]["nfieldlines"] == 12
    assert captured["kwargs"]["compute_transport_diagnostics"] is False
    assert row.donor_label == "campaign/donor"
    assert row.surface_path.endswith("surf_opt_boozer_surface.json")
    assert row.hardware_constraints_ok is True
    assert row.topology_broken is False
    assert row.invariant_torus_fraction is None
    assert row.kam_fraction is None
    assert json.loads(row.seed_contract_json) == {"nfieldlines": 12}

    payload = module.calibration_payload(
        rows=[row],
        run_dirs=[run_dir],
        nfieldlines=12,
        tmax=50.0,
        nphis=4,
        kam_width_ratio=0.25,
        inset_fraction=0.05,
        frontier_invariant_torus_min=0.0,
        selection_margin=1.0 / 12.0,
    )
    assert payload["summary"]["hw_clean_donor_count"] == 1
    assert payload["summary"]["hw_clean_evaluated_donor_count"] == 0
    assert payload["summary"]["recommended_frontier_invariant_torus_min"] is None
    csv_path = tmp_path / "calibration.csv"
    module.write_csv(csv_path, [row])
    assert "campaign/donor" in csv_path.read_text(encoding="utf-8")
