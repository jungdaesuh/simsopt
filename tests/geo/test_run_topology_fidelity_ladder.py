import json
from types import SimpleNamespace

from geo._frontier_test_helpers import ensure_examples_import_path

ensure_examples_import_path()

from banana_opt.design_only_fields import build_design_only_results_fields
import run_topology_fidelity_ladder as ladder


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
        assert tier_record["evaluation_error_type"] == "DesignOnlyTopologyFieldError"
