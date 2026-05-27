import json
import subprocess
import sys

from geo._frontier_test_helpers import EXAMPLE_ROOT


FRONTIER_TRAJECTORY_SCRIPT = EXAMPLE_ROOT / "frontier_pareto_trajectory.py"


def run_frontier_trajectory(run_dir, output_dir):
    subprocess.run(
        [
            sys.executable,
            str(FRONTIER_TRAJECTORY_SCRIPT),
            str(run_dir),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    return json.loads(
        (output_dir / "frontier_pareto_trajectory.json").read_text(encoding="utf-8")
    )


def test_frontier_pareto_trajectory_uses_topology_archive_and_root_metadata(tmp_path):
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "out"
    run_dir.mkdir()
    output_dir.mkdir()
    (run_dir / "topology_archive.jsonl").write_text(
        json.dumps(
            {
                "accepted_iteration": 1,
                "J": 12.5,
                "checkpoint_objective_total": 13.0,
                "invariant_torus_fraction": 1.0 / 12.0,
                "kam_fraction": 0.75,
                "frontier_invariant_torus_min": 0.30,
                "frontier_kam_min": 0.20,
                "frontier_certification_ok": False,
                "frontier_certification_reason": "invariant_torus_fraction_below_min",
                "frontier_certification_hardware_ok": True,
                "survival_fraction": 1.0,
                "topology_broken": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "log.txt").write_text(
        "\n".join(
            (
                "ITERATION 1",
                "Objective J                         = 1.250000e+01",
                "nonQS ratio                         = 5.000000e-04 (dJ = 1.0)",
                "Boozer Residual                     = 2.000000e-05 (dJ = 1.0)",
                "Iotas (actual)                      = 0.1200, 0.1800",
                "Volume                              = 0.0900, 0.1100",
                "Hardware Constraints OK             = True",
            )
        ),
        encoding="utf-8",
    )
    (run_dir / "results.json").write_text(
        json.dumps(
            {
                "iterations": 1,
                "SEARCH_OBJECTIVE_J": 12.5,
                "NONQS_RATIO": 5.0e-4,
                "BOOZER_RESIDUAL": 2.0e-5,
                "FINAL_IOTA": 0.18,
                "FINAL_VOLUME": 0.11,
                "FRONTIER_INVARIANT_TORUS_FRACTION": 1.0 / 12.0,
                "FRONTIER_INVARIANT_TORUS_MIN": 0.30,
                "FRONTIER_KAM_FRACTION": 0.75,
                "FRONTIER_KAM_MIN": 0.20,
                "HARDWARE_CONSTRAINTS_OK": True,
                "FRONTIER_CERTIFICATION_OK": False,
            }
        ),
        encoding="utf-8",
    )

    payload = run_frontier_trajectory(run_dir, output_dir)
    rows = payload["rows"]

    archive_row = [row for row in rows if row["source_kind"] == "topology_archive"][0]
    assert archive_row["source_kind"] == "topology_archive"
    assert archive_row["invariant_torus_fraction"] == 1.0 / 12.0
    assert archive_row["invariant_torus_min"] == 0.30
    assert archive_row["kam_fraction"] == 0.75
    assert archive_row["kam_min"] == 0.20
    assert archive_row["frontier_certification_ok"] is False
    assert (
        archive_row["frontier_certification_reason"]
        == "invariant_torus_fraction_below_min"
    )
    assert archive_row["qa_error"] == 5.0e-4
    assert archive_row["boozer_residual"] == 2.0e-5
    assert archive_row["iota"] == 0.18
    assert archive_row["hardware_ok"] is True
    final_row = [row for row in rows if row["source_kind"] == "final_results"][0]
    assert final_row["frontier_certification_ok"] is False
    assert final_row["invariant_torus_fraction"] == 1.0 / 12.0
    assert final_row["invariant_torus_min"] == 0.30
    assert final_row["kam_fraction"] == 0.75
    assert final_row["kam_min"] == 0.20


def test_frontier_pareto_trajectory_keeps_legacy_kam_separate_without_semantics(tmp_path):
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "out"
    run_dir.mkdir()
    output_dir.mkdir()
    (run_dir / "topology_archive.jsonl").write_text(
        json.dumps(
            {
                "accepted_iteration": 2,
                "J": 4.0,
                "kam_fraction": 0.75,
                "frontier_kam_min": 0.20,
                "frontier_certification_ok": True,
                "frontier_certification_reason": "legacy_certified",
                "survival_fraction": 1.0,
                "topology_broken": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "solver_state_checkpoint.json").write_text(
        json.dumps(
            {
                "accepted_iterations": 3,
                "accepted_incumbent": {
                    "search_eval": {
                        "total": 5.0,
                        "frontier_kam_fraction": 0.80,
                        "frontier_kam_min": 0.25,
                        "frontier_certification_ok": True,
                    },
                    "surface_status": {"iotas": [0.18], "volumes": [0.11]},
                    "accepted_hardware_status": {"success": True},
                },
            }
        ),
        encoding="utf-8",
    )

    payload = run_frontier_trajectory(run_dir, output_dir)
    archive_row = [
        row for row in payload["rows"] if row["source_kind"] == "topology_archive"
    ][0]
    checkpoint_row = [
        row for row in payload["rows"] if row["source_kind"] == "solver_state_checkpoint"
    ][0]

    assert archive_row["invariant_torus_fraction"] is None
    assert archive_row["invariant_torus_min"] is None
    assert archive_row["kam_fraction"] == 0.75
    assert archive_row["kam_min"] == 0.20
    assert checkpoint_row["invariant_torus_fraction"] is None
    assert checkpoint_row["invariant_torus_min"] is None
    assert checkpoint_row["kam_fraction"] == 0.80
    assert checkpoint_row["kam_min"] == 0.25


def test_frontier_pareto_trajectory_joins_resumed_iteration_log(tmp_path):
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "out"
    run_dir.mkdir()
    output_dir.mkdir()
    (run_dir / "topology_archive.jsonl").write_text(
        json.dumps(
            {
                "accepted_iteration": 6,
                "J": 12.5,
                "invariant_torus_fraction": 0.5,
                "kam_fraction": 0.5,
                "survival_fraction": 1.0,
                "topology_broken": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "log.txt").write_text(
        "\n".join(
            (
                "ITERATION 6",
                "nonQS ratio                         = 6.000000e-04 (dJ = 1.0)",
                "Boozer Residual                     = 3.000000e-05 (dJ = 1.0)",
                "Iotas (actual)                      = 0.1200, 0.1900",
                "Volume                              = 0.0900, 0.1200",
                "Hardware Constraints OK             = True",
            )
        ),
        encoding="utf-8",
    )

    payload = run_frontier_trajectory(run_dir, output_dir)
    archive_row = [row for row in payload["rows"] if row["source_kind"] == "topology_archive"][0]

    assert archive_row["accepted_iteration"] == 6
    assert archive_row["qa_error"] == 6.0e-4
    assert archive_row["boozer_residual"] == 3.0e-5
    assert archive_row["iota"] == 0.19
    assert archive_row["volume"] == 0.12
    assert archive_row["hardware_ok"] is True


def test_frontier_pareto_trajectory_reads_posthoc_topology_file(tmp_path):
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "out"
    run_dir.mkdir()
    output_dir.mkdir()
    (run_dir / "results_best_accepted.partial.json").write_text(
        json.dumps(
            {
                "iterations": 9,
                "SEARCH_OBJECTIVE_J": -0.701,
                "NONQS_RATIO": 6.5e-5,
                "BOOZER_RESIDUAL": 9.6e-7,
                "FINAL_IOTA": 0.2077,
                "FINAL_VOLUME": 0.064,
                "HARDWARE_CONSTRAINTS_OK": False,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "topology_eval_posthoc.json").write_text(
        json.dumps(
            {
                "survival_fraction": 1.0,
                "broken": False,
                "invariant_torus_fraction": 1.0 / 12.0,
                "kam_fraction": 1.0 / 12.0,
            }
        ),
        encoding="utf-8",
    )

    payload = run_frontier_trajectory(run_dir, output_dir)
    posthoc_row = [
        row for row in payload["rows"] if row["source_kind"] == "topology_posthoc"
    ][0]

    assert posthoc_row["accepted_iteration"] == 9
    assert posthoc_row["topology_source_artifact_path"].endswith(
        "topology_eval_posthoc.json"
    )
    assert posthoc_row["hardware_ok"] is False
    assert posthoc_row["invariant_torus_fraction"] == 1.0 / 12.0
    assert posthoc_row["kam_fraction"] == 1.0 / 12.0
    assert posthoc_row["survival_fraction"] == 1.0


def test_frontier_pareto_trajectory_writes_outputs_for_empty_run(tmp_path):
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "out"
    run_dir.mkdir()
    output_dir.mkdir()

    payload = run_frontier_trajectory(run_dir, output_dir)

    assert (output_dir / "frontier_pareto_trajectory.png").is_file()
    assert (output_dir / "frontier_pareto_trajectory.json").is_file()
    assert (output_dir / "frontier_pareto_trajectory.csv").is_file()
    assert payload["rows"] == []
