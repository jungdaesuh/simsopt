"""Contract tests for the single-stage 11-vs-51 matrix generator + consumer.

``benchmarks/perlmutter/build_single_stage_matrix.py`` is the SSOT for the
production matrix; ``submit_single_stage_matrix.py`` consumes its manifest to
emit ``sbatch`` commands. After the migration off the monolithic on-device lane
(which OOMs at production resolution), the matrix must be exactly the **8
host-driven cells** ``{scipy-jax (11), scipy-jax-fullgraph (51)} × {cpu, gpu} ×
{mpol2, mpol10}``, quasi-newton inner only, with **no ``ondevice`` cell**.

These tests pin that observable contract -- the manifest shape and the sbatch
export the submit consumer actually produces -- so a regression (re-adding
ondevice, re-introducing an inner-LS axis, or breaking the mpol10 warm-start
gate) fails loudly. They are behavior-distinguishing: they would fail against the
pre-migration 24-cell ondevice generator.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.benchmark_timing_labels import MIXED_PARITY_REFERENCE
from benchmarks.perlmutter.build_single_stage_matrix import build_manifest
from benchmarks.perlmutter.submit_single_stage_matrix import sbatch_command, select

REPO_ROOT = Path(__file__).resolve().parents[2]


def _export_kv(cmd: list[str]) -> dict[str, str]:
    """KEY=VALUE pairs from the single ``--export=`` argument of an sbatch cmd."""
    export_arg = next(a for a in cmd if a.startswith("--export="))
    kv: dict[str, str] = {}
    for item in export_arg[len("--export="):].split(","):
        if "=" in item:
            key, value = item.split("=", 1)
            kv[key] = value
    return kv


def test_matrix_is_eight_host_driven_cells_without_ondevice():
    manifest = build_manifest("testsha")
    cells = manifest["cells"]
    assert len(cells) == 8
    backends = sorted(c["optimizer_backend"] for c in cells)
    assert backends == ["scipy-jax"] * 4 + ["scipy-jax-fullgraph"] * 4
    # The on-device *outer* monolith is gone (it OOMs): no cell runs the ondevice
    # outer optimizer, and the coupling map names only host-driven lanes. (ondevice
    # still legitimately appears as the inner Boozer backend on the fullgraph lane.)
    assert all(c["optimizer_backend"] != "ondevice" for c in cells)
    assert "ondevice" not in json.dumps(manifest["formulation_backend_coupling"])
    assert manifest["formulation_backend_coupling"] == {
        "11": "scipy-jax",
        "51": "scipy-jax-fullgraph",
    }


def test_formulation_dim_is_coupled_to_backend_and_reference():
    for cell in build_manifest("testsha")["cells"]:
        assert cell["outer_optimizer"] == "host-scipy"
        if cell["formulation_dim"] == 11:
            assert cell["optimizer_backend"] == "scipy-jax"
            # 11-dim reduced lane is dim-mismatched vs the 51 cpp reference.
            assert cell["dim_matched_reference"] is False
        elif cell["formulation_dim"] == 51:
            assert cell["optimizer_backend"] == "scipy-jax-fullgraph"
            assert cell["dim_matched_reference"] is True
        else:
            pytest.fail(f"unexpected formulation_dim {cell['formulation_dim']}")


def test_inner_solver_is_quasi_newton_only():
    manifest = build_manifest("testsha")
    assert manifest["inner_boozer_least_squares"] == "quasi-newton"
    # No inner-LS axis: every cell leaves the LS-algorithm env empty so the
    # launcher omits BOOZER_LEAST_SQUARES_ALGORITHM and the child falls back to
    # its quasi-newton default. No lm / lm-minpack cells exist.
    ls_values = {c["env"]["PROD_BOOZER_LS_ALGORITHM"] for c in manifest["cells"]}
    assert ls_values == {""}


def test_every_cell_satisfies_submit_consumer_schema():
    manifest = build_manifest("testsha")
    # Exactly the fields submit_single_stage_matrix.py reads off each cell.
    required = {"id", "status", "tier", "formulation_dim", "platform",
                "env", "account", "sbatch_extra", "launcher",
                "timing_classification", "supports_performance_headline"}
    for cell in manifest["cells"]:
        assert required <= set(cell), required - set(cell)
    # The consumer reads the warm-start requirement off the tier, not the cell.
    assert manifest["tiers"]["mpol2"]["warm_start"] is None
    assert manifest["tiers"]["mpol10"]["warm_start"] is not None


def test_matrix_cells_are_labeled_mixed_parity_reference_not_headline():
    manifest = build_manifest("testsha")
    assert "mixed-parity-reference" in " ".join(manifest["notes"])
    for cell in manifest["cells"]:
        assert cell["timing_classification"] == MIXED_PARITY_REFERENCE
        assert cell["timing_includes_gpu_target"] is (cell["platform"] == "gpu")
        assert cell["timing_includes_cpu_reference"] is True
        assert cell["supports_performance_headline"] is False
        assert cell["headline_timing_classification"] is None
        assert cell["env"]["PROD_TIMING_CLASSIFICATION"] == MIXED_PARITY_REFERENCE
        assert cell["env"]["PROD_SUPPORTS_PERFORMANCE_HEADLINE"] == "0"


def test_production_launchers_default_to_decomposed_backend():
    for launcher_name in (
        "single_stage_production_cpu.slurm",
        "single_stage_production_gpu.slurm",
    ):
        launcher = (
            REPO_ROOT / "benchmarks" / "perlmutter" / launcher_name
        ).read_text(encoding="utf-8")
        assert (
            'PROD_OPTIMIZER_BACKEND="${PROD_OPTIMIZER_BACKEND:-scipy-jax-decomposed}"'
            in launcher
        )
        assert (
            'PROD_TIMING_CLASSIFICATION="${PROD_TIMING_CLASSIFICATION:-mixed-parity-reference}"'
            in launcher
        )
        assert "benchmark_timing_label.json" in launcher


def test_submit_emits_fullgraph_backend_without_ls_override():
    manifest = build_manifest("testsha")
    [cell] = select(manifest["cells"], status=None, tier="mpol2", dim=51,
                    platform="cpu", ids=None)
    cmd = sbatch_command(cell, tiers=manifest["tiers"], repo_root="/tmp/co",
                         run_root_base="/tmp/runs", warm_start=None)
    kv = _export_kv(cmd)
    assert kv["PROD_OPTIMIZER_BACKEND"] == "scipy-jax-fullgraph"
    assert kv["PROD_TIMING_CLASSIFICATION"] == MIXED_PARITY_REFERENCE
    assert kv["PROD_SUPPORTS_PERFORMANCE_HEADLINE"] == "0"
    # Empty LS algorithm is dropped from the export -> child uses quasi-newton.
    assert "PROD_BOOZER_LS_ALGORITHM" not in kv
    assert kv["PROD_MPOL"] == "2"


def test_submit_gates_mpol10_on_warm_start():
    manifest = build_manifest("testsha")
    [cell] = select(manifest["cells"], status=None, tier="mpol10", dim=51,
                    platform="gpu", ids=None)
    # Production tier without the continuation donor must fail closed.
    with pytest.raises(SystemExit):
        sbatch_command(cell, tiers=manifest["tiers"], repo_root="/tmp/co",
                       run_root_base="/tmp/runs", warm_start=None)
    cmd = sbatch_command(cell, tiers=manifest["tiers"], repo_root="/tmp/co",
                         run_root_base="/tmp/runs", warm_start="/tmp/donor")
    assert _export_kv(cmd)["PROD_WARM_START_RUN_DIR"] == "/tmp/donor"


def test_fullgraph_cells_force_ondevice_inner_boozer():
    # The fullgraph inner Boozer solve defaults to scipy on CPU, which
    # jax_cpu_parity rejects at boozer_surface.py:5659 (gate b5f97fdf9). Fullgraph
    # cells must force ondevice; the reduced lane already resolves to ondevice by
    # the child default and is left unset (forcing it would also flip the harness
    # Newton-polish-policy resolution).
    for cell in build_manifest("testsha")["cells"]:
        if cell["optimizer_backend"] == "scipy-jax-fullgraph":
            assert cell["inner_boozer_optimizer_backend"] == "ondevice"
            assert cell["env"]["PROD_BOOZER_OPTIMIZER_BACKEND"] == "ondevice"
        else:
            assert cell["inner_boozer_optimizer_backend"] == ""
            assert cell["env"]["PROD_BOOZER_OPTIMIZER_BACKEND"] == ""


def test_submit_exports_boozer_backend_only_for_fullgraph():
    manifest = build_manifest("testsha")
    [full] = select(manifest["cells"], status=None, tier="mpol2", dim=51,
                    platform="cpu", ids=None)
    full_kv = _export_kv(sbatch_command(full, tiers=manifest["tiers"],
                                        repo_root="/tmp/co", run_root_base="/tmp/runs",
                                        warm_start=None))
    assert full_kv["PROD_BOOZER_OPTIMIZER_BACKEND"] == "ondevice"
    [reduced] = select(manifest["cells"], status=None, tier="mpol2", dim=11,
                       platform="cpu", ids=None)
    reduced_kv = _export_kv(sbatch_command(reduced, tiers=manifest["tiers"],
                                           repo_root="/tmp/co", run_root_base="/tmp/runs",
                                           warm_start=None))
    # empty value dropped from the export -> reduced lane uses the child default
    assert "PROD_BOOZER_OPTIMIZER_BACKEND" not in reduced_kv
