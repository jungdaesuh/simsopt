from __future__ import annotations

import json
from pathlib import Path
import zipfile

from benchmarks.prepare_cuda_gpu_lowres_tests import (
    CUDA_DETERMINISM_XLA_FLAG,
    LowresCudaPrepConfig,
    build_manifest,
    render_shell_runner,
)
from benchmarks.single_stage_smoke_defaults import DEFAULT_STAGE2_BS_PATH
from benchmarks.single_stage_smoke_fixture import (
    DEFAULT_SMOKE_MPOL,
    DEFAULT_SMOKE_NPHI,
    DEFAULT_SMOKE_NTHETA,
    DEFAULT_SMOKE_NTOR,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_cuda_lowres_manifest_pins_strict_transfer_guard(tmp_path: Path) -> None:
    boozer_zip = tmp_path / "boozer_surfaces.zip"
    with zipfile.ZipFile(boozer_zip, mode="w") as archive:
        archive.writestr("row01_surf_opt_boozer_surface.json", '{"@class":"SIMSON"}')

    run_dir = tmp_path / "runs" / "campaign" / "mpol=8-ntor=6-test"
    _write_json(run_dir / "surf_opt.json", {"@class": "SIMSON"})
    _write_json(run_dir / "results.json", {"FINAL_IOTA": 0.15})

    config = LowresCudaPrepConfig(
        boozer_surface_zip=boozer_zip,
        autoresearch_runs_dir=tmp_path / "runs",
        stage2_bs_path=DEFAULT_STAGE2_BS_PATH,
        output_dir=tmp_path / "out",
        stage2_nphi=31,
        stage2_ntheta=16,
        stage2_maxiter=3,
        single_stage_nphi=DEFAULT_SMOKE_NPHI,
        single_stage_ntheta=DEFAULT_SMOKE_NTHETA,
        single_stage_mpol=DEFAULT_SMOKE_MPOL,
        single_stage_ntor=DEFAULT_SMOKE_NTOR,
        single_stage_outer_maxiter=10,
        candidate_limit=5,
    )

    manifest = build_manifest(config)

    runtime_contract = manifest["runtime_contract"]
    assert isinstance(runtime_contract, dict)
    assert runtime_contract["platform"] == "cuda"
    assert runtime_contract["backend_mode"] == "jax_gpu_parity"
    assert runtime_contract["strict_backend"] is True
    assert runtime_contract["transfer_guard"] == "disallow"
    assert runtime_contract["deterministic_xla_flag"] == CUDA_DETERMINISM_XLA_FLAG

    input_artifacts = manifest["input_artifacts"]
    assert isinstance(input_artifacts, dict)
    boozer_inventory = input_artifacts["boozer_surface_zip"]
    assert isinstance(boozer_inventory, dict)
    assert boozer_inventory["entry_count"] == 1

    runs_inventory = input_artifacts["autoresearch_runs"]
    assert isinstance(runs_inventory, dict)
    assert runs_inventory["candidate_count"] == 1

    commands = manifest["commands"]
    assert isinstance(commands, list)
    commands_by_name = {
        str(command["name"]): command
        for command in commands
        if isinstance(command, dict)
    }
    assert set(commands_by_name) == {
        "cuda_backend_unit_guardrails",
        "stage2_cuda_lowres_target_e2e",
        "single_stage_cuda_init_parity",
        "single_stage_cuda_outer_loop_strict_transfer",
        "single_stage_cuda_target_lane_memory_profile",
    }

    for command in commands_by_name.values():
        env = command["env"]
        assert isinstance(env, dict)
        assert env["SIMSOPT_JAX_TRANSFER_GUARD"] == "disallow"
        assert env["XLA_FLAGS"] == CUDA_DETERMINISM_XLA_FLAG
        assert env["JAX_PLATFORMS"] == "cuda,cpu"

    stage2_command = commands_by_name["stage2_cuda_lowres_target_e2e"]["command"]
    assert isinstance(stage2_command, list)
    assert "--optimizer-backend" in stage2_command
    assert "ondevice" in stage2_command

    memory_command = commands_by_name["single_stage_cuda_target_lane_memory_profile"][
        "command"
    ]
    assert isinstance(memory_command, list)
    assert "--profile-target-lane-memory-analysis" in memory_command
    assert "--profile-target-lane-only" in memory_command

    shell_runner = render_shell_runner(manifest)
    assert "SIMSOPT_JAX_TRANSFER_GUARD=disallow" in shell_runner
    assert "stage2_cuda_lowres_target_e2e" in shell_runner
