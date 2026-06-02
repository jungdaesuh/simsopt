#!/usr/bin/env python3
"""Prepare a strict CUDA low-resolution proof packet.

The packet is intentionally a preparation artifact: it inventories the supplied
external artifacts, pins the repo fixture used by the runnable low-resolution
Stage 2 and single-stage lanes, and writes a shell runner for a CUDA host.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
import zipfile

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.single_stage_smoke_defaults import (
    DEFAULT_PLASMA_SURF_FILENAME,
    DEFAULT_STAGE2_BS_PATH,
)
from benchmarks.single_stage_smoke_fixture import (
    DEFAULT_EQUILIBRIA_DIR,
    DEFAULT_IOTA_TARGET,
    DEFAULT_SMOKE_MPOL,
    DEFAULT_SMOKE_NPHI,
    DEFAULT_SMOKE_NTHETA,
    DEFAULT_SMOKE_NTOR,
    DEFAULT_VOL_TARGET,
)
from benchmarks.validation_ladder_contract import (
    TIER3_SINGLE_STAGE_OUTER_LOOP_RUNG,
    single_stage_proof_contract,
)


DEFAULT_OUTPUT_DIR = REPO_ROOT / ".artifacts" / "cuda_gpu_lowres_tests"
CUDA_DETERMINISM_XLA_FLAG = "--xla_gpu_exclude_nondeterministic_ops=true"


@dataclass(frozen=True)
class LowresCudaPrepConfig:
    boozer_surface_zip: Path
    autoresearch_runs_dir: Path
    stage2_bs_path: Path
    output_dir: Path
    stage2_nphi: int
    stage2_ntheta: int
    stage2_maxiter: int
    single_stage_nphi: int
    single_stage_ntheta: int
    single_stage_mpol: int
    single_stage_ntor: int
    single_stage_outer_maxiter: int
    candidate_limit: int
    warm_start_run_dir: Path | None = None


def _repo_path(path: Path) -> str:
    resolved = path.resolve()
    repo_root = REPO_ROOT.resolve()
    if resolved == repo_root or repo_root in resolved.parents:
        return resolved.relative_to(repo_root).as_posix()
    return str(resolved)


def _resolve_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as infile:
        for chunk in iter(lambda: infile.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as infile:
        loaded = json.load(infile)
    if not isinstance(loaded, dict):
        raise ValueError(f"expected JSON object in {path}")
    return loaded


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as outfile:
        json.dump(payload, outfile, indent=2, sort_keys=True)
        outfile.write("\n")


def _zip_inventory(path: Path) -> dict[str, object]:
    entries: list[dict[str, object]] = []
    with zipfile.ZipFile(path) as archive:
        for info in sorted(archive.infolist(), key=lambda item: item.filename):
            if info.is_dir():
                continue
            data = archive.read(info.filename)
            entries.append(
                {
                    "name": info.filename,
                    "size_bytes": int(info.file_size),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            )
    return {
        "path": _repo_path(path),
        "sha256": _sha256_file(path),
        "entry_count": len(entries),
        "entries": entries,
    }


def _stage2_seed_inventory(stage2_bs_path: Path) -> dict[str, object]:
    results_path = stage2_bs_path.with_name("results.json")
    if not stage2_bs_path.is_file():
        raise FileNotFoundError(f"missing Stage 2 seed: {stage2_bs_path}")
    if not results_path.is_file():
        raise FileNotFoundError(f"missing Stage 2 seed results: {results_path}")
    results = _load_json(results_path)
    keys = (
        "MAJOR_RADIUS",
        "TOROIDAL_FLUX",
        "order",
        "banana_surf_radius",
        "TF_CURRENT_A",
        "HARDWARE_CONSTRAINTS_OK",
        "FIELD_ERROR",
        "OPTIMIZER_SUCCESS",
    )
    return {
        "biot_savart_path": _repo_path(stage2_bs_path),
        "biot_savart_sha256": _sha256_file(stage2_bs_path),
        "results_path": _repo_path(results_path),
        "results_sha256": _sha256_file(results_path),
        "results_summary": {key: results.get(key) for key in keys},
    }


def _warm_start_run_inventory(path: Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    surf_opt_path = path / "surf_opt.json"
    results_path = path / "results.json"
    if not surf_opt_path.is_file():
        raise FileNotFoundError(f"missing warm-start surf_opt.json: {surf_opt_path}")
    if not results_path.is_file():
        raise FileNotFoundError(f"missing warm-start results.json: {results_path}")
    biot_savart_path = path / "biot_savart_opt.json"
    return {
        "run_dir": _repo_path(path),
        "surf_opt_sha256": _sha256_file(surf_opt_path),
        "results_sha256": _sha256_file(results_path),
        "has_biot_savart_opt": biot_savart_path.is_file(),
        "biot_savart_sha256": (
            _sha256_file(biot_savart_path) if biot_savart_path.is_file() else None
        ),
    }


def _discover_warm_start_candidates(
    runs_dir: Path,
    *,
    limit: int,
) -> dict[str, object]:
    candidates: list[dict[str, object]] = []
    total = 0
    for surf_opt_path in sorted(runs_dir.rglob("surf_opt.json")):
        run_dir = surf_opt_path.parent
        if not (run_dir / "results.json").is_file():
            continue
        total += 1
        if len(candidates) >= limit:
            continue
        biot_savart_path = run_dir / "biot_savart_opt.json"
        candidates.append(
            {
                "run_dir": _repo_path(run_dir),
                "has_biot_savart_opt": biot_savart_path.is_file(),
                "resolution_hint": run_dir.name,
            }
        )
    return {
        "path": _repo_path(runs_dir),
        "candidate_count": total,
        "listed_candidate_limit": int(limit),
        "listed_candidates": candidates,
    }


def _cuda_env(output_dir: Path, *, cache_label: str) -> dict[str, str]:
    cache_dir = output_dir / "jax_compilation_cache" / cache_label
    return {
        "PYTHONPATH": os.pathsep.join([str(REPO_ROOT), str(REPO_ROOT / "src")]),
        "SIMSOPT_BACKEND_MODE": "jax_gpu_parity",
        "SIMSOPT_BACKEND_STRICT": "1",
        "SIMSOPT_JAX_PLATFORM": "cuda",
        "SIMSOPT_JAX_BACKEND": "cuda",
        "SIMSOPT_JAX_TRANSFER_GUARD": "disallow",
        "SIMSOPT_JAX_CUDA_LIBRARY_MODE": "bundled",
        "SIMSOPT_JAX_GPU_PREALLOCATE": "false",
        "JAX_ENABLE_X64": "1",
        "JAX_PLATFORMS": "cuda,cpu",
        "JAX_COMPILATION_CACHE_DIR": str(cache_dir),
        "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0",
        "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES": "-1",
        "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "all",
        "XLA_FLAGS": CUDA_DETERMINISM_XLA_FLAG,
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    }


def _command(
    *,
    name: str,
    purpose: str,
    command: list[str],
    env: dict[str, str],
    outputs: dict[str, str],
    acceptance: list[str],
) -> dict[str, object]:
    return {
        "name": name,
        "purpose": purpose,
        "env": env,
        "command": command,
        "expected_outputs": outputs,
        "acceptance": acceptance,
    }


def _commands(config: LowresCudaPrepConfig) -> list[dict[str, object]]:
    output_dir = config.output_dir
    stage2_output = output_dir / "stage2_cuda_lowres_e2e.json"
    single_stage_parity_output = output_dir / "single_stage_cuda_init_parity.json"
    outer_loop_output = output_dir / "single_stage_cuda_outer_loop_probe.json"
    target_lane_output_root = output_dir / "single_stage_target_lane_memory"
    stage2_bs_path = _repo_path(config.stage2_bs_path)
    equilibria_dir = _repo_path(Path(DEFAULT_EQUILIBRIA_DIR))
    warm_start_args = (
        []
        if config.warm_start_run_dir is None
        else ["--warm-start-run-dir", _repo_path(config.warm_start_run_dir)]
    )

    return [
        _command(
            name="cuda_backend_unit_guardrails",
            purpose=(
                "Verify CUDA determinism and GPU-memory helper guardrails before "
                "running physics probes."
            ),
            env=_cuda_env(output_dir, cache_label="backend-unit-guardrails"),
            command=[
                "python",
                "-m",
                "pytest",
                "-q",
                "tests/test_backend.py",
                "-k",
                "cuda_determinism or gpu_memory",
            ],
            outputs={},
            acceptance=[
                "pytest exits 0",
                "CUDA determinism flag checks pass",
                "GPU memory selector/unit tests pass",
            ],
        ),
        _command(
            name="stage2_cuda_lowres_target_e2e",
            purpose=(
                "Run low-resolution Stage 2 CPU-vs-CUDA endpoint, matched-state "
                "parity, performance, and warm timing comparison."
            ),
            env=_cuda_env(output_dir, cache_label="stage2-lowres-e2e"),
            command=[
                "python",
                "benchmarks/stage2_e2e_comparison.py",
                "--platform",
                "cuda",
                "--optimizer-backend",
                "ondevice",
                "--nphi",
                str(config.stage2_nphi),
                "--ntheta",
                str(config.stage2_ntheta),
                "--maxiter",
                str(config.stage2_maxiter),
                "--output-json",
                _repo_path(stage2_output),
            ],
            outputs={"json": _repo_path(stage2_output)},
            acceptance=[
                "JSON exists",
                "payload passed is true",
                "provenance backend is cuda or gpu",
                "provenance transfer_guard is disallow",
                "timings include JAX elapsed fields",
            ],
        ),
        _command(
            name="single_stage_cuda_init_parity",
            purpose=(
                "Run reduced single-stage CPU-vs-CUDA init parity with durable "
                "per-lane artifacts."
            ),
            env=_cuda_env(output_dir, cache_label="single-stage-init-parity"),
            command=[
                "python",
                "benchmarks/single_stage_init_parity.py",
                "--platform",
                "cuda",
                "--stage2-bs-path",
                stage2_bs_path,
                *warm_start_args,
                "--optimizer-backend",
                "ondevice",
                "--benchmark-mode",
                "--nphi",
                str(config.single_stage_nphi),
                "--ntheta",
                str(config.single_stage_ntheta),
                "--mpol",
                str(config.single_stage_mpol),
                "--ntor",
                str(config.single_stage_ntor),
                "--maxiter",
                "0",
                "--output-json",
                _repo_path(single_stage_parity_output),
                "--case-artifacts-dir",
                _repo_path(output_dir / "single_stage_init_cases"),
            ],
            outputs={
                "json": _repo_path(single_stage_parity_output),
                "case_artifacts_dir": _repo_path(output_dir / "single_stage_init_cases"),
            },
            acceptance=[
                "JSON exists",
                "payload passed is true",
                "provenance backend is cuda or gpu",
                "strict_transfer_support status is supported",
            ],
        ),
        _command(
            name="single_stage_cuda_outer_loop_strict_transfer",
            purpose=(
                "Run the reduced real single-stage outer-loop proof under CUDA "
                "strict transfer guard."
            ),
            env=_cuda_env(output_dir, cache_label="single-stage-outer-loop"),
            command=[
                "python",
                "benchmarks/single_stage_outer_loop_probe.py",
                "--platform",
                "cuda",
                "--stage2-bs-path",
                stage2_bs_path,
                "--maxiter",
                str(config.single_stage_outer_maxiter),
                "--profile-target-lane",
                "--profile-target-lane-batch-size",
                "1",
                "--deterministic-gpu-reductions",
                "--output-json",
                _repo_path(outer_loop_output),
            ],
            outputs={"json": _repo_path(outer_loop_output)},
            acceptance=[
                "JSON exists",
                "payload passed is true",
                "provenance transfer_guard is disallow",
                "probe objective_decreased is true",
                "probe finite_result_keys are all true",
            ],
        ),
        _command(
            name="single_stage_cuda_target_lane_memory_profile",
            purpose=(
                "Compile and profile the single-stage target lane, including XLA "
                "memory_analysis byte counts."
            ),
            env=_cuda_env(output_dir, cache_label="single-stage-target-lane-memory"),
            command=[
                "python",
                "examples/single_stage_optimization/SINGLE_STAGE/"
                "single_stage_banana_example.py",
                "--backend",
                "jax",
                "--optimizer-backend",
                "ondevice",
                "--stage2-bs-path",
                stage2_bs_path,
                *warm_start_args,
                "--plasma-surf-filename",
                DEFAULT_PLASMA_SURF_FILENAME,
                "--equilibria-dir",
                equilibria_dir,
                "--output-root",
                _repo_path(target_lane_output_root),
                "--nphi",
                str(config.single_stage_nphi),
                "--ntheta",
                str(config.single_stage_ntheta),
                "--mpol",
                str(config.single_stage_mpol),
                "--ntor",
                str(config.single_stage_ntor),
                "--vol-target",
                str(DEFAULT_VOL_TARGET),
                "--iota-target",
                str(DEFAULT_IOTA_TARGET),
                "--maxiter",
                "0",
                "--benchmark-mode",
                "--disable-target-lane-success-filter",
                "--profile-target-lane-only",
                "--profile-target-lane-memory-analysis",
                "--profile-target-lane-batch-size",
                "1",
                "--target-lane-profile-progress-json",
                _repo_path(output_dir / "single_stage_target_lane_profile_progress.json"),
            ],
            outputs={
                "output_root": _repo_path(target_lane_output_root),
                "profile_progress_json": _repo_path(
                    output_dir / "single_stage_target_lane_profile_progress.json"
                ),
            },
            acceptance=[
                "run exits 0",
                "results payload contains TARGET_LANE_PROFILE",
                "TARGET_LANE_PROFILE contains memory_analysis",
                "provenance transfer_guard is disallow",
            ],
        ),
    ]


def build_manifest(config: LowresCudaPrepConfig) -> dict[str, object]:
    if not config.boozer_surface_zip.is_file():
        raise FileNotFoundError(f"missing Boozer surface zip: {config.boozer_surface_zip}")
    if not config.autoresearch_runs_dir.is_dir():
        raise FileNotFoundError(
            f"missing autoresearch runs dir: {config.autoresearch_runs_dir}"
        )
    outer_contract = single_stage_proof_contract(TIER3_SINGLE_STAGE_OUTER_LOOP_RUNG)
    return {
        "schema": "simsopt.cuda_lowres_test_packet",
        "schema_version": 1,
        "title": "CUDA low-resolution Stage 2 and single-stage proof packet",
        "repo_root": str(REPO_ROOT),
        "runtime_contract": {
            "platform": "cuda",
            "backend_mode": "jax_gpu_parity",
            "strict_backend": True,
            "transfer_guard": "disallow",
            "x64": True,
            "deterministic_xla_flag": CUDA_DETERMINISM_XLA_FLAG,
            "host_device_transfer_policy": (
                "No silent host-device transfer: CUDA commands run with "
                "SIMSOPT_JAX_TRANSFER_GUARD=disallow."
            ),
        },
        "input_artifacts": {
            "boozer_surface_zip": _zip_inventory(config.boozer_surface_zip),
            "autoresearch_runs": _discover_warm_start_candidates(
                config.autoresearch_runs_dir,
                limit=config.candidate_limit,
            ),
            "stage2_seed": _stage2_seed_inventory(config.stage2_bs_path),
            "warm_start_run": _warm_start_run_inventory(config.warm_start_run_dir),
        },
        "low_resolution": {
            "stage2": {
                "nphi": int(config.stage2_nphi),
                "ntheta": int(config.stage2_ntheta),
                "maxiter": int(config.stage2_maxiter),
            },
            "single_stage": {
                "nphi": int(config.single_stage_nphi),
                "ntheta": int(config.single_stage_ntheta),
                "mpol": int(config.single_stage_mpol),
                "ntor": int(config.single_stage_ntor),
                "init_parity_maxiter": 0,
                "outer_loop_maxiter": int(config.single_stage_outer_maxiter),
                "outer_loop_contract": outer_contract,
            },
        },
        "commands": _commands(config),
    }


def _render_shell_command(command: list[str]) -> str:
    rendered = []
    for token in command:
        if token == "python":
            rendered.append('"${PYTHON}"')
        else:
            rendered.append(shlex.quote(token))
    return " ".join(rendered)


def render_shell_runner(manifest: dict[str, object]) -> str:
    commands = manifest["commands"]
    if not isinstance(commands, list):
        raise ValueError("manifest commands must be a list")
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        'PYTHON="${PYTHON:-python}"',
        f"cd {shlex.quote(str(REPO_ROOT))}",
        "",
    ]
    for command_payload in commands:
        if not isinstance(command_payload, dict):
            raise ValueError("command entry must be an object")
        name = str(command_payload["name"])
        env = command_payload["env"]
        command = command_payload["command"]
        if not isinstance(env, dict) or not isinstance(command, list):
            raise ValueError(f"invalid command payload for {name}")
        lines.append(f"echo {shlex.quote('== ' + name + ' ==')}")
        env_prefix = " ".join(
            f"{key}={shlex.quote(str(value))}"
            for key, value in sorted(env.items())
        )
        lines.append(f"{env_prefix} {_render_shell_command([str(item) for item in command])}")
        lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    outer_contract = single_stage_proof_contract(TIER3_SINGLE_STAGE_OUTER_LOOP_RUNG)
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a CUDA low-resolution proof manifest and shell runner from "
            "the supplied Stage 2/single-stage artifacts."
        )
    )
    parser.add_argument("--boozer-surface-zip", required=True)
    parser.add_argument("--autoresearch-runs-dir", required=True)
    parser.add_argument(
        "--stage2-bs-path",
        default=str(DEFAULT_STAGE2_BS_PATH),
        help="Stage 2 biot_savart_opt.json seed with adjacent results.json.",
    )
    parser.add_argument("--warm-start-run-dir", default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--manifest-json", default=None)
    parser.add_argument("--shell-script", default=None)
    parser.add_argument("--stage2-nphi", type=int, default=31)
    parser.add_argument("--stage2-ntheta", type=int, default=16)
    parser.add_argument("--stage2-maxiter", type=int, default=3)
    parser.add_argument("--single-stage-nphi", type=int, default=DEFAULT_SMOKE_NPHI)
    parser.add_argument("--single-stage-ntheta", type=int, default=DEFAULT_SMOKE_NTHETA)
    parser.add_argument("--single-stage-mpol", type=int, default=DEFAULT_SMOKE_MPOL)
    parser.add_argument("--single-stage-ntor", type=int, default=DEFAULT_SMOKE_NTOR)
    parser.add_argument(
        "--single-stage-outer-maxiter",
        type=int,
        default=int(outer_contract["default_maxiter"]),
    )
    parser.add_argument("--candidate-limit", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = _resolve_path(args.output_dir)
    manifest_path = (
        output_dir / "cuda_gpu_lowres_manifest.json"
        if args.manifest_json is None
        else _resolve_path(args.manifest_json)
    )
    shell_script_path = (
        output_dir / "run_cuda_gpu_lowres_tests.sh"
        if args.shell_script is None
        else _resolve_path(args.shell_script)
    )
    config = LowresCudaPrepConfig(
        boozer_surface_zip=_resolve_path(args.boozer_surface_zip),
        autoresearch_runs_dir=_resolve_path(args.autoresearch_runs_dir),
        stage2_bs_path=_resolve_path(args.stage2_bs_path),
        output_dir=output_dir,
        stage2_nphi=int(args.stage2_nphi),
        stage2_ntheta=int(args.stage2_ntheta),
        stage2_maxiter=int(args.stage2_maxiter),
        single_stage_nphi=int(args.single_stage_nphi),
        single_stage_ntheta=int(args.single_stage_ntheta),
        single_stage_mpol=int(args.single_stage_mpol),
        single_stage_ntor=int(args.single_stage_ntor),
        single_stage_outer_maxiter=int(args.single_stage_outer_maxiter),
        candidate_limit=int(args.candidate_limit),
        warm_start_run_dir=(
            None if args.warm_start_run_dir is None else _resolve_path(args.warm_start_run_dir)
        ),
    )
    manifest = build_manifest(config)
    _write_json(manifest_path, manifest)
    shell_script_path.parent.mkdir(parents=True, exist_ok=True)
    shell_script_path.write_text(render_shell_runner(manifest), encoding="utf-8")
    shell_script_path.chmod(0o755)
    print(f"manifest: {_repo_path(manifest_path)}")
    print(f"runner:   {_repo_path(shell_script_path)}")


if __name__ == "__main__":
    main()
