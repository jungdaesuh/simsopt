#!/usr/bin/env python3
"""Prepare a strict CUDA low-resolution proof packet.

The packet is intentionally a preparation artifact: it inventories the supplied
external artifacts, pins the repo fixture used by the runnable low-resolution
Stage 2 and single-stage lanes, and writes a shell runner for a CUDA host.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
import zipfile

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.single_stage_smoke_defaults import (
    DEFAULT_STAGE2_BS_PATH,
    DEFAULT_STAGE2_RUNTIME_SPEC_PATH,
)
from benchmarks.single_stage_smoke_fixture import (
    DEFAULT_IOTA_TARGET,
    DEFAULT_SMOKE_MPOL,
    DEFAULT_SMOKE_NPHI,
    DEFAULT_SMOKE_NTHETA,
    DEFAULT_SMOKE_NTOR,
)
from benchmarks.validation_ladder_contract import (
    TIER3_SINGLE_STAGE_OUTER_LOOP_RUNG,
    single_stage_proof_contract,
)
from simsopt_jax.core.surface_fourier_indices import stellsym_scatter_indices


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
    jax_runtime_seed_source: Path = DEFAULT_STAGE2_RUNTIME_SPEC_PATH


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


def _require_mapping(payload: dict[str, object], key: str) -> dict[str, object]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"expected object field {key!r}")
    return value


def _array_payload(values: object) -> dict[str, object]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "dtype": "float64",
        "shape": [int(size) for size in array.shape],
        "data": array.reshape(-1).tolist(),
    }


def _array_from_payload(payload: dict[str, object], *, field_name: str) -> np.ndarray:
    shape = payload.get("shape")
    data = payload.get("data")
    if not isinstance(shape, list) or not isinstance(data, list):
        raise ValueError(f"{field_name} must contain list shape and data fields")
    return np.asarray(data, dtype=np.float64).reshape(
        tuple(int(size) for size in shape)
    )


def _single_stage_half_period_quadpoints(
    *,
    nphi: int,
    ntheta: int,
    nfp: int,
) -> tuple[np.ndarray, np.ndarray]:
    phi = (np.arange(int(nphi), dtype=np.float64) + 0.5) / (2.0 * int(nfp) * int(nphi))
    theta = np.arange(int(ntheta), dtype=np.float64) / int(ntheta)
    return phi, theta


def _theta_basis(quadpoints_theta: np.ndarray, mpol: int) -> np.ndarray:
    theta = 2.0 * np.pi * np.asarray(quadpoints_theta, dtype=np.float64)
    cos_modes = np.arange(0, int(mpol) + 1, dtype=np.float64)
    sin_modes = np.arange(1, int(mpol) + 1, dtype=np.float64)
    return np.concatenate(
        (
            np.cos(theta[:, None] * cos_modes[None, :]),
            np.sin(theta[:, None] * sin_modes[None, :]),
        ),
        axis=1,
    )


def _phi_basis(
    quadpoints_phi: np.ndarray,
    *,
    ntor: int,
    nfp: int,
) -> np.ndarray:
    phi = 2.0 * np.pi * np.asarray(quadpoints_phi, dtype=np.float64)
    cos_modes = int(nfp) * np.arange(0, int(ntor) + 1, dtype=np.float64)
    sin_modes = int(nfp) * np.arange(1, int(ntor) + 1, dtype=np.float64)
    return np.concatenate(
        (
            np.cos(phi[:, None] * cos_modes[None, :]),
            np.sin(phi[:, None] * sin_modes[None, :]),
        ),
        axis=1,
    )


def _surface_xyz_tensor_design_matrix(
    *,
    mpol: int,
    ntor: int,
    nfp: int,
    stellsym: bool,
    quadpoints_phi: np.ndarray,
    quadpoints_theta: np.ndarray,
) -> np.ndarray:
    theta = _theta_basis(quadpoints_theta, int(mpol))
    phi = _phi_basis(quadpoints_phi, ntor=int(ntor), nfp=int(nfp))
    basis_values = phi[:, None, None, :] * theta[None, :, :, None]
    basis_values = np.reshape(
        basis_values,
        (
            phi.shape[0],
            theta.shape[0],
            (2 * int(mpol) + 1) * (2 * int(ntor) + 1),
        ),
    )
    phi_angles = 2.0 * np.pi * np.asarray(quadpoints_phi, dtype=np.float64)
    cos_phi = np.cos(phi_angles)[:, None, None]
    sin_phi = np.sin(phi_angles)[:, None, None]
    zeros = np.zeros_like(basis_values)
    x_block = np.stack(
        (
            basis_values * cos_phi,
            basis_values * sin_phi,
            zeros,
        ),
        axis=2,
    )
    y_block = np.stack(
        (
            -basis_values * sin_phi,
            basis_values * cos_phi,
            zeros,
        ),
        axis=2,
    )
    z_block = np.stack(
        (
            zeros,
            zeros,
            basis_values,
        ),
        axis=2,
    )
    design_matrix = np.reshape(
        np.concatenate((x_block, y_block, z_block), axis=3),
        (-1, 3 * basis_values.shape[2]),
    )
    if not bool(stellsym):
        return design_matrix
    return design_matrix[:, stellsym_scatter_indices(int(mpol), int(ntor))]


def _reproject_xyz_tensor_surface_dofs(
    surface_payload: dict[str, object],
    *,
    mpol: int,
    ntor: int,
    nphi: int,
    ntheta: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if surface_payload.get("surface_class") != "SurfaceXYZTensorFourier":
        raise ValueError(
            "runtime seed reprojection only supports SurfaceXYZTensorFourier; "
            f"got {surface_payload.get('surface_class')!r}"
        )
    nfp = int(surface_payload["nfp"])
    stellsym = bool(surface_payload["stellsym"])
    target_phi, target_theta = _single_stage_half_period_quadpoints(
        nphi=int(nphi),
        ntheta=int(ntheta),
        nfp=nfp,
    )
    source_dofs = _array_from_payload(
        _require_mapping(surface_payload, "dofs"),
        field_name="surface.dofs",
    ).reshape(-1)
    source_design = _surface_xyz_tensor_design_matrix(
        mpol=int(surface_payload["mpol"]),
        ntor=int(surface_payload["ntor"]),
        nfp=nfp,
        stellsym=stellsym,
        quadpoints_phi=target_phi,
        quadpoints_theta=target_theta,
    )
    target_gamma = source_design @ source_dofs
    target_design = _surface_xyz_tensor_design_matrix(
        mpol=int(mpol),
        ntor=int(ntor),
        nfp=nfp,
        stellsym=stellsym,
        quadpoints_phi=target_phi,
        quadpoints_theta=target_theta,
    )
    projected_dofs, _, _, _ = np.linalg.lstsq(
        target_design,
        target_gamma,
        rcond=None,
    )
    return np.asarray(projected_dofs, dtype=np.float64), target_phi, target_theta


def write_runtime_seed_spec(
    *,
    source_path: Path,
    output_path: Path,
    mpol: int,
    ntor: int,
    nphi: int,
    ntheta: int,
) -> None:
    source_payload = _load_json(source_path)
    surface_payload = _require_mapping(source_payload, "surface")
    projected_dofs, target_phi, target_theta = _reproject_xyz_tensor_surface_dofs(
        surface_payload,
        mpol=int(mpol),
        ntor=int(ntor),
        nphi=int(nphi),
        ntheta=int(ntheta),
    )
    output_payload = copy.deepcopy(source_payload)
    output_surface = _require_mapping(output_payload, "surface")
    output_surface["dofs"] = _array_payload(projected_dofs)
    output_surface["mpol"] = int(mpol)
    output_surface["ntor"] = int(ntor)
    output_surface["quadpoints_phi"] = _array_payload(target_phi)
    output_surface["quadpoints_theta"] = _array_payload(target_theta)
    output_payload["quadrature"] = {
        "nphi": int(nphi),
        "ntheta": int(ntheta),
    }
    _write_json(output_path, output_payload)


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


def _runtime_seed_source_inventory(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"missing JAX runtime seed source: {path}")
    payload = _load_json(path)
    surface = payload.get("surface")
    quadrature = payload.get("quadrature")
    return {
        "path": _repo_path(path),
        "sha256": _sha256_file(path),
        "schema": payload.get("schema"),
        "schema_version": payload.get("schema_version"),
        "surface": (
            {
                "mpol": surface.get("mpol"),
                "ntor": surface.get("ntor"),
            }
            if isinstance(surface, dict)
            else None
        ),
        "quadrature": (
            {
                "nphi": quadrature.get("nphi"),
                "ntheta": quadrature.get("ntheta"),
            }
            if isinstance(quadrature, dict)
            else None
        ),
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
        # SAFE narrow autotune cache only. The broad "all" mode forces nvlink
        # through the container CUDA toolkit vs the wheel-bundled NVIDIA libs,
        # which blocked the RunPod cu1290 image (runtime.py:2391-2397 uses this
        # narrow mode for the same reason). Keep aligned with apply_jax_runtime_config.
        "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "xla_gpu_per_fusion_autotune_cache_dir",
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
    stage2_output_root = output_dir / "stage2_cuda_lowres_direct_gpu"
    stage2_results = stage2_output_root / "results.json"
    stage2_biotsavart = stage2_output_root / "stage2_biotsavart.json"
    stage2_boozersurface = stage2_output_root / "stage2_boozersurface.json"
    stage2_surface = stage2_output_root / "stage2_surface.json"
    stage2_log = stage2_output_root / "stage2.log"
    runtime_seed_spec = output_dir / "single_stage_lowres_jax_runtime_spec.json"
    single_stage_init_root = output_dir / "single_stage_cuda_init_soft_penalty"
    single_stage_outer_root = output_dir / "single_stage_cuda_outer_loop_soft_penalty"
    single_stage_restart_root = output_dir / "single_stage_cuda_restart_chain"
    runtime_seed_spec_path = _repo_path(runtime_seed_spec)

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
            name="stage2_cuda_lowres_direct_gpu",
            purpose=(
                "Run the low-resolution Stage 2 JAX soft-penalty CUDA lane and "
                "record restartable output artifacts."
            ),
            env=_cuda_env(output_dir, cache_label="stage2-lowres-direct-gpu"),
            command=[
                "python",
                "examples/single_stage_optimization/STAGE_2/banana_coil_solver.py",
                "--output-dir",
                _repo_path(stage2_output_root),
                "--nphi",
                str(config.stage2_nphi),
                "--ntheta",
                str(config.stage2_ntheta),
                "--maxiter",
                str(config.stage2_maxiter),
                "--banana-order",
                "1",
                "--overwrite",
            ],
            outputs={
                "results_json": _repo_path(stage2_results),
                "biotsavart_json": _repo_path(stage2_biotsavart),
                "boozersurface_json": _repo_path(stage2_boozersurface),
                "surface_json": _repo_path(stage2_surface),
                "log": _repo_path(stage2_log),
                "output_root": _repo_path(stage2_output_root),
            },
            acceptance=[
                "run exits 0",
                "results JSON exists",
                "restart BiotSavart/BoozerSurface/surface JSON outputs exist",
                "log exists",
            ],
        ),
        _command(
            name="single_stage_cuda_runtime_seed_spec",
            purpose=(
                "Reproject the immutable checked-in single-stage JAX runtime "
                "seed spec to the requested low-resolution CUDA proof rung."
            ),
            env=_cuda_env(output_dir, cache_label="single-stage-runtime-seed-spec"),
            command=[
                "python",
                "benchmarks/prepare_cuda_gpu_lowres_tests.py",
                "--write-runtime-seed-spec",
                "--jax-runtime-seed-source",
                _repo_path(config.jax_runtime_seed_source),
                "--output-dir",
                _repo_path(output_dir),
                "--single-stage-nphi",
                str(config.single_stage_nphi),
                "--single-stage-ntheta",
                str(config.single_stage_ntheta),
                "--single-stage-mpol",
                str(config.single_stage_mpol),
                "--single-stage-ntor",
                str(config.single_stage_ntor),
            ],
            outputs={"json": runtime_seed_spec_path},
            acceptance=[
                "JSON exists",
                "schema is simsopt.single_stage.jax_runtime_spec",
                "surface/quadrature shape matches the low-resolution rung",
            ],
        ),
        _command(
            name="single_stage_cuda_init_soft_penalty",
            purpose=(
                "Run the current single-stage JAX soft-penalty CUDA driver with "
                "zero optimizer iterations from the generated Stage 2 "
                "BoozerSurface seed."
            ),
            env=_cuda_env(output_dir, cache_label="single-stage-init-soft-penalty"),
            command=[
                "python",
                "examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py",
                _repo_path(stage2_boozersurface),
                str(DEFAULT_IOTA_TARGET),
                "--maxiter",
                "0",
                "--nphi",
                str(config.single_stage_nphi),
                "--ntheta",
                str(config.single_stage_ntheta),
                "--mpol",
                str(config.single_stage_mpol),
                "--ntor",
                str(config.single_stage_ntor),
                "--output-dir",
                _repo_path(single_stage_init_root),
                "--overwrite",
            ],
            outputs={
                "results_json": _repo_path(single_stage_init_root / "results.json"),
                "boozersurface_json": _repo_path(
                    single_stage_init_root / "single_stage_boozersurface.json"
                ),
                "biotsavart_json": _repo_path(
                    single_stage_init_root / "single_stage_biotsavart.json"
                ),
                "surface_json": _repo_path(
                    single_stage_init_root / "single_stage_surface.json"
                ),
                "log": _repo_path(single_stage_init_root / "single_stage.log"),
            },
            acceptance=[
                "run exits 0",
                "results JSON exists",
                "driver is single_stage_jax_soft_penalty",
                "restart BiotSavart/BoozerSurface/surface JSON outputs exist",
            ],
        ),
        _command(
            name="single_stage_cuda_outer_loop_soft_penalty",
            purpose=(
                "Run the current single-stage JAX soft-penalty CUDA driver with "
                "a reduced outer optimizer budget under strict transfer guard."
            ),
            env=_cuda_env(output_dir, cache_label="single-stage-outer-loop"),
            command=[
                "python",
                "examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py",
                _repo_path(stage2_boozersurface),
                str(DEFAULT_IOTA_TARGET),
                "--maxiter",
                str(config.single_stage_outer_maxiter),
                "--nphi",
                str(config.single_stage_nphi),
                "--ntheta",
                str(config.single_stage_ntheta),
                "--mpol",
                str(config.single_stage_mpol),
                "--ntor",
                str(config.single_stage_ntor),
                "--output-dir",
                _repo_path(single_stage_outer_root),
                "--overwrite",
            ],
            outputs={
                "results_json": _repo_path(single_stage_outer_root / "results.json"),
                "boozersurface_json": _repo_path(
                    single_stage_outer_root / "single_stage_boozersurface.json"
                ),
                "biotsavart_json": _repo_path(
                    single_stage_outer_root / "single_stage_biotsavart.json"
                ),
                "surface_json": _repo_path(
                    single_stage_outer_root / "single_stage_surface.json"
                ),
                "log": _repo_path(single_stage_outer_root / "single_stage.log"),
            },
            acceptance=[
                "run exits 0",
                "results JSON exists",
                "driver is single_stage_jax_soft_penalty",
                "optimizer nfev is positive",
                "finite objective and iota diagnostics are reported",
            ],
        ),
        _command(
            name="single_stage_cuda_restart_chain_smoke",
            purpose=(
                "Reload the generated single-stage BoozerSurface output once to "
                "prove the current JAX soft-penalty driver emits chainable "
                "restart artifacts."
            ),
            env=_cuda_env(output_dir, cache_label="single-stage-restart-chain"),
            command=[
                "python",
                "examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py",
                _repo_path(single_stage_init_root / "single_stage_boozersurface.json"),
                str(DEFAULT_IOTA_TARGET),
                "--maxiter",
                "0",
                "--nphi",
                str(config.single_stage_nphi),
                "--ntheta",
                str(config.single_stage_ntheta),
                "--mpol",
                str(config.single_stage_mpol),
                "--ntor",
                str(config.single_stage_ntor),
                "--output-dir",
                _repo_path(single_stage_restart_root),
                "--overwrite",
            ],
            outputs={
                "results_json": _repo_path(single_stage_restart_root / "results.json"),
                "boozersurface_json": _repo_path(
                    single_stage_restart_root / "single_stage_boozersurface.json"
                ),
            },
            acceptance=[
                "run exits 0",
                "results JSON exists",
                "driver is single_stage_jax_soft_penalty",
                "restart BoozerSurface JSON output exists",
            ],
        ),
    ]


def build_manifest(config: LowresCudaPrepConfig) -> dict[str, object]:
    if not config.boozer_surface_zip.is_file():
        raise FileNotFoundError(
            f"missing Boozer surface zip: {config.boozer_surface_zip}"
        )
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
            "jax_runtime_seed_source": _runtime_seed_source_inventory(
                config.jax_runtime_seed_source
            ),
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
            f"{key}={shlex.quote(str(value))}" for key, value in sorted(env.items())
        )
        lines.append(
            f"{env_prefix} {_render_shell_command([str(item) for item in command])}"
        )
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
    parser.add_argument("--boozer-surface-zip", default=None)
    parser.add_argument("--autoresearch-runs-dir", default=None)
    parser.add_argument(
        "--write-runtime-seed-spec",
        action="store_true",
        help=(
            "Only write the low-resolution single-stage runtime seed spec into "
            "--output-dir, then exit."
        ),
    )
    parser.add_argument(
        "--stage2-bs-path",
        default=str(DEFAULT_STAGE2_BS_PATH),
        help="Stage 2 biot_savart_opt.json seed with adjacent results.json.",
    )
    parser.add_argument(
        "--jax-runtime-seed-source",
        default=str(DEFAULT_STAGE2_RUNTIME_SPEC_PATH),
        help=(
            "Immutable single-stage JAX runtime seed spec to reproject to the "
            "requested low-resolution rung."
        ),
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
    runtime_seed_source = _resolve_path(args.jax_runtime_seed_source)
    if args.write_runtime_seed_spec:
        runtime_seed_spec_path = (
            output_dir / "single_stage_lowres_jax_runtime_spec.json"
        )
        write_runtime_seed_spec(
            source_path=runtime_seed_source,
            output_path=runtime_seed_spec_path,
            mpol=int(args.single_stage_mpol),
            ntor=int(args.single_stage_ntor),
            nphi=int(args.single_stage_nphi),
            ntheta=int(args.single_stage_ntheta),
        )
        print(f"runtime_seed_spec: {_repo_path(runtime_seed_spec_path)}")
        return

    if args.boozer_surface_zip is None:
        raise ValueError("--boozer-surface-zip is required outside runtime-spec mode")
    if args.autoresearch_runs_dir is None:
        raise ValueError(
            "--autoresearch-runs-dir is required outside runtime-spec mode"
        )

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
            None
            if args.warm_start_run_dir is None
            else _resolve_path(args.warm_start_run_dir)
        ),
        jax_runtime_seed_source=runtime_seed_source,
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
