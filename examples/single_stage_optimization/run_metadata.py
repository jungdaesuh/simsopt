from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import os
import subprocess
import sys
from typing import Any

from simsopt_jax.config import (
    get_backend_config,
    get_backend_mode,
    get_backend_policy,
    get_compilation_cache_dir,
    get_compilation_cache_policy,
    get_jax_platform,
    get_provenance_label,
    get_transfer_guard,
    is_backend_strict,
    query_active_gpu_memory_mb,
)


def _resolved_path(path: str | os.PathLike[str]) -> str:
    return str(Path(path).resolve())


def _git_sha(repo_root: str | os.PathLike[str]) -> str | None:
    completed = subprocess.run(
        ["git", "-C", _resolved_path(repo_root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return None
    sha = completed.stdout.strip()
    return sha or None


def build_runtime_provenance(
    *,
    title: str,
    repo_root: str | os.PathLike[str],
    script_path: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
    argv: list[str],
    jax_module,
    jaxlib_version: str,
) -> dict[str, Any]:
    gpu_memory_mb = query_active_gpu_memory_mb()
    backend_config = get_backend_config()
    backend_policy = get_backend_policy(backend_config.mode)
    provenance = {
        "title": title,
        "generated_at_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "repo_root": _resolved_path(repo_root),
        "repo_sha": _git_sha(repo_root),
        "script_path": _resolved_path(script_path),
        "output_root": _resolved_path(output_root),
        "argv": list(argv),
        "cwd": os.getcwd(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "jax": str(jax_module.__version__),
        "jaxlib": str(jaxlib_version),
        "backend": str(jax_module.default_backend()),
        "devices": [str(device) for device in jax_module.devices()],
        "x64_enabled": bool(jax_module.config.x64_enabled),
        "backend_mode": get_backend_mode(),
        "backend_label": get_provenance_label(),
        "runtime_dtype": backend_policy.runtime_dtype,
        "host_dtype": backend_policy.host_dtype,
        "tolerance_tier": backend_policy.tolerance_tier,
        "backend_strict": bool(is_backend_strict()),
        "transfer_guard": get_transfer_guard(),
        "jax_platform_request": get_jax_platform(),
        "compilation_cache_policy": get_compilation_cache_policy(),
        "compilation_cache_dir": get_compilation_cache_dir(),
        "persistent_cache_min_compile_time_secs": (
            backend_config.persistent_cache_min_compile_time_secs
        ),
        "jax_persistent_cache_min_compile_time_secs": (
            jax_module.config.jax_persistent_cache_min_compile_time_secs
        ),
        "environment": {
            "SIMSOPT_BACKEND_MODE": os.environ.get("SIMSOPT_BACKEND_MODE"),
            "SIMSOPT_BACKEND_STRICT": os.environ.get("SIMSOPT_BACKEND_STRICT"),
            "SIMSOPT_JAX_TRANSFER_GUARD": os.environ.get(
                "SIMSOPT_JAX_TRANSFER_GUARD"
            ),
            "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "SIMSOPT_JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": os.environ.get(
                "SIMSOPT_JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"
            ),
        },
    }
    if gpu_memory_mb is not None:
        provenance["gpu_memory_mb"] = float(gpu_memory_mb)
    return provenance


def build_artifact_manifest(
    output_root: str | os.PathLike[str],
    *,
    required_files: tuple[str, ...],
    optional_files: tuple[str, ...] = (),
    planned_files: tuple[str, ...] = (),
) -> dict[str, Any]:
    resolved_root = Path(output_root).resolve()
    planned = frozenset(planned_files)

    def _entry(filename: str) -> dict[str, Any]:
        path = resolved_root / filename
        return {
            "path": str(path),
            "exists": bool(path.exists() or filename in planned),
        }

    return {
        "output_root": str(resolved_root),
        "required": {filename: _entry(filename) for filename in required_files},
        "optional": {filename: _entry(filename) for filename in optional_files},
    }
