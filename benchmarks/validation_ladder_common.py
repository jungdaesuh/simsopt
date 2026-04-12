"""Shared helpers for the JAX GPU validation ladder probes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import re
import shlex
import subprocess
import sys
import threading
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from repo_bootstrap import (
    apply_cuda_toolchain_env as _apply_cuda_toolchain_env,
    bootstrap_local_simsopt as _bootstrap_local_simsopt,
)
from benchmarks import validation_ladder_contract as ladder_contract

SRC_ROOT = REPO_ROOT / "src"
_JAX_PLATFORM_ENV_VARS = (
    "JAX_PLATFORMS",
    "SIMSOPT_JAX_PLATFORM",
    "SIMSOPT_JAX_BACKEND",
)
_JAX_CUDA_MEMORY_ENV_VARS = ("XLA_PYTHON_CLIENT_PREALLOCATE",)
_JAX_COMPILATION_CACHE_ENV_VAR = "JAX_COMPILATION_CACHE_DIR"
_JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_ENV_VAR = (
    "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"
)
_JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_ENV_VAR = (
    "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES"
)
_JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES_ENV_VAR = (
    "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES"
)
_SIMSOPT_DISABLE_COMPILATION_CACHE_ENV_VAR = "SIMSOPT_DISABLE_JAX_COMPILATION_CACHE"
_SIMSOPT_COMPILATION_CACHE_POLICY_ENV_VAR = "SIMSOPT_JAX_COMPILATION_CACHE_POLICY"
_SIMSOPT_BACKEND_MODE_ENV_VAR = "SIMSOPT_BACKEND_MODE"
_SIMSOPT_BACKEND_STRICT_ENV_VAR = "SIMSOPT_BACKEND_STRICT"
_SIMSOPT_TRANSFER_GUARD_ENV_VAR = "SIMSOPT_JAX_TRANSFER_GUARD"
_TARGET_LANE_ACCEPTED_STEP_SYNC_ENV_VAR = "TARGET_LANE_ACCEPTED_STEP_SYNC"
_TRUTHY_ENV_VALUES = frozenset({"1", "true", "yes", "on"})
_BENCHMARK_COMPILATION_CACHE_ENV_DEFAULTS = {
    _JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_ENV_VAR: "0",
    _JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_ENV_VAR: "-1",
    _JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES_ENV_VAR: "all",
}
_REQUESTED_PLATFORM_RUNTIME_BACKENDS = {
    "cpu": frozenset({"cpu"}),
    "cuda": frozenset({"cuda", "gpu"}),
}
OPTIMIZER_DRIFT_TOLERANCES = ladder_contract.OPTIMIZER_DRIFT_TOLERANCES
_SHORT_RUN_SMOKE_MAXITER = ladder_contract.SHORT_RUN_SMOKE_MAXITER
TIER3_SINGLE_STAGE_OUTER_LOOP_RUNG = ladder_contract.TIER3_SINGLE_STAGE_OUTER_LOOP_RUNG
optimizer_drift_tolerances = ladder_contract.optimizer_drift_tolerances
resolve_probe_lane = ladder_contract.resolve_probe_lane
short_run_geometry_rel_tolerance = ladder_contract.short_run_geometry_rel_tolerance
short_run_stage2_final_objective_rel_tolerance = (
    ladder_contract.short_run_stage2_final_objective_rel_tolerance
)
evaluate_grouped_adjoint_memory_budget = (
    ladder_contract.evaluate_grouped_adjoint_memory_budget
)
evaluate_tier5_performance_budget = ladder_contract.evaluate_tier5_performance_budget
grouped_adjoint_memory_budget = ladder_contract.grouped_adjoint_memory_budget
single_stage_proof_contract = ladder_contract.single_stage_proof_contract
tier5_performance_budget = ladder_contract.tier5_performance_budget


def preparse_platform(argv: list[str]) -> str:
    """Read --platform before JAX import so scripts can pin the runtime device."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--platform", choices=("auto", "cpu", "cuda"), default="auto")
    args, _ = parser.parse_known_args(argv)
    return args.platform


def apply_requested_platform(platform: str) -> None:
    """Pin JAX to a specific platform before importing the package."""
    _apply_platform_env(os.environ, platform)


def apply_compilation_cache_policy(
    cache_dir: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Set a stable JAX compilation-cache policy before importing JAX."""
    disable_raw = os.environ.get(_SIMSOPT_DISABLE_COMPILATION_CACHE_ENV_VAR, "")
    if disable_raw.strip().lower() in _TRUTHY_ENV_VALUES:
        os.environ.pop(_JAX_COMPILATION_CACHE_ENV_VAR, None)
        for env_name in _BENCHMARK_COMPILATION_CACHE_ENV_DEFAULTS:
            os.environ.pop(env_name, None)
        os.environ[_SIMSOPT_COMPILATION_CACHE_POLICY_ENV_VAR] = "disabled"
        return current_compilation_cache_metadata()

    resolved_dir = cache_dir or os.environ.get(_JAX_COMPILATION_CACHE_ENV_VAR)
    if resolved_dir is None:
        os.environ.pop(_JAX_COMPILATION_CACHE_ENV_VAR, None)
        for env_name in _BENCHMARK_COMPILATION_CACHE_ENV_DEFAULTS:
            os.environ.pop(env_name, None)
        os.environ[_SIMSOPT_COMPILATION_CACHE_POLICY_ENV_VAR] = "disabled"
        return current_compilation_cache_metadata()
    resolved_path = Path(resolved_dir).expanduser()
    policy = "explicit"
    resolved_path.mkdir(parents=True, exist_ok=True)
    os.environ[_JAX_COMPILATION_CACHE_ENV_VAR] = str(resolved_path)
    for env_name, env_value in _BENCHMARK_COMPILATION_CACHE_ENV_DEFAULTS.items():
        os.environ.setdefault(env_name, env_value)
    os.environ[_SIMSOPT_COMPILATION_CACHE_POLICY_ENV_VAR] = policy
    metadata = current_compilation_cache_metadata()
    return metadata


def benchmark_compilation_cache_dir(label: str) -> Path:
    """Return the default persistent-cache directory for one benchmark label."""
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", label).strip("-")
    if not normalized:
        raise ValueError("benchmark compilation cache label must not be empty.")
    return REPO_ROOT / ".artifacts" / "jax_compilation_cache" / normalized


def apply_benchmark_compilation_cache_policy(
    label: str,
    *,
    requested_platform: str,
) -> dict[str, Any]:
    """Enable a stable cache dir for benchmark probes unless the run is CPU-only."""
    if requested_platform == "cpu":
        return apply_compilation_cache_policy()
    return apply_compilation_cache_policy(benchmark_compilation_cache_dir(label))


def repo_pythonpath_env(
    *,
    platform: str = "auto",
    disable_compilation_cache: bool = False,
    clear_backend_guardrails: bool = False,
) -> dict[str, str]:
    """Return an environment that resolves in-repo imports for subprocess probes."""
    env = dict(os.environ)
    _apply_platform_env(env, platform)
    env.pop(_SIMSOPT_DISABLE_COMPILATION_CACHE_ENV_VAR, None)
    env.pop(_SIMSOPT_COMPILATION_CACHE_POLICY_ENV_VAR, None)
    env.pop(_TARGET_LANE_ACCEPTED_STEP_SYNC_ENV_VAR, None)
    if clear_backend_guardrails:
        env.pop(_SIMSOPT_BACKEND_MODE_ENV_VAR, None)
        env.pop(_SIMSOPT_BACKEND_STRICT_ENV_VAR, None)
        env.pop(_SIMSOPT_TRANSFER_GUARD_ENV_VAR, None)
    if disable_compilation_cache:
        env.pop(_JAX_COMPILATION_CACHE_ENV_VAR, None)
        for env_name in _BENCHMARK_COMPILATION_CACHE_ENV_DEFAULTS:
            env.pop(env_name, None)
        env[_SIMSOPT_DISABLE_COMPILATION_CACHE_ENV_VAR] = "1"
        env[_SIMSOPT_COMPILATION_CACHE_POLICY_ENV_VAR] = "disabled"
    pythonpath_entries = [str(REPO_ROOT), str(SRC_ROOT)]
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    return env


def _apply_platform_env(env: dict[str, str], platform: str) -> None:
    """Apply or clear all JAX platform selectors used by this repo."""
    for key in _JAX_PLATFORM_ENV_VARS:
        env.pop(key, None)
    for key in _JAX_CUDA_MEMORY_ENV_VARS:
        env.pop(key, None)
    if platform == "auto":
        return
    env["JAX_PLATFORMS"] = platform
    env["SIMSOPT_JAX_PLATFORM"] = platform
    env["SIMSOPT_JAX_BACKEND"] = platform
    if platform == "cuda":
        env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        _apply_cuda_toolchain_env(env)


def _x64_enabled(jax_module) -> bool:
    """Return whether the active JAX runtime is operating in float64 mode."""
    config = getattr(jax_module, "config", None)
    config_x64 = getattr(config, "jax_enable_x64", None)
    if config_x64 is not None:
        return bool(config_x64)
    return bool(jax_module.numpy.zeros(1).dtype == jax_module.numpy.float64)


def require_x64_runtime(jax_module, *, context: str) -> None:
    """Raise when a target-lane runtime attempts to proceed without x64."""
    if _x64_enabled(jax_module):
        return
    raise RuntimeError(f"{context} requires jax_enable_x64=True before import/use.")


def require_requested_platform_runtime(
    jax_module,
    *,
    requested_platform: str,
    context: str,
) -> None:
    """Raise when the active JAX backend does not honor an explicit platform request."""
    if requested_platform == "auto":
        return
    actual_backend = str(jax_module.default_backend()).lower()
    expected_backends = _REQUESTED_PLATFORM_RUNTIME_BACKENDS[requested_platform]
    if actual_backend in expected_backends:
        return
    devices = [str(device) for device in jax_module.devices()]
    raise RuntimeError(
        f"{context} requested JAX platform '{requested_platform}' but initialized "
        f"backend '{actual_backend}' on devices {devices}."
    )


def current_compilation_cache_metadata() -> dict[str, Any]:
    """Return the active compilation-cache policy as provenance metadata."""
    disable_raw = os.environ.get(_SIMSOPT_DISABLE_COMPILATION_CACHE_ENV_VAR, "")
    disabled = disable_raw.strip().lower() in _TRUTHY_ENV_VALUES
    cache_dir = os.environ.get(_JAX_COMPILATION_CACHE_ENV_VAR)
    policy = os.environ.get(_SIMSOPT_COMPILATION_CACHE_POLICY_ENV_VAR)
    metadata = {
        "compilation_cache_enabled": bool(cache_dir) and not disabled,
        "compilation_cache_dir": None if disabled or cache_dir is None else cache_dir,
        "compilation_cache_policy": "disabled" if disabled else "unset",
    }
    if policy:
        metadata["compilation_cache_policy"] = policy
    elif metadata["compilation_cache_enabled"]:
        metadata["compilation_cache_policy"] = "env"
    return metadata


def current_backend_guardrail_metadata() -> dict[str, Any]:
    """Return the active backend-mode contract from the environment."""
    strict_raw = os.environ.get(_SIMSOPT_BACKEND_STRICT_ENV_VAR, "")
    return {
        "backend_mode": os.environ.get(_SIMSOPT_BACKEND_MODE_ENV_VAR),
        "backend_strict": strict_raw.strip().lower() in _TRUTHY_ENV_VALUES,
        "transfer_guard": os.environ.get(_SIMSOPT_TRANSFER_GUARD_ENV_VAR),
    }


def describe_compile_behavior(
    *,
    uses_subprocesses: bool,
) -> str:
    """Describe whether the current process expects true cold-compile timing."""
    metadata = current_compilation_cache_metadata()
    if metadata["compilation_cache_enabled"]:
        if uses_subprocesses:
            return (
                "persistent compilation cache enabled; subprocess launches may reuse "
                "cached compilations"
            )
        return (
            "persistent compilation cache enabled; cached executables may reduce "
            "first-call compile cost"
        )
    if uses_subprocesses:
        return (
            "persistent compilation cache disabled; subprocess timings include "
            "first-call compilation"
        )
    return "persistent compilation cache disabled; first-call compilation is included"


def bootstrap_local_simsopt() -> None:
    """Force imports to resolve against this repo's source tree."""
    _bootstrap_local_simsopt(SRC_ROOT)


def get_git_sha() -> str:
    """Return the exact repo SHA for provenance."""
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def relative_error(actual: float, reference: float) -> float:
    """Stable scalar relative error helper."""
    return float(abs(actual - reference) / (abs(reference) + 1e-30))


def max_relative_error(actual: np.ndarray, reference: np.ndarray) -> float:
    """Return the maximum elementwise relative error."""
    actual_arr = np.asarray(actual, dtype=float)
    reference_arr = np.asarray(reference, dtype=float)
    denom = np.maximum(np.abs(reference_arr), 1e-30)
    return float(np.max(np.abs(actual_arr - reference_arr) / denom))


def l2_relative_error(actual: np.ndarray, reference: np.ndarray) -> float:
    """Return the vector L2 relative error."""
    actual_arr = np.asarray(actual, dtype=float)
    reference_arr = np.asarray(reference, dtype=float)
    return float(
        np.linalg.norm(actual_arr - reference_arr)
        / (np.linalg.norm(reference_arr) + 1e-30)
    )


def max_pointwise_geometry_drift(
    actual_points: np.ndarray,
    reference_points: np.ndarray,
) -> tuple[float, float]:
    """Return max absolute and relative pointwise geometry drift."""
    actual_arr = np.asarray(actual_points, dtype=float).reshape(-1, 3)
    reference_arr = np.asarray(reference_points, dtype=float).reshape(-1, 3)
    pointwise = np.linalg.norm(actual_arr - reference_arr, axis=1)
    geometry_scale = max(float(np.max(np.linalg.norm(reference_arr, axis=1))), 1e-30)
    return float(np.max(pointwise)), float(np.max(pointwise) / geometry_scale)


def peak_rss_mb() -> float:
    """Return the process max RSS in MB using platform-correct units."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return float(rss) / (1024.0 * 1024.0)
    return float(rss) / 1024.0


def query_gpu_memory_mb() -> float | None:
    """Return coarse memory usage for the active CUDA device when available."""
    from simsopt.config import query_active_gpu_memory_mb

    return query_active_gpu_memory_mb()


def _distributed_provenance_fields(config) -> dict[str, Any]:
    return {
        "distributed_enabled": bool(config.enabled),
        "distributed_initialized": bool(config.initialized),
        "distributed_process_count": config.num_processes,
        "distributed_process_id": config.process_id,
        "distributed_coordinator_address": config.coordinator_address,
        "distributed_local_device_ids": (
            None if config.local_device_ids is None else list(config.local_device_ids)
        ),
    }


def maybe_initialize_distributed_runtime() -> dict[str, Any]:
    """Initialize distributed JAX when the repo runtime contract requests it."""
    from simsopt.config import maybe_initialize_distributed_jax

    config = maybe_initialize_distributed_jax()
    return _distributed_provenance_fields(config)


def _current_sharding_metadata() -> dict[str, Any]:
    from simsopt.config import (
        get_distributed_runtime_config,
        get_sharding_tuning,
    )

    sharding = get_sharding_tuning()
    distributed = get_distributed_runtime_config()
    return {
        "sharding_strategy": sharding.strategy,
        "sharding_active": bool(sharding.active),
        "sharding_axis_name": sharding.mesh_axis_name,
        "sharding_device_count": int(sharding.device_count),
        "sharding_local_device_count": int(sharding.local_device_count),
        "sharding_min_points_to_shard": int(sharding.min_points_to_shard),
        "sharding_min_pairwise_rows_to_shard": int(
            sharding.min_pairwise_rows_to_shard
        ),
        **_distributed_provenance_fields(distributed),
    }


def build_provenance(
    jax_module, jaxlib_module, *, title: str, extra: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Collect shared provenance fields for ladder outputs."""
    compilation_cache = current_compilation_cache_metadata()
    backend_guardrails = current_backend_guardrail_metadata()
    provenance = {
        "title": title,
        "repo_sha": get_git_sha(),
        "jax": jax_module.__version__,
        "jaxlib": jaxlib_module.__version__,
        "backend": jax_module.default_backend(),
        "devices": [str(device) for device in jax_module.devices()],
        "x64_enabled": _x64_enabled(jax_module),
        "peak_rss_mb": peak_rss_mb(),
        **backend_guardrails,
        **compilation_cache,
        **_current_sharding_metadata(),
    }
    gpu_memory_mb = query_gpu_memory_mb()
    if gpu_memory_mb is not None:
        provenance["gpu_memory_mb"] = gpu_memory_mb
    if extra:
        provenance.update(extra)
    return provenance


def print_provenance(provenance: dict[str, Any]) -> None:
    """Emit a stable human-readable provenance block."""
    print(f"\n{'=' * 70}")
    print(provenance["title"])
    print(f"{'=' * 70}")
    print(f"repo sha:     {provenance['repo_sha']}")
    print(f"jax:          {provenance['jax']}")
    print(f"jaxlib:       {provenance['jaxlib']}")
    print(f"backend:      {provenance['backend']}")
    print(f"devices:      {provenance['devices']}")
    print(f"x64 enabled:  {provenance['x64_enabled']}")
    if "lane" in provenance:
        print(f"lane:         {provenance['lane']}")
    if "fixture" in provenance:
        print(f"fixture:      {provenance['fixture']}")
    if "config_label" in provenance:
        print(f"config:       {provenance['config_label']}")
    if "platform_request" in provenance:
        print(f"platform arg: {provenance['platform_request']}")
    if "compile_behavior" in provenance:
        print(f"compile:      {provenance['compile_behavior']}")
    if provenance.get("backend_mode") is not None:
        print(f"mode:         {provenance['backend_mode']}")
    print(f"strict:       {provenance['backend_strict']}")
    if provenance.get("transfer_guard") is not None:
        print(f"guard:        {provenance['transfer_guard']}")
    print(f"cache policy: {provenance['compilation_cache_policy']}")
    if provenance.get("compilation_cache_dir"):
        print(f"cache dir:    {provenance['compilation_cache_dir']}")
    if "peak_rss_mb" in provenance:
        print(f"peak RSS:     {provenance['peak_rss_mb']:.1f} MB")
    if "gpu_memory_mb" in provenance:
        print(f"GPU memory:   {provenance['gpu_memory_mb']:.1f} MB")


def write_json(path: str | os.PathLike[str], payload: dict[str, Any]) -> None:
    """Write JSON payloads for probe outputs."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as outfile:
        json.dump(payload, outfile, indent=2)


def load_json(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load a JSON payload."""
    with Path(path).open("r", encoding="utf-8") as infile:
        return json.load(infile)


def _stream_subprocess_pipe(
    pipe,
    sink,
    chunks: list[str],
) -> None:
    """Mirror subprocess output to the parent stream while retaining it."""
    while True:
        chunk = pipe.read(4096)
        if not chunk:
            break
        chunks.append(chunk)
        sink.write(chunk)
        sink.flush()
    pipe.close()


def run_python_script(
    script_path: str | os.PathLike[str],
    args: list[str],
    *,
    env: dict[str, str] | None = None,
    cwd: str | os.PathLike[str] | None = None,
    bootstrap_repo: bool = False,
    stream_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run a Python helper script using the current interpreter."""
    if bootstrap_repo:
        command = [
            sys.executable,
            "-c",
            (
                "import runpy, sys; "
                "repo_root, script_path, *script_args = sys.argv[1:]; "
                "sys.path.insert(0, repo_root); "
                "sys.path.insert(0, repo_root + '/src'); "
                "from benchmarks.validation_ladder_common import bootstrap_local_simsopt; "
                "bootstrap_local_simsopt(); "
                "sys.argv = [script_path, *script_args]; "
                "runpy.run_path(script_path, run_name='__main__')"
            ),
            str(REPO_ROOT),
            str(script_path),
            *args,
        ]
    else:
        command = [sys.executable, str(script_path), *args]
    child_env = dict(env) if env is not None else dict(os.environ)
    child_env.setdefault("PYTHONUNBUFFERED", "1")
    if not stream_output:
        result = subprocess.run(
            command,
            cwd=str(cwd or REPO_ROOT),
            env=child_env,
            capture_output=True,
            text=True,
        )
    else:
        process = subprocess.Popen(
            command,
            cwd=str(cwd or REPO_ROOT),
            env=child_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if process.stdout is None or process.stderr is None:
            raise RuntimeError("Failed to capture subprocess stdout/stderr pipes.")
        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []
        stdout_thread = threading.Thread(
            target=_stream_subprocess_pipe,
            args=(process.stdout, sys.stdout, stdout_chunks),
        )
        stderr_thread = threading.Thread(
            target=_stream_subprocess_pipe,
            args=(process.stderr, sys.stderr, stderr_chunks),
        )
        stdout_thread.start()
        stderr_thread.start()
        returncode = process.wait()
        stdout_thread.join()
        stderr_thread.join()
        result = subprocess.CompletedProcess(
            command,
            returncode,
            stdout="".join(stdout_chunks),
            stderr="".join(stderr_chunks),
        )
    if result.returncode != 0:
        formatted_command = " ".join(shlex.quote(part) for part in command)
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        details: list[str] = []
        if stdout:
            details.append(f"stdout:\n{stdout}")
        if stderr:
            details.append(f"stderr:\n{stderr}")
        detail_block = "\n\n".join(details) if details else "no stdout/stderr captured"
        raise RuntimeError(
            f"Subprocess failed with exit code {result.returncode}: {formatted_command}\n\n{detail_block}"
        )
    return result


def find_single_file(root: str | os.PathLike[str], pattern: str) -> Path:
    """Resolve a single probe artifact under a temporary output root."""
    matches = list(Path(root).rglob(pattern))
    if len(matches) != 1:
        match_display = ", ".join(str(match) for match in matches) or "<none>"
        raise RuntimeError(
            f"Expected exactly one {pattern!r} under {root}, found {match_display}"
        )
    return matches[0]
