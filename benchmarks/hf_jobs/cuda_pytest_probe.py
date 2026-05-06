#!/usr/bin/env python3
"""Emit CUDA proof payloads for hardware-gated pytest lanes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import threading
import time


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from benchmarks.validation_ladder_common import (  # noqa: E402
    apply_benchmark_compilation_cache_policy,
    apply_requested_platform,
    bootstrap_local_simsopt,
    build_provenance,
    describe_compile_behavior,
    maybe_initialize_distributed_runtime,
    parity_ladder_tolerances,
    peak_rss_mb,
    query_gpu_memory_mb,
    require_requested_platform_runtime,
    require_x64_runtime,
    write_json,
)


PROBE_LANES = {
    "boozer_well_conditioned_adjoint": "exact_well_conditioned_adjoint",
    "reduction_cancellation_stress": "reduction_cpu_gpu",
}
CUDA_BACKENDS = frozenset({"cuda", "gpu"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a CUDA pytest proof lane and write a JSON payload."
    )
    parser.add_argument("--name", choices=tuple(PROBE_LANES), required=True)
    parser.add_argument("--platform", choices=("cuda",), default="cuda")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.pytest_args[:1] == ["--"]:
        args.pytest_args = args.pytest_args[1:]
    if not args.pytest_args:
        parser.error("pytest arguments are required after --")
    return args


def _sample_gpu_memory(stop_event: threading.Event, samples_mb: list[float]) -> None:
    while not stop_event.is_set():
        gpu_memory_mb = query_gpu_memory_mb()
        if gpu_memory_mb is not None:
            samples_mb.append(float(gpu_memory_mb))
        stop_event.wait(0.25)


def _runtime_contract_failures(provenance: dict[str, object]) -> list[str]:
    failures: list[str] = []
    backend = str(provenance["backend"]).lower()
    if backend not in CUDA_BACKENDS:
        failures.append(f"expected CUDA backend, got {backend!r}")
    if provenance["x64_enabled"] is not True:
        failures.append("expected x64_enabled=True")
    if not provenance.get("nvidia_smi_gpus"):
        failures.append("missing nvidia_smi_gpus")
    if not provenance.get("cuda_driver_version"):
        failures.append("missing cuda_driver_version")
    if not provenance.get("cuda_runtime_version"):
        failures.append("missing cuda_runtime_version")
    if "peak_rss_mb" not in provenance:
        failures.append("missing peak_rss_mb")
    if "peak_gpu_memory_mb" not in provenance:
        failures.append("missing peak_gpu_memory_mb")
    return failures


def main() -> None:
    args = parse_args()
    apply_requested_platform(str(args.platform))
    apply_benchmark_compilation_cache_policy(
        f"cuda_pytest_probe_{args.name}",
        requested_platform=str(args.platform),
    )
    bootstrap_local_simsopt()

    import jax
    import jaxlib
    import pytest

    maybe_initialize_distributed_runtime()
    jax.config.update("jax_enable_x64", True)
    require_x64_runtime(jax, context=f"CUDA pytest proof {args.name}")
    require_requested_platform_runtime(
        jax,
        requested_platform=str(args.platform),
        context=f"CUDA pytest proof {args.name}",
    )

    lane = PROBE_LANES[args.name]
    output_json = Path(args.output_json)
    command_argv = [sys.executable, "-m", "pytest", *args.pytest_args]
    gpu_memory_samples_mb: list[float] = []
    initial_gpu_memory_mb = query_gpu_memory_mb()
    if initial_gpu_memory_mb is not None:
        gpu_memory_samples_mb.append(float(initial_gpu_memory_mb))
    stop_event = threading.Event()
    sampler = threading.Thread(
        target=_sample_gpu_memory,
        args=(stop_event, gpu_memory_samples_mb),
        daemon=True,
    )

    start = time.perf_counter()
    sampler.start()
    pytest_exit_code = int(pytest.main(args.pytest_args))
    stop_event.set()
    sampler.join()
    final_gpu_memory_mb = query_gpu_memory_mb()
    if final_gpu_memory_mb is not None:
        gpu_memory_samples_mb.append(float(final_gpu_memory_mb))
    elapsed_s = time.perf_counter() - start

    provenance = build_provenance(
        jax,
        jaxlib,
        title=f"CUDA pytest proof: {args.name}",
        extra={
            "lane": lane,
            "fixture": args.name,
            "pytest_args": list(args.pytest_args),
            "command_argv": command_argv,
            "compile_behavior": describe_compile_behavior(uses_subprocesses=False),
        },
    )
    provenance["peak_rss_mb"] = max(float(provenance["peak_rss_mb"]), peak_rss_mb())
    if gpu_memory_samples_mb:
        provenance["peak_gpu_memory_mb"] = max(gpu_memory_samples_mb)

    runtime_failures = _runtime_contract_failures(provenance)
    failures = (
        []
        if pytest_exit_code == 0
        else [f"pytest exit code {pytest_exit_code}"]
    )
    failures.extend(runtime_failures)
    payload = {
        "passed": not failures,
        "elapsed_s": elapsed_s,
        "failures": failures,
        "provenance": provenance,
        "bundle_provenance": {
            "runner": "benchmarks/hf_jobs/cuda_pytest_probe.py",
            "fake": False,
            "probe_name": args.name,
            "default_backend": provenance["backend"],
            "devices": provenance["devices"],
            "xla_flags": provenance["xla_flags"],
            "cuda_force_ptx_jit": provenance["cuda_force_ptx_jit"],
            "cuda_disable_ptx_jit": provenance["cuda_disable_ptx_jit"],
        },
        "proof_parity": {
            "lane": lane,
            **parity_ladder_tolerances(lane),
        },
    }
    write_json(output_json, payload)
    print(json.dumps(payload, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
