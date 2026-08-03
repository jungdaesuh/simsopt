"""Same-process prepared L-BFGS warm-soak evidence harness."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

import numpy as np
from simsopt_jax.geo.optimizers.private import _common

from benchmarks import custom_quasi_newton_runtime as runtime
from benchmarks.fixtures.custom_quasi_newton import fixture
from benchmarks.process_gpu_monitor import (
    GpuMemoryUnavailableReason,
    ProcessGpuMemoryMonitor,
    cpu_gpu_memory_unavailable,
    process_gpu_memory_artifact,
)

# Tolerate Linux allocator arena growth and page rounding without masking a leak.
RSS_PLATEAU_SLACK_KIB = 2048
VRAM_PLATEAU_SLACK_MIB = 2
WARM_SECONDS_RATIO_LIMIT = 1.2
_DISCARDED_RUNS = 1
_FUSED_CACHE_KEY = "lbfgsb-fused-stepwise"


@runtime_checkable
class _CacheSizedJittedCallable(Protocol):
    """The PjitFunction cache-count surface required by this evidence harness."""

    def _cache_size(self) -> int: ...


@dataclass(frozen=True)
class VramRecord:
    """Per-run memory evidence for this process, or an explicit unavailable state."""

    availability: Literal["available", "unavailable"]
    reason: GpuMemoryUnavailableReason | None
    vram_mib: int | None


@dataclass(frozen=True)
class WarmRunRecord:
    """One synchronized warm solve and its immediately sampled process state."""

    run_index: int
    retained: bool
    warm_seconds: float
    rss_kib: int
    vram: VramRecord
    executable_count: int


@dataclass(frozen=True)
class PlateauVerdict:
    """Equal-half retained-run comparisons and their fail-closed overall verdict."""

    retained_run_count: int
    compared_runs_per_half: int
    executable_count_identical: bool
    executable_count_positive: bool
    rss_plateau: bool
    rss_slack_kib: int
    vram_applicable: bool
    vram_plateau: bool | None
    vram_slack_mib: int
    warm_seconds_plateau: bool
    warm_seconds_ratio_limit: float
    first_half_max_rss_kib: int
    last_half_max_rss_kib: int
    first_half_max_vram_mib: int | None
    last_half_max_vram_mib: int | None
    first_half_median_warm_seconds: float
    last_half_median_warm_seconds: float
    plateau: bool


@dataclass(frozen=True)
class DeviceIdentity:
    backend: str
    jax_device: str


@dataclass(frozen=True)
class SoakConfig:
    device: Literal["cpu", "gpu"]
    fixture: Literal["coil47"]
    runs: int
    discarded_runs: int
    retained_runs: int
    maxiter: int
    maxcor: int
    provider: Literal["custom"] = "custom"
    method: Literal["lbfgs"] = "lbfgs"
    run_mode: Literal["fused_stepwise"] = "fused_stepwise"


@dataclass(frozen=True)
class SoakArtifact:
    schema_version: Literal[1]
    git_commit: str
    git_clean: bool
    device_identity: DeviceIdentity
    config: SoakConfig
    runs: tuple[WarmRunRecord, ...]
    plateau_verdict: PlateauVerdict


def _executable_count(jitted_callable: object) -> int:
    """Read the exact PjitFunction executable-cache count or fail closed."""

    if not isinstance(jitted_callable, _CacheSizedJittedCallable):
        raise TypeError(
            "prepared fused L-BFGS jitted callable does not expose "
            "PjitFunction._cache_size(); executable-count evidence is unavailable"
        )
    return jitted_callable._cache_size()


def _prepared_fused_jitted_callable(prepared: runtime._PreparedCustom) -> object:
    """Recover the fused JIT retained by the runner's objective-owned cache."""

    solver_cache = getattr(
        prepared.objective,
        _common._PRIVATE_SOLVER_CACHE_ATTR,
        None,
    )
    if not isinstance(solver_cache, Mapping):
        raise TypeError(
            "prepared fused L-BFGS objective has no readable private solver cache"
        )
    matches = tuple(
        cached_callable
        for cache_key, cached_callable in solver_cache.items()
        if isinstance(cache_key, tuple)
        and bool(cache_key)
        and cache_key[0] == _FUSED_CACHE_KEY
    )
    if len(matches) != 1:
        raise RuntimeError(
            "prepared fused L-BFGS objective must retain exactly one fused jitted "
            f"callable, found {len(matches)}"
        )
    _executable_count(matches[0])
    return matches[0]


def _vram_record(
    *,
    device: Literal["cpu", "gpu"],
    gpu_uuid: str | None,
    run_solve: Callable[[], None],
) -> tuple[float, VramRecord]:
    """Time one synchronized solve while measuring this PID's GPU memory."""

    provider_pid = os.getpid()
    if device == "cpu":
        started = time.perf_counter()
        run_solve()
        warm_seconds = time.perf_counter() - started
        measurement = cpu_gpu_memory_unavailable(provider_pid=provider_pid)
    else:
        if gpu_uuid is None:
            raise RuntimeError("GPU warm soak has no authenticated GPU UUID")
        monitor = ProcessGpuMemoryMonitor(
            gpu_uuid=gpu_uuid,
            provider_pid=provider_pid,
        )
        monitor.start()
        started = time.perf_counter()
        run_solve()
        warm_seconds = time.perf_counter() - started
        measurement = monitor.finish()
    artifact = process_gpu_memory_artifact(measurement)
    return warm_seconds, VramRecord(
        availability=artifact.availability,
        reason=artifact.unavailable_reason,
        vram_mib=artifact.peak_used_memory_mib,
    )


def _validate_device_binding(
    device: Literal["cpu", "gpu"],
    *,
    backend: str,
    platform: str,
) -> None:
    """Require the requested evidence lane to match the initialized JAX device."""

    if device == "cpu" and (backend != "cpu" or platform != "cpu"):
        raise RuntimeError(
            "requested CPU warm soak requires JAX backend/platform 'cpu', "
            f"got backend={backend!r}, platform={platform!r}"
        )
    if device == "gpu" and (
        backend not in {"cuda", "gpu", "rocm"}
        or platform not in {"cuda", "gpu", "rocm"}
    ):
        raise RuntimeError(
            "requested GPU warm soak requires a JAX GPU backend/platform, "
            f"got backend={backend!r}, platform={platform!r}"
        )


def compute_plateau_verdict(
    retained_runs: Sequence[WarmRunRecord],
    *,
    vram_applicable: bool,
) -> PlateauVerdict:
    """Compare equal first/last retained halves; ignore one middle run if odd."""

    if len(retained_runs) < 2:
        raise ValueError("plateau verdict requires at least two retained runs")
    half_size = len(retained_runs) // 2
    first_half = retained_runs[:half_size]
    last_half = retained_runs[-half_size:]

    executable_count_identical = (
        len({record.executable_count for record in retained_runs}) == 1
    )
    executable_count_positive = all(
        record.executable_count > 0 for record in retained_runs
    )
    first_half_max_rss_kib = max(record.rss_kib for record in first_half)
    last_half_max_rss_kib = max(record.rss_kib for record in last_half)
    rss_plateau = (
        last_half_max_rss_kib <= first_half_max_rss_kib + RSS_PLATEAU_SLACK_KIB
    )

    first_half_median_warm_seconds = statistics.median(
        record.warm_seconds for record in first_half
    )
    last_half_median_warm_seconds = statistics.median(
        record.warm_seconds for record in last_half
    )
    warm_seconds_plateau = (
        last_half_median_warm_seconds
        <= WARM_SECONDS_RATIO_LIMIT * first_half_median_warm_seconds
    )

    first_half_max_vram_mib: int | None = None
    last_half_max_vram_mib: int | None = None
    vram_plateau: bool | None = None
    if vram_applicable:
        vram_available = all(
            record.vram.availability == "available" and record.vram.vram_mib is not None
            for record in retained_runs
        )
        if vram_available:
            first_half_max_vram_mib = max(
                record.vram.vram_mib
                for record in first_half
                if record.vram.vram_mib is not None
            )
            last_half_max_vram_mib = max(
                record.vram.vram_mib
                for record in last_half
                if record.vram.vram_mib is not None
            )
            vram_plateau = (
                last_half_max_vram_mib
                <= first_half_max_vram_mib + VRAM_PLATEAU_SLACK_MIB
            )
        else:
            vram_plateau = False

    applicable_checks = [
        executable_count_identical,
        executable_count_positive,
        rss_plateau,
        warm_seconds_plateau,
    ]
    if vram_applicable:
        applicable_checks.append(vram_plateau is True)
    return PlateauVerdict(
        retained_run_count=len(retained_runs),
        compared_runs_per_half=half_size,
        executable_count_identical=executable_count_identical,
        executable_count_positive=executable_count_positive,
        rss_plateau=rss_plateau,
        rss_slack_kib=RSS_PLATEAU_SLACK_KIB,
        vram_applicable=vram_applicable,
        vram_plateau=vram_plateau,
        vram_slack_mib=VRAM_PLATEAU_SLACK_MIB,
        warm_seconds_plateau=warm_seconds_plateau,
        warm_seconds_ratio_limit=WARM_SECONDS_RATIO_LIMIT,
        first_half_max_rss_kib=first_half_max_rss_kib,
        last_half_max_rss_kib=last_half_max_rss_kib,
        first_half_max_vram_mib=first_half_max_vram_mib,
        last_half_max_vram_mib=last_half_max_vram_mib,
        first_half_median_warm_seconds=first_half_median_warm_seconds,
        last_half_median_warm_seconds=last_half_median_warm_seconds,
        plateau=all(applicable_checks),
    )


def run_soak(
    *,
    device: Literal["cpu", "gpu"],
    fixture_name: Literal["coil47"],
    runs: int,
    output_json: Path,
    maxiter: int,
    maxcor: int,
) -> SoakArtifact:
    """Prepare once, execute repeated fused warm solves, and persist evidence."""

    if runs < 3:
        raise ValueError("runs must be at least 3 (one discarded, two retained)")
    if maxiter <= 0:
        raise ValueError("maxiter must be positive")
    if maxcor <= 0:
        raise ValueError("maxcor must be positive")

    runtime_device_identity = runtime._device_identity(device)
    _validate_device_binding(
        device,
        backend=runtime_device_identity.backend,
        platform=runtime_device_identity.platform,
    )

    fixture_case = fixture(fixture_name)
    initial_parameters = np.asarray(fixture_case.initial, dtype=np.float64)
    prepared = runtime._prepare_custom(
        fixture_case,
        initial_parameters,
        maxcor=maxcor,
        run_mode="fused_stepwise",
    )
    fused_jitted_callable = _prepared_fused_jitted_callable(prepared)
    prepared = replace(
        prepared,
        program=replace(prepared.program, fused_solve=fused_jitted_callable),
    )

    def run_solve() -> None:
        runtime._run_custom(
            fixture_case,
            initial_parameters,
            maxiter=maxiter,
            maxcor=maxcor,
            method="lbfgs",
            run_mode="fused_stepwise",
            prepared=prepared,
        )

    run_records: list[WarmRunRecord] = []
    for run_index in range(runs):
        warm_seconds, vram = _vram_record(
            device=device,
            gpu_uuid=runtime_device_identity.gpu_uuid,
            run_solve=run_solve,
        )
        run_records.append(
            WarmRunRecord(
                run_index=run_index,
                retained=run_index >= _DISCARDED_RUNS,
                warm_seconds=warm_seconds,
                rss_kib=runtime._current_rss_kib(),
                vram=vram,
                executable_count=_executable_count(fused_jitted_callable),
            )
        )

    retained_records = tuple(record for record in run_records if record.retained)
    plateau_verdict = compute_plateau_verdict(
        retained_records,
        vram_applicable=device == "gpu",
    )
    git_commit, git_clean = runtime._checkout_provenance()
    artifact = SoakArtifact(
        schema_version=1,
        git_commit=git_commit,
        git_clean=git_clean,
        device_identity=DeviceIdentity(
            backend=runtime_device_identity.backend,
            jax_device=runtime_device_identity.jax_device,
        ),
        config=SoakConfig(
            device=device,
            fixture=fixture_name,
            runs=runs,
            discarded_runs=_DISCARDED_RUNS,
            retained_runs=len(retained_records),
            maxiter=maxiter,
            maxcor=maxcor,
        ),
        runs=tuple(run_records),
        plateau_verdict=plateau_verdict,
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(asdict(artifact), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return artifact


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--fixture", choices=("coil47",), required=True)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--maxiter", type=int, default=20)
    parser.add_argument("--maxcor", type=int, default=10)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    artifact = run_soak(
        device=args.device,
        fixture_name=args.fixture,
        runs=args.runs,
        output_json=args.output_json,
        maxiter=args.maxiter,
        maxcor=args.maxcor,
    )
    return 0 if artifact.plateau_verdict.plateau else 1


if __name__ == "__main__":
    raise SystemExit(main())
