"""Lean, reproducible custom quasi-Newton runtime measurements.

This runner qualifies solver mechanics on deterministic fixtures.  ``coil47``
and ``boozer`` are source-owned VMEC-free physics contracts with matched
SIMSOPT-native and JAX objective callbacks.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import TracebackType
from typing import Literal, NoReturn, Protocol, Self, cast

import jax
import jax.numpy as jnp
import numpy as np
import optax
import scipy
from scipy import optimize
from simsopt_jax.backend.runtime import resolve_jax_execution_profile
from simsopt_jax.geo.optimizer_host_lbfgs import (
    line_search_value_and_grad_more_thuente_host,
    minimize_bfgs_host_core,
)
from simsopt_jax.geo.optimizers.private import (
    PreparedLBFGS,
    _result_converters,
    prepare_lbfgs_private,
)
from simsopt_jax.geo.optimizers.private._bfgs import (
    _bfgs_inverse_hessian_update,
    _bfgs_memory_contract,
    _compiled_step_memory_analysis,
    _minimize_bfgs_private,
)
from simsopt_jax.geo.optimizers.private._lbfgs import _minimize_lbfgs_private
from simsopt_jax.runtime.host_boundary import host_transfer_audit
from simsopt_jax.solve.endpoint_certificate import (
    OptimizationEndpointCertificate,
    _stopping_reason,
    certify_optimization_endpoint,
    status_convention_for,
)

from benchmarks.boozer_trial_diagnostic import (
    TrialProvider,
    run_boozer_host_diagnostic,
    validate_boozer_trial_trace,
)
from benchmarks.fixtures.custom_quasi_newton import (
    AcceptedIncumbentInnerState,
    Fixture,
    NativeValueAndGrad,
    ScientificEndpointEvidence,
    fixture,
    fixture_accepted_incumbent,
    fixture_method,
)
from benchmarks.process_gpu_monitor import (
    ProcessGpuMemoryMeasurement,
    ProcessGpuMemoryMonitor,
    cpu_gpu_memory_unavailable,
    process_gpu_memory_artifact,
)

Provider = Literal["native", "custom", "optax"]
Method = Literal["bfgs", "lbfgs"]

_RUNNER_SCHEMA_VERSION = 9


class _OptaxValueAndGrad(Protocol):
    def __call__(
        self,
        params: jax.Array,
        *,
        state: optax.OptState,
    ) -> tuple[jax.Array, jax.Array]: ...


class _OptaxStep(Protocol):
    def __call__(
        self,
        params: jax.Array,
        state: optax.OptState,
    ) -> tuple[jax.Array, optax.OptState, jax.Array, jax.Array]: ...


class _OptaxFinalValueAndGrad(Protocol):
    def __call__(self, params: jax.Array) -> tuple[jax.Array, jax.Array]: ...


class _OptaxObjective(Protocol):
    def __call__(self, params: jax.Array) -> jax.Array: ...


@dataclass(frozen=True)
class _PreparedOptax:
    """Compiled Optax programs reused by cold and warm solver runs."""

    solver: optax.GradientTransformationExtraArgs
    step: _OptaxStep
    final_value_and_grad: _OptaxFinalValueAndGrad
    initial_value: jax.Array
    initial_gradient: jax.Array
    objective: _OptaxObjective
    initial_parameter_bits: tuple[int, ...]
    parameter_shape: tuple[int, ...]
    maxcor: int


@dataclass(frozen=True)
class _PreparedCustom:
    """Compiled custom L-BFGS programs reused by cold and warm runs."""

    program: PreparedLBFGS
    objective: _OptaxObjective
    initial_parameter_bits: tuple[int, ...]
    parameter_shape: tuple[int, ...]
    maxcor: int


_PROVIDER_CHILD_TIMEOUT_SECONDS = 120
_BOOZER_PROVIDER_CHILD_TIMEOUT_SECONDS = 1800
_PROVIDER_CHILD_RSS_LIMIT_KIB = 8 * 1024 * 1024
_PROVIDER_CHILD_TERM_GRACE_SECONDS = 5
_PROVIDER_CHILD_POLL_SECONDS = 0.1
_GPU_MEMORY_ARTIFACT_NAME = "gpu_memory.json"
_BOOZER_TRIAL_TRACE_ARTIFACT_NAME = "boozer_trial_trace.json"
_SOLVER_FTOL = 0.0
_SOLVER_GTOL = 1.0e-10
_SOLVER_MAXLS = 20
_FIXTURE_TOLERANCES = {
    "objective_abs": 1.0e-12,
    "objective_rel": 1.0e-10,
    "parameters_abs": 1.0e-10,
    "gradient_inf_norm_abs": 1.0e-8,
}
_RUNTIME_ENVIRONMENT_KEYS = (
    "JAX_PLATFORMS",
    "JAX_ENABLE_X64",
    "SIMSOPT_BACKEND_MODE",
    "SIMSOPT_BACKEND_STRICT",
    "SIMSOPT_PRECISION",
    "XLA_PYTHON_CLIENT_PREALLOCATE",
    "XLA_PYTHON_CLIENT_ALLOCATOR",
    "XLA_PYTHON_CLIENT_MEM_FRACTION",
)


def _checkout_provenance() -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    clean = not bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return commit, clean


def _runtime_environment_payload() -> dict[str, str | None]:
    return {key: os.environ.get(key) for key in _RUNTIME_ENVIRONMENT_KEYS}


def _validate_intent_environment(device: str, intent: str) -> str:
    """Require the CLI intent to select the matching runtime profile."""

    profile = resolve_jax_execution_profile(device, intent)
    configured_mode = os.environ.get("SIMSOPT_BACKEND_MODE")
    if configured_mode != profile.mode:
        raise RuntimeError(
            "requested execution profile does not match "
            f"SIMSOPT_BACKEND_MODE={configured_mode!r}; expected {profile.mode!r}"
        )
    return profile.mode


@dataclass(frozen=True)
class TransferMeasurement:
    phase: str
    calls: int
    leaves: int
    bytes: int


@dataclass(frozen=True)
class PhaseRSSMeasurement:
    phase: str
    start_rss_kib: int
    peak_rss_kib: int
    end_rss_kib: int
    sample_count: int
    scope: str


@dataclass(frozen=True)
class DeviceIdentity:
    requested_device: str
    backend: str
    platform: str
    jax_device: str
    device_kind: str
    device_id: int | None
    process_index: int | None
    gpu_uuid: str | None
    gpu_model: str | None
    compute_capability: str | None
    total_memory_bytes: int | None
    driver_version: str | None
    cuda_version: str | None
    visible_devices: str | None
    hostname: str
    scheduler_job_id: str | None


@dataclass(frozen=True)
class WorkMeasurement:
    accepted_iterations: int
    objective_evaluations: int | None
    transfer_calls: int
    transfer_leaves: int
    transfer_bytes: int
    advance_observations: int | None


@dataclass(frozen=True)
class _NvidiaSmiIdentity:
    index: int
    uuid: str
    model: str
    total_memory_bytes: int
    driver_version: str


@dataclass(frozen=True)
class Measurement:
    case: str
    provider: Provider
    method: Method
    device: str
    intent: str
    solver_route: str
    device_identity: DeviceIdentity
    dimension: int
    maxiter: int
    maxcor: int
    fixture_build_seconds: float
    fixture_build_peak_rss_kib: int
    solver_start_rss_kib: int
    solver_peak_rss_kib: int
    solver_peak_rss_delta_kib: int
    initial_parameters: tuple[float, ...]
    final_parameters: tuple[float, ...]
    initial_objective: float
    initial_gradient_inf_norm: float
    final_objective: float
    final_gradient_inf_norm: float
    iterations: int
    evaluations: int | None
    status: int | None
    success: bool
    stopping_reason: str
    preparation_seconds: float
    first_execution_seconds: float
    cold_seconds: float
    warm_seconds: float
    peak_rss_kib: int
    peak_rss_scope: str
    ru_maxrss_kib: int
    peak_vram_mib: float | None
    process_pid: int
    certificate: str
    warm_transfer_audit: tuple[TransferMeasurement, ...]
    work_counters: WorkMeasurement
    inner_success: bool
    parameters_finite: bool
    observables_finite: bool
    constraint_norm: float | None
    endpoint_certificate: OptimizationEndpointCertificate
    scientific_observables: dict[str, float]
    scientific_certification_seconds: float
    diagnostic_artifacts: dict[str, str | None]
    phase_rss: tuple[PhaseRSSMeasurement, ...]
    fixture_metadata: tuple[tuple[str, object], ...]
    fixture_contract: dict[str, object]
    algorithm_memory_contract: dict[str, int | bool] | None


def _solver_route(
    provider: Provider,
    method: Method,
    *,
    intent: str = "parity",
    accepted_incumbent: bool = False,
) -> str:
    if provider == "native":
        return "scipy_bfgs" if method == "bfgs" else "scipy_lbfgsb"
    if provider == "optax":
        return "optax_lbfgs"
    if method == "bfgs":
        # The route names the emitting driver so persisted rows stay
        # self-describing: the host core under accepted-incumbent
        # continuation and the private on-device solver use different
        # status vocabularies.
        return (
            "custom_bfgs_host_incumbent"
            if accepted_incumbent
            else "custom_bfgs_private"
        )
    return "fused_stepwise" if intent == "fast" else "stepwise"


def _parse_nvidia_smi_identity_rows(output: str) -> tuple[_NvidiaSmiIdentity, ...]:
    """Parse the fixed CSV projection used to authenticate a CUDA device."""

    records: list[_NvidiaSmiIdentity] = []
    for line in output.splitlines():
        fields = tuple(field.strip() for field in line.split(","))
        if len(fields) != 5:
            raise ValueError(f"invalid nvidia-smi identity row: {line!r}")
        index, uuid, model, memory_mib, driver = fields
        records.append(
            _NvidiaSmiIdentity(
                index=int(index),
                uuid=uuid,
                model=model,
                total_memory_bytes=int(memory_mib) * 1024 * 1024,
                driver_version=driver,
            )
        )
    return tuple(records)


def _selected_nvidia_smi_identity(device_id: int | None) -> _NvidiaSmiIdentity:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    records = _parse_nvidia_smi_identity_rows(completed.stdout)
    selector = os.environ.get("CUDA_VISIBLE_DEVICES")
    if selector is not None:
        selectors = tuple(part.strip() for part in selector.split(",") if part.strip())
        if len(selectors) != 1:
            raise RuntimeError(
                "strict GPU qualification requires exactly one visible device"
            )
        selected = selectors[0]
        if selected.startswith("GPU-"):
            matches = tuple(record for record in records if record.uuid == selected)
        else:
            physical_index = int(selected)
            matches = tuple(
                record for record in records if record.index == physical_index
            )
    else:
        matches = tuple(record for record in records if record.index == device_id)
    if len(matches) != 1:
        raise RuntimeError("could not bind the JAX CUDA device to one nvidia-smi UUID")
    return matches[0]


def _device_identity(requested_device: str) -> DeviceIdentity:
    """Bind one measurement process to its exact host and JAX device."""

    devices = tuple(jax.devices())
    device = devices[0]
    device_id_value = getattr(device, "id", None)
    process_index_value = getattr(device, "process_index", None)
    client = getattr(device, "client", None)
    cuda_version_value = getattr(client, "platform_version", None)
    device_id = int(device_id_value) if isinstance(device_id_value, int) else None
    gpu_identity = (
        _selected_nvidia_smi_identity(device_id) if requested_device == "gpu" else None
    )
    device_compute_capability = (
        getattr(device, "compute_capability", None)
        if requested_device == "gpu"
        else None
    )
    return DeviceIdentity(
        requested_device=requested_device,
        backend=str(jax.default_backend()),
        platform=str(getattr(device, "platform", "")),
        jax_device=str(device),
        device_kind=str(getattr(device, "device_kind", "")),
        device_id=device_id,
        process_index=(
            int(process_index_value) if isinstance(process_index_value, int) else None
        ),
        gpu_uuid=None if gpu_identity is None else gpu_identity.uuid,
        gpu_model=None if gpu_identity is None else gpu_identity.model,
        # The CUDA runtime (via the bound JAX device) owns the compute
        # capability; nvidia-smi's compute_cap projection did not exist
        # before the r510 drivers, and the runtime is the same authority
        # that already proves CUDA-ness through jax_device.
        compute_capability=(
            str(device_compute_capability)
            if device_compute_capability is not None
            else None
        ),
        total_memory_bytes=(
            None if gpu_identity is None else gpu_identity.total_memory_bytes
        ),
        driver_version=(None if gpu_identity is None else gpu_identity.driver_version),
        cuda_version=(
            str(cuda_version_value)
            if requested_device == "gpu" and cuda_version_value is not None
            else None
        ),
        visible_devices=(
            os.environ.get("CUDA_VISIBLE_DEVICES")
            if requested_device == "gpu"
            else None
        ),
        hostname=platform.node(),
        scheduler_job_id=(
            os.environ.get("SLURM_JOB_ID") or os.environ.get("PBS_JOBID")
        ),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_repo_path(path: Path) -> str:
    return path.resolve().relative_to(Path(__file__).resolve().parents[1]).as_posix()


def _fixture_source_path(fixture_case: Fixture) -> Path | None:
    metadata = dict(fixture_case.metadata)
    candidates = [fixture_case.source, metadata.get("source_example")]
    for candidate in candidates:
        if not isinstance(candidate, str):
            continue
        path = Path(candidate)
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[1] / path
        if path.is_file():
            return path.resolve()
    return None


def _fixture_contract_payload(
    fixture_case: Fixture,
    initial: np.ndarray,
    *,
    initial_objective: float,
    initial_gradient_inf_norm: float,
    method: Method,
    maxiter: int,
    maxcor: int,
    device: str,
    intent: str,
) -> dict[str, object]:
    generator_path = (
        Path(__file__).resolve().parent / "fixtures" / "custom_quasi_newton.py"
    )
    source_path = _fixture_source_path(fixture_case)
    if source_path is None:
        source_path = generator_path
    metadata = dict(fixture_case.metadata)
    return {
        "generator_path": _relative_repo_path(generator_path),
        "generator_sha256": _sha256_file(generator_path),
        "source_path": _relative_repo_path(source_path),
        "source_sha256": _sha256_file(source_path),
        "certificate": fixture_case.certificate,
        "fixture_metadata": metadata,
        "initial_parameters": [float(value) for value in initial],
        "expected_initial_observables": {
            "objective": float(initial_objective),
            "gradient_inf_norm": float(initial_gradient_inf_norm),
        },
        "solver_options": {
            "device": device,
            "intent": intent,
            "maxcor": int(maxcor),
            "maxfun": None,
            "maxiter": int(maxiter),
            "ftol": _SOLVER_FTOL,
            "gtol": _SOLVER_GTOL,
            "maxls": _SOLVER_MAXLS,
            "method": method,
        },
        "tolerances": dict(_FIXTURE_TOLERANCES),
    }


def _sync(value):
    leaves = jax.tree_util.tree_leaves(value)
    for leaf in leaves:
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()
    return value


def _native_value_and_grad_or_none(
    fixture_case: Fixture,
) -> NativeValueAndGrad | None:
    callback = fixture_case.native_value_and_grad
    if callback is None and fixture_case.source.startswith("source_owned_"):
        raise ValueError(
            f"native provider is unavailable for unmatched source-owned fixture "
            f"{fixture_case.name!r}"
        )
    return callback


def _peak_rss_kib() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _current_rss_kib() -> int:
    """Return the process resident set at the solver boundary."""

    try:
        status = Path("/proc/self/status").read_text(encoding="utf-8")
    except FileNotFoundError:
        return _peak_rss_kib()
    for line in status.splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1])
    return _peak_rss_kib()


class _RSSPhase:
    """Sample this process's RSS during one explicitly named phase."""

    def __init__(self, phase: str, *, interval_seconds: float = 0.01) -> None:
        self.phase = phase
        self.interval_seconds = float(interval_seconds)
        self.start_rss_kib = 0
        self.peak_rss_kib = 0
        self.end_rss_kib = 0
        self.sample_count = 0
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> Self:
        with self._lock:
            self.start_rss_kib = _current_rss_kib()
            self.peak_rss_kib = self.start_rss_kib
            self.sample_count = 1
        self._stop.clear()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        with self._lock:
            self.end_rss_kib = _current_rss_kib()
            self.peak_rss_kib = max(self.peak_rss_kib, self.end_rss_kib)
            self.sample_count += 1

    def _sample(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            rss_kib = _current_rss_kib()
            with self._lock:
                self.peak_rss_kib = max(self.peak_rss_kib, rss_kib)
                self.sample_count += 1

    def measurement(self) -> PhaseRSSMeasurement:
        with self._lock:
            if self.sample_count < 2:
                raise RuntimeError(f"RSS phase {self.phase!r} was not completed")
            start_rss_kib = self.start_rss_kib
            peak_rss_kib = self.peak_rss_kib
            end_rss_kib = self.end_rss_kib
            sample_count = self.sample_count
        return PhaseRSSMeasurement(
            phase=self.phase,
            start_rss_kib=start_rss_kib,
            peak_rss_kib=peak_rss_kib,
            end_rss_kib=end_rss_kib,
            sample_count=sample_count,
            scope="self_proc_status_poll_10ms",
        )


def _child_rss_kib(pid: int) -> int | None:
    """Read the direct Linux child RSS without process-name matching."""

    try:
        status = Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    for line in status.splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1])
    return None


def _stop_provider_child(
    child: subprocess.Popen[str],
    reason: str,
) -> NoReturn:
    child.terminate()
    try:
        stdout, stderr = child.communicate(timeout=_PROVIDER_CHILD_TERM_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        child.kill()
        stdout, stderr = child.communicate()
        suffix = "; TERM grace expired, sent KILL"
    else:
        suffix = "; child exited after TERM"
    raise RuntimeError(
        f"provider child {child.pid} {reason}{suffix}\n"
        f"stdout:\n{stdout}\nstderr:\n{stderr}"
    )


def _provider_child_timeout_seconds(cases: str, override: int | None = None) -> int:
    """Return the bounded child watchdog for the selected fixture set.

    ``override`` is the explicit CLI bound for long-budget campaigns; it must
    be positive and is recorded verbatim in run provenance.
    """

    if override is not None:
        if override <= 0:
            raise ValueError(
                "provider child timeout override must be a positive second count"
            )
        return override
    selected_cases = {name.strip() for name in cases.split(",")}
    if "boozer" in selected_cases:
        return _BOOZER_PROVIDER_CHILD_TIMEOUT_SECONDS
    return _PROVIDER_CHILD_TIMEOUT_SECONDS


def _run_provider_child_process(
    command: list[str],
    *,
    gpu_uuid: str | None = None,
    timeout_seconds: int | None = None,
) -> ProcessGpuMemoryMeasurement:
    """Run one direct provider child and measure its GPU memory externally.

    ``gpu_uuid`` is the authenticated GPU the child must run on; ``None``
    selects the explicit CPU evidence instead of NVIDIA sampling.
    """

    effective_timeout_seconds = (
        _PROVIDER_CHILD_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds
    )
    child = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    monitor = (
        None
        if gpu_uuid is None
        else ProcessGpuMemoryMonitor(gpu_uuid=gpu_uuid, provider_pid=child.pid)
    )
    if monitor is not None:
        monitor.start()
    started = time.monotonic()
    while child.poll() is None:
        elapsed = time.monotonic() - started
        if elapsed >= effective_timeout_seconds:
            _stop_provider_child(
                child,
                f"exceeded {effective_timeout_seconds}-second watchdog",
            )
        rss_kib = _child_rss_kib(child.pid)
        if rss_kib is not None and rss_kib >= _PROVIDER_CHILD_RSS_LIMIT_KIB:
            _stop_provider_child(
                child,
                f"exceeded {_PROVIDER_CHILD_RSS_LIMIT_KIB // (1024 * 1024)}-GiB RSS watchdog "
                f"at {rss_kib} KiB",
            )
        time.sleep(_PROVIDER_CHILD_POLL_SECONDS)
    _stdout, stderr = child.communicate()
    measurement = (
        cpu_gpu_memory_unavailable(provider_pid=child.pid)
        if monitor is None
        else monitor.finish()
    )
    if child.returncode != 0:
        raise RuntimeError(
            f"provider child failed with exit code {child.returncode}:\n{stderr}"
        )
    return measurement


def _initial_value_and_grad(
    fixture_case: Fixture,
    x0: np.ndarray,
    *,
    provider: Provider | None = None,
) -> tuple[float, np.ndarray]:
    if provider == "native":
        callback = _native_value_and_grad_or_none(fixture_case)
        if callback is not None:
            return callback(np.asarray(x0, dtype=np.float64))
    value_and_grad = fixture_case.value_and_grad or jax.value_and_grad(
        fixture_case.objective
    )
    value, gradient = value_and_grad(jnp.asarray(x0, dtype=jnp.float64))
    _sync((value, gradient))
    return float(value), np.asarray(jax.device_get(gradient), dtype=np.float64)


def _dense_bfgs_update_memory_analysis(
    dimension: int,
    dtype: np.dtype,
) -> dict[str, int | bool]:
    """Measure XLA buffers for the isolated dense BFGS update."""

    n = int(dimension)
    x_dtype = np.dtype(dtype)
    hessian = jnp.eye(n, dtype=x_dtype)
    step = jnp.linspace(
        jnp.asarray(0.25, dtype=x_dtype),
        jnp.asarray(0.75, dtype=x_dtype),
        n,
    )
    gradient_delta = jnp.linspace(
        jnp.asarray(0.5, dtype=x_dtype),
        jnp.asarray(1.0, dtype=x_dtype),
        n,
    )
    identity = jnp.eye(n, dtype=x_dtype)

    def update(hessian_arg, step_arg, gradient_delta_arg, identity_arg):
        return _bfgs_inverse_hessian_update(
            hessian_arg,
            step_arg,
            gradient_delta_arg,
            base_identity_host=identity_arg,
        )

    compiled = (
        jax.jit(update)
        .lower(
            hessian,
            step,
            gradient_delta,
            identity,
        )
        .compile()
    )
    report = _compiled_step_memory_analysis(compiled)
    return {
        "dense_update_argument_bytes": report["compiled_step_argument_bytes"],
        "dense_update_output_bytes": report["compiled_step_output_bytes"],
        "dense_update_alias_bytes": report["compiled_step_alias_bytes"],
        "dense_update_temp_bytes": report["compiled_step_temp_bytes"],
        "dense_update_peak_live_bytes": report["compiled_step_peak_live_bytes"],
        "dense_update_compiled_memory_is_update_only": True,
    }


def _run_custom(
    fixture_case: Fixture,
    x0: np.ndarray,
    *,
    maxiter: int,
    maxcor: int,
    method: Literal["bfgs", "lbfgs"],
    run_mode: str = "stepwise",
    prepared: _PreparedCustom | None = None,
) -> tuple[
    object,
    int | None,
    int | None,
    bool,
    tuple[TransferMeasurement, ...],
    dict[str, int] | None,
    AcceptedIncumbentInnerState | None,
]:
    if prepared is not None:
        if method != "lbfgs":
            raise ValueError("prepared custom programs support only L-BFGS")
        if prepared.program.run_mode != run_mode:
            raise ValueError("prepared custom program run mode does not match request")
        return _run_prepared_custom(
            fixture_case,
            x0,
            maxiter=maxiter,
            maxcor=maxcor,
            prepared=prepared,
        )
    fused_value_and_grad = fixture_case.value_and_grad
    objective = fused_value_and_grad or fixture_case.objective
    uses_fused_value_and_grad = fused_value_and_grad is not None
    object.__setattr__(objective, "_simsopt_cache_jit_value_and_grad", True)
    x_device = jnp.asarray(x0, dtype=jnp.float64)
    memory_analysis: dict[str, int] | None = None
    final_incumbent_state: AcceptedIncumbentInnerState | None = None

    def record_memory_analysis(report: dict[str, int]) -> None:
        nonlocal memory_analysis
        memory_analysis = dict(report)

    with host_transfer_audit() as transfer_audit:
        if method == "bfgs":
            incumbent_factory = (
                fixture_case.accepted_incumbent_host_value_and_grad
            )
            if incumbent_factory is None:
                result = _minimize_bfgs_private(
                    objective,
                    x_device,
                    maxiter=maxiter,
                    gtol=_SOLVER_GTOL,
                    x_dtype=jnp.float64,
                    value_and_grad=uses_fused_value_and_grad,
                    memory_analysis_callback=record_memory_analysis,
                )
            else:
                incumbent_controller = incumbent_factory()
                initial_parameters = np.asarray(x0, dtype=np.float64)
                initial_value_and_gradient = incumbent_controller.value_and_grad(
                    initial_parameters
                )
                result = minimize_bfgs_host_core(
                    incumbent_controller.value_and_grad,
                    initial_parameters,
                    maxiter=maxiter,
                    gtol=_SOLVER_GTOL,
                    maxls=_SOLVER_MAXLS,
                    initial_value_and_grad=initial_value_and_gradient,
                    line_search_value_and_grad=(
                        line_search_value_and_grad_more_thuente_host
                    ),
                    callback=incumbent_controller.accept,
                )
                final_incumbent_state = incumbent_controller.current_inner_state
        else:
            result = _minimize_lbfgs_private(
                objective,
                x_device,
                maxiter=maxiter,
                maxcor=maxcor,
                ftol=_SOLVER_FTOL,
                gtol=_SOLVER_GTOL,
                maxls=_SOLVER_MAXLS,
                x_dtype=jnp.float64,
                run_mode=run_mode,
            )
        if method == "bfgs":
            if fixture_case.accepted_incumbent_host_value_and_grad is None:
                _result_converters._private_bfgs_result_to_optimize_result(result)
        else:
            _result_converters._private_lbfgs_result_to_optimize_result(result)
        transfer_summary = tuple(
            TransferMeasurement(
                phase=entry.phase,
                calls=entry.calls,
                leaves=entry.leaves,
                bytes=entry.bytes,
            )
            for entry in transfer_audit.summary()
        )
    _sync(result)
    status = int(np.asarray(result.status))
    return (
        result,
        int(np.asarray(result.nfev)),
        status,
        bool(np.asarray(result.converged)),
        transfer_summary,
        memory_analysis,
        final_incumbent_state,
    )


def _prepare_custom(
    fixture_case: Fixture,
    x0: np.ndarray,
    *,
    maxcor: int,
    run_mode: str = "stepwise",
) -> _PreparedCustom:
    """Prepare the custom L-BFGS state machine for fair warm timing."""

    if fixture_case.value_and_grad is not None:
        raise ValueError("prepared custom comparison requires a scalar fixture")
    objective = fixture_case.objective
    object.__setattr__(objective, "_simsopt_cache_jit_value_and_grad", True)
    params = jnp.asarray(x0, dtype=jnp.float64)
    program = prepare_lbfgs_private(
        objective,
        params,
        cache_owner=objective,
        maxcor=maxcor,
        ftol=_SOLVER_FTOL,
        gtol=_SOLVER_GTOL,
        maxls=_SOLVER_MAXLS,
        x_dtype=jnp.float64,
        run_mode=run_mode,
    )
    return _PreparedCustom(
        program=program,
        objective=objective,
        initial_parameter_bits=_parameter_bits(x0),
        parameter_shape=tuple(int(size) for size in params.shape),
        maxcor=maxcor,
    )


def _run_prepared_custom(
    fixture_case: Fixture,
    x0: np.ndarray,
    *,
    maxiter: int,
    maxcor: int,
    prepared: _PreparedCustom,
) -> tuple[
    object,
    int | None,
    int | None,
    bool,
    tuple[TransferMeasurement, ...],
    dict[str, int] | None,
    AcceptedIncumbentInnerState | None,
]:
    params = jnp.asarray(x0, dtype=jnp.float64)
    if (
        prepared.objective is not fixture_case.objective
        or prepared.initial_parameter_bits != _parameter_bits(x0)
        or prepared.parameter_shape != tuple(int(size) for size in params.shape)
        or prepared.maxcor != maxcor
    ):
        raise ValueError("prepared custom program does not match the requested run")
    with host_transfer_audit() as transfer_audit:
        result = prepared.program.run(params, maxiter=maxiter, maxfun=None)
        _result_converters._private_lbfgs_result_to_optimize_result(result)
        transfer_summary = tuple(
            TransferMeasurement(
                phase=entry.phase,
                calls=entry.calls,
                leaves=entry.leaves,
                bytes=entry.bytes,
            )
            for entry in transfer_audit.summary()
        )
    _sync(result)
    status = int(np.asarray(result.status))
    return (
        result,
        int(np.asarray(result.nfev)),
        status,
        bool(np.asarray(result.converged)),
        transfer_summary,
        None,
        None,
    )


def _run_native(
    fixture_case: Fixture,
    x0: np.ndarray,
    *,
    maxiter: int,
    maxcor: int,
    method: Method,
) -> tuple[
    optimize.OptimizeResult,
    int | None,
    int | None,
    bool,
    tuple[TransferMeasurement, ...],
    dict[str, int] | None,
    AcceptedIncumbentInnerState | None,
]:
    native_callback = _native_value_and_grad_or_none(fixture_case)
    value_and_grad = (
        None
        if native_callback is not None
        else jax.jit(
            fixture_case.value_and_grad or jax.value_and_grad(fixture_case.objective)
        )
    )
    calls = 0

    def scipy_value_and_grad(x: np.ndarray) -> tuple[float, np.ndarray]:
        nonlocal calls
        calls += 1
        if native_callback is not None:
            value, gradient = native_callback(
                np.asarray(x, dtype=np.float64),
            )
            return float(value), np.asarray(gradient, dtype=np.float64)
        assert value_and_grad is not None
        value, gradient = value_and_grad(jnp.asarray(x, dtype=jnp.float64))
        _sync((value, gradient))
        return float(value), np.asarray(gradient, dtype=np.float64)

    options = {
        "maxiter": maxiter,
        "gtol": _SOLVER_GTOL,
    }
    if method == "lbfgs":
        options.update({"maxcor": maxcor, "ftol": _SOLVER_FTOL})
    result = optimize.minimize(
        scipy_value_and_grad,
        np.asarray(x0, dtype=np.float64),
        jac=True,
        method="BFGS" if method == "bfgs" else "L-BFGS-B",
        options=options,
    )
    return result, calls, int(result.status), bool(result.success), (), None, None


def _parameter_bits(x0: np.ndarray) -> tuple[int, ...]:
    values = np.ascontiguousarray(np.asarray(x0, dtype=np.float64).reshape(-1))
    return tuple(int(value) for value in values.view(np.uint64))


def _run_optax(
    fixture_case: Fixture,
    x0: np.ndarray,
    *,
    maxiter: int,
    maxcor: int,
    prepared: _PreparedOptax | None = None,
) -> tuple[
    object,
    int | None,
    int | None,
    bool,
    tuple[TransferMeasurement, ...],
    dict[str, int] | None,
    AcceptedIncumbentInnerState | None,
]:
    if prepared is None:
        prepared = _prepare_optax(fixture_case, x0, maxcor=maxcor)
    params = jnp.asarray(x0, dtype=jnp.float64)
    parameter_bits = _parameter_bits(x0)
    if (
        prepared.objective is not fixture_case.objective
        or prepared.initial_parameter_bits != parameter_bits
        or prepared.parameter_shape != tuple(int(size) for size in params.shape)
        or prepared.maxcor != maxcor
    ):
        raise ValueError("prepared Optax program does not match the requested run")
    state = prepared.solver.init(params)
    value = prepared.initial_value
    gradient = prepared.initial_gradient
    iterations = 0
    status: int | None
    finite_initial = bool(
        jnp.all(jnp.isfinite(params))
        & jnp.isfinite(value)
        & jnp.all(jnp.isfinite(gradient))
    )
    if not finite_initial:
        status = 6
    elif float(jnp.max(jnp.abs(gradient))) <= _SOLVER_GTOL:
        status = 0
    else:
        status = 1
        for iterations in range(1, maxiter + 1):
            params, state, value, gradient = prepared.step(params, state)
            decrease_error = cast(
                jax.Array,
                optax.tree.get(state, "decrease_error"),
            )
            curvature_error = cast(
                jax.Array,
                optax.tree.get(state, "curvature_error"),
            )
            _sync((params, state, value, gradient))
            finite_endpoint = bool(
                jnp.all(jnp.isfinite(params))
                & jnp.isfinite(value)
                & jnp.all(jnp.isfinite(gradient))
            )
            if not finite_endpoint:
                status = 6
                break
            line_search_failed = bool(
                (~jnp.isfinite(decrease_error))
                | (~jnp.isfinite(curvature_error))
                | (decrease_error > 0.0)
                | (curvature_error > 0.0)
            )
            if line_search_failed:
                status = 2
                break
            if float(jnp.max(jnp.abs(gradient))) <= _SOLVER_GTOL:
                status = 0
                break
    _sync((value, gradient))
    return (
        (params, value, gradient, iterations),
        None,
        status,
        status == 0,
        (),
        None,
        None,
    )


def _prepare_optax(
    fixture_case: Fixture,
    x0: np.ndarray,
    *,
    maxcor: int,
) -> _PreparedOptax:
    """Build and compile the fixed-shape Optax programs once.

    The compiled callables are deliberately kept separate from the Python
    iteration budget.  This makes the second measurement a solver-only timing
    rather than a repeated optimizer/JIT-construction timing.
    """

    if fixture_case.value_and_grad is not None:
        raise ValueError("Optax comparison requires a scalar objective fixture")
    params = jnp.asarray(x0, dtype=jnp.float64)
    solver = optax.lbfgs(memory_size=maxcor)
    state = solver.init(params)
    value_and_grad = cast(
        _OptaxValueAndGrad,
        optax.value_and_grad_from_state(fixture_case.objective),
    )

    def step(
        step_params: jax.Array,
        step_state: optax.OptState,
    ) -> tuple[jax.Array, optax.OptState, jax.Array, jax.Array]:
        value, gradient = value_and_grad(step_params, state=step_state)
        updates, next_state = solver.update(
            gradient,
            step_state,
            step_params,
            value=value,
            grad=gradient,
            value_fn=fixture_case.objective,
        )
        updated_params = cast(jax.Array, optax.apply_updates(step_params, updates))
        next_value = cast(jax.Array, optax.tree.get(next_state, "value"))
        next_gradient = cast(jax.Array, optax.tree.get(next_state, "grad"))
        return updated_params, next_state, next_value, next_gradient

    jitted_step = jax.jit(step)
    compiled_step = cast(_OptaxStep, jitted_step.lower(params, state).compile())
    jitted_final_value_and_grad = jax.jit(jax.value_and_grad(fixture_case.objective))
    compiled_final_value_and_grad = cast(
        _OptaxFinalValueAndGrad,
        jitted_final_value_and_grad.lower(params).compile(),
    )
    initial_value, initial_gradient = compiled_final_value_and_grad(params)
    _sync((initial_value, initial_gradient))
    return _PreparedOptax(
        solver=solver,
        step=compiled_step,
        final_value_and_grad=compiled_final_value_and_grad,
        initial_value=initial_value,
        initial_gradient=initial_gradient,
        objective=fixture_case.objective,
        initial_parameter_bits=_parameter_bits(x0),
        parameter_shape=tuple(int(size) for size in params.shape),
        maxcor=maxcor,
    )


def _measurement(
    fixture_case: Fixture,
    provider: Provider,
    device: str,
    intent: str,
    x0: np.ndarray,
    *,
    maxiter: int,
    maxcor: int,
    method: Method,
    fixture_build_seconds: float,
    fixture_build_peak_rss_kib: int,
    fixture_phase_rss: PhaseRSSMeasurement | None = None,
) -> Measurement:
    initial_objective, initial_gradient = _initial_value_and_grad(
        fixture_case,
        x0,
        provider=provider,
    )
    fixture_contract = _fixture_contract_payload(
        fixture_case,
        x0,
        initial_objective=initial_objective,
        initial_gradient_inf_norm=float(np.max(np.abs(initial_gradient))),
        method=method,
        maxiter=maxiter,
        maxcor=maxcor,
        device=device,
        intent=intent,
    )
    phase_rss: list[PhaseRSSMeasurement] = []
    if fixture_phase_rss is not None:
        phase_rss.append(fixture_phase_rss)
    accepted_incumbent_bfgs = bool(
        provider == "custom"
        and method == "bfgs"
        and fixture_case.accepted_incumbent_host_value_and_grad is not None
    )
    algorithm_memory_contract = (
        _bfgs_memory_contract(fixture_case.expected_dimension, np.float64)
        if method == "bfgs" and not accepted_incumbent_bfgs
        else None
    )
    if algorithm_memory_contract is not None:
        with _RSSPhase("algorithm_memory_analysis") as memory_phase:
            algorithm_memory_contract.update(
                _dense_bfgs_update_memory_analysis(
                    fixture_case.expected_dimension,
                    np.dtype(np.float64),
                )
            )
        phase_rss.append(memory_phase.measurement())
    solver_start_rss_kib = _current_rss_kib()
    prepared_custom: _PreparedCustom | None = None
    prepared_optax: _PreparedOptax | None = None
    preparation_seconds = 0.0
    if provider == "custom" and method == "lbfgs":
        run_mode = "fused_stepwise" if intent == "fast" else "stepwise"
        preparation_started = time.perf_counter()
        with _RSSPhase("preparation") as preparation_phase:
            prepared_custom = _prepare_custom(
                fixture_case,
                x0,
                maxcor=maxcor,
                run_mode=run_mode,
            )
        phase_rss.append(preparation_phase.measurement())
        preparation_seconds = time.perf_counter() - preparation_started
    elif provider == "optax":
        preparation_started = time.perf_counter()
        with _RSSPhase("preparation") as preparation_phase:
            prepared_optax = _prepare_optax(fixture_case, x0, maxcor=maxcor)
        phase_rss.append(preparation_phase.measurement())
        preparation_seconds = time.perf_counter() - preparation_started

    def run_once():
        if provider == "native":
            # Native inner solvers may retain their last accepted state; cold
            # and warm measurements must start from the same native state.
            if fixture_case.native_reset is not None:
                fixture_case.native_reset()
            return _run_native(
                fixture_case,
                x0,
                maxiter=maxiter,
                maxcor=maxcor,
                method=method,
            )
        if provider == "custom":
            return _run_custom(
                fixture_case,
                x0,
                maxiter=maxiter,
                maxcor=maxcor,
                method=method,
                run_mode=(
                    "fused_stepwise"
                    if intent == "fast" and method == "lbfgs"
                    else "stepwise"
                ),
                prepared=prepared_custom,
            )
        if method != "lbfgs":
            raise ValueError("Optax comparison supports only method='lbfgs'")
        return _run_optax(
            fixture_case,
            x0,
            maxiter=maxiter,
            maxcor=maxcor,
            prepared=prepared_optax,
        )

    started = time.perf_counter()
    with _RSSPhase("cold_solver") as cold_phase:
        (
            cold_result,
            _cold_evaluations,
            _cold_status,
            _cold_success,
            _cold_transfer_audit,
            _cold_memory_analysis,
            _cold_incumbent_state,
        ) = run_once()
        _sync(cold_result)
    phase_rss.append(cold_phase.measurement())
    first_execution_seconds = time.perf_counter() - started
    cold_seconds = preparation_seconds + first_execution_seconds

    started = time.perf_counter()
    with _RSSPhase("warm_solver") as warm_phase:
        (
            result,
            evaluations,
            status,
            success,
            warm_transfer_audit,
            warm_memory_analysis,
            final_incumbent_state,
        ) = run_once()
        _sync(result)
    phase_rss.append(warm_phase.measurement())
    warm_seconds = time.perf_counter() - started
    solver_phases = tuple(
        phase
        for phase in phase_rss
        if phase.phase in {"preparation", "cold_solver", "warm_solver"}
    )
    if not solver_phases:
        raise RuntimeError("solver execution produced no RSS phase measurements")
    solver_peak_rss_kib = max(phase.peak_rss_kib for phase in solver_phases)
    solver_peak_rss_delta_kib = max(
        phase.peak_rss_kib - phase.start_rss_kib for phase in solver_phases
    )

    if provider == "optax":
        params, final_value, gradient, iterations = result
        final_objective = float(final_value)
        final_gradient_inf_norm = float(jnp.max(jnp.abs(gradient)))
        iteration_count = int(iterations)
        final_parameters = np.asarray(jax.device_get(params), dtype=np.float64)
    elif provider == "native":
        final_objective = float(result.fun)
        final_gradient_inf_norm = float(np.max(np.abs(np.asarray(result.jac))))
        iteration_count = int(result.nit)
        final_parameters = np.asarray(result.x, dtype=np.float64)
    else:
        final_objective = float(np.asarray(result.f_k))
        final_gradient_inf_norm = float(np.max(np.abs(np.asarray(result.g_k))))
        iteration_count = int(np.asarray(result.k))
        final_parameters = np.asarray(result.x_k, dtype=np.float64)

    finite_endpoint = bool(
        np.isfinite(final_objective)
        and np.isfinite(final_gradient_inf_norm)
        and np.all(np.isfinite(final_parameters))
    )
    status_convention = status_convention_for(
        provider,
        method,
        accepted_incumbent=bool(
            provider == "custom"
            and method == "bfgs"
            and fixture_case.accepted_incumbent_host_value_and_grad is not None
        ),
    )
    stopping_reason = _stopping_reason(
        provider_success=success,
        provider_status=status,
        status_convention=status_convention,
        iterations=iteration_count,
        max_iterations=maxiter,
        finite=finite_endpoint,
    )
    scientific_certification_started = time.perf_counter()
    scientific_endpoint = (
        fixture_case.native_scientific_endpoint
        if provider == "native"
        else fixture_case.scientific_endpoint
    )
    endpoint_evidence = (
        ScientificEndpointEvidence(inner_success=True, observables=())
        if scientific_endpoint is None
        else scientific_endpoint(final_parameters, final_incumbent_state)
    )
    scientific_certification_seconds = (
        time.perf_counter() - scientific_certification_started
    )
    scientific_observables = {
        key: float(value)
        for key, value in endpoint_evidence.observables
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    parameters_finite = bool(np.all(np.isfinite(final_parameters)))
    observables_finite = bool(
        np.isfinite(final_objective)
        and np.isfinite(final_gradient_inf_norm)
        and all(np.isfinite(value) for value in scientific_observables.values())
    )
    endpoint_certificate = certify_optimization_endpoint(
        provider_success=success,
        provider_status=status,
        status_convention=status_convention,
        iterations=iteration_count,
        max_iterations=maxiter,
        initial_gradient_inf_norm=float(np.max(np.abs(initial_gradient))),
        final_gradient_inf_norm=final_gradient_inf_norm,
        inner_success=endpoint_evidence.inner_success,
        parameters_finite=parameters_finite,
        observables_finite=observables_finite,
        constraint_norm=None,
    )
    if endpoint_certificate.stopping_reason != stopping_reason:
        raise RuntimeError("endpoint stopping-reason owners disagree")

    fixture_contract["final_certificate_fields"] = {
        "final_objective": final_objective,
        "final_gradient_inf_norm": final_gradient_inf_norm,
        "iterations": iteration_count,
        "evaluations": evaluations,
        "status": status,
        "success": success,
        "stopping_reason": stopping_reason,
        "final_parameters": [float(value) for value in final_parameters],
    }

    if algorithm_memory_contract is not None and warm_memory_analysis is not None:
        algorithm_memory_contract.update(warm_memory_analysis)

    transfer_calls = sum(entry.calls for entry in warm_transfer_audit)
    transfer_leaves = sum(entry.leaves for entry in warm_transfer_audit)
    transfer_bytes = sum(entry.bytes for entry in warm_transfer_audit)
    advance_observations = sum(
        entry.calls for entry in warm_transfer_audit if entry.phase == "advance"
    )
    if (
        provider == "custom"
        and method == "lbfgs"
        and intent == "fast"
        and advance_observations > iteration_count + 1
    ):
        raise RuntimeError(
            "fused_stepwise advance observations exceed the runner transfer gate"
        )

    return Measurement(
        case=fixture_case.name,
        provider=provider,
        method=method,
        device=device,
        intent=intent,
        solver_route=_solver_route(
            provider,
            method,
            intent=intent,
            accepted_incumbent=bool(
                provider == "custom"
                and method == "bfgs"
                and fixture_case.accepted_incumbent_host_value_and_grad is not None
            ),
        ),
        device_identity=_device_identity(device),
        dimension=fixture_case.expected_dimension,
        maxiter=maxiter,
        maxcor=maxcor,
        fixture_build_seconds=fixture_build_seconds,
        fixture_build_peak_rss_kib=fixture_build_peak_rss_kib,
        solver_start_rss_kib=solver_start_rss_kib,
        solver_peak_rss_kib=solver_peak_rss_kib,
        solver_peak_rss_delta_kib=solver_peak_rss_delta_kib,
        initial_parameters=tuple(float(value) for value in x0),
        final_parameters=tuple(float(value) for value in final_parameters),
        initial_objective=initial_objective,
        initial_gradient_inf_norm=float(np.max(np.abs(initial_gradient))),
        final_objective=final_objective,
        final_gradient_inf_norm=final_gradient_inf_norm,
        iterations=iteration_count,
        evaluations=evaluations,
        status=status,
        success=success,
        stopping_reason=stopping_reason,
        preparation_seconds=preparation_seconds,
        first_execution_seconds=first_execution_seconds,
        cold_seconds=cold_seconds,
        warm_seconds=warm_seconds,
        peak_rss_kib=max(phase.peak_rss_kib for phase in phase_rss),
        peak_rss_scope="self_proc_status_phase_max",
        ru_maxrss_kib=_peak_rss_kib(),
        peak_vram_mib=None,
        process_pid=os.getpid(),
        certificate=fixture_case.certificate,
        warm_transfer_audit=warm_transfer_audit,
        work_counters=WorkMeasurement(
            accepted_iterations=iteration_count,
            objective_evaluations=evaluations,
            transfer_calls=transfer_calls,
            transfer_leaves=transfer_leaves,
            transfer_bytes=transfer_bytes,
            advance_observations=advance_observations,
        ),
        inner_success=endpoint_evidence.inner_success,
        parameters_finite=parameters_finite,
        observables_finite=observables_finite,
        constraint_norm=None,
        endpoint_certificate=endpoint_certificate,
        scientific_observables=scientific_observables,
        scientific_certification_seconds=scientific_certification_seconds,
        diagnostic_artifacts={"memory_trace": None, "trial_trace": None},
        fixture_metadata=fixture_case.metadata,
        fixture_contract=fixture_contract,
        algorithm_memory_contract=algorithm_memory_contract,
        phase_rss=tuple(phase_rss),
    )


def _validate_monitored_gpu_identity(
    identity: object,
    *,
    provider: Provider,
    monitored_gpu_uuid: str | None,
) -> None:
    """Bind one child-reported device identity to the externally monitored GPU."""

    if not isinstance(identity, dict):
        raise TypeError(f"provider child {provider!r} omitted device identity")
    reported_gpu_uuid = cast(dict[str, object], identity).get("gpu_uuid")
    if reported_gpu_uuid != monitored_gpu_uuid:
        raise ValueError(
            f"provider child {provider!r} reported GPU UUID {reported_gpu_uuid!r}, "
            f"which differs from the monitored GPU UUID {monitored_gpu_uuid!r}"
        )


def _run_provider_child(
    *,
    provider: Provider,
    cases: str,
    device: str,
    intent: str,
    method: Method,
    maxiter: int,
    maxcor: int,
    output: Path,
    capture_boozer_trial_trace: bool = False,
    provider_child_timeout_seconds: int | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    monitored_gpu_uuid = (
        _selected_nvidia_smi_identity(None).uuid if device == "gpu" else None
    )
    command = [
        sys.executable,
        __file__,
        "--device",
        device,
        "--intent",
        intent,
        "--providers",
        provider,
        "--cases",
        cases,
        "--method",
        method,
        "--maxiter",
        str(maxiter),
        "--maxcor",
        str(maxcor),
        "--output",
        str(output),
        "--provider-child",
    ]
    if capture_boozer_trial_trace:
        command.append("--capture-boozer-trial-trace")
    memory_measurement = _run_provider_child_process(
        command,
        gpu_uuid=monitored_gpu_uuid,
        timeout_seconds=_provider_child_timeout_seconds(
            cases, provider_child_timeout_seconds
        ),
    )
    measurements_path = output / "measurements.json"
    payload = json.loads(measurements_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"provider child {provider!r} wrote a non-object payload")
    if payload.get("schema_version") != _RUNNER_SCHEMA_VERSION:
        raise ValueError(f"provider child {provider!r} wrote the wrong schema")
    if payload.get("provider_child") is not True:
        raise ValueError(f"provider child {provider!r} omitted child provenance")
    if capture_boozer_trial_trace:
        if payload.get("capture_boozer_trial_trace") is not True:
            raise ValueError(
                f"provider child {provider!r} omitted Boozer trial capture provenance"
            )
    elif "capture_boozer_trial_trace" in payload:
        raise ValueError(
            f"provider child {provider!r} added unrequested Boozer trial capture"
        )
    if payload.get("requested_device") != device or payload.get("method") != method:
        raise ValueError(f"provider child {provider!r} request contract mismatched")
    if payload.get("runtime_environment") != _runtime_environment_payload():
        raise ValueError(f"provider child {provider!r} runtime environment mismatched")
    _validate_monitored_gpu_identity(
        payload.get("device_identity"),
        provider=provider,
        monitored_gpu_uuid=monitored_gpu_uuid,
    )
    measurements = payload.get("measurements")
    if not isinstance(measurements, list):
        raise TypeError(f"provider child {provider!r} wrote an invalid payload")
    if capture_boozer_trial_trace and len(measurements) != 1:
        raise ValueError("Boozer trial capture requires exactly one measurement row")
    memory_artifact = process_gpu_memory_artifact(memory_measurement)
    typed_measurements: list[dict[str, object]] = []
    for raw_measurement in measurements:
        if not isinstance(raw_measurement, dict):
            raise TypeError(f"provider child {provider!r} wrote a non-object row")
        typed_measurement = cast(dict[str, object], raw_measurement)
        if typed_measurement.get("provider") != provider:
            raise ValueError(f"provider child {provider!r} wrote another provider")
        _validate_monitored_gpu_identity(
            typed_measurement.get("device_identity"),
            provider=provider,
            monitored_gpu_uuid=monitored_gpu_uuid,
        )
        diagnostic_artifacts = typed_measurement.get("diagnostic_artifacts")
        if not isinstance(diagnostic_artifacts, dict):
            raise TypeError(f"provider child {provider!r} omitted diagnostic artifacts")
        trial_trace = cast(dict[str, object], diagnostic_artifacts).get("trial_trace")
        if capture_boozer_trial_trace:
            if typed_measurement.get("case") != "boozer":
                raise ValueError("trial trace is attached to a non-Boozer row")
            if trial_trace != _BOOZER_TRIAL_TRACE_ARTIFACT_NAME:
                raise ValueError("provider child omitted the Boozer trial trace path")
            production_route = typed_measurement.get("solver_route")
            if not isinstance(production_route, str) or not production_route:
                raise TypeError("Boozer trial trace row omitted its solver route")
            expected_production_route = _solver_route(
                provider,
                method,
                intent=intent,
                accepted_incumbent=bool(
                    provider == "custom"
                    and method == "bfgs"
                    and fixture_accepted_incumbent("boozer")
                ),
            )
            if production_route != expected_production_route:
                raise ValueError(
                    "Boozer trial trace row solver route differs from the request"
                )
            measurement_maxiter = typed_measurement.get("maxiter")
            if not isinstance(measurement_maxiter, int) or isinstance(
                measurement_maxiter, bool
            ):
                raise TypeError("Boozer trial trace row omitted maxiter")
            if measurement_maxiter != maxiter:
                raise ValueError(
                    "Boozer trial trace row maxiter differs from the request"
                )
            measurement_evaluations = typed_measurement.get("evaluations")
            if not isinstance(measurement_evaluations, int) or isinstance(
                measurement_evaluations, bool
            ):
                raise TypeError("Boozer trial trace row omitted evaluations")
            raw_final_parameters = typed_measurement.get("final_parameters")
            if not isinstance(raw_final_parameters, list):
                raise TypeError(
                    "Boozer trial trace row omitted final parameters"
                )
            final_parameters = np.asarray(raw_final_parameters, dtype=np.float64)
            measurement_final_objective = typed_measurement.get("final_objective")
            if not isinstance(measurement_final_objective, (int, float)) or isinstance(
                measurement_final_objective, bool
            ):
                raise TypeError("Boozer trial trace row omitted final objective")
            measurement_final_gradient = typed_measurement.get(
                "final_gradient_inf_norm"
            )
            if not isinstance(measurement_final_gradient, (int, float)) or isinstance(
                measurement_final_gradient, bool
            ):
                raise TypeError(
                    "Boozer trial trace row omitted final gradient norm"
                )
            measurement_status = typed_measurement.get("status")
            if not isinstance(measurement_status, int) or isinstance(
                measurement_status, bool
            ):
                raise TypeError("Boozer trial trace row omitted status")
            validate_boozer_trial_trace(
                output / _BOOZER_TRIAL_TRACE_ARTIFACT_NAME,
                expected_provider=provider,
                expected_production_route=expected_production_route,
                expected_maxiter=maxiter,
                expected_evaluations=measurement_evaluations,
                expected_final_parameters=final_parameters,
                expected_final_objective=float(measurement_final_objective),
                expected_final_gradient_inf_norm=float(measurement_final_gradient),
                expected_final_status=measurement_status,
            )
        elif trial_trace is not None:
            raise ValueError("provider child emitted an unrequested trial trace")
        typed_measurements.append(
            {
                **typed_measurement,
                "peak_vram_mib": memory_artifact.peak_used_memory_mib,
                "diagnostic_artifacts": {
                    **cast(dict[str, object], diagnostic_artifacts),
                    "memory_trace": _GPU_MEMORY_ARTIFACT_NAME,
                },
            }
        )
    memory_artifact_path = output / _GPU_MEMORY_ARTIFACT_NAME
    memory_artifact_path.write_text(
        json.dumps(asdict(memory_artifact), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    measurements_path.write_text(
        json.dumps(
            {**payload, "measurements": typed_measurements}, indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )
    git_commit = payload.get("git_commit")
    git_clean = payload.get("git_clean")
    if not isinstance(git_commit, str) or not git_commit:
        raise TypeError(f"provider child {provider!r} omitted git commit")
    if not isinstance(git_clean, bool):
        raise TypeError(f"provider child {provider!r} omitted clean state")
    child_provenance: dict[str, object] = {
        "provider": provider,
        "measurements_path": f"{provider}/measurements.json",
        "measurements_sha256": _sha256_file(measurements_path),
        "gpu_memory_path": f"{provider}/{_GPU_MEMORY_ARTIFACT_NAME}",
        "gpu_memory_sha256": _sha256_file(memory_artifact_path),
        "measurement_count": len(typed_measurements),
        "git_commit": git_commit,
        "git_clean": git_clean,
        "runtime_environment": payload["runtime_environment"],
        "requested_device": payload["requested_device"],
        "method": payload["method"],
        "device_identity": payload.get("device_identity"),
    }
    if capture_boozer_trial_trace:
        trial_trace_path = output / _BOOZER_TRIAL_TRACE_ARTIFACT_NAME
        child_provenance.update(
            trial_trace_path=(
                f"{provider}/{_BOOZER_TRIAL_TRACE_ARTIFACT_NAME}"
            ),
            trial_trace_sha256=_sha256_file(trial_trace_path),
        )
    return typed_measurements, child_provenance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--intent", choices=("fast", "parity"), required=True)
    parser.add_argument("--providers", default="custom")
    parser.add_argument("--cases", default="rosenbrock")
    parser.add_argument("--method", choices=("bfgs", "lbfgs"))
    parser.add_argument("--maxiter", type=int, default=20)
    parser.add_argument("--maxcor", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capture-boozer-trial-trace", action="store_true")
    parser.add_argument("--provider-child-timeout-seconds", type=int, default=None)
    parser.add_argument("--provider-child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    _validate_intent_environment(args.device, args.intent)
    case_names = tuple(name.strip() for name in args.cases.split(","))
    if not case_names or any(not name for name in case_names):
        raise ValueError("at least one nonempty fixture name is required")
    method: Method = args.method or fixture_method(case_names[0])
    if any(fixture_method(name) != method for name in case_names):
        raise ValueError("selected fixtures require different solver methods")
    providers = [provider.strip() for provider in args.providers.split(",")]
    invalid = set(providers).difference({"native", "custom", "optax"})
    if invalid:
        raise ValueError(f"unknown providers: {sorted(invalid)}")
    if args.capture_boozer_trial_trace:
        if case_names != ("boozer",):
            raise ValueError("Boozer trial capture requires only the Boozer fixture")
        if method != "bfgs":
            raise ValueError("Boozer trial capture requires method='bfgs'")
        if any(provider not in {"native", "custom"} for provider in providers):
            raise ValueError(
                "Boozer trial capture supports only native and custom providers"
            )
    selected_cases: list[Fixture] = []
    provider_children: list[dict[str, object]] = []
    if args.provider_child:
        if len(providers) != 1:
            raise ValueError("provider child requires exactly one provider")
        backend = jax.default_backend()
        devices = cast(tuple[object, ...], tuple(jax.devices()))
        device_platforms = tuple(
            str(getattr(device, "platform", "")) for device in devices
        )
        if args.device == "cpu":
            if backend != "cpu" or any(
                platform != "cpu" for platform in device_platforms
            ):
                raise RuntimeError(
                    f"requested CPU execution, got backend={backend!r}, "
                    f"devices={devices!r}"
                )
        elif (
            len(devices) != 1
            or backend not in {"cuda", "gpu", "rocm"}
            or any(
                platform not in {"cuda", "gpu", "rocm"} for platform in device_platforms
            )
        ):
            raise RuntimeError(
                "requested strict single-GPU execution, "
                f"got backend={backend!r}, devices={devices!r}"
            )
        fixture_build_started = time.perf_counter()
        with _RSSPhase("fixture_build") as fixture_phase:
            selected_cases = [fixture(name) for name in case_names]
        fixture_phase_rss = fixture_phase.measurement()
        fixture_build_seconds = time.perf_counter() - fixture_build_started
        fixture_build_peak_rss_kib = _peak_rss_kib()
        measurements = [
            _measurement(
                fixture_case,
                providers[0],
                args.device,
                args.intent,
                np.asarray(fixture_case.initial, dtype=np.float64),
                maxiter=args.maxiter,
                maxcor=args.maxcor,
                method=method,
                fixture_build_seconds=fixture_build_seconds,
                fixture_build_peak_rss_kib=fixture_build_peak_rss_kib,
                fixture_phase_rss=fixture_phase_rss,
            )
            for fixture_case in selected_cases
        ]
    else:
        backend = "provider-child"
        devices = ()
        measurements = []
        for provider in providers:
            child_output = args.output / provider
            child_measurements, child_provenance = _run_provider_child(
                provider=cast(Provider, provider),
                cases=args.cases,
                device=args.device,
                intent=args.intent,
                method=method,
                maxiter=args.maxiter,
                maxcor=args.maxcor,
                output=child_output,
                capture_boozer_trial_trace=args.capture_boozer_trial_trace,
                provider_child_timeout_seconds=args.provider_child_timeout_seconds,
            )
            measurements.extend(child_measurements)
            provider_children.append(child_provenance)
    args.output.mkdir(parents=True, exist_ok=True)
    measurement_payload = [
        asdict(measurement) if isinstance(measurement, Measurement) else measurement
        for measurement in measurements
    ]
    if args.provider_child and args.capture_boozer_trial_trace:
        if len(selected_cases) != 1 or len(measurement_payload) != 1:
            raise RuntimeError("Boozer trial capture requires one child measurement")
        row = measurement_payload[0]
        if not isinstance(row, dict):
            raise TypeError("Boozer trial capture requires an object measurement")
        production_evaluations = row.get("evaluations")
        if not isinstance(production_evaluations, int) or isinstance(
            production_evaluations, bool
        ):
            raise TypeError("Boozer measurement omitted evaluations")
        production_final_objective = row.get("final_objective")
        if not isinstance(production_final_objective, (int, float)) or isinstance(
            production_final_objective, bool
        ):
            raise TypeError("Boozer measurement omitted final objective")
        production_final_gradient_inf_norm = row.get("final_gradient_inf_norm")
        if not isinstance(
            production_final_gradient_inf_norm, (int, float)
        ) or isinstance(production_final_gradient_inf_norm, bool):
            raise TypeError("Boozer measurement omitted final gradient norm")
        production_final_status = row.get("status")
        if not isinstance(production_final_status, int) or isinstance(
            production_final_status, bool
        ):
            raise TypeError("Boozer measurement omitted final status")
        raw_production_final_parameters = row.get("final_parameters")
        if not isinstance(raw_production_final_parameters, (list, tuple)):
            raise TypeError("Boozer measurement omitted final parameters")
        production_final_parameters = np.asarray(
            raw_production_final_parameters, dtype=np.float64
        )
        trial_trace_path = args.output / _BOOZER_TRIAL_TRACE_ARTIFACT_NAME
        diagnostic_result = run_boozer_host_diagnostic(
            selected_cases[0],
            provider=cast(TrialProvider, providers[0]),
            manifest_path=trial_trace_path,
            production_evaluations=production_evaluations,
            production_final_objective=float(production_final_objective),
            production_final_gradient_inf_norm=float(
                production_final_gradient_inf_norm
            ),
            production_final_status=production_final_status,
            production_final_parameters=production_final_parameters,
            maxiter=args.maxiter,
            maxls=_SOLVER_MAXLS,
            gtol=_SOLVER_GTOL,
        )
        if diagnostic_result.trial_trace != trial_trace_path:
            raise RuntimeError("Boozer diagnostic wrote an unexpected trial path")
        diagnostic_artifacts = row.get("diagnostic_artifacts")
        if not isinstance(diagnostic_artifacts, dict):
            raise TypeError("Boozer measurement omitted diagnostic artifacts")
        measurement_payload[0] = {
            **row,
            "diagnostic_artifacts": {
                **cast(dict[str, object], diagnostic_artifacts),
                "trial_trace": _BOOZER_TRIAL_TRACE_ARTIFACT_NAME,
            },
        }
    if not measurement_payload:
        raise RuntimeError("runner produced no measurements")
    if args.provider_child:
        device_identity_payload = asdict(_device_identity(args.device))
        devices_payload = [str(device) for device in devices]
    else:
        first_identity = measurement_payload[0].get("device_identity")
        if not isinstance(first_identity, dict):
            raise TypeError("provider child omitted device identity")
        device_identity_payload = first_identity
        backend = str(first_identity.get("backend"))
        devices_payload = sorted(
            {
                str(cast(dict[str, object], row["device_identity"])["jax_device"])
                for row in measurement_payload
            }
        )
    git_commit, orchestrator_git_clean = _checkout_provenance()
    if args.provider_child:
        provider_children = []
        git_clean = orchestrator_git_clean
    else:
        child_commits = {cast(str, child["git_commit"]) for child in provider_children}
        if child_commits != {git_commit}:
            raise RuntimeError(
                "provider child commit does not match the orchestrator checkout"
            )
        git_clean = orchestrator_git_clean and all(
            child["git_clean"] is True for child in provider_children
        )
    payload = {
        "schema_version": _RUNNER_SCHEMA_VERSION,
        "argv": list(sys.argv),
        "requested_device": args.device,
        "backend": backend,
        "devices": devices_payload,
        "device_identity": device_identity_payload,
        "python_version": platform.python_version(),
        "method": method,
        "jax_version": jax.__version__,
        "jaxlib_version": importlib.metadata.version("jaxlib"),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "optax_version": optax.__version__,
        "runtime_environment": _runtime_environment_payload(),
        "git_commit": git_commit,
        "git_clean": git_clean,
        "orchestrator_git_clean": orchestrator_git_clean,
        "provider_child": args.provider_child,
        "provider_children": provider_children,
        "provider_child_timeout_seconds": _provider_child_timeout_seconds(
            args.cases, args.provider_child_timeout_seconds
        ),
        "provider_child_rss_limit_kib": _PROVIDER_CHILD_RSS_LIMIT_KIB,
        "provider_child_term_grace_seconds": _PROVIDER_CHILD_TERM_GRACE_SECONDS,
        "measurements": measurement_payload,
    }
    if args.capture_boozer_trial_trace:
        payload["capture_boozer_trial_trace"] = True
    (args.output / "measurements.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
