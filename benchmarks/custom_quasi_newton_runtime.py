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

from benchmarks.fixtures.custom_quasi_newton import (
    Fixture,
    NativeValueAndGrad,
    fixture,
)

Provider = Literal["native", "custom", "optax"]
Method = Literal["bfgs", "lbfgs"]


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
_PROVIDER_CHILD_RSS_LIMIT_KIB = 8 * 1024 * 1024
_PROVIDER_CHILD_TERM_GRACE_SECONDS = 5
_PROVIDER_CHILD_POLL_SECONDS = 0.1
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
class Measurement:
    case: str
    provider: Provider
    method: Method
    device: str
    intent: str
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
    cold_seconds: float
    warm_seconds: float
    peak_rss_kib: int
    peak_rss_scope: str
    process_pid: int
    certificate: str
    warm_transfer_audit: tuple[TransferMeasurement, ...]
    phase_rss: tuple[PhaseRSSMeasurement, ...]
    fixture_metadata: tuple[tuple[str, object], ...]
    fixture_contract: dict[str, object]
    algorithm_memory_contract: dict[str, int | bool] | None


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


def _run_provider_child_process(command: list[str]) -> None:
    child = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    started = time.monotonic()
    while child.poll() is None:
        elapsed = time.monotonic() - started
        if elapsed >= _PROVIDER_CHILD_TIMEOUT_SECONDS:
            _stop_provider_child(
                child,
                f"exceeded {_PROVIDER_CHILD_TIMEOUT_SECONDS}-second watchdog",
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
    if child.returncode != 0:
        raise RuntimeError(
            f"provider child failed with exit code {child.returncode}:\n{stderr}"
        )


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
    prepared: _PreparedCustom | None = None,
) -> tuple[
    object,
    int | None,
    int | None,
    bool,
    tuple[TransferMeasurement, ...],
    dict[str, int] | None,
]:
    if prepared is not None:
        if method != "lbfgs":
            raise ValueError("prepared custom programs support only L-BFGS")
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

    def record_memory_analysis(report: dict[str, int]) -> None:
        nonlocal memory_analysis
        memory_analysis = dict(report)

    with host_transfer_audit() as transfer_audit:
        if method == "bfgs":
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
            result = _minimize_lbfgs_private(
                objective,
                x_device,
                maxiter=maxiter,
                maxcor=maxcor,
                ftol=_SOLVER_FTOL,
                gtol=_SOLVER_GTOL,
                maxls=_SOLVER_MAXLS,
                x_dtype=jnp.float64,
            )
        if method == "bfgs":
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
    )


def _prepare_custom(
    fixture_case: Fixture,
    x0: np.ndarray,
    *,
    maxcor: int,
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
    return result, calls, int(result.status), bool(result.success), (), None


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


def _stopping_reason(
    *,
    iterations: int,
    maxiter: int,
    status: int | None,
    success: bool,
) -> str:
    """Classify the terminal state without treating finite decrease as success."""

    if success:
        return "converged"
    if status == 99:
        return "callback-stopped"
    if status == 6:
        return "nonfinite"
    if status == 2:
        return "line-search-failed"
    if iterations >= maxiter:
        return "iteration-limit"
    return "failed"


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
    algorithm_memory_contract = (
        _bfgs_memory_contract(fixture_case.expected_dimension, np.float64)
        if method == "bfgs"
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
        preparation_started = time.perf_counter()
        with _RSSPhase("preparation") as preparation_phase:
            prepared_custom = _prepare_custom(fixture_case, x0, maxcor=maxcor)
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
        ) = run_once()
        _sync(cold_result)
    phase_rss.append(cold_phase.measurement())
    cold_seconds = preparation_seconds + time.perf_counter() - started

    started = time.perf_counter()
    with _RSSPhase("warm_solver") as warm_phase:
        (
            result,
            evaluations,
            status,
            success,
            warm_transfer_audit,
            warm_memory_analysis,
        ) = run_once()
        _sync(result)
    phase_rss.append(warm_phase.measurement())
    warm_seconds = time.perf_counter() - started
    solver_peak_rss_kib = max(solver_start_rss_kib, _peak_rss_kib())

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

    stopping_reason = _stopping_reason(
        iterations=iteration_count,
        maxiter=maxiter,
        status=status,
        success=success,
    )

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

    return Measurement(
        case=fixture_case.name,
        provider=provider,
        method=method,
        device=device,
        intent=intent,
        dimension=fixture_case.expected_dimension,
        maxiter=maxiter,
        maxcor=maxcor,
        fixture_build_seconds=fixture_build_seconds,
        fixture_build_peak_rss_kib=fixture_build_peak_rss_kib,
        solver_start_rss_kib=solver_start_rss_kib,
        solver_peak_rss_kib=solver_peak_rss_kib,
        solver_peak_rss_delta_kib=max(0, solver_peak_rss_kib - solver_start_rss_kib),
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
        cold_seconds=cold_seconds,
        warm_seconds=warm_seconds,
        peak_rss_kib=_peak_rss_kib(),
        peak_rss_scope="provider_child_process_lifetime",
        process_pid=os.getpid(),
        certificate=fixture_case.certificate,
        warm_transfer_audit=warm_transfer_audit,
        fixture_metadata=fixture_case.metadata,
        fixture_contract=fixture_contract,
        algorithm_memory_contract=algorithm_memory_contract,
        phase_rss=tuple(phase_rss),
    )


def _run_provider_child(
    *,
    provider: Provider,
    cases: str,
    device: str,
    intent: str,
    maxiter: int,
    maxcor: int,
    output: Path,
) -> list[dict[str, object]]:
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
        "--maxiter",
        str(maxiter),
        "--maxcor",
        str(maxcor),
        "--output",
        str(output),
        "--provider-child",
    ]
    _run_provider_child_process(command)
    payload = json.loads((output / "measurements.json").read_text(encoding="utf-8"))
    measurements = payload.get("measurements")
    if not isinstance(measurements, list):
        raise TypeError(f"provider child {provider!r} wrote an invalid payload")
    return measurements


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
    parser.add_argument("--provider-child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    _validate_intent_environment(args.device, args.intent)
    backend = jax.default_backend()
    devices = cast(tuple[object, ...], tuple(jax.devices()))
    device_platforms = tuple(str(getattr(device, "platform", "")) for device in devices)
    if args.device == "cpu":
        if backend != "cpu" or any(platform != "cpu" for platform in device_platforms):
            raise RuntimeError(
                f"requested CPU execution, got backend={backend!r}, devices={devices!r}"
            )
    elif backend not in {"cuda", "gpu", "rocm"} or any(
        platform not in {"cuda", "gpu", "rocm"} for platform in device_platforms
    ):
        raise RuntimeError(
            f"requested strict GPU execution, got backend={backend!r}, devices={devices!r}"
        )
    fixture_build_started = time.perf_counter()
    with _RSSPhase("fixture_build") as fixture_phase:
        selected_cases = [fixture(name.strip()) for name in args.cases.split(",")]
    fixture_phase_rss = fixture_phase.measurement()
    fixture_build_seconds = time.perf_counter() - fixture_build_started
    fixture_build_peak_rss_kib = _peak_rss_kib()
    method: Method = args.method or selected_cases[0].method
    if any(fixture_case.method != method for fixture_case in selected_cases):
        raise ValueError("selected fixtures require different solver methods")
    providers = [provider.strip() for provider in args.providers.split(",")]
    invalid = set(providers).difference({"native", "custom", "optax"})
    if invalid:
        raise ValueError(f"unknown providers: {sorted(invalid)}")
    if args.provider_child:
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
        measurements = []
        for provider in providers:
            child_output = args.output / provider
            measurements.extend(
                _run_provider_child(
                    provider=provider,
                    cases=args.cases,
                    device=args.device,
                    intent=args.intent,
                    maxiter=args.maxiter,
                    maxcor=args.maxcor,
                    output=child_output,
                )
            )
    args.output.mkdir(parents=True, exist_ok=True)
    measurement_payload = [
        asdict(measurement) if isinstance(measurement, Measurement) else measurement
        for measurement in measurements
    ]
    git_commit, git_clean = _checkout_provenance()
    payload = {
        "schema_version": 6,
        "argv": list(sys.argv),
        "requested_device": args.device,
        "backend": backend,
        "devices": [str(device) for device in devices],
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
        "provider_child_timeout_seconds": _PROVIDER_CHILD_TIMEOUT_SECONDS,
        "provider_child_rss_limit_kib": _PROVIDER_CHILD_RSS_LIMIT_KIB,
        "provider_child_term_grace_seconds": _PROVIDER_CHILD_TERM_GRACE_SECONDS,
        "measurements": measurement_payload,
    }
    (args.output / "measurements.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
