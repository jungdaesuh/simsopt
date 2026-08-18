"""Native/GPU speed benchmark for the finite-build Stage-II example.

Plan: ``docs/jax_gpu_finitebuild_native_speed_implementation_plan.md``.

The benchmark reuses the parity case's frozen input bundle and native
evaluator construction (``examples/jax/parity/cases/
native_stage_two_optimization_finitebuild.py``) so there is exactly one
physics specification.  Every measured leg runs in its own subprocess with a
fully scrubbed and pinned threading environment; the orchestrator serializes
GPU use behind a phase-start pid allowlist and records subprocess wall time.
Raw JSON rows bind the runtime identity, and the terminal manifest enumerates
every expected leg with its SHA-256.  The validator recomputes gate
eligibility, medians, paired ratios, and the verdict from raw rows alone, and
enforces one runtime identity across the run (git commit, changed-file
hashes, ``simsoptpp`` binary, JAX version and device list): missing,
nonfinite, mismatched, or partial rows make the result ``NOT_PRODUCED``,
never a win or loss.

Subcommands (run from the repository root with ``.`` and ``src`` on
``PYTHONPATH``, e.g. ``PYTHONPATH=.:src .venv-qn-gpu/bin/python``):

    baseline
        Value/gradient/diagnostics at the frozen initial state and two
        deterministic perturbed states, for the independent native and JAX
        evaluators.
    gate
        Freezes the scientific quality contract from one untimed native
        reference run of the shipped 400-step/400-history formulation.  Its
        stopping callback captures every accepted iterate, so the truncated
        anchor -- the like-for-like endpoint every budget-truncated lane is
        compared against -- comes from that run's own trajectory instead of
        from a replay (native solves at ``OMP>1`` do not reproduce across
        processes).
    kernel-canary
        Warm value/gradient program against native across the preregistered
        OpenMP sweep; applies the 1.10x proceed/close criterion.
    hlo-capture
        Lowers and compiles the full-scale value/gradient program and retains
        StableHLO, optimized HLO, the operation census, and the cost analysis.
    hlo-diff <before-run> <after-run>
        Classifies a pre/post refactor pair ``DCE_NULL`` or ``CHANGED`` from
        the census and cost analysis (raw HLO text is not the predicate).
    native-matrix --gate <contract>
        Native OpenMP x history time-to-quality: independent fresh-process
        solves that stop at the frozen gate rung, plus one untimed
        shipped-default disclosure lane.
    jax-sweep --gate <contract>
        JAX history sweep over the preregistered coarse GPU budget ladder.
    freeze-selection <native-matrix-run> <jax-sweep-run> --output <path>
        Binds both validated selections and their manifests into one document.
    final-pairs --gate <contract> --selection <path> [--cache-policy ...]
        Five interleaved native/GPU pairs under the frozen selections.  Both
        lanes are split the same way: a warm leg owns the warm statistic and
        a single-solve leg owns the subprocess-wall statistic.
    validate <run-dir>
        Recomputes the phase verdict from raw rows alone.
    leg --spec <path> --row <path>
        Internal: runs one leg inside its own pinned subprocess.

Every GPU endpoint is re-evaluated through the independent native evaluator
(``native-endpoint-eval``) and the gate clauses are applied to that native
re-evaluation, never to the JAX lane's self-report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import resource
import statistics
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = REPO_ROOT / ".artifacts" / "stage_two_finitebuild_native_gpu"

ROW_SCHEMA = "stage-two-finitebuild-native-gpu-row-v1"
GATE_SCHEMA = "stage-two-finitebuild-quality-contract-v3"
MANIFEST_SCHEMA = "stage-two-finitebuild-native-gpu-manifest-v1"
VERDICT_SCHEMA = "stage-two-finitebuild-native-gpu-verdict-v1"
SELECTION_SCHEMA = "stage-two-finitebuild-selection-v1"

CASE_ID = "native-stage-two-optimization-finitebuild"
SCALE = "native_default"

# ---------------------------------------------------------------------------
# Plan-frozen contract constants.  Preregistered in
# ``docs/jax_gpu_finitebuild_native_speed_implementation_plan.md`` before any
# timed configuration was ranked; never revised post-hoc from measurements.
# ---------------------------------------------------------------------------
GATE_OBJECTIVE_MARGIN = 1.001
GATE_ENDPOINT_RTOL = 5.0e-2
GATE_ENDPOINT_ATOL = 1.0e-9
GATE_GRADIENT_NORM_MARGIN = 1.05
GATE_GRADIENT_NORM_FLOOR = 1.0e-12
# The plan freezes one 1.05 slack and the contract spends it twice: once as
# the one-sided quality caps on the nonnegative objective terms, once as the
# gradient-norm slack.  Numerically identical, deliberately separate keys --
# a future revision of one must not silently move the other.
GATE_QUALITY_CAP_MARGIN = 1.05
KERNEL_CANARY_MINIMUM_RATIO = 1.10
FINAL_MEDIAN_MINIMUM_RATIO = 1.10
FINAL_EVERY_PAIR_MINIMUM_RATIO = 1.00
FINAL_PAIR_COUNT = 5

NATIVE_OMP_SWEEP = (2, 4, 8, 16, 32, 48)
NATIVE_HISTORY_SWEEP = (10, 20, 40, 400)
JAX_HISTORY_SWEEP = (10, 20, 40)
# Amended 2026-08-17 (plan: budget-parity amendment): the fp64 rung is crossed
# at median native iteration 736 off the reference trajectory, so a ladder
# capped at 400 denies the GPU lane the 2x-reference allowance the native
# callback-stop protocol already grants (NATIVE_STOP_MAX_STEPS = 800).  560
# and 800 extend the coarse ladder to budget parity; the sweep kill criterion
# is final at b <= 800.
GPU_BUDGET_SWEEP = (40, 80, 160, 240, 400, 560, 800)
SELECTION_REPETITIONS = 3
FINAL_WARM_REPETITIONS = 3
GATE_REFERENCE_OMP = 8
GATE_REFERENCE_STEPS = 400
# Iteration cap for native stop-at-target legs.  The rung sits at 1.001x a
# 400-iteration reference trajectory while sibling trajectories fork ~1%
# two-sided, so a cap equal to the reference budget would leave sub-fork
# headroom and fail roughly half of all repetitions on noise.  The cap is
# not a selected or compared quantity -- the stop rule decides the work --
# so doubling it changes no gate clause.  Preregistered 2026-08-17 before
# any native-matrix or final-pairs run existed.
NATIVE_STOP_MAX_STEPS = 2 * GATE_REFERENCE_STEPS
GATE_REFERENCE_HISTORY = 400
WARM_VALUE_GRAD_CALLS = 20
KERNEL_CANARY_REPETITIONS = 3
# Host threads a GPU leg is allowed: the device owns the arithmetic, and a
# wide host mask only adds launch-thread contention.
GPU_HOST_OMP_THREADS = 8

BASELINE_STATE_LABELS = ("initial", "perturbed_a", "perturbed_b")
BASELINE_PERTURBATION_SCALES = (1.0e-4, -1.0e-3)

# ---------------------------------------------------------------------------
# Measured calibration constants.  These are NOT preregistered: they were
# calibrated from the 2026-08-17 baseline run of this benchmark on this box,
# and each one carries the observation it was fitted to.  They bound
# floating-point reduction-order noise between two independent evaluators;
# any real physics defect sits orders of magnitude above them.
# ---------------------------------------------------------------------------
BASELINE_RTOL = 2.0e-8
BASELINE_ATOL = 2.0e-10
# Gradient components accumulate 7.37M source-target interactions, so GPU-XLA
# and CPU-simsoptpp reduction orders separate per component.  Measured
# 2026-08-17: max absolute difference 2.5e-9, max difference relative to the
# gradient scale 7e-8.  The per-component rule below keeps ~4x margin on the
# absolute term and ~14x on the relative term.
BASELINE_GRADIENT_ATOL = 1.0e-8
BASELINE_GRADIENT_RTOL = 1.0e-6
# Minimum clearance is a min-reduction over pairwise distances, not a sum, so
# it carries no accumulation error: measured 2026-08-17 relative difference
# 2.4e-16 (one ulp).  1e-12 is ~4000x that and still rejects any geometric
# disagreement.
BASELINE_CLEARANCE_RTOL = 1.0e-12
# Ceiling on how far a GPU lane's self-reported endpoint metrics may sit from
# the independent native re-evaluation of the same solution vector before the
# two lanes are declared to have diverged (a physics fork, not noise).
LANE_CROSS_CHECK_RTOL = 5.0e-2

VERDICT_WIN = "WIN"
VERDICT_CLOSED = "CLOSED_BOUNDED_NEGATIVE"
VERDICT_NOT_PRODUCED = "NOT_PRODUCED"
DECISION_PROCEED = "PROCEED"

# Inherited threading configuration changes OpenMP reduction order, which
# forks native floating-point trajectories between launch contexts (proven
# 2026-08-04).  Scrub by prefix rather than by name -- the same convention as
# ``benchmarks/run_jax_native_example_measurements.py`` -- because a single
# unscrubbed ``KMP_AFFINITY`` or ``OMP_PROC_BIND`` silently re-pins the leg.
_NUMERICAL_ENVIRONMENT_PREFIXES = (
    "OMP_",
    "GOMP_",
    "KMP_",
    "MKL_",
    "OPENBLAS_",
    "NUMEXPR_",
    "VECLIB_",
    "BLIS_",
    "NUMBA_",
)
_SCRUBBED_ENVIRONMENT_PREFIXES = _NUMERICAL_ENVIRONMENT_PREFIXES + (
    "JAX_",
    "XLA_",
    "SIMSOPT_",
)
_JAX_ENVIRONMENT = {
    "SIMSOPT_BACKEND_MODE": "jax_gpu_fast",
    "SIMSOPT_BACKEND_STRICT": "1",
    "SIMSOPT_PRECISION": "fp64",
    "JAX_PLATFORMS": "cuda",
    "JAX_ENABLE_X64": "1",
    "CUDA_VISIBLE_DEVICES": "0",
    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    "XLA_FLAGS": "--xla_gpu_exclude_nondeterministic_ops=true",
    "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES": "-1",
    "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0",
    # Plan STRICT_GPU_ENV parity.  Every in-leg device-to-host read routes
    # through ``host_array`` or the typed optimizer's terminal conversion and
    # every host-to-device staging is an explicit ``device_put``, so the
    # implicit-only guard cannot fire on a legitimate path.
    "SIMSOPT_JAX_TRANSFER_GUARD": "disallow",
    "JAX_TRANSFER_GUARD": "disallow",
}


class BenchmarkError(RuntimeError):
    """A benchmark phase cannot produce trustworthy raw evidence."""


# ---------------------------------------------------------------------------
# JSON, hashing, and atomic publication
# ---------------------------------------------------------------------------


def _json_safe(value: object) -> object:
    """Replace nonfinite floats with their IEEE names, recursively.

    Canonical bytes must be reproducible by any JSON reader, so ``NaN`` and
    ``Infinity`` literals -- which are not JSON -- are published as strings.
    ``_finite`` reads them back as nonfinite (``float("NaN")`` is ``nan``), so
    a poisoned value still fails every downstream numeric clause.
    """
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if value == math.inf:
            return "Infinity"
        if value == -math.inf:
            return "-Infinity"
        return value
    if isinstance(value, Mapping):
        return {str(name): _json_safe(entry) for name, entry in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(entry) for entry in value]
    return value


def _canonical_json_bytes(value: object) -> bytes:
    payload = json.dumps(
        _json_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return (payload + "\n").encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _write_json_exclusive(path: Path, document: object) -> str:
    """Write a document once, refusing to overwrite, and return its SHA-256."""
    payload = _canonical_json_bytes(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return _sha256_bytes(payload)


def _replace_json_atomic(path: Path, document: object) -> str:
    """Atomically publish a document that terminates a phase (the manifest)."""
    payload = _canonical_json_bytes(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return _sha256_bytes(payload)


def _read_json(path: Path) -> object:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


# ---------------------------------------------------------------------------
# Runtime identity binding
# ---------------------------------------------------------------------------


def _git_identity() -> dict[str, object]:
    def run(*arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout

    # ``-z`` is the machine-parsing format: NUL-separated fields, no quoting,
    # and rename/copy entries emit the new path first followed by the original
    # path as its own field ("R  new\0old").  Line splitting drops paths that
    # contain spaces or non-ASCII bytes, which would silently under-report a
    # dirty tree.
    fields = run("status", "--porcelain", "-z").split("\0")
    if fields and fields[-1] == "":
        fields.pop()
    status_entries: list[str] = []
    changed_hashes: dict[str, str] = {}
    index = 0
    while index < len(fields):
        entry = fields[index]
        index += 1
        status_entries.append(entry)
        path_text = entry[3:]
        if entry[:1] in {"R", "C"} and index < len(fields):
            status_entries.append(fields[index])
            index += 1
        candidate = REPO_ROOT / path_text
        if candidate.is_file():
            changed_hashes[path_text] = _sha256_file(candidate)
    return {
        "commit": run("rev-parse", "HEAD").strip(),
        "status": status_entries,
        "changed_file_sha256": changed_hashes,
    }


def _simsoptpp_identity() -> dict[str, str]:
    import importlib.util

    specification = importlib.util.find_spec("simsoptpp")
    if specification is None or specification.origin is None:
        raise BenchmarkError("simsoptpp extension is unavailable")
    origin = Path(specification.origin).resolve()
    return {"path": str(origin), "sha256": _sha256_file(origin)}


def _threading_identity() -> dict[str, object]:
    """Every numerical-threading variable actually visible to this process."""
    return {
        "environment": {
            name: value
            for name, value in sorted(os.environ.items())
            if name.startswith(_NUMERICAL_ENVIRONMENT_PREFIXES)
        },
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "cpu_count": os.cpu_count(),
    }


def _gpu_compute_processes() -> tuple[dict[str, object], ...]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise BenchmarkError(f"nvidia-smi query failed: {completed.stderr.strip()}")
    rows: list[dict[str, object]] = []
    for line in completed.stdout.strip().splitlines():
        pid_text, memory_text = (field.strip() for field in line.split(","))
        pid = int(pid_text)
        name = subprocess.run(
            ["ps", "-o", "comm=", "-p", str(pid)],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        rows.append({"pid": pid, "used_memory_mib": int(memory_text), "name": name})
    return tuple(rows)


def _gpu_identity() -> dict[str, str]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise BenchmarkError(
            f"nvidia-smi device query failed: {completed.stderr.strip()}"
        )
    name, uuid, driver_version, memory_total = (
        field.strip() for field in completed.stdout.strip().splitlines()[0].split(",")
    )
    return {
        "name": name,
        "uuid": uuid,
        "driver_version": driver_version,
        "memory_total": memory_total,
    }


def _cache_digest(root: Path) -> str:
    """Full content digest of a compilation cache root (orchestrator-only).

    This walks and hashes every cache entry, so it must never run inside a
    timed leg; the orchestrator samples it around the subprocess instead.
    """
    if not root.is_dir():
        return "absent"
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digest.update(str(path.relative_to(root)).encode("utf-8"))
            digest.update(_sha256_file(path).encode("utf-8"))
    return digest.hexdigest()


def _cache_state(root: Path | None) -> dict[str, int]:
    """Cheap in-leg cache evidence: entry count and total bytes, no hashing."""
    if root is None or not root.is_dir():
        return {"files": 0, "bytes": 0}
    files = 0
    total_bytes = 0
    for path in root.rglob("*"):
        if path.is_file():
            files += 1
            total_bytes += path.stat().st_size
    return {"files": files, "bytes": total_bytes}


def _runtime_identity(*, lane: str) -> dict[str, object]:
    identity: dict[str, object] = {
        "lane": lane,
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "git": _git_identity(),
        "threading": _threading_identity(),
        "timestamp_ns": time.time_ns(),
    }
    import numpy
    import scipy

    identity["numpy_version"] = numpy.__version__
    identity["scipy_version"] = scipy.__version__
    identity["simsoptpp"] = _simsoptpp_identity()
    # ``simsopt.geo`` objectives are jax-jitted, so a "native" leg can still
    # import JAX transitively.  Record whether it did and which backend it
    # resolved to, in both lanes: a native leg that quietly initialized CUDA
    # is not a native measurement.
    identity["jax_imported"] = "jax" in sys.modules
    if identity["jax_imported"]:
        import jax

        identity["jax_default_backend"] = str(jax.default_backend())
        # fp64 state is part of both lanes' physics specification: the
        # 2026-08-17 taint showed a native leg whose transitive JAX ran
        # float32 publishes gradients that are not the declared physics.
        identity["jax_enable_x64"] = bool(jax.config.read("jax_enable_x64"))
    # nvidia-smi is lane-agnostic: a native leg sharing the box with GPU
    # compute is as compromised as a GPU leg would be.
    identity["gpu_compute_processes"] = _gpu_compute_processes()
    if lane == "jax":
        import jax
        import jaxlib

        devices = jax.local_devices()
        identity["jax_version"] = jax.__version__
        identity["jaxlib_version"] = jaxlib.__version__
        identity["jax_platforms"] = os.environ.get("JAX_PLATFORMS")
        identity["xla_flags"] = os.environ.get("XLA_FLAGS")
        identity["devices"] = [
            {"platform": device.platform, "kind": device.device_kind}
            for device in devices
        ]
        identity["gpu"] = _gpu_identity()
    return identity


def _peak_memory(*, lane: str) -> dict[str, object]:
    peak: dict[str, object] = {
        "host_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    }
    if lane == "jax":
        import jax

        statistics_map = jax.local_devices()[0].memory_stats() or {}
        peak["gpu_peak_bytes_in_use"] = statistics_map.get("peak_bytes_in_use")
    return peak


# ---------------------------------------------------------------------------
# Frozen problem construction (single physics specification)
# ---------------------------------------------------------------------------


def _parity_case():
    from examples.jax.parity.cases import (
        native_stage_two_optimization_finitebuild as case,
    )

    return case


def _frozen_bundle(scratch: Path):
    from examples.jax.parity.input_bundle import load_input_bundle

    case = _parity_case()
    bundle = case.create_input(scratch, SCALE)
    _, arrays = load_input_bundle(scratch, bundle)
    return bundle, arrays


def _initial_parameters(arrays: Mapping[str, object]):
    import numpy as np

    return np.asarray(arrays["initial_parameters"], dtype=np.float64)


def _baseline_states(arrays: Mapping[str, object]) -> tuple[tuple[str, object], ...]:
    import numpy as np

    initial = _initial_parameters(arrays)
    direction = np.asarray(arrays["taylor_direction"], dtype=np.float64)
    states = [("initial", initial)]
    for label, scale in zip(
        BASELINE_STATE_LABELS[1:], BASELINE_PERTURBATION_SCALES, strict=True
    ):
        states.append((label, initial + scale * direction))
    return tuple(states)


def _native_context(bundle):
    import numpy as np

    case = _parity_case()
    evaluator = case.build_native_evaluator(bundle)
    objective_scale = float(bundle.configuration["objective_scale"])

    def unscaled_state(parameters: np.ndarray) -> dict[str, object]:
        evaluator.objective.x = parameters
        squared_flux = float(evaluator.flux.J())
        length_penalty = float(evaluator.length_term.J())
        distance_penalty = float(evaluator.distance_term.J())
        gradient = np.asarray(evaluator.objective.dJ(), dtype=np.float64)
        return {
            "objective": squared_flux + length_penalty + distance_penalty,
            "gradient": gradient.tolist(),
            "gradient_inf_norm": float(np.max(np.abs(gradient))),
            "squared_flux": squared_flux,
            "length_penalty": length_penalty,
            "distance_penalty": distance_penalty,
            "minimum_clearance": float(evaluator.distance.shortest_distance()),
            "coil_lengths": [float(length.J()) for length in evaluator.lengths],
        }

    def scaled_value_and_grad(parameters: np.ndarray):
        evaluator.objective.x = parameters
        return (
            objective_scale * float(evaluator.objective.J()),
            objective_scale * np.asarray(evaluator.objective.dJ(), dtype=np.float64),
        )

    return evaluator, unscaled_state, scaled_value_and_grad


def _jax_objective_parts(bundle):
    from simsopt_jax.backend.runtime import get_runtime_jax_device
    from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
    from simsopt_jax_adapters.objectives import (
        finite_build_stage_two_diagnostics,
        make_finite_build_stage_two_objective,
    )
    from simsopt_jax_adapters.objectives.flux import SquaredFluxJAX

    case = _parity_case()
    surface, _, _, coils, config = case._build_geometry(bundle.configuration)
    field = BiotSavartJAX(coils)
    flux = SquaredFluxJAX(surface, field)
    # One flux specification feeds both programs: deriving it twice would
    # stage two independent copies of the same frozen surface arrays.
    flux_spec = flux.fixed_surface_flux_spec()
    objective = make_finite_build_stage_two_objective(field, flux_spec, config)
    diagnostics = finite_build_stage_two_diagnostics(field, flux_spec, config)
    return objective, diagnostics, get_runtime_jax_device()


def _jax_context(bundle, arrays):
    import jax
    import numpy as np
    from simsopt_jax.solve.serial import (
        TraceableArrayFunction,
        TraceableParametricScalarProblem,
    )

    objective, diagnostics, device = _jax_objective_parts(bundle)
    objective_scale = float(bundle.configuration["objective_scale"])

    def scaled_objective(parameters: jax.Array, scale: jax.Array) -> jax.Array:
        return scale * objective(parameters)

    def device_scalar(value: float) -> jax.Array:
        return jax.device_put(np.asarray(value, dtype=np.float64), device)

    initial_device = jax.device_put(
        np.asarray(arrays["initial_parameters"], dtype=np.float64), device
    )
    problem = TraceableParametricScalarProblem(
        objective_fn=scaled_objective,
        objective_parameter=device_scalar(objective_scale),
        x=initial_device,
    )
    diagnostics_program = TraceableArrayFunction(diagnostics, initial_device)
    return {
        "problem": problem,
        "diagnostics_program": diagnostics_program,
        "device": device,
        "device_scalar": device_scalar,
        "objective_scale": objective_scale,
        "initial_device": initial_device,
    }


def _jax_unscaled_state(context, parameters) -> dict[str, object]:
    import jax
    import numpy as np
    from simsopt_jax.runtime.host_boundary import host_array
    from simsopt_jax_adapters.objectives.finite_build_stage_two import (
        FINITE_BUILD_DIAGNOSTIC_FIELDS,
    )

    problem = context["problem"]
    problem.set_objective_parameter(context["device_scalar"](1.0))
    value, gradient = problem.value_and_grad(parameters)
    diagnostic_values = context["diagnostics_program"](parameters)
    jax.block_until_ready((value, gradient, diagnostic_values))
    published_value = float(host_array(value))
    published_gradient = np.asarray(host_array(gradient), dtype=np.float64)
    published_diagnostics = np.asarray(host_array(diagnostic_values), dtype=np.float64)
    problem.set_objective_parameter(
        context["device_scalar"](context["objective_scale"])
    )
    field_count = len(FINITE_BUILD_DIAGNOSTIC_FIELDS)
    packed_fields = dict(
        zip(
            FINITE_BUILD_DIAGNOSTIC_FIELDS,
            (float(value) for value in published_diagnostics[:field_count]),
            strict=True,
        )
    )
    return {
        "objective": published_value,
        "gradient": published_gradient.tolist(),
        "gradient_inf_norm": float(np.max(np.abs(published_gradient))),
        **packed_fields,
        "coil_lengths": published_diagnostics[field_count:].tolist(),
    }


def _bundle_fingerprints(bundle) -> dict[str, str]:
    return {
        "input_fingerprint": bundle.input_fingerprint,
        "configuration_fingerprint": bundle.configuration_fingerprint,
    }


# ---------------------------------------------------------------------------
# Leg implementations (each runs inside a pinned subprocess)
# ---------------------------------------------------------------------------


def _leg_native_eval(specification: Mapping[str, object]) -> dict[str, object]:
    scratch = Path(str(specification["scratch"]))
    bundle, arrays = _frozen_bundle(scratch)
    _, unscaled_state, _ = _native_context(bundle)
    states = {
        label: unscaled_state(parameters)
        for label, parameters in _baseline_states(arrays)
    }
    return {"fingerprints": _bundle_fingerprints(bundle), "states": states}


def _leg_jax_eval(specification: Mapping[str, object]) -> dict[str, object]:
    scratch = Path(str(specification["scratch"]))
    bundle, arrays = _frozen_bundle(scratch)
    context = _jax_context(bundle, arrays)
    import jax

    states = {}
    for label, parameters in _baseline_states(arrays):
        device_parameters = jax.device_put(parameters, context["device"])
        states[label] = _jax_unscaled_state(context, device_parameters)
    return {"fingerprints": _bundle_fingerprints(bundle), "states": states}


def _leg_native_value_grad(specification: Mapping[str, object]) -> dict[str, object]:
    scratch = Path(str(specification["scratch"]))
    construction_start = time.perf_counter()
    bundle, arrays = _frozen_bundle(scratch)
    _, _, scaled_value_and_grad = _native_context(bundle)
    initial = _initial_parameters(arrays)
    construction_seconds = time.perf_counter() - construction_start

    first_start = time.perf_counter()
    scaled_value_and_grad(initial)
    first_execute_seconds = time.perf_counter() - first_start

    warm_seconds = []
    for _ in range(int(specification["warm_calls"])):
        warm_start = time.perf_counter()
        scaled_value_and_grad(initial)
        warm_seconds.append(time.perf_counter() - warm_start)
    return {
        "fingerprints": _bundle_fingerprints(bundle),
        "timings": {
            "construction_seconds": construction_seconds,
            "first_execute_seconds": first_execute_seconds,
            "warm_value_grad_seconds": warm_seconds,
        },
    }


def _leg_jax_value_grad(specification: Mapping[str, object]) -> dict[str, object]:
    import jax

    scratch = Path(str(specification["scratch"]))
    construction_start = time.perf_counter()
    bundle, arrays = _frozen_bundle(scratch)
    context = _jax_context(bundle, arrays)
    problem = context["problem"]
    initial = context["initial_device"]
    jax.block_until_ready(initial)
    construction_seconds = time.perf_counter() - construction_start

    first_start = time.perf_counter()
    jax.block_until_ready(problem.value_and_grad(initial))
    first_execute_seconds = time.perf_counter() - first_start

    warm_seconds = []
    for _ in range(int(specification["warm_calls"])):
        warm_start = time.perf_counter()
        jax.block_until_ready(problem.value_and_grad(initial))
        warm_seconds.append(time.perf_counter() - warm_start)
    return {
        "fingerprints": _bundle_fingerprints(bundle),
        "timings": {
            "construction_seconds": construction_seconds,
            "first_execute_seconds": first_execute_seconds,
            "warm_value_grad_seconds": warm_seconds,
        },
    }


def _leg_native_solve(specification: Mapping[str, object]) -> dict[str, object]:
    import numpy as np
    from scipy.optimize import minimize

    scratch = Path(str(specification["scratch"]))
    max_steps = int(specification["max_steps"])
    history = int(specification["history"])
    record_trace = bool(specification.get("record_trace", False))
    # ``stop_at_scaled_target``: terminate the solve through ``StopIteration``
    # at the first accepted iterate whose scaled objective clears the frozen
    # rung, so the solve wall IS the time to that quality.  Measured 2026-08-17:
    # native solves at OMP>1 are not reproducible across processes (~1% endpoint
    # fork at 400 iterations from OpenMP reduction arrival order), so a budgeted
    # replay of another process's crossing iteration measures a different
    # trajectory; each solve must find its own crossing.
    raw_stop_target = specification.get("stop_at_scaled_target")
    stop_target = None if raw_stop_target is None else float(raw_stop_target)
    # ``derive_gate_anchor``: keep every accepted iterate so the gate's
    # truncated anchor comes from this run's own trajectory rather than a
    # replay that would land somewhere else.
    derive_gate_anchor = bool(specification.get("derive_gate_anchor", False))
    # ``warm_protocol`` discards one full solve before the timed ones so the
    # native lane is measured under the same warmed-process protocol as the
    # JAX lane; without it the timed solve is the process's first solve.
    warm_protocol = bool(specification.get("warm_protocol", False))
    # Timed solves after the discarded one, matching the JAX lane's field of
    # the same name so the warm statistic is a median of the same count in
    # both lanes.  A wall leg leaves it at 1: its subprocess wall is the
    # metric, and repeating the solve inside it would corrupt that wall.
    warm_repetitions = int(specification.get("warm_repetitions", 1))
    if warm_repetitions < 1:
        raise BenchmarkError(
            "native-solve legs publish their endpoint from a timed solve; "
            f"warm_repetitions={warm_repetitions} would leave none"
        )

    construction_start = time.perf_counter()
    bundle, arrays = _frozen_bundle(scratch)
    _, unscaled_state, scaled_value_and_grad = _native_context(bundle)
    initial = _initial_parameters(arrays)
    construction_seconds = time.perf_counter() - construction_start

    initial_state = unscaled_state(initial)
    trace: list[float] = []
    iterates: list = []
    stop_state = None

    def observe(intermediate_result) -> None:
        nonlocal stop_state
        value = float(intermediate_result.fun)
        if record_trace or derive_gate_anchor:
            trace.append(value)
        if derive_gate_anchor:
            iterates.append(np.array(intermediate_result.x, dtype=np.float64))
        if stop_target is not None and value <= stop_target:
            stop_state = np.array(intermediate_result.x, dtype=np.float64)
            raise StopIteration

    callback = (
        observe
        if (record_trace or derive_gate_anchor or stop_target is not None)
        else None
    )

    def solve():
        return minimize(
            scaled_value_and_grad,
            initial,
            jac=True,
            method="L-BFGS-B",
            callback=callback,
            options={
                "maxiter": max_steps,
                "maxcor": history,
                "gtol": 1.0e-20,
                "ftol": 1.0e-20,
            },
            tol=1.0e-20,
        )

    first_execute_seconds = None
    if warm_protocol:
        # The discarded solve runs the same callback, so a stop leg's warm-up
        # stops at its own crossing exactly like the timed solves after it.
        first_start = time.perf_counter()
        solve()
        first_execute_seconds = time.perf_counter() - first_start

    # A list, matching the JAX lane's repetition list so every consumer takes
    # a median of the same type at the same count.
    warm_solve_seconds = []
    for _ in range(warm_repetitions):
        # The published trace and stop state belong to the last timed solve,
        # not to their concatenation; a traced leg is single-repetition anyway.
        trace.clear()
        iterates.clear()
        stop_state = None
        solve_start = time.perf_counter()
        result = solve()
        warm_solve_seconds.append(time.perf_counter() - solve_start)

    # scipy 1.17.1, measured on this host: a ``StopIteration`` callback stop
    # returns ``status=99``, ``success=False``, ``message='`callback` raised
    # `StopIteration`.'``, ``nit`` = the accepted iterations completed, and
    # ``x``/``fun`` = exactly the stopping iterate.  The endpoint is published
    # from ``result.x`` and this guard pins that identity.
    if stop_state is not None and not np.array_equal(
        np.asarray(result.x, dtype=np.float64), stop_state
    ):
        raise BenchmarkError(
            "the stopped solve did not return its stopping iterate; the "
            "endpoint would not be the measured time-to-quality state"
        )

    publication_start = time.perf_counter()
    endpoint = unscaled_state(np.asarray(result.x, dtype=np.float64))
    endpoint_publication_seconds = time.perf_counter() - publication_start
    row: dict[str, object] = {
        "fingerprints": _bundle_fingerprints(bundle),
        "solver": {
            "driver": "scipy_lbfgsb_finite_build",
            "max_steps": max_steps,
            "history": history,
            "objective_scale": float(bundle.configuration["objective_scale"]),
            "status": int(result.status),
            "success": bool(result.success),
            "nit": int(result.nit),
            "nfev": int(result.nfev),
            "stopped_at_target": stop_state is not None,
        },
        "initial_parameters": initial.tolist(),
        "initial_state": initial_state,
        "endpoint": {"solution": np.asarray(result.x).tolist(), **endpoint},
        "scaled_objective_trace": trace if record_trace else None,
        "timings": {
            "construction_seconds": construction_seconds,
            "first_execute_seconds": first_execute_seconds,
            "warm_solve_seconds": warm_solve_seconds,
            "endpoint_publication_seconds": endpoint_publication_seconds,
        },
    }
    if derive_gate_anchor:
        row.update(
            _gate_anchor(
                trace,
                iterates,
                unscaled_state,
                converged_objective=float(endpoint["objective"]),
                objective_scale=float(bundle.configuration["objective_scale"]),
                anchor_margin=float(specification["anchor_margin"]),
            )
        )
    return row


def _gate_anchor(
    trace: Sequence[float],
    iterates: Sequence,
    unscaled_state,
    *,
    converged_objective: float,
    objective_scale: float,
    anchor_margin: float,
) -> dict[str, object]:
    """Evaluate the gate anchor on the converged run's own trajectory.

    The anchor is the first accepted iterate of this very solve whose scaled
    objective clears ``anchor_margin`` times the converged objective.  It is
    taken from the captured iterate rather than from a truncated replay: a
    replay is a different process, and native solves at ``OMP>1`` fork by ~1%
    across processes.  Untimed.
    """
    anchor_budget = first_qualifying_iteration(
        trace, objective_scale * anchor_margin * converged_objective
    )
    if anchor_budget is None:
        raise BenchmarkError(
            "the converged reference trajectory never cleared its own target "
            "objective; the gate anchor cannot be derived"
        )
    anchor_iterate = iterates[anchor_budget - 1]
    return {
        "anchor_budget": anchor_budget,
        "anchor_endpoint": {
            "solution": anchor_iterate.tolist(),
            **unscaled_state(anchor_iterate),
        },
    }


def _leg_native_endpoint_eval(
    specification: Mapping[str, object],
) -> dict[str, object]:
    """Re-evaluate a foreign lane's published solution on the native oracle.

    ``parameters`` is the subject leg's endpoint solution vector and
    ``subject_leg_id`` names that leg.  Untimed: this is the independent
    scientific evaluation the gate clauses are applied to, never a
    measurement.
    """
    import numpy as np

    scratch = Path(str(specification["scratch"]))
    parameters = np.asarray(specification["parameters"], dtype=np.float64)
    bundle, _ = _frozen_bundle(scratch)
    _, unscaled_state, _ = _native_context(bundle)
    return {
        "fingerprints": _bundle_fingerprints(bundle),
        "subject_leg_id": str(specification["subject_leg_id"]),
        "endpoint": {"solution": parameters.tolist(), **unscaled_state(parameters)},
    }


def _leg_jax_solve(specification: Mapping[str, object]) -> dict[str, object]:
    import jax
    import numpy as np
    from simsopt_jax.solve.dispatch import minimize
    from simsopt_jax.solve.driver import Driver
    from simsopt_jax.solve.simsopt.contracts import SimsoptLBFGSBOptions

    scratch = Path(str(specification["scratch"]))
    max_steps = int(specification["max_steps"])
    history = int(specification["history"])
    warm_repetitions = int(specification.get("warm_repetitions", 1))

    construction_start = time.perf_counter()
    bundle, arrays = _frozen_bundle(scratch)
    context = _jax_context(bundle, arrays)
    problem = context["problem"]
    initial = context["initial_device"]
    jax.block_until_ready(initial)
    construction_seconds = time.perf_counter() - construction_start

    initial_state = _jax_unscaled_state(context, initial)
    options = SimsoptLBFGSBOptions(
        maxiter=max_steps,
        maxfun=max_steps * 20,
        gtol=float(bundle.configuration["atol"]),
        ftol=float(bundle.configuration["rtol"]),
        maxcor=history,
    )

    # The cache-marked solver callable, exactly as serial_solve_jax passes it:
    # the public bound method is unmarked, so the fused L-BFGS executable
    # cache would re-trace on every solve and "warm" would include lowering.
    solver_value_and_grad = problem._solver_value_and_grad_fn

    def solve():
        return minimize(
            solver_value_and_grad,
            initial,
            driver=Driver.SIMSOPT_LBFGSB,
            options=options,
        )

    first_start = time.perf_counter()
    result = solve()
    first_execute_seconds = time.perf_counter() - first_start

    warm_solve_seconds = []
    for _ in range(warm_repetitions):
        warm_start = time.perf_counter()
        result = solve()
        warm_solve_seconds.append(time.perf_counter() - warm_start)

    publication_start = time.perf_counter()
    solution_host = np.asarray(result.x, dtype=np.float64)
    solution_device = jax.device_put(solution_host, context["device"])
    endpoint = _jax_unscaled_state(context, solution_device)
    endpoint_publication_seconds = time.perf_counter() - publication_start
    return {
        "fingerprints": _bundle_fingerprints(bundle),
        "solver": {
            "driver": "simsopt_lbfgsb",
            "max_steps": max_steps,
            "history": history,
            "objective_scale": float(bundle.configuration["objective_scale"]),
            "status": int(result.status),
            "success": bool(result.success),
            "nit": int(result.nit),
            "nfev": int(result.nfev),
        },
        "initial_state": initial_state,
        "endpoint": {"solution": solution_host.tolist(), **endpoint},
        "timings": {
            "construction_seconds": construction_seconds,
            "first_execute_seconds": first_execute_seconds,
            "warm_solve_seconds": warm_solve_seconds,
            "endpoint_publication_seconds": endpoint_publication_seconds,
        },
    }


_HLO_OPCODE_PATTERN = re.compile(r"= .*?([a-z][a-z0-9_\-]*)\(")

_OBJECTIVE_SOURCE = "src/simsopt_jax_adapters/objectives/finite_build_stage_two.py"
_PARITY_CASE_SOURCE = (
    "examples/jax/parity/cases/native_stage_two_optimization_finitebuild.py"
)
_BENCHMARK_SOURCE = "benchmarks/stage_two_finitebuild_native_gpu.py"
# The plan is the preregistration: every frozen constant, ladder, and kill
# criterion above is quoted from it, so a gate that does not bind the plan's
# bytes cannot prove which criteria it was frozen under.
_PLAN_SOURCE = "docs/jax_gpu_finitebuild_native_speed_implementation_plan.md"


def hlo_operation_census(optimized_hlo: str) -> dict[str, int]:
    """Count optimized-HLO operations by opcode, ignoring instruction ids."""
    census: dict[str, int] = {}
    for line in optimized_hlo.splitlines():
        match = _HLO_OPCODE_PATTERN.search(line)
        if match is not None:
            opcode = match.group(1)
            census[opcode] = census.get(opcode, 0) + 1
    return census


def _leg_jax_hlo(specification: Mapping[str, object]) -> dict[str, object]:
    import jax

    scratch = Path(str(specification["scratch"]))
    artifact_directory = Path(str(specification["artifact_dir"]))
    bundle, arrays = _frozen_bundle(scratch)
    objective, _, device = _jax_objective_parts(bundle)
    initial_device = jax.device_put(_initial_parameters(arrays), device)
    # The plan's plain-jit instrument embeds device-resident closure constants,
    # and lowering materializes them host-side — an implicit D2H the strict
    # guard rejects.  Artifact generation is not a timed boundary, so allow it
    # for exactly this scope; the production solve path closure-converts and
    # never needs this.
    with jax.transfer_guard("allow"):
        lowered = jax.jit(jax.value_and_grad(objective)).lower(initial_device)
        stablehlo_text = lowered.as_text()
        compiled = lowered.compile()
        optimized_text = compiled.as_text()
    cost_analysis = {
        str(name): float(value)
        for name, value in dict(compiled.cost_analysis()).items()
        if isinstance(value, (int, float))
    }
    artifact_directory.mkdir(parents=True, exist_ok=True)
    (artifact_directory / "stablehlo.txt").write_text(stablehlo_text)
    (artifact_directory / "optimized_hlo.txt").write_text(optimized_text)
    objective_source = REPO_ROOT / _OBJECTIVE_SOURCE
    return {
        "fingerprints": _bundle_fingerprints(bundle),
        "objective_source": {
            "path": _OBJECTIVE_SOURCE,
            "sha256": _sha256_file(objective_source),
        },
        "artifacts": {
            "stablehlo_sha256": _sha256_bytes(stablehlo_text.encode("utf-8")),
            "optimized_hlo_sha256": _sha256_bytes(optimized_text.encode("utf-8")),
        },
        "census": hlo_operation_census(optimized_text),
        "cost_analysis": cost_analysis,
    }


_LEG_IMPLEMENTATIONS = {
    "native-eval": ("native", _leg_native_eval),
    "jax-eval": ("jax", _leg_jax_eval),
    "native-value-grad": ("native", _leg_native_value_grad),
    "jax-value-grad": ("jax", _leg_jax_value_grad),
    "native-solve": ("native", _leg_native_solve),
    "native-endpoint-eval": ("native", _leg_native_endpoint_eval),
    "jax-solve": ("jax", _leg_jax_solve),
    "jax-hlo": ("jax", _leg_jax_hlo),
}


def _run_leg(specification_path: Path, row_path: Path) -> None:
    specification = _read_json(specification_path)
    if not isinstance(specification, dict):
        raise BenchmarkError("leg specification must be a JSON object")
    kind = str(specification["kind"])
    lane, implementation = _LEG_IMPLEMENTATIONS[kind]
    affinity = specification.get("cpu_affinity")
    if affinity is not None:
        os.sched_setaffinity(0, {int(core) for core in affinity})
    cache_root_text = os.environ.get("JAX_COMPILATION_CACHE_DIR")
    cache_root = Path(cache_root_text) if cache_root_text else None
    row: dict[str, object] = {
        "schema": ROW_SCHEMA,
        "leg_id": specification["leg_id"],
        "phase": specification["phase"],
        "kind": kind,
        "lane": lane,
        "specification": specification,
        "gate_sha256": specification.get("gate_sha256"),
    }
    cache_before: dict[str, int] | None = None
    if lane == "jax":
        row["gpu_compute_processes_before"] = _gpu_compute_processes()
        cache_before = _cache_state(cache_root)
    payload = implementation(specification)
    if lane == "jax":
        row["compilation_cache"] = {
            "root": cache_root_text,
            "before": cache_before,
            "after": _cache_state(cache_root),
        }
    row.update(
        {
            "identity": _runtime_identity(lane=lane),
            "peak_memory": _peak_memory(lane=lane),
            **payload,
        }
    )
    _write_json_exclusive(row_path, row)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _new_run_directory(phase: str) -> Path:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_directory = ARTIFACT_ROOT / f"{stamp}-{phase}-{os.getpid()}"
    run_directory.mkdir(parents=True, exist_ok=False)
    (run_directory / "rows").mkdir()
    (run_directory / "specs").mkdir()
    return run_directory


def _pinned_threading_environment(omp_threads: int) -> dict[str, str]:
    """The explicit replacement pins written after the prefix scrub."""
    thread_count = str(omp_threads)
    return {
        "OMP_NUM_THREADS": thread_count,
        "OMP_DYNAMIC": "FALSE",
        "OMP_SCHEDULE": "STATIC",
        "OPENBLAS_NUM_THREADS": thread_count,
        "MKL_NUM_THREADS": thread_count,
        "VECLIB_MAXIMUM_THREADS": thread_count,
        "NUMEXPR_NUM_THREADS": thread_count,
    }


def _leg_environment(
    lane: str, *, omp_threads: int | None, cache_root: Path | None
) -> dict[str, str]:
    """Inherit the shell (loader paths) but pin every load-bearing variable.

    ``omp_threads=None`` publishes the shipped default: the inherited
    numerical-threading configuration is still scrubbed, but nothing is pinned
    in its place, so the runtime picks its own thread count.  Only the untimed
    disclosure lane uses it.
    """
    environment = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(_SCRUBBED_ENVIRONMENT_PREFIXES)
    }
    environment.update(
        {
            "PYTHONPATH": f"{REPO_ROOT}:{REPO_ROOT / 'src'}",
            # Every leg is single-rank by design; with this pin MPI_Init
            # never runs and any future communicator use aborts loudly.
            "MPI4PY_RC_INITIALIZE": "false",
            # Dormant while the mpi4py pin is set (MPI_Init never executes);
            # load-bearing the moment a future MPI leg drops that pin: during
            # MPI_Init, hwloc's ``gl`` plugin connects to X displays, and this
            # box has a permanently wedged Xwayland listener that blocks such
            # connects forever in state S (measured 2026-08-17) — a hang a
            # wall-clock leg would misreport as a timeout.  Enabling MPI is
            # therefore a two-line spec edit: drop the mpi4py pin, keep this.
            "HWLOC_COMPONENTS": "-gl",
        }
    )
    if omp_threads is not None:
        environment.update(_pinned_threading_environment(omp_threads))
    if lane == "jax":
        environment.update(_JAX_ENVIRONMENT)
        if cache_root is None:
            raise BenchmarkError("JAX legs require an explicit compilation cache root")
        environment["JAX_COMPILATION_CACHE_DIR"] = str(cache_root)
    else:
        # ``simsopt.geo`` objectives are jax-jitted, so a native leg imports
        # JAX transitively; without this pin that import can initialize CUDA
        # and put a second context on the device a GPU leg is measured on.
        environment["JAX_PLATFORMS"] = "cpu"
        # The same transitive import defaults to float32, and the prefix scrub
        # above removes any inherited x64 setting.  Unpinned, the jax-jitted
        # pieces of the native objective evaluate in fp32: measured 2026-08-17
        # at the jax-sweep h20-b400 endpoint, the native gradient forked from
        # the GPU lane's self-report by up to 2.6e-6 per component (8.9% of
        # the inf norm, DOF block 38-68) while J moved only 3e-13; FD
        # arbitration convicted the native value (see
        # docs/jax_gpu_finitebuild_fp64_taint_diagnostic.md).  fp64 is part
        # of the native lane's physics specification, not a tuning knob.
        environment["JAX_ENABLE_X64"] = "1"
    return environment


def _native_affinity_fields(omp_threads: int) -> dict[str, object]:
    """Affinity spec fields for a native leg at ``omp_threads``.

    This box is SMT-2: ``os.cpu_count()`` counts 64 logical CPUs over 32
    physical cores.  Forcing ``range(48)`` would double-book 16 physical
    cores against their own siblings while leaving 16 cores idle, which
    penalizes exactly the wide configurations; the 2026-08-16 GSCO receipts'
    best native lane at OMP=48 ran with the full inherited mask.  So pin only
    while the request fits in distinct physical cores.
    """
    physical_cores = (os.cpu_count() or 2) // 2
    if omp_threads <= physical_cores:
        return {
            "cpu_affinity": list(range(omp_threads)),
            "affinity_policy": "pinned-physical",
        }
    return {"affinity_policy": "unpinned-smt-overflow"}


# Rows are published evidence: record only the load-bearing variables the
# benchmark itself pins.  Never serialize the inherited shell environment —
# it carries unrelated credentials.
_ATTESTED_ENVIRONMENT_NAMES = frozenset(
    {
        "PYTHONPATH",
        "MPI4PY_RC_INITIALIZE",
        "CUDA_VISIBLE_DEVICES",
        "HWLOC_COMPONENTS",
    }
)


def _attested_environment(environment: Mapping[str, str]) -> dict[str, str]:
    # Everything under a scrubbed prefix was set by this benchmark, so
    # attesting the whole prefix cannot leak an inherited secret.
    return {
        name: value
        for name, value in environment.items()
        if name in _ATTESTED_ENVIRONMENT_NAMES
        or name.startswith(_SCRUBBED_ENVIRONMENT_PREFIXES)
    }


# Timed legs demand a quiet box: competing CPU load skews native wall time
# exactly as competing GPU compute skews device legs.  The CPU gate is a
# 2-second /proc/stat busy-fraction delta (1-minute loadavg was ruled too
# slow by the 2026-08-16 GSCO campaigns); the GPU gate is an at-phase-start
# pid allowlist (name matching is the disclosed 2026-08-16 defect) plus a
# sampled utilization ceiling.
TIMED_LEG_MAX_CPU_BUSY_FRACTION = 0.15
CPU_BUSY_SAMPLE_SECONDS = 2.0
GPU_IDLE_MAX_UTILIZATION_PERCENT = 5.0


def _proc_stat_totals() -> tuple[int, int]:
    with open("/proc/stat", encoding="ascii") as handle:
        fields = [int(value) for value in handle.readline().split()[1:]]
    idle = fields[3] + fields[4]
    return sum(fields), idle


def _cpu_busy_fraction() -> float:
    total_before, idle_before = _proc_stat_totals()
    time.sleep(CPU_BUSY_SAMPLE_SECONDS)
    total_after, idle_after = _proc_stat_totals()
    total_delta = total_after - total_before
    if total_delta <= 0:
        raise BenchmarkError("/proc/stat did not advance during the CPU sample")
    return 1.0 - (idle_after - idle_before) / total_delta


def _require_quiet_cpu() -> None:
    busy_fraction = _cpu_busy_fraction()
    if busy_fraction > TIMED_LEG_MAX_CPU_BUSY_FRACTION:
        raise BenchmarkError(
            f"host CPU busy fraction {busy_fraction:.2f} exceeds "
            f"{TIMED_LEG_MAX_CPU_BUSY_FRACTION}; rerun when the box is quiet "
            f"(loadavg {os.getloadavg()[0]:.1f})"
        )


def _gpu_utilization_percent() -> float:
    samples = []
    for _ in range(3):
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise BenchmarkError(
                f"nvidia-smi utilization query failed: {completed.stderr.strip()}"
            )
        samples.append(float(completed.stdout.strip().splitlines()[0]))
        time.sleep(0.2)
    return _median(samples)


def _gpu_pid_allowlist() -> tuple[int, ...]:
    """Compute contexts resident at phase start; anything new is a competitor."""
    return tuple(int(process["pid"]) for process in _gpu_compute_processes())


def _require_idle_gpu(pid_allowlist: Sequence[int]) -> None:
    competing = tuple(
        process
        for process in _gpu_compute_processes()
        if int(process["pid"]) not in set(pid_allowlist)
    )
    if competing:
        raise BenchmarkError(
            f"GPU compute processes appeared after phase start: {competing}"
        )
    utilization = _gpu_utilization_percent()
    if utilization > GPU_IDLE_MAX_UTILIZATION_PERCENT:
        raise BenchmarkError(
            f"GPU utilization {utilization:.0f}% exceeds "
            f"{GPU_IDLE_MAX_UTILIZATION_PERCENT}%; a resident context is busy"
        )


# The worst-case leg is a warm-native final pair: one discarded plus
# FINAL_WARM_REPETITIONS timed stop-solves, each capped at
# NATIVE_STOP_MAX_STEPS iterations.  1.5 s/iteration is ~10x the measured
# OMP=8 rate (~150 ms), covering the narrow end of the OMP sweep and load;
# deriving the timeout from the cap keeps a future cap change from silently
# crossing it.  A leg that exceeds this has hung, and hanging forever would
# silently strand the whole phase.
LEG_TIMEOUT_SECONDS = int(1.5 * (1 + FINAL_WARM_REPETITIONS) * NATIVE_STOP_MAX_STEPS)


def _execute_leg(
    run_directory: Path,
    specification: Mapping[str, object],
    environment: Mapping[str, str],
    pid_allowlist: Sequence[int] | None = None,
) -> None:
    leg_id = str(specification["leg_id"])
    specification_path = run_directory / "specs" / f"{leg_id}.json"
    row_path = run_directory / "rows" / f"{leg_id}.json"
    _write_json_exclusive(specification_path, specification)
    is_jax_leg = str(specification.get("kind", "")).startswith("jax")
    if bool(specification.get("timed", True)):
        _require_quiet_cpu()
        # Lane-symmetric: a native wall or warm leg sharing the box with
        # foreign GPU compute is as compromised as a device leg, because that
        # compute contends for the same host cores and memory bandwidth.
        if pid_allowlist is None:
            raise BenchmarkError(
                f"timed leg {leg_id} launched without a phase-start GPU pid allowlist"
            )
        _require_idle_gpu(pid_allowlist)
    cache_root_text = environment.get("JAX_COMPILATION_CACHE_DIR")
    cache_root = Path(cache_root_text) if cache_root_text else None
    cache_digest_before = (
        _cache_digest(cache_root) if is_jax_leg and cache_root is not None else None
    )
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "leg",
        "--spec",
        str(specification_path),
        "--row",
        str(row_path),
    ]
    launch_start = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=dict(environment),
            capture_output=True,
            text=True,
            check=False,
            timeout=LEG_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as error:
        raise BenchmarkError(
            f"leg {leg_id} exceeded {LEG_TIMEOUT_SECONDS}s and was killed"
        ) from error
    process_wall_seconds = time.perf_counter() - launch_start
    row_produced = row_path.is_file()
    launch_row = {
        "schema": ROW_SCHEMA,
        "leg_id": leg_id,
        "record": "launch",
        "command": command,
        "environment": _attested_environment(environment),
        "gate_sha256": specification.get("gate_sha256"),
        "returncode": completed.returncode,
        "process_wall_seconds": process_wall_seconds,
        "row_produced": row_produced,
        "row_sha256": _sha256_file(row_path) if row_produced else "absent",
        "compilation_cache_digest": {
            "before": cache_digest_before,
            "after": (
                _cache_digest(cache_root)
                if is_jax_leg and cache_root is not None
                else None
            ),
        },
        "stdout_head": completed.stdout[:2000],
        "stdout_tail": completed.stdout[-4000:],
        "stderr_head": completed.stderr[:2000],
        "stderr_tail": completed.stderr[-4000:],
    }
    _write_json_exclusive(run_directory / "rows" / f"{leg_id}.launch.json", launch_row)
    if completed.returncode != 0 or not row_produced:
        raise BenchmarkError(
            f"leg {leg_id} failed (returncode={completed.returncode}); "
            f"stderr tail: {completed.stderr[-2000:]}"
        )


def _publish_manifest(
    run_directory: Path,
    *,
    phase: str,
    expected_leg_ids: Sequence[str],
    gate_sha256: str | None,
    cache_policy: str | None = None,
    aborted_reason: str | None = None,
) -> None:
    rows: dict[str, str] = {}
    for leg_id in expected_leg_ids:
        for suffix in (".json", ".launch.json"):
            relative = f"rows/{leg_id}{suffix}"
            path = run_directory / relative
            rows[relative] = _sha256_file(path) if path.is_file() else "absent"
    gate_path = run_directory / "gate" / "quality_contract.json"
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "phase": phase,
        "run_directory": run_directory.name,
        "expected_legs": list(expected_leg_ids),
        "rows": rows,
        "gate_sha256": gate_sha256,
        "gate_present": gate_path.is_file(),
        "cache_policy": cache_policy,
        # ``None`` on clean completion.  A partial manifest published by the
        # try/finally otherwise says only that legs are missing; this names
        # the exception that stopped the phase, so an abort is never read as
        # a phase that merely produced fewer rows.
        "aborted_reason": aborted_reason,
        "orchestrator_identity": _runtime_identity(lane="native"),
    }
    _replace_json_atomic(run_directory / "manifest.json", manifest)


def _abort_reason(error: BaseException) -> str:
    """The manifest's record of why a phase stopped short."""
    return f"{type(error).__name__}: {error}"


def _bind_gpu_pid_allowlist(run_directory: Path) -> tuple[int, ...]:
    """Freeze the phase-start GPU residents and publish them with the run."""
    # One sample only: a second nvidia-smi call could see a different set and
    # publish an allowlist that does not match the recorded processes.  The
    # named processes are already bound in every row's runtime identity.
    allowlist = _gpu_pid_allowlist()
    _write_json_exclusive(
        run_directory / "gpu_pid_allowlist.json",
        {"pids": list(allowlist), "sampled_at_ns": time.time_ns()},
    )
    return allowlist


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------


def _phase_baseline(arguments: argparse.Namespace) -> Path:
    run_directory = _new_run_directory("baseline")
    pid_allowlist = _bind_gpu_pid_allowlist(run_directory)
    legs: list[str] = []
    aborted_reason: str | None = None
    try:
        for kind, omp in (
            ("native-eval", GATE_REFERENCE_OMP),
            ("jax-eval", GPU_HOST_OMP_THREADS),
        ):
            lane = _LEG_IMPLEMENTATIONS[kind][0]
            leg_id = kind
            specification: dict[str, object] = {
                "kind": kind,
                "leg_id": leg_id,
                "phase": "baseline",
                "scratch": str(run_directory / "scratch" / leg_id),
                "omp_threads": omp,
                "timed": False,
            }
            if lane == "jax":
                specification["cpu_affinity"] = list(range(GPU_HOST_OMP_THREADS))
            else:
                specification.update(_native_affinity_fields(omp))
            cache_root = run_directory / "jax-cache" if lane == "jax" else None
            legs.append(leg_id)
            _execute_leg(
                run_directory,
                specification,
                _leg_environment(lane, omp_threads=omp, cache_root=cache_root),
                pid_allowlist,
            )
    except BaseException as error:
        aborted_reason = _abort_reason(error)
        raise
    finally:
        _publish_manifest(
            run_directory,
            phase="baseline",
            expected_leg_ids=legs,
            gate_sha256=None,
            aborted_reason=aborted_reason,
        )
    return run_directory


def _gate_reference_specification(
    run_directory: Path, leg_id: str
) -> dict[str, object]:
    return {
        "kind": "native-solve",
        "leg_id": leg_id,
        "phase": "gate",
        "role": "gate-reference",
        "scratch": str(run_directory / "scratch" / leg_id),
        "max_steps": GATE_REFERENCE_STEPS,
        "history": GATE_REFERENCE_HISTORY,
        "record_trace": True,
        "derive_gate_anchor": True,
        "anchor_margin": GATE_OBJECTIVE_MARGIN,
        "omp_threads": GATE_REFERENCE_OMP,
        **_native_affinity_fields(GATE_REFERENCE_OMP),
        "timed": False,
    }


def _phase_gate(arguments: argparse.Namespace) -> Path:
    run_directory = _new_run_directory("gate")
    environment = _leg_environment(
        "native", omp_threads=GATE_REFERENCE_OMP, cache_root=None
    )
    reference_leg = "native-reference"
    legs: list[str] = []
    gate_sha256: str | None = None
    aborted_reason: str | None = None
    try:
        legs.append(reference_leg)
        _execute_leg(
            run_directory,
            _gate_reference_specification(run_directory, reference_leg),
            environment,
        )
        reference_row = _read_json(run_directory / "rows" / f"{reference_leg}.json")
        gate = _derive_quality_contract(reference_row)
        gate_sha256 = _write_json_exclusive(
            run_directory / "gate" / "quality_contract.json", gate
        )
    except BaseException as error:
        aborted_reason = _abort_reason(error)
        raise
    finally:
        _publish_manifest(
            run_directory,
            phase="gate",
            expected_leg_ids=legs,
            gate_sha256=gate_sha256,
            aborted_reason=aborted_reason,
        )
    print(f"gate sha256 {gate_sha256}")
    return run_directory


_GATE_ENDPOINT_FIELDS = (
    "solution",
    "objective",
    "gradient",
    "gradient_inf_norm",
    "squared_flux",
    "length_penalty",
    "distance_penalty",
    "minimum_clearance",
    "coil_lengths",
)


def _require_sound_gate_endpoint(label: str, endpoint: Mapping[str, object]) -> None:
    for name in _GATE_ENDPOINT_FIELDS:
        if name not in endpoint:
            raise BenchmarkError(f"gate {label} endpoint lacks {name}")
        if not _all_finite(endpoint[name]):
            raise BenchmarkError(f"gate {label} endpoint {name} is not finite")
    if float(endpoint["minimum_clearance"]) <= 0.0:
        raise BenchmarkError(f"gate {label} endpoint minimum clearance is not positive")


def _source_fingerprints(git_commit: object) -> dict[str, object]:
    return {
        "git_commit": git_commit,
        "objective_module_sha256": _sha256_file(REPO_ROOT / _OBJECTIVE_SOURCE),
        "parity_case_sha256": _sha256_file(REPO_ROOT / _PARITY_CASE_SOURCE),
        "benchmark_sha256": _sha256_file(REPO_ROOT / _BENCHMARK_SOURCE),
        "plan_sha256": _sha256_file(REPO_ROOT / _PLAN_SOURCE),
    }


def _derive_quality_contract(
    reference_row: Mapping[str, object],
) -> dict[str, object]:
    """Freeze the scientific contract from the single converged reference run.

    The converged endpoint sets the target objective; the anchor endpoint --
    the same run's own first accepted iterate to clear that target -- is the
    like-for-like anchor every budget-truncated lane is measured against.
    Both come from one trajectory, so no cross-process reproducibility is
    assumed anywhere in the derivation.
    """
    converged_endpoint = reference_row["endpoint"]
    anchor_endpoint = reference_row["anchor_endpoint"]
    reference_budget = int(reference_row["anchor_budget"])
    _require_sound_gate_endpoint("converged", converged_endpoint)
    _require_sound_gate_endpoint("anchor", anchor_endpoint)
    solver = reference_row["solver"]
    # The reference carries no stopping target and both tolerances are 1e-20,
    # so L-BFGS-B cannot converge out early: a short reference is an abnormal
    # termination, and a gate frozen from one would understate the converged
    # quality every later lane is held to.
    if int(solver["nit"]) != int(solver["max_steps"]):
        raise BenchmarkError(
            f"gate reference stopped at {solver['nit']} of {solver['max_steps']} "
            f"iterations (status {solver.get('status')})"
        )
    initial_state = reference_row["initial_state"]
    if not _finite(initial_state["objective"]):
        raise BenchmarkError("gate initial objective is not finite")
    # Self-consistency: the anchor is the first iterate of this trajectory to
    # clear the target, so it clears it by construction.  A violation is an
    # internal error in the anchor capture, never a measurement outcome.
    target = GATE_OBJECTIVE_MARGIN * float(converged_endpoint["objective"])
    if not float(anchor_endpoint["objective"]) <= target:
        raise BenchmarkError(
            "gate is internally inconsistent: anchor objective "
            f"{anchor_endpoint['objective']} misses the target {target} it was "
            "selected on; the anchor capture is broken"
        )
    return {
        "schema": GATE_SCHEMA,
        "case_id": CASE_ID,
        "fingerprints": reference_row["fingerprints"],
        "initial_parameters": reference_row["initial_parameters"],
        "source_fingerprints": _source_fingerprints(
            reference_row["identity"]["git"]["commit"]
        ),
        "identity": reference_row["identity"],
        "solver": {"converged_reference": solver},
        "initial_objective": initial_state["objective"],
        "initial_state": initial_state,
        "converged_endpoint": converged_endpoint,
        "reference_endpoint": anchor_endpoint,
        "reference_budget": reference_budget,
        "target_objective": GATE_OBJECTIVE_MARGIN
        * float(converged_endpoint["objective"]),
        "tolerances": {
            "objective_margin": GATE_OBJECTIVE_MARGIN,
            "endpoint_rtol": GATE_ENDPOINT_RTOL,
            "endpoint_atol": GATE_ENDPOINT_ATOL,
            "gradient_norm_margin": GATE_GRADIENT_NORM_MARGIN,
            "gradient_norm_floor": GATE_GRADIENT_NORM_FLOOR,
            "quality_cap_margin": GATE_QUALITY_CAP_MARGIN,
        },
    }


def _phase_kernel_canary(arguments: argparse.Namespace) -> Path:
    run_directory = _new_run_directory("kernel-canary")
    pid_allowlist = _bind_gpu_pid_allowlist(run_directory)
    legs: list[str] = []
    aborted_reason: str | None = None
    try:
        for repetition in range(KERNEL_CANARY_REPETITIONS):
            cycle: list[tuple[str, str, int]] = [
                ("jax-value-grad", "jax", GPU_HOST_OMP_THREADS)
            ]
            cycle.extend(
                ("native-value-grad", "native", omp) for omp in NATIVE_OMP_SWEEP
            )
            offset = repetition % len(cycle)
            for kind, lane, omp in cycle[offset:] + cycle[:offset]:
                leg_id = f"{kind}-omp{omp}-rep{repetition}"
                specification: dict[str, object] = {
                    "kind": kind,
                    "leg_id": leg_id,
                    "phase": "kernel-canary",
                    "scratch": str(run_directory / "scratch" / leg_id),
                    "warm_calls": WARM_VALUE_GRAD_CALLS,
                    "omp_threads": omp,
                }
                if lane == "native":
                    specification.update(_native_affinity_fields(omp))
                else:
                    specification["cpu_affinity"] = list(range(GPU_HOST_OMP_THREADS))
                cache_root = run_directory / "jax-cache" if lane == "jax" else None
                legs.append(leg_id)
                _execute_leg(
                    run_directory,
                    specification,
                    _leg_environment(lane, omp_threads=omp, cache_root=cache_root),
                    pid_allowlist,
                )
    except BaseException as error:
        aborted_reason = _abort_reason(error)
        raise
    finally:
        _publish_manifest(
            run_directory,
            phase="kernel-canary",
            expected_leg_ids=legs,
            gate_sha256=None,
            aborted_reason=aborted_reason,
        )
    return run_directory


def _phase_hlo_capture(arguments: argparse.Namespace) -> Path:
    run_directory = _new_run_directory("hlo-capture")
    pid_allowlist = _bind_gpu_pid_allowlist(run_directory)
    leg_id = "jax-hlo"
    legs: list[str] = []
    aborted_reason: str | None = None
    try:
        legs.append(leg_id)
        _execute_leg(
            run_directory,
            {
                "kind": "jax-hlo",
                "leg_id": leg_id,
                "phase": "hlo-capture",
                "scratch": str(run_directory / "scratch" / leg_id),
                "artifact_dir": str(run_directory / "hlo"),
                "omp_threads": GPU_HOST_OMP_THREADS,
                "cpu_affinity": list(range(GPU_HOST_OMP_THREADS)),
                "timed": False,
            },
            _leg_environment(
                "jax",
                omp_threads=GPU_HOST_OMP_THREADS,
                cache_root=run_directory / "jax-cache",
            ),
            pid_allowlist,
        )
    except BaseException as error:
        aborted_reason = _abort_reason(error)
        raise
    finally:
        _publish_manifest(
            run_directory,
            phase="hlo-capture",
            expected_leg_ids=legs,
            gate_sha256=None,
            aborted_reason=aborted_reason,
        )
    return run_directory


def _load_gate(gate_path: Path) -> tuple[dict, str]:
    gate = _read_json(gate_path)
    if not isinstance(gate, dict) or gate.get("schema") != GATE_SCHEMA:
        raise BenchmarkError(f"not a quality contract: {gate_path}")
    return gate, _sha256_bytes(_canonical_json_bytes(gate))


def _gate_scaled_target(gate: Mapping[str, object]) -> float:
    """The scaled objective every timed native solve stops at.

    The frozen contract publishes the target in unscaled units and the
    reference solver record carries the one frozen objective scale, so the
    stopping rung is derived from the gate rather than restated anywhere.
    """
    return float(gate["solver"]["converged_reference"]["objective_scale"]) * float(
        gate["target_objective"]
    )


def first_qualifying_iteration(
    trace: Sequence[float], scaled_target: float
) -> int | None:
    """Smallest accepted-iteration count whose objective clears the target."""
    for index, value in enumerate(trace):
        if float(value) <= scaled_target:
            return index + 1
    return None


def _native_matrix_configs() -> tuple[tuple[int, int], ...]:
    return tuple(
        (omp, history) for omp in NATIVE_OMP_SWEEP for history in NATIVE_HISTORY_SWEEP
    )


SHIPPED_DEFAULT_DISCLOSURE_ROLE = "shipped-default-disclosure"
NATIVE_MATRIX_TIMED_ROLE = "time-to-quality"


def _phase_native_matrix(arguments: argparse.Namespace) -> Path:
    gate, gate_sha256 = _load_gate(arguments.gate)
    scaled_target = _gate_scaled_target(gate)
    run_directory = _new_run_directory("native-matrix")
    # The timed solves are the selection metric, so this CPU-only phase binds
    # a GPU allowlist too: foreign device compute contends for the same host.
    pid_allowlist = _bind_gpu_pid_allowlist(run_directory)
    _write_json_exclusive(run_directory / "gate" / "quality_contract.json", gate)
    legs: list[str] = []
    aborted_reason: str | None = None
    try:
        # Disclosure only: what a user gets with no threading pin at all.
        # Reported beside the matrix and never used as a denominator or a
        # selection candidate.
        disclosure_leg = "native-shipped-default-disclosure"
        legs.append(disclosure_leg)
        _execute_leg(
            run_directory,
            {
                "kind": "native-solve",
                "leg_id": disclosure_leg,
                "phase": "native-matrix",
                "role": SHIPPED_DEFAULT_DISCLOSURE_ROLE,
                "scratch": str(run_directory / "scratch" / disclosure_leg),
                "max_steps": NATIVE_STOP_MAX_STEPS,
                "history": GATE_REFERENCE_HISTORY,
                "record_trace": True,
                "stop_at_scaled_target": scaled_target,
                "affinity_policy": "unpinned-shipped-default",
                "gate_sha256": gate_sha256,
                "timed": False,
            },
            _leg_environment("native", omp_threads=None, cache_root=None),
            pid_allowlist,
        )

        # Time to quality: every repetition is an independent fresh-process
        # solve that finds its own crossing of the frozen rung, so nothing
        # here assumes one process's trajectory reproduces in another.
        configs = _native_matrix_configs()
        for repetition in range(SELECTION_REPETITIONS):
            offset = repetition % len(configs)
            for omp, history in configs[offset:] + configs[:offset]:
                leg_id = f"native-time-to-quality-omp{omp}-h{history}-rep{repetition}"
                legs.append(leg_id)
                _execute_leg(
                    run_directory,
                    {
                        "kind": "native-solve",
                        "leg_id": leg_id,
                        "phase": "native-matrix",
                        "role": NATIVE_MATRIX_TIMED_ROLE,
                        "scratch": str(run_directory / "scratch" / leg_id),
                        "max_steps": NATIVE_STOP_MAX_STEPS,
                        "history": history,
                        "record_trace": False,
                        "stop_at_scaled_target": scaled_target,
                        "omp_threads": omp,
                        **_native_affinity_fields(omp),
                        "gate_sha256": gate_sha256,
                    },
                    _leg_environment("native", omp_threads=omp, cache_root=None),
                    pid_allowlist,
                )
    except BaseException as error:
        aborted_reason = _abort_reason(error)
        raise
    finally:
        _publish_manifest(
            run_directory,
            phase="native-matrix",
            expected_leg_ids=legs,
            gate_sha256=gate_sha256,
            aborted_reason=aborted_reason,
        )
    return run_directory


def _native_oracle_leg_id(subject_leg_id: str) -> str:
    return f"native-endpoint-{subject_leg_id}"


def _execute_native_oracle_leg(
    run_directory: Path,
    *,
    phase: str,
    subject_leg_id: str,
    gate_sha256: str,
) -> Mapping[str, object]:
    """Re-evaluate a GPU leg's published solution on the native evaluator.

    Untimed and pinned at the gate reference thread count, so it can run
    between timed legs without competing with one.
    """
    subject_row = _read_json(run_directory / "rows" / f"{subject_leg_id}.json")
    leg_id = _native_oracle_leg_id(subject_leg_id)
    _execute_leg(
        run_directory,
        {
            "kind": "native-endpoint-eval",
            "leg_id": leg_id,
            "phase": phase,
            "role": "endpoint-eval",
            "scratch": str(run_directory / "scratch" / leg_id),
            "subject_leg_id": subject_leg_id,
            "parameters": subject_row["endpoint"]["solution"],
            "omp_threads": GATE_REFERENCE_OMP,
            **_native_affinity_fields(GATE_REFERENCE_OMP),
            "gate_sha256": gate_sha256,
            "timed": False,
        },
        _leg_environment("native", omp_threads=GATE_REFERENCE_OMP, cache_root=None),
    )
    return _read_json(run_directory / "rows" / f"{leg_id}.json")


def _phase_jax_sweep(arguments: argparse.Namespace) -> Path:
    gate, gate_sha256 = _load_gate(arguments.gate)
    run_directory = _new_run_directory("jax-sweep")
    pid_allowlist = _bind_gpu_pid_allowlist(run_directory)
    _write_json_exclusive(run_directory / "gate" / "quality_contract.json", gate)
    legs: list[str] = []
    aborted_reason: str | None = None
    try:
        for history in JAX_HISTORY_SWEEP:
            for budget in GPU_BUDGET_SWEEP:
                leg_id = f"jax-sweep-h{history}-b{budget}"
                legs.append(leg_id)
                _execute_leg(
                    run_directory,
                    {
                        "kind": "jax-solve",
                        "leg_id": leg_id,
                        "phase": "jax-sweep",
                        "role": "sweep",
                        "scratch": str(run_directory / "scratch" / leg_id),
                        "max_steps": budget,
                        "history": history,
                        "warm_repetitions": SELECTION_REPETITIONS,
                        "omp_threads": GPU_HOST_OMP_THREADS,
                        "cpu_affinity": list(range(GPU_HOST_OMP_THREADS)),
                        "gate_sha256": gate_sha256,
                    },
                    _leg_environment(
                        "jax",
                        omp_threads=GPU_HOST_OMP_THREADS,
                        cache_root=run_directory / "jax-cache",
                    ),
                    pid_allowlist,
                )
                legs.append(_native_oracle_leg_id(leg_id))
                oracle_row = _execute_native_oracle_leg(
                    run_directory,
                    phase="jax-sweep",
                    subject_leg_id=leg_id,
                    gate_sha256=gate_sha256,
                )
                if evaluate_quality_gate(gate, oracle_row["endpoint"])["eligible"]:
                    break
    except BaseException as error:
        aborted_reason = _abort_reason(error)
        raise
    finally:
        _publish_manifest(
            run_directory,
            phase="jax-sweep",
            expected_leg_ids=legs,
            gate_sha256=gate_sha256,
            aborted_reason=aborted_reason,
        )
    return run_directory


# Per pair both lanes are split the same way, into four timed legs.  A warm
# leg runs the discard-then-repeat protocol and owns the warm statistic; a
# wall leg runs exactly one solve in a fresh process and owns the subprocess
# wall statistic.  One leg cannot own both metrics: the warm protocol's
# discarded solve and its extra repetitions are inside any wall that leg
# publishes, which would charge the native wall numerator for work the GPU
# wall leg never did.
FINAL_PAIR_WARM_NATIVE_ROLE = "warm-native"
FINAL_PAIR_WALL_NATIVE_ROLE = "wall-native"
FINAL_PAIR_NATIVE_ROLES = (FINAL_PAIR_WARM_NATIVE_ROLE, FINAL_PAIR_WALL_NATIVE_ROLE)
FINAL_PAIR_JAX_ROLES = ("warm", "wall")
FINAL_PAIR_ROLES = FINAL_PAIR_NATIVE_ROLES + FINAL_PAIR_JAX_ROLES
# Roles whose specification must declare the discard-first warm protocol.
# Only the native lane records the field; the JAX lane expresses the same
# split through its own ``warm_repetitions``.
FINAL_PAIR_WARM_PROTOCOL_ROLES = frozenset({FINAL_PAIR_WARM_NATIVE_ROLE})


def _phase_final_pairs(arguments: argparse.Namespace) -> Path:
    gate, gate_sha256 = _load_gate(arguments.gate)
    selection = _read_json(arguments.selection)
    if selection.get("gate_sha256") != gate_sha256:
        raise BenchmarkError("selection was frozen against a different gate")
    cache_policy = str(arguments.cache_policy)
    run_directory = _new_run_directory(f"final-pairs-{cache_policy}")
    pid_allowlist = _bind_gpu_pid_allowlist(run_directory)
    _write_json_exclusive(run_directory / "gate" / "quality_contract.json", gate)
    _write_json_exclusive(run_directory / "selection.json", selection)
    native = selection["native"]
    jax_selection = selection["jax"]
    legs: list[str] = []
    aborted_reason: str | None = None

    shared_cache = run_directory / "jax-cache"
    try:
        if cache_policy == "primed":
            leg_id = "jax-prime"
            legs.append(leg_id)
            _execute_leg(
                run_directory,
                {
                    "kind": "jax-solve",
                    "leg_id": leg_id,
                    "phase": "final-pairs",
                    "role": "prime",
                    "scratch": str(run_directory / "scratch" / leg_id),
                    "max_steps": int(jax_selection["budget"]),
                    "history": int(jax_selection["history"]),
                    "warm_repetitions": 0,
                    "omp_threads": GPU_HOST_OMP_THREADS,
                    "cpu_affinity": list(range(GPU_HOST_OMP_THREADS)),
                    "gate_sha256": gate_sha256,
                },
                _leg_environment(
                    "jax",
                    omp_threads=GPU_HOST_OMP_THREADS,
                    cache_root=shared_cache,
                ),
                pid_allowlist,
            )

        for pair_index in range(FINAL_PAIR_COUNT):
            wall_cache = (
                shared_cache
                if cache_policy == "primed"
                else run_directory / f"jax-cache-fresh-pair{pair_index}"
            )
            native_legs = []
            # ``warm``: discard-first protocol, FINAL_WARM_REPETITIONS timed
            # solves, median vs the GPU lane's median of the same count.
            # ``wall``: one solve in a fresh process, whose launch wall is the
            # wall numerator.  Matrix repetitions stay single-solve — see
            # NATIVE_MATRIX_SELECTION_METRIC_ANNOTATION.
            for metric, warm_protocol, warm_repetitions in (
                ("warm", True, FINAL_WARM_REPETITIONS),
                ("wall", False, 1),
            ):
                native_leg_id = f"native-{metric}-pair{pair_index}"
                native_legs.append(
                    (
                        native_leg_id,
                        {
                            "kind": "native-solve",
                            "leg_id": native_leg_id,
                            "phase": "final-pairs",
                            "role": f"{metric}-native",
                            "pair_index": pair_index,
                            "scratch": str(run_directory / "scratch" / native_leg_id),
                            "max_steps": NATIVE_STOP_MAX_STEPS,
                            "history": int(native["history"]),
                            "record_trace": False,
                            "stop_at_scaled_target": _gate_scaled_target(gate),
                            "warm_protocol": warm_protocol,
                            "warm_repetitions": warm_repetitions,
                            "omp_threads": int(native["omp"]),
                            **_native_affinity_fields(int(native["omp"])),
                            "gate_sha256": gate_sha256,
                        },
                        _leg_environment(
                            "native", omp_threads=int(native["omp"]), cache_root=None
                        ),
                    )
                )
            jax_legs = []
            for role, warm_repetitions, cache_root in (
                ("warm", FINAL_WARM_REPETITIONS, shared_cache),
                ("wall", 0, wall_cache),
            ):
                leg_id = f"jax-{role}-pair{pair_index}"
                jax_legs.append(
                    (
                        leg_id,
                        {
                            "kind": "jax-solve",
                            "leg_id": leg_id,
                            "phase": "final-pairs",
                            "role": role,
                            "pair_index": pair_index,
                            "scratch": str(run_directory / "scratch" / leg_id),
                            "max_steps": int(jax_selection["budget"]),
                            "history": int(jax_selection["history"]),
                            "warm_repetitions": warm_repetitions,
                            "omp_threads": GPU_HOST_OMP_THREADS,
                            "cpu_affinity": list(range(GPU_HOST_OMP_THREADS)),
                            "gate_sha256": gate_sha256,
                        },
                        _leg_environment(
                            "jax",
                            omp_threads=GPU_HOST_OMP_THREADS,
                            cache_root=cache_root,
                        ),
                    )
                )
            # Alternate which lane's block leads; the two legs of a lane stay
            # adjacent so no cross-lane leg lands between them.
            block = (
                [*native_legs, *jax_legs]
                if pair_index % 2 == 0
                else [*jax_legs, *native_legs]
            )
            for leg_id, specification, environment in block:
                legs.append(leg_id)
                _execute_leg(run_directory, specification, environment, pid_allowlist)
            # The native oracle re-evaluations run after the whole timed block
            # so a CPU leg never overlaps a timed measurement.
            for jax_leg_id, _, _ in jax_legs:
                legs.append(_native_oracle_leg_id(jax_leg_id))
                _execute_native_oracle_leg(
                    run_directory,
                    phase="final-pairs",
                    subject_leg_id=jax_leg_id,
                    gate_sha256=gate_sha256,
                )
    except BaseException as error:
        aborted_reason = _abort_reason(error)
        raise
    finally:
        _publish_manifest(
            run_directory,
            phase="final-pairs",
            expected_leg_ids=legs,
            gate_sha256=gate_sha256,
            cache_policy=cache_policy,
            aborted_reason=aborted_reason,
        )
    return run_directory


# ---------------------------------------------------------------------------
# Validation: pure reductions over raw rows
# ---------------------------------------------------------------------------


def _finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _all_finite(values: object) -> bool:
    if isinstance(values, list):
        return all(_all_finite(entry) for entry in values)
    return _finite(values)


def evaluate_quality_gate(
    gate: Mapping[str, object], endpoint: Mapping[str, object]
) -> dict[str, object]:
    """Apply every frozen gate clause to one endpoint; fail-closed on shape.

    ``endpoint`` is always a NATIVE evaluation: either a native lane's own
    endpoint or the native re-evaluation of a GPU lane's published solution.
    The anchor is the gate's reference endpoint -- the converged run's own
    first qualifying iterate -- so a truncated lane is compared with a
    truncated reference.
    """
    failures: list[str] = []
    tolerances = gate["tolerances"]
    reference = gate["reference_endpoint"]
    for name in _GATE_ENDPOINT_FIELDS:
        if name not in endpoint:
            return {"eligible": False, "failures": [f"missing endpoint field {name}"]}
        if not _all_finite(endpoint[name]):
            failures.append(f"nonfinite endpoint field {name}")
    if failures:
        return {"eligible": False, "failures": failures}
    objective = float(endpoint["objective"])
    if not objective <= float(gate["target_objective"]):
        failures.append(
            f"objective {objective} misses target {gate['target_objective']}"
        )
    if not objective < float(gate["initial_objective"]):
        failures.append("objective does not improve on the common initial state")
    rtol = float(tolerances["endpoint_rtol"])
    atol = float(tolerances["endpoint_atol"])
    # Two numerically equal slacks under two names: the caps below and the
    # gradient-norm ratio are separate clauses of the contract.
    cap_margin = float(tolerances["quality_cap_margin"])
    margin = float(tolerances["gradient_norm_margin"])
    # One-sided quality caps.  Squared flux and both penalties are
    # nonnegative, and converging BETTER than the truncated reference is the
    # outcome this benchmark is hunting for -- never a gate failure.
    for name in ("squared_flux", "length_penalty", "distance_penalty"):
        observed = float(endpoint[name])
        cap = atol + cap_margin * float(reference[name])
        if not observed <= cap:
            failures.append(f"{name} {observed} exceeds cap {cap}")
    # Two-sided geometry bands: a different geometric regime is a failure in
    # either direction, however good the objective looks.
    observed_clearance = float(endpoint["minimum_clearance"])
    expected_clearance = float(reference["minimum_clearance"])
    if abs(observed_clearance - expected_clearance) > atol + rtol * abs(
        expected_clearance
    ):
        failures.append(
            f"minimum_clearance {observed_clearance} outside rtol={rtol}, "
            f"atol={atol} of {expected_clearance}"
        )
    observed_lengths = endpoint["coil_lengths"]
    expected_lengths = reference["coil_lengths"]
    if len(observed_lengths) != len(expected_lengths):
        failures.append("coil length count mismatch")
    else:
        for index, (observed, expected) in enumerate(
            zip(observed_lengths, expected_lengths, strict=True)
        ):
            if abs(float(observed) - float(expected)) > atol + rtol * abs(
                float(expected)
            ):
                failures.append(f"coil length {index} outside tolerance")
    norm_ratio = float(endpoint["gradient_inf_norm"]) / max(
        float(reference["gradient_inf_norm"]),
        float(tolerances["gradient_norm_floor"]),
    )
    if not norm_ratio <= margin:
        failures.append(f"gradient infinity norm ratio {norm_ratio} exceeds {margin}")
    if not observed_clearance > 0.0:
        failures.append("minimum clearance is not positive")
    return {"eligible": not failures, "failures": failures}


# Metrics both lanes publish for the same solution vector.  They are compared
# only to detect a lane fork; the native value is the one that is gated.
_LANE_CROSS_CHECK_METRICS = (
    "objective",
    "squared_flux",
    "length_penalty",
    "distance_penalty",
    "minimum_clearance",
    "gradient_inf_norm",
)


def _lane_cross_check(
    reported: Mapping[str, object], oracle: Mapping[str, object]
) -> list[str]:
    """Names where a lane's self-report disagrees with the native oracle."""
    disagreements: list[str] = []
    for name in _LANE_CROSS_CHECK_METRICS:
        if name not in reported or name not in oracle:
            disagreements.append(f"{name} (missing)")
            continue
        observed = float(reported[name])
        expected = float(oracle[name])
        if abs(observed - expected) > GATE_ENDPOINT_ATOL + LANE_CROSS_CHECK_RTOL * abs(
            expected
        ):
            disagreements.append(f"{name} ({observed} vs {expected})")
    reported_lengths = reported.get("coil_lengths")
    oracle_lengths = oracle.get("coil_lengths")
    if not isinstance(reported_lengths, list) or not isinstance(oracle_lengths, list):
        disagreements.append("coil_lengths (missing)")
    elif len(reported_lengths) != len(oracle_lengths):
        disagreements.append("coil_lengths (length mismatch)")
    else:
        for index, (observed, expected) in enumerate(
            zip(reported_lengths, oracle_lengths, strict=True)
        ):
            if abs(float(observed) - float(expected)) > (
                GATE_ENDPOINT_ATOL + LANE_CROSS_CHECK_RTOL * abs(float(expected))
            ):
                disagreements.append(f"coil_lengths[{index}]")
    return disagreements


def _gate_through_native_oracle(
    gate: Mapping[str, object],
    jax_row: Mapping[str, object],
    oracle_rows: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Gate one GPU endpoint through its native re-evaluation.

    ``produced`` is False when the evidence itself is broken: no paired native
    re-evaluation, or two lanes that disagree about the same solution vector
    (a physics fork, which is never a speed verdict).
    """
    leg_id = str(jax_row.get("leg_id"))
    oracle = oracle_rows.get(leg_id)
    if oracle is None:
        return {
            "produced": False,
            "reason": f"{leg_id} has no native endpoint re-evaluation",
        }
    if not isinstance(jax_row.get("endpoint"), Mapping):
        return {"produced": False, "reason": f"{leg_id} published no endpoint"}
    disagreements = _lane_cross_check(jax_row["endpoint"], oracle["endpoint"])
    if disagreements:
        return {
            "produced": False,
            "reason": (
                f"{leg_id} self-report diverged from its native re-evaluation "
                f"beyond rtol={LANE_CROSS_CHECK_RTOL}: {disagreements}"
            ),
        }
    evaluation = evaluate_quality_gate(gate, oracle["endpoint"])
    return {
        "produced": True,
        "eligible": bool(evaluation["eligible"]),
        "failures": evaluation["failures"],
    }


def _median(values: Sequence[float]) -> float:
    return float(statistics.median(values))


# How many mismatching gradient components to name per state before
# truncating: enough to show whether a fork is one component or the whole
# vector, without publishing a 675-entry list per state.
BASELINE_MISMATCH_REPORT_LIMIT = 10


def reduce_baseline(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Compare native and JAX evaluator states; fail-closed on any mismatch."""
    by_kind: dict[str, Mapping[str, object]] = {}
    for row in rows:
        kind = str(row.get("kind"))
        if kind in by_kind:
            return {
                "verdict": VERDICT_NOT_PRODUCED,
                "reason": f"duplicate {kind} evaluator row",
            }
        by_kind[kind] = row
    native = by_kind.get("native-eval")
    jax_row = by_kind.get("jax-eval")
    if native is None or jax_row is None:
        return {"verdict": VERDICT_NOT_PRODUCED, "reason": "missing evaluator rows"}
    if native["fingerprints"] != jax_row["fingerprints"]:
        return {"verdict": VERDICT_NOT_PRODUCED, "reason": "fingerprint mismatch"}
    mismatches: list[str] = []
    for label in BASELINE_STATE_LABELS:
        native_state = native["states"].get(label)
        jax_state = jax_row["states"].get(label)
        if native_state is None or jax_state is None:
            return {
                "verdict": VERDICT_NOT_PRODUCED,
                "reason": f"missing state {label}",
            }
        for name in (
            "objective",
            "squared_flux",
            "length_penalty",
            "distance_penalty",
            "gradient",
            "minimum_clearance",
            "coil_lengths",
        ):
            if name not in native_state or name not in jax_state:
                return {
                    "verdict": VERDICT_NOT_PRODUCED,
                    "reason": f"missing {label}:{name}",
                }
            if not _all_finite(native_state[name]) or not _all_finite(jax_state[name]):
                return {
                    "verdict": VERDICT_NOT_PRODUCED,
                    "reason": f"nonfinite {label}:{name}",
                }
        for name in (
            "objective",
            "squared_flux",
            "length_penalty",
            "distance_penalty",
        ):
            observed = float(jax_state[name])
            expected = float(native_state[name])
            if abs(observed - expected) > BASELINE_ATOL + BASELINE_RTOL * abs(expected):
                mismatches.append(f"{label}:{name}")
        native_gradient = native_state["gradient"]
        jax_gradient = jax_state["gradient"]
        if len(native_gradient) != len(jax_gradient):
            return {
                "verdict": VERDICT_NOT_PRODUCED,
                "reason": f"gradient length mismatch at {label}",
            }
        # Per-component: the reduction-order noise scales with the component,
        # not with the vector's largest entry, so a scale-relative rule would
        # hide a fork in the small components.
        reported = 0
        for index, (expected, observed) in enumerate(
            zip(native_gradient, jax_gradient, strict=True)
        ):
            if abs(float(observed) - float(expected)) > (
                BASELINE_GRADIENT_ATOL + BASELINE_GRADIENT_RTOL * abs(float(expected))
            ):
                reported += 1
                if reported <= BASELINE_MISMATCH_REPORT_LIMIT:
                    mismatches.append(f"{label}:gradient[{index}]")
        if reported > BASELINE_MISMATCH_REPORT_LIMIT:
            mismatches.append(
                f"{label}:gradient (+{reported - BASELINE_MISMATCH_REPORT_LIMIT} more)"
            )
        native_lengths = native_state["coil_lengths"]
        jax_lengths = jax_state["coil_lengths"]
        if len(native_lengths) != len(jax_lengths):
            return {
                "verdict": VERDICT_NOT_PRODUCED,
                "reason": f"coil length count mismatch at {label}",
            }
        for index, (expected, observed) in enumerate(
            zip(native_lengths, jax_lengths, strict=True)
        ):
            if abs(float(observed) - float(expected)) > (
                BASELINE_ATOL + BASELINE_RTOL * abs(float(expected))
            ):
                mismatches.append(f"{label}:coil_lengths[{index}]")
        observed_clearance = float(jax_state["minimum_clearance"])
        expected_clearance = float(native_state["minimum_clearance"])
        if abs(observed_clearance - expected_clearance) > (
            BASELINE_CLEARANCE_RTOL * abs(expected_clearance)
        ):
            mismatches.append(f"{label}:minimum_clearance")
    return {
        "verdict": "IDENTITY_OK" if not mismatches else VERDICT_NOT_PRODUCED,
        "mismatches": mismatches,
    }


def reduce_kernel_canary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Best-native versus GPU warm value/gradient medians and the 1.10x gate."""
    native_medians: dict[int, list[float]] = {}
    gpu_samples: list[float] = []
    for row in rows:
        if str(row.get("record", "")) == "launch":
            continue
        timings = row.get("timings")
        if not isinstance(timings, Mapping):
            return {"verdict": VERDICT_NOT_PRODUCED, "reason": "row without timings"}
        warm = timings.get("warm_value_grad_seconds")
        if not isinstance(warm, list) or not warm or not _all_finite(warm):
            return {
                "verdict": VERDICT_NOT_PRODUCED,
                "reason": f"invalid warm samples in {row.get('leg_id')}",
            }
        if row.get("kind") == "native-value-grad":
            omp = int(row["specification"]["omp_threads"])
            native_medians.setdefault(omp, []).append(_median(warm))
        elif row.get("kind") == "jax-value-grad":
            gpu_samples.append(_median(warm))
        else:
            return {
                "verdict": VERDICT_NOT_PRODUCED,
                "reason": f"unexpected kind {row.get('kind')}",
            }
    if set(native_medians) != set(NATIVE_OMP_SWEEP):
        return {
            "verdict": VERDICT_NOT_PRODUCED,
            "reason": "incomplete native OMP sweep",
        }
    # Every lane must contribute the same number of repetitions: a missing or
    # a duplicated repetition moves the median it feeds.
    incomplete = {
        omp: len(samples)
        for omp, samples in sorted(native_medians.items())
        if len(samples) != KERNEL_CANARY_REPETITIONS
    }
    if incomplete or len(gpu_samples) != KERNEL_CANARY_REPETITIONS:
        return {
            "verdict": VERDICT_NOT_PRODUCED,
            "reason": (
                f"expected {KERNEL_CANARY_REPETITIONS} repetitions per lane; "
                f"native {incomplete or 'complete'}, gpu {len(gpu_samples)}"
            ),
        }
    per_omp = {omp: _median(samples) for omp, samples in native_medians.items()}
    best_omp = min(per_omp, key=per_omp.__getitem__)
    best_native_seconds = per_omp[best_omp]
    gpu_seconds = _median(gpu_samples)
    ratio = best_native_seconds / gpu_seconds
    return {
        "verdict": DECISION_PROCEED
        if ratio >= KERNEL_CANARY_MINIMUM_RATIO
        else VERDICT_CLOSED,
        "native_median_seconds_by_omp": per_omp,
        "best_native": {"omp": best_omp, "median_seconds": best_native_seconds},
        "gpu_median_seconds": gpu_seconds,
        "ratio": ratio,
        "minimum_ratio": KERNEL_CANARY_MINIMUM_RATIO,
    }


# The matrix repetitions are deliberately single-solve, unlike the final
# pairs' warm legs, so the statistic they publish must say so.  The label
# travels with the verdict document rather than living only in this comment.
NATIVE_STOP_ANNOTATION = (
    "native stop legs terminate via a per-iteration callback at the frozen "
    "rung under a preregistered cap of NATIVE_STOP_MAX_STEPS; the callback "
    "costs O(microseconds) against ~150 ms iterations, charged to the "
    "native lane (a <=1e-5 relative pro-GPU bias, four orders under the "
    "1.10 gate)"
)
NATIVE_MATRIX_SELECTION_METRIC_ANNOTATION = (
    "per-configuration statistic is fresh-process solve time to the frozen "
    "gate rung: one callback-stopped solve per subprocess, with no discarded "
    "warm-up solve; the cold-start delta scales with thread count but is "
    "O(ms) against O(10s) solves, so it cannot reorder configurations at the "
    f"{FINAL_MEDIAN_MINIMUM_RATIO} margin"
)


def reduce_native_matrix(
    gate: Mapping[str, object], rows: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    """Recompute native time-to-quality, gate eligibility, and the fastest lane.

    Every repetition is an independent solve that stops at its own crossing of
    the frozen rung.  A repetition qualifies only if it actually crossed
    (``stopped_at_target``) and its stop endpoint clears the gate, and a
    configuration is a denominator candidate only if all of its repetitions
    qualify: a lane that reaches the rung only sometimes is not a lane the
    speed claim can be divided by.
    """
    gate_sha256 = _sha256_bytes(_canonical_json_bytes(gate))
    timed: dict[tuple[int, int], list[Mapping[str, object]]] = {}
    disclosure: Mapping[str, object] | None = None
    launch_walls: dict[str, float] = {}
    for row in rows:
        if str(row.get("record", "")) == "launch":
            launch_walls[str(row["leg_id"])] = float(row["process_wall_seconds"])
            continue
        if row.get("gate_sha256") != gate_sha256:
            return {
                "verdict": VERDICT_NOT_PRODUCED,
                "reason": f"row {row.get('leg_id')} bound to a different gate",
            }
        specification = row["specification"]
        role = str(specification.get("role"))
        if role == SHIPPED_DEFAULT_DISCLOSURE_ROLE:
            if disclosure is not None:
                return {
                    "verdict": VERDICT_NOT_PRODUCED,
                    "reason": "duplicate shipped-default disclosure row",
                }
            disclosure = row
            continue
        if role != NATIVE_MATRIX_TIMED_ROLE:
            return {
                "verdict": VERDICT_NOT_PRODUCED,
                "reason": f"unexpected role {role} in {row.get('leg_id')}",
            }
        config = (
            int(specification["omp_threads"]),
            int(specification["history"]),
        )
        timed.setdefault(config, []).append(row)
    if set(timed) != set(_native_matrix_configs()):
        return {
            "verdict": VERDICT_NOT_PRODUCED,
            "reason": "incomplete time-to-quality matrix",
        }
    miscounted = {
        f"omp{omp}-h{history}": len(config_rows)
        for (omp, history), config_rows in sorted(timed.items())
        if len(config_rows) != SELECTION_REPETITIONS
    }
    if miscounted:
        return {
            "verdict": VERDICT_NOT_PRODUCED,
            "reason": (
                f"every configuration owes exactly {SELECTION_REPETITIONS} "
                f"time-to-quality repetitions: {miscounted}"
            ),
        }
    table: dict[str, dict[str, object]] = {}
    qualifying: dict[tuple[int, int], float] = {}
    for config, config_rows in sorted(timed.items()):
        omp, history = config
        times: list[float] = []
        iterations: list[int] = []
        failures: list[str] = []
        for repetition in config_rows:
            leg_id = str(repetition["leg_id"])
            solver = repetition["solver"]
            iterations.append(int(solver["nit"]))
            if not bool(solver.get("stopped_at_target", False)):
                # Rung-unreachability: the callback never fired, so this
                # configuration has no time-to-quality at all -- not a slow
                # one.  Name the actual termination: cap exhaustion only when
                # the solver spent its whole budget; otherwise it stopped on
                # its own criteria below the cap.
                cap = int(repetition["specification"]["max_steps"])
                nit = int(solver["nit"])
                if nit >= cap:
                    failures.append(
                        f"{leg_id} exhausted its {cap}-iteration cap "
                        "without reaching the frozen gate rung"
                    )
                else:
                    failures.append(
                        f"{leg_id} stopped at nit {nit} of its {cap}-iteration "
                        f"cap (solver status {solver.get('status')}) without "
                        "reaching the frozen gate rung"
                    )
            else:
                gate_verdict = evaluate_quality_gate(gate, repetition["endpoint"])
                if not gate_verdict["eligible"]:
                    failures.append(f"{leg_id}: {gate_verdict['failures']}")
            repetition_times = repetition["timings"]["warm_solve_seconds"]
            if not isinstance(repetition_times, list) or not repetition_times:
                return {
                    "verdict": VERDICT_NOT_PRODUCED,
                    "reason": f"omp{omp}-h{history} repetition has no solve samples",
                }
            times.append(_median(repetition_times))
        if not _all_finite(times):
            return {
                "verdict": VERDICT_NOT_PRODUCED,
                "reason": f"omp{omp}-h{history} has nonfinite timings",
            }
        entry: dict[str, object] = {
            "repetitions": len(config_rows),
            "qualifying_repetitions": len(config_rows) - len(failures),
            "nit": iterations,
            "median_fresh_process_solve_seconds": _median(times),
            "eligible": not failures,
            "failures": failures,
        }
        table[f"omp{omp}-h{history}"] = entry
        if not failures:
            qualifying[config] = _median(times)
    if not qualifying:
        return {
            "verdict": VERDICT_NOT_PRODUCED,
            "reason": (
                "no native configuration reached the frozen gate rung in every "
                "repetition; the per-configuration table names the crossings "
                "that were missed"
            ),
            "table": table,
        }
    best_config = min(qualifying, key=qualifying.__getitem__)
    best_entry = table[f"omp{best_config[0]}-h{best_config[1]}"]
    return {
        "verdict": "NATIVE_SELECTED",
        "table": table,
        "selection_metric": NATIVE_MATRIX_SELECTION_METRIC_ANNOTATION,
        "native_stop_cap": NATIVE_STOP_MAX_STEPS,
        "native_stop_annotation": NATIVE_STOP_ANNOTATION,
        "shipped_default_disclosure": _disclosure_report(
            gate, disclosure, launch_walls
        ),
        "selected": {
            "omp": best_config[0],
            "history": best_config[1],
            # Informational: the selected lane has no frozen iteration budget,
            # it stops at the rung it measures.
            "median_nit": _median(best_entry["nit"]),
            "median_fresh_process_solve_seconds": qualifying[best_config],
        },
    }


def _disclosure_report(
    gate: Mapping[str, object],
    disclosure: Mapping[str, object] | None,
    launch_walls: Mapping[str, float],
) -> dict[str, object] | None:
    """Report the unpinned shipped-default lane beside the selection.

    Informational only: it is never a selection candidate and never the
    denominator of a ratio, because its thread count is whatever the host
    happened to choose.
    """
    if disclosure is None:
        return None
    leg_id = str(disclosure["leg_id"])
    trace = disclosure.get("scaled_objective_trace")
    budget = None
    if isinstance(trace, list) and _all_finite(trace):
        budget = first_qualifying_iteration(
            trace,
            float(disclosure["solver"]["objective_scale"])
            * float(gate["target_objective"]),
        )
    solve_seconds = disclosure["timings"]["warm_solve_seconds"]
    return {
        "leg_id": leg_id,
        "budget": budget,
        "endpoint_objective": disclosure["endpoint"]["objective"],
        "solve_seconds": (
            _median(solve_seconds)
            if isinstance(solve_seconds, list) and solve_seconds
            else None
        ),
        "process_wall_seconds": launch_walls.get(leg_id),
    }


def reduce_jax_sweep(
    gate: Mapping[str, object], rows: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    """First qualifying budget per history; fastest qualifying history wins.

    Eligibility is decided on the paired ``native-endpoint-eval`` row, never
    on the GPU lane's self-reported endpoint.
    """
    gate_sha256 = _sha256_bytes(_canonical_json_bytes(gate))
    by_history: dict[int, dict[int, Mapping[str, object]]] = {}
    oracle_rows: dict[str, Mapping[str, object]] = {}
    for row in rows:
        if str(row.get("record", "")) == "launch":
            continue
        if row.get("gate_sha256") != gate_sha256:
            return {
                "verdict": VERDICT_NOT_PRODUCED,
                "reason": f"row {row.get('leg_id')} bound to a different gate",
            }
        if str(row.get("kind")) == "native-endpoint-eval":
            subject = str(row["subject_leg_id"])
            if subject in oracle_rows:
                return {
                    "verdict": VERDICT_NOT_PRODUCED,
                    "reason": f"duplicate native re-evaluation of {subject}",
                }
            oracle_rows[subject] = row
            continue
        specification = row["specification"]
        if str(specification.get("role")) != "sweep" or row.get("lane") != "jax":
            return {
                "verdict": VERDICT_NOT_PRODUCED,
                "reason": (
                    f"unexpected {row.get('lane')} row with role "
                    f"{specification.get('role')} in {row.get('leg_id')}"
                ),
            }
        history = int(specification["history"])
        budget = int(specification["max_steps"])
        if budget in by_history.get(history, {}):
            return {
                "verdict": VERDICT_NOT_PRODUCED,
                "reason": f"duplicate history {history} budget {budget} row",
            }
        by_history.setdefault(history, {})[budget] = row
    if set(by_history) != set(JAX_HISTORY_SWEEP):
        return {
            "verdict": VERDICT_NOT_PRODUCED,
            "reason": "incomplete history sweep",
        }
    table: dict[str, dict[str, object]] = {}
    qualifying: dict[int, tuple[int, float]] = {}
    for history in JAX_HISTORY_SWEEP:
        budgets = sorted(by_history[history])
        expected_prefix = list(GPU_BUDGET_SWEEP[: len(budgets)])
        if budgets != expected_prefix:
            return {
                "verdict": VERDICT_NOT_PRODUCED,
                "reason": f"history {history} budgets are not an ascending "
                f"prefix of the preregistered ladder: {budgets}",
            }
        entry: dict[str, object] = {"qualifying_budget": None}
        table[f"h{history}"] = entry
        for position, budget in enumerate(budgets):
            row = by_history[history][budget]
            evaluation = _gate_through_native_oracle(gate, row, oracle_rows)
            if not evaluation["produced"]:
                return {
                    "verdict": VERDICT_NOT_PRODUCED,
                    "reason": evaluation["reason"],
                }
            eligible = bool(evaluation["eligible"])
            if eligible:
                if position != len(budgets) - 1:
                    return {
                        "verdict": VERDICT_NOT_PRODUCED,
                        "reason": f"history {history} kept sweeping past a "
                        f"qualifying budget {budget}",
                    }
                warm = row["timings"]["warm_solve_seconds"]
                if (
                    not isinstance(warm, list)
                    or len(warm) != SELECTION_REPETITIONS
                    or not _all_finite(warm)
                ):
                    return {
                        "verdict": VERDICT_NOT_PRODUCED,
                        "reason": (
                            f"history {history} published {warm} warm samples, "
                            f"expected {SELECTION_REPETITIONS} finite values"
                        ),
                    }
                entry["qualifying_budget"] = budget
                entry["median_solve_seconds"] = _median(warm)
                qualifying[history] = (budget, _median(warm))
            elif position == len(budgets) - 1 and len(budgets) != len(GPU_BUDGET_SWEEP):
                return {
                    "verdict": VERDICT_NOT_PRODUCED,
                    "reason": f"history {history} sweep stopped early without "
                    "a qualifying budget",
                }
    if not qualifying:
        return {
            "verdict": VERDICT_CLOSED,
            "reason": "no JAX history reached the frozen endpoint contract",
            "table": table,
        }
    best_history = min(qualifying, key=lambda history: qualifying[history][1])
    return {
        "verdict": "JAX_SELECTED",
        "table": table,
        "selected": {
            "history": best_history,
            "budget": qualifying[best_history][0],
            "median_solve_seconds": qualifying[best_history][1],
        },
    }


def _selection_mismatch(
    selection: Mapping[str, object], key: str, specification: Mapping[str, object]
) -> str | None:
    """Name the first frozen-selection field this pair leg contradicts."""
    if key == "native":
        # No ``max_steps``: the native lane has no frozen iteration budget
        # under the time-to-quality protocol -- it stops at the frozen rung,
        # and the cap it never reaches is not a selected quantity.
        expected = {
            "history": int(selection["native"]["history"]),
            "omp_threads": int(selection["native"]["omp"]),
        }
    else:
        # The selection document freezes no GPU host-thread count; the
        # benchmark's own constant is the contract for that field.
        expected = {
            "max_steps": int(selection["jax"]["budget"]),
            "history": int(selection["jax"]["history"]),
            "omp_threads": GPU_HOST_OMP_THREADS,
        }
    for name, value in expected.items():
        if name not in specification:
            return f"{key} leg does not record {name}"
        if int(specification[name]) != value:
            return f"{key} leg {name}={specification[name]} is not the frozen {value}"
    return None


def reduce_final_pairs(
    gate: Mapping[str, object],
    selection: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Recompute the terminal speed verdict from the five interleaved pairs.

    Each pair holds four timed legs -- both lanes split the same way -- plus
    one native re-evaluation of each GPU endpoint.  ``warm-native`` and
    ``warm`` run the repeated warm protocol and are the only source of the
    warm ratio; ``wall-native`` and ``wall`` run one solve per fresh process
    and their launch walls are the only source of the wall ratio.  So neither
    metric inflates the other's protocol in either lane, and no GPU lane gates
    itself: the native legs are their own oracle, the GPU legs are gated
    through their ``native-endpoint-eval`` rows.
    """
    gate_sha256 = _sha256_bytes(_canonical_json_bytes(gate))
    pairs: dict[int, dict[str, Mapping[str, object]]] = {}
    oracle_rows: dict[str, Mapping[str, object]] = {}
    launch_walls: dict[str, float] = {}
    for row in rows:
        if str(row.get("record", "")) == "launch":
            launch_walls[str(row["leg_id"])] = float(row["process_wall_seconds"])
            continue
        if row.get("gate_sha256") != gate_sha256:
            return {
                "verdict": VERDICT_NOT_PRODUCED,
                "reason": f"row {row.get('leg_id')} bound to a different gate",
            }
        if str(row.get("kind")) == "native-endpoint-eval":
            subject = str(row["subject_leg_id"])
            if subject in oracle_rows:
                return {
                    "verdict": VERDICT_NOT_PRODUCED,
                    "reason": f"duplicate native re-evaluation of {subject}",
                }
            oracle_rows[subject] = row
            continue
        specification = row["specification"]
        role = str(specification.get("role"))
        if role == "prime":
            continue
        if role not in FINAL_PAIR_ROLES:
            return {
                "verdict": VERDICT_NOT_PRODUCED,
                "reason": f"unexpected pair role {role} in {row.get('leg_id')}",
            }
        # The role names its lane, so a role on the wrong lane is a mislabeled
        # measurement, not a slower one.
        lane_key = "native" if role in FINAL_PAIR_NATIVE_ROLES else "jax"
        if str(row.get("lane")) != lane_key:
            return {
                "verdict": VERDICT_NOT_PRODUCED,
                "reason": (
                    f"{row.get('leg_id')}: role {role} published on the "
                    f"{row.get('lane')} lane"
                ),
            }
        # Protocol symmetry: a wall leg that ran the discard-first protocol
        # publishes a wall containing a solve the paired GPU wall leg never
        # ran, and a warm leg without it publishes a cold first solve.
        if lane_key == "native":
            warm_protocol = bool(specification.get("warm_protocol", False))
            if warm_protocol != (role in FINAL_PAIR_WARM_PROTOCOL_ROLES):
                return {
                    "verdict": VERDICT_NOT_PRODUCED,
                    "reason": (
                        f"{row.get('leg_id')}: role {role} carries "
                        f"warm_protocol={warm_protocol}"
                    ),
                }
        mismatch = _selection_mismatch(selection, lane_key, specification)
        if mismatch is not None:
            return {
                "verdict": VERDICT_NOT_PRODUCED,
                "reason": f"{row.get('leg_id')}: {mismatch}",
            }
        pair_index = int(specification["pair_index"])
        legs = pairs.setdefault(pair_index, {})
        if role in legs:
            return {
                "verdict": VERDICT_NOT_PRODUCED,
                "reason": f"duplicate {role} leg in pair {pair_index}",
            }
        legs[role] = row
    if sorted(pairs) != list(range(FINAL_PAIR_COUNT)):
        return {
            "verdict": VERDICT_NOT_PRODUCED,
            "reason": f"expected {FINAL_PAIR_COUNT} pairs, found {sorted(pairs)}",
        }
    warm_ratios: list[float] = []
    wall_ratios: list[float] = []
    gate_failures: list[str] = []
    for pair_index in range(FINAL_PAIR_COUNT):
        legs = pairs[pair_index]
        if set(legs) != set(FINAL_PAIR_ROLES):
            return {
                "verdict": VERDICT_NOT_PRODUCED,
                "reason": (
                    f"pair {pair_index} lacks the {'/'.join(FINAL_PAIR_ROLES)} legs"
                ),
            }
        for key in FINAL_PAIR_NATIVE_ROLES:
            native_row = legs[key]
            native_steps = int(native_row["specification"]["max_steps"])
            # The native lane is measured as time to the frozen rung, so a leg
            # whose callback never fired produced no measurement:
            # rung-unreachability, not a slow lane.  Cap exhaustion is named
            # only when the solver spent its whole budget.
            if not bool(native_row["solver"].get("stopped_at_target", False)):
                native_nit = int(native_row["solver"]["nit"])
                termination = (
                    f"exhausted its {native_steps}-iteration cap"
                    if native_nit >= native_steps
                    else (
                        f"stopped at nit {native_nit} of its "
                        f"{native_steps}-iteration cap"
                    )
                )
                return {
                    "verdict": VERDICT_NOT_PRODUCED,
                    "reason": (
                        f"pair {pair_index} {key} {termination} "
                        "without reaching the frozen gate rung "
                        "(rung-unreachability), so it timed no quality "
                        f"(status {native_row['solver'].get('status')})"
                    ),
                }
            # The native legs are self-evaluated because they ARE the oracle.
            native_eligibility = evaluate_quality_gate(gate, native_row["endpoint"])
            if not native_eligibility["eligible"]:
                gate_failures.append(
                    f"pair {pair_index} {key}: {native_eligibility['failures']}"
                )
        for key in FINAL_PAIR_JAX_ROLES:
            evaluation = _gate_through_native_oracle(gate, legs[key], oracle_rows)
            if not evaluation["produced"]:
                return {
                    "verdict": VERDICT_NOT_PRODUCED,
                    "reason": evaluation["reason"],
                }
            if not evaluation["eligible"]:
                gate_failures.append(
                    f"pair {pair_index} {key}: {evaluation['failures']}"
                )
        # Only the warm legs feed the warm metric, and only the wall legs feed
        # the wall metric -- in both lanes, at symmetric repetition counts.
        expected_repetitions = {
            FINAL_PAIR_WARM_NATIVE_ROLE: FINAL_WARM_REPETITIONS,
            FINAL_PAIR_WALL_NATIVE_ROLE: 1,
            "warm": FINAL_WARM_REPETITIONS,
            "wall": 0,
        }
        for role_key, expected in expected_repetitions.items():
            declared = int(legs[role_key]["specification"].get("warm_repetitions", 1))
            if declared != expected:
                return {
                    "verdict": VERDICT_NOT_PRODUCED,
                    "reason": f"pair {pair_index} {role_key} declares "
                    f"warm_repetitions={declared}, contract requires {expected}",
                }
        native_warm_samples = legs[FINAL_PAIR_WARM_NATIVE_ROLE]["timings"][
            "warm_solve_seconds"
        ]
        jax_warm_samples = legs["warm"]["timings"]["warm_solve_seconds"]
        for label, samples in (
            ("native", native_warm_samples),
            ("GPU", jax_warm_samples),
        ):
            if (
                not isinstance(samples, list)
                or len(samples) != FINAL_WARM_REPETITIONS
                or not _all_finite(samples)
            ):
                return {
                    "verdict": VERDICT_NOT_PRODUCED,
                    "reason": f"pair {pair_index} {label} warm samples must be "
                    f"{FINAL_WARM_REPETITIONS} finite entries",
                }
        native_warm = _median(native_warm_samples)
        jax_warm = _median(jax_warm_samples)
        pair_walls = {
            "native": launch_walls.get(
                str(legs[FINAL_PAIR_WALL_NATIVE_ROLE]["leg_id"])
            ),
            "jax": launch_walls.get(str(legs["wall"]["leg_id"])),
        }
        if any(value is None for value in pair_walls.values()):
            return {
                "verdict": VERDICT_NOT_PRODUCED,
                "reason": f"pair {pair_index} lacks launch wall records",
            }
        if not all(
            _finite(value) for value in (native_warm, jax_warm, *pair_walls.values())
        ):
            return {
                "verdict": VERDICT_NOT_PRODUCED,
                "reason": f"pair {pair_index} has nonfinite timings",
            }
        # Every denominator is a measured GPU duration: a nonpositive one is
        # a broken clock, not an infinite speedup.
        if jax_warm <= 0.0 or float(pair_walls["jax"]) <= 0.0:
            return {
                "verdict": VERDICT_NOT_PRODUCED,
                "reason": f"pair {pair_index} has a nonpositive GPU denominator",
            }
        warm_ratios.append(native_warm / jax_warm)
        wall_ratios.append(float(pair_walls["native"]) / float(pair_walls["jax"]))
    if gate_failures:
        return {
            "verdict": VERDICT_CLOSED,
            "reason": "endpoint quality gate failures",
            "gate_failures": gate_failures,
            "warm_solve_ratios": warm_ratios,
            "process_wall_ratios": wall_ratios,
        }
    speed_pass = (
        _median(warm_ratios) >= FINAL_MEDIAN_MINIMUM_RATIO
        and _median(wall_ratios) >= FINAL_MEDIAN_MINIMUM_RATIO
        and all(ratio > FINAL_EVERY_PAIR_MINIMUM_RATIO for ratio in warm_ratios)
        and all(ratio > FINAL_EVERY_PAIR_MINIMUM_RATIO for ratio in wall_ratios)
    )
    return {
        "verdict": VERDICT_WIN if speed_pass else VERDICT_CLOSED,
        "warm_solve_ratios": warm_ratios,
        "process_wall_ratios": wall_ratios,
        "warm_solve_median_ratio": _median(warm_ratios),
        "process_wall_median_ratio": _median(wall_ratios),
        "median_minimum_ratio": FINAL_MEDIAN_MINIMUM_RATIO,
        "every_pair_minimum_ratio": FINAL_EVERY_PAIR_MINIMUM_RATIO,
    }


def _load_run_rows(run_directory: Path) -> tuple[Mapping[str, object], list[dict]]:
    manifest_path = run_directory / "manifest.json"
    if not manifest_path.is_file():
        raise BenchmarkError("run directory has no terminal manifest")
    manifest = _read_json(manifest_path)
    rows: list[dict] = []
    for relative, expected_sha in manifest["rows"].items():
        path = run_directory / relative
        if expected_sha == "absent" or not path.is_file():
            raise BenchmarkError(f"manifest names an absent row: {relative}")
        observed_sha = _sha256_file(path)
        if observed_sha != expected_sha:
            raise BenchmarkError(
                f"row {relative} hash mismatch: {observed_sha} != {expected_sha}"
            )
        document = _read_json(path)
        if not isinstance(document, dict):
            raise BenchmarkError(f"row {relative} is not a JSON object")
        rows.append(document)
    return manifest, rows


def _gate_from_run(run_directory: Path) -> dict[str, object]:
    """Read the quality contract this run bound itself to."""
    try:
        gate = _read_json(run_directory / "gate" / "quality_contract.json")
    except (OSError, json.JSONDecodeError) as error:
        raise BenchmarkError(f"gate artifact unreadable: {error}") from error
    if not isinstance(gate, dict):
        raise BenchmarkError("gate artifact is not a JSON object")
    return gate


def _selection_from_run(run_directory: Path) -> dict[str, object]:
    """Read the frozen selection the final pairs were launched under."""
    try:
        selection = _read_json(run_directory / "selection.json")
    except (OSError, json.JSONDecodeError) as error:
        raise BenchmarkError(f"selection artifact unreadable: {error}") from error
    if not isinstance(selection, dict) or "native" not in selection:
        raise BenchmarkError("selection artifact is not a selection document")
    return selection


def _cross_row_identity_failure(
    data_rows: Sequence[Mapping[str, object]],
) -> str | None:
    """Name the first runtime-identity field that forks across the run.

    One run is one machine state.  A commit, a dirty-file hash, a
    ``simsoptpp`` binary, a JAX version, or a device list that changes
    mid-run means the rows are not comparable with each other.
    """
    checks: dict[str, set[bytes]] = {
        "git.commit": set(),
        "git.changed_file_sha256": set(),
        "simsoptpp.sha256": set(),
    }
    jax_versions: set[bytes] = set()
    jax_devices: set[bytes] = set()
    for row in data_rows:
        identity = row.get("identity")
        identity = identity if isinstance(identity, Mapping) else {}
        git = identity.get("git")
        git = git if isinstance(git, Mapping) else {}
        simsoptpp = identity.get("simsoptpp")
        simsoptpp = simsoptpp if isinstance(simsoptpp, Mapping) else {}
        checks["git.commit"].add(_canonical_json_bytes(git.get("commit")))
        checks["git.changed_file_sha256"].add(
            _canonical_json_bytes(git.get("changed_file_sha256"))
        )
        checks["simsoptpp.sha256"].add(_canonical_json_bytes(simsoptpp.get("sha256")))
        if row.get("lane") == "jax":
            jax_versions.add(_canonical_json_bytes(identity.get("jax_version")))
            jax_devices.add(_canonical_json_bytes(identity.get("devices")))
    for name, values in checks.items():
        if len(values) > 1:
            return f"rows disagree on identity.{name}"
    if len(jax_versions) > 1:
        return "JAX rows disagree on identity.jax_version"
    if len(jax_devices) > 1:
        return "JAX rows disagree on identity.devices"
    return None


def _threading_conformance_failure(
    data_rows: Sequence[Mapping[str, object]],
) -> str | None:
    """A pinned leg must have observed the threading its spec declared.

    The measured OpenMP-reduction fork made per-process threading the axis
    every comparison stands on; a leg whose child observed different
    threading than the orchestrator declared is mislabeled evidence.
    """
    for row in data_rows:
        identity = row.get("identity")
        identity = identity if isinstance(identity, Mapping) else {}
        threading = identity.get("threading")
        if not isinstance(threading, Mapping):
            continue
        specification = row.get("specification")
        specification = specification if isinstance(specification, Mapping) else {}
        declared_omp = specification.get("omp_threads")
        if declared_omp is not None:
            environment = threading.get("environment")
            observed = (
                environment.get("OMP_NUM_THREADS")
                if isinstance(environment, Mapping)
                else None
            )
            if observed != str(int(declared_omp)):
                return (
                    f"{row.get('leg_id')} declared omp_threads={declared_omp} "
                    f"but its child observed OMP_NUM_THREADS={observed}"
                )
        declared_affinity = specification.get("cpu_affinity")
        if declared_affinity is not None:
            observed_affinity = threading.get("cpu_affinity")
            if observed_affinity != sorted(int(cpu) for cpu in declared_affinity):
                return (
                    f"{row.get('leg_id')} declared cpu_affinity "
                    f"{declared_affinity} but its child observed "
                    f"{observed_affinity}"
                )
    return None


def _fp64_conformance_failure(
    data_rows: Sequence[Mapping[str, object]],
) -> str | None:
    """Any leg that imported JAX must have observed fp64 enabled.

    Both lane environments pin ``JAX_ENABLE_X64``, but a pin in the launch
    row is an orchestrator claim; only the child's own observation counts.
    The 2026-08-17 taint showed a native leg whose transitive JAX ran
    float32 publishes gradients that are not the declared physics, while
    every identity and objective check stays green.
    """
    for row in data_rows:
        identity = row.get("identity")
        identity = identity if isinstance(identity, Mapping) else {}
        if identity.get("jax_imported") and identity.get("jax_enable_x64") is not True:
            return (
                f"{row.get('leg_id')} imported JAX but observed "
                f"jax_enable_x64={identity.get('jax_enable_x64')!r}"
            )
    return None


# Phases whose rows are measured against a previously frozen gate.  The gate
# phase itself derives the pins; baseline and the kernel canary consume none.
_GATE_CONSUMING_PHASES = frozenset({"native-matrix", "jax-sweep", "final-pairs"})

# The rung is derived from these sources' physics: a consumer built from
# different physics is measuring a different problem, so equality is
# fail-closed.
_PHYSICS_SOURCE_PINS: tuple[tuple[str, str], ...] = (
    ("objective_module_sha256", _OBJECTIVE_SOURCE),
    ("parity_case_sha256", _PARITY_CASE_SOURCE),
)
# Harness and plan sources may legitimately move between the gate commit and a
# consumer commit through dated amendments; their drift is published, never
# silently swallowed and never an equality requirement.
_DISCLOSED_SOURCE_PINS: tuple[tuple[str, str], ...] = (
    ("benchmark_sha256", _BENCHMARK_SOURCE),
    ("plan_sha256", _PLAN_SOURCE),
)


def _run_source_sha256(identity_git: Mapping[str, object], path: str) -> str:
    """The named source's content hash as the run actually executed it.

    A dirty file's content is bound by the row's own ``changed_file_sha256``;
    a clean file's content is bound by the row's commit.
    """
    changed = identity_git.get("changed_file_sha256")
    if isinstance(changed, Mapping) and path in changed:
        return str(changed[path])
    commit = str(identity_git.get("commit"))
    completed = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise BenchmarkError(
            f"cannot resolve {path} at commit {commit} for gate source conformance"
        )
    return hashlib.sha256(completed.stdout).hexdigest()


def _gate_source_conformance(
    gate: Mapping[str, object], data_rows: Sequence[Mapping[str, object]]
) -> tuple[str | None, dict[str, object]]:
    """Bind the gate's pinned sources to the consuming run's own sources.

    The gate freezes source fingerprints at derivation time; without this
    check they were republished but never enforced, so a consumer running
    changed physics would still validate (fail-open).  Physics pins are
    fail-closed; harness/plan pins are disclosed drift.
    """
    pinned = gate.get("source_fingerprints")
    if not isinstance(pinned, Mapping):
        return "gate publishes no source fingerprints", {}
    identity_git = next(
        (
            row["identity"]["git"]
            for row in data_rows
            if isinstance(row.get("identity"), Mapping)
            and isinstance(row["identity"].get("git"), Mapping)
        ),
        None,
    )
    if identity_git is None:
        return "no data row carries git identity for gate source conformance", {}
    for key, path in _PHYSICS_SOURCE_PINS:
        run_sha = _run_source_sha256(identity_git, path)
        if run_sha != pinned.get(key):
            return (
                (
                    f"gate pins {key}={str(pinned.get(key))[:12]} but the run"
                    f" executed {run_sha[:12]} — the frozen rung refers to"
                    " different physics"
                ),
                {},
            )
    drift: dict[str, object] = {}
    for key, path in _DISCLOSED_SOURCE_PINS:
        run_sha = _run_source_sha256(identity_git, path)
        drift[key] = {
            "gate": pinned.get(key),
            "run": run_sha,
            "identical": run_sha == pinned.get(key),
        }
    return None, drift


def _worktree_evidence(
    data_rows: Sequence[Mapping[str, object]], manifest: Mapping[str, object]
) -> dict[str, object]:
    identities = [
        row["identity"] for row in data_rows if isinstance(row.get("identity"), Mapping)
    ]
    orchestrator = manifest.get("orchestrator_identity")
    if not identities and isinstance(orchestrator, Mapping):
        identities = [orchestrator]
    statuses = [
        identity["git"].get("status", [])
        for identity in identities
        if isinstance(identity.get("git"), Mapping)
    ]
    commits = [
        identity["git"].get("commit")
        for identity in identities
        if isinstance(identity.get("git"), Mapping)
    ]
    dirty_counts = [len(status) for status in statuses]
    return {
        "evidence_grade": (
            "clean-tree"
            if all(count == 0 for count in dirty_counts)
            else "dirty-tree-diagnostic"
        ),
        "worktree": {
            "commit": commits[0] if commits else None,
            "dirty_file_count": max(dirty_counts) if dirty_counts else 0,
        },
    }


# The JAX lane has no host trajectory (the fused loop reports no accepted
# steps), so its budget is not a measured crossing iteration.
_JAX_BUDGET_ANNOTATION = (
    "the JAX max_steps budget is an upper bound on the true crossing "
    "iteration, taken from the preregistered coarse ladder "
    f"{list(GPU_BUDGET_SWEEP)}"
)


def _solver_counters(
    data_rows: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    counters: dict[str, dict[str, object]] = {}
    for row in data_rows:
        solver = row.get("solver")
        if isinstance(solver, Mapping):
            counters[str(row.get("leg_id"))] = {
                "nit": solver.get("nit"),
                "nfev": solver.get("nfev"),
                "status": solver.get("status"),
                "max_steps": solver.get("max_steps"),
            }
    return counters


def validate_run(run_directory: Path) -> dict[str, object]:
    """Recompute the phase verdict from raw rows; fail-closed everywhere."""
    try:
        manifest, rows = _load_run_rows(run_directory)
    except (BenchmarkError, json.JSONDecodeError, KeyError, OSError) as error:
        return {
            "schema": VERDICT_SCHEMA,
            "verdict": VERDICT_NOT_PRODUCED,
            "reason": str(error),
        }
    phase = str(manifest.get("phase"))
    data_rows = [row for row in rows if str(row.get("record", "")) != "launch"]
    try:
        missing_fingerprints = [
            str(row.get("leg_id")) for row in data_rows if "fingerprints" not in row
        ]
        if missing_fingerprints:
            raise BenchmarkError(
                f"data rows without input fingerprints: {missing_fingerprints}"
            )
        fingerprints = {_canonical_json_bytes(row["fingerprints"]) for row in data_rows}
        if len(fingerprints) > 1:
            raise BenchmarkError("input fingerprints differ between rows")
        identity_failure = _cross_row_identity_failure(data_rows)
        if identity_failure is not None:
            raise BenchmarkError(identity_failure)
        threading_failure = _threading_conformance_failure(data_rows)
        if threading_failure is not None:
            raise BenchmarkError(threading_failure)
        fp64_failure = _fp64_conformance_failure(data_rows)
        if fp64_failure is not None:
            raise BenchmarkError(fp64_failure)
        gate_source_drift: dict[str, object] | None = None
        if phase in _GATE_CONSUMING_PHASES:
            conformance_failure, gate_source_drift = _gate_source_conformance(
                _gate_from_run(run_directory), data_rows
            )
            if conformance_failure is not None:
                raise BenchmarkError(conformance_failure)
        if phase == "baseline":
            reduction = reduce_baseline(data_rows)
        elif phase == "kernel-canary":
            reduction = reduce_kernel_canary(data_rows)
        elif phase == "gate":
            gate = _gate_from_run(run_directory)
            reduction = {
                "verdict": "GATE_FROZEN",
                "gate_sha256": _sha256_bytes(_canonical_json_bytes(gate)),
                "target_objective": gate["target_objective"],
                "reference_budget": gate["reference_budget"],
                "source_fingerprints": gate["source_fingerprints"],
            }
        elif phase == "final-pairs":
            reduction = reduce_final_pairs(
                _gate_from_run(run_directory),
                _selection_from_run(run_directory),
                rows,
            )
        elif phase == "native-matrix":
            reduction = reduce_native_matrix(_gate_from_run(run_directory), rows)
        elif phase == "jax-sweep":
            reduction = reduce_jax_sweep(_gate_from_run(run_directory), data_rows)
        elif phase == "hlo-capture":
            if len(data_rows) != 1 or "census" not in data_rows[0]:
                reduction = {
                    "verdict": VERDICT_NOT_PRODUCED,
                    "reason": "expected exactly one jax-hlo row with a census",
                }
            else:
                row = data_rows[0]
                reduction = {
                    "verdict": "HLO_CAPTURED",
                    "objective_source": row["objective_source"],
                    "artifacts": row["artifacts"],
                    "census_operation_count": sum(row["census"].values()),
                }
        else:
            reduction = {
                "verdict": VERDICT_NOT_PRODUCED,
                "reason": f"unknown phase {phase}",
            }
        if gate_source_drift is not None:
            reduction["gate_source_drift"] = gate_source_drift
    except (
        BenchmarkError,
        json.JSONDecodeError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
        IndexError,
        ZeroDivisionError,
        StopIteration,
    ) as error:
        return {
            "schema": VERDICT_SCHEMA,
            "phase": phase,
            "verdict": VERDICT_NOT_PRODUCED,
            "reason": f"{type(error).__name__}: {error}",
        }
    verdict: dict[str, object] = {
        "schema": VERDICT_SCHEMA,
        "phase": phase,
        "cache_policy": manifest.get("cache_policy"),
        **_worktree_evidence(data_rows, manifest),
        **reduction,
    }
    if phase in {"jax-sweep", "final-pairs"}:
        verdict["solver_counters"] = _solver_counters(data_rows)
        verdict["solver_counters_annotation"] = _JAX_BUDGET_ANNOTATION
    if phase in {"native-matrix", "final-pairs"}:
        verdict["native_stop_cap"] = NATIVE_STOP_MAX_STEPS
        verdict["native_stop_annotation"] = NATIVE_STOP_ANNOTATION
    return verdict


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def reduce_hlo_diff(
    before: Mapping[str, object], after: Mapping[str, object]
) -> dict[str, object]:
    """Classify a pre/post refactor HLO pair: DCE_NULL or CHANGED.

    An empty census or cost analysis is not evidence of "no change": it is a
    capture that failed, so it classifies ``NOT_PRODUCED`` rather than
    ``DCE_NULL``.
    """
    for label, row in (("before", before), ("after", after)):
        if not row.get("census"):
            return {
                "classification": VERDICT_NOT_PRODUCED,
                "reason": f"{label} row has an empty operation census",
            }
        if not row.get("cost_analysis"):
            return {
                "classification": VERDICT_NOT_PRODUCED,
                "reason": f"{label} row has an empty cost analysis",
            }
    if before.get("fingerprints") != after.get("fingerprints"):
        return {
            "classification": VERDICT_NOT_PRODUCED,
            "reason": "the two captures used different input bundles",
        }
    census_before = dict(before["census"])
    census_after = dict(after["census"])
    census_delta = {
        opcode: census_after.get(opcode, 0) - census_before.get(opcode, 0)
        for opcode in sorted(set(census_before) | set(census_after))
        if census_after.get(opcode, 0) != census_before.get(opcode, 0)
    }
    cost_before = dict(before["cost_analysis"])
    cost_after = dict(after["cost_analysis"])
    cost_delta = {
        name: (cost_before.get(name), cost_after.get(name))
        for name in sorted(set(cost_before) | set(cost_after))
        if cost_before.get(name) != cost_after.get(name)
    }
    stablehlo_changed = (
        before["artifacts"]["stablehlo_sha256"]
        != after["artifacts"]["stablehlo_sha256"]
    )
    return {
        "classification": "DCE_NULL"
        if not census_delta and not cost_delta
        else "CHANGED",
        "stablehlo_changed": stablehlo_changed,
        "optimized_hlo_sha256": {
            "before": before["artifacts"].get("optimized_hlo_sha256"),
            "after": after["artifacts"].get("optimized_hlo_sha256"),
        },
        "census_delta": census_delta,
        "cost_delta": cost_delta,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        allow_abbrev=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="phase", required=True)
    subparsers.add_parser("baseline")
    subparsers.add_parser("gate")
    subparsers.add_parser("kernel-canary")
    subparsers.add_parser("hlo-capture")
    diff_parser = subparsers.add_parser("hlo-diff")
    diff_parser.add_argument("before_run", type=Path)
    diff_parser.add_argument("after_run", type=Path)
    matrix_parser = subparsers.add_parser("native-matrix")
    matrix_parser.add_argument("--gate", type=Path, required=True)
    sweep_parser = subparsers.add_parser("jax-sweep")
    sweep_parser.add_argument("--gate", type=Path, required=True)
    freeze_parser = subparsers.add_parser("freeze-selection")
    freeze_parser.add_argument("native_matrix_run", type=Path)
    freeze_parser.add_argument("jax_sweep_run", type=Path)
    freeze_parser.add_argument("--output", type=Path, required=True)
    pairs_parser = subparsers.add_parser("final-pairs")
    pairs_parser.add_argument("--gate", type=Path, required=True)
    pairs_parser.add_argument("--selection", type=Path, required=True)
    pairs_parser.add_argument(
        "--cache-policy", choices=("primed", "fresh"), default="primed"
    )
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("run_directory", type=Path)
    leg_parser = subparsers.add_parser("leg")
    leg_parser.add_argument("--spec", type=Path, required=True)
    leg_parser.add_argument("--row", type=Path, required=True)
    arguments = parser.parse_args(argv)

    if arguments.phase == "leg":
        _run_leg(arguments.spec, arguments.row)
        return 0
    if arguments.phase == "validate":
        verdict = validate_run(arguments.run_directory.resolve())
        print(json.dumps(verdict, indent=2, sort_keys=True))
        return 0 if verdict["verdict"] not in {VERDICT_NOT_PRODUCED} else 1
    if arguments.phase == "hlo-diff":
        rows = []
        for run in (arguments.before_run, arguments.after_run):
            _, run_rows = _load_run_rows(run.resolve())
            captures = [row for row in run_rows if row.get("kind") == "jax-hlo"]
            if not captures:
                raise BenchmarkError(f"{run} holds no jax-hlo capture row")
            rows.append(captures[0])
        print(json.dumps(reduce_hlo_diff(*rows), indent=2, sort_keys=True))
        return 0
    if arguments.phase == "freeze-selection":
        selections: list[dict[str, object]] = []
        gate_shas: list[object] = []
        for run, expected in (
            (arguments.native_matrix_run, "NATIVE_SELECTED"),
            (arguments.jax_sweep_run, "JAX_SELECTED"),
        ):
            verdict = validate_run(run.resolve())
            if verdict["verdict"] != expected:
                raise BenchmarkError(f"{run} did not validate to {expected}: {verdict}")
            manifest_path = run.resolve() / "manifest.json"
            manifest = _read_json(manifest_path)
            gate_shas.append(manifest["gate_sha256"])
            selections.append(
                {
                    **verdict["selected"],
                    "run_directory": str(run.resolve()),
                    "manifest_sha256": _sha256_file(manifest_path),
                }
            )
        if gate_shas[0] != gate_shas[1]:
            raise BenchmarkError("selection runs bind different gates")
        document = {
            "schema": SELECTION_SCHEMA,
            "gate_sha256": gate_shas[0],
            "native": selections[0],
            "jax": selections[1],
        }
        selection_sha256 = _write_json_exclusive(arguments.output, document)
        print(f"selection sha256 {selection_sha256}")
        print(json.dumps(document, indent=2, sort_keys=True))
        return 0
    phase_runners = {
        "baseline": _phase_baseline,
        "gate": _phase_gate,
        "kernel-canary": _phase_kernel_canary,
        "hlo-capture": _phase_hlo_capture,
        "native-matrix": _phase_native_matrix,
        "jax-sweep": _phase_jax_sweep,
        "final-pairs": _phase_final_pairs,
    }
    run_directory = phase_runners[arguments.phase](arguments)
    verdict = validate_run(run_directory)
    print(f"run directory {run_directory}")
    print(json.dumps(verdict, indent=2, sort_keys=True))
    return 0 if verdict["verdict"] not in {VERDICT_NOT_PRODUCED} else 1


if __name__ == "__main__":
    raise SystemExit(main())
