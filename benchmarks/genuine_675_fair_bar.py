"""Genuine-675 fair-bar campaign harness.

Charter (frozen): docs/jax_gpu_genuine675_fair_bar_plan.md on
pr/jax-port-squashed at commit 7b6d69041 (sha256 embedded below).  This
orchestrator re-measures the archived flat-675 GPU-vs-native comparison
under the program law: OMP-swept pinned native denominator, symmetric
discarded primers, interleaved pairs, preregistered process-wall timer,
hard work-matching/endpoint/oracle gates, per-row campaign-contract sha
binding, and fail-closed verdicts.

The timed instrument is the archived lane driver
(benchmarks/genuine_675_dynamic_lane.py) invoked unmodified; this harness
owns only environment construction, sequencing, gating, and evidence.
Subcommands: mint-manifest, selftest-loader, phase1, probe, native-matrix,
pairs, validate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

# The measurement instrument lives in a worktree pinned at the archived
# launch-source commit (the frozen input bundle binds it); this harness file
# lives on pr/jax-port-squashed and must always run with
# PYTHONPATH=<source-root>:<source-root>/src so every simsopt_jax import
# below resolves from the pinned instrument tree, never from this tree.
DEFAULT_SOURCE_ROOT = Path("/home/jungdaesuh/code/columbia/simsopt-genuine675-fairbar")
DEFAULT_OUTPUT_ROOT = Path(
    "/home/jungdaesuh/simsopt_mixed_artifacts/genuine675_fair_bar"
)
INSTRUMENT_COMMIT = "1c23f6c5f8964c74cc60f63d81b7f93f2db852f3"

from simsopt_jax.runtime.genuine_675_dynamic import Genuine675LbfgsbPolicy
from simsopt_jax.runtime.single_stage_fullspace_675 import GENUINE_FULLSPACE_675
from simsopt_jax.runtime.validation_ladder_common import repo_pythonpath_env

# --------------------------------------------------------------------------
# Frozen campaign identity
# --------------------------------------------------------------------------

CHARTER_SHA256 = "537d621b456dd15688fd960e88c6f15c66f6a03739245d5c160b3ada7e8f0fdb"
CHARTER_COMMIT = "2f0381cde"
CHARTER_PATH = "docs/jax_gpu_genuine675_fair_bar_plan.md (pr/jax-port-squashed)"

SOURCE_MANIFEST_SHA256 = (
    "84febc05d195d84c0802205b2b4c85ea1fa38faa7ff856efca7c12d980647c0c"
)
SOURCE_MANIFEST_SEMANTIC_SHA256 = (
    "8dc7149eb1a878b4efc95d2212f620b5c4771e2232991c1852e1e836d32ebd7c"
)
SOURCE_MEMBER_FILES: Mapping[str, Mapping[str, object]] = {
    "candidate_source": {
        "relative_path": "candidate_source.events.ndjson",
        "sha256": "d25cc2985c1d30fe07f06c40d62d37b90c0f290de7b3eb5b224eea097f91ea2a",
        "byte_count": 9632504,
    },
    "equilibrium": {
        "relative_path": "equilibrium.nc",
        "sha256": "b62feb914799f00a970a92fa0c268de289a0418cb3743948699adf75668e578a",
        "byte_count": 548773,
    },
    "historical_launch_identity": {
        "relative_path": "historical_launch_identity.json",
        "sha256": "cc3ca5677f44ae1851a7807084fb0ca39d55eaf59f94b4cb6d4803e477387012",
        "byte_count": 15083,
    },
    "native_biot_savart": {
        "relative_path": "native_biot_savart.json",
        "sha256": "0415ae937c78b9f2d68e8463a9176e8f330a9aa172eece160341afccdc29429d",
        "byte_count": 165261,
    },
    "runtime_spec": {
        "relative_path": "single_stage_jax_runtime_spec.json",
        "sha256": "8dff9142c6859141d96ce91e5e9d08eb69b4cecaaeabf56579ea930f6192c0b7",
        "byte_count": 1010944,
    },
    "vessel_material": {
        "relative_path": "vessel_material.json",
        "sha256": "c9c68617e6833640ec080f738178fff2156905c150153146e684872b72441300",
        "byte_count": 331870,
    },
}

CAMPAIGN_MANIFEST_SCHEMA = "genuine-675-fair-bar-input.v1"
ROW_SCHEMA = "genuine-675-fair-bar-row.v1"
RUN_MANIFEST_SCHEMA = "genuine-675-fair-bar-manifest.v1"

BUDGET_CONTINUITY = 3
BUDGET_HEADLINE = 50
OMP_MATRIX = (1, 2, 4, 8, 16, 32, 64)
PHYSICAL_CORES = tuple(range(32))
ALL_CPUS = tuple(range(64))
MATRIX_REPS = 3
PAIR_COUNT = 5
NOT_PRODUCED_ABORT = 3
WIN_MEDIAN_THRESHOLD = 1.10
SEQUENCE_RTOL = 1.0e-10
ENDPOINT_OBJECTIVE_RTOL = 1.0e-10
ENDPOINT_GRADIENT_RTOL = 1.0e-8
IDLE_LOAD_MAX = 1.0
IDLE_GPU_UTIL_MAX = 5
IDLE_POLL_SECONDS = 30
IDLE_MAX_WAIT_SECONDS = 7200
NARROWING_FACTOR = 2.0

SOURCE_ROOT = DEFAULT_SOURCE_ROOT
OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT

NATIVE_LANE = "native_cpp_cpu"
GPU_LANE = "jax_gpu_fp64"
FIXED_BUDGET_MODE = "fixed_budget_diagnostic"


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def policy_for_budget(budget: int) -> Genuine675LbfgsbPolicy:
    return Genuine675LbfgsbPolicy.production_unbounded(
        maxiter=budget,
        common_objective_target=None,
    )


def contract_sha256(*, campaign_manifest_sha256: str, budget: int) -> str:
    return _sha256_bytes(
        _canonical_bytes(
            {
                "charter_sha256": CHARTER_SHA256,
                "formulation_semantic_sha256": GENUINE_FULLSPACE_675.semantic_sha256,
                "campaign_input_manifest_sha256": campaign_manifest_sha256,
                "policy_semantic_sha256": policy_for_budget(budget).semantic_sha256,
            }
        )
    )


# --------------------------------------------------------------------------
# Campaign input manifest (the dated reclassification, charter section
# "Input-bundle eligibility clause")
# --------------------------------------------------------------------------


def mint_campaign_manifest(source_manifest: Path, output: Path) -> str:
    observed = _sha256_file(source_manifest)
    if observed != SOURCE_MANIFEST_SHA256:
        raise ValueError(
            "Source manifest bytes moved: "
            f"observed {observed}, frozen {SOURCE_MANIFEST_SHA256}."
        )
    payload = {
        "schema": CAMPAIGN_MANIFEST_SCHEMA,
        "charter": {
            "path": CHARTER_PATH,
            "commit": CHARTER_COMMIT,
            "sha256": CHARTER_SHA256,
        },
        "source_manifest": {
            "sha256": SOURCE_MANIFEST_SHA256,
            "semantic_sha256": SOURCE_MANIFEST_SEMANTIC_SHA256,
        },
        "performance_eligible": True,
        "reclassification": (
            "Dated pre-evidence reclassification per the frozen charter: the "
            "member bytes are identical to the archived parity bundle; the "
            "old contract's performance_eligible=false bound the old "
            "campaign's evidentiary intent, and both lanes here receive the "
            "same bytes, so input-quality concerns cancel in the ratio. "
            "Scope limit carried to the receipt: the starting candidate is "
            "one mid-trajectory native iterate, not an ensemble."
        ),
        "files": {
            name: dict(entry) for name, entry in sorted(SOURCE_MEMBER_FILES.items())
        },
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return _sha256_file(output)


def load_campaign_manifest(manifest_path: Path, input_dir: Path) -> str:
    """Fail-closed campaign-manifest load: verify schema and every member.

    Returns the campaign manifest file sha256 on success.  Raises on any
    schema, hash, or byte-count mismatch (charter: a hash mismatch at load
    is NOT_PRODUCED).
    """
    payload = json.loads(manifest_path.read_text())
    if payload.get("schema") != CAMPAIGN_MANIFEST_SCHEMA:
        raise ValueError(f"Campaign manifest schema is {payload.get('schema')!r}.")
    if payload.get("charter", {}).get("sha256") != CHARTER_SHA256:
        raise ValueError("Campaign manifest binds a foreign charter.")
    if payload.get("performance_eligible") is not True:
        raise ValueError("Campaign manifest must declare performance_eligible.")
    if payload.get("source_manifest", {}).get("sha256") != SOURCE_MANIFEST_SHA256:
        raise ValueError("Campaign manifest binds a foreign source manifest.")
    files = payload.get("files")
    if not isinstance(files, dict) or sorted(files) != sorted(SOURCE_MEMBER_FILES):
        raise ValueError("Campaign manifest member set differs from the freeze.")
    for name, frozen in SOURCE_MEMBER_FILES.items():
        entry = files[name]
        if (
            entry.get("sha256") != frozen["sha256"]
            or entry.get("byte_count") != frozen["byte_count"]
            or entry.get("relative_path") != frozen["relative_path"]
        ):
            raise ValueError(f"Campaign manifest entry {name!r} drifted.")
        member = input_dir / str(frozen["relative_path"])
        if not member.is_file():
            raise ValueError(f"Member file missing: {member}.")
        size = member.stat().st_size
        if size != frozen["byte_count"]:
            raise ValueError(
                f"Member {name!r} byte count {size} != {frozen['byte_count']}."
            )
        observed = _sha256_file(member)
        if observed != frozen["sha256"]:
            raise ValueError(
                f"Member {name!r} sha256 {observed} != {frozen['sha256']}."
            )
    return _sha256_file(manifest_path)


def selftest_loader(source_manifest: Path, scratch: Path) -> dict[str, object]:
    """Charter-required loader test: tampered-byte and missing-file cases."""
    input_dir = source_manifest.parent
    scratch.mkdir(parents=True, exist_ok=True)
    campaign_manifest = scratch / "campaign_input_manifest.json"
    mint_campaign_manifest(source_manifest, campaign_manifest)

    load_campaign_manifest(campaign_manifest, input_dir)

    tampered_dir = scratch / "tampered"
    if tampered_dir.exists():
        shutil.rmtree(tampered_dir)
    tampered_dir.mkdir()
    for entry in SOURCE_MEMBER_FILES.values():
        rel = str(entry["relative_path"])
        shutil.copyfile(input_dir / rel, tampered_dir / rel)
    victim = tampered_dir / str(
        SOURCE_MEMBER_FILES["historical_launch_identity"]["relative_path"]
    )
    corrupted = bytearray(victim.read_bytes())
    corrupted[0] ^= 0xFF
    victim.write_bytes(bytes(corrupted))
    try:
        load_campaign_manifest(campaign_manifest, tampered_dir)
    except ValueError as error:
        tampered_rejected = "sha256" in str(error)
    else:
        tampered_rejected = False

    missing_dir = scratch / "missing"
    if missing_dir.exists():
        shutil.rmtree(missing_dir)
    missing_dir.mkdir()
    for name, entry in SOURCE_MEMBER_FILES.items():
        if name == "vessel_material":
            continue
        rel = str(entry["relative_path"])
        shutil.copyfile(input_dir / rel, missing_dir / rel)
    try:
        load_campaign_manifest(campaign_manifest, missing_dir)
    except ValueError as error:
        missing_rejected = "missing" in str(error)
    else:
        missing_rejected = False

    passed = tampered_rejected and missing_rejected
    return {
        "schema": "genuine-675-fair-bar-loader-selftest.v1",
        "intact_load": True,
        "tampered_byte_rejected": tampered_rejected,
        "missing_file_rejected": missing_rejected,
        "passed": passed,
    }


# --------------------------------------------------------------------------
# Child environments, affinity, and provenance
# --------------------------------------------------------------------------

_SITECUSTOMIZE = '''\
"""Fair-bar in-child provenance capture.

Affinity is recorded twice: at interpreter start (before any OpenMP
runtime initializes, i.e. the mask the launcher actually granted) and at
exit (after OMP_PROC_BIND may have re-bound the master thread).
"""
import atexit
import json
import os

_AFFINITY_AT_IMPORT = sorted(os.sched_getaffinity(0))


def _fair_bar_provenance() -> None:
    out = os.environ.get("FAIR_BAR_PROVENANCE_OUT")
    if not out:
        return
    record = {"pid": os.getpid()}
    record["sched_affinity_at_import"] = _AFFINITY_AT_IMPORT
    record["sched_affinity_at_exit"] = sorted(os.sched_getaffinity(0))
    record["cpu_count"] = os.cpu_count()
    record["env"] = {
        name: os.environ.get(name)
        for name in (
            "OMP_NUM_THREADS",
            "OMP_PLACES",
            "OMP_PROC_BIND",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "JAX_ENABLE_X64",
            "JAX_PLATFORMS",
        )
    }
    libgomp = None
    openmp_max_threads = None
    try:
        with open("/proc/self/maps") as handle:
            for line in handle:
                if "libgomp" in line:
                    libgomp = line.rsplit(" ", 1)[-1].strip()
                    break
    except OSError:
        pass
    if libgomp is not None:
        try:
            import ctypes

            openmp_max_threads = ctypes.CDLL(libgomp).omp_get_max_threads()
        except (OSError, AttributeError):
            openmp_max_threads = None
    record["libgomp_path"] = libgomp
    record["omp_get_max_threads"] = openmp_max_threads
    try:
        with open("/proc/cpuinfo") as handle:
            for line in handle:
                if line.startswith("model name"):
                    record["cpu_model"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    with open(out, "w") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)


atexit.register(_fair_bar_provenance)
'''


def write_provenance_shim(run_root: Path) -> Path:
    shim_dir = run_root / "provenance_shim"
    shim_dir.mkdir(parents=True, exist_ok=True)
    (shim_dir / "sitecustomize.py").write_text(_SITECUSTOMIZE)
    return shim_dir


@dataclass(frozen=True)
class NativeConfig:
    """One native threading configuration from the charter's matrix."""

    label: str
    omp_threads: int | None  # None = unpinned July-condition disclosure leg
    affinity: tuple[int, ...] | None

    @classmethod
    def pinned(cls, threads: int) -> NativeConfig:
        affinity = PHYSICAL_CORES if threads <= 32 else ALL_CPUS
        return cls(label=f"omp{threads}", omp_threads=threads, affinity=affinity)

    @classmethod
    def unpinned_default(cls) -> NativeConfig:
        return cls(label="unpinned-default", omp_threads=None, affinity=None)


def native_environment(
    config: NativeConfig,
    *,
    shim_dir: Path,
) -> dict[str, str]:
    environment = repo_pythonpath_env(
        environment=os.environ,
        platform="cpu",
        backend_mode="native_cpu",
        backend_strict=True,
        mixed_precision_enabled=False,
    )
    if config.omp_threads is not None:
        environment["OMP_NUM_THREADS"] = str(config.omp_threads)
        environment["OMP_PLACES"] = "cores"
        environment["OMP_PROC_BIND"] = "close"
        environment["OPENBLAS_NUM_THREADS"] = "1"
    environment["PYTHONPATH"] = f"{shim_dir}{os.pathsep}{environment['PYTHONPATH']}"
    return environment


def gpu_environment(
    *,
    cache_dir: Path,
    shim_dir: Path,
) -> dict[str, str]:
    environment = repo_pythonpath_env(
        environment=os.environ,
        platform="cuda",
        backend_mode="jax_gpu_parity",
        backend_strict=True,
        mixed_precision_enabled=False,
        production_cuda=True,
    )
    resolved = str(cache_dir.resolve())
    environment["JAX_COMPILATION_CACHE_DIR"] = resolved
    environment["SIMSOPT_JAX_COMPILATION_CACHE_DIR"] = resolved
    environment["PYTHONPATH"] = f"{shim_dir}{os.pathsep}{environment['PYTHONPATH']}"
    return environment


SCRUBBED_THREADING_NAMES = (
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def enforce_child_conformance(
    *,
    lane: str,
    config_label: str,
    omp_threads: int | None,
    affinity: tuple[int, ...] | None,
    provenance: Mapping[str, object],
) -> None:
    """Fail a leg closed unless the child-observed state matches the pins.

    The declared environment proves what was requested; only the child's
    resolved state (env echo, libgomp thread count, granted affinity mask)
    proves what actually ran — the finite-build fp64-taint lesson.
    """
    if not provenance:
        raise RuntimeError(
            f"{lane} ({config_label}): no child provenance was captured."
        )
    child_env = provenance.get("env")
    has_env_echo = isinstance(child_env, Mapping)
    if not has_env_echo:
        raise RuntimeError(f"{lane} ({config_label}): provenance has no env echo.")
    failures: list[str] = []
    if child_env.get("JAX_ENABLE_X64") != "1":
        failures.append("x64_unobserved")
    for name in SCRUBBED_THREADING_NAMES:
        if child_env.get(name) is not None:
            failures.append(f"{name.lower()}_leaked")
    if lane == NATIVE_LANE:
        if omp_threads is not None:
            expected = {
                "OMP_NUM_THREADS": str(omp_threads),
                "OMP_PLACES": "cores",
                "OMP_PROC_BIND": "close",
                "OPENBLAS_NUM_THREADS": "1",
            }
            for name, value in expected.items():
                if child_env.get(name) != value:
                    failures.append(f"{name.lower()}_unobserved")
            resolved = provenance.get("omp_get_max_threads")
            if resolved != omp_threads:
                failures.append(
                    f"omp_get_max_threads_resolved_{resolved}_declared_{omp_threads}"
                )
        else:
            for name in (
                "OMP_NUM_THREADS",
                "OMP_PLACES",
                "OMP_PROC_BIND",
                "OPENBLAS_NUM_THREADS",
            ):
                if child_env.get(name) is not None:
                    failures.append(f"disclosure_{name.lower()}_set")
    granted = provenance.get("sched_affinity_at_import")
    expected_mask = sorted(affinity) if affinity is not None else sorted(ALL_CPUS)
    if granted != expected_mask:
        failures.append(
            f"affinity_granted_{len(granted) if isinstance(granted, list) else '?'}"
            f"_expected_{len(expected_mask)}"
        )
    if failures:
        raise RuntimeError(
            f"{lane} ({config_label}) child-conformance failures: {failures}"
        )


def wait_for_idle_box() -> float:
    """Fail-closed idle gate: block timed legs while foreign compute is live."""
    waited = 0.0
    while True:
        load1 = os.getloadavg()[0]
        gpu_util = _gpu_utilization_percent()
        if load1 <= IDLE_LOAD_MAX and gpu_util <= IDLE_GPU_UTIL_MAX:
            return waited
        if waited >= IDLE_MAX_WAIT_SECONDS:
            raise RuntimeError(
                f"Box never went idle (load {load1:.2f}, gpu {gpu_util}%) "
                f"within {IDLE_MAX_WAIT_SECONDS}s."
            )
        time.sleep(IDLE_POLL_SECONDS)
        waited += IDLE_POLL_SECONDS


def _gpu_utilization_percent() -> int:
    completed = subprocess.run(
        (
            "nvidia-smi",
            "--query-gpu=utilization.gpu",
            "--format=csv,noheader,nounits",
        ),
        capture_output=True,
        text=True,
        check=True,
    )
    return max(int(line) for line in completed.stdout.split() if line.strip())


# --------------------------------------------------------------------------
# Leg execution against the archived lane driver
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LegResult:
    lane: str
    budget: int
    timed: bool
    launcher_wall_seconds: float
    process_wall_seconds: float
    optimizer_seconds: float
    compact_candidate_evaluations: int
    accepted_callback_count: int
    accepted_objectives: tuple[float, ...]
    per_eval_seconds: tuple[float, ...]
    certificate_attempt_count: int
    certificate_seconds: float
    endpoint_objective: float
    endpoint_gradient_inf_norm: float
    endpoint_candidate: Mapping[str, object]
    endpoint_inner_state: tuple[float, float]
    termination_reason: str
    scipy_status: Mapping[str, object]
    lane_json_sha256: str
    provenance: Mapping[str, object]
    row_path: Path


def run_leg(
    *,
    lane: str,
    budget: int,
    environment: dict[str, str],
    omp_threads: int | None,
    affinity: tuple[int, ...] | None,
    leg_root: Path,
    timed: bool,
    role: str,
    config_label: str,
    campaign_manifest_sha256: str,
    source_manifest: Path,
    git_identity: Mapping[str, object],
) -> LegResult:
    leg_root.mkdir(parents=True, exist_ok=False)
    output_path = leg_root / "lane.json"
    provenance_out = leg_root / "child_provenance.json"
    environment = dict(environment)
    environment["FAIR_BAR_PROVENANCE_OUT"] = str(provenance_out)
    policy = policy_for_budget(budget)
    command = (
        sys.executable,
        str(SOURCE_ROOT / "benchmarks" / "genuine_675_dynamic_lane.py"),
        "--platform",
        "cpu" if lane == NATIVE_LANE else "cuda",
        "--lane",
        lane,
        "--input-manifest",
        str(source_manifest),
        "--output-json",
        str(output_path),
        "--maxiter",
        str(budget),
        "--measurement-mode",
        FIXED_BUDGET_MODE,
        "--expected-policy-sha256",
        policy.semantic_sha256,
        "--expected-formulation-sha256",
        GENUINE_FULLSPACE_675.semantic_sha256,
    )
    idle_wait = wait_for_idle_box() if timed else 0.0

    def _preexec() -> None:
        if affinity is not None:
            os.sched_setaffinity(0, affinity)

    started = time.perf_counter()
    with (leg_root / "stdout.log").open("xb") as stdout, (leg_root / "stderr.log").open(
        "xb"
    ) as stderr:
        completed = subprocess.run(
            command,
            cwd=SOURCE_ROOT,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            preexec_fn=_preexec if affinity is not None else None,
            check=False,
        )
    launcher_wall = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"Lane {lane} (budget {budget}, {role}) exited "
            f"{completed.returncode}; see {leg_root / 'stderr.log'}."
        )
    lane_bytes = output_path.read_bytes()
    payload = json.loads(lane_bytes)
    result = payload["result"]
    trace = result["trace"]
    accepted = [entry for entry in trace if entry["accepted_by_optimizer"]]
    final_certificate = result["final_certificate"]
    provenance: Mapping[str, object] = (
        json.loads(provenance_out.read_text()) if provenance_out.is_file() else {}
    )
    enforce_child_conformance(
        lane=lane,
        config_label=config_label,
        omp_threads=omp_threads,
        affinity=affinity,
        provenance=provenance,
    )
    leg = LegResult(
        lane=lane,
        budget=budget,
        timed=timed,
        launcher_wall_seconds=launcher_wall,
        process_wall_seconds=float(payload["process_wall_seconds"]),
        optimizer_seconds=float(result["optimizer_seconds"]),
        compact_candidate_evaluations=int(
            result["work_counts"]["compact_candidate_evaluations"]
        ),
        accepted_callback_count=int(result["work_counts"]["accepted_callback_count"]),
        accepted_objectives=tuple(
            float(entry["returned_objective_value"]) for entry in accepted
        ),
        per_eval_seconds=tuple(
            float(entry["proposal_evaluation_seconds"]) for entry in trace
        ),
        certificate_attempt_count=int(
            result["work_counts"]["certificate_attempt_count"]
        ),
        certificate_seconds=float(
            result["phase_timings"]["cached_canonical_certificate_seconds"]
        ),
        endpoint_objective=float(final_certificate["objective_value"]),
        endpoint_gradient_inf_norm=max(
            abs(float(g)) for g in final_certificate["gradient"]["full_675"]
        ),
        endpoint_candidate=final_certificate["candidate"],
        endpoint_inner_state=tuple(
            float(v) for v in final_certificate["y_certificate"]["solution"]
        ),
        termination_reason=str(result["termination_reason"]),
        scipy_status=result["scipy"],
        lane_json_sha256=_sha256_bytes(lane_bytes),
        provenance=provenance,
        row_path=leg_root / "row.json",
    )
    row = {
        "schema": ROW_SCHEMA,
        "role": role,
        "config": config_label,
        "lane": leg.lane,
        "budget": leg.budget,
        "timed": leg.timed,
        "idle_wait_seconds": idle_wait,
        "launcher_wall_seconds": leg.launcher_wall_seconds,
        "process_wall_seconds": leg.process_wall_seconds,
        "optimizer_seconds": leg.optimizer_seconds,
        "compact_candidate_evaluations": leg.compact_candidate_evaluations,
        "accepted_callback_count": leg.accepted_callback_count,
        "accepted_objectives": list(leg.accepted_objectives),
        "per_eval_seconds": list(leg.per_eval_seconds),
        "certificate_attempt_count": leg.certificate_attempt_count,
        "certificate_seconds": leg.certificate_seconds,
        "endpoint_objective": leg.endpoint_objective,
        "endpoint_gradient_inf_norm": leg.endpoint_gradient_inf_norm,
        "endpoint_inner_state": list(leg.endpoint_inner_state),
        "termination_reason": leg.termination_reason,
        "scipy": dict(leg.scipy_status),
        "lane_json_sha256": leg.lane_json_sha256,
        "child_provenance": dict(provenance),
        "environment_echo": {
            name: environment.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OMP_PLACES",
                "OMP_PROC_BIND",
                "OPENBLAS_NUM_THREADS",
                "JAX_ENABLE_X64",
                "JAX_PLATFORMS",
            )
        },
        "requested_affinity": list(affinity) if affinity is not None else None,
        "policy_semantic_sha256": policy.semantic_sha256,
        "formulation_semantic_sha256": GENUINE_FULLSPACE_675.semantic_sha256,
        "campaign_input_manifest_sha256": campaign_manifest_sha256,
        "campaign_contract_sha256": contract_sha256(
            campaign_manifest_sha256=campaign_manifest_sha256,
            budget=budget,
        ),
        "git": dict(git_identity),
    }
    leg.row_path.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
    return leg


def run_oracle(
    *,
    candidate: Mapping[str, object],
    anchor: tuple[float, float],
    oracle_root: Path,
    source_manifest: Path,
) -> Mapping[str, object]:
    oracle_root.mkdir(parents=True, exist_ok=False)
    request_path = oracle_root / "request.json"
    output_path = oracle_root / "oracle.json"
    request_path.write_text(
        json.dumps(
            {
                "coil_coordinates": candidate["coil_coordinates"],
                "vessel_coordinates": candidate["vessel_coordinates"],
                "surface_coordinates": candidate["surface_coordinates"],
                "anchor_iota": anchor[0],
                "anchor_G": anchor[1],
            },
            sort_keys=True,
        )
        + "\n"
    )
    environment = repo_pythonpath_env(
        environment=os.environ,
        platform="cpu",
        backend_mode="native_cpu",
        backend_strict=True,
        mixed_precision_enabled=False,
    )
    completed = subprocess.run(
        (
            sys.executable,
            str(Path(__file__).resolve().parent / "genuine_675_fair_bar_oracle.py"),
            "--platform",
            "cpu",
            "--input-manifest",
            str(source_manifest),
            "--candidate-json",
            str(request_path),
            "--output-json",
            str(output_path),
            "--expected-formulation-sha256",
            GENUINE_FULLSPACE_675.semantic_sha256,
            "--source-root",
            str(SOURCE_ROOT),
        ),
        cwd=SOURCE_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Oracle failed: {completed.stderr[-2000:]}")
    return json.loads(output_path.read_text())


# --------------------------------------------------------------------------
# Gates (charter: "Work-matching and endpoint gates")
# --------------------------------------------------------------------------


def evaluate_pair_gates(
    native: LegResult,
    gpu: LegResult,
    oracle: Mapping[str, object],
) -> dict[str, object]:
    failures: list[str] = []
    if native.compact_candidate_evaluations != gpu.compact_candidate_evaluations:
        failures.append("eval_count_mismatch")
    if native.accepted_callback_count != gpu.accepted_callback_count:
        failures.append("accept_count_mismatch")
    sequence_ok = len(native.accepted_objectives) == len(gpu.accepted_objectives)
    if sequence_ok:
        for lhs, rhs in zip(native.accepted_objectives, gpu.accepted_objectives):
            if abs(lhs - rhs) > SEQUENCE_RTOL * max(abs(lhs), abs(rhs)):
                sequence_ok = False
                break
    if not sequence_ok:
        failures.append("accepted_objective_sequence_divergence")
    objective_gap = abs(native.endpoint_objective - gpu.endpoint_objective) / abs(
        native.endpoint_objective
    )
    if objective_gap > ENDPOINT_OBJECTIVE_RTOL:
        failures.append("endpoint_objective_gap")
    gradient_gap = abs(
        native.endpoint_gradient_inf_norm - gpu.endpoint_gradient_inf_norm
    ) / abs(native.endpoint_gradient_inf_norm)
    if gradient_gap > ENDPOINT_GRADIENT_RTOL:
        failures.append("endpoint_gradient_gap")
    oracle_gap = abs(
        float(oracle["objective_value"]) - native.endpoint_objective
    ) / abs(native.endpoint_objective)
    if oracle_gap > ENDPOINT_OBJECTIVE_RTOL:
        failures.append("oracle_objective_gap")
    for leg in (native, gpu):
        if leg.termination_reason != "scipy_completed":
            failures.append(f"{leg.lane}_termination_{leg.termination_reason}")
    return {
        "passed": not failures,
        "failures": failures,
        "endpoint_objective_relative_gap": objective_gap,
        "endpoint_gradient_relative_gap": gradient_gap,
        "oracle_objective_relative_gap": oracle_gap,
    }


def matched_prefix_length(native: Sequence[float], gpu: Sequence[float]) -> int:
    matched = 0
    for lhs, rhs in zip(native, gpu):
        if abs(lhs - rhs) > SEQUENCE_RTOL * max(abs(lhs), abs(rhs)):
            break
        matched += 1
    return matched


# --------------------------------------------------------------------------
# Phase drivers
# --------------------------------------------------------------------------


def _git_describe(root: Path) -> dict[str, object]:
    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ("git", "status", "--porcelain"),
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return {
        "commit": commit,
        "dirty_file_count": 0 if not dirty else len(dirty.splitlines()),
    }


def _git_identity() -> dict[str, object]:
    instrument = _git_describe(SOURCE_ROOT)
    if instrument["commit"] != INSTRUMENT_COMMIT:
        raise RuntimeError(
            f"Instrument worktree is at {instrument['commit']}, not the "
            f"frozen launch-source commit {INSTRUMENT_COMMIT}."
        )
    return {
        "instrument": instrument,
        "harness": _git_describe(Path(__file__).resolve().parent.parent),
    }


def _new_run_root(phase: str) -> Path:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    root = OUTPUT_ROOT / f"{stamp}-{phase}-{os.getpid()}"
    root.mkdir(parents=True, exist_ok=False)
    return root


def _finish_run(
    run_root: Path,
    *,
    phase: str,
    verdict: str,
    campaign_manifest_sha256: str,
    extra: Mapping[str, object],
) -> None:
    rows = {}
    for row_path in sorted(run_root.rglob("row.json")):
        rows[str(row_path.relative_to(run_root))] = _sha256_file(row_path)
    for name in ("campaign_input_manifest.json",):
        candidate = run_root / name
        if candidate.is_file():
            rows[name] = _sha256_file(candidate)
    manifest = {
        "schema": RUN_MANIFEST_SCHEMA,
        "phase": phase,
        "verdict": verdict,
        "charter_sha256": CHARTER_SHA256,
        "campaign_input_manifest_sha256": campaign_manifest_sha256,
        "formulation_semantic_sha256": GENUINE_FULLSPACE_675.semantic_sha256,
        "git": _git_identity(),
        "rows": rows,
        **dict(extra),
    }
    (run_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"phase": phase, "verdict": verdict, "run": str(run_root)}))


def _prepare_run(
    args: argparse.Namespace, phase: str
) -> tuple[Path, Path, str, Path, dict[str, object]]:
    run_root = _new_run_root(phase)
    campaign_manifest = run_root / "campaign_input_manifest.json"
    campaign_sha = mint_campaign_manifest(args.source_manifest, campaign_manifest)
    load_campaign_manifest(campaign_manifest, args.source_manifest.parent)
    shim_dir = write_provenance_shim(run_root)
    return run_root, campaign_manifest, campaign_sha, shim_dir, _git_identity()


def _native_leg(
    *,
    config: NativeConfig,
    budget: int,
    run_root: Path,
    shim_dir: Path,
    leg_name: str,
    timed: bool,
    role: str,
    campaign_sha: str,
    source_manifest: Path,
    git_identity: Mapping[str, object],
    primer: bool,
) -> LegResult:
    if primer:
        primer_root = run_root / f"{leg_name}-primer"
        run_leg(
            lane=NATIVE_LANE,
            budget=budget,
            environment=native_environment(config, shim_dir=shim_dir),
            omp_threads=config.omp_threads,
            affinity=config.affinity,
            leg_root=primer_root,
            timed=False,
            role=f"{role}-primer",
            config_label=config.label,
            campaign_manifest_sha256=campaign_sha,
            source_manifest=source_manifest,
            git_identity=git_identity,
        )
    return run_leg(
        lane=NATIVE_LANE,
        budget=budget,
        environment=native_environment(config, shim_dir=shim_dir),
        omp_threads=config.omp_threads,
        affinity=config.affinity,
        leg_root=run_root / leg_name,
        timed=timed,
        role=role,
        config_label=config.label,
        campaign_manifest_sha256=campaign_sha,
        source_manifest=source_manifest,
        git_identity=git_identity,
    )


def _gpu_leg(
    *,
    budget: int,
    run_root: Path,
    shim_dir: Path,
    cache_dir: Path,
    leg_name: str,
    timed: bool,
    role: str,
    campaign_sha: str,
    source_manifest: Path,
    git_identity: Mapping[str, object],
    primer: bool,
) -> LegResult:
    if primer:
        primer_root = run_root / f"{leg_name}-primer"
        run_leg(
            lane=GPU_LANE,
            budget=budget,
            environment=gpu_environment(cache_dir=cache_dir, shim_dir=shim_dir),
            omp_threads=None,
            affinity=None,
            leg_root=primer_root,
            timed=False,
            role=f"{role}-primer",
            config_label="gpu",
            campaign_manifest_sha256=campaign_sha,
            source_manifest=source_manifest,
            git_identity=git_identity,
        )
    return run_leg(
        lane=GPU_LANE,
        budget=budget,
        environment=gpu_environment(cache_dir=cache_dir, shim_dir=shim_dir),
        omp_threads=None,
        affinity=None,
        leg_root=run_root / leg_name,
        timed=timed,
        role=role,
        config_label="gpu",
        campaign_manifest_sha256=campaign_sha,
        source_manifest=source_manifest,
        git_identity=git_identity,
    )


def cmd_phase1(args: argparse.Namespace) -> None:
    run_root, _manifest, campaign_sha, shim_dir, git = _prepare_run(args, "phase1")
    config = NativeConfig.pinned(args.omp_threads)
    native_cold = _native_leg(
        config=config,
        budget=BUDGET_CONTINUITY,
        run_root=run_root,
        shim_dir=shim_dir,
        leg_name="native-cold",
        timed=False,
        role="phase1-native-cold",
        campaign_sha=campaign_sha,
        source_manifest=args.source_manifest,
        git_identity=git,
        primer=False,
    )
    native_primed = _native_leg(
        config=config,
        budget=BUDGET_CONTINUITY,
        run_root=run_root,
        shim_dir=shim_dir,
        leg_name="native-primed",
        timed=False,
        role="phase1-native-primed",
        campaign_sha=campaign_sha,
        source_manifest=args.source_manifest,
        git_identity=git,
        primer=True,
    )
    gpu = _gpu_leg(
        budget=BUDGET_CONTINUITY,
        run_root=run_root,
        shim_dir=shim_dir,
        cache_dir=run_root / "gpu-cache",
        leg_name="gpu",
        timed=False,
        role="phase1-gpu",
        campaign_sha=campaign_sha,
        source_manifest=args.source_manifest,
        git_identity=git,
        primer=True,
    )
    oracle = run_oracle(
        candidate=gpu.endpoint_candidate,
        anchor=gpu.endpoint_inner_state,
        oracle_root=run_root / "oracle",
        source_manifest=args.source_manifest,
    )
    gates = evaluate_pair_gates(native_primed, gpu, oracle)
    conformance_failures = []
    for leg, expected_x64 in ((native_primed, "1"), (gpu, "1")):
        child_env = leg.provenance.get("env", {}) if leg.provenance else {}
        if child_env.get("JAX_ENABLE_X64") != expected_x64:
            conformance_failures.append(f"{leg.lane}_x64_unobserved")
    child_env = native_primed.provenance.get("env", {})
    if child_env.get("OMP_NUM_THREADS") != str(args.omp_threads):
        conformance_failures.append("native_omp_pin_unobserved")
    primed_delta = native_cold.process_wall_seconds - native_primed.process_wall_seconds
    verdict = (
        "PHASE1_OK" if gates["passed"] and not conformance_failures else "NOT_PRODUCED"
    )
    _finish_run(
        run_root,
        phase="phase1",
        verdict=verdict,
        campaign_manifest_sha256=campaign_sha,
        extra={
            "gates": gates,
            "conformance_failures": conformance_failures,
            "native_primed_vs_cold_delta_seconds": primed_delta,
        },
    )


def cmd_probe(args: argparse.Namespace) -> None:
    run_root, _manifest, campaign_sha, shim_dir, git = _prepare_run(args, "probe")
    config = NativeConfig.pinned(args.omp_threads)
    native = _native_leg(
        config=config,
        budget=BUDGET_HEADLINE,
        run_root=run_root,
        shim_dir=shim_dir,
        leg_name="native",
        timed=False,
        role="probe-native",
        campaign_sha=campaign_sha,
        source_manifest=args.source_manifest,
        git_identity=git,
        primer=True,
    )
    gpu = _gpu_leg(
        budget=BUDGET_HEADLINE,
        run_root=run_root,
        shim_dir=shim_dir,
        cache_dir=run_root / "gpu-cache",
        leg_name="gpu",
        timed=False,
        role="probe-gpu",
        campaign_sha=campaign_sha,
        source_manifest=args.source_manifest,
        git_identity=git,
        primer=True,
    )
    matched = matched_prefix_length(native.accepted_objectives, gpu.accepted_objectives)
    full_match = (
        matched == len(native.accepted_objectives) == len(gpu.accepted_objectives)
        and native.compact_candidate_evaluations == gpu.compact_candidate_evaluations
    )
    _finish_run(
        run_root,
        phase="probe",
        verdict="PROBE_MATCHED" if full_match else "PROBE_FORKED",
        campaign_manifest_sha256=campaign_sha,
        extra={
            "matched_accepted_prefix": matched,
            "native_accepted": len(native.accepted_objectives),
            "gpu_accepted": len(gpu.accepted_objectives),
            "native_evals": native.compact_candidate_evaluations,
            "gpu_evals": gpu.compact_candidate_evaluations,
            "certificate_cadence": {
                "native": {
                    "attempts": native.certificate_attempt_count,
                    "seconds": native.certificate_seconds,
                },
                "gpu": {
                    "attempts": gpu.certificate_attempt_count,
                    "seconds": gpu.certificate_seconds,
                },
            },
        },
    )


def cmd_native_matrix(args: argparse.Namespace) -> None:
    run_root, _manifest, campaign_sha, shim_dir, git = _prepare_run(
        args, f"native-matrix-b{args.budget}"
    )
    configs = [NativeConfig.pinned(threads) for threads in args.omp_set]
    results: dict[str, dict[str, object]] = {}
    for config in configs:
        reps = []
        termination_reasons = []
        for rep in range(MATRIX_REPS):
            leg = _native_leg(
                config=config,
                budget=args.budget,
                run_root=run_root,
                shim_dir=shim_dir,
                leg_name=f"{config.label}-rep{rep}",
                timed=True,
                role="matrix",
                campaign_sha=campaign_sha,
                source_manifest=args.source_manifest,
                git_identity=git,
                primer=True,
            )
            reps.append(leg.process_wall_seconds)
            termination_reasons.append(leg.termination_reason)
        eligible_config = all(
            reason == "scipy_completed" for reason in termination_reasons
        )
        results[config.label] = {
            "reps_process_wall_seconds": reps,
            "median": statistics.median(reps),
            "dispersion_max_over_min": max(reps) / min(reps),
            "termination_reasons": termination_reasons,
            "eligible": eligible_config,
        }
    disclosure = _native_leg(
        config=NativeConfig.unpinned_default(),
        budget=args.budget,
        run_root=run_root,
        shim_dir=shim_dir,
        leg_name="unpinned-default",
        timed=True,
        role="disclosure",
        campaign_sha=campaign_sha,
        source_manifest=args.source_manifest,
        git_identity=git,
        primer=True,
    )
    eligible_results = {
        label: entry for label, entry in results.items() if entry["eligible"]
    }
    if not eligible_results:
        raise RuntimeError(
            "No native matrix configuration completed scipy in all reps."
        )
    best = min(eligible_results.items(), key=lambda item: float(item[1]["median"]))
    best_median = float(best[1]["median"])
    headline_eligible = sorted(
        label
        for label, entry in eligible_results.items()
        if float(entry["median"]) <= NARROWING_FACTOR * best_median
    )
    _finish_run(
        run_root,
        phase=f"native-matrix-b{args.budget}",
        verdict="NATIVE_SELECTED",
        campaign_manifest_sha256=campaign_sha,
        extra={
            "budget": args.budget,
            "matrix": results,
            "unpinned_default_process_wall_seconds": (disclosure.process_wall_seconds),
            "selected_config": best[0],
            "selected_median_process_wall_seconds": best_median,
            "headline_eligible_configs": headline_eligible,
            "narrowing_factor": NARROWING_FACTOR,
        },
    )


def cmd_pairs(args: argparse.Namespace) -> None:
    run_root, _manifest, campaign_sha, shim_dir, git = _prepare_run(
        args, f"pairs-b{args.budget}"
    )
    config = NativeConfig.pinned(args.omp_threads)
    ratios: list[float] = []
    not_produced = 0
    pair_reports: list[dict[str, object]] = []
    for pair_index in range(PAIR_COUNT):
        native_first = pair_index % 2 == 0
        cache_dir = run_root / f"gpu-cache-pair{pair_index}"

        def _native(pair_index: int = pair_index) -> LegResult:
            return _native_leg(
                config=config,
                budget=args.budget,
                run_root=run_root,
                shim_dir=shim_dir,
                leg_name=f"pair{pair_index}-native",
                timed=True,
                role="pair",
                campaign_sha=campaign_sha,
                source_manifest=args.source_manifest,
                git_identity=git,
                primer=True,
            )

        def _gpu(
            pair_index: int = pair_index, cache_dir: Path = cache_dir
        ) -> LegResult:
            return _gpu_leg(
                budget=args.budget,
                run_root=run_root,
                shim_dir=shim_dir,
                cache_dir=cache_dir,
                leg_name=f"pair{pair_index}-gpu",
                timed=True,
                role="pair",
                campaign_sha=campaign_sha,
                source_manifest=args.source_manifest,
                git_identity=git,
                primer=True,
            )

        try:
            if native_first:
                native = _native()
                gpu = _gpu()
            else:
                gpu = _gpu()
                native = _native()
            oracle = run_oracle(
                candidate=gpu.endpoint_candidate,
                anchor=gpu.endpoint_inner_state,
                oracle_root=run_root / f"pair{pair_index}-oracle",
                source_manifest=args.source_manifest,
            )
        except RuntimeError as error:
            not_produced += 1
            pair_reports.append(
                {
                    "pair": pair_index,
                    "order": "native-first" if native_first else "gpu-first",
                    "not_produced": True,
                    "failure": str(error),
                }
            )
            if not_produced >= NOT_PRODUCED_ABORT:
                _finish_run(
                    run_root,
                    phase=f"pairs-b{args.budget}",
                    verdict="NOT_PRODUCED",
                    campaign_manifest_sha256=campaign_sha,
                    extra={"budget": args.budget, "pairs": pair_reports},
                )
                return
            continue
        gates = evaluate_pair_gates(native, gpu, oracle)
        report: dict[str, object] = {
            "pair": pair_index,
            "order": "native-first" if native_first else "gpu-first",
            "gates": gates,
        }
        if gates["passed"]:
            ratio = native.process_wall_seconds / gpu.process_wall_seconds
            ratios.append(ratio)
            report["ratio_process_wall"] = ratio
            report["native_process_wall_seconds"] = native.process_wall_seconds
            report["gpu_process_wall_seconds"] = gpu.process_wall_seconds
            report["ratio_optimizer_wall"] = (
                native.optimizer_seconds / gpu.optimizer_seconds
            )
        else:
            not_produced += 1
            report["not_produced"] = True
        pair_reports.append(report)
        if not_produced >= NOT_PRODUCED_ABORT:
            _finish_run(
                run_root,
                phase=f"pairs-b{args.budget}",
                verdict="NOT_PRODUCED",
                campaign_manifest_sha256=campaign_sha,
                extra={"budget": args.budget, "pairs": pair_reports},
            )
            return
    if len(ratios) == PAIR_COUNT:
        median_ratio = statistics.median(ratios)
        if median_ratio >= WIN_MEDIAN_THRESHOLD and all(r > 1.0 for r in ratios):
            verdict = "WIN"
        else:
            verdict = "CLOSED_BOUNDED_NEGATIVE"
    else:
        verdict = "NOT_PRODUCED"
    _finish_run(
        run_root,
        phase=f"pairs-b{args.budget}",
        verdict=verdict,
        campaign_manifest_sha256=campaign_sha,
        extra={
            "budget": args.budget,
            "native_config": config.label,
            "pairs": pair_reports,
            "ratios_process_wall": ratios,
            "ratio_min": min(ratios) if ratios else None,
            "ratio_median": statistics.median(ratios) if ratios else None,
            "ratio_max": max(ratios) if ratios else None,
        },
    )


def cmd_fresh_pair(args: argparse.Namespace) -> None:
    """One both-cold disclosure pair per budget (no primers, cold caches)."""
    run_root, _manifest, campaign_sha, shim_dir, git = _prepare_run(
        args, f"fresh-pair-b{args.budget}"
    )
    config = NativeConfig.pinned(args.omp_threads)
    native = _native_leg(
        config=config,
        budget=args.budget,
        run_root=run_root,
        shim_dir=shim_dir,
        leg_name="native",
        timed=True,
        role="fresh",
        campaign_sha=campaign_sha,
        source_manifest=args.source_manifest,
        git_identity=git,
        primer=False,
    )
    gpu = _gpu_leg(
        budget=args.budget,
        run_root=run_root,
        shim_dir=shim_dir,
        cache_dir=run_root / "gpu-cache-fresh",
        leg_name="gpu",
        timed=True,
        role="fresh",
        campaign_sha=campaign_sha,
        source_manifest=args.source_manifest,
        git_identity=git,
        primer=False,
    )
    ratio = native.process_wall_seconds / gpu.process_wall_seconds
    _finish_run(
        run_root,
        phase=f"fresh-pair-b{args.budget}",
        verdict="FRESH_REPORTED",
        campaign_manifest_sha256=campaign_sha,
        extra={
            "budget": args.budget,
            "ratio_process_wall": ratio,
            "native_process_wall_seconds": native.process_wall_seconds,
            "gpu_process_wall_seconds": gpu.process_wall_seconds,
        },
    )


def cmd_mint_manifest(args: argparse.Namespace) -> None:
    output = args.output
    sha = mint_campaign_manifest(args.source_manifest, output)
    load_campaign_manifest(output, args.source_manifest.parent)
    print(json.dumps({"campaign_manifest": str(output), "sha256": sha}))


def cmd_selftest_loader(args: argparse.Namespace) -> None:
    run_root = _new_run_root("selftest-loader")
    report = selftest_loader(args.source_manifest, run_root / "scratch")
    (run_root / "selftest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report))
    if not report["passed"]:
        raise SystemExit(1)


def cmd_validate(args: argparse.Namespace) -> None:
    run_root = args.run_dir
    manifest = json.loads((run_root / "manifest.json").read_text())
    if manifest.get("schema") != RUN_MANIFEST_SCHEMA:
        raise ValueError("Not a fair-bar run manifest.")
    if manifest.get("charter_sha256") != CHARTER_SHA256:
        raise ValueError("Run manifest binds a foreign charter.")
    campaign_sha = manifest["campaign_input_manifest_sha256"]
    problems: list[str] = []
    recorded_manifest_sha = manifest["rows"].get("campaign_input_manifest.json")
    if recorded_manifest_sha != campaign_sha:
        problems.append("campaign_manifest_sha_link_broken")
    for rel, expected_sha in manifest["rows"].items():
        path = run_root / rel
        if not path.is_file():
            problems.append(f"missing:{rel}")
            continue
        if _sha256_file(path) != expected_sha:
            problems.append(f"sha_mismatch:{rel}")
            continue
        if rel.endswith("row.json"):
            row = json.loads(path.read_text())
            expected_contract = contract_sha256(
                campaign_manifest_sha256=campaign_sha,
                budget=int(row["budget"]),
            )
            if row.get("campaign_contract_sha256") != expected_contract:
                problems.append(f"contract_mismatch:{rel}")
    result = {
        "run": str(run_root),
        "phase": manifest.get("phase"),
        "stored_verdict": manifest.get("verdict"),
        "row_count": len(manifest["rows"]),
        "problems": problems,
        "validation": "OK" if not problems else "FAILED",
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if problems:
        raise SystemExit(1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path(
            "/home/jungdaesuh/simsopt_mixed_artifacts/"
            "genuine675-r3-input-1c23f6c5-20260721-r1/manifest.json"
        ),
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    sub = parser.add_subparsers(dest="command", required=True)

    mint = sub.add_parser("mint-manifest")
    mint.add_argument("--output", type=Path, required=True)
    mint.set_defaults(func=cmd_mint_manifest)

    selftest = sub.add_parser("selftest-loader")
    selftest.set_defaults(func=cmd_selftest_loader)

    phase1 = sub.add_parser("phase1")
    phase1.add_argument("--omp-threads", type=int, default=16)
    phase1.set_defaults(func=cmd_phase1)

    probe = sub.add_parser("probe")
    probe.add_argument("--omp-threads", type=int, default=16)
    probe.set_defaults(func=cmd_probe)

    matrix = sub.add_parser("native-matrix")
    matrix.add_argument("--budget", type=int, required=True)
    matrix.add_argument(
        "--omp-set",
        type=lambda text: tuple(int(v) for v in text.split(",")),
        default=OMP_MATRIX,
    )
    matrix.set_defaults(func=cmd_native_matrix)

    pairs = sub.add_parser("pairs")
    pairs.add_argument("--budget", type=int, required=True)
    pairs.add_argument("--omp-threads", type=int, required=True)
    pairs.set_defaults(func=cmd_pairs)

    fresh = sub.add_parser("fresh-pair")
    fresh.add_argument("--budget", type=int, required=True)
    fresh.add_argument("--omp-threads", type=int, required=True)
    fresh.set_defaults(func=cmd_fresh_pair)

    validate = sub.add_parser("validate")
    validate.add_argument("run_dir", type=Path)
    validate.set_defaults(func=cmd_validate)

    args = parser.parse_args()
    global SOURCE_ROOT, OUTPUT_ROOT
    SOURCE_ROOT = args.source_root.resolve()
    OUTPUT_ROOT = args.output_root.resolve()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    import simsopt_jax.runtime.single_stage_fullspace_675 as _formulation_module

    formulation_file = Path(_formulation_module.__file__).resolve()
    if not formulation_file.is_relative_to(SOURCE_ROOT):
        raise RuntimeError(
            "simsopt_jax resolved outside the instrument tree: "
            f"{formulation_file} (expected under {SOURCE_ROOT}); launch with "
            "PYTHONPATH=<source-root>:<source-root>/src."
        )
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
