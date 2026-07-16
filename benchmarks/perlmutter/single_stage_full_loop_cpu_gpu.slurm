#!/bin/bash -l
#SBATCH -A m4680_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 36:00:00
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --gpus-per-node=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --hint=nomultithread
#SBATCH --job-name=banana-full-loop
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

module load python/3.13-26.1.0
unset PYTHONPATH
export PYTHONNOUSERSITE=1

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR}}"
PROFILE="${PROFILE:-pilot}"
SEED_DIR="${SEED_DIR:?Set SEED_DIR to the immutable banana seed directory}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${SCRATCH}/simsopt-full-loop/${SLURM_JOB_ID}}"
IOTA_TARGET="${IOTA_TARGET:?Set IOTA_TARGET explicitly from the seed provenance}"
HASH_LOCK_PATH="${HASH_LOCK_PATH:-}"
VENV_ROOT="${VENV_ROOT:-${TMPDIR:-/tmp}/simsopt-full-loop-envs-${UID}}"
PAIR_EXECUTION_MODE="${PAIR_EXECUTION_MODE:-slurm-step}"

validate_external_write_root() {
    local root_label="$1"
    local repo_root_real
    local write_root_real
    repo_root_real="$(realpath -- "$2")"
    write_root_real="$(realpath -m -- "$3")"
    case "${write_root_real}/" in
        "${repo_root_real}/"*)
            echo "${root_label} must be outside REPO_ROOT" >&2
            echo "REPO_ROOT=${repo_root_real}" >&2
            echo "${root_label}=${write_root_real}" >&2
            return 2
            ;;
    esac
}

validate_disjoint_write_roots() {
    local first_label="$1"
    local first_root_real
    local second_label="$3"
    local second_root_real
    first_root_real="$(realpath -m -- "$2")"
    second_root_real="$(realpath -m -- "$4")"
    case "${first_root_real}/" in
        "${second_root_real}/"*)
            echo "${first_label} and ${second_label} must not overlap" >&2
            echo "${first_label}=${first_root_real}" >&2
            echo "${second_label}=${second_root_real}" >&2
            return 2
            ;;
    esac
    case "${second_root_real}/" in
        "${first_root_real}/"*)
            echo "${first_label} and ${second_label} must not overlap" >&2
            echo "${first_label}=${first_root_real}" >&2
            echo "${second_label}=${second_root_real}" >&2
            return 2
            ;;
    esac
}

reserve_fresh_directory() {
    local root_label="$1"
    local root_path="$2"
    mkdir -p -- "$(dirname -- "${root_path}")"
    if ! mkdir -- "${root_path}"; then
        echo "${root_label} must not exist before launch: ${root_path}" >&2
        return 2
    fi
}

prepare_private_venv_root() {
    local root_path="$1"
    local owner_uid
    local permissions
    if [[ -L "${root_path}" ]]; then
        echo "VENV_ROOT must not be a symbolic link: ${root_path}" >&2
        return 2
    fi
    if [[ ! -e "${root_path}" ]]; then
        if ! mkdir --mode=700 -- "${root_path}"; then
            echo "VENV_ROOT could not be reserved privately: ${root_path}" >&2
            return 2
        fi
    fi
    if [[ ! -d "${root_path}" ]]; then
        echo "VENV_ROOT must be a directory: ${root_path}" >&2
        return 2
    fi
    owner_uid="$(stat -c %u -- "${root_path}")"
    permissions="$(stat -c %a -- "${root_path}")"
    if [[ "${owner_uid}" != "${UID}" || "${permissions}" != "700" ]]; then
        echo "VENV_ROOT must be owned by the current user with mode 700" >&2
        echo "VENV_ROOT=${root_path}" >&2
        echo "owner_uid=${owner_uid} permissions=${permissions}" >&2
        return 2
    fi
}

validate_external_write_root "ARTIFACT_ROOT" "${REPO_ROOT}" "${ARTIFACT_ROOT}"
validate_external_write_root "VENV_ROOT" "${REPO_ROOT}" "${VENV_ROOT}"
REPO_ROOT="$(realpath -- "${REPO_ROOT}")"
ARTIFACT_ROOT="$(realpath -m -- "${ARTIFACT_ROOT}")"
VENV_ROOT_REAL="$(realpath -m -- "${VENV_ROOT}")"
VENV_DIR="${VENV_ROOT_REAL}/job-${SLURM_JOB_ID}-py313"
validate_disjoint_write_roots \
    "ARTIFACT_ROOT" "${ARTIFACT_ROOT}" "VENV_DIR" "${VENV_DIR}"

SURFACE_PATH="${SURFACE_PATH:-${SEED_DIR}/surf_opt.json}"
BIOTSAVART_FILE="${BIOTSAVART_FILE:-${SEED_DIR}/biot_savart_opt.json}"
BOOZER_STATE_PATH="${BOOZER_STATE_PATH:-${SEED_DIR}/surf_opt_boozer_state.json}"

case "${PROFILE}" in
    pilot)
        EVIDENCE_CLASS="diagnostic"
        PAIR_COUNT="${PAIR_COUNT:-1}"
        MPOL="${MPOL:-10}"
        NTOR="${NTOR:-10}"
        NPHI="${NPHI:-65}"
        NTHETA="${NTHETA:-32}"
        OUTER_MAXITER="${OUTER_MAXITER:-3}"
        OUTER_FTOL="${OUTER_FTOL:-1e-9}"
        OUTER_GTOL="${OUTER_GTOL:-1e-6}"
        BOOZER_BFGS_MAXITER="${BOOZER_BFGS_MAXITER:-300}"
        BOOZER_NEWTON_MAXITER="${BOOZER_NEWTON_MAXITER:-20}"
        ;;
    full)
        EVIDENCE_CLASS="production"
        if [[ -z "${HASH_LOCK_PATH}" ]]; then
            echo \
                "Full production evidence requires HASH_LOCK_PATH from a successful pilot artifact" \
                >&2
            exit 2
        fi
        PAIR_COUNT="${PAIR_COUNT:-4}"
        MPOL="${MPOL:-10}"
        NTOR="${NTOR:-10}"
        NPHI="${NPHI:-255}"
        NTHETA="${NTHETA:-64}"
        OUTER_MAXITER="${OUTER_MAXITER:-150}"
        OUTER_FTOL="${OUTER_FTOL:-1e-12}"
        OUTER_GTOL="${OUTER_GTOL:-1e-12}"
        BOOZER_BFGS_MAXITER="${BOOZER_BFGS_MAXITER:-1500}"
        BOOZER_NEWTON_MAXITER="${BOOZER_NEWTON_MAXITER:-40}"
        ;;
    *)
        echo "PROFILE must be pilot or full; got ${PROFILE}" >&2
        exit 2
        ;;
esac

validate_pair_count() {
    local profile="$1"
    local pair_count="$2"
    if [[ ! "${pair_count}" =~ ^[1-9][0-9]*$ ]]; then
        echo "PAIR_COUNT must be a positive integer; got ${pair_count}" >&2
        return 2
    fi
    if [[ "${profile}" == "full" ]] \
        && ((pair_count < 2 || pair_count % 2 != 0)); then
        echo "Full production evidence requires an even PAIR_COUNT >= 2; got ${pair_count}" >&2
        return 2
    fi
}

validate_pair_execution_mode() {
    local profile="$1"
    local execution_mode="$2"
    local visible_devices
    case "${execution_mode}" in
        slurm-step)
            ;;
        direct)
            if [[ "${profile}" != "pilot" ]]; then
                echo "Direct pair execution is diagnostic-only and requires PROFILE=pilot" >&2
                return 2
            fi
            if [[ -z "${SLURM_STEP_ID:-}" ]]; then
                echo "Direct pair execution requires the launcher to run inside a Slurm step" >&2
                return 2
            fi
            visible_devices="${CUDA_VISIBLE_DEVICES:-}"
            if [[ -z "${visible_devices}" || "${visible_devices}" == *,* \
                || "${visible_devices}" =~ [[:space:]] ]]; then
                echo \
                    "Direct pair execution requires exactly one CUDA_VISIBLE_DEVICES selector" \
                    >&2
                return 2
            fi
            ;;
        *)
            echo \
                "PAIR_EXECUTION_MODE must be slurm-step or direct; got ${execution_mode}" \
                >&2
            return 2
            ;;
    esac
}

configure_pair_launcher() {
    local execution_mode="$1"
    PAIR_LAUNCHER=()
    if [[ "${execution_mode}" == "slurm-step" ]]; then
        PAIR_LAUNCHER=(
            srun
            --ntasks=1
            --cpus-per-task="${SLURM_CPUS_PER_TASK}"
            --gpus-per-task=1
            --cpu-bind=cores
        )
    fi
}

validate_pair_count "${PROFILE}" "${PAIR_COUNT}"
validate_pair_execution_mode "${PROFILE}" "${PAIR_EXECUTION_MODE}"

CONSTRAINT_WEIGHT="${CONSTRAINT_WEIGHT:-1.0}"
OUTER_MAXCOR="${OUTER_MAXCOR:-300}"
OUTER_MAXLS="${OUTER_MAXLS:-20}"
BOOZER_BFGS_TOL="${BOOZER_BFGS_TOL:-1e-10}"
BOOZER_NEWTON_TOL="${BOOZER_NEWTON_TOL:-1e-11}"

prepare_private_venv_root "${VENV_ROOT}"
VENV_ROOT="$(realpath -- "${VENV_ROOT}")"
if [[ "${VENV_ROOT}" != "${VENV_ROOT_REAL}" ]]; then
    echo "VENV_ROOT changed while it was being reserved" >&2
    exit 2
fi
VENV_DIR="${VENV_ROOT}/job-${SLURM_JOB_ID}-py313"
reserve_fresh_directory "ARTIFACT_ROOT" "${ARTIFACT_ROOT}"
reserve_fresh_directory "VENV_DIR" "${VENV_DIR}"

MACHINE_ROOT="${ARTIFACT_ROOT}/machine"
INPUT_ROOT="${ARTIFACT_ROOT}/inputs"
PAIR_ROOT="${ARTIFACT_ROOT}/pairs"
ENVIRONMENT_ROOT="${ARTIFACT_ROOT}/environment"
mkdir -p "${MACHINE_ROOT}" "${INPUT_ROOT}" "${PAIR_ROOT}" \
    "${ENVIRONMENT_ROOT}"

GPU_MONITOR_PID=""
GPU_MONITOR_SENTINEL="${MACHINE_ROOT}/gpu_monitor.running"
GPU_PROCESS_MEMORY_CSV="${MACHINE_ROOT}/gpu_process_memory.csv"
GPU_SAMPLER_QUERY_CSV="${MACHINE_ROOT}/gpu_sampler_queries.csv"
GPU_SAMPLER_ERROR_LOG="${MACHINE_ROOT}/gpu_sampler_errors.log"
GPU_PROCESS_MEMORY_SUMMARY="${MACHINE_ROOT}/gpu_process_memory_summary.json"
CURRENT_PAIR_PATH="${MACHINE_ROOT}/current_pair.txt"
ENVIRONMENT_LOCK_MANIFEST="${ENVIRONMENT_ROOT}/manifest.json"
ENVIRONMENT_LOCK_SNAPSHOT="${ENVIRONMENT_ROOT}/requirements.lock"

summarize_gpu_process_memory() {
    local python_executable="python"
    if [[ -x "${VENV_DIR}/bin/python" ]]; then
        python_executable="${VENV_DIR}/bin/python"
    fi
    GPU_PROCESS_MEMORY_CSV="${GPU_PROCESS_MEMORY_CSV}" \
        GPU_SAMPLER_QUERY_CSV="${GPU_SAMPLER_QUERY_CSV}" \
        GPU_SAMPLER_ERROR_LOG="${GPU_SAMPLER_ERROR_LOG}" \
        GPU_PROCESS_MEMORY_SUMMARY="${GPU_PROCESS_MEMORY_SUMMARY}" \
        PAIR_ROOT="${PAIR_ROOT}" \
        PROFILE="${PROFILE}" \
        PAIR_COUNT="${PAIR_COUNT}" \
        "${python_executable}" - <<'PY'
from __future__ import annotations

import csv
from datetime import datetime
import json
import os
from pathlib import Path


process_csv_path = Path(os.environ["GPU_PROCESS_MEMORY_CSV"])
query_csv_path = Path(os.environ["GPU_SAMPLER_QUERY_CSV"])
error_log_path = Path(os.environ["GPU_SAMPLER_ERROR_LOG"])
summary_path = Path(os.environ["GPU_PROCESS_MEMORY_SUMMARY"])
pair_root = Path(os.environ["PAIR_ROOT"])
profile = os.environ["PROFILE"]
expected_pair_count = int(os.environ["PAIR_COUNT"])


def parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


expected_pair_names = tuple(
    f"pair-{pair_index:02d}"
    for pair_index in range(1, expected_pair_count + 1)
)
by_pair: dict[str, dict[str, object]] = {
    pair_name: {
        "query_count": 0,
        "query_failure_count": 0,
        "process_sample_count": 0,
        "process_ids": set(),
        "values": [],
        "jax_process_sample_count": 0,
        "jax_process_ids": set(),
        "jax_values": [],
    }
    for pair_name in expected_pair_names
}
query_count = 0
query_failure_count = 0
if query_csv_path.is_file():
    with query_csv_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            query_count += 1
            pair = row["pair"].strip() or "unassigned"
            failed = row["status"].strip() != "ok"
            query_failure_count += int(failed)
            if pair in by_pair:
                pair_summary = by_pair[pair]
                pair_summary["query_count"] += 1
                pair_summary["query_failure_count"] += int(failed)

jax_windows: dict[str, tuple[datetime, datetime]] = {}
for pair_name in expected_pair_names:
    execution_path = pair_root / pair_name / "jax" / "execution.json"
    if not execution_path.is_file():
        continue
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    jax_windows[pair_name] = (
        parse_timestamp(str(execution["started_at_utc"])),
        parse_timestamp(str(execution["ended_at_utc"])),
    )

overall_values: list[int] = []
overall_pids: set[int] = set()

if process_csv_path.is_file():
    with process_csv_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            pair = row["pair"].strip() or "unassigned"
            if pair not in by_pair:
                continue
            pid = int(row["pid"])
            used_memory_mib = int(row["used_memory_mib"])
            sampled_at = parse_timestamp(row["timestamp_utc"])
            pair_summary = by_pair[pair]
            pair_summary["process_sample_count"] += 1
            pair_summary["process_ids"].add(pid)
            pair_summary["values"].append(used_memory_mib)
            if pair in jax_windows:
                started_at, ended_at = jax_windows[pair]
                if started_at <= sampled_at <= ended_at:
                    pair_summary["jax_process_sample_count"] += 1
                    pair_summary["jax_process_ids"].add(pid)
                    pair_summary["jax_values"].append(used_memory_mib)
            overall_pids.add(pid)
            overall_values.append(used_memory_mib)

serialized_by_pair = {
    pair: {
        "query_count": values["query_count"],
        "query_failure_count": values["query_failure_count"],
        "process_sample_count": values["process_sample_count"],
        "process_ids": sorted(values["process_ids"]),
        "maximum_used_memory_mib": (
            max(values["values"]) if values["values"] else None
        ),
        "jax_execution_window_recorded": pair in jax_windows,
        "jax_process_sample_count": values["jax_process_sample_count"],
        "jax_process_ids": sorted(values["jax_process_ids"]),
        "maximum_jax_used_memory_mib": (
            max(values["jax_values"]) if values["jax_values"] else None
        ),
    }
    for pair, values in sorted(by_pair.items())
}
all_queries_succeeded = query_count > 0 and query_failure_count == 0
every_pair_has_jax_process_sample = all(
    int(serialized_by_pair[pair_name]["jax_process_sample_count"]) > 0
    for pair_name in expected_pair_names
)
production_gate_passed = (
    all_queries_succeeded and every_pair_has_jax_process_sample
)
payload = {
    "schema_version": 1,
    "metric": "nvidia-smi compute-process used_memory",
    "unit": "MiB",
    "sampling_interval_seconds": 1,
    "scope": (
        "all compute processes on the exclusive Slurm node"
        if profile == "full"
        else "diagnostic samples from all compute processes visible on the "
        "allocated node"
    ),
    "note": (
        "JAX samples are compute-process rows whose timestamps fall inside "
        "the recorded JAX lane execution window. GPU process memory is "
        "separate from GNU-time host RSS."
    ),
    "sampler_queries": {
        "query_count": query_count,
        "successful_query_count": query_count - query_failure_count,
        "failure_count": query_failure_count,
        "query_log": str(query_csv_path),
        "failure_log": str(error_log_path),
    },
    "overall": {
        "sample_count": len(overall_values),
        "process_ids": sorted(overall_pids),
        "maximum_used_memory_mib": (
            max(overall_values) if overall_values else None
        ),
    },
    "by_pair": serialized_by_pair,
    "evidence_gate": {
        "profile": profile,
        "expected_pair_count": expected_pair_count,
        "all_queries_succeeded": all_queries_succeeded,
        "every_pair_has_jax_process_sample": every_pair_has_jax_process_sample,
        "passed": production_gate_passed,
        "required_for_claim": profile == "full",
    },
}
temporary_path = summary_path.with_suffix(summary_path.suffix + ".tmp")
temporary_path.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
temporary_path.replace(summary_path)
PY
}

aggregate_pairs() {
    local python_executable="python"
    if [[ -x "${VENV_DIR}/bin/python" ]]; then
        python_executable="${VENV_DIR}/bin/python"
    fi
    ARTIFACT_ROOT="${ARTIFACT_ROOT}" \
        PROFILE="${PROFILE}" \
        EVIDENCE_CLASS="${EVIDENCE_CLASS}" \
        PAIR_EXECUTION_MODE="${PAIR_EXECUTION_MODE}" \
        PAIR_COUNT="${PAIR_COUNT}" \
        GPU_PROCESS_MEMORY_SUMMARY="${GPU_PROCESS_MEMORY_SUMMARY}" \
        ENVIRONMENT_LOCK_MANIFEST="${ENVIRONMENT_LOCK_MANIFEST}" \
        "${python_executable}" - <<'PY'
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import statistics


artifact_root = Path(os.environ["ARTIFACT_ROOT"])
pair_root = artifact_root / "pairs"
profile = os.environ["PROFILE"]
evidence_class = os.environ["EVIDENCE_CLASS"]
pair_execution_mode = os.environ["PAIR_EXECUTION_MODE"]
expected_pair_count = int(os.environ["PAIR_COUNT"])
gpu_memory_summary_path = Path(os.environ["GPU_PROCESS_MEMORY_SUMMARY"])
environment_lock_manifest_path = Path(os.environ["ENVIRONMENT_LOCK_MANIFEST"])
gpu_memory_summary = (
    json.loads(gpu_memory_summary_path.read_text(encoding="utf-8"))
    if gpu_memory_summary_path.is_file()
    else None
)
environment_lock_manifest = (
    json.loads(environment_lock_manifest_path.read_text(encoding="utf-8"))
    if environment_lock_manifest_path.is_file()
    else None
)
environment_lock_snapshot_path = artifact_root / "environment" / "requirements.lock"


def sha256_file(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


recorded_lock_sha256 = (
    str(environment_lock_manifest.get("lock_sha256"))
    if environment_lock_manifest is not None
    and environment_lock_manifest.get("lock_sha256")
    else None
)
lock_snapshot_sha256 = (
    sha256_file(environment_lock_snapshot_path)
    if environment_lock_snapshot_path.is_file()
    else None
)
dependency_lock_digest_matches = bool(
    recorded_lock_sha256 is not None
    and recorded_lock_sha256 == lock_snapshot_sha256
)
gpu_memory_gate = (
    gpu_memory_summary["evidence_gate"]
    if gpu_memory_summary is not None
    else {
        "profile": profile,
        "expected_pair_count": expected_pair_count,
        "all_queries_succeeded": False,
        "every_pair_has_jax_process_sample": False,
        "passed": False,
        "required_for_claim": profile == "full",
    }
)
dependency_lock_gate_passed = bool(
    environment_lock_manifest is not None
    and dependency_lock_digest_matches
    and (
        profile != "full"
        or environment_lock_manifest.get("secure_install_verified") is True
    )
)
pairs: list[dict[str, object]] = []
successful_speedups: list[float] = []
successful_cpu_wall_seconds: list[float] = []
successful_jax_wall_seconds: list[float] = []
successful_cpu_host_rss_kib: list[int] = []
successful_jax_host_rss_kib: list[int] = []

for pair_index in range(1, expected_pair_count + 1):
    pair_name = f"pair-{pair_index:02d}"
    pair_dir = pair_root / pair_name
    status_path = pair_dir / "pair_status.tsv"
    status_values: dict[str, str] = {}
    if status_path.is_file():
        for line in status_path.read_text(encoding="utf-8").splitlines():
            key, value = line.split("\t", 1)
            status_values[key] = value

    comparison_path = pair_dir / "comparison.json"
    comparison: dict[str, object] | None = None
    if comparison_path.is_file():
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))

    status = status_values.get("adjudicated_status", "not-run")
    order = status_values.get(
        "order",
        "cpu-jax" if pair_index % 2 == 1 else "jax-cpu",
    )
    pair_payload: dict[str, object] = {
        "pair_index": pair_index,
        "pair_name": pair_name,
        "order": order,
        "pair_execution_mode": status_values.get("pair_execution_mode"),
        "status": status,
        "comparator_exit_code": (
            int(status_values["comparator_exit_code"])
            if "comparator_exit_code" in status_values
            else None
        ),
        "comparison_status": status_values.get("comparison_status"),
        "comparison_claim_ready": (
            status_values.get("comparison_claim_ready") == "true"
            if "comparison_claim_ready" in status_values
            else None
        ),
        "cpu_optimizer_success": (
            status_values.get("cpu_optimizer_success") == "true"
            if "cpu_optimizer_success" in status_values
            else None
        ),
        "jax_optimizer_success": (
            status_values.get("jax_optimizer_success") == "true"
            if "jax_optimizer_success" in status_values
            else None
        ),
        "started_at_utc": status_values.get("started_at_utc"),
        "finished_at_utc": status_values.get("finished_at_utc"),
        "comparison_path": (
            str(comparison_path.relative_to(artifact_root))
            if comparison_path.is_file()
            else None
        ),
    }
    if gpu_memory_summary is not None:
        pair_payload["gpu_process_memory"] = gpu_memory_summary["by_pair"].get(
            pair_name,
            {
                "query_count": 0,
                "query_failure_count": 0,
                "process_sample_count": 0,
                "process_ids": [],
                "maximum_used_memory_mib": None,
                "jax_execution_window_recorded": False,
                "jax_process_sample_count": 0,
                "jax_process_ids": [],
                "maximum_jax_used_memory_mib": None,
            },
        )
    if comparison is not None:
        performance = comparison["performance"]
        parity = comparison["parity"]
        pair_payload["performance"] = performance
        pair_payload["required_failures"] = parity["required_failures"]
        pair_payload["advisory_failures"] = parity["advisory_failures"]
        pair_payload["host_rss_kib"] = {
            "cpu": performance["cpu_host_max_rss_kib"],
            "jax": performance["jax_host_max_rss_kib"],
        }
        if status == "passed":
            successful_speedups.append(float(performance["cpu_over_jax_speedup"]))
            successful_cpu_wall_seconds.append(
                float(performance["cpu_wall_seconds"])
            )
            successful_jax_wall_seconds.append(
                float(performance["jax_wall_seconds"])
            )
            successful_cpu_host_rss_kib.append(
                int(performance["cpu_host_max_rss_kib"])
            )
            successful_jax_host_rss_kib.append(
                int(performance["jax_host_max_rss_kib"])
            )
    pairs.append(pair_payload)

successful_pair_count = len(successful_speedups)
repeat_design_valid = (
    expected_pair_count >= 2 and expected_pair_count % 2 == 0
    if profile == "full"
    else expected_pair_count >= 1
)
pair_execution_modes_match = all(
    pair["pair_execution_mode"] == pair_execution_mode for pair in pairs
)
all_pairs_passed = (
    repeat_design_valid
    and pair_execution_modes_match
    and successful_pair_count == expected_pair_count
)
production_evidence_gates_passed = bool(
    gpu_memory_gate["passed"] and dependency_lock_gate_passed
)
aggregate_claim_ready = bool(
    profile == "full" and all_pairs_passed and production_evidence_gates_passed
)
speedup_summary = {
    "metric": "native_simsopt_cpu_wall_seconds / jax_cuda_wall_seconds",
    "successful_pair_count": successful_pair_count,
    "median": (
        statistics.median(successful_speedups) if successful_speedups else None
    ),
    "minimum": min(successful_speedups) if successful_speedups else None,
    "maximum": max(successful_speedups) if successful_speedups else None,
}


def distribution(values: list[float] | list[int]) -> dict[str, float | int | None]:
    return {
        "median": statistics.median(values) if values else None,
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
    }


payload = {
    "schema_version": 1,
    "status": (
        "passed"
        if aggregate_claim_ready
        else "diagnostic"
        if profile == "pilot" and all_pairs_passed
        else "failed"
    ),
    "profile": profile,
    "evidence_class": evidence_class,
    "pair_execution_mode": pair_execution_mode,
    "claim_ready": aggregate_claim_ready,
    "expected_pair_count": expected_pair_count,
    "successful_pair_count": successful_pair_count,
    "failed_pair_count": expected_pair_count - successful_pair_count,
    "repeat_design_valid": repeat_design_valid,
    "pair_execution_modes_match": pair_execution_modes_match,
    "lane_order_policy": "alternating cpu-jax / jax-cpu",
    "speedup": speedup_summary,
    "wall_seconds": {
        "cpu": distribution(successful_cpu_wall_seconds),
        "jax": distribution(successful_jax_wall_seconds),
    },
    "host_max_rss_kib": {
        "cpu": distribution(successful_cpu_host_rss_kib),
        "jax": distribution(successful_jax_host_rss_kib),
    },
    "pairs": pairs,
    "input_manifest": "inputs/manifest.json",
    "dependency_lock": {
        "manifest": "environment/manifest.json",
        "lock_sha256": (
            recorded_lock_sha256
        ),
        "snapshot_sha256": lock_snapshot_sha256,
        "digest_matches_snapshot": dependency_lock_digest_matches,
        "secure_install_verified": (
            environment_lock_manifest.get("secure_install_verified")
            if environment_lock_manifest is not None
            else False
        ),
        "evidence_gate_passed": dependency_lock_gate_passed,
    },
    "memory": {
        "host_rss": "Per-lane GNU-time MaxRSS is recorded in each pair.",
        "gpu_process_memory_summary": "machine/gpu_process_memory_summary.json",
        "gpu_process_memory": gpu_memory_summary,
        "evidence_gate": gpu_memory_gate,
    },
    "production_evidence_gates_passed": production_evidence_gates_passed,
}
output_path = artifact_root / "comparison.json"
temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
temporary_path.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
temporary_path.replace(output_path)
if profile == "full" and not aggregate_claim_ready:
    raise SystemExit(1)
PY
}

finalize_job() {
    local exit_code=$?
    local last_pair=""
    set +e
    rm -f "${GPU_MONITOR_SENTINEL}"
    if [[ -n "${GPU_MONITOR_PID}" ]]; then
        wait "${GPU_MONITOR_PID}"
    fi
    rm -f "${CURRENT_PAIR_PATH}"
    nvidia-smi -q > "${MACHINE_ROOT}/nvidia_smi_after.txt" 2>&1
    nvidia-smi --query-gpu=index,name,uuid,memory.used,memory.total \
        --format=csv,noheader > "${MACHINE_ROOT}/gpu_inventory_after.csv" 2>&1
    sacct -j "${SLURM_JOB_ID}" \
        --format=JobIDRaw,State,ExitCode,Elapsed,MaxRSS,AllocTRES%80 -P \
        > "${MACHINE_ROOT}/sacct_final.txt" 2>&1
    if [[ -f "${MACHINE_ROOT}/last_pair.txt" ]]; then
        last_pair="$(<"${MACHINE_ROOT}/last_pair.txt")"
    fi
    if ! summarize_gpu_process_memory; then
        exit_code=1
    fi
    if ! aggregate_pairs; then
        exit_code=1
    fi
    FINAL_EXIT_CODE="${exit_code}" \
        FINAL_PROFILE="${PROFILE}" \
        FINAL_EVIDENCE_CLASS="${EVIDENCE_CLASS}" \
        FINAL_PAIR_EXECUTION_MODE="${PAIR_EXECUTION_MODE}" \
        FINAL_CURRENT_PAIR="${last_pair}" \
        FINAL_STATUS_PATH="${MACHINE_ROOT}/job_status.json" \
        python - <<'PY'
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path


exit_code = int(os.environ["FINAL_EXIT_CODE"])
payload = {
    "schema_version": 1,
    "status": "completed" if exit_code == 0 else "failed",
    "exit_code": exit_code,
    "profile": os.environ["FINAL_PROFILE"],
    "evidence_class": os.environ["FINAL_EVIDENCE_CLASS"],
    "pair_execution_mode": os.environ["FINAL_PAIR_EXECUTION_MODE"],
    "last_pair": os.environ["FINAL_CURRENT_PAIR"] or None,
    "finished_at_utc": datetime.now(timezone.utc).isoformat(),
}
output_path = Path(os.environ["FINAL_STATUS_PATH"])
temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
temporary_path.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
temporary_path.replace(output_path)
PY
    trap - EXIT
    exit "${exit_code}"
}

trap finalize_job EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

for required_file in \
    "${REPO_ROOT}/pyproject.toml" \
    "${SURFACE_PATH}" \
    "${BIOTSAVART_FILE}" \
    "${BOOZER_STATE_PATH}"
do
    if [[ ! -f "${required_file}" ]]; then
        echo "Required input does not exist: ${required_file}" >&2
        exit 2
    fi
done

git -C "${REPO_ROOT}" rev-parse HEAD > "${MACHINE_ROOT}/git_commit.txt"
git -C "${REPO_ROOT}" rev-parse 'HEAD^{tree}' > "${MACHINE_ROOT}/git_tree.txt"
git -C "${REPO_ROOT}" status --porcelain=v1 --untracked-files=normal \
    > "${MACHINE_ROOT}/git_status.txt"
sed -E '/^\?\? slurm-[0-9]+\.(out|err)$/d' \
    "${MACHINE_ROOT}/git_status.txt" > "${MACHINE_ROOT}/git_source_status.txt"
if [[ -s "${MACHINE_ROOT}/git_source_status.txt" ]]; then
    echo "Refusing performance run from a dirty source tree" >&2
    cat "${MACHINE_ROOT}/git_source_status.txt" >&2
    exit 2
fi

SURFACE_ORIGINAL="$(realpath -- "${SURFACE_PATH}")"
BIOTSAVART_ORIGINAL="$(realpath -- "${BIOTSAVART_FILE}")"
BOOZER_STATE_ORIGINAL="$(realpath -- "${BOOZER_STATE_PATH}")"
mkdir -p "${INPUT_ROOT}/surface" "${INPUT_ROOT}/biotsavart" "${INPUT_ROOT}/boozer_state"
SURFACE_SNAPSHOT="${INPUT_ROOT}/surface/$(basename -- "${SURFACE_ORIGINAL}")"
BIOTSAVART_SNAPSHOT="${INPUT_ROOT}/biotsavart/$(basename -- "${BIOTSAVART_ORIGINAL}")"
BOOZER_STATE_SNAPSHOT="${INPUT_ROOT}/boozer_state/$(basename -- "${BOOZER_STATE_ORIGINAL}")"
cp --reflink=auto --preserve=mode,timestamps -- \
    "${SURFACE_ORIGINAL}" "${SURFACE_SNAPSHOT}"
cp --reflink=auto --preserve=mode,timestamps -- \
    "${BIOTSAVART_ORIGINAL}" "${BIOTSAVART_SNAPSHOT}"
cp --reflink=auto --preserve=mode,timestamps -- \
    "${BOOZER_STATE_ORIGINAL}" "${BOOZER_STATE_SNAPSHOT}"
chmod a-w "${SURFACE_SNAPSHOT}" "${BIOTSAVART_SNAPSHOT}" \
    "${BOOZER_STATE_SNAPSHOT}"

REQUESTED_SURFACE_PATH="${SURFACE_PATH}" \
    REQUESTED_BIOTSAVART_PATH="${BIOTSAVART_FILE}" \
    REQUESTED_BOOZER_STATE_PATH="${BOOZER_STATE_PATH}" \
    ORIGINAL_SURFACE_PATH="${SURFACE_ORIGINAL}" \
    ORIGINAL_BIOTSAVART_PATH="${BIOTSAVART_ORIGINAL}" \
    ORIGINAL_BOOZER_STATE_PATH="${BOOZER_STATE_ORIGINAL}" \
    SNAPSHOT_SURFACE_PATH="${SURFACE_SNAPSHOT}" \
    SNAPSHOT_BIOTSAVART_PATH="${BIOTSAVART_SNAPSHOT}" \
    SNAPSHOT_BOOZER_STATE_PATH="${BOOZER_STATE_SNAPSHOT}" \
    INPUT_MANIFEST_PATH="${INPUT_ROOT}/manifest.json" \
    IOTA_TARGET="${IOTA_TARGET}" \
    python - <<'PY'
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path


def sha256_file(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


inputs = {}
for name, environment_suffix in (
    ("surface", "SURFACE"),
    ("biotsavart", "BIOTSAVART"),
    ("boozer_state", "BOOZER_STATE"),
):
    original_path = Path(os.environ[f"ORIGINAL_{environment_suffix}_PATH"])
    snapshot_path = Path(os.environ[f"SNAPSHOT_{environment_suffix}_PATH"])
    original_sha256 = sha256_file(original_path)
    snapshot_sha256 = sha256_file(snapshot_path)
    if original_sha256 != snapshot_sha256:
        raise RuntimeError(f"Input snapshot hash mismatch for {name}")
    inputs[name] = {
        "requested_path": os.environ[f"REQUESTED_{environment_suffix}_PATH"],
        "resolved_original_path": str(original_path),
        "original_sha256": original_sha256,
        "snapshot_path": str(snapshot_path),
        "snapshot_sha256": snapshot_sha256,
    }

payload = {
    "schema_version": 1,
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "iota_target": float(os.environ["IOTA_TARGET"]),
    "inputs": inputs,
}
output_path = Path(os.environ["INPUT_MANIFEST_PATH"])
temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
temporary_path.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
temporary_path.replace(output_path)
PY

SOURCE_SHA="$(<"${MACHINE_ROOT}/git_commit.txt")"
SOURCE_TREE="$(<"${MACHINE_ROOT}/git_tree.txt")"
python -m venv "${VENV_DIR}"
printf '%s\n' "${VENV_DIR}" > "${MACHINE_ROOT}/venv_path.txt"

if [[ "${PROFILE}" == "pilot" ]]; then
    "${VENV_DIR}/bin/python" -m pip install --upgrade \
        'pip==26.1.2' 'setuptools==83.0.0' 'wheel==0.47.0'
    "${VENV_DIR}/bin/python" -m pip install \
        'scikit-build-core' 'pybind11<3.0.0' 'setuptools_scm>=8.0' \
        'numpy>=2.0,<2.5'
    CMAKE_BUILD_TYPE=Release CXXFLAGS='-O3' \
        "${VENV_DIR}/bin/python" -m pip install \
        --no-build-isolation \
        -e "${REPO_ROOT}[JAX_GPU,test,ALGS]" \
        'shapely==2.1.2' \
        'numba==0.65.1'
    "${VENV_DIR}/bin/python" -m pip check

    PILOT_LOCK_INPUT="${ENVIRONMENT_ROOT}/resolved_requirements.txt"
    PILOT_WHEELHOUSE="${ENVIRONMENT_ROOT}/wheelhouse"
    PILOT_ARTIFACT_MANIFEST="${ENVIRONMENT_ROOT}/downloaded_artifacts.json"
    "${VENV_DIR}/bin/python" -m pip freeze --all --exclude-editable \
        > "${PILOT_LOCK_INPUT}"
    mkdir -p "${PILOT_WHEELHOUSE}"
    "${VENV_DIR}/bin/python" -m pip download \
        --disable-pip-version-check \
        --no-deps \
        --only-binary=:all: \
        --dest "${PILOT_WHEELHOUSE}" \
        --requirement "${PILOT_LOCK_INPUT}"

    PILOT_LOCK_INPUT="${PILOT_LOCK_INPUT}" \
        PILOT_WHEELHOUSE="${PILOT_WHEELHOUSE}" \
        PILOT_ARTIFACT_MANIFEST="${PILOT_ARTIFACT_MANIFEST}" \
        ENVIRONMENT_LOCK_SNAPSHOT="${ENVIRONMENT_LOCK_SNAPSHOT}" \
        SOURCE_SHA="${SOURCE_SHA}" \
        SOURCE_TREE="${SOURCE_TREE}" \
        "${VENV_DIR}/bin/python" - <<'PY'
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version


def sha256_file(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


lock_input_path = Path(os.environ["PILOT_LOCK_INPUT"])
wheelhouse_path = Path(os.environ["PILOT_WHEELHOUSE"])
lock_path = Path(os.environ["ENVIRONMENT_LOCK_SNAPSHOT"])
artifact_manifest_path = Path(os.environ["PILOT_ARTIFACT_MANIFEST"])

requirements: dict[str, tuple[str, Version]] = {}
for line_number, raw_line in enumerate(
    lock_input_path.read_text(encoding="utf-8").splitlines(),
    start=1,
):
    line = raw_line.strip()
    if not line or line.startswith("#"):
        continue
    requirement = Requirement(line)
    specifiers = tuple(requirement.specifier)
    if (
        requirement.url is not None
        or requirement.extras
        or requirement.marker is not None
        or len(specifiers) != 1
        or specifiers[0].operator != "=="
    ):
        raise RuntimeError(
            "Pilot resolution did not produce one exact index requirement "
            f"at line {line_number}: {line!r}"
        )
    canonical_name = canonicalize_name(requirement.name)
    if canonical_name in requirements:
        raise RuntimeError(f"Duplicate resolved requirement: {canonical_name}")
    requirements[canonical_name] = (
        requirement.name,
        Version(specifiers[0].version),
    )

artifacts: dict[str, tuple[Path, Version, str]] = {}
for wheel_path in sorted(wheelhouse_path.iterdir()):
    if wheel_path.suffix != ".whl":
        raise RuntimeError(
            f"Production lock generation requires wheels; got {wheel_path.name}"
        )
    distribution, version, _, _ = parse_wheel_filename(wheel_path.name)
    canonical_name = canonicalize_name(distribution)
    if canonical_name in artifacts:
        raise RuntimeError(
            f"Expected one selected wheel for {canonical_name}; found multiple"
        )
    artifacts[canonical_name] = (
        wheel_path,
        version,
        sha256_file(wheel_path),
    )

missing = sorted(requirements.keys() - artifacts.keys())
unexpected = sorted(artifacts.keys() - requirements.keys())
if missing or unexpected:
    raise RuntimeError(
        f"Resolved requirement/wheel mismatch: missing={missing}, unexpected={unexpected}"
    )

lock_lines = [
    "# Generated by the Perlmutter pilot from exact resolved wheels.",
    "# Install with: python -m pip install --require-hashes "
    "--only-binary=:all: -r requirements.lock",
]
serialized_artifacts = []
for canonical_name in sorted(requirements):
    requirement_name, requirement_version = requirements[canonical_name]
    wheel_path, wheel_version, wheel_sha256 = artifacts[canonical_name]
    if wheel_version != requirement_version:
        raise RuntimeError(
            f"Version mismatch for {canonical_name}: "
            f"requirement={requirement_version}, wheel={wheel_version}"
        )
    lock_lines.extend(
        (
            f"{requirement_name}=={requirement_version} \\",
            f"    --hash=sha256:{wheel_sha256}",
        )
    )
    serialized_artifacts.append(
        {
            "name": canonical_name,
            "version": str(requirement_version),
            "filename": wheel_path.name,
            "size_bytes": wheel_path.stat().st_size,
            "sha256": wheel_sha256,
        }
    )

lock_path.write_text("\n".join(lock_lines) + "\n", encoding="utf-8")
artifact_manifest_path.write_text(
    json.dumps(
        {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_commit": os.environ["SOURCE_SHA"],
            "source_tree": os.environ["SOURCE_TREE"],
            "python": sys.version,
            "platform": platform.platform(),
            "lock_input_sha256": sha256_file(lock_input_path),
            "lock_sha256": sha256_file(lock_path),
            "artifacts": serialized_artifacts,
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY
    rm -rf "${PILOT_WHEELHOUSE}"
    chmod a-w "${PILOT_LOCK_INPUT}" "${PILOT_ARTIFACT_MANIFEST}" \
        "${ENVIRONMENT_LOCK_SNAPSHOT}"
    ENVIRONMENT_LOCK_SHA256="$(sha256sum "${ENVIRONMENT_LOCK_SNAPSHOT}" | cut -d' ' -f1)"
    LOCK_ORIGIN="pilot-generated"
    SECURE_INSTALL_VERIFIED="false"
    echo "Pilot generated production lock: ${ENVIRONMENT_LOCK_SNAPSHOT}" >&2
    echo \
        "Submit PROFILE=full with HASH_LOCK_PATH=${ENVIRONMENT_LOCK_SNAPSHOT}" \
        >&2
else
    if [[ ! -f "${HASH_LOCK_PATH}" ]]; then
        echo "HASH_LOCK_PATH does not exist: ${HASH_LOCK_PATH}" >&2
        exit 2
    fi
    HASH_LOCK_ORIGINAL="$(realpath -- "${HASH_LOCK_PATH}")"
    cp --preserve=mode,timestamps -- \
        "${HASH_LOCK_ORIGINAL}" "${ENVIRONMENT_LOCK_SNAPSHOT}"
    chmod a-w "${ENVIRONMENT_LOCK_SNAPSHOT}"
    ENVIRONMENT_LOCK_SHA256="$(sha256sum "${ENVIRONMENT_LOCK_SNAPSHOT}" | cut -d' ' -f1)"
    LOCK_ORIGIN="${HASH_LOCK_ORIGINAL}"

    "${VENV_DIR}/bin/python" -m pip install \
        --require-hashes \
        --only-binary=:all: \
        --requirement "${ENVIRONMENT_LOCK_SNAPSHOT}"
    CMAKE_BUILD_TYPE=Release CXXFLAGS='-O3' \
        "${VENV_DIR}/bin/python" -m pip install \
        --no-deps \
        --no-build-isolation \
        -e "${REPO_ROOT}"
    "${VENV_DIR}/bin/python" -m pip check
    SECURE_INSTALL_VERIFIED="true"
fi

"${VENV_DIR}/bin/python" -m pip freeze --all --exclude-editable \
    > "${ENVIRONMENT_ROOT}/installed_requirements.txt"
ENVIRONMENT_LOCK_SNAPSHOT="${ENVIRONMENT_LOCK_SNAPSHOT}" \
    INSTALLED_REQUIREMENTS="${ENVIRONMENT_ROOT}/installed_requirements.txt" \
    "${VENV_DIR}/bin/python" - <<'PY'
from __future__ import annotations

import os
from pathlib import Path
import re


def canonicalize_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def exact_versions(path: Path, *, lock: bool) -> dict[str, str]:
    versions: dict[str, str] = {}
    pattern = re.compile(r"^([A-Za-z0-9_.-]+)==([^\s\\]+)")
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("--hash="):
            continue
        match = pattern.match(line)
        if match is None:
            kind = "lock" if lock else "installed requirement"
            raise RuntimeError(
                f"Non-exact {kind} at {path}:{line_number}: {line!r}"
            )
        name = canonicalize_name(match.group(1))
        if name in versions:
            raise RuntimeError(f"Duplicate requirement for {name} in {path}")
        versions[name] = match.group(2)
    return versions


lock_versions = exact_versions(
    Path(os.environ["ENVIRONMENT_LOCK_SNAPSHOT"]),
    lock=True,
)
installed_versions = exact_versions(
    Path(os.environ["INSTALLED_REQUIREMENTS"]),
    lock=False,
)
if lock_versions != installed_versions:
    missing = sorted(lock_versions.keys() - installed_versions.keys())
    unexpected = sorted(installed_versions.keys() - lock_versions.keys())
    changed = {
        name: {
            "lock": lock_versions[name],
            "installed": installed_versions[name],
        }
        for name in sorted(lock_versions.keys() & installed_versions.keys())
        if lock_versions[name] != installed_versions[name]
    }
    raise RuntimeError(
        "Installed environment does not match dependency lock: "
        f"missing={missing}, unexpected={unexpected}, changed={changed}"
    )
PY

PROFILE="${PROFILE}" \
    EVIDENCE_CLASS="${EVIDENCE_CLASS}" \
    PAIR_EXECUTION_MODE="${PAIR_EXECUTION_MODE}" \
    LOCK_ORIGIN="${LOCK_ORIGIN}" \
    ENVIRONMENT_LOCK_SNAPSHOT="${ENVIRONMENT_LOCK_SNAPSHOT}" \
    ENVIRONMENT_LOCK_SHA256="${ENVIRONMENT_LOCK_SHA256}" \
    SECURE_INSTALL_VERIFIED="${SECURE_INSTALL_VERIFIED}" \
    SOURCE_SHA="${SOURCE_SHA}" \
    SOURCE_TREE="${SOURCE_TREE}" \
    ENVIRONMENT_LOCK_MANIFEST="${ENVIRONMENT_LOCK_MANIFEST}" \
    "${VENV_DIR}/bin/python" - <<'PY'
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path


def sha256_file(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


lock_path = Path(os.environ["ENVIRONMENT_LOCK_SNAPSHOT"])
lock_sha256 = sha256_file(lock_path)
if lock_sha256 != os.environ["ENVIRONMENT_LOCK_SHA256"]:
    raise RuntimeError("Dependency lock changed after selection")
payload = {
    "schema_version": 1,
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "profile": os.environ["PROFILE"],
    "evidence_class": os.environ["EVIDENCE_CLASS"],
    "pair_execution_mode": os.environ["PAIR_EXECUTION_MODE"],
    "lock_origin": os.environ["LOCK_ORIGIN"],
    "lock_snapshot": str(lock_path),
    "lock_sha256": lock_sha256,
    "secure_install_verified": os.environ["SECURE_INSTALL_VERIFIED"] == "true",
    "install_contract": {
        "dependencies": "pip --require-hashes --only-binary=:all:",
        "repository": "pip --no-deps --no-build-isolation -e",
    },
    "source_commit": os.environ["SOURCE_SHA"],
    "source_tree": os.environ["SOURCE_TREE"],
}
manifest_path = Path(os.environ["ENVIRONMENT_LOCK_MANIFEST"])
manifest_path.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

export PYTHONHASHSEED=0
export JAX_ENABLE_X64=1
export JAX_ENABLE_COMPILATION_CACHE=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export SIMSOPT_JAX_CUDA_LIBRARY_MODE=bundled
export SIMSOPT_MIXED_PRECISION=0
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

printf '%s\n' "${PAIR_EXECUTION_MODE}" \
    > "${MACHINE_ROOT}/pair_execution_mode.txt"
module list > "${MACHINE_ROOT}/modules.txt" 2>&1
lscpu > "${MACHINE_ROOT}/lscpu.txt"
nvidia-smi -q > "${MACHINE_ROOT}/nvidia_smi_before.txt"
nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version \
    --format=csv,noheader > "${MACHINE_ROOT}/gpu_inventory.csv"
scontrol show job "${SLURM_JOB_ID}" > "${MACHINE_ROOT}/slurm_job.txt"
"${VENV_DIR}/bin/python" -m pip freeze --all > "${MACHINE_ROOT}/pip_freeze.txt"
sha256sum "${SURFACE_ORIGINAL}" "${BIOTSAVART_ORIGINAL}" \
    "${BOOZER_STATE_ORIGINAL}" > "${MACHINE_ROOT}/original_input_sha256.txt"
sha256sum "${SURFACE_SNAPSHOT}" "${BIOTSAVART_SNAPSHOT}" \
    "${BOOZER_STATE_SNAPSHOT}" > "${MACHINE_ROOT}/snapshot_input_sha256.txt"

"${VENV_DIR}/bin/python" - <<'PY' > "${MACHINE_ROOT}/python_runtime.txt"
from __future__ import annotations

import hashlib
from pathlib import Path
import platform
import sys

import simsoptpp


extension = Path(simsoptpp.__file__).resolve()
digest_builder = hashlib.sha256()
with extension.open("rb") as handle:
    while chunk := handle.read(1024 * 1024):
        digest_builder.update(chunk)
print(f"python={sys.version}")
print(f"platform={platform.platform()}")
print(f"simsoptpp={extension}")
print(f"simsoptpp_sha256={digest_builder.hexdigest()}")
PY

printf 'timestamp_utc,pair,gpu_uuid,pid,used_memory_mib\n' \
    > "${GPU_PROCESS_MEMORY_CSV}"
printf 'timestamp_utc,pair,status,exit_code\n' > "${GPU_SAMPLER_QUERY_CSV}"
: > "${GPU_SAMPLER_ERROR_LOG}"
touch "${GPU_MONITOR_SENTINEL}"
(
    sampler_last_error="${MACHINE_ROOT}/gpu_sampler_last_error.txt"
    exec 3>> "${GPU_PROCESS_MEMORY_CSV}"
    exec 4>> "${GPU_SAMPLER_QUERY_CSV}"
    exec 5>> "${GPU_SAMPLER_ERROR_LOG}"
    while [[ -e "${GPU_MONITOR_SENTINEL}" ]]
    do
        timestamp="$(date --utc +%Y-%m-%dT%H:%M:%S.%NZ)"
        pair_name="unassigned"
        if [[ -s "${CURRENT_PAIR_PATH}" ]]; then
            pair_name="$(<"${CURRENT_PAIR_PATH}")"
        fi
        if sample="$(nvidia-smi \
            --query-compute-apps=gpu_uuid,pid,used_memory \
            --format=csv,noheader,nounits 2> "${sampler_last_error}")"
        then
            printf '%s,%s,ok,0\n' "${timestamp}" "${pair_name}" >&4
            if [[ -n "${sample}" ]]; then
                while IFS= read -r process_record
                do
                    printf '%s,%s,%s\n' \
                        "${timestamp}" "${pair_name}" "${process_record}" >&3
                done <<< "${sample}"
            fi
        else
            sampler_exit_code=$?
            printf '%s,%s,failed,%s\n' \
                "${timestamp}" "${pair_name}" "${sampler_exit_code}" >&4
            printf '[%s] pair=%s exit_code=%s\n' \
                "${timestamp}" "${pair_name}" "${sampler_exit_code}" >&5
            sed 's/^/  /' "${sampler_last_error}" >&5
        fi
        sleep 1
    done
    rm -f "${sampler_last_error}"
) &
GPU_MONITOR_PID=$!

configure_pair_launcher "${PAIR_EXECUTION_MODE}"

FAILED_PAIRS=0
for ((pair_index = 1; pair_index <= PAIR_COUNT; pair_index++))
do
    pair_name="pair-$(printf '%02d' "${pair_index}")"
    pair_dir="${PAIR_ROOT}/${pair_name}"
    if ((pair_index % 2 == 1)); then
        pair_order="cpu-jax"
    else
        pair_order="jax-cpu"
    fi
    printf '%s\n' "${pair_name}" > "${CURRENT_PAIR_PATH}"
    printf '%s\n' "${pair_name}" > "${MACHINE_ROOT}/last_pair.txt"
    pair_started_at="$(date --utc +%Y-%m-%dT%H:%M:%S.%NZ)"
    echo "Starting ${pair_name}/${PAIR_COUNT} with order ${pair_order}" >&2

    if "${PAIR_LAUNCHER[@]}" "${VENV_DIR}/bin/python" \
        "${REPO_ROOT}/benchmarks/single_stage_full_loop_compare.py" \
        --surface-path "${SURFACE_SNAPSHOT}" \
        --biotsavart-file "${BIOTSAVART_SNAPSHOT}" \
        --boozer-state-path "${BOOZER_STATE_SNAPSHOT}" \
        --output-root "${pair_dir}" \
        --order "${pair_order}" \
        --mode "${EVIDENCE_CLASS}" \
        --mpol "${MPOL}" \
        --ntor "${NTOR}" \
        --nphi "${NPHI}" \
        --ntheta "${NTHETA}" \
        --constraint-weight "${CONSTRAINT_WEIGHT}" \
        --iota-target "${IOTA_TARGET}" \
        --environment-lock-sha256 "${ENVIRONMENT_LOCK_SHA256}" \
        --maxiter "${OUTER_MAXITER}" \
        --outer-maxcor "${OUTER_MAXCOR}" \
        --outer-maxls "${OUTER_MAXLS}" \
        --outer-ftol "${OUTER_FTOL}" \
        --outer-gtol "${OUTER_GTOL}" \
        --boozer-bfgs-tol "${BOOZER_BFGS_TOL}" \
        --boozer-bfgs-maxiter "${BOOZER_BFGS_MAXITER}" \
        --boozer-newton-tol "${BOOZER_NEWTON_TOL}" \
        --boozer-newton-maxiter "${BOOZER_NEWTON_MAXITER}"
    then
        comparator_exit_code=0
    else
        comparator_exit_code=$?
    fi
    pair_finished_at="$(date --utc +%Y-%m-%dT%H:%M:%S.%NZ)"
    mkdir -p "${pair_dir}"

    comparison_status="missing"
    comparison_claim_ready="false"
    cpu_optimizer_success="false"
    jax_optimizer_success="false"
    if [[ -f "${pair_dir}/comparison.json" ]]; then
        read -r comparison_status comparison_claim_ready cpu_optimizer_success \
            jax_optimizer_success < <(
            PAIR_DIR="${pair_dir}" "${VENV_DIR}/bin/python" - <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path


pair_dir = Path(os.environ["PAIR_DIR"])
comparison = json.loads(
    (pair_dir / "comparison.json").read_text(encoding="utf-8")
)
cpu_results = json.loads(
    (pair_dir / "cpu" / "results.json").read_text(encoding="utf-8")
)
jax_results = json.loads(
    (pair_dir / "jax" / "results.json").read_text(encoding="utf-8")
)
print(
    comparison["status"],
    str(bool(comparison["claim_ready"])).lower(),
    str(bool(cpu_results["optimizer"]["success"])).lower(),
    str(bool(jax_results["optimizer"]["success"])).lower(),
)
PY
        )
    fi

    adjudicated_status="failed"
    if [[ "${comparator_exit_code}" -eq 0 ]]; then
        if [[ "${PROFILE}" == "pilot" && "${comparison_status}" == "diagnostic" ]]; then
            adjudicated_status="passed"
        elif [[ "${PROFILE}" == "full" && "${comparison_status}" == "passed" \
            && "${comparison_claim_ready}" == "true" ]]; then
            adjudicated_status="passed"
        fi
    fi
    if [[ "${adjudicated_status}" != "passed" ]]; then
        FAILED_PAIRS=$((FAILED_PAIRS + 1))
    fi

    {
        printf 'pair_index\t%s\n' "${pair_index}"
        printf 'pair_name\t%s\n' "${pair_name}"
        printf 'order\t%s\n' "${pair_order}"
        printf 'profile\t%s\n' "${PROFILE}"
        printf 'evidence_class\t%s\n' "${EVIDENCE_CLASS}"
        printf 'pair_execution_mode\t%s\n' "${PAIR_EXECUTION_MODE}"
        printf 'started_at_utc\t%s\n' "${pair_started_at}"
        printf 'finished_at_utc\t%s\n' "${pair_finished_at}"
        printf 'comparator_exit_code\t%s\n' "${comparator_exit_code}"
        printf 'comparison_status\t%s\n' "${comparison_status}"
        printf 'comparison_claim_ready\t%s\n' "${comparison_claim_ready}"
        printf 'cpu_optimizer_success\t%s\n' "${cpu_optimizer_success}"
        printf 'jax_optimizer_success\t%s\n' "${jax_optimizer_success}"
        printf 'adjudicated_status\t%s\n' "${adjudicated_status}"
    } > "${pair_dir}/pair_status.tsv"
    echo "Finished ${pair_name}: ${adjudicated_status}" >&2
done

rm -f "${CURRENT_PAIR_PATH}"
echo "Full-loop artifact root: ${ARTIFACT_ROOT}"
if ((FAILED_PAIRS > 0)); then
    echo "${FAILED_PAIRS}/${PAIR_COUNT} pairs failed adjudication" >&2
    exit 1
fi
