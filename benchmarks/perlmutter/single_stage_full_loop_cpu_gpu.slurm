#!/bin/bash -l
#SBATCH -A m4680
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 36:00:00
#SBATCH -N 2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --hint=nomultithread
#SBATCH --job-name=banana-full-loop
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH hetjob
#SBATCH -A m4680_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 36:00:00
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
# Keep SMT enabled on the shared GPU slice: 32 logical CPUs are 16 physical cores.
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1

set -euo pipefail

write_gpu_memory_hook() {
    local pair_dir="$1"
    local pair_name="$2"
    local actual_node="$3"
    local python_executable="$4"
    local process_memory_csv="$5"
    local sampler_query_csv="$6"
    local inventory_csv="$7"

    GPU_HOOK_PAIR_DIR="${pair_dir}" \
        GPU_HOOK_PAIR_NAME="${pair_name}" \
        GPU_HOOK_NODE="${actual_node}" \
        GPU_HOOK_PROCESS_MEMORY_CSV="${process_memory_csv}" \
        GPU_HOOK_SAMPLER_QUERY_CSV="${sampler_query_csv}" \
        GPU_HOOK_INVENTORY_CSV="${inventory_csv}" \
        "${python_executable}" - <<'PY'
from __future__ import annotations

import csv
import json
import os
from pathlib import Path


pair_dir = Path(os.environ["GPU_HOOK_PAIR_DIR"])
pair_name = os.environ["GPU_HOOK_PAIR_NAME"]
node = os.environ["GPU_HOOK_NODE"]
process_memory_csv = Path(os.environ["GPU_HOOK_PROCESS_MEMORY_CSV"])
sampler_query_csv = Path(os.environ["GPU_HOOK_SAMPLER_QUERY_CSV"])
inventory_csv = Path(os.environ["GPU_HOOK_INVENTORY_CSV"])

sample_count = 0
gpu_uuids: set[str] = set()
process_ids: set[int] = set()
first_sample_at_utc: str | None = None
last_sample_at_utc: str | None = None
maximum_used_memory_mib: int | None = None
with process_memory_csv.open(encoding="utf-8", newline="") as handle:
    for row in csv.DictReader(handle):
        if row["pair"].strip() != pair_name or row["node"].strip() != node:
            continue
        timestamp_utc = row["timestamp_utc"].strip()
        gpu_uuid = row["gpu_uuid"].strip()
        pid = int(row["pid"])
        used_memory_mib = int(row["used_memory_mib"])
        sample_count += 1
        gpu_uuids.add(gpu_uuid)
        process_ids.add(pid)
        if first_sample_at_utc is None:
            first_sample_at_utc = timestamp_utc
        last_sample_at_utc = timestamp_utc
        maximum_used_memory_mib = (
            used_memory_mib
            if maximum_used_memory_mib is None
            else max(maximum_used_memory_mib, used_memory_mib)
        )

query_count = 0
query_failure_count = 0
with sampler_query_csv.open(encoding="utf-8", newline="") as handle:
    for row in csv.DictReader(handle):
        if row["pair"].strip() != pair_name or row["node"].strip() != node:
            continue
        query_count += 1
        query_failure_count += int(row["status"].strip() != "ok")

gpu_inventory: list[dict[str, str | int]] = []
with inventory_csv.open(encoding="utf-8", newline="") as handle:
    for row in csv.reader(handle, skipinitialspace=True):
        gpu_inventory.append(
            {
                "index": int(row[0]),
                "name": row[1].strip(),
                "uuid": row[2].strip(),
                "memory_total_mib": int(row[3].strip().removesuffix(" MiB")),
                "driver_version": row[4].strip(),
            }
        )

payload = {
    "schema_version": 1,
    "metric": "nvidia-smi compute-process used_memory",
    "unit": "MiB",
    "pair": pair_name,
    "node": node,
    "slurm_step_id": os.environ["SLURM_STEP_ID"],
    "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
    "gpu_inventory": gpu_inventory,
    "sample_count": sample_count,
    "gpu_uuids": sorted(gpu_uuids),
    "process_ids": sorted(process_ids),
    "first_sample_at_utc": first_sample_at_utc,
    "last_sample_at_utc": last_sample_at_utc,
    "maximum_used_memory_mib": maximum_used_memory_mib,
    "sampler_queries": {
        "query_count": query_count,
        "successful_query_count": query_count - query_failure_count,
        "failure_count": query_failure_count,
        "all_succeeded": query_count > 0 and query_failure_count == 0,
    },
}
output_path = pair_dir / "jax_gpu" / "gpu_process_memory.json"
temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
temporary_path.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
temporary_path.replace(output_path)
PY
}

run_lane_step() {
    local lane="$1"
    local expected_node="$2"
    local pair_dir="$3"
    local python_executable="$4"
    local comparator_path="$5"
    local process_memory_csv="$6"
    local sampler_query_csv="$7"
    local sampler_error_log="$8"
    local expected_cpu_count="$9"
    local expected_omp_thread_count="${10}"
    local pair_name
    local actual_node
    local assigned_gpu_id=""
    local visible_devices="${CUDA_VISIBLE_DEVICES:-}"
    local -a step_nodes=()

    pair_name="$(basename -- "${pair_dir}")"
    actual_node="$(hostname -s)"
    mapfile -t step_nodes < <(scontrol show hostnames "${SLURM_STEP_NODELIST:?}")
    if ((${#step_nodes[@]} != 1)) \
        || [[ "${SLURM_STEP_NUM_NODES:?}" != "1" ]] \
        || [[ "${step_nodes[0]}" != "${expected_node}" ]] \
        || [[ "${actual_node}" != "${expected_node}" ]]; then
        echo \
            "${lane} step node mismatch: expected=${expected_node} actual=${actual_node} step_nodes=${step_nodes[*]}" \
            >&2
        return 2
    fi

    if [[ "${SLURM_CPUS_PER_TASK:?}" != "${expected_cpu_count}" ]]; then
        echo \
            "${lane} step CPU mismatch: expected=${expected_cpu_count} actual=${SLURM_CPUS_PER_TASK}" \
            >&2
        return 2
    fi
    if [[ "${OMP_NUM_THREADS:?}" != "${expected_omp_thread_count}" ]]; then
        echo \
            "${lane} OMP thread mismatch: expected=${expected_omp_thread_count} actual=${OMP_NUM_THREADS}" \
            >&2
        return 2
    fi

    if [[ "${lane}" == "jax_gpu" ]]; then
        if [[ -z "${visible_devices}" || "${visible_devices}" == *,* \
            || "${visible_devices}" =~ [[:space:]] ]]; then
            echo "jax_gpu requires exactly one CUDA_VISIBLE_DEVICES selector" >&2
            return 2
        fi
        assigned_gpu_id="${SLURM_STEP_GPUS:-}"
        if [[ -z "${assigned_gpu_id}" || "${assigned_gpu_id}" == *,* \
            || "${assigned_gpu_id}" =~ [[:space:]] ]]; then
            echo "jax_gpu requires exactly one Slurm GPU assigned to its step" >&2
            return 2
        fi
    else
        if [[ -n "${visible_devices}" || -n "${SLURM_STEP_GPUS:-}" ]]; then
            echo "${lane} must not receive a GPU or CUDA visibility" >&2
            return 2
        fi
    fi

    {
        printf 'lane\t%s\n' "${lane}"
        printf 'expected_node\t%s\n' "${expected_node}"
        printf 'actual_node\t%s\n' "${actual_node}"
        printf 'slurm_step_id\t%s\n' "${SLURM_STEP_ID:?}"
        printf 'slurm_step_nodelist\t%s\n' "${SLURM_STEP_NODELIST}"
        printf 'slurm_step_gpus\t%s\n' "${SLURM_STEP_GPUS:-}"
        printf 'slurm_cpus_per_task\t%s\n' "${SLURM_CPUS_PER_TASK}"
        printf 'omp_num_threads\t%s\n' "${OMP_NUM_THREADS}"
        printf 'cuda_visible_devices\t%s\n' "${visible_devices}"
    } > "${pair_dir}/${lane}_slurm_step.tsv"
    lscpu > "${pair_dir}/${lane}_lscpu.txt"

    local lane_exit_code=0
    if [[ "${lane}" == "jax_gpu" ]]; then
        local gpu_monitor_sentinel="${pair_dir}/jax_gpu_monitor.running"
        local sampler_last_error="${pair_dir}/jax_gpu_sampler_last_error.txt"
        local gpu_memory_hook_returncode
        local gpu_monitor_pid

        nvidia-smi --id="${assigned_gpu_id}" -q \
            > "${pair_dir}/jax_gpu_nvidia_smi_before.txt"
        nvidia-smi --id="${assigned_gpu_id}" \
            --query-gpu=index,name,uuid,memory.total,driver_version \
            --format=csv,noheader > "${pair_dir}/jax_gpu_inventory.csv"
        touch "${gpu_monitor_sentinel}"
        (
            exec 3>> "${process_memory_csv}"
            exec 4>> "${sampler_query_csv}"
            exec 5>> "${sampler_error_log}"
            while [[ -e "${gpu_monitor_sentinel}" ]]
            do
                local_timestamp="$(date --utc +%Y-%m-%dT%H:%M:%S.%NZ)"
                if sample="$(nvidia-smi --id="${assigned_gpu_id}" \
                    --query-compute-apps=gpu_uuid,pid,used_memory \
                    --format=csv,noheader,nounits 2> "${sampler_last_error}")"
                then
                    printf '%s,%s,%s,ok,0\n' \
                        "${local_timestamp}" "${pair_name}" "${actual_node}" >&4
                    if [[ -n "${sample}" ]]; then
                        while IFS= read -r process_record
                        do
                            printf '%s,%s,%s,%s\n' \
                                "${local_timestamp}" "${pair_name}" \
                                "${actual_node}" "${process_record}" >&3
                        done <<< "${sample}"
                    fi
                else
                    sampler_exit_code=$?
                    printf '%s,%s,%s,failed,%s\n' \
                        "${local_timestamp}" "${pair_name}" "${actual_node}" \
                        "${sampler_exit_code}" >&4
                    printf '[%s] pair=%s node=%s exit_code=%s\n' \
                        "${local_timestamp}" "${pair_name}" "${actual_node}" \
                        "${sampler_exit_code}" >&5
                    sed 's/^/  /' "${sampler_last_error}" >&5
                fi
                sleep 1
            done
            rm -f "${sampler_last_error}"
        ) &
        gpu_monitor_pid=$!

        if "${python_executable}" "${comparator_path}" run-lane \
            --output-root "${pair_dir}" --lane "${lane}"
        then
            lane_exit_code=0
        else
            lane_exit_code=$?
        fi
        rm -f "${gpu_monitor_sentinel}"
        wait "${gpu_monitor_pid}"
        if write_gpu_memory_hook \
            "${pair_dir}" "${pair_name}" "${actual_node}" \
            "${python_executable}" "${process_memory_csv}" \
            "${sampler_query_csv}" "${pair_dir}/jax_gpu_inventory.csv"
        then
            gpu_memory_hook_returncode=0
        else
            gpu_memory_hook_returncode=$?
            if [[ "${lane_exit_code}" -eq 0 ]]; then
                lane_exit_code="${gpu_memory_hook_returncode}"
            fi
        fi
        nvidia-smi --id="${assigned_gpu_id}" -q \
            > "${pair_dir}/jax_gpu_nvidia_smi_after.txt"
        nvidia-smi --id="${assigned_gpu_id}" \
            --query-gpu=index,name,uuid,memory.used,memory.total \
            --format=csv,noheader > "${pair_dir}/jax_gpu_inventory_after.csv"
    elif "${python_executable}" "${comparator_path}" run-lane \
        --output-root "${pair_dir}" --lane "${lane}"
    then
        lane_exit_code=0
    else
        lane_exit_code=$?
    fi
    return "${lane_exit_code}"
}

if [[ "${1:-}" == "__run-lane-step" ]]; then
    shift
    run_lane_step "$@"
    exit $?
fi

module load python/3.13-26.1.0
unset PYTHONPATH
export PYTHONNOUSERSITE=1

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR}}"
LAUNCHER_PATH="$(realpath -- "${BASH_SOURCE[0]}")"
PROFILE="${PROFILE:-pilot}"
SEED_DIR="${SEED_DIR:?Set SEED_DIR to the immutable banana seed directory}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${SCRATCH}/simsopt-full-loop/${SLURM_JOB_ID}}"
IOTA_TARGET="${IOTA_TARGET:?Set IOTA_TARGET explicitly from the seed provenance}"
HASH_LOCK_PATH="${HASH_LOCK_PATH:-}"
VENV_ROOT="${VENV_ROOT:-${SCRATCH:?SCRATCH must be set}/simsopt-full-loop-envs-${UID}}"
PREPARED_VENV_DIR="${PREPARED_VENV_DIR:-}"
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

resolve_source_scm_provenance() {
    local repo_root="$1"
    local source_sha="$2"
    local description
    description="$(
        git -C "${repo_root}" describe \
            --tags --long --abbrev=40 --match 'v[0-9]*' "${source_sha}"
    )"
    if [[ ! "${description}" =~ ^(v[0-9][0-9A-Za-z.+]*)(-([0-9]+))-g([0-9a-f]{40})$ ]]; then
        echo \
            "Unsupported SIMSOPT release description for package version: ${description}" \
            >&2
        return 2
    fi
    SOURCE_SCM_DESCRIPTION="${description}"
    SOURCE_SCM_TAG="${BASH_REMATCH[1]}"
    SOURCE_SCM_DISTANCE="${BASH_REMATCH[3]}"
    SOURCE_SCM_NODE="g${BASH_REMATCH[4]}"
    if [[ "${SOURCE_SCM_NODE}" != "g${source_sha}" ]]; then
        echo "Source SCM node does not match the recorded source commit" >&2
        echo "source_commit=${source_sha}" >&2
        echo "source_scm_node=${SOURCE_SCM_NODE}" >&2
        return 2
    fi
    printf -v SOURCE_SCM_METADATA \
        '{node="%s", distance=%s, dirty=false}' \
        "${SOURCE_SCM_NODE}" "${SOURCE_SCM_DISTANCE}"
}

format_source_package_version() {
    local python_executable="$1"
    SOURCE_SCM_TAG="${SOURCE_SCM_TAG}" \
        SOURCE_SCM_DISTANCE="${SOURCE_SCM_DISTANCE}" \
        SOURCE_SCM_NODE="${SOURCE_SCM_NODE}" \
        "${python_executable}" - <<'PY'
from __future__ import annotations

import os

from setuptools_scm import Configuration
from setuptools_scm.version import format_version, meta


configuration = Configuration(root=".", dist_name="simsopt")
scm_version = meta(
    tag=os.environ["SOURCE_SCM_TAG"],
    distance=int(os.environ["SOURCE_SCM_DISTANCE"]),
    node=os.environ["SOURCE_SCM_NODE"],
    dirty=False,
    config=configuration,
)
print(format_version(scm_version))
PY
}

reserve_fresh_directory() {
    local root_label="$1"
    local root_path="$2"
    local directory_mode="${3:-}"
    mkdir -p -- "$(dirname -- "${root_path}")"
    if [[ -n "${directory_mode}" ]]; then
        if mkdir --mode="${directory_mode}" -- "${root_path}"; then
            return
        fi
    elif mkdir -- "${root_path}"; then
        return
    fi
    if [[ -e "${root_path}" ]]; then
        echo "${root_label} must not exist before launch: ${root_path}" >&2
    else
        echo "${root_label} could not be created: ${root_path}" >&2
    fi
    return 2
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

validate_prepared_venv() {
    local prepared_venv="$1"
    local owner_uid
    local parent_owner_uid
    local parent_permissions
    local permissions
    local python_abi
    if [[ -L "${prepared_venv}" ]]; then
        echo "PREPARED_VENV_DIR must not be a symbolic link: ${prepared_venv}" >&2
        return 2
    fi
    if [[ ! -d "${prepared_venv}" ]]; then
        echo "PREPARED_VENV_DIR must be an existing directory: ${prepared_venv}" >&2
        return 2
    fi
    owner_uid="$(stat -c %u -- "${prepared_venv}")"
    permissions="$(stat -c %a -- "${prepared_venv}")"
    if [[ "${owner_uid}" != "${UID}" ]]; then
        echo "PREPARED_VENV_DIR must be owned by the current user" >&2
        echo "PREPARED_VENV_DIR=${prepared_venv}" >&2
        echo "owner_uid=${owner_uid}" >&2
        return 2
    fi
    if [[ "${permissions}" == "770" ]]; then
        parent_owner_uid="$(stat -c %u -- "$(dirname -- "${prepared_venv}")")"
        parent_permissions="$(stat -c %a -- "$(dirname -- "${prepared_venv}")")"
        if [[
            "${parent_owner_uid}" != "${UID}"
            || "${parent_permissions}" != "700"
        ]]; then
            echo \
                "Mode-770 PREPARED_VENV_DIR requires an owner-only mode-700 parent" \
                >&2
            echo "PREPARED_VENV_DIR=${prepared_venv}" >&2
            echo \
                "parent_owner_uid=${parent_owner_uid} parent_permissions=${parent_permissions}" \
                >&2
            return 2
        fi
    elif [[ "${permissions}" != "700" ]]; then
        echo \
            "PREPARED_VENV_DIR must use mode 700, or mode 770 under an owner-only parent" \
            >&2
        echo "PREPARED_VENV_DIR=${prepared_venv}" >&2
        echo "permissions=${permissions}" >&2
        return 2
    fi
    if [[ ! -x "${prepared_venv}/bin/python" ]]; then
        echo "PREPARED_VENV_DIR has no executable bin/python" >&2
        return 2
    fi
    python_abi="$(
        "${prepared_venv}/bin/python" -c \
            'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'
    )"
    if [[ "${python_abi}" != "3.13" ]]; then
        echo "PREPARED_VENV_DIR must use Python 3.13; found ${python_abi}" >&2
        return 2
    fi
    if ! (
        cd /
        "${prepared_venv}/bin/python" - <<'PY'
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec


try:
    installed_simsopt = version("simsopt")
except PackageNotFoundError:
    installed_simsopt = None
if installed_simsopt is not None:
    raise RuntimeError(
        "PREPARED_VENV_DIR must not contain SIMSOPT before source installation: "
        f"found {installed_simsopt}"
    )
if find_spec("simsoptpp") is not None:
    raise RuntimeError(
        "PREPARED_VENV_DIR must not contain simsoptpp before source installation"
    )
PY
    ); then
        return 2
    fi
}

validate_external_write_root "ARTIFACT_ROOT" "${REPO_ROOT}" "${ARTIFACT_ROOT}"
REPO_ROOT="$(realpath -- "${REPO_ROOT}")"
ARTIFACT_ROOT="$(realpath -m -- "${ARTIFACT_ROOT}")"
if [[ -n "${PREPARED_VENV_DIR}" ]]; then
    if [[ -L "${PREPARED_VENV_DIR}" || ! -d "${PREPARED_VENV_DIR}" ]]; then
        echo \
            "PREPARED_VENV_DIR must be an existing non-symbolic-link directory" \
            >&2
        echo "PREPARED_VENV_DIR=${PREPARED_VENV_DIR}" >&2
        exit 2
    fi
    validate_external_write_root \
        "PREPARED_VENV_DIR" "${REPO_ROOT}" "${PREPARED_VENV_DIR}"
    VENV_DIR="$(realpath -- "${PREPARED_VENV_DIR}")"
    PREPARED_VENV_DIR="${VENV_DIR}"
    VENV_ROOT_REAL=""
else
    validate_external_write_root "VENV_ROOT" "${REPO_ROOT}" "${VENV_ROOT}"
    VENV_ROOT_REAL="$(realpath -m -- "${VENV_ROOT}")"
    VENV_DIR="${VENV_ROOT_REAL}/job-${SLURM_JOB_ID}-py313"
fi
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
        PAIR_COUNT="${PAIR_COUNT:-1}"
        MPOL="${MPOL:-10}"
        NTOR="${NTOR:-10}"
        NPHI="${NPHI:-255}"
        NTHETA="${NTHETA:-64}"
        OUTER_MAXITER="${OUTER_MAXITER:-1500}"
        OUTER_FTOL="${OUTER_FTOL:-1e-12}"
        OUTER_GTOL="${OUTER_GTOL:-1e-12}"
        BOOZER_BFGS_MAXITER="${BOOZER_BFGS_MAXITER:-1500}"
        BOOZER_NEWTON_MAXITER="${BOOZER_NEWTON_MAXITER:-50}"
        ;;
    *)
        echo "PROFILE must be pilot or full; got ${PROFILE}" >&2
        exit 2
        ;;
esac
if [[ -n "${PREPARED_VENV_DIR}" && "${PROFILE}" != "full" ]]; then
    echo "PREPARED_VENV_DIR is supported only for PROFILE=full" >&2
    exit 2
fi

validate_pair_count() {
    local profile="$1"
    local pair_count="$2"
    if [[ ! "${pair_count}" =~ ^[1-9][0-9]*$ ]]; then
        echo "PAIR_COUNT must be a positive integer; got ${pair_count}" >&2
        return 2
    fi
}

validate_pair_execution_mode() {
    local execution_mode="$2"
    case "${execution_mode}" in
        slurm-step)
            ;;
        direct)
            echo \
                "Direct pair execution cannot satisfy the three-node benchmark topology" \
                >&2
            return 2
            ;;
        *)
            echo \
                "PAIR_EXECUTION_MODE must be slurm-step; got ${execution_mode}" \
                >&2
            return 2
            ;;
    esac
}

validate_allocated_nodes() {
    if ((${#CPU_ALLOCATION_NODES[@]} != 2)); then
        echo \
            "Heterogeneous benchmark requires exactly 2 CPU nodes; got ${#CPU_ALLOCATION_NODES[@]}" \
            >&2
        return 2
    fi
    if ((${#GPU_ALLOCATION_NODES[@]} != 1)); then
        echo \
            "Heterogeneous benchmark requires exactly 1 GPU node; got ${#GPU_ALLOCATION_NODES[@]}" \
            >&2
        return 2
    fi
    if [[ "${CPU_ALLOCATION_NODES[0]}" == "${CPU_ALLOCATION_NODES[1]}" \
        || "${CPU_ALLOCATION_NODES[0]}" == "${GPU_ALLOCATION_NODES[0]}" \
        || "${CPU_ALLOCATION_NODES[1]}" == "${GPU_ALLOCATION_NODES[0]}" ]]; then
        echo "Three-node benchmark requires 3 distinct allocated nodes" >&2
        printf 'allocated_node=%s\n' \
            "${CPU_ALLOCATION_NODES[@]}" "${GPU_ALLOCATION_NODES[@]}" >&2
        return 2
    fi
}

discover_allocation_topology() {
    if [[ "${SLURM_HET_SIZE:-}" != "2" ]]; then
        echo \
            "Benchmark requires a 2-component heterogeneous allocation; got SLURM_HET_SIZE=${SLURM_HET_SIZE:-unset}" \
            >&2
        return 2
    fi
    if [[ -z "${SLURM_JOB_NODELIST_HET_GROUP_0:-}" \
        || -z "${SLURM_JOB_NODELIST_HET_GROUP_1:-}" ]]; then
        echo "Heterogeneous component node lists are required" >&2
        return 2
    fi
    if [[ -z "${SLURM_JOB_ID_HET_GROUP_0:-}" \
        || -z "${SLURM_JOB_ID_HET_GROUP_1:-}" \
        || "${SLURM_JOB_ID:?}" != "${SLURM_JOB_ID_HET_GROUP_0}" ]]; then
        echo "SLURM_JOB_ID must be the heterogeneous group-0 leader ID" >&2
        return 2
    fi

    CPU_ALLOCATION_NODES=()
    GPU_ALLOCATION_NODES=()
    mapfile -t CPU_ALLOCATION_NODES \
        < <(scontrol show hostnames "${SLURM_JOB_NODELIST_HET_GROUP_0}")
    mapfile -t GPU_ALLOCATION_NODES \
        < <(scontrol show hostnames "${SLURM_JOB_NODELIST_HET_GROUP_1}")
    if ! validate_allocated_nodes; then
        return 2
    fi

    CPU_COMPONENT_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK_HET_GROUP_0:-}"
    GPU_COMPONENT_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK_HET_GROUP_1:-}"
    CPU_LANE_CPUS_PER_TASK="${CPU_LANE_CPUS_PER_TASK:-64}"
    JAX_GPU_CPUS_PER_TASK="${JAX_GPU_CPUS_PER_TASK:-${GPU_COMPONENT_CPUS_PER_TASK}}"
    if [[ ! "${CPU_COMPONENT_CPUS_PER_TASK}" =~ ^[1-9][0-9]*$ \
        || ! "${GPU_COMPONENT_CPUS_PER_TASK}" =~ ^[1-9][0-9]*$ \
        || ! "${CPU_LANE_CPUS_PER_TASK}" =~ ^[1-9][0-9]*$ \
        || ! "${JAX_GPU_CPUS_PER_TASK}" =~ ^[1-9][0-9]*$ ]]; then
        echo "Heterogeneous component CPU counts must be positive integers" >&2
        return 2
    fi
    if [[ "${CPU_LANE_CPUS_PER_TASK}" -gt "${CPU_COMPONENT_CPUS_PER_TASK}" \
        || "${JAX_GPU_CPUS_PER_TASK}" -gt "${GPU_COMPONENT_CPUS_PER_TASK}" ]]; then
        echo "Lane CPU counts must not exceed their component allocations" >&2
        return 2
    fi
    # The shared GPU component exposes two hardware threads per physical core.
    JAX_GPU_OMP_NUM_THREADS="${JAX_GPU_OMP_NUM_THREADS:-16}"
    if [[ ! "${JAX_GPU_OMP_NUM_THREADS}" =~ ^[1-9][0-9]*$ \
        || "${JAX_GPU_OMP_NUM_THREADS}" -gt "${JAX_GPU_CPUS_PER_TASK}" ]]; then
        echo \
            "JAX_GPU_OMP_NUM_THREADS must be a positive integer no greater than the GPU step CPU count" \
            >&2
        return 2
    fi

    ALLOCATION_LAYOUT="perlmutter-het-cpu2-gpu1"
    LANE_ASSIGNMENT_POLICY="concurrent typed nodes; CPU roles swap by pair index and JAX GPU remains on the GPU node"
    ALLOCATED_NODES=(
        "${CPU_ALLOCATION_NODES[@]}"
        "${GPU_ALLOCATION_NODES[@]}"
    )
    CPU_HET_ARGS=(--het-group=0)
    GPU_HET_ARGS=(--het-group=1)
}

assign_pair_nodes() {
    local pair_index="$1"
    local rotation_index=$(((pair_index - 1) % 2))
    native_cpu_node="${CPU_ALLOCATION_NODES[rotation_index]}"
    jax_cpu_node="${CPU_ALLOCATION_NODES[(rotation_index + 1) % 2]}"
    jax_gpu_node="${GPU_ALLOCATION_NODES[0]}"
}

launch_pair_lanes() {
    local pair_dir="$1"
    local native_cpu_node="$2"
    local jax_cpu_node="$3"
    local jax_gpu_node="$4"
    local -a common_step_arguments=(
        --nodes=1
        --ntasks=1
        --cpu-bind=cores
        --exact
        --mpi=none
    )
    local -a cpu_step=(
        srun
        "${CPU_HET_ARGS[@]}"
        "${common_step_arguments[@]}"
        --cpus-per-task="${CPU_LANE_CPUS_PER_TASK}"
        --hint=nomultithread
    )
    local -a gpu_step=(
        srun
        "${GPU_HET_ARGS[@]}"
        "${common_step_arguments[@]}"
        --cpus-per-task="${JAX_GPU_CPUS_PER_TASK}"
    )
    local -a lane_wrapper=(
        bash
        "${LAUNCHER_PATH}"
        __run-lane-step
    )

    "${cpu_step[@]}" --nodelist="${native_cpu_node}" --gres=none \
        --output="${pair_dir}/native_cpu_slurm_stdout.log" \
        --error="${pair_dir}/native_cpu_slurm_stderr.log" \
        env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS="${CPU_LANE_CPUS_PER_TASK}" \
        "${lane_wrapper[@]}" native_cpu "${native_cpu_node}" "${pair_dir}" \
        "${VENV_DIR}/bin/python" "${COMPARATOR_PATH}" \
        "${GPU_PROCESS_MEMORY_CSV}" "${GPU_SAMPLER_QUERY_CSV}" \
        "${GPU_SAMPLER_ERROR_LOG}" "${CPU_LANE_CPUS_PER_TASK}" \
        "${CPU_LANE_CPUS_PER_TASK}" &
    NATIVE_CPU_STEP_PID=$!

    "${cpu_step[@]}" --nodelist="${jax_cpu_node}" --gres=none \
        --output="${pair_dir}/jax_cpu_slurm_stdout.log" \
        --error="${pair_dir}/jax_cpu_slurm_stderr.log" \
        env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS="${CPU_LANE_CPUS_PER_TASK}" \
        "${lane_wrapper[@]}" jax_cpu "${jax_cpu_node}" "${pair_dir}" \
        "${VENV_DIR}/bin/python" "${COMPARATOR_PATH}" \
        "${GPU_PROCESS_MEMORY_CSV}" "${GPU_SAMPLER_QUERY_CSV}" \
        "${GPU_SAMPLER_ERROR_LOG}" "${CPU_LANE_CPUS_PER_TASK}" \
        "${CPU_LANE_CPUS_PER_TASK}" &
    JAX_CPU_STEP_PID=$!

    "${gpu_step[@]}" --nodelist="${jax_gpu_node}" --gpus-per-task=1 \
        --gpu-bind=single:1 \
        --output="${pair_dir}/jax_gpu_slurm_stdout.log" \
        --error="${pair_dir}/jax_gpu_slurm_stderr.log" \
        env OMP_NUM_THREADS="${JAX_GPU_OMP_NUM_THREADS}" \
        "${lane_wrapper[@]}" jax_gpu "${jax_gpu_node}" "${pair_dir}" \
        "${VENV_DIR}/bin/python" "${COMPARATOR_PATH}" \
        "${GPU_PROCESS_MEMORY_CSV}" "${GPU_SAMPLER_QUERY_CSV}" \
        "${GPU_SAMPLER_ERROR_LOG}" "${JAX_GPU_CPUS_PER_TASK}" \
        "${JAX_GPU_OMP_NUM_THREADS}" &
    JAX_GPU_STEP_PID=$!
}

wait_for_pair_lanes() {
    if wait "${NATIVE_CPU_STEP_PID}"; then
        native_cpu_returncode=0
    else
        native_cpu_returncode=$?
    fi
    if wait "${JAX_CPU_STEP_PID}"; then
        jax_cpu_returncode=0
    else
        jax_cpu_returncode=$?
    fi
    if wait "${JAX_GPU_STEP_PID}"; then
        jax_gpu_returncode=0
    else
        jax_gpu_returncode=$?
    fi
}

validate_pair_count "${PROFILE}" "${PAIR_COUNT}"
validate_pair_execution_mode "${PROFILE}" "${PAIR_EXECUTION_MODE}"

CONSTRAINT_WEIGHT="${CONSTRAINT_WEIGHT:-1.0}"
OUTER_MAXCOR="${OUTER_MAXCOR:-300}"
OUTER_MAXLS="${OUTER_MAXLS:-20}"
BOOZER_BFGS_TOL="${BOOZER_BFGS_TOL:-1e-10}"
BOOZER_NEWTON_TOL="${BOOZER_NEWTON_TOL:-1e-11}"

if [[ -n "${PREPARED_VENV_DIR}" ]]; then
    validate_prepared_venv "${VENV_DIR}"
    exec {PREPARED_VENV_LOCK_FD}>"${VENV_DIR}.lock"
    if ! flock -n "${PREPARED_VENV_LOCK_FD}"; then
        echo "PREPARED_VENV_DIR is already in use: ${VENV_DIR}" >&2
        exit 2
    fi
else
    prepare_private_venv_root "${VENV_ROOT}"
    VENV_ROOT="$(realpath -- "${VENV_ROOT}")"
    if [[ "${VENV_ROOT}" != "${VENV_ROOT_REAL}" ]]; then
        echo "VENV_ROOT changed while it was being reserved" >&2
        exit 2
    fi
    VENV_DIR="${VENV_ROOT}/job-${SLURM_JOB_ID}-py313"
fi
reserve_fresh_directory "ARTIFACT_ROOT" "${ARTIFACT_ROOT}"
if [[ -z "${PREPARED_VENV_DIR}" ]]; then
    reserve_fresh_directory "VENV_DIR" "${VENV_DIR}" 700
fi

MACHINE_ROOT="${ARTIFACT_ROOT}/machine"
INPUT_ROOT="${ARTIFACT_ROOT}/inputs"
PAIR_ROOT="${ARTIFACT_ROOT}/pairs"
ENVIRONMENT_ROOT="${ARTIFACT_ROOT}/environment"
mkdir -p "${MACHINE_ROOT}" "${INPUT_ROOT}" "${PAIR_ROOT}" \
    "${ENVIRONMENT_ROOT}"

GPU_PROCESS_MEMORY_CSV="${MACHINE_ROOT}/gpu_process_memory.csv"
GPU_SAMPLER_QUERY_CSV="${MACHINE_ROOT}/gpu_sampler_queries.csv"
GPU_SAMPLER_ERROR_LOG="${MACHINE_ROOT}/gpu_sampler_errors.log"
GPU_PROCESS_MEMORY_SUMMARY="${MACHINE_ROOT}/gpu_process_memory_summary.json"
ENVIRONMENT_LOCK_MANIFEST="${ENVIRONMENT_ROOT}/manifest.json"
ENVIRONMENT_LOCK_SNAPSHOT="${ENVIRONMENT_ROOT}/requirements.lock"
COMPARATOR_PATH="${REPO_ROOT}/benchmarks/single_stage_full_loop_compare.py"

ALLOCATED_NODES=()
CPU_ALLOCATION_NODES=()
GPU_ALLOCATION_NODES=()
CPU_HET_ARGS=()
GPU_HET_ARGS=()
discover_allocation_topology
printf '%s\n' "${ALLOCATED_NODES[@]}" > "${MACHINE_ROOT}/allocated_nodes.txt"
printf '%s\n' "${CPU_ALLOCATION_NODES[@]}" \
    > "${MACHINE_ROOT}/cpu_allocated_nodes.txt"
printf '%s\n' "${GPU_ALLOCATION_NODES[@]}" \
    > "${MACHINE_ROOT}/gpu_allocated_nodes.txt"
{
    printf 'allocation_layout\t%s\n' "${ALLOCATION_LAYOUT}"
    printf 'leader_job_id\t%s\n' "${SLURM_JOB_ID}"
    printf 'cpu_component_job_id\t%s\n' "${SLURM_JOB_ID_HET_GROUP_0}"
    printf 'gpu_component_job_id\t%s\n' "${SLURM_JOB_ID_HET_GROUP_1}"
    printf 'cpu_component_account\t%s\n' "${SLURM_JOB_ACCOUNT_HET_GROUP_0:-}"
    printf 'gpu_component_account\t%s\n' "${SLURM_JOB_ACCOUNT_HET_GROUP_1:-}"
    printf 'cpu_component_qos\t%s\n' "${SLURM_JOB_QOS_HET_GROUP_0:-}"
    printf 'gpu_component_qos\t%s\n' "${SLURM_JOB_QOS_HET_GROUP_1:-}"
    printf 'cpu_component_nodelist\t%s\n' \
        "${SLURM_JOB_NODELIST_HET_GROUP_0}"
    printf 'gpu_component_nodelist\t%s\n' \
        "${SLURM_JOB_NODELIST_HET_GROUP_1}"
    printf 'cpu_component_cpus_per_task\t%s\n' \
        "${CPU_COMPONENT_CPUS_PER_TASK}"
    printf 'gpu_component_cpus_per_task\t%s\n' \
        "${GPU_COMPONENT_CPUS_PER_TASK}"
    printf 'cpu_lane_cpus_per_task\t%s\n' "${CPU_LANE_CPUS_PER_TASK}"
    printf 'jax_gpu_cpus_per_task\t%s\n' "${JAX_GPU_CPUS_PER_TASK}"
    printf 'jax_gpu_omp_num_threads\t%s\n' "${JAX_GPU_OMP_NUM_THREADS}"
    printf 'gpu_component_allocation\t%s\n' \
        "${SLURM_JOB_GPUS_HET_GROUP_1:-}"
    printf 'lane_assignment_policy\t%s\n' "${LANE_ASSIGNMENT_POLICY}"
} > "${MACHINE_ROOT}/slurm_allocation.tsv"

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
        "sampled_nodes": set(),
        "process_sample_count": 0,
        "process_ids": set(),
        "values": [],
        "jax_gpu_process_sample_count": 0,
        "jax_gpu_process_ids": set(),
        "jax_gpu_values": [],
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
                pair_summary["sampled_nodes"].add(row["node"])

jax_gpu_windows: dict[str, tuple[datetime, datetime]] = {}
for pair_name in expected_pair_names:
    execution_path = pair_root / pair_name / "jax_gpu" / "execution.json"
    if not execution_path.is_file():
        continue
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    started_at_utc = execution.get("started_at_utc")
    ended_at_utc = execution.get("ended_at_utc")
    if (
        execution.get("status") != "passed"
        or not isinstance(started_at_utc, str)
        or not isinstance(ended_at_utc, str)
    ):
        continue
    jax_gpu_windows[pair_name] = (
        parse_timestamp(started_at_utc),
        parse_timestamp(ended_at_utc),
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
            pair_summary["sampled_nodes"].add(row["node"])
            pair_summary["process_sample_count"] += 1
            pair_summary["process_ids"].add(pid)
            pair_summary["values"].append(used_memory_mib)
            if pair in jax_gpu_windows:
                started_at, ended_at = jax_gpu_windows[pair]
                if started_at <= sampled_at <= ended_at:
                    pair_summary["jax_gpu_process_sample_count"] += 1
                    pair_summary["jax_gpu_process_ids"].add(pid)
                    pair_summary["jax_gpu_values"].append(used_memory_mib)
            overall_pids.add(pid)
            overall_values.append(used_memory_mib)

serialized_by_pair = {
    pair: {
        "query_count": values["query_count"],
        "query_failure_count": values["query_failure_count"],
        "sampled_nodes": sorted(values["sampled_nodes"]),
        "process_sample_count": values["process_sample_count"],
        "process_ids": sorted(values["process_ids"]),
        "maximum_used_memory_mib": (
            max(values["values"]) if values["values"] else None
        ),
        "jax_gpu_execution_window_recorded": pair in jax_gpu_windows,
        "jax_gpu_process_sample_count": values["jax_gpu_process_sample_count"],
        "jax_gpu_process_ids": sorted(values["jax_gpu_process_ids"]),
        "maximum_jax_gpu_used_memory_mib": (
            max(values["jax_gpu_values"]) if values["jax_gpu_values"] else None
        ),
    }
    for pair, values in sorted(by_pair.items())
}
all_queries_succeeded = query_count > 0 and query_failure_count == 0
every_pair_has_jax_gpu_process_sample = all(
    int(serialized_by_pair[pair_name]["jax_gpu_process_sample_count"]) > 0
    for pair_name in expected_pair_names
)
production_gate_passed = (
    all_queries_succeeded and every_pair_has_jax_gpu_process_sample
)
payload = {
    "schema_version": 2,
    "metric": "nvidia-smi compute-process used_memory",
    "unit": "MiB",
    "sampling_interval_seconds": 1,
    "scope": (
        "all compute processes on the GPU assigned to each JAX-GPU lane step"
        if profile == "full"
        else "diagnostic samples from compute processes on the GPU assigned "
        "to each JAX-GPU lane step"
    ),
    "note": (
        "JAX-GPU samples are compute-process rows whose timestamps fall inside "
        "the recorded JAX-GPU lane execution window. Every nvidia-smi query is "
        "restricted to the lane's Slurm-assigned global GPU ID. GPU process "
        "memory is separate from GNU-time host RSS."
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
        "every_pair_has_jax_gpu_process_sample": every_pair_has_jax_gpu_process_sample,
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
        ALLOCATION_LAYOUT="${ALLOCATION_LAYOUT}" \
        LANE_ASSIGNMENT_POLICY="${LANE_ASSIGNMENT_POLICY}" \
        CPU_COMPONENT_CPUS_PER_TASK="${CPU_COMPONENT_CPUS_PER_TASK}" \
        GPU_COMPONENT_CPUS_PER_TASK="${GPU_COMPONENT_CPUS_PER_TASK}" \
        CPU_LANE_CPUS_PER_TASK="${CPU_LANE_CPUS_PER_TASK}" \
        JAX_GPU_CPUS_PER_TASK="${JAX_GPU_CPUS_PER_TASK}" \
        JAX_GPU_OMP_NUM_THREADS="${JAX_GPU_OMP_NUM_THREADS}" \
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
allocation_layout = os.environ["ALLOCATION_LAYOUT"]
lane_assignment_policy = os.environ["LANE_ASSIGNMENT_POLICY"]
cpu_component_cpus_per_task = int(os.environ["CPU_COMPONENT_CPUS_PER_TASK"])
gpu_component_cpus_per_task = int(os.environ["GPU_COMPONENT_CPUS_PER_TASK"])
cpu_lane_cpus_per_task = int(os.environ["CPU_LANE_CPUS_PER_TASK"])
jax_gpu_cpus_per_task = int(os.environ["JAX_GPU_CPUS_PER_TASK"])
jax_gpu_omp_num_threads = int(os.environ["JAX_GPU_OMP_NUM_THREADS"])
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
        "every_pair_has_jax_gpu_process_sample": False,
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
successful_native_cpu_over_jax_cpu_speedups: list[float] = []
successful_native_cpu_over_jax_gpu_speedups: list[float] = []
successful_jax_cpu_over_jax_gpu_speedups: list[float] = []
successful_native_cpu_wall_seconds: list[float] = []
successful_jax_cpu_wall_seconds: list[float] = []
successful_jax_gpu_wall_seconds: list[float] = []
successful_native_cpu_host_rss_kib: list[int] = []
successful_jax_cpu_host_rss_kib: list[int] = []
successful_jax_gpu_host_rss_kib: list[int] = []

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
    pair_payload: dict[str, object] = {
        "pair_index": pair_index,
        "pair_name": pair_name,
        "order": status_values.get("order", "concurrent"),
        "pair_execution_mode": status_values.get("pair_execution_mode"),
        "allocation_layout": status_values.get("allocation_layout"),
        "lane_assignment_policy": status_values.get("lane_assignment_policy"),
        "status": status,
        "nodes": {
            lane: status_values.get(f"{lane}_node")
            for lane in ("native_cpu", "jax_cpu", "jax_gpu")
        },
        "lane_returncodes": {
            lane: (
                int(status_values[f"{lane}_returncode"])
                if f"{lane}_returncode" in status_values
                else None
            )
            for lane in ("native_cpu", "jax_cpu", "jax_gpu")
        },
        "lane_statuses": {
            lane: status_values.get(f"{lane}_status")
            for lane in ("native_cpu", "jax_cpu", "jax_gpu")
        },
        "prepare_returncode": (
            int(status_values["prepare_returncode"])
            if "prepare_returncode" in status_values
            else None
        ),
        "adjudicate_returncode": (
            int(status_values["adjudicate_returncode"])
            if "adjudicate_returncode" in status_values
            else None
        ),
        "comparator_exit_code": (
            int(status_values["comparator_exit_code"])
            if "comparator_exit_code" in status_values
            else None
        ),
        "execution_timestamps_complete": (
            status_values.get("execution_timestamps_complete") == "true"
        ),
        "all_lanes_overlapped": status_values.get("all_lanes_overlapped") == "true",
        "overlap_seconds": (
            float(status_values["overlap_seconds"])
            if "overlap_seconds" in status_values
            else None
        ),
        "comparison_status": status_values.get("comparison_status"),
        "comparison_claim_ready": (
            status_values.get("comparison_claim_ready") == "true"
            if "comparison_claim_ready" in status_values
            else None
        ),
        "optimizer_success": {
            lane: (
                status_values.get(f"{lane}_optimizer_success") == "true"
                if f"{lane}_optimizer_success" in status_values
                else None
            )
            for lane in ("native_cpu", "jax_cpu", "jax_gpu")
        },
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
                "sampled_nodes": [],
                "process_sample_count": 0,
                "process_ids": [],
                "maximum_used_memory_mib": None,
                "jax_gpu_execution_window_recorded": False,
                "jax_gpu_process_sample_count": 0,
                "jax_gpu_process_ids": [],
                "maximum_jax_gpu_used_memory_mib": None,
            },
        )
    if comparison is not None:
        performance = comparison["performance"]
        parity = comparison["parity"]
        pair_payload["performance"] = performance
        pair_payload["failures"] = comparison["failures"]
        pair_payload["required_failures"] = parity["required_failures"]
        pair_payload["advisory_failures"] = parity["advisory_failures"]
        pair_payload["host_rss_kib"] = (
            performance["host_max_rss_kib"] if performance is not None else None
        )
        if status == "passed" and performance is not None:
            speedups = performance["speedups"]
            wall_seconds = performance["wall_seconds"]
            host_max_rss_kib = performance["host_max_rss_kib"]
            successful_native_cpu_over_jax_cpu_speedups.append(
                float(speedups["native_cpu_over_jax_cpu"])
            )
            successful_native_cpu_over_jax_gpu_speedups.append(
                float(speedups["native_cpu_over_jax_gpu"])
            )
            successful_jax_cpu_over_jax_gpu_speedups.append(
                float(speedups["jax_cpu_over_jax_gpu"])
            )
            successful_native_cpu_wall_seconds.append(
                float(wall_seconds["native_cpu"])
            )
            successful_jax_cpu_wall_seconds.append(float(wall_seconds["jax_cpu"]))
            successful_jax_gpu_wall_seconds.append(float(wall_seconds["jax_gpu"]))
            successful_native_cpu_host_rss_kib.append(
                int(host_max_rss_kib["native_cpu"])
            )
            successful_jax_cpu_host_rss_kib.append(
                int(host_max_rss_kib["jax_cpu"])
            )
            successful_jax_gpu_host_rss_kib.append(
                int(host_max_rss_kib["jax_gpu"])
            )
    pairs.append(pair_payload)

successful_pair_count = len(successful_native_cpu_over_jax_gpu_speedups)
measurement_design_valid = expected_pair_count >= 1
repeat_design_valid = expected_pair_count >= 2
pair_execution_modes_match = all(
    pair["pair_execution_mode"] == pair_execution_mode for pair in pairs
)
pair_allocation_layouts_match = all(
    pair["allocation_layout"] == allocation_layout for pair in pairs
)
pair_lane_assignment_policies_match = all(
    pair["lane_assignment_policy"] == lane_assignment_policy for pair in pairs
)
pair_topologies_valid = all(
    pair["all_lanes_overlapped"] is True
    and len(set(pair["nodes"].values())) == 3
    and None not in pair["nodes"].values()
    for pair in pairs
)
all_pairs_passed = (
    measurement_design_valid
    and pair_execution_modes_match
    and pair_allocation_layouts_match
    and pair_lane_assignment_policies_match
    and pair_topologies_valid
    and successful_pair_count == expected_pair_count
)
production_evidence_gates_passed = bool(
    gpu_memory_gate["passed"] and dependency_lock_gate_passed
)
aggregate_claim_ready = bool(
    profile == "full" and all_pairs_passed and production_evidence_gates_passed
)
def distribution(values: list[float] | list[int]) -> dict[str, float | int | None]:
    return {
        "median": statistics.median(values) if values else None,
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
    }


speedup_summary = {
    "successful_pair_count": successful_pair_count,
    "native_cpu_over_jax_cpu": distribution(
        successful_native_cpu_over_jax_cpu_speedups
    ),
    "native_cpu_over_jax_gpu": distribution(
        successful_native_cpu_over_jax_gpu_speedups
    ),
    "jax_cpu_over_jax_gpu": distribution(successful_jax_cpu_over_jax_gpu_speedups),
}


payload = {
    "schema_version": 2,
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
    "allocation_layout": allocation_layout,
    "claim_ready": aggregate_claim_ready,
    "expected_pair_count": expected_pair_count,
    "successful_pair_count": successful_pair_count,
    "failed_pair_count": expected_pair_count - successful_pair_count,
    "measurement_design_valid": measurement_design_valid,
    "repeat_design_valid": repeat_design_valid,
    "pair_execution_modes_match": pair_execution_modes_match,
    "pair_allocation_layouts_match": pair_allocation_layouts_match,
    "pair_lane_assignment_policies_match": pair_lane_assignment_policies_match,
    "pair_topologies_valid": pair_topologies_valid,
    "measurement_class": (
        "single_matched_triplet"
        if expected_pair_count == 1
        else "repeated_cpu-swapped_triplets"
    ),
    "measurement_interpretation": (
        "one matched measurement; no statistical inference"
        if expected_pair_count == 1
        else "repeated matched measurements with deterministic CPU-role swap "
        "and a fixed GPU node"
    ),
    "lane_assignment_policy": lane_assignment_policy,
    "resources": {
        "cpu_component_cpus_per_task": cpu_component_cpus_per_task,
        "gpu_component_cpus_per_task": gpu_component_cpus_per_task,
        "native_cpu": {
            "cpus_per_task": cpu_lane_cpus_per_task,
            "omp_num_threads": cpu_lane_cpus_per_task,
        },
        "jax_cpu": {
            "cpus_per_task": cpu_lane_cpus_per_task,
            "omp_num_threads": cpu_lane_cpus_per_task,
        },
        "jax_gpu": {
            "cpus_per_task": jax_gpu_cpus_per_task,
            "omp_num_threads": jax_gpu_omp_num_threads,
            "gpus_per_task": 1,
        },
    },
    "speedup": speedup_summary,
    "wall_seconds": {
        "native_cpu": distribution(successful_native_cpu_wall_seconds),
        "jax_cpu": distribution(successful_jax_cpu_wall_seconds),
        "jax_gpu": distribution(successful_jax_gpu_wall_seconds),
    },
    "host_max_rss_kib": {
        "native_cpu": distribution(successful_native_cpu_host_rss_kib),
        "jax_cpu": distribution(successful_jax_cpu_host_rss_kib),
        "jax_gpu": distribution(successful_jax_gpu_host_rss_kib),
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
    sacct -j "${SLURM_JOB_ID}" \
        --format=JobIDRaw,State,ExitCode,Elapsed,NodeList,NNodes,NTasks,MaxRSS,AllocTRES%80 -P \
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

SOURCE_SHA="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
SOURCE_TREE="$(git -C "${REPO_ROOT}" rev-parse 'HEAD^{tree}')"
printf '%s\n' "${SOURCE_SHA}" > "${MACHINE_ROOT}/git_commit.txt"
printf '%s\n' "${SOURCE_TREE}" > "${MACHINE_ROOT}/git_tree.txt"
git -C "${REPO_ROOT}" status --porcelain=v1 --untracked-files=normal \
    > "${MACHINE_ROOT}/git_status.txt"
sed -E '/^\?\? slurm-[0-9]+\.(out|err)$/d' \
    "${MACHINE_ROOT}/git_status.txt" > "${MACHINE_ROOT}/git_source_status.txt"
if [[ -s "${MACHINE_ROOT}/git_source_status.txt" ]]; then
    echo "Refusing performance run from a dirty source tree" >&2
    cat "${MACHINE_ROOT}/git_source_status.txt" >&2
    exit 2
fi
resolve_source_scm_provenance "${REPO_ROOT}" "${SOURCE_SHA}"
printf '%s\n' "${SOURCE_SCM_DESCRIPTION}" \
    > "${MACHINE_ROOT}/source_scm_description.txt"

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

verify_installed_requirements_match_lock() {
    local installed_requirements="$1"
    "${VENV_DIR}/bin/python" -m pip freeze --all --exclude-editable \
        > "${installed_requirements}"
    ENVIRONMENT_LOCK_SNAPSHOT="${ENVIRONMENT_LOCK_SNAPSHOT}" \
        INSTALLED_REQUIREMENTS="${installed_requirements}" \
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
}

if [[ -z "${PREPARED_VENV_DIR}" ]]; then
    python -m venv "${VENV_DIR}"
fi
printf '%s\n' "${VENV_DIR}" > "${MACHINE_ROOT}/venv_path.txt"

if [[ "${PROFILE}" == "pilot" ]]; then
    "${VENV_DIR}/bin/python" -m pip install --upgrade \
        'pip==26.1.2' 'setuptools==83.0.0' 'wheel==0.47.0'
    "${VENV_DIR}/bin/python" -m pip install \
        'scikit-build-core' 'pybind11<3.0.0' 'setuptools_scm>=8.0' \
        'numpy>=2.0,<2.5'
    SOURCE_PACKAGE_VERSION="$(
        format_source_package_version "${VENV_DIR}/bin/python"
    )"
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SIMSOPT="${SOURCE_PACKAGE_VERSION}" \
        SETUPTOOLS_SCM_PRETEND_METADATA_FOR_SIMSOPT="${SOURCE_SCM_METADATA}" \
        CMAKE_BUILD_TYPE=Release CXXFLAGS='-O3' \
        "${VENV_DIR}/bin/python" -m pip install \
        --no-build-isolation \
        -e "${REPO_ROOT}[JAX_GPU]" \
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
    DEPENDENCY_INSTALL_CONTRACT="pilot-resolved environment"
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

    if [[ -z "${PREPARED_VENV_DIR}" ]]; then
        "${VENV_DIR}/bin/python" -m pip install \
            --require-hashes \
            --only-binary=:all: \
            --requirement "${ENVIRONMENT_LOCK_SNAPSHOT}"
        DEPENDENCY_INSTALL_CONTRACT="pip --require-hashes --only-binary=:all:"
        SECURE_INSTALL_VERIFIED="true"
    else
        DEPENDENCY_INSTALL_CONTRACT=\
"prepared dependency-only venv; exact lock revalidated before source install"
        SECURE_INSTALL_VERIFIED="false"
        verify_installed_requirements_match_lock \
            "${ENVIRONMENT_ROOT}/prepared_venv_requirements.txt"
    fi
    SOURCE_PACKAGE_VERSION="$(
        format_source_package_version "${VENV_DIR}/bin/python"
    )"
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SIMSOPT="${SOURCE_PACKAGE_VERSION}" \
        SETUPTOOLS_SCM_PRETEND_METADATA_FOR_SIMSOPT="${SOURCE_SCM_METADATA}" \
        CMAKE_BUILD_TYPE=Release CXXFLAGS='-O3' \
        "${VENV_DIR}/bin/python" -m pip install \
        --no-deps \
        --no-build-isolation \
        -e "${REPO_ROOT}"
    "${VENV_DIR}/bin/python" -m pip check
fi

printf '%s\n' "${SOURCE_PACKAGE_VERSION}" \
    > "${MACHINE_ROOT}/source_package_version.txt"
EXPECTED_SOURCE_PACKAGE_VERSION="${SOURCE_PACKAGE_VERSION}" \
    EXPECTED_SOURCE_COMMIT_ID="g${SOURCE_SHA:0:9}" \
    "${VENV_DIR}/bin/python" - <<'PY'
from __future__ import annotations

from importlib.metadata import version
import os

import simsopt
from simsopt import _version


expected_version = os.environ["EXPECTED_SOURCE_PACKAGE_VERSION"]
installed_version = version("simsopt")
if installed_version != expected_version:
    raise RuntimeError(
        "Installed SIMSOPT distribution version does not match source provenance: "
        f"expected={expected_version}, installed={installed_version}"
    )
if simsopt.__version__ != expected_version:
    raise RuntimeError(
        "Imported SIMSOPT version does not match source provenance: "
        f"expected={expected_version}, imported={simsopt.__version__}"
    )
expected_commit_id = os.environ["EXPECTED_SOURCE_COMMIT_ID"]
if _version.commit_id != expected_commit_id:
    raise RuntimeError(
        "Generated SIMSOPT commit ID does not match source provenance: "
        f"expected={expected_commit_id}, generated={_version.commit_id}"
    )
PY

verify_installed_requirements_match_lock \
    "${ENVIRONMENT_ROOT}/installed_requirements.txt"

PROFILE="${PROFILE}" \
    EVIDENCE_CLASS="${EVIDENCE_CLASS}" \
    PAIR_EXECUTION_MODE="${PAIR_EXECUTION_MODE}" \
    LOCK_ORIGIN="${LOCK_ORIGIN}" \
    DEPENDENCY_INSTALL_CONTRACT="${DEPENDENCY_INSTALL_CONTRACT}" \
    PREPARED_VENV_DIR="${PREPARED_VENV_DIR}" \
    ENVIRONMENT_LOCK_SNAPSHOT="${ENVIRONMENT_LOCK_SNAPSHOT}" \
    ENVIRONMENT_LOCK_SHA256="${ENVIRONMENT_LOCK_SHA256}" \
    SECURE_INSTALL_VERIFIED="${SECURE_INSTALL_VERIFIED}" \
    SOURCE_SHA="${SOURCE_SHA}" \
    SOURCE_TREE="${SOURCE_TREE}" \
    SOURCE_SCM_DESCRIPTION="${SOURCE_SCM_DESCRIPTION}" \
    SOURCE_SCM_TAG="${SOURCE_SCM_TAG}" \
    SOURCE_SCM_DISTANCE="${SOURCE_SCM_DISTANCE}" \
    SOURCE_SCM_NODE="${SOURCE_SCM_NODE}" \
    SOURCE_PACKAGE_VERSION="${SOURCE_PACKAGE_VERSION}" \
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
        "dependencies": os.environ["DEPENDENCY_INSTALL_CONTRACT"],
        "repository": "pip --no-deps --no-build-isolation -e",
        "repository_version_source": (
            "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SIMSOPT"
        ),
        "repository_metadata_source": (
            "SETUPTOOLS_SCM_PRETEND_METADATA_FOR_SIMSOPT"
        ),
    },
    "source_commit": os.environ["SOURCE_SHA"],
    "source_tree": os.environ["SOURCE_TREE"],
    "source_scm_description": os.environ["SOURCE_SCM_DESCRIPTION"],
    "source_scm_tag": os.environ["SOURCE_SCM_TAG"],
    "source_scm_distance": int(os.environ["SOURCE_SCM_DISTANCE"]),
    "source_scm_node": os.environ["SOURCE_SCM_NODE"],
    "source_package_version": os.environ["SOURCE_PACKAGE_VERSION"],
    "prepared_venv_dir": os.environ["PREPARED_VENV_DIR"] or None,
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
export OMP_NUM_THREADS="${CPU_LANE_CPUS_PER_TASK}"
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

printf '%s\n' "${PAIR_EXECUTION_MODE}" \
    > "${MACHINE_ROOT}/pair_execution_mode.txt"
printf '%s\n' "${ALLOCATION_LAYOUT}" \
    > "${MACHINE_ROOT}/allocation_layout.txt"
module list > "${MACHINE_ROOT}/modules.txt" 2>&1
hostname -s > "${MACHINE_ROOT}/batch_host.txt"
lscpu > "${MACHINE_ROOT}/batch_host_lscpu.txt"
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

printf 'timestamp_utc,pair,node,gpu_uuid,pid,used_memory_mib\n' \
    > "${GPU_PROCESS_MEMORY_CSV}"
printf 'timestamp_utc,pair,node,status,exit_code\n' > "${GPU_SAMPLER_QUERY_CSV}"
: > "${GPU_SAMPLER_ERROR_LOG}"

COMPARATOR_PREPARE_ARGS=(
    --python "${VENV_DIR}/bin/python"
    --surface-path "${SURFACE_SNAPSHOT}"
    --biotsavart-file "${BIOTSAVART_SNAPSHOT}"
    --boozer-state-path "${BOOZER_STATE_SNAPSHOT}"
    --mode "${EVIDENCE_CLASS}"
    --mpol "${MPOL}"
    --ntor "${NTOR}"
    --nphi "${NPHI}"
    --ntheta "${NTHETA}"
    --constraint-weight "${CONSTRAINT_WEIGHT}"
    --iota-target "${IOTA_TARGET}"
    --environment-lock-sha256 "${ENVIRONMENT_LOCK_SHA256}"
    --maxiter "${OUTER_MAXITER}"
    --outer-maxcor "${OUTER_MAXCOR}"
    --outer-maxls "${OUTER_MAXLS}"
    --outer-ftol "${OUTER_FTOL}"
    --outer-gtol "${OUTER_GTOL}"
    --boozer-bfgs-tol "${BOOZER_BFGS_TOL}"
    --boozer-bfgs-maxiter "${BOOZER_BFGS_MAXITER}"
    --boozer-newton-tol "${BOOZER_NEWTON_TOL}"
    --boozer-newton-maxiter "${BOOZER_NEWTON_MAXITER}"
)

FAILED_PAIRS=0
for ((pair_index = 1; pair_index <= PAIR_COUNT; pair_index++))
do
    pair_name="pair-$(printf '%02d' "${pair_index}")"
    pair_dir="${PAIR_ROOT}/${pair_name}"
    assign_pair_nodes "${pair_index}"
    printf '%s\n' "${pair_name}" > "${MACHINE_ROOT}/last_pair.txt"
    pair_started_at="$(date --utc +%Y-%m-%dT%H:%M:%S.%NZ)"
    echo \
        "Starting ${pair_name}/${PAIR_COUNT}: native_cpu=${native_cpu_node} jax_cpu=${jax_cpu_node} jax_gpu=${jax_gpu_node}" \
        >&2

    if "${VENV_DIR}/bin/python" "${COMPARATOR_PATH}" prepare \
        "${COMPARATOR_PREPARE_ARGS[@]}" \
        --output-root "${pair_dir}" \
        --native-cpu-node "${native_cpu_node}" \
        --jax-cpu-node "${jax_cpu_node}" \
        --jax-gpu-node "${jax_gpu_node}"
    then
        prepare_returncode=0
    else
        prepare_returncode=$?
    fi

    native_cpu_returncode=125
    jax_cpu_returncode=125
    jax_gpu_returncode=125
    if [[ "${prepare_returncode}" -eq 0 ]]; then
        launch_pair_lanes \
            "${pair_dir}" "${native_cpu_node}" "${jax_cpu_node}" "${jax_gpu_node}"
        wait_for_pair_lanes
    fi

    mkdir -p "${pair_dir}"
    if "${VENV_DIR}/bin/python" "${COMPARATOR_PATH}" adjudicate \
        --output-root "${pair_dir}" \
        --native-cpu-returncode "${native_cpu_returncode}" \
        --jax-cpu-returncode "${jax_cpu_returncode}" \
        --jax-gpu-returncode "${jax_gpu_returncode}"
    then
        adjudicate_returncode=0
    else
        adjudicate_returncode=$?
    fi
    pair_finished_at="$(date --utc +%Y-%m-%dT%H:%M:%S.%NZ)"

    comparison_status="missing"
    comparison_claim_ready="false"
    native_cpu_optimizer_success="false"
    jax_cpu_optimizer_success="false"
    jax_gpu_optimizer_success="false"
    if [[ -f "${pair_dir}/comparison.json" ]]; then
        read -r comparison_status comparison_claim_ready \
            native_cpu_optimizer_success jax_cpu_optimizer_success \
            jax_gpu_optimizer_success < <(
            PAIR_DIR="${pair_dir}" "${VENV_DIR}/bin/python" - <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path


pair_dir = Path(os.environ["PAIR_DIR"])
comparison = json.loads(
    (pair_dir / "comparison.json").read_text(encoding="utf-8")
)


def optimizer_success(lane: str) -> bool:
    results_path = pair_dir / lane / "results.json"
    if not results_path.is_file():
        return False
    results = json.loads(results_path.read_text(encoding="utf-8"))
    return bool(results["optimizer"]["success"])


print(
    comparison["status"],
    str(bool(comparison["claim_ready"])).lower(),
    *(str(optimizer_success(lane)).lower() for lane in (
        "native_cpu",
        "jax_cpu",
        "jax_gpu",
    )),
)
PY
        )
    fi

    read -r execution_timestamps_complete all_lanes_overlapped overlap_seconds < <(
        PAIR_DIR="${pair_dir}" "${VENV_DIR}/bin/python" - <<'PY'
from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path


pair_dir = Path(os.environ["PAIR_DIR"])
executions = []
for lane in ("native_cpu", "jax_cpu", "jax_gpu"):
    execution_path = pair_dir / lane / "execution.json"
    if not execution_path.is_file():
        break
    executions.append(json.loads(execution_path.read_text(encoding="utf-8")))

timestamps = [
    (item.get("started_at_utc"), item.get("ended_at_utc"))
    for item in executions
]
if len(executions) != 3 or not all(
    isinstance(started_at_utc, str) and isinstance(ended_at_utc, str)
    for started_at_utc, ended_at_utc in timestamps
):
    print("false false 0.0")
else:
    starts = [
        datetime.fromisoformat(started_at_utc)
        for started_at_utc, _ended_at_utc in timestamps
    ]
    ends = [
        datetime.fromisoformat(ended_at_utc)
        for _started_at_utc, ended_at_utc in timestamps
    ]
    overlap_seconds = (min(ends) - max(starts)).total_seconds()
    print("true", str(overlap_seconds > 0.0).lower(), max(0.0, overlap_seconds))
PY
    )

    adjudicated_status="failed"
    if [[ "${prepare_returncode}" -eq 0 \
        && "${native_cpu_returncode}" -eq 0 \
        && "${jax_cpu_returncode}" -eq 0 \
        && "${jax_gpu_returncode}" -eq 0 \
        && "${adjudicate_returncode}" -eq 0 \
        && "${all_lanes_overlapped}" == "true" ]]; then
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

    native_cpu_status="failed"
    jax_cpu_status="failed"
    jax_gpu_status="failed"
    if [[ "${native_cpu_returncode}" -eq 0 ]]; then
        native_cpu_status="passed"
    fi
    if [[ "${jax_cpu_returncode}" -eq 0 ]]; then
        jax_cpu_status="passed"
    fi
    if [[ "${jax_gpu_returncode}" -eq 0 ]]; then
        jax_gpu_status="passed"
    fi

    {
        printf 'pair_index\t%s\n' "${pair_index}"
        printf 'pair_name\t%s\n' "${pair_name}"
        printf 'order\tconcurrent\n'
        printf 'profile\t%s\n' "${PROFILE}"
        printf 'evidence_class\t%s\n' "${EVIDENCE_CLASS}"
        printf 'pair_execution_mode\t%s\n' "${PAIR_EXECUTION_MODE}"
        printf 'allocation_layout\t%s\n' "${ALLOCATION_LAYOUT}"
        printf 'lane_assignment_policy\t%s\n' "${LANE_ASSIGNMENT_POLICY}"
        printf 'native_cpu_node\t%s\n' "${native_cpu_node}"
        printf 'jax_cpu_node\t%s\n' "${jax_cpu_node}"
        printf 'jax_gpu_node\t%s\n' "${jax_gpu_node}"
        printf 'started_at_utc\t%s\n' "${pair_started_at}"
        printf 'finished_at_utc\t%s\n' "${pair_finished_at}"
        printf 'prepare_returncode\t%s\n' "${prepare_returncode}"
        printf 'native_cpu_returncode\t%s\n' "${native_cpu_returncode}"
        printf 'jax_cpu_returncode\t%s\n' "${jax_cpu_returncode}"
        printf 'jax_gpu_returncode\t%s\n' "${jax_gpu_returncode}"
        printf 'adjudicate_returncode\t%s\n' "${adjudicate_returncode}"
        printf 'comparator_exit_code\t%s\n' "${adjudicate_returncode}"
        printf 'native_cpu_status\t%s\n' "${native_cpu_status}"
        printf 'jax_cpu_status\t%s\n' "${jax_cpu_status}"
        printf 'jax_gpu_status\t%s\n' "${jax_gpu_status}"
        printf 'execution_timestamps_complete\t%s\n' "${execution_timestamps_complete}"
        printf 'all_lanes_overlapped\t%s\n' "${all_lanes_overlapped}"
        printf 'overlap_seconds\t%s\n' "${overlap_seconds}"
        printf 'comparison_status\t%s\n' "${comparison_status}"
        printf 'comparison_claim_ready\t%s\n' "${comparison_claim_ready}"
        printf 'native_cpu_optimizer_success\t%s\n' "${native_cpu_optimizer_success}"
        printf 'jax_cpu_optimizer_success\t%s\n' "${jax_cpu_optimizer_success}"
        printf 'jax_gpu_optimizer_success\t%s\n' "${jax_gpu_optimizer_success}"
        printf 'adjudicated_status\t%s\n' "${adjudicated_status}"
    } > "${pair_dir}/pair_status.tsv"
    echo "Finished ${pair_name}: ${adjudicated_status}" >&2
done

echo "Full-loop artifact root: ${ARTIFACT_ROOT}"
if ((FAILED_PAIRS > 0)); then
    echo "${FAILED_PAIRS}/${PAIR_COUNT} pairs failed adjudication" >&2
    exit 1
fi
