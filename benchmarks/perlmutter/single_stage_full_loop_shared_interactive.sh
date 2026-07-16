#!/bin/bash -l

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(realpath -- "${SCRIPT_DIR}/../..")}"
BENCHMARK_LAUNCHER="${REPO_ROOT}/benchmarks/perlmutter/single_stage_full_loop_cpu_gpu.slurm"

CPU_ACCOUNT="${CPU_ACCOUNT:-m4680}"
GPU_ACCOUNT="${GPU_ACCOUNT:-m4680_g}"
INTERACTIVE_WALLTIME="${INTERACTIVE_WALLTIME:-04:00:00}"
INTERACTIVE_WAIT_SECONDS="${INTERACTIVE_WAIT_SECONDS:-600}"
INTERACTIVE_JOB_NAME="${INTERACTIVE_JOB_NAME:-banana-full-loop-interactive}"

if [[ ! -r "${BENCHMARK_LAUNCHER}" ]]; then
    echo "Benchmark launcher is not readable: ${BENCHMARK_LAUNCHER}" >&2
    exit 2
fi
if ! command -v salloc >/dev/null; then
    echo "salloc is required to launch the Perlmutter interactive benchmark" >&2
    exit 2
fi

export REPO_ROOT BENCHMARK_LAUNCHER

# With Perlmutter's use_interactive_step setting, the no-command salloc shell
# runs on group 0 without consuming resources needed by the three sibling steps.
# The shared GPU slice keeps SMT enabled, so 32 logical CPUs are 16 physical cores.
salloc \
    --job-name="${INTERACTIVE_JOB_NAME}" \
    --immediate="${INTERACTIVE_WAIT_SECONDS}" \
    --account="${CPU_ACCOUNT}" \
    --qos=interactive \
    --time="${INTERACTIVE_WALLTIME}" \
    --constraint=cpu \
    --nodes=2 \
    --ntasks=2 \
    --ntasks-per-node=1 \
    --cpus-per-task=128 \
    --hint=nomultithread \
    : \
    --account="${GPU_ACCOUNT}" \
    --qos=shared_interactive \
    --time="${INTERACTIVE_WALLTIME}" \
    --constraint=gpu \
    --nodes=1 \
    --ntasks=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=32 \
    --gpus-per-task=1 \
    <<'SALLOC_COMMANDS'
exec bash -l -- "${BENCHMARK_LAUNCHER}"
SALLOC_COMMANDS
