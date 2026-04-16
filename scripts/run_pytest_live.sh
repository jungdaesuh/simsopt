#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${CONDA_ENV_NAME:-columbia-jax-0.9.2}"
ARTIFACT_DIR="${ROOT_DIR}/.artifacts/pytest"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="${ARTIFACT_DIR}/full_${TIMESTAMP}.log"
JUNIT_PATH="${ARTIFACT_DIR}/junit_${TIMESTAMP}.xml"

mkdir -p "${ARTIFACT_DIR}"

if [ "$#" -eq 0 ]; then
  set -- tests
fi

echo "log: ${LOG_PATH}"
echo "junit: ${JUNIT_PATH}"

PYTHONUNBUFFERED=1 \
SIMSOPT_STAGE2_TEST_STREAM_LOGS=1 \
conda run -n "${ENV_NAME}" env \
  PYTHONPATH="${ROOT_DIR}/tests:${ROOT_DIR}/src" \
  python -m pytest "$@" \
  -vv -ra --tb=short --durations=100 \
  --capture=tee-sys \
  -o log_cli=true \
  -o log_cli_level=INFO \
  -o console_output_style=progress \
  --junitxml="${JUNIT_PATH}" \
  2>&1 | tee "${LOG_PATH}"
