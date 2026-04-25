#!/usr/bin/env bash
set -euo pipefail

EXPECTED_JAX_VERSION="${SIMSOPT_HF_JOB_EXPECTED_JAX_VERSION:-0.9.2}"
CUDA_LIBRARY_MODE="${SIMSOPT_JAX_CUDA_LIBRARY_MODE:-bundled}"
VENV_DIR="/opt/venv"
BOOTSTRAP_SOURCED=0
BOOTSTRAP_JAX_SMOKE_JSON="${SIMSOPT_HF_BOOTSTRAP_JAX_SMOKE_JSON:-bootstrap_jax_smoke.json}"

if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
  BOOTSTRAP_SOURCED=1
fi

activate_runtime_env() {
  export PATH="${VENV_DIR}/bin:${PATH}"
  export VIRTUAL_ENV="${VENV_DIR}"
  export SIMSOPT_JAX_CUDA_LIBRARY_MODE="${CUDA_LIBRARY_MODE}"
}

bootstrap_finish() {
  local status="${1:-0}"
  if [[ "${BOOTSTRAP_SOURCED}" -eq 1 ]]; then
    return "${status}"
  fi
  exit "${status}"
}

verify_runtime_versions() {
  python - "${EXPECTED_JAX_VERSION}" "${BOOTSTRAP_JAX_SMOKE_JSON}" <<'PY'
import json
import sys
from pathlib import Path

import jax
import jaxlib
import numpy
import scipy

expected = sys.argv[1]
smoke_path = Path(sys.argv[2])
default_backend = str(jax.default_backend()).lower()
payload = {
    "jax": jax.__version__,
    "jaxlib": jaxlib.__version__,
    "numpy": numpy.__version__,
    "scipy": scipy.__version__,
    "default_backend": default_backend,
    "devices": [str(device) for device in jax.devices()],
}
print(json.dumps(payload, sort_keys=True))
smoke_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
if jax.__version__ != expected or jaxlib.__version__ != expected:
    raise SystemExit(
        f"Expected JAX/JAXLIB {expected}, got "
        f"jax={jax.__version__} jaxlib={jaxlib.__version__}"
    )
if default_backend not in {"gpu", "cuda"}:
    raise SystemExit(
        "Expected GPU JAX backend during HF proof bootstrap, got "
        f"{default_backend!r} on devices {payload['devices']}"
    )
PY
}


if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "Production GPU proof requires a prebuilt runtime at ${VENV_DIR}/bin/python." >&2
  bootstrap_finish 1
fi

activate_runtime_env
verify_runtime_versions
bootstrap_finish 0
