#!/usr/bin/env bash
# Ruff pin for the JAX workflows and for humans, who call this instead of
# unpinned `uvx ruff`. It matches upstream's pin in .github/workflows/linting.yml
# (ruff-action `version: 0.15.0`); change both together.
#
# Pair a 0.16.x bump with the auto-fixable hits that version introduces.
set -euo pipefail

RUFF_VERSION=0.15.0
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

run_ruff() {
  if command -v uvx >/dev/null 2>&1; then
    uvx --from "ruff==${RUFF_VERSION}" ruff "$@"
    return
  fi
  python3 -m pip install -q "ruff==${RUFF_VERSION}"
  python3 -m ruff "$@"
}

usage() {
  cat <<'EOF'
Usage:
  scripts/lint.sh              ruff check .
  scripts/lint.sh check PATH…  ruff check PATH…
  scripts/lint.sh format PATH… ruff format --check PATH…
  scripts/lint.sh --version    print the pin and the ruff binary version
EOF
}

if [[ $# -eq 0 ]]; then
  run_ruff check .
  exit 0
fi

case "$1" in
  -h | --help)
    usage
    ;;
  --version)
    echo "ruff==${RUFF_VERSION}"
    run_ruff --version
    ;;
  check)
    shift
    if [[ $# -eq 0 ]]; then
      run_ruff check .
    else
      run_ruff check "$@"
    fi
    ;;
  format)
    shift
    if [[ $# -eq 0 ]]; then
      echo "scripts/lint.sh format requires paths" >&2
      exit 2
    fi
    run_ruff format --check "$@"
    ;;
  *)
    run_ruff check "$@"
    ;;
esac
