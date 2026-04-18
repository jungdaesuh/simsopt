#!/usr/bin/env bash
#
# Build simsoptpp with a named variant of compile flags.
# Re-installs the editable package into the currently-active conda env,
# using a separate cmake build dir per variant so rebuilds are incremental.
#
# Only one variant is "active" in a given env at a time — `pip install -e`
# replaces simsoptpp.so in the env's site-packages. For truly parallel
# variants (e.g. to run ASan tests while a native benchmark runs), create
# a separate env with ENV_PREFIX=./env-<variant> scripts/dev/env-create.sh
# and activate it before running this script.
#
# Variants:
#   native       -O3 -march=native -ffp-contract=fast     (default dev)
#   asan         AddressSanitizer + UBSan, -O1            (A1, A3 gates)
#   tsan         ThreadSanitizer, -O1                     (A3, B3 gates)
#   no-xsimd     NO_XSIMD=1 (exercises scalar path)       (A2 gate)
#   westmere     -O3 -march=westmere                      (CI wheel baseline)
#   x86-64-v3    -O3 -march=x86-64-v3 -ffp-contract=fast  (proposed baseline; C1)
#   debug        -O0 -g, no sanitizers
#
# Usage:
#     scripts/dev/build.sh [variant]
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

variant="${1:-native}"
case "$variant" in
    native)
        flags="-O3 -march=native -ffp-contract=fast"
        env_extra=""
        ;;
    asan)
        flags="-O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined"
        env_extra=""
        ;;
    tsan)
        flags="-O1 -g -fno-omit-frame-pointer -fsanitize=thread"
        env_extra=""
        ;;
    no-xsimd)
        flags="-O3 -march=native"
        env_extra="NO_XSIMD=1"
        ;;
    westmere)
        flags="-O3 -march=westmere"
        env_extra=""
        ;;
    x86-64-v3)
        flags="-O3 -march=x86-64-v3 -ffp-contract=fast"
        env_extra=""
        ;;
    debug)
        flags="-O0 -g"
        env_extra=""
        ;;
    *)
        echo "error: unknown variant '$variant'" >&2
        echo "valid: native asan tsan no-xsimd westmere x86-64-v3 debug" >&2
        exit 1
        ;;
esac

if ! command -v pip >/dev/null 2>&1; then
    echo "error: pip not on PATH — activate your conda env first" >&2
    echo "       conda activate ./env" >&2
    exit 1
fi

build_dir="build-$variant"
echo "==> Building simsoptpp variant=$variant"
echo "    Build dir:  $build_dir"
echo "    CXX_FLAGS:  $flags"
[[ -n "$env_extra" ]] && echo "    Env:        $env_extra"

if [[ -n "$env_extra" ]]; then
    eval "export $env_extra"
fi

# scikit-build-core reads build-dir from config settings; CMAKE_CXX_FLAGS_RELEASE
# is appended to default release flags rather than replacing them.
pip install -e . --no-build-isolation -v \
    -C build-dir="$build_dir" \
    -C cmake.args="-DCMAKE_BUILD_TYPE=Release" \
    -C cmake.define.CMAKE_CXX_FLAGS_RELEASE="$flags"

echo ""
echo "==> Built variant=$variant; simsoptpp.so is now active in the env"
echo "    Test with:  scripts/dev/test.sh"
