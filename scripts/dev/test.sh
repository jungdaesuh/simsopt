#!/usr/bin/env bash
#
# Run tests against the currently-active simsoptpp build variant.
# Auto-detects ASan/TSan from the installed .so and sets sanitizer env
# vars accordingly. Passes remaining args through to pytest.
#
# Usage:
#     scripts/dev/test.sh                    # full suite
#     scripts/dev/test.sh tests/field        # one dir
#     scripts/dev/test.sh -k biot_savart     # pytest filter
#     scripts/dev/test.sh -n auto            # parallel (pytest-xdist)
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if ! command -v pytest >/dev/null 2>&1; then
    echo "error: pytest not on PATH — activate your conda env first" >&2
    exit 1
fi

so_path="$(python -c 'import simsoptpp, os; print(os.path.realpath(simsoptpp.__file__))' 2>/dev/null || true)"

sanitizer=""
if [[ -n "$so_path" && -f "$so_path" ]]; then
    # strings / nm work on both ELF and Mach-O; grep for sanitizer symbols.
    if strings "$so_path" 2>/dev/null | grep -q "AddressSanitizer"; then
        sanitizer="asan"
    elif strings "$so_path" 2>/dev/null | grep -q "ThreadSanitizer"; then
        sanitizer="tsan"
    fi
fi

case "$sanitizer" in
    asan)
        echo "==> ASan build detected; enabling leak detection"
        export ASAN_OPTIONS="detect_leaks=1:abort_on_error=0:halt_on_error=0:symbolize=1"
        export UBSAN_OPTIONS="print_stacktrace=1:halt_on_error=0"
        # macOS requires the ASan runtime to be dlopen-able by Python.
        # clang from conda-forge puts it in $CONDA_PREFIX/lib/clang/*/lib/darwin.
        if [[ "$(uname)" == "Darwin" && -n "${CONDA_PREFIX:-}" ]]; then
            asan_dylib="$(ls "$CONDA_PREFIX"/lib/clang/*/lib/darwin/libclang_rt.asan_osx_dynamic.dylib 2>/dev/null | head -1 || true)"
            if [[ -n "$asan_dylib" ]]; then
                export DYLD_INSERT_LIBRARIES="$asan_dylib"
                echo "    DYLD_INSERT_LIBRARIES=$asan_dylib"
            fi
        fi
        ;;
    tsan)
        echo "==> TSan build detected; enabling thread-race detection"
        export TSAN_OPTIONS="halt_on_error=0:second_deadlock_stack=1:history_size=7"
        ;;
    *)
        ;;
esac

exec pytest "$@"
