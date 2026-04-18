# Isolated dev environment

Tooling to build and test `simsopt` with a fully in-tree conda env and per-variant cmake build dirs. Enables the sanitizer, non-XSIMD, and wheel-baseline builds required by `PERFORMANCE_AUDIT.md`.

## One-time setup

```bash
# Requires miniforge / mambaforge on PATH.
scripts/dev/env-create.sh        # creates ./env with python, compilers, deps
conda activate ./env
```

## Per-variant build

```bash
scripts/dev/build.sh native      # default dev (-march=native)
scripts/dev/build.sh asan        # AddressSanitizer + UBSan
scripts/dev/build.sh tsan        # ThreadSanitizer
scripts/dev/build.sh no-xsimd    # scalar path (NO_XSIMD=1)
scripts/dev/build.sh westmere    # current CI wheel baseline
scripts/dev/build.sh x86-64-v3   # proposed wheel baseline (C1)
scripts/dev/build.sh debug       # -O0 -g, no sanitizers
```

Each variant writes to `build-<variant>/`. Only one variant is active in the env at a time — rerun `build.sh <variant>` to switch.

## Running tests

```bash
scripts/dev/test.sh                    # full suite
scripts/dev/test.sh tests/field        # one dir
scripts/dev/test.sh -k biot_savart     # filter
scripts/dev/test.sh -n auto            # parallel
```

`test.sh` auto-detects ASan/TSan builds from the loaded `.so` and sets the runtime env vars (`ASAN_OPTIONS`, `DYLD_INSERT_LIBRARIES` on macOS, etc.).

## Parallel variants (optional)

To run, say, an ASan test suite while a native benchmark runs, create a second env:

```bash
ENV_PREFIX=./env-asan scripts/dev/env-create.sh
conda activate ./env-asan
scripts/dev/build.sh asan
```

## Resetting

```bash
rm -rf env build-*                # wipe everything
rm -rf build-<variant>            # wipe one variant
```
