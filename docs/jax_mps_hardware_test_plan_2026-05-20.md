# JAX MPS Hardware Test Plan

Date: 2026-05-20

Purpose: define the local Apple Silicon MPS hardware proof lane separately
from the Perlmutter CUDA proof plan in
`docs/full_repo_banana_e2e_cpu_gpu_test_plan_2026-05-19.md`.

This document is an execution checklist. It does not claim the MPS waves have
passed.

## Scope

- [ ] Test one explicit repo snapshot.
- [ ] Prove that the local JAX runtime sees a real MPS device.
- [ ] Run the opt-in MPS smoke tests.
- [ ] Run a small non-banana MPS fixture against a CPU baseline.
- [ ] Run the banana single-stage float32 smoke path on real MPS hardware.
- [ ] Keep MPS artifacts separate from CUDA/Perlmutter artifacts.
- [ ] Report accepted versus rejected artifacts explicitly.

## Non-Goals

- [ ] Do not treat MPS as float64 production parity.
- [ ] Do not merge MPS results into Perlmutter CUDA signoff.
- [ ] Do not silently reroute MPS failures to CPU.
- [ ] Do not use `jax_metal_smoke`; the current selector is
  `jax_mps_smoke`.
- [ ] Do not loosen float64 CPU/CUDA tolerances based on MPS behavior.
- [ ] Do not claim full repo MPS support from a smoke subset.

## Current Local Hardware And Runtime Facts

Observed on 2026-05-20 before this document was written:

- Local host: Apple M3 Max MacBook Pro, 30-core Apple GPU, Metal 4.
- Main repo interpreter `.conda/jax/bin/python`:
  - `jax==0.10.0`
  - default backend `cpu`
  - devices `[CpuDevice(id=0)]`
  - `jax.devices("mps")` fails with `Unknown backend mps`
- MPS interpreter `.conda/jax-mps/bin/python`:
  - `jax==0.10.0`
  - default backend `mps`
  - devices `[MpsDevice(id=0)]`
  - JAX prints the upstream warning that MPS is experimental.

Implication: local MPS hardware proof must run under `.conda/jax-mps`, not
under `.conda/jax`.

Official-doc boundary:

- JAX's standard installation docs do not treat Mac GPU as a supported
  production backend; the JAX table marks Apple GPU experimental and the Mac
  GPU section directs standard users to CPU installation.
- Apple's Metal JAX page documents a separate experimental Metal plug-in path
  and says unsupported data types include `np.float64`, `np.complex64`, and
  `np.complex128`.
- JAX's `JAX_PLATFORMS` option is fail-closed: every listed platform must
  initialize, and the first platform in the list becomes the default.
- JAX transfer guard `disallow` rejects implicit host/device transfers while
  allowing explicit `jax.device_put*()` and `jax.device_get()` calls.
- This repo-local plan tests the installed `.conda/jax-mps` interpreter and
  backend id `mps`. Do not infer that a passing artifact validates Apple's
  `jax-metal` package, standard JAX CUDA behavior, or float64 production
  parity.

Official sources checked:

- https://docs.jax.dev/en/latest/installation.html
- https://docs.jax.dev/en/latest/config_options.html#platforms
- https://docs.jax.dev/en/latest/default_dtypes.html
- https://docs.jax.dev/en/latest/transfer_guard.html
- https://developer.apple.com/metal/jax/

## Required Inputs

- [ ] Exact repo SHA to test: `<repo_sha>`.
  - Fill from `git rev-parse HEAD` immediately before Wave 0.
- [ ] Source mode:
  - [ ] clean committed SHA, or
  - [ ] dirty-tree proof with explicit `git diff` manifest.
- [ ] Runtime:

  ```bash
  export MPS_PYTHON="$PWD/.conda/jax-mps/bin/python"
  export CPU_PYTHON="$PWD/.conda/jax/bin/python"
  export RESULTS_ROOT="$PWD/.artifacts/jax_mps_hardware_20260520"
  export SINGLE_STAGE_JAX_RUNTIME_SEED_SPEC="$PWD/.artifacts/parity/20260507-cpp-jaxcpu-single-stage/m1-cases/single_stage_jax_runtime_seed_spec.json"
  ```

- [ ] The MPS runtime imports `jax_plugins.mps`.
- [ ] `SINGLE_STAGE_JAX_RUNTIME_SEED_SPEC` exists before the banana waves run.

## Common MPS Environment

Use this environment for every MPS command:

```bash
export PYTHONPATH="$PWD:$PWD/src"
export JAX_ENABLE_X64=0
export JAX_PLATFORMS=mps
export SIMSOPT_BACKEND_MODE=jax_mps_smoke
export SIMSOPT_BACKEND_STRICT=1
export SIMSOPT_JAX_PLATFORM=mps
export SIMSOPT_EXAMPLE_PARITY_JAX_PLATFORM=mps
export SIMSOPT_JAX_TRANSFER_GUARD=disallow
```

The MPS lane is float32 smoke by policy. CPU and CUDA production parity remain
float64 lanes.

## Wave 0: Source And Runtime Preflight

Purpose: prove the local command is using the intended source and real MPS
runtime before running expensive fixtures.

```bash
mkdir -p "${RESULTS_ROOT}/wave0_mps_preflight"

git rev-parse HEAD | tee "${RESULTS_ROOT}/wave0_mps_preflight/repo_sha.txt"
git status --short --untracked-files=no \
  | tee "${RESULTS_ROOT}/wave0_mps_preflight/git_status_short.txt"

"${MPS_PYTHON}" - <<'PY' | tee "${RESULTS_ROOT}/wave0_mps_preflight/jax_mps_preflight.json"
import json
import importlib.metadata as metadata

import jax
import jax.numpy as jnp
import numpy as np

from simsopt.backend import get_backend_policy, set_backend

set_backend("jax_mps_smoke")
mps_device = jax.devices("mps")[0]
value = jax.device_put(np.asarray([1.0, 2.0], dtype=np.float32), mps_device)
delta = jax.device_put(np.asarray([3.0, 4.0], dtype=np.float32), mps_device)
value = jnp.add(value, delta).block_until_ready()

payload = {
    "jax": jax.__version__,
    "jax_mps_version": metadata.version("jax-mps"),
    "backend": jax.default_backend(),
    "devices": [str(device) for device in jax.devices()],
    "device_platforms": [device.platform for device in jax.devices()],
    "x64": bool(jax.config.read("jax_enable_x64")),
    "policy_runtime_dtype": get_backend_policy().runtime_dtype,
    "policy_tolerance_tier": get_backend_policy().tolerance_tier,
    "value_dtype": str(value.dtype),
    "value_device": str(value.device),
    "value_device_platform": value.device.platform,
}
print(json.dumps(payload, indent=2, sort_keys=True))
assert payload["backend"] == "mps"
assert payload["x64"] is False
assert payload["policy_runtime_dtype"] == "float32"
assert payload["policy_tolerance_tier"] == "float32_smoke"
assert "mps" in payload["device_platforms"]
assert payload["value_device_platform"] == "mps"
PY
```

Acceptance:

- [ ] `backend` is exactly `mps`.
- [ ] At least one device reports `platform == "mps"`.
- [ ] x64 is false.
- [ ] `jax-mps` is installed and its version is recorded.
- [ ] `jax-metal` is either absent or explicitly recorded separately; this plan
  does not mix `jax-mps` and `jax-metal` artifacts.
- [ ] backend policy reports `runtime_dtype=float32`.
- [ ] backend policy reports `tolerance_tier=float32_smoke`.
- [ ] `git status` is recorded. Dirty-tree evidence is allowed only if the
  report labels it as dirty-tree evidence.

## Wave 1: Opt-In MPS Smoke Tests

Purpose: run the repo's hardware-marked MPS smoke tests.

```bash
mkdir -p "${RESULTS_ROOT}/wave1_mps_pytest"

"${MPS_PYTHON}" -m pytest \
  tests/test_jax_mps_smoke.py \
  tests/test_mps_smoke_dtype.py \
  -m mps \
  -ra --tb=short --durations=50 \
  --junitxml="${RESULTS_ROOT}/wave1_mps_pytest/mps_smoke.xml" \
  | tee "${RESULTS_ROOT}/wave1_mps_pytest/mps_smoke.log"
```

Acceptance:

- [ ] Both MPS smoke tests pass.
- [ ] Skips are accepted only if the preflight already proves no real MPS
  backend is available. On this machine, a skip is a failure.

## Wave 2: Non-Banana MPS Fixture Smoke

Purpose: prove one known-small JAX wrapper path on real MPS against a CPU
baseline before running banana.

CPU baseline:

```bash
mkdir -p "${RESULTS_ROOT}/wave2_non_banana"

SIMSOPT_BACKEND_MODE=jax_cpu_parity \
SIMSOPT_JAX_TRANSFER_GUARD=disallow \
JAX_PLATFORMS=cpu \
JAX_ENABLE_X64=1 \
"${CPU_PYTHON}" benchmarks/non_banana_example_cpp_jax_cpu_parity.py \
  --fixtures surface_area_volume_simple \
  --lanes cpu_cpp,jax_cpu \
  --output-json "${RESULTS_ROOT}/wave2_non_banana/cpu_surface_area_volume_simple.json"
```

MPS smoke:

```bash
SIMSOPT_BACKEND_MODE=jax_mps_smoke \
SIMSOPT_BACKEND_STRICT=1 \
SIMSOPT_JAX_TRANSFER_GUARD=disallow \
JAX_PLATFORMS=mps \
JAX_ENABLE_X64=0 \
"${MPS_PYTHON}" benchmarks/non_banana_example_cpp_jax_cpu_parity.py \
  --fixtures surface_area_volume_simple \
  --lanes cpu_cpp,jax_mps \
  --baseline-json "${RESULTS_ROOT}/wave2_non_banana/cpu_surface_area_volume_simple.json" \
  --output-json "${RESULTS_ROOT}/wave2_non_banana/mps_surface_area_volume_simple.json"
```

Acceptance:

- [ ] CPU baseline artifact passes.
- [ ] MPS artifact passes.
- [ ] MPS artifact records `backend_mode=jax_mps_smoke`.
- [ ] MPS artifact records `runtime_dtype=float32` and
  `tolerance_tier=float32_smoke`.
- [ ] Any CPU backend in the MPS artifact invalidates the wave.

Optional diagnostic inventory:

```bash
SIMSOPT_BACKEND_MODE=jax_cpu_parity \
SIMSOPT_JAX_TRANSFER_GUARD=disallow \
JAX_PLATFORMS=cpu \
JAX_ENABLE_X64=1 \
"${CPU_PYTHON}" benchmarks/non_banana_example_cpp_jax_cpu_parity.py \
  --fixtures all-supported \
  --lanes cpu_cpp,jax_cpu \
  --output-json "${RESULTS_ROOT}/wave2_non_banana/cpu_all_supported_diagnostic_baseline.json"

SIMSOPT_BACKEND_MODE=jax_mps_smoke \
SIMSOPT_BACKEND_STRICT=1 \
SIMSOPT_JAX_TRANSFER_GUARD=disallow \
JAX_PLATFORMS=mps \
JAX_ENABLE_X64=0 \
"${MPS_PYTHON}" benchmarks/non_banana_example_cpp_jax_cpu_parity.py \
  --fixtures all-supported \
  --lanes cpu_cpp,jax_mps \
  --baseline-json "${RESULTS_ROOT}/wave2_non_banana/cpu_all_supported_diagnostic_baseline.json" \
  --output-json "${RESULTS_ROOT}/wave2_non_banana/mps_all_supported_diagnostic.json"
```

The all-supported command is diagnostic until the current MPS wrapper inventory
is closed. It does not gate banana smoke acceptance.

## Wave 3: Banana Single-Stage MPS Smoke

Purpose: run the banana target-lane path on real MPS hardware and require the
artifact gate to decide accepted versus rejected status.

```bash
mkdir -p "${RESULTS_ROOT}/wave3_banana_single_stage"

SIMSOPT_BACKEND_MODE=jax_mps_smoke \
SIMSOPT_BACKEND_STRICT=1 \
SIMSOPT_JAX_TRANSFER_GUARD=disallow \
JAX_PLATFORMS=mps \
JAX_ENABLE_X64=0 \
"${MPS_PYTHON}" examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py \
  --backend jax \
  --optimizer-backend scipy-jax \
  --mpol 2 \
  --ntor 2 \
  --nphi 31 \
  --ntheta 16 \
  --maxiter 7 \
  --minimal-artifacts \
  --jax-runtime-seed-spec "${SINGLE_STAGE_JAX_RUNTIME_SEED_SPEC}" \
  --output-root "${RESULTS_ROOT}/wave3_banana_single_stage"
```

Acceptance:

- [ ] Process exits `0`.
- [ ] Exactly one run directory is written under the output root.
- [ ] `results.json` exists.
- [ ] `REJECTED.json` does not exist.
- [ ] `results.json` reports `OPTIMIZER_SUCCESS=true`.
- [ ] `results.json` records `backend_mode=jax_mps_smoke`,
  `runtime_dtype=float32`, and `tolerance_tier=float32_smoke`.
- [ ] `results.json` records MPS execution for the target-lane JAX value/grad
  path. `--optimizer-backend scipy-jax` is host optimizer control only; it is
  not accepted as a CPU reroute.
- [ ] Objective, final DOFs, and optimizer gradient finiteness fields are
  accepted by the strict result gate.
- [ ] If the process exits nonzero or writes `REJECTED.json`, classify the
  failure from that artifact; do not infer success from diagnostic side files.

## Wave 4: Banana MPS Gradient Diagnosis

Purpose: collect per-term adjoint evidence on real MPS for comparison with CPU
float32 smoke and CUDA float64 parity.

```bash
mkdir -p "${RESULTS_ROOT}/wave4_banana_gradient_diagnosis"

SIMSOPT_BACKEND_MODE=jax_mps_smoke \
SIMSOPT_BACKEND_STRICT=1 \
SIMSOPT_JAX_TRANSFER_GUARD=disallow \
JAX_PLATFORMS=mps \
JAX_ENABLE_X64=0 \
"${MPS_PYTHON}" examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py \
  --backend jax \
  --optimizer-backend scipy-jax \
  --mpol 2 \
  --ntor 2 \
  --nphi 31 \
  --ntheta 16 \
  --maxiter 7 \
  --minimal-artifacts \
  --diagnose-target-lane-gradient \
  --jax-runtime-seed-spec "${SINGLE_STAGE_JAX_RUNTIME_SEED_SPEC}" \
  --output-root "${RESULTS_ROOT}/wave4_banana_gradient_diagnosis"
```

Acceptance:

- [ ] `target_lane_gradient_diagnosis.json` exists.
- [ ] Total gradient summary is finite.
- [ ] Every per-term report records solve status, residual,
  residual-relative, and iterations (`null` is valid for unknown GMRES
  iterations).
- [ ] Any NaN-filled term is a blocker for accepted MPS banana smoke until the
  failed solve status is explained.

## Wave 5: Report

Create `REPORT.md` under `${RESULTS_ROOT}` with:

- [ ] repo SHA and dirty-tree status.
- [ ] local hardware facts.
- [ ] MPS Python path and package versions.
- [ ] preflight JSON path.
- [ ] pytest JUnit/log paths.
- [ ] non-banana CPU and MPS artifact paths.
- [ ] banana run directory and accepted/rejected marker path.
- [ ] gradient diagnosis artifact path.
- [ ] verdict table:

  | Area | Artifact | Backend | Result | Notes |
  | --- | --- | --- | --- | --- |
  | MPS preflight |  |  |  |  |
  | MPS smoke pytest |  |  |  |  |
  | Non-banana surface fixture |  |  |  |  |
  | Banana single-stage |  |  |  |  |
  | Banana gradient diagnosis |  |  |  |  |

## CUDA Separation Rule

MPS evidence from this plan is local Apple-GPU float32 smoke evidence only.
CUDA release-grade proof remains in
`docs/full_repo_banana_e2e_cpu_gpu_test_plan_2026-05-19.md` and must run on
Perlmutter under `jax_gpu_parity` / `JAX_PLATFORMS=cuda,cpu` / x64.

Do not combine MPS and CUDA artifacts into one pass/fail verdict. The final
release table should carry separate columns:

```text
CPU float64 oracle | JAX CPU float64 | JAX CUDA float64 | MPS float32 smoke
```

## Blocker Rules

- [ ] Missing MPS backend blocks all MPS hardware claims.
- [ ] Any MPS command that records CPU backend is invalid for MPS signoff.
- [ ] Any accepted artifact containing non-finite active objective, state, or
  gradient values is invalid.
- [ ] Inactive zero-weight terms must have `active=false` or equivalent
  dependency metadata and zero weighted contribution. A non-finite inactive raw
  diagnostic value must not poison the accepted objective, but it must not be
  counted as active physics evidence.
- [ ] Any `results.json` refusal is a real failure unless explicitly accepted
  as a rejected-run diagnostic.
- [ ] Float32 MPS smoke failures do not weaken CPU/CUDA float64 production
  parity.
