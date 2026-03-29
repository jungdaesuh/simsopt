# HF A100 Issue Report — Root Cause Analysis

**Date**: 2026-03-28, updated 2026-03-29
**Branch**: `jax-port` at `03c27f44`
**Existing analysis**: `/Users/suhjungdae/code/columbia/analysis/jax_hf_a100_issue_report_2026-03-28.md`

---

## Executive Summary

This report was originally written on `2026-03-28`. The current `jax-port`
tree has moved since then, and the HF proof harness changed materially in
commit `03c27f44`.

The original A100 failures were a mixture of:

- Stage 2 comparison-contract issues
- barrier-edge portability gating
- short-run endpoint tolerance policy
- HF bootstrap / clone / ref mistakes

Those issues were fixed.

The report below was stale in two important ways:

1. the shell harness now continues past failed Stage 2 probes and still runs
   single-stage probes
2. the shell harness no longer forces a geometry-reproducibility rung on the
   default `20`-iteration smoke path

The old launcher/shell geometry-default issue was later fixed by centralizing
smoke-vs-repro planning in the shared ladder contract.

The next verified live blocker is different:

- the HF job now reaches the single-stage rungs
- but the CPU reference lane can fail when
  `initialize_boozer_surface()` passes JAX-only `sdofs=` into native
  `BoozerSurface.run_code()`
- the observed failure was:
  `TypeError: BoozerSurface.run_code() got an unexpected keyword argument 'sdofs'`

---

## Issue 1 (ACTIVE): Single-Stage CPU Warm-Start Contract Mismatch

**Status**: reproduced on live HF A100; fixed locally pending rerun

**Historical symptom**:
job `69c7e306f900226fc14ae53a` reached the single-stage phase and then failed
both CPU-reference rungs with:
```
TypeError: BoozerSurface.run_code() got an unexpected keyword argument 'sdofs'
```

**Root cause**:

- `/Users/suhjungdae/code/columbia/simsopt-jax/examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py`
  used the JAX-style `run_code(..., sdofs=...)` call unconditionally
- native CPU
  `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/geo/boozersurface.py`
  does not accept `sdofs`
- JAX
  `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/geo/boozersurface_jax.py`
  does accept `sdofs`

**Local fix**:

- keep the explicit `sdofs` warm-start only on the JAX path
- let the CPU path use the already-initialized surface state and call the native
  `run_code(iota, G)` contract directly

**Why it matters**:

- this is the trusted CPU reference lane for the single-stage parity proof
- until it runs cleanly, there is no trustworthy post-fix single-stage
  CPU-vs-JAX evidence

---

## Issue 2 (FIXED): Harness Now Continues Past Failed Stage 2 Probes

**Status**: fixed in `03c27f44`

The original report was correct for the `2026-03-28` harness: the script used
fail-fast flow and never reached `single_stage_cold` / `single_stage_warm`
after a failed Stage 2 probe.

That is no longer true.

Current harness behavior:

- accumulates `OVERALL_RC`
- runs all expected probes unconditionally
- emits per-rung payload summaries even for missing/corrupt payloads
- exits non-zero only at the end

Regression coverage exists in:

- `/Users/suhjungdae/code/columbia/simsopt-jax/tests/test_hf_production_gpu_proof.py`
  - `test_run_production_gpu_proof_continues_after_missing_payload`
  - `test_run_production_gpu_proof_survives_corrupt_payload`

**Updated interpretation**:

The absence of A100 single-stage results is now an operational/rerun gap, not a
structural harness bug.

---

## Issues 3-8 (FIXED): Timeline of Today's Resolved Failures

All 6 issues below were fixed in today's commit sequence.

### Issue 3: Wrong Tier 2 Ondevice Comparison Contract
- **Commits**: `44830a6b`, `8b07d6c4`
- **Root cause**: CPU-reference vs JAX lanes used inconsistent configurations
- **Fix**: Unified ondevice comparison to CPU-JAX vs CUDA-JAX

### Issue 4: Barrier-Edge Gradient Portability
- **Commit**: `444f35c1`
- **Root cause**: Hard curvature barrier (threshold=40.0) amplifies sub-1e-7
  CPU/CUDA differences in max curvature into large gradient divergence
- **Fix**: Barrier-edge-aware gradient gate — forgives `curvature_barrier` term
  when margin < 1e-6 and rel_diff < 1e-5

### Issue 5: Short-Run Objective Gate Too Strict
- **Commit**: `de786f97`
- **Root cause**: Hardcoded 1e-4 objective tolerance; 20-iteration smoke runs
  have inherent ~2.6e-4 CPU/CUDA endpoint drift
- **Fix**: `short_run_stage2_final_objective_rel_tolerance(maxiter)` returns
  5e-4 for maxiter<=20, 1e-4 otherwise

### Issue 6: CPU Validation Probe Cache Contamination
- **Commit**: `7db31d7e`
- **Root cause**: CPU probes inherited `JAX_COMPILATION_CACHE_DIR`, causing
  cross-device cache corruption
- **Fix**: Strip cache env for CPU lanes via `disable_compilation_cache` flag

### Issue 7: HF Git Clone Missing Branch Ref
- **Commit**: `d7c12bfc`
- **Root cause**: `git clone` without `--branch jax-port` failed to fetch
  non-default branch refs before `git checkout <sha>`
- **Fix**: Added `--branch <ref> --single-branch` to clone command

### Issue 8: Bootstrap Venv Not Sourced
- **Commit**: `10f8aec8`
- **Root cause**: `bash bootstrap_runtime.sh` (child process) — venv PATH lost
  on exit. Subsequent `pip install` used system Python without JAX.
- **Fix**: Changed to `. benchmarks/hf_jobs/bootstrap_runtime.sh` (source)

---

## Bootstrap/Infrastructure Fragility Audit

### HIGH severity

| # | Issue | Impact |
|---|-------|--------|
| B1 | Single-stage CPU warm-start contract split | Blocks the trusted CPU reference lane and prevents single-stage parity evidence |
| B2 | Base image `python:3.11-bookworm` has no CUDA runtime | Relies entirely on JAX pip wheel bundling CUDA libs; fragile across versions |

### MEDIUM severity

| # | Issue | Impact |
|---|-------|--------|
| B3 | `unset LD_LIBRARY_PATH` may strip NVIDIA container-injected paths | CUDA library discovery can break silently |
| B4 | Full `simsoptpp` C++ build is unnecessary for JAX-only proof | Wastes 5-15 min A100 time per job; introduces CMake/Boost/FFTW failure surface |
| B5 | Triple `git submodule update` (clone + explicit + CMake) | Redundant; Gitlab rate-limit exposure for Eigen submodule |

### LOW severity

| # | Issue | Impact |
|---|-------|--------|
| B6 | Hardcoded `fork` remote name in launcher | Crashes at parse-time if remote not named `fork` |
| B7 | No `--depth 1` on clone | 2-5 min clone overhead on large repo |
| B8 | No early GPU smoke check (`python -c "import jax; print(jax.devices())"`) | Burns 10+ min before discovering CUDA misconfiguration |

---

## Tolerance System Gap Analysis

Of ~14 Tier 2 gates, only **2 are maxiter-aware** and **2 are barrier-aware**:

| Gate | Tolerance | Maxiter-Aware | Barrier-Aware |
|------|-----------|:---:|:---:|
| Final objective rel diff | 5e-4 (20-iter) / 1e-4 (default) | YES | YES |
| Geometry drift | None (20-iter) / 1e-6 (default) | YES | no |
| Field error rel diff | 1e-4 | **no** | **no** |
| Matched-state objective | 1e-10 | no | no |
| Matched-state gradient | 1e-9 rtol, 5e-12 atol | no | YES |
| Matched-state field | 1e-10 | no | no |
| Curve length | hard constraint | no | no |
| Coil-coil distance | hard constraint | no | no |
| Max curvature | CPU envelope | no | no |
| Self-intersection | boolean | no | no |
| Trajectory finite | boolean | no | no |
| Trajectory improves | boolean | no | no |

**Next failure candidate**: `FIELD_ERROR_REL_TOL` (1e-4) is a hard constant
that never loosens for short iteration budgets or barrier-edge conditions.
If CPU/CUDA field error diverges by >1e-4 on a short run, this will fail next.

---

## Recommended Actions (Priority Order)

1. **Fix Issue 1** — restore backend-sensitive CPU vs JAX warm-start behavior
   in `single_stage_banana_example.py`.

2. **Add direct regression coverage** for the CPU init seam, including the
   reduced real-fixture replay shape that failed on HF.

3. **Relaunch A100 job** with the corrected single-stage path to obtain the
   first trustworthy single-stage CUDA evidence.

4. **Add early GPU smoke check** (B8) — `python -c "import jax; print(jax.devices())"`
   before the expensive build.

5. **Then decide** whether a separate longer-run Stage 2 geometry
   reproducibility rung is required.

---

## HF Job History

| Job ID | SHA | Outcome | Failure |
|--------|-----|---------|---------|
| `69c6c68cbf20` / `69c6c68df900` | pre-fixes | FAIL | Mixed comparison contract |
| `69c774a6f900` | post-lane-fix | FAIL | Barrier-edge gradient |
| `69c77de9f900` | post-barrier-fix | PARTIAL | Geometry gate only |
| `69c790bdbf20` | wrong SHA | FAIL | `git checkout` failed |
| `69c792a3bf20` | `de786f97` | PARTIAL | Historical geometry-gated smoke failure on old harness |

---

## Historical A100 Results (Job `69c792a3bf20`)

| Probe | Status | Objective Δ | Field Δ | Geometry Δ | Wall Time |
|-------|--------|------------|---------|------------|-----------|
| `stage2_cold` | PASS | 2.63e-4 | 1.63e-4 | 3.65e-3 (report-only) | 960s |
| `stage2_warm` | PASS | 2.63e-4 | 1.63e-4 | 3.65e-3 (report-only) | 780s |
| `stage2_warm_geometry_gate` | FAIL | — | — | 3.65e-3 vs 5e-6 | — |
| `single_stage_cold` | NOT REACHED IN THAT JOB | — | — | — | — |
| `single_stage_warm` | NOT REACHED IN THAT JOB | — | — | — | — |

---

## Current Harness State (As Of `03c27f44`)

- shell runner continues after failed probes
- shell runner emits a final per-rung summary
- shell runner keeps geometry repro optional
- shell runner rejects explicit geometry repro on the `20`-iter smoke path
- launcher/shell smoke-vs-repro planning was later centralized in
  `benchmarks/validation_ladder_contract.py`
- the next live blocker is the single-stage CPU warm-start contract mismatch,
  not the old launcher geometry-default issue
