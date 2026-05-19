# JAX-Native Round-3 — Curated TODOs (2026-05-18, refreshed 2026-05-19)

Derived from `docs/jax_native_round3_performance_todos_2026-05-18.md`
after cross-validating each item against the source. Items the round-3
doc lists as "confirmed" but whose source-level work has already
landed are reclassified here as either "evidence-only" or "closed".

**Refresh delta (since 2026-05-18 19:00):**
- **N21 closed** — design pivoted to single public-boundary
  materialization; real-fixture transfer-guard test passes.
- **N24, N26 benchmark probes added** — `benchmarks/per_coil_unit_field_vmap_probe.py`
  and `benchmarks/framedcurve_vjp_scaling.py` (recorded artifacts still
  pending).
- **N33 equal-gradient proof added** —
  `test_exact_linearization_residency_dual_instance_gradient_path_matches`.
- **N34 dual-instance cache observation added**; worked restart
  example still open.
- **N30 Perlmutter plan added** —
  `docs/perlmutter_gpu_test_plan_2026-05-19.md`. Still hardware-gated.

## Critical path

The single remaining item whose absence blocks declaring the JAX GPU
port delivered.

- [x] **N30 — multi-GPU speedup proof for N11 + N12 sharding**
  - Planning: `docs/perlmutter_gpu_test_plan_2026-05-19.md`,
    `docs/full_repo_banana_e2e_cpu_gpu_test_plan_2026-05-19.md`.
  - Prerequisites:
    - [x] Resolve CUDA toolchain mismatch by running on Perlmutter
      with JAX/JAXLIB `0.10.0`.
    - [x] Multi-GPU hardware (≥ 4 devices in one box).
  - Action:
    - [x] 1 / 2 / 4-GPU wall-time + HBM-peak + HLO-collective-bytes
      sweep on `integral_BdotN_surface_sharded`
      (`src/simsopt/jax_core/integral_bdotn.py:240`).
    - [x] Same sweep on seed-batch scoring
      (`src/simsopt/geo/surfaceobjectives_jax.py:5088`).
    - [x] Write `docs/jax_multi_gpu_proof_2026-05-19.md`.
  - Acceptance: > 1.5× at 2 GPUs, > 2.5× at 4 GPUs vs 1-GPU baseline,
    parity preserved on `benchmarks/single_stage_init_parity.py`.
  - Closeout: Perlmutter jobs `53168131` and `53168132` passed the
    pre-sharded steady-state proof. `integral_BdotN_surface_sharded` measured
    2.03× at 2 GPUs and 3.87× at 4 GPUs on the regular job; seed-batch scoring
    measured 1.93× at 2 GPUs and 3.78× at 4 GPUs. The failed non-pre-sharded
    probes are retained as diagnostic evidence that repeated placement, not
    sharded compute, was the bottleneck. Follow-up Perlmutter job `53170493`
    passed `benchmarks/single_stage_init_parity.py` on four A100s with active
    point sharding after the private optimizer and Boozer penalty geometry
    placement fixes.

## Source-open (partial)

Real code or example work remaining, beyond pure measurement.

- [ ] **N33 — selective host-residency, production memory probe**
  - Done: `linearization_residency={"device","host"}` accepted; dense
    LS factors placed on CPU and restaged for runtime solve callbacks;
    **equal-gradient two-solver proof landed**
    (`tests/geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXExactPath::test_exact_linearization_residency_dual_instance_gradient_path_matches`).
  - Action:
    - [ ] Production-N memory probe (peak HBM / RSS at host vs device
      residency on a real fixture).

- [ ] **N34 — checkpoint/restart workflow, worked example**
  - Done: `with_cpu_device_for_construction()` exported; GPU OOM docs
    prescribe checkpoint/restart and reject transparent
    compiled-JIT retargeting; dual-instance cache-observation test
    exists.
  - Action:
    - [ ] Worked command-level restart example (script + recorded
      run).
    - [ ] Real GPU hardware proof for the CPU-pinned + GPU-routed
      two-instance compile-cache path.

## Hardware-gated (CPU correctness already in)

Source state is what it should be on CPU; the missing artifact
requires real GPU access.

- [ ] **N27** — CUDA TF32 speedup comparison for parity-mode
  `matmul_precision`. Source pinned `highest` for parity modes;
  needs real CUDA timings.
- [ ] **N28** — Real CUDA pre-import subprocess matrix. Determinism
  validation lands in source; only import-smoke covers it today.
- [ ] **N31** — Real GPU allocator log + memory-limit proof. Runtime
  owns the env policy and `SIMSOPT_*` overrides; needs an actual
  allocator-pressure recording.

## Evidence-only (source complete; only artifact missing)

Goal is satisfied at source level; benchmark / counter probe exists
or only the recorded artifact is missing.

- [ ] **N22** — `jax.lax.while_loop` manual-LS path.
  - Source: in (`src/simsopt/geo/optimizer_jax.py:1305, 1405, 2390`
    and `optimizer_jax_private/_bfgs.py`, `_line_search.py`).
  - Missing: dedicated benchmark improvement recording.
- [ ] **N23** — Direct-coil scalar pulls.
  - Source: in (helpers return JAX scalars; public-boundary
    materializes once).
  - Missing: transfer-count counter + real wrapper sweep.
- [ ] **N24** — `_per_coil_unit_field` scaling probe.
  - Source: vmap landed at
    `src/simsopt/field/biotsavart_jax_backend.py:209`.
  - Probe: `benchmarks/per_coil_unit_field_vmap_probe.py`.
  - Missing: recorded probe artifact + HLO report.
- [ ] **N26** — Framed-curve VJP fan-out scaling probe.
  - Source: collapsed to 9 sites (target was <22).
  - Probe: `benchmarks/framedcurve_vjp_scaling.py`.
  - Missing: recorded wall-time benchmark artifact.
- [ ] **N32** — `max_dense_jacobian_bytes` large-N scaling fixture.
  - Source: CPU 4 GB / GPU 256 MB defaults +
    `SIMSOPT_MAX_DENSE_JACOBIAN_BYTES_CPU/GPU` env overrides routed
    through `BackendPolicy` (`src/simsopt/backend/runtime.py`).
  - Missing: large-N scaling-limit fixture recording.

## Closed

- [x] **N21 — public-boundary single host materialization** *(closed
  2026-05-18)*
  - Design pivoted from "NaN sentinel" to "single public-boundary
    materialization". `_host_bool(success)` at
    `src/simsopt/geo/surfaceobjectives_jax.py:1902` is the
    *designated* single materialization site, not a leak.
  - Acceptance test:
    `tests/integration/test_single_stage_jax_cpu_reference.py::TestCompositeObjective::test_public_wrapper_dj_boundaries_allow_strict_transfer_guard_real_fixture`
    runs `BoozerResidualJAX.dJ()`, `IotasJAX.dJ()`,
    `NonQuasiSymmetricRatioJAX.dJ()` inside
    `jax.transfer_guard("disallow")` on a real Boozer fixture.
- [x] **N25 — `_grouped_field` JIT-static-`group_count`**. Probe
  artifact: `docs/grouped_field_distribution_probe_2026-05-18.md`.
- [x] **N29 — single-device `jax_gpu_parity`** documented; runtime
  emits info log for multi-device parity default. Real multi-GPU
  proof intentionally folded into N30.

## Goal-bullet rollup (round-3 doc lines 41-58)

- [x] M5 IFT adjoint zero host transfers under
  `transfer_guard("disallow")` — closed via **N21**
  (single-materialization design + real-fixture acceptance).
- [x] Adapter-layer Python `for`-over-jitted-kernel sites vectorized
  or jit-stable — met at source via N24 / N25 / N26.
- [x] `apply_jax_runtime_config()` is single source of truth for GPU
  memory env vars — met at source via N31.
- [x] `max_dense_jacobian_bytes` defaults platform-aware and
  env-overridable — met at source via N32.
- [x] Selective host-residency path with equal-gradient proof —
  landed via **N33** (production memory probe still pending).
- [x] N11 / N12 multi-GPU speedup measured on real hardware — closed
  via **N30** and `docs/jax_multi_gpu_proof_2026-05-19.md`.
- [ ] Documented checkpoint-and-restart OOM-recovery workflow —
  partial via **N34** (helper + docs + cache-observation test exist;
  worked restart example + real GPU proof missing).

## What's actually on the critical path

No multi-GPU hardware blocker remains. **N34**'s worked example is small
documentation work, not blocking. Everything else is either closed at source or
benchmark-artifact-only closure paperwork.

## Provenance

- Reviewed branch: `gpu-purity-stage2-20260405` at HEAD `4da847c72`.
- Source-state cross-checks performed against
  `src/simsopt/backend/runtime.py`,
  `src/simsopt/geo/boozersurface_jax.py`,
  `src/simsopt/geo/surfaceobjectives_jax.py`,
  `src/simsopt/geo/framedcurve_jax.py`,
  `src/simsopt/field/biotsavart_jax_backend.py`,
  `src/simsopt/jax_core/biotsavart.py`,
  `src/simsopt/_core/jax_host_boundary.py`.
- Closeout table re-used from
  `docs/jax_native_round3_performance_todos_2026-05-18.md:1097-1108`
  (revised by commit `0957ed5d7`).
- Refresh trigger: commits `2a42bda09..4da847c72`, notably
  `deefbb9bc` ("fix: keep Boozer adjoints transfer-clean"),
  `cbc0cf0de` ("fix: centralize JAX runtime policy"),
  `bd381d5b4` ("fix: vectorize grouped Biot-Savart paths"),
  `3cd654e7a` ("docs: harden Perlmutter GPU test plan").
