# JAX-Native Round-3 — Curated TODOs (2026-05-18)

Derived from `docs/jax_native_round3_performance_todos_2026-05-18.md`
after cross-validating each item against the source. Items the round-3
doc lists as "confirmed" but whose source-level work has already
landed are reclassified here as either "evidence-only" or "closed".

## Critical path

Items whose absence blocks declaring the JAX GPU port delivered.

- [ ] **N21 — eliminate residual host pull on the M5 IFT adjoint path**
  - Source residual: `src/simsopt/geo/surfaceobjectives_jax.py:1895`
    still does `if not bool(np.asarray(success)):` inside
    `_checked_boozer_linear_solve`. Trips
    `jax.transfer_guard("disallow")` over `BoozerResidualJAX.dJ`,
    `IotasJAX.dJ`, `NonQuasiSymmetricRatioJAX.dJ`.
  - Action: hoist the success-check to the public wrapper boundary so
    exactly one host scalar is materialized per outer iteration.
  - Acceptance: `transfer_guard("disallow")` sweep over the helper
    region passes on a real M5 fixture.

- [ ] **N30 — multi-GPU speedup proof for N11 + N12 sharding** *(hardware-gated)*
  - Prerequisites:
    - [ ] Resolve Runpod CUDA toolchain mismatch
      (`docs/source/jax_gpu_setup.rst:421-466`): rebuild `jaxlib`
      against host CUDA or pin a known-good jaxlib for H100.
    - [ ] Multi-GPU hardware (≥ 4 devices in one box).
  - Action:
    - [ ] 1 / 2 / 4-GPU wall-time + HBM-peak + HLO-collective-bytes
      sweep on `integral_BdotN_surface_sharded`
      (`src/simsopt/jax_core/integral_bdotn.py:240`).
    - [ ] Same sweep on seed-batch scoring
      (`src/simsopt/geo/surfaceobjectives_jax.py:5088`).
    - [ ] Write `docs/jax_multi_gpu_proof_2026-XX-XX.md`.
  - Acceptance: > 1.5× at 2 GPUs, > 2.5× at 4 GPUs vs 1-GPU baseline,
    parity preserved on `benchmarks/single_stage_init_parity.py`.

## Source-open (partial)

Real code or example work remaining, beyond pure measurement.

- [ ] **N33 — selective host-residency, end-to-end proof**
  - Done: `linearization_residency={"device","host"}` accepted; dense
    LS factors placed on CPU and restaged for runtime solve callbacks.
    Covered by
    `tests/geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXClass::test_linearization_residency_host_places_dense_factors_on_cpu`.
  - Action:
    - [ ] Equal-gradient two-solver proof — host-resident factors
      yield byte-equivalent (or tolerance-bounded) gradient vs
      device-resident.
    - [ ] Memory probe at production-relevant N.

- [ ] **N34 — checkpoint/restart workflow, worked example**
  - Done: `with_cpu_device_for_construction()` exported; GPU OOM docs
    prescribe checkpoint/restart and reject transparent
    compiled-JIT retargeting.
  - Action:
    - [ ] Worked command-level restart example (script + recorded run).
    - [ ] CPU-pinned + GPU-routed two-instance compile-cache proof.

## Hardware-gated (CPU correctness already in)

Source state is what it should be on CPU; the missing artifact requires
real GPU access.

- [ ] **N27** — CUDA TF32 speedup comparison for parity-mode
  `matmul_precision`. Source pinned `highest` for parity modes; needs
  real CUDA timings.
- [ ] **N28** — Real CUDA pre-import subprocess matrix. Determinism
  validation lands in source; only import-smoke covers it today.
- [ ] **N31** — Real GPU allocator log + memory-limit proof. Runtime
  owns the env policy and `SIMSOPT_*` overrides; needs an actual
  allocator-pressure recording.

## Evidence-only (source complete, no code change)

Goal is satisfied at source level; only telemetry / benchmark
artifacts are missing.

- [ ] **N22** — `jax.lax.while_loop` manual-LS path: benchmark
  improvement not recorded.
- [ ] **N23** — Direct-coil scalar pulls: transfer-count counter +
  real wrapper sweep not recorded.
- [ ] **N24** — `_per_coil_unit_field` (`src/simsopt/field/biotsavart_jax_backend.py:180-211`):
  vmap landed; scaling benchmark / HLO report not recorded.
- [ ] **N26** — Framed-curve VJP fan-out: collapsed to 9 sites
  (target was <22); dedicated wall-time benchmark not recorded.
- [ ] **N32** — `max_dense_jacobian_bytes`: CPU 4 GB / GPU 256 MB
  defaults + `SIMSOPT_MAX_DENSE_JACOBIAN_BYTES_CPU/GPU` env overrides
  already live at `src/simsopt/backend/runtime.py:58-162, 1348-1362`
  and routed through `BackendPolicy`; large-N scaling-limit fixture
  not recorded.

## Closed

- [x] **N25** — `_grouped_field` JIT-static-`group_count`
  (`src/simsopt/jax_core/biotsavart.py:734-740`). Probe artifact:
  `docs/grouped_field_distribution_probe_2026-05-18.md`.
- [x] **N29** — Single-device `jax_gpu_parity` documented; runtime
  emits info log for multi-device parity default. Real multi-GPU
  proof intentionally folded into N30.

## Goal-bullet rollup (round-3 doc lines 41-58)

- [ ] M5 IFT adjoint zero host transfers under
  `transfer_guard("disallow")` — open via **N21**.
- [x] Adapter-layer Python `for`-over-jitted-kernel sites vectorized
  or jit-stable — met at source via N24 / N25 / N26.
- [x] `apply_jax_runtime_config()` is single source of truth for GPU
  memory env vars — met at source via N31.
- [x] `max_dense_jacobian_bytes` defaults platform-aware and
  env-overridable — met at source via N32.
- [ ] Selective host-residency path for warm-start factors — partial
  via **N33** (path exists; proof missing).
- [ ] N11 / N12 multi-GPU speedup measured on real hardware — open via
  **N30**, multi-GPU + CUDA-toolchain prerequisites.
- [ ] Documented checkpoint-and-restart OOM-recovery workflow —
  partial via **N34** (helper + docs exist; worked example missing).

## Provenance

- Reviewed branch: `gpu-purity-stage2-20260405` at HEAD `2a42bda09`.
- Source-state cross-checks performed against
  `src/simsopt/backend/runtime.py`,
  `src/simsopt/geo/boozersurface_jax.py`,
  `src/simsopt/geo/surfaceobjectives_jax.py`,
  `src/simsopt/geo/framedcurve_jax.py`,
  `src/simsopt/field/biotsavart_jax_backend.py`,
  `src/simsopt/jax_core/biotsavart.py`.
- Closeout table re-used from
  `docs/jax_native_round3_performance_todos_2026-05-18.md:1097-1107`.
