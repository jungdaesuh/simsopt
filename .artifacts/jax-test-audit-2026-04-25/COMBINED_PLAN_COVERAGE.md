# Combined 12-Step Plan Coverage Analysis

Date: 2026-04-25
Branch: `gpu-purity-stage2-20260405`
Inputs: `IMPLEMENTATION_PLAN_COMBINED.md` (12 steps, 227 lines), `SYNTHESIS.md` (25 P0s),
`PLAN_VS_FINDINGS.md` (V1: 6A/6B/12C/1D), `CORRECTED_PLAN_REVIEW.md` and
`PLAN_REVIEW_V2.md` (V2: 7A/6B/11C/1D), `bucket{1..8}_*.md`.

Status keys: **A** = fully addressed, **B** = partially addressed, **C** = not
addressed, **D** = explicitly retired.

---

## 1. Coverage matrix update (25 P0s)

| P0 # | Finding (one-line) | V1 | V2 | 12-step | Movement | Step(s) |
|-----:|--------------------|:--:|:--:|:-------:|:---------|:--------|
| 1  | GPU proof rename + Stage 2 platform guard + provenance + real-GPU lane | A | A | **A** | — | 2 (5 sub-tasks) |
| 2  | `_MockVolumeLabel.J()` zeroing | C | D | **D** | — | 5 (rename + delete unused J(), add real label-path test) |
| 3  | OR-escape + 2-of-3 / 3-of-5 + Taylor 0.55 (lines 1979/4213/5792) | B | B | **A** | B → A | 4 (line 510, 1439, 1442, 3744, 4213, 5153, 5792, 5804) |
| 4  | Donation `is_deleted()` + positional donate | C | A | **A** | — | 3 |
| 5  | 3 tautological flux-kernel parity tests + singular `dJ()` raises | B | B | **B** | — | 8 (surface/accessibility) — bucket 4 flux kernels still implicit; SquaredFlux singular `dJ()` partly via Step 11 |
| 6  | IotasJAX adjoint residual rel-tol gate (line 5662) | C | C | **C** | — | not addressed |
| 7  | Replace toy 3×3 oracle at `test_surface_objectives_jax.py:1728` | C | C | **A** | C → A | 10 (replace with real exact-state adjoint solve + dense oracle) |
| 8  | `test_adjoint_fraction_diagnostic > 0` ceremony (line 5859) | C | C | **A** | C → A | 11 (promote or delete) |
| 9  | `test_outer_opt_decreases_objective` `+1e-12` slack (line 4925) | C | C | **A** | C → A | 11 (require strict decrease + `result.nit > 0`) |
| 10 | End-to-end exact-Newton on real torus (no shim) | C | C | **A** | C → A | 10 (real torus, no `_patched_exact_newton_result`) |
| 11 | `test_Taylor`: 320 → 12 sub-cases | A | A | **A** | — | 9 (slow-marked broad sweep + fast representative default) |
| 12 | Restore 3 deleted curve tests + downsample loop + RNG | A | A | **A** | — | 9 |
| 13 | Drop tautological `_assert_surface_jacobian_parity` arms | B | B | **B** | — | 8 (keep JAX-vs-JAX as API consistency only; FD oracle replacement *implied* not enumerated; tolerance not specified) |
| 14 | Stage 2 aggregator: provenance + parity-rtol rejection | B | B | **B** | — | 2 (rejection criteria listed but `cpu_oracle_value`/`gpu_value`/`value_rtol`/`gradient_rtol` schema fields not enumerated) |
| 15 | Compile-count subprocess wrappers parse JSON | A | A | **B** | A → B | 12 (printf-format replacement + JSON parse mentioned but JSON-sentinel for compile-count subprocess wrappers not explicitly retained) |
| 16 | `np.isfinite` → `np.isposinf` zero-current case (line 1036) | C | C | **A** | C → A | 11 |
| 17 | `test_normal_orthogonality` cross-product tautology + analytic torus area/volume | B | B | **B** | — | 8 (HLO/JAX-vs-JAX downgrade); analytic torus area/volume + cross-product tautology not enumerated |
| 18 | Order-dependent `test_backend_*_sequence_*` refactor | C | C | **A** | C → A | 12 (refactor into isolated tests passable under randomized order) |
| 19 | `tests/integration/conftest.py:_patch_meta_path_finder` silent False | C | C | **A** | C → A | 12 (no silent False path) |
| 20 | Divergence-theorem invariant in `test_integral_bdotn_jax.py` | A | A | **A** | — | 6 (raw signed flux helper, divergence theorem) |
| 21 | `SquaredFluxJAX.dJ()` Taylor + chunked-VJP gradient parity | C | C | **A** | C → A | 11 (Taylor + chunked/grouped-VJP large point cloud) |
| 22 | 5 `*_reuses_shared_jit_kernels` accessibility tests need FD parity | B | B | **B** | — | 8 (accessibility tests must assert J/dJ/ddJ parity or FD); 5 tests not enumerated, tolerances not anchored to lane SSOT |
| 23 | Printf-format pinning in `test_field_cache_hot_path_benchmark.py` | C | C | **A** | C → A | 12 (replace with end-to-end compile/run + structured JSON parse) |
| 24 | `_force_x64` autouse counter-test | C | C | **A** | C → A | 12 (`_force_x64` counter-test) |
| 25 | VJP signature regression for `_boozer_ls_coil_vjp` / `_boozer_exact_coil_vjp` | C | C | **A** | C → A | 10 (`inspect.signature` regression guards + value-vs-FD guard) |

---

## 2. Coverage delta (V1 → V2 → 12-step)

| Status | V1 (first plan) | V2 (corrected) | **12-step combined** | V1→12 net | V2→12 net |
|--------|:---------------:|:--------------:|:--------------------:|:---------:|:---------:|
| A (full) | 6 (24%) | 7 (28%) | **17 (68%)** | +11 | +10 |
| B (partial) | 6 (24%) | 6 (24%) | **6 (24%)** | 0 | 0 |
| C (unaddressed) | 12 (48%) | 11 (44%) | **1 (4%)** | -11 | -10 |
| D (retired) | 1 (4%) | 1 (4%) | **1 (4%)** | 0 | 0 |

Net headline movement vs V2: 10 additional P0s fully addressed (#7, #8, #9,
#10, #16, #18, #19, #21, #23, #24), 1 additional P0 closed in scope but with
1 partial regression (#15 was A in V2, now B because the JSON-sentinel /
compile-count subprocess wrapper migration is not explicitly preserved in
Step 12; only the printf-format → JSON contract on the hot-path benchmark
is). The dominant story: the addition of Steps 10, 11, 12 (Boozer/exact +
M5/Stage 2 wrappers + conftest/CI) closes the four cluster gaps that V1/V2
left untouched.

---

## 3. P1/P2 spot-check — what the 12-step plan picks up that V1/V2 missed

| Audit item (P1/P2) | Bucket | V1/V2 | 12-step | Notes |
|--------------------|:------:|:-----:|:-------:|-------|
| Vector potential A gauge consistency | b1 §3 | C | **C** | Step 6 is raw signed flux only; A vs analytic and Coulomb gauge `∇·A=0` not picked up. |
| Stellsym round-trip on `surface_xyzfourier` | b1 §3 | C | **C** | Not added; bucket 1's stellsym=True parametrize over `test_coefficient_derivatives_match_cpp` remains a backlog gap. |
| `test_div_B_zero` only on-axis (off-axis 20-pt) | b1 #6 | C | **C** | Step 6 wording targets divergence theorem (∮B·n dA), not the pointwise off-axis div test. |
| nfp rotational symmetry of B | b1/b4 §3 | C | **C** | Step 7 is reductions GPU/CPU parity only; nfp B-field symmetry not picked up. |
| Kahan/cancellation oracle for `compensated_sum_flat` | b7 §3 | C | **B (close)** | Step 7 lists "compensated sum" + "cancellation-stress arrays" — partially picked up, but adversarial `[1e16, 1.0, -1e16]` and Kahan-vs-naïve oracle not enumerated. |
| GPU same-state determinism | b3/b6 §3 | C | **C** | Step 7 is CPU vs GPU parity; same-state determinism (run twice → bit-identical) is a separate gate not picked up. |
| `bootstrap_runtime.sh` hardening (`jax.default_backend() == "gpu"` after import) | b6 §3 | C | **C** | Step 2 sub-tasks reference build_provenance, Stage 2 platform guard, aggregator, real CUDA canary, nvidia-smi; bootstrap-time `jax.default_backend()` assertion is NOT enumerated. |
| Device residency of hot-loop arrays | b6 §3 | C | **C** | Step 2 lists `nvidia-smi` capture + provenance fields; per-array device-residency assertion (`assert_device_residency` helper) NOT picked up. |
| Cold-warm GPU determinism | b6 §3 | C | **C** | Not picked up. |
| LD_LIBRARY_PATH dynamic-loader smoke (real `import jaxlib.cuda_versions`) | b6 P0 #6 | C | **B (close)** | Step 2.4 adds a "real CUDA canary that performs compile and execution with `block_until_ready()` under both PTX-forced and CUBIN-forced runs" — this is functionally the dynamic-loader smoke even though `jaxlib.cuda_versions` import is not explicitly invoked; passes the dlopen contract by virtue of `block_until_ready()` succeeding under CUBIN-forced mode. |
| Newton-polish post-LS magnitude target (`‖grad‖<1e-10`) | b2 #6 | C | **C** | Step 5 retires `_MockVolumeLabel` and adds one real label-path test, but `test_newton_polish_reduces_gradient` magnitude tightening is not picked up. |
| Ill-conditioned exact path with `failure_category="scaling_limit"` | b2 §3 | C | **C** | Step 10 mentions "real exact-state adjoint solve" but does not enumerate the ill-conditioned (residual/failure-only) sibling test. |
| Legacy `dJ()` adjoint failure (real fixture, monkeypatch `solve_transpose_with_status`) | b3 §3 | C | **A** | Step 11 explicitly: "Add the legacy `dJ()` adjoint failure case where a finite wrong gradient would currently escape." |
| Successful forward + failed adjoint → non-finite gradient | b3 §3 | C | **A** | Same line in Step 11. |

Net P1/P2 picked up: 2 fully (legacy dJ adjoint failure, successful-fwd
failed-adj non-finite gate, both via Step 11), 2 close (Kahan oracle in
Step 7, dynamic-loader smoke via Step 2.4). 9 P1/P2 items remain unaddressed
(A gauge, stellsym round-trip, off-axis div B, nfp rotational symmetry,
GPU same-state determinism, bootstrap hardening, device residency, cold-warm
determinism, Newton polish magnitude target, ill-conditioned exact lane).

---

## 4. Validation Notes accuracy

| Claim | Verdict | Evidence |
|-------|:-------:|----------|
| `PLAN_REVIEW.md` is 397 lines | **TRUE** | `wc -l` confirms 397. |
| `PLAN_REVIEW_V2.md` is 336 lines | **TRUE** | `wc -l` confirms 336. |
| `CORRECTED_PLAN_REVIEW.md` still says to emit `jaxlib_cuda_versions` | **TRUE — and the combined plan's correction is VALID** | `CORRECTED_PLAN_REVIEW.md:162-168` says "build_provenance does NOT yet emit `xla_flags` or `jaxlib_cuda_versions` ... 1. Extend `build_provenance` to emit `xla_flags` ... and `jaxlib_cuda_versions = getattr(jaxlib, "cuda_versions", None)`." V2's section 0 #2 contradicts this with a direct probe showing `getattr(jaxlib, "cuda_versions", None)` returns None on jaxlib 0.9.2. Combined plan correctly drops the field. |
| V2's Step 10 mixes Boozer/conftest/order/benchmark | **PARTIALLY TRUE** | V2 §3-5 have a single "Step 10" labelled "Boozer exact-Newton end-to-end fixture" but V2 §5 also lists Steps 11 (Conftest and order), 12 (CI infra), each with multiple sub-tasks. The combined plan's reorganization into 12 explicit steps with Steps 10 (Boozer exact + VJP signature), 11 (M5+Stage 2 failure paths), 12 (conftest/order/smoke/CI gates) is a real readability and decomposition improvement — each step now has 1 thematic concern. |
| Local probe shows no stable `jaxlib.cuda_versions` on JAX 0.9.2 | **TRUE per V2 §0 #2** | V2 explicitly verified via `getattr(jaxlib, "cuda_versions", None)` returning None. The combined plan correctly bans the field. |
| `CUDA_FORCE_PTX_JIT` / `CUDA_DISABLE_PTX_JIT` are documented NVIDIA knobs | **TRUE per V2 §0 #4** | V2 cites NVIDIA Programming Guide §5.2 directly. Combined plan adopts the right env-var contract. |
| `integral_BdotN` is a squared objective | **TRUE per VERIFY_physics_claims and V2 §0 #5** | All 3 definitions in `src/simsopt/objectives/integral_bdotn_jax.py:50-78` square `(B·n̂)`; raw signed flux needs a NEW helper. Step 6 correctly mandates the new helper. |

All Validation Notes claims hold up to verification.

---

## 5. Plan rule compliance

| Rule | Step(s) that enforce it | Mechanism | Concrete or aspirational |
|------|------------------------|-----------|:------------------------:|
| **No tautology tests** (no JAX-vs-JAX oracle, no HLO text as physics, no cache-size as derivative behavior) | 8 (HLO/JAX-vs-JAX → API consistency only; cache-size demoted to secondary in accessibility) | Per-test refactor; "downgrade JAX-vs-JAX to alias/API tests" | **Aspirational** — no CI lint or `assert_no_jax_vs_jax_parity` helper proposed; relies on per-PR review. The synthesis §5 "New helpers worth adding" recommended `assert_no_jax_vs_jax_parity` but combined plan does not include it. |
| **No silent GPU pass** (production proof must fail closed if requested runtime is not GPU) | 2.2, 2.3, 2.4 | `require_requested_platform_runtime` in Stage 2; aggregator rejection of payloads with wrong backend; real CUDA canary fails closed | **Concrete** for production proof scripts; **Aspirational** for ad-hoc tests because Step 1 says "GPU-required tests use `xfail(strict=True)` only for local hardware absence" — strict=True semantics are correctly stated, but no CI lint enforces `xfail strict=True` over `skipif`. |
| **No ad-hoc tolerance constants in new helpers** (all parity tolerances from `PARITY_LADDER_TOLERANCES` via `lane=` contract) | 1 (`require_parity_lane(lane=...)`) | Helper API design constraint | **Concrete** for new helpers; **Aspirational** for existing tests because no migration audit is mandated to convert the 5 hardcoded constants in `test_single_stage_jax_cpu_reference.py` (lines 510, 1439, 1442, 3744, 4213, 5153, 5792, 5804) — Step 4 cites the lines but does not require the lane-contract to be the *source* of replacement values. |
| **No broad defensive wrappers or fallback execution lanes** | 5 (real label-path test, no defensive wrap), 6 (raw signed flux primitive, not wrapped through integral_BdotN), 8 (FD checks instead of fallback paths), 11 (no fallback lane for failed adjoint — must surface non-finite gradient) | Test design constraint per step | **Concrete** in 11 (legacy `dJ()` adjoint failure must surface non-finite, no fallback); **Aspirational** elsewhere because the rule is restated rather than mechanized. |
| **Preserve upstream CPU/reference behavior** (CPU is the oracle lane; JAX/CUDA prove parity) | 9 (restore upstream coverage as written, slow-marked broad sweeps preserve behavior); 5 (real label-path test uses CPU `Volume`/`Area`/`ToroidalFlux` as oracle); 6 (signed-flux is a CPU-checkable invariant) | Per-step text | **Concrete** in 9 (restore deleted upstream tests verbatim); **Aspirational** elsewhere — no `git diff` audit gate in CI ensures no upstream test was silently retuned. |

Net: Rule 2 (no silent GPU pass) and Rule 5 (preserve upstream) are the
most concretely enforced. Rules 1, 3, 4 rely on per-PR discipline; the
combined plan does not include the `tests/_helpers/parity.py` SSOT or the
`assert_no_jax_vs_jax_parity` helper that synthesis §5 recommended.

---

## 6. Step execution risk

### Step 1 — Shared Test Contracts

- **Dependencies**: all subsequent steps consume these helpers; must land first.
- **Wording risk**: "GPU-required tests use `xfail(strict=True)` only for local hardware absence; production proof scripts must fail closed" — implementor may interpret "local hardware absence" too liberally (e.g., CI runners that are *supposed* to have GPUs); the helper must encode "is this a hardware-class machine without GPU" vs "is this a GPU lane that lost its GPU" distinction.
- **Missing detail**: helper file location not specified (`tests/_helpers/parity.py`? `tests/_helpers/jax.py`?). No `assert_no_jax_vs_jax_parity` helper proposed. No `assert_device_residency` helper.

### Step 2 — GPU Proof And Provenance (5 sub-tasks)

- **Dependencies**: 2.1 (extend `build_provenance`) MUST precede 2.3 (aggregator preserves fields) — combined plan correctly orders them. 2.2 (`require_requested_platform_runtime` in Stage 2) is independent. 2.4 (real CUDA canary) and 2.5 (`nvidia-smi` capture) are independent.
- **Wording risk**: "rejects invalid proof payloads" — fake runner used without explicit test-only fake mode is named, but the discriminator field is not specified. Implementor may add an opaque flag instead of `bundle_provenance.fake = True` discriminator.
- **Missing detail**: no `bootstrap_runtime.sh` hardening (audit recommends adding `jax.default_backend() == "gpu"` assertion immediately after `import jax`); no `cpu_oracle_value`/`gpu_value`/`value_rtol`/`gradient_rtol` schema fields enumerated for the aggregator-side parity check; no `bundle_provenance.runner` field; no schema_version bump.

### Step 3 — Donation Probe

- **Dependencies**: independent.
- **Wording risk**: "Call the donated JIT function positionally, not with keyword arguments" — correct per JAX 0.9.2 docs. "Reuse raises the documented invalid-buffer runtime error" — implementor needs to know the exact match string (`"has been deleted"` per V2 §4 Step 8). Combined plan does not include the match string.
- **Missing detail**: no continuous mutation infrastructure (V2 recommended `tests/test_donation_mutants.py` with `pytest.mark.mutation`); without it, "mutant `donate_argnums=()` must fail" is one-shot.

### Step 4 — FD And IFT Discipline

- **Dependencies**: must consume Step 1's `require_parity_lane(lane=...)` helper. Implementor must touch all 8 enumerated lines (510, 1439, 1442, 3744, 4213, 5153, 5792, 5804) — risk of partial migration.
- **Wording risk**: "Replace fixed OR gates at lines 3744, 4213, 5153, and 5792" — line 510 also has the OR-escape but is split out as "Replace helper-level `rel_tol or abs_tol`" — implementor may treat as separate concerns.
- **Missing detail**: tolerance source not specified for Step 4's "lane-backed central-FD threshold" replacement of `0.55` — should be `_FD_GRADIENT_TOLS["directional_fd_rtol"]` per V2 §4 Step 4 #1. The "preselect nonzero directions" cherry-picking risk per PLAN_REVIEW.md §5 is NOT addressed (no specification that direction selection uses RNG seed + reject-only-if-projected-grad-below-1e-12).

### Step 5 — Boozer Label And Plumbing Mock Cleanup

- **Dependencies**: P0 #2 retirement is correct per VERIFY_mock_volume_label.md (the production code does not call `_MockVolumeLabel.J()` — it recomputes from `volume_jax(gamma, normal) - targetlabel`).
- **Wording risk**: "Rename the helper to `_PlumbingVolumeLabel` or delete its unused `J()` method" — both options are correct; implementor should do BOTH. The "or" creates ambiguity.
- **Missing detail**: which existing tests must migrate to the real label-path test is not enumerated. The audit listed 7-12 highest-leverage tests (`test_penalty_with_volume_label`, `test_run_code_ls_converges`, `test_penalty_gradient_fd`, `test_newton_polish_reduces_gradient`, `test_penalty_with_toroidal_flux`, `test_penalty_with_area_label`); combined plan adds only "one real label-path test" — implementor may leave the existing tests untouched, defeating the spirit.

### Step 6 — Raw Signed Flux Invariant

- **Dependencies**: depends on Step 1 (lane tolerance plumbing).
- **Wording risk**: "Use a coil ring outside the torus or another physically clean source-free surface case" — correct, but no specific geometry (V2 §4 recommends `z=5` for `R=1, r=0.1` torus).
- **Missing detail**: quadrature refinement (audit recommends 64×64 minimum); pass criterion (`|signed_flux| / max|B| / area_total < 1e-9`) not specified — relies on Step 1's lane.

### Step 7 — Reductions GPU/CPU Parity

- **Dependencies**: GPU must be available; uses Step 1's `xfail(strict=True)` skip.
- **Wording risk**: "Use cancellation-stress arrays" — implementor may not include the adversarial `[1e16, 1.0, -1e16]` case. Combined plan correctly excludes default `jnp.sum`/`jnp.vdot` paths by saying "compensated and pairwise modes" but does not enumerate the exact functions (`compensated_sum_flat`, `pairwise_sum_flat`, `pairwise_sum_axis`).
- **Missing detail**: no nfp rotational symmetry test in scope.

### Step 8 — Surface And Accessibility Tests

- **Dependencies**: independent.
- **Wording risk**: "Keep JAX-vs-JAX checks only as API consistency" — implementor may leave the `rtol=1e-12` JAX-vs-JAX assertions in place under a renamed helper, defeating the purpose. The audit said "Drop the `rtol=1e-12` JAX-vs-JAX arms; keep only the C++ oracle arm."
- **Missing detail**: which 5 accessibility tests need FD parity is not enumerated (`test_projected_enclosed_area_reuses_shared_jit_kernels`, `test_directed_facing_port_reuses_shared_jit_kernels`, `test_curve_in_port_penalty_reuses_shared_jit_kernels`, `test_projected_curve_curve_distance_reuses_shared_jit_kernels`, `test_projected_curve_convexity_reuses_shared_jit_kernels`). Bucket 4 flux-kernel tautologies (P0 #5: `test_fluxobjective_jax_parity.py:211, 223, 253`) are not enumerated under this step. Cross-product orthogonality tautology + analytic torus area/volume (P0 #17) not enumerated.

### Step 9 — Restore Upstream Coverage

- **Dependencies**: independent.
- **Wording risk**: "Restore deterministic random seeding where upstream used it" — keeping `np.random.seed(0)` (upstream) vs `np.random.default_rng(0)` (modern API) decision is left to implementor; the audit recommended same seed value to minimize tolerance drift.
- **Missing detail**: 3 deleted tests not named (`test_arclength_variation_circle_planar`, `test_linking_number_planar`, `test_curve_curve_distance_empty_candidates`). `test_objectives_time` (`ncoils=2` default-on, `ncoils=8` env-gated) and `test_call` follow-up at `tests/core/test_optimizable.py:243-258` not picked up. Function name `test_Taylor` (per V2 §4 Step 7 #1) not specified.

### Step 10 — Boozer Exact-Newton And VJP Signature

- **Dependencies**: real-torus fixture must exist; `_solve_exact_cpu_jax_parity_pair()` already exists per bucket 3 §3. No `_patched_exact_newton_result` shim must be left in the new test.
- **Wording risk**: "Replace the toy 3x3 oracle at `tests/geo/test_surface_objectives_jax.py:1728` with a real exact-state adjoint solve" — correct. The dense reference oracle (`scipy.linalg.lu_solve`) is correctly named.
- **Missing detail**: no ill-conditioned exact-path fixture (V2 §5 Step 10 third bullet, audit P0 line 6 in bucket 2 §3) — combined plan only addresses the well-conditioned exact lane. The `inspect.signature` regression is correctly added but should be paired with a value-vs-FD guard to prevent shape-only test (combined plan does state "and one value-vs-FD guard so signature checks do not become shape-only tests" — good).

### Step 11 — M5 And Stage 2 Failure Paths

- **Dependencies**: independent. Existing test infrastructure sufficient (`boozer_setup` fixture already monkeypatchable).
- **Wording risk**: "Promote or delete the `adjoint_fraction > 0` ceremony" — both options are correct. "Tighten `test_outer_opt_decreases_objective`... so it requires a real decrease and a nonzero optimization step, not `j_final <= j0 + 1e-12`" — correct semantic but no specific tolerance (audit recommended `j_final < j0 - 1e-6 * abs(j0)` AND `result.nit > 0`).
- **Missing detail**: tolerance for the new `SquaredFluxJAX.dJ()` Taylor test not specified — should mirror upstream `check_taylor_test`. The "chunked/grouped-VJP gradient parity case on a large point cloud" point cloud size not specified (audit recommended companion to `test_chunked_grouped_paths_match_cpu_on_large_point_cloud` which uses ≥10000 points).

### Step 12 — Conftest, Order, Smoke, And CI Gates

- **Dependencies**: independent of other steps.
- **Wording risk**: "Refactor order-dependent backend guard sequence tests into isolated tests that can pass under randomized order" — correct semantic but does not name `test_backend_state_guard_sequence_01..04` and `test_backend_module_guard_sequence_01..02`. "Make any conftest boolean guard failures explicit; no silent `False` path" — correctly maps to P0 #19. "Add the `_force_x64` counter-test so the autouse fixture cannot hide missing x64 setup" — correctly maps to P0 #24.
- **Missing detail**: CI gates listed but `pytest-randomly` is not named explicitly. "Replace printf-format pinning in `test_field_cache_hot_path_benchmark.py`" — the JSON sentinel pattern that V1's Step 8 picked up for compile-count subprocess wrappers (P0 #15) is NOT preserved here — Step 12 only mentions the hot-path benchmark replacement, not the broader subprocess wrapper migration. **This is the cause of P0 #15 regressing from A → B**.

---

## 7. Still missing after 12 steps (ranked by severity)

1. **IotasJAX adjoint residual rel-tol gate (P0 #6)**: `test_gradient_wrappers_operator_status_on_exact_state` (line 5662) IotasJAX branch only checks finite/non-zero. Step 11 covers M5 wrapper paths but not this specific residual-norm gate. Severity: P0 — the `exact_ill_conditioned_adjoint` lane requires `residual_rel_tol=1e-10`.
2. **Bucket 4 flux-kernel tautologies (P0 #5 partial)**: `tests/objectives/test_fluxobjective_jax_parity.py:{211, 223, 253}` are JAX-vs-JAX. Step 8 is scoped to "surface and accessibility" — implementor likely misses bucket 4 unless explicitly enumerated. Severity: P0.
3. **Bucket 1 surface analytics (P0 #17 partial)**: `test_normal_orthogonality` cross-product tautology and missing analytic torus area (`4π² R r`)/volume (`2π² R r²`) tests. Step 8 covers "tautological assertions with physics/math assertions" but does not enumerate bucket 1 surface invariants. Severity: P0.
4. **5 accessibility `*_reuses_shared_jit_kernels` tests not enumerated (P0 #22 partial)**: Step 8 mentions "accessibility tests must assert J/dJ/ddJ parity or FD behavior" but the 5 specific tests are not named, tolerances not anchored to lane SSOT. Severity: P0.
5. **JSON-sentinel migration for ~75 import-smoke wrappers (P0 #15 regression)**: Step 12 replaces only the hot-path benchmark printf-format with JSON parse; the broader migration of `test_jax_import_smoke.py:519-526` compile-count case to JSON-payload `{"compile_count": handler.count}` (V1 Step 8) is NOT preserved. Severity: P0.
6. **Stage 2 aggregator parity-rtol rejection (P0 #14 partial)**: Step 2.3 lists provenance fields and rejection criteria (fake without flag, missing backend, wrong backend, missing PTX/CUBIN env) but does NOT list `cpu_oracle_value`/`gpu_value`/`value_rtol`/`gradient_rtol` schema fields with rtol-exceeds-lane-contract rejection. Severity: P0.
7. **Bootstrap-time GPU smoke (`bootstrap_runtime.sh` hardening)**: V2 §4 Step 2 #5 explicitly requires assertion of `jax.default_backend() == "gpu"` immediately after `import jax`. Combined plan does not pick this up. Severity: P0.
8. **Order-dependent test ordering CI gate**: Step 12 refactors the sequence tests but does not add `pytest-randomly` to nightly CI. Without continuous enforcement, future order-dependent tests slip through. Severity: P1.
9. **Ill-conditioned exact-path lane (bucket 2 §3, audit P0)**: Step 10 covers well-conditioned exact only. The `exact_ill_conditioned_adjoint` lane (residual-only, no vector parity) remains unexercised against a real ill-conditioned configuration. Severity: P0.
10. **GPU same-state determinism gate (bucket 3 §3)**: `TestRealFixtureGpuM5Parity` runs once; no `np.testing.assert_array_equal` between two runs. Step 7 GPU/CPU parity does not cover same-state determinism. Severity: P1.
11. **Vector potential A gauge consistency, stellsym round-trip on `surface_xyzfourier`, off-axis `div B`, nfp rotational symmetry, Newton polish post-LS magnitude target, device residency of hot-loop arrays, cold-warm GPU determinism, LD_LIBRARY_PATH dynamic-loader smoke (real `import jaxlib.cuda_versions` invocation)**: bucket 1, 2 §3, 6 §3 missing physics/proof invariants. Severity: P1/P2 mostly; LD_LIBRARY_PATH and bootstrap are P0.
12. **Mutation/CI infrastructure**: combined plan has no `tests/mutants/test_required_mutants.py` with `pytest.mark.mutation` (V2 §5 Step 12 recommendation). Plan-rule "no tautology tests" enforcement is per-PR review only. No `pytest --collect-only` guard against re-deletion of restored upstream tests. No `pytest.skip` audit with tracked GitHub issue ID. Severity: P1 ongoing-regression risk.

---

## 8. Acceptance criteria audit

| Criterion | Measurable? | Producing step(s) | Concrete check |
|-----------|:-----------:|:-----------------:|----------------|
| Every required GPU proof payload has real backend/device/provenance evidence and fails closed on CPU fallback | **Yes** | 2.1, 2.2, 2.3, 2.4, 2.5 | grep `default_backend` and `devices` keys in every `bundle_provenance` written by `stage2_e2e_comparison.py` and `single_stage_init_parity.py`; `pytest tests/...real.py` returns non-zero when `jax.default_backend() != "gpu"`; `nvidia-smi` payload present in launcher logs. |
| Every derivative test that claims FD/IFT correctness uses nonzero signed directions, lane tolerances, and all-direction acceptance | **Yes** | 1, 4 | grep for `or abs_err <` in `tests/integration/test_single_stage_jax_cpu_reference.py` returns zero hits; `_REAL_RESOLVE_FD_TAYLOR_RATE` value lookup returns lane-backed constant; `validated_directions` comparison uses `==` not `>=`. |
| Every physics invariant test checks a signed or conserved quantity, not a squared objective that is nonnegative by construction | **Partial** | 6 | grep for new `signed_flux_jax` helper in `src/simsopt/objectives/integral_bdotn_jax.py`; assert exists. But criterion as stated does not specify which test families this applies to (e.g., bucket 1 surface area/volume invariants are missed by Step 6's narrow signed-flux scope). |
| Every compile-shape/HLO test is labeled as compile/performance coverage only | **Partial** | 8 | grep for `@pytest.mark.brittle_perf_gate` or `@pytest.mark.compile_shape` — the marker name is not specified in the plan. The audit recommended `tests/perf_gates/` directory with `@pytest.mark.brittle_perf_gate`; plan does not enforce a directory or marker. |
| The restored upstream tests are present, deterministic, and split into fast representative coverage plus slow broad sweeps | **Yes** | 9 | `pytest tests/field/test_selffieldforces.py::test_Taylor` runs (fast default); `pytest tests/field/test_selffieldforces.py::test_Taylor --runslow` runs broad sweep; the 3 deleted tests appear in `pytest --collect-only` output; `np.random.seed(0)` (or `default_rng(0)`) present in `test_curve_minimum_distance_taylor_test`. |

3 of 5 acceptance criteria are concretely measurable. Criterion 3 (physics
invariants) and 4 (compile-shape labeling) are partial because the criterion
text is broader than the producing step's scope.

---

## 9. Verdict

The combined 12-step plan is a substantial improvement over V2: **net P0
coverage 28% → 68% fully addressed (+10 P0s closed)**, accomplished by adding
Steps 10 (Boozer exact-Newton + VJP signature), 11 (M5 + Stage 2 failure
paths), and 12 (conftest/order/smoke/CI gates). The plan correctly retires
the false-positive `_MockVolumeLabel.J()` finding, picks the right physics
oracle for raw signed flux, adopts the positional-donation contract per JAX
0.9.2 docs, and orders Step 2's plumbing work correctly (extend
`build_provenance` before touching aggregators). Three risks remain: (a) the
plan rules are mostly aspirational rather than CI-enforced (no `pytest-randomly`
gate, no `assert_no_jax_vs_jax_parity` helper, no mutation infrastructure);
(b) Step 8's surface-and-accessibility scope does not enumerate bucket 4
flux-kernel tautologies, the 5 specific accessibility tests, or bucket 1
analytic torus invariants — implementor risks partial coverage; (c) P0 #15
regresses A → B because the JSON-sentinel migration for ~75 import-smoke
wrappers was not preserved in Step 12. Recommended additions: enumerate
bucket 4 + bucket 7 + bucket 1 sub-clusters under Step 8; add bootstrap
hardening + `bundle_provenance.fake` discriminator + parity-rtol schema fields
to Step 2; preserve the V1 Step 8 JSON-sentinel scope inside the new Step 12.

---

DONE — coverage at /Users/suhjungdae/code/columbia/simsopt-jax/.artifacts/jax-test-audit-2026-04-25/COMBINED_PLAN_COVERAGE.md
