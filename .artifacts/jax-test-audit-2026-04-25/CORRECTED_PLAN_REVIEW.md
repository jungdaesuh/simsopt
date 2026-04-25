# Corrected Plan Review — 2026-04-25

Branch: `gpu-purity-stage2-20260405` · HEAD `42b68f33d`
Inputs: revised 9-step plan + `SYNTHESIS.md` + `PLAN_VS_FINDINGS.md` +
`PLAN_REVIEW.md` + spot-checked source (`benchmarks/validation_ladder_common.py:468`,
`benchmarks/stage2_e2e_comparison.py:33,59`,
`tests/integration/test_single_stage_jax_cpu_reference.py:1439,1441-1442,1979,5804`,
`benchmarks/hf_jobs/run_production_gpu_proof.sh:249-289`,
`src/simsopt/jax_core/reductions.py:13-16`).

---

## 1. Headline assessment

The corrected plan is a **net upgrade** over the first plan on three axes —
correctly retiring the false-positive `_MockVolumeLabel.J()` finding (per
`VERIFY_mock_volume_label.md`), explicitly re-using existing SSOT helpers
(`build_provenance`, `require_requested_platform_runtime`), and adopting
the right physics oracle for Step 6 (raw signed flux, not `integral_BdotN`).
The strict signed directional gate in Step 4, the positional-donation
contract in Step 8, and the real CUDA compile/run rung in Step 2 each close
distinct cherry-picking risks the first plan left open.

However, the corrections do not move any P0 from "C → A". They tighten the
language inside Steps 1-9 but still leave **all 12 unaddressed P0s
(#6, #7, #8, #9, #10, #16, #18, #19, #21, #23, #24, #25) untouched**. The
five-cluster gap analysis from `PLAN_REVIEW.md` (Boozer exact-Newton, M5
wrapper failure, Stage 2 sign/Taylor, conftest/order/smoke, CI infra) is
unchanged. The corrected plan needs Steps 10-14 (verbatim from
`PLAN_REVIEW.md` §4) to actually close the audit.

---

## 2. Coverage delta — 25 P0 findings

Status keys: **A** = fully addressed, **B** = partial, **C** = not addressed,
**D** = explicitly retired.

| P0 # | Finding (one-line) | First plan | Corrected plan | Movement |
|-----:|--------------------|:----------:|:--------------:|:---------|
| 1 | GPU proof rename + Stage 2 platform guard + provenance + real-GPU lane | A | A | — |
| 2 | `_MockVolumeLabel.J()` zeroing | C | **D** | first plan ignored; corrected plan correctly retires |
| 3 | OR-escape + 2-of-3 + Taylor 0.55 (lines 1979/4213/5792) | B | B | strengthened wording (strict signed); still misses line 1439 |
| 4 | Donation `is_deleted()` + positional donate | C | A | C → A |
| 5 | 3 tautological flux-kernel parity tests + singular `dJ()` | B | B | step 5 still scoped to surface/accessibility, flux kernels not enumerated |
| 6 | IotasJAX adjoint residual rel-tol gate (line 5662) | C | C | — |
| 7 | Replace toy 3×3 oracle with real `BoozerSurfaceJAX(boozer_type='exact')` | C | C | — |
| 8 | `test_adjoint_fraction_diagnostic > 0` ceremony (line 5859) | C | C | — |
| 9 | `test_outer_opt_decreases_objective` `+1e-12` slack (line 4925) | C | C | — |
| 10 | End-to-end exact-Newton on real torus (no `_patched_exact_newton_result`) | C | C | — |
| 11 | `test_Taylor`: 320 → 12 sub-cases | A | A | — |
| 12 | Restore 3 deleted curve tests + downsample loop + RNG | A | A | — |
| 13 | Drop tautological `_assert_surface_jacobian_parity` arms | B | B | "alias" framing replaced with "API/compile-shape only" — slight tightening but still no FD-oracle replacement spec |
| 14 | Stage 2 aggregator: provenance + parity-rtol rejection | B | B | "preserve full build_provenance" cited but aggregator-side rejection still not spec'd |
| 15 | Compile-count subprocess wrappers parse JSON | A | A | — |
| 16 | `np.isfinite` → `np.isposinf` zero-current case (line 1036) | C | C | — |
| 17 | `test_normal_orthogonality` cross-product tautology + analytic torus area/volume | B | B | step 5 says "API/compile-shape only" — bucket 1 surface invariants still not enumerated |
| 18 | Order-dependent `test_backend_*_sequence_*` refactor | C | C | — |
| 19 | `tests/integration/conftest.py:_patch_meta_path_finder` silent False | C | C | — |
| 20 | Divergence-theorem invariant in `test_integral_bdotn_jax.py` | A | A | — (signed-flux phrasing in step 6 is **better** than first plan's `integral_BdotN`) |
| 21 | `SquaredFluxJAX.dJ()` Taylor + chunked-VJP gradient parity | C | C | — |
| 22 | 5 `*_reuses_shared_jit_kernels` accessibility tests need FD parity | B | B | step 5 still doesn't enumerate the 5 tests; plan correctly says "do not count cache size as physics" |
| 23 | Printf-format pinning in `test_field_cache_hot_path_benchmark.py` | C | C | — |
| 24 | `_force_x64` autouse counter-test | C | C | — |
| 25 | VJP signature regression for `_boozer_ls_coil_vjp` / `_boozer_exact_coil_vjp` | C | C | — |

### Aggregate counts

| Status | First plan | Corrected plan | Net change |
|--------|:----------:|:--------------:|:----------:|
| A (full) | 6 | **7** | +1 |
| B (partial) | 6 | 6 | 0 |
| C (unaddressed) | 12 | **11** | -1 |
| D (retired) | 1 | 1 | 0 |

**Net movement: P0 #2 corrected (C → D, retired); P0 #4 closed (C → A,
positional donate). One P0 fully closed; one false-positive correctly
retired. Eleven P0s still C; six still B.**

---

## 3. Strengths of the corrected plan

- **Correctly retires P0 #2** (`_MockVolumeLabel.J()`). Confirmed by
  `VERIFY_mock_volume_label.md`: `boozersurface_jax.py` has zero executable
  `label.J()` calls; the penalty label is recomputed from
  `volume_jax(gamma, normal) - targetlabel` regardless of the mock. The
  user's pushback was correct and saved a wave of misdirected refactor
  work.
- **Re-uses existing SSOT** — `build_provenance` (lines 468-492 of
  `benchmarks/validation_ladder_common.py`) and
  `require_requested_platform_runtime`. The first plan implicitly
  re-invented these; the corrected plan explicitly reuses them, eliminating
  drift risk.
- **Positional donation contract** (Step 8). JAX 0.9.2 `donate_argnums`
  contract is positional-only; the corrected plan calls this out
  explicitly. The first plan said "mutant must fail" but did not
  specify the calling convention. A keyword-style donation call
  silently no-ops without raising.
- **Bug-fix #5: raw signed magnetic-flux conservation** (Step 6) — not
  `integral_BdotN`. `integral_BdotN` returns `∫|B·n|² dA / ∫|B|² dA`
  (always non-negative), which can never test Gauss's law `∮ B·n dA = 0`.
  The corrected plan picks the correct invariant.
- **Bug-fix #6: strict signed directional checks on preselected nonzero
  directions** (Step 4). Forces sign sensitivity that the first plan's
  "deterministic direction selection avoiding near-zero" left vulnerable
  to cherry-picking. Combined with separating "near-zero" tests into a
  separate suite, this closes the wrong-sign IFT blind spot.
- **Real CUDA compile/run proof** (Step 2). The first plan only added
  provenance fields. The corrected plan adds an end-to-end CUDA compile
  with `block_until_ready` synchronization — the load-bearing rung that
  would have caught the 2026-04-20 Runpod cubin incident.
- **Five mutants in Step 9** (vs. zero explicit mutants in the first
  plan). The CPU-fallback-under-cuda mutant directly validates Step 2;
  the wrong-sign-gradient mutant validates Step 4; the JAX-vs-JAX-only
  oracle mutant validates Step 5.

---

## 4. Remaining gaps (table — needs new Steps 10+)

| P0 # | One-line summary | Severity | Suggested step |
|-----:|------------------|:--------:|:---------------|
| 6 | IotasJAX adjoint residual rel-tol gate at line 5662 | P0 | **Step 10** (Boozer exact + adjoint) |
| 7 | Replace toy 3×3 oracle at `test_surface_objectives_jax.py:1728` with real `BoozerSurfaceJAX(boozer_type='exact')` + `scipy.linalg.lu_solve` | P0 | **Step 10** |
| 8 | `test_adjoint_fraction_diagnostic > 0` ceremony (line 5859) — promote or delete | P0 | **Step 12** (M5 wrappers) |
| 9 | `test_outer_opt_decreases_objective` `+1e-12` slack (line 4925) → require `< -1e-6 · |j0|` AND `nit > 0` | P0 | **Step 12** |
| 10 | End-to-end exact-Newton fixture on real torus (replace `_patched_exact_newton_result(jacobian=identity)` shim) | P0 | **Step 10** |
| 16 | `np.isfinite` → `np.isposinf` parity for zero-current singular case (line 1036) | P0 | **Step 12** |
| 18 | Order-dependent `test_backend_state_guard_sequence_*` / `test_backend_module_guard_sequence_*` refactor | P0 | **Step 13** (conftest/order) |
| 19 | `tests/integration/conftest.py:_patch_meta_path_finder` silent `return False` → `pytest.skip(...)` | P0 | **Step 13** |
| 21 | `SquaredFluxJAX.dJ()` Taylor test + chunked-VJP gradient parity on large point cloud | P0 | **Step 12** |
| 23 | `test_field_cache_hot_path_benchmark.py` printf-format pinning → end-to-end compile+JSON | P0 | **Step 13** |
| 24 | `_force_x64` autouse counter-test in `test_run_code_benchmark_common.py` | P0 | **Step 13** |
| 25 | `inspect.signature` regression guard for `_boozer_ls_coil_vjp` / `_boozer_exact_coil_vjp` | P0 | **Step 11** (VJP signature) |

Plus ungapped audit infrastructure recommendations:
- **`pytest -p random` CI gate** (synthesis §5) — to catch P0 #18 + future
  order-dependent leaks. Suggested **Step 14**.
- **`pytest --collect-only` guard** against re-deletion of restored
  upstream tests. Suggested **Step 14**.

---

## 5. Corrections needed within existing 9 steps

### Step 1 — "Use Existing SSOT Helpers" (too generic)
- Helper API must accept `lane=` keyword that **reads tolerance + ε from
  `PARITY_LADDER_TOLERANCES`**. Audit found `_REAL_RESOLVE_FD_TAYLOR_RATE
  = 0.55` (`tests/integration/test_single_stage_jax_cpu_reference.py:1439`)
  hardcoded — not under any lane. Without a `lane=` constraint in the
  shared helper, the same per-call hardcoding will reappear.
- "Add only small test-local helpers" risks a 4th competing helper
  surface. Specify location: `tests/_helpers/parity.py` and
  `tests/_helpers/jax.py`.
- "Reuse `tests/conftest.py` backend/parity fixtures" needs the
  device-aware skip predicate spelled out: `pytest.mark.xfail(condition=no_gpu,
  strict=True)`, NEVER `pytest.mark.skipif`. CPU-only CI silently going
  green is the exact failure mode the GPU-proof audit flagged.

### Step 2 — "preserve full build_provenance" (incomplete plumbing)
- **`build_provenance` does NOT yet emit `xla_flags` or `jaxlib_cuda_versions`**
  (verified: lines 468-492 of `benchmarks/validation_ladder_common.py`
  have only `repo_sha`, `jax`, `jaxlib`, `backend`, `devices`, `x64_enabled`,
  `peak_rss_mb`, plus optional `gpu_memory_mb`). The plan must:
  1. Extend `build_provenance` to emit `xla_flags` (snapshot of `XLA_*`
     and `JAX_PLATFORMS` env vars, captured **after** `import jax`) and
     `jaxlib_cuda_versions = getattr(jaxlib, "cuda_versions", None)`.
  2. Then update the aggregator to preserve those fields.
- **Aggregator-side rejection logic is not specified.** The aggregator at
  `benchmarks/hf_jobs/run_production_gpu_proof.sh:249-289` currently only
  reads `passed`/`elapsed_s`/`failures` from each payload (verified). The
  corrected plan says "preserve provenance" but does not say the
  aggregator must REJECT bundles when (a) `bundle_provenance.fake == True`
  without `SIMSOPT_FAKE_GPU=1`, (b) `default_backend != "gpu"` on the
  real-GPU lane, (c) `value_rtol`/`gradient_rtol` exceeds the lane
  contract.
- **Stage 2 line correction**: `require_requested_platform_runtime`
  insertion goes after the existing `require_x64_runtime(jax, context=...)`
  call at **line 59** of `benchmarks/stage2_e2e_comparison.py` (verified).

### Step 3 — "Boozer Math And Labels" (vague target list)
- Step 3 retires the false-positive `_MockVolumeLabel.J()` correctly but
  doesn't name the cosmetic follow-up: rename `_MockVolumeLabel` →
  `_PlumbingVolumeLabel` and delete the dead `J()` method. Future audits
  will keep flagging the existing name.
- Real-label tests should target lane `derivative_heavy` (rtol=1e-8,
  atol=1e-10) per `PARITY_LADDER_TOLERANCES`. Plan does not anchor.
- "Target-label quadratic penalty checks" should enumerate the test
  scope: `(label_val - targetlabel)² · weight/2` with both gradient and
  value parity vs. CPU `Volume(surface).J()` / `Area(surface).J()` /
  `ToroidalFlux(surface, field).J()`.

### Step 4 — "Derivative Tests" (line list incomplete)
- Confirmed FD-escape pattern occurs at **5 sites** per `PLAN_REVIEW.md`
  §1: `tests/integration/test_single_stage_jax_cpu_reference.py:{510,
  3744, 4213, 5153, 5792}`. Plan must enumerate all five.
- The Taylor rate constant `_REAL_RESOLVE_FD_TAYLOR_RATE = 0.55` lives at
  **line 1439** (verified). Plan says "central FD with fixed epsilon
  ladders" but does not name the constant. Required: change to `0.4` (the
  audit's recommendation; theoretical floor for symmetric central FD is
  ~0.25).
- The majority gates `_REAL_RESOLVE_FD_MIN_STABLE_SAMPLES = 3` (line 1441)
  and `_REAL_RESOLVE_FD_MIN_STABLE_EPS = 2` (line 1442) and the inline
  `validated_directions >= 2` (line 5804) all need to drop to "all
  surviving directions." Plan says "strict signed directional checks on
  nonzero directions" but does not name the constants.
- "Preselect nonzero directions" cherry-picking risk: must specify
  protocol: uniformly random with RNG-seeded reproducibility, reject only
  directions where projected gradient magnitude is below `1e-12`, require
  `len(stable) >= 0.8 * len(sampled)` else fail with "fixture geometry
  is degenerate."

### Step 5 — "Surface And Accessibility" (still implicitly skips bucket 4)
- Plan title scopes to "surface/accessibility" but bucket 4 has three
  TAUTOLOGICAL flux-kernel parity tests at
  `tests/objectives/test_fluxobjective_jax_parity.py:{211, 223, 253}`
  (P0 #5). Implementor will likely miss them. Must split into 5a (bucket
  1 surface), 5b (bucket 7 accessibility), 5c (bucket 4 flux kernels).
- "Add CPU/reference or FD parity for J, dJ, and selected Hessian-vector
  paths" is over-spec for accessibility (bucket 7 mostly needs J/dJ only)
  and under-spec for flux kernels (need CPU `SquaredFlux` parity).
- HLO text-count gates: do not delete; move to `tests/perf_gates/` with
  `@pytest.mark.brittle_perf_gate`.

### Step 6 — "Field And Reductions" (scoping question)
- The audit identified `pairwise_sum_axis`, `pairwise_sum_flat`,
  `compensated_sum_flat`, `scalar_square_sum` in
  `src/simsopt/jax_core/reductions.py:13-16` (verified). The corrected
  plan's "fixed-order parity only when claimed" is correct: **only**
  `compensated_sum_flat` and `pairwise_sum_*` claim deterministic order;
  `scalar_square_sum` dispatches on `mode` (compensated vs pairwise).
  Plan should be explicit: GPU/CPU parity for `compensated_sum_flat` and
  `pairwise_sum_axis`/`pairwise_sum_flat` only; default `jnp.sum`/`jnp.vdot`
  paths excluded.
- "Closed-surface ∮B·n dA" must specify coil ring **outside** the torus
  (e.g., at `z=5` for `R=1, r=0.1` torus) or `|∮B·n dA|/(max|B|·area_total)
  < 1e-9` will trivially fail.
- Add nfp rotational-symmetry B-field test (would have caught Y/Z
  stellsym DOF bug per CLAUDE.md history).
- Add Kahan/cancellation oracle for `compensated_sum_flat` on
  `[1e16, 1.0, -1e16]`.

### Step 7 — "Restore Upstream Coverage" (naming + missing siblings)
- Function name is `test_Taylor` in `tests/field/test_selffieldforces.py`,
  not `test_force_objectives_taylor_test` (per `VERIFY_plan_claims.md`
  §5).
- Add `test_objectives_time` (default-on for `ncoils=2`; gate `ncoils=8`
  on `SIMSOPT_RUN_FIELD_TIMING=1`) and `test_call` follow-up at
  `tests/core/test_optimizable.py:243-258`.
- Name the 3 deleted tests verbatim: `test_arclength_variation_circle_planar`,
  `test_linking_number_planar`, `test_curve_curve_distance_empty_candidates`.
- HEAD's `distance_threshold = 0.4 if "CurveHelical" else 0.2` conflict
  needs resolution before restoring upstream uniform `0.4`.
- Use `np.random.default_rng(0)` (modern API, same seed value to
  minimize tolerance drift from upstream `np.random.seed(0)`).

### Step 8 — "Smoke And Donation" (mechanism unspecified)
- Donation mutant test as a real fixture: `tests/test_donation_mutants.py`
  parametrizes over `donate_argnums = (0,)` vs `()` and asserts the latter
  fails the existing donation probe. Without a continuous mechanism
  (separate test file + marker), "mutant must fail" is one-shot.
- JSON sentinel schema: wrappers must validate `payload["case"] ==
  expected_case_name` AND `payload["invariant"] in {"compile_count",
  "device_residency", "no_simsoptpp_import", ...}`. Plan mentions both
  fields but not wrapper-side schema enforcement.
- Donation assertion uses both `points.is_deleted() is True` AND
  `pytest.raises(RuntimeError, match="has been deleted")` against
  `jnp.asarray(points)` (both work in JAX 0.9.2 per direct probe in
  `VERIFY_plan_claims.md` claim #1).

### Step 9 — "Validation" (mutant infrastructure ad-hoc)
- "JAX-vs-JAX-only oracle must fail" mutant: must specify mechanism.
  Options:
  - One-shot pytest test that monkeypatches the oracle path and asserts
    failure → low confidence; rots after refactor.
  - Separate `tests/mutants/test_required_mutants.py` with
    `@pytest.mark.mutation` run nightly → continuous enforcement.
  - Real `mutmut`/`cosmic-ray` integration → most rigorous but heaviest
    dev cost.
  Plan must pick one; recommended: separate-file + `pytest.mark.mutation`.
- Add three more mutants beyond the listed five:
  - `_real_label.J()` returns 0.0 → step 4 must fail.
  - GPU reduction summands reordered → step 6 GPU parity must fail.
  - `use_jax_curve=True` branch silently disabled in `test_Taylor` →
    step 7 must fail.
- Add mutants for unaddressed P0s if Steps 10-13 are added: real-Newton
  fixture mutant, IotasJAX adjoint residual mutant, etc.

---

## 6. Final ranking — remaining work to actually close the audit

Ranked by impact-to-effort. Each item closes one or more P0s.

| Rank | Action | P0s closed | Effort | Why this order |
|-----:|--------|:----------:|:------:|:---------------|
| 1 | **Add Step 10** (real exact-Newton on torus + ill-cond exact + 3×3 oracle replacement) | #7, #10 (+ bucket 2 §3 ill-cond) | M-L | Largest unaddressed cluster: entire `TestBoozerSurfaceJAXExactPath` validates plumbing only. Single-highest leverage step. |
| 2 | **Step 2 plumbing fix**: extend `build_provenance` BEFORE updating aggregator; add aggregator rejection logic | #1 (close gap), #14 (close gap) | M | Without this, the very thing Step 2 claims to fix (provenance preservation) silently records `None`. Verified: `xla_flags` and `jaxlib_cuda_versions` are NOT yet in `build_provenance`. |
| 3 | **Add Step 12** (M5 wrappers + SquaredFluxJAX Taylor + zero-current `np.isposinf` + outer-opt strict decrease + adjoint-fraction diagnostic) | #8, #9, #16, #21 | M | Closes 4 P0s in one focused module; existing test files; no new fixtures needed. |
| 4 | **Add Step 13** (conftest/order/_force_x64/printf-format) | #18, #19, #23, #24 | S-M | 4 P0s, mostly XS each; touches independent files. Easy parallel work. |
| 5 | **Add Step 11** (VJP signature regression + value-vs-FD parity replacing shape-only test) | #25 | XS-S | 1 P0 but extremely high catch-rate per LOC; protects against the documented historical CPU-vs-JAX 2-arg-vs-4-arg drift. |
| 6 | **Step 3 line-list expansion** (5 sites instead of 3, add line 1439, 1441, 1442, 5804) | #3 (close gap), #6 (add IotasJAX adjoint residual rel-tol) | S | Step 3 is "B" status because line 1439 (`0.55`) is missed and line 5804 (`>= 2`) is also unaddressed. |
| 7 | **Step 5 split into 5a/5b/5c** with explicit oracle per cluster | #5 (close gap), #13 (close gap), #17 (close gap), #22 (close gap) | M | Moves 4 P0s from B → A by enumerating bucket 1 / 4 / 7 separately. Otherwise implementor will miss bucket 4 flux kernels. |
| 8 | **Step 6 specifics**: GPU `xfail strict=True`, coil ring outside torus, nfp rotational symmetry, Kahan/cancellation oracle | #20 (already A but harden), bucket 4 §3 (nfp), bucket 7 §3 (Kahan) | S | Cheap correctness hardening; prevents Step 6 from being trivially-passing on CPU-only CI. |
| 9 | **Add Step 14** (CI infra: `pytest -p random`, `pytest --collect-only` guard, `pytest.skip` audit) | future-#18, future-restore re-deletion | S | Covers ongoing regression risk; not a P0 closure but prevents recurrence. |
| 10 | Step 1 `lane=` SSOT enforcement + `xfail strict=True` for GPU skip | systemic — guards Steps 3, 4, 6 | XS-S | Prevents the very tolerance-drift the audit found (0.55 hardcoded) from recurring. |
| 11 | Step 4 `_MockVolumeLabel` rename + dead-`J()` deletion (cosmetic) | none (P0 #2 retired) | XS | Stops future audits from re-flagging. |

**Bottom line**: with Steps 10-14 added (estimated effort: 1 L + 2 M + 2 S =
~4-6 days), the corrected plan covers ~92% of the P0 set fully (all 11
remaining C's plus the 6 B's promoted to A). Without them, the corrected
plan is at 7/25 fully closed — better than the first plan's 6/25 but
still well short of the audit's stated bar.

---

DONE — corrected-plan review at /Users/suhjungdae/code/columbia/simsopt-jax/.artifacts/jax-test-audit-2026-04-25/CORRECTED_PLAN_REVIEW.md
