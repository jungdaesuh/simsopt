# Corrected Plan Review (V2) — 2026-04-25

Branch: `gpu-purity-stage2-20260405` · HEAD `42b68f33d`
Inputs: corrected 9-step plan + 8 plan-bug-fix items from user + bucket1..8 +
SYNTHESIS + PLAN_VS_FINDINGS + PLAN_REVIEW + 5 verification reports
(`VERIFY_mock_volume_label.md`, `VERIFY_plan_claims.md`,
`VERIFY_physics_claims.md`, `VERIFY_jax_cuda_claims.md`,
`CORRECTED_PLAN_REVIEW.md`).

---

## 0. All 8 plan bug fixes verified against source / docs

| # | Bug fix claim | Verdict | Evidence |
|--:|---|:---:|---|
| 1 | Donation: assert input invalid/deleted, NOT output buffer aliasing; positional `donate_argnums` | ✅ CORRECT | jax 0.9.2 probe + JAX docs (`buffer_donation.html`, `jax.jit`): donated buffers "can be overwritten and marked deleted in the caller", reusing them raises `RuntimeError: ... invalid buffer`. Output aliasing is XLA hint, not JAX guarantee. |
| 2 | jaxlib 0.9.2 lacks stable CUDA-version attrs; reuse `build_provenance` SSOT + launcher-side CUDA env vars | ✅ CORRECT | Direct probe: `getattr(jaxlib, "cuda_versions", None)` evaluates to `None` on 0.9.2; no `jax_cuda12_pjrt`, no `jaxlib.cuda` submodule. Drop the field; capture `nvidia-smi` driver/GPU-name + `jax.default_backend()` + `jax.devices()` repr at launcher. |
| 3 | GPU proof must fail closed under `--platform cuda`, not skip | ✅ CORRECT | `require_requested_platform_runtime` (`benchmarks/validation_ladder_common.py:288-305`) raises `RuntimeError` when actual `default_backend()` doesn't match requested platform. Only `--platform auto` is no-op. |
| 4 | `CUDA_FORCE_PTX_JIT` / `CUDA_DISABLE_PTX_JIT` are canonical PTX/cubin knobs; `CUDA_VISIBLE_DEVICES` alone is insufficient | ✅ CORRECT | NVIDIA Programming Guide §5.2: `CUDA_FORCE_PTX_JIT=1` ignores embedded CUBIN; `CUDA_DISABLE_PTX_JIT=1` requires compatible CUBIN. `CUDA_VISIBLE_DEVICES` only controls device enumeration. |
| 5 | `integral_BdotN` squares/normalizes; raw signed `∮B·n dA` needs new helper | ✅ CORRECT | All 3 definitions in `src/simsopt/objectives/integral_bdotn_jax.py:50-78` and C++ `integral_BdotN.cpp:93-103` square `(B·n̂)`. `residual_BdotN` is pre-weighted by `√weight`. **No existing raw signed-flux helper in the codebase.** New ~3-line `signed_flux_jax(B, normal)` primitive needed. |
| 6 | FD: strict signed directional checks on preselected nonzero directions; near-zero gets separate test | ✅ CORRECT | Closes the cherry-picking risk PLAN_REVIEW.md flagged in step 3 of the first plan. Aligns with central FD theory: error scales as `eps²` for smooth functions. |
| 7 | HLO/StableHLO text checks are compile-shape/perf tests, NOT correctness | ✅ CORRECT | Bucket 1 (`test_surface_rzfourier_jax.py`) flagged 3 HLO heuristic gates as XLA implementation-detail tests. Reclassification is the right disposition. |
| 8 | Restore upstream coverage without rewriting CPU-lane behavior; representative default + slow exhaustive sweep | ✅ CORRECT | Bucket 8: `test_Taylor` collapsed 320→12 (96.25% reduction); 4 of 10 per-coil-sum objectives dropped; `for downsample in [1,2,3]` loop dropped; 3 upstream tests deleted. Restoration is correct disposition. |

**All 8 bug fixes verified.** SIMSOPT Boozer residual `G·B − |B|²·(x_φ + ι·x_θ)`
also confirmed identical across JAX (`boozer_residual_jax.py:110-117`),
Python (`surfaceobjectives.py:596-598`), and C++ (`boozerresidual_impl.h:60-66`).
`Volume.J() = (1/3)∮r·n dA` (enclosed volume, no normalization) confirmed.

---

## 1. Coverage delta vs first plan

| Status | First plan | Corrected plan | Movement |
|---|:---:|:---:|---|
| **A** (fully addressed) | 6 (24%) | **7 (28%)** | +1 (P0 #4 moved C→A: positional donation contract) |
| **B** (partially addressed) | 6 (24%) | 6 (24%) | 0 |
| **C** (unaddressed) | 12 (48%) | **11 (44%)** | -1 |
| **D** (correctly retired) | 1 (4%) | 1 (4%) | 0 |

Net: corrected plan covers **28% fully + 24% partially + 44% unaddressed + 4%
retired** of the 25 P0 items. Real improvement vs first plan: +1 fully
addressed; one P0 item moved into "explicitly handled" via the donation
positional-contract bug-fix.

The improvements come not from raw coverage growth but from **technical
correctness** — the first plan would have implemented `output buffer aliasing`
checks that don't match JAX's documented contract, would have included
`jaxlib_cuda_versions` as a useful field on jax 0.9.2 (where it's None), and
would have used `integral_BdotN` for a closed-surface flux test that always
yields a non-negative number.

---

## 2. Strengths of the corrected plan

1. **Explicitly retires the false MockVolumeLabel P0** (saves a wave of
   misdirected refactor work; bucket2 P0 #2 was wrong, user pushed back
   correctly, plan author maintains the position).
2. **Raw signed flux instead of `integral_BdotN`** (subtle physics bug fix —
   `integral_BdotN` is non-negative by construction, so `≈ 0` is automatic and
   tells you nothing about whether the closed-surface integrand sums to zero).
3. **Real CUDA compile/run rung** (`block_until_ready` synchronized) — would
   have caught the 2026-04-20 Runpod cubin incident; provenance recording
   alone proves nothing.
4. **Drops invented `jaxlib_cuda_versions` field** — capturing what JAX
   doesn't expose is theatrical; the launcher-side `nvidia-smi` capture is
   the right replacement.
5. **Strict signed directional checks on preselected nonzero directions**
   closes the cherry-picking risk — PLAN_REVIEW.md flagged this as a new risk
   introduced by the first plan; corrected plan kills it directly.
6. **`use existing SSOT helpers`** in step 1 (`build_provenance`,
   `require_requested_platform_runtime`, conftest fixtures) avoids re-inventing
   APIs that already exist and have lane-aware semantics.
7. **Validation step 9 has 5 explicit mutants** — each maps to a specific
   plan step's invariant, not just a generic "run tests."

---

## 3. Remaining gaps (11 P0s still unaddressed)

| P0 # | Summary | Severity | Why the plan misses it | Suggested fix |
|--:|---|:---:|---|---|
| #6 | IotasJAX adjoint residual rel-tol gate (line 5662) | P0 | Step 4 covers FD/IFT directional gradients but not adjoint residual norm gate | Add to step 4: assert `adjoint_residual_rel <= 1e-10` for IotasJAX branch |
| #7 | Toy 3×3 oracle at `test_surface_objectives_jax.py:1728` | P0 | Step 5 mentions accessibility (bucket 7), not exact-well-conditioned-adjoint plumbing in bucket 3 | Add Step 10: replace toy 3×3 with real `BoozerSurfaceJAX(boozer_type='exact')` + `scipy.linalg.lu_solve` |
| #8 | `test_adjoint_fraction_diagnostic` `>0` ceremony (line 5859) | P0 | Plan does not promote/delete this | Add to step 4: replace with meaningful upper bound |
| #9 | `test_outer_opt_decreases_objective` `+1e-12` slack (line 4925) | P0 | Plan does not mention strict-decrease | Add to step 4: drop slack, require strict monotone decrease |
| #10 | TestBoozerSurfaceJAXExactPath cluster: ~14 tests run against `_patched_exact_newton_result(jacobian=identity)` shim | P0 | Plan does not require shim removal | Add Step 10: ONE end-to-end exact-Newton on real torus, no shim |
| #16 | `np.isfinite`-only on zero-current singular case (`tests/integration/test_stage2_jax.py:1036`) | P0 | Step 6 is field/reductions, not Stage 2 singular-case | Add to step 6: pin `np.isposinf` parity (singular-current limit is +∞, not just finite) |
| #18 | Order-dependent `test_backend_state_guard_sequence_*` and `test_backend_module_guard_sequence_*` | P0 | Plan does not touch ordering/sequence tests | Add Step 10: refactor into single-test-per-sequence layout; add `pytest -p random` CI gate |
| #19 | `tests/integration/conftest.py:_patch_meta_path_finder` returning `False` silently | P0 | Plan does not touch conftest skip-vs-silent-False | Add to step 8: replace silent `return False` with `pytest.skip("editable install required")` |
| #21 | `SquaredFluxJAX.dJ()` Taylor + chunked-VJP gradient parity on large point cloud | P0 | Step 5 covers surface/accessibility, not flux | Add to step 5c (new sub-step): explicit Taylor test + chunked-VJP parity |
| #23 | `test_field_cache_hot_path_benchmark.py` printf-format pinning | P0 | Plan does not mention this benchmark | Add Step 10: end-to-end compile + parse output JSON; drop printf-format pinning |
| #24 | `_force_x64` autouse fixture blinkers all tests in `test_run_code_benchmark_common.py` | P0 | Plan does not address autouse counter-test | Add Step 10: counter-test that drops autouse, asserts behavior degrades |

P0 #25 (`inspect.signature` VJP regression guard) is also still missing —
classify with #10 as Boozer exact-Newton/adjoint cluster.

---

## 4. Sharpenings needed within existing 9 steps

### Step 1 — SSOT helpers
- Add `lane=` keyword to all FD/Taylor/parity helpers; consume tolerances from
  `benchmarks/validation_ladder_contract.py::PARITY_LADDER_TOLERANCES` SSOT.
  Reject ad-hoc tolerance overrides — otherwise the same hardcoded `0.55`
  Taylor rate at `test_single_stage_jax_cpu_reference.py:1439` will reappear.
- GPU device-aware skip MUST be `pytest.mark.xfail(condition=no_gpu, strict=True)`,
  NOT `pytest.mark.skipif`. With `skipif`, a CI runner that loses its GPU
  silently goes green — this is exactly the failure mode the original GPU-proof
  audit flagged.
- Place helpers under `tests/_helpers/parity.py` (not scattered across test
  files).

### Step 2 — GPU proof
1. **Plumbing order matters**: `build_provenance` (`benchmarks/validation_ladder_common.py:468-492`)
   does NOT yet emit `xla_flags`, `cuda_force_ptx_jit`, `cuda_disable_ptx_jit`,
   or `nvidia_smi_capture`. Step 2 must extend the emitter BEFORE updating the
   aggregator at `run_production_gpu_proof.sh:249-289`.
2. **Aggregator must REJECT, not just preserve**:
   - When any payload's `provenance.fake_runner == True` and
     `SIMSOPT_FAKE_GPU != "1"`: fail loud.
   - When any payload's `provenance.backend != "gpu"` on the real-GPU lane:
     fail loud.
   - When any `comparison.value_rtol` or `comparison.gradient_rtol` exceeds
     the parity-ladder lane contract: fail loud.
   Step 2 says "preserve full build_provenance" but plumbing is at the
   aggregator; preservation alone doesn't enforce.
3. **Stage 2 line correction**: insert `require_requested_platform_runtime(jax,
   requested_platform=REQUESTED_PLATFORM, context="Stage 2 end-to-end
   comparison")` after `require_x64_runtime(...)` at line 59 (NOT line 46 —
   line 46 is `apply_requested_platform`).
4. **Real CUDA compile/run rung specifics**: minimal payload should be
   `arr = jax.device_put(jnp.arange(64.0)); out = jax.jit(lambda x: x*x)(arr).block_until_ready();
   assert out.devices() == {jax.devices('gpu')[0]}` — i.e., the OUTPUT must
   be GPU-resident, not just the input.
5. **`bootstrap_runtime.sh` hardening**: add `python -c "import jax; assert
   jax.default_backend() == 'gpu'"` immediately after `import jax`, plus
   write `bootstrap_jax_smoke.json` artifact alongside the proof bundles.
   Currently `bootstrap_runtime.sh:30-54` validates only `jax.__version__`.
6. Add `bundle_provenance.fake = bool` discriminator field so the aggregator
   can refuse GREEN unless `SIMSOPT_FAKE_GPU=1`.

### Step 3 — Boozer math and labels
- Add concrete cosmetic action: rename `_MockVolumeLabel` →
  `_PlumbingVolumeLabel` (or delete the dead `J()` method). This stops future
  audits from re-flagging it.
- Real-label tests target lane `derivative_heavy` (`rtol=1e-8, atol=1e-10`)
  per `PARITY_LADDER_TOLERANCES`. Enumerate the 7-12 highest-leverage tests to
  migrate to real labels.
- Add `post-Newton ‖grad‖ < 1e-10` magnitude target to
  `test_newton_polish_reduces_gradient` (P0 #6 in bucket 2).
- For target-label quadratic-penalty checks, perturb `target_label` by
  `delta` and verify `dJ/d(target_label) = -2 * (J_label - target_label) *
  weight` (closed-form derivative of the squared penalty).

### Step 4 — Derivative tests
1. **Name the line numbers**:
   - `_REAL_RESOLVE_FD_TAYLOR_RATE = 0.55` at `tests/integration/test_single_stage_jax_cpu_reference.py:1439`
     → change to `0.4` (or per-lane, read from `PARITY_LADDER_TOLERANCES`)
   - `_REAL_RESOLVE_FD_MIN_STABLE_SAMPLES = 3` at line 1441 → keep as 3 but
     this is the all-of-N gate (correct semantic per plan)
   - `_REAL_RESOLVE_FD_MIN_STABLE_EPS = 2` at line 1442 → tighten to 3
   - `validated_directions >= 2` at line 5804 → change to `validated_directions == 3`
2. **OR-escape pattern at 5 sites, not 3**: `tests/integration/test_single_stage_jax_cpu_reference.py:{510, 3744, 4213, 5153, 5792}`.
   First plan named only 3; corrected plan must name all 5.
3. **"Preselected nonzero directional derivatives" specifics**: deterministic
   fixture that constructs surface DOFs for which the analytical gradient
   has known nonzero magnitude in 3 orthogonal coordinate directions; assert
   FD matches with sign in ALL 3.
4. **Near-zero separate tests**: assert FD result is bounded by a small
   absolute floor; do NOT mix into sign-sensitive parity checks.

### Step 5 — Surface and accessibility
Split into 5a/5b/5c with explicit oracle per sub-step:
- **5a (bucket 1, surface)**: drop the rtol=1e-12 JAX-vs-JAX arms in
  `_assert_surface_jacobian_parity` / `_assert_area_volume_gradient_parity`
  (`tests/geo/test_surface_rzfourier_jax.py:619-696`); keep the C++ oracle
  arm; add FD oracle (`eps=1e-5, rtol=1e-7, atol=1e-9`). Drop
  `test_normal_orthogonality` cross-product tautology. Add analytic torus
  area/volume tests (`area = 4π² R r`, `volume = 2π² R r²`).
- **5b (bucket 7, accessibility)**: enumerate the 5
  `*_reuses_shared_jit_kernels` tests; for each, ADD `J/dJ` FD parity at
  `h=1e-6, rtol=1e-6` (lane `derivative_heavy`). Hessian-vector is over-spec;
  drop from accessibility scope.
- **5c (bucket 4, flux kernels)**: explicitly named — replace the three
  `_flux_kernel_value_and_grad` self-comparison tests at
  `tests/objectives/test_fluxobjective_jax_parity.py:{211, 223, 253}` with
  CPU `SquaredFlux` parity OR analytic-zero algebra. Add `SquaredFluxJAX.dJ()`
  Taylor test (P0 #21). Pin `SquaredFluxJAX.dJ()` raises `ObjectiveFailure`
  for the singular zero-current case.
- HLO text-count gates: relocate to `tests/perf_gates/` with
  `@pytest.mark.brittle_perf_gate` (don't delete — useful as soft alerts).

### Step 6 — Field and reductions
1. **Add the `signed_flux_jax(B, normal)` helper** (~3 lines per
   VERIFY_physics_claims.md): `signed_flux = jnp.sum(B * normal) * dphi *
   dtheta`. Place in `src/simsopt/objectives/integral_bdotn_jax.py` next to
   the squared variants, with a docstring explicitly distinguishing it from
   `integral_BdotN`.
2. **Closed-surface Gauss-law specifics**: coil ring at `z=5` (well outside a
   `R=1, r=0.1` torus); refine quadrature to 64×64 minimum; expect
   `|signed_flux| / max|B| / area_total < 1e-9`.
3. **GPU marker**: `pytest.mark.xfail(condition=no_gpu, strict=True)`, NOT
   `skipif`.
4. **Reduction parity scoping**: bucket 7 §3 names `pairwise_sum_*`,
   `compensated_sum_flat`, `scalar_square_sum` as functions claiming
   fixed-order parity. Plan correctly excludes default `jnp.sum`/`vdot` —
   verify the source contract (in `src/simsopt/jax_core/reductions.py`)
   matches this scoping before writing tests.
5. **Add Kahan/cancellation oracle** for `compensated_sum_flat` on
   adversarial `[1e16, 1.0, -1e16]` (bucket 7 §3).
6. **Add nfp rotational-symmetry test** for the B-field — bucket 4 §3
   missing coverage that would have caught the historical Y/Z stellsym DOF
   bug.

### Step 7 — Restore upstream coverage
1. **Naming**: function is `test_Taylor` in
   `tests/field/test_selffieldforces.py:1720`, not
   `test_force_objectives_taylor_test`.
2. **Restore the per-coil-sum objective variants** that HEAD dropped (4 of 10
   upstream objectives): `sum([LpCurveTorque(coils[i], coils2, ...) for i
   in range(len(coils))])` and analogous `SquaredMeanTorque/Force/LpCurveForce`
   sums.
3. **Resolve `distance_threshold` conflict** in
   `subtest_curve_minimum_distance_taylor_test`: HEAD's `0.4 if "CurveHelical"
   else 0.2` is incompatible with restoring the upstream uniform `0.4`. Verify
   the upstream `0.4` branch under HEAD's helical path before restoring.
4. **Slow-sweep gating**: use the existing `pytest.mark.slow` decorator
   (`tests/conftest.py:434`) and run on `pytest --runslow` in nightly CI.
5. **Add `test_objectives_time` and `test_call`** to step 7 explicitly (sibling
   regressions per bucket 8 P0 #2 and #8). `test_objectives_time` should
   default-on for `ncoils=2`, env-gate `ncoils=8` row only.
6. **Name the 3 deleted tests verbatim**: `test_arclength_variation_circle_planar`,
   `test_linking_number_planar`, `test_curve_curve_distance_empty_candidates`
   — restore from upstream commit rather than re-implementing.
7. **Seed migration**: keep `np.random.seed(0)` if the test is restored
   verbatim (don't introduce drift from `default_rng(0)` migration unless
   re-tuning tolerances).

### Step 8 — Smoke and donation
1. **JSON sentinel schema**: wrappers must validate `payload["case"] ==
   expected_case_name` AND `payload["invariant"] in {"compile_count",
   "device_residency", "no_simsoptpp_import", ...}`.
2. **Donation mutant test as a continuous fixture**:
   `tests/test_donation_mutants.py` with `pytest.mark.mutation` — monkeypatch
   `donate_argnums = ()` and assert the existing donation probe test now
   fails. Run nightly.
3. **Donation assertion uses BOTH predicates** (per JAX docs):
   `points.is_deleted() is True` AND
   `pytest.raises(RuntimeError, match="has been deleted")` against
   `jnp.asarray(points)`.
4. **Enumerate the ~14 highest-leverage `test_jax_import_smoke.py` wrappers**
   to migrate first (out of ~75). Don't try to migrate all 75 in one PR.
5. **Fix `tests/integration/conftest.py:_patch_meta_path_finder`** to
   `pytest.skip("editable install required")` instead of silent `return
   False` (P0 #19).

### Step 9 — Validation
- Add three more mutants:
  - `_real_label.J()` returns 0.0 → step 3's tests must fail.
  - GPU reduction on a 1024-element array reordered → step 6 GPU parity
    must fail.
  - `use_jax_curve=True` branch silently disabled in `test_Taylor` → step 7
    must fail.
- Place mutant tests in `tests/mutants/test_required_mutants.py` with
  `pytest.mark.mutation`, run nightly. Without a separate file + marker,
  "required mutant checks" become a one-time manual exercise.
- Add a `pytest --collect-only | grep -c <restored_test_names>` guard in
  CI so re-deletions of restored tests are caught immediately.

---

## 5. Required Step 10 additions

The 11 unaddressed P0 items cluster into 4 themes that need a dedicated
Step 10 (or sub-numbered into the existing 9). The single highest-impact
addition remains: **end-to-end exact-Newton fixture on a real torus**.

### Step 10 — Boozer exact-Newton end-to-end fixture (P0 #7, #10, #25)
- Add ONE end-to-end exact-Newton test on a real torus (no
  `_patched_exact_newton_result`, no `jacobian = identity` shim). Lane:
  `exact-well-conditioned-adjoint` (`rtol=1e-6, atol=1e-8, residual ≤ 1e-10`).
- Replace toy 3×3 oracle at `tests/geo/test_surface_objectives_jax.py:1728`
  with a real `BoozerSurfaceJAX(boozer_type='exact')` fixture. Use
  `scipy.linalg.lu_solve` against the materialized PLU as the dense oracle;
  assert operator-vs-dense vector parity per the lane contract.
- Add an ill-conditioned exact-path test that asserts
  `failure_category="scaling_limit"` OR operator residual `≤ 1e-10`, with
  NO vector parity claim.
- Add `inspect.signature` regression for `_boozer_ls_coil_vjp` and
  `_boozer_exact_coil_vjp` (P0 #25).

### Step 11 — Conftest and order discipline (P0 #18, #19, #24)
Already enumerated in step 8 sharpenings above; promote to standalone step
to avoid orphaning under "smoke and donation."

### Step 12 — CI infrastructure (synthesis §5)
- `pytest -p random` (or `pytest-randomly`) on one CI job to catch
  order-dependent tests.
- `pytest --collect-only` guard against re-deleted tests.
- `pytest.skip(...)` audit: every skip must have a tracked GitHub issue ID;
  default policy is skip-without-issue-ID = CI failure.
- Add `test_field_cache_hot_path_benchmark.py` end-to-end replacement
  (P0 #23): compile + parse output JSON; drop printf-format pinning.

---

## 6. Final assessment

The corrected plan is **technically sounder** than the first plan in 6
specific ways (raw signed flux, dropped invented `jaxlib_cuda_versions`,
positional donation contract, real CUDA compile/run rung, strict signed
directional checks on preselected directions, retired MockVolumeLabel). Net
coverage moves from 24% → 28% fully addressed.

**The corrected plan is ready to implement** for steps 1-9 with the line-number
sharpenings in §4 above. The 11 unaddressed P0 items (44%) need one or two
additional steps (step 10-12 in §5) to fully close the audit. Without those,
the suite remains incapable of catching:
- Boozer exact-Newton convergence regressions on a real torus (P0 #10)
- Order-dependent test failures under `pytest -p random` (P0 #18)
- Adjoint-failure-yields-finite-wrong-gradient bugs in legacy `dJ()` (bucket 3 §3)
- VJP signature drift on `_boozer_ls_coil_vjp` / `_boozer_exact_coil_vjp` (P0 #25)
- Squared-flux dJ regressions on large point clouds (P0 #21)

The single most important sharpening within the existing 9 steps remains:
**Step 2's `build_provenance` extension MUST come BEFORE the aggregator
update**. Otherwise the aggregator preserves nothing meaningful because the
fields don't exist yet.

The user's pushback on MockVolumeLabel (still correctly retained), the
positional-donation reframing (matches JAX docs), the `integral_BdotN`
substitution (matches source code), and the CUDA cubin proof rung (matches
NVIDIA docs) all hold up to verification. The corrected plan deserves to
move forward.
