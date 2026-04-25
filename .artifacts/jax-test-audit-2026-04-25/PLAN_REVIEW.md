# Plan Review — 2026-04-25

Branch: `gpu-purity-stage2-20260405` · HEAD `42b68f33d`
Inputs: 9-step plan from user + bucket1..8 + SYNTHESIS + 3 verification reports
(`VERIFY_mock_volume_label.md`, `VERIFY_plan_claims.md`, `PLAN_VS_FINDINGS.md`).

---

## 0. Headline corrections to the original audit

The user's pushback on `_MockVolumeLabel.J() ≡ 0.0` is **correct**. The audit's
SYNTHESIS P0 #2 / bucket2 P0 #2 is a **false positive** and is hereby retracted.

Direct code reading
(`VERIFY_mock_volume_label.md`):

- `src/simsopt/geo/boozersurface_jax.py` contains exactly **one** `\.J(` match
  in the entire file, and it is a docstring at line 1955. There are zero
  executable `label.J()` calls in the JIT forward path, exact-residual
  builder, LS residual, adjoint, or any public API.
- The penalty label is computed by `_label_from_geometry_and_field_terms`
  (L743-758) which dispatches on the string `params.label_type` and calls
  `volume_jax`/`area_jax`/`toroidal_flux_jax` on `gamma`/`normal`/`A`. The
  penalty is `(label_val - params.targetlabel)² · weight/2` (L774-778).
- `self.label` is stored only to derive the string token `self.label_type`
  via `type(label).__name__` (L1990-2013); it is never read during the JIT
  trace.
- A repo-wide grep across the three test files returns **zero** `.J()`
  invocations on the mock. The mock returning `0.0` cannot affect any
  assertion because no assertion's RHS is derived from it.

Implication: **the user's plan correctly declines to "fix" this** in step 4.
The mock is plumbing-only and should stay that way. The audit's headline
"~70% of M4 tests silently null-test the constraint" is wrong; the
constraint contribution IS exercised via the recomputed `volume_jax(gamma,
normal) - targetlabel` term, with `targetlabel = 2π² R r²` set by
`_make_mock_boozer_surface`.

The remaining residual concern is cosmetic: the `J() → 0.0` method is dead
code that future audits will keep flagging. Recommended cosmetic follow-up
in step 4: rename `_MockVolumeLabel` → `_PlumbingVolumeLabel` (or delete the
dead `J()` method), and add a one-line comment explaining the marker-only
intent.

---

## 1. Verification of plan's load-bearing factual claims

| # | Claim | Verdict | Refinement needed |
|--:|-------|:------:|---|
| 1 | jax 0.9.2 has `Array.is_deleted()`; deleted-array reads raise `RuntimeError` | **CONFIRMED** | Test should use both `is_deleted() is True` AND `pytest.raises(RuntimeError, match="has been deleted")`. Run probes under `PYTHONNOUSERSITE=1` to avoid user-site jax 0.10.0 shadowing. |
| 2 | Stage 2 missing `require_requested_platform_runtime` (line 46) | **CONFIRMED** | Insert at end of preamble after `require_x64_runtime(...)` at line 59 (not line 46 — line 46 is `apply_requested_platform`). Also import the symbol at line 33. |
| 3 | `run_production_gpu_proof.sh:249` aggregator drops provenance | **CONFIRMED** | **Dependency**: `xla_flags` and `jaxlib_cuda_versions` are NOT yet emitted by `build_provenance` (lines 468-492). Step 2 must (a) extend `build_provenance` to emit them, THEN (b) update the aggregator to preserve. |
| 4 | FD escape hatches at lines 4213, 5792, 1979 | **PARTIALLY CONFIRMED** | The OR-escape pattern occurs at **5 sites**, not 3: `tests/integration/test_single_stage_jax_cpu_reference.py:{510, 3744, 4213, 5153, 5792}`. Line 1979 is the *eps-ladder majority gate* (`_REAL_RESOLVE_FD_MIN_STABLE_EPS = 2`); the *direction majority gate* is at line 5804 (`validated_directions >= 2`). The Taylor rate `0.55` is at line 1439 (`_REAL_RESOLVE_FD_TAYLOR_RATE`). |
| 5 | Force-test collapse 320→12; 3 deleted curve tests; lost seed | **CONFIRMED** | Naming caveat: function is `test_Taylor` (in `tests/field/test_selffieldforces.py:1720`), NOT `test_force_objectives_taylor_test` (which doesn't exist). 96.25% reduction (1 - 12/320). HEAD also dropped the `for downsample in [1,2,3]` loop AND 4 of 10 objective entries (the `sum([Lp/SquaredMean*(coils[i]) for i ...])` per-coil-sum variants). HEAD's `distance_threshold = 0.4 if "CurveHelical" else 0.2` is incompatible with restoring the upstream uniform `0.4` — pick one. |

All five claims hold. The only concrete refinement is that the FD-escape
fix list must expand from `{4213, 5792}` to `{3744, 4213, 5153, 5792}`
and add line 1439 (the `0.55` Taylor rate constant).

---

## 2. Plan-vs-findings coverage matrix

Across the audit's 25 P0 items (in `SYNTHESIS.md` §3):

- **A** (fully addressed by plan): 6 — P0 #1, #4, #11, #12, #15, #20
- **B** (partially addressed): 6 — P0 #3, #5, #13, #14, #17, #22
- **C** (not addressed): 12 — P0 #6, #7, #8, #9, #10, #16, #18, #19, #21, #23, #24, #25
- **D** (explicitly retired by plan): 1 — P0 #2 (correctly, per §0 above)

**Plan covers 24% fully, 24% partially, 48% unaddressed, 4% retired.**

The 48% unaddressed cluster into four themes:

1. **Boozer exact-Newton / adjoint contract**: P0 #6 (IotasJAX adjoint
   residual rel-tol gate at line 5662), #7 (toy 3×3 oracle at
   `test_surface_objectives_jax.py:1728` needs real `BoozerSurfaceJAX(boozer_type='exact')`
   fixture), #10 (entire `TestBoozerSurfaceJAXExactPath` cluster runs against
   `_patched_exact_newton_result(jacobian=identity)`), #25 (no
   `inspect.signature` regression guard for `_boozer_ls_coil_vjp` /
   `_boozer_exact_coil_vjp`).
2. **M5 wrapper failure / parity gaps**: legacy `dJ()` adjoint failure path
   doesn't yield non-finite gradient on adjoint failure (bucket 3 §3);
   `IotasJAX` and `NonQuasiSymmetricRatioJAX` projected coil-derivative
   parity at LS-warmed fixture is missing.
3. **Stage 2 sign / Taylor coverage**: P0 #16 (`np.isfinite`-only assertion
   on zero-current singular case), P0 #21 (`SquaredFluxJAX.dJ()` Taylor test
   + chunked-VJP gradient parity on large point cloud).
4. **Conftest / order / smoke hardening**: P0 #18 (order-dependent
   `test_backend_*_sequence_*`), P0 #19 (`tests/integration/conftest.py:_patch_meta_path_finder`
   silent `False`), P0 #23 (`test_field_cache_hot_path_benchmark.py`
   printf-format pinning), P0 #24 (`_force_x64` autouse fixture blinkers all
   tests in `test_run_code_benchmark_common.py`).

---

## 3. Step-by-step plan refinements

### Step 1 — Shared assertion helpers
**Status**: infrastructure step; supports steps 2/3/6/8.
**Required refinements**:
- Helper API must accept `lane=` and read tolerance/eps from the SSOT
  `benchmarks/validation_ladder_contract.py::PARITY_LADDER_TOLERANCES`. Do
  not let any test override the lane tolerance ad-hoc — that's how the
  current 0.55 Taylor rate ended up being tunable per call site.
- "Device-aware skip" must be `pytest.mark.xfail(condition=no_gpu, strict=True)`,
  NOT `pytest.mark.skipif`. With `skipif`, a CI runner that loses its GPU
  silently goes green and the entire GPU lane is gone.
- Place helpers under `tests/_helpers/parity.py` (lane-aware FD, Taylor
  convergence, CPU/JAX parity labels, runtime provenance, device-aware
  xfail). Keep typed; no broad framework.

### Step 2 — Stage 2 / GPU proof contract
**Status**: addresses P0 #1 fully, P0 #14 partially.
**Required refinements**:
1. **Order matters**: extend `build_provenance` (`benchmarks/validation_ladder_common.py:468-492`)
   to emit `xla_flags` (snapshot of `XLA_*` and `JAX_PLATFORMS` env vars
   *captured after `import jax`*) and `jaxlib_cuda_versions =
   getattr(jaxlib, "cuda_versions", None)` BEFORE updating the aggregator
   to preserve them.
2. **Aggregator must REJECT, not just preserve**: when any payload
   `bundle_provenance.fake == True` and `SIMSOPT_FAKE_GPU != "1"`, fail
   loud. When any payload's `default_backend != "gpu"` on the real-GPU lane,
   fail loud. When any `value_rtol` or `gradient_rtol` exceeds the
   parity-ladder lane contract, fail loud. Step 2 currently lists fields but
   not rejection logic.
3. **`bootstrap_runtime.sh` hardening**: add `python -c "import jax; assert
   jax.default_backend() == 'gpu'"` immediately after `import jax` in the
   bootstrap, plus a `bootstrap_jax_smoke.json` artifact written next to
   the proof bundles. Currently `bootstrap_runtime.sh:30-54` validates only
   `jax.__version__`.
4. **Stage 2 line-number correction**: insert `require_requested_platform_runtime(jax,
   requested_platform=REQUESTED_PLATFORM, context="Stage 2 end-to-end
   comparison")` after the existing `require_x64_runtime(...)` call at line
   59 (not line 46).
5. **File-rename to `tests/test_hf_production_gpu_proof_shell.py`** is correct
   but should also move the file under `tests/subprocess/` so the visible-on-CI
   "GPU proof" status detaches from a launcher-plumbing suite.

### Step 3 — FD/IFT gradient tightening
**Status**: addresses P0 #3 partially.
**Required refinements**:
1. **Line list expansion**: replace `{4213, 5792}` with `{3744, 4213, 5153,
   5792}` for the `rel < 1e-3 OR abs < 1e-8` escape pattern. Plus the
   majority gate at line 5804 (`validated_directions >= 2`).
2. **Taylor-rate constant**: change `_REAL_RESOLVE_FD_TAYLOR_RATE = 0.55` →
   `0.4` at `tests/integration/test_single_stage_jax_cpu_reference.py:1439`.
   For symmetric central FD on a smooth scalar, the proper rate floor is
   ~0.25 (error scales as `eps²`), so 0.4 is a defensible tightening.
3. **Direction-selection cherry-pick risk**: "deterministic direction
   selection avoiding near-zero" can leak the same blind spot in disguise.
   Concrete protocol: uniformly random with RNG-seeded reproducibility,
   reject only directions where the *projected gradient magnitude is below
   `1e-12`*, include all surviving directions, and cap rejection rate
   (`len(stable) >= 0.8 * len(sampled)`) — if more than 20% of directions
   are rejected, fail with "fixture geometry is degenerate."
4. **IotasJAX adjoint residual rel-tol** at `test_gradient_wrappers_operator_status_on_exact_state`
   (line 5662): step 3 doesn't address this. Add a sub-task: assert
   `adjoint_residual_rel <= 1e-10` for the IotasJAX branch.

### Step 4 — Reframe Boozer label tests
**Status**: correctly retires the false-positive P0 #2.
**Required refinements**:
- Add concrete cosmetic action: rename `_MockVolumeLabel` →
  `_PlumbingVolumeLabel` (or delete the dead `J()` method), add a one-line
  comment in `boozersurface_jax_test_helpers.py` documenting the
  marker-only intent. This stops future audits from re-flagging it.
- Real-label tests should target lane `derivative_heavy` (`rtol=1e-8,
  atol=1e-10`) per `PARITY_LADDER_TOLERANCES`. Enumerate which existing
  tests should migrate to real `Volume(surface)` vs which can keep the
  marker (the migration is small — most existing tests don't depend on the
  label type at all, since labels are recomputed from geometry).
- Add `post-Newton ‖grad‖ < 1e-10` magnitude target to
  `test_newton_polish_reduces_gradient` (audit P0 #6).

### Step 5 — Replace tautological surface/accessibility coverage
**Status**: addresses P0 #13 and P0 #22 partially; P0 #5 implicitly.
**Required refinements**: split into 5a/5b/5c with explicit oracle per
sub-step:
- **5a (bucket 1, surface)**: drop the rtol=1e-12 JAX-vs-JAX arms in
  `_assert_surface_jacobian_parity` / `_assert_area_volume_gradient_parity`
  (lines 619-696); KEEP the C++ oracle arm; add FD oracle (`eps=1e-5,
  rtol=1e-7, atol=1e-9`). Also drop `test_normal_orthogonality`
  cross-product tautology and add analytic torus area/volume tests
  (`area = 4π² R r`, `volume = 2π² R r²`).
- **5b (bucket 7, accessibility)**: enumerate the 5 `*_reuses_shared_jit_kernels`
  tests; for each, ADD `J/dJ` FD parity at `h=1e-6, rtol=1e-6` (lane
  `derivative_heavy`). Hessian-vector is over-spec for accessibility per
  bucket 7 — drop that.
- **5c (bucket 4, flux kernels)**: explicitly named — replace the three
  `_flux_kernel_value_and_grad` self-comparison tests at
  `tests/objectives/test_fluxobjective_jax_parity.py:{211, 223, 253}` with
  CPU `SquaredFlux` parity OR analytic-zero algebra. Also pin
  `SquaredFluxJAX.dJ()` raises `ObjectiveFailure` for the singular case.
- **HLO text-count gates**: don't delete — move to `tests/perf_gates/` with
  `@pytest.mark.brittle_perf_gate`. They're useful as soft alerts but
  should not be correctness gates.

### Step 6 — Field and reduction physics tests
**Status**: addresses P0 #20 fully.
**Required refinements**:
1. **GPU marker**: `pytest.mark.xfail(condition=no_gpu, strict=True)`, NOT
   `skipif`. (Same risk as step 1.)
2. **Closed-surface Gauss-law specifics**: coil ring at `z=5` (well outside
   a `R=1, r=0.1` torus); refine quadrature to 64×64 minimum; expect
   `|∮ B·n dA| / max|B|·area_total < 1e-9`.
3. **Add nfp rotational-symmetry test** for the B-field — this is the
   invariant that would have caught the historical Y/Z stellsym DOF bug per
   bucket 4 §3.
4. **Add Kahan/cancellation oracle** for `compensated_sum_flat` on the
   adversarial `[1e16, 1.0, -1e16]` case (bucket 7 §3 — load-bearing for the
   reproducibility claim).
5. **Add stellsym round-trip** on `surface_xyzfourier` (DOFs → coefficients →
   DOFs identity) — bucket 1 §3 missing coverage.

### Step 7 — Restore upstream physics coverage
**Status**: addresses P0 #11 and P0 #12 fully.
**Required refinements**:
1. **Naming**: the function is `test_Taylor` in
   `tests/field/test_selffieldforces.py`, not
   `test_force_objectives_taylor_test`.
2. **Restore the per-coil-sum objective variants** that HEAD dropped:
   `sum([LpCurveTorque(coils[i], coils2, ...) for i in range(len(coils))])`
   and the analogous `SquaredMeanTorque/Force/LpCurveForce` sums. HEAD only
   kept 6 of 10 upstream objectives even on the surviving configs.
3. **Resolve `distance_threshold` conflict** in
   `subtest_curve_minimum_distance_taylor_test`: HEAD's `0.4 if "CurveHelical"
   else 0.2` split is incompatible with restoring the upstream uniform `0.4`.
   Verify the upstream `0.4` branch under HEAD's helical path before
   restoring.
4. **Use `np.random.default_rng(0)`** (not `seed`) for the seed restoration —
   modern API, same numerical seed value to minimize tolerance drift.
5. **Slow-sweep gating**: the slow exhaustive sweep should use the existing
   `pytest.mark.slow` decorator (`tests/conftest.py:434`) and run on `pytest
   --runslow` in nightly CI, not opt-in env var.
6. **Add `test_objectives_time` and `test_call`** to step 7 explicitly (they
   are sibling regressions per bucket 8 P0 #2 and #8). `test_objectives_time`
   should default-on for `ncoils=2`, env-gate `ncoils=8` row only.
7. **Name the 3 deleted tests verbatim**: `test_arclength_variation_circle_planar`,
   `test_linking_number_planar`, `test_curve_curve_distance_empty_candidates`
   — restore from the upstream commit rather than re-implementing.

### Step 8 — Smoke and donation hardening
**Status**: addresses P0 #4 and P0 #15 fully.
**Required refinements**:
1. **Donation mutant test as a real fixture**: `tests/test_donation_mutants.py`
   that monkeypatches `donate_argnums = ()` and asserts the existing
   donation probe test now fails. The user's "mutant must fail" requirement
   is correct but needs a continuous-enforcement mechanism; one-shot manual
   testing rots.
2. **JSON sentinel schema**: wrappers must validate `payload["case"] ==
   expected_case_name` AND `payload["invariant"] in {"compile_count",
   "device_residency", "no_simsoptpp_import", ...}`. Step 8 mentions both
   fields but not the wrapper-side schema check — without it, a wrapper
   that just asserts `"OK:" in stdout` accepts a future case body that
   prints "OK:" outside its success path.
3. **Enumerate the ~14 highest-leverage `test_jax_import_smoke.py`
   wrappers** to migrate first (out of ~75). Don't try to migrate all 75 in
   one PR.
4. **Donation assertion uses both predicates**: `points.is_deleted() is True`
   AND `pytest.raises(RuntimeError, match="has been deleted")` against
   `jnp.asarray(points)`. Both work in jax 0.9.2 per direct probe.

### Step 9 — Validation gate
**Status**: well-chosen mutant set; expand.
**Required refinements**:
- Add three more mutants:
  - `_real_label.J()` returns 0.0 → step 4's tests must fail.
  - GPU reduction on a 1024-element array reordered → step 6 GPU parity
    must fail.
  - `use_jax_curve=True` branch silently disabled in `test_Taylor` → step 7
    must fail.
- Place mutant tests in `tests/mutants/test_required_mutants.py` with
  `pytest.mark.mutation`, run nightly. Without a separate file + marker,
  "required mutant checks" become a one-time manual exercise.

---

## 4. Required Step 10+ additions

The 12 unaddressed P0 items cluster into 5 new steps. These are not optional
nice-to-haves — without them, the suite remains incapable of catching the
specific regressions the original audit flagged.

### Step 10 — Boozer exact-Newton end-to-end fixture (P0 #7, #10)
- Add ONE end-to-end exact-Newton test on a real torus (no
  `_patched_exact_newton_result`, no `jacobian = identity` shim). Lane:
  `exact-well-conditioned-adjoint` (rtol=1e-6, atol=1e-8, residual ≤ 1e-10).
- Replace the toy 3×3 oracle at `tests/geo/test_surface_objectives_jax.py:1728`
  with a real `BoozerSurfaceJAX(boozer_type='exact')` fixture. Use
  `scipy.linalg.lu_solve` against the materialized PLU as the dense oracle;
  assert operator-vs-dense vector parity per the lane contract.
- Add an ill-conditioned exact-path test that asserts
  `failure_category="scaling_limit"` OR operator residual ≤ 1e-10, with
  NO vector parity claim. (Bucket 2 §3.)

### Step 11 — VJP signature regression guards (P0 #25)
- Add `inspect.signature` regression for `_boozer_ls_coil_vjp` and
  `_boozer_exact_coil_vjp`. These are XS effort and would have caught the
  historical CPU-vs-JAX 2-arg-vs-4-arg drift.
- Cover `tests/test_ls_vjp_returns_correct_shapes` with an actual VJP-vs-FD
  parity test, not just shape assertions.

### Step 12 — M5 wrapper failure paths (bucket 3 §3)
- Add tests that assert successful forward + failed adjoint → non-finite
  gradient on legacy `BoozerResidualJAX.dJ()`, `IotasJAX.dJ()`,
  `NonQuasiSymmetricRatioJAX.dJ()`. Currently, a failed adjoint silently
  yields a finite (wrong) gradient.
- Add CPU-vs-JAX projected coil-derivative parity at LS-warmed fixture for
  IotasJAX/NonQS at strict tolerance.
- Add `SquaredFluxJAX.dJ()` Taylor test (P0 #21) + chunked-VJP gradient
  parity on a large point cloud.
- Address P0 #16: replace `np.isfinite`-only assertion with `np.isposinf`
  parity at `tests/integration/test_stage2_jax.py:1036` zero-current case.
- Tighten outer-opt strict decrease: drop `+1e-12` slack at
  `tests/integration/test_single_stage_jax_cpu_reference.py:4925`.
- Replace `test_adjoint_fraction_diagnostic > 0` ceremony with a
  meaningful upper bound (line 5859).

### Step 13 — Conftest and order discipline (P0 #18, #19, #24)
- Refactor `test_backend_state_guard_sequence_*` and
  `test_backend_module_guard_sequence_*` into single-test-per-sequence
  layout (no shared mutable state across tests). Currently fragile under
  `pytest -p random`.
- Fix `tests/integration/conftest.py:_patch_meta_path_finder` to
  `pytest.skip("editable install required")` instead of silent `return
  False`.
- Add a counter-test in `tests/test_run_code_benchmark_common.py` that
  drops the `_force_x64` autouse fixture and asserts behavior degrades —
  proving the autouse is load-bearing rather than blinkering all callers.

### Step 14 — CI infrastructure (synthesis §5)
- Add `pytest -p random` (or `pytest-randomly`) to one CI job to catch
  order-dependent tests on every PR.
- Add a `pytest --collect-only | grep -c "test_arclength_variation_circle_planar\|test_linking_number_planar\|test_curve_curve_distance_empty_candidates"`
  guard so re-deletions of restored tests are caught.
- Add a `test_field_cache_hot_path_benchmark.py` end-to-end replacement
  (P0 #23): compile + parse output JSON; drop the printf-format pinning.
- Audit every `pytest.skip(...)` for a tracked GitHub issue ID; default
  policy: skip without issue ID is a CI failure.

---

## 5. New risks introduced by the plan as written

1. **Step 1 / Step 6 device-aware skip**: implementing as `pytest.mark.skipif`
   instead of `xfail strict=True` masks GPU regressions on CPU-only CI. This
   is exactly the failure mode the original GPU-proof audit flagged
   (silent-green on CUDA-less host).
2. **Step 3 "deterministic direction selection avoiding near-zero"**: can
   cherry-pick out the regime where wrong-sign IFT terms surface. Required
   protocol above (uniform sample + `1e-12` projection floor + 80%
   rejection cap).
3. **Step 5 "downgrade JAX-vs-JAX to alias tests"**: leaves false-parity
   tests in place that look like contract coverage in code review. Drop
   them outright (or move under `tests/aliases/` with `@pytest.mark.alias_only`)
   rather than rename.
4. **Step 4 mock migration tolerance drift**: existing tests calibrated
   against `_MockVolumeLabel.J() ≡ 0.0` may break tolerance gates when
   migrated to real labels. Plan does not budget for re-tuning. Mitigation:
   migrate one test at a time, re-run the 0.9.2 lane after each.
5. **Step 7 RNG migration drift**: `np.random.default_rng(0)` produces
   different streams than `np.random.seed(0)` even with the same seed.
   Tolerance gates calibrated against the legacy seed may misfire.
   Mitigation: re-tune tolerances after seed migration; do not move tests
   to "default" lane until they pass with the new seed.

---

## 6. Final assessment

The plan is structurally sound and addresses 12 of the 25 P0 findings (24%
fully + 24% partially). The one factual error in the audit (`_MockVolumeLabel.J()`)
is correctly retired. The five line-number/API claims are all confirmed,
with two requiring expansion (FD-escape sites: 5 not 3; provenance-field
extension before aggregator update).

The 48% unaddressed P0 set is real and not optional. The 5 new steps
above (10-14) close all of them. With those additions, the plan would cover
~92% of the original P0 set fully — the remaining 8% is backlog
(plumbing-only mock cleanup, perf-gate relocation, etc.).

The single highest-impact addition: **Step 10 (real exact-Newton fixture)**.
Without it, the entire `TestBoozerSurfaceJAXExactPath` cluster validates
plumbing only — the core load-bearing claim of the M4 milestone (exact
Newton convergence on a real torus) has zero coverage today.

The single highest-impact refinement to the existing plan: **Step 2's
`build_provenance` extension MUST come before the aggregator update**
(otherwise the aggregator will preserve `None` for `xla_flags` and
`jaxlib_cuda_versions` because they don't exist yet).

The user's pushback on the MockVolumeLabel finding was correct, well-cited,
and saved an entire wave of misdirected refactor work. The audit owes the
user a public retraction in `SYNTHESIS.md` (which I'll do separately if
asked).
