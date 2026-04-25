# Plan vs. Findings — 9-step plan cross-check against 2026-04-25 audit

Date: 2026-04-25
Branch: `gpu-purity-stage2-20260405` (HEAD `42b68f33d`)
Inputs: 9-step plan (provided) + bucket1..8 + SYNTHESIS.md.

This document does **not** re-audit tests. It maps the plan onto the 25 P0
findings catalogued in `SYNTHESIS.md` §3 and cross-references the wider 195-issue
backlog spread across `bucket{1..8}_*.md`.

---

## 1. Coverage matrix — 25 P0 findings (from SYNTHESIS §3)

Status: **A** = fully addressed by plan; **B** = partially addressed (gaps
listed); **C** = not addressed; **D** = explicitly retired by plan author.

| P0 # | Findings (one-line) | Plan step(s) | Status | Notes |
|-----:|---------------------|--------------|:----:|-------|
| 1 | GPU proof file: rename + `require_requested_platform_runtime` + provenance fields + real-GPU lane | 2 | **A** | Step 2 enumerates rename/payload extension/aggregator update/lane split exactly. |
| 2 | `_MockVolumeLabel.J() ≡ 0.0` silences M4 constraint physics | 4 | **D** | Plan author has explicitly declined to implement P0 #2 ("DO NOT implement"); replaces with real-label tests that perturb `target_label`/geometry. The audit's recommended fix (compute `surface_volume`) is rejected; plan's alternative (real-label coverage) addresses the physics gap from a different angle. |
| 3 | OR-escape + 2-of-3 / 3-of-5 + Taylor 0.55 in resolve-FD helpers (lines 4213, 5792, 1979) | 3 | **B** | Step 3 explicitly drops `or abs < 1e-8`, requires all directions, mentions deterministic direction selection and stricter halving-eps central-FD. **MISSING**: explicit Taylor rate change from 0.55 → 0.4 at line 1979 (`_REAL_RESOLVE_FD_TAYLOR_RATE`). Plan does not list `tests/integration/test_single_stage_jax_cpu_reference.py:1979` by line number — only 4213 and 5792 are cited. |
| 4 | Donation probe never asserts `points.is_deleted()` | 8 | **A** | Step 8 spells out "donation probe holds donated input, calls positionally, blocks, asserts `is_deleted()`; mutant `donate_argnums=()` must fail." |
| 5 | Three TAUTOLOGICAL kernel "contract" tests at `test_fluxobjective_jax_parity.py:211, 223, 253` | 5 | **B** | Step 5 says "downgrade JAX-vs-JAX to alias tests" and "add real J/dJ/Hessian-vector/FD checks" — but step 5 is scoped to "surface/accessibility" tests. Step 5's wording ("HLO text-count gates") points at bucket 1 + bucket 7, not bucket 4 flux kernels. **Bucket 4 P0 #5 is not explicitly enumerated**. The flux-kernel singular-grad case is also unaddressed (audit recommends `SquaredFluxJAX.dJ()` raise `ObjectiveFailure`). |
| 6 | `test_gradient_wrappers_operator_status_on_exact_state` IotasJAX branch needs residual rel-tol gate (line 5662) | — | **C** | Plan does not mention the residual-rel-tol assertion for the IotasJAX branch. Step 3 covers FD/IFT directional gradients; this finding is about adjoint operator solve residual norm. |
| 7 | Replace toy 3×3 oracle at `test_surface_objectives_jax.py:1728` with real `BoozerSurfaceJAX(boozer_type='exact')` fixture + `scipy.linalg.lu_solve` | — | **C** | Plan does not address this. Step 5 mentions accessibility (which is bucket 7), not exact-well-conditioned-adjoint lane plumbing in bucket 3. |
| 8 | `test_adjoint_fraction_diagnostic` is a `>0` ceremony (line 5859) | — | **C** | Not addressed. Plan does not promote/delete this. |
| 9 | `test_outer_opt_decreases_objective` `+1e-12` slack (line 4925) | — | **C** | Not addressed. Plan does not mention strict-decrease assertion. |
| 10 | TestBoozerSurfaceJAXExactPath cluster needs ONE end-to-end exact-Newton test on real torus (no `_patched_exact_newton_result`) | — | **C** | Not addressed. Plan does not require any `_patched_exact_newton_result` removal. |
| 11 | `test_force_objectives_taylor_test`: 320 → 12 sub-cases | 7 | **A** | Step 7 explicitly: "for force-objective Taylor: default covering set + slow exhaustive sweep; default must cover both CPU/JAX curve paths, both regularization families, nonzero threshold, downsampling, multiple nfp." |
| 12 | Restore `downsample` loop + `np.random.seed(0)` + `use_jax_curve` parametrization + 3 deleted upstream tests | 7 | **A** | Step 7: "Restore upstream physics coverage — restore 3 deleted curve-objective tests; reintroduce `np.random.default_rng(seed)`." |
| 13 | Drop tautological `_assert_surface_jacobian_parity` / `_assert_area_volume_gradient_parity` JAX-vs-JAX arms (lines 619-696) | 5 | **B** | Step 5 reads "remove HLO text-count gates; downgrade JAX-vs-JAX to alias tests" — covers the JAX-vs-JAX parity arms in spirit. **MISSING**: explicit FD-oracle replacement (audit recommends `eps=1e-5, rtol=1e-7, atol=1e-9`). Also "alias tests" framing is weaker than what the audit recommends ("Drop the rtol=1e-12 JAX-vs-JAX arms; keep only the C++ oracle arm"). |
| 14 | `stage2_e2e_comparison.py`: `require_requested_platform_runtime` + bundle parity fields + aggregator rejection | 2 | **B** | Step 2 lists `require_requested_platform_runtime` and "extend payloads with backend/devices/jaxlib/CUDA/XLA flags/repo SHA/probe name" — covers provenance. **MISSING**: aggregator rejection of bundles when value/gradient rtol exceeds the parity-ladder contract (audit P0 #14 explicitly demands rejection logic, not just schema fields). |
| 15 | Compile-count subprocess wrappers must parse JSON payload `compile_count == 1` | 8 | **A** | Step 8: "subprocess emit JSON sentinel with case/checked/invariant fields." Captures compile-count visibility. |
| 16 | Singular zero-current parity must pin `np.isposinf` parity, not just `np.isfinite` | — | **C** | Not addressed. Plan does not mention `tests/integration/test_stage2_jax.py:1036`. |
| 17 | Replace `test_normal_orthogonality` tautology + add analytic surface area/volume tests | 5 | **B** | Step 5 mentions "real J/dJ/Hessian-vector/FD checks for accessibility objectives" — accessibility lives in bucket 7 (`test_accessibility.py`). Bucket 1 surface analytic tests (torus area/volume) are NOT explicitly covered. The cross-product orthogonality tautology is also not enumerated. |
| 18 | Order-dependent `test_backend_state_guard_sequence_*` and `test_backend_module_guard_sequence_*` need single-test refactor | — | **C** | Not addressed. Plan does not touch ordering/sequence tests. |
| 19 | `tests/integration/conftest.py:_patch_meta_path_finder` returning `False` silently | — | **C** | Not addressed. Plan does not mention the conftest skip-vs-silent-False fix. |
| 20 | Add divergence-theorem invariant to `test_integral_bdotn_jax.py` | 6 | **A** | Step 6: "Add field and reduction physics tests — closed-surface ∮B·n dA conservation." Direct match. |
| 21 | Add `test_squaredfluxjax_dJ_taylor_test` + chunked-VJP large-point-cloud parity | — | **C** | Not addressed. Plan does not mention Taylor test on `SquaredFluxJAX.dJ()` or chunked-VJP gradient parity. |
| 22 | 5 `*_reuses_shared_jit_kernels` accessibility tests need FD parity at `h=1e-6, rtol=1e-6` | 5 | **B** | Step 5 says "add real J/dJ/Hessian-vector/FD checks for accessibility objectives." This is the bucket 7 accessibility cluster. **Coverage is on point** but the plan wording is generic and doesn't enumerate which 5 tests; risk is implementor only fixes 1 or 2. |
| 23 | TAUTOLOGICAL printf-format pinning in `test_field_cache_hot_path_benchmark.py` — needs end-to-end compile+JSON parse | — | **C** | Not addressed. Plan does not mention this benchmark. |
| 24 | Add counter-test in `test_run_code_benchmark_common.py` that drops the `_force_x64` autouse | — | **C** | Not addressed. |
| 25 | Replace shape-only `test_ls_vjp_returns_correct_shapes` + add `inspect.signature` regression for `_boozer_ls_coil_vjp`/`_boozer_exact_coil_vjp` | — | **C** | Not addressed. Plan does not introduce VJP signature regression guards. |

### Aggregate counts

- **A** (fully addressed): 6 — P0 #1, #4, #11, #12, #15, #20
- **B** (partial): 6 — P0 #3, #5, #13, #14, #17, #22
- **C** (not addressed): 12 — P0 #6, #7, #8, #9, #10, #16, #18, #19, #21, #23, #24, #25
- **D** (explicitly retired): 1 — P0 #2

Plan covers **24 % fully**, **24 % partially**, **48 % unaddressed**, **4 %
explicitly retired**.

---

## 2. Step-by-step assessment

### Step 1 — Shared assertion helpers (FD, Taylor, parity labels, runtime provenance, device-aware skip)

- **Coverage**: This step is infrastructure, not a finding-fix. It supports
  steps 2/3/6/8 by giving them shared idioms. No P0 finding directly maps to
  step 1.
- **Missing detail**:
  - No mention of the SSOT contract in
    `benchmarks/validation_ladder_contract.py::PARITY_LADDER_TOLERANCES`. The
    audit explicitly recommends a `taylor_test_with_floor(lane=...)` helper that
    *consumes* the SSOT lane label so tolerances cannot drift in the test layer.
  - No mention of where the helpers live (`tests/_helpers/parity.py`?
    `tests/_helpers/jax.py`?) — risk of three teams writing competing helpers.
  - "Device-aware skip" is named but its predicate is not specified. The audit
    explicitly warns (bucket 6, §6) that any new `pytest.mark.skipif(jax.default_backend() != "gpu", ...)`
    must default to `strict=True` so a missing GPU on the GPU lane is RED, not
    skipped. Step 1 does not call this out.
- **Risk**:
  - "Device-aware skip" implemented incorrectly **masks real GPU regressions**.
    If the predicate is "skip when no GPU is present", a CI job that loses its
    GPU device silently goes green.
  - Sharing a `central_fd` helper makes it tempting to centralise `eps` choice
    — the audit (bucket 1 #7, #8) shows that `eps=1e-7` is wrong for some
    kernels and `eps=1e-5` is wrong for others. A single shared `eps` default
    would either over- or under-shoot for many call sites.
- **Suggested refinement**: Helper API must (a) accept `lane=` and read the
  tolerance/eps from `PARITY_LADDER_TOLERANCES`, (b) make the skip predicate
  explicitly `xfail strict=True` for GPU lanes, (c) reject calls that try to
  override the lane tolerance ad-hoc.

### Step 2 — Stage 2 GPU proof contract

- **Coverage**: Addresses P0 #1 fully (rename + payload extension + aggregator +
  lane split). Addresses P0 #14 partially (provenance fields, but rejection
  logic for out-of-tolerance bundles is not enumerated).
- **Missing detail**:
  - **Aggregator rejection logic**: Audit P0 #14 explicitly demands "reject the
    bundle when any rtol exceeds the parity-ladder contract." Step 2 lists the
    fields but doesn't say the aggregator must fail-loud on tolerance breach.
  - **Bundle provenance disambiguation**: Audit recommends `bundle_provenance.runner` and
    `bundle_provenance.fake = True` so the aggregator refuses GREEN unless
    `SIMSOPT_FAKE_GPU=1`. Step 2's "backend/devices/jaxlib/CUDA/XLA flags" list
    omits the explicit fake-vs-real provenance discriminator.
  - **`bootstrap_runtime.sh` strengthening** (audit P0-adjacent #5 in bucket 6's
    playbook) is omitted: the audit wants `jax.default_backend() == "gpu"`
    asserted *immediately after* `import jax` in the bootstrap, plus a
    `bootstrap_jax_smoke.json` artifact. Step 2 only touches the proof
    aggregator, not the bootstrap.
  - **LD_LIBRARY_PATH dynamic-loader smoke** (audit P0 #6 in bucket 6) is
    omitted. Currently `test_run_production_gpu_proof_preserves_ld_library_path`
    only echoes env vars; audit demands a real `import jaxlib.cuda_versions`
    invocation under the real lane.
  - **CPU-vs-GPU parity assertion in the bundle aggregator** (audit P0 #7 in
    bucket 6) is implicit but not enumerated. Step 2 mentions "backend/devices"
    but not `cpu_oracle_value` / `gpu_value` / `value_rtol` / `gradient_rtol`
    fields.
- **Risk**:
  - Renaming the file alone (without moving it under `tests/subprocess/`) keeps
    the visible-on-CI status of "GPU proof" attached to a launcher plumbing
    suite. The audit recommends moving it under `tests/subprocess/`.
  - Splitting "launcher / schema / real-GPU" lanes is correct, but if the
    real-GPU lane uses `pytest.mark.skipif` (not `xfail strict=True`), a CI
    runner without a GPU silently goes green and the entire purpose is lost.
- **Suggested refinement**: Add to step 2: aggregator MUST fail when (a) any
  payload `bundle_provenance.fake == True` without `SIMSOPT_FAKE_GPU=1`, (b)
  any payload `value_rtol` or `gradient_rtol` exceeds the lane contract, (c)
  any payload's `default_backend != "gpu"` on the real-GPU lane. Add bootstrap
  hardening as a sub-task.

### Step 3 — Tighten FD/IFT gradient tests

- **Coverage**: Directly addresses P0 #3 — the OR-escape + majority gate
  pattern. Mentions deterministic direction selection and stricter
  halving-eps central FD.
- **Missing detail**:
  - **Line 1979** (`_REAL_RESOLVE_FD_TAYLOR_RATE = 0.55`): plan cites only
    lines 4213 and 5792. The synthesis (§1.2) explicitly calls out 1979 as the
    helper file. Without changing this line, the resolve-FD helpers (used by
    NQSR/BoozerResidual) keep the 0.55 rate. Audit recommends 0.4.
  - **`_REAL_RESOLVE_FD_MIN_STABLE_SAMPLES = 3`** (the 3-of-5 majority gate):
    plan says "require ALL nonzero directions to pass" — this is the right
    semantic but doesn't name the constant or the file. Implementor risk if
    this constant lives in a different place than the assertion.
  - **`_ADJOINT_RESIDUAL_REL_TOL`** at the IotasJAX adjoint check: P0 #6 wants
    `<=1e-10`. Step 3 doesn't address this.
- **Risk**:
  - **"Deterministic direction selection avoiding near-zero" can be cherry-picking.**
    If the implementor selects directions that yield large-magnitude gradients,
    the test stops sampling the near-zero regime where adjoint sign errors
    surface. The bucket-3 audit notes the 2-of-3 majority gate hides
    wrong-sign IFT terms on individual directions; a deterministic-but-skewed
    direction set could leak the same blind spot in disguise.
  - **Stricter halving-eps central FD**: if the new tolerance is set at
    `eps=1e-5, rtol=1e-7` without ε-ladder, a kernel with an `O(eps)` truncation
    artifact (pole-near 1/|B| weighting, see bucket 2 #25) will fail; the
    implementor will then loosen the tolerance again.
  - **"Near-zero gradient tests separated"**: this risks creating a separate
    "easy" suite that becomes the de-facto spec, while the strict suite is
    relegated to nightly-only. Audit recommends keeping near-zero in the same
    suite but with a `fd_gradient` lane label and a `rtol=1e-5` floor.
- **Suggested refinement**: Add "include line 1979 / `_REAL_RESOLVE_FD_TAYLOR_RATE`"
  explicitly. Specify direction selection as "uniformly random with
  RNG-seeded reproducibility, reject only directions where the *projected
  gradient magnitude is below `1e-12`*; include all surviving directions."
  Cap the cherry-picking risk by requiring `len(stable_samples) >= 0.8 * len(direction_samples)`.

### Step 4 — Reframe Boozer label tests

- **Coverage**: This is P0 #2 *retired* — the audit recommended computing real
  `surface_volume`; plan author explicitly declines and replaces with real-label
  perturbation tests.
- **Missing detail**:
  - "Keep mock for plumbing only or rename" is unspecific. Without a concrete
    rename or guard, future tests will continue to import `_MockVolumeLabel`
    for physics tests by accident.
  - The audit lists ~70 % of M4 tests as affected. Plan does not enumerate
    which existing tests must be migrated to real labels vs. which can keep
    the mock.
  - Audit's "real label tests perturbing target_label, perturbing geometry,
    comparing against CPU label values" is correct; missing is the FD/Taylor
    tolerance lane (presumably `derivative_heavy`, `rtol=1e-8`).
- **Risk**:
  - **"Keep mock for plumbing only or rename"**: if the rename happens but
    existing tests are not migrated, the new physics tests are appended on
    top of the existing tautological tests. Test count goes up; signal
    doesn't.
  - **Migration risk**: ~70 % of `TestBoozerSurfaceJAXClass` tests share
    `_MockVolumeLabel`. Migrating them naively to real `Volume(surface)` will
    change penalty values and may break tolerance gates that were calibrated
    against the mock-zeroed constraint contribution. Plan does not budget for
    re-tuning these tolerances.
- **Suggested refinement**: Specify (a) rename `_MockVolumeLabel` →
  `_PlumbingVolumeLabel` and document its non-physics use, (b) enumerate the
  7-12 highest-leverage tests to migrate to real labels (`test_penalty_with_volume_label`,
  `test_run_code_ls_converges`, `test_penalty_gradient_fd`, `test_newton_polish_reduces_gradient`,
  `test_penalty_with_toroidal_flux`, `test_penalty_with_area_label`, etc.),
  (c) require new tests use lane `derivative_heavy` (rtol=1e-8) and document
  the migration in `CLAUDE.md`.

### Step 5 — Replace tautological surface/accessibility coverage

- **Coverage**: Addresses P0 #13 (surface_rzfourier JAX-vs-JAX) and P0 #22
  (5 accessibility tests) partially. Wording is generic.
- **Missing detail**:
  - **Bucket 4 P0 #5** (three TAUTOLOGICAL kernel tests in
    `test_fluxobjective_jax_parity.py:211, 223, 253`) is implicitly covered by
    "downgrade JAX-vs-JAX to alias tests" — but step 5 is titled
    "surface/accessibility coverage" and does not explicitly mention bucket 4.
    Implementor risk: bucket 4 tautologies are missed.
  - **Bucket 1 P0 #17** (`test_normal_orthogonality` cross-product tautology
    + missing analytic torus area/volume) is not enumerated.
  - **Bucket 7 P0 #22** "FD parity at `h=1e-6, rtol=1e-6`" is named at the
    correct lane but tolerance is not anchored to the SSOT
    `PARITY_LADDER_TOLERANCES` lane (audit recommends the `derivative_heavy`
    or `fd_gradient` lane).
  - **Hessian-vector / Hessian** for accessibility objectives: step 5 mentions
    "Hessian-vector/FD checks" but accessibility objectives mostly need J/dJ
    only (per bucket 7). This is over-spec on accessibility but under-spec on
    flux/surface.
- **Risk**:
  - "Alias tests" is weaker than the audit's "drop the JAX-vs-JAX arms
    entirely." If the alias tests stay, they look like parity tests at a
    glance and will be cited in code review as covering the contract — but
    they don't.
  - Replacing the C++ oracle arm without a fallback risks losing C++ parity
    coverage (audit explicitly says "keep only the C++ oracle arm").
  - HLO text-count gates being deleted is correct but the plan does not say
    where they should go (audit recommends `tests/perf_gates/` with
    `@pytest.mark.brittle_perf_gate`).
- **Suggested refinement**: Split step 5 into 5a (surface_rzfourier
  tautologies — bucket 1), 5b (accessibility — bucket 7), 5c (flux kernel
  tautologies — bucket 4). For each, enumerate (a) which assertions to drop,
  (b) which oracle to use (C++ for 5a, FD for 5b, CPU `SquaredFlux` for 5c).
  Move HLO-text gates to `tests/perf_gates/` rather than delete.

### Step 6 — Field and reduction physics tests

- **Coverage**: Addresses P0 #20 (Gauss-law / closed-surface ∮B·n dA = 0)
  fully. Adds GPU-vs-CPU reduction parity for `pairwise_sum_*`,
  `compensated_sum_flat`, `scalar_square_sum` — directly answers bucket 7's
  largest gap (GPU/CPU bitwise reduction parity).
- **Missing detail**:
  - "Under GPU marker" — implementor risk: if the marker is `pytest.mark.skipif(no_gpu)`
    rather than `xfail strict=True`, no-GPU CI silently goes green and the
    parity gate never fires. Audit (bucket 6 §6) explicitly warns about this.
  - **nfp rotational symmetry** (bucket 4 §3, bucket 1 §3): not mentioned.
    Audit calls out nfp B-field symmetry as the test that would have caught
    the historical Y/Z stellsym DOF bug. Step 6 only mentions ∮B·n dA.
  - **Stellsym round-trip on `surface_xyzfourier`** (bucket 1): not mentioned.
  - **Long-wire limit, vector-potential gauge** (bucket 1): not mentioned.
  - **Compensated sum cancellation case `[1e16, 1.0, -1e16]`** (bucket 7):
    not mentioned. Plan only lists "GPU-vs-CPU reduction parity," not the
    Kahan-vs-naïve oracle test.
- **Risk**:
  - **GPU parity tests under wrong skip predicate**: on a CPU-only CI, every
    single GPU parity test silently passes. The whole point of the test is
    moot.
  - **"Closed-surface ∮B·n dA conservation"** has a subtlety: the test must
    use coils *outside* the torus (else the integral is non-zero by Gauss's
    law). Plan doesn't say this explicitly; an implementor could put coils
    *inside* the torus and the test would correctly fail — but for the wrong
    reason.
- **Suggested refinement**: Specify the GPU marker as `xfail strict=True`,
  not `skipif`. Add nfp rotational-symmetry test as a sub-task. Specify that
  Gauss-law test uses a coil ring at `z=5` (well outside a `R=1, r=0.1`
  torus). Add Kahan/cancellation oracle test for `compensated_sum_flat`.

### Step 7 — Restore upstream physics coverage

- **Coverage**: Addresses P0 #11 (force objectives Taylor sweep 320 → 12) and
  P0 #12 (downsample loop, RNG, deleted tests, `use_jax_curve` parametrization).
  Direct enumeration: "default covering set + slow exhaustive sweep; default
  must cover both CPU/JAX curve paths, both regularization families, nonzero
  threshold, downsampling, multiple nfp."
- **Missing detail**:
  - **`test_objectives_time` env-gated to `SIMSOPT_RUN_FIELD_TIMING=1`**
    (audit S1 + bucket 8 P0 #2): not mentioned. Audit recommends restoring
    `ncoils=2` row default-on; gate `ncoils=8` row on env var.
  - **`test_call` follow-up** in `tests/core/test_optimizable.py:243-258`
    (bucket 8 P0 #8): not mentioned. 5 lines were silently dropped.
  - **3 deleted tests** named verbatim: `test_arclength_variation_circle_planar`,
    `test_linking_number_planar`, `test_curve_curve_distance_empty_candidates`.
    Plan says "restore 3 deleted curve-objective tests" — direct match but
    the plan should name them so the implementor uses the upstream commit
    rather than re-implementing.
  - **`np.random.default_rng(seed)`** vs upstream's `np.random.seed(0)`:
    upstream uses module-state seed; plan correctly migrates to `default_rng`
    (modern pattern). Good but unstated migration consequence: any other test
    in the same module that depended on `np.random` global state changes
    behaviour.
- **Risk**:
  - **"Default covering set + slow exhaustive sweep"** as two separate runs:
    if the slow sweep is gated on a CI env var, it never runs in dev and
    regressions slip through to nightly only. Plan should specify what `slow`
    means (a `pytest.mark.slow` decorator already exists per `tests/conftest.py:434`).
  - **Restoring tests verbatim from upstream** can fail if the JAX-port
    refactored downstream APIs. Plan doesn't budget for API adaptation.
  - **`np.random.default_rng(seed)`** in `test_curve_minimum_distance_taylor_test`:
    if the test was originally tuned against `np.random.seed(0)`, the
    direction sample distribution will change and the tolerance gate may
    misfire. Plan does not require re-tuning.
- **Suggested refinement**: Add `test_objectives_time` and `test_call` to step
  7 explicitly. Specify the slow sweep gates on `SIMSOPT_RUN_FULL_FORCE_SWEEP=1`
  (or reuse `pytest.mark.slow`). For the seed migration, require
  `default_rng(0)` (same seed value) to minimize tolerance drift.

### Step 8 — Harden smoke and donation

- **Coverage**: Addresses P0 #4 (donation probe) fully and P0 #15 (compile-count
  JSON sentinel) fully. JSON sentinel pattern matches the audit's gold-standard
  `test_single_stage_surface_reprojection_probe_emits_structured_cpu_result`.
- **Missing detail**:
  - **~75 wrappers in `test_jax_import_smoke.py`** (audit T4): plan says
    "subprocess emit JSON sentinel" — direct match — but doesn't enumerate
    that ~75 wrappers need to migrate. Implementor risk: migrating only the
    compile-count case but not the import smoke wrappers.
  - **Donation mutant test**: plan says "mutant `donate_argnums=()` must
    fail." This is correct but doesn't specify the mechanism (mutation testing
    via `mutmut`? CI-only mutation gate? Manual test?). Without the mechanism,
    the mutant test is aspirational.
  - **`tests/subprocess/jax_runtime_cases.py:36`** (module-level
    `_prefer_local_simsopt_source_tree()` collapsing all subprocess tests on
    one import error, bucket 5 #28): not mentioned.
- **Risk**:
  - **"Mutant `donate_argnums=()` must fail"**: if the implementor interprets
    this as "add a mutant test once and forget," there is no continuous
    enforcement. Mutation testing is an ongoing process, not a one-shot test.
  - **JSON sentinel without schema validation**: if the wrapper just asserts
    `"OK:" in stdout`, a future case body that prints "OK:" outside its
    success path passes silently. Audit recommends the JSON have a `case`
    field and an `invariant` field — plan mentions both but the wrapper-side
    schema check is implicit.
- **Suggested refinement**: Specify (a) the donation mutant test is a real
  pytest test that *patches* `donate_argnums` to `()` and asserts the existing
  donation probe test now fails (parametrize over donation invariants), (b)
  the wrapper validates `payload["case"] == expected_case_name` and
  `payload["invariant"] == "compile_count" / "device_residency" / ...`, (c)
  enumerate the ~14 most important `test_jax_import_smoke.py` wrappers to
  migrate first.

### Step 9 — Validation gate

- **Coverage**: Excellent meta-step. The required mutant checks are
  well-chosen:
  - "removing Stage 2 platform guard fails" — validates step 2.
  - "fake GPU payload without backend/devices fails" — validates step 2.
  - "flipping IFT gradient sign fails" — validates step 3.
  - "donate_argnums=() fails" — validates step 8.
  - "replacing surface CPU oracle with same JAX function not accepted" —
    validates step 5.
- **Missing detail**:
  - **No mutant for step 4** (real-label perturbation): a mutant that returns
    `0.0` from `_real_label.J()` should still fail.
  - **No mutant for step 6** (Gauss-law / GPU reduction parity): a mutant that
    reorders reduction summands on GPU should fail.
  - **No mutant for step 7** (force objective Taylor sweep): a mutant that
    silently disables the `use_jax_curve=True` branch should fail.
  - **No mutant for the Boozer exact-Newton path** (P0 #10) or the IotasJAX
    adjoint residual gate (P0 #6) — both unaddressed by the plan, so no
    validation gate either.
- **Risk**:
  - **Mutant tests as pytest fixtures** versus **CI-only mutation tooling**:
    the plan does not specify. Without a clear mechanism, "required mutant
    checks" become a one-time manual exercise rather than an ongoing gate.
  - **"Replacing surface CPU oracle with same JAX function not accepted"** is
    a vague mutant. A real implementation needs a `pytest.mark.mutation` test
    that monkeypatches the oracle and asserts the test fails with a specific
    error message.
- **Suggested refinement**: Add three more mutants: (a) `_real_label.J()`
  returns 0.0 → step 4's tests must fail, (b) GPU reduction on a 1024-element
  array reordered → step 6 GPU parity must fail, (c) `use_jax_curve=True`
  branch disabled in `test_force_objectives_taylor_test` → step 7 must fail.
  Specify the mutant infrastructure: separate `tests/mutants/test_required_mutants.py`
  file with `pytest.mark.mutation`, run nightly.

---

## 3. Findings NOT addressed (cross-referenced from buckets 1-8)

This section enumerates audit findings that survive across the bucket reports
but are NOT picked up by any of the 9 steps. Severity follows the audit's own
classification (P0 = highest, P1/P2 = backlog).

### Bucket 1 (M1 kernels)
- **Bucket 1 #6**: `test_div_B_zero` only checks on-axis; no off-axis 20-point
  test. **Severity**: P1. **Mitigation**: add to step 6 as a sub-task ("plus
  off-axis div-B at 20 random points").
- **Bucket 1 #15**: `test_surface_rz_geometry_hlo_probe_entrypoint_uses_local_package`
  is a subprocess test that only asserts a JSON-payload boolean produced by a
  benchmark probe. **Severity**: P1. **Mitigation**: move to `benchmarks/` test
  suite or add to step 5's "remove HLO text-count gates" scope.
- **Bucket 1 #17**: `test_on_axis_field` uses `nquad=256` for `rtol=1e-12`.
  Wasteful but correct. **Severity**: P2. **Mitigation**: backlog.
- **Bucket 1 §3**: missing **vector potential A gauge consistency** and
  **Coulomb-gauge `∇·A=0`** tests. **Severity**: P1. **Mitigation**: backlog
  or add to step 6.
- **Bucket 1 §3**: missing **stellarator-symmetry round-trip** check on
  `surface_xyzfourier`. **Severity**: P1. **Mitigation**: backlog or 10th plan
  step.

### Bucket 2 (M3/M4 Boozer)
- **Bucket 2 #1**: `_MockVolumeLabel` cluster — explicitly retired by step 4.
  Backlog item: actually migrate the 30+ tests using it (plan only mentions
  "real label tests").
- **Bucket 2 #6**: `test_newton_polish_reduces_gradient` — only checks
  direction-of-improvement, no magnitude target. **Severity**: P0 (audit
  cites). **Not in step 4**. **Mitigation**: add to step 4 ("post-Newton ‖grad‖
  < 1e-10 on real torus").
- **Bucket 2 #13**: `test_exact_well_conditioned_operator_adjoint_cpu_gpu_same_state_parity`
  uses synthetic A=diag fixture, not a real Jacobian. **Severity**: P0 (audit
  cites). **Not in plan**. **Mitigation**: add as 10th plan step or fold into
  step 4 (real exact-Newton fixture).
- **Bucket 2 #16 & #43**: `_successful_exact_newton_result` patched with
  `jacobian = identity` across ~14 tests (TestBoozerSurfaceJAXExactPath cluster).
  **Severity**: P0 (matches synthesis P0 #10). **Not addressed**. **Mitigation**:
  10th plan step "Add ONE end-to-end exact-Newton test on a real torus."
- **Bucket 2 §3**: missing **ill-conditioned exact-path test**
  (`failure_category="scaling_limit"` OR operator residual `≤1e-10`, NO vector
  parity claim). **Severity**: P0. **Mitigation**: 10th plan step.
- **Bucket 2 §3**: missing **`inspect.signature` regression for
  `_boozer_exact_coil_vjp` and `_boozer_ls_coil_vjp`**. **Severity**: P0
  (matches synthesis P0 #25). **Not addressed**. **Mitigation**: small XS test,
  add to step 4 or 9.

### Bucket 3 (M5 single-stage)
- **Bucket 3 #4 / #5 / #11**: `test_adjoint_fraction_diagnostic` (line 5859),
  `test_device_native_adjoint_solve_satisfies_runtime_operator` (line 2502),
  `test_vjp_produces_finite_derivative` (line 2544) — all weak. **Severity**:
  P0/P1. **Not addressed**. **Mitigation**: fold into step 3.
- **Bucket 3 §3**: missing **End-to-end IFT-vs-FD on real reduced fixture for
  IotasJAX at multiple `iota_target_shift` values** (no re-solve, just shift
  target). **Severity**: P0. **Not addressed**. **Mitigation**: 10th plan step.
- **Bucket 3 §3**: missing **Successful forward + failed adjoint → non-finite
  gradient on legacy `BoozerResidualJAX.dJ()` / `IotasJAX.dJ()` /
  `NonQuasiSymmetricRatioJAX.dJ()`**. **Severity**: P0. **Not addressed**.
  **Mitigation**: 10th plan step.
- **Bucket 3 §3**: missing **CPU-vs-JAX projected coil-derivative parity at
  LS-warmed fixture for IotasJAX/NonQS** at strict tolerance. **Severity**:
  P0. **Not addressed**. **Mitigation**: 10th plan step.
- **Bucket 3 §3**: missing **GPU same-state determinism gate** (run twice,
  `np.testing.assert_array_equal`). **Severity**: P0. **Not addressed**.
  **Mitigation**: add to step 6 (GPU parity) or 10th plan step.

### Bucket 4 (Stage 2 / flux)
- **Bucket 4 #4** (= synthesis P0 #16): `np.isfinite`-only assertion on
  zero-current singular case. **Severity**: P0. **Not addressed**.
  **Mitigation**: fold into step 5 or 6.
- **Bucket 4 #5 / #6 / #7**: basin-stability and trajectory-parity tests
  calibrated at 1e-2 / 1e-3 / 1e-6 with no per-iteration check. **Severity**:
  P1. **Mitigation**: backlog.
- **Bucket 4 #17**: `test_j_changes_after_dof_mutation` is inequality-only;
  audit recommends FD-vs-grad consistency check. **Severity**: P1.
  **Mitigation**: backlog.
- **Bucket 4 #24 / #25**: `test_b_vjp_includes_curvecwsfouriercpp_surface_derivative`
  + `test_b_vjp_includes_curvecwsfourier_surface_derivative` weak (only
  `isfinite + norm > 0`). **Severity**: P1. **Mitigation**: backlog.
- **Bucket 4 §3**: missing **`SquaredFluxJAX.dJ()` Taylor test** (matches
  synthesis P0 #21). **Severity**: P0. **Not addressed**. **Mitigation**: 10th
  plan step.
- **Bucket 4 §3**: missing **chunked-VJP gradient parity on large point cloud**
  (matches synthesis P0 #21). **Severity**: P0. **Not addressed**.
  **Mitigation**: 10th plan step.
- **Bucket 4 §3**: missing **`set_points()` after construction** raises-vs-silent
  decision. **Severity**: P1. **Mitigation**: backlog (open question for user).

### Bucket 5 (backend / runtime / smoke)
- **Bucket 5 #7 / #8** (= synthesis P0 #18): `test_backend_state_guard_sequence_*`
  / `test_backend_module_guard_sequence_*` ordering anti-pattern. **Severity**:
  P0. **Not addressed**. **Mitigation**: 10th plan step.
- **Bucket 5 #9** (= synthesis P0 #19): `tests/integration/conftest.py:_patch_meta_path_finder`
  silent `return False`. **Severity**: P0. **Not addressed**. **Mitigation**:
  10th plan step (XS effort).
- **Bucket 5 #21** (= synthesis P0 #24): `_force_x64` autouse blinkers all
  tests in `test_run_code_benchmark_common.py`. **Severity**: P0. **Not
  addressed**. **Mitigation**: 10th plan step (XS).
- **Bucket 5 #23**: `_BACKEND_RUNTIME_ENV_VARS` SSOT drift across 3 files.
  **Severity**: P1. **Not addressed**. **Mitigation**: backlog.
- **Bucket 5 #28**: `tests/subprocess/jax_runtime_cases.py:36` module-level
  imports collapse all subprocess tests on one import error. **Severity**: P1.
  **Mitigation**: backlog or step 8 sub-task.
- **Bucket 5 #29**: ~14 `fake_run_python_script` monkeypatches without
  integration counterpart. **Severity**: P1. **Mitigation**: backlog.
- **Bucket 5 §3**: missing **`apply_jax_runtime_config()` against real JAX
  module**. **Severity**: P1. **Mitigation**: backlog.

### Bucket 6 (GPU proof)
- **Bucket 6 §3**: missing **device residency of hot-loop arrays**. **Severity**:
  P0. Step 2 mentions "devices" in payload but not residency assertions.
  **Mitigation**: add to step 2 (sub-task: `assert_device_residency` helper).
- **Bucket 6 §3**: missing **forward AND backward executed on device** with
  `exe.runtime_executable.execution_count_by_device` instrumentation. **Severity**:
  P0. **Not addressed**. **Mitigation**: 10th plan step or fold into step 2.
- **Bucket 6 §3**: missing **`jax.transfer_guard("disallow")` audit log**.
  **Severity**: P0. **Not addressed**. **Mitigation**: fold into step 2.
- **Bucket 6 §3**: missing **`xla_flags_seen_at_jax_init`** captured *after*
  `import jax`. **Severity**: P0. Step 2 mentions XLA flags but not the
  *when-captured* contract. **Mitigation**: clarify step 2.
- **Bucket 6 §3**: missing **CUDA wheel resolved to compatible cubin**
  recording in bundle (`jaxlib.cuda_versions`). **Severity**: P0. Step 2
  mentions `jaxlib_cuda_versions` — partial. **Mitigation**: clarify step 2's
  "dynamic loader smoke" sub-task.
- **Bucket 6 §3**: missing **cold-warm GPU determinism**. **Severity**: P0.
  **Not addressed**. **Mitigation**: add to step 6 (GPU determinism) or 10th
  plan step.
- **Bucket 6 §3**: missing **bootstrap-time GPU smoke** (`bootstrap_runtime.sh`
  must assert `jax.default_backend() == "gpu"`). **Severity**: P0. **Not
  addressed**. **Mitigation**: 10th plan step or fold into step 2.

### Bucket 7 (core / curve / misc)
- **Bucket 7 #14** (= synthesis P0 #23 and bucket 8 P0 #2): `test_field_cache_hot_path_benchmark.py`
  TAUTOLOGICAL printf-format pinning. **Severity**: P0. **Not addressed**.
  **Mitigation**: 10th plan step (S effort).
- **Bucket 7 §2 #6-#10**: `test_jax_core_specs.py` 5 weak/tautological tests
  (Literal vs discriminator, factory validation gaps). **Severity**: P1.
  **Mitigation**: backlog.
- **Bucket 7 §3 (Reductions)**: missing **`compensated_sum_flat` cancellation
  case** (`[1e16, 1.0, -1e16]`). **Severity**: P1. **Mitigation**: add to step
  6 (reduction physics).
- **Bucket 7 §3 (Specs)**: missing **`make_coil_symmetry_spec` rotmat
  dimensionality validation**. **Severity**: P1. **Mitigation**: backlog.
- **Bucket 7 §5 (Import-cycle)**: ZERO tests exercise the cold-state lazy-import
  seam in `simsopt.geo.curve` ↔ `simsopt.jax_core`. **Severity**: P1.
  **Mitigation**: backlog (separate `tests/test_import_seam.py` file).

### Bucket 8 (modified upstream)
- **Bucket 8 #2** (= synthesis P0 #11 sub-issue): `test_objectives_time` env-gated
  → default CI now skips. **Severity**: P0. Step 7 covers force-objectives
  Taylor but not the timing test. **Mitigation**: add to step 7.
- **Bucket 8 #6** (= synthesis P0 #5 in spirit): `_FluxObjectiveFakeField`
  zero-against-zero test. **Severity**: P0/P1. **Not directly in plan**.
  **Mitigation**: add to step 5 (or migrate to step 4 mock-discipline scope).
- **Bucket 8 #8** (= synthesis P0 #11 cousin): `test_call` follow-up dropped.
  **Severity**: P0. **Not in step 7**. **Mitigation**: add to step 7
  explicitly.
- **Bucket 8 #9 / #10**: `test_call_boozer_residual_falls_back_to_alpha_only_signature`
  and ds/ds2 variants only mock the C++ residual. **Severity**: P1.
  **Mitigation**: backlog.
- **Bucket 8 #11**: `test_quadratic_penalty_hostifies_jax_scalar_objective`
  call-count monkeypatch (synthesis T7 backlog). **Severity**: P1.
  **Mitigation**: backlog.

### Cross-cutting
- **SSOT drift on `_BACKEND_RUNTIME_ENV_VARS`** (bucket 5 #23 / #22 /
  synthesis SSOT-drift section). **Severity**: P1. **Mitigation**: backlog.
- **No `pytest -p random` (pytest-randomly) gate** to catch order-dependent
  tests (synthesis §5 CI changes). **Severity**: P0. **Not addressed**.
  **Mitigation**: 10th plan step (CI infrastructure).
- **No `pytest --collect-only` guard against re-introduced deleted tests**
  (synthesis §5 CI changes). **Severity**: P1. **Mitigation**: backlog.
- **No `pytest.skip` audit** with tracked GitHub issue ID requirement
  (synthesis §5 CI changes). **Severity**: P1. **Mitigation**: backlog.

---

## 4. New risks introduced by the plan

These are risks the plan *creates* by virtue of how it is written, independent
of the audit findings.

1. **Step 1's "device-aware skip helper" can mask GPU regressions.** If the
   skip predicate is `pytest.mark.skipif(jax.default_backend() != "gpu", ...)`,
   a CI runner without a GPU silently passes the test. The audit explicitly
   warns (bucket 6 §6) that GPU lanes must use `xfail strict=True`, not skip.
   Plan does not encode this in step 1.

2. **Step 3's "deterministic direction selection avoiding near-zero" risks
   cherry-picking.** Replacing 5 random directions with a deterministic
   "non-near-zero" selection can systematically exclude the regime where
   wrong-sign IFT terms surface. The audit's intent is to reject directions
   where the *projected gradient is below the FD floor*, not to avoid
   "near-zero" in some absolute sense. If the implementor chooses directions
   that yield large gradient magnitude, the test passes for the wrong reason.

3. **Step 5's "downgrade JAX-vs-JAX to alias tests"** (instead of "drop and
   replace with C++/FD oracle") leaves alias tests that look like parity tests
   in code review. A maintainer six months from now sees `test_assert_jax_jax_alias`
   and assumes it covers the contract.

4. **Step 4's "keep mock for plumbing only or rename"** is unspecific. If the
   rename happens but the existing tests are not migrated, the new physics
   tests are appended and the old tautological tests stay. Test count goes
   up; signal does not.

5. **Step 8's "mutant `donate_argnums=()` must fail"** without a mutation
   testing infrastructure becomes a one-shot manual exercise. Without
   continuous enforcement, a future refactor that disables donation reverts
   the contract silently.

6. **Step 9's mutant validation list is incomplete.** Only 5 mutants are
   listed. Steps 4, 6, 7 do not have corresponding mutants. A mutant test for
   `_real_label.J() returns 0.0` would close the loop on step 4; without it,
   step 4 has no objective acceptance criterion.

7. **Step 7's "reintroduce `np.random.default_rng(seed)`"** changes the
   seeding API from upstream's `np.random.seed(0)`. Any other test in the
   same module that depends on global RNG state changes behaviour. The plan
   does not enumerate which sibling tests this affects.

8. **Step 2's bundle schema extension without a `schema_version` bump** risks
   breaking downstream consumers (Hugging Face dashboard / Runpod sign-off
   pipeline per synthesis §7 Q1). Plan adds new fields but does not specify
   whether the consumer protocol must also be updated.

---

## 5. Suggested plan amendments

These slot into the existing 9-step structure. Numbering preserves the
original 1-9; new numbers 10-13 add what's missing.

### Amend step 1
- Specify helper API: accept `lane=` keyword that reads tolerance from
  `PARITY_LADDER_TOLERANCES`. Reject ad-hoc tolerance overrides.
- "Device-aware skip" is `xfail strict=True` for GPU lanes, NEVER a soft
  skip. Predicate is `pytest.importorskip("jax")` + `jax.default_backend() == "gpu"`.

### Amend step 2
- Add aggregator rejection logic: bundle MUST fail when any payload has
  `bundle_provenance.fake == True` without `SIMSOPT_FAKE_GPU=1`, or any payload
  `value_rtol`/`gradient_rtol` exceeds the lane contract, or any payload's
  `default_backend != "gpu"` on the real-GPU lane.
- Add `bootstrap_runtime.sh` strengthening: assert `jax.default_backend() == "gpu"`
  immediately after `import jax`; emit `bootstrap_jax_smoke.json`.
- Add LD_LIBRARY_PATH dynamic-loader smoke: real `import jaxlib.cuda_versions`
  invocation under the real lane.
- Add device-residency assertion + `jax.transfer_guard("disallow")` audit log
  to the bundle.
- Bump `schema_version` so downstream consumers can opt-in.

### Amend step 3
- Add line **1979** explicitly (`_REAL_RESOLVE_FD_TAYLOR_RATE = 0.55` →
  `0.4`) and the constant `_REAL_RESOLVE_FD_MIN_STABLE_SAMPLES = 3` →
  `len(direction_samples)`.
- Specify direction selection: uniformly random with RNG-seeded reproducibility,
  reject only directions where projected gradient magnitude is below the FD
  floor (`1e-12`), require at least 80% of sampled directions to survive.
- Add `assert residual_rel <= 1e-10` to `test_gradient_wrappers_operator_status_on_exact_state`
  IotasJAX branch (covers P0 #6).

### Amend step 4
- Rename `_MockVolumeLabel` → `_PlumbingVolumeLabel`; document in CLAUDE.md.
- Enumerate 7-12 highest-leverage tests to migrate to real `Volume(surface)`
  labels (`test_penalty_with_volume_label`, `test_run_code_ls_converges`,
  `test_penalty_gradient_fd`, `test_newton_polish_reduces_gradient`,
  `test_penalty_with_toroidal_flux`, `test_penalty_with_area_label`).
- Use `derivative_heavy` lane (rtol=1e-8) for new label-perturbation FD
  tests.

### Amend step 5
- Split into 5a (bucket 1 surface_rzfourier tautologies), 5b (bucket 7
  accessibility), 5c (bucket 4 flux kernel tautologies). Enumerate
  per-cluster oracle (C++ for 5a, FD for 5b, CPU `SquaredFlux` for 5c).
- Move HLO-text gates to `tests/perf_gates/` with `@pytest.mark.brittle_perf_gate`,
  do NOT delete.
- Add bucket 1 P0 #17 (cross-product orthogonality + analytic torus area/volume).

### Amend step 6
- GPU markers MUST be `xfail strict=True` for parity lanes, not `skipif`.
- Add nfp rotational-symmetry B-field test (bucket 4 §3, bucket 1 §3).
- Specify Gauss-law test uses coil ring at `z=5` outside `R=1, r=0.1` torus.
- Add Kahan/cancellation oracle test for `compensated_sum_flat` (bucket 7
  §3 reductions): `[1e16, 1.0, -1e16]`.
- Add GPU same-state determinism gate (run twice, `np.testing.assert_array_equal`).

### Amend step 7
- Add `test_objectives_time` un-skip (`ncoils=2` row default-on; `ncoils=8`
  row gated on `SIMSOPT_RUN_FIELD_TIMING=1`).
- Add `test_call` follow-up restoration (`tests/core/test_optimizable.py:243-258`).
- Name the 3 deleted tests verbatim: `test_arclength_variation_circle_planar`,
  `test_linking_number_planar`, `test_curve_curve_distance_empty_candidates`.
- Specify `default_rng(0)` (same seed value) to minimize tolerance drift from
  upstream `np.random.seed(0)`.

### Amend step 8
- Specify donation mutant test mechanism: pytest test that monkey-patches
  `donate_argnums = ()` and asserts the existing donation probe fails.
- Specify wrapper schema validation: `payload["case"] == expected_case_name`
  AND `payload["invariant"] == "compile_count" / "device_residency" / ...`.
- Add `tests/subprocess/jax_runtime_cases.py:36` import-collapse fix as
  sub-task.

### Amend step 9
- Add three more mutants:
  - `_real_label.J()` returns 0.0 → step 4 must fail.
  - GPU reduction summands reordered → step 6 GPU parity must fail.
  - `use_jax_curve=True` branch disabled → step 7 must fail.
- Specify mutant infrastructure: `tests/mutants/test_required_mutants.py`
  with `@pytest.mark.mutation`, run nightly.

### NEW — step 10: Boozer exact-Newton + adjoint contract closure

Address P0 #7, #10, bucket 2 #16, bucket 2 §3 (ill-conditioned exact lane).

- Add ONE end-to-end exact-Newton test on a real torus (no
  `_patched_exact_newton_result`): assert `‖res["residual"]‖ < 1e-10`,
  `‖J^T r‖ < 1e-10`, `res["jacobian_materialized"] is True`.
- Add ill-conditioned exact-path test: low-iota or near-axis configuration,
  assert `failure_category="scaling_limit"` OR operator residual `≤1e-10`,
  NO vector-parity claim.
- Add `inspect.signature` regression for `_boozer_exact_coil_vjp` and
  `_boozer_ls_coil_vjp` (CLAUDE.md M4 signature `(lm, booz_surf, iota, G[, weight_inv_modB])`).
- Replace toy 3×3 oracle at `test_surface_objectives_jax.py:1728` with real
  `BoozerSurfaceJAX(boozer_type='exact')` fixture + `scipy.linalg.lu_solve`.

### NEW — step 11: Legacy public dJ() failure + IotasJAX/NQSR coil-derivative parity

Address bucket 3 §3 missing coverage (3 sub-items).

- Add real-fixture FAILURE-of-adjoint test for legacy `BoozerResidualJAX.dJ()` /
  `IotasJAX.dJ()` / `NonQuasiSymmetricRatioJAX.dJ()`: monkeypatch
  `solve_transpose_with_status` to return `(garbage, False)`, assert
  `not np.all(np.isfinite(grad))`.
- Add CPU-vs-JAX projected coil-derivative parity at LS-warmed fixture for
  IotasJAX and NonQuasiSymmetricRatioJAX (`rtol=1e-6, atol=1e-9`).
- Add per-axis FD on `NonQuasiSymmetricRatioJAX.dJ()` on the controlled LS
  fixture (`rtol=1e-4` per direction).

### NEW — step 12: SquaredFluxJAX Taylor + chunked-VJP gradient parity

Address P0 #21.

- Add `test_squaredfluxjax_dJ_taylor_test` mirroring upstream
  `check_taylor_test` across all three definitions.
- Add chunked-VJP large-point-cloud parity test (companion to
  `test_chunked_grouped_paths_match_cpu_on_large_point_cloud` which only
  checks B).

### NEW — step 13: Conftest hardening + smoke restitution

Address P0 #18, #19, #23, #24 + bucket 5 #28.

- `tests/integration/conftest.py:_patch_meta_path_finder`: replace silent
  `return False` with `pytest.skip(...)` at module load.
- `tests/test_backend.py:1772-1830`: convert
  `test_backend_state_guard_sequence_*` and `test_backend_module_guard_sequence_*`
  to single-test mutation/restore lifecycles; eliminate module-level
  `_backend_module_guard_reloaded` dict.
- `tests/test_run_code_benchmark_common.py:7-9`: add counter-test
  (`test_resolver_rejects_float32_runtime`) that drops the autouse override.
- `tests/test_field_cache_hot_path_benchmark.py:8-66`: replace 3 TAUTOLOGICAL
  tests with one end-to-end compile+JSON-parse smoke.
- `tests/subprocess/jax_runtime_cases.py:36`: wrap module-level
  `_prefer_local_simsopt_source_tree()` and top-level imports in try/except so
  one import error does not collapse ~50 subprocess tests.

### NEW — CI infrastructure (slot before step 9 or add to step 9)

- Add `pytest-randomly` to nightly CI to catch order-dependent tests
  (synthesis §5).
- Add `pytest --collect-only` guard that fails when a known-deleted upstream
  test re-appears under the same name with a different body.
- Add `pytest.skip` audit: any new `pytest.skip` without a tracked GitHub
  issue ID fails CI.

---

DONE — plan-vs-findings at /Users/suhjungdae/code/columbia/simsopt-jax/.artifacts/jax-test-audit-2026-04-25/PLAN_VS_FINDINGS.md

Counts:
- P0 fully addressed: 6 / 25 (#1, #4, #11, #12, #15, #20)
- P0 partially addressed: 6 / 25 (#3, #5, #13, #14, #17, #22)
- P0 not addressed: 12 / 25 (#6, #7, #8, #9, #10, #16, #18, #19, #21, #23, #24, #25)
- P0 explicitly retired: 1 / 25 (#2)

Top-5 missing-from-plan items:
1. Real end-to-end exact-Newton fixture (P0 #10) + ill-conditioned exact lane
   (bucket 2 §3) — entire `TestBoozerSurfaceJAXExactPath` cluster validates
   plumbing only.
2. `inspect.signature` regression for `_boozer_ls_coil_vjp` /
   `_boozer_exact_coil_vjp` (P0 #25) — no automated guard against the
   historical CPU-vs-JAX signature drift.
3. `test_squaredfluxjax_dJ_taylor_test` + chunked-VJP gradient parity on
   large point cloud (P0 #21).
4. Order-dependent `test_backend_*_sequence_*` refactor (P0 #18) +
   `tests/integration/conftest.py:_patch_meta_path_finder` silent-False fix
   (P0 #19).
5. Replace toy 3×3 exact-well-conditioned test at
   `test_surface_objectives_jax.py:1728` with real `BoozerSurfaceJAX`
   exact-path fixture (P0 #7).

Top-3 new risks introduced by the plan:
1. Step 1's "device-aware skip helper" without `xfail strict=True` masks
   GPU regressions on CPU-only CI.
2. Step 3's "deterministic direction selection avoiding near-zero" can
   cherry-pick out the regime where wrong-sign IFT terms surface.
3. Step 5's "downgrade JAX-vs-JAX to alias tests" leaves false-parity tests
   in place that look like contract coverage in code review.
