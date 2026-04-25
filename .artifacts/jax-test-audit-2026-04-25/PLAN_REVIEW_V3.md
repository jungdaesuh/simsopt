# Combined Plan Review (V3) — 2026-04-25

Branch: `gpu-purity-stage2-20260405` · HEAD `42b68f33d`
Subject: `IMPLEMENTATION_PLAN_COMBINED.md` (227 lines, 12 steps).

Inputs: `IMPLEMENTATION_PLAN_COMBINED.md` + `SYNTHESIS.md` + `PLAN_VS_FINDINGS.md` +
`PLAN_REVIEW.md` (V1) + `PLAN_REVIEW_V2.md` (V2) + `CORRECTED_PLAN_REVIEW.md` +
4 verification reports + `COMBINED_PLAN_COVERAGE.md` (V3).

---

## 0. Headline

The combined 12-step plan is a **net upgrade over V2** and the strongest plan
in this thread. Coverage of the 25 P0 audit findings jumps from 28% (V2) to
**68% fully addressed** — only **1 P0** (#6 IotasJAX adjoint residual rel-tol)
remains unaddressed. **Move forward**.

Trajectory:

| Plan | A (full) | B (partial) | C (unaddressed) | D (retired) | % full |
|---|:---:|:---:|:---:|:---:|:---:|
| V1 (first plan) | 6 | 6 | 12 | 1 | 24% |
| V2 (corrected) | 7 | 6 | 11 | 1 | 28% |
| **Combined 12-step** | **17** | **6** | **1** | **1** | **68%** |

The 10 newly-closed P0s vs V2: #7 (toy 3×3 oracle), #8 (adjoint_fraction
ceremony), #9 (outer_opt slack), #10 (exact-Newton end-to-end), #16
(np.isfinite → np.isposinf), #18 (order-dependent guards), #19 (conftest
silent-False), #21 (SquaredFluxJAX dJ Taylor + chunked-VJP), #23
(printf-format pinning), #24 (`_force_x64` autouse), plus #25 (VJP signature
regression — was C in V2, now A under Step 10).

---

## 1. Verification of the plan's load-bearing claims

All factual claims in "Validation Notes" + "Corrections to V2" + cited line
numbers across the source tree verified against actual code at HEAD:

| Claim | Verdict | Evidence |
|---|:---:|---|
| PLAN_REVIEW.md is 397 lines | ✅ TRUE | `wc -l` |
| PLAN_REVIEW_V2.md is 336 lines | ✅ TRUE | `wc -l` |
| CORRECTED_PLAN_REVIEW.md still recommends `jaxlib_cuda_versions` | ✅ TRUE | `CORRECTED_PLAN_REVIEW.md:162-168` says "Extend `build_provenance` to emit ... `jaxlib_cuda_versions = getattr(jaxlib, 'cuda_versions', None)`"; V2 dropped this; combined plan's drop is correct. |
| `build_provenance` doesn't yet emit xla_flags / cuda_force_ptx_jit / cuda_disable_ptx_jit | ✅ TRUE | `validation_ladder_common.py:468` defines `build_provenance`; current emitted fields are `repo_sha`, `jax`, `jaxlib`, `backend`, `devices`, `x64_enabled`, `peak_rss_mb`, optional `gpu_memory_mb`. |
| All FD-discipline line numbers (510, 1439, 1442, 3744, 4213, 5153, 5792, 5804) | ✅ TRUE | Direct grep on `tests/integration/test_single_stage_jax_cpu_reference.py` confirms each line content matches plan. |
| `test_stage2_jax.py:1036-1054` zero-current singular boundary | ✅ TRUE | Function `test_singular_zero_current_objectives_boundary_is_documented` at line 1036; assertions `not np.isfinite(cpu_j/jax_j)` at lines 1053-1054. |
| `test_surface_objectives_jax.py:1728` toy 3×3 oracle | ✅ TRUE | `def test_iotas_jax_exact_well_conditioned_gradient_matches_dense_projection` at line 1728. |
| `test_single_stage_jax_cpu_reference.py:5859-5917` adjoint_fraction diagnostic | ✅ TRUE | `def test_adjoint_fraction_diagnostic` at line 5859 with explicit comment "does NOT fail if the fraction is below 10%". |
| `test_single_stage_jax_cpu_reference.py:4925-4960` outer_opt strict decrease | ✅ TRUE | `def test_outer_opt_decreases_objective` at line 4925. |

Every line number cited in the plan is real; every existing-source claim
about what's missing is real. The plan reads accurately against current HEAD.

---

## 2. Strengths over V2

1. **Steps 10/11/12 close 10 previously-untouched P0s** — exactly the four
   thematic clusters V2 left unaddressed (Boozer exact-Newton + VJP signature,
   M5/Stage 2 wrapper failure paths, conftest/order/CI gates).
2. **Validation Notes are source-backed and self-correcting** — the plan
   explicitly bans the invented `jaxlib_cuda_versions` field per the JAX 0.9.2
   probe, adopts NVIDIA-documented `CUDA_FORCE_PTX_JIT`/`CUDA_DISABLE_PTX_JIT`
   knobs, and keeps the false MockVolumeLabel P0 retired with a cosmetic
   rename + dead-method removal.
3. **Step 2 sub-task ordering is correct**: extend `build_provenance` first,
   then add Stage 2 platform guard, then aggregator rejection, then real CUDA
   canary with `block_until_ready()` (which doubles as the dynamic-loader
   smoke), then `nvidia-smi` capture. This matches the V2 sharpening
   ("plumbing order matters").
4. **Implementation Rules** are explicit and enforceable (no tautology, no
   silent GPU pass, no ad-hoc tolerances, no defensive wrappers, preserve
   upstream CPU semantics) — these correspond to the cross-cutting themes
   T1-T12 in `SYNTHESIS.md` §2 and give code review a clear yes/no test.
5. **Acceptance Criteria** map back to specific evidence-producing steps
   (every GPU proof payload has real backend/device/provenance; every
   derivative test uses nonzero signed directions + lane tolerances + all-
   direction acceptance; every physics invariant is signed/conserved; every
   compile-shape test is labeled as such; restored upstream tests are
   present, deterministic, and fast/slow-split).
6. **Claims V2's percentages are bookkeeping** — and provides a concrete
   acceptance criterion instead. Correct posture; this stops percentage chase.

---

## 3. Remaining gaps

### P0 #6 IotasJAX adjoint residual rel-tol (only unaddressed P0)
`test_single_stage_jax_cpu_reference.py:5662` checks the IotasJAX adjoint
operator solve status but does NOT assert
`adjoint_residual_rel <= 1e-10` for the `exact_ill_conditioned_adjoint`
lane. Add to Step 10 or Step 11: a single-line assertion:
```python
assert iotas_status["adjoint_residual_rel"] <= 1e-10, (
    f"IotasJAX adjoint residual rel exceeds ill-conditioned lane gate"
)
```

### P0 #15 mild regression A → B
V2's Step 8 explicitly said "subprocess emit JSON sentinel with case/checked/
invariant fields" — covering ~75 wrappers in `test_jax_import_smoke.py`
including the compile-count case. Combined Step 12 narrows JSON-parse scope
to the hot-path benchmark only. Restore the broader subprocess JSON-sentinel
migration: add to Step 12 a sub-bullet "migrate ~14 highest-leverage
`test_jax_import_smoke.py` wrappers to JSON sentinel schema (case +
invariant fields)."

### Step-level under-enumeration (B-status P0s that could move to A)
- **Step 8 (P0 #5, #13, #17, #22)**: bucket 4 flux-kernel tautologies at
  `test_fluxobjective_jax_parity.py:{211, 223, 253}`, the 5 specific
  `*_reuses_shared_jit_kernels` accessibility tests, the cross-product
  tautology in `test_normal_orthogonality`, and the analytic torus
  area/volume tests are NOT enumerated. Implementor will likely under-cover.
- **Step 14 (P0 #14)**: aggregator rejection criteria are listed but the
  payload schema fields (`cpu_oracle_value`, `gpu_value`, `value_rtol`,
  `gradient_rtol`) and `bundle_provenance.fake` discriminator are not
  enumerated; `bootstrap_runtime.sh` `jax.default_backend() == "gpu"`
  assertion missing.

### P1/P2 items the plan still misses (9 of 14 spot-checked)
- Vector potential A gauge consistency (bucket 1 §3)
- Stellsym round-trip on `surface_xyzfourier` (bucket 1 §3)
- Off-axis 20-pt `test_div_B_zero` (bucket 1 #6)
- nfp rotational symmetry of B (bucket 1/4 §3) — would have caught the
  historical Y/Z stellsym DOF bug
- GPU same-state determinism (run twice, bit-identical) (bucket 3/6 §3)
- `bootstrap_runtime.sh` hardening (bucket 6 §3) — Step 2 covers other
  bootstrap concerns but not the `jax.default_backend()` assertion at import
- Per-array device residency assertion for hot-loop arrays (bucket 6 §3)
- Cold-warm GPU determinism (bucket 6 §3)
- `test_newton_polish_reduces_gradient` `‖grad‖ < 1e-10` magnitude target
  (bucket 2 #6)
- Ill-conditioned exact path test with `failure_category="scaling_limit"`
  (bucket 2 §3)

These are all P1/P2 by audit severity; backlog is acceptable.

---

## 4. Step-level execution risks

### Step 4 — preselection cherry-pick risk
Plan says "preselected nonzero directions" but doesn't specify the
selection protocol. Implementor risk: pick directions where the gradient
magnitude is large, missing the regime where wrong-sign IFT terms surface.
Concrete protocol from V2:
- Uniformly random with RNG-seeded reproducibility,
- Reject only directions where projected gradient magnitude is below `1e-12`,
- Cap rejection rate at 20% (if more rejected, fail "fixture geometry is
  degenerate").

### Step 5 — ambiguous OR
"Rename the helper to `_PlumbingVolumeLabel` OR delete its unused `J()`
method" reads as either-or. It should be both: rename for clarity AND delete
the dead method to prevent re-flagging. Also the 7-12 highest-leverage tests
to migrate to real `Volume(surface)` are not enumerated.

### Step 8 — alias-vs-drop blindspot
"Keep JAX-vs-JAX checks only as API consistency" risks leaving the
`rtol=1e-12` JAX-vs-JAX assertions in place under a renamed helper that
looks like contract coverage in code review. Audit explicitly said "DROP the
`rtol=1e-12` JAX-vs-JAX arms; KEEP only the C++ oracle arm." Make this
explicit: drop the assertion, do not just rename the wrapper.

### Step 10 — well-conditioned only
"Real exact-state adjoint solve backed by the production operator and a
dense reference such as `scipy.linalg.lu_solve`" covers the
well-conditioned lane. Add a sibling sub-task: ill-conditioned exact path
that asserts `failure_category="scaling_limit"` OR operator residual
`≤ 1e-10`, with NO vector parity claim.

### Step 12 — CI gate enforcement mechanism
"Add CI gates for `pytest --collect-only`, randomized order, skip/xfail
audit, and the hot-path benchmark replacement" doesn't say where (which
.github/workflows file) or what predicate. Implementor risk: gates added
locally, missing from CI. Specify:
- Add to `.github/workflows/jax_smoke.yml` a job `pytest -p random tests/`
- Add a job `pytest --collect-only tests/geo/test_curve_objectives.py | grep
  -c "test_arclength_variation_circle_planar\|test_linking_number_planar\|
  test_curve_curve_distance_empty_candidates"` expecting `>= 3`
- Skip/xfail audit: `grep -rE "@pytest\.mark\.(skip|skipif|xfail)" tests/ |
  grep -v "issue=GH-" | wc -l` expecting `== 0`

---

## 5. Plan rule compliance

| Rule | Step(s) enforcing | Mechanism | Concrete or aspirational |
|---|---|---|:---:|
| No tautology tests | 8 (HLO/JAX-vs-JAX as API only), 9 (mutant) | Step 8 reclassification + Step 9 mutant "JAX-vs-JAX-only oracle must fail" | **Concrete** (mutant gate) |
| No silent GPU pass | 1 (xfail strict=True), 2 (require_requested_platform_runtime), 7 (GPU parity not CPU-only-pass), 9 (mutant) | xfail strict + runtime guard + Step 9 "CPU fallback under --platform cuda must fail" mutant | **Concrete** (mutant + runtime guard) |
| No ad-hoc tolerance constants in new helpers | 1 (`require_parity_lane(lane=...)`) | Lane SSOT through `PARITY_LADDER_TOLERANCES` | **Concrete** (helper API) |
| No broad defensive wrappers / fallback execution lanes | 11 (legacy dJ adjoint failure must yield non-finite, not a fallback finite gradient) | Step 11 explicit + Step 9 wrong-sign mutant | **Concrete** |
| Preserve upstream CPU/reference behavior | 9 (restore upstream sweep without rewriting CPU) | Reintroduction of deleted tests + slow-marked broad sweep | **Concrete** (regression-by-collect-only CI gate would close the loop) |

All 5 rules are concrete, not aspirational. This is unusually strong for a
test-quality plan.

---

## 6. Acceptance Criteria — measurability

| Criterion | Measurable? | Producing step(s) | Concrete check |
|---|:---:|---|---|
| Every required GPU proof payload has real backend/device/provenance evidence and fails closed on CPU fallback | ✅ Yes | 2 (require_requested_platform_runtime + aggregator rejection + real CUDA canary) | grep aggregator output for `provenance.backend=="gpu"` AND `default_backend=="gpu"`; mutant test that removes guard must fail |
| Every derivative test that claims FD/IFT correctness uses nonzero signed directions, lane tolerances, and all-direction acceptance | ✅ Yes | 1, 4 | grep tests for `assert rel < 1e-3 or abs < 1e-8` patterns expecting 0 hits; helper API rejects ad-hoc tolerance |
| Every physics invariant test checks a signed or conserved quantity, not a squared objective that is nonnegative by construction | ✅ Yes | 6 (raw signed flux), 11 (zero-current `np.isposinf`) | code review: physics tests use `signed_flux_jax` not `integral_BdotN`; `test_singular_zero_current_objectives_boundary_is_documented` uses `np.isposinf` |
| Every compile-shape/HLO test is labeled as compile/performance coverage only | ✅ Yes | 8 (HLO as API/compile-shape only) | grep for `hlo_text` checks expecting `@pytest.mark.compile_only` or in `tests/perf_gates/` |
| Restored upstream tests are present, deterministic, and split into fast representative coverage plus slow broad sweeps | ✅ Yes | 9 | `pytest --collect-only` finds the 3 restored tests + `pytest -m slow` runs the broad sweep |

All 5 acceptance criteria are measurable with concrete checks. CI gates from
Step 12 close the loop.

---

## 7. Suggested final amendments

To convert the 6 remaining B-status P0s to A and close the 1 remaining C, add:

1. **Step 4 sub-bullet**: specify direction-selection protocol (RNG-seeded
   uniform + `1e-12` projection floor + 20% rejection cap).
2. **Step 5 sub-bullet**: rename to `_PlumbingVolumeLabel` AND delete dead
   `J()` method (not OR); enumerate the 7-12 tests to migrate to real
   `Volume(surface)`.
3. **Step 8 sub-bullets** (split into 8a/8b/8c):
   - 8a: drop `_assert_surface_jacobian_parity` / `_assert_area_volume_gradient_parity`
     JAX-vs-JAX arms (don't just rename); add FD oracle (`eps=1e-5,
     rtol=1e-7, atol=1e-9`); add analytic torus area/volume tests; drop
     `test_normal_orthogonality` cross-product tautology
   - 8b: enumerate 5 `*_reuses_shared_jit_kernels` accessibility tests; for
     each, ADD `J/dJ` FD parity at `h=1e-6, rtol=1e-6` (lane
     `derivative_heavy`)
   - 8c: replace `test_fluxobjective_jax_parity.py:{211, 223, 253}` with
     CPU `SquaredFlux` parity OR analytic-zero algebra
4. **Step 10 sub-bullet**: add ill-conditioned exact-path sibling test
   (`failure_category="scaling_limit"` OR residual `≤ 1e-10`, NO vector
   parity claim).
5. **Step 11 sub-bullet** (closes P0 #6): add IotasJAX adjoint residual
   rel-tol gate at `test_single_stage_jax_cpu_reference.py:5662`.
6. **Step 12 sub-bullets**:
   - Migrate ~14 highest-leverage `test_jax_import_smoke.py` wrappers to
     JSON sentinel (don't lose V2's broader scope; restores P0 #15 to A)
   - Specify CI workflow file (`.github/workflows/jax_smoke.yml`) and
     concrete gate predicates (collect-only count, skip-without-issue grep)
   - Add `bootstrap_runtime.sh` `assert jax.default_backend() == 'gpu'`
     check + `bootstrap_jax_smoke.json` artifact
7. **Step 2.3 sub-bullet**: enumerate aggregator payload schema fields
   required for parity-tolerance rejection (`cpu_oracle_value`, `gpu_value`,
   `value_rtol`, `gradient_rtol`, `bundle_provenance.fake`).

With these 7 amendments, coverage moves from 17A/6B/1C to **22A/2B/0C** = 88%
fully addressed. The remaining 2 B's (P0 #14 partial-by-design,
P0 #17 stellsym/torus-analytic backlog) are acceptable.

---

## 8. Final assessment

The combined 12-step plan is **production-ready** with the 7 sub-bullet
amendments in §7. Without them, it still closes 17 of 25 P0s (68%) — a major
improvement over V1 (24%) and V2 (28%).

The plan correctly:
- Retires the false MockVolumeLabel P0
- Drops the invented `jaxlib_cuda_versions` field per direct probe
- Substitutes raw signed flux for `integral_BdotN` (which squares everywhere)
- Uses positional donation contract per JAX docs
- Adds NVIDIA-documented PTX/CUBIN knobs (not just `CUDA_VISIBLE_DEVICES`)
- Adds a real CUDA compile/run rung (would have caught the 2026-04-20 Runpod
  cubin incident)
- Gives concrete enforcement for all 5 implementation rules
- Maps acceptance criteria to evidence-producing steps

The plan does NOT:
- Specify direction-selection protocol (Step 4)
- Enumerate which tests to migrate from `_MockVolumeLabel` (Step 5)
- Enumerate the bucket 4 flux-kernel tautologies (Step 8)
- Add the IotasJAX adjoint residual rel-tol gate (P0 #6)
- Specify CI workflow file or gate predicates (Step 12)
- Restore V2's broader subprocess JSON-sentinel scope (Step 12)

These omissions are addressable as 7 sub-bullets in §7. None require new
steps; all are within the existing 12-step structure.

**Verdict: ship it.** Combined plan is the right artifact to drive
implementation. Apply §7 amendments before kicking off the work.
