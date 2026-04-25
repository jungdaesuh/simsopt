# JAX Test Quality Implementation Plan - Combined Current Version

Date: 2026-04-25

This plan combines the strongest parts of the original implementation plan,
`PLAN_REVIEW.md`, `CORRECTED_PLAN_REVIEW.md`, and `PLAN_REVIEW_V2.md`, after
rechecking the current tree and the relevant official docs.

## Validation Notes

Source-backed items retained from V2:

- JAX donation tests must validate the positional donated-input contract:
  after a `donate_argnums=(0,)` call, the caller-owned input must be deleted or
  invalid for reuse. Do not assert output aliasing as the primary contract.
  Official source: https://docs.jax.dev/en/latest/buffer_donation.html
- `jax.default_backend()` and `jax.devices()` are the stable JAX runtime facts
  to record for backend/device proof. Official sources:
  https://docs.jax.dev/en/latest/_autosummary/jax.default_backend.html and
  https://docs.jax.dev/en/latest/_autosummary/jax.devices.html
- Local probe on JAX/JAXLIB 0.9.2 shows no stable `jaxlib.cuda_versions`
  attribute. Do not require a `jaxlib_cuda_versions` field in the proof bundle.
- CUDA PTX/CUBIN validation should use NVIDIA's documented
  `CUDA_FORCE_PTX_JIT` and `CUDA_DISABLE_PTX_JIT` knobs. `CUDA_VISIBLE_DEVICES`
  only constrains device enumeration. Official source:
  https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html
- `integral_BdotN` is a squared objective, not a signed flux integral:
  `src/simsopt/objectives/integral_bdotn_jax.py:50-78` and
  `src/simsoptpp/integral_BdotN.cpp:93-103` square `B dot n`.
- `Volume.J()` is the SIMSOPT volume label; the false `_MockVolumeLabel.J()`
  P0 remains retired. Official source:
  https://simsopt.readthedocs.io/stable/simsopt.geo.html

Corrections to the pasted V2 comparison:

- Current artifact line counts are stale in the pasted text:
  `PLAN_REVIEW.md` is 397 lines and `PLAN_REVIEW_V2.md` is 336 lines.
- `CORRECTED_PLAN_REVIEW.md` still says to emit `jaxlib_cuda_versions`; this
  contradicts V2 and the local JAX/JAXLIB 0.9.2 probe. Drop that field.
- V2's Step 10 table mixes Boozer, conftest, order, and benchmark work under
  one label, then later splits Step 10/11/12. This plan uses one clear 12-step
  structure.
- Treat V2's P0 percentages as report bookkeeping, not an implementation
  acceptance criterion. Close concrete tests and contracts instead.

## Implementation Rules

- No tautology tests: no JAX-vs-JAX correctness oracle, no HLO text as physics
  proof, no cache-size-only pass condition for derivative behavior.
- No silent GPU pass: required GPU proof must fail closed when the requested
  runtime is not actually GPU.
- No ad-hoc tolerance constants in new helpers. All parity tolerances come from
  `PARITY_LADDER_TOLERANCES` through a `lane=` contract.
- No broad defensive wrappers or fallback execution lanes. The tests should
  exercise the intended production path directly.
- Preserve upstream CPU/reference behavior. CPU tests are the oracle lane; JAX
  and CUDA tests prove parity with that lane.

## Step 1 - Shared Test Contracts

Create or tighten the smallest shared helpers needed by the changed tests:

- `assert_all_directions_match_fd(...)`: all preselected nonzero signed
  directions must pass. No OR escape between weak relative and weak absolute
  tolerances for nonzero directions.
- `assert_taylor_rate(...)`: central finite-difference error must decrease at
  the expected rate; near-zero derivative cases must be separate tests with
  explicit zero-derivative expectations.
- `require_parity_lane(lane=...)`: resolve tolerances only through
  `PARITY_LADDER_TOLERANCES`.
- GPU-required tests use `xfail(strict=True)` only for local hardware absence;
  production proof scripts must fail closed.

## Step 2 - GPU Proof And Provenance

Patch plumbing in this order:

1. Extend `build_provenance` in `benchmarks/validation_ladder_common.py` before
   touching aggregators. It currently records JAX/JAXLIB/backend/devices but
   not `xla_flags`, `cuda_force_ptx_jit`, or `cuda_disable_ptx_jit`.
2. Add `require_requested_platform_runtime(...)` to
   `benchmarks/stage2_e2e_comparison.py` immediately after JAX initializes,
   matching the existing `single_stage_init_parity.py` pattern.
3. Update `benchmarks/hf_jobs/run_production_gpu_proof.sh` so the summary
   preserves provenance fields and rejects invalid proof payloads:
   fake runner used without an explicit test-only fake mode, missing backend or
   device provenance, real GPU lane with `default_backend` not `gpu` or `cuda`,
   missing PTX/CUBIN env captures, or parity tolerance outside the ladder.
   The payload schema must make this check explicit: include CPU oracle value,
   GPU value, value rtol, gradient rtol, bundle provenance, and a fake-runner
   discriminator. Reject rtol values that exceed the lane contract.
4. Add a real CUDA canary that performs compile and execution with
   `block_until_ready()` under both PTX-forced and CUBIN-forced runs.
5. Record launcher-side driver/GPU facts with `nvidia-smi` output when present.
6. Add the same GPU backend assertion to `benchmarks/hf_jobs/bootstrap_runtime.sh`
   after `import jax`; workflow-local CUDA checks are not enough for the HF proof
   bootstrap path.

Do not add `jaxlib_cuda_versions` as a required field.

## Step 3 - Donation Probe

Replace output-alias assumptions with JAX's positional donation contract:

- Call the donated JIT function positionally, not with keyword arguments.
- Assert the donated input is deleted through `is_deleted()` or that reuse
  raises the documented invalid-buffer runtime error.
- Keep value equality checks as secondary correctness checks.

## Step 4 - FD And IFT Discipline

Fix the current escape hatches in
`tests/integration/test_single_stage_jax_cpu_reference.py`:

- Replace helper-level `rel_tol or abs_tol` at line 510 with strict nonzero
  signed-direction comparison.
- Replace fixed OR gates at lines 3744, 4213, 5153, and 5792.
- Change `_REAL_RESOLVE_FD_TAYLOR_RATE = 0.55` at line 1439 to the lane-backed
  central-FD threshold.
- Change `_REAL_RESOLVE_FD_MIN_STABLE_EPS = 2` at line 1442 to require all
  eps ladder points intended by the lane.
- Change `validated_directions >= 2` at line 5804 to require all three
  preselected nonzero directions.
- Direction selection is deterministic and auditable: draw from an RNG-seeded
  uniform direction set, reject only directions whose projected reference
  derivative is below `1e-12`, and fail the test if rejection exceeds 20%.

## Step 5 - Boozer Label And Plumbing Mock Cleanup

Keep the audit's `_MockVolumeLabel.J() == 0` P0 retired:

- Do not refactor production Boozer label logic for this false positive.
- Rename the helper to `_PlumbingVolumeLabel` and delete its unused `J()` method
  so future audits do not mistake it for physics coverage.
- Enumerate every physics-claiming test that reaches `_make_mock_boozer_surface`,
  `_make_mock_boozer_surface_exact`, `_make_mock_boozer_surface_mixed_quad`, or
  direct `_MockVolumeLabel` construction. Migrate those tests to real
  `Volume(surface)` coverage; keep plumbing-only tests on the plumbing helper.
- Add one real label-path test for volume/area/toroidal-flux label value and
  gradient using current geometry-derived JAX label code, not the plumbing mock.

## Step 6 - Raw Signed Flux Invariant

Add a raw signed closed-surface flux helper local to the test or to the
smallest existing JAX-core module:

- Compute the signed surface integral of `B dot n dA`, not squared
  `integral_BdotN`.
- Use a coil ring outside the torus or another physically clean source-free
  surface case.
- Assert the signed flux is near zero by the divergence theorem, with the
  tolerance from the lane.
- Keep `integral_BdotN` tests as squared-objective parity tests only.

## Step 7 - Reductions GPU/CPU Parity

Add device parity for `src/simsopt/jax_core/reductions.py`:

- Compare CPU and GPU reductions for pairwise and compensated modes.
- Use cancellation-stress arrays and fixed shapes.
- Require deterministic parity within the ladder tolerance.
- Do not treat a CPU-only run as passing the GPU parity claim.

## Step 8 - Surface And Accessibility Tests

Replace tautological assertions with physics/math assertions:

- 8a, surface geometry: keep JAX-vs-JAX checks only as API consistency.
  Correctness must compare against CPU/analytic derivatives or FD. Add analytic
  torus area and volume checks, and replace cross-product orthogonality
  tautologies with independent geometry assertions.
- 8b, accessibility: add J/dJ/ddJ parity or FD behavior to these cache-only
  tests:
  `test_projected_enclosed_area_reuses_shared_jit_kernels`,
  `test_directed_facing_port_reuses_shared_jit_kernels`,
  `test_curve_in_port_penalty_reuses_shared_jit_kernels`,
  `test_projected_curve_curve_distance_reuses_shared_jit_kernels`, and
  `test_projected_curve_convexity_reuses_shared_jit_kernels`.
- 8c, flux kernels: replace the JAX-vs-JAX-style kernel contracts at
  `tests/objectives/test_fluxobjective_jax_parity.py:211`,
  `tests/objectives/test_fluxobjective_jax_parity.py:223`, and
  `tests/objectives/test_fluxobjective_jax_parity.py:253` with independent
  NumPy/analytic expectations for value and derivative.
- HLO/StableHLO text checks may remain only as compile-shape/performance
  contract tests, not as numerical correctness tests.

## Step 9 - Restore Upstream Coverage

Restore the upstream coverage that was removed or narrowed:

- Reintroduce the three deleted upstream `test_curve_objectives.py` cases.
- Restore the broad `test_force_objectives_taylor_test` sweep as a slow-marked
  test, while keeping a representative fast default.
- Restore deterministic random seeding where upstream used it.

## Step 10 - Boozer Exact-Newton And VJP Signature

Close the exact-Newton plumbing-only gap:

- Add one end-to-end `BoozerSurfaceJAX(boozer_type="exact")` fixture on a real
  torus with no `_patched_exact_newton_result(jacobian=identity)` shim.
- Replace the toy 3x3 oracle at
  `tests/geo/test_surface_objectives_jax.py:1728` with a real exact-state
  adjoint solve backed by the production operator and a dense reference such as
  `scipy.linalg.lu_solve` where materialized.
- Add an ill-conditioned exact-path sibling test. It should assert the
  residual/failure contract only: either `failure_category == "scaling_limit"`
  or residual norm is within the lane gate, with no vector-parity requirement.
- Add `inspect.signature` regression guards for `_boozer_ls_coil_vjp` and
  `_boozer_exact_coil_vjp`, and one value-vs-FD guard so signature checks do
  not become shape-only tests.

## Step 11 - M5 And Stage 2 Failure Paths

Close remaining wrapper and singular-boundary checks:

- Add the legacy `dJ()` adjoint failure case where a finite wrong gradient
  would currently escape.
- Add the exact-path IotasJAX adjoint residual-rel gate currently missing near
  `tests/integration/test_single_stage_jax_cpu_reference.py:5662`. Compute the
  transpose residual directly from the exact `adjoint_state` and assert the
  lane's residual tolerance; do not depend on a new status-field side channel.
- Add a `SquaredFluxJAX.dJ()` Taylor test and a chunked/grouped-VJP gradient
  parity case on a large point cloud. Existing CPU parity is not enough.
- Change the zero-current singular boundary in
  `tests/integration/test_stage2_jax.py:1036-1054` from `not np.isfinite(...)`
  to `np.isposinf(...)`, matching the intended contract and the existing
  `np.isposinf(snapshot["J"])` style elsewhere.
- Promote or delete the `adjoint_fraction > 0` ceremony at
  `tests/integration/test_single_stage_jax_cpu_reference.py:5859-5917`.
- Tighten `test_outer_opt_decreases_objective` at lines 4925-4960 so it
  requires a real decrease and a nonzero optimization step, not `j_final <= j0
  + 1e-12`.

## Step 12 - Conftest, Order, Smoke, And CI Gates

Close the remaining infrastructure P0s:

- Refactor order-dependent backend guard sequence tests into isolated tests
  that can pass under randomized order.
- Make any conftest boolean guard failures explicit; no silent `False` path.
- Add the `_force_x64` counter-test so the autouse fixture cannot hide missing
  x64 setup.
- Replace printf-format pinning in `test_field_cache_hot_path_benchmark.py`
  with an end-to-end compile/run and structured JSON parse.
- Restore the broader subprocess JSON-sentinel scope from the earlier plan:
  migrate the highest-leverage `tests/test_jax_import_smoke.py` wrappers to a
  structured payload with case name, checked invariant fields, and measured
  counters instead of accepting only `rc == 0`.
- Add CI gates in `.github/workflows/jax_smoke.yml` for `pytest --collect-only`,
  `pytest-randomly` randomized order, skip/xfail audit, import-smoke JSON
  sentinel validation, and the hot-path benchmark replacement.

## Acceptance Criteria

- Every required GPU proof payload has real backend/device/provenance evidence
  and fails closed on CPU fallback.
- Every derivative test that claims FD/IFT correctness uses nonzero signed
  directions, lane tolerances, and all-direction acceptance.
- Every physics invariant test checks a signed or conserved quantity, not a
  squared objective that is nonnegative by construction.
- Every compile-shape/HLO test is labeled as compile/performance coverage only.
- The restored upstream tests are present, deterministic, and split into fast
  representative coverage plus slow broad sweeps.
