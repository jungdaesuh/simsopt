# JAX Upstream Final Upgrades Implementation Plan

**Status:** Completeness-audited and ready for implementation
**Last updated:** 2026-07-23

## Purpose

Define an executable, reviewable path for porting the final reusable FP64,
mixed-precision, optimizer, field-kernel, and correctness upgrades from the
`simopt-jax-clean-local` work into the clean upstream-PR branch
`pr/jax-port-squashed` without importing example applications, generated
evidence, or research-campaign infrastructure.

This plan treats source commits as provenance. It does not authorize merging
the source branch, replacing target files wholesale, or replaying every source
commit.

## Goals

- Preserve the current hardened FP64 behavior and make the dense-IR solver an
  explicit, tested capability.
- Add an explicit mixed-precision execution mode whose FP32 proposals are
  accepted only through FP64 certificate and fallback rules.
- Port the reusable online Biot-Savart, dense-HVP, surface-scatter, SciPy
  lifecycle, and derivative/replay corrections.
- Keep production modules and focused tests independent of
  `examples.single_stage_optimization`.
- Produce a commit series that can be reviewed by subsystem and validated on
  CPU and GPU without relying on historical campaign artifacts.

## Non-Goals

- Merging or cleaning `simopt-jax-clean-local` in place.
- Porting any file under `examples/`.
- Porting generated artifacts, `.Codex/` records, historical result payloads,
  or source-capture/remediation archives.
- Porting `genuine_675`, `canonical11`, fullspace-675, hardware-soft-penalty,
  attestation, validation-ladder, deadline-supervision, or campaign-publication
  machinery.
- Cherry-picking the 392-file remediation commit `5df801e1b` wholesale.
- Adding Diffrax or its tracing/Poincare integrations in this change. Those
  require a separate dependency and import-boundary review.
- Making dense-IR or mixed precision the implicit default for existing users.
- Porting persistent Boozer factor snapshots, HVP rematerialization environment
  controls, persistent compilation-cache threshold tuning, coil-spec
  memoization, surface-tangent finite-build support, general native/JAX curve
  cleanup beyond the narrow host-ownership/strict-transfer slice selected below,
  or broad objective batching in this series.

## Current Context

### Confirmed repository facts

- The implementation code baseline beneath the planning-only commits is
  `d2cdd7a8f`. Do not assume the branch tip still equals that commit: Phase 1
  must capture the actual clean `pr/jax-port-squashed` tip as `upgrade_base`
  immediately before implementation.
- The locally inspected upstream snapshot is `631cbe736`. On 2026-07-23,
  `git ls-remote upstream_check refs/heads/master` reported remote HEAD
  `51b22454f5a0faffbfc712b4a9a8d45738f12781`, so the local
  `upstream_check/master` reference is stale and must be refreshed before
  implementation.
- The stable source provenance anchor is
  `5fb96818885fe0f839f088f624914e411f2f25f8`.
- The current source branch tip and dirty source worktree are not safe merge or
  file-copy sources. In particular,
  `a79ddd4eaf6e46b2a7562b1c5379afd57221b7e1` is a retention merge whose
  integration ancestry includes a tree with 1,378 changed paths and 947,229
  deletions relative to the stable anchor; it is not a port source.
- A 2026-07-23 immutable-ref audit found no additional trustworthy production
  source. Of the 314 commits in `5fb968188..a79ddd4e`, 301 are synthetic
  reconstruction commits; the remainder are campaign/attestation changes,
  planning/merge records, or behavior already represented by the stable-anchor
  semantics selected below. The dirty source worktree is deliberately not
  completeness evidence: it is mutable, uncommitted, and excluded as a port
  source regardless of its current path counts.
- The target already contains the base JAX port and FP64 hardening, including
  patch-equivalent dense-solve, rejected-gradient, residual-J LSMR, and Boozer
  warm-start fixes.
- The target does not contain `src/simsopt_jax/numerical_policy.py`,
  `src/simsopt_jax/core/biotsavart_online.py`,
  `src/simsopt_jax/runtime/exact_numeric_identity.py`,
  `src/simsopt_jax/geo/optimizers/_evaluation_lifecycle.py`, or
  `src/simsopt_jax/geo/optimizers/_evaluation_provider.py`.
- The target does not expose the source branch's mixed-precision or
  `hybrid_final_dense_ir` implementation symbols.
- Source tests and benchmark harnesses contain direct imports from
  `examples.single_stage_optimization`; those consumers cannot be copied into
  an example-free upstream slice.

### Source provenance map

| Capability | Source provenance | Port rule |
| --- | --- | --- |
| FP64 dense-IR/factor-once Newton | selected portions of `1d055547e`, `139c05880`, `9bd9661b9`, `8d4a1103b`, `db6906fc9`, then `37b65c7af`, `ad3cc28b7`, `01baba0d5`, `4abc6982e`, `3a64837b2` | Port the complete non-default typed-mode and retry lineage, not the source's changed default. Include all four declared solver values or remove unsupported values from the public type. |
| Host/dense operator correctness | selected portions of `9ca8929f5`, `7d488caacc`, `ecdba5011`, `aa7612a05`, `5df801e1b`, and `e7b74254a` | Keep host dense-Hessian materialization independent, short-circuit exact-zero RHS solves, use bounded static CUDA chunk sizing, and keep Newton step damping out of returned accepted-state Hessians except for explicitly augmented residual-J LSMR operators. |
| Mixed dtype and solver foundation | `35eda35e3`, selected non-`lsmr_ir` portions of `e726e161f`, `37fc6be86`, `303ca6ed1`, selected `1f71046a7`, and `8120b0ede` | Reconcile hunks with target FP64 hardening; retain target defaults, preserve explicit FP32 placement, port replicated-scalar sharding, and package integration tests so mixed-directory collection loads both conftest owners. |
| Mixed correctness fixes | `93a3e0271`, `9c3b6523e`, `d5ea716e3`, `c9299557b`, `111d22758`, `18b39f0c3`, `5604c4263`, `c2450d7bb`, `dd6d7fcc7`, and selected `7d488caacc` | Port final semantics, including the transfer-safe fixed reduction tree, not intermediate states. |
| Pairwise and curve-objective SSOT | production portions of `8e3d2a784` | Port the core owner, geo compatibility re-export, core curve kernels, adapter routing, and focused parity tests as one slice; exclude project-specific owners and tests. |
| Typed policy and certificate/fallback closure | selected portions of `1f71046a7`, `5df801e1b`, `84bfcfc6b`, `0412de980`, `add41e95c`, `b35d9a9cc`, `c338f48ca`, `9e13791ca`, `e7b74254a`, `0391dd82d`, `aa47aa741` | Exclude campaign, attestation, and evidence-publication files; retain the strict-transfer host-boundary fix. |
| Online Biot-Savart | production portions of `2afc66397`, then `12f7eb254` | Port implementation and focused field tests only. |
| Dense HVP and surface placement | `b9732104b`, `7f6bf6192`, `5742d81b8` | Port as independent performance/correctness slices. |
| Adapter transfer/compile correctness | selected production portions of `2a134a677`, `7c934adc2`, `0412de980`, and `8120b0ede` | Port explicit quadrature boundaries, the exact lowerable production gradient, cached-report staging, and spec-backed ownership; exclude compile-evidence and campaign bundles. |
| SciPy evaluation ownership and factor routing | earlier routing invariant from `950fb5ca7`, reconciled with production portions of `e6746f04a`, `9183695f0`, `dbfb3238c`, then `f35f83515` | Port provider/lifecycle abstractions and explicit adjoint/forward factor authority without phase/campaign analyzers. |
| Native derivative, transfer, and replay correctness | `e4c008e80`, `df4b5b711`, the native curve host-ownership, derivative, and curve-surface ownership portions of `5df801e1b`, and selected portions of `5e3208281` | Port the explicit public-method allowlist, native curve strict-transfer boundary, and physical partials only; exclude broad batching, reduced-objective/certificate, and genuine-675 owners. |
| Deterministic paired statistics | `benchmarks/paired_bca.py` from `1f71046a7` plus example-independent contract assertions | Reuse the dependency-light BCa utility and deterministic balanced schedule; do not port the campaign comparator. |
| Test/bootstrap and public documentation companions | selected `1f71046a7`, `5df801e1b`, `ad73aa0f1`, `9e4b7c23f`, and the documentation-only `82e9b88d3` hunk | Package integration tests for collision-free mixed collection; port only fixtures required by selected final signatures; add the eager surface/native-curve strict-transfer regressions and public CurveSurfaceDistance/controller documentation. |
| QFM reuse and diagnostics | `0d4f82ddc` | Candidate follow-up after the core precision stack is stable. |

## Rationale

The source history contains validated numerical work, but it also contains
superseded implementations, explicit reversions, branch reconstructions, and
large evidence systems. A source-tree replacement would overwrite target-only
FP64 hardening, while raw cherry-picks do not apply cleanly to the target.

The implementation therefore proceeds by behavioral subsystem. Each phase
ports the final source semantics into the target abstractions, adds focused
observable tests, and closes compatibility gates before the next phase. FP64
and mixed precision remain one architecture, but the existing FP64 route stays
the compatibility default and mixed precision is selected explicitly.

## Assumptions

- `5fb96818885fe0f839f088f624914e411f2f25f8` remains available in the shared
  Git object database for provenance inspection.
- Upstream maintainers will review the explicit typed precision/solver policy
  selected below; changing that public spelling during review requires updating
  the caller inventory, migration example, compatibility tests, and rollback
  notes together.
- JAX CPU validation is available locally or in CI, while GPU performance and
  memory validation will run on an explicitly identified CUDA machine.
- No generated benchmark artifact is required to build, import, or test the
  production packages.
- QFM and Diffrax/Poincare work can remain outside the precision-port critical
  path.

## Resolved Design Decisions

- The upgrade range is based on the exact target commit captured before
  implementation, initially `d2cdd7a8f`. All containment and rollback checks
  compare against this `upgrade_base`, not against the older upstream merge
  base, because the target branch already contains the base JAX port.
- Precision selection is an optional typed keyword on the existing public
  runtime entry points: `set_backend(..., precision=None)` and
  `use_runtime(..., precision=None)`. `None` is an omission sentinel and is not
  a valid stored or environment value. Define
  `PrecisionSelection = Literal["mode_default", "fp64", "mixed"]` and
  `ResolvedPrecision = Literal["fp32_smoke", "fp64", "mixed"]`, add a defaulted
  `precision="mode_default"` field to `BackendConfig`, and expose the resolved
  precision, compute dtype, and optional FP64 certificate dtype through
  `BackendPolicy`. `mode_default` preserves every current mode: the existing
  `jax_cpu_float32_smoke` route remains full FP32 with no FP64 certificate,
  while the other current modes retain their existing FP64 behavior.
- Do not add or retain `SIMSOPT_MIXED_PRECISION`. The target has never exposed
  it. Add `SIMSOPT_PRECISION` only as the normalized subprocess/configuration
  transport for the same typed field: explicit `set_backend()`/`use_runtime()`
  non-`None` keyword, then `SIMSOPT_PRECISION`, then `"mode_default"`. Thus an
  explicit `precision="mode_default"` overrides an inherited environment value.
  The central runtime resolver owns this precedence, and `set_backend()` mirrors
  the normalized `PrecisionSelection` into the environment—not the derived
  `ResolvedPrecision`—so child processes preserve the parent selection. Reject `mixed` for
  `native_cpu` and `jax_cpu_float32_smoke`; reject explicit `fp64` for the smoke
  mode instead of silently changing that mode's contract.
- Design-it-twice result: the selected orthogonal precision field avoids
  multiplying every CPU/GPU fast/parity backend mode by FP64/mixed variants.
  Dedicated mixed backend modes were rejected because they duplicate mode
  defaults and amplify every future backend-policy change. A separate public
  `PrecisionPolicy` object was rejected because it would be a shallow second
  owner beside `BackendConfig`/`BackendPolicy`.
- Information-hiding test: `backend/runtime.py` alone owns precision parsing,
  explicit/environment/default precedence, subprocess serialization, and dtype
  resolution. Kernels and adapters consume resolved typed dtype helpers and do
  not read environment variables. Changing the public spelling therefore
  changes the runtime owner, its exports, compatibility tests, and docs—not
  every numerical caller.
- Mixed FP32 execution uses `matmul_precision="highest"` to prohibit TF32.
  `mode_default` and explicit FP64 retain the target mode's existing matmul
  setting, including `"default"` for smoke/fast modes where that is the current
  compatibility behavior.
- Dense-IR remains opt-in for this series. Any proposal to make it the default
  is a separate API change after upstream performance and compatibility review.
- The source's tiny-solution exception for unsafe condition estimates is not
  part of this series. Exact-zero RHS receives its own successful zero-solution
  path, while every nonzero solve retains the target's fail-closed condition
  policy.
- Newton stabilization owns iteration-direction regularization, not the
  accepted-state Hessian returned for adjoint/factor construction. Add one
  `adjoint_hessian_stabilization()` owner: it returns zero for dense and CG
  final/adjoint linearizations and retains the configured stabilization only for
  residual-J LSMR formulations whose augmented operator is
  `[J; sqrt(stab) I]`.
- The source's mixed residual-J `lsmr_ir` comparator is not part of this series.
  The current source selects it through import-time environment state, while
  this plan requires typed public boundaries and cache identity. Porting it
  requires a separate typed selector, adapter propagation, cache-key audit, and
  its focused solver tests; citing `e726e161f` elsewhere does not select this
  comparator.
- Preserve the target's existing native-policy GPU-helper behavior in this
  series. The `8120b0ede` short circuit that suppresses imported-JAX CUDA device
  and memory reporting under a non-CUDA policy changes observable runtime
  telemetry and is independent of the selected precision/sharding hunks; review
  it separately with an explicit compatibility decision.
- The source anchor's production adapter still runs the outer BFGS stage in the
  FP64 runtime dtype. Phase 4 deliberately extends proposal compute into FP32;
  that new exposure adopts the proposal-isolation contract recorded in
  `docs/mixed_online_biotsavart_matrix_free_single_stage_implementation_plan_2026-07-20.md`
  at the source anchor. Its snapshot and whole-pipeline fallback are outer
  orchestration requirements, not additional attempts inside the bounded mixed
  Newton primitive.
- Define
  `TraceableNewtonLinearSolver = Literal["operator_gmres", "dense_lu",
  "hybrid_final_dense_lu", "hybrid_final_dense_ir"]`. Add
  `linear_solver: TraceableNewtonLinearSolver = "operator_gmres"` to
  `newton_polish_traceable()` and the corresponding public
  `BoozerSurfaceJAX.options["newton_linear_solver"]` selector. The adapter must
  validate and pass this value explicitly; the runner cache key must include
  it. Do not port the source branch's import-time
  `SIMSOPT_TRACEABLE_NEWTON_LINEAR_SOLVER` selector or its aliases. Document the
  source-to-target migration to the typed keyword/option and test that the old
  environment spelling cannot change the target default.
- QFM commit `0d4f82ddc` is deferred to a follow-up PR.
- The only new benchmark entrypoint in this series is
  `benchmarks/jax_precision_upgrade_gate.py`, with contract tests in
  `tests/integration/test_jax_precision_upgrade_gate.py`.
  `benchmarks/paired_bca.py` is its reusable, dependency-light statistical
  utility, not a second benchmark owner. Both must use native, synthetic
  fixtures and must not read campaign artifacts or import examples.
- Authoritative GPU signoff runs on one Perlmutter GPU allocation. FP64 and
  mixed lanes run sequentially on the same recorded GPU UUID, environment, and
  source commit. If that allocation is unavailable, GPU signoff is blocked;
  results from unmatched devices are diagnostic only.

## Implementation Plan

1. Freeze the integration boundary and provenance manifest.
   - [ ] Confirm `git status --short --branch` is clean on
     `pr/jax-port-squashed` before implementation begins.
   - [ ] Run `git fetch upstream_check master`, record the refreshed upstream
     HEAD, and inspect `631cbe736..upstream_check/master` for conflicts with the
     planned public/runtime surfaces before editing.
   - [ ] Capture `upgrade_base=$(git rev-parse pr/jax-port-squashed)` and record
     the target HEAD, refreshed upstream HEAD, upstream merge base, and source
     anchor in the first implementation commit message or PR notes.
   - [ ] Create `jax-upstream-final-upgrades` from the captured
     `pr/jax-port-squashed` commit.
   - [ ] For every source commit in the provenance table, inspect both its
     complete file list and the final state at `5fb968188`; do not infer scope
     from the subject line alone.
   - [ ] Preserve target commit `ef4c8681588a31291a14111d355b82032dcef430`,
     especially the explicit `jax.device_get()` host boundary in
     `src/simsopt_jax/solve/minimize_runtime.py` and target-only sharding
     helpers. Do not copy source-anchor hunks that broadly allow transfers or
     remove those target safeguards.
   - [ ] Reject any candidate patch that adds or modifies `examples/`, generated
     artifacts, `.Codex/`, or campaign-only runtime modules.
   - [ ] Maintain a per-phase staged-file manifest and verify it with
     `git diff --cached --name-only` before each commit.

2. Establish the typed numerical and runtime policy boundary.
   - [ ] Add `src/simsopt_jax/numerical_policy.py` with only the policy,
     certificate, fallback, and typed evidence structures required by
     production backend, optimizer, host-boundary, and adapter code.
     The production allowlist is the mixed dense-IR accuracy policy and
     singleton; optimizer history capacities/source, correction limits, Armijo
     coefficient, and backend code; host-boundary two-word key and PRNG
     implementation types; and adapter refinement, solve, fallback,
     fresh-versus-replay challenge, probability/sampling, and K-seed evidence
     consumed by selected production callers.
   - [ ] Do not include benchmark report schemas or historical campaign payload
     migration, comparator schemas, cross-lane limits, final-gradient campaign
     policy, or producer-lineage schemas unless a selected production caller
     demonstrably requires them.
   - [ ] Add the defaulted `precision="mode_default"` stored selection to
     `BackendConfig`, add the omission-sentinel signature
     `precision: PrecisionSelection | None = None` to `set_backend()` and
     `use_runtime()`, and extend `BackendPolicy` in
     `src/simsopt_jax/backend/runtime.py` with the resolved precision, compute
     dtype, and `CertificateDType | None`. Keep runtime and host result dtype
     FP64 in mixed mode; keep `jax_cpu_float32_smoke` fully FP32.
   - [ ] Preserve every existing mode's current default dtype and behavior.
   - [ ] Set JAX matmul precision to `"highest"` only for resolved mixed FP32
     execution so TF32 cannot weaken the numerical contract. Preserve the
     target's current matmul setting for `mode_default`, explicit FP64, and
     `jax_cpu_float32_smoke`; add tests for both TF32 blocking and compatibility
     defaults.
   - [ ] Export `PrecisionSelection`, `ResolvedPrecision`, and the resolved
     precision helpers through the
     existing backend/config public surface. Do not add a parallel
     `PrecisionPolicy` object. Parse and synchronize only `SIMSOPT_PRECISION`;
     reject the source branch's `SIMSOPT_MIXED_PRECISION` spelling.
   - [ ] Port the required helpers and exports in
     `src/simsopt_jax/backend/dtypes.py`, `src/simsopt_jax/config.py`, and
     `src/simsopt_jax/runtime/host_boundary.py`.
   - [ ] Port the selected `1f71046a7` placement semantics so
     `_device_put_preserving_dtype()` and `explicit_device_array()` preserve an
     explicitly requested FP32 dtype rather than reapplying the runtime FP64
     dtype. Add the direct regression to
     `tests/test_backend_dtypes_reference_sharding.py`.
   - [ ] Add a backend/precision compatibility matrix and tests proving that
     omitted precision preserves every existing mode, including
     `jax_cpu_float32_smoke`, and that unsupported native/smoke combinations
     fail loudly.
   - [ ] Inventory every in-repository caller of `BackendConfig`,
     `BackendPolicy`, `set_backend()`, and `use_runtime()`; add a migration
     example for explicit mixed selection and a constructor-compatibility test
     for the defaulted field.
   - [ ] Extend the static, typed cases in
     `tests/subprocess/jax_runtime_cases.py` and add subprocess tests proving the
     normalized `PrecisionSelection` in `SIMSOPT_PRECISION` round-trips through
     `set_backend()` and a fresh interpreter. Cover omitted selection honoring
     the environment, explicit `mixed` winning over a conflict, explicit
     `mode_default` clearing a conflict, and smoke-mode propagation retaining
     `SIMSOPT_PRECISION=mode_default` rather than serializing `fp32_smoke`.
     Invoke those named cases through the existing subprocess harness; do not
     generate Python source or use `exec`, `compile`, or `python -c`.
   - [ ] Add `SIMSOPT_PRECISION` to the root `tests/conftest.py` runtime
     environment snapshot/restore owner so precision selection cannot leak
     between tests. Do not add the source branch's obsolete
     `SIMSOPT_MIXED_PRECISION` spelling.
   - [ ] Port the selected `1f71046a7` mixed-collection bootstrap atomically:
     add `tests/integration/__init__.py` and change
     `tests/integration/conftest.py` to import
     `._backend_test_helpers` relatively. The package marker must give the
     integration conftest a qualified module name so it cannot shadow the root
     `tests/conftest.py` when one pytest invocation collects root, geo, and
     integration paths.
   - [ ] Reject every value outside `{"mode_default", "fp64", "mixed"}` with an actionable
     `ValueError` from the runtime resolver before JAX runtime initialization;
     test invalid explicit and environment inputs through the public entrypoint
     and a fresh subprocess.

3. Port the FP64 dense-IR solver as an opt-in capability.
   - [ ] Reconcile the factor-once dense-IR materialization and solve path into
     `src/simsopt_jax/geo/optimizers/optimizer.py`.
   - [ ] Port strict-cap retry routing through dense LU and the lazy-chord factor
     reuse behavior. Reconstruct the selected non-default lineage from
     `1d055547e`, `139c05880`, `9bd9661b9`, `8d4a1103b`, and `db6906fc9`
     before applying `37b65c7af`; do not copy the `9bd9661b9` change to the
     default `operator_gmres` route.
   - [ ] Preserve existing fail-closed condition, backward-error, and
     ill-conditioning gates already present on the target.
   - [ ] Generate dense operator columns without materializing a quadratic
     identity constant, using the final behavior from `b9732104b`.
   - [ ] Port `_materialize_dense_hessian_host()` as an independent host
     implementation from selected `9ca8929f5`; it must not call the device
     `lax.map` materializer. Preserve the existing monkeypatch regression and
     add `tests/geo/test_optimizer_jax_item19.py` to focused validation.
   - [ ] Port the exact-zero-RHS operator-solve short circuit from selected
     `7d488caacc`: return a successful zero solution without invoking GMRES.
     Do not port the separate tiny-nonzero-solution condition exception.
   - [ ] Port the static CUDA dense-operator chunk auto-sizing finalized by
     `ecdba5011` and `aa7612a05`, retaining explicit override and the
     conservative non-CUDA default. Keep the helper private.
   - [ ] Port `adjoint_hessian_stabilization()` from selected `5df801e1b` and
     its final application from `e7b74254a`. Apply it consistently when
     materializing the host-controlled, bounded-mixed, and traceable Newton
     final Hessians. Damping may change the Newton step, but the returned
     accepted-state Hessian must be undamped for dense and CG adjoint authority;
     preserve stabilization only for explicitly augmented residual-J LSMR
     formulations. Adapt the helper to the target's existing adjoint selector:
     support the selected `"lsmr_j"` path, and do not import the source-only
     `HessianLinearSolver.LSMR_IR` enum member or its environment-selected
     comparator.
   - [ ] Add the exact typed `linear_solver` keyword and
     `BoozerSurfaceJAX.options["newton_linear_solver"]` contract from the
     resolved decisions. Keep `operator_gmres` as the target default; require
     explicit `hybrid_final_dense_ir` selection during this PR, validate the
     four canonical values, and do not accept the source environment aliases.
   - [ ] Include the selector in every affected runner/cache identity and result
     label so changing it cannot reuse a compilation or report the wrong lane.
   - [ ] Add observable tests to `tests/geo/test_adjoint_cg_solver.py` and
     `tests/geo/test_boozersurface_jax_private.py` for factor reuse, nonsymmetric
     column ordering, retry behavior, and FP64 certificate acceptance.
   - [ ] Add the `4abc6982e`/`3a64837b2` regressions: an x-dependent Hessian
     near-target fixture that distinguishes factorization at the correct state,
     plus fail-loud equivalence tests for ill-conditioned and rejected solves.
   - [ ] Port the focused final-Hessian ownership regressions from
     `tests/geo/test_boozersurface_jax.py`: host and traceable Newton step
     damping with an undamped returned Hessian, solver-specific stabilization
     ownership, bounded-mixed final-Hessian behavior, undamped dense adjoint
     runtime state, and cache/bundle identity independence from dense Newton
     step damping. Adapt the solver-specific parameterization to the selected
     dense, CG, and existing `"lsmr_j"` routes; do not copy the deferred
     `LSMR_IR` case.

4. Thread mixed compute dtype through pure kernels and private optimizers.
   - [ ] Port compute-dtype propagation through
     `src/simsopt_jax/core/_device_scalars.py`, `_math_utils.py`,
     `biotsavart.py`, `curve_geometry.py`, `curve_kernels.py`, `field.py`,
     `specs.py`, `surface_fourier_kernels.py`, and `surface_rzfourier.py`, plus
     `src/simsopt_jax/geo/label_constraints.py` and
     `src/simsopt_jax/geo/boozer_residual.py`. In
     `toroidal_flux_jax()`, construct the quadrature divisor in the active
     vector-potential dtype instead of forcing FP64 so the residual graph
     preserves the resolved compute dtype.
   - [ ] In `boozer_residual.py`, port the final `303ca6ed1` behavior reconciled
     with `9c3b6523e`: preserve caller dtype in decision splits and residual
     inputs, use explicit scalar/vector compute dtype, stage literals relative
     to live arrays, keep CPU-ordered values/gradients in `B.dtype`, request
     compute dtype from surface geometry, and thread `dtype=B.dtype` through
     composed residual and coil-VJP paths.
   - [ ] Extend `tests/geo/test_label_constraints_jax.py` with a direct
     `toroidal_flux_jax()` dtype regression: pass matching FP32 vector-potential
     and tangent inputs and require an FP32 scalar result, while retaining the
     corresponding FP64 result contract. Numerical value and derivative parity
     alone do not prove this dtype boundary.
   - [ ] Port the selected `8e3d2a784` pairwise/curve-objective slice as one
     coherent owner migration: add
     `src/simsopt_jax/core/_pairwise_reductions.py`, convert
     `src/simsopt_jax/geo/_pairwise_reductions.py` to a compatibility
     re-export, move the reusable length/curvature/curve-distance kernels into
     `core/curve_kernels.py`, and route
     `src/simsopt_jax_adapters/geo/curve_objectives.py` through those kernels.
     Exclude project-specific application owners, specs/exports, and tests.
   - [ ] Port the `7d488caacc` fixed reduction tree in
     `src/simsopt_jax/core/reductions.py`; remove the short-axis
     `lax.slice_in_dim` path so VJP on three-component axes remains valid under
     `jax.transfer_guard("disallow")`. Reconcile the same commit's selected
     static selector/update portion of `curve_geometry.py`.
   - [ ] Port mixed-dtype support through private BFGS, LBFGS, common optimizer,
     line-search, and result-conversion modules.
   - [ ] Ensure proposal arrays and explicitly compute-dtype intermediates may
     use FP32 while kernel-declared FP64 reductions/results, decision state,
     host-materialized public results, and certificate computations retain their
     declared FP64 contract.
   - [ ] Preserve transfer-guard-safe scalar and constant staging; do not add
     host callbacks to traced hot paths. Port the selected `8120b0ede`
     behavior that maps a rank-zero scalar derived from a nonreplicated
     `NamedSharding` reference to replicated `PartitionSpec()` and stages
     `curve_kernels.py` literals through the shared helper.
   - [ ] Resolve and validate the immutable numerical policy once at the public
     boundary before tracing. Thread its retained dtype, tolerance,
     certificate, and fallback fields explicitly through kernels and cached
     runners; kernels must not reconstruct thresholds or reread environment
     state. Include every behavior-affecting retained field in cache identity.
   - [ ] Add independent policy-mutation tests that change one retained field at
     a time and observe the intended kernel/certificate behavior, proving the
     production path consumes the numerical-policy owner rather than duplicated
     constants.
   - [ ] Add focused dtype and strict-transfer tests without importing example
     modules. Include a replicated-scalar `NamedSharding` regression,
     `tests/core/test_reductions.py` short-axis VJP coverage, and dense/chunked
     value-and-gradient parity in `tests/geo/test_curve_objectives_jax.py`.
   - [ ] Port the selected `5df801e1b` eager
     `SurfaceRZFourier` linear-derivative regression in
     `tests/geo/test_surface_rzfourier_transfer_guard_jax.py`. Exercise
     `_surface_rz_fourier_derivative_lin_from_spec()` outside `jit` with
     device-resident quadrature under `jax.transfer_guard("disallow")`; compiled
     scalar-gradient coverage alone does not exercise the selected
     device-relative zero/sign construction.
   - [ ] Create or port
     `tests/geo/test_traceable_bundle_mixed_lowering.py` and
     `tests/geo/test_traceable_predictor_dtype_guard.py`; these files do not
     exist on the target baseline and must be present before their validation
     command is run.

5. Port mixed dense-IR certification and fallback.
   - [ ] Port FP32 proposal-matrix construction into `optimizer.py`. Certify
     against the live matrix-free FP64 operator through FP64 matvecs; do not
     materialize a separate FP64 certificate matrix. A single canonical FP64
     dense refactor/solve is allowed only as the bounded fallback attempt.
   - [ ] Port bounded refinement history, contraction checks, effective linear
     tolerance normalization, factor authority, and fallback termination
     semantics.
   - [ ] Preserve the randomized contraction-certificate contract: pass the
     complete two-word `uint32` Threefry key, mint fresh entropy only after the
     live operator inputs are frozen, distinguish fresh-run authority from an
     explicitly requested replay key, and preserve the exact key through
     serialization even when x64 is disabled. Label the finite-PRNG sampling
     evidence separately from the ideal-Gaussian probability-model bound; do
     not present the latter as an unconditional PRNG failure probability.
   - [ ] Require a live FP64 certificate before accepting a mixed proposal;
     never return an uncertified FP32 endpoint.
   - [ ] Reserve the single canonical FP64 fallback for proposal,
     refinement/tolerance, contraction, or condition rejection. After that
     attempt, a nonfinite or out-of-tolerance final gradient or adjoint result
     fails closed; it does not trigger a third attempt.
   - [ ] Implement the bounded mixed Newton state machine exactly: take one FP32
     dense-IR proposal from the FP64 incumbent; after an accepted step that
     remains above tolerance, take the second logical attempt with fresh FP32
     factors against the live FP64 operator and permit one conditional FP64
     refactor only if that factor attempt is rejected. If the first proposal is
     rejected, go directly to the canonical FP64 attempt at the same incumbent.
     Do not add an unconditional FP64 rerun or a third bounded-Newton attempt.
   - [ ] Because Phase 4 newly moves the outer pre-Newton/BFGS proposal from its
     source-anchor FP64 runtime dtype to FP32 compute, capture an immutable
     snapshot of the FP64 decision vector, warm start, solver/cache identity
     tokens, and accepted-state ownership. Never publish the FP32 BFGS endpoint:
     re-evaluate the original and proposal seeds with the live FP64 objective
     and gradient, then reuse `_newton_candidate_status` as the sole
     stationarity/Armijo merit owner with unit step and the exact convention
     `dx = original_seed - proposal_seed`, matching its
     `x_next = x - alpha * dx` update. Do not add a second threshold or formula.
     After this seed gate passes, use the proposal only as input to the bounded
     mixed Newton state machine, whose final live-FP64 certificate authorizes
     the returned result. On a seed/certificate failure, discard mixed state,
     restore the snapshot, and run the complete canonical FP64
     pre-Newton/Newton pipeline once.
   - [ ] Buffer or suppress callbacks from speculative mixed work until final
     certification. Publish only the accepted attempt's ordered lifecycle; on
     canonical fallback, discard speculative events so external observers never
     receive an abandoned trajectory that array/token restoration cannot undo.
   - [ ] Keep mixed and FP64 Newton producers distinct until they normalize into
     the common public result contract.
   - [ ] Port only production-required portions of
     `src/simsopt_jax_adapters/geo/factor_handoff_identity.py`.
   - [ ] Add its direct production dependency
     `src/simsopt_jax/runtime/exact_numeric_identity.py` from selected
     `5df801e1b`. Test that tree identity binds structure, dtype, shape, and
     bytes, and rejects object/nonfinite leaves in a new dependency-light
     `tests/test_exact_numeric_identity.py`; do not import the source campaign
     probe.
   - [ ] Port the `950fb5ca7` routing invariant: an explicit adjoint selector
     overrides supplied factor metadata, and selecting LSMR for the adjoint
     path never changes the K1 forward predictor, with or without supplied
     factors.
   - [ ] Add tests for FP32-BFGS-proposal-to-bounded-Newton handoff, exact
     snapshot restoration on every outer fallback trigger,
     final-gradient/adjoint fail-closed
     behavior inside the bounded primitive, outer canonical fallback after a
     failed final certificate, the exact seed-gate direction and shared
     stationarity/Armijo threshold owner, speculative callback isolation, full key
     round-trip/fresh-versus-replay authority, and the forward/adjoint
     factor-routing invariant.
   - [ ] Add an explicit regression proving the bounded mixed primitive is
     independent of the default Newton runner and its iteration shape; adapt
     only the example-independent assertion from the source test to the typed
     precision API.
   - [ ] Create or port `tests/test_runtime_host_boundary.py` for the exact
     two-word `uint32` contraction-probe key, strict transfer-guard behavior,
     post-freeze fresh entropy, explicit replay authority, and x64-disabled
     round-trip. Keep the test independent of examples and historical
     artifacts.
   - [ ] Do not add `src/simsopt_jax/newton_telemetry.py` unless a production
     API—not a benchmark consumer—requires its schema.

6. Reconcile Boozer and surface-objective adapters.
   - [ ] Port mixed compute/certificate routing into
     `src/simsopt_jax_adapters/geo/boozer_surface.py` while preserving current
     FP64 result keys and public behavior.
   - [ ] Reconcile every Boozer adapter consumer of Newton stabilization with
     the Phase 3 final/adjoint owner. In `get_adjoint_runtime_state()`, pass
     `adjoint_hessian_stabilization(newton_stab)` into the Hessian solve and
     retain the residual closure for the existing `"lsmr_j"` route. In the
     penalty-kernel signature/bundle path, key on that resolved final/adjoint
     stabilization so changing dense/CG step damping does not split an
     equivalent undamped adjoint bundle, and thread the same resolved value
     through traceable objective state/signatures rather than storing raw Newton
     step damping as adjoint identity. Do not add the deferred `"lsmr_ir"`
     selector or source environment dispatch.
   - [ ] Port large-constant staging, runtime-dtype predictor solves, seeded/K2
     FP64 certificate rules, and accepted-state ownership into
     `surface_objectives_traceable.py`.
   - [ ] Port the selected `2a134a677` device-quadrature boundary: use the
     explicit host boundary in `surface_objectives.py` only when host data is
     required, and stage already-device-resident quadrature directly in
     `surface_objectives_traceable.py`. Add the source strict-transfer
     regression without example imports.
   - [ ] Port only the production lowerable-gradient seam from `7c934adc2`:
     construct one `lowerable_total_gradient_for`, reuse it in lazy and eager
     production routes, and expose that exact callable to prewarm/lowering.
     Exclude the commit's compile-evidence and campaign bundle.
   - [ ] Port only the required production changes in
     `surface_objectives.py` and
     `src/simsopt_jax_adapters/field/biotsavart_backend.py`. In the adapter's
     per-coil unit-field path, stage points, coil geometry, and unit current in
     the resolved compute dtype before `vmap` or `lax.map`, then return each
     kernel result unchanged. The real Biot-Savart kernels retain their declared
     FP64 quadrature accumulation and output dtype even when their elementwise
     inputs are FP32; do not add a helper-local cast in either direction. Port
     the per-coil boundary regression to the typed precision API and assert FP32
     staged operands together with preservation of the kernel's FP64 result.
   - [ ] Preserve spec-backed Biot-Savart graph identity through rotated/scaled
     curve and current wrappers, give spec-backed fields the required
     `clear_points()` behavior, and include symmetry-current owner segments in
     cache identity using selected `7d488caacc`, `8120b0ede`, and
     `5df801e1b` production hunks. Do not port campaign consumers.
   - [ ] Introduce `_evaluation_lifecycle.py` and `_evaluation_provider.py` as
     optimizer-owned abstractions, then route `_shared.py`, `optimizer.py`, and
     `reference.py` through them.
   - [ ] Port SciPy callback resolution against the latest exact evaluation from
     `f35f83515`; do not synthesize accepted state from an inexact callback
     vector.
   - [ ] Add public-surface tests for accepted, rejected, duplicate, and
     unresolved SciPy trial lifecycles.
   - [ ] Propagate clamped dimension and static-basis metadata as immutable
     runtime state through the geometry dispatcher and the public
     `BoozerSurfaceJAX` entrypoint. Include both in cache identities and result
     labels; test the public entrypoint, not only private helpers, with values
     that distinguish each route.
   - [ ] In `boozer_residual.py::_surface_geometry_from_dofs`, construct one
     immutable `SurfaceXYZTensorFourierSpec` carrying compact `int32` indices
     and `clamped_dims`, then call the spec-owned geometry kernels. Port the
     `7f6bf6192` companion fixture changes in
     `tests/geo/boozersurface_jax_test_helpers.py` so clones and mocks preserve
     `clamped_dims`.
   - [ ] Add the `aa47aa741` strict-transfer regression at the lazy reporting
     boundary, proving host booleans are staged outside traced/device code and
     do not cause an implicit host-to-device transfer.
   - [ ] Complete the selected `0412de980` reporting boundary by staging both
     the cached `outer_raw_terms` presence flag and every cached raw-term leaf
     relative to `solved_x`; add a strict-transfer regression.
   - [ ] Reconcile only the dependency-complete fixture portions of
     `ad73aa0f1` in `tests/geo/test_surface_objectives_jax.py`: typed
     certificate-key helpers and the final `newton_trace_capacity` mock/state
     contract. Do not copy validation-ladder, campaign-specific, or evidence
     imports from the source test wholesale.
   - [ ] Port the production `newton_trace_capacity` owner and propagation
     finalized by `add41e95c`: `BoozerSurfaceJAX` must return the full configured
     `newton_maxiter` capacity for every production lane, and
     `surface_objectives_traceable.py` must carry that static capacity through
     cache identity, traceable state, pack/pad helpers, compiled bundles, and
     every forward path. Do not port the earlier policy-dependent bounded-mixed
     capacity from `5df801e1b`; all JAX branches must share one static trace
     shape.
   - [ ] Port the unconditional `9e4b7c23f` private minimizer fixture update in
     `tests/geo/test_boozersurface_jax.py`: the selected on-device
     quasi-Newton branch already consumes `converged`, `failed`, and `k`, so its
     fake result must provide those fields in addition to `x_k`.

7. Port standalone reusable performance upgrades.
   - [ ] Add `src/simsopt_jax/core/biotsavart_online.py` from the final
     production state and connect it through `core/field.py` without campaign
     selectors.
   - [ ] Port the matching backend half of the dispatch from `2afc66397`: the
     mixed online source-tile constant,
     `FieldKernelTuning.mixed_biot_savart_source_tile_size`, and its policy
     population. Test tuning and strict-transfer dispatch so `core/field.py`
     cannot dereference a missing field.
   - [ ] Preserve bitwise agreement between the direct primal entry point and
     the custom-JVP primal while fusing primal and tangent source traversal.
   - [ ] Port the stellarator-symmetry index-scatter representation across
     `surface_fourier.py`, `surface_fourier_kernels.py`, `boozer_residual.py`,
     and the Boozer adapter.
   - [ ] Stage the scatter zero on the active device and verify that strict
     transfer guards do not introduce a host scalar.
   - [ ] Add focused online Biot-Savart, additive/homogeneous JVP, scatter
     equivalence, and device-placement tests.
   - [ ] Create or port `tests/field/test_biotsavart_online.py` and
     `tests/geo/test_surface_fourier_device_placement.py`; both are absent from
     the target baseline.
   - [ ] Add `benchmarks/jax_precision_upgrade_gate.py` and its contract test.
     Pre-register a count `N >= 20` with `N % 4 == 0` fresh independent paired
     process blocks on the same GPU UUID, using the temporally balanced
     predeclared schedule `(AB, BA, BA, AB) * (N / 4)`. Tamper-test the complete
     schedule, not only equal aggregate AB/BA counts. In each process, use at
     least one warmup followed
     by at least three synchronized repetitions and reduce those repetitions to
     one lane median. Admit a timing pair only after both lanes pass their
     scientific correctness and FP64-certificate gates. Require the median
     paired FP64/mixed speedup and the lower bound of a deterministic one-sided
     95% BCa paired bootstrap confidence interval, using 10,000 replicates and
     seed `20260714`, to be strictly greater than `1.0`; treat the ratio of
     aggregate lane medians as diagnostic only. Report both the explicitly
     labeled FP64/mixed time ratio and the percentage time reduction
     `100 * (1 - mixed_time / fp64_time)` with units; never describe either as
     “times lower.” Report peak device memory and
     fallback frequency without substituting static compiled-buffer estimates
     for observed peak memory.
   - [ ] Add the dependency-light `benchmarks/paired_bca.py` utility from
     `1f71046a7`, or reproduce its final algorithm exactly in the new gate:
     tie-aware bias correction, jackknife acceleration, deterministic
     bootstrap, one-sided bounds, and fail-loud degenerate-sample handling.
     Adapt its pure deterministic/distinctness/degenerate assertions into
     `tests/integration/test_jax_precision_upgrade_gate.py`; do not import the
     source campaign comparator.
   - [ ] Make the parent wrapper preserve a nonzero child exit code and attach
     the child's status and stderr before any success-only RSS/GPU telemetry is
     interpreted. Add a contract test for child failure precedence.
   - [ ] Derive every scientific pass boolean with one canonical validator that
     reloads the persisted raw metrics, requires finite values and dtype
     evidence, and applies the exact committed thresholds. Missing, null,
     nonfinite, malformed, or contradictory metrics must produce a failed gate.

8. Port native derivative and replay correctness fixes.
   - [ ] Add full-gradient projection behavior for fully fixed Optimizable
     lineages in `src/simsopt/_core/derivative.py`.
   - [ ] Reconcile the source-anchor companion behavior in the same module:
     `Derivative.__call__()` returns `np.empty((0,), dtype=np.float64)` when no
     free lineage contributes instead of calling `np.concatenate([])`, and
     `derivative_dec` preserves the wrapped callable's metadata with
     `functools.wraps`.
   - [ ] Implement that projection without copying the source commit's new
     function-local import. Reuse the existing validated derivative path or
     move the runtime type dependency to a static module boundary without
     creating an import cycle.
   - [ ] Port the explicit public replay allowlist from `df4b5b711`:
     `NonQuasiSymmetricRatio._fixed_surface_value()`,
     `NonQuasiSymmetricRatio.fixed_surface_value_and_derivative()`, and
     `BoozerResidual.fixed_surface_value_derivative_and_y_partial()`. Reconcile
     only the later `G: float | None` annotation from `5e3208281`.
   - [ ] Exclude `BoozerSurfaceReducedAdjointCertificate`,
     `BoozerSurfaceReducedObjective`, and
     `boozer_surface_y_stationarity_outer_vjp()` from this series; they require
     the broader genuine-675/remediation dependency chain.
   - [ ] Port the narrow native curve host-ownership slice from selected
     `5df801e1b`. In `src/simsopt/geo/curve.py`, implement native
     `dincremental_arclength_by_dcoeff_vjp()` and `kappa_impl()` with NumPy-owned
     operations; retain the JAX kernel in an explicit `JaxCurve.kappa_impl()`.
     In `src/simsopt/geo/curveobjectives.py`, make native `CurveLength.J()` and
     `dJ()` use the host incremental-arclength value and its analytic constant
     cotangent instead of sending NumPy arrays through a jitted JAX gradient.
     This is a required strict-transfer boundary, not authorization to port the
     commit's broad curve-objective batching.
   - [ ] Port `tests/geo/test_curve_length_transfer_guard.py` and the focused
     native/JAX curvature regression in `tests/geo/test_curve.py`. Require native
     `CurveLength.J()/dJ()` and native curvature to run under
     `jax.transfer_guard("disallow")`, while preserving JAX-curve numerical
     behavior.
   - [ ] Port the native curve-surface physical-partial slice from selected
     `5df801e1b`: add
     `src/simsopt/geo/_curve_surface_distance_owners.py`, make
     `CurveSurfaceDistance` depend on the surface, and return the surface VJP
     from `dJ(partials=True)`. Port the matching public
     `CurveSurfaceDistanceJAX` owner and four-input derivative behavior.
   - [ ] Update the native `CurveSurfaceDistance.dJ` and adapter
     `CurveSurfaceDistanceJAX.dJ` docstrings to state that derivatives cover
     both curve and surface DOFs. Port the focused introspection assertions so
     callable-owned documentation and `docs/source/geo.rst` cannot drift apart.
   - [ ] Preserve the physical derivative basis for fixed surfaces, including
     the selected `5df801e1b` Boozer label-gradient fix after
     `surface.fix_all()`.
   - [ ] Ensure replay values and primitive partials come from the same FP64
     source of truth.
   - [ ] Add focused tests that construct native Optimizable fixtures directly;
     do not import the single-stage example package. Include owner and
     directional finite-difference coverage for native and JAX curve-surface
     objectives and the fixed-surface Boozer label-gradient regression.
   - [ ] Add an independent finite-difference directional-derivative spot check
     for one fixed-surface objective so value replay, primitive partials, and
     the physical derivative basis are not validated only against each other.

9. Close packaging, documentation, and source-boundary work.
   - [ ] Export only intentional public precision and solver-policy symbols from
     package `__init__.py` files.
   - [ ] Update `docs/source/jax_gpu_setup.rst` with explicit FP64 and mixed
     selection, certificate/fallback behavior, synchronized timing, and
     compatibility defaults.
   - [ ] Update the migration SSOT `docs/source/jax_migration.rst` with
     `precision=`, `SIMSOPT_PRECISION`, compatibility defaults, removal of the
     source-only environment spellings, and typed opt-in dense-IR selection.
   - [ ] Update `docs/source/geo.rst` so the public
     `CurveSurfaceDistance` description states that derivatives cover both curve
     and surface DOFs.
   - [ ] Port the independent `82e9b88d3` documentation correction in
     `src/simsopt_jax/core/magnetic_axis_helpers.py`: describe the implemented
     local DOPRI5 exponent instead of the incorrect PI(0.7, 0.4) controller.
   - [ ] Document that mixed precision changes proposal computation but not the
     FP64 acceptance authority.
   - [ ] Confirm that no new dependency is required for this plan. Keep Diffrax
     and Poincare work in a separate follow-up plan and PR.
   - [ ] Run a source-boundary audit proving production modules, selected tests,
     and selected benchmarks do not import from `examples/`.
   - [ ] Port the example-independent AST boundary from
     `tests/jax/solve/test_import_boundaries.py` so both `src/` and `examples/`
     are forbidden from importing `benchmarks`.
   - [ ] Add the CPU-safe focused tests to `.github/workflows/jax_smoke.yml`,
     install the repository-supported JAX/JAXLIB pins there, and add the
     strict-GPU correctness subset to `.github/workflows/jax_gpu_parity.yml`.
     Keep the authoritative paired performance signoff on Perlmutter; do not
     port source campaign workflows wholesale.
   - [ ] Keep the deferred QFM change `0d4f82ddc` out of this series.
   - [ ] Audit three explicit exclusion seams before finalizing the manifest:
     retain the target `backend/runtime.py` and `config.py` base without the
     source runtime-attestation imports/exports; keep
     `surface_objectives.py` on the existing numerical grouped-Biot-Savart path
     without dispatch-evidence plumbing; and adapt only selected assertions from
     source tests that otherwise import validation-ladder or project-specific owners.

10. Prepare the upstream review series.
    - [ ] Keep policy/runtime, FP64 dense-IR, mixed kernels, certification,
      adapters, performance kernels, and replay fixes in separate reviewable
      commits.
    - [ ] Include source provenance hashes in commit messages without claiming
      that the original commits were cherry-picked intact.
    - [ ] For each commit, inspect `git show --stat`, `git diff --check`, and the
      staged file manifest before proceeding.
    - [ ] Rebase after the phase-local tests pass but before the final gates; if
      conflict resolution is required, continue with
      `GIT_EDITOR=true git rebase --continue`. Rerun the complete static,
      focused CPU, relevant non-JAX, and authoritative GPU signoff on the
      rebased commit. If no content changes, prove tree identity and still
      refresh the recorded source commit and provenance before signoff.
    - [ ] Prepare PR notes describing observable API/default changes, caller
      inventory, compatibility tests, migration examples, and rollback by
      commit slice.
    - [ ] Make rollback executable: revert dependent slices in reverse order,
      with policy/runtime last; rerun the FP64 compatibility shard after each
      revert; and verify that the pre-series `upgrade_base` public mode/default
      behavior is restored. This series has no persisted-data migration.

## Validation Plan

### Environment and baseline

- [ ] Start from a clean checkout and install the documented CPU environment
  with `python -m pip install -e ".[JAX,dev,DOCS]"` and install the documented
  Doxygen system prerequisite; use
  `python -m pip install -e ".[JAX_GPU,dev]"` on the GPU worker. The repository
  pins JAX/JAXLIB `0.10.0` in `pyproject.toml`; both distributions require
  Python `>=3.11`, despite the broader base-project Python declaration. Record
  `python --version` and fail environment bootstrap before installation when
  that prerequisite is not met.
- [ ] Record `command -v python`, the import origins and versions of `simsopt`,
  `simsoptpp`, `jax`, and `jaxlib`, and `python -m pip check` before baseline
  tests. Do not use an environment whose imports resolve outside the checkout.
- [ ] On the untouched target commit, run and record the existing-file FP64
  shards listed below before adding mixed mode. Treat baseline failures as
  blockers or separately documented pre-existing failures, not as upgrade
  regressions.
- [ ] Preserve the captured `upgrade_base` through final review and use it for
  every new-series containment and rollback comparison.

### Static and source-boundary checks

- [ ] Run `git diff --check` after every phase and
  `git diff --check "$upgrade_base"..HEAD` across the complete upgrade range.
- [ ] Run `python -m compileall -q src/simsopt src/simsopt_jax src/simsopt_jax_adapters`.
- [ ] Run
  `python scripts/jax_where_division_lint.py src/simsopt_jax src/simsopt_jax_adapters`
  and `python -m pytest -q tests/test_jax_import_smoke.py` to preserve the
  existing JAX CI lint and static import boundary.
- [ ] Run `ruff check src/simsopt src/simsopt_jax src/simsopt_jax_adapters`, then
  pipe the NUL-delimited changed Python manifest from
  `git diff --name-only -z --diff-filter=ACMR "$upgrade_base"..HEAD -- '*.py'`
  through `xargs -0 -r ruff check`.
- [ ] Pipe that same changed Python manifest through
  `xargs -0 -r ruff format --check`.
- [ ] Run
  `rg -n '(from|import) examples|examples\.single_stage_optimization' src/simsopt src/simsopt_jax src/simsopt_jax_adapters`
  and require no production imports. Repeat the same scan over the changed test
  and benchmark manifest and require no matches there.
- [ ] Run the AST boundary in
  `tests/jax/solve/test_import_boundaries.py` and require that neither `src/`
  nor `examples/` imports `benchmarks`.
- [ ] Run `git diff --name-only "$upgrade_base"..HEAD -- examples` and require
  no output. Comparing against the upstream merge base is invalid here because
  it includes pre-existing files from the base JAX port.
- [ ] Audit every changed production, test, and benchmark path for `importlib`,
  `__import__`, loader execution, `exec`, `compile`, `python -c`, and
  function-local imports; require no newly introduced dynamic, generated-code,
  or local imports.
- [ ] Run `sphinx-build -W -b html docs/source /tmp/simsopt-jax-docs-build` after
  updating `docs/source/jax_gpu_setup.rst`,
  `docs/source/jax_migration.rst`, and `docs/source/geo.rst`.
- [ ] Run one mixed-directory collection command after adding the integration
  package marker:
  `python -m pytest --collect-only -q tests/test_backend_dtypes_reference_sharding.py tests/geo/test_boozersurface_jax.py tests/integration/test_factor_once_adjoint_phase2.py`.
  Require both root and integration conftest owners to load without module-name
  collision.

### Focused CPU tests

- [ ] Run `python -m pytest -q tests/test_backend_dtypes_reference_sharding.py tests/test_backend_strict_jax_device_detection.py tests/test_runtime_host_boundary.py`.
- [ ] Run `python -m pytest -q tests/test_exact_numeric_identity.py tests/jax/solve/test_import_boundaries.py`.
- [ ] Run `python -m pytest -q tests/geo/test_adjoint_cg_solver.py tests/geo/test_boozersurface_jax_private.py tests/geo/test_optimizer_jax_item19.py`.
- [ ] Run `python -m pytest -q tests/geo/test_boozersurface_jax.py tests/geo/test_surface_objectives_jax.py`.
- [ ] Run `python -m pytest -q tests/geo/test_curve_objectives_jax.py`.
- [ ] Run `python -m pytest -q tests/geo/test_curve.py tests/geo/test_curve_length_transfer_guard.py`.
- [ ] Run `python -m pytest -q tests/geo/test_surface_rzfourier_transfer_guard_jax.py`.
- [ ] Run `python -m pytest -q tests/geo/test_optimizer_jax_reference.py tests/geo/test_traceable_bundle_mixed_lowering.py tests/geo/test_traceable_predictor_dtype_guard.py`.
- [ ] Run `python -m pytest -q tests/field/test_biotsavart_jax.py tests/field/test_biotsavart_online.py tests/geo/test_surface_fourier_device_placement.py`.
- [ ] Run `python -m pytest -q tests/geo/test_label_constraints_jax.py tests/geo/test_boozer_residual_jax.py tests/geo/test_curvexyzfouriersymmetries_spec_jax.py tests/geo/test_surface_fourier_jax.py` and require the label-constraint, residual, curve-spec, and surface-from-DOFs paths to preserve the selected compute dtype.
- [ ] Run `python -m pytest -q tests/core/test_derivative.py tests/core/test_reductions.py tests/geo/test_surface_objectives.py tests/geo/test_boozersurface.py tests/integration/test_factor_once_adjoint_phase2.py`.
- [ ] Run `python -m pytest -q tests/integration/test_jax_precision_upgrade_gate.py`.
- [ ] Implement native synthetic snapshot, seed-gate, state-isolation, and
  whole-pipeline fallback regressions from the source-anchor design contract in
  `docs/mixed_online_biotsavart_matrix_free_single_stage_implementation_plan_2026-07-20.md`.
  Add the separate target-side speculative callback-isolation regression
  required by Phase 5; do not attribute it to that source document.
  Reuse only the example-independent bounded-Newton fallback fixtures that
  actually exist in the source anchor's
  `tests/integration/test_mixed_precision_bfgs_newton_ab_gate.py`; do not copy
  its campaign schemas or artifact consumers.
- [ ] Run the existing target FP64 regression tests before enabling any mixed
  mode so target-only hardening cannot regress unnoticed.

### Precision and fallback checks

- [ ] Verify identical default route, dtype, and observable results before and
  after the port when no mixed mode is selected.
- [ ] Verify mixed proposals use FP32 compute buffers while live matrix-free
  certificate matvecs, final gradients, and accepted public results satisfy the
  FP64 contract; assert no FP64 certificate matrix is materialized before the
  single canonical fallback.
- [ ] Force each mixed rejection condition and verify deterministic canonical
  FP64 fallback: nonfinite proposal, failed refinement, contraction failure,
  unsafe condition estimate, and tolerance miss. Separately force a
  final-gradient and final-adjoint miss after the canonical attempt and verify
  fail-closed termination with no additional attempt.
- [ ] Verify the FP32 BFGS endpoint is used only as the seed for the bounded
  mixed Newton state machine, never as a returned result. Verify its final
  live-FP64 certificate, exact two-logical-attempt trace, conditional single
  FP64 refactor, and that every outer fallback starts from a byte-identical
  pre-mixed decision/warm-start/token/accepted-state snapshot.
- [ ] Verify the FP64 seed gate evaluates both original and proposal seeds,
  calls `_newton_candidate_status` with unit step and
  `dx = original_seed - proposal_seed`, and shares its exact stationarity and
  Armijo thresholds without a duplicate acceptance formula.
- [ ] Under both fresh and explicit-replay contraction probes, verify the full
  two-word `uint32` key round-trip with x64 disabled, post-freeze entropy
  ordering, and distinct finite-PRNG versus ideal-Gaussian evidence labels.
- [ ] Verify supplied factor reuse is accepted only when its exact identity and
  certificate authority match the current state.
- [ ] Verify SciPy callbacks accept the latest exact duplicate and leave unknown
  callback states unresolved rather than fabricating evidence.
- [ ] Verify the exact-zero RHS path returns a successful zero solution without
  entering GMRES, while a nonzero ill-conditioned solve remains fail-closed and
  does not receive the excluded tiny-solution exception.
- [ ] With nonzero Newton stabilization, verify iteration steps differ from the
  undamped route while dense/CG returned final Hessians and adjoint cache
  identities remain those of the accepted-state undamped operator. Separately
  verify residual-J LSMR retains stabilization in its augmented operator.

### GPU validation

- [ ] In one Perlmutter allocation, record `jax.__version__`, `jaxlib` version,
  device model, GPU UUID, driver, backend, x64 state, Python/import origins,
  dependency check, and source commit before GPU execution.
- [ ] Run the focused FP64 and mixed kernel/optimizer tests sequentially on that
  same GPU UUID with `jax.transfer_guard("disallow")` active around the public
  solver entry points, not only around pure kernels.
- [ ] Compare FP64 and mixed fixed-state objectives and gradients using the
  committed numerical policy tolerances.
- [ ] Measure compile time separately from warm execution time. Synchronize every
  timed result with `jax.block_until_ready()` or a result leaf's
  `.block_until_ready()` so asynchronous dispatch cannot produce false
  speedups. Run the committed example-independent fixture and enforce its paired
  timing gates; record observed peak VRAM, allocator peak when available, host
  RSS, factor reuse, and fallback frequency.
- [ ] Require the mixed lane to preserve FP64 acceptance authority; performance
  improvement alone is not a pass condition.

### Final review gates

- [ ] Run the repository's applicable non-JAX regression suite for every
  modified `simsopt` module.
- [ ] Confirm no test or benchmark selected for the PR reads historical artifact
  paths or imports example-owned schemas.
- [ ] Review the final diff for new configuration parameters, public API changes,
  stale comments, renamed symbols, and added dependencies.
- [ ] Verify every new public precision selection or solver mode has a
  compatibility test, migration example, and rollback path.

## Risks and Mitigations

- Risk: Replacing source files wholesale removes target-only FP64 hardening.
  Mitigation: Port behavior by phase, review target and source hunks together,
  and run existing FP64 tests before mixed tests.

- Risk: The source's self-selecting dense-IR default changes observable solver
  behavior for existing users.
  Mitigation: Keep the target default and ship dense-IR as an explicit mode in
  this PR.

- Risk: An environment-only mixed toggle hides an internally important policy
  decision and is difficult to document or type-check.
  Mitigation: make the typed runtime keyword/config field authoritative, parse
  only the normalized `SIMSOPT_PRECISION` transport in the same resolver, and
  test explicit-over-environment precedence plus subprocess propagation.

- Risk: FP32 proposals pass a local residual gate while violating FP64 state,
  objective, gradient, or caller-output parity.
  Mitigation: retain independent live FP64 seed/final certification, keep the
  FP32 BFGS endpoint seed-only, restore the immutable pre-mixed snapshot before
  the outer canonical FP64 fallback, and validate final public outputs rather
  than certificate stationarity alone.

- Risk: Campaign/evidence code enters production because it resides under
  `src/`.
  Mitigation: use the explicit Non-Goals and module allowlist; reject
  `genuine_675`, attestation, publication, and remediation owners regardless of
  directory.

- Risk: Tests silently retain example dependencies after `examples/` is omitted.
  Mitigation: use direct native/JAX fixtures and enforce the source-boundary
  `rg` checks in validation.

- Risk: Mixed dtype staging introduces host-device transfers or recompilation.
  Mitigation: retain device-relative scalar construction, replicate rank-zero
  scalar sharding explicitly, use the transfer-safe fixed reduction tree, stage
  large constants once, test public solver/adapter entrypoints under strict
  transfer guards, and measure compile count and warm execution.

- Risk: Native curve objectives or eager surface derivatives accidentally send
  host NumPy values through JAX after the strict-transfer gate is enabled.
  Mitigation: keep native arclength/curvature/CurveLength operations host-owned,
  preserve an explicit JAX-curve override, and run the native curve plus eager
  `SurfaceRZFourier` regressions under `jax.transfer_guard("disallow")`.

- Risk: Newton step damping contaminates the final Hessian used for adjoint
  solves, factors, or cache identity.
  Mitigation: centralize final/adjoint stabilization ownership, return the
  undamped accepted-state Hessian for dense and CG routes, and retain damping
  only where it is mathematically part of an augmented residual-J LSMR operator.

- Risk: `tests/integration/conftest.py` shadows the root test configuration
  during mixed-directory collection, silently bypassing precision environment
  isolation.
  Mitigation: package `tests/integration`, use its relative helper import, and
  run an explicit root/geo/integration collection gate.

- Risk: A selected adapter or kernel is ported without its direct runtime,
  policy, tuning, compatibility-re-export, or fixture dependency.
  Mitigation: treat the exact-identity helper, online-field tuning, pairwise
  owner migration, curve-objective routing, clamped-surface fixture, and
  deterministic BCa utility as atomic dependency slices with focused tests.

- Risk: Copying a broad source commit changes target defaults or imports
  campaign/remediation code while fixing one reachable production seam.
  Mitigation: use the symbol/file allowlists in each phase, preserve
  `operator_gmres` and target transfer hardening, and exclude the source's
  tiny-solution exception, compile-evidence bundle, reduced-objective chain,
  and campaign schemas.

- Risk: The source contains an implementation followed by a later revert.
  Mitigation: compare with the final tree at `5fb968188`; do not replay
  superseded commits such as the reverted mixed Biot-Savart reduction directly.

- Risk: The combined change remains too large for upstream review.
  Mitigation: retain the phase-aligned commit series and allow maintainers to
  review or merge precision foundation, dense-IR, mixed mode, and performance
  kernels independently.

- Risk: A containment check against the upstream merge base reports old JAX-port
  example changes as part of this upgrade or hides rollback boundaries.
  Mitigation: freeze `upgrade_base` at the target tip and use that exact commit
  for all upgrade-series file, diff, and rollback checks.

- Risk: JAX asynchronous dispatch makes compile or warm execution timings look
  faster than the work actually completes.
  Mitigation: synchronize each timed result and compare paired runs on the same
  recorded GPU UUID.

## Completion Criteria

- [ ] Existing backend modes retain their prior default behavior and pass their
  prior regression tests, including the full-FP32 `jax_cpu_float32_smoke` mode.
- [ ] Dense-IR is available through an explicit solver selection and passes
  factor-reuse, independent host-materialization, exact-zero RHS, condition,
  retry, final-Hessian stabilization-ownership, and fallback tests for every
  declared typed solver value.
- [ ] Mixed precision is explicitly selectable and cannot return an endpoint
  without FP64 acceptance authority.
- [ ] Online Biot-Savart, dense-HVP, and surface-scatter upgrades pass their
  focused parity, dispatch-tuning, graph-identity, and placement tests.
- [ ] Core reductions, pairwise/curve-objective routing, scalar sharding, and
  adapter quadrature/reporting boundaries pass strict-transfer and dtype tests.
- [ ] SciPy evaluation lifecycle and derivative/replay correctness tests pass,
  including fixed-surface replay, native curve strict-transfer ownership, and
  native/JAX curve-surface physical partials.
- [ ] Eager `SurfaceRZFourier` derivatives and native CurveLength/curvature
  execute under strict transfer guards, and mixed-directory pytest collection
  loads both root and integration conftest owners.
- [ ] The deterministic paired benchmark uses the predeclared balanced schedule,
  validated BCa implementation, synchronized timing, and fail-closed scientific
  gate.
- [ ] Production code and selected validation code have no imports from
  `examples/`.
- [ ] No campaign artifact, remediation, genuine-675, canonical11, or generated
  evidence module is included.
- [ ] Static checks, focused CPU tests, GPU parity/performance gates, and relevant
  non-JAX regressions pass from a clean checkout.
- [ ] The PR description includes the API evolution, compatibility, migration,
  rollback, provenance, and validation evidence required for upstream review.

## Review-Time Change Control

- Upstream may request a different public precision spelling, but that is an API
  review change, not an implementation-time choice. Update the resolved design,
  caller inventory, compatibility tests, docs, migration example, and rollback
  notes atomically before adopting it.
- An unavailable Perlmutter GPU allocation blocks final GPU signoff. Do not
  replace the matched-device gate with results from different GPU models or
  separate environments.
- Any additional benchmark, QFM change, Diffrax integration, campaign schema, or
  example import is scope expansion and requires a separate plan/PR.
- Mixed residual-J `lsmr_ir`, native-policy GPU telemetry short-circuiting,
  atomic spec publication, and the legacy Biot-Savart static loader are separate
  behavior/dependency changes. Do not partially copy them from otherwise
  selected commits; each requires its own typed boundary, compatibility review,
  and focused tests.
- Persistent factor snapshots, HVP rematerialization controls, surface-tangent
  finite-build support, coil-spec memoization commit `ca0c420d75`, general
  batching commit `b24bad015`, and other post-anchor product features remain
  separate follow-ups unless this plan and its dependency/validation inventory
  are explicitly reopened.
- The broad `e3ac8a3d0` oracle-reference comment cleanup is documentation-only
  follow-up scope; update only comments in files already touched by this series
  unless maintainers request the complete cleanup.
