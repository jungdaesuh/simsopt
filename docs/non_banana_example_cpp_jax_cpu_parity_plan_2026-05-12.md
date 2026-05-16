# Non-Banana Example C++/JAX CPU Parity Plan - 2026-05-12

Status: plan drafted after inspecting the current example tree at
`c6fcebfd5848`. The working tree was already dirty when this document was
created; this plan is additive and does not classify unrelated uncommitted
files.

Review update: this document was reviewed against the current source and
official JAX/SIMSOPT documentation on 2026-05-12. The review fixed two
material plan bugs: position/orientation coils are not P0 JAX parity fixtures
until their oriented-curve spec contract exists, and CPU/JAX parity commands
must force JAX x64 plus explicit synchronization for measured JAX execution.

Second review update: a follow-up plan review on 2026-05-12 fixed these
additional pre-implementation issues:

1. The minimal Stage-II example's exact `QuadraticPenalty(sum(CurveLength),
   LENGTH_TARGET, "max")` term has no exact native-JAX path today and is now
   classified up front as `QuadraticPenalty_over_sum_CurveLength_max`.
2. `SquaredFluxJAX` definition coverage is now an explicit Phase 0 gate; current
   source and tests cover `"quadratic flux"`, `"normalized"`, and `"local"`.
3. Test commands now separate mandatory parity environment from diagnostic
   hardening and record the required dirty-tree posture.
4. Orphan non-banana examples are classified explicitly.
5. Comparison statistics, tolerance-bucket defaults, verdict semantics, and test
   runtime environment are specified before implementation starts.

## Goal

Use existing SIMSOPT examples, excluding the banana Stage 2 and banana
single-stage entrypoints, to build research-grade CPU-only C++/JAX parity
evidence.

Primary trust chain for this plan:

```text
existing SIMSOPT CPU/C++/SciPy example behavior
  -> JAX CPU matches at identical fixed states
  -> short optimizer/integration diagnostics remain consistent
```

GPU proof is out of scope for this document. No task here requires or authorizes
starting GPU jobs.

## Requirements Analysis

User requirements captured from the 2026-05-12 investigation:

- [ ] Find usable examples other than banana Stage 2 and banana single-stage.
- [ ] Prefer examples that already exercise production-relevant SIMSOPT
      surfaces instead of toy-only construction.
- [ ] Treat existing CPU/C++/SciPy SIMSOPT behavior as the oracle.
- [ ] Prove JAX CPU against the CPU/C++ oracle before any JAX-vs-JAX or GPU
      claim.
- [ ] Use fixed-state value/gradient parity as the first proof surface.
- [ ] Keep full optimizer trajectories as integration diagnostics, not the
      primary oracle.
- [ ] Do not loosen tolerances to hide drift.
- [ ] Keep banana-specific validation separate from this non-banana fixture
      plan.
- [ ] Keep all work CPU-only unless the user later asks for GPU proof.

Engineering requirements inferred from the current codebase and prior parity
docs:

- [ ] Reuse `benchmarks/validation_ladder_contract.py` tolerance semantics where
      applicable instead of adding one-off tolerance constants.
- [ ] Use `direct_kernel` only for direct kernel quantities, `derivative_heavy`
      for derivative-heavy kernel checks, `ls_wrapper_gradient` for wrapper
      value/gradient checks, and `gpu_runtime` only if a later GPU proof is
      explicitly in scope.
- [ ] Preserve the upstream CPU example scripts as user-facing examples. Avoid
      importing scripts directly if they execute optimization at import time.
- [ ] Build pure fixture constructors that reproduce the example initial states
      without VTK writes, plotting, or optimizer side effects.
- [ ] Build CPU and JAX lanes from a shared immutable fixture specification.
      Do not share mutable `BiotSavart`, `BiotSavartJAX`, objective, or surface
      adapter instances across lanes or threads.
- [ ] Record exact fixture inputs: source example, grid resolution, active DOF
      names, active/fixed mask, current scaling, surface source, target field,
      perturbation seed, JAX x64 status, and platform.
- [ ] Require `JAX_ENABLE_X64=1` for parity commands. A run without x64 is a
      smoke test only, not a parity proof.
- [ ] Compare objective components when the objective is composite; total `J`
      alone is not enough for research-grade diagnosis.
- [ ] Compare gradients in an explicitly named coordinate basis. If CPU uses
      `JF.x` and JAX uses a different field/coil basis, the mapping must be
      recorded and tested.
- [ ] Fail closed for unsupported native-JAX components. A mixed lane that uses
      CPU helpers inside a claimed JAX component is a diagnostic, not a pass.

## Official Docs And Current-Source Checks

The review used these source-of-truth constraints:

- [ ] JAX `jit` expects pure transformed functions. Python side effects such as
      printing run at trace/cache boundaries, not reliably at every execution.
      Fixture construction and reporting must stay outside jitted kernels.
- [ ] JAX dispatch can be asynchronous. Any timing or execution-complete claim
      must call `.block_until_ready()` or `jax.block_until_ready(...)` at the
      measurement boundary.
- [ ] JAX defaults away from x64 unless explicitly enabled. All parity commands
      must set `JAX_ENABLE_X64=1` or assert the same setting in process.
- [ ] SIMSOPT ReadTheDocs exposes the public `Optimizable`, `BiotSavart`,
      `SquaredFlux`, `CurveLength`, and `BoozerSurface` APIs. These public APIs
      define the CPU oracle behavior for this plan.
- [ ] Current source says `BiotSavartJAX` requires curves with immutable JAX
      specs and rejects unsupported geometry contracts. It also states the
      mutable adapter is not safe for concurrent shared use because `set_points`
      mutates cached evaluation points.
- [ ] Current source says `SquaredFluxJAX` captures fixed surface geometry,
      field-evaluation points, and field DOF layout at construction. The harness
      must rebuild the objective after any point or fixed/free layout change.
- [ ] Current tests document that the legacy composite `SquaredFluxJAX +
      CurveLength` pattern keeps `CurveLength` on the CPU path. Such a mixed
      composite can be used as a diagnostic, but it cannot be labeled as a fully
      native JAX objective-bundle proof.

Official documentation references checked:

- `https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html`
- `https://docs.jax.dev/en/latest/async_dispatch.html`
- `https://docs.jax.dev/en/latest/default_dtypes.html`
- `https://simsopt.readthedocs.io/latest/simsopt_user.field.html`
- `https://simsopt.readthedocs.io/latest/simsopt_user.objectives.html`
- `https://simsopt.readthedocs.io/latest/simsopt_user.geo.html`

## Why Fixed-State First

A fixed-state parity test asks:

```text
Given the same coils, currents, surface, quadrature grid, target, fixed/free
DOF layout, and dof vector x, do CPU/C++ and JAX CPU return the same J(x) and
dJ(x)?
```

That directly tests the physics and derivative implementation.

A full optimizer trajectory also includes optimizer history, line-search
candidate selection, Wolfe/termination thresholds, rollback behavior, and
floating-point path sensitivity. Small harmless differences in `J` or `dJ` can
change a later line-search decision and make the path diverge without proving a
physics bug. Therefore:

- [ ] Use fixed-state parity as the correctness oracle.
- [ ] Use deterministic perturbation checks to test local behavior around the
      initial state.
- [ ] Use optimizer runs only after fixed-state parity passes, and compare final
      public behavior envelopes rather than requiring iterate-by-iterate
      identity.

## Candidate Fixture Matrix

| Priority | Example | Useful parity surface | Why it is useful | First gate |
| --- | --- | --- | --- | --- |
| P0 | `examples/1_Simple/stage_two_optimization_minimal.py:70` | `SurfaceRZFourier`, symmetric coils, `BiotSavart`, `SquaredFlux`; length penalty support gate | Cleanest non-banana fixed-state Stage-II fixture. Small enough to isolate value/gradient drift. The exact length penalty is unsupported-native-JAX today. | Same-state `SquaredFluxJAX`, `dJ`, `B`, `B dot n`; length penalty classified unsupported |
| P1 | `examples/3_Advanced/curves_CWS_example.py:18` | Saved optimized CWS `BiotSavart` JSON, VMEC boundary, `SquaredFlux(definition="local")`, `BdotN` | Production-like read-only artifact fixture without an optimizer loop. | Same-state field, `BdotN`, local `SquaredFlux` |
| P1 | `examples/2_Intermediate/stage_two_optimization.py:107` | Full Stage-II composite: flux, length, coil-coil distance, coil-surface distance, curvature, mean-squared-curvature | Broader objective bundle after the minimal flux/length proof passes. | Native-supported component parity; CPU-only components classified as gaps |
| P2 | `examples/2_Intermediate/stage_two_optimization_planar_coils.py:98` | `CurvePlanarFourier`, `BiotSavart`, `SquaredFlux`, geometry penalties, linking number | Adds planar-coil coverage where JAX-side curve support exists. | Same-state flux/geometry components; linking-number handling classified |
| Gap gate | `examples/1_Simple/optimize_coil_position_orientation.py:54` | TF/windowpane coils, position/orientation/current DOFs, `SquaredFlux` | Useful example, but `OrientedCurveXYZFourier` currently lacks the immutable native curve spec required by `BiotSavartJAX`. | CPU fixture plus explicit unsupported-native-JAX finding until spec support exists |
| P2 | `examples/2_Intermediate/boozer.py:42` | `BoozerSurface`, `boozer_surface_residual`, `ToroidalFlux`, `Area` | Good Boozer surface contract fixture, but solver trajectory is fragile. | Fixed-state residual/label checks before solve trajectory |
| P2 | `examples/2_Intermediate/boozerQA.py:67` | `BoozerSurface`, `NonQuasiSymmetricRatio`, `Iotas`, `MajorRadius`, length penalty | High-value solved-state scalar parity after base Boozer residual/state parity is stable. | Fixed solved-state scalar outputs; public wrapper/adjoint parity remains in the dedicated BoozerSurfaceJAX lanes |
| P3 | `examples/2_Intermediate/stage_two_optimization_finite_beta.py:112` | `SquaredFlux(target=vc.B_external_normal)` | Covers nonzero target normal field, but VirtualCasing makes setup heavier. | Target-array shape/value hash, then `J`/`dJ` |
| P3 | `examples/3_Advanced/stage_two_optimization_finitebuild.py:123` | Multifilament finite-build coils, `BiotSavart`, `SquaredFlux`, length and distance penalties | Important upstream example class, but multifilament construction needs a native-spec support check before any JAX pass claim. | CPU fixture plus native-spec support classification |
| P3 | `examples/1_Simple/qfm.py:37` | `QfmResidual`, `QfmSurface`, volume/area/toroidal-flux labels | Useful if expanding beyond the core coil/flux/Boozer parity surface. | Fixed-state QFM residual and label checks |
| Diagnostic | `examples/2_Intermediate/stage_two_optimization_stochastic.py:127` | Deterministic stochastic perturbation samples, `MPIObjective`, JAX optimizer wrapper over Python Optimizable calls | Stress diagnostic only; not primary C++/JAX physics parity. | Frozen sample state, reduced-N smoke |
| CPU backlog | `examples/3_Advanced/coil_forces.py:135` | Force and energy penalties | Good CPU regression target, but outside the clean current JAX parity surface. | CPU-only regression until JAX force wrappers exist |

## Explicitly Classified Orphan Examples

These examples are intentionally not first-class fixtures in this plan:

| Example group | Classification |
| --- | --- |
| `examples/single_stage_optimization/` | Out of scope: banana Stage 2 and banana single-stage research workspace, already excluded by this plan's top-level non-banana scope. |
| `examples/3_Advanced/single_stage_optimization.py`, `single_stage_optimization_finite_beta.py`, `single_stage_optimization_curveCWSfourier.py` | Out of scope for this plan: non-banana single-stage examples need a separate single-stage non-banana plan. |
| `examples/3_Advanced/optimize_qs_and_islands_simultaneously.py` | Out of scope: VMEC/island optimization surface, not a Stage-II fixed-state C++/JAX coil-objective fixture. |
| `examples/2_Intermediate/boozerQA_ls_mpi.py` | Diagnostic sibling of `boozerQA.py`; MPI/LS behavior belongs after the fixed-state Boozer wrapper fixture. |
| `examples/2_Intermediate/B_external_normal.py` | Upstream setup for finite-beta target-field generation; referenced through the finite-beta Phase 7 fixture, not a parity fixture by itself. |
| `examples/1_Simple/permanent_magnet_simple.py`, `examples/2_Intermediate/permanent_magnet_*.py` | Out of scope: PermanentMagnetGrid optimization surface, no native JAX parity wrapper claim in this plan. |
| `examples/2_Intermediate/wireframe_*.py`, `examples/3_Advanced/wireframe_gsco_multistep.py` | Out of scope: wireframe optimization surface, no native JAX parity wrapper claim in this plan. |
| `examples/2_Intermediate/QH_fixed_resolution.py`, `QH_fixed_resolution_boozer.py`, `resolution_increase.py`, `resolution_increase_boozer.py`, `eliminate_magnetic_islands.py`, `strain_optimization.py`, `tracing_boozer.py`, `vmec_adjoint.py`, `constrained_optimization.py`, `QSC.py`, `free_boundary_vmec.py` | Out of scope: VMEC, tracing, QSC, or single-purpose diagnostic examples rather than Stage-II/Boozer-objective fixed-state parity fixtures. |
| `examples/1_Simple/just_a_quadratic.py`, `logger_example.py`, `minimize_curve_length.py`, `surf_vol_area.py`, `tracing_fieldlines_*.py`, `tracing_particle.py` | Out of scope: tutorial, logging, scalar geometry, or tracing examples outside the current C++/JAX coil-objective proof surface. |
| `examples/stellarator_benchmarks/*.py` | Out of scope: benchmark problem corpus, not part of this non-banana example parity harness. |

## Non-Goals

- [ ] Do not use banana Stage 2 or banana single-stage entrypoints in this
      plan.
- [ ] Do not extend `_pre_newton_census_gate_failures` to non-banana examples in
      this plan. That strict byte-identity gate remains banana/single-stage
      release infrastructure until a separate plan scopes non-banana coverage.
- [ ] Do not start GPU jobs.
- [ ] Do not treat JAX CPU vs JAX CPU self-consistency as CPU/C++ preservation.
- [ ] Do not treat full optimizer trajectory matching as the first oracle.
- [ ] Do not import example scripts that execute optimizers, write VTK files, or
      mutate global state at import time.
- [ ] Do not rewrite upstream examples just to make harnessing easier unless a
      later implementation pass explicitly scopes that refactor.
- [ ] Do not add defensive fallback paths that silently skip unsupported
      objective components.
- [ ] Do not classify unsupported JAX terms as passing. Mark them unsupported
      with exact component names.
- [ ] Do not label a mixed CPU/JAX composite as native JAX parity.
- [ ] Do not mutate field evaluation points or fixed/free DOF layout after
      constructing `SquaredFluxJAX`; rebuild the objective from the fixture spec.

## Planned Artifact Shape

Target harness, subject to implementation review:

- [ ] Add a benchmark/reporting entrypoint, for example
      `benchmarks/non_banana_example_cpp_jax_cpu_parity.py`.
- [ ] Add reusable fixture builders, for example
      `benchmarks/non_banana_example_parity_fixtures.py`.
- [ ] Add focused tests, for example
      `tests/integration/test_non_banana_example_cpp_jax_cpu_parity.py`.

Expected JSON schema:

- [ ] `schema_version`
- [ ] `git`
- [ ] `dirty_tree_summary`
- [ ] `jax_platform = "cpu"`
- [ ] `jax_enable_x64`
- [ ] `jax_backend`
- [ ] `jax_devices`
- [ ] `fixture_id`
- [ ] `source_example`
- [ ] `fixture_inputs`
- [ ] `dof_contract`
- [ ] `native_spec_contract`
- [ ] `lanes.cpu_cpp`
- [ ] `lanes.jax_cpu`
- [ ] `comparisons.cpu_cpp_vs_jax_cpu`
- [ ] `unsupported_components`
- [ ] `mixed_lane_diagnostics`
- [ ] `passed`
- [ ] `failures`

Each comparison entry must include:

- [ ] `quantity`
- [ ] `component`
- [ ] `tolerance_bucket`
- [ ] `tolerance_rtol`
- [ ] `tolerance_atol`
- [ ] `max_abs_diff`
- [ ] `max_rel_diff`
- [ ] `argmax_index`
- [ ] `argmax_dof_name`
- [ ] `verdict`

Allowed verdicts:

- `pass`: all required native-supported quantities pass their tolerance bucket
  and there are no unsupported required components.
- `partial`: all native-supported quantities pass, at least one component is
  explicitly listed in `unsupported_components`, and no native-supported
  component fails.
- `unsupported`: the fixture cannot construct a native JAX lane because a
  required native-spec or wrapper contract is absent.
- `fail`: at least one native-supported quantity exceeds its tolerance bucket or
  violates the fixture contract.

Required lane outputs:

- [ ] total objective `J` for fully native-supported fixtures, or the
      native-supported objective subtotal for partial fixtures
- [ ] objective components with stable names
- [ ] native-supported gradient in the declared basis
- [ ] gradient norms
- [ ] active DOF names
- [ ] active DOF vector hash
- [ ] fixed/free mask hash
- [ ] native curve spec hashes for every coil curve in the JAX lane
- [ ] surface point hash
- [ ] unit-normal hash
- [ ] `B` sample hash and max/mean statistics
- [ ] `B dot n` or target-subtracted `B dot n` statistics
- [ ] wall time split into setup, compile, and execution when JAX JIT is used,
      with `.block_until_ready()` or `jax.block_until_ready(...)` at the timing
      boundary

## Default Tolerance Mapping

Phase 0 may tighten a mapping when the current source justifies it, but it must
not invent looser fixture-local tolerances.

| Quantity | Default bucket |
| --- | --- |
| `B` at fixed surface points | `direct_kernel` |
| surface `gamma`, `unitnormal` | `direct_kernel` |
| raw `integral_BdotN` / fixed-surface flux kernel | `direct_kernel` |
| `B dot n` arrays and summaries | `direct_kernel` |
| `SquaredFluxJAX` wrapper value and gradient | `ls_wrapper_gradient` |
| `SquaredFluxJAX` finite-difference directional derivative checks | `fd_gradient` |
| Boozer residual at fixed state | `direct_kernel` |
| Boozer residual Jacobian/JVP/VJP | `derivative_heavy` |
| `IotasJAX`, `MajorRadiusJAX`, `NonQuasiSymmetricRatioJAX` wrapper values and gradients | `branch_stable_resolve` plus `fd_gradient` diagnostics unless a stricter direct wrapper lane is already available |
| BoozerQA copied-solved-state scalar helpers | `direct_kernel`; not a public `BoozerSurfaceJAX` wrapper/adjoint lane |
| QFM residual at fixed state | `direct_kernel` |
| Composite objective total over mixed native/unsupported components | no pass bucket; report `partial` or `unsupported` |

## Math, Physics, And Computation Gates

- [ ] Surface geometry parity uses the same `nphi`, `ntheta`, range, `nfp`,
      stellarator symmetry, quadrature ordering, `gamma`, and `unitnormal`
      arrays.
- [ ] Field parity compares the vector field `B` at identical surface points
      before reducing to `B dot n` or `SquaredFlux`.
- [ ] Flux parity records the `SquaredFlux` definition:
  - [ ] `"quadratic flux"`
  - [ ] `"normalized"`
  - [ ] `"local"`
- [ ] `SquaredFluxJAX` definition support is checked in Phase 0 before any
      fixture pass claim. Current source accepts `"quadratic flux"`,
      `"normalized"`, and `"local"`, and current tests cover singular
      `"normalized"` / `"local"` behavior.
- [ ] Target-field parity records whether the target is zero, an explicit array,
      or `VirtualCasing.B_external_normal`.
- [ ] Target arrays must match surface shape exactly. Shape mismatch is a failed
      fixture contract, not a reshaping fallback.
- [ ] Composite objective parity compares raw component values before weights,
      weighted component values, total `J`, and the gradient contribution for
      each native-supported component.
- [ ] Gradient parity reports both absolute and relative differences plus the
      owning component and DOF name for the worst slice.
- [ ] Perturbation checks reuse deterministic directions and never draw random
      directions without recording the seed and vector hash.
- [ ] Boozer checks compare original residual vectors and label values before
      any solver behavior is interpreted.
- [ ] If a transformed/preconditioned solve is later introduced, the final
      original physical residual remains the acceptance quantity.

## Regression Review Gates

Upstream compatibility:

- [ ] Existing example scripts remain runnable as examples; the harness must not
      convert them into library modules with import-time side effects.
- [ ] Public CPU/C++/SciPy behavior remains the oracle and is not routed through
      JAX target internals.
- [ ] Existing tests for `BiotSavart`, `SquaredFlux`, curve objectives,
      `BoozerSurface`, and JAX wrappers stay in their current ownership lanes.

Downstream compatibility:

- [ ] JSON artifacts remain schema-versioned and include enough metadata for
      later manifest/report consumers.
- [ ] Touch `docs/jax_parity_manifest.md` only in commits that also carry the
      backing current-HEAD JSON artifact under `.artifacts/parity/`.
- [ ] Historical `.artifacts/parity/` files are not upgraded into current proof
      by documentation wording.

End-to-end compatibility:

- [ ] Fixture builder produces CPU and JAX lane objects from one immutable
      fixture spec.
- [ ] Fixture builder returns independent mutable field/objective instances per
      lane. The immutable spec can be shared; `BiotSavart`, `BiotSavartJAX`,
      `SquaredFlux`, and `SquaredFluxJAX` instances cannot be shared across
      lanes or threads.
- [ ] Evaluator emits JSON with lane outputs, comparisons, unsupported
      components, and failure details.
- [ ] Report reader can classify each fixture as `pass`, `partial`,
      `unsupported`, or `fail`.
- [ ] CI-safe tests cover at least the P0 fixture and one unsupported-component
      classification path.

Stale-code checks:

- [ ] Every referenced example path, source class, benchmark, test, and artifact
      path exists in the current tree at run time.
- [ ] Every documented command is executable from the repository root.
- [ ] Any removed or renamed example is removed from pass claims in the same
      change that removes or renames it.

## Implementation Plan

### Phase 0 - Baseline And Fixture Contract

- [ ] Record current `HEAD`, dirty-tree status, Python/JAX/SIMSOPT versions,
      and platform metadata before each parity run.
- [ ] Confirm all candidate source examples still exist in the current tree.
- [ ] Classify `QuadraticPenalty_over_sum_CurveLength_max` as
      unsupported-native-JAX unless an exact native implementation for
      `QuadraticPenalty(sum(CurveLength), "max")` lands in the same change.
- [ ] Confirm `SquaredFluxJAX` keyword coverage for `"quadratic flux"`,
      `"normalized"`, and `"local"` from current source/tests before enabling
      CWS local-flux pass claims.
- [ ] Define fixture IDs:
  - [ ] `minimal_stage2_flux_length_gap`
  - [ ] `cws_saved_local_flux`
  - [ ] `full_stage2_composite`
  - [ ] `planar_stage2_composite`
  - [ ] `position_orientation_flux_support_gate`
  - [ ] `boozer_surface_basic`
  - [ ] `boozer_qa_wrappers`
  - [ ] `finite_beta_target_flux`
  - [ ] `finitebuild_multifilament_support_gate`
  - [ ] `qfm_surface`
- [ ] Define the exact basis for every fixture:
  - [ ] CPU optimizer basis
  - [ ] JAX optimizer basis
  - [ ] mapping between bases, if different
  - [ ] fixed/free DOF mask provenance
- [ ] Define deterministic perturbation vectors:
  - [ ] seed `1` for example-aligned Taylor direction checks
  - [ ] at least one small random perturbation
  - [ ] at least one structured current-only or geometry-only perturbation
- [ ] Confirm the Default Tolerance Mapping applies as written, or record a
      justification for any stricter per-fixture bucket.
- [ ] Fail fixture-contract setup if `jax.config.jax_enable_x64` is false.
- [ ] Verify every JAX-lane field exposes `coil_dof_extraction_spec()` and every
      curve it depends on exposes `to_spec()` or another native immutable spec
      contract before constructing `BiotSavartJAX`.
- [ ] Verify fixture builders create independent `BiotSavart`/`BiotSavartJAX`
      and objective instances for each lane; only immutable specs may be shared.

Test runtime environment:

- [ ] Primary implementation/testing environment is `conda run -n jax`
      with a simsoptpp-backed editable install when CPU/C++ parity is required.
- [ ] `tests/integration/conftest.py` must either patch the scikit-build finder
      in candidate-fixed-style environments or report that no finder was
      detected; both outcomes are recorded in test output.
- [ ] Candidate-fixed integration runs are optional cross-environment checks
      until the harness proves `BiotSavartJAX` and the new benchmark imports are
      available there.

Acceptance gate:

- [ ] The harness can print fixture metadata without evaluating JAX.
- [ ] Unsupported components are explicitly listed, not skipped silently.
- [ ] No GPU devices are requested or required.
- [ ] Mixed CPU/JAX diagnostics cannot set `passed=true`.

### Phase 1 - P0 Minimal Stage-II Fixed-State Parity

Source example:

- `examples/1_Simple/stage_two_optimization_minimal.py`

Todos:

- [ ] Build a pure CPU fixture constructor for the exact initial state:
  - [ ] `SurfaceRZFourier.from_vmec_input(...)`
  - [ ] `create_equally_spaced_curves(...)`
  - [ ] `Current(1.0) * 1e5`
  - [ ] one fixed current
  - [ ] `coils_via_symmetries(...)`
  - [ ] `BiotSavart`
  - [ ] `SquaredFlux`
  - [ ] `CurveLength`
  - [ ] `QuadraticPenalty(sum(Jls), LENGTH_TARGET, "max")`
- [ ] Build the matching native JAX CPU flux lane from the same fixture spec:
  - [ ] `BiotSavartJAX`
  - [ ] `SquaredFluxJAX`
  - [ ] fixed surface captured at construction
  - [ ] fixed/free layout captured before objective construction
- [ ] For the length penalty, choose exactly one path:
  - [ ] Unsupported native-JAX component:
        `QuadraticPenalty_over_sum_CurveLength_max`.
  - [ ] Native target-bundle path only if the same change implements exact
        `QuadraticPenalty(sum(CurveLength), "max")` semantics in the declared
        basis.
      Do not pass the full objective through a CPU `CurveLength` substitute.
- [ ] Compare CPU/C++ and JAX CPU:
  - [ ] native-supported `J`
  - [ ] `SquaredFlux` component
  - [ ] native-supported length penalty component, or unsupported classification
  - [ ] native-supported gradient
  - [ ] `B`
  - [ ] `B dot n`
  - [ ] surface `gamma`
  - [ ] unit normals
- [ ] Run deterministic perturbation checks around the initial state.
- [ ] Add a small CI-safe test for this fixture.

Acceptance gate:

- [ ] P0 minimal fixed-state `SquaredFluxJAX` value and gradient parity passes
      on CPU.
- [ ] The length-penalty term is recorded in `unsupported_components` as
      `QuadraticPenalty_over_sum_CurveLength_max` unless exact native support
      has landed.
- [ ] Failure output identifies the first drifting component and gradient slice.

### Phase 2 - P1 Saved CWS Artifact Fixture

Source example:

- `examples/3_Advanced/curves_CWS_example.py`

Todos:

- [ ] Load both saved CWS artifact cases:
  - [ ] `optimization_cws_singlestage_nfp2_QA_ncoils3_axiTorus`
  - [ ] `optimization_cws_singlestage_nfp3_QA_ncoils4_axiTorus`
- [ ] Record hashes for:
  - [ ] saved `biot_savart_opt_maxmode*.json`
  - [ ] VMEC input file
  - [ ] surface `gamma`
  - [ ] unit normals
  - [ ] active coil/current DOFs
- [ ] Verify loaded coil curves expose native immutable specs before constructing
      `BiotSavartJAX`.
- [ ] Compare `BiotSavart` field values at the VMEC boundary.
- [ ] Compare `BdotN` mean/max and full flattened arrays.
- [ ] Compare `SquaredFlux(definition="local")` only after the Phase 0
      `SquaredFluxJAX` definition-coverage gate confirms `"local"` support in
      current source/tests.
- [ ] Keep VTK writing out of the parity harness.

Acceptance gate:

- [ ] Saved CWS artifacts reproduce CPU/C++ vs JAX CPU fixed-state field and
      local-flux parity without running an optimizer.

### Phase 3 - P1 Full Stage-II Composite Fixture

Source example:

- `examples/2_Intermediate/stage_two_optimization.py`

Todos:

- [ ] Reproduce the full initial objective:
  - [ ] `SquaredFlux`
  - [ ] `sum(CurveLength)`
  - [ ] `CurveCurveDistance`
  - [ ] `CurveSurfaceDistance`
  - [ ] `LpCurveCurvature`
  - [ ] `MeanSquaredCurvature`
  - [ ] `QuadraticPenalty(..., "max")`
- [ ] Classify each component:
  - [ ] directly JAX-supported
  - [ ] CPU-only for now
  - [ ] unsupported and excluded from pass claims
- [ ] Do not use CPU-only geometry penalties inside the JAX lane to pass the
      native composite objective.
- [ ] Compare component values before comparing total `J`.
- [ ] Compare gradient slices by component where supported.
- [ ] Add a short optimizer diagnostic only after fixed-state parity passes:
  - [ ] `maxiter=1`
  - [ ] final value envelope
  - [ ] no iterate-by-iterate oracle claim

Acceptance gate:

- [ ] Supported composite components pass fixed-state CPU/C++ vs JAX CPU parity.
- [ ] Unsupported components are reported as open gaps, not silently dropped.

### Phase 4 - P2 Planar-Coil Fixture

Source example:

- `examples/2_Intermediate/stage_two_optimization_planar_coils.py`

Todos:

- [ ] Reproduce `create_equally_spaced_planar_curves(...)`.
- [ ] Verify JAX curve specs cover the planar curve DOF contract.
- [ ] Compare flux and geometry-penalty components shared with Phase 3.
- [ ] Classify `LinkingNumber`:
  - [ ] supported in JAX parity lane
  - [ ] CPU-only diagnostic
  - [ ] unsupported gap
- [ ] Add focused failure messages for planar curve basis drift.

Acceptance gate:

- [ ] Planar flux and supported geometry components match CPU/C++ at fixed
      state.

### Phase 5 - Position/Orientation Support Gate

Source example:

- `examples/1_Simple/optimize_coil_position_orientation.py`

Current source-backed finding:

- `create_equally_spaced_oriented_curves(...)` returns
  `OrientedCurveXYZFourier`.
- `BiotSavartJAX` requires curves with immutable native specs.
- `OrientedCurveXYZFourier` currently does not expose `to_spec()`, so this
  example is a support-gap fixture, not a P0 native-JAX parity fixture.

Todos:

- [ ] Build the CPU TF plus windowpane coil fixture without optimizer execution.
- [ ] Preserve active DOFs:
  - [ ] TF geometry fixed
  - [ ] windowpane curve geometry fixed except `x0`, `y0`, `z0`
  - [ ] windowpane orientation DOFs `yaw`, `pitch`, `roll` free
  - [ ] TF/windowpane currents free except the fixed seed current
- [ ] Emit a precise unsupported-native-JAX result if oriented curves still lack
      immutable native specs.
- [ ] If `OrientedCurveXYZFourier.to_spec()` or an equivalent immutable spec is
      implemented later, add native parity checks:
  - [ ] active DOF names and masks
  - [ ] current-only perturbation
  - [ ] position-only perturbation
  - [ ] orientation-only perturbation
  - [ ] `SquaredFlux` value and gradient

Acceptance gate:

- [ ] The fixture cannot pass native JAX parity until oriented-curve immutable
      spec support exists and is covered by tests.

### Phase 6 - P2 Boozer Surface And QA Wrapper Fixtures

Source examples:

- `examples/2_Intermediate/boozer.py`
- `examples/2_Intermediate/boozerQA.py`

Todos:

- [x] Start with fixed-state residual checks before full solves:
  - [x] `boozer_surface_residual`
  - [x] iota/G inputs
  - [x] label value
  - [x] residual norm
- [x] Add label fixtures:
  - [x] `Area`
  - [x] `ToroidalFlux`
  - [x] `Volume`
- [ ] For `boozer.py`, compare:
  - [x] pre-solve residual vector
  - [ ] post-LBFGS contract fields as diagnostics only
  - [ ] post-Levenberg-Marquardt final residual envelope
- [x] For `boozerQA.py`, compare solved-state scalar outputs:
  - [x] `NonQuasiSymmetricRatio`
  - [x] `Iotas`
  - [x] `MajorRadius`
  - [x] length penalty classified as unsupported native JAX component
- [ ] Keep solver path differences separate from public wrapper value/gradient parity.

Acceptance gate:

- [x] Boozer fixed-state residual and copied-solved-state scalar outputs agree
      where the harness claims support. Public `BoozerSurfaceJAX`
      wrapper/adjoint parity is not claimed by this fixture.
- [ ] Solver trajectory drift is reported as integration behavior, not as the
      first oracle.

### Phase 7 - Lower-Priority Expansion

Finite-beta target flux:

- [ ] Add `examples/2_Intermediate/stage_two_optimization_finite_beta.py` only
      after zero-target flux fixtures pass.
- [ ] Record `VirtualCasing` provenance and target-array hash.
- [ ] Compare `SquaredFlux(target=vc.B_external_normal)`.

Finite-build multifilament:

- [ ] Add `examples/3_Advanced/stage_two_optimization_finitebuild.py` as a
      support-gate fixture.
- [ ] Record multifilament grid parameters and base-curve/current hashes.
- [ ] Verify every generated filament curve exposes a native immutable spec
      before any JAX pass claim.
- [ ] Classify unsupported finite-build geometry as a native-spec gap, not a
      parity pass.

QFM:

- [ ] Add `examples/1_Simple/qfm.py` after Boozer/surface-objective support is
      stable.
- [ ] Compare `QfmResidual` at fixed state before running constrained surface
      optimizers.

Stochastic:

- [ ] Treat `examples/2_Intermediate/stage_two_optimization_stochastic.py` as a
      deterministic stress diagnostic only.
- [ ] Freeze seed, sample count, and perturbed coil states.
- [ ] Use reduced sample counts for CI.

Coil forces:

- [ ] Keep `examples/3_Advanced/coil_forces.py` as CPU regression backlog until
      force and energy JAX wrappers exist.

### Phase 8 - Reporting And Documentation

- [ ] Add a per-fixture summary table:
  - [ ] pass/fail
  - [ ] unsupported components
  - [ ] max value absolute/relative diff
  - [ ] max gradient absolute/relative diff
  - [ ] first failing component
  - [ ] first failing gradient slice
- [ ] Add exact commands used for CPU-only runs.
- [ ] Write JSON artifacts under `.artifacts/parity/` with a date-stamped
      directory.
- [ ] Touch `docs/jax_parity_manifest.md` only in commits that also carry the
      backing current-HEAD JSON artifact under `.artifacts/parity/`.
- [ ] Add a review-results subsection to this document or a follow-up report
      listing plan bugs found, fixes applied, and residual unsupported surfaces.
- [ ] Keep historical artifacts labeled as historical; do not use them as
      current-HEAD proof.

Acceptance gate:

- [ ] A reader can determine which example surfaces are proven, partial, or
      unsupported from one current-HEAD report.

## Test Plan

Mandatory parity environment:

```bash
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
```

Recommended CPU diagnostic environment:

```bash
export JAX_TRACEBACK_FILTERING=off
export JAX_DEBUG_NANS=1
```

Initial CPU-only command set, to be refined during implementation:

```bash
conda run -n jax python -m pytest -q \
  -m "not private_optimizer_runtime" \
  tests/integration/test_non_banana_example_cpp_jax_cpu_parity.py
```

```bash
conda run -n jax python benchmarks/non_banana_example_cpp_jax_cpu_parity.py \
  --fixtures minimal_stage2_flux_length_gap,cws_saved_local_flux \
  --git-sha "$(git rev-parse HEAD)" \
  --dirty-policy record \
  --output-json .artifacts/parity/20260512-non-banana-examples/cpu-jax-cpu.json
```

Extended local command set:

```bash
conda run -n jax python benchmarks/non_banana_example_cpp_jax_cpu_parity.py \
  --fixtures all-supported \
  --git-sha "$(git rev-parse HEAD)" \
  --dirty-policy record \
  --output-json .artifacts/parity/20260512-non-banana-examples/all-supported-cpu.json
```

## Definition Of Done

- [ ] P0 fixed-state fixtures pass CPU/C++ vs JAX CPU value and gradient parity
      for native-supported quantities.
- [ ] The minimal Stage-II fixture's exact length penalty is either implemented
      natively or explicitly reported as
      `QuadraticPenalty_over_sum_CurveLength_max` in `unsupported_components`.
- [ ] P1 saved-artifact and full-composite fixtures either pass or report exact
      unsupported components.
- [ ] P2 Boozer/planar fixtures have fixed-state coverage before any optimizer
      trajectory claims.
- [ ] Position/orientation and finite-build fixtures are either covered by native
      immutable specs or reported as support gaps.
- [ ] All JSON artifacts include current git SHA and dirty-tree metadata.
- [ ] All pass claims state the exact fixture, quantity, tolerance bucket, and
      max difference.
- [ ] All timed JAX measurements synchronize with `block_until_ready`.
- [ ] No GPU proof is claimed from this CPU-only plan.
- [ ] No banana Stage 2 or banana single-stage proof is mixed into this
      non-banana example plan.
