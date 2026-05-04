# Boozer Full Parity Plan

Status: draft implementation plan as of 2026-05-04.

Current tree inspected at `bee115caedfb`. The working tree was dirty when this
plan was written; this document is intentionally additive and does not judge
uncommitted implementation changes outside the Boozer parity surface.

## Goal

Close the BoozerSurfaceJAX parity contract against legacy BoozerSurface for:

- math kernels
- solver results
- guard behavior
- derivatives and adjoints
- supported public APIs

The explicit exclusion is legacy mutable object identity semantics. JAX does
not need to match CPU pointer identity, cached result object identity, or dirty
flag internals. JAX must instead prove equivalent immutable runtime-state
behavior.

## Scope

In scope:

- `tests/geo/test_boozersurface.py`
- `tests/geo/test_boozersurface_jax.py`
- `tests/geo/test_boozersurface_jax_private.py`
- `tests/geo/test_boozer_derivatives_jax.py`
- `tests/geo/test_boozer_residual_jax.py`
- `tests/geo/boozersurface_jax_test_helpers.py`
- `tests/integration/test_single_stage_jax_cpu_reference.py`
- `src/simsopt/geo/boozersurface_jax.py`
- `src/simsopt/geo/boozer_residual_jax.py`
- `src/simsopt/geo/surfaceobjectives_jax.py`
- `src/simsopt/geo/_boozersurface_current_guard.py`

Dependency and audit-only surfaces:

- `src/simsopt/geo/_simsoptpp_boozer_compat.py`: legacy simsoptpp
  compatibility shim. Audit it explicitly as CPU/C++ compatibility or
  `unsupported_jax_contract`; do not add a JAX fallback through it.
- `src/simsopt/geo/optimizer_jax.py`: owns `jax_minimize`,
  `newton_polish`, and `newton_exact` solver behavior consumed by P3/P5.
- `src/simsopt/geo/label_constraints_jax.py`: owns `volume_jax`,
  `area_jax`, and `compute_G_from_currents` for label parity paths.
- `tests/geo/conftest.py`: checked on 2026-05-04; it only restores
  `sys.modules` state after geo tests and is not a Boozer fixture owner.

Out of scope:

- GPU performance tuning.
- Full single-stage outer optimizer trajectory parity.
- Requiring SciPy Wolfe-internal step identity.
- Emulating legacy mutable object identity in JAX.
- Adding defensive fallbacks around unsupported compatibility paths.

## Contract Terms

`strict_parity` means CPU BoozerSurface and JAX BoozerSurfaceJAX are evaluated
from the same fixture state and must match the same public behavior or numeric
quantity within the named tolerance lane.

`jax_native_equivalent` means the CPU behavior is inherently mutable, but the
supported product guarantee has a JAX-native equivalent over immutable runtime
state.

`intentional_exclusion` means the legacy behavior is object identity or mutable
cache identity only. It is documented and not implemented in JAX.

`unsupported_jax_contract` means the legacy CPU path exists, but the JAX backend
must reject it explicitly because it conflicts with the immutable target-lane
contract.

Canonical tolerance lane names are the snake_case keys from
`benchmarks/validation_ladder_contract.py::PARITY_LADDER_TOLERANCES`. Prose may
use hyphenated English, but matrix entries and acceptance gates must use these
exact keys:

- `direct_kernel`
- `ls_wrapper_gradient`
- `derivative_heavy`
- `exact_well_conditioned_adjoint`
- `exact_ill_conditioned_adjoint`
- `branch_stable_resolve`
- `fd_gradient`
- `gpu_runtime`
- `reduction_cpu_gpu`

## Current Legacy Baseline

Legacy `tests/geo/test_boozersurface.py` currently collects 23 top-level
BoozerSurface tests. The parity closure target is not "copy the file." The
target is to classify each legacy contract and ensure every non-mutable
contract has a strict JAX proof or a deliberate unsupported-JAX proof.

| Legacy test | Current classification | Required action |
| --- | --- | --- |
| `test_call_boozer_residual_falls_back_to_alpha_only_signature` | CPU/C++ compatibility only | Keep CPU-only. Add no JAX fallback. Document as `unsupported_jax_contract` if referenced. |
| `test_call_boozer_residual_ds_falls_back_to_alpha_only_signature` | CPU/C++ compatibility only | Keep CPU-only. Add no JAX fallback. Document as `unsupported_jax_contract` if referenced. |
| `test_call_boozer_residual_ds2_falls_back_to_alpha_only_signature` | CPU/C++ compatibility only | Keep CPU-only. Add no JAX fallback. Document as `unsupported_jax_contract` if referenced. |
| `test_solver_signatures_do_not_expose_vectorize` | public API guard | Add JAX signature guard for supported public methods. |
| `test_constructor_rejects_spoof_surface_names` | guard | Add strict JAX constructor/type rejection if not already complete. |
| `test_duplicate_import_surface_helpers_accept_canonical_surface_shapes` | CPU helper compatibility | Keep CPU-only unless JAX exposes the same helper. Do not add hidden import fallback. |
| `test_run_code_rejects_G_none_with_free_currents` | guard | Add strict JAX public-path parity or explicit unsupported path proof. |
| `test_run_code_rejects_G_none_with_free_legacy_currents` | guard | Already has JAX coverage; verify exact same free-current/fixed-current contract. |
| `test_none_G_coil_gradient_callback_rejects_free_currents` | guard/derivative | Add strict JAX adjoint callback guard parity. |
| `test_none_G_coil_gradient_callback_allows_explicit_G` | guard/derivative | Add strict JAX adjoint callback allow-path parity. |
| `test_residual` | math | Add same NCSX exact-surface JAX scalar residual proof. |
| `test_boozer_penalty_constraints_gradient` | derivative | Expand CPU/JAX parity matrix over surface type, stellsym, and optimize_G. |
| `test_boozer_penalty_constraints_hessian` | derivative | Add full matrix parity or documented FD/Taylor equivalence lane. |
| `test_boozer_constrained_jacobian` | derivative | Add exact constrained residual/Jacobian CPU/JAX parity matrix. |
| `test_boozer_surface_optimisation_convergence` | solver result | Add solver-result parity over the supported LS/exact combinations. |
| `test_boozer_serialization` | mutable identity | Exclude pointer identity; add immutable runtime-state round-trip replacement. |
| `test_run_code` | mixed solver/mutable | Split: solver parts require parity; "second call does not run" stays JAX-native idempotence only. |
| `test_minimize_boozer_penalty_constraints_ls_manual` | solver result | Manual LS is supported in JAX; extend existing manual API coverage to CPU/JAX public result parity. |
| `test_need_to_run_code_false` | mutable identity | Exclude `res1 is res2`; add immutable same-input same-result replacement. |
| `test_minimize_boozer_exact_constraints_newton_G_None` | solver result | Add JAX exact G=None result contract. |
| `test_minimize_boozer_exact_constraints_newton_stellsym_false` | solver result | Add non-stellsym exact JAX result parity. |
| `test_boozer_surface_quadpoints` | guard/math setup | Existing JAX coverage should become strict one-for-one parity. |
| `test_boozer_surface_type_assert` | guard | Add explicit JAX accepted/rejected surface family matrix. |

## Mutable Identity Policy

Mutable identity tests are not copied into BoozerSurfaceJAX. They are replaced
by immutable state tests.

### Serialization

Legacy behavior:

- `BoozerSurface` is serialized through SIMSON/GSON.
- The regenerated object preserves mutable object graph identity:
  `bs_regen.label.surface is bs_regen.surface`.

JAX contract:

- [ ] Do not require `label.surface is surface`.
- [ ] Do not add a host object graph fallback to reconstruct mutable identity.
- [ ] Add a JAX runtime-state round-trip test for:
  - surface DOFs
  - quadpoints
  - `stellsym`, `nfp`, `mpol`, `ntor`
  - label type and target value
  - coil set spec metadata
  - solved runtime state when a solve has been run
- [ ] The round-trip assertion compares arrays and structured metadata, not
  Python object identity.

Implementation notes:

- Put the replacement in `tests/geo/test_boozersurface_jax.py`.
- Prefer existing runtime-state APIs:
  - `BoozerSurfaceJAX.coil_set_spec`
  - `BoozerSurfaceJAX.get_solved_runtime_state()`
  - `BoozerSurfaceJAX.get_adjoint_runtime_state()`
- If object-level JSON is not a supported JAX public API, do not add it just for
  this test. Test the supported immutable runtime state instead.

### Cached Result Identity

Legacy behavior:

- `need_to_run_code=False` makes the second call return the exact same result
  object.
- The legacy test literally asserts `res1 is res2`; this is an object-identity
  assertion that piggybacks on cache reuse.

JAX contract:

- [ ] Do not require `res1 is res2`.
- [ ] Do not make the target lane preserve a mutable result object for parity.
- [ ] Add same-input same-result coverage for:
  - same `coil_set_spec`
  - same `sdofs`
  - same `iota`
  - same `G`
  - same options
- [ ] Assert numeric equality of result arrays and scalar metadata.
- [ ] Assert stable compiled/runtime callable reuse where the current code
  supports it.
- [ ] Assert callable rebuild when target label, runtime spec, or options
  change.

Implementation notes:

- Existing callable reuse tests already live near
  `test_run_code_traceable_exact_reuses_stable_residual_callable` and
  `test_run_code_traceable_lm_reuses_stable_residual_and_objective_callables`.
- Extend these into the parity matrix rather than adding a host `res` cache
  compatibility layer.

### Dirty Flag Behavior

Legacy behavior:

- `BoozerSurface.need_to_run_code` controls whether `run_code` recomputes or
  returns cached state.

JAX contract:

- [ ] Keep public idempotence behavior where it already exists.
- [ ] Do not use the dirty flag as the parity oracle.
- [ ] The parity oracle is explicit runtime input state plus explicit result
  state.
- [ ] Traceable objective calls must not mutate `need_to_run_code`.
- [ ] Existing mutable wrapper paths can remain for public compatibility, but
  new target-lane proofs must enter through explicit runtime state or traceable
  APIs. They must not use hidden host-wrapper re-entry to satisfy target-lane
  correctness.

Implementation notes:

- Keep `test_run_code_idempotent` as wrapper compatibility.
- Add a separate immutable traceable-state test that proves the same solve can
  be reproduced without reading or mutating wrapper dirty state.

## Implementation Workstreams

### P0: Boozer Parity Matrix SSOT

Goal: make the full test contract auditable before editing solver code.

- [x] Add `docs/boozer_full_parity_plan_2026-05-04.md` as the human-readable
  plan.
- [x] Link this plan from `docs/jax_parity_manifest.md`.
- [ ] Add or update a test-side matrix, preferably in a lightweight helper such
  as `tests/geo/boozer_legacy_parity_contract.py`.
- [ ] Record every legacy test name from `tests/geo/test_boozersurface.py`.
- [ ] For each legacy test, record:
  - category: `strict_parity`, `jax_native_equivalent`,
    `intentional_exclusion`, or `unsupported_jax_contract`
  - owning JAX test file
  - owning JAX test function or planned function
  - tolerance lane, when numeric
  - whether `simsoptpp` is required
  - whether CUDA is required
- [ ] Add a non-runtime contract test that fails when a legacy BoozerSurface test
  appears without a matrix entry.

Acceptance:

- [ ] A reader can answer "which JAX test covers this legacy Boozer test?"
  without searching the repo manually.
- [ ] Mutable exclusions are explicit and narrow.
- [ ] No runtime fallback behavior is added.

### P1: Shared Fixture Layer

Goal: remove accidental fixture drift between CPU and JAX parity tests.

- [ ] Centralize NCSX Boozer parity fixture builders in
  `tests/geo/boozersurface_jax_test_helpers.py`.
- [ ] Provide fixture constructors for:
  - `SurfaceXYZFourier`
  - `SurfaceXYZTensorFourier`
  - `stellsym=True`
  - `stellsym=False`
  - `optimize_G=True`
  - `optimize_G=False`
  - fixed-current `G=None`
  - explicit `G`
- [ ] Ensure CPU and JAX fixtures use:
  - copied DOF arrays from the same source
  - identical quadrature points
  - identical current values
  - identical target labels
  - identical `weight_inv_modB`
  - identical `constraint_weight`
- [ ] Add one helper that returns a pair:
  - CPU `BoozerSurface`
  - JAX `BoozerSurfaceJAX`
- [ ] Add one helper that returns immutable inputs:
  - `coil_set_spec`
  - `sdofs`
  - `iota`
  - `G`
  - options snapshot

Acceptance:

- [ ] Tests no longer duplicate G/current formulas in multiple places.
- [ ] Any CPU/JAX mismatch can be traced to implementation behavior, not
  fixture construction drift.

### P2: Math Kernel Parity

Goal: CPU and JAX evaluate the same Boozer math at the same state.

Legacy anchors:

- `test_residual`
- `test_boozer_penalty_constraints_gradient`
- `test_boozer_penalty_constraints_hessian`
- `test_boozer_constrained_jacobian`

Existing JAX anchors:

- `TestBoozerResidualCPUParity`
- `tests/geo/test_boozer_residual_jax.py`
- `test_full_penalty_objective_parity`
- `test_penalty_value_and_gradient_cpu_parity_tensor_matrix`
- `tests/geo/test_boozer_derivatives_jax.py`

Tasks:

- [ ] Add NCSX exact-surface residual parity for the legacy
  `test_residual` state:
  - same `get_exact_surface()`
  - same `tf_target=0.41431152`
  - same `iota=-0.44856192`
  - same `weight=1.0`
  - compare CPU scalar objective against JAX scalar objective
- [ ] Expand penalty value/gradient parity over:
  - `SurfaceXYZFourier`
  - `SurfaceXYZTensorFourier`
  - `stellsym=True`
  - `stellsym=False`
  - `optimize_G=True`
  - `optimize_G=False`
- [ ] Add Hessian parity or an explicitly named Hessian-equivalence lane:
  - direct CPU Hessian vs JAX Hessian where both are available:
    `derivative_heavy` with `second_derivative_rtol=1e-6` and
    `second_derivative_atol=1e-8`
  - otherwise CPU/JAX directional second derivative with the same `h1`, `h2`:
    `fd_gradient` with `directional_fd_rtol=1e-5` and
    `directional_fd_atol=1e-7`
  - no claim of direct Hessian parity if only FD/Taylor evidence exists
- [ ] Add exact constrained residual/Jacobian parity over the same matrix:
  - compare residual vector
  - compare Jacobian-vector product for seeded direction `h`
  - compare dense Jacobian only for small fixtures where materialization is
    supported
- [ ] Use `parity_ladder_tolerances` for numeric gates instead of ad hoc
  tolerances. Matrix entries must name the canonical snake_case lane.

Acceptance:

- [ ] Same-state residual parity passes.
- [ ] Same-state penalty value/gradient parity passes for the full matrix.
- [ ] Hessian coverage is either direct parity or clearly labeled
  `fd_gradient` FD/Taylor-equivalence.
- [ ] Exact constrained Jacobian coverage is no longer mock-only.

### P3: Solver Result Parity

Goal: CPU and JAX solve to equivalent public results for supported solver
paths.

Legacy anchors:

- `test_boozer_surface_optimisation_convergence`
- `test_run_code`
- `test_minimize_boozer_penalty_constraints_ls_manual`
- `test_minimize_boozer_exact_constraints_newton_G_None`
- `test_minimize_boozer_exact_constraints_newton_stellsym_false`

Existing JAX anchors:

- `TestRunCodeLSParity`
- `test_exact_solve_parity`
- `test_public_manual_ls_api_supports_baseline_demo_sequence`
- `test_public_manual_ls_api_increases_damping_after_worsening_trial`
- real fixture on-device parity tests in
  `tests/integration/test_single_stage_jax_cpu_reference.py`

Tasks:

- [ ] Split legacy `test_run_code` into:
  - solve result parity
  - mutable idempotence exclusion
  - fixed-current `G=None` guard
  - exact solve without explicit `G`
- [ ] Add supported LS/LBFGS result parity:
  - CPU `BoozerSurface.run_code`
  - JAX `BoozerSurfaceJAX.run_code`
  - same initial surface
  - same `iota`
  - same `G`
  - same solver options
- [ ] Compare public result fields:
  - `success`
  - `type`
  - `iota`
  - `G`
  - residual norm
  - label value and label error
  - final surface DOFs when representation is the same
- [ ] Add exact Newton `G=None` JAX result contract:
  - if supported, compare final `G is None`, residual norm, and success/failure
    class
  - if unsupported, assert a strict unsupported-JAX error
- [ ] Add non-stellsym exact result parity:
  - use Area label as the legacy test does
  - route the JAX label through `area_jax` in
    `src/simsopt/geo/label_constraints_jax.py`
  - assert `lm` length and residual quality
  - compare CPU/JAX final public metrics
- [ ] Extend existing manual LS support to CPU/JAX result parity:
  - keep `method="manual"` as a supported JAX public API path
  - use the existing manual API tests as anchors
  - compare public result fields against the matching CPU manual LS case

Acceptance:

- [ ] Supported solver paths match in public result semantics.
- [ ] Unsupported solver paths fail early and explicitly.
- [ ] No test requires identical iteration trace or SciPy line-search internals.
- [ ] Existing mutable wrappers may stay, but target-lane solver proofs do not
  depend on hidden host-wrapper re-entry.

### P4: Guard Parity

Goal: CPU and JAX enforce the same safety contracts at public boundaries.

Legacy anchors:

- `test_solver_signatures_do_not_expose_vectorize`
- `test_constructor_rejects_spoof_surface_names`
- `test_run_code_rejects_G_none_with_free_currents`
- `test_run_code_rejects_G_none_with_free_legacy_currents`
- `test_none_G_coil_gradient_callback_rejects_free_currents`
- `test_none_G_coil_gradient_callback_allows_explicit_G`
- `test_boozer_surface_quadpoints`
- `test_boozer_surface_type_assert`

Tasks:

- [ ] Verify public method signatures do not expose unsupported legacy knobs.
- [ ] Ensure `G=None` with free currents rejects through the shared current
  guard.
- [ ] Ensure `G=None` with fixed currents is allowed.
- [ ] Ensure adjoint/gradient callback paths apply the same fixed/free current
  logic.
- [ ] Ensure spoofed surface names fail by real type/contract, not by string
  matching.
- [ ] Ensure unsupported surfaces fail explicitly.
- [ ] Keep the JAX accepted surface matrix clear:
  - accepted: supported XYZ Fourier/Tensor runtime surfaces
  - rejected: unsupported host-only surfaces unless a native JAX contract exists
- [ ] Keep quadpoint/stellsym mask coverage one-for-one with legacy cases.

Acceptance:

- [ ] Same public guard behavior on CPU and JAX where JAX supports the path.
- [ ] Explicit unsupported-JAX errors where JAX does not support the path.
- [ ] No hidden `_coils` fallback or dynamic compatibility route is introduced.

### P5: Derivative And Adjoint Parity

Goal: derivative correctness is proven at the same contract level as values.

Tasks:

- [ ] Add direct CPU/JAX wrapper gradient parity for supported labels:
  - `IotasJAX`
  - `NonQuasiSymmetricRatioJAX`
  - `BoozerResidualJAX`
- [ ] Add or expand finite-difference checks over the shared NCSX fixtures.
- [ ] Add batched RHS adjoint parity for matrix cotangents.
- [ ] Keep exact ill-conditioned adjoints in
  `exact_ill_conditioned_adjoint`: residual/failure-class shape only,
  `residual_rel_tol=1e-10`, and no vector parity requirement.
- [ ] Keep exact well-conditioned adjoints in
  `exact_well_conditioned_adjoint`: vector parity required with
  `adjoint_rtol=1e-6`, `adjoint_atol=1e-8`, `gradient_rtol=1e-6`, and
  `gradient_atol=1e-8`.
- [ ] Use `derivative_heavy` for direct CPU/JAX derivative matrices and
  `fd_gradient` for directional FD/Taylor evidence.
- [ ] For every derivative test, state whether the proof is:
  - direct CPU/JAX value/gradient parity
  - CPU finite difference vs JAX analytic
  - JAX Taylor/FD self-consistency
  - residual/failure-only coverage

Acceptance:

- [ ] Derivative tests are not just smoke tests.
- [ ] Every derivative parity claim names its oracle.
- [ ] Every derivative parity claim names its canonical tolerance lane.
- [ ] Ill-conditioned carve-outs cannot be mistaken for vector parity.

### P6: Supported API Parity

Goal: all supported public APIs have a clear parity or JAX-native contract.

Supported API surface:

- `run_code`
- `minimize_boozer_penalty_constraints_LBFGS`
- `minimize_boozer_penalty_constraints_ls` for supported methods
- `solve_residual_equation_exactly_newton`
- `get_solved_runtime_state`
- `get_adjoint_runtime_state`
- `run_code_traceable`
- `run_code_functional`

Tasks:

- [ ] For each API, document whether it is:
  - CPU/JAX parity
  - JAX-native equivalent
  - unsupported in JAX
- [ ] Assert public result schema keys for parity APIs.
- [ ] Assert runtime-state schema keys for JAX-native APIs.
- [ ] Keep `run_code_traceable` and `run_code_functional` explicit-state APIs.
- [ ] Do not route target-lane correctness through mutable wrapper re-entry.

Acceptance:

- [ ] Public APIs either match CPU behavior or reject unsupported modes.
- [ ] JAX-native APIs have explicit immutable state tests.
- [ ] There are no silent fallbacks to legacy mutable host behavior.

## Final Validation Ladder

### Current Pass/Fail Watermark

Baseline captured on 2026-05-04 at `bee115caedfb` with
`JAX_ENABLE_X64=True JAX_PLATFORM_NAME=cpu`.

Collection baseline:

- `pytest --collect-only -q tests/geo/test_boozersurface.py`: 23 tests
  collected.
- `pytest --collect-only -q tests/geo/test_boozersurface_jax.py`: 301 tests
  collected.
- `pytest --collect-only -q tests/geo/test_boozersurface_jax_private.py
  tests/geo/test_boozer_derivatives_jax.py tests/geo/test_boozer_residual_jax.py
  tests/integration/test_single_stage_jax_cpu_reference.py -k "boozer or
  Boozer"`: 155 selected, 157 deselected, 312 total.

The collection bundles and execution bundles do not partition the same files.
The third collection bullet groups private, derivative, residual, and
integration Boozer selections by `-k`, while the execution baseline exercises
residual and derivative tests in the first bundle and private plus integration
tests in the third bundle.

Execution baseline:

- `pytest -q tests/geo/test_boozersurface.py tests/geo/test_boozer_residual_jax.py
  tests/geo/test_boozer_derivatives_jax.py`: 59 passed, 14 skipped, 56 subtests
  passed in 312.83 s.
- `pytest -q tests/geo/test_boozersurface_jax.py`: 297 passed, 4 skipped in
  118.63 s.
- `pytest -q tests/geo/test_boozersurface_jax_private.py
  tests/integration/test_single_stage_jax_cpu_reference.py -k "boozer or
  Boozer"`: 103 passed, 2 failed, 157 deselected in 339.24 s.

Current failing integration tests:

- `tests/integration/test_single_stage_jax_cpu_reference.py::TestBoozerResidualGradientFD::test_end_to_end_dJ_vs_fd`
- `tests/integration/test_single_stage_jax_cpu_reference.py::TestBoozerResidualAdjointFD::test_boozer_residual_resolve_fd`

The Definition of Done cannot claim CPU validation closure until these failures
are fixed or deliberately reclassified in the parity matrix with a named
contract lane.

Run these after the implementation tasks land:

```bash
JAX_ENABLE_X64=True JAX_PLATFORM_NAME=cpu pytest -q tests/geo/test_boozersurface.py
JAX_ENABLE_X64=True JAX_PLATFORM_NAME=cpu pytest -q tests/geo/test_boozersurface_jax.py
JAX_ENABLE_X64=True JAX_PLATFORM_NAME=cpu pytest -q tests/geo/test_boozersurface_jax_private.py
JAX_ENABLE_X64=True JAX_PLATFORM_NAME=cpu pytest -q tests/geo/test_boozer_residual_jax.py
JAX_ENABLE_X64=True JAX_PLATFORM_NAME=cpu pytest -q tests/geo/test_boozer_derivatives_jax.py
JAX_ENABLE_X64=True JAX_PLATFORM_NAME=cpu pytest -q tests/integration/test_single_stage_jax_cpu_reference.py -k "boozer or Boozer"
```

Optional CUDA proof after CPU closure:

```bash
SIMSOPT_JAX_PLATFORM=cuda JAX_ENABLE_X64=True JAX_PLATFORM_NAME=cuda pytest -q tests/geo/test_boozersurface_jax.py -k "gpu or parity"
SIMSOPT_JAX_PLATFORM=cuda JAX_ENABLE_X64=True JAX_PLATFORM_NAME=cuda pytest -q tests/integration/test_single_stage_jax_cpu_reference.py -k "boozer and gpu"
```

## Definition Of Done

- [ ] Every legacy BoozerSurface test has a matrix entry.
- [ ] Every non-mutable legacy math test has strict CPU/JAX parity or an
  explicitly named equivalence lane.
- [ ] Every supported solver path has CPU/JAX public result parity.
- [ ] Every unsupported solver path has a strict unsupported-JAX test.
- [ ] Every guard path either matches CPU behavior or rejects by explicit JAX
  contract.
- [ ] Every derivative claim states its oracle and tolerance lane.
- [ ] Mutable identity behavior is limited to the three documented exclusions:
  - serialization pointer identity
  - cached `res` object identity
  - dirty-flag identity behavior
- [ ] JAX replacement tests prove immutable runtime-state equivalence.
- [x] `docs/jax_parity_manifest.md` links to this plan.
- [ ] CPU validation ladder passes relative to the current pass/fail watermark,
  with no unexplained failures.
- [ ] CUDA validation ladder passes if GPU parity is claimed.

## Non-Negotiables

- [ ] Do not loosen tolerances to hide drift.
- [ ] Do not add defensive fallback paths.
- [ ] Do not add hidden host-wrapper re-entry to satisfy target-lane tests.
- [ ] Do not claim full solver trajectory parity from final metric closeness.
- [ ] Do not claim direct Hessian or adjoint vector parity from FD-only evidence.
- [ ] Do not use mutable object identity as the JAX parity oracle.
