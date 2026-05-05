# Boozer Full Parity Plan

Status: CPU implementation closure as of 2026-05-05.

Implementation note: closure landed in `73d85b88e` and the follow-up stale
Boozer backend-policy cleanup landed in `5011b56bf`. This remains a CPU
closure: no CUDA Boozer parity claim is made until the optional CUDA validation
ladder below is run on hardware.

Current tree initially inspected at `bee115caedfb`. The working tree was dirty
when this plan was written; this document is intentionally additive and does
not judge uncommitted implementation changes outside the Boozer parity surface.

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
| `test_boozer_penalty_constraints_hessian` | derivative | Column-complete CPU/C++ Hessian oracle parity via `test_penalty_hessian_column_complete_cpu_parity_matrix`; retain seeded directional HVP coverage. |
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

- [x] Do not require `label.surface is surface`.
- [x] Do not add a host object graph fallback to reconstruct mutable identity.
- [x] Add a JAX runtime-state round-trip test for:
  - surface DOFs
  - quadpoints
  - `stellsym`, `nfp`, `mpol`, `ntor`
  - label type and target value
  - coil set spec metadata
  - solved runtime state when a solve has been run
- [x] The round-trip assertion compares arrays and structured metadata, not
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

- [x] Do not require `res1 is res2`.
- [x] Do not make the target lane preserve a mutable result object for parity.
- [x] Add same-input same-result coverage for:
  - same `coil_set_spec`
  - same `sdofs`
  - same `iota`
  - same `G`
  - same options
- [x] Assert numeric equality of result arrays and scalar metadata.
- [x] Assert stable compiled/runtime callable reuse where the current code
  supports it.
- [x] Assert callable rebuild when target label, runtime spec, or options
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

- [x] Keep public idempotence behavior where it already exists.
- [x] Do not use the dirty flag as the parity oracle.
- [x] The parity oracle is explicit runtime input state plus explicit result
  state.
- [x] Traceable objective calls must not mutate `need_to_run_code`.
- [x] Existing mutable wrapper paths can remain for public compatibility, but
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
- [x] Add or update a test-side matrix, preferably in a lightweight helper such
  as `tests/geo/boozer_legacy_parity_contract.py`.
- [x] Record every legacy test name from `tests/geo/test_boozersurface.py`.
- [x] For each legacy test, record:
  - category: `strict_parity`, `jax_native_equivalent`,
    `intentional_exclusion`, or `unsupported_jax_contract`
  - owning JAX test file
  - owning JAX test function or planned function
  - tolerance lane, when numeric
  - whether `simsoptpp` is required
  - whether CUDA is required
- [x] Add a non-runtime contract test that fails when a legacy BoozerSurface test
  appears without a matrix entry.

Acceptance:

- [x] A reader can answer "which JAX test covers this legacy Boozer test?"
  without searching the repo manually.
- [x] Mutable exclusions are explicit and narrow.
- [x] No runtime fallback behavior is added.

### P1: Shared Fixture Layer

Goal: remove accidental fixture drift between CPU and JAX parity tests.

- [x] Centralize NCSX Boozer parity fixture builders in
  `tests/geo/boozersurface_jax_test_helpers.py`.
- [x] Provide fixture constructors for:
  - `SurfaceXYZFourier`
  - `SurfaceXYZTensorFourier`
  - `stellsym=True`
  - `stellsym=False`
  - `optimize_G=True`
  - `optimize_G=False`
  - fixed-current `G=None`
  - explicit `G`
- [x] Ensure CPU and JAX fixtures use:
  - copied DOF arrays from the same source
  - identical quadrature points
  - identical current values
  - identical target labels
  - identical `weight_inv_modB`
  - identical `constraint_weight`
  - covered by
    `TestUpstreamFactoryBoozerMatrix::test_penalty_case_uses_copied_matching_cpu_jax_fixtures`
- [x] Add one helper that returns a pair:
  - CPU `BoozerSurface`
  - JAX `BoozerSurfaceJAX`
- [x] Add one helper that returns immutable inputs:
  - `coil_set_spec`
  - `sdofs`
  - `iota`
  - `G`
  - options snapshot

Acceptance:

- [x] Tests no longer duplicate G/current formulas in multiple places.
- [x] Any CPU/JAX mismatch can be traced to implementation behavior, not
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
- `test_penalty_value_and_gradient_cpu_parity_matrix`
- `tests/geo/test_boozer_derivatives_jax.py`

Tasks:

- [x] Add NCSX exact-surface residual parity for the legacy
  `test_residual` state:
  - same `get_exact_surface()`
  - same `tf_target=0.41431152`
  - same `iota=-0.44856192`
  - same `weight=1.0`
  - compare CPU scalar objective against JAX scalar objective
- [x] Expand penalty value/gradient parity over:
  - `SurfaceXYZFourier`
  - `SurfaceXYZTensorFourier`
  - `stellsym=True`
  - `stellsym=False`
  - `optimize_G=True`
  - `optimize_G=False`
  - fixed root cause: label terms now evaluate on explicit `label.surface`
    runtime metadata, matching upstream CPU label quadrature instead of
    silently reusing the Boozer residual surface grid
- [x] Add Hessian parity:
  - column-complete CPU/C++ Hessian oracle parity via
    `test_penalty_hessian_column_complete_cpu_parity_matrix` using
    `derivative_heavy` with `second_derivative_rtol=1e-6` and
    `second_derivative_atol=1e-8`
  - retain same-state CPU Hessian vs JAX Hessian-vector product directional
    parity over the full surface/stellsym/`optimize_G` matrix as seeded
    operator-path coverage
- [x] Add exact constrained residual/Jacobian parity over the same matrix:
  - compare residual vector
  - compare Jacobian-vector product for seeded direction `h`
  - compare dense Jacobian only for small fixtures where materialization is
    supported
  - implemented with reduced real NCSX exact-constraints fixtures over both
    supported upstream surface families, both `stellsym` modes, and both
    `optimize_G` modes; CPU dense Jacobian is the oracle, JAX JVP is the target
    proof, and dense JAX Jacobian materialization is limited to the small
    reference fixture
- [x] Use `parity_ladder_tolerances` for numeric gates instead of ad hoc
  tolerances. Matrix entries must name the canonical snake_case lane.

Acceptance:

- [x] Same-state residual parity passes.
- [x] Same-state penalty value/gradient parity passes for the full matrix.
- [x] Hessian coverage is either direct parity or clearly labeled
  `fd_gradient` FD/Taylor-equivalence.
- [x] Exact constrained Jacobian coverage is no longer mock-only.

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

- [x] Split legacy `test_run_code` into:
  - solve result parity
  - mutable idempotence exclusion
  - fixed-current `G=None` guard
  - exact solve without explicit `G`
- [x] Add supported LS result parity and public LBFGS routing/schema coverage:
  - CPU `BoozerSurface.run_code`
  - JAX `BoozerSurfaceJAX.run_code`
  - same initial surface
  - same `iota`
  - same `G`
  - same solver options
  - LBFGS public API coverage verifies supported option routing and result
    schema without claiming Wolfe/iteration trace identity
- [x] Compare public result fields:
  - `success`
  - `type`
  - `iota`
  - `G`
  - residual norm
  - label value and label error
  - final surface DOFs when representation is the same
- [x] Add exact Newton `G=None` JAX result contract:
  - if supported, compare final `G is None`, residual norm, and success/failure
    class
  - if unsupported, assert a strict unsupported-JAX error
- [x] Add non-stellsym exact result parity:
  - use Area label as the legacy test does
  - route the JAX label through `area_jax` in
    `src/simsopt/geo/label_constraints_jax.py`
  - assert `lm` length and residual quality
  - compare CPU/JAX final public metrics
- [x] Extend existing manual LS support to CPU/JAX result parity:
  - keep `method="manual"` as a supported JAX public API path
  - use the existing manual API tests as anchors
  - compare public result fields against the matching CPU manual LS case

Acceptance:

- [x] Supported solver paths match in public result semantics.
- [x] Unsupported solver paths fail early and explicitly.
- [x] No test requires identical iteration trace or SciPy line-search internals.
- [x] Existing mutable wrappers may stay, but target-lane solver proofs do not
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

- [x] Verify public method signatures do not expose unsupported legacy knobs.
- [x] Ensure `G=None` with free currents rejects through the shared current
  guard.
- [x] Ensure `G=None` with fixed currents is allowed.
- [x] Ensure adjoint/gradient callback paths apply the same fixed/free current
  logic.
- [x] Ensure spoofed surface names fail by real type/contract, not by string
  matching.
- [x] Ensure unsupported surfaces fail explicitly.
- [x] Keep the JAX accepted surface matrix clear:
  - accepted: supported RZ Fourier, XYZ Fourier, and XYZ Tensor runtime
    surfaces with native JAX geometry contracts
  - rejected: unsupported host-only surfaces unless a native JAX contract exists
- [x] Keep quadpoint/stellsym mask coverage one-for-one with legacy cases.

Acceptance:

- [x] Same public guard behavior on CPU and JAX where JAX supports the path.
- [x] Explicit unsupported-JAX errors where JAX does not support the path.
- [x] No hidden `_coils` fallback or dynamic compatibility route is introduced.

### P5: Derivative And Adjoint Parity

Goal: derivative correctness is proven at the same contract level as values.

Tasks:

- [x] Add direct CPU/JAX wrapper gradient parity for supported labels:
  - `IotasJAX`
  - `NonQuasiSymmetricRatioJAX`
  - `BoozerResidualJAX`
- [x] Add or expand finite-difference checks over the shared NCSX fixtures.
- [x] Add batched RHS adjoint parity for matrix cotangents.
  - exact operator adjoint now solves column-batched RHS matrices through the
    same operator GMRES path and compares against dense JAX and PLU references
- [x] Keep true exact ill-conditioned adjoints, when present, in
  `exact_ill_conditioned_adjoint`: residual/failure-class shape only,
  `residual_rel_tol=1e-10`, and no vector parity requirement.
- [x] Keep the current exact operator-status fixture explicit as mixed-RHS
  coverage: Iotas satisfies the residual-success contract, while NQS exercises
  the residual/failure-only branch.
- [x] Keep exact well-conditioned adjoints in
  `exact_well_conditioned_adjoint`: vector parity required with
  `adjoint_rtol=1e-6`, `adjoint_atol=1e-8`, `gradient_rtol=1e-6`, and
  `gradient_atol=1e-8`.
- [x] Use `derivative_heavy` for direct CPU/JAX derivative matrices, including
  column-complete Boozer Hessian parity, and `fd_gradient` for directional
  FD/Taylor evidence.
- [x] For every derivative test, state whether the proof is:
  - direct CPU/JAX value/gradient parity
  - CPU finite difference vs JAX analytic
  - JAX Taylor/FD self-consistency
  - residual/failure-only coverage

Acceptance:

- [x] Derivative tests are not just smoke tests.
- [x] Every derivative parity claim names its oracle.
- [x] Every derivative parity claim names its canonical tolerance lane.
- [x] Ill-conditioned carve-outs cannot be mistaken for vector parity.

### P6: Supported API Parity

Goal: all supported public APIs have a clear parity or JAX-native contract.

Supported API surface:

- `run_code`
- `minimize_boozer_penalty_constraints_LBFGS`
- `minimize_boozer_penalty_constraints_ls` for supported methods
- `minimize_boozer_penalty_constraints_newton`
- `solve_residual_equation_exactly_newton`
- `minimize_boozer_exact_constraints_newton`
- `get_solved_runtime_state`
- `get_adjoint_runtime_state`
- `run_code_traceable`
- `run_code_functional`

| API | Contract | Current proof anchor |
| --- | --- | --- |
| `run_code` | CPU/JAX parity for supported LS/exact public result schema and branch-stable result metrics | `test_run_code_ls_converges`, `test_run_code_exact_converges`, `TestRunCodeLSParity::test_ls_solve_parity`, `TestExactSolveCPUJAXParity::test_exact_solve_parity` |
| `minimize_boozer_penalty_constraints_LBFGS` | CPU/JAX parity API shape on the supported reference/ondevice lanes | `test_lbfgs_public_api_uses_options_default_when_limited_memory_omitted` |
| `minimize_boozer_penalty_constraints_ls` | CPU/JAX parity API shape for `method="lm"` and `method="manual"` | `test_public_ls_api_routes_ondevice_lm`, `test_public_manual_ls_api_supports_baseline_demo_sequence` |
| `minimize_boozer_penalty_constraints_newton` | CPU/JAX parity API shape for supported Newton polish lanes | `test_public_newton_api_routes_without_legacy_vectorize_kwarg` |
| `solve_residual_equation_exactly_newton` | CPU/JAX parity API shape for exact public solves | `test_exact_result_dict_keys` |
| `minimize_boozer_exact_constraints_newton` | CPU/JAX parity API shape and exact-constraint public result contract | `test_public_exact_constraints_newton_restores_cpu_api`, `test_public_exact_constraints_newton_nonstellsym_stays_native_without_root`, `test_public_exact_constraints_newton_nonstellsym_uses_full_jacobian_solve` |
| `get_solved_runtime_state` | JAX-native equivalent over immutable solved state | `test_get_solved_runtime_state_uses_cached_dofs` |
| `get_adjoint_runtime_state` | JAX-native equivalent over immutable adjoint callbacks and streamable group VJPs | `test_get_adjoint_runtime_state_exposes_runtime_callbacks_and_stream` |
| `run_code_traceable` | JAX-native explicit-state target-lane API | `test_run_code_traceable_exact_uses_operator_only_newton`, `test_run_code_traceable_ls_routes_lm_ondevice` |
| `run_code_functional` | JAX-native explicit-state alias for `run_code_traceable` | `test_run_code_functional_aliases_run_code_traceable_schema` |

Tasks:

- [x] For each API, document whether it is:
  - CPU/JAX parity
  - JAX-native equivalent
  - unsupported in JAX
- [x] Assert public result schema keys for parity APIs.
- [x] Assert runtime-state schema keys for JAX-native APIs.
- [x] Keep `run_code_traceable` and `run_code_functional` explicit-state APIs.
- [x] Do not route target-lane correctness through mutable wrapper re-entry.

Acceptance:

- [x] Public APIs either match CPU behavior or reject unsupported modes.
- [x] JAX-native APIs have explicit immutable state tests.
- [x] There are no silent fallbacks to legacy mutable host behavior.

## Optimizer Decision Memo

Release parity is a final-physics contract, not an optimizer-trajectory
contract. The product gate is CPU/JAX agreement on solver success/failure,
residual norms, public result schemas, final objective values, and final
physics quantities such as `iota`, `G`, label value/error, and anchored axis-z.

SciPy step-by-step trajectory parity remains diagnostic only. Iteration counts,
Wolfe line-search internals, and individual trial-step sequences can differ
between CPU/SciPy and JAX lanes as long as the accepted final state satisfies
the same public result and final-physics acceptance envelope.

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

Execution baseline before this closure:

- `pytest -q tests/geo/test_boozersurface.py tests/geo/test_boozer_residual_jax.py
  tests/geo/test_boozer_derivatives_jax.py`: 59 passed, 14 skipped, 56 subtests
  passed in 312.83 s.
- `pytest -q tests/geo/test_boozersurface_jax.py`: 297 passed, 4 skipped in
  118.63 s.
- `pytest -q tests/geo/test_boozersurface_jax_private.py
  tests/integration/test_single_stage_jax_cpu_reference.py -k "boozer or
  Boozer"`: 103 passed, 2 failed, 157 deselected in 339.24 s.

CPU closure evidence captured on 2026-05-05 with
`JAX_ENABLE_X64=True JAX_PLATFORM_NAME=cpu`:

- `pytest -q tests/geo/test_boozersurface.py`: 23 passed, 56 subtests
  passed in 267.46 s.
- `pytest -q tests/geo/test_boozersurface_jax.py`: 346 passed, 4 skipped in
  216.80 s.
- `pytest -q tests/geo/test_boozersurface_jax_private.py`: 85 passed in
  26.11 s.
- `pytest -q tests/geo/test_boozer_residual_jax.py`: 14 passed, 14 skipped in
  5.27 s.
- `pytest -q tests/geo/test_boozer_derivatives_jax.py`: 22 passed in
  22.73 s.
- `pytest -q tests/integration/test_single_stage_jax_cpu_reference.py -k
  "boozer or Boozer"`: 21 passed, 157 deselected in 393.43 s.

Boozer Hessian oracle addendum captured on 2026-05-05 with
`JAX_ENABLE_X64=True JAX_PLATFORMS=cpu`:

- `pytest -q tests/geo/test_boozersurface_jax.py -k "penalty_hessian"`:
  16 passed, 343 deselected in 135.72 s.
- `pytest -q tests/geo/test_boozersurface_jax.py
  tests/geo/test_boozer_derivatives_jax.py -k "hessian or derivative"`:
  45 passed, 336 deselected in 158.43 s.

Current CPU Boozer validation has no unexplained failures. The two baseline
integration failures are closed by the fixed same-state derivative tolerance
contract and the branch-stable re-solve FD contract. The backend-selection test
double was also corrected to preserve the real `SurfaceXYZTensorFourier` class
contract instead of replacing the class with a `MagicMock`.

Result-contract cleanup rerun captured on 2026-05-05 with
`JAX_ENABLE_X64=True JAX_PLATFORM_NAME=cpu`:

- `pytest -q tests/geo/test_boozersurface.py tests/geo/test_boozersurface_jax.py
  tests/geo/test_boozersurface_jax_private.py
  tests/geo/test_boozer_residual_jax.py
  tests/geo/test_boozer_derivatives_jax.py
  tests/geo/test_surface_objectives_jax.py
  tests/integration/test_single_stage_jax_cpu_reference.py -k "not gpu"`:
  758 passed, 1 skipped, 65 deselected, 56 subtests passed in 1554.75 s.

CPU closure commands:

```bash
JAX_ENABLE_X64=True JAX_PLATFORMS=cpu pytest -q tests/geo/test_boozersurface.py
JAX_ENABLE_X64=True JAX_PLATFORMS=cpu pytest -q tests/geo/test_boozersurface_jax.py
JAX_ENABLE_X64=True JAX_PLATFORMS=cpu pytest -q tests/geo/test_boozersurface_jax_private.py
JAX_ENABLE_X64=True JAX_PLATFORMS=cpu pytest -q tests/geo/test_boozer_residual_jax.py
JAX_ENABLE_X64=True JAX_PLATFORMS=cpu pytest -q tests/geo/test_boozer_derivatives_jax.py
JAX_ENABLE_X64=True JAX_PLATFORMS=cpu pytest -q tests/integration/test_single_stage_jax_cpu_reference.py -k "boozer or Boozer"
```

Optional CUDA proof after CPU closure:

```bash
SIMSOPT_JAX_PLATFORM=cuda JAX_ENABLE_X64=True JAX_PLATFORMS=cuda pytest -q tests/geo/test_boozersurface_jax.py -k "gpu or parity"
SIMSOPT_JAX_PLATFORM=cuda JAX_ENABLE_X64=True JAX_PLATFORMS=cuda pytest -q tests/integration/test_single_stage_jax_cpu_reference.py -k "boozer and gpu"
```

## Definition Of Done

- [x] Every legacy BoozerSurface test has a matrix entry.
- [x] Every non-mutable legacy math test has strict CPU/JAX parity or an
  explicitly named equivalence lane.
- [x] Every supported solver path has CPU/JAX public result parity.
- [x] Every unsupported solver path has a strict unsupported-JAX test.
- [x] Every guard path either matches CPU behavior or rejects by explicit JAX
  contract.
- [x] Every derivative claim states its oracle and tolerance lane.
- [x] Mutable identity behavior is limited to the three documented exclusions:
  - serialization pointer identity
  - cached `res` object identity
  - dirty-flag identity behavior
- [x] JAX replacement tests prove immutable runtime-state equivalence.
- [x] `docs/jax_parity_manifest.md` links to this plan.
- [x] CPU validation ladder passes relative to the current pass/fail watermark,
  with no unexplained failures.
- [x] CUDA validation ladder passes if GPU parity is claimed. No new CUDA
  parity claim is made by this CPU closure, so the optional CUDA proof commands
  remain the required gate before claiming hardware parity.

## Non-Negotiables

- [x] Do not loosen tolerances to hide drift.
- [x] Do not add defensive fallback paths.
- [x] Do not add hidden host-wrapper re-entry to satisfy target-lane tests.
- [x] Do not claim full solver trajectory parity from final metric closeness.
- [x] Do not claim adjoint vector parity from FD-only evidence; Boozer Hessian
  direct parity is now column-complete CPU/C++ oracle coverage.
- [x] Do not use mutable object identity as the JAX parity oracle.
