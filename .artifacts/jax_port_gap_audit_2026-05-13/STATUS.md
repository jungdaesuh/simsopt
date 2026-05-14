# Wave 1 STATUS — JAX parity-test remediation

**Date**: 2026-05-13
**Branch**: `gpu-purity-stage2-20260405`
**Base HEAD at goal start**: `6bfe0dd69` (which sits on top of audit parent
`d258c9285` and base/audit HEAD `da44735ab`).
**Worktree**: `/Users/suhjungdae/code/columbia/simsopt-jax`
**Goal doc**: `.artifacts/jax_port_gap_audit_2026-05-13/GOAL_remediation.md`

## Summary

Closed the four Wave 1 checklist items from `GOAL_remediation.md`:

- **T1** — Boozer residual unit tests now use the C++
  `simsoptpp.boozer_residual` symbol as the oracle (via the ABI-tolerant
  wrapper `simsopt.geo.boozersurface._call_boozer_residual`) at all four
  tautology sites originally at `:163-188`, `:192-207`, `:402`, and `:432`.
  The vector→scalar boundary is documented and asserted via
  `0.5 * sum(r**2) / r.size` against the C++ scalar.
- **T2** — `TestBiotSavartJaxChunkedParity` renamed to
  `TestBiotSavartJaxChunkedSelfConsistency`; the class and helper
  docstrings now make the JAX-vs-JAX dense-reference contract explicit
  (not a C++ oracle). A new direct C++ `BiotSavart.B_vjp(v)` parity
  assertion (`test_B_vjp_parity_ncsx`) lives in
  `TestBiotSavartJaxCppParity` and uses the `derivative_heavy` lane.
- **T3** — `_flux_kernel_value_and_grad` docstring declares it a
  fixed-surface edge-contract helper used only by three specific
  call sites; local hard-coded parity tolerances (`_VALUE_RTOL`,
  `_VALUE_ATOL`, `_GRADIENT_RTOL`, `_GRADIENT_ATOL`) have been
  replaced by `PARITY_LADDER_TOLERANCES` entries (`direct_kernel`
  for value, `derivative_heavy` first-derivative row for gradient).
- **G0** — This artifact.
- **G1** — Focused tests pass except one pre-existing unrelated failure
  (see below).
- **G2** — Skipped: no shared-helper changes; unit Boozer and BiotSavart
  parity match the integration parity that already exists in
  `tests/integration/test_single_stage_jax_cpu_reference.py:8156`,
  so no integration slice is needed.

## Files Edited

- `tests/geo/test_boozer_residual_jax.py`
- `tests/field/test_biotsavart_jax.py`
- `tests/objectives/test_fluxobjective_jax_parity.py`
- `.artifacts/jax_port_gap_audit_2026-05-13/STATUS.md` (this file)

Diff scope (vs `HEAD = 6bfe0dd69`):

```
tests/field/test_biotsavart_jax.py                | 102 +++-
tests/geo/test_boozer_residual_jax.py             | 265 +++++++-----
tests/objectives/test_fluxobjective_jax_parity.py |  69 +++-
3 files changed, 302 insertions(+), 134 deletions(-)
```

Pre-existing unrelated dirty files in `git status` (e.g. `docs/banana_*.md`,
`.artifacts/...`, `examples/...`) were not staged or modified.

## Stale Items Pruned From The Original Goal

The Wave 1 scope in `GOAL_remediation.md` already pruned the following
stale claims from the first-draft `/goal`; this STATUS confirms the
remediation rests on the pruned scope:

- `BiotSavartJAX.A() / dA_by_dX() / d2A_by_dXdX()` and `A_vjp / A_and_dA_vjp`
  already exist — no reimplementation attempted (`src/simsopt/field/biotsavart_jax_backend.py:517,523,529,1616,1620`).
- `MagneticFieldSum` / `MagneticFieldMultiply` already support JAX-native
  composition through the existing class path plus strict-mode guards
  (`src/simsopt/field/magneticfield.py:12-37`) — no adapter classes
  created.
- Boozer residual direct C++ parity already exists in the integration
  layer (`tests/integration/test_single_stage_jax_cpu_reference.py:8156`).
  The Wave 1 work added unit-level C++ oracle coverage in
  `tests/geo/test_boozer_residual_jax.py`; integration coverage was
  preserved and not duplicated.
- SquaredFlux parity tests already use CPU `SquaredFlux(...)` (instantiated
  at `tests/objectives/test_fluxobjective_jax_parity.py:171,201`) — the
  Wave 1 work clarified classification and tightened the helper boundary
  rather than reclassifying.

## T1 — Boozer Residual Unit C++ Oracle Details

### Tautology sites remediated

- `:163-188` `test_matches_numpy` (weighted scalar). Inline NumPy formula
  was a literal re-implementation of the JAX kernel formula.
- `:192-207` `test_no_weight` (unweighted scalar). Same anti-pattern.
- `:402` inside `test_vector_parity_near_tolerance_floor` (vector).
  `_numpy_boozer_residual_reference` (defined at `:89`) re-implemented
  the JAX formula in NumPy.
- `:432` inside `test_scalar_residual_norm_near_tolerance_floor` (scalar).
  Same helper.

### How they are now anchored

- Two scalar tests (`test_scalar_matches_cpp_oracle_weighted`,
  `test_scalar_matches_cpp_oracle_unweighted`) call the new
  `_cpp_boozer_residual_scalar` helper which wraps
  `simsoptpp.boozer_residual` via
  `simsopt.geo.boozersurface._call_boozer_residual`. The C++ kernel
  returns the unnormalised half-sum-of-squares (see
  `src/simsoptpp/boozerresidual_impl.h:74`); dividing by
  `num_res = 3 * nphi * ntheta` recovers the JAX
  `boozer_residual_scalar` convention. Lane: `direct_kernel`
  (`rtol=1e-10`, `atol=1e-12`).
- Both near-floor stress tests
  (`test_vector_reduction_near_tolerance_floor_matches_cpp_scalar`,
  `test_scalar_residual_norm_near_tolerance_floor_matches_cpp_oracle`)
  now anchor on the same C++ scalar oracle. The vector test documents
  the vector→scalar boundary explicitly because the public C++ API
  exposes only the scalar `boozer_residual`
  (`src/simsoptpp/boozerresidual_py.cpp:4`); the JAX vector is reduced
  via `0.5 * sum(r**2) / r.size`, matching the JAX scalar definition.
  Lane assertions retain the existing `boozer_residual_floor_scalar`
  parity-lane tier so CPU/GPU floor regressions remain observable.
- The dead pairwise-summation helpers (`_numpy_pairwise_sum_flat`,
  `_numpy_pairwise_reduce_last_axis`, `_numpy_pairwise_sum_last_axis`)
  and the tautological `_numpy_boozer_residual_reference` were
  deleted.
- The remaining `_numpy_cpu_ordered_boozer_scalar_reference` is kept
  because the cpu-ordered reduction test was not flagged in the
  goal's four tautology sites and it tests a specific reduction-order
  claim with a bit-identical `rtol=0.0, atol=0.0` assertion.

### Module-scope import strategy

- `from benchmarks.validation_ladder_contract import parity_ladder_tolerances`
  is added at module import time (`tests/conftest.py` already adds the
  repo root to `sys.path`).
- `from simsopt.geo.boozersurface import _call_boozer_residual` is
  imported inside `_cpp_boozer_residual_scalar` so non-C++ tests
  remain runnable without `simsoptpp`.
- Each C++ oracle test calls `pytest.importorskip("simsoptpp")` first
  so the module continues to import in pure-JAX environments.

## T2 — BiotSavart Chunked Reference Reclassification Details

- `TestBiotSavartJaxChunkedParity` renamed to
  `TestBiotSavartJaxChunkedSelfConsistency`. Class docstring at the
  rename site now states explicitly that the dense reference is the
  non-chunked JAX kernel evaluated via `jax.vmap` / `jax.jacfwd` /
  `jax.vjp` against `module._one_point_dense` — not the C++
  `simsoptpp.BiotSavart` symbol. The class is therefore a Tier-4
  self-consistency probe per `tests/REVIEWER_ORACLE_LINT.md`.
- `_dense_reference_fields`, `_dense_B_reference`, and `_dense_B_vjp`
  helpers now carry docstrings labelling them as self-consistency
  helpers, not C++ oracles, and they reference
  `TestBiotSavartJaxCppParity` as the location of direct C++
  oracle assertions.
- New direct-oracle parity assertion `test_B_vjp_parity_ncsx` in
  `TestBiotSavartJaxCppParity` compares `BiotSavartJAX.B_vjp(v)`
  against `simsopt.field.biotsavart.BiotSavart.B_vjp(v)` on identical
  coils/points/cotangent. Both `Derivative` objects are evaluated per
  coil. Tolerance is sourced from `PARITY_LADDER_TOLERANCES`'s
  `derivative_heavy` first-derivative row (`rtol=1e-8`, `atol=1e-10`).
- Pre-existing unused import `grouped_field_sharding_summary` was
  removed (one-line F401 cleanup) so `ruff check` passes cleanly on
  the touched file; this is the only otherwise-unrelated change in
  the file.

## T3 — SquaredFlux Helper Classification Details

- Confirmed at `:169,199` (originally `:169,199` in the goal anchor;
  ruff-format kept the call-site lines stable after the local-tolerance
  refactor) that CPU/JAX value and gradient parity uses
  `SquaredFlux(...)` — CPU C++ — as the oracle.
- Added a docstring above `_flux_kernel_value_and_grad` (now at
  the equivalent location after formatting) declaring it a fixed-surface
  edge-contract helper used only by:
  - `test_quadratic_flux_zero_normals_contract`
  - `test_degenerate_normals_do_not_perturb_valid_flux_contracts`
  - `test_singular_zero_field_contract`
  The docstring also names the closed-form NumPy oracle
  (`_single_valid_flux_value_and_gradient`) cited by the second test.
- Replaced module-level constants `_VALUE_RTOL` / `_VALUE_ATOL` /
  `_GRADIENT_RTOL` / `_GRADIENT_ATOL` with
  `parity_ladder_tolerances("direct_kernel")` for value and
  `parity_ladder_tolerances("derivative_heavy")` first-derivative row
  for gradient. Both `_assert_flux_value_parity` and
  `_assert_flux_gradient_parity` now read from the SSOT lane dicts
  and document the lane source in their docstrings.
- **Deliberate tolerance trade-off (Crucible reviewer flagged, retained
  by design)**: commit `7e8e8f622` (2026-04-10, "test: tighten
  fluxobjective wrapper parity") originally ratcheted the local
  constants to `_VALUE_RTOL=1e-12`/`_VALUE_ATOL=1e-15`/
  `_GRADIENT_RTOL=1e-11`/`_GRADIENT_ATOL=1e-14` — tighter than the lane
  SSOT (`direct_kernel` `rtol=1e-10`/`atol=1e-12` for value;
  `derivative_heavy` `first_derivative_rtol=1e-8`/
  `first_derivative_atol=1e-10` for gradient). The Wave 1 T3 instruction
  asked for SSOT compliance (`Replace local hard-coded parity
  tolerances with entries from PARITY_LADDER_TOLERANCES where the test
  is an actual parity test`), and `CLAUDE.md` is explicit that
  `PARITY_LADDER_TOLERANCES` OWNS the lane-specific precision contract.
  Loosening to the SSOT for the CPU/JAX parity tests is therefore the
  documented direction: the lane is the contract floor, and future
  tightening (if the kernel achieves consistently better than 1e-10
  on every supported platform) should ratchet the lane SSOT itself
  rather than re-introducing local overrides. The historical 1e-12
  pass at HEAD demonstrates the kernel currently does better than the
  lane minimum, but that headroom is not a contractual guarantee.
- **Chunked-vs-dense flux self-consistency carve-out (Crucible
  reviewer fix)**: `test_squaredfluxjax_large_point_cloud_grouped_vjp_matches_dense`
  is a JAX-vs-JAX Tier-4 self-consistency check (chunked vs dense on
  the same kernel) per `tests/REVIEWER_ORACLE_LINT.md`, NOT a CPU/JAX
  parity test. The T3 instruction explicitly scoped its tolerance
  replacement to "where the test is an actual parity test." Routing
  this self-consistency test through the parity-lane SSOT would
  silently widen the chunking-bug detection window for the flux path,
  diverging from the comparable BiotSavart chunked-vs-dense
  self-consistency tests that use explicit `atol=1e-14` inline. The
  test now uses inline `rtol=1e-12, atol=1e-15` for value and
  `rtol=1e-11, atol=1e-14` for gradient — the historical floor for
  this specific self-consistency contract — and the test docstring
  documents the rationale.
- Directional FD coverage for every entry in
  `_SQUARED_FLUX_DEFINITIONS` already exists via
  `@pytest.mark.parametrize("definition", _SQUARED_FLUX_DEFINITIONS)`
  on `test_squaredfluxjax_gradient_matches_directional_taylor_fd`
  (all three of `"quadratic flux"`, `"normalized"`, `"local"` are
  parametrised). No extension needed.

## G1 — Focused Validation Commands And Outcomes

```
.conda/jax-0.9.2/bin/python -m ruff check \
  tests/geo/test_boozer_residual_jax.py \
  tests/field/test_biotsavart_jax.py \
  tests/objectives/test_fluxobjective_jax_parity.py
# → All checks passed!

.conda/jax-0.9.2/bin/python -m ruff format --check \
  tests/geo/test_boozer_residual_jax.py \
  tests/field/test_biotsavart_jax.py \
  tests/objectives/test_fluxobjective_jax_parity.py
# → 3 files already formatted

.conda/jax-0.9.2/bin/python -m pytest tests/geo/test_boozer_residual_jax.py -v
# → 15 passed, 15 skipped in 8.90s  (skips are gpu_parity-only)

.conda/jax-0.9.2/bin/python -m pytest tests/field/test_biotsavart_jax.py -v
# → 23 passed in 23.14s

.conda/jax-0.9.2/bin/python -m pytest tests/objectives/test_fluxobjective_jax_parity.py -v
# → 1 failed, 27 passed, 28 skipped in 14.59s
```

### Pre-existing failure not introduced by Wave 1

`test_squaredfluxjax_requires_native_field_contract[cpu_parity]` fails
**both before and after** the Wave 1 edits — verified by `git stash` to
the unmodified `HEAD = 6bfe0dd69` and re-running the same test:

```
NotImplementedError: SquaredFluxJAX requires a field exposing integer
_dof_layout_version for drift detection.

AssertionError: Regex pattern did not match.
  Expected regex: 'coil_dof_extraction_spec'
  Actual message: 'SquaredFluxJAX requires a field exposing integer
                   _dof_layout_version for drift detection.'
```

Root cause (pre-existing, unrelated to Wave 1):

- `_NonNativeFakeField` in
  `tests/objectives/test_fluxobjective_jax_parity.py` does not expose
  `_dof_layout_version`.
- `SquaredFluxJAX.__init__` (`src/simsopt/objectives/fluxobjective_jax.py:73`)
  raises `NotImplementedError` on a missing `_dof_layout_version` BEFORE
  reaching the original `coil_dof_extraction_spec` rejection that the
  test's regex was written against.
- The test still proves rejection of non-native fields via
  `NotImplementedError`; only the regex on the error message is stale.

The failure is therefore documented per `GOAL_remediation.md`
Acceptance Criterion 5 ("Focused validation passes, or failures are
recorded with exact failing tests and root-cause notes in
`STATUS.md`"). It is NOT a regression introduced by this `/goal`;
remediation belongs to a separate scope (regex update or fake-field
augmentation).

## G2 — Regression Gate

Skipped. Justification:

- The Wave 1 edits touch only test code; no shared helper or production
  module was modified.
- Unit-level C++ Boozer residual parity assertions (`direct_kernel`
  lane) duplicate the lane that the integration parity fixture at
  `tests/integration/test_single_stage_jax_cpu_reference.py:8156`
  already validates against. Adding both passes does not require
  re-running the broader public pure-JAX command in CLAUDE.md.
- New `BiotSavartJAX.B_vjp(v)` parity (`derivative_heavy` lane) is
  consistent with the existing integration coverage
  (`tests/integration/test_stage2_jax.py:1658`) — same kernels, same
  symbols, same tolerance lane.

If a downstream Wave or Crucible review surfaces a shared-helper drift,
the public pure-JAX command in `CLAUDE.md` should be run as the next
gate.

## Acceptance Criteria Self-Check

1. ✅ Boozer residual unit tests contain a direct C++ scalar oracle for
   weighted (`test_scalar_matches_cpp_oracle_weighted`) and unweighted
   (`test_scalar_matches_cpp_oracle_unweighted`) residual modes. The
   vector tests cite the `0.5 * sum(r**2) / r.size` vector→scalar
   boundary explicitly in their docstring.
2. ✅ No test in the touched files presents JAX-dense BiotSavart output
   as an independent C++ parity oracle.
   `TestBiotSavartJaxChunkedSelfConsistency` is labelled
   self-consistency in class name, class docstring, and individual
   helper docstrings.
3. ✅ SquaredFlux parity tests use `SquaredFlux` CPU methods as the
   oracle (instantiated at `:171,201`); `_flux_kernel_value_and_grad`
   is documented as an edge-contract helper used only by the three
   listed tests.
4. ✅ `.artifacts/jax_port_gap_audit_2026-05-13/STATUS.md` exists (this
   file) and records changes, pruned stale claims, and validation
   outcomes.
5. ✅ Focused validation passes for all goal-scoped sites. The single
   unrelated `test_squaredfluxjax_requires_native_field_contract`
   failure is pre-existing and recorded above with exact failing test
   name and root-cause analysis.
