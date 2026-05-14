# GOAL - Wave 1 JAX parity-test remediation (validated 2026-05-14)

**/goal-ready objective:** Validated against base HEAD `da44735ab` (the audit
parent); this audit / goal doc was committed as `d258c9285`. Remediate the
still-valid Wave 1 parity-test gaps from the 2026-05-13 JAX port audit:
move Boozer residual unit coverage onto a direct C++ oracle, make BiotSavart
chunked tests stop presenting JAX-dense references as parity oracles, reconcile
the SquaredFlux parity-test helper classification, run the focused affected
tests, and write the audit status artifact.

This file intentionally narrows the original "close every gap" roadmap. The
full port/parity backlog is larger than one `/goal` and contains stale claims
against the current tree. Use this document for `/goal`; use the four audit
inputs below only as background.

## Inputs

Read these before editing:

- `.artifacts/jax_port_gap_audit_2026-05-13/cpp_port_gap.md`
- `.artifacts/jax_port_gap_audit_2026-05-13/python_port_gap.md`
- `.artifacts/jax_port_gap_audit_2026-05-13/jax_cpp_parity_test_gaps.md`
- `.artifacts/jax_port_gap_audit_2026-05-13/jax_python_test_mirror_gaps.md`
- `CLAUDE.md`
- `tests/REVIEWER_ORACLE_LINT.md`
- `benchmarks/validation_ladder_contract.py::PARITY_LADDER_TOLERANCES`

Branch: `gpu-purity-stage2-20260405`

Worktree: `/Users/suhjungdae/code/columbia/simsopt-jax`

## Current-Tree Corrections

These corrections are part of the goal contract. Do not reintroduce the stale
claims from the first draft.

- `BiotSavartJAX.A()`, `dA_by_dX()`, and `d2A_by_dXdX()` already exist in
  `src/simsopt/field/biotsavart_jax_backend.py:517,523,529` (with subclass
  overrides at `:1447,1451,1458`), backed by
  `src/simsopt/jax_core/biotsavart.py:617` (`biot_savart_A`) and `:694`
  (`grouped_biot_savart_A`). `A_vjp` / `A_and_dA_vjp` are at
  `biotsavart_jax_backend.py:1616,1620`.
- `tests/field/test_biotsavart_A_direct_kernel_closeout.py` already provides
  direct-kernel CPU-oracle coverage for `BiotSavartJAX.A()`.
- `MagneticFieldSum` / `MagneticFieldMultiply` already have JAX-native
  composition behavior through the existing classes, with strict-mode guards
  defined in `src/simsopt/field/magneticfield.py:12-37`
  (`_is_jax_native_field`, `_raise_if_strict_jax_mixed_composition`) and
  invoked at `:226,279`, plus pure JAX primitives in
  `src/simsopt/jax_core/magneticfield_composition.py`.
- Boozer residual direct C++ parity exists in integration coverage:
  `tests/integration/test_single_stage_jax_cpu_reference.py` co-imports
  `BoozerResidual` (CPU) and `BoozerResidualJAX` (lines 108, 154). The
  strongest direct-parity references are
  `TestBoozerResidualCPUParity:8156` (JAX `boozer_residual_scalar` vs C++
  `sopp.boozer_residual` at the same state),
  `TestBoozerResidualDerivativeCPUParity:5536` (composed Jacobian vs CPU
  oracle), and the wrapper-gradient parity fixture at `:4818`
  (`test_real_fixture_ondevice_parity_and_wrapper_gradients`). The focused
  unit file `tests/geo/test_boozer_residual_jax.py` carries unit-level
  NumPy-formula tautologies at four sites: inline formulas at `:163-188`
  (`test_matches_numpy`, weighted) and `:192-207` (`test_no_weight`,
  unweighted) inside `TestBoozerResidualScalar`, plus
  `_numpy_boozer_residual_reference` (defined `:89`) at `:402` and `:432`.
  The goal is unit-level oracle cleanup at all four sites.
- `tests/objectives/test_fluxobjective_jax_parity.py` already compares
  `SquaredFluxJAX.dJ()` against `SquaredFlux.dJ()` using CPU `SquaredFlux(...)`
  instantiated at `:169,199`, and includes directional FD coverage. The
  `_flux_kernel_value_and_grad` helper (defined `:279`) is used only by three
  edge-contract tests: `test_quadratic_flux_zero_normals_contract` (`:370`),
  `test_degenerate_normals_do_not_perturb_valid_flux_contracts` (`:383`), and
  `test_singular_zero_field_contract` (`:404`). None pose as a C++ parity
  oracle. The goal is to preserve that classification with an explicit
  comment / docstring, not to reclassify.
- `.artifacts/jax_port_gap_audit_2026-05-13/STATUS.md` does not exist yet and
  must be created by this goal.

## Scope

In scope:

- `tests/geo/test_boozer_residual_jax.py`
- `tests/field/test_biotsavart_jax.py`
- `tests/objectives/test_fluxobjective_jax_parity.py`
- `.artifacts/jax_port_gap_audit_2026-05-13/STATUS.md`
- Any minimal test helper edits required by those files.

Out of scope:

- New MHD ports, surface bootstrap ports, `CoilSetJAX`, `QfmSurfaceJAX`,
  ScalarPotentialRZ SymPy codegen, and broad curve/surface objective mirrors.
- Rewriting already-landed `MagneticFieldSum` / `MagneticFieldMultiply`
  composition.
- Re-implementing existing BiotSavart `A` / `dA_by_dX` / `d2A_by_dXdX`.
- Committing or staging unrelated dirty files.

## Checklist

- [ ] **G0 - Status artifact**
  - [ ] Create `.artifacts/jax_port_gap_audit_2026-05-13/STATUS.md`.
  - [ ] Record current HEAD, edited files, stale items pruned from the original
        goal, focused validation commands, and final pass/fail status.

- [ ] **T1 - Boozer residual unit C++ oracle**
  - [ ] Read `tests/geo/test_boozer_residual_jax.py`. Inline NumPy-formula
        tautologies live at `:163-188` (`test_matches_numpy`, weighted) and
        `:192-207` (`test_no_weight`, unweighted), both inside
        `TestBoozerResidualScalar`. The named helper
        `_numpy_boozer_residual_reference` (definition `:89`) is called at
        `:402` and `:432`. Existing direct integration parity:
        `TestBoozerResidualCPUParity:8156`,
        `TestBoozerResidualDerivativeCPUParity:5536`, and the
        wrapper-gradient parity fixture at `:4818` in
        `tests/integration/test_single_stage_jax_cpu_reference.py`.
  - [ ] Add direct C++ oracle checks at all four tautology sites
        (`:163-188`, `:192-207`, `:402`, `:432`) for weighted and unweighted
        scalar residuals. Two equivalent oracle entry points:
        (a) call the top-level `simsoptpp.boozer_residual` symbol directly
        (`src/simsoptpp/boozerresidual_py.cpp:4`); or
        (b) call `_call_boozer_residual` at
        `src/simsopt/geo/boozersurface.py:104` — a thin Python wrapper
        around the same C++ symbol. Note: `boozersurface.py:4` imports
        `simsoptpp` at module top, so any use of (b) requires `simsoptpp`
        to be importable.
  - [ ] For `boozer_residual_vector`, do not invent a component oracle: the
        public C++ API exposes only the scalar `boozer_residual` and the
        derivative variants `boozer_residual_ds` / `_ds2`
        (`boozerresidual_py.cpp:4,11,22`). Compare the JAX vector to the
        C++ scalar via `0.5 * sum(r²) / r.size` (matching the NumPy
        reference at `tests/geo/test_boozer_residual_jax.py:97`) and
        document the vector→scalar boundary explicitly.
  - [ ] Keep finite-difference gradient/Hessian tests only where they are
        genuinely FD tests. Do not label FD-only checks as C++ parity.
  - [ ] Leave non-C++ tests runnable in pure-JAX environments. C++ oracle
        tests may use module-scope or per-test
        `pytest.importorskip("simsoptpp")` — top-level module name; there
        is no `simsopt._simsoptpp`.

- [ ] **T2 - BiotSavart chunked reference classification**
  - [ ] Read all `_dense_B_vjp` and `_dense_reference_fields` call sites in
        `tests/field/test_biotsavart_jax.py`. Helpers are defined at `:252`
        (`_dense_reference_fields`) and `:294` (`_dense_B_vjp`); current
        callsites are `_dense_B_vjp` at `:585` and `_dense_reference_fields`
        at `:621,862,980`, all inside `TestBiotSavartJaxChunkedParity`
        (`:526`, docstring `:527`: "Directly compare chunked low-level
        kernels against dense references."). The adjacent
        `TestBiotSavartJaxCppParity` (`:480`) is a genuine C++-parity class
        (uses `pytest.importorskip("simsoptpp")` and compares JAX against
        `bs.B()` (`:494`, asserted `:503`) and `bs.dB_by_dX()` (`:509`,
        asserted `:518`)) — leave it alone.
        Validated 2026-05-14: both `_dense_*` helpers are pure JAX autodiff
        on the same `module._one_point_dense` kernel (`jax.vjp`, `jax.vmap`
        + `jax.jacfwd`), so they are chunked-vs-dense self-consistency
        probes, not C++ oracles. Rename `TestBiotSavartJaxChunkedParity`
        (e.g., to `TestBiotSavartJaxChunkedSelfConsistency`) and rewrite
        its `:527` docstring to make clear the dense reference is the
        non-chunked JAX kernel, not a C++ oracle.
  - [ ] If a test is a chunked-vs-dense implementation self-consistency test,
        rename/comment it as such and keep it out of direct C++ parity claims.
  - [ ] Add or move direct C++ oracle assertions for any still-missing
        production parity surface into the C++ parity section of the file.
  - [ ] `B_vjp` parity must compare `BiotSavartJAX.B_vjp(v)` with
        `simsopt.field.BiotSavart.B_vjp(v)` on identical coils/points/cotangent
        and use the `derivative_heavy` lane from the validation-ladder SSOT.
  - [ ] Do not delete chunking tests just because their reference is JAX-dense;
        delete only misleading parity wording or duplicate dead helpers.

- [ ] **T3 - SquaredFlux helper classification**
  - [ ] Confirm CPU/JAX value and gradient parity uses CPU `SquaredFlux(...)`
        (instantiated at `:169,199`) for `.J()` / `.dJ()` — verified at the
        2026-05-14 review.
  - [ ] Add an explicit docstring/comment block above
        `_flux_kernel_value_and_grad` (`:279`) declaring it a fixed-surface
        edge-contract helper used only by `:370,383,404` (zero normals,
        degenerate normals, singular zero field). It must not be cited as a
        parity oracle.
  - [ ] Replace local hard-coded parity tolerances with entries from
        `PARITY_LADDER_TOLERANCES` where the test is an actual parity test.
  - [ ] Extend the directional FD gradient coverage if it is not exercised for
        every entry in `_SQUARED_FLUX_DEFINITIONS`.

- [ ] **G1 - Focused validation**
  - [ ] `ruff check` on every changed Python file.
  - [ ] `ruff format --check` on every changed Python file.
  - [ ] `.conda/jax-0.9.2/bin/python -m pytest tests/geo/test_boozer_residual_jax.py -v`
  - [ ] `.conda/jax-0.9.2/bin/python -m pytest tests/field/test_biotsavart_jax.py -v`
  - [ ] `.conda/jax-0.9.2/bin/python -m pytest tests/objectives/test_fluxobjective_jax_parity.py -v`

- [ ] **G2 - Regression gate**
  - [ ] Run the public pure-JAX command from `CLAUDE.md` if the focused tests
        pass and the change touches shared helpers.
  - [ ] Run a targeted integration parity slice only if unit-level Boozer or
        BiotSavart results disagree with existing integration parity.

## Acceptance Criteria

The `/goal` is complete when:

1. Boozer residual unit tests contain a direct C++ scalar oracle for weighted
   and unweighted residual modes, and vector tests clearly state the scalar-norm
   oracle boundary.
2. No test presents JAX-dense BiotSavart output as an independent parity oracle.
   Chunked-vs-dense tests are labelled as self-consistency checks.
3. SquaredFlux parity tests use `SquaredFlux` CPU methods as the oracle; the
   fixed-surface helper is documented as an edge-contract helper only.
4. `.artifacts/jax_port_gap_audit_2026-05-13/STATUS.md` exists and records
   what was fixed, what was pruned as stale, and the validation outcomes.
5. Focused validation passes, or failures are recorded with exact failing tests
   and root-cause notes in `STATUS.md`.

## All Waves

Wave 1 is the active `/goal` slice described above. Waves 2-7 are written here
as complete follow-on waves, but each later wave must be revalidated against the
live tree before it becomes a new `/goal` or implementation PR. This is
intentional: the first draft contained stale claims at `da44735ab`.

### Wave 1 - Tautology And Oracle Classification Fixes

Status: active `/goal` scope. The top-level **Checklist** (G0/T1/T2/T3/G1/G2)
and **Acceptance Criteria** sections above are the SSOT for this wave. This
subsection is kept as a stable anchor so cross-wave references resolve.

### Wave 2 - BiotSavart Derivative Ladder

Status: follow-on wave. Partially stale. `A`, `dA_by_dX`, and `d2A_by_dXdX`
already exist; do not reimplement them. The likely live gap is the
coil-current derivative ladder and direct parity around it.

Files likely touched:

- `src/simsopt/field/biotsavart_jax_backend.py`
- `src/simsopt/jax_core/biotsavart.py`
- `src/simsopt/field/biotsavart_jax.py`
- `tests/field/test_biotsavart_jax.py`
- `tests/field/test_biotsavart_jax_parity.py`
- `.artifacts/jax_port_gap_audit_2026-05-13/STATUS.md`

Workstreams:

- [ ] **W2-B0 - Current-tree revalidation**
  - [ ] Confirm current public API on `BiotSavartJAX`, `SpecBackedBiotSavartJAX`,
        and `simsopt.jax_core.biotsavart`.
  - [ ] Confirm CPU shape contracts from `simsopt.field.biotsavart.BiotSavart`.
  - [ ] Confirm current direct C++ parity tests for `B`, `dB_by_dX`, `A`,
        `dA_by_dX`, `d2B_by_dXdX`, `d2A_by_dXdX`, and `B_vjp`.
  - [ ] Validated 2026-05-14 against base/audit HEAD `da44735ab`:
        `dA_by_dcoilcurrents`, `d2A_by_dXdcoilcurrents`,
        `d3A_by_dXdXdcoilcurrents`, and their B-side analogs
        (`dB_by_dcoilcurrents` family) are not present in
        `src/simsopt/field/biotsavart_jax_backend.py`. Reconfirm absence in
        `biotsavart_jax_backend.py` and `simsopt/jax_core/biotsavart.py`
        before implementing W2-B1 / W2-B2.
  - [ ] Update `STATUS.md` with the live gap list before implementation.

- [ ] **W2-B1 - B-side coil-current derivatives**
  - [ ] Implement `dB_by_dcoilcurrents` if missing.
  - [ ] Implement `d2B_by_dXdcoilcurrents` if missing.
  - [ ] Implement `d3B_by_dXdXdcoilcurrents` if missing.
  - [ ] Preserve tensor convention:
        `dB_by_dX[p, j, l] = partial_j B_l(x_p)`.
  - [ ] Preserve mixed quadrature grouping for TF/banana mixed coil sets.
  - [ ] Avoid `simsoptpp` imports in source modules.

- [ ] **W2-B2 - A-side coil-current derivatives**
  - [ ] Keep existing `A`, `dA_by_dX`, and `d2A_by_dXdX`.
  - [ ] Implement `dA_by_dcoilcurrents` if missing.
  - [ ] Implement `d2A_by_dXdcoilcurrents` if missing.
  - [ ] Implement `d3A_by_dXdXdcoilcurrents` if missing.
  - [ ] Match CPU axis conventions from `BiotSavart`.

- [ ] **W2-B3 - Direct C++ parity**
  - [ ] Add direct oracle tests for every newly exposed current derivative.
  - [ ] Add direct oracle `B_vjp` coverage if Wave 1 did not already land it
        in the canonical parity section.
  - [ ] Use `direct_kernel` for values and `derivative_heavy` for first/second
        derivatives.
  - [ ] Reuse existing fixtures where possible; do not create a parallel test
        dialect.

- [ ] **W2-B4 - Coil-current Taylor mirrors**
  - [ ] Mirror `tests/field/test_biotsavart.py` current Taylor tests for `B`.
  - [ ] Mirror vector-potential current Taylor tests for `A`.
  - [ ] Use C++ value parity plus type-3 FD/Taylor convergence as separate
        oracles.
  - [ ] Record observed convergence orders in the test or `STATUS.md`.

Validation:

- [ ] `ruff check` / `ruff format --check` on changed files.
- [ ] `.conda/jax-0.9.2/bin/python -m pytest tests/field/test_biotsavart_jax.py tests/field/test_biotsavart_jax_parity.py -v`
- [ ] Run the public pure-JAX command from `CLAUDE.md` if shared helpers change.

Acceptance:

- [ ] New JAX current-derivative methods shape-match CPU `BiotSavart`.
- [ ] Direct C++ parity exists for new methods.
- [ ] Existing `A`/`dA`/`d2A` behavior is not regressed or duplicated.

### Wave 3 - MHD Reductions

Status: follow-on wave. Needs fresh import/path validation before coding.

Files likely touched:

- `src/simsopt/mhd/boozer_jax.py` or another existing MHD JAX module
- `src/simsopt/mhd/vmec_diagnostics_jax.py`
- `tests/mhd/test_boozer_jax.py`
- `tests/mhd/test_vmec_diagnostics_jax.py`
- pinned fixtures under `tests/test_files/` only if existing fixtures are
  insufficient

Workstreams:

- [ ] **W3-M0 - Revalidate MHD targets**
  - [ ] Locate current `Quasisymmetry`, `QuasisymmetryRatioResidual`,
        `IotaTargetMetric`, `IotaWeighted`, and `WellWeighted` definitions.
  - [ ] Check whether any JAX equivalents already exist.
  - [ ] Identify the smallest existing VMEC/Boozer fixture that can pin
        deterministic outputs.

- [ ] **W3-M1 - `Quasisymmetry` and `QuasisymmetryRatioResidual`**
  - [ ] Port the pure NumPy reduction math to JAX if still missing.
  - [ ] Keep VMEC/Booz_xform objects at the boundary; use arrays in the JAX hot
        path.
  - [ ] Add value parity against the existing NumPy implementation.
  - [ ] Add gradient parity or FD coverage only if the upstream class exposes a
        gradient contract.

- [ ] **W3-M2 - `IotaTargetMetric`, `IotaWeighted`, `WellWeighted`**
  - [ ] Port only the pure reduction kernels.
  - [ ] Preserve constructor and `.J()` / `.dJ()` contracts where applicable.
  - [ ] Add pinned fixture parity tests against the CPU/NumPy originals.

- [ ] **W3-M3 - Shared MHD validation**
  - [ ] Centralize fixture setup if both Boozer and VMEC diagnostics need the
        same VMEC output.
  - [ ] Document any non-portable Fortran/MPI/file-IO boundary as out of scope.

Validation:

- [ ] `.conda/jax-0.9.2/bin/python -m pytest tests/mhd/test_boozer*.py tests/mhd/test_vmec_diagnostics*.py -v`
- [ ] Run focused import-smoke tests if new modules are exported.

Acceptance:

- [ ] Each ported MHD reduction has a CPU/NumPy oracle test on pinned data.
- [ ] File IO and VMEC/SPEC execution are not moved into JAX scope.

### Wave 4 - Curve Geometry And Objectives

Status: follow-on wave. Current `src/simsopt/jax_core/curve_geometry.py` must
be checked first; do not assume pure functions still live only in
`src/simsopt/geo/curve.py`.

Files likely touched (revalidated 2026-05-14):

- `src/simsopt/jax_core/curve_geometry.py`
- `src/simsopt/geo/curve.py`
- `src/simsopt/geo/curveobjectives.py`
- existing curve-JAX / closeout tests in the current tree:
  `tests/geo/test_curve_item05_closeout.py`,
  `tests/geo/test_curveobjectives_item07_closeout.py`,
  `tests/geo/test_curvexyzfouriersymmetries_spec_jax.py`,
  `tests/geo/test_framedcurve_jax_item18.py`,
  `tests/geo/test_framedcurve_jax_wrappers_item18.py`. Pick or create the
  W4-C2 mirror file during W4-C0 — do not assume
  `tests/geo/test_curve_jax.py` or `tests/geo/test_curve_objectives_jax.py`
  exist (they do not).

Workstreams:

- [ ] **W4-C0 - Revalidate curve geometry location**
  - [ ] Locate `kappa`, `torsion`, `incremental_arclength`, and their gradient
        / VJP helpers in the current tree.
  - [ ] Identify any import cycle risk between `simsopt.geo.curve` and
        `simsopt.jax_core`.
  - [ ] Decide whether the needed action is move, re-export, or no-op.

- [ ] **W4-C1 - Lift or normalize curve geometry API**
  - [ ] Move only missing pure functions into `jax_core/curve_geometry.py`.
  - [ ] Keep backward-compatible imports from `geo/curve.py`.
  - [ ] Avoid dynamic imports. If a cycle blocks the move, document the root
        cycle and stop.

- [ ] **W4-C2 - Curve-objective FD-Taylor mirrors**
  - [ ] Revalidate missing mirrors for `CurveCurveDistance`,
        `LpCurveTorsion`, `MeanSquaredCurvature`, and `ArclengthVariation`.
  - [ ] Add type-3 FD-Taylor tests against the JAX value path.
  - [ ] Add CPU value parity where the CPU object is independent and not a
        wrapper around the same JAX kernel.
  - [ ] Preserve or extend existing `FramedCurveTwist` / `LinkingNumber`
        closeout coverage rather than duplicating it.

- [ ] **W4-C3 - C++ curve VJP parity**
  - [ ] Revalidate `kappa_by_dcoeff_vjp` and related VJP coverage.
  - [ ] Add C++ oracle parity only for still-missing VJP surfaces.
  - [ ] Use `derivative_heavy` tolerances.

Validation:

- [ ] `.conda/jax-0.9.2/bin/python -m pytest tests/geo/test_curve.py tests/geo/test_curve_objectives.py tests/geo/test_curve_item05_closeout.py tests/geo/test_curveobjectives_item07_closeout.py tests/geo/test_curvexyzfouriersymmetries_spec_jax.py tests/geo/test_framedcurve_jax_item18.py tests/geo/test_framedcurve_jax_wrappers_item18.py -v`

Acceptance:

- [ ] No duplicate curve-geometry implementation survives without a reason.
- [ ] New objective mirrors cite their oracle type.
- [ ] Import-smoke paths remain acyclic.

### Wave 5 - Surfaces

Status: follow-on wave. Partially stale. Current
`tests/geo/test_surface_fourier_jax.py` already contains some `fit_to_curve`
and projected-normal coverage; revalidate before adding new modules.

Files likely touched:

- `src/simsopt/geo/surface_bootstrap_jax.py` if still needed
- `src/simsopt/geo/surface_fourier_jax.py`
- `src/simsopt/jax_core/surface_rzfourier.py`
- `tests/geo/test_surface_bootstrap_jax.py`
- `tests/geo/test_surface_fourier_jax.py`
- `tests/geo/test_surface_xyz_curvatures_jax.py`
- `tests/geo/test_surface_rzfourier_jax.py`

Workstreams:

- [ ] **W5-S0 - Revalidate surface coverage**
  - [ ] Check current object-API and pure-kernel coverage for `fit_to_curve`,
        `least_squares_fit`, `_extend_via_normal_for_nonuniform_phi`, and
        `extend_via_projected_normal`.
  - [ ] Check current RZ, XYZ, XYZTensor, Henneberg, and Garabedian surface
        coverage.
  - [ ] Record which missing rows are real in `STATUS.md`.

- [ ] **W5-S1 - Surface bootstrap kernels**
  - [ ] Port only missing JAX hot-path bootstrap kernels.
  - [ ] Mirror C++ signatures and semantics from `python_surfaces.cpp`.
  - [ ] Add C++ oracle parity tests on pinned fixtures.
  - [ ] Do not add Optimizable wrappers unless the live contract requires them.

- [ ] **W5-S2 - XYZ/XYZTensor fundamental forms and curvatures**
  - [ ] Revalidate RZ-only vs XYZ/XYZTensor gaps for `first_fund_form`,
        `second_fund_form`, `surface_curvatures`, and `_by_dcoeff` variants.
  - [ ] Build on existing `surface_fourier_jax.py` evaluators.
  - [ ] Preserve stellsym DOF conventions from `CLAUDE.md`.
  - [ ] Add C++ oracle tests for both stellsym modes where possible.

- [ ] **W5-S3 - Gauss-Bonnet mirror expansion**
  - [ ] Revalidate current RZ Gauss-Bonnet tests.
  - [ ] Add XYZ/XYZTensor/Henneberg mirrors only for still-uncovered surface
        kinds.
  - [ ] Keep self-intersection and projected-normal API tests separate from
        curvature scalar tests.

Validation:

- [ ] `.conda/jax-0.9.2/bin/python -m pytest tests/geo/test_surface_fourier_jax.py tests/geo/test_surface_rzfourier_jax.py -v`
- [ ] Add new focused surface test files to the command as they are created.

Acceptance:

- [ ] Every new surface kernel has a direct CPU/C++ or pinned-data oracle.
- [ ] No new surface module duplicates existing working JAX kernels.

### Wave 6 - Field Composition And Analytic Gaps

Status: follow-on wave. First-draft `MagneticFieldSumJAX` /
`MagneticFieldMultiplyJAX` adapter request is stale. Existing composition uses
`MagneticFieldSum` / `MagneticFieldMultiply` with JAX-native children plus
strict fallback guards.

Files likely touched:

- `src/simsopt/field/magneticfield.py`
- `src/simsopt/jax_core/magneticfield_composition.py`
- `src/simsopt/field/dommaschk_jax.py`
- `src/simsopt/field/reiman_jax.py`
- `src/simsopt/field/scalar_potential_rz_jax.py`
- `tests/field/test_magnetic_field_composition_jax.py`
- `tests/field/test_magneticfieldclasses_jax_item15.py`

Workstreams:

- [ ] **W6-F0 - Revalidate field-composition surface**
  - [ ] Confirm current strict-mode behavior for all-JAX and mixed CPU/JAX
        compositions.
  - [ ] Confirm pure JAX primitives cover the production composition path.
  - [ ] Do not create separate `MagneticFieldSumJAX` or
        `MagneticFieldMultiplyJAX` classes unless current evidence proves the
        existing class path cannot satisfy the contract.

- [ ] **W6-F1 - Composition coverage**
  - [ ] Add missing tests only for uncovered production combinations such as
        BiotSavart + analytic field or nested composition.
  - [ ] Preserve existing strict fallback tests.
  - [ ] Add VJP/gradient coverage only where child fields expose the required
        contract.

- [ ] **W6-F2 - `BifieldMultiply` gradient parity**
  - [ ] Revalidate the CPU original and current JAX composition tests.
  - [ ] Add gradient parity for the live JAX composition path if still missing.
  - [ ] Use a CPU object or closed-form expression as oracle, not JAX-vs-JAX.

- [ ] **W6-F3 - Dommaschk/Reiman analytic mirrors**
  - [ ] Revalidate current `B`, `dB`, `A`, and `d2B_by_dXdX` coverage for
        Dommaschk and Reiman wrappers.
  - [ ] Add Taylor/curl/divergence-free checks only where upstream has a
        corresponding invariant.
  - [ ] Do not force unsupported CPU surfaces into JAX scope.

- [ ] **W6-F4 - ScalarPotentialRZMagneticField**
  - [ ] Decide whether the SymPy-to-JAX printer is required.
  - [ ] If required and out of scope, mark deferred with the exact blocker.
  - [ ] If implemented, add closed-form or CPU-oracle tests for `B`, `dB`, and
        any supported `A`/higher-derivative surfaces.

Validation:

- [ ] `.conda/jax-0.9.2/bin/python -m pytest tests/field/test_magnetic_field_composition_jax.py tests/field/test_magneticfieldclasses_jax_item15.py -v`
- [ ] Add scalar-potential focused tests only if W6-F4 is implemented.

Acceptance:

- [ ] Composition behavior is tested through the real public path.
- [ ] No stale adapter class names are introduced without evidence.
- [ ] Analytic-field tests cite independent oracles.

### Wave 7 - Composite And Ancillary Ports

Status: long-tail follow-on wave. Each item should become its own freshly
validated `/goal` or PR-sized implementation plan.

Files likely touched:

- `src/simsopt/field/coilset_jax.py`
- `src/simsopt/geo/qfmsurface_jax.py`
- `src/simsopt/jax_core/wireframe.py`
- `src/simsopt/geo/surfaceobjectives_jax.py`
- `tests/field/test_coilset_jax.py`
- `tests/geo/test_qfm_jax.py`
- `tests/field/test_wireframefield_jax_item30.py`
- `tests/geo/test_surface_objectives_jax.py`
- `tests/geo/test_boozersurface_jax.py`

Workstreams:

- [ ] **W7-X0 - Revalidate long-tail contracts**
  - [ ] Confirm which original audit items are still unported, partial, or
        already handled.
  - [ ] Split any item larger than one PR into a separate implementation plan.
  - [ ] Identify whether each gap is a pure kernel, an Optimizable adapter, a
        solver, or a test-only mirror.

- [ ] **W7-X1 - `CoilSetJAX` / `ReducedCoilSetJAX`**
  - [ ] Revalidate `src/simsopt/field/coilset.py` and current
        `CoilSetDofExtractionSpec`.
  - [ ] Decide whether to implement a public Optimizable wrapper, lower-level
        pure kernels, or both.
  - [ ] Preserve existing CPU behavior at the boundary.
  - [ ] Add parity tests against `CoilSet` / `ReducedCoilSet` on a small pinned
        fixture.

- [ ] **W7-X2 - `QfmSurfaceJAX`**
  - [ ] Revalidate `QfmResidualJAX` and optimizer support.
  - [ ] Add a solver adapter only if residual and Jacobian surfaces are ready.
  - [ ] Compare against CPU `QfmSurface` convergence on pinned fixtures.
  - [ ] Keep SciPy-only features out of the JAX hot path unless explicitly
        required.

- [ ] **W7-X3 - Wireframe `compute(derivatives=2)`**
  - [ ] Revalidate whether JAX wireframe has `d2B_by_dXdX`.
  - [ ] Implement analytic `d2B_by_dXdX` only if still missing.
  - [ ] Add C++ oracle parity and Taylor tests.
  - [ ] Keep cache orchestration differences documented as non-portable.

- [ ] **W7-X4 - Surface objective Optimizable wrappers**
  - [ ] Revalidate existing pure helpers for area, volume, and toroidal flux.
  - [ ] Add `AreaJAX`, `VolumeJAX`, and `ToroidalFluxJAX` Optimizable wrappers
        only if wrapper absence is still a live contract gap.
  - [ ] Add parity tests against `Area`, `Volume`, and `ToroidalFlux` CPU
        objectives.

- [ ] **W7-X5 - BoozerSurface convergence-history mirrors**
  - [ ] Revalidate current BoozerSurface solver convergence and trajectory
        tests.
  - [ ] Mirror only the CPU convergence-history fixtures that remain
        unmirrored.
  - [ ] Keep solver-trajectory claims separate from fixed-state value/gradient
        parity.

Validation:

- [ ] Run focused tests for each implemented W7 item.
- [ ] Run the public pure-JAX command from `CLAUDE.md` after any shared solver
      or objective wrapper change.
- [ ] Run private optimizer and integration suites only for solver/objective
      changes that affect Stage 2 or single-stage paths.

Acceptance:

- [ ] Each W7 item has its own current-tree evidence block in `STATUS.md`.
- [ ] New wrappers preserve CPU boundary contracts and JAX hot-path contracts.
- [ ] No unrelated long-tail item is folded into a PR without a fresh scope
      check.

## Notes For Executors

- Follow `tests/REVIEWER_ORACLE_LINT.md`: JAX-vs-JAX, re-export identity, and
  host wrappers that secretly route through JAX are not parity oracles.
- Do not import `simsoptpp` from JAX source modules. Tests may import C++ oracle
  symbols.
- Do not add defensive try/except wrappers, dynamic imports, or `Any` casts.
- Do not relax tolerances to make a test pass. If parity fails at the correct
  lane, record the failure and stop.
