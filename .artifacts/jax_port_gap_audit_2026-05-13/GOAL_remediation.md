# GOAL - JAX port-gap remediation waves (reviewed 2026-05-14)

**Original Wave 1 objective:** Validated against base HEAD `da44735ab` (the
audit parent); this audit / goal doc was committed as `d258c9285`. Remediate the
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

## Current Status

- Wave 1 is closed. `STATUS.md` records the implementation, follow-up review,
  and final public pure-JAX regression gate.
- Wave 2 is the next concrete `/goal` candidate. W2-B0 pre-implementation
  revalidation is recorded in `STATUS.md` at current HEAD `d773344d1`.
- Waves 3-7 are follow-on plans. They were expanded from the audit inputs and
  current-tree spot checks, but each wave still needs a fresh pre-state block
  before implementation.

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
- `.artifacts/jax_port_gap_audit_2026-05-13/STATUS.md` already exists in the
  current tree. Wave work should append/update the scoped outcome section and
  must not overwrite unrelated status history.

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
  - [ ] Update `.artifacts/jax_port_gap_audit_2026-05-13/STATUS.md`.
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

Wave 1 is retained as the historical closeout scope described above. Wave 2 is
the next concrete `/goal` candidate. Waves 3-7 are written here as complete
follow-on waves, but each later wave must be revalidated against the live tree
before it becomes a new `/goal` or implementation PR. This is intentional: the
first draft contained stale claims at `da44735ab`. Waves 3-7 were expanded on
2026-05-14 from the four audit inputs plus current-tree spot checks at HEAD
`d773344d1`; line numbers remain advisory.

### Wave 1 - Tautology And Oracle Classification Fixes

Status: closed. The top-level **Checklist** (G0/T1/T2/T3/G1/G2) and
**Acceptance Criteria** sections above are the historical SSOT for this wave;
`STATUS.md` records the implementation and validation closeout. This subsection
is kept as a stable anchor so cross-wave references resolve.

### Wave 2 - BiotSavart Derivative Ladder Closeout

Status: next concrete `/goal` candidate. W2-B0 pre-implementation revalidation
is recorded in `STATUS.md` at HEAD `d773344d1`. Scope = close the remaining
BiotSavart derivative-ladder gaps surfaced by the
2026-05-13 audit: one missing spatial-Hessian wrapper method, the entire
six-method coil-current derivative ladder, and the still-missing direct
C++ parity assertions for `dA_by_dX`, `d2B_by_dXdX`, and `d2A_by_dXdX`.
B-side `B`/`dB_by_dX` parity is complete; `B_vjp` direct C++ parity
landed in Wave 1 (`test_biotsavart_jax.py:542`); A-value parity lives in
`tests/field/test_biotsavart_A_direct_kernel_closeout.py`.

#### Current-Tree Corrections (verified 2026-05-14 at HEAD `d773344d1`)

These corrections are part of the Wave 2 contract. Do not reintroduce
the stale claims from the first draft.

- **`BiotSavartJAX` already exposes (do not reimplement):** `B`
  (`biotsavart_jax_backend.py:1439`), `dB_by_dX` (`:1465`), `A` (`:1447`),
  `dA_by_dX` (`:1451`), `d2A_by_dXdX` (`:1458`), `B_and_dB` (`:1476`),
  `B_vjp` (`:1539`), `A_vjp` (`:1616`), `A_and_dA_vjp` (`:1620`),
  `B_and_dB_vjp` (`:1628`), plus the native-pullback variants `*_pullback_native`
  / `*_cotangents`.
- **Confirmed live gaps on `BiotSavartJAX`** (no method present at
  `biotsavart_jax_backend.py` 1439-1634 method block):
  - `d2B_by_dXdX()` — the unit kernel `biot_savart_d2B_by_dXdX` already
    exists at `src/simsopt/jax_core/biotsavart.py:585` (declared in
    `__all__` at `:40`), but the grouped wrapper layer in
    `src/simsopt/jax_core/field.py` (where every other
    `grouped_biot_savart_*_from_spec` lives — see `:495`,
    `:506`, `:517`, `:531` for B/A/dA/d2A) does not import
    `biot_savart_d2B_by_dXdX`, has no entry for it in the
    `_empty_grouped_field_result` dispatch (`:52-67`), and exposes no
    `grouped_biot_savart_d2B_by_dXdX_from_spec`. Therefore neither
    `SpecBackedBiotSavartJAX` (method block `:511-:545`) nor
    `BiotSavartJAX` (method block `:1439-:1634`) can expose a
    `d2B_by_dXdX()` method without first lifting the grouped helper.
    Contrast with the symmetric A-side path:
    `grouped_biot_savart_d2A_by_dXdX_from_spec` is defined at
    `jax_core/field.py:531`, imported at `biotsavart_jax_backend.py:42`,
    and consumed by `d2A_by_dXdX()` at `:529` and `:1458`.
  - `dB_by_dcoilcurrents(compute_derivatives=0)` — CPU contract at
    `src/simsopt/field/biotsavart.py:30` returns
    `list[ndarray[npoints, 3]]` of length `ncoils`.
  - `d2B_by_dXdcoilcurrents(compute_derivatives=1)` — CPU contract at
    `biotsavart.py:40` returns `list[ndarray[npoints, 3, 3]]`.
  - `d3B_by_dXdXdcoilcurrents(compute_derivatives=2)` — CPU contract at
    `biotsavart.py:50` returns `list[ndarray[npoints, 3, 3, 3]]`.
  - `dA_by_dcoilcurrents(compute_derivatives=0)` — CPU contract at
    `biotsavart.py:132` returns `list[ndarray[npoints, 3]]`.
  - `d2A_by_dXdcoilcurrents(compute_derivatives=1)` — CPU contract at
    `biotsavart.py:142` returns `list[ndarray[npoints, 3, 3]]`.
  - `d3A_by_dXdXdcoilcurrents(compute_derivatives=2)` — CPU contract at
    `biotsavart.py:152` returns `list[ndarray[npoints, 3, 3, 3]]`.
- **Direct C++ parity already covered:** `B`
  (`test_biotsavart_jax.py:507`), `dB_by_dX` (`:522`), `B_vjp` (`:542`
  added by Wave 1), and `A` value (`test_biotsavart_A_direct_kernel_closeout.py`).
  The existing `B` row still has an inline `rtol=1e-10` at
  `test_biotsavart_jax.py:520`; Wave 2 should migrate it to
  `parity_ladder_tolerances("direct_kernel")` while adding the new rows,
  so the C++ parity block has one tolerance SSOT.
- **Confirmed missing direct C++ parity** (no co-import of the
  corresponding `simsopt.field.BiotSavart` Python method or `simsoptpp`
  symbol with the JAX path at the lane tolerance):
  - `dA_by_dX` (M-1 mirror in `jax_cpp_parity_test_gaps.md`).
  - `d2B_by_dXdX` (M-2 in `jax_cpp_parity_test_gaps.md`).
  - `d2A_by_dXdX` (M-2 in `jax_cpp_parity_test_gaps.md`).
- **Analytic / Taylor coverage already present** in
  `tests/field/test_biotsavart_jax_parity.py::TestBiotSavartParitySuite`
  (`:229`): `dB_by_dX` Taylor (`:316`), `dA_by_dX` Taylor (`:293`),
  `d2B_by_dXdX` symmetry + Taylor (`:367,377,400`), `B_vjp` channel-wise
  Taylor (`:415`), aggregate current-linearity exact-FD across `B`,
  `dB`, `A`, `dA`, `d2B`, `d2A` (`:490`, using helper
  `_assert_current_linearity` at `:205`). These are oracle type 3
  (FD/Taylor) — they do not satisfy
  `derivative_heavy.requires_direct_cpp_oracle` on their own. The
  aggregate current-linearity test does NOT validate the per-coil
  decomposition that `dB_by_dcoilcurrents` exposes; that decomposition
  is what Wave 2 must additionally test.
- **Tolerance lane mapping** from
  `benchmarks/validation_ladder_contract.py::PARITY_LADDER_TOLERANCES`
  (validated 2026-05-14):
  - `dB_by_dcoilcurrents`, `dA_by_dcoilcurrents` (per-coil B/A at unit
    current) — `direct_kernel` (`rtol=1e-10`, `atol=1e-12`).
  - `d2B_by_dXdcoilcurrents`, `d2A_by_dXdcoilcurrents`,
    `d3B_by_dXdXdcoilcurrents`, `d3A_by_dXdXdcoilcurrents` — these are
    per-coil unit-current B/A spatial-derivative bundles. Use
    `derivative_heavy` (`first_derivative_*` for `d{B,A}_by_dXdc`,
    `second_derivative_*` for `d{B,A}_by_dXdXdc`).
  - `d2B_by_dXdX`, `d2A_by_dXdX`, `dA_by_dX` parity — use
    `derivative_heavy.first_derivative_*` for `dA_by_dX` and
    `derivative_heavy.second_derivative_*` for `d2B_by_dXdX` /
    `d2A_by_dXdX`.
- **External authority checks (2026-05-14):**
  - JAX documentation models `jax.jacfwd` as pushing a Euclidean basis
    through JVPs and transposing the result, and documents `vmap` as
    stacking mapped outputs. That matches the local raw `jacfwd^2` axis
    analysis in `jax_core/biotsavart.py:462-476`.
  - JAX `shard_map` documentation describes rank-preserving block mapping
    with output assembly controlled by `out_specs`. Therefore the grouped
    d2B helper must have an explicit 4D output-spec path when point-axis
    sharding is active.
  - SIMSOPT latest user docs list the full current-derivative ladder
    (`d{B,A}_by_dcoilcurrents`, `d2{B,A}_by_dXdcoilcurrents`,
    `d3{B,A}_by_dXdXdcoilcurrents`) as part of the CPU `BiotSavart`
    public API; the current tree's `src/simsopt/field/biotsavart.py`
    is the authoritative shape/line contract for this fork.
  - NVIDIA's CUDA Programming Guide documents CPU/GPU floating-point
    differences around fused multiply-add and dot-product accumulation.
    Wave 2 remains a CPU-only C++/JAX parity slice; do not claim CUDA
    parity from these tests. Any GPU/CUDA signoff belongs in a later
    `gpu_runtime` lane with deterministic-XLA provenance.

#### Files In Scope

- `src/simsopt/field/biotsavart_jax_backend.py` — expose the seven
  missing methods on both `BiotSavartJAX` (method block `:1439-:1634`)
  and `SpecBackedBiotSavartJAX` (method block `:511-:545`). The two
  classes do not inherit from each other; both surface the same public
  API and both must remain in sync.
- `src/simsopt/jax_core/field.py` — add `grouped_biot_savart_d2B_by_dXdX`
  / `grouped_biot_savart_d2B_by_dXdX_from_spec` /
  `grouped_biot_savart_d2B_by_dXdX_from_inputs` mirroring the d2A trio
  at `:531-:540`; import `biot_savart_d2B_by_dXdX` next to
  `biot_savart_d2A_by_dXdX` (`jax_core/field.py:18`); extend the
  `_empty_grouped_field_result` kernel dispatch (`:52-67`) and the
  `_field_out_specs` sharding dispatch (`:113-125`) so the 4D Hessian
  branch explicitly covers `biot_savart_d2B_by_dXdX` instead of relying
  on the current catch-all 4D return.
- `src/simsopt/jax_core/__init__.py` — export the new
  `grouped_biot_savart_d2B_by_dXdX_from_spec` and
  `grouped_biot_savart_d2B_by_dXdX_from_inputs` helpers next to the
  existing d2A exports, so public `simsopt.jax_core` imports stay
  consistent.
- `src/simsopt/jax_core/biotsavart.py` — no kernel changes required;
  `biot_savart_d2B_by_dXdX` (`:585`) is already in `__all__` (`:40`).
- `tests/field/test_biotsavart_jax.py` — extend
  `TestBiotSavartJaxCppParity` (`:497`) with the missing parity rows
  using the existing `_ncsx_biotsavart_parity_fixture` (`:341`) whose
  5-tuple return is `(bs, points_np, gammas_np, gds_np, currents_np)`.
- `tests/field/test_biotsavart_jax_parity.py` — add per-coil
  current-linearity tests using the existing
  `_assert_current_linearity` helper template (`:205`) — see W2-B6.
- `.artifacts/jax_port_gap_audit_2026-05-13/STATUS.md` — append a Wave 2
  outcome section.

#### Out Of Scope

- Re-implementing `A`, `dA_by_dX`, `d2A_by_dXdX`, `B_vjp`, `A_vjp`,
  `B_and_dB_vjp`, `A_and_dA_vjp` (all already present).
- Rewriting the `TestBiotSavartJaxChunkedSelfConsistency` self-consistency
  probes — Wave 1 already classified them.
- Adding a `BiotSavartJAX.compute(derivatives=N)` batched cache entrypoint;
  the cpp_port_gap audit classifies the bundled cache fill as
  NON-PORTABLE-by-design (per-method calls are the JAX-native shape).
- `WireframeField::compute(derivatives=2)` Hessian — that lives in a
  later wave and uses a different kernel.
- Mixed CPU-JAX `MagneticFieldSum` composition (covered by Wave 6).

#### Workstreams

- [ ] **W2-B0 - Current-tree revalidation (small)**
  - [ ] Re-grep `biotsavart_jax_backend.py` for the seven method names
        in the gap list and confirm absence on both `BiotSavartJAX`
        (`:878`) and `SpecBackedBiotSavartJAX` (`:405`).
  - [ ] Re-grep `jax_core/field.py` for `grouped_biot_savart_d2B_by_dXdX`
        and confirm absence (and that `biot_savart_d2B_by_dXdX` is not
        imported there).
  - [ ] Confirm `_ncsx_biotsavart_parity_fixture` (`:341`) still
        returns a 5-tuple `(bs, points_np, gammas_np, gds_np,
        currents_np)` from a `simsoptpp`-backed
        `simsopt.field.BiotSavart`.
  - [ ] Append the verified gap list to
        `.artifacts/jax_port_gap_audit_2026-05-13/STATUS.md` before
        starting W2-B1.

- [ ] **W2-B1 - Expose `d2B_by_dXdX()` on both wrapper classes
      (smallest fix first)**
  - [ ] In `simsopt/jax_core/field.py`: import
        `biot_savart_d2B_by_dXdX` next to the existing
        `biot_savart_d2A_by_dXdX` import (`:18`).
  - [ ] Extend `_empty_grouped_field_result` (`:52-67`) with a single
        new branch `if kernel is biot_savart_d2B_by_dXdX: return
        _zeros_float64((point_count, 3, 3, 3))` — mirror the d2A
        branch (`:58`).
  - [ ] Extend `_field_out_specs` (`:113-125`) so
        `biot_savart_d2A_by_dXdX` and `biot_savart_d2B_by_dXdX` share
        an explicit `(point_axis_name, None, None, None)` sharding
        entry. The current default path happens to return that spec for d2A,
        but Wave 2 should make the Hessian kernels explicit.
  - [ ] Add `grouped_biot_savart_d2B_by_dXdX_from_spec` /
        `_from_inputs` mirroring the d2A trio at `:531-:540` — single
        call into `_accumulate_grouped_field(points, coil_spec,
        biot_savart_d2B_by_dXdX)`. No new low-level integrand.
  - [ ] In `biotsavart_jax_backend.py`: add the new grouped helper to
        the import block (`:36-50`) and add `d2B_by_dXdX()` methods on
        both `SpecBackedBiotSavartJAX` (next to `d2A_by_dXdX` at
        `:529`) and `BiotSavartJAX` (next to `d2A_by_dXdX` at `:1458`).
        Each method body is a single call into the new grouped helper
        with `(self._points_jax, self.coil_set_spec())`.
  - [ ] In `simsopt/jax_core/__init__.py`, add matching import and
        `__all__` entries for the new grouped d2B helpers.
  - [ ] Tensor convention: the unit JAX kernel at
        `jax_core/biotsavart.py:462-476` transposes raw `jacfwd²` axes
        `(component, d1, d2)` to `(d1, d2, component)`, then vmaps
        over points to produce shape `(npoints, 3, 3, 3)` with
        `d2B[p, j, k, l] = ∂_j ∂_k B_l(x_p) = ∂_k ∂_j B_l(x_p)`. This
        matches the CPU C++ pybind docstring
        (`simsoptpp/python_magneticfield.cpp:31`,
        "`\partial_k\partial_j B_l(x_i)`") by Schwarz symmetry — no
        runtime reordering needed at the JAX/CPU boundary.
  - [ ] Mixed-quadrature grouping (TF + banana coexistence) flows
        through the existing `_accumulate_grouped_field` path used by
        the d2A helper. No new grouping code needed.

- [ ] **W2-B2 - B-side coil-current derivatives**
  - [ ] Implement `BiotSavartJAX.dB_by_dcoilcurrents(compute_derivatives=0)`
        as a per-coil unit-current B-field bundle. Signature,
        Python-list structure, per-entry shape `(npoints, 3)`, and
        per-coil ordering match
        `simsopt.field.biotsavart.BiotSavart.dB_by_dcoilcurrents`
        (`biotsavart.py:30`).
  - [ ] Implement `d2B_by_dXdcoilcurrents(compute_derivatives=1)` —
        Python list with `(npoints, 3, 3)` entries, matching
        `biotsavart.py:40`.
  - [ ] Implement `d3B_by_dXdXdcoilcurrents(compute_derivatives=2)` —
        Python list with `(npoints, 3, 3, 3)` entries, matching
        `biotsavart.py:50`.
  - [ ] Implementation: build each list entry `b_k`, `db_k`, `d2b_k`
        by calling the existing per-point kernels
        (`biot_savart_B` / `biot_savart_dB_by_dX` /
        `biot_savart_d2B_by_dXdX`) with a single-coil input and
        `currents = jnp.array([1.0])`. The math identity
        `dB/dI_k = b_k(x)` follows from
        `B(x) = Σ_k I_k · b_k(x)` (already validated as exact in
        `test_B_and_dB_linearity_in_current` at
        `test_biotsavart_jax_parity.py:490`). Mirror this construction
        on `SpecBackedBiotSavartJAX`.
  - [ ] `compute_derivatives` argument: accept for signature parity
        with the CPU class. The JAX path has no fieldcache, so the
        argument value has no runtime effect; do not branch on it and
        do not validate it (per `CLAUDE.md` "no defensive checks").
        Document this in the docstring.
  - [ ] No `simsoptpp` imports in source modules.

- [ ] **W2-B3 - A-side coil-current derivatives**
  - [ ] Implement `dA_by_dcoilcurrents(compute_derivatives=0)` —
        Python list with `(npoints, 3)` entries, matching
        `biotsavart.py:132`. Body:
        per-coil `biot_savart_A(points, [γ_k], [γ'_k], [1.0])`.
  - [ ] Implement `d2A_by_dXdcoilcurrents(compute_derivatives=1)` —
        Python list with `(npoints, 3, 3)` entries, matching
        `biotsavart.py:142`. Body:
        per-coil `biot_savart_dA_by_dX(...)` at unit current.
  - [ ] Implement `d3A_by_dXdXdcoilcurrents(compute_derivatives=2)` —
        Python list with `(npoints, 3, 3, 3)` entries, matching
        `biotsavart.py:152`. Body:
        per-coil `biot_savart_d2A_by_dXdX(...)` at unit current.
  - [ ] Same `compute_derivatives` semantics as W2-B2.

- [ ] **W2-B4 - Direct C++ parity (closeout for already-existing
      methods)**
  - [ ] Replace the existing inline `rtol=1e-10` in
        `test_B_parity_ncsx` (`test_biotsavart_jax.py:520`) with
        `parity_ladder_tolerances("direct_kernel")` values. Keep
        self-consistency tests' intentionally inline tight floors
        separate from C++ parity-lane tolerances.
  - [ ] Add `test_dA_by_dX_parity_ncsx` to `TestBiotSavartJaxCppParity`
        comparing `BiotSavartJAX.dA_by_dX()` against
        `BiotSavart.dA_by_dX()` at the
        `derivative_heavy.first_derivative` lane.
  - [ ] Add `test_d2B_by_dXdX_parity_ncsx` comparing
        `BiotSavartJAX.d2B_by_dXdX()` against
        `BiotSavart.d2B_by_dXdX()` at the
        `derivative_heavy.second_derivative` lane (depends on W2-B1).
  - [ ] Add `test_d2A_by_dXdX_parity_ncsx` comparing
        `BiotSavartJAX.d2A_by_dXdX()` against
        `BiotSavart.d2A_by_dXdX()` at the same lane.
  - [ ] Each test must co-import a `simsoptpp`-backed `BiotSavart` and
        cite the oracle type per
        `tests/REVIEWER_ORACLE_LINT.md`. Reuse
        `_ncsx_biotsavart_parity_fixture` rather than building a
        parallel fixture. Tolerances must come from
        `PARITY_LADDER_TOLERANCES` via
        `parity_ladder_tolerances(...)` (matching the existing
        `_DERIVATIVE_HEAVY_TOLS` usage at `test_biotsavart_jax.py`).
        No inline `rtol=`/`atol=` literals.

- [ ] **W2-B5 - Direct C++ parity (coil-current ladder)**
  - [ ] Add `test_dB_by_dcoilcurrents_parity_ncsx` comparing the JAX
        list-of-arrays against `BiotSavart.dB_by_dcoilcurrents()` on
        identical coils/points at the `direct_kernel` lane (per-coil
        unit-current B is a value-like quantity).
  - [ ] Add the symmetric `test_dA_by_dcoilcurrents_parity_ncsx`.
  - [ ] Add `test_d2B_by_dXdcoilcurrents_parity_ncsx` and
        `test_d2A_by_dXdcoilcurrents_parity_ncsx` at
        `derivative_heavy.first_derivative`.
  - [ ] Add `test_d3B_by_dXdXdcoilcurrents_parity_ncsx` and
        `test_d3A_by_dXdXdcoilcurrents_parity_ncsx` at
        `derivative_heavy.second_derivative`.
  - [ ] Each test must compare list element-by-element (preserving
        per-coil ordering — the CPU list ordering is the live coil
        ordering from `BiotSavart._coils`) and reuse
        `_ncsx_biotsavart_parity_fixture`. Use
        `len(jax_list) == len(cpu_list)` as the structural check, then
        per-index `assert_allclose`.

- [ ] **W2-B6 - Per-coil current-linearity FD coverage**
  - [ ] In `test_biotsavart_jax_parity.py`, add per-coil
        current-linearity tests for `dB_by_dcoilcurrents` and
        `dA_by_dcoilcurrents`. For each coil k, perturb only that
        coil's current by `+ε` and `-ε` (with all other currents fixed).
        Verify both equivalent exact-linear identities:
        `(B(I_k+ε)-B(I_k-ε))/(2ε) == dB_by_dcoilcurrents[k]` and
        `B(I_k+ε)-B(I_k) == ε * dB_by_dcoilcurrents[k]` (same for `A`).
        Biot-Savart is exactly linear in I, so a single FD step is exact
        to machine precision (no convergence series needed). This mirrors
        the helper pattern in `_assert_current_linearity` (`:205`), and
        corresponds to the upstream test
        `test_biotsavart_coil_current_taylortest`
        (`simsopt/tests/field/test_biotsavart.py:276`) and its
        vector-potential analog (`:402`).
  - [ ] These are type-3 (FD-on-the-JAX-stack) oracles, distinct from
        W2-B5's direct C++ list-equality oracles. Do not present them
        as parity oracles.
  - [ ] The existing aggregate `test_B_and_dB_linearity_in_current`
        (`:490`) covers Σ_k linearity but does NOT validate the
        per-coil decomposition — keep both.

#### Validation

- [ ] `ruff check` and `ruff format --check` on every changed
      Python file.
- [ ] `.conda/jax-0.9.2/bin/python -m pytest tests/field/test_biotsavart_jax.py tests/field/test_biotsavart_jax_parity.py tests/field/test_biotsavart_A_direct_kernel_closeout.py tests/test_jax_import_smoke.py -k 'biotsavart or grouped_biot_savart' -v`
- [ ] Public pure-JAX command from `CLAUDE.md` if any shared
      helper (kernel-cache, grouped wrapper, or `jax_core/__init__.py`
      public export) is touched.
- [ ] No `tests/integration/` run is required unless a Stage 2 path is
      observed to regress.

#### Acceptance Criteria

The Wave 2 `/goal` is complete when:

1. `BiotSavartJAX` and `SpecBackedBiotSavartJAX` both expose a
   `d2B_by_dXdX()` method whose value matches
   `BiotSavart.d2B_by_dXdX()` on the NCSX parity fixture at the
   `derivative_heavy.second_derivative` lane. `jax_core/field.py`
   exposes `grouped_biot_savart_d2B_by_dXdX_from_spec` /
   `_from_inputs`; `jax_core/__init__.py` re-exports both helpers; and
   `_empty_grouped_field_result` / `_field_out_specs` recognize the new
   kernel.
2. `BiotSavartJAX` and `SpecBackedBiotSavartJAX` both expose the six
   coil-current derivative methods
   (`{d,d2,d3}{B,A}_by_d[X[X]]dcoilcurrents`) with CPU-matching
   signatures, default `compute_derivatives` arguments, Python-list
   return structure, per-entry shapes, and per-coil ordering. The JAX
   methods return per-coil JAX arrays, not host-materialized NumPy
   arrays; the `compute_derivatives` argument is accepted but not
   branched on.
3. `tests/field/test_biotsavart_jax.py::TestBiotSavartJaxCppParity`
   contains direct-C++ parity rows for `dA_by_dX`, `d2B_by_dXdX`,
   `d2A_by_dXdX`, and all six coil-current derivative methods, each
   citing its oracle type and using `PARITY_LADDER_TOLERANCES` entries
   via `parity_ladder_tolerances(...)`; the pre-existing `B` and
   `dB_by_dX` rows also use the same SSOT lane constants. No inline
   tolerance literals remain in direct C++ parity rows.
4. `tests/field/test_biotsavart_jax_parity.py` carries per-coil
   current-linearity coverage for `dB_by_dcoilcurrents` and
   `dA_by_dcoilcurrents` — distinct from the W2-B4/W2-B5 direct-C++
   parity rows. The aggregate `test_B_and_dB_linearity_in_current`
   (`:490`) remains green.
5. Existing `A`/`dA_by_dX`/`d2A_by_dXdX`/`B`/`dB_by_dX`/`B_vjp` parity
   tests remain green; existing chunked self-consistency tests
   (`TestBiotSavartJaxChunkedSelfConsistency`, `:578`) remain green;
   existing Taylor invariants remain green. No `simsoptpp` import is
   introduced into any `src/simsopt/**` module.
6. `STATUS.md` records the verified gap list (pre-implementation), the
   landed method/test changes, and the focused validation outcomes.

### Wave 3 - MHD Reductions

Status: follow-on wave. Spot-checked 2026-05-14 at HEAD `d773344d1`.
This wave is a pure-reduction MHD slice, not a VMEC/SPEC/BOOZXFORM runner
port. The current tree has no `src/simsopt/mhd/*_jax.py` module and no
`tests/mhd/*jax*` test file. Existing JAX surface objectives such as
`IotasJAX` and `NonQuasiSymmetricRatioJAX` are in
`src/simsopt/geo/surfaceobjectives_jax.py`; they do not close the MHD
post-processing classes in `src/simsopt/mhd`.

#### Current-Tree Corrections

- `src/simsopt/mhd/boozer.py:244` defines `Quasisymmetry`; its live CPU
  mirror test is `tests/mhd/test_boozer.py:83`.
- `src/simsopt/mhd/vmec_diagnostics.py:32` defines
  `QuasisymmetryRatioResidual`; `:373`, `:486`, and `:595` define
  `IotaTargetMetric`, `IotaWeighted`, and `WellWeighted`; related tests live
  in `tests/mhd/test_vmec_diagnostics.py:35,97,264,323,379`.
- `src/simsopt/mhd/bootstrap.py:27,173,405,517,635` defines the trapped
  fraction / Redl bootstrap pipeline. This is larger than the first MHD
  reduction slice because it mixes SciPy splines, optimization, profile
  objects, and geometry containers.
- `vmec.run()`, `boozer.run()`, BOOZXFORM output generation, VMEC adjoint
  re-runs in `IotaTargetMetric.dJ()` / `IotaWeighted.dJ()` /
  `WellWeighted.dJ()`, NetCDF reads, MPI grouping, and plotting remain
  non-portable boundaries for this wave.
- The JAX port target is frozen-array payload reduction: consume arrays from
  CPU VMEC/Boozer fixtures and evaluate the same algebra on JAX. Do not move
  external executables or file IO into the JAX hot path.

#### Files In Scope

- `src/simsopt/jax_core/mhd_reductions.py` or a similarly named pure-kernel
  module for frozen VMEC/Boozer payload reductions.
- `src/simsopt/mhd/boozer_jax.py` only if a thin public adapter is needed
  around `Quasisymmetry`.
- `src/simsopt/mhd/vmec_diagnostics_jax.py` only if thin public adapters are
  needed around VMEC-diagnostic reductions.
- `tests/mhd/test_boozer_jax.py`
- `tests/mhd/test_vmec_diagnostics_jax.py`
- pinned fixtures under `tests/test_files/` only if existing fixtures are
  insufficient

#### Out Of Scope

- Running VMEC, SPEC, BOOZXFORM, `virtual_casing`, `pyoculus`, or MPI.
- Porting `Vmec`, `Spec`, `Boozer`, `VirtualCasing`, or file readers.
- Shape-gradient / adjoint rerun implementations for `dJ()` unless a current
  tree revalidation proves they can be expressed from already-frozen arrays.
- Redl bootstrap geometry and profile-object ports. Record them as W3-deferred
  unless this wave is intentionally split before implementation.

#### Workstreams

- [ ] **W3-M0 - Revalidate MHD targets**
  - [ ] Locate current `Quasisymmetry`, `QuasisymmetryRatioResidual`,
        `IotaTargetMetric`, `IotaWeighted`, and `WellWeighted` definitions.
  - [ ] Confirm there are still no `src/simsopt/mhd/*_jax.py` modules and no
        MHD-specific JAX tests.
  - [ ] Distinguish pure `.J()` reductions from adjoint rerun `.dJ()` paths:
        `.J()` reductions are candidate JAX hot paths; VMEC rerun shape
        gradients are out of this wave by default.
  - [ ] Identify the smallest existing VMEC/Boozer fixture that can pin
        deterministic outputs.
  - [ ] Append the verified live gap list and any deferred Redl/bootstrap rows
        to `STATUS.md` before implementation.

- [ ] **W3-M1 - `Quasisymmetry` and `QuasisymmetryRatioResidual`**
  - [ ] For Boozer `Quasisymmetry`, freeze the required `bx` payload
        (`bmnc_b`, `xm_b`, `xn_b`, `nfp`, `s_to_index`) into arrays and port
        the mode-selection, normalization (`B00`, `symmetric`), and weight
        modes (`even`, `stellopt`, `stellopt_ornl`) without calling
        `boozer.run()` inside the JAX kernel.
  - [ ] For VMEC `QuasisymmetryRatioResidual`, split the code into
        CPU-only geometry extraction and a pure residual reducer over frozen
        arrays. Do not port `vmec_compute_geometry` unless W3-M0 proves it is
        small enough for the same PR.
  - [ ] Add CPU/NumPy value parity on pinned payloads for at least QA,
        QH, and QP mode-selection cases where fixture data supports them.
  - [ ] If gradients are added, use FD/Taylor against the JAX value path or an
        explicit CPU derivative contract. Do not claim gradient parity from
        JAX-vs-JAX autodiff alone.

- [ ] **W3-M2 - `IotaTargetMetric`, `IotaWeighted`, `WellWeighted`**
  - [ ] Extract pure `.J()` reducers over `s_half_grid`, `ds`, `wout.iotas`,
        `wout.vp`, and user-supplied weight/target arrays.
  - [ ] Keep user-callable target/weight functions on the host boundary:
        evaluate them to arrays before entering the JAX reduction.
  - [ ] Preserve the CPU object contracts in adapters only if adapters are
        added. A pure-kernel-only slice is acceptable if it is exported and
        documented as such.
  - [ ] Mark `shape_gradient()` and `.dJ()` as deferred unless implemented
        with a frozen-array contract and an independent oracle.

- [ ] **W3-M3 - Redl/bootstrap triage**
  - [ ] Revalidate `compute_trapped_fraction`, `j_dot_B_Redl`,
        `RedlGeomVmec`, `RedlGeomBoozer`, and `VmecRedlBootstrapMismatch`.
  - [ ] Split Redl work into a separate wave if it still requires SciPy
        splines/minimize/quad or profile-object adapters.
  - [ ] If any scalar Redl algebra is ported here, test it against direct
        CPU function calls on array fixtures and keep geometry extraction out
        of JAX scope.

- [ ] **W3-M4 - Shared MHD validation**
  - [ ] Centralize fixture setup if both Boozer and VMEC diagnostics need the
        same VMEC output.
  - [ ] Add import-smoke coverage for any new public MHD JAX module.
  - [ ] Document every non-portable Fortran/MPI/file-IO boundary in
        `STATUS.md`.

#### Validation

- [ ] `.conda/jax-0.9.2/bin/python -m pytest tests/mhd/test_boozer*.py tests/mhd/test_vmec_diagnostics*.py -v`
- [ ] Run focused import-smoke tests if new modules are exported.
- [ ] Run the public pure-JAX command from `CLAUDE.md` only if new modules are
      imported by the supported JAX surface/objective path.

#### Acceptance Criteria

The Wave 3 `/goal` is complete when:

1. Every implemented MHD reducer has a CPU/NumPy oracle test on pinned
   payloads, and every test states whether the oracle is CPU object output,
   frozen-array NumPy algebra, or FD/Taylor.
2. VMEC/SPEC/BOOZXFORM execution, MPI grouping, NetCDF reads, plotting, and
   adjoint reruns are outside the JAX hot path and documented as such.
3. Redl/bootstrap rows are either explicitly deferred with blockers or split
   into their own current-tree implementation plan.
4. `STATUS.md` records the verified MHD gap list, files changed, and validation
   outcomes.

### Wave 4 - Curve Geometry And Objectives

Status: follow-on wave. Spot-checked 2026-05-14 at HEAD `d773344d1`.
This wave closes curve-core API normalization and the highest-value missing
curve objective mirrors. It should not duplicate already working
`CurveXYZFourierSymmetries`, `FramedCurve`, linking-number, or distance
candidate coverage.

#### Current-Tree Corrections

- `src/simsopt/geo/curve.py:213,229,259` still owns
  `incremental_arclength_pure`, `kappa_pure`, and `torsion_pure`; their
  grad/VJP helpers sit nearby (`kappagrad*`, `kappavjp*`, torsion VJPs).
  These are pure JAX functions but are not first-class `jax_core` exports.
- `src/simsopt/jax_core/curve_geometry.py:300` already has
  `_curve_geometry_with_third_derivative_from_dofs`, including a
  `CurvePerturbedSpec` path. Revalidate the actual supported spec set before
  adding any third-derivative code; do not assume it is XYZFourier-only.
- Existing current-tree curve JAX tests include
  `tests/geo/test_curve_item05_closeout.py`,
  `tests/geo/test_curvexyzfouriersymmetries_spec_jax.py`,
  `tests/geo/test_framedcurve_jax_item18.py`,
  `tests/geo/test_framedcurve_jax_wrappers_item18.py`,
  `tests/geo/test_distance_jax.py`, and `tests/geo/test_linking_number_jax.py`.
  There is still no `tests/geo/test_curve_jax.py` or
  `tests/geo/test_curve_objectives_jax.py`.
- The high-value test-mirror gaps from
  `tests/geo/test_curve_objectives.py:634-911` are FD/Taylor mirrors for
  `CurveLength`, `LpCurveCurvature`, `LpCurveCurvatureBarrier`,
  `LpCurveTorsion`, `CurveCurveDistance`, `CurveCurveDistanceBarrier`,
  `ArclengthVariation`, and `MeanSquaredCurvature`.

#### Files In Scope

- `src/simsopt/jax_core/curve_geometry.py`
- `src/simsopt/geo/curve.py`
- `src/simsopt/geo/curveobjectives.py`
- `tests/geo/test_curve_item05_closeout.py`
- `tests/geo/test_curveobjectives_item07_closeout.py` or a new
  `tests/geo/test_curve_objectives_jax.py` if W4-C0 chooses a dedicated file
- `tests/geo/test_curveperturbed.py` only as a CPU reference source; create a
  JAX mirror instead of editing CPU tests unless the CPU fixture needs a helper

#### Out Of Scope

- VTK, plotting, serialization, MAKEGRID loading, and CPU-only DOF-name tests.
- Rewriting `CurveCWSFourier` surface-bound machinery unless W4-C0 proves the
  XYZTensor branch is the smallest live blocker.
- Replacing `FramedCurveTwist` / `LinkingNumber` coverage that already exists.
- Broad curve-objective refactors unrelated to missing oracle mirrors.

#### Workstreams

- [ ] **W4-C0 - Revalidate curve geometry location**
  - [ ] Locate `kappa`, `torsion`, `incremental_arclength`, and their gradient
        / VJP helpers in the current tree.
  - [ ] Identify any import cycle risk between `simsopt.geo.curve` and
        `simsopt.jax_core`.
  - [ ] Revalidate supported spec types for
        `_curve_geometry_with_third_derivative_from_dofs`:
        `CurveXYZFourierSpec`, `CurveRZFourierSpec`,
        `CurvePlanarFourierSpec`, `CurveHelicalSpec`,
        `CurveXYZFourierSymmetriesSpec`, `CurvePerturbedSpec`, and
        `CurveCWSFourierRZSpec`.
  - [ ] Decide whether the needed action is move, re-export, named wrapper, or
        no-op; record the decision in `STATUS.md`.

- [ ] **W4-C1 - Lift or normalize curve geometry API**
  - [ ] Move only missing pure functions into `jax_core/curve_geometry.py`.
  - [ ] Keep backward-compatible imports from `geo/curve.py`.
  - [ ] Avoid dynamic imports. If a cycle blocks the move, document the root
        cycle and stop.
  - [ ] Add named wrappers for public curve geometry quantities and VJP/grad
        surfaces that external JAX consumers need (`incremental_arclength`,
        `kappa`, `torsion`, their DOF gradients/VJPs), instead of forcing
        callers to reach into `jax.vjp` closures.
  - [ ] Do not remove the CPU `sopp.Curve` trampoline paths unless every CPU
        subclass using them has a current-tree JAX replacement.

- [ ] **W4-C2 - Curve-objective FD-Taylor mirrors**
  - [ ] Revalidate missing mirrors for `CurveLength`,
        `LpCurveCurvature`, `LpCurveCurvatureBarrier`, `LpCurveTorsion`,
        `CurveCurveDistance`, `CurveCurveDistanceBarrier`,
        `ArclengthVariation`, and `MeanSquaredCurvature`.
  - [ ] Add type-3 FD-Taylor tests against the JAX value path.
  - [ ] Add CPU value parity where the CPU object is independent and not a
        wrapper around the same JAX kernel.
  - [ ] Preserve or extend existing `FramedCurveTwist` / `LinkingNumber`
        closeout coverage rather than duplicating it.
  - [ ] For pairwise distance objectives, keep chunked-vs-dense self-consistency
        separate from value-gradient Taylor validation.

- [ ] **W4-C3 - C++ curve VJP parity**
  - [ ] Revalidate `kappa_by_dcoeff_vjp` and related VJP coverage.
  - [ ] Add C++ oracle parity only for still-missing VJP surfaces.
  - [ ] Use `derivative_heavy` tolerances.
  - [ ] If no direct C++ VJP symbol exists for a curve wrapper, classify the
        test as FD/Taylor or CPU-wrapper parity; do not label it direct C++.

- [ ] **W4-C4 - CurvePerturbed and surface-bound curve triage**
  - [ ] Revalidate `CurvePerturbed` gaps from `tests/geo/test_curveperturbed.py`
        and decide whether they are real JAX gaps or CPU fixture smoke.
  - [ ] Revalidate `CurveCWSFourier` RZ versus XYZTensor behavior. The RZ spec
        path exists; the XYZTensor host-surface branch should be deferred unless
        needed by the current production lane.
  - [ ] Keep any deferred surface-bound curve item tied to Wave 5 surface scope.

#### Validation

- [ ] `.conda/jax-0.9.2/bin/python -m pytest tests/geo/test_curve.py tests/geo/test_curve_objectives.py tests/geo/test_curve_item05_closeout.py tests/geo/test_curveobjectives_item07_closeout.py tests/geo/test_curvexyzfouriersymmetries_spec_jax.py tests/geo/test_framedcurve_jax_item18.py tests/geo/test_framedcurve_jax_wrappers_item18.py -v`
- [ ] Add any new dedicated JAX curve-objective mirror file to the command.
- [ ] Run `tests/test_jax_import_smoke.py -k jax_core` if exports in
      `simsopt.jax_core` or `simsopt.geo` change.

#### Acceptance Criteria

The Wave 4 `/goal` is complete when:

1. Curve geometry helpers have a single JAX-core ownership story: either moved,
   re-exported, or explicitly kept in `geo/curve.py` with a documented reason.
2. Every new curve objective mirror cites its oracle type and is not
   JAX-vs-JAX or wrapper-vs-kernel tautology.
3. Missing third-derivative / VJP surfaces are either implemented across the
   verified spec set or deferred with exact unsupported spec names.
4. Existing curve closeout tests remain green and import paths stay acyclic.
5. `STATUS.md` records implemented mirrors, deferred curve rows, and validation
   outcomes.

### Wave 5 - Surfaces

Status: follow-on wave. Spot-checked 2026-05-14 at HEAD `d773344d1`.
This wave should not assume all surface audit rows are still missing. The
current tree already has broad RZ/XYZ/XYZTensor/Henneberg/Garabedian JAX
coverage, object-API parity for several host constructors, and generic
surface-curvature helpers in `surfaceobjectives_jax.py`.

#### Current-Tree Corrections

- `tests/geo/test_surface_fourier_jax.py:1552-1553` already exercises
  `extend_via_normal` and `extend_via_projected_normal` object-API parity for
  `SurfaceXYZFourier` and `SurfaceXYZTensorFourier`.
- `tests/geo/test_surface_rzfourier_jax.py:1480` already exercises
  `SurfaceRZFourier.extend_via_normal` object-API parity.
- `src/simsopt/geo/surfaceobjectives_jax.py:411,419` provides generic
  `surface_curvatures_jax_from_dofs` and
  `surface_dsurface_curvatures_jax_from_dofs`; tests at
  `tests/geo/test_surface_objectives_jax.py:5552,5573` sweep surface types. Do
  not reintroduce the stale claim that XYZ/XYZTensor curvatures are absent
  without re-grepping this path.
- RZ fundamental-form helpers remain explicitly named in
  `src/simsopt/jax_core/surface_rzfourier.py:872-914` and are tested in
  `tests/geo/test_surface_rzfourier_jax.py:254-292,673-687`.
- `SurfaceRZFourier` still exposes CPU third-parametric-derivative `_lin`
  methods at `src/simsopt/geo/surfacerzfourier.py:402-417`; no matching JAX
  helper names were found in the spot check.

#### Files In Scope

- `src/simsopt/geo/surface_fourier_jax.py`
- `src/simsopt/jax_core/surface_rzfourier.py`
- `src/simsopt/jax_core/surface_fourier.py` if direct XYZ/XYZTensor
  fundamental-form helpers are still needed after W5-S0
- `src/simsopt/geo/surface_bootstrap_jax.py` only if W5-S0 proves a real
  bootstrap-kernel gap remains
- `tests/geo/test_surface_fourier_jax.py`
- `tests/geo/test_surface_rzfourier_jax.py`
- `tests/geo/test_surface_objectives_jax.py`
- `tests/geo/test_surface_henneberg_jax.py`
- `tests/geo/test_surface_garabedian_jax.py`
- new surface test files only when the existing files would become too broad

#### Out Of Scope

- Serialization, VTK, VMEC/FOCUS/NESCOIL/PyQSC readers, plotting, and
  CPU-only DOF-name/indexing tests.
- New `SurfaceScaledJAX` or `SurfaceRZPseudospectralJAX` adapters unless W5-S0
  proves they block the supported JAX product surface.
- Replacing CPU object methods that only mutate host DOFs. JAX kernels should
  be pure functions over specs/dofs.
- Duplicating already-passing RZ/XYZ/XYZTensor/Henneberg/Garabedian parity
  rows.

#### Workstreams

- [ ] **W5-S0 - Revalidate surface coverage**
  - [ ] Check current object-API and pure-kernel coverage for `fit_to_curve`,
        `least_squares_fit`, `_extend_via_normal_for_nonuniform_phi`, and
        `extend_via_projected_normal`.
  - [ ] Check current RZ, XYZ, XYZTensor, Henneberg, and Garabedian surface
        coverage.
  - [ ] Re-grep `surfaceobjectives_jax.py` before adding curvature helpers;
        generic curvature reducers may already cover the requested surface
        kind.
  - [ ] Reclassify each original audit row as implemented, stale, deferred, or
        still missing.
  - [ ] Record which missing rows are real in `STATUS.md`.

- [ ] **W5-S1 - Third parametric derivatives and bootstrap kernels**
  - [ ] Add RZ third `_lin` helpers only if they are still required by a live
        QFM/regularization path. Mirror CPU names:
        `gammadash1dash1dash1_lin`, `gammadash1dash1dash2_lin`,
        `gammadash1dash2dash2_lin`, `gammadash2dash2dash2_lin`.
  - [ ] Prefer autodiff over the existing RZ surface basis rather than
        hand-rolled formula duplication.
  - [ ] Port only missing JAX hot-path bootstrap kernels.
  - [ ] Mirror C++ signatures and semantics from `python_surfaces.cpp`.
  - [ ] Add C++ oracle parity tests on pinned fixtures.
  - [ ] Do not add Optimizable wrappers unless the live contract requires them.

- [ ] **W5-S2 - XYZ/XYZTensor fundamental forms and curvatures**
  - [ ] Revalidate RZ-only vs XYZ/XYZTensor gaps for `first_fund_form`,
        `second_fund_form`, `surface_curvatures`, and `_by_dcoeff` variants.
  - [ ] If only named helper exports are missing, add thin wrappers over the
        generic `_surface_geometry_second_derivatives_from_dofs` /
        `surface_curvatures_jax_from_dofs` path rather than new math.
  - [ ] Build on existing `surface_fourier_jax.py` evaluators.
  - [ ] Preserve stellsym DOF conventions from `CLAUDE.md`.
  - [ ] Add C++ oracle tests for both stellsym modes where possible.

- [ ] **W5-S3 - Surface Taylor and Gauss-Bonnet mirrors**
  - [ ] Revalidate current RZ Gauss-Bonnet tests.
  - [ ] Add XYZ/XYZTensor/Henneberg mirrors only for still-uncovered surface
        kinds.
  - [ ] Add FD-Taylor mirrors for `dgamma_by_dphi`, `dgamma_by_dtheta`,
        `d2gamma_by_dtheta2`, `d2gamma_by_dphi2`, and
        `d2gamma_by_dthetaphi` if still unmirrored from
        `tests/geo/test_surface_taylor.py:593-686`.
  - [ ] Keep self-intersection and projected-normal API tests separate from
        curvature scalar tests.

- [ ] **W5-S4 - Surface bootstrap/object API triage**
  - [ ] Confirm whether `fit_to_curve` and `least_squares_fit` are needed as
        JAX kernels or only as CPU initialization helpers.
  - [ ] Keep `extend_via_normal` object-API coverage already present unless a
        pure JAX offset kernel is required by the supported pipeline.
  - [ ] If `SurfaceScaled` is needed, implement it as a pure spec/dof transform
        and test against the CPU object method; otherwise record as deferred.

#### Validation

- [ ] `.conda/jax-0.9.2/bin/python -m pytest tests/geo/test_surface_fourier_jax.py tests/geo/test_surface_rzfourier_jax.py tests/geo/test_surface_objectives_jax.py tests/geo/test_surface_henneberg_jax.py tests/geo/test_surface_garabedian_jax.py -v`
- [ ] Add new focused surface test files to the command as they are created.
- [ ] Run `tests/test_jax_import_smoke.py -k surface` if public exports change.

#### Acceptance Criteria

The Wave 5 `/goal` is complete when:

1. Surface audit rows are reclassified against the live tree; stale curvature
   and projected-normal claims are not carried forward.
2. Every new surface kernel has a direct CPU/C++ or pinned-data oracle and
   states its oracle type.
3. No new surface module duplicates existing working JAX kernels or generic
   helpers.
4. Deferred host-object features (`fit_to_curve`, LSQ bootstrap,
   `SurfaceScaled`, `SurfaceRZPseudospectral`) are documented with exact
   blockers and not silently treated as implemented.
5. `STATUS.md` records implemented rows, stale-pruned rows, deferred rows, and
   validation outcomes.

### Wave 6 - Field Composition And Analytic Gaps

Status: follow-on wave. Spot-checked 2026-05-14 at HEAD `d773344d1`.
This wave is a field-composition and analytic-field closeout. The first-draft
request for new `MagneticFieldSumJAX` / `MagneticFieldMultiplyJAX` classes is
stale unless current evidence proves the existing public classes cannot carry
the contract.

#### Current-Tree Corrections

- `src/simsopt/field/magneticfield.py:12-37` defines strict JAX-native field
  guards and invokes them in `MagneticFieldMultiply.__init__` (`:226`) and
  `MagneticFieldSum.__init__` (`:279`).
- `src/simsopt/jax_core/magneticfield_composition.py` already exposes pure
  JAX composition primitives for `B`, `dB`, `d2B`, `A`, `dA`, and `d2A`
  sums/scales.
- `tests/field/test_magnetic_field_composition_jax.py:153,273` already
  asserts `d2B_by_dXdX` composition parity for sum and multiply. Do not
  reintroduce stale "only first order" wording without revalidating the
  file.
- `tests/field/test_magneticfieldclasses_jax_item15.py:107` already covers
  `ToroidalFieldJAX.d2B_by_dXdX` versus CPU.
- `src/simsopt/field/interpolated_field_jax.py:246,265` exposes
  `GradAbsB_cyl` and `B_cyl` on the interpolated-field wrapper; broader
  `B_cyl` / `A_cyl` / `GradAbsB_cyl` access on arbitrary JAX magnetic fields
  remains a separate revalidation item.

#### Files In Scope

- `src/simsopt/field/magneticfield.py`
- `src/simsopt/jax_core/magneticfield_composition.py`
- `src/simsopt/field/circular_coil_jax.py`
- `src/simsopt/field/dipole_field_jax.py`
- `src/simsopt/field/dommaschk_jax.py`
- `src/simsopt/field/reiman_jax.py`
- `src/simsopt/field/mirror_model_jax.py`
- `src/simsopt/field/poloidal_field_jax.py`
- `src/simsopt/field/scalar_potential_rz_jax.py`
- `tests/field/test_magnetic_field_composition_jax.py`
- `tests/field/test_magneticfieldclasses_jax_item15.py`
- `tests/field/test_circular_coil_jax.py`
- `tests/field/test_dipole_field_jax_item26.py`
- `tests/field/test_scalar_potential_rz_jax_item23.py`

#### Out Of Scope

- Creating new adapter class names just to match the stale audit.
- Porting helical coil wrappers, MGrid, or tracing branches in this wave.
- SymPy-to-JAX printer work for `ScalarPotentialRZMagneticField` unless W6-F0
  explicitly selects that blocked item as the whole wave.
- Relaxing strict mixed CPU/JAX composition guards without a tested public-path
  contract.

#### Workstreams

- [ ] **W6-F0 - Revalidate field-composition surface**
  - [ ] Confirm current strict-mode behavior for all-JAX and mixed CPU/JAX
        compositions.
  - [ ] Confirm pure JAX primitives cover the production composition path.
  - [ ] Do not create separate `MagneticFieldSumJAX` or
        `MagneticFieldMultiplyJAX` classes unless current evidence proves the
        existing class path cannot satisfy the contract.
  - [ ] Re-grep direct tests for `B`, `dB_by_dX`, `d2B_by_dXdX`, `A`,
        `dA_by_dX`, and `d2A_by_dXdX` composition rows before adding coverage.
  - [ ] Record stale-pruned composition claims in `STATUS.md`.

- [ ] **W6-F1 - Composition coverage**
  - [ ] Add missing tests only for uncovered production combinations such as
        BiotSavart + analytic field or nested composition.
  - [ ] Preserve existing strict mixed CPU/JAX guard tests.
  - [ ] Add VJP/gradient coverage only where child fields expose the required
        contract.
  - [ ] Add nested sum/multiply tests only if they exercise a production path
        not already covered by existing flat sum/scale tests.

- [ ] **W6-F2 - `test_BifieldMultiply` / `MagneticFieldMultiply` parity**
  - [ ] Revalidate the CPU `test_BifieldMultiply` mirror and current JAX
        `MagneticFieldMultiply` composition tests. `BifieldMultiply` is a
        legacy test-method name, not a public class to port.
  - [ ] Add missing scalar-multiply parity for the live JAX composition path,
        including non-stellsym, A/dA/d2A, and any still-uncovered derivative
        cases from `tests/field/test_magneticfields.py:921`.
  - [ ] Use a CPU object or closed-form expression as oracle, not JAX-vs-JAX.

- [ ] **W6-F3 - Dommaschk/Reiman analytic mirrors**
  - [ ] Revalidate current `B`, `dB`, `A`, and `d2B_by_dXdX` coverage for
        Dommaschk and Reiman wrappers.
  - [ ] Add Taylor/curl/divergence-free checks only where upstream has a
        corresponding invariant.
  - [ ] Do not force unsupported CPU surfaces into JAX scope.
  - [ ] If `_A_impl` or `_d2B_by_dXdX_impl` remains absent on a wrapper, decide
        whether the CPU class actually exposes the same method. Mark de-facto
        parity when CPU also lacks the method.

- [ ] **W6-F4 - Other analytic-field partials**
  - [ ] Revalidate `CircularCoilJAX` `_A_impl` / `_d2B_by_dXdX_impl`,
        `DipoleFieldJAX` `_d2B_by_dXdX_impl`, `PoloidalFieldJAX`
        `_A_impl` / `_d2B_by_dXdX_impl`, and `MirrorModelJAX`
        `_A_impl` / `_d2B_by_dXdX_impl`.
  - [ ] Add only the methods with a CPU or closed-form contract. Do not fill
        CPU-missing methods with synthetic JAX surfaces.
  - [ ] Extend tests in the existing wrapper-specific files rather than
        creating duplicate suites.

- [ ] **W6-F5 - Cylindrical accessors**
  - [ ] Revalidate `B_cyl`, `A_cyl`, and `GradAbsB_cyl` across
        `BiotSavartJAX`, `CircularCoilJAX`, and analytic wrappers.
  - [ ] Preserve existing `InterpolatedFieldJAX.B_cyl` /
        `GradAbsB_cyl` behavior.
  - [ ] Add cartesian-to-cylindrical adapter tests only for wrappers that
        expose the CPU method.

- [ ] **W6-F6 - ScalarPotentialRZMagneticField**
  - [ ] Decide whether the SymPy-to-JAX printer is required.
  - [ ] If required and out of scope, mark deferred with the exact blocker.
  - [ ] If implemented, add closed-form or CPU-oracle tests for `B`, `dB`, and
        any supported `A`/higher-derivative surfaces.
  - [ ] Existing `tests/field/test_scalar_potential_rz_jax_item23.py` covers
        `B`/`dB` against the CPU SymPy-derived wrapper; do not relabel it as a
        direct C++ oracle.

#### Validation

- [ ] `.conda/jax-0.9.2/bin/python -m pytest tests/field/test_magnetic_field_composition_jax.py tests/field/test_magneticfieldclasses_jax_item15.py tests/field/test_circular_coil_jax.py tests/field/test_dipole_field_jax_item26.py tests/field/test_scalar_potential_rz_jax_item23.py -v`
- [ ] Add wrapper-specific focused tests only if W6-F4/W6-F5/W6-F6 implements
      new methods.
- [ ] Run the public pure-JAX command from `CLAUDE.md` if shared
      `magneticfield.py` composition behavior changes.

#### Acceptance Criteria

The Wave 6 `/goal` is complete when:

1. Composition behavior is tested through the real public
   `MagneticFieldSum` / `MagneticFieldMultiply` path, not invented adapter
   classes.
2. Existing composition d2B coverage remains green, and any new product or
   nested-composition coverage cites an independent CPU or closed-form oracle.
3. Analytic-wrapper partials are either implemented with CPU/closed-form
   parity or deferred with exact CPU-contract evidence.
4. Cylindrical accessor coverage distinguishes generic magnetic-field methods
   from `InterpolatedFieldJAX`-specific support.
5. `STATUS.md` records stale-pruned adapter claims, implemented rows, deferred
   rows, and validation outcomes.

### Wave 7 - Composite And Ancillary Ports

Status: long-tail follow-on wave. Spot-checked 2026-05-14 at HEAD
`d773344d1`. This section is not one PR. It is a queue of PR-sized slices that
need independent `/goal` plans after Waves 1-6 land or are deliberately
skipped.

#### Current-Tree Corrections

- `src/simsopt/field/coilset.py:18,383` still defines CPU
  `CoilSet` / `ReducedCoilSet`; no `src/simsopt/field/coilset_jax.py` was
  found in the spot check.
- `src/simsopt/geo/qfmsurface.py:9,117,147,183` defines CPU
  `QfmSurface` and SciPy LBFGS/SLSQP methods. `QfmResidualJAX` exists in
  `src/simsopt/geo/surfaceobjectives_jax.py:691`, and pure residual helpers
  exist near `:567`, but no `QfmSurfaceJAX` adapter was found.
- `src/simsopt/jax_core/wireframe.py:473-499` exposes `wireframe_B`,
  `wireframe_dB_by_dX`, and `wireframe_B_and_dB_by_dX`; no
  `wireframe_d2B_by_dXdX` name was found.
- `src/simsopt/geo/surfaceobjectives.py:220,278,336` defines CPU
  `Area`, `Volume`, and `ToroidalFlux`. Pure JAX helpers exist, but no
  `AreaJAX`, `VolumeJAX`, or `ToroidalFluxJAX` class names were found.
- `src/simsopt/objectives/least_squares.py:30` and
  `src/simsopt/objectives/constrained.py:27` remain CPU Optimizable problem
  wrappers. JAX optimization currently routes through `target_minimize` in
  `src/simsopt/geo/optimizer_jax.py:3372`.
- `src/simsopt/field/normal_field.py:20,522` and
  `src/simsopt/field/mgrid.py:22` remain CPU/file-oriented ancillary
  surfaces.

#### Files In Scope By Slice

- `src/simsopt/field/coilset_jax.py`
- `src/simsopt/geo/qfmsurface_jax.py`
- `src/simsopt/jax_core/wireframe.py`
- `src/simsopt/geo/surfaceobjectives_jax.py`
- `src/simsopt/objectives/least_squares_jax.py` only if W7-X6 is selected
- `src/simsopt/objectives/constrained_jax.py` only if W7-X6 is selected
- `src/simsopt/field/normal_field_jax.py` only if W7-X7 is selected
- `src/simsopt/field/mgrid_jax.py` or `src/simsopt/jax_core/mgrid.py` only if
  W7-X8 is selected
- `tests/field/test_coilset_jax.py`
- `tests/geo/test_qfm_jax.py`
- `tests/field/test_wireframefield_jax_item30.py`
- `tests/geo/test_surface_objectives_jax.py`
- `tests/geo/test_boozersurface_jax.py`
- `tests/objectives/test_least_squares_jax.py` / `test_constrained_jax.py`
  only if W7-X6 is selected

#### Out Of Scope

- Folding multiple long-tail items into one broad PR.
- File readers/writers (`MGrid` NetCDF read, VTK, FOCUS/MAKEGRID) unless the
  selected slice is explicitly a host IO compatibility task.
- Claiming solver trajectory parity from fixed-state value/gradient parity.
- New public wrappers that do not preserve CPU boundary contracts and JAX
  hot-path contracts.

#### Workstreams

- [ ] **W7-X0 - Revalidate long-tail contracts**
  - [ ] Confirm which original audit items are still unported, partial, or
        already handled.
  - [ ] Split any item larger than one PR into a separate implementation plan.
  - [ ] Identify whether each gap is a pure kernel, an Optimizable adapter, a
        solver, or a test-only mirror.
  - [ ] For the selected slice, write a `STATUS.md` pre-state block with live
        file paths, oracle type, and exact out-of-scope rows.

- [ ] **W7-X1 - `CoilSetJAX` / `ReducedCoilSetJAX`**
  - [ ] Revalidate `src/simsopt/field/coilset.py` and current
        `CoilSetDofExtractionSpec`.
  - [ ] Decide whether to implement a public Optimizable wrapper, lower-level
        pure kernels, or both.
  - [ ] Preserve existing CPU behavior at the boundary.
  - [ ] Add parity tests against `CoilSet` / `ReducedCoilSet` on a small pinned
        fixture.
  - [ ] Route flux penalties through `BiotSavartJAX` + `SquaredFluxJAX` only
        after verifying that Wave 2 and Wave 6 parity surfaces are available.
  - [ ] Do not port coil file IO, serialization, or mutating helper methods in
        the first CoilSet slice.

- [ ] **W7-X2 - `QfmSurfaceJAX`**
  - [ ] Revalidate `QfmResidualJAX` and optimizer support.
  - [ ] Add a solver adapter only if residual and Jacobian surfaces are ready.
  - [ ] Compare against CPU `QfmSurface` convergence on pinned fixtures.
  - [ ] Keep SciPy-only features out of the JAX hot path unless explicitly
        required.
  - [ ] Implement label-constraint support only for labels with JAX adapters or
        pure JAX helpers (`Area`, `Volume`, `ToroidalFlux`, etc.).
  - [ ] Separate value/gradient parity from convergence/iteration-history
        parity.

- [ ] **W7-X3 - Wireframe `compute(derivatives=2)`**
  - [ ] Revalidate whether JAX wireframe has `d2B_by_dXdX`.
  - [ ] Implement analytic `d2B_by_dXdX` only if still missing.
  - [ ] Add C++ oracle parity and Taylor tests.
  - [ ] Keep cache orchestration differences documented as non-portable.
  - [ ] Add `wireframe_B_and_dB_and_d2B` only if a production caller needs the
        bundled cache shape; otherwise expose the functional Hessian first.

- [ ] **W7-X4 - Surface objective Optimizable wrappers**
  - [ ] Revalidate existing pure helpers for area, volume, and toroidal flux.
  - [ ] Add `AreaJAX`, `VolumeJAX`, and `ToroidalFluxJAX` Optimizable wrappers
        only if wrapper absence is still a live contract gap.
  - [ ] Add parity tests against `Area`, `Volume`, and `ToroidalFlux` CPU
        objectives.
  - [ ] Preserve `dJ_by_dsurfacecoefficients` and `parameter_derivatives`
        semantics where CPU objectives expose them.
  - [ ] Do not duplicate `AspectRatioJAX` / `PrincipalCurvatureJAX` patterns
        unless the new wrappers share the same Optimizable boundary.

- [ ] **W7-X5 - BoozerSurface convergence-history mirrors**
  - [ ] Revalidate current BoozerSurface solver convergence and trajectory
        tests.
  - [ ] Mirror only the CPU convergence-history fixtures that remain
        unmirrored.
  - [ ] Keep solver-trajectory claims separate from fixed-state value/gradient
        parity.
  - [ ] Candidate CPU fixtures from the test-mirror audit:
        `tests/geo/test_boozersurface.py:534,732,864,907`.
  - [ ] Record iteration counts, residual norms, branch choices, and any
        accepted tolerance lane in the test or `STATUS.md`.

- [ ] **W7-X6 - Least-squares and constrained problem wrappers**
  - [ ] Revalidate CPU `LeastSquaresProblem` and `ConstrainedProblem` public
        contracts.
  - [ ] Decide whether a JAX problem wrapper is needed or whether
        `target_minimize` remains the intended public JAX solve API.
  - [ ] If wrappers are added, mirror CPU residual arithmetic, parent/DOF
        wiring, bounds, and scalar objective contracts on small fixtures.
  - [ ] Keep MPI wrappers out of scope.

- [ ] **W7-X7 - NormalField / CoilNormalField**
  - [ ] Revalidate which `NormalField` rows are pure Fourier coefficient
        algebra versus CPU Optimizable plumbing.
  - [ ] Port only the real-space / coefficient transforms needed by CoilSet or
        SPEC coupling.
  - [ ] Add parity tests against `NormalField` / `CoilNormalField` for
        coefficient packing, real-space reconstruction, and coilset
        correspondence if selected.

- [ ] **W7-X8 - MGrid evaluator**
  - [ ] Treat NetCDF file reading as host IO. Port only the grid evaluator /
        interpolation surface if needed.
  - [ ] Reuse `jax_core/regular_grid_interp.py` where possible.
  - [ ] Add parity tests against a CPU `MGrid` fixture after loading on host.

- [ ] **W7-X9 - Tracing and particle post-processing tails**
  - [ ] Revalidate `compute_resonances`, `compute_toroidal_transits`,
        `compute_poloidal_transits`, and missing stopping-criterion behavior.
  - [ ] Port only pure post-processing or already-supported JAX-tracer branches.
  - [ ] Keep collisional / non-vacuum `sopp.particle_*` branches out of scope
        unless selected as a dedicated tracing wave.

#### Validation

- [ ] Run focused tests for each implemented W7 item.
- [ ] Run the public pure-JAX command from `CLAUDE.md` after any shared solver
      or objective wrapper change.
- [ ] Run private optimizer and integration suites only for solver/objective
      changes that affect Stage 2 or single-stage paths.
- [ ] For each new public module, add/import it through existing lazy export
      smoke tests or a focused import-smoke row.

#### Acceptance Criteria

The Wave 7 queue is actionable when each selected `/goal` slice satisfies:

1. The slice has its own current-tree evidence block in `STATUS.md` before
   implementation starts.
2. New wrappers preserve CPU boundary contracts and JAX hot-path contracts.
3. Solver slices distinguish fixed-state parity, gradient parity, convergence
   parity, and iteration-history parity.
4. File IO, serialization, MPI, and plotting rows are either host-boundary
   setup or explicitly deferred.
5. No unrelated long-tail item is folded into a PR without a fresh scope check.

## Notes For Executors

- Follow `tests/REVIEWER_ORACLE_LINT.md`: JAX-vs-JAX, re-export identity, and
  host wrappers that secretly route through JAX are not parity oracles.
- Do not import `simsoptpp` from JAX source modules. Tests may import C++ oracle
  symbols.
- Do not add defensive try/except wrappers, dynamic imports, or `Any` casts.
- Do not relax tolerances to make a test pass. If parity fails at the correct
  lane, record the failure and stop.
- Before closing a wave, re-check the relevant official docs/source contracts:
  JAX transform/sharding/transfer semantics, SIMSOPT public CPU APIs, and CUDA
  floating-point behavior for any GPU claim.
- Treat upstream CPU/SIMSOPT behavior as the first oracle. A JAX CPU fixed-state
  parity pass is not a CUDA/GPU proof and is not a downstream optimizer E2E
  proof.
- For downstream regression closure, run the public pure-JAX command from
  `CLAUDE.md` after shared JAX exports, field composition, surface/objective, or
  solver changes, and run the affected private Stage 2 / single-stage /
  optimizer suites only when those call paths are touched.
- Keep host IO, explicit `jax.device_put` / `jax.device_get` transfers, and
  transfer-guard checks at validation boundaries. Do not hide host transfers in
  kernels or public wrappers.
