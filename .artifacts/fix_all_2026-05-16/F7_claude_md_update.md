# F7 — CLAUDE.md drift fixes (2026-05-16)

Target file: `/Users/suhjungdae/code/columbia/simsopt-jax/CLAUDE.md`
HEAD at fix time: `f455402ed`

## §1 Corrections applied

- Replaced the M5 adapter-pattern bullet to reflect that runtime evaluation
  is fully JAX-pure (`_surface_geometry_from_dofs` from `solved_state.sdofs`)
  for both value and gradient; CPU surface objects are only the
  spec/DOF source at construction.
- Updated `_normalize_solver_options` citation from
  `boozersurface_jax.py:3122` to `:3134` (def) and from `:3185-3186` to
  `:3197-3198` (exact strip), pinning the SHA `f455402ed`.
- Updated the `Adjoint / warm-start operator solves` PLU citation from
  `boozersurface_jax.py:3514-3540` to `:3500-3580` and from
  `surfaceobjectives_jax.py:3167-3220` to `:3170-3260`, pinning HEAD `f455402ed`.
- Updated the `Note on linear_solve_factors` bullet citations
  consistently: `boozersurface_jax.py:3523-3580` for the LS SciPy
  reference runtime callbacks, `surfaceobjectives_jax.py:3170-3260` for
  `_traceable_solve_plu_linearization`, with HEAD `f455402ed`.
- No new content added. No other paragraphs touched.

## §2 Diff summary

Four bullets in the "Key Conventions" section changed:

1. `M5 adapter pattern` — full rewrite (factual correction).
2. `Exact Boozer scaling-limit contract` — only the parenthetical line
   citation `(boozersurface_jax.py:3122, ... :3185-3186)` updated to
   `(boozersurface_jax.py:3134, ... :3197-3198; current as of HEAD f455402ed)`.
3. `Adjoint / warm-start operator solves` — PLU citation updated.
4. `Note on linear_solve_factors` — PLU citations updated.

## §3 Verification (citation by citation)

### `_normalize_solver_options` def
- Old in CLAUDE.md: `boozersurface_jax.py:3122`.
- Grep evidence: `grep -n "def _normalize_solver_options" src/simsopt/geo/boozersurface_jax.py`
  -> `3134:def _normalize_solver_options(raw_options, boozer_type):`.
- New value: `boozersurface_jax.py:3134`. Off by +12 (file drifted).

### Exact strip site
- Old: `boozersurface_jax.py:3185-3186`.
- Grep evidence: `grep -n "optimizer_backend"` shows
  `3197:    if boozer_type == "exact":` followed by
  `3198:        normalized_options.pop("optimizer_backend", None)`.
- New value: `boozersurface_jax.py:3197-3198`. Off by +12.

### LS PLU runtime block in `boozersurface_jax.py`
- Old: `:3514-3540`.
- Read evidence: the `optimizer_backend == "scipy"` LS runtime block
  with `H_host = P_host @ L_host @ U_host` and the matching
  `apply_forward`/`apply_transpose`/`solve_forward`/`solve_transpose`
  closures runs from `3523:` (if-guard) to `3580:` (closing
  `pack_callbacks` call). The PLU-shared callback block immediately
  above runs from `3500-3521`. Used the inclusive range `3500-3580` for
  the general "see also" citation and `3523-3580` for the dense-PLU
  SciPy reference block specifically.
- New values pinned to HEAD `f455402ed`.

### `_traceable_solve_plu_linearization`
- Old: `surfaceobjectives_jax.py:3167-3220`.
- Grep evidence:
  `grep -n "def _traceable_solve_plu_linearization" surfaceobjectives_jax.py`
  -> `3170:def _traceable_solve_plu_linearization(`.
  Body runs through ~line 3260 (success returns and helpers).
- New value: `surfaceobjectives_jax.py:3170-3260`. Off by +3 on the start,
  but the body length is longer than the old citation suggested.

### `stellsym_scatter_indices`
- CLAUDE.md cites the function only by name and file, no line.
- Grep evidence:
  `grep -n "def stellsym_scatter_indices" src/simsopt/geo/surface_fourier_jax.py`
  -> `1154:`. Still in `surface_fourier_jax.py`. No update needed.

### `BiotSavartJAX._coil_dof_state_token`
- Cited in CLAUDE.md only as an attribute name, not a line number.
- Grep evidence:
  `grep -n "_coil_dof_state_token" src/simsopt/field/biotsavart_jax_backend.py`
  -> assignments at lines 460, 499, 1023, 1050 with the token factory at 91.
  Attribute is present on `SpecBackedBiotSavartJAX` and `BiotSavartJAX`.
  No update needed.

### `_traceable_solve_state_token`
- Cited only by name.
- Grep evidence:
  `grep -n "_traceable_solve_state_token" src/simsopt/geo/boozersurface_jax.py`
  -> token factory at 134, mutation sites at 722 and 3268. Attribute
  present. No update needed.

### M5 adapter pattern claim
- Old claim: wrappers call `surface.gamma()` / `label.J()` for value
  computation.
- Read evidence:
  - `BoozerResidualJAX._compute_value_from_solved_state` at line 2391
    of `surfaceobjectives_jax.py` calls
    `_evaluate_direct_coil_objective_value(...)` against
    `solved_state.iota`, `solved_state.G`, `solved_state.sdofs`. No CPU
    `gamma()` or `label.J()`.
  - `IotasJAX._compute_value_from_solved_state` at 2499 returns
    `solved_state.iota` directly.
  - `MajorRadiusJAX._compute_value` at 2523 calls
    `surface_major_radius_jax_from_dofs(self._surface_spec(), sdofs)`,
    pure JAX.
  - `NonQuasiSymmetricRatioJAX._compute_value` at 2615 calls
    `_qs_ratio_pure(sdofs, coil_set_spec, ...)`, pure JAX.
  Grep for CPU eval calls in this file:
  `grep -n "surface.gamma\\(\\)\\|label.J\\(\\)" surfaceobjectives_jax.py`
  finds one match at line 863 inside an unrelated helper, none in any
  of the four wrapper value paths.
- Conclusion: the old claim was wrong. Replaced with the corrected text
  describing pure-JAX runtime and CPU surfaces as construction-time
  spec/DOF source.

### File / class / test existence sweep
- All files referenced in M1-M6 module tables exist:
  `src/simsopt/field/biotsavart_jax.py`,
  `src/simsopt/geo/surface_fourier_jax.py`,
  `src/simsopt/geo/boozer_residual_jax.py`,
  `src/simsopt/objectives/integral_bdotn_jax.py`,
  `src/simsopt/field/biotsavart_jax_backend.py`,
  `src/simsopt/objectives/fluxobjective_jax.py`,
  `src/simsopt/geo/boozersurface_jax.py`,
  `src/simsopt/geo/optimizer_jax.py`,
  `src/simsopt/geo/label_constraints_jax.py`,
  `src/simsopt/geo/surfaceobjectives_jax.py`,
  `src/simsopt/backend.py`.
- All test files in the Validation/M-section tables exist:
  `tests/test_jax_import_smoke.py`, `tests/test_benchmark_helpers.py`,
  `tests/test_run_code_benchmark_common.py`,
  `tests/integration/test_jax_native_path.py`,
  `tests/integration/test_single_stage_jax.py`,
  `tests/integration/test_stage2_jax.py`,
  `tests/geo/test_boozersurface_jax.py`,
  `tests/geo/test_boozer_derivatives_jax.py`,
  `tests/geo/test_boozer_residual_jax.py`,
  `tests/geo/test_surface_fourier_jax.py`,
  `tests/objectives/test_integral_bdotn_jax.py`,
  `tests/field/test_biotsavart_jax.py`,
  `tests/REVIEWER_ORACLE_LINT.md`.
- Doc paths exist: `docs/parity_dual_mode_contract_2026-05-08.md`,
  `docs/source/jax_acceptance.rst`, `docs/source/jax_gpu_setup.rst`,
  `.github/workflows/jax_smoke.yml`, `envs/jax.yml`.
- All key classes/functions resolve:
  `BoozerResidualJAX` 2302, `IotasJAX` 2465,
  `NonQuasiSymmetricRatioJAX` 2571,
  `make_traceable_objective_runtime_bundle` 5596 in
  `surfaceobjectives_jax.py`; `BoozerSurfaceJAX` 3202 in
  `boozersurface_jax.py`; `SquaredFluxJAX` 172 in
  `fluxobjective_jax.py`; `BiotSavartJAX` 984 and
  `SpecBackedBiotSavartJAX` 450 in `biotsavart_jax_backend.py`;
  `VALID_OPTIMIZER_BACKENDS = frozenset({"scipy", "ondevice"})` at
  `optimizer_jax.py:151`.

No additional citations required updates.
