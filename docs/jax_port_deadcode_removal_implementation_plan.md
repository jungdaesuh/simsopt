# JAX Port Dead-Code Removal Implementation Plan

## Purpose

Remove genuinely unused code from the JAX port (`src/simsopt_jax/` and
`src/simsopt_jax_adapters/`) identified by a verified dead-code audit, while
explicitly protecting symbols that *look* dead to static tooling but are reached
through dynamic dispatch (GSONable serialization, reflective `getattr`, pytree
hooks, C++ virtual overrides). This file is the executable checklist for that
cleanup and the record of what must **not** be touched and why.

## Goals

- Delete the **69** confirmed-dead symbols (def + body, and any now-orphaned
  re-imports), grouped by file, with no behavior change.
- Remove the one transitive orphan exposed by the cleanup
  (`_extract_full_subgraph_dofs`).
- Leave the test suite, `ruff`, and import smoke green.
- Keep an auditable separation between *deleted*, *protected* (verified live),
  and *deferred* (parity-surface judgment call) symbols.

## Non-Goals

- Removing the **7 deferred** winding-surface `*factor` / second-derivative
  `d2modBd*` parity-surface symbols — those are a maintainer judgment call
  (see Open Questions), not part of this cleanup.
- Touching the **2 protected** `_upstream` attributes — deleting them breaks
  GSONable serialization (proven below).
- Any refactor, rename, or behavior change beyond deleting dead definitions
  (the single exception is the `tangent_dash` → `_tangent_dash` discard rename).
- Re-running or re-litigating the audit itself.

## Current Context

- Audit pipeline: `vulture --min-confidence 60` produced 296 raw candidates →
  per-file LLM classifier (repo-wide `rg`, export/override/dispatch checks) →
  adversarial refuter per `DEAD` verdict → deterministic aggregation.
- `ruff --select F401,F811,F841` already passes on both packages, so unused
  *imports* and *locals* are not in scope — only dead defs/attrs/aliases are.
- Authoritative artifacts (regenerated after validation):
  - `/tmp/deadcode/actionable_dead.json` — the 69 to delete.
  - `/tmp/deadcode/verified_keep.json` — the 2 `_upstream` (never delete).
  - `/tmp/deadcode/defer.json` — the 7 parity-surface symbols.
  - `/tmp/deadcode/DEADCODE_AUDIT.md` — human-readable report.
- Repo test entrypoint: `python -m pytest` from the repo root works with the
  base miniforge `python3` — `tests/conftest.py` calls
  `bootstrap_local_simsopt(_REPO_ROOT / "src")`, which puts the working-tree
  `src/` (and the compiled `simsoptpp`) on the path. **Standalone** imports
  outside pytest do *not* get that bootstrap and need `PYTHONPATH=src`
  (e.g. `PYTHONPATH=src python3 -c "import simsopt_jax"`). The packages are a
  src-layout (`scikit-build-core`, `wheel.packages = simsopt, simsopt_jax,
  simsopt_jax_adapters`) and are not pip-installed in the base env.

## Rationale

The adapter layer mirrors the upstream simsopt CPU/C++ public surface and routes
almost everything through dynamic dispatch that `vulture`/`rg` cannot see
(`return_fn_map`, JAX pytree hooks, `@custom_jvp.defjvp`, C++ `sopp` virtual
overrides, getattr-keyed scalar tables, GSONable `_<arg>` serialization). That
makes ~75% of raw flags false positives, so each deletion is gated on a
repo-wide usage check plus a dispatch/override/serialization check rather than a
single grep. Two validation passes caught two false negatives of the same
"grep-invisible live code" class (`d2modBdzeta2`, `_upstream`); both are now
protected, and a family-consistency rule (never split sibling method families)
moved the `*factor` VJP partners into the deferred bucket.

## Assumptions

- The audit's repo-wide search scope (`src/`, `tests/`, `examples/`,
  `benchmarks/`, `docs/`) is the complete consumer set; nothing imports these
  packages from outside the repo at runtime in a way that bypasses those paths.
- Deferred `*factor` / `d2modBd*` symbols stay until a maintainer decides on
  parity-surface scope; this plan does not remove them.
- The working tree is otherwise clean enough that `pytest` is green *before*
  starting (establish a baseline first — see Validation Plan step 0).

## Implementation Plan

> Per file: delete the listed definition **and its body**. After each phase, run
> the per-phase import smoke (`PYTHONPATH=src python3 -c "import <module>"`). Checking a box means
> the symbol is deleted *and* `rg '\b<symbol>\b'` shows no remaining reference
> (except intended same-name symbols in the original `src/simsopt/` tree).
>
> Line numbers were verified against the working tree on 2026-06-19; the **symbol
> name is the source of truth, the line number is only a hint.** Several files
> here (e.g. `optimizer.py`) are under concurrent edits and drift — if a line does
> not match, locate the target with `rg -n 'def <symbol>'` / `rg -n '<symbol>'`
> before editing rather than trusting the offset.

### Phase 0 — Baseline & guards

- [ ] Capture a green baseline: `python -m pytest -q` (record pass count).
- [ ] Add a temporary guard test asserting the **protected** symbols still exist
      and serialize (see Validation Plan), so a later edit cannot silently
      delete them.

### Phase 1 — Zero-risk leaf removals (pure private/module-local, def-only repo-wide)

   - [ ] `src/simsopt_jax/geo/_pairwise_reductions.py:131` — function `_pairwise_rowwise_min_distance`
   - [ ] `src/simsopt_jax/geo/_pairwise_reductions.py:176` — function `_pairwise_rowwise_pnorm_distance`
   - [ ] `src/simsopt_jax/runtime/host_boundary.py:81` — function `strict_scalar_grad` (stale duplicate of `surface_objectives._strict_scalar_grad`)
   - [ ] `src/simsopt_jax/runtime/host_boundary.py:87` — function `strict_scalar_value_and_grad` (stale duplicate)
   - [ ] `src/simsopt_jax/geo/optimizers/optimizer.py:4515` — function `_solve_dense_square_operator_system_with_status`
   - [ ] `src/simsopt_jax/geo/optimizers/optimizer.py:4707` — function `_solve_jacobian_system` (distinct from live `_solve_jacobian_system_with_status` at :4736)
   - [ ] `src/simsopt_jax/core/biotsavart.py:161` — function `_tree_concatenate`
   - [ ] `src/simsopt_jax/core/_sympy_to_jax.py:16` — variable `ConditionEvaluator` (unused type alias)
   - [ ] `src/simsopt_jax/core/curve_planar_fourier.py:98` — function `jaxplanarcurve_pure` (port copy; the live same-name fn is in `src/simsopt/geo/curveplanarfourier.py` — do **not** touch that one)
   - [ ] `src/simsopt_jax_adapters/geo/surface_objectives.py:1019` — function `_take_runtime_row`
   - [ ] `src/simsopt_jax_adapters/geo/surface_objectives.py:2016` — function `_solve_boozer_forward`
   - [ ] `src/simsopt_jax_adapters/objectives/stage2_target.py:278` — function `_curve_group_arrays`
   - [ ] `src/simsopt_jax_adapters/field/force.py:54` — variable `Biot_savart_prefactor`
   - [ ] `src/simsopt_jax/core/curve_geometry.py:533` — variable `tangent_dash` — **rename** to `_tangent_dash` (tuple unpack must keep three names; match the centroid branch at line ~550). Do not delete the name.

### Phase 2 — `curve_contract.py` cluster (with transitive cascade)

   - [ ] `src/simsopt_jax_adapters/geo/curve_contract.py:28` — function `_install_curve_jax_contract` (simsopt installs the same contract inline; this reimplementation is never invoked). Deleting it removes its inner attribute assignments automatically:
       - [ ] :34 attribute `gamma_impl_jax`
       - [ ] :47 attribute `dgamma_by_dcoeff_jax`
       - [ ] :51 attribute `dgammadash_by_dcoeff_jax`
       - [ ] :55 attribute `dgammadashdash_by_dcoeff_jax`
   - [ ] `src/simsopt_jax_adapters/geo/curve_contract.py:12` — function `_as_runtime_jax_float64` (uncalled passthrough to `as_jax_float64`)
   - [ ] `src/simsopt_jax_adapters/geo/curve_contract.py:83` — function `_curve_jax_arg_from_full_dofs`
   - [ ] `src/simsopt_jax_adapters/geo/curve_contract.py:69` — function `_extract_full_subgraph_dofs` — **cascade orphan**: only caller is `_curve_jax_arg_from_full_dofs` (line 84); remove after the line-83 deletion.
   - [ ] `src/simsopt_jax_adapters/geo/curve_contract.py:92` — function `_optimizable_local_full_dofs_from_full_dofs`

### Phase 3 — Import-coupled removal

   - [ ] `src/simsopt_jax/geo/optimizers/private/_result_converters.py:200` — function `_scipy_result_is_continuable`
   - [ ] `src/simsopt_jax/geo/optimizers/private/__init__.py:37` — remove the now-dangling `_scipy_result_is_continuable,` re-import.

### Phase 4 — `biotsavart_backend.py`

   - [ ] `:819` function `_is_legacy_curve_xyzfourier`
   - [ ] `:1340` attribute `_curve_dof_size` (assignment in `_introspect_coils`, never read)
   - [ ] `:1453` attribute `_curve_dof_size` (second assignment; remove together with :1340)
   - [ ] `:1774` method `_base_curve_geometry` (wrapper forwarding to `_base_curve_geometry_with_timings`; no caller)
   - [ ] `:1954` variable `B_cotangents` (unused alias of `B_pullback_native`)
   - [ ] `:2005` variable `A_cotangents` (unused alias)
   - [ ] `:2014` variable `dA_by_dX_cotangents` (unused alias)
   - [ ] `:2023` variable `dB_by_dX_cotangents` (unused alias)

### Phase 5 — `curvecwsfourier.py` vestigial JAX attrs + orphaned `_2d` helpers

   - [ ] `:332` attribute `gamma_impl_jax`
   - [ ] `:336` attribute `gammac_jax`
   - [ ] `:337` attribute `gammas_jax`
   - [ ] `:342` attribute `dgamma_by_dcoeff_jax`
   - [ ] `:346` attribute `dgamma_by_dsurf_jax`
   - [ ] `:355` attribute `gammacdash_jax`
   - [ ] `:356` attribute `gammasdash_jax`
   - [ ] `:361` attribute `dgammadash_by_dcoeff_jax`
   - [ ] `:365` attribute `dgammadash_by_dsurf_jax`
   - [ ] `:382` attribute `gammacdashdash_jax`
   - [ ] `:383` attribute `gammasdashdash_jax`
   - [ ] `:390` attribute `dgammadashdash_by_dcoeff_jax`
   - [ ] `:396` attribute `dgammadashdash_by_dsurf_jax`
   - [ ] `:419` attribute `gammasdashdashdash_jax`
   - [ ] `:438` attribute `dgammadashdashdash_by_dsurf_jax`
   - [ ] `:453` attribute `dgamma_2d_by_dcoeff_vjp`
   - [ ] `:467` attribute `dgammadash_2d_by_dcoeff_vjp`
   - [ ] `:483` attribute `dgammadashdash_2d_by_dcoeff_vjp`
   - [ ] `:755` method `gammadash_2d_impl`
   - [ ] `:839` method `gammadashdash_2d`
   - [ ] `:843` method `gammadashdash_2d_impl`
   - ⚠️ Do **not** touch `zfactor`/`rfactor`/`dzfactor_by_dcoeff_vjp`/`drfactor_by_dcoeff_vjp` in this file — deferred (Non-Goals / Open Questions). Verify the `*_impl` variants that **override** a base `Curve` hook (e.g. `gamma_impl`, `dgamma_by_dcoeff_vjp_impl`) are untouched — only the `_2d`/`_2d_impl` non-overrides above are dead.

### Phase 6 — Remaining adapter aliases & write-only attributes

   - [ ] `src/simsopt_jax_adapters/field/boozer_field.py:272` — method `get_points_ref` (`BoozerRadialInterpolantJAX`; not an override of `BoozerMagneticField`)
   - [ ] `src/simsopt_jax_adapters/field/boozer_field.py:566` — method `get_points_ref` (`BoozerAnalyticJAX`)
   - [ ] `src/simsopt_jax_adapters/field/boozer_field.py:799` — method `get_points_ref` (`InterpolatedBoozerFieldJAX`)
   - ⚠️ Do **not** touch `boozer_field.py` `_upstream` (:206, :224) or `d2modBd*` (:886/:889/:892) — protected/deferred.
   - [ ] `src/simsopt_jax_adapters/geo/boozer_surface.py:2669` — function `_boozer_exact_coil_vjp_groups`
   - [ ] `src/simsopt_jax_adapters/geo/boozer_surface.py:3644` — function `_default_ls_optimizer_backend`
   - [ ] `src/simsopt_jax_adapters/geo/boozer_surface.py:4059` — attribute `_label_surface_runtime_state`
   - [ ] `src/simsopt_jax_adapters/geo/boozer_surface.py:4762` — method `_make_penalty_optimizer_state`
   - [ ] `src/simsopt_jax_adapters/geo/boozer_surface.py:4881` — method `_make_penalty_objective_host_jax_with`
   - [ ] `src/simsopt_jax_adapters/geo/qfm_surface.py:61` — attribute `qfm` (write-only; native API mirror)
   - [ ] `src/simsopt_jax_adapters/geo/qfm_surface.py:234` — method `minimize_qfm_penalty_constraints_LBFGS` (JAX-adapter alias; keep native `QfmSurface` method)
   - [ ] `src/simsopt_jax_adapters/geo/qfm_surface.py:281` — method `minimize_qfm_exact_constraints_SLSQP` (JAX-adapter alias)
   - [ ] `src/simsopt_jax_adapters/objectives/flux.py:310` — attribute `_normal_jax` (already captured in `self._flux_spec`)
   - [ ] `src/simsopt_jax_adapters/objectives/flux.py:311` — attribute `_target_jax`
   - [ ] `src/simsopt_jax_adapters/objectives/flux.py:510` — attribute `J_and_dJ` (alias of live `value_and_dJ`)
   - [ ] `src/simsopt_jax_adapters/geo/framed_curve.py:333` — method `jax_alpha`
   - [ ] `src/simsopt_jax_adapters/geo/framed_curve.py:340` — method `jax_alphadash`
   - [ ] `src/simsopt_jax_adapters/field/interpolated.py:171` — attribute `_skip_callable` (ctor arg is `skip`; the `skip` predicate is consumed via `_build_skip_callback`/`skip_cb`, so this is not a GSON fallback)

## Validation Plan

- [ ] **Step 0 (before edits):** `python -m pytest -q` green baseline recorded.
- [ ] **Protected-symbol guard:** a test that constructs each `_upstream`-bearing
      wrapper and calls `as_dict()` (round-trips through `from_dict`) — must pass
      both before and after, proving `_upstream` was not removed. Reference:
      GSONable `as_dict` at `src/simsopt/_core/json.py:190-206` reads
      `getattr(self, "_upstream")` as the fallback for ctor arg `upstream`.
- [ ] **Per-phase import smoke:** `PYTHONPATH=src python3 -c "import simsopt_jax, simsopt_jax_adapters"`
      after each phase (catches deleted-but-still-referenced symbols immediately).
      (Verified working 2026-06-19; plain `import simsopt_jax` without `PYTHONPATH=src`
      fails with `ModuleNotFoundError` because the package is not pip-installed.)
- [ ] **Lint:** `ruff check src/simsopt_jax src/simsopt_jax_adapters` passes
      (catches any newly-dangling import / redefinition).
- [ ] **Re-run the detector** to confirm no *new* orphans were created and the
      deleted set is gone:
      `python3 -m vulture src/simsopt_jax src/simsopt_jax_adapters --min-confidence 60`
      — expect the 69 removed entries absent; investigate any new function/method
      flags (transitive orphans like `_extract_full_subgraph_dofs`).
- [ ] **Grep guard for protected/deferred:** confirm `_upstream`, `d2modBdtheta2`,
      `d2modBdzeta2`, `d2modBdthetadzeta`, `zfactor`, `rfactor`,
      `dzfactor_by_dcoeff_vjp`, `drfactor_by_dcoeff_vjp` still present.
- [ ] **Full suite:** `python -m pytest -q` — pass count ≥ baseline, no new
      failures. Pay attention to serialization (`as_dict`/`from_dict`), Boozer
      field scalar, QFM, flux, and CWS-curve tests.

## Risks and Mitigations

- Risk: Deleting a symbol reached only via reflective `getattr` / GSONable
  `_<arg>` fallback (the `d2modBdzeta2` / `_upstream` class of false negative).
  Mitigation: the protected list + the `as_dict` guard test + the underscore-attr
  vs ctor-arg sweep already run during the audit; re-run that sweep if new
  underscore attributes are removed.
- Risk: Splitting a sibling method family (deleting one of three `d2modBd*` or
  one VJP of a `*factor` pair) and breaking parity-surface consistency.
  Mitigation: the deferred bucket keeps each family whole; Phase 5/6 ⚠️ notes
  call out the do-not-touch siblings.
- Risk: A `*_impl` deletion that is actually a C++/base virtual override (e.g.
  `_B_impl`, `gamma_impl`, `dgamma_by_dcoeff_vjp_impl`) — these are dispatched,
  not called by name. Mitigation: only non-overriding `_2d`/`_2d_impl` helpers
  are in scope; verify against the base class before deleting any `*_impl`.
- Risk: Transitive orphans beyond `_extract_full_subgraph_dofs`.
  Mitigation: the post-removal `vulture` re-run (Validation) surfaces them.
- Risk: Same-name symbols in the original `src/simsopt/` tree (e.g.
  `jaxplanarcurve_pure`) get edited by mistake. Mitigation: every task pins the
  full path under `src/simsopt_jax*/`.

## Completion Criteria

- [ ] All 69 actionable boxes + the `_extract_full_subgraph_dofs` cascade +
      the `private/__init__.py:37` import are checked.
- [ ] `tangent_dash` renamed to `_tangent_dash` (not deleted).
- [ ] `ruff` clean; import smoke clean; `python -m pytest` pass count ≥ baseline.
- [ ] Post-removal `vulture` shows the removed set gone and no unexplained new
      function/method flags.
- [ ] Protected symbols (`_upstream` ×2, `d2modBd*` ×3) and deferred symbols
      (`zfactor`, `rfactor`, `*factor` VJPs ×2) still present and tested.
- [ ] Diff scoped to `src/simsopt_jax*/` (+ this plan); no edits under
      `src/simsopt/`, `src/simsoptpp/`.

## Open Questions

- Winding-surface `*factor` family (`zfactor`, `rfactor`, `dzfactor_by_dcoeff_vjp`,
  `drfactor_by_dcoeff_vjp`, and the then-orphaned `dzfactor_by_dcoeff`/
  `drfactor_by_dcoeff`): keep as upstream-parity public API, or trim because the
  consumer module was not ported? — maintainer decision. If "trim", they become a
  follow-up removal phase (delete as a whole family).
- Second-derivative `d2modBd*` family: keep all three for C++/`SYMMETRY_EXPLOIT_SCALARS`
  parity (recommended — they are live via reflective dispatch), or is the scalar
  table itself being slimmed? If the table shrinks, revisit as a unit.
- Should the deferred-family decision and the protected-symbol guard test be
  captured as a permanent regression test, or is the one-off guard sufficient?
