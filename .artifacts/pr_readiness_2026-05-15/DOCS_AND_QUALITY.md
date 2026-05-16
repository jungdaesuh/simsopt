# PR Readiness — Docs & Code Quality Audit
Branch: `gpu-purity-stage2-20260405` (HEAD `5c488d809`)
Date: 2026-05-15
Scope: docs, code hygiene, reviewer-facing surface (read-only on source)

## RST docs completeness — gaps

The three JAX docs exist on disk but **none are wired into the Sphinx toctree**.
`docs/source/index.rst:64-108` lists the `toctree` directives and references
none of the JAX RSTs. As a result they will not be built by Sphinx, will not
have stable URLs, and reviewers cannot find them from the entry page.

| Doc | Path | In TOC? | Notes |
|----|----|----|----|
| GPU setup | `docs/source/jax_gpu_setup.rst` | No | 481 lines, covers conda env, env vars, HF Jobs, Lightning, Runpod ops, troubleshooting |
| Acceptance criteria | `docs/source/jax_acceptance.rst` | No | 148 lines, parity gates, validation checklist |
| Migration guide | `docs/source/jax_migration.rst` | No | 112 lines, CPU→JAX API mapping |

No `grep` hit for the strings `jax_gpu_setup`, `jax_acceptance`, or
`jax_migration` anywhere under `docs/source/`. The CLAUDE.md cites these
as the canonical references but they are orphaned files.

Other doc gaps:

- **No API reference page** for the JAX public surface. `docs/source/fields.rst`,
  `docs/source/geo.rst`, `docs/source/index.rst:100` `simsopt` autodoc page do
  not autodoc `BiotSavartJAX`, `SquaredFluxJAX`, `BoozerSurfaceJAX`,
  `BoozerResidualJAX`, `IotasJAX`, `NonQuasiSymmetricRatioJAX`, or
  `simsopt.backend`.
- **17 env vars undocumented.** The runtime reads 29 distinct
  `SIMSOPT_*` env flags from `src/simsopt/`; only 12 appear in the three JAX
  RSTs. Undocumented ones used by code:
  `SIMSOPT_JAX_COIL_CHUNK_SIZE`, `SIMSOPT_JAX_POINT_CHUNK_SIZE`,
  `SIMSOPT_JAX_QUADRATURE_BLOCK_SIZE`, `SIMSOPT_JAX_PENALTY_POINT_CHUNK_SIZE`,
  `SIMSOPT_JAX_SHARDING`, `SIMSOPT_JAX_SHARDING_AXIS`,
  `SIMSOPT_JAX_COIL_SHARDING_AXIS`, `SIMSOPT_JAX_MIN_COILS_TO_SHARD`,
  `SIMSOPT_JAX_MIN_PAIRWISE_ROWS_TO_SHARD`, `SIMSOPT_JAX_MIN_POINTS_TO_SHARD`,
  `SIMSOPT_JAX_DEBUG_NANS`, `SIMSOPT_JAX_TRANSFER_GUARD`,
  `SIMSOPT_JAX_GEO_MODULES`, `SIMSOPT_JAX_COMPILATION_CACHE_DIR`,
  `SIMSOPT_JAX_DISTRIBUTED_INIT`, `SIMSOPT_TARGET_LANE_STRICT`,
  `SIMSOPT_LBFGS_DEBUG`, `SIMSOPT_TRACEABLE_DIAG_PROGRESS`.

## Docstring coverage on JAX public surface — table

Spot-checked top-level public class/function on each M1-M6 module.

| Module | Top-level public symbol | Docstring | Notes |
|----|----|----|----|
| `src/simsopt/field/biotsavart_jax.py` | re-export shim only | Module docstring present | Forwards to `simsopt.jax_core.biotsavart` |
| `src/simsopt/field/biotsavart_jax_backend.py` | `BiotSavartJAX` (line 975) | Yes (rich, multi-paragraph) | r"""...""" |
| `src/simsopt/objectives/fluxobjective_jax.py` | `SquaredFluxJAX` (line 150) | Yes (Args block, contract notes) | r"""...""" |
| `src/simsopt/objectives/integral_bdotn_jax.py` | `integral_BdotN` | Module docstring; function: short | Three definitions documented |
| `src/simsopt/geo/surface_fourier_jax.py` | `surface_gamma` / `stellsym_scatter_indices` | Yes | Stellsym convention explained at 1166-1169 |
| `src/simsopt/geo/boozer_residual_jax.py` | `boozer_residual_scalar` | Module docstring rich; per-function docstrings present | |
| `src/simsopt/geo/boozersurface_jax.py` | `BoozerSurfaceJAX` (line 3190) | Yes (Args block, threading note, M3 references) | |
| `src/simsopt/geo/optimizer_jax.py` | `jax_minimize` (module-level) | Module docstring documents all methods | Tags target-lane vs reference |
| `src/simsopt/geo/label_constraints_jax.py` | `toroidal_flux_jax`, `compute_G_from_currents` | Yes | Both have Args/Returns |
| `src/simsopt/geo/surfaceobjectives_jax.py` | `BoozerResidualJAX`, `IotasJAX`, `NonQuasiSymmetricRatioJAX` | Yes (math + Args for each) | |
| `src/simsopt/backend.py` | facade exports only | Module docstring 1-line | Re-exports from `simsopt.backend.runtime` |

**Verdict:** Public docstrings present on every M1-M6 entry. No `TODO`/`FIXME`
inside docstrings on JAX surface (`grep` confirmed). Minor: `backend.py` is
a one-line facade docstring — could enumerate the API.

## CLAUDE.md drift — table

| Claim | Current truth | Impact |
|----|----|----|
| Stellsym DOF convention: cos-cos + sin-sin for x, cos-sin + sin-cos for y/z | Matches `surface_fourier_jax.py:1166-1179` (`stellsym_scatter_indices`) | No drift |
| M4 VJP signature `(lm, booz_surf, iota, G)` | Matches `boozersurface_jax.py:2269` and `:2386` (uses `_booz_surf, _iota, _G` for unused) | No drift |
| `BoozerSurfaceJAX.get_adjoint_runtime_state()` exists as runtime SSOT | Defined at `boozersurface_jax.py:3702` | No drift |
| `linear_solve_factors` load-bearing in LS lane at `boozersurface_jax.py:3418-3475` | Range still hits the LS callback construction zone (lines around 3410-3480 contain the `apply_forward`/`apply_transpose`/`solve_forward`/`linear_solve_factors` block) | Range still accurate |
| Exact normalizer in `_normalize_solver_options` strips `optimizer_backend` | `_normalize_solver_options` at `boozersurface_jax.py:3122`; the strip itself is the `pop("optimizer_backend", None)` at `boozersurface_jax.py:3185-3186`. CLAUDE.md, `_cpp_compatible_probe.py`, and historical artifacts updated to cite the actual function and line range. | Citation refreshed |
| `_pre_newton_census_gate_failures` in `benchmarks/single_stage_init_parity.py` | Defined at `benchmarks/single_stage_init_parity.py:2110`, consumed at `:2181` | No drift |
| `PARITY_LADDER_TOLERANCES` in `benchmarks/validation_ladder_contract.py` | Defined at `benchmarks/validation_ladder_contract.py:52` | No drift |
| M5 modules all exist (`BoozerResidualJAX`, `IotasJAX`, `NonQuasiSymmetricRatioJAX`) | Present at `surfaceobjectives_jax.py:2299, 2462, 2568` (plus `MajorRadiusJAX` at 2507, not listed in CLAUDE.md) | `MajorRadiusJAX` not mentioned in CLAUDE.md M5 section |
| Memory file: tautology tests in `test_boozer_residual_jax`, `test_biotsavart_jax`, `test_fluxobjective_jax_parity` | `test_biotsavart_jax.py:248-300` helpers explicitly relabeled "NOT a C++ parity oracle"; `_flux_kernel_value_and_grad` at `test_fluxobjective_jax_parity.py:332` clearly labeled "NOT a CPU/JAX parity oracle"; `test_boozer_residual_jax.py` near lines 402/432 cites `simsoptpp.boozer_residual` and `boozer_residual_floor_scalar` tier | **Tautology audit appears resolved** — helpers retained but reclassified |

Tier "drift" overall: low. Three orphaned RSTs are the real blocker.

## Lint state — JAX modules

Ran `ruff check` against all 11 JAX surface modules (from CLAUDE.md M1-M6
table) using the repo-local `.conda/jax-0.9.2/bin/ruff`:

```
All checks passed!
```

No issues. The only `# noqa` directives in the surface are:

- `src/simsopt/field/biotsavart_jax.py:19` — `noqa: E402` for the
  `sys.path.insert`-then-import shim. Legitimate.
- `src/simsopt/backend.py:10` — `noqa: E402` for re-export from
  `simsopt.backend.runtime` after `__path__` manipulation. Legitimate.
- `src/simsopt/geo/boozersurface_jax.py:994,1145,1148,1301` —
  `noqa: PLC0415` on conditional `from .surface_fourier_jax_cpu_ordered import …`
  and `from ..jax_core.biotsavart_cpu_ordered import …` inside parity-only
  branches. Lazy imports for an optional CPU-ordered parity backend.

**Reviewer-facing verdict: CLEAN.** Zero ruff issues.

## Type-cleanliness — `Any` / `# type: ignore` audit

- `grep -nE "from typing import.*Any|: Any[^_]|-> Any\b|cast\(Any"` across
  all 11 JAX surface modules: **0 hits**.
- `grep -nE "# type: ignore"`: **0 hits**.

Honors the user's "NO `Any` casts" rule on the JAX surface.

## Code-hygiene findings — TODO/FIXME/print/breakpoint residue

- `breakpoint(` / `pdb.set_trace(`: **0 hits anywhere in `src/simsopt/`**.
- `TODO`/`FIXME` in changed src: **only upstream CPU code** (5 in
  `_core/optimizable.py`, 1 each in `geo/boozersurface.py`,
  `geo/surfacegarabedian.py`, `geo/surfacerzfourier.py`). **Zero in JAX
  surface modules.**
- `print(` in JAX surface: 6 hits, all gated:
  - `boozersurface_jax.py:5154,5415` — gated on `if verbose:` (LS / Newton
    solver result reporting).
  - `boozersurface_jax.py:5743,5851,5853` — gated on `if verbose and ...:`
    (exact-mode materialization diagnostics).
  - `surfaceobjectives_jax.py:194` — `_traceable_diag_progress()` helper
    gated on env flag `SIMSOPT_TRACEABLE_DIAG_PROGRESS`.

No raw debug residue. All printing is either verbose-gated or env-flag gated.

## Backwards-compat hacks — list

`grep -nE "_old|_legacy|deprecated|backward.?compat|removed:"` against the
JAX surface returns **one** result:

- `optimizer_jax.py:1801` — phrase "Used for backward-compatible reporting
  under the [pre-existing] ..." in a docstring describing a reporting field.
  This is documentation of a reporting alias, not a runtime hack.

No `_old` / `_legacy` modules exist on the JAX surface. The two env-flag pairs
(`SIMSOPT_BACKEND`/`STAGE2_BACKEND`, `SIMSOPT_JAX_PLATFORM`/`SIMSOPT_JAX_BACKEND`)
are explicitly documented as "legacy alias" in `jax_gpu_setup.rst` rows 95-117
and behave as soft aliases, not dead branches.

**Reviewer-facing verdict:** no hidden compat hacks. The legacy env aliases
are intentional and documented (in the orphaned RST — which still has the
toctree gap, but the documentation is at least written).

## Test oracle lint — tautology tests still present?

The 2026-05-13 audit flagged three test files as containing tautologies.
Working-tree state on `gpu-purity-stage2-20260405`:

- **`tests/field/test_biotsavart_jax.py`**:
  - `_dense_reference_fields` (line 248) and `_dense_B_vjp` (line 303) now
    carry explicit docstrings: *"This is a chunked-vs-dense self-consistency
    helper, **NOT a C++ parity oracle** ... Direct C++ parity assertions live
    in `TestBiotSavartJaxCppParity`."* The C++ parity class begins at line 498.
  - Verdict: **resolved by relabeling**, but reviewers should still confirm
    that callsites at lines 884, 920, 1159, 1279 are within chunking-probe
    tests (not parity tests).

- **`tests/objectives/test_fluxobjective_jax_parity.py`**:
  - `_flux_kernel_value_and_grad` (line 332) docstring: *"NOT a CPU/JAX parity
    oracle. ... Do not cite this helper as a parity oracle in new tests; route
    CPU/JAX parity through `SquaredFlux`/`SquaredFluxJAX` ..."*
  - Real parity tests at lines 379-394 use `objective_cpu.J()` / `.dJ()` from
    upstream `SquaredFlux` as the oracle (line 211: `objective_cpu = SquaredFlux(...)`).
  - Verdict: **resolved**.

- **`tests/geo/test_boozer_residual_jax.py`**:
  - `_numpy_boozer_residual_reference` cited by the memory file at lines 402, 432.
    The `test_scalar_residual_norm_near_tolerance_floor_matches_cpp_oracle`
    test (line 451) carries: *"Oracle: C++ reference symbol `simsoptpp.boozer_residual`
    (acceptable oracle type 1). Lanes: `direct_kernel` (rtol=1e-10, atol=1e-12)
    for the scalar reduction and the parity-lane `boozer_residual_floor_scalar`
    tier ..."*
  - Verdict: **resolved**.

The `tests/REVIEWER_ORACLE_LINT.md` policy is documented and the three flagged
files now route real parity through C++ oracles (`SquaredFlux`,
`simsoptpp.boozer_residual`, `TestBiotSavartJaxCppParity`).

## Examples / runbooks — what reviewers can actually run

`find examples -name "*jax*"` reveals only:

- `examples/single_stage_optimization/jax_host_boundary.py` — a small helper
  module, not a runnable demo.
- `examples/single_stage_optimization/STAGE_2/banana_coil_solver.py` is
  **JAX-aware** (accepts `--backend jax` and reads `SIMSOPT_BACKEND`; see
  lines 655-660). Not labeled as JAX-aware in the README.
- `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py`
  uses `BiotSavartJAX`/`SquaredFluxJAX`/`BoozerSurfaceJAX` (per `grep`).

The `examples/single_stage_optimization/README.md` predates the JAX port. It
mentions nothing about the `--backend jax` flag, `SIMSOPT_BACKEND`, JAX
prerequisites, or which output directories are JAX-specific. Reviewers will
not discover the JAX path from the README.

`examples/1_Simple`, `examples/2_Intermediate`, `examples/3_Advanced` contain
**no `*_jax*` files** — the standard tutorial ladder is CPU-only.

**Gap:** no top-level "run me first to verify JAX works" example. The
`benchmarks/jax_feasibility_spike.py` is the closest substitute and lives
outside `examples/`.

## Reviewer-facing blockers (ranked)

1. **[BLOCKER] JAX RSTs are orphaned.** `docs/source/jax_gpu_setup.rst`,
   `docs/source/jax_acceptance.rst`, `docs/source/jax_migration.rst` not
   referenced in any toctree (`docs/source/index.rst:64-108`). Sphinx will
   not build them. A reviewer landing on the docs site sees nothing about
   the JAX backend. **Fix: add a `JAX backend` caption section to
   `index.rst` with these three RSTs.** ~5 lines.

2. **[BLOCKER] No JAX API autodoc.** `BiotSavartJAX`, `SquaredFluxJAX`,
   `BoozerSurfaceJAX`, `BoozerResidualJAX`, `IotasJAX`,
   `NonQuasiSymmetricRatioJAX`, and `simsopt.backend.*` do not appear in
   any `.. autoclass::` / `.. automodule::`. Reviewers cannot click through
   to the public surface. **Fix: extend `fields.rst`, `geo.rst`, or add a
   `jax_api.rst` with autoclass directives.**

3. **[MAJOR] 17 env vars read by JAX runtime are undocumented.** See list
   above. A reviewer encountering production behavior driven by, e.g.,
   `SIMSOPT_JAX_TRANSFER_GUARD` or `SIMSOPT_JAX_COIL_CHUNK_SIZE` cannot
   look them up. **Fix: extend the env-vars section in `jax_gpu_setup.rst`.**

4. **[MAJOR] Examples README does not mention the JAX backend.** Stage 2
   and single-stage examples support `--backend jax`, but the README at
   `examples/single_stage_optimization/README.md` does not flag this.
   **Fix: add a JAX section to the README; or add a minimal
   `examples/8_JAX/` runnable demo.**

5. **[MINOR] CLAUDE.md M5 section omits `MajorRadiusJAX`.** Class exists
   at `surfaceobjectives_jax.py:2507` and is exported in `__all__` but is
   not listed in the M5 table. **Fix: add a row.**

6. **[RESOLVED] Drift citation `boozersurface_jax.py:3097` updated.**
   Refreshed to cite `_normalize_solver_options` at `:3122` with the
   strip itself at `:3185-3186` across CLAUDE.md, `_cpp_compatible_probe.py`,
   and historical jax_port_goal plan artifacts as part of W1.2 of the
   BoozerSurface LS deepdive plan.

7. **[NICE-TO-HAVE] `backend.py` facade docstring is one line.** Could
   enumerate the public re-exported API. ~10 lines.

What is **not** a blocker (intentionally listed so reviewers can stop
worrying about it):

- Lint: clean (`ruff check` passes on all 11 modules).
- Type-cleanliness: zero `Any` / zero `# type: ignore`.
- Hygiene: zero `breakpoint(` / `pdb.set_trace(` / debug `print(` in src.
- Docstrings on M1-M6 public classes: present and informative.
- Tautology tests: resolved by reclassification + real C++ oracles.
- Backwards-compat hacks: none on the JAX surface.
