# `simsopt.solve.jax` caller inventory

**Status:** v0 audit, produced as the §SOFTWARE_DESIGN.md "API evolution gate / Caller inventory" artifact for the spec in `docs/solve_jax_api_spec_2026-05-19.md`.

**Method:** repository-wide grep against the union of public optimizer entry points and old-string-API tokens. Includes every `*.py` under `src/`, `tests/`, `benchmarks/`, `examples/`, `scripts/`. Excludes `thirdparty/`, `.miniforge/`, `.conda/`, `build/`, and (per the search scope) all generated artifacts.

**How to reproduce:** the exact commands used to assemble this inventory are listed in §9. Run any of them to validate or refresh.

---

## 1. Top-line numbers

| Surface | Total external call sites |
|---|---|
| `jax_minimize(...)` direct callers | **65** |
| `jax_least_squares(...)` direct callers | **6** |
| `reference_minimize(...)` direct callers | **25** |
| `reference_least_squares(...)` direct callers | **8** |
| `target_minimize(...)` direct callers | **43** |
| `target_least_squares(...)` direct callers | **23** |
| `BoozerSurfaceJAX(...)` constructor calls (files) | **14** |
| `ReferenceOptimizerContract` / `TargetOptimizerContract` callers (files) | **5** |
| Files using `optimizer_backend=` | **23** |
| Files using `least_squares_algorithm=` | **13** |
| Files using `limited_memory=` (True/False) | **9** |

Per-method-string external call counts (excludes the API-defining files in `src/simsopt/geo/optimizer_jax*.py` and `optimizer_jax_private/`):

| Old method string | External hits | Production-source hits | Test hits |
|---|---:|---:|---:|
| `lbfgs` | 11 | 0 | 11 |
| `lbfgs-trace` | 3 | 1 (example) | 2 |
| `lbfgs-scipy-jax` | 3 | 1 (example) | 2 |
| `lbfgs-scipy-jax-fullgraph` | 2 | 0 | 2 |
| `lbfgs-ondevice` | 50 | 0 | 50 |
| `bfgs` | 7 | 0 | 7 |
| `bfgs-ondevice` | 19 | 0 | 19 |
| `adam` | 3 | 1 (example) | 2 |
| `adam-ondevice` | 2 | 0 | 2 |
| `lm` | 16 | 2 (`boozersurface_jax.py`) + 1 upstream baseline | 13 |
| `lm-ondevice` | 6 | 0 | 6 |
| `lm-minpack-ondevice` | 4 | 0 | 4 |
| `optimistix-lm-ondevice` | 3 | 0 | 3 |
| **Total** | **~129** | **~5** | **~124** |

Headline: the migration is overwhelmingly **a test-suite migration**. Production `src/` code has fewer than 10 string-based optimizer call sites. The bulk of the work is rewriting tests and one large example file.

---

## 2. Per old method string — file:line and Driver mapping

For each old string, the new-API target (per spec §8) and every external call site. Sorted within each block.

### `method="lbfgs"` → `Driver.SCIPY_LBFGSB`

| File:line | Form | Notes |
|---|---|---|
| `tests/geo/test_boozersurface_jax.py:1271` | `method="lbfgs"` | inside `jax_minimize` test |
| `tests/geo/test_boozersurface_jax.py:1435` | `method="lbfgs"` | rosenbrock parity |
| `tests/geo/test_boozersurface_jax.py:1611` | `method="lbfgs"` | callback test |
| `tests/geo/test_boozersurface_jax.py:1619` | `method="lbfgs"` | callback test |
| `tests/geo/test_boozersurface_jax.py:2275` | `method="lbfgs"` | tolerance comparison test |
| `tests/geo/test_boozersurface_jax_private.py:865` | `method="lbfgs"` | private LBFGS-B test |
| `tests/geo/test_optimizer_jax_item19.py:44` | `ReferenceOptimizerContract(method="lbfgs")` | contract resolution |
| `tests/geo/test_single_stage_example.py:2609` | `ReferenceOptimizerContract(method="lbfgs")` | contract instantiation |
| `tests/geo/test_single_stage_example.py:4900` | `optimizer_method="lbfgs"` | kwarg-passthrough |
| `tests/integration/test_stage2_jax.py:1292` | `method="lbfgs"` | Stage 2 |
| `tests/test_benchmark_helpers.py:4356` | `reference_optimizer_method="lbfgs"` | benchmark helper |

**Risk:** trivial — every call site is a kwarg substitution.

### `method="lbfgs-trace"` → `Driver.SIMSOPT_TRACE_LBFGS`

| File:line | Form | Notes |
|---|---|---|
| `examples/.../single_stage_banana_example.py:7852` | `ReferenceOptimizerContract(method="lbfgs-trace")` | **production example** — debug fallback path |
| `tests/geo/test_boozersurface_jax.py:1579` | `method="lbfgs-trace"` | "must not enter scipy_minimize" guard |
| `tests/geo/test_single_stage_example.py:4906` | `optimizer_method="lbfgs-trace"` | kwarg-passthrough |

**Risk:** low. One example caller relies on the trace surface; the migration must preserve `record_optimizer_state_trace` and `invalid_step_log` fields per §7.3 of the spec.

### `method="lbfgs-scipy-jax"` → `Driver.SCIPY_LBFGSB` + per-term JAX value/grad callable

| File:line | Form | Notes |
|---|---|---|
| `examples/.../single_stage_banana_example.py:7836` | `TargetOptimizerContract(method="lbfgs-scipy-jax")` | **production example** |
| `tests/geo/test_single_stage_example.py:2582, 2588` | `TargetOptimizerContract(method="lbfgs-scipy-jax")` | contract tests |

**Risk:** low. Callers already wire per-term JAX value/grad; only the dispatch string changes.

### `method="lbfgs-scipy-jax-fullgraph"` → `Driver.SCIPY_LBFGSB` + fullgraph value/grad callable

| File:line | Form | Notes |
|---|---|---|
| `tests/test_benchmark_helpers.py:5326, 5335` | `outer_optimizer_method="lbfgs-scipy-jax-fullgraph"` / `expected_jax_outer_optimizer_method=...` | benchmark helper kwargs |

**Risk:** low. Caller already builds a fullgraph callable; the migration consolidates the string distinction into a callable-source choice (per spec §8).

### `method="lbfgs-ondevice"` → `Driver.SIMSOPT_LBFGSB`

50 call sites — the largest single migration block. Concentrated in:

| File | Count |
|---|---:|
| `tests/geo/test_boozersurface_jax_private.py` | 17 |
| `tests/geo/test_single_stage_example.py` | 10 |
| `tests/integration/test_single_stage_jax_cpu_reference.py` | 5 |
| `tests/subprocess/jax_runtime_cases.py` | 4 |
| `tests/subprocess/import_smoke_cases.py` | 4 |
| `tests/geo/test_single_stage_alm_integration.py` | 3 |
| `tests/integration/test_stage2_target_lane_purity.py` | 1 |
| `tests/geo/test_boozersurface_jax.py` | 1 |
| `tests/geo/test_optimizer_jax_item19.py` | 1 |
| Other | 4 |

(Full file:line dump available via the §9 reproduction commands; suppressed here for brevity.)

**Risk:** trivial per call site, but high migration *volume*. Likely one dedicated PR per file in this list.

### `method="bfgs"` → `Driver.SCIPY_BFGS`

| File:line | Form | Notes |
|---|---|---|
| `tests/geo/test_boozersurface_jax.py:1238, 1317, 1465, 2224, 4901, 8508` | `method="bfgs"` | 6 sites; standard test patterns |
| `tests/test_benchmark_helpers.py:5390` | `outer_optimizer_method="bfgs"` | benchmark helper kwarg |

**Risk:** trivial.

### `method="bfgs-ondevice"` → `Driver.SIMSOPT_BFGS`

19 call sites:

| File | Count |
|---|---:|
| `tests/geo/test_boozersurface_jax_private.py` | 9 |
| `tests/test_benchmark_helpers.py` | 4 |
| `tests/subprocess/import_smoke_cases.py` | 4 |
| `tests/subprocess/jax_runtime_cases.py` | 2 |

**Risk:** trivial per site.

### `method="adam"` → bridge: `Driver.SIMSOPT_ADAM_HOST`; target: `Driver.OPTAX_ADAM`

| File:line | Form | Notes |
|---|---|---|
| `examples/2_Intermediate/stage_two_optimization_stochastic.py:201` | `method="adam"` | **production stochastic stage-two example** |
| `tests/geo/test_boozersurface_jax.py:4827, 4848` | `method="adam"` | 2 sites |

**Risk:** medium. The example is a public-facing intermediate tutorial; the migration must preserve its result behavior. Optax target promotion is gated on parity validation.

### `method="adam-ondevice"` → bridge: `Driver.SIMSOPT_ADAM`; target: `Driver.OPTAX_ADAM`

| File:line | Form | Notes |
|---|---|---|
| `tests/geo/test_boozersurface_jax.py:4866` | `method="adam-ondevice"` | trace-safe Adam test |
| `tests/subprocess/import_smoke_cases.py:984` | `method="adam-ondevice"` | smoke |

**Risk:** medium (same reason as `adam`).

### `method="lm"` → `Driver.SIMSOPT_LM_GMRES_HOST`

| File:line | Form | Notes |
|---|---|---|
| `src/simsopt/geo/boozersurface.py:1251` | `method="lm"` | **upstream baseline kwarg default** (not a jax_least_squares caller; uses `scipy.optimize.least_squares` directly per upstream pattern) |
| `src/simsopt/geo/boozersurface_jax.py:5798` | `method="lm"` | **production kwarg default** in `minimize_boozer_penalty_constraints_ls` |
| `src/simsopt/geo/boozersurface_jax.py:5888` | `reference_least_squares(method="lm")` | **production call** routing to in-tree GMRES LM |
| `tests/geo/test_boozersurface.py:910, 923` | `method="lm"` | upstream Boozer baseline tests |
| `tests/geo/test_boozersurface_jax.py:3049, 3121, 3136` | `method="lm"` | 3 sites |
| `tests/geo/test_boozersurface_jax.py:4790, 4811, 4946` | `method="lm"` | 3 sites (some `jax_least_squares` direct calls) |
| `tests/geo/test_lm_damping_parity.py:255, 324` | `method="lm"` | LM damping parity |
| `tests/geo/test_lm_minpack_qr_parity.py:227` | `method="lm"` | LM/MINPACK parity |
| `tests/subprocess/import_smoke_cases.py:2069` | `method="lm"` | smoke |

**Risk:** medium. Critical correctness point: `Driver.SIMSOPT_LM_GMRES_HOST` is the lossless translation. **`Driver.SCIPY_LM` is NOT the migration target.** Spec §8 line 562 makes this explicit; reviewers reading the inventory must double-check no auto-replace tool changes `method="lm"` → `Driver.SCIPY_LM`.

The two `boozersurface.py:1251` and `boozersurface_jax.py:5798` lines are *kwarg defaults*, not call sites — they define the public method signature `minimize_boozer_penalty_constraints_ls(..., method="lm", ...)`. Migration here changes the default to a `Driver.*` value; the public method retains a single-method contract.

### `method="lm-ondevice"` → bridge: `Driver.SIMSOPT_LM_GMRES`; target: `Driver.OPTIMISTIX_LM` + `LinearSolver.LSMR`

| File:line | Form |
|---|---|
| `tests/geo/test_lm_optimistix_contract.py:206, 246` | `method="lm-ondevice"` |
| `tests/geo/test_single_stage_example.py:1475` | `method="lm-ondevice"` in `stage_callback` |
| `tests/test_benchmark_helpers.py:5389` | `boozer_optimizer_method="lm-ondevice"` |
| `tests/subprocess/import_smoke_cases.py:1021, 1058` | direct `jax_least_squares(method="lm-ondevice")` |

**Risk:** medium. Bridge driver covers the lossless path; target (Optimistix LSMR) is gated on the parity validation listed in spec §3 / §15.

### `method="lm-minpack-ondevice"` → bridge: `Driver.SIMSOPT_LM_QR`; target: `Driver.OPTIMISTIX_LM` + `LinearSolver.QR`

| File:line | Form |
|---|---|
| `tests/geo/test_lm_minpack_qr_parity.py:239, 468, 490, 511` | `method="lm-minpack-ondevice"` |

**Risk:** medium. Also requires the MINPACK info-code audit (do any callers depend on info=4 or info=8) before the bridge can be removed.

### `method="optimistix-lm-ondevice"` → `Driver.OPTIMISTIX_LM` (+ `LinearSolver.LSMR` default)

| File:line | Form |
|---|---|
| `tests/geo/test_lm_optimistix_contract.py:40, 138, 161` | `method="optimistix-lm-ondevice"` |

**Risk:** trivial. Already opt-in lane; only the dispatch string changes.

---

## 3. `optimizer_backend=` callers (23 files)

These are *wrapper-policy* uses (per spec §8 preamble) — passed into `BoozerSurfaceJAX(...)` LS options, into `resolve_optimizer_backend_method(...)`, or into benchmark/example CLI options. They are **not** direct `jax_minimize(...)` kwargs.

Production source:

| File | Use |
|---|---|
| `src/simsopt/geo/boozersurface_jax.py` | `BoozerSurfaceJAX` LS options (`__setattr__` hook, `optimizer_backend` in {`auto`, `scipy`, `ondevice`}) |
| `src/simsopt/geo/optimizer_jax.py` | API definition (`VALID_OPTIMIZER_BACKENDS`, `resolve_optimizer_backend_method`) |
| `src/simsopt/geo/optimizer_jax_private/_common.py` | private optimizer runtime gates |
| `src/simsopt/geo/surfaceobjectives_jax.py` | passes through to BoozerSurfaceJAX |

Benchmarks:

| File |
|---|
| `benchmarks/_cpp_compatible_probe.py` |
| `benchmarks/cpu_run_code_benchmark.py` |
| `benchmarks/gpu_run_code_benchmark.py` |
| `benchmarks/run_code_parity_probe.py` |
| `benchmarks/single_stage_backend_routing.py` |
| `benchmarks/single_stage_cpp_jax_state_parity.py` |

Examples:

| File |
|---|
| `examples/.../single_stage_banana_example.py` (**23 uses**) |

Tests:

| File |
|---|
| `tests/geo/test_boozersurface_jax.py` |
| `tests/geo/test_optimizer_jax_item19.py` |
| `tests/geo/test_single_stage_alm_integration.py` |
| `tests/geo/test_single_stage_example.py` |
| `tests/integration/test_factor_once_adjoint_phase2.py` |
| `tests/integration/test_section6_public_lane_split.py` |
| `tests/integration/test_single_stage_jax_cpu_reference.py` |
| `tests/integration/test_single_stage_physics_parity.py` |
| `tests/integration/test_stage2_jax.py` |
| `tests/test_backend.py` |
| `tests/test_benchmark_helpers.py` |
| `tests/test_cpp_compatible_probe_phase3.py` |

**Migration shape:** preserved as wrapper/CLI options until `BoozerSurfaceJAX` (and the benchmark/example helpers) move to the `outer_driver=` / `inner_driver=` typed contract per spec §10.

---

## 4. `least_squares_algorithm=` callers (13 files)

Values in use: `quasi-newton`, `lm`, `lm-minpack`, `optimistix-lm`.

| Category | Files |
|---|---|
| Production source | `src/simsopt/geo/boozersurface_jax.py`, `src/simsopt/geo/optimizer_jax.py` |
| Benchmarks | `benchmarks/single_stage_backend_routing.py`, `single_stage_outer_loop_probe.py`, `single_stage_smoke_fixture.py` |
| Examples | `examples/.../single_stage_banana_example.py` (6 uses), `examples/.../STAGE_2/banana_coil_solver.py` |
| Tests | `tests/geo/test_boozersurface_jax.py`, `tests/geo/test_single_stage_example.py`, `tests/integration/test_section6_public_lane_split.py`, `tests/integration/test_stage2_jax.py`, `tests/test_benchmark_helpers.py`, `tests/subprocess/section6_fixture_probe.py` |

**Migration shape:** the `least_squares_algorithm` string maps to `inner_driver=Driver.SIMSOPT_LM_GMRES_HOST | SIMSOPT_LM_GMRES | SIMSOPT_LM_QR | OPTIMISTIX_LM` per the `(optimizer_backend, least_squares_algorithm)` pair (spec §10).

---

## 5. `limited_memory=` callers (9 files)

`limited_memory=True` → L-BFGS-B family; `limited_memory=False` → dense BFGS family.

| Category | Files |
|---|---|
| Production source | `src/simsopt/geo/boozersurface_jax.py` (lines 5791, 5847 — Boozer inner-solve uses `False` at small n), `src/simsopt/geo/boozersurface.py` (upstream baseline) |
| Examples | `examples/.../single_stage_banana_example.py` |
| Tests | `tests/geo/test_boozersurface_jax_private.py`, `tests/geo/test_boozersurface_jax.py`, `tests/geo/test_single_stage_example.py`, `tests/test_benchmark_helpers.py`, `tests/geo/test_boozersurface_jax.py:3727` (`bogus` validation test) |
| Benchmarks | `benchmarks/single_stage_backend_routing.py` |

**Migration shape:** in the typed API, `limited_memory` collapses — `Driver.SCIPY_LBFGSB`/`SIMSOPT_LBFGSB` are the `True` branches; `Driver.SCIPY_BFGS`/`SIMSOPT_BFGS` are the `False` branches. Callers select the driver directly instead of toggling the boolean.

---

## 6. `BoozerSurfaceJAX(...)` constructor calls (14 files)

This is the Tier 3 wrapper migration surface. Per spec §10 the constructor will accept `outer_driver=` and `inner_driver=` instead of `optimizer_backend=` and `least_squares_algorithm=`.

Production source: `src/simsopt/geo/boozersurface_jax.py`.

Tests:
- `tests/geo/boozersurface_jax_test_helpers.py` (shared factory)
- `tests/geo/test_boozersurface_jax.py`
- `tests/geo/test_lm_damping_parity.py`
- `tests/geo/test_lm_minpack_qr_parity.py`
- `tests/geo/test_lm_optimistix_contract.py`
- `tests/geo/test_single_stage_example.py`
- `tests/integration/test_single_stage_jax_cpu_reference.py`

Benchmarks:
- `benchmarks/_cpp_compatible_probe.py`
- `benchmarks/parity/boozer_derivative_input_repro.py`
- `benchmarks/production_boozer_parity_probe.py`
- `benchmarks/run_code_benchmark_common.py`
- `benchmarks/run_code_parity_probe.py`
- `benchmarks/traceable_target_lane_compile_shape.py`

Examples: **none directly**. The single-stage banana example uses helper indirection rather than instantiating `BoozerSurfaceJAX` itself.

QFM/BiotSavart JAX constructors are confirmed to **not** accept `optimizer_backend` or `least_squares_algorithm` kwargs (verified at `qfmsurface_jax.py:56` and `biotsavart_jax_backend.py:330+`); the wrapper migration is narrowed to `BoozerSurfaceJAX` only.

---

## 7. `ReferenceOptimizerContract` / `TargetOptimizerContract` callers (5 files)

| File |
|---|
| `src/simsopt/geo/optimizer_jax.py` (definition) |
| `examples/.../single_stage_banana_example.py` |
| `tests/geo/test_optimizer_jax_item19.py` |
| `tests/geo/test_single_stage_alm_integration.py` |
| `tests/geo/test_single_stage_example.py` |

These contract dataclasses already serve as a typed selector at the boundary. They are the **natural anchor for the migration** — the migration replaces their internal `method: str` field with a `driver: Driver` field while preserving the dataclass shape externally.

---

## 8. Hot-spot files (highest call-density)

Ordered by total old-API touch points (method/backend/lsa/limited_memory/BoozerSurfaceJAX combined):

| Rank | File | Approximate touch points | Migration shape |
|---|---|---:|---|
| 1 | `examples/.../single_stage_banana_example.py` | 66+ | one large dedicated PR; CLI-driven, wire `Driver` enum through the option-resolution layer |
| 2 | `tests/geo/test_boozersurface_jax_private.py` | ~30 | private-optimizer parity tests; mechanical rewrite |
| 3 | `tests/geo/test_boozersurface_jax.py` | ~25 | broad coverage of every method string; PR per logical test class |
| 4 | `tests/geo/test_single_stage_example.py` | ~15 | exercise single-stage banana path; co-migrate with the example |
| 5 | `tests/test_benchmark_helpers.py` | ~10 | benchmark contract validation; preserves the benchmark CLI shape |
| 6 | `tests/subprocess/import_smoke_cases.py` | ~12 | import-time smoke; small mechanical PR |
| 7 | `tests/subprocess/jax_runtime_cases.py` | ~6 | runtime smoke |
| 8 | `tests/integration/test_single_stage_jax_cpu_reference.py` | ~5 | CPU↔JAX parity tests; verify shim translation byte-equality |
| 9 | `tests/geo/test_lm_optimistix_contract.py` | ~5 | already exercises the Optimistix lane |
| 10 | `tests/geo/test_lm_minpack_qr_parity.py` | ~4 | MINPACK-style LM parity; verify `lm-minpack-ondevice` bridge stays sound |

---

## 9. Reproduction commands

Every count and file:line above came from one of these commands. Run any of them to validate or refresh.

```sh
# Per-method-string per-line (excluding the API-defining files)
for pattern in 'method="lbfgs"' 'method="lbfgs-trace"' 'method="lbfgs-scipy-jax"' \
               'method="lbfgs-scipy-jax-fullgraph"' 'method="lbfgs-ondevice"' \
               'method="bfgs"' 'method="bfgs-ondevice"' 'method="adam"' \
               'method="adam-ondevice"' 'method="lm"' 'method="lm-ondevice"' \
               'method="lm-minpack-ondevice"' 'method="optimistix-lm-ondevice"'; do
  echo "## $pattern ##"
  grep -rn "$pattern" --include="*.py" src tests benchmarks examples scripts \
    | grep -v '^src/simsopt/geo/optimizer_jax\.py:\|^src/simsopt/geo/optimizer_jax_reference\.py:\|^src/simsopt/geo/optimizer_host_lbfgs\.py:\|^src/simsopt/geo/optimizer_jax_private/'
done

# Dispatch-function caller counts
for fn in jax_minimize jax_least_squares reference_minimize reference_least_squares \
          target_minimize target_least_squares; do
  printf "%-25s %d\n" "$fn" "$(grep -rn "$fn\s*(" --include="*.py" src tests benchmarks examples scripts | wc -l)"
done

# Files using each wrapper kwarg
grep -rln 'optimizer_backend\s*=\s*["'"'"']' --include="*.py" src tests benchmarks examples scripts | sort -u
grep -rln 'least_squares_algorithm\s*=' --include="*.py" src tests benchmarks examples scripts | sort -u
grep -rln 'limited_memory\s*=\s*\(True\|False\)' --include="*.py" src tests benchmarks examples scripts | sort -u
grep -rln 'BoozerSurfaceJAX\s*(' --include="*.py" src tests benchmarks examples scripts | sort -u
grep -rln 'ReferenceOptimizerContract\s*(\|TargetOptimizerContract\s*(' --include="*.py" src tests benchmarks examples scripts | sort -u

# Production-source LM callers (the load-bearing dispatch)
grep -n "method=.lm.\|reference_least_squares\|levenberg_marquardt" src/simsopt/geo/boozersurface_jax.py | head -20
```

---

## 10. Recommended PR sequence

Ordered by risk and dependency. Each row is one self-contained PR.

| # | PR | Risk | Reverts cleanly? |
|---|---|---|---|
| 1 | **Scaffolding**: create `src/simsopt/solve/jax/` package, `Driver` enum, dispatch table forwarding to existing `optimizer_jax.py` entry points, compat shim emitting `DeprecationWarning`. **No caller changes.** | Tier 2 (additive) | yes (single-file revert) |
| 2 | Migrate `tests/subprocess/import_smoke_cases.py` and `tests/subprocess/jax_runtime_cases.py` — smoke tests are the easiest sanity check for the shim. | Tier 1b | yes |
| 3 | Migrate `tests/geo/test_boozersurface_jax.py` (broad coverage; uses every method string family). | Tier 1b | yes (single test file) |
| 4 | Migrate `tests/geo/test_boozersurface_jax_private.py` (private optimizer parity; high LBFGS-ondevice density). | Tier 1b | yes |
| 5 | Migrate `tests/test_benchmark_helpers.py` and `tests/integration/test_single_stage_jax_cpu_reference.py`. | Tier 1b | yes |
| 6 | Migrate `tests/geo/test_lm_*.py` (LM parity contracts). | Tier 1b | yes |
| 7 | Migrate `tests/geo/test_single_stage_example.py`. | Tier 1b | yes |
| 8 | Migrate the benchmark CLI surface (`benchmarks/*.py`) — preserve CLI strings, translate at the option-resolution boundary. | Tier 1b–2 | yes |
| 9 | Migrate `examples/.../single_stage_banana_example.py`. Single largest PR (66 touch points). | Tier 2 | yes (single example) |
| 10 | Migrate `examples/2_Intermediate/stage_two_optimization_stochastic.py` (Adam example). | Tier 1b | yes |
| 11 | **Wrapper migration**: `BoozerSurfaceJAX` accepts `outer_driver` / `inner_driver` typed kwargs. Tier 3 (API change). Update all 14 `BoozerSurfaceJAX(...)` callers in tests/benchmarks. | Tier 3 | needs API-evolution gate |
| 12 | **OptimizerContract migration**: `ReferenceOptimizerContract`/`TargetOptimizerContract` move from `method: str` to `driver: Driver` fields. Update 5 callers. | Tier 3 | needs API-evolution gate |
| 13 | **Production-source migration**: `src/simsopt/geo/boozersurface_jax.py` (3 sites including the `minimize_boozer_penalty_constraints_ls` default), `src/simsopt/geo/surfaceobjectives_jax.py` (passthrough), `src/simsopt/geo/optimizer_jax_private/_common.py` (gates). | Tier 3 | needs gate; touches the public Boozer API |
| 14 | **Deprecation** — log entries should drop to zero for the deprecated strings across one release cycle. Then PR 15. | n/a | n/a |
| 15 | **Removal** — delete `simsopt.geo.optimizer_jax.{jax_minimize, jax_least_squares}` and the string-based dispatch. Replace with `ImportError` pointing at `simsopt.solve.jax`. | Tier 3 | irreversible — gate per spec §13 |

The bottom-of-stack (PRs 1–10) is **pure mechanical rewriting** with the shim catching any miss. The top (PRs 11–15) is real API change and needs the gate.

---

## 11. Open questions surfaced by the audit

| # | Question | Why it matters |
|---|---|---|
| Q1 | Does `tests/geo/test_boozersurface_jax.py:3727`'s `"bogus"` validation test (line `resolve_optimizer_backend_method("bogus", limited_memory=False)`) need a new equivalent for `Driver` validation? | The "bogus value rejected" contract is a real public API guarantee. The new API should have a typed equivalent (`ValueError` on unknown `Driver` value) and a test for it. |
| Q2 | Do any callers actually read MINPACK `info` codes 4 or 8 from `lm-minpack-ondevice` results? | Pre-spec §3 bridge-removal gate. If no caller depends on those codes, `SIMSOPT_LM_QR` can be retired after Optimistix QR parity. |
| Q3 | The single-stage banana example's CLI accepts `--optimizer-backend` and `--least-squares-algorithm`. Should the migrated CLI continue accepting old strings as aliases, or strictly take the new `Driver` enum names? | Affects external researchers running the autoresearch pipeline. If they have shell scripts pinning the old strings, the CLI alias is load-bearing. |
| Q4 | `tests/integration/test_section6_public_lane_split.py` and `tests/subprocess/section6_fixture_probe.py` reference a "Section 6 public lane" — what is this lane and does it set a backwards-compat requirement that the migration must preserve? | Audit needs to confirm this isn't a hidden contract surface. |
| Q5 | `src/simsopt/geo/boozersurface_jax.py:5798` is a public method default. Changing it from `method="lm"` to `driver=Driver.SIMSOPT_LM_GMRES_HOST` is a Tier 3 API change to the Boozer public surface. Are the call sites of `minimize_boozer_penalty_constraints_ls(...)` enumerated? | Required for the PR 13 gate. |
| Q6 | Several `subprocess` tests run smoke cases in fresh interpreter processes (`tests/subprocess/*.py`). These bypass any module-level fixture; do they exercise the compat shim from a fresh import? | Required to validate spec §9 ("shim guarantees one DeprecationWarning per process"). |

---

## 12. What this inventory does NOT cover

- **External scripts and notebooks.** Anything outside the repo tree (autoresearch run configs, researcher notebooks, downstream packages depending on `simsopt-jax`) is not in this audit. Spec §12 deprecation timeline assumes a structured-log signal will surface these; this audit cannot substitute for that.
- **Indirect callers via `**kwargs` forwarding.** If a helper accepts `**options` and forwards to `jax_minimize`, the inner `method=` string may be hidden behind data-driven dispatch. The grep finds the literal strings but not the dynamic ones.
- **Future research workloads.** The audit is a snapshot of the tree at HEAD as of 2026-05-19. New tests landing on `gpu-purity-stage2-20260405` between now and the migration ship will need a re-scan.
- **Test fixture file dependencies.** Some tests load JSON / pickle fixtures that may serialize method strings; those are not captured by source-grep.

---

## Appendix A: changelog

- 2026-05-19 v0 — initial audit produced by source-grep over HEAD `gpu-purity-stage2-20260405`.
