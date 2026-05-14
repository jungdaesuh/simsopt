# Crucible Review — JAX Silent-Fallback Removal

Date: 2026-05-13
Branch: gpu-purity-stage2-20260405
Reviewer: Crucible (adversarial verification gate)
Plan under review: `.artifacts/jax-silent-fallback-removal-2026-05-13/PLAN.md`
Patchset state: uncommitted working-tree changes against `HEAD = ca276abbd`
(two production-source files: `src/simsopt/backend/runtime.py`,
`src/simsopt/geo/optimizer_jax.py`), four new test files:
`tests/test_backend_strict_jax_device_detection.py`,
`tests/geo/test_optimizer_jax_silent_fallback_removal.py`,
`tests/jax_core/test_tree_signature.py`,
`tests/geo/test_surface_quadrature_grid_rejection.py`.

## Verdict: PASS

All `[x] applied` items in PLAN.md verified against the actual source. The
two source edits are sound. All 4 new test files pass under re-run
(105/105 in 2.63 s). Ruff check, ruff format, and `git diff --check`
return clean on all touched files. No new silent-fallback pattern was
introduced in the scoped JAX paths; `grep -rn "except Exception"` returns
zero hits across `simsopt.backend`, `simsopt.jax_core`,
`simsopt.geo/optimizer_jax*`, `simsopt.geo/boozersurface_jax`,
`simsopt.geo/surfaceobjectives_jax`, and `simsopt.objectives/fluxobjective_jax`.

## Findings

### Critical (must fix before merge)
- None.

### Major (should fix; explain if not fixing)
- None.

### Minor (nice to have)

1. **§5 partial-deny dtype gate.** The added gate is
   `np.dtype(x0.dtype).kind != "O"`, which rejects only object dtype.
   It does not explicitly reject `M` (datetime64), `m` (timedelta64),
   `S`/`U` (bytes/str). In current callers those dtypes would still
   reach `_optimizer_flat_vector` / `ravel_pytree` and raise loudly
   downstream, so this does not violate the §1 contract — but a
   strict allow-list (`kind in "biufc"`) would surface the contract
   violation one stack frame earlier. PLAN.md explicitly chose the
   deny-list form, so this is a documented trade-off rather than a
   defect. Recommended follow-up only.

2. **§4 cache-marker robustness.** `_cached_jit_value_and_grad` writes
   two different attributes via `setattr` (the marker
   `_CACHEABLE_VALUE_AND_GRAD_ATTR` upstream in
   `_mark_cacheable_jit_value_and_grad`, then
   `_CACHED_VALUE_AND_GRAD_ATTR` here). PLAN §4 argues "if the
   marker succeeded, the cache attr also succeeds." That is true for
   `__dict__`-based callables (the production case) but not strictly
   true for an object whose `__slots__` contains the marker name but
   not the cache name. No production caller has that shape, but a
   future `__slots__` instance that includes only the marker slot
   would surface `AttributeError` from the cache install — which is
   still the §1 contract behavior (loud failure), just not the same
   behavior the marker site exhibits. Acceptable; document as future
   audit hook.

3. **§7 detector tolerance.** `_canonicalize_traceable_exact_quadrature`
   uses `np.allclose(quadpoints_phi, shifted_half_period_phi)` with
   numpy defaults (`rtol=1e-05`, `atol=1e-08`). Any caller passing a
   visibly-different shifted phi grid falls through and raises
   `ValueError` from `_compute_stellsym_mask_indices_for_grid` —
   which is the documented loud-failure contract. The detector is
   correctly narrow: the §7 contract holds.

## Verified

### Contract compliance (PLAN.md §§1–9)

- **§2 non-JAX early-exit (`_build_sharding_tuning`).** Confirmed at
  `backend/runtime.py:974-990`. Non-JAX policy short-circuits with
  `strategy="none"`, `local_device_count=0`, `device_count=0`. No
  device probes invoked.
- **§2 narrow device probes to `RuntimeError`.** Confirmed at
  `backend/runtime.py:730-749`. `_detect_local/global_jax_device_count`
  catch only `ImportError`; any post-import `RuntimeError` from
  `jax.local_devices` / `jax.devices` propagates.
- **§2 drop `inspect_array_sharding_summary` catch.** Confirmed at
  `jax_core/sharding.py:401-417`. No `try/except`; pre-check on
  `isinstance(value, jax.Array)` gates the inspector call.
- **§2 drop `_jax_distributed_runtime_is_initialized` catch.**
  Confirmed at `backend/runtime.py:1294-1302`. Plain
  `return bool(is_initialized())` with no exception handling.
- **§3 narrow CUDA probe.** Confirmed at `backend/runtime.py:776-795`.
  `_detect_imported_jax_cuda_device_index` catches only
  `RuntimeError` from `local_devices(backend="gpu")`; the rest of
  the function uses `getattr(..., None)` guards.
- **§4 direct `setattr` (no wrapper).** Confirmed at
  `optimizer_jax.py:396-402, 405-408, 424-439` and
  `optimizer_jax_private/_common.py:205-228`. All four sites use
  unconditional `setattr` without `try/except`. Identity is
  preserved (`marked is fun`).
- **§4.1 strict `_field_dof_layout_version`.** Confirmed at
  `fluxobjective_jax.py:69-77`. `_strict_field_dof_layout_version`
  raises `NotImplementedError` if the attribute is missing or
  non-int. Production callers at L164 (`__init__`) and L300
  (drift check) use it.
- **§4.2 DELETE.** `_coerce_dense_hess_inv` and
  `_make_bfgs_continuation_state` removed; `grep -rn` across
  `src/simsopt` returns zero hits.
- **§5 `_is_flat_optimizer_vector` dtype gate.** Confirmed at
  `optimizer_jax.py:367-372`. New gate `np.dtype(x0.dtype).kind != "O"`
  rejects 1-D object-dtype ndarrays. Test
  `test_is_flat_optimizer_vector_rejects_object_dtype_array` passes
  after pycache invalidation.
- **§6 dead `except TypeError` removal.** All 5 helpers
  (`_runtime_cache_tree_signature` in `boozersurface_jax.py`;
  `_traceable_cache_tree_signature`, `_traceable_contract_tree_signature`,
  `_traceable_runtime_hostify_tree`, `_traceable_runtime_deviceify_tree`
  in `surfaceobjectives_jax.py`) call `tree_flatten`/`tree_map`
  directly with no `try/except`. JAX 0.10.0 contract that unregistered
  classes are leaves is pinned at runtime in
  `tests/jax_core/test_tree_signature.py::test_tree_flatten_treats_unregistered_class_as_leaf`.
- **§7 in-bundle canonicalization.** Confirmed at
  `surfaceobjectives_jax.py:1512-1561`. The `try/except ValueError`
  rescue is gone; replaced by an explicit `np.allclose` match against
  `Surface.get_phi_quadpoints(nphi, RANGE_HALF_PERIOD, nfp)`. Any
  unrecognized grid raises `ValueError` from
  `_compute_stellsym_mask_indices_for_grid`. Test
  `test_unknown_non_canonical_phi_grid_raises` pins this.
- **`_devices_for_platform` DELETE.** Confirmed at
  `jax_core/sharding.py:75-84`. No `try/except`; `RuntimeError`
  from `jax.devices(backend=...)` propagates.
- **§8 boundary-parser inline comments.** Verified at four sites:
  `_split_xla_flag_tokens` (L352-361, catches `ValueError` from
  `shlex.split`), `_parse_visible_cuda_device_index` (L762-773,
  catches `ValueError` from `int(...)`),
  `_parse_nvidia_smi_indexed_value_row` (L820-830, catches
  `ValueError` from `int(float(...))` / `float(...)`),
  `_query_gpu_metric_mb_from_nvidia_smi` (L833-857, catches
  `(FileNotFoundError, subprocess.CalledProcessError)`). Each
  comment accurately names the exception(s) caught and labels the
  rationale ("external-input parse contract" / "external-tool
  availability boundary"). Comments are accurate. No behavior
  change.
- **§9 sites kept unchanged.** Spot-checked
  `boozersurface_jax.py:676-682` (TypeError re-raise),
  `surfaceobjectives_jax.py:217-222` (KeyError → ValueError),
  `optimizer_jax.py:482-507` (lazy private-package import,
  re-raises non-package `ModuleNotFoundError`). All still in place.

### Source-edit soundness

- **`runtime.py` §8 comments.** Pure comment additions — no behavior
  delta. Each comment block accurately describes the narrow
  exception type and rationale.
- **`optimizer_jax.py` §5 dtype gate.** The change
  `return x0.ndim == 1 and np.dtype(x0.dtype).kind != "O"`:
  - `np.dtype(x0.dtype)` is a no-op for normal dtypes (numpy
    re-uses the dtype object) — verified empirically and in numpy
    docs.
  - For `jax.Array`, `arr.dtype` is a numpy dtype, so
    `np.dtype(arr.dtype).kind` is well-defined.
  - The §1 contract is preserved: object-dtype 1-D arrays now
    return False (rejecting silent acceptance), driving the caller
    through the pytree-adapter path which will raise loudly on
    object-dtype contents.
  - Does not break valid callers (the parametrize cases in
    `test_optimizer_jax_silent_fallback_removal.py` exercise
    jax.Array, np.ndarray, list/tuple of mixed scalars, ragged
    list, 2-D array, empty list, list-of-arrays, dict — all 11
    cases pass).

### Test-result verification

- **105/105 new tests pass** under
  `.conda/jax-0.9.2/bin/python -m pytest tests/test_backend_strict_jax_device_detection.py tests/geo/test_optimizer_jax_silent_fallback_removal.py tests/jax_core/test_tree_signature.py tests/geo/test_surface_quadrature_grid_rejection.py -q`
  in **2.63 s**. An initial run reported 1 failure because of stale
  `.pyc` files dating from the pre-edit module; clearing
  `__pycache__` resolved it. This is a session hygiene note, not a
  patch defect — the on-disk source has the §5 change and the
  test passes against it.
- **ruff check** on six touched files (2 source + 4 tests): `All
  checks passed!`
- **ruff format --check** on the same six files: `6 files already
  formatted`.
- **git diff --check**: no whitespace errors, no merge markers.
- **Adversarial grep** for new broad except clauses in scoped JAX
  paths: `grep -rn "except Exception"` against
  `src/simsopt/{backend,jax_core,geo/optimizer_jax.py,geo/optimizer_jax_private,geo/optimizer_jax_reference.py,geo/boozersurface_jax.py,geo/surfaceobjectives_jax.py,objectives/fluxobjective_jax.py,field/biotsavart_jax_backend.py,field/_jax_common.py,geo/_distance_jax.py,backend.py}`
  returned zero hits.

### Test-quality spot checks (per file)

- **`tests/test_backend_strict_jax_device_detection.py`** — 38 tests.
  Each test exercises the real runtime function. Monkeypatches use
  `sys.modules['jax']` substitution (real `import jax` path) and
  `subprocess.run` substitution (real boundary path). The
  `inspect_array_sharding_summary` test (L286-309) patches
  `sharding_module.jax.debug.inspect_array_sharding`, which is the
  exact symbol the function reads via `getattr(getattr(jax, "debug",
  None), "inspect_array_sharding", None)`. No mocks hide real
  behavior. Parametrize for `(local_devices, devices)` and for
  garbage env-value cases is non-bogus; each value tests a
  distinct branch (sentinel, UUID, MIG-UUID, whitespace).
- **`tests/geo/test_optimizer_jax_silent_fallback_removal.py`** — 25
  tests. `_SlottedCallable` correctly forces `AttributeError` on
  marker write (slot layout excludes the marker name). §4.2 deletion
  tests check both `hasattr(private_pkg, ...)` and direct
  `from ..._result_converters import _coerce_dense_hess_inv`
  raising `ImportError` — defends against partial re-introduction.
  §5 test cases cover every accepted/rejected branch.
- **`tests/jax_core/test_tree_signature.py`** — 37 tests. The
  `_register_flatten_bomb()` helper creates a fresh registered
  pytree class per test (no public unregister), so duplicate
  registration is avoided. The `_ALL_HELPERS` parametrize includes
  both signature builders and transforms — the rationale documented
  inline. JAX 0.10 contract is pinned at runtime
  (`test_tree_flatten_treats_unregistered_class_as_leaf`).
- **`tests/geo/test_surface_quadrature_grid_rejection.py`** — 5
  tests. Uses real `Surface.get_phi_quadpoints` /
  `Surface.get_theta_quadpoints`; no mocks. Each test pins a
  distinct §7 contract clause (canonical passthrough, shifted-VMEC
  rewrite, unknown rejection, non-stellsym passthrough, mask
  equivalence after rewrite). No silent skip/xfail.

### Pre-existing failures (not from this work)

Per the brief these were verified pre-existing via `git stash` in the
prior session. I did not re-run the slow suites
(`tests/geo/test_boozersurface_jax.py`,
`tests/integration/`, full smoke); the brief explicitly does not
require it. Flagging here so they are not attributed to this patchset:

- `tests/geo/test_surface_objectives_jax.py` — 2 failures (pre-existing).
- `tests/geo/test_boozersurface_jax.py` — 1 Newton-polish failure (pre-existing).
- `tests/test_jax_import_smoke.py` — 1 timeout
  (`test_grouped_biot_savart_coil_collective_parity_and_lowering`,
  per PLAN.md §7 "Test status" — pre-existing flake under the 60 s
  pytest timeout when run in the full smoke suite).

Targeted re-runs done during this review:

- `tests/geo/test_surface_objectives_jax.py -k "mark_cacheable or value_and_grad"`:
  7 passed, 238 deselected (8.63 s) — confirms the §4 cache-marker
  edits do not regress the closest surface-objectives tests.
- `tests/geo/test_surface_objectives_jax.py -k canonicalize`:
  no matches (245 deselected) — string references at L2163/L2298 are
  fixtures, not test selectors, so no relevant test sub-suite to
  exercise.

## Recommendations

1. **Pycache hygiene in CI.** The initial test re-run produced a
   spurious failure because a stale
   `src/simsopt/geo/__pycache__/optimizer_jax.cpython-311.pyc`
   shadowed the §5 edit (the `.pyc` mtime was 1 minute older than
   the `.py`, but Python loaded the cached bytecode anyway). Recommend
   adding `find src tests -name "*.pyc" -delete` to the validation
   gate before running the new tests, or invoking
   `PYTHONDONTWRITEBYTECODE=1`. This is a process improvement, not
   a patch defect.

2. **Tighten §5 to an allow-list (future).** Replace the deny-list
   gate with `kind in "biufc"` so date/string/bytes 1-D arrays are
   rejected at the same layer as object-dtype arrays. Current
   behavior is acceptable under the §1 contract (downstream
   `_optimizer_flat_vector` raises), but the loud failure surfaces
   one stack frame later than ideal.

3. **Cosmetic rename of `_canonicalize_traceable_exact_quadrature`.**
   PLAN.md §7 "Open follow-ups" already notes that the function name
   is mildly misleading now that it literally canonicalizes. Defer
   to a wider naming sweep.

4. **§4 future audit hook.** If a future caller passes a
   `__slots__` instance that includes the marker slot but not the
   cache slot, the cache-install `setattr` at
   `optimizer_jax.py:438` and
   `optimizer_jax_private/_common.py:223` will raise
   `AttributeError`. That is still the §1 contract (loud), but
   distinct from the marker-write behavior. Document this
   asymmetry in the §4 inline comments if a future audit revisits
   the section.
