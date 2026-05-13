# JAX Silent-Fallback Removal — Plan & TODOs

Date: 2026-05-13
Branch: `gpu-purity-stage2-20260405`
Scope: JAX-ported code under `src/simsopt/{backend,jax_core}/`,
`src/simsopt/{field,geo,objectives,solve}/*_jax*.py`,
`src/simsopt/{field/_jax_common.py,geo/_distance_jax.py}`,
`src/simsopt/backend.py`,
`src/simsopt/geo/optimizer_jax_private/`.

---

## Context

The JAX port has accumulated 28 `try`/`except` sites and several
silent-default `getattr(..., None)` patterns. Many swallow JAX runtime
or optimizer cache failures and return a "neutral" value (`0`, `None`,
`False`, `()`, identity matrix) instead of surfacing the broken
contract.

The codebase already has the correct seam for compatibility paths:

- `src/simsopt/backend/runtime.py:1604-1633`
  - `raise_if_strict_jax_fallback(component, detail)` — raises `RuntimeError` in strict mode.
  - `warn_if_jax_fallback(component, detail)` — one-shot `RuntimeWarning` otherwise.

Existing call sites already route through this API
(`geo/boozersurface_jax.py:4908-4921`,
`geo/optimizer_jax.py:251-260, 3327`,
`geo/optimizer_jax_reference.py:404-408, 491-495`,
`geo/surfaceobjectives_jax.py:3465-3472`). The work below extends this
contract to the remaining sites and removes the silent-neutral returns.

A re-audit on 2026-05-13 found **zero** prior findings retired. Two new
silent-fallback sites were added under
`src/simsopt/geo/optimizer_jax_private/`. Acting now keeps the pattern
from spreading further.

> **Line-number stability note.** All file:line references in this
> document are point-in-time from 2026-05-13 and shift as the
> implementation lands. Always `grep -n` the symbol name before
> editing rather than trusting a line ref.

---

## Implementation status (2026-05-13 working tree)

Sections marked `applied` are visible in `git diff` at the working-tree
root; `pending` items have no code change yet.

| Section | Status | Notes |
|---|---|---|
| §1 contract | applied (doc-only) | Bullets below mirror what landed in code |
| §2 non-JAX early-exit (`_build_sharding_tuning`) | applied | Sharding tuning now skips device probes when `policy.backend != "jax"` |
| §2 narrow device probes to `RuntimeError` | applied | `_detect_local/global_jax_device_count` no longer swallow `Exception` |
| §2 drop `inspect_array_sharding_summary` catch | applied | Pre-check on `isinstance(value, jax.Array)` makes the catch dead |
| §2 drop `_jax_distributed_runtime_is_initialized` catch | applied | `is_initialized()` never raises (verified against JAX 0.10.0) |
| §3 narrow CUDA probe | applied | `_detect_imported_jax_cuda_device_index` narrowed to `RuntimeError` |
| §4 cache wrapper | resolved (no wrapper) | First attempt added a `_CacheableMarkedCallable` class; broke an identity assertion. Reverted to direct `setattr` without `try/except`. All production callers pass Python functions where `setattr` succeeds; a future builtin caller would surface `AttributeError`. Test passes. |
| §4.1 strict `_field_dof_layout_version` | applied | Helper `_strict_field_dof_layout_version` added at `fluxobjective_jax.py:69` |
| §4.2 hybrid BFGS Hessian-continuation | applied as DELETE | Both `_coerce_dense_hess_inv` and `_make_bfgs_continuation_state` removed (dead-code path, zero external callers) |
| §5 tighten `_is_flat_optimizer_vector` | applied | Explicit `(int, float, np.generic)` element check |
| §6 delete dead `except TypeError` branches | applied | All 5 sites cleaned inline — no shared helper module |
| §7 quadrature grid | applied as in-bundle canonicalization (2026-05-13 round 3) | Round 1 silent rescue deleted; round 2 fully pure adapter exposed `test_transfer_guard_disallow_...`; round 3 reinstates a narrowly-scoped, explicit canonicalization step inside `_canonicalize_traceable_exact_quadrature` for the documented `Surface.get_phi_quadpoints(RANGE_HALF_PERIOD)` family. Any other unrecognized grid still raises. See §7 for the option-1/2/3 trade-off |
| `_devices_for_platform` | applied as DELETE (2026-05-13 round 2) | Earlier narrowing to `except RuntimeError: return ()` flagged by reviewer as still being neutral-return; now the `RuntimeError` propagates — config-error-on-CUDA-host is loud |
| §8 boundary-parser annotations | pending | Inline comments only — does not change behavior |
| Validation suite | pending | See "Validation gates" |

### Parallel review pass (2026-05-13)

Three parallel reviewers (JAX docs verification, downstream consumer
enumeration, math/computation review) returned with these
simplifications. Most were applied as DELETE per KISS/YAGNI; §7
landed differently after the round-2 regression — see
"Implementation status" above for the final state per section.

- **§4.2 → DELETE applied.** `_coerce_dense_hess_inv` and its caller
  `_make_bfgs_continuation_state` had zero external callers. The
  function dressed a hard BFGS reseed (identity ``H_0`` per Nocedal
  & Wright Alg. 6.1) as a "continuation" — mathematically valid but
  mislabeled. Both deleted along with the now-unused `warnings` /
  `jnp` imports in `_result_converters.py`. The §4.2 body below is
  preserved as the *historical* strict-fallback-routing alternative
  in case future requirements re-introduce the function.
- **§7 → in-bundle canonicalization (round 3).** Rounds 1–2
  attempted a clean DELETE; round 2 surfaced a real downstream test
  regression because `build_real_single_stage_init_fixture` supplies
  a shifted VMEC half-period grid that the rescue had been
  canonicalizing. Round 3 keeps the §7 source change for the silent
  `try/except` rescue (gone), but adds one explicit detection step
  inside `_canonicalize_traceable_exact_quadrature` that recognizes
  the named family
  (`Surface.get_phi_quadpoints(nphi, RANGE_HALF_PERIOD, nfp)`) via
  `np.allclose` and substitutes the unshifted canonical
  `half_phi`/`full_theta`. Any other unrecognized grid still raises
  `ValueError` from `_compute_stellsym_mask_indices_for_grid`. See
  §7 below for the full rationale and the rejected options 2/3.
- **§6 → DELETE applied.** Verified against JAX 0.10.0:
  `jax.tree_util.tree_flatten` and `tree_map` treat unregistered
  classes as leaves; they do not raise `TypeError`. All 5
  `except TypeError` branches were unreachable. Removed inline at
  each site. **No shared `_tree_signature.py` module created** — the
  helper proposal in §6 body is YAGNI for one-line `tree_flatten` /
  `tree_map` wrappers.
- **§2 narrow, don't drop.** `jax.devices()` / `jax.local_devices()`
  raise specifically `RuntimeError` for unavailable backends.
  `jax.debug.inspect_array_sharding` raises `AttributeError` on
  non-JAX inputs, but the `inspect_array_sharding_summary` pre-check
  (`isinstance(value, jax.Array)`) makes that branch unreachable —
  catch was **dropped** rather than narrowed. `jax.distributed.is_initialized()`
  never raises (returns `bool`) — catch dropped.
- **§4 — direct `setattr`, no wrapper.**
  `_mark_cacheable_jit_value_and_grad` is used as a decorator at
  `surfaceobjectives_jax.py` (decorator + multiple call sites),
  `tests/subprocess/jax_runtime_cases.py` (4 sites),
  `boozersurface_jax.py`, and `stage2_target_objective_jax.py`
  (3 sites). All call sites pass plain Python callables. A
  short-lived `_CacheableMarkedCallable` wrapper experiment was
  reverted because it broke `bundle["..."] is marked["fun"]`
  identity assertions; direct `setattr` without `try/except` is
  KISS and fails loudly for any future builtin/`__slots__` caller.
- **§4 thread-safety verified.** Existing `threading.Lock()`
  (non-reentrant) is correct: double-checked install pattern, build
  outside the lock. Pattern preserved in
  `_cached_jit_value_and_grad` and `_cached_private_solver`.

Six earlier findings validated against the code and folded into the sections
below:
1. §2 splits non-JAX backend (early-exit, no probing) from JAX-mode
   failures (raise). Required because `_build_sharding_tuning` at
   `backend/runtime.py:964-977` still probes JAX devices even when
   `policy.backend != "jax"` forces `strategy = "none"`.
2. §6 separates **signature builders** (value-hash vs structural-meta)
   from **tree transforms** (hostify/deviceify). The original
   "one SSOT helper" collapsed three distinct contracts. The two
   signature builders have different leaf semantics (value hash at
   `surfaceobjectives_jax.py:1709-1723` vs structural meta at
   `:1739-1770`) and the hostify/deviceify helpers are `tree_map`
   transforms, not signature builders.
3. §4 enumerates the cache-marker attributes that consumers read
   directly via `getattr` on the marked callable:
   `_CACHEABLE_VALUE_AND_GRAD_ATTR`, `_CACHED_VALUE_AND_GRAD_ATTR`,
   `_STRUCTURED_SOLVER_CACHE_TOKEN_ATTR`, `_PRIVATE_SOLVER_CACHE_ATTR`,
   and the separate `_simsopt_value_and_grad` flag. Tests at
   `tests/geo/test_surface_objectives_jax.py:889, 4156` inspect these
   attributes directly. Identity is preserved (no wrapper), so
   `marked is fun`.
4. §4.2 originally proposed routing the three Hessian-continuation
   branches through `warn_if_jax_fallback()` /
   `raise_if_strict_jax_fallback()`. **Superseded.** Reviewer C
   showed `_coerce_dense_hess_inv` and its caller were dead code,
   so both were deleted instead. The strict-fallback routing
   write-up is preserved under §4.2 as the historical alternative
   if the function ever returns.
5. Validation gates updated to the prefix-based env (`conda run -p
   .conda/jax-0.9.2` or `.conda/jax-0.9.2/bin/python`) because this
   checkout's env is at `/Users/suhjungdae/code/columbia/simsopt-jax/.conda/jax-0.9.2`,
   not a named `jax-0.9.2`.
6. `inspect_array_sharding_summary` directive made consistent across
   §2 and the success criteria: narrow or surface, never broad.

---

## Contract (apply consistently)

- [ ] JAX backend/runtime failures **raise**; they do not become `0`, `None`, `False`, or `()`.
- [ ] Optional external-tool availability may return `None` only at real process/tool boundaries (e.g. `nvidia-smi` not installed, `CUDA_VISIBLE_DEVICES` unparsable).
- [ ] Pytree/cache helpers use explicit type/attribute checks, **not** `try/except` as control flow.
- [ ] Compatibility fallbacks route through `raise_if_strict_jax_fallback()` / `warn_if_jax_fallback()` or are deleted.
- [ ] No new broad `except Exception:` clause in JAX-scoped paths.

---

## §2. Harden runtime and sharding device detection

Files:
- `src/simsopt/backend/runtime.py`
- `src/simsopt/jax_core/sharding.py`

**Precondition — non-JAX backends must not probe JAX at all.** Today
`_build_sharding_tuning` at `backend/runtime.py:964-977` forces
`strategy = "none"` for `policy.backend != "jax"` but still calls
`_detect_local_jax_device_count(policy)` and
`_detect_global_jax_device_count(policy)`. If we make those raise on
post-import JAX failures, non-JAX consumers break. So the contract is:

- Non-JAX path: never probe JAX. Skip the device-count calls.
- JAX path: device probes must raise on real failures.

- [ ] **`backend/runtime.py:964-977`** `_build_sharding_tuning` — when
  `policy.backend != "jax"`, return early with `strategy = "none"`,
  `local_device_count = 0`, `device_count = 0`, and no probe calls.
  Audit other callers of `_detect_local_jax_device_count` /
  `_detect_global_jax_device_count` and apply the same non-JAX
  early-exit so the JAX-mode probes can be made strict.
- [ ] **`backend/runtime.py:728-735`** `_detect_local_jax_device_count`
  — keep `except ImportError: return 0` as a **boundary parser**
  (simsopt is importable without JAX installed; this is the same
  category of contract as nvidia-smi-not-on-PATH in §8). Remove the
  broad `except Exception: return 0`; let `jax.local_devices(...)`
  raise `RuntimeError` directly when the JAX backend exists but
  enumeration fails. (The §2 precondition above already guarantees
  this function is only called in JAX mode.)
- [ ] **`backend/runtime.py:740-747`** `_detect_global_jax_device_count`
  — same treatment.
- [x] **`jax_core/sharding.py:78-84`** `_devices_for_platform` —
  catch dropped entirely. Caller has already gated on JAX mode and
  resolved `policy.jax_platform`; if `jax.devices(backend=...)`
  raises `RuntimeError`, the requested platform is genuinely
  unavailable (config error, not graceful degradation). The
  `RuntimeError` propagates and surfaces as a loud failure rather
  than degrading to no-mesh placement.
- [ ] **`jax_core/sharding.py:410-413`**
  `inspect_array_sharding_summary` — **DROP the catch** (decided).
  Reviewer A confirmed: `jax.debug.inspect_array_sharding` raises
  `AttributeError` only for non-jax inputs, but this function
  pre-checks `isinstance(value, jax.Array)` before calling
  `inspect_fn`. The catch is unreachable in production. Removing it
  surfaces a real bug (e.g., a future JAX version regression in
  `inspect_array_sharding`) rather than masking it. Annotation alone
  does not satisfy the contract.
- [ ] **`backend/runtime.py:1289-1291`**
  `_jax_distributed_runtime_is_initialized` — keep the `return False`
  pre-checks when `jax`/`distributed`/`is_initialized` are absent, but
  remove the `except Exception` around the actual `is_initialized()`
  call. A working `is_initialized()` raising is a real bug, not a
  diagnostic miss.
- [ ] Add unit tests that monkeypatch `jax.devices`,
  `jax.local_devices`, and `jax.distributed.is_initialized` to raise
  and assert the error is surfaced (not swallowed) under
  `backend == "jax"`. Also test `backend != "jax"` and assert no JAX
  probe is performed (e.g. via monkeypatch counting calls).

---

## §3. Make CUDA runtime probing explicit

File:
- `src/simsopt/backend/runtime.py`

Note: the existing function split is already close to the target
(`_detect_imported_jax_cuda_device_index`, `_visible_cuda_device_selector`,
`_parse_visible_cuda_device_index`, `_detect_active_jax_cuda_device_index`,
`_detect_active_jax_cuda_device_selector`). The substantive work is
removing the `except Exception` swallows, not renaming.

- [ ] **`backend/runtime.py:782-785`** `_detect_imported_jax_cuda_device_index`
  — remove the `except Exception: return None` around
  `is_initialized()`; only treat "not callable" / "not enabled" as
  None.
- [ ] **`backend/runtime.py:790-792`** same function — remove
  `except Exception: return None` around `local_devices(backend="gpu")`.
  A JAX-imported runtime where `local_devices` raises is a hard
  failure.
- [ ] **Keep** `backend/runtime.py:765-767`
  (`int(CUDA_VISIBLE_DEVICES)` `except ValueError`) — env parsing
  contract.
- [ ] **Keep** `backend/runtime.py:822-824` (nvidia-smi row
  `except ValueError`) — output parsing contract.
- [ ] **Keep** `backend/runtime.py:839-846` (`subprocess.run` of
  `nvidia-smi` with `(FileNotFoundError, subprocess.CalledProcessError)`)
  — external-tool availability boundary.
- [ ] Add tests for: jax-imported + `local_devices` raises → assert
  raise; CUDA_VISIBLE_DEVICES garbage → assert `None`; no nvidia-smi
  on PATH → assert `None`.

---

## §4. Clean optimizer cache tagging

Files:
- `src/simsopt/geo/optimizer_jax.py`
- `src/simsopt/geo/optimizer_jax_private/_common.py`
- `src/simsopt/geo/optimizer_jax_private/_lbfgs.py` (consumer)
- `src/simsopt/geo/surfaceobjectives_jax.py` (consumer)

There were **four** copies of the same anti-pattern:
`try: setattr(... cache ...) except (AttributeError, TypeError):
pass | return compiled`.

### Resolution: direct `setattr`, no wrapper (applied)

The `try/except` is removed at all four sites. `setattr` runs
unconditionally on the marker attribute. Production callers
(`surfaceobjectives_jax.py` decorator + call sites,
`stage2_target_objective_jax.py`, `boozersurface_jax.py`,
`tests/subprocess/jax_runtime_cases.py`) all pass plain Python
callables on which `setattr` always succeeds. A future caller
passing a builtin or `__slots__` instance now raises `AttributeError`
loudly — the desired §1 contract behavior.

Marker attributes (`_CACHEABLE_VALUE_AND_GRAD_ATTR`,
`_CACHED_VALUE_AND_GRAD_ATTR`,
`_STRUCTURED_SOLVER_CACHE_TOKEN_ATTR`,
`_PRIVATE_SOLVER_CACHE_ATTR`) are written and read directly on the
callable; identity is preserved (`marked is fun`), so existing
identity-based test assertions continue to hold.

The cache install pattern remains a double-checked
`threading.Lock()` (`_SCALAR_VALUE_AND_GRAD_CACHE_LOCK` for the
public path, `_PRIVATE_SOLVER_CACHE_LOCK` for the private path),
with the compiled function built outside the lock and only
installed inside.

The `_CacheableMarkedCallable` wrapper class experiment is
documented under "Test identity conflict" below for historical
reference and rejected. No wrapper module is needed.

### Test identity conflict (resolved 2026-05-13 round 2)

Option (b) was chosen: **wrapper class reverted; direct `setattr`
without `try/except`.** All current production callers
(`surfaceobjectives_jax.py`, `stage2_target_objective_jax.py`,
`boozersurface_jax.py`, `tests/subprocess/jax_runtime_cases.py`)
pass plain Python callables that accept `setattr`. A future caller
passing a builtin or `__slots__` instance will surface
`AttributeError` immediately — the desired fail-loud behavior under
the §1 contract. The test at
`tests/geo/test_surface_objectives_jax.py:885` now passes unchanged.

The wrapper-class alternative is rejected because it solves a
problem that does not exist in current usage (no production caller
passes a non-Python callable) while breaking identity preservation.

### Sites fixed (applied)

- [x] **`_mark_cacheable_jit_value_and_grad`** — `setattr(fun, _CACHEABLE_VALUE_AND_GRAD_ATTR, True)` runs unconditionally; `try/except` removed.
- [x] **`_mark_structured_private_solver_cacheable`** — same treatment; `setattr(fun, _STRUCTURED_SOLVER_CACHE_TOKEN_ATTR, cache_token)` runs unconditionally.
- [x] **`_cached_jit_value_and_grad`** — outer `try/except (AttributeError, TypeError): return compiled` removed; the lock acquisition cannot raise those, and `setattr` is now guaranteed-safe because the marker check above only matches callables that already accepted a `setattr` for the marker.
- [x] **`_cached_private_solver` in `optimizer_jax_private/_common.py`** — fourth copy of the pattern cleaned the same way.
- [x] **`surfaceobjectives_jax.py:1705`** `value_and_grad._simsopt_value_and_grad = True` — left as a plain attribute assignment on a closure (no wrapper). The `_simsopt_value_and_grad` flag is a separate marker from the cache markers and is consumed only at `:1835`; no wrapper conformance issue.
- [x] Monkeypatch flow at `tests/geo/test_surface_objectives_jax.py:850-863` verified by `pytest -k test_traceable_objective_bundle_marks_value_and_grad_cacheable` → passes.
- [ ] Add a focused test asserting that a callable without `__dict__` (e.g., a `__slots__` class instance) now raises `AttributeError` from `_mark_cacheable_jit_value_and_grad` — documents the loud-failure contract.

### §4.1 SquaredFluxJAX field DOF layout version

File:
- `src/simsopt/objectives/fluxobjective_jax.py`

The field always sets `_dof_layout_version = 0`
(`field/biotsavart_jax_backend.py:417, 915, 937`). The `None` default
silently passes drift detection if the attribute is ever missing
(`None == None`).

- [ ] **`fluxobjective_jax.py:153`** replace
  `getattr(field, "_dof_layout_version", None)` with a required-attribute
  fetch raising `NotImplementedError` (same shape as
  `_strict_field_coil_dof_extraction_spec` at lines 59-66).
- [ ] **`fluxobjective_jax.py:289`** same treatment.
- [ ] Test that a field-like mock without `_dof_layout_version` raises
  at construction time, not silently at drift-check time.

### §4.2 Hybrid BFGS Hessian-continuation (applied — DELETE)

The applied fix was **DELETE** (reviewer C: zero external callers,
function dresses a hard BFGS reseed as a continuation). The
strict-fallback routing described below is preserved only as a
*historical alternative* in case a future requirement re-introduces
`_make_bfgs_continuation_state`.

Files removed/cleaned:
- `src/simsopt/geo/optimizer_jax_private/_result_converters.py`:
  `_coerce_dense_hess_inv` and the unused `warnings` / `jnp` imports
  deleted.
- `src/simsopt/geo/optimizer_jax_private/_bfgs.py`:
  `_make_bfgs_continuation_state` deleted; the
  `from ._result_converters import _coerce_dense_hess_inv` removed.
- `src/simsopt/geo/optimizer_jax_private/__init__.py`: both symbols
  removed from the package re-exports.

<details>
<summary>Historical strict-fallback routing (not applied)</summary>

If a future change re-introduces the function, route the three
branches through `raise_if_strict_jax_fallback()` /
`warn_if_jax_fallback()` rather than raw `warnings.warn`. Use
distinct `component` namespaces so `_warned_jax_fallbacks` dedup
treats them independently:
`"hybrid_bfgs.hess_inv.missing"`,
`"hybrid_bfgs.hess_inv.densify_failed"`,
`"hybrid_bfgs.hess_inv.shape_mismatch"`.

</details>

---

## §5. Remove ambiguous flat-vector detection

File:
- `src/simsopt/geo/optimizer_jax.py`

- [ ] **`optimizer_jax.py:368-377`** `_is_flat_optimizer_vector` — replace
  `try: np.asarray(x0) except Exception: return False` with explicit
  accepted types:
  - `jax.Array` with `ndim == 1` (already handled).
  - `np.ndarray` with `ndim == 1` (already handled).
  - `list` / `tuple` of `(int, float, np.generic)` — preserve current
    mixed-numeric acceptance (`np.float64` mixed with Python floats
    works today).
  - Anything else → `False`, **not** a swallowed exception.
- [ ] For ragged / object-dtype input that previously returned `False`,
  decide at the call site of `_prepare_optimizer_pytree_adapter`: it
  already accepts non-flat as pytree, so `False` remains correct;
  document the path.
- [ ] Tests: ragged list, object array, mixed `np.float64` + `float`,
  empty list, list of arrays.

---

## §6. Make pytree signature helpers deterministic

Files:
- `src/simsopt/geo/boozersurface_jax.py`
- `src/simsopt/geo/surfaceobjectives_jax.py`

Five duplicated defensive patterns, but **three distinct contracts**.
Do not collapse them into one `signature_leaf()`:

| Site | Kind | Leaf semantics |
|---|---|---|
| `boozersurface_jax.py:806-815` (`_runtime_cache_tree_signature`) | signature | value hash (`_runtime_cache_leaf_signature` at `boozersurface_jax.py:778-803` uses repr/scalar value) |
| `surfaceobjectives_jax.py:1726-1736` (`_traceable_cache_tree_signature`) | signature | **value hash** via `blake2b(array.tobytes())` for arrays — `_traceable_cache_leaf_signature` at `:1709-1723` |
| `surfaceobjectives_jax.py:1773-1783` (`_traceable_contract_tree_signature`) | signature | **structural meta only** — `_traceable_contract_leaf_signature` at `:1739-1770` returns `(shape, dtype)` for non-scalar arrays, scalar value only for size ≤ 1 |
| `surfaceobjectives_jax.py:1802-1807` (`_traceable_runtime_hostify_tree`) | **transform** | `tree_map(_traceable_runtime_hostify_leaf, tree)` — moves arrays to host |
| `surfaceobjectives_jax.py:1821-1826` (`_traceable_runtime_deviceify_tree`) | **transform** | `tree_map(_traceable_runtime_deviceify_leaf, tree)` — moves arrays to device |

Each uses `try: jax.tree_util.tree_{flatten,map}(tree) except TypeError:
leaf_fn(tree)`. JAX treats unregistered classes as leaves rather than
raising, so the `except TypeError` branch is unreachable for typical
objects; replacing with explicit checks makes this auditable.

### Plan (applied — KISS / YAGNI)

The originally-proposed `jax_core/_tree_signature.py` module was
**rejected**: it would wrap two one-line JAX calls behind named
helpers that add no behavior. Reviewer A confirmed
`tree_flatten`/`tree_map` never raise `TypeError` for unregistered
classes, so the `except TypeError` branches are dead. The applied
fix is to delete each branch inline; the surrounding signature/
transform functions are unchanged otherwise.

- [x] **`boozersurface_jax.py`** `_runtime_cache_tree_signature` —
  `try/except TypeError` removed; calls `jax.tree_util.tree_flatten`
  directly.
- [x] **`surfaceobjectives_jax.py`** `_traceable_cache_tree_signature`,
  `_traceable_contract_tree_signature`,
  `_traceable_runtime_hostify_tree`,
  `_traceable_runtime_deviceify_tree` — same treatment at each site.
- [x] All five distinct leaf functions (value-hash, structural-meta,
  hostify, deviceify, runtime-cache) retained unchanged — their
  semantics are intentionally different.
- [ ] Add a focused regression test in
  `tests/jax_core/test_tree_helpers.py` (smaller scope than the
  original `test_tree_signature.py` proposal):
  - `tree_flatten` on an unregistered class returns `[obj]` and a
    leaf treedef (documents the JAX 0.10 contract we're relying on);
  - sentinel test that registers a pytree node with a flatten that
    raises `RuntimeError`, asserting the signature helper propagates
    (no silent catch).

---

## §7. Surface quadrature — in-bundle canonicalization (applied 2026-05-13 round 3)

### Resolution chosen

`_canonicalize_traceable_exact_quadrature` is no longer a pure
adapter. It now detects exactly one known input family — the VMEC
half-period shifted phi grid produced by
`Surface.get_phi_quadpoints(nphi, RANGE_HALF_PERIOD, nfp)` — and,
if `np.allclose` matches, substitutes both `quadpoints_phi` and
`quadpoints_theta` with the unshifted canonical
`half_phi`/`full_theta` family. Any other input falls through to
`_compute_stellsym_mask_indices_for_grid` and raises `ValueError`
loudly if the grid is unrecognized.

This is option 1 from the previous round, applied **inside the
traceable bundle** (not at the fixture or `BoozerSurfaceJAX.__init__`).
It is still a form of grid substitution, but unlike the previous
silent rescue it is:

- **Explicit**: detected by name (`Surface.get_phi_quadpoints` with
  `RANGE_HALF_PERIOD`), not by catching a `ValueError`.
- **Narrowly scoped**: matches exactly one documented family; any
  other shifted/non-canonical grid still raises.
- **Not a `try/except` fallback**: no exception is swallowed.

The reviewer's framing (`accept and document, or move the fix
elsewhere`) was offered as the valid options after the first delete.
The team chose accept-and-document.

### Why option 1 over options 2 / 3

| Option | Cost | Risk |
|---|---|---|
| 1. In-bundle canonicalization | ~10 lines in one function | None — `BoozerSurfaceJAX` behavior unchanged |
| 2. SSOT fix at `BoozerSurfaceJAX.__init__` | Constructor change | Could affect solve numerics across all consumers |
| 3. Extend `surface_stellsym_mask_for_grid` for the shifted family | Mask-builder derivation + tests | Most invasive; requires verifying the reflected mask under `phi = 0.25/nfp` |

Option 1 keeps the change localized to the one consumer that needs
it (`make_traceable_objective_runtime_bundle`). The `BoozerSurfaceJAX`
solve and other consumers see no change. Trade-off acknowledged:
the function name is now mildly misleading ("canonicalize" is now
literal, not aspirational).

### Direct callers of `build_real_single_stage_init_fixture` (verified by `grep -rln`)

```
benchmarks/adjoint_fd_validation.py
benchmarks/grouped_adjoint_memory_probe.py
benchmarks/single_stage_cpp_jax_state_parity.py
benchmarks/single_stage_smoke_fixture.py        # definition
benchmarks/traceable_target_lane_compile_shape.py
tests/integration/test_single_stage_jax_cpu_reference.py   # 8+ call sites
tests/integration/test_single_stage_physics_parity.py
tests/subprocess/jax_runtime_cases.py
tests/subprocess/section6_fixture_probe.py
```

Direct callers are concentrated in `benchmarks/*` and
`tests/{integration,subprocess}/*`. The previously-listed
`tests/geo/test_single_stage_*`, `tests/test_jax_import_smoke.py`,
and `tests/test_benchmark_helpers.py` reference the single-stage
machinery only **indirectly** (via subprocess runners or other
helpers), so they were removed from the "direct caller" list.
Either way the shifted VMEC grid is broadly reused, which is why
the chosen fix sits in `_canonicalize_traceable_exact_quadrature`
rather than at each call site.

### Test status

- `tests/test_jax_import_smoke.py::test_transfer_guard_disallow_enforces_single_stage_target_runtime_boundaries`
  passes (`1 passed in 112.17s`) — confirms the in-bundle
  canonicalization resolves the round-2 regression.
- `test_grouped_biot_savart_coil_collective_parity_and_lowering`
  passes in isolation (`1 passed in 49.28s`); under the full smoke
  suite it can cross the 60 s pytest timeout. Pre-existing flake;
  not a regression from this work.

### Open follow-ups

- If a future caller supplies a different shifted family that the
  in-bundle detector does not recognize, the loud `ValueError` from
  `_compute_stellsym_mask_indices_for_grid` will surface it.
  Reviewer recommendation: extend the in-bundle detector with the
  new family name (still explicit), or migrate to option 3 then.
- Consider renaming `_canonicalize_traceable_exact_quadrature` to
  reflect that it actually canonicalizes (currently the docstring
  is the only place "canonicalize" is qualified). Cosmetic; defer
  until the wider naming sweep.

### Sites changed (applied — round 3)

File:
- `src/simsopt/geo/surfaceobjectives_jax.py`

- [x] **`_canonicalize_traceable_exact_quadrature`** — the original
  `try/except ValueError` block and the heuristic grid rebuild have
  been removed. In their place is a single explicit detection step:
  if `booz_jax.stellsym` and `quadpoints_phi` matches
  `Surface.get_phi_quadpoints(nphi, RANGE_HALF_PERIOD, nfp)` under
  `np.allclose`, both `quadpoints_phi` and `quadpoints_theta` are
  substituted with the unshifted canonical
  `half_phi` / `full_theta` family. Otherwise the inputs flow through
  unchanged. The mask is then built by
  `_compute_stellsym_mask_indices_for_grid`, which raises
  `ValueError` for any non-canonical grid that did not match the
  named family.
- [x] Docstring rewritten to describe the in-bundle canonicalization
  contract — no longer says "normalize at the surface constructor."
- [ ] Add a regression test in
  `tests/geo/test_surface_quadrature_grid_rejection.py` covering:
  canonical grid passthrough (assert no substitution); shifted VMEC
  half-period grid is detected and canonicalized to
  `half_phi`/`full_theta`; an unknown non-canonical grid raises
  `ValueError` from `_compute_stellsym_mask_indices_for_grid`.

### Round history (audit trail)

- **Round 1 (2026-05-13).** Tried to delete the silent `try/except
  ValueError` rescue. Edit did not persist to the working tree.
- **Round 2 (2026-05-13).** Re-applied the delete and verified by
  `grep`. Smoke suite then surfaced
  `test_transfer_guard_disallow_enforces_single_stage_target_runtime_boundaries`
  failing on a shifted VMEC half-period grid from
  `build_real_single_stage_init_fixture` — confirmed broad
  fixture reuse via `rg -l`.
- **Round 3 (2026-05-13).** Option 1 chosen (in-bundle
  canonicalization) and applied as described above. The previously-
  failing test passes (`1 passed in 112.17s`).

---

## §8. Keep boundary parsing, label it as such

File:
- `src/simsopt/backend/runtime.py`

- [ ] **Keep** `runtime.py:355-357` `_split_xla_flag_tokens` — `shlex` boundary.
- [ ] **Keep** `runtime.py:765-767` `_parse_visible_cuda_device_index` — env boundary.
- [ ] **Keep** `runtime.py:822-824` `_parse_nvidia_smi_indexed_value_row` — output parser.
- [ ] **Keep** `runtime.py:839-846` `_query_gpu_metric_mb_from_nvidia_smi` — subprocess boundary.
- [ ] Add inline comments labelling each as "external-input parse
  contract" (not a fallback). No new `try/catch` introduced anywhere
  else in this file.

---

## §9. Sites kept unchanged (re-raise / boundary / strict-fallback)

Verified clean and aligned with §1 contract. No edits required.

- [x] `boozersurface_jax.py:676-682` — `TypeError → TypeError` with
  explicit VJP-signature message.
- [x] `surfaceobjectives_jax.py:217-222` — `KeyError → ValueError` with
  explicit unknown-term message.
- [x] `optimizer_jax.py:490-498` — `ModuleNotFoundError` re-raise for
  non-private package failures.
- [x] `jax_core/tracing.py:1915-1924` — `getattr(..., None)` followed
  by explicit `raise TypeError(...)`; acceptable dispatch.
- [x] All strict-fallback API call sites
  (`boozersurface_jax.py:4908-4921`,
  `optimizer_jax.py:251-260, 3327`,
  `optimizer_jax_reference.py:404-408, 491-495`,
  `surfaceobjectives_jax.py:3465-3472`).

---

## Validation gates

Run in this order. Expect pre-existing upstream warnings; only zero
regression on touched files matters.

**Env note.** This checkout uses a prefix-based conda env at
`./.conda/jax-0.9.2`, not a named `jax-0.9.2`. The commands below use
`conda run -p .conda/jax-0.9.2 python`; equivalent forms are
`.conda/jax-0.9.2/bin/python -m pytest ...` or activating the env
with `conda activate ./.conda/jax-0.9.2`. The CLAUDE.md examples that
say `conda run -n jax-0.9.2` assume a named env elsewhere on the host
and will not resolve in this worktree.

- [ ] `ruff check <changed-files>`
- [ ] `ruff format <changed-files>`
- [ ] `conda run -p .conda/jax-0.9.2 python -m pytest -q tests/test_backend.py`
- [ ] `conda run -p .conda/jax-0.9.2 python -m pytest -q tests/test_jax_import_smoke.py`
- [ ] `conda run -p .conda/jax-0.9.2 python -m pytest -q tests/geo/test_boozersurface_jax.py -m "not private_optimizer_runtime"`
- [ ] `conda run -p .conda/jax-0.9.2 python -m pytest -q tests/geo/test_surface_objectives_jax.py`
- [ ] `conda run -p .conda/jax-0.9.2 python -m pytest -q tests/geo/test_optimizer_jax_item19.py` (closest extant optimizer test; add a new `tests/geo/test_optimizer_jax_silent_fallback_removal.py` for the new coverage)
- [ ] New tests: `tests/test_backend_strict_jax_device_detection.py` (§§2, 3)
- [ ] New tests: `tests/geo/test_optimizer_jax_silent_fallback_removal.py` (§§4, 4.1, 4.2, 5)
- [ ] New tests: `tests/jax_core/test_tree_signature.py` (§6)
- [ ] New tests: `tests/geo/test_surface_quadrature_grid_rejection.py` (§7)
- [ ] M2+M5 integration: `/Users/suhjungdae/code/hbt-compare/envs/candidate-fixed/bin/python -m pytest -q tests/integration/`
- [ ] Private optimizer lane: `conda run -p .conda/jax-0.9.2 python -m pytest tests/geo/test_boozersurface_jax.py tests/integration/test_single_stage_jax.py -m "private_optimizer_runtime"`
- [ ] `git diff --check`

---

## Success criteria

- [x] Zero broad `except Exception:` in scoped JAX paths. Boundary
  parsers in §8 use narrow exception classes (`ValueError`,
  `FileNotFoundError`, `subprocess.CalledProcessError`,
  `ImportError`) — not `Exception`. `inspect_array_sharding_summary`
  is dropped entirely (pre-check makes the catch unreachable).
- [x] Zero silent neutral-return for JAX runtime, optimizer cache,
  pytree signature, or sharding device-enumeration failures.
- [~] Quadrature-grid path: there is **no** silent
  `try/except`-driven rescue. There is one explicit canonicalization
  step in `_canonicalize_traceable_exact_quadrature` that maps a
  single named family (`Surface.get_phi_quadpoints(RANGE_HALF_PERIOD)`)
  to its unshifted equivalent. Any other unrecognized grid raises
  `ValueError` from `_compute_stellsym_mask_indices_for_grid`. Not
  a silent fallback, but not a pure adapter either — see §7
  "Resolution chosen" for the trade-off.
- [ ] Non-JAX backends never probe JAX device APIs (applied in code;
  regression test pending in `tests/test_backend_strict_jax_device_detection.py`).
- [x] All non-boundary fallbacks either route through
  `raise_if_strict_jax_fallback()` / `warn_if_jax_fallback()` or were
  deleted as dead code (§4.2). No raw
  `warnings.warn(..., RuntimeWarning, ...)` for fallback events in
  the scoped paths.
- [x] Pytree signature helpers: dead `except TypeError` branches
  removed inline at each of the 5 sites. Two signature contracts
  (value-hash, structural-meta) and two transforms
  (hostify, deviceify) keep their distinct leaf functions — no
  shared helper module (YAGNI).
- [x] Optimizer cache marking uses direct `setattr` without
  `try/except`; identity is preserved (`marked is fun`); all four
  cache-write call sites cleaned. Production callers all pass
  Python functions; future builtins/`__slots__` raise loudly.
- [x] `_field_dof_layout_version` is a required-attribute fetch via
  `_strict_field_dof_layout_version`.
- [x] `_coerce_dense_hess_inv` cannot silently degrade to identity —
  deleted along with its sole caller.

---

## Open questions for the implementer

- [x] **§7 fixture canonicalization** — resolved 2026-05-13 round 3:
  option 1 (in-bundle canonicalization inside
  `_canonicalize_traceable_exact_quadrature`) chosen. The
  previously-failing test passes. See §7 "Resolution chosen" for
  the rationale and the rejected options 2 / 3.

- [x] **§4 wrapper test conflict** — resolved 2026-05-13 round 2:
  wrapper class reverted; direct `setattr` without `try/except`.
  See §4 "Test identity conflict" for rationale.
- [ ] **§2/§3 regression tests** — write
  `tests/test_backend_strict_jax_device_detection.py` covering: (i)
  non-JAX backend never invokes `jax.devices` / `jax.local_devices`
  (assert via monkeypatch call counter); (ii) JAX-mode + monkeypatch
  raise → caller observes `RuntimeError`; (iii) JAX-mode + GPU
  backend unavailable → `_detect_imported_jax_cuda_device_index`
  returns `None` (narrow `RuntimeError` catch). Also assert
  `inspect_array_sharding_summary` returns the base summary for
  non-jax inputs (pre-check path) and propagates JAX errors for
  jax.Array inputs.
- [ ] **§4.2 follow-up** — if a future hybrid-BFGS continuation is
  re-introduced, route through `raise_if_strict_jax_fallback()` /
  `warn_if_jax_fallback()` with distinct
  `component="hybrid_bfgs.hess_inv.<branch>"` namespaces. See the
  collapsed "Historical strict-fallback routing" block in §4.2.
- [ ] **§2 callers audit** — confirmed: only
  `_build_sharding_tuning` calls
  `_detect_local_jax_device_count`/`_detect_global_jax_device_count`
  from production code (the other 14 hits are test monkeypatches).
  Early-exit is sufficient.
