# JAX Field Adapter Audit — `src/simsopt/field/*_jax*.py`

Reviewer: Opus 4.7 (1M context)
Branch: `gpu-purity-stage2-20260405`
JAX/jaxlib: 0.10.0 / 0.10.0 (Python 3.11, NumPy 2.x)
Date: 2026-05-16

## Executive Summary

The JAX field adapter layer falls into three architectural buckets:

1. **Cache-driven analytic ports** (`toroidal_field_jax.py`, `poloidal_field_jax.py`, `mirror_model_jax.py`, `reiman_jax.py`, `dommaschk_jax.py`, `circular_coil_jax.py`, `scalar_potential_rz_jax.py`, `dipole_field_jax.py`, `interpolated_field_jax.py`, `wireframefield_jax.py`) — subclass `MagneticField` and use the C++ field cache. Each `_B_impl` / `_dB_by_dX_impl` rebuilds the host-resident output buffer from a JAX kernel call. These are tight, generally faithful to the CPU oracle, and bit-identity tested in their respective test files. Issues are mostly DRY (multiple local `_points_device` helpers), performance smells (per-call allocations), and minor parity-attribute leaks.

2. **Free-standing Optimizable adapters** (`biotsavart_jax_backend.py` — `BiotSavartJAX`, `SpecBackedBiotSavartJAX`, `SingleStageRuntimeSpecBiotSavartJAX`) — do **not** inherit from `MagneticField`/`sopp.BiotSavart`, instead exposing a parallel API that takes JAX arrays end-to-end. The class is the most complex in the JAX field tree (2037 LOC) and the most rigorously instrumented (coil DOF state token, native pullback type, fast path for uniform `CurveXYZFourier`). Findings are largely about (a) the introspected fast path being unused on the hot path (`coil_set_spec()` always goes through the generic immutable-spec rebuild), (b) `SpecBackedBiotSavartJAX.x.setter` writing into `_dofs.full_x` with what is implicitly assumed to be a free-DOF vector — a latent shape-mismatch bug for any spec that ever has fixed DOFs, and (c) `_points_jax = None` initialization with no informative error if `B()` is called before `set_points`.

3. **Frozen-state Boozer adapters** (`boozermagneticfield_jax.py` — `BoozerRadialInterpolantJAX`, `BoozerAnalyticJAX`, `InterpolatedBoozerFieldJAX`) — `Optimizable` parallel classes that capture the CPU `BoozerMagneticField` state at construction time. Their public surface mirrors the CPU API but they do **not** subclass `sopp.BoozerMagneticField` (so the C++ cache and `recompute_bell` are reimplemented as a Python `dict` cache). Issues: `as_dict` does not delegate through `Optimizable.as_dict` for one of the three classes; `BoozerAnalyticJAX`/`InterpolatedBoozerFieldJAX` lack `_simsopt_jax_native_field` but that marker is only consumed by `MagneticField.__add__` composition (the Boozer classes do not have that operator), so the omission is currently inert; `InterpolatedBoozerFieldJAX._ensure_spec` mutates a `frozen=True` dataclass's `specs` dict in place — documented and intentional, but is a maintenance liability.

The audit found **no BLOCKER-class correctness issues**, but a non-trivial number of HIGH issues centered around contract subtleties (composition with `BiotSavartJAX`/Boozer wrappers, `SpecBackedBiotSavartJAX` shape assumption) and several MEDIUM issues (DRY, performance, missing markers). The aliases in the out-of-scope `force.py` (`B2EnergyJAX = B2Energy`, `LpCurveForceJAX = LpCurveForce`) confirm the CLAUDE.md flag about alias-only ports — none of those identity aliases live in the field-layer `*_jax*.py` modules under review here, but the issue is relevant when characterizing the JAX field surface as a whole.

---

## Per-module findings

### `_jax_common.py`

#### NIT-1. Unused helper used by only 6 of 9 candidate modules
- **File:line**: `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/field/_jax_common.py:11-21`
- **Category**: DRY / Code organization
- **What's wrong**: `points_device()` is meant to be the SSOT for host-points→device staging. It is imported by 6 modules (`circular_coil_jax`, `dommaschk_jax`, `mirror_model_jax`, `poloidal_field_jax`, `reiman_jax`, `toroidal_field_jax`). Three other modules — `dipole_field_jax.py:56`, `interpolated_field_jax.py:55`, `scalar_potential_rz_jax.py:33` — define their own `_points_device` instead of importing the shared helper.
- **Why it matters**: Three trivial duplicates of the same `_as_jax_float64(points)` body. Future changes to the device-staging policy (e.g. tightening transfer_guard rules) must be applied in 4 places instead of 1.
- **Suggested fix**: Replace the local `_points_device` definitions in `dipole_field_jax.py`, `interpolated_field_jax.py`, and `scalar_potential_rz_jax.py` with `from ._jax_common import points_device as _points_device`. (The scalar_potential version is a `self._points_device` method that already wraps `np.asarray(self.get_points_cart_ref(), ...)`; that has a different signature and is acceptable.)

---

### `biotsavart_jax.py` (re-export shim)

#### POSITIVE
- Pure re-export shim (55 lines). Maintains the historical import path for the kernel functions while delegating to `simsopt.jax_core.biotsavart`. Clean.

#### NIT-2. `_ensure_src_root_on_path` mutates `sys.path`
- **File:line**: `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/field/biotsavart_jax.py:11-17`
- **Category**: Code smell
- **What's wrong**: Module-level `sys.path.insert(0, src_root)` runs at every import. The function `_ensure_src_root_on_path` and the trailing call appear to support a legacy direct-load test that loaded the module by path. With the normal package import flow, this is unnecessary and pollutes `sys.path`.
- **Why it matters**: Adding the `src/` root to `sys.path` means any top-level `simsopt*` directory under `src/` is importable as a top-level package, which can mask shadowing bugs (e.g. `simsoptpp` vs `src/simsoptpp/` namespace pkg).
- **Suggested fix**: Remove `_ensure_src_root_on_path()`/`sys.path.insert(...)`. Confirm with the test suite (`tests/field/test_biotsavart_jax.py`) that direct-load is not relied upon (the importlib direct-load pattern documented in CLAUDE.md targets a different problem and does not require this `sys.path` injection).

---

### `biotsavart_jax_backend.py`

The main coil-tree adapter and the most consequential file in the JAX field layer.

#### HIGH-1. `SpecBackedBiotSavartJAX.x.setter` writes a full vector into `_dofs.full_x` regardless of fixed DOFs
- **File:line**: `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/field/biotsavart_jax_backend.py:496-523`
- **Category**: Optimizable contract / latent bug
- **What's wrong**: `_set_coil_dofs(coil_dofs)` stores the input into `self._x` as-is. Then `x.setter` does `self._dofs.full_x = host_array(self._x, dtype=np.float64)`, and `local_x.setter` does `self._dofs.free_x = host_array(self._x, dtype=np.float64)`. The Optimizable setter contract distinguishes "full" (all DOFs incl. fixed) from "free" (only un-fixed). If `BiotSavartSpec.coil_dofs` ever represents the FREE subset and any DOFs are fixed, then `_x.shape != full_x.shape`, and writing `_dofs.full_x = _x` produces a shape mismatch.
- **Why it matters**: Today the spec-backed path always passes the free-DOF vector (which equals full when no DOFs are fixed), so the bug is latent. The moment any caller builds a `BiotSavartSpec` with a partially-fixed DOF graph, this setter blows up — and the call site that exercises it (a single-stage seed reconstruction) only iterates the free vector. The defensive guard `if hasattr(self, "_dofs")` at line 504 confirms the author was aware that the local-DOF table needs to stay coherent, but the path through `x.setter` doesn't differentiate free from full.
- **Suggested fix**: Decide and document whether `BiotSavartSpec.coil_dofs` is the free vector or the full vector, then route `x.setter` through the matching `_dofs.free_x` / `_dofs.full_x` accessor. Cross-link the contract to `make_biot_savart_spec` and `make_coil_set_dof_extraction_spec`. Add an assertion `assert coil_dofs.shape[0] == self.dof_size` at the top of `_set_coil_dofs` so the failure mode surfaces at the call site.

#### HIGH-2. `BiotSavartJAX._uses_uniform_curve_xyz_fourier_fastpath` is gated only on the explicit-DOF helper, not on `coil_set_spec()`
- **File:line**: `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/field/biotsavart_jax_backend.py:1092-1152` (introspection), `:1333-1397` (`_coil_arrays_in_order_from_dofs`), `:1559-1569` (`coil_set_spec`)
- **Category**: Performance / Dead optimization
- **What's wrong**: `_introspect_coils` walks the coil tree to detect uniform `CurveXYZFourier`+`Current` graphs and populates `_unique_base_curves`/`_unique_base_currents`/`_coil_descs`/`_curve_quadpoints_jax`. This is then consumed **only** by `_coil_arrays_in_order_from_dofs` (line 1333), which is itself only used by `grouped_coil_arrays_from_dofs(coil_dofs)` (line 1399) — the explicit-DOF path for traceable wrappers. The actual hot path `B()` / `dB_by_dX()` / `B_vjp()` goes through `coil_set_spec() → _coil_set_spec_from_explicit_state → _coil_set_spec_from_dofs_immutable_specs(self.x) → coil_specs_from_dofs(self.x) → coil_specs_from_dof_extraction_spec(...)`, which uses `curve_spec_with_dofs(...)` and goes through the generic immutable-spec rebuild — bypassing the fast path entirely.
- **Why it matters**: The fast path was advertised in the class docstring at lines 1004-1011 as a major performance lever ("When all coils use `CurveXYZFourier` ... the JAX-native path is enabled"). In practice it is never engaged on the steady-state forward path. The `_introspect_coils` cost is paid at construction, but the documented payoff (avoiding CPU geometry round-trips) only materializes for callers that hand explicit DOF vectors via `grouped_coil_arrays_from_dofs` — i.e. traceable single-stage runtime entrypoints.
- **Suggested fix**: Either (a) wire the fast path into `coil_set_spec()` so steady-state `B()` / `dB_by_dX()` use `jaxfouriercurve_pure` directly, or (b) remove the fast-path code (introspection, `_unique_base_*`, `_coil_descs`, `_curve_quadpoints_jax`) and consolidate on the immutable-spec rebuild. The docstring should reflect whichever choice is made.

#### MEDIUM-3. `_per_coil_unit_field` quietly bypasses coil-axis sharding
- **File:line**: `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/field/biotsavart_jax_backend.py:122-150`
- **Category**: GPU sharding / Performance
- **What's wrong**: `dB_by_dcoilcurrents()` and siblings call `_per_coil_unit_field`, which Python-loops over every coil in every group and calls the per-coil kernel separately. The docstring at lines 130-135 admits "coil-axis collective reduction does not apply." On a CUDA target with `SIMSOPT_JAX_SHARDING=coil_groups`, this short-circuits the collective path. For a coilset with N=100 coils, this is N kernel launches and N traced graphs.
- **Why it matters**: `dB_by_dcoilcurrents` is hit by every Stage 2 derivative path that wants per-coil contributions (squared-flux current-sensitivity, taps for force / regularizers). On GPU, sequential kernel launches kill throughput.
- **Suggested fix**: Either (a) move the per-coil computation into a `jax.vmap` over a stacked coil-batch with `currents=ones`, or (b) document that consumers should prefer the grouped pullback APIs and treat `dB_by_dcoilcurrents` as a debug-only diagnostic. The current "JAX kernel cache for compile-time reuse within a quadrature group" relies on identical traces, which the per-call slicing `group.gammas[position:position+1]` should preserve, but the executor still launches N times.

#### MEDIUM-4. `BiotSavartJAX` does not implement `set_points_cart` / `set_points_cyl` / `B_cyl` / `GradAbsB_cyl`
- **File:line**: `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/field/biotsavart_jax_backend.py:984` (class), `:1413-1437` (`set_points`)
- **Category**: API parity / Composition
- **What's wrong**: `BiotSavartJAX` only exposes `set_points`, not `set_points_cart` / `set_points_cyl`. The CPU `BiotSavart` exposes both because it inherits from `MagneticField`. Consequence: code paths that call `field.set_points_cart(xyz)` (e.g. `MagneticFieldSum._set_points_cb` at `magneticfield.py:288`, `enclosed_current` at `wireframefield.py:153`) will `AttributeError` if handed a `BiotSavartJAX`. Similarly, `BiotSavartJAX` lacks `B_cyl()` and `GradAbsB_cyl()`, so it cannot be nested as the `field=` argument to `InterpolatedFieldJAX` (whose sampler at `interpolated_field_jax.py:122-124` calls `source_field.B_cyl()` / `source_field.GradAbsB_cyl()`).
- **Why it matters**: Real consequence — `InterpolatedFieldJAX(BiotSavartJAX(coils), …)` fails. Workaround is to wrap with the CPU `BiotSavart` instead, which defeats the GPU purity goal at the boundary. CLAUDE.md does flag this as the "BiotSavartJAX is parallel, not a MagneticField subclass" design, but the composition gap should be explicit in the docstring and the `_simsopt_jax_native_field` discovery logic should be aware of it.
- **Suggested fix**: Either (a) add `set_points_cart` / `set_points_cyl` thin aliases to `set_points`, plus `B_cyl()` / `GradAbsB_cyl()` host-side rotations of `B()` for the InterpolatedFieldJAX sampler boundary, or (b) document the composition incompatibility explicitly in the class docstring and add a `TypeError` with a directive message when `InterpolatedFieldJAX` is handed a `BiotSavartJAX`.

#### MEDIUM-5. `_points_jax = None` initialization with no informative error
- **File:line**: `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/field/biotsavart_jax_backend.py:1019` (init), `:1581` (`B()`)
- **Category**: UX / Error reporting
- **What's wrong**: `__init__` sets `self._points_jax = None`. The `B()` method passes it directly to `grouped_biot_savart_B_from_spec(self._points_jax, self.coil_set_spec())`. If `B()` is called before `set_points`, the kernel call receives `None` and the failure is a deep-stack TypeError inside the kernel rather than a clear "you must call set_points first" message.
- **Why it matters**: The CPU `BiotSavart` raises an informative error in C++ when `B()` is called pre-`set_points`. The JAX wrapper should preserve this UX.
- **Suggested fix**: Add a guard at the top of `coil_set_spec()` (or each public field accessor):
  ```python
  if self._points_jax is None:
      raise RuntimeError("BiotSavartJAX.set_points(...) must be called before B/dB/A.")
  ```

#### MEDIUM-6. `SpecBackedBiotSavartJAX.save()` always raises
- **File:line**: `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/field/biotsavart_jax_backend.py:720-721`
- **Category**: Serialization / Optimizable contract
- **What's wrong**: `save(self, _path)` unconditionally raises `RuntimeError("JAX runtime seed specs split runtime from host export.")`. This means a `SingleStageRuntimeSpecBiotSavartJAX` cannot participate in any pipeline that round-trips through SIMSON serialization.
- **Why it matters**: If a single-stage runtime spec is part of an Optimizable graph being serialized for restart, the dump phase will crash on this class. Either the spec-backed path is meant to never be serialized (in which case it should be excluded from `as_dict`/`from_dict` discovery), or it should round-trip through the same `BiotSavartSpec` host format used at construction.
- **Suggested fix**: Either document this as an intentional non-serializable runtime adapter (and add a typecheck in the parent serializer to redirect to the corresponding `BiotSavart` host export), or implement `as_dict` / `from_dict` over the underlying `BiotSavartSpec`.

#### LOW-7. `_block_until_ready` recurses through Derivative / dict / list / tuple but not through `BiotSavartFieldPullback`
- **File:line**: `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/field/biotsavart_jax_backend.py:102-119`
- **Category**: Timing / Correctness
- **What's wrong**: The helper iterates `Derivative.data`, dicts, lists, tuples, and pytree leaves. `BiotSavartFieldPullback` is a pytree-registered dataclass containing tuples of `(jax.Array, jax.Array, jax.Array)`. The `jax.tree_util.tree_leaves` fallback at the bottom (lines 117-119) should handle it. However, the fact that pytree leaves are only reached if the value didn't match earlier branches means perf-critical paths spend an `isinstance(...)` check series before reaching the right branch. Minor.
- **Suggested fix**: Just always use `jax.tree_util.tree_leaves` and block on each leaf. Drop the explicit per-type branches. The complexity reduction is worth more than the micro-optimization.

#### LOW-8. `_unwrap_coil_curve_and_current_objects` import is module-side-effect coupled
- **File:line**: `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/field/biotsavart_jax_backend.py:75`
- **Category**: Coupling / Import structure
- **What's wrong**: `from ._coil_graph import _unwrap_coil_curve_and_current_objects` — the helper lives in the (CPU-only) `_coil_graph.py` module. This couples the JAX backend to a CPU-side module that may import simsoptpp transitively.
- **Why it matters**: Slight import-time risk for pure-JAX runtimes where simsoptpp is intentionally unavailable. Worth a sanity check that `_coil_graph.py` does not pull simsoptpp at import time.
- **Suggested fix**: Verify `_coil_graph` is `simsoptpp`-free (a one-time import audit). If it isn't, factor out the unwrap helper to a smaller pure-Python utility.

#### POSITIVE
- The coil-state-token contract (`_coil_dof_state_token` advanced both on direct `x` writes via `_set_global_coil_dofs` and on ancestor invalidation via `set_recompute_flag`) is implemented precisely as CLAUDE.md specifies — line 1048-1059. `_suppress_dependency_coil_dof_state` correctly debounces the dual-call path during a global set.
- `BiotSavartFieldPullback` pytree registration (lines 243-263) cleanly separates the data tuples (children) from the coil-index lists (static metadata). Round-trip-safe under `jax.tree_util.tree_flatten` / `tree_unflatten`.
- `B_pullback_native` / `A_pullback_native` / `dA_by_dX_pullback_native` / `dB_by_dX_pullback_native` form a uniform pullback boundary distinct from the public `Derivative`-projecting `*_vjp` methods. Good separation of native pytree boundary from the SIMSOPT `Derivative` graph.
- `Optimizable.update_free_dof_size_indices()` override at line 1041-1046 correctly clears the cached free-position dict and rebuilds the DOF extraction spec when the free-DOF layout changes — this is the right hook for in-place fix/unfix of DOFs.

#### Verdict
**HIGH** — functional in production, but with two non-trivial latent issues (`SpecBackedBiotSavartJAX` shape assumption, fast-path being unused on hot path) and a real composition gap with `InterpolatedFieldJAX`.

---

### `boozermagneticfield_jax.py`

Three classes — `BoozerRadialInterpolantJAX`, `BoozerAnalyticJAX`, `InterpolatedBoozerFieldJAX` — all `Optimizable` parallel ports of the CPU `BoozerMagneticField` subclasses.

#### HIGH-9. `BoozerRadialInterpolantJAX.as_dict` does NOT call `super().as_dict(...)`
- **File:line**: `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/field/boozermagneticfield_jax.py:913-924`
- **Category**: Serialization / Optimizable contract
- **What's wrong**: The method manually constructs the `@module`/`@class`/`@name`/`@version` metadata block instead of delegating to `Optimizable.as_dict(serial_objs_dict)`. Other JAX wrappers in the field tree (`dipole_field_jax.py:306`, `dommaschk_jax.py:138`, `wireframefield_jax.py` does not implement, `interpolated_field_jax.py:308`, `toroidal_field_jax.py:77`) all call `super().as_dict(serial_objs_dict=...)` and then append their own keys.
- **Why it matters**: `Optimizable.as_dict` (line 1633 in `_core/optimizable.py`) writes a `"dofs"` block from `self._dofs.as_dict2(...)` when `len(self.local_full_x)` is non-zero. The Boozer JAX wrapper has `x0=np.asarray([])` so the dofs block is empty either way; but if the SIMSON encoder adds any new bookkeeping (provenance, version stamps) at the `Optimizable.as_dict` level, this class silently drops it.
- **Suggested fix**: Replace the manual metadata block with `d = super().as_dict(serial_objs_dict)` and add the JAX-specific keys after.

#### MEDIUM-10. `InterpolatedBoozerFieldJAX._ensure_spec` mutates a `frozen=True` dataclass's dict field
- **File:line**: `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/field/boozermagneticfield_jax.py:1473-1476`
- **Category**: JAX best practices / Pytree invariants
- **What's wrong**: The `InterpolatedBoozerFieldFrozenState` dataclass is `frozen=True` but holds a mutable `dict` field `specs`. The wrapper relies on appending to that dict to lazy-build per-scalar interpolants on first access. The pattern is documented in `jax_core/interpolated_boozer_field.py:178-208` ("Mutability contract"), and the dataclass is explicitly **not** registered as a pytree. So there is no JIT/autodiff correctness bug, but the practice is fragile: anyone who later registers the dataclass as a pytree (or relies on dataclass-frozen semantics for caching) will silently break the lazy-build path.
- **Why it matters**: Defensive design: a `frozen=True` dataclass is meant to be immutable. Mutating a referenced dict is a Python-semantic loophole, not a design pattern. Future maintainers who don't read the contract docstring will be surprised.
- **Suggested fix**: Two options. (a) Drop `frozen=True` on `InterpolatedBoozerFieldFrozenState`, since the class isn't actually immutable. (b) Move the `specs` dict out of the dataclass and onto the wrapper (`InterpolatedBoozerFieldJAX._lazy_specs: dict[str, RegularGridInterpolant3DSpec]`), keeping the frozen state purely immutable. Either is cleaner than the current "frozen but mutable" pattern.

#### MEDIUM-11. `BoozerAnalyticJAX` and `InterpolatedBoozerFieldJAX` lack `as_dict`/`from_dict`
- **File:line**: `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/field/boozermagneticfield_jax.py:1065` (`BoozerAnalyticJAX`), `:1282` (`InterpolatedBoozerFieldJAX`)
- **Category**: Serialization / Optimizable contract
- **What's wrong**: Neither class implements `as_dict` / `from_dict`. The CPU siblings `BoozerAnalytic`/`InterpolatedBoozerField` also don't, so there is parity — but parity to a CPU class that itself can't round-trip is not a strong defense for the JAX port.
- **Why it matters**: Any pipeline that serializes a `SIMSON(opt_graph_with_boozer_analytic_jax)` and reloads it will only get the dataclass-frozen state if `from_dict` is supplied. Currently the JSON load path will work because the GSON decoder is lenient, but the lack of `from_dict` means the field is not exercising the documented `from_frozen_state` constructor on reload.
- **Suggested fix**: Implement `as_dict`/`from_dict` symmetric with `BoozerRadialInterpolantJAX:913-939`, using `_frozen_state_to_host`/`_frozen_state_from_host` analogues for `BoozerAnalyticFrozenState` and `InterpolatedBoozerFieldFrozenState`.

#### MEDIUM-12. `freeze_boozer_radial_state` and `_frozen_state_from_host` use `jnp.asarray` instead of `_as_jax_float64`
- **File:line**: `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/field/boozermagneticfield_jax.py:219-220`, `:241-242`, `:262-263`, `:283-284`, `:309-310`
- **Category**: GPU transfer-guard / GPU strict-cuda hardening
- **What's wrong**: CLAUDE.md notes the JAX 0.10.0 strict-cuda transfer-guard policy: implicit host-to-device transfers should go through `jax.device_put` (i.e. through `_as_jax_float64`). The freeze helpers use `jnp.asarray(np_array, dtype=jnp.float64)`, which under modern JAX is functionally `device_put` for NumPy inputs but in `transfer_guard("disallow")` is sometimes rejected. The `_as_jax_float64` helper in `jax_core/_math_utils.py` is the documented SSOT for explicit `device_put`.
- **Why it matters**: These are construction-time staging (not hot-path), so even if `transfer_guard` fires it would only fail at `freeze_boozer_radial_state` invocation, not on every modB call. Still, the pattern is inconsistent with the rest of the codebase, which uniformly uses `_as_jax_float64` at host→device boundaries.
- **Suggested fix**: Replace `jnp.asarray(np.asarray(x, dtype=np.float64), dtype=jnp.float64)` patterns with `_as_jax_float64(np.asarray(x, dtype=np.float64))`. Also drop the redundant double-cast: `_as_jax_float64` already casts to float64.

#### LOW-13. `BoozerAnalyticJAX` / `BoozerRadialInterpolantJAX` / `InterpolatedBoozerFieldJAX` are missing `_simsopt_jax_native_field` markers
- **File:line**: `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/field/boozermagneticfield_jax.py:819`, `:1065`, `:1318` (only InterpolatedBoozer has the marker)
- **Category**: Composition guard
- **What's wrong**: The `_simsopt_jax_native_field = True` marker is consumed by `_is_jax_native_field` in `magneticfield.py:14`. The marker only matters for `MagneticFieldSum`/`MagneticFieldMultiply` composition. Since Boozer wrappers are not `MagneticField` subclasses, they cannot participate in `__add__`/`__mul__` composition. So the marker is currently inert for these classes — but adding it would future-proof against any later strict-mode composition guard for Boozer fields.
- **Suggested fix**: Add `_simsopt_jax_native_field = True` to all three Boozer wrappers for consistency. Cheap and forward-compatible.

#### LOW-14. `BoozerRadialInterpolantJAX._nfp` extraction via deep attribute chain
- **File:line**: `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/field/boozermagneticfield_jax.py:845`
- **Category**: Fragile coupling
- **What's wrong**: `self._nfp = int(getattr(upstream.booz.bx, "nfp", 1))` digs through `upstream.booz.bx.nfp`. If the upstream `BoozerRadialInterpolant` ever restructures its internal `booz`/`bx` attributes (e.g. wraps `bx` in a property or renames `booz`), this silently falls back to `nfp=1`.
- **Why it matters**: The default of 1 is a wrong default for any non-axisymmetric Boozer profile. A failure mode of "I silently got the wrong nfp" is far worse than "I crashed at construction."
- **Suggested fix**: Replace the `getattr(..., 1)` default with `getattr(upstream, "nfp", None)`; if None, raise an error rather than silently defaulting.

#### NIT-15. `BoozerRadialInterpolantJAX.set_points` returns `self`; CPU returns `None`
- **File:line**: `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/field/boozermagneticfield_jax.py:902`
- **Category**: API parity (minor)
- **What's wrong**: Docstring claims "Returns `self` to match the upstream `set_points` signature." The CPU `sopp.BoozerMagneticField.set_points_*` in turn returns the C++ field object — verified by inspecting `pyboozermagneticfield.h`. So the parity claim is correct, but the comment is worth a sanity check against the actual C++ pybind11 return type.
- **Suggested fix**: Confirm with the C++ binding (or test the CPU class's return value at the Python boundary) and document precisely.

#### POSITIVE
- `BoozerRadialInterpolantFrozenState` is correctly registered as a JAX dataclass pytree (line 165-199) with proper `data_fields` / `meta_fields` split. The `stellsym` and `no_K` flags being meta-fields means JIT specialization on them happens correctly.
- All `set_points` methods validate the `(n, 3)` shape (line 896-899, 1166-1169, 1424-1427) — better than the CPU sibling.
- The cache invalidation pattern (`self._cache.clear()` on `set_points`, plus `clear_cached_properties()`) is symmetric with the CPU `invalidate_cache` semantics.
- `from_frozen_state` constructors (lines 849-864, 1117-1138, 1360-1388) cleanly separate the host-frozen-state path from the upstream-CPU path. The InterpolatedBoozer variant correctly tracks `_field = None` and uses it to refuse lazy-build requests on reloaded wrappers.
- The K-spline `psi0` factor is captured at freeze time with documentation that `upstream.psi0` mutations require rebuilding the frozen state — line 302-303 and 829-831. Correct contract.

#### Verdict
**MEDIUM** — Solid frozen-state pattern; the `as_dict` skip-super and the missing `as_dict`/`from_dict` on two of the three classes are real gaps. The frozen-dataclass-with-mutable-dict is a code smell that has a clean fix.

---

### `circular_coil_jax.py`

#### LOW-16. `Inorm` is private internal but `I` is a derived property
- **File:line**: `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/field/circular_coil_jax.py:39`, `:45-46`
- **Category**: Convention
- **What's wrong**: `self.Inorm = float(I) * 4e-7` stores `Inorm` (current normalized by `mu0/pi`); `I` property at line 45-46 returns `self.Inorm * 25e5`. The constant chain (`4e-7` and `25e5`) is the standard `mu0/pi` decomposition but is presented as bare floats with no comment. The CPU class does the same — but the magic numbers should be linked to `mu0 = 4*pi*1e-7`.
- **Suggested fix**: Add a comment: `# mu0 / pi == 4e-7 H/m`, `# 1 / (4e-7) ≈ 25e5`. Or factor out a module-level `_INORM_TO_AMPERES = 1.0 / (4e-7)` constant.

#### POSITIVE
- Uses `_jax_common.points_device`. Clean drop-in replacement. `jax_B_dB_at` provides the JAX-native single-point evaluator separate from the host-cache `_B_impl`/`_dB_by_dX_impl`. Matches the established pattern.

#### Verdict
**LOW** — solid, faithful to CPU, no correctness issues.

---

### `dipole_field_jax.py`

#### MEDIUM-17. `dipole_grid_arr.shape != dipole_vectors_arr.shape` check is too restrictive
- **File:line**: `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/field/dipole_field_jax.py:222-231`
- **Category**: API parity
- **What's wrong**: The validation requires `dipole_vectors.shape == dipole_grid.shape`. But CPU `DipoleField` allows `dipole_vectors.reshape(ndipoles, 3)` — implicitly accepting any layout that has `ndipoles * 3` elements (e.g. a `(ndipoles*3,)` flat vector or `(ndipoles, 3, 1)` shape). The CPU class is permissive; the JAX wrapper is stricter.
- **Why it matters**: Code that worked against the CPU class will fail against the JAX one. Cross-port substitutability is the explicit goal.
- **Suggested fix**: Mirror the CPU `dipole_vectors.reshape(ndipoles, 3)` pattern: accept any input that reshapes cleanly to `(ndipoles, 3)`, raise only if the total element count differs.

#### LOW-18. `self.dipole_vectors = m_vec` aliases an attribute that's not on the CPU class
- **File:line**: `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/field/dipole_field_jax.py:243-244`
- **Category**: API parity / Dead attribute
- **What's wrong**: After `_expand_symmetries`, the JAX class stores `self.m_vec = m_vec` AND `self.dipole_vectors = m_vec`. The CPU class only stores `self.m_vec` (the expanded manifold). `self.dipole_vectors` is the **input** to the CPU constructor but isn't kept as an instance attribute after expansion.
- **Why it matters**: A caller who reads `field.dipole_vectors` on a JAX field gets the post-expansion (full-symmetry) vectors, but on a CPU field would `AttributeError`. Worse: a downstream consumer that conditionally branches on `hasattr(field, "dipole_vectors")` would behave differently between CPU and JAX ports.
- **Suggested fix**: Either remove `self.dipole_vectors = m_vec`, or document that it is the **expanded** manifold and not the half-period input.

#### LOW-19. `_expand_symmetries` is a 90-line pure-NumPy copy of `_dipole_fields_from_symmetries`
- **File:line**: `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/field/dipole_field_jax.py:67-160`
- **Category**: DRY / Maintenance
- **What's wrong**: The expansion logic is duplicated from `magneticfieldclasses.py:_dipole_fields_from_symmetries` (lines 634-712). Two implementations of the same physics expand-symmetry rule. Any bug found in one needs to be ported to the other.
- **Why it matters**: As CLAUDE.md notes, "Confirmed NOT bugs: nfp factor in volume/area — correct because nfp cancels with quadrature step." Physics-arithmetic bugs in expand-symmetry would be subtle and only surface as silent parity failures. Single source of truth is the only safe pattern.
- **Suggested fix**: Either (a) hoist `_dipole_fields_from_symmetries` to a pure NumPy module-level function that both classes call, or (b) accept the duplication explicitly and add a comment in both pointing to the matching block in the other.

#### Verdict
**MEDIUM** — clean code, but the input-shape strictness and the duplicated expand-symmetry logic are real parity / maintenance liabilities.

---

### `dommaschk_jax.py`

#### MEDIUM-20. `_toroidal_baseline_B_dB` instantiates a fresh `ToroidalFieldJAX` on every `_B_impl` call
- **File:line**: `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/field/dommaschk_jax.py:28-41`, used at `:105`, `:135`
- **Category**: Performance
- **What's wrong**: Every call to `DommaschkJAX._B_impl(B)` and `._dB_by_dX_impl(dB)` constructs a new `ToroidalFieldJAX(R0=1.0, B0=1.0)`, calls `baseline.set_points_cart(points)`, and computes `baseline.B()` / `baseline.dB_by_dX()`. The construction also pulls a fresh `sopp.MagneticField` cache buffer (since `ToroidalFieldJAX` is a `MagneticField`).
- **Why it matters**: For Dommaschk fields evaluated repeatedly (e.g. tracing), this is non-trivial overhead. The class already has the `jax_B_at` / `jax_B_dB_at` methods that compute the baseline directly from `toroidal_B(ToroidalFieldSpec(R0=1.0, B0=1.0), points)` — that's the pure-JAX path. The host-cache path should reuse the same pattern.
- **Suggested fix**: Replace the call to `_toroidal_baseline_B_dB` with direct calls to `toroidal_B` / `toroidal_dB` on a module-level `_BASELINE_SPEC = ToroidalFieldSpec(R0=1.0, B0=1.0)`. Then `_B_impl(B)` and `_dB_by_dX_impl(dB)` use the spec directly, no allocation per call.

#### NIT-21. `coeffs` is stored as a Python list and re-cast on every `_current_spec()`
- **File:line**: `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/field/dommaschk_jax.py:78`, `:93-98`
- **Category**: Performance
- **What's wrong**: `self.coeffs = coeffs` (line 78) stores the user-supplied input as-is. `_current_spec()` then casts `np.asarray(self.coeffs, dtype=np.float64)` on every call. The CPU class also stores `self.coeffs` as input but doesn't have this re-cast cycle.
- **Suggested fix**: Cache the spec at construction (`self._spec = self._build_spec(...)`) and only rebuild if `self.coeffs` is mutated (which the design seems to allow but doesn't guard). If `coeffs` is immutable, store as a `np.ndarray` and reuse the cached spec.

#### POSITIVE
- Construction validates `mn` and `coeffs` shapes (lines 63-75). Better than the CPU sibling, which silently accepts mis-shaped inputs.

#### Verdict
**MEDIUM** — repeat `ToroidalFieldJAX` construction is a real performance miss but doesn't break correctness.

---

### `interpolated_field_jax.py`

#### HIGH-22. `_dB_by_dX_impl` is deliberately not implemented; consumers will hit C++ trampoline NotImplementedError
- **File:line**: `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/field/interpolated_field_jax.py:21-27`
- **Category**: API parity / Documentation
- **What's wrong**: The class docstring documents that `_dB_by_dX_impl` is intentionally absent because the CPU `InterpolatedField` doesn't implement it either (raises inside the C++ binding). The CPU class then exposes `_GradAbsB_impl` instead. The JAX wrapper shadows `GradAbsB()` at the Python level (line 241-264). This means: `InterpolatedFieldJAX(field, …).dB_by_dX()` will hit the C++ trampoline and raise. **This is intentional.** But the docstring is buried inside the class, not on the `dB_by_dX` method itself.
- **Why it matters**: A user who knows about `dB_by_dX()` from the base `MagneticField` API will hit a confusing error path. The CPU sibling has the same gap, but the JAX port should surface the limitation more loudly.
- **Suggested fix**: Override `dB_by_dX()` on `InterpolatedFieldJAX` with a clear `RuntimeError("InterpolatedFieldJAX does not expose dB_by_dX in Cartesian coordinates. Use the source field directly, or call GradAbsB() for the physical gradient table.")`.

#### MEDIUM-23. The interpolant evaluation kernel emits NaN on out-of-bounds queries when `extrapolate=False`
- **File:line**: `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/jax_core/regular_grid_interp.py:622-630`, consumed by `interpolated_field_jax.py:133-140`
- **Category**: JAX best practices / Gradient correctness
- **What's wrong**: When `out_of_bounds_ok=False`, the kernel routes out-of-bounds samples to `jnp.nan` instead of raising (the C++ binding raises). The host-side `_checked_host_result` then materializes and inspects for `np.isnan(result).any()` before raising. The pattern is documented at the kernel layer (lines 622-626). However, **`jax.grad`** through this kernel for a point that's in-bounds will compute a finite gradient via the `jnp.where(in_kept_cell, result, jnp.nan)` selection. JAX's `jnp.where` gradient is `p * df/dx + (1-p) * dg/dx` — if `dg/dx` (the gradient of `jnp.nan` w.r.t. `x`) is computed eagerly, it propagates `NaN * 0 = NaN`. This is a known JAX gradient gotcha.
- **Why it matters**: A consumer that uses `InterpolatedFieldJAX.jax_B_at(point)` with `extrapolate=False` and computes `jax.grad(jax_B_at)(point)` may get NaN gradients even for points well inside the domain. The pure-evaluation path is fine; only `jax.grad`/`jax.jacfwd`/`jax.jacrev` through the kernel hit this gotcha.
- **Suggested fix**: Wrap the `result` computation with `jnp.where(in_kept_cell, xlocal, jnp.zeros_like(xlocal))` (and similarly for `ylocal`, `zlocal`) **before** the basis-value computation, so the false branch has gradient zero by construction. Alternatively, document the gradient policy explicitly: "do not `jax.grad` through `jax_B_at` with `extrapolate=False` if any query may be out-of-bounds."

#### MEDIUM-24. `B_cyl` shadows the C++ trampoline and bypasses the `data_Bcyl` cache
- **File:line**: `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/field/interpolated_field_jax.py:285-306`
- **Category**: API parity
- **What's wrong**: `B_cyl()` is overridden at the Python level to compute `self.B()` (which goes through `_B_impl` and the C++ cache) and rotate host-side. This bypasses the C++ `data_Bcyl` cache field. The docstring acknowledges this (lines 287-295) but does not warn that consumers expecting the C++ cache (via `Bcyl_ref()`) will get stale or empty data.
- **Why it matters**: `MagneticField` exposes `B_cyl()` and (via the C++ binding) `Bcyl_ref()`. The C++ binding's `Bcyl_ref()` reads `data_Bcyl` — which is populated by the C++ `_B_cyl_impl`. The JAX wrapper's `B_cyl()` does NOT populate `data_Bcyl`. So `field.B_cyl()` returns the rotated value, but `field.Bcyl_ref()` (used internally by some C++ consumers) returns stale data.
- **Suggested fix**: Either (a) also fill `data_Bcyl` via the C++ binding, or (b) document the limitation clearly in the class docstring, including the `Bcyl_ref()` divergence.

#### POSITIVE
- Eager spec build at construction (line 187-212) keeps the JAX kernel cache hot and avoids any host round-trips at evaluation time except for the explicit `np.asarray(...)` materialization.
- The `_build_skip_callback` adapter (line 79-101) cleanly handles both `skip=None` and user-supplied skip predicates, and rebases between the rectangular-kernel arg names and the cylindrical mesh-node coordinates.
- `_checked_host_result` (line 133-140) gives the host post-hoc error detection consistent with the C++ binding's "raise on out-of-bounds".

#### Verdict
**MEDIUM** — The `dB_by_dX` parity gap and the `jax.grad` NaN gradient gotcha both deserve explicit documentation; the `B_cyl` cache bypass is real and should be either fixed or documented.

---

### `magneticfieldclasses_jax.py`

#### POSITIVE
- Pure back-compat shim (26 lines). Clean re-export of the per-class modules. No issues.

#### Verdict
**LOW** — no findings.

---

### `mirror_model_jax.py`

#### NIT-25. `Z_m` attribute uses CPU-style snake-case-ish capitalization
- **File:line**: `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/field/mirror_model_jax.py:30-33`
- **Category**: Convention
- **What's wrong**: The constructor parameter and attribute is `Z_m` matching the CPU class. This is fine but worth noting since the rest of the JAX module uses snake_case.
- **Suggested fix**: None — CPU parity wins. Leave as-is.

#### POSITIVE
- Tight, minimal drop-in port. Uses `_jax_common.points_device`. Three public scalar attributes, one frozen spec. Clean.

#### Verdict
**LOW** — no findings.

---

### `poloidal_field_jax.py`

#### POSITIVE
- Same pattern as `mirror_model_jax.py`. Minimal port, clean.

#### Verdict
**LOW** — no findings.

---

### `reiman_jax.py`

#### LOW-26. `self.k = list(k)` and `self.epsilonk = list(epsilonk)` lose array semantics
- **File:line**: `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/field/reiman_jax.py:42-43`
- **Category**: API parity
- **What's wrong**: The constructor stores `k` and `epsilonk` as Python lists, then `_build_spec` reads them back through `tuple(int(v) for v in self.k)` and `_as_jax_float64(self.epsilonk)`. CPU `Reiman` stores them as lists too (it doesn't normalize). So parity is preserved, but the JAX wrapper would benefit from storing them as immutable NumPy arrays at construction.
- **Suggested fix**: `self.k = np.asarray(k, dtype=np.int64)` and `self.epsilonk = np.asarray(epsilonk, dtype=np.float64)`. Update `_build_spec` accordingly. Same default behavior at the API boundary, but defensive against external mutation.

#### POSITIVE
- Construction validates `len(k) == len(epsilonk)` (lines 35-39). Good defensive shape check.

#### Verdict
**LOW** — clean port.

---

### `sampling_jax.py`

#### POSITIVE
- Pure re-export shim (13 lines). Clean.

#### Verdict
**LOW** — no findings.

---

### `scalar_potential_rz_jax.py`

#### LOW-27. `_points_device` is a method instead of using the shared `_jax_common` helper
- **File:line**: `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/field/scalar_potential_rz_jax.py:33-34`
- **Category**: DRY
- **What's wrong**: `_points_device` is a method on the class that returns `_as_jax_float64(np.asarray(self.get_points_cart_ref(), dtype=np.float64))`. This is structurally different from the other modules' module-level `_points_device(points)` helper because it pulls `self.get_points_cart_ref()` inline. Refactoring to use the shared helper would simplify the body of `_B_impl` / `_dB_by_dX_impl`.
- **Suggested fix**: Replace `self._points_device()` with `_points_device(np.asarray(self.get_points_cart_ref(), dtype=np.float64))` using the shared helper from `_jax_common`. Pull the conversion to the impl sites.

#### POSITIVE
- SymPy-to-JAX lowering happens once at construction via `scalar_potential_rz_kernels(self.phi_parsed)` (line 31). Runtime evaluation stays inside pure JAX kernels. Good adapter pattern.

#### Verdict
**LOW** — minor DRY issue.

---

### `toroidal_field_jax.py`

#### POSITIVE
- Implements `_B_impl`, `_dB_by_dX_impl`, `_d2B_by_dXdX_impl`, `_A_impl`, `_dA_by_dX_impl` — full coverage of the CPU surface. Clean uniform pattern.

#### Verdict
**LOW** — no findings.

---

### `wireframefield_jax.py`

#### MEDIUM-28. `_points_device` is not initialized in `__init__`; `_B_impl` will AttributeError before any `set_points_*`
- **File:line**: `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/field/wireframefield_jax.py:49-58`, `:83-93`
- **Category**: UX / Error reporting
- **What's wrong**: `WireframeFieldJAX.__init__` does NOT initialize `self._points_device`. It is first set by `set_points_cart` / `set_points_cyl`. If a caller invokes `B()` before either `set_points_*`, the C++ trampoline calls `_B_impl(B)` which then `AttributeError`s on `self._points_device`. The CPU `WireframeField` also requires `set_points` first, but raises a clearer C++ error.
- **Why it matters**: A caller pre-`set_points` gets an obscure `AttributeError: 'WireframeFieldJAX' object has no attribute '_points_device'` instead of a clear "you must call set_points first" message.
- **Suggested fix**: Initialize `self._points_device = None` in `__init__` and add a guard at the top of `_B_impl` / `_dB_by_dX_impl` / `dB_by_dsegmentcurrents` that raises `RuntimeError("WireframeFieldJAX.set_points_cart(...) must be called before B / dB / dB_by_dsegmentcurrents.")` if `self._points_device is None`.

#### LOW-29. Wireframe snapshot at construction not propagated when `wframe.currents` mutates
- **File:line**: `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/field/wireframefield_jax.py:49-58`
- **Category**: Stale-cache pattern
- **What's wrong**: `_snapshot_wireframe_arrays(wframe)` captures `wframe.currents` at construction. The `_currents_device` JAX array is computed once and never refreshed. The CPU sibling does the same (the C++ binding's `sopp.WireframeField.__init__` copies into internal state), so parity is preserved. But it is a footgun for callers who don't know the contract.
- **Why it matters**: A caller who mutates `wframe.currents = new_array` and expects `field.B()` to reflect the change will get the stale snapshot. Both CPU and JAX share this — but it's worth flagging.
- **Suggested fix**: Document the snapshot semantics on the class docstring (currently mentioned at line 41-45 but only in passing). Or expose a `refresh_currents()` method that rebuilds `_currents_device` from `self.wireframe.currents`.

#### POSITIVE
- `_clear_segment_current_cache` (line 60-62) and `set_points_cart`/`set_points_cyl`/`clear_cached_properties` calling it (lines 64-81) correctly invalidate the per-segment cache when points or DOFs change.
- `dB_by_dsegmentcurrents` returns a list of `n_segments` separate NumPy arrays matching the CPU sibling's interface (line 105-131). Parity preserved.
- `dBnormal_by_dsegmentcurrents_matrix` (line 133-158) mirrors the CPU implementation faithfully, including the `area_weighted=False` default and the `np.linalg.norm(normal, axis=2)` normalization.

#### Verdict
**LOW** — clean port, with the pre-`set_points` AttributeError as the only real UX issue.

---

## Cross-cutting positive notes

- **Tensor convention**: `dB_by_dX[p, j, l] = ∂_j B_l(x_p)` is preserved throughout. The kernel layer (`jax_core/biotsavart.py`, `jax_core/dipole_field.py`, `jax_core/analytic_pure_fields.py`) implements it correctly, and the adapter layer does not transpose accidentally.
- **`_simsopt_jax_native_field` discovery**: 11 of 14 class-bearing modules carry the marker. The 3 without it (`BiotSavartJAX`, `BoozerAnalyticJAX`, `BoozerRadialInterpolantJAX`) are all non-`MagneticField` Optimizable parallels, so the marker is inert. Consistent if not uniform.
- **Lazy import / try-except-ImportError**: `field/__init__.py` correctly guards `_JAX_FIELD_MODULES` and `_JAX_FIELD_SIMSOPTPP_MODULES` on `_has_jax` and `_has_simsoptpp`. Pure-JAX-no-simsoptpp installs and CPU-only installs both work.
- **JIT discipline**: No accidental `static_argnames` misuse spotted. The kernel layer's `@jax.jit` decorators (e.g. `_evaluate_batch_jit`, `_dipole_field_B_jit`, `_wireframe_segment_B_contributions_jit`) use `static_argnames` only on shape/policy flags, not on data.
- **No `jax.array(...)` in hot paths**: All host-to-device staging goes through `_as_jax_float64` (or `_as_jax_int64` in wireframe) which explicitly calls `jax.device_put`. Compatible with `transfer_guard("disallow")`.
- **Boundary `int()`/`bool()` casts**: Applied where needed in result dicts. The `BoozerRadialInterpolantJAX.from_dict` and related paths use `bool(...)` / `int(...)` on JAX scalars before storing.
- **CLAUDE.md flagged `_set_points_cb` divergence**: The JAX wrappers override `set_points_cart` / `set_points_cyl` (where needed) rather than `_set_points_cb`. For the wireframe and dipole adapters, this is the right hook because they have to refresh the device-resident points buffer. For the simpler analytic adapters that re-read `get_points_cart_ref()` each call, no override is needed — and they don't have one. Correct.

## Notes on aliases (out-of-scope but relevant)

The CLAUDE.md / OpenMemory note about "B2EnergyJAX / LpCurveForceJAX alias-identity caveat" applies to `src/simsopt/field/force.py`:
- `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/field/force.py:1320` — `B2EnergyJAX = B2Energy`
- `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/field/force.py:2284` — `LpCurveForceJAX = LpCurveForce`

These are identity re-exports, not real ports. None of the `*_jax*.py` field modules in scope here are alias-only — every one of them implements its own evaluator on top of the `jax_core` kernels. The `force.py` issue is real but lives outside the audit scope; flagging here for cross-reference.

## Summary by severity

| Severity | Count |
|----------|-------|
| BLOCKER  | 0 |
| HIGH     | 4 (HIGH-1, HIGH-2, HIGH-9, HIGH-22) |
| MEDIUM   | 11 |
| LOW      | 12 |
| NIT      | 4 |

## Top-3 fix priorities

1. **HIGH-1** — `SpecBackedBiotSavartJAX.x.setter` shape assumption. Add the assertion now; latent failure mode is silent shape mismatch on any fixed-DOF spec.
2. **HIGH-2** — Either wire `_uses_uniform_curve_xyz_fourier_fastpath` into `coil_set_spec()` or remove it. The current state misleads readers about the hot-path performance characteristics of `BiotSavartJAX`.
3. **HIGH-22** — Override `InterpolatedFieldJAX.dB_by_dX()` with an explicit error message instead of relying on the C++ trampoline NotImplementedError.

## Files audited (with absolute paths)

- `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/field/_jax_common.py`
- `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/field/biotsavart_jax.py`
- `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/field/biotsavart_jax_backend.py`
- `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/field/boozermagneticfield_jax.py`
- `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/field/circular_coil_jax.py`
- `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/field/dipole_field_jax.py`
- `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/field/dommaschk_jax.py`
- `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/field/interpolated_field_jax.py`
- `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/field/magneticfieldclasses_jax.py`
- `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/field/mirror_model_jax.py`
- `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/field/poloidal_field_jax.py`
- `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/field/reiman_jax.py`
- `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/field/sampling_jax.py`
- `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/field/scalar_potential_rz_jax.py`
- `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/field/toroidal_field_jax.py`
- `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/field/wireframefield_jax.py`

Cross-referenced against:
- `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/field/biotsavart.py`
- `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/field/boozermagneticfield.py`
- `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/field/magneticfield.py`
- `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/field/magneticfieldclasses.py`
- `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/field/wireframefield.py`
- `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/field/__init__.py`
- `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsoptpp/magneticfield.h`
- `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsoptpp/pymagneticfield.h`
- `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/jax_core/biotsavart.py`
- `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/jax_core/dipole_field.py`
- `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/jax_core/regular_grid_interp.py`
- `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/jax_core/field.py`
- `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/jax_core/interpolated_boozer_field.py`
- `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/jax_core/boozer_analytic.py`
- `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/_core/optimizable.py`
- `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/field/force.py` (alias-only confirmation)
