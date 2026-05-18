# simsopt Convention Baseline for JAX Port Audits

**Date:** 2026-05-16
**Branch:** `gpu-purity-stage2-20260405`
**Purpose:** Reference document capturing non-negotiable simsopt conventions
that any port (including the JAX port) must comply with. Reviewers cross-check
JAX modules under `src/simsopt/{field,geo,objectives}/*_jax*.py` and
`src/simsopt/jax_core/*.py` against the rules below.

Each section gives: **(a)** a one-sentence rule, **(b)** authoritative source
file(s) and line(s), **(c)** the typical violation pattern, **(d)** why it
matters. Citations are real; if a claim could not be anchored in code, it is
flagged "unverified".

---

## 1. Optimizable contract — graph DAG, lineage, DOF surface

**Rule.** Every Optimizable subclass must (i) declare its parent set through
`depends_on=[...]` at `Optimizable.__init__`, (ii) leave `local_dof_size`,
`local_full_dof_size`, `local_dof_names`, `local_full_x`, `unique_dof_lineage`,
`dof_indices`, and `dofs_free_status` to the base class, and (iii) override
only `recompute_bell()` (cache invalidation) and optionally
`set_recompute_flag()` (state-token bookkeeping). It must never reimplement
ancestor traversal, DOF concatenation, free/full splitting, or `__add__` /
`__mul__` aggregation by hand.

**Source.**
- `src/simsopt/_core/optimizable.py:567-627` — class docstring fixing the
  graph/DOF/return-fn model.
- `src/simsopt/_core/optimizable.py:630-751` — `__init__` accepting `dofs`,
  `external_dof_setter`, `depends_on`, `opt_return_fns`, `funcs_in`.
- `src/simsopt/_core/optimizable.py:962-1051` — `_get_ancestors`,
  `unique_dof_lineage`, `_update_full_dof_size_indices`,
  `update_free_dof_size_indices` (recursive walk down the DAG of children).
- `src/simsopt/_core/optimizable.py:1100-1172` — canonical
  `x` / `full_x` / `local_x` / `local_full_x` getters & setters.
- `src/simsopt/_core/optimizable.py:1174-1182` — `set_recompute_flag` walks
  the children set; subclasses extend, not replace.
- `src/simsopt/_core/optimizable_contract.py:1-22` — duck-type contract
  (`unique_dof_lineage`, `local_full_dof_size`, `local_dof_size`, `x`,
  `dof_size`) consumed by `Derivative` and `OptimizableDefaultDict`.

**Typical violation.**
- Building a flat DOF vector by `np.concatenate([coil.x for coil in coils])`
  instead of consuming `Optimizable.x` (which already concatenates over
  `unique_dof_lineage`).
- Overriding `x` or `full_x` to do anything other than relay through the base
  setter; the base setter is the one that triggers
  `DOFs._flag_recompute_opt` and propagates `set_recompute_flag` to children.
- Bypassing `depends_on` and manually wiring `_children`/`_add_child`.
- Returning a Python list from `dof_names` rather than letting the base
  class build it from `unique_dof_lineage`.

**Why it matters.** `Derivative.__call__` (`src/simsopt/_core/derivative.py:206-244`)
and gradient assembly (`src/simsopt/_core/derivative.py:45-60`) walk
`unique_dof_lineage` and `dofs.dep_opts()` directly — any shadow DOF surface
breaks gradient unrolling silently.

---

## 2. DOF management — `set_dofs`, caches, dtype

**Rule.** All DOF mutations flow through `DOFs._flag_recompute_opt`
(`src/simsopt/_core/optimizable.py:209-220`) which fires
`Optimizable.local_dof_setter` (if registered via `external_dof_setter`) and
calls `set_recompute_flag` on every dependent. DOF storage is always
`np.float64`; bounds are concatenated through `lineage`. JAX ports must not
introduce a parallel mutable DOF cache.

**Source.**
- `src/simsopt/_core/optimizable.py:132-181` — `DOFs.__init__` enforcing
  `np.array(x, dtype=np.double, copy=True)` and uniqueness of names.
- `src/simsopt/_core/optimizable.py:364-405` — `free_x` / `full_x` setters
  call `_flag_recompute_opt`.
- `src/simsopt/_core/optimizable.py:687-693` — when `external_dof_setter` is
  passed, the constructor pushes the initial DOFs into the C++/JAX backend.
- `src/simsopt/geo/surfacexyztensorfourier.py:126-131,186-191` —
  canonical use: `set_dofs_impl` is passed as `external_dof_setter`, and
  `set_dofs` simply assigns `self.local_full_x = dofs`.
- `src/simsopt/field/biotsavart_jax_backend.py:1048-1067` —
  `_advance_coil_dof_state()` and the `_set_global_coil_dofs` indirection
  show how a port may add a state token *on top of* the base
  setter, but never instead of it.

**Typical violation.**
- Holding a JAX `Array` in `self._x` and bypassing `self._dofs` so
  `_flag_recompute_opt` never fires.
- Casting to `jnp.float32` at the boundary — see §15 "no `cast` to `any`,
  double precision default".

**Why it matters.** Cache invalidation (§12) and adjoint-state tokens
(§14) all key off `recompute_bell` notifications. A silent DOF mutation =
stale objective and a stale adjoint.

---

## 3. Derivative protocol — `Derivative`, `@derivative_dec`, lineage projection

**Rule.** Functions named `dJ()` on Optimizables must:
1. Build a `Derivative({owner_optimizable: numpy_block, ...})` mapping
   parent Optimizable instances to per-parent
   `local_full_dof_size`-length numpy arrays.
2. Be wrapped with `@derivative_dec`
   (`src/simsopt/_core/derivative.py:254-270`) so the unwrapped `dJ()` form
   returns a flat `np.ndarray` over free DOFs.
3. Return per-parent blocks of length `local_full_dof_size`
   (i.e., **including fixed entries** — slicing to free DOFs happens in
   `Derivative.__call__`).
4. Project chain-rule sums via `Derivative.__add__` / `__iadd__`
   (`src/simsopt/_core/derivative.py:146-190`) — never roll a custom sum.

**Source.**
- `src/simsopt/_core/derivative.py:13-25` — `OptimizableDefaultDict` produces
  zero blocks of `local_full_dof_size`.
- `src/simsopt/_core/derivative.py:45-60` —
  `_iter_local_free_derivative_blocks` walks `unique_dof_lineage`,
  multiplies entries by `local_dofs_free_status`, and sums shared DOF
  carriers.
- `src/simsopt/_core/derivative.py:63-141` — the chain-rule recipe spelled
  out in the docstring.
- `src/simsopt/_core/derivative.py:28-35` — `_coerce_derivative_block`
  permits a JAX `Array` block but copies it to NumPy through
  `jax.device_get`. **Returning JAX leaves directly is allowed; they will be
  cast at composition time.**
- `src/simsopt/objectives/fluxobjective.py:85-133` — canonical
  pattern: `@derivative_dec`, then `dJ()` returns
  `self.field.B_vjp(dJdB)` (which is itself a `Derivative`).
- `src/simsopt/field/biotsavart.py:60-119` — `B_vjp`/`B_and_dB_vjp` sum
  `Derivative` objects via `coils[i].vjp(...)` (`Coil.vjp` returns a
  `Derivative`).
- `src/simsopt/objectives/fluxobjective_jax.py:142-164` —
  `_field_dofs_gradient_to_derivative`: how a JAX flat gradient is
  splayed back into per-parent `local_full_dof_size` blocks using
  `local_dofs_free_status`. This is the canonical reverse direction.

**Typical violation.**
- Returning a single concatenated `np.ndarray` from `dJ()` shaped to the
  free-DOF length, expecting `derivative_dec` to convert it — it cannot.
  `Derivative` keys are Optimizables, not lengths.
- Forgetting `@derivative_dec`, in which case `obj.dJ()` returns the
  `Derivative` container, breaking SciPy-style callers.
- Producing per-parent blocks of `local_dof_size` (free-only) rather than
  `local_full_dof_size`.

**Why it matters.** The Derivative dataclass is the *contract* by which
gradients compose across the DAG; a port that silently bypasses it produces
mismatched gradients during outer optimization.

---

## 4. File layout and module organization

**Rule.** JAX modules live alongside their C++ counterparts; pure JAX kernels
live under `src/simsopt/jax_core/`; ports of *Optimizable adapters* live as
`<module>_jax.py` or `<module>_jax_backend.py` in `simsopt.{field,geo,objectives}`.
Every JAX subpackage `__init__.py` exposes JAX classes through the
`build_lazy_export_map` registry guarded by `_has_simsoptpp` and `_has_jax`
probes; `jax_core/*.py` **never imports `simsoptpp`** (`sopp.*` strings appear
only in docstrings/comments).

**Source.**
- `src/simsopt/jax_core/__init__.py:1-100+` — public surface of pure JAX
  kernels and specs.
- `src/simsopt/field/__init__.py:1-66` —
  `_has_simsoptpp` and `_has_jax` probes; `_CPU_FIELD_MODULES`,
  `_JAX_FIELD_MODULES`, `_JAX_FIELD_SIMSOPTPP_MODULES`.
- `src/simsopt/geo/__init__.py:1-105` — same pattern plus
  `_DYNAMIC_JAX_EXPORTS` for forward declarations and
  `_ORDERED_JAX_CPU_GEO_BLOCK` for ordered import.
- `src/simsopt/objectives/__init__.py:1-50` — guarded `from .fluxobjective_jax
  import *` inside `try/except`.
- `src/simsopt/_lazy_exports.py:7-46` — `build_lazy_export_map` parses
  literal `__all__` lists from each module and rejects duplicate exports
  across submodules.
- `src/simsopt/__init__.py:1-58` — top-level applies the JAX runtime
  config only when `should_eagerly_configure_jax()` returns True; it does
  not import `simsoptpp` at module top level (the `try/except`
  swallows `ImportError` and `AttributeError`).
- `src/simsopt/field/magneticfield.py:1-9,43-92` — JAX-native
  composites guarded with `_simsopt_jax_native_field` flag and
  `_raise_if_strict_jax_mixed_composition`.

**Typical violation.**
- A new pure JAX kernel imports `simsoptpp as sopp` for "convenience".
- A `<module>_jax.py` adapter omits `__all__` or adds a non-literal
  expression to it — `build_lazy_export_map` raises `RuntimeError`
  (`_lazy_exports.py:18-20`).
- Hard-coding `from .biotsavart_jax_backend import BiotSavartJAX` at the
  package top level — defeats the CPU-only install path and breaks the
  "no JAX in CPU-only env" smoke test.

**Why it matters.** simsopt must be importable on hosts without JAX (and on
hosts without simsoptpp); the lazy-export probes guarantee that.

---

## 5. Tensor axis convention — `dB_by_dX[p, j, l]`

**Rule.** For magnetic fields, surface and curve geometries:
`dB_by_dX[p, j, l] = ∂_j B_l(x_p)`. Axis 1 is the spatial derivative
direction; axis 2 is the field component. The same convention applies to
`dA_by_dX[p, j, l]`, `d2B_by_dXdX[p, j, k, l]`, surface
`dgamma_by_dcoeff[..., l, i]` (l = Cartesian component, i = coefficient
index), and similar.

**Source.**
- `simsopt-jax/CLAUDE.md:182` — the official rule statement.
- `src/simsopt/jax_core/biotsavart.py:597-635` — `biot_savart_dB_by_dX`
  routes through `_get_kernel(_Integrand.B, _DiffMode.JACOBIAN)` and is
  expected to produce `(npoints, 3, 3)` with derivative axis first.
- `tests/integration/test_stage2_jax.py:1809-1810` — explicit
  contract test "validates the forward Jacobian tensor convention".

**Typical violation.**
- Internal kernels may legitimately store `(component, derivative)` order
  for cache efficiency, but the **public** kernel/adapter must transpose
  before returning. The audit at
  `.artifacts/parity_audit_2026-05-16/08_wireframe_field.md:53-61` flags
  the wireframe `dB[p, k, m]` deliberately-divergent layout and treats it
  as a documentation-only concern.
- VJP cotangents that feed `field.B_vjp(v)` shaped `(npoints, 3)` must
  obey the same component-axis convention as `field.B()` returns
  (`src/simsopt/field/biotsavart.py:94-119`); reshaping `(npoints, 3, 3)
  → (npoints, 3)` requires matching axis 2 of `dB` to the field-component
  axis of `v`.

**Why it matters.** A single transposed axis between Python and C++ produces
silently wrong gradients with the right magnitude — the worst kind of bug.

---

## 6. Quadrature, nfp factor, stellsym DOF ordering

**Rules.**
- **nfp factor.** The CPU surface integration scheme multiplies the per-point
  weight by `1/(nfp * nphi * ntheta)`; geometric quantities (Volume, Area,
  ToroidalFlux) cancel one `nfp` factor in the periodicity loop. A port must
  match the CPU integrand exactly, including this cancellation. CLAUDE.md
  records this as a confirmed *non*-bug in the 2026-03-18 review.
- **Surface coefficient layout for `SurfaceXYZTensorFourier`.**
  - Rows (theta basis): `{1, cos θ, …, cos(mpol·θ), sin θ, …, sin(mpol·θ)}`.
  - Columns (phi basis): `{1, cos(nfp·φ), …, cos(ntor·nfp·φ),
    sin(nfp·φ), …, sin(ntor·nfp·φ)}`.
- **Stellsym DOF convention.** For `SurfaceXYZTensorFourier`:
  - `x` uses **cos-cos + sin-sin** quadrants (even-even).
  - `y` and `z` both use **cos-sin + sin-cos** quadrants (odd-odd).
  - i.e., `y` transforms like `z` under stellsym (φ,θ) → (−φ,−θ).
- **Public DOF order = `[x, y, z]` flat blocks of size `(2*mpol+1) * (2*ntor+1)`.**
  Within each block, scatter follows
  `stellsym_scatter_indices(mpol, ntor)` row-major over `(m, n)`.

**Source.**
- `src/simsopt/geo/surface_fourier_jax.py:8-21` — basis layout docstring
  (mirrors `src/simsoptpp/surfacexyztensorfourier.h`).
- `src/simsopt/geo/surface_fourier_jax.py:1128-1180` —
  `_is_stellsym_xy`, `_is_stellsym_z`, `stellsym_scatter_indices` with
  the explicit `cos-cos+sin-sin` (x) and `cos-sin+sin-cos` (y, z)
  comment.
- `src/simsopt/geo/surfacexyztensorfourier.py:53-178` —
  C++ `sopp.SurfaceXYZTensorFourier`-anchored DOF order, `skip()` rule,
  and `_make_names_helper` order.
- `CLAUDE.md:184-187` — repeats the convention and warns that any port
  must verify the scatter against the CPU `set_dofs_impl` ordering.

**Typical violation.**
- Using cos-cos+sin-sin for y/z (treating y like x rather than like z).
- Dropping the `nfp` factor in JAX kernels because "JAX kernel is per-period";
  see CLAUDE.md confirmed-not-bug entry.
- Reordering blocks to `[x_block, x_block_sin, y_block, ...]` to "compress"
  fixed entries — breaks lineage projection in §3.

**Why it matters.** Both VMEC bootstrap and Boozer solver state-parity gates
read DOF byte vectors; a permuted stellsym scatter is invisible at the
objective level but produces gibberish surface geometry.

---

## 7. Public vs private API

**Rule.** Public exports live in module-level `__all__` and are surfaced via
the `_lazy_exports` registry. Private helpers begin with `_`, live in
`_*_jax.py` modules (e.g., `_distance_jax.py`), or are tucked into
`simsopt._core` for internal-only protocols. The `jax_core/` subpackage is a
public *kernel* layer; the adapter wrappers under `simsopt.{field,geo,objectives}`
are public *Optimizable* surfaces.

**Source.**
- `src/simsopt/__init__.py:1-58` — the "ATTENTION" comment forbids
  abusing the top-level by importing all symbols — sub-packages follow
  the same rule.
- `src/simsopt/geo/__init__.py:23-65` — `_BASE_CPU_GEO_MODULES`,
  `_JAX_GEO_MODULES`, `_DYNAMIC_JAX_EXPORTS`, `_ORDERED_JAX_CPU_GEO_BLOCK`
  enumerate the public modules.
- `src/simsopt/geo/_distance_jax.py` (path-only) — leading-underscore
  private helper module.
- `src/simsopt/_core/jax_host_boundary.py:14-101` — `host_array`,
  `host_scalar`, `host_float`, etc. — private to `simsopt._core` but
  imported by many JAX adapters; **adapters must call them, not roll
  their own `jax.device_get` boilerplate.**

**Typical violation.**
- Exporting an internal helper by accident through `__all__`.
- Bypassing `host_array` / `host_scalar` and calling
  `np.asarray(jax.device_get(...))` inline.
- Adding a `simsopt.<sub>.foo_jax_internal` module whose `__all__` is
  empty — `build_lazy_export_map` will still scan it; prefer a leading
  underscore.

**Why it matters.** The lazy-export probe rejects duplicate exports across
submodules (`_lazy_exports.py:30-36`); a colliding name between
`xxx_jax.py` and `xxx_jax_backend.py` raises at import.

---

## 8. Parity ladder & oracle lint

**Rule.** Every JAX↔C++ parity claim must (a) cite one of the four lanes in
`benchmarks/validation_ladder_contract.py::PARITY_LADDER_TOLERANCES` and
(b) cite an independent oracle (C++ symbol / closed-form / pinned dataset /
FD) per `tests/REVIEWER_ORACLE_LINT.md`. JAX-vs-JAX assertions, re-export
`is` checks, and "host wrapper that routes through JAX" comparisons are
banned tautologies.

**Source.**
- `benchmarks/validation_ladder_contract.py:52-100` — defines `direct_kernel`
  (`rtol=1e-10, atol=1e-12`, same-state, C++ oracle required),
  `ls_wrapper_gradient`, `derivative_heavy` (scalar `1e-10`, first deriv
  `1e-8`, second deriv `1e-6`), `reporting_contract`,
  `direct_hessian_oracle`, `exact_well_conditioned_adjoint`, and so on.
- `tests/REVIEWER_ORACLE_LINT.md:1-80+` — full lint rules and examples.
- CLAUDE.md "Parity ladder SSOT" — repeats the contract.

**Typical violation.**
- `assert_allclose(wrapper.J(), kernel.J())` when the wrapper calls the
  kernel directly (re-export tautology).
- `assert jax_path(x) == host_path(x)` where `host_path` invokes the same
  JAX kernel.
- Choosing tolerances empirically rather than deriving them from the lane
  contract.
- Asserting parity using a NumPy reimplementation of the same formula.

**Why it matters.** Tier-1 tautology rejection is the only thing that
distinguishes a parity test from a regression test of the JAX code with
itself.

---

## 9. Backend selection

**Rule.** Two orthogonal axes:
- **Code-path backend.** `SIMSOPT_BACKEND={cpu,jax}` (legacy
  `STAGE2_BACKEND=jax`).
- **JAX device platform.** `SIMSOPT_JAX_PLATFORM={cpu,cuda,metal}`
  (legacy `SIMSOPT_JAX_BACKEND`).

The SSOT is the **mode** API — `SIMSOPT_BACKEND_MODE` selects one of
`native_cpu`, `jax_cpu_fast`, `jax_cpu_parity`, `jax_gpu_fast`,
`jax_gpu_parity`, `jax_metal_smoke`. The legacy env vars are read/written by
`set_backend()` for compatibility. Programmatic access:
`from simsopt.backend import get_backend, is_jax_backend, get_jax_platform,
set_backend`.

**Source.**
- `src/simsopt/backend/runtime.py:35-40` — env var names.
- `src/simsopt/backend/runtime.py:106-191` — `VALID_BACKEND_MODES`,
  `_MODE_TO_RUNTIME`, `_MODE_POLICY_DEFAULTS`.
- `src/simsopt/backend/runtime.py:1626-1655` —
  `raise_if_strict_jax_fallback` and `warn_if_jax_fallback` —
  CPU-fallback rejection on strict modes.
- `src/simsopt/backend/runtime.py:1716-1774` — `apply_jax_runtime_config`,
  `set_backend`.
- `src/simsopt/backend.py:1-105` — public facade re-exporting the runtime
  surface as `simsopt.backend`.
- `src/simsopt/__init__.py:21-37` — eager runtime config when an explicit
  selector is set.

**Typical violation.**
- Reading `os.environ["SIMSOPT_BACKEND"]` inline; use
  `get_backend()`/`is_jax_backend()`.
- Adding a new private env var to bypass the mode contract.
- Calling `jax.config.update("jax_platforms", ...)` directly inside an
  Optimizable adapter — must be done via `apply_jax_runtime_config()` so
  the deterministic XLA flag validation fires
  (`src/simsopt/backend/runtime.py:1693-1735`).

**Why it matters.** Strict modes (`jax_cpu_parity`, `jax_gpu_parity`) reject
CPU fallback to enforce parity contracts; ad hoc env reads defeat the
strict mode.

---

## 10. Adjoint / IFT runtime SSOT

**Rule.**
- `BoozerSurfaceJAX.get_adjoint_runtime_state()`
  (`src/simsopt/geo/boozersurface_jax.py:3713-3761`) is the SSOT for the
  *exact*-lane adjoint runtime callbacks (forward/transpose matvec, forward
  /transpose solve, grouped VJP stream, dense factor availability flag).
- In the **exact** lane the `PLU` field of `res` is **debug metadata only**;
  the adjoint must always reach the solve through
  `linear_solve_backend="operator"`. The exact path strips
  `optimizer_backend` from public options
  (`boozersurface_jax.py:3122`, `:3185-3186`) so user-visible exact results
  never carry dense factors.
- In the **LS** lane the `PLU` factors *are* load-bearing runtime data: the
  SciPy reference callbacks at `boozersurface_jax.py:3514-3540` and the
  traceable adjoint `_traceable_solve_plu_linearization`
  (`surfaceobjectives_jax.py:3167-3220`) consume them under
  `lax.stop_gradient` so forward and adjoint Hessian actions are
  bit-identical when `decision_size² × 8 ≤ max_dense_jacobian_bytes`.
- M5 wrappers (`BoozerResidualJAX`, `IotasJAX`, `NonQuasiSymmetricRatioJAX`)
  must compute `dJ/d_coils = ∂J/∂coils − adj^T ∂g/∂coils` and route the
  inner-solve via `get_adjoint_runtime_state()`. A successful traceable
  forward solve with a *failed* adjoint solve must surface a non-finite
  gradient, **never** a finite direct gradient or failure-penalty
  fallback.

**Source.**
- `src/simsopt/geo/boozersurface_jax.py:3200-3286` — `BoozerSurfaceJAX`
  class header including the SSOT statement: "object wrapper is
  intentionally stateful and should be treated as thread-confined".
- `src/simsopt/geo/boozersurface_jax.py:3713-3761` —
  `get_adjoint_runtime_state`.
- `src/simsopt/geo/boozersurface_jax.py:3768-3771` —
  `recompute_bell` only sets `need_to_run_code = True`.
- `src/simsopt/geo/boozersurface_jax.py:3514-3540` — SciPy LS callbacks
  build `H_host = P @ L @ U` from `self.res["PLU"]`.
- `src/simsopt/geo/surfaceobjectives_jax.py:3167-3220` —
  `_traceable_solve_plu_linearization`.
- CLAUDE.md "Adjoint / warm-start operator solves" and "Note on
  `linear_solve_factors`" — explicit statement.

**Typical violation.**
- Exact-lane wrappers reading `self.res["PLU"]` directly instead of
  calling `solve_transpose` from `get_adjoint_runtime_state()`.
- Returning a `Derivative` from an adjoint-failed solve where the gradient
  vector has finite values copied from the direct partial — the M5
  contract requires the gradient to be non-finite so the outer optimizer
  rejects the step.

**Why it matters.** Implicit differentiation correctness *is* the M5
contract; mishandling the adjoint produces silently-wrong gradients that
look reasonable.

---

## 11. JSON serialization (GSONable)

**Rule.** Any new Optimizable adapter must respect the `GSONable` protocol
implemented by `simsopt._core.json.SIMSON`. The default `as_dict` on
`Optimizable` (`src/simsopt/_core/optimizable.py:1633-1638`) serializes the
DOFs only when `local_full_x` is non-empty; ports may override `as_dict` to
add fields, but must keep the `@module`/`@class` and `dofs` keys consistent.
Reconstruction uses `from_dict` and `GSONDecoder.process_decoded`.

**Source.**
- `src/simsopt/_core/json.py:55-100` — module/class redirect indirection.
- `src/simsopt/_core/optimizable.py:1633-1673` — `as_dict`, `save`,
  `from_str`, `from_file`, `load`, `save`.
- `src/simsopt/field/biotsavart.py:223-239` — canonical example:
  `BiotSavart.as_dict` adds `points`; `from_dict` rebuilds the coil list
  then restores the points.

**Typical violation.**
- A new adapter overrides `as_dict` and forgets to call `super().as_dict`,
  so the `dofs` block disappears.
- Storing a JAX `Array` directly in `as_dict` output (it cannot be
  json-encoded). Cast to NumPy via `host_array` first.

**Why it matters.** Autoresearch/single-stage workflows persist Optimizable
trees as JSON; a port that breaks JSON round-trips silently corrupts
checkpoints.

---

## 12. JIT-closure rules at the boundary

**Rule.**
- A JAX adapter may capture **immutable** data inside a JIT closure at
  construction time (`SquaredFluxJAX` captures the *fixed-surface*
  `gamma`, `normal`, and target arrays).
- A JAX adapter must **not** capture mutable handles whose state could
  shift under it; if the captured data must change, the captured JIT must
  be invalidated (set tokens, see §14).
- Mutating the source of a JIT closure after construction is an error;
  `SquaredFluxJAX` records the surface DOF fingerprint at construction and
  raises if a later call detects a different fingerprint
  (`src/simsopt/objectives/fluxobjective_jax.py:87-104,200+`).

**Source.**
- `src/simsopt/objectives/fluxobjective_jax.py:1-21,87-104` — docstring
  pins the contract: "The fixed surface is captured … once at construction
  time and kept on JAX arrays for the lifetime of the objective. … mutating
  the surface's free DOFs after construction raises a `RuntimeError`."
- `src/simsopt/objectives/fluxobjective_jax.py:200-end` (init) — captures
  the surface fingerprint.
- CLAUDE.md "JIT closure strategy" — repeats the rule.

**Typical violation.**
- Capturing a mutable Python list inside a JIT function and expecting
  appends to be visible (they are not — JIT traces once).
- Forgetting to bump the cache signature (§14) when the closure changes,
  causing the traceable bundle cache to return a stale entrypoint.

**Why it matters.** A surface mutation after `SquaredFluxJAX` construction
will be silently ignored — the wrong objective will be optimized.

---

## 13. Error handling and result dicts

**Rule.**
- Solver result dicts (`boozer_surface.run_code()` `res`, `_run_*` returns)
  use a fixed key set: `iota`, `G`, `sdofs`, `success`, `primal_success`,
  `iter`, `fun`, `gradient`, `residual`, `weight_inv_modB`, `vjp`, `PLU`,
  `LU_PIV`, `hessian`, `linearization_kind`,
  `linear_solve_backend`, `dense_linear_solve_factors_available`,
  `failure_category`, `failure_stage`, `jacobian_materialized`,
  `dense_jacobian_shape`, `dense_jacobian_bytes`, `max_dense_jacobian_bytes`,
  `optimizer_method`, `pre_newton`, `adjoint_linear_solve_available`,
  `vjp_groups`.
- JAX integer/boolean scalars must be cast to Python `int()`/`bool()`
  before storing in result dicts: `"iter": int(result.nit)`,
  `"success": bool(result.success)`.
- Exact-lane scaling-limit failures are predictable reporting limits, **not**
  adjoint-availability errors:
  `failure_category="scaling_limit"`,
  `failure_stage="dense_jacobian_finalization"`.

**Source.**
- `src/simsopt/geo/boozersurface_jax.py:3370-3406,3713-3761,6040-6100` —
  result-dict population sites and the use of `bool(...)`/`int(...)` casts.
- CLAUDE.md "JAX scalar boundary conversions" and "Exact Boozer
  scaling-limit contract" — restate the rules.
- `src/simsopt/_core/jax_host_boundary.py:14-44` — `host_scalar`,
  `host_float`, `host_int`, `host_bool`, `host_inf_norm` — the
  blessed conversion helpers.

**Typical violation.**
- Returning `result.nit` (a numpy/JAX scalar) into a dict that later flows
  through `json.dumps` — fails to serialize.
- Reusing a "failed-adjoint" code path for "exact dense-Jacobian
  too large" — the latter is recoverable diagnostic data, the former is a
  gradient correctness incident.

**Why it matters.** Downstream tooling (parity probes, autoresearch
launchers, JSON checkpoints) consumes these dicts; missing keys or wrong
types break runs deep in the optimization loop.

---

## 14. Ancestor invalidation token pattern

**Rule.** Mutable JAX adapters expose deterministic state tokens that
JIT-cache signatures key off:
- **`BoozerSurfaceJAX._traceable_solve_state_token`** — bumped on every new
  solve. Wrappers use it as the "solved baseline freshness" key.
- **`BiotSavartJAX._coil_dof_state_token`** — bumped on aggregate `x` /
  `full_x` writes *and* on ancestor DOF invalidation through
  `set_recompute_flag`. The same field name is used on
  `SpecBackedBiotSavartJAX`.
- **`_traceable_runtime_cache_signature`** — exposed by success filters
  (e.g., for traceable objective bundles); used for semantic cache sharing.
  When absent the cache key holds a live callable-reference signature
  compared by `is`, **never** `id(callable)` or user-equality on the
  callable.
- **Solved-state token rule:** `make_traceable_objective_runtime_bundle()`
  caches against a signature of solved baseline, objective kwargs, coil
  runtime state, coil reconstruction layout, and success filter. Rebuild
  after any captured input changes — never mutate captured objects in
  place.

**Source.**
- `src/simsopt/geo/boozersurface_jax.py:133` (`_new_traceable_solve_state_token`),
  `:721`, `:3266`, `:3768-3771`.
- `src/simsopt/field/biotsavart_jax_backend.py:91-93`
  (`_new_coil_dof_state_token`), `:460`, `:499`, `:1023`, `:1050`,
  `:1052-1059` (set_recompute_flag override).
- CLAUDE.md "Traceable runtime bundle cache contract" — explicit
  specification.

**Typical violation.**
- Using Python `id(callable)` as a cache key, which is reused after gc.
- Failing to override `set_recompute_flag` on a port, so ancestor DOF
  changes do not invalidate the cached coil state token.
- Forgetting to invalidate the bundle when the surface DOF fingerprint
  changes (related rule, §12).

**Why it matters.** A stale traceable bundle silently returns the wrong
gradient with no error signal.

---

## 15. Other non-negotiables

**a. No dynamic imports.** `await import(...)` is banned per global rules.
`importlib.import_module` is used only by `_lazy_exports.py` and
`simsopt.geo.__init__.py` for the ordered geo block.

**b. No `cast` to `any`.** Python ports should rely on the duck-type
contract (`optimizable_contract.has_*_contract`) rather than `cast(...)`.

**c. Double precision default.** JAX runs with `jax_enable_x64=True` —
enforced eagerly in `src/simsopt/__init__.py:26-32`. Float arrays are
`np.float64` / `jnp.float64`. The shared `_math_utils.as_jax_float64` is
the conversion gate
(`src/simsopt/jax_core/_math_utils.py:57`).

**d. No defensive guards.** Per global rules, don't add try/except blocks
beyond what is required. The only blanket guards in tree are at module
import boundaries (`try: import jax; except ImportError: ...`).

**e. No `.env` files.** Per global rules.

**f. No simsoptpp imports in `jax_core/`.** Cross-checked by grep — only
docstrings reference `sopp.*` in `jax_core/*.py`. JAX adapter modules under
`field/*_jax_backend.py` and `geo/*_jax.py` may import `simsoptpp` *via*
the adapter pattern (CPU surface at the boundary, JAX on the hot path),
but the kernel layer stays pure.

**g. Strict mode rejects mixed composition.**
`_raise_if_strict_jax_mixed_composition`
(`src/simsopt/field/magneticfield.py:23-41`) rejects
`MagneticFieldSum` / `MagneticFieldMultiply` mixes of JAX-native and
CPU-only fields when strict-JAX mode is active. JAX-native fields opt
in by setting the **class** attribute `_simsopt_jax_native_field = True`.

**h. Stellsym-symmetric DOFs are stored *flat*, not as a mask.** When a
DOF is `fixed`, it remains in `full_x` but is excluded from `x`. Do not
introduce alternate representations.

**i. Optimizable equality is name-based.** `Optimizable.__eq__` uses
`self.name == other.name`
(`src/simsopt/_core/optimizable.py:772-782`); the auto-generated name
includes the `_id.id` counter. Do not assume `is` identity.

**j. `weakref` is used for the children set.** Mutating
`Optimizable._children` directly is reserved for the base class
(`src/simsopt/_core/optimizable.py:879-903`).

**k. C++ ANGLE_RECOMPUTE brace pattern.** When touching
`src/simsoptpp/surfacerzfourier.cpp` VJP loops, the `if(i % ANGLE_RECOMPUTE
== 0)` blocks require explicit `{}` braces (bare `if` only guards the first
statement). CLAUDE.md records this as a fixed bug — preserve the rule.

**l. Floating-point reproducibility cross-machine.** Byte-identity CPU↔JAX
state parity is **not** a portable invariant on the LS path; cross-machine
absolute thresholds set `sdofs_inf ≤ 1e-11`. Reserve `rtol=1e-12` for
same-state direct-kernel parity on a single machine.

**m. Read-only spec-backed views.** `SpecBackedCurve`, `SpecBackedCoil`,
`SpecBackedCurrent`, `SpecBackedBiotSavartJAX`
(`src/simsopt/field/biotsavart_jax_backend.py:290-490`) are read-only
adapters that reconstruct C++-style API from immutable specs. They expose
`dgamma_by_dcoeff_vjp`, `dgammadash_by_dcoeff_vjp`, etc., that route
through `curve_pullback_from_dofs` — not the C++ vjp_graph.

---

## Quick checklist (1-line rules)

1. Subclass `Optimizable`, pass `depends_on=[...]`, never rebuild the DOF
   surface by hand.
2. Mutate DOFs only through `local_full_x` / `x` / `full_x` setters; let
   the base class fire `recompute_bell`.
3. `dJ()` returns a `Derivative({parent: local_full_dof_size_block, ...})`
   and is wrapped with `@derivative_dec`.
4. JAX kernels live under `jax_core/` and never import `simsoptpp`;
   adapters live as `*_jax.py` / `*_jax_backend.py` and are surfaced
   through `_lazy_exports`.
5. Public Jacobian convention is `dB_by_dX[p, j, l] = ∂_j B_l(x_p)`.
6. Stellsym DOF order: `x` = cos-cos + sin-sin, `y`,`z` = cos-sin + sin-cos.
   Surface DOF blocks are `[x, y, z]`. `nfp` factor cancels with
   `1/(nfp * nphi * ntheta)` per-period quadrature.
7. Private helpers start with `_`; `host_array`/`host_scalar` are the
   blessed device→host conversions.
8. Tests cite a parity-ladder lane and an independent oracle (C++ / closed
   form / dataset / FD). JAX-vs-JAX, re-export `is`, and host-routes-JAX
   comparisons are tautologies.
9. Use `SIMSOPT_BACKEND_MODE` (mode SSOT) or `SIMSOPT_BACKEND` /
   `SIMSOPT_JAX_PLATFORM` (legacy). Programmatic: `simsopt.backend.*`.
10. `BoozerSurfaceJAX.get_adjoint_runtime_state()` is the exact-lane
    adjoint SSOT; exact `PLU` is debug-only, LS `PLU` is load-bearing.
11. Respect `GSONable`; cast JAX arrays to NumPy before serialization.
12. JIT closures capture immutable surface arrays; mutating the source
    raises.
13. Solver result dicts use fixed key set; cast JAX scalars with
    `int()`/`bool()`/`host_scalar`. Scaling-limit ≠ adjoint failure.
14. State tokens (`_traceable_solve_state_token`,
    `_coil_dof_state_token`, `_traceable_runtime_cache_signature`) key the
    traceable cache. Override `set_recompute_flag` to advance the coil
    token on ancestor invalidation.
15. JAX runs `float64`; no dynamic imports; no `cast(any)`; no defensive
    try/except.
