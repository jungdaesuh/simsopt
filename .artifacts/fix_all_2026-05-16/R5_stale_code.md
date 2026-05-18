# R5: Stale, Dead, and Obsolete Code Audit

**Scope:** `src/simsopt/jax_core/`, `src/simsopt/{field,geo,objectives,solve}/*_jax*.py`,
`src/simsopt/geo/_distance_jax.py`, `src/simsopt/field/_jax_common.py`,
`src/simsopt/backend.py`.

**Method:** grep for marker comments, alias patterns, deprecated JAX APIs, and
unused symbol references; cross-check candidates against the rest of the repo
(`src/`, `tests/`, `benchmarks/`).

**Headline:** the in-scope code is already very clean. The scan turned up
**zero** TODO/FIXME/XXX/HACK/WIP/DEPRECATED/OBSOLETE comments, **zero**
deprecated JAX-API call sites (`pmap`, `device_put_sharded`, `register_pytree_node`,
runtime `PRNGKey` calls), and **zero** commented-out code blocks (every 3+ line
"#" block we sampled is documentation, not stripped code).

The remaining findings are narrow: three unused symbols, two confirmed
alias-only ports, one over-cautious lazy import (already documented),
one DRY violation across three modules, one latent dead fallback path
(already H-17 in the convention review), one stale shim docstring, two
stale "JAX 0.9.2" version pins, and one stale facade export. None of
these block release.

---

## 1. TODO / FIXME / XXX / HACK / WIP / DEPRECATED / OBSOLETE markers

`grep -rEn "TODO|FIXME|XXX|HACK|WIP|DEPRECATED|OBSOLETE"` across the in-scope
set returns **no matches**. (Upstream `_core/optimizable.py`,
`geo/surfacerzfourier.py`, `geo/surfacegarabedian.py`, `mhd/spec.py`, and
`geo/boozersurface.py` carry historical TODOs, but those files are out of scope
for this audit and inherited from the upstream simsopt port.)

**Verdict:** clean — no action.

---

## 2. Commented-out code blocks (3+ consecutive comment lines)

A sweep for 4+ consecutive comment lines flagged candidates in
`surface_fourier_jax.py`, `boozer_residual_jax.py`, `optimizer_jax.py`,
`surfaceobjectives_jax.py`, `boozersurface_jax.py`,
`analytic_pure_fields.py`, `field.py`, `specs.py`, `sharding.py`,
`tracing.py`, `interpolated_boozer_field.py`, `surface_henneberg.py`,
`_elliptic.py`, `regular_grid_interp.py`, `pm_optimization.py`,
`biotsavart.py`, `biotsavart_cpu_ordered.py`, `surface_fourier.py`,
`magnetic_axis_helpers.py`, `fluxobjective_jax.py`,
`boozermagneticfield_jax.py`, `dipole_field_jax.py`,
`surface_fourier_jax_cpu_ordered.py`.

Each one inspected is a doc/math block (rationale, equations, parity
contract, JAX-API caveat) — *not* stripped code. Examples:

- `surface_fourier_jax.py:328-339` — ASCII block describing the
  ``clamped_dims`` BC enforcer.
- `boozersurface_jax.py:3462-3476` — explanation of Phase-2 factor-once
  dispatch and the `optimizer_backend == "scipy"` carve-out.
- `magnetic_axis_helpers.py:602-614` — closed-form 2x2 eigenvalue
  derivation, justifying why `jnp.linalg.eig` is not used.
- `biotsavart.py:120-129` — domain-edge divergence policy vs the C++
  oracle.

**Verdict:** clean — no action.

---

## 3. Unused module-level symbols

### 3.1 `scalar_at_axis0` in `simsopt.jax_core._math_utils`

- **File:line:** `src/simsopt/jax_core/_math_utils.py:76`
- **Category:** Unused public function.
- **Snippet:**
  ```python
  def scalar_at_axis0(array, index: int) -> jax.Array:
      selector = np.zeros(int(array.shape[0]), dtype=np.float64)
      selector[int(index)] = 1.0
      return jnp.dot(array, _explicit_device_array(selector, dtype=np.float64))
  ```
- **Evidence:** `grep -rn 'scalar_at_axis0' src/ tests/ benchmarks/ examples/`
  returns only the definition line.
- **Verdict:** UNUSED.
- **Suggested action:** DELETE. The neighbouring `scalar_like` /
  `zero_padding_like` / `pad_axis` / `zeros` / `eye` are all referenced;
  `scalar_at_axis0` is dead.

### 3.2 `optimizable_full_dofs_from_map_spec` in `simsopt.jax_core.curve_geometry`

- **File:line:** `src/simsopt/jax_core/curve_geometry.py:281`
- **Category:** Unused public function.
- **Snippet:**
  ```python
  def optimizable_full_dofs_from_map_spec(
      map_spec: OptimizableDofMapSpec,
      owner_dofs,
  ):
      return _mapped_full_dofs(map_spec, owner_dofs)
  ```
- **Evidence:** `grep -rn 'optimizable_full_dofs_from_map_spec' src/ tests/
  benchmarks/ examples/` returns only the definition. The sibling
  `optimizable_input_dofs_from_map_spec` (line 288) is exported and
  imported by `jax_core/field.py:36` and `field/force.py:38`.
- **Verdict:** UNUSED.
- **Suggested action:** DELETE — or DOCUMENT if intended as a planned API
  for completeness alongside `_input_` variant.

### 3.3 `BiotSavartBPullback` in `simsopt.field.biotsavart_jax_backend`

- **File:line:** `src/simsopt/field/biotsavart_jax_backend.py:265`
- **Category:** Public alias-only export with no external consumer.
- **Snippet:**
  ```python
  BiotSavartBPullback = BiotSavartFieldPullback
  ```
- **Evidence:** `grep -rn 'BiotSavartBPullback' src/ tests/ benchmarks/`
  returns the definition (line 265) and its `__all__` entry (line 81)
  only. Outside callers all use `BiotSavartFieldPullback`.
- **Verdict:** OBSOLETE alias.
- **Suggested action:** DELETE both the alias and the `__all__` entry.

---

## 4. Lazy-import shims

### 4.1 Confirmed over-cautious: `_get_grouped_biot_savart`

- **File:line:** `src/simsopt/geo/boozer_residual_jax.py:509-513`
- **Category:** Unnecessary lazy import.
- **Snippet:**
  ```python
  def _get_grouped_biot_savart():
      """Lazily import grouped Biot-Savart (avoids simsopt top-level)."""
      from simsopt.field.biotsavart_jax import grouped_biot_savart_B
      return grouped_biot_savart_B
  ```
- **Reason:** The `biotsavart_jax` shim only depends on
  `simsopt.jax_core.biotsavart`; no transitive `simsopt.geo.*` cycle
  exists. The convention review (`.artifacts/jax_convention_review_2026-05-16/05_review_geo_small.md:176-180`)
  already flagged this as "low/cosmetic — could be promoted to top-level
  import."
- **Verdict:** STALE caution.
- **Suggested action:** DELETE the wrapper and promote the import to the
  module top of `boozer_residual_jax.py`.

### 4.2 Necessary cycle breakers (KEEP)

- `boozer_residual_jax.py:479-491` (`_get_surface_fns`) and
  `boozer_residual_jax.py:494-506` (`_get_surface_xyzfourier_fns`) defer
  to `simsopt.geo.surface_fourier_jax`, which through
  `jax_core/__init__.py` → `jax_core.surface_henneberg`,
  `jax_core.surface_fourier` re-enters `simsopt.geo.surface_fourier_jax`
  (mitigated symmetrically by the bottom-of-file import workaround at
  `surface_fourier_jax.py:2766-2768`).
- `jax_core/curve_geometry.py:163` (`from simsopt.geo.orientedcurve
  import centercurve_pure`) and `jax_core/curve_geometry.py:199`
  (`from simsopt.geo.curvexyzfouriersymmetries import
  jaxXYZFourierSymmetriescurve_pure`) — break the `simsopt.geo.curve` ↔
  `simsopt.jax_core` cycle documented in
  `MEMORY/project_curve_jax_core_import_cycle.md`.
- `jax_core/_math_utils.py:11` (`from simsopt.backend import
  maybe_initialize_distributed_jax`) is run-time only; `simsopt.backend`
  is the public facade that itself imports from `_math_utils` indirectly
  through other modules.

**Verdict:** all three retained shims are KEEP.

---

## 5. Dead fallback paths

### 5.1 Already known: `_traceable_solve_hessian_linearization` live-solver fallback

- **File:line:** `src/simsopt/geo/surfaceobjectives_jax.py:3076-3106`
- **Category:** Dead/latent fallback flagged as H-17 in
  `.artifacts/jax_convention_review_2026-05-16/00_SYNTHESIS.md:211-216`.
- **Snippet:**
  ```python
  def _traceable_solve_hessian_linearization(
      booz_jax, solved_x, rhs, coil_set_spec, objective_kwargs,
      *, linear_solve_factors, linear_solve_tol, linear_solve_stab, transpose,
  ):
      if linear_solve_factors is not None:
          return _traceable_solve_plu_linearization(
              linear_solve_factors, rhs, linear_solve_tol=linear_solve_tol,
              transpose=transpose,
          )
      # ↓ fallback only fires when factors is None; for the LS lane factors
      #   are always populated. This branch is currently unreachable.
      objective_fn = _make_boozer_penalty_objective_closure(...)
      return _optimizer_jax._solve_hessian_least_squares_system_with_status(...)
  ```
- **Verdict:** DEAD fallback (per H-17 — currently unreachable, but a
  correctness trap if a future change passes `factors=None` for the LS
  lane: this fallback would silently call the LIVE solver inside `jit`,
  violating the "traceable adjoint must NOT call the live solver inside
  jit" rule from CLAUDE.md `Adjoint / warm-start operator solves`).
- **Suggested action:** Per the H-17 recommendation, REPLACE the
  live-solver fallback with a NaN-emission branch (e.g.
  `_traceable_adjoint_gradient_or_nan`) so the failure mode is explicit
  rather than silently incorrect.

### 5.2 Static `baseline_linear_solve_factors = None` in
`_build_traceable_objective_state`

- **File:line:** `src/simsopt/geo/surfaceobjectives_jax.py:3936`
- **Category:** Latent-only, related to 5.1.
- **Snippet:**
  ```python
  linearization_kind = booz_jax.res["linearization_kind"]
  baseline_linear_solve_factors = None
  linear_solve_tol = booz_jax._linear_solve_tolerance()
  ```
- **Reason:** This `None` initial value flows into `solved_linear_solve_factors`
  passed to `_traceable_solve_hessian_linearization`. For LS lanes the
  factors are repopulated downstream (the LS forward path stores them
  under `lax.stop_gradient`); but combined with finding 5.1 this means
  the fallback IS reached if a future change does not repopulate before
  calling the adjoint. Fixing 5.1 resolves both.
- **Verdict:** UNCLEAR — defensive `None` init that may be load-bearing
  if downstream rewrites factors. Tied to fix for 5.1.
- **Suggested action:** Leave as-is until 5.1 is fixed; then DOCUMENT
  the contract that LS lanes must populate factors before invoking the
  linearization helper.

---

## 6. Alias-only "JAX port" symbols

Two known cases plus one not previously listed:

### 6.1 `B2EnergyJAX = B2Energy`

- **File:line:** `src/simsopt/field/force.py:1320`
- **Category:** Alias-only port (no JAX implementation).
- **Snippet:**
  ```python
  B2EnergyJAX = B2Energy
  ```
- **Verdict:** OBSOLETE — the name implies a JAX port that doesn't
  exist. Already noted as M-24 in the convention review.
- **Suggested action:** Either DELETE the alias (and update callers to
  use `B2Energy`) or RENAME to clarify it is not a JAX port. Adding a
  one-line docstring noting the equivalence would suffice if external
  callers depend on the name.

### 6.2 `LpCurveForceJAX = LpCurveForce`

- **File:line:** `src/simsopt/field/force.py:2284`
- **Category:** Alias-only port (no JAX implementation).
- **Snippet:**
  ```python
  LpCurveForceJAX = LpCurveForce
  ```
- **Verdict:** OBSOLETE — same pattern as 6.1.
- **Suggested action:** Same — DELETE or DOCUMENT.

### 6.3 `BiotSavartBPullback = BiotSavartFieldPullback`

Already covered in 3.3 (no external consumer). Same character as 6.1/6.2
but no `JAX` suffix in the alias name — listed here for completeness.

### 6.4 Non-issue: `FieldCollectiveConfig = CoilGroupCollectiveConfig`

- **File:line:** `src/simsopt/jax_core/sharding.py:72`
- **Verdict:** KEEP — used internally as a type-annotation alias
  (`sharding.py:382`). Cosmetic rename only.

### 6.5 Non-issue: `_surface_dmajor_radius_jax_from_dofs` and siblings

- **Files:** `src/simsopt/geo/surfaceobjectives_jax.py:574-576`
- **Verdict:** KEEP — private module aliases used as static methods on
  `_SurfaceScalarMetricJAX` subclasses (lines 783-784, 2529) plus a test
  helper. Cosmetic-only.

### 6.6 Non-issue: `_rotation_eval` / `_rotationdash_eval`

- **File:line:** `src/simsopt/geo/framedcurve.py:835-836`
- **Verdict:** KEEP — used at lines 448, 451.

---

## 7. Stale docstrings

### 7.1 `interpolated_field_jax._points_device` mirrors a stale module

- **File:line:** `src/simsopt/field/interpolated_field_jax.py:55-61`
- **Category:** Stale docstring reference.
- **Snippet:**
  ```python
  def _points_device(points: np.ndarray):
      """Stage host points to a JAX float64 device array via the strict-safe
      ``jax.device_put`` path. Mirrors the helper in
      :mod:`simsopt.field.magneticfieldclasses_jax`.
      """
      return _as_jax_float64(points)
  ```
- **Reason:** `simsopt.field.magneticfieldclasses_jax` is now a pure
  re-export shim (`magneticfieldclasses_jax.py:1-27` only re-exports
  `CircularCoilJAX, DommaschkJAX, MirrorModelJAX, PoloidalFieldJAX,
  ReimanJAX, ToroidalFieldJAX`). The actual `points_device` helper is
  the canonical `simsopt.field._jax_common.points_device` (line 11).
  The docstring points readers at a module that no longer hosts the
  helper.
- **Verdict:** STALE.
- **Suggested action:** Either DELETE the local `_points_device` (and
  import `points_device` from `_jax_common`, see §11) or UPDATE the
  docstring to reference `simsopt.field._jax_common`.

### 7.2 `optimizer_jax.py` "JAX 0.9.2" reference in module docstring

- **File:line:** `src/simsopt/geo/optimizer_jax.py:73-77`
- **Snippet:**
  ```text
  The private methods live in ``optimizer_jax_private/`` and intentionally mirror
  the JAX 0.9.2 optimizer semantics so the line-search and iteration behavior
  stay stable across this project. ...
  The reference source is the upstream
  ``jax-v0.9.2`` tag (``a659757d768587a81d095a9fab5f0c36f8beb218``).
  ```
- **Reason:** Per `CLAUDE.md` the local environment is JAX 0.10.0. The
  derivation tag is historically correct (the implementation was lifted
  from `jax-v0.9.2`), but the prose reads as "we target 0.9.2 semantics"
  which is now ambiguous against the 0.10.0 runtime.
- **Verdict:** STALE wording (the historical attribution is correct,
  but the sentence "intentionally mirror the JAX 0.9.2 optimizer
  semantics" is misleading at the 0.10.0 runtime).
- **Suggested action:** DOCUMENT — clarify "Derived from JAX 0.9.2
  (commit `a659757d…`); the runtime is JAX 0.10.0 and the implementation
  is now self-contained and does not pin to 0.9.2 behavior."

### 7.3 `surfaceobjectives_jax._traceable_objective_gradient_parts`
"JAX 0.9.2" inline comment

- **File:line:** `src/simsopt/geo/surfaceobjectives_jax.py:3741-3745`
- **Snippet:**
  ```python
      if not depends_on_coil_dofs:
          # Some diagnostic terms depend only on the solved inner state, so
          # their explicit coil derivative is exactly zero. Avoid autodiff on
          # these constant-in-coils scalars under strict transfer guard because
          # JAX 0.9.2 instantiates host scalar zeros for null tangent paths.
          direct_grad = _runtime_zeros_like(coil_dofs)
  ```
- **Verdict:** STALE — pins the rationale to 0.9.2, but the same
  behavior would have to be re-checked against 0.10.0. The mitigation
  (using `_runtime_zeros_like`) is still correct, so the code is fine;
  only the comment wording is stale.
- **Suggested action:** DOCUMENT — drop the explicit "0.9.2" version
  pin or add "(verified through 0.10.0)".

---

## 8. Deprecated JAX APIs

- `pmap`, `device_put_sharded`, `device_put_replicated`, `PmapSharding`,
  `register_pytree_node` (bare), and runtime `PRNGKey()` calls: **zero
  matches** across the in-scope set.
- The only `PRNGKey` hits are docstring references in
  `jax_core/sampling.py:6, 56, 94, 132` (the public sampling functions
  accept a `key` arg whose docstring still says "a ``jax.random.PRNGKey``").
  In JAX 0.10.0 the preferred constructor is `jax.random.key`, but
  `PRNGKey` is still functional (warning-free) and the docstring does
  not pin behavior to the deprecated constructor.

**Verdict:** clean — no action required. (Optional: refresh the
`PRNGKey` mentions in `jax_core/sampling.py` docstrings to point at the
typed-key API for forward compatibility.)

---

## 9. Backward-compat shims without forward-compat partners

### 9.1 `simsopt.backend.os` re-export

- **File:line:** `src/simsopt/backend.py:102-104`
- **Category:** Stale facade export.
- **Snippet:**
  ```python
  __all__ = (
      ...,
      "warn_if_jax_fallback",
      # Keep ``os`` available on the facade for existing tests/helpers.
      "os",
  )
  ```
- **Evidence:** `grep -rn 'from simsopt.backend import.*os' src/ tests/
  benchmarks/ examples/` returns no matches; no callsite uses
  `simsopt.backend.os` either. The "existing tests/helpers" rationale
  no longer applies.
- **Verdict:** OBSOLETE.
- **Suggested action:** DELETE the `os` entry from `__all__` and the
  unused `import os` at `backend.py:3`.

### 9.2 Legacy env-var pair `STAGE2_BACKEND` / `SIMSOPT_JAX_BACKEND`

- **File:line:** `src/simsopt/backend/runtime.py:36, 38`
- **Category:** Legacy env-var compatibility (not a stale shim).
- **Verdict:** KEEP — `runtime.py:96-100,1217-1225` actively translate
  legacy env vars into the SSOT `SIMSOPT_BACKEND_MODE` resolution, and
  `CLAUDE.md` documents both pairs as supported.

---

## 10. Module-level `sys.path.insert`

### 10.1 `biotsavart_jax.py` direct-path loader

- **File:line:** `src/simsopt/field/biotsavart_jax.py:11-17`
- **Category:** Module-level sys.path manipulation.
- **Snippet:**
  ```python
  def _ensure_src_root_on_path() -> None:
      src_root = str(Path(__file__).resolve().parents[2])
      if src_root not in sys.path:
          sys.path.insert(0, src_root)


  _ensure_src_root_on_path()
  ```
- **Verdict:** STALE (per NIT-2 in the convention review). The shim's
  docstring (`biotsavart_jax.py:1-5`) says "preserves the historical
  import and direct-path loader contract". With editable install
  active, this is no longer necessary; with the `tests/integration/conftest.py`
  meta-path-finder patch (per CLAUDE.md), no consumer relies on the
  manual `sys.path.insert`.
- **Suggested action:** DELETE the helper and the call site. The
  `from simsopt.jax_core.biotsavart import (...)` line just below will
  resolve correctly under any reasonable install mode.

(Sweep `grep -rn 'sys.path' src/simsopt/` confirms this is the **only**
in-scope occurrence — no similar siblings to clean up.)

---

## 11. DRY violations across modules

### 11.1 `_points_device` duplicated across three field modules

- Canonical: `src/simsopt/field/_jax_common.py:11` — `points_device(points)`.
- Imported and aliased by:
  - `circular_coil_jax.py:15`
  - `dommaschk_jax.py:20`
  - `mirror_model_jax.py:14`
  - `poloidal_field_jax.py:14`
  - `reiman_jax.py:15`
  - `toroidal_field_jax.py:17`
- **Duplicate definitions:**
  - `src/simsopt/field/interpolated_field_jax.py:55-61` — local
    `_points_device(points)` with identical body.
  - `src/simsopt/field/dipole_field_jax.py:56-64` — local
    `_points_device(points)` with identical body.
- **Snippet (representative):**
  ```python
  def _points_device(points: np.ndarray):
      """Stage host points to a JAX float64 device array..."""
      return _as_jax_float64(points)
  ```
- **Verdict:** DRY violation. The convention review reported
  "duplicated in 4 modules"; the count above is 2 module-level
  duplicates plus 1 unrelated method (`scalar_potential_rz_jax.py:33`
  is a `self._points_device(self)` instance method, not a duplicate)
  plus 1 instance-attribute name reuse
  (`wireframefield_jax.py:67, 73; dipole_field_jax.py:251, 256, …`).
- **Suggested action:** DELETE both local function definitions and
  replace with `from ._jax_common import points_device as _points_device`
  (matching the six modules already using the canonical helper).
  Updating the stale docstring (§7.1) falls out of the same change.

(No other DRY duplicates surfaced in the unused-symbol sweep across
`jax_core/` and `field/`, `geo/`, `objectives/`, `solve/` `*_jax*.py`.)

---

## Prioritized cleanup list

Ordered by risk/benefit. None of these are PR blockers; all are
maintenance cleanups.

| # | Item | Action | Files | Risk |
|---|---|---|---|---|
| 1 | `_traceable_solve_hessian_linearization` live-solver fallback (H-17) | Replace fallback with NaN-emission | `geo/surfaceobjectives_jax.py:3088-3106` | Medium — preserves correctness contract |
| 2 | `_points_device` DRY violation in 2 field modules | Import `points_device` from `_jax_common` | `field/interpolated_field_jax.py:55-61`, `field/dipole_field_jax.py:56-64` | Low |
| 3 | `_get_grouped_biot_savart` over-cautious lazy import | Promote to module-top import | `geo/boozer_residual_jax.py:509-513, 626, 854` | Low |
| 4 | `B2EnergyJAX = B2Energy`, `LpCurveForceJAX = LpCurveForce` alias-only ports | Delete aliases or add explicit "not a JAX port" docstring | `field/force.py:1320, 2284` | Low (API surface) |
| 5 | Unused `BiotSavartBPullback` alias | Delete alias and `__all__` entry | `field/biotsavart_jax_backend.py:81, 265` | Low |
| 6 | Unused `scalar_at_axis0` | Delete | `jax_core/_math_utils.py:76-79` | Low |
| 7 | Unused `optimizable_full_dofs_from_map_spec` | Delete | `jax_core/curve_geometry.py:281-285` | Low |
| 8 | Stale `sys.path.insert` shim | Delete `_ensure_src_root_on_path()` and its invocation | `field/biotsavart_jax.py:11-17` | Low |
| 9 | Stale `simsopt.backend.os` facade export | Drop from `__all__` and remove `import os` | `backend.py:3, 102-104` | Low |
| 10 | Stale "JAX 0.9.2" comment in `surfaceobjectives_jax.py:3745` | Update or drop the version pin | `geo/surfaceobjectives_jax.py:3741-3745` | Trivial (comment only) |
| 11 | Ambiguous "JAX 0.9.2 semantics" in `optimizer_jax.py` module docstring | Clarify that the runtime is JAX 0.10.0 | `geo/optimizer_jax.py:73-77` | Trivial (comment only) |
| 12 | Stale docstring "Mirrors the helper in `magneticfieldclasses_jax`" | Update reference or remove with §11 fix | `field/interpolated_field_jax.py:55-61` | Trivial |

Items 6, 7, 8, 9 can be batched as a single "delete unused symbols"
commit. Items 4, 5 can be batched as "remove alias-only public ports".
Item 1 should travel with the H-17 fix landed by the parity-audit team.
Items 10, 11, 12 are comment-only fixups.
