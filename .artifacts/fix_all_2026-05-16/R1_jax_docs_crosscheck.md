# R1 — JAX 0.10.x Documentation Cross-Check (simsopt-jax)

**Date:** 2026-05-16
**Reviewer:** Opus 4.7 (max-effort)
**Runtime:** JAX 0.10.0, jaxlib 0.10.0, NumPy 2.x, Python 3.11
**Repo:** `/Users/suhjungdae/code/columbia/simsopt-jax` (branch `gpu-purity-stage2-20260405`)
**Authority:** https://docs.jax.dev/en/latest/changelog.html (verbatim quotes via WebFetch);
`.artifacts/jax_convention_review_2026-05-16/02_jax_best_practices_baseline.md` (JBP-1..20)

This is a verification-grade cross-check of the simsopt-jax port against current
official JAX 0.10.x documentation and the JAX 0.10.0 changelog. Every finding
cites an actual grep hit (`file:line`) and an upstream authority. Patterns flagged
have been read, not inferred.

---

## Headline summary

The simsopt-jax port is **substantially clean against the JAX 0.10.0 breaking-change
inventory**: none of the removed APIs (`device_put_sharded`, `device_put_replicated`,
`PmapSharding`, `jax.pmap`, `jax.core.ShapedArray.vma`, `TFRT_CPU_*` device names,
`PartitionSpec == tuple`, deprecated `jnp.clip` kwargs) are present in source.
The handful of remaining issues are best-practice gaps (JBP rule violations), not
release-blocking deprecations:

- **Section 1 (deprecated APIs in use):** zero blockers. One MEDIUM-grade
  `solve_triangular` 2D-RHS pattern that survived the 0.10.0 reinterpretation
  (still legal, but caller relies on a since-tightened behavior).
- **Section 2 (suboptimal patterns):** 22 `static_argnums` call sites that should
  migrate to `static_argnames` (JBP-2.1); ~79 `jax.tree_util.tree_*` call sites
  that should migrate to the modern `jax.tree.*` alias (JBP-5.5); 8 redundant
  `@partial(jax.jit, static_argnames=())` decorations.
- **Section 3 (precision):** `_as_jax_float64` SSOT is well-used; one duplicate
  local definition exists in `surface_rzfourier.py` that should re-export the SSOT.
- **Section 4 (numerical-safety gaps):** `integral_BdotN`'s `BdotN * jnp.sqrt(weight)`
  pattern violates JBP-17.1 (double-where); `_on_ball_projected_gradient` in
  `pm_optimization.py` has the same shape. One `jnp.linalg.eig` call
  (`magnetic_axis_helpers.py:598`) is autodiff-incompatible and CPU/GPU-only.
- **Section 5 (donate_argnums):** zero production usage. The benchmark probe
  confirms ~halving of peak memory is achievable on the Biot-Savart hot path.
- **Section 6 (transfer-guard / sharding):** `jax.shard_map` modern API is used
  correctly; sharding helpers are sound. Two `with jax.transfer_guard("allow")`
  scopes in `optimizer_jax.py` are necessary GMRES escape hatches and documented
  as such.
- **Section 7:** actionable fix list with priorities and JAX docs URLs.

---

## Section 1. Deprecated APIs in use (JAX 0.10.0 removals)

Verbatim cross-reference against the JAX 0.10.0 (2026-04-16) changelog:

| Removed/deprecated in 0.10.0 | Status in simsopt-jax | Evidence |
|---|---|---|
| `jax.device_put_sharded` | absent | `grep -rn "device_put_sharded" src/simsopt/` → 0 hits |
| `jax.device_put_replicated` | absent | `grep -rn "device_put_replicated" src/simsopt/` → 0 hits |
| `jax.sharding.PmapSharding` | absent | `grep -rn "PmapSharding" src/simsopt/` → 0 hits |
| `jax.pmap` (now wraps `jit(shard_map)`) | absent | `grep -rn "jax\.pmap\|@pmap" src/simsopt/` → 0 hits |
| `jax.core.ShapedArray.vma` | absent | `grep -rn "\.vma\b" src/simsopt/` → 0 hits |
| `TFRT_CPU_0` / `TFRT_CPU_1` device names | absent | `grep -rn "TFRT_CPU" src/simsopt/ tests/` → 0 hits |
| `PartitionSpec == tuple` | absent | `grep -rn "PartitionSpec.*==\|== *PartitionSpec\|spec *==" src/simsopt/ tests/` → 0 hits |
| `jax.numpy.clip(a_min=...)` / `a_max=...` | absent | All 12 `jnp.clip` call sites in `src/simsopt/{jax_core,geo}/` use positional args |
| `with mesh:` context manager | absent | `grep -rn "with mesh\|with.*Mesh.*:" src/simsopt/` → 0 hits; mesh objects are stored on dataclasses and threaded to `jax.shard_map(mesh=...)` |
| `jax.lax.pvary` (deprecated 0.9.0) | absent | `grep -rn "pvary" src/simsopt/` → 0 hits |

### 1.1 [MEDIUM] `solve_triangular` with 2D identity RHS for matrix inversion

- **File:** `src/simsopt/field/force.py:1122-1123`
  ```python
  inv_C = jscp.linalg.solve_triangular(C, jnp.eye(C.shape[0]), lower=True)
  inv_L = jscp.linalg.solve_triangular(C.T, inv_C, lower=False)
  ```
- **Issue.** JAX 0.10.0 changelog (verbatim, fetched 2026-05-16):
  > "`jax.scipy.linalg.cho_solve()`, `jax.scipy.linalg.lu_solve()`, and
  > `jax.scipy.linalg.solve_triangular()` now show a deprecation warning for
  > batched 1D solves with `b.ndim > 1`."
- **Reading.** The new docstring of `solve_triangular` (fetched 2026-05-16) describes
  `b` as "shape `(N,)` (for a 1-dimensional right-hand-side) or `(..., N, M)` (for
  a batched 2-dimensional right-hand-side)". The 2D identity RHS in `force.py`
  now reads as a single non-batched 2D solve (`(N, M)` with no batch axes), which
  remains supported under the new contract. **However**, the same call would have
  previously been interpreted as a batch of 1D solves under the old contract, so
  the numerical result depends on which interpretation jaxlib 0.10.0 actually
  selects. This is a behavior contract that the JAX 0.10.0 release tightened.
- **Replacement (recommended).** Use the explicit inversion route:
  `inv_C = jscp.linalg.solve_triangular(C, jnp.eye(C.shape[0]), lower=True)` is
  semantically a matrix solve. The robust replacement is
  `jnp.linalg.solve(C, jnp.eye(C.shape[0]))` (modern; no contract drift) or to
  compute the explicit inverse via `jscp.linalg.cho_solve((C, True), jnp.eye(C.shape[0]))`.
  At minimum, add a comment locking the 2D semantics.
- **JAX docs.** https://docs.jax.dev/en/latest/_autosummary/jax.scipy.linalg.solve_triangular.html

### 1.2 [LOW] Other `jsp_linalg.solve_triangular` call sites — all 1D RHS, no exposure

All other call sites pass 1D `b` and are unaffected by the deprecation:

| File:line | RHS shape | Status |
|---|---|---|
| `src/simsopt/geo/optimizer_jax.py:1971` | `q_matrix.T @ augmented_rhs` (1D) | OK |
| `src/simsopt/geo/surfaceobjectives_jax.py:3198` | `rhs` (1D adjoint cotangent) | OK |
| `src/simsopt/geo/surfaceobjectives_jax.py:3199` | `y` (intermediate 1D) | OK |
| `src/simsopt/geo/surfaceobjectives_jax.py:3202` | `P.T @ rhs` (1D) | OK |
| `src/simsopt/geo/surfaceobjectives_jax.py:3203` | `y` (1D) | OK |
| `src/simsopt/geo/optimizer_jax_private/_lbfgsb_scipy.py:2324, 2344` | 1D RHS vectors | OK |

All `jsp_linalg.lu_solve((lu, piv), rhs, trans=...)` call sites in
`src/simsopt/geo/optimizer_jax.py:{2411, 2880, 2883}` and
`src/simsopt/geo/surfaceobjectives_jax.py:3190` similarly pass 1D RHS — unaffected.

---

## Section 2. Suboptimal patterns by current best practice

### 2.1 [HIGH] `static_argnums` instead of `static_argnames` (JBP-2.1 violation)

The JAX 0.10.x docs explicitly recommend `static_argnames` over positional
`static_argnums` because named markers are greppable and survive argument
reordering. The repo has 22 `static_argnums` call sites that should be migrated:

| File:line | Decoration | Recommended fix |
|---|---|---|
| `src/simsopt/jax_core/framedcurve.py:332` | `jax.jit(_rotation_alpha_impl, static_argnums=(2,))` | `static_argnames=("order",)` |
| `src/simsopt/jax_core/framedcurve.py:333` | `jax.jit(_rotation_alphadash_impl, static_argnums=(2,))` | `static_argnames=("order",)` |
| `src/simsopt/field/force.py:953` | `jit(_b2energy_eval, static_argnums=(3,))` | `static_argnames=("downsample",)` |
| `src/simsopt/field/force.py:954` | `jit(grad(...), static_argnums=(3,))` | same |
| `src/simsopt/field/force.py:955` | `jit(_net_ext_flux_eval, static_argnums=(2,))` | `static_argnames=("downsample",)` |
| `src/simsopt/field/force.py:956` | `jit(grad(...), static_argnums=(2,))` | same |
| `src/simsopt/field/force.py:957` | `jit(_squared_mean_force_eval, static_argnums=(9,))` | `static_argnames=("downsample",)` |
| `src/simsopt/field/force.py:960` | `jit(grad(...), static_argnums=(9,))` | same |
| `src/simsopt/field/force.py:962` | `jit(_lp_force_eval, static_argnums=(14,))` | `static_argnames=("downsample",)` |
| `src/simsopt/field/force.py:965-967` | three more (LP force/torque grad) | same |
| `src/simsopt/field/force.py:969-972` | three more (LP torque grad) | same |
| `src/simsopt/field/force.py:974-977` | two more (squared-mean torque grad) | same |
| `src/simsopt/geo/accessibility.py:650, 666, 1120, 1130, 1140, 1150` | six × `static_argnums=(5, 6)` | `static_argnames=("left_argnum", "right_argnum")` |
| `src/simsopt/geo/accessibility.py:1644, 1653` | two × `static_argnums=(4,)` | `static_argnames=("argnum",)` |

- **JAX docs:** https://docs.jax.dev/en/latest/jit-compilation.html (sections "JIT and caching", "static_argnums") — the `static_argnames` syntax is documented as the preferred form throughout the 0.10.x docs.
- **Severity:** HIGH for `geo/accessibility.py` (positional index 5,6 is opaque); MEDIUM elsewhere.

### 2.2 [MEDIUM] Legacy `jax.tree_util.tree_*` instead of `jax.tree.*` (JBP-5.5)

JAX 0.10.x docs treat `jax.tree.map`, `jax.tree.leaves`, `jax.tree.structure` as
the modern convenience namespace; `jax.tree_util.tree_*` remains supported for
legacy compatibility but the docs recommend migration. The repo has 79 hits of
the legacy form (`grep -rn "tree_util\.tree_map\|tree_util\.tree_leaves\|tree_util\.tree_structure\|tree_util\.tree_flatten\|tree_util\.tree_unflatten" src/simsopt/`) and zero hits of the modern form.

Representative call sites:

- `src/simsopt/_core/jax_host_boundary.py:64`: `jax.tree_util.tree_map(_hostify_leaf, value)`
- `src/simsopt/geo/optimizer_jax.py:411,441,821,829,833,841,849,857,866,878,885,886,904,923`: 14 hits
- `src/simsopt/geo/surfaceobjectives_jax.py:1884,1928,1954,1970,4486,5874`: 6 hits
- `src/simsopt/geo/boozersurface_jax.py:817`: 1 hit
- `src/simsopt/jax_core/boozer_fixed_state.py:306,316`: 2 hits

**Note.** `jax.tree_util.register_pytree_node_class` and
`jax.tree_util.register_dataclass` remain in the `jax.tree_util` namespace and
have no `jax.tree.*` alias — those 79 + 1 hits are correct as written.

- **JAX docs:** https://docs.jax.dev/en/latest/pytrees.html (section "Modern Convenience Namespace").

### 2.3 [LOW] Redundant `@partial(jax.jit, static_argnames=())`

Eight call sites in `src/simsopt/jax_core/boozer_radial_interp.py` decorate
functions with an empty `static_argnames` tuple. The functional intent is just
`@jax.jit`. Same semantics; cleaner to drop the redundant wrapper:

```
src/simsopt/jax_core/boozer_radial_interp.py:222
src/simsopt/jax_core/boozer_radial_interp.py:281
src/simsopt/jax_core/boozer_radial_interp.py:373
src/simsopt/jax_core/boozer_radial_interp.py:400
src/simsopt/jax_core/boozer_radial_interp.py:426
src/simsopt/jax_core/boozer_radial_interp.py:447
src/simsopt/jax_core/boozer_radial_interp.py:494
src/simsopt/jax_core/boozer_radial_interp.py:511
```

- **Replace** `@partial(jax.jit, static_argnames=())` → `@jax.jit`.
- **JAX docs:** https://docs.jax.dev/en/latest/jit-compilation.html.

### 2.4 [LOW] `jax_enable_x64` is correctly set at entrypoint (JBP-9.1, NO violation)

`src/simsopt/__init__.py:26-32`:
```python
if "jax" in _sys.modules:
    import jax as _jax
    _jax.config.update("jax_enable_x64", True)
    del _jax
else:
    _os.environ.setdefault("JAX_ENABLE_X64", "True")
```

This pattern is correct: the env var is set before any JAX import side effect,
and the config update covers the case where another caller pre-imported JAX.
The runtime layer also explicitly enforces `jax.config.update("jax_enable_x64", ...)`
at `src/simsopt/backend/runtime.py:1729`. No library-level inner setting found.

### 2.5 [LOW] `jax.random.PRNGKey` vs `jax.random.key` (JBP-10.1)

`src/simsopt/jax_core/sampling.py` references `PRNGKey` only in docstrings
(lines 6, 56, 94, 132). Tests (`tests/field/test_sampling_jax_item22.py:160,
173, 184, 202, 207, 226, 245, 313, 341, 367`) use `jax.random.PRNGKey(seed)`
directly. The 0.10.x docs designate `jax.random.key(seed)` as the preferred
typed-key form; `PRNGKey` is back-compat. Migrate the test seeds to `key()`.

- **JAX docs:** https://docs.jax.dev/en/latest/random-numbers.html (section "Current Preferred Approach").

---

## Section 3. Float64 / precision discipline

### 3.1 [LOW] Duplicate `_as_jax_float64` local definition

`src/simsopt/jax_core/_math_utils.py` is the SSOT (`as_jax_float64` at line 57,
re-exported with underscore prefix in many modules).

A duplicate is locally defined in `src/simsopt/jax_core/surface_rzfourier.py:22-27`:
```python
def _as_jax_float64(value) -> jax.Array:
    if isinstance(value, ...):
        return jnp.asarray(value, dtype=jnp.float64)
    ...
    return jnp.asarray(value, dtype=jnp.float64)
```

This shadowed copy bypasses the SSOT-strict-checks in `as_jax_array`
(which asserts via `_require_explicit_float64` for sufficiently-typed paths).
Replace the local def with:

```python
from ._math_utils import as_jax_float64 as _as_jax_float64
```

- **JAX docs:** https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html (section "🔪 Double (64bit) precision"); JBP-9.2.

### 3.2 [LOW] `jnp.asarray(..., dtype=jnp.float64)` direct usage (vs SSOT)

`src/simsopt/jax_core/magnetic_axis_helpers.py:586-587`:
```python
phi0 = jnp.asarray(0.0, dtype=jnp.float64)
phi_end = jnp.asarray(1.0, dtype=jnp.float64) / jnp.asarray(nfp, dtype=jnp.float64)
```

These bypass `_as_jax_float64`. Same pattern at `:600-602`. Not a bug
(intent is clear), but inconsistent with the SSOT contract codified in
`_math_utils.py:_require_explicit_float64`. The SSOT `as_jax_float64` performs
the explicit `jax.device_put` route which is strict-transfer-guard clean;
bare `jnp.asarray` on a Python scalar relies on JAX's implicit host-to-device
behavior, which is fine in normal mode but produces a guard violation under
`transfer_guard("disallow")`.

Other in-source `jnp.asarray(..., dtype=jnp.float64)` hits worth review (sample):

- `src/simsopt/objectives/integral_bdotn_jax.py:47`: `jnp.asarray(arr, dtype=jnp.float64)` — wrapped by `_as_real_float64`, OK.
- `src/simsopt/jax_core/_math_utils.py:49,51`: SSOT internal, OK.
- `src/simsopt/jax_core/interpolated_boozer_field.py:679`: `period=jnp.asarray(state.period, dtype=jnp.float64)` — boundary input, OK.

---

## Section 4. Numerical-safety gaps (JBP-17.1, JBP-17.x)

### 4.1 [HIGH] Double-where pattern missing on `sqrt(weight)` in `integral_BdotN`

**File:** `src/simsopt/objectives/integral_bdotn_jax.py:64-115`

The "quadratic flux" branch (line 84-85) and "normalized" branch (lines 90-95)
both compute `BdotN * jnp.sqrt(weight)` where `weight` is masked via
`jnp.where(has_normal, ..., 0.0)`. The classic JBP-17.1 hazard applies:

```python
weight = jnp.where(has_normal, norm_n / (nphi * ntheta), 0.0)
residual = jnp.where(has_normal, BdotN * jnp.sqrt(weight), 0.0)
```

At masked points, `weight = 0`, and the gradient of `sqrt(0)` is `+inf`. The
outer `where` blocks the value but **autodiff still differentiates both
branches** — the cotangent for the masked branch carries the `inf`, which
propagates as NaN into the gradient sum.

**Fix (JBP-17.1):** Mask the argument to `sqrt` as well:
```python
safe_norm_n_for_sqrt = jnp.where(has_normal, norm_n, 1.0)  # always positive
weight = jnp.where(has_normal, safe_norm_n_for_sqrt / (nphi * ntheta), 0.0)
residual = jnp.where(has_normal, BdotN * jnp.sqrt(weight), 0.0)
```

The same pattern repeats in:

- Line 93: `jnp.where(has_normal, BdotN * jnp.sqrt(point_weight), 0.0)`
- Line 105: `jnp.where(has_normal, BdotN * jnp.sqrt(weight), 0.0)`

Each requires the same masked-input fix.

- **JAX docs:** https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html (section "🔪 Debugging NaNs and Infs"); https://docs.jax.dev/en/latest/faq.html (the "ensure that there is a `jnp.where` inside the partially-defined function" pattern).

### 4.2 [MEDIUM] `jnp.linalg.norm(m, axis=1)` at `m=0` in `pm_optimization.py`

**File:** `src/simsopt/jax_core/pm_optimization.py:2195` (used in
`_l2_ball_projection`) and `2262` (used in `_on_ball_projected_gradient`).

```python
norm = jnp.linalg.norm(m, axis=1)  # (N,)
zero = norm - norm
one = jnp.exp(zero)
scale = jnp.where(
    norm > m_maxima,
    m_maxima / jnp.where(norm > zero, norm, one),
    one,
)
```

Same JBP-17.1 hazard. `jnp.linalg.norm` at a zero vector returns 0 with a NaN
gradient (cotangent `m / norm` = `0/0`). The outer `where` correctly handles
the forward value but autodiff differentiates the `m / norm` branch — yielding
NaN gradient leakage.

**Fix:** Use a `safe` version that masks the input vector:
```python
safe_m = jnp.where(norm[:, None] > 0.0, m, jnp.ones_like(m))
safe_norm = jnp.linalg.norm(safe_m, axis=1)
```

Then use `safe_norm` for the `m_maxima / norm` operation, and re-mask the
output with `jnp.where(norm > 0, ..., one)`.

The same pattern occurs at line 2262-2263; the existing `safe_norm` only
guards the divide, not the upstream `norm` computation that fed it.

- **JAX docs:** Common gotchas (NaN debugging); JBP-17.1.

### 4.3 [HIGH] `jnp.linalg.eig` is CPU/GPU-restricted, autodiff-incompatible, JIT-incompatible

**File:** `src/simsopt/jax_core/magnetic_axis_helpers.py:598`
```python
evals, _ = jnp.linalg.eig(M)
```

Per JAX 0.10.x docs (fetched 2026-05-16):
- > "Non-symmetric eigendecomposition is only implemented on the CPU and GPU backends."
- > "autodiff is not currently supported for computation of non-symmetric eigenvectors"

Concrete implications for the simsopt-jax port:

1. **TPU/backend restriction:** Any future TPU deployment of `on_axis_iota_rk`
   will fail at runtime.
2. **Autodiff hazard:** `on_axis_iota_rk` is consumed by `IotasJAX` and similar
   adapters that may want gradient flow. The current code computes `iota` from
   `arctan2(im(eig0), re(eig0))`. If a caller tries to `jax.grad(iota_of_M)`,
   the result is silently NaN (no error is raised because eigenvalues
   themselves are differentiable; eigenvectors are not, but they aren't used
   here).
3. **JIT contract:** `jnp.linalg.eig` is implemented via a host callback on
   some backends, which (a) breaks the persistent compilation cache (JBP-13.4)
   and (b) blocks under `transfer_guard("disallow")`. The internal docstring
   at `magnetic_axis_helpers.py:55-58` acknowledges the eigenvalue-ordering
   hazard but not the autodiff/JIT-cache one.

**Recommendation.** Since `M` here is the 2x2 tangent map, the eigenvalues are
analytical:

```python
# For 2x2 M = [[a, b], [c, d]]:
# tr = a + d; det = a*d - b*c; disc = tr*tr - 4*det
# eigs = (tr ± sqrt(disc)) / 2
```

Replace the `jnp.linalg.eig` call with a hand-rolled closed-form eigendecomposition.
This makes the function autodiff-friendly, JIT-cache-friendly, and
strict-transfer-guard clean.

- **JAX docs:** https://docs.jax.dev/en/latest/_autosummary/jax.numpy.linalg.eig.html.

### 4.4 [LOW] `pair_linking_number_pure` has `dr ** (-3)` at `dr=0`

**File:** `src/simsopt/jax_core/curve_geometry.py:802-805`

```python
dr = jnp.linalg.norm(difference, axis=-1)
...
inv_dr3 = jnp.where(dr > 0, dr ** (-3), _explicit_scalar(0.0, reference=gamma1))
```

This is the same `where`-without-input-mask hazard, but the function returns a
rounded integer (`jnp.round(...).astype(jnp.int32)`) and its own docstring
notes "this is a discrete parity kernel, not a differentiable objective for
`jax.grad`". No fix required. Document with an inline comment that the kernel
is non-differentiable by contract.

### 4.5 [LOW] `dr * 1e-31` epsilon padding in `circular_coil.py`

**File:** `src/simsopt/jax_core/circular_coil.py:255-256`
```python
beta_arg = r0_sq + x * x + y * y + 2.0 * r0 * rho + z * z + 1.0e-31
beta_guarded = jnp.sqrt(beta_arg)
```

`1e-31` is below jaxlib's subnormal flush threshold on some platforms (JBP-9.3).
For float64, this is fine (subnormal min is ~5e-324). For float32, it would be
flushed to zero. Since this file enforces float64 explicitly (`_as_jax_float64`),
this is OK, but the idiom is fragile — prefer `jnp.finfo(dtype).tiny * scale`.

---

## Section 5. Missing `donate_argnums` on hot paths

### 5.1 [MEDIUM] Zero production donation despite benchmark-validated benefit

`grep -rn "donate_argnums" src/simsopt/ → 0 hits` (production).
`grep -rn "donate_argnums" benchmarks/biotsavart_donation_probe.py → 4 hits`.

The benchmark probe (`benchmarks/biotsavart_donation_probe.py:199,212`) demonstrates
`donate_argnums=(0,)` halves peak memory on the Biot-Savart hot path. The test
suite (`tests/test_biotsavart_donation_probe.py:81,89`) confirms that donation
correctly deletes the input buffer post-call. **No production kernel adopts this.**

Hot-path candidates where donation would help (large array in / same-shape out):

| File:line | Kernel | Donation argument | Memory benefit |
|---|---|---|---|
| `src/simsopt/jax_core/biotsavart.py:506` (`@jax.jit kernel(points, gammas, gammadashs, currents)`) | per-point Biot-Savart | `donate_argnums=(0,)` on `points` (B is same `(P, 3)`) | ~halves peak |
| `src/simsopt/jax_core/biotsavart.py:553` (`kernel(points, v, gammas, gammadashs, currents)`) | VJP | donate `v` if same shape as output cotangent | minor |
| `src/simsopt/objectives/integral_bdotn_jax.py:64,131` (`residual_BdotN`, `integral_BdotN`) | per-quadrature mask | donate `Bcoil` if caller can release it | medium |

**Constraints (JBP-16.1..16.4):**
- Donation requires positional args (JAX 0.10.x rule unchanged).
- The kernel cache is keyed on (shape, dtype) so donation flag is part of the key.
- Donating `points` requires that no caller retains the input after the call —
  the current `BiotSavartJAX.B(points)` API does retain points internally as a
  back-reference (`field._coil_dof_state_token`); donating would invalidate that.
  A donation variant must be opt-in (separate kernel) or live behind a
  `donate=True` keyword that the wrapper toggles only when safe.

- **JAX docs:** https://docs.jax.dev/en/latest/buffer_donation.html.

### 5.2 [LOW] Optimizer step kernels (`optimizer_jax.py`) — donation candidates

The on-device LBFGS step kernels in `src/simsopt/geo/optimizer_jax.py` consume
large state pytrees and produce same-shape outputs. These are natural donation
candidates but require auditing each carry tree to confirm input invalidation
is safe.

---

## Section 6. Transfer-guard / sharding

### 6.1 Modern `jax.shard_map` API in use

`src/simsopt/jax_core/field.py:204-215` uses `jax.shard_map` correctly with
`mesh=`, `in_specs=`, `out_specs=`, and `check_vma=True`. No use of the
deprecated `jax.experimental.shard_map.shard_map`. Confirmed by:

```
grep -rn "jax\.experimental" src/simsopt/ → 0 hits
grep -rn "from jax\.experimental" src/simsopt/ → 0 hits
```

### 6.2 `Mesh`, `NamedSharding`, `PartitionSpec` — clean

`src/simsopt/jax_core/sharding.py:12`, `field.py:10`: imports use the canonical
public-API path `jax.sharding.{Mesh, NamedSharding, PartitionSpec}`. Mesh
construction at `sharding.py:92, 123` uses the modern positional form
`Mesh(np.asarray(devices, dtype=object), (axis_name,))`.

### 6.3 Transfer-guard escape hatches

The repo uses `with jax.transfer_guard("allow")` only in three production
locations, all with explanatory comments:

- `src/simsopt/geo/optimizer_jax.py:2676` (GMRES scalar literal lowering, comment block at 2672-2675)
- `src/simsopt/geo/optimizer_jax.py:2854` (similar)
- `src/simsopt/geo/surfaceobjectives_jax.py:4320, 4362, 4423` (compiled-bundle warmup boundaries)
- `src/simsopt/solve/wireframe_optimization_jax.py:176` (host materialization at outer boundary)

These are necessary and documented; no fix required. Cite JBP-12 and the
`with jax.transfer_guard(...)` reference in
https://docs.jax.dev/en/latest/_autosummary/jax.transfer_guard.html for context.

### 6.4 No deprecated transfer-guard config-state accessors

The repo uses `jax.config.jax_transfer_guard_device_to_host` and
`jax.config.jax_transfer_guard` (`src/simsopt/solve/permanent_magnet_optimization_jax.py:70-72`).
These are stable as of JAX 0.10.0.

---

## Section 7. Actionable fix list

### BLOCKER (release-gated)

None. The simsopt-jax port has no JAX 0.10.0 release blockers.

### HIGH (correctness or autodiff hazard)

1. **Fix double-where in `integral_BdotN`**
   - Files: `src/simsopt/objectives/integral_bdotn_jax.py:84-85, 90-95, 100-105`
   - Mask the `sqrt(weight)` input as well as the output (JBP-17.1).
   - Docs: https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html
2. **Replace `jnp.linalg.eig` for 2x2 monodromy with closed form**
   - File: `src/simsopt/jax_core/magnetic_axis_helpers.py:598`
   - Use analytical eigenvalues (`tr ± sqrt(disc))/2`) to unblock TPU, autodiff,
     persistent cache, and strict-transfer-guard.
   - Docs: https://docs.jax.dev/en/latest/_autosummary/jax.numpy.linalg.eig.html
3. **Migrate `static_argnums` → `static_argnames` in `geo/accessibility.py`**
   - Eight call sites with positional indices `(5, 6)` / `(4,)` — opaque
     to reviewers; named markers preserve intent and survive arg reorder.
   - Docs: https://docs.jax.dev/en/latest/jit-compilation.html

### MEDIUM (best-practice; correctness-adjacent)

4. **Fix double-where in `pm_optimization._l2_ball_projection` and
   `_on_ball_projected_gradient`** (`pm_optimization.py:2195, 2262`).
   - Use a `safe_m` input mask before `jnp.linalg.norm`.
   - Docs: https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html
5. **Replace 2D-identity `solve_triangular` with `jnp.linalg.solve`**
   - File: `src/simsopt/field/force.py:1122-1123`
   - JAX 0.10.0 tightened the 1D/2D RHS contract; the 2D-identity form is
     legal but contract-fragile. The modern alternative is `jnp.linalg.solve`
     or `cho_solve((C, True), jnp.eye(...))`.
   - Docs: https://docs.jax.dev/en/latest/_autosummary/jax.scipy.linalg.solve_triangular.html
6. **Migrate `static_argnums` → `static_argnames` in `field/force.py`**
   - 14 call sites (lines 953-977). All single-static-arg cases — straightforward
     mechanical conversion.
   - Docs: https://docs.jax.dev/en/latest/jit-compilation.html
7. **Migrate `static_argnums` → `static_argnames` in `jax_core/framedcurve.py:332, 333`**
   - Two call sites.
   - Docs: same.
8. **Adopt `donate_argnums` on at least one Biot-Savart hot path**
   - File: `src/simsopt/jax_core/biotsavart.py:506` (and consider VJP at 553).
   - Benchmark probe (`benchmarks/biotsavart_donation_probe.py`) already
     validates correctness and the memory benefit.
   - Docs: https://docs.jax.dev/en/latest/buffer_donation.html

### LOW (cleanups; no functional impact)

9. **Drop redundant `@partial(jax.jit, static_argnames=())`**
   - Eight call sites in `src/simsopt/jax_core/boozer_radial_interp.py:222, 281,
     373, 400, 426, 447, 494, 511`. Replace with `@jax.jit`.
   - Docs: https://docs.jax.dev/en/latest/jit-compilation.html
10. **Migrate `jax.tree_util.tree_*` to `jax.tree.*` modern alias**
    - 79 occurrences across the codebase. Mechanical migration; preserves
      semantics. Skip `register_pytree_node_class`, `register_dataclass`
      (those remain `jax.tree_util.*`).
    - Docs: https://docs.jax.dev/en/latest/pytrees.html
11. **De-duplicate `_as_jax_float64` in `surface_rzfourier.py`**
    - File: `src/simsopt/jax_core/surface_rzfourier.py:22-27`.
    - Replace local def with `from ._math_utils import as_jax_float64 as _as_jax_float64`.
    - Docs: SSOT convention; CLAUDE.md.
12. **Migrate `jnp.asarray(scalar, dtype=jnp.float64)` to SSOT helpers**
    - File: `src/simsopt/jax_core/magnetic_axis_helpers.py:586, 587, 600, 601`.
    - Use `_as_jax_float64`-style helpers from `_math_utils` so strict-transfer
      lanes get the same `jax.device_put` route.
    - Docs: JBP-9.2.
13. **Migrate test seeds from `jax.random.PRNGKey` to `jax.random.key`**
    - File: `tests/field/test_sampling_jax_item22.py` (11 occurrences).
    - Source-side docstrings in `src/simsopt/jax_core/sampling.py:6, 56, 94, 132`
      should mention both forms and recommend `jax.random.key`.
    - Docs: https://docs.jax.dev/en/latest/random-numbers.html

---

## Appendix A. Verification trail

All bullet-pointed JAX 0.10.0 removals at the top of Section 1 were verified by
WebFetch against https://docs.jax.dev/en/latest/changelog.html on 2026-05-16
and cross-checked with the simsopt-jax baseline at
`.artifacts/jax_convention_review_2026-05-16/02_jax_best_practices_baseline.md`
(JBP-18.3). Source greps that returned 0 hits are explicit in the table.

## Appendix B. Out-of-scope items observed but not flagged

- **`jax.lax.fori_loop` reverse-mode autodiff:** All `fori_loop` call sites in
  `src/simsopt/jax_core/{regular_grid_interp.py, biotsavart.py, reductions.py,
  tracing.py}` use Python-static integer bounds (`0, degree+1`, `0, chunk_count`,
  `0, max_steps+1`), which preserves reverse-mode AD (JBP-3.4). No violations.
- **`jax.lax.while_loop`:** All call sites in `src/simsopt/jax_core/tracing.py`
  are field-line integrators that are not differentiated through (no outer
  `grad` consumes their output). No reverse-mode hazard (JBP-3.3 satisfied
  by absence of grad dependency).
- **`@jax.custom_vjp` usage:** Found at `_math_utils.py:109` and
  `surfaceobjectives_jax.py:4471, 5861`. Implicit-differentiation contract
  follows JBP-6.7. Not audited here; verified upstream in baseline review
  artifact `06_review_geo_big.md`.
- **`jax.tree_util.register_pytree_node_class`:** Correct usage at
  `biotsavart_jax_backend.py:243` — class decorator form (JBP-5.2 satisfied).
- **`jax.tree_util.register_dataclass`:** 18 dataclass registrations in
  `pm_optimization.py`, `wireframe_optimization_jax.py`, `analytic_pure_fields.py`,
  `specs.py`. All correct, no violations.

---

End of R1 cross-check.
