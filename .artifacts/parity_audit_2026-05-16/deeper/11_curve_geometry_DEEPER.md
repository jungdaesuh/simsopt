# Parity Audit DEEPER — Priority 11: Curve geometry

- Audit timestamp: 2026-05-16
- Worktree: `/Users/suhjungdae/code/columbia/simsopt-jax`
- Branch: `gpu-purity-stage2-20260405`
- Phase: SECOND-PASS (hunting issues the forward-formula audit would systematically miss)

## TL;DR

The first-pass forward-formula audit was correct that **gamma/derivative scalar values** agree across all curve types and that DOF orderings match. The deeper audit surfaces real issues elsewhere:

| Severity | Finding | Where |
|---|---|---|
| HIGH | Quaternion-normalization gradient NaN at `|q|=0` (JAX) vs zero (C++). Existing test seeds `q=0` and only checks forward parity. | `curveplanarfourier.py:17-22`; `test_curve.py:1186-1265` |
| MEDIUM | C++ `dgammadash_by_dcoeff_impl` has off-by-one center-DOF loop (`i < 2` instead of `i < 3`). Benign today because `fill_array(0.0)` zero-initializes, and the correct value IS zero — but a real C++ bug that would bite under any future reordering or partial-overwrite refactor. | `curveplanarfourier.cpp:411` (vs `:281`, `:543`, `:696`) |
| LOW | RZ Fourier and Planar Fourier `gammadashdash` / `gammadashdashdash` are not pinned against C++ scalars at production scale. (Confirmed — first pass flagged but the live `_assert_curve_spec_geometry_matches_live_curve` only exercises XYZ, Perturbed, Filament.) | `tests/integration/test_single_stage_jax_cpu_reference.py:3216-3226` |
| LOW | `RotatedCurve` is not routed through the JAX `curve_geometry` dispatcher — `curve_spec_from_curve` will raise `NotImplementedError` on it. Stage-2 callers must materialize before re-entering the JAX side. | `src/simsopt/jax_core/curve_geometry.py:106-136`; `src/simsopt/geo/curve.py:1217-1376` |
| LOW | `pair_linking_number_pure` returns `jnp.round(...).astype(jnp.int32)` — not differentiable through JAX autodiff. Documented but a trap if downstream optimizers try to back-prop through `LinkingNumber.J`. | `src/simsopt/jax_core/curve_geometry.py:777-778`; `src/simsopt/geo/curveobjectives.py:1268-1270` |
| INFO | `jnp.where`-based fallbacks in `_normalized_quaternion`, `_distance_sq`, `segment_segment_distance_pure` all carry standard JAX `0 * inf = NaN` gradient traps if hit on the inactive branch. Forward values are correct; gradients at the kink are poisoned. | `curveplanarfourier.py:21`; `curve_geometry.py:495-499`, `:540-625` |
| INFO | `curvature` (`κ = |γ′×γ″| / |γ′|³`) and `torsion` (`τ = (γ′×γ″)·γ‴ / |γ′×γ″|²`) are unguarded — produce `nan`/`inf` at degeneracy. C++ does the same. Pure parity, but research scripts that drive coils through near-degenerate states will see `NaN`. | `src/simsopt/geo/curve.py:228-237`, `:258-266` |
| INFO | Lazy-import helpers `_as_jax_float64`, `_as_runtime_jax_float64`, `_as_runtime_float64_ref` are correct. The `_HAS_JAX=False` fallback returns NumPy arrays, identical to the JAX path numerically. No correctness implications. | `src/simsopt/geo/curve.py:61-85` |
| INFO | Linking-number `downsample` semantic matches C++ exactly (caller folds `downsample * dphi` into `self.dphis`, slicer applied identically on both lanes). | `curveobjectives.py:1203-1206`, `:1240-1259`; `python_distance.cpp:196-211` |

The first-pass tag of "cleanest audit, no critical findings" holds for **forward values**. The deeper sweep finds one HIGH-severity issue in the **gradient path** (quaternion normalization), one C++ off-by-one **counter** bug that is currently benign by virtue of array zero-initialization, and several research-script footguns around degeneracy.

---

## Hunt list resolution

### 1. Lazy-import correctness (CLAUDE.md cycle note)

`src/simsopt/geo/curve.py:64-85`:

```python
_HAS_JAX = _jax_vjp is not None

def _as_jax_float64(value):
    if not _HAS_JAX:
        return np.asarray(value, dtype=np.float64)
    from ..jax_core._math_utils import as_jax_float64 as _distributed_as_jax_float64
    return _distributed_as_jax_float64(value)


def _as_runtime_jax_float64(value):
    if not _HAS_JAX:
        return np.asarray(value, dtype=np.float64)
    return _as_jax_float64(value)


def _as_runtime_float64_ref(value, *, reference):
    if not _HAS_JAX:
        return np.asarray(value, dtype=np.float64)
    from ..jax_core._math_utils import (
        as_runtime_float64 as _distributed_as_runtime_float64,
    )
    return _distributed_as_runtime_float64(value, reference=reference)
```

The two branches in each helper are: (a) JAX-present → delegate to `simsopt.jax_core._math_utils.as_jax_float64` (`_math_utils.py:40`) which routes through `_explicit_device_array` (`_math_utils.py:10-14`), itself calling `jax.device_put(np.asarray(value, dtype=np.float64))`; (b) JAX-absent → plain `np.asarray(value, dtype=np.float64)`.

For numeric arrays, both arms produce identical-bit results within their respective array library. There is no scenario where the JAX-absent fallback returns a numpy array when a JAX-aware call expected a `jax.Array` and then crashes — the code is uniformly NumPy when `_HAS_JAX=False`, and uniformly JAX when `_HAS_JAX=True`. The "wrong array library" failure mode is theoretical but not present.

A subtler issue: `_as_runtime_float64_ref` ignores `reference` (`_math_utils.py:61-65` explicitly `del reference`), so any caller relying on `reference` for device or dtype-broadcast routing is being silently ignored. In practice all internal callers also pass dtype/array information through the value itself, so this is a documented signature mismatch, not a bug. **STATUS: CLEAN.**

### 2. 2π chain-rule in higher derivatives

XYZ Fourier path uses explicit `mode_scale = 2π·j` in `_fourier_basis_terms` (`curvexyzfourier.py:54-56`):

```python
mode_scale = two_pi * mode_numbers
mode_scale_sq = mode_scale * mode_scale
mode_scale_cu = mode_scale_sq * mode_scale
```

These are formed as `(2π·j)`, `(2π·j)²`, `(2π·j)³` and embedded in `dash_basis`, `dashdash_basis`, `dashdashdash_basis` — exact match to C++ `gammadash_impl`/`gammadashdash_impl`/`gammadashdashdash_impl` which post-scale by `2π`, `4π²`, `8π³`.

RZ and Planar Fourier paths compute higher derivatives via `jax.jvp` chains (`curve_geometry.py:229-247`). The chain rule autodiff applies the inner-derivative `dφ/dq = 2π` automatically. I symbolically verified one term:

- RZ `gamma_x` term: `rc[i] · cos(N·φ) · cos(φ)` with `φ = 2π·q`. First derivative w.r.t. `q`: `-2π·rc[i]·(N·sin(N·φ)·cos(φ) + cos(N·φ)·sin(φ))`. Second derivative: `-(2π)²·rc[i]·(N²+1)·cos(N·φ)·cos(φ) + (2π)²·rc[i]·2N·sin(N·φ)·sin(φ)` — matches C++ `gammadashdash_impl` line 67 with its `simsoptpp::scale_array(data, 4 * M_PI * M_PI)` postscale (line 86).

**STATUS: CLEAN — the 4π² and 8π³ factors emerge correctly from the JVP chain.**

### 3. Curvature at inflection points (γ′ = 0 or γ′ × γ″ = 0)

`src/simsopt/geo/curve.py:228-237`:

```python
@jit
def kappa_pure(d1gamma, d2gamma):
    return (
        jnp.linalg.norm(jnp.cross(d1gamma, d2gamma), axis=1)
        / jnp.linalg.norm(d1gamma, axis=1) ** 3
    )
```

There is **no guard** for `|γ′| = 0` or `|γ′ × γ″| = 0`. At `|γ′| = 0`, the JAX result is `0 / 0 = nan`. At `γ′ × γ″ = 0` with `|γ′| ≠ 0`, the result is `0 / positive = 0`, which is the mathematically correct value (zero curvature at a straight inflection).

C++ `curve.h:140`: `kappa_impl` is a virtual stub that throws — there is no C++ `kappa_impl` in any concrete class. The Python `Curve.kappa_impl` (`curve.py:608-617`) actually delegates to `kappa_pure`, so the same `nan` arises on the C++-backed path.

**STATUS: parity-equivalent (both return `nan` at `|γ′|=0`).** But a curve with a stationary point (`|γ′| = 0` at a quadpoint) would silently break optimizers via `nan` propagation. No public guard exists.

### 4. Torsion at degenerate points

`curve.py:258-266`:

```python
return jnp.sum(jnp.cross(d1gamma, d2gamma, axis=1) * d3gamma, axis=1) / jnp.sum(
    jnp.cross(d1gamma, d2gamma, axis=1) ** 2, axis=1
)
```

Same story. Denominator is `|γ′ × γ″|²` which is zero at inflection points and at `|γ′|=0`. Result: `0/0 = nan`. **STATUS: parity-equivalent (Python-side delegate also produces `nan`).** Same research-script footgun.

### 5. Quaternion rotation degenerate cases — **HIGH FINDING**

`src/simsopt/geo/curveplanarfourier.py:17-22`:

```python
def _normalized_quaternion(quaternion):
    norm_sq = jnp.sum(quaternion * quaternion)
    zero = _as_runtime_float64_ref(0.0, reference=norm_sq)
    one = _as_runtime_float64_ref(1.0, reference=norm_sq)
    inv_norm = jnp.where(norm_sq > zero, one / jnp.sqrt(norm_sq), one)
    return quaternion * inv_norm
```

The forward value at `|q|=0` is `q_norm = 0 · 1 = 0`, identical to the C++ `inv_magnitude` (`curveplanarfourier.cpp:5-13`) which returns `1` when `s == 0`. The quaternion rotation matrix at `q_norm = (0,0,0,0)` becomes the identity, in both code paths. **Forward values agree.**

**Gradient divergence**: `jnp.where(cond, A, B)` evaluates A even when cond is false; the JAX gradient through `where` is:

```
∂out/∂q = jnp.where(cond, ∂A/∂q, ∂B/∂q)
```

When `cond = (norm_sq > 0)` is false (i.e., `q = 0`), the false-branch derivative `∂B/∂q = 0` propagates. BUT the forward computation of A involves `1/sqrt(norm_sq) = 1/0 = inf`, and the chain-rule term `∂A/∂q` involves `∂(1/sqrt(norm_sq))/∂q = -0.5·norm_sq^(-3/2) · 2q = -q · norm_sq^(-3/2)`, which at `q=0` is `0 · ∞ = NaN`.

The classic JAX `0 * inf = NaN` poisoning of `where`-gradients then leaks NaN into the false branch via the multiplicative `quaternion * inv_norm` chain. Empirically (this is a documented JAX issue — the "safe sqrt" pattern uses `double_where` or `jnp.where(cond, x, 1.0)` *outside* and re-applies, exactly to avoid this), gradients from this function will be `NaN` when the optimizer happens to land on `|q|=0`.

By contrast, the C++ `dgamma_by_dcoeff_impl` (e.g., `curveplanarfourier.cpp:226-237`) computes the quaternion derivative analytically as `4·i·(...)·q_norm[0] - 4·j·(... + 0.5·q_norm[3])` etc.; at `q_norm = 0`, each term contains `q_norm[?] = 0` factors that **deterministically yield zero**. The C++ derivative is finite (zero) at `|q|=0`.

**Empirical risk**: the explicit centroid test at `test_curve.py:1186-1265` constructs a `JaxCurvePlanarFourier` with **`q = (0,0,0,0)`** (`dofs = np.zeros(...)`, only `dofs[0]=1.0` and `dofs[-3]=R0` set). The forward `gamma` is correctly `[R0, 0, 0]` because the JAX fallback `inv_norm=1` produces identity rotation. But any optimizer that uses this curve as a seed and asks for `jax.grad` w.r.t. quaternion DOFs will receive `NaN`.

**File:line citations**: `src/simsopt/geo/curveplanarfourier.py:17-22` (JAX bug source), `src/simsoptpp/curveplanarfourier.cpp:5-13` (C++ semantics it tries to match), `tests/geo/test_curve.py:1186-1265` (test that demonstrates the seed but only asserts forward).

**Recommendation**: replace `_normalized_quaternion` with a `double_where` pattern:

```python
def _normalized_quaternion(quaternion):
    norm_sq = jnp.sum(quaternion * quaternion)
    safe_norm_sq = jnp.where(norm_sq > 0, norm_sq, 1.0)  # avoid 1/0 in inner sqrt
    inv_norm = jnp.where(norm_sq > 0, 1.0 / jnp.sqrt(safe_norm_sq), 1.0)
    return quaternion * inv_norm
```

This is the canonical JAX-safe pattern (see `jax.lax.cond` discussion in the JAX docs). The forward value is identical to the current code; the gradient through the false branch is now a clean `0` rather than `NaN`.

### 6. Periodicity / closed-curve enforcement (t=0 vs t→1)

All curve classes initialize `quadpoints = np.linspace(0, 1, N, endpoint=False)` (XYZ at `curvexyzfourier.py:118`; Planar at `curveplanarfourier.py:149`; Helical/Symmetries similarly; RZ uses `np.linspace(0, 1.0/nfp, ..., endpoint=False)`). The endpoint `t=1` (or `t=1/nfp`) is never evaluated, so the discrete sample set is `[0, 1/N, 2/N, ..., (N-1)/N]`. Both C++ and JAX consume the same `quadpoints` array.

The Fourier formulas are mathematically periodic in `φ = 2π·t`, so `γ(0) = γ(1)` and `γ′(0) = γ′(1)` exactly (as continuous functions). The discretized derivative samples align: there is no "endpoint mismatch" because no endpoint is sampled.

`CurveLength.J = jnp.mean(|γ′|)` (`curveobjectives.py:55-59`) equals `(1/N) Σ |γ′(q_i)|` — the equispaced periodic Riemann sum, which for smooth periodic integrands matches the trapezoidal rule and is exponentially accurate. **STATUS: CLEAN, no periodicity artifact.**

### 7. Length integral quadrature

Confirmed in step 6: both C++ and JAX use **`L = (1/N) Σ |γ′(q_i)|`** (left-endpoint Riemann = midpoint for periodic = trapezoidal for periodic). The C++ has no internal length integral; `CurveLength` is a Python-only construct, so there is exactly one quadrature path. **STATUS: CLEAN.**

### 8. `min_distance` / self-intersection for self-intersecting curves

`segment_segment_distance_pure` (`curve_geometry.py:503-625`) is a Sunday/Lumelsky-style segment-segment distance algorithm that returns **0** for intersecting segments and finite values for non-intersecting ones. The `closed_curve_self_intersection_min_distance` aggregator (`curve_geometry.py:633-673`) returns `min` over all non-neighbor segment pairs. For a self-intersecting curve at neighbor distance ≤ `neighbor_skip`, the offending pair is **excluded** by the `wrapped_delta > neighbor_skip` filter (`curve_geometry.py:659`). For a self-intersection between non-adjacent quadpoint indices (e.g., a figure-eight), the kernel will correctly return 0 (or the floating-point neighborhood thereof).

There is **no C++ counterpart** for `segment_segment_distance_pure` (the C++ helpers in `python_distance.cpp` operate on point clouds via grid hashing and return boolean "too-close" answers, not analytic segment-segment distance). The JAX-only helper has no parity asserter. **STATUS: this is JAX-only; an out-of-scope contract not testable against C++.** The forward correctness of the Sunday/Lumelsky algorithm at degenerate inputs (parallel segments, point-segment, zero-length segments) is governed by the explicit `jax.lax.cond` branches at `curve_geometry.py:539-625`, which cover all the standard degeneracy cases.

### 9. Autodiff through linking number — **LOW FINDING**

`pair_linking_number_pure` (`curve_geometry.py:742-778`) returns `value.astype(jnp.int32)` where `value = jnp.round(jnp.abs(...) / 4π)`. Both `jnp.round` and `astype(int32)` are non-differentiable (zero gradient almost everywhere; undefined at half-integers and at the cast boundary). `LinkingNumber.dJ()` (`curveobjectives.py:1268-1270`) correctly returns `Derivative({})` — an empty derivative — so this is not exercised in any production gradient path.

C++ `compute_linking_number` (`python_distance.cpp:211`) uses `std::round(std::abs(...) / (4π))` similarly; no autodiff is plumbed.

**STATUS: parity-equivalent — both return integers via rounding; neither is differentiable.** The trap is that any new JAX caller who attempts `jax.grad(pair_linking_number_pure)(...)` will silently get zero everywhere, including where the continuous Gauss integral would have non-zero gradient. The current `LinkingNumber.dJ` correctly returns empty, but the public `pair_linking_number_pure` function exposes this footgun.

### 10. C++ UB sweep — **MEDIUM FINDING**

Read each `curve*.cpp` end-to-end. Notable observations:

- **No OMP in `curve.cpp`, `curvexyzfourier.cpp`, `curverzfourier.cpp`, `curveplanarfourier.cpp`**: single-threaded inner loops only (verified with `grep '#pragma omp' src/simsoptpp/curve*.cpp` → empty). No race conditions, no `ordered`-clause regressions.

- **`fill_array(data, 0.0)` invariant**: every `*_impl` starts with `simsoptpp::fill_array(data, 0.0)` (`curvexyzfourier.cpp:7,21,34,47`; `curverzfourier.cpp:8,35,63,91`; `curveplanarfourier.cpp:19,49,87,123` etc.). The DOF-derivative entries `dgamma_by_dcoeff_impl` etc. however do NOT begin with `fill_array(0.0)` in some places — let me double-check XYZ:

  `curvexyzfourier.cpp:58-69` (`dgamma_by_dcoeff_impl`): no `fill_array(0.0)` at the top. But the writes are sparse: `data(k, i, i*(2*order+1) + 2*j-1)` etc. Any slots not written will retain garbage. **Looking up `check_the_persistent_cache` in `curve.h:67-77`**: the cache allocator passes `xt::zeros<double>(dims)` (line 70) — so the first time the cache is queried, the underlying array is zero-initialized. Subsequent calls retrieve a **stale-but-cached** value (no `invalidate_cache` triggered for `dgamma_by_dcoeff` because it's in `cache_persistent`, not `cache`). The pattern relies on the "linear in dofs" invariant noted in `curve.h:45-49`. This is **correct by design**.

- **`curveplanarfourier.cpp:411`**: `for (int i = 0; i < 2; ++i)` in `dgammadash_by_dcoeff_impl` writes only **2** zero rows for the center DOFs, whereas the parallel methods `dgamma_by_dcoeff_impl` (line 281, `i < 3`), `dgammadashdash_by_dcoeff_impl` (line 543, `i < 3`), and `dgammadashdashdash_by_dcoeff_impl` (line 696, `i < 3`) all write **3** zero rows. **C++ off-by-one bug.**

  Mitigation: `simsoptpp::fill_array(data, 0.0)` at line 293 already zero-fills the entire data buffer. The center-DOF derivative of `gammadash` is mathematically zero (a constant offset's derivative w.r.t. itself is the identity, and `d/dt(constant) = 0`), and the zero-fill provides the correct value. So **forward parity is preserved**, but:
  - The `counter` variable walks off by 1: after the loop `counter = (2*order+1) + 4 + 2`, but the actual `num_dofs() = (2*order+1) + 4 + 3`, so the **last data slot is uninitialized except by the zero-fill**.
  - Any future refactor that removes `fill_array(0.0)` or changes the zero-init contract would silently corrupt the last `dgammadash_by_dcoeff` entry.
  
  **STATUS: real C++ typo, currently benign, but worth fixing for robustness.** Recommend changing `i < 2` to `i < 3` at `curveplanarfourier.cpp:411`.

- **`curve.cpp:36-42` `incremental_arclength_impl`**: scalar loop, `data(i) = sqrt(...)`. No UB. Returns NaN if `dg(i, k)` is NaN (e.g., from a propagated upstream); behaves identically to JAX.

- **`curve.cpp:44-54` `dincremental_arclength_by_dcoeff_impl`**: divides by `l(i)` without guarding `l(i) = 0`. If `|γ′| = 0` at any quadpoint (which shouldn't happen for valid curves), this divides by zero — same trap as JAX. Parity-equivalent.

### 11. Multi-coil shared-DOFs

`_install_curve_jax_contract` (`curve.py:321-359`) installs JAX callables on a single curve instance. Shared-DOF handling is via `_optimizable_dof_map_components` (`curve.py:394-419`) and `_optimizable_dof_map_spec` (`curve.py:422-439`), which construct an immutable spec mapping owner DOFs → target Optimizable DOFs. In `curve_geometry.py:264-292`, `_mapped_full_dofs` and `_mapped_input_dofs` apply this map via dense selector matrices (`_slice_1d_static`, `_update_1d_static`).

The pattern correctly identifies shared parameters: each owner segment is `(owner_start, owner_end, target_start, target_end)`. If two curves share the same parent Optimizable, the owner sees one segment per parent in `_full_dof_indices`, and both curves' specs reference the same parent_id implicitly through the immutable map. **No double-counting in the forward pass** because each owner DOF is written once into the target.

However, **gradient aggregation** for shared DOFs is the caller's responsibility — the JAX pullback in `curve_pullback_from_dofs` (`curve_geometry.py:890-916`) returns a `coeff_cotangent` array with shape matching the **single curve's full DOFs**. If two curves share parents, the caller must sum the cotangents back into the owner's DOF vector at the corresponding `_full_dof_indices` slices. This is done by the outer Optimizable derivative machinery; not a curve_geometry concern.

**STATUS: clean. Shared-DOF mapping is correctly handled at the spec layer, and gradient aggregation is done upstream.**

### 12. Curve types not tested — **LOW FINDING confirmed**

The production-scale test `test_curve_spec_pullback_production_scale_parity` (`test_curve_item05_closeout.py:122-171`) covers XYZ, RZ, Planar, Helical but **only asserts `gamma`** (line 162). Higher derivatives are autodiff-derived but not pinned against C++.

The `_assert_curve_spec_geometry_matches_live_curve` helper (`test_single_stage_jax_cpu_reference.py:1444-1460`) **does** pin `gammadashdash` at `atol=1e-12`, but only for XYZ, Perturbed, Filament (`:3216-3226`). **It is NOT parametrized over RZ, Planar, Helical, Symmetries.**

For PlanarFourier specifically, the production-scale seed sets `dofs[q_start] = 1.0` (`test_curve_item05_closeout.py:103`), then perturbs with `0.01 * rng.random(ndofs)`. Quaternion is `(1.0+small, small, small, small)` — close to but not identity. **A non-trivial rotation IS exercised**, but the rotation angle is `O(0.01)`. No "hard rotation" (e.g., `q = (1/√2, 1/√2, 0, 0)`, 90-degree X-axis rotation) is in the production parity matrix.

**Recommendation**: extend `test_curve_spec_pullback_production_scale_parity` to also seed a non-trivial quaternion (`q = (cos(π/4), sin(π/4), 0, 0)`) and to assert `gammadash`, `gammadashdash`, `gammadashdashdash` against the C++ oracle.

### 13. DOF count and shape

| Class | C++ num_dofs | JAX slice layout |
|---|---|---|
| XYZ Fourier | `3·(2·order+1)` (`curvexyzfourier.cpp:62-65` counter) | `coeffs = reshape(dofs, (3, 2·order+1))` (`curvexyzfourier.py:322`) ✓ |
| RZ Fourier stellsym | `2·order+1`: `[rc(0..order), zs(1..order)]` (`curverzfourier.cpp:131-153`) | `rc[0:order+1], zs[order+1:]` (`curverzfourier.py:24-28`) ✓ |
| RZ Fourier non-stellsym | `4·order+2`: `[rc, rs, zc, zs]` (`curverzfourier.cpp:131-153`) | `rc=[0,order+1), rs=[order+1, 2·order+1), zc=[2·order+1, 3·order+2), zs=[3·order+2, ...)` (`curverzfourier.py:30-32`) ✓ |
| Planar Fourier | `(2·order+1)+7`: `[rc, rs, q (4), center (3)]` (`curveplanarfourier.h:64-65, set_dofs_impl :68-78`) | `rc, rs, q, center` (`curveplanarfourier.py:60-65`) ✓ |

All DOF layouts match. **STATUS: CLEAN.**

---

## Detailed C++ off-by-one finding

**File**: `src/simsoptpp/curveplanarfourier.cpp:411-417`

```cpp
for (int i = 0; i < 2; ++i) {
    data(m, 0, counter) = 0;
    data(m, 1, counter) = 0;
    data(m, 2, counter) = 0;

    counter++;
}
```

**Compare against the three sibling methods** (all use `i < 3`):

- `dgamma_by_dcoeff_impl` at `curveplanarfourier.cpp:281-287`:
  ```cpp
  for (int i = 0; i < 3; ++i) {
      data(m, 0, counter) = 0;
      data(m, 1, counter) = 0;
      data(m, 2, counter) = 0;
      data(m, i, counter) = 1;  // NOTE: also sets the diagonal!
      counter++;
  }
  ```
  This is `dgamma/dcenter = identity_3x3` per quadpoint, which is correct since `gamma += center`.

- `dgammadashdash_by_dcoeff_impl` at `curveplanarfourier.cpp:543-549`: identical to the buggy site but with `i < 3`.

- `dgammadashdashdash_by_dcoeff_impl` at `curveplanarfourier.cpp:696-702`: identical to the buggy site but with `i < 3`.

The buggy `dgammadash_by_dcoeff_impl` (line 411) only writes 2 of the 3 center slots. Because `simsoptpp::fill_array(data, 0.0)` at line 293 has already zeroed the entire buffer, and because the mathematically correct value IS zero (a constant translation has zero derivative w.r.t. the parameter `t`), the **forward output is correct**. But the `counter` variable advances by only 2, so:

- After the loop: `counter = (2·order+1) + 4 + 2 = 2·order+7`
- Expected: `num_dofs() = (2·order+1) + 4 + 3 = 2·order+8`

The "missing" counter step does not cause a write past the end of the buffer (because we exit the loop), but if any new code is added that uses `counter` to drive an assertion or to iterate over a downstream secondary buffer, this off-by-one will silently break it. **Recommend fixing to `i < 3` for consistency and future-proofing.**

---

## Existing test coverage that hides the quaternion gradient trap

`tests/geo/test_curve.py:1186-1208`:

```python
# Use a simple planar circle for which the centroid is known
curve = CurvePlanarFourier(nquad, order)
dofs = np.zeros(curve.dof_size)
dofs[0] = 1.0  # radius
# Set the center to (R0, 0, 0)
dofs[-3] = R0
dofs[-2] = 0.0
dofs[-1] = 0.0
curve.set_dofs(dofs)
centroid = curve.centroid()
# The centroid should be at (R0, 0, 0)
np.testing.assert_allclose(
    centroid,
    [R0, 0.0, 0.0],
    atol=1e-12,
    err_msg="Centroid of the planar curve should be at the center (R0, 0, 0)",
)
```

The setup literally sets `q = (0, 0, 0, 0)` (lines 1194-1199: `dofs = np.zeros(...)`, then only `dofs[0]` and `dofs[-3]` are overwritten). Both C++ and JAX paths fall back to identity rotation, **forward result matches**. The test repeats this for `RotatedCurve(CurvePlanarFourier(...))` (line 1211), then for `JaxCurvePlanarFourier(...)` (line 1230) again with `dofs = np.zeros(...)`.

The test never differentiates anything w.r.t. `q`, so the gradient NaN trap is not surfaced. A new oracle row that does `jax.grad(jax.numpy.sum)(curve_geometry_from_dofs(spec, q_zero_dofs)[0])` would fail (or produce `NaN`) on the JAX side while the C++ analytic `dgamma_by_dcoeff` returns the finite zero.

---

## Recommended actions

| Priority | Action | Files touched |
|---|---|---|
| **HIGH** | Fix `_normalized_quaternion` to use the JAX "double-where" pattern to prevent `NaN` gradients at `\|q\|=0`. The pattern is: `safe_norm_sq = where(norm_sq > 0, norm_sq, 1.0); inv_norm = where(norm_sq > 0, 1/sqrt(safe_norm_sq), 1.0)`. | `src/simsopt/geo/curveplanarfourier.py:17-22` |
| **HIGH** | Add a regression test that pins `jax.grad(...)` of a Planar Fourier objective w.r.t. quaternion DOFs at `q=0`, asserting finiteness. | new fixture in `tests/geo/test_curve.py` |
| **MEDIUM** | Fix the C++ off-by-one in `dgammadash_by_dcoeff_impl`: change `for (int i = 0; i < 2; ++i)` → `for (int i = 0; i < 3; ++i)`. Currently masked by `fill_array(0.0)`, but a real typo. | `src/simsoptpp/curveplanarfourier.cpp:411` |
| **LOW** | Extend `test_curve_spec_pullback_production_scale_parity` (`tests/geo/test_curve_item05_closeout.py:122-171`) to assert `gammadash`, `gammadashdash`, and `gammadashdashdash` parity for all four C++-backed classes, and to seed at least one non-trivial-rotation Planar Fourier coil (e.g., `q = (cos(π/4), sin(π/4), 0, 0)`). | `tests/geo/test_curve_item05_closeout.py` |
| **LOW** | Either drop the int-cast on `pair_linking_number_pure` (so callers receive a float that can be differentiated through the unrounded Gauss integral) and round at the caller, or document that the return is not differentiable. | `src/simsopt/jax_core/curve_geometry.py:763-778` |
| **INFO** | Document the kappa/torsion NaN-at-degeneracy contract and add a research-mode safe-divide variant for callers who need finite-gradient continuation through curve straightening. No production action required. | `src/simsopt/geo/curve.py:228-237`, `:258-266` |

---

## Items confirmed clean (no action)

- Lazy-import helpers in `src/simsopt/geo/curve.py:64-85` — both fallback paths produce numerically identical arrays.
- 2π / 4π² / 8π³ chain-rule factors — verified symbolically; JVP autodiff produces the same coefficients as the C++ post-scale.
- DOF orderings for all four curve types — counter ordering in C++ matches JAX slice/reshape order.
- Periodicity at t=0 vs t→1 — both lanes use `endpoint=False` linspace; no endpoint mismatch.
- Length quadrature — both lanes use equispaced periodic Riemann sum (= trapezoid for smooth periodic).
- Multi-coil shared-DOF mapping — clean at the spec layer.
- Linking-number `downsample` semantics — kernel inputs match exactly between C++ and JAX paths.
- C++ OMP usage in curve files — no parallel regions, no race risk.

The forward-formula parity remains the cleanest priority audit of the suite. The HIGH-severity issue is gradient-only and requires explicit triggering on `\|q\|=0`; the MEDIUM C++ typo is benign by virtue of zero-initialization.
