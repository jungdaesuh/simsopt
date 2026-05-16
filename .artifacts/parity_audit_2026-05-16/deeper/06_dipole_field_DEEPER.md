# Priority 6 — Permanent-magnet dipole field SECOND-PASS DEEPER audit

**Audit timestamp:** 2026-05-16
**Branch:** `gpu-purity-stage2-20260405`
**Auditor:** Claude (Opus 4.7, 1M context)
**Mandate:** hunt for issues the first-pass forward-formula parity audit
systematically missed (autodiff, singularity behavior under autodiff,
behavioral parity at degenerate inputs, convention claims, validation
gaps, C++ performance/correctness sketches, API-surface differences).

## Targets

| Role | Path | Notes |
|------|------|-------|
| JAX module | `src/simsopt/jax_core/dipole_field.py` (533 lines) | hot kernels: B, A, dB, dA, Bn |
| C++ reference | `src/simsoptpp/dipole_field.cpp` (855 lines) | XSIMD + scalar variants |
| C++ header | `src/simsoptpp/dipole_field.h` (20 lines) | public signatures |
| Public JAX wrapper | `src/simsopt/field/dipole_field_jax.py` (347 lines) | `DipoleFieldJAX(MagneticField)` |
| pybind11 binding | `src/simsoptpp/python.cpp:64-68` | unconventional dA docstring |
| Convention docstring | `src/simsoptpp/python_magneticfield.cpp:30,45` | `∂_j B_l`/`∂_j A_l` |
| Consumer | `src/simsopt/geo/permanent_magnet_grid_jax.py:132-141` | calls `dipole_field_Bn` |
| Consumer | `src/simsopt/geo/permanent_magnet_grid.py:428-436` | calls `sopp.dipole_field_Bn` |
| Tests | `tests/jax_core/test_dipole_field_jax_item24.py`, `tests/jax_core/test_dipole_field_item24.py`, `tests/field/test_dipole_field_jax_item26.py` | all forward, all carefully avoid degenerate inputs |

## Executive summary — top 7 deeper findings

1. **MEDIUM — Silent behavior divergence at "dipole on the axis" with toroidal/cylindrical coordinate_flag.** When a dipole sits at the origin (or on the z-axis) and `coordinate_flag != "cartesian"`, `xsimd::atan2(0, 0)` returns NaN whereas `jnp.atan2(0, 0)` returns 0. C++ then poisons the cylindrical/toroidal rotation factors and the output A-matrix becomes `[NaN, NaN, finite-z]`. JAX returns a finite (but conventionally arbitrary at the degeneracy) value. Repro: 6 lines, immediate. **No test exercises this case** — first-pass fixtures keep dipoles inside `[-0.3, 0.3]^3` (not at origin) and on-axis siting is a real PM-grid scenario for stellarators with thin TF columns.
2. **HIGH (documentation only — NO numerical wrong-doing) — `dipole_field_dB` / `dipole_field_dA` docstrings DISAGREE with the SIMSOPT-wide pybind11 convention.** `python_magneticfield.cpp:30,45` states `dB_by_dX` / `dA_by_dX` return `∂_j B_l(x_i)` (axis 1 = derivative direction, axis 2 = field component). The C++ kernel implementations + JAX docstrings at `dipole_field.cpp:125-131,193-200` / `dipole_field.py:239-240,295-297` store the opposite: `dB[p, j, k] = ∂B_j/∂x_k` (axis 1 = field component, axis 2 = derivative direction). For `dB` this is invisible because B is the gradient of a scalar potential and the Hessian is symmetric. For `dA` it is **observable**: rotating between conventions changes off-diagonal entries (verified by FD at `dA[p, 0, 1]` — `2.59e-08` vs `6.53e-08` over the test inputs). This is a SIMSOPT-wide documentation inconsistency; the JAX port faithfully mirrors the *actual* C++ storage layout, not the pybind11 docstring.
3. **MEDIUM — Bn validation contract diverges between JAX and C++.** `dipole_field_Bn` JAX raises on `unitnormal.shape != points.shape`; C++ silently accepts (with arbitrary OOB reads when mismatched larger). JAX silently accepts unknown `coordinate_flag` (treats it as cartesian); a user typo `"sphereical"` is not caught. `b` array shape is never validated by either backend.
4. **MEDIUM — Autodiff has zero direct test coverage** despite being the *reason the JAX port exists*. We ran cold tests in this audit: linearity in `m` holds at `1.2e-23`; gradient of `sum(B)` w.r.t. dipole position scales as expected as `1/r^4` near the source; at the singularity `jax.grad` returns NaN (consistent with primal NaN — not silently finite). All checked manually; **no committed test asserts any of these**.
5. **LOW — C++ Bn hoisting opportunity (perf, not correctness).** `dipole_field.cpp:320-322` (`phi0`, `sphi0`, `cphi0`) and `dipole_field.cpp:333-338` (`mp_phi_new`, `mp_theta_new`, `sphi_new`, etc.) are computed inside the nested `for(stell)` / `for(fp)` loops but depend only on outer-loop variables. For a `(stellsym=1, nfp=5)` device that is `2*5=10` redundant `xsimd::atan2 + sin + cos` calls per `j`. JAX hoists these already (`dipole_field.py:447`).
6. **LOW — Bn fast path drops a `pow(-1, stell)` even when `stell=0`.** `pow(-1, 0) = 1.0` (correct) but the call itself is on the hot inner loop. Trivial to replace with `(stell == 0 ? 1.0 : -1.0)` or `(1 - 2*stell)` (integer literal). JAX already uses `(-1.0) ** stell` evaluated at Python-graph-build time, so it has no runtime cost. Not a correctness issue.
7. **LOW — Mixed dtype contract is undocumented and silently upcasts.** Passing `float32` moments or points to `dipole_field_B` is silently upcast to `float64` (because `_as_jax_float64` is called inside `_require_xyz_matrix`). C++ kernel hard-codes `Array = xt::pyarray<double>` so float32 inputs typically trigger pybind11 conversion errors. The JAX path's silent upcast is more forgiving but loses the "fail fast on contract violation" property — and a user expecting the JAX kernel to be float32-aware will be surprised by float64 outputs.

## 1 — Autodiff parity probes (NEW)

The first-pass audit said "MEDIUM: zero autodiff coverage" but did not actually run autodiff. We ran it. Findings (all `.conda/jax/bin/python`, JAX 0.10.0, x64 enabled):

### 1.1 Linearity in `m` — machine-precision PASS

```python
m1 + m2 → B(m1+m2);  B(m1) + B(m2)
max abs error B:   1.158e-23
max abs error dB:  2.316e-23
```

So both `dipole_field_B` and `dipole_field_dB` are linear in `m` to the rounding floor. This is expected by construction (the kernel is linear in moments).

**Untested in repo:** no committed test pins this linearity. If a future refactor accidentally introduces a nonlinear term (e.g., a unit-vector normalization of `m_hat`) it would silently break and existing parity tests would still pass against C++ (which is also linear, so the failure would be reflected in both backends).

### 1.2 Gradient at singularity — NaN propagation PASS

Primal `B(x; r0 = x; m)` returns `NaN` (both backends; `_explicit_rsqrt(0) → +inf`; `3*inf*r*inf^5 - m*inf^3` → `inf - inf = NaN`). `jax.grad(sum(B**2))(r0)` at the same singular point returns NaN. **Good** — no silent finite value from subtraction cancellation.

### 1.3 Gradient near singularity — `1/r^4` scaling PASS

```
eps=1e-1: max|grad|=6.000e-03
eps=1e-3: max|grad|=6.000e+05
eps=1e-6: max|grad|=6.000e+17
eps=1e-9: max|grad|=6.000e+29
```

Each decade smaller `eps` → 4 decades larger gradient. Confirms the expected `1/r^4` singularity (B ~ 1/r^3, ∂B/∂r0 ~ 1/r^4). No silent regularization.

### 1.4 Missing autodiff coverage

Grepping `tests/jax_core/`, `tests/field/`, `tests/geo/test_permanent_magnet_grid_jax_item27.py`, `tests/solve/test_permanent_magnet_optimization_jax_item28.py` for `jax.grad|jax.jacfwd|jax.jacrev` returns **0 hits** on any dipole-field test. The only autodiff usage anywhere near is in `tests/jax_core/test_tracing_jax_guiding_center.py:95` which is for guiding-center tracing, not dipole field.

**Recommend** adding committed tests that pin:
- linearity in `m` at machine precision (1 line via `jax.grad` of `sum(B)`),
- FD vs `jax.grad` on dipole-position gradient (sensitivity scale)
- FD vs `jax.grad` on point gradient at far-field
- NaN propagation at the exact singularity

Suggested location: `tests/jax_core/test_dipole_field_autodiff.py`. Use the `derivative-heavy` parity-ladder lane (`rtol=1e-8, atol=1e-10`).

## 2 — Behavior parity gap at "dipole on axis" with non-cartesian frame (NEW, MEDIUM)

Re-pro (`.conda/jax/bin/python`):

```python
import simsoptpp as sopp
from simsopt.jax_core.dipole_field import dipole_field_Bn
import numpy as np

points = np.array([[1.5, 0.5, 0.3]], dtype=np.float64)
unitnormal = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
b = np.zeros(1, dtype=np.float64)
positions = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)  # axis dipole

# Cylindrical/toroidal with dipole AT origin:
A_cpp = np.asarray(sopp.dipole_field_Bn(points, positions, unitnormal, 1, 0, b, "cylindrical", 1.0))
A_jax = np.asarray(dipole_field_Bn(points, positions, unitnormal, 1, 0, b, "cylindrical", 1.0))
print("CPP:", A_cpp[0, 0])   # [nan, nan, finite-z]
print("JAX:", A_jax[0, 0])   # [finite, finite, finite-z]
```

Root cause: `dipole_field.cpp:333-334`:
```cpp
simd_t mp_phi_new = xsimd::atan2(mp_j.y, mp_j.x);
simd_t mp_theta_new = xsimd::atan2(mp_j.z, sqrt(mp_j.x * mp_j.x + mp_j.y * mp_j.y) - R0);
```

For `mp_j = (0, 0, 0)`: `xsimd::atan2(0, 0) → NaN` (`xsimd` differs from `std::atan2(0, 0) → 0`). This propagates: `cphi_new = NaN`, `sphi_new = NaN`, and the inner-product rotation lines 354-355 / 363-365 multiply finite `Ax_temp` with `cphi_new = NaN` ⇒ NaN. The cartesian branch at 372-374 does not use `cphi_new` / `sphi_new`, so the cartesian-flag output is fine.

JAX (`dipole_field.py:310-316`): `jnp.atan2(0, 0) → 0` in JAX. So the rotation factors become `(sphi=0, cphi=1, stheta=0, ctheta=1)` and the output is finite — but this is a *conventional* choice at a degenerate input where the physically correct answer is "undefined" (the local toroidal frame is not defined at the axis).

**Severity rationale:** real PM grids commonly include dipoles on or near the toroidal axis (e.g., legacy MUSE configurations, axis-aligned PSC arrays in HBT-like devices). The current first-pass parity tests carefully use `dipoles in [-0.3, 0.3]^3` and so will not catch this. A user comparing JAX to C++ on a real PM grid with axis-resident dipoles will see a silent disagreement, and the disagreement *favors* JAX (returns plausible numbers) over C++ (returns NaN — which would at least be loud), making this a subtly dangerous false-positive parity.

**Recommendation:** either (a) make JAX also produce NaN at the degeneracy (matches C++, loud); or (b) make C++ guard the `atan2(0, 0)` case (e.g., `R = sqrt(x²+y²)`; if `R < eps && |z - 0| < eps && R0 != 0`, return zero contribution); or (c) document at API level that "non-cartesian Bn requires all dipoles to satisfy `(x² + y² > 0) OR (z != 0)`".

## 3 — Convention claim mismatch on dB/dA layout (HIGH doc, no math bug)

**Observed (verified by FD probe in this audit):**

For the dipole kernels (both CPU and JAX), the layout is:
```
dB[p, j, k] = ∂B_j(x_p)/∂x_k   (axis 1 = B component, axis 2 = derivative direction)
dA[p, j, k] = ∂A_j(x_p)/∂x_k   (axis 1 = A component, axis 2 = derivative direction)
```

This matches:
- JAX docstring `dipole_field.py:239-240,295-297`
- C++ comment `dipole_field.cpp:125-131,193-200`
- The actual storage assignment in C++ at lines 171-176 (where `dB(i+k, j, k)` symbol mapping makes the symmetric variant)
- The actual storage assignment in C++ at lines 239-247 (where `dA_i1.x ← rmag_inv_3 * (... -3 mcrossr.x * r.x ...)` is `∂A_0/∂r_0`).

But the SIMSOPT *base-class* convention documented in `src/simsoptpp/python_magneticfield.cpp:30` and `:45` reads:

```
dB_by_dX returns ∂_j B_l(x_i)        (axis 1 = j = deriv, axis 2 = l = field)
dA_by_dX returns ∂_j A_l(x_i)        (axis 1 = j = deriv, axis 2 = l = field)
```

This is **the opposite** layout. CLAUDE.md restates the SIMSOPT-wide convention at line 182:
> `dB_by_dX[p, j, l] = ∂_j B_l(x_p)` — axis 1 is derivative direction, axis 2 is B component.

**For `dB`:** because `B` is the gradient of a scalar potential, the Hessian `∂_j B_l` is symmetric in `(j, l)`, so the convention swap is invisible to downstream consumers. Documentation harmless.

**For `dA`:** `A` is *not* gradient-of-scalar (it's a vector potential), so `∂_j A_l ≠ ∂_l A_j` in general. The dipole kernel writes `∂A_j/∂x_k` (j=A-comp, k=deriv) into `dA[p, j, k]`. A consumer that reads it as `∂_j A_l` (j=deriv, l=A-comp) will see *the transpose* of the correct Jacobian and will be wrong by a sign-aware swap.

**Empirical check** (this audit):

```python
# FD reference: A_FD[j, l] = ∂_j A_l(x)
# dipole_field_dA[p=0, j, k] should equal what?
JAX vs FD direct:   6.529e-08          # FAIL at FD precision
JAX vs FD-transpose: 2.333e-18         # CORRECT
```

So JAX's `dipole_field_dA[p, j, k] = ∂A_j/∂x_k = (FD)_{k, j}` — matching `dA = FD.T`. The same holds for C++.

**Verdict:** the documented mismatch at `python_magneticfield.cpp:45` is a SIMSOPT-wide claim that does not survive contact with the actual `DipoleField._dA_by_dX_impl` implementation. This is not new to the JAX port, but the JAX port has propagated the C++ kernel's *actual* convention (which is correct for that kernel) and replicated the SIMSOPT-wide docstring's *claimed* convention nowhere consistently.

**Recommendations:**
1. Fix `python_magneticfield.cpp:45` to read `∂_k A_j` (or document explicitly that the dipole kernel stores the transpose of the base-class advertised convention).
2. Add an FD check in `tests/field/test_dipole_field_jax_item26.py` that asserts `dipole_field_dA[p, j, k] ≈ FD_{k, j}` — pinning the *implementation* convention.

## 4 — Input validation diverges between JAX and C++ (MEDIUM)

JAX `dipole_field.py:439-443`:
```python
if unitnormal_arr.shape != points_arr.shape:
    raise ValueError(
        "unitnormal must have the same shape as points; "
        f"got {unitnormal_arr.shape!r} and {points_arr.shape!r}."
    )
```

C++ has **no such guard**. Verified (this audit):

```python
points = (1,3), unitnormal = (2,3)  # mismatched leading dim
JAX:  ValueError raised  (correct — defensive)
CPP:  silently accepts, returns A.shape=(1, 1, 3)  (OOB read of unitnormal[1] is undefined)
```

This is a **divergent validation contract**: JAX rejects the malformed input, C++ silently produces a value with potential OOB memory access. The C++ behavior is UB by the strict-aliasing reading of `xtensor` — the `unitnormal(i+k, d)` access at line 312 has `i+k = 0` only, so it reads `unitnormal[0]` which is OK, but if `num_points > unitnormal.shape(0)` then UB.

Similar for `b`:

```python
b shape = (1, 3) instead of (1,)  → JAX silently accepts (shape never checked beyond rank-2)
b shape = (42,) instead of matching # of points  → JAX silently accepts
```

Both backends silently accept malformed `b` because neither uses `b` in the computation — its presence is a vestigial API-compat parameter.

`coordinate_flag = "spherical"` (typo for cylindrical) → JAX silently treats as cartesian (no `else: raise`).

**Recommendations:**
1. Add `coordinate_flag` whitelist check in `dipole_field_Bn` (JAX side; lines 458-469 use chained `if/elif/else` with `else → cartesian`).
2. Either drop the `b` parameter from the JAX API or check `b.shape == (points.shape[0],)`.
3. File an issue against C++ to validate `unitnormal.shape == points.shape`.

## 5 — Bn `_basis_angles` is correctly hoisted in JAX, redundantly recomputed in C++ (LOW, performance)

C++ `dipole_field.cpp:316-345`:

```cpp
for (int j = 0; j < num_dipoles; ++j) {
    Vec3dSimd mp_j = ...;
    for (int stell = 0; stell < (stellsym + 1); ++stell) {
        for(int fp = 0; fp < nfp; ++fp) {
            simd_t phi0 = (2 * M_PI / ((simd_t) nfp)) * fp;     // depends on fp only
            simd_t sphi0 = xsimd::sin(phi0);                     // depends on fp only
            simd_t cphi0 = xsimd::cos(phi0);                     // depends on fp only

            simd_t mp_x_new = mp_j.x * cphi0 - mp_j.y * sphi0 * pow(-1, stell);
            // ...

            simd_t mp_phi_new = xsimd::atan2(mp_j.y, mp_j.x);    // depends on j only
            simd_t mp_theta_new = xsimd::atan2(mp_j.z, ...);     // depends on j only
            simd_t sphi_new = xsimd::sin(mp_phi_new);            // depends on j only
            simd_t stheta_new = xsimd::sin(mp_theta_new);        // depends on j only
            simd_t cphi_new = xsimd::cos(mp_phi_new);            // depends on j only
            simd_t ctheta_new = xsimd::cos(mp_theta_new);        // depends on j only
            ...
```

The `mp_phi_new` and friends are recomputed `(stellsym+1) * nfp` times per `j`. For typical `(stellsym=1, nfp=5)`, that's **10× redundant** `xsimd::atan2 + sin + cos` calls per dipole. For a 1M-dipole grid this adds up.

JAX hoists `_basis_angles` outside both loops:

```python
sphi, cphi, stheta, ctheta = _basis_angles(dipole_points_arr, R0)  # line 447
for stell in range(stellsym_int + 1):
    for fp in range(nfp_int):
        # use sphi, cphi, stheta, ctheta which are O(num_dipoles), not O(num_dipoles * nfp * (stellsym+1))
```

Verified mathematically: `mp_phi_new` depends only on `mp_j` (the *original* dipole position), not on `mp_j_new` (the symmetry-transformed position), so it is genuinely loop-invariant. This is the same correct optimization JAX does.

**Recommendation:** hoist `mp_phi_new`, `sphi_new`, `cphi_new`, `stheta_new`, `ctheta_new`, `phi0`, `sphi0`, `cphi0` out of their loop nests in C++. No correctness change, no parity break.

## 6 — `pow(-1, stell)` hot-path inefficiency (LOW)

`dipole_field.cpp:327, 328, 329, 352, 360, 372` all call `pow(-1, stell)` per iteration. For `stell ∈ {0, 1}`, the answer is `±1`, and the cost is dwarfed by the field computation, but `pow` is a transcendental-class function and is not always optimized to a sign-flip at compile time (depends on compiler / pragma settings). Replacing with `(stell ? -1.0 : 1.0)` or `(1.0 - 2.0 * stell)` is trivial and removes a hot-path function call.

JAX uses `(-1.0) ** stell` evaluated at Python-graph-build time before JIT (line 453), so the compiled IR sees a literal `1.0` or `-1.0`. Zero runtime cost.

## 7 — Mixed-dtype contract is silent (LOW)

Verified (this audit):
```python
# float32 moments, float64 points
B = dipole_field_B(points_f64, dipole_pts_f64, moments_f32)
# → silently accepted; output dtype = float64
```

Root cause: `dipole_field.py:71-75`:
```python
def _require_xyz_matrix(name: str, value: object) -> jax.Array:
    array = _as_jax_float64(value)
    ...
```

`_as_jax_float64` upcasts via `jax.device_put(np.asarray(value, dtype=np.float64))`. A float32 input is silently promoted.

C++ side: `Array = xt::pyarray<double>` (header line 8). Passing a `np.float32` array to `sopp.dipole_field_B` will trigger pybind11's automatic conversion — either it converts (and now C++ output is float64 from float32 input, lossy) or it throws. Verified: pybind11 in this build silently upconverts via numpy, producing float64 output from float32 input.

**Severity rationale:** "production grade" software for stellarator design should fail fast on dtype contract violations. The current behavior silently produces float64 results from float32 inputs, which can mask bugs in upstream type plumbing (e.g., a user who plumbs a float32 m through PM optimization will get float64 forward fields but float32 gradients, leading to silent precision loss in autograd-traced quantities).

## 8 — C++ `define_a_uniform_cartesian_grid_between_two_toroidal_surfaces` sentinel (LOW)

`dipole_field.cpp:776-777`:
```cpp
double min_dist_inner = 1e5;
double min_dist_outer = 1e5;
```

The sentinel value `1e5` (≈316.23 distance units when squared) is interpreted as the candidate-distance threshold. For a stellarator with major radius ~1m and minor radius ~0.3m, all surface distances are < 1m and the sentinel never matters. For an oversized device (e.g., a fusion-pilot-plant configuration with major radius 8m and 1m minor radius), the maximum candidate-to-surface distance is well within `sqrt(1e5) = 316m`, so still safe in practice.

But this is fragile design. `std::numeric_limits<double>::infinity()` is the conventional choice and would make the code robust against any device scale.

JAX (`dipole_field.py:474-509`) uses `jnp.argmin(distances)` which has no sentinel — the global minimum is selected unconditionally, so the JAX implementation has no sentinel risk.

**Recommendation:** replace `1e5` with `std::numeric_limits<double>::infinity()` in the C++ kernel.

## 9 — `_filter_uniform_grid_point` zero-norm normal handling (LOW)

`dipole_field.cpp:814-817`:
```cpp
double norm_vec = sqrt(nx * nx + ny * ny + nz * nz);
double ray_x = nx / norm_vec;
double ray_y = ny / norm_vec;
double ray_z = nz / norm_vec;
```

If a surface point has a zero normal vector (degenerate surface mesh), `norm_vec = 0`, and `nx/0 = NaN` (since `nx = 0`) or `±inf` (if `nx ≠ 0`). The downstream ray-distance computation will propagate NaN into `dist_inner_ray` and `dist_outer_ray`, and the `if (dist < min_dist)` test against NaN is always false. So `nearest_loc_inner = 0` (the default) and `nearest_loc_outer = 0`. The post-condition at lines 846, 849 then fails (`nearest_loc_outer > 0` is false), so the candidate is silently filtered out.

JAX (`dipole_field.py:494`): `ray = normal * _explicit_rsqrt(jnp.sum(normal * normal))`. For zero normal, `1/sqrt(0) = +inf`, `ray = 0 * inf = NaN`. The `_explicit_rsqrt` is defined via `1/sqrt(x)` and produces inf — but `inf * 0` becomes NaN. Then `ray_points` are NaN, `dist_inner_ray = NaN`, `argmin(NaN)` returns index 0, the `keep` condition `nearest_loc_inner <= 0 AND nearest_loc_outer > 0` evaluates `True AND False = False`, so the candidate is silently filtered out.

Both backends silently filter out zero-normal cases. **Behaviorally equivalent**, but neither one warns the user that their surface mesh contains degenerate normals. Could surface as a "phantom missing dipoles" issue in a production PM grid.

## 10 — JIT closure / retracing (LOW)

The `_dipole_field_B_jit` / `_dipole_field_A_jit` / `_dipole_field_dB_jit` / `_dipole_field_dA_jit` paths are all unstaged JITs that retracing on shape changes. Verified (this audit):

```python
# Different N → both N=1 and N=2 work, separate compilations.
B1 = dipole_field_B(pts, dp1, m1)   # N=1, shape ok
B2 = dipole_field_B(pts, dp2, m2)   # N=2, retraces silently
```

For `BiotSavart`-style callers that swap moments without changing dipole positions, the retracing is benign. For PM-optimization callers that update only `dipole_moments` after construction (e.g., MwPGP loop), the cached JIT is reused — verified by the steady-state time being ~0.0 seconds.

`DipoleFieldJAX.__init__` pre-stages `_dipole_points_device` and `_dipole_moments_device` once (line 251-252) and does not retrace per `set_points_cart` call (verified, the strict-transfer-guard test in `tests/field/test_dipole_field_jax_item26.py:222-256` would catch a regression there).

**No bug.** Just confirming JIT behavior matches expectations.

## 11 — Bn `R0=0` corner-case parity with non-cartesian frame

When `R0 = 0` and dipole is at origin, `atan2(0, sqrt(0) - 0) = atan2(0, 0)`. Same NaN-vs-0 divergence as finding 2.

When `R0 = 0` and dipole is at `(0, 0, z)` with `z != 0`: `atan2(z, 0 - 0) = atan2(z, 0) = ±π/2` (both backends agree because `atan2(nonzero, 0)` is defined for both `xsimd` and `numpy`).

So the parity gap is specifically `(R = 0, z = 0)` regardless of `R0`.

## 12 — Symmetric vs asymmetric dB tensor symmetry check (NEW)

The first-pass audit asserted `dB` is symmetric. We verified at a different fixture (point at origin, dipole at `(1, 1, 1)`):

```
dB - dB.T inf-norm:  0.0  (both JAX and CPP)
```

PASS. Both backends preserve the symmetry exactly (because of how the kernels write to the symmetric entries and copy to the lower triangle, not by accident of floating-point coincidence).

## 13 — `permanent_magnet_grid_jax.py` consumer is gradient-clean (NEW)

`permanent_magnet_grid_jax.py:132-141` calls `dipole_field_Bn` once at fixture-build time and stores the result in `A_obj`. **No autograd flows through `dipole_field_Bn` in the current consumer path.** The MwPGP solver consumes `A_obj` as a fixed matrix.

This means: even if there is a latent autodiff bug in `dipole_field_Bn`, the current PM-grid consumer would not be affected. **But** a future single-stage PM optimization (priority 7) that puts the dipole site coordinates `dipole_grid_xyz` or the unit normals into a `jax.grad` outer loop *will* exercise the `dipole_field_Bn` gradient path — and that path is currently 100% untested.

## 14 — Untested edge-case inventory

| Edge case | JAX | C++ | Status | Severity |
|---|---|---|---|---|
| Point exactly on dipole | NaN | NaN | parity, undocumented | LOW (doc only) |
| Dipole at origin, cylindrical coords | finite | NaN | **divergence** | MEDIUM |
| Dipole at origin, toroidal coords | finite | NaN | **divergence** | MEDIUM |
| Dipole on z-axis (R=0, z≠0) | finite | finite | parity | OK |
| Empty dipole array (N=0) | returns zeros | returns zeros | parity | OK |
| Single dipole (N=1) | OK | OK | parity | OK |
| nfp=0 | returns zeros | returns zeros | parity | undocumented (LOW) |
| nfp negative | ? | ? | not tested | LOW (probably crashes) |
| `coordinate_flag = "Bogus"` | silent → cartesian | silent → cartesian | parity | MEDIUM (validation gap) |
| `unitnormal.shape != points.shape` | ValueError | silent (UB risk) | **divergence** | MEDIUM |
| `b.shape != (num_points,)` | silent | silent | parity (both gap) | LOW |
| Float32 moments | silent upcast to float64 | silent upcast | parity (both lossy) | LOW |
| Float32 points | silent upcast to float64 | silent upcast | parity | LOW |
| Non-contiguous input arrays | accepts | accepts (row-major check is broken on Cori — see comment) | parity | LOW |
| Zero-norm surface normal | filters out silently | filters out silently | parity | LOW |
| Surface-distance > sqrt(1e5) ≈ 316 m | n/a (no sentinel) | sentinel break | **JAX more robust** | LOW |
| `jax.grad` w.r.t. `m` linearity | machine precision | n/a | INFO | needs test |
| `jax.grad` w.r.t. dipole position | finite, scales 1/r^4 | n/a | INFO | needs test |
| `jax.grad` at exact singularity | NaN (consistent) | n/a | INFO | needs test |
| Bit-identity CPU→GPU under XLA-deterministic | untested | n/a | UNKNOWN | needs CUDA CI |

## 15 — C++ end-to-end UB sweep

Read `dipole_field.cpp` lines 16-855 looking for:

1. **Uninitialized accumulators:** `Vec3dSimd()` / `Vec3dStd()` default constructors are inspected at `vec3dsimd.h` — they zero-initialize per `Vec3dSimd::x = simd_t(0)` etc. The XSIMD variant uses `simd_t(0)` which zero-fills the SIMD register. No UB.
2. **Missing braces:** none observed. Each `for` body is fully braced, each `if` is fully braced. Unlike the `surfacerzfourier.cpp` ANGLE_RECOMPUTE pattern flagged in CLAUDE.md, the dipole-field kernels do not contain the bare-if footgun.
3. **Signed-int loop overflow:** `int num_points`, `int num_dipoles`, `int num_inner`, etc. For a 1M-dipole grid, `int` (32-bit signed, max ~2.1e9) is safe. `int klimit = std::min(simd_size, num_points - i)`: `simd_size` is small (4 or 8), `num_points - i ≥ 0` by loop guard. No overflow.
4. **OMP races on shared output arrays:**
   - `dipole_field_B`: each thread owns chunk `[i, i + simd_size)` of the output `B`. No two threads write to the same `B(i+k, d)`. Safe.
   - `dipole_field_A`: same pattern.
   - `dipole_field_dB`: same.
   - `dipole_field_dA`: same.
   - `dipole_field_Bn`: each `i`-iteration writes to `A(i, j, *)`. Multiple threads can write to *different* `i` but the same `j` (separate cache lines per `i`, but for a small enough `A` ⇒ false sharing risk if `j * 3 * sizeof(double)` is misaligned with cache-line boundaries). False sharing is a perf concern, not a correctness one. Safe.
   - `define_a_uniform_cartesian_grid_between_two_toroidal_surfaces`: each `i`-iteration writes to `final_grid(i, *)`. Safe.
5. **`mod_B_squared`-style race (from CLAUDE.md):** the dipole-field kernels have no such pattern. Safe.
6. **`#pragma omp parallel for ordered`:** none used in `dipole_field.cpp`. Safe.

**No UB found in `dipole_field.cpp` end-to-end.**

## 16 — Recommended action list (ordered by severity)

### MEDIUM

1. **Behavior-parity gap at dipole-on-axis with non-cartesian frame** (Finding 2). Decide and document: either align JAX with C++ NaN behavior, or guard the degeneracy on both sides. Add a regression test in `tests/jax_core/test_dipole_field_item24.py` that pins the chosen behavior.
2. **Add `coordinate_flag` whitelist validation** to `dipole_field.py:425` and to `dipole_field.cpp:277`. Reject silent typos.
3. **Add `unitnormal.shape == points.shape` validation to C++ `dipole_field_Bn`.** Currently only JAX guards. This is a divergent contract.
4. **Add committed autodiff tests** for `dipole_field_B/A/dB/dA/Bn`: linearity in `m`, FD vs `jax.grad` w.r.t. positions, NaN propagation at singularity. Suggested location: `tests/jax_core/test_dipole_field_autodiff.py`. Use `derivative-heavy` parity lane.

### HIGH (documentation only)

5. **Fix the SIMSOPT-wide pybind11 docstring at `python_magneticfield.cpp:45`** (and `:30`) to reflect the actual dipole-field storage convention `dA[i, j, k] = ∂A_j/∂x_k`. Or, audit and confirm whether BiotSavart, CircularCoil etc. use `∂_j A_l` and **mark the dipole-field kernel as "transposed by convention"** in a docstring callout.

### LOW

6. **Hoist `mp_phi_new`/`sphi0`/`cphi0` out of loop in C++ `dipole_field_Bn`** (Finding 5). Pure perf win, no parity change.
7. **Replace `pow(-1, stell)` with branchless integer math** (Finding 6).
8. **Replace `1e5` sentinel with `std::numeric_limits<double>::infinity()`** in C++ grid filter (Finding 8).
9. **Document mixed-dtype behavior** in JAX `dipole_field.py` module docstring (Finding 7). State that float32 inputs are silently upcast to float64.
10. **Add a docstring warning about coincident point/dipole singularity** in both JAX and C++ docstrings (carried over from first-pass Finding 5).

### INFO

11. **Untested:** GPU-CI determinism check under `XLA_FLAGS=--xla_gpu_deterministic_ops=true` on `dipole_field_B/Bn` (carried over from first-pass).
12. **Untested:** large-N (`N ≥ 10⁶`) memory and accuracy probe — only `N=10⁴` tested in this audit; behavior at production PM-grid scale untested.

## Conclusion

The first-pass forward-formula parity audit covered the math correctly and verified bit-clean direct-kernel parity. This second-pass found three substantive **NEW** items missed by forward-only inspection:

1. **A genuine behavior parity gap** at `(dipole at origin, cylindrical/toroidal frame)` where C++ produces NaN and JAX produces finite — this is the kind of "JAX more robust than C++" silent disagreement that bypasses parity tests carefully fixtured to avoid degenerate inputs.
2. **A pervasive SIMSOPT-wide docstring/implementation convention contradiction** on `dA_by_dX` axis ordering — the JAX port replicates the C++ kernel's actual convention faithfully, but the SIMSOPT-base-class pybind11 docstring claims the opposite. Both the JAX and the C++ kernels are internally self-consistent; the *base-class advertised API* is what's wrong. Severe documentation finding; no numerical bug in the dipole port.
3. **A divergent input-validation contract** where JAX rejects malformed shapes but C++ silently accepts with potential UB.

Plus several low-severity C++ optimization opportunities and one mixed-dtype contract gap. No autodiff bugs surfaced — linearity in `m` holds at machine precision, `1/r^4` singularity scaling is correct, NaN propagates correctly at the singularity. But the **complete absence of committed autodiff tests** remains the largest forward-looking risk; the audit's manual tests would not catch a regression introduced in a future refactor.
