# Parity Audit 05 — 3D Regular-Grid Interpolation

**Audit timestamp:** 2026-05-16
**Auditor scope:** PRIORITY 5 — Item 13 (rectangular kernel) and item 15-sub (cylindrical wrapper)

## Files audited

| Role | File | Lines |
|------|------|-------|
| JAX kernel | `src/simsopt/jax_core/regular_grid_interp.py` | 660 |
| C++ kernel impl | `src/simsoptpp/regular_grid_interpolant_3d_impl.h` | 252 |
| C++ kernel decl | `src/simsoptpp/regular_grid_interpolant_3d.h` | 344 |
| C++ template insts (py) | `src/simsoptpp/regular_grid_interpolant_3d_py.cpp` | 8 |
| C++ template insts (c)  | `src/simsoptpp/regular_grid_interpolant_3d_c.cpp` | 11 |
| C++ binding | `src/simsoptpp/python_magneticfield.cpp` | (only RegularGridInterpolant3D registration at L72-81) |
| JAX cylindrical wrapper | `src/simsopt/jax_core/interpolated_field.py` | 511 |
| C++ cylindrical wrapper | `src/simsoptpp/magneticfield_interpolated.h` | 660+ (L1-211 reviewed) |
| Public CPU wrapper | `src/simsopt/field/magneticfieldclasses.py` | L847-916 |
| Public JAX wrapper | `src/simsopt/field/interpolated_field_jax.py` | 329 |
| JAX classifier | `src/simsopt/jax_core/surface_classifier.py` | (relevant excerpts) |
| Tests (item 13) | `tests/jax_core/test_regular_grid_interp_item13.py` | 478 |
| Tests (item 15-sub) | `tests/field/test_interpolated_field_jax_item15.py` | 479 |
| CPU oracle tests | `tests/field/test_interpolant.py` | 188 |
| Boozer interpolant tests | `tests/field/test_interpolated_boozer_field_jax.py` | 760 |

---

## Executive summary — top findings

1. **HIGH — Cell-locator sign asymmetry at lower-bound OOB.** The C++ kernel uses C-style `int(x)` truncation toward zero (`regular_grid_interpolant_3d_impl.h:78-80, 96-98`). JAX uses `jnp.floor(...).astype(int32)` which is floor (toward `-inf`) (`regular_grid_interp.py:510-512`). For an OOB query at `x < xmin` whose `(x - xmin)/(xmax - xmin)` evaluates to a *negative* fraction, C++ produces `xidx = 0` and `in_bounds_x = True`, while JAX produces `xidx = -1` and `in_bounds_x = False`. With `out_of_bounds_ok=True` C++ falls through to `evaluate_local` which then misses the cell map and returns silently (leaving `res` untouched). JAX correctly routes to NaN/zero. **The classifications diverge near `xmin` for any query that lands in `(xmin - hx, xmin)` due to floating-point noise even after the soft `_EPS_` clamp** — these are exactly the queries that the soft clamp was designed to hide.

2. **HIGH — `evaluate_local`'s "miss & leave-result-unchanged" semantics with `out_of_bounds_ok=True`.** When the locator picks a *valid* cell index that maps to a skipped cell, C++ `evaluate_local` (`regular_grid_interpolant_3d_impl.h:117-123`) returns early WITHOUT writing to `res`, preserving the caller's pre-existing buffer contents. The JAX kernel unconditionally writes zero (`regular_grid_interp.py:549-557, 564`). This mismatch is visible in `tests/field/test_interpolant.py:91-96` where the CPU oracle is asserted against a pre-populated buffer of `1.0` and verified to remain `1.0` for OOB. The JAX cross-oracle test (`tests/jax_core/test_regular_grid_interp_item13.py:186-196`) explicitly compares against zero instead — see "documented zero/leave-unchanged contract" comment at L334-337. The contract divergence is acknowledged in source comments but is not tested as a parity break.

3. **MEDIUM — Reduction-order difference between C++ FMA-stencil and JAX `jnp.einsum`.** The C++ inner loop is a hand-rolled triple FMA chain `sumk += val * pkz; sumj += sumk * pjy; sumi += sumj * pix` (`regular_grid_interpolant_3d_impl.h:185-197`), so component sums are accumulated in cell-major / k-fastest order. The JAX implementation uses `jnp.einsum("i,j,k,ijkl->l", pkx, pky, pkz, local_vals)` (`regular_grid_interp.py:550-557`) whose reduction order is XLA-chosen and not guaranteed to be Kahan-stable or i/j/k-fastest. For `degree=4, value_size=3` cubic-Lagrange cells this typically agrees to ~1e-13 (within the `direct_kernel` lane), but high-degree (>3) low-magnitude cells where two basis weights are nearly antisymmetric can produce 1 ULP / 1 LSB divergence per cell that the test suite does not currently exercise.

---

## Function-by-function parity matrix

| Concern | C++ symbol | JAX symbol | Parity | Notes |
|---------|------------|------------|--------|-------|
| Equispaced 1D nodes | `UniformInterpolationRule` (`.h:312-326`) | `UniformInterpolationRule` (`regular_grid_interp.py:66-84`) | OK | Same `i/degree` Python construction; both Python lists are float64. |
| Chebyshev 1D nodes | `ChebyshevInterpolationRule` (`.h:329-343`) | `ChebyshevInterpolationRule` (`regular_grid_interp.py:87-105`) | OK | Same `-0.5*cos(i*pi/degree)+0.5` recipe. |
| Lagrange denominators | `InterpolationRule::build_scalings` (`.h:30-37`) | `_build_scalings` (`regular_grid_interp.py:53-63`) | OK | Same nested loop, same accumulation order. |
| Basis function eval | `InterpolationRule::basis_fun` (`.h:44-52`) | `_basis_values` (`regular_grid_interp.py:433-455`) | DIVERGENT-RED-ORDER | C++ accumulates `res *= (x - nodes[i])` in `i=0,1,2,..., skip idx`. JAX builds a `(degree+1, degree+1)` mask matrix and uses `jnp.prod(factors, axis=0)`. XLA may reduce in any order. ULP-level diff. |
| Mesh node table | `linspace(min, max, n+1, true)` (`_impl.h:240-252`) | `np.linspace(min, max, n+1)` (`regular_grid_interp.py:354-356`) | OK | NumPy uses the standard endpoint formula identical to the C++ `(max-min)/(n-1)` step. |
| DOF coordinate table | `xdof[i*degree+j] = xmesh[i] + nodes[j]*hx` (`.h:213-227`) | Same expression (`regular_grid_interp.py:208-216`) | OK | Bit-identical when float64 inputs match. |
| Skip-cell logic (8 corners AND) | `.h:190-205` | `regular_grid_interp.py:381-391` | OK | Same 8-corner conjunction. |
| Cell locator | `int(nx*(x-xmin)/(xmax-xmin))` (`_impl.h:96-98`) | `jnp.floor(nx*(x-xmin)/(xmax-xmin)).astype(int32)` (`regular_grid_interp.py:510-512`) | **DIVERGENT** | See finding 1. C++ truncates toward zero, JAX floors toward `-inf`. |
| Soft boundary clamp `_EPS_=1e-13` | `_impl.h:88-94` | `regular_grid_interp.py:496-507` | OK semantically | Same value `1e-13`. But the inequalities differ subtly: C++ uses `if(x >= xmax) x -= _EPS_; else if (x <= xmin) x += _EPS_;` (`_impl.h:89-90`) — exclusive branches. JAX runs two independent `jnp.where` ops on the *clamped* state. The compositions agree for ordinary inputs. |
| OOB handling (`out_of_bounds_ok=False`) | `throw std::runtime_error(...)` (`_impl.h:99-106`) | `result = jnp.where(in_kept_cell, result, jnp.nan)` (`regular_grid_interp.py:563-565`) | INTENTIONALLY DIVERGENT | Documented at `regular_grid_interp.py:557-562`. JAX cannot raise from inside `jit`. NaN sentinel surfaces to host. |
| OOB handling (`out_of_bounds_ok=True`) | early return, leaves caller buffer untouched (`_impl.h:118-123`) | Returns zero (`regular_grid_interp.py:549-557`) | **DIVERGENT** | See finding 2. JAX cannot represent "leave-unchanged" inside a pure-functional kernel. |
| Skipped-cell handling | `all_local_vals_map.find(cell_idx) == end()` early return (`_impl.h:117-123`) | `sentinel_row` redirect → zero row (`regular_grid_interp.py:526-528, 549`) | OK in OOB-zero mode; **DIVERGENT** in OOB-strict mode | C++ raises in strict mode for both "outside" and "skipped"; JAX produces NaN for both. Comment at `regular_grid_interp.py:528-533` acknowledges the unification. |
| Tensor-product contraction | hand-rolled FMA loop (`_impl.h:148-200`) | `jnp.einsum("i,j,k,ijkl->l", ...)` (`regular_grid_interp.py:550-557`) | DIVERGENT-RED-ORDER | See finding 3. |
| Cell table padding | `padded_value_size = round_up(value_size, simdcount)` (`.h:295`) | None (uses exact `value_size`) | OK semantically | JAX has no SIMD lane-padding concept; result shape is exact `value_size`. Does not affect numerical parity. |
| `estimate_error` | random sample, RMS over `value_size`, mean ± std bracket (`_impl.h:204-236`) | `regular_grid_interp.py:618-649` | RNG-DIVERGENT | C++ uses `std::default_random_engine` (LCG) with default seed; JAX uses `np.random.default_rng(seed)` (PCG64). Brackets are different. The JAX test only asserts the bracket is finite / `<atol` for the polynomial-exactness case. |

---

## Detailed findings

### Finding A (HIGH, finding 1): C++ `int(...)` vs JAX `jnp.floor(...)` cell-locator sign asymmetry

**Locations:**

- C++ (`src/simsoptpp/regular_grid_interpolant_3d_impl.h:96-98`):
  ```cpp
  int xidx = int(nx*(x-xmin)/(xmax-xmin)); // find idx so that xmesh[xidx] <= x <= xs[xidx+1]
  int yidx = int(ny*(y-ymin)/(ymax-ymin));
  int zidx = int(nz*(z-zmin)/(zmax-zmin));
  ```

- JAX (`src/simsopt/jax_core/regular_grid_interp.py:510-512`):
  ```python
  xidx_raw = jnp.floor(nx * (x_clamped - xmin) / (xmax - xmin)).astype(jnp.int32)
  yidx_raw = jnp.floor(ny * (y_clamped - ymin) / (ymax - ymin)).astype(jnp.int32)
  zidx_raw = jnp.floor(nz * (z_clamped - zmin) / (zmax - zmin)).astype(jnp.int32)
  ```

**Mechanism.** The C++ standard says `int(d)` truncates toward zero for finite `d`. For a query at `x = xmin - epsilon` where the soft clamp at `_impl.h:89-90` did *not* fire (because `epsilon > _EPS_ = 1e-13`), the C++ computes `int(nx * negative_fraction) = 0`, treating the query as if it were in cell 0. The subsequent OOB check `if(xidx < 0 || xidx >= nx)` passes, so C++ silently extrapolates from the cell-0 polynomial into the OOB region.

JAX computes `jnp.floor(nx * negative_fraction).astype(int32) = -1` (for any negative fraction whose magnitude exceeds `1/nx`), then the in-bounds check at `regular_grid_interp.py:514-517` rejects, so JAX routes through NaN or zero per `out_of_bounds_ok`.

**Impact on tracing.** This is consumed transitively by `tests/jax_core/test_tracing_jax_levelset_events.py:370-371` and by `surface_classifier.py:88` through `evaluate_batch`. The classifier's own `in_bounds` short-circuit at `surface_classifier.py:90-98` masks the divergence for the classifier case. The bare-interpolant case (e.g. `interpolated_field_B` for a particle that just left the `r > rmax` boundary) does not.

**Severity rationale.** This affects particle-loss classification at the boundary. A particle that crosses below `rmin` will be treated as "extrapolated from cell 0" by the CPU and "lost / NaN" by JAX, producing divergent step-by-step trajectories near the wall. The CPU behavior is closer to "extrapolation" semantics — which IS the upstream contract — so the JAX behavior is actually *more correct* in the strict sense, but the parity claim cannot hold.

**Recommended action:**
- Implement C-style truncation in JAX: `xidx_raw = jnp.trunc(...).astype(jnp.int32)` to match C++ exactly. This still maps `-0.5` to `0`, so the in-bounds check at L514-517 will *also* need to allow `-1` through to the cell-0 polynomial (which is what C++ does implicitly). Or, equivalently, drop the `in_bounds_x` rejection for the case where the float index is in `[-1, 0)` and force `xidx = 0`.
- Alternative: add an explicit parity test that constructs a 1D-cell interpolant with `nx = 2`, samples at `x = xmin - 0.1 * hx`, and asserts CPU = JAX. This will fail today.

---

### Finding B (HIGH, finding 2): "Leave-result-unchanged" on OOB-OK / skipped path

**Locations:**

- C++ (`src/simsoptpp/regular_grid_interpolant_3d_impl.h:117-123`):
  ```cpp
  auto got = all_local_vals_map.find(cell_idx);
  if (got == all_local_vals_map.end()) {
      if(out_of_bounds_ok)
          return;
      else
          throw std::runtime_error(...);
  }
  ```

  The `return` is `void`; the `res` pointer is NOT zeroed. Whatever was previously written into the caller's buffer remains.

- JAX (`src/simsopt/jax_core/regular_grid_interp.py:549, 564`):
  ```python
  local_vals = cell_table[row_idx]  # row_idx = sentinel_row when OOB or skipped
  result = jnp.einsum("i,j,k,ijkl->l", pkx, pky, pkz, local_vals, optimize=True)
  ...
  if not out_of_bounds_ok:
      result = jnp.where(in_kept_cell, result, jnp.nan)
  return result
  ```

  The sentinel row at `cell_table[-1]` is forced to zero (`regular_grid_interp.py:294-297`). The einsum over a zero local-vals cube returns zero regardless of basis values. So `out_of_bounds_ok=True` unconditionally produces zero.

**Witness.** Upstream `tests/field/test_interpolant.py:91-96` asserts that the OOB-OK CPU path leaves a pre-populated buffer of `1.0` unchanged:
```python
fhxyz = np.ones((nsamples, dim))
interpolant.evaluate_batch(xyz, fhxyz)
assert np.allclose(fhxyz, 1., atol=1e-14, rtol=1e-14)
```

The JAX item-13 test at `tests/jax_core/test_regular_grid_interp_item13.py:186-196` works around this by asserting against zero instead:
```python
lax_result = np.asarray(evaluate_batch(spec_lax, xyz_oob))
np.testing.assert_allclose(lax_result, np.zeros_like(lax_result), ...)
```

The contract divergence is acknowledged in source comments at `regular_grid_interp.py:184-200` ("Slots for skipped DOFs stay at zero" ... "they will never be read because cell_to_row routes skipped cells away") and at `regular_grid_interp.py:557-562` ("the C++ binding leaves the caller buffer unchanged, which is not representable in a pure-functional kernel"). The skipped-cell zero-test at `tests/jax_core/test_regular_grid_interp_item13.py:266-272` matches the documented JAX semantic, NOT the upstream contract.

**Impact on consumers.** Any caller that
1. Pre-populates the result buffer before calling `evaluate_batch`, AND
2. Has a real CPU codepath that depends on the pre-population surviving,

will observe a divergence. The JAX wrapper `InterpolatedFieldJAX` at `interpolated_field_jax.py:214-220` allocates a fresh buffer per call so the divergence is invisible to that wrapper. The skip-mask cross-oracle test at `tests/jax_core/test_regular_grid_interp_item13.py:330-404` only checks "kept" points, NOT skipped points.

**Severity rationale.** This is HIGH because the semantic is documented in CPU sources, exercised by the CPU oracle test, and the JAX behavior diverges *silently*. If a downstream caller (existing or future) ever interleaves OOB and in-domain queries in a single `evaluate_batch` and pre-populates the buffer, JAX will overwrite valid pre-population with zeros while CPU will preserve it.

**Recommended action:**
- Document the divergence in the `RegularGridInterpolant3DSpec` docstring as an explicit "non-portable" semantic; mark it MEDIUM-portable rather than CPU-byte-identical.
- Add a parity test that exercises the skip-and-mixed-batch path: build a CPU oracle, evaluate at a mixed batch of kept + skipped points with a pre-populated `100.0` buffer, and document that the JAX path *cannot* reproduce this. The existing test at `tests/jax_core/test_regular_grid_interp_item13.py:266-272` should be renamed `test_skip_region_yields_zero_inside_skipped_cells_in_jax_lane` to clarify it is a JAX-lane semantic, not a CPU parity test.

---

### Finding C (MEDIUM, finding 3): Tensor-product reduction order — FMA vs `jnp.einsum`

**Locations:**

- C++ inner loop (`src/simsoptpp/regular_grid_interpolant_3d_impl.h:179-199`, non-SIMD branch):
  ```cpp
  for(int l=0; l<padded_value_size; l += simdcount) {
      double sumi(0.);
      for (int i = 0; i < degree+1; ++i) {
          double sumj(0.);
          for (int j = 0; j < degree+1; ++j) {
              double sumk(0.);
              for (int k = 0; k < degree+1; ++k) {
                  sumk += (*val_ptr) * pkzs[k]; val_ptr += padded_value_size;
              }
              sumj += sumk * pkys[j];
          }
          sumi += sumj * pkxs[i];
      }
      res[l] = sumi;
  }
  ```

- C++ SIMD branch (`_impl.h:150-172`): uses `xsimd::fma` — fused multiply-add gives one rounding instead of two, but the loop order is the same: k-fastest → j → i → l.

- JAX (`src/simsopt/jax_core/regular_grid_interp.py:550-557`):
  ```python
  local_vals = cell_table[row_idx]  # (degree+1, degree+1, degree+1, value_size)
  result = jnp.einsum(
      "i,j,k,ijkl->l",
      pkx, pky, pkz, local_vals,
      optimize=True,
  )
  ```

**Analysis.**
1. The C++ loop is k-fastest with sequential left-to-right accumulation. Three temporaries (`sumk`, `sumj`, `sumi`) carry the partial product.
2. The C++ SIMD path uses FMAs (one rounding per multiply-add), which is *different from* the non-SIMD path (two roundings: one for `*`, one for `+=`). So C++ already has two intra-language reduction-order paths that may differ at the 1 ULP level.
3. JAX `jnp.einsum("i,j,k,ijkl->l", ...)` is an XLA HLO `dot` that XLA may decompose as a sequence of `reduce` ops in any order it chooses; the `optimize=True` flag explicitly *allows* XLA to choose. There is no guarantee that the JAX summation matches either of the two C++ paths bit-for-bit.

**Empirical evidence.** The `test_cpp_cross_oracle` test (`tests/jax_core/test_regular_grid_interp_item13.py:277-327`) reports passing at `_DIRECT_KERNEL` tolerance (`rtol=1e-10, atol=1e-13` per `benchmarks/validation_ladder_contract.py`). It is parametrized over `dim in {1,3,4}, degree in {1,2,3}`. The `degree=4` case from upstream `test_interpolant.py:62` is NOT in the parametrization. The chosen sample density is 128 points across `_DEFAULT_XRANGE = (1.0, 4.0, 20)` so cell residuals are typically O(1e-13). Degree-4 large-magnitude polynomials with near-cancellation cells can grow this to O(1e-11), still under `direct_kernel`'s `atol=1e-10` but not at the `1e-13` floor the CPU test asserts (`test_interpolant.py:57`).

**Severity rationale.** MEDIUM because the divergence sits under the published `direct_kernel` tolerance budget today. It can break a future tighter parity gate if anyone introduces one. It is also a deviation from the canonical "byte-identity CPU↔JAX" claim that the CLAUDE.md parity-ladder SSOT carves out only for same-state, single-machine evaluations.

**Recommended action:**
- Add a degree-4 case to the cross-oracle parametrization (matches the CPU oracle's existing coverage at `test_interpolant.py:62`).
- For the byte-identity gate (if/when required for interpolated fields), implement a reduction-order-preserving JAX path: replace the einsum with explicit nested `jax.lax.scan` so the i/j/k summation order matches C++. Per profiling guidance in `_impl.h:147-149`, an optimization to barycentric form is also on the table.

---

### Finding D (MEDIUM): Locator divisor uses `(xmax-xmin)` instead of `hx`

**Locations:**

- C++ (`src/simsoptpp/regular_grid_interpolant_3d_impl.h:96`): `int xidx = int(nx*(x-xmin)/(xmax-xmin));`
- JAX (`src/simsopt/jax_core/regular_grid_interp.py:510`): `xidx_raw = jnp.floor(nx * (x_clamped - xmin) / (xmax - xmin)).astype(jnp.int32)`

**Analysis.** Both kernels compute `nx * (x - xmin) / (xmax - xmin)` rather than the algebraically-equivalent `(x - xmin) / hx`. These are NOT bit-identical because `hx = (xmax - xmin) / nx` was already pre-divided. The chosen form does match across both kernels, so this is a parity-positive choice. Mentioned here only for completeness: the local-coordinate computation at the next step *does* divide by `hx` (`_impl.h:107-109` and `regular_grid_interp.py:535-537`), so the two computations are not perfectly self-consistent in either kernel. This is benign for in-cell evaluation but means that a query exactly at a mesh node (e.g. `x = xmesh[xidx+1]`) may land in cell `xidx+1` with `xlocal ≈ 0` instead of cell `xidx` with `xlocal = 1`, depending on which formula is used. Since both kernels use the same formula, the choice is shared. Net: OK.

---

### Finding E (LOW): Sentinel row vs unordered_map miss — different cache locality, same result

**Locations:**

- C++ (`src/simsoptpp/regular_grid_interpolant_3d_impl.h:117-118`):
  ```cpp
  auto got = all_local_vals_map.find(cell_idx);
  if (got == all_local_vals_map.end()) { ... }
  ```

  The C++ uses an `std::unordered_map<int, AlignedPaddedVec>` keyed by flat cell index. Skipped cells have no entry, so the lookup fails and the early-return fires.

- JAX (`src/simsopt/jax_core/regular_grid_interp.py:255-298, 526-533`): packs all retained cells into a contiguous `(cells_to_keep+1, degree+1, ..., value_size)` array. A `cell_to_row[cell_idx]` int32 table maps a flat cell index either to a row in the data table or to a sentinel row (the last row, forced to zero). The `gather` is then a single indexed read.

**Analysis.** The JAX layout is far better for GPU memory traffic (a single coalesced gather instead of a hash lookup). The behavioral difference is finding-B above. Otherwise identical.

---

## (a) Basis function parity

The Lagrange basis is

  p_i(x) = scalings[i] · Π_{j != i} (x - nodes[j])

Both kernels compute this. C++ does it scalar-by-scalar in a tight loop. JAX builds a `(degree+1, degree+1)` "exclude i=j" mask matrix `factors[i, idx] = (diffs[i] if i != idx else 1.0)` and does a column-wise `jnp.prod`. The mask trick is mathematically equivalent (each column is missing exactly one factor, replaced by `1.0`) but the reduction order over `axis=0` is not the C++ left-to-right.

For `degree <= 3` the difference is below 1 ULP. For `degree >= 4` with widely-varying node spacings (Chebyshev) it can hit 1 ULP. The polynomial-exactness test at `tests/jax_core/test_regular_grid_interp_item13.py:72-105` covers `degree in [1, 2, 3, 4]` but the cross-oracle test at L275-327 only covers `degree in [1, 2, 3]`. **Coverage gap noted under "Test coverage gaps" below.**

The basis values themselves are computed once per axis per query (`pkx, pky, pkz`), so there is no inner-loop amplification of any ULP-level mismatch — each component of the final result is at most `(degree+1)^3` such ULPs accumulated, which is `O(1e-13)` for degree 4 in float64. Within `direct_kernel` budget.

## (b) Cell-locator parity

See finding A (HIGH). The two divergences are:

1. C++ `int(...)` truncates toward zero, JAX `jnp.floor(...)` truncates toward `-inf`. Affects negative fractions at the lower boundary.
2. JAX adds an explicit `in_bounds_x = (xidx_raw >= 0) & (xidx_raw < nx)` check (`regular_grid_interp.py:514-517`) that C++ does NOT have when `out_of_bounds_ok=True`. C++ relies on the soft `_EPS_` clamp + `int(...)` to silently land "barely-OOB" queries in cell 0.

JAX clips the locator to a valid range (`regular_grid_interp.py:521-523`) before the gather to keep the gather safe even when `in_bounds = False`. This is a valid JAX pattern but means the sentinel-row redirect at `regular_grid_interp.py:528` is the actual OOB sentinel; the cell-0 polynomial is never evaluated outside the sentinel branch.

## (c) Boundary / OOB parity

| Case | C++ behavior | JAX behavior | Parity |
|------|--------------|--------------|--------|
| `x = xmax` exactly | `_EPS_` clamp triggers, falls in cell `nx-1` (`_impl.h:89`) | Same `_EPS_` clamp, falls in cell `nx-1` | OK |
| `x = xmin` exactly | `_EPS_` clamp triggers, falls in cell `0` (`_impl.h:90`) | Same `_EPS_` clamp, falls in cell `0` | OK |
| `x = xmax + 1e-12` | clamp does not fire (epsilon larger than `_EPS_`), `int(nx*1.0+epsilon) = nx`, OOB check fires (`_impl.h:100-101`) | clamp does not fire, `floor(nx*1.0+epsilon) = nx`, in-bounds check fires | OK |
| `x = xmin - 1e-12` | clamp does not fire, `int(nx*negative)` = `0`, `xidx in [0, nx)` so OOB check passes silently | clamp does not fire, `floor(nx*negative) = -1`, in-bounds check rejects | **DIVERGENT** (finding A) |
| In-domain on skipped cell, OOB-OK=True | early return, `res` unchanged | sentinel-row redirect → `result = 0` | **DIVERGENT** (finding B) |
| In-domain on skipped cell, OOB-OK=False | C++ throws | JAX returns NaN | OK semantically (intentional, documented) |
| OOB, OOB-OK=True | early return, `res` unchanged | sentinel-row redirect → `result = 0` | **DIVERGENT** (finding B) |
| OOB, OOB-OK=False | C++ throws | JAX returns NaN | OK semantically (intentional, documented) |

## (d) Derivative parity

**There is no analytic derivative kernel in either C++ or JAX.** Both kernels expose only the value `evaluate_batch`. The JAX kernel does NOT implement `dF/dx`, `dF/dy`, `dF/dz` directly.

Downstream consumers requiring derivatives use the **separately-interpolated** `GradAbsB` table:

- `interp_GradAbsB` in `magneticfield_interpolated.h:48` (C++) and `GradAbsB_spec` in `interpolated_field.py:91` (JAX).
- The wrappers sample `field.GradAbsB_cyl()` on the same mesh as `B_cyl()` and build a second interpolant whose values *are* the gradient components.

**This is correct upstream behavior** (the interpolant of a derivative is not the same as the derivative of an interpolant). Both kernels match. There is NO `dB/dX` for the Cartesian Jacobian, which is documented at `interpolated_field_jax.py:21-27`:
> The wrapper deliberately does NOT implement `_dB_by_dX_impl`. The CPU `InterpolatedField` does not implement it either (it raises a runtime error inside the C++ binding).

**JAX autodiff through `evaluate_batch`.** The JAX kernel is `jax.jit`-traceable and (by inspection) is also `jax.grad` / `jax.jacfwd` traceable. The basis-values function `_basis_values` uses smooth `jnp.prod`. The `jnp.floor` cell locator is non-differentiable but is wrapped in `astype(int32)` so JAX treats the gather as constant w.r.t. the input coordinates — the derivative therefore captures only the *intra-cell* basis-coordinate derivative, NOT cell-boundary jumps. This is mathematically what one wants for a degree-`d` Lagrange interpolant in the open interior of each cell. **The C++ has no such autodiff path.**

If a downstream JAX consumer ever differentiates `interpolated_field_B` w.r.t. its query point, the result is the analytic gradient of the piecewise polynomial — which is NOT the same as evaluating the `GradAbsB` interpolant. The two will agree only for analytic source fields whose gradient table was built from `nabla |B|` analytically. **There is no test covering this path.**

## (e) Classifier parity

The `surface_classifier.signed_distance_to_cartesian_classifier` (`src/simsopt/jax_core/surface_classifier.py:40-103`):

1. Maps `(x, y, z) → (r, phi, z)` with `r = sqrt(x² + y²)`, `phi = mod(arctan2(y, x), 2π)`, `z = z` (L81-86).
2. Calls `evaluate_batch(interpolant_spec, rphiz)` to get the signed distance (L88).
3. Independently checks domain inclusion via `r in [xmin, xmax]`, `phi in [ymin, ymax]`, `z in [zmin, zmax]` (L90-97).
4. Returns `sign(dist) if in_bounds else -1.0` via `jnp.sign(jnp.where(in_bounds, dist_flat, -1.0))` (L98).

The CPU `SurfaceClassifier` (`opensource/simsopt/src/simsopt/geo/surface.py:925`) uses `simsopt.LevelsetStoppingCriterion` which calls the C++ kernel's `evaluate` and returns the sign. The CPU path inherits the **finding A** divergence at `r < rmin` or `r > rmax` because the C++ `int(...)` will silently extrapolate. The JAX classifier sidesteps this by performing its own `in_bounds` check BEFORE consulting the interpolant value. **This is a parity-positive divergence but it means JAX and CPU classifiers disagree at the `r = rmin` boundary by exactly the difference between "extrapolated signed distance" (CPU) and "always -1" (JAX).**

For tracing: the classifier-style "particle is lost" decision will be different at the wall by exactly the resolution of one cell-edge. This is acceptable in practice but should be noted in the parity ladder as a `tracing.classifier_boundary` deviation.

## (f) Batched / vmap parity

| Aspect | C++ | JAX |
|--------|-----|-----|
| Outer batching | `for (int i = 0; i < npoints; ++i)` (`_impl.h:62-65`), serial host loop | `jax.vmap(evaluate_one)(xyz)` (`regular_grid_interp.py:567`) |
| Inner SIMD | `xsimd::fma` over `value_size` lanes if compiled with `USE_XSIMD` (`_impl.h:126-172`) | XLA-chosen vectorization over `value_size` and batch axis |
| Per-point state | `pkxs`, `pkys`, `pkzs` heap-allocated, member-state buffers (shared!) (`.h:115`) | Closure-local stack arrays per `evaluate_one` invocation |
| Determinism | The C++ member-state buffer (`pkxs` etc.) is mutated per call. **This is not thread-safe** in the C++ kernel; concurrent `evaluate_batch` calls would race. | JAX `vmap` is functional, all per-point state is local to the traced function. |

**Note on threading.** The C++ kernel's `pkxs`/`pkys`/`pkzs` member buffers (declared at `regular_grid_interpolant_3d.h:115`) are mutated inside `evaluate_local`. The `evaluate_batch` driver loop at `_impl.h:62-65` is serial, so this is benign for the canonical CPU path, but any future OpenMP parallelization of the batch loop would race on these buffers. JAX has no such issue.

**`interpolate_batch` parity.** The JAX `build_regular_grid_interpolant_3d` evaluates `f` on the entire retained DOF set in a single call (`regular_grid_interp.py:274`), while C++ chunks into batches of size 16384 (`_impl.h:15-29`). The semantic result is identical assuming `f` is purely functional. This is a parity-neutral implementation choice.

---

## Symmetry-wrapper parity (item 15-sub)

The cylindrical wrapper in `src/simsopt/jax_core/interpolated_field.py` mirrors the C++ `exploit_symmetries_points`, `apply_symmetries_to_B_cyl`, and `apply_symmetries_to_GradAbsB_cyl` in `src/simsoptpp/magneticfield_interpolated.h:90-136`.

| Concern | C++ | JAX | Parity |
|---------|-----|-----|--------|
| Stellsym detection | `if(z < 0 && stellsym)` (`magneticfield_interpolated.h:102`) | `reflect = stellsym & (z < 0.0)` (`interpolated_field.py:158`) | OK |
| z reflection | `z = -z` (L103) | `z_pre = jnp.where(reflect, -z, z)` (L159) | OK |
| phi reflection | `phi = 2*M_PI - phi` (L104) | `phi_pre = jnp.where(reflect, 2.0*pi - phi, phi)` (L160) | OK |
| nfp fold | `int(phi/period); phi -= phi_mult * period` (L105-106) — C-style truncation again! | `jnp.where(phi<0, phi+2pi, phi); jnp.mod(phi_wrapped, period)` (L132-133) | **Subtle** — different math, see below |
| `B_r` sign flip | `field(i, 0) = -field(i, 0)` if symmetries[i] (L122-125) | `B_r = B_cyl_fold[:, 0] * sign_br` (L174) | OK |
| `GradAbsB` flip | indices 1, 2 (L130-135) | indices 1, 2 (L189-191) | OK |
| Output rotation | `cos(phi)*B_r - sin(phi)*B_phi` (L72-74) using ORIGINAL phi | `cos(phi) * F_r - sin(phi) * F_phi` (L207-212) using ORIGINAL phi | OK |

**The nfp-fold subtlety.** C++ (`magneticfield_interpolated.h:105-106`) does:

```cpp
int phi_mult = int(phi/period);
phi = phi - phi_mult * period;
```

This is C-style truncation. For a `phi` that came out of the stellsym branch with `phi_pre = 2π - phi_original` and `phi_original` close to 0, the post-reflection value is close to `2π`, and `int(2π / period)` may be `nfp` (rounding into the last period) instead of `nfp - 1`. The subsequent subtraction puts `phi` slightly *negative* — which then drives `int(nx * negative)` in the locator to 0 (C++) but to `-1` (JAX) under finding A.

JAX (`interpolated_field.py:132-133`) uses:
```python
phi_wrapped = jnp.where(phi < 0.0, phi + 2.0 * jnp.pi, phi)
return jnp.mod(phi_wrapped, period)
```

This is well-defined for negative inputs because of the explicit `where`, and `jnp.mod` always returns in `[0, period)`. For `phi_pre` slightly larger than `2π` due to FP roundoff at the stellsym reflection, `jnp.mod` correctly wraps to a small positive remainder.

**Severity rationale.** This compounds finding A. A query at `(x, y, z) = (R0, 0, -small)` with stellsym and nfp > 1 will:
- in C++: get phi-reflected to `~2π`, then `int(2π/period) ≈ nfp` so post-mod phi is slightly negative, then locator returns `phi_idx = 0` (C-truncation), evaluates cell 0.
- in JAX: get phi-reflected to `~2π`, then `jnp.mod` returns slightly positive, locator returns `phi_idx = 0` (floor), evaluates cell 0.

These can agree for typical query phases but the worst-case ULP-level path is different. Currently the parity-test fixtures avoid the boundary cases (`tests/field/test_interpolated_field_jax_item15.py:124-133` margins `0.02` away from `_R_RANGE` endpoints).

---

## Test coverage gaps

1. **No `degree=4` cross-oracle test for the rectangular kernel.** `tests/jax_core/test_regular_grid_interp_item13.py:275-276` only parametrizes `degree in [1, 2, 3]`. The CPU oracle test at `tests/field/test_interpolant.py:62` covers `degree in [1, 2, 3, 4]`. **Recommend** adding `degree=4` to the cross-oracle parametrization.
2. **No locator-boundary parity test.** Finding A's mismatch at `x = xmin - epsilon` is invisible to the current test suite. **Recommend** adding a test that builds a small 2-cell interpolant and asserts CPU vs JAX at `x in [xmin - 0.5*hx, xmin - 1e-12, xmin, xmin + 1e-12]`.
3. **No skip-mask + OOB-OK pre-populated-buffer test.** Finding B's "leave-result-unchanged" CPU semantic vs JAX's zero semantic is acknowledged in source comments but not asserted as a documented parity gap.
4. **No Chebyshev cross-oracle test.** `test_uniform_vs_chebyshev_nodes` (`tests/jax_core/test_regular_grid_interp_item13.py:108-141`) asserts JAX-vs-closed-form for both rules. The cross-oracle test (`test_cpp_cross_oracle`) only uses `UniformInterpolationRule`. **Recommend** parametrizing the cross-oracle test over both rule factories.
5. **No autodiff-through-`evaluate_batch` test.** The JAX kernel is differentiable in the open interior of each cell; this is implicitly exercised by `boozersurface_jax` callers but not directly. **Recommend** adding a test that calls `jax.grad(lambda x: evaluate_batch(spec, x[None])[0, 0])(point)` and checks against analytic gradient for a linear separable polynomial.
6. **No stellsym + nfp-fold boundary test.** The integration tests at `tests/field/test_interpolated_field_jax_item15.py:286-318` cover combined fold but with safe-margin sample distributions. The nfp-fold subtlety described above is not exercised.
7. **No `estimate_error` cross-oracle test.** The JAX `estimate_error` uses different RNG (PCG64) than C++ (`std::default_random_engine` LCG), so direct numeric comparison is impossible. The CPU side has no `estimate_error` test either.
8. **No high-multiplicity (large `value_size`) cross-oracle test.** `value_size in [1, 3, 4]` is the cross-oracle parametrization; the C++ uses `padded_value_size` SIMD lanes when `value_size > simdcount` so `value_size in [5, 8]` would exercise different SIMD code-paths. **Recommend** adding `value_size in [5, 8]` to the cross-oracle parametrization.

---

## Recommended actions ordered by severity

1. **HIGH (Finding A).** Resolve the cell-locator divergence. Two options:
   - Make JAX match C++: use `jnp.trunc(...).astype(int32)` and relax the in-bounds check for `[-1, 0)` to land in cell 0 (matches C++ silent extrapolation).
   - OR document the divergence as intentional (JAX is strictly correct), and tighten the `interpolated_field_jax` wrapper to reject `r < rmin` / `phi < phimin` / `z < zmin` queries explicitly before they reach the kernel.
   Add the locator-boundary parity test from coverage gap #2.

2. **HIGH (Finding B).** Either:
   - Pre-zero the caller buffer on CPU before `evaluate_batch` to make the CPU and JAX semantics agree on the "zero result for OOB-OK" path. This is an upstream change to `InterpolatedField::_B_cyl_impl` (`magneticfield_interpolated.h:38-43`) — pass a freshly-zeroed `B_cyl` to `evaluate_batch`.
   - OR document the divergence as a known semantic gap; add an explicit parity test that demonstrates the gap and pins the JAX behavior; rename the existing `test_oob_behavior_returns_nan_when_strict` lax-mode assertion to reflect the JAX-only contract.

3. **MEDIUM (Finding C).** Add `degree=4` and `value_size in [5, 8]` to the cross-oracle parametrization. If the test passes within `direct_kernel` tolerance, document the einsum-vs-FMA ULP budget. If it fails, swap the einsum for a `jax.lax.scan`-based explicit k/j/i reduction.

4. **MEDIUM (nfp-fold subtlety).** Add an item-15-sub test that exercises `phi` close to `2π/nfp` with stellsym enabled; assert CPU vs JAX within `direct_kernel` tolerance.

5. **LOW (coverage gaps 4, 5, 6, 7).** Add the suggested tests. None of these are urgent but they close the audit trail for future tightening.

6. **INFO (Finding D, E).** No action required. Documented for completeness.

---

## Cross-references for downstream priorities

- **Priority 4 (tracing).** The Cartesian particle drivers in `simsopt/jax_core/tracing.py` consume the interpolated field through `interpolated_field_B`. Findings A and B propagate into tracing decisions at the wall and at skipped cells. The classifier in `surface_classifier.py:88-101` partially mitigates finding A by short-circuiting on in-bounds, but the bare `interpolated_field_B` call inside the RHS does not.
- **Boozer interpolant.** `src/simsopt/jax_core/interpolated_boozer_field.py` builds 16+ separate `RegularGridInterpolant3DSpec` instances (one per scalar field component). Each inherits findings A, B, C. Tests at `tests/field/test_interpolated_boozer_field_jax.py` (760 lines) cover the cylindrical wrapper but inherit the same kernel-level test coverage gaps.
