# PRIORITY 8 — Wireframe Magnetic-Field DEEPER Parity Audit (Second Pass)

**Audit timestamp:** 2026-05-16
**Scope:** edge-case, dtype-overflow, mutation, autodiff, UB, and consumer-side
parity issues that a forward-formula audit would systematically miss.

**Reference forward-pass audit:**
`.artifacts/parity_audit_2026-05-16/08_wireframe_field.md` established
op-for-op bit-identity of the closed-form Biot-Savart formula
(`_MU0_OVER_4PI = 1e-7` = C++ `fak`, JAX `jnp.cross` = right-hand-rule
C++ helpers, `lax.scan` matches C++ `axpy_array` order, the 9-component
analytic Jacobian agrees line-by-line). This pass focuses on what
forward-formula parity cannot catch: non-numerical contract surfaces,
dtype/mutation/UB hazards, and downstream consumer compatibility.

## Top findings by severity

| # | Severity | Title | Locations |
| - | -------- | ----- | --------- |
| 1 | **HIGH** | C++ uses variable-length arrays (VLAs, non-standard C++); pointer aliasing potential | `magneticfield_wireframe.cpp:39-40` |
| 2 | **HIGH** | int32 narrowing at JAX/C++ ingress with no overflow guard; ToroidalWireframe emits int64 | `wireframefield_jax.py:26`, `wireframe.py:115`, `wireframe_toroidal.py:163-164` |
| 3 | **MEDIUM** | `WireframeFieldJAX` does not invalidate the per-segment contribution cache when `set_points()` changes | `wireframefield_jax.py:60-70,107-120` |
| 4 | **MEDIUM** | `WireframeFieldJAX` is not a `MagneticFieldSum`-compatible JAX-native field for `B_vjp`; mixing with another JAX-native field triggers no error but silently lacks VJP | `magneticfield.py:308-309`, `wireframefield_jax.py:47` |
| 5 | **MEDIUM** | C++ `set_array_to_zero(ddB)` writes through the dummy `_dummyhess` reference even at `derivatives=0`; benign but wasteful and obscures the intent | `magneticfield_wireframe.cpp:17-20, 35` |
| 6 | **MEDIUM** | JAX rank-1 `points` input (single-point) silently mis-broadcasts via `factor[:, None]`; C++ would assert via `xt::pytensor` rank check | `wireframe.py:133, 188-195` |
| 7 | **MEDIUM** | `dB_by_dsegmentcurrents(1)` is parity-tested but the JAX path internally returns B contributions, NOT dB contributions; CPU and JAX *both* return the B-cache regardless of argument; semantic ambiguity, not divergence | `wireframefield_jax.py:94-120`, `wireframefield.py:30-61`, test:`test_wireframefield_jax_item30.py:79-84` |
| 8 | **LOW** | Singular-regime behaviour: division-by-zero at `r = a`, `r = b`, or `r` on the segment is shared CPU/JAX behaviour, but no parity test locks down the contract | `wireframe.py:64-75`, no test |
| 9 | **LOW** | Cross-machine fp reproducibility for the contribution path is bounded only empirically; the `jnp.sum` reduction order is *not* guaranteed by JAX, so different XLA backends could differ | `wireframe.py:261-268, 297-304` |
| 10 | **LOW** | `seg_signs` accepts both list and ndarray; no shape/length validation that `len(seg_signs) == nodes.shape[0]` | `wireframe.py:468, 246`, no test |
| 11 | **LOW** | `WireframeFieldJAX` stores reference to `self.wireframe`; mutating the wireframe after construction does not invalidate the device snapshot (intentional and tested), but `dB_by_dsegmentcurrents` then accesses `self.wireframe.n_segments` (live), creating a divergence point | `wireframefield_jax.py:118-119` |
| 12 | **INFO** | `wireframe_segment_dB_by_dX_contributions` is publicly exported (`__all__`), has no in-tree consumer, and no direct C++ parity test (only structurally validated through the `B_and_dB` combined path) | `wireframe.py:100, 273-306` |
| 13 | **INFO** | Test fixture topologies are tiny (10-node closed loop, `n_phi=4, n_theta=6` torus with 48 segments); realistic wireframe topologies have 1000s of segments and are not exercised | `tests/jax_core/test_wireframe*.py` |
| 14 | **INFO** | Layout-convention `dB[p, k, m]` (component-first) deliberately differs from the abstract simsopt-jax `[p, j, l]` (derivative-first) convention; no downstream consumer currently mixes them, but the lint risk is real | `wireframe.py:27-40`, contrast with `biot_savart_jax.py` |

## Detailed findings

### 1. HIGH — C++ variable-length arrays (`magneticfield_wireframe.cpp:39-40`)

```cpp
// Store pointers to the nodes array for each half period (in nodes vector)
double* halfPrd_ptr[nHalfPrds];        // VLA (non-standard ISO C++)
double seg_signs[nHalfPrds];           // VLA (non-standard ISO C++)
```

`nHalfPrds = this->nodes.size()` is a runtime value (`magneticfield_wireframe.cpp:22`), so this is a C99-style VLA, **not** valid ISO C++. GCC accepts it under `-Wno-vla`; clang accepts it; MSVC rejects it. The xsimd codebase compiles with both GCC and clang and these lines are silent (no diagnostic surfaced).

**Risk profile:**
* If a hostile or accidental caller pushes `nHalfPrds > 1024 * 256` (the typical stack-size limit / 8 bytes), stack overflow with no error path. The JAX path never has this issue — `seg_signs` is `(nHalfPrds,)`-shaped heap-allocated.
* MSVC builds would fail outright; the worktree apparently isn't tested on MSVC, but this is a portability landmine.
* The local copy at line 43 (`seg_signs[j] = this->seg_signs[j]`) is redundant — `this->seg_signs[j]` could be used directly inside the kernel call. The local copy was probably added because `this->seg_signs` is a `vector<double>` from a member, and the original author wanted plain-pointer semantics for OMP, but OMP is commented out (`magneticfield_wireframe.cpp:50-51`). Dead weight.

**Parity impact:** none for normal-size wireframes. For pathological inputs, C++ crashes while JAX produces a result.

**Recommended fix:** replace the VLAs with `std::vector<double*>` and `std::vector<double>` of size `nHalfPrds`. This is a one-line behavioural-no-op patch that hardens the C++ side.

### 2. HIGH — Integer-width narrowing at JAX ingress; no overflow guard

`ToroidalWireframe.segments` is constructed as **int64**:

```python
# wireframe_toroidal.py:163-164
self.segments = \
    np.ascontiguousarray(np.zeros((self.n_segments, 2)).astype(np.int64))
```

The C++ wrapper consumes `PyIntArray = xt::pyarray<int>` (32-bit signed on every platform that simsopt targets). xtensor converts via pybind11 — silently truncating on overflow if `n_nodes > 2^31` (no guard in either path).

The JAX wrapper does:

```python
# wireframefield_jax.py:26
segments = np.array(wframe.segments, dtype=np.int32, order="C", copy=True)
```

This is **a silent narrowing copy**. If `wframe.segments` contains values `>= 2^31`, the cast wraps modulo `2^32`. There is no `astype(np.int32, casting='safe')` guard; `casting="unsafe"` is the NumPy default. The same silent narrowing happens on the C++ side via pybind11's array binding.

Then `wireframe.py:115`:

```python
segments_jax = _as_jax_int32(segments)
```

`_as_jax_int32` (`_math_utils.py:57-58`) wraps `as_jax_array(..., dtype=jnp.int32)` — another silent narrowing.

**Realistic threshold:** `n_nodes = n_phi * n_theta` for `ToroidalWireframe`. To exceed int32, you need `n_phi * n_theta > 2^31 ≈ 2.1e9`. Not reachable in practice (CPU memory exhausts well before that). **But** the wireframe-optimization workflow constructs free-cell lookups via `np.int64`-indexed segment keys (`wireframe_toroidal.py:134, 140, 254-255`), and a follow-on grid refinement could push past the limit. There is no defensive check.

**Severity rationale:** dtype-mismatch silent wraparound on the boundary is a HIGH-class bug class even when not currently triggered. The fix is one-liner `assert wframe.segments.max() < 2**31` at boundary.

### 3. MEDIUM — Per-segment contribution cache is not invalidated on `set_points()`

```python
# wireframefield_jax.py:60-70
def set_points_cart(self, xyz):
    result = super().set_points_cart(xyz)
    self._points_device = _as_jax_float64(np.asarray(xyz, dtype=np.float64))
    return result
```

`super().set_points_cart()` invalidates the parent `MagneticField` cache. The JAX wrapper then updates `self._points_device`. **But** `self._dB_by_dcoilcurrents`, populated by `dB_by_dsegmentcurrents` (`wireframefield_jax.py:116-119`), is *not* invalidated.

The CPU wrapper does not have this hazard because `dB_by_dsegmentcurrents` uses the C++ `field_cache` and checks `fieldcache_get_status` (`wireframefield.py:53-54`):

```python
if any([not self.fieldcache_get_status(f'B_{i}')
        for i in range(self.wireframe.n_segments)]):
    assert compute_derivatives >= 0
    self.compute(compute_derivatives)
```

The C++ `MagneticField::invalidate_cache()` cascades to the wireframe `field_cache` (`magneticfield_wireframe.h:108-112`), so after `set_points()`, the next `dB_by_dsegmentcurrents` call recomputes.

The JAX wrapper has no equivalent. Once `self._dB_by_dcoilcurrents` is populated for one `points` set, a subsequent `set_points()` does not clear it, but the **next** `dB_by_dsegmentcurrents` call always recomputes regardless (it does not check a status flag). So the bug is latent: the cache field exists, but it's only used as a return-list buffer, not as a memoization. The mismatch is purely cosmetic — *unless* a future change makes the JAX wrapper consult `self._dB_by_dcoilcurrents` for early-return.

**Recommended fix:** explicitly `del self._dB_by_dcoilcurrents` (or set to `None`) in the overridden `set_points_cart`/`set_points_cyl` to lock in the no-stale-cache contract.

### 4. MEDIUM — `WireframeFieldJAX` `_simsopt_jax_native_field=True` but no `B_vjp` implementation

```python
# wireframefield_jax.py:47
class WireframeFieldJAX(MagneticField):
    _simsopt_jax_native_field = True
```

`MagneticFieldSum._raise_if_strict_jax_mixed_composition`
(`magneticfield.py:23-40`) and `_is_jax_native_field`
(`magneticfield.py:12-20`) gate composition based on this attribute. A `MagneticFieldSum([WireframeFieldJAX(...), BiotSavartJAX(...)])` is accepted as JAX-native, and the sum's `_B_impl` calls each child's `_B_impl`
(`magneticfield.py:290-293`).

But `MagneticFieldSum.B_vjp` (`magneticfield.py:308-309`) iterates over children and calls `bf.B_vjp(v)`. `WireframeFieldJAX` does **not** implement `B_vjp`. The CPU `WireframeField` inherits `B_vjp` from `sopp.WireframeField` via the C++ MagneticField inheritance — meaning a sum with the CPU wireframe field has `B_vjp` working, but a sum with the JAX wireframe field would raise `AttributeError`.

**Trigger condition:** the wireframe contributions are **linear** in currents, so optimization paths that adjoint through `B(currents)` would need `B_vjp` (a Jacobian transpose with respect to coil/segment dofs). In practice, `optimize_wireframe_jax` uses the analytic `Amat` matrix instead of `B_vjp` (since the system is linear), so this gap is not triggered today. But the marker `_simsopt_jax_native_field = True` claims more than the wrapper delivers.

**Recommended fix:** either drop the `_simsopt_jax_native_field` marker (so composition triggers the CPU-fallback path), or add a stub `B_vjp` that explicitly raises with a descriptive message, since the field has no free dofs from the perspective of upstream optimization.

### 5. MEDIUM — `set_array_to_zero(ddB)` runs on a dummy at derivatives=0

`magneticfield_wireframe.cpp:17-20, 30-35`: `Array dummyjac`, `Array dummyhess`, `Tensor3 _dummyjac`, `Tensor4 _dummyhess` coexist with shadowed names. At `derivatives == 0`, `dB` and `ddB` references bind to the `Tensor3`/`Tensor4` dummies; `set_array_to_zero(dB)` and `set_array_to_zero(ddB)` write through the dummies to zeros then never read them. Pure waste, no UB. **Severity rationale:** maintainability hazard from name shadowing + dead writes; not a parity issue.

### 6. MEDIUM — JAX rank-1 input silent mis-broadcast

```python
# wireframe.py:133
return _MU0_OVER_4PI * factor[:, None] * jnp.cross(diff0, diff1)
```

If `points` has rank-1 shape `(3,)` (single point), then `diff0` and `diff1` are `(3,)`, `norm_diff0` is `()` (scalar), and `factor` is `()`. Then `factor[:, None]` raises:

```
IndexError: tuple index out of range
```

Tested via mental trace and consistent with JAX's stricter rules. So the *failure mode* is loud, not silent — but error message is opaque.

**What if `points` has shape `(M, N, 3)` (extra batch axis)?** Then `factor` has shape `(M, N)`, and `factor[:, None]` becomes `(M, 1, N)`, which broadcasts against `jnp.cross(diff0, diff1).shape = (M, N, 3)` incorrectly (silent shape error — would produce garbage values via implicit broadcasting). This *is* a silent bug for batched callers.

**Equivalent C++ behaviour:** the C++ kernel hard-codes 1-D `pointsx`/`pointsy`/`pointsz` (`magneticfield_wireframe.h:34-36`) via the `AlignedPaddedVec`. A multi-dim ndarray would be caught at the `fill_points` cast on line 47-50 (single-axis assumption baked in).

**Recommended fix:** add a `points.ndim == 2 and points.shape[-1] == 3` assertion at the public entry points (`wireframe_segment_B`, `wireframe_B`, `wireframe_dB_by_dX`, `wireframe_B_and_dB_by_dX`).

### 7. MEDIUM — `dB_by_dsegmentcurrents` semantic ambiguity

The CPU wrapper `dB_by_dsegmentcurrents(compute_derivatives)`
(`wireframefield.py:30-61`) accepts `compute_derivatives ∈ {0, 1}`. But the docstring promises:

> If zero, will provide derivatives of the magnetic field with respect to
> segment currents. If one, will provide derivatives of the first spatial
> derivatives of the magnetic field with respect to segment currents.

And then **always returns** `self.fieldcache_get_or_create(f'B_{i}', ...)` — i.e., **B contributions, never dB contributions**. The `compute_derivatives` argument only controls what gets cached internally as a *side effect*: the `dB_{i}` array is computed and stashed in the field cache if `compute_derivatives=1`, but **the return value is still the B-cache**.

The JAX wrapper inherits this:

```python
# wireframefield_jax.py:107-120
contributions = np.asarray(
    _wireframe_segment_B_contributions_jit(...),  # B contributions
    dtype=np.float64,
)
self._dB_by_dcoilcurrents = [
    np.ascontiguousarray(contributions[i])
    for i in range(self.wireframe.n_segments)
]
return self._dB_by_dcoilcurrents
```

So both wrappers return B contributions. Test
`test_public_B_dB_and_segment_contributions_match_cpu`
(`test_wireframefield_jax_item30.py:73-84`) parity-checks **both** the
`compute_derivatives=0` and `compute_derivatives=1` returns — and both pass because **both** sides return B contributions. The test thus has no signal on whether dB-segment-current parity is real.

**Parity-claim wise:** identical behaviour CPU and JAX. **Semantic-claim wise:** the docstring is wrong on both sides, but they're wrong in the same way. The `wireframe_segment_dB_by_dX_contributions` JAX function exists but is never reachable through `dB_by_dsegmentcurrents` — confirming the first-pass INFO finding that it's unused. (Note: the JAX wrapper accepts but does not act on `compute_derivatives=1` beyond raising for `compute_derivatives > 1`; it does not differentiate the JAX path for `=0` vs `=1`.)

**Recommended fix:** fix the docstrings on both sides to describe actual behaviour, or wire up `dB_by_dsegmentcurrents(1)` to actually call `wireframe_segment_dB_by_dX_contributions` and return that. Either way, the test should distinguish the two cases.

### 8. LOW — Singular-regime contract is shared but untested

The closed form diverges at three regimes (collinear-on-segment, `r = a`, `r = b`); both kernels emit `inf`/`nan` and the JAX docstring (`wireframe.py:64-75`) explicitly says "No defensive floors are inserted." But there is no test that asserts this. A future "harden the JAX kernel against NaN" patch could silently introduce a guard that diverges from C++.

**Recommended fix:** add a contract test that constructs a point at `r = node0` and asserts `assert not np.all(np.isfinite(B_jax))` and `assert not np.all(np.isfinite(B_cpp))`.

### 9. LOW — `jnp.sum` reduction-order divergence: CPU vs GPU portability

The contribution paths (`wireframe.py:261-268, 297-304`) use
`jnp.sum(jax.vmap(half_period_B)(...), axis=0)`. XLA's `reduce` op does not
guarantee summation order; on CPU it typically reduces sequentially, but on
GPU it can use a tree reduction. The `direct_kernel` lane (`rtol=1e-10`) is
tight; for small `n_halfprds`, both reductions agree to machine precision, but
for higher `nfp` or numerically pathological configurations, GPU vs CPU could
drift. The CLAUDE.md "Floating-point reproducibility across machines" note
already documents that this kind of cross-hardware drift is real for other
kernels.

**Recommended fix:** the first-pass already recommended replacing
`jnp.sum(jax.vmap(...))` with `lax.scan` in the contribution paths. This pass
re-affirms; it is the only way to guarantee CPU == GPU == C++ parity for
arbitrary `nfp`.

### 10. LOW — No validation that `seg_signs.size == nodes.shape[0]`

```python
# wireframe.py:467-468
_as_jax_float64(seg_signs).reshape((-1,)),
```

Reshape silently allows length mismatch. If a caller passes `seg_signs = [1.0, -1.0, 1.0]` but `nodes.shape[0] == 4` (4 half-periods), the subsequent `lax.scan` over `(node0_by_segment, node1_by_segment, seg_signs)` raises a `ScopeMismatchError` because `lax.scan` enforces matching length on stacked args. So the failure is loud here. Good.

For the `vmap` path in `wireframe_segment_B_contributions`, an analogous length mismatch causes `vmap` to raise. Also loud.

The C++ side reads `seg_signs[j]` for `j ∈ [0, nHalfPrds)` where
`nHalfPrds = this->nodes.size()`, but `this->seg_signs` is a `vector<double>`
of whatever length the caller supplied. C++ has no out-of-bounds check — a
short `seg_signs` would read past the end (UB).

**Recommended fix:** add an explicit length check in `WireframeField::compute()` and at the JAX wrapper boundary. This is a hardening item, not a parity issue.

### 11. LOW — `WireframeFieldJAX` holds live reference to `self.wireframe`

```python
# wireframefield_jax.py:51-58
self.wireframe = wframe
self.nodes, self.segments, self.seg_signs, self.currents = (
    _snapshot_wireframe_arrays(wframe)
)
```

The nodes/currents are snapshotted (deep-copied) but `self.wireframe = wframe` is a *reference*. Then:

```python
# wireframefield_jax.py:118
for i in range(self.wireframe.n_segments)
```

`self.wireframe.n_segments` is read at call time, not snapshot time. If the caller mutates `wframe` after construction (e.g., calls
`wframe.update_segments(...)`), `WireframeFieldJAX.dB_by_dsegmentcurrents()` accesses a stale snapshot via `self._currents_device` but a live `n_segments` count. If `n_segments` shrank, the loop terminates early (silently dropping segments). If `n_segments` grew, the index `i` exceeds `contributions.shape[0]` and raises `IndexError`.

The CPU wrapper has the same issue (`wireframefield.py:54`):

```python
for i in range(self.wireframe.n_segments)
```

So both wrappers share this hazard. Test `test_current_snapshot_matches_cpu_wrapper_semantics` only verifies that mutating `wireframe.currents` is ignored — not `wireframe.n_segments`. **Parity-wise identical**, but a future ToroidalWireframe API that adds dynamic segment growth would surface this.

**Recommended fix:** snapshot `self._n_segments = wframe.n_segments` at construction.

### 12. INFO — `wireframe_segment_dB_by_dX_contributions` is dead public API

Reaffirmed from first pass. The function is exported in `__all__`
(`wireframe.py:100`), tested only structurally
(`test_combined_B_and_dB_matches_separate`), and never consumed inside `src/`
or `benchmarks/`. The `dB_by_dsegmentcurrents` wrapper always returns
B contributions regardless of `compute_derivatives`. The C++ field cache
*does* hold per-segment dB arrays internally, but no public Python accessor
returns them — the JAX export is for parity completeness, not for use.

**Recommended action:** decide whether to add a direct C++ parity test
(probing `field_cache.get(IndexedFieldCacheKind::dB, i)` via pybind11) or
remove the JAX export. The current state is "dead code with dead test."

### 13. INFO — Test fixture realism

Existing test fixtures:
* `test_wireframe_jax_item29.py::_closed_loop_nodes`: 10 nodes, 10 segments,
  single half-period.
* `test_wireframe_item29.py::_wireframe_case`: `ToroidalWireframe(n_phi=4,
  n_theta=6)` -> `n_segments = 48`, `n_halfprds = 4` (nfp=2 stellsym),
  `n_nodes = (4+1)*6 = 30`.

Production wireframes (e.g., NCSX, W7-X) typically use `n_phi ≈ 32, n_theta ≈ 64` -> `n_segments ≈ 4096`. The current test fixtures exercise neither the n-segment scan throughput nor the int32 cast boundary in any meaningful way.

**Recommended action:** add a `n_phi=16, n_theta=32, nfp=4` parity fixture
(yielding `n_segments = 1024, n_halfprds = 8`) under a `pytest.mark.slow` marker.

### 14. INFO — Layout-convention asymmetry

`wireframe.py:27-40` documents `dB[p, k, m]` (component-first), matching C++
`dB_by_dX(p, k, m) = fak * dB_dX_i[k].m`. The CLAUDE.md abstract convention
says `dB[p, j, l] = ∂_j B_l` (derivative-first). The two are transposes of each
other. `test_dB_layout_convention_via_finite_difference`
(`test_wireframe_jax_item29.py:307-350`) confirms the *component-first*
layout by FD at a single point.

Forensic confirmation that this matches both `simsoptpp.WireframeField` and
`simsoptpp.BiotSavart`:
* `wireframe_field_impl.h:121-123`:
  `dB_by_dX(i+j, k, 0/1/2) = fak * dB_dX_i[k].x/y/z`  (k=component, then m)
* `python_magneticfield.cpp:30` docstring says `\partial_j B_l(x_i)` — but the
  *storage* puts `l` (component) before `j` (deriv), making the docstring
  confusing.

The biot_savart_jax.py module uses the same component-first storage too
(despite the doc claiming `[p, j, l]`). The "deriv-first" convention is
documented but not implemented anywhere in the worktree. So **no downstream
consumer currently mixes the conventions** — the lint risk is purely
documentation hygiene.

**Recommended action:** update CLAUDE.md to reflect the actual
component-first storage convention used throughout the worktree, or add a
named-axis comment to each affected module.

## C++ end-to-end UB sweep

Read `wireframe_field_impl.h` (239 lines), `magneticfield_wireframe.cpp` (127 lines), `magneticfield_wireframe.h` (126 lines). Findings:

* **`wireframe_field_impl.h:107-114`** — the SIMD overshoot guard at the end of the i-loop is correct: `jlimit = std::min(simd_size, num_points-i)` bounds the inner loop. No out-of-bounds writes.
* **`wireframe_field_impl.h:23, 147`** — `B`, `dB_by_dX`, `d2B_by_dXdX` are passed by reference and used uninitialized for the SIMD overshoot, but the overshoot bytes are *discarded* (`jlimit` bounds the write loop). No UB.
* **`wireframe_field_impl.h:43-45, 166-168`** — `node0_vec` and `node1_vec` are constructed from `node0[0..2]`, `node1[0..2]`. If `node0.size() < 3`, UB. The Python caller always passes 3-element vectors, but no check.
* **`magneticfield_wireframe.cpp:25`** — `double* currents_ptr = &(this->currents(0));` is unused (the loops index `currents[i]` directly via xtensor). Dead variable, no harm.
* **`magneticfield_wireframe.cpp:39-40`** — VLA. See Finding 1.
* **`magneticfield_wireframe.cpp:50-51`** — `#pragma omp parallel for` is commented out with a "not thread safe" note. The `field_cache.get()` calls and `axpy_array` calls write through xtensor refs; if OMP were enabled, races would occur on the cache. Confirmed serial. Not currently a UB issue but a foot-gun if someone uncomments the pragma without auditing `IndexedFieldCache::get`.
* **`magneticfield_wireframe.cpp:54-60`** — `set_array_to_zero(Bi)` and the equivalent for `dBi` are done inside the i-loop, *before* the j-loop accumulates. So Bi/dBi start zero for each segment. Correct.
* **`magneticfield_wireframe.cpp:64-65`** — `segments_ptr[2*i]` and `segments_ptr[2*i + 1]` rely on row-major layout of `this->segments`. xt::pytensor is row-major by default; `xt::pyarray` is row-major too unless overridden. OK.
* **`magneticfield_wireframe.cpp:101-119`** — three separate accumulation loops at the end (B, dB, ddB). Each does `xt::noalias(B) = B + current * Bi`. `xt::noalias` is correct here because `B` and `Bi` are distinct buffers.

**No ordered-pragma issues, no race conditions in the current code path. No missing braces. No `mod_B_squared`-style accumulator races.** The wireframe C++ is cleaner than `biot_savart.cpp` was prior to the code-review fixes.

## Untested edge-case inventory

| Edge case | Tested? | Expected behaviour | Severity if violated |
| --- | --- | --- | --- |
| `r = node0_hp` (zero distance) | No | `nan`/`inf` (same on both) | LOW |
| `r = node1_hp` (zero distance) | No | `nan`/`inf` (same on both) | LOW |
| `r` collinear *between* node0 and node1 | No | `nan`/`inf` (same on both) | LOW |
| `node0 == node1` (zero-length edge) | No | `0 * inf = nan` (same on both) | LOW |
| `n_halfprds >= 8` (nfp >= 4) | No | Reduction-order divergence on contribution path | MEDIUM |
| `n_segments > 2^15` realistic torus | No | Throughput / int32 boundary | MEDIUM |
| `n_segments > 2^31` (overflow) | No | Silent wraparound | HIGH (latent) |
| `seg_signs.size != nodes.shape[0]` | No | C++ UB / JAX shape error | LOW |
| `points.ndim != 2` | No | JAX shape error / C++ UB | MEDIUM |
| `set_points` then mutate `wireframe.n_segments` | No | Stale cache / IndexError | LOW |
| `MagneticFieldSum([WireframeFieldJAX, BiotSavartJAX]).B_vjp(v)` | No | AttributeError on missing `B_vjp` | MEDIUM |
| `dB_by_dsegmentcurrents(2)` | Yes (raises NotImplementedError on JAX; throws logic_error on CPU) | Different exception types | LOW |
| `WireframeFieldJAX` with shared-node multi-coil sets | N/A | nfp-symmetric stellarator only via `ToroidalWireframe` | -- |
| Autodiff through `wireframe_B` w.r.t. nodes | No | Not used in production (analytic Amat instead) | INFO |
| Autodiff through `wireframe_B` w.r.t. currents | No | Linear; would yield `wireframe_segment_B_contributions` exactly | INFO |
| Strict transfer guard on GPU | CPU-only test | GPU-side reduction order may differ from CPU | INFO |

## Prompt cross-questions (cleaned)

* **Cross-product orientation (Q2):** C++ `cross()` (`vec3dsimd.h:166-172`) emits
  `(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x)` via
  `xsimd::fms`. JAX `jnp.cross` uses the same right-hand-rule formula. Same
  frame, same sign. The `fms` fused op is one-ulp tighter than naïve JAX
  arithmetic but the `direct_kernel` lane absorbs the difference. **No sign
  divergence.**
* **Autodiff w.r.t. nodes (Q5):** both kernels emit the **analytic** Jacobian
  w.r.t. point positions. Neither uses autodiff internally. `jax.jacfwd(...,
  argnums=1)` would work and (modulo singular regimes) would yield the same
  result as `wireframe_dB_by_dX` composed with `dgamma_by_dnodes` — untested
  but not a production concern.
* **Autodiff w.r.t. currents (Q6):** the field is linear in currents:
  `B(currents) = sum_i currents[i] * unit_B_i(...)`.
  `jax.jacfwd(wireframe_B, argnums=4)` should yield
  `wireframe_segment_B_contributions(...)` exactly. **Untested.** A trivial
  parity test would lock down the linearity invariant.
* **Multi-coil sets / shared nodes (Q7):** N/A by design. The data model is a
  *single* wireframe topology; shared nodes are just nodes connecting two
  segments, each carrying an independent current. Kirchhoff sum is implicit.
* **JIT closure (Q10):** `_wireframe_B_jit` etc. take device arrays as args,
  not as closures. Attribute swap on the wrapper is safe-ish (re-traces on
  dtype/shape mismatch); in-place mutation of `self.nodes` does NOT propagate
  to `self._nodes_device` (`_as_jax_float64` copied). Same isolation as the
  CPU pybind11 vector copy.
* **B_vjp / B_jacobian (Q11):** CPU wrapper inherits `B_vjp` from
  `sopp.MagneticField`; JAX wrapper does not implement either. No production
  consumer (Stage-2 squared-flux or wireframe-opt) hits this — wireframe-opt
  uses the analytic `Amat` matrix. But the `_simsopt_jax_native_field = True`
  marker (`wireframefield_jax.py:47`) over-claims; see Finding 4.

## Conclusion

The wireframe JAX port is, as the first-pass concluded, a clean line-by-line
mirror of the C++ closed form. The deeper hazards are:

1. **HIGH — C++ VLAs and silent int32 narrowing** at the dtype boundary are
   real but currently unreachable. Both should be hardened. The VLA fix is
   trivial; the int32 fix is one assertion at the wrapper boundary.
2. **MEDIUM — Public-API completeness on the JAX wrapper**: missing `B_vjp`
   while claiming `_simsopt_jax_native_field = True`, semantic ambiguity in
   `dB_by_dsegmentcurrents`, dead `_dB_by_dcoilcurrents` cache field, and
   silent mis-broadcast for non-2D points. None block production, but each is
   a future-bug seed.
3. **MEDIUM — Sub-tolerance reduction-order portability**: the contribution
   path's `jnp.sum` is not order-stable across XLA backends; the first-pass
   already flagged this; it remains the only structural divergence between
   JAX and C++ that is not bounded by construction.
4. **LOW — Test coverage gaps**: singular-regime contract, autodiff
   linearity-in-currents invariant, realistic-size topology stress, and
   stellsym-mirror parity at higher nfp are all uncovered.

No new HIGH-severity correctness defects beyond what the first pass found.
The HIGH items here are all latent-bug classes (VLA portability, int32 wraparound),
not active bit-identity failures.

## Recommended action queue

1. **HIGH-1**: replace C++ VLAs with `std::vector` (4-line patch).
2. **HIGH-2**: assert `wframe.segments.max() < 2**31` at `WireframeFieldJAX.__init__`
   and at `WireframeField.__init__` (or in `wireframe_toroidal.py`).
3. **MEDIUM-3**: invalidate `self._dB_by_dcoilcurrents` in
   `WireframeFieldJAX.set_points_cart` / `set_points_cyl`.
4. **MEDIUM-4**: implement `WireframeFieldJAX.B_vjp` (or remove the
   `_simsopt_jax_native_field` marker until it's implemented).
5. **MEDIUM-6**: add `assert points.ndim == 2 and points.shape[-1] == 3` at
   public entry points.
6. **MEDIUM-7**: fix docstrings on `dB_by_dsegmentcurrents`; decide whether
   to wire up `wireframe_segment_dB_by_dX_contributions` or remove the export.
7. **MEDIUM-9**: replace `jnp.sum(jax.vmap(...))` with `lax.scan` in
   `wireframe_segment_B_contributions` and `wireframe_segment_dB_by_dX_contributions`
   (first-pass recommendation, re-affirmed).
8. **LOW-8**: add a singular-regime contract test
   (`assert not all_finite`).
9. **LOW-11**: snapshot `self._n_segments = wframe.n_segments` at
   construction.
10. **LOW-13**: add a `pytest.mark.slow` `n_phi=16, n_theta=32, nfp=4` parity
    fixture.
11. **INFO-14**: align CLAUDE.md tensor-convention note with the actual
    component-first storage used in `wireframe.py` and `biot_savart_jax.py`.

The recommended audit lane for follow-up work is `direct_kernel`
(`rtol=1e-10`, `atol=1e-12`) for all new tests except autodiff-vs-analytic,
which warrants `derivative_heavy` (`first_derivative_rtol=1e-8`).
