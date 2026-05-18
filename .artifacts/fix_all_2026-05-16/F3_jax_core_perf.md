# F3 — jax_core performance fixes (2026-05-16)

Two surgical fixes applied to `src/simsopt/jax_core/`:

| ID   | File                                          | Issue                                                                | Fix                                                                  |
|------|-----------------------------------------------|----------------------------------------------------------------------|----------------------------------------------------------------------|
| H-14 | `jax_core/magnetic_axis_helpers.py`           | `jnp.linalg.eig` on a 2x2 fell back to a LAPACK host callback        | Closed-form 2x2 eigenvalue, on-device pure JAX                       |
| H-15 | `jax_core/biotsavart.py`                      | `_make_kernel` LRU cache key carried `jax.default_backend()`         | Dropped platform from `_make_kernel` cache key                       |

---

## Fix 1 — H-14: closed-form 2x2 eigenvalue in `on_axis_iota_rk`

### Issue

`src/simsopt/jax_core/magnetic_axis_helpers.py:598` called `jnp.linalg.eig(M)` on
the final 2x2 monodromy matrix to extract the on-axis iota. In JAX 0.10.0 the
eig path for non-Hermitian inputs lowers to a host-resident LAPACK callback, so
each call performed a device→host→device round trip. That defeated the strict
transfer-guard discipline of the rest of the integrator (`while_loop`,
Dormand-Prince RK4(5)) and made the `transfer_guard("disallow")` test of the
JIT-compiled kernel a latent failure.

### Fix

Replaced the eigenvalue extraction with a closed-form quadratic on the trace
and determinant. For `M = [[a, b], [c, d]]`:

```
tr   = a + d
det  = a*d - b*c
disc = tr*tr - 4*det
```

Eigenvalues are `λ_± = 0.5 * (tr ± sqrt(disc))`. The CPU oracle consumes
`arctan2(imag(eig[0]), real(eig[0]))` of LAPACK's first eigenvalue; for the
physical stellarator on-axis tangent map this is always the complex-conjugate
branch with `disc < 0`, in which case LAPACK returns the root with positive
imaginary part first. The closed form reproduces that ordering on the relevant
branch and remains deterministic (and JIT-traceable) on the non-physical real
branch:

- complex branch (`disc < 0`): `real = tr/2`, `imag = +sqrt(-disc)/2`
- real branch (`disc >= 0`): `real = (tr + sqrt(disc))/2`, `imag = 0`

The branch select uses `jnp.where(disc >= 0, …, …)` rather than a Python `if`,
so the closure traces under JIT. No fallback to `jnp.linalg.eig` for any edge
case (the closed form handles `disc = 0` exactly via both arms collapsing to
`tr/2`).

### Before / after

```diff
-    M = y_final.reshape((2, 2))
-    evals, _ = jnp.linalg.eig(M)
-    eig0 = evals[0]
-    nfp_arr = jnp.asarray(nfp, dtype=jnp.float64)
-    two_pi = jnp.asarray(2.0 * jnp.pi, dtype=jnp.float64)
-    iota = jnp.arctan2(jnp.imag(eig0), jnp.real(eig0)) * nfp_arr / two_pi
-    return iota, steps_taken, succeeded
+    M = y_final.reshape((2, 2))
+    # Closed-form 2x2 eigenvalue: jnp.linalg.eig on a non-Hermitian 2x2
+    # falls back to a host-resident LAPACK call in JAX 0.10.0 ...
+    a = M[0, 0]; b = M[0, 1]; c = M[1, 0]; d = M[1, 1]
+    tr = a + d
+    det_M = a * d - b * c
+    disc = tr * tr - 4.0 * det_M
+    sqrt_real = jnp.sqrt(jnp.maximum(disc, 0.0))
+    sqrt_imag = jnp.sqrt(jnp.maximum(-disc, 0.0))
+    is_complex = disc < 0.0
+    real_part = 0.5 * (tr + jnp.where(is_complex, 0.0, sqrt_real))
+    imag_part = 0.5 * jnp.where(is_complex, sqrt_imag, 0.0)
+    nfp_arr = jnp.asarray(nfp, dtype=jnp.float64)
+    two_pi = jnp.asarray(2.0 * jnp.pi, dtype=jnp.float64)
+    iota = jnp.arctan2(imag_part, real_part) * nfp_arr / two_pi
+    return iota, steps_taken, succeeded
```

Module-level docstring (lines 53-58) and `on_axis_iota_rk` docstring updated to
describe the closed-form path and the strict-transfer-guard guarantee.

### Verification

1. **Closed-form vs `jnp.linalg.eig` smoke test** (task-prescribed fixture):

   ```
   M = jnp.array(np.random.RandomState(0).randn(2, 2))
   ...
   max abs diff: 4.440892098500626e-16
   ```

   Well below the `1e-12` threshold.

2. **`ruff check` + `ruff format`** on both touched files: all checks pass, no
   formatting changes.

3. **`tests/field/test_magnetic_axis_helpers_jax_item21.py`** — all 15 tests
   pass, including:
   - `TestOnAxisIotaParity::test_jax_kernel_matches_cpu_oracle[hsx|ncsx|giuliani]`
     — CPU oracle parity on three production zoo configs (`derivative_heavy`
     lane: `scalar_value_rtol = 1e-10`).
   - `TestJitAndRepeatability::test_compiled_kernel_with_jax_field_runs_under_strict_transfer_guard`
     — the test that the previous `jnp.linalg.eig` path could have failed under
     `jax.transfer_guard("disallow")`.

---

## Fix 2 — H-15: drop platform from `_make_kernel` LRU cache key

### Issue

`src/simsopt/jax_core/biotsavart.py::_make_kernel` was decorated
`@lru_cache(maxsize=32)` with `platform` (the result of `jax.default_backend()`)
in the key. The factory body unconditionally `del platform` — XLA already
specializes the JIT-compiled closure by device at lowering time, so the cpu and
cuda closures were functionally identical Python objects. Including the
platform string wasted cache slots on duplicate closures.

### Fix

1. Removed `platform` from `_make_kernel`'s signature and from its
   `lru_cache` key.
2. Removed the `del platform` (no longer needed).
3. Updated `_get_kernel` to stop passing `jax.default_backend()`.
4. `_make_B_vjp_kernel` (out of scope for the cache-key change per the task,
   but a caller of `_make_kernel`) was updated to stop threading `platform`
   into `_make_kernel`. Its own `lru_cache` key still carries `platform` —
   that path was intentionally left as-is, with a `del platform` inside the
   body and a clarifying docstring sentence.

### Before / after

```diff
 @lru_cache(maxsize=32)
 def _make_kernel(
     integrand_key,
     diff_mode,
     coil_cs,
     quad_bs,
     point_cs,
     point_vma_axis_name,
-    platform,
 ):
-    """...
-    ``platform`` is the active JAX backend string ...
-    """
-    del platform
+    """...
+    The active JAX backend string is intentionally *not* part of the key:
+    XLA specializes the compiled kernel by device at lowering time, so a
+    single Python closure is correct across cpu/cuda.
+    """
     integrand = _INTEGRANDS[integrand_key]
```

```diff
 def _get_kernel(integrand_key, diff_mode, *, point_vma_axis_name=None):
     coil_cs, quad_bs, point_cs = _read_tuning_config()
     return _make_kernel(
         integrand_key,
         diff_mode,
         coil_cs,
         quad_bs,
         point_cs,
         point_vma_axis_name,
-        jax.default_backend(),
     )
```

```diff
 @lru_cache(maxsize=16)
 def _make_B_vjp_kernel(coil_cs, quad_bs, point_cs, platform):
+    del platform
     forward_kernel = _make_kernel(
         _Integrand.B,
         _DiffMode.VALUE,
         coil_cs,
         quad_bs,
         point_cs,
         None,
-        platform,
     )
```

### Verification

1. **`ruff check` + `ruff format`** on both touched files: all checks pass, no
   formatting changes.

2. **`tests/field/test_biotsavart_jax.py`** — all 40 tests pass, including:
   - `TestBiotSavartJaxChunkedSelfConsistency::test_backend_cache_invalidation_clears_kernel_cache`
     — verifies `_make_kernel.cache_clear()` and `cache_info()` accessors
     still behave (signature change is invisible).
   - `TestBiotSavartJaxChunkedSelfConsistency::test_B_vjp_rebuilds_when_tuning_changes_in_process`
     — verifies the VJP kernel still rebuilds when tuning changes.
   - All C++ parity tests on ncsx (B, dB, A, dA, d2B, d2A, B_vjp,
     d{B,A}_by_dcoilcurrents, d2{B,A}_by_dXd_currents, d3{B,A}_by_dXdXdcurrents).

---

## Files changed

- `src/simsopt/jax_core/magnetic_axis_helpers.py` — module/function docstring
  refresh, closed-form 2x2 eigenvalue at the iota extraction site.
- `src/simsopt/jax_core/biotsavart.py` — `_make_kernel` signature/key shrink,
  `_get_kernel` and `_make_B_vjp_kernel` callsite updates, docstring refresh.
