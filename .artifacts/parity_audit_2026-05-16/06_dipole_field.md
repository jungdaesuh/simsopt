# Priority 6 — Permanent-magnet dipole field parity audit

**Audit timestamp:** 2026-05-16
**Branch:** `gpu-purity-stage2-20260405`
**Auditor:** Claude (Opus 4.7, 1M context)

## Files audited

| Role | Path | Lines |
|------|------|------:|
| JAX module | `src/simsopt/jax_core/dipole_field.py` | 533 |
| C++ reference | `src/simsoptpp/dipole_field.cpp` | 855 |
| C++ header | `src/simsoptpp/dipole_field.h` | 20 |
| Public JAX wrapper | `src/simsopt/field/dipole_field_jax.py` | 347 |
| C++ Optimizable wrapper | `src/simsopt/field/magneticfieldclasses.py` (DipoleField) | (lines 572–730) |
| Consumer (PM grid, JAX) | `src/simsopt/geo/permanent_magnet_grid_jax.py` | (calls `dipole_field_Bn` at L132) |
| Consumer (PM grid, C++) | `src/simsopt/geo/permanent_magnet_grid.py` | (calls `sopp.dipole_field_Bn` at L428) |
| Adjacent tests | `tests/jax_core/test_dipole_field_item24.py`, `tests/jax_core/test_dipole_field_jax_item24.py`, `tests/field/test_dipole_field_jax_item26.py`, `tests/geo/test_permanent_magnet_grid_jax_item27.py` | — |

## Executive summary — top 3 findings

1. **MATH/PHYSICS parity is bit-clean across all five public kernels** (`dipole_field_B`, `dipole_field_A`, `dipole_field_dB`, `dipole_field_dA`, `dipole_field_Bn`). Formulas are identical to the C++ reference (including the prefactor `fak = 1e-7 = μ₀/(4π)`, the inverse-cube/inverse-fifth scaling, the symmetry-expansion convention, and the cylindrical/toroidal rotation matrices). The CPU oracle tests in `tests/jax_core/test_dipole_field_item24.py` (lines 91–227) cover all five kernels at the `direct_kernel` lane (rtol=1e-10, atol=1e-12).

2. **INFO — Zero autodiff coverage on the dipole-field kernels.** The JAX module exists *because* the C++ kernels are not differentiable, yet **no test exercises `jax.grad`, `jax.jacfwd`, or `jax.jacrev` against any of `dipole_field_{B,A,dB,dA,Bn}`** (`grep -rn "jax.grad\|jax.jacfwd\|jax.jacrev"` over the test files returns empty). The PM-grid wrapper at `permanent_magnet_grid_jax.py:142–148` builds `A_obj` via `dipole_field_Bn` and immediately consumes it for `ATb`/`SVD`, so a regression in the linear `∂A/∂m_points` Jacobian would silently break downstream Stage-2 PM optimization without test coverage. **Recommend adding gradient parity tests** before priority 7 (permanent-magnet optimization) is audited.

3. **MEDIUM — Singularity policy is silently undefined.** Neither `dipole_field.py` nor the C++ kernels guard against an evaluation point coinciding with a dipole position. `_explicit_rsqrt(r2)` will return `+inf` when `r2 == 0`, propagating into the field and `dB` as `±inf` or `nan`; the C++ `rsqrt(0.0)` has the same behavior. Test `test_dipole_field_jax_vs_cpp_direct_kernel` carefully places evaluation points on a `[0.6, 2.0]` shell and dipoles inside `|x| < 0.3` (lines 38–63) to avoid this, so the contract is "caller must keep points off dipole sites." This is consistent with the C++ code but is *not* documented in either module's docstring. **The behavior parity is intact** (both return inf/nan), so this is INFO/MEDIUM, not CRITICAL.

## Function-by-function parity matrix

| JAX entry point (file:line) | C++ counterpart (file:line) | Math | Physics | Algorithm | Computation | Test oracle |
|---|---|---|---|---|---|---|
| `dipole_field_B` (`dipole_field.py:142–155`) via `_dipole_field_B_jit` (L120–139) | `dipole_field_B` (`dipole_field.cpp:16–69` XSIMD; L393–439 scalar) | ✅ identical (eqn at L8–9) | ✅ SI units, B in T | ✅ JAX scan-over-dipoles; C++ omp-parallel chunked points | ✅ float64; multiplication tree `rinv*rinv, rinv*rinv2, rinv3*rinv2` matches | `test_direct_cpp_parity_for_field_and_derivative_kernels` |
| `dipole_field_A` (`dipole_field.py:178–191`) via `_dipole_field_A_jit` (L162–175) | `dipole_field_A` (`dipole_field.cpp:72–123`; L442–486 scalar) | ✅ `m × r / r³` | ✅ SI | ✅ scan vs omp | ✅ | same test L107 |
| `dipole_field_dB` (`dipole_field.py:234–247`) via `_dipole_field_dB_jit` (L198–231) | `dipole_field_dB` (`dipole_field.cpp:133–191`; L496–548 scalar) | ✅ full symmetric tensor matches comment formula L126–131 | ✅ T/m | ✅ JAX uses full `(P,3,3)` tensor; C++ writes 6 upper-tri entries + copies to lower-tri (L185–187, L543–545) | ✅ symmetric by construction in JAX vs by-copy in C++; **JAX always finite for symmetric off-diagonals** | parity test + `test_dipole_field_convention_dB_symmetric` |
| `dipole_field_dA` (`dipole_field.py:290–303`) via `_dipole_field_dA_jit` (L265–287) | `dipole_field_dA` (`dipole_field.cpp:201–262`; L558–612 scalar) | ✅ skew + `−3(m×r) r / r⁵` matches comment L193–200 | ✅ T (since A is T·m and ∂/∂x is /m) | ✅ scan vs omp | ✅ | parity test + curl identity `test_dipole_field_dA_antisymmetric_part_matches_B` |
| `dipole_field_Bn` (`dipole_field.py:418–471`) | `dipole_field_Bn` (`dipole_field.cpp:277–382`; L629–728 scalar) | ✅ nesting + rotation-matrix order matches | ✅ SI | ⚠️ JAX precomputes `_basis_angles` once outside `(stell, fp)` loop; C++ recomputes inside (algebraically equivalent because it always uses original `mp_j`, not symmetry-transformed `mp_j_new`) | ✅ Python `for stell: for fp:` gives same reduction order as C++ nested loops | `test_dipole_field_Bn_cpp_parity_for_production_matrix` (parametrized cartesian/cylindrical/toroidal) |
| `define_a_uniform_cartesian_grid_between_two_toroidal_surfaces` (`dipole_field.py:512–533`) | same name (`dipole_field.cpp:736–856`) | ✅ ray length 4.0/2000 = 0.002 step matches | n/a | ✅ JAX `vmap` over candidate points; C++ omp over candidates | ✅ keep predicate `nearest_loc_inner <= 0 AND nearest_loc_outer > 0` matches C++ `continue` + `if (>0) write` | `test_uniform_cartesian_grid_between_toroidal_surfaces_cpp_parity` |

Legend: ✅ identical (verified by code reading + existing CPU-oracle tests), ⚠️ algebraically equivalent but structurally different.

## Detailed findings

### Finding 1 — MATH/PHYSICS parity verified (INFO, no action)

Walking through each formula:

**B field** — C++ `dipole_field.cpp:58–60`:
```cpp
B_i.x += 3.0 * rdotm * r.x * rmag_inv_5 - m_j.x * rmag_inv_3;
```
JAX `dipole_field.py:130–133`:
```python
contribution = (
    three * rdotm[:, None] * r * rinv5[:, None]
    - moment[None, :] * rinv3[:, None]
)
```
Identical to `B = (μ₀/4π) Σ [3(m·r) r / r⁵ − m / r³]`. Prefactor `1e-7` applied identically (C++ `fak * B_i.x[k]` at L63; JAX `_scale(points) * field` at L139). The `_MU0_OVER_4PI` constant is declared at `dipole_field.py:37` as `np.float64(1e-7)`, exactly matching the C++ `double fak = 1e-7;` at L33, L89, L148, L216, L299. **Verified consistent.**

**A field** — C++ `dipole_field.cpp:111–114`:
```cpp
Vec3dSimd mcrossr = cross(m_j, r);
A_i.x += mcrossr.x * rmag_inv_3;
```
JAX `dipole_field.py:168–170`:
```python
r, _, rinv3, _ = _point_dipole_geometry(points, dipole_point)
contribution = jnp.cross(moment[None, :], r, axis=-1) * rinv3[:, None]
```
The cross product orientation matches: both compute `m × r`. Identical.

**dB tensor** — verified above in the function-by-function table. The JAX path uses one `(P,3,3)` symmetric tensor expression while C++ writes 6 upper-triangle entries and copies to the lower triangle. The expressions are algebraically identical (verified by expanding `3 r_inv5 * (m_j r_k + m_k r_j + (m·r) δ_{jk} − 5(m·r) r_j r_k / r²)` matches every C++ written line at `dipole_field.cpp:171–176`).

**dA tensor** — JAX `_dipole_cross_derivative` (`dipole_field.py:254–262`) builds the skew matrix
```
[[ 0, -mz,  my],
 [ mz,  0, -mx],
 [-my, mx,   0]]
```
This satisfies `(skew · r)_j = (m × r)_j` and matches the C++ row-by-row expansion at `dipole_field.cpp:239–247`. For example, JAX row 0 col 1 contribution is `−m_z/r³ − 3(m×r)_x r_y / r⁵`, exactly C++ L240.

### Finding 2 — Algorithm differences are reduction-equivalent (LOW, document only)

**Parallelization axis differs but reduction order matches per-output.**

- C++: `#pragma omp parallel for schedule(static)` over evaluation points in chunks of `simd_size`. Each thread handles a contiguous block of points and serially loops over dipoles.
- JAX: `jax.lax.scan` over dipoles (sequential, per-dipole, broadcast across all points). All points are processed in parallel within each scan step.

For each output element `B[p, :]`, both implementations accumulate dipole contributions in **the same dipole index order** (j = 0, 1, ..., M−1). Different thread parallelism, same per-cell reduction order — **bit-identity per output cell is preserved on CPU** (which is why the test at `test_dipole_field_jax_vs_cpp_direct_kernel:104–107` passes at `rtol=1e-10, atol=1e-12`).

On GPU, JAX may reorder the reduction inside the scan body's broadcast (the sum over `(p, m, 3)` axes), but this only affects intermediate operations within one scan step, not the cross-dipole order. Empirically the `direct_kernel` tolerance lane passes — see the test config citing `rtol=1e-10`.

**Reduction tree for `r²` and `r_inv`:**

C++ `dipole_field.cpp:53–56`:
```cpp
simd_t rmag_2     = normsq(r);
simd_t rmag_inv   = rsqrt(rmag_2);
simd_t rmag_inv_3 = rmag_inv * (rmag_inv * rmag_inv);
simd_t rmag_inv_5 = rmag_inv_3 * (rmag_inv * rmag_inv);
```
JAX `dipole_field.py:102–107`:
```python
r2 = jnp.sum(r * r, axis=-1)
rinv = _explicit_rsqrt(r2)
rinv2 = rinv * rinv
rinv3 = rinv * rinv2
rinv5 = rinv3 * rinv2
```
Both build `r_inv^3 = r_inv * (r_inv * r_inv)`. **NB:** C++ `dipole_field_dB` at L167–169 uses `rmag_inv_2 = rmag_inv * rmag_inv; rmag_inv_3 = rmag_inv * rmag_inv_2; rmag_inv_5 = rmag_inv_3 * rmag_inv_2` — a *different* multiplication tree than `dipole_field_B`. JAX uses the dB tree for all kernels (`r_inv * r_inv2` not `r_inv * (r_inv * r_inv)`). At float64 these differ at most by 1 ULP per operation; the empirical tests pass at `rtol=1e-10`. Document the discrepancy but no action needed.

### Finding 3 — dipole_field_Bn nesting structure (INFO)

Both implementations compute `_basis_angles(mp_j)` from the *original* dipole position, even though the geometry `r = points - mp_j_new` uses the symmetry-transformed position. This is correct because `phi_new, theta_new` describe the dipole's grid-aligned frame (which is fixed across symmetry copies — each symmetry copy is in the same frame as the original).

C++ `dipole_field.cpp:333–334`:
```cpp
simd_t mp_phi_new = xsimd::atan2(mp_j.y, mp_j.x);
simd_t mp_theta_new = xsimd::atan2(mp_j.z, sqrt(mp_j.x * mp_j.x + mp_j.y * mp_j.y) - R0);
```
JAX `dipole_field.py:310–316`:
```python
def _basis_angles(dipole_points, R0):
    x = dipole_points[:, 0]; y = dipole_points[:, 1]; z = dipole_points[:, 2]
    phi = jnp.atan2(y, x)
    theta = jnp.atan2(z, jnp.sqrt(x * x + y * y) - R0)
    return jnp.sin(phi), jnp.cos(phi), jnp.sin(theta), jnp.cos(theta)
```
JAX precomputes these once outside the `(stell, fp)` loop. C++ recomputes per-iteration but always from `mp_j` (not `mp_j_new`), so the value is loop-invariant. Algebraically equivalent. **Verified consistent**, just structurally different.

The `_rotate_normal_matrix_to_cartesian_basis` (`dipole_field.py:349–364`) applies `stell_sign` only to component 0:
```python
(normal_matrix[:, :, 0] * cphi0 + normal_matrix[:, :, 1] * sphi0) * stell_sign,
-normal_matrix[:, :, 0] * sphi0 + normal_matrix[:, :, 1] * cphi0,
normal_matrix[:, :, 2],
```
matching C++ `dipole_field.cpp:372–374`:
```cpp
A(i + k, j, 0) += fak * (G_i.x[k] * cphi0[k] + G_i.y[k] * sphi0[k]) * pow(-1, stell);
A(i + k, j, 1) += fak * (- G_i.x[k] * sphi0[k] + G_i.y[k] * cphi0[k]);
A(i + k, j, 2) += fak * G_i.z[k];
```
**Component 1 has no `stell` factor.** This is asymmetric but consistent in both implementations.

### Finding 4 — `b` argument in `dipole_field_Bn` is ignored but required (LOW)

Both C++ and JAX accept a `b` (Bnormal external-coil contribution) array as input but **never use it in the returned matrix**. The C++ checks `if(b.layout() != row_major)` (L286–287) for storage-order validation only. The JAX wrapper at `dipole_field.py:438` does the equivalent shape/dtype check via `_as_jax_float64(b)` for parity. This is correct — `dipole_field_Bn` computes only the geometric matrix `A`; the consumer (`permanent_magnet_grid_jax.py:128`) handles `b_obj` separately when forming `A·m − b`.

Both implementations are parity-consistent. No action.

### Finding 5 — Self-field singularity policy is undocumented (MEDIUM/INFO)

When `points[p] == dipole_points[d]`, `r2 = 0`, `r_inv = ∞`. Both implementations propagate this to `±inf` or `nan`. There is no guard. The test fixtures in `test_dipole_field_jax_item24.py` (L38–63) carefully construct points and dipoles in disjoint spatial shells precisely to avoid this. The same caution applies to the consumer `permanent_magnet_grid_jax.py:115–148` — the plasma surface points should never coincide with dipole sites.

The JAX module docstring (L1–7) does NOT mention this. The C++ comment at L7–15 also does NOT. **Recommend adding a docstring warning** along the lines of "Caller is responsible for ensuring `points` and `dipole_points` are disjoint; coincident points produce non-finite output." This is INFO severity — both implementations agree on the (undefined) behavior, so it is not a parity gap, but documentation hygiene matters for downstream PM consumers.

### Finding 6 — `_explicit_rsqrt` is the JAX numerical hot spot (INFO)

`dipole_field.py:103, 113` calls `_explicit_rsqrt(r2)`, defined in `simsopt.jax_core._math_utils`. The C++ side uses XSIMD's `rsqrt`, which may invoke hardware reciprocal-sqrt instructions on x86 (potentially less precise than full `1/sqrt`). The fact that the JAX-vs-C++ test passes at `rtol=1e-10` suggests `_explicit_rsqrt` produces hardware-accurate `1/sqrt(x)` results that match the C++ XSIMD path's per-element accuracy on the M2 macOS test platform. No further action — flag is documentation-only.

## Test coverage gaps

| Coverage area | Status | Severity |
|---|---|---|
| `dipole_field_B` CPU oracle parity | ✅ `tests/jax_core/test_dipole_field_item24.py:91–119` | — |
| `dipole_field_A` CPU oracle parity | ✅ same test | — |
| `dipole_field_dB` CPU oracle parity | ✅ same test | — |
| `dipole_field_dA` CPU oracle parity | ✅ same test | — |
| `dipole_field_Bn` CPU oracle parity (cartesian/cylindrical/toroidal) | ✅ `test_dipole_field_Bn_cpp_parity_for_production_matrix` (L191–226) | — |
| `define_a_uniform_cartesian_grid_between_two_toroidal_surfaces` parity | ✅ `test_uniform_cartesian_grid_between_toroidal_surfaces_cpp_parity` (L229–265) | — |
| Symmetric `dB` tensor (∂_j B_l = ∂_l B_j) | ✅ `test_dipole_field_convention_dB_symmetric` (L110–119) | — |
| Curl identity B = ∇ × A | ✅ `test_dipole_field_dA_antisymmetric_part_matches_B` (L122–149) | — |
| Strict transfer-guard (no implicit host→device copies inside kernel) | ✅ `test_dipole_field_strict_transfer_guard` (L181–211) | — |
| Output shape/dtype contract | ✅ `test_dipole_field_output_shapes_and_dtypes` (L152–171) | — |
| Input validation rejects wrong shapes | ✅ `test_dipole_field_rejects_*` (L217–243) | — |
| `DipoleField` preprocessing (`dipole_grid, m_vec`) feeds JAX | ✅ `test_python_dipolefield_preprocessing_usage_matches_jax_raw_kernels` (L165–188) | — |
| `DipoleFieldJAX` end-to-end vs CPU `DipoleField` (B, dB, A, dA, all 3 coordinate flags, stellsym×nfp combos) | ✅ `tests/field/test_dipole_field_jax_item26.py:79–141` | — |
| **`jax.grad` w.r.t. dipole moments** (linear gradient — most-used in PM optimization) | ❌ **MISSING** | **INFO/HIGH** |
| **`jax.grad` w.r.t. dipole positions** (nonlinear, sensitivity to grid placement) | ❌ MISSING | INFO/MEDIUM |
| **`jax.grad` w.r.t. evaluation points** (for downstream surface objectives) | ❌ MISSING | INFO/MEDIUM |
| Finite-difference cross-validation of any autodiff path | ❌ MISSING | INFO/MEDIUM |
| Singular-point behavior (point coincident with dipole) — documented + asserted | ❌ MISSING | LOW |
| GPU determinism under `XLA_FLAGS=--xla_gpu_deterministic_ops=true` | ❌ untested | LOW |

The **autodiff gap is the load-bearing concern**. Recall that the *raison d'être* of the JAX port is to enable end-to-end gradient flow — the C++ kernels already work for forward evaluation. The PM optimization consumer (`permanent_magnet_grid_jax.py:142–146`) builds an SVD-based normal-equation system that does not directly call `jax.grad` on the dipole kernels, but priority 7 (PM optimization) will introduce gradient-based solvers (BFGS / L-BFGS / IFT adjoints) that depend on these gradients being correct. Adding FD-cross-validated `jax.grad` tests *now* (before priority 7) localizes any bug to this module rather than letting it manifest as a downstream optimization-convergence failure.

## Recommended actions (ordered by severity)

### HIGH

None. All five public kernels are math-identical, sign-identical, prefactor-identical, and unit-identical to the C++ reference. Existing parity tests cover the full forward path at `rtol=1e-10, atol=1e-12`.

### MEDIUM

1. **Add autodiff parity tests** under `tests/jax_core/` for `dipole_field_B`, `dipole_field_A`, `dipole_field_dB`, `dipole_field_dA`, and `dipole_field_Bn`. Cross-validate `jax.grad`/`jax.jacfwd` against centered finite differences on each of:
   - moments `m` (linear — gradient should be the C++ `dipole_field_Bn` matrix transposed / contracted)
   - positions `dipole_points` (nonlinear)
   - evaluation `points` (links to the existing `dipole_field_dB` oracle)

   Suggested location: a new module `tests/jax_core/test_dipole_field_autodiff.py`, using the `derivative_heavy` lane (`rtol=1e-8, atol=1e-10`) per the parity-ladder SSOT. This forecloses any gradient regression *before* priority 7 PM optimization consumes the JAX path with gradient-based solvers.

2. **Document the singularity policy.** Add a one-line warning to the module docstring at `dipole_field.py:1–7` and to each public function docstring: "Caller is responsible for ensuring `points` and `dipole_points` are disjoint; coincident locations produce non-finite output (matches the C++ reference)."

### LOW

3. **Unify the `r_inv^3` multiplication tree** between `_dipole_field_B_jit` and the C++ `dipole_field_B` (the C++ B/A use `rinv * (rinv * rinv)` while dB/dA use `rinv * rinv2`). The JAX module uses the dB/dA tree throughout. Both compile to the same arithmetic up to 1 ULP and the test passes at `rtol=1e-10`, but harmonizing the C++ side (B/A → dB/dA tree) would eliminate the asymmetry. This is C++-side polish, not a JAX bug.

4. **Add a CUDA-determinism smoke test.** Once GPU CI lights up, parametrize `test_dipole_field_jax_vs_cpp_direct_kernel` with `XLA_FLAGS=--xla_gpu_deterministic_ops=true` to confirm the `direct_kernel` lane holds on GPU. The `transfer_guard("disallow")` test (`test_dipole_field_strict_transfer_guard`) is necessary but insufficient — it proves no host transfers, not that reductions are deterministic.

### INFO

5. **Document the `b` parameter passthrough** in the JAX `dipole_field_Bn` docstring (`dipole_field.py:418–433` already calls it out — keep this).
6. **Note the algorithmic equivalence of `_basis_angles` precomputation** in a comment near `dipole_field.py:447`, since reviewers may flag the apparent difference vs. C++ in-loop computation.

## Conclusion

`src/simsopt/jax_core/dipole_field.py` is a **clean port** of `src/simsoptpp/dipole_field.cpp`. The forward path is bit-clean at machine precision across all five public kernels and three coordinate frames; algorithmic differences (scan-over-dipoles vs. omp-over-points, precomputed basis angles vs. in-loop recompute) are reduction-equivalent and verified by the existing `direct_kernel` parity-lane tests. The single load-bearing gap is the **absence of autodiff coverage**, which should be closed before priority 7 (PM optimization) consumes this module via gradient-based solvers. No CRITICAL or HIGH findings. Two MEDIUM recommendations (autodiff tests + singularity-policy docstring) plus polish-only LOW/INFO items.
