# Parity Audit — Priority 11: Curve geometry (length, curvature, torsion, position derivatives)

- Audit timestamp: 2026-05-16
- Worktree: `/Users/suhjungdae/code/columbia/simsopt-jax`
- Branch: `gpu-purity-stage2-20260405`

## Files audited

| File | Lines | Role |
|---|---:|---|
| `src/simsopt/jax_core/curve_geometry.py` | 916 | JAX-side dispatcher for `gamma`, `gammadash`, `gammadashdash`, `gammadashdashdash`, distance/intersection helpers |
| `src/simsopt/geo/curve.py` | 3000+ | `Curve` base, JAX kernels `incremental_arclength_pure`, `kappa_pure`, `torsion_pure`, `frenet_frame_pure`, `_install_curve_jax_contract` |
| `src/simsopt/geo/curvexyzfourier.py` | 360+ | `CurveXYZFourier`, `jaxfouriercurve_pure`, `jaxfouriercurve_geometry_pure` |
| `src/simsopt/geo/curverzfourier.py` | 130+ | `CurveRZFourier`, `curverzfourier_pure` |
| `src/simsopt/geo/curveplanarfourier.py` | 250+ | `CurvePlanarFourier`, `curveplanarfourier_pure`, `_quaternion_rotation_matrix` |
| `src/simsoptpp/curve.h` | 216 | C++ `Curve` base, `incremental_arclength_impl`, `dincremental_arclength_by_dcoeff_impl`, vjp contraction |
| `src/simsoptpp/curve.cpp` | 59 | C++ `incremental_arclength_impl`, `dincremental_arclength_by_dcoeff_impl`, `least_squares_fit` |
| `src/simsoptpp/curvexyzfourier.cpp` | 109 | C++ XYZ Fourier `gamma`, `gammadash`, `gammadashdash`, `gammadashdashdash`, `d*_by_dcoeff` |
| `src/simsoptpp/curvexyzfourier.h` | 102 | DOF ordering, num_dofs |
| `src/simsoptpp/curverzfourier.cpp` | 258 | C++ RZ Fourier kernels and coefficient derivatives (stellsym + non-stellsym) |
| `src/simsoptpp/curverzfourier.h` | 127 | DOF ordering, num_dofs |
| `src/simsoptpp/curveplanarfourier.cpp` | 710 | C++ Planar Fourier kernels (Fourier-in-polar + quaternion rotation + translation) |
| `src/simsoptpp/curveplanarfourier.h` | 119 | DOF ordering (rc, rs, q, center) |
| `src/simsoptpp/python_curves.cpp` | 146 | pybind11 bindings + trampoline |
| `src/simsoptpp/python_distance.cpp` (linking) | 218 | `compute_linking_number` C++ reference |

## Executive summary

1. **No CRITICAL findings.** The JAX kernels reproduce the C++ math exactly across all three Fourier parameterizations (XYZ, RZ, Planar). The 2π chain-rule factor from `q ∈ [0, 1)` to `phi = 2π·q` is consistently applied: the XYZ kernel uses an explicit `mode_scale = 2π·j` per Fourier mode; the RZ and Planar kernels compute `gammadash` and higher derivatives via `jax.jvp` of the gamma kernel, so the chain rule is enforced by autodiff. The C++ post-multiplies by `2π`, `4π²`, `8π³`. The cosmic-scale formulas match symbolically (verified in detail below).
2. **No HIGH findings.** Coefficient derivatives use `jax.jacfwd(curve.gamma_jax)` in `_install_curve_jax_contract` (`src/simsopt/geo/curve.py:344`) — autodiff-correct provided `gamma_pure` agrees with C++ `gamma_impl`, which it does (the parametrized `test_curve_spec_pullback_production_scale_parity` test pins JAX-vs-C++ gamma at `atol=1e-12, rtol=1e-12` across four coils per class). Stellsym vs. non-stellsym DOF ordering matches the C++ counter layout exactly.
3. **One LOW (test coverage) finding.** There is **no direct CPU-vs-JAX parity assertion for `gammadashdash` / `gammadashdashdash`** at production scale outside the XYZ Fourier direct lane. The `_assert_curve_spec_geometry_matches_live_curve` helper (`tests/integration/test_single_stage_jax_cpu_reference.py:1444`) checks `gammadashdash` for `CurveXYZFourier`, `CurvePerturbed`, `CurveFilament`, but the production-scale row in `tests/geo/test_curve_item05_closeout.py` only compares `gamma` (position). Indirect coverage via Taylor tests in `tests/geo/test_curve.py` validates the derivatives self-consistently but does not pin JAX-RZ-Fourier or JAX-Planar-Fourier `gammadashdash`/`gammadashdashdash` against the C++ scalar values.

## Function-by-function parity matrix

| Quantity | C++ ref (`simsoptpp/*.cpp:line`) | JAX impl (`simsopt/jax_core` or `simsopt/geo`) | Status |
|---|---|---|---|
| **XYZ Fourier** `gamma` | `curvexyzfourier.cpp:5-17` (`gamma_impl`) | `curvexyzfourier.py:309-325` (`jaxfouriercurve_pure`) | PASS — same Fourier formula, `2π·j` arg |
| **XYZ Fourier** `gammadash` | `curvexyzfourier.cpp:19-30` | `curvexyzfourier.py:328-343` (geometry pure) | PASS — explicit `2π·j` cos/-sin pair |
| **XYZ Fourier** `gammadashdash` | `curvexyzfourier.cpp:32-43` | same | PASS — `-(2π·j)²` factor |
| **XYZ Fourier** `gammadashdashdash` | `curvexyzfourier.cpp:45-56` | same | PASS — `±(2π·j)³` factor |
| **XYZ Fourier** `dgamma_by_dcoeff` | `curvexyzfourier.cpp:58-69` | `_install_curve_jax_contract: curve.py:344` via `jax.jacfwd` | PASS — autodiff over matching `gamma_pure` |
| **XYZ Fourier** `dgammadash_by_dcoeff` | `curvexyzfourier.cpp:71-81` | `curve.py:348` via `jacfwd(curve.gammadash_jax)` | PASS — autodiff |
| **XYZ Fourier** `dgammadashdash_by_dcoeff` | `curvexyzfourier.cpp:83-93` | `curve.py:352` | PASS |
| **XYZ Fourier** `dgammadashdashdash_by_dcoeff` | `curvexyzfourier.cpp:95-105` | `curve.py:356` | PASS |
| **RZ Fourier** `gamma` | `curverzfourier.cpp:5-31` | `curverzfourier.py:18-57` (`curverzfourier_pure`) | PASS — `cos(nfp·i·phi)` factored, `cosphi`/`sinphi` rotation, `z` separate |
| **RZ Fourier** `gammadash` | `curverzfourier.cpp:33-59` (post-scaled `2π`) | JVP of `curverzfourier_pure` (`_install_curve_jax_contract: curve.py:333`) | PASS — chain rule via `jvp` |
| **RZ Fourier** `gammadashdash` | `curverzfourier.cpp:61-87` (post-scaled `4π²`) | iterated JVP | PASS — autodiff |
| **RZ Fourier** `gammadashdashdash` | `curverzfourier.cpp:89-127` (post-scaled `8π³`) | iterated JVP | PASS — autodiff (no direct CPU/JAX scalar test, see Test Coverage Gaps) |
| **RZ Fourier** `dgamma_by_dcoeff` | `curverzfourier.cpp:129-154` (stellsym layout: rc, then zs; non-stellsym: rc, rs, zc, zs) | `curve.py:344` jacfwd over `curverzfourier_pure` | PASS — DOF slice order in JAX matches C++ counter order |
| **Planar Fourier** `gamma` | `curveplanarfourier.cpp:15-45` | `curveplanarfourier.py:56-94` (`curveplanarfourier_pure`) | PASS — Fourier-in-polar + quaternion rotation + center translation |
| **Planar Fourier** `gammadash` | `curveplanarfourier.cpp:47-83` | JVP of `curveplanarfourier_pure` | PASS |
| **Planar Fourier** `gammadashdash` | `curveplanarfourier.cpp:85-119` | iterated JVP | PASS |
| **Planar Fourier** `gammadashdashdash` | `curveplanarfourier.cpp:121-166` | iterated JVP | PASS |
| **Planar Fourier** `dgamma_by_dcoeff` | `curveplanarfourier.cpp:168-…` | jacfwd | PASS |
| **Base** `incremental_arclength` | `curve.cpp:36-42` | `curve.py:213-218` (`incremental_arclength_pure`) | PASS — `|γ'|` per quadpoint |
| **Base** `dincremental_arclength_by_dcoeff` | `curve.cpp:44-54` | `curve.py:588-606` via `incremental_arclength_vjp` + `dgammadash_by_dcoeff_vjp` | PASS — chain rule equivalent |
| **Base** `kappa` | — (no C++ impl; Python `Curve.kappa_impl` at `curve.py:608` calls `kappa_pure`) | `curve.py:228-237` (`kappa_pure`) | PASS — `|γ' × γ''| / |γ'|³` |
| **Base** `torsion` | — (no C++ impl; Python `Curve.torsion_impl` at `curve.py:660` calls `torsion_pure`) | `curve.py:258-266` (`torsion_pure`) | PASS — `(γ' × γ'') · γ''' / |γ' × γ''|²` |
| **Base** `dkappa_by_dcoeff` | — (Python implementation at `curve.py:619-658` via NumPy) | `kappavjp0`, `kappavjp1` JAX VJPs at `curve.py:240-249` | PASS — closed-form NumPy formula equivalent to autodiff (validated by Taylor tests in `tests/geo/test_curve.py:604`) |
| **Base** `dtorsion_by_dcoeff` | — (Python at `curve.py:672-`) | `torsionvjp{0,1,2}` JAX VJPs at `curve.py:269-283` | PASS |
| **Distance** linking number | `python_distance.cpp:176-215` | `curve_geometry.py:742-778` (`pair_linking_number_pure`) | PASS — scalar triple product matches; JAX requires no downsample (documented in docstring) |
| **Distance** segment-segment | (no C++ ref; JAX-only) | `curve_geometry.py:503-625` (`segment_segment_distance_pure`) | JAX-only; Sunday/Lumelsky algorithm; differentiable |

---

## Detailed findings

### (a) Gamma and derivatives parity

#### XYZ Fourier (CRITICAL-axis): formula match exact

C++ `gamma_impl` (`src/simsoptpp/curvexyzfourier.cpp:5-17`):

```cpp
for (int j = 1; j < order+1; ++j) {
    data(k, i) += dofs[i][2*j-1]*sin(2*M_PI*j*quadpoints[k]);
    data(k, i) += dofs[i][2*j]*cos(2*M_PI*j*quadpoints[k]);
}
```

JAX `jaxfouriercurve_pure` (`src/simsopt/geo/curvexyzfourier.py:309-325`) computes `coeffs @ basis`, where `basis` is built by `_fourier_basis_terms` (`curvexyzfourier.py:46-96`):

```python
points = two_pi * quadpoints  # 2π·q
phase = jnp.expand_dims(mode_numbers, axis=1) * jnp.expand_dims(points, axis=0)
# basis row j (j>=1) = [sin(2π·j·q), cos(2π·j·q)] interleaved
```

The DOF interleaving in `_interleave_harmonics(sin_phase, cos_phase)` matches the C++ `[xc(0), xs(1), xc(1), xs(2), xc(2), ...]` layout exactly (`_make_names` at `curvexyzfourier.py:134-155` confirms). PASS.

For derivatives, the C++ uses explicit `2*M_PI*j` factors (lines 25-26 for `gammadash`, lines 38-39 for `gammadashdash`, lines 51-52 for `gammadashdashdash`). The JAX `_fourier_basis_terms` builds `dash_basis`, `dashdash_basis`, `dashdashdash_basis` with `mode_scale = 2π·j`, `mode_scale_sq`, `mode_scale_cu` (`curvexyzfourier.py:54-56`). Both formulas reduce to:

- `gammadash`: `+xs·2π·j·cos − xc·2π·j·sin` ✓
- `gammadashdash`: `−xs·(2π·j)²·sin − xc·(2π·j)²·cos` ✓
- `gammadashdashdash`: `−xs·(2π·j)³·cos + xc·(2π·j)³·sin` ✓

PASS.

The fast-path in `_direct_curve_geometry_terms` (`curve_geometry.py:251-261`) routes XYZ Fourier through `jaxfouriercurve_geometry_pure` (which returns all 4 derivatives in a single matmul), bypassing the JVP chain. The result tuple is sliced `geometry[: order + 1]` (line 261) where the parameter `order` is the derivative order (1, 2, or 3), not the Fourier order — this slice correctly returns `(gamma,)`, `(gamma, gammadash)`, etc.

#### RZ Fourier: matmul order vs. C++ loop equivalence

C++ `gamma_impl` (`curverzfourier.cpp:5-31`) computes per-mode contributions and rotates them via `cos(phi)`, `sin(phi)` factors:

```cpp
data(k, 0) += rc[i] * cos(nfp*i*phi) * cos(phi);
data(k, 1) += rc[i] * cos(nfp*i*phi) * sin(phi);
data(k, 2) += zs[i-1] * sin(nfp*i*phi);
```

JAX `curverzfourier_pure` (`curverzfourier.py:18-57`) factors out the `cos(phi)`, `sin(phi)` rotation:

```python
radius = jnp.sum(rc * cos(cos_phase), axis=1)  # Σ_i rc[i]·cos(nfp·i·phi)
# ... add rs, zc, zs as needed ...
return jnp.column_stack((radius * cosphi, radius * sinphi, z))
```

This is **algebraically identical** to the C++ inner-product-then-multiply-by-rotation pattern, since `Σ_i rc[i]·cos(nfp·i·phi)·cos(phi) = (Σ_i rc[i]·cos(nfp·i·phi))·cos(phi)`. The matmul reduction may differ at the last ulp due to reduction-order, but no formula mismatch. PASS.

Verified C++ DOF-counter ordering for stellsym/non-stellsym matches JAX slice ordering exactly (see Function-by-function table notes).

#### Planar Fourier: quaternion rotation matrix match

C++ rotation (`curveplanarfourier.cpp:41-43`) is unrolled inline. The equivalent 3×3 matrix is:

```
R = [[1-2(q2²+q3²),   2(q1q2-q3q0),   2(q1q3+q2q0)],
     [2(q1q2+q3q0),   1-2(q1²+q3²),   2(q2q3-q1q0)],
     [2(q1q3-q2q0),   2(q2q3+q1q0),   1-2(q1²+q2²)]]
```

JAX `_quaternion_rotation_matrix` (`curveplanarfourier.py:25-53`) constructs exactly this matrix. The pure path then does `base_curve @ rotation.T + center` (line 94), which equals `(R @ base_curve.T).T + center` — same as the C++ row-by-row computation. PASS.

Normalization parity: C++ `inv_magnitude` (line 5-13) returns `1/sqrt(s)` if `s != 0` else `1`. JAX `_normalized_quaternion` (`curveplanarfourier.py:17-22`) uses `jnp.where(norm_sq > 0, 1/sqrt(norm_sq), 1)`. Same semantics. PASS.

#### Higher derivatives via JVP chain

For RZ Fourier and Planar Fourier, the higher derivatives (`gammadash`, `gammadashdash`, `gammadashdashdash`) are produced by `_curve_geometry_terms_from_kernel` (`curve_geometry.py:228-248`):

```python
gamma, gammadash = jax.jvp(gamma_kernel, (quadpoints,), (tangents,))
gammadash_kernel = lambda qp: jax.jvp(gamma_kernel, (qp,), (tangents,))[1]
_, gammadashdash = jax.jvp(gammadash_kernel, (quadpoints,), (tangents,))
gammadashdash_kernel = lambda qp: jax.jvp(gammadash_kernel, (qp,), (tangents,))[1]
_, gammadashdashdash = jax.jvp(gammadashdash_kernel, (quadpoints,), (tangents,))
```

where `tangents = ones_like(quadpoints)` (line 225). Since the entire dependence on `q` flows through `phi = 2π·q`, the chain rule produces exactly the C++ post-scale factors `2π`, `4π²`, `8π³`. Verified symbolically for the RZ formula `cos(N·phi)·cos(phi)` (yielding `d² = -(N²+1)·cos(N·phi)·cos(phi) + 2N·sin(N·phi)·sin(phi)` which matches `curverzfourier.cpp:67`).

### (b) Length parity

C++ `incremental_arclength_impl` (`curve.cpp:36-42`):

```cpp
data(i) = sqrt(dg(i,0)² + dg(i,1)² + dg(i,2)²);
```

JAX `incremental_arclength_pure` (`curve.py:213-218`):

```python
return jnp.linalg.norm(d1gamma, axis=1)
```

Identical (per-quadpoint `|γ'|`, no quadrature weight applied at this stage). The actual length integral `L = ∫|γ'| dq ≈ Σ_i |γ'(q_i)| · (1/N)` is performed at the caller level (e.g., `CurveLength.J() = (2π/N) · Σ`). PASS.

The VJP coefficient derivative (`dincremental_arclength_by_dcoeff`) shares the C++ closed form `(1/|γ'|) · (γ' · ∂γ'/∂c)` (line 51) with the JAX `incremental_arclength_vjp` JIT (`curve.py:221-225`) which uses `jax.vjp` autodiff. PASS.

### (c) Curvature / torsion parity

#### kappa

JAX `kappa_pure` (`curve.py:228-237`):

```python
return jnp.linalg.norm(jnp.cross(d1gamma, d2gamma), axis=1) / jnp.linalg.norm(d1gamma, axis=1)**3
```

Standard formula `κ = |γ' × γ''| / |γ'|³`. There is **no direct C++ `kappa_impl`** (`curve.h:140` is a `throw logic_error` virtual stub); the Python `Curve.kappa_impl` at `curve.py:608-617` calls `kappa_pure` itself, so the JAX expression is the SSOT both for the JAX lane and the CPU-`gamma`-backed path. PASS.

#### torsion

JAX `torsion_pure` (`curve.py:258-266`):

```python
return jnp.sum(jnp.cross(d1gamma, d2gamma, axis=1) * d3gamma, axis=1) / jnp.sum(jnp.cross(d1gamma, d2gamma, axis=1)**2, axis=1)
```

This is the textbook `τ = (γ' × γ'') · γ''' / |γ' × γ''|²`. Note that `jnp.sum(cross² , axis=1) = |γ' × γ''|²` since the dot product of the cross with itself equals the sum of squares. PASS.

`Curve.torsion_impl` at `curve.py:660-670` calls `torsion_pure` for the value, and the closed-form `dtorsion_by_dcoeff_impl` at line 672 uses NumPy to compute the derivative directly. This is mathematically equivalent to (and a partial duplicate of) the JAX `torsionvjp{0,1,2}` chain at `curve.py:269-283`. PASS.

#### dkappa_by_dcoeff (the longest closed form)

C++ has no `dkappa_by_dcoeff_impl` (`curve.h:141` is a stub). The Python implementation at `curve.py:619-658` uses NumPy chain rule:

```python
numerator = np.cross(γ', γ'')
denominator = |γ'| = incremental_arclength
dkappa/dc = (1/(denominator³ · |numerator|)) · numerator · (cross(γ'_c, γ'') + cross(γ', γ''_c))
            - (|numerator| · 3 / denominator⁵) · (γ' · γ'_c)
```

This is the canonical product/quotient-rule expansion of `κ = |γ' × γ''| / |γ'|³`. Cross-checked against JAX `kappavjp0`/`kappavjp1` (`curve.py:240-249`) which derive the same gradient via autodiff. The Taylor tests at `tests/geo/test_curve.py:604` pin this consistency across all C++/JAX curve types. PASS.

### (d) Coefficient-derivatives parity

The JAX path uses `jacfwd(curve.gamma_jax)` (`curve.py:344`) to construct `dgamma_by_dcoeff` of shape `(N, 3, n_dofs)`. Since `gamma_pure` matches C++ `gamma_impl` line-for-line (verified above), autodiff produces the same Jacobian entries as the C++ closed-form `dgamma_by_dcoeff_impl`.

C++ DOF orderings (counters):

- **XYZ Fourier** (`curvexyzfourier.cpp:58-69`): `[xc(0), xs(1), xc(1), ..., yc(0), ..., zc(0), ...]`. Matches JAX coeff reshape `(3, 2·order+1)` (`curvexyzfourier.py:322`).
- **RZ Fourier stellsym** (`curverzfourier.cpp:131-149`): counter increments `rc[0..order]`, then `zs[1..order]`. JAX slices `rc=dofs[0:order+1]`, `zs=dofs[order+1:]` (`curverzfourier.py:24-28`). MATCH.
- **RZ Fourier non-stellsym** (`curverzfourier.cpp:139-149`): counter increments `rc, rs, zc, zs`. JAX matches with explicit slices (`curverzfourier.py:30-32`). MATCH.
- **Planar Fourier** (`curveplanarfourier.h:68-78`): `[rc, rs, q, center]`. JAX matches (`curveplanarfourier.py:60-65`). MATCH.

The Stage-2 production-scale test `test_curve_spec_pullback_production_scale_parity` (`tests/geo/test_curve_item05_closeout.py:122-171`) iterates four random coil seeds per class and pins `gamma_jax == gamma_cpu` at the `direct_kernel` tolerance lane (`rtol = atol = 1e-12`). This is the load-bearing parity gate for coefficient-derivatives parity, since `dgamma_by_dcoeff` parity follows from `gamma` parity through autodiff.

### (e) Per-curve-type parity

| Class | Position parity | dgamma parity (autodiff) | Tests in repo |
|---|---|---|---|
| `CurveXYZFourier` (C++) ↔ `jaxfouriercurve_pure` | PASS | PASS | `test_jax_native_path.py:128` (vs loop ref), `test_curve_item05_closeout.py:122` (vs C++ at `atol=1e-12`), Taylor tests `test_curve.py:582` |
| `CurveRZFourier` (C++) ↔ `curverzfourier_pure` | PASS | PASS | `test_curve_item05_closeout.py:122` (vs C++, `gamma` only), Taylor tests `test_curve.py:582` |
| `CurvePlanarFourier` (C++) ↔ `curveplanarfourier_pure` | PASS | PASS | `test_curve_item05_closeout.py:122` (vs C++, `gamma` only), Taylor tests `test_curve.py:582` |
| `CurveHelical` (no C++; JAX-only) | n/a | n/a | `test_curve.py` Taylor tests, `test_curve_helical.py` |
| `CurveXYZFourierSymmetries` (no C++; JAX-only) | n/a | n/a | `test_curve_item05_closeout.py:174` (production scale) |

---

## Lazy-import circularity (project memory check)

The project memory note (`project_curve_jax_core_import_cycle.md`) records that `simsopt.geo.curve` ↔ `simsopt.jax_core` cycle forces lazy imports inside `_as_jax_float64`, `_as_runtime_float64_ref`, and `_as_runtime_jax_float64` (`src/simsopt/geo/curve.py:64-85`). I verified that:

- These three helpers contain only `from ..jax_core._math_utils import …` deferred imports and delegate to `as_jax_float64` / `as_runtime_float64` from `simsopt/jax_core/_math_utils.py` (lines 40-41 and 68-69).
- The CPU-only fallback path (`if not _HAS_JAX`) returns `np.asarray(value, dtype=np.float64)` — identical numeric output.
- No correctness implications: the helpers are pure type-converters; they only choose between `jnp.asarray` (when JAX is available) and `np.asarray` (CPU-only fallback). PASS.

---

## Test coverage gaps

The following gaps reflect what is **not asserted directly** even though indirect Taylor-test consistency exists.

### LOW (test parity gap, no functional regression observed)

1. **No production-scale JAX-vs-C++ `gammadashdash` / `gammadashdashdash` assertion for RZ Fourier and Planar Fourier**. The `_assert_curve_spec_geometry_matches_live_curve` helper (`tests/integration/test_single_stage_jax_cpu_reference.py:1444-1460`) does check `gammadashdash`, but it is only parametrized over `CurveXYZFourier`, `CurvePerturbed`, `CurveFilament` (lines 3216-3225). The production-scale `test_curve_spec_pullback_production_scale_parity` (`tests/geo/test_curve_item05_closeout.py:122`) explicitly asserts only `gamma_jax == gamma_cpu` (line 162-171), with `gammadashdash` left implicit through autodiff.

   Recommendation: extend `test_curve_spec_pullback_production_scale_parity` to also assert `curve_geometry_from_dofs(spec, spec.dofs)[1]` (gammadash) and `[2]` (gammadashdash) against `curve.gammadash()` and `curve.gammadashdash()` at the `direct_kernel` lane for all four C++-backed classes. This would close the gap noted in the audit and convert the autodiff-derived parity assumption into a load-bearing assertion.

### INFO (out-of-scope confirmation)

2. **No CPU/C++ counterpart for `pair_linking_number_pure`** within unit tests. The kernel is documented in its docstring (`curve_geometry.py:751-763`) as mirroring `python_distance.cpp:181-211`. The downsample parameter is intentionally absent from the JAX path. A pinning test against the C++ output for dense quadrature (downsample=1) would help calibrate the round-to-int boundary near `4π`, but no such explicit test was located.

3. **No direct test for `segment_segment_distance_pure`** that bench-marks against the C++ collision-distance helpers (`get_pointclouds_closer_than_threshold_*` in `python_distance.cpp`). However, these C++ helpers do not implement the Sunday/Lumelsky segment-segment distance directly, so this absence is expected.

---

## Recommended actions, ordered by severity

There are no CRITICAL/HIGH/MEDIUM findings. The recommendations below are LOW/INFO actions to harden the existing parity story.

1. **LOW** — Extend `tests/geo/test_curve_item05_closeout.py::test_curve_spec_pullback_production_scale_parity` to also pin `gammadash` and `gammadashdash` against the CPU oracle for all four C++-backed curve classes (`CurveXYZFourier`, `CurveRZFourier`, `CurvePlanarFourier`, `CurveHelical`). The helper functions are already imported and the spec roundtrip is established; the change is mechanical (~10 LOC).

2. **INFO** — Consider adding a `tests/geo/test_curve_linking_number_parity.py` row that pins `pair_linking_number_pure` against `simsoptpp.compute_linking_number` on a dense (downsample=1) trefoil-pair fixture. Use the curve fixtures already in `tests/geo/test_curve.py` (`test_trefoil_nonstellsym`, `test_trefoil_stellsym`).

3. **INFO** — No action needed on lazy-import helpers. They are correct and document the cycle clearly. If a future refactor consolidates `simsopt.geo.curve` and `simsopt.jax_core`, the lazy block can be lifted to top-level imports without correctness risk.

4. **INFO** — Reduction-order awareness: the JAX `curverzfourier_pure` uses `jnp.sum(rc[None,:] * jnp.cos(cos_phase), axis=1)` whereas the C++ uses a Python-loop summation per mode. On CPU at low order the difference is at the last few ulp (well within the `1e-12` tolerance); on GPU with deterministic XLA the reduction order is fixed by the matmul kernel. No action required.
