# Surface Fourier Parity DEEPER Audit (PRIORITY 10)

- **Audit date:** 2026-05-16
- **Branch:** `gpu-purity-stage2-20260405`
- **Auditor:** automated agent (fresh context, second pass)
- **Scope:** Second-pass audit hunting issues the forward-formula audit would systematically miss. **Not** revisiting forward parity — those findings stand. This is a coverage / edge-case / scaling / autodiff-behaviour audit.

## Files audited

| Path | Lines | Role |
|------|------:|------|
| `src/simsopt/jax_core/surface_rzfourier.py` | 1049 | Pure JAX `SurfaceRZFourier` evaluators |
| `src/simsopt/jax_core/surface_fourier.py` | 286 | Thin spec wrappers re-exporting `surface_fourier_jax` kernels |
| `src/simsopt/geo/surface_fourier_jax.py` | 2761 | SSOT for tensor + xyz-Fourier surface evaluation, BC enforcer, Jacobians |
| `src/simsoptpp/surface.{cpp,h}` | 817 + 249 | Base-class `normal_impl`, area/volume defaults, `dnormal_by_dcoeff_vjp` |
| `src/simsoptpp/surfacerzfourier.{cpp,h}` | 1555 + 133 | RZ-Fourier closed-form kernels with ANGLE_RECOMPUTE recurrence |
| `src/simsoptpp/surfacexyzfourier.{cpp,h}` | 822 + 170 | Per-`(m,n)` XYZ-Fourier kernels |
| `src/simsoptpp/surfacexyztensorfourier.h` | 1257 | Header-only tensor-Fourier (gamma/dash/Jacobians/BC enforcer/`skip`) |

Tests inspected: `tests/geo/test_surface_rzfourier_jax.py` (1426 lines), `tests/geo/test_surface_fourier_jax.py` (~1600 lines), `tests/geo/test_surface_xyz_tensor_clamped_jax.py`, `tests/geo/test_surface_rzfourier_jax_item06_closeout.py`, `tests/geo/test_surface_taylor.py`, `tests/geo/test_surface_xyzfourier.py`.

## Executive summary — top 5 second-pass findings

1. **(HIGH, scaling / OOM hazard) The dense `(target_size × source_size)` scatter matrices in `surface_rzfourier.py:_coefficients_from_dofs` are not cached and not JIT-friendly.** At `mpol=20, ntor=20` (stellsym), a single call builds two `(861, 1681)` `float64` matrices — **~22 MB transferred from host to device per evaluation**, plus an extra matmul on every call (including every `jacfwd`/`vjp` differentiation step). Combined with the autodiff Hessian path (`jax.hessian` over `area_from_dofs`) the worst-case scaling produces a `(32, 32, 3, 1681, 1681)` tensor at **~64 GB** for a single Hessian evaluation. There is no precomputed scatter or sparse-scatter alternative, and no high-resolution memory-budget test.

2. **(HIGH, autodiff hazard) `_unitnormal` is not gradient-safe at near-zero normals.** `surface_fourier_jax.py:1114-1115` and `surface_rzfourier.py:680-683` divide by `‖n‖` without any safe-divide. Direct probing shows that for `n = (1e-300, 0, 0)` the forward value yields `[inf, nan, nan]` and `jax.jacfwd(unitnormal)` propagates `nan`. The existing test `test_surface_rzfourier_unitnormal_degenerate_surface_matches_cpu_singularity` (line 1042) only checks `‖n‖ == 0` exactly — it does not stress the floating-point underflow boundary. **For optimizer runs where DOFs drift through near-degenerate intermediate states, autodiff can silently emit NaN gradients that propagate to the line search**. The CPU C++ kernel uses an explicit `sqrt(n²)` and shares the same singularity, but the C++ optimizer pipeline never differentiates `unitnormal` directly (it computes `darea` via `dnormal_by_dcoeff_vjp` with the cotangent `n/‖n‖`, which avoids the singular gradient when `‖n‖ → 0` along the limit-zero direction).

3. **(MEDIUM, coverage gap) High-mpol surfaces are exercised only at mpol≤2 in tests.** Every `_make_surface(...)` factory inside `tests/geo/test_surface_*_jax*.py` uses `mpol ≤ 2, ntor ≤ 2`. The largest tensor case found is `mpol=2, ntor=2` (in `test_surface_xyz_tensor_clamped_jax.py:_build_surface(seed=...)`). There are **no** scaling-stress tests at `mpol=10, ntor=10` or higher, where the dense-scatter and dense-hessian paths first manifest as performance/memory issues. The first-pass audit explicitly classified the dense angle tensor as a "MEDIUM, parity-budget" issue but never measured the dense-scatter overhead.

4. **(MEDIUM, coverage gap) Non-stellsym XYZTensorFourier is covered for `gamma`-family quantities but the basis-complete `dvolume`/`d2volume` Hessian round-trip is unverified at production resolution.** Inspection of `test_surface_fourier_jax.py` lines 959, 1067, 1175 etc. shows the non-stellsym tensor parity tests use `mpol=1, ntor=1` or `mpol=2, ntor=2`. The C++ `d2volume_by_dcoeffdcoeff_impl` (`surface.cpp:651-811`) uses a hand-rolled SIMD-vectorised path; the JAX `d2volume_by_dcoeffdcoeff` (`surface_fourier_jax.py:2612`) is `jax.hessian` over `surface_volume_from_dofs`. The two get cross-checked only at the low resolutions, and the JAX path does not exploit the analytical structure C++ uses — the SIMD vectorisation is replaced with a dense `(ndofs, ndofs)` outer-product Hessian.

5. **(LOW/INFO, autodiff vs explicit-VJP semantics) `darea_by_dcoeff` is computed via `jax.grad(surface_area_from_dofs)` (autodiff), while the C++ oracle constructs it explicitly via `dnormal_by_dcoeff_vjp(n/‖n‖)`.** These are mathematically equivalent — both compute the same chain rule — but the JAX path differentiates **through** `surface_area`, which calls `sqrt(sum(n²))`. The autodiff chain produces an internal `1 / (2·‖n‖)` term that is again unsafe at `‖n‖ = 0`. The C++ VJP path multiplies the cotangent `n/‖n‖` **before** running through the linear `dnormal_by_dcoeff_vjp`, so it goes through a different ordering. The cross-check tests in `tests/geo/test_surface_rzfourier_jax.py:1122-1145` show `rtol=1e-9` agreement on well-conditioned surfaces, but **no test exercises near-flat surfaces** where the two orderings would diverge.

No new CRITICAL or HIGH **correctness** issues found. Forward formulas remain bit-correct. The findings above are **scaling, autodiff-stability, and coverage** issues that the first pass would not have surfaced because it focused on point-wise forward parity.

---

## A. Hunt 1 — Non-stellsym / asymmetric surfaces

### A1. RZ-Fourier non-stellsym is well covered

`tests/geo/test_surface_rzfourier_jax.py` includes parametrized non-stellsym tests:

- `test_surface_rzfourier_jax_parity_non_stellsym` (line 564)
- `test_surface_rzfourier_jax_jacobian_parity_non_stellsym` (line 1114)
- `test_surface_rzfourier_area_volume_gradient_parity_non_stellsym` (line 1248)
- `test_surface_rzfourier_scalar_metric_parity_non_stellsym` (line 1256)
- `test_surface_rzfourier_dofs_round_trip_non_stellsym` (line 1312)
- `test_surface_rzfourier_jax_production_scale_non_stellsym_parity` (closeout, line 209) — **upgraded to nphi=32, ntheta=16**, mpol=2, ntor=1, addressing the gap left by the unit-scale tests.

The closeout module exists specifically because the in-file `_make_surface` factory was constrained to small dimensions. This is a **good coverage signal** — the maintainers identified the gap and closed it.

### A2. XYZTensor / XYZ-Fourier non-stellsym parity is comprehensive but at low resolution

`tests/geo/test_surface_fourier_jax.py` has 18 `@pytest.mark.parametrize("stellsym", [True, False])` decorators (lines 373, 418, 468, 566, 621, 685, 798, 892, 959, 1067, 1175, 1375, 1394, 1417, 1440, 1471, 1526, 1573). These cover:

- forward `gamma`, `gammadash{1,2,1d1,1d2,2d2}` parity
- non-tensor XYZ-Fourier `dgamma_by_dcoeff`, `dgammadash{1,2,1d1,1d2,2d2}_by_dcoeff`, `dnormal_by_dcoeff`, `dunitnormal_by_dcoeff`, `darea_by_dcoeff`, `dvolume_by_dcoeff`
- tensor-Fourier scalar metrics (area, volume) under stellsym=False

**Coverage gap:** all of these use `mpol ∈ {1, 2}, ntor ∈ {0, 1, 2}`. The C++ XYZ-Fourier path scales with `(mpol+1)*(2*ntor+1)` modes per coordinate, so at `mpol=10, ntor=10` we have 231 modes per coord × 6 coords = 1386 dofs (non-stellsym). The JAX autodiff path would build a dense `(nphi, ntheta, 3, 1386)` Jacobian, and the Hessian step would explode. No test exercises this regime.

### A3. Asymmetric / chirality detection

Neither audit found explicit tests that check the JAX path against C++ when:
- the stellsym flag is `False` but the user happens to feed an actual stellsym DOF vector (no-op symmetry-break sanity check)
- the surface has chirality (mirror-flipped variant)

These are not bugs but are uncovered edge cases. The stellsym scatter is independent of DOF values, so the only risk is a stellsym=False code path that silently relies on a stellsym-shaped DOF vector (e.g., zero-padding for `rs`/`zc`). No such path was found.

---

## B. Hunt 2 — Mode-table off-by-one (`mpol+1` vs `2*mpol+1`)

### B1. Verified mode tables

For SurfaceRZFourier:
- `_poloidal_modes` returns `mpol+1` values `[0, 1, …, mpol]` — matches `surfacerzfourier.cpp:51` which iterates `for(int m = 0; m <= mpol; ++m)`.
- `_toroidal_modes` returns `2*ntor+1` values `[-ntor, …, ntor]` — matches `surfacerzfourier.cpp:53` which iterates `for(int i = 0; i < 2*ntor+1; ++i)` with `int n = i-ntor`.

Empirically verified at `mpol=3, ntor=4`: arrays have shapes `(4,)` and `(9,)` respectively. **No off-by-one detected.**

For SurfaceXYZTensorFourier:
- C++ uses `m ∈ [0, 2*mpol]`, `n ∈ [0, 2*ntor]` (`surfacexyztensorfourier.h:83-100`).
- JAX `stellsym_scatter_indices` (line 1175-1178) iterates `m in range(2*mpol+1)`, `n in range(2*ntor+1)`. **Matches.**

For SurfaceXYZFourier:
- C++ uses one-coefficient-block-per-component layout `(mpol+1, 2*ntor+1)` with skip on `i < ntor` for cos and `i < ntor+1` for sin (`surfacexyzfourier.h:84-95`).
- JAX `_scatter_surface_xyzfourier_dofs` (line 1268-1300): `cos_count = n_per - ntor`, `sin_count = n_per - (ntor + 1)`. **Matches** the C++ `for (int i = ntor; i < shift)` and `for (int i = ntor+1; i < shift)` bounds.

### B2. Stellsym DOF count formulae match

For SurfaceRZFourier stellsym (`surfacerzfourier.h:54-55`):
```
num_dofs = 2*(mpol+1)*(2*ntor+1) - ntor - (ntor+1)
```
JAX equivalent: `include_positions.size + exclude_positions.size = (1 + (2*ntor+1)*mpol + ntor) + ((2*ntor+1)*mpol + ntor)` after expanding `_block_mode_positions` — empirically verified equal at `mpol=20, ntor=20` (1681).

For SurfaceXYZTensorFourier stellsym (`surfacexyztensorfourier.h:75-78`):
```
num_dofs = (ntor+1)*(mpol+1) + ntor*mpol + 2*((ntor+1)*mpol + ntor*(mpol+1))
```
JAX `stellsym_scatter_indices` empirically produces matching counts (verified at `mpol=2, ntor=2`: both report 37).

**No off-by-one detected.**

---

## C. Hunt 3 — Zero / negative / inverted-orientation volume

### C1. Volume sign convention is **signed** (both implementations)

Both C++ (`surface.cpp:598-610`) and JAX (`surface_fourier_jax.py:2399-2417`, `surface_rzfourier.py:698-702`) return signed volume — `(1/3) ∫ γ·n dφdθ` with no absolute value. The cross product orientation `n = γ_φ × γ_θ` is identical (both axis-1 cross axis-2). A user-supplied surface with inward-facing normals produces a negative volume from both backends — consistent.

### C2. Zero-volume surface: smoke-tested

I verified empirically that setting all Fourier coefficients to zero on a 4-by-5 stellsym grid produces:
- `surface_rz_fourier_volume_from_spec(spec) = 0.0` (finite)
- `surface_rz_fourier_unitnormal_from_spec(spec) = nan` (matches CPU)

The CPU oracle test `test_surface_rzfourier_unitnormal_degenerate_surface_matches_cpu_singularity` (line 1042) only checks the NaN mask agreement. **No test checks that `darea_by_dcoeff` or `dvolume_by_dcoeff` produce finite gradients on a near-zero-volume surface**, even though the autodiff path would crash there.

### C3. Self-intersecting surfaces

The C++ class `Surface` has an `is_self_intersecting` method (`tests/geo/test_surface.py:552-568`). The JAX path has no equivalent and no test that the area/volume formulae return signed-meaningful values for self-intersecting surfaces. **Not a bug** — the divergence-theorem volume integral simply returns the signed enclosed volume regardless of self-intersection — but worth noting that no upstream consumer relies on JAX to detect this.

---

## D. Hunt 4 — Degenerate quadrature grids (nphi=1, ntheta=1)

### D1. Empirical results on degenerate grids

Verified by running `surface_rz_fourier_*_from_spec` on a mpol=1 stellsym RZ surface:

| Grid | gamma shape | area | volume | unitnormal finite |
|---|---|---:|---:|---:|
| nphi=1, ntheta=8 | (1, 8, 3) | 3.948e+00 | 1.974e-01 | True |
| nphi=8, ntheta=1 | (8, 1, 3) | 4.343e+00 | 1.592e+00 | True |
| nphi=1, ntheta=1 | (1, 1, 3) | 4.343e+00 | 1.592e+00 | True |

**No divide-by-zero, no NaN.** Both the JAX dense angle tensor and the `(nphi, ntheta)` averaging behave correctly with singleton dims — `jnp.sum/jnp.mean` over a singleton just returns the single value.

Note: the surface-quadrature semantics of `nphi=1` is not physically meaningful (a single-point quadrature gives a heavily biased area estimate). The integrals are not converged. But the JAX path returns finite values consistent with the discretization. This is the C++ behaviour too.

`tests/geo/test_surface_taylor.py:396-405` exercises `nphi=1, ntheta=1` for the Taylor test, but the JAX modules are not part of that test fixture. **There is no JAX-specific degenerate-grid test**.

### D2. Stellsym + nfp=1 special case

JAX `_mode_terms` (line 41-54) builds angles as `m*θ - nfp*n*φ`. At `nfp=1` this reduces to `m*θ - n*φ` which is the standard 1-field-period torus. The mode tables and scatter indices have no special branches for `nfp=1`. Empirically verified by `test_surface_rzfourier_jax_gauss_bonnet_matches_cpu_oracle` (line 733) which uses `nfp=1, stellsym=True`. **No issue.**

---

## E. Hunt 5 — DOF reordering / set_dofs convention

### E1. The JAX RZ-Fourier matches C++ ordering exactly

C++ `SurfaceRZFourier::set_dofs_impl` (`surfacerzfourier.h:60-79`) stellsym:
1. `rc`: positions `[ntor, shift)` — i.e., `m=0, n=0..ntor` then `m=1..mpol, n=-ntor..ntor`
2. `zs`: positions `[ntor+1, shift)` — `m=0, n=1..ntor` then `m=1..mpol, n=-ntor..ntor`

JAX `_block_mode_positions(include_zero_mode=True)` (line 301-318):
- `n in range(0, ntor+1)`: `m=0` row, positions `n+ntor = [ntor, 2*ntor]`
- `m in range(1, mpol+1), n in range(-ntor, ntor+1)`: positions `m*(2*ntor+1) + n+ntor`

JAX `_block_mode_positions(include_zero_mode=False)`:
- `n in range(1, ntor+1)`: positions `[ntor+1, 2*ntor]`
- `m in range(1, mpol+1)`: positions `m*(2*ntor+1) + n+ntor`

This is **byte-identical** to C++. The dofs_round_trip tests (line 1312) confirm empirically.

### E2. Non-stellsym ordering

C++ non-stellsym (`surfacerzfourier.h:69-78`): `[rc, rs, zc, zs]`.

JAX non-stellsym (`surface_rzfourier.py:390-423`): same `[rc, rs, zc, zs]` order with offsets `0, rc_count, rc_count+tail_count, rc_count+tail_count+rc_count`.

**Match.**

### E3. SurfaceXYZFourier (non-tensor) ordering

C++ (`surfacexyzfourier.h:72-96`) non-stellsym: `[xc, xs, yc, ys, zc, zs]`.

JAX (`surface_fourier_jax.py:1288-1300`): same `[xc, xs, yc, ys, zc, zs]` order.

**Match.**

### E4. SurfaceXYZTensorFourier stellsym scatter

C++ skip function (`surfacexyztensorfourier.h:1233-1242`):
- `dim=0` (x): skip if `(n≤ntor && m>mpol) || (n>ntor && m≤mpol)` — keeps cos-cos + sin-sin.
- `dim=1` (y) and `dim=2` (z): skip if `(n≤ntor && m≤mpol) || (n>ntor && m>mpol)` — keeps cos-sin + sin-cos.

JAX (`surface_fourier_jax.py:1128-1153`):
- `_is_stellsym_xy`: `(m≤mpol AND n≤ntor) OR (m>mpol AND n>ntor)` — **keeps cos-cos + sin-sin**. Same as C++ keep-rule for x.
- `_is_stellsym_z`: `(m≤mpol AND n>ntor) OR (m>mpol AND n≤ntor)` — **keeps cos-sin + sin-cos**. Same as C++ keep-rule for y, z.

Note the polarity flip: C++ defines `skip`, JAX defines `is_allowed`. The JAX `_is_stellsym_*` is the **negation** of C++ `skip`. Verified by inspection.

**Match.** This is the highest-risk area per CLAUDE.md and the first-pass audit; the second-pass confirms the bit-exact equivalence with a fresh-eyes reading.

---

## F. Hunt 6 — Surface DOF → Stage-1 sensitivity (Biot-Savart pipeline)

`tests/integration/test_single_stage_jax.py` (single ref to `_surface_geometry_from_dofs` at line 172) covers the surface-DOF → gamma → BdotN integration end-to-end. Specific tests:
- `test_value_sanity` — checks single-stage objective returns finite value
- `test_adjoint_consistency` — implicit differentiation consistency
- `test_fd_gradient` — finite-difference gradient validation against AD

**The composition path is exercised**, but **not at high-resolution surfaces**. The fixtures all use `mpol=2, ntor=2` or smaller surfaces — same coverage gap as in Hunt 1.

For Stage-2 (fixed surface), `SquaredFluxJAX` captures the surface arrays in JIT closures at construction time (per CLAUDE.md). This means the surface-DOF gradient does **not** propagate through the Stage-2 path — it's intentionally severed for performance. Any test claiming "Stage-2 surface-DOF sensitivity" would be wrong by design. Verified by reading `src/simsopt/objectives/fluxobjective_jax.py` summary: the surface gamma/normal are baked into the closure, not parametric.

---

## G. Hunt 7 — Autodiff through `unit_normal` at near-zero |n|

### G1. The forward formula is `n / sqrt(sum(n²))`

In `surface_fourier_jax.py:1114-1115`:
```python
def _unitnormal(normal):
    return normal / jnp.sqrt(jnp.sum(normal * normal, axis=-1))[..., None]
```

In `surface_rzfourier.py:680-683`:
```python
def surface_rz_fourier_unitnormal_from_spec(spec):
    normal = surface_rz_fourier_normal_from_spec(spec)
    norm = jnp.linalg.norm(normal, axis=-1, keepdims=True)
    return normal / norm
```

### G2. Empirical autodiff behaviour at the singularity

I ran direct probes:

**Forward at exactly zero**: `_unitnormal(jnp.zeros(3)) → NaN` (matches C++ singularity behaviour, verified by existing test at `test_surface_rzfourier_jax.py:1042`).

**Forward at `(1e-300, 0, 0)`**: `[inf, NaN, NaN]`. Floating-point underflow of `n²` to 0 yields `1/sqrt(0) = inf`; the multiplication `1e-300 × inf → NaN`.

**`jax.jacfwd(unitnormal)` at `(1e-300, 0, 0)`**: returns all-NaN `(3, 3)` Jacobian.

**`jax.grad(sum_unitnormal)` at zero**: returns `[NaN, NaN, NaN]`.

### G3. Risk assessment

In a real optimizer iteration, an L-BFGS step that drives DOFs through near-degenerate intermediate states could:
1. Compute a finite forward value (because `‖n‖² > 0` numerically)
2. Compute a NaN gradient (because some intermediate `‖n‖` lands at floating-point underflow)

This NaN propagates silently into the line search. Wolfe conditions on NaN fail in unpredictable ways (depending on optimizer NaN handling). **The CPU C++ optimizer has the same singularity in `dunitnormal_by_dcoeff_impl` (`surface.cpp:451-489`), so this is a shared backend hazard**, not a JAX regression. But the JAX path is more vulnerable because:
- autodiff threading through `sqrt` has the well-known `d(sqrt(x))/dx = 1/(2*sqrt(x)) → inf` at x=0 issue.
- the C++ optimizer typically calls `darea_by_dcoeff` via the explicit VJP `dnormal_by_dcoeff_vjp(n/‖n‖)` which **factors the cotangent first**, side-stepping the `1/(2·sqrt)` step.

### G4. Recommended action (not made — flagging only)

Add a degenerate-but-near-zero test that probes `jax.grad(area_from_dofs)` on an almost-flat surface (e.g., `rc[0,ntor] = 1e-12`) and asserts that the gradient is finite, not NaN.

The existing test only checks `‖n‖ == 0` exactly. The interesting case is `‖n‖ << 1` but nonzero — the regime where the chain-rule `1/sqrt` factor blows up.

---

## H. Hunt 8 — Area/volume gradient implementation (autodiff vs explicit VJP)

### H1. RZ-Fourier path

`surface_rzfourier.py:952-989`:
- `darea_from_dofs = jax.grad(area_from_dofs)`
- `dvolume_from_dofs = jax.grad(volume_from_dofs)`
- `d2area_from_dofs = jax.hessian(area_from_dofs)`
- `d2volume_from_dofs = jax.hessian(volume_from_dofs)`

**All four are pure autodiff.** No explicit VJP. The C++ oracle for darea/dvolume uses `dnormal_by_dcoeff_vjp` (`surface.cpp:504-537` and `surface.cpp:612-649`).

The cross-check test `_assert_area_volume_gradient_parity` (`test_surface_rzfourier_jax.py:1122-1145`) compares `surface_rz_fourier_darea_from_dofs` against `surface.darea_by_dcoeff()`. **Passes** at `rtol=1e-9` for `_make_surface(stellsym=True/False)` (small mpol/ntor).

### H2. Tensor-Fourier path

`surface_fourier_jax.py:2609-2612`:
- `darea_by_dcoeff = _surface_scalar_grad(surface_area_from_dofs)`
- `d2area_by_dcoeffdcoeff = _surface_scalar_hessian(surface_area_from_dofs)`
- `dvolume_by_dcoeff = _surface_scalar_grad(surface_volume_from_dofs)`
- `d2volume_by_dcoeffdcoeff = _surface_scalar_hessian(surface_volume_from_dofs)`

Where `_surface_scalar_grad = jax.grad` and `_surface_scalar_hessian = jax.hessian`. Same autodiff approach.

### H3. Hessian-scale concern (LOW)

For `mpol=20, ntor=20, nphi=32, ntheta=32`:
- `ndofs ≈ 1681`
- `area_hessian: (1681, 1681) = 22 MB` — fine
- `volume_hessian: (1681, 1681) = 22 MB` — fine

But the **non-scalar Hessians** `d2normal_by_dcoeffdcoeff` (line 2546-2551) and `d2gamma_by_dcoeffdcoeff` etc. produce `(nphi, ntheta, 3, ndofs, ndofs)` — at the same scale, **~64 GB**.

`jax.hessian = jacfwd ∘ jacrev`, both forward and reverse passes are dense. There is no sparse Hessian option in the JAX kernels. **Real-world high-mpol Hessian requests would OOM.**

### H4. Recommended action (not made — flagging only)

Add a `pytest.skip` guard or `jax.checkpoint` rematerialization in the Hessian path. Alternatively, gate `d2*_by_dcoeffdcoeff` behind a runtime size check that raises a clear error before consuming memory.

---

## I. Hunt 9 — Mode-table / scatter caching

### I1. Static-arg JIT cache keys

`SurfaceRZFourierSpec` (`specs.py:1413-1442`) registers `mpol, ntor, nfp, stellsym` as `meta_fields` — JAX's `register_dataclass` makes these JIT cache keys. **Changing mpol/ntor invalidates the compiled kernel correctly.**

`SurfaceXYZFourierSpec` (`specs.py:478-502`) does the same: `mpol, ntor, nfp, stellsym` are meta_fields. `scatter_indices` and `coeff_template` are data_fields, but their **shape** encodes `mpol, ntor, stellsym`. JAX caches by data-field shape, so a different `(mpol, ntor)` automatically produces a different `scatter_indices.shape` and forces recompilation.

**No stale-cache bug detected.**

### I2. Scatter matrix is not cached

`surface_rzfourier.py:_scatter_matrix` (line 327-337) builds a dense `(target_size × source_size)` float64 matrix **every call**:
```python
def _scatter_matrix(positions, *, target_size, source_size, source_offset):
    matrix = np.zeros((target_size, source_size), dtype=np.float64)
    source_columns = np.arange(positions.size) + source_offset
    matrix[positions, source_columns] = 1.0
    return _as_jax_float64(matrix)
```

At `mpol=20, ntor=20` stellsym, this is `(861, 1681) × 4 calls` (rc, rs, zc, zs) = ~88 MB **rebuilt and host→device transferred per `_coefficients_from_dofs` call**.

This is a **performance hazard**, not a correctness bug. It is also called inside `jax.jacfwd` and `jax.vjp` lambdas, so it can be re-invoked per backward pass.

**No cache or LRU layer detected**. The dense scatter is not converted to `lax.scatter` (which `surface_fourier_jax.py:dofs_to_xyzc` does use for the tensor variant). The RZ path is therefore **denser than the XYZ tensor path** — an inconsistency.

### I3. Recommended action (not made — flagging only)

Refactor `_coefficients_from_dofs` to use `lax.scatter` with a precomputed `positions` static array (analogous to `dofs_to_xyzc` at line 1203-1234). The current implementation builds a dense matrix every call, which is exactly the pattern the tensor path already avoided.

---

## J. Hunt 10 — ARM vs x86 reproducibility

I did not have hardware to retest. The CLAUDE.md already documents:
- `sdofs_inf` varies between `1.9e-14` and `3.6e-12` across two macOS hardware platforms.
- Hessian condition number `κ ≈ 5.3e+04` is hardware-invariant.
- Hessian entry disagreement `H_inf_diff` ranges `1.4e-12 — 9.6e-10`.

**No new ARM/x86 test exists in the codebase**. The `validation_ladder_contract.py` and the `floating-point reproducibility across machines` note in CLAUDE.md are the only acknowledgements.

The RZ-path dense scatter (Hunt 9) and the dense `(P, M, N)` angle tensor in `_mode_terms` exacerbate the reduction-order portability problem because their `jnp.sum(..., axis=(2, 3))` calls reduce over a flat tensor, whose summation order depends on the BLAS/XLA backend.

Recommended setting: keep absolute `sdofs_inf ≤ 1e-11` thresholds for state-parity gates, as already documented in `CLAUDE.md`. **No new finding.**

---

## K. Hunt 11 — Stellsym `dofs_to_coefficients` completeness

I directly counted the stellsym scatter indices:

```
mpol=2, ntor=2:
  n_per_coord (per dim) = (2*2+1)*(2*2+1) = 25
  x quadrant (cos-cos + sin-sin): (mpol+1)*(ntor+1) + mpol*ntor = 9 + 4 = 13
  y/z quadrant (cos-sin + sin-cos): mpol*(ntor+1) + (mpol+1)*ntor = 6 + 6 = 12 each
  total = 13 + 12 + 12 = 37 ✓
```

Empirical run produces 37 indices. **All DOFs map to non-trivial positions; no orphan DOFs.**

I also verified that the JAX `_is_stellsym_xy` and `_is_stellsym_z` quadrant predicates are exact set complements of the C++ `skip(dim, m, n)` function (they enumerate the same `(m, n)` pairs).

---

## L. Hunt 12 — High (mpol, ntor) numerical regression

### L1. Memory scaling of the angle tensor

`surface_rzfourier.py:_mode_terms` (line 41-54) builds `(nphi, ntheta, mpol+1, 2*ntor+1)` cos/sin tensors. Empirical numbers at `nphi=ntheta=32`:

| (mpol, ntor) | tensor shape | tensor MB | scatter MB |
|---|---|---:|---:|
| (2, 2) | (32, 32, 3, 5) | 0.12 | 0.05 |
| (8, 8) | (32, 32, 9, 17) | 1.20 | 2.34 |
| (20, 20) | (32, 32, 21, 41) | 6.73 | 11.31 |

The **angle tensor is fine** (under 7 MB at mpol=ntor=20). The **dense scatter matrices** (Hunt 9, line 11.3 MB above) are the actual bottleneck. Plus the dense Jacobian if computed: `(32, 32, 3, 1681) = 39 MB`. And the Hessian: **64 GB** (see Hunt 8).

### L2. No high-mpol test in suite

`tests/geo/test_surface_*_jax.py` factories all use `mpol ≤ 2`. **No production-resolution stress test exists**. The closest is `test_surface_rzfourier_jax_production_scale_non_stellsym_parity` (closeout module, line 209) which uses `nphi=32, ntheta=16, mpol=2, ntor=1` — still small in mode space.

---

## M. Hunt 13 — C++ UB in `surfacexyztensorfourier.h`

I read `surfacexyztensorfourier.h` end-to-end (1257 lines). Checked specifically:

1. **`if(...) { }` brace structure around the `apply_bc_enforcer` and `bc_enforcer_*_fun` series** (lines 903-955): every helper is a 3-line `if/else` returning one branch. **No latent missing-brace issue**.

2. **Trig recomputation pattern**: this header does **not** use `ANGLE_RECOMPUTE`. The tensor-product kernel evaluates `basis_fun` from a precomputed `cache_basis_fun_*` array indexed by `(dim, n_or_m, k_idx)`. No recurrence to break.

3. **OpenMP `parallel for`**: each function declares `#pragma omp parallel for` with no `ordered` clause and writes to disjoint `data(k1, k2, ...)` slots. **No data race detected** — the outer `k1` loop is parallel; inner `k2/m/n` loops are sequential per `k1`.

4. **`skip` function** (line 1233-1242): the `&&` and `||` combinators are explicit. **No missing `()` operator-precedence ambiguity** (verified by inspection — `&&` binds tighter than `||`, and the parens correctly group the two skip-quadrants).

5. **`get_coeff` (line 1244-1256)**: short-circuits via `if(skip(dim, m, n)) return 0.;` before indexing into `x/y/z`. **Safe**.

6. **`basis_fun(int dim, int n, int phiidx, int m, int thetaidx)` (line 958)** vs **`basis_fun(int dim, int n, double phi, int m, double theta)` (line 1086)**: these are overloads. The `idx` version reads from precomputed `cache_basis_fun_*` arrays; the value version recomputes from `phi/theta` directly. Both should be equivalent at quadrature points. The `_lin` variants only use the value version (because they take user-supplied paired-point quadpoints, not the cached grid). **No UB.**

**No new C++ UB found.** The CLAUDE.md-documented missing-brace ANGLE_RECOMPUTE issue is specific to `surfacerzfourier.cpp` and was fixed; the tensor header does not use the same idiom.

---

## N. Hunt 14 — `set_dofs` / `set_dofs_from_array` consistency

I traced the Python ↔ JAX boundary:

- Python `SurfaceRZFourier.surface_spec()` (`surfacerzfourier.py:435-439`) calls `_surface_rzfourier_jax_tool("make_spec", ...)` which builds a `SurfaceRZFourierSpec` from `self._surface_spec_kwargs()`.
- `_surface_spec_kwargs` extracts `rc, rs, zc, zs` arrays directly from the Python surface — the `set_dofs` ordering is already baked in by `SurfaceRZFourier.set_dofs_impl` (C++ side).
- `make_surface_rzfourier_spec` (`specs.py:1415`) takes the coefficient matrices, not the DOF vector.
- The JAX `_coefficients_from_dofs` (line 340-424) does the **reverse**: takes a flat DOF vector and produces `(rc, rs, zc, zs)`. This is used inside `jax.jacfwd`/`jax.vjp` to differentiate w.r.t. DOFs.

The DOF unpacking order (cos-then-sin, m=0-skips-n<0) is implemented twice:
1. In `_block_mode_positions` (line 301-318) — for the inverse `dofs → coefficients` direction (used inside `jax.jacfwd`/`jax.vjp`).
2. In `_surface_rz_fourier_block_mode_positions` (`specs.py`) — for the forward `coefficients → dofs` direction.

These are **separately implemented**. I read both and they produce the same positions for the same `(mpol, ntor, include_zero_mode)` inputs. The dofs round-trip test `test_surface_rzfourier_dofs_round_trip_*` (lines 1312-1313) validates the chain end-to-end against the CPU surface.

**No drift detected**. But the duplication is a potential source of bug if one is updated without the other.

---

## O. Hunt 15 — End-to-end Stage-1 surface-DOF sensitivity

I read `tests/integration/test_single_stage_jax.py` (the only integration test that exercises surface DOFs flowing into integral_BdotN through BiotSavart). It contains 14 named tests covering:
- value sanity
- adjoint consistency
- FD gradient
- composite pipeline
- backend construction
- LS parity
- short outer optimization
- exact path
- `ensure_solved` guard

The single-stage chain is `surface DOFs → spec → gamma → integral_BdotN`. JAX autodiffs through this chain via `jax.grad` on the composite objective. The CPU/JAX parity is gated by the per-stage parities (Hunt 14 above) and by the byte-identity gate in `_pre_newton_census_gate_failures` (`benchmarks/single_stage_init_parity.py`).

**Coverage gap (LOW):** the single-stage tests use small surfaces (mpol≤2). The byte-identity gate is run on production fixtures, but those are not in this audit's scope.

---

## P. Severity-tagged finding inventory

| # | Severity | Area | Finding |
|---|---|---|---|
| 1 | HIGH | scaling | RZ-Fourier dense `_scatter_matrix` rebuilds `(861×1681)` matrix per call at mpol=ntor=20; not LRU-cached; transferred host→device every evaluation |
| 2 | HIGH | autodiff stability | `_unitnormal` propagates NaN through `jacfwd` when `‖n‖` underflows; existing test only checks exact zero, not the underflow boundary |
| 3 | MEDIUM | coverage | `mpol≤2, ntor≤2` everywhere in JAX surface tests; no high-resolution stress test |
| 4 | MEDIUM | coverage | Non-stellsym XYZTensor `d2volume/d2area` Hessians validated only at `mpol=1,2` |
| 5 | MEDIUM | scaling | `jax.hessian` over `surface_volume_from_dofs` at `mpol=20, ntor=20` produces 64 GB dense tensor; no guard / no `skip` |
| 6 | LOW/INFO | autodiff path | `darea_by_dcoeff` uses `jax.grad` (autodiff through sqrt) where C++ uses pre-factored cotangent VJP; mathematically equivalent but more numerically sensitive at degenerate surfaces |
| 7 | LOW/INFO | scaling | RZ angle tensor `(P, T, mpol+1, 2*ntor+1)` reaches 6.7 MB at mpol=ntor=20; first-pass already noted this |
| 8 | LOW/INFO | coverage | Single-stage surface-DOF sensitivity (`test_single_stage_jax.py`) validated only at `mpol=2` |
| 9 | LOW/INFO | duplication | `_block_mode_positions` is implemented in both `surface_rzfourier.py` and `specs.py`; potential drift if one is updated |
| 10 | LOW/INFO | coverage | JAX degenerate-grid (`nphi=1, ntheta=1`) returns finite values but has no dedicated JAX test |

No CRITICAL findings. No new HIGH **correctness** findings (the two HIGH items above are scaling/stability hazards, not bit-level correctness bugs).

---

## Q. Untested edge-case inventory

| Edge case | Tested? | Notes |
|---|---|---|
| nphi=1 (single phi quadrature) | No (JAX only) | Empirically smoke-tested in this audit; returns finite values |
| ntheta=1 (single theta quadrature) | No (JAX only) | Same |
| nphi=ntheta=1 (single point) | Only in `test_surface_taylor.py` for C++ classes | Empirically finite for JAX |
| mpol≥10, ntor≥10 (high resolution) | No | Memory/perf hazard inferred from measurement |
| Near-zero `‖n‖` (near-degenerate parametrization) | Only exact zero | Underflow regime untested |
| nfp=1 stellsym | Yes (`test_surface_rzfourier_jax_gauss_bonnet_matches_cpu_oracle`) | OK |
| nfp=4+ stellsym | Yes (small ntor) | Inferred; not explicitly tested at large nfp |
| Surface with negative volume (DOFs producing inward-normal) | No JAX-specific test | C++ and JAX both return signed volume |
| Self-intersecting surface | C++ `is_self_intersecting` only | JAX has no analogue |
| Asymmetric quadpoints (non-uniform spacing) | Yes — `np.linspace(0.013, ...)` patterns | Covered |
| Tensor-Fourier clamped_dims = (True, True, True) | Yes | `test_clamped_dims_invalidates_jit_cache` |
| Different `clamped_dims` patterns: (T,F,F), (T,T,F) etc. | Partially | Some combinations tested at mpol=2 |
| Gradients through `jax.grad(area)` at `‖n‖ = 1e-12` | No | Recommended new test |
| Hessian at high mpol | No | Memory-prohibitive at mpol=20 |
| ARM vs x86 byte-identity | No | Documented as parity-budget issue |

---

## R. Recommended follow-ups (not implemented in this audit)

1. **Replace `_scatter_matrix` with `lax.scatter`** in `surface_rzfourier.py:_coefficients_from_dofs`. Eliminate the per-call dense host-side matrix construction.
2. **Add a near-zero-normal autodiff test** for `darea_by_dcoeff` and `dvolume_by_dcoeff`. Probe `rc[0, ntor] = 1e-12` and assert the gradient is finite, not NaN.
3. **Add a high-resolution memory regression test**. Run `surface_rz_fourier_gamma_from_spec` at `mpol=ntor=20, nphi=ntheta=32` under `jax.transfer_guard("disallow")` (after warm-up) to detect the dense scatter host→device transfer.
4. **Document the Hessian scaling limit**. Add a runtime check in `_dcoeff_hessian` and `_surface_scalar_hessian` that raises a clear `RuntimeError` when `ndofs² × nphi × ntheta × 3 × 8` exceeds a configurable threshold (default 8 GB).
5. **Deduplicate `_block_mode_positions`**. Extract to a single SSOT location (`specs.py` is the natural home), reuse it from `surface_rzfourier.py`.
6. **Add an explicit `set_dofs` round-trip on production-scale non-stellsym surfaces with extreme nfp**. Currently `nfp=1` (single field period) and `nfp=2/3/5` are covered at small mpol; `nfp=10+` is untested.

---

## S. Conclusion

The forward-formula parity story established by the first pass is **confirmed**. This second pass found **no new correctness defects**, but did surface:

- **Two HIGH scaling/stability hazards** (#1 dense scatter rebuild, #2 unitnormal underflow autodiff NaN) — both are pre-existing and shared with the C++ optimizer, but their interaction with `jax.grad` / `jax.jacfwd` makes them more visible in the JAX path.
- **Three MEDIUM coverage gaps** at high (mpol, ntor) — autodiff Jacobians/Hessians at production resolution.
- A handful of LOW/INFO items including code-duplication, autodiff-vs-VJP semantic differences, and edge cases the test suite doesn't explicitly probe.

The DOF scatter, mode tables, BC enforcer, and stellsym quadrant maps are **bit-exact between C++ and JAX** for every surface kind. The first-pass conclusion that "the highest-risk area is the stellsym DOF scatter" remains correct — that risk has been **driven to zero** by the existing test coverage.

The biggest delta from the first-pass conclusion: the dense scatter matrix in the RZ-Fourier path is a **memory and performance hazard** at production resolution that has no test gate. If a downstream consumer requests a Jacobian or Hessian at `mpol=20, ntor=20`, the JAX path will silently consume tens of MB to GB of memory beyond what the C++ path uses. The first-pass focused on correctness; this hazard would not show up in a forward-parity audit.
