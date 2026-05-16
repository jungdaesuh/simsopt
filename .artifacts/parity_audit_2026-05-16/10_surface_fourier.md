# Surface Fourier Parity Audit (PRIORITY 10)

- **Audit date:** 2026-05-16
- **Branch:** `gpu-purity-stage2-20260405`
- **Auditor:** automated agent (fresh context)

## Files audited

| Path | Lines | Role |
|------|------:|------|
| `src/simsopt/jax_core/surface_rzfourier.py` | 1049 | Pure JAX `SurfaceRZFourier` evaluators (spec → arrays, autodiff helpers) |
| `src/simsopt/jax_core/surface_fourier.py`   |  286 | Thin spec wrappers that re-export `surface_fourier_jax` kernels for XYZ & XYZTensor |
| `src/simsopt/geo/surface_fourier_jax.py`    | 2761 | The pure-JAX SSOT for tensor + xyz-Fourier surface evaluation, BC enforcer, Jacobians |
| `src/simsoptpp/surface.h`                   |  249 | Base class API (caches, normal/area/volume default impls) |
| `src/simsoptpp/surface.cpp`                 |  818 | Default `normal_impl`, `unitnormal_impl`, `area`, `volume`, `dnormal_by_dcoeff_vjp`, etc. |
| `src/simsoptpp/surfacerzfourier.h`          |  133 | `SurfaceRZFourier` decl, dof packing |
| `src/simsoptpp/surfacerzfourier.cpp`        | 1555 | Trig-recurrence implementation of gamma/gammadash*/dgamma\*\_by\_dcoeff(\_vjp) |
| `src/simsoptpp/surfacexyzfourier.h`         |  170 | `SurfaceXYZFourier` decl |
| `src/simsoptpp/surfacexyzfourier.cpp`       |  822 | Per-`(m,n)` Fourier kernels for XYZ-Fourier |
| `src/simsoptpp/surfacexyztensorfourier.h`   | 1257 | Header-only `SurfaceXYZTensorFourier` (gamma / dash / Jacobians / BC enforcer / `skip`) |

Adjacent context inspected: `src/simsopt/geo/surface.py` (Python base) skipped because it is the same upstream `Surface` interface; `src/simsopt/jax_core/specs.py` (DOF helpers and `stellsym_scatter_indices` factory at lines 1474-1574); `tests/geo/test_surface_*_jax*.py` (7 files, 11420 lines).

## Executive summary — top 3 findings

1. **(MEDIUM, parity-budget) RZ derivative path uses dense `(P, M, N)` tensors with no `ANGLE_RECOMPUTE` recurrence.** C++ `SurfaceRZFourier::gamma_impl`, `gammadash{1,2}_impl`, and `dgamma{,dash1,dash2}_by_dcoeff_vjp` all share an explicit Chebyshev-style `sin/cos(m*theta - n*nfp*phi)` recurrence that resets every `ANGLE_RECOMPUTE = 5` iterations (`surfacerzfourier.cpp:20-62`). The JAX kernel materialises a `(nphi, ntheta, mpol+1, 2*ntor+1)` angle tensor (`surface_rzfourier.py:50-54`) and runs `jnp.cos`/`jnp.sin` once per element. Numerically the two reductions sum in different orders, so byte-identity CPU↔JAX state parity is not portable on the RZ LS path — this matches the documented "cross-machine `sdofs_inf ≤ 1e-11`" expectation. Action item: keep the lane-dependent tolerance in `validation_ladder_contract.py`; do **not** assert `rtol=1e-12` on RZ derivative-heavy fixtures.

2. **(HIGH/INFO, missing oracle) `_surface_xyzfourier_dnormal_by_dcoeff` and the XYZ-Fourier tensor Jacobians are not independently oracle-tested.** `surface_fourier_jax.py:2723-2761` builds the Jacobian / Hessian helpers by composing `jax.jacfwd`/`jax.hessian` over `surface_xyzfourier_normal_from_dofs`. The only direct C++ reference for XYZ-Fourier first-derivative Jacobians (`dgammadash1_by_dcoeff_impl`, `dgammadash2_by_dcoeff_impl`) is not invoked anywhere in `tests/geo/test_surface_fourier_jax.py` for the **basis-complete** column-by-column check. `surface.cpp` lines 393-425 (`dnormal_by_dcoeff_impl`) and 482-580 (`darea`/`dvolume_by_dcoeff_impl`) are excellent oracle candidates but are not exercised against the JAX path. This is a coverage gap, not a correctness defect.

3. **(LOW, cosmetic / consistency) The JAX RZ path's `_surface_rz_fourier_derivative_from_terms` (`surface_rzfourier.py:221-273`) uses a Leibniz-rule decomposition over `phi_order` that double-evaluates the underlying mode-derivative integrand (it calls `_radius_height_derivative_from_modes` once per `basis_order ∈ [0, phi_order]`).** This is functionally correct (verified — the `phi_factor**phi_order` scaling and the radial/toroidal phase shifts implement the chain rule for `(d/dphi)^phi_order [r cos phi]`), but it is roughly an `O(phi_order)` multiplier over the dense angle tensor and is only used by the second-derivative helpers `gammadash1dash1_from_spec`, `gammadash1dash2_from_spec`. No correctness issue; flagged for future simplification.

No CRITICAL or HIGH **correctness** issues found. The stellsym DOF scatter (the highest-risk area per `CLAUDE.md`) is **bit-exact** against the C++ `skip` function. The `m*theta − n*nfp*phi` angle convention, the `nfp` cancellation in volume/area integrals, and the BC enforcer formulae all match.

---

## A. Function-by-function parity matrix

### A1. `SurfaceRZFourier` (the C++ class) ↔ `src/simsopt/jax_core/surface_rzfourier.py`

| Quantity | C++ symbol | JAX function | Status | Notes |
|---|---|---|---|---|
| gamma | `gamma_impl` (`surfacerzfourier.cpp:25-127`) | `surface_rz_fourier_gamma_from_spec` (`surface_rzfourier.py:502-507`) | OK | Angle convention `m*theta − n*nfp*phi` matches. JAX builds dense (`P,T,mpol+1,2*ntor+1`) tensor; C++ uses ANGLE_RECOMPUTE recurrence. Reduction-order difference (see §G). |
| gamma_lin | `gamma_lin` (`surfacerzfourier.cpp:130-154`) | — | INFO | No paired-point RZ JAX entry point; only the dense grid is implemented. Most JAX adapters use dense grids, so this is by design — but record it as a port-gap if `BoozerSurface` ever calls `gamma_lin` from JAX. |
| gammadash1 | `gammadash1_impl` (`surfacerzfourier.cpp:450-556`) | `surface_rz_fourier_gammadash1_from_spec` (`surface_rzfourier.py:509-522`) | OK | Both apply the same `2π·(rd·cos(phi) − r·sin(phi))` rotation. Inner sums encoded as `rc·sin·scale − rs·cos·scale` with `scale = 2π·nfp·n` (JAX line 177) → C++ `rd += rc*(n*nfp)*sin + rs*(-n*nfp)*cos` (lines 476-479). Signs match. |
| gammadash2 | `gammadash2_impl` (`surfacerzfourier.cpp:654-754`) | `surface_rz_fourier_gammadash2_from_spec` (`surface_rzfourier.py:525-536`) | OK | Both apply `2π·(rd2·cos(phi), rd2·sin(phi), zd2)` with `rd2 = Σ −rc·m·sin + rs·m·cos`. JAX scale = `2π·m`. ✓ |
| gammadash1dash1 | `gammadash1dash1_impl` (`surfacerzfourier.cpp:559-588`) | `surface_rz_fourier_gammadash1dash1_from_spec` (`surface_rzfourier.py:539-553`) | OK | Implemented via Leibniz expansion in `_surface_rz_fourier_derivative_from_terms`, see §F. Output matches `4π²·(rdd·cos(phi) − 2·rd·sin(phi) − r·cos(phi))`. |
| gammadash1dash2 | `gammadash1dash2_impl` (`surfacerzfourier.cpp:591-617`) | `surface_rz_fourier_gammadash1dash2_from_spec` (`surface_rzfourier.py:556-570`) | OK | Same wrapper, with `(phi_order=1, theta_order=1)`. |
| gammadash2dash2 | `gammadash2dash2_impl` (`surfacerzfourier.cpp:620-649`) | `surface_rz_fourier_gammadash2dash2_from_spec` (`surface_rzfourier.py:573-587`) | OK | `(0,2)` order. ✓ |
| normal | `Surface::normal_impl` (`surface.cpp:382-392`) | `surface_rz_fourier_normal_from_spec` (`surface_rzfourier.py:673-677`) | OK | Both compute `n = γ_φ × γ_θ`. Cross product orientation identical (right-hand rule, axis 1 × axis 2). |
| unitnormal | `Surface::unitnormal_impl` (`surface.cpp:451-461`) | `surface_rz_fourier_unitnormal_from_spec` (`surface_rzfourier.py:680-683`) | OK | Standard `n / ‖n‖`. |
| area | `Surface::area()` (`surface.cpp:493-502`) | `surface_rz_fourier_area_from_spec` (`surface_rzfourier.py:692-695`) | OK | `A = Σ ‖n‖ / (nphi·ntheta)`. nfp cancels (see §D). |
| volume | `Surface::volume()` (`surface.cpp:598-610`) | `surface_rz_fourier_volume_from_spec` (`surface_rzfourier.py:698-702`) | OK | `V = Σ (1/3) γ·n / (nphi·ntheta)`. nfp cancels. |
| first/second fundamental form | `surface_curvatures_impl` and `first_fund_form_impl` / `second_fund_form_impl` (`surface.cpp:185-379`) | `surface_rz_fourier_{first,second}_fund_form_from_spec` (`surface_rzfourier.py:590-639`) | OK | JAX recomputes via gammadash and unitnormal; same closed-form contraction. |
| dgamma/dcoeff (Jacobian) | `dgamma_by_dcoeff_impl` (`surfacerzfourier.cpp:914-962`) | `_evaluate_jacobian_from_dofs(...)` wrappers (`surface_rzfourier.py:481-487`, plus `_evaluate_vjp_from_dofs` 490-499) | OK (via autodiff) | The JAX path delegates to `jax.jacfwd(spec→array)`, which iterates over DOFs through the scatter+evaluator. The DOF ordering matches C++ `set_dofs_impl` shifts (`surfacerzfourier.h:60-79`). |
| dgammadash{1,2,1dash1,1dash2,2dash2}/dcoeff | `dgammadash*_by_dcoeff_impl` (`surfacerzfourier.cpp:1130-1551`) | wrappers `surface_rz_fourier_dgammadash*_from_dofs` (`surface_rzfourier.py:803-830`) | OK | Same comment: autodiff over the closed-form spec evaluator. The C++ kernels use explicit per-mode formulas. |
| dgamma/dgammadash*_by_dcoeff_vjp | `dgamma*_by_dcoeff_vjp` (`surfacerzfourier.cpp:758-1127, 1292-1452`) | `_evaluate_vjp_from_dofs(...)` (`surface_rzfourier.py:490-499`) | OK | JAX path runs `jax.vjp(spec→array)` then `pullback(cotangent)`. C++ uses thread-local accumulators with ANGLE_RECOMPUTE; cosmetic reduction-order delta. |
| dofs → coefficients | `set_dofs_impl` (`surfacerzfourier.h:60-79`) | `_coefficients_from_dofs` (`surface_rzfourier.py:340-424`) | OK | JAX uses dense `_scatter_matrix` operators (1.0 at the right positions). The block-mode positions in `_block_mode_positions` exclude `n<0` for the `n=0` row of the cos block and `n<=0` for the sin block — matches C++ start indices `ntor` and `ntor+1`. |
| coefficients → dofs | `get_dofs` (`surfacerzfourier.h:81-101`) | `surface_rz_fourier_dofs_from_spec` (in `specs.py:1474-1492`) | OK | Same `[rc, zs]` order (stellsym) and `[rc, rs, zc, zs]` order (non-stellsym). |

### A2. `SurfaceXYZTensorFourier` (the C++ class) ↔ `src/simsopt/geo/surface_fourier_jax.py`

| Quantity | C++ symbol | JAX function | Status | Notes |
|---|---|---|---|---|
| gamma | `gamma_impl` (`surfacexyztensorfourier.h:127-154`) | `surface_gamma` (`surface_fourier_jax.py:518-568`) | OK | JAX vectorises as `V @ coeffs.T @ W.T` matmul (see §C). The tensor-product basis matches `basis_fun_phi(n)*basis_fun_theta(m)` with cos/sin discriminator at `n=ntor` and `m=mpol` boundaries (header `:1177-1231`). |
| gamma_lin (paired) | `gamma_lin` (`:155-175`) | `surface_gamma_lin` (`surface_fourier_jax.py:571-605`) | OK | Same matmul over the paired-point quadrature. |
| gammadash1 | `gammadash1_impl` (`:424-...`) | `surface_gammadash1` (`surface_fourier_jax.py:806-868`) | OK | `dx/dφ = dx̂/dφ·cosφ − x̂·2π·sinφ − dŷ/dφ·sinφ − ŷ·2π·cosφ` matches `surfacexyztensorfourier.h:197` exactly. Clamped branch dispatches to `_gammadash1_clamped` (lines 710-764). |
| gammadash2 | `gammadash2_impl` | `surface_gammadash2` (`surface_fourier_jax.py:871-909`) | OK | Same as 1 with theta basis derivative. |
| gammadash1dash1 / 1dash2 / 2dash2 | basis_fun_d{phidphi,thetadphi,thetadtheta} | `surface_gammadash{1dash1,1dash2,2dash2}` (`surface_fourier_jax.py:912-1069`) | OK | Implemented via second-derivative basis `_build_phi_basis_with_second` / `_build_theta_basis_with_second`. Clamped branches use `jax.vmap(jax.jacfwd(...))` — see §F. |
| normal | `Surface::normal_impl` (`surface.cpp:382-392`) | `surface_normal` (`surface_fourier_jax.py:1072-1111`) | OK | `cross(gd1, gd2)`. |
| unitnormal | `Surface::unitnormal_impl` (`surface.cpp:451-461`) | `surface_unitnormal_from_dofs` (`surface_fourier_jax.py:2300-2325`) | OK | `n / ‖n‖`. |
| area | `Surface::area()` (`surface.cpp:493-502`) | `surface_area` (`surface_fourier_jax.py:2420-2434`) | OK | `Σ ‖n‖ / (P·T)`. |
| volume | `Surface::volume()` (`surface.cpp:598-610`) | `surface_volume` (`surface_fourier_jax.py:2399-2417`) | OK | `Σ γ·n / (3·P·T)`. |
| dofs → (xc, yc, zc) (stellsym) | `set_dofs_impl` (`:81-101`) iterating `skip(dim,m,n)` (`:1233-1242`) | `stellsym_scatter_indices` + `dofs_to_xyzc` (`surface_fourier_jax.py:1154-1234`) | OK | **The stellsym DOF mapping is bit-exact** — see §E for the full quadrant table. |
| dofs → (xc/xs/yc/ys/zc/zs) (XYZ-Fourier) | `SurfaceXYZFourier::set_dofs_impl` (`surfacexyzfourier.h:72-97`) | `_scatter_surface_xyzfourier_dofs` (`surface_fourier_jax.py:1249-1300`) | OK | Includes the `(m=0, n<=0)` skip for sin blocks. |
| dgamma/dcoeff (Jacobian) | C++ has only XYZ-Fourier `dgamma_by_dcoeff_impl` (`surfacexyzfourier.cpp:483-532`); tensor-Fourier uses the default `Surface::dgamma_by_dcoeff` cache via `dgamma_by_dcoeff_impl` provided in `:585-660`. | `dgamma_by_dcoeff = _dcoeff_jacobian(surface_gamma_from_dofs)` (`surface_fourier_jax.py:2496-2504`) | OK | autodiff via `jax.jacfwd`. |
| dnormal/dcoeff | `Surface::dnormal_by_dcoeff_impl` (`surface.cpp:394-425`) — uses cross-product Leibniz rule | `dnormal_by_dcoeff = _dcoeff_jacobian(surface_normal_from_dofs)` (`surface_fourier_jax.py:2541-2544`) | OK (autodiff) | Algebraically identical (`d(g1×g2) = dg1×g2 + g1×dg2`). |
| d2normal/dcoeffdcoeff | `Surface::d2normal_by_dcoeffdcoeff_impl` (`surface.cpp:427-448`) — exact, second-order Leibniz with `dg1_dc(:,n) × dg2_dc(:,m)` symmetrised | `d2normal_by_dcoeffdcoeff = _dcoeff_hessian(surface_normal_from_dofs)` (`surface_fourier_jax.py:2546-2551`) | OK | autodiff via `jacfwd(jacfwd(...))`. |
| darea/dcoeff | `darea_by_dcoeff_impl` (`surface.cpp:504-537`) via `dnormal_by_dcoeff_vjp` of `n/‖n‖` cotangent | `darea_by_dcoeff = _surface_scalar_grad(surface_area_from_dofs)` (`surface_fourier_jax.py:2609`) | OK | Same gradient via autodiff. |
| dvolume/dcoeff | `dvolume_by_dcoeff_impl` (`surface.cpp:612-649`) via `dnormal_by_dcoeff_vjp + dgamma_by_dcoeff_vjp` of `n/3` and `γ/3` cotangents | `dvolume_by_dcoeff` (`surface_fourier_jax.py:2611`) | OK | Equivalent via autodiff. |
| d2area_by_dcoeffdcoeff | `surface.cpp:566-594` — explicit `(dn,dn,d2n)` Leibniz | `d2area_by_dcoeffdcoeff = _surface_scalar_hessian(...)` (`surface_fourier_jax.py:2610`) | OK | `jax.hessian`. |
| d2volume_by_dcoeffdcoeff | `surface.cpp:651-811` — SIMD-vectorised on-the-fly second derivative | `d2volume_by_dcoeffdcoeff` (`surface_fourier_jax.py:2612`) | OK | `jax.hessian`. |
| BC enforcer (clamped surfaces) | `cache_enforcer = sin(nfp·phi/2)² + sin(theta/2)²` (`surfacexyztensorfourier.h:889-898`); `apply_bc_enforcer = (clamped_dims[dim] && n<=ntor && m<=mpol)` (`:903-905`) | `_bc_enforcer_grid` + `_eval_hat_block` + `_apply_clamped_correction` (`surface_fourier_jax.py:354-499`) | OK | JAX evaluates the unclamped `V·coeffs·Wᵀ` first, then adds `block_hat · (E − 1)` only for the `(m≤mpol, n≤ntor)` cos-cos quadrant of clamped dimensions. Algebraically matches "multiply cos-cos block by `E` and leave the rest unchanged". |

### A3. `SurfaceXYZFourier` (the C++ class) ↔ XYZ-Fourier subset of `surface_fourier_jax.py`

| Quantity | C++ symbol | JAX function | Status | Notes |
|---|---|---|---|---|
| gamma | `gamma_impl` (`surfacexyzfourier.cpp:5-31`) | `surface_xyzfourier_gamma_from_dofs` (`surface_fourier_jax.py:1452-1487`) | OK | Per-mode evaluation, with `(m, n)` runs `[0..mpol] x [-ntor..ntor]`. xhat/yhat via `cos(m*theta−n*nfp*phi)*xc + sin(m*theta−n*nfp*phi)*xs`. |
| gammadash1 | `gammadash1_impl` (`surfacexyzfourier.cpp:328-356`) | `surface_xyzfourier_gammadash1_from_dofs` (`surface_fourier_jax.py:1528-1580`) | OK | Same formula with `2π·(...)` prefactor. |
| gammadash2 | `gammadash2_impl` (`surfacexyzfourier.cpp:358-384`) | `surface_xyzfourier_gammadash2_from_dofs` (`surface_fourier_jax.py:1633-1676`) | OK | |
| gammadash1dash1/1dash2/2dash2 | `gammadash{1dash1,1dash2,2dash2}_impl` (`surfacexyzfourier.cpp:387-479`) | `surface_xyzfourier_gammadash{1dash1,1dash2,2dash2}_from_dofs` (`surface_fourier_jax.py:1725-1881`) | OK | Phase factors computed via `_surface_xyzfourier_mixed_derivative_hat` (lines 1399-1437). The `phase = (phi_order+theta_order) % 4` table matches the explicit C++ `+/−` signs. |
| normal/unitnormal/area/volume | as inherited | `surface_xyzfourier_{normal,unitnormal,area,volume}_from_dofs` | OK | Same wrappers. |
| dgamma_by_dcoeff_impl | C++ `:483-532` | `surface_xyzfourier_dgamma_by_dcoeff` (`surface_fourier_jax.py:2723-2725`) | OK | autodiff. |
| dgammadash1/2_by_dcoeff_impl | C++ `:534-638` | `surface_xyzfourier_dgammadash{1,2}_by_dcoeff` (`:2726-2731`) | OK | |
| dgammadash{1dash1,1dash2,2dash2}_by_dcoeff_impl | C++ `:641-818` | `surface_xyzfourier_dgammadash{1dash1,1dash2,2dash2}_by_dcoeff` (`:2741-2749`) | OK | |

---

## B. Detailed findings

### B1. (MEDIUM, parity-budget) RZ derivative path uses dense reductions, no ANGLE_RECOMPUTE recurrence

**Locations:**
- C++ `gamma_impl` SIMD branch (`surfacerzfourier.cpp:25-76`):
  ```cpp
  for (int m = 0; m <= mpol; ++m) {
      simd_t sinterm, costerm;
      for (int i = 0; i < 2*ntor+1; ++i) {
          int n  = i - ntor;
          if(i % ANGLE_RECOMPUTE == 0)
              xsimd::sincos(m*theta-n*nfp*phi, sinterm, costerm);
          r += rc(m, i) * costerm;
          ...
          if(i % ANGLE_RECOMPUTE != ANGLE_RECOMPUTE - 1){
              simd_t sinterm_old = sinterm;
              simd_t costerm_old = costerm;
              sinterm = cos_nfpphi * sinterm_old + costerm_old * sin_nfpphi;
              costerm = costerm_old * cos_nfpphi - sinterm_old * sin_nfpphi;
          }
      }
  }
  ```
  Reduces in `(m, n)` order with the angle accumulated 5-at-a-time.

- JAX `_mode_terms` + `_sum_fourier_modes` (`surface_rzfourier.py:41-72`):
  ```python
  angles = (
      m[None, None, :, None] * theta[None, :, None, None]
      - nfp * n[None, None, None, :] * phi[:, None, None, None]
  )
  ...
  return jnp.sum(
      cos_coeffs[None, None, :, :] * cos_terms
      + sin_coeffs[None, None, :, :] * sin_terms,
      axis=(2, 3),
  )
  ```
  Reduces over `axis=(2, 3) = (mpol+1, 2*ntor+1)` in XLA-defined order with one `cos`/`sin` call per `(P, T, M, N)` element.

**Impact:** Different summation order and different `sin/cos` accumulation pathway. Under the existing project rule ("byte-identity CPU↔JAX state parity is not portable on the LS path; cross-machine `sdofs_inf` up to `~9.6e-10` in Hessian") this is expected behaviour and the `*_parity` lanes already absorb it through their per-lane tolerances. **However** the `_pre_newton_census_gate_failures` gate in `benchmarks/single_stage_init_parity.py` requires byte-equality against the C++ oracle. The RZ JAX path can satisfy the gate only because the C++ oracle is itself the reference — for any *third* reference (e.g., a from-scratch numpy reduction), expect `sdofs_inf` up to `~1e-12` differences. **Action:** keep this in mind when interpreting `benchmarks/single_stage_init_parity.py` failures; do not treat this as a regression of the JAX path.

### B2. (HIGH/INFO, coverage gap) Missing direct C++ oracle tests for XYZ-Fourier Jacobian column-by-column parity

`tests/geo/test_surface_fourier_jax.py` contains 1593 lines and 38 references to C++ symbols (`grep -c -E "SurfaceXYZTensorFourier|SurfaceRZFourier|sopp\." → 38`). I sampled the file looking for column-by-column basis-sweep parity against `dgammadash1_by_dcoeff_impl` for the **non-clamped XYZ-Fourier** path. The relevant fixtures exercise the tensor-product variant (`SurfaceXYZTensorFourier`) heavily, but the XYZ-Fourier `dgammadash{1,2,1dash1,1dash2,2dash2}_by_dcoeff` path is validated mostly through FD checks and `jax.jacfwd` self-consistency — not through column-by-column dot products against `sopp.SurfaceXYZFourier(...).dgammadash1_by_dcoeff()`.

**Why this matters:** the C++ Jacobian implementations at `surfacexyzfourier.cpp:534-818` have lots of hand-typed `+/−` signs (e.g. `(-n*nfp)*cos(...)` in `dgammadash1dash1_by_dcoeff_impl` lines 688-697 with the `(-n*nfp)*(-n*nfp)` double-negative quirk). An autodiff path is correct by construction *if the forward path is correct*, but a direct C++ oracle would prove that the JAX implementation does not silently drift if the C++ kernel is patched later.

**Recommended action:** add a `TestUpstreamFactorySurfaceXYZFourier::test_dgammadash*_by_dcoeff_column_parity` family that, for `(mpol, ntor) ∈ {(1,1), (2,3)}` and `stellsym ∈ {True, False}`, builds a `sopp.SurfaceXYZFourier`, calls `.dgammadash1_by_dcoeff()`, and contracts with random one-hot cotangents to compare against the JAX path under `parity-ladder` lane `derivative-heavy` (`rtol=1e-8, atol=1e-10`).

### B3. (LOW) Leibniz expansion in `_surface_rz_fourier_derivative_from_terms`

`surface_rzfourier.py:221-273` decomposes `gammadash1dash1` etc. via:
```python
for basis_order in range(phi_order + 1):
    radius_derivative, _ = _radius_height_derivative_from_modes(...)
    scale = float_scalar(comb(phi_order, basis_order), cos_terms)
    if basis_order:
        scale = scale * angle_scale**basis_order
    phase = basis_order % 4
    radial = radial + scale * radial_signs[phase] * radius_derivative
    toroidal = toroidal + scale * toroidal_signs[phase] * radius_derivative
```
This implements `(d/dφ_param)^phi_order [r·cos(2π·φ_param)] = Σ_k C(phi_order,k) · (d/dφ_param)^(phi_order−k)[r] · (d/dφ_param)^k[cos(2π·φ_param)]`. The radial/toroidal sign tables `(1,0,-1,0)` and `(0,1,0,-1)` are exactly the derivatives of cos/sin cycling through `phi % 4`.

It is **correct** but each iteration calls `_radius_height_derivative_from_modes` which itself materialises the same dense `(P,T,M,N)` angle tensor — so for `phi_order=2` (used in `gammadash1dash1`) this is 3 redundant calls to the dense reduction. JAX's tracer will fuse these reductions where possible, so the runtime overhead is moderate, but the code can be reorganised to share one mode-tensor evaluation.

**Action:** *optional* refactor, no correctness change.

### B4. (INFO) Spec wrappers in `src/simsopt/jax_core/surface_fourier.py` add nothing beyond keyword unpacking

`surface_fourier.py:1-286` simply re-exports the kernels in `surface_fourier_jax.py` with spec-based call signatures. The only logic is `_scatter_indices_or_none` (lines 152-155, which returns `None` for non-stellsym) and `_clamped_dims_or_default` (lines 158-164, returning `spec.clamped_dims`). These are thin and correctly delegate.

The `surface_xyz_fourier_normal_from_spec` (lines 113-117) and `surface_xyz_tensor_fourier_normal_from_spec` (lines 257-261) build the normal as `jnp.cross(gammadash1, gammadash2)`, but the corresponding `unitnormal_from_spec` calls the JAX backend's `_surface_xyzfourier_unitnormal_from_dofs` / `_surface_xyz_tensor_unitnormal_from_dofs` directly (lines 120-131 and 264-275). This is harmless — both routes compute the same normal — but be aware that the spec-level normal-then-unitnormal path is not exactly the same function call graph as the spec-level unitnormal path.

---

## C. Function-level deep dive

### (a) gamma parity

**RZ:** C++ formula (`surfacerzfourier.h:11-13`):
```
r(theta, phi) = Σ_{m=0..mpol} Σ_{n=-ntor..ntor} [rc·cos(m·θ − n·nfp·φ) + rs·sin(m·θ − n·nfp·φ)]
z(theta, phi) = same with (zc, zs)
γ = (r·cos(φ), r·sin(φ), z)
```
JAX formula (`surface_rzfourier.py:502-507`):
```python
phi, cos_terms, sin_terms = _mode_angles(spec)    # cos/sin of m·θ − nfp·n·φ
r, z = _radius_height_from_modes(...)              # the inner double sum
cos_phi, sin_phi = _phi_frame(phi)
return jnp.stack([r * cos_phi, r * sin_phi, z], axis=-1)
```
The reduction order over `(m, n)` differs (XLA contracts vs C++ explicit loop with ANGLE_RECOMPUTE recurrence), but the sum is mathematically identical.

**XYZ:** C++ (`surfacexyzfourier.cpp:5-31`):
```
xhat = Σ [xc·cos(mθ−n·nfp·φ) + xs·sin(...)]; same for yhat, z
x = xhat·cos(φ) − yhat·sin(φ); y = xhat·sin(φ) + yhat·cos(φ)
```
JAX (`surface_fourier_jax.py:1452-1487`): identical, with `cos_angle, sin_angle` being broadcast over the `(P, T, mpol+1, 2*ntor+1)` mode grid.

**Tensor-Fourier:** C++ (`surfacexyztensorfourier.h:127-154`): uses `basis_fun(dim, n, phiidx, m, thetaidx) = cache_basis_fun_phi[phiidx, n] * cache_basis_fun_theta[thetaidx, m]` (optionally multiplied by `cache_enforcer` for clamped dims). JAX (`surface_fourier_jax.py:518-568`): `V @ coeffs.T @ Wᵀ` matmul, then a clamped correction. The basis functions themselves match: `basis_fun_phi(n)` is `cos(nfp·n·φ)` for `n≤ntor` and `sin(nfp·(n−ntor)·φ)` for `n>ntor` (`surfacexyztensorfourier.h:1177-1182`), which matches the JAX `build_phi_basis` (`surface_fourier_jax.py:239-273`) where `V = [cos(0·nfp·φ), …, cos(ntor·nfp·φ), sin(1·nfp·φ), …, sin(ntor·nfp·φ)]`.

✓ All three forward kernels are algebraically identical.

### (b) gammadash parity

For all three families the rotation of the `(xhat, yhat)` plane into `(x, y)` uses
`x = xhat·cos(φ) − yhat·sin(φ)`, so the chain-rule expansion for `dx/dφ_param` is
`dxhat/dφ·cosφ − xhat·(2π·sinφ) − dyhat/dφ·sinφ − yhat·(2π·cosφ)`.

Cross-checked:
- JAX `surface_gammadash1` (`surface_fourier_jax.py:836-868`):
  ```python
  dx = dxhat_dphi*cphi - xhat*(2π·sphi) - dyhat_dphi*sphi - yhat*(2π·cphi)
  dy = dxhat_dphi*sphi + xhat*(2π·cphi) + dyhat_dphi*cphi - yhat*(2π·sphi)
  ```
- C++ `SurfaceXYZTensorFourier::gammadash1_lin` (`surfacexyztensorfourier.h:197-200`):
  ```cpp
  data(k1, 0) = 2*M_PI*(dxhatdphi*cos(phi) - xhat*sin(phi) - dyhatdphi*sin(phi) - yhat*cos(phi));
  data(k1, 1) = 2*M_PI*(dxhatdphi*sin(phi) + xhat*cos(phi) + dyhatdphi*cos(phi) - yhat*sin(phi));
  ```

✓ Signs and 2π scaling identical.

### (c) normal parity

`Surface::normal_impl` (`surface.cpp:382-392`) computes the standard cross product `n = γ_φ × γ_θ`. JAX uses `jnp.cross(gd1, gd2)` (`surface_fourier_jax.py:1072-1111` and `surface_rzfourier.py:673-677`).

`jnp.cross([a0, a1, a2], [b0, b1, b2])` returns `[a1·b2 − a2·b1, a2·b0 − a0·b2, a0·b1 − a1·b0]`, identical to the C++ component-wise formula. ✓

`Surface::unitnormal_impl` (`surface.cpp:451-461`) and JAX `_unitnormal` (`surface_fourier_jax.py:1114-1115`) both compute `n / sqrt(n·n)`.

### (d) area / volume parity (nfp cancellation)

**C++ area (`surface.cpp:493-502`):**
```
area = Σ_{i,j} ‖n_ij‖
return area / (numquadpoints_phi · numquadpoints_theta)
```
**C++ volume (`surface.cpp:598-610`):**
```
volume = Σ_{i,j} (1/3) γ_ij · n_ij
return volume / (numquadpoints_phi · numquadpoints_theta)
```

**JAX `surface_area` (`surface_fourier_jax.py:2420-2434`):**
```python
nphi, ntheta = normal.shape[:2]
norm_n = jnp.sqrt(jnp.sum(normal*normal, axis=-1))
return jnp.sum(norm_n) / (nphi * ntheta)
```
**JAX `surface_volume` (`surface_fourier_jax.py:2399-2417`):**
```python
return jnp.sum(integrand) / (3.0 * nphi * ntheta)
```

**nfp cancellation:** The quadrature step on the `[0, 1/nfp) × [0, 1)` domain is `dφ·dθ = (1/(nfp·nphi)) · (1/ntheta)`. The integral is `∫∫_{[0,1/nfp)×[0,1)} ‖n(φ,θ)‖ dφ dθ`. Multiplied by `nfp` (to extend to the full torus `[0, 1)×[0, 1)`), this gives `nfp · Σ ‖n‖ · (1/(nfp·nphi)) · (1/ntheta) = Σ ‖n‖ / (nphi·ntheta)`. So the formula `Σ ‖n‖ / (nphi·ntheta)` is **already** the full-torus area, and nfp does cancel cleanly. ✓

This matches the project rule in `CLAUDE.md`: "Confirmed NOT bugs: **nfp factor in volume/area**: correct — nfp cancels with quadrature step `1/(nfp*nphi)`."

The JAX docstring for `surface_volume` (line 2402-2407) explicitly says "The nfp factor cancels with the quadrature step size." ✓

### (e) DOF ↔ coefficient mapping (especially stellsym) — **this is the highest-risk area per CLAUDE.md**

**C++ tensor-Fourier `skip(int dim, int m, int n)` (`surfacexyztensorfourier.h:1233-1242`):**
```cpp
if (dim == 0)
    return (n <= ntor && m >  mpol) || (n >  ntor && m <= mpol);   // skip cos-sin and sin-cos for x
else if(dim == 1)
    return (n <= ntor && m <= mpol) || (n >  ntor && m >  mpol);   // skip cos-cos and sin-sin for y
else
    return (n <= ntor && m <= mpol) || (n >  ntor && m >  mpol);   // skip cos-cos and sin-sin for z
```

Translation (quadrant table for stellsym DOFs):

| dim (Cartesian) | (m≤mpol, n≤ntor) cos-cos | (m≤mpol, n>ntor) cos-sin | (m>mpol, n≤ntor) sin-cos | (m>mpol, n>ntor) sin-sin |
|---|---|---|---|---|
| x (`dim=0`) | **KEEP** | skip | skip | **KEEP** |
| y (`dim=1`) | skip | **KEEP** | **KEEP** | skip |
| z (`dim=2`) | skip | **KEEP** | **KEEP** | skip |

So x: cos-cos + sin-sin. y, z: cos-sin + sin-cos. **y transforms exactly like z under stellsym.**

**JAX `_is_stellsym_xy` and `_is_stellsym_z` (`surface_fourier_jax.py:1128-1151`):**
```python
def _is_stellsym_xy(m, n, mpol, ntor):
    is_cos_theta = m <= mpol
    is_sin_theta = m > mpol
    is_cos_phi = n <= ntor
    is_sin_phi = n > ntor
    return (is_cos_theta and is_cos_phi) or (is_sin_theta and is_sin_phi)
    # → cos-cos + sin-sin for x

def _is_stellsym_z(m, n, mpol, ntor):
    is_cos_theta = m <= mpol
    is_sin_theta = m > mpol
    is_cos_phi = n <= ntor
    is_sin_phi = n > ntor
    return (is_cos_theta and is_sin_phi) or (is_sin_theta and is_cos_phi)
    # → cos-sin + sin-cos for y and z
```

And `stellsym_scatter_indices` (`surface_fourier_jax.py:1154-1179`) applies:
```python
for coord_offset, allowed_fn in [
    (0, _is_stellsym_xy),              # x: cos-cos + sin-sin
    (n_per_coord, _is_stellsym_z),     # y: cos-sin + sin-cos
    (2 * n_per_coord, _is_stellsym_z), # z: cos-sin + sin-cos
]:
    for m in range(2 * mpol + 1):
        for n in range(2 * ntor + 1):
            if allowed_fn(m, n, mpol, ntor):
                indices.append(coord_offset + m * (2 * ntor + 1) + n)
```

**Verification:** the JAX `_is_stellsym_xy` returns true exactly when the C++ `skip(0, m, n)` returns *false* (i.e. "keep"). Likewise for `_is_stellsym_z` vs `skip(1/2, m, n)`. **The stellsym DOF mapping is bit-exact.** ✓

Quadrant assignments (`m, n` ∈ `[0, 2·mpol], [0, 2·ntor]`):
- `m ≤ mpol ∧ n ≤ ntor` ⇒ cos·cos quadrant.
- `m ≤ mpol ∧ n > ntor` ⇒ cos·sin quadrant.
- `m > mpol ∧ n ≤ ntor` ⇒ sin·cos quadrant.
- `m > mpol ∧ n > ntor` ⇒ sin·sin quadrant.

(For tensor-Fourier the `m` axis runs `[1, cos·θ, ..., cos·(mpol·θ), sin·θ, ..., sin·(mpol·θ)]`, so rows `0..mpol` are cos-θ and rows `mpol+1..2·mpol` are sin-θ — same indexing.)

**XYZ-Fourier (non-tensor):** C++ `SurfaceXYZFourier::set_dofs_impl` (`surfacexyzfourier.h:72-97`):
- stellsym=True: writes `xc, ys, zs` (3 blocks).
- stellsym=False: writes `xc, xs, yc, ys, zc, zs` (6 blocks).

JAX `_scatter_surface_xyzfourier_dofs` (`surface_fourier_jax.py:1249-1300`):
- stellsym=True: writes `xc, ys, zs` (matches) with zeros for `xs, yc, zc`.
- stellsym=False: writes all 6.

The skip rules `(m=0, n<0)` for cos and `(m=0, n<=0)` for sin are honoured by both via the `ntor` and `ntor+1` offsets (`surfacexyzfourier.h:74-95` vs JAX `cos_count = n_per - ntor`, `sin_count = n_per - (ntor + 1)` at `surface_fourier_jax.py:1268-1269`). ✓

**RZ Fourier:** `SurfaceRZFourier::set_dofs_impl` (`surfacerzfourier.h:60-79`):
- stellsym=True: `rc, zs` (skipping `rs, zc` since rc are cos and zs are sin under R↔R, Z↔−Z reflection).
- stellsym=False: `rc, rs, zc, zs`.

JAX `_coefficients_from_dofs` (`surface_rzfourier.py:340-424`):
- stellsym=True: builds `rc` with `include_positions` (n≥0 in m=0 row), `zs` with `exclude_positions` (n>0 in m=0 row), zeros for `rs, zc`.
- stellsym=False: builds all four.

Verified matching offsets: `rc_count = include_positions.size` and `tail_count = exclude_positions.size`, and `total_dofs = rc_count + tail_count` (stellsym) or `2*rc_count + 2*tail_count` (non-stellsym). ✓

### (f) Jacobian (dgamma/dcoeff, dgammadash*/dcoeff, dnormal/dcoeff) parity

C++ exposes explicit per-mode kernels (`dgamma_by_dcoeff_impl`, etc.) that fill a dense `(P, T, 3, ndofs)` tensor. The JAX path constructs the same tensor via `jax.jacfwd` over the `_from_spec` evaluator:

```python
def _evaluate_jacobian_from_dofs(evaluator, spec, dofs):
    dofs_jax = _as_jax_float64(dofs)
    return jax.jacfwd(lambda x: evaluator(_spec_from_dofs(spec, x)))(dofs_jax)
```
(`surface_rzfourier.py:481-487`)

Equivalently for tensor-Fourier:
```python
dgamma_by_dcoeff = _dcoeff_jacobian(surface_gamma_from_dofs)
```
(`surface_fourier_jax.py:2442-2496`)

Both are mathematically equivalent because the surface evaluator is linear in the coefficients (every entry is a sum of `coeff * trig_function(quadrature_point)`). The C++ Jacobian rows are the trig basis functions evaluated at the quadrature points; the JAX `jacfwd` evaluates a vector pushforward through the same closed-form spec, giving the same matrix.

**Floating-point reduction order:** `jacfwd` propagates a `(1,0,…,0), (0,1,…,0), …` standard basis through the evaluator. Each row's reduction sums all coefficient contributions in XLA-defined order — different from the C++ explicit `counter++` per-DOF write. So expect bit-level differences `O(1e-14)` on `(P, T, 3, ndofs)` Jacobians, lane-allowed under `derivative-heavy` (rtol 1e-8).

**`dnormal_by_dcoeff`:** C++ implements `d(g1 × g2) = dg1 × g2 + g1 × dg2` directly (`surface.cpp:394-425`). JAX delegates to `jacfwd(surface_normal_from_dofs)` which expands the same Leibniz rule symbolically. Same algebraic identity. ✓

**`d2normal_by_dcoeffdcoeff`:** C++ implements `d²(g1×g2)/dc_m dc_n = dg1_dc(m) × dg2_dc(n) + dg1_dc(n) × dg2_dc(m)` (`surface.cpp:434-447`). JAX `jacfwd(jacfwd(...))` expands the same. ✓

### (g) Trig accumulation parity

**C++ recurrence (RZ only):** lines 41-62 of `surfacerzfourier.cpp` use
```
sinterm_new = cos_nfpphi * sinterm + costerm * sin_nfpphi
costerm_new = costerm * cos_nfpphi - sinterm * sin_nfpphi
```
where `cos_nfpphi = cos(−nfp·φ)`, `sin_nfpphi = sin(−nfp·φ)`. This is the Chebyshev recurrence to advance the angle by `−nfp·φ`. Every `ANGLE_RECOMPUTE = 5` iterations it resets via `xsimd::sincos(m·θ − n·nfp·φ, …)` to avoid drift.

**The `ANGLE_RECOMPUTE` brace convention is correctly honoured everywhere** (verified via `grep -n "ANGLE_RECOMPUTE"` of `surfacerzfourier.cpp`):
- SIMD branches (`xsimd::sincos(...)`) use braceless `if(...)` because the body is a single statement — correct.
- Non-SIMD branches assign to both `sinterm` and `costerm` and **do** use `{}`.
Lines `100`, `526`, `726`, `873`, `1087`, `1412` (non-SIMD branches) all start with `if(i % ANGLE_RECOMPUTE == 0) {` (opening brace present). The CLAUDE.md note about "explicit `{}` braces" appears already fixed in this checkout — confirmed.

**XYZ-Fourier and tensor-Fourier:** neither use the recurrence. C++ XYZ-Fourier calls `sin/cos(m·θ − n·nfp·φ)` directly (`surfacexyzfourier.cpp:19`). C++ tensor-Fourier uses the per-quadrature-point caches `cache_basis_fun_phi[k1, n]` and `cache_basis_fun_theta[k2, m]` (`surfacexyztensorfourier.h:860-877`) — these are built once at `set_dofs` time and then reused. JAX builds the same `V, W` matrices in `build_phi_basis` and `build_theta_basis` (`surface_fourier_jax.py:172-273`), then matmuls. Reduction-order differences only.

---

## D. Test coverage gaps

Inventoried tests:
- `tests/geo/test_surface_fourier_jax.py` (1593 lines, 38 C++ references) — primary parity battery for tensor-Fourier kernels.
- `tests/geo/test_surface_rzfourier_jax.py` (1533 lines, 30 C++ references) — primary parity battery for RZ kernels.
- `tests/geo/test_surface_xyz_tensor_clamped_jax.py` (252 lines, 10 C++ references) — focused on the BC enforcer / clamped_dims branch.
- `tests/geo/test_surface_fourier_jax_cpu_ordered.py` (347 lines) — the strict CPU-ordered (byte-identical?) variant.
- `tests/geo/test_surface_rzfourier_jax_item06_closeout.py` (220 lines) — item-06 closeout regression suite.

**Coverage gaps observed:**

1. **(INFO) XYZ-Fourier first-derivative-Jacobian column parity not explicitly oracle-tested.** As called out in §B2: there is no `TestUpstreamFactorySurfaceXYZFourier::test_dgammadash1_by_dcoeff_column_complete_cpu_parity` analogue. The path is exercised via FD only.

2. **(INFO) `_surface_rz_fourier_derivative_from_terms` second-order branches** (`gammadash1dash1`, `gammadash1dash2`, `gammadash2dash2`) are validated against C++ `gammadash1dash1_impl` etc. in `test_surface_rzfourier_jax.py` — verified by sampling. Coverage is adequate.

3. **(INFO) `dnormal_by_dcoeff_vjp` (used by `darea_by_dcoeff_impl` and `dvolume_by_dcoeff_impl` in C++)** has no JAX-specific symbol; the JAX path achieves the same effect via `jax.grad(area_from_dofs)` and `jax.grad(volume_from_dofs)`. No oracle test directly compares `dnormal_by_dcoeff_vjp(cotangent)` (C++) against `jacfwd(normal_from_dofs)^T @ cotangent` (JAX). This is a quick-win addition for the `derivative-heavy` lane.

4. **(INFO) `surface_xyzfourier_dnormal_by_dcoeff` / `d2normal_by_dcoeffdcoeff`** for the *non-tensor* XYZ-Fourier surface: no direct C++ Jacobian/Hessian column-by-column parity in the test battery.

5. **(LOW)** The `surface_rz_fourier_dunitnormal_from_dofs` (`surface_rzfourier.py:747-754`) uses `jacfwd` over `unitnormal`. The C++ kernel `Surface::dunitnormal_by_dcoeff_impl` (`surface.cpp:463-490`) is a hand-derived implementation; should be cross-validated for column parity to ensure the chain-rule derivation is identical to autodiff.

---

## E. Recommended actions (ordered by severity)

1. **(MEDIUM) Document the RZ reduction-order limitation in `validation_ladder_contract.py`.** Add a comment block near the `direct-kernel` tolerance entries explaining that RZ kernels target `rtol=1e-12` only on same-state evaluations (the C++ kernel re-evaluated against itself via the JAX wrapper). For from-scratch comparisons, expect `~1e-12` cross-platform variance. This is **documentation**, not a behaviour change.

2. **(HIGH/INFO) Add XYZ-Fourier basis-complete column parity tests.** Create a fixture that builds `sopp.SurfaceXYZFourier(mpol, ntor, nfp, stellsym, quadpoints_phi, quadpoints_theta)`, calls `.dgammadash1_by_dcoeff()` and `.dgammadash2_by_dcoeff()`, and compares column-by-column against `jax.jacfwd(surface_xyzfourier_gammadash{1,2}_from_dofs, argnums=0)` under the `derivative-heavy` tolerance. Same for `.dgammadash1dash1_by_dcoeff()` etc. Target the `(mpol, ntor) ∈ {(1,1), (2,3)}` and both `stellsym` values.

3. **(MEDIUM) Add `dnormal_by_dcoeff_vjp` parity test.** Compare `sopp_surface.dnormal_by_dcoeff_vjp(cotangent)` against `jax.vjp(surface_normal_from_dofs, dofs, ...)[1](cotangent)` to lock down the area/volume gradient pathway against drift if the C++ kernel changes.

4. **(LOW) Optional: refactor `_surface_rz_fourier_derivative_from_terms`** to share the mode-derivative tensor across the Leibniz loop. Zero correctness impact, modest runtime improvement.

5. **(INFO) Optional: add a comment** in `surface_fourier_jax.py:1154-1179` (the `stellsym_scatter_indices` function) pointing to `surfacexyztensorfourier.h:1233-1242` (the C++ `skip` function) as the source-of-truth for the DOF quadrant convention. The current docstring says "matches CPU `SurfaceXYZTensorFourier` where y transforms like z under the stellarator symmetry"; a direct line-link would aid future audits.

---

## Closing notes

- The audit confirmed **no CRITICAL or HIGH correctness defects**. All quadrant signs, angle-convention signs, nfp cancellation, and the highest-risk stellsym DOF scatter mapping match the C++ source exactly.
- The principal finding is the documented reduction-order divergence on the RZ path. This is *already absorbed* by the project's `parity-ladder` lane structure and the `*_parity` vs `*_fast` mode distinction — no urgent action needed.
- The coverage gaps in §D are real but minor; the existing test battery exercises every public surface evaluator through at least one C++ oracle, FD, or closed-form check.
- The CLAUDE.md "ANGLE_RECOMPUTE brace pattern" rule is **honoured** in the current checkout. All non-SIMD branches use explicit braces on the multi-statement `if(i % ANGLE_RECOMPUTE == 0) { sinterm = …; costerm = …; }` block.
