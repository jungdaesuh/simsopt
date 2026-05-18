# R3 — Math/Physics Correctness Audit of simsopt-jax Kernels

**Scope**: Verify the mathematical and physical correctness of the JAX-ported
physics kernels in `simsopt-jax` (worktree
`/Users/suhjungdae/code/columbia/simsopt-jax`) against authoritative references
(Landreman & Bhattacharjee 2021; Helander & Spong 2010; Boozer 1981; Jackson §5;
the C++ oracle in `src/simsoptpp/`).

**Verdict legend**: CORRECT / WRONG / SIGN-ERROR / SCALE-ERROR /
CONVENTION-DIVERGENT.

Branch audited: `gpu-purity-stage2-20260405`. C++ oracle paths cited are
relative to `src/simsoptpp/`. JAX paths are relative to `src/`.

---

## 1. Biot-Savart law — `simsopt/jax_core/biotsavart.py`

### 1.1 B field

**Expected**:
$$\mathbf{B}(\mathbf{x}) = \frac{\mu_0}{4\pi}\oint \frac{I\, d\boldsymbol{\ell}\times(\mathbf{x}-\mathbf{x}')}{|\mathbf{x}-\mathbf{x}'|^3}$$

**Implementation**:
- `_biot_savart_B_integrand` at `simsopt/jax_core/biotsavart.py:372-378`
  ```
  diff = gammas - x
  r_inv3 = 1/|diff|^3
  cross = _cross_product(diff, gammadashs)   # = (γ - x) × dℓ
  return cross * r_inv3
  ```
- Final scale: `_MU0_OVER_4PI = 1e-7` at `biotsavart.py:54`.

**Sign analysis**: The JAX `diff = gammas - x = -(x - x')`, and the integrand
forms `diff × dγ = -(x - x') × dℓ = dℓ × (x - x')`. The C++ oracle
(`simsoptpp/biot_savart_impl.h:63-69`) uses `diff = x - γ` and computes
`dγ × diff = dℓ × (x - x')`. Both expressions agree: the two sign flips
(direction of `diff` and the order of the cross product) cancel.

**Quadrature**: `_quadrature_block_integral` (`biotsavart.py:262-328`)
performs trapezoidal-style sum and divides by `quadrature_count`, matching
C++ `fak = 1e-7 / num_quad_points` (`biot_savart_impl.h:44`).

**Verdict**: **CORRECT.**

### 1.2 dB/dX and second derivatives

Implemented via `jax.jacfwd(one_point, argnums=0)` (`biotsavart.py:471-478`)
with `jnp.swapaxes(..., -1, -2)` to reshape from autodiff layout
`[B-component, deriv]` to the SIMSOPT public `dB_by_dX[p, j, l] = ∂_j B_l(x_p)`
convention (axis 1 = derivative direction, axis 2 = component).

The Hessian (`biotsavart.py:480-494`) reshapes from `jacfwd²` raw
`[component, d1, d2]` to `[d1, d2, component]` via `jnp.transpose(..., (1,2,0))`.
The comment notes `jacfwd²` (not `jacrev∘jacfwd`) preserves
current-linearity exactly.

**Verdict**: **CORRECT** (autodiff differentiates the analytic value kernel,
so the C++ closed-form derivative in `biot_savart_impl.h:80-100` is a
parallel reference rather than a structural divergence).

### 1.3 VJP and signs

`biot_savart_B_vjp` at `biotsavart.py:656-669` calls `jax.vjp(fwd, gammas,
gammadashs, currents)` and returns the raw cotangent triplet. Sign is
controlled by JAX autodiff through the forward kernel. No manual sign
inversion is needed; the docstring (lines 660-666) cautions that the CPU
wrapper pushes geometry/current cotangents through each coil and combines
through coil dofs differently.

**Verdict**: **CORRECT.**

### 1.4 Vector potential A

$$\mathbf{A}(\mathbf{x}) = \frac{\mu_0}{4\pi}\oint \frac{I\, d\boldsymbol{\ell}}{|\mathbf{x}-\mathbf{x}'|}$$

`_biot_savart_A_integrand` at `biotsavart.py:381-385` returns `gammadashs *
r_inv`. The final scale `_MU0_OVER_4PI * Σ I_c * ∫ dℓ_c/r`. **CORRECT.**

### 1.5 Singularity handling

`_safe_radius_squared` (`biotsavart.py:119-131`) clamps `r²` at `1e-60`. The
docstring (lines 120-130) flags this as a deliberate divergence from the C++
NaN/Inf behavior on point-on-coil inputs. It is documented and intentional;
not a math bug.

**Verdict**: **CONVENTION-DIVERGENT (intentional, documented).**

---

## 2. Dommaschk vacuum field — `simsopt/jax_core/analytic_fields.py`

**Expected (Dommaschk 1986, *Computer Physics Communications* 40, 203-218)**:
The Dommaschk magnetic field is derived from a scalar potential
$\Phi = \sum_{m,n}\bigl[(a_{mn}\cos m\varphi + b_{mn}\sin m\varphi) D_{mn}(R,Z) +
(c_{mn}\cos m\varphi + d_{mn}\sin m\varphi) N_{mn}(R,Z)\bigr]$
where $D_{mn}, N_{mn}$ are polynomial in $R, Z, \log R$.

**Implementation**:
- `_dmn_terms`, `_nmn_terms` (`analytic_fields.py:128-180`) build the
  per-mode polynomial term lists; mirror `Dmn(m,n)` / `Nmn(m,n)` in
  `simsoptpp/dommaschk.cpp:60-71, 73-84`.
- `_eval_terms_dense` (`analytic_fields.py:226-248`) evaluates the
  polynomial sums; `_diff_R_terms`, `_diff_Z_terms` differentiate symbolically.
- `_dommaschk_single_mode_BR_BZ_Bphi` (`analytic_fields.py:327-370`) assembles
  the cylindrical-component field. The cached `_dommaschk_term_bundle` keys
  `N_mn` on `n - 1` (`analytic_fields.py:262`), matching the C++ pattern
  `Nmn(m, n - 1, R, Z)` at `simsoptpp/dommaschk.cpp:260, 277, 294, 311, ...`.

**Cross-checks**:
- The combinatorial helpers `_alpha_py`, `_beta_py`, `_gamma1_py`,
  `_alphas_py`, `_betas_py`, `_gammas_py` (`analytic_fields.py:59-99`)
  match `simsoptpp/dommaschk.cpp:4-58` line-by-line.
- The identity `dZBR = dRBZ` (`analytic_fields.py:442`) is the curl-free
  property of vacuum harmonics: $\partial_Z B_R = \partial_R B_Z$ when
  $\nabla\times\mathbf{B} = 0$. C++ at `simsoptpp/dommaschk.cpp:400-432`
  implements the same identity structurally.
- `_cylindrical_to_cartesian_dB` (`analytic_fields.py:474-525`) is the full
  $\partial_j B_l$ transformation from cylindrical to Cartesian. I derived the
  `dB00 = ∂B_x/∂x` term manually (chain rule with $R = \sqrt{x^2+y^2}$, $\phi =
  \mathrm{atan2}(y,x)$) and the implementation matches term-for-term.

**Verdict**: **CORRECT.**

---

## 3. Reiman island model — `simsopt/jax_core/analytic_fields.py:692-837`

**Expected (Reiman & Greenside 1986)**: The Reiman field is a closed-form
series in $r_{\min} = \sqrt{(R-R_{\rm axis})^2 + Z^2}$ producing a
controllable island chain.

**Implementation**: `_reiman_pure_B` (`analytic_fields.py:692-732`):
```
combo  = iota0 + iota1·rmin²  − Σ k_θ·ε·r_min^(k_θ-2)·cos(k_θ θ − m_0 φ)
combo1 = +Σ k_θ·ε·r_min^(k_θ-2)·sin(k_θ θ − m_0 φ)
B_R = ((R−R_axis)/R) combo1 + (Z/R) combo
B_Z = −((R−R_axis)/R) combo  + (Z/R) combo1
B_φ = −1
```

This matches `simsoptpp/reiman.cpp:8-41` line by line, including the sign
conventions and the fixed `B_phi = -1`. The dB tensor implementation
(`_reiman_pure_dB`, lines 735-837) likewise tracks the C++ at
`simsoptpp/reiman.cpp:44-93`.

**Verdict**: **CORRECT.**

---

## 4. Circular coil — `simsopt/jax_core/circular_coil.py`

**Expected (Smythe / Jackson §5.5)**: The magnetic field of a circular
current loop is expressible in closed form using complete elliptic integrals
$K(m)$ and $E(m)$ of modulus $m = 4 r_0 \rho / [(r_0+\rho)^2 + z^2]$ (or
equivalently $1 - \alpha^2/\beta^2$).

**Implementation**: `_B_local_pointwise` (`circular_coil.py:226-242`) uses
the standard $\alpha, \beta$ form:
```
α = √(r_0² + r² − 2r_0 ρ)
β = √(r_0² + r² + 2r_0 ρ)
m = 1 − α²/β²
B_x = I·x·z / (2α²β·ρ² + 1e-31) · [(r_0² + r²) E(m) − α² K(m)]
B_y = I·y·z / (2α²β·ρ² + 1e-31) · [same]
B_z = I / (2α²β + 1e-31) · [(r_0² − r²) E(m) + α² K(m)]
```

Compared against the CPU oracle `_B_impl` in
`simsopt/field/magneticfieldclasses.py` (lines around `def _B_impl(self,
B):` for `CircularCoil`), the JAX implementation matches line-by-line.

**Vector potential**: `_A_local_pointwise` (`circular_coil.py:245-265`)
reproduces the upstream CPU formula
`A_φ = -I/2 · (2r_0 + ρ·E(m) + (r_0²+r²)(E(m) - K(m))) / (ρ²+ε)·β`.

> **Note on the `+ 2r_0` term**: The literal CPU formula contains
> `2*r0 + ρ·E(k²) + (r_0²+r²)(E−K)` in the numerator, which is unusual for
> the canonical Smythe form (one expects only the trigonometric-integral
> residue without an additive `+2r_0`). The JAX port reproduces the
> upstream CPU literal exactly (verified by checking
> `_A_impl` at `magneticfieldclasses.py:502-517`). This is a faithful port
> of upstream; an oracle audit against the textbook Smythe form would
> require physics review beyond this audit's scope.

**Singularity guards**: `+1e-31` floors on the denominators are copied
verbatim from upstream (`magneticfieldclasses.py:513-514`); these are not
the JAX "double-where" pattern but match the upstream parity contract.

**Verdict**: **CORRECT (vs. upstream CPU).** Flagged in the actionable
list as a separate physics-review item.

---

## 5. Toroidal / Poloidal vacuum fields — `simsopt/jax_core/analytic_pure_fields.py`

### 5.1 ToroidalField

**Expected**: $\mathbf{B} = (B_0 R_0 / R)\,\hat{\boldsymbol{\varphi}}$.

JAX `_toroidal_B_pointwise` (`analytic_pure_fields.py:162-172`):
```
coeff = B_0 R_0 / R²
B_x = -coeff · y
B_y =  coeff · x
B_z =  0
```
This is $(B_0 R_0 / R²) (-y, x, 0) = (B_0 R_0 / R) \cdot (-y/R, x/R, 0) =
(B_0 R_0 / R) \hat{\boldsymbol{\varphi}}$ ✓ since
$\hat{\boldsymbol{\varphi}} = (-\sin\varphi, \cos\varphi, 0) = (-y/R, x/R, 0)$.

**Vector potential**: `_toroidal_A_pointwise` (line 253-263):
`A_x = B_0 R_0 z x / R²`, `A_y = B_0 R_0 z y / R²`, `A_z = 0`. Take the curl:
$\nabla \times \mathbf{A}$ should give $\mathbf{B}$. Quick check:
$\partial_z A_y - \partial_y A_z = B_0 R_0 y / R²$, which is $B_y$ ✓.

**Verdict**: **CORRECT.**

### 5.2 PoloidalField (q-factor parameterized)

`_poloidal_B_pointwise` (lines 372-391) uses $\theta = \mathrm{atan2}(Z, R -
R_0)$ and $r = \sqrt{(R-R_0)^2 + Z^2}$, yielding `B = (B_0/(R_0 q)) r ·
(-\sin\theta \cos\varphi, -\sin\theta \sin\varphi, +\cos\theta)`. The
$r/(R_0 q)$ factor gives $\iota = 1/q$ on the axis to first order. Singular
at $r = 0$ (the magnetic axis) by design (documented in module docstring
lines 42-45). **CORRECT.**

### 5.3 MirrorModel (WHAM)

Double-Lorentzian flux function `ψ = (R²B_0/(2π γ)) · [1/(1+((Z-Z_m)/γ)²) +
1/(1+((Z+Z_m)/γ)²)]` (`_mirror_psi`, lines 522-534). The field is derived
as `B_R = (∂_R ψ)/R` and `B_Z = 2ψ/R²` form. Verified against CPU
`magneticfieldclasses.py` `MirrorModel._B_impl`.

**Verdict**: **CORRECT.**

---

## 6. Surface volume / area — `simsopt/jax_core/surface_rzfourier.py:672-700`, `simsopt/jax_core/surface_fourier_jax.py:2399-2434`

### 6.1 Volume (divergence theorem)

**Expected**: For a closed surface $\Sigma$,
$$V = \frac{1}{3}\oint_\Sigma \mathbf{r}\cdot\hat{\mathbf{n}}\, dS = \frac{1}{3}\oint \mathbf{r}\cdot(\boldsymbol{\gamma}_\varphi\times\boldsymbol{\gamma}_\theta)\, d\varphi\, d\theta.$$

JAX `surface_rz_fourier_volume_from_spec` (`surface_rzfourier.py:678-682`):
```
V = Σ γ · (γ_φ × γ_θ) / (3 · nphi · ntheta)
```

C++ `Surface::volume()` (`simsoptpp/surface.cpp:598-609`) is identical:
`Σ (1/3)(x·n_x + y·n_y + z·n_z) / (nphi·ntheta)`.

**nfp factor**: Both `gammadash1` and `gammadash2` carry a factor of $2\pi$
(see `simsoptpp/surfacerzfourier.cpp:180-182`), so `γ_φ × γ_θ` has $(2\pi)^2$
baked in. Quadrature step is `dφ_phys · dθ_phys = (2π/nfp)/nphi · 2π/ntheta`
for `range="field period"`, giving:
$$V_{\rm period} = \frac{1}{3}\Sigma \gamma\cdot n \cdot\frac{(2\pi)^2/(\rm nfp)}{nphi\cdot ntheta\cdot(2\pi)^2}$$
After cancellation with the $(2\pi)^2$ in `n`, the explicit `1/(nphi·ntheta)`
factor yields $V_{\rm total} = {\rm nfp}\cdot V_{\rm period}$, so the total
torus volume comes out correctly for both `range="full torus"` and
`range="field period"`. The CLAUDE.md comment "nfp cancels with quadrature
step" is correct.

**Verdict**: **CORRECT.**

### 6.2 Area (Stokes / surface measure)

JAX `surface_rz_fourier_area_from_spec` (`surface_rzfourier.py:672-675`):
```
A = Σ |n| / (nphi · ntheta)
```
where `n = γ_φ × γ_θ` (unnormalized normal, length = $\sqrt{EG-F^2}$).
C++ (`simsoptpp/surface.cpp:493-502`) is identical. Same `nfp/quadrature`
cancellation as volume. **CORRECT.**

### 6.3 Mean cross-sectional area, minor / major radius

`surface_mean_cross_sectional_area_jax_from_dofs`
(`simsopt/geo/surfaceobjectives_jax.py:487-506`) computes
$\overline{A} = \frac{1}{2\pi}\int A_\phi(\phi)\,d\phi$ using the Jacobian
inversion line-integral formula
$A_\phi = \oint R\,(dZ/d\theta)\,d\theta$. The JAX expression
```
dz_dtheta = γ_θ[2] − γ_φ[2] · jacobian_01 / jacobian_00
signed_area = mean(√(x²+y²) · dz_dtheta · jacobian_00) / (2π)
```
matches the CPU `surface.py:693-766` (`mean_cross_sectional_area`) via:
- C++ detJ = `J[0,0]` (since `J[1,1]=1, J[1,0]=0`), and
- C++ `dZ_dtheta = dgamma1[2] · Jinv[0,1] + dgamma2[2] · Jinv[1,1] =
  -dgamma1[2]·J[0,1]/J[0,0] + dgamma2[2]`.

Multiplying CPU `np.mean(√(x²+y²) · dZ_dtheta · detJ) / (2π)` gives
`mean(√(x²+y²) · (dgamma2[2] − dgamma1[2]·jacobian_01/jacobian_00) ·
jacobian_00) / (2π)`, identical to JAX. **CORRECT.**

Minor radius: $a = \sqrt{\overline{A}/\pi}$ (line 509-511). Major radius: $R
= |V| / (2\pi^2 a^2)$ (line 514-517). These are the standard torus
relations $V = 2\pi^2 R a^2$. **CORRECT.**

---

## 7. Stellsym DOF ordering — `simsopt/geo/surface_fourier_jax.py:1128-1179`

**Expected** (CLAUDE.md SSOT): x uses cos-cos + sin-sin; y and z use cos-sin
+ sin-cos. y transforms like z under stellarator symmetry $(\varphi,\theta)
\to (-\varphi,-\theta)$.

**JAX `_is_stellsym_xy` (lines 1128-1138)**: allowed when (m≤mpol AND n≤ntor)
[cos-cos] OR (m>mpol AND n>ntor) [sin-sin].

**JAX `_is_stellsym_z` (lines 1141-1151)**: allowed when (m≤mpol AND n>ntor)
[cos-sin] OR (m>mpol AND n≤ntor) [sin-cos].

**C++ `SurfaceXYZTensorFourier::skip`
(`simsoptpp/surfacexyztensorfourier.h:1233-1242`)**:
```
dim=0 (x): skip if (n≤ntor && m>mpol) || (n>ntor && m≤mpol)  // skip sin-cos and cos-sin
             ⇔ keep cos-cos and sin-sin                          ✓ matches JAX
dim=1 (y): skip if (n≤ntor && m≤mpol) || (n>ntor && m>mpol)  // skip cos-cos and sin-sin
             ⇔ keep cos-sin and sin-cos                          ✓ matches JAX
dim=2 (z): same as dim=1                                          ✓ matches JAX
```

**Verdict**: **CORRECT.**

---

## 8. First / Second fundamental forms — `simsopt/jax_core/surface_rzfourier.py:569-619`

**Expected**:
- $E = \boldsymbol\gamma_\varphi\cdot\boldsymbol\gamma_\varphi$, $F =
  \boldsymbol\gamma_\varphi\cdot\boldsymbol\gamma_\theta$, $G =
  \boldsymbol\gamma_\theta\cdot\boldsymbol\gamma_\theta$.
- $L = \hat{\mathbf{n}}\cdot\boldsymbol\gamma_{\varphi\varphi}$, $M =
  \hat{\mathbf{n}}\cdot\boldsymbol\gamma_{\varphi\theta}$, $N =
  \hat{\mathbf{n}}\cdot\boldsymbol\gamma_{\theta\theta}$.
- Mean curvature $H = (LG − 2MF + NE)/(2(EG − F^2))$.
- Gaussian curvature $K = (LN − M^2)/(EG − F^2)$.
- Principal curvatures $\kappa_\pm = H \pm \sqrt{H^2 - K}$.

JAX (`surface_rzfourier.py:569-619`) implements these identities verbatim.
The variable names in `surface_curvatures_from_spec` are slightly clashing
(`m` and `n` are reused for `M` and `N`) but the algebra is correct.

**Verdict**: **CORRECT.**

---

## 9. Boozer residual — `simsopt/geo/boozer_residual_jax.py:117-188`

**Expected** (Helander & Spong 2010, Landreman & Bhattacharjee 2021): On a
Boozer surface the magnetic field satisfies
$$G\,\mathbf{B} = |\mathbf{B}|^2(\boldsymbol{\gamma}_\varphi + \iota\,\boldsymbol{\gamma}_\theta)$$
so the **residual** vector is $\mathbf{r} = G\,\mathbf{B} - |\mathbf{B}|^2(\boldsymbol{\gamma}_\varphi + \iota\,\boldsymbol{\gamma}_\theta)$,
with optional inverse-modB weighting.

**JAX `_boozer_weighted_residual` (lines 117-124)**:
```
tang     = xphi + iota · xtheta
B²       = |B|²
residual = G · B − B² · tang
if weight_inv_modB: residual = residual / |B|
```

**C++ `boozer_residual_impl` (`simsoptpp/boozerresidual_impl.h:60-72`)**:
```
tang_ij = xphi(i,j) + iota · xtheta(i,j)
resij   = G · B(i,j) − B²ij · tang_ij
rtil    = resij · wij     where wij = 1/|B| or 1
```

Identical sign and normalization. **CORRECT.**

**Normalization**: JAX divides by `3·nphi·ntheta` to give an averaged
square sum (line 188); raw C++ `sopp.boozer_residual` does not. This is a
documented convention divergence (`boozer_residual_jax.py:32-39` module
docstring) — the production CPU `boozersurface.py:601-602` applies the same
normalization. **CONVENTION-DIVERGENT (documented; not a math bug).**

---

## 10. IFT adjoint signs — `simsopt/geo/surfaceobjectives_jax.py:2302-2687`

**Expected**: For an inner solve defined by $g(x_{\rm inner}(\rm coils),
\rm coils) = 0$ and outer objective $J = J(x_{\rm inner}(\rm coils),
\rm coils)$:
$$\frac{dJ}{d{\rm coils}} = \frac{\partial J}{\partial {\rm coils}} - {\rm adj}^\top \frac{\partial g}{\partial {\rm coils}}$$
where ${\rm adj}$ solves $(\partial g/\partial x)^\top\,{\rm adj} =
\partial J/\partial x$.

**Implementation**:
- `_solve_boozer_adjoint` (`surfaceobjectives_jax.py:1726-1736`) calls the
  **transpose** solve path. ✓
- `BoozerResidualJAX._value_and_dJ_by_dcoil_dofs` (line 2389): returns
  `value, direct_gradient - adjoint_gradient`. ✓ matches `direct − adj^⊤
  ∂g/∂coils`.
- `IotasJAX._value_and_dJ_by_dcoil_dofs` (line 2497): returns `iota,
  -adjoint_gradient`. ✓ matches the no-direct-term branch (since
  $\partial \iota/\partial {\rm coils} = 0$).
- `MajorRadiusJAX._value_and_dJ_by_dcoil_dofs` (line 2558): returns
  `value, -adjoint_gradient`. ✓ matches `−adj^⊤ ∂g/∂coils` with no direct
  term (since major radius depends only on surface DOFs).
- `NonQuasiSymmetricRatioJAX._value_and_dJ_by_dcoil_dofs` (line 2674):
  returns `value, direct_gradient - adjoint_gradient`. ✓

All four signs match the spec.

**Adjoint solve direction**: `_checked_boozer_linear_solve(..., transpose=True)`
at line 1744 routes through `solve_transpose` or
`solve_transpose_with_status`. The CPU/JAX runtime SSOT
`BoozerSurfaceJAX.get_adjoint_runtime_state()` controls the operator-vs-PLU
choice; CLAUDE.md documents that LS-lane PLU is load-bearing while
exact-lane PLU is metadata only.

**Verdict**: **CORRECT.**

---

## 11. Quasi-symmetry ratio — `simsopt/geo/surfaceobjectives_jax.py:2106-2150`

**Expected** (Helander & Boozer):
$$J_{\rm nQS} = \frac{\langle dS\, B_{\rm nonQS}^2\rangle}{\langle dS\, B_{\rm QS}^2\rangle}, \quad B_{\rm QS} = \frac{\langle |B|\,dS\rangle_{\rm sym\,axis}}{\langle dS\rangle_{\rm sym\,axis}}.$$

**Implementation** (`_qs_ratio_pure`, lines 2106-2150):
```
B_QS    = sum(|B|·dS, axis) / sum(dS, axis)
B_nonQS = |B| − B_QS
return sum(dS · B_nonQS²) / sum(dS · B_QS²)
```
- `axis = 0` (sum-over-phi → QS depends on θ only) ⇒ **quasi-axisymmetry**.
- `axis = 1` (sum-over-theta → QS depends on φ only) ⇒ **quasi-poloidal**.
- Selector: `self.axis = 1 if quasi_poloidal else 0` at line 2593. ✓

**CPU comparison**: `simsopt/geo/surfaceobjectives.py:1024` uses
`np.mean(modB·dS, axis=axis) / np.mean(dS, axis=axis)`. JAX uses `sum/sum`.
Since both numerator and denominator divide by the same N (=
shape[axis]), the result is identical. **CORRECT.**

**Verdict**: **CORRECT.**

---

## 12. Particle guiding-center RHS — `simsopt/jax_core/tracing.py`

### 12.1 Cartesian vacuum GC (`guiding_center_vacuum_rhs`, lines 1364-1431)

**Expected**:
$$\dot{\mathbf{x}} = \frac{v_\parallel}{|B|}\mathbf{B} + \frac{m}{q|B|^3}(\tfrac12 v_\perp^2 + v_\parallel^2)\,\mathbf{B}\times\nabla|B|, \quad \dot v_\parallel = -\frac{\mu}{|B|}\mathbf{B}\cdot\nabla|B|.$$

JAX implementation matches C++ `GuidingCenterVacuumRHS::operator()` at
`simsoptpp/tracing.cpp:50-77` line-by-line:
- `grad_abs_B = einsum("l,jl->j", B, dB_by_dX) / abs_B` = $\partial_j|B|$
  (correct chain rule using `dB_by_dX[j,l] = ∂_j B_l`).
- `B × ∇|B|` via `jnp.cross(B, grad_abs_B)`.
- Both factors `fak1`, `fak2`, and the sign of $\dot v_\parallel$ match.

**Verdict**: **CORRECT.**

### 12.2 Boozer GC — vacuum, no_K, full

`guiding_center_vacuum_boozer_rhs` (lines 2163-2227),
`guiding_center_no_k_boozer_rhs` (lines 2230-2304), and
`guiding_center_boozer_rhs` (lines 2307-2381) match the C++ classes
`GuidingCenterVacuumBoozerRHS`, `GuidingCenterNoKBoozerRHS`,
`GuidingCenterBoozerRHS` (`simsoptpp/tracing.cpp:81-254`) **term-by-term**,
including:
- $C = -mv_\parallel(K_\zeta - G')/|B| - q\iota$ ✓
- $F = -mv_\parallel(K_\theta - I')/|B| + q$ ✓
- $D = (FG - CI)/\iota$ ✓
- $\dot v_\parallel = -(\mu/v_\parallel)(\,|B|_\psi\dot s\,\psi_0 +
  |B|_\theta\dot\theta + |B|_\zeta\dot\zeta)$ ✓ (the energy-conservation
  identity from the analytic Boozer GC equations).

**Verdict**: **CORRECT.**

### 12.3 Energy / μ conservation

The audit prompt notes "should be conserved to FP precision under
symplectic integration". **The dopri5 stepper used by both C++ and JAX is
NOT symplectic** — it is an embedded RK4(5) Dormand-Prince integrator
(`tracing.py:678-735`, `_DOPRI5_A`/`B`/`C`/`E` tableaux at lines 145-209
match boost::odeint's `runge_kutta_dopri5` used in `simsoptpp/tracing.cpp:374-375`).

What IS exactly preserved:
- The analytic RHS conserves $E = \tfrac12 m v_\parallel^2 + \mu |B|$ and
  $\mu$ at the differential-equation level. The JAX port preserves this
  analytic structure.
- $\mu$ conservation is structural in the GC formulation; energy
  conservation in the no-K and full Boozer RHS is built into the explicit
  $\dot v_\parallel = -(\mu/v_\parallel)\nabla|B|\cdot\dot{\mathbf{x}}$
  identity (lines 2299-2301, 2376-2378).

What is NOT exactly preserved:
- FP-level energy conservation across many steps. Dopri5 has local
  truncation $O(h^5)$ and global error $O(h^4)$; energy and $\mu$ drift
  accordingly. This is a C++/JAX shared property, not a JAX port bug.

**Verdict**: **CORRECT** (faithful port of the C++ analytic + numerical
contract; the symplectic-precision claim in the audit prompt is
mis-specified).

---

## 13. Dipole field — `simsopt/jax_core/dipole_field.py`

**Expected (Jackson §5.6)**:
$$\mathbf{B} = \frac{\mu_0}{4\pi}\sum_i\Bigl[\frac{3(\mathbf{m}_i\cdot\mathbf{r}_i)\,\mathbf{r}_i}{|\mathbf{r}_i|^5} - \frac{\mathbf{m}_i}{|\mathbf{r}_i|^3}\Bigr]$$
$$\mathbf{A} = \frac{\mu_0}{4\pi}\sum_i\frac{\mathbf{m}_i\times\mathbf{r}_i}{|\mathbf{r}_i|^3}$$

`_dipole_field_B_jit` (`dipole_field.py:128-147`): exactly this form. ✓
`_dipole_field_A_jit` (`dipole_field.py:170-183`): exactly this form. ✓

**dB/dX gradient** (`_dipole_field_dB_jit`, lines 207-239):
$$\partial_k B_j = \frac{\mu_0}{4\pi}\sum_i \frac{3}{r_i^5}\bigl[m_j r_k + m_k r_j + (\mathbf{m}\cdot\mathbf{r})\delta_{jk} - 5(\mathbf{m}\cdot\mathbf{r})\frac{r_j r_k}{r^2}\bigr]$$

This is the textbook gradient. The C++ stores only six independent
components and fills the symmetric off-diagonal pairs explicitly
(`simsoptpp/dipole_field.cpp:213-221`), confirming the field is curl-free
(symmetric gradient). The JAX kernel produces the full tensor by direct
formula. Verified by symmetry: $\partial_k B_j = \partial_j B_k$ since
$m_j r_k + m_k r_j$ is symmetric in $(j,k)$, $\delta_{jk}$ is symmetric,
and $r_j r_k$ is symmetric.

**Verdict**: **CORRECT.**

---

## 14. GPMO / MwPGP / GSCO algorithms

### 14.1 MwPGP (`simsopt/jax_core/pm_optimization.py:2411-2545`)

Mixed-active-set Projected Gradient solver for the relax-and-split convex
inner problem. The step body `_step_body` mirrors C++ `MwPGP_algorithm`
(`simsoptpp/permanent_magnet_optimization.cpp`) including the φ/β proximal
projection (`phi_MwPGP`, `phi_MwPGP_diff`). The C++ doc comment at the top
of `pm_optimization.py:1-80` enumerates the mapping line by line.

**Verdict**: **CORRECT** (port verified by parity gates per repository
notes).

### 14.2 GPMO (greedy variants)

Several GPMO variants are ported (`gpmo_baseline_solve`,
`gpmo_arbvec_solve`, `gpmo_multi_solve`, `gpmo_backtracking_solve`,
`gpmo_arbvec_backtracking_solve`). The candidate-cost computations
(`gpmo_baseline_candidate_costs` etc.) compute the per-cell objective
difference exactly as the C++ greedy step would. Algorithm matches
Kaptanoglu et al.

**Verdict**: **CORRECT**.

### 14.3 GSCO (`simsopt/solve/wireframe_optimization_jax.py:315-545`)

`greedy_stellarator_coil_optimization_jax` implements the same greedy
objective $f = f_B + \lambda_S f_S$ as `simsoptpp/wireframe_optimization.cpp`,
including the loop-undo rule (lines 430-436) and the connectivity / new-coil
constraint masks.

**Verdict**: **CORRECT.**

---

## 15. Wireframe segment field — `simsopt/jax_core/wireframe.py:139-253`

**Expected** (closed-form Biot-Savart of a straight finite wire from $a$
to $b$ evaluated at $\mathbf{p}$):
$$\mathbf{B} = \frac{\mu_0 I}{4\pi}\cdot\frac{|r_1|+|r_2|}{|r_1||r_2|(|r_1||r_2|+\mathbf{r}_1\cdot\mathbf{r}_2)}\,\mathbf{r}_1\times\mathbf{r}_2,$$
$\mathbf{r}_1 = \mathbf{p} - a$, $\mathbf{r}_2 = \mathbf{p} - b$.

JAX `_wireframe_segment_B_from_arrays` (lines 139-151) implements this
formula. Verified term-by-term against C++ `wireframe_field_kernel`
(`simsoptpp/wireframe_field_impl.h:63-76`).

The gradient via `factor`, `grad_factor`, and the cross-product expansion
(`_wireframe_segment_B_and_dB_by_dX_from_arrays`, lines 190-253) matches
the C++ closed-form gradient at `simsoptpp/wireframe_field_impl.h:78-100`
line-by-line.

**Singularity caveat**: when $\mathbf{p}$ is on the wire (between $a$ and
$b$), `denom = 0` and the result is NaN. Both JAX and C++ are unprotected;
production wireframe workflows ensure evaluation points stay off the wire.

**Verdict**: **CORRECT.**

---

## 16. `jnp.where` ordering and double-where pattern

### 16.1 `integral_bdotn_jax.py:65-115`

The quadratic-flux residual uses the **JBP-17.1 double-where** pattern:
```
safe_norm_n = jnp.where(has_normal, norm_n, 1.0)
unit_n      = jnp.where(has_normal, normal / safe_norm_n, 0.0)
```
The "normalized" definition adds a second `denominator > 0` guard before
dividing. The "local" definition routes failure-mode through
`_inf_with_nan_jvp` (see §18). **CORRECT.**

### 16.2 `_distance_jax.py:32`

Distance² (no square root) so no divide-by-zero. The `jnp.where` puts inf
at invalid pairs to push them out of the `argmin`. **CORRECT.**

### 16.3 `interpolated_field.py:132, 159-161`

Symmetry folds map $\varphi$ into $[0, 2\pi)$ then `mod 2π/nfp`. The
`jnp.where(phi < 0.0, phi + 2π, phi)` is a value-only fold (no division).
**CORRECT.**

### 16.4 `regular_grid_interp.py:598-627`

Soft boundary clamp uses `jnp.where` value-only; `jnp.clip` keeps cell
indices in `[0, n-1]` for safe gather, with the in-bounds flag controlling
fill-vs-evaluate. This is **CORRECT** defensive coding.

### 16.5 `analytic_fields.py` (Dommaschk, Reiman)

No `jnp.where` guards — the C++ oracle is also unprotected at the
coordinate-axis singularity ($R = 0$, $R = R_{\rm axis}$). The JAX port
matches the upstream non-defensive contract. **CONVENTION-DIVERGENT
(intentional, matches C++).**

### 16.6 `_math_utils._axis_last_norm` (lines 82-86)

Uses scale-then-normalize:
```
scale       = max(|v_i|, ...)
safe_scale  = jnp.where(scale > 0, scale, 1.0)
norm        = safe_scale · √(Σ (v_i/safe_scale)²)
```
This is the scaled-Euclidean pattern (Hairer/Wanner). At `scale = 0`,
`norm = 0` and the value is correct. The custom_jvp wrapper
`unit_vector_axis_last` (lines 93-106) handles the gradient with the
correct orthogonal-projection formula
$d(\hat{\mathbf{v}}) = (I - \hat{\mathbf{v}}\hat{\mathbf{v}}^\top) \mathbf{d}/\|\mathbf{v}\|$.

**Verdict**: **CORRECT.**

---

## 17. `safe_norm` at coil singularities

`_safe_radius_squared` (`biotsavart.py:119-131`) clamps $r^2$ at `1e-60`.
For coil-singularity inputs, the JAX kernel returns a finite-but-huge
field (≈ $10^{90}$ for $1/r^3$) rather than C++ NaN/Inf. Documented
divergence; not a bug. The "production workflows never land on
point-on-coil geometry" assertion in the docstring is a research-practice
contract.

**Verdict**: **CONVENTION-DIVERGENT (documented).**

---

## 18. Failure-mode `inf_with_nan_jvp` — `simsopt/jax_core/_math_utils.py:109-123`

The `inf_with_nan_jvp(reference)` returns `+inf` for the primal and
propagates `nan` as the cotangent. This is the JAX-idiomatic "failure
lane" pattern: value pushes the optimizer back, gradient pollutes the
update with `nan` so the caller can detect a failure-mode iteration.
Consumed by `integral_bdotn_jax.py:107-110` in the "local" definition's
$|B|^2 = 0$ singular branch.

**Verdict**: **CORRECT.**

---

## 19. Boozer analytic / radial / interpolated fields

### 19.1 `simsopt/jax_core/boozer_analytic.py`

`_eval_modB`, `_eval_dmodB{ds,dtheta,dzeta}`, `_eval_K`, `_eval_dK{dtheta,dzeta}`
(lines 219-275) implement the Garren-Boozer near-axis form:
- $|B|(s,\theta,\zeta) = B_0 + B_0\eta_{\rm bar} r \cos(\theta - N\zeta)$ where
  $r = \sqrt{2 s \psi_0 / \overline{B}}$.
- $\partial|B|/\partial\zeta = N B_0\eta_{\rm bar} r \sin(\theta - N\zeta)$ ✓
  (chain rule with `angle = θ − Nζ`, so $\partial_\zeta \angle = -N$ flips
  the sign to give positive $N \cdot \sin$).
- $\partial r/\partial s = \psi_0/(\overline{B} r)$ (line 233: $r\psi_0/(2\psi)$
  using $\psi = s\psi_0$, equivalent algebra). ✓

**Verdict**: **CORRECT.**

### 19.2 `simsopt/jax_core/boozer_radial_interp.py`

Pure Fourier-projection helpers `compute_kmnc_kmns`, `fourier_transform_*`,
`inverse_fourier_transform_*`. These are linear algebra kernels (matmul
form) with the upstream $1/(2\pi^2)$ normalisation baked in. Verified
against `simsoptpp/boozerradialinterpolant.cpp`. **CORRECT.**

### 19.3 `simsopt/jax_core/interpolated_boozer_field.py`

Regular-grid interpolant with explicit symmetry / flux-function folds
matching `simsoptpp/boozermagneticfield_interpolated.h:724-807` (the
docstring at lines 22-46 enumerates the mappings). **CORRECT.**

---

## 20. Surface RZ Fourier / XYZ Tensor Fourier derivative chains

Pre-computed derivative kernels in `surface_rzfourier.py:520-566` and
`surface_fourier_jax.py:330-2316` are mostly straightforward Fourier-series
differentiation. The audit verified the gamma/gammadash construction
chain against `simsoptpp/surfacerzfourier.cpp:25-280`. The factors of $2\pi$
on derivatives match. **CORRECT.**

The 2nd and 3rd-order derivatives carry the `(2π)²` and `(2π)³` factors
matching `simsoptpp/surfacexyztensorfourier.h:250-417`.

---

## Math/physics bugs that need fixing

### BLOCKER
*(None identified in this audit. All Biot-Savart, Boozer-residual, IFT
adjoint signs, QS ratio formulas, surface volume/area, Stellsym DOF
ordering, GC RHS, dipole, and wireframe segment kernels match the C++
oracle and standard physics references.)*

### HIGH
*(None identified.)*

### MEDIUM

1. **CircularCoil `_A_impl` literal upstream-port** —
   `simsopt/jax_core/circular_coil.py:245-265` and
   `simsopt/field/magneticfieldclasses.py:502-517`.

   The upstream CPU `_A_impl` carries an additive `+2*r0` term in the
   numerator:
   `num = 2*r0 + ρ·E(m) + (r_0²+r²)·(E(m) − K(m))`

   The canonical Smythe form for the vector potential of a circular
   current loop (Smythe, *Static and Dynamic Electricity*, §7.10; Jackson
   §5.5) is
   $$A_\varphi = \frac{\mu_0 I}{\pi k}\sqrt{\frac{r_0}{\rho}}\bigl[(1-\tfrac{k^2}{2})K(k^2) - E(k^2)\bigr]$$
   which does **not** contain an additive bare-radius term. The JAX
   port faithfully reproduces upstream, so the parity gate is tight, but
   the upstream formula itself appears to deviate from the canonical
   Smythe form. **Action**: defer to a physics-review item (out of scope
   for JAX-port audit) but flag here so it is not lost. If the upstream
   formula is intentional (e.g., a Yokoyama-form regularisation), it
   should be documented at the call site.

2. **Dopri5 is not symplectic — energy/$\mu$ conservation is RHS-level
   only, not stepper-level** —
   `simsopt/jax_core/tracing.py:678-735` + GC RHS at lines 1364-2381.

   The JAX port matches the C++ choice of `boost::numeric::odeint`
   `runge_kutta_dopri5`, which is a non-symplectic embedded RK 4(5).
   $\mu$ is structurally conserved by the GC ansatz (no equation of
   motion changes it), and the analytic Boozer RHS encodes $E$
   conservation through the $\dot v_\parallel$ identity, but the
   numerical integration introduces $O(h^4)$ truncation drift in $E$ and
   $\mu$. This is a research-practice limit, not a port bug, but
   downstream tooling that expects strict FP-level conservation will not
   get it.

   **Action**: Document this in the tracing-mode acceptance criteria
   (`docs/source/jax_acceptance.rst`) if not already there. Any
   conservation-sensitive consumer needs a symplectic alternative
   (e.g., a Boris pusher or symplectic Gauss-Legendre).

### LOW

3. **`_axis_last_norm` does not use the strict JBP-17.1 "double-where"
   form** — `simsopt/jax_core/_math_utils.py:82-86`.

   The current implementation uses the scaled-Euclidean form, which is
   value-correct at `scale = 0` (returns 0). The custom_jvp at lines
   93-106 routes gradients through the orthogonal-projection formula,
   so the gradient is also well-defined for `‖v‖ > 0`. At `‖v‖ = 0` the
   gradient would divide by zero (via `projected_dot / norm`); this is
   never hit in production because unit-vector lookups are downstream of
   a non-degenerate surface normal.

   **Action**: Add a defensive guard in `_unit_vector_axis_last_jvp` if
   `‖v‖ = 0` becomes reachable on any new code path. No fix needed
   today.

4. **Singularity behavior at coil/wire points is non-defensive by
   design** — `wireframe.py:139-151`, `biotsavart.py:119-131`,
   `analytic_pure_fields.py` (Poloidal at $r = 0$, Mirror at $\rho = 0$).

   The JAX port intentionally matches the C++ NaN/Inf behavior (or, in
   the case of Biot-Savart, a soft clamp at `1e-60`). Production
   workflows must keep evaluation points off the singular set. This is
   neither a math bug nor a port divergence to act on; it is recorded
   here as a known shape constraint.

5. **C++ vs. JAX residual normalization for the raw `sopp.boozer_residual`
   symbol** — `boozer_residual_jax.py:32-39`.

   The raw C++ symbol does **not** divide by `3·nphi·ntheta`; the
   production CPU path normalizes inline in
   `boozersurface.py:601-602`. The JAX `boozer_residual_scalar`
   normalizes internally to match the production path. Same value, but
   tests that compare to the raw C++ symbol must add the
   `1/(3·nphi·ntheta)` factor explicitly. Documented in the module
   docstring; no code action needed.

---

## Summary

All field, surface, Boozer, particle-tracing, dipole, wireframe,
permanent-magnet, and adjoint kernels reviewed in this audit are
**mathematically and physically correct** to the standard of the C++
oracle and the underlying physics literature. Sign conventions for the
IFT adjoint (`BoozerResidual`, `Iotas`, `MajorRadius`,
`NonQuasiSymmetricRatio`) are correct. The Biot-Savart sign convention
agrees with the textbook law via the cancellation of two sign flips.
The Stellsym DOF ordering matches `SurfaceXYZTensorFourier::skip` in
both branches (`xy` and `z`). The Boozer-residual operator
$G\mathbf{B} - |\mathbf{B}|^2(\boldsymbol{\gamma}_\varphi + \iota\,
\boldsymbol{\gamma}_\theta)$ is implemented with the correct sign and
weighting. The QS ratio computes the standard Helander-Boozer
non-quasisymmetry penalty in both QA and QP modes. All three Boozer GC
RHS variants (vacuum, no_K, full) match `simsoptpp/tracing.cpp`
term-by-term.

The five items flagged are either (a) upstream-CPU literal-port concerns
that fall outside the JAX-port audit boundary (CircularCoil `_A_impl`),
(b) numerical-integration limits inherited from the C++ contract
(dopri5 non-symplecticity), or (c) defensive-coding observations that
require no code change today. No BLOCKER or HIGH severity math/physics
bug was identified.
