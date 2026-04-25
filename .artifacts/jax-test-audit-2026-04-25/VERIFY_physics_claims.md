# Physics/Math Claim Verification — Revised JAX Test Plan

Date: 2026-04-25
Repo: `/Users/suhjungdae/code/columbia/simsopt-jax`
Branch: `gpu-purity-stage2-20260405`

Sources verified: actual source files in `src/simsopt/`, `src/simsoptpp/`. No reliance on
external SIMSOPT documentation.

---

## Claim 1 — `integral_BdotN` squares/normalizes flux (cannot be reused for raw Gauss-law test)

> "Closed-surface flux test must be raw signed flux. Do not use `integral_BdotN`; that
> objective squares/normalizes flux. The physical invariant is signed
> `sum(B · normal) * dphi * dtheta ≈ 0` on a closed surface not intersecting the current
> wire."

### Public API of `src/simsopt/objectives/integral_bdotn_jax.py`

`__all__ = ["integral_BdotN", "residual_BdotN"]`

#### `residual_BdotN(Bcoil, target, normal, definition)`

Three branches, all squared (the *residual* is `(B·n̂ − target) · √weight`, which the
caller squares):

`integral_bdotn_jax.py:50-78`

```python
BdotN = jnp.sum(Bcoil * unit_n, axis=-1) - target

if definition == "quadratic flux":
    weight   = jnp.where(has_normal, norm_n / (nphi * ntheta), 0.0)
    residual = jnp.where(has_normal, BdotN * jnp.sqrt(weight), 0.0)
elif definition == "normalized":
    B2          = jnp.sum(Bcoil * Bcoil, axis=-1)
    denominator = pairwise_sum_flat(B2 * norm_n)
    point_weight = jnp.where(has_normal, norm_n / safe_denominator, 0.0)
    residual = ... BdotN * jnp.sqrt(point_weight) ...
elif definition == "local":
    B2     = jnp.sum(Bcoil * Bcoil, axis=-1)
    weight = jnp.where(has_normal, norm_n / (safe_B2 * (nphi * ntheta)), 0.0)
    residual = ... BdotN * jnp.sqrt(weight) ...
```

Note the `unit_n = normal / |normal|` step then immediately undone via a `√norm_n` weight.
The integrand reconstructed by the caller is `(BdotN)² · norm_n` — i.e. `(B·n̂)² · |n|`,
i.e. quadratic flux per unit area times area element.

#### `integral_BdotN(...)` (`integral_bdotn_jax.py:85-121`)

```python
residual = residual_BdotN(Bcoil, target, normal, definition=definition)
return 0.5 * scalar_square_sum(residual, ...)
```

So all three branches are quadratic in `B·n̂`:

| definition       | scalar value                                         |
|------------------|-----------------------------------------------------|
| `quadratic flux` | `0.5 · (1/(nphi·ntheta)) · Σ (B·n̂ − target)² · |n|` |
| `normalized`     | `0.5 · Σ (B·n̂ − target)² · |n|  /  Σ |B|² · |n|`    |
| `local`          | `0.5 · (1/(nphi·ntheta)) · Σ (B·n̂ − target)² / |B|² · |n|` |

#### C++ counterpart `src/simsoptpp/integral_BdotN.cpp`

Identical math. `integral_BdotN.cpp:93-103`:

```cpp
if (definition_int == DEFINITION_QUADRATIC_FLUX) {
    numerator_sum += (BcoildotN * BcoildotN) * normN;
} else if (definition_int == DEFINITION_NORMALIZED){
    numerator_sum   += (BcoildotN * BcoildotN) * normN;
    denominator_sum += mod_B_squared * normN;
} else if (definition_int == DEFINITION_LOCAL) {
    if (mod_B_squared > 0.0) {
        numerator_sum += (BcoildotN * BcoildotN) / mod_B_squared * normN;
    }
}
```

with `result = 0.5 * numerator_sum / (nphi * ntheta)` (or normalized variant). Squared in
all three definitions. Module docstring at `integral_bdotn_jax.py:1-22` explicitly calls
these "quadratic-flux-like surface integrals".

### Verdict: CORRECT

None of the three definitions reduces to the raw signed flux `∮ B·n dA`. The integrand is
always `(B·n̂)² · |n|` (quadratic) or that ratio normalized by `Σ|B|²|n|`. There is no way
to extract signed `B·n̂ · |n|` from this kernel without a sign cancellation that defeats
the purpose. Even the residual vector pre-square is `(B·n̂ − target) · √weight` — the
square-root weight makes it useless as a raw flux source as well.

### Implication

A Gauss-law / closed-surface verification test cannot reuse `integral_bdotn_jax`. The
plan must add a new helper, e.g.

```python
def signed_flux_jax(B, normal):
    # raw signed ∮ B·n dA, no square, no normalization
    nphi, ntheta = B.shape[:2]
    return jnp.sum(B * normal) / (nphi * ntheta)
```

and use it as the closed-surface invariant.

---

## Claim 2 — Boozer residual formula is `G·B − |B|²·(x_phi + iota·x_theta)`

### JAX source (`src/simsopt/geo/boozer_residual_jax.py:110-117`)

```python
def _boozer_weighted_residual(G, iota, B, xphi, xtheta, weight_inv_modB):
    tang = xphi + iota * xtheta
    B2 = pairwise_sum_axis(B * B, axis=-1)
    residual = G * B - B2[..., None] * tang

    if weight_inv_modB:
        residual = _safe_inverse_modB(B2)[..., None] * residual
    return residual
```

Module docstring at `boozer_residual_jax.py:18-26`:

> The residual at each grid point is
> `r̃_{ij} = w_{ij} · [G · B_{ij} − |B_{ij}|² · (x_φ + ι · x_θ)]`
> with `w = 1/|B|` when *weight_inv_modB* is True, else `w = 1`.

### CPU Python reference (`src/simsopt/geo/surfaceobjectives.py:537-605`)

Function docstring (line 545):

> `G·B_BS(x) − ‖B_BS(x)‖² · (x_φ + ι·x_θ)`

Code (lines 596-598):

```python
tang = xphi + iota * xtheta
B2   = np.sum(B**2, axis=2)
residual = G * B - B2[..., None] * tang
```

(Optional `weight_inv_modB` multiplies by `1/|B|` afterward.)

A second-order Python reference at `surfaceobjectives.py:1527-1562`
(`boozer_surface_residual_dB`) repeats the same formula in its docstring:
`d/dB[ G*B_BS(x) - ||B_BS(x)||^2 * (x_phi + iota * x_theta) ]`.

### C++ counterpart (`src/simsoptpp/boozerresidual_impl.h:60-66`)

```cpp
double tang_ij0 = xphi(i,j,0) + iota * xtheta(i,j,0);
double tang_ij1 = xphi(i,j,1) + iota * xtheta(i,j,1);
double tang_ij2 = xphi(i,j,2) + iota * xtheta(i,j,2);

double resij0 = G * B(i,j,0) - B2ij * tang_ij0;
double resij1 = G * B(i,j,1) - B2ij * tang_ij1;
double resij2 = G * B(i,j,2) - B2ij * tang_ij2;
```

Identical sign convention, identical ordering, identical optional `wij = 1/|B|` weight.

### Verdict: CORRECT

The residual is exactly `G·B − |B|²·(x_φ + ι·x_θ)` in the JAX kernel, the Python
reference, and the C++ kernel. No sign, ordering, or normalization variance.

### Implication

The plan can take the residual formula as established. Tests that assert the residual
expression should match this exact form (modulo the optional `1/|B|` weight, which is a
test-time choice).

---

## Claim 3 — `Volume.J()` returns enclosed volume

### CPU `Volume.J()` (`src/simsopt/geo/surfaceobjectives.py:275-330`)

```python
class Volume(Optimizable):
    """Wrapper class for volume label."""

    def J(self):
        """Compute the volume enclosed by the surface."""
        return self.surface.volume()
```

`surface.volume()` is implemented in C++ at `src/simsoptpp/surface.cpp:597-610`:

```cpp
template<class Array>
double Surface<Array>::volume() {
    double volume = 0.;
    auto n = this->normal();
    auto xyz = this->gamma();
    for (int i = 0; i < numquadpoints_phi; ++i) {
        for (int j = 0; j < numquadpoints_theta; ++j) {
            volume += (1./3) * (xyz(i,j,0)*n(i,j,0) + xyz(i,j,1)*n(i,j,1) + xyz(i,j,2)*n(i,j,2));
        }
    }
    return volume / (numquadpoints_phi * numquadpoints_theta);
}
```

This is exactly `V = (1/3) ∮ r · n dA` with `n = γ_φ × γ_θ` (unnormalized) and the
quadrature step folded into the trailing division. Divergence theorem on `∇·r = 3` gives
`∫_V 3 dV = ∮ r·n dA`, hence `V = (1/3) ∮ r·n dA`. Standard enclosed volume.

### JAX counterpart `volume_jax` (`src/simsopt/geo/label_constraints_jax.py:14-15`)

```python
from .surface_fourier_jax import surface_volume as volume_jax
```

`surface_volume` lives in `src/simsopt/geo/surface_fourier_jax.py:784-802`:

```python
def surface_volume(gamma, normal):
    """Compute the volume enclosed by a toroidal surface.

    Uses the divergence theorem:
    ``V = (1/3) ∫∫ γ · n dφ dθ``
    where ``n = gammadash1 × gammadash2`` is the unnormalized normal.

    The ``nfp`` factor cancels with the quadrature step size.
    """
    nphi, ntheta = gamma.shape[:2]
    integrand = jnp.sum(gamma * normal, axis=-1)  # (nphi, ntheta)
    return jnp.sum(integrand) / _as_jax_float64(3.0 * nphi * ntheta)
```

Same expression as the C++ version, modulo the `nfp` cancellation note documented in the
"Key Conventions" section of `CLAUDE.md`.

### Verdict: CORRECT

Both the CPU `Volume.J()` (via `surface.volume()` in C++) and the JAX `volume_jax`
compute the unnormalized enclosed volume of the closed surface via the divergence
theorem `V = (1/3) ∮ r·n dA`. No surface-area variant, no normalization factor.

### Implication

The plan can rely on `Volume.J()` (CPU) or `volume_jax(gamma, normal)` (JAX) as the
ground-truth enclosed-volume scalar with no additional rescaling. Either is appropriate
for an enclosed-volume invariance / parity assertion.

---

## Claim 4 — Existing closed-surface signed-flux helper?

Searched `src/simsopt/` for any helper computing raw signed `∮ B·n dA` over a closed
toroidal surface without squaring or normalizing.

### Candidates examined

| symbol | location | what it actually computes |
|---|---|---|
| `integral_BdotN`, `residual_BdotN` | `objectives/integral_bdotn_jax.py` | quadratic / normalized / local (all squared, see Claim 1) |
| `IntegralBdotN` (C++ via `simsoptpp`) | `simsoptpp/integral_BdotN.cpp` | same three squared definitions |
| `SquaredFlux`, `SquaredFluxJAX` | `objectives/fluxobjective.py`, `objectives/fluxobjective_jax.py` | wrap `integral_BdotN` (squared) |
| `fixed_surface_flux_integral_from_B` | `jax_core/objectives_flux.py:62-69` | calls `integral_BdotN_jax` (squared) |
| `fixed_surface_flux_residual_from_B` | `jax_core/objectives_flux.py:72-79` | calls `residual_BdotN_jax` (already weighted by `√|n|`, not raw `B·n·|n|`) |
| `ToroidalFlux.J()` | `geo/surfaceobjectives.py:333+` | flux through a `phi=const` ribbon via `∮ A·t dl`, NOT a closed surface |
| `toroidal_flux_jax` | `geo/label_constraints_jax.py:25-46` | same — `Σ A·γ_θ / nθ`, flux at one phi slice |
| `_net_fluxes_pure`, `net_ext_fluxes_pure`, `_net_ext_flux_eval`, `NetFluxes` | `field/force.py:949,1318,1411,1457` | flux through a coil loop via line integral `∮ A·dℓ`, NOT `∮ B·n dA` over a closed surface |
| `coilset.flux_penalty` | `field/coilset.py:230-240` | wraps `SquaredFlux` (squared) |
| `MinToroidalFluxStoppingCriterion`, `MaxToroidalFluxStoppingCriterion` | `field/tracing.py:751,770` | tracer stop condition, not a flux integral |
| `pflux_profile`, `tflux_profile` | `mhd/spec.py:640,678` | profile getters/setters in SPEC equilibrium wrapper |

No helper computes raw signed `∮ B·n dA` over a closed toroidal surface.

### Verdict: WRONG that one exists / CORRECT that one is needed

There is no existing primitive for the Gauss-law invariant `∮_S B·n dA = 0` on a closed
toroidal surface that does not enclose the current wire. Every existing flux helper
either squares the integrand, normalizes it, or computes a fundamentally different
flux (line integral around a coil loop, or flux through a `phi=const` cap).

### Implication

The plan needs a new helper. Minimal form:

```python
# src/simsopt/objectives/integral_bdotn_jax.py  (or a new module)
def signed_flux_jax(B, normal):
    """Raw signed flux ∮ B·n dA over a closed surface (Gauss-law test).

    B:      (nphi, ntheta, 3)  field on the closed surface
    normal: (nphi, ntheta, 3)  unnormalized normal = γ_φ × γ_θ

    Returns scalar (B*normal already includes the area element via |n|).
    """
    nphi, ntheta = B.shape[:2]
    return jnp.sum(B * normal) / (nphi * ntheta)
```

(Sign is determined by the orientation of `normal`. Convention: outward normal makes the
sum positive when the wire is enclosed and zero when it is not.)

---

## Summary

| Claim | Verdict |
|---|---|
| 1. `integral_BdotN` squares/normalizes — needs new raw-flux helper | CORRECT |
| 2. Boozer residual is `G·B − |B|²·(x_φ + ι·x_θ)` | CORRECT |
| 3. `Volume.J()` returns enclosed volume `(1/3) ∮ r·n dA` | CORRECT |
| 4. Existing signed-flux helper exists | WRONG (none exists; new helper required) |

All four claims in the revised JAX test plan are physically and numerically grounded.
The corrected plan should add a small `signed_flux_jax(B, normal)` primitive and use it
for the Gauss-law closed-surface check; existing modules already provide everything else.
