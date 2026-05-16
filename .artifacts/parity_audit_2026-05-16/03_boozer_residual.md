# Parity Audit — Boozer Residual & Derivatives

**Audit timestamp:** 2026-05-16
**Branch:** `gpu-purity-stage2-20260405`
**Scope:** Boozer residual objective JAX implementation vs. C++ reference

## Files audited

| Path | Lines |
|------|------:|
| `src/simsopt/geo/boozer_residual_jax.py` | 802 |
| `src/simsoptpp/boozerresidual_impl.h` | 558 |
| `src/simsoptpp/boozerresidual_py.cpp` | 32 |
| `src/simsoptpp/boozerresidual_py.h` | 9 |
| `src/simsoptpp/python.cpp` (lines 100-138, `boozer_dresidual_dc`) | — |
| `src/simsopt/geo/surfaceobjectives.py` lines 540-760 (CPU oracle `boozer_surface_residual`) | — |
| `src/simsopt/geo/boozersurface_jax.py` (consumers) | — |
| `tests/geo/test_boozer_residual_jax.py` | 574 |
| `tests/geo/test_boozer_derivatives_jax.py` | 994 |

## Executive summary

1. **HIGH (semantic divergence between the JAX scalar and the C++
   `sopp.boozer_residual`):** the C++ public symbol
   (`boozerresidual_impl.h:74`) returns `Σ ½ r²` — *no* `num_res`
   division — whereas `boozer_residual_scalar`
   (`boozer_residual_jax.py:158`) returns `Σ ½ r² / (3·nphi·ntheta)`.
   The parity tests at `tests/geo/test_boozer_residual_jax.py:99-101`
   re-scale the C++ value before comparing, but every other JAX
   downstream (LS penalty in `boozersurface_jax.py:1623`,
   `surfaceobjectives_jax.py:2208`) carries the JAX normalization. CPU
   consumers (`boozersurface.py:788, 802`) read the C++ unnormalized
   scalar back into `BoozerSurface.res["residual"]` and into the
   SciPy LBFGS-B/Newton callbacks. **JAX `J` and CPU `J` are NOT the
   same scalar by a factor of `1/(3·nphi·ntheta)`**; any cross-runtime
   reuse of the penalty value (e.g. weighting in single-stage outer
   objectives, or runtime gates that compare against absolute
   thresholds) must be aware of this rescale.

2. **HIGH (composed pipeline uses `jax.jacfwd` even though
   `n_res ≪ n_dofs`):** `boozer_residual_jacobian_composed`
   (`boozer_residual_jax.py:738-739`) calls
   `jax.jacfwd(_boozer_residual_vector_composed)` to materialize the
   `(n_res, n_dofs)` Jacobian. For a typical BoozerExact problem
   `n_res = 3·nphi·ntheta = 3·8·8 = 192` and `n_dofs ≈ (2mpol+1)(2ntor+1)·6 + 2`
   often `≪ 192`, so forward-mode is the correct choice here. *But*
   the function evaluates the residual twice (`r =
   _boozer_residual_vector_composed(x, **kwargs)` on line 738, then
   the value-producing tape inside `jacfwd` on line 739); the same
   `(gamma, B)` is computed twice. Switching to
   `jax.jvp`-on-`vmap` or to a fused `value_and_jacfwd` pattern
   would halve the surface evaluation work without changing math.
   This is a performance issue, not a correctness issue.

3. **MEDIUM (composed pipeline silently overrides user-provided
   `G` when `optimize_G=False`):** `_unpack_decision_vector`
   (`boozer_residual_jax.py:545-556`) ignores the `G` slot of the
   decision vector when `optimize_G=False` and instead computes
   `G = μ₀·Σ|I_k|` from the coil currents. The CPU oracle
   `surfaceobjectives.boozer_surface_residual` (lines 577-584) uses
   the *same* formula when `G is None`, so the math is consistent — but
   the JAX path never accepts a user-supplied `G` together with
   `optimize_G=False`. The C++ kernel always takes an explicit `G`
   (the CPU `boozer_surface_residual` only auto-fills when the
   caller passed `G=None`). A caller who reasonably expects to pin
   `G` and exclude it from the decision vector has no JAX route to
   do so; the docstrings do not flag this.

---

## Function-by-function parity matrix

| JAX callable (`boozer_residual_jax.py`) | C++/CPU counterpart | Lane / oracle status |
|----------------------------------------|----------------------|----------------------|
| `boozer_residual_scalar` (L117-158, default reduction) | `sopp.boozer_residual` (`boozerresidual_py.cpp:4-9`, `boozerresidual_impl.h:13/346, deriv=0`) | `direct_kernel` lane, anchored at `test_boozer_residual_jax.py:170` (after `1/num_res` rescale); near-floor parity at L382 |
| `boozer_residual_scalar(reduction_mode="cpu_ordered")` (L149-150, fori_loop) | Same C++ kernel; mirrors the outer `i,j,3-component` accumulation order verbatim | Tested at L201-231 against a numpy reference. NOT tested against the C++ binary, only against a Python re-implementation — see Test gap T1 |
| `boozer_residual_scalar(reduction_mode="strict_oracle")` | None (compensated summation reserved for oracle investigations) | Tested at L289-326 against `math.fsum`; no C++ counterpart |
| `boozer_residual_grad` (L201-243) | `sopp.boozer_residual_ds` (`boozerresidual_py.cpp:11-20`, `boozerresidual_impl.h:13/346, deriv=1`) WHEN `nsurfdofs == 0`; **diverges** when `nsurfdofs > 0` (JAX returns zeros, CPU returns nonzero) | FD-only validation against iota/G at L342-376; surface-DOF entries explicitly checked to be zero at L551-570. Scope-limited by design; documented |
| `boozer_residual_hessian` (L246-285) | `sopp.boozer_residual_ds2` (`boozerresidual_py.cpp:22-31`, `boozerresidual_impl.h:13/346, deriv=2`) WHEN `nsurfdofs == 0`; otherwise the surface-DOF blocks are zero | FD-symmetry + FD-of-gradient at L498-527; **NO C++ Hessian parity check in this file**; the composed Hessian is covered in `test_boozer_derivatives_jax.py::TestBoozerHessianComposed` against FD only |
| `boozer_residual_vector` (L293-317) | None (C++ exposes only the scalar through `sopp.boozer_residual`; the vector form is constructed inline by `boozer_surface_residual` in `surfaceobjectives.py:592-611`) | No public C++ symbol oracle; floor-regime parity check at `test_boozer_residual_jax.py:382` uses `0.5·Σ r²/r.size` → C++ scalar as the boundary oracle |
| `boozer_residual_scalar_and_grad_cpu_ordered` (L320-446) | `sopp.boozer_residual_ds(..., deriv=1)` summed point-by-point — manual C++ chain rule including the `1/|B|` Jacobian terms | Tested only indirectly through `boozersurface_jax.py` LS path bit-identity gates; the function itself is **not** directly tested against `sopp.boozer_residual_ds` in this file (see Test gap T2) |
| `boozer_penalty_composed` (L599-657) | None (high-level composition that the C++ side performs by chaining `surface.gamma()` → `BiotSavart` → `sopp.boozer_residual` in the SciPy callback at `boozersurface.py:1697-1705`) | FD-only validation in `test_boozer_derivatives_jax.py::TestBoozerPenaltyGradComposed` |
| `boozer_penalty_grad_composed` (L660-673) | None directly; functional equivalent of the C++ chain at `surfaceobjectives.py:615-661` for derivatives=1 | FD-only via `check_grads` and JVP-vs-grad in `test_boozer_derivatives_jax.py:280-295` |
| `boozer_residual_jacobian_composed` (L717-740) | `surfaceobjectives.boozer_surface_residual(..., derivatives=1)` returning `(r, J)`, with chain `sopp.boozer_dresidual_dc` + `dgamma_by_dcoeff` + `dB_by_dX` | `derivative-heavy` lane FD validation in `test_boozer_derivatives_jax.py::TestBoozerResidualJacobianComposed`. **No direct CPU-derivative oracle parity test** for this pair |
| `boozer_residual_coil_vjp` (L743-802) | CPU chain `boozer_surface_residual_dB` → `B_vjp` → `sopp.biot_savart_vjp_graph` (referenced in docstring L765-766) | Scalarization-vs-`jax.grad` consistency in `test_boozer_derivatives_jax.py::TestBoozerResidualCoilVJP`. No direct C++ VJP oracle, only JAX-internal consistency |
| `_split_decision_vector`, `_inverse_modB`, `_pack`, `_unpack`, `_boozer_weighted_residual`, `_boozer_objective_from_packed`, `_cpu_ordered_boozer_square_sum`, `_get_surface_fns`, `_get_surface_xyzfourier_fns`, `_get_grouped_biot_savart`, `_surface_geometry_from_dofs`, `_unpack_decision_vector`, `_composed_pipeline`, `_boozer_residual_vector_composed` | Private helpers — no public C++ counterparts | Covered transitively |

---

## Detailed findings

### F1 — HIGH — Scalar normalization divergence between JAX and CPU public APIs

The CPU public scalar `sopp.boozer_residual` (`boozerresidual_py.cpp:4-9`)
delegates to `boozer_residual_impl<Array, 0>`. Inside the kernel
(`boozerresidual_impl.h:74` and the non-SIMD fallback at L372):

```cpp
res += 0.5*(rtil_ij0*rtil_ij0 + rtil_ij1*rtil_ij1 + rtil_ij2*rtil_ij2);
```

There is no `/num_res` normalization. The C++ scalar is

    J_cpp = Σ_ij ½ ‖r_ij‖²

The JAX scalar (`boozer_residual_jax.py:146-158`):

```python
num_res = _as_runtime_float64(3 * nphi * ntheta, reference=B)
rtil = _boozer_weighted_residual(...)
...
return _as_runtime_float64(0.5, reference=rtil) * square_sum / num_res
```

normalizes by `num_res = 3·nphi·ntheta`. So

    J_jax = J_cpp / (3·nphi·ntheta)

The parity test (`tests/geo/test_boozer_residual_jax.py:92-101`)
correctly applies the conversion:

```python
val_raw = _call_boozer_residual(float(G), float(iota), xphi_host, xtheta_host, B_host, ...)
num_res = 3 * B_host.shape[0] * B_host.shape[1]
return float(val_raw) / num_res
```

This is mathematically consistent inside the JAX side: every JAX
consumer that adds a penalty term to `J_boozer` is built around the
normalized form (e.g.
`boozersurface_jax.py:1623`, `surfaceobjectives_jax.py:2208`).

But the *CPU* consumer of the same kernel works in the unnormalized
form. `boozersurface.py:788, 802` calls
`_call_boozer_residual_ds`/`_call_boozer_residual_ds2` and stores the
unnormalized `val` / `dval` into `BoozerSurface.res["residual"]` and
hands them to SciPy's L-BFGS-B / Newton solver as the penalty
objective. Therefore:

- If a user re-uses `BoozerSurface.res["residual"]` (a CPU run) to
  set absolute tolerances or compare against `BoozerSurfaceJAX.res`
  values, the values differ by `3·nphi·ntheta`.
- The traceable-bundle helper in `surfaceobjectives_jax.py` (see
  `BoozerResidualJAX` wrapper) uses the JAX normalization; mixing
  with CPU-derived absolute thresholds is unsafe.

**Recommended action:** Either (a) drop the `/num_res` factor in
`boozer_residual_scalar` and apply it explicitly inside the LS
penalty (matching CPU exactly), or (b) document the rescale in the
module docstring (it is partially documented at the top of
`boozer_residual_jax.py` lines 31-34 but the "matching the C++
normalization" comment is incorrect — C++ does NOT carry this
normalization). At minimum, the module docstring should be corrected
and the LS-penalty callsites in `boozersurface_jax.py` should
explicitly call out the rescale.

### F2 — HIGH — `boozer_residual_jacobian_composed` performance: duplicate residual evaluation

`boozer_residual_jacobian_composed` at L717-740:

```python
def boozer_residual_jacobian_composed(x, **kwargs):
    r = _boozer_residual_vector_composed(x, **kwargs)
    J = jax.jacfwd(_boozer_residual_vector_composed)(x, **kwargs)
    return r, J
```

This pattern evaluates the full `DOFs → gamma → Biot-Savart → residual`
pipeline twice. `jax.jacfwd` internally also calls the function to
produce the primal output but does not surface it back to the caller.
The CPU counterpart at `surfaceobjectives.boozer_surface_residual`
(L612-660) shares the forward residual and the derivative arrays in a
single pass.

This is *not* a correctness issue, but at the parity ladder's typical
nphi=ntheta=16 fixture with full coil set, the duplicate forward pass
adds ~one full Biot-Savart evaluation. JAX provides
`jax.linearize(f, x)` returning `(y, jvp_fn)`, or one can wrap
`jax.jacfwd` with `has_aux=True` to keep the primal. Replacing
L738-739 with

```python
primal_and_jacfwd = jax.jacfwd(_boozer_residual_vector_composed, has_aux=False)
# or: y, jvp_fn = jax.linearize(_boozer_residual_vector_composed, x)
```

(combined with a `jvp` sweep over identity basis vectors) would
eliminate the duplicate residual call. `jax.jacrev` is **not**
preferred here because `n_res = 3·nphi·ntheta` is generally larger
than `n_dofs` for BoozerExact (`192` vs e.g. `~80`), and `jacfwd`
runs in `O(n_dofs)` forward sweeps which is correct.

**Recommended action:** Replace L738-739 with a single-pass
`linearize`+basis-sweep or a `jacfwd` with primal capture. Add a
runtime check that warns when `n_res ≫ n_dofs` so the choice
between `jacfwd`/`jacrev` is auditable.

### F3 — MEDIUM — Silent rebinding of `G` when `optimize_G=False`

`_unpack_decision_vector` (`boozer_residual_jax.py:545-556`):

```python
def _unpack_decision_vector(x, coil_arrays, optimize_G):
    sdofs, iota, G = _split_decision_vector(x, optimize_G=optimize_G)
    if optimize_G:
        return sdofs, iota, G
    all_currents = jnp.concatenate([c for _, _, c in coil_arrays])
    mu0 = _as_runtime_float64(4.0e-7 * np.pi, reference=all_currents)
    return sdofs, iota, mu0 * jnp.sum(jnp.abs(all_currents))
```

The CPU oracle `surfaceobjectives.boozer_surface_residual`
(L577-584):

```python
user_provided_G = G is not None
if not user_provided_G:
    G = 2.0 * np.pi * np.sum([np.abs(c.current.get_value()) for c in biotsavart.coils]) \
        * (4 * np.pi * 10 ** (-7) / (2 * np.pi))
```

Algebraically `2π · Σ|I| · (4π·10⁻⁷ / 2π) = 4π·10⁻⁷ · Σ|I| = μ₀·Σ|I|`,
matching `compute_G_from_currents` in
`label_constraints_jax.py:49-61`. The constant is correct.

However: the CPU branch is gated by `if G is None`. Callers can pin
`G` to an arbitrary value (e.g. when iterating on a single coilset
across many surfaces with hand-chosen `G_guess`). In the JAX composed
pipeline the `optimize_G` flag silently re-derives `G` whenever it is
False, *regardless of what the caller packed into `x`*. There is no
"fixed-but-not-optimized G" mode, and the docstring at L599-636 does
not mention this. A caller who places a chosen `G` value as the last
entry of `x` and passes `optimize_G=False` will see that entry
discarded.

Additionally, `optimize_G=False` defeats differentiation through `G`
because `G` is rebuilt from `coil_arrays` (which are inputs); this is
mathematically correct *only when the coils are also being
differentiated*. In a fixed-coil regime (e.g. M5 single-stage
post-stage-2 hot-start) the `G` recomputation injects extra terms
that the CPU oracle does not have.

**Recommended action:** Introduce a `g_mode={"auto","explicit","from_currents"}`
or accept a separate `G_fixed=...` kwarg; explicitly document the
existing behavior.

### F4 — MEDIUM — Composed Hessian path has no C++ oracle parity check

The C++ side exposes `sopp.boozer_residual_ds2` for the full
second-derivative tensor including surface-DOF blocks; per CLAUDE.md
this is the source for the `direct-hessian-oracle` parity lane
(rtol=1e-8). The JAX side exposes (i) `boozer_residual_hessian` whose
surface-DOF blocks are *zero* by construction (M1 limitation,
documented), and (ii) `jax.hessian(boozer_penalty_composed)` which
computes the full surface-DOF Hessian via autodiff.

In `tests/geo/test_boozer_derivatives_jax.py::TestBoozerHessianComposed`:

- `test_hessian_symmetry` checks `H == H.T` only.
- `test_hessian_fd` checks the directional FD of the gradient.
- `test_hessian_taylor_convergence` checks the second-order Taylor
  remainder.

None of these compare the JAX Hessian against `sopp.boozer_residual_ds2`.
The CLAUDE.md note states the `direct-hessian-oracle` lane is
covered by `TestUpstreamFactoryBoozerMatrix::test_penalty_hessian_column_complete_cpu_parity_matrix`,
which lives outside this file. **For the public composed-Hessian path
specifically (`jax.hessian(boozer_penalty_composed)`), there is no
local oracle parity test**, only FD self-consistency.

**Recommended action:** Add a column-by-column parity test against
`sopp.boozer_residual_ds2` for `jax.hessian(boozer_penalty_composed)`
on a small surface (e.g. mpol=ntor=2, nphi=ntheta=8) — this gives a
direct numerical oracle that does not exist today.

### F5 — MEDIUM — `boozer_residual_scalar_and_grad_cpu_ordered` has no direct CPU oracle test

This function (L320-446) is the most complex JAX kernel in the
module: it manually unrolls the chain rule for the `1/|B|`
Jacobian (lines 419-425) to match `boozerresidual_impl.h:141-145`:

JAX:
```python
if weight_inv_modB:
    dmodB = 0.5 * dB2 * wij
    dw = -dmodB * rB2
else:
    dw = jnp.zeros_like(dB2)
drtil0 = dres0 * wij + dw * res0
```

C++ (SIMD path, `boozerresidual_impl.h:141-143`):
```cpp
auto dmodB_ijm = 0.5 * dB2_ijm * wij;
auto dw_ijm = weight_inv_modB ? -dmodB_ijm * rB2ij : simd_t(0.);
auto drtil_ij0m = xsimd::fma(dresij0m , bw_ij , dw_ijm * resij0);
```

These match line-for-line. The G derivative (`grad_G = rtil · wij · B`
in JAX L440 vs `drtil_ij0_dG = wij * dres_ij0_dG` in C++ L194-196 with
`dres_ij0_dG = B(i,j,0)`) and the iota derivative (`dres0_iota =
-B2 · xtheta` in JAX L429 vs `dres_ij0iota = -B2ij * xtheta(i,j,0)`
in C++ L176) also match.

The forward residual numerics (lines 364-377) mirror the C++ point
accumulation exactly. The chain rule for `dB^j_l = (∂_j B_l)·dx^j`
(L382-390) matches the C++ inner loop at L128-130 modulo SIMD lane
packing.

**Concern:** every line is *believed* identical to the C++ kernel,
but there is no test in this file (or in `tests/geo/test_boozer_residual_jax.py`)
that asserts:

```
sopp.boozer_residual_ds(G, iota, B, dB, xphi, xtheta, dx_ds, dxphi_ds, dxtheta_ds, wim)
== boozer_residual_scalar_and_grad_cpu_ordered(G, iota, B, dB, ...)
```

against synthetic inputs. The CLAUDE.md `ls-wrapper-gradient` lane
mentions same-state direct-kernel parity, but it is exercised through
the larger `boozersurface_jax.py` LS bit-identity census, not as a
focused unit test of this kernel.

**Recommended action:** Add a unit-level parity test that builds
synthetic `(B, dB_dx, dx_ds, ...)` arrays and asserts
`(value, gradient)` equality against
`_call_boozer_residual_ds(...)` from
`simsopt.geo.boozersurface`, with `0.5·Σr²` and `Σr·∇r` rescaled to
match the JAX `1/num_res` normalization.

### F6 — LOW — Anonymous "scalar" returned by `boozer_residual_scalar` carries the surface area weighting only implicitly

The CLAUDE.md note about `nfp · 1/(nfp·nphi)` cancellation for
volume/area applies elsewhere; the Boozer residual scalar has no
nfp factor. The CPU kernel sums over the supplied `(nphi, ntheta)`
grid as-is. The JAX kernel matches. **No issue**, but the docstring
in `boozer_residual_jax.py` claims this matches the "C++
normalization" (line 33), which contradicts F1.

### F7 — LOW — `_inverse_modB` calls `_explicit_rsqrt(B²)` on a sum-of-squares

`_inverse_modB(B2) = 1/√(|B|²)` per `boozer_residual_jax.py:85-87`.
The C++ kernel uses `sqrt(rB2ij)` where `rB2ij = 1/B²`
(`boozerresidual_impl.h:55-56`). Mathematically equivalent. The JAX
choice goes through an explicit rsqrt primitive (per the helpers in
`jax_core/_math_utils.py`) for GPU-determinism; the precision will
differ from the C++ `1/sqrt` by 1-2 ulp at the CPU/GPU boundary.
Acceptable inside the `direct-kernel` lane.

### F8 — LOW — `_boozer_weighted_residual` summation over the 3-component norm uses `pairwise_sum_axis`

`_boozer_weighted_residual` at L90-97:

```python
B2 = pairwise_sum_axis(B * B, axis=-1)
```

This dispatches to the pairwise reducer in `jax_core/reductions.py`.
With only 3 components, pairwise vs. naive is identical
mathematically (both reduce to `b0² + b1² + b2²`), but the precise
contraction order can differ by one FMA. The C++ kernel writes
`B(i,j,0)*B(i,j,0) + B(i,j,1)*B(i,j,1) + B(i,j,2)*B(i,j,2)`
explicitly (a left-to-right `((a+b)+c)` sum). The `pairwise_sum_axis`
helper for 3 elements likely produces `(a+b) + c` as well, but it is
worth confirming for the `direct_kernel` 1e-10 contract.

**Recommended action:** Add a one-line comment in
`boozer_residual_jax.py:92` clarifying the 3-element reduction order
and/or replace with explicit `B[..., 0]**2 + B[..., 1]**2 + B[..., 2]**2`
to keep parity easy to audit.

### F9 — INFO — JAX path differs in `dB/dx` index convention from documentation

Docstring at L309-310:
```
xphi:   (nphi, ntheta, 3) toroidal tangent.
xtheta: (nphi, ntheta, 3) poloidal tangent.
```

But CLAUDE.md states: `dB_by_dX[p, j, l] = ∂_j B_l(x_p)` — axis 1 is
the derivative direction, axis 2 is the B component. The JAX kernel
at L382-390 uses `dB_dX[i,j,0,0]·dx0 + dB_dX[i,j,1,0]·dx1 +
dB_dX[i,j,2,0]·dx2` which contracts the third axis with `dx`
(derivative direction). The fourth axis is the B component. This
matches the convention.

The C++ kernel at L128-130 uses
`dB_dx(i,j,0,0)·dx_ds_ij0m + dB_dx(i,j,1,0)·dx_ds_ij1m +
dB_dx(i,j,2,0)·dx_ds_ij2m`, identical.

**No bug** — but worth re-affirming in the JAX docstring since the
4D layout is not described in the function header.

### F10 — INFO — `_inverse_modB` returns NaN/inf on zero-field surfaces; test exists

`test_weighted_zero_field_is_nonfinite` (`test_boozer_residual_jax.py:262-287`)
covers this. The C++ kernel divides by `B²` unconditionally at
`boozerresidual_impl.h:55` and would produce NaN; behavior is
preserved.

### F11 — INFO — Composed VJP `boozer_residual_coil_vjp` has no `weight_inv_modB=True` test path

`TestBoozerResidualCoilVJP` at L756-878 only exercises
`weight_inv_modB=False`. The function accepts the flag and forwards
it to `boozer_residual_vector`. With `weight_inv_modB=True` the VJP
must additionally include the `dw/d(coil)` term through `|B|`.
Without a test the path is unverified, although it is exercised
transitively through the M5 single-stage adjoint pipeline.

**Recommended action:** Parametrize `TestBoozerResidualCoilVJP` over
both `weight_inv_modB` values.

---

## Test coverage gaps

- **T1 — `cpu_ordered` reduction has no C++ binary oracle.**
  `test_cpu_ordered_reduction_matches_ordered_numpy_reference`
  compares to a Python re-implementation, not to the actual
  `sopp.boozer_residual` accumulation order. The bit-identity claim
  relies on this matching, and the only check is JAX-vs-Python (not
  JAX-vs-C++). Adding a `cpu_ordered`-vs-`_call_boozer_residual`
  parity assertion at `rtol=0.0, atol=0.0` would close the loop.

- **T2 — `boozer_residual_scalar_and_grad_cpu_ordered` is untested
  at the unit level.** See F5 for the recommended test sketch.

- **T3 — `boozer_residual_jacobian_composed` has no C++ derivative
  oracle.** The CPU equivalent
  `boozer_surface_residual(..., derivatives=1)` returns
  `(r, J)` where `J[i, k] = ∂r_i / ∂x_k`. A small-fixture parity
  test comparing JAX Jacobian columns against the CPU column-by-column
  (with G-derivative absent on the CPU side and JAX's
  `_unpack_decision_vector` reconciled for `optimize_G=True`) would
  close one of the `derivative-heavy` lane gaps.

- **T4 — `jax.hessian(boozer_penalty_composed)` has no C++ oracle**
  inside this directory; see F4. Existing matrix-column parity test
  is in a different module.

- **T5 — `boozer_residual_coil_vjp(weight_inv_modB=True)` is
  untested.** See F11.

- **T6 — `optimize_G=False` semantics are not tested for the
  unintended-G-discard behavior.** See F3. A simple test that packs
  a wrong `G` into `x[-1]`, sets `optimize_G=False`, and asserts the
  returned scalar matches one computed with the auto-derived `G`
  would document the contract.

---

## Recommended actions, ordered by severity

1. **HIGH (F1)** Correct the module docstring at
   `boozer_residual_jax.py:31-34` — the JAX normalization does *not*
   match the C++ kernel — and audit every cross-runtime threshold
   that compares JAX `J_boozer` against CPU `J_boozer`. Consider
   removing the `/num_res` factor inside the kernel and applying it
   at the LS-penalty boundary so the kernel matches the C++ public
   API bit-for-bit.

2. **HIGH (F2)** Refactor `boozer_residual_jacobian_composed` to
   evaluate the forward residual once (via `jax.linearize` or
   `jacfwd` with primal capture). For the typical
   BoozerExact fixture this saves one full Biot-Savart evaluation
   per Jacobian call.

3. **MEDIUM (F3)** Document — and ideally fix — the `optimize_G=False`
   behavior of `_unpack_decision_vector`. Either error out if the
   caller packs a `G` into `x` with `optimize_G=False`, or accept an
   explicit `G_value=...` kwarg that bypasses the
   `μ₀·Σ|I|` re-derivation.

4. **MEDIUM (F4, T4)** Add a C++ Hessian oracle parity test for
   `jax.hessian(boozer_penalty_composed)` in
   `test_boozer_derivatives_jax.py`.

5. **MEDIUM (F5, T2)** Add a unit-level C++ parity test for
   `boozer_residual_scalar_and_grad_cpu_ordered` against
   `_call_boozer_residual_ds`.

6. **MEDIUM (T3)** Add a C++ Jacobian oracle parity test for
   `boozer_residual_jacobian_composed`.

7. **LOW (F8)** Replace `pairwise_sum_axis(B*B, axis=-1)` with the
   explicit 3-term sum, or document the implicit equivalence.

8. **LOW (F11, T5)** Parametrize the coil-VJP test over both
   `weight_inv_modB` values.

9. **INFO (T6)** Add a test asserting `optimize_G=False` discards any
   user-packed `G[-1]` entry — to make the contract auditable.

10. **INFO (F9)** Add the 4D `dB_by_dX` axis convention to the
    `boozer_residual_scalar_and_grad_cpu_ordered` docstring.

---

## Closing notes

The raw `boozer_residual_scalar` and `boozer_residual_vector`
forward pipelines are mathematically faithful to
`boozer_residual_impl<T, 0>`. The chain-rule kernel in
`boozer_residual_scalar_and_grad_cpu_ordered` is a careful
line-for-line port of the C++ first-derivative kernel (verified F5).
The composed-pipeline gradient (`boozer_penalty_grad_composed`) is
the standard JAX reverse-mode through Biot-Savart, with no
mathematical equivalent on the CPU side until the full Hessian is
assembled. The principal risks are (i) the scalar-normalization
divergence (F1), which is consistent inside the JAX subtree but
inconsistent with CPU absolute thresholds, and (ii) test coverage:
several kernels lack direct C++ oracles and rely on FD or
JAX-vs-JAX self-consistency.
