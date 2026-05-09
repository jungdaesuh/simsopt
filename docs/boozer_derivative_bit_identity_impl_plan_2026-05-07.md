# Boozer Derivative Bit-Identity Zeroing — Detailed Implementation Plan

- Date: 2026-05-07 (revised 2026-05-08)
- Branch: `gpu-purity-stage2-20260405`
- Status: Phases 0–3, 6, 7 landed in commit `e61370cdf` (2026-05-08);
  Phase 4 entry checklist authoritative in §10 (revised), §19, §20, §21
- Companion strategy doc: `docs/boozer_derivative_bit_identity_zeroing_plan_2026-05-07.md`
  (older `optimization_barrier`/`reduce_precision` wording superseded by §21)
- Companion dual-mode doc: `docs/parity_dual_mode_contract_2026-05-08.md`
  (diagnostic reporting context only; does not loosen the strict gate)
- Companion in-flight slice: `docs/boozer_bfgs_pre_newton_contract_impl_plan_2026-05-07.md`

## 1. Why This Plan Exists

The 2026-05-07 BFGS pre-Newton contract slice closed the larger physics drift,
re-armed the Newton non-worsening guard, and added a CPU-ordered Boozer
value/gradient route. The strict acceptance gate is still red:

- `.artifacts/parity/20260507-bfgs-prenewton-cpuordered-vg-m1/result.json`
  - `passed`: `false`
  - same-candidate `max_candidate_abs_diff`: `0.0`
  - same-candidate `max_objective_abs_diff`: `2.220446049250313e-16`
  - first adapter-evaluation split: index `2`, decision-vector max abs diff
    `1.6063539387545234e-15`
  - `boozer_solve.pre_newton_state`: `4.519706948979962e-09`
  - `boozer_solve.pre_newton_objective_gradient`: `1.5830200208379184e-10`
  - `iota_penalty.adjoint`: `2.318643055332359e-10`
  - final physics drift: ~1e-15 (within ladder, but the gate is stricter)

A six-lane root-cause investigation
(`.artifacts/bit-identity-deepdive-2026-05-07/agent_{1..6}_*.md`) converged on
three structural cross-implementation mismatches:

1. **Surface geometry**
   - JAX matmul reduction order `(V @ coeffs.T) @ W.T` at
     `src/simsopt/geo/surface_fourier_jax.py:304` vs C++ nested-(m,n)
     scalar accumulation at `src/simsoptpp/surfacexyztensorfourier.h:437`.
   - 2π factor distributed inside basis/product terms at
     `surface_fourier_jax.py:255, 466` vs applied once outside at
     `surfacexyztensorfourier.h:451–455`.
2. **Biot-Savart**
   - `r_inv3 = r_inv * inv(r2)` at `src/simsopt/jax_core/biotsavart.py:357` vs
     `r_inv*r_inv*r_inv` at `src/simsoptpp/biot_savart_impl.h:66`.
   - Pairwise tree reduction across `nq` quadrature points (XLA) vs sequential
     `+=` at `biot_savart_impl.h:62`.
3. **Boozer residual gradient**
   - 3-term `a*b + c*d + e*f` sums in
     `boozer_residual_scalar_and_grad_cpu_ordered`
     (`boozer_residual_jax.py:320–432`) were observed by Lane 4 to lower to
     XLA's preferred FMA shape `fma(e, f, fma(a, b, c*d))` on aarch64, while
     xsimd writes `fma(a, b, fma(c, d, e*f))` at
     `boozerresidual_impl.h:128–148`. Treat this as a candidate until the
     x86_64 side track below reproduces it on the production target.

Lane 6 ladder names **forward `gamma` evaluation** as the FIRST kernel that
ULP-drifts (2–10 ULP). The cascade flows: `gamma` → `gammadash{1,2}` →
`dgamma_by_dcoeff`/`dgammadash{1,2}_by_dcoeff` (autodiff inheritance) →
Biot-Savart (B and dB/dX depend on `gamma`) → Boozer residual gradient.

The 1.606e-15 first-evaluation gradient diff (Lane 1) is propagated by SciPy
L-BFGS-B's first step verbatim (`x1 = x0 − α H g`, ratio = −1.0 exact), then
amplified by BFGS curvature to 4.5e-9 in `pre_newton_state`. Final physics
re-collapses to ~1e-15 because the inner solve still converges, but the strict
contract requires the BFGS trajectory itself to be byte-identical.

## 2. Contract (Hard Constraints — Do Not Change)

From `docs/boozer_derivative_bit_identity_zeroing_plan_2026-05-07.md:142–160`,
gate code at `benchmarks/single_stage_init_parity.py:1905`, regression test at
`tests/test_benchmark_helpers.py:1415`:

- **Definition Of Zero** — all of the following must hold on the CPU parity
  lane:
  - The same outer candidate reaches the same inner Boozer pre-Newton decision
    vector bytes before SciPy sees the value/gradient callback.
  - Every traced derivative input array has matching `np.float64` shape,
    C/F memory-order contract, and raw byte digest.
  - Fixed-candidate CPU and JAX Boozer LS scalar and gradient have
    `max_abs_diff == 0.0`.
  - Strict `parity_bug_census` has no divergent `boozer_solve.pre_newton_*`.
  - Same-candidate replay and final physics gates remain at least as good as
    the current post-fix artifact.
- **Non-goals**
  - Do not loosen tolerances.
  - Do not add production fallbacks, defensive retries, or route switches.
  - Do not replace upstream CPU C++/pybind code.
  - Do not claim success from JAX-vs-JAX agreement.
  - Do not pin SciPy optimizer status/counter differences as the root cause
    until raw derivative inputs have been proven byte-identical.

**Oracle determinism scope.** "CPU C++ as oracle" is a per-build, per-host
claim: byte identity is fixed under (input, CMake flags
`-march=native -ffp-contract=fast` at `CMakeLists.txt:59,61,63`,
`xsimd::simd_type<double>::size` for the host ISA, libm version, oracle
function path). Cross-machine parity is out of scope; that is a separate ladder.

The 16 `#pragma omp parallel for` directives in
`src/simsoptpp/surfacexyztensorfourier.h` parallelize outer phi loops with
disjoint output cells (no `reduction(+:val)`), so they do not break the
per-build determinism premise.

## 3. Strategy

> Trace raw inputs first. Prove the owner by substitution. Then implement the
> smallest CPU-oracle-order JAX parity kernel for that owner.

Production fast JAX kernels stay fast and untouched. Parity twins live in
dedicated `*_cpu_ordered` modules and are selected only by the existing backend
parity policy (`SIMSOPT_BACKEND_MODE=jax_cpu_parity`/`jax_gpu_parity` via
`is_parity_mode()`), not by a second Boozer-specific env var. The JAX product
path keeps `einsum`/matmul/`jacfwd` for performance.

## 4. Pre-requisites

- [ ] Confirm working tree clean of unrelated edits before starting Phase 0.
- [ ] `.conda/jax-0.9.2/bin/python` available (env shows JAX 0.10.0; document
      version drift if relevant).
- [ ] Existing artifact preserved:
      `.artifacts/parity/20260507-bfgs-prenewton-cpuordered-vg-m1/result.json`.
- [ ] All six deep-dive reports preserved at
      `.artifacts/bit-identity-deepdive-2026-05-07/agent_{1..6}_*.md`.

## 5. Side Track (Run In Parallel With Phase 0)

Lane 4 and Lane 5 reproducers were lost from `/tmp` and their FMA-fusion claims
were derived on aarch64 (local macOS) only. Phase 6 strategy depends on what
fusion shape XLA picks on x86_64 production (RunPod A100/H200).

- [ ] Recreate `lane4_repro.py` for the 9-pattern FMA fusion sweep on x86_64.
  - File target: `benchmarks/parity/lane4_fma_fusion_repro_x86.py`.
    Keep this as a benchmark/artifact reproducer unless it can run portably in
    CI; the ordinary pytest suite must not hard-code one host's codegen shape.
  - Fixture: 5000 random `(a, b, c, d, e, f)` triples in
    `[-1, 1]^6`, run on x86_64, compare JAX `a*b + c*d + e*f` against all 9
    explicit fusion shapes.
- [ ] Recreate `lane5_repro.py` for the HLO dump + FMA detection.
  - Use current JAX AOT APIs: `jax.jit(fn).lower(*args).as_text(dialect="hlo")`
    or `jax.jit(fn).lower(*args).compiler_ir("hlo")`. Do not use
    `jax.xla_computation`; it is deleted in installed JAX 0.10.0.
  - HLO alone is insufficient for a final FMA verdict because `fma` may appear
    during later backend codegen. Preserve optimized IR/assembly when available.
- [ ] Probe `XLA_FLAGS=--xla_cpu_enable_fast_math=false` only as an ablation
      after the baseline reproducer matches the failing artifact under the
      artifact's original env. Do not make this flag part of the baseline
      contract unless the side track proves it is required and supported.
- [ ] If x86_64 picks a different fusion shape than aarch64, update Phase 6
      remediation choice accordingly.

### Acceptance Gate

- [ ] Documented `(host, ISA, JAX version)` → `(fusion shape, HLO ops)` mapping
      committed under
      `.artifacts/bit-identity-deepdive-2026-05-07/lane4_x86_repro/` and
      `.artifacts/bit-identity-deepdive-2026-05-07/lane5_x86_repro/`.

## 6. Phase 0 — Freeze The Oracle Artifact

**Context:** the current artifact (`20260507-bfgs-prenewton-cpuordered-vg-m1`)
is the failing baseline. Phase 1 needs a smaller, deterministic reproducer that
runs one outer candidate and dumps raw Boozer LS inputs from both CPU and JAX
paths.

### Tasks

- [ ] Create reproducer entry point.
  - Script path: `benchmarks/parity/boozer_derivative_input_repro.py`.
  - Picks one outer candidate (the same one used by the failing artifact;
    extract via `parity_bug_census` first divergent pair).
  - Runs one inner Boozer LS call on each backend (CPU, JAX) at that candidate.
  - Honors `--dump-arrays <DIR>` to write `candidate.json`,
    `census.ndjson`, and `census_summary.json` (the SHA-256 column on
    every boundary array). **Per-array `.npy` byte dumps were planned but
    not implemented in the Phase 0/1 slice (commit `e61370cdf`); see §10
    P4.1 for the deferred work and the recommended
    `--dump-arrays-as-npy <DIR>` extension.**
- [ ] Record and replay the runtime environment in the repro script:
  - First run with the failing artifact's captured env exactly as recorded.
  - Then run the deterministic local subprocess variant with `JAX_ENABLE_X64=true`,
    `JAX_PLATFORMS=cpu`, and `OMP_NUM_THREADS=1`; record whether it preserves
    the same first divergent layer and magnitudes.
  - Record, but do not rewrite, `XLA_FLAGS` for the baseline. Fast-math toggles
    belong to the FMA ablation side track.
  - Include existing strict-parity flags from
    `benchmarks/single_stage_init_parity.py`.
- [ ] Add reproducer artifact directory:
  `.artifacts/parity/20260507-boozer-deriv-input-repro-m1/`.
- [ ] Confirm reproducer reports the same first-divergence layer
      (`boozer_solve.pre_newton_state`) as the postfix artifact.

### Acceptance Gate

- [ ] One-command reproducer exists and reproduces the strict-gate failure on
      a single fixed candidate in <60s wall-clock (measure on local host).
- [ ] Documented in Phase 0 acceptance section of strategy doc.

## 7. Phase 1 — Fixed-Candidate Derivative Input Census

**Context:** comparing final scalar+gradient is too late. The trace must
compare the derivative inputs **before** the Boozer residual derivative code
contracts them. The Boozer LS callback boundary inputs are:

```
boozer_residual_ds(G, iota, B, dB_dx, xphi, xtheta,
                   dx_ds, dxphi_ds, dxtheta_ds, weight_inv_modB)
```

(verified at `src/simsoptpp/boozerresidual_py.cpp:11` and
`src/simsopt/geo/boozer_residual_jax.py:320`).

Upstream of the boundary, the producers are:

| Boundary array | CPU producer | JAX producer |
|---|---|---|
| `gamma` | `surface.gamma()` | `_surface_geometry_from_dofs` |
| `xphi`  | `surface.gammadash1()` | `_surface_geometry_from_dofs` |
| `xtheta`| `surface.gammadash2()` | `_surface_geometry_from_dofs` |
| `dx_ds` | `surface.dgamma_by_dcoeff()` | `jax.jacfwd(geometry_arrays)` |
| `dxphi_ds` | `surface.dgammadash1_by_dcoeff()` | `jax.jacfwd(geometry_arrays)` |
| `dxtheta_ds` | `surface.dgammadash2_by_dcoeff()` | `jax.jacfwd(geometry_arrays)` |
| `B` | `BiotSavart.B()` | `_field_terms_for_local_label` |
| `dB_dX` | `BiotSavart.dB_by_dX()` | same |

(file:line map in `bit_identity_zeroing_plan_2026-05-07.md:46–119`.)

### Schema

JSON object per array, written one per line to NDJSON:

```json
{
  "array_name": "gamma",
  "producer": "cpu" | "jax",
  "stage": "boozer_ls_callback_input",
  "dtype": "float64",
  "shape": [nphi, ntheta, 3],
  "strides": [..],
  "contiguity": "C" | "F",
  "sha256_float64_bytes": "<hex>",
  "norm_l2": <double>,
  "norm_linf": <double>
}
```

Plus a paired `_diff` record per array:

```json
{
  "array_name": "gamma",
  "stage": "boozer_ls_callback_input",
  "max_abs_diff": <double>,
  "argmax_abs_diff": [i, j, c],
  "first_unequal_byte_index": <int | null>,
  "first_unequal_numeric_index": [i, j, c] | null,
  "n_bit_different_entries": <int>,
  "byte_identical": <bool>
}
```

### Tasks

- [ ] Define `boozer_derivative_input_census` schema as a typed dataclass.
  - File: `benchmarks/parity/boozer_derivative_input_census.py` (diagnostic
    namespace; not in the production package import path).
  - Use `@dataclasses.dataclass(frozen=True)` for SSOT/immutable
    contract.
- [ ] Implement CPU capture helper.
  - Function: `capture_cpu_boozer_inputs(boozer_surface, candidate, *,
    weight_inv_modB)` → dict of named arrays.
  - Factor a private `_boozer_penalty_vectorized_inputs(...)` helper in
    `src/simsopt/geo/boozersurface.py` for the arrays currently materialized
    before `_call_boozer_residual_ds`. As of HEAD `e61370cdf` (2026-05-08),
    the helper is defined at `boozersurface.py:643`, the call from
    `boozer_penalty_constraints_vectorized` is at `boozersurface.py:750`,
    and the consuming `_call_boozer_residual_ds(...)` is at
    `boozersurface.py:769`.
  - `boozer_penalty_constraints_vectorized(...)` calls that helper, and the
    benchmark census calls the same helper. Do not add a public `_capture_cb`
    kwarg or callback seam to `BoozerSurface.boozer_penalty_constraints_vectorized`.
- [ ] Implement JAX capture helper.
  - Function: `capture_jax_boozer_inputs(spec, candidate, *, weight_inv_modB)`.
  - Factor a private `_boozer_penalty_value_and_grad_inputs_cpu_ordered(...)`
    helper in `src/simsopt/geo/boozersurface_jax.py` for the arrays
    currently built before the `boozer_residual_scalar_and_grad_cpu_ordered`
    call. As of HEAD `e61370cdf` (2026-05-08), the inputs helper is at
    `boozersurface_jax.py:1352`, the consuming v+g function is at
    `:1437`, and the residual call site itself is at `:1483`.
  - `_boozer_penalty_value_and_grad_cpu_ordered(...)` calls that helper, and the
    benchmark census calls the same helper.
  - Materialize JAX arrays to NumPy via `jax.device_get(...)` to enforce host
    boundary parity.
- [ ] Compute SHA-256 over raw float64 bytes:
  - `np.ascontiguousarray(arr, dtype=np.float64).tobytes()` → `hashlib.sha256(...).hexdigest()`.
  - Validate `arr.dtype == np.float64` first. If not, emit a layout/type failure
    record and stop that census lane; do not cast away the evidence.
  - Record original stride metadata and the bytes used at the residual boundary.
    If the boundary requires contiguous packing, make that packing explicit in
    the producer helper rather than silently re-casting inside the digest.
- [ ] Diff helper:
  - `compare_census_arrays(cpu, jax)` → list of paired `_diff` records.
  - First-unequal-byte-index via `np.frombuffer(...).view(np.uint8)` differential.
  - First-unequal-numeric-index via `np.where(cpu != jax)`.
- [ ] Wire census into the Phase 0 reproducer:
  - Add `--census` flag.
  - Emit NDJSON to `.artifacts/parity/20260507-boozer-deriv-input-repro-m1/census.ndjson`.
  - Emit a one-line summary: name and metadata of the FIRST array where
    `byte_identical=False`.
- [ ] Add red test:
  - File: `tests/geo/test_boozer_derivative_input_census.py`.
  - Do not commit an `xfail` as the red test; this repo audits skip/xfail
    markers and the strict contract forbids hiding the divergence.
  - During Phases 0-1, keep the failing owner proof as a benchmark artifact.
    Promote it to pytest in the same slice that makes the assertion pass.
  - Assert the census names the expected first owner while fixed-candidate
    byte-identity assertions pass after Phase 2/3.

### Acceptance Gate

- [ ] Census names the first non-byte-identical derivative input array.
- [ ] NDJSON artifact preserved.
- [ ] Red test in place.

## 8. Phase 2 — Surface Geometry And Surface Derivative Bit Identity

**Context:** Lane 6 ladder strongly suggests `gamma` is the first owner.
Lane 2 confirmed two specific algorithmic divergences for `gammadash1`/`gammadash2`:
matmul reduction order and 2π distribution. Phase 2 implements CPU-ordered JAX
parity twins.

### CPU Reference Algorithm (per `surfacexyztensorfourier.h`)

```cpp
for (int k1 = 0; k1 < numquadpoints_phi; ++k1) {     // OMP-parallel outer
  double phi    = 2*M_PI*quadpoints_phi[k1];
  double sinphi = sin(phi);
  double cosphi = cos(phi);
  for (int k2 = 0; k2 < numquadpoints_theta; ++k2) {
    double xhat = 0, yhat = 0; // sequential accumulator
    for (int m = 0; m <= 2*mpol; ++m) {
      for (int n = 0; n <= 2*ntor; ++n) {
        xhat += get_coeff(0, m, n) * basis_fun(0, n, k1, m, k2);
        yhat += get_coeff(1, m, n) * basis_fun(1, n, k1, m, k2);
        // ...
      }
    }
    double xdash = xhatdash * cosphi - yhatdash * sinphi
                 - xhat    * sinphi  - yhat    * cosphi;
    data(k1, k2, 0) = 2*M_PI * xdash;     // 2π applied once at outer level
    // ...
  }
}
```

### Tasks

- [ ] New module: `src/simsopt/geo/surface_fourier_jax_cpu_ordered.py`.
  - Pure-JAX, no simsoptpp imports (M1 contract).
  - Single source of truth for parity twins; production matmul kernels stay in
    `src/simsopt/geo/surface_fourier_jax.py` untouched.
- [ ] Implement `surface_gamma_cpu_ordered(...)`.
  - `lax.fori_loop` over `(k1, k2)` flattened, inner `lax.fori_loop` over
    `(m, n)` flattened.
  - Match basis_fun signature: cos-cos / cos-sin / sin-cos / sin-sin per
    stellsym scatter convention.
  - Accumulate via scalar `xhat += a*b` (no `einsum`, no matmul).
- [ ] Implement `surface_gammadash1_cpu_ordered(...)` and
      `surface_gammadash2_cpu_ordered(...)`.
  - Apply `2*pi` exactly once at outer assembly (line analogous to
    C++:451–455), not distributed inside `dV` or basis terms.
  - Match the C++ formula `xdash = xhatdash*cosphi - yhatdash*sinphi
    - xhat*sinphi - yhat*cosphi` operator-for-operator.
- [ ] Implement `dgamma_by_dcoeff_cpu_ordered(...)` and
      `dgammadash{1,2}_by_dcoeff_cpu_ordered(...)`.
  - **Decision required**: analytic kernel (mirroring
    `surfacexyztensorfourier.h::dgamma_by_dcoeff_impl`) vs `jax.jacfwd` over
    the new `gamma_cpu_ordered`.
  - Default decision: write analytic kernels. `jax.jacfwd` over parity twins may
    still choose derivative arithmetic and batching order that differs from the
    C++ analytic derivative routines.
  - Validate against C++ via Phase 1 census.
- [ ] Wire existing parity-policy selection.
  - Add an internal policy argument to `_surface_geometry_and_derivatives_from_dofs`
    at `src/simsopt/geo/boozersurface_jax.py:1128`, derived from
    `is_parity_mode()` at the caller boundary rather than a new env var.
  - In parity policy, dispatch to `surface_*_cpu_ordered` instead of
    `_surface_geometry_from_dofs` + `jax.jacfwd`.
  - Plumb the policy through `_boozer_penalty_value_and_grad_cpu_ordered` at
    `boozersurface_jax.py:1437` (and the inputs helper
    `_boozer_penalty_value_and_grad_inputs_cpu_ordered` at `:1352`)
    and include it in any closure cache key it changes.
- [ ] Census in parity-mode must report `gamma`, `gammadash1`, `gammadash2`,
      and their coeff derivatives as byte-identical.
- [ ] Test:
  - `tests/geo/test_surface_fourier_jax_cpu_ordered.py`.
  - Fixture matrix: `(SurfaceXYZTensorFourier, mpol, ntor, nfp, stellsym,
    nphi, ntheta)` over the production set.
  - Assert byte identity against `surface.gamma()`, `gammadash1()`,
    `gammadash2()`, `dgamma_by_dcoeff()`, `dgammadash1_by_dcoeff()`,
    `dgammadash2_by_dcoeff()`.

### Acceptance Gate

- [ ] Census in parity mode shows surface-side arrays byte-identical or
      documents the exact remaining first-mismatch with arithmetic-order
      reason.

## 9. Phase 3 — Biot-Savart Field Derivative Bit Identity

(See strategy doc `:334–379`.)

**Context:** even with surface bytes matching, `B` and `dB_dX` can differ
because the JAX path uses `jax.jacfwd` / `jax.linearize` over a kernel whose
algebra differs from C++'s xsimd-FMA implementation.

### CPU Reference Algorithm (per `biot_savart_impl.h:62–73`)

```cpp
for (int j = 0; j < num_quad_points; ++j) {
  auto diff             = point_i - Vec3dSimd(gamma_j_ptr[3*j+0], ...);
  auto norm_diff_2      = normsq(diff);
  auto norm_diff_inv    = rsqrt(norm_diff_2);
  auto norm_diff_3_inv  = norm_diff_inv*norm_diff_inv*norm_diff_inv;
  auto cross_           = cross(dgamma_by_dphi_j_simd, diff);
  B_i.x = xsimd::fma(cross_.x, norm_diff_3_inv, B_i.x);
  // ...
}
```

### Tasks

- [ ] New module: `src/simsopt/jax_core/biotsavart_cpu_ordered.py`.
  - Pure JAX, no simsoptpp.
- [ ] Implement `biot_savart_B_cpu_ordered(...)` mirroring C++ algebra:
  - `diff = point - gamma` (sign convention matches C++).
  - `cross(dgamma, diff)` (operand order matches C++).
  - `r_inv = rsqrt(diff @ diff)`.
  - `r_inv3 = r_inv * r_inv * r_inv` (NOT `r_inv * inv(r2)`).
  - Sequential `lax.fori_loop` over quadrature points (no pairwise tree).
- [ ] Implement `dB_by_dX_cpu_ordered(...)` mirroring C++ derivative kernel
      operator-for-operator.
- [ ] Implement `biot_savart_VJP_cpu_ordered(...)` if Phase 1 census shows VJP
      path is consumed.
- [ ] Wire existing parity-policy selection into
      `_field_terms_for_local_label`/`_field_terms_for_toroidal_flux`
      (`src/simsopt/geo/boozersurface_jax.py:1046–1126`; the
      `_local_label` def begins at `:1046` and `_toroidal_flux` at `:1089`)
      by importing the new
      parity kernel from `src/simsopt/jax_core/biotsavart_cpu_ordered.py`.
  - Do not edit `src/simsopt/jax_core/biotsavart.py`; that file remains the
    production hot-path kernel factory. For `coil_set_spec` inputs, add a local
    Boozer parity helper that iterates `grouped_field_inputs_from_spec(...)` and
    calls `_evaluate_grouped_field_group(..., biot_savart_B_and_dB_cpu_ordered)`.
- [ ] Test:
  - `tests/field/test_biotsavart_jax_cpu_ordered.py`.
  - Fixture: identical coil set + quadrature points used by production
    Boozer LS.
  - Assert byte identity against `BiotSavart.B()`, `BiotSavart.dB_by_dX()`.

### Acceptance Gate

- [ ] Census in parity mode shows `B` and `dB_dX` byte-identical.

## 10. Phase 4 — Residual Derivative Assembly Identity (revised 2026-05-08)

(See strategy doc `:381–431`.)

> **Revision history.** The original Phase 4 plan (commit `fc58b90f5`,
> 2026-05-07) recommended `jax.lax.optimization_barrier` and
> `jax.lax.reduce_precision` as FMA-fusion levers. Empirical probes on
> 2026-05-08 (recorded in §19) **retracted** both. This revision replaces
> §10 with the candidate direction. See §19 for the supporting evidence and
> §20 for the expanded ablation surface.

**Context.** Phase 2/3 left a 1–2 ULP gradient drift after the cpu_ordered
surface and Biot-Savart twins reduced the dominant `gamma` and `B`
contributions. The residual lives in the gradient assembly inside
`boozer_residual_scalar_and_grad_cpu_ordered`
(`src/simsopt/geo/boozer_residual_jax.py:320–432`). The C++ oracle uses
explicit `xsimd::fma` nests at
`src/simsoptpp/boozerresidual_impl.h:128, 137, 148`, so on the C++ side
fusion is *forced* — the JAX side either fuses to the same nesting shape or
it doesn't. This phase aligns the JAX expression structure to the C++
nesting, then verifies via post-optimization LLVM IR + object disassembly,
and validates with a boundary-pinned residual-gradient byte test.

### Local probes, candidate levers, and dead ends (2026-05-08)

* **`jax.lax.optimization_barrier`** — DEAD END at the LLVM-fusion layer.
  XLA's `OptimizationBarrierExpander` (`xla/service/cpu/cpu_compiler.cc:943`)
  deletes every `kOptimizationBarrier` HLO op via `ReplaceAllUsesWith(arg)`
  before LLVM IR is emitted. The HLO-level barrier prevents CSE/algebraic
  simplification across it, but LLVM's mul+add → fma combine is unaffected.
  Empirical probe on JAX 0.9.2 (HEAD on 2026-05-08): barrier present in
  pre-opt HLO, absent in optimized HLO, mul+add fused in post-opt LLVM IR.
  The autodiff-rule question is moot at this layer.
* **`jax.lax.reduce_precision(x, 11, 52)`** — DEAD END for float64. Per
  OpenXLA `ReducePrecision` op semantics, "setting [exponent_bits,
  mantissa_bits] to those of the type results in an op that is a no-op."
  `(11, 52)` is exactly float64, so no rounding boundary is inserted. Local
  probe confirmed plain `fmul`/`fadd` survive in optimized LLVM IR.
* **`xla_cpu_enable_fast_math=false`** — necessary but not sufficient. It
  controls per-IR-instruction `FastMathFlags` but XLA's CPU target options
  unconditionally set `AllowFPOpFusion = llvm::FPOpFusion::Fast`
  (`xla/service/cpu/cpu_aot_loader.cc:55-58`, comment "Always allow FMA
  fusion"). The LLVM combine therefore fires regardless.
* **`xsimd::rsqrt<double>` parity** — NOT a residual source on this branch.
  Verified via direct read of `src/simsoptpp/simdhelpers.h:152, 163`:
  both active overloads are `1./sqrt(r2)` / `1./std::sqrt(r2)`. The
  approximate `_mm_rsqrt_ps` and AVX-512 Newton-refinement paths are
  literally commented out (AVX-512 + Newton-refinement at lines 139-149
  is inside a `/* ... */` block comment 132-151; the inline
  `_mm_rsqrt_ps` ps-roundtrip is at line 156). JAX's
  `_explicit_rsqrt = 1.0 / jnp.sqrt(x)` should match modulo libm.
* **Explicit grouping (local candidate lever).** Right-nesting
  `tail = c*d + e*f; result = a*b + tail` in JAX produces, on aarch64,
  object code shaped `fmul e*f; fmadd c*d + tail; fmadd a*b + tail` —
  matching the C++ `xsimd::fma(a, b, xsimd::fma(c, d, e*f))` nesting at
  `boozerresidual_impl.h:128`. Empirical reproducer recorded in §19. This
  is not a production acceptance proof until the same shape is confirmed for
  float64 on the x86_64 production target.

### Tasks (revised)

- [ ] **P4.1 — Pin boundary inputs.** The current census artifact at
      `.artifacts/parity/20260507-boozer-deriv-input-repro-m1/cpu_ordered_full/`
      contains only `candidate.json`, `census.ndjson`, and
      `census_summary.json` — there is no per-array byte dump. The capture
      helpers in `benchmarks/parity/boozer_derivative_input_census.py` and
      `benchmarks/parity/boozer_derivative_input_repro.py` only digest
      `tobytes()` for the SHA-256 column; they do not persist arrays. P4.1
      must EXTEND the capture helper (or the repro driver) to write two
      distinct artifacts:
      - producer snapshots: `cpu_<name>.npy` and `jax_<name>.npy`, preserving
        the current CPU-vs-JAX boundary-input census evidence;
      - a canonical residual-input bundle: `canonical_<name>.npy`, copied
        from the CPU oracle producer and used as the single input set for
        both the C++ and JAX residual calls in P4.5/P4.5b.
      Suggested CLI surface (final flag name is implementer's choice; this
      is the recommended shape):
      ```
      python benchmarks/parity/boozer_derivative_input_repro.py \
          --census --parity-policy cpu_ordered \
          --dump-arrays-as-npy .artifacts/parity/<DATE>-residual-pinned-inputs/
      ```
      Implementation: write the CPU side via
      `np.save(<dir>/cpu_<name>.npy, np.ascontiguousarray(arr, dtype=np.float64))`,
      write the JAX side via
      `np.save(<dir>/jax_<name>.npy, np.asarray(jax.device_get(arr), dtype=np.float64))`,
      and write `canonical_<name>.npy` from the CPU side for every name in
      `CENSUS_BOUNDARY_ARRAY_ORDER`. Emit a sibling `manifest.json` listing
      the role (`cpu`, `jax`, `canonical`), name, dtype, shape, and sha256
      for every file. The producer snapshots remain upstream diagnostics:
      current `cpu_ordered_full` has `n_byte_identical_arrays = 0`, so they
      cannot be used to claim residual isolation. The canonical bundle is
      what makes P4.5/P4.5b feed byte-identical inputs to both residual
      implementations.
- [ ] **P4.2 — Baseline C++ object disassembly.** Disassemble the simsoptpp
      `.o` for the xsimd::fma sites at `boozerresidual_impl.h:128, 137, 148`
      with `objdump -d` (and on the production target host if applicable,
      see Side Track §5). Save the reference instruction sequence under
      `.artifacts/parity/<DATE>-fma-shape-baseline/` so P4.4 has a known
      target shape per host.
- [ ] **P4.3 — Restructure JAX expressions to right-nested form.** See §20
      for the full ablation surface (≥14 candidate sites, *not* the three
      originally listed). Apply ONE site at a time per Phase 5 ablation
      discipline; measure with P4.5 between each change.
- [ ] **P4.4 — Verify shape via dump-to plus object disassembly.**
      Run with `XLA_FLAGS="--xla_dump_to=<dir>"` (no other flags — see §19
      for why `--xla_dump_llvm_ir=true` is rejected by this build). The
      dump auto-emits `.ir-no-opt.ll`, `.ir-with-opt.ll`,
      `.obj-file.*.o`, and pre/post-optimization HLO. Inspect via:
      * `.ir-with-opt.ll` for the *association* shape (which mul-add pair
        is grouped together).
      * `objdump -d *.obj-file.*.o` for the *codegen-level* FMA emission
        (`vfmadd231sd`/`vfmadd213sd` on x86, `fmadd`/`fmla` on aarch64).
      LLVM IR alone is INSUFFICIENT — aarch64 codegen welds `fmul + fadd`
      into `fmadd` without materializing `@llvm.fmuladd`. Diff against the
      P4.2 C++ baseline at the corresponding instruction sites.
- [ ] **P4.5 — Boundary-pinned residual-only byte test.**
      Add `tests/geo/test_boozer_residual_pinned_input_byte_parity.py`
      under the `parity_census` marker. Load the P4.1
      `canonical_<name>.npy` bundle once, then pass the same host arrays to
      `_call_boozer_residual_ds` (CPU C++) and to
      `boozer_residual_scalar_and_grad_cpu_ordered` (JAX). Do not feed
      separate `cpu_<name>.npy` and `jax_<name>.npy` producer snapshots into
      this test; those arrays are allowed to remain non-byte-identical and
      belong to the upstream census. Assert `max|grad_jax - grad_cpu| == 0.0`.
      **This test is the
      *first-tier* arbiter — it isolates the residual kernel only.**
      Object-shape match in P4.4 is a *necessary* condition, not
      sufficient — vector lane behavior, sub-expression reordering, and
      sites outside the explicit FMA pattern can still produce drift.
- [ ] **P4.5b — Boundary-pinned full penalty byte test (second-tier
      arbiter).** P4.5 only proves the raw residual kernel is byte-clean.
      The full SciPy gradient that BFGS sees is built at
      `boozersurface_jax.py:1497-1525` and adds a label term plus rz axis
      penalty:
      ```python
      label_value, label_gradient = jax.value_and_grad(
          _label_value_from_surface_dofs)(...)
      rl = weight_sqrt * (label_value - targetlabel)
      rz = weight_sqrt * _surface_sample_z(geometry.gamma)
      # _surface_sample_z (defined at boozersurface_jax.py:797) selects
      # geometry.gamma[0, 0, 2] from the 3D (nphi, ntheta, 3) array via
      # two _select_axis0 reductions. The rz-axis DERIVATIVE below is
      # 4-indexed because geometry_derivative.gamma is (nphi, ntheta, 3, ndofs).
      surface_size = optimizer_state.surface_dofs.shape[0]
      surface_gradient = (
          gradient[:surface_size]                            # residual term
          + rl * weight_sqrt * label_gradient                # label term
          + rz * weight_sqrt * geometry_derivative.gamma[0, 0, 2, :]  # rz term
      )
      ```
      Each of those last two added terms is its own product-sum, and the
      final `gradient + rl·label + rz·dgamma` itself is a 3-term sum
      (another FMA-shape candidate not in §20). P4.5b extends P4.5 by
      pinning the same canonical boundary inputs and additionally pinning the label
      geometry / coil_set_spec, then comparing
      `_boozer_penalty_value_and_grad_cpu_ordered` (JAX) against
      `boozer_penalty_constraints_vectorized` (CPU) at byte level for both
      `value` and the full `surface_gradient` vector. Pass criterion:
      `max_abs_diff == 0.0` on both outputs.
- [ ] **P4.6 — Re-run strict single_stage_init_parity gate.** After P4.5
      AND P4.5b report `max_abs_diff == 0.0` on pinned inputs, run
      `benchmarks/single_stage_init_parity.py` against the failing
      artifact's candidate end-to-end. Acceptance: parity_bug_census reports
      no divergent `boozer_solve.pre_newton_*`. Emit a new artifact under
      `.artifacts/parity/<DATE>-derivative-bit-identity-zeroing-pass/`. If
      `pre_newton_state.max_abs_diff` is still nonzero with both pinned
      tests byte-clean, the divergence has moved DOWNSTREAM of the penalty
      assembly (likely `iota_penalty.adjoint` or BFGS state representation,
      not the inner-LS gradient) — escalate to P4.7.
- [ ] **P4.7 — If gap persists: contract decision (policy, NOT a fix).**
      Only enter this branch after P4.5 AND P4.5b both report byte-clean.
      If P4.6 still shows non-zero `pre_newton_state` once both pinned
      tests pass and P4.4 confirms object-shape parity:
      * Document the residual as instruction-selection-level divergence not
        reachable from JAX-source-level changes.
      * Surface contract-level alternatives to the project owner as POLICY
        decisions, NOT root-cause closures. **Under v2 3b semantics
        (`docs/parity_dual_mode_contract_2026-05-08.md` §10) the
        production tolerance contract is locked: no `rtol=1e-N` lane
        relaxes the strict byte-identity gate. The originally-listed
        `strict_pre_newton_state` lane at `rtol=1e-12` is therefore
        REMOVED from this list.** Remaining policy alternatives:
        - Adopt noise-tolerant BFGS (Shi-Xie-Byrd-Nocedal 2020 lengthening,
          arXiv:2010.04352) — addresses BFGS amplification at algorithm
          level, not at gradient bit identity.
        - Accept the residual gap as a **build-host fingerprint** and
          quarantine the affected gate as build-pinned: pass on the
          local x86_64 build host, document explicitly that strict-mode
          bytes are valid only on this build (cross-host pinning is a
          dual-mode plan DM-E #3 slice).
      * Neither of these is a "fix" — they're acceptance-contract changes.
        They MUST be merged as explicit policy decisions, not as silent
        Phase 4 closure.

### Acceptance Gate

- [ ] P4.5 reports `max_abs_diff == 0.0` on the residual-only pinned byte
      test (first-tier arbiter — isolates the residual kernel).
- [ ] P4.5b reports `max_abs_diff == 0.0` on the full
      `_boozer_penalty_value_and_grad_cpu_ordered` pinned byte test
      (second-tier arbiter — isolates the full inner-LS gradient including
      label and rz terms).
- [ ] P4.4 shows JAX object-code FMA shape matches the C++ baseline at the
      restructured sites (necessary condition, documented per host).
- [ ] P4.6 strict gate passes OR P4.7 contract decision is documented and
      approved (only after P4.5 and P4.5b are both byte-clean).

## 11. Phase 5 — Test-Only Ablations To Prove Ownership

(See strategy doc `:432–453`.)

**Context:** prove which CPU-ordered substitution actually closes the gate;
no code is committed without this proof.

### Tasks

- [ ] Test: parity-mode surface only (Phase 2 substitution alone). Census
      should show surface arrays byte-identical, BS and downstream still
      diff. Measure `pre_newton_state` shrinkage.
- [ ] Test: parity-mode BS only (Phase 3 substitution alone). Surface still
      diffs in the live pipeline, so this ablation must feed fixed CPU
      `gamma`/`gammadash` arrays from the census directly into the BS parity
      kernel. Otherwise the surface delta cascades and the owner proof is
      invalid.
- [ ] Test: parity-mode surface + BS (Phase 2 + 3). Census all arrays
      byte-identical. If gradient still differs → Phase 4 needed.
- [ ] Test: full parity mode. Gate passes.
- [ ] Bisect any remaining residual divergence by selectively swapping the
      3-term sums in `boozer_residual_scalar_and_grad_cpu_ordered`.

### Acceptance Gate

- [ ] Each substitution's contribution to gate closure is documented.

## 12. Phase 6 — Implement The Smallest Root Fix

(See strategy doc `:454–484`.)

**Context:** based on Phase 5's ablation results, ship only the substitutions
that are necessary to close the gate.

### Tasks

- [ ] Decide which `*_cpu_ordered` modules are wired into the strict-parity
      lane based on Phase 5.
- [ ] Use the existing backend mode SSOT. Do not add
      `SIMSOPT_BOOZER_PARITY_MODE` or any second selector.
  - The route is `SIMSOPT_BACKEND_MODE=jax_cpu_parity`/`jax_gpu_parity` via
    `src/simsopt/backend/runtime.py:is_parity_mode`.
  - Propagate any needed policy through the existing
    `_make_penalty_value_and_grad_cpu_ordered_with` factory and its cache key at
    `boozersurface_jax.py:3680`.
- [ ] Production path remains untouched outside existing parity modes.
- [ ] Document the backend-policy extension in `CLAUDE.md` and in the strategy
      doc Phase 6 acceptance section.

### Acceptance Gate

- [ ] Strict-parity lane runs under `SIMSOPT_BACKEND_MODE=jax_cpu_parity` and
      gate passes; production modes stay unchanged.

## 13. Phase 7 — Regression And Release Gates

(See strategy doc `:485+`.)

### Tasks

- [ ] Full benchmark under strict parity policy:
      `parity_bug_census` reports no divergent `boozer_solve.pre_newton_*`.
- [ ] Final physics ≤ existing artifact thresholds.
- [ ] All existing unit and integration tests pass on both
      production backend modes and parity backend modes.
- [ ] New regression tests:
  - [ ] Census byte-identity of all surface and BS arrays.
  - [ ] Fixed-candidate CPU/JAX Boozer LS scalar + gradient `max_abs_diff == 0.0`.
  - [ ] `tests/test_benchmark_helpers.py:1415` continues to lock the gate.
- [ ] Update validation note:
  `.artifacts/parity/20260507-bfgs-prenewton-cpuordered-vg-m1/VALIDATION_NOTE.md`
  with a "superseded by 2026-05-07 derivative bit-identity zeroing" tag, and
  emit a new artifact under
  `.artifacts/parity/<DATE>-derivative-bit-identity-zeroing-pass/`.

### Acceptance Gate

- [ ] All Phase 7 tasks ticked.
- [ ] Final artifact `passed=true`.

## 14. Risk Register

| Risk | Mitigation |
|---|---|
| ~~`jax.lax.optimization_barrier` behavior under autodiff transforms~~ — RETRACTED 2026-05-08 (see §19.2, §21). The HLO `optimization_barrier` is erased by `OptimizationBarrierExpander` before LLVM IR is emitted; the autodiff-rule question is moot at the LLVM-fusion layer. | Risk closed. Phase 4 investigates explicit grouping as a local candidate lever (see §19.5) rather than `optimization_barrier`; production acceptance still requires x86_64 float64 object-code proof. |
| Cross-machine byte identity breaks (laptop dev vs RunPod prod) | Out of scope; document explicitly. Each host has its own per-build oracle. |
| Adding `reduction(+:val)` to C++ surface kernels in future would break per-build determinism | Add CI lint that blocks new OMP reductions in `surfacexyztensorfourier.h`, `biot_savart_impl.h`, `boozerresidual_impl.h`. |
| `jax.jacfwd` over `*_cpu_ordered` may still choose derivative arithmetic/batching order that differs from C++ analytic derivative routines | Default to analytic dgamma_by_dcoeff kernels; do not rely on jacfwd for parity twins. |
| M5 IFT adjoint and exact-Newton paths still drift | Out of scope here; track in `iota_penalty.adjoint` and exact-Newton parity ladder lanes separately. Phase 4 only addresses LS pre-Newton path. |
| `OMP_NUM_THREADS` not pinned in some test environments | Pin to 1 in the Phase 0 subprocess env and any promoted parity tests via local `monkeypatch`; do not mutate global `conftest.py` defaults unless a real shared reduction is found. |
| JAX version drift (CLAUDE.md says 0.9.2; env shows 0.10.0) | Document the version actually used; treat conda env spec as source of truth for parity claims. |

## 15. Files Touched (Summary)

### New (parity-only)

- `src/simsopt/geo/surface_fourier_jax_cpu_ordered.py`
- `src/simsopt/jax_core/biotsavart_cpu_ordered.py`
- `benchmarks/parity/__init__.py`
- `benchmarks/parity/boozer_derivative_input_census.py`
- `benchmarks/parity/boozer_derivative_input_repro.py`
- `benchmarks/parity/lane4_fma_fusion_repro_x86.py`
- `benchmarks/parity/lane5_hlo_dump_repro_x86.py`
- `tests/geo/test_boozer_derivative_input_census.py`
- `tests/geo/test_surface_fourier_jax_cpu_ordered.py`
- `tests/field/test_biotsavart_jax_cpu_ordered.py`

### Edited (private input helpers + parity-policy wiring)

- `src/simsopt/geo/boozersurface.py` — private boundary-input helper
  `_boozer_penalty_vectorized_inputs` at line 643 (consumer in
  `boozer_penalty_constraints_vectorized` at line 750).
- `src/simsopt/geo/boozersurface_jax.py` — private boundary-input helper
  `_boozer_penalty_value_and_grad_inputs_cpu_ordered` at line 1352;
  local Biot-Savart parity helper around `_field_terms_for_local_label`
  (line 1046) / `_field_terms_for_toroidal_flux` (line 1089);
  parity-policy plumbing through
  `_surface_geometry_and_derivatives_from_dofs` (line 1128) and
  `_boozer_penalty_value_and_grad_cpu_ordered` (line 1437); auto-select
  via `is_parity_mode()` in
  `_make_penalty_value_and_grad_cpu_ordered_with` factory (line 3680).
- `src/simsopt/geo/boozer_residual_jax.py` — Phase 4 restructuring
  surface in `boozer_residual_scalar_and_grad_cpu_ordered` at lines
  320–432 (≥14 candidate FMA-shape sites — see §20).

### Untouched (do not modify within this slice)

- `src/simsoptpp/**` — CPU C++/pybind oracle stays read-only.
- `benchmarks/single_stage_init_parity.py:1905` gate function.†
- `tests/test_benchmark_helpers.py:1415` gate regression test.†
- `benchmarks/validation_ladder_contract.py::PARITY_LADDER_TOLERANCES`.†
- Production JAX hot-path kernels: `surface_fourier_jax.py`,
  `jax_core/biotsavart.py`, `jax_core/field.py`, `field/biotsavart_jax_backend.py`
  — all keep matmul/einsum/jacfwd hot path.

> **† Carve-out for the dual-mode contract.** The three "†" files are
> *untouched within the bit-identity slice itself*, but the parallel
> dual-mode workstream (`docs/parity_dual_mode_contract_2026-05-08.md`
> §2.6, slices DM-A/DM-B) explicitly modifies them to add the
> `PARITY_LADDER_REPORTING_CONTEXT["pre_newton_state_empirical"]`
> reporting context and the optional `severity_context` argument on the
> gate function. That carve-out is
> narrowly scoped: gate pass/fail decisions, the strict byte-identity
> contract, and existing `PARITY_LADDER_TOLERANCES` entries are
> **not** permitted to change. If DM-A/B lands before Phase 4's gate
> work, Phase 4 inherits the augmented file shape; this plan's
> implementer simply re-runs against the post-DM-A/B state.

## 16. Today's First Checkbox (revised 2026-05-08)

> **Historical note.** The original §16 (commit `fc58b90f5`, 2026-05-07)
> directed implementers to scaffold
> `benchmarks/parity/boozer_derivative_input_repro.py`. That file plus
> `boozer_derivative_input_census.py` and the three new test files
> (`tests/geo/test_boozer_derivative_input_census.py`,
> `tests/geo/test_surface_fourier_jax_cpu_ordered.py`,
> `tests/field/test_biotsavart_jax_cpu_ordered.py`) all landed in commit
> `e61370cdf` (2026-05-08). Phases 0–3, 6, 7 are done. The first action
> is now Phase 4 entry.

Start here:

- [ ] **Phase 4 entry — P4.1**: extend
      `benchmarks/parity/boozer_derivative_input_repro.py` and/or the
      capture helpers in `benchmarks/parity/boozer_derivative_input_census.py`
      with a `--dump-arrays-as-npy <DIR>` mode (or equivalent flag) that
      writes producer snapshots plus a canonical CPU-oracle residual-input
      bundle and a `manifest.json` with role/name/sha256 integrity pairs.
      See §10 P4.1 for the full spec.

After P4.1, P4.2 (C++ object disassembly baseline) and P4.3 (restructure
JAX 3-term sums per §20) are the next steps. P4.4 verifies via post-opt
LLVM IR + `objdump`. P4.5 + P4.5b are the byte-test arbiters that gate
P4.6 / P4.7.

## 17. References

- Strategy: `docs/boozer_derivative_bit_identity_zeroing_plan_2026-05-07.md`
- In-flight slice: `docs/boozer_bfgs_pre_newton_contract_impl_plan_2026-05-07.md`
- Failing artifact: `.artifacts/parity/20260507-bfgs-prenewton-cpuordered-vg-m1/`
- Six-lane root-cause:
  `.artifacts/bit-identity-deepdive-2026-05-07/agent_{1..6}_*.md`
- Surviving lane reproducers: `/tmp/lane{2,3,6}_repro*.py` (lanes 4 and 5
  must be recreated; see Side Track).
- Gate code: `benchmarks/single_stage_init_parity.py:1905`
- Gate regression: `tests/test_benchmark_helpers.py:1415`
- CPU oracle entry: `src/simsoptpp/boozerresidual_py.cpp:11`
- JAX entry: `src/simsopt/geo/boozer_residual_jax.py:320`
- Surface CPU: `src/simsoptpp/surfacexyztensorfourier.h` (16 OMP directives,
  all on outer phi loops with disjoint output cells; per-build deterministic).
- Surface JAX matmul: `src/simsopt/geo/surface_fourier_jax.py:304`.
- Biot-Savart CPU: `src/simsoptpp/biot_savart_impl.h:62–73`.
- Biot-Savart JAX integrand: `src/simsopt/jax_core/biotsavart.py:351–359`.
- CMake flags: `CMakeLists.txt:59,61,63`.
- JAX AOT/HLO API: <https://docs.jax.dev/en/latest/aot.html> and
  <https://docs.jax.dev/en/latest/changelog.html> (`jax.xla_computation`
  deleted; use `jax.jit(fn).lower(...).compiler_ir("hlo")`).
- JAX optimization barrier:
  <https://docs.jax.dev/en/latest/_autosummary/jax.lax.optimization_barrier.html>
  (HLO-level primitive — prevents HLO-graph CSE / algebraic-simplifier
  rewriting / kernel-fusion grouping across the barrier. NOT a Phase 4
  FMA-fusion blocker here: per §19.2, `OptimizationBarrierExpander`
  erases the HLO op before LLVM IR is emitted, so the LLVM-level
  `fmul + fadd → fmadd` combine fires regardless. The autodiff-rule
  question the docstring raises is moot at this layer.)
- NVIDIA CUDA floating point/FMA:
  <https://docs.nvidia.com/cuda/archive/11.0/floating-point/index.html>
  (FMA is one rounding step; separate multiply/add is two; compiler flags or
  intrinsics control contraction).
- Official SIMSOPT Boozer residual API:
  <https://simsopt.readthedocs.io/stable/simsopt.geo.html#simsopt.geo.surfaceobjectives.boozer_surface_residual>
  (residual and derivative inputs are the surface, iota, G, and Biot-Savart
  field values).

## 18. Probe Log (2026-05-07)

> **Status 2026-05-08.** This section's recommendation to "rerun the
> relevant subset at Phase 4 entry … before adopting `optimization_barrier`
> in the residual gradient path" is RETRACTED. §19.2 establishes that
> `optimization_barrier` is erased by `OptimizationBarrierExpander` before
> LLVM IR is emitted, so the autodiff-rule question this section was
> guarding does not block FMA fusion regardless of which transform stack
> runs above the barrier. The §18 entries below remain accurate as a JAX
> 0.10.0 capability snapshot but should NOT be read as Phase 4 entry
> tasks. Phase 4 entry checklist is in §10 (revised) and §16 (revised).

Recorded against the runtime resolved by `.conda/jax-0.9.2/bin/python`.

### JAX runtime

- JAX version: `0.10.0` (env name retains the `jax-0.9.2` label; CLAUDE.md
  reference is stale on this point and should be updated when the version
  decision is finalized).
- `jax.xla_computation`: absent (deleted; AOT APIs are the replacement).
- `jax.jit(fn).lower(args).as_text("hlo")`: works.
- `jax.jit(fn).lower(args).compiler_ir("hlo")`: works.
- `jax.lax.optimization_barrier` under `jax.jit`: works at the HLO level
  (probe `f = lambda x: ob(x*2.0) + 1.0` at `x=3.0` returned `7.0`).
- `jax.lax.optimization_barrier` under `jax.vmap`: works at the HLO level
  (probe over `[1.0, 2.0, 3.0]` returned `[3.0, 5.0, 7.0]`).
- ~~`jax.lax.optimization_barrier` under `jax.grad`/`jax.jacfwd`/
  `jax.value_and_grad`: NOT PROBED — verify before adopting~~. **The
  autodiff-coverage question is RETRACTED 2026-05-08 (§19.2): the HLO
  barrier is erased by `OptimizationBarrierExpander` before LLVM IR is
  emitted, so even if the autodiff rules preserve the HLO op, it does
  not block the LLVM-level `fmul + fadd → fmadd` combine. Phase 4 must
  NOT use `optimization_barrier` as an FMA-fusion lever; see §10
  (revised) for the local candidate approach (explicit grouping +
  object-code verification).**
- `jax.lax.fma` / `jnp.fma`: absent in JAX 0.10.0. ~~`optimization_barrier`
  is one of the available levers~~ — RETRACTED, see entry above. The
  local candidate lever for Phase 4 is explicit right-nested grouping at
  the JAX expression level (§19.5), with verification via object
  disassembly (§19.6); there is no JAX-source-level FMA primitive to call
  directly.

### CPU oracle build

- C++ build flags: `-march=native -ffp-contract=fast` confirmed at
  `CMakeLists.txt:59,61,63`.
- xsimd FMA in oracle kernels: confirmed at `vec3dsimd.h:145–147`,
  `biot_savart_impl.h:71–73`, `boozerresidual_impl.h:128–148`.
- OMP usage in `surfacexyztensorfourier.h`: 16 `#pragma omp parallel for`
  directives, all on outer phi loops with disjoint output cells; no
  `reduction(+:val)`. Per-build deterministic.
- OMP usage in `boozerresidual_impl.h`: zero pragmas; single-threaded.
- OMP usage in `biot_savart_impl.h`: zero pragmas; SIMD via xsimd width.

### Backend SSOT

- `is_parity_mode()` defined at `src/simsopt/backend/runtime.py:1199`.
- `_MODE_ENV = "SIMSOPT_BACKEND_MODE"` at `runtime.py:38`.
- `VALID_BACKEND_MODES` includes `jax_cpu_parity` and `jax_gpu_parity` at
  `runtime.py:102–103`.
- `_make_penalty_value_and_grad_cpu_ordered_with` factory at
  `boozersurface_jax.py:3680` is the cache-keyed wiring point.

### Environment

- `OMP_NUM_THREADS`: not pinned globally in repo; Phase 0 must set it at the
  subprocess level for any byte-comparison run.
- `XLA_FLAGS`: not part of the baseline parity contract; treated as ablation
  surface in §5.

## 19. Phase 4 Empirical Validations (2026-05-08)

This section records the probes that retracted the original §10 levers
(`optimization_barrier`, `reduce_precision`) and identified explicit
grouping as the local candidate path. All probes ran on the local aarch64
host (Apple Silicon)
against the JAX 0.9.2 conda env at `.conda/jax-0.9.2/bin/python`. The
production target is RunPod x86_64 (A100/H100); per the Side Track in §5,
those probes are pending and may pick a different fma shape — re-run before
applying any restructuring conclusion to production.

### 19.1 `--xla_dump_llvm_ir=true` is rejected by this build

Running with `XLA_FLAGS="--xla_dump_to=/tmp/xla --xla_dump_llvm_ir=true"`
exits with:

```
F0508 06:17:32.820936 60095240 parse_flags_from_env.cc:234] Unknown flag in
XLA_FLAGS: --xla_dump_llvm_ir=true
```

The correct invocation is plain `XLA_FLAGS="--xla_dump_to=<dir>"`. The dump
directory automatically receives:

```
module_*.jit_*.before_optimizations.txt           # pre-opt HLO
module_*.jit_*.cpu_after_optimizations.txt        # post-opt HLO
module_*.jit_*.__compute_module_*.ir-no-opt.ll    # pre-opt LLVM IR
module_*.jit_*.__compute_module_*.ir-with-opt.ll  # post-opt LLVM IR
module_*.jit_*.obj-file.__compute_module_*.o      # codegen object file
module_*.jit_*.debug_options                      # XLA flag snapshot
```

No `--xla_dump_llvm_ir` flag exists in current OpenXLA. LLVM IR + object
emission is gated by the `xla_dump_to` directory only.

### 19.2 `optimization_barrier` is erased before LLVM

Source-level evidence: `OptimizationBarrierExpander` is registered in the
CPU pipeline at `xla/service/cpu/cpu_compiler.cc:943` and the pass body at
`xla/hlo/transforms/expanders/optimization_barrier_expander.cc` deletes
every `kOptimizationBarrier` HLO via `ReplaceAllUsesWith(arg)`. The pass
runs after CSE/algebraic simplification (preventing rewriting across the
HLO-level barrier) but before LLVM IR emission, so by the time LLVM sees
the IR there is no barrier left to interpose between `fmul` and `fadd`.

Empirical confirmation on JAX 0.9.2: a probe `f(a,b,c) = barrier(a*b) + c`
showed the `OptimizationBarrierOp` present in the pre-opt HLO, absent in
post-opt HLO, and the resulting LLVM IR contained an unprotected `fmul`
followed by `fadd` — both eligible for the LLVM combine.

The barrier still does what the docstring claims at the HLO level (no CSE
across it, no fusion grouping across it). **It does not block intra-kernel
mul+add → fma combine.**

### 19.3 `reduce_precision(x, 11, 52)` is a no-op for float64

OpenXLA `ReducePrecision` op semantics
(<https://openxla.org/xla/operation_semantics#reduceprecision>):

> "Setting [exponent_bits, mantissa_bits] to those of the [type] results
> in an op that is a no-op."

`(exponent_bits=11, mantissa_bits=52)` matches IEEE-754 binary64 exactly.
A standalone probe wrapping `reduce_precision(a*b, 11, 52)` on float64
produced post-optimization LLVM IR with the same plain `fmul`/`fadd` as the
unwrapped expression — no inserted bit-manipulation barrier, no rounding
boundary.

To force a round, the precision must be reduced (e.g. `(8, 23)` for float32
equivalent), which is a precision-loss operation, not a fusion blocker.
There is no JAX-level primitive in 0.9.2 that inserts an LLVM-visible
volatile-style fence at full float64 precision.

### 19.4 Active rsqrt on this branch is full-precision `1./sqrt(x)`

`src/simsoptpp/simdhelpers.h:139-149` (AVX-512 + Newton-refinement) is
inside a `/* ... */` block comment opened at line 132 and closed at
line 151 — the entire AVX-512 path is dead code. Lines 152-158 (active
SIMD path):

```cpp
inline simd_t rsqrt(const simd_t& r2){
    //On my avx2 machine, computing the sqrt and then the inverse is actually a
    //bit faster. just keeping this line here to remind myself how to compute
    //the approximate inverse square root in that case.
    //simd_t rinv = _mm256_cvtps_pd(_mm_rsqrt_ps(_mm256_cvtpd_ps(r2)));
    return 1./sqrt(r2);
}
```

The `_mm_rsqrt_ps` ps-roundtrip is commented out. Lines 163-165 (scalar
overload) call `1./std::sqrt(r2)`. The earlier hypothesis that
"single-precision rsqrt is the dominant remaining source on B/dB_dX" is
therefore **wrong on this branch** and was retracted.

### 19.5 Explicit grouping reproduces C++ FMA nesting on aarch64

Probe: `f(a,b,c,d,e,f) = a*b + (c*d + e*f)` with `XLA_FLAGS=--xla_dump_to`.

Post-optimization LLVM IR (`*.ir-with-opt.ll`) — no FMA intrinsic emitted:

```llvm
%23 = fmul float %21, %22        ; a*b
%26 = fmul float %24, %25        ; c*d
%29 = fmul float %27, %28        ; e*f
%30 = fadd float %26, %29        ; c*d + e*f
%31 = fadd float %23, %30        ; a*b + (c*d + e*f)
```

Object disassembly (`objdump -d *.o`) — codegen welds pairs into `fmadd`:

```
50: 1e250884   fmul   s4, s4, s5         ; e*f
54: 1f031042   fmadd  s2, s2, s3, s4     ; c*d + (e*f)    ← fused
58: 1f010800   fmadd  s0, s0, s1, s2     ; a*b + tail     ← fused
```

This shape — `fmul; fmadd; fmadd` with the trailing free multiply at the
deepest position — is exactly what C++
`xsimd::fma(a, b, xsimd::fma(c, d, e*f))` produces at
`boozerresidual_impl.h:128`. **Explicit grouping is therefore an empirically
supported local candidate lever, not a guess.** It is not production-accepted
until verified as float64 on the x86_64 production target.

The example was on float32 (Python literals defaulted to f32 in the probe);
the structural conclusion holds for f64 because the same SelectionDAG /
MachineCombiner passes apply. Verify on x86_64 via the Side Track §5 before
locking the restructuring shape for production.

### 19.6 Methodology takeaways for future Phase 4 rounds

* **LLVM IR alone is insufficient** to detect FMA fusion. Aarch64 codegen
  (and likely x86_64) welds `fmul + fadd → fmadd` *after* LLVM IR
  optimization, without ever materializing `@llvm.fmuladd`. Always check
  the disassembled object file.
* **HLO/LLVM IR is the right layer** for *association/grouping* (which
  mul-add pair is bound together) but not for *fusion emission* (whether
  the pair becomes a single fma machine instruction).
* **`--xla_dump_to` alone gives you everything**: HLO before/after, LLVM IR
  before/after, the object file, and the runtime debug-options snapshot.
  No additional flags required for the probe.
* **The C++ explicit `xsimd::fma`/`fms` calls force fusion on the C++ side
  unconditionally.** The JAX side reaches the same shape iff the source
  expression's grouping matches AND the SelectionDAG cost model picks the
  same operand as the FMA multiplier. Both must be verified per host.

## 20. Full Phase 4 Ablation Surface

The original Phase 4 wording named "three 3-term sums" as restructure
targets. Direct read of `src/simsopt/geo/boozer_residual_jax.py:354-431`
plus cross-check against `boozerresidual_impl.h:108-160` shows ≥14
candidate FMA-shape sites in the JAX residual gradient kernel. These are
the *first targets*, not exhaustive — the boundary-pinned byte test (P4.5)
is the ground truth.

| # | JAX line(s) | JAX expression | C++ counterpart | Family |
|---|---|---|---|---|
| 1 | 382-384 | `dB0 = dB_dX[i,j,0,0]*dx0 + dB_dX[i,j,1,0]*dx1 + dB_dX[i,j,2,0]*dx2` | `boozerresidual_impl.h:128` `dBij0m = xsimd::fma(a,b,xsimd::fma(c,d,e*f))` | 3-term right-nested fma |
| 2 | 385-387 | `dB1` analogous | `:129` | 3-term right-nested fma |
| 3 | 388-390 | `dB2_component` analogous | `:130` | 3-term right-nested fma |
| 4 | 391-393 | `dB2 = 2.0 * (B0*dB0 + B1*dB1 + B2_component*dB2_component)` | `:132` `dB2_ijm = 2*(Bij0*dBij0m + Bij1*dBij1m + Bij2*dBij2m)` | 3-term sum × scalar prefactor |
| 5 | 395 | `dtang0 = iota * dxtheta_ds[i,j,0,:] + dxphi_ds[i,j,0,:]` | `:133` `tang_ij0m = xsimd::fma(it, dxtheta_ds_ij0m, dxphi_ds_ij0m)` | 2-term fma |
| 6 | 396 | `dtang1` analogous | `:134` | 2-term fma |
| 7 | 397 | `dtang2` analogous | `:135` | 2-term fma |
| 8 | 399 | `dres0 = G * dB0 - (dB2 * tang0 + B2 * dtang0)` | `:137` `dresij0m = xsimd::fms(GG, dBij0m, xsimd::fma(dB2_ijm, btang_ij0, B2ij*tang_ij0m))` | fms wrapping a 3-term fma |
| 9 | 400 | `dres1` analogous | `:138` | fms + fma |
| 10 | 401 | `dres2` analogous | `:139` | fms + fma |
| 11 | 408-410 | `drtil0 = dres0*wij + dw*res0` (×3 for drtil0/1/2) | `:143-145` `drtil_ij0m = xsimd::fma(dresij0m, bw_ij, dw_ijm*resij0)` | 2-term fma + free mul |
| 12 | 411 | `surface_grad = rtil0*drtil0 + rtil1*drtil1 + rtil2*drtil2` | `:148` `dresm = xsimd::fma(brtil_ij0, drtil_ij0m, xsimd::fma(brtil_ij1, drtil_ij1m, brtil_ij2*drtil_ij2m))` | 3-term right-nested fma |
| 13 | 417-421 | `iota_grad = rtil0*dres0_iota*wij + rtil1*dres1_iota*wij + rtil2*dres2_iota*wij` | `boozerresidual_impl.h:176-183` `dres(ndofs+0) += rtil_ij0*drtil_ij0iota + rtil_ij1*drtil_ij1iota + rtil_ij2*drtil_ij2iota` | 3-term scalar sum (C++ uses `double`, no SIMD `xsimd::fma` nest) |
| 14 | 425 | `G_grad = rtil0*wij*B0 + rtil1*wij*B1 + rtil2*wij*B2_component` | `boozerresidual_impl.h:190-197` `dres(ndofs+1) += rtil_ij0*drtil_ij0_dG + rtil_ij1*drtil_ij1_dG + rtil_ij2*drtil_ij2_dG` | 3-term scalar sum (C++ uses `double`, no SIMD `xsimd::fma` nest) |

Sites 13 and 14 DO have C++ counterparts in
`boozerresidual_impl.h:176-183` (iota) and `:190-197` (G), but the C++
side accumulates them in **scalar `double`** rather than in `xsimd::fma`
nests — the iota/G columns sit outside the SIMD-vectorized inner loop.
Because `-ffp-contract=fast` still permits scalar `mul + add → vfmadd*sd`
codegen, the C++ side likely *does* fuse these scalars into FMAs, but the
exact shape (right-nested vs left-nested) is compiler-controlled rather
than source-pinned. P4.4 disassembly must verify per host. The byte test
in P4.5 / P4.5b remains the arbiter for these sites because the source
expression alone does not determine the FMA shape.

### Ablation discipline

1. Patch ONE site at a time.
2. Run P4.5 (boundary-pinned byte test) before and after.
3. If P4.5 `max_abs_diff` decreases, keep the patch and proceed to the next
   candidate site.
4. If P4.5 `max_abs_diff` is unchanged or worse, revert and try a different
   site or a different grouping shape.
5. Sites 1-3 (the dB triple) are the most likely starting point because
   they have a direct C++ counterpart with explicit `xsimd::fma` and
   contribute most directly to the gradient first-step drift observed in
   the failing artifact.

### Watch points outside the explicit-FMA pattern

Even with all 14 sites restructured to match the C++ shape, residual drift
may persist from:

* **SIMD lane width.** xsimd reduces over `simd_t` lanes (4 or 8 doubles
  per register on AVX2/AVX-512); JAX `lax.fori_loop` is scalar. The
  per-element FMA shape can match while the lane-level reduction tree
  outside the 3-term arithmetic differs.
* **`fms` lowering.** C++ `xsimd::fms(a, b, c) = a*b - c` lowers to
  `vfnmadd*sd` (negated FMA) on x86. JAX's plain subtraction `a*b - c`
  may or may not select the same negated FMA at codegen — verify in P4.4.
* **The trailing free multiply.** `e*f` inside `xsimd::fma(c, d, e*f)` is
  not fma-protected. A sibling expression in scope that touches `e` or `f`
  can pull `e*f` into a different fusion candidate. JAX's grouping pins
  the *shape* but not the *operand binding*.
* **`2*(...)` triple-sum (site 4).** The scalar prefactor 2 may lower to
  a multiply or get folded into the inner FMA depending on the compiler.
  Worth checking whether the C++ form `2*(a*b + c*d + e*f)` matches the
  JAX form to within a single rounding.

## 21. Phase 4 Stale-Wording Cleanup (2026-05-08)

The prior §10 referenced `optimization_barrier` and `reduce_precision` as
candidate levers and directed implementers to "probe `optimization_barrier`
under the exact transform stack" before proceeding to grouping. Both of
those steps have now been retracted:

* The `optimization_barrier` autodiff-rule question is moot: the barrier
  is erased before LLVM regardless of which transform stack runs above it
  (§19.2).
* `reduce_precision(11, 52)` is a no-op (§19.3); the precision-reducing
  variant (`reduce_precision(x, 8, 23)`) is a precision-loss operation,
  not a fusion blocker.
* `--xla_dump_llvm_ir=true` is not a valid flag (§19.1); replace any
  occurrence with plain `--xla_dump_to=<dir>`.

Implementers picking up Phase 4 from this doc should treat §10 (revised),
§19, and §20 as authoritative and ignore the older wording in the strategy
doc `docs/boozer_derivative_bit_identity_zeroing_plan_2026-05-07.md` until
that doc is also revised.
