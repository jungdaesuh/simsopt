# Boozer Derivative Bit-Identity Zeroing — Detailed Implementation Plan

- Date: 2026-05-07
- Branch: `gpu-purity-stage2-20260405`
- Status: not-started
- Companion strategy doc: `docs/boozer_derivative_bit_identity_zeroing_plan_2026-05-07.md`
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
     (`boozer_residual_jax.py:324–445`) were observed by Lane 4 to lower to
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
  - Honors `--dump-arrays <DIR>` to write raw float64 arrays as `.npy`
    plus per-array metadata JSON.
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
`src/simsopt/geo/boozer_residual_jax.py:324`).

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
    before `_call_boozer_residual_ds` at `boozersurface.py:720`.
  - `boozer_penalty_constraints_vectorized(...)` calls that helper, and the
    benchmark census calls the same helper. Do not add a public `_capture_cb`
    kwarg or callback seam to `BoozerSurface.boozer_penalty_constraints_vectorized`.
- [ ] Implement JAX capture helper.
  - Function: `capture_jax_boozer_inputs(spec, candidate, *, weight_inv_modB)`.
  - Factor a private `_boozer_penalty_value_and_grad_inputs_cpu_ordered(...)`
    helper in `src/simsopt/geo/boozersurface_jax.py` for the arrays currently
    built before the `boozer_residual_scalar_and_grad_cpu_ordered` call at line
    1197.
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
    at `src/simsopt/geo/boozersurface_jax.py:1043`, derived from
    `is_parity_mode()` at the caller boundary rather than a new env var.
  - In parity policy, dispatch to `surface_*_cpu_ordered` instead of
    `_surface_geometry_from_dofs` + `jax.jacfwd`.
  - Plumb the policy through `_boozer_penalty_value_and_grad_cpu_ordered` at
    `boozersurface_jax.py:1143` and include it in any closure cache key it
    changes.
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
      (`src/simsopt/geo/boozersurface_jax.py:980–1033`) by importing the new
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

## 10. Phase 4 — Residual Derivative Assembly Identity

(See strategy doc `:381–431`.)

**Context:** if all raw arrays are byte-identical but the gradient still
differs, the bug is in residual derivative assembly. The current CPU-ordered
JAX kernel is `boozer_residual_scalar_and_grad_cpu_ordered` at
`src/simsopt/geo/boozer_residual_jax.py:324–445`. Lane 4 identified candidate
3-term FMA-fusion mismatches at lines 384–401 (`dB0`, `dB1`, `dB2`), 419–421
(`surface_grad`), 427–438 (`iota_grad`/`G_grad`).

### Tasks

- [ ] Confirm Phase 1 census post-Phase-2/3: are raw inputs byte-identical?
- [ ] If gradient still differs:
  - [ ] Probe `jax.lax.optimization_barrier` as a candidate (not assumed) tool.
        Official JAX docs say it prevents compiler fusion and movement across
        the barrier, and that it has no effect outside a compiled function. The
        docs also state it has no derivative or batching rules.
        Local JAX 0.10.0 probes recorded 2026-05-07 (see §18) verify
        `optimization_barrier` under `jax.jit` and `jax.vmap` only; `jax.grad`,
        `jax.jacfwd`, and `jax.value_and_grad` were NOT probed. Phase 4 entry
        must rerun the probe under the exact transform stack used by
        `_boozer_penalty_value_and_grad_cpu_ordered` (`jax.jacfwd` for surface
        geometry derivatives at
        `_surface_geometry_and_derivatives_from_dofs:1069`, plus
        `jax.value_and_grad` for the label term at
        `boozersurface_jax.py:1211`) before placing a barrier in the residual
        gradient path. Whether the barrier blocks intra-kernel FMA contraction
        at LLVM codegen is also empirical; verify with a small reproducer
        first.
  - [ ] If `optimization_barrier` does not block FMA contraction, restructure
        each 3-term sum to match xsimd's nesting shape
        `fma(a, b, fma(c, d, e*f))`. Two pieces are needed and grouping only
        fixes the first:
        - **Associativity (grouping):** force the reduction tree so the inner
          sum binds `c*d + e*f` and the outer binds `a*b + tail`.
          ```python
          # original free-form (XLA chooses fusion per host; Lane 4 observed
          # p3 = fma(e, f, fma(a, b, c*d)) on aarch64):
          # a*b + c*d + e*f
          #
          # parity-twin target shape: fma(a, b, fma(c, d, e*f))
          tail = (c * d) + (e * f)
          result = (a * b) + tail
          ```
        - **FMA factor selection:** even after grouping, XLA still chooses
          which operand becomes the FMA's multiplier. `c*d + e*f` may lower
          to `fma(c, d, e*f)` (target) or `fma(e, f, c*d)` (off-target); the
          same ambiguity applies to `(a*b) + tail`. After grouping, dump
          optimized HLO via `jit().lower(*).compiler_ir("hlo")` and inspect
          backend IR/asm where available to verify the lowering. If XLA
          picks an off-target factor, wrap the binding side in
          `optimization_barrier` (subject to the jacfwd/value_and_grad probe
          above) or fall back to a manual fused multiply-add helper if a
          future JAX version exposes one.
  - [ ] Audit `_boozer_penalty_value_and_grad_cpu_ordered`
        (`boozersurface_jax.py:1143–1242`) for label-value/gradient and z-axis
        pinning 3-term sums; restructure analogously.
- [ ] If x86_64 FMA reproducer (Side Track) shows different fusion shape, pick
      the matching restructuring.

### Acceptance Gate

- [ ] Fixed-candidate CPU/JAX Boozer LS scalar + gradient: `max_abs_diff == 0.0`.

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
    `boozersurface_jax.py:3390`.
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
| `jax.lax.optimization_barrier` behavior under the exact transform stack or backend FMA contraction differs from local probes | Probe the exact `jax.jacfwd`/`jax.value_and_grad` usage and backend codegen before adopting it; then use explicit parenthesization if required. Lane 4/5 x86_64 reproducers (Side Track) inform this. |
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

- `src/simsopt/geo/boozersurface.py` — private boundary-input helper at line
  720 area.
- `src/simsopt/geo/boozersurface_jax.py` — private boundary-input helper at
  line 1190 area; local Biot-Savart parity helper around
  `_field_terms_for_local_label`/`_field_terms_for_toroidal_flux`; parity-policy
  plumbing through
  `_surface_geometry_and_derivatives_from_dofs` (line 1043) and
  `_boozer_penalty_value_and_grad_cpu_ordered` (line 1143).
- `src/simsopt/geo/boozer_residual_jax.py` — possible Phase 4 restructuring at
  lines 384–445.

### Untouched (do not modify)

- `src/simsoptpp/**` — CPU C++/pybind oracle stays read-only.
- `benchmarks/single_stage_init_parity.py:1905` gate function.
- `tests/test_benchmark_helpers.py:1415` gate regression test.
- `benchmarks/validation_ladder_contract.py::PARITY_LADDER_TOLERANCES`.
- Production JAX hot-path kernels: `surface_fourier_jax.py`,
  `jax_core/biotsavart.py`, `jax_core/field.py`, `field/biotsavart_jax_backend.py`
  — all keep matmul/einsum/jacfwd hot path.

## 16. Today's First Checkbox

Start here:

- [ ] **Phase 0 task 1**: scaffold
      `benchmarks/parity/boozer_derivative_input_repro.py` skeleton with the
      env-capture block, `argparse` for `--candidate-source` and
      `--dump-arrays`, and a placeholder `main()` that prints the resolved
      candidate ID.

After that's done, Phase 0 task 2 (extract candidate from existing artifact),
then Phase 1 task 1 (census schema dataclass), in that order.

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
- JAX entry: `src/simsopt/geo/boozer_residual_jax.py:324`
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
  (prevents compiler fusions; no derivative or batching rules).
- NVIDIA CUDA floating point/FMA:
  <https://docs.nvidia.com/cuda/archive/11.0/floating-point/index.html>
  (FMA is one rounding step; separate multiply/add is two; compiler flags or
  intrinsics control contraction).
- Official SIMSOPT Boozer residual API:
  <https://simsopt.readthedocs.io/stable/simsopt.geo.html#simsopt.geo.surfaceobjectives.boozer_surface_residual>
  (residual and derivative inputs are the surface, iota, G, and Biot-Savart
  field values).

## 18. Probe Log (2026-05-07)

Recorded against the runtime resolved by `.conda/jax-0.9.2/bin/python`. Re-run
the relevant subset at Phase 4 entry under the exact transform stack before
adopting `optimization_barrier` in the residual gradient path.

### JAX runtime

- JAX version: `0.10.0` (env name retains the `jax-0.9.2` label; CLAUDE.md
  reference is stale on this point and should be updated when the version
  decision is finalized).
- `jax.xla_computation`: absent (deleted; AOT APIs are the replacement).
- `jax.jit(fn).lower(args).as_text("hlo")`: works.
- `jax.jit(fn).lower(args).compiler_ir("hlo")`: works.
- `jax.lax.optimization_barrier` under `jax.jit`: works
  (probe `f = lambda x: ob(x*2.0) + 1.0` at `x=3.0` returned `7.0`).
- `jax.lax.optimization_barrier` under `jax.vmap`: works
  (probe over `[1.0, 2.0, 3.0]` returned `[3.0, 5.0, 7.0]`).
- `jax.lax.optimization_barrier` under `jax.grad`: NOT PROBED — verify before
  adopting in any gradient path.
- `jax.lax.optimization_barrier` under `jax.jacfwd`: NOT PROBED — verify before
  Phase 4 inserts the barrier under
  `_surface_geometry_and_derivatives_from_dofs`.
- `jax.lax.optimization_barrier` under `jax.value_and_grad`: NOT PROBED —
  verify before Phase 4 inserts the barrier in the label-gradient path at
  `boozersurface_jax.py:1211`.
- `jax.lax.fma` / `jnp.fma`: absent. Phase 4 cannot rely on a primitive
  `fma`; restructuring or `optimization_barrier` are the available levers.

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
  `boozersurface_jax.py:3390` is the cache-keyed wiring point.

### Environment

- `OMP_NUM_THREADS`: not pinned globally in repo; Phase 0 must set it at the
  subprocess level for any byte-comparison run.
- `XLA_FLAGS`: not part of the baseline parity contract; treated as ablation
  surface in §5.
