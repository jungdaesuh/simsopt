# Boozer Derivative Bit-Identity Zeroing Plan - 2026-05-07

## Purpose

This plan scopes the deeper follow-up to the remaining strict CPU C++/pybind
versus JAX CPU mismatch:

> The root cause is localized, but not fully eliminated: the remaining strict
> mismatch is most likely non-bit-identical derivative input generation between
> CPU C++/pybind paths and JAX paths, especially surface derivatives and
> Biot-Savart derivatives. The JAX path uses `jax.jacfwd` / `jax.linearize`;
> CPU uses the upstream C++/pybind derivative routines. Matching those
> bit-for-bit is a deeper derivative-bit-identity project, not a small Boozer
> BFGS wiring fix.

The goal is to turn that statement from a strong diagnosis into a closed result:
either zero the remaining strict mismatch, or prove exactly which derivative
input cannot be made byte-identical without replacing one side's numerical
algorithm.

## Current Evidence

The post-fix parity artifact
`.artifacts/parity/20260507-bfgs-prenewton-postfix-m1/result.json` shows the
larger behavioral drift is closed, but the strict census is not fully zeroed.

- Same-candidate replay:
  - `status`: `pass`
  - `max_candidate_abs_diff`: `0.0`
  - `max_objective_abs_diff`: `2.220446049250313e-16`
  - `max_optimizer_gradient_abs_diff`: `8.3322237998118e-13`
- Final physics drift:
  - `final_iota_abs_diff`: `4.575333167888829e-16`
  - `final_volume_rel_diff`: `1.3892094770373822e-15`
  - `field_error_rel_diff`: `5.964756209488326e-15`
- Remaining strict census divergences:
  - `boozer_solve.pre_newton_state`: `4.519706948979962e-09`
  - `boozer_solve.pre_newton_objective_gradient`: `1.5830200208379184e-10`
  - `iota_penalty.adjoint`: `2.318643055332359e-10`

Interpretation: the objective wiring and candidate replay are no longer the main
issue. The first material strict divergence is the inner Boozer pre-Newton solve
state. The iota adjoint drift is downstream unless a raw-array trace disproves
that.

## Code Map

### CPU C++/pybind path

- `src/simsopt/geo/boozersurface.py:641` defines
  `BoozerSurface.boozer_penalty_constraints_vectorized(...)`, the fast CPU
  scalarized penalty path used by the legacy LS solve.
- `src/simsopt/geo/boozersurface.py:691` sets CPU surface dofs on the mutable
  surface object before evaluating geometry.
- `src/simsopt/geo/boozersurface.py:695` to `src/simsopt/geo/boozersurface.py:710`
  read CPU surface geometry and derivative arrays:
  `gamma`, `gammadash1`, `gammadash2`, `dgamma_by_dcoeff`,
  `dgammadash1_by_dcoeff`, and `dgammadash2_by_dcoeff`.
- `src/simsopt/geo/boozersurface.py:701` to `src/simsopt/geo/boozersurface.py:710`
  set Biot-Savart points, call `compute(derivatives)`, and read `B` plus
  `dB_by_dX`.
- `src/simsopt/geo/boozersurface.py:720` dispatches first derivatives through
  `_call_boozer_residual_ds(...)`.
- `src/simsopt/geo/boozersurface.py:117` to
  `src/simsopt/geo/boozersurface.py:161` call `sopp.boozer_residual_ds`.
- `src/simsopt/field/biotsavart.py:10` subclasses `sopp.BiotSavart`.
- `src/simsopt/field/biotsavart.py:60` to
  `src/simsopt/field/biotsavart.py:119` project CPU Biot-Savart VJPs through
  `sopp.biot_savart_vjp_graph`.
- `src/simsopt/geo/surfaceobjectives.py:49` dispatches
  `sopp.boozer_dresidual_dc` for the CPU residual-Jacobian path consumed by
  outer `BoozerResidual` objectives.
- `src/simsopt/geo/surfaceobjectives.py:586` to
  `src/simsopt/geo/surfaceobjectives.py:624` build CPU Boozer residual
  derivative inputs from surface geometry, surface coefficient derivatives,
  `B`, `dB_by_dX`, and `dB_dc`.
- `src/simsopt/geo/surfaceobjectives.py:1375` to
  `src/simsopt/geo/surfaceobjectives.py:1417` consume the residual Jacobian,
  project `dJ_by_dB` through `BiotSavart.B_vjp(...)`, solve the Boozer adjoint,
  and form the downstream coil derivative.

### JAX CPU path

- `src/simsopt/geo/boozersurface_jax.py:3390` builds the host-SciPy
  value/gradient parity closure via
  `_make_penalty_value_and_grad_cpu_ordered_with(...)`.
- `src/simsopt/geo/boozersurface_jax.py:1143` implements
  `_boozer_penalty_value_and_grad_cpu_ordered(...)`.
- `src/simsopt/geo/boozersurface_jax.py:1043` to
  `src/simsopt/geo/boozersurface_jax.py:1073` evaluate JAX surface geometry and
  compute `dgamma`, `dxphi`, and `dxtheta` with `jax.jacfwd`.
- `src/simsopt/geo/boozersurface_jax.py:1190` to
  `src/simsopt/geo/boozersurface_jax.py:1209` feed JAX `B`, `dB_dX`, surface
  geometry, and surface derivative arrays into
  `boozer_residual_scalar_and_grad_cpu_ordered(...)`.
- `src/simsopt/geo/boozer_residual_jax.py:324` to
  `src/simsopt/geo/boozer_residual_jax.py:445` hand-accumulate the
  CPU-ordered Boozer scalar and first derivative in JAX.
- `src/simsopt/geo/boozer_residual_jax.py:485` to
  `src/simsopt/geo/boozer_residual_jax.py:541` are the pure JAX surface
  geometry SSOT for Boozer.
- `src/simsopt/jax_core/biotsavart.py:451` to
  `src/simsopt/jax_core/biotsavart.py:486` compute Biot-Savart spatial
  derivatives with `jax.jacfwd` and combined `B, dB` with `jax.linearize`.
- `src/simsopt/jax_core/field.py:509` to
  `src/simsopt/jax_core/field.py:514` accumulate grouped `B` and `dB_dX`.
- `src/simsopt/field/biotsavart_jax_backend.py:1465` to
  `src/simsopt/field/biotsavart_jax_backend.py:1485` expose JAX `dB_by_dX()` and
  `B_and_dB()`.
- `src/simsopt/geo/boozersurface_jax.py:1801` to
  `src/simsopt/geo/boozersurface_jax.py:1859` implement exact-solve coil VJPs
  with `jax.vjp`.
- `src/simsopt/geo/boozersurface_jax.py:1935` to
  `src/simsopt/geo/boozersurface_jax.py:1972` implement LS-path coil VJPs with
  `jax.grad`.
- `src/simsopt/geo/boozersurface_jax.py:1991` to
  `src/simsopt/geo/boozersurface_jax.py:2025` begin the grouped LS VJP path that
  streams field-term cotangents group by group.

### JAX API facts that matter

Context7 `/google/jax` docs fetched on 2026-05-07 confirm:

- `jax.jacfwd` computes Jacobians by forward-mode automatic differentiation.
- `jax.linearize` returns a primal output plus a reusable linearized function at
  the input point.
- JAX defaults to 32-bit unless `jax_enable_x64` is set at startup.

These APIs provide mathematically valid derivatives, but they do not promise the
same floating-point operation order, reduction order, memory layout, or byte
identity as upstream C++ derivative kernels.

## Definition Of Zero

For this project, "zeroed" means all of the following are true on the CPU parity
lane:

- The same outer candidate reaches the same inner Boozer pre-Newton decision
  vector bytes before SciPy sees the value/gradient callback.
- For every traced derivative input array, CPU and JAX materialize to the same
  `np.float64` shape, C/F memory-order contract, and raw byte digest.
- The fixed-candidate CPU and JAX Boozer LS scalar and gradient have
  `max_abs_diff == 0.0`.
- The strict `parity_bug_census` has no divergent `boozer_solve.pre_newton_*`
  layer.
- The existing same-candidate replay and final physics gates remain at least as
  good as the current post-fix artifact.

GPU bit identity is not part of this gate. GPU remains a separate CPU-oracle
parity lane because CPU/GPU execution order and compiler lowering are not a
byte-identity contract.

## Non-Goals

- Do not loosen tolerances to hide the remaining mismatch.
- Do not add production fallbacks, defensive retries, or route switches.
- Do not replace upstream CPU C++/pybind code.
- Do not claim success from JAX-vs-JAX agreement.
- Do not make SciPy optimizer status/counter differences the root cause until
  fixed-candidate raw derivative inputs have been proven byte-identical.

## Requirements

### Functional Requirements

- [ ] Preserve the trust chain: SIMSOPT CPU C++/SciPy -> JAX CPU -> JAX GPU.
- [ ] Compare CPU and JAX at the same outer candidate and the same inner Boozer
      decision vector.
- [ ] Compare raw derivative inputs before residual scalar/gradient contraction:
  - [ ] `surface_dofs`
  - [ ] `iota`
  - [ ] `G`
  - [ ] `gamma`
  - [ ] `gammadash1`
  - [ ] `gammadash2`
  - [ ] `dgamma_by_dcoeff`
  - [ ] `dgammadash1_by_dcoeff`
  - [ ] `dgammadash2_by_dcoeff`
  - [ ] Biot-Savart evaluation points
  - [ ] `B`
  - [ ] `dB_by_dX`
  - [ ] `d2B_by_dXdX` for Hessian/exact follow-up checks
  - [ ] label value and label gradient
  - [ ] z-axis pinning gradient
  - [ ] residual Jacobian `J`
  - [ ] residual Hessian `H` where requested
  - [ ] masks and stellsym scatter indices
  - [ ] normalization factors
  - [ ] `G` construction from currents when `optimize_G=False`
  - [ ] coil `gamma`, `gammadash`, and current inputs to VJP paths
- [ ] Record for each compared array:
  - [ ] producing function
  - [ ] dtype
  - [ ] shape
  - [ ] strides
  - [ ] C/F contiguity
  - [ ] raw `np.float64` SHA-256 digest
  - [ ] `max_abs_diff`
  - [ ] `argmax_abs_diff`
  - [ ] first unequal raw-byte index
  - [ ] first unequal numeric index
- [ ] Prove whether the first mismatch enters through surface geometry,
      surface coefficient derivatives, Biot-Savart field derivatives, label
      derivatives, or residual derivative assembly.
- [ ] Keep all instrumentation test-only or benchmark-only unless the final
      fix needs a production arithmetic change.

### Numerical Requirements

- [ ] Run with `JAX_ENABLE_X64=True`.
- [ ] Force CPU parity runs through `JAX_PLATFORMS=cpu`.
- [ ] Disable accidental GPU execution for the zeroing gate.
- [ ] Materialize JAX arrays at the same host boundary used by SciPy before
      byte comparison.
- [ ] Avoid mixed `float64`/Python-float comparisons in the trace artifacts.
- [ ] Compare values before and after reshapes used by both paths.
- [ ] Compare arrays in the exact flattened order consumed by
      `sopp.boozer_residual_ds` and by
      `boozer_residual_scalar_and_grad_cpu_ordered(...)`.

### Architecture Requirements

- [ ] Single source of truth for trace schema.
- [ ] No production fallback from JAX to CPU derivative routines.
- [ ] No duplicate derivative formulas beyond a temporary test harness used for
      isolation.
- [ ] Keep CPU oracle code read-only unless a pybind test hook is absolutely
      required to expose an intermediate.
- [ ] Keep the target-lane JAX product path separate from CPU parity
      instrumentation.

## Implementation Plan

### Phase 0 - Freeze The Oracle Artifact

Context: the current artifact is good enough to define the starting failure, but
the next phase needs a smaller fixed-state reproducer.

- [ ] Preserve the current post-fix artifact path and summary in the new report.
- [ ] Add a short fixed-state reproducer command that runs only one candidate and
      dumps raw Boozer LS inputs.
- [ ] Pin environment variables in the command:
  - [ ] `JAX_ENABLE_X64=True`
  - [ ] `JAX_PLATFORMS=cpu`
  - [ ] existing parity/strict flags used by `benchmarks/single_stage_init_parity.py`
- [ ] Confirm the reproducer still reports:
  - [ ] same-candidate replay pass
  - [ ] first strict divergence at `boozer_solve.pre_newton_state`
  - [ ] no new unrelated divergent family before Boozer solve

Acceptance gate:

- [ ] A minimal artifact reproduces the same first-divergence family/layer as
      `.artifacts/parity/20260507-bfgs-prenewton-postfix-m1/result.json`.

### Phase 1 - Add A Fixed-Candidate Derivative Input Census

Context: comparing the final scalar and gradient is too late. The trace must
compare the derivative inputs before the Boozer residual derivative code
contracts them.

Files likely involved:

- `benchmarks/single_stage_init_parity.py`
- `tests/geo/test_boozersurface_jax.py`
- `tests/geo/boozersurface_jax_test_helpers.py`
- `src/simsopt/geo/boozersurface_jax.py` only if a narrow test seam is needed

Todos:

- [ ] Define `boozer_derivative_input_census` as the trace schema.
- [ ] Implement a test helper that captures CPU raw arrays from
      `BoozerSurface.boozer_penalty_constraints_vectorized(..., derivatives=1)`.
- [ ] Implement a matching JAX helper that captures arrays from
      `_boozer_penalty_value_and_grad_cpu_ordered(...)` before calling
      `boozer_residual_scalar_and_grad_cpu_ordered(...)`.
- [ ] Emit one JSON object per array with producer, dtype, shape, strides,
      digest, norms, and first-diff metadata.
- [ ] Add a red test that names the first mismatching array instead of only
      reporting aggregate gradient drift.

Acceptance gate:

- [ ] The report identifies the first non-byte-identical derivative input array
      by name and index.

### Phase 2 - Surface Geometry And Surface Derivative Bit Identity

Context: the CPU path reads surface derivatives from C++/pybind-backed surface
methods. The JAX path computes geometry through pure surface spec functions and
then uses `jax.jacfwd` for derivative arrays.

CPU producers:

- `surface.gamma()`
- `surface.gammadash1()`
- `surface.gammadash2()`
- `surface.dgamma_by_dcoeff()`
- `surface.dgammadash1_by_dcoeff()`
- `surface.dgammadash2_by_dcoeff()`

JAX producers:

- `_surface_geometry_from_dofs(...)`
- `_surface_geometry_and_derivatives_from_dofs(...)`
- `jax.jacfwd(geometry_arrays)(surface_dofs)`

Todos:

- [ ] Build direct fixed-dof tests for each supported Boozer surface kind:
  - [ ] `SurfaceRZFourier`
  - [ ] `SurfaceXYZFourier`
  - [ ] `SurfaceXYZTensorFourier`
- [ ] Compare geometry values byte-for-byte:
  - [ ] `gamma`
  - [ ] `gammadash1`
  - [ ] `gammadash2`
- [ ] Compare coefficient derivatives byte-for-byte:
  - [ ] `dgamma_by_dcoeff`
  - [ ] `dgammadash1_by_dcoeff`
  - [ ] `dgammadash2_by_dcoeff`
- [ ] If geometry values match but derivatives do not, compare manual analytic
      JAX derivative kernels against `jax.jacfwd`.
- [ ] If both JAX derivative routes differ from CPU, trace CPU surface derivative
      formula order in `src/simsoptpp/surface.cpp` and pybind registration.
- [ ] Decide whether to implement explicit CPU-ordered JAX surface derivative
      kernels for the Boozer parity path.

Acceptance gate:

- [ ] Surface derivative inputs are either byte-identical, or the exact first
      array and arithmetic-order reason are documented.

### Phase 3 - Biot-Savart Field Derivative Bit Identity

Context: even if surface derivatives match, `dB_by_dX` can differ because the CPU
path uses `sopp.BiotSavart.compute(derivatives)` while JAX uses grouped pure JAX
Biot-Savart kernels, `jax.jacfwd`, and `jax.linearize`.

CPU producers:

- `biotsavart.set_points(...)`
- `biotsavart.compute(derivatives)`
- `biotsavart.B()`
- `biotsavart.dB_by_dX()`

JAX producers:

- `grouped_biot_savart_B_and_dB_from_spec(...)`
- `biot_savart_B_and_dB(...)`
- `jax.linearize(...)`
- `jax.jacfwd(one_point, argnums=0)(...)`

Todos:

- [ ] Compare Biot-Savart input bundles before field evaluation:
  - [ ] point array bytes
  - [ ] coil `gamma` bytes
  - [ ] coil `gammadash` bytes
  - [ ] current bytes
  - [ ] grouped coil order
  - [ ] quadrature point order
- [ ] Compare field outputs:
  - [ ] `B`
  - [ ] `dB_by_dX`
- [ ] Compare direct `biot_savart_dB_by_dX(...)` against combined
      `biot_savart_B_and_dB(...)` on the JAX side.
- [ ] Compare JAX `jax.jacfwd` field Jacobian against JAX `jax.linearize`
      field Jacobian to isolate internal JAX route drift.
- [ ] Compare JAX grouped accumulation order against CPU coil loop order.
- [ ] If `B` matches and `dB_by_dX` does not, isolate the first component and
      point where the derivative differs.
- [ ] If `B` itself does not match, resolve point ordering, coil grouping, current
      scaling, or quadrature ordering before touching derivative code.

Acceptance gate:

- [ ] Biot-Savart field inputs and outputs are either byte-identical, or the
      exact first field/derivative mismatch is documented.

### Phase 4 - Residual Derivative Assembly Identity

Context: if the raw arrays match but the gradient still differs, the bug is in
the JAX residual derivative assembly rather than derivative input generation.
This phase must cover both the LS pre-Newton scalar-gradient path and the outer
`BoozerResidual` / iota-adjoint path, because the current strict census also
reports downstream `iota_penalty.adjoint` drift.

Files:

- `src/simsopt/geo/boozer_residual_jax.py`
- `src/simsopt/geo/boozersurface.py`
- `src/simsopt/geo/surfaceobjectives.py`
- `src/simsopt/geo/boozersurface_jax.py`
- `src/simsoptpp/boozerresidual_impl.h`
- `src/simsoptpp/boozerresidual_py.cpp`

Todos:

- [ ] Feed CPU-recorded raw arrays into
      `boozer_residual_scalar_and_grad_cpu_ordered(...)`.
- [ ] Compare against CPU `sopp.boozer_residual_ds(...)` at the same arrays.
- [ ] Compare per-point contributions before the final `num_res` normalization.
- [ ] Compare gradient accumulation order for each surface dof and tail variable.
- [ ] Compare handling of `optimize_G=False`, including CPU's full gradient then
      `Jnl[:-1]` path versus JAX's packed gradient path.
- [ ] Compare label and z-pin gradient contributions separately from Boozer
      residual gradient contributions.
- [ ] Compare CPU `boozer_surface_residual(..., derivatives=1)` against JAX
      `boozer_residual_jacobian_composed(...)` at identical fixed arrays.
- [ ] Compare CPU `_call_boozer_dresidual_dc(...)` inputs and outputs against
      the corresponding JAX derivative contraction.
- [ ] Compare `Jtil.T @ rtil` construction before it enters the Boozer adjoint
      solve.
- [ ] Compare `dJ_by_dB` before CPU `BiotSavart.B_vjp(...)` and JAX field VJP
      paths consume it.
- [ ] Compare exact-path coil VJP and LS-path coil VJP inputs:
  - [ ] adjoint vector `lm`
  - [ ] solved `sdofs`
  - [ ] solved `iota`
  - [ ] solved `G`
  - [ ] field points
  - [ ] grouped coil arrays
  - [ ] coil index lists

Acceptance gate:

- [ ] With identical raw inputs, CPU and JAX residual scalar/gradient bytes match,
      and the outer residual/JVP/VJP ingredients match, or the first
      arithmetic-order divergence is isolated to one formula.

### Phase 5 - Test-Only Ablations To Prove Ownership

Context: ablations are allowed as diagnostics only. They must not become
production fallbacks.

Todos:

- [ ] Run JAX residual assembly with CPU-recorded surface derivative arrays.
- [ ] Run JAX residual assembly with CPU-recorded Biot-Savart `B` and `dB_by_dX`.
- [ ] Run JAX residual assembly with both CPU-recorded surface derivatives and
      CPU-recorded Biot-Savart arrays.
- [ ] Run CPU residual assembly, where possible, with JAX-recorded arrays.
- [ ] Record which substitution first turns the fixed-candidate gradient into
      byte identity.
- [ ] Delete or quarantine ablation hooks if they are not needed as permanent
      tests.

Acceptance gate:

- [ ] A single owner layer is proven by substitution, not guessed from aggregate
      optimizer traces.

### Phase 6 - Implement The Smallest Root Fix

Context: the fix depends on the proven owner layer.

Decision tree:

- [ ] If surface derivative generation owns the mismatch:
  - [ ] Implement CPU-ordered explicit JAX derivative kernels for the consumed
        Boozer surface kind.
  - [ ] Route only the CPU-parity Boozer value/gradient closure through those
        kernels if the target lane should keep the native JAX route.
- [ ] If Biot-Savart derivative generation owns the mismatch:
  - [ ] Align point/coil/current ordering first.
  - [ ] Align reduction order next.
  - [ ] Only then consider replacing `jax.linearize`/`jax.jacfwd` with an
        explicit derivative kernel.
- [ ] If residual assembly owns the mismatch:
  - [ ] Change `boozer_residual_scalar_and_grad_cpu_ordered(...)` to mirror the
        CPU formula and order exactly.
- [ ] If no practical bit-identical implementation exists without duplicating a
      large upstream C++ kernel:
  - [ ] Document that CPU/JAX scientific parity is closed and strict byte
        identity is out of product scope.
  - [ ] Keep the strict census as a known diagnostic gate rather than weakening
        production correctness criteria.

Acceptance gate:

- [ ] The chosen fix is a root fix for the proven owner layer and does not add
      production fallbacks.

### Phase 7 - Regression And Release Gates

Todos:

- [ ] Add targeted tests for the first mismatching raw derivative array.
- [ ] Add a fixed-candidate scalar/gradient byte-identity test.
- [ ] Re-run the existing Boozer parity suites:
  - [ ] `tests/geo/test_boozer_residual_jax.py`
  - [ ] `tests/geo/test_boozersurface_jax.py`
  - [ ] `tests/geo/test_boozersurface_jax_private.py`
  - [ ] `tests/geo/test_surface_objectives_jax.py`
- [ ] Re-run the single-stage parity benchmark that produced the current
      artifact.
- [ ] Confirm:
  - [ ] `same_candidate_replay.status == "pass"`
  - [ ] `max_candidate_abs_diff == 0.0`
  - [ ] no divergent `boozer_solve.pre_newton_*` census layer
  - [ ] final iota/volume/field-error drift remains at current or better scale
- [ ] Update `docs/boozer_bfgs_pre_newton_contract_impl_plan_2026-05-07.md`
      with the final outcome and artifact path.

## Expected Deliverables

- [ ] A fixed-candidate derivative input census artifact.
- [ ] A concise root-cause report naming the first non-byte-identical array and
      producer.
- [ ] Targeted tests that fail on the current mismatch and pass after the fix or
      contract decision.
- [ ] A minimal code patch only if the owner layer is fixable without production
      fallbacks.
- [ ] Updated parity documentation with the final gate status.

## Open Questions

- [ ] Does the first mismatch appear in surface coefficient derivatives before
      Biot-Savart sees any points?
- [ ] Does `dB_by_dX` differ when `B` is byte-identical?
- [ ] Does JAX `jax.jacfwd` produce the same field Jacobian as JAX
      `jax.linearize` for this kernel and shape?
- [ ] Is CPU C++ using a different derivative formula or only a different
      accumulation order?
- [ ] Is the strict byte-identity requirement product-critical, or is the current
      scientific parity plus same-candidate replay enough for release?
