# Boozer Hessian CPU/C++ Oracle Parity Implementation Plan

Date: 2026-05-05

## Context

Current Boozer penalty Hessian coverage is valid but does not yet make a
full direct-Hessian parity claim. The existing upstream matrix test compares a
single seeded bilinear projection:

```text
h1.T @ H_cpu @ h2  vs  h1.T @ JAX_HVP(h2)
```

The CPU side is already backed by the SIMSOPT CPU/C++ path:

- `BoozerSurface.boozer_penalty_constraints_vectorized(..., derivatives=2)`
- `_call_boozer_residual_ds2(...)`
- `simsoptpp.boozer_residual_ds2(...)`

The remaining gap is the evidence class. A single bilinear projection proves a
directional second-derivative check, not column-complete Hessian parity.

The upgraded claim should be:

```text
Every JAX Hessian-vector basis column matches the CPU/C++ scalar Boozer
penalty Hessian column for the same unsolved upstream fixture state.
```

The unsolved initial `case.x` is intentional. Because the residual is nonzero,
the test exercises the full scalar Hessian, including residual-weighted
second-derivative terms, not only the Gauss-Newton `J.T @ J` contribution.

## Scope

In scope:

- Add column-complete CPU/C++ oracle parity for the scalar Boozer penalty
  Hessian.
- Keep the existing directional HVP test.
- Use the existing `derivative_heavy` second-derivative tolerance lane.
- Update docs after the test passes.

Out of scope:

- CUDA/GPU parity claims.
- Tolerance loosening.
- Production fallbacks or defensive production code.
- Solver trajectory parity.

## Contract

Use the SSOT tolerance lane from
`benchmarks/validation_ladder_contract.py::PARITY_LADDER_TOLERANCES`:

```text
lane: derivative_heavy
second_derivative_rtol = 1e-6
second_derivative_atol = 1e-8
```

Same-state inputs must match across CPU and JAX:

- surface DOFs
- `iota`
- `G` / `optimize_G`
- quadrature
- label and target label
- `constraint_weight`
- `weight_inv_modB`
- surface type
- `stellsym`
- coil data

All JAX inputs for this parity test must be explicitly float64:

- `x = jnp.asarray(case.x, dtype=jnp.float64)`
- `basis = jnp.eye(x.size, dtype=jnp.float64)`

Use `JAX_ENABLE_X64=True` and `JAX_PLATFORMS=cpu` for validation commands.
Older Boozer docs still contain `JAX_PLATFORM_NAME=cpu`; this plan uses the
repo runtime/benchmark convention and does not perform that broader doc cleanup.

## Implementation Plan

### 1. Add a Hessian column-complete evaluator

Add a test helper near `_evaluate_upstream_boozer_penalty_case` in
`tests/geo/boozersurface_jax_test_helpers.py`.

Responsibilities:

- Call `case.cpu_boozer.boozer_penalty_constraints_vectorized(...)` with
  `derivatives=2`.
- Keep only the CPU Hessian from that call. Value and gradient parity are
  already covered by `_evaluate_upstream_boozer_penalty_case`.
- Build the JAX scalar objective using
  `case.jax_boozer._make_penalty_objective_with(...)`.
- Build `grad_fn = jax.grad(objective)`.
- Build one HVP function using `jax.jvp(grad_fn, (x,), (v,))`.
- Sweep standard basis vectors and assemble the JAX Hessian by columns.
  Implement the default path with `jax.vmap` over one HVP function:

  ```python
  x = jnp.asarray(case.x, dtype=jnp.float64)
  basis = jnp.eye(x.size, dtype=jnp.float64)

  def hvp_single(v):
      return jax.jvp(grad_fn, (x,), (v,))[1]

  rows = jax.vmap(hvp_single)(basis)
  jax_hessian = np.asarray(jax.device_get(rows)).T
  ```

  Add a local comment in the helper:

  ```python
  # Standard basis for column-complete oracle parity; do not replace with
  # _seeded_hessian_direction_pair, which covers one random projection.
  ```

  If a future fixture grows beyond the current roughly 365-DOF worst case,
  replace only the mapping primitive with `jax.lax.map` or explicit chunking.
  Do not downgrade the test to one seeded random direction.
- Return `cpu_hessian` and `jax_hessian` as host NumPy arrays, using
  `jax.device_get` before `np.asarray`.

Do not use dense `jax.hessian()` for the full upstream matrix. The default
fixtures use `mpol=5`, `ntor=5`, and include the large
`SurfaceXYZTensorFourier`, `stellsym=False`, `optimize_G=True` case. A dense
Hessian transform across the whole matrix risks excessive compile time and
memory. A column-complete HVP sweep gives the same evidence with predictable
memory.

### 2. Add the main parity test

Add a test in `TestUpstreamFactoryBoozerMatrix`:

```text
test_penalty_hessian_column_complete_cpu_parity_matrix
```

Use the existing upstream parameter matrix:

- `SurfaceXYZFourier`
- `SurfaceXYZTensorFourier`
- `stellsym=True`
- `stellsym=False`
- `optimize_G=True`
- `optimize_G=False`

Assertions:

```python
tolerances = parity_ladder_tolerances("derivative-heavy")
rtol = tolerances["second_derivative_rtol"]
atol = tolerances["second_derivative_atol"]

np.testing.assert_allclose(
    jax_hessian,
    cpu_hessian,
    rtol=rtol,
    atol=atol,
    err_msg="CPU/JAX penalty Hessian column-complete mismatch",
)
```

Also assert:

- `cpu_hessian.shape == jax_hessian.shape`
- both Hessians are finite
- `cpu_hessian` is symmetric with a strict CPU/C++ guard:

  ```python
  np.testing.assert_allclose(
      cpu_hessian,
      cpu_hessian.T,
      rtol=1e-12,
      atol=1e-12,
      err_msg="CPU Hessian asymmetric; check CPU/C++ oracle regression",
  )
  ```

- `jax_hessian` is symmetric under the same `rtol` / `atol`

These are test assertions only. They do not add defensive production behavior.
The test docstring should note that the column-complete basis sweep executes
roughly one HVP per decision variable across the upstream matrix and is expected
to live on the auto-marked slow Boozer lane.

### 3. Keep the existing directional test

Keep `test_penalty_hessian_directional_cpu_parity_matrix`.

Reason: the directional test directly exercises the seeded JVP-of-gradient
operator path. The new column-complete test upgrades the oracle evidence; the
existing test remains useful operator-path coverage.

### 4. Optional dense sanity cross-check

Optionally add one dense `jax.hessian(objective)(x)` sanity check for the
explicitly reduced fixture only:

```text
SurfaceXYZFourier, mpol=2, ntor=2, stellsym=True, optimize_G=False
```

Purpose:

- verify that the column-complete HVP assembly agrees with dense JAX Hessian
  materialization once

Do not run dense `jax.hessian()` over the full upstream parameter matrix.

### 5. Update documentation after tests pass

Update:

- `CLAUDE.md`
  - Replace the stale Boozer Hessian TODO sentence with:

    ```text
    second-derivative/Hessian direct C++ parity is covered by
    TestUpstreamFactoryBoozerMatrix::test_penalty_hessian_column_complete_cpu_parity_matrix
    using column-complete CPU/C++ Hessian basis-sweep parity at rtol=1e-6,
    atol=1e-8; the seeded directional HVP test is retained as operator-path
    coverage.
    ```

- `docs/jax_parity_manifest.md`
  - Extend the existing `derivative_heavy` evidence row to include:

    ```text
    tests/geo/test_boozersurface_jax.py::TestUpstreamFactoryBoozerMatrix::test_penalty_hessian_column_complete_cpu_parity_matrix
    ```

- `docs/boozer_full_parity_plan_2026-05-04.md`
  - Update the `test_boozer_penalty_constraints_hessian` row from "full matrix
    parity or documented FD/Taylor equivalence" to:

    ```text
    Column-complete CPU/C++ Hessian oracle parity via
    test_penalty_hessian_column_complete_cpu_parity_matrix; retain seeded
    directional HVP coverage.
    ```

  - Update the Hessian parity section to state that direct column-complete
    Hessian parity exists and the seeded directional HVP test is retained.
  - Clarify the "Do not claim direct Hessian or adjoint vector parity from
    FD-only evidence" line as:

    ```text
    Do not claim adjoint vector parity from FD-only evidence; Boozer Hessian
    direct parity is now column-complete CPU/C++ oracle coverage.
    ```

  - Update the upstream-listing entry for `test_boozer_penalty_constraints_hessian`
    if it repeats the old directional-only status.

## Validation

Focused Hessian gate:

```bash
JAX_ENABLE_X64=True JAX_PLATFORMS=cpu ./.conda/jax-0.9.2/bin/python -m pytest -q tests/geo/test_boozersurface_jax.py -k 'penalty_hessian'
```

Derivative slice:

```bash
JAX_ENABLE_X64=True JAX_PLATFORMS=cpu ./.conda/jax-0.9.2/bin/python -m pytest -q tests/geo/test_boozersurface_jax.py tests/geo/test_boozer_derivatives_jax.py -k 'hessian or derivative'
```

Diff hygiene:

```bash
git diff --check
```

Doc-drift guard:

```bash
! grep -RIn "second-derivative.*TOD[O]" CLAUDE.md docs/
```

## Acceptance Criteria

- The new column-complete Hessian parity test passes over the upstream matrix.
- The test uses `derivative_heavy` `second_derivative_*` tolerances.
- The existing directional HVP Hessian test remains.
- Docs accurately distinguish direct column-complete Hessian parity from
  directional HVP coverage.
- No stale Hessian direct-parity TODO wording remains in `CLAUDE.md` or `docs/`.
- No fallback path, no tolerance loosening, no CUDA parity claim.
