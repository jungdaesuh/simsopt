# Phase 7 validation note — Boozer derivative bit-identity zeroing

**Date:** 2026-05-07
**Refreshed:** 2026-05-08 — regenerated the census artifacts after the
Crucible F3/F4 diagnostic fixes so stored NDJSON reflects sign-of-zero bit
differences.
**Plan:** [`docs/boozer_derivative_bit_identity_impl_plan_2026-05-07.md`](../../../docs/boozer_derivative_bit_identity_impl_plan_2026-05-07.md)
**Branch:** `gpu-purity-stage2-20260405`

## Acceptance ladder (Phase 7)

The Phase 7 strict gate at
`benchmarks/single_stage_init_parity.py:1905` reports a
`boozer_solve.pre_newton_state` divergence of `4.5e-9` (post-BFGS
amplification of a `1.6e-15` first-step gradient mismatch). This slice
implemented the surface (Phase 2) and Biot-Savart (Phase 3) CPU-ordered
parity twins and wired them through `is_parity_mode()` in Phase 6. The
strict pre-Newton gate **is not closed in this slice** — Phase 4
(residual-FMA assembly) is deferred and tracked as a residual.

What this slice **does** ship:

| Phase | Deliverable | Status |
|---|---|---|
| 0 | Reproducer skeleton + candidate extraction | done |
| 1 | Census schema, capture helpers, NDJSON, red test | done — `gamma` is the documented first owner |
| 2 | Surface CPU-ordered twins (`surface_fourier_jax_cpu_ordered.py`) | done — gamma drift narrows 6.66e-16 → 2.22e-16 |
| 3 | Biot-Savart CPU-ordered twins (`biotsavart_cpu_ordered.py`) | done — dB drift narrows 4.00e-15 → 3.11e-15 |
| 4 | Residual FMA restructuring | **deferred** (see Residual section) |
| 5 | Ablations | implicit via Phase 1 census diff (`production/` vs `cpu_ordered_full/`) |
| 6 | Auto-select via `is_parity_mode()` | done — factory cache key bifurcates production vs parity twins |
| 7 | Regression and release gates | this note |

## Per-array drift snapshot (synthetic NCSX fixture)

`tests/geo/test_boozer_derivative_input_census.py` and the reproducer at
`benchmarks/parity/boozer_derivative_input_repro.py` produce the
following ladder. Both rows pin against the C++ oracle:

| Boundary array | Production (matmul + jacfwd) | cpu_ordered (parity twins) |
|---|---|---|
| gamma | 6.66e-16 | 2.22e-16 |
| xphi | 5.33e-15 | 1.78e-15 |
| xtheta | 3.33e-16 | 1.11e-16 |
| dx_ds | 0.0 (sign-of-zero only) | 0.0 (sign-of-zero only) |
| dxphi_ds | 1.42e-14 | 7.11e-15 |
| dxtheta_ds | 3.55e-15 | 3.55e-15 |
| B | 1.55e-15 | 1.11e-15 |
| dB_dX | 4.00e-15 | 3.11e-15 |

All scalars (`G_value`, `iota`, `weight_inv_modB`) are byte-identical in
both modes.

## Tests added (regression coverage)

- `tests/geo/test_boozer_derivative_input_census.py` — 10 tests; pins the
  ladder shape, schema round-trips, scalar invariants, dtype-mismatch
  reporting, sign-of-zero bit counting, and NDJSON I/O.
- `tests/geo/test_surface_fourier_jax_cpu_ordered.py` — 10 tests
  (3 parametric × 3 kernels + 1 routing); fixture matrix
  `(SurfaceXYZTensorFourier, mpol, ntor, nfp, stellsym, nphi, ntheta)`
  documented in the file. Asserts cpu_ordered drift is no worse than
  production and stays within the 1e-13 ULP ceiling.
- `tests/field/test_biotsavart_jax_cpu_ordered.py` — 4 tests; B, dB/dX,
  no-regression vs production matmul, and `_field_terms_for_local_label`
  parity_policy routing.
- Existing `tests/geo/test_boozersurface_jax.py` (365 tests) — full pass
  on JAX 0.9.2 lane after factory edits.

## Files touched

### New

- `benchmarks/parity/__init__.py`
- `benchmarks/parity/boozer_derivative_input_repro.py`
- `benchmarks/parity/boozer_derivative_input_census.py`
- `src/simsopt/geo/surface_fourier_jax_cpu_ordered.py`
- `src/simsopt/jax_core/biotsavart_cpu_ordered.py`
- `tests/geo/test_boozer_derivative_input_census.py`
- `tests/geo/test_surface_fourier_jax_cpu_ordered.py`
- `tests/field/test_biotsavart_jax_cpu_ordered.py`

### Edited (private input helpers + parity-policy wiring)

- `src/simsopt/geo/boozersurface.py` — extracted
  `_boozer_penalty_vectorized_inputs` boundary helper at the
  `_call_boozer_residual_ds` callsite.
- `src/simsopt/geo/boozersurface_jax.py` — extracted
  `_boozer_penalty_value_and_grad_inputs_cpu_ordered`,
  added `_BoozerPenaltyVectorizedInputs` pytree, threaded `parity_policy`
  through `_surface_geometry_and_derivatives_from_dofs`,
  `_field_terms_for_local_label`, and the
  `_make_penalty_value_and_grad_cpu_ordered_with` factory.
- `pyproject.toml` — registered `parity_census` pytest marker.

### Untouched (per plan §15)

- `src/simsoptpp/**`
- `benchmarks/single_stage_init_parity.py`
- `benchmarks/validation_ladder_contract.py`
- Production JAX hot paths in `surface_fourier_jax.py`,
  `jax_core/biotsavart.py`, `jax_core/field.py`,
  `field/biotsavart_jax_backend.py`.

## Crucible review outcome

Crucible (max Opus 4.7 subagents, 4 lenses run; lenses 3/4 skipped — git
history not consulted) surfaced findings; the actionable ones were fixed
in this slice:

* **F7 (Mistake Book Pattern 4 — missing dtype on `jnp.array` in the
  kernel layer):** fixed. All `jnp.array([...])` constructions in
  `src/simsopt/jax_core/biotsavart_cpu_ordered.py` now carry
  `dtype=gammas.dtype`.
* **F9 (Mistake Book Pattern 6 — dead `n_full` allocation):** fixed.
  Removed in `src/simsopt/geo/surface_fourier_jax_cpu_ordered.py:76`.
* **F4 / F3 (census diagnostic ambiguity around `+0.0`/`-0.0`):** fixed.
  `_first_unequal_byte_index` now returns ``None`` for shape mismatches
  (no longer collides with a real first-byte mismatch at index 0); the
  per-array `n_bit_different_entries` is now a bytewise count over
  `np.uint64`, so sign-of-zero divergences are surfaced faithfully.
* **F1 (label-gradient parity bypass):** fixed. `parity_policy` now
  flows through `_label_value_from_surface_dofs` and
  `_geometry_from_surface_dofs`; under
  ``SIMSOPT_BACKEND_MODE=jax_cpu_parity`` the label-side surface
  geometry routes through the cpu_ordered twins (same `xyztensorfourier`
  whitelist as the residual side; non-tensor surface kinds keep the
  matmul path).
* **F8 (Mistake Book Pattern 5 — untracked files):** acknowledged
  as a handoff obligation. The shippable scope is the explicit source/test
  parity slice plus this single validation artifact subtree; do not sweep
  unrelated `.artifacts/` directories into the commit.

## Residual / deferred work

1. **Phase 4 (residual-FMA restructuring).** The remaining 1–2 ULP gap
   on `gamma`, `B`, and the gradient cascade is FMA-fusion (plan §1
   item 3). Closing it requires either:
   - A `jax.lax.optimization_barrier` probe under the exact transform
     stack (`jax.jacfwd` for surface derivatives at line 1069 + the
     `jax.value_and_grad` for label terms at line 1211); the plan §18
     probe log only verified the barrier under `jax.jit` and `jax.vmap`,
     **not** under those autodiff transforms.
   - The Side Track x86_64 FMA-fusion reproducer
     (`benchmarks/parity/lane4_fma_fusion_repro_x86.py` and
     `lane5_hlo_dump_repro_x86.py` per plan §5) to confirm the fusion
     shape XLA picks on the production target before adopting any
     restructuring.

   **Phase 4 is therefore not yet ready to ship.** The strict gate at
   `benchmarks/single_stage_init_parity.py:1905` will continue to report
   a non-zero `boozer_solve.pre_newton_state` until Phase 4 lands.

2. **Artifact-driven reproducer.** The Phase 0 / Phase 1 reproducer uses
   a synthetic NCSX fixture, which exercises the same boundary helpers
   the failing artifact drives. Re-running the strict
   `single_stage_init_parity.py` gate against the failing-artifact
   candidate end-to-end (with parity_policy routed through the
   single-stage runtime) is Phase 7 follow-up work and was not in this
   slice's scope.

3. **`-0.0` vs `+0.0` sign-bit handling on `dx_ds`.** Production-mode
   `dx_ds` has 2310 zero entries with differing sign bits between CPU and
   JAX; the cpu_ordered route narrows this to 834 entries. In both cases
   `max_abs_diff = 0` while `byte_identical = False`. Numerically harmless;
   the regenerated census records this faithfully. Closing it would require
   a canonicalization pass on either producer; outside this slice's scope.

## Validation commands run (this slice)

```
ruff check <changed-files>          # PASS
ruff format <changed-files>          # applied
pytest tests/geo/test_boozer_derivative_input_census.py -v   # 10 passed after F3/F4 refresh
pytest tests/geo/test_surface_fourier_jax_cpu_ordered.py -v  # 10 passed
pytest tests/field/test_biotsavart_jax_cpu_ordered.py -v     # 4 passed
pytest tests/geo/test_boozersurface_jax.py -m "not private_optimizer_runtime"  # 365 passed
pytest tests/geo/test_boozersurface_jax_private.py -q        # 85 passed
pytest tests/geo/test_single_stage_example.py -q             # 323 passed, 20 subtests passed
pytest tests/geo/test_boozer_residual_jax.py                 # 15 passed (15 skipped — pre-existing)
benchmarks/parity/boozer_derivative_input_repro.py --census  # regenerated production NDJSON
benchmarks/parity/boozer_derivative_input_repro.py --census --parity-policy production --dump-arrays .artifacts/parity/20260507-boozer-deriv-input-repro-m1/production
benchmarks/parity/boozer_derivative_input_repro.py --census --parity-policy cpu_ordered --dump-arrays .artifacts/parity/20260507-boozer-deriv-input-repro-m1/cpu_ordered
benchmarks/parity/boozer_derivative_input_repro.py --census --parity-policy cpu_ordered --dump-arrays .artifacts/parity/20260507-boozer-deriv-input-repro-m1/cpu_ordered_full
```

The strict `single_stage_init_parity.py` benchmark is **not re-run in
this slice** — Phase 4 is the gate for that.

## Next slice (Phase 4 entry checklist)

Per plan §10:

- [ ] Keep Phase 4 staged separately from unrelated artifact directories.
- [ ] Re-run the `optimization_barrier` probe under
      `jax.jacfwd(_surface_geometry_and_derivatives_from_dofs)` and
      `jax.value_and_grad(_label_value_from_surface_dofs)`.
- [ ] Land the Side Track lane 4/5 x86_64 reproducers under
      `.artifacts/bit-identity-deepdive-2026-05-07/lane{4,5}_x86_repro/`.
- [ ] Decide between explicit-grouping restructuring vs
      `optimization_barrier`-guarded fma fixed point.
- [ ] Re-run the strict `single_stage_init_parity.py` gate; emit a new
      artifact under
      `.artifacts/parity/<DATE>-derivative-bit-identity-zeroing-pass/`.
