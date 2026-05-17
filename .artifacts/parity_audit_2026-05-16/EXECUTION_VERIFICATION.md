# Execution Verification - ISSUES_CHECKLIST.md

**Date:** 2026-05-17
**Repo HEAD:** `8d7577f8a`
**Source under audit:** `.artifacts/parity_audit_2026-05-16/ISSUES_CHECKLIST.md`
**Local interpreter:** `.conda/jax/bin/python` (`jax==0.10.0`, backend `cpu`,
devices `[CpuDevice(id=0)]`)
**Scope:** Re-check the incomplete/partial rows called out by the prior
verification report, patch the remaining local gaps, and separate local CPU
proof from external CUDA/platform signoff. Follow-up strict-contract reviews
also covered stale code and silent fallback behavior in the touched
permanent-magnet solve wrapper, the shared tracing event localizer, and the
magnetic-axis helper documentation.

## Final Verdict

The prior verification report is superseded. Its "Confirmed NOT EXECUTED" table
was stale against the live tree, and several "Not Resolved" entries were either
fixed already, fixed in this pass, or based on an incorrect physics/action
statement.

Local actionable rows from this report are now resolved or explicitly
dispositioned. The follow-up review fixed additional local issues:
CPU/JAX `prox_l0` now respects zero-capacity dipoles, explicit JAX `m0`
validation no longer silently drops out under tracing/transfer guard, and
oversized MwPGP `alpha` is rejected instead of warning-and-continuing. A later
tracing review also made `bracket_root_jax` inactive for non-brackets and
removed stale `dB_by_dX`/reverse-mode wording from the magnetic-axis helper.
The P8 gates remain external signoff gates and are not claimed complete from
this CPU-only workstation.

## Official Documentation Checked

- JAX `jnp.fmax` docs: `fmax` returns the finite operand when paired with one
  `nan`, and returns `nan` only when both operands are `nan`.
- JAX transfer guard docs: explicit transfers such as `jax.device_get` are
  distinguished from implicit transfers; strict tests should catch unintended
  host/device movement.
- JAX gotchas docs: JIT-compatible code must keep array shapes static; dynamic
  result shapes require restructuring.
- JAX `lax.while_loop` docs: loop-carried values have fixed shape/dtype, and
  `while_loop` is not reverse-mode differentiable because XLA needs static
  memory bounds.
- SIMSOPT Boozer docs: Boozer fields expose both covariant
  `B = G grad zeta + I grad theta + K grad psi` and contravariant
  `B = (x_zeta + iota x_theta) / sqrt(g)` forms. The correct checked identity
  for item F-DH21 is the covariant toroidal component
  `B dot x_zeta = G`, not `B dot grad zeta = G`.
- SIMSOPT tracing docs: public tracing APIs define `tmax` as the trace duration;
  status and step-budget exhaustion must remain visible rather than being
  hidden by fabricated endpoint extrapolation.
- NVIDIA CUDA programming-guide docs: GPU execution/scheduling and floating
  point reduction order are not a local CPU proof. CUDA determinism, production
  scale, and cross-platform parity stay in `P8_EXTERNAL_SIGNOFF.md`.

## Corrected Disposition

| ID | Final status | Evidence |
|----|--------------|----------|
| F1/H7(a) NaN projection | Resolved before this pass | `src/simsopt/jax_core/pm_optimization.py` uses `m_maxima**0` plus `jnp.fmax`; tests cover zero and `nan` `m_maxima`. |
| F1/H7 prox zero-capacity contract | Resolved in follow-up review | CPU/JAX `prox_l0` and `prox_l1` normalize only positive `m_maxima`; `test_zero_mmax_helpers_match_cpu_without_nan` covers nonzero moments on zero-capacity dipoles. |
| F1/H7(e) near-axis PM grid | Resolved in this pass | `tests/geo/test_pm_grid.py` now includes `r = 1e-12` and `1e-15` cylindrical cells and asserts finite positive `m_maxima`. |
| Explicit JAX `m0` validation | Resolved in follow-up review | `setup_initial_condition_jax` validates eager explicit `m0` with explicit `jax.device_get`; traced explicit `m0` is rejected so infeasible values cannot bypass the CPU contract. |
| R07-A2 oversized alpha | Resolved in follow-up review | Explicit `alpha > 2/lambda_max(H)` now raises `ValueError`; the previous diagnostic warning path was removed. |
| F-DH11 surface-DOF drift | Resolved before this pass | `tests/objectives/test_fluxobjective_jax_parity.py::test_squaredfluxjax_rejects_surface_dof_mutation_after_construction`. |
| F-DH22 `mn_factor` extrapolation | Resolved before this pass | `tests/field/test_boozermagneticfield_jax_item33.py::test_mn_factor_extrapolation_below_first_retained_knot`. |
| R04-A2 max-step warning | Resolved before this pass | `src/simsopt/field/tracing.py` emits a structured warning; `tests/field/test_tracing_jax_item16.py::test_compute_fieldlines_jax_warns_on_step_budget_exhaustion` covers it. |
| F-DH17 C++ `atan2(0,0)` | Resolved in current working tree | C++ non-cartesian dipole path uses the scalar zero-angle convention and rejects invalid coordinate flags; targeted parity/input-validation tests pass. |
| F13d dipole singularity docs | Resolved in this pass | CPU and JAX public wrappers now state the raw point-dipole singularity contract. |
| F-DH21 Boozer identity | Resolved as physics correction | Prior `B dot grad zeta = G` wording was wrong; current test checks `B dot x_zeta = G`, the correct covariant toroidal component. |
| F-DH6 conservation over bounce | Resolved in this pass | `tests/jax_core/test_tracing_jax_conservation.py` now uses trapped parameters and asserts at least one parallel-velocity sign change before checking mu/energy. |
| F3/H2 `dtmax` formula | Resolved by public wrapper contract | Public wrappers compute the C++ quarter-turn `dtmax`; lower-level specs intentionally accept explicit `dtmax`. |
| F-DH7 bracket localizer edge cases | Resolved in follow-up review | `src/simsopt/jax_core/tracing.py` now leaves non-brackets inactive and keeps the false-position candidate finite; `test_bracket_root_keeps_equal_residual_no_bracket_result_finite` covers the equal-residual no-bracket poison case. |
| F-DH8 post-loop back-fill | Dispositioned: no budget-exhausted backfill | Accepted steps are clamped to `tmax` for normal exits. For status=1 step-budget exhaustion, back-filling to `tmax` would hide the failure; the wrapper warning/status is the correct contract. |
| F-DH Row 3 `weight_inv_modB` partial | Dispositioned as intentional contract | Module docs state public scalar/vector helpers default to weighted residuals; internal composed/vector VJP paths default false for legacy least-squares call sites. Both branches have C++ oracle coverage. |
| F-DH Row 4 lost-particle edge cases | Resolved in follow-up review | Item 14 tests now cover both static interpolation-cuboid face classification and an active guiding-centre trace that exits through the classifier cuboid face mid-integration, plus multiple criteria firing on the same accepted step and exact-zero levelset behavior. |
| Magnetic-axis helper stale convention wording | Resolved in follow-up review | `src/simsopt/jax_core/magnetic_axis_helpers.py` now states that the derivative matrix is consumed in upstream CPU-helper order without transposing field-specific layouts, and the reverse-mode AD text now mirrors official JAX `lax.while_loop` semantics rather than pinning an exception string. |
| Checklist stale line numbers/closeout text | Resolved in follow-up review | `ISSUES_CHECKLIST.md` now records that local rows are closed by live verification plus this pass, while P8 remains external; the stale tracing references in the P1 block were rewritten against the current helper-based structure. |

## Validation Run

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src JAX_ENABLE_X64=True JAX_PLATFORM_NAME=cpu \
  .conda/jax/bin/python -m pytest -q \
  tests/geo/test_pm_grid.py::PermanentMagnetGridTesting::test_cylindrical_grid_chopping_removes_axis_cells \
  tests/jax_core/test_pm_optimization_jax_item25.py::TestPMKernelHelpers::test_projection_zero_mmax_zero_row_is_finite \
  tests/solve/test_permanent_magnet_optimization_jax_item28.py::test_projection_nan_mmax_matches_cpu_fmax_contract \
  tests/objectives/test_fluxobjective_jax_parity.py::test_squaredfluxjax_rejects_surface_dof_mutation_after_construction \
  tests/field/test_boozermagneticfield_jax_item33.py::test_mn_factor_extrapolation_below_first_retained_knot \
  tests/field/test_boozermagneticfield_jax_item33.py::test_covariant_toroidal_identity_matches_G \
  tests/field/test_tracing_jax_item16.py::test_compute_fieldlines_jax_warns_on_step_budget_exhaustion \
  tests/jax_core/test_tracing_jax_conservation.py \
  tests/jax_core/test_dipole_field_item24.py::test_dipole_field_Bn_on_axis_noncartesian_matches_cpp \
  tests/jax_core/test_dipole_field_item24.py::test_dipole_field_Bn_rejects_invalid_coordinate_flag \
  tests/field/test_dipole_field_jax_item26.py::TestDipoleFieldJAXInputValidation::test_rejects_invalid_coordinate_flag
```

Result: `15 passed, 1 skipped, 3 warnings in 11.44s`.

Additional partial-row verification:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src JAX_ENABLE_X64=True JAX_PLATFORM_NAME=cpu \
  .conda/jax/bin/python -m pytest -q \
  tests/geo/test_boozer_residual_jax.py::TestBoozerResidualScalar::test_scalar_matches_cpp_oracle \
  tests/jax_core/test_tracing_jax_item14.py::test_levelset_classifier_grid_faces_remain_classified \
  tests/jax_core/test_tracing_jax_item14.py::test_trace_guiding_center_stops_after_exiting_classifier_cuboid_face \
  tests/jax_core/test_tracing_jax_item14.py::test_trace_fieldline_first_stopping_criterion_wins_same_step \
  tests/jax_core/test_tracing_jax_item14.py::test_trace_fieldline_levelset_zero_does_not_stop
```

Result: `6 passed, 2 skipped, 2 warnings in 5.82s`.

Follow-up strict-contract verification:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src JAX_ENABLE_X64=True JAX_PLATFORM_NAME=cpu \
  .conda/jax/bin/python -m pytest -q \
  tests/solve/test_permanent_magnet_optimization_jax_item28.py
```

Result: `29 passed in 8.79s`.

Focused strict-contract slice:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src JAX_ENABLE_X64=True JAX_PLATFORM_NAME=cpu \
  .conda/jax/bin/python -m pytest -q \
  tests/solve/test_permanent_magnet_optimization_jax_item28.py::test_zero_mmax_helpers_match_cpu_without_nan \
  tests/solve/test_permanent_magnet_optimization_jax_item28.py::test_setup_initial_condition_jax_explicit_m0_validates_under_guard \
  tests/solve/test_permanent_magnet_optimization_jax_item28.py::test_setup_initial_condition_jax_rejects_infeasible_explicit_m0 \
  tests/solve/test_permanent_magnet_optimization_jax_item28.py::test_setup_initial_condition_jax_rejects_traced_explicit_m0 \
  tests/solve/test_permanent_magnet_optimization_jax_item28.py::test_relax_and_split_jax_rejects_oversized_alpha \
  tests/solve/test_permanent_magnet_optimization_jax_item28.py::test_relax_and_split_jax_jits_under_strict_transfer_guard
```

Result: `6 passed in 3.22s`.

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src JAX_ENABLE_X64=True JAX_PLATFORM_NAME=cpu \
  .conda/jax/bin/python -m pytest -q \
  tests/solve/test_permanent_magnet_optimization_jax_item28.py::test_relax_and_split_jax_default_alpha_includes_smooth_terms \
  tests/solve/test_permanent_magnet_optimization_jax_item28.py::test_relax_and_split_jax_matches_direct_mwpgp_fixed_step \
  tests/solve/test_permanent_magnet_optimization_jax_item28.py::test_relax_and_split_jax_matches_cpu_after_multiple_outer_steps
```

Result: `3 passed in 4.46s`.

Tracing/magnetic-axis follow-up review:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src JAX_ENABLE_X64=True JAX_PLATFORM_NAME=cpu \
  .conda/jax/bin/python -m pytest -q \
  tests/jax_core/test_tracing_jax_item14.py
```

Result after simplifier pass: `35 passed, 2 warnings in 8.91s`.

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src JAX_ENABLE_X64=True JAX_PLATFORM_NAME=cpu \
  .conda/jax/bin/python -m pytest -q \
  tests/field/test_magnetic_axis_helpers_jax_item21.py
```

Result after simplifier pass: `15 passed in 8.95s`.

Diff hygiene:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src JAX_ENABLE_X64=True JAX_PLATFORM_NAME=cpu \
  .conda/jax/bin/python -m ruff check \
  src/simsopt/jax_core/tracing.py \
  src/simsopt/jax_core/magnetic_axis_helpers.py \
  tests/jax_core/test_tracing_jax_item14.py \
  tests/field/test_magnetic_axis_helpers_jax_item21.py
PYTHONNOUSERSITE=1 PYTHONPATH=src JAX_ENABLE_X64=True JAX_PLATFORM_NAME=cpu \
  .conda/jax/bin/python -m ruff format --check \
  src/simsopt/jax_core/tracing.py \
  src/simsopt/jax_core/magnetic_axis_helpers.py \
  tests/jax_core/test_tracing_jax_item14.py \
  tests/field/test_magnetic_axis_helpers_jax_item21.py
git diff --check
```

Result: `ruff check` passed; `ruff format --check` passed; `git diff --check` passed.

## Residual External Gates

`P8_EXTERNAL_SIGNOFF.md` remains open by design. This local pass did not produce
current-SHA CUDA, H200/production-scale, cross-platform, long-soak, or
concurrency proof artifacts.
