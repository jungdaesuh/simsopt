# Item 02 Coverage Matrix

Status: complete for CPU/JAX oracle closure. CUDA proof is not claimed.

Upstream audit SHA:
`1b0cc3a96063197cdbdd01559e04c25456fbe6ff`.

Grep commands used to enumerate this matrix:

```
git -C /Users/suhjungdae/code/columbia/simsopt-jax grep -nE \
  "selffield|SelfField|self_field|B_regularized|regularization_circ|regularization_rect" \
  tests/ src/
git -C /Users/suhjungdae/code/opensource/simsopt grep -nE \
  "selffield|B_regularized|regularization_circ|regularization_rect" \
  tests/
```

## Source rows (`src/simsopt/field/selffield.py` symbols and downstream callers)

| Source | Path / Node | Intent | Classification | Evidence |
| --- | --- | --- | --- | --- |
| current | `src/simsopt/field/selffield.py:21` | `Biot_savart_prefactor` host scalar | `not_applicable` | One-shot import-time scalar; no hot-path host transfer. |
| current | `src/simsopt/field/selffield.py:26-60` | `_rectangular_xsection_k` / `_rectangular_xsection_delta` auxiliary functions | `covered_by_unit_parity` | `tests/field/test_selffieldforces.py::SpecialFunctionsTests::test_k_square` (`test_selffieldforces.py:58`); `test_delta_square` (`test_selffieldforces.py:64`); `test_symmetry` (`test_selffieldforces.py:70`); `test_limits` (`test_selffieldforces.py:83`). |
| current | `src/simsopt/field/selffield.py:63` | `regularization_circ(a) = a^2 / sqrt(e)` | `covered_by_unit_parity` | `tests/field/test_selffieldforces.py::SpecialFunctionsTests::test_regularization_circ` (`test_selffieldforces.py:109`); `test_regularization_functions_transform_under_strict_transfer_guard` (`test_selffieldforces.py:158`). |
| current | `src/simsopt/field/selffield.py:80` | `regularization_rect(a, b) = a*b*exp(-25/6 + K(a,b))` | `covered_by_unit_parity` | `tests/field/test_selffieldforces.py::SpecialFunctionsTests::test_regularization_rect` (`test_selffieldforces.py:125`); `test_regularization_functions_transform_under_strict_transfer_guard` (`test_selffieldforces.py:158`). |
| current | `src/simsopt/field/selffield.py:97` | `@jit B_regularized_singularity_term` analytic singular term | `covered_by_unit_parity` | `tests/field/test_selffieldforces.py::CoilForcesTest::test_b_regularized_pure_jit_vmap_strict_transfer_guard_matches_wrapper` (`test_selffieldforces.py:191`); production-scale closed-form parity in `tests/field/test_selffield_item02_closeout.py::test_b_regularized_pure_matches_circular_closed_form_oracle_at_production_scale`. |
| current | `src/simsopt/field/selffield.py:116` | `@jit B_regularized_pure` regularized self-field kernel | `covered_by_unit_parity` | `tests/field/test_selffieldforces.py::CoilForcesTest::test_b_regularized_pure_jit_vmap_strict_transfer_guard_matches_wrapper` (`test_selffieldforces.py:191`); `test_circular_coil` (`test_selffieldforces.py:264`); `tests/field/test_selffield_item02_closeout.py::test_b_regularized_pure_matches_circular_closed_form_oracle_at_production_scale`; `::test_b_regularized_pure_wrong_regularization_breaks_closed_form_parity`. |
| current | `src/simsopt/field/coil.py:135` | `RegularizedCoil.B_regularized()` public wrapper (transfer-guard `allow` boundary) | `covered_by_unit_parity` | `tests/field/test_selffieldforces.py::CoilForcesTest::test_regularized_coil_self_field_methods_use_strict_transfer_boundary` (`test_selffieldforces.py:237`). |
| current | `src/simsopt/field/coil.py:154` | `RegularizedCoil.self_force()` Lorentz-force wrapper | `covered_by_unit_parity` | `test_regularized_coil_self_field_methods_use_strict_transfer_boundary` (`test_selffieldforces.py:237`); `test_circular_coil` self-force assertions (`test_selffieldforces.py:264`). |
| current | `src/simsopt/field/coil.py:280` | `CircularRegularizedCoil` consumer of `regularization_circ` | `covered_by_unit_parity` | `tests/field/test_selffieldforces.py::test_circular_regularized_coil_subclass` (`test_selffieldforces.py:2703`). |
| current | `src/simsopt/field/coil.py:300` | `RectangularRegularizedCoil` consumer of `regularization_rect` | `covered_by_unit_parity` | `tests/field/test_selffieldforces.py::test_rectangular_regularized_coil_subclass` (`test_selffieldforces.py:2782`). |
| current | `src/simsopt/field/coil.py:710` | `coils_via_symmetries` consumer of `regularization_circ` | `covered_by_unit_parity` | `tests/field/test_coil.py::test_coil_serialization_round_trip` (`test_coil.py:17,336,369`). |
| current | `src/simsopt/field/force.py:2016,2394` | bulk `vmap(B_regularized_pure, in_axes=(0,0,0,None,0,0))` over coils | `covered_by_integration_parity` | `tests/field/test_selffieldforces.py::test_force_objectives` (`test_selffieldforces.py:1191`); `test_net_force_and_torque` (`test_selffieldforces.py:1073`); `test_Taylor` / `test_Taylor_broad_upstream_sweep` (`test_selffieldforces.py:2000,2012`). |
| current | `src/simsopt/util/coil_optimization_helper_functions.py:134,183,184,708,727,864` | helper / config paths consuming `regularization_circ` | `wrapper_only` | Exercised indirectly by `tests/field/test_selffieldforces.py::test_objectives_time` and downstream integration fixtures. No new item 02 surface introduced. |

## Repository test rows (`tests/`)

Every test that mentions a selffield symbol in the repo tree (output of
the repo `git grep` above), grouped by file:

### `tests/field/test_coil.py`

| Path / Node | Intent | Classification | Evidence |
| --- | --- | --- | --- |
| `tests/field/test_coil.py:17` | imports `regularization_circ` for `RegularizedCoil` serialization tests | `covered_by_unit_parity` | round-trip coverage at lines 336 and 369. |
| `tests/field/test_coil.py:336` | `coils_via_symmetries` with `regularization_circ(0.05)` | `covered_by_unit_parity` | exercised by repo `test_coil.py` serialization round-trip. |
| `tests/field/test_coil.py:369` | `RegularizedCoil` subclass round-trip | `covered_by_unit_parity` | same. |

### `tests/field/test_selffieldforces.py` (36 tests; selffield-touching rows)

| Path / Node | Intent | Classification | Evidence |
| --- | --- | --- | --- |
| `tests/field/test_selffieldforces.py::test_k_square` (`:58`) | Closed-form `k(a, b)` for square cross-section | `covered_by_unit_parity` | Direct closed-form comparison. |
| `::test_delta_square` (`:64`) | Closed-form `delta(a, b)` for square cross-section | `covered_by_unit_parity` | Direct closed-form comparison. |
| `::test_symmetry` (`:70`) | `_rectangular_xsection_k`, `_rectangular_xsection_delta` symmetric under swap | `covered_by_unit_parity` | Algebraic identity. |
| `::test_limits` (`:83`) | Limiting-case behavior of rectangular helpers | `covered_by_unit_parity` | Limiting-case test. |
| `::test_regularization_circ` (`:109`) | `a**2 / sqrt(e)` scalar identity | `covered_by_unit_parity` | Direct identity. |
| `::test_regularization_rect` (`:125`) | Rectangular regularization symmetry / scale | `covered_by_unit_parity` | Direct properties. |
| `::test_regularization_functions_transform_under_strict_transfer_guard` (`:158`) | `jit` / `vmap` / `grad` under strict transfer guard | `covered_by_unit_parity` | `jax.transfer_guard("disallow")` context. |
| `::test_b_regularized_pure_jit_vmap_strict_transfer_guard_matches_wrapper` (`:191`) | Strict-transfer `jit` and `vmap` parity with wrapper | `covered_by_unit_parity` | `jax.transfer_guard("disallow")` context, device-resident inputs. |
| `::test_regularized_coil_self_field_methods_use_strict_transfer_boundary` (`:237`) | Public-wrapper strict boundary for `B_regularized()` / `self_force()` | `covered_by_unit_parity` | Both `CurveXYZFourier` and `JaxCurveXYZFourier` subtests. |
| `::test_circular_coil` (`:264`) | Closed-form circular / rectangular oracle (N_quad=500) | `covered_by_unit_parity` | Closed-form Eq. (98); also exercises `coil_coil_inductances`. |
| `::test_force_convergence` (`:677`) | Quadpoint independence of force objective | `covered_by_integration_parity` | Production force pipeline. |
| `::test_hsx_coil` (`:698`) | HSX coil (`nquadpoints=160`) parity vs CoilForces.jl | `covered_by_integration_parity` | Production-scale single-coil oracle. |
| `::test_coil_force_requires_regularized_coil` (`:1050`) | Force objective rejects non-regularized coils | `wrapper_only` | API guard. |
| `::test_net_force_and_torque` (`:1073`) | Net force / torque on `ncoils=4` | `covered_by_integration_parity` | Production-scale multi-coil. |
| `::test_force_objectives` (`:1191`) | Production force objectives, ncoils=4 | `covered_by_integration_parity` | Production-scale multi-coil. |
| `::test_force_and_torque_objectives_with_different_quadpoints` (`:1539`) | Mixed-quadpoint coil lists | `covered_by_integration_parity` | Production-scale. |
| `::test_downsample_must_divide_quadpoints` (`:1739`) | Argument guard | `wrapper_only` | API guard. |
| `::test_mixed_quadpoints_in_coil_lists_raises` (`:1788`) | Argument guard | `wrapper_only` | API guard. |
| `::test_Taylor` (`:2000`) | Taylor / finite-difference gradient parity, narrow | `covered_by_unit_parity` | FD-validated. |
| `::test_Taylor_broad_upstream_sweep` (`:2012`) | Taylor / FD parity, broad upstream sweep | `covered_by_unit_parity` | FD-validated, broad sweep. |
| `::test_objectives_time` (`:2022`) | Performance smoke | `wrapper_only` | Reporting only. |
| `::test_regularized_coil_requirement` (`:2173`) | Wrapper requirement | `wrapper_only` | API guard. |
| `::test_source_coils_coarse_and_fine` (`:2221`) | Coarse/fine source coil bookkeeping | `covered_by_integration_parity` | Production force pipeline. |
| `::test_force_objectives_reuse_packed_state_until_dofs_change` (`:2372`) | Cache reuse | `wrapper_only` | Runtime contract. |
| `::test_force_objectives_incrementally_refresh_only_dirty_source_coil` (`:2443`) | Cache incremental refresh | `wrapper_only` | Runtime contract. |
| `::test_force_objectives_share_coil_state_across_objectives` (`:2469`) | Cache sharing | `wrapper_only` | Runtime contract. |
| `::test_force_objectives_lazily_materialize_target_second_derivatives` (`:2498`) | Lazy materialization | `wrapper_only` | Runtime contract. |
| `::test_force_objectives_share_overlapping_reordered_coil_groups` (`:2535`) | Coil-group sharing | `wrapper_only` | Runtime contract. |
| `::test_shared_coil_state_reuses_precomputed_curve_specs_on_refresh` (`:2554`) | Spec reuse | `wrapper_only` | Runtime contract. |
| `::test_shared_coil_state_packs_composite_current_graphs_without_live_current_calls` (`:2578`) | Composite current packing | `wrapper_only` | Runtime contract. |
| `::test_shared_coil_state_uses_full_graph_dofs_for_filament_specs` (`:2602`) | Full-graph DOFs | `wrapper_only` | Runtime contract. |
| `::test_shared_coil_state_rejects_curves_without_spec_or_jax_path` (`:2635`) | Spec rejection | `wrapper_only` | API guard. |
| `::test_lpcurveforces_taylor_test` (`:2654`) | `LpCurveForce` Taylor test | `covered_by_integration_parity` | FD gradient parity for force objective. |
| `::test_circular_regularized_coil_subclass` (`:2703`) | `CircularRegularizedCoil` subclass | `covered_by_unit_parity` | Subclass round-trip. |
| `::test_rectangular_regularized_coil_subclass` (`:2782`) | `RectangularRegularizedCoil` subclass | `covered_by_unit_parity` | Subclass round-trip. |
| `::test_regularized_coil_methods_comprehensive` (`:2863`) | Comprehensive `B_regularized` shape / API | `covered_by_unit_parity` | Shape and method coverage. |

### `tests/field/test_selffield_item02_closeout.py` (new file — this item)

| Path / Node | Intent | Classification | Evidence |
| --- | --- | --- | --- |
| `tests/field/test_selffield_item02_closeout.py::test_b_regularized_pure_matches_circular_closed_form_oracle_at_production_scale` | Production-scale (ncoils=4, nquadpoints=128) closed-form circular-coil oracle vs `B_regularized_pure` | `covered_by_unit_parity` | imports `parity_ladder_tolerances("direct_kernel")`; runs under `jax.transfer_guard("disallow")`. |
| `::test_b_regularized_pure_wrong_regularization_breaks_closed_form_parity` | Negative control: wrong `a' = 1.5 a` produces ~6.3e-2 relative deviation, six orders above lane tolerance | `covered_by_unit_parity` | Sign-stable, tolerance-busting negative control. |
| `::test_b_regularized_pure_oracle_negative_control_runs_under_process_strict_guard` | Redundant strict transfer guard gate | `covered_by_unit_parity` | Context-manager strict guard. |
| `::test_b_regularized_pure_x_y_components_vanish_for_circular_coil[0-3]` | x/y components vanish at machine precision (4 parametrized cases) | `covered_by_unit_parity` | Direct kernel-tolerance assertion. |

## Upstream test rows (`/Users/suhjungdae/code/opensource/simsopt/tests`)

Audited at upstream SHA `1b0cc3a96063197cdbdd01559e04c25456fbe6ff`. The
upstream test set has 23 tests in `tests/field/test_selffieldforces.py`
(versus the repo's 36); the differences are jax-port additions (strict
transfer-guard, JAX curve types, force objective cache contracts).
Every upstream test maps cleanly onto a repo test of the same name (or a
repo superset of it):

| Path / Node | Intent | Classification | Evidence |
| --- | --- | --- | --- |
| `upstream:tests/field/test_selffieldforces.py::test_k_square` (`:41`) | Closed-form k value | `oracle_only` | Repo `test_k_square` covers identical kernel. |
| `upstream:::test_delta_square` (`:47`) | Closed-form delta value | `oracle_only` | Repo `test_delta_square`. |
| `upstream:::test_symmetry` (`:53`) | Rectangular helper symmetry | `oracle_only` | Repo `test_symmetry`. |
| `upstream:::test_limits` (`:66`) | Limiting cases of rectangular helpers | `oracle_only` | Repo `test_limits`. |
| `upstream:::test_regularization_circ` (`:84`) | `a^2 / sqrt(e)` | `oracle_only` | Repo `test_regularization_circ`. |
| `upstream:::test_regularization_rect` (`:100`) | Rectangular regularization | `oracle_only` | Repo `test_regularization_rect`. |
| `upstream:::test_circular_coil` (`:134`) | Closed-form circular / rectangular oracle | `oracle_only` | Repo `test_circular_coil`; jax-port superset adds strict-transfer subtests. |
| `upstream:::test_force_convergence` (`:465`) | Quadpoint convergence | `oracle_only` | Repo `test_force_convergence`. |
| `upstream:::test_hsx_coil` (`:482`) | HSX coil vs CoilForces.jl | `oracle_only` | Repo `test_hsx_coil`. |
| `upstream:::test_coil_force_requires_regularized_coil` (`:512`) | API guard | `oracle_only` | Repo equivalent (line 1050). |
| `upstream:::test_net_force_and_torque` (`:535`) | Net force / torque, ncoils=4 | `oracle_only` | Repo `test_net_force_and_torque`. |
| `upstream:::test_force_objectives` (`:613`) | Production force objectives | `oracle_only` | Repo `test_force_objectives`. |
| `upstream:::test_force_and_torque_objectives_with_different_quadpoints` (`:802`) | Mixed-quadpoint coils | `oracle_only` | Repo `test_force_and_torque_objectives_with_different_quadpoints`. |
| `upstream:::test_downsample_must_divide_quadpoints` (`:936`) | API guard | `oracle_only` | Repo `test_downsample_must_divide_quadpoints`. |
| `upstream:::test_mixed_quadpoints_in_coil_lists_raises` (`:977`) | API guard | `oracle_only` | Repo `test_mixed_quadpoints_in_coil_lists_raises`. |
| `upstream:::test_Taylor` (`:1017`) | Taylor / FD parity | `oracle_only` | Repo `test_Taylor`. |
| `upstream:::test_objectives_time` (`:1154`) | Performance smoke | `oracle_only` | Repo `test_objectives_time`. |
| `upstream:::test_regularized_coil_requirement` (`:1253`) | API guard | `oracle_only` | Repo `test_regularized_coil_requirement`. |
| `upstream:::test_source_coils_coarse_and_fine` (`:1299`) | Coarse/fine source coil bookkeeping | `oracle_only` | Repo `test_source_coils_coarse_and_fine`. |
| `upstream:::test_lpcurveforces_taylor_test` (`:1345`) | `LpCurveForce` Taylor test | `oracle_only` | Repo `test_lpcurveforces_taylor_test`. |
| `upstream:::test_circular_regularized_coil_subclass` (`:1374`) | `CircularRegularizedCoil` | `oracle_only` | Repo `test_circular_regularized_coil_subclass`. |
| `upstream:::test_rectangular_regularized_coil_subclass` (`:1453`) | `RectangularRegularizedCoil` | `oracle_only` | Repo `test_rectangular_regularized_coil_subclass`. |
| `upstream:::test_regularized_coil_methods_comprehensive` (`:1534`) | Comprehensive method coverage | `oracle_only` | Repo `test_regularized_coil_methods_comprehensive`. |
| `upstream:tests/field/test_coil.py:17` | `regularization_circ` consumer in coil tests | `oracle_only` | Repo `tests/field/test_coil.py` mirrors upstream. |

## Coverage completeness

- No matrix row is `unclassified`. Every row carries a classification.
- Every `covered_by_unit_parity` / `covered_by_integration_parity` row
  cites a JAX parity test path that resolves on disk at the commit's
  tree.
- `wrapper_only` rows are explicitly API guards or runtime cache
  contracts.
- `not_applicable` row (`Biot_savart_prefactor`) is annotated with the
  reason (one-shot import-time scalar).
- No `blocked` rows.

## Stale-upstream guard

`upstream_audit_sha` recorded at the top of this matrix matches the
upstream HEAD audited in the source SHA stamp.
