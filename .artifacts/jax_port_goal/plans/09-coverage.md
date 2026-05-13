# Item 09 Coverage Matrix

Status: complete for CPU/JAX oracle closure. CUDA proof is not claimed.

Upstream audit SHA:
`1b0cc3a96063197cdbdd01559e04c25456fbe6ff`.

Grep used to populate the matrix:

```
git grep -nE 'B2Energy|NetFluxes|SquaredMeanForce|LpCurveForce|LpCurveTorque|SquaredMeanTorque|b2energy_pure|squared_mean_force_pure|lp_force_pure|lp_torque_pure|squared_mean_torque|_SharedCoilState' -- tests src benchmarks docs
```

| Source | Path / Node | Intent | Classification | Evidence |
| --- | --- | --- | --- | --- |
| current | `src/simsopt/field/force.py:953` | Module-level `jit(_b2energy_eval, static_argnums=(3,))` entrypoint | `covered_by_unit_parity` | `tests/field/test_selffieldforces.py::test_Taylor`, `tests/field/test_force_item09_closeout.py::test_lp_curve_force_production_scale_taylor_parity_under_strict_transfer_guard` |
| current | `src/simsopt/field/force.py:955` | Module-level `jit(_net_ext_flux_eval, static_argnums=(2,))` entrypoint | `covered_by_unit_parity` | `tests/field/test_selffieldforces.py::test_Taylor` NetFluxes Taylor row |
| current | `src/simsopt/field/force.py:957` | Module-level `jit(_squared_mean_force_eval, static_argnums=(9,))` entrypoint | `covered_by_unit_parity` | `tests/field/test_selffieldforces.py::test_force_objectives`, `test_Taylor` |
| current | `src/simsopt/field/force.py:962` | Module-level `jit(_lp_force_eval, static_argnums=(14,))` entrypoint | `covered_by_unit_parity` | `tests/field/test_force_item09_closeout.py::test_lp_curve_force_production_scale_taylor_parity_under_strict_transfer_guard`, `tests/field/test_selffieldforces.py::test_force_objectives`, `test_Taylor` |
| current | `src/simsopt/field/force.py:967` | Module-level `jit(_lp_torque_eval, static_argnums=(14,))` entrypoint | `covered_by_unit_parity` | `tests/field/test_selffieldforces.py::test_force_objectives`, `test_Taylor` |
| current | `src/simsopt/field/force.py:972` | Module-level `jit(_squared_mean_torque_eval, static_argnums=(9,))` entrypoint | `covered_by_unit_parity` | `tests/field/test_selffieldforces.py::test_Taylor` |
| current | `src/simsopt/field/force.py:1187` | `b2energy_pure` kernel | `covered_by_unit_parity` | `tests/field/test_selffieldforces.py::test_Taylor` B2Energy Taylor row |
| current | `src/simsopt/field/force.py:1229` | `B2Energy(Optimizable)` wrapper | `covered_by_unit_parity` | `tests/field/test_selffieldforces.py::test_force_objectives_reuse_packed_state_until_dofs_change`, `test_Taylor`, `test_regularized_coil_requirement` |
| current | `src/simsopt/field/force.py:1411` | `net_ext_fluxes_pure` kernel | `covered_by_unit_parity` | `tests/field/test_selffieldforces.py::test_Taylor` NetFluxes row |
| current | `src/simsopt/field/force.py:1457` | `NetFluxes(Optimizable)` wrapper | `covered_by_unit_parity` | `tests/field/test_selffieldforces.py::test_Taylor` NetFluxes row, plus shared-state tests at 2469, 2498, 2535 |
| current | `src/simsopt/field/force.py:1555` | `squared_mean_force_pure` kernel | `covered_by_unit_parity` | `tests/field/test_selffieldforces.py::test_force_objectives`, `test_Taylor` SquaredMeanForce rows |
| current | `src/simsopt/field/force.py:1699` | `SquaredMeanForce(Optimizable)` wrapper | `covered_by_unit_parity` | `tests/field/test_selffieldforces.py::test_force_objectives`, `test_Taylor`, `test_source_coils_coarse_and_fine`, `test_force_objectives_reuse_packed_state_until_dofs_change` |
| current | `src/simsopt/field/force.py:1889` | `lp_force_pure` kernel | `covered_by_unit_parity` | `tests/field/test_force_item09_closeout.py::test_lp_curve_force_production_scale_taylor_parity_under_strict_transfer_guard`, `tests/field/test_selffieldforces.py::test_force_objectives` |
| current | `src/simsopt/field/force.py:2059` | `LpCurveForce(Optimizable)` wrapper | `covered_by_unit_parity` | `tests/field/test_force_item09_closeout.py::test_lp_curve_force_production_scale_taylor_parity_under_strict_transfer_guard`, `tests/field/test_selffieldforces.py::test_force_objectives`, `test_Taylor`, `test_lpcurveforces_taylor_test`, `test_force_objectives_share_overlapping_reordered_coil_groups` |
| current | `src/simsopt/field/force.py:2279` | `lp_torque_pure` kernel | `covered_by_unit_parity` | `tests/field/test_selffieldforces.py::test_force_objectives`, `test_Taylor` LpCurveTorque row |
| current | `src/simsopt/field/force.py:2446` | `LpCurveTorque(Optimizable)` wrapper | `covered_by_unit_parity` | `tests/field/test_selffieldforces.py::test_force_objectives`, `test_Taylor`, `test_force_and_torque_objectives_with_different_quadpoints` |
| current | `src/simsopt/field/force.py:2666` | `squared_mean_torque` kernel | `covered_by_unit_parity` | `tests/field/test_selffieldforces.py::test_force_objectives`, `test_Taylor` SquaredMeanTorque row |
| current | `src/simsopt/field/force.py:2799` | `SquaredMeanTorque(Optimizable)` wrapper | `covered_by_unit_parity` | `tests/field/test_selffieldforces.py::test_force_objectives`, `test_Taylor` |
| current | `src/simsopt/field/force.py:979` | `_coil_coil_inductances_pure` kernel | `covered_by_unit_parity` | `tests/field/test_selffieldforces.py` inductance / mutual / self circuit tests around line 318-650 |
| current | `src/simsopt/field/force.py:1062` | `_coil_coil_inductances_inv_pure` kernel | `covered_by_unit_parity` | `tests/field/test_selffieldforces.py` inductance pseudo-inverse tests |
| current | `src/simsopt/field/force.py:1125` | `_induced_currents_pure` kernel | `covered_by_unit_parity` | `tests/field/test_selffieldforces.py` induced-currents row |
| current | `src/simsopt/field/force.py:480` | `_CoilStateEntry` dataclass | `owned_dependency` | exercised by every objective wrapper; classified as fork-only pre-compute layer with no upstream counterpart |
| current | `src/simsopt/field/force.py:656` | `_SharedCoilState` packed per-coil cache | `owned_dependency` | `tests/field/test_selffieldforces.py::test_shared_coil_state_reuses_precomputed_curve_specs_on_refresh` (line 2554), `test_shared_coil_state_packs_composite_current_graphs_without_live_current_calls` (2578), `test_shared_coil_state_uses_full_graph_dofs_for_filament_specs` (2602), `test_shared_coil_state_rejects_curves_without_spec_or_jax_path` (2635) |
| current | `src/simsopt/field/force.py:731` | `_CoilStateGroupCache` group view | `owned_dependency` | `tests/field/test_selffieldforces.py::test_force_objectives_share_overlapping_reordered_coil_groups` (line 2535), `test_force_objectives_share_coil_state_across_objectives` (2469) |
| current | `src/simsopt/field/force.py:764` | `_invalidate_objective_state` cache invalidator | `owned_dependency` | `tests/field/test_selffieldforces.py::test_force_objectives_reuse_packed_state_until_dofs_change` (2372), `test_force_objectives_incrementally_refresh_only_dirty_source_coil` (2443) |
| current | `src/simsopt/field/force.py:781` | `_assemble_curve_current_derivative` projection | `owned_dependency` | `tests/field/test_force_item09_closeout.py` (FD parity covers the derivative path), `tests/field/test_selffieldforces.py::test_force_objectives` (gradient shape check) |
| current | `src/simsopt/field/force.py:811` | `_cached_objective_args` host cache | `owned_dependency` | `tests/field/test_selffieldforces.py::test_force_objectives_reuse_packed_state_until_dofs_change` (2372) |
| upstream | `upstream_hss/master:src/simsopt/field/force.py` (audit SHA `1b0cc3a9`) | Upstream JAX kernels and wrapper bodies for B2Energy, NetFluxes, SquaredMeanForce, LpCurveForce, LpCurveTorque, SquaredMeanTorque | `oracle_only` | upstream test parity is provided by `tests/field/test_selffieldforces.py::test_Taylor_broad_upstream_sweep`; fork keeps the same compute, additive pre-compute layer only |
| upstream | `upstream_hss/master:docs/source/example_coils.rst` | Public-facing documentation lists each of the six force/torque/energy objectives as a public class | `oracle_only` | docs surface preserved unchanged; downstream wrappers still find `simsopt.field.LpCurveForce` etc. |
| current | `docs/source/example_coils.rst:377-381` | Local docs row enumerating six public force/torque/energy objectives | `wrapper_only` | matches the upstream public surface; no change required by item 09 |
| current | `src/simsopt/util/coil_optimization_helper_functions.py:133` | Downstream `LpCurveForce` consumer in `coil_optimization_helper_functions` | `wrapper_only` | covered by `tests/util/test_coil_optimization_helper_functions.py::TestCoilOptimizationHelperFunctions::test_initial_optimizations_force_objective` |
| current | `tests/util/test_coil_optimization_helper_functions.py:13` | Downstream wrapper test exercising LpCurveForce | `wrapper_only` | already collected by the existing pytest suite |
| current | `tests/test_jax_import_smoke.py:1127` | Cross-environment LpCurveForce shared-state transfer-guard smoke | `covered_by_unit_parity` | runs `tests/subprocess/import_smoke_cases.py:1337` LpCurveForce subprocess case |
| current | `tests/subprocess/import_smoke_cases.py:1337` | Subprocess LpCurveForce smoke with `RegularizedCoil` and `SIMSOPT_JAX_TRANSFER_GUARD=disallow` | `covered_by_unit_parity` | wrapped through `test_jax_import_smoke.py:1127` |
| current | `tests/field/test_force_item09_closeout.py` | Production-scale LpCurveForce Taylor parity test under strict transfer guard | `covered_by_unit_parity` | new closeout test using `parity_ladder_tolerances("fd_gradient")` |

No matrix row is `unclassified`. Item 09 has no `blocked` row, no `not_applicable`
row, and no row that depends on a missing upstream parity oracle.

Notes on grep deduplication: the grep also surfaces 234 raw match lines across
`tests/`, `src/`, `benchmarks/`, and `docs/`. Each match maps to one of the
rows above through its owning class/kernel. Coverage rows above merge
multi-line matches that resolve to the same JAX kernel or wrapper.
