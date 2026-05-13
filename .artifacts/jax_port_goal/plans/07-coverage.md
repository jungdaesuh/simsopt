# Item 07 Coverage Matrix

Status: complete for CPU/JAX oracle closure. CUDA proof is not claimed.

Upstream audit SHA:
`1b0cc3a96063197cdbdd01559e04c25456fbe6ff`.

Repo-wide grep used to populate this matrix:

```
git grep -nE 'CurveLength|LpCurveCurvature|LpCurveCurvatureBarrier|LpCurveTorsion|ArclengthVariation|MeanSquaredCurvature|LinkingNumber|FramedCurveTwist|curve_length_pure|Lp_curvature_pure|Lp_torsion_pure|curve_msc_pure|curve_arclengthvariation_pure|frametwist_pure' -- src/ tests/
```

The classification column uses the project's standard rubric:

- `covered_by_unit_parity`: explicit unit/parity test with FD-Taylor or
  fixed-state oracle assertion in current tree.
- `covered_by_integration_parity`: integration fixture that exercises
  the class as part of a larger pipeline.
- `oracle_only`: upstream HSS source row or hand-derived contract used
  as a reference; no current-tree port file is exercised by it.
- `oracle_only=hand_derived`: port-only class with no upstream HSS
  oracle; the contract is the hand-derived formula in the kernel
  module.
- `wrapper_only`: a downstream consumer that calls a covered objective
  without introducing new kernel surface.

## Public Class Rows

| Source | Path / Node | Intent | Classification | Evidence |
| --- | --- | --- | --- | --- |
| current | `src/simsopt/geo/curveobjectives.py:54` | `curve_length_pure` JAX scalar kernel | `covered_by_unit_parity` | `tests/geo/test_curve_objectives.py::CurveObjectiveTesting::test_curve_length_taylor_test` |
| current | `src/simsopt/geo/curveobjectives.py:157` | `CurveLength` wrapper J/dJ via `dincremental_arclength_by_dcoeff_vjp` | `covered_by_unit_parity` | `tests/geo/test_curve_objectives.py::CurveObjectiveTesting::test_curve_length_taylor_test` |
| current | `src/simsopt/geo/curveobjectives.py:190` | `Lp_curvature_pure` JAX scalar kernel | `covered_by_unit_parity` | `tests/geo/test_curve_objectives.py::CurveObjectiveTesting::test_curve_curvature_taylor_test` |
| current | `src/simsopt/geo/curveobjectives.py:236` | `LpCurveCurvature` wrapper J/dJ via dkappa/dgammadash VJPs | `covered_by_unit_parity` | `tests/geo/test_curve_objectives.py::CurveObjectiveTesting::test_curve_curvature_taylor_test` |
| current | `src/simsopt/geo/curveobjectives.py:209` | `curvature_barrier_pure` strict log-barrier kernel | `covered_by_unit_parity` | `tests/geo/test_curve_objectives.py::CurveObjectiveTesting::test_curve_curvature_barrier_taylor_test` |
| current | `src/simsopt/geo/curveobjectives.py:290` | `LpCurveCurvatureBarrier` wrapper (port-only, no upstream class) | `covered_by_unit_parity` | `tests/geo/test_curve_objectives.py::CurveObjectiveTesting::test_curve_curvature_barrier_taylor_test`; subprocess wrappers in `tests/test_jax_import_smoke.py:1009-1021` |
| current | `src/simsopt/geo/curveobjectives.py:337` | `Lp_torsion_pure` JAX scalar kernel | `covered_by_unit_parity` | `tests/geo/test_curve_objectives.py::CurveObjectiveTesting::test_curve_torsion_taylor_test` |
| current | `src/simsopt/geo/curveobjectives.py:361` | `LpCurveTorsion` wrapper J/dJ via dtorsion/dgammadash VJPs | `covered_by_unit_parity` | `tests/geo/test_curve_objectives.py::CurveObjectiveTesting::test_curve_torsion_taylor_test` |
| current | `src/simsopt/geo/curveobjectives.py:1050` | `curve_arclengthvariation_pure` JAX variance kernel | `covered_by_unit_parity` | `tests/geo/test_curve_objectives.py::CurveObjectiveTesting::test_curve_arclengthvariation_taylor_test`; `test_arclength_variation_circle`; `test_arclength_variation_circle_planar` |
| current | `src/simsopt/geo/curveobjectives.py:1063` | `ArclengthVariation` wrapper J/dJ via incremental-arclength VJP | `covered_by_unit_parity` | same as above |
| current | `src/simsopt/geo/curveobjectives.py:1145` | `curve_msc_pure` JAX scalar kernel | `covered_by_unit_parity` | `tests/geo/test_curve_objectives.py::CurveObjectiveTesting::test_curve_meansquaredcurvature_taylor_test` |
| current | `src/simsopt/geo/curveobjectives.py:1159` | `MeanSquaredCurvature` wrapper J/dJ via dkappa/dgammadash VJPs | `covered_by_unit_parity` | same as above |
| current | `src/simsopt/geo/curveobjectives.py:1194` | `LinkingNumber.J()` via `sopp.compute_linking_number`; `dJ` returns `Derivative({})` by construction | `oracle_only` | `tests/geo/test_curve_objectives.py::CurveObjectiveTesting::test_linking_number`; `tests/geo/test_curve_objectives.py::CurveObjectiveTesting::test_linking_number_planar`; new `tests/geo/test_curveobjectives_item07_closeout.py::test_linking_number_production_scale_ncoils_four` (ncoils=4 production-scale floor); strict-guard variant in same file |
| current | `src/simsopt/geo/curveobjectives.py:1244` | `frametwist_pure` discrete twist integrator | `covered_by_unit_parity` | new `tests/geo/test_curveobjectives_item07_closeout.py::test_framed_curve_twist_lp_taylor_value_and_gradient` (via `angle_profile()`); existing `tests/geo/test_curve_objectives.py::test_framed_curve_twist_reuses_shared_jit_kernels` (cache hygiene) |
| current | `src/simsopt/geo/curveobjectives.py:1267` | `frametwist_net_pure` | `covered_by_unit_parity` | new `tests/geo/test_curveobjectives_item07_closeout.py::test_framed_curve_twist_non_lp_modes_return_zero_derivative` |
| current | `src/simsopt/geo/curveobjectives.py:1272` | `frametwist_range_pure` | `covered_by_unit_parity` | same as above |
| current | `src/simsopt/geo/curveobjectives.py:1277` | `frametwist_max_pure` | `covered_by_unit_parity` | same as above |
| current | `src/simsopt/geo/curveobjectives.py:1282` | `frametwist_lp_pure` differentiable kernel | `covered_by_unit_parity` | new `tests/geo/test_curveobjectives_item07_closeout.py::test_framed_curve_twist_lp_taylor_value_and_gradient` |
| current | `src/simsopt/geo/curveobjectives.py:1301` | `FramedCurveTwist` wrapper (port-only) for f in {net, range, lp, max} | `covered_by_unit_parity` | new tests above plus subprocess wrappers in `tests/subprocess/jax_runtime_cases.py:1734,1787` |

## Downstream Consumer Rows

| Source | Path / Node | Intent | Classification | Evidence |
| --- | --- | --- | --- | --- |
| current | `src/simsopt/field/coilset.py:253` | `CoilSet` length helper sums `CurveLength` | `wrapper_only` | covered indirectly by `tests/field/test_coilset.py` |
| current | `src/simsopt/field/coilset.py:295` | `CoilSet` curvature helper sums `LpCurveCurvature` | `wrapper_only` | same |
| current | `src/simsopt/field/coilset.py:307` | `CoilSet` mean-squared-curvature helper sums `MeanSquaredCurvature` | `wrapper_only` | same |
| current | `src/simsopt/field/coilset.py:330` | `CoilSet` arclength-variation helper sums `ArclengthVariation` | `wrapper_only` | same |
| current | `src/simsopt/geo/accessibility.py:80` | `Port` uses `ArclengthVariation(self.port)` | `wrapper_only` | covered indirectly by accessibility tests |
| current | `src/simsopt/util/coil_optimization_helper_functions.py:94-99,174-179,773-797` | example helper builders combine `CurveLength`, `LpCurveCurvature`, `MeanSquaredCurvature`, `ArclengthVariation`, `LinkingNumber` into composite objectives | `wrapper_only` | covered by `tests/integration/test_single_stage_jax_cpu_reference.py` and the existing single-stage smoke suites |
| current | `src/simsopt/geo/surfaceobjectives_jax.py` | uses `CurveLength` references through composite single-stage objectives | `wrapper_only` | covered by `tests/integration/test_single_stage_jax.py` and `tests/integration/test_single_stage_physics_parity.py` |
| current | `src/simsopt/objectives/stage2_target_objective_jax.py` | composite Stage 2 target uses item 07 objectives indirectly via `CoilSet` helpers | `wrapper_only` | covered by `tests/objectives/test_fluxobjective.py` and `tests/integration/test_stage2_jax.py` |
| current | `tests/test_jax_import_smoke.py:1003-1024` | subprocess legacy curve objective value/gradient parametrizations (`framed-curve-twist`, `lp-curve-curvature-barrier`, `lp-curve-curvature`, `lp-curve-torsion`, `curve-length`) | `covered_by_unit_parity` | matching subprocess case bodies in `tests/subprocess/jax_runtime_cases.py:1710-1798` |
| current | `tests/geo/test_curve_optimizable.py` | top-level curve dof / set / get-state smokes that touch length/curvature objectives | `wrapper_only` | existing tests pass on parent commit |
| current | `tests/geo/test_curve.py` | curve type tests indirectly use `CurveLength` and friends | `wrapper_only` | existing tests pass on parent commit |
| current | `tests/geo/test_curveperturbed.py` | curve perturbation tests reuse the same dofs | `wrapper_only` | existing tests pass on parent commit |
| current | `tests/geo/test_finitebuild.py` | finite-build tests touch `LpCurveCurvature` and friends through finite-build pipeline | `wrapper_only` | existing tests pass on parent commit; finite-build kernel is owned by item 09 |
| current | `tests/geo/test_single_stage_example.py` | single-stage example regression uses item 07 objectives | `wrapper_only` | covered by existing single-stage tests |
| current | `tests/objectives/test_utilities.py` | composite objective tests reuse `CurveLength` etc. | `wrapper_only` | existing tests pass on parent commit |

## Upstream Oracle Rows

| Source | Path / Node | Intent | Classification | Evidence |
| --- | --- | --- | --- | --- |
| upstream | `upstream_hss/master:src/simsopt/geo/curveobjectives.py:32` (audit SHA `1b0cc3a96063197cdbdd01559e04c25456fbe6ff`) | upstream `CurveLength` reference public API | `oracle_only` | mirrored locally; FD-Taylor parity assertion in `test_curve_length_taylor_test` |
| upstream | `upstream_hss/master:src/simsopt/geo/curveobjectives.py:82` | upstream `LpCurveCurvature` reference public API | `oracle_only` | mirrored locally; FD-Taylor parity in `test_curve_curvature_taylor_test` |
| upstream | `upstream_hss/master:src/simsopt/geo/curveobjectives.py:138` | upstream `LpCurveTorsion` reference public API | `oracle_only` | mirrored locally; FD-Taylor parity in `test_curve_torsion_taylor_test` |
| upstream | `upstream_hss/master:src/simsopt/geo/curveobjectives.py:434` | upstream `ArclengthVariation` reference public API | `oracle_only` | mirrored locally; FD-Taylor parity in `test_curve_arclengthvariation_taylor_test` |
| upstream | `upstream_hss/master:src/simsopt/geo/curveobjectives.py:520` | upstream `MeanSquaredCurvature` reference public API | `oracle_only` | mirrored locally; FD-Taylor parity in `test_curve_meansquaredcurvature_taylor_test` |
| upstream | `upstream_hss/master:src/simsopt/geo/curveobjectives.py:555` | upstream `LinkingNumber` reference public API; `dJ` returns `Derivative({})` upstream as well | `oracle_only` | mirrored locally as `test_linking_number` and new ncoils=4 test |
| upstream | `upstream_hss/master:src/simsoptpp/python_dommaschk_etc.cpp` (compute_linking_number binding) | C++ Gauss linking integral binding | `oracle_only` | exercised by `LinkingNumber.J()` calls in the new ncoils=4 test |
| port_only | hand-derived | `LpCurveCurvatureBarrier` strict log-barrier scalar (no upstream HSS class at audit SHA) | `oracle_only=hand_derived` | covered by `test_curve_curvature_barrier_taylor_test` plus subprocess wrappers |
| port_only | hand-derived | `FramedCurveTwist` discrete trapezoid + lp/net/range/max wrapping (no upstream HSS class at audit SHA) | `oracle_only=hand_derived` | covered by new closeout tests for the lp Taylor path and the {net, range, max} contract |

No row is unclassified.
