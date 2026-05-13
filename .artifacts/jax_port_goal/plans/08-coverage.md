# Item 08 Coverage Matrix

Upstream audit SHA: `1b0cc3a96063197cdbdd01559e04c25456fbe6ff`.

Grep commands re-run at item 08 close time:

- `git -C /Users/suhjungdae/code/columbia/simsopt-jax grep -nE
  "strain_optimization|LPBinormalCurvatureStrainPenalty|LPTorsionalStrainPenalty|CoilStrain|torstrain_pure|binormstrain_pure"
  tests/ src/`
- `git -C /Users/suhjungdae/code/opensource/simsopt grep -nE
  "strain_optimization|LPBinormalCurvatureStrainPenalty|LPTorsionalStrainPenalty|CoilStrain"
  tests/`

Both `tests/` greps return a single test file each. No `simsoptpp_symbol`
exists for this item (Python+JAX-only module).

| Source | Path / Node | Intent | Classification | Evidence |
| --- | --- | --- | --- | --- |
| current | `tests/geo/test_strainopt.py::CoilStrainTesting::test_strain_opt` | Circular-coil rotation-DOF optimization drives `LPTorsional*` + `LPBinormal*` penalties to zero | `covered_by_integration_parity` | Wrapper J/dJ exercised through `scipy.optimize.minimize`; asserts both penalties < 1e-12 after 100 BFGS iterations. Already-green on HEAD: `3 passed, 8 subtests passed`. |
| current | `tests/geo/test_strainopt.py::CoilStrainTesting::test_torsion` (and `subtest_torsion`) | Production-scale NCSX (`coil_order=6, points_per_period=120`) FD-gradient parity for `LPTorsionalStrainPenalty.dJ` | `covered_by_integration_parity` | Central-difference contraction test against `dJ()` on NCSX coil 0 at the production-scale grid. Already-green on HEAD. |
| current | `tests/geo/test_strainopt.py::CoilStrainTesting::test_binormal_curvature` (and `subtest_binormal_curvature`) | Production-scale FD-gradient parity for `LPBinormalCurvatureStrainPenalty.dJ` plus the Frenet-zero-rotation zero-strain control | `covered_by_integration_parity` | Same NCSX `coil_order=6, points_per_period=120` fixture; FD contraction when frame is non-trivial; `J() < 1e-12` on Frenet+ZeroRotation. Already-green on HEAD. |
| current | `tests/geo/test_strainopt_item08_closeout.py::test_lp_torsional_penalty_production_scale_matches_numpy_reference_under_strict_guard` | Strict-transfer-guard production-scale `J()` parity vs host NumPy reference; `dJ()` shape+finiteness invariance under device transitions | `covered_by_unit_parity` | New item-08 test. Imports tolerance from `parity_ladder_tolerances("direct_kernel")`. Red-step on parent `a9da18fac` failed with `Disallowed host-to-device transfer`; HEAD passes under and without `SIMSOPT_JAX_TRANSFER_GUARD=disallow`. |
| current | `tests/geo/test_strainopt_item08_closeout.py::test_lp_binormal_penalty_zero_twist_circle_vanishes_in_frenet_frame` | Negative control: zero-twist circular `CurveXYZFourier`+Frenet+ZeroRotation yields binormal strain at the `direct-kernel` floor | `covered_by_unit_parity` | New item-08 test. Same `parity_ladder_tolerances("direct_kernel")` tolerance source. Red-step on parent failed at the strict-guard boundary; HEAD passes under and without `SIMSOPT_JAX_TRANSFER_GUARD=disallow`. |
| upstream | `tests/geo/test_strainopt.py::CoilStrainTesting::test_strain_opt` | Circular-coil rotation-DOF optimization to zero strain | `oracle_only` | Upstream Python+JAX test file is the parity oracle for the math contract; the current-repo file is a refactored superset that already covers the same assertions. |
| upstream | `tests/geo/test_strainopt.py::CoilStrainTesting::test_torsion` | Production-scale `LPTorsionalStrainPenalty` FD-gradient parity | `oracle_only` | Same NCSX `coil_order=6, points_per_period=120` fixture as the current-repo file; the current `subtest_torsion` is the JAX-side equivalent. |
| upstream | `tests/geo/test_strainopt.py::CoilStrainTesting::test_binormal_curvature` | Production-scale `LPBinormalCurvatureStrainPenalty` FD-gradient parity + Frenet zero control | `oracle_only` | Same fixture and control as upstream `test_torsion`; current `subtest_binormal_curvature` matches. |
| current | `src/simsopt/geo/__init__.py:49` | `"strain_optimization"` package export | `not_applicable` | Module re-export, not a parity oracle. |
| upstream | `src/simsopt/geo/strain_optimization.py:1-193` | Upstream Python+JAX module that owns the math contract | `oracle_only` | Used as the math oracle; the current-repo refactor only changes jit-closure scope and the `gammadash` host-staging boundary. |

No row is `unclassified`. Empty-oracle case does not apply: every
upstream symbol has a directly-mapped current-repo coverage row, and the
upstream `tests/geo/test_strainopt.py` test file is the math oracle for
the integrand / pointwise strain definitions.

No matrix row cites a test that is decorated with
`@pytest.mark.skip`, `@pytest.mark.skipif`, `@pytest.mark.xfail`, or
wrapped in a module / class / function-scope `pytest.skip(...)`. Every
cited path resolves on disk at HEAD and is collected by `pytest
--collect-only`.
