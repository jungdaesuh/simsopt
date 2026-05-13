# Item 05 Coverage Matrix

Status: complete for CPU/JAX oracle closure on seven curve classes that already
expose immutable specs. CUDA proof is not claimed.

Upstream audit SHA:
`1b0cc3a96063197cdbdd01559e04c25456fbe6ff`.

## Curve / spec coverage rows

| Source | Path / Node | Intent | Classification | Evidence |
| --- | --- | --- | --- | --- |
| current | `src/simsopt/geo/curvexyzfourier.py:170-178` | `CurveXYZFourier.to_spec` -> `CurveXYZFourierSpec` | `covered_by_unit_parity` | `tests/integration/test_single_stage_jax_cpu_reference.py::TestBiotSavartJAXReference::test_curvexyzfourier_spec_pullback_matches_curve_methods`; `tests/field/test_biotsavart_jax_parity.py::TestCurveTypeParametrization` (ncoils=1 nquad=100) |
| current | `src/simsopt/geo/curverzfourier.py:131-141` | `CurveRZFourier.to_spec` -> `CurveRZFourierSpec` | `covered_by_unit_parity` | `tests/integration/test_single_stage_jax_cpu_reference.py::TestBiotSavartJAXReference::test_curverzfourier_spec_pullback_matches_curve_methods`; `tests/field/test_biotsavart_jax_parity.py::TestCurveTypeParametrization` (ncoils=1 nquad=100) |
| current | `src/simsopt/geo/curveplanarfourier.py:185-193` | `CurvePlanarFourier.to_spec` -> `CurvePlanarFourierSpec` | `covered_by_unit_parity` | `tests/integration/test_single_stage_jax_cpu_reference.py::TestBiotSavartJAXReference::test_curveplanarfourier_spec_pullback_matches_curve_methods`; `tests/field/test_biotsavart_jax_parity.py::TestCurveTypeParametrization` (ncoils=1 nquad=100) |
| current | `src/simsopt/geo/curvehelical.py:138-150` | `CurveHelical.to_spec` -> `CurveHelicalSpec` | `covered_by_unit_parity` | `tests/integration/test_single_stage_jax_cpu_reference.py::TestBiotSavartJAXReference::test_curvehelical_spec_pullback_matches_curve_methods`; `tests/field/test_biotsavart_jax_parity.py::TestCurveTypeParametrization` (ncoils=1 nquad=100) |
| current | `src/simsopt/geo/curveperturbed.py:260-289` | `CurvePerturbed.to_spec` -> `CurvePerturbedSpec` | `covered_by_unit_parity` | `tests/integration/test_single_stage_jax_cpu_reference.py::TestBiotSavartJAXReference::test_curveperturbed_spec_pullback_matches_curve_methods` |
| current | `src/simsopt/geo/curvecwsfourier.py:172-228` | `CurveCWSFourierCPP` -> RZ surface fallback in `curve_spec_from_curve` -> `CurveCWSFourierRZSpec` | `covered_by_unit_parity` | `tests/integration/test_single_stage_jax_cpu_reference.py::TestBiotSavartJAXReference::test_curvecwsfouriercpp_spec_pullback_matches_curve_and_surface_methods` |
| current | `src/simsopt/geo/curve.py:2008-2088` | `CurveCWSFourier` (JAX) -> RZ surface fallback in `curve_spec_from_curve` -> `CurveCWSFourierRZSpec` | `covered_by_unit_parity` | `tests/integration/test_single_stage_jax_cpu_reference.py::TestBiotSavartJAXReference::test_curvecwsfourier_spec_pullback_matches_curve_and_surface_methods` |
| current | `src/simsopt/geo/finitebuild.py:208-238` | `CurveFilament.to_spec` -> `CurveFilamentSpec` | `covered_by_unit_parity` | `tests/integration/test_single_stage_jax_cpu_reference.py::TestBiotSavartJAXReference::test_curvefilament_spec_pullback_matches_curve_methods` |
| current | `src/simsopt/geo/curvexyzfouriersymmetries.py:60-162` | `CurveXYZFourierSymmetries` (no `to_spec`) | `blocked` | `tests/geo/test_curve_item05_closeout.py::test_curvexyzfouriersymmetries_spec_routing_is_documented_blocker` documents the architecture limitation and skips with explicit blocker text. The pure-JAX `gamma_pure` path is itself jit-compiled JAX, so existing `tests/geo/test_curve.py` indirectly exercises the kernel via `curve.gamma()`. |

## Production-scale floor rows (item 05 new tests)

| Curve class | ncoils | nquadpoints | Test |
| --- | --- | --- | --- |
| `CurveXYZFourier` | 4 | 64 | `tests/geo/test_curve_item05_closeout.py::test_curve_spec_pullback_production_scale_parity[CurveXYZFourier]` |
| `CurveRZFourier` | 4 | 64 | `tests/geo/test_curve_item05_closeout.py::test_curve_spec_pullback_production_scale_parity[CurveRZFourier]` |
| `CurvePlanarFourier` | 4 | 64 | `tests/geo/test_curve_item05_closeout.py::test_curve_spec_pullback_production_scale_parity[CurvePlanarFourier]` |
| `CurveHelical` | 4 | 64 | `tests/geo/test_curve_item05_closeout.py::test_curve_spec_pullback_production_scale_parity[CurveHelical]` |

## Existing parity coverage cross-references

| Source | Path / Node | Intent | Classification | Evidence |
| --- | --- | --- | --- | --- |
| current | `tests/field/test_biotsavart_jax_parity.py:719-829` | `_CURVE_SPEC_FACTORIES` parametrization across XYZ/RZ/Helical/Planar at ncoils=1, nquad=100 | `covered_by_unit_parity` | `TestCurveTypeParametrization` 5 tests x 4 curve types = 20 cases, all under the same `parity_ladder_tolerances("direct_kernel")` regime |
| current | `tests/integration/test_single_stage_jax_cpu_reference.py:3056-3163` | spec pullback against curve methods for 8 variants | `covered_by_unit_parity` | 8 distinct `test_curve*_spec_pullback_matches_curve_methods` cases plus the additional spec-exposure and live-curve geometry tests |
| current | `tests/geo/test_curve.py` (`curvetypes` list, ll. 156-194) | upstream-style CPU parametrization including `CurveXYZFourierSymmetries{1,2,3}` | `oracle_only` | `Testing` class exercises `curve.gamma()` parity, serialization, Taylor tests etc. |
| upstream | `upstream_hss/master:src/simsopt/geo/curvexyzfouriersymmetries.py` | upstream public `CurveXYZFourierSymmetries` API and `gamma_pure` | `oracle_only` | inline `curve.gamma()` parity; the upstream class itself has no immutable JAX spec and is not part of the upstream public spec contract |

No matrix row is unclassified. The `blocked` row for
`CurveXYZFourierSymmetries` is the one architecture-limitation entry and is
recorded under section 5 of the goal prompt as an `architecture` candidate;
the test docstring restates the blocker and the source pointer.
