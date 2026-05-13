# Item 13 Coverage Matrix

Status: complete for CPU/JAX oracle closure. CUDA proof is not claimed.

Upstream audit SHA: `1b0cc3a96063197cdbdd01559e04c25456fbe6ff`.

| Source | Path / Node | Intent | Classification | Evidence |
| --- | --- | --- | --- | --- |
| current | `src/simsoptpp/regular_grid_interpolant_3d.h:67-309` | C++ `RegularGridInterpolant3D` class header | `oracle_only` | parity oracle consumed by `simsoptpp.RegularGridInterpolant3D` from JAX cross-oracle tests |
| current | `src/simsoptpp/regular_grid_interpolant_3d_impl.h:14-252` | C++ `interpolate_batch`, `evaluate_batch`, `evaluate_local`, `estimate_error`, `linspace` implementations | `oracle_only` | parity oracle; JAX kernel matches the same math contract within `direct_kernel` rtol/atol |
| current | `src/simsoptpp/regular_grid_interpolant_3d_c.cpp` | C++ template instantiation | `oracle_only` | binding glue; consumed transitively through the Python binding |
| current | `src/simsoptpp/regular_grid_interpolant_3d_py.cpp` | pybind11 binding for `RegularGridInterpolant3D`, `UniformInterpolationRule`, `ChebyshevInterpolationRule` | `oracle_only` | the JAX cross-oracle tests construct the C++ binding directly through `simsoptpp` |
| current | `src/simsopt/jax_core/regular_grid_interp.py` | new JAX kernel: rules, spec, build, evaluate, error | `covered_by_unit_parity` | `tests/jax_core/test_regular_grid_interp_item13.py` (28 tests, all green) |
| current | `tests/field/test_interpolant.py::Testing::test_regular_grid_interpolant_exact` | upstream-style closed-form polynomial-exactness test against `simsoptpp` | `covered_by_unit_parity` | `tests/jax_core/test_regular_grid_interp_item13.py::test_polynomial_exactness` (parametrized over `dim` and `degree`) |
| current | `tests/field/test_interpolant.py::Testing::test_out_of_bounds` | upstream-style OOB-behavior test against `simsoptpp` | `covered_by_unit_parity` | `tests/jax_core/test_regular_grid_interp_item13.py::test_oob_behavior_returns_nan_when_strict` (JAX `NaN` semantic) |
| current | `tests/field/test_interpolant.py::Testing::test_skip` | upstream-style skip-region test against `simsoptpp` | `covered_by_unit_parity` | `tests/jax_core/test_regular_grid_interp_item13.py::test_skip_region_yields_zero_inside_skipped_cells` and `::test_cpp_cross_oracle_with_skip_mask` |
| current | `tests/field/test_interpolant.py::Testing::test_convergence_order` | upstream-style convergence-order test for `UniformInterpolationRule` | `oracle_only` | this is a convergence-rate property of the rule, not a same-state byte-parity claim; the JAX `test_polynomial_exactness` and Chebyshev-vs-uniform tests cover the underlying degree-exact reproduction |
| current | `src/simsopt/field/magneticfieldclasses.py:855-903` | `InterpolatedField` consumes `sopp.InterpolatedField` and `RegularGridInterpolant3D` | `wrapper_only` | downstream wrapper owned by item 15; item 13 deliberately leaves the public C++ binding wrappers in place |
| current | `src/simsopt/field/tracing.py:744` | tracing classifier accepts `sopp.RegularGridInterpolant3D` | `wrapper_only` | downstream wrapper owned by item 16; item 13 does not touch the tracing classifier surface |
| current | `src/simsopt/geo/surface.py:972-973` | surface distance interpolant uses `sopp.UniformInterpolationRule` + `sopp.RegularGridInterpolant3D` | `wrapper_only` | downstream consumer; not scoped to item 13 |
| upstream | `upstream_hss/master:src/simsoptpp/regular_grid_interpolant_3d.h` | upstream C++ kernel matches the current-tree copy at audit SHA | `oracle_only` | identical to the current-tree header within byte parity; same parity oracle |
| upstream | `upstream_hss/master:tests/field/test_interpolant.py` | upstream parity oracle for the C++ kernel | `oracle_only` | `git -C /Users/suhjungdae/code/opensource/simsopt grep -nE "RegularGridInterpolant3D"` shows the same upstream test file at audit SHA `1b0cc3a96063197cdbdd01559e04c25456fbe6ff` |

No matrix row is unclassified. The empty-oracle row pair is covered by
new parity tests built against (a) the hand-derived separable-polynomial
closed-form oracle and (b) the C++ `simsoptpp.RegularGridInterpolant3D`
binding, per the prompt section 4a empty-oracle guidance.

## Cited Parity-Test Execution Evidence

```bash
JAX_ENABLE_X64=True JAX_PLATFORMS=cpu \
  .conda/jax-0.9.2/bin/python -m pytest \
  tests/jax_core/test_regular_grid_interp_item13.py -q
```

Result line: `28 passed in 3.30s`.

Every test asserts against tolerances imported from
`benchmarks.validation_ladder_contract.parity_ladder_tolerances`.
`grep -E "(atol|rtol)\s*=\s*[0-9eE.+-]+"
tests/jax_core/test_regular_grid_interp_item13.py` matches only a
docstring reference to the lane tolerance literal, not a live argument
literal.
