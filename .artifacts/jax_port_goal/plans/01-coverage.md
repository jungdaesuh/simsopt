# Item 01 Coverage Matrix

Status: complete for CPU/JAX oracle closure. CUDA proof is not claimed.

Upstream audit SHA:
`1b0cc3a96063197cdbdd01559e04c25456fbe6ff`.

| Source | Path / Node | Intent | Classification | Evidence |
| --- | --- | --- | --- | --- |
| current | `src/simsopt/field/coilobjective.py:12` | JAX scalar `CurrentPenalty` kernel | `covered_by_unit_parity` | `tests/field/test_coilobjective.py::test_current_penalty_matches_scalar_cpu_oracle` |
| current | `src/simsopt/field/coilobjective.py:36` | Legacy wrapper value boundary | `covered_by_unit_parity` | `tests/field/test_coilobjective.py::test_current_penalty_wrapper_uses_explicit_transfer_boundary` |
| current | `src/simsopt/field/coilobjective.py:40` | Legacy wrapper derivative projection | `covered_by_unit_parity` | scalar, scaled, and sum-current tests all pass |
| current | `tests/field/test_coilobjective.py:62` | Pure JAX strict `jit` / `vmap` path | `covered_by_unit_parity` | `8 passed` in focused suite |
| current | `src/simsopt/geo/_distance_jax.py` | JAX candidate cullers | `covered_by_unit_parity` | `tests/geo/test_distance_jax.py` candidate equality and public-wrapper no-C++ tests |
| current | `src/simsopt/geo/curveobjectives.py:701` | `CurveCurveDistance.compute_candidates` JAX-mode routing | `covered_by_unit_parity` | `test_curve_curve_distance_uses_jax_candidate_culler` |
| current | `src/simsopt/geo/curveobjectives.py:938` | `CurveSurfaceDistance.compute_candidates` JAX-mode routing | `covered_by_unit_parity` | `test_curve_surface_distance_uses_jax_candidate_culler` |
| current | `src/simsopt/geo/curveobjectives.py:420` | `cc_distance_pure` dense/chunked reduction | `covered_by_unit_parity` | `test_pairwise_penalty_chunking_matches_dense_paths`; subprocess legacy value/gradient smokes |
| current | `src/simsopt/geo/curveobjectives.py:480` | `CurveCurveDistanceBarrier` reduction | `covered_by_unit_parity` | infeasible barrier and finite zero-gradient chunking tests |
| current | `src/simsopt/geo/curveobjectives.py:816` | `cs_distance_pure` dense/chunked reduction | `covered_by_unit_parity` | chunked/dense gradient strict-transfer tests |
| current | `src/simsopt/geo/_pairwise_reductions.py:135` | rowwise min helper with row sharding | `covered_by_unit_parity` | `test_pairwise_penalty_accepts_explicit_row_sharding`; subprocess row-sharding smoke |
| current | `src/simsopt/geo/_pairwise_reductions.py:237` | shared minimum-distance reduction | `covered_by_unit_parity` | pairwise chunking and strict-transfer tests |
| current | `src/simsopt/geo/curveobjectives.py:1190` | deprecated `MinimumDistance` alias | `wrapper_only` | inherits `CurveCurveDistance`; no separate kernel |
| current | `src/simsopt/geo/curveobjectives.py:1428` | `MinCurveCurveDistance` rowwise wrapper | `covered_by_unit_parity` | rowwise p-norm/min helper coverage via shared tests |
| upstream | `upstream_hss/master:src/simsopt/geo/curveobjectives.py:208` | CPU `CurveCurveDistance` public API and oracle semantics | `oracle_only` | candidate culler parity and legacy wrapper smokes |
| upstream | `upstream_hss/master:src/simsopt/geo/curveobjectives.py:328` | CPU `CurveSurfaceDistance` public API and oracle semantics | `oracle_only` | candidate culler parity and legacy wrapper smokes |
| upstream | `upstream_hss/master:src/simsoptpp/python_distance.cpp:174` | C++ within/between culler binding | `oracle_only` | `tests/geo/test_distance_jax.py` exact candidate-set comparisons |
| upstream | `upstream_hss/master:docs/source/geo.rst:63` | Official docs list `CurveCurveDistance` and `CurveSurfaceDistance` as public geo objectives | `oracle_only` | Context7 SIMSOPT docs consulted for public API |
| current | `benchmarks/non_banana_example_parity_fixtures.py` | non-banana composite consumers of distance objectives | `wrapper_only` | item 04 harness remains partial; item 01 does not claim full composite fixture closure |
| current | `src/simsopt/field/coilset.py` and `src/simsopt/util/coil_optimization_helper_functions.py` | downstream constructors using distance objectives | `wrapper_only` | covered indirectly by public wrapper tests and existing downstream tests |

No matrix row is unclassified. `CurrentPenalty` is a current-tree-only item, so
its oracle row is the hand-derived formula rather than an upstream source row.
