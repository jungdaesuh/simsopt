# Item 20 Coverage

Coverage rows follow `.artifacts/jax_port_goal/CLOSEOUT_TEMPLATE.md`
(introduced after the 2026-05-13 audit, AI-2). Each row cites the test,
the independent oracle, and the parity-ladder lane.

| Coverage row | Test | Oracle | Lane |
| --- | --- | --- | --- |
| Filament gammas analytic | `tests/geo/test_finitebuild_jax_ssot_item20.py::test_build_filament_gammas_matches_planar_circle_closed_form` | Closed-form filament position on planar circle: for base curve `γ(t)=(R cos 2πt, R sin 2πt, 0)` and centroid frame, filament `i` with offset `(dn, db)` sits at `γ + dn·N + db·B`, written out as an analytic array; replaces the pre-2026-05-13 `test_build_filament_gammas_matches_create_multifilament_grid` and `test_build_filament_gamma_and_dash_matches_grid_first_filament` which the audit (#3) flagged as JAX-vs-JAX | `direct_kernel`, `rtol=1e-10`, `atol=1e-12` |
| Compute offsets arithmetic | `tests/geo/test_finitebuild_jax_ssot_item20.py::test_compute_filament_offsets_matches_grid_construction` | Direct Python arithmetic: `_compute_filament_offsets` is a pure-Python helper over the grid index space; the test pins the offset list against hand-computed values for a `numfilaments_n=3`, `numfilaments_b=3`, `gapsize_n`, `gapsize_b` configuration | n/a (pure-Python helper) |
| Compiled strict transfer | `tests/geo/test_finitebuild_jax_ssot_item20.py::test_compiled_filament_builders_run_under_strict_transfer_guard` | Runtime invariant: compiled builders must execute under `SIMSOPT_JAX_TRANSFER_GUARD=disallow` without host↔device transfer | n/a (transfer-boundary invariant) |
| Multifilament grid + spec geometry | `tests/geo/test_finitebuild_jax_item20.py::test_multifilament_grid_preserves_offsets_and_spec_geometry` | Cross-check: spec geometry roundtrip preserves offsets emitted by `create_multifilament_grid` and matches the spec-driven JAX builder against the upstream CPU `CurveFilament.gamma()` call (CPU oracle path is the simsoptpp `Curve.gamma()` upstream contract; same-state values) | `direct_kernel`, `rtol=1e-10`, `atol=1e-12` |
| CurveFilament gamma VJP FD | `tests/geo/test_finitebuild_jax_item20.py::test_curvefilament_jax_gamma_vjp_matches_central_fd` | Central finite-difference of `CurveFilament.gamma()` (independent gradient oracle); replaces the pre-2026-05-13 `test_curvefilament_jax_vjps_match_public_derivative_methods` which the audit (#4) flagged as JAX-vs-JAX | `fd_gradient`, `directional_fd_rtol=1e-5`, `directional_fd_atol=1e-7` |
| CurveFilament gammadash VJP FD | `tests/geo/test_finitebuild_jax_item20.py::test_curvefilament_jax_gammadash_vjp_matches_central_fd` | Central FD of `CurveFilament.gammadash()` (independent gradient oracle) | `fd_gradient`, `directional_fd_rtol=1e-5`, `directional_fd_atol=1e-7` |
| Spec pullback FD | `tests/geo/test_finitebuild_jax_item20.py::test_curvefilament_spec_pullback_matches_central_fd` | Central FD of the spec-pullback path (independent gradient oracle) | `fd_gradient`, `directional_fd_rtol=1e-5`, `directional_fd_atol=1e-7` |

Bench pointer: `.artifacts/jax_port_goal/bench/20.json`.

## Audit note (2026-05-13)

Findings #3 and #4 in `.artifacts/jax-test-audit-2026-05-13/TEST_QUALITY_TODOS.md`
flagged the prior coverage rows as tautological:

- The deleted `test_build_filament_gammas_matches_create_multifilament_grid`
  and `test_build_filament_gamma_and_dash_matches_grid_first_filament`
  ran the SSOT helper `build_filament_gammas` against the host
  `CurveFilament.gamma()`, but both paths jit `gamma_jax` from
  `src/simsopt/jax_core/finitebuild.py` — a JAX-vs-JAX comparison.
- The deleted `test_curvefilament_jax_vjps_match_public_derivative_methods`
  compared two JAX-backed VJP routes (Optimizable derivative vs spec
  pullback) — no independent gradient oracle.

The replacement tests anchor against closed-form planar-circle filament
positions (analytic oracle) and central finite-difference (independent
gradient oracle).
