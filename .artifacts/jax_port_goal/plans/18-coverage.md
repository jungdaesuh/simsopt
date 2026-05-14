# Item 18 Coverage

Coverage rows follow `.artifacts/jax_port_goal/CLOSEOUT_TEMPLATE.md`
(introduced after the 2026-05-13 audit, AI-2). Each row cites the test,
the independent oracle, and the parity-ladder lane.

| Coverage row | Test | Oracle | Lane |
| --- | --- | --- | --- |
| Rotated centroid frame analytic | `tests/geo/test_framedcurve_jax_item18.py::test_rotated_centroid_frame_matches_planar_circle_analytic` | Closed-form planar circle `γ(t) = (R cos 2πt, R sin 2πt, 0)`, tangent `T=(-sin 2πt, cos 2πt, 0)`, base centroid `N₀=(cos 2πt, sin 2πt, 0)`, `B₀=(0,0,-1)`; rotated frame `N = cos(α) N₀ − sin(α) B₀`, `B = sin(α) N₀ + cos(α) B₀` | `direct_kernel`, `rtol=1e-10`, `atol=1e-12` |
| Rotated frenet frame analytic | `tests/geo/test_framedcurve_jax_item18.py::test_rotated_frenet_frame_matches_planar_circle_analytic` | Same planar-circle closed form; Frenet normal `N₀` points radially inward, `B₀` aligned with `-ẑ`; rotation applied analytically | `direct_kernel`, `rtol=1e-10`, `atol=1e-12` |
| Centroid frame reduces at α=0 | `tests/geo/test_framedcurve_jax_item18.py::test_centroid_frame_matches_rotated_at_alpha_zero` | Algebraic identity: `rotated_*_frame(α=0)` must equal the base `*_frame` (degenerate-rotation invariant) | `direct_kernel`, `rtol=1e-10`, `atol=1e-12` |
| Frenet frame reduces at α=0 | `tests/geo/test_framedcurve_jax_item18.py::test_frenet_frame_matches_rotated_at_alpha_zero` | Same degenerate-rotation invariant | `direct_kernel`, `rtol=1e-10`, `atol=1e-12` |
| Frame orthonormality | `tests/geo/test_framedcurve_jax_item18.py::test_rotated_frame_is_orthonormal` | Algebraic identity: `T·N = N·B = B·T = 0`, `|T|=|N|=|B|=1` (orthonormal-triad invariant) | `direct_kernel`, `rtol=1e-10`, `atol=1e-12` |
| Wrapper DOF round-trip | `tests/geo/test_framedcurve_jax_wrappers_item18.py::test_frame_rotation_jax_dof_round_trip` | Optimizable DOF contract: `set_dofs(get_dofs())` is identity (independent of any kernel evaluation) | n/a (Optimizable contract) |
| Wrapper dependency graph | `tests/geo/test_framedcurve_jax_wrappers_item18.py::test_framed_curve_jax_dependency_graph` | Optimizable graph contract: declared dependencies match expected base-curve + rotation parents | n/a (Optimizable contract) |
| Wrapper drives outputs via DOFs (FD) | `tests/geo/test_framedcurve_jax_wrappers_item18.py::test_frame_rotation_jax_dofs_drive_wrapper_outputs_via_fd` | Central finite-difference of wrapper output w.r.t. DOFs (independent gradient oracle); replaces the pre-2026-05-13 `*_jax_matches_host` tests which the audit (#2) flagged as JAX-vs-JAX | `fd_gradient`, `directional_fd_rtol=1e-5`, `directional_fd_atol=1e-7` |
| Strict transfer guard | `tests/geo/test_framedcurve_jax_item18.py::test_kernels_run_under_strict_transfer_guard` | Runtime invariant: kernel must execute under `SIMSOPT_JAX_TRANSFER_GUARD=disallow` without host↔device transfer | n/a (transfer-boundary invariant) |

Bench pointer: `.artifacts/jax_port_goal/bench/18.json`.

## Audit note (2026-05-13)

Findings #1 and #2 in `.artifacts/jax-test-audit-2026-05-13/TEST_QUALITY_TODOS.md`
flagged the prior coverage rows as tautological:

- The deleted `test_rotated_centroid_frame_matches_upstream` and
  `test_rotated_frenet_frame_matches_upstream` compared a function to
  itself because `src/simsopt/geo/framedcurve.py` re-exports the JAX
  kernels under the "upstream" alias.
- The deleted `test_frame_rotation_jax_matches_host`,
  `test_zero_rotation_jax_matches_host`,
  `test_framed_curve_frenet_jax_matches_host`, and
  `test_framed_curve_centroid_jax_matches_host` compared the JAX
  wrapper to the "host" `FramedCurveFrenet`/`FramedCurveCentroid`, but
  those host classes invoke the same JAX kernel — a tautology.

The replacement tests anchor against closed-form planar-circle frames
(analytic oracle), the α=0 reduction identity, the orthonormality
invariant, and FD-vs-JAX (independent gradient oracle).
