# Item 04 JAX Transform And Memory Strategy

Status: comprehensive for the in-scope item-04 modules. CPU-only validation
(no CUDA proof claimed).

## Compiled Boundary

The hot path consumes immutable specs and explicit DOF arrays per the
M0 contract; no compiled boundary reads mutable `Optimizable` wrapper state.
The principal compiled entrypoints are:

- `boozer_residual_scalar`, `boozer_residual_grad`, `boozer_residual_hessian`,
  `boozer_residual_vector`, `boozer_residual_scalar_and_grad_cpu_ordered`
  in `src/simsopt/geo/boozer_residual_jax.py:117-431` — pure functions over
  precomputed `B`, `xphi`, `xtheta`, `iota`, `G` arrays. Cited in
  `src/simsopt/geo/boozer_residual_jax.py:61-69` `__all__`.
- Composed pipeline: `_boozer_residual_vector_composed`,
  `boozer_residual_jacobian_composed`, `boozer_residual_coil_vjp`,
  `boozer_penalty_grad_composed` (`src/simsopt/geo/boozer_residual_jax.py:661-784`)
  pull through `_surface_geometry_from_dofs` and grouped Biot-Savart
  evaluation; consumed by `BoozerResidualJAX`.
- `BoozerSurfaceJAX` traceable runtime bundle:
  `make_traceable_objective_runtime_bundle`
  (`src/simsopt/geo/surfaceobjectives_jax.py:132,3355,3445,3557`), which
  caches `TraceableObjectiveSeededValueAndGrad` against deterministic
  signatures of the solved baseline state, objective kwargs, and coil /
  runtime specs (`tests/geo/test_surface_objectives_jax.py::test_get_cached_traceable_runtime_entry_*`).
  Per CLAUDE.md "Boozer target-lane compile-cache fix": stable closure
  identity lives in `BoozerSurfaceJAX` while dynamic `GroupedCoilSetSpec`
  travels as explicit solver args.
- LS forward + adjoint: `_traceable_solve_plu_linearization`
  (`src/simsopt/geo/surfaceobjectives_jax.py:3026,3079`) consumes
  `(lu, piv)` factors stored under `lax.stop_gradient` in the LS lane only.
- Exact-lane operator solve: `_traceable_solve_exact_linearization`
  (`src/simsopt/geo/surfaceobjectives_jax.py:3105`) routes through
  `_run_operator_gmres` in `optimizer_jax.py`; never materializes dense
  PLU for the runtime path.
- `BoozerSurfaceJAX.get_adjoint_runtime_state()`
  (`src/simsopt/geo/boozersurface_jax.py:_BoozerAdjointRuntimeState`,
  declared at line 595) is the runtime SSOT for adjoint callbacks /
  operator-vs-dense routing.

## Static Shape Strategy

- Immutable specs (`GroupedCoilSetSpec`,
  `_BoozerSurfaceRuntimeState`, `_BoozerSolvedRuntimeState`,
  `_BoozerPenaltyVectorizedInputs`) are dataclasses; shape metadata
  (`mpol`, `ntor`, `nfp`, `stellsym`, `nphi`, `ntheta`,
  `scatter_indices`, `mask_indices`) is hashable static metadata.
- Decision vector layout is fixed per spec:
  `(n_active_surface_dofs, [iota,], [G])` with stellsym-aware scatter
  to surface basis indices via `stellsym_scatter_indices` and
  `compute_stellsym_mask_indices_for_grid`. `_split_decision_vector`
  (`boozer_residual_jax.py:99`) and `_as_boozer_penalty_optimizer_state`
  (`boozersurface_jax.py:747`) split host scalars from device DOFs at the
  jit boundary.
- Static arguments under `jax.jit`: `optimize_G`, `label_type`,
  `weight_inv_modB`, `surface_kind`, `phi_idx`,
  `(label_)scatter_indices`, `(label_)mask_indices`.
- Traceable bundle reuses stable public boundaries
  (`tests/geo/test_surface_objectives_jax.py::test_make_traceable_objective_runtime_bundle_reuses_stable_public_boundaries`)
  to avoid recompilation on coil-DOF generation changes that do not affect
  static metadata.

## Transform Inventory

- `jit`: throughout the hot path; explicit `@jax.jit` on the value+grad
  bundle (`surfaceobjectives_jax.py:5724`). Traceable bundle wraps the
  inner solve in jit via `_compile_seeded_value_and_grad`. The Boozer
  residual scalar/grad/hessian primitives at
  `boozer_residual_jax.py:117-291` are also jitted by their consumers.
- `vmap`: used in `surface_curvatures_jax_from_dofs`,
  `_surface_dprincipal_curvature_jax_from_dofs`, and inside the surface
  metric helpers (`surfaceobjectives_jax.py:411-540`); also used by
  exact batched adjoints to flatten RHS columns. **No new `vmap` in this
  item; the exact lane batched adjoint solves one RHS at a time through
  the operator seam (`_normalize_solver_options` at
  `boozersurface_jax.py:3185-3186`).**
- `scan` / `fori_loop`: used by the iterative solvers in
  `optimizer_jax.py` (operator GMRES, Newton outer loop). Not item-04
  owned by the API surface but consumed via
  `levenberg_marquardt_traceable`, `newton_exact_traceable`,
  `newton_polish_traceable`.
- `checkpoint` / `remat`: N/A in the item-04 hot path; the runtime bundle
  does not introduce remat. (Pairwise-distance remat lives in item 07
  / item 01 helpers and is not item-04 scope.)
- `shard_map`, `pmap`, collectives: N/A. `git grep`
  `shard_map\|psum\|all_reduce\|pjit` in the in-scope source modules
  returns zero hits for item 04. The grouped Biot-Savart sharding is
  introduced by item 10 (`jax_core.field`) and consumed transparently;
  item 04 introduces no new collective.
- `custom_vjp`: applied to the traceable forward / adjoint solve
  (`surfaceobjectives_jax.py::test_traceable_custom_vjp_surfaces_adjoint_solve_failure_as_nan_gradient`
  cites the contract). Failed adjoint solve must surface NaN gradient
  rather than a finite direct-gradient or penalty fallback (per CLAUDE.md
  "Adjoint / warm-start operator solves").

## Why Transform Structure Matches SIMSOPT Math Contract

- Boozer residual scalar = `1/2 * sum r_i^2 / w_i` over surface grid;
  the JAX kernel reuses CPU `_boozer_penalty_vectorized_inputs` layout
  byte-for-byte (`tests/geo/test_boozer_residual_pinned_input_byte_parity.py`).
- The composed pipeline preserves the M3 contract: surface DOFs →
  `_surface_geometry_from_dofs` → grouped Biot-Savart → residual vector
  / scalar / gradient / hessian. No basis change, no preconditioning at
  the residual reporting boundary. Residual evidence is reported in the
  original physical basis after any preconditioning, normalization, or
  basis transform (matches plan section 4c).
- IFT adjoint:
  `dJ/d_coils = ∂J/∂coils − adj^T ∂g/∂coils`, where `adj` solves the
  transposed inner linearization (`get_adjoint_runtime_state()` is the
  SSOT). The LS lane reuses `(lu, piv)` factors under
  `lax.stop_gradient` only when
  `decision_size² × 8 ≤ max_dense_jacobian_bytes`; otherwise it stays on
  the operator path. Exact lane never falls back to dense at runtime.

## Dense Materialization And Donation

- Dense Jacobian budget: enforced by
  `max_dense_jacobian_bytes` (default budget plumbed via runtime contract).
  `tests/geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXExactPath::test_ls_surface_exact_newton_has_default_dense_jacobian_ceiling`
  and `::test_newton_exact_skips_dense_jacobian_when_ceiling_is_exceeded`
  exercise the ceiling. `test_run_code_exact_reports_scaling_limit_failure_without_fake_success`
  asserts predictable exact-mode reporting limits (per CLAUDE.md
  "Exact Boozer scaling-limit contract").
- LS lane dense PLU: load-bearing runtime data
  (`boozersurface_jax.py:3418-3475`, `surfaceobjectives_jax.py:3017-3055`);
  used by SciPy reference runtime callbacks AND by traceable adjoint when
  factors fit the byte budget. Exact lane dense PLU is metadata only
  (`linear_solve_backend="operator"`,
  `dense_linear_solve_factors_available`,
  `failure_category="scaling_limit"`,
  `failure_stage="dense_jacobian_finalization"`).
- Largest dense arrays (production-scale fixture in
  `tests/integration/test_single_stage_jax_cpu_reference.py`,
  `nphi=31`, `ntheta=16`, mpol=2, ntor=2): residual vector
  `float64[~1488]` (3·nphi·ntheta + 2), Jacobian `float64[~1488 × ndof]`,
  PLU `float64[~ndof × ndof]`. Always compared against
  `max_dense_jacobian_bytes` before materialization.
- `donate_argnums` / `donate_argnames`: **not used by item-04 modules.**
  `git grep "donate_argnums\|donate_argnames"` in the 8 in-scope source
  files returns 0 hits; the public wrappers (`BoozerSurfaceJAX`,
  `BoozerResidualJAX`, `IotasJAX`, `MajorRadiusJAX`,
  `NonQuasiSymmetricRatioJAX`) keep `Optimizable` wrapper state alive
  for downstream `Derivative` projection and would corrupt donor-array
  reuse. No HLO donation evidence is required because no donation is
  claimed.

## Decision-vector Shape Strategy

- `optimize_G=True`: x = [active_sdofs..., iota, G] (n_active+2).
- `optimize_G=False`: x = [active_sdofs..., iota] (n_active+1); G is
  derived from currents inside the kernel via
  `compute_G_from_currents` (`label_constraints_jax.py:49-61`).
- Stellsym=True: `active_sdofs` is the reduced free DOF set; the spec
  carries `scatter_indices` so the kernel reconstructs the full surface
  basis. Stellsym=False: full DOF set. Both branches are exercised by
  `TestStellsymMaskCPUJAXParity`, `TestParametrizedPenaltyGradientTaylor`,
  and `TestUpstreamFactoryBoozerMatrix`.

## HLO / Benchmark Artifact

- `.artifacts/jax_port_goal/bench/04.json` records the "no hot-path
  change" justification per section 4c. Item 04 closes existing JAX
  coverage with two NEW integration parity tests; the kernel surface
  is already exercised by the existing benchmarks
  (`benchmarks/jax_derivative_benchmark.py`,
  `benchmarks/non_banana_example_cpp_jax_cpu_parity.py`,
  `benchmarks/single_stage_smoke_fixture.py`,
  `benchmarks/single_stage_init_parity.py`).
- No new compiled hot path is introduced; the new tests reuse existing
  JAX kernels at fixed surface DOFs and copy CPU-solved state for QA
  wrappers. The traceable runtime bundle remains the SSOT; the harness
  does not bypass it.

## Multi-Device CPU Proxy

- N/A for item 04. The in-scope modules do not introduce any
  `shard_map`, `psum`, `all_reduce`, or `pjit`. The multi-device tests
  in `tests/test_jax_import_smoke.py` belong to items 01/10 (grouped
  Biot-Savart, pairwise penalties); item 04 inherits the proxy
  transparently through `BiotSavartJAX`.

## CUDA Status

- `not_claimed`. User requested CPU JAX only. JAX transfer-guard docs
  note that fetching CPU buffers is always allowed, so CPU validation
  cannot prove CUDA device residency. Real CUDA proof would require an
  approved GPU run with current-SHA artifact.
