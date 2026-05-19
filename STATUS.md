# simsopt-jax - Project Status, Backlog & Documentation Index

> Single entry point for the JAX port. This file summarizes the current tree,
> the validation evidence, the remaining TODOs, and the source documents to
> consult before making release or cleanup decisions.

| Field | Value |
|---|---|
| **Worktree** | `/Users/suhjungdae/code/columbia/simsopt-jax` |
| **Branch** | `gpu-purity-stage2-20260405` |
| **HEAD** | `0b2a69bf6` *fix: align lbfgs-ondevice fullgraph parity* |
| **Runtime probed** | Python 3.11.15 / JAX 0.10.0 / jaxlib 0.10.0 / NumPy 2.4.3 |
| **Local backend** | CPU only: `jax.devices() == [CpuDevice(id=0)]` |
| **Last refresh** | 2026-05-18 |
| **Parent repo** | `columbia/simsopt` |

## Purpose

The simsopt-jax worktree builds a JAX-native execution lane parallel to
simsopt's existing C++ (`simsoptpp`) backend, so stellarator optimization
workflows can run on GPU and use end-to-end autodiff.

This is a rewrite project, not a backend swap. JAX only becomes meaningfully
better than `simsoptpp` if it replaces a large fraction of the single-stage
inner loop: forward field, Boozer residual, derivative path, and
optimizer-facing objective. A field-only port is not enough.

### Goals

1. Run Stage 2 on GPU without depending on `simsoptpp` for the hot field path.
2. Run the single-stage Boozer inner solve on GPU with implicit differentiation.
3. Replace the hand-maintained field-derivative stack with JAX autodiff where practical.
4. Keep the CPU `simsoptpp` path as the reference implementation until parity and workflow gates pass.

### Non-goals

- Do not delete or degrade the `simsoptpp` CPU implementation.
- Do not make `simsoptpp` implicitly JAX-compatible.
- Do not require bitwise-identical CPU/GPU trajectories outside the documented parity lanes.
- Do not start by rewriting unrelated field types or frozen comparison worktrees.

## Architecture

```text
CPU oracle lane:
  src/simsoptpp/                     76 C++/header files, ~19K LOC
  pybind11 + xtensor + Eigen + SIMD

JAX lane:
  src/simsopt/jax_core/              39 pure-JAX Python files, ~22.8K LOC
  src/simsopt/{field,geo,objectives,solve}/*jax*.py
                                     ~32K wrapper/adaptor LOC

Contract:
  Existing SIMSOPT C++/SciPy behavior
    -> JAX CPU fixed-state parity
    -> JAX GPU parity
    -> JAX CPU/GPU agreement
```

### Parity-mode contract

| Mode | Purpose | Byte-identity to C++? | Speed vs C++ |
|---|---|---|---|
| `native_cpu` | C++ reference oracle. No JAX. | n/a; this is the oracle | 1.0x |
| `jax_cpu_parity`, `jax_gpu_parity` | Verification lanes | yes, within documented tolerance/build | 5-20x slower |
| `jax_cpu_fast`, `jax_gpu_fast` | Research speed-opt-out | no parity claim by construction | fastest |

All modes share one strict scalar-objective gate contract. Fast lanes do not get
relaxed tolerances; they are speed promises without a parity claim.

## Current State

### Ported surface

Concrete file mapping is in `04_review_field.md`, `05_review_geo_small.md`,
`06_review_geo_big.md`, and `07_review_obj_solve.md`. The latest broad status
writeup is `docs/jax_port_status_2026-05-15.md`, but that document predates
current HEAD and should be treated as dated evidence.

| Area | Status |
|---|---|
| BiotSavart B/dB/A/dA + VJP | Ported (`jax_core/biotsavart.py`, `field/biotsavart_jax_backend.py`) |
| Surface tensor/RZ Fourier + Henneberg + Garabedian | Ported |
| Curves (XYZ/RZ/Planar Fourier, framed, finite-build) | Ported |
| Boozer analytic + radial interp + interpolated + residual | Ported |
| Particle tracing (fieldline, GC, full-orbit, events) | Ported; forward-mode only on while-loop integrators unless later migrated |
| Permanent magnet (GPMO, MwPGP, relax-and-split) | Ported |
| Wireframe field + RCLS + GSCO | Ported |
| `integral_BdotN` | Ported; zero-normal masking now protects inactive non-finite inputs before AD |
| Regular grid interp 3D | Ported |
| Dommaschk / Reiman / Dipole / Circular coil / Mirror / Toroidal / Poloidal / Scalar-potential | Ported |
| BoozerSurface inner solve (M4) | Ported (LS + exact lanes, operator-backed adjoint) |
| Single-stage IFT wrappers (M5) | Ported (`BoozerResidualJAX`, `IotasJAX`, `NonQuasiSymmetricRatioJAX`, `MajorRadiusJAX`) |
| Stage 2 fused objective | Ported (`objectives/stage2_target_objective_jax.py`) |
| On-device L-BFGS-B optimizer | Ported SciPy 1.17.1 Fortran lbfgsb parity path; RESTART/cache/ddot perf fixes present; full-`JF.x` CPU bridge proof present; GPU proof remains open |
| Backend selection | Ported (`SIMSOPT_BACKEND`, `SIMSOPT_JAX_PLATFORM`) |

### Latest parity evidence

| Metric | Value |
|---|---|
| Artifact | `.artifacts/parity/20260518-current-head-cpu/all-fixtures.json` |
| Artifact SHA | `a43ea274bc821f5c831c1bf21e29ff88822be7ac` |
| Artifact commit | `0b2a69bf6` |
| Worktree HEAD | `0b2a69bf6` |
| Parity fixtures | 27 |
| Pass / partial / fail | 21 / 6 / 0 |
| Supported CPU/JAX comparisons | 251 |
| Failed comparisons | 0 |
| Worst absolute diff | `1.6675243386998773e-05` (`tracing_particle_gc_vac_reduced_endpoint::trajectory_endpoint`) |
| Worst relative diff | `0.00019296860502983134` (`boozer_surface_basic::boozer_residual`) |
| Version probe command | `.conda/jax/bin/python` interpreter recorded from the live run |
| GPU/CUDA parity | Not proven; artifact records `jax_gpu.status = runtime_required` |
| Dirty-tree policy | `record`; artifact metadata records the current dirty worktree |

This matrix is current CPU/JAX regression evidence for the dirty worktree at
`0b2a69bf6`. It is not a clean release artifact until the intended source/report
slice is committed and the matrix is regenerated or explicitly re-verified
against that clean commit.

### Parity audit closeout

`.artifacts/parity_audit_2026-05-16/ISSUES_CHECKLIST.md` no longer has local
unchecked checklist items. Its header records `0 remaining unchecked local
checklist TODOs`, with the original local items checked or explicitly
dispositioned in `EXECUTION_VERIFICATION.md`.

The P8 rows were not completed locally. They were reclassified as external
hardware/platform signoff gates in `P8_EXTERNAL_SIGNOFF.md`, because this
workstation only has CPU JAX available.

### Port-gap inventory

`.artifacts/jax_port_gap_audit_2026-05-13/cpp_port_gap.md` remains the detailed
symbol inventory: 78 PORTED, 23 PARTIAL, 7 UNPORTED, 11 NON-PORTABLE, and 8
UNCLEAR across 127 public C++ symbols. Treat it as a port-completeness backlog,
not the same thing as release-blocking parity signoff.

Live 2026-05-18 resolution of the audit's eight UNCLEAR questions:

- **Q1 GSCO parity: closed.** Current tests include fixed-state, large-grid,
  undo/stop/eligibility, wrapper, and public-CPU parity coverage in
  `tests/solve/test_wireframe_optimization_jax_item31.py`.
- **Q2 `boozer_dresidual_dc`: closed as JAX-autodiff parity.** The C++ helper
  remains the CPU oracle; JAX uses the composed residual Jacobian and coil VJP,
  with dedicated parity tests against the C++ derivative route.
- **Q3 `RotatedCurve` VJP matmult: closed as CPU-wrapper-only.** The JAX spec
  layer documents standalone `RotatedCurve` as non-spec CPU wrapper behavior;
  coil placement uses `CoilSymmetrySpec` rotation/scale in JAX.
- **Q4 curve third derivative and VJP: closed.** Geometry emits
  `gammadashdashdash` through the generic spec ladder, including perturbed and
  non-XYZ direct curve specs. Named public VJP wrappers now cover `gamma`,
  `gammadash`, `gammadashdash`, and `gammadashdashdash`, with CPU tensor
  contraction parity tests for XYZ, RZ, Planar, Helical, XYZ-symmetry, and
  Perturbed specs.
- **Q5 non-RZ surface curvature/fundamental forms: closed.** `jax_core`
  now exposes first fundamental form, second fundamental form, surface
  curvatures, and dof-Jacobian helpers for both `SurfaceXYZFourierSpec` and
  `SurfaceXYZTensorFourierSpec`, with CPU parity tests against the public
  `Surface` methods and `_by_dcoeff` methods.
- **Q6 cylindrical magnetic-field accessors: closed.** `BiotSavartJAX` and
  `SpecBackedBiotSavartJAX` expose explicit cylindrical accessors, and the
  remaining wrappers inherit the shared CPU cylindrical boundary.
- **Q7 tracing `get_phi`: closed.** `simsopt.jax_core.tracing.get_phi` is now a
  public wrapper over the JAX continuous-branch helper and is tested against
  `simsoptpp.get_phi`.
- **Q8 `Surface::scale`: closed as CPU mutation / fit boundary.** The C++ API
  constructs scaled target grid values around each phi-slice theta mean and then
  calls `least_squares_fit`, mutating the object. That is not a simple immutable
  spec `dofs * scale` transform. The supported JAX boundary is CPU object
  `scale(...)` followed by `surface_spec()`, with object-API parity coverage for
  `SurfaceXYZFourier` and `SurfaceXYZTensorFourier`.

Live S-1/S-2 re-audit notes:

- **SurfaceRZFourier third paired `_lin` derivatives: closed.** `jax_core`
  now exposes spec and dof helpers for `gammadash1dash1dash1_lin`,
  `gammadash1dash1dash2_lin`, `gammadash1dash2dash2_lin`, and
  `gammadash2dash2dash2_lin`, all backed by the existing RZ Fourier derivative
  kernel and pinned against the CPU pybind object methods.
- **Non-RZ surface paired `_lin` derivatives: closed.** `jax_core` now exposes
  spec and dof helpers for the second- and third-order paired `_lin` derivative
  methods on `SurfaceXYZFourier` and `SurfaceXYZTensorFourier`, pinned against
  the CPU pybind object methods.
- **Curve scalar derivative wrappers: closed.** `jax_core` now exposes Jacobian
  and VJP helpers for `dincremental_arclength_by_dcoeff`,
  `dkappa_by_dcoeff`, and `dtorsion_by_dcoeff`, pinned against CPU derivative
  methods across XYZ, RZ, Planar, and Helical curve specs.
- **Wireframe second spatial derivatives: reclassify out of JAX-port
  implementation backlog.** The upstream C++ `WireframeField` path raises
  `logic_error("Second spatial derivatives not implemented for WireframeField")`
  when `derivatives >= 2`, so a JAX `wireframe_d2B_by_dXdX` helper would not be
  mirroring an implemented C++ oracle.
- **BiotSavart current-derivative cache bundle: closed for public JAX APIs.**
  `BiotSavartJAX` and `SpecBackedBiotSavartJAX` expose explicit
  `dB_by_dcoilcurrents`, `d2B_by_dXdcoilcurrents`,
  `d3B_by_dXdXdcoilcurrents`, and `dA_by_dcoilcurrents` methods, with NCSX
  CPU parity tests. The C++ `compute(...)` / `compute_A(...)` cache-filler
  entrypoints remain cache orchestration rather than a separate required JAX
  kernel surface.
- **Surface LS / fit / offset rows: CPU object-mutation boundary.** C++
  `Surface::least_squares_fit`, `fit_to_curve`,
  `_extend_via_normal_for_nonuniform_phi`, and
  `extend_via_projected_normal` all build target grid values and call the
  mutating Eigen `FullPivHouseholderQR` fit path. The current supported JAX
  boundary is CPU object method execution followed by `surface_spec()`, pinned
  by non-RZ and RZ object-API parity tests. A pure immutable JAX surface-fit
  helper would be new API work, not a current release-blocking parity gap.
- **Curve `least_squares_fit`: closed as CPU object-mutation boundary.** The C++
  method also uses Eigen `FullPivHouseholderQR` and mutates object DOFs. A live
  call-site scan found no in-tree Python/JAX use of curve `least_squares_fit`;
  surface fitting has its own object boundary above. A live solver probe showed
  NumPy/SciPy least-squares routes match CPU DOFs for full-rank XYZ, RZ, and
  Helical curves, but choose a different rank-deficient `CurvePlanarFourier`
  nullspace DOF vector while preserving the fitted geometry. The supported JAX
  boundary is CPU object `least_squares_fit(...)` followed by
  `curve_spec_from_curve(...)`, now pinned for XYZ, RZ, Planar, and Helical
  curves. A pure immutable JAX curve-fit helper would be new API work requiring
  an explicit QR/nullspace contract.

## Outstanding Work

Source order for this section:

1. Current-tree validation in this refresh.
2. `.artifacts/parity_audit_2026-05-16/EXECUTION_VERIFICATION.md`.
3. `.artifacts/parity_audit_2026-05-16/P8_EXTERNAL_SIGNOFF.md`.
4. `.artifacts/fix_all_2026-05-16/00_SYNTHESIS.md` for investigation provenance. If a synthesis checkmark conflicts with live code, live code wins.

### Release gates

- [x] **Current dirty-tree CPU parity matrix.** Regenerated the native
  CPU/C++/SciPy -> JAX CPU matrix for worktree HEAD `0b2a69bf6`; the
  27-fixture artifact has 21 pass / 6 partial / 0 fail and 251 supported
  CPU/JAX comparisons with 0 failures.
- [ ] **Clean post-commit CPU parity matrix.** After the intended source/report
  slice is committed, regenerate or explicitly re-verify the 27-fixture matrix
  against the clean commit/tag.
- [ ] **CUDA/GPU acceptance.** Produce current-SHA CUDA/H200, cross-platform,
  long-soak, and concurrency artifacts per
  `.artifacts/parity_audit_2026-05-16/P8_EXTERNAL_SIGNOFF.md`.

### Current non-CUDA unresolved list

- [ ] **Clean post-commit CPU parity matrix.** The dirty-tree CPU matrix is
  green/partial-only; rerun or explicitly re-verify it after the intended
  source/report slice is committed.
- [ ] **Stash disposition.** Three inspected stashes contain tracked changes
  outside the current worktree diff; keep them until the owner explicitly
  approves dropping or replaying them.

No release-scope port-completeness residual remains after the live S-1/S-2/S-3
re-audit. Surface LS/fit/offset and curve `least_squares_fit` are
documented/tested CPU object-mutation boundaries; BiotSavart cache entrypoints
and Wireframe second derivatives are cache-oracle / non-implemented-oracle
boundaries. Pure immutable JAX least-squares fit helpers would be future API
work, not part of the current release contract.

### Closed in this refresh

- [x] **L-BFGS-B ddot perf.** `_lbfgsb_ddot` now uses vectorized masked products
  plus `reduce_sum`, with a structural jaxpr test rejecting elementwise
  `scan`/`while`/`cond` lowering.
- [x] **Exact Boozer dense diagnostics.** Quiet exact solves now skip dense PLU
  and dense condition estimates while keeping the exact adjoint path
  operator-backed.
- [x] **On-axis iota 2x2 eig.** `on_axis_iota_rk` now extracts the first
  eigenvalue angle with a closed-form 2x2 formula instead of `jnp.linalg.eig`.
- [x] **`integral_BdotN` double-where hardening.** Zero-normal points now mask
  inactive `Bcoil`, target, and residual terms before square-root weighting;
  value and gradients stay finite when inactive inputs are non-finite.
- [x] **`SpecBackedBiotSavartJAX` DOF setter contract.** `x` now writes
  free-DOF semantics through `_dofs.free_x`, matching upstream
  `Optimizable.x`.
- [x] **`BiotSavartJAX` fast-path wiring.** The uniform `CurveXYZFourier`
  explicit-state path now feeds `coil_set_spec()` through grouped JAX curve
  arrays; focused strict-mode and immutable-spec correctness tests pass.
  Standalone performance evidence still belongs with the release
  benchmark/parity gates.
- [x] **CLAUDE.md validation drift.** Validation commands now use the repo-local
  `.conda/jax/bin/python`; private optimizer commands point at the current
  private-runtime files and stale fixed pass counts were removed.
- [x] **`jax_core` package-boundary cleanup.** `wireframe.py` now imports the
  index-range check from neutral host contracts, and plain `import simsopt`
  no longer eagerly imports `_core` or probes `simsoptpp`; root exports resolve
  through lazy `__getattr__`.
- [x] **B-1 layering/import-bootstrap proof.** A current-tree scan finds no
  remaining `simsopt.jax_core` imports from `simsopt.geo`, `simsopt.field`, or
  `simsopt.objectives`, and an import probe confirms both `import simsopt` and
  `import simsopt.jax_core` leave `simsoptpp` and `simsopt._core` unloaded.
  Accessing the legacy root export `simsopt.make_optimizable` still lazy-loads
  `_core` by design. The `case_import_jax_core_specs` subprocess smoke now
  blocks `simsoptpp` and asserts `_core` / `simsoptpp` remain unloaded.
- [x] **Compile diagnostics focused test root cause.** The diagnostics test no
  longer boots the full single-stage/Boozer subprocess. It now tests recorder
  accounting, the logging context manager, and source-level runtime wiring to
  `results["JAX_COMPILE_DIAGNOSTICS"]` / `jax_compile_diagnostics.json`.
  The cached-pyc `TypeError` was not reproduced; the current focused test run
  passes.
- [x] **Force `*JAX` public aliases disposition.** `B2EnergyJAX` and
  `LpCurveForceJAX` are removed from the `force.py` and `simsopt.field`
  export surfaces; the closeout test asserts both aliases stay absent.
- [x] **Surface Fourier float64 helper consolidation.**
  `surface_fourier_jax.py` now imports the shared `jax_core._math_utils`
  float64/int32/zeros helpers instead of carrying local duplicates.
- [x] **Framed-curve JAX API fill.** `FramedCurve*JAX` now expose
  `frame_twist`, `dframe_twist_by_dcoeff_vjp`, and rotated-frame VJP helpers;
  the new test checks the curve-DOF twist VJP against finite differences.
- [x] **BiotSavart cylindrical API boundary.** `BiotSavartJAX` and
  `SpecBackedBiotSavartJAX` now provide `set_points_cart`, `set_points_cyl`,
  `get_points_cart`, `get_points_cyl`, `AbsB`, `GradAbsB`, `B_cyl`, `A_cyl`,
  and `GradAbsB_cyl`, including CPU-compatible setter chaining and supplied
  cylindrical-`phi` round-trip semantics.
- [x] **JAX field cylindrical audit.** A live wrapper scan shows the remaining
  per-class JAX field wrappers inherit from `MagneticField` and therefore use
  the shared CPU cylindrical boundary, while `InterpolatedFieldJAX` and
  BiotSavart own explicit cylindrical caches/accessors. A focused inherited
  accessor test now pins `B_cyl`, `A_cyl` where implemented, `GradAbsB_cyl`,
  `get_points_cyl`, and `get_points_cart` parity for `ToroidalFieldJAX`,
  `PoloidalFieldJAX`, `MirrorModelJAX`, `ReimanJAX`, `DommaschkJAX`, and
  `CircularCoilJAX`.
- [x] **Tracing `get_phi` public wrapper.** The private continuous-branch helper
  is now exposed as `simsopt.jax_core.tracing.get_phi`, and the edge/tie parity
  test calls the public JAX wrapper against `simsoptpp.get_phi`.
- [x] **Named curve geometry VJP wrappers.** `simsopt.jax_core` now exports
  `curve_gamma_vjp_from_dofs`, `curve_gammadash_vjp_from_dofs`,
  `curve_gammadashdash_vjp_from_dofs`, and
  `curve_gammadashdashdash_vjp_from_dofs`; focused tests pin CPU derivative
  tensor-contraction parity across XYZ, RZ, Planar, Helical, XYZ-symmetry, and
  Perturbed specs.
- [x] **Non-RZ surface fundamental forms.** `SurfaceXYZFourierSpec` and
  `SurfaceXYZTensorFourierSpec` now have direct JAX first/second
  fundamental-form, curvature, and dof-Jacobian helpers, exported through
  `simsopt.jax_core` and pinned against CPU `Surface` oracles.
- [x] **`Surface::scale` JAX disposition.** C++ `Surface::scale` is documented
  here as a mutating least-squares-fit object method, not an immutable spec
  multiply. The existing non-RZ object API parity test confirms scaled CPU
  objects still materialize matching JAX specs.
- [x] **SurfaceRZFourier third paired `_lin` helpers.** `simsopt.jax_core`
  now exports spec and dof helpers for the four third-order paired RZ surface
  derivative methods and validates them against the CPU pybind methods.
- [x] **Non-RZ surface paired `_lin` helpers.** `simsopt.jax_core` now exports
  spec and dof helpers for the second- and third-order paired `_lin` derivative
  methods on `SurfaceXYZFourier` and `SurfaceXYZTensorFourier`, with focused
  CPU pybind parity coverage.
- [x] **Curve scalar derivative wrappers.** `simsopt.jax_core` now exports
  Jacobian and VJP helpers for incremental arclength, curvature, and torsion
  derivatives with CPU derivative parity coverage across production direct
  curve specs.
- [x] **`_per_coil_unit_field` sharding contract review.** The helper remains a
  list-shaped per-coil Python loop by design, bypassing coil-axis collectives;
  a focused `SIMSOPT_JAX_SHARDING=coil_groups` test now pins parity and output
  shape for that contract.
- [x] **Per-coil derivative performance follow-up reclassification.** The
  remaining vectorized/CUDA-sharded derivative-kernel question is a GPU
  performance item, not a non-CUDA correctness TODO. The non-CUDA contract is
  the reviewed list-shaped helper above.
- [x] **Force triangular-solve migration.** `force.py` now solves each identity
  RHS column through `_solve_triangular_columns(...)`; the live source scan has
  no remaining `static_argnums` hits in `src/`.
- [x] **Permanent-magnet double-where cleanup.** `projection_l2_balls(...)`
  now avoids inactive divide-by-zero branches and zero-radius sqrt-gradient
  hazards while preserving transfer-guard behavior and CPU projection parity.
- [x] **JAX tree API migration.** Remaining live
  `jax.tree_util.tree_map/tree_leaves` usages in the Stage 2 and single-stage
  entrypoints were migrated to `jax.tree.map` / `jax.tree.leaves`; the live
  tree-API scan is clean.
- [x] **Unused BiotSavart pullback alias cleanup.** The stale
  `BiotSavartBPullback` export alias was removed; `BiotSavartFieldPullback`
  remains the public type.
- [x] **Parity harness provenance command cleanup.**
  `non_banana_example_cpp_jax_cpu_parity.py` now records the actual
  `sys.executable` interpreter in `version_probe_command`; the current
  all-fixture CPU artifact was regenerated with repo-local `.conda/jax/bin/python`
  provenance, and the non-banana parity command examples now use the same
  repo-local interpreter instead of `conda run -n jax`.
- [x] **`backend.py` `os` re-export cleanup.** The facade no longer imports or
  exports `os`, and a live scan found no `simsopt.backend.os` or
  `from simsopt.backend import os` consumers.
- [x] **Private optimizer JAX-version prose cleanup.** Live optimizer docs now
  distinguish the pinned upstream `jax-v0.9.2` provenance source from the
  checked local JAX/JAXLIB 0.10.0 runtime, and the Optimistix benchmark usage
  points at repo-local `.conda/jax/bin/python`.
- [x] **LM production-scope disposition.** The original Track 1
  byte-identical MINPACK `lmder` port was abandoned at Phase 0 because the
  production Boozer QR shape is not bit-identical, then replaced by the
  owner-approved CPU tolerance-equivalent dense-QR lane
  `least_squares_algorithm="lm-minpack"` / `method="lm-minpack-ondevice"`.
  The revised Track 1 route, BoozerSurface routing, and focused parity tests
  are present in the live tree. Track 3 remains an optional CPU diagnostic
  Optimistix/Lineax lane, not a promoted production solver. The original
  "first JAX MINPACK port" publication framing is therefore moot unless a new
  novelty assessment is opened.
- [x] **`lbfgs_ondevice_quadratic_smokes` timeout investigation.** The
  previously reported subprocess timeout no longer reproduces on the current
  tree; the isolated transfer-guard smoke passes on CPU/x64 in 35.87s.
- [x] **L-BFGS full-`JF.x` CPU bridge proof.** The prior abnormal /
  zero-progress `lbfgs-ondevice` comparison was traced to the compact coil-only
  objective path. The current bridge routes the CPU-order 51D `JF.x`
  value/gradient through an ordered host callback into the private JAX
  L-BFGS-B driver; the reduced comparison matches CPU/SciPy final objective to
  about `1.2e-9` with 4 value/gradient evaluations in each lane. This closes
  the non-CUDA optimizer correctness TODO. A pure device-side fullgraph
  objective remains GPU-purity architecture work, not a current local CPU
  correctness blocker.
- [x] **Boozer CPU RNG/test cleanup.** Boozer derivative tests now use local
  generators instead of mutating NumPy global RNG state, and direct CPU
  `BoozerSurface.minimize_boozer_penalty_constraints_LBFGS()` calls on exact
  surfaces receive the same `record_scipy_callback_trace=False` option default
  as LS surfaces.
- [x] **Deterministic seed cleanup slice 1.** Four legacy tests now use local
  `np.random.default_rng(...)` generators instead of mutating global NumPy RNG
  state: `tests/core/test_derivative.py`,
  `tests/objectives/test_fluxobjective.py`,
  `tests/objectives/test_utilities.py`, and
  `tests/field/test_interpolant.py`.
- [x] **Deterministic seed cleanup slice 2.** Geometry Taylor/surface helper
  tests now use local RNG objects instead of `np.random.seed` in
  `tests/geo/test_curve.py`, `tests/geo/test_surface_taylor.py`,
  `tests/geo/test_surface_objectives.py`, and
  `tests/geo/test_surface_rzfourier.py`.
- [x] **Deterministic seed cleanup slice 3.** Small solve/field/strain tests no
  longer mutate global NumPy RNG state in `tests/solve/test_constrained.py`,
  `tests/geo/test_strainopt.py`, `tests/field/test_magneticfields.py`, and
  `tests/field/test_coilset.py`.
- [x] **Deterministic seed cleanup slice 4.** SurfaceXYZFourier and BiotSavart
  legacy tests now keep perturbation randomness local in
  `tests/geo/test_surface_xyzfourier.py` and
  `tests/field/test_biotsavart.py`; the BiotSavart perturbed-curve helper no
  longer consumes global RNG state.
- [x] **Deterministic seed cleanup slice 5.** Particle tracing tests now use
  local `RandomState` streams for sampled trajectory indices and initial
  conditions instead of mutating global NumPy RNG state.
- [x] **Deterministic seed cleanup slice 6.** Coil and finite-build tests now
  use local RNG streams for curve perturbations, makegrid comparison points,
  VTK helper curves, filament derivative directions, and finite-build
  perturbation directions.
- [x] **Deterministic seed cleanup slice 7.** The final repo-wide
  `np.random.seed` call sites in
  `tests/field/test_boozermagneticfields.py`,
  `tests/mhd/test_vmec_diagnostics.py`, `tests/geo/test_qfm.py`, and
  `tests/geo/test_curve_objectives.py` now use local `RandomState` streams.
  A live `rg -n "np\.random\.seed" tests` scan is clean.

### Latest local validation evidence

The full-suite runs below are current for `0b2a69bf6` plus the earlier
fix-all source slice. The focused runs cover the final PM/tree cleanup patches
made after those full suites.

| Scope | Result |
|---|---|
| Full integration suite: `tests/integration/ -v` | 450 passed, 6 skipped |
| Public pure-JAX suite from `CLAUDE.md` | 848 passed, 114 skipped |
| Private optimizer runtime suite | 50 passed, 224 deselected |
| Benchmark/helper suite | 270 passed, 2 skipped |
| Current dirty-tree CPU/JAX all-fixture parity artifact | 27 fixtures; 21 pass / 6 partial / 0 fail; 251 comparisons pass, 0 fail |
| B-1 current-tree layering/import smoke | no forbidden `jax_core` -> `geo`/`field`/`objectives` imports; `test_import_package_root` + tightened `test_import_jax_core_specs` pass; `import simsopt.jax_core` leaves `simsoptpp` and `simsopt._core` unloaded |
| Explicit downstream `tests/integration/test_single_stage_jax.py` gate | 7 passed |
| Compile diagnostics focused regression | 4 passed |
| `lbfgs_ondevice_quadratic_smokes` focused rerun | 1 passed in 35.87s |
| Boozer CPU RNG/option-contract focused reruns | 8 passed, 49 subtests passed |
| Deterministic seed cleanup slice 1 | 28 passed, 25 subtests passed |
| Deterministic seed cleanup slice 2 | 7 passed, 70 subtests passed |
| Deterministic seed cleanup slice 3 | 5 passed |
| Deterministic seed cleanup slice 4 | 13 passed, 4 subtests passed |
| Deterministic seed cleanup slice 5 | 11 passed |
| Deterministic seed cleanup slice 6 | 6 passed, 16 subtests passed |
| Deterministic seed cleanup slice 7 | 16 passed, 5 skipped, 128 subtests passed |
| JAX field inherited cylindrical accessor audit | 6 passed |
| Tracing `get_phi` public wrapper parity | ruff passed; `test_get_phi_matches_cpp_get_phi_edges`: 8 passed |
| Named curve geometry VJP wrappers | ruff passed; 25 focused CPU tensor-contraction parity tests passed |
| Non-RZ surface fundamental forms and curvature Jacobians | ruff passed; 8 focused CPU parity tests passed |
| `Surface::scale` object boundary | `TestSurfaceFourierObjectApiParity::test_scale_object_api_parity`: 4 passed |
| SurfaceRZFourier third paired `_lin` helpers | ruff passed; `test_surface_rzfourier_third_paired_lin_helpers_match_cpu`: 2 passed, 2 skipped |
| Non-RZ surface paired `_lin` helpers | ruff passed; `test_higher_paired_lin_wrappers_match_cpp`: 4 passed |
| Curve scalar derivative wrappers | ruff passed; `test_curve_scalar_derivative_wrappers_match_cpu_derivatives`: 12 passed |
| Curve least-squares CPU boundary | `test_curve_least_squares_fit_cpu_boundary_materializes_jax_spec`: 4 passed; full `test_curve_item05_closeout.py`: 41 passed |
| Public pure-JAX CPU suite | `tests/test_jax_import_smoke.py tests/field/test_biotsavart_jax.py tests/geo/test_surface_fourier_jax.py tests/geo/test_boozer_residual_jax.py tests/objectives/test_integral_bdotn_jax.py tests/geo/test_boozer_derivatives_jax.py tests/geo/test_boozersurface_jax.py tests/integration/test_jax_native_path.py -m "not private_optimizer_runtime"`: 860 passed, 114 skipped |
| Private optimizer runtime suite | `tests/geo/test_boozersurface_jax_private.py tests/integration/test_section6_public_lane_split.py tests/integration/test_single_stage_jax_cpu_reference.py -m "private_optimizer_runtime"`: 50 passed, 224 deselected |
| Benchmark/runtime helper suite | `tests/test_run_code_benchmark_common.py tests/test_benchmark_helpers.py`: 270 passed, 2 skipped |
| M2+M5 integration suite | `tests/integration/`: 450 passed, 6 skipped |
| Track 1 revised `lm-minpack-ondevice` parity suite | 11 passed |
| L-BFGS full-`JF.x` CPU bridge proof | report shows `lbfgs-ondevice` final objective `1.113211083463847`, CPU/SciPy `1.1132110846645535`, diff `-1.20e-9`, 4/4 nfev/njev; focused callback/routing tests: 3 passed |
| Stage 2 / single-stage tree API focused rerun | 8 passed |
| PM JAX item file after double-where cleanup | 59 passed |
| PM solve-wrapper projection parity focused rerun | 3 passed |

### Stash inspection

The three stashes were inspected on 2026-05-18. None has an untracked-file
payload, but none is safe to drop automatically because each contains tracked
changes in paths not represented by the current worktree diff:

| Stash | Base | Tracked paths | Distinct tracked paths not in current diff | Disposition |
|---|---|---:|---:|---|
| `stash@{0}` | `d75ebcb7b` | 26 | 10 | Keep until owner explicitly approves dropping or replaying |
| `stash@{1}` | `be850cb72` | 34 | 10 | Keep until owner explicitly approves dropping or replaying |
| `stash@{2}` | `cadc6139e` | 41 | 24 | Keep until owner explicitly approves dropping or replaying |

### High-priority local TODOs

No unchecked high-priority local TODOs remain after excluding CUDA/GPU/hardware
acceptance and post-commit workflow gates. The old live optimizer item is
closed by the full-`JF.x` host-callback bridge evidence above. A pure
device-side fullgraph objective remains future GPU-purity architecture work, not
a current CPU correctness blocker.

### Medium / cleanup backlog

No current medium/cleanup item remains in this consolidated status file. The
historical fix-all synthesis still contains unchecked checkboxes, but those rows
are either closed above, superseded by live-code validation, or explicitly
outside the JAX-port release scope. Use the port-gap inventory above for
historical provenance and future API-expansion decisions.

## Active Workstreams

| Workstream | Tracking artifact | Status |
|---|---|---|
| GPU purity / strict-cuda hardening | Branch `gpu-purity-stage2-20260405` | Active |
| L-BFGS-B parity / on-device optimizer | `.artifacts/lbfgsb_parity_review_2026-05-16/`, `docs/scipy_lbfgsb_jax_parity_plan_2026-05-15.md`, `docs/lbfgs_ondevice_full_jfx_bridge_report_2026-05-18.md` | RESTART/cache/ddot perf fixes present; full-`JF.x` CPU bridge proof closed; GPU/pure-device objective remains future hardening |
| LM MINPACK / Optimistix lanes | `.artifacts/lm_minpack_port_plan_2026-05-16/PLAN.md`, Track 3 artifacts | Revised Track 1 `lm-minpack-ondevice` implemented and tested; Track 3 Optimistix/Lineax remains optional CPU diagnostic; CUDA proof remains external |
| Parity audit closeout | `.artifacts/parity_audit_2026-05-16/ISSUES_CHECKLIST.md` | Local checklist closed; P8 external signoff remains |
| `jax_core` layering cleanup | `.artifacts/fix_all_2026-05-16/X1_blocker_plan.md` | Closed for current tree; no forbidden `jax_core` -> `geo`/`field`/`objectives` imports and root `_core`/`simsoptpp` bootstrap is lazy |
| CUDA evidence / GPU acceptance | `docs/source/jax_acceptance.rst`, P8 external signoff | Blocked on real CUDA/H200/platform runs |
| CLAUDE.md doc refresh | `CLAUDE.md` validation section | Validation commands refreshed; record pass counts from each run |

## Validation Playbook

Full commands and rationale live in `CLAUDE.md`. Prefer the repo-local
interpreter and the explicit CPU/x64 env prefix for reproducible local evidence:

```bash
# Ruff
.conda/jax/bin/python -m ruff check <changed-files>
.conda/jax/bin/python -m ruff format <changed-files>

# Public pure-JAX unit tests
PYTHONNOUSERSITE=1 PYTHONPATH=src JAX_ENABLE_X64=True JAX_PLATFORM_NAME=cpu \
  .conda/jax/bin/python -m pytest -q \
  tests/test_jax_import_smoke.py \
  tests/field/test_biotsavart_jax.py \
  tests/geo/test_surface_fourier_jax.py \
  tests/geo/test_boozer_residual_jax.py \
  tests/objectives/test_integral_bdotn_jax.py \
  tests/geo/test_boozer_derivatives_jax.py \
  tests/geo/test_boozersurface_jax.py \
  tests/integration/test_jax_native_path.py \
  -m "not private_optimizer_runtime"

# Private optimizer lane
PYTHONNOUSERSITE=1 PYTHONPATH=src JAX_ENABLE_X64=True JAX_PLATFORM_NAME=cpu \
  .conda/jax/bin/python -m pytest -q \
  tests/geo/test_boozersurface_jax_private.py \
  tests/integration/test_section6_public_lane_split.py \
  tests/integration/test_single_stage_jax_cpu_reference.py \
  -m "private_optimizer_runtime"
```

### Backend selection

```bash
SIMSOPT_BACKEND=jax
SIMSOPT_JAX_PLATFORM=cuda
```

```python
from simsopt.backend import get_backend, get_jax_platform, is_jax_backend
```

### Pre-PR signals

| Signal | Status |
|---|---|
| Parity matrix | `.artifacts/parity/20260518-current-head-cpu/all-fixtures.json`; current dirty-tree CPU/JAX evidence, clean post-commit rerun still required |
| Local parity-audit checklist | 0 unchecked local checklist TODOs in `ISSUES_CHECKLIST.md` |
| External P8 signoff | Open; see `P8_EXTERNAL_SIGNOFF.md` |
| PR-readiness artifacts | `.artifacts/pr_readiness_2026-05-16/`; inspect provenance before treating as current |

## Documentation Index

### Start here for current state

| Doc | What it tells you |
|---|---|
| `.artifacts/parity_audit_2026-05-16/EXECUTION_VERIFICATION.md` | Local parity-audit closeout and what was actually verified |
| `.artifacts/parity_audit_2026-05-16/P8_EXTERNAL_SIGNOFF.md` | External CUDA/platform evidence still required |
| `.artifacts/parity_audit_2026-05-16/ISSUES_CHECKLIST.md` | Closed local checklist; historical item-level audit trail |
| `.artifacts/fix_all_2026-05-16/00_SYNTHESIS.md` | Doc-review/fix-all synthesis and provenance; verify against live code before copying checkmarks |
| `.artifacts/lm_minpack_port_plan_2026-05-16/PLAN.md` | LM Track 1/2/3 plan context |
| `.artifacts/lbfgsb_parity_review_2026-05-16/00_SYNTHESIS.md` | L-BFGS-B parity audit context |
| `docs/jax_port_status_2026-05-15.md` | Dated ported-surface table and parity-evidence summary |
| `.artifacts/jax_port_gap_audit_2026-05-13/cpp_port_gap.md` | Symbol-by-symbol C++ -> JAX status |

### Foundational

| Doc | What it tells you |
|---|---|
| `CLAUDE.md` | Module layout, conventions, validation commands; record counts from current runs |
| `/Users/suhjungdae/code/columbia/analysis/jax_port_plan.md` | Full milestone plan: goals, non-goals, scope decisions |
| `/Users/suhjungdae/code/columbia/analysis/jax_port_m0_contract.md` | M0 contract decisions |
| `docs/parity_dual_mode_contract_2026-05-08.md` | Parity / fast / native_cpu mode contract |
| `docs/source/jax_gpu_setup.rst` | GPU environment setup and runbook |
| `docs/source/jax_acceptance.rst` | CPU-vs-JAX acceptance criteria |

### Detailed implementation plans

| Doc | Topic |
|---|---|
| `docs/scipy_lbfgsb_jax_parity_plan_2026-05-15.md` | L-BFGS-B port plan |
| `docs/non_banana_example_cpp_jax_cpu_parity_plan_2026-05-12.md` | Non-banana CPU/JAX parity |
| `docs/boozer_hessian_cpp_oracle_parity_impl_plan_2026-05-05.md` | Boozer Hessian parity |
| `docs/example_cpp_jax_cpu_gpu_parity_expansion_plan_2026-05-14.md` | GPU parity expansion |
| `docs/release_grade_cpp_jax_behavior_preservation_plan_2026-05-02.md` | Release-grade behavior preservation |
| `docs/boozer_full_parity_plan_2026-05-04.md` | Boozer full parity |
| `docs/boozer_derivative_bit_identity_impl_plan_2026-05-07.md` | Boozer derivative bit identity |
| `docs/banana_jax_native_port_todos_2026-05-05.md` | Banana native port todos |
| `docs/cpu_cpp_jax_cpu_full_trajectory_parity_plan_2026-04-28.md` | Trajectory parity |

### Issue trackers

| Doc | Topic |
|---|---|
| `examples/single_stage_optimization/ISSUES.md` | Single-stage example issue tracker |
| `docs/test_single_stage_example_refactor_todos_2026-05-07_v2.md` | Single-stage example refactor todos |
| `docs/jax_gpu_port_todos_2026-04-08.md` | GPU port todos |
| `docs/jax_parity_reduction_todos_2026-04-10.md` | Parity reduction todos |
| `/Users/suhjungdae/code/columbia/analysis/jax_backend_execution_todos_2026-03-31.md` | Backend execution todos |

### Older reviews

Kept for traceability; do not start here:

- `.artifacts/jax-test-audit-2026-04-25/`
- `.artifacts/review-2026-05-10/`
- `.artifacts/parity/20260514-partial-closeout/`
- `/Users/suhjungdae/code/columbia/analysis/jax_port_review_checklist_2026-04-02.md`
- `/Users/suhjungdae/code/columbia/analysis/jax_port_code_review_2026-04-01.md`

## Maintenance

Refresh this file after any new audit, current-HEAD parity matrix, Tier-1
remediation closeout, or active-workstream ownership change. Close checklist
items in their source artifact first, then summarize the state here.
