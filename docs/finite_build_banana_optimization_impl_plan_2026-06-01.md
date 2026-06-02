# Finite-Build Banana Stage-2 Optimization Implementation Plan

## Purpose

This plan defines the work to add a **finite-build optimization mode** to the HBT
banana Stage-2 coil solver
(`examples/single_stage_optimization/STAGE_2/banana_coil_solver.py`). In this mode
each banana coil is represented during optimization as a **multi-filament winding
pack** (Singh et al. 2020, via `simsopt.geo.create_multifilament_grid`) instead of
a zero-thickness filament, with **pack-rotation degrees of freedom** and a
**pack-to-pack clearance penalty**. This ports the upstream FOCUS-like finite-build
Stage-2 pattern (`examples/3_Advanced/stage_two_optimization_finitebuild.py`) onto
the banana geometry so coils are optimized as real conductors with cross-section.

This is distinct from, and complementary to, the already-implemented
`banana_opt/finitebuild_export.py` (export/evaluation-only conversion of a solved
thin-filament artifact). That tool finite-builds *after* optimization; this plan
finite-builds *during* optimization so the field fit (`SquaredFlux`) and geometry
penalties see the real pack.

## Goals

- Add an opt-in finite-build mode to `banana_coil_solver.py`, default **OFF** so
  the existing thin-filament behavior is byte-for-byte preserved when unset.
- When enabled, replace the single banana centerline field source with a
  multi-filament pack built from that same centerline curve, expanded by the
  existing banana symmetry, so `SquaredFlux` fits the real pack field.
- Add the pack-rotation `FrameRotation` degrees of freedom to the optimizer so the
  winding-pack orientation is optimized along the coil.
- Preserve the total banana current: split it across filaments so the pack's net
  current equals the thin-filament banana current, with a single shared banana
  current DOF.
- Keep all existing centerline shape constraints (length, curvature, poloidal
  extent, projected ellipse width, self-intersection) acting on the **centerline**
  curve, unchanged.
- Keep the coil-to-coil and coil-to-surface clearance penalties acting on the
  symmetry-expanded **centerlines** (pack centers), so intra-pack filament pairs
  do not pollute the clearance penalty.
- Emit metadata distinguishing finite-build artifacts (filament counts, gap sizes,
  rotation order, frame, per-pack filament current) and a finite-build-aware
  `COIL_GROUPS` manifest so downstream tools partition the artifact correctly.
- Report a finite-build buildability diagnostic: minimum centerline radius of
  curvature vs the pack binormal half-extent (a pack cannot bend tighter than its
  half-build).
- Add focused tests for coil counts, current conservation, rotation-DOF presence,
  objective construction, artifact reload, and thin-mode parity (mode-off
  regression).

## Non-Goals

- Do not change the thin-filament solve path's numerics, defaults, or outputs when
  finite-build mode is off.
- Do not change `src/simsopt/geo/finitebuild.py` or other core Simsopt primitives
  unless implementation reveals a real upstream bug.
- Do not add plasma-current physics; plasma current remains a VMEC / proxy concern.
- Do not combine finite-build with the `jhalpern30` banana replay path in this task
  (the replay builder has bespoke current/sign logic); fail loudly if combined.
- Do not add per-filament independent currents or per-filament shape DOFs; the pack
  shares one centerline and one rotation profile (the Singh et al. model).
- Do not run long production VMEC sweeps or Boozer-surface boots as part of this
  task; validation is smoke-level + contract-level.
- Do not wire finite-build into the autoresearch `scripts/` launchers in this task
  (a follow-up once the solver-side contract is proven).

## Current Context

- The Stage-2 coil constructor is the single source of truth for coil assembly:
  `banana_opt/stage2_geometry.py:initialize_coils` (`stage2_geometry.py:483-574`).
  - Master banana centerline: `CurveCWSFourierCPP` on the coil winding surface,
    `stage2_geometry.py:509-517` (DOFs `phic(0)`, `thetac(0)`, `phic(1)`,
    `thetas(1)`, ... up to `order`).
  - Standard (non-jhalpern30) field source: a single base curve + single
    `ScaledCurrent(Current(1), banana_init_current_A)` expanded by
    `coils_via_symmetries([banana_curve], [...], surf_coils.nfp, surf_coils.stellsym)`
    (`stage2_geometry.py:527-532`).
  - `jhalpern30` field source: `build_jhalpern30_banana_coils(...)`
    (`stage2_geometry.py:519-525`) — out of scope here.
  - Final coil order: `coils = tf_coils + banana_coils + proxy_coils + vf_coils`
    (`stage2_geometry.py:561`). Returns `bs, curves, banana_curve, banana_coils,
    proxy_coils, vf_coils|vf_build_result` (`stage2_geometry.py:572-574`).
- The objective is assembled in `banana_coil_solver.py:2657-2713`:
  - `Jf = SquaredFlux(new_surf, new_bs)` (`:2657`) — field fit over the full
    BiotSavart, so the banana coils carry the optimizable field.
  - `objective_curves = [coil.curve for coil in new_banana_coils]` (`:2618`) feeds
    `Jccdist = CurveCurveDistance(objective_curves, CC_THRESHOLD)` (`:2659-2661`)
    and `Jcsdist = CurveSurfaceDistance(objective_curves, lcfs_surf, CS_THRESHOLD)`
    (`:2662`).
  - Centerline-only penalties on `new_banana_curve`: `CurveLength` (`:2658`),
    `LpCurveCurvature` (`:2665`), `PoloidalExtent` (`:2666-2670`),
    `ProjectedEllipseWidth` (`:2671-2675`), `CurveSelfIntersect` (`:2676-2682`).
  - `JF` is the weighted sum (`:2703-2712`); `BASE_OBJECTIVE = SQUARED_FLUX_WEIGHT
    * Jf` (`:2713`) for the ALM path.
  - DOFs: `dofs = BASE_OBJECTIVE.x if CONSTRAINT_METHOD == "alm" else JF.x`
    (`:2804`). Optimizer is scipy L-BFGS-B (penalty) or `minimize_alm` (ALM).
- The finite-build primitive is `simsopt.geo.create_multifilament_grid(curve,
  numfilaments_n, numfilaments_b, gapsize_n, gapsize_b, rotation_order=None,
  rotation_scaling=None, frame='centroid')` (`src/simsopt/geo/finitebuild.py:65`).
  It returns `numfilaments_n * numfilaments_b` `CurveFilament` objects that share
  the input curve and **one** `FrameRotation` (the rotation DOFs; `ZeroRotation`
  when `rotation_order is None`). Each `CurveFilament` is a `Curve`, so it is a
  valid input to `coils_via_symmetries` / `apply_symmetries_to_curves`.
- The upstream finite-build Stage-2 example
  (`examples/3_Advanced/stage_two_optimization_finitebuild.py`) demonstrates the
  canonical split this plan reuses: the **field** `bs` is built from the
  symmetry-expanded **filaments**, while `CurveCurveDistance` is applied to the
  symmetry-expanded **centerlines** (`curves = apply_symmetries_to_curves(
  base_curves, ...)`), and the per-coil current is `(I/nfil)` shared across the
  `nfil` filaments.
- Existing tests to extend/mirror: `tests/geo/test_banana_coil_solver_cli_alm.py`,
  `tests/geo/test_banana_objective_modules.py`,
  `tests/geo/test_banana_helper_modules.py`, `tests/geo/test_finitebuild.py`.
- The banana-only exporter already centralizes finite-build defaults discussion in
  `docs/finite_build_biotsavart_generator_impl_plan_2026-06-01.md`; no Stage-2
  hardware contract owns filament counts or gap sizes today, so finite-build
  geometry must be explicit CLI input (with documented local defaults near the
  parser).

## Rationale

The existing solver already factors the banana family into three roles — the
field-carrying coils (`new_banana_coils` -> `new_bs`), the master centerline
(`new_banana_curve`, used by every shape penalty), and the symmetry-expanded
centerlines (`objective_curves`, used by the clearance penalties). Finite build
only needs to change **what carries the field**: swap the single banana base curve
for a filament pack built from the same centerline, while leaving the centerline
and its symmetry-expanded copies as the geometry/clearance references. This is the
smallest change that makes `SquaredFlux` and the clearance penalty finite-build
aware without rewriting the objective or the optimizer, and it matches the upstream
example exactly.

Building the pack inside `initialize_coils` (rather than rebuilding in the solver
after the fact) keeps coil construction in one place and avoids duplicating the
symmetry/current knowledge. The constructor's return contract is left unchanged:
the returned `banana_coils` simply become the filament pack, and the solver derives
the two finite-build-aware references it needs — the clearance centerlines (via
`apply_symmetries_to_curves([new_banana_curve], ...)`) and the shared net-current
optimizable (via the first coil's `current.current_to_scale`).

### Design-it-twice

- **Design A (chosen): centerline-preserving, finite-build-before-symmetry inside
  `initialize_coils`, with the solver deriving the clearance centerlines and net
  current.** Build `filaments = create_multifilament_grid(banana_curve, ...)`;
  expand with `coils_via_symmetries(filaments, [filament_current]*nfil, nfp,
  stellsym)` where `filament_current = ScaledCurrent(banana_net_current, 1/nfil)`.
  Pros: one centerline DOF set + one rotation DOF set + one banana-current DOF
  (preserves existing semantics); shape penalties unchanged; matches upstream;
  no return-contract / test-mock change. Cons: clearance/`CurveSurfaceDistance`
  measured on centerlines, not outer filaments (a small, documented approximation).
- **Design B (rejected): rebuild banana coils as filaments in the solver `main`
  after `initialize_coils` returns.** Rejected: there are two `initialize_coils`
  call sites; this leaks coil-construction knowledge out of the constructor SSOT
  and must re-derive symmetry/current handling at the call site.
- **Design C (rejected): finite-build each already-expanded symmetry copy directly
  (finite-build-after-symmetry).** Rejected: not proven field-equivalent to
  finite-build-before-symmetry for a given frame/rotation (flagged in
  `full_finitebuild_vmec_impl_plan_2026-06-01.md`); it also multiplies the rotation
  DOFs per copy and breaks the single shared current/rotation semantics.

## Assumptions

- Finite-build mode targets the standard (non-jhalpern30) banana path first. The
  jhalpern30 replay path is rejected with a clear error when finite-build is on.
- The pack shares the master centerline curve and one `FrameRotation`; the
  centerline DOFs and rotation DOFs are both free during optimization (rotation
  only when `rotation_order` is not `None`).
- `frame='centroid'` is the robust default (fewer curve-derivative requirements
  than `frenet`); `frenet` is allowed but treated as opt-in/experimental.
- Centerline clearance with the existing `CC_THRESHOLD`/`CS_THRESHOLD` is an
  acceptable first-version approximation; a build-aware clearance (threshold +
  pack half-extent) is a documented follow-up.
- Filament counts and gap sizes are explicit CLI inputs; if defaults are provided
  they live next to the parser and are documented, since no hardware contract owns
  them.
- The output BiotSavart with `CurveFilament` + `FrameRotation` serializes and
  reloads through the existing simsopt GSON path (to be verified by a reload test).

## Implementation Plan

1. Add finite-build CLI surface to `banana_coil_solver.py:parse_args` (default off).
   - [x] `--finite-build` (store_true) master toggle.
   - [x] `--finitebuild-numfilaments-n`, `--finitebuild-numfilaments-b` (ints).
   - [x] `--finitebuild-gapsize-n`, `--finitebuild-gapsize-b` (floats, meters).
   - [x] `--finitebuild-rotation-order` (int; a negative value maps to `None` = no
         rotation DOFs, resolved in `resolve_finite_build_settings`).
   - [x] `--finitebuild-frame` (`centroid`|`frenet`, default `centroid`).
   - [x] `validate_finite_build_cli_args`: positive counts, finite positive gaps,
         and reject `--finite-build` with `--stage2-bs-path` (seeded finite-build is
         a follow-up). Called from `parse_args` AND the `main(parsed_args=...)`
         re-validation block. The non-jhalpern30 guard lives in `initialize_coils`
         (the construction SSOT), not the CLI validator.

2. Extend `initialize_coils` (`stage2_geometry.py`) with a finite-build contract.
   - [x] Add one keyword param `finite_build=None` taking a `FiniteBuildSettings`
         (frozen dataclass: filament counts, gap sizes, rotation order, frame, with
         `nfilaments` / `pack_half_extent_{n,b}_m` properties). `None` = thin mode.
         Settings are built by `resolve_finite_build_settings(args)` and threaded via
         `_build_initialize_coils_kwargs`.
   - [x] When `finite_build` is None, behavior is byte-identical to today.
   - [x] When True and mode is non-jhalpern30, delegate to a module-level helper
         `build_finite_build_banana_coils(master_curve, total_banana_current_A,
         finite_build, surf_coils)` that:
         - builds `filaments = create_multifilament_grid(banana_curve, n, b, gn, gb,
           rotation_order=..., frame=...)` (`nfil = n*b`);
         - structures the current as `banana_net_current = ScaledCurrent(Current(1),
           total_banana_current_A)` and `filament_current =
           ScaledCurrent(banana_net_current, 1/nfil)`, then passes
           `[filament_current] * nfil` with `filaments` to `coils_via_symmetries(...)`
           so net banana current is preserved, one current DOF drives the pack, AND
           the NET-current optimizable stays recoverable (see step 3) so current
           bounds/metadata are correct with unchanged thresholds.
   - [x] When True and mode is jhalpern30: raise a clear `ValueError`.
   - [x] Keep `initialize_coils`' return contract unchanged (the returned
         `banana_coils` are now the filament pack; `banana_curve` is the centerline).
         The clearance centerlines are computed in the solver (step 3), avoiding a
         return-contract change at the call site and the test mock.

3. Wire the solver objective to the centerline clearance curves and net current.
   - [x] At the post-construction merge point, set
         `finite_build_settings = resolve_finite_build_settings(args)` and
         `FINITE_BUILD = finite_build_settings is not None`. When finite-build,
         recover `banana_current_optimizable = new_banana_coils[0].current
         .current_to_scale` (the shared NET current) and set `objective_curves =
         apply_symmetries_to_curves([new_banana_curve], nfp, stellsym)` (pack
         centerlines, equal to thin-mode `[coil.curve for coil in new_banana_coils]`
         because `coils_via_symmetries` is itself `apply_symmetries_to_curves`).
         Thin mode keeps `banana_current_optimizable = new_banana_coils[0].current`
         and `objective_curves = [coil.curve for coil in new_banana_coils]`.
   - [x] Thread `banana_current_optimizable` through every banana-current consumer
         (the penalty/ALM current bound, `_capture_stage2_artifact_state` — whose
         `new_banana_coils` param is replaced by `banana_current_optimizable` — and
         the final net-current metadata/prints) so net-current semantics hold in both
         modes.
   - [x] Leave `Jf = SquaredFlux(new_surf, new_bs)` unchanged: `new_bs` carries the
         filament field automatically in finite-build mode.
   - [x] Leave all centerline penalties (`CurveLength`, `LpCurveCurvature`,
         `PoloidalExtent`, `ProjectedEllipseWidth`, `CurveSelfIntersect`) on
         `new_banana_curve` unchanged.
   - [x] The rotation `FrameRotation` DOFs enter `JF.x` / `BASE_OBJECTIVE.x` through
         the filament -> `Jf` dependency (no manual DOF plumbing); verified by a
         DOF-count delta test (rotation_order=1 adds 3 DOFs; thin current DOF count
         unchanged).

4. Finite-build buildability diagnostic.
   - [x] `_finite_build_artifact_metadata` computes `min_radius_of_curvature = 1 /
         max(centerline kappa)` and pack half-extents `h_b = 0.5 * (numfilaments_b -
         1) * gapsize_b`, `h_n` (via `FiniteBuildSettings.pack_half_extent_{b,n}_m`).
   - [x] Reported in `results.json` as `FINITEBUILD_MIN_CURVATURE_RADIUS_M`,
         `FINITEBUILD_PACK_HALF_EXTENT_{B,N}_M`, and
         `FINITEBUILD_CURVATURE_OK = min_radius > h_b`. (Diagnostic only in this task;
         a hard penalty is a documented follow-up.)

5. Artifact metadata + manifest.
   - [x] The `COIL_GROUPS` manifest already uses `num_banana_coils =
         len(new_banana_coils)`, which equals `nfil * symmetry_copies` in
         finite-build mode, so downstream partitioning
         (`coil_groups.partition_coils_by_manifest`,
         `stage2_single_stage_handoff.partition_loaded_stage2_coils`) stays correct
         with no change.
   - [x] Add `results.json` fields via `_finite_build_artifact_metadata`:
         `FINITE_BUILD_ENABLED`, `FINITEBUILD_NUMFILAMENTS_N/_B`,
         `FINITEBUILD_GAPSIZE_N/_B_M`, `FINITEBUILD_ROTATION_ORDER`,
         `FINITEBUILD_FRAME`, `FINITEBUILD_FILAMENTS_PER_BANANA`,
         `BANANA_FILAMENT_CURRENT_A`, and the step-4 diagnostic fields. Applied to
         the primary artifact `results`; the niche secondary-artifact path omits
         them (documented follow-up).
   - [x] `BANANA_CURRENT_A` stays the net banana current (read from the recovered
         net-current optimizable), so existing consumers read the physical current
         unchanged.

6. Keep finite-build isolated and reversible.
   - [x] No change to thin-mode defaults, output schema values, or numerics
         (thin-mode path is byte-identical; finite-build is gated on `--finite-build`).
   - [x] `--finite-build` help documents that finite-build changes the field model
         and the banana coil count in the output artifact.

## Validation Plan

- [ ] Add `tests/geo/test_banana_finite_build_optimization.py`:
  - [ ] Mode-off regression: with `--finite-build` unset, `initialize_coils`
        returns the same banana coil count, the same `objective_curves`, and the
        same initial objective value as before (parity vs current behavior).
        Covered indirectly: the 307 pre-existing solver/objective tests pass
        unchanged, and `test_rotation_order_adds_only_rotation_dofs...` asserts the
        no-rotation finite-build pack has the SAME DOF count as the thin coil (one
        shared curve + one shared current). A dedicated full-`initialize_coils`
        mode-off objective-parity test is a follow-up (needs a VMEC fixture).
  - [x] Coil count: `test_coil_count_is_filaments_times_symmetry_copies`.
  - [x] Current conservation: `test_net_current_conserved_across_one_pack` (sum per
        pack == net; each filament == net/nfil) and
        `test_net_current_optimizable_recoverable_from_first_coil`.
  - [x] Single current DOF: `test_rotation_order_adds_only_rotation_dofs...` — the
        no-rotation finite-build pack DOF count equals the thin coil's, proving the
        filaments share one current DOF (and the master curve).
  - [x] Rotation DOFs: `test_rotation_order_adds_only_rotation_dofs...` and
        `test_field_objective_and_gradient_finite_through_rotation_dofs`
        (rotation_order=1 -> +3 DOFs; rotation_order=None -> none).
  - [~] Clearance source: enforced in the solver (`objective_curves =
        apply_symmetries_to_curves([new_banana_curve], ...)` in finite-build mode).
        A direct unit assertion was dropped because the minimal test master sits on
        a symmetry plane (coincident copies, `shortest_distance == 0`, NaN gradient
        — the very failure mode the design avoids). The choice is exercised by the
        solver tests; a non-degenerate clearance unit test is a follow-up.
  - [x] Field objective + gradient smoke: `Jf.J()` and `Jf.dJ()` finite through the
        filament pack and rotation DOFs
        (`test_field_objective_and_gradient_finite_through_rotation_dofs`). (Field
        eval also checked in `test_field_evaluates_finite`.) A formal Taylor test
        was not added.
  - [x] Field evaluation: `test_field_evaluates_finite` (`set_points` + `B()` finite,
        expected shape). A save/reload roundtrip of the finite-build `BiotSavart`
        was not added (CurveFilament/FrameRotation GSON serialization is already
        covered by `tests/geo/test_finitebuild.py`).
  - [x] jhalpern30 guard: `test_jhalpern30_with_finite_build_raises`.
  - [x] Solver CLI/metadata helpers: `resolve_finite_build_settings`,
        `validate_finite_build_cli_args` (off / valid / seeded / nonpositive /
        nonfinite), and `_finite_build_artifact_metadata` (counts, filament current,
        buildable + unbuildable curvature diagnostic) — `SolverFiniteBuildHelpersTest`.
- [x] Focused + regression tests run (all pass):
  `tests/geo/test_banana_finite_build_optimization.py` (19),
  `tests/geo/test_ishw_deliverables.py`, `tests/geo/test_banana_objective_modules.py`,
  `tests/geo/test_banana_helper_modules.py`, `tests/geo/test_banana_coil_solver_cli_alm.py`
  (204), `tests/geo/test_single_stage_example.py -k "stage2/initialize/capture/..."`
  (103), `tests/geo/test_finitebuild.py`,
  `tests/geo/test_banana_modularization_parity.py`,
  `tests/geo/test_stage2_single_stage_handoff.py`,
  `tests/geo/test_stage2_track_b_wrappers.py` (142). CLI `--help` lists all 7 flags.
- [x] Manual full-solve smoke (RAN, passed): `banana_coil_solver.py --finite-build
  --finitebuild-numfilaments-n 2 --finitebuild-numfilaments-b 3
  --finitebuild-rotation-order 1 --maxiter 5 --nphi 32 --ntheta 16` on
  `wout_nfp5ginsburg_000_014417_iota15.nc` (fresh init, penalty method). Ran 5
  iterations to the iteration limit (HW violation expected from a cold 5-step start).
  Verified: `BANANA_CURRENT_A = -9999.94` (NET, not net/nfil) and
  `BANANA_FILAMENT_CURRENT_A = -1666.66` (= net/6); `NUM_BANANA_COILS = 60` (6 x 10
  symmetry copies) with a correct contiguous `COIL_GROUPS` manifest; finite-build
  metadata + buildability diagnostic present
  (`FINITEBUILD_MIN_CURVATURE_RADIUS_M = 0.0502` vs `PACK_HALF_EXTENT_B_M = 0.04`,
  `FINITEBUILD_CURVATURE_OK = True`, borderline). The saved `biot_savart_opt.json`
  reloads to 101 coils, preserves the net pack current, and returns finite `B()`.

## Risks and Mitigations

- Risk: double symmetry expansion (expanding already-expanded filaments) duplicates
  coils / corrupts signs.
  Mitigation: finite-build-before-symmetry only — build the pack from the single
  master centerline, then expand once via `coils_via_symmetries`.
- Risk: clearance penalty blows up because `CurveCurveDistance` sees intra-pack
  filament pairs separated by ~gap size.
  Mitigation: apply clearance to the symmetry-expanded **centerlines**, never the
  filaments (verified by the clearance-source test).
- Risk: thin-mode behavior drifts because the constructor return contract changed.
  Mitigation: mode-off regression test asserting identical coil count,
  `objective_curves`, and initial objective; default off everywhere.
- Risk: `frenet` frame needs higher curve derivatives that `CurveCWSFourierCPP` may
  not provide.
  Mitigation: default `centroid`; cover `frenet` only behind explicit opt-in and
  test it; fail loudly if a derivative is missing rather than silently degrading.
- Risk: rotation DOFs are silently ignored (not picked up by `JF.x`).
  Mitigation: assert the DOF-count delta when finite-build + `rotation_order` are
  set; test the DOF count explicitly.
- Risk: downstream tools mis-partition the larger banana coil count.
  Mitigation: write a finite-build-aware `COIL_GROUPS` manifest and test round-trip
  partitioning of the saved artifact.
- Risk: the centerline can curve tighter than the pack half-build (unbuildable).
  Mitigation: report the curvature-radius-vs-half-build diagnostic; flag a
  follow-up to make it a hard penalty if needed.
- Risk: current object aliasing couples output filament currents to a source seed.
  Mitigation: construct fresh `Current`/`ScaledCurrent` for the pack; test the
  source object is not mutated.

## Completion Criteria

- [ ] `--finite-build` opt-in works end-to-end on the standard banana path and is
  default off; thin-mode outputs/numerics are unchanged.
- [ ] In finite-build mode the field comes from the symmetry-expanded filament pack,
  the rotation DOFs are optimized, and net banana current is preserved.
- [ ] Clearance and all centerline shape penalties remain correct (clearance on
  centerlines; shape on the master centerline).
- [ ] The saved BiotSavart reloads and carries a finite-build-aware `COIL_GROUPS`
  manifest and finite-build metadata.
- [ ] The buildability diagnostic (min curvature radius vs pack half-build) is
  reported in the CLI summary and `results.json`.
- [ ] Focused finite-build, objective, and CLI tests pass locally.
- [ ] The jhalpern30 + finite-build combination fails loudly.

## Open Questions

- Should finite-build clearance use the outer-filament envelope (build-aware
  threshold) from the start, or is centerline clearance + the buildability
  diagnostic sufficient for the first version?
- Should the curvature-radius-vs-half-build relationship be a hard objective term
  (penalty) now, or remain a reported diagnostic until a real run needs it?
- Should one shared pack geometry serve all banana symmetry copies (current plan),
  or will any future case need per-copy filament settings?
- Should a future task wire `--finite-build` into the autoresearch `scripts/`
  launchers and the single-stage path, or keep it Stage-2-solver-only?
- Should finite-build artifacts be accepted as single-stage warm-start seeds, or
  remain optimization outputs that the existing export/handoff contracts consume?
