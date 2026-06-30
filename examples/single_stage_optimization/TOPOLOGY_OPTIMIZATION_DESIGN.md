# Topology-Aware Multi-Surface Confinement Optimization — Design & Implementation Plan

> Status: IMPLEMENTED-IN-WORKING-TREE, NOT YET COMMITTED/MERGED. Last updated: 2026-06-24 (header reconciled).
> All four phases are IMPLEMENTED + crucible-reviewed per the Review Log at the bottom of this file
> (Phase 1a VolumeBoozer, 1b iota/volume profile objectives, 2 `boozer_surface_family.py` [347 LOC/11 tests],
> 3 `topology/chirikov_overlap.py` [314 LOC/21 tests], 4 `topology/mather_dw.py` [276 LOC/27 tests]) and present
> in the tree. The prior "DESIGN / Phase 1-4 not applied" header was STALE — it contradicted this doc's own
> Review Log. REMAINING to truly close: (a) commit/merge — several new modules are untracked (Pattern-5 commit
> blockers flagged in the Review Log: `git add boozer_surface_family.py`, `topology/chirikov_overlap.py`,
> `topology/mather_dw.py` + their tests before commit, else a clean checkout ModuleNotFounds); (b) tick the
> Completion-Criteria checkboxes below once merged; (c) the multi-surface *optimization* runs themselves
> (the concurrent/Perlmutter thread) are separate from this code being implemented.
> Scope: extend the banana-coil single-stage optimizer from a single Boozer surface to
> multi-surface, topology-aware confinement optimization (nested family, per-surface
> iota/volume profile, Greene residue [already wired], WBA [already coded], Chirikov
> overlap [new], Mather ΔW [new]).
> Authoritative file:line references below were VERIFIED against the tree on 2026-06-23 via
> the doc-review-fix-loop — objective/gradient/optimizer/topology/preset symbols and the
> latent `MagneticWellVolumeShortfall` bug were all confirmed by reading source (not just grep).

## Purpose

Capture, in a durable spec, the design produced by a read-only multi-agent investigation
so it can be executed and reviewed against a written plan rather than a chat transcript.

The motivation is physical. A field-line-topology investigation of the certified
`slid_clean_R0p9095_2026-06-22` design found that its **edge is a non-resonant
X-point divertor**: a genuine hyperbolic X-point separatrix at (R,Z)=(1.0258, 0)
(return-map Jacobian det≈1.000 [symplectic], eigenvalues 0.895/1.117, Greene residue
−0.003), bounding a ~0.18 m-thick stochastic island-overlap layer formed by overlapping
high-m, n=5 island chains in a **nontwist** (iota→0, negative-shear, shearless-curve)
edge. The current single-surface objective never sees any of this. The goal is to give
the optimizer topology-aware levers (KAM / residue / WBA / Mather / Chirikov) so it can
optimize the confined region, not just converge one flux surface.

## Goals

- A reusable nested-Boozer-surface **family** helper with tolerant (truncation) behavior.
- **Per-surface iota-profile and volume objectives** with live coil-DOF gradients.
- Fix the **latent coil-DOF-dead volume gradient** in `MagneticWellVolumeShortfall`.
- A **Chirikov island-overlap scalar** (diagnostic + objective).
- A **Mather ΔW** last-torus-flux certification module.
- An explicit architecture: **cheap topology metrics in the inner loop, expensive
  certification at checkpoints**.
- Every new term **default-OFF (weight 0)** so current single-surface runs stay
  byte-identical until explicitly enabled.

## Non-Goals

- Re-implementing residue / WBA / converse-KAM — they already exist in the fork (see
  Current Context). We reuse, not rebuild.
- Folding the legacy 1–2 surface path (`initialize_surface_data_in_config_order`) into
  the family helper — that is a separate warm-start/resume lane; Tier-2 follow-up.
- Running a full multi-surface re-optimization — that is execution, not this plan.
- Mather ΔW in the optimizer inner loop — too expensive; certification-only.
- Mather's rigorous guarantees *at* the shearless curve — nontwist breaks Aubry–Mather
  there; ΔW is reported as heuristic and paired with converse-KAM.

## Current Context

Confirmed facts (file:line verified 2026-06-23). Paths relative to
`examples/single_stage_optimization/` (= `SS`); driver lives under `SS/SINGLE_STAGE/`.
Artifact root for the cited topology figures/data is
`/Users/suhjungdae/code/columbia/autoresearch/campaigns/balance_pareto_singlestage_2026-06-17/runs/slid_clean_R0p9095_2026-06-22`.
That path is outside this repo; the repo-root `runs/slid_clean_R0p9095_2026-06-22/` path
does not exist in this checkout.

**Objective + single-surface gap**
- Objective assembler `build_total_objective` — `SS/banana_opt/single_stage_objectives.py:660`.
- Per-surface aggregator `average_surface_objectives` — `single_stage_objectives.py:59`.
- Per-surface term lists exist (`surface_iota_terms = [Iotas(s) for s …]` at
  `SS/SINGLE_STAGE/single_stage_banana_example.py:9413`, `nonQSs:9414`, `brs:9419`), but
  **iota and volume objectives read the OUTER surface only** — both read-VERIFIED:
  iota `build_single_stage_iota_objective(surface_iota_terms[-1], …)` at `:9560-9561`, and
  volume `surface_volume_term = Volume(outer_surface)` at `:9559`. `JVolume` is `None` in
  default (target) goal-mode.
- Existing per-surface/profile terms (default-off): `IotaShearShortfall:81`, `RationalIotaAvoidance:202`, `MagneticWellVolumeShortfall:415`.

**Multi-surface machinery already present (default-off)**
- A native nested **continuation** helper already exists:
  `initialize_published_surface_data_from_stage2_seed` — `single_stage_banana_example.py:5100`
  (volume-contraction marcher `contract_surface_to_target_volume` — `:4857`). It marches
  edge→inner, warm-starts `(iota, G, DOFs)`, and enforces nesting/volume-ordering
  postconditions. The inventory line "no native nested-volume family (write a loop)" is
  therefore **overstated** — the loop exists; we factor + generalize it.
- Multi-surface presets are just `label_fractions` tuples — `interior_covering_deep` =
  `(0.2, 0.4, 0.6, 0.8, 1.0)`, 5 surfaces (`banana_opt/surface_mode_contracts.py:63,67`).

**Gradients / adjoints**
- `Iotas` (`src/simsopt/geo/surfaceobjectives.py:1001`) `.dJ(partials=True)` returns a
  **live coil-DOF gradient** via the BoozerSurface adjoint. Confirmed by
  `tests/geo/test_surface_objectives.py:396` (`test_iotas_derivative`, Taylor-tested).
- `Volume` (`surfaceobjectives.py:192`) is **coil-DOF-dead** — VERIFIED by reading the
  class: `__init__` does `super().__init__(depends_on=[self.surface])` (coils are NOT a
  parent) and `dJ -> Derivative({self.surface: self.surface.dvolume_by_dcoeff()})`
  (`:226-234`), so projecting onto coil DOFs gives **ZERO**. This is the "trap": any volume
  term built from bare `Volume(surface)` has a coil-DOF-dead gradient. (`dvolume_by_dcoeff`
  is the geometric C++ derivative, `src/simsoptpp/surface.h:240`.)
- `MajorRadius` (`surfaceobjectives.py:699`) is the proven **coil-live** adjoint template
  (forward_backward over the cached PLU + `dconstraint_dcoils_vjp`); it is constructed from
  the **`boozer_surface`** (not `.surface`) — e.g.
  `MajorRadius(surface_data[-1]["boozer_surface"])` at `single_stage_banana_example.py:9816`.
  `VolumeBoozer` must take the `boozer_surface` the same way.
- Optimizer: ALM (`SS/alm_utils.py`) over `scipy.optimize.minimize(method="L-BFGS-B", jac=True)`
  (`alm_utils.py:3632-3633`) with analytic Boozer-adjoint gradients.

**Topology package already shipped** (`SS/banana_opt/topology/`)
- `greene_residue.py` — `greene_residue_from_trace:22`, `classify_greene_residue:29`,
  `greene_residue_diagnostic_from_matrix:42`.
- `periodic_orbit.py` — `solve_periodic_orbit:229`, `discover_periodic_orbit:320`
  (RK4 + damped Newton, monodromy; returns O/X orbits + trajectories).
- `kam_birkhoff.py` — `weighted_birkhoff_average:167` (**WBA already exists**).
- `residue_objective.py` — `BiotSavartGreeneResidueObjective:245` (differentiable Greene
  residue objective; analytic adjoint VJP in `residue_sensitivity.py`,
  `branch_resolved` FD mode markers `:52-53`).
- `converse_kam.py`, `iota_profile.py` present.
- `BiotSavart` exposes `.A()`, `.A_cyl()`, `.A_vjp()`, `.A_and_dA_vjp()` (needed for the
  Mather field-line action ∮A·dl).

## Rationale

- **Profile, not averaging.** With negative shear the physics target is an iota *profile*
  iota(s) that differs per surface. Averaging the per-surface iotas to one scalar would
  fight shear (and the existing `IotaShearShortfall`). So the per-surface extension is a
  profile residual, not an average.
- **Reuse over rebuild.** Residue, WBA, converse-KAM, periodic-orbit solving, island-width
  ranking, and the nested continuation already exist. New code is limited to: one
  `VolumeBoozer` adjoint wrapper, two profile-objective builders, a Chirikov scalar, and a
  Mather ΔW module — each composing existing, tested primitives.
- **Cheap-inner / expensive-certify.** Greene residue (wired) + WBA (coded) + Chirikov
  (pendulum tier) are cheap enough for the inner loop; Mather ΔW (2 periodic-orbit solves
  per convergent × q-turn tangent integrations) is orders of magnitude costlier → it is a
  certification tool run at checkpoints.
- **Truncation = the default-mode-Poincaré analog.** Default-mode Poincaré tolerates lost
  edge lines and reports a survival fraction (42/50) without aborting; the surface family
  should likewise keep the surfaces that converge and report "k/N, last good at R=…".
  Same physics, same boundary: a non-converging inner surface and a lost field line both
  hit the stochastic edge / X-point. **But** the optimizer inner loop needs a *fixed*
  surface count for a smooth objective, so truncation is applied at (re)init and then held
  fixed within each inner loop (re-truncate at outer-loop / restart boundaries).

## Assumptions

- Per-surface Boozer adjoints remain valid for an N-surface family (each surface is an
  independent adjoint solve, as already exercised by the nested-spacing constraint). **To
  be Taylor-tested**, not assumed in code.
- `surface_data[i]["target_volume"]` entries are strictly decreasing (nesting) — true by
  construction of `contract_surface_to_target_volume`.
- The **shear-shape** iota target (edge value anchored to the existing `iota_target` + a
  target slope dι/ds) is the desired default form, over explicit per-surface iota targets.
- Standard-map calibration values: Mather golden-torus breakup K_c ≈ 0.971635; Chirikov
  onset s ≈ 1 (refined "2/3 rule" s_crit ≈ 0.67).

## Decisions (settled 2026-06-23)

- **Surface-family policy: truncation** — truncate-at-init then hold fixed per inner loop;
  re-truncate at outer-loop/restart. `min_surfaces` floor below which the run genuinely
  fails: **default 3** (the magnetic-well V″(s) proxy needs 3 surfaces). Strict
  (fail-if-<N) remains available as the validation-gate analog.
- **N surfaces: 5** (`interior_covering_deep`, s-fractions 0.2…1.0).
- **iota target form: shear-shape** (edge anchor + negative target slope).
- **All new terms default-OFF** (weight 0); single-surface mode reduces to current
  behavior; builders return `None` for <2 (or <3) surfaces.
- **Label stays `Volume`** for the family (the pipeline is volume-keyed end-to-end;
  normalized toroidal flux `s` remains the spacing coordinate, handing off to a Volume
  target for the constrained solve).

## Implementation Plan

Build order is by leverage/effort. Phase 0 needs the cores free; Phases 1–4 are mostly
edits + light unit tests (less CPU-bound than tracing).

1. **Phase 0 — Diagnostic baseline (NO new code; uses tools that already exist).**
   - [ ] Run `kam_birkhoff` WBA classifier over a radial seed comb on the current design
         → baseline **last-KAM-surface radius** and **edge chaotic fraction**.
   - [ ] Run the Greene-residue probe / `run_residue_probe.py` → residue profile incl. the
         edge X-point (expect ≈ −0.003 at R≈1.0258).
   - [ ] Run `converse_kam` over the edge band → independent non-existence certificate of
         the loss boundary.
   - [ ] Record baselines under the artifact root above (or copy them into an intentional
         repo-local path if this plan is promoted to tracked docs). These are the numbers
         the new objectives will move.

2. **Phase 1 — Unit #1: per-surface profile objectives + the latent-bug fix (smallest, highest leverage).**
   - [ ] Add `VolumeBoozer(Optimizable)` to `src/simsopt/geo/surfaceobjectives.py`, a
         clone of `MajorRadius` (read-verified `:699-748`). EXACTLY TWO lines of `compute()`
         change: `self._J = surface.major_radius()` -> `surface.volume()`, and
         `dj_ds = surface.dmajor_radius_by_dcoeff()` -> `surface.dvolume_by_dcoeff()`.
         Everything else is identical and inherited: `depends_on=[boozer_surface]`,
         `run_code_from_last_solution()`, cached `res['PLU']`/`res['vjp']`,
         `adj = forward_backward(P,L,U,dJ_ds)`,
         `dconstraint_dcoils_vjp(adj, booz_surf, iota, G)`, `self._dJ = -1*adj_times_dg_dcoil`
         -> **live coil-DOF volume gradient** (the whole reason this wrapper exists).
   - [ ] Build `IotaProfileShortfall` (shear-shape target) by composing
         `average_surface_objectives([QuadraticPenalty(surface_iota_terms[i], target_i)…])`
         in `single_stage_objectives.py` — reuse the existing `surface_iota_terms` list;
         **no new Iotas list, no new class needed** (iota gradient already live).
   - [ ] Build `VolumeProfileShortfall` from `[VolumeBoozer(e["boozer_surface"])…]` against
         `e["target_volume"]`, weighted-averaged; soft objective (NOT an ALM constraint).
   - [ ] Fix the latent bug **at its source — the driver, not the well class**. The class is
         fine: it calls `.dJ(partials=True)` on the volume terms it is *given*
         (`single_stage_objectives.py:510-512`). The dead gradient comes from the driver
         building those terms as bare `Volume(entry["boozer_surface"].surface)`
         (`single_stage_banana_example.py:9862-9863`, passed at `:9868`). Build them as
         `VolumeBoozer(entry["boozer_surface"])` instead -> the well's descent gradient
         becomes coil-live. (It is wired into descent at `single_stage_objectives.py:822-823`,
         so today it is a silent no-op whenever `MAGNETIC_WELL_WEIGHT > 0`.)
   - [ ] Thread two new params into `build_total_objective:660` + `evaluate_total_objective`,
         default `None`/0.0, added in the iota-family block (mirror the `JShear` pattern);
         add CLI weights (default 0). Builders return `None` for <2 surfaces.

3. **Phase 2 — Unit A: factor the nested-volume family helper + truncation.**
   - [ ] Extract the continuation body (`single_stage_banana_example.py:5100-5186`) into a
         reusable `banana_opt/boozer_surface_family.py::build_boozer_surface_family(...)`;
         make `initialize_published_surface_data_from_stage2_seed` a thin caller (DRY).
   - [ ] Replace the all-or-nothing raise with **fail-closed-to-last-good**: use the
         non-raising `banana_opt/stage2_single_stage_handoff.py:1425`
         `attempt_initialize_boozer_surface` helper (currently not imported by the driver);
         accept a shell iff solved AND nested-inside the last accepted AND volume-ordered;
         stop at the first failure.
   - [ ] Add `min_surfaces` (default 3, matching the magnetic-well floor) +
         `allow_truncation` (default true for this family helper; false for strict
         validation); raise if accepted < `min_surfaces`; in strict mode also raise if
         accepted < `len(label_fractions)`; never fabricate a surface.
   - [ ] Return `_surface_data_entry`-shaped list so all per-surface terms consume it unchanged.

4. **Phase 3 — Unit D: Chirikov island-overlap scalar.**
   - [ ] New `topology/chirikov_overlap.py`: s = (W_m/2 + W_{m+1}/2)/|r_{m+1}−r_m| over the
         n=5 chains from `iota_profile` rational crossings; W from `residue_seed_builder`
         widths / `periodic_orbit` O-X separation; aggregate = max (diagnostic) or
         soft-LSE (objective).
   - [ ] **Nontwist handling:** when `|local_shear| < floor` (shearless band) route twin
         pairs through measured **O/X separation** (finite), not the 1/√(shear) pendulum
         width; config threshold (2/3 vs 1), not hard-coded.

5. **Phase 4 — Unit C: Mather ΔW certification module.**
   - [ ] New `topology/mather_dw.py` reusing `solve_periodic_orbit` for O (minimizing) and
         X (minimax) orbits; action W = ∮A·dl via `BiotSavart.A()` on the returned
         trajectories (dl exact from the field-line ODE; closed-loop gauge-invariant).
   - [ ] ΔW(p_i/q_i) = W_X − W_O over noble convergents → flux Φ(ω*); ΔW→0 ⇔ torus intact.
   - [ ] Run **certification-only** (offline / checkpoints), NOT inner loop; at the
         shearless curve label ΔW heuristic and cross-certify with `converse_kam`.

## Validation Plan

- [ ] **VolumeBoozer Taylor/FD test** (the load-bearing one): Taylor-test J/dJ wrt coil DOFs,
      mirroring `test_major_radius_derivative`; the bare-`Volume` path must FAIL this, the
      wrapper must PASS (atol ≈ 2e-8). Proves the dead-gradient fix.
- [ ] **Iota-profile gradient test**: Taylor-test the composed term wrt `bs.x` on a ≥2-surface
      stack (rides the proven `Iotas` adjoint).
- [ ] **Byte-identical default regression**: with all new weights 0 and single-surface,
      `build_total_objective` output is byte-identical to a captured pre-change baseline.
- [ ] **Family nesting/ordering test**: `cross_sections_are_nested` for every adjacent pair
      and `np.all(np.diff(volumes) > 0)`; default truncation returns the outer band with
      provenance and stays ordered; strict validation mode raises and fabricates nothing.
- [ ] **Chirikov standard-map calibration**: s crosses 1 when analytic half-widths sum to the
      spacing; nontwist regression returns the twin pair and stays finite (no 1/√0).
- [ ] **Mather standard-map calibration**: ΔW(golden) → 0 for K < K_c = 0.971635 and lifts
      off > 0 above it; gauge-invariance unit test (A → A+∇f leaves W unchanged).
- [ ] **Cross-consistency on the real map**: ΔW→0 agrees with Greene-residue breakup and WBA
      torus survival for a monotone-band torus.
- [ ] **Reviewer gate**: each implemented unit closed by a `crucible` adversarial review to PASS
      (and `py_compile` + `ruff check` + `ruff format` clean on changed lines).

## Risks and Mitigations

- Risk: gradients computed *through* a field-line trace are noisy.
  Mitigation: prefer implicit-function (for fixed points) and finite-difference over naive
  autodiff through the integrator; reserve adjoint VJP for the cheap pendulum/action terms.
- Risk: nontwist (iota→0) breaks the standard formulas — Chirikov width 1/√(shear)→∞, Mather
  min/minimax ill-posed at the shearless curve.
  Mitigation: Chirikov twin-pair O/X separation; Mather flagged heuristic + paired with
  converse-KAM (rigorous non-existence) and the nontwist-residue indicator.
- Risk: surface count flickering between optimizer iterations makes the objective
  discontinuous → L-BFGS-B fails.
  Mitigation: truncate-at-init then hold fixed within each inner loop; re-truncate only at
  outer-loop/restart boundaries.
- Risk: enabling `MagneticWellVolumeShortfall` as a descent term today silently does nothing
  (dead volume gradient).
  Mitigation: the `VolumeBoozer` swap in Phase 1; a regression test asserting nonzero dJ.
- Risk: Mather ΔW cost blows up the optimization.
  Mitigation: certification-only; if used in-loop at all, a single dominant convergent with
  the envelope adjoint, full sequence only at certification.

## Completion Criteria

- [ ] Phase 1 merged: `VolumeBoozer` + iota/volume profile terms + well-bug fix, all
      default-OFF, Taylor tests PASS, byte-identical default regression PASS, crucible PASS.
- [ ] Phase 2 merged: `build_boozer_surface_family` with truncation, nesting/truncation tests
      PASS, refactor proven byte-identical on a fixed seed, crucible PASS.
- [ ] Phase 3 merged: Chirikov scalar with standard-map + nontwist tests PASS, crucible PASS.
- [ ] Phase 4 merged: Mather ΔW certification with standard-map K_c + gauge tests PASS,
      crucible PASS.
- [ ] This document updated to reflect what shipped (checkboxes ticked, file:line of new code).

## Open Questions

- Green-light to start **Phase 0 (baseline)** then **Phase 1 (Unit #1)**? (Awaiting user.)
- Should Phase 0 baselines run now, or after current local CPU/GPU capacity is checked?
- Doc home: kept here (`SS/TOPOLOGY_OPTIMIZATION_DESIGN.md`, matching the other top-level
  pipeline docs) — move to `autoresearch/docs/` if preferred.

## Provenance

Design produced 2026-06-23 by a read-only multi-agent investigation (units: nested-volume
family, per-surface iota/volume, Mather ΔW, Chirikov overlap; plus a prior scoping pass:
objective architecture, simsopt capability inventory, WBA/topology-objective menu). The
edge-topology motivation (non-resonant X-point divertor; X-point R=1.0258, det≈1,
λ=0.895/1.117, residue −0.003; ~0.18 m stochastic island-overlap layer; nontwist edge) was
established by a separate two-wave field-line-topology investigation on the same design.
Figures:
`/Users/suhjungdae/code/columbia/autoresearch/campaigns/balance_pareto_singlestage_2026-06-17/runs/slid_clean_R0p9095_2026-06-22/{connlength_vs_R,edge_poincare_offsym_zoom,edge_manifold_legs,iota_vs_R_edge}.png`.

## Review Log (loop-driven crucible, 2026-06-23)

Appended by the topology-optimization /loop (crucible + fix per implemented item). Reflects the working tree (uncommitted) at review time.

- **Phase 1a — VolumeBoozer**: IMPLEMENTED + crucible-PASS. `surfaceobjectives.py::VolumeBoozer` (coil-live volume Boozer adjoint); Taylor/FD + dead-gradient + unsolved-surface contract tests (`tests/geo/test_surface_objectives.py::VolumeBoozerTests`, plus added to the `IotasTests` contract test).
- **Phase 1b — iota/volume profile builders + driver wiring**: IMPLEMENTED (concurrent session) + crucible-reviewed. `build_single_stage_iota_profile_objective` (shear-shape target) + `build_single_stage_volume_profile_objective` (coil-live `VolumeBoozer` terms), default-OFF, threaded into `build_total_objective`/`evaluate_total_objective`. Builder gradient/None/ValueError tests added: `tests/geo/test_single_stage_profile_objectives.py` (10 pass, incl. real 2-surface Boozer Taylor).
- **CRITICAL found + FIXED (iter 2)**: the new profile kwargs were threaded into `build_`/`evaluate_total_objective` + the driver wrappers but NOT the `evaluate_base_objective`/`evaluate_alm_objective` IMPLS → unconditional `TypeError` on every run's finalization + every ALM eval (broke default-OFF). Fixed: 4 params added to both impl sigs; threaded through `evaluate_base_objective`'s `_optional`/sum/breakdown (mirroring `JMagneticWell`); `evaluate_alm_objective` forwards by keyword. Verified: `SingleStageObjectiveModuleTests` 39 pass, introspection clean, ruff clean. (Advisory: `evaluate_total_objective`'s breakdown DICT still omits the profile per-term entries — not a crash; its scalar via `build_total_objective` is correct.)
- **Advisory — `band_shear_load`/`BandShearLoad`** (`topology/iota_profile.py`): defined + tested but NO production caller and never committed → Pattern-6 dead scaffold. Wire into the residue/topology diagnostic path or document as an offline diagnostic, else drop.
- Phases 2 (family helper), 3 (Chirikov), 4 (Mather ΔW): NOT yet implemented.

### Review Log — Phase 2 (loop iter 3, 2026-06-23)

- **Phase 2 — `build_boozer_surface_family` + truncation**: IMPLEMENTED (concurrent session, `banana_opt/boozer_surface_family.py`, 347 lines) + crucible-PASS (functionally correct). DRY thin-caller refactor of `initialize_published_surface_data_from_stage2_seed` is byte-identical to the legacy strict continuation (`_solve_or_raise` verbatim; same march/warm-start/provenance/postconditions). Truncation accept-logic correct (solved ∧ nested ∧ volume-ordered, stop-at-first-failure, never fabricates). `tests/geo/test_boozer_surface_family.py` 8 pass.
- **🟠 COMMIT BLOCKER (Pattern 5):** `banana_opt/boozer_surface_family.py` is UNTRACKED but imported at module-top by the tracked-modified driver (`single_stage_banana_example.py:141`) → a clean checkout would `ModuleNotFoundError`. `git add` it + `tests/geo/test_boozer_surface_family.py` (+ iter-2's `tests/geo/test_single_stage_profile_objectives.py`) before any commit.
- **Advisory (test gap):** `_require_relaxed_family_postconditions` raise paths (the doc-mandated band `np.diff>0` + adjacent-nesting checks) and the non-finite warm-start-G raise are untested — the per-step march guard makes a bad band practically unreachable, and the strict-path sibling postcondition is covered at the driver level (`test_single_stage_example.py`), so this is advisory not blocking. Add direct unit tests of `_require_relaxed_family_postconditions` (disordered band → "strictly"; non-nested band → "must nest") and a non-finite-G march test when the file stabilizes.
- **Advisory (doc/code drift):** `build_boozer_surface_family(allow_truncation=False)` defaults strict; this doc's Decisions/Phase-2 say the family helper defaults `True`. No live bug (the sole caller passes `False` explicitly for byte-identity). Reconcile: recommend keep `False` default + amend the Decisions line, since flipping is riskier with an explicit-False caller already present.
- Phases 3 (Chirikov), 4 (Mather ΔW): still NOT implemented. `band_shear_load` still has no production caller (Pattern-6 advisory from iter 2 unaddressed).

### Review Log — Phase 3 (loop iter 4, 2026-06-23)

- **Phase 3 — Chirikov island-overlap scalar**: IMPLEMENTED (concurrent session, `banana_opt/topology/chirikov_overlap.py`, 314 lines) + crucible-PASS (physics correct). Pendulum full-width `C=4` derivation correct; Chirikov `s = Σ(half-widths)/spacing` with NO off-by-2 (s=1 at full-widths [1,1]/spacing 1); LSE is a numerically-stable smooth-MAX (not min); nontwist routing correctly raises on `1/√0` and routes to the measured O/X separation (never fabricates); edge cases (zero spacing, <2 crossings, non-finite) handled. Not yet wired into the driver (standalone diagnostic/objective, as designed). `tests/geo/test_chirikov_overlap.py` 20→**21 pass**.
- **FIXED (iter 4):** added `test_analytic_pendulum_half_widths_summing_to_spacing_cross_one` — the doc's headline calibration ("s crosses 1 when the ANALYTIC half-widths sum to spacing") was previously tested only with a literal width; the new test drives `pendulum_full_width(0.0625, 1.0)==1.0` → `s==1.0` and brackets ±10%, so a pendulum-coefficient regression is now caught at the onset.
- **Advisory (low):** `CHIRIKOV_TWO_THIRDS_THRESHOLD` (refined s_crit≈0.67) is exported but untested (constant hard to mis-set; primary s≈1 onset is covered).
- **Pattern-5 note:** `chirikov_overlap.py` + test are untracked but NOT yet imported by production (only the test imports it), so no clean-checkout crash today — still `git add` before commit.
- Phase-2 deferred test gaps (relaxed band-postcondition raise paths + non-finite-G) remain open advisories (files now stable; low value — practically unreachable via the march guard, strict sibling covered at driver level).
- Phase 4 (Mather ΔW): still NOT implemented.

### Review Log — iter 5 (caught-up, 2026-06-23)

- **No new implementation since iter 4** (Phase 4 / Mather ΔW still absent; concurrent actor paused). All implemented phases (1a/1b/2/3) remain reviewed + fixed; my prior fixes verified intact (no collision).
- **FIXED (deferred Phase-2 gap, files now stable):** added `RelaxedBandPostconditionTests` to `tests/geo/test_boozer_surface_family.py` (8→11 pass) — directly exercises `_require_relaxed_family_postconditions` raise paths (volumes-not-strictly-increasing → "strictly"; adjacent-pair-not-nested → "must nest") + the clean-band pass. This closes the doc's Phase-2 Validation headline (`cross_sections_are_nested` per adjacent pair + `np.all(np.diff(volumes) > 0)`), previously unreachable via the march guard.
- **Remaining open items (all low-value / not-code or owner=elsewhere):** Pattern-5 commit blocker (`git add` the untracked new modules before commit — user's git domain); `CHIRIKOV_TWO_THIRDS_THRESHOLD` untested (constant, hard to mis-set); `allow_truncation` default vs doc-decision drift (reconcile doc); `band_shear_load` dead scaffold (wire or drop — actor's domain); non-finite warm-start-G raise untested (deliberate guard).
- **Loop status: caught up to the implementation.** Phases 1a/1b/2/3 done+reviewed+fixed; Phase 4 awaits implementation by the concurrent author, after which the loop will review it.

### Review Log — Phase 4 + ALL PHASES COMPLETE (loop iter 6, 2026-06-23)

- **Phase 4 — Mather ΔW certification**: IMPLEMENTED (`banana_opt/topology/mather_dw.py`, 276 lines, standalone/offline as designed) + crucible-PASS (physics correct). Gauge-invariant closed-loop ∮A·dl (telescopes to round-off on planar AND non-planar loops; Stokes flux = B0·πR²); ΔW = W_X − W_O sign correct; golden/noble convergents hand-verified (Fibonacci/Pell); standard-map K_c≈0.9716 calibration is GENUINE (textbook Morse indices X=1/O=0; ΔW(5/8) lifts 7.9e-9→5.0e-5→7.6e-4 across K_c from real Newton-solved orbits; mutation test catches a rigged orbit pair). `tests/geo/test_mather_dw.py` 27 pass.
- **FIXED (iter 6):** clarified the `mather_delta_w_from_orbits` docstring — a `solve_periodic_orbit` result exposes orbit samples as cylindrical `states` (N,2) + `phi_grid`, not a Cartesian `(N,3)` trajectory; the docstring now names the `[R cosφ, R sinφ, Z]` conversion (closes the doc-spec-vs-reality gap; these routines expect (N,3)).
- **Advisory (low):** `MatherCertificate` is a passive container (no deepest-convergent enforcement — moot, no builder ships); the doc's real-map cross-consistency (ΔW vs Greene-residue/WBA) is explicitly **scoped out** in the module docstring (disclosed, not silently missing).

## ✅ ALL FOUR PHASES IMPLEMENTED + CRUCIBLE-REVIEWED + FIXED (2026-06-23)

| Phase | Status | Tests |
|---|---|---|
| 1a VolumeBoozer | PASS | Taylor/FD + dead-gradient + contract |
| 1b iota/volume profile builders | PASS (CRITICAL crash fixed) | builder gradient + default-OFF |
| 2 boozer_surface_family | PASS | 11 (incl. band-postcondition) |
| 3 Chirikov overlap | PASS | 21 (incl. analytic calibration) |
| 4 Mather ΔW | PASS | 27 (gauge + standard-map K_c) |

**Remaining (owner = user / author, not loop-fixable):** the Pattern-5 commit blocker (`git add` the untracked new modules: `boozer_surface_family.py`, `topology/chirikov_overlap.py`, `topology/mather_dw.py`, and the 4 new test files, before committing); `band_shear_load` dead scaffold (wire or drop); `allow_truncation` default vs doc-decision drift; two minor untested guards (Chirikov 2/3 threshold, non-finite-G); the real-map cross-consistency test (scoped out).

> **Update (iter 8):** Phases 1-4 above were committed in `953c7a5d2` (so the Pattern-5 blocker is resolved). The `allow_truncation` drift is also resolved — it is now an explicit wired arg (see iter 8 below).

### Review Log — iter 8 (NEW post-commit topology work, 2026-06-23)

Resumed after the Phases 1-4 commit (`953c7a5d2`). Reviewed three newer topology changes with 3 parallel adversarial subagents; adjudicated + fixed myself; all changes re-verified under the miniforge interpreter.

- **🔴 CRITICAL found + FIXED — commit `268506155` (`--surface-allow-truncation` / `--surface-min-surfaces` wiring) was a SILENT NO-OP.** The policy was published as a module global in `__main__` ~400 lines AFTER the setup-time family build had already read it via `globals().get(...)`, so top-to-bottom execution always read the unset global → `False` → the flag never reached `build_boozer_surface_family`. The default-OFF path was byte-identical, masking it. **Root fix (not a band-aid):** thread `allow_truncation`/`min_surfaces` as EXPLICIT args through `initialize_surface_data_for_contract` → `initialize_published_surface_data_from_stage2_seed` → `build_boozer_surface_family`, passed straight from `args` at the single call site, and delete the `globals()` indirection + the dead late assignments. This eliminates the temporal-coupling bug *class* (no global to misorder) and makes the wiring unit-testable. These functions already take all other config as explicit args, so the `globals()` read was the lone anomaly. `tests/geo/test_surface_truncation_wiring.py` rewritten to drive the policy as args (no module globals) + a `_require_published_G_consistency` spy + a new `initialize_surface_data_for_contract` forwarding test (the propagation hop the bug broke). 4 wiring + 11 family = 15 pass.
- **🟡 MINOR-1 FIXED — truncated bands skipped the G-consistency QA.** A truncated band correctly skips the strict inner0-based name postcondition, but it also dropped `_require_published_G_consistency` (which is name-agnostic — every shell shares the outer G). Now: `if provenance.truncated: _require_published_G_consistency(...) else: <full strict postcondition>`.
- **🟡 MINOR-2 FIXED — env-var truthiness.** `default=bool(os.environ.get("SURFACE_ALLOW_TRUNCATION"))` treated `"0"`/`"false"` as True. Now parses truthy strings (`in ("1","true","yes","on")`).
- **✅ `band_shear_load` / `BandShearLoad`** (`topology/iota_profile.py` +108, `test_iota_profile.py` +92): PASS. The branch-aware nearest-resonance formula `min(|ι−(round(ι−f)+f)|)` was proven exact (0.0 error over dense grids + every half-integer banker's-rounding tie, q≤7/13/24); shear adjacency, per-bracket count, and edge cases hold; tests pin exact numerics. Still a dead scaffold (no production caller) — the pre-existing wire-or-drop advisory, owner=user.
- **✅ `converse_kam.py` +5** (`np.ascontiguousarray` for a column-strided RK4-state slice fed to `set_points`): PASS, monotonically safe; `$MF` repro confirmed the crash without it / success with it. No unfixed live twins (every other `set_points` site in `topology/` is fed fresh-contiguous arrays; the one structural lookalike, `mather_dw.py:279`, is dead/unwired — advisory).
- **🩹 PRE-EXISTING defect found + FIXED — 3 stale-stub tests in `tests/geo/test_single_stage_example.py`** (`published_stage2_seed_initializes_outer_to_inner`, `..._rejects_solved_G_drift`, `..._rejects_nonmonotone_actual_volumes`). They patched the driver's `initialize_boozer_surface`, but Phase-2's `build_boozer_surface_family` solves via `attempt_initialize_boozer_surface` (resolved in the `boozer_surface_family` module namespace) → the stub was bypassed → the real solver ran on a `FakeSurface` and died on `quadpoints_theta`. **Broken since `953c7a5d2` (git-proven; the test file predates it), missed by iters 2-3** (the "covered at the driver level" claim was wrong). Fix (test-only): redirect the solve patch to `family.attempt_initialize_boozer_surface` and wrap the fakes' returns in the `BoozerInitializationResult` result shape (`result.success`/`result.boozer_surface`). Proven NON-VACUOUS: removing the drift / making volumes monotone flips each rejection test pass→"RuntimeError not raised", so they reject via the real driver guards, not the crash.
- **Verification (independently re-run):** 28 pass across `test_surface_truncation_wiring` + `test_boozer_surface_family` + the `test_single_stage_example` published-init group; `ruff check` clean on both edited test files; `git status` shows ONLY 3 files touched this iter (the driver + 2 test files) — no production solve/handoff/family code modified. The driver's 3 `F401`s are pre-existing/unrelated (in the rotation-aware import block, proven identical on HEAD) — left for the geometry-track author.
- **Still open (owner = user/author):** `band_shear_load` wire-or-drop; the `mather_dw.py:279` dead-code contiguity hardening; the Chirikov 2/3-threshold + non-finite-G untested guards; the real-map cross-consistency test (scoped out). All low-value / not loop-fixable.
