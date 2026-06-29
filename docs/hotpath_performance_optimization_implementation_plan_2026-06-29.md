# Hot-Path Performance Optimization Implementation Plan (2026-06-29)

## Purpose

Convert the 14-subsystem adversarially-verified hot-path performance audit
(workflow `wf_84dfb068-dd5`, branch `surrogate-confinement-v2`) into an
executable, trackable work plan. The audit produced **44 confirmed** on-hot-path
findings (2 high / 3 medium / 8 low / 31 micro), 11 rejected false positives, and
2 uncertain. This file orders them by *default-hot-path impact × safety* so the
zero-risk wins land first and the behavior-affecting / opt-in-lane changes are
gated behind explicit verification.

## Goals

- Eliminate the documented per-iteration `CurveCurveDistance` brute-force stall
  (the 30-min A100 startup park) with no change to optimizer numerics.
- Remove the systemic "N separate `grad(J, argnums=k)` calls" anti-pattern
  (2–6× redundant forward+backward) at ~10 sites, numerically identically.
- Cut per-eval framework overhead (`Optimizable.x` setter, FD copy, solve-log
  flush) that is paid on *every* `J(x)` for every config.
- Land the high-value opt-in-lane wins (DESC `use_jit`, residue trace/Hessian
  reuse) gated behind their feature flags, when those lanes are exercised.
- Prove "no numerical change" for every change marked numerically-identical via a
  before/after `J`/`dJ` equivalence gate.

## Non-Goals

- No profiling-driven re-architecture of the C++ `simsoptpp` kernels.
- No change to the Boozer adjoint linear-solver backend (operator-GMRES vs dense
  LU) — out of scope here; tracked elsewhere.
- No micro-optimization of cold paths (plotting, JSON/checkpoint I/O, CLI
  parsing, one-time seed/equilibrium construction) — explicitly rejected by the
  audit.
- Not chasing the 11 rejected findings (trace-time-unrolled `for` loops inside
  `@jit`, `.item()`/`float()` inside a jit trace, O(n²) over `nfp`/`ncoils≈4`).

## Current Context

- Test harness: `./.conda-env/bin/python` (Python 3.11.15) — the interpreter the
  repo tests pass under; every validation command below uses it. Do NOT validate
  with ambient `python` from this shell (`/Users/suhjungdae/.local/bin/python`,
  Python 3.14.3): `PYTHONPATH=src python -m pytest …` cannot even load
  `tests/conftest.py`, failing with `ImportError: cannot import name 'Curve' from
  'simsoptpp'`. The failure resolves `simsoptpp` as a namespace package at
  `src/simsoptpp/` (`__file__ is None`, no `__init__.py`, no `.so`). Verified
  2026-06-29: that pytest command fails under ambient `python` and passes
  (15 passed, 39 subtests) under `.conda-env`. Separate caveat: the Homebrew
  miniforge base interpreter (`/opt/homebrew/Caskroom/miniforge/base/bin/python`,
  Python 3.13.12; also current `python3`) can import the compiled site-packages
  `simsoptpp` extension and passed the focused curve-objectives test in this
  checkout, but it is not the canonical validation interpreter for this plan.
  Relevant tests: `tests/field/test_selffieldforces.py`,
  `tests/geo/test_boozersurface.py`, `tests/field/test_biotsavart.py`,
  `tests/geo/test_curve_objectives.py`.
- Original checkout baseline for this review: branch `surrogate-confinement-v2`,
  HEAD `c15e39414`, dirty tree present. Latest committed hot-path baseline before
  the current micro slice is `f7f5b3007` (`02778c0da` + `095348cf4` landed Phases
  1-3 and the first Phase 4.3 host-sync item; `f7f5b3007` landed the Phase 4.1
  BiotSavart safe pieces and the Phase 4.2 `objectives/utilities.py` PLU
  allocation cleanup). Current repo `HEAD` may include unrelated non-hotpath
  commits (for example the Sobolev diagnostics slice `45d2b3ae1`). Phase 5 DESC
  anchors remain dirty-tree scoped: `examples/single_stage_optimization/banana_opt/desc_bridge/objective_factory.py`
  and `examples/single_stage_optimization/DESC_JOINT/run_desc_joint_banana.py`
  are untracked in the current tree, while `desc_bridge/runtime_coilset.py` is
  added. Treat Phase 5 as dirty-tree-scoped until those files are committed or
  the plan is explicitly scoped to this worktree.
- Audit provenance: the workflow ID `wf_84dfb068-dd5` is a Claude Code runtime
  artifact (session transcript), not committed to the repo, so it does not appear in
  repo `rg`. The cited project memory `project_perf_audit_hotpath_2026_06_29` DOES
  exist — in the file-based agent memory store at
  `~/.claude/projects/-Users-suhjungdae-code-columbia-simsopt/memory/project_perf_audit_hotpath_2026_06_29.md`
  (a different store than OpenMemory MCP, where it was never written; querying
  OpenMemory and finding nothing is expected, not missing provenance). Before
  treating the 44-finding count as SSOT, cite that concrete memory path (or export
  the workflow result) rather than the runtime ID.
- `simsopt.geo.jit` forces the CPU platform for the geo JAX kernels (per
  `jit.py:2`), so each separate grad jit is also a separate XLA dispatch.
- Confirmed code shapes (read 2026-06-29):
  - `CurveCurveDistance.shortest_distance` (`geo/curveobjectives.py:240-248`) and
    `CurveSurfaceDistance.shortest_distance` (`:372-378`) fall to an all-pairs
    full-resolution `scipy.cdist` double loop when `candidates` is empty.
  - `shortest_distance_among_candidates` already caps its result at
    `minimum_distance` via `min([self.minimum_distance] + [...])`
    (`:235`, `:370`), so reporting `minimum_distance` in the empty case is
    *consistent* with existing semantics (empty ⇒ true min ≥ `minimum_distance`).
  - `MeanSquaredForce` (`field/force.py:204-271`) and `LpCurveForce`
    (`:92-159`) each build 5 separate `grad(self.J_jax, argnums=k)` jits and call
    all 5 in `dJ()` (verified: `MeanSquaredForce.dJ` body runs `:250-269`).
  - `boozersurface.py:718-719` calls the slow non-vectorized
    `boozer_penalty_constraints(..., derivatives=0)` after the LS Newton loop to
    populate `res['residual']`, re-triggering a full `BiotSavart` set_points +
    compute at already-converged points.
  - `Optimizable.x` setter (`_core/optimizable.py:1060-1065`) materializes
    `list(self.dof_indices.values())` per call; `x`/`full_x` getters
    (`:1057-1058`, `:1073-1074`) fancy-index-copy each block then concatenate.
  - `FiniteDifference.jac` (`_core/finite_difference.py:88-111`) allocates
    `np.copy(x0)` inside the per-DOF loop.
  - `serial.py:124-129` writes + `flush()`es the objective log every evaluation.

## Rationale

The audit's wall-time numbers were originally reasoned (per-call cost × frequency)
from code reading; the safe phases are now **measured** on CPU (geo kernels are
CPU-forced per `jit.py:2`, so CPU A/B is representative) — see *Measured results*
below. The plan front-loads
changes that are **numerically identical** (provable by a bit-for-bit `J`/`dJ`
equivalence harness) so they can ship on static confidence alone, and isolates
the two behavior-affecting changes (boozer diagnostic-residual removal, exact
Newton iterative-refinement gating) behind consumer audits + convergence checks.
The systemic multi-primal-grad fix is grouped into one phase because every site
uses the identical idiom and the fix template is the same — a single reviewed
pattern applied N times.

## Measured results (2026-06-29, CPU A/B)

Faithful before/after on this checkout (`.conda-env` py3.11). Phase-1/2 use the
committed methods vs reconstructed old paths (CCD brute force; 5-separate vs
1-tuple grad on the same `J_jax`); Phase-3 framework items use a two-worktree A/B
(`c15e39414` before vs HEAD).

| Win | Setup | Result |
|---|---|---|
| **Phase 1** `CurveCurveDistance` empty-candidate (the stall) | 18 curves×100 pts / 12×400 pts, well-separated | **6,600× / 68,000×** per call (2.2 ms / 11.4 ms → ~0); O(n²·m²) growth; `≥ threshold` gate unchanged |
| **Phase 2** force multi-primal grad | same `J_jax`, 80 quad pts | **3.68×** (1.28 → 0.35 ms); numerically equivalent (max\|Δ\| 6e-5, fp-reassoc) |
| **Phase 3** `Optimizable.x` index cache | worktree A/B, 7 opts / 306 dofs | ~2% (23.2 → 22.7 µs) — setter dominated by recompute propagation (left untouched) |
| **Phase 3** `FiniteDifference.jac` | worktree A/B, 1000 dofs, trivial fn | ~2.4% (3.98 → 3.89 ms); `np.copy` hoist alone saves 0.1–0.7 ms/jac (2–3×) but fn/setter dominate |
| **Phase 3** serial `flush()` removal | inline per-eval log write | 1.8–22 µs/eval (scales with dof count) |

Conclusion: the `impact × safety` ordering is validated end-to-end. Phase 1 is 4–5
orders of magnitude per call (the documented 30-min stall); Phase 2 is a ~4×
default-path win (coil-force is on by default); Phase 3 items are correctly "micro"
(~2–3%). The dominant per-eval framework cost — Optimizable DAG recompute
propagation — was deliberately left untouched (its memo regressed force/strain
Taylor checks and was reverted).

## Assumptions

- **Confirmed (consumer audit 2026-06-29):** `res['residual']` for the
  `type == "ls"` Boozer solve is diagnostic-only. The LS trust-gate norm helper
  `compute_boozer_constrained_residual_norm`
  (`stage2_single_stage_handoff.py:932-952`, `kind == "ls"` branch) reads
  `res['jacobian']`/`res['gradient']`, NOT the residual (its docstring states the
  LS residual "is *not* what the solver checks"). The only `res['residual']`
  consumers are the EXACT-Newton path (`handoff.py:956`, `res.get("residual")`),
  verbose prints (`boozersurface.py:1172`, `boozer_finite_current.py:858`), and a
  failure-string formatter (`desc_joint_validation_launcher.py:677`). **Therefore
  the Phase 1 Task 2 fix must stay scoped to the LS solver
  (`minimize_boozer_penalty_constraints_newton`) and must NOT touch the exact
  solver, where the residual is a real gate input.**
- `shortest_distance()` does not feed `J`/`dJ` or the optimizer in the current
  grep, but it is not merely a print diagnostic: several callers persist/read it
  as a clearance metric (`STAGE_2/banana_coil_solver.py`,
  `VMEC_SINGLE_STAGE/vmec_single_stage_banana.py`,
  `banana_opt/single_stage_geometry.py`, `banana_opt/stage2_objectives.py`,
  and downstream summary/report scripts). Capping the empty-candidate value at
  `minimum_distance` is optimizer-safe, but it intentionally converts those
  metric sites into lower-bound/pass-fail values. Exact-achieved-clearance
  consumers need a separate exact helper or a renamed/labeled metric.
- Existing `tests/geo/test_curve_objectives.py` encodes the old empty-candidate
  `CurveSurfaceDistance.shortest_distance()` behavior (`:402-411` expects exact
  distance `>` the capped candidate distance when candidates are empty). Phase 1
  must update that test and add a `CurveCurveDistance` empty-candidate assertion.
- Official JAX docs (`/jax-ml/jax`, checked 2026-06-29) confirm that
  `grad(..., argnums=(0, 1, ...))` and `value_and_grad(..., argnums=(...))`
  return a tuple of per-primal gradients from one transformed call.
- `jax.vjp`/`grad` over multiple primals returns the full cotangent tuple from one
  forward+backward — standard JAX semantics; the refactor is exact.
- Reverse-mode `grad(J, argnums=(0,...,k))` yields identical cotangents to N
  separate `grad(J, argnums=i)` calls.

## Implementation Plan

### Phase 0 — Baseline & equivalence harness (gate for "no numerical change")

1. Establish before/after equivalence tooling.
   - [ ] Write `scratchpad`/throwaway script that, for a fixed small config, calls
         `J()` and `dJ()` on each object to be refactored (`MeanSquaredForce`,
         `LpCurveForce`, `CurveCurveDistance`, `CurveSurfaceDistance`, the framed
         curve penalties) and pickles the outputs as the pre-change baseline.
   - [ ] Record current outputs on `surrogate-confinement-v2` HEAD before any edit.
   - [ ] Confirm test baseline is green:
         `./.conda-env/bin/python -m pytest tests/field/test_selffieldforces.py tests/geo/test_boozersurface.py tests/field/test_biotsavart.py -q`
   - [ ] Use the existing curve-objective test file:
         `rg --files tests/geo | rg 'test_curve_objectives\.py$'`.
         Do not use `ls tests/geo | grep -i curveobj`; it misses
         `tests/geo/test_curve_objectives.py`.
   - [ ] Record current `tests/geo/test_curve_objectives.py` behavior before the
         Phase 1 semantic change, especially `CurveSurfaceDistance` lines
         `:402-411`, which currently assert exact empty-candidate distance is
         greater than the capped candidate distance.

### Phase 1 — Safe default-hot-path wins (highest value, lowest risk)

1. `CurveCurveDistance` / `CurveSurfaceDistance` empty-candidate fallback
   (P3 medium; the documented stall). Numerically identical for `J`/`dJ`
   (untouched); changes only the diagnostic `shortest_distance()` value, and only
   in the empty-candidate case, to the already-used capped semantics.
   - [x] Caller audit done (2026-06-29): all ~28 `shortest_distance()` callers are
         reporting/certification only (diagnostic `outstr` prints, or `float(...)`
         recorded as a metric); NONE feed `J`/`dJ` or the optimizer. Caveat: several
         record it as an *achieved-clearance metric* (`banana_coil_solver.py:3750,
         5822`; `VMEC_SINGLE_STAGE/vmec_single_stage_banana.py:1017,1019`;
         `single_stage_geometry.py:1086-1087`; hardware-validity certification), so
         capping UNDER-reports true clearance when coils are well-separated — fine
         for a `≥ threshold` gate, a fidelity loss if exact clearance is recorded
         (handled by the conditional below).
   - [ ] `geo/curveobjectives.py:240-248` — in `CurveCurveDistance.shortest_distance`,
         when `len(self.candidates) == 0` `return self.minimum_distance`
         (true min is provably ≥ threshold; matches `min([minimum_distance]+…)` cap).
   - [ ] `geo/curveobjectives.py:372-378` — same fix in
         `CurveSurfaceDistance.shortest_distance`.
   - [ ] For the achieved-clearance-metric callers above, decide per call site:
         (a) confirm the metric is only consumed as a `≥ threshold` pass/fail (cap is
         fine), or (b) if exact clearance must be recorded, give those sites an exact
         path — honor `self.downsample` and precompute the downsampled gamma list
         once outside the comprehension (a one-time `scipy.cKDTree` per curve +
         nearest-neighbor query) — do NOT keep full-resolution `cdist`.
   - [ ] Update `tests/geo/test_curve_objectives.py`: revise the existing
         `CurveSurfaceDistance` empty-candidate assertion (`:402-411`) and add a
         `CurveCurveDistance` empty-candidate assertion so both classes lock the
         capped-return contract.
   - [ ] Extend the metric audit beyond the original three sites:
         `banana_opt/stage2_objectives.py:2254,2827`,
         `src/simsopt/util/permanent_magnet_helper_functions.py:161`, advanced /
         intermediate example prints, and summary readers that consume the
         persisted metrics. If exact values are kept, record them under a distinct
         key so capped lower-bound values are not mislabeled as exact clearance.

2. Boozer LS-Newton tail: stop recomputing the residual via the slow path
   (P3 low; **behavior-touching — LS solver ONLY**). Consumer audit already done
   (see Assumptions): the LS gate routes around `res['residual']`; the EXACT solver
   genuinely consumes it. Scope this change to
   `minimize_boozer_penalty_constraints_newton` only; leave
   `solve_residual_equation_exactly_newton` untouched.
   - [ ] In `boozersurface.py:718-719` (LS solver), either (a) build `r` from the
         already-cached `biotsavart.B()` (still set at the converged points from the
         loop's final `derivatives=2` eval at `:714`) + `surface.gammadash1/2` in
         NumPy, or (b) gate the `boozer_penalty_constraints(..., derivatives=0)` call
         behind `options.get('verbose')`. Do not call the non-vectorized path
         unconditionally.
   - [ ] Regression guard after editing — re-run the consumer grep and confirm no
         new LS-path `res['residual']` consumer appeared:
         `rg -n "res\['residual'\]|res\.get\(\"residual\"\)|get\(\"residual\"\)|\['residual'\]" src examples tests -g '*.py'`.

### Phase 2 — Systemic multi-primal-grad sweep (numerically identical, ~10 sites)

Replace N single-argnum grad jits with ONE multi-primal grad/`value_and_grad`.
Fix template:
`self.dJ_all = jit(lambda *a: grad(self.J_jax, argnums=(0,1,2,3,4))(*a))`, then
unpack the tuple in `dJ()`.

1. `field/force.py` (P2 medium + P3 low).
   - [ ] `MeanSquaredForce` (`:204-271`) — one `argnums=(0,1,2,3,4)` grad; unpack
         `dJ_dgamma…dJ_dB` from a single call in `dJ()` (`:261-269`).
   - [ ] `LpCurveForce` (`:92-159`) — identical refactor in `dJ()` (`:137-159`).
   - [ ] Optional deeper win: switch `J_jax` to
         `value_and_grad(J_pure, argnums=(0,1,2,3,4))` and cache `(value, grads)`
         keyed on `set_points` so `J()` and `dJ()` share the single forward
         (currently `J()` does a 6th forward).
2. `geo/framedcurve.py` (P3 low ×2 + P4 micro).
   - [ ] Strain penalty binormal/torsion (`:90-118, 181-243, 319-343, 357-457,
         545-569, 630-674`) — replace `binormgrad_vjp0,1,2,4,5` (and Frenet 6×)
         with one multi-primal `vjp(self.binorm, *primals)[1](v)`; unpack
         `grad0…grad5`. Capture the primal output to also serve `J()`'s separate
         `frame_binormal_curvature()` forward.
   - [ ] Finite-build filament frame (`:245-300, 459-506, 676-723, 832-862,
         916-946, 976-1014`) — single multi-primal
         `vjp(rotated_*_frame, …)[1]((v0,v1,v2))` and
         `vjp(rotated_*_frame_dash, …)[1]((v0,v1,v2))`; remove per-arg closures.
3. `geo/curveobjectives.py` (P4 low).
   - [ ] `CurveCurveDistance` (`:211-214`) — one `argnums=(0,1,2,3)` grad; unpack in
         `dJ()` (`:299-302`).
   - [ ] `CurveSurfaceDistance` (`:351-352`) — one `argnums=(0,1)` grad.
4. `examples/.../banana_opt` constraint terms (P4 low + P4/P5 micro).
   - [ ] `self_intersect.py` (`:335-512`) — collapse the two separate grads over the
         O(N²) distance-matrix graph into one multi-primal grad.
   - [ ] `fold_buildability.py` — `RotationAwareCurvatureExcessPenalty.dJ`
         (`:263-339`, 5 grads → 1) and TWO two-grad terms:
         `CurveSurfaceGeodesicCurvature` (grad defs `:57-68`, dJ `:92-99`) and
         `NormalizedCurveCurvatureHinge` (grad defs `:142-151`, dJ `:156-164`).
   - [ ] `hardware_keepout.py` — `CurveHardwareKeepout.dJ` second grad pass when
         `R0` free (`:1043-1073`) and the SDF family double trilinear-interp
         (`:1155-1683`, `argnums=0` then `argnums=3`): compute both cotangents in
         one pass.
   - [ ] `poloidal_extent.py:194-197` and `ellipse_width.py:165-167` — live grep
         finds the same two-primal grad pattern in banana constraints. Triage
         whether those constraints are active in the default production lane; if
         yes, include them in this Phase 2 sweep, otherwise explicitly defer them
         as non-default hot path.

### Phase 3 — Per-eval framework micro overhead (universal; every `J(x)`)

1. `_core/optimizable.py`.
   - [ ] `x` setter (`:1060-1065`) — cache the total free-dof count instead of
         `list(self.dof_indices.values())[-1][-1]` per call; same for the
         `dof_indices.values()` materialization at `:1062, 1213, 1232, 1294, 1313`.
   - [ ] `x`/`full_x` getters (`:1057-1058, 1073-1074`) — avoid per-block
         fancy-index copy + concatenate churn where a preallocated buffer / cached
         layout suffices.
   - [ ] DAG walk — `set_recompute_flag` (`:1125-1133`) re-walks `self._children`
         with NO visited memo, so shared descendants are revisited once per
         propagation path; add a memo. The cached graph builders
         `update_free_dof_size_indices` (`:938-967`) and
         `_update_full_dof_size_indices` (`:969-1002`) — which carry explicit
         `# TODO: This is slow … walks the graph repeatedly` comments — are NOT
         per-eval (rebuilt only on fix/unfix or parent add/remove), but their
         recursive walk is worth memoizing too. (Corrected: earlier refs
         `:154-165, 309-324` were `DOFs._flag_recompute_opt` / `DOFs.free_x.setter`,
         unrelated to this.)
2. `solve/serial.py:124-129`, residual callback `:139`, and sibling
   `serial_solve` callback `:241` — buffer the objective/residual-log writes and
   drop the per-evaluation `flush()` (flush on close / periodically), removing a
   syscall per eval. Mirror or explicitly defer the analogous `solve/mpi.py`
   flushes (`:187, :197, :390, :443`) so the scope is not ambiguous.
3. `_core/finite_difference.py` — hoist `np.copy(x0)` out of the per-DOF loop in
   BOTH branches (`x = np.copy(x0)` at `:89` centered, `:106` forward); reuse one
   scratch vector and restore the perturbed entry each iteration.
4. `banana_opt/single_stage_geometry.py:1177, 1192-1203` — cache the
   `inspect.signature()` reflection result (per surface class) instead of
   recomputing it on every line-search Boozer re-solve guard.

### Phase 4 — Remaining micro-wins (opportunistic; grouped by file)

1. `field/biotsavart.py` (P5 micro).
   - [ ] Skip the current-derivative branch when all coil currents are fixed
         (`:82-90, 117-119`).
   - [ ] Vectorize the per-coil current-derivative reduction (`:83-85, 174-176`).
   - [ ] Assemble the derivative tree once instead of per-coil build + N−1 dict
         merges (`:87-90, 178-181`).
2. `geo` array-op / allocation micro (P5).
   - [x] `curvexyzfourier.py:240-244` — avoid `np.concatenate` dof rebuild per
         gamma/derivative call by switching the shared `JaxCurve` internal JAX
         kernels to `local_full_x`. Public `get_dofs()` copy/alias behavior stays
         unchanged, and the same internal fast path covers `OrientedCurveXYZFourier`.
   - [x] `orientedcurve.py:56` — `rotate_pure` returned
         `v @ Myaw @ Mpitch @ Mroll` (left-assoc ⇒ three `(N,3)@(3,3)` products);
         the current continuation precombines one 3×3 rotation and does a single
         point-cloud matmul, with a focused yaw/pitch/roll order regression test.
   - [x] `surface.py:679-689` — replace batched `np.linalg.det/inv` on the
         structured 2×2 (J[1,0]=0, J[1,1]=1) with the analytic inverse.
   - [x] `objectives/utilities.py:24-48` — replace dense permutation matmul
         (`P@…`, 3 dense n×n factors) with O(n) pivot indexing in
         `forward_solve`/`forward_backward`.
   - [ ] `framedcurve.py:746-759` — stop re-transferring quadpoints to device /
         recomputing `FrameRotation.alpha/alphadash` multiple times per eval.
   - [ ] `curveobjectives.py` — `MeanSquaredCurvature.J` (`:538`) and
         `ArclengthVariation.J` (`:495`) return `float(...)`, a host sync. CAVEAT:
         only a win if the value stays inside a JAX graph downstream (fused into a
         larger jitted objective); at the scipy boundary a python float is required,
         so this is a no-op there. Verify the consumer first. Low value.
3. `banana_opt` micro (P5).
   - [x] `stage2_objectives.py:2993-3000` — dedup the double
         `Jc.curve.kappa()`/`np.max(kappa)` on the default smooth-curvature path
         by reusing the hard-path `kappa` array for the smooth signed constraint.
         Custom injected curvature helpers keep the existing four-argument
         contract.
   - [ ] `stage2_objectives.py:3226-3268` — cache ALM constraint metadata /
         activity tolerances (depend only on smoothing params + static
         config/thresholds, e.g. fixed `Jc.threshold` / `Jccdist.minimum_distance`)
         instead of rebuilding every inner eval.
   - [ ] `boozer_finite_current.py:178, 189` — do NOT drop the array `.copy()`
         unless the caller alias proof is written down and covered by a regression
         test. These helpers currently return the original object only for
         `I_value == 0.0`; for nonzero transforms, the copy preserves input
         immutability. If the tensors are freshly computed and consume-once, prove
         that at each caller and add a no-alias test before using in-place updates.
   - [ ] `single_stage_objectives.py:1326-1403` — cache the simsopt `Optimizable`
         sum/scale graph instead of reconstructing it on every trial objective eval.
   - [x] `edge_iota_proxy.py:391-392` (the 401-line proxy module — NOT
         `edge_delivered_iota.py`; both files exist) — remove the redundant
         `BiotSavart.set_points` (points already set at `:142` in
         `_banana_cyl_B_on_contours`). Regression covers that `B()` and `B_vjp()`
         still see the contour points while the value/gradient path calls
         `set_points` once.
   - [ ] `edge_iota_proxy.py:186-200` — replace unbuffered `np.add.at` scatter
         (`:190`) with a buffered accumulation.
   - [ ] `hardware_keepout.py:1034-1041` — `J()` does `res += float(self.J_jax(...))`
         inside the per-candidate loop, forcing one host sync PER candidate;
         accumulate in a JAX scalar and call `float()` ONCE after the loop (N → 1).
         (NOTE: the `:992-1008` `compute_candidates()` AABB rescan is NOT redundant —
         within-eval cached, re-run across evals only because `recompute_bell` nulls
         it on DOF change, i.e. geometry actually moved; only a cheaper/incremental
         scan would help. Low value.)
   - [ ] `boozer_topology_bridge.py:673-726` — the `S_HEL` gradient does a full
         `fft2 + ifft2` (`:709`, `:723`) on top of the value-path `rfft2`. NOTE:
         **documented-intentional** trade-off (docstring `:697-700`: "one extra FFT
         pair, paid once per gradient call" for a cleaner adjoint) — optional / low
         priority; revisit only if it shows up in a profile.
   - [ ] `desc_bridge/runtime_coilset.py:613-639` — avoid rebuilding the full
         per-coil params list (`:633`) in `_merge_params` on the scoped-params path
         (the normal optimized-coil case; a fast-path early-return at `:616-627`
         already skips the rebuild for full-length params).
4. Behavior-affecting micro (isolate; needs convergence check).
   - [ ] `boozersurface.py:1133-1134` — gate the exact-Newton iterative refinement
         on `norm < 1e-9` (matching sibling solvers) instead of applying it every
         step. **safe=False**: changes Newton iterate path → must pass the Boozer
         convergence/parity tests and a single-stage smoke before adopting.

### Phase 5 — Opt-in lane wins (gated behind feature flags; high value when on)

1. DESC joint lane (P2 high). Flag `--desc-objective-use-jit`
   (`BooleanOptionalAction`, already CLI-overridable); today's default is
   `DEFAULT_DESC_OBJECTIVE_USE_JIT = False` (`objective_factory.py:81`).
   Current checkout caveat: the two main DESC anchors named below are untracked in
   this dirty tree (`objective_factory.py`,
   `DESC_JOINT/run_desc_joint_banana.py`), so this phase is not actionable from a
   clean checkout until that implementation is landed or this plan is explicitly
   bound to the dirty tree.
   - [ ] `desc_bridge/objective_factory.py:81, 423, 561-565` — default
         `use_jit=True` for the optimizer (multi-eval) lane; keep `use_jit=False`
         only for the single-shot smoke eval. Prefer auto-select on
         `optimizer.optimize` + memory budget over a blanket flip.
   - [ ] (Uncertain finding) `objective_factory.py:82, 424, 564` — evaluate
         defaulting `deriv_mode='batched'` (vmapped single Jacobian pass) with a
         `'blocked'` memory-pressure fallback.
2. Residue topology lane (P2 high + P3 medium; `--residue-objective-weight > 0`).
   - [ ] `topology/residue_sensitivity.py:856-922, 2040-2059` — trace the converged
         orbit ONCE with `with_tape=True` (reuse the Newton solver's final state)
         and feed the shared records to both the state-gradient and the field-VJP
         passes (eliminates 1–2 full retraces/branch/grad-eval).
   - [ ] `topology/residue_sensitivity.py:1812-1823, 1908-1996` — gather all
         `4×steps` recorded stage points into one `(N,3)` array, call
         `field.set_points(all_points)` + `field.d2B_by_dXdX()` ONCE (returns
         `(N,3,3,3)`), index per stage, and share that batched Hessian across both
         passes (collapses ~3072 single-point pybind calls into one batched C++
         call).
3. ALM solve-orchestration (uncertain; `STAGE_2/banana_coil_solver.py`).
   - [ ] `:3749-3760` — gate the exact brute-force distance/curvature certification
         so it runs only on the final incumbent (and on accepted iterates that
         already beat the best `field_objective`), not unconditionally every
         accepted iterate. (Largely subsumed once Phase 1 Task 1 lands the
         candidate-aware `shortest_distance`.)

### Implementation Status — 2026-06-29

- Phase 1: landed in `02778c0da` with focused `curve_objectives` and Boozer tests.
- Phase 2: landed in `02778c0da` for `force.py`; follow-up landed in this slice for
  `framedcurve.py`, `curveobjectives.py`, `self_intersect.py`,
  `fold_buildability.py`, `hardware_keepout.py`, `poloidal_extent.py`, and
  `ellipse_width.py`. Remaining single-arg gradients in fixed-R0 or single-primal
  paths are intentionally not tuple-gradient families.
- Phase 3: landed in `02778c0da` for finite difference copies and solver flushes;
  follow-up landed in `095348cf4` for `Optimizable` index iteration and
  `single_stage_geometry.py` `run_code` signature caching. The recursive DAG-walk
  memoization item is deferred: a visited-set implementation changed shared-graph
  recompute/update traversal semantics and regressed force/strain Taylor tests, so
  it needs a separate design and regression proof rather than a hot-path sweep edit.
- Phase 4.1 (`field/biotsavart.py`): partial continuation slice. The safe pieces
  are implemented: vectorized current-cotangent reductions and one-pass
  `sum_derivatives` assembly for `B_vjp`, `B_and_dB_vjp`, `A_vjp`, and
  `A_and_dA_vjp`. The all-fixed-current branch skip is explicitly deferred: it
  preserves optimizer-visible free gradients, but it would remove nonzero fixed
  current entries from the public `Derivative(..., as_derivative=True)` view unless
  a contract-preserving lazy/current-partial strategy is designed.
- Phase 4.2 (`geo` array/allocation micro): partial continuation slice. The dense
  permutation-matrix products in `objectives/utilities.py` `forward_solve` /
  `forward_backward` are replaced by O(n) pivot indexing and factored residual
  matvecs. The current continuation also precombines
  `orientedcurve.rotate_pure`'s yaw/pitch/roll matrices before multiplying the
  point cloud, and replaces `surface.mean_cross_sectional_area`'s batched
  structured-2x2 `det`/`inv` allocation with analytic entries. The current
  continuation also routes `JaxCurve`'s internal gamma/derivative kernels through
  `local_full_x`, avoiding per-call public `get_dofs()` concatenation for
  `JaxCurveXYZFourier` and `OrientedCurveXYZFourier` without changing public DOF
  behavior. The framed-curve
  VJP consolidation was already covered by Phase 2; the remaining items still
  need isolated consumer audits or timing proof before changing allocation/host-sync
  behavior.
- Phase 4.3 (`banana_opt` micro): `hardware_keepout.py` per-candidate host sync
  landed with hardware keepout tests. The stage2 curvature path now evaluates
  `Jc.curve.kappa()` once when the default smooth-curvature helper is injected,
  while preserving the custom-helper four-argument contract. The edge-iota proxy
  gradient path now reuses the contour points already installed for `B()`, avoiding
  the second identical `BiotSavart.set_points` before `B_vjp()`. The other items
  are deferred. In particular, `boozer_finite_current.py` copies stay until a
  no-alias proof and regression test exist, and DESC bridge/runtime items belong
  to the dirty-tree DESC lane.
- Phase 4.4 (`boozersurface.py` exact-Newton iterative refinement): deferred;
  behavior-affecting convergence-path change requires an isolated Boozer parity
  plus single-stage smoke gate.
- Phase 5.1 (DESC joint lane): deferred; the named DESC anchors are still dirty or
  untracked in this checkout, so this cannot be clean-checkout landed here.
- Phase 5.2 (residue topology lane): deferred; opt-in residue objective path needs
  a dedicated `with_tape` trace/Hessian batching design and residue smoke test.
- Phase 5.3 (ALM solve-orchestration certification gating): deferred; behavior
  affects certification cadence and is partly subsumed by the Phase 1
  candidate-aware distance change.

## Validation Plan

- [ ] **Equivalence gate (numerically-identical changes — all of Phase 2, Phase 1
      Task 1, Phase 4 except 4.4):** rerun the Phase 0 harness post-change and
      assert `J` matches bit-for-bit and `dJ` matches to ≤ 1e-12 relative
      (same VJP, reordered) for every refactored object.
- [ ] `./.conda-env/bin/python -m pytest tests/field/test_selffieldforces.py -q`
      (force.py Phase 2 + biotsavart Phase 4).
- [ ] `./.conda-env/bin/python -m pytest tests/geo/test_curve_objectives.py -q`
      (Phase 1 Task 1 `shortest_distance` semantic contract and candidate
      invalidation behavior).
- [ ] `./.conda-env/bin/python -m pytest tests/geo/test_boozersurface.py tests/geo/test_boozer_trust_gate.py -q`
      (Phase 1 Task 2 + Phase 4.4 boozer changes).
- [ ] `./.conda-env/bin/python -m pytest tests/field/test_biotsavart.py -q`
      (Phase 4 biotsavart).
- [ ] Full geo/field regression:
      `./.conda-env/bin/python -m pytest tests/geo tests/field -q` before sign-off.
- [ ] `./.conda-env/bin/python -m ruff check` on every touched file.
- [ ] **Caller-audit gate (Phase 1 Task 1 & 2):** the two `rg` audits return no
      consumer that depends on the removed/changed value.
- [ ] **DESC lane cleanliness gate (Phase 5):** `git ls-files` contains every DESC
      path named by the phase, or the implementation note states that Phase 5 is
      validated only against the current dirty tree.
- [ ] **Behavior gate (Phase 4.4 & Phase 5):** Boozer convergence parity test +
      one single-stage smoke eval show unchanged convergence (iteration count,
      final `J`) within tolerance; residue/DESC lane smoke matches pre-change
      objective when the flag is on.
- [ ] (Optional, if a GPU pod is available) timing A/B on one production
      single-stage eval to confirm the predicted ms savings on Phase 1/2/3.

### Validation Evidence — 2026-06-29 Continuation Slice

- `PYTHONNOUSERSITE=1 ./.conda-env/bin/python -m pytest tests/field/test_biotsavart.py -q`
  — 18 passed, 12 subtests passed.
- `PYTHONNOUSERSITE=1 ./.conda-env/bin/python -m pytest tests/field/test_selffieldforces.py -q`
  — 11 passed.
- `PYTHONNOUSERSITE=1 ./.conda-env/bin/python -m pytest tests/objectives/test_utilities.py -q`
  — 6 passed.
- `PYTHONNOUSERSITE=1 ./.conda-env/bin/python -m pytest tests/geo/test_boozersurface.py tests/geo/test_boozer_trust_gate.py -q`
  — 73 passed, 183 subtests passed.
- `PYTHONNOUSERSITE=1 ./.conda-env/bin/python -m pytest tests/geo/test_surface_objectives.py tests/geo/test_single_stage_surface_stack_spacing_gradient.py -q`
  — 26 passed, 159 subtests passed.
- `PYTHONNOUSERSITE=1 ./.conda-env/bin/python -m ruff check src/simsopt/field/biotsavart.py src/simsopt/objectives/utilities.py tests/field/test_biotsavart.py tests/objectives/test_utilities.py`
  — clean.
- `PYTHONNOUSERSITE=1 ./.conda-env/bin/python -m pytest tests/geo/test_curve.py::Testing::test_rotate_pure_preserves_yaw_pitch_roll_order -q`
  — 1 passed.
- `PYTHONNOUSERSITE=1 ./.conda-env/bin/python -m pytest tests/geo/test_curve.py::Testing::test_jaxcurve_internal_calls_use_local_full_x -q`
  — 1 passed, 2 subtests passed.
- `PYTHONNOUSERSITE=1 ./.conda-env/bin/python -m pytest tests/geo/test_curve.py -q`
  — 28 passed, 388 subtests passed.
- `PYTHONNOUSERSITE=1 ./.conda-env/bin/python -m pytest tests/field/test_coil.py -q`
  — 10 passed, 8 subtests passed.
- `PYTHONNOUSERSITE=1 ./.conda-env/bin/python -m ruff check src/simsopt/geo/curve.py src/simsopt/geo/orientedcurve.py tests/geo/test_curve.py`
  — clean.
- `PYTHONNOUSERSITE=1 ./.conda-env/bin/python -m py_compile src/simsopt/geo/curve.py src/simsopt/geo/orientedcurve.py tests/geo/test_curve.py`
  — clean.
- `PYTHONNOUSERSITE=1 ./.conda-env/bin/python -m pytest tests/geo/test_surface_taylor.py -q -k surface_coefficient_derivative`
  — 1 passed, 18 deselected, 12 subtests passed.
- `PYTHONNOUSERSITE=1 MPLCONFIGDIR=/tmp ./.conda-env/bin/python -m pytest tests/geo/test_surface_xyzfourier.py -q -k 'aspect_ratio_compare_with_cross_sectional_computation or mean_cross_sectional_area_raises_for_singular_phi_mapping'`
  — 2 passed, 8 deselected.
- `PYTHONNOUSERSITE=1 ./.conda-env/bin/python -m ruff check src/simsopt/geo/surface.py tests/geo/test_surface_xyzfourier.py`
  — clean.
- `PYTHONNOUSERSITE=1 ./.conda-env/bin/python -m py_compile src/simsopt/geo/surface.py tests/geo/test_surface_xyzfourier.py`
  — clean.
- Dense-formula equivalence probe for `Surface.mean_cross_sectional_area`
  (`get_exact_surface()` helper): current analytic value matched the previous
  dense `np.linalg.det/inv` computation with absolute delta `5.551e-17`.
- `PYTHONNOUSERSITE=1 ./.conda-env/bin/python -m pytest tests/geo/test_banana_objective_modules.py::Stage2ObjectiveModuleTests -q`
  — 62 passed.
- `PYTHONNOUSERSITE=1 ./.conda-env/bin/python -m ruff check examples/single_stage_optimization/banana_opt/stage2_objectives.py tests/geo/test_banana_objective_modules.py`
  — clean.
- `PYTHONNOUSERSITE=1 ./.conda-env/bin/python -m py_compile examples/single_stage_optimization/banana_opt/stage2_objectives.py tests/geo/test_banana_objective_modules.py`
  — clean.
- `PYTHONNOUSERSITE=1 ./.conda-env/bin/python -m pytest tests/geo/test_edge_iota_proxy.py -q`
  — 9 passed.
- `PYTHONNOUSERSITE=1 ./.conda-env/bin/python -m ruff check examples/single_stage_optimization/banana_opt/edge_iota_proxy.py tests/geo/test_edge_iota_proxy.py`
  — clean.
- `PYTHONNOUSERSITE=1 ./.conda-env/bin/python -m py_compile examples/single_stage_optimization/banana_opt/edge_iota_proxy.py tests/geo/test_edge_iota_proxy.py`
  — clean.

## Risks and Mitigations

- Risk: `shortest_distance()` empty-case change silently alters a real consumer
  (not just a diagnostic print).
  Mitigation: Phase 1 Task 1 `rg` gate plus explicit exact-vs-capped metric
  decision per caller. The new value equals the existing capped semantics, so any
  `>= minimum_distance` check is unaffected, but exact achieved-clearance metrics
  must either keep an exact helper or be labeled as lower-bound metrics.
- Risk: Removing the Boozer LS diagnostic residual breaks a downstream reader.
  Mitigation: consumer grep before editing; fall back to "reuse cached field"
  (compute the same `r` without the redundant `set_points + compute`) if any
  consumer exists.
- Risk: Multi-primal grad refactor changes results if a `J_jax` closes over
  mutable state or the argnums order is mismatched on unpack.
  Mitigation: equivalence gate (bit-identical `J`, ≤1e-12 `dJ`) per object; keep
  argnums order identical to the original separate-grad order.
- Risk: Gating exact-Newton iterative refinement (Phase 4.4) changes convergence.
  Mitigation: isolated commit; Boozer convergence/parity tests + single-stage
  smoke must match before adoption; revert if iteration count regresses.
- Risk: DESC `use_jit=True` default increases compile time / device memory on the
  single-shot smoke path.
  Mitigation: scope the flip to the multi-eval optimizer lane only; keep the
  smoke path eager.
- Risk: `Optimizable.x` setter/getter changes affect a hot framework contract used
  everywhere.
  Mitigation: rely on the full `tests/geo tests/field` regression; keep the public
  return types identical (only avoid redundant allocations).
- Risk: Copy removal in `boozer_finite_current.py` mutates arrays that a caller
  still owns.
  Mitigation: leave the copies in place unless each caller proves consume-once
  ownership and a no-alias regression test covers the in-place path.

## Completion Criteria

- [ ] Phase 1 landed: empty-candidate `shortest_distance` returns the capped
      lower bound; no full-res `cdist` on the well-separated hot path; the LS
      diagnostic-residual recompute removed or cached.
- [ ] Phase 2 landed: the six target files (`force.py`, `framedcurve.py`,
      `curveobjectives.py`, `self_intersect.py`, `fold_buildability.py`,
      `hardware_keepout.py`) contain no single-argnum `grad(...)` *family* in a `dJ`
      method that should be one tuple-argnums grad. (NB: a repo-wide
      `rg -n "argnums=[0-9]" src examples` returns many single-arg grads, most
      legitimate — scope the check to these files' `dJ` methods, not a global count.)
- [ ] Phase 3 landed: `Optimizable.x` setter/getter no longer materialize
      `dof_indices.values()` per call; FD loop has no per-iteration `np.copy`;
      solve log no longer flushes per eval.
- [ ] Equivalence gate green for all numerically-identical changes.
- [ ] `tests/geo` + `tests/field` regression green under `.conda-env` python;
      `ruff` clean on touched files.
- [ ] Phases 4–5 tracked: each item either landed-with-gate or explicitly deferred
      with a one-line reason.
- [ ] Project memory updated (`project_perf_audit_hotpath_2026_06_29`) with
      landed-vs-deferred status only after the implementation actually lands.

## Open Questions

- ~~Does any production caller of `CurveCurveDistance.shortest_distance` /
  `CurveSurfaceDistance.shortest_distance` need the *exact* distance above the
  threshold (vs. the capped diagnostic)?~~ **RESOLVED 2026-06-29:** no caller feeds
  `J`/`dJ` or the optimizer (all reporting/certification). Remaining sub-decision:
  the achieved-clearance-metric sites (`banana_coil_solver.py`, `VMEC`,
  `single_stage_geometry.py`, `stage2_objectives.py`, example/status reporters) —
  gate-only (cap is fine) vs. exact-record-needed (give them the `cKDTree` path or
  distinct exact helper)? Owner call per site; see Phase 1 Task 1.
- ~~Is `res['residual']` ever consumed for `type == "ls"` outside diagnostics?~~
  **RESOLVED 2026-06-29 (consumer audit):** No — the LS gate uses
  `jacobian`/`gradient`; only the EXACT path (`handoff.py:956`) + verbose prints
  read the residual. Phase 1 Task 2 is safe when scoped to the LS solver
  (see Assumptions).
- ~~Is the coil-force objective (`MeanSquaredForce`/`LpCurveForce`) active in the
  default single-stage banana lane, or only in dedicated force runs?~~
  **RESOLVED 2026-06-29:** active by default for
  `SINGLE_STAGE/single_stage_banana_example.py` unless `COIL_FORCE_WEIGHT` or
  `--coil-force-weight` disables it. The parser default is
  `SINGLE_STAGE_COIL_FORCE_WEIGHT_DEFAULT` (`hardware_contracts.py:166`, value
  `1.0`) and `JCoilForce` is constructed when `FORCE_WEIGHT > 0.0`
  (`single_stage_banana_example.py:3619-3626, 10298-10300`). Phase 2 Task 1 is
  default-path for that runner.
- ~~Is the audit artifact for `wf_84dfb068-dd5` available in the repo or artifact
  store?~~ **PARTIALLY RESOLVED 2026-06-29:** the workflow ID is not repo-local, but
  the project memory with the 44-finding count exists at
  `~/.claude/projects/-Users-suhjungdae-code-columbia-simsopt/memory/project_perf_audit_hotpath_2026_06_29.md`.
  Remaining action: cite that concrete path or export the workflow result before
  treating the 44-count as SSOT in downstream implementation work.
- Should the DESC `use_jit` default flip be a CLI default change or an
  auto-select on `allow_high_memory` + a memory budget? (Owner decision before
  Phase 5 Task 1.)
- Are GPU-pod timing A/Bs in scope for sign-off, or is the equivalence + test gate
  sufficient given the static cost model?
