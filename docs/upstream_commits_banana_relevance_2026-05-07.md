# Upstream simsopt commits not in `surrogate-confinement-v2` — banana-relevance audit

## Revision history

- **v2 (2026-05-07)** — Anchored to `upstream_check/master = 1b0cc3a96` (true hiddenSymmetries HEAD, 636 commits ahead of fork). Re-derived all PR groupings via `git log <merge>^1..<merge>^2`. Demoted `#586`/`#509`/`#463`/`#563`/`#576`/`#558`/`#567` to LOW after grep-verifying that banana production code (`examples/single_stage_optimization/`) has zero callers of the relevant APIs. Rewrote `#486` conflict basis to cite the actual fork-side surface (`RefinedBoozerResidual` reading `boozer_surface.res` / `weight_inv_modB` / `PLU`). Added `#509` `force_and_torque_overhaul` plus `#593`, `#597`, `#599`, `#602`, `#605`, `#606`, `#608`, `#610`, `#611`, `#616` (all in `upstream_check/master` but absent from local `master`). Appendix now 635 commits.
- **v1 (2026-05-07)** — Initial cut. Anchored to local `master` (which lags upstream by 315 commits). Overstated banana relevance for force/torque, prob.bounds, surface flips, and condense_spectrum. Cited a non-existent `simsopt.jax_core` module as a conflict surface for #528. Failed Codex xhigh review.

## Header

- **Local HEAD**: `2dae544b2` (surrogate-confinement-v2)
- **Upstream target**: `upstream_check/master` = `upstream_hss/master` = `1b0cc3a96` (hiddenSymmetries master, 2026-04-09)
- **Local `master` (lagging mirror)**: `21117aa8d` — do not use as merge target; lags upstream by 315 commits including PR #509
- **Merge-base**: `539c0f98`
- **Upstream commits not in fork**: 636 (`git log HEAD..upstream_check/master --oneline | wc -l`) across 51 PR merges

## Top-line

The honest read of 636 upstream commits across 51 PRs against banana coil optimization as it exists *today*: **almost all of it is irrelevant or LOW-priority for banana**. Banana's actual hot path is narrow — `BoozerSurface.run_code`, `RefinedBoozerResidual` reading `boozer_surface.res`/`weight_inv_modB`/`PLU`, `CurveCWSFourierCPP`, `CurveCurveDistance` / `CurveSurfaceDistance` / `LpCurveCurvature`, `scipy.optimize.minimize(..., bounds=...)`, and particle/fieldline tracing for Poincaré. Most upstream work in this 636-commit window targets areas banana does not touch.

After grep-verifying every claim, the partition is **1 HIGH**, **3 MEDIUM**, **8 LOW**, **39 IRRELEVANT** themes (the remainder are pure CI/conda/docs noise). The single HIGH item is **PR #486** (`ubuntu24` — disguised BoozerSurface rewrite with `vectorize=False` removed, three-IC BFGS warm start, regularization-by-default, `weight_inv_modB` defaulting); banana DOES wire into `BoozerSurface.run_code` (12 banana files) and `RefinedBoozerResidual` reads the result dict, so PR #486's reshape is a direct hit. MEDIUM is the `#528` `downsample` parameter on `CurveCurveDistance` (banana uses it heavily), `#519` `change_resolution` returning a copy (banana calls it on surfaces during stage-II prep), and a small bundle of surgical bug fixes (`#557` `mgrid` 64-bit, `#543` `cross_section`, `#535` poincare angle convention).

LOW captures the *substantial* upstream features banana does NOT use today but might want eventually: PRs #509 + #586 + #558 (force/torque/RegularizedCoil overhaul — would become HIGH if banana wires force/torque into HBT-2 hardware contracts), `#463` `prob.bounds` plumbing (banana uses scipy directly), `#563`/`#576` surface flip + spectral helpers, `#567` `VmecGeometryResults` (banana doesn't call `vmec_compute_geometry`), `#492`/`#550`/`#575` lasym enablement (banana is stellsym today). IRRELEVANT is CI runner bumps, conda packaging, doc/example polish, mayavi/vtk pins, etc.

## HIGH priority for banana

### PR #486 — Ubuntu 24 + BoozerSurface rewrite (disguised in the PR title)

- **Merge SHA**: `f0b67c5c5` (2026-01-16, 70 commits in branch range)
- **Banana benefit**: banana uses `BoozerSurface` heavily — 12 banana files reference it. PR #486 collapses `boozer_penalty_constraints` into a single vectorized solver path, removes the `vectorize=False` branch entirely, adds three-initial-condition BFGS warm start, regularization-by-default, and changes `weight_inv_modB` defaults. Reduces the rate of Newton divergence the fork's ALM driver currently sees on tight stellsym BoozerSurface fits. Also brings non-stellsym (`stellsym=False`, `G=None`, manual option) coverage banana will need for HBT-2 lasym configs.
- **Files**: `src/simsopt/geo/boozersurface.py` (+189/-393 net trim), `tests/geo/test_boozersurface.py` (+342)
- **Conflict surface**: banana does NOT wrap `boozer_penalty_constraints` directly. Banana builds `BoozerSurface` and calls `boozer_surface.run_code(iota, G)` (`examples/single_stage_optimization/banana_opt/stage2_single_stage_handoff.py:497`). Then `RefinedBoozerResidual` consumes `boozer_surface.res["weight_inv_modB"]`, `boozer_surface.res["iota"]`, `boozer_surface.res["G"]`, and `PLU` (`examples/single_stage_optimization/banana_opt/boozer_residuals.py:118-136`). PR #486 changes `weight_inv_modB` defaulting and the `boozer_surface.res` dict shape — verify `RefinedBoozerResidual._weight_inv_modB()` still returns the right value before merging.
- **Conflict risk**: **high** — fork's `0bc13f225` (Boozer lifecycle) and `4fa639aa8` / `b8c45d363` (Newton `lu()` + `forward_solve` + helper extraction) heavily overlap PR #486's residual-vector reshape. Plan a guard branch and re-run the fork's `frontier_evaluator` smoke before merging.

## MEDIUM priority for banana

### PR #528 — JAX-curve overhaul + `downsample=` on `CurveCurveDistance`

- **Merge SHA**: `63918918d` (2025-07-11, 9 commits in branch range)
- **Commits** (verified via `git log 63918918d^1..63918918d^2`): `427bfbf9d`, `1a620a9ff`, `c3a053fc8`, `dcf805c74`, `3fb044beb`, `870aba883`, `de3018773`, `bb187b186`, `82a462efb`, `a0250ca10`
- **Banana benefit**: `CurveCurveDistance.__init__` gains `downsample=1` (verified at `src/simsopt/geo/curveobjectives.py:204` post-merge; module-level helper `cc_distance_pure(downsample=1)` at `:161`). Banana uses `CurveCurveDistance` in stage-II and single-stage drivers (`examples/single_stage_optimization/STAGE_2/banana_coil_solver.py:1633`, `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:3707`); thinning quadrature points by a stride during JAX evaluation directly cuts banana's stage-II Jacobian time on coil-coil distance terms.
- **What does NOT come with this PR**: only `CurveCurveDistance` gains `downsample`. `CurveSurfaceDistance`, `LpCurveCurvature`, `LpCurveTorsion`, `MeanSquaredCurvature`, and `ArclengthVariation` do **not** have `downsample` on their `__init__`. The v1 doc overstated this.
- **Files**: `src/simsopt/geo/curve.py`, `src/simsopt/geo/curveobjectives.py`, `src/simsopt/geo/curveplanarfourier.py`, `src/simsopt/geo/curvexyzfourier.py`, `src/simsopt/geo/jit.py`, `tests/geo/test_curve.py`, `tests/geo/test_curve_objectives.py`
- **Conflict surface**: there is NO `simsopt.jax_core` module on this branch (`ls src/simsopt/jax_core*` returns no matches). The real conflict is fork's `CurveCWSFourierCPP` (`examples/single_stage_optimization/banana_opt/stage2_geometry.py:386`) which sits alongside upstream's `JaxCurvePlanarFourier` additions, and the inline JAX `CurveCWSFourier` class in `src/simsopt/geo/curve.py` which shares the file with upstream's vectorized `jaxfouriercurve_pure` rewrite.
- **Conflict risk**: **medium** — file-level overlap on `curve.py` and `curveobjectives.py` is real but the API additions are mostly disjoint from CWS internals. Pull after #486.

### PR #519 — `change_resolution` returns a copy + dofs preserved

- **Merge SHA**: `c94af3986` (2026-01-07)
- **Banana benefit**: banana stage-II preconditioning calls `surface.change_resolution(...)`. Upstream fixes the silent in-place mutation that was a known footgun. Adopt early as low-risk hygiene.
- **Files**: `src/simsopt/geo/surfacerzfourier.py`, `tests/geo/test_surface_rzfourier.py`
- **Conflict risk**: **low**. Behavioral change (returns new object) — sweep banana callers to ensure they pick up the return value.

### Surgical bug-fix bundle (no API surface)

| PR | Merge SHA | What it fixes | Banana relevance |
|---|---|---|---|
| `#557` | `ab9096ff6` | `mgrid` netCDF v2 / 64-bit overflow in `mgrid_field.py` writer | banana's free-boundary diagnostics use mgrid; netCDF correctness |
| `#543` | `61d65d73b` | `Surface.cross_section` typing fix | banana's fork-only `SurfaceSurfaceDistance` uses cross_section indirectly |
| `#535` | `8e063db61` | `plot_poincare_data` cross_section angle convention | banana's Poincaré plotter expects the upstream convention |
| `#552` | `69f188cac` | `test_flux_through_disk` precision fix | nominal (test) |

- **Conflict risk**: **low**. Each is a single-function fix.

## LOW priority for banana

The following upstream PRs are real upgrades to simsopt but banana code does NOT use the affected APIs today. Verified by ripgrep over `examples/single_stage_optimization/`.

### PR #509 — `force_and_torque_overhaul` (the big one)

- **Merge SHA**: `f8c9be314` (2026-03-01, 221 commits in branch range — substantial)
- **Why LOW for banana**: zero callers of `coil_force` / `self_force` / `RegularizedCoil` / `LpCurveTorque` / `SquaredMeanForce` / `SquaredMeanTorque` / `B2Energy` / `NetFluxes` in banana code. HBT-2 hardware contracts cover length, spacing, curvature, currents, and surface limits — **not** coil force or torque. Verified at `examples/single_stage_optimization/banana_opt/hardware_contracts.py:6` and `examples/single_stage_optimization/banana_opt/hardware_constraint_schema.py:100`.
- **Future scope**: this PR + #586 + #558 collectively become **HIGH** the moment banana adds force/torque/inductance into HBT-2 contracts. They land an entire engineering-constraint vocabulary that maps directly to HBT-2 hardware-spec language.

### PR #586 — RegularizedCoil class

- **Merge SHA**: `1da1db777` (2026-01-28, 7 commits)
- **Why LOW for banana**: same reasoning as #509. No `RegularizedCoil` callers; force/torque is unused.
- **Conflict risk** (corrected from v1): **low**. Banana doesn't call the function form `coil_force(coil, allcoils, regularization)` — that exists only in `src/simsopt/field/force.py:15` and is not invoked from banana code. v1's "high conflict" was unfounded.

### PR #558 — `coil_optimization_helper_functions.py` extraction

- **Merge SHA**: `913c71676` (2026-01-23, 5 commits)
- **Why LOW for banana**: banana has no callers of `coil_optimization_helper_functions`, `vacuum_stage_II_optimization`, `build_stage_II_data_array`, or `make_stage_II_pareto_plots`. Banana's frontier engine (`banana_opt/frontier_*.py`) is a multi-objective NSGA-3 + multilane-local engine with typed archive state; upstream's helpers are a single-objective scalarization scan. They solve different problems.
- **Future scope**: HIGH alongside #509 if banana adds force/torque scans.

### PR #463 — `bounds=prob.bounds` plumbing in solver wrappers

- **Merge SHA**: `c88f7bab5` (2026-01-02, 4 commits)
- **Why LOW for banana**: banana does NOT use `least_squares_serial_solve` / `least_squares_mpi_solve` / `LeastSquaresProblem` / `prob.bounds`. Banana calls `scipy.optimize.minimize(..., bounds=...)` directly (`examples/single_stage_optimization/alm_utils.py:2808`, `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:3912`).

### PR #563 — Surface flip / rotate transformations

- **Merge SHA**: `155ad620c` (2025-11-30, 7 commits)
- **Why LOW for banana**: zero callers of `flip_theta`, `flip_phi`, `flip_z`, `rotate_half_field_period`, `shift_theta_by_half` in banana code.

### PR #576 — `condense_spectrum` / spectral hygiene

- **Merge SHA**: `106733770` (2026-01-08, 20 commits)
- **Why LOW for banana**: zero callers of `condense_spectrum` / `spectral_width` / `plot_spectral_condensation`.

### PR #492 + #550 + #575 — `lasym=True` enablement across vmec / vmec_diagnostics / Boozer

- **Merge SHAs**: `080096159`, `d053f9085`, `098a6feac`
- **Why LOW for banana**: banana is stellsym today. Enables future non-stellsym studies but no immediate benefit.

### PR #567 — `VmecGeometryResults` typed dataclass

- **Merge SHA**: `14cfff2f9` (2026-01-12, 6 commits)
- **Why LOW for banana**: banana code does NOT call `vmec_compute_geometry` or `vmec_fieldlines` (verified via ripgrep). Banana reads VMEC outputs through other paths.

### Other LOW themes

- **QUASR DB / `get_data` API consolidation** (PRs `#532`, `#583`, `#591`): banana doesn't use per-config getters or QUASR seeds.
- **`coils_to_vtk()` writer** (folded into `#586`): banana uses `curves_to_vtk` already.
- **`JaxCurvePlanarFourier`** (folded into `#528`): banana uses `CurveCWSFourierCPP`, not planar JAX curves.
- **Sphinx role refresh + GitHub link cleanup** (PRs `#525`, `#545`, `#565`): docs only.
- **`Optimizable.derivatives` doc** (PR `#545`): clarifies an API banana uses correctly.

## IRRELEVANT (catalogued for completeness)

- **Python 3.13 enablement, conda recipe & singularity bumps** (PRs `#537`, `#538`, `#588`, `#552`): runner / packaging.
- **Ubuntu 24 / macOS 13 retire / hostedtoolcache fix** (CI portion of PR `#486`, plus `#577`, `#571`, `#572`, `#580`): runner bumps.
- **Concurrency cancel-in-progress** (PR `#536`): CI throughput.
- **Wheel build, conda-verify removal, pybind11 churn** (PRs `#534`, `#537`, `#538`): packaging.
- **CI test refactors / coverage knobs** (PR `#529` wf coverage, PR `#560` wireframe indexing, PR `#608` coil-helper coverage, PR `#605` PM test numerical-roundoff, `#552` precision fix): test-only.
- **Singularity → Apptainer migration** (PR `#611`): build infra.
- **Defaults `freebound.sp` SPEC update** (PR `#541`): SPEC vacuum-balance defaults; banana doesn't run SPEC.
- **README link fixes** (PR `#565`).
- **Docstring SyntaxWarning escape fixes** (PRs `#549`, `#568`).
- **Matplotlib local-import** (PR `#592`): one-line lazy import.
- **Examples polish** (PRs `#532` examples touch-ups, `#556` `QH_fixed_resolution.py` MPI count, `#525` GH-link cross-references): examples / docs only.
- **`fix_issue_553` wireframe indexing** (PR `#560`): one-line `dtype=int`.
- **`hotfix_ground_kaptanu`** (PR `#585`) and its revert (PR `#597`): the fork carries its own ground/bentley_ottmann compat shim (`f160ba381`) so this is a no-op.
- **`coils-to-vtk-cumulative-indexing` fix** (PR `#606`): banana doesn't use `coils_to_vtk`.
- **`ag/star_lite_a` STAR-Lite-A config** (PRs `#602`, `#610`): adds `STAR_Lite-A.json` config; banana doesn't use the configs zoo.
- **`mp_boozer_doc_fix`** (PR `#616`): doc-only.
- **`mp_doc_fix`** (PR `#599`): doc-only.
- **`jmh/fieldline_docs`** (PR `#593`): doc-only.

## Future-scope HIGH (if banana adds force/torque constraints)

If banana wires coil-coil force, torque, induced flux, or self-force into HBT-2 hardware contracts, three currently-LOW PRs become **HIGH together** and must be pulled as a coherent unit:

1. **PR #586** — `RegularizedCoil` class: hierarchical coil dof representation with regularization owned by the coil.
2. **PR #509** — `force_and_torque_overhaul`: introduces `B2Energy`, `NetFluxes`, `SquaredMeanForce`, `LpCurveTorque`, `SquaredMeanTorque`, plus JAX-pure inductance helpers, `downsample` plumbing on force objectives.
3. **PR #558** — `coil_optimization_helper_functions.py` extraction: pure-additive new module hosting force-aware coil-optimization scaffolds.

Without all three, force/torque constraints can't be expressed cleanly. The total surface is `src/simsopt/field/coil.py` + `src/simsopt/field/force.py` + `src/simsopt/field/selffield.py` + `src/simsopt/util/coil_optimization_helper_functions.py` (new file) + matching tests.

## Recommended pull-in order (4 stages)

1. **Stage 1 — IRRELEVANT noise** (~39 themes, ~250 commits): rebase-noise only. Take all to flatten the diff. CI/conda/runner/docs.
2. **Stage 2 — Surgical bug fixes (no API surface)** (PRs `#557`, `#543`, `#535`, `#519`): each is a single-function fix or behavioral tightening. Banana picks up minor correctness improvements with near-zero conflict.
3. **Stage 3 — MEDIUM** (PR `#528` JAX-curve overhaul with `downsample` on `CurveCurveDistance`): files overlap with fork's CWS work in `src/simsopt/geo/curve.py`; do a careful merge after Stage 2.
4. **Stage 4 — HIGH (highest conflict)** (PR `#486` BoozerSurface rewrite): isolate on a guard branch. Re-run the fork's `RefinedBoozerResidual` against `boozer_surface.res`/`weight_inv_modB`/`PLU` semantics on a frozen seed before merging back. Expect to merge-edit the fork's `0bc13f225`/`4fa639aa8`/`b8c45d363` Boozer lifecycle + Newton perf work against upstream's residual-vector reshape.

LOW items (`#509`, `#586`, `#558`, `#463`, `#563`, `#576`, `#492`, `#550`, `#575`, `#567`, configs/QUASR, etc.) can be deferred indefinitely without affecting banana research. They become urgent only if banana scope expands into force/torque, lasym, or cross-config benchmark studies.

## Appendix: full commit list (oldest → newest)

```
1b39b700c 2024-07-29 Alan Kaptanoglu Fixed the m_maxima parameter and plotting in the PM4Stell example.
e9766791a 2024-09-05 Alan Kaptanoglu Merge branch 'master' of https://github.com/hiddenSymmetries/simsopt
e92288709 2024-09-12 Alan Kaptanoglu Merge branch 'master' of https://github.com/hiddenSymmetries/simsopt
4a0141e1d 2024-09-23 Alan Kaptanoglu Merge branch 'master' of https://github.com/hiddenSymmetries/simsopt
d36ef4baf 2024-09-26 Alan Kaptanoglu Tests passing nicely. Trying to get a version of stage_two_optimization.py working fully with Jax objects. Had to make some changes for correctly dealing with coils obtained by symmetries, and rotated curves, and so far can plot the curves and initial surface errors correctly.
9384734bb 2024-09-26 Alan Kaptanoglu Seems I have gotten the stage two example working with all Jax objects, and now optimizing over the sum of the net forces squared over all the coils.
6f499b95e 2024-09-26 Alan Kaptanoglu got force minimization working to some extent.
af2d133c4 2024-09-26 Alan Kaptanoglu Playing with the force min example.
75c082b89 2024-09-26 Alan Kaptanoglu Started attempt at torques.
efd3d5b0a 2024-09-26 Alan Kaptanoglu First attempt at getting a full planar coil optimization working with jax.
6316ef853 2024-09-27 Alan Kaptanoglu After lots of debugging, believe I have gotten the torques and total energy and inductance calculations working. Started a version of the self force from hurwitz paper but havent finished, and not sure how I will debug.
279242779 2024-09-27 Alan Kaptanoglu Attempted to get the self force working and immediately find issue that the frenet frame is not implemented for the JaxCurves so need to do this myself in the curve.py file. Also tested that the torque and tve objectives along have good-looking taylor tests and do sensible things when run them on the basic stage two example.
5a42a44d5 2024-09-29 Alan Kaptanoglu Added some more examples for scanning and baselining. Added optimization objectives that compute the net torques and net forces from one set of coils onto another set of coils with different parameters (e.g. TF coils vs dipole coils with differing number of quadpoints).
e78b2478f 2024-09-29 Alan Kaptanoglu Did some more fiddling with the QA example to see improvements. With 26 dipole coils and some fiddling, can get to 9e-4 or so solution in Bn_over_B. Feels like plenty still to hyperparameter tune there. Still need to test and debug the self force calculation and double check all the tve stuff. Then run a self force scan and add to paper.
cf9a995b5 2024-09-30 Alan Kaptanoglu Didnt do much beyond mess around a bit trying to find what the issue is in the self force calculations.
78c86f05c 2024-10-01 Alan Kaptanoglu Scans are too slow on laptop so saving current status and going to try on desktop later.
c18c648f1 2024-10-02 Alan Kaptanoglu Some more fiddling with the examples. Saving to try this on a gpu on greene.
2de04a337 2024-10-03 Alan Kaptanoglu Tried speeding up some of the biotsavart stuff but no luck with jax so far.
4dc12b47c 2024-10-04 Alan Kaptanoglu Merge branch 'master' into planar_coil_arrays
34cf164ee 2024-10-04 Alan Kaptanoglu Merged with coil_forces branch
1d3fa2778 2024-10-04 Alan Kaptanoglu Added some of my metrics. Comparing pointwise and net force minimizations. Net force minimization does minimize net forces better but at cost of much higher pointwise forces. Maybe can combine the objectives.
1194dbadd 2024-10-04 Alan Kaptanoglu Saving example for running at home on laptop to generate the figures I need.
b3885f645 2024-10-09 Alan Kaptanoglu Added reactor scale examples and new speedup python code for gamma calculation from JaxCurves. Switching to Greene to try gpus again.
5fbf6360d 2024-10-14 Alan Kaptanoglu Performed a lot of debugging, including trying to speed up Sienas force calculations by directly computing BiotSavart, rather than calling a BiotSavart object in the loop, which gets very slow with many coils as the number of Optimizable objects balloon. Self forces also appear intolerably high, even for basic planar TF coils. Need to discuss this.
fcd3dff8b 2024-10-14 Alan Kaptanoglu Merged with recent coil_forces branch changes.
426cf6bb8 2024-10-14 Alan Kaptanoglu Merged with laptop changes
8eb1b6bdd 2024-10-15 Alan Kaptanoglu prepping to try pareto scans with net force and torques.
31a1d92d3 2024-10-15 Alan Kaptanoglu Getting ready to merge with laptop changes, minimal changes to examples in this commit.
1089ac9f9 2024-10-15 Alan Kaptanoglu Merge branch 'planar_coil_arrays' of https://github.com/hiddenSymmetries/simsopt into planar_coil_arrays
76319eb8b 2024-10-17 Alan Kaptanoglu Made a bunch of fixes in the direct force and torque calculations. Trying to get various examples in order now and generate Pareto plots.
92cc3780d 2024-10-17 Alan Kaptanoglu Tweaked the examples a tiny bit.
68219b4ea 2024-10-17 Alan Kaptanoglu Some more tweaks to the examples to get rid of interlocking coils and other issues.
97d9acbe0 2024-10-17 Alan Kaptanoglu More example tweaking.
9801fe582 2024-10-23 Alan Kaptanoglu Added updated examples.
bf3dab064 2024-10-23 Alan Kaptanoglu Trying to get the fixed surface dipole example working better.
02473f8c1 2024-10-25 Alan Kaptanoglu Added poincare plotting and example updates.
09f4aaa2b 2024-10-25 Alan Kaptanoglu Merge branch 'planar_coil_arrays' of https://github.com/hiddenSymmetries/simsopt into planar_coil_arrays
c57a07e8f 2024-10-25 Alan Kaptanoglu Added new example updates.
3bdf16987 2024-10-25 Alan Kaptanoglu Merge branch 'planar_coil_arrays' of https://github.com/hiddenSymmetries/simsopt into planar_coil_arrays
16bb776a3 2024-10-25 Alan Kaptanoglu Rerunning the fixed orientations one to remove the interlinking coils in the solution. '
8a2e57b94 2024-10-25 Alan Kaptanoglu Made some example changes. Need to increase the force weighting on the fixed orientation examples since Bnormal errors still seem good enough for poincare plots but pointwise forces are too large, especially on the problematic coils in the inboard side.
aea2f371a 2024-10-29 Alan Kaptanoglu Updated the torques to be computed with respect to the coil barycenter. rerunning some examples. Almost have a working QH example with fixed coils but need to tune the force weight a bit.
beb74ec03 2024-10-29 Alan Kaptanoglu More example tweaking. QA example essentially there with params QA_fixed_orientations_n36_p2.25e+00_c3.50e+00_lw1.00e-03_lt1.30e+02_lkw1.00e+03_cct8.00e-01_ccw1.00e+01_cst1.50e+00_csw1.00e+02_fw1.00e-20_fww0.000000e+00_tw1.00e-24_tww1.000000e-24
b3d7ac492 2024-10-29 Alan Kaptanoglu Merge branch 'master' of https://github.com/hiddenSymmetries/simsopt
4802d6e26 2024-10-29 Alan Kaptanoglu Merge branch 'master' into planar_coil_arrays
5ae14c0d0 2024-11-01 Alan Kaptanoglu Made some more useful changes. Got the MixedLpCurve class running faster by downsampling the calculation, but for some reason the initial compilation is very slow. Still trying to understand how to avoid spawning all the weak references to child processes during optimization with the normal LpCurveForce object, and added downsampling to this too. This is probably worth trying to figure out definitively.
b08061a41 2024-11-01 Alan Kaptanoglu Isolated the weak reference spawning in the jacobian calculation of Jforce but need to figure out a fix.
45dd058e2 2024-11-02 Alan Kaptanoglu Updating laptop branch.
cc527f796 2024-11-02 Alan Kaptanoglu Merge branch 'planar_coil_arrays' of https://github.com/hiddenSymmetries/simsopt into planar_coil_arrays
3f1963611 2024-11-03 Alan Kaptanoglu Still struggling to find out why JaxCurves seem to spawn so many optimizable weak references, especially when B_vjp is used in the dJ calculation in the various Force objectives. For now, seem to have got around it. Finally implemented the corrected jacobian terms for the CurvePlanarFourier objects from Alex, and these are running much faster, including with forces. No issue with generating huge numbers of child processes IF one cleans up the children spawning after every call to the force J or dJ calls. Code ready for a dramatic clean up and finalization.
b0243bf9f 2024-11-06 Alan Kaptanoglu Fixed a bug from when i added openmp loops in the c++ files for curvexyzfourier and curveplanarfourier. Might be worth going back and getting this working at some point for speed. Added option to downsample the curve-curve distance calculation. Got the Lp and SquaredMean forces and torques working, including checking the jacobians, allowing for downsampling, removing as many weak references as possible. Tried tiny speedup in the biot_savart_vjp_kernal calculation. Fixed the center function for the torque calculations. Tried edits in derivative file to no success, to try and speedup the calculations. Got the QH example fully running again with pointwise forces and net torques optimized. Remains to clean up the code and generate the new examples.
ff7027460 2024-11-08 Alan Kaptanoglu Did nothing beyond tune the QH example a bit.
ec33cdd8a 2024-11-12 Alan Kaptanoglu Trying to finalize examples.
cefa7229a 2024-11-21 Alan Kaptanoglu Got the QA example working well, including getting QFMs working and so forth.
c7ef9480b 2024-11-21 Alan Kaptanoglu Merge branch 'master' of https://github.com/hiddenSymmetries/simsopt
ddc1e166d 2024-11-21 Alan Kaptanoglu Merge branch 'master' into planar_coil_arrays
4c4048cc0 2024-11-25 Alan Kaptanoglu Added continuation script for QH.
7727f8f94 2024-11-25 Alan Kaptanoglu Merged with desktop changes.
ae9878125 2024-11-27 Alan Kaptanoglu Cleaned up the post processing plots. Going to reorganize and delete lots of old files now.
318b99335 2024-11-27 Alan Kaptanoglu Moved all the planar coil files into separate folder.
c2ab3eba8 2024-11-27 Alan Kaptanoglu Got the self field unit tests working again, just some normalization factors missing and test_update_points has incorrect check I think.
a53d85a88 2024-11-27 Alan Kaptanoglu Merge branch 'master' of https://github.com/hiddenSymmetries/simsopt
6e1e6d9c2 2024-11-27 Alan Kaptanoglu Merge branch 'master' into planar_coil_arrays
260640795 2024-11-27 Alan Kaptanoglu QH example tweaks but mostly just merging ith desktop code.
f1fb5f762 2024-11-27 Alan Kaptanoglu Merge branch 'planar_coil_arrays' of https://github.com/hiddenSymmetries/simsopt into planar_coil_arrays
b47a168e0 2024-11-27 Alan Kaptanoglu Removing some more files. Adding henneberg example.
be71b5e86 2024-11-27 Alan Kaptanoglu Okay now adding henneberg example
525c1705e 2024-11-29 Alan Kaptanoglu Added henneberg example.
a2e278da5 2024-12-01 Alan Kaptanoglu Tweaked all the examples. Tried to rename some stuff. Got the henneberg solution working better.
b9c7fc4e9 2024-12-03 Alan Kaptanoglu Updating pareto script. Going to try running it on greene.
6fc842dc6 2024-12-04 Alan Kaptanoglu Adding pareto updates to run on greene in parallel.
810a42e6c 2024-12-05 Alan Kaptanoglu Additional tweaking, switching to laptop.
e60873037 2024-12-10 Alan Kaptanoglu Merging with desktop changes.
3286f1ffe 2024-12-10 Alan Kaptanoglu Merge branch 'planar_coil_arrays' of https://github.com/hiddenSymmetries/simsopt into planar_coil_arrays
a80b16758 2024-12-11 Alan Kaptanoglu Testing merging.
9e629e103 2024-12-11 Alan Kaptanoglu Merge branch 'master' into planar_coil_arrays
67468730a 2024-12-12 Alan Kaptanoglu Tweaking examples so they reproduce the results in the paper.
664ef7741 2024-12-13 Alan Kaptanoglu Getting all the eamples reproducing the paper results.
b5f2beef5 2024-12-13 Alan Kaptanoglu Autopepped all the files, which seems hasnt been done in a while on main branch.
d84735e35 2024-12-13 Alan Kaptanoglu Fixed the force tests with downsampling.
fa54c272d 2024-12-13 Alan Kaptanoglu Did some linting with ruff.
18a8c6fba 2024-12-13 Alan Kaptanoglu Cleaning up code and writing up tve calculation with my current method of doing so.
d05ccfeda 2024-12-14 Alan Kaptanoglu Still linting. Added henneberg example without dipoles.
b8933f5fd 2024-12-14 Alan Kaptanoglu Got linting fully working. Added some tests of the self and mutual inductances.
329d26697 2024-12-15 Alan Kaptanoglu Working on getting all unit tests running properly on the github CI and getting TVE objective fully working in the coil_forces example.
6cc7cfb76 2024-12-15 Alan Kaptanoglu Got the TVE working
742954c5c 2024-12-15 akaptano Adding pareto change from greene runs.
a4c3be889 2024-12-15 Alan Kaptanoglu Removed omp functionality from dipoles to check for a race condition during github CI.
a9158f1b3 2024-12-15 Alan Kaptanoglu Moved all the coil force stuff to single folder inside 3_Advanced.
28a0b4b4b 2024-12-15 Alan Kaptanoglu Deleting old files.
b19ad4627 2024-12-15 akaptano Tweaked pareto runs to do the same for TVE.
4f3aea519 2024-12-15 akaptano Merge branch 'planar_coil_arrays' of https://github.com/hiddenSymmetries/simsopt into planar_coil_arrays
8cd3ef70a 2024-12-15 Alan Kaptanoglu Put back the omp, didnt seem to fix the unit test.
c6e5c1bda 2024-12-15 akaptano Merged with laptop changes that moved coil force examples.
2ef63db56 2024-12-16 akaptano Fixed the input file in the pareto scan.
dcd1a50f5 2024-12-16 Alan Kaptanoglu played with the tve a bit more in the coil force scan.
27930110c 2024-12-16 Alan Kaptanoglu Still trying to get unit tests working on github CI, but cannot reproduce the test_Bn and test_curve_optimizable errors seen there so far.
a9e13784e 2024-12-16 Alan Kaptanoglu Did tiny bit of linting.
3ec484cc9 2024-12-16 Alan Kaptanoglu Got the examples running better.
a7571fa4f 2024-12-16 Alan Kaptanoglu Removed some old stuff from RotatedCurve related to previous optimization tools with jax, in attempt to get the github CI working.
2cea45928 2024-12-17 akaptano Did some TVE runs but results look a big strange so going to debug a bit to see if things make sense.
d6327e8ac 2024-12-17 akaptano Merge branch 'planar_coil_arrays' of https://github.com/hiddenSymmetries/simsopt into planar_coil_arrays
88ee6ad90 2024-12-17 Alan Kaptanoglu Downsampled tve calculation looks quite accurate still and speeds up the pareto runs. Checked that it looks like tve is being optimized reasonably. Going to try another set of small runs to verify if anything changes.
d0d729ff0 2024-12-17 akaptano Fixed a dumb bug in the TVE where the magnitude of the currents were being used without the signs in LijIiIj. Rerunning the TVE scan.
220c0d6c3 2024-12-18 akaptano Finished up the TVE scan.
0f97eae89 2024-12-19 Alan Kaptanoglu Took first stab at redoing the passive coil terms.
965500c9e 2024-12-20 Alan Kaptanoglu Merge branch 'planar_coil_arrays' of https://github.com/hiddenSymmetries/simsopt into planar_coil_arrays
2fc5da8bd 2024-12-20 Alan Kaptanoglu Added psc example test.
18540a086 2024-12-20 Alan Kaptanoglu Amazingly, example seems to be running properly but its too slow because each PSCCurrent does the whole Linv * psi solve required to solve for all the currents, and only need to do one of these each iteration. So probably need to make a class PSCArray that has a list of current objects and just keeps them updated.
bc80c687e 2024-12-22 Alan Kaptanoglu Still working on getting the PSC jacobian integrated properly. Is tricky because the PSC biot savart object doesnt want to update the currents and Bfield if the currents are not being directly optimized. So currently the PSCCurrents have their own get_value() overwrite and the Bfield object needs to clear its cache every time the loss function is called during optimization, to make sure its using the new current value. Moreover, tried to move over most of the machinery into a PSCArray class object, since we only want to do have to do a single I = -Linv * psi solve for all the currents and same for the Jacobian. Still debugging this in the two simple cases (1) fixed PSC coils and TF coils varying and (2) fixed TF coils and PSCs varying in orientation.
31ead00f6 2024-12-25 Alan Kaptanoglu Think I went down a wrong road trying to combine the biotsavart objects because its unclear then how to separate out the A() and A_vjp() contributions from the two different sets of coils. I think there might be a way to keep two separate biotsavart objects but still associate the vjp contributions from one with the other.
5eb4f4ff7 2024-12-26 Alan Kaptanoglu Got the PSC array working much more effectively by avoiding a call to the compute all the currents every time the PSCCurrent get_value() function is called. Now recompute_currents() function is called by the PSCArray every time function() is called. This seemed to also fix the issue that the final Bnormal errors from the PSCs did not seem to respect the discrete symmetries. Now looks like everything is working well when finite differences are used. Still no luck finalizing the jacobian calculation, which is also too slow right now.
f154608d2 2024-12-27 Alan Kaptanoglu Made the jacobian calculation much faster by swapping jacfwd with jacrev, since there are many more inputs than outputs. Still debugging jacobian calculation.
d08042b7b 2024-12-27 Alan Kaptanoglu Debugging. Added a test to see if I can even get dI_dgammas through jax agreeing with the finite differences, and so far its not. No idea why. However, during testing, it occurs to me that Im missing an important term in the Jacobian. The vector potential of the TF fields is evaluated at the points along each PSC curve, and there is a dA_dX term associated with it.
15df42552 2024-12-27 Alan Kaptanoglu I tentatively may have fixed the jacobian calculation by just directly computing A() in the coil_currents function. Will test this and clean this up tomorrow.
c6cc8164c 2024-12-28 Alan Kaptanoglu Fixed some lingering bugs. Seems the jacobian is fully working, including with all the TF and PSC dofs, including letting the PSC shapes change
af29321b3 2024-12-28 Alan Kaptanoglu Continuing to clean up the code and speed up the jacobian calculation. Replaced all the dJ calculations with their vjp counterparts. Cleaning up the child processes during the calls to the psc_array now allows me to scale to 50 psc coils or so and still reasonably compute the jacobian.
3a9997d92 2024-12-28 Alan Kaptanoglu Fixed a bug in the A() calculation where I wasnt dividing by the number of quadpoints during the biotsavart integral.
9143cff52 2024-12-28 Alan Kaptanoglu Moved the passive coil example files. Fixed lingering bug that the jacobian was wrong when there were no PSC dofs. Requires the fix in the magneticfieldsum class and a call to invalidate_cache() in fun() during optimization. Only need this cache call when the PSCs have no dofs.
f7907d442 2024-12-29 Alan Kaptanoglu Working on getting a final solution for the schuett-henneberg configuration.
90ea15413 2024-12-30 Alan Kaptanoglu Got the henneberg example finalized. Working on a QH example now. Saving so I can switch to desktop.
bfc20a7e4 2025-01-04 Alan Kaptanoglu Finalizing poincare plots and other postprocessing thngs.
6786968bc 2025-01-04 Alan Kaptanoglu Added CSX files.
f3c7ff453 2025-01-12 Alan Kaptanoglu Saving final status of the psc branch before cleaning it up.
ec870e4d1 2025-01-13 Alan Kaptanoglu Committing last changes from desktop, will merge with latest changes in a moment.
4c66b9365 2025-01-13 Alan Kaptanoglu Merge branch 'passive_coil_arrays' of https://github.com/hiddenSymmetries/simsopt into passive_coil_arrays
25d8c494f 2025-01-13 Alan Kaptanoglu Merge branch 'passive_coil_arrays' of https://github.com/hiddenSymmetries/simsopt into passive_coil_arrays
07f3c35c1 2025-01-13 Alan Kaptanoglu Merge branch 'planar_coil_arrays' into passive_coil_arrays
a4499fc25 2025-02-04 Philipp Jurasic use problem.bounds when available in solvers
d938aa2f0 2025-02-04 Philipp Jurasic test if bounds are respected in least_squares_solve
036f69844 2025-02-28 Alan Kaptanoglu Some small reorganizing and rerunning for setting up the dipole array zenodo package. Still lots of clean up needed.
a019f4419 2025-03-07 Alan Kaptanoglu Small changes to compare with florians coils.
c0be48a9a 2025-03-11 missing-user Merge branch 'hiddenSymmetries:master' into warn-unused-bounds
ec8483247 2025-03-14 Alan Kaptanoglu small change to coil forces script.
3c8724834 2025-03-14 Alan Kaptanoglu Merge branch 'master' of https://github.com/hiddenSymmetries/simsopt
b095f2248 2025-03-14 Alan Kaptanoglu Merged with main
5a74c3bcb 2025-03-14 Alan Kaptanoglu Ran ruff, still fixes to make.
a9e6435ef 2025-03-14 Alan Kaptanoglu Did some linting and added Taylor test for all the force and dipole relevant terms.
c2f63a09a 2025-03-14 Alan Kaptanoglu Linting and added doc files.
2487bba78 2025-03-15 Alan Kaptanoglu Got CSX passive coil example cleaned up, and only requiring one script to run both the initial and continuation optimizations. Need to do similarly for all the other planar and passive coil scripts to really neaten this branch up.
3330e6f6b 2025-03-16 Alan Kaptanoglu Made some renaming changes and tried to get both the initial optimization and continuation scripts in the same python file for the QA, QH, Schuett-Henneberg QA examples. Trying to finish this up for the planar coil arrays before going through the passive coil examples more carefully.
9262a73ca 2025-03-16 Alan Kaptanoglu Still in the process of dipole array script clean up and linting and tweaking.
2c984c72a 2025-03-16 Alan Kaptanoglu Almost got the dipole array examples fully in order. Need to get the doc example fully done now.
fd39c9f0c 2025-03-17 Alan Kaptanoglu Tweaking and cleaning the examples. QH example with continuation not quite reproducing paper results. Henneberg example looks good.
3c199fd13 2025-03-17 Alan Kaptanoglu Moved all the postprocessing into a single script. Got the examples working pretty well.
cb42ed7bf 2025-03-17 Alan Kaptanoglu Combined the nodipole scripts.
9cc8c3dc1 2025-03-20 Alan Kaptanoglu Added in Jakes tokamak stellarator hybrid example.
4953de85b 2025-03-24 Alan Kaptanoglu Fixed SchuettHenneberg example to include the bootstrap current contribution. Looks about the same, although there is a weird line in the paraview plot. Not sure this is a real issue though, since the Bfield is perfectly stellarator and field period symmetric.
b2bc210cc 2025-03-26 Alan Kaptanoglu Added some linting and improvements on Jakes script, including use the MixedLpCurveForce and other functions that avoid making new BiotSavart objects. Next step is to add a LP QA example for Pedro and try to make the MixedLPCurveForce calculations faster.
b8163fdd7 2025-03-26 Alan Kaptanoglu Added Jakes grid setup to a LP QA reactor scale example. Got it running reasonably well for Pedro.
b68e4937f 2025-03-26 Alan Kaptanoglu Did some linting
c5cc6986e 2025-03-27 Alan Kaptanoglu Merge with wireframe update
185567fea 2025-04-16 Alan Kaptanoglu Minor fix in the import file.
202c1ba7c 2025-04-17 Alan Kaptanoglu Edited the muse examples.
f37767040 2025-04-20 Alan Kaptanoglu Cleaned up the passive coil scripts in anticipation for making a Zenodo repository and pull request in May.
b5382ef83 2025-04-20 Alan Kaptanoglu Did a round of autopep
07a2e1ea2 2025-04-20 Alan Kaptanoglu Linted a bit more. Ran ruff and recommit. Cleaned up the SchuettHenneberg tutorial a bit.
ebe50297d 2025-04-20 Alan Kaptanoglu Renamed the folder.
36eeb254d 2025-04-21 Alan Kaptanoglu Added a detailed dipole array tutorial example, as well as renamed some fields, added some passive coil array functionality, added some pictures to the docs.
26174350b 2025-04-21 Alan Kaptanoglu Did some linting.
9477c0940 2025-04-21 Alan Kaptanoglu Deleted old file from Jake.
16d6ece46 2025-04-21 Alan Kaptanoglu Fixed a bug from a conflict between an initialize_coils function from the permanent magnet helper file and a similar function coming from dipole array helper file.
57f8c8bbd 2025-04-21 Alan Kaptanoglu Reorganizing and renaming files while I debug the tests.
a19dbc1d6 2025-04-21 Alan Kaptanoglu Renamed dipole array reproducing files.
c45aafac6 2025-04-21 Alan Kaptanoglu Made a bunch of files executable to run all the example scripts. Fixed a tiny issue in the psc taylor tests (there was no issue but the error check was too stringent).
210a3a25b 2025-04-21 Alan Kaptanoglu Fixed some lingering issues running the serial examples.
764c06e03 2025-04-21 Alan Kaptanoglu Added the zenodo links for both papers.
2905a8ab5 2025-04-21 Alan Kaptanoglu Added all the dipole scripts to run at low resolution if CI is going.
704b3567b 2025-04-21 Alan Kaptanoglu Finalized linting.
9930aee07 2025-04-21 Alan Kaptanoglu Fixed the CI I think.
3612d5712 2025-04-21 Alan Kaptanoglu Think I fixed an error in running the coil force examples.
031cb0bae 2025-04-25 Bharat Medasani Use ubunut 24.04 in place of other ubuntu runners
6fce511f2 2025-04-25 Bharat Medasani Update test_boozersurface.py
94b15de17 2025-04-25 Alan Kaptanoglu Merge branch 'master' into passive_coil_arrays
fa889c6fe 2025-04-26 Alan Kaptanoglu Linted and added a lot coverage.
caf4ce8ac 2025-04-26 Alan Kaptanoglu Attempting to rerun code coverage on pull request updates.
85018595b 2025-04-26 Alan Kaptanoglu Fixed some little bugs in the tests, including a factor of two missing in the coil center calculation and reverted the change to tests.yml. Added some additional testing, and found a bug where the JaxPlanarFourierCurve gives NaN values for everything if you accidentally initialize it with the quaternion dofs zeroed out. To mimic CurvePlanarFourier, there is now a safe division for the quaternion norm, and the bug is fixed.
612307984 2025-04-26 Alan Kaptanoglu Attempted to fix the planar curve test failure in the CI, which appears to be differing c++ results on different machines. Changed the way that the equally spaced planar curve function accepts some additional arguments related to the geometry of the setup.
fcd7f3589 2025-04-27 Alan Kaptanoglu Started fixing a substantial bug I found in the course of checking out the unit test failure in the CI. Looks like the jacobians of all the objective terms depending on the passive coil currents (except the SquaredFlux) were not quite right, since they did not correctly compute the Jacobian with respect to the currents, which requires tracking through the psc_array and calling vjp_setup as needed. Did this so far with the MixedLpCurveForce and similar terms, which fixed the Jacobian calulations. However, havent tried with the MeanSquaredForce or LpCurveForce yet, since this might be tricky with the BiotSavart objects they generate. Additionally, need to speed up the MixedLpCurveForce and similar calculations.
abd0b4f9f 2025-04-27 Alan Kaptanoglu Got rid of the pyplot show.
760a76851 2025-04-28 Alan Kaptanoglu Think I fixed the test error. Going to try and see if the tests pass now.
3bb65ff1c 2025-04-28 Alan Kaptanoglu Tiny change to get the Taylor test working on CI.
b79f48e40 2025-04-29 Alan Kaptanoglu Got the passive coil jacobians looking good with the mixed force and torque objectives. Did some documenting in force.py. Updated the passive coil examples to use the right force and torque objectives (no rerun needed since these all ran without minimizing the forces and torques). Added some tests.
ca2e8269b 2025-04-29 Alan Kaptanoglu Linted and fixed the Taylor test.
55bd04261 2025-04-29 Alan Kaptanoglu Fixed a small error introduced in the inductance calculation when reformatting. Fixed the QASH passive coil example.
c82ee229b 2025-04-30 Alan Kaptanoglu Added some wireframe tests missing from coverage. Wireframe optimization when bnorm_target is passed does not seem to work correctly. It returns the same optimized currents independent of bnorm_target. Will leave this to Ken to fix. Reran some of the QASH files to make sure the virtual casing calculations are working correctly. Added functionality in the QFMs to consider a Bnormal_target, which makes sense only if the QFM surface stays very close to the original surface.
e245139e0 2025-04-30 Alan Kaptanoglu Got the QFM with nontrivial Bnormal_plasma working in the jacobian calculation. Checked this works with QASH examples. Obviously cannot push this too far because it is sort of a hack.
6322a443f 2025-05-01 Kenneth Hammond Fix wireframe optimization test with target bnormal
8960f0483 2025-05-02 Alan Kaptanoglu Finalized the passive coil example scripts and debugged a little error in the force scripts that was causing it to fail the CI. During the Taylor test, Jf.x was being modified for each objective so it needs to be reset after each objective is tested.
e3dce5506 2025-05-02 Alan Kaptanoglu Merge branch 'passive_coil_arrays' of https://github.com/hiddenSymmetries/simsopt into passive_coil_arrays
f413dec7f 2025-05-02 Alan Kaptanoglu Did tiny linting on change from Ken.
678c23a66 2025-05-02 Alan Kaptanoglu Think I fixed the issue with the force and torque and other objectives when python=3.11. Apparently erasing all the biotsavart and coil children objects only affects the jacobian and other calculations in python 3.10 and 3.11 -- somehow python 3.9 does not register this since the jacobians are all calculated correctly there. Unfortunately this probably means that objectives relying on their own biotsavart objects will be even slower to calculate than before, since these erasures were controlling the growth of these optimizable graph objects, which blow up with more and more coils.
4862fde36 2025-05-02 Alan Kaptanoglu Added comments about the optimizable graph blow up issue. Will open a simsopt issue about this soon.
734fe0874 2025-05-05 Alan Kaptanoglu Added much more rigorous taylor tests for the force terms. Still seeing occasional failures of the jacobian calculation in lpcurveforce and meansquaredforce. Notably, these are the classes that depend on a biotsavart object. Also notably, I do not seem to see these errors in LpCurveTorque, which should be equivalent in a lot of ways.
9315d3412 2025-05-05 Alan Kaptanoglu Attempting a major overhaul of the force and torque calculations. Deprecating the old objectives that use BiotSavart objects so we can avoid optimizable graph dependencies. Tried to speed up the other calculations and replace them as the default. Need to lint and update the examples.
ba4fabb5c 2025-05-05 Alan Kaptanoglu slow work trying to get the new force objectives into shape.
b336ba72b 2025-05-05 Alan Kaptanoglu Finally got something decent working in the new coil coil force and torque objectives. basically the problem is that the calculation is very slow if any of the coils have different numbers of quadpoints. Now the default is just to downsample all the coils to the lowest number of quadpoints out there. This should be good enough for the forces. Calculations are now only a little bit slower than the BiotSavart versions.
cde8bf934 2025-05-06 Alan Kaptanoglu Changed all the dipole and other example files to use the new force and torque objectives. Got these objectives working pretty well in jax even if all the curves have different number of quadpoints. Reran and updated the tests. Remains to lint and fix small error still in test_Taylor.
2e1d3bba1 2025-05-06 Alan Kaptanoglu Linted the code.
432a4fa82 2025-05-06 Alan Kaptanoglu Tried fixing the Taylor tests. It looks like the issue that the jacobians blow up every once in a while is still there, even though the coil forces and torques were completely revamped. Presumably this is an issue with singularity somewhere in the calculation.
785d407f2 2025-05-07 Alan Kaptanoglu Merge branch 'master' of https://github.com/hiddenSymmetries/simsopt
7e5c606dd 2025-05-07 Alan Kaptanoglu Resolved merge conflicts with main branch with autopep pull request merged in earlier yesterday.
b8ea42ddc 2025-05-07 Alan Kaptanoglu Merge branch 'master' into passive_coil_arrays
8202f249a 2025-05-07 Alan Kaptanoglu Linted after merge with main.
69708410c 2025-05-08 Alan Kaptanoglu Added some little fixes from Misha review. Deleted 3_Advanced/inputs/ which I think I had accidentally added to the branch.
87f936f48 2025-05-08 Alan Kaptanoglu Deleted all the changes to non-relevant example files. deleted all the dipole and passive coil array examples and docs. Still need to delete the tests and source code.
5bcaaec6a 2025-05-08 Alan Kaptanoglu Deleted a lot more files. Revert a few other changes related to the QFM hack, PSCArrays, and so on.
535f3760f 2025-05-08 Alan Kaptanoglu Fixed the init files which I had accidentally swapped.
c7dc84c4c 2025-05-08 Alan Kaptanoglu Reverted the name change in coil_initialization in the permanent magnet helper functions and the pm tests. Think this is pretty ready to open a pull request.
5b274e890 2025-05-08 Alan Kaptanoglu Readded some files that shouldnt have been deleted, and reverted some files that made negligible changes.
d70cd3312 2025-05-08 Alan Kaptanoglu Fixed the CI error. Did some linting and documentation, responding to Misha and Andrew comments on the original passive_coil_arrays pull request.
0a8bae2dd 2025-05-08 Alan Kaptanoglu Made the fixes initially from Matt. Cleaned up the coil_force_optimization examples. Tried to fix the unit test errors.
003b4e6cb 2025-05-08 Alan Kaptanoglu Making a number of simplifications along the lines of Matt Landremans suggestions. Renamed the TVE class. Simplified the computing and plotting of force and torque terms. Moved that plotting to a coils_to_vtk function. Reran the coil force examples. Not fully passing the unit tests yet
578e59bea 2025-05-08 Alan Kaptanoglu Think I polished off lingering failing unit tests from all the changes to the force and coil objectives syntax.
799f5feeb 2025-05-08 Alan Kaptanoglu Fixed a few more unit tests that were perturbed by the changes to the force objectives.
0ebb93b13 2025-05-09 Alan Kaptanoglu Fixed the ruff check and added coverage, which had decreased because of some changes I made, e.g. I did not delete the old deprecated functions in force.py. Add a test for coils_to_vtk.
37ca7bc4f 2025-05-09 Alan Kaptanoglu Fixed coils_to_vtk working.
420c44a10 2025-05-09 Alan Kaptanoglu Deprecated functions still making the Taylor test fail, so removed them from it and just check the jacobian is the right size. Seems like the new force and torque terms actually do pass all the Taylor tests well. Added a docstring to test_Taylor
3e3374dc6 2025-05-09 Alan Kaptanoglu added some documentation. got the self force functions to also use the default regularization that comes with the coil.
18b83199d 2025-05-09 Alan Kaptanoglu Forgot to rename the coils in the last commit.
34da1dade 2025-05-09 Stefan Buller Fixed decorator to skip test without VMEC or booz_xform installed
4f4d8c10d 2025-05-12 Alan Kaptanoglu Fixed the test_flux_through_disk test by adding more quadrature points. Fixed the linting issue.
f63dfc8fa 2025-05-12 Alan Kaptanoglu Merge branch 'master' into ubuntu24
f8188df29 2025-05-13 Alan Kaptanoglu 1. The CurvePlanarFourier class now is initialized without the stellsym and nfp arguments, which did nothing in the class. Also the C++ documentation was incorrect, there is no factor of nfp in the cos and sin series. 2. Added docstrings to a number of functions in CurvePlanarFourier and JaxPlanarFourier. 3. Added a _make_names function for CurvePlanarFourier and JaxPlanarFourier so that the dof names are much more clear and correspond to the syntax in the docstrings. 4. Added curve tests that check setting the dofs and dof names and verify CurvePlanarFourier and JaxPlanarFourier provide the same output. 5. In create_equally_spaced_curves and create_planar_equally_spaced_curves, the code with and without the jax_flag is now the same because the syntax/functionality of both CurvePlanarFourier and JaxPlanarFourier is now identical. 6. Reran the linting, all the unit tests, and all the example files to verify I did not break anything.
9953db675 2025-05-13 Alan Kaptanoglu Made the recommended fixes from Andrew G. Cleaned up the coil force pareto script functionality and added docstrings. Retested everything and reran linting and unittests and all the examples successfully.
55f53fb9b 2025-05-13 Alan Kaptanoglu Few more small changes from Andrews suggestion, fixed a lot of docstrings in the force calculations to output the correct math expression and correct units.
c5e3f4904 2025-05-13 Alan Kaptanoglu Fix the tests, which failed because optimization_tools.py in examples uses pandas for plotting.
1cbdb5b0f 2025-05-13 Alan Kaptanoglu Finished off remaining test failures and other comments from Andrew. In particular, added lots of docstrings for the curveplanarellipticalcylindrical class, added pandas to the dependencies for the CI to run the examples, fixed a few things in optimization_tools.py
d772aa55d 2025-05-13 Alan Kaptanoglu Merged with master branch changes.
02a019d3c 2025-05-14 Alan Kaptanoglu Merged with curve helical changes from master branch. Added curvehelical checks in curve objectives tests. Think I fixed the linking number test for planar coils.
4f3787d11 2025-05-14 Alan Kaptanoglu Merge branch 'master' into force_and_torque_overhaul
8aca04774 2025-05-14 Alan Kaptanoglu Added a tiny bit of coverage. Checks now that shortest_distance still works if there are no candidates in the list. Added some documentation to the Current classes and removed some unused functions in JaxCurrent.
ef96aadc0 2025-05-14 Alan Kaptanoglu Merge branch 'master' into force_and_torque_overhaul
fc1645424 2025-05-14 Alan Kaptanoglu Made a bunch of docstring and other changes along the lines of Mishas comments.
1eae0b241 2025-05-14 Alan Kaptanoglu Finished off lingering import issues from renaming some parameters.
b2c66c231 2025-05-14 Alan Kaptanoglu Merge branch 'master' into force_and_torque_overhaul
26de6e7d1 2025-05-14 Alan Kaptanoglu Fixed some docstrings and other things.
e49682395 2025-05-15 Alan Kaptanoglu Added tiny amount of coverage. Fixed the setup_uniform_grid function to now guarantee that circular coils of radius R on the grid will not overlap for any reason (including after symmetrization over the symmetry planes).
4384f74b6 2025-05-15 Chris Make change_resolution return copy
67335eb43 2025-05-15 Chris Update example to new change_resolution
a927f3971 2025-05-15 Chris Update test to change_resolution fix
9181ef932 2025-05-15 Chris Update example with new change_resolution
619a8265b 2025-05-15 Chris remove unused import
066a0048c 2025-05-15 Alan Kaptanoglu Removed deprecated force objectives, curveplanarellipticalcylindrical class, tiny changes to jacobian calculation of qfm objective, JaxCurrent class.
42e81db95 2025-05-15 Alan Kaptanoglu Got the tests running again after all the deletions.
fb1207a99 2025-05-16 Chris Copy over only existant fourier modes
67bc16c67 2025-05-18 Bharat Medasani Update the cross references in geo module
53093515e 2025-05-18 Bharat Medasani Update cross references in index
98c4da9e6 2025-05-18 Bharat Medasani update outdated installation instructions
9ef2e1376 2025-05-18 Bharat Medasani Small fixes in overview
65f49b352 2025-05-18 Bharat Medasani Fix cross references and minor issues
57659d016 2025-05-19 Bharat Medasani Add cross references in quasisymmetry example
d3ee345d0 2025-05-19 Bharat Medasani Add gh links to examples
5e56cac77 2025-05-19 Bharat Medasani Add examples in the link text
5328a1ce4 2025-05-19 Chris Copy dofs correctly this time
fff5a7ea9 2025-05-19 Chris Fix last test using old copy call sig
d04ad4c9c 2025-05-20 Chris Merge branch 'master' into cbs/change_resolution_fix
a931cf90e 2025-05-26 Alan Kaptanoglu Merge branch 'master' into force_and_torque_overhaul
82a462efb 2025-05-28 Alan Kaptanoglu Added the changes with jaxcurves and additional tests.
a0250ca10 2025-05-28 Alan Kaptanoglu Added new curveobjectives with updated docstrings and downsample parameter.
4a997ff5b 2025-05-28 Alan Kaptanoglu Added wf coverage increases.
72e087273 2025-05-28 Alan Kaptanoglu Accidentally deleted wf test file.
d73f9e36b 2025-05-29 Alan Kaptanoglu Update test_wf_optimization.py
bb187b186 2025-05-29 Alan Kaptanoglu Fix merge fixes in the curveplanarfourier docstring.
725c15300 2025-05-29 Alan Kaptanoglu Merge branch 'master' of https://github.com/hiddenSymmetries/simsopt
de3018773 2025-05-29 Alan Kaptanoglu Merge branch 'master' into jax_curve_PR
5eed80a1c 2025-06-12 Alan Kaptanoglu Merge pull request #529 from hiddenSymmetries/wf_coverage
870aba883 2025-06-12 Alan Kaptanoglu Fixed things from matt and misha suggestions.
3fb044beb 2025-06-12 Alan Kaptanoglu Merge branch 'master' into jax_curve_PR
dcf805c74 2025-06-13 Alan Kaptanoglu ADded better unit test error messages for curves, along lines of Misha suggestions.
148cb564c 2025-06-20 Armin Ulrich refactor: unify coil configuration loaders into get_data API and update docstring
c40dd1f3e 2025-06-20 Armin Ulrich docs: improved docstring of get_data function
446c1645e 2025-06-20 Armin Ulrich docs: addressed issue in get_data docstring
a7917df90 2025-06-24 Armin Ulrich feat(configs): merge lhd_like into get_data, return nfp & bs correctly
2d05b767c 2025-07-10 Alan Kaptanoglu Merge branch 'master' of https://github.com/hiddenSymmetries/simsopt
c3a053fc8 2025-07-10 Alan Kaptanoglu Made docstring change suggested by Matt in the PR.
1a620a9ff 2025-07-10 Alan Kaptanoglu Made last docstring and other changes in the PR.
4e403a881 2025-07-11 Bharat Medasani Revert to pybind11 2.13.6
ed66b1d0f 2025-07-11 Bharat Medasani Revert pybind11 < 3.0
0ddd80be1 2025-07-11 Bharat Medasani Merge pull request #534 from hiddenSymmetries/pybind11_fix
840493ca4 2025-07-11 Alan Kaptanoglu Merge branch 'master' into jax_curve_PR
427bfbf9d 2025-07-11 Alan Kaptanoglu Fixed silly syntax error.
8f07751ad 2025-07-11 Alan Kaptanoglu Merge branch 'jax_curve_PR' into force_and_torque_overhaul
d1954bf82 2025-07-11 Alan Kaptanoglu Fixed testing issues from merging with jax_curve_PR.
63918918d 2025-07-11 Alan Kaptanoglu Merge pull request #528 from hiddenSymmetries/jax_curve_PR
9e69e128a 2025-07-11 Alan Kaptanoglu Merge branch 'master' into force_and_torque_overhaul
e39413d40 2025-07-11 Alan Kaptanoglu Corrected the example files to use the new jax curve flag.
261263066 2025-07-13 Alan Kaptanoglu Made a new util file with the coil pareto scans and other coil optimization related functions. Moved some functions from permanent magnet helpers, which was getting full of functions that were not specific to permanent magnets.
be428e717 2025-07-14 Alan Kaptanoglu Fixed some unit test and example issues coming from moving all the pareto scan functionality to a helper function file. Added a simple unit test for the new helper function file, but probably coverage is very bad on this right now. Will have to take a look at this more carefully once the coverage updates online.
77e926d13 2025-07-14 Alan Kaptanoglu Think I fixed the example failures.
0aeb36956 2025-07-14 Alan Kaptanoglu Tried to get the unit test folders created correctly.
9bb1ffc2e 2025-07-14 Alan Kaptanoglu Trying to remove the unit test issue.
900c7c07a 2025-07-14 Alan Kaptanoglu Trying a different way to save the files.
ec108855a 2025-07-15 Alan Kaptanoglu Improved the code coverage, which had depreciated since the migration of a bunch of functions to coil_optimization_helper_functions.
af1545a9a 2025-07-15 Alan Kaptanoglu Hopefully fixed remaining unit test issues.
3cd4aaf86 2025-07-15 Alan Kaptanoglu Added paretoset package to be downloaded for unit tests.
b22f0820a 2025-07-18 Kenneth Hammond Update plot_poincare_data to new cross_section angle convention
8e063db61 2025-07-19 Matt Landreman Merge pull request #535 from hiddenSymmetries/kch/fix_poincare_plotter
a17162be8 2025-07-21 Bharat Medasani Fix wrong file names
d156f2a61 2025-07-21 Bharat Medasani Update example_quasisymmetry.rst
9d89d43ac 2025-07-21 Bharat Medasani [skip ci] Fix merge conflicts
a5643906e 2025-07-21 Bharat Medasani [skip ci] Update example_quasisymmetry.rst
72be80d72 2025-07-21 Bharat Medasani [skip ci] Update example_quasisymmetry.rst
91c21e291 2025-07-21 Bharat Medasani [skip ci] Fix virtual casing link
4228f49b3 2025-07-21 Bharat Medasani Cancel in progress workflows if new code is pushed
ba4271236 2025-07-21 Bharat Medasani Delete a comment in conda.yml to test the concurrency
26da7f8a9 2025-07-21 Bharat Medasani Fix pybind11 version in conda build
784e9ec81 2025-07-21 Bharat Medasani Fix bug in version specifier for pybind11
4c1824407 2025-07-21 Bharat Medasani Try a new approach for numpy spec in conda
6d4640e65 2025-07-21 Bharat Medasani Add numpy >2.0
b54eb5f79 2025-07-21 Bharat Medasani Update the python version and singularity version
253e3ff55 2025-07-21 Bharat Medasani Add scikitbuild-core to host
655374af2 2025-07-21 Bharat Medasani Add numpy to host from build
5bbafe5bc 2025-07-21 Bharat Medasani Merge pull request #525 from hiddenSymmetries/mbk/gh_links
5e481fdec 2025-07-21 Bharat Medasani replace main with ref_protected to account for protected branches
d71728457 2025-07-21 Bharat Medasani Merge branch 'mbk/wf-concurrency' into mbk/conda-numpy
f3af06238 2025-07-22 Bharat Medasani Merge pull request #536 from hiddenSymmetries/mbk/wf-concurrency
ee9b90e19 2025-07-22 Bharat Medasani Merge pull request #537 from hiddenSymmetries/mbk/conda-numpy
c1dd68c7d 2025-07-22 Bharat Medasani Remove unused nptyping from requirements
8c905e667 2025-07-23 Chris Update defaults_freebound.sp with converged file after SPEC Update
4eaeb7749 2025-07-23 Bharat Medasani Merge pull request #541 from hiddenSymmetries/cbs/spec_defaultfbfix
6b164cb0e 2025-07-23 Bharat Medasani Merge remote-tracking branch 'origin/master' into mbk/numpy-typing
d88a0e272 2025-07-23 Bharat Medasani Merge branch 'master' into mbk/python313
b362480fc 2025-07-23 Bharat Medasani remove spec and vmec matrix options
ea9ff983f 2025-07-23 Bharat Medasani Merge pull request #539 from hiddenSymmetries/mbk/numpy-typing
14be7f125 2025-07-23 Bharat Medasani Remove spec vmec options in extensive tests
47cde177d 2025-08-04 mishapadidar fixed a typing bug in the arguments of the Surface.cross_section function, and updated the documentation
94d385d7b 2025-08-06 mishapadidar updated Derivatives section of optimizable page
1ff576c20 2025-08-06 Bharat Medasani Merge pull request #545 from hiddenSymmetries/mp_derivatives_section2
489486d62 2025-08-07 mishapadidar covered valueerror with unit test in cross_section
61d65d73b 2025-08-07 Andrew Giuliani Merge pull request #543 from hiddenSymmetries/mp_cross_section_fix
6081c7c23 2025-08-08 Armin Ulrich docs: address feedback mapping for get_data API and relocate bs note
8b7be73a1 2025-08-09 Armin Ulrich docs/configs: update get_data docs with LHD-like parameters and symmetry notes; mark old getters as deprecated; fixed conflicts.
26257ac76 2025-08-11 Stefan Buller Merge branch 'master' of github.com:hiddenSymmetries/simsopt into lasym_boozer Unclear what the conflict was.
ab82ca038 2025-08-11 Stefan Buller Added missing file
161ff15b3 2025-08-14 Alan Kaptanoglu Merge branch 'master' of https://github.com/hiddenSymmetries/simsopt
5962a023e 2025-08-27 Philipp Jurasic Silence SyntaxWarnings from unescaped \ in strings
adb8aebd9 2025-08-27 Matt Landreman Merge pull request #549 from jurasic-pf/docstring-escape-backslash
8051df989 2025-08-28 Alan Kaptanoglu Merge branch 'master' of https://github.com/hiddenSymmetries/simsopt
4e6ffce32 2025-09-05 jhLiu526 Update vmec.py
829e8b686 2025-09-05 Armin Ulrich Merge branch 'hiddenSymmetries:master' into config_zoo_update
a1c0c0bf7 2025-09-05 Armin Ulrich fix: unify base_curves/base_currents usage and current_sum handling
3697d4ba8 2025-09-05 Armin Ulrich fix: Added comment to clarify all_curves usage due to symmetry expansion
1bf45b132 2025-09-11 Armin Ulrich fix(docs): clarify get_data docstring for base_curves (unique coils, no symmetry copies)
53faa35d1 2025-09-12 Chris Smiet Update test_boozersurface.py
660e72526 2025-09-12 Chris Smiet Merge pull request #532 from armulrich/config_zoo_update
ba3459760 2025-09-12 Bharat Medasani Merge branch 'master' into mbk/python313
0e86ee956 2025-09-12 Bharat Medasani Bug fix in singularity.yml
7ce223ef0 2025-09-13 Bharat Medasani Remove conda-verify package
6af9995d5 2025-09-13 Bharat Medasani Change python 3.13 to 3.12
b083d5759 2025-09-15 Bharat Medasani Switch python in tests to 3.11
9a07d65d3 2025-09-15 Bharat Medasani Reduce the precision for the failing test
0da1f076b 2025-09-15 Bharat Medasani Fix bug in singularity.yml
a99588e1c 2025-09-15 Bharat Medasani Add back python 3.13
69f188cac 2025-09-16 Bharat Medasani Merge pull request #552 from hiddenSymmetries/fix_annoying_test
b16b6d372 2025-09-16 Bharat Medasani Merge conflict resolved
640c21729 2025-09-24 Rogerio Jorge Update QH_fixed_resolution.py and input parameters for more clear and faster optimization
de27e0d66 2025-09-24 mishapadidar changed mgrid to write netcdf_file using the version=2 option. this allows for writing to 64bit arrays
2cdd86cea 2025-09-28 jhLiu526 Update test_vmec.py
8dd3af2b0 2025-09-29 jhLiu526 Update test_vmec.py
4c7c787ee 2025-10-02 Chris Smiet Fix test method definition for data transfer during optimization
d053f9085 2025-10-03 Chris Smiet Merge pull request #550 from jhLiu526/master
8bd454c50 2025-10-11 Rogerio Jorge Update QH_fixed_resolution.py to correct MPI process count in example usage
6d239fa7f 2025-10-12 Alan Kaptanoglu Merge branch 'master' into force_and_torque_overhaul
dbc53f395 2025-10-12 Alan Kaptanoglu Added small fix from merge.
299e8f974 2025-10-14 Matt Landreman Added flip_z method for SurfaceRZFourier
d9348d3f7 2025-10-14 Matt Landreman Added the function rotate_half_field_period
55b718dad 2025-10-14 Matt Landreman Removed some lines from test_change_resolution that weren't doing anything
e9e5b4eeb 2025-10-15 Alan Kaptanoglu Merge branch 'master' of https://github.com/hiddenSymmetries/simsopt
b80af5da0 2025-10-15 Alan Kaptanoglu Merge branch 'master' into force_and_torque_overhaul
5c26f49eb 2025-10-15 Alan Kaptanoglu Added permanent magnet and coil routine updates, along with workflow file changes.
d6512892b 2025-10-15 Alan Kaptanoglu Reduced time for running the simple PM script.
080096159 2025-10-15 Bharat Medasani Merge pull request #492 from daringli/lasym_boozer
0b514ed15 2025-10-15 Bharat Medasani Merge pull request #556 from hiddenSymmetries/rj/simplify_opts
3787d1b47 2025-10-16 Bharat Medasani Merge branch 'master' into mbk/python313
1a241541f 2025-10-16 Bharat Medasani Remove 3.9 due to EOL
a7dc86991 2025-10-16 Alan Kaptanoglu Merge branch 'master' into permanent_magnet_helpers_update
93c3acdd4 2025-10-17 Bharat Medasani Revert to ubuntu 22.04 in workflows
ee6ca5556 2025-10-24 Kenneth Hammond Initialize index array as integer type
eadfa074f 2025-10-24 Matt Landreman Added shift_theta_by_half
e6c451678 2025-10-29 Bharat Medasani Merge pull request #560 from hiddenSymmetries/fix_issue_553
8c775f219 2025-11-04 Alan Kaptanoglu Made misha's recommended changes, cleaned up the tests and documentation and reduced redundancy of the functions.
9d1f9f2a3 2025-11-04 Alan Kaptanoglu Fixed some lingering issues from changing the names and functionality of the helper functions. Got the tests and examples running well again.
654c3f159 2025-11-11 Matt Landreman Added flip_phi()
a784300b5 2025-11-11 Matt Landreman Fix linting issue
d642f4cca 2025-11-16 Matt Landreman Added flip_theta()
71987953b 2025-11-23 Matt Landreman Merge branch 'master' into ml/surf_transformations
466b3e53f 2025-11-24 Matt Landreman Fix broken links on readme
26f3c8db9 2025-11-24 Misha Padidar Merge pull request #565 from hiddenSymmetries/20251124-fix-readme-links
8a3e418b5 2025-11-26 Philipp Jurasic Documented vmec diagnostics output type, replaced Struct
af17f2388 2025-11-26 Philipp Jurašić Merge branch 'hiddenSymmetries:master' into vmec-diagnostics-documented-output-type
135b61480 2025-11-26 Philipp Jurasic Escape strings correctly, so tests and imports become less noisy
89e7b1cea 2025-11-26 Philipp Jurasic Union to support older python version
5ffe0c953 2025-11-26 Philipp Jurasic Merge branch 'vmec-diagnostics-documented-output-type' of https://github.com/jurasic-pf/simsopt into vmec-diagnostics-documented-output-type
a64ab38d9 2025-11-28 Matt Landreman Remove SPEC from failing github actions workflows
ea1472b35 2025-11-28 Matt Landreman address codecov error
32af83411 2025-11-28 Matt Landreman Merge pull request #571 from hiddenSymmetries/20251128-disable-SPEC-in-CI
3c1483fbd 2025-11-28 Matt Landreman Check whether lowering ubuntu version resolves test failures
3547b0e69 2025-11-28 Matt Landreman Merge pull request #572 from hiddenSymmetries/20251128-fix-ci
f43dcec40 2025-11-28 Matt Landreman Merge branch 'master' into ml/surf_transformations
e39069711 2025-11-29 Alan Kaptanoglu Merge branch 'master' into ubuntu24
9aa5f5950 2025-11-29 Alan Kaptanoglu Fix incorrect merge with master.
297e210cd 2025-11-29 Alan Kaptanoglu Just removing the converge flag to false since the serialization test doesnt care about convergence anyways.
155ad620c 2025-11-30 Matt Landreman Merge pull request #563 from hiddenSymmetries/ml/surf_transformations
7c518a8ec 2025-11-30 Matt Landreman Merge pull request #568 from jurasic-pf/unescaped-string-syntax-warning
9036a766f 2025-11-30 Alan Kaptanoglu Remove plotting from wireframe examples to avoid crashes in CI. Also trying to debug pip-not-found errors in the CI.
fbdf82c7a 2025-11-30 Alan Kaptanoglu Another attempt at a fix.
662e484ad 2025-11-30 Alan Kaptanoglu Reverting CI changes which didnt seem to help.
e54b742d5 2025-11-30 Alan Kaptanoglu Merge branch 'master' into ubuntu24
3072f40ab 2025-11-30 Matt Landreman Initial commit of condense_spectrum() and tests
083056211 2025-11-30 Matt Landreman Merge in master
ab9096ff6 2025-11-30 Matt Landreman Merge pull request #557 from hiddenSymmetries/mp_mgrid_fix
741b184c5 2025-11-30 Alan Kaptanoglu Attempt to fix small scale boozer surface errors. Now passes in python 312 on my mac but CI still showing big errors. Also attempting to get pip recognized correctly, which appears to be a ubuntu and python312 issue.
d6e9c9973 2025-11-30 Alan Kaptanoglu Keep trying to fix pip issues in python312.
586b21bdd 2025-11-30 Alan Kaptanoglu More cursor suggested fies.
14c0ec542 2025-11-30 Alan Kaptanoglu Merge branch 'master' into ubuntu24
93b2d5db5 2025-11-30 Alan Kaptanoglu Another attempt at ci fixes.
c617c4052 2025-11-30 Alan Kaptanoglu Added some debugging lines.
1c0864b4c 2025-11-30 Alan Kaptanoglu Trying another strateg to get the python312 tests running.
e36de41db 2025-11-30 Alan Kaptanoglu Another try at python312 in the CI.
74f5eb296 2025-11-30 Alan Kaptanoglu Cursor really struggling with this one...
0f141addb 2025-11-30 Alan Kaptanoglu Think Im very close to resolving the python312 errors.
78558b30d 2025-11-30 Alan Kaptanoglu Make pip installation step more verbose for debugging
9d186d4bf 2025-11-30 Alan Kaptanoglu Add debugging to Configure VMEC2000 step to diagnose pip issue
ae2a9ce22 2025-11-30 Alan Kaptanoglu Add comprehensive pip diagnostics to debug why pip disappears
8054a09c5 2025-11-30 Alan Kaptanoglu Fix: Add site-packages to PYTHONPATH if missing from sys.path
5cb56018b 2025-11-30 Alan Kaptanoglu Fix: ALWAYS set PYTHONPATH to site-packages (not conditional)
612b6b74b 2025-11-30 Alan Kaptanoglu Fix: Use check-latest to avoid corrupted Python cache
3878e4524 2025-11-30 Alan Kaptanoglu Exclude Python 3.12 from Ubuntu 24.04 (corrupted hostedtoolcache), remove debugging
5e49d049f 2025-12-01 Alan Kaptanoglu Attempt to fix boozer tests on ubuntu24 branch by adding a little regularization to the solve.
dec0d1815 2025-12-01 Alan Kaptanoglu Try to solve issues with further regularization.
dd4fab3c5 2025-12-02 Stefan Buller Adds support for non-stellarator symmetric configurations in vmec_compute_geometry.
bdb915050 2025-12-02 Stefan Buller Added a test for non-stellarator symmetric vmec_compute_geometry() invocations.
61855039d 2025-12-02 Stefan Buller Updated comment to better describe the test.
c5df5ece2 2025-12-02 Philipp Jurašić Merge branch 'hiddenSymmetries:master' into vmec-diagnostics-documented-output-type
2ee90f734 2025-12-02 Philipp Jurasic Expanded latex docs
130aaae75 2025-12-02 Philipp Jurasic Minor fix in docs
1332d63f5 2025-12-02 Alan Kaptanoglu Trying some boozersurface changes to robustify against finding different minima in newton method.
2d0d7d38e 2025-12-02 Alan Kaptanoglu Quick fix to get the CI closer to passing.
a6e2c8f49 2025-12-02 Alan Kaptanoglu Simplify implementation, revert pip changes on extensive_test.yml file, add some regularization to all the boozer checks by default.
f9f2387a4 2025-12-03 Alan Kaptanoglu Attempt to stabilize the newton solve, which is diverging randomly in the ubuntu24 CI.
e6dc6e9e2 2025-12-03 Alan Kaptanoglu Remove test that the surface is not self-intersecting. When Newton method diverges (as sometimes in the ubuntu24 CI), we now detect this but cannot seem to stop it. So removed this check if the CI is running. Also added to tests.yml a temporary change to solely run the boozersurface tests, so I dont have to wait forever to see if the CI issues have been fixed after every change.
cf69df308 2025-12-03 Alan Kaptanoglu Another attempt to run the boozertests separately.
f99f575d1 2025-12-03 Alan Kaptanoglu Forgot to install coverage package before running test.
7ffcaef40 2025-12-03 Alan Kaptanoglu Giving up basically. The Newton divergence seems to be unavoidable for some reason, so just not running the assert checks for convergence anymore. The method runs, and I dont think it is our fault that newton is diverging.
d97f64973 2025-12-03 Alan Kaptanoglu Ignore more failing tests.
25e54eb4b 2025-12-03 Matt Landreman Trying a few approaches for spectral condensation
40dfa36f6 2025-12-03 Alan Kaptanoglu Adding back in boozer tests. Changing tests.yml to only run tests/geo right now for speedup of seeing the issue in the CI. Adding some debug statements in a moment to figure out further what the issues might be in the boozersurface tests.
65e2c9f16 2025-12-03 Alan Kaptanoglu Printing out vectorize along with the residuals now.
d6e08391f 2025-12-04 Matt Landreman More tests for condense_spectrum
c1232df91 2025-12-04 Matt Landreman Tidying up
809c5bbbe 2025-12-04 Matt Landreman Tried constrained spectral method for spectral condensation
f5fad4c0a 2025-12-05 Matt Landreman spectral condensation: polish docstrings
028de7016 2025-12-05 Matt Landreman condense_spectrum is now more efficient - eliminated solve
bba78b3d3 2025-12-05 Matt Landreman Try to fix non-xsimd CI fail with different python version
517d559c7 2025-12-05 Alan Kaptanoglu Did some various fix attempts on the boozer surface tests. The vectorize=False tests were failing on the CI, and this seems to be because vectorize=False and vectorize=True were normalizing differently before optimization began. I have now standardized that, and otherwise added some stabilizing and adaptive newton updates. Lets see if it fixes the CI.
28a25d7ce 2025-12-05 Matt Landreman Try to get non-xsimd CI working
b82313b12 2025-12-05 Alan Kaptanoglu Increase number of iterations for BFGS (may be needed for the non-vectorized, which still achieves much worse final errors) with hope that will fix the newton solve. Also added a random seed for another optimization test that randomly failed (seems unrelated to the boozer stuff).
e103c8880 2025-12-05 Alan Kaptanoglu Added BFGS to try three initial conditions where iota is perturbed slightly. Seems to make things more robust on my laptop. Nonvectorized is performing as well as vectorized on my laptop now. Lets see how it does on the CI.
6423c9be7 2025-12-05 Alan Kaptanoglu Narrowed it down to the vectorized false code being the problem, removing it from CI and then adding a warning. Reverted other changes to the boozersurface stuff.
def5f7a10 2025-12-05 Alan Kaptanoglu Revert to running all the tests in the CI. Add random seed to iota test that randomly fails on the CI every once in a while, which is completely unrelated to the boozersurface stuff.
c80d9fbee 2025-12-06 Matt Landreman condense_spectrum: added constrained methods again
b67d3a7b3 2025-12-07 Alan Kaptanoglu removed non-vectorized code, to see if coverage can be much improved.
4445874e5 2025-12-07 Matt Landreman Try reverting changes to non-xsimd tests
73e2a71d4 2025-12-07 Alan Kaptanoglu Attempt to run coverage checks with VMEC and SPEC functionality. Right now, mhd/spec.py and field/boozermagneticfield.py have very low coverage because they are not run with VMEC or SPEC installed. So the coverage in the main branch appears artificially low.
be77d87a8 2025-12-07 Alan Kaptanoglu Changed it so tests with VMEC are run with MPI.
621d10e1b 2025-12-07 Matt Landreman Add comment about exponential spectral scaling
03784a358 2025-12-07 Alan Kaptanoglu Get rid of optimization for the serialization test, which is not needed and sometimes hangs in the CI.
8ba794b9c 2025-12-07 Alan Kaptanoglu Forgot to put spec back into packages list and remove warning.
9ed133e21 2025-12-07 Alan Kaptanoglu Reverting change since it appears spec will not work at all right now in the CI.
4a986c0a4 2025-12-07 Alan Kaptanoglu Try to bring back python 3.12 in CI.
cb71dd0ef 2025-12-07 Alan Kaptanoglu Adding debug print statements to determine issue.
1b6192a3d 2025-12-07 Alan Kaptanoglu More pip debugging on python212.
2b7496288 2025-12-07 Alan Kaptanoglu Python312 debugging
b1bbb52d5 2025-12-07 Alan Kaptanoglu Trying more complex solutions now.
de31b54d1 2025-12-07 Alan Kaptanoglu Reverting that change, trying python full.
1a1c88ad9 2025-12-07 Alan Kaptanoglu Attempt to retain CI working in python312 now, but getting rid of all the debugging print statements that were added.
4bc381021 2025-12-07 Alan Kaptanoglu In process of debug print statement deleting, managed to break the tests again. lets see if all runs til the end before tweaking more.
e8e490abc 2025-12-07 Alan Kaptanoglu Attempt to only delete the debugging statements, and keep the python312 working in the CI.
098a6feac 2025-12-08 Chris Smiet Merge pull request #575 from daringli/lasym_vmec_compute_geom
06e216223 2025-12-08 Alan Kaptanoglu Broke everything with an empty if statement in the .yml file. Getting rid of it now.
62077a52d 2025-12-08 Alan Kaptanoglu Merge branch 'master' into ubuntu24
39ea2a949 2025-12-08 Alan Kaptanoglu Increase boozersurface coverage, including non-stellsym, G=None, and manual option.
ccae3c947 2025-12-08 Alan Kaptanoglu Improve boozersurface coverage, fix the tests.
01078b1ed 2025-12-08 Alan Kaptanoglu Attempted to fix bugs introduced in the method=manual least squares solves. Attempted to reduce the complexity of the python312 installation in the CI.
7f140ee7a 2025-12-08 Alan Kaptanoglu Attempt to make the solve more robust in the stellsym=False and other checks that are working locally but failing the CI. Added more descriptive errors messages so I can see the issue.
f727b4753 2025-12-08 Alan Kaptanoglu New test trying to improve coverage so far giving more issues than they are worth. One more try before reverting.
d2908eab9 2025-12-09 Alan Kaptanoglu Attempt to resolve random taylor test error occasionally in testIotas. Attempt to slightly reduce the number of changes in extensive_test.yml
efd1de152 2025-12-13 Alan Kaptanoglu Took plot_2d out of the flag to avoid running it during CI. Simplified the extensive test yml file.
8ddf3b80b 2025-12-16 Matt Landreman Simplify the change to extensive_test
a90cbb193 2025-12-16 Matt Landreman In wireframe examples, CI can skip only plt.savefig, not adjacent statements
030fe1c19 2025-12-16 Matt Landreman macos 13 has been retired from github actions
fe23d47ad 2025-12-16 Matt Landreman Address 2 of @smiet's requests
71b809ee4 2025-12-16 Matt Landreman Refactor condense_spectrum as @smiet requested
28b570591 2025-12-16 Matt Landreman condense_spectrum: polish docstring
d2162990c 2025-12-16 Matt Landreman macos-13 has been retired in github actions
acfd3340f 2025-12-17 Matt Landreman Try macos 15 instead of 14
c24c416e6 2025-12-17 Bharat Medasani Merge pull request #577 from hiddenSymmetries/retire_macos_13
94626ade9 2025-12-17 Matt Landreman Merge master into ubuntu24
ede1c46bd 2025-12-17 Bharat Medasani Update extensive_test.yml
e84d7cd53 2025-12-17 Bharat Medasani Update .coveragerc
f8cdc2af1 2025-12-17 Bharat Medasani Update extensive_test.yml
cdb8b74e4 2025-12-17 Bharat Medasani Merge remote-tracking branch 'origin/ubuntu24-patch' into ubuntu24
6a7e31ff9 2025-12-23 Matt Landreman condense_spectrum: handle edge case of n_phi=1
8cacad818 2025-12-23 Matt Landreman Merge branch 'master' into ml/spectral_condensation
a5b6e7e1b 2025-12-24 Matt Landreman condense_spectrum: address Misha's comment
33004dd76 2025-12-24 Matt Landreman trace_particles_boozer: vpar_inits must be 1D
6d606155d 2025-12-24 Matt Landreman Fix test issues related to converting 1-element array to scalar
784591b09 2025-12-24 Matt Landreman Missed one instance of 1D array -> float in tracing_boozer.py
93e36b199 2025-12-25 Matt Landreman Try to fix subtest_boozer_serialization from hanging
34f1dd1e1 2025-12-25 Matt Landreman Merge pull request #580 from hiddenSymmetries/20251224-fix-ci
c903af5c9 2025-12-25 Matt Landreman Merge branch 'master' into ml/spectral_condensation
0ca56d507 2026-01-01 Matt Landreman condense_spectrum: fix quadpoints in returned surface to match original
a64eca36a 2026-01-01 Matt Landreman Merge remote-tracking branch 'upstream/master' into warn-unused-bounds
09e22b368 2026-01-01 Matt Landreman Implement requests by @mishapadidar for PR #463
634abd00d 2026-01-01 Matt Landreman Fix test_solve_quadratic_bounds
c88f7bab5 2026-01-02 Matt Landreman Merge pull request #463 from missing-user/warn-unused-bounds
6241f45cb 2026-01-03 Philipp Jurašić Clear signature for sphinx docs
b895afbf6 2026-01-06 Matt Landreman condense_spectrum: scale constraints so they are order 1
22e325df1 2026-01-06 Matt Landreman Merge branch 'master' into ml/spectral_condensation
495bb0032 2026-01-07 Chris change resolution returns copy, more docstrings
f2bc71c66 2026-01-07 Chris test cpp fn on changed-res surf
c94af3986 2026-01-07 Bharat Medasani Merge pull request #519 from hiddenSymmetries/cbs/change_resolution_fix
eca62b5e4 2026-01-07 Matt Landreman Address requests by @smiet
106733770 2026-01-08 Matt Landreman Merge pull request #576 from hiddenSymmetries/ml/spectral_condensation
ea286acb7 2026-01-10 Philipp Jurašić Explained missing arguments in sphinx docs
14cfff2f9 2026-01-12 Misha Padidar Merge pull request #567 from jurasic-pf/vmec-diagnostics-documented-output-type
018459376 2026-01-13 Alan Kaptanoglu Merge branch 'master' into ubuntu24
865256b6d 2026-01-16 Alan Kaptanoglu Merge branch 'master' into permanent_magnet_helpers_update
f0b67c5c5 2026-01-16 Andrew Giuliani Merge pull request #486 from hiddenSymmetries/ubuntu24
86cd8c8ae 2026-01-16 Alan Kaptanoglu Merge branch 'master' into permanent_magnet_helpers_update
620db818a 2026-01-21 Alan Kaptanoglu Attempt to deal with strange issue with no ground package installed.
e02b9a594 2026-01-22 Chris implement QUASR loader in get_data style
9251d0b94 2026-01-22 Chris get_data tests
b683094f3 2026-01-22 Chris fix typo
44538f715 2026-01-22 Chris rename test_zoo
2f667ac20 2026-01-22 Chris add test quasr file
f149116c5 2026-01-22 Chris test mocking quasr response
9ebc9166f 2026-01-22 Chris add integration tests actually accessing QUASR
068e2da1f 2026-01-22 Chris add cache limit of 100 for quasr downloads
df9243c7b 2026-01-22 Chris fix mock for system without requests
e012baf86 2026-01-22 Chris silence ruff
b09f48b4e 2026-01-22 Chris fix quasr integration test call
be2d78d9a 2026-01-22 Chris install requests in workflow including integrated tests
ab262da95 2026-01-21 Alan Kaptanoglu Attempt to deal with strange issue with no ground package installed.
61e15c33c 2026-01-22 Chris fix ground imports
edb66d8cf 2026-01-22 Bharat Medasani Merge pull request #585 from hiddenSymmetries/cbs/hotfix_ground_kaptanu
2c509c7b6 2026-01-22 Bharat Medasani Merge master
71cabe9e2 2026-01-23 Alan Kaptanoglu Merge pull request #538 from hiddenSymmetries/mbk/python313
551ac48a0 2026-01-23 Alan Kaptanoglu Merge branch 'master' into permanent_magnet_helpers_update
913c71676 2026-01-23 Bharat Medasani Merge pull request #558 from hiddenSymmetries/permanent_magnet_helpers_update
c0576bd24 2026-01-23 Alan Kaptanoglu Merge branch 'master' into force_and_torque_overhaul
929ff8b0a 2026-01-23 Alan Kaptanoglu Fix CI error in tests.
f9d96731c 2026-01-23 Alan Kaptanoglu Fix CI error from old import. Got rid of cold_starts.sh file (copied it into a comment in initiation.py). Updated the tests to run the stuff with pandas and paretoset to maximize coverage.
0d2ce16b7 2026-01-23 Alan Kaptanoglu Added the functionality for RegularizedCoil class, in anticipation for updating the force and torque calculations. Got the unit tests working again I think.
b2333b3b6 2026-01-23 Alan Kaptanoglu Fix linting error.
424cf1c0e 2026-01-23 Alan Kaptanoglu Fix CI error from old incorrect import.
23b004577 2026-01-27 Chris Merge remote-tracking branch 'origin/master' into cbs/new_quasr_getter
e90cb24fd 2026-01-27 Chris address @mishapadidar issues
3ee8854cb 2026-01-27 Chris address @andrewgiuliani s request
fa04b8715 2026-01-28 Alan Kaptanoglu Added more unit tests
c46af24dd 2026-01-28 Alan Kaptanoglu Fix CI unit test error.
6e5500a8d 2026-01-28 Alan Kaptanoglu Refactored code along Bharats lines, so coil force functionality are methods of RegularizedCoil. Fixed some syntax errors and updated the unit tests.
e819d560b 2026-01-28 Alan Kaptanoglu Another attempt to fix NaN CI error.
1da1db777 2026-01-28 Bharat Medasani Merge pull request #586 from hiddenSymmetries/regularized_coil_class
690d1003c 2026-01-28 Alan Kaptanoglu Think merge with latest master is complete and consistent. Unit tests running again.
26aa2eef6 2026-01-28 Alan Kaptanoglu Attempt to fix unit test errors in CI.
29c88a781 2026-01-28 Bharat Medasani Update singularity.yml
c6a19a67f 2026-01-28 Bharat Medasani Update singularity.yml
6ebe6033d 2026-01-28 Bharat Medasani Remove Python 3.9 from build configuration
d8b48e1f4 2026-01-28 Bharat Medasani Merge pull request #588 from hiddenSymmetries/conda-patch
54537ca0f 2026-01-29 Chris hit last lines
bc23eec7e 2026-01-30 Chris hit cache fallback cache path handling logic in tests
8bdcc4fc7 2026-01-30 Chris hit requests unavailable raise
7dec46c65 2026-01-30 Andrew Giuliani Merge pull request #583 from hiddenSymmetries/cbs/new_quasr_getter
1f5db340b 2026-01-30 Chris only test on python 3.13
0f7927be6 2026-01-30 Chris Merge remote-tracking branch 'origin/master' into cbs/new_quasr_getter
d56c26e06 2026-01-30 Andrew Giuliani Merge pull request #591 from hiddenSymmetries/cbs/new_quasr_getter
74b379e97 2026-01-30 Alan Kaptanoglu Reduced small changes from master to make merge easier. Some changes were unnecessary anyways.
184036e21 2026-01-30 Alan Kaptanoglu Merge branch 'master' into force_and_torque_overhaul
94ec45294 2026-01-31 Alan Kaptanoglu try to fix the unit test and increase coverage.
dc33cff7b 2026-02-01 Andrew Giuliani make matplotlib a local import
21117aa8d 2026-02-01 Alan Kaptanoglu Merge pull request #592 from hiddenSymmetries/ag/matplotlib
55928e9ec 2026-02-02 Alan Kaptanoglu Merge branch 'master' into force_and_torque_overhaul
d660ea8a8 2026-02-02 Alan Kaptanoglu Rescaled all the force and torque objectives to use forces in MN/M and torques in M. Updated the coil force examples and the unit tests.
6489eae15 2026-02-02 Alan Kaptanoglu Rescaled B2Energy to MJ for the same reason.
7d5750ba6 2026-02-02 Alan Kaptanoglu Did some debugging to try and get to the heart of issue 486. Tracked it down to C++ caching potentially, JaxCurve has no problem passing all the taylor tests.
7bfa57bc3 2026-02-03 Alan Kaptanoglu Fix tiny typo in docs.
af46efebb 2026-02-09 Alan Kaptanoglu Delete the extraneous examples.
13c969ea8 2026-02-09 Jake Halpern Adding documentation of starting phi location in compute_fieldines and ouptut values in vmec_fieldlines
b58bcd845 2026-02-09 Jake Halpern Minor rewording
83143aa97 2026-02-09 Jake Halpern Updating docstrings after checking them against the built html with Misha's help
6776edf87 2026-02-09 Alan Kaptanoglu Merge pull request #593 from jhalpern30/jmh/fieldline_docs
d18ff1704 2026-02-09 Andrew Giuliani Revert "Attempt to deal with strange issue with no ground package installed."
6e26640ab 2026-02-09 Andrew Giuliani forcing ground==9 and bentley_ottmann==8
43cd767e6 2026-02-09 Andrew Giuliani fixing typo
9fd0ab824 2026-02-09 Bharat Medasani Remove version constraint on mpi4py
a504ad23f 2026-02-09 Bharat Medasani Merge pull request #597 from hiddenSymmetries/revert-585-cbs/hotfix_ground_kaptanu
92846982b 2026-02-10 Alan Kaptanoglu Merge branch 'master' into force_and_torque_overhaul
ef9a80c0d 2026-02-10 Alan Kaptanoglu Attempt to finalize force and torque overhaul from Mishas recent comments. Removed the subsample functionality, since this was downsampling all the coils to the one with the lowest number of quadrature points. Will need to get the passive coil array branch updated with these changes afterwards. Also cleaned up force.py, added documentation and simplified duplicate code.
a21f501b3 2026-02-10 mishapadidar fixed documentation for coil.py
daf36d8aa 2026-02-11 Alan Kaptanoglu Merge pull request #599 from hiddenSymmetries/mp_doc_fix
34f571f06 2026-02-11 Alan Kaptanoglu Second round of fixes in february from misha and andrew, improving docs and clarifying functionality.
a1565b7ce 2026-02-11 Alan Kaptanoglu Merge branch 'master' into force_and_torque_overhaul
2b580370f 2026-02-11 Alan Kaptanoglu Added some value error checks for source_coils since it should not be optional and should not be the exact same as target_coils. updated some of the docs and syntax. Tried to finalize changes to comments from Andrew and Misha, in particular checks now whether downsample is a multiple of nquadpoints.
0c41db1b7 2026-02-11 Alan Kaptanoglu Add unit tests checking for valueerrors.
98a46adea 2026-02-11 Alan Kaptanoglu Fix linting error.
4cb94279e 2026-02-17 Alan Kaptanoglu Overhauled the force and torque classes to now accept source_coils_coarse and source_coils_fine. Going to push this, check coverage, verify it works on the passive_coil_arrays branch, and then finalize.
b1eb1114f 2026-02-17 Bharat Medasani Fix mayavi installation in singularity and docker containers
a4c7ddaf8 2026-02-17 Alan Kaptanoglu Test didnt work if errors got low enough, taylor test now doesnt try if the error is lower than 1e-8, which is more than sufficient.
cc7570ba8 2026-02-17 Alan Kaptanoglu Merge branch 'master' into force_and_torque_overhaul
fae8edb15 2026-02-18 Alan Kaptanoglu Get coverage back up.
44e6f9229 2026-02-18 Alan Kaptanoglu Fix unit test failure from empty db_filtered.
f8c9be314 2026-03-01 Alan Kaptanoglu Merge pull request #509 from hiddenSymmetries/force_and_torque_overhaul
7e9d63533 2026-03-19 Andrew Giuliani adding STAR_Lite-A to configs.zoo
8ed84038d 2026-03-20 Andrew Giuliani added arxiv link
32a6517d9 2026-03-20 Andrew Giuliani editing the documentation
94a21710e 2026-03-20 Andrew Giuliani Merge pull request #602 from hiddenSymmetries/ag/star_lite_a
b5906bbcd 2026-03-23 Alan Kaptanoglu Fix atol=1e-15 for the pm tests to avoid occasional fails from numerical roundoff.
35775162c 2026-03-23 Alan Kaptanoglu Fix coils_to_vtk indexing for unequal points per coil
b573e6b06 2026-03-23 Alan Kaptanoglu Copy fixes from passive coil arrays branch that increase coverage of the helpers and check bad args.
a2c7539ee 2026-03-23 Alan Kaptanoglu Reduce atol a bit.
39149aa5d 2026-03-24 Misha Padidar Merge pull request #608 from hiddenSymmetries/fix/increase_coil_optimization_helper_coverage
f850cb862 2026-03-24 Misha Padidar Merge pull request #605 from hiddenSymmetries/fix/pm_test_failures_numerical_roundoff
4883116f1 2026-03-24 Misha Padidar Merge pull request #606 from hiddenSymmetries/fix/coils-to-vtk-cumulative-indexing
b738f029b 2026-03-24 Bharat Medasani Update the singularity definition file for simsopt
faacbbb9f 2026-03-26 Andrew Giuliani fixing G values
9206a325b 2026-03-26 Andrew Giuliani Merge pull request #610 from hiddenSymmetries/ag/star_lite_a
2ca29ab6c 2026-04-03 mishapadidar updated documentation of Boozer
a747cad86 2026-04-03 mishapadidar fixed typo
e39fb41d5 2026-04-03 Bharat Medasani Format links in README for better readability
34262cc46 2026-04-05 mishapadidar changed docstring
3100c4bec 2026-04-05 Bharat Medasani Merge pull request #616 from hiddenSymmetries/mp_boozer_doc_fix
b79c22b0a 2026-04-06 Bharat Medasani Update ci/singularity.def
e2b9d4190 2026-04-06 Bharat Medasani Update ci/singularity.def
1b0cc3a96 2026-04-09 Bharat Medasani Merge pull request #611 from hiddenSymmetries/apptainer
```
