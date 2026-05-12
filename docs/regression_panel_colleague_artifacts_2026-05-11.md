# Regression Panel from Colleague Artifacts — Implementation Plan

**Date:** 2026-05-11
**Branch:** surrogate-confinement-v2
**Owner:** TBD
**Status:** Not started
**Trigger:** Need to validate the simsopt-core commits authored by Jung Dae Suh on this branch produced no math-layer regression, given that direct end-to-end replay is infeasible (HW-spec drift, ALM/frontier/basin outer loops, mixed core+banana commits).

**Commit-window definition:** `git log master..HEAD --author="Jung Dae Suh" --oneline -- src/simsopt src/simsoptpp` — 26 commits at the HEAD this plan was authored against (`9c1829aea`). The unfiltered count `master..HEAD -- src/simsopt src/simsoptpp` is **124** (includes upstream-merged commits authored by Antoine Baillod, Elizabeth, Frank Fu, brought in by accessibility-branch merges; those are not in-scope for this regression panel). When this plan is implemented, record the HEAD SHA in `tests/regression/COMMIT_WINDOW.md` and freeze the in-scope SHA list there.

---

## 1. Goal

Establish a forward-pinned regression panel that asserts the simsopt math layer (`src/simsopt/`, `src/simsoptpp/`) produces stable numerical outputs at fixed configurations supplied by a collaborator's artifacts. The panel must:

- Be invariant to outer-loop drift (ALM/frontier/basin code in `examples/single_stage_optimization/banana_opt/` can change without invalidating the panel).
- Be invariant to HW-spec drift (audit floors in `banana_opt/hardware_*.py` can change without invalidating the panel).
- Catch math-layer regressions in the 11 Tier-A commits whose codepaths the artifacts exercise (see §3).
- Be deterministic and ULP-tight where the underlying math is closed-form.

## 2. Non-goals

- Validating ALM convergence behavior, frontier scalarization, or basin selection. Those need their own forward-pinned tests on user-side runners.
- Replaying the colleague's optimizer trajectory. Trajectory diffs are infeasible across different optimizers; we only compare evaluations at the colleague's terminal `x`.
- Validating commits whose codepaths the artifacts do not exercise (see §3 coverage gaps).
- Asserting that the colleague's `x` is optimal under our objective — it is a state, not a stationary point of any objective we ship.

## 3. Coverage map

The 4 artifacts (`bsurf_opt_{01,02,10,20}kA.json`) share a single graph schema and finite-current label family, and they scan serialized `BoozerSurface.I` values (μ₀·I_enc): 1.257e-04 → 2.513e-03, a 20× range across 4 points. They do **not** differ only in `I`: coil currents and other optimized DOFs were independently adjusted by the colleague's optimizer. Do not use cross-artifact differences as a linear-in-`I` oracle. They serialize:

- 1 `BoozerSurface` with `I≠0` (finite enclosed current)
- 1 `SurfaceXYZTensorFourier` (mpol=12, ntor=12, nfp=5, stellsym=True)
- 1 `SurfaceRZFourier`
- 1 `BiotSavart` with 51 `Coil`
- 41 `CurveXYZFourier` + 1 `CurveCWSFourierCPP` + 9 `RotatedCurve`
- 47 `ScaledCurrent` + 23 `Current`
- 1 `Volume`

Mapping to Tier-A commits:

| Commit | Math layer | Exercised by | Needs solved state? |
|---|---|---|---|
| `315a3b107` | Biot-Savart cache (C++) | `bs.B`, `bs.dB_by_dX` at fixed eval points | No |
| `b8c45d363` | Boozer residual hot path (kernel) | finite-`I` Path B: wrapper call plus raw kernel call with `G_eff = G + iota * I` | No solved state for Path B; yes for physical `BoozerResidual` |
| `4fa639aa8` | Boozer residual derivative reuse (kernel) | finite-`I` Path B derivative kernel with `G_eff = G + iota * I` | No solved state for Path B; yes for physical `BoozerResidual` |
| `01828e4f6` | `BoozerResidual.dJ_by_dB` solved-state guard | Either skip OR use sidecar; see §3.1 | **Yes** — sidecar |
| `8efc93ed7` | finite-current Boozer routing | finite-`I` wrapper with fixed nonzero `iota`, compared to raw kernel at `G_eff = G + iota * I` | No solved state for Path B; yes for physical `BoozerResidual` |
| `0bc13f225` | banana invalidation / Boozer lifecycle | mutate-and-restore cache probe (BiotSavart side only; Boozer side gated on solved state) | No (BS only) |
| `238bed3ca` | Boozer ALM compat | constructor parity with `I` kwarg via bridge loader | No |
| `d3688c6ea` | tensor-surface derivative (C++) | `surface.gamma`, `surface.normal` | No |
| `a91f4bbe0` | separable tensor-surface enforcer | `surface.gamma` at non-separable DOFs | No |
| `a30aef73e` | curve objective geometry reuse | `curve.gamma`, `curve.dgamma_by_dcoeff` | No |
| `a81c50f6a` | curve distance downsample contract | `MinimumDistance.J`, `.dJ` on coil set | No |

**Without a solved-state sidecar:** 7 of 11 commits are clean to exercise from artifacts alone (315a3b107, 0bc13f225 [BS side], 238bed3ca, d3688c6ea, a91f4bbe0, a30aef73e, a81c50f6a), and 3 more are covered at kernel/wrapper level by Path B (b8c45d363, 4fa639aa8, 8efc93ed7). That Path B coverage is intentionally lower than a physical solved finite-`I` residual test.

**With a complete solved-state sidecar (§3.1):** all 11 commits have deterministic coverage. Finite-`I` physical residual coverage must use the examples-side `RefinedBoozerResidual` / `BoozerResidualExact` path; upstream `BoozerResidual` remains useful only for the solved-state guard/vacuum residual contract.

### 3.1 Solved-state-dependent commits — required sidecar or kernel-only path

`BoozerResidual.compute()` (`src/simsopt/geo/surfaceobjectives.py:1108`) and `BoozerResidual.dJ_by_dB()` (after `01828e4f6`) both call `self.boozer_surface.run_code_from_last_solution()` which requires `self.res` populated. The colleague's artifacts do **not** serialize `res` (no `iota`, `G`, `PLU`, `weight_inv_modB`, residual). Loading an artifact and calling upstream `BoozerResidual.J()` therefore either raises (`RuntimeError: BoozerSurface has no solved state`) or — worse, depending on `need_to_run_code` — silently triggers a fresh solve, which is platform-dependent and contradicts the frozen-state design. Even with a sidecar, upstream `BoozerResidual` computes the vacuum residual with `G`; finite-`I` physical residuals must route through `examples/single_stage_optimization/banana_opt/boozer_residuals.py`.

Two paths:

- **Path A (preferred, requires colleague):** request a solved-state sidecar per artifact, `bsurf_opt_NNkA.solved_state.json`:
  ```json
  {
    "iota": <float>,
    "G": <float>,
    "weight_inv_modB": <bool>,
    "boozer_type": "ls" | "exact",
    "residual_norm": <float>,
    "src_simsopt_version": "0.1.dev5590+gfc208e657.d20260406",
    "res_payload": {
      "type": "ls | exact",
      "PLU": "<serialized PLU or reproducible reconstruction data>",
      "vjp": "<exact-path VJP reconstruction data if boozer_type=exact>"
    }
  }
  ```
  At load time, decode the artifact as `BoozerSurfaceFiniteI`, set `boozer_surface.res = {...}` from the complete sidecar, set `boozer_surface.need_to_run_code = False`, then evaluate the examples-side finite-`I` objective (`RefinedBoozerResidual` for the recorded grid contract or `BoozerResidualExact` for the single-stage compatibility contract) and the lower-level finite-`I` wrapper/kernel. The scalar fields alone are enough for `RefinedBoozerResidual.dJ_by_dB()`, but not for full `RefinedBoozerResidual.J()` / `dJ()` because the current implementation consumes `PLU` and, on the exact path, a reconstructable `vjp`. This evaluates the actual finite-`I` physics quantity the colleague computed without fresh-solving. Do not use upstream `BoozerResidual.J()` as the finite-`I` physical residual oracle; use it only for a separately labeled vacuum/guard regression.

- **Path B (no sidecar):** drop `BoozerResidual` from the panel. Replace with finite-`I` wrapper/kernel evaluation at deterministic chosen `(iota=0.27, G=μ₀·sum(|I_coil|))`. Compute `I` from the loaded `BoozerSurfaceFiniteI`, evaluate `boozer_surface_residual_finite_I(..., iota, G, I=I)`, and compare `0.5 * ||residual||²` to the raw C++ scalar kernel called with `G_eff = G + iota * I`. Use the same nonzero `iota` for derivative checks. This tests finite-`I` routing and the **C++ kernel itself** (the changes in `8efc93ed7`, `b8c45d363`, `4fa639aa8`), not the physically-correct solved Boozer residual on this surface. Document clearly in the test: "this test exercises finite-I wrapper-to-kernel numerics at a frozen surface, not a meaningful solved Boozer residual."

Decision: start with Path B (no colleague dependency); upgrade to finite-`I` Path A if the colleague can supply sidecars.

Coverage gaps (not exercised by these artifacts at all):

- `78dbd74bb` (derivative aggregation across optimizable graph) — needs a synthetic composite-objective fixture.
- `e9a94b1d0` (field-line tracing / topology gate) — covered separately by `tests/field/test_fieldline.py` (already touched by the commit).

## 4. Approach

Treat each artifact as a frozen state `x` produced by an external optimizer. Evaluate state functions `f(x)` purely in the math layer; do not invoke any outer-loop code. Compare HEAD outputs to HEAD-pinned snapshots that ship with the test. The colleague's outer loop (no ALM, no frontier, no basin) is irrelevant because we do not run it.

Additional oracle: Biot-Savart is exactly linear in coil currents. On one loaded artifact, scale each unique leaf `Current` DOF by 2.0 in memory, re-evaluate `B`, restore the original DOFs, and assert `B_scaled == 2*B_original` at ULP-tight tolerance. This is a reference-free internal-consistency check that catches scale-dependent bugs the snapshots alone cannot. Do not scale `ScaledCurrent.scale`, and do not compare across artifacts.

## 5. Implementation steps

### 5.1 Bridge loader for `BoozerSurface{I=…}` → `BoozerSurfaceFiniteI`

**Why:** `459da8fab` (2026-05-10) removed the `I=` kwarg from `BoozerSurface.__init__` and relocated finite-I support to `examples/single_stage_optimization/banana_opt/boozer_finite_current.py::BoozerSurfaceFiniteI`. Existing serialized graphs with `I≠0` fail to deserialize at HEAD with `TypeError: BoozerSurface.__init__() got an unexpected keyword argument 'I'`.

**Where:** `examples/single_stage_optimization/banana_opt/json_compat.py` (new file).

**What:** Subclass `simsopt._core.json.GSONDecoder` (it has no registry/hook API — `process_decoded` at `src/simsopt/_core/json.py:448` does direct `__import__` + `cls_.from_dict`). The subclass rewrites the decoded JSON tree before delegation: any serialized `@class == "BoozerSurface"` dict carrying an `I` field is changed to `BoozerSurfaceFiniteI`'s module/class, then `super().process_decoded(...)` runs normally. The pre-rewrite is required because `SIMSON.from_dict` uses the base decoder internally once it receives `simsopt_objs`. Callers use `load_boozer_finite_i(path)` or `load(path, cls=BoozerFiniteIDecoder)` from `simsopt._core.optimizable.load`, which accepts a custom decoder via the `cls=` kwarg (`src/simsopt/_core/optimizable.py:1642-1645`).

**Size:** ~40 LOC (slightly larger than the original ~30 because subclassing requires reproducing the dict-rewrite path, not just registering a callback).

**Validation:** All 4 artifacts deserialize without error.

### 5.2 Snapshot generator

**Where:** `tests/regression/_generate_colleague_snapshots.py` (script, not a pytest module — underscored to exclude from collection).

**What:** Loads each of the 4 artifacts, evaluates the invariant set defined in §6, writes one snapshot JSON per artifact under `tests/regression/colleague_artifact_snapshots/bsurf_opt_NNkA.snapshot.json`. Captures the HEAD SHA, simsopt git version, numpy/scipy versions, `OMP_NUM_THREADS`, platform (`sys.platform` + `platform.machine()`), and BLAS info (`numpy.show_config()` summary) in the snapshot.

**Cache hygiene:** Immediately after `load(...)`, call `bs.set_points(eval_points)` on every BiotSavart before evaluating anything. This invalidates any cached state carried over from the colleague's Perlmutter run (which is why the 10kA/20kA serialized BiotSavart objects are 4× larger than 01kA/02kA — they carry Perlmutter-computed cache state). Same for surface objects: explicitly recompute `surface.gamma()` rather than relying on a possibly-cached attribute.

**Size:** ~120 LOC.

**Run once** at the HEAD that defines the baseline. Re-run only when intentional math changes ship; gate re-runs behind explicit reviewer sign-off.

### 5.3 Regression test module

**Where:** `tests/regression/test_colleague_artifact.py`.

**What:** Parametrized test over the 4 currents. Loads the artifact via the bridge loader, re-evaluates the invariant set, asserts against the snapshot using per-category tolerances (§6). Includes:

- `test_surface_dofs_roundtrip` — bit-equal round-trip of serialized DOFs.
- `test_biot_savart_at_eval_points` — B and dB at fixed eval points.
- `test_boozer_residual_kernel` — finite-`I` wrapper call at deterministic nonzero `iota=0.27` and `G=μ₀·sum(|I_coil|)`, comparing `0.5 * ||residual||²` to direct `sopp.boozer_residual(...)` with `G_eff = G + iota * I`. Tests finite-`I` wrapper-to-kernel numerics only, not physically-correct solved residual (§3.1 Path B). Upgrade to `RefinedBoozerResidual` / `BoozerResidualExact` if a complete solved-state sidecar arrives (Path A).
- `test_surface_geometry` — γ, normal sampled on a fixed (φ, θ) grid.
- `test_first_coil_geometry` — γ, dγ/dc on coil 0.
- `test_volume_label` — `Volume.J()`.
- `test_minimum_distance` — coil-coil minimum distance and derivative if applicable.
- `test_in_memory_biot_savart_linearity` (one artifact, in-memory) — load one artifact, snapshot `B0 = bs.B(pts)`, scale each unique leaf `Current` DOF by α=2.0 in memory, evaluate `B1 = bs.B(pts)`, restore, assert `B1 == 2*B0` at rtol=1e-13. Reference-free physics oracle. Replaces the cross-artifact `test_linear_in_I` which was invalid because the colleague's coil currents are not uniformly scaled across artifacts (`bsurf_opt_01kA.json:4499` has scale=-10000, `bsurf_opt_20kA.json:4499` has scale=-16000; the `I` label is decoupled from coil-current values, since the optimizer independently adjusted currents to hit the enclosed-current target). Do not also scale `ScaledCurrent.scale`, because nested `ScaledCurrent` chains would multiply the effective current more than once.
- `test_cache_invalidation_probe` (one artifact) — eval B, mutate coil DOF, restore DOF, re-eval B, assert bit-equal.

**Size:** ~180 LOC.

### 5.4 Eval-point and grid generators

**Where:** `tests/regression/_fixtures.py`.

**What:** Deterministic generators for the fixed eval-point sets used in tests (so eval points are not stored in the snapshot JSON, only the outputs are). Examples:

- 100 eval points on a Halton/Sobol grid in a bounding box around the colleague's plasma surface, seed-pinned.
- A 16×16 (φ, θ) grid on the colleague's surface for γ/normal sampling.

**Size:** ~40 LOC.

### 5.5 CI / pre-commit wiring — **N/A by user directive (2026-05-12)**

Originally proposed to add `tests/regression/` to a GitHub Actions workflow with `OMP_NUM_THREADS=1`. Subsequently scoped out: the panel is a **local Darwin/arm64 forward gate** invoked manually:

```sh
OMP_NUM_THREADS=1 python -m pytest tests/regression/ -q
```

Active workflow note: `.github/workflows/tests.yml:185-201` runs `coverage run -m unittest discover` over `tests/{configs,core,field,geo,mhd,objectives,solve,util}`. `tests/regression/` is **not** in that list and is not intended to gate CI.

The `tox.ini` `OMP_NUM_THREADS=1` setting in `[testenv]` (`4515bcba6`) remains as documentation-of-intent for any future re-enablement of tox-based CI; the active workflow does not use tox. `tests/README.md` documents the local-only acceptance line.

## 6. Test design

### 6.1 Snapshot file format

One file per artifact, JSON, schema:

```jsonc
{
  "artifact": "bsurf_opt_01kA.json",
  "head_sha": "<sha at generation>",
  "simsopt_version": "<git describe>",
  "numpy_version": "...",
  "scipy_version": "...",
  "omp_num_threads": "1",
  "I": 1.2566370614359172e-04,
  "surface_dofs_sha256": "<hash of x_surf as raw bytes>",
  "volume": 6.283...e-02,
  "boozer_residual_kernel": {
    "path": "finite_I_wrapper_vs_raw_kernel",
    "iota": 0.27,
    "G": <float>,
    "I": 1.2566370614359172e-04,
    "G_eff": <float>,
    "value": <float>,
    "norm": <float>
  },
  "biot_savart_eval": {
    "fixture_seed": 1234,
    "n_points": 100,
    "B_sha256": "<hash of B array bytes>",
    "B_sample_first10": [[Bx,By,Bz], ...],
    "dB_sha256": "<hash of dB array bytes>",
    "dB_sample_first10_flat": [...]
  },
  "surface_geometry": {
    "grid_phi_n": 16, "grid_theta_n": 16,
    "gamma_sha256": "...",
    "gamma_sample_first10": [...],
    "normal_sha256": "..."
  },
  "coil0_geometry": {
    "gamma_sha256": "...",
    "dgamma_dcoeff_sha256": "..."
  },
  "minimum_distance": { "value": <float>, "n_coils": 51 }
}
```

SHA-256 of the raw array bytes is the primary comparison; a 10-element prefix is stored as human-readable diagnostic for failure triage.

### 6.2 Tolerances

Per assertion category:

| Category | rtol | atol | Justification |
|---|---|---|---|
| Surface DOF round-trip | 0 | 0 | Pure serialize/deserialize, must be bit-equal |
| Volume (closed-form) | 1e-14 | 0 | Polynomial in surface DOFs |
| Curve γ (Fourier sums) | 1e-14 | 0 | Pure trig sums, ULP-tight |
| dγ/dc | 1e-13 | 0 | One extra polynomial degree |
| Biot-Savart B | 1e-13 | 0 | Closed-form 1/R²; loop sums absorb ULP |
| Biot-Savart dB | 1e-12 | 0 | Derivative of 1/R² |
| Boozer residual kernel value | 1e-12 | 0 | Finite-`I` wrapper vs raw kernel at fixed nonzero `iota` |
| Boozer residual kernel derivative | 1e-11 | 0 | Includes mixed B, B-derivative terms after `G_eff` substitution |
| MinimumDistance | 1e-12 | 0 | min over differentiable distance function |
| In-memory current-linearity oracle | 1e-13 | 0 | Biot-Savart is exactly linear in effective coil currents |
| Cache invalidation probe | 0 | 0 | Restored DOF must produce bit-equal eval |

SHA-256 comparison is the primary check; the per-element tolerance check is a secondary diagnostic when the hash mismatches.

### 6.3 Determinism env

Required for ULP-tight reproducibility:

- `OMP_NUM_THREADS=1` — operator must set this in the shell before invoking the panel. `tests/regression/conftest.py` does **not** mutate the environment (intentional, post-correction); it skips the panel with a loud reason if the env is wrong. The acceptance line in `tests/README.md` makes the requirement explicit. (CI is N/A by user directive — see §5.5.)
- Pinned `numpy` and `scipy` versions in the project's lockfile. `conftest.py` reads `numpy_version` from the snapshot `_meta` and skips on mismatch.
- Document BLAS implementation if cross-platform (e.g., Accelerate on Darwin vs MKL on Linux) — expect small ULP-level cross-platform drift in matrix products; if observed, relax `dB` tolerance to 1e-11 and document in the snapshot file.

### 6.4 Platform pinning — important

The colleague's artifacts were produced on **Perlmutter (NERSC; x86_64, AMD EPYC + Cray libsci / GCC)**. This panel runs on **macOS Silicon (Darwin ARM64; Accelerate / Apple Clang)**. The two platforms differ in:

- BLAS implementation (libsci vs Accelerate) — different reduction order in matmul
- libm transcendentals (different vendor implementations of `sin`/`cos`/`sqrt`)
- C++ compiler (GCC vs Apple Clang) — different FMA emission, different vectorization
- Register width / extended-precision intermediates (x86_64 may use 80-bit; ARM64 does not)

These differences are **irrelevant to this panel** because the artifact stores only IEEE-754 DOFs (bit-equal across platforms via JSON text) and we evaluate all state functions **locally on Darwin Silicon**. The snapshot is therefore a Darwin Silicon baseline by construction. There is no Perlmutter number to match.

Implication: snapshots generated on macOS Silicon are valid on macOS Silicon only. If CI eventually runs on Linux x86_64, do not weaken tolerances — instead, publish a second platform-keyed snapshot (`*.linux-x86_64.snapshot.json`) and parametrize the test by platform. Same-platform same-commit reproducibility remains ULP-tight per §6.2; cross-platform reproducibility is an explicit non-goal.

## 7. Verification (validating the test itself)

Before trusting a green run, prove the test fails when math drifts.

### 7.1 Negative-control injection

Apply a 1e-10 multiplicative perturbation to `bs.B()` in a throwaway branch; assert the regression test reports failure on the `biot_savart_eval` assertion. Revert. This proves the test has the resolution it claims.

### 7.2 Tier-A commit reversal sanity check

On a throwaway branch, revert one pure-core Tier-A commit (e.g., `315a3b107` Biot-Savart cache) and rebuild. Run the regression test:

- If the commit was numerically equivalent (perf-only), the test still passes — consistent with the commit message claim.
- If the commit shifted numerics, the test reports which artifact(s) and which invariant(s) failed.

Document the outcome per Tier-A commit. This produces a per-commit numerics-impact ledger.

### 7.3 In-memory Biot-Savart linearity sanity check

Independently of any snapshot, on a single loaded artifact: snapshot `B0 = bs.B(pts)`, scale each unique leaf `Current` DOF by α=2.0 in memory, evaluate `B1 = bs.B(pts)`, restore, assert `B1 == 2*B0` at rtol=1e-13. Do not also scale `ScaledCurrent.scale`; serialized graphs can contain nested `ScaledCurrent` chains, and scaling both leaves and scales would over-scale the effective coil current. This is a reference-free physics oracle that catches scale-dependent regressions the snapshots alone cannot.

**Note:** the original draft of this section asserted cross-artifact linearity (`B(01kA) * 20 == B(20kA)`). That assertion is **wrong** for these artifacts: the `I` label on `BoozerSurface` is decoupled from coil-current values — the optimizer independently adjusted currents to hit a target enclosed current, and uniform 20× scaling does **not** hold across artifacts (e.g., one ScaledCurrent goes -10000 → -16000, a factor of 1.6, while 47 others stay at -80000). The cross-artifact check has been replaced with the in-memory variant above.

## 8. Risks and gaps

- **R1: BoozerSurface upstream signature drift.** Future upstream changes to `BoozerSurface.__init__` may break the bridge loader. Mitigation: bridge loader lives in `banana_opt/json_compat.py`, isolated from `src/`; it can be updated without touching upstream-compatible code.
- **R2: Cross-platform ULP drift.** Colleague's artifacts were produced on Perlmutter (x86_64 + libsci + GCC); panel runs on Darwin Silicon (ARM64 + Accelerate + Apple Clang). See §6.4. This is a **non-issue for the regression test** because the JSON stores only IEEE-754 DOFs (bit-equal across platforms) and the panel evaluates state functions locally — there is no Perlmutter number to match. If CI is ever added on a second platform (Linux x86_64), publish a platform-keyed snapshot rather than relax tolerances. Same-platform same-commit ULP-tightness (§6.2) is preserved.
- **R3: Snapshot rot under intentional math changes.** When a future commit intentionally changes a math output, the snapshot must be regenerated. Mitigation: require explicit reviewer sign-off and a commit message field listing every snapshot field changed and why. Track in `tests/regression/SNAPSHOTS.md`.
- **R4: Coverage gaps (`78dbd74bb`, `e9a94b1d0`).** Two Tier-A commits unexercised. Mitigation: §11 adds synthetic fixtures for derivative aggregation; tracing covered by existing `tests/field/test_fieldline.py`.
- **R5: Outer-loop bugs invisible to this panel.** ALM/frontier/basin code regressions are not caught here. Mitigation: separate forward-pinned tests on user-side runners (out of scope for this plan).
- **R6: Colleague artifact provenance.** Files were built from simsopt `fc208e657` (2026-04-06), not present in our remotes. We cannot rebuild them. Mitigation: treat the artifacts as immutable inputs and version-pin them in-repo (or by content hash) under `tests/regression/fixtures/`.

## 9. Acceptance criteria

The plan is complete when all of the following hold:

- [ ] **AC1.** All 4 artifacts load via the bridge loader without exception.
- [ ] **AC2.** Snapshots exist at `tests/regression/colleague_artifact_snapshots/bsurf_opt_{01,02,10,20}kA.snapshot.json` and were generated by `_generate_colleague_snapshots.py` against the documented HEAD SHA.
- [ ] **AC3.** `pytest tests/regression/test_colleague_artifact.py -v` passes locally on macOS/Darwin (`OMP_NUM_THREADS=1`).
- [ ] **AC4.** Negative-control injection (§7.1) demonstrably fails the test.
- [ ] **AC5.** In-memory Biot-Savart current-linearity sanity check (single artifact, unique leaf `Current` DOFs ×2 in memory, assert B doubles at rtol=1e-13) passes. The original cross-artifact assertion was removed — see §7.3 for why.
- [ ] **AC6.** Cache invalidation probe passes (bit-equal after mutate-and-restore).
- [ ] **AC7.** Per-Tier-A-commit numerics-impact ledger (§7.2) produced and committed to `tests/regression/TIER_A_LEDGER.md`. Each entry: commit SHA, "no observed shift in panel invariants" or "shifted [field], delta [magnitude]", or "unexercised by this panel" / "joint/stacked evidence only" where applicable. Verdicts must be scoped to what the panel actually observes — do not claim "numerically equivalent" when the codepath is not exercised.
- [ ] **AC8.** ~~CI runs `tests/regression/` and is green.~~ **N/A by user directive (2026-05-12).** Replaced with: `OMP_NUM_THREADS=1 python -m pytest tests/regression/ -q` passes locally on Darwin/arm64, with `tests/README.md` documenting the local-only acceptance line and `tests/regression/conftest.py` enforcing platform + env gating.
- [ ] **AC9.** `tests/regression/README.md` documents: what the panel proves, what it does not prove, how to regenerate snapshots, how to add a new artifact.

## 10. Sequencing

Order of work, in commit-sized chunks:

1. **Step A.** Bridge loader (§5.1) + one-liner script that loads all 4 artifacts and prints class counts + I values. Validates load path before going further. *Single commit, ~50 LOC.*
2. **Step B.** Eval-point fixtures (§5.4). *Single commit, ~40 LOC.*
3. **Step C.** Snapshot generator (§5.2). Run it, commit the 4 snapshot files. *Two commits: code, then generated artifacts.*
4. **Step D.** Regression test module (§5.3). *Single commit, ~180 LOC.*
5. **Step E.** Verification (§7.1, §7.3). Negative-control commits do not land; they prove the test resolution. Produce the impact ledger (§7.2). *No code commit; ledger is markdown.*
6. **Step F.** CI wiring (§5.5) + README. *Single commit.*

Each step is independently mergeable and individually testable.

## 11. Follow-ups (out of scope; tracked here for context)

- **F1.** Synthetic fixture for `78dbd74bb` (derivative aggregation across composite objectives). ~60 LOC; uses a 2-objective sum on a small coil set.
- **F2.** Confirm `tests/field/test_fieldline.py` exercises `e9a94b1d0` (tracing/topology). If yes, document the mapping in `tests/regression/TIER_A_LEDGER.md`. If no, add a fixed-seed tracing micro-test.
- **F3.** Extend the panel as new colleague artifacts arrive — same loader, new parametrize entry, new snapshot. Adding an artifact is a config change, not a code change.
- **F4.** Optional: persist eval-point sets in the snapshot (rather than regenerate). Tradeoff: snapshot files grow ~3×; gain reproducibility independent of `_fixtures.py` evolution. Decision deferred until first snapshot regeneration occurs.
- **F5.** *Optional cross-platform sanity (if colleague has run logs).* If the colleague can supply any scalar quantities they evaluated on Perlmutter at the same `x` (e.g., final objective value, Boozer residual norm, Volume), do a one-time loose comparison at rtol=1e-8 to confirm Darwin Silicon agrees with Perlmutter on the physics within reasonable BLAS-platform drift. This is **not a regression test** — it's a one-shot sanity check that there is no gross numerical disagreement between platforms. Result is logged in `tests/regression/CROSS_PLATFORM_SANITY.md`. If it disagrees beyond 1e-8 on closed-form quantities like Volume, that is a real bug, not platform drift.

## 12. Operational notes

- The 4 artifacts under `/Users/suhjungdae/code/columbia/banana_drivers/inputs/` are read-only inputs to this plan. Do not modify them. If they are moved into the repo (recommended for reproducibility), preserve content hash.
- Per-platform snapshots may eventually be needed. If a Linux-CI snapshot diverges from the macOS-Darwin snapshot at runtime, the resolution is to publish a second snapshot file under a platform-keyed name (`bsurf_opt_01kA.linux.snapshot.json`), not to weaken tolerances.
- Snapshot regeneration requires a written justification entry in `tests/regression/SNAPSHOTS.md` (commit, date, fields changed, why). Treat snapshot files as protected per `CODEOWNERS` if available.
