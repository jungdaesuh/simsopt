# JAX Clean Branch Reconciliation Commit Classification - 2026-06-11

## Source Boundary

- Clean worktree: `/Users/suhjungdae/code/columbia/simsopt-pr-jax-port-clean`
- Clean branch: `pr/jax-port-clean`
- Clean HEAD: `a4b4a583e45d75e83118ce8a9291aa415caff402`
- Donor worktree: `/Users/suhjungdae/code/columbia/simsopt-pr-jax-port-pure`
- Donor branch: `pr/jax-port-pure`
- Donor HEAD: `98f3efe037d6a94d95f2a437d3ba96652addc723`
- Merge base: `fc28d62f8e84e8f194ac5d1e74e360693b0ec368`

Clean dirty and untracked files at classification time:

```text
docs/jax_clean_branch_reconciliation_implementation_plan.md
docs/jax_gpu_integration_batches_2026-06-05/batch_001_paths.txt
docs/jax_gpu_integration_batches_2026-06-05/batch_002_paths.txt
docs/jax_gpu_integration_batches_2026-06-05/batch_003_paths.txt
docs/jax_gpu_integration_batches_2026-06-05/batch_004_paths.txt
docs/jax_gpu_integration_batches_2026-06-05/batch_005_paths.txt
docs/jax_gpu_integration_batches_2026-06-05/batch_006_paths.txt
docs/jax_gpu_integration_batches_2026-06-05/batch_008_paths.txt
docs/jax_gpu_integration_batches_2026-06-05/batch_009_paths.txt
docs/jax_gpu_integration_batches_2026-06-05/batch_010_paths.txt
docs/jax_gpu_integration_batches_2026-06-05/batch_018_paths.txt
docs/jax_gpu_integration_batches_2026-06-05/batch_019_paths.txt
docs/jax_gpu_integration_batches_2026-06-05/batch_020_paths.txt
docs/jax_gpu_integration_batches_2026-06-05/batch_021_paths.txt
docs/jax_gpu_integration_test_paths_2026-06-05.txt
scripts/jax_gpu_failed_stale_tests_signoff.py
src/simsopt_jax/core/dipole_field.py
src/simsopt_jax/core/pm_optimization.py
src/simsopt_jax/core/surface_fourier_kernels.py
src/simsopt_jax/solve/permanent_magnet.py
tests/conftest.py
tests/jax/core/test_dipole_field_item24.py
tests/jax/core/test_pm_optimization_jax_item25.py
tests/solve/test_permanent_magnet_optimization_jax_item28.py
tests/test_gpu_transfer_guard_harness.py
```

Donor dirty files at classification time:

```text
docs/jax_stage2_single_stage_artifacts_2026-06-08.md
scripts/jax_gpu_failed_stale_tests_signoff.py
src/simsopt/field/sampling.py
src/simsopt/field/tracing.py
src/simsopt_jax/core/pm_optimization.py
src/simsopt_jax/core/surface_fourier_kernels.py
src/simsopt_jax_adapters/geo/surface_objectives.py
tests/jax/core/test_pm_optimization_jax_item25.py
tests/test_gpu_transfer_guard_harness.py
```

## Commit Classification

| Commit | Subject | Classification | Clean action |
| --- | --- | --- | --- |
| `57385db6d` | `feat: add isolated JAX package port` | Superseded by clean | Clean has reconstructed equivalent base in `c9b6282a3`; do not wholesale replay the broad port commit. |
| `8dd641aaf` | `docs: add JAX test coverage expansion plan` | Diagnostic/doc-only | Leave as donor history unless a later docs pass explicitly needs its coverage-plan prose. |
| `ce39eda8a` | `Fix JAX GPU stale test collection` | Superseded by clean / out of scope | Broad legacy and source-layout changes conflict with clean boundary; clean has replacement signoff work. |
| `47cad0598` | `Update JAX GPU cleanup proof status` | Diagnostic-only | Status doc for donor-era proof, not clean signoff authority. |
| `c4ca6dae7` | `Remove local dynamic smoke imports` | Superseded by clean | Clean branch already carries static import smoke cleanup in its reconstructed history. |
| `77a944c51` | `Remove stale JAX owner facades` | Superseded by clean / out of scope | Broad legacy facade changes are outside the current clean slice. |
| `67622bd16` | `Remove Stage 2 dynamic test reloads` | Superseded by clean | Clean replacement commits already remove dynamic reload behavior for the signoff surface. |
| `bea38b7a6` | `Add JAX GPU stale-test signoff harness` | Superseded by clean | Clean carries the harness and inventories; current work refreshes stale clean inventory instead of replaying donor paths. |
| `0ddebd78b` | `Fix transfer guard signoff probe` | Superseded by clean | Covered by clean harness replacement commits. |
| `bd4c65659` | `Align RunPod signoff harness transfer guard` | Superseded by clean | Covered by clean harness replacement commits and current dry-run validation. |
| `4c896797a` | `Fix RunPod signoff cache mode` | Superseded by clean | Covered by clean harness replacement commits; donor cache env behavior is not copied blindly. |
| `55480dda1` | `Fix regular grid NaN native bounds` | Out of scope | Touches native C++ bounds behavior; excluded from the clean PR boundary for this slice. |
| `1b607d74a` | `Fix pure JAX GPU signoff blockers` | Mixed, superseded by clean | Broad donor fixes were replaced by clean commits `b1cc98645` through `56d85b14a`; no wholesale cherry-pick. |
| `113dc6b9f` | `docs: refresh JAX coverage expansion plan` | Diagnostic/doc-only | Donor coverage status, not clean signoff authority. |
| `5a4304462` | `test: expand JAX legacy parity coverage` | Out of scope / superseded by clean | Broad legacy parity expansion is outside the current clean signoff slice. |
| `a5b72d4d4` | `fix: address JAX GPU review findings` | Mixed, superseded by clean | Clean replacement commits own the relevant review fixes; no wholesale replay. |
| `0abd8d93e` | `fix: restore tracing metadata and parity contracts` | Superseded by clean | Clean commit `7c9ad0e36` owns tracing metadata in legacy core. |
| `33be4b965` | `fix: close jax gpu root failures` | Mixed, superseded by clean | Clean replacement commits cover relevant root failures; broad extras are outside this slice. |
| `177ecb392` | `docs: plan BoozerSurfaceJAX LOC reduction` | Out of scope | Deferred Boozer cleanup, not required for clean branch signoff. |
| `14677a684` | `refactor: dedupe BoozerSurfaceJAX residual paths` | Out of scope | Deferred Boozer cleanup, not required for clean branch signoff. |
| `6e97f7984` | `test: simplify BoozerSurfaceJAX exact residual coverage` | Out of scope | Deferred Boozer cleanup, not required for clean branch signoff. |
| `33c71901d` | `fix: silence force docstring escape warning` | Out of scope | Legacy docstring warning outside clean signoff slice. |
| `72c8128cf` | `refactor(jax): isolate legacy package from jax adapters` | Superseded by clean | Clean reconstructed branch already has its own isolated package/adapters boundary. |
| `21b4f7315` | `docs: plan clean JAX PR reconstruction` | Diagnostic/doc-only | Useful historical context; clean authority remains `docs/jax_clean_pr_reconstruction_audit.md` plus the current reconciliation plan. |
| `df0b9f845` | `docs: curate stage2 single-stage test artifacts` | Doc-only useful | Port or rewrite only after verifying every artifact path and preserving Stage 2 versus single-stage/BoozerSurface contracts. |
| `454a5c2db` | `test: isolate strict transfer guard signoff lane` | Clean-required, partially ported | Clean harness keeps focused selector isolation but removes stale donor selectors and uses clean-specific output fields. |
| `e421deb3f` | `fix(jax): repair targeted GPU stale selectors` | Clean-required, partially ported | PM projection behavior is ported as a clean-specific CPU-wrapper parity fix; unrelated donor hunks are not replayed. |
| `d698d26bc` | `fix(jax): align dipole axis basis convention` | Rejected for clean source | Donor commit fails the live native oracle here. Focused test failed in both clean and pure, then passed after restoring the non-finite SIMD convention in clean. |
| `bbb918d74` | `test(jax): allow native grid NaN error variant` | Out of scope | Native regular-grid test tolerance change is not part of current clean signoff slice. |
| `7248cdba8` | `fix(jax): refresh banana example driver paths` | Out of scope | Example path refresh is outside current clean signoff slice. |
| `5d2817493` | `refactor(jax): simplify Boozer penalty closures` | Out of scope | Deferred Boozer cleanup, not required for clean branch signoff. |
| `24aebddc4` | `docs: update clean JAX PR sync plan` | Diagnostic/doc-only | Donor sync-plan status is superseded by this clean reconciliation plan. |
| `98f3efe03` | `docs: plan split Perlmutter CPU/GPU benchmarks` | Doc-only useful | Rewrite useful benchmark-run requirements into clean docs before final CPU/GPU reruns. |

## Donor Dirty Patch Classification

- `src/simsopt_jax/core/surface_fourier_kernels.py`: donor dirty patch rejected after Crucible review. Its weighted split preserved strict-guard staging but let `0 * NaN` contaminate other coordinate blocks. Clean now uses a true `jnp.split` and adds a direct `_split_flat_to_xyzc` regression test.
- `scripts/jax_gpu_failed_stale_tests_signoff.py`: clean-required behavior, not patch-identical. Clean intentionally drops stale/missing donor focused selectors and records focused lane/repro counts separately.
- `src/simsopt_jax/core/pm_optimization.py` and `tests/jax/core/test_pm_optimization_jax_item25.py`: clean-required behavior, not patch-identical. Clean extends donor zero-radius handling to exact CPU-wrapper parity for signed zero, infinities, and NaNs.
- `docs/jax_stage2_single_stage_artifacts_2026-06-08.md`: doc-only useful pending path verification before port/rewrite.
- `src/simsopt/field/sampling.py`, `src/simsopt/field/tracing.py`, and `src/simsopt_jax_adapters/geo/surface_objectives.py`: out of scope for the current clean signoff slice unless later validation proves a clean-source failure requires them.

## Validation Evidence Recorded During Classification

- `../simsopt-jax/.miniforge/bin/python3.13 scripts/jax_gpu_failed_stale_tests_signoff.py --dry-run --repo . --python-bin ../simsopt-jax/.miniforge/bin/python3.13 --results-dir /tmp/clean-signoff-dry-run-after-final-review-pass` passed. Its summary records 130 requested/present integration paths, 0 missing paths, 8 focused deselectors, 0 current/new/stale hits, and `failures: []`.
- `../simsopt-jax/.miniforge/bin/python3.13 -m pytest tests/jax/core/test_dipole_field_item24.py::test_dipole_field_Bn_on_axis_noncartesian_matches_cpp -q` failed on clean and donor before the clean-only correction, then passed after restoring the non-finite SIMD convention.
- `../simsopt-jax/.miniforge/bin/python3.13 -m pytest tests/jax/core/test_dipole_field_item24.py -q` initially exposed two stale assertions that expected native `simsoptpp.dipole_field_Bn` to raise for bad coordinate flags and unitnormal shape mismatches; the current native extension returns arrays for those cases. After narrowing those tests to the JAX validation contract, the file passed: `23 passed, 1 warning`.
- `../simsopt-jax/.miniforge/bin/python3.13 -m pytest tests/test_gpu_transfer_guard_harness.py -q` passed: `17 passed`.
- `../simsopt-jax/.miniforge/bin/python3.13 -m pytest tests/jax/core/test_pm_optimization_jax_item25.py::TestPMKernelHelpers tests/solve/test_permanent_magnet_optimization_jax_item28.py::test_projection_and_prox_helpers_match_cpu_oracles tests/solve/test_permanent_magnet_optimization_jax_item28.py::test_projection_helper_matches_cpu_oracle_for_edge_mmax tests/solve/test_permanent_magnet_optimization_jax_item28.py::test_zero_mmax_helpers_match_cpu_without_nan -q` passed: `18 passed, 8 warnings`.
- `../simsopt-jax/.miniforge/bin/python3.13 -m pytest tests/jax/core/test_pm_optimization_jax_item25.py::TestPMKernelHelpers -q` passed: `15 passed`.
- `../simsopt-jax/.miniforge/bin/python3.13 -m pytest tests/solve/test_permanent_magnet_optimization_jax_item28.py -q` passed: `48 passed, 9 warnings`.
- `../simsopt-jax/.miniforge/bin/python3.13 -m pytest tests/test_jax_import_smoke.py::test_transfer_guard_disallow_allows_surface_xyztensorfourier_gamma_from_dofs -q` skipped locally because no GPU was available.
- `../simsopt-jax/.miniforge/bin/python3.13 -m pytest tests/geo/test_surface_fourier_jax.py::test_split_flat_to_xyzc_keeps_nan_blocks_isolated -q` passed after replacing the donor weighted split with `jnp.split`.
- `../simsopt-jax/.miniforge/bin/python3.13 -m pytest tests/geo/test_surface_fourier_jax.py -q` passed after the surface split fix: `147 passed`.
- `../simsopt-jax/.miniforge/bin/python3.13 -m ruff check scripts/jax_gpu_failed_stale_tests_signoff.py src/simsopt_jax/core/dipole_field.py src/simsopt_jax/core/pm_optimization.py src/simsopt_jax/core/surface_fourier_kernels.py src/simsopt_jax/solve/permanent_magnet.py tests/conftest.py tests/geo/test_surface_fourier_jax.py tests/jax/core/test_dipole_field_item24.py tests/jax/core/test_pm_optimization_jax_item25.py tests/solve/test_permanent_magnet_optimization_jax_item28.py tests/test_gpu_transfer_guard_harness.py` passed after the final review delta.
- `git diff --check` passed after the final review delta.
- The Crucible reviewer loop reached strict PASS for this code/test slice. Initial findings on the donor weighted surface split and tracked/untracked doc wording were fixed, the delta reviewers returned PASS, and all six reviewer agents were closed.
