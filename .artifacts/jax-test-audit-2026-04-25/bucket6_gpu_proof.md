# Bucket 6 audit: GPU production proof + Runpod continuation runners

Branch: `gpu-purity-stage2-20260405`. Date: 2026-04-25. Auditor: Claude Opus 4.7.

Files in scope:
- `/Users/suhjungdae/code/columbia/simsopt-jax/tests/test_hf_production_gpu_proof.py`
- `/Users/suhjungdae/code/columbia/simsopt-jax/tests/test_runpod_single_stage_continuation.py`
- `/Users/suhjungdae/code/columbia/simsopt-jax/tests/subprocess/hf_production_gpu_fake_runner.py`

Context files:
- `/Users/suhjungdae/code/columbia/simsopt-jax/benchmarks/hf_jobs/launch_production_gpu_proof.py`
- `/Users/suhjungdae/code/columbia/simsopt-jax/benchmarks/hf_jobs/run_production_gpu_proof.sh`
- `/Users/suhjungdae/code/columbia/simsopt-jax/benchmarks/hf_jobs/bootstrap_runtime.sh`
- `/Users/suhjungdae/code/columbia/simsopt-jax/benchmarks/stage2_e2e_comparison.py`
- `/Users/suhjungdae/code/columbia/simsopt-jax/benchmarks/single_stage_init_parity.py`
- `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/backend/runtime.py`

---

## Executive summary

The branch's stated purpose is to "PROVE on a real GPU runtime that JAX is using cuda, that forward AND backward passes execute device-side, and that numerical results match the CPU oracle to ladder precision." None of the 31 tests in `tests/test_hf_production_gpu_proof.py` verify any of that. Every test in the proof bucket uses a fake runner (`tests/subprocess/hf_production_gpu_fake_runner.py`) that emits `{"passed": True, "elapsed_s": 1.0, "failures": []}` blindly, without ever importing JAX, touching CUDA, or computing anything. The shell driver (`run_production_gpu_proof.sh`) reads only `passed`/`failures`/`elapsed_s` from those payloads — there is no schema field for `default_backend`, `jax_devices`, `gpu_buffer_residency`, or `xla_compile_target` in the proof bundle, so even on a real H200 the bundle would not surface whether the GPU actually executed the workload. The `bootstrap_runtime.sh` only validates `jax.__version__`, never `jax.default_backend()`. The Stage 2 probe `stage2_e2e_comparison.py` itself omits `require_requested_platform_runtime`, so a CPU-fallback Stage 2 run would be reported as a passing GPU proof. Every test in the bucket is a launcher/bash plumbing test masquerading as a GPU proof test, and there is no `pytest.skip`/`xfail` to abuse — but there is also no real-GPU lane the suite can ever exercise.

---

## 1. Per-file summary

| File | Total tests | Tautological | Loose tol | Weak | Meaningless | Fake-runner contamination | Skip/xfail abuse | Well-tightened | Priority |
|------|------------:|-------------:|----------:|-----:|------------:|--------------------------:|-----------------:|---------------:|----------|
| `tests/test_hf_production_gpu_proof.py` | 31 | 0 | 0 | 14 | 6 | 11 (every shell-runner test) | 0 | 0 | **P0** |
| `tests/test_runpod_single_stage_continuation.py` | 22 | 0 | 0 | 18 | 4 | 0 (does not exercise GPU) | 0 | 0 | **P1** |
| `tests/subprocess/hf_production_gpu_fake_runner.py` | 0 (helper) | n/a | n/a | n/a | n/a | n/a | n/a | n/a | helper |

Notes:
- "Weak" rows count tests whose only assertion is on argv plumbing, JSON keys, return codes, env-var preservation, or stdout substrings.
- "Fake-runner contamination" counts tests that drive the proof shell with the fake runner substituted in for the real probe scripts. The label applies because the test asserts behavior on the assumption that this fake bundle equals a "real proof".
- "Meaningless" rows include tests of substring matches in dry-run stdout, help-text smoke, ad-hoc-bootstrap-mode rejection, hf-CLI lookup, and seed contract preflight on a synthetic remote git repo (verifies plumbing only, not the proof contract).
- The fake runner helper itself has no tests; it is a stand-in that lets the proof shell complete without touching JAX.

---

## 2. Top issues

| # | File:line | Test name | Classification | Quote | Tightening recommendation |
|---|-----------|-----------|----------------|-------|---------------------------|
| 1 | `tests/test_hf_production_gpu_proof.py:113` | `test_run_production_gpu_proof_continues_after_missing_payload` | **FAKE-RUNNER CONTAMINATION** + WEAK | `assert (results_dir / "single_stage_cold.json").is_file()` and `assert (results_dir / "single_stage_warm.json").is_file()` | The test substitutes `FAKE_PROOF_SCRIPT` for the real probes via `_copy_executable(FAKE_PROOF_SCRIPT, benchmarks_dir / "stage2_e2e_comparison.py")` (lines 71–75) and then asserts the bundle has files. There is no SIMSOPT_FAKE_GPU sandbox flag — this is a CI smoke for the `run_probe` / heartbeat / payload aggregator, not a GPU proof. Move under `tests/subprocess/test_run_production_gpu_proof_shell.py` and rename so the file's name does not promise "GPU proof". |
| 2 | `tests/test_hf_production_gpu_proof.py:152` | `test_run_production_gpu_proof_survives_corrupt_payload` | FAKE-RUNNER CONTAMINATION + WEAK | `assert "corrupt payload" in completed.stdout` and `assert '"corrupt_payload": true' in completed.stdout` | Stdout-substring assertion. As above — covers payload aggregator JSON shape, not proof. Same renaming and same gating recommendation. |
| 3 | `tests/test_hf_production_gpu_proof.py:183` | `test_run_production_gpu_proof_adds_optional_repro_rung` | FAKE-RUNNER CONTAMINATION | `assert "--geometry-rel-tol" in repro_calls[0]["argv"]` | Asserts the shell propagated argv, then accepts the fake's `passed:true` as success (`assert completed.returncode == 0`). Fine as a plumbing test, but has no business sitting in a file named `test_hf_production_gpu_proof.py`. |
| 4 | `tests/test_hf_production_gpu_proof.py:381` | `test_run_production_gpu_proof_preserves_ld_library_path` | FAKE-RUNNER CONTAMINATION + WEAK | `assert {record["cuda_library_mode"] for record in call_records} == {"bundled"}` and `assert all("--xla_gpu_deterministic_ops=true" in str(record["xla_flags"]).split() for record in call_records)` | Verifies the fake runner saw the right env vars. Does NOT verify that XLA, when actually initialized, honored that flag. The XLA determinism field on `BackendPolicy` is itself reporting/acceptance metadata (see `runtime.py:91-94`). For a real proof, the bundle must include the result of `jax.config.read("jax_xla_flags")` and the active `default_backend()`. |
| 5 | `tests/test_hf_production_gpu_proof.py:423` | `test_run_production_gpu_proof_uses_repo_artifact_compilation_cache_by_default` | FAKE-RUNNER CONTAMINATION + WEAK | `assert {record["jax_compilation_cache_dir"] for record in call_records} == {str(expected_cache_dir)}` and `assert expected_cache_dir.is_dir()` | Tests env-var threading; no assertion that JAX actually wrote into that cache. Acceptable as a plumbing test, mis-located. |
| 6 | `tests/test_hf_production_gpu_proof.py:587` | `test_launch_production_gpu_proof_help_works_without_site_packages` | MEANINGLESS | `assert "Launch the production GPU proof on Hugging Face Jobs." in completed.stdout` | Help-text substring; this is a docstring contract test, not a proof test. |
| 7 | `tests/test_hf_production_gpu_proof.py:599` | `test_resolve_hf_cli_requires_hf_on_path` | WEAK | `with pytest.raises(RuntimeError, match="Could not find the Hugging Face CLI")` | Tests a `RuntimeError` message string. Useful for the launcher, irrelevant to the proof contract. |
| 8 | `tests/test_hf_production_gpu_proof.py:655` | `test_launch_production_gpu_proof_dry_run_omits_smoke_geometry_override` | WEAK | `assert '"effective_geometry_rel_tol": null' in completed.stdout` | Substring assertion on JSON encoded as text in dry-run stdout. The field is plumbed through `build_stage2_hf_plan`, not validated against any GPU-side measurement. |
| 9 | `tests/test_hf_production_gpu_proof.py:680` | same test | WEAK | `assert 'export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_deterministic_ops=true"' in completed.stdout` | Verifies the launcher emits a shell `export` line. Does not verify the flag is actually honored by the running XLA. |
| 10 | `tests/test_hf_production_gpu_proof.py:687-693` | same test | WEAK | `assert 'SIMSOPT_HF_JOB_JAX_GPU_WHEEL_SPEC="jax[cuda12]==0.9.2"' in completed.stdout` | Hard-coded substring; would pass even if `jax[cuda12]` cannot find a CUDA wheel for the running platform. |
| 11 | `tests/test_hf_production_gpu_proof.py:749` | `test_launch_production_gpu_proof_requires_explicit_image_or_env` | WEAK | `assert "requires a prebuilt image via SIMSOPT_HF_GPU_IMAGE or --image" in completed.stderr` | Plumbing for an `argparse` SystemExit message. |
| 12 | `tests/test_hf_production_gpu_proof.py:806` | `test_launch_production_gpu_proof_rejects_ad_hoc_always_bootstrap` | WEAK | `assert "invalid choice: 'always'" in completed.stderr` | argparse choices test; not specific to GPU proof. |
| 13 | `tests/test_hf_production_gpu_proof.py:871` | `test_launch_production_gpu_proof_rejects_remote_sha_not_on_repo_ref` | WEAK | `assert "is not present on" in completed.stderr or "is not reachable from repo ref" in completed.stderr` | Synthetic git remote test. Useful, but a launcher concern. |
| 14 | `tests/test_runpod_single_stage_continuation.py:374` | `test_build_remote_execution_script_threads_thresholds_and_paths` | WEAK | `self.assertIn('apt-get install -y --no-install-recommends "cuda-toolkit-${required_release//./-}"', script)` | Asserts a shell-substring of the generated remote script. Real CUDA toolkit installation, GPU detection, and JAX import success are NOT exercised. |
| 15 | `tests/test_runpod_single_stage_continuation.py:381-385` | same test | WEAK | `self.assertIn('python -m pip install -e ".[JAX_GPU,dev]"', script)` and `self.assertIn('expected = "0.9.2"', script)` | String matches on generated shell payload. The remote script's actual exit status, gradient accuracy, or `default_backend` is never inspected. |
| 16 | `tests/test_runpod_single_stage_continuation.py:1044` | `test_fetch_results_marks_profile_fetch_optional` | WEAK | Mocks `scp_from_remote` and asserts `report.optional_failures == ("profiling_report:...",)` | Tests mock-call bookkeeping; asserts the profiling report is "optional" but never asserts that a successfully-fetched report contains evidence the GPU ran. |
| 17 | `tests/test_runpod_single_stage_continuation.py:946` | `test_fetch_results_salvages_later_artifacts_after_summary_fetch_failure` | WEAK | Asserts `report.required_failures == ("summary:...",)` after mocking `scp_from_remote` to raise on the summary | Bookkeeping test; the proof would be salvaged into a state where summary is missing — which means an attacker (or a bug) could mark a run "salvaged" with no actual GPU-side result. |
| 18 | `tests/test_hf_production_gpu_proof.py:381` | `test_run_production_gpu_proof_preserves_ld_library_path` | FAKE-RUNNER CONTAMINATION | The test sets `"LD_LIBRARY_PATH": "/cuda/lib:/driver/lib"` and asserts the fake runner saw it. | The point of an LD_LIBRARY_PATH proof is to verify the runtime can `dlopen` the bundled CUDA libraries. The fake runner is a no-op `print` script — it cannot fail to dlopen anything. Move the assertion to a dynamic loader smoke that imports `jaxlib.cuda_versions`. |
| 19 | (entire file) `tests/test_hf_production_gpu_proof.py` | n/a | structural | The file does not contain a single `pytest.mark.skipif(jax.devices(...))`-style guard, no real-GPU lane, and no `SIMSOPT_FAKE_GPU` env-var sandbox separator. Currently, every test runs against the fake runner unconditionally. |
| 20 | (file) `benchmarks/stage2_e2e_comparison.py:46-59` | n/a | structural | The Stage 2 probe calls `apply_requested_platform` and `require_x64_runtime` but does NOT call `require_requested_platform_runtime` like `single_stage_init_parity.py:73-77` does. A real Stage 2 run on a CUDA-less host would silently fall back to CPU and report `passed: True`. Add the runtime guard. |

---

## 3. Missing coverage

A "production GPU proof" should at minimum verify all of the following on real hardware. None of these are covered today.

- **Active runtime backend**: assert `jax.default_backend() == "gpu"` (or `"cuda"`) inside the actual probe processes (Stage 2 + single-stage). `single_stage_init_parity.py` does this via `require_requested_platform_runtime`; `stage2_e2e_comparison.py` does not.
- **Device residency of hot-loop arrays**: assert that gamma, normal, target field, coil tangents, and the gradient buffer all live on `cuda:N`. The proof bundle today contains zero device metadata.
- **Forward and backward both ran on device**: instrument the JIT-compiled forward and backward to record the executable target (`exe.runtime_executable.execution_count_by_device`) and dump it into the bundle. No such field exists in the bundle schema.
- **No host syncs in the optimizer hot loop**: capture a `jax.transfer_guard("disallow")` audit log; failure to use disallow on the proof lane would indicate host fallback.
- **Numerical parity to ladder precision**: compare GPU value/gradient to a CPU oracle at `rtol=1e-8 atol=1e-10` (first derivative) and `rtol=1e-6 atol=1e-8` (second derivative). The probe scripts compute parity, but the test doesn't assert the parity tolerances against the ladder contract — the test only asserts `payload.passed`.
- **Determinism flag was set BEFORE jax import**: the bundle should include `xla_flags_seen_at_jax_init` captured by re-reading the env from inside the probe AFTER `import jax`. Currently, the test only verifies the env-var was passed in (no proof XLA actually picked it up).
- **CUDA wheel resolved to compatible cubin**: known issue per project memory 2026-04-20 (Runpod jaxlib cubin v12.9 vs system nvlink). The proof bundle should record `jaxlib.cuda_versions` and the resolved CUDA driver/runtime versions.
- **GPU ran the same numerical answer on cold and warm starts**: the warm payload should reference the cold payload's solution and assert end-to-end determinism within `gpu_reduction_order_max_ulp` (10 ulp, `runtime.py:146`). Today no cross-payload reproducibility check exists.
- **Real-GPU sandbox separation**: there is no `SIMSOPT_FAKE_GPU=1` sandbox marker. Without it, the suite cannot tell whether it ran the fake CI plumbing or the real probe.
- **Refusal to accept fake bundles**: the test must refuse to mark a payload as a "proof" if the payload was emitted by `hf_production_gpu_fake_runner.py`. Today the bundle has no provenance field, so this is impossible to enforce.
- **Bootstrap-time GPU smoke**: `bootstrap_runtime.sh:30-54` only validates `jax.__version__`. It must also assert `jax.default_backend() == "gpu"` after `import jax` and dump `jax.devices()` to a `bootstrap_jax_smoke.json`. The existing `jax_cuda_smoke.json` artifact (e.g. in `.artifacts/runpod_prod_signoff/h200-prod-signoff-20260425T012000Z/`) demonstrates the pattern — but the proof shell never invokes that smoke.
- **Runpod toolkit smoke**: `tests/test_runpod_single_stage_continuation.py` only inspects the generated shell as text. There is no end-to-end test that runs the script in a containerized GPU sandbox or even validates the toolkit-installation function reaches `dpkg --compare-versions` correctly under multiple host versions.

---

## 4. Tightening playbook (P0)

1. **Rename `tests/test_hf_production_gpu_proof.py` -> `tests/subprocess/test_hf_production_gpu_proof_shell.py`**. The current 31 tests cover the bash driver and launcher argv plumbing. They are not GPU proof tests. Renaming removes the contractual lie and signals to readers (and to CI dashboards) that this file does not exercise CUDA.

2. **Introduce a real `tests/test_hf_production_gpu_proof_real.py` lane gated by `SIMSOPT_FAKE_GPU=0` + `pytest.importorskip("jax")` + `jax.default_backend() == "gpu"`**. This lane runs the actual `stage2_e2e_comparison.py` and `single_stage_init_parity.py` against a CPU oracle and asserts the ladder tolerances (`rtol=1e-8, atol=1e-10` for first-derivative; `rtol=1e-6, atol=1e-8` for second-derivative). Refuse to skip when CUDA is missing — the test is `xfail strict=True` if a GPU was expected.

3. **Add explicit fake-vs-real provenance to the proof bundle**. Both the real probes and the fake runner must write a `bundle_provenance` field. Real: `{"runner": "stage2_e2e_comparison.py", "default_backend": jax.default_backend(), "devices": [str(d) for d in jax.devices()], "jaxlib_cuda_versions": jaxlib.cuda_versions._versions, "xla_flags_at_init": os.environ.get("XLA_FLAGS")}`. Fake: `{"runner": "hf_production_gpu_fake_runner.py", "fake": True}`. The shell aggregator must refuse to set OVERALL_RC=0 when ANY payload has `bundle_provenance.fake == True` unless `SIMSOPT_FAKE_GPU=1`.

4. **Add `require_requested_platform_runtime` to `benchmarks/stage2_e2e_comparison.py` (mirror `single_stage_init_parity.py:73-77`)**. Without this guard, the Stage 2 lane silently falls back to CPU on a CUDA-less host. This is a runtime hole, not just a test hole, but the test bucket inherits the consequence.

5. **Strengthen `bootstrap_runtime.sh:verify_runtime_versions` to also assert `jax.default_backend() == "gpu"` and emit a `bootstrap_jax_smoke.json` capturing `default_backend` and `devices`**. The existing `jax_cuda_smoke.json` artifact pattern shows what this should look like.

6. **Add an LD_LIBRARY_PATH dynamic-loader smoke**. Replace the fake-runner LD_LIBRARY_PATH echoback with a real `python -c "import jaxlib.cuda_versions; print(jaxlib.cuda_versions._versions)"` invocation gated under the real lane. Today's test passes even if `dlopen` would fail on the production image.

7. **Add a CPU-vs-GPU parity assertion in the bundle aggregator**. The current shell aggregator only emits `{"passed": payload.get("passed"), "elapsed_s": payload.get("elapsed_s"), "failures": payload.get("failures")}` (`run_production_gpu_proof.sh:271-285`). Extend the schema to require `cpu_oracle_value`, `gpu_value`, `value_rtol`, `gradient_rtol`, and reject the bundle when any rtol exceeds the parity-ladder contract.

8. **Wire a continuation/Runpod end-to-end smoke under `pytest.mark.runpod`**. Today `tests/test_runpod_single_stage_continuation.py` only inspects shell strings. Add (or move out of the audit bucket) an `@pytest.mark.runpod` integration test that boots a sandboxed container, runs `build_remote_execution_script(plan)`, and asserts the resulting JAX import reports `default_backend == "gpu"`.

---

## 5. Fake-vs-real audit table

For each test in `tests/test_hf_production_gpu_proof.py`: today's behavior on the fake-runner CI lane (default), today's behavior on real GPU (impossible — there is no real-GPU lane), and what the right behavior should be after tightening.

| Line | Test | Passes on fake today? | Passes on real GPU? | SHOULD pass on fake? | SHOULD pass on real GPU? |
|------|------|:---------------------:|:-------------------:|:--------------------:|:------------------------:|
| 85 | `test_run_production_gpu_proof_requires_single_stage_seed` | yes | yes (no seed -> exit 1) | yes | yes |
| 113 | `test_run_production_gpu_proof_continues_after_missing_payload` | yes (fake returns rc=3 for warm) | n/a (real does not have a `missing` mode) | yes, under SIMSOPT_FAKE_GPU=1 only | n/a — real lane would never simulate "missing payload"; this is a fault-injection test |
| 152 | `test_run_production_gpu_proof_survives_corrupt_payload` | yes (fake writes `{bad json`) | n/a | yes, under SIMSOPT_FAKE_GPU=1 only | n/a |
| 183 | `test_run_production_gpu_proof_adds_optional_repro_rung` | yes | argv-plumbing only | yes | n/a (this is a launcher concern) |
| 223 | `test_run_production_gpu_proof_omits_boozer_override_by_default` | yes | yes | yes | yes |
| 258 | `test_run_production_gpu_proof_threads_single_stage_seed_contract` | yes | yes | yes | yes |
| 307 | `test_run_production_gpu_proof_threads_single_stage_benchmark_mode` | yes | yes | yes | yes |
| 343 | `test_run_production_gpu_proof_threads_single_stage_success_filter_bypass` | yes | yes | yes | yes |
| 381 | `test_run_production_gpu_proof_preserves_ld_library_path` | yes (fake just echoes env) | n/a (would need dlopen smoke) | yes | yes (with real dlopen smoke replacing the fake echoback) |
| 423 | `test_run_production_gpu_proof_uses_repo_artifact_compilation_cache_by_default` | yes | n/a (cache behavior depends on real JIT) | yes | yes (with cache-population assertion) |
| 462 | `test_build_stage2_hf_plan_keeps_smoke_jobs_geometry_report_only` | yes (pure unit) | yes | yes | yes |
| 472 | `test_build_stage2_hf_plan_requires_long_run_for_explicit_geometry_repro` | yes | yes | yes | yes |
| 480 | `test_build_stage2_hf_plan_adds_repro_rung_for_long_run_override` | yes | yes | yes | yes |
| 494 | `test_build_stage2_hf_plan_reports_default_long_run_geometry_gate` | yes | yes | yes | yes |
| 587 | `test_launch_production_gpu_proof_help_works_without_site_packages` | yes | yes | yes | yes |
| 605 | `test_resolve_hf_cli_requires_hf_on_path` | yes (monkeypatched) | yes | yes | yes |
| 616 | `test_resolve_repo_defaults_prefers_current_branch_upstream_remote` | yes (monkeypatched) | yes | yes | yes |
| 645 | `test_resolve_default_repo_url_rejects_ambiguous_nonstandard_remotes` | yes | yes | yes | yes |
| 661 | `test_launch_production_gpu_proof_dry_run_omits_smoke_geometry_override` | yes (substr in stdout) | yes | yes | yes |
| 703 | `test_launch_production_gpu_proof_dry_run_threads_single_stage_benchmark_mode` | yes | yes | yes | yes |
| 729 | `test_launch_production_gpu_proof_dry_run_threads_single_stage_success_filter_bypass` | yes | yes | yes | yes |
| 749 | `test_launch_production_gpu_proof_requires_explicit_image_or_env` | yes | yes | yes | yes |
| 778 | `test_launch_production_gpu_proof_requires_single_stage_seed` | yes | yes | yes | yes |
| 806 | `test_launch_production_gpu_proof_rejects_ad_hoc_always_bootstrap` | yes | yes | yes | yes |
| 824 | `test_launch_production_gpu_proof_reports_default_long_run_geometry_gate` | yes | yes | yes | yes |
| 852 | `test_launch_production_gpu_proof_rejects_smoke_geometry_override` | yes | yes | yes | yes |
| 871 | `test_launch_production_gpu_proof_rejects_remote_sha_not_on_repo_ref` | yes | yes | yes | yes |
| 910 | `test_launch_production_gpu_proof_accepts_matching_remote_repo_ref_and_sha` | yes | yes | yes | yes |
| 940 | `test_launch_production_gpu_proof_rejects_host_absolute_seed_path` | yes | yes | yes | yes |
| 970 | `test_launch_production_gpu_proof_rejects_seed_path_missing_from_target_sha` | yes | yes | yes | yes |
| 1002 | `test_launch_production_gpu_proof_allows_explicit_long_run_geometry_rung` | yes | yes | yes | yes |

Conclusion: **0 of 31 tests in `test_hf_production_gpu_proof.py` exercise GPU code paths**. The file's name is misleading — it is a launcher/shell-runner plumbing suite. None of these tests would catch a regression where the proof shell silently falls back to CPU on a real CUDA host. The renamed-and-split layout (P0 #1 + #2) is the correct fix.

For `tests/test_runpod_single_stage_continuation.py`, every test inspects either a generated shell-script string, a path resolution, or a mocked SCP call. The file is purely `LaunchPlan` / `build_remote_execution_script` plumbing. Same pattern as above; the file is correctly scoped if renamed to `test_runpod_single_stage_continuation_launcher.py` to remove the false promise.

---

## 6. Skip/xfail audit

There are zero `pytest.skip`, `pytest.skipif`, `pytest.xfail`, or conditional skips in the three audit files. Confirmed by:

```
grep -n "skip\|xfail\|skipif\|skip_unless" tests/test_hf_production_gpu_proof.py tests/test_runpod_single_stage_continuation.py tests/subprocess/hf_production_gpu_fake_runner.py
```

(Only hit: `tests/test_runpod_single_stage_continuation.py:391` where `'skipping env update."'` is an asserted shell-string substring, NOT a pytest skip.)

This means the suite is honest in one direction: **no test silently swallows a regression via skip-abuse**. But it is dishonest in the other direction: **it reports GREEN for plumbing tests on a file titled "production GPU proof", which is a stronger lie than `xfail` suppression.** The fix is at the framework level (rename + add a real GPU lane), not at the per-test level.

When the real-GPU lane is added per P0 #2, any new `pytest.mark.skipif(jax.default_backend() != "gpu", reason=...)` MUST default to `strict=True` so a missing GPU on the proof CI is a red, not a skip.

---

## Cross-cutting findings

- **The Stage 2 probe `benchmarks/stage2_e2e_comparison.py:46-59` is missing the `require_requested_platform_runtime` guard** that `single_stage_init_parity.py:73-77` uses. This is a runtime contract gap that the tests inherit. Adding the call is a one-line fix and would catch silent CPU fallback on the Stage 2 lane.
- **The bundle schema is impoverished**. `passed`, `elapsed_s`, `failures`, `missing_payload`, `corrupt_payload` — that's it (`run_production_gpu_proof.sh:271-285`). Add `bundle_provenance`, `default_backend`, `devices`, `jaxlib_cuda_versions`, `xla_flags`, `cpu_oracle_value`, `gpu_value`, `value_rtol`, `gradient_rtol`.
- **The fake runner is too friendly**. `tests/subprocess/hf_production_gpu_fake_runner.py:50-55` writes `{"passed": True, ...}` unconditionally. It has only three modes: `ok`, `missing`, `corrupt`. There is no mode that simulates "GPU initialized but compute fell back to CPU" — exactly the regression a real GPU proof should catch.
- **`bootstrap_runtime.sh` does not verify the wheel actually loaded CUDA**. Per the project memory, the open issue (2026-04-20) is exactly that on Runpod the cubin v12.9 mismatches nvlink. The bootstrap currently passes when JAX/jaxlib are 0.9.2, regardless of whether the GPU backend will initialize. Add a `jax.default_backend()` assertion immediately after `import jax`.
- **The current test suite would have been GREEN during the 2026-04-20 Runpod incident**. None of the 31 tests would detect a jaxlib cubin mismatch, because the assertion surface stops at "argv was passed correctly" and "fake returner returned 0".
