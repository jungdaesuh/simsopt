# Single-Stage JAX GPU Parity Audit — Issue Checklist

**Date**: 2026-03-29
**Branch**: `jax-port` at `274fefbd`
**Scope**: All issues uncovered during code review and parity chain analysis

---

## 1. Bugs Fixed in This Session

- [x] **CPU warm-start `sdofs` contract mismatch**
  - File: `/Users/suhjungdae/code/columbia/simsopt-jax/examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:692-695`
  - `initialize_boozer_surface()` unconditionally passed `sdofs=` to `run_code()`, crashing native `BoozerSurface` with TypeError.
  - Fixed: `run_boozer_solve()` dispatcher — CPU omits `sdofs=`, JAX passes it.
  - Impact: **Unblocked the trusted CPU reference lane for single-stage parity.** Without this, every HF A100 single-stage probe crashed on the CPU side.

- [x] **No GPU platform verification**
  - Files: `/Users/suhjungdae/code/columbia/simsopt-jax/benchmarks/validation_ladder_common.py:130-149`, `/Users/suhjungdae/code/columbia/simsopt-jax/benchmarks/single_stage_init_parity.py:62-66`
  - JAX silently falls back to CPU when no GPU is available.
  - Fixed: `require_requested_platform_runtime()` raises RuntimeError on platform mismatch.
  - Impact: **Prevents false "GPU parity" claims.** A CPU-vs-CPU comparison labeled as GPU evidence would be misleading.

- [x] **Inconsistent `iterative_refinement` in adjoint solves**
  - File: `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/geo/surfaceobjectives_jax.py:60`
  - `_solve_boozer_adjoint()` defaulted to `iterative_refinement=False`, while the traceable path used `True`.
  - Fixed: enabled `iterative_refinement=True` in both paths.
  - Impact: **Eliminates ~1e-2 adjoint precision loss on ill-conditioned PLU systems.** The scipy-backed M5 path now matches the on-device path's numerical refinement.

- [x] **`float(current_value)` host round-trip in coil extraction**
  - File: `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/field/biotsavart_jax_backend.py:844`
  - Non-JAX-native fallback path forced current scalars through Python `float()`.
  - Fixed: pass `current_value` directly; `group_coil_data` already calls `jnp.asarray`.
  - Impact: **Removes unnecessary GPU-to-host-to-GPU transfer per coil** in the fallback path. Minor perf improvement.

- [x] **`surface_self_intersection_check_available` AttributeError**
  - File: `/Users/suhjungdae/code/columbia/simsopt-jax/examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:832`
  - `surface_module.LineString` raises AttributeError when shapely isn't installed.
  - Fixed: `getattr(surface_module, "LineString", None)`.
  - Impact: **Prevents crash in HF container environments** where shapely is not available.

## 2. Bugs Fixed in Prior Session (commit `03c27f44`)

- [x] **HF harness abort on failed Stage 2 probe**
  - File: `/Users/suhjungdae/code/columbia/simsopt-jax/benchmarks/hf_jobs/run_production_gpu_proof.sh:124-151`
  - `set -e` + bare `wait` killed the script before single-stage probes could run.
  - Fixed: `|| OVERALL_RC=1` accumulation and `if wait; then rc=0; else rc=$?; fi`.
  - Impact: **Single-stage probes now always run** regardless of Stage 2 outcome. Previously, any Stage 2 failure meant zero single-stage evidence from the entire A100 job.

- [x] **HF harness unconditional geometry repro rung**
  - Files: `/Users/suhjungdae/code/columbia/simsopt-jax/benchmarks/hf_jobs/run_production_gpu_proof.sh:62-84`, `/Users/suhjungdae/code/columbia/simsopt-jax/benchmarks/validation_ladder_contract.py:91-123`
  - Default `GEOMETRY_REL_TOL="5e-6"` forced a geometry gate on 20-iter smoke runs.
  - Fixed: centralized in `build_stage2_hf_plan()`, smoke runs use `report-only` policy.
  - Impact: **Smoke runs no longer fail on expected geometry drift.** The 3.65e-3 geometry drift at 20 iterations is normal; gating on 5e-6 was a false failure.

- [x] **Launcher `--geometry-rel-tol` always passed**
  - File: `/Users/suhjungdae/code/columbia/simsopt-jax/benchmarks/hf_jobs/launch_production_gpu_proof.py:115-122`
  - Default `"5e-6"` string was always emitted in the job command.
  - Fixed: default is `None`, flag conditionally emitted via `_build_optional_stage2_geometry_flag()`.
  - Impact: **Launcher and shell harness now agree on geometry policy.** Eliminates the smoke-vs-repro desync that caused the original A100 failures.

- [x] **Launcher eager git calls at parse time**
  - File: `/Users/suhjungdae/code/columbia/simsopt-jax/benchmarks/hf_jobs/launch_production_gpu_proof.py:86-112`
  - `--repo-url` default called `_git_output("remote", "get-url", "fork")` at import.
  - Fixed: deferred to `_resolve_repo_defaults()` with multi-remote fallback chain.
  - Impact: **Launcher no longer crashes when "fork" remote doesn't exist.** Works in any git environment.

- [x] **Missing/corrupt payload handling**
  - File: `/Users/suhjungdae/code/columbia/simsopt-jax/benchmarks/hf_jobs/run_production_gpu_proof.sh:139-149`
  - Shell harness didn't check if output JSON existed before parsing.
  - Fixed: explicit file-exists check, corrupt-JSON detection, per-probe payload summary.
  - Impact: **Diagnostic clarity.** Operators can now distinguish "probe crashed" from "probe produced bad data" from "probe passed" in HF job logs.

---

## 3. Parity Tolerance Gaps

Precision targets below assume `newton_tol` tightened to 1e-12 (costs ~5 extra Newton iterations). Native C++ `test_convergence_cpp_and_notcpp_same` proves Python==C++ to atol=1e-11 with `tol=1e-10`. JAX implements the same float64 math — the achievable floor is the same.

### Tier 3 End-to-End Probe

- [ ] **`final_iota_abs_tol` = 1e-3 (3 sig figs)**
  - File: `/Users/suhjungdae/code/columbia/simsopt-jax/benchmarks/validation_ladder_contract.py:24`
  - Consumed by: `/Users/suhjungdae/code/columbia/simsopt-jax/benchmarks/single_stage_init_parity.py:63`
  - Native C++ proves Python==C++ to 1e-11 (`/Users/suhjungdae/code/columbia/simsopt/tests/geo/test_boozersurface.py:392`).
  - Target: **1e-10**. Bottleneck: inner solver convergence — tighten `newton_tol` to 1e-12.
  - Impact: **11 sig fig iota agreement proves JAX and C++ converge to the same stationary point.** Users switching backends will see identical published results.

- [ ] **`final_volume_rel_tol` = 1e-6**
  - File: `/Users/suhjungdae/code/columbia/simsopt-jax/benchmarks/validation_ladder_contract.py:25`
  - M1 surface parity is ~1e-13 (`/Users/suhjungdae/code/columbia/simsopt-jax/tests/geo/test_surface_fourier_jax.py`).
  - Target: **1e-10**.
  - Impact: **Volume is a simple integral of converged surface geometry.** At 1e-10 the volume agreement is limited only by surface DOF agreement, not by the tolerance gate itself.

- [ ] **`field_error_rel_tol` = 1e-4**
  - File: `/Users/suhjungdae/code/columbia/simsopt-jax/benchmarks/validation_ladder_contract.py:26`
  - M1 B-field parity is ~1e-10 (`/Users/suhjungdae/code/columbia/simsopt-jax/tests/field/test_biotsavart_jax.py`).
  - Target: **1e-8**.
  - Impact: **Field error is the primary coil quality metric.** 1e-8 proves B-field evaluation on the converged surface matches to float64 composition noise.

- [ ] **`surface_geometry_rel_tol` = 1e-5**
  - File: `/Users/suhjungdae/code/columbia/simsopt-jax/benchmarks/validation_ladder_contract.py:27`
  - Target: **1e-9** (with scipy backend on both sides and tightened newton_tol).
  - Impact: **Proves the converged plasma boundary is the same surface to 9 digits**, not just a nearby one. Matches the native C++ convergence test precision.

### Micro-Parity Tests

- [ ] **LS `run_code` iota parity = 1e-6**
  - File: `/Users/suhjungdae/code/columbia/simsopt-jax/tests/integration/test_single_stage_jax.py` (TestRunCodeLSParity)
  - Native `test_convergence_cpp_and_notcpp_same` achieves atol=1e-11 (`/Users/suhjungdae/code/columbia/simsopt/tests/geo/test_boozersurface.py:392`).
  - Target: **1e-11** (match native precision with tightened solver tol).
  - Impact: **The primary regression gate for the LS pipeline.** 1e-11 is the proven C++ floor — matching it eliminates the weakest link in the parity chain.

- [ ] **Exact Newton iota/G parity = 1e-5**
  - File: `/Users/suhjungdae/code/columbia/simsopt-jax/tests/integration/test_single_stage_jax.py` (TestExactSolveCPUJAXParity)
  - Target: **1e-10**.
  - Impact: **Newton converges quadratically to the same root.** With tightened tol, both sides reach the same fixed point to float64 gradient noise.

- [ ] **`IotasJAX.J()` = finiteness only (no CPU comparison)**
  - File: `/Users/suhjungdae/code/columbia/simsopt-jax/tests/integration/test_single_stage_jax.py` (TestIotasValue)
  - Target: add `assert_allclose(iotas_jax.J(), iotas_cpu.J(), rtol=1e-12)`.
  - Impact: **Both read `res["iota"]` which is a Python float from the same solver.** Should agree to near-machine-precision. Currently passes even if values are completely different.

- [ ] **`NonQuasiSymmetricRatioJAX.J()` = finiteness only**
  - File: `/Users/suhjungdae/code/columbia/simsopt-jax/tests/integration/test_single_stage_jax.py` (TestNonQSRatioValue)
  - Target: add CPU parity assertion with rtol=1e-8.
  - Impact: **QS ratio involves an auxiliary-grid integral — more accumulated ops, but still deterministic float64.** 1e-8 is achievable and catches algorithmic divergence.

- [ ] **`BoozerResidualJAX.J()` = magnitude bound only**
  - File: `/Users/suhjungdae/code/columbia/simsopt-jax/tests/integration/test_single_stage_jax.py` (TestBoozerResidualValue)
  - Target: add `assert_allclose(br_jax.J(), br_cpu.J(), rtol=1e-12)`.
  - Impact: **Same penalty formula, same inputs, same surface state.** Should agree to near-machine-precision. A wrong value means the inner solver optimizes the wrong objective.

- [ ] **IFT adjoint FD tolerance = rel<1e-3**
  - File: `/Users/suhjungdae/code/columbia/simsopt-jax/tests/integration/test_single_stage_jax.py` (TestIotasJAXResolveFD, TestNonQSRatioJAXResolveFD)
  - Native `IotasTests` uses eps=2^(-13 to -19) with atol=2e-8 (`/Users/suhjungdae/code/columbia/simsopt/tests/geo/test_surface_objectives.py:328`).
  - Target: **atol=2e-8** (match native Taylor test protocol with same eps schedule).
  - Impact: **Proves the IFT adjoint gradient is correct to the same precision as the native C++ implementation.** 2e-8 is the native floor; matching it eliminates the FD precision gap.

### M1/M2 Component Parity

- [ ] **M1 B-field vs C++ = rtol=1e-10**
  - File: `/Users/suhjungdae/code/columbia/simsopt-jax/tests/field/test_biotsavart_jax.py` (TestBiotSavartJaxCppParity)
  - Target: **rtol=1e-13**.
  - Impact: **Near-machine-precision B-field agreement.** The native `cpp_notcpp` penalty test achieves atol=1e-13 for the value. JAX Biot-Savart should match since it's the same summation in float64.

- [ ] **M2 SquaredFlux gradient vs C++ = rtol=1e-9**
  - File: `/Users/suhjungdae/code/columbia/simsopt-jax/tests/integration/test_stage2_jax.py`
  - Target: **rtol=1e-11**.
  - Impact: **Matches native penalty gradient precision (atol=1e-11).** Proves the VJP chain rule through coil geometry agrees with C++ hand-coded derivatives to the composition noise floor.

---

## 4. Missing Parity Tests

### M5 Value Parity (JAX wrapper vs CPU wrapper)

- [ ] **`IotasJAX.J()` vs `Iotas.J()`**
  - Where to add: `/Users/suhjungdae/code/columbia/simsopt-jax/tests/integration/test_single_stage_jax.py` (new test in TestIotasValue)
  - Reference: `/Users/suhjungdae/code/columbia/simsopt/tests/geo/test_surface_objectives.py:291` (native Iotas tests)
  - Expected: rtol=1e-10 (both read `res["iota"]` as Python float).
  - Impact: **Closes the "finiteness-only" gap.** Proves the JAX iota wrapper returns the same number as the CPU wrapper on the same solved surface.

- [ ] **`NonQuasiSymmetricRatioJAX.J()` vs `NonQuasiSymmetricRatio.J()`**
  - Where to add: `/Users/suhjungdae/code/columbia/simsopt-jax/tests/integration/test_single_stage_jax.py` (new test in TestNonQSRatioValue)
  - Reference: `/Users/suhjungdae/code/columbia/simsopt/tests/geo/test_surface_objectives.py:332`
  - Expected: rtol=1e-6 (QS ratio involves more accumulated ops on auxiliary grid).
  - Impact: **The QS ratio is what users actually minimize in single-stage.** Without value parity, JAX could optimize a different objective.

- [ ] **`BoozerResidualJAX.J()` vs `BoozerResidual.J()`**
  - Where to add: `/Users/suhjungdae/code/columbia/simsopt-jax/tests/integration/test_single_stage_jax.py` (new test in TestBoozerResidualValue)
  - Reference: `/Users/suhjungdae/code/columbia/simsopt/tests/geo/test_surface_objectives.py:372`
  - Impact: **Proves the penalty objective that drives the inner solve is identical across backends.**

### M5 Gradient Parity (JAX IFT vs CPU IFT)

- [ ] **`IotasJAX.dJ()` vs `Iotas.dJ()`**
  - Where to add: `/Users/suhjungdae/code/columbia/simsopt-jax/tests/integration/test_single_stage_jax.py`
  - Reference: `/Users/suhjungdae/code/columbia/simsopt/tests/geo/test_surface_objectives.py:310-328`
  - Impact: **Direct gradient comparison bypasses FD limitations.** If both implementations are correct, they should agree to ~1e-10. This is stronger than FD validation at 1e-3.

- [ ] **`NonQuasiSymmetricRatioJAX.dJ()` vs `NonQuasiSymmetricRatio.dJ()`**
  - Where to add: same file
  - Reference: `/Users/suhjungdae/code/columbia/simsopt/tests/geo/test_surface_objectives.py:351-369`
  - Impact: Same.

- [ ] **`BoozerResidualJAX.dJ()` vs `BoozerResidual.dJ()`**
  - Where to add: same file
  - Reference: `/Users/suhjungdae/code/columbia/simsopt/tests/geo/test_surface_objectives.py:383`
  - Impact: Same.

### Boozer Penalty C++ Parity (mirror Columbia fork `cpp_notcpp`)

**IMPORTANT**: The `test_boozer_penalty_constraints_cpp_notcpp` and `test_convergence_cpp_and_notcpp_same` tests exist only in the **Columbia fork** (`/Users/suhjungdae/code/columbia/simsopt/tests/geo/test_boozersurface.py:385-516`), NOT in the upstream `hiddenSymmetries/simsopt` repo (`/Users/suhjungdae/code/opensource/simsopt/tests/geo/test_boozersurface.py`). The upstream removed the non-vectorized `boozer_penalty_constraints()` method entirely — only the C++-backed `boozer_penalty_constraints_vectorized()` remains. The Columbia fork retained both methods and the comparison test. Precision claims referencing the `cpp_notcpp` test are valid only against the Columbia fork baseline.

- [ ] **JAX `_boozer_penalty_objective` vs C++ `boozer_penalty_constraints_vectorized` — value**
  - JAX impl: `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/geo/boozersurface_jax.py:148-212`
  - C++ impl: `/Users/suhjungdae/code/opensource/simsopt/src/simsopt/geo/boozersurface.py:206` (only vectorized version exists upstream)
  - Columbia fork test: `/Users/suhjungdae/code/columbia/simsopt/tests/geo/test_boozersurface.py:447-516` (atol=1e-13)
  - Impact: **Proves JAX penalty formula matches C++ to near-machine-precision.** The Columbia fork test covers 8 grid configs x 2 surface types x 2 stellsym modes. Mirroring this for JAX would be the single strongest parity evidence.

- [ ] **JAX penalty gradient vs C++ gradient**
  - Columbia fork tolerance: atol=1e-11.
  - Upstream alternative: Taylor test in `/Users/suhjungdae/code/opensource/simsopt/tests/geo/test_boozersurface.py:45-108` proves gradient correctness via FD convergence (error < 0.55x per halving).
  - Impact: **The penalty gradient drives BFGS convergence.** Direct comparison is only possible against the Columbia fork. Against upstream, Taylor test convergence rate is the proxy.

- [ ] **JAX penalty Hessian vs C++ Hessian**
  - Columbia fork tolerance: atol=1e-10.
  - Upstream alternative: Taylor test in `/Users/suhjungdae/code/opensource/simsopt/tests/geo/test_boozersurface.py:58-145`.
  - Impact: **The Hessian drives Newton polish.** Hessian agreement proves the second-order structure matches.

### Solver Convergence Parity

- [ ] **JAX LBFGS+Newton final state vs C++ final state on NCSX**
  - Columbia fork test: `/Users/suhjungdae/code/columbia/simsopt/tests/geo/test_boozersurface.py:385-429` (atol=1e-11)
  - Upstream equivalent: `/Users/suhjungdae/code/opensource/simsopt/tests/geo/test_boozersurface.py:203-315` — tests convergence (residual < 1e-9, area constraint < 1e-9) but does NOT compare vectorized vs non-vectorized final state. The `vectorize` parameter was removed upstream.
  - Impact: **The Columbia fork gold standard (atol=1e-11) is the strictest test available.** Against upstream, the proxy is: both converge to residual < 1e-9 on the same NCSX/HSX/Giuliani fixtures.

### Real Fixture Solver Parity

- [ ] **LS `run_code` parity on real Columbia fixture (50+ coils, mpol=8, ntor=6)**
  - Current test: `/Users/suhjungdae/code/columbia/simsopt-jax/tests/integration/test_single_stage_jax.py` (TestRunCodeLSParity — 2 coils, mpol=ntor=2)
  - Fixture: `/Users/suhjungdae/code/columbia/simsopt-jax/benchmarks/single_stage_smoke_fixture.py`
  - Impact: **Toy fixture hides accumulated rounding across 50 coils and a 255x64 grid.** Need real-scale evidence.

- [ ] **Exact Newton parity on real fixture**
  - Same gap. Exact path uses full residual Jacobian which is much larger at production scale.
  - Impact: Same.

### GPU-Specific Tests

- [ ] **Any test exercising `SIMSOPT_JAX_PLATFORM=cuda`**
  - No file — doesn't exist.
  - Impact: **Zero empirical evidence that JAX-on-CUDA produces correct results.** All current parity is CPU-only. The first A100 run will be the first GPU evidence.

- [ ] **GPU vs CPU-JAX numerical agreement measurement**
  - No file — doesn't exist.
  - Impact: **Unknown whether CUDA reduction order, cuBLAS, or XLA GPU codegen introduces additional noise.** Need to measure before setting GPU-specific tolerances.

- [ ] **GPU memory behavior under production grid (nphi=255, ntheta=64, 50+ coils)**
  - No file — doesn't exist.
  - Impact: **OOM on GPU would crash the HF proof job silently.** No test exercises production-scale memory.

### Taylor Test Mirrors from Native Repo

- [ ] **`IotasJAX.dJ()` Taylor test**
  - Native: `/Users/suhjungdae/code/columbia/simsopt/tests/geo/test_surface_objectives.py:292-328` — eps=2^(-13 to -19), atol=2e-8, across exact/ls x Volume/ToroidalFlux x optimize_G x weight_inv_modB.
  - JAX: `/Users/suhjungdae/code/columbia/simsopt-jax/tests/integration/test_single_stage_jax.py` (TestIotasJAXResolveFD — rel<1e-3).
  - Impact: **Native achieves 5 orders tighter gradient validation.** Mirroring the native Taylor test structure would prove JAX gradients match C++ gradient accuracy.

- [ ] **`NonQuasiSymmetricRatioJAX.dJ()` Taylor test**
  - Native: `/Users/suhjungdae/code/columbia/simsopt/tests/geo/test_surface_objectives.py:332-369`
  - Impact: Same.

- [ ] **`BoozerResidualJAX.dJ()` Taylor test**
  - Native: `/Users/suhjungdae/code/columbia/simsopt/tests/geo/test_surface_objectives.py:372-400`
  - JAX: `/Users/suhjungdae/code/columbia/simsopt-jax/tests/integration/test_single_stage_jax.py` (TestBoozerResidualGradientFD — fixed-surface only)
  - Impact: **Native tests the full re-solve Taylor test. JAX only tests the fixed-surface direct term.** The adjoint term (the hard part) is untested at native precision.

---

## 5. Infrastructure / Harness Issues

### HF Production GPU Proof

- [ ] **First successful A100 single-stage run**
  - File: `/Users/suhjungdae/code/columbia/simsopt-jax/benchmarks/hf_jobs/run_production_gpu_proof.sh:186-214`
  - Structurally unblocked by `274fefbd`. Needs relaunch.
  - Impact: **No GPU parity evidence exists until this runs.** All claims about GPU correctness are theoretical.

- [ ] **No early GPU smoke check in HF job**
  - File: `/Users/suhjungdae/code/columbia/simsopt-jax/benchmarks/hf_jobs/run_production_gpu_proof.sh` (after line 88, before probes)
  - Add: `python -c "import jax; print(jax.devices())"` before the build step.
  - Impact: **Saves 10-15 min of wasted A100 time** when CUDA is misconfigured. Current flow: clone → build simsoptpp → discover no GPU.

- [ ] **Full simsoptpp C++ build unnecessary for JAX-only proof**
  - File: `/Users/suhjungdae/code/columbia/simsopt-jax/benchmarks/hf_jobs/run_production_gpu_proof.sh:321` (`pip install -v -e .`)
  - Impact: **5-15 min A100 time per job** spent compiling C++ (CMake, Boost, FFTW). The JAX proof only needs the Python package.

- [ ] **`unset LD_LIBRARY_PATH` may strip NVIDIA container paths**
  - File: `/Users/suhjungdae/code/columbia/simsopt-jax/benchmarks/hf_jobs/run_production_gpu_proof.sh:89`
  - Impact: **CUDA library discovery can break silently** in NVIDIA container runtimes that inject paths via `LD_LIBRARY_PATH`.

- [ ] **No `--depth 1` on HF clone**
  - File: `/Users/suhjungdae/code/columbia/simsopt-jax/benchmarks/hf_jobs/launch_production_gpu_proof.py:312-316`
  - Impact: **2-5 min clone overhead** downloading full git history on A100.

### Test Suite Gaps

- [ ] **`test_section6_public_lane_split.py` requires simsoptpp**
  - File: `/Users/suhjungdae/code/columbia/simsopt-jax/tests/integration/test_section6_public_lane_split.py:17-19` (`pytest.importorskip("simsoptpp")`)
  - Impact: **Claimed as "public lane" coverage but can't run in the public conda env.** Misleading CI coverage.

- [ ] **`test_benchmark_helpers.py` single-stage cases are monkeypatched**
  - File: `/Users/suhjungdae/code/columbia/simsopt-jax/tests/test_benchmark_helpers.py` (all `test_single_stage_init_case_*`)
  - Impact: **Never spawns real CPU lane.** Only tests the comparison/gating logic, not the actual parity computation.

- [ ] **`boozer_setup` fixture defaults to `optimizer_backend="ondevice"`**
  - File: `/Users/suhjungdae/code/columbia/simsopt-jax/tests/integration/test_single_stage_jax.py` (boozer_setup fixture)
  - Impact: **scipy backend absent from M5 integration tests.** No test exercises the scipy-backed IFT adjoint path end-to-end. The scipy path is the trusted reference.

### Issue Report

- [ ] **`jax_hf_a100_issue_report_2026-03-28.md` tracked in repo root**
  - File: `/Users/suhjungdae/code/columbia/simsopt-jax/jax_hf_a100_issue_report_2026-03-28.md`
  - Impact: **Contains absolute local paths.** Should be moved to `docs/` or an issue tracker.

---

## 6. Documentation / Contract Gaps

- [ ] **Tier 4 `full_resolve_fd_rel_tol` = 1e-2 rationale undocumented**
  - File: `/Users/suhjungdae/code/columbia/simsopt-jax/benchmarks/validation_ladder_contract.py:34`
  - 100x looser than `fixed_surface_fd_rel_tol` (1e-3 at line 33). No comment explains why.
  - Impact: **Reviewers can't assess whether 1e-2 is a principled choice or a workaround.** Documenting the rationale (branch-switching under perturbation) prevents future tightening attempts from breaking.

- [ ] **`run_code_traceable()` JAX scalar convention undocumented**
  - File: `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/geo/boozersurface_jax.py:1089-1103`
  - `result["iota"]` and `result["G"]` are JAX arrays, not Python floats (unlike `run_code()`).
  - Impact: **A consumer expecting `float(result["iota"])` to work will get surprising JAX tracer behavior inside JIT.** Docstring should warn.

- [ ] **M5 adapter seam (CPU value, JAX gradient) untested**
  - CPU value: `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/geo/surfaceobjectives_jax.py:356-365` (`_surface_geometry_from_dofs`)
  - CPU surface: `surface.gamma()` in the Optimizable
  - Impact: **The M5 adapter uses CPU `surface.gamma()` for value but JAX `_surface_geometry_from_dofs()` for gradient.** If these produce different geometry from the same DOFs, the gradient is wrong. No test verifies they agree.

- [ ] **`default-long-run-gate` geometry policy never exercised on HF**
  - File: `/Users/suhjungdae/code/columbia/simsopt-jax/benchmarks/validation_ladder_contract.py:112`
  - Has contract + unit tests but no production evidence.
  - Impact: **The contract defines 3 geometry policies but only 1 is ever used in practice.** The other 2 are tested in isolation but never on real hardware.

---

## Priority Order for C++ Replacement Readiness

### P0 — Blocking (must fix before claiming replacement parity)

1. **First successful A100 single-stage run** — GPU evidence
2. **Add M5 value parity tests** — `IotasJAX.J()` (rtol=1e-12) / `NonQuasiSymmetricRatioJAX.J()` (rtol=1e-8) / `BoozerResidualJAX.J()` (rtol=1e-12) vs CPU equivalents
3. **Add JAX vs C++ penalty function parity** — mirror Columbia fork `cpp_notcpp` at atol=1e-13 (value), atol=1e-11 (gradient), atol=1e-10 (Hessian). NOTE: the non-vectorized Python baseline was removed from upstream `hiddenSymmetries/simsopt`; compare JAX directly against `boozer_penalty_constraints_vectorized` (C++) instead
4. **Tighten LS `run_code` iota parity** — 1e-6 to **1e-11** (match native `test_convergence_cpp_and_notcpp_same`)
5. **Tighten `newton_tol`** — 1e-9 to **1e-12** in solver options for parity tests (costs ~5 extra Newton iterations)

Why: without these, you cannot claim JAX produces the same answer as C++. With them, you achieve 11 sig fig agreement — the same precision the native test suite proves internally.

### P1 — High (tighten before production use)

6. **Tighten Tier 3 e2e** — iota 1e-3 to **1e-10**, volume 1e-6 to **1e-10**, field 1e-4 to **1e-8**, geometry 1e-5 to **1e-9**
7. **Add M5 gradient parity** — `IotasJAX.dJ()` vs `Iotas.dJ()` direct comparison (expected ~rtol=1e-10)
8. **Run solver parity on real Columbia fixture** — not just toy 2-coil; measure actual agreement at production scale
9. **Mirror native Taylor tests** — Iotas/NonQSRatio/BoozerResidual at native precision (eps=2^(-13 to -19), atol=2e-8)
10. **Tighten exact Newton parity** — 1e-5 to **1e-10**

Why: these prove the optimizer follows the same descent direction, converges to the same minimum, on the real problem, to 10+ significant figures.

### P2 — Medium (harden before upstream merge)

11. **Add scipy-backed M5 integration test** — currently only ondevice tested
12. **GPU vs CPU-JAX numerical calibration** — after first A100 data; expect ~1e-12 per-op noise, ~1e-10 through solver
13. **Tighten M1 B-field vs C++** — 1e-10 to **1e-13** (match native penalty value precision)
14. **Tighten M2 SquaredFlux gradient** — 1e-9 to **1e-11** (match native penalty gradient precision)
15. **Document Tier 4 `full_resolve_fd_rel_tol` rationale**
16. **Early GPU smoke check in HF job**
17. **Test M5 adapter seam** — verify `surface.gamma()` == `_surface_geometry_from_dofs()` on same DOFs

Why: these harden the implementation against subtle regressions and make the codebase mergeable upstream.

### P3 — Low (cleanup / nice-to-have)

18. **Move issue report out of repo root**
19. **HF clone `--depth 1` optimization**
20. **Document `run_code_traceable()` JAX scalar convention**
21. **`test_section6_public_lane_split.py` simsoptpp dependency audit**

Why: polish.
