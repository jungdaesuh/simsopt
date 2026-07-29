# SIMSOPT native CPU versus JAX CPU/GPU parity and performance

**Evidence date:** 2026-07-29

**Scope:** the 26 external-solver-free, one-to-one native-example mirrors

**Scientific precision:** FP64

**Authority revision:** `11340c829690fdc0652e47588f5da549829c056a`

**Authority run:** `20260729T005942Z-5ade9aee`

**Authority verdict:** 26/26 cases, 78/78 lane receipts, and 1,248/1,248
comparisons passed

## Executive summary

- All 26 bounded native examples passed in three independently launched lanes:
  native SIMSOPT/simsoptpp CPU, JAX CPU, and strict JAX GPU on an NVIDIA
  GeForce RTX 5090.
- All 416 declared native-CPU/JAX-GPU comparison routes passed their predefined
  tolerances. Across all three lane pairs, 1,064 comparisons were numerical
  `allclose` checks, 180 were exact checks, and 4 were `not_worse` checks.
- Initial-state agreement is generally exact or at FP64 roundoff. The largest
  relative difference among the representative initial scalars below is
  `8.06e-13%`.
- All 26 native and GPU lanes passed the normalized bounded-workflow predicate.
  This is not a claim that every underlying optimizer converged: raw statuses
  include fixed iteration budgets, iteration-limit exits, and unsuccessful
  outer solves with accepted finite endpoints.
- Final objectives can differ when the solver sequences accept different
  bounded-run endpoints. The largest representative non-near-zero difference
  is planar Stage-II, for which the GPU objective is 16.2633% lower. Standard
  and stochastic Stage-II differ by 1.56037% and 0.736265%, respectively.
- The authority run retains one parent-observed, isolated-child wall time and
  launched-child RSS measurement for each of its 78 lane launches, together with
  child peak RSS and GPU allocation. It does **not** isolate compile time or
  provide controlled-cold or warmed repetitions.
- Summed across the 26 one-shot bounded executions, native CPU used 84.740877 s,
  JAX CPU used 457.572343 s, and JAX GPU used 1,196.250171 s. The GPU total was
  14.1166x the native total (+1,311.657%). These are validation-campaign costs,
  not steady-state benchmark results.
- A separate five-workload JAX fast-versus-parity benchmark has cold and warm
  timing, RSS, and VRAM measurements. It does not provide a native C++ baseline
  and is therefore reported separately.

This is a bounded-scale parity result. Native-default-scale certification
remains `not_run`, and no table in this report claims that JAX GPU is generally
faster than native C++.

## How to read the numerical tables

For a scalar observable, “raw” is the literal recorded FP64 scalar. For a
vector or tensor, “raw” is its L2 norm and the table also gives the maximum
elementwise absolute difference. Relative difference is

```text
100 * abs(JAX_GPU - native_CPU) / abs(native_CPU).
```

For vector or tensor rows, relative error is

```text
100 * norm(JAX_GPU - native_CPU, 2) / norm(native_CPU, 2).
```

This normwise error detects direction and component changes that subtracting
the two vector norms can hide. When the native scalar or vector norm is zero,
the relative result is undefined and is shown as `—`. When a native scalar is
merely close to zero, the percentage is mathematically valid but
ill-conditioned; the absolute difference is the scientifically useful number.

Each row selects one representative primary observable so that all 26 cases
fit in a reviewable table. This is not a replacement for the complete
machine-readable record: the authority artifact contains every residual,
gradient, Jacobian, constraint, parameter, trajectory, and diagnostic route
used in the 1,248-check verdict.

## Initial-state parity: native CPU versus strict JAX GPU

| Case | Representative observable | Representation | Native CPU raw | JAX GPU raw | Max elementwise absolute difference | Relative difference |
|---|---|---:|---:|---:|---:|---:|
| boozer | `initial:residual_norm` | scalar | 52.8758273275 | 52.8758273275 | 0 | 0% |
| boozerqa | `initial:objective` | scalar | 0.000400219206543 | 0.000400219206543 | 3.7947076037e-19 | 9.48157e-14% |
| coil-forces | `initial:objective` | scalar | 0.122521880457 | 0.122521880457 | 0 | 0% |
| just-a-quadratic | `initial:objective_sum_squares` | scalar | 36 | 36 | 0 | 0% |
| minimize-curve-length | `initial:objective_sum_squares` | scalar | 2344.17312592 | 2344.17312592 | 0 | 0% |
| permanent-magnet-muse | `initial:objective_sum_squares` | scalar | 0.000781385130239 | 0.000781385130239 | 0 | 0% |
| permanent-magnet-pm4stell | `initial:objective_sum_squares` | scalar | 0.282544904313 | 0.282544904313 | 0 | 0% |
| permanent-magnet-qa | `initial:objective_sum_squares` | scalar | 0.0121912800423 | 0.0121912800423 | 0 | 0% |
| permanent-magnet-simple | `initial:objective_sum_squares` | scalar | 0.381486930553 | 0.381486930553 | 0 | 0% |
| qfm | `initial:qfm_value` | scalar | 0.0166141725672 | 0.0166141725672 | 1.73472347598e-17 | 1.04412e-13% |
| single-stage-boozer-vacuum-optimization | `initial:objective` | scalar | 0.000390284322085 | 0.000390284322085 | 3.14418630021e-18 | 8.05614e-13% |
| stage-two-optimization | `initial:objective` | scalar | 0.213188718113 | 0.213188718113 | 2.77555756156e-17 | 1.30193e-14% |
| stage-two-optimization-finitebuild | `initial:objective` | scalar | 0.0325812054749 | 0.0325812054749 | 3.46944695195e-17 | 1.06486e-13% |
| stage-two-optimization-minimal | `initial:objective` | scalar | 0.0343532545247 | 0.0343532545247 | 6.93889390391e-18 | 2.01987e-14% |
| stage-two-optimization-planar-coils | `initial:objective` | scalar | 23.5002138641 | 23.5002138641 | 3.5527136788e-15 | 1.51178e-14% |
| stage-two-optimization-stochastic | `initial:objective` | scalar | 0.0124321823096 | 0.0124321823096 | 3.46944695195e-18 | 2.7907e-14% |
| strain-optimization | `initial:objective` | scalar | 3.34830877827e-05 | 3.34830877827e-05 | 6.77626357803e-21 | 2.02379e-14% |
| surf-vol-area | `first:initial:objective_sum_squares` | scalar | 16.5820795293 | 16.5820795293 | 8.52651282912e-14 | 5.142e-13% |
| tracing-fieldlines-ncsx | `initial:states` | L2 norm, n=9 | 2.89251899502 | 2.89251899502 | 0 | 0% |
| tracing-fieldlines-qa | `initial:states` | L2 norm, n=9 | 2.17237142283 | 2.17237142283 | 0 | 0% |
| tracing-particle | `initial:states` | L2 norm, n=12 | 1081582.28636 | 1081582.28636 | 0 | 0% |
| wireframe-gsco-modular | `initial:total_objective` | scalar | 0.128984983324 | 0.128984983324 | 2.77555756156e-17 | 2.15185e-14% |
| wireframe-gsco-multistep | `initial:normal_objective` | scalar | 0.315256579924 | 0.315256579924 | 0 | 0% |
| wireframe-gsco-sector-saddle | `initial:total_objective` | scalar | 0.13282745365 | 0.13282745365 | 0 | 0% |
| wireframe-rcls-basic | `initial:total_objective` | scalar | 0.14536452631 | 0.14536452631 | 0 | 0% |
| wireframe-rcls-with-ports | `initial:total_objective` | scalar | 0.124199627545 | 0.124199627545 | 1.38777878078e-17 | 1.11738e-14% |

## Final-result parity: native CPU versus strict JAX GPU

| Case | Representative observable | Representation | Native CPU raw | JAX GPU raw | Max elementwise absolute difference | Relative difference |
|---|---|---:|---:|---:|---:|---:|
| boozer | `area:residual_norm` | scalar | 1.84773545566e-12 | 5.53361289101e-13 | 1.29437416656e-12 | 70.0519% |
| boozerqa | `final:objective` | scalar | 5.22338676454e-05 | 5.2233867645e-05 | 3.69204721049e-16 | 7.0683e-10% |
| coil-forces | `final:objective` | scalar | 0.00342419993098 | 0.00342419993098 | 1.96457433654e-16 | 5.73732e-12% |
| just-a-quadratic | `final:objective_sum_squares` | scalar | 0 | 2.50167514568e-28 | 2.50167514568e-28 | — |
| minimize-curve-length | `final:objective_sum_squares` | scalar | 355.305758558 | 355.305758439 | 1.18418256534e-07 | 3.33285e-08% |
| permanent-magnet-muse | `final:objective_sum_squares` | scalar | 0.000757256579622 | 0.000757256579622 | 2.16840434497e-19 | 2.8635e-14% |
| permanent-magnet-pm4stell | `final:objective_sum_squares` | scalar | 0.00728022393316 | 0.00728022393316 | 2.08166817117e-17 | 2.85935e-13% |
| permanent-magnet-qa | `final:objective_sum_squares` | scalar | 0.00221072852511 | 0.00221074238204 | 1.38569253082e-08 | 0.000626804% |
| permanent-magnet-simple | `final:objective_sum_squares` | scalar | 0.36520685775 | 0.36520685775 | 2.22044604925e-16 | 6.07997e-14% |
| qfm | `area:exact:qfm_value` | scalar | 0.00114616121041 | 0.00114616114267 | 6.77404924689e-11 | 5.91021e-06% |
| single-stage-boozer-vacuum-optimization | `final:objective` | scalar | 0.000270780779834 | 0.000270780779834 | 5.96311194867e-18 | 2.20219e-12% |
| stage-two-optimization | `final:objective` | scalar | 1.44163437495e-06 | 1.4641292451e-06 | 2.24948701511e-08 | 1.56037% |
| stage-two-optimization-finitebuild | `final:objective` | scalar | 0.00531308145509 | 0.00531308145511 | 1.79188261451e-14 | 3.37259e-10% |
| stage-two-optimization-minimal | `final:objective` | scalar | 1.00446449822e-18 | 1.20783727368e-19 | 8.83680770853e-19 | 87.9753% |
| stage-two-optimization-planar-coils | `final:objective` | scalar | 0.00115153962093 | 0.000964260707242 | 0.000187278913688 | 16.2633% |
| stage-two-optimization-stochastic | `final:objective` | scalar | 1.30238424265e-05 | 1.31197324422e-05 | 9.58900157359e-08 | 0.736265% |
| strain-optimization | `final:objective` | scalar | 2.74672297411e-07 | 2.74672297411e-07 | 1.05879118407e-22 | 3.85474e-14% |
| surf-vol-area | `second:final:objective_sum_squares` | scalar | 5.84866405512e-29 | 1.63782623215e-25 | 1.63724136574e-25 | 279934% |
| tracing-fieldlines-ncsx | `final:states` | L2 norm, n=9 | 2.46162360311 | 2.461486805 | 0.0247930225297 | 1.22397% |
| tracing-fieldlines-qa | `final:states` | L2 norm, n=9 | 1.64776087127 | 1.64786477306 | 0.000391307954554 | 0.0280405% |
| tracing-particle | `final:positions` | L2 norm, n=9 | 2.5315466482 | 2.53154820654 | 2.77404235009e-06 | 0.000173747% |
| wireframe-gsco-modular | `final:total_objective` | scalar | 0.0240232063435 | 0.0240232063435 | 2.08166817117e-17 | 8.66524e-14% |
| wireframe-gsco-multistep | `final:normal_objective` | scalar | 0.0116353641155 | 0.0116353641155 | 1.73472347598e-18 | 1.49091e-14% |
| wireframe-gsco-sector-saddle | `final:total_objective` | scalar | 0.022028207044 | 0.022028207044 | 1.73472347598e-17 | 7.87501e-14% |
| wireframe-rcls-basic | `final:total_objective` | scalar | 0.0072593110819 | 0.0072593110819 | 2.60208521397e-18 | 3.58448e-14% |
| wireframe-rcls-with-ports | `final:total_objective` | scalar | 3.12796971953e-06 | 3.12796971953e-06 | 1.38489886876e-19 | 4.42747e-12% |

Three large percentages above are near-zero denominator effects:

- Boozer residuals are both around `1e-12`; the GPU residual is smaller.
- Stage-II minimal differs by only `8.8368e-19` in absolute objective.
- Surface geometry differs by only `1.6372e-25` in absolute objective.

Planar Stage-II is different: its `16.2633%` reduction is a real difference
between accepted bounded-run endpoints, not a near-zero percentage artifact.

## Solver, iteration, evaluation, and status parity

`—` means that the lane did not publish that counter. A missing counter is not
treated as zero.

| Case | Native driver | JAX GPU driver | iterations native/GPU | function evaluations native/GPU | Jacobian evaluations native/GPU | normalized workflow status native/GPU |
|---|---|---|---:|---:|---:|---|
| boozer | `simsopt_scipy_lbfgsb_manual_lm` | `simsopt_jax_bfgs_lm` | —/— | —/— | —/— | converged/converged |
| boozerqa | `simsopt_scipy_bfgs_with_boozer_newton` | `simsopt_jax_host_bfgs_with_traceable_boozer_newton` | 5/5 | 7/7 | 7/7 | converged/converged |
| coil-forces | `scipy_lbfgsb_two_stage_force` | `simsopt_lbfgsb` | 6/6 | 10/10 | 10/10 | converged/converged |
| just-a-quadratic | `simsopt_least_squares_serial_solve` | `simsopt_lm_gmres` | —/4 | —/5 | —/5 | converged/converged |
| minimize-curve-length | `simsopt_least_squares_serial_solve` | `simsopt_bfgs` | —/16 | —/37 | —/37 | converged/converged |
| permanent-magnet-muse | `simsoptpp_gpmo_arbvec_backtracking` | `simsopt_jax_gpmo_arbvec_backtracking` | 20/20 | 20/20 | —/— | converged/converged |
| permanent-magnet-pm4stell | `simsoptpp_gpmo_arbvec_backtracking` | `simsopt_jax_gpmo_arbvec_backtracking` | 20/20 | 20/20 | —/— | converged/converged |
| permanent-magnet-qa | `simsoptpp_mwpgp_relax_and_split` | `simsopt_jax_mwpgp_relax_and_split` | 2/2 | 20/20 | —/— | converged/converged |
| permanent-magnet-simple | `simsoptpp_gpmo_baseline` | `simsopt_jax_gpmo_baseline` | 40/40 | 40/40 | —/— | converged/converged |
| qfm | `simsopt_lbfgsb_then_slsqp_qfm_sequence` | `simsopt_bfgs_augmented_lagrangian_qfm_sequence` | 210/76 | 294/126 | 222/126 | converged/converged |
| single-stage-boozer-vacuum-optimization | `simsopt_scipy_bfgs_with_boozer_newton` | `simsopt_jax_host_bfgs_with_traceable_boozer_newton` | 2/2 | 5/5 | 5/5 | converged/converged |
| stage-two-optimization | `scipy_lbfgsb_two_stage` | `simsopt_lbfgsb` | 100/100 | 197/183 | 197/183 | converged/converged |
| stage-two-optimization-finitebuild | `scipy_lbfgsb_finite_build` | `simsopt_lbfgsb` | 3/3 | 6/6 | 6/6 | converged/converged |
| stage-two-optimization-minimal | `scipy_lbfgsb` | `simsopt_bfgs` | 76/76 | 103/78 | 103/78 | converged/converged |
| stage-two-optimization-planar-coils | `scipy_lbfgsb_two_stage` | `simsopt_lbfgsb` | 100/100 | 277/149 | 277/149 | converged/converged |
| stage-two-optimization-stochastic | `scipy_lbfgsb` | `simsopt_bfgs` | 20/20 | 25/22 | 25/22 | converged/converged |
| strain-optimization | `scipy_lbfgsb_native_strain_objective` | `simsopt_jax_lbfgsb_strain_objective` | 50/50 | 54/54 | 54/54 | converged/converged |
| surf-vol-area | `simsopt_least_squares_serial_solve` | `simsopt_lm_gmres` | —/30 | —/32 | —/32 | converged/converged |
| tracing-fieldlines-ncsx | `simsoptpp_lsoda_fieldline` | `simsopt_jax_dopri5_fieldline` | —/— | —/— | —/— | converged/converged |
| tracing-fieldlines-qa | `simsoptpp_lsoda_fieldline` | `simsopt_jax_dopri5_fieldline` | —/— | —/— | —/— | converged/converged |
| tracing-particle | `simsoptpp_lsoda_guiding_center` | `simsopt_jax_dopri5_guiding_center` | —/— | —/— | —/— | converged/converged |
| wireframe-gsco-modular | `simsopt_cpp_gsco` | `simsopt_jax_gsco` | 40/40 | 40/40 | —/— | converged/converged |
| wireframe-gsco-multistep | `simsopt_cpp_multistep_gsco` | `simsopt_jax_multistep_gsco` | 80/80 | 80/80 | —/— | converged/converged |
| wireframe-gsco-sector-saddle | `simsopt_cpp_gsco` | `simsopt_jax_gsco` | 40/40 | 40/40 | —/— | converged/converged |
| wireframe-rcls-basic | `simsopt_wireframe_rcls_and_field_postprocessing` | `simsopt_jax_wireframe_rcls_and_field_postprocessing` | 1/1 | 1/1 | —/— | converged/converged |
| wireframe-rcls-with-ports | `simsopt_wireframe_rcls_and_field_postprocessing` | `simsopt_jax_wireframe_rcls_and_field_postprocessing` | 1/1 | 1/1 | —/— | converged/converged |

The nonzero counter differences are:

| Case | Iterations native/GPU | GPU iteration difference | Function evaluations native/GPU | GPU evaluation difference |
|---|---:|---:|---:|---:|
| qfm | 210/76 | -63.810% | 294/126 | -57.143% |
| stage-two-optimization | 100/100 | 0% | 197/183 | -7.107% |
| stage-two-optimization-minimal | 76/76 | 0% | 103/78 | -24.272% |
| stage-two-optimization-planar-coils | 100/100 | 0% | 277/149 | -46.209% |
| stage-two-optimization-stochastic | 20/20 | 0% | 25/22 | -12.000% |

Every other case with native and GPU iteration counters has a 0% iteration
difference. Every other case with native and GPU function-evaluation counters
has a 0% evaluation difference. Negative percentages mean that the GPU lane's
solver sequence used fewer published iterations or evaluations; they do not
by themselves imply shorter wall time.

The status column reproduces the authority schema's `normalized_status`; it is
a workflow-level acceptance label. It must not be read as the underlying
solver's raw convergence result. Examples of retained raw statuses include
`fixed_iteration_budget_complete`, iteration-limit exits, and Boozer
`inner=True;outer=False`.

The driver names show that “parity” means matched scientific behavior, not
necessarily the same implementation:

- native SciPy L-BFGS-B is paired with SIMSOPT-owned JAX BFGS or L-BFGS-B;
- native serial least squares is paired with SIMSOPT-owned JAX LM-GMRES;
- native LSODA tracing is paired with SIMSOPT-owned JAX DOPRI5;
- simsoptpp GPMO, MwPGP, and GSCO are paired with their SIMSOPT-owned JAX
  implementations;
- Boozer QA and vacuum single-stage retain a host-controlled BFGS outer loop,
  while the JAX objective, derivatives, and traceable Boozer inner solve run on
  the selected JAX device.

## Peak memory in the 26-case authority run

Host values are child-reported `ru_maxrss` converted to MiB. GPU values are
JAX `peak_bytes_in_use` converted to MiB. “GPU host versus native” is
`100 * (GPU_host/native_host - 1)`.

These measurements cover imports, compilation/warmup, and one bounded
execution. They are not steady-state memory measurements and should not be
added together as if they were simultaneous allocations.

| Case | Native host peak MiB | GPU host peak MiB | GPU host versus native | GPU device peak MiB |
|---|---:|---:|---:|---:|
| boozer | 589.777 | 1366.789 | +131.747% | 256.890 |
| boozerqa | 590.723 | 2116.461 | +258.283% | 64.217 |
| coil-forces | 1021.707 | 1996.391 | +95.398% | 64.414 |
| just-a-quadratic | 591.387 | 967.773 | +63.645% | 0.015 |
| minimize-curve-length | 590.191 | 1131.457 | +91.710% | 64.010 |
| permanent-magnet-muse | 584.734 | 1111.637 | +90.110% | 64.379 |
| permanent-magnet-pm4stell | 588.941 | 1109.617 | +88.409% | 64.614 |
| permanent-magnet-qa | 959.680 | 985.500 | +2.691% | 0.062 |
| permanent-magnet-simple | 585.004 | 943.215 | +61.232% | 0.265 |
| qfm | 589.414 | 1337.363 | +126.897% | 64.547 |
| single-stage-boozer-vacuum-optimization | 1021.707 | 2113.984 | +106.907% | 64.094 |
| stage-two-optimization | 960.680 | 1869.266 | +94.577% | 64.198 |
| stage-two-optimization-finitebuild | 1021.707 | 2055.238 | +101.157% | 65.750 |
| stage-two-optimization-minimal | 591.766 | 1532.977 | +159.051% | 64.172 |
| stage-two-optimization-planar-coils | 961.180 | 1994.211 | +107.475% | 64.198 |
| stage-two-optimization-stochastic | 963.180 | 1525.055 | +58.335% | 64.242 |
| strain-optimization | 963.180 | 1322.875 | +37.345% | 0.143 |
| surf-vol-area | 588.367 | 1290.062 | +119.261% | 64.008 |
| tracing-fieldlines-ncsx | 633.621 | 1501.008 | +136.894% | 357.344 |
| tracing-fieldlines-qa | 615.801 | 1421.016 | +130.759% | 128.239 |
| tracing-particle | 592.527 | 1680.988 | +183.698% | 703.567 |
| wireframe-gsco-modular | 963.680 | 977.094 | +1.392% | 0.255 |
| wireframe-gsco-multistep | 1021.707 | 1021.707 | +0.000% | 0.168 |
| wireframe-gsco-sector-saddle | 963.680 | 980.781 | +1.775% | 0.255 |
| wireframe-rcls-basic | 1016.055 | 1133.922 | +11.600% | 70.139 |
| wireframe-rcls-with-ports | 1021.707 | 1136.531 | +11.238% | 73.047 |

The largest child-reported host peak is 2,219,270,144 bytes
(2,116.461 MiB) for Boozer QA. The separate parent-observed launched-child
measurement reaches 2,292,080,640 bytes for the strict-GPU Boozer-vacuum
single-stage lane. The largest GPU allocation is 737,743,616 bytes
(703.567 MiB) for particle tracing.

## Wall time, cold time, compile time, and warm time

### What is and is not measured

| Evidence set | Native/GPU matched wall time | Cold | Compile-only | Warm | Host RSS | GPU memory |
|---|---:|---:|---:|---:|---:|---:|
| 26-case authority run | one isolated launch per lane | not controlled | not measured | not measured | yes | yes |
| Five-workload JAX fast/parity benchmark | JAX-only, no native baseline | yes | not isolated | 7 paired repetitions | yes | yes |
| VMEC-hybrid local run | metrics not retained in a receipt | not established | not measured | not measured | not claim-grade | not retained |

The 26-case campaign took 29 minutes 21 seconds from the run identifier
timestamp to its completion marker. This is orchestration wall time for all 78
independent lane launches.

Compile-only time is unavailable. The cold measurements below include process
startup, imports, persistent-cache state, JIT compilation, execution,
synchronization, and receipt publication. Subtracting a warm median from a
cold time would not produce a defensible compile time.

### Authority-run isolated end-to-end wall time

The authority runner measured each child with a parent `perf_counter()` around
the complete subprocess. Each value therefore includes process startup,
imports, the lane's existing cache state, JIT/compilation where applicable,
execution, synchronization, and receipt publication. There is one observation
per lane, without a controlled cache protocol or repetitions. The percentages
are useful for describing this validation campaign, but they are not
steady-state speedups.

Parent-observed RSS is the launched child process's peak `VmHWM` sampled by the
runner. It does not aggregate descendants and is distinct from the child
`ru_maxrss` values in the memory table above.

| Case | Native CPU wall s | JAX CPU wall s | JAX GPU wall s | GPU wall versus native | Native/GPU parent peak MiB | GPU parent RSS versus native |
|---|---:|---:|---:|---:|---:|---:|
| just-a-quadratic | 1.727878 | 3.532206 | 3.718977 | +115.234% | 668.766/1057.055 | +58.061% |
| minimize-curve-length | 2.273584 | 4.191700 | 6.433507 | +182.968% | 668.953/1200.742 | +79.496% |
| permanent-magnet-simple | 2.361102 | 3.375123 | 3.221544 | +36.442% | 667.766/1033.398 | +54.755% |
| qfm | 2.138333 | 24.945247 | 50.368931 | +2255.524% | 669.277/1398.262 | +108.921% |
| stage-two-optimization-minimal | 1.999803 | 14.893482 | 34.007180 | +1600.527% | 675.957/1608.121 | +137.903% |
| surf-vol-area | 1.958091 | 14.074629 | 25.744723 | +1214.787% | 667.660/1350.230 | +102.233% |
| tracing-fieldlines-ncsx | 2.787933 | 18.352228 | 96.843037 | +3373.650% | 725.695/1534.457 | +111.446% |
| tracing-fieldlines-qa | 1.936870 | 16.879116 | 100.467023 | +5087.080% | 693.188/1496.129 | +115.833% |
| tracing-particle | 2.167364 | 15.054966 | 54.828442 | +2429.729% | 677.535/1706.012 | +151.797% |
| boozer | 2.274912 | 14.070808 | 20.632089 | +806.940% | 670.844/1422.680 | +112.073% |
| boozerqa | 2.612852 | 49.579738 | 89.308499 | +3318.047% | 671.023/2180.355 | +224.930% |
| permanent-magnet-muse | 2.925915 | 3.016964 | 3.768538 | +28.799% | 674.664/1193.648 | +76.925% |
| permanent-magnet-pm4stell | 2.553192 | 3.312094 | 4.839412 | +89.544% | 676.320/1187.941 | +75.648% |
| permanent-magnet-qa | 2.005148 | 4.280223 | 4.939642 | +146.348% | 668.254/1076.441 | +61.083% |
| stage-two-optimization | 5.340888 | 35.972989 | 118.463526 | +2118.049% | 816.059/1927.379 | +136.181% |
| stage-two-optimization-planar-coils | 6.577998 | 52.538032 | 144.810948 | +2101.444% | 821.664/2051.332 | +149.656% |
| stage-two-optimization-stochastic | 5.175548 | 12.744761 | 32.563875 | +529.187% | 821.824/1596.633 | +94.279% |
| strain-optimization | 5.565510 | 8.502953 | 29.058550 | +422.118% | 925.230/1405.730 | +51.933% |
| wireframe-gsco-modular | 2.275638 | 3.018582 | 4.447346 | +95.433% | 668.895/1070.988 | +60.113% |
| wireframe-gsco-sector-saddle | 2.542881 | 3.388137 | 4.706880 | +85.100% | 668.676/1070.855 | +60.146% |
| wireframe-rcls-basic | 2.687815 | 3.154913 | 3.945360 | +46.787% | 680.238/1226.441 | +80.296% |
| wireframe-rcls-with-ports | 2.113715 | 9.073668 | 4.592282 | +117.261% | 682.684/1222.152 | +79.022% |
| coil-forces | 9.154687 | 47.992761 | 147.314503 | +1509.170% | 994.172/2051.227 | +106.325% |
| single-stage-boozer-vacuum-optimization | 2.299332 | 46.252876 | 84.067757 | +3556.182% | 670.219/2185.898 | +226.147% |
| stage-two-optimization-finitebuild | 7.493834 | 42.481764 | 118.380292 | +1479.703% | 901.500/2078.082 | +130.514% |
| wireframe-gsco-multistep | 1.790057 | 2.892382 | 4.777306 | +166.880% | 669.051/1053.172 | +57.413% |
| **Sum of lane elapsed times** | **84.740877** | **457.572343** | **1196.250171** | **+1311.657%** | — | — |

The authority GPU lane is slower than native CPU in every bounded case. That
observation is specific to fresh isolated validation launches and includes JAX
startup and compilation effects. It does not predict throughput for a larger
same-process workload that amortizes compilation.

### JAX GPU fast versus JAX GPU parity: cold and warmed measurements

This local RTX 5090 diagnostic is not native C++ versus JAX GPU evidence. It is
included to document the measured fast-mode behavior. Each cold run started
with an empty profile-specific persistent cache. Warm statistics use seven
balanced paired repetitions. “Paired warm speedup” is the benchmark's median
of paired `parity/fast` ratios; positive percentage means fast is faster.

| Workload | Cold parity/fast s | Warm median parity/fast s | Paired warm speedup | 95% lower bound | Host peak fast/parity ratio | VRAM peak fast/parity ratio |
|---|---:|---:|---:|---:|---:|---:|
| traceable-least-squares | 1.570356 / 1.635742 | 1.266591 / 1.347578 | 1.005617x (+0.562%) | 0.909396x | 1.035965x (+3.597%) | 1.234432x (+23.443%) |
| curve-length-optimization | 3.206125 / 3.657130 | 2.327011 / 2.110266 | 1.025516x (+2.552%) | 0.999536x | 1.103043x (+10.304%) | 1.234432x (+23.443%) |
| surface-geometry-optimization | 6.916989 / 7.002769 | 3.486808 / 3.413276 | 1.036187x (+3.619%) | 0.857579x | 1.078408x (+7.841%) | 1.467153x (+46.715%) |
| coil-flux-optimization | 3.737068 / 4.033779 | 2.472577 / 2.240982 | 1.104171x (+10.417%) | 0.992904x | 1.071184x (+7.118%) | 1.468864x (+46.886%) |
| fieldline-and-particle-tracing | 3.462484 / 3.986709 | 2.434236 / 2.260534 | 1.058354x (+5.835%) | 1.001966x | 1.013405x (+1.341%) | 1.059480x (+5.948%) |

The raw median time ratio can differ from the median of paired speedup ratios
because process order was balanced within each pair. The paired ratio and its
bootstrap lower bound are the prespecified policy statistics.

Only field-line/particle tracing passed every fast-mode promotion gate in this
run. The overall fast policy was not promoted: four workloads missed the
one-sided warm-speed confidence gate, and surface/coil exceeded the unchanged
1.25 VRAM-ratio ceiling.

### VMEC-hybrid evidence gap

VMEC remains a CPU/MPI host solve in the hybrid example; only the JAX-owned
coil slice moves between CPU and GPU. A local run at ancestor revision
`e07a30635` left CPU/GPU VMEC inputs, `wout`, `threed1`, and finite-difference
working files under
`.artifacts/jax-authority-evidence/vmec-hybrid-e07a/`.

The run's JSON stdout containing objectives, evaluation counts, wall time, RSS,
configuration fingerprint, and hardware identity was never persisted. Exact
VMEC-hybrid performance numbers are therefore excluded from this report:
working outputs alone cannot reconstruct or authenticate those metrics. The
hybrid workflow needs a machine-readable, checksum-bound CPU/GPU receipt
before it can appear in a performance table.

## Evidence provenance and limitations

Primary authority evidence:

```text
.artifacts/jax-example-parity/20260729T005942Z-5ade9aee/
```

The archived `summary.json` SHA-256 is
`fa235cacb0f3e4fa7abc6e8ff4b2f888b2e20c7392bd1531eeb983abad67d66a`.
The run is bound to a clean checkout, exact source hashes, exact input bundles,
FP64, strict GPU transfer guards, and an independently replayed fail-closed
audit.

Fast/parity performance evidence:

```text
.artifacts/jax-example-execution-modes/20260727T064728Z-gpu-829eb1cbf037/
```

This run used JAX/JAXLIB 0.10.2, an NVIDIA GeForce RTX 5090, and an AMD Ryzen
Threadripper 9970X. Its artifact labels itself non-certifying and its decision
is `promoted: false`. SHA-256:

```text
artifact.json  6213e4a6a826c4d45ae82d63c644f3c24074d0e0225160d0eb45dcb612f73a4f
decision.json  cc7b5ade9ee50be1d64547127e484f7bc1e92e14ff7cfe77eb8afb39f65ce3d8
```

The exact declared per-case route matrix is owned by
[`examples/jax/parity_manifest.json`](../examples/jax/parity_manifest.json),
whose authority-bound SHA-256 is
`dd532a55f7baeb841fc2f8c0df1c8c4368438ccbdd6c8e9ad11d2d41ae616b78`.
The archived artifacts are host-local and Git-ignored; their retention and
audit command are documented in
`.artifacts/jax-authority-evidence/README.md`.

Limitations:

1. The scientific authority campaign is bounded-scale, not native-default
   scale.
2. Its memory numbers include compilation and execution and are not
   steady-state measurements.
3. It contains one end-to-end elapsed time per lane, but no isolated compile
   time, controlled-cold repetitions, or warmed repetitions.
4. The five-workload cold/warm benchmark compares JAX fast against JAX parity,
   not native C++ against JAX GPU.
5. The authority elapsed times are one-shot validation-campaign diagnostics
   and cannot support a steady-state speed claim.
6. Percent differences near a zero native result are ill-conditioned; raw
   absolute differences and the declared scientific tolerances control the
   verdict.
7. No Perlmutter data is used in this report.

## Bottom line

The evidence supports a strong bounded scientific-parity claim: all applicable
declared routes across the 26 one-to-one relationships pass native CPU, JAX
CPU, and strict RTX 5090 GPU validation. Derivatives, constraints,
trajectories, and counters are compared only for the cases whose parity
manifest declares those observables.

The evidence does **not** yet support a broad performance claim. Current
authority-run timing shows all 26 isolated bounded GPU launches slower than
their native CPU counterparts, while the JAX-only fast/parity diagnostic shows
modest warmed gains for some workload classes and a statistically clean gain
only for the combined field-line/particle-tracing workload. A future native
SIMSOPT-CPU/JAX-GPU performance campaign should record synchronized cold
end-to-end time, compile-only time, same-process warmed time, process-tree RSS,
and process-attributed VRAM in one machine-readable receipt per representative
workload.
