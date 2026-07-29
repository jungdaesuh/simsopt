# SIMSOPT native CPU versus JAX CPU/GPU parity and performance

**Evidence date:** 2026-07-29

**Scope:** the 26 external-solver-free, one-to-one native-example mirrors

**Scientific precision:** FP64

**Authority revision:** `11340c829690fdc0652e47588f5da549829c056a`

**Authority run:** `20260729T005942Z-5ade9aee`

**Authority verdict:** 26/26 cases, 78/78 lane receipts, and 1,248/1,248
comparisons passed

## Executive summary

- All 26 bounded examples passed native CPU, JAX CPU, and strict RTX 5090 GPU
  lanes.
- All 416 native-CPU/JAX-GPU routes passed. Across all lane pairs, 1,064 checks
  were `allclose`, 180 exact, and 4 `not_worse`.
- Representative initial values agree exactly or at FP64 roundoff.
- Workflow acceptance does not imply solver convergence. Some runs ended at
  fixed budgets or iteration limits.
- Planar Stage-II has the largest non-near-zero final gap: its GPU objective is
  16.2633% lower.
- Authority timings are single validation launches, not steady-state
  benchmarks. Total GPU time was 14.1166x native CPU time.
- The separate fast/parity benchmark has no native C++ baseline.

This report covers bounded scale. Native-default scale remains `not_run`.

## How to read the numerical tables

For scalars, “raw” is the recorded FP64 value. For arrays, it is the L2 norm;
the table also reports the maximum elementwise absolute difference. Scalar
relative difference is

```text
100 * abs(JAX_GPU - native_CPU) / abs(native_CPU).
```

Array relative error is

```text
100 * norm(JAX_GPU - native_CPU, 2) / norm(native_CPU, 2).
```

This normwise metric captures component changes. `—` marks an undefined
relative result. Near zero, use the absolute difference. Each row shows one
representative observable; the artifact retains all declared routes.

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

| Case | Representative observable | Representation | Native CPU raw | JAX GPU raw | Max elementwise absolute difference | Relative difference | Iterations native/GPU | Note |
|---|---|---:|---:|---:|---:|---:|---:|---|
| boozer | `area:residual_norm` | scalar | 1.84773545566e-12 | 5.53361289101e-13 | 1.29437416656e-12 | 70.0519% | —/— | Near zero; use the absolute difference. |
| boozerqa | `final:objective` | scalar | 5.22338676454e-05 | 5.2233867645e-05 | 3.69204721049e-16 | 7.0683e-10% | 5/5 | — |
| coil-forces | `final:objective` | scalar | 0.00342419993098 | 0.00342419993098 | 1.96457433654e-16 | 5.73732e-12% | 6/6 | — |
| just-a-quadratic | `final:objective_sum_squares` | scalar | 0 | 2.50167514568e-28 | 2.50167514568e-28 | — | —/4 | — |
| minimize-curve-length | `final:objective_sum_squares` | scalar | 355.305758558 | 355.305758439 | 1.18418256534e-07 | 3.33285e-08% | —/16 | — |
| permanent-magnet-muse | `final:objective_sum_squares` | scalar | 0.000757256579622 | 0.000757256579622 | 2.16840434497e-19 | 2.8635e-14% | 20/20 | — |
| permanent-magnet-pm4stell | `final:objective_sum_squares` | scalar | 0.00728022393316 | 0.00728022393316 | 2.08166817117e-17 | 2.85935e-13% | 20/20 | — |
| permanent-magnet-qa | `final:objective_sum_squares` | scalar | 0.00221072852511 | 0.00221074238204 | 1.38569253082e-08 | 0.000626804% | 2/2 | — |
| permanent-magnet-simple | `final:objective_sum_squares` | scalar | 0.36520685775 | 0.36520685775 | 2.22044604925e-16 | 6.07997e-14% | 40/40 | — |
| qfm | `area:exact:qfm_value` | scalar | 0.00114616121041 | 0.00114616114267 | 6.77404924689e-11 | 5.91021e-06% | 210/76 | — |
| single-stage-boozer-vacuum-optimization | `final:objective` | scalar | 0.000270780779834 | 0.000270780779834 | 5.96311194867e-18 | 2.20219e-12% | 2/2 | — |
| stage-two-optimization | `final:objective` | scalar | 1.44163437495e-06 | 1.4641292451e-06 | 2.24948701511e-08 | 1.56037% | 100/100 | Both stages hit the iteration limit. |
| stage-two-optimization-finitebuild | `final:objective` | scalar | 0.00531308145509 | 0.00531308145511 | 1.79188261451e-14 | 3.37259e-10% | 3/3 | Iteration limit reached. |
| stage-two-optimization-minimal | `final:objective` | scalar | 1.00446449822e-18 | 1.20783727368e-19 | 8.83680770853e-19 | 87.9753% | 76/76 | Near-zero objective; final curve length differs by 21.38%. |
| stage-two-optimization-planar-coils | `final:objective` | scalar | 0.00115153962093 | 0.000964260707242 | 0.000187278913688 | 16.2633% | 100/100 | Both stages hit the limit; GPU reached a lower objective. |
| stage-two-optimization-stochastic | `final:objective` | scalar | 1.30238424265e-05 | 1.31197324422e-05 | 9.58900157359e-08 | 0.736265% | 20/20 | Iteration limit reached. |
| strain-optimization | `final:objective` | scalar | 2.74672297411e-07 | 2.74672297411e-07 | 1.05879118407e-22 | 3.85474e-14% | 50/50 | Iteration limit reached. |
| surf-vol-area | `second:final:objective_sum_squares` | scalar | 5.84866405512e-29 | 1.63782623215e-25 | 1.63724136574e-25 | 279934% | —/30 | Near zero; area, volume, and parameters agree to about `1e-13`. |
| tracing-fieldlines-ncsx | `final:states` | L2 norm, n=9 | 2.46162360311 | 2.461486805 | 0.0247930225297 | 1.22397% | —/— | LSODA/DOPRI5 at shared `1e-7` tolerance. |
| tracing-fieldlines-qa | `final:states` | L2 norm, n=9 | 1.64776087127 | 1.64786477306 | 0.000391307954554 | 0.0280405% | —/— | — |
| tracing-particle | `final:positions` | L2 norm, n=9 | 2.5315466482 | 2.53154820654 | 2.77404235009e-06 | 0.000173747% | —/— | — |
| wireframe-gsco-modular | `final:total_objective` | scalar | 0.0240232063435 | 0.0240232063435 | 2.08166817117e-17 | 8.66524e-14% | 40/40 | Fixed iteration budget. |
| wireframe-gsco-multistep | `final:normal_objective` | scalar | 0.0116353641155 | 0.0116353641155 | 1.73472347598e-18 | 1.49091e-14% | 80/80 | Fixed iteration budget. |
| wireframe-gsco-sector-saddle | `final:total_objective` | scalar | 0.022028207044 | 0.022028207044 | 1.73472347598e-17 | 7.87501e-14% | 40/40 | Fixed iteration budget. |
| wireframe-rcls-basic | `final:total_objective` | scalar | 0.0072593110819 | 0.0072593110819 | 2.60208521397e-18 | 3.58448e-14% | 1/1 | — |
| wireframe-rcls-with-ports | `final:total_objective` | scalar | 3.12796971953e-06 | 3.12796971953e-06 | 1.38489886876e-19 | 4.42747e-12% | 1/1 | — |

## Solver, iteration, evaluation, and status parity

`—` means the lane did not publish that counter.

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

All other paired iteration and evaluation counters match. Negative percentages
mean fewer GPU iterations or evaluations, not shorter wall time.

`normalized_status` is workflow acceptance, not raw solver convergence. Raw
statuses include fixed budgets, iteration limits, and Boozer
`inner=True;outer=False`.

Parity targets scientific behavior, not identical implementations:

- native SciPy L-BFGS-B is paired with SIMSOPT-owned JAX BFGS or L-BFGS-B;
- native serial least squares is paired with SIMSOPT-owned JAX LM-GMRES;
- native LSODA tracing is paired with SIMSOPT-owned JAX DOPRI5;
- simsoptpp GPMO, MwPGP, and GSCO are paired with their SIMSOPT-owned JAX
  implementations;
- Boozer QA and vacuum single-stage use host-controlled BFGS around JAX
  objectives, derivatives, and inner solves.

## Peak memory in the 26-case authority run

Host values are child `ru_maxrss`; device values are JAX
`peak_bytes_in_use`. Both are MiB. Measurements include imports,
compilation/warmup, and one bounded execution; they are not steady state.

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

Maxima: child host 2,116.461 MiB (Boozer QA), parent-observed child 2,185.898
MiB (Boozer-vacuum GPU), and device 703.567 MiB (particle tracing).

## Wall time, cold time, compile time, and warm time

### What is and is not measured

| Evidence set | Native/GPU matched wall time | Cold | Compile-only | Warm | Host RSS | GPU memory |
|---|---:|---:|---:|---:|---:|---:|
| 26-case authority run | one isolated launch per lane | not controlled | not measured | not measured | yes | yes |
| Five-workload JAX fast/parity benchmark | JAX-only, no native baseline | yes | not isolated | 7 paired repetitions | yes | yes |
| VMEC-hybrid local run | metrics not retained in a receipt | not established | not measured | not measured | not claim-grade | not retained |

The 78-launch campaign took 29 minutes 21 seconds. Compile-only time was not
measured. Cold time includes startup, imports, cache state, JIT, execution,
synchronization, and receipt publication.

### Authority-run isolated end-to-end wall time

Each value wraps one complete subprocess with `perf_counter()`. Cache state was
not controlled and runs were not repeated. Parent RSS is the launched child's
peak `VmHWM`; it excludes descendants.

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

GPU was slower in every isolated bounded launch. These runs include JAX startup
and compilation; they do not predict warmed throughput.

### JAX GPU fast versus JAX GPU parity: cold and warmed measurements

This RTX 5090 diagnostic compares JAX fast and parity modes, not native C++ and
JAX GPU. Cold runs used empty profile caches; warm results use seven balanced
pairs. Paired speedup is the median `parity/fast` ratio.

| Workload | Cold parity/fast s | Warm median parity/fast s | Paired warm speedup | 95% lower bound | Host peak fast/parity ratio | VRAM peak fast/parity ratio |
|---|---:|---:|---:|---:|---:|---:|
| traceable-least-squares | 1.570356 / 1.635742 | 1.266591 / 1.347578 | 1.005617x (+0.562%) | 0.909396x | 1.035965x (+3.597%) | 1.234432x (+23.443%) |
| curve-length-optimization | 3.206125 / 3.657130 | 2.327011 / 2.110266 | 1.025516x (+2.552%) | 0.999536x | 1.103043x (+10.304%) | 1.234432x (+23.443%) |
| surface-geometry-optimization | 6.916989 / 7.002769 | 3.486808 / 3.413276 | 1.036187x (+3.619%) | 0.857579x | 1.078408x (+7.841%) | 1.467153x (+46.715%) |
| coil-flux-optimization | 3.737068 / 4.033779 | 2.472577 / 2.240982 | 1.104171x (+10.417%) | 0.992904x | 1.071184x (+7.118%) | 1.468864x (+46.886%) |
| fieldline-and-particle-tracing | 3.462484 / 3.986709 | 2.434236 / 2.260534 | 1.058354x (+5.835%) | 1.001966x | 1.013405x (+1.341%) | 1.059480x (+5.948%) |

Only field-line/particle tracing passed every promotion gate. Four workloads
missed the warm-speed confidence gate; surface and coil also exceeded the 1.25
VRAM ceiling. Fast mode was not promoted.

### VMEC-hybrid evidence gap

VMEC stays on CPU/MPI; only the JAX coil slice moves between CPU and GPU. A run
at `e07a30635` left working files under
`.artifacts/jax-authority-evidence/vmec-hybrid-e07a/`.

Its JSON stdout was not retained, so exact performance numbers are excluded.
A checksum-bound CPU/GPU receipt is still required.

## Evidence provenance and limitations

Primary authority evidence:

```text
.artifacts/jax-example-parity/20260729T005942Z-5ade9aee/
```

The archived `summary.json` SHA-256 is
`fa235cacb0f3e4fa7abc6e8ff4b2f888b2e20c7392bd1531eeb983abad67d66a`.
The run binds source hashes, inputs, FP64, transfer guards, and a replayed
fail-closed audit.

Fast/parity performance evidence:

```text
.artifacts/jax-example-execution-modes/20260727T064728Z-gpu-829eb1cbf037/
```

This non-certifying run used JAX/JAXLIB 0.10.2, an RTX 5090, and a Threadripper
9970X. Its decision is `promoted: false`. SHA-256:

```text
artifact.json  6213e4a6a826c4d45ae82d63c644f3c24074d0e0225160d0eb45dcb612f73a4f
decision.json  cc7b5ade9ee50be1d64547127e484f7bc1e92e14ff7cfe77eb8afb39f65ce3d8
```

The route matrix is owned by
[`examples/jax/parity_manifest.json`](../examples/jax/parity_manifest.json),
whose authority-bound SHA-256 is
`dd532a55f7baeb841fc2f8c0df1c8c4368438ccbdd6c8e9ad11d2d41ae616b78`.
Artifacts are host-local and Git-ignored. Retention and replay instructions:
`.artifacts/jax-authority-evidence/README.md`.

Limitations:

1. Authority scale is bounded, not native default.
2. Memory includes compilation and execution.
3. Authority timing has one end-to-end sample per lane; no compile-only or
   warmed samples.
4. The cold/warm benchmark is JAX fast versus JAX parity.
5. One-shot authority timings do not establish steady-state speed.
6. Near-zero relative differences are ill-conditioned; use absolute values and
   declared tolerances.
7. No Perlmutter data is used in this report.

## Bottom line

All declared bounded routes pass native CPU, JAX CPU, and strict RTX 5090 GPU
validation. Coverage remains case-specific to the parity manifest.

No broad performance claim is supported. All isolated GPU launches were
slower than native CPU. Only field-line/particle tracing showed a statistically
clean warmed fast-mode gain. A future native-CPU/JAX-GPU campaign needs cold,
compile-only, warmed, RSS, and VRAM measurements in machine-readable receipts.
