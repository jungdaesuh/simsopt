# SIMSOPT native CPU versus JAX CPU/GPU parity and performance

**Evidence date:** 2026-07-29

**Scope:** the 26 external-solver-free, one-to-one native-example mirrors

**Scientific precision:** FP64

**Authority revision:** `11340c829690fdc0652e47588f5da549829c056a`

**Authority run:** `20260729T005942Z-5ade9aee`

**Authority verdict:** 26/26 cases, 78/78 lane receipts, and 1,248/1,248
comparisons passed

## Executive summary

- All 26 bounded examples pass native CPU, JAX CPU, and strict RTX 5090 GPU
  lanes: 1,248/1,248 comparisons (1,064 `allclose`, 180 exact, 4 `not_worse`),
  including all 416 native-CPU/JAX-GPU routes.
- Initial states agree exactly or at FP64 roundoff. The largest real final gap
  is planar Stage-II, where the GPU objective ends 16.3% lower (better) after
  both lanes hit the 100-iteration cap. (Stale at HEAD, checked 2026-08-24: a
  2026-08-02 mirror fix changed this case's objective — see the dated
  correction under the final-result table.)
- Passing means workflow acceptance, not solver convergence; several runs end
  at fixed budgets.
- Timing is one isolated launch per lane, startup and JIT included. Total GPU
  time was 14.1x native CPU. No steady-state claim follows, and the separate
  fast/parity benchmark has no native baseline. (A 2026-08-24 re-measurement
  at HEAD on an A100 host puts the same protocol at 9.47x there — see "Wall
  time at HEAD"; two hosts, two environments, not a trend.)
- Everything here is bounded scale; native-default remains `not_run`. (True of
  every measurement below and of the manifest at this report's authority
  revision. The live manifest has since promoted one route to `native_default`
  — see the pin note under Limitations.)

## How to read the tables

“Raw” is the recorded FP64 value for scalars and the L2 norm for arrays, with
the maximum elementwise difference alongside. Relative difference is
`100 * |JAX_GPU - native_CPU| / |native_CPU|`, normwise for arrays; `—` marks
an undefined result, and near zero the absolute column is the one to read.
Each row is one representative observable; the artifact retains every declared
route.

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
| boozerqa | `final:objective` | scalar | 5.22338676454e-05 | 5.2233867645e-05 | 3.69204721049e-16 | 7.0683e-10% | 5/5 | Fixed iteration budget. |
| coil-forces | `final:objective` | scalar | 0.00342419993098 | 0.00342419993098 | 1.96457433654e-16 | 5.73732e-12% | 6/6 | — |
| just-a-quadratic | `final:objective_sum_squares` | scalar | 0 | 2.50167514568e-28 | 2.50167514568e-28 | — | —/4 | — |
| minimize-curve-length | `final:objective_sum_squares` | scalar | 355.305758558 | 355.305758439 | 1.18418256534e-07 | 3.33285e-08% | —/16 | — |
| permanent-magnet-muse | `final:objective_sum_squares` | scalar | 0.000757256579622 | 0.000757256579622 | 2.16840434497e-19 | 2.8635e-14% | 20/20 | — |
| permanent-magnet-pm4stell | `final:objective_sum_squares` | scalar | 0.00728022393316 | 0.00728022393316 | 2.08166817117e-17 | 2.85935e-13% | 20/20 | — |
| permanent-magnet-qa | `final:objective_sum_squares` | scalar | 0.00221072852511 | 0.00221074238204 | 1.38569253082e-08 | 0.000626804% | 2/2 | — |
| permanent-magnet-simple | `final:objective_sum_squares` | scalar | 0.36520685775 | 0.36520685775 | 2.22044604925e-16 | 6.07997e-14% | 40/40 | — |
| qfm | `area:exact:qfm_value` | scalar | 0.00114616121041 | 0.00114616114267 | 6.77404924689e-11 | 5.91021e-06% | 210/76 | — |
| single-stage-boozer-vacuum-optimization | `final:objective` | scalar | 0.000270780779834 | 0.000270780779834 | 5.96311194867e-18 | 2.20219e-12% | 2/2 | Fixed iteration budget. |
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

> **Row falsified at HEAD, and a 2026-08-24 re-validation of this table.**
> The `stage-two-optimization-planar-coils` row is correct for the authority
> revision but no longer describes the current code: commit `fd200f564`
> (2026-08-02) aligned the mirror's mean-squared-curvature penalty with the
> upstream example's `identity` target mode (the parity case previously used
> `"max"`), which changes the optimization problem itself. Re-measured at HEAD
> on 2026-08-24 (bounded scale, jax/jaxlib 0.10.0, RTX 5090 for the GPU lane;
> a replay environment, not the authority environment): native
> `final:objective` = 0.00221946821602, JAX CPU 0.00103232035072, strict GPU
> 0.00103939304979, and **all lanes converge** instead of hitting the
> iteration cap. The row's raw values, difference cells, `100/100` iteration
> cell, and Note are stale, as is the executive summary's "16.3% lower"
> bullet — at HEAD the GPU endpoint is ~53% below native.
>
> The other rows were re-validated in the same replay pass (native and JAX
> CPU lanes for all bounded cases): 23 of 26 native endpoints reproduce the
> table to its printed precision. Two environment-sensitivity effects
> surfaced, each present identically at the authority revision and at HEAD
> (so neither is a code regression): `permanent-magnet-qa`'s MwPGP endpoint
> lands 3.3e-05 (relative) away from the table in this environment (its
> parity gate still passes), and `tracing-fieldlines-ncsx` — the only gate
> failure of the pass — misses its Poincaré-positions bucket at one of 66
> crossings (0.041 versus `atol` 0.030) from chaotic amplification of
> environment-level FP differences; both lanes' trajectories are bitwise
> identical across the two revisions.
> `single-stage-boozer-vacuum-optimization` cannot be re-validated at
> bounded scale at HEAD (the runner fails closed on its promoted scale tier;
> see the pin note under Evidence provenance). Separately, the pm4stell
> native-versus-JAX last-bit mechanism was discharged on 2026-08-24: with
> FMA contraction disabled, the rebuilt native GPMO state is bitwise
> identical to the archived JAX dump — 0 of 5,826 dipole rows differ
> (commit `b2c099489`, `docs/jax_gpu_pm_gpmo_campaign_plan.md`).
>
> A second-architecture leg (2026-08-24, NVIDIA A100-PCIE-40GB on a
> different host, AMD EPYC 7452, glibc 2.31, jax/jaxlib 0.10.0, same
> revision) re-ran the three-lane bounded matrix for 25 of the 26 cases —
> the 26th, `single-stage-boozer-vacuum-optimization`, is excluded by the
> fail-closed scale refusal above, not by any failure. Result: **1,108 of
> 1,122 executed comparisons pass**; of the 23 cases that produced
> verdicts, 19 are fully clean and 4 carry failures. These fractions must
> not be read against the header's 1,248/1,248 — that was a clean pass in
> one pinned environment; this one contains failures. All 14 failures
> were quantified from the retained value arrays as exact
> allclose-overshoot factors (max elementwise |a-b| divided by its own
> `atol + rtol*|b|` bound). By class: the chaotic tracers miss modestly —
> fieldlines-ncsx (24/30) at 1.60x-1.83x, worst 5.5e-2 against `atol`
> 3e-2, and fieldlines-qa (26/30) at 1.05x-1.41x. The most
> consequential number in the set is also the most benign-looking:
> fieldlines-qa's **native-cpu:jax-cpu** states pair at 1.05x over a
> loose absolute tolerance — a CPU-versus-CPU divergence with no device
> involved, establishing that the cross-host trajectory divergence exists
> independent of any GPU. planar-coils' jax-cpu:jax-gpu final objective
> misses at 1.32x, but its first-phase objective misses at 5.26x (3.5e-4
> absolute). And wireframe-rcls-with-ports' near-zero
> `initial:constraint_residual` (88/90) overshoots its very tight bound
> by 466x — an absolute miss of only 4.7e-10, but against `atol` 1e-12 +
> `rtol` 1e-10 on a dense linear-solve residual, making it the one
> failure that is neither small relative to its own gate nor a chaotic
> trajectory. The absolute misses are small everywhere; the relative
> overshoots are not uniformly so, and no blanket "all marginal" summary
> survives these numbers. One inversion worth naming: field-line
> tracing supplies 10 of these 14 failures yet is the one workload that
> cleared every warmed fast-mode gate below — chaotic amplification of
> floating-point differences is the plausible mechanism, plausible but not
> proven. Two cases could not complete there:
> finitebuild (the manifest-route gap; see the controlled re-measurement)
> and permanent-magnet-qa, whose lane writes `.vtu` grid files into the
> checkout and trips the runner's repository-integrity guard on any fresh
> clone. pm-qa's three lanes still agree internally to ~3e-6 relative on
> that host, but its endpoint lands 16% from this report's value — the
> MwPGP environment sensitivity noted above, at much larger scale across
> architectures. For that case, lane-to-lane parity within an environment
> is the invariant that holds; endpoints do not transport between
> environments.
>
> Scope of this re-validation: endpoints and parity gates, plus a targeted
> GPU-lane re-measurement of six cases with same-environment
> authority-revision controls. For the timing, memory, and precision
> verdicts see "2026-08-24 controlled re-measurement" under the wall-time
> section.

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

All other paired counters match. Fewer GPU evaluations does not mean less wall
time. `normalized_status` records workflow acceptance; raw statuses include
fixed budgets, iteration limits, and Boozer `inner=True;outer=False`. (Format
note, 2026-08-24: since the fail-closed endpoint certificate landed —
`91e1133b9`, 2026-08-02 — the Boozer cases can also emit a seven-field raw
status (`inner`/`outer`/`certificate`/`stopping_reason`/`initial_stationary`/
`terminal_stationary`/`constraints_satisfied`) and normalize budget-capped
endpoints to `budget_exhausted`, but only when the case enforces the endpoint
certificate. The bounded parity path does not, so a bounded replay at HEAD
still reproduces the two-field form and `converged` statuses above — verified
2026-08-24.)

Parity targets scientific behavior, not identical implementations:

- native SciPy L-BFGS-B is paired with SIMSOPT-owned JAX BFGS or L-BFGS-B;
- native serial least squares is paired with SIMSOPT-owned JAX LM-GMRES;
- native LSODA tracing is paired with SIMSOPT-owned JAX DOPRI5;
- simsoptpp GPMO, MwPGP, and GSCO are paired with their SIMSOPT-owned JAX
  implementations;
- Boozer QA and vacuum single-stage use host-controlled BFGS around JAX
  objectives, derivatives, and inner solves.

## Peak memory in the 26-case authority run

Host values are child `ru_maxrss`, device values JAX `peak_bytes_in_use`, both
MiB, covering imports, compilation, and one bounded execution — not steady
state.

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

The 78-launch campaign took 29 minutes 21 seconds. No compile-only or warmed
samples exist: each wall time is one full subprocess, startup through receipt
publication, unrepeated.

Hardware and threading (added 2026-08-24): the authority run's host-local
artifacts live on the same RTX 5090 / Threadripper 9970X box as the
fast/parity benchmark below. The parity runtime pins every native-lane child
to `OMP_NUM_THREADS=1` (for determinism; `examples/jax/parity/runtime.py`,
present at the authority revision), so the native wall times below are
single-threaded. A threaded native baseline would be faster, which makes the
14.1x total a floor on the GPU-lane deficit, not an overstatement of it.

### 2026-08-24 controlled re-measurement: what is and is not still current

Six cases (coil-forces, boozerqa, wireframe-gsco-multistep, tracing-particle,
planar-coils, finitebuild) were re-run at HEAD on the same RTX 5090 box in a
fresh CUDA environment (jax/jaxlib 0.10.0, matching the authority lane's
recorded version), with same-environment runs at the authority revision as
controls to separate code change from environment change. Verdicts:

- **Precision: current.** Every re-measured GPU endpoint reproduces the
  final-result table to its printed precision (2e-13 to 7e-12 relative);
  the 2026-08-14 solver fusing/caching did not move physics.
- **Device memory: current.** Peak device bytes are identical to the table
  in every re-measured case.
- **Host memory: one real change.** The fused/cached lanes cost more host
  RSS at HEAD: the coil-forces GPU lane went 2041 -> 2328 MiB (+14%), with
  the authority revision reproducing the table's value in the same
  environment. Other re-measured cases sit within about 5% of the table.
- **GPU wall time: the table is environment-bound, not stale.**
  Same-environment controls put HEAD near parity with the authority
  revision (coil-forces 86.9 -> 84.5 s and tracing-particle 96.9 -> 94.9 s,
  both within 3%; boozerqa 108.6 -> 99.5 s, -8%). The large differences
  from the table (coil-forces 147.3 s there versus ~86 s here;
  tracing-particle 54.8 s there versus ~96 s here) appear at the **same
  revision** and run in both directions — they are environment effects.
  Consequence: the wall-time table above is valid only in its authority
  environment, and quoting its GPU numbers against any other environment
  or hardware is invalid.
- **JAX CPU wall time: no single-sample claim survives controls.** Within
  one environment, HEAD versus the authority revision moves in both
  directions (coil-forces 48.6 -> 41.3 s; boozerqa 26.9 -> 37.7 s), and
  the same lane differs by up to 40% between this pass's two replay
  environments at one revision. These one-shot launches cannot separate
  code effects from compile/environment variance; per-case receipts with
  repeated interleaved pairs (see the addendum) are the only wall-time
  evidence class that can.
- **finitebuild could not be re-measured through the harness:** at HEAD its
  lanes publish `initial:minimum_clearance` and `final:minimum_clearance`
  (added by `ead83eaef`, 2026-08-18) with no comparison routes in
  `parity_manifest.json`, so a full three-lane bounded replay fails closed
  ("complete direct lane-pair matrix") — reproduced on two machines. Until
  routes for those observables are added, only 24 of 26 cases can replay
  the full three-lane bounded matrix at HEAD (this case, plus the promoted
  single-stage route).

### Wall time at HEAD: the 2026-08-24 A100 three-lane run

This is the current-code counterpart to the authority wall-time table
above, measured under the same protocol — one isolated cold launch per
lane per case, startup and compilation included, native lane pinned to
`OMP_NUM_THREADS=1` — at revision `f5a3c9821` on a quiet dedicated host
(AMD EPYC 7452, NVIDIA A100-PCIE-40GB, glibc 2.31, jax/jaxlib 0.10.0).
It is a **different host and environment** from the authority table:
the two tables are each valid within their own environment, and their
totals (14.1x there, 9.47x here) are two measurements, not a trend.

| Case | Native CPU wall s | JAX CPU wall s | JAX GPU wall s | GPU/native |
|---|---:|---:|---:|---:|
| boozer | 3.79 | 19.38 | 28.46 | 7.51x |
| boozerqa | 3.23 | 56.37 | 108.73 | 33.62x |
| coil-forces | 12.37 | 59.96 | 117.23 | 9.47x |
| just-a-quadratic | 2.94 | 4.21 | 5.28 | 1.79x |
| minimize-curve-length | 2.95 | 5.93 | 9.58 | 3.25x |
| permanent-magnet-muse | 4.39 | 4.08 | 5.62 | 1.28x |
| permanent-magnet-pm4stell | 3.37 | 4.09 | 5.71 | 1.69x |
| permanent-magnet-qa | — | — | — | blocked |
| permanent-magnet-simple | 2.91 | 3.67 | 4.18 | 1.44x |
| qfm | 3.57 | 32.59 | 84.81 | 23.76x |
| stage-two-optimization | 8.61 | 44.03 | 79.15 | 9.19x |
| stage-two-optimization-finitebuild | — | — | — | blocked |
| stage-two-optimization-minimal | 3.29 | 20.07 | 49.33 | 15.01x |
| stage-two-optimization-planar-coils | 10.48 | 59.31 | 101.96 | 9.73x |
| stage-two-optimization-stochastic | 8.15 | 19.57 | 40.95 | 5.03x |
| strain-optimization | 8.65 | 9.99 | 24.26 | 2.81x |
| surf-vol-area | 2.99 | 18.63 | 40.67 | 13.62x |
| tracing-fieldlines-ncsx | 3.18 | 39.04 | 102.64 | 32.30x |
| tracing-fieldlines-qa | 3.18 | 38.30 | 100.44 | 31.61x |
| tracing-particle | 3.05 | 24.87 | 71.01 | 23.28x |
| wireframe-gsco-modular | 2.95 | 5.15 | 6.50 | 2.21x |
| wireframe-gsco-multistep | 3.05 | 4.34 | 5.99 | 1.96x |
| wireframe-gsco-sector-saddle | 2.98 | 5.05 | 5.83 | 1.96x |
| wireframe-rcls-basic | 3.14 | 3.76 | 4.75 | 1.51x |
| wireframe-rcls-with-ports | 3.26 | 3.94 | 5.01 | 1.54x |
| **Total (23 measured cases)** | **106.45** | **486.30** | **1008.09** | **9.47x** |

Scope of the 9.47x total, stated plainly: it is computed over a case set
that excludes one case by scale refusal
(`single-stage-boozer-vacuum-optimization`, promoted to `native_default`),
excludes two by the bugs documented above (permanent-magnet-qa's `.vtu`
side-writes; finitebuild's unrouted `minimum_clearance` observables), and
contains four cases with failing parity comparisons (quantified in the
re-validation note: absolute misses at most 5.5e-2, allclose overshoots
from 1.05x to 466x on one near-zero residual). The
per-case spread is 1.28x-33.62x; the launch-bound structure is the same
as the authority run's: cold XLA compilation dominates the JAX lanes at
bounded scale, so this table measures deployment cost of one-shot runs,
not device throughput. The GPU-favorable regime — warmed or
persistent-cache solves at the examples' own scales — is the addendum's
per-example certified evidence, and this table does not supersede it.

### Authority-run isolated end-to-end wall time

Parent RSS is the launched child's peak `VmHWM`; it excludes descendants.

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

GPU was slower in every launch.

> **Scope note (added 2026-08-24):** true of these one-shot cold launches,
> which include startup and compilation, and of nothing else. Under a
> different measurement class — warmed or persistent-cache solves at the
> examples' reference scales — two of the rows above are now certified GPU
> **wins** (`stage-two-optimization-finitebuild` 13.58x warmed solve,
> `wireframe-gsco-multistep` 3.5x device solve); see the addendum at the end
> of this report.

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

VMEC stays on CPU/MPI; only the JAX coil slice moves devices. The local run at
`e07a30635` left only VMEC working files
(`.artifacts/jax-authority-evidence/vmec-hybrid-e07a/`); its JSON stdout was
not retained, so no numbers are quoted here. A checksum-bound CPU/GPU receipt
is still owed.

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

> **Pin superseded (checked 2026-08-24).** That SHA-256 is the manifest as of
> the authority revision `11340c829`, and it is the value this report's results
> must be replayed against — it is correct here and must not be "refreshed."
> The **live** file no longer matches it: commit `3c6dfea62` (2026-08-05,
> "feat(bench): promote single-stage speed campaign") edited one route, so
> `examples/jax/parity_manifest.json` at HEAD hashes to
> `cd9e3c2d191e7934bd5240a9de3f2f7eb79d79e66789aaccf24ed9452588bf30`.
> The single change is `native-single-stage-boozer-vacuum-optimization`,
> `scale_tier: bounded` → `native_default` (plus its `comparison_routes`); no
> route was added or removed, and the other 26 declarations are byte-identical.
> Verify with
> `git diff 11340c829 HEAD -- examples/jax/parity_manifest.json`.
> In consequence the live manifest declares 1,260 applicable comparisons over
> 420 routes (192 exact) versus the 1,248/416/180 this report's verdict
> counts; the whole delta sits inside the promoted route's declarations
> (45 → 57). A bounded replay at HEAD also executes at most 25 of the 26
> cases: `run_parity.py` fails closed on the promoted route's scale tier, so
> the 26th case replays only at the authority revision (or at
> `native_default` scale). Separately, the finitebuild case's three-lane
> replay fails closed at HEAD on unrouted `minimum_clearance` observables
> (see the 2026-08-24 controlled re-measurement), leaving 24 of 26 cases
> fully replayable.

Limitations:

1. Bounded scale only; native-default is `not_run`. **Superseded for one route
   (2026-08-24):** this was true of the manifest at the authority revision and
   remains true of every measurement in this report, but the live manifest now
   declares `native-single-stage-boozer-vacuum-optimization` at
   `scale_tier: native_default` (see the pin note above). This report's rows for
   that example — lines in the initial-state, final-state, solver, and memory
   tables — were measured at the **bounded** tier and are not native-default
   evidence for it. Bounded scale is a design choice, not a shortcut:
   pinned budgets keep the three lanes' work identical and the 26-case
   matrix affordable to replay on any revision, where shipped-scale runs
   cost hours and amplify environment noise past comparability.
   Example-scale GPU evidence is certified per example instead — see the
   addendum and `docs/jax_example_device_assignment.md`.
2. Timing and memory are one sample per lane, imports and compilation
   included; no compile-only or warmed measurements, so no steady-state speed
   claim.
3. The cold/warm benchmark compares JAX fast to JAX parity, not JAX to native.
4. Near-zero relative differences are ill-conditioned; use absolute values and
   the declared tolerances.
5. No Perlmutter data.

## Bottom line

Every declared bounded route passes on all three lanes; coverage is exactly
what the parity manifest declares, nothing broader. No performance claim
survives this evidence: every isolated GPU launch was slower than native CPU,
and only field-line/particle tracing cleared every warmed fast-mode gate. A
quotable native-CPU/JAX-GPU speed comparison still needs cold, compile-only,
and warmed samples with RSS and VRAM in machine-readable receipts.

## Addendum — first native_default-scale GPU result (2026-08-15)

The bottom line above is a bounded-scale verdict and stands unchanged for the
`20260729T005942Z-5ade9aee` evidence it summarizes. On 2026-08-15 the
wireframe-gsco-multistep mirror produced the repository's first
native_default-scale native-CPU/JAX-GPU receipt with full-precision physics:
the final 19,200-segment currents vector is bitwise identical (0 ULP) between
the native example (`OMP_NUM_THREADS=32`) and the strict fp64 RTX 5090 lane,
and the warmed device solve is ~3.5x faster than the best measured native
configuration (5.77-5.93 s vs 20.49 s). Scope: this single mirror only; the
receipt and its qualifiers live in
`docs/receipts/wireframe_gsco_multistep_native_default_receipt.md`. The
bounded-scale rows above are unchanged.

## Addendum — per-example device assignment record (2026-08-16)

Per-example "run this on CPU or GPU" advice, with its evidence class and
mechanism for every manifest example, now lives in
[`docs/jax_example_device_assignment.md`](jax_example_device_assignment.md) and
is drift-gated by `tests/test_jax_example_device_assignment.py`.

## Addendum — native_default-scale results since 2026-08-16 (2026-08-24)

The bounded-scale verdict above stands unchanged for the
`20260729T005942Z-5ade9aee` evidence it summarizes, and **no newer aggregate
parity run exists**: that run is still the newest under
`.artifacts/jax-example-parity/`, so every parity, timing, and memory row above
is current as published. What follows is not a revision of those rows. It is a
different measurement class — native_default or reference scale, warmed or
persistent-cache, per example rather than aggregate — and it is summarized here
only so a reader of this report is not left with the bounded numbers as the
program's last word.

Why the numbers below cannot replace the numbers above: this report times **one
cold isolated launch per lane with imports and compilation included**
(Limitation 2), at bounded scale (Limitation 1). The results below are warmed
or persistent-cache solves at the examples' own reference scales. The aggregate
14.1x GPU-slower figure remains correct for what it measures — a launch-bound
artifact, not a device gap — and pairing it against a warm solve ratio compares
two different experiments.

Current per-example placements live in
[`docs/jax_example_device_assignment.md`](jax_example_device_assignment.md),
which is the SSOT for this and is drift-gated by
`tests/test_jax_example_device_assignment.py`. As of 2026-08-23 its 40 rows
read: 3 `gpu`/`measured-certified`, 2 `gpu`/`measured-diagnostic`, 4
`cpu`/`measured-diagnostic`, 21 `cpu`/`census-structural`, 10 `unmeasured`.
(30 of the 40 are *placed*; that record defines `unmeasured` as not placed.)

The three certified GPU wins, each with the scope its receipt states:

1. **`native-stage-two-optimization-finitebuild`** — 13.58x warmed solve
   (median of five pair ratios; the receipt's median native 45.23 s and median
   GPU 3.353 s divide to 13.49x — the 13.58x is the median of the per-pair
   ratios, not the ratio of the medians) and 3.11x warm persistent-cache wall
   (50.1 s vs 16.11 s) over the fastest qualifying swept native lane, five
   interleaved pairs per rung with every pair > 1.00 and every GPU endpoint
   bitwise identical to the frozen crossing solution. **Repeated-workload only:**
   a fresh empty-cache process loses 0.88x to ~42 s of XLA compile — measured,
   reported, and never claimed as a cold-start win.
   `docs/receipts/stage_two_finitebuild_native_gpu_successor.md`
2. **`native-wireframe-gsco-multistep`** — 3.5x warmed device solve
   (5.77-5.93 s vs 20.49 s best native), 19,200-segment currents vector bitwise
   identical (0 ULP). Already recorded in the 2026-08-15 addendum above.
   `docs/receipts/wireframe_gsco_multistep_native_default_receipt.md`
3. **`flat675-single-stage-coupled-optimization`** — the certified frozen-bundle
   configuration, reached through the example's `--bundle` mode: 1.67x at equal
   budget 3, **7.70x at equal budget 37** (headline) and 7.36x quality-matched,
   all on process wall, five interleaved pairs per rung with every pair > 1.00.
   The configuration the example ships by default carries no timing claim of its
   own. Repeated-workload only: the cold fused child pays ~150 s of XLA compile,
   measured at N=1. `docs/receipts/flat675_fused_campaign.md`

Two qualifiers a reader should carry out of this addendum. First, every
certified win above is a **warm or persistent-cache** claim, and none of the
three carries a cold-start claim. Their cold measurements differ and must not
be summarized as one: finite-build loses cold (0.88x, ~42 s compile) and GSCO
likewise, while flat675's cold results **split** — 0.47x at B3 but 1.53x at B37
and 1.54x at BQ, i.e. the fused lane wins cold at two of its three rungs
(`docs/receipts/flat675_fused_campaign.md:107-111`, all N=1, reported and never
claimed). Compile cost is disclosed in each receipt rather than amortized away.
Second, 10 of the 40 rows are `unmeasured` — including both
`wireframe-gsco` siblings, which two dated diagnostics of the *same shipped
configuration* place in opposite directions, so this record's own rule unplaces
them pending re-adjudication. Placement is per example and per scale; there is
still no repository-wide native-CPU/JAX-GPU speed claim, and the closing
paragraph above continues to state what one would require.

The bounded-scale rows above are unchanged.
