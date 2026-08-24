# SIMSOPT native CPU versus JAX CPU/GPU parity and performance

**Evidence date:** 2026-07-29

**Scope:** the 26 external-solver-free, one-to-one native-example mirrors

**Scientific precision:** FP64

**Authority revision:** `11340c829690fdc0652e47588f5da549829c056a`

**Authority run:** `20260729T005942Z-5ade9aee`

**Authority verdict:** 26/26 cases, 78/78 lane receipts, and 1,248/1,248
comparisons passed

## Executive summary

**Performance in one place:** "Wall time and GPU speed — one table
(2026-08-24)" — per example: cold three-lane wall times at HEAD on an
A100, the cold GPU-vs-native ratio, the enlarged-workload
GPU-vs-native ratio, and its confidence level.

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
  at HEAD on an A100 host puts the same protocol at 9.47x there — see the
  one-table section; two hosts, two environments, not a trend.)
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

> **HEAD re-validation (2026-08-24):**
>
> - **Planar-coils row falsified.** `fd200f564` (2026-08-02) changed the
>   problem (curvature target `"max"` → `identity`). At HEAD all lanes
>   converge: native 0.00221946821602, JAX CPU 0.00103232035072, GPU
>   0.00103939304979 (~53% below native). This row and the "16.3%" bullet
>   are stale.
> - **23 of 26 native endpoints reproduce** the table to printed precision.
> - **Environment sensitivity, not code drift** (identical at authority
>   revision and HEAD): pm-qa MwPGP endpoint 3.3e-5 off-table (gate
>   passes); fieldlines-ncsx misses 1 of 66 Poincaré crossings (0.041 vs
>   `atol` 0.030) — the only gate failure; trajectories bitwise identical
>   across revisions.
> - `single-stage-boozer-vacuum-optimization`: no bounded replay at HEAD
>   (fail-closed on its promoted scale tier).
> - **pm4stell repaired** (`aa04f698c`): the θ=π removal predicate is now
>   exact in both lanes; re-certified at full scale, FMA on, 163 dipoles
>   bitwise. The FMA-off diagnostic matched the JAX dump (0/5,826,
>   `b2c099489`) but with differing input hashes — suggestive, not causal.
>   Kernel rebuilt twice: `41b2ca79…` → `95190afa…` (predicate) →
>   `d4a6e028…` (exact branch only at literal π; earlier receipts valid
>   for literal-π inputs, near-π branch uncertified). Bounded rows
>   unaffected; PM/force/strain/curve suites re-ran green.
> - **A100 second-architecture leg** (EPYC 7452, glibc 2.31, jax 0.10.0):
>   25 of 26 cases, **1,108/1,122 comparisons pass**, 19/23 clean — not
>   comparable to the header's single-environment 1,248/1,248. All 14
>   failures quantified as exact allclose overshoots
>   (max |a−b| / (atol + rtol·|b|)): fieldlines-ncsx 1.60–1.83x (worst
>   5.5e-2 vs `atol` 3e-2); fieldlines-qa 1.05–1.41x, incl. a
>   **CPU-vs-CPU** pair at 1.05x — the divergence needs no GPU;
>   planar-coils 1.32x final / 5.26x first phase; rcls-with-ports
>   near-zero residual 466x (4.7e-10 absolute). Absolute misses small,
>   relative overshoots not — no "all marginal" summary survives.
>   Tracing = 10 of 14 failures yet cleared every warmed fast-mode gate;
>   chaotic FP amplification plausible, not proven.
> - Could not run there: finitebuild (route gap, Appendix B); pm-qa
>   (`.vtu` side-writes trip the integrity guard — lanes agree ~3e-6,
>   endpoint 16% off: endpoints do not transport between environments).
> - Scope: endpoints + parity gates + six GPU re-measurements with
>   authority-revision controls; timing/memory/precision in Appendix B.

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
note, 2026-08-24: since `91e1133b9` the Boozer cases can emit a seven-field
raw status and normalize capped endpoints to `budget_exhausted`, but only
when the endpoint certificate is enforced; the bounded parity path does not,
so bounded replays still reproduce the two-field `converged` statuses above —
verified 2026-08-24.)

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

## Wall time and GPU speed — one table (2026-08-24)

Wall-time columns: the 2026-08-24 run at HEAD (`f5a3c9821`) on a quiet
A100-PCIE-40GB / EPYC 7452 host (glibc 2.31, jax/jaxlib 0.10.0) — one
isolated **cold** launch per lane per case at bounded size, startup and
XLA compile included, native lane pinned to `OMP_NUM_THREADS=1`
(threaded native would be faster: the cold GPU deficits are floors).
One unrepeated subprocess per cell; warm numbers come from the
per-example receipts.

Both ratios state GPU speed as a multiple of native C++ (higher = GPU
faster): **cold** = native s / GPU s from this table; **enlarged** =
warm or realistic-scale receipt (bigger resolution, grid, filaments,
ensemble, or coupling DOFs). Per-row evidence:
`docs/jax_example_device_assignment.md`. † = cold ratio from the case's
own 5090 receipt (blocked in / absent from this run).

| Example | Native CPU s | JAX CPU s | JAX GPU s | GPU vs native, cold default size | GPU vs native, enlarged workload | Confidence |
|---|---:|---:|---:|---:|---|---|
| stage-two-optimization-finitebuild | — | — | — | 0.9x † | **13.6x** warm solve | certified |
| flat675 single-stage | — | — | — | 0.5x † | **7.7x** at budget 37 | certified |
| wireframe-gsco-multistep | 3.05 | 4.34 | 5.99 | 0.51x | **3.5x**, bitwise | certified |
| permanent-magnet-simple | 2.91 | 3.67 | 4.18 | 0.70x | **5.2x**, bitwise | measured once |
| wireframe-gsco-modular | 2.95 | 5.15 | 6.50 | 0.45x | **5.2x** at 96x100 grid | measured once |
| wireframe-gsco-sector-saddle | 2.98 | 5.05 | 5.83 | 0.51x | **4.4x** at 96x100 | measured once |
| permanent-magnet-pm4stell | 3.37 | 4.09 | 5.71 | 0.59x | — pre-repair 3.0x not comparable | unmeasured — predicate repaired `aa04f698c`, parity re-certified; timing rung pending |
| permanent-magnet-muse | 4.39 | 4.08 | 5.62 | 0.78x | **2.9x** at nphi=64 | measured once |
| coil-forces | 12.37 | 59.96 | 117.23 | 0.11x | **1.6x** warm | measured once |
| stage-two-optimization-stochastic | 8.15 | 19.57 | 40.95 | 0.20x | **1.2-1.4x**, grows with ensemble | measured once |
| projected-route single-stage | — | — | — | — | only device that finishes the script | measured once |
| stage-two-optimization | 8.61 | 44.03 | 79.15 | 0.11x | 0.3x warm — native faster | measured once |
| stage-two-optimization-planar-coils | 10.48 | 59.31 | 101.96 | 0.10x | 0.3x warm — native faster | measured once |
| wireframe-rcls-with-ports | 3.26 | 3.94 | 5.01 | 0.65x | 0.6x device solve — native faster | measured once |
| permanent-magnet-qa | — | — | — | — | plausible at nphi=64 | blocked (`.vtu` bug) |
| wireframe-rcls-basic | 3.14 | 3.76 | 4.75 | 0.66x | 0.6x at native_default — still too small | measured once |
| tracing-fieldlines-ncsx | 3.18 | 39.04 | 102.64 | 0.03x | — GPU OOM at reference scale (native 78 s, JAX CPU 518 s) | blocked (GPU memory) |
| tracing-fieldlines-qa | 3.18 | 38.30 | 100.44 | 0.03x | — GPU OOM at reference scale (native 171 s, JAX CPU 253 s) | blocked (GPU memory) |
| tracing-particle | 3.05 | 24.87 | 71.01 | 0.04x | — GPU OOM at reference scale (native 109 s, JAX CPU 1435 s) | blocked (GPU memory) |
| boozer | 3.79 | 19.38 | 28.46 | 0.13x | stays on CPU — too small to fill a GPU | settled |
| boozerqa | 3.23 | 56.37 | 108.73 | 0.03x | stays on CPU | settled |
| just-a-quadratic | 2.94 | 4.21 | 5.28 | 0.56x | stays on CPU | settled |
| minimize-curve-length | 2.95 | 5.93 | 9.58 | 0.31x | stays on CPU | settled |
| qfm | 3.57 | 32.59 | 84.81 | 0.04x | stays on CPU | settled |
| stage-two-optimization-minimal | 3.29 | 20.07 | 49.33 | 0.07x | stays on CPU | settled |
| strain-optimization | 8.65 | 9.99 | 24.26 | 0.36x | stays on CPU | settled |
| surf-vol-area | 2.99 | 18.63 | 40.67 | 0.07x | stays on CPU | settled |
| single-stage-boozer-vacuum-optimization | — | — | — | — | excluded: runs at `native_default` scale only | excluded |
| all other device-assignment rows (12 of 40) | — | — | — | — | no at-scale GPU claim | settled |
| **Total (23 measured cases)** | **106.45** | **486.30** | **1008.09** | **0.11x** | — | — |

- "Certified" = sealed multi-run receipt, verified physics; "measured
  once" = one dated measurement.
- One rule: the GPU wins once a step carries ~1e7 elements of parallel
  work (resolution, segments, filaments, ensembles, coupling DOFs — not
  nested solves).
- 2026-08-24 A100 probes (diagnostic, `native_default`, cold):
  rcls-basic 0.6x; all three tracing GPU lanes OOM in
  `build_regular_grid_interpolant_3d`
  (`src/simsopt_jax_adapters/field/interpolated.py`, 29–60 GiB) before
  tracing begins — blocked by construction memory, not disproven on
  speed. Fix: chunked/streamed interpolant build.
- Cold total (0.11x) excludes single-stage (scale refusal), pm-qa, and
  finitebuild (Appendix B bugs); 4 included cases have failing
  comparisons (Appendix B). Cold = one-shot deployment cost
  (compile-dominated); enlarged = device throughput; neither supersedes
  the other.
- The A100 and Appendix A's 5090 are different environments — 9.47x and
  14.1x are two measurements, not a trend. Memory: peak-memory section +
  Appendix A. VMEC-hybrid: no claim-grade metrics.

### Appendix A — archived authority wall times (2026-07-29, RTX 5090)

The pinned original measurement this report is built on — superseded for
current-code reading by the one-table above, retained for replay and as
the only per-case memory record. 78 launches, 29 m 21 s, same protocol,
on the RTX 5090 / Threadripper 9970X box, native lane pinned to
`OMP_NUM_THREADS=1` (`examples/jax/parity/runtime.py`) — so the 14.1x
total is a floor on the GPU deficit.

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

### Appendix B — 2026-08-24 controlled re-measurement: what is and is not still current

Six cases re-run at HEAD on the 5090 (fresh CUDA env, jax 0.10.0), with
same-environment authority-revision controls to separate code from
environment:

- **Precision: current.** Every re-measured GPU endpoint reproduces the
  table (2e-13 to 7e-12 relative); solver fusing/caching moved no physics.
- **Device memory: current.** Peak device bytes identical everywhere.
- **Host memory: one change.** coil-forces GPU lane 2041 → 2328 MiB
  (+14%, code-caused; controls confirm); others within ~5%.
- **GPU wall time: environment-bound, not stale.** HEAD ≈ authority
  revision within the same env (±3–8%), while the same revision swings
  ±77% across envs, both directions. The Appendix A GPU numbers are valid
  only in their authority environment.
- **JAX CPU wall time: no single-sample claim survives controls** (moves
  both directions; up to 40% between envs at one revision). Only repeated
  interleaved pairs (addendum receipts) are claim-grade wall-time
  evidence.
- **finitebuild: replay fails closed at HEAD** — `minimum_clearance`
  observables (`ead83eaef`) have no manifest routes; reproduced on two
  machines. 24 of 26 cases fully replayable until routes are added.

### Appendix C — JAX GPU fast versus JAX GPU parity (not a native comparison)

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

> **Pin superseded (2026-08-24):**
>
> - That SHA-256 is the manifest at authority revision `11340c829` —
>   replay against it; do not "refresh" it.
> - Live file: `cd9e3c2d…bf30`. Sole change: `3c6dfea62` promoted the
>   single-stage route `bounded` → `native_default` (routes 45 → 57;
>   other 26 declarations byte-identical).
> - Live totals 1,260 / 420 / 192 vs this report's 1,248 / 416 / 180.
> - At HEAD: bounded replay runs ≤ 25 of 26 cases (promoted tier), and
>   finitebuild fails closed on unrouted observables (Appendix B) —
>   24 of 26 fully replayable.

Limitations:

1. Bounded scale only; native-default is `not_run`. (Still true of every
   measurement here; the live manifest has since promoted the single-stage
   route to `native_default` — see the pin note — and this report's rows for
   that example remain bounded-tier evidence.) Bounded scale is a design
   choice: pinned budgets keep the three lanes' work identical and the
   matrix cheap to replay on any revision. Example-scale GPU evidence is
   certified per example instead (addendum;
   `docs/jax_example_device_assignment.md`).
2. Timing and memory are one sample per lane, imports and compilation
   included; no compile-only or warmed measurements, so no steady-state speed
   claim.
3. The cold/warm benchmark compares JAX fast to JAX parity, not JAX to native.
4. Near-zero relative differences are ill-conditioned; use absolute values and
   the declared tolerances.
5. No Perlmutter data.

## Bottom line

- Every declared bounded route passes on all three lanes; coverage is
  exactly what the manifest declares.
- No performance claim from this evidence: every isolated GPU launch was
  slower than native; only tracing cleared every warmed fast-mode gate.
- A quotable native-vs-GPU speed comparison needs cold, compile-only, and
  warmed samples in machine-readable receipts — delivered per example in
  the addendum below.

## Addendum — per-example device assignment record (2026-08-16)

Per-example CPU-or-GPU placement, with evidence class and mechanism, lives
in [`docs/jax_example_device_assignment.md`](jax_example_device_assignment.md)
(drift-gated by `tests/test_jax_example_device_assignment.py`).

## Addendum — native_default-scale results (2026-08-24)

A different measurement class — warmed or persistent-cache, per example, at
reference scale. It cannot replace the cold bounded rows above, and no newer
aggregate parity run exists. The 40-row assignment record reads (2026-08-23):
3 gpu certified, 2 gpu diagnostic, 4 cpu diagnostic, 21 cpu structural,
10 unmeasured.

The three certified GPU wins (each **warm/persistent-cache only** — cold
loses or splits, disclosed in each receipt):

1. **finitebuild** — **13.58x** warmed solve, 3.11x persistent-cache wall;
   endpoints bitwise; cold 0.88x (~42 s compile).
   `docs/receipts/stage_two_finitebuild_native_gpu_successor.md`
2. **wireframe-gsco-multistep** — **3.5x** warmed device solve
   (5.77–5.93 s vs 20.49 s best native); currents bitwise (0 ULP).
   `docs/receipts/wireframe_gsco_multistep_native_default_receipt.md`
3. **flat675 single-stage** (`--bundle`) — **7.70x** at equal budget 37
   (1.67x at budget 3, 7.36x quality-matched), process wall, five
   interleaved pairs, every pair > 1.00; cold splits (0.47x B3 / 1.53x
   B37 / 1.54x BQ, N=1). `docs/receipts/flat675_fused_campaign.md`

10 of 40 rows remain `unmeasured` (incl. both `wireframe-gsco` siblings,
unplaced after two diagnostics disagreed). Placement is per example and
per scale; there is no repository-wide native-vs-GPU speed claim. The
bounded-scale rows above are unchanged.
