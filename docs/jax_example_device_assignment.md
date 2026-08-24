# JAX example device assignment

Which device each `examples/jax` example should be *run* on for speed, and what
evidence backs that choice. One row per manifest example, no exceptions.

## What this record is — and is not

This is a **performance assignment advisory**. It answers "if I want this
example to finish fastest, where do I launch it?" and nothing else.

It is **not a capability gate**. Every example that declares a GPU scope in
`examples/jax/manifest.json` must keep working on GPU, and parity
certification runs must keep exercising their GPU lanes regardless of what the
`device` column says. A `cpu` row means "the GPU lane is slower here", never
"do not run the GPU lane".

The record exists because the alternative is worse: the same measurement gets
re-litigated every time someone asks "should this run on the GPU?", and the
answer lives in a session transcript or a host-local campaign directory
instead of the repository. `tests/test_jax_example_device_assignment.py`
keeps the table honest — it fails if an example is added to the manifest
without an assignment, if a row names an example that does not exist, if a
`cpu` row does not open with one of the three mechanism families, if a `gpu`
row does not cite a **git-tracked file under `docs/receipts/`**, if the device
and evidence-class columns contradict each other, if any cited in-repo path is
missing, or if this document's legends stop enumerating exactly the values the
test accepts.

## Where the evidence lives

The point of this record is to stop device knowledge living in session
transcripts. That promise is only kept if the record is honest about which
evidence a reader can actually open, so every claim below carries its
provenance:

- **In-repo.** Reviewable from a clone, and drift-gated by the test:
  `docs/receipts/wireframe_gsco_multistep_native_default_receipt.md`,
  `docs/receipts/wireframe_gsco_siblings_native_default.md`,
  `docs/receipts/projected_route_example_promotion.md`,
  `docs/single_stage_jax_gpu_projected_route_certification_plan.md`,
  `docs/jax_porting_progress_report.md`, the example sources under
  `examples/jax/` that every `census-structural` row is read from, and — new
  on 2026-08-23 — the 171 backlog-probe artifacts committed at `fbab4f2b8` under
  `docs/receipts/evidence/` with their shared executed-order ledger
  `docs/receipts/evidence/probe_leg_ledger.jsonl` (the dated `*_20260823.*` glob
  also matches the concurrent nested-LS campaign's gate artifacts, which are not
  probe evidence and carry no diagnostic label). Those 172 files are
  **diagnostic-not-certifying by their own label**: each timed leg publishes
  per-leg identity (commit, observed OMP, JAX platform and x64 state, device
  list, concurrent GPU processes) — except the quartet probes' unmodifiable
  native children, which publish parent-side identity and disclose
  `child_identity_available: false`; native denominators are OMP-swept where a
  sweep ran (single-lane GPU artifacts carry none); endpoint dumps are full
  precision except the quartet's native objective, printed at two significant
  digits — enough to place a row, never enough to certify one.
- **Host-local campaign artifacts** — *not* in this repository, not reviewable
  from a clone, not reproducible without this workstation. Marked
  **[host-local]** at every use: `~/simsopt-campaigns/winnable-six-20260815/`
  (44-leg `receipt.json`, `silicon_probe_results.json`),
  `~/simsopt-campaigns/ndparity-boozer-vacuum-20260814/`, and
  `~/simsopt-campaigns/gsco-siblings-20260816/` (the two 2026-08-16
  wireframe-GSCO sibling legs, box-state records and parity captures), and
  `~/simsopt-campaigns/projected-route-example-promotion-20260816/` (the
  2026-08-17 projected-route example campaign: 41 leg JSONs, of which **37**
  carry a whole-leg sampled contention record — the sampler was added after the
  plumbing and first probe legs had run, and two of the four unsampled legs feed
  published statistics, which its receipt §7 names; plus
  `artifacts/receipt_numbers.json` binding every headline numeral of its receipt
  to a source file and pointer).
- **Session-audit classification** — the 2026-08-13 audit's grouping of the
  mirrors into winnable / never-winnable / unmeasured classes. It has **no
  artifact of any kind**; it is a reasoned classification, and this document
  is its first written record. Marked **[session-audit]** at every use. Where
  it disagrees with a measurement, the measurement wins and the row is
  `unmeasured`.

## The mechanism law

The assignments below are **workload-shape facts, not hardware caveats**.
Three statements carry all of them:

1. **Wide batched work with large reductions goes to the GPU.** The kernels
   compile to memory-bound code, so bandwidth decides, and the fp64 1:64 cap
   does not bite on this class. **[host-local:
   `~/simsopt-campaigns/winnable-six-20260815/silicon_probe_results.json`,
   2026-08-14, fp64, median of 30 warm reps]** the RTX 5090 matches an
   A100-PCIE-40GB on the Biot-Savart-shaped value-and-gradient kernel
   (0.661 ms vs 0.680 ms) and beats it on a 472 MB GEMV (0.62 ms vs 0.97 ms).
   Where a mirror loses on the 5090, it loses on any GPU for the same reason.
2. **Narrow sequential dependency chains stay on the CPU.** When each step
   depends on the last and the per-step work is small, kernel launch latency
   dominates. **[host-local: same probe]** on a 1000-step dependent chain the
   Threadripper 9970X CPU beats the 5090 by 5.7x host-driven and 75x fused,
   and beats the A100 by more. No GPU wins this shape. Field-line and particle
   tracing, nested Boozer inner Newton solves, and 3-to-30-DOF toy problems
   are all this shape.
3. **The reduction dimension and the per-step work volume decide, not
   arithmetic intensity.** The two measured members of the same greedy
   fixed-matrix family split on exactly this. **[host-local:
   `~/simsopt-campaigns/winnable-six-20260815/receipt.json`, 2026-08-15,
   self-labelled diagnostic-not-certifying, commit 2e6166505]** GSCO multistep
   reduces over 1024 plasma-surface rows against a 19,200-segment wireframe
   and wins 3.5x at 99% device utilization and 442 W — this one is also
   certified in-repo, see the receipt on its row; GPMO MUSE reduces over 256
   rows in tiny per-candidate dot products and loses 4.05x at 69-73%
   utilization and only 120 W. **The 2026-08-23 retime keeps the law and moves
   the numbers**: at the same shipped 16x16 MUSE scale the loss is 0.64x, not
   4.05x, and at nφ=64 — 4,096 reduction rows over 7,530 dipoles — the same
   variant *wins* 2.9x bitwise. Reduction width still decides; what the older
   pair of numbers hid is that the crossover sits inside this one family, so
   every GPMO row below is stated at a named scale and none of them generalizes
   to another.

**Contention rider.** **[host-local:
`~/simsopt-campaigns/winnable-six-20260815/receipt.json`]** on a shared box the
native OpenMP lanes collapse under third-party load — 50x (GSCO) to 228x
(GPMO MUSE) at 1-minute loadavg ~240 — because each greedy iteration
fork/joins a small parallel region, while the device-resident JAX lanes
degrade only 1.2-2.1x. Every `cpu` assignment below is a *quiet-box*
assignment; on a contended host the GPU lane is robustness-preferred even
where it is slower at rest. Note also that the shipped native examples set no
thread count, so an unset `OMP_NUM_THREADS` lands on the pathological
all-hardware-threads configuration (3.2x for GSCO, 12x for GPMO on a 64-thread
box, same source) — a native baseline measured that way is not the baseline
these assignments compare against.

**Scope guard.** Running many instances at once — a batch of particles, a
sweep of surfaces, an ensemble of coil sets — is a *different workload*, not a
faster mirror. Batched-instance execution can be an excellent GPU fit even for
rows marked `cpu` here. This table says nothing about it.

## Device values

The `device` column holds exactly one of these four, and nothing else:

- **cpu** — the CPU lane is faster; the row states which mechanism family
  makes the GPU lose.
- **gpu** — the GPU lane is faster, evidenced by a git-tracked receipt file
  under `docs/receipts/` named on the row.
- **either** — reserved for a row with *measured parity* between the lanes. No
  row carries it today; on a contended host the contention rider above would
  break such a tie toward the GPU.
- **unmeasured** — not placed. Either nothing was measured, or the available
  evidence points both ways.

## Evidence-class legend

- **measured-certified** — a committed receipt in this repository establishes
  the assignment at `native_default` scale, with physics agreement resolved.
- **measured-diagnostic** — a timed `native_default`-scale comparison exists
  and is self-labelled diagnostic-not-certifying; the direction is measured,
  the magnitude is dated and conditional.
- **census-structural** — no `native_default` timing for this example; the
  assignment follows from the mechanism law applied to a workload shape read
  directly out of the example source (problem size, dependency structure,
  reduction width).
- **unmeasured** — not grounded. Either no evidence, or the available evidence
  points both ways. These rows are an invitation to measure, not a
  recommendation.

## Mechanism families

A `cpu` row's mechanism cell must **open** with one of exactly these three
family names, so that the reason a row is on the CPU cannot be buried, negated,
or implied:

- **sequential chain** — each step depends on the previous one and the
  per-step work is too small to cover a kernel launch.
- **tiny problem** — the whole problem is fixed and small at every execution
  scale, so there is nothing to parallelize.
- **narrow matrix** — the reduction dimension of the fixed response matrix is
  too short to fill the device.

## Assignment table

Bounded-scale timings are excluded by construction, and **no row below is
derived from one**. In-repo, `docs/jax_porting_progress_report.md` states the
verdict twice: its bottom line, "every isolated GPU launch was slower than
native CPU", and its Limitation 2, "one sample per lane, imports and
compilation included; no compile-only or warmed measurements, so no
steady-state speed claim". Bounded runs are dominated by startup, compile, and
uncacheable tracing; they are launch-bound artifacts, not performance
evidence. **[session-audit]** aggregating that run's 26 mirrors gives 14.1x
GPU-slower overall, which is the size of the artifact, not of any device gap.

| Example ID | Device | Evidence class | Mechanism / receipt |
| --- | --- | --- | --- |
| traceable-least-squares | cpu | census-structural | tiny problem — three independent weighted residuals, fixed size at every scale |
| curve-length-optimization | cpu | census-structural | tiny problem — order-2 CurveXYZFourier, 15 DOF, no native_default branch |
| surface-geometry-optimization | cpu | census-structural | tiny problem — mpol=1 / ntor=0 axisymmetric torus, fixed size |
| coil-flux-optimization | cpu | census-structural | tiny problem — one free coil current against an mpol=1 / ntor=0 surface |
| qfm-surface-optimization | cpu | census-structural | tiny problem — bounded mpol=1 / ntor=1 QFM surface, fixed size |
| permanent-magnet-optimization | cpu | census-structural | tiny problem — two-dipole fixed-state problem on a 1x6 quadrature grid |
| fieldline-and-particle-tracing | cpu | census-structural | sequential chain — ODE event walk, one dependent step per integration substep |
| boozer-surface-optimization | cpu | census-structural | sequential chain — inner Boozer Newton solve on an mpol=1 / ntor=1 surface |
| wireframe-optimization | cpu | census-structural | tiny problem — 4x6 toy ToroidalWireframe |
| coil-force-and-finite-build | cpu | census-structural | tiny problem — order-1 finite-build coil, fixed size at every scale |
| single-stage-vacuum-optimization | unmeasured | unmeasured | manifest status is `planned` and the script is absent from the tree |
| projected-route-single-stage-boozer-vacuum-optimization | gpu | measured-diagnostic | wide coupled projected route — the GPU is the only device this script's own success criterion was met on. Interleaved A/B at native_default: on the CPU backend all three protocol attempts end `LINE_SEARCH_COLLAPSE` (the reported final attempt ran 74 iterations; the artifact does not publish the first two attempts' counts) and the run publishes `retry_exhausted` after 1294.4 s, while on GPU 8/8 legs reach `OBJECTIVE_TARGET_REACHED` at iteration 399 with a bitwise-identical 716-coordinate endpoint; matched truncated budgets put one CPU-backend iteration at 8.33x one GPU iteration (a mild upper bound — the matched CPU legs pinned 16 threads and the one uncapped datapoint is ~2% faster). Read with its bar: the *native two-stage mirror* reaches the same endpoint quality in 31.3 s (OMP=8), so this script is not the fast way to this physics — **5.14x** at matched process wall and **8.48x** to matched endpoint using this row's own warm median (265.6 s, 6 legs), or 5.07x and 8.37x pooling the 8 warm+cold GPU legs (pooled median 262.1 s) as the receipt's boundary table does. All seven pre-registered endpoint-physics gates pass through the native evaluator — `docs/receipts/projected_route_example_promotion.md` |
| native-just-a-quadratic | cpu | census-structural | tiny problem — three independent DOFs |
| native-minimize-curve-length | cpu | census-structural | tiny problem — order-4 CurveRZFourier, 8 free DOF |
| native-permanent-magnet-simple | gpu | measured-diagnostic | wide fixed-matrix GPMO — warm baseline solve **5.2x** at this mirror's own `native_default` (16x16 half-period grid, `downsample=4`, 14,336 dipoles, a 256 x 43,008 response matrix, K=500): GPU 0.0306 s against the swept-native optimum 0.1586 s at OMP=32. Six native legs sweep OMP 2/4/8/16/32/48 (1.21 / 0.79 / 0.35 / 0.20 / 0.159 / 6.07 s — the OMP=48 leg is this box's pathological collapse, disclosed and never the denominator) against two interleaved GPU legs; all eight legs publish the same SHA-256 for the placed moment array, so the endpoint is bitwise identical (0 ULP) — `docs/receipts/evidence/pm_simple16_jaxgpu_4_20260823.json` and `docs/receipts/evidence/pm_simple16_native_omp32_20260823.json`. Both are self-labelled diagnostic-not-certifying: one date, one box, cold in-process 0.10-0.46 s reported and not claimed. Certification is chartered as Rung A of `docs/jax_gpu_pm_gpmo_campaign_plan.md` (draft) |
| native-qfm | unmeasured | unmeasured | no native_default timing; the port also swaps SLSQP for an augmented-Lagrangian solve, so a timing would not be matched work |
| native-stage-two-optimization-minimal | unmeasured | unmeasured | no native_default timing for the stage-two coil family |
| native-surf-vol-area | cpu | census-structural | tiny problem — mpol=1 / ntor=0 surface, two sequential scalar targets |
| native-tracing-fieldlines-ncsx | cpu | census-structural | sequential chain — sequential ODE event walk; [session-audit] never-winnable class |
| native-tracing-fieldlines-qa | cpu | census-structural | sequential chain — sequential ODE event walk; [session-audit] never-winnable class |
| native-tracing-particle | cpu | census-structural | sequential chain — sequential ODE event walk; [session-audit] never-winnable class |
| native-boozer | cpu | census-structural | sequential chain — host-side stage-1 loop around a latency-serialized inner Boozer solve |
| native-boozerqa | cpu | census-structural | sequential chain — one inner Newton solve per outer evaluation |
| native-permanent-magnet-muse | cpu | measured-diagnostic | narrow matrix — 256-row (16x16) GPMO reduction at the scale this mirror ships. The 4.05x figure this row used to carry [host-local: `~/simsopt-campaigns/winnable-six-20260815/receipt.json`] is **superseded**: the 2026-08-23 matched-work retime puts the GPU at 0.64x here — warm 4.71 s against the swept-native optimum 3.01 s at OMP=32 (sweep 4/8/16/32 = 16.76 / 8.35 / 4.25 / 3.01 s), endpoints bitwise identical (0 ULP, one moments SHA-256 over six legs) and measured post-cond-skip, so the frozen-step objection to the old number is discharged and the loss is 1.6x rather than 4x — `docs/receipts/evidence/muse_shipped_native_omp32_20260823.json`. The direction is scale-dependent and this row is the shipped scale: at nφ=64, which the source's own comment names for high-resolution runs, the same variant puts the GPU **2.9x ahead** bitwise (7.32 s vs 21.22 s at OMP=32) — a configuration no mirror ships, outside this row, chartered as Rung B of `docs/jax_gpu_pm_gpmo_campaign_plan.md` (draft) |
| native-permanent-magnet-pm4stell | unmeasured | unmeasured | Measured fast and **blocked on physics**, which is not a placement. At nφ=64 (the high-resolution configuration, not this mirror's shipped N=16) the 2026-08-23 probe timed the GPU 3.0x faster — warm 9.51 s vs 28.7 s at OMP=32 — but the greedy selection **forks inside the first dewyrming sweep** — run by the last iteration, k=200, of the 201-iteration probe (the `(k % nBacktracking) == 0` gate has no warm-up guard) — from digest-identical inputs (same `A_obj` and `b_obj` SHA-256): 10 of 5,826 moment rows differ, five exactly-antiparallel removal pairs, the placed-dipole count splits 133 native / 139 GPU, and each lane stays bitwise stable across its own repeats. The mechanism is adjudicated (2026-08-23 static phase; **replay-CONFIRMED 2026-08-24**: a separate `-ffp-contract=off` native rebuild reproduces the JAX endpoint bitwise — 139 placed, 0 of 5,826 rows differ — `docs/receipts/evidence/pm4stell64_fork_k201_native_ffpoff_20260824.npz`): an FMA-contraction difference — native's `-ffp-contract=fast` local build against XLA's uncontracted dot — at the equality-grade removal test cos <= cos(pi) = -1.0, every deciding pair within 0-2 ULP of -1; no semantic difference between the lanes, and the CI native build (`-march=westmere`, no FMA) would side with the GPU, so the charter adopts an exact-arithmetic predicate repair (a solver change, requiring parity re-certification before any timing). Both endpoint states are archived — `docs/receipts/evidence/pm4stell64_fork_k201_native_20260823.npz`. Speed over a trajectory the lanes do not share is not a device recommendation; this is the blocked rung of `docs/jax_gpu_pm_gpmo_campaign_plan.md` (draft); **the exact-predicate repair LANDED and lane-parity re-certified 2026-08-24** (FMA-on native rebuild and 5090 JAX lane bitwise-identical at the fork scale, 163 placed each — §Blocked rung repair record), and the row stays unplaced until a chartered timing rung runs under the frozen instrument |
| native-permanent-magnet-qa | unmeasured | unmeasured | Half-measured and fail-closed. At nφ=64 (the full-resolution configuration the source comments name, not this mirror's shipped nφ=16) the 2026-08-23 probe sized the relax-and-split grid at 29,286 dipoles and 2.88 GB of staged arrays and timed the native lane at ~32.4 s, OMP-insensitive between 16 and 32 (32.43 s and 32.39-34.23 s) — `docs/receipts/evidence/qa64_native_omp32_solve_20260823.json`. The JAX lane never produces a comparable number: it fail-closes on the MwPGP step-size bound (the `_mwpgp_spec` validator in `src/simsopt_jax/solve/permanent_magnet.py`), and that refusal is adjudicated (2026-08-23, P3.5 of the backlog plan) as a **false reject from the probe's own staging**: the probe rescales the grid in place (native `rescale_for_opt` folds 1/nu into `ATA_scale`), computes the explicit alpha from the shifted scale — an alpha *inside* the true bound by its 1e-5 margin — then stages the shifted grid into `PermanentMagnetGridJAX.from_cpu`, whose validator re-applies the shift and rejects a legitimate step. No step-rule divergence exists: with the un-rescaled grid staged, both lanes' default formulas coincide exactly and the validator passes the same alpha it rejected. The refusal is now reproduced with operands archived (2026-08-24, `docs/receipts/evidence/qa64_jaxgpu_solve_refusal_20260824.log`: alpha 1837891.6017776239 vs bound 1837741.1007418663, +8.2e-5 relative — the adjudicated double-shift signature; the original failed leg wrote no artifact and the cited JSON holds only the native half); the production validator is vindicated and **the instrument-side staging fix is applied 2026-08-24** (probe stages the raw grid before the host rescale; the leg now completes rc=0, all-finite endpoint over 29,286 dipoles, `docs/receipts/evidence/qa64_jaxgpu_solve_20260824.json` — diagnostic only, no denominator) — `docs/jax_gpu_pm_gpmo_campaign_plan.md` (draft) puts `permanent_magnet_QA` / relax-and-split explicitly out of scope for timing |
| native-stage-two-optimization | cpu | measured-diagnostic | sequential chain — 400 host-driven L-BFGS steps whose per-step coil objective is too small to cover the launch. Shipped-vs-shipped 2026-08-23: **0.30x**, warm mirror solve 34.5 s against the native minimize region's best leg 10.4 s (six fresh native processes at OMP 4/8/16, 10.4-11.9 s). The ratio mixes optimizer policy (mirror `maxcor` 300 vs the native script's 10) with a disclosed 1.33x quadrature asymmetry (the mirror pins `numquadpoints=100`, the script takes the 75-point default, so the GPU lane does 1.33x more Biot-Savart work per evaluation), and the GPU window also carries host construction and the Taylor test that the native denominator excludes; all three corrections run **for** the GPU and none of them closes a 3.3x gap — `docs/receipts/evidence/quartet_stage_two_native_omp8_20260823.json` |
| native-stage-two-optimization-planar-coils | cpu | measured-diagnostic | sequential chain — same shape and same shipped-vs-shipped comparison as the stage-two mirror above: **0.33x**, warm mirror solve 15.3 s against the best native leg 5.0 s. The native legs are noisy (5.0-39.1 s over six fresh processes at OMP 4/8/16), so the denominator is the best leg and not a median — the *worst* native leg would have minted a false 2.6x GPU win. The same policy, 1.33x quadrature and timed-window disclosures apply and all run for the GPU — `docs/receipts/evidence/quartet_planar_native_omp16_20260823.json` |
| native-stage-two-optimization-stochastic | unmeasured | unmeasured | A 2026-08-23 diagnostic measures the GPU **ahead** — warm 1.35x at `maxcor=10` (22.15 s vs 29.99 s at OMP=16) and 1.24x at `maxcor=400` (24.62 s vs 30.53 s), both at a matched nit=400 with one bitwise-shared training-sample set injected into both lanes and endpoints inside the `native_workflow` tolerance bucket; cold in-process costs 157-214 s of XLA compile, and the sample-tile lever measured neutral-to-negative (24.94-25.21 s vs 24.62 s untiled). Above the charter's bar, but the native side is N=1 per OMP value and OMP 2/48 were never run, which is not a denominator this record places a row on. Certification is chartered as R1/R2 of `docs/jax_gpu_stochastic_stage_two_campaign_plan.md` (draft); the row moves on that campaign's receipt, not on this probe — `docs/receipts/evidence/stoch_jaxgpu_mc400_a_20260823.json` |
| native-strain-optimization | cpu | census-structural | tiny problem — 21 DOF (rotation order 10), host-bound by construction |
| native-wireframe-gsco-modular | unmeasured | unmeasured | **Two dated diagnostics point opposite ways**, so this record's own rule unplaces the row. The sealed 2026-08-16 receipt measured warm GPU device solve 0.552 s vs 0.492 s best native (0.89x) at shipped scale — `docs/receipts/wireframe_gsco_siblings_native_default.md`. The 2026-08-23 probe re-measured the same shipped configuration and came back **reversed**, ~1.6x *for* the GPU: warm median 0.408 s vs the sampled-native optimum 0.669 s at OMP=32. Two things moved at once and neither half may be kept alone — the GPU lane got faster (0.408 vs 0.552 s) *and* the OMP=48 legs are pathological on this box in this session (33.9 s where the receipt measured 0.492 s and called OMP=48 what a native user should run), a ~69x native regression at the receipt's own best configuration that is a suspected defect until proven otherwise. Currents bitwise identical (0 ULP) on both dates — `docs/receipts/evidence/gsco_modular_shipped_native_omp32_20260823.json`. At the *reference* configuration both sources name in dead comments (96x100, 19,200 segments, 20,000 iterations) the same probe measured **5.2x** GPU bitwise; that is a different configuration and does not place this row. Both are chartered in `docs/jax_gpu_gsco_siblings_reference_scale_campaign_plan.md` (draft), whose R3 rung re-adjudicates this shipped-scale conflict |
| native-wireframe-gsco-sector-saddle | unmeasured | unmeasured | **Two dated diagnostics point opposite ways**, so this record's own rule unplaces the row. The sealed 2026-08-16 receipt measured warm GPU device solve 0.653 s vs 0.518 s best native (0.79x) at shipped scale — `docs/receipts/wireframe_gsco_siblings_native_default.md`. The 2026-08-23 probe re-measured the same shipped configuration and came back **reversed**, ~1.35x *for* the GPU: warm 0.388 s vs the sampled-native optimum 0.524 s at OMP=32. Same two-sided movement as the modular sibling — a faster GPU lane *and* an OMP=48 leg that costs 30.1 s here against the receipt's 0.518 s, a ~58x native regression at the receipt's own best configuration and a suspected defect until proven otherwise. Currents bitwise identical (0 ULP) on both dates — `docs/receipts/evidence/gsco_sector_saddle_shipped_native_omp32_20260823.json`. At the *reference* configuration the source names in dead comments (96x100, 20,000 iterations, `break_width=4`, `gsco_cur_frac=0.03`) the same probe measured **4.4x** GPU bitwise; different configuration, does not place this row. Both are chartered in `docs/jax_gpu_gsco_siblings_reference_scale_campaign_plan.md` (draft), whose R4 rung re-adjudicates this shipped-scale conflict |
| native-wireframe-rcls-basic | unmeasured | unmeasured | single dense regularized least-squares solve; [host-local: silicon probe] the dense-solve crossover sits between n=169 (CPU 2.4x) and n=716 (GPU ~5x), and this problem has not been placed against it |
| native-wireframe-rcls-with-ports | cpu | measured-diagnostic | narrow matrix — 497 free segments under 254 port constraints, a 1521 x 243 augmented least-squares system, below the dense-solve crossover [host-local: silicon probe] that sits between n=169 and n=716. Matched solve window 2026-08-23: warm GPU device solve 0.035 s vs 0.020 s native (medians; **0.57x**, a kill under the probe's own warm rule), native swept OMP 2/4/8/16 with the optimum at 8; the whole `optimize_wireframe` window is 1.08x for the GPU and still under the campaign bar, and the two lanes' objectives agree to 1.4e-14 relative against the OMP=8 denominator leg (bit-identity is not expected — the device path reassociates the same algebra) — `docs/receipts/evidence/marginal_rcls_native_omp8_20260823.json` |
| native-coil-forces | unmeasured | unmeasured | A 2026-08-23 diagnostic measures the GPU **~1.6x ahead** — warm mirror solve 24.5 s against the native minimize region's best leg 40.0 s, out of six noisy legs spanning 40-82 s at OMP 4/8/16 — and the mirror's endpoint objective is the better of the two (2.77e-5 vs 2.9e-5). Not a placement: this is shipped-vs-shipped, so the ratio carries the same unmatched-policy and timed-window disclosures as the stage-two rows, and the native denominator's own spread (2.0x) is wider than the gap it is being asked to prove. Chartered as the R3 rung of `docs/jax_gpu_stochastic_stage_two_campaign_plan.md` (draft) — `docs/receipts/evidence/quartet_coil_forces_native_omp16_20260823.json` |
| native-single-stage-boozer-vacuum-optimization | cpu | census-structural | sequential chain — latency-serialized inner Boozer Newton per outer evaluation; [session-audit] never-winnable class in this nested formulation; native_default lane artifacts [host-local: `~/simsopt-campaigns/ndparity-boozer-vacuum-20260814/`] |
| native-single-stage-optimization | cpu | census-structural | sequential chain — VMEC equilibrium and its finite-difference derivatives run host/MPI-serial; the manifest gives this example a `jax_slice_only` GPU scope |
| native-stage-two-optimization-finitebuild | gpu | measured-certified | 13.58x warm solve (45.23 s vs 3.353 s) and 3.11x warm persistent-cache process wall (50.1 s vs 16.11 s) over the fastest qualifying native lane (omp2-h400, swept optimum), five interleaved pairs each, every pair > 1.00, endpoint quality oracle-verified under the successor v4 contract and every GPU endpoint bitwise-identical to the frozen crossing solution. Repeated-workload/persistent-cache win only: a fresh-empty-cache process loses 0.88x to the ~42 s XLA compile (measured, reported, no cold-start claim) — `docs/receipts/stage_two_finitebuild_native_gpu_successor.md` |
| native-wireframe-gsco-multistep | gpu | measured-certified | 3.5x warmed device solve (5.77-5.93 s vs 20.49 s best native) with a bitwise-identical 19,200-segment currents vector — `docs/receipts/wireframe_gsco_multistep_native_default_receipt.md` |
| flat675-single-stage-coupled-optimization | gpu | measured-certified | The evidence class describes the CERTIFIED FROZEN-BUNDLE CONFIGURATION, which this example reaches through its `--bundle` mode: the sealed campaign measured that configuration, from the single archived start candidate, at 1.67x (equal budget 3), 7.70x (equal budget 37, headline) and 7.36x (quality-matched), all on process wall, five interleaved pairs per rung with every pair > 1.00 and all quality gates green. The configuration this example SHIPS BY DEFAULT runs the same production fused lane on repository geometry and carries no timing claim of its own — it is placed here by the certified configuration's device, not by a measurement of itself. Repeated-workload win only: the cold fused child pays the full XLA compile (~150 s), measured at N=1, disclosed and never claimed — `docs/receipts/flat675_fused_campaign.md` |

## Summary counts

40 manifest examples: **5 gpu**, **25 cpu**, **0 either**, **10 unmeasured**.

Restricted to the 27 `native-*` mirrors: 3 gpu, 15 cpu, 9 unmeasured.

The 15 `cpu` mirrors are 10 of the 2026-08-13 session audit's 11 never-winnable
mirrors (3 tracing, 3 nested-Boozer, and 4 of the audit's 5 tiny/fixed-size —
`native-permanent-magnet-simple` is the fifth and left the family on 2026-08-23:
its `native_default` scale turned out to build a 256-row reduction over 14,336
dipoles rather than a fixed 2x2 grid, and it is now placed `gpu` on a measured
5.2x), plus four rows placed on measured losses rather than structure —
`native-permanent-magnet-muse` (0.64x at shipped scale),
`native-wireframe-rcls-with-ports` (0.57x on the matched solve window),
`native-stage-two-optimization` and
`native-stage-two-optimization-planar-coils` (0.30x and 0.33x shipped-vs-shipped)
— and `native-single-stage-optimization` (VMEC host lane, `planned`, outside the
26 measured). The two 2026-08-16 wireframe-GSCO siblings **left this group on
2026-08-23**: a second dated diagnostic reversed their direction, so they are
`unmeasured` rather than `cpu`.

All four mirrors the 2026-08-13 session audit classed as winnable are still
`unmeasured`, and after 2026-08-23 each one names a *specific* reason rather than
a missing number. `native-permanent-magnet-pm4stell` was timed 3.0x faster on the
GPU and is blocked on a greedy fork at k=201 — speed over a trajectory the lanes
do not share. `native-permanent-magnet-qa` has a native time and no GPU
counterpart at all: that lane fail-closes on the MwPGP step-size bound.
`native-wireframe-gsco-modular` and `native-wireframe-gsco-sector-saddle` were
`cpu` here between 2026-08-16 and 2026-08-23 and are now unplaced: the second
diagnostic reversed the direction, and this record does not choose between two
dated measurements that disagree. Their reference-scale configuration, where the
same probe measured 5.2x and 4.4x, is a different workload and places nothing.

`native-permanent-magnet-simple` is the one row the 2026-08-23 evidence *closed*:
the audit classed it never-winnable on a workload shape that was a mirror defect,
the two readings that replaced it disagreed, and a matched-work A/B at the real
`native_default` now puts the GPU 5.2x ahead with a bitwise-identical endpoint.
Three `native-*` rows are `unmeasured` for the older reason — no timing at all
(`native-qfm`, `native-stage-two-optimization-minimal`,
`native-wireframe-rcls-basic`) — and two more,
`native-stage-two-optimization-stochastic` and `native-coil-forces`, are
`unmeasured` while holding a diagnostic that points *at* the GPU (1.35x/1.24x and
~1.6x): above the bar, under-replicated, and chartered rather than placed.

## Scope note and amendment procedure

These assignments were derived at the **2026-08-13 / 2026-08-16 evidence
state**: the 2026-08-13 six-agent `examples/jax` audit, the 2026-08-14
three-device silicon probe, the 2026-08-15 winnable-six campaign, the
2026-08-16 full-precision GSCO promotion, and the 2026-08-16 GSCO-siblings
campaign. Hardware: RTX 5090 plus Threadripper
9970X, with an A100-PCIE-40GB cross-check on the kernel probe only.

**Benchmarks-path campaign receipts (2026-08-19; example row added
2026-08-19, see the log).** Two sealed campaign receipts measure the
flat-675 formulation. When they were recorded the formulation had no
`examples/jax` mirror and therefore no row above; F4/C2 added
`flat675-single-stage-coupled-optimization`, whose row cites the second of
them under the scope law stated in that row — the receipts certify the
frozen-bundle configuration, not the repository-geometry configuration the
example ships by default. The timed instruments remain `benchmarks/`
harnesses, and the fused lane's production module is
`src/simsopt_jax/examples/single_stage_flat675.py`:
`docs/receipts/genuine675_fair_bar.md` — the July host-loop instrument at
8.07x (B3) and 25.87x (B37) on `process_wall_seconds` (10.33x on the
archived claim's own optimizer-wall timer at B3) — and
`docs/receipts/flat675_fused_campaign.md` — the production fused lane at
1.67x / 7.70x / 7.36x (B3 / B37 / quality-matched BQ), also
`process_wall_seconds`, which supersedes the archived flat-675 "9.8x" as
the citable fused production number while the host-loop lane remains
faster at those budgets on that timer. Timers are named because the
example rows above use different ones (device-solve wall for GSCO;
warm-solve and process wall for finite-build); none of these figures are
mutually commensurate without their timer.

**2026-08-23 backlog probe (nine families, rows amended in the log below).**
A single-day probe pass measured nine example families against swept-OMP native
denominators and committed its artifacts in-tree under
`docs/receipts/evidence/` with a shared executed-order ledger. Every artifact is
self-labelled `diagnostic-not-certifying`, so the pass may move a row to
`measured-diagnostic` and may unplace a row, and may **not** mint
`measured-certified`: it moved one row to `gpu` (5.2x, bitwise), three rows to
`cpu` on measured losses, restated one `cpu` row whose 4.05x figure it
superseded with 0.64x, and unplaced the two wireframe-GSCO siblings whose
direction it reversed. Three certifying campaign charters were drafted out of it
— `docs/jax_gpu_pm_gpmo_campaign_plan.md`,
`docs/jax_gpu_stochastic_stage_two_campaign_plan.md` and
`docs/jax_gpu_gsco_siblings_reference_scale_campaign_plan.md`, all tracked back
to `docs/jax_gpu_examples_backlog_native_speed_implementation_plan.md` — and
**a charter is not a receipt**: no row below moves on one.

To amend, **append a dated row to the log below and edit the table row it
refers to in the same commit**. Do not rewrite or delete log entries; a
superseded assignment stays visible as the entry that superseded it. Moving a
row to `gpu` requires committing a receipt first: the consistency test accepts
the row only if it names a path that `git ls-files` reports as a tracked
regular file under `docs/receipts/`, so a host-local campaign directory, an
untracked file, or a directory will not satisfy it. The same rule gates the
`measured-certified` evidence class.

| Date | Example ID | Change | Authority |
| --- | --- | --- | --- |
| 2026-08-16 | (all) | Record created at the 2026-08-13 / 2026-08-16 evidence state | `docs/receipts/wireframe_gsco_multistep_native_default_receipt.md` and the 2026-08-13 audit |
| 2026-08-16 | native-wireframe-gsco-modular | unmeasured / census-structural → cpu / measured-diagnostic. Ten-round interleaved A/B: warm GPU device solve 0.552 s vs 0.612 s fair native (OMP=32) and 0.492 s best native (OMP=48) — 1.11x and 0.89x on the kernel, 1.01x and 0.89x on the numerical region. Placed `cpu` rather than `either` because best-configured native wins, a cold JAX process loses 2.75x, and the warm advantage needs a persistent compile cache nothing configures by default. Currents vector bitwise identical (0 ULP over 88 native x 15 GPU legs). The 10.3x lead over the shipped `OMP_NUM_THREADS`-unset default is the 64-thread OpenMP collapse on this box, not a GPU win. | `docs/receipts/wireframe_gsco_siblings_native_default.md` |
| 2026-08-16 | native-wireframe-gsco-sector-saddle | unmeasured / census-structural → cpu / measured-diagnostic. Ten-round interleaved A/B: warm GPU device solve 0.653 s vs 0.560 s fair native (OMP=32) and 0.518 s best native (OMP=48) — 0.86x and 0.79x on the kernel, 0.85x and 0.81x on the numerical region. Currents vector bitwise identical (0 ULP over 88 native x 14 GPU legs). Same shipped-default caveat as the modular sibling (7.0x). | `docs/receipts/wireframe_gsco_siblings_native_default.md` |
| 2026-08-17 | projected-route-single-stage-boozer-vacuum-optimization | unmeasured / unmeasured → gpu / measured-diagnostic. First timing of the script itself (the row previously recorded only that the certified 2.304x belongs to the benchmarks-path root run). Placed `gpu` on this record's own semantics — where to launch *this* example to finish fastest — because the CPU backend does not finish it at all: all 3 protocol attempts end `LINE_SEARCH_COLLAPSE` (the reported final attempt ran 74 iterations; the first two attempts' counts are not published by the example) and it reports `retry_exhausted` after 1294.4 s, while 8/8 GPU legs reach the objective target at iteration 399 with one bitwise-identical endpoint digest, and matched truncated budgets price one CPU-backend iteration at 8.33x one GPU iteration. The row states its bar in the same cell: the native two-stage mirror at its measured thread optimum (OMP=8, swept over 2/4/8/16/32/unset) beats this script 5.14x at matched process wall and 8.48x to matched endpoint quality on the row's own warm median (5.07x and 8.37x on the receipt's pooled warm+cold median), so the GPU placement is a device recommendation and never a formulation recommendation. Endpoint physics equivalence passes seven pre-registered gates on two independent endpoint pairs, both recomputed through the native evaluator; the campaign also recorded that a strict reuse of the native inner solve's 1e-13 stopping rule would have false-rejected the route by 94x at a measured objective cost of 1.08e-19. | `docs/receipts/projected_route_example_promotion.md` |
| 2026-08-17 | native-stage-two-optimization-finitebuild | reason updated; device stays unmeasured / unmeasured. The dedicated GPU speed route closed `CLOSED_BOUNDED_NEGATIVE` on its preregistered Step-3 kill: the fused GPU lane crossed the frozen objective rung (h10-b560 converged 0.52% below the 1.001x target with every quality cap and geometry band clean) but the endpoint it publishes failed the gradient infinity-norm landing clause (1.98 vs the 1.05 cap — disclosed pre-evidence as a landing condition the reference's own converged endpoint fails at 2.08), and the kill is final at budget parity (b <= 800). 23 of 24 native configurations fail the same contract. No native-vs-GPU solve-time comparison was produced (selection never froze), so no device recommendation is minted; the 13.03x warm value/grad kernel canary is a kernel measurement, not a solve claim. | `docs/receipts/stage_two_finitebuild_native_gpu.md` |
| 2026-08-18 | native-stage-two-optimization-finitebuild | unmeasured / unmeasured → gpu / measured-certified. The successor campaign (new preregistration, symmetric first-crossing endpoints + window-median gradient clause) produced the comparison its predecessor could not: five interleaved pairs, warm solve 13.58x median (13.01–13.96, native omp2-h400 45.23 s vs GPU h10/k*=500 3.353 s) and warm persistent-cache process wall 3.11x median (2.90–3.24), every pair > 1.00 on both timers, every GPU endpoint bitwise-identical to the frozen crossing solution and oracle-gated under the frozen v4 contract (freeze audit passed; the native denominator landed within 1% of the charter's pre-registered estimate). Fresh-empty-cache measured separately and lost (0.88x, ~42 s XLA compile): the row certifies a repeated-workload/persistent-cache win only. Pairs ran clean-tree at 66003ee45 in a pinned worktree; the shipped module reproduces the crossing solution bitwise at the production commit (see receipt disclosure). | `docs/receipts/stage_two_finitebuild_native_gpu_successor.md` |
| 2026-08-19 | flat675-single-stage-coupled-optimization | Row created: **gpu / measured-certified**. The flat-675 formulation gained its first `examples/jax` mirror in F4/C2 (`examples/jax/3_Advanced/single_stage_flat675.py`), so the same-dated scope-note entry below — written when the formulation had no example row — is superseded on that one point and stands otherwise. The class is scoped in the row itself and is NOT extended past the receipt: the sealed campaign measured the frozen-bundle configuration at one archived start (1.67x / 7.70x / 7.36x process wall at budgets 3 / 37 / quality-matched), and the example reaches exactly that configuration through `--bundle`. The clone-runnable default builds the same production fused lane from repository test-file geometry and is not timed here or anywhere; it inherits the device recommendation, not the number. Cold start stays a disclosure: the fused child pays ~150 s of XLA compile on first solve, N=1, reported not claimed. | `docs/receipts/flat675_fused_campaign.md` |
| 2026-08-19 | (scope note) | Recorded two sealed benchmarks-path campaign receipts for the flat-675 formulation, which has no example mirror and changes no assignment row: the fair-bar re-adjudication of the archived flat-675 "9.8x" (host-loop instrument, 8.07x B3 / 25.87x B37 process wall; the archived claim survives strengthened) and the F3 fused production campaign (1.67x / 7.70x / 7.36x process wall, superseding the 9.8x as the citable fused production number; host-loop remains faster at those budgets on that timer). Timers named in the scope note; example assignment counts unchanged. | `docs/receipts/genuine675_fair_bar.md` and `docs/receipts/flat675_fused_campaign.md` |
| 2026-08-23 | native-permanent-magnet-simple | cpu / census-structural → **unmeasured / unmeasured**; no timing is minted here. Two things changed together. (1) The mirror was defective: `solve()` accepted an execution-scale argument and discarded it, so `--smoke` and the native default both built the bounded 2x2 / `downsample=100` grid, and the old cell ("tiny problem — fixed 2x2 quadrature grid (4 reduction rows), no native_default branch") described that defect rather than the example. The mirror now branches on that argument, aligning native_default with the values its parity case had already frozen (`examples/jax/parity/cases/native_permanent_magnet_simple.py`): 16x16 quadrature, `downsample=4`, 14,336 dipoles, a 256 x 43,008 response matrix, K=500 — and the iteration budgets are now pinned to that case by test rather than being a second independent literal. Bounded scale is byte-for-byte unchanged in its physics; only the published observable shape changed, from the full (ndipoles, 3) moment array to the placed rows, their dipole indices, the row count, and a SHA-256 of the full array (the native_default JSON payload was 244 KB of ~96% zeros). (2) With the real workload in view the `tiny problem` family no longer applies, and the replacement is not `narrow matrix` on the MUSE read: MUSE's 4.05x GPU loss times ArbVec-backtracking, a different algorithm variant whose frozen-step skip postdates that number, while the only direct measurement of *this* mirror's baseline kernel — a host-local N=3 diagnostic dated 2026-07-26 — put the GPU ~2.0x faster at the same 16x16 workload. Two uncommitted, unmatched readings that disagree are not a placement, so the row is unplaced until the pre-registered matched-work A/B (P3.2) runs. | `examples/jax/1_Simple/permanent_magnet_simple.py`, `examples/jax/parity/cases/native_permanent_magnet_simple.py`, and `docs/jax_gpu_examples_backlog_native_speed_implementation_plan.md` |
| 2026-08-23 | native-permanent-magnet-simple | unmeasured / unmeasured → **gpu / measured-diagnostic**. The matched-work A/B pre-registered as P3.2 ran and settled the conflict the same-dated entry above left open, in the GPU's favour: warm baseline solve 0.0306 s against the swept-native optimum 0.1586 s (OMP=32) at the mirror's own `native_default` (16x16, `downsample=4`, K=500) — **5.2x**. Six native legs (OMP 2/4/8/16/32/48) against two interleaved GPU legs, all eight publishing one moments SHA-256, so the endpoint is bitwise identical (0 ULP). The OMP=48 leg costs 6.07 s and is disclosed as this box's collapse, never used as the denominator. The 2026-07-26 host-local ~2.0x reading is superseded by this one, and the MUSE 4.05x reading that pointed the other way is superseded on its own row. Diagnostic only: one date, one box, self-labelled diagnostic-not-certifying; certification is Rung A of the GPMO charter draft. | `docs/receipts/evidence/pm_simple16_jaxgpu_4_20260823.json`, `docs/receipts/evidence/pm_simple16_native_omp32_20260823.json` and `docs/jax_gpu_pm_gpmo_campaign_plan.md` |
| 2026-08-23 | native-permanent-magnet-muse | Cell rewritten; device and class unchanged (`cpu` / `measured-diagnostic`). The 4.05x figure is **superseded**: a matched-work retime at the shipped nφ=16 scale measures 0.64x (warm 4.71 s vs the swept-native optimum 3.01 s at OMP=32, sweep 4/8/16/32), endpoints bitwise identical over six legs, and post-cond-skip — so the objection that the old number predated the frozen-step `lax.cond` skip is discharged and the loss is 1.6x, not 4x. The placement does not change; its magnitude does. Recorded in the same cell: at nφ=64 the same variant wins 2.9x bitwise, which is a configuration this mirror does not ship and this row does not cover (Rung B of the GPMO charter draft). | `docs/receipts/evidence/muse_shipped_native_omp32_20260823.json` and `docs/receipts/evidence/muse64_native_omp32_20260823.json` |
| 2026-08-23 | native-wireframe-rcls-with-ports | unmeasured / unmeasured → **cpu / measured-diagnostic**. The dense-solve crossover question this row was holding is answered: n_free=497 with 254 port constraints sits below it. Matched solve window, warm GPU device solve 0.035 s vs 0.020 s native (medians; **0.57x**, a kill under the probe's own warm rule; native swept OMP 2/4/8/16, optimum at 8); the whole `optimize_wireframe` window is 1.08x for the GPU and still under the campaign bar, so no charter follows. Objectives agree to 1.4e-14 relative against the OMP=8 denominator leg — bit-identity is not expected here, the device path reassociates the same equality-constrained least squares. `native-wireframe-rcls-basic` is a smaller problem of the same shape and stays `unmeasured`: it was not run. | `docs/receipts/evidence/marginal_rcls_native_omp8_20260823.json` and `docs/receipts/evidence/marginal_rcls_jaxgpu_a_20260823.json` |
| 2026-08-23 | native-stage-two-optimization | unmeasured / unmeasured → **cpu / measured-diagnostic**. First timing for the stage-two coil family: shipped-vs-shipped **0.30x**, warm mirror solve 34.5 s against the native minimize region's best leg 10.4 s (six fresh native processes, OMP 4/8/16, 10.4-11.9 s). Three asymmetries are disclosed and all three run *for* the GPU — optimizer policy (mirror `maxcor` 300 vs native 10), 1.33x more Biot-Savart work per evaluation on the mirror (`numquadpoints=100` vs the 75-point default), and a GPU window that includes host construction and the Taylor test the native denominator excludes. None of them closes a 3.3x gap, so the placement is safe in the direction it is made. | `docs/receipts/evidence/quartet_stage_two_native_omp8_20260823.json` and `docs/receipts/evidence/quartet_stage_two_jaxgpu_a_20260823.json` |
| 2026-08-23 | native-stage-two-optimization-planar-coils | unmeasured / unmeasured → **cpu / measured-diagnostic**. Same comparison and same disclosures as the stage-two mirror: **0.33x**, warm mirror solve 15.3 s against the best native leg 5.0 s. The native legs are noisy (5.0-39.1 s over six fresh processes at OMP 4/8/16) and the denominator is the best leg, not a median — taking the worst leg would have minted a 2.6x GPU win out of native variance, which is the failure mode this record's OMP law exists to prevent. | `docs/receipts/evidence/quartet_planar_native_omp16_20260823.json` and `docs/receipts/evidence/quartet_planar_jaxgpu_a_20260823.json` |
| 2026-08-23 | native-stage-two-optimization-stochastic | Cell rewritten; device and class unchanged (`unmeasured` / `unmeasured`). A diagnostic now exists and it points *at* the GPU — warm 1.35x at `maxcor=10` (22.15 s vs 29.99 s at OMP=16) and 1.24x at `maxcor=400` (24.62 s vs 30.53 s), matched nit=400, one bitwise-shared training-sample set injected into both lanes, endpoints inside the `native_workflow` bucket, cold in-process 157-214 s of XLA compile, sample-tile lever neutral-to-negative. The row stays unplaced anyway: N=1 native rep per OMP value and OMP 2/48 never run is not a denominator. Chartered as R1/R2; the row moves on that campaign's receipt. | `docs/receipts/evidence/stoch_jaxgpu_mc400_a_20260823.json`, `docs/receipts/evidence/stoch_native_mc400_omp16_20260823.json` and `docs/jax_gpu_stochastic_stage_two_campaign_plan.md` |
| 2026-08-23 | native-coil-forces | Cell rewritten; device and class unchanged (`unmeasured` / `unmeasured`). The finite-build coil-force family has its first timing: GPU **~1.6x** ahead, warm mirror solve 24.5 s against the native minimize region's best leg 40.0 s of six noisy legs spanning 40-82 s (OMP 4/8/16), with the mirror's endpoint objective the better of the two (2.77e-5 vs 2.9e-5). Unplaced because it is shipped-vs-shipped with the same unmatched-policy and timed-window disclosures as the stage-two rows, and because a native denominator whose own spread is 2.0x cannot carry a 1.6x claim. Chartered as the R3 rung of the stochastic campaign draft. | `docs/receipts/evidence/quartet_coil_forces_native_omp16_20260823.json` and `docs/jax_gpu_stochastic_stage_two_campaign_plan.md` |
| 2026-08-23 | native-wireframe-gsco-modular | cpu / measured-diagnostic → **unmeasured / unmeasured**. Not a correction of the 2026-08-16 receipt and not an adoption of the new probe: **two dated diagnostics of the same shipped configuration point opposite ways**, and this record's own rule for that is `unmeasured`. Sealed 2026-08-16: 0.552 s GPU vs 0.492 s native (0.89x). Probe 2026-08-23: 0.408 s (median) GPU vs 0.669 s at OMP=32 (~1.6x *for* the GPU). Two things moved together and neither half may be kept alone — the GPU lane got faster, *and* OMP=48, the configuration the receipt called what a native user should run, costs 33.9 s here against its 0.492 s, a ~69x native regression that is a suspected box defect until proven otherwise. Currents bitwise identical (0 ULP) on both dates, so the disagreement is timing only. Reference scale (96x100, 20,000 it) measured 5.2x GPU bitwise and is a different configuration; the charter draft carries both, with R3 re-adjudicating this shipped-scale conflict. | `docs/receipts/wireframe_gsco_siblings_native_default.md`, `docs/receipts/evidence/gsco_modular_shipped_native_omp32_20260823.json`, `docs/receipts/evidence/gsco_modular_reference_jaxgpu_a_20260823.json` and `docs/jax_gpu_gsco_siblings_reference_scale_campaign_plan.md` |
| 2026-08-23 | native-wireframe-gsco-sector-saddle | cpu / measured-diagnostic → **unmeasured / unmeasured**, on the same two-dated conflict as its modular sibling and by the same rule. Sealed 2026-08-16: 0.653 s GPU vs 0.518 s native (0.79x). Probe 2026-08-23: 0.388 s GPU vs 0.524 s at OMP=32 (~1.35x *for* the GPU). Both sides moved — faster GPU lane, and an OMP=48 leg at 30.1 s against the receipt's 0.518 s, a ~58x native regression at the receipt's own best configuration. Currents bitwise identical (0 ULP) on both dates. Reference scale measured 4.4x GPU bitwise and places nothing here; R4 of the charter draft re-adjudicates the shipped scale. | `docs/receipts/wireframe_gsco_siblings_native_default.md`, `docs/receipts/evidence/gsco_sector_saddle_shipped_native_omp32_20260823.json`, `docs/receipts/evidence/gsco_sector_saddle_reference_jaxgpu_a_20260823.json` and `docs/jax_gpu_gsco_siblings_reference_scale_campaign_plan.md` |
| 2026-08-23 | native-permanent-magnet-pm4stell | unmeasured / census-structural → **unmeasured / unmeasured**; the device does not move. The class does, because the old cell's structural read — "the same 256-row reduction as MUSE" — is no longer why this row is unplaced, and a `native_default`-scale timing now exists. At nφ=64 the GPU is 3.0x faster (9.51 s vs 28.7 s at OMP=32), but the greedy selection **forks at exactly k=201**, one past the example's `nBacktracking = 200`, from digest-identical inputs: the lanes agree through k=200, then 10 of 5,826 moment rows differ and the placed count splits 133 native / 139 GPU, each lane internally bitwise stable. Fork dumps archived on both lanes. Speed over an unshared trajectory is not a recommendation; this is the GPMO charter's blocked rung. Note the timing is at nφ=64, not this mirror's shipped N=16. | `docs/receipts/evidence/pm4stell64_fork_k201_native_20260823.npz`, `docs/receipts/evidence/pm4stell64_native_omp32_20260823.json` and `docs/jax_gpu_pm_gpmo_campaign_plan.md` |
| 2026-08-23 | native-permanent-magnet-qa | unmeasured / census-structural → **unmeasured / unmeasured**; the device does not move. As with `pm4stell`, the structural rationale is retired because half of a measurement now exists: at nφ=64 the native relax-and-split lane runs ~32.4 s and is OMP-insensitive between 16 and 32, on a grid sized here at 29,286 dipoles and 2.88 GB of staged arrays. The other half does not exist — the JAX lane fail-closes on the MwPGP step-size bound, the native formula 2(1-1e-5)/ATA_scale exceeding 2/lambda_max(H) by ~1e-4 relative in the failed leg's ValueError (that leg wrote no artifact, so the operands are unarchived; the cited JSON holds only the native half). Which step rule the shared work uses is a matched-work design decision, and it is chartered nowhere: the GPMO charter draft puts relax-and-split out of scope by name, so the item is parked at P3.5 of the backlog plan. Note the timing is at nφ=64, not this mirror's shipped nφ=16. | `docs/receipts/evidence/qa64_native_omp32_solve_20260823.json` and `docs/jax_gpu_examples_backlog_native_speed_implementation_plan.md` |
| 2026-08-23 | (scope note) | Append-only correction record for commit `6ccaa28f7`: a doc review that re-derived every quoted number from the probe artifacts corrected three same-dated log entries above **in place**, hours after their creating commit — `native-wireframe-rcls-with-ports` (0.034 s → 0.035 s, both cells now medians, matching the 0.57x they always implied), `native-wireframe-gsco-modular` (warm 0.405 s → median 0.408 s, ~1.65x → ~1.6x; the ~65x OMP=48 regression → ~69x, 65 was the sector denominator; reference 5.3x → 5.2x, the only ratio a consistent statistic pair supports), and `native-permanent-magnet-qa` (the MwPGP step-bound operands re-attributed to the failed leg's unarchived ValueError — the cited JSON holds only the native half). The matching assignment cells were edited in the same commit. In-place editing deviated from this procedure's append-only rule; this row restores the trail, and the originals remain in git at `a91295194`. No device or evidence class moved. | `docs/jax_gpu_gsco_siblings_reference_scale_campaign_plan.md`, `docs/jax_gpu_pm_gpmo_campaign_plan.md` and `docs/jax_gpu_stochastic_stage_two_campaign_plan.md` (the same review's charter fixes) |
| 2026-08-23 | native-permanent-magnet-pm4stell | Cell rewritten; device and class unchanged (`unmeasured` / `unmeasured`). The pre-registered two-hypothesis fork adjudication completed its static phase: the fork lives inside the first dewyrming sweep (H2's site) as a 0–2 ULP near-tie broken differently per lane (H1's kind of cause, by FMA contraction rather than the reduction order H1 posited) — the native local build FMA-contracts the 3-term cosine (`-ffp-contract=fast`) where XLA does not, at this case's equality-grade `cos <= cos(pi) = -1.0` removal test. The 201 placements entering the sweep are identical in both lanes; the split is 34-vs-31 removed antiparallel pairs (three direct threshold flips, two cascades); `muse-64` cannot fork because `nAdjacent=1` makes its sweep a structural no-op. Neither lane is buggy, and the CI native build (`-march=westmere`, no FMA3) would agree with the GPU — so "match native" is ill-defined and the charter adopts the pre-registered exact-arithmetic predicate repair (solver change → parity re-certification before any timing), with a confirming `-ffp-contract=off` native replay pre-registered for a quiet box. This entry also corrects the earlier entry's k-phrasing: the fork is at the sweep run by the last iteration k=200, not "one past" `nBacktracking`. Row stays unplaced. | `docs/jax_gpu_pm_gpmo_campaign_plan.md` (§Blocked rung, adjudication record), `docs/receipts/evidence/pm4stell64_fork_k201_native_20260823.npz` and `src/simsoptpp/permanent_magnet_optimization.cpp` |
| 2026-08-23 | native-permanent-magnet-qa | Cell rewritten; device and class unchanged (`unmeasured` / `unmeasured`). The MwPGP step-bound refusal reported in the earlier entry is adjudicated as a **false reject from the probe's staging, not a step-rule divergence**: the probe calls native `rescale_for_opt` (which folds `1/nu` into `ATA_scale` in place), computes its explicit alpha from the shifted scale — inside the true contraction bound by the formula's own 1e-5 margin — then stages the already-shifted grid through `PermanentMagnetGridJAX.from_cpu`, whose `_mwpgp_spec` validator (contracted on the raw spectral scale) re-applies the shift and rejects a legitimate step. The observed ~1e-4 gap fits both a missing-shift and a double-shift reading; the code discriminates, and it is the double shift. Consequence: staged un-rescaled, both lanes' default step rules coincide exactly, so the "matched-work step-rule decision" the earlier entry called for does not exist; the production validator is vindicated (it caught a real instrument inconsistency fail-closed) and the repair is instrument-side, owned by the future QA charter. Full chain with file:line cites at P3.5 of the backlog plan. Row stays unplaced. | `docs/jax_gpu_examples_backlog_native_speed_implementation_plan.md` (P3.5 adjudication), `src/simsopt/geo/permanent_magnet_grid.py` and `src/simsopt_jax/solve/permanent_magnet.py` |
| 2026-08-24 | native-permanent-magnet-qa | Operand archival for the entry above: the pre-registered qa-64 jax-gpu leg rerun reproduced the MwPGP refusal fail-closed (rc=1) and the ValueError operands are now archived — alpha 1837891.6017776239 vs bound 1837741.1007418663, +8.2e-5 relative, matching the adjudicated double-shift prediction to three digits. Disclosure: the leg ran `nice -n 19` while a concurrent campaign's native timed leg held the box (loadavg ~55); a refusal capture carries no timing claim, so contention does not touch its validity. The qa assignment cell's "wrote no artifact" parenthetical was updated in the same commit. Device and class unchanged; row stays unplaced. | `docs/receipts/evidence/qa64_jaxgpu_solve_refusal_20260824.log` |
| 2026-08-24 | native-permanent-magnet-pm4stell | Pre-registered confirming replay executed and **CONFIRMED** — the static-phase FMA adjudication holds in its strongest form. A separate contract-off rebuild of `simsoptpp` (`-ffp-contract=off` substituted for `-ffp-contract=fast`, no code changed) rerun at the archived configuration (k=201, history off, OMP=8, repeat 2) produced a native endpoint **bitwise-identical to the archived JAX GPU dump** (139 placed rows, 0 of 5,826 rows differ; both repeats removed 31 antiparallel pairs where the production build removes 34): the entire 133-vs-139 fork is FMA contraction, and neither lane is buggy. The artifact's identity block names the loaded binary (sha `7c560e6b…` contract-off vs production `41b2ca79…`) — load-bearing, because a first same-morning attempt is VOID: the probe re-execs its native leg as a scrubbed child, a parent-process preload never reached it, and that leg ran the production binary while labeled ffp-off (caught by exactly this identity check; its ledger line 08:48Z stands as an executed-leg fact, its artifacts were removed from the evidence tree). Disclosures: box not quiet (`nice -n 19`, concurrent campaign legs) — admissible for a bitwise endpoint, no timing minted; donor pybind11 2.13.6 headers in the rebuild (binding glue only). Device and class unchanged; the exact-arithmetic predicate repair + parity re-certification still gate any placement. | `docs/receipts/evidence/pm4stell64_fork_k201_native_ffpoff_20260824.npz`, `docs/receipts/evidence/pm4stell64_native_ffpoff_20260824.json` and `docs/jax_gpu_pm_gpmo_campaign_plan.md` (§Blocked rung, confirming-replay record) |
| 2026-08-24 | native-permanent-magnet-pm4stell | **Exact-predicate repair LANDED, lane parity re-certified at the fork scale.** At `thresh_angle = pi` the dewyrming removal is now evaluated exactly (componentwise negation, first qualifying neighbor) in BOTH the C++ kernel and the JAX solver — the same integer-exact convention plain `GPMO_backtracking` always used; general angles keep the FP-dot path. Re-cert: a rebuilt native kernel with FMA contraction ON and the 5090 JAX lane produced bitwise-identical k=201 endpoints (`np.array_equal` true, 163 placed each — the exact predicate removes the 19 truly-antiparallel pairs where the pre-repair FP predicates removed 34/31 incl. rounded near-ties). Committed regression pins the discriminators (exact removed / one-ULP-off kept / general-angle oracle parity). Disclosed: the shipped prebuilt `.so` is NOT rebuilt (source now leads it; suite green against the old binary, near-tie-free at test scales); concurrent `nice -19` legs admissible for bitwise endpoints. Device/class unchanged — timing rungs stay chartered, now unblocked. | `docs/jax_gpu_pm_gpmo_campaign_plan.md` (§Blocked rung, repair record), `docs/receipts/evidence/pm4stell64_fork_k201_native_predicate_20260824.npz`, `docs/receipts/evidence/pm4stell64_fork_k201_jaxgpu_predicate_20260824.npz` and `tests/jax/core/test_pm_optimization_jax_item25.py` |
| 2026-08-24 | native-permanent-magnet-qa | **Instrument-side staging fix applied; the refusing lane now completes.** `run_jax_relax_split` stages the RAW grid before the host `rescale_for_opt` shift (the P3.5 double-shift root cause; ordering is now the documented contract), so the explicit alpha passes the validator that false-tripped. First-ever completed qa-64 jax-gpu solve leg: rc=0, both continuation solves finite over all 29,286 dipoles, diagnostic 5.34 s cold / 4.93 s repeat_retrace — cold/retrace numbers, NO denominator, no claim. Ordering-contract regression pinned in item28 (fixed order accepted == default-step formula; buggy order refused with the archived signature). Transient disclosed: one `CUDA_ERROR_ILLEGAL_ADDRESS` crash on a just-vacated card; clean-GPU retry green. Device and class unchanged. | `docs/receipts/evidence/qa64_jaxgpu_solve_20260824.json`, `docs/receipts/evidence/qa64_rs_jaxgpu_20260824.npz` and `tests/solve/test_permanent_magnet_optimization_jax_item28.py` |
