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
  `docs/jax_porting_progress_report.md`, and the example sources under
  `examples/jax/` that every `census-structural` row is read from.
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
   utilization and only 120 W.

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
| native-permanent-magnet-simple | cpu | census-structural | tiny problem — fixed 2x2 quadrature grid (4 reduction rows), no native_default branch |
| native-qfm | unmeasured | unmeasured | no native_default timing; the port also swaps SLSQP for an augmented-Lagrangian solve, so a timing would not be matched work |
| native-stage-two-optimization-minimal | unmeasured | unmeasured | no native_default timing for the stage-two coil family |
| native-surf-vol-area | cpu | census-structural | tiny problem — mpol=1 / ntor=0 surface, two sequential scalar targets |
| native-tracing-fieldlines-ncsx | cpu | census-structural | sequential chain — sequential ODE event walk; [session-audit] never-winnable class |
| native-tracing-fieldlines-qa | cpu | census-structural | sequential chain — sequential ODE event walk; [session-audit] never-winnable class |
| native-tracing-particle | cpu | census-structural | sequential chain — sequential ODE event walk; [session-audit] never-winnable class |
| native-boozer | cpu | census-structural | sequential chain — host-side stage-1 loop around a latency-serialized inner Boozer solve |
| native-boozerqa | cpu | census-structural | sequential chain — one inner Newton solve per outer evaluation |
| native-permanent-magnet-muse | cpu | measured-diagnostic | narrow matrix — 256-row (16x16) GPMO reduction; [host-local: `~/simsopt-campaigns/winnable-six-20260815/receipt.json`] GPU 4.05x slower (2.03x iteration-normalized), 69-73% utilization at 120 W, dated pre-cond-skip and not matched-work |
| native-permanent-magnet-pm4stell | unmeasured | census-structural | [session-audit] classed it winnable (fixed-matrix GEMV), but its source builds the same 256-row native_default reduction as the MUSE mirror that measured a 4.05x GPU loss — unresolved, no receipt |
| native-permanent-magnet-qa | unmeasured | census-structural | [session-audit] classed it winnable (fixed-matrix GEMV), but its source builds the same 256-row native_default reduction as the MUSE mirror that measured a 4.05x GPU loss — unresolved, no receipt |
| native-stage-two-optimization | unmeasured | unmeasured | no native_default timing for the stage-two coil family |
| native-stage-two-optimization-planar-coils | unmeasured | unmeasured | no native_default timing for the stage-two coil family |
| native-stage-two-optimization-stochastic | unmeasured | unmeasured | no native_default timing for the stage-two coil family |
| native-strain-optimization | cpu | census-structural | tiny problem — 21 DOF (rotation order 10), host-bound by construction |
| native-wireframe-gsco-modular | cpu | measured-diagnostic | sequential chain — 2,000-iteration loop-carried greedy over 4,800 segments; the 1024-row reduction it shares with the certified multistep win is not enough once per-step work drops to a quarter. Interleaved A/B: warm GPU solve 0.552 s vs 0.492 s best native (0.89x), 0.89x across the numerical region, a tie against OMP=32, and 2.75x behind cold; currents vector bitwise identical — `docs/receipts/wireframe_gsco_siblings_native_default.md` |
| native-wireframe-gsco-sector-saddle | cpu | measured-diagnostic | sequential chain — 2,000-iteration loop-carried greedy over 4,800 segments; the 1024-row reduction it shares with the certified multistep win is not enough once per-step work drops to a quarter. Interleaved A/B: warm GPU solve 0.653 s vs 0.518 s best native (0.79x), 0.81x across the numerical region, 0.86x even against OMP=32; currents vector bitwise identical — `docs/receipts/wireframe_gsco_siblings_native_default.md` |
| native-wireframe-rcls-basic | unmeasured | unmeasured | single dense regularized least-squares solve; [host-local: silicon probe] the dense-solve crossover sits between n=169 (CPU 2.4x) and n=716 (GPU ~5x), and this problem has not been placed against it |
| native-wireframe-rcls-with-ports | unmeasured | unmeasured | single dense regularized least-squares solve; [host-local: silicon probe] the dense-solve crossover sits between n=169 (CPU 2.4x) and n=716 (GPU ~5x), and this problem has not been placed against it |
| native-coil-forces | unmeasured | unmeasured | no native_default timing for the finite-build coil-force family |
| native-single-stage-boozer-vacuum-optimization | cpu | census-structural | sequential chain — latency-serialized inner Boozer Newton per outer evaluation; [session-audit] never-winnable class in this nested formulation; native_default lane artifacts [host-local: `~/simsopt-campaigns/ndparity-boozer-vacuum-20260814/`] |
| native-single-stage-optimization | cpu | census-structural | sequential chain — VMEC equilibrium and its finite-difference derivatives run host/MPI-serial; the manifest gives this example a `jax_slice_only` GPU scope |
| native-stage-two-optimization-finitebuild | gpu | measured-certified | 13.58x warm solve (45.23 s vs 3.353 s) and 3.11x warm persistent-cache process wall (50.1 s vs 16.11 s) over the fastest qualifying native lane (omp2-h400, swept optimum), five interleaved pairs each, every pair > 1.00, endpoint quality oracle-verified under the successor v4 contract and every GPU endpoint bitwise-identical to the frozen crossing solution. Repeated-workload/persistent-cache win only: a fresh-empty-cache process loses 0.88x to the ~42 s XLA compile (measured, reported, no cold-start claim) — `docs/receipts/stage_two_finitebuild_native_gpu_successor.md` |
| native-wireframe-gsco-multistep | gpu | measured-certified | 3.5x warmed device solve (5.77-5.93 s vs 20.49 s best native) with a bitwise-identical 19,200-segment currents vector — `docs/receipts/wireframe_gsco_multistep_native_default_receipt.md` |

## Summary counts

39 manifest examples: **3 gpu**, **25 cpu**, **0 either**, **11 unmeasured**.

Restricted to the 27 `native-*` mirrors: 2 gpu, 15 cpu, 10 unmeasured.

The 15 `cpu` mirrors are the 2026-08-13 session audit's 11 never-winnable mirrors (3 tracing,
3 nested-Boozer, 5 tiny/fixed-size), plus `native-permanent-magnet-muse`
(measured loss, not structural), `native-single-stage-optimization` (VMEC
host lane, `planned`, outside the 26 measured), and the two 2026-08-16
wireframe-GSCO siblings (measured loss, not structural).

Two mirrors the 2026-08-13 session audit classed as winnable are marked `unmeasured`
here, because a device recommendation needs a receipt and they have none:
`native-permanent-magnet-pm4stell` and `native-permanent-magnet-qa`. For
those two GPMO mirrors the nearest measurement — the same 256-row reduction in
`native-permanent-magnet-muse` — points the other way, so `unmeasured` is a
statement about a genuine conflict, not merely a missing number. The other two
the audit classed winnable, `native-wireframe-gsco-modular` and
`native-wireframe-gsco-sector-saddle`, have now been measured and lose: the
audit's classification is superseded for both.

## Scope note and amendment procedure

These assignments were derived at the **2026-08-13 / 2026-08-16 evidence
state**: the 2026-08-13 six-agent `examples/jax` audit, the 2026-08-14
three-device silicon probe, the 2026-08-15 winnable-six campaign, the
2026-08-16 full-precision GSCO promotion, and the 2026-08-16 GSCO-siblings
campaign. Hardware: RTX 5090 plus Threadripper
9970X, with an A100-PCIE-40GB cross-check on the kernel probe only.

**Benchmarks-path campaign receipts outside this table's scope
(2026-08-19).** Two sealed campaign receipts measure the flat-675
formulation, which has no `examples/jax` mirror and therefore no row above
(the timed instruments are `benchmarks/` harnesses; the fused lane's
production module is `src/simsopt_jax/examples/single_stage_flat675.py`):
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
| 2026-08-19 | (scope note) | Recorded two sealed benchmarks-path campaign receipts for the flat-675 formulation, which has no example mirror and changes no assignment row: the fair-bar re-adjudication of the archived flat-675 "9.8x" (host-loop instrument, 8.07x B3 / 25.87x B37 process wall; the archived claim survives strengthened) and the F3 fused production campaign (1.67x / 7.70x / 7.36x process wall, superseding the 9.8x as the citable fused production number; host-loop remains faster at those budgets on that timer). Timers named in the scope note; example assignment counts unchanged. | `docs/receipts/genuine675_fair_bar.md` and `docs/receipts/flat675_fused_campaign.md` |
