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
  `docs/single_stage_jax_gpu_projected_route_certification_plan.md`,
  `docs/jax_porting_progress_report.md`, and the example sources under
  `examples/jax/` that every `census-structural` row is read from.
- **Host-local campaign artifacts** — *not* in this repository, not reviewable
  from a clone, not reproducible without this workstation. Marked
  **[host-local]** at every use: `~/simsopt-campaigns/winnable-six-20260815/`
  (44-leg `receipt.json`, `silicon_probe_results.json`) and
  `~/simsopt-campaigns/ndparity-boozer-vacuum-20260814/`.
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
| projected-route-single-stage-boozer-vacuum-optimization | unmeasured | unmeasured | wide coupled projected route; the certified 2.304x engine-boundary GPU win belongs to the benchmarks-path root run (`docs/single_stage_jax_gpu_projected_route_certification_plan.md`), not to this script |
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
| native-wireframe-gsco-modular | unmeasured | census-structural | [session-audit] classed it winnable; its source does share the 1024-row GSCO reduction of the certified multistep win, but a 48x50 wireframe carries roughly a quarter of the per-step work — no native_default receipt |
| native-wireframe-gsco-sector-saddle | unmeasured | census-structural | [session-audit] classed it winnable; its source does share the 1024-row GSCO reduction of the certified multistep win, but a 48x50 wireframe carries roughly a quarter of the per-step work — no native_default receipt |
| native-wireframe-rcls-basic | unmeasured | unmeasured | single dense regularized least-squares solve; [host-local: silicon probe] the dense-solve crossover sits between n=169 (CPU 2.4x) and n=716 (GPU ~5x), and this problem has not been placed against it |
| native-wireframe-rcls-with-ports | unmeasured | unmeasured | single dense regularized least-squares solve; [host-local: silicon probe] the dense-solve crossover sits between n=169 (CPU 2.4x) and n=716 (GPU ~5x), and this problem has not been placed against it |
| native-coil-forces | unmeasured | unmeasured | no native_default timing for the finite-build coil-force family |
| native-single-stage-boozer-vacuum-optimization | cpu | census-structural | sequential chain — latency-serialized inner Boozer Newton per outer evaluation; [session-audit] never-winnable class in this nested formulation; native_default lane artifacts [host-local: `~/simsopt-campaigns/ndparity-boozer-vacuum-20260814/`] |
| native-single-stage-optimization | cpu | census-structural | sequential chain — VMEC equilibrium and its finite-difference derivatives run host/MPI-serial; the manifest gives this example a `jax_slice_only` GPU scope |
| native-stage-two-optimization-finitebuild | unmeasured | unmeasured | no native_default timing for the stage-two coil family |
| native-wireframe-gsco-multistep | gpu | measured-certified | 3.5x warmed device solve (5.77-5.93 s vs 20.49 s best native) with a bitwise-identical 19,200-segment currents vector — `docs/receipts/wireframe_gsco_multistep_native_default_receipt.md` |

## Summary counts

39 manifest examples: **1 gpu**, **23 cpu**, **0 either**, **15 unmeasured**.

Restricted to the 27 `native-*` mirrors: 1 gpu, 13 cpu, 13 unmeasured.

The 13 `cpu` mirrors are the 2026-08-13 session audit's 11 never-winnable mirrors (3 tracing,
3 nested-Boozer, 5 tiny/fixed-size), plus `native-permanent-magnet-muse`
(measured loss, not structural) and `native-single-stage-optimization` (VMEC
host lane, `planned`, outside the 26 measured).

Four mirrors the 2026-08-13 session audit classed as winnable are marked `unmeasured`
here, because a device recommendation needs a receipt and they have none:
`native-permanent-magnet-pm4stell`, `native-permanent-magnet-qa`,
`native-wireframe-gsco-modular`, `native-wireframe-gsco-sector-saddle`. For
the two GPMO mirrors the nearest measurement — the same 256-row reduction in
`native-permanent-magnet-muse` — points the other way, so `unmeasured` is a
statement about a genuine conflict, not merely a missing number.

## Scope note and amendment procedure

These assignments were derived at the **2026-08-13 / 2026-08-16 evidence
state**: the 2026-08-13 six-agent `examples/jax` audit, the 2026-08-14
three-device silicon probe, the 2026-08-15 winnable-six campaign, and the
2026-08-16 full-precision GSCO promotion. Hardware: RTX 5090 plus Threadripper
9970X, with an A100-PCIE-40GB cross-check on the kernel probe only.

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
