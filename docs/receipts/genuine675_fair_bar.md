# Genuine-675 fair-bar re-adjudication — terminal receipt

**Verdict: `WIN` at both rungs** (2026-08-19, preregistered dual rule). The
archived 2026-07-21 "flat genuine-675 fp64-GPU 9.8× native" claim — flagged
suspect for a failed triad, N=1, an unpinned 64-thread native, and GPU-only
priming — was re-adjudicated under the frozen fair-bar charter with the
bit-identical July instrument and **survives, strengthened**:

| Rung | Native (median, swept omp16) | GPU (median) | Pair ratios (process wall) | Median | Anchor / GPU | Gate |
| --- | --- | --- | --- | --- | --- | --- |
| **B3** (archived budget) | 54.97 s | 6.80 s | 8.04 / 8.05 / 8.07 / 8.09 / 8.38 | **8.07×** | 58.702 / 6.80 = **8.63×** | ≥1.10 ✔, every pair >1.00 ✔ |
| **B37** (headline) | 287.74 s | 11.11 s | 25.47 / 25.81 / 25.87 / 25.98 / 26.08 | **25.87×** | 287.50 / 11.11 = **25.87×** | ≥1.10 ✔, every pair >1.00 ✔ |

On the archived claim's own timer (optimizer wall), the B3 re-adjudication
gives median **10.33×** — the archived "9.8×" *understated* the July
instrument's speedup once the denominator was swept and both lanes primed.
Its honest process-wall basis (7.47×) is likewise superseded by the 8.07×
measured here. The primary timer of this receipt is `process_wall_seconds`
of the timed child; optimizer-wall ratios are report-only.

The fresh-cache disclosure pairs (both lanes cold, no primers, fresh XLA
cache) also favored the GPU: B3 native 55.09 s vs GPU 19.56 s (2.82×); B37
native 286.81 s vs GPU 23.91 s (11.99×). **Reported, not claimed** — these
are `FRESH_REPORTED` disclosures at N=1 per rung, and the charter mints a
cold-start claim only from a cold pair set satisfying the five-pair win
rule, which N=1 cannot. (Contrast the finite-build campaign, whose full
five-pair fresh run was a bounded negative; here the single disclosure pair
per rung points the other way but is not evidence of the same strength.)

Charter: `docs/jax_gpu_genuine675_fair_bar_plan.md`, frozen `7b6d69041`,
amended pre-evidence A1–A3a; append-only charter-sha lineage
`92e6a657… → 537d621b… → 1d82aece… → be4b262c… → 2dea1522… → 6ca00d03…`.
Rows bind the sha current at execution (phase 1 / B3 matrix / probe at
`be4b262c…`, B37 matrix at `2dea1522…`, both pairs phases and both fresh
pairs at `6ca00d03…`); `validate` accepts exactly this lineage and returns
`OK` for all eight run directories.

## What was measured, exactly

Both lanes solve the archived genuine-675 flat single-stage problem (675
outer DOFs = 11 coil + 3 vessel + 661 surface; inner (iota, G) by
closed-form 2-column QR) from the archived start candidate, under the
archived L-BFGS-B policy (`maxcor=300, maxls=8, ftol=0, gtol=1e-3`,
policy sha `fc349892…`), at fixed budgets B3 and B37:

- **Native lane** (denominator): the July instrument worktree, detached
  clean at `1c23f6c5` — C++/simsoptpp evaluation under the host SciPy
  driver. Config selected by a five-point OMP sweep **at each rung**
  (medians of 3, seconds — B3: omp1 70.10, omp2 61.46, omp4 56.48,
  omp8 53.73, **omp16 52.70**; B37: omp1 376.96, omp2 341.59, omp4 300.24,
  omp8 297.01, **omp16 287.57**). omp16 is **SMT-assisted** (16 threads on
  the campaign's 8 physical cores); it is the fastest config at both rungs
  and can only lower the bar. One discarded primer child per timed process.
- **GPU lane** (numerator): the same instrument's JAX fp64 lane on the
  RTX 5090 — the identical host-SciPy-loop code path that produced the
  archived claim (NOT the new fused lane; that is the separate F3
  campaign). Warm legs run with a persistent XLA cache primed by a
  discarded primer child.
- **Interleaved pairs**, alternating order (native-first / gpu-first), five
  per rung; per-pair fail-closed gates all passed: child-observed
  conformance (env echo, resolved `omp_get_max_threads`, granted
  affinity), bracketed partition-integrity (entry AND exit), matched
  compact evaluation counts (B3: 9/9; B37: 49/49 per pair), endpoint
  objective agreement ≤1e-10 relative (B3 endpoints agree to ~5e-14;
  B37 to ~2e-11; gradient-∞ agrees to 8 digits), and native-oracle
  cross-evaluation of GPU endpoints.

### Endpoint quality (B3 pair 0, representative)

| | objective | inner (iota, G) |
| --- | --- | --- |
| native | 1.8133486877704454 | 0.1496019311863628, 2.010619295352492 |
| GPU | 1.8133486877705427 | 0.14960193118636242, 2.010619295352476 |

## Scope

- **This campaign's verdict speaks to optimization launched from this one
  archived mid-trajectory native iterate, not to an ensemble of start
  candidates.**
- The claim is the July *instrument's* GPU-vs-native speed at matched work
  under fair discipline. It is a host-loop lane: each L-BFGS-B iterate
  crosses the host boundary. The production fused lane is chartered
  separately (F3, `docs/jax_gpu_flat675_fused_campaign_plan.md`) and its
  receipt supersedes the archived 9.8× as the program's citable flat-675
  number going forward; **this receipt owns the past** — it is the sole
  adjudication of the archived claim.
- Anchor law: the B3 anchor is the fastest archived native wall
  (58.702 s); the archived July condition also produced 77.046, 82.039,
  and 87.310 s walls — the adjudication is against the *best* archived
  native. The B37 anchor is the archived sustained per-eval mean
  (52.807/9 s) × the pairs' matched eval count (49), i.e. 287.50 s —
  pair-derived per Amendment 2a. Anchors were timed on an uncontended box;
  under the partition protocol contention can only make the anchor rule
  harder to pass.

## Headline rung B50 → B37 (preregistered contingency)

The frozen headline was B50. The cross-implementation probe (untimed, both
lanes at B50) found byte-agreement of the accepted-objective sequence up to
iterate 37 and a fork at 38 (both lanes 50/50 accepts, 66/66 evaluations,
certificate cadence non-scaling at 2 attempts/lane — cross-arch fp64 drift
through L-BFGS-B curvature pairs, both endpoints healthy). Amendment 3
fixed the headline at the matched prefix, B37, before any timed headline
evidence existed. `PROBE_FORKED` is recorded in the evidence.

## Partition protocol disclosure (Amendment 2)

The box was NOT idle: a foreign compute campaign ran throughout. The
campaign reserved CCD0 (CPUs {0–7, 32–39}, private L3) fail-closed; foreign
compute was confined to {8–31, 40–63} by a 5-second daemon plus runner
sweeps; every timed leg ran a bracketed partition-integrity gate (foreign
affinity audit, reserved-busy <20%, GPU ≤5% for native legs) at entry AND
exit. Contamination, had any leaked, is bounded by the anchor rule (the
uncontended archive can only be *harder* to beat from a contended box).

## Harness-dirty disclosure

Harness git state is captured **twice per phase**: once at CLI start,
stamped on that phase's **rows**, and once at phase finish, stamped in that
phase's **manifest**. Long phases can therefore legitimately record
different states in the two places; both are tracked evidence and both are
stated here:

| phase | window (EDT) | rows (start) | manifest (finish) |
| --- | --- | --- | --- |
| phase 1, B3 matrix, probe | 04:01–04:50 | `475e967c5` clean | `475e967c5` clean |
| B37 matrix | 04:52–07:36 | `23c147f32` clean | `fb0ad88d9` dirty=1 |
| pairs B3 | 07:36–07:51 | `fb0ad88d9` dirty=1 | `fb0ad88d9` dirty=1 |
| pairs B37 | 07:51–08:45 | `fb0ad88d9` dirty=1 | `a3639915a` clean |
| fresh B3 / B37 | 08:45–08:52 | `a3639915a` clean | `a3639915a` clean |

The single dirty entry, wherever it appears, is the then-untracked F3
charter draft (`docs/jax_gpu_flat675_fused_campaign_plan.md`, committed
later as `b7ec63b6e`). During the long B37-matrix and pairs-B37 windows,
disjoint commits landed on the branch (among them `e4ef23765` 06:53,
`fb0ad88d9` 07:22, `b7ec63b6e` 07:58, `020fb0c8b` 08:16, `a3639915a`
08:31 — flat-675 port and F3 charter/instrument work touching no fair-bar
file — and the harness rebind `37fed39d1` 05:27, which could not affect
the already-running B37-matrix process: that process loaded `23c147f32`'s
harness at 04:52 and its manifest binds precisely that harness's charter
sha), which is why a phase's finish capture can post-date its start
capture by several commits. The timed children are isolated from harness-tree churn by the
fail-closed import-origin guard, which resolves every import from the
clean instrument tree. The harness `.py` itself advanced across phases
only along the charter amendment chain (`475e967c5` → `23c147f32` →
`fb0ad88d9` carry the successive charter-sha rebinds; that is the lineage
mechanism `validate` accepts, and each phase's rows were produced by the
harness at their recorded commit). The instrument worktree records
`dirty_file_count = 0` in every row and every manifest of every phase.

## A100 supplementary lane (non-verdict)

The same instrument on landau's A100 (driver 595.71.05, jax 0.10.0
native): trajectory reproduction vs archived native at 3.5–5.2e-14; warm
process walls median 15.16 s (B3) and 18.06 s (B50), 5 reps each, bitwise
endpoints across reps. Third-silicon corroboration only; never part of any
verdict; raw runs archived at
`~/simsopt_mixed_artifacts/genuine675_fair_bar/a100-supplementary/`.

## Evidence

Tracked under `docs/receipts/evidence/genuine675_fair_bar/` — the
`manifest.json` of each of the eight run directories (phase 1, B3 matrix,
B50 probe, B37 matrix, B3 pairs, B37 pairs, fresh B3, fresh B37). Each
manifest embeds the charter sha, the campaign-input-manifest sha — which
tracks the charter lineage: `1df14169…` (phase 1, B3 matrix, probe, under
charter `be4b262c…`), `fca9b46a…` (B37 matrix, under `2dea1522…`),
`2a381125…` (both pairs phases and both fresh pairs, under `6ca00d03…`);
all three bind the same frozen input bundle `84febc05…` — the formulation
semantic sha (`0fe8e9e7…`, constant across all eight), the per-row
contract sha, and a `rows` map binding every row/lane/provenance file by
sha256. Re-validate
any run directory with:

```
PYTHONPATH=<WT>:<WT>/src <v0c-python> benchmarks/genuine_675_fair_bar.py \
  validate ~/simsopt_mixed_artifacts/genuine675_fair_bar/<run-dir>
```

(`<WT>` = the instrument worktree at `1c23f6c5`.) All eight return
`"validation": "OK"` with the stored verdicts
`PHASE1_OK / NATIVE_SELECTED ×2 / PROBE_FORKED / WIN ×2 /
FRESH_REPORTED ×2`.

Harness: `benchmarks/genuine_675_fair_bar.py` +
`benchmarks/genuine_675_fair_bar_oracle.py` (commit chain `12dafb27b` …
`fb0ad88d9`; six adversarial review rounds to strict PASS pre-execution).
Instrument: `1c23f6c5` (clean, enforced). Runtime env: v0c
(python 3.11.15, stock jax 0.10.0).
