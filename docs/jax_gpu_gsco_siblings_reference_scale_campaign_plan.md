# Wireframe GSCO siblings at reference scale — certifying campaign charter

**Status: DRAFT — NOT FROZEN.** This is the `[charter]` deliverable of task **P2.4**
(`docs/jax_gpu_examples_backlog_native_speed_implementation_plan.md, Phase 2 task P2.4`). A draft
charter **mints nothing**: no ratio in it is a claim, no row in
`docs/jax_example_device_assignment.md` moves on it, no receipt may cite it. It becomes a
preregistration only when its own review rounds close it and it is committed with `FROZEN`,
a freeze date, and §2's instrument sha filled in. Until then every number in it comes from
`diagnostic-not-certifying` probe artifacts: motivation, never evidence.

## 1. Why this campaign exists (motivation only)

The two `examples/2_Intermediate` GSCO siblings are placed `cpu` by the sealed 2026-08-16
receipt `docs/receipts/wireframe_gsco_siblings_native_default.md` at *shipped* scale (48x50,
4,800 segments, 2,000 iterations): warm device solve 0.89x (modular) / 0.79x (sector-saddle)
of native `OMP_NUM_THREADS=48` over ten interleaved rounds, currents bitwise identical (0
ULP) on every leg. Its discriminator is per-step work volume, not reduction dimension: the
certified 3.5x multistep sibling
(`docs/receipts/wireframe_gsco_multistep_native_default_receipt.md`) runs 7 x 2,500
iterations over 19,200 segments. But both native sources name a *reference*
configuration in dead comments no selector reaches: 96x100 (19,200 segments, 9,600 cells),
20,000 iterations, plasma 32^2, and for sector-saddle `break_width=4`, `gsco_cur_frac=0.03`,
`lambda_S=10**-7.5`. The P2.2/P2.3 probe (in-tree artifacts
`docs/receipts/evidence/gsco_{modular,sector_saddle}_{reference,shipped}_{native_omp{16,32,48},jaxgpu_{a,b}}_20260823.json`,
each self-labelled `diagnostic-not-certifying`) measured at reference scale:

| sibling | GPU warm | native OMP=16 | OMP=32 | OMP=48 | accepted iters |
| --- | --- | --- | --- | --- | --- |
| modular | 5.25–5.33 s (median 5.27) | 30.0–30.5 s | 27.4–28.0 s | 153.8–154.1 s | 8,031 |
| sector-saddle | 8.05–10.01 s (median 8.08; two of six warm samples ≈10.0 s) | 37.7–38.8 s | 35.2–37.6 s | 233.1–234.2 s | 12,589 |

— ~5.2x and ~4.4x, the best native OMP=32 leg over the six-sample GPU warm median
(27.4/5.27 and 35.2/8.08; any consistent statistic pair gives the same two ratios to one
decimal), final 19,200-entry currents vector
**bitwise identical** (0 ULP) for both siblings, GPU per-iteration cost nearly flat 2,000 ->
20,000 iterations (0.20 -> 0.26 ms/it). Not a claim; the reason to spend a campaign.

**The conflict this charter must also resolve.** The same probe re-measured *shipped* scale
and came back in the **opposite direction** from the sealed receipt: GPU warm 0.404–0.413 s
(modular, median 0.408) against a sampled-native optimum of 0.669 s (OMP=32) — ~1.6x *for* the GPU where
the receipt recorded 0.79–0.89x against it. Two things moved at once and the charter must
not keep only the flattering half: the GPU lane got faster (0.41 vs the receipt's 0.552 s)
*and* the probe's OMP=48 legs are pathological on this box in this session (33.9 s modular,
30.1 s sector-saddle, where the receipt measured 0.492 / 0.518 s and called OMP=48 the
configuration a native user should run). Native regressions of ~69x (modular, 33.94/0.492)
and ~58x (sector-saddle, 30.10/0.518) at the receipt's own best
configuration are a defect until proven otherwise, and adopting them would mint exactly the
false win this repository's OMP law exists to kill. Hence §4's shipped-scale re-adjudication
rungs.

## 2. Instrument (frozen at freeze time)

`benchmarks/wireframe_gsco_siblings_reference_scale.py` with
`benchmarks/probe_conventions.py`, tracked and clean at drafting time (last touched by
commit `76f1b5f37`; the former's content sha256
`98480d3e25c8c3cdc32e41ca89ec9db49657b56834ee5d40a2471c5831b44102`, tree `451baf2b5`). **The
freezing commit records the instrument's commit sha and content sha256 here, and no evidence
leg may run against an instrument whose content sha256 differs from the frozen one**; a
post-freeze change requires a dated pre-evidence amendment (§10) that re-freezes the sha and
invalidates every leg recorded before it. Example scripts are not imported, executed, or
edited; the configuration is built through public APIs only. Before freeze the instrument
gains one capability it lacks — a driver running §3's pair schedule and emitting a per-rung
reduction; no gate, no new physics path.

## 3. Win rule (per-sibling, per-scale rungs)

Backlog §Campaign protocol item 2 (`…backlog_native_speed_implementation_plan.md §Campaign protocol item 2`),
inherited from `docs/jax_gpu_finitebuild_native_speed_successor_plan.md:198-206`:

- **Five interleaved pairs per rung, alternating order** (native-first, GPU-first, …), each
  leg a fresh process, GPU legs serialized, affinity pinned, quiet-gated. **WIN** requires
  median paired `native_seconds / gpu_seconds >= 1.10` **and every one of the five pairs `>
  1.00`**. Anything else is `CLOSED_BOUNDED_NEGATIVE`; `NOT_PRODUCED` stays broken evidence,
  never a verdict.
- Ratio of record is the **solve window** on both lanes (`solve_call_s` native,
  `warm_solve_s` / `warm_repeat_*_s` GPU); the matrix build is timed separately on both
  lanes and subtracted from neither. Process wall is recorded but **not ratioed** — the
  native lane renders plots and writes VTK, the JAX lane does not (sealed-receipt rule).
  This is a **declared deviation from the inherited two-timer law**: the cited
  `…successor_plan.md:198-206` requires the ≥ 1.10 median on warm persistent-cache
  `process_wall_seconds` as well, and the PM and stochastic sibling charters keep both
  timers. It follows the sealed sibling receipt's single-timer rule instead because the
  native wall is inflated by rendering work the JAX lane does not perform; the freeze
  review must either ratify this single-timer rule or restore the wall gate before
  freezing.
- **Four rungs, adjudicated independently** — R1 modular and R2 sector-saddle at *reference*
  scale (each mints at most a benchmarks-path claim naming that configuration), R3 modular
  and R4 sector-saddle at *shipped* scale (each may re-adjudicate its shipped-scale row,
  `unmeasured` since the same-commit conflict amendment superseded its 2026-08-16 `cpu`
  placement).
  One rung's verdict is never evidence for another.

## 4. Scope law

- **R1/R2 claims are scoped to the named reference configuration** — 96x100, 20,000
  iterations, plasma 32^2, and for sector-saddle `break_width=4`, `gsco_cur_frac=0.03`,
  `lambda_S=10**-7.5` — restated wherever the ratio is quoted. That configuration has **no
  `examples/jax` mirror and no selector in the shipped examples**, so an R1/R2 win moves
  **no example row**: it is a dated scope-note entry in
  `docs/jax_example_device_assignment.md` (the flat-675 precedent, under that record's
  "Scope note and amendment procedure"), naming its timer, the two shipped-scale rows (now
  `unmeasured`) untouched.
- **R3/R4 are the only path by which the 2026-08-16 rows may move.** They may supersede
  `docs/receipts/wireframe_gsco_siblings_native_default.md` only through this campaign's own
  certified protocol, in a receipt that (a) states the superseded numbers, (b) reproduces or
  explicitly refutes that receipt's OMP=48 denominator under §5's full sweep, and (c)
  accounts for the both-lanes movement of §1. **A rung that wins only because the probe's
  OMP=48 pathology reappears is `NOT_PRODUCED`, not a win** (§9).
- Row movement follows that record's procedure verbatim: append a dated log row and edit the
  table row it refers to **in the same commit**, never rewriting a log entry; `gpu` requires
  a receipt tracked as a regular file under `docs/receipts/`, enforced by
  `tests/test_jax_example_device_assignment.py`.

## 5. Fair-native denominator

Backlog item 3 (`…backlog_native_speed_implementation_plan.md §Campaign protocol item 3`),
`…finitebuild_native_speed_successor_plan.md:186-189`, sweep-down correction
`docs/receipts/stage_two_minimal_coupled_route.md:574-583`:

- **Full sweep `OMP_NUM_THREADS` in {2,4,8,16,32,48}, per sibling per scale.** The probe
  sampled only {16,32,48} and this charter refuses that denominator; the sweep-down half is
  not optional — the OMP law exists because narrow problems optimize below 16, and it has
  already killed two false wins.
- **The denominator is the swept optimum**, median of its own repetitions, measured in the
  same interleaved schedule as the pairs it denominates. Every OMP value's median publishes,
  pathological ones included; a non-monotone sweep is called out, never silently reduced to
  its minimum. **The shipped default (`OMP_NUM_THREADS` unset) is disclosed separately and
  is never a denominator** — the 64-thread collapse is a trap the siblings receipt named.
  The GPU host pin is `OMP_NUM_THREADS=8` (`GPU_HOST_OMP_THREADS` of
  `benchmarks/stage_two_finitebuild_native_gpu.py`), stated per leg; both lanes read the pin
  back from libgomp and refuse to run on mismatch.
- **Mode-2 window, disclosed.** The native lane is timed in `optimize_wireframe`'s
  precomputed-matrix mode (`Amat`/`bvec`) with `bnorm_obj_matrices` timed separately, as the
  JAX lane splits `bnorm_obj_matrices_jax` from `gsco_wireframe_jax`. Two residual
  asymmetries ride in that window, disclosed and never subtracted: (a) the native window
  still holds one post-solve `WireframeField` construction over the solution currents, which
  `optimize_wireframe` performs in both modes and the JAX window has no equivalent of —
  **pro-GPU**, so the campaign times it standalone at both scales, publishes it as a
  fraction of the native median, and republishes any rung above 2% with and without it; (b)
  `print_interval=max_iter` leaves the C++ kernel writing two progress lines instead of
  ~`max_iter/100`, a small **anti-GPU** bias, disclosed and uncorrected. The sector-saddle
  topology-accessor cost stays in the kernel figure on both lanes, per the sealed receipt.

## 6. Warm/cold scoping

Backlog item 4 (`…backlog_native_speed_implementation_plan.md §Campaign protocol item 4`):

- Claims are **warm same-process** and **warm persistent-cache** only. Persistent cache via
  `JAX_COMPILATION_CACHE_DIR` with `JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0` and
  `JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0`; the entry count is recorded before and
  after every timed solve and **a warm leg whose count grows is disqualified as warm**
  (`cache_entries_before` / `_after_first_solve` / `_after_second_solve`).
- **Cold is measured and disclosed, never folded into any ratio.** A cold claim exists only
  if the cold numbers independently pass §3; the probe's reference-scale cold solves
  (5.39–6.09 s modular, 8.26–8.28 s sector-saddle) suggest they might — measure, do not
  assume. Module-import
  time is excluded from every ratio, both lanes, and reported.

## 7. Physics gate

Backlog item 5 (`…backlog_native_speed_implementation_plan.md §Campaign protocol item 5`):

- **Bitwise identity of the final segment-currents vector, both lanes, every pair.** GSCO is
  exact greedy arithmetic; the multistep receipt proved 0 ULP over 19,200 entries, the
  siblings receipt over 88 x 14–15 legs. Anything short of bitwise is a kill (§9), not a
  tolerance conversation — the `native_workflow` bucket
  (`src/simsopt_jax/parity_tolerances.py`) is the floor this family clears by the maximum
  possible margin. Accepted-iteration count, support, nonzero count and the current ladder
  must agree exactly and publish per rung. Comparison runs through `--compare`, whose
  **configuration-sidecar identity gate** refuses two dumps whose (sibling, scale,
  resolution, iteration, penalty) fields disagree; every `.npy` carries its sidecar.
- **Input-fidelity note the campaign must bound or accept explicitly.** Both lanes digest
  the `A` and `b` they solve against. The probe found `b` **equal** across lanes at both
  scales (`target_vector_sha256 9f1dcbc35c…`) while `A` **differs** (`eb6bd4ecc3…` native vs
  `368a4bd0a3…` JAX at reference; `d9f1b92a83…` vs `dac60da1e3…` at shipped):
  `bnorm_obj_matrices_jax` reduces in a different order — a last-bit-level difference the
  probe observed but published nothing to bound (the artifacts hold only the digests, so no
  magnitude may be quoted). **Before any pair
  runs this campaign must** publish the elementwise max relative and absolute `A` difference
  at both scales and either (i) feed the JAX lane the native `A` so the compared problem is
  digest-identical, or (ii) accept the difference in the frozen text with the measured bound
  stated, arguing the greedy accepted set is insensitive at that magnitude — then treat any
  bitwise mismatch as evidence that argument was wrong. Option (i) is preferred; the choice
  is pre-evidence, not amendable after.

## 8. Provenance and box conditions

Backlog item 6 (`…backlog_native_speed_implementation_plan.md §Campaign protocol item 6`):

- Every leg publishes `benchmarks.probe_conventions.runtime_identity(lane)` — git commit +
  per-changed-file sha256, hostname, python/jax/jaxlib/numpy/scipy versions, `simsoptpp`
  sha256, jax devices and default backend, `jax_enable_x64`, XLA flags, threading
  environment and cpu affinity, loadavg, `nvidia-smi` compute processes — and appends to
  `docs/receipts/evidence/probe_leg_ledger.jsonl`, interleave order per rung. **Cross-leg
  conformance is fail-closed**: a rung whose legs disagree on instrument sha, `simsoptpp`
  sha, jax version, x64 state or device is `NOT_PRODUCED`.
- **Quiet box**: 2-second `/proc/stat` utilization gate below 15% at each leg start,
  recorded; any leg with a foreign GPU compute process is discarded and rerun. The sealed
  receipt's classifier defect (`_is_baseline` substring-matching `"code"` against the repo
  path, mis-labelling sibling campaign processes as baseline) is a known trap: match by pid
  and command, never substring. Clean tree at every pair leg (`git_dirty_files: []`).

## 9. Kill criteria (fail-closed; none amendable post-evidence)

- **Any pair `<= 1.00`, on any rung R1–R4, closes that rung**
  `CLOSED_BOUNDED_NEGATIVE`; median `>= 1.10` is the second gate, not the first.
- **Physics non-bitwise** on any pair, either scale: `NOT_PRODUCED` — a forked greedy
  trajectory is an instrument or input defect, not a slow lane; investigate before rerun.
- **Box contention voids the leg** (foreign GPU process, failed quiet gate, dirty tree); a
  rung that cannot assemble five clean pairs is `NOT_PRODUCED`, as is any **cross-leg
  conformance failure** or **instrument sha drift** within a rung.
- **Sweep pathology guard (R3/R4)**: if the shipped-scale sweep reproduces the probe's
  OMP=48 pathology (OMP=48 slower than OMP=32 by more than 2x) the rung is `NOT_PRODUCED`
  until root-caused — that sweep cannot be reconciled with the sealed receipt's denominator
  and a win over it measures the defect. Root-cause it; do not adopt it.

## 10. Amendment discipline

- §§2–9 are **frozen text** at the freezing commit. Dated amendments are permitted **only
  before the evidence they govern exists**, are append-only, never edit frozen text in
  place, and each cites its empirical basis by artifact path and sha256 (backlog item 1,
  `…backlog_native_speed_implementation_plan.md §Campaign protocol item 1`;
  `…finitebuild_native_speed_successor_plan.md:247-251`;
  `docs/jax_gpu_flat675_fused_campaign_plan.md:267-271,293-294`).
- **Non-amendable post-evidence, on any lane's evidence:** the win rule (§3), the scope law
  (§4), the denominator rule including the full OMP set (§5), warm/cold scoping (§6), the
  physics gate and the `A`-digest option chosen in §7, and every kill criterion (§9).
- Amendable pre-evidence only: rung ordering, repetition counts above the five-pair floor,
  and the instrument sha (which re-freezes and invalidates prior legs). A verdict requiring
  a frozen clause to change is a `CLOSED_BOUNDED_NEGATIVE` or a successor charter.

## 11. Phases

- **P-0 Freeze.** Review rounds close; instrument sha recorded in §2; §7's `A`-digest option
  chosen and written in; status becomes `FROZEN` with a date. Nothing below runs before that
  commit exists. **P-1 Input fidelity**: publish the `A`/`b` bound at both scales and execute
  the chosen §7 option, failing closed if the bound exceeds what §7 accepted.
- **P-2 Native sweep.** Full `{2,4,8,16,32,48}` sweep per sibling per scale, interleaved and
  quiet-gated; swept optimum fixed as each rung's denominator; shipped-default leg
  disclosed; §9's pathology guard evaluated on the shipped sweeps first.
- **P-3 Warm/cold.** Cache-entry-count proof of warmth; cold legs at both scales; the
  post-solve `WireframeField` construction timed standalone (§5).
- **P-4 Pairs.** Five interleaved alternating-order pairs per rung, R1–R4 in order, each
  pair's currents compared bitwise through `--compare` before its time is admitted. **P-5
  Publication** per §12.

## 12. Evidence layout, receipt, and scoreboard

Backlog item 7 (`…backlog_native_speed_implementation_plan.md §Campaign protocol item 7`):

- Tracked evidence bundle under `docs/receipts/evidence/gsco_siblings_reference_scale/`:
  per-leg artifact JSON, `.npy` currents dumps and their configuration sidecars, the sweep
  reduction, the pair schedule with per-pair ratios, the `--compare` verdicts, the
  input-fidelity record and the leg-ledger slice. The 2026-08-23 probe artifacts stay under
  `docs/receipts/evidence/`: motivation, never evidence.
- Tracked terminal receipt at `docs/receipts/wireframe_gsco_siblings_reference_scale.md`:
  every rung's verdict, the timer names, every pair ratio, the full OMP sweep table, the
  shipped-default disclosure, the cold numbers, the bitwise verdicts, and for R3/R4 the
  reconciliation with `docs/receipts/wireframe_gsco_siblings_native_default.md` — all of it
  published regardless of verdict.
- `docs/jax_example_device_assignment.md` amended **in the same commit** as the measured
  verdict — a scope-note entry for R1/R2 (no row moves); for a winning R3/R4 a dated log row
  plus the table-row edit it refers to — with `pytest
  tests/test_jax_example_device_assignment.py` green, and the backlog plan's §Probe outcomes
  row and P2.4 checkbox updated in that same commit.
