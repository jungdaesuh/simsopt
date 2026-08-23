# Stochastic stage-two native-vs-GPU speed: certifying campaign charter

**Status: DRAFT — NOT FROZEN.** `[charter]` deliverable of task **P1.4**
(`docs/jax_gpu_examples_backlog_native_speed_implementation_plan.md, Phase 1 task P1.4`). A draft charter **mints nothing**:
no ratio in it is a claim, no row in `docs/jax_example_device_assignment.md` moves on it, no receipt cites it. It
becomes a preregistration only when committed with `FROZEN`, a freeze date, §2's shas, and §5.1's decision.
**Protocol authority:** the backlog plan's §Campaign protocol (`…backlog…plan.md:186-222`), cited clause-by-clause
below and never restated as if it originated here; shape follows
`docs/jax_gpu_finitebuild_native_speed_successor_plan.md`, whose warm/persistent-cache claim shape this campaign
inherits. **Slug:** `stochastic_stage_two_native_gpu`.

## 1. Why this campaign exists (motivation only)

The 2026-08-23 P1.1/P1.3 and P4.2 probes (`benchmarks/stochastic_stage_two_probe.py`,
`benchmarks/marginal_quartet_probes.py`, self-stamped `diagnostic-not-certifying`) supply **no admissible
number**: ≤3 timed solves per leg, no interleaving, no quiet gate, no five-pair rule, an incomplete sweep. Their
artifacts (`docs/receipts/evidence/stoch_*_20260823.json`/`.npz`, `quartet_coil_forces_*_20260823.json`) are
tracked since `fbab4f2b8` — but they are self-labelled diagnostic and belong to no `docs/receipts/` campaign
receipt, so clause 7 gives them no evidentiary standing. Every number below is re-measured
here or it does not exist.

Shipped stochastic scale — 16 training samples, 64×16 surface (1,024 points), order 24, 360 quadrature points,
budget 400, 1 MPI rank, shared-sample injection, matched `nit=400` both lanes:

| rung | policy | GPU warm (s) | native sampled-best (s) | ratio | GPU cold-in-process (s) |
| --- | --- | --- | --- | --- | --- |
| stochastic-mc10 | `maxcor=10` both lanes | 22.150 / 22.147 | 29.994 (OMP=16) | **1.35×** | 156.9 |
| stochastic-mc400 | `maxcor=400` both lanes | 24.617 / 24.619 | 30.525 (OMP=16) | **1.24×** | 213.9 |

Sweep observed: mc10 41.320 (4), 32.257 (8), **29.994 (16)**, 33.405 (32); mc400 41.402 (4), 33.200 (8), **30.525
(16)**, 50.720 (32) — `OMP ∈ {2,48}` **not measured**, a *sampled* optimum §5 refuses; and each native value is
one fresh-process solve (each artifact's only solve, labelled `cold_in_process`), an N=1 statistic §6 will not
admit. Endpoints GPU
`2.879160e-05` vs native `2.864290e-05` (mc10) and `2.368170e-05` vs `2.371265e-05` (mc400, GPU marginally
better); fingerprint `c96661e2c9…` identical on every leg. **Cold is a heavy bounded negative** (157–214 s of XLA
compile) against warm 22–25 s, so the claim shape is warm/persistent-cache only — the finite-build precedent
(`docs/receipts/stage_two_finitebuild_native_gpu_successor.md`).

**The sample-tile lever measured neutral-to-negative and is NOT chartered.** At `maxcor=400`, warm: tile 4 →
25.047/25.046 s, tile 8 → 24.938/24.967 s, tile 16 → 25.210/25.224 s vs **24.617/24.619 s untiled**. The
sequential-scan objective (`sample_tile=None`, `src/simsopt_jax/objectives/stochastic_stage_two.py:64-68`) is both
the production lane and the chartered configuration; the tiled path stays unchartered and no leg may set
`--sample-tile`.

**Coil-forces rung (P4.2).** The mirror (`examples/jax/3_Advanced/coil_forces.py`) timed 24.542 / 24.757 s warm
(`solve_call_seconds`, persistent-cache repeats) against the native script (`examples/3_Advanced/coil_forces.py`)
`minimize_region_seconds` best 40.034 s at OMP=16 — ≈**1.6×** — with the mirror's endpoint objective **better**
(`2.7704e-05` vs the script's stdout-parsed `2.9e-05`). Native is noisy (40–82 s over `{4,8,16}`, two cold
processes each) and the comparison is **shipped-vs-shipped**: `maxcor` 300 native vs `min(max_steps, 300)`,
tolerances unmatched (`policy_matched: false`), which §5.1 must resolve before freeze.

## 2. Instruments (frozen at freeze time)

Both tracked and clean at drafting (commit `76f1b5f37`); `benchmarks/probe_conventions.py` supplies identity,
ledger, interleave and the pinned environment:

| rung | instrument | content sha256 at drafting |
| --- | --- | --- |
| stochastic-mc10, -mc400 | `benchmarks/stochastic_stage_two_probe.py` | `7bca9f01162e1159a839b6d7de631a931445d94b86364baf91d342e13ca635bc` |
| coil-forces | `benchmarks/marginal_quartet_probes.py` | `93888b30cf5c4239436ac4b478b8a01e62cb2504a9e1ab405a47aacf789944bd` |
| both | `benchmarks/probe_conventions.py` | `84f12403b2dc7ac61673b8baa445b514ccb9db329ea52737fe8c47d70274edd2` |

**Freeze rule** (clause 1, `…backlog…plan.md:190-193`; `…finitebuild…successor_plan.md:7-8,247-251`): the freezing
commit records each instrument's commit sha and content sha256 here, and **no evidence leg may run against an
instrument whose content sha256 differs from the frozen one**, or before that commit exists; a post-freeze change
needs a dated pre-evidence amendment (§10) re-freezing the sha and invalidating every prior leg. Shipped scripts
are never edited. **Required pre-freeze work**:

1. **Certifying driver mode** — one invocation times one leg today and interleaving is operator-owned; the driver
   must run §3's alternation in fresh processes and publish the **executed** order from the ledger (in-file or a
   thin sibling per `benchmarks/nested_ls_a100_banana_omp.py`; frozen here).
2. **Quiet gates enforced, not merely available** — `probe_conventions.cpu_utilization_delta` exists and
   **neither** instrument records it (both record `gpu_compute_processes`); sample CPU busy-ness around every
   timed leg, discarding fail-closed any leg sharing the box.
3. **Process-wall timer on both lanes** (§3's second timer: the stochastic probe publishes in-process solve
   seconds only, the quartet probe has it natively but not for the JAX lane) plus a **cache-entry warmth proof**
   (§6).
4. **Ranks × OMP support** — the probe hardcodes `mpi_ranks: 1` and pins `MPI4PY_RC_INITIALIZE=false`, while the
   native objective is `MPIObjective(Jfs, comm_world, needs_splitting=True)`
   (`examples/2_Intermediate/stage_two_optimization_stochastic.py:142`); §5's denominator law needs a rank lever.
5. **U3 harness-clone gaps** named by P1.4 — retarget `_parity_case`, factor `build_native_evaluator` out of
   `examples/jax/parity/cases/native_stage_two_optimization_stochastic.py`, add `objective_scale`; the probe calls
   its native lane "the clone", and no certifying campaign runs on one undiffed against that case's `_native`.

## 3. Win rule

Plan clause 2 (`…backlog…plan.md:194-198`; `…finitebuild…successor_plan.md:198-206`), **independently per rung** —
the two stochastic rungs are two policies, not two samples of one:

> **Five interleaved pairs per rung, alternating order**, each leg a fresh process, GPU legs serialized, affinity
> pinned, quiet-gated. **WIN** requires median paired `native_seconds / gpu_seconds ≥ 1.10` **and every one of the
> five pairs `> 1.00`**, on **both** the solve timer and warm persistent-cache `process_wall_seconds`. Anything
> else is `CLOSED_BOUNDED_NEGATIVE`; `NOT_PRODUCED` stays broken evidence, never a verdict.

- **Three rungs:** R1 `stochastic-mc10`, R2 `stochastic-mc400`, R3 `coil-forces`.
- **Solve timer of record.** R1/R2: the `minimize` call and nothing else, both lanes. R3: native
  `minimize_region_seconds` against the mirror's `solve_call_seconds`, conservative for the GPU by the probe's
  disclosure (the mirror carries host construction and the Taylor evaluation; the native region carries both
  `minimize` calls and the VTK writes between them).
- **No dual-anchor clause**: no archived anchor of the kind `docs/jax_gpu_flat675_fused_campaign_plan.md:226-245`
  binds exists here; the two rows this charter can move — `native-stage-two-optimization-stochastic` and
  `native-coil-forces` — are `unmeasured` in `docs/jax_example_device_assignment.md`, and the stage-two /
  planar-coils rows sit at `cpu`/`measured-diagnostic` outside this charter's scope (§4).

## 4. Scope law

- **R1/R2 claims are scoped to shipped scale and policy by name** — the §1 configuration, the frozen rank count,
  `maxcor` 10 or 400 matched across lanes, `sample_tile=None` — restated wherever the ratio is quoted; a
  `maxcor=10` win is not a `maxcor=400` win. R1/R2 are the only path moving
  `native-stage-two-optimization-stochastic` off `unmeasured`, through this campaign's receipt (§11); R3 the only
  path for `native-coil-forces`, scoped by §5.1.
- **The stage-two and planar-coils rows are out of scope**: their P4.1 kill numbers are recorded as
  `cpu`/`measured-diagnostic` by the backlog plan's P6.2 (`…backlog…plan.md` §P6.2), independent of this
  charter, and no verdict here is evidence about them.
- Row movement follows `docs/jax_example_device_assignment.md`'s own procedure: append a dated log row and edit
  the table row it refers to **in the same commit**, never rewriting a log entry; `gpu` requires a receipt tracked
  as a regular file under `docs/receipts/`, enforced by `tests/test_jax_example_device_assignment.py:248-331`.

## 5. Fair-native denominator

Plan clause 3 (`…backlog…plan.md:199-207`; `…finitebuild…successor_plan.md:186-189`; sweep-down correction
`docs/receipts/stage_two_minimal_coupled_route.md:574-583`):

- **Full OMP sweep `{2,4,8,16,32,48}`, every value measured, per rung and per policy.** The probes sampled
  `{4,8,16,32}` (stochastic) and `{4,8,16}` (coil-forces); this charter refuses both — the OMP law exists because
  narrow problems optimize below 16, and it has killed two false wins. **A denominator from an incomplete sweep is
  `NOT_PRODUCED`.** `--omp` is mandatory on every timed leg; the shipped default (unset threads) is disclosed
  separately, never a denominator; the GPU host pin is stated per leg (probes ran `OMP=8`) and read back from
  libgomp.
- **Two-dimensional ranks × OMP for the stochastic family** (clause 3's carve-out): the native example splits its
  16 sample objectives across MPI ranks, so the denominator is the optimum over a pinned, disclosed ranks × OMP
  grid, not over OMP at one rank — the single-rank probe number is **not** that denominator. The grid is frozen at
  freeze time; an oversubscribing cell publishes, never drops.
- **Maxcor policy law, per rung.** R1/R2 are *matched-policy* rungs: one `maxcor` for both lanes (10 = mirror
  default, `examples/jax/2_Intermediate/stage_two_optimization_stochastic.py:51,263`; 400 = native default,
  `examples/2_Intermediate/stage_two_optimization_stochastic.py:198`), same budget, same tolerances (native
  `tol=1e-15` → `ftol=gtol=1e-15`, explicit on the JAX side). **Policy and hardware are never mixed inside one
  stochastic ratio.**
- **Exclusions symmetric, as the instrument already enforces**: the five-epsilon Taylor test and the 256-sample
  out-of-sample loop (`examples/2_Intermediate/stage_two_optimization_stochastic.py:186-191,217-229`) sit outside
  the timed window on both lanes, the bundle is not materialized, and callback and print cost stays charged to
  native. The JAX lane's bypass of `serial_solve_jax`'s whole-objective evaluations and log epilogue
  (`src/simsopt_jax/solve/serial.py`) is **pro-GPU**: disclosed and timed standalone.

### 5.1 R3 matched-policy decision — open at draft, closed at freeze

Two legitimate options; one goes into the frozen text. **(a) Match `maxcor` and tolerance across lanes**, as R1/R2
do — but neither shipped lane runs that configuration, so the number says nothing about the device row's
semantics. **(b) Charter shipped-vs-shipped explicitly** — each lane timed as it ships, both policies published on
both legs (`native_policy`, `mirror_policy`) — **plus an endpoint-quality gate**: the mirror's endpoint must be no
worse than the native script's, or `NOT_PRODUCED`.

**Recommended: (b).** It matches what a device-assignment row means — "where do I launch *this example*, as
shipped" — the only question such a row answers; and the endpoint gate protects it, since the objection to a
policy-mixed ratio is that the faster lane may simply be optimizing less. The probe measured the mirror's endpoint
*better* (`2.7704e-05` vs `2.9e-05`), so the gate is a live constraint, not a formality. Its weakness is bounded
and disclosed: the native objective is parsed from a `.1e` stdout line (two digits), so the frozen text either
accepts that bound or lands a fuller-precision capture pre-freeze. Under (b) the ratio **must never be quoted as a
matched-work number**, and the receipt says so beside it.

## 6. Warm/cold scoping

Plan clause 4 (`…backlog…plan.md:208-211`). Claims are **warm same-process** and **warm persistent-cache** only;
the lever is `JAX_COMPILATION_CACHE_DIR` via `--compile-cache` with `JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0`
and `JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0` already pinned
(`benchmarks/stochastic_stage_two_probe.py:243-245`); absent = cold lane, and the scrub-then-pin environment stops
an inherited cache warming it. **Cold is measured and published, never folded into a warm claim**: the probe's
156.9 s and 213.9 s colds against warm 22.2 / 24.6 s are why the claim is warm-scoped, and a cold claim exists
only if the cold numbers independently pass §3, which here they will not. Cold/warm is symmetric (index 0 is
`cold_in_process` and excluded on **both** lanes); the campaign publishes fresh-process empty-cache colds, not
in-process first solves; module-import time is excluded from every ratio.

## 7. Physics gate

Plan clause 5 (`…backlog…plan.md:212-214`). Not exact greedy arithmetic, so the gate is the **`native_workflow`
tolerance bucket** (`src/simsopt_jax/parity_tolerances.py`), not bitwise:

- **Endpoint agreement, every timed pair**, through the instrument's **metadata-gated** `compare_endpoints`, which
  refuses two endpoints disagreeing on `scale`, `budget`, `maxcor`, `training_sha256` or
  `initial_parameters_sha256`, so a difference can never be a configuration difference in a physics costume.
  `dofs_within_bucket` and `objective_within_bucket` must both be true; iteration counts publish per side,
  ungated.
- **Shared-sample fingerprint identity is mandatory**: both lanes draw the perturbations from one PCG64DXSM stream
  through `materialize_stochastic_coil_perturbations` (`src/simsopt_jax/examples/stochastic_samples.py:133`, the
  PCG64DXSM stream at `:158`)
  and record the bundle `sha256`; a pair whose `training_sha256` differs is `NOT_PRODUCED`.
- **The nit-at-exhaustion disclosure must be confirmed on the first GPU pair.** The publication gate assumes the
  fused JAX driver reports `nit == budget` at exhaustion as SciPy does; if it reports `budget-1` the gate refuses
  the leg fail-closed rather than minting a number (`benchmarks/stochastic_stage_two_probe.py:223,435-439`). The
  probe legs show `nit: 400, status: 1` on every leg, with `nfev` 405 on all mc400 legs and 418–422 on the mc10
legs (418 on both legs of the mc10 timed pair) — a *diagnostic* observation the first certifying GPU pair
  re-confirms, updating that line in the same commit. R3 additionally uses §5.1's endpoint gate with its two-digit
  bound disclosed.

## 8. Provenance, ledger, quiet gates

Plan clause 6 (`…backlog…plan.md:215-217`). Every leg publishes `probe_conventions.runtime_identity(lane)` —
commit plus per-changed-file sha256, host, library versions, `simsoptpp` sha256, jax devices/backend/x64, XLA
flags, threading environment, observed OpenMP threads, loadavg, unfiltered `nvidia-smi` processes plus `own_pid` —
and appends one line per leg to `docs/receipts/evidence/probe_leg_ledger.jsonl`, the executed-order proof. The
environment is `probe_conventions.pinned_environment`, the native lane *positively* pinned to `JAX_PLATFORMS=cpu`,
`JAX_ENABLE_X64=1`, `CUDA_VISIBLE_DEVICES=""` so its transitive JAX import takes no CUDA context and evaluates
nothing in fp32. **Cross-leg conformance is fail-closed**: legs of a rung disagreeing on instrument sha, commit,
`simsoptpp` sha, jax version, x64 state, device or pins make it `NOT_PRODUCED`. The unmodifiable native
coil-forces child publishes `parent_identity`, `child_pid` and `child_environment_pinned` — nothing about its own
state. Clean tree at every pair leg.

## 9. Kill criteria (fail-closed; none amendable post-evidence)

- Median paired ratio `< 1.10` on either required timer, or **any single pair `≤ 1.00`** → that rung closes
  `CLOSED_BOUNDED_NEGATIVE`.
- Endpoints outside the `native_workflow` bucket, a `training_sha256` / `initial_parameters_sha256` mismatch, or a
  `compare_endpoints` identity refusal → `NOT_PRODUCED`; a forked trajectory is an instrument or input defect, not
  a slow lane.
- An incomplete OMP sweep or ranks × OMP grid, an unset `--omp` on a timed leg, a denominator taken from the
  unset-threads shipped default, or an instrument content sha changing between freeze and the last pair →
  `NOT_PRODUCED`.
- A solve refused by the publication gate (`nit < 1`, or short of the budget without converging) → leg discarded,
  never published as a fast leg; a warm leg whose cache-entry count grows → not warm; foreign CPU or GPU compute
  during a timed leg → leg discarded and re-run, three discards on a rung → `NOT_PRODUCED`.
- Any leg with `--sample-tile` set → `NOT_PRODUCED`, outside the chartered configuration. R3 only: mirror endpoint
  worse than the native script's under §5.1(b) → `NOT_PRODUCED`.

## 10. Amendment discipline and execution order

§§2–9 and §11 are **frozen text** at the freezing commit. Dated amendments are permitted **only before the
evidence they govern exists**, are append-only, never edit frozen text in place, and each cites its empirical
basis by artifact path and sha256 (clause 1, `…backlog…plan.md:190-193`; `…finitebuild…successor_plan.md:247-251`;
`docs/jax_gpu_flat675_fused_campaign_plan.md:267-271,293-294`). **Non-amendable post-evidence:** the win rule
(§3), the scope law (§4), the denominator rule including the full OMP set and the ranks × OMP law (§5), the §5.1
option chosen, warm/cold scoping (§6), the physics gate (§7), every kill criterion (§9). Amendable pre-evidence
only: rung ordering, repetitions above the five-pair floor, the rank set, and the instrument shas (re-freezing
invalidates prior legs). Execution order: **freeze** (§2, §5.1) → **native grid** (§5) → **warm/cold** (§6) →
**pairs** (R1 → R2 → R3) → **publication** (§11), nothing running before the freezing commit exists.

## 11. Evidence layout, receipt, and scoreboard

Plan clause 7 (`…backlog…plan.md:218-222`):

- Tracked evidence bundle under `docs/receipts/evidence/stochastic_stage_two_native_gpu/`: per-leg artifact JSON,
  endpoint `.npz` pairs, the sweep reductions, the pair schedule and ratios, the `compare_endpoints` reports, the
  leg-ledger slice. The 2026-08-23 probe artifacts stay under `docs/receipts/evidence/`: motivation, never
  evidence.
- Tracked terminal receipt at `docs/receipts/stochastic_stage_two_native_gpu.md`: every rung's verdict, both timer
  names, every pair ratio, the sweep tables, the shipped-default disclosure, the cold numbers, the endpoint-bucket
  verdicts, the sample fingerprint, §5.1's statement for R3, and the tile measurement as a recorded non-lever —
  **all published whatever the verdict**.
- `docs/jax_example_device_assignment.md` amended **in the same commit** as the verdict. **Pre-registered
  consequences:** an R1 or R2 WIN moves `native-stage-two-optimization-stochastic` from `unmeasured/unmeasured` to
  `gpu/measured-certified` citing this receipt, the warm/persistent-cache scope and cold loss stated in the row
  itself (the finite-build row is the template); a close leaves it `unmeasured` unless the evidence establishes
  the native lane faster, in which case `cpu/measured-certified`. An R3 WIN moves `native-coil-forces` the same
  way, §5.1(b)'s semantics named in the row. The stage-two and planar-coils rows are untouched (§4).
- `pytest tests/test_jax_example_device_assignment.py` green in that commit, with the backlog plan's §Probe
  outcomes row and P1.4 entry updated alongside.
