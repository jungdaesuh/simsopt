# Projected-route certification protocol — single-stage VMEC-free, GPU beats native

Status: PHASES 1–3 LANDED (CPU machinery, GPU attempt-protocol launcher,
shipped example), phase 4 NOT STARTED. No root has been opened.
Route under certification: `projected-lagrangian-newton-cg`
(`src/simsopt_jax/geo/optimizers/projected_lbfgs.py`, entry
`run_projected_lbfgs`), configuration frozen at
`benchmarks/rehearse_single_stage_projected_route_cpu.py::CERTIFIED_ROUTE_OPTIONS`.

This document is the single source of truth for what the projected route
claims, what evidence discharges the claim, and in what order that evidence may
be produced. It supersedes nothing: the DIAG5 plan
(`single_stage_jax_gpu_native_equivalent_quality_diag5_native_binding_recovery_plan.md`)
remains the frozen SSOT for the *predecessor* GNTR route, whose roots are all
spent, and whose bytes must not be edited — its plan sha is bound into five
constant sites and four published review records.

---

## 1. The claim

> On the audited full single-stage VMEC-free examples workload, the custom JAX
> GPU route reaches the native C++ reference's endpoint objective, at strictly
> better feasibility, in less wall time than native spent — compile included.

Discharged by three numbers and one boundary:

| Quantity | Contract | Provenance |
|---|---|---|
| Quality | terminal objective `Φ ≤ 4.4822246533126125e-08` | native reference's 1000-BFGS-iteration endpoint; native was budget-exhausted, not converged, so the claim is *"reach native's endpoint"*, never *"converge better than native"* |
| Feasibility | `‖c‖_inf ≤ 1e-10` at **every** recorded iterate, raw (unscaled) equalities | the route's own `feasibility_tolerance`; native enforces its constraints by inner Newton elimination and has no comparable published figure |
| Speed | wall `< 287.30421751597896 s` | native reference wall for the same workload |
| Boundary | wall = engine **compile + solve**, excluding problem bootstrap and the identity binding gate | see §3 |

Reference measurements the claim is being asked to reproduce under receipt
(RTX 5090, commit `5ab98d15b`, compile-inclusive):

| Arm | wall (s) | Φ | iterations | status |
|---|---|---|---|---|
| Q1 | 168.02226769109257 | 4.4441279448375905e-08 | 357 | OBJECTIVE_TARGET_REACHED |
| Q2 | 190.94799917202909 | 4.4782553066432176e-08 | 401 | OBJECTIVE_TARGET_REACHED |

Evidence sealed read-only at
`~/simsopt-campaigns/projected-lbfgs-goal-evidence-20260812/`.

### 1.1 Quality parity is defined per term, not on the total

A total objective and a feasibility number cannot distinguish *reached the same
physics* from *reached the same scalar*. Every receipt therefore carries a
**per-term endpoint ledger**: the five raw objective terms (`non_qs`,
`residual`, `iota`, `major_radius`, `length`), the eight observables (`iota`,
`G`, `volume`, `major_radius`, `total_length`, `non_qs_ratio`,
`boozer_residual_scalar`, `boozer_residual_rms`), both constraint blocks, and
the two non-geometry state components — for the run's endpoint **and** for the
sealed native endpoint, both evaluated through one executable of this
repository's objective.

Measured against native at the banked 5090 endpoints — every number below is
the CPU re-evaluation of §12.8, not a recollection:

* `volume` is machine-identical (7.6e-16 / 5.0e-15 relative), being an equality
  constraint.
* `iota` agrees to 2.2e-07 (Q1) and 3.9e-06 (Q2); `major_radius` to 5.6e-06
  (Q1) and 1.0e-06 (Q2); `total_length` to 2.3e-06 (Q1) and 3.0e-08 (Q2).
  **An earlier revision of this section claimed 1e-6 or better on all four and
  the bands were drawn from that claim; the measurement refutes it** — see
  §12.8, adjudication 2.
* non-QS is slightly *better* than native (0.885% on Q1, 0.087% on Q2).
* Boozer residuals are machine-zero on both sides (5.8e-13 / 3.8e-12 absolute
  against a 1e-10 band).
* **`G` is the one materially differing observable**, 0.785% (Q1) and 0.934%
  (Q2) below native.

`G` is **reported, never gated**. The non-QS term is a field-scale-invariant
ratio and nothing in the shared objective pins the net poloidal current, so `G`
is a flat valley direction along which distinct equal-quality minima exist —
consistent with the terminal-to-native scaled distance of ~2.3. A `G` gate
would manufacture a false reject on a direction the objective deliberately
leaves free, which is the V260/ρ-floor failure class this campaign has now hit
three times.

`total_length` is **gated on the longer side only** for the same reason in a
weaker form: its penalty is `0.5·max(L − target, 0)²`, exactly flat below the
target, and the banked Q1 latch sits in that flat region with `raw.length`
exactly `0.0`. Below native's length the objective pins nothing; above it, it
does.

Owner of the two sets: `PINNED_ENDPOINT_QUALITY_TERMS` and
`INFORMATIONAL_ENDPOINT_OBSERVABLES` in the rehearsal module. An artifact may
not restate them — validation compares the recorded sets against the module's
before reading the ledger. The ledger is *reported* on every attempt and
*gated* on the attempt that discharges the claim: a latch at the certified
budget, and nothing else (`endpoint_ledger_is_gated`; §12.8, adjudication 1).

### 1.2 What is NOT claimed

* Not a convergence claim. Native's endpoint is a budget exhaustion point.
* Not a hardware-general speed claim. The speed result is RTX 5090 specific
  (§7).
* Not a claim about any other route in this repository. The GNTR/FTR/SQP
  families are closed with recorded negative verdicts.

**After the root ran, §12.14 is the authoritative claim statement.** It records
the measured ratio family, the exact NOT-claimed list (reviews, native-bar
provenance, parity scope), and the two gaps closed on 2026-08-13. Quote §12.14,
not this section, when stating the result outside this document.

---

## 2. Problem identity is bound by observables, never by the problem sha

The exact-numeric sha of the bootstrapped `FullSpaceProblem`
(`exact_numeric_tree_sha256`) is **not reproducible across processes**, not
even on one box:

| Launch | problem sha | bootstrap sha |
|---|---|---|
| CPU (this repo) | `e6df89b6…` | `9d3dd46e…` |
| RTX 5090 Q1 | `3ca54b8b…` | `ee2b65ed…` |
| RTX 5090 Q2 | `21b0efc3…` | `8e1357f4…` |
| A100 smoke | `9ac1c0b4…` | `e8c14ee7…` |
| A100 Q1/Q2/Q3 | `1c8dddd0…` | `b5c2afbc…` |

Two runs of the same commit on the same GPU produced different shas. **No
certification gate may reference the problem sha or the bootstrap sha.** They
are recorded as provenance and explicitly marked non-binding
(`problem_identity.sha_is_binding = false`); an artifact that claims otherwise
is refused at validation.

Identity is instead bound by reproducing the bootstrap point's observables,
which agree to ~1e-14 relative across every backend the route has run on:

| Observable | CPU reference | gate |
|---|---|---|
| `objective` | 8.44421289101312e-05 | relative ≤ 1e-10 |
| `gradient_norm` | 0.011322200934491376 | relative ≤ 1e-8 (sums 716 contributions) |
| `projected_gradient_norm` | 0.0006791192049597319 | relative ≤ 1e-6 (through a Gram of condition 3.6e6) |
| `feasibility_inf` | 1.8708818992040455e-14 | absolute ≤ 1e-10 on **both** sides |

Feasibility is gated absolutely on purpose: both sides sit four decades below
the tolerance the route enforces, so a relative comparison of them would
measure rounding and nothing else.

Owner: `benchmarks/rehearse_single_stage_projected_route_cpu.py`
(`CPU_BOOTSTRAP_OBSERVABLES`, `BOOTSTRAP_BINDING_RELATIVE_TOLERANCES`,
`bind_problem_identity`). The GPU lane must import these, not restate them.

---

## 3. Timing convention

The certified wall is **engine compile + engine solve**, i.e.
`ProjectedLbfgsRun.compile_seconds + ProjectedLbfgsRun.solve_seconds`. Building
the `FullSpaceProblem` and running the identity gate are setup that native's
287.30 s does not contain either (bootstrap costs ≈14.5 s on the 5090, ≈7.2 s
on this CPU box). Every artifact states the boundary explicitly in a
`timing_boundary` field so the comparison is checkable rather than assumed.

**Compile is INSIDE the claim.** The predecessor route's gate excluded compile
while native's bar excluded nothing, which is not a comparison. Two lanes are
published:

* **Warm lane (timed, certified).** Persistent JAX compilation cache, primed by
  a preflight process, so the certified run's compile is a cache load. Config,
  verified on `.venv-qn-gpu` across separate processes:
  * `JAX_COMPILATION_CACHE_DIR=<dir>`
  * `jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)`
  * `jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)`

  Both knobs are required: the defaults skip small and fast kernels, which are
  most of this route's bundle. Measured probe: 0.031 s warm against 0.095 s
  cold in a second process, cache entries on disk.
* **Cold lane (documented, not timed against the bar).** One full cold-compile
  run published alongside, so the cache is an accounting device and not a
  hiding place.

The cache directory is part of the sealed artifact's evidence: its path, its
entry count and its aggregate digest are recorded, so a reader can tell warm
from cold.

---

## 4. The latch is a stochastic draw — pre-registered attempt design

The A100 replication settled a question the 5090 runs could not: **the route's
quality latch is a draw, not a deterministic outcome.**

| Box | latched | walls (s) |
|---|---|---|
| RTX 5090 | 2/2 | 168.02, 190.95 |
| A100 | 2/3 | 316.87, 301.02 (no-latch arm: 264.14) |

The A100 no-latch arm terminated `LINE_SEARCH_COLLAPSE` at `Φ ≈ 8.96e-8`: the
carried period-4 projector reached true tangency ≈ 12.5 at ages 2–3, the
retraction hit its 8-correction cap, and the step scale fell to 9.46e-7, below
the 1e-6 floor. The 5090 flirted with the same mode (tangency max 3.98 on Q1,
5.30 on Q2) and latched anyway. The trajectory band across runs is driven by
the bootstrap nondeterminism of §2, not by silicon.

A single-trajectory one-shot root can therefore be burned by a no-latch draw
that indicts nothing. The root protocol must be pre-registered **before the
root is opened**, and it is:

> **Frozen attempt protocol.** One root authorizes exactly **N = 3 sequential
> attempts** of the frozen configuration, run in one supervised session on one
> GPU. Attempts run in order and stop at the first that reaches
> `OBJECTIVE_TARGET_REACHED`. Every attempt — latching or not — publishes its
> full telemetry into the same sealed artifact as `attempts[k]`. The claim is
> discharged **iff the first latching attempt's wall is under the bar**; the
> artifact additionally reports the latch rate `k/N` and the wall of every
> attempt.

Semantics that must be recorded in the artifact, not inferred:

| Outcome | Artifact verdict | Root disposition |
|---|---|---|
| An attempt latches under the bar, at the pre-registered conformance | `CLAIM_DISCHARGED` | root spent, successfully |
| All N attempts complete, none latches | `NO_LATCH_IN_PROTOCOL` | root spent; the claim is *not* refuted, the draw failed. A successor root requires **new user authorization** and is never automatic (§12.1) |
| An attempt latches over the bar, **or** under it at any conformance other than `PREREGISTERED` | `QUALITY_ONLY` | root spent; quality replicated, speed not claimed |
| Any attempt fails a gate (identity, feasibility, receipt) | `GATE_REFUSED:<gate>` | root spent; this is a defect report, not a science result |

There is no undefined outcome. Roots 1–4 of the predecessor route all died in
stages whose semantics had never been written down.

`CLAIM_DISCHARGED` is conditioned on `attempt_protocol.conformance` as well as
on the wall — see §12.8, adjudication 3. Telemetry that is not in the §6 gate
order can never produce an outcome outside this table: a GPU-memory sampler
failure is absorbed as degraded telemetry, not raised (§12.8, adjudication 4).

---

## 5. Receipts are tolerance-certified, state bindings are exact

The predecessor route's fourth root failed **after a complete solve**, at
receipt publication, because the terminal validator demanded bitwise equality
between values produced by two independently compiled executables (measured
divergence ≈ 4 ULP). That gate class is banned.

| Comparison class | Rule |
|---|---|
| Cross-executable numeric values (endpoint objective, objective terms, observables) | relative `1e-11` with absolute floor `1e-19`, via `certify_agreement` in `benchmarks/single_stage_native_equivalent_quality_diagnostic_receipt.py` |
| Raw equality residuals against the native reference | the frozen raw-equality tolerances (`1e-10` relative, `1e-12` absolute) |
| State SHAs, same-source copies, artifact-tree digests | **exact** |
| Feasibility | absolute threshold on both sides, never relative |
| Free directions of the shared objective (`G`) | reported, never gated (§1.1) |

There is exactly one implementation of the tolerance rule
(`certify_agreement`, made public in this phase). Any second implementation is
a twin, and twins drift — see mistake-book P153.

---

## 6. Gate order (a certified run executes exactly this sequence)

1. **Environment gate** — platform, x64 and allocator pinned by equality; a
   missing variable is a different run, not an error.
2. **Execution-source binding** — every imported module under
   `benchmarks/`, `examples/`, `src/` must hash to its entry in
   `benchmarks/single_stage_native_equivalent_quality_gntr3_execution_sources.json`,
   asked *after* the imports happened. This is the gate that catches the
   scikit-build-core editable finder outranking `PYTHONPATH`.
3. **Problem bootstrap** — builds the `FullSpaceProblem` and reads every data
   file it needs (`src/simsopt/configs/NCSX.dat` among them; its absence spent
   a root).
4. **Identity binding** — §2.
5. **Lowering pre-gate** — §6.1.
6. **Engine compile** (timed, warm-cache lane).
7. **Solve** (timed) — attempt protocol of §4.
8. **Endpoint certification** — §5 — and the per-term endpoint ledger of
   §1.1, gated on the pinned set on the attempt that discharges the claim: a
   latch at the certified budget (§12.8, adjudication 1).
9. **Independent re-validation** of the receipt bytes, in-process, against the
   still-writable staging tree — it GATES step 10 rather than annotating it. A
   refusal leaves an unsealed tree carrying the refusal and publishes nothing
   (§12.8, adjudication 5). It re-derives the claim rather than checking the
   receipt against itself, and refuses a receipt that is not complete (§12.8,
   adjudication 7).
10. **Sealed publication** — 0444/0555, artifact manifest written last,
    `renameat2(RENAME_NOREPLACE)`, parent fsync, sealed modes re-checked from
    the published tree.

Every external resource the protocol depends on — the NVIDIA tooling, the GPU
UUID the receipt names, the sealed native endpoint of §12.8 adjudication 6, and
the temporary, cache and output storage every child writes through (§12.8,
adjudication 9) — is preflighted before the first child is spawned and before
the staging tree exists, so none of them can spend the root at step 3.

### 6.1 Lowering pre-gate (`.lower()` without `.compile()`)

`lower_projected_lbfgs_kernels` lowers exactly the kernels the configuration
selects, with every argument beyond the start point taken abstractly from
`jax.eval_shape`. Nothing reaches a backend compiler — proved by a test that
counts `jax._src.compiler.compile_or_get_cached` calls and requires zero.

The gate is **self-referential and carries no frozen size constant**: the route
is lowered at the rehearsal budget and at the certified budget, and the two
must produce identical IR. That is the invariant the entire bounded-rehearsal
idea rests on — the loop runs on the host, so no kernel may carry the attempt
budget. When it fails, a short rehearsal is compiling a strictly smaller
program than the run it stands in for, which is precisely the regression the
predecessor's fused loop shipped when its safeguard unrolled the attempt body
three times and its CPU compile grew to seventy minutes.

Per-kernel IR sizes and `stablehlo.while` counts are published as evidence, so
a blowup is visible to a reviewer without any threshold having to be guessed.

---

## 7. Supplementary lane: A100 replicates quality, not speed

Evidence sealed read-only at
`~/simsopt-campaigns/projected-lbfgs-a100-replication-20260813/`
(`A100_REPLICATION_verdict.json`, schema `proj-lbfgs-a100-replication-v1`).

* Quality **replicated**: 2/3 arms reached `Φ ≤ 4.48e-8` with feasibility
  ≤ 1e-10 throughout and monotone descent.
* Speed **not replicated**: every latching A100 arm sits above the bar
  (301.02 s and 316.87 s, i.e. 1.05× and 1.10×).
* Mechanism: compile is 3.08× slower (host-clock bound), launch-latency
  dominated phases run 1.4–2.5× slower, while dense fp64 phases are *faster* on
  the A100. Hardware: A100-PCIE-40GB, driver 470.256.02 with the cuda-compat
  12-6 shim, jax/jaxlib 0.10.0.

The certified claim is stated for the RTX 5090 and cites this lane as
replication of quality on independent silicon.

---

## 8. Phase 1 — bounded-rehearsal CPU gate (LANDED)

Rationale (standing user verdict): a full-budget CPU qualification gate is not
usable — hours of uncached compile against second-scale failures. The successor
shape is a bounded rehearsal.

`benchmarks/rehearse_single_stage_projected_route_cpu.py` runs the whole
sequence of §6 on CPU at a three-attempt budget, publishes a sealed artifact
and re-validates it, in about two minutes. Measured phase costs on this box
(64-core Ryzen 9 9970X, CPU backend, chain wall 114.4 s): bootstrap ≈8.0 s,
identity ≈25.1 s (pays the first point-evaluation compile), lowering pre-gate
≈17.2 s (two traces per kernel per budget, no compile), engine compile
≈23.8 s, three attempts ≈36.1 s. The published lowering evidence at this
freeze totals 65.2 MB of IR across the six selected kernels.

Substitution soundness rests on two asserted properties:

1. **Same program.** The lowering pre-gate proves rehearsal IR ≡ certified IR.
2. **Same route.** Every budget is `CERTIFIED_ROUTE_OPTIONS` with
   `maximum_iterations` replaced, and the artifact publishes the resulting
   delta (`certified_options_delta`), which must be `{"maximum_iterations": 3}`
   and nothing else.

The rehearsal claims nothing scientific: three attempts cannot reach the
target, and the receipt says so in a `quality_claim` field rather than leaving
a reader to infer it from a status code.

**Rehearse-before-root rule.** The rehearsal's own entry path is executed as
launched, in a subprocess, with zero monkeypatched module constants, by
`tests/benchmarks/test_rehearse_single_stage_projected_route_cpu.py`. A
fifty-nine-test suite was green while the predecessor's launcher raised
`NameError` in its first phase, because the suite imported the module and
monkeypatched the very constant that was missing. Every other test in that file
reads the artifact that subprocess published.

---

## 9. Phases 2–4

The order below is the execution order, and it is binding: the examples script
lands **before** the root opens (§12.6).

**Phase 2 — GPU lane. LANDED** as
`benchmarks/run_single_stage_projected_route_gpu_root.py`. Persistent-cache
preflight and the warm/cold pair of §3; the attempt protocol of §4; artifact
schema extended from the rehearsal's with `attempts[]`, the cache evidence, the
GPU runtime identity, and the sealed source snapshot of §12.4.

Two shapes the build had to settle. **Each attempt is its own process**: a
second attempt run in the launcher's process inherits the first's `jax.jit`
caches and reports a compile of milliseconds, and the claim is discharged by
the first LATCHING attempt's wall, which is not necessarily the first
attempt's. And **the cold lane runs first against the empty cache**, so one run
is both the honest cold measurement and the process that primes what the timed
attempts load — the supervisor reuses the campaign's `publish_immutable_snapshot`,
its `_enumerated_source_roots` role bindings and its PID-and-device-bound GPU
memory monitor, while the frozen configuration, the identity gate, the lowering
pre-gate, the endpoint ledger and the sealing primitives are imported from the
rehearsal module rather than re-spelled. Re-validation RECOMPUTES rather than
reads back: the verdict from the attempts, the sealed 0555/0444 modes from the
tree, the pinned-term verdicts from the ledger's own published terms, and the
claim's quality quantity as a NUMBER — the recorded `objective_target` and the
recorded terminal objective are both compared against `NATIVE_TARGET_OBJECTIVE`,
because `OBJECTIVE_TARGET_REACHED` is the optimizer reporting against whatever
target it was configured with, not against this claim's. A child whose stdout
did not carry its canonical document is classified `PROTOCOL_FAILURE` and
published; it does not escape the outcome space and abort the root unpublished.

Measured on the RTX 5090 at a bounded three-iteration budget (not a root):
cold compile 13.8 s and solve 43.1 s against warm compile 3.9 s and solve
9.9 s, the two processes agreeing on Φ and on every recorded feasibility.

**Phase 3 — examples landing. LANDED** as
`examples/jax/3_Advanced/single_stage_boozer_vacuum_projected_route.py`, a new
script beside the nested mirror rather than a rewrite of it. The mirror's value
is that it reproduces the native example term for term — its correctness file
asserts that numerically against `examples/3_Advanced/` at 1e-12 relative — and
switching its formulation in place would have deleted that evidence to satisfy
a wording point. The new script runs the certified configuration on the same
audited workload, and
`tests/jax/examples/test_single_stage_boozer_vacuum_projected_route_example.py`
pins its options to `CERTIFIED_ROUTE_OPTIONS` field for field so the two
spellings cannot drift.

It is deliberately **not** registered in `examples/jax/manifest.json`. A `ready`
manifest entry is executed by the tree's strict-transfer-guard lane, and the
route's host-driven loop reads device scalars with `float()` and `bool()`,
which that guard refuses; registering it would require re-plumbing the engine's
host boundary through `simsopt_jax.runtime.host_boundary`, and the engine under
certification must not change between the reviews and the one-shot root. The
registration is the first item of phase 4's follow-on work, not of this freeze.

**Phase 4 — refreeze, reviews and root.** Refreeze (§10, which the examples
file makes mandatory) → four independent GO reviews under a fresh reviews root
→ authority JSON → one preflight → the one-shot root. Same-role reviewer reuse
across phases is allowed; cross-role is not.

---

## 10. Refreeze recipe (execute in this order, manifest LAST)

Any new file under `benchmarks/`, `examples/` or `src/` changes the
execution-source membership. The counts are twins and must move together —
grep the **old value** repo-wide, never the symbol (mistake-book P153; the
A1/A1b incident cost a crucible cycle).

1. `benchmarks/qualify_single_stage_native_equivalent_quality_gntr3_cpu.py`
   → `_BROAD_EXECUTION_SOURCE_COUNTS` per-root counts.
2. `benchmarks/single_stage_native_equivalent_quality_successor_authority.py`
   → `DIAG5_EXECUTION_SOURCE_ENTRY_COUNT`.
3. `benchmarks/run_single_stage_native_equivalent_quality_campaign.py`
   → `_DIAG5_BOOTSTRAP_EXECUTION_ENTRY_COUNT` (the import-free bootstrap twin).
4. Any literal assertion of the same value in
   `tests/benchmarks/test_run_single_stage_native_equivalent_quality_campaign.py`
   and `tests/benchmarks/test_single_stage_native_equivalent_quality_diagnostic_receipt.py`.
5. **Manifest last**: regenerate
   `benchmarks/single_stage_native_equivalent_quality_gntr3_execution_sources.json`
   with `benchmarks/regenerate_execution_source_manifest.py`. It does not
   restate the membership rule — it imports
   `_diag4_execution_source_membership`, the same rule the certification gates
   validate against (members = `rglob("*.py")` under the three broad roots ∪
   the 25 qualified paths ∪ the 11 frozen-numerical paths, minus the manifest;
   canonical JSON, `entries_sha256` recomputed). It refuses every membership
   change the operator did not name: a path entering or leaving without
   `--admit`, an admitted path that is already a member or that the rule does
   not select, and any final count other than `--expect-count`. A refusal
   writes nothing.

   ```
   PYTHONPATH=src:. .venv-qn-cpu/bin/python \
       benchmarks/regenerate_execution_source_manifest.py \
       --admit <new/broad/root/file.py> --expect-count <n>
   ```

6. Validate all suites green, then commit.

At this freeze: **614 entries** (benchmarks 116, examples 157, src 327, plus 14
non-broad qualified paths). The two new members are phase 2's GPU launcher
(`benchmarks/run_single_stage_projected_route_gpu_root.py`) and phase 3's
shipped example
(`examples/jax/3_Advanced/single_stage_boozer_vacuum_projected_route.py`); both
entered the DIAG2 allowlist by the step-4b rule, and the previous freeze was 612
(benchmarks 115, examples 156).

Do **not** touch the DIAG5 plan doc. Its prose "591 paths (113 `benchmarks`,
156 `examples`, 322 `src`)" is a historical statement about the DIAG5 freeze
and is already superseded by the constants; editing it would invalidate
`DIAG5_PLAN_SHA256`, `DIAG5_BLANK_PLAN_SHA256`, `DIAG5_BLANK_PLAN_SIZE_BYTES`
and the four published DIAG5 review records.

The DIAG2 allowlist **does** grow, and growing it is step 4b of the recipe. A
live source snapshot is enumerated from the tree
(`run_single_stage_native_equivalent_quality_campaign._enumerated_source_roots`),
so any new broad-root file enters it and pushes the filtered entry count past
the frozen DIAG1 baseline of 576. The designed remedy is to add the path to
`DIAG2_SOURCE_DELTA_ALLOWLIST` — a new file is not an edit to a reviewed
numerical source, so excluding it by path leaves the frozen count and digest
still describing the DIAG1 tree — and to extend the expected difference set in
`test_diag3_snapshot_closure_is_frozen_against_later_diag2_growth`, which
exists precisely to force that growth through review.

Refreezing the DIAG2 baseline itself (576 → 58x) was **vetoed**: the filter is
path-membership based, so a refreeze would invalidate every historical
artifact. Widening the test's error matcher was also vetoed (mistake-book
P154): the alternation blinds the ordered count/digest legs, and the count leg
is the one that fires first for a new file.

**Step 4c — editing a file listed in `DIAG2_FROZEN_NUMERICAL_ENTRIES`.** Those
eleven paths are pinned by exact digest, and the pins name the DIAG1 tree, not
the live one — several already differ from the checked-out bytes. Moving one
therefore does **not** mean refreezing the pin (see the veto above); the
designed remedy is `_DIAG2_ARCHIVED_FROZEN_SOURCE` in
`tests/benchmarks/_diag2_fixture.py`, which retains the historical bytes
(zlib + base85, keyed by path, its digest asserted against the pin) so the
contract suite keeps replaying DIAG1's tree while the live file moves on. Add
the entry, keep the mapping sorted by path, then continue to step 5.

---

## 11. Operating rules carried forward

* **Never retry a spent root.** Any process start spends it. Each recovery is a
  new timestamped namespace, a plan revision, a refreeze and fresh reviews.
* **Enumerate the failure class, not the instance.** One fix per root turns
  every unreached stage into a serial root burner. Before opening root N+1,
  audit-hook the whole remaining resource surface of the failed stage class.
* **Evidence in `/tmp` is volatile.** Seal to `~/simsopt-campaigns/` with mode
  0444 and a recorded sha256 before the session ends.
* **Tests run one file per process**
  (`JAX_PLATFORMS=cpu JAX_ENABLE_X64=true PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
  PYTHONPATH=src`), always with an explicit `--basetemp` outside tmpfs — the
  per-user tmpfs quota has twice mass-failed suites and once killed the harness
  itself.
* **GPU launches set `TMPDIR` outside tmpfs — and the launcher enforces it.**
  XLA spills PTX through the system temporary directory, and on a full `/tmp`
  the spill fails with `RESOURCE_EXHAUSTED: … Disk quota exceeded` *inside the
  bootstrap gate* — a third instance of the quota class, and the one that would
  spend the root. Measured 2026-08-13: the bounded GPU smoke published
  `GATE_REFUSED:bootstrap` for exactly this reason, and passed the whole chain
  when relaunched with `TMPDIR` pointed off tmpfs. `TMPDIR` is not one of the
  pinned environment variables (§6 step 1) because it names no property of the
  run; it is an operator precondition — but it is **not** "checked the same way
  free space is", because free space cannot see it: the same box reported 12.29
  GiB available and 571 769 free inodes while a one-byte write returned
  `EDQUOT` and left a zero-length file. `preflight_external_resources` therefore
  *writes* a probe byte in the resolved temporary directory, in `--cache-dir`
  and in `output_root.parent`, refuses any of the three on a tmpfs filesystem
  type, publishes what it resolved, and launches the children with `TMPDIR` set
  to it (§12.8, adjudication 9). A redirected write under a spent quota returns
  exit 0 with an empty result, so no capacity check and no exit code substitutes
  for the probe.
* **Sealed trees are 0555/0444.** Removal restores modes first and uses a
  Python walk; recursive-force shell globs are blocked by policy.

---

## 12. Adjudicated decisions

These were open questions when phase 1 landed. All six are now ruled on, and
the rulings are binding on the phases that follow. §12.8 adds six more, ruled
on after the first four-role GO review round returned unanimous NO-GO.

### 12.1 No-latch disposition — successor roots are never automatic

If all N attempts miss, the artifact records `NO_LATCH_IN_PROTOCOL` (§4) and
the arc **stops**. A successor root requires **new user authorization**: a
fresh timestamped namespace, a plan revision, a refreeze and fresh reviews, as
§11 requires of any recovery. No agent, and no rule in this document, may open
one on its own. The draw failing is not evidence against the route, and it is
not licence to spend another root either.

### 12.2 N = 3 — accepted

The observed latch rate is 4/5 across two boxes, which puts three consecutive
misses at roughly 1%. It is a judgement, not a measurement, and a larger N
costs GPU hours linearly. Accepted as the frozen protocol.

### 12.3 Tangency-driven collapse — certify the configuration as-is

`projector_refresh_period = 4` with `projector_tangency_tolerance = 0.0` is
certified exactly as the banked numbers were produced. The A100 no-latch arm's
mechanism (carried projector true tangency 12.494 at ages 2–3, retraction at
its 8-correction cap, step scale 9.456e-7 below the 1e-6 floor) is a **known,
unmitigated, accepted** risk of that pair: a positive tangency tolerance would
trade wall for latch probability, and changing it means re-measuring the bar.
The mechanism is recorded here, and both fields — with every other field of the
configuration — are pinned by
`tests/benchmarks/test_rehearse_single_stage_projected_route_cpu.py::test_certified_options_are_the_configuration_the_latches_used`,
so an optimizer default changed elsewhere cannot silently redefine what is
being certified.

### 12.4 Rehearsal scope — the GPU root artifact carries a sealed snapshot

The CPU rehearsal keeps its import-hash binding as it is: it hashes what
actually imported, which catches the editable-finder class more cheaply than an
`execve` into a copied tree and is the right cost at a two-minute budget.

The **GPU root artifact MUST additionally carry a sealed source snapshot**,
published through the existing snapshot machinery
(`publish_immutable_snapshot` and the source-role bindings that
`run_single_stage_native_equivalent_quality_campaign.py` already drives), not a
new mechanism. A one-shot root's evidence must remain readable after the tree
has moved on; an import-hash list points at bytes that no longer exist.

### 12.5 Manifest generator — landed in `benchmarks/`

Resolved: the fail-closed regenerator is
`benchmarks/regenerate_execution_source_manifest.py` (§10 step 5). Promoting it
made it a member of the manifest it writes, which is why the phase-1 freeze
came out at 612 and not 611. The current freeze is §10's 614; this paragraph
records why the count moved when the regenerator landed, and is not a second
statement of what the count is.

### 12.6 Examples landing order — before the root

The examples script lands in **phase 3, before the root opens** (§9). The
claim's wording ("the audited full single-stage VMEC-free examples workload")
is only literally true once the script ships, and landing it first puts the
certified bytes and the shipped bytes in one freeze instead of forcing a
post-certification refreeze that no review covered.

### 12.7 Phase-3 examples shape and manifest registration — ACCEPTED

Both halves of §9's phase-3 paragraph are ruled on, not merely described:

* **A new file beside the nested mirror, not a rewrite of it — ACCEPTED.** The
  mirror's value is evidence: its correctness file asserts, numerically at 1e-12
  relative, that it reproduces `examples/3_Advanced/` term for term. Switching
  its formulation in place would have deleted that evidence to satisfy a wording
  point, and the wording is satisfied by shipping the coupled formulation
  alongside it. The two spellings are held together by
  `tests/jax/examples/test_single_stage_boozer_vacuum_projected_route_example.py`,
  which pins the new script's options to `CERTIFIED_ROUTE_OPTIONS` field for
  field.
* **Deferral of `examples/jax/manifest.json` registration to phase 4 —
  ACCEPTED.** A `ready` manifest entry is executed by the tree's
  strict-transfer-guard lane, which refuses the `float()`/`bool()` device reads
  the route's host-driven loop performs. Registering it now would require
  re-plumbing the engine's host boundary through
  `simsopt_jax.runtime.host_boundary` — a change to the engine under
  certification, landing between the four independent reviews and the one-shot
  root, which §9 forbids. The registration is therefore phase 4 follow-on work,
  and the shipped script's execution evidence at this freeze is its own contract
  test, not a manifest lane.

---

### 12.8 Rulings on the first review round (all four roles NO-GO)

The pre-root reviews at `a3ea4983d`
(`~/simsopt-campaigns/projected-route-root-reviews-20260813T084549Z/`) returned
NO-GO from all four roles, on two criticals, four majors and a long minor tail.
Six rulings follow; they are binding, and the remediation implements exactly
them. The reviews' verdict on everything else stands: the manifest arithmetic,
the identity binding, the timing boundary, the lowering pre-gate, the snapshot
custody and the tolerance classes were found sound and are unchanged.

**1. The pinned-term endpoint gate fires only on the latching attempt.**
Quality parity is a claim *about the latch*: it decides whether the endpoint
that reached native's objective reached native's physics. Gated instead on
every certified-budget attempt, a non-latching attempt fails `weighted_total`'s
`not_worse` band with certainty — its objective is above the target by the
definition of not latching, and the A100 no-latch arm measures an excess of
≈1.0 against a 1e-6 band — so the first stochastic miss (≈1 in 5 by the
campaign's own measured rate) publishes `GATE_REFUSED:endpoint_ledger`, breaks
the attempt loop after one of three attempts, and makes
`COMPLETED_WITHOUT_LATCH` and therefore `NO_LATCH_IN_PROTOCOL` unreachable at a
root — dissolving exactly the insurance §4 was written to buy. Non-latching
attempts publish the ledger **ungated** and the protocol continues to the next
draw. Owner: `endpoint_ledger_is_gated` in the rehearsal module, asked by both
lanes.

**2. The bands are re-derived from equal-minima geometry.**
All five objective weights are 1.0 and the geometry penalties are
`0.5·(x − target)²` summed into a total of nonnegative terms, so at the
certified quality every penalized observable satisfies
`|x − target| ≤ √(2Φ) = 2.99407e-04`, and two *legitimate* endpoints of equal
certified quality may differ by up to `2√(2Φ) = 5.98814e-04` in that
coordinate. Dividing by the native value gives the **admissibility ceiling**:
above it the gate refuses nothing the objective permits, below the measured
spread it manufactures a false reject.

The geometry terms are gated against the native reference with bands taken as
*the next decade at or above ten times the worst deviation the two banked 5090
latches show*, capped by that ceiling. Measured on CPU through this
repository's objective (`BoundCase`; bootstrap 7.3 s, ledger 1.8 s), relative
to native:

| term | native | Q1 | Q2 | worst | ceiling | band | margin |
|---|---|---|---|---|---|---|---|
| `observable.iota` | −0.4062027259574152 | 2.210e-07 | 3.925e-06 | 3.925e-06 | 1.474e-03 | **relative 1e-4** | 25× over evidence, 14.7× under ceiling |
| `observable.major_radius` | 1.467443804809453 | 5.602e-06 | 1.031e-06 | 5.602e-06 | 4.081e-04 | **relative 1e-4** | 17.9× over evidence, 4.1× under ceiling |
| `observable.total_length` | 20.98916289206094 | −2.282e-06 | −2.986e-08 | both shorter | one-sided | **not_worse 1e-4** | see below |
| `observable.volume` | −0.2904457582995848 | 7.6e-16 | 5.0e-15 | 5.0e-15 | equality constraint | **relative 1e-6** | unchanged |

The predecessor bands were `relative 1e-6` on all four, drawn from a prose
claim in §1.1 that the banked endpoints "agree to 1e-6 relative or better". The
measurement refutes that claim: the gate the root was about to run for the
first time **refuses Q2 on iota and both arms on major radius** — the
campaign's own banked evidence, refused by the gate meant to certify it, with
§12.1 barring an automatic successor root. That is the V260 shell gate and the
SQP ρ-floor a third time, and it is why §1.1's numbers are now stated from the
re-evaluation rather than from memory.

Flat and hinged directions stay reported-never-gated on their free side. `G` is
free in both directions and remains informational. `total_length` is free
*below* its hinge — `0.5·max(L − target, 0)²` is exactly flat there, and Q1's
terminal sits in that region with `raw.length` exactly `0.0` — so it is judged
`not_worse`: unconstrained shorter, gated longer. Its band sits above the
ceiling on purpose, because a 1e-4 relative excess on L is a 2.1e-3 absolute
excess whose penalty alone is 49× the certified Φ, so no endpoint at the
certified quality can fail it for a legitimate reason.

`not_worse` stays for the QS terms (`raw.non_qs`, `observable.non_qs_ratio`)
and for `weighted_total`, on the latch only. The absolute legs (`constraint.*`,
`raw.residual`) are unchanged at 1e-10 and pass with ≥26× margin (worst measured
3.8e-12).

**Amended in the round-2 remediation: the one-sided bands are derived from this
same geometry, not asserted beside it.** The first version of this ruling
derived the ceiling for the *penalized observables* and left the QS leg where it
found it. That is the defect this ruling exists to close, one term over. The
construction transfers, through a different bound: a penalized observable is
constrained via its penalty, which is where the square root comes from, while a
term that IS one of the objective's raw summands is constrained *directly* —
all five weights are 1.0 and every raw term is nonnegative, so a latch's
`Φ ≤ NATIVE_TARGET_OBJECTIVE` bounds each summand by that same number. Owner:
`equal_minima_raw_term_ceiling` in the rehearsal module, beside
`EQUAL_MINIMA_PENALTY_SPREAD`.

| term | native | ceiling | band | placement |
|---|---|---|---|---|
| `raw.non_qs` | 4.480897876285335e-08 | 2.961e-04 | **not_worse 1e-4** | 2.96× under ceiling |
| `observable.non_qs_ratio` | (the same variable) | 2.961e-04 | **not_worse 1e-4** | 2.96× under ceiling |
| `weighted_total` | 4.482224653311689e-08 | 2.061e-13 | **not_worse 1e-6** | 4.85e6× *above* ceiling — inert |

The predecessor QS band of 1e-6 sat **296× under its own ceiling** — two decades
tighter than the placement every geometry term received — on the term carrying
**99.97 % of the NATIVE objective** (99.93 % at Q1's terminal, 99.97 % at Q2's —
the share is a property of the endpoint, not of the term; §12.9 states the
correction and this line carried the uncorrected form). Its refusal budget is
`(Φ* − native) − native·band`: a latch is refused iff its geometry penalties
plus its overshoot below the target fall under that number. At 1e-6 the nearer
banked latch (Q2) cleared refusal by **3.95×**, while the two banked arms differ
*from each other* in that very quantity by 7.98e-03, i.e. **9.2× that surviving
margin** — a gate whose margin is smaller than the observed run-to-run spread of
the quantity it gates is not calibrated. And the engine latches on the *first*
iterate at or under the target, so a tight landing (Q2 landed 3.97e-11 under) is
an ordinary outcome, not a tail event. At 1e-4 — the decade below the ceiling,
the placement `observable.major_radius` takes against its own 4.08× — Q2's
margin is 5.94× and Q1's is 46.7×. Nothing banked moves from pass to fail in
either direction; this is a placement, not a widening that admits the evidence.

`weighted_total` is the opposite case and keeps 1e-6 deliberately. Its ceiling
**is** the cross-executable gap between this repository's re-evaluated native
total and the frozen literal, so any band a latch could fail would have to be
tighter than a ULP class §5 exists to tolerate. At 1e-6 the term is **inert** —
it can neither refuse a latch nor false-reject one — and its substantive gate is
the latch number itself, `terminal_objective ≤ NATIVE_TARGET_OBJECTIVE`,
re-derived from the published bytes. `observable.total_length` is inert for the
same reason (3.51× above its own 2.853e-05 ceiling), as stated above. Two of the
ten pinned terms being one measurement (`raw.non_qs` and
`observable.non_qs_ratio` bind the same variable) is recorded here rather than
changed: dropping one would narrow the published set an artifact is judged on.

*(An earlier revision of this ruling stated the `weighted_total` cross-executable
offset as 2.1e-08. The measured value is 2.061e-13 — the conclusion is unchanged
and 6.7 decades stronger, not "nearly two".)*

**3. `derive_verdict` consults budget conformance.**
`CLAIM_DISCHARGED` requires `attempt_protocol.conformance == PREREGISTERED` as
well as a latch under the bar. Conformance is one label derived from the three
facts §3 and §12.2 freeze together — N = 3, the certified budget, the cold lane
— by a single owner (`attempt_protocol_conformance`); re-validation recomputes
it from the published fields before recomputing the verdict with it. Bounded
runs cap at `QUALITY_ONLY` (a latch under a lower cap is still a true
measurement) or `NO_LATCH_IN_PROTOCOL`. Without this, a `--iterations 400` run
minted the campaign's headline verdict and a zero exit code beside
`quality_claim: NOT_CLAIMED_AT_BOUNDED_BUDGET` and a per-term physics gate that
never ran — and the suite ratified that shape rather than refusing it.

**4. Sampler failures are absorbed as modeled degraded telemetry.**
`ProcessGpuMemoryMonitor.finish` re-raises whatever its polling thread stored,
and the supervisor called it with no handler anywhere between the raise and the
interpreter. One `nvidia-smi` query timing out among the ~10⁴ this protocol
performs, or one unparseable `[N/A]` row contributed by an unrelated process on
the box, discarded up to four completed GPU runs — possibly including the latch
— with nothing sealed and nothing published: the undefined outcome §4 exists to
eliminate, produced by a subsystem that appears nowhere in the §6 gate order.
Process GPU-memory sampling has no veto. Every failure of that observer — a
procfs binding refusal, an argv mismatch, a thread that would not start, a
query failure, an exhausted sample cap — is absorbed into the monitor module's
own unavailability union as `reason: "sampler-failed"` and published as
evidence; attempts and verdict publish regardless. The absorption is bounded to
that union and that observer, and no other exception path is widened.

**5. Re-validation runs before sealing and gates publication.**
§6 step 10 ran *after* `seal_and_sync` and `renameat2`, so any refusal it
raised left a sealed, immutable 0444 artifact carrying an intact `verdict`
field that the launcher's own validator had rejected, with the only record on a
stderr §11 calls volatile. The receipt is now re-derived from the bytes on disk
while the staging tree is still writable. A refusal writes
`root-validation-refusal.json` into that tree, leaves it unsealed and
unrenamed, publishes nothing at the final name, and propagates. Only the sealed
modes are checked after the rename, because they are the one property that does
not exist yet at the moment the receipt is judged. §6's gate order is
renumbered accordingly: re-validation is step 9, sealed publication step 10.

**6. The native endpoint reference is pinned and verified at load.**
It is the one input the chain reads from outside the repository: no
execution-source entry covers it, no source snapshot seals it, its enclosing
`.partial-` directory is owner-writable, and nothing compared its digest to
anything — so replacing it silently redefined what quality parity means, and
deleting it (ordinary housekeeping on the staging directory of a publication
that never completed) spent the root at the bootstrap gate. Both digests the
producing artifact records are now frozen constants, verified on every load,
along with the naming convention itself. They are **different quantities**:
`benchmarks/single_stage_native_equivalent_reference.py:418` writes
`arrays/{content_sha256}.npy`, where the content digest is taken over the
C-order float64 buffer, while the file digest covers the `.npy` container
around it. Reading the basename as the file digest is the trap this ruling
closes.

| quantity | value |
|---|---|
| `NATIVE_ENDPOINT_STATE_CONTENT_SHA256` (the basename) | `2639a955ede349edfdd7f5083776ae3ed0151627f3468d3430ca46029d63a912` |
| `NATIVE_ENDPOINT_STATE_FILE_SHA256` | `2ec9a9e38e9e4262c4b5dac49f418d1396572ddb22472f9ada979582fe6bf070` |

Both are published in the endpoint ledger and in the supervisor's preflight, so
a reader of the sealed bytes can re-identify the reference the gate used. Two of
the three facts `load_native_endpoint_state` checks are substantive; the third,
the basename convention, is an *identity* — `NATIVE_ENDPOINT_STATE_PATH` is
built with an f-string from the content digest, so its stem is that constant by
construction and cannot fail. It is kept because building the path from the
constant is what stops the two from drifting, which is worth more than a third
independent check would be; it is recorded here so it is not read as one.

#### Minors closed in the same remediation

Re-validation additionally re-derives each attempt's **outcome** from its own
evidence, return code and timeout flag; the **certified options delta** from
the published options against the frozen configuration; the **engine wall**
from `engine_compile + engine_solve` (an IEEE addition of the same two
published doubles, so agreement is exact); and the **claim's feasibility
tolerance** alongside the other two claim numbers. `latch_rate` is published as
*k*/N over the attempts authorized, the denominator §4 names, with the cold
lane deliberately outside it — a fourth full-budget draw that is not part of
the protocol and can only make the rate conservative. `gpu_runtime_identity`
records the interpreter and its prefix, since §3 pins the warm-cache behaviour
to a named venv. `_solve_payload` sanitizes every terminal scalar through
`json_scalar`, so a nonfinite value publishes null rather than killing a
completed solve at the canonical encoder. The execution-source gate refuses any
module of this distribution that resolved outside the checkout instead of being
structurally blind to it, and the launcher's own import closure is now bound in
a fresh interpreter by the suite. The certified-budget ledger branch is
executed against the real objective by the rehearsal suite (the native endpoint
against itself), so the root is no longer its first execution. The
external-resource preflight is described in §6.

#### Round-2 rulings (reviews at `5d4f6e855`, all four roles NO-GO)

The second review round returned NO-GO on five majors and no critical: every
round-1 blocker was confirmed closed by execution, and what remained was one
coherent theme plus one precondition. Four rulings follow, binding in the same
way; ruling 2 above is amended in place rather than restated.

**7. Re-validation re-derives the claim, and refuses a receipt that is not
whole.** Ruling 5 promoted `validate_root_artifact` from a diagnostic to *the*
gate on publication, and the function was not taught what that made it
responsible for: it remained a consistency check over the fields it happened to
find. Four sealed `CLAIM_DISCHARGED` artifacts were published through the real
`publish_root` and re-validated clean by two reviewers independently — one whose
latching attempt's ledger was **ungated**, carrying no per-term verdicts at all;
one carrying a self-consistent gate whose `passed` was **false**; and two whose
`attempt_protocol.maximum_iterations` said 700 while the attempts' own options
said 400, which is field for field the defect ruling 3 closed in the launcher
and left open in the validator. A fifth carried none of its custody blocks. The
gate now re-derives, from the sealed bytes:

* `endpoint_ledger_is_gated(iterations, latched)` for every attempt, against the
  published `gated_at_this_budget` — the one decision field that was read rather
  than derived, and the one that switches §1.1's physics gate on;
* `pinned_term_gate.passed`, *required* on a gated ledger. Equality of a
  faithfully recorded failure with its own recomputation is a consistency check,
  not a quality gate;
* every attempt's own `options.maximum_iterations` against the budget the
  conformance label is derived from, and `quality_claim` on both the attempt and
  the root from that same budget;
* §4's draw statistics — `attempts_run`, `latch_count`, `latch_rate`,
  `preregistered_attempts`, `stop_rule`, `certified_maximum_iterations` — which
  were pure read-backs beside a conformance label that was not;
* the wall of **every** attempt, not only the first latching one;
* the options **key set** against `CERTIFIED_ROUTE_OPTIONS.__dataclass_fields__`,
  because a delta derived over the keys an attempt published makes a truncated
  options block derive an empty delta and pass;
* the GPU the receipt names against the frozen `GPU_UUID` of §1.2, refused at
  launch as well so a mismatch costs nothing rather than costing the root;
* and **completeness**: the root document, `attempt_protocol`, every supervised
  attempt record and every child document have frozen key sets. *(Amended by
  ruling 11. The mechanism half of this bullet was accurate and the consequence
  half was false: the sets were exactly one level deep, so a receipt missing its
  preflight — or carrying an empty source snapshot, a null cache accounting and
  an emptied telemetry block — did pass for a whole one, executed through the
  real publication path by three of the four round-3 roles. Completeness is now
  recursive; see §12.9.)*

The suite is the other half of this ruling. `test_a_published_root_revalidates_from_its_sealed_bytes`
*ratified* the ungated shape — it published `CLAIM_DISCHARGED` at the certified
budget with `_synthetic_ledger(gated=False)` and asserted acceptance, which is
the pathology ruling 3 named as the reason the round-1 defect went unseen,
reproduced inside the fix for it. It now refuses that shape, and every fixture
publishes the complete document the supervisor publishes.

**8. Ruling 1 is carried to the cold lane.** *(SUPERSEDED IN ITS SECOND HALF by
ruling 13 of §12.9: the lane's OUTCOME no longer feeds conformance or the
verdict. Conformance's cold leg is the lane's AUTHORIZATION. What follows is the
defect ruling 8 correctly identified and the mechanism it chose; §12.9 keeps the
first and replaces the second.)* The cold lane is a fourth
full-budget draw at the certified budget, run first and outside the attempt
loop, and its outcome was never inspected. A lane that **latched and failed the
per-term quality gate** published `GATE_REFUSED:endpoint_ledger`, primed the
cache the timed attempts were then measured against, left
`conformance: PREREGISTERED` untouched — because that leg read the
`--no-cold-lane` *flag*, not the lane — and let the protocol mint the headline
verdict beside it: the strongest available counter-evidence to the quality
claim, sealed into the same tree with no effect on anything, and a certified
wall that is a warm number whose cold counterpart does not exist. Conformance's
cold-lane leg is now the lane's own outcome (`cold_lane_measured`). A lane that
**missed** still measured a cold compile and still primed the cache, so a miss
conforms — ruling 1's whole point is that a stochastic miss indicts nothing.
Every other outcome leaves the protocol in the state `--no-cold-lane` leaves it
in, and is labelled the same way: the verdict caps at `QUALITY_ONLY` instead of
the loop breaking, so every attempt still runs and every attempt's telemetry
still publishes.

**9. §11's temporary-storage rule is enforced in code, not in prose.** The rule
landed in §11 and was implemented by zero lines: `TMPDIR` appeared nowhere
in this repository except two sentences of this document, and the check those
sentences named — free space, "checked the same way" — is *provably blind* to
the binding limit. Measured on the certifying box while the condition was live:
`/tmp` reported **12.29 GiB available and 571 769 free inodes** while a one-byte
write returned `EDQUOT` and left a zero-length file behind. A per-uid tmpfs quota
is invisible to every capacity API and visible to exactly one thing, which is a
write. Meanwhile the unsafe configuration was the *default* — XLA spills from
C++, where the rule is `TMPDIR` or `/tmp` with no fallthrough, unlike Python's
`tempfile`, and that asymmetry is precisely why every Python path on the box kept
working while the spill died inside the bootstrap gate. `preflight_external_resources`
now **writes, fsyncs and removes a probe byte** in the resolved temporary
directory, in `--cache-dir` and in `output_root.parent`, refusing on any errno,
and refuses any of the three on a tmpfs filesystem type — because an empty tmpfs
passes a write probe and then fills during the run, and §11 enumerates the
failure *class*. The resolved directory, its filesystem type, its `st_dev` and
the (advisory) capacity number are published in the receipt, so a reader of the
sealed bytes can tell a root that ran under safe storage from one that did not.
The children are launched with `TMPDIR` **set** to the preflighted directory
rather than inheriting the operator's, so the rule is enforced against the
directory that is actually used. All of it runs before the staging tree exists
and before a second of compute, so a refusal costs an error message.

**10. Durability reaches the refusal record.** `seal_and_sync` is what fsyncs a
published tree, and it is exactly the step a refusal never reaches, so the
receipt, the manifest and `root-validation-refusal.json` lived only in the page
cache — the same "the only record is on a stderr §11 calls volatile" failure
ruling 5 exists to eliminate, displaced one step later. All three are fsynced
where they are written.

**Deferred, with reasons.** The order-dependence of `max()` over a sequence
containing NaN (numerics advisory A1) stays: recorded rows are provably finite
at this tree and the terminal point is separately gated by a nonfinite-refusing
`certify_agreement`, so a guard there would be an unreachable branch. The four
tautological read-backs (`timed_against_bar`, `sha_is_binding`, `bound`,
`budget_independent`; protocol-receipt finding 5) stay: each is a tamper check
whose substantive gate exists upstream. *(Round 2 correctly observed that the
"the artifact-tree digest covers the hand-edit case" half of that reason did not
hold while a wholly fabricated artifact could re-validate clean. **That half is
withdrawn rather than repaired**: rulings 7, 11, 12, 15 and 16 raised the cost of
a fabricated artifact — every block and every leaf shaped, the physics reference
frozen, the module-byte custody re-derived against the manifest — but a
hand-assembled `CLAIM_DISCHARGED` tree whose `source_snapshot` names a directory
it does not carry still re-validates, because nothing compares that block's
digest to the tree. The read-backs stay on the first half of the reason, which
never depended on it; the snapshot-digest tie is deferred with its reason in
§12.10.)* Sanitizing the endpoint ledger's own term rows would
require the pinned-term gate to model null terms, changing the verdict
contract; the rows are finite at any finite iterate, and a nonfinite one stays
contained as a published `PROTOCOL_FAILURE`. Constraining `--output-root` to
`~/simsopt-campaigns/` (reproducibility finding 7) stays operator-enforced per
§11. `src/simsopt/configs/NCSX.dat` (finding 11) stays covered by the identity
gate and the sealed source snapshot rather than by the module-hash gate. The
output namespace is still claimed only at the `renameat2`; §11 constrains the
root to one supervised session, so the misleading docstring is corrected rather
than the mechanism changed (protocol-receipt advisory 7).

**Deferred from round 2, with reasons.** *The unguarded supervisor window* stays
scoped exactly as ruling 4 scoped it — the one new indexing hazard inside it is
closed (`_attempt_outcome` requires `latched` to *be* a boolean rather than
indexing for it, so a canonical document of another shape is the
`PROTOCOL_FAILURE` the closed outcome space already has), and no other exception
path is widened. *No gate requires a timed attempt to have run warm*
(reproducibility NEW-2): the mechanism that would silently skip §3's warm lane
is a `--cache-dir` whose writes fail, and ruling 9 now probes that directory by
writing to it; the gate itself is refused because its direction is
conservative — a cold compile only inflates the wall against the bar, so no
false speed claim is reachable — and a gate that refuses an honest slow run is
the false-reject class this campaign has paid for three times. *No
`--preflight-only` CLI* (reproducibility NEW-3): under ruling 9 the whole
preflight runs before the staging tree exists and before any compute, so a
refusal costs an error message, and adding a zero-exit non-root lane to `main`
would break the invariant that its exit code is 0 **iff** `CLAIM_DISCHARGED`.
*The admissibility test is bound to a transcribed fixture* (numerics M2): the
fixture is verified correct at this tree and re-deriving it in the suite would
make the tests read a sealed campaign root, which is what pinning it as literals
deliberately avoids; it is a drift risk, not a present error. *`XLA_FLAGS` and
the rest of the inherited environment are neither pinned nor recorded*
(adversarial N6): recording the whole child environment would seal an operator's
shell into a published artifact, and enumerating a "performance-relevant" subset
is the twin-constant class §5 forbids. *`thread.join()` has no timeout*
(adversarial N11): the absorption catches exceptions, not hangs, and a join
timeout that abandons a live sampling thread trades a rare hang for a thread
still writing after the attempt closed. *A distribution module presented as a
namespace package escapes the redirected-module refusal* (reproducibility NEW-5,
adversarial N5b): a namespace package carries no code and the import dies at the
first symbol; the `.so`'s bytes are sealed into the source snapshot through the
`native_extension` role. The claim in the minors list above should be read as
"refuses any module of this distribution that resolved outside the checkout
**and has a file**". *`QUALITY_ONLY` at a bounded budget is minted with the
per-term ledger never gated* (protocol-receipt NEW-7): plan-sanctioned by ruling
3 — a latch under a lower cap is still a true measurement — and `quality_claim`
beside it is now re-derived and refused if restated, so the disambiguation is
machine-checked rather than conventional. *The sampler perturbs the wall it
certifies* (adversarial N12) and *the absorption discards samples already
collected* (N4): both are telemetry-direction findings, the first inflationary
and therefore incapable of minting a false pass.
 The second, N4, discards a
partial sample series when the observer fails and publishes the whole attempt as
`availability: unavailable, reason: "sampler-failed"` — so what is lost is a peak
number the receipt then declines to state, in the direction that under-claims,
and no gate reads it. Both are deferred on that ground and neither is reachable
from a claim.

---

### 12.9 Rulings on the third review round (two roles NO-GO)

The round-3 reviews at `cc1c710c0`
(`~/simsopt-campaigns/projected-route-root-reviews-round3-20260813T114359Z/`)
returned NO-GO from reproducibility-engineering and adversarial-redteam on three
majors, with numerics-physics and protocol-receipt at GO. All three majors are
in the layer that decides what a sealed receipt is allowed to say; none is in
the engine or the physics. Four rulings follow, binding in the same way.

**11. Completeness is recursive, or it is nothing.** Ruling 7 froze the key sets
of the root document, `attempt_protocol`, every supervised record and every
child document — and stopped there. Three roles independently reached the same
consequence from different entry points: below those four names nothing was
checked, and PRESENT-BUT-NULL was indistinguishable from absent for every block
no reader indexed into. Executed through the real `publish_root`, which
re-validates before it seals: a `CLAIM_DISCHARGED` root with **no preflight
block at all** published and re-validated clean, as did one whose supervisor
held nothing but `gpu_uuid`, one with an empty `source_snapshot`, one with a
null `compilation_cache`, one with an emptied `timing_seconds`, one whose
`claim` carried a key no producer emits, and ones with every attempt's
`gpu_memory`, `runtime_identity`, `execution_sources` and `environment` nulled.
The plan sentence those falsify is this document's own (ruling 7's completeness
bullet), and ruling 9's promise that "a reader of the sealed bytes can tell a
root that ran under safe storage from one that did not" rested on it.

The shape is now frozen as a TREE and walked recursively from one listing
(`_validate_document_shape`), so there is no second enumeration to drift: the
claim, the supervisor, its runtime identity, its preflight, each of the three
storage probes, the source snapshot and its worktree identity, the root's cache
accounting and telemetry, each supervised record's `gpu_memory`, and each child
document's environment, runtime identity, cache accounting, solve payload,
endpoint agreement, endpoint ledger and timing block. The preflight's contents
are additionally RE-DERIVED rather than read: ruling 6's two digests and the
reference path against the frozen constants, the pinned device against the
published inventory, and every probed directory against
`REFUSED_STORAGE_FILESYSTEM_TYPES` and its own write result — because a record
that is merely published is a record an artifact can invent. Where the reference
file is still on the box it is re-loaded, which re-verifies both digests at
validation instead of only at launch.

The suite is the other half again. The fixture every published-root test flowed
through published `supervisor: {"gpu_uuid": …, "gpu_zero_asserted": false,
"preflight": {}}`, so the suite asserted that the campaign's headline verdict
needs no native-reference digests, no device inventory and no storage evidence —
the exact pathology ruling 7 named, one level down, inside the fix for it. Every
fixture now publishes what its producer publishes, the nested shapes are bound
to the producers by execution rather than to a second listing, and the ten
escape shapes above are refused by name.

**12. An artifact may not supply the reference its physics gate is judged
against.** `gate_endpoint_ledger` takes BOTH sides from the document it is
handed. Ruling 7 made it *the* gate on publication and bound whether it ran and
whether it passed; it did not bind what it ran against. Executed: a ledger
publishing `terminal == native == 1.0` on all ten pinned terms recomputes every
verdict to `measured = 0.0, passed = true`, passes the gate, and seals as
`CLAIM_DISCHARGED` beside `solve.terminal_objective = 4.48e-8` — an internal
contradiction nothing noticed. This is round-2 N1's defect class displaced one
level: the quality half of the claim was a self-consistency check between two
numbers the receipt supplied about itself.

The native side is therefore a FROZEN CONSTANT of this campaign
(`NATIVE_ENDPOINT_PINNED_TERMS`), measured through this repository's objective
at the digest-pinned endpoint, and re-validation does three things with it: it
compares the receipt's published native side to the literals term by term; it
recomputes the gate FROM the literals and requires that recomputation to pass;
and it compares the ledger's own `native_state_sha256`,
`native_state_content_sha256` and reference filename to ruling 6's constants.
The existing self-consistency recomputation stays beside them — it is what
proves the run recorded its own arithmetic faithfully — but it is no longer the
gate.

The comparison class is per term, and this is load-bearing. The two lanes
evaluate one array through two independently compiled executables, so bitwise
equality is the demand that refused the predecessor route's fourth root after a
complete solve. Measured on this box, CPU against the 5090, at the same
endpoint: the three `absolute` terms sit at machine zero where the RELATIVE
deviation reaches 23x (`constraint.volume`, −2.332e-18 against −5.551e-17) while
the absolute deviation is 5.3e-17; the seven others agree to 1.95e-14 relative
or better (`raw.non_qs`, the dominant term). So absolute terms are judged
against their own gate band and the rest against
`NATIVE_ENDPOINT_REFERENCE_RELATIVE_TOLERANCE = 1e-11` — 512.98x above the worst
measured deviation and five decades under the tightest relative band any pinned
term carries, so it cannot false-reject an honest lane.

Two figures in the previous statement of this ruling were wrong and are
corrected here rather than restated. *"Nor let a forged reference widen a gate by
a fraction of its band"* is **false for the three `absolute` terms**, whose
native-side tolerance IS the gate band (1e-10): a forged `constraint.volume`
native of +1.0e-10 moves the admissible terminal window from [−1e-10, 1e-10] to
[0, 2e-10], a widening of a FULL band, and the same holds for
`constraint.boozer|inf` and `raw.residual` (measured: 1.000 band units for those
three, 1e-7 or 1e-5 for the other seven). The protection is sound for a stronger
reason than that sentence gave: the gate that DECIDES,
`gate_endpoint_ledger_against_frozen_native`, substitutes the frozen literals for
the published native side and never reads the reference the artifact publishes —
all three maximal forgeries are refused by it. And recomputing the gate against
the literals shifts each `measured` by at most 1.9477e-14 against a smallest band
of **1e-10**, not 1e-6: the worst shift as a fraction of a band is 2.3377e-05, on
`constraint.boozer|inf`. That is still four decades under the tightest banked
margin on that term (26.6x), so the conclusion — *passing* rather than *matching
bitwise* is what is demanded — holds by an enormous margin.

**13. THE COLD LANE IS DIAGNOSTICS, NEVER DISPOSITION.** Ruling 8 fed the lane's
own outcome into the conformance label through a predicate that cannot tell
counter-evidence from infrastructure: `cold_lane_measured` returns the same
`False` for a lane that failed the per-term quality gate and for one that died
on `GATE_REFUSED:bootstrap`, `:environment`, `:execution_sources`, `:solve`
(OOM) or `:attempt_publication`, that hit the 3600 s timeout, or whose stdout
did not parse. The lane is the FIRST GPU process of the session, against a cache
that must start empty, which is precisely where a first-compile timeout, an OOM
or a bootstrap fault lands — and §12.8 itself calls it "a fourth full-budget
draw that is not part of the protocol". Either way the previous revision
labelled a run that ran the pre-registered N, the certified budget and the lane
`BOUNDED_SMOKE`, capped its verdict at `QUALITY_ONLY` — which §4's table
disposes as **root spent** — and minted a `conformance: BOUNDED_SMOKE` beside
`quality_claim: CERTIFIED_BUDGET`, a pair `derive_verdict`'s own contract says
cannot occur and the sole stated reason for deferring protocol-receipt NEW-7.
§12.1 governs `NO_LATCH_IN_PROTOCOL` and nothing else, so the operator could not
tell from the artifact whether the arc stopped. The suite published exactly that
receipt and asserted acceptance.

The ruling: **attempt conformance and verdict derivation are governed solely by
the attempts.** `attempt_protocol_conformance` takes `cold_lane_authorized` —
whether the lane RAN, which is a pre-registration fact — and never the lane's
outcome, so `--no-cold-lane` still demotes and a lane that ran and then refused
does not. `derive_verdict` never sees the lane; its docstring says so. Any
cold-lane outcome other than a latch or a miss is published as a new top-level
`cold_lane_anomaly` block carrying the lane's outcome, the gate that refused it,
its exit status, whether it timed out, its supervised wall and its path in the
tree — re-derived at re-validation, so a receipt cannot hide an anomalous lane
behind a null. `PREREGISTERED` stands if the attempts ran pre-registered.

This is a deliberate reversal of ruling 8's second half, and the cost is stated:
a cold lane that latches and FAILS the per-term quality gate no longer caps
anything. That was ruling 8's real content and both GO roles endorsed its
direction. It is given up because the predicate cannot separate that case from
an OOM, because the price of the conflation is a spent one-shot root on an
infrastructure fault, and because the counter-evidence is not lost — it is
published in full, under a name, beside three timed attempts that each run the
same gate on their own endpoint. The lane's own ledger is still gated, still
published and still validated; what changes is that it decides nothing.

**14. The suite provisions its own storage.** Ruling 9's three tests are the
only machine evidence for the rule, and under this box's default environment
they were the three that failed: pytest derives `tmp_path` from `$TMPDIR`,
`/tmp` here is tmpfs, and the filesystem-type refusal fired before the assertion
each test makes — so an operator running the pre-root suite the obvious way saw
the newest ruling's tests in red pointing at the code under test rather than at
the environment, and the EDQUOT-class write probe the ruling is BUILT AROUND was
never exercised at all. §11's `--basetemp` rule is an operating instruction, not
a property of the repository. The suite now resolves its own directory off tmpfs
(`/var/tmp`, then `~/.cache`) through a fixture and the three tests use it, so
the whole file is green under the default environment and the write leg runs.

#### Minors closed in the same remediation

`filesystem_type` skipped mount-point TIES, so among several mounts sharing one
point the FIRST line of `/proc/self/mountinfo` won while the kernel resolves the
LAST: `mount -t tmpfs tmpfs /var/tmp/scratch` published as the ext4 underneath
it, passed the tmpfs refusal, passed the write probe an empty tmpfs always
passes, and then filled during the run. One character (`<` for `<=`), pinned by
a test against this namespace's one real stacked mount. — XLA's temporary
directory is a CANDIDATE LIST in TSL's order, `TEST_TMPDIR` → `TMPDIR` → `TMP` →
`/tmp`, and all three names are carried by both shipped binaries beside the
resolver's terminal `LOG(FATAL)`; the launcher read and overrode one of them, so
a shell holding `TEST_TMPDIR` sent every child's spill through a directory the
preflight never probed. All three are resolved, and all three are set in the
child environment. — A RELATIVE `TMPDIR` was accepted, probed against the
supervisor's working directory and forwarded to children launched with
`cwd=REPOSITORY`, so the certified directory and the used one were different and
the used one did not exist, which falls XLA's resolver through to `/tmp`: a
complete bypass reached by a value no check rejected. Non-absolute directories
are refused, and both the declared and the RESOLVED path are published, so a
symlinked temporary directory can be re-identified from the sealed bytes. — The
certified wall was derived from two halves neither of which was constrained in
sign or magnitude (`engine_compile = −1e6` beside `engine_solve = 1e6 + 100`
derives 100 s exactly) and was never compared to `supervised_seconds`, which the
receipt requires, publishes and no reader read. Both halves must be finite and
nonnegative and the derived wall must fit inside the wall the supervisor
observed. — `ATTEMPT_STOP_RULE` was re-derived as a STRING while the sequence it
describes was unconstrained, so a receipt could publish three `LATCHED` attempts
and `latch_rate: 3/3` on a loop that breaks after one, or four draws under three
authorized. The list is now bound to the rule: at most `authorized_attempts`
draws, every attempt before the last `COMPLETED_WITHOUT_LATCH`, consecutive
indices in their own directories, and no attempt directory in the tree the
receipt does not publish. — An attempt's declared EXECUTION CONTEXT was
published and never re-derived, so a `CLAIM_DISCHARGED` receipt could name the
CPU as the backend that produced the certified wall, state another timing
boundary, carry another route's schema or index, or declare an environment the
route forbids; all five are compared. — Re-validation is floating-point work and
silently depended on `JAX_ENABLE_X64`: with x64 disabled `jnp.asarray(...,
float64)` downcasts without raising, and a third party re-validating a GENUINE
sealed root was told "published terminal state differs from its hash", a message
indicting the artifact rather than the reader's shell. It asserts its own
precision first. — Ruling 10 reached the refusal record's contents but not its
tree's NAME: an unfsynced `mkdir` leaves the directory entry in the page cache,
so a power loss in the refusal window could take the freshly fsynced
`root-validation-refusal.json` with it. The staging tree's parent is fsynced at
creation. — `probe_writable_storage` reported "does not exist" for a path that
exists and is not a directory. — Two ordering statements around the preflight
("a refusal leaves the filesystem exactly as it found it") are approximate: the
cache directory `mkdir` and `bind_gpu_backend`'s CUDA context both precede it.
Corrected in place rather than papered over. — Three documentation corrections
from numerics-physics: `weighted_total`'s stated ceiling (2.061e-13) bounds the
ENGINE's Φ while the gate measures the STANDALONE re-evaluation, which §5
certifies equal only within `DIAG4_ENDPOINT_AGREEMENT_RELATIVE_TOLERANCE`, so
the honest ceiling is 1.0206e-11 — 49.52x (1.695 decades) above the stated
figure, with the shipped 1e-6 band still inert against either (9.80e4x);
`equal_minima_raw_term_ceiling`
carries its real precondition (corrected in §12.10 to "a quantity a latch BOUNDS
ABOVE by `NATIVE_TARGET_OBJECTIVE`", the narrower "raw objective summand" having
excluded the `weighted_total` this module derives a ceiling for)
and the note that out of domain it returns a well-formed meaningless negative;
and "99.93 % of the objective" is Q1's TERMINAL share, while the term carries
99.97 % of the native objective.

**Deferred from round 3, with reasons.** *`XLA_FLAGS` routes around ruling 9 and
moves the certified wall, unrecorded* (adversarial N19, protocol-receipt's one
MARGINAL deferral): the adjacent and larger half of this — the temporary-directory
name class — is closed above, which removes the mechanism by which an ordinary
shell reaches unprobed storage. What remains is an operator who deliberately
points `--xla_dump_to` at tmpfs or changes autotuning, and round 2's reason
stands: recording the whole child environment seals an operator's shell into a
published artifact and enumerating a "performance-relevant" subset is the
twin-constant class §5 forbids. It rests on §11 operator discipline and is
recorded as a known, accepted exposure of the wall-time claim. *The timed
child's DEVICE is never bound, only its backend string* (protocol-receipt NEW-5
residue): `gpu_runtime_identity` records `device_kind` and no UUID, and JAX
exposes no device UUID this chain can compare to §1.2's constant without a new
dependency; the supervisor pins the device at launch, refuses a mismatch before
compute and refuses it again at re-validation, and this is a single-GPU box.
Latent, and a new binding is a change to the launcher between the reviews and
the root. *The re-validated physics gate has no tie to the published TERMINAL
STATE* (protocol-receipt N3-8): closing it means re-running the objective inside
the validator, which is ~9 s of CPU and affordable — but the validator runs
inside the supervisor, and §6's whole timing argument rests on the supervisor
building no problem and compiling no kernel. Ruling 12 ties the gate's
REFERENCE side to the digest-pinned file without evaluating anything; tying its
TERMINAL side to the sealed array requires a lane that is not the supervisor,
and that is post-root work. *The admissibility fixture drifts by 1–4 ULP from
the live objective* (numerics M2, re-measured this round at worst 3.868e-16
relative, nine decades under the tightest band): unchanged, and re-verified
correct at this tree. *`max()` over a NaN-containing sequence is
order-dependent* (numerics A1): unchanged, and this round grounded the deferral
on the engine's own finiteness invariants rather than on inspection. *The
`--preflight-only` deferral's stated reason is contradicted by `--attempt-child`*
(protocol-receipt N3-5): the deferral stands on its other half — under ruling 9
the whole preflight runs before the staging tree exists and before any compute,
so a refusal already costs an error message — and the exit-code argument beside
it should not be relied on, since `main` already has a non-root lane.

---

### 12.10 Rulings on the fourth review round (three roles NO-GO)

The round-4 reviews at `167d71d87`
(`~/simsopt-campaigns/projected-route-root-reviews-round4-20260813T130000Z/`)
returned NO-GO from protocol-receipt, reproducibility-engineering and
adversarial-redteam, with numerics-physics at GO. Four majors, no critical, none
in the engine or the physics. Round 4 was also the first round in which the
adversarial forgery attempt SUCCEEDED after a remediation: six defective
receipts were published through the real `publish_root` and re-validated clean
from their sealed bytes, four of them as `CLAIM_DISCHARGED`.

Three roles reached ONE defect from three different entry points, and the
orchestrator's own note names the pattern behind it: *each round's completeness
fix stops exactly one level above the fact it protects.* Round 2 froze the
top-level keys; round 3 froze the blocks one level down; round 4 froze a tree
and left `execution_sources` — a REQUIRED key — with no shape at all, plus 164
`_ANY` leaves under the blocks it did freeze. A fix that enumerates shapes by
hand reproduces this once per round. Ruling 15 is therefore structural and is
the ruling the other three rest on.

**15. THE SHAPE IS THE KEY SET.** There were two hand-written enumerations —
five `*_REQUIRED_KEYS` frozensets ("the keys that must be present") and three
`*_NESTED_SHAPES` maps ("the blocks that have shapes"), keyed by a subset of
them — and the drift between them was live: an AST census showed
`execution_sources` was the only key of `ATTEMPT_EVIDENCE_REQUIRED_KEYS` that no
code in the module ever read, and the suite could not see it because a test that
enumerates the shapes which EXIST is structurally incapable of detecting a shape
that is ABSENT. That is how the previous revision shipped 75 green tests over
the hole, and how the plan came to assert that "the ten escape shapes above are
refused by name" while the ninth was refused in no form at all.

The ruling: **there is one structure, and the key sets are derived from it.**
Every document the receipt is built from is a node of one shape tree
(`RECEIPT_SHAPES`); each `*_REQUIRED_KEYS` frozenset is now literally
`frozenset(*_SHAPE)`, so a required key with no shape is unrepresentable rather
than merely absent. `_ANY` is gone: every leaf declares what its producer writes
there (`_STRING`, `_NUMBER`, `_BOOL`, `_LIST`, `_MAPPING`, their nullable forms,
and `_NULL`), `bool` is excluded from the number leaves explicitly because it is
a subclass of `int`, and a leaf refusal names the defect instead of surfacing as
whatever `TypeError` the first reader to touch it raised. Where a mapping or a
list is still admitted without an inner shape, the place and the REASON are
declared in `UNSHAPED_LEAVES`, and a `_dispatched(...)` node names the function
that validates it instead.

The suite's half is a COVERAGE meta-test, not another enumeration
(`test_the_shape_tree_covers_its_own_required_key_sets`): it asserts that every
required key set is exactly its shape's key set, that every node of every tree is
one of the four admitted kinds, and that a walk of the trees finds exactly the
unshaped leaves the module declares — so a block added without a shape, or an
unshaped leaf added without a reason, fails the suite rather than the next
review. Every producer a CPU process can run is bound to its shape BY EXECUTION,
including the four the previous revision called "the three that are only
reachable with a device" (a miscount, and wrong for at least three of them). The
one genuinely device-gated producer is the preflight, which queries the pinned
GPU's inventory; it stays bound through the bounded GPU smoke.

**16. A `CLAIM_DISCHARGED` RECEIPT SAYS WHICH BYTES EXECUTED.**
`execution_sources` is the published residue of the module-hash gate — the
manifest binding, the bound modules, the unmanifested-repository set and the
interpreter-installation set — and it is the only place in a sealed receipt
where a reader can see the child's source binding. It answers the question a
sealed source snapshot cannot: did the bytes the snapshot contains actually RUN?
The predecessor route lost a root to a scikit-build-core editable finder that
outranked `PYTHONPATH`, which is the class this block exists to catch. At the
previous revision it had no shape and zero readers, so `null`, `{}`, `"a
string"` and `{"bound_modules": []}` all published through the real
`publish_root` and re-validated clean, on the timed attempt and on the cold
lane — and the suite fixture published `{"bound_modules": []}`, a receipt
asserting that ZERO source modules executed, against a producer that publishes
four keys and 297 bound modules.

It now carries a full frozen shape AND is RE-DERIVED: the published manifest
evidence must equal this repository's own manifest recomputed from its bytes
(path, schema, both digests, entry count), every bound module must hash and
size to the entry the manifest holds for it, the bound set may not be empty, and
it must contain the three modules the certified chain cannot run without — the
launcher, the rehearsal module every shared primitive comes from, and the engine
under certification — named by the files this process imported rather than by a
second spelling of their paths. The same treatment reaches the two other blocks
that had no shape and two read booleans between them: `problem_identity` is
re-derived in full from its published measurements through
`problem_identity_evidence`, the producer's own owner (§2's identity binding, now
checked rather than asserted), and `lowering_pre_gate` has its budgets, its
kernel list and its IR total re-derived. A fixture that publishes less than its
producer does is a fixture asserting a shape the protocol cannot produce; the
suite's now derive from the manifest and from the producers.

**17. THE COLD LANE DOES NOT GATE PUBLICATION EITHER — ruling 13, completed.**
Ruling 13 took the lane out of `attempt_protocol_conformance` and out of
`derive_verdict` and said "what changes is that it decides nothing". Executed, it
still decided the largest thing there is: `validate_root_artifact` ran the FULL
discharging-attempt validation on the lane, inside the publication gate, so a
cold lane that latched and missed one pinned band — or that merely MISSED the
latch with one infeasible recorded iterate — raised, `publish_root` wrote its
refusal record and re-raised, and `seal_and_sync` was never reached. No
artifact, no verdict, and the cold lane plus all three timed attempts already
spent. §4's table disposes `QUALITY_ONLY` as *root spent*; this outcome is
strictly worse, because there is nothing for an operator to read. Two roles
found it independently (adversarial-redteam F1 as a live root-burn path,
numerics-physics N12 as the same mechanism), and it is reachable on the
sanctioned launcher path with no flags: the lane is a fourth full-budget draw at
the same budget, run first, against an empty cache, and the campaign's measured
miss rate is one in five.

The ruling: **inside `publish_root` the cold lane gets anomaly-recording
validation only.** It is validated for SHAPE and for HONESTY — the record is
complete and typed, its outcome is re-derived from its own evidence and exit
status, its execution context, custody blocks, budget, options, engine-wall
algebra, ledger scope, ledger arithmetic and terminal-state hash are all
re-derived, it may not be timed against the bar and it may not have run warm —
and it is never validated for the CLAIM. The five comparisons that decide
whether a draw discharges §1's claim (`certify_native_reference`, the per-term
gate's verdict, that gate recomputed from the frozen literals,
`certify_agreement`, and the feasibility bound) run on the pre-registered
attempts alone. An honest cold miss or band-fail publishes: a lane that refused
a gate publishes as `cold_lane_anomaly` with the refusing gate named, and a lane
that latched past a band publishes its whole ledger for a reader to recompute,
beside three timed attempts each of which runs every gate on its own endpoint.
The split has one owner each — `_validate_attempt_record` for the record's own
facts, `_validate_attempt` for the claim, `_validate_cold_lane` for the lane.

**18. A PRE-REGISTRATION FACT IS BOUND TO THE TREE.** After ruling 13,
`cold_lane_authorized` is the lane's ONLY channel to `PREREGISTERED` and
therefore to `CLAIM_DISCHARGED`, and ruling 13 did not bind it: the sole check
was that a record existed somewhere in the receipt, `COLD_LANE_DIRECTORY`
appeared zero times in the validator, and the "every draw on disk is a draw in
the receipt" sweep iterated only `attempts/`. A lane record whose
`artifact_relative_path` said `attempts/attempt-1` was validated against attempt
1's own sealed array and passed, so a root with **no cold-lane directory in the
tree at all** minted the headline verdict. The lane's path must now be
`cold-lane` and its index 0, and the `cold-lane` directory must exist in the
artifact tree if and only if the receipt claims the lane ran — the supervisor
creates it before it launches the lane, so it exists for every outcome the lane
has, including a refusal.

**The six forged receipts are re-published by name.** Round 4's forgeries are
kept as named tests rather than folded into the tables, because a forgery that
once worked is the only evidence that a fix works:
`test_round4_forgery_i_hollowed_custody_blocks_are_refused`,
`test_round4_forgery_i2_a_null_execution_sources_block_is_refused`,
`test_round4_forgery_h_a_nulled_leaf_is_refused`,
`test_round4_forgery_j_a_wrongly_typed_leaf_is_refused`,
`test_round4_forgery_k_a_cold_lane_aliased_onto_an_attempt_is_refused` and
`test_round4_forgery_m_a_restated_relative_difference_is_refused`. Ruling 17's
own direction is pinned the other way by
`test_an_unlucky_cold_lane_draw_cannot_refuse_the_whole_publication`, which
publishes a root whose cold lane failed the per-term gate and another whose cold
lane recorded an infeasible iterate, and requires both to seal
`CLAIM_DISCHARGED` — while a cold lane that lies about its backend or hollows its
custody block is still refused.

#### Minors closed in the same remediation

The ledger's `relative_difference` column was checked for its key set and never
for its arithmetic, so every entry could read `0.0` beside sides that disagree;
it is re-derived from the two sides through `endpoint_relative_differences`, one
owner shared by the producer, the validator and the suite (adversarial F5). —
Every `_ANY` leaf admitted present-but-null and any scalar type: 164 of them, and
the reachable ones included the whole cache accounting ruling 9's readability
promise leans on, every storage probe's identity, the source snapshot's digest,
`chain_wall`, and every attempt's telemetry and solve payload. Closed by ruling
15's typed leaves (protocol-receipt N4-3, reproducibility R4-2, adversarial F3).
— `preflight.visible_gpu_uuids: null` refused with `TypeError: argument of type
'NoneType' is not iterable` rather than a named refusal; every leaf refusal now
names the defect (protocol-receipt N4-6). — The preflight published the
temporary directory TWICE, once as the resolved XLA spill path and once as the
probe that cleared it, and never compared them, so a receipt could name one
directory beside a probe of another; they are one fact and are compared
(protocol-receipt N4-4, first half). — The ruling-14 suite fixture applied
neither of ruling 9's own rules to itself: it accepted a RELATIVE `TMPDIR`, which
resolves against pytest's rootdir — this frozen repository — and it type-checked
its candidates without write-probing them, so a `/var/tmp` that is off tmpfs but
read-only or quota-exhausted is selected and fails later at `mkdir` as an
unhandled `OSError` mid-suite. The fixture now selects through the launcher's own
`probe_writable_storage`, which refuses a non-absolute path, refuses a RAM
filesystem, and then writes a byte (reproducibility R4-3, R4-4). — Four frozen
shapes were bound in the suite to fixtures under a comment that miscounted them
("the three that are only reachable with a device" naming four) and misstated
why: `gpu_runtime_identity`, `probe_writable_storage`,
`configure_persistent_compilation_cache` and `_gpu_memory_payload` all run on
CPU, and all four are now bound to their producers by execution (adversarial F6,
reproducibility R4-6). — Three numerics documentation defects, each corrected in
place with the measured figure: ruling 12's "a fraction of its band" is false for
the three absolute terms (N8), its "~2e-14 against a smallest band of 1e-6"
pairs the right magnitude with the wrong band — the smallest band is 1e-10 and
the worst shift is 2.3377e-05 of a band (N9), and `weighted_total`'s corrected
ceiling is 1.0206e-11, which is 49.52x rather than "two decades" above the
derived figure and leaves the shipped band inert by 9.80e4x rather than 4.8e4x
(N11; the module carried two different multipliers for one ratio). —
`equal_minima_raw_term_ceiling`'s stated precondition excluded the
`weighted_total` the module derives a ceiling for two dozen lines below and the
suite calls it on; restated as "a quantity a latch BOUNDS ABOVE by
`NATIVE_TARGET_OBJECTIVE`" (N10). — §12.8's deferral sentence "Ruling 7 closes
that" was still refuted by execution; the half it rested on is WITHDRAWN in
place, with what actually remains open stated beside it (protocol-receipt N4-2).
— The plan's ruling-2 body still carried the unqualified "99.93 % of the
objective" that §12.9 corrects 490 lines later, and the module's band table
printed `observable.iota` without its sign (numerics N15).

**Deferred from round 4, with reasons.** *The preflight's storage record is
re-derived only against itself* (protocol-receipt N4-4, second half): the
temporary directory is now tied to its probe, but `filesystem_type` and
`one_byte_write` remain the artifact's own strings, because re-deriving them
means reading `/proc/self/mountinfo` on the READER's box — which makes a sealed
root un-re-validatable anywhere except the machine that produced it, and
reproducibility-engineering's third-party re-validation is a property this
protocol has and intends to keep. The half that decides is already frozen: the
launcher computes the real type through the corrected mount walk before any
compute, and a run on tmpfs never reaches the staging tree. — *`source_snapshot`
may name a directory the tree does not carry* (protocol-receipt N4-2's X11): the
snapshot's own manifest digest is not compared to the sealed subtree, so a
hand-assembled root with no `source-snapshot` directory re-validates. Tying it
means re-deriving `publish_immutable_snapshot`'s manifest over the sealed tree
inside the validator, which is new code on the publication path between the
reviews and the root; the artifact manifest already covers every byte in the
tree against tampering AFTER publication, and this is a forgery-only exposure of
a receipt nobody produced. Filed for the successor revision. — *Two attempt
records with byte-identical evidence in two directories publish clean*
(protocol-receipt N4-5's AL5b): with `artifact_relative_path` now pinned to
`attempts/attempt-{i}` and the lane to `cold-lane`, this reads as "two draws that
produced the same terminal state", which is not per se false. — *The supervisor's
own `runtime_identity.backend` may say `cpu`* (N4-5's H06): the supervisor is not
where the certified wall comes from and the attempt's backend is bound. — *The
suite fixture does not record which storage it resolved, and leaves a shared base
behind* (reproducibility R4-5): pytest output is not an artifact of this
protocol, and the per-session directory is removed. — *`certify_native_reference`
is a cross-backend equality gate one decade tighter than the campaign's own
cross-backend precedent, and it runs after the spend* (numerics N7), and *it runs
before the `gated` branch, so a native-side deviation would destroy even the
`NO_LATCH_IN_PROTOCOL` receipt §12.1 needs* (N13): both are accepted with the
margin measured — 513x against the cited measurement, ~340x against an
independent numpy-vs-XLA reimplementation, ~85x against the worst case implied by
the session-to-session spread in sixteen banked `bootstrap_equivalence` records —
and numerics-physics explicitly recommends against retuning the constant between
the reviews and the one-shot root. Its own role rated the placement defensible;
retuning is the larger risk. — *Twin native-side constants: the rehearsal suite's
`BANKED_ENDPOINT_LEDGER` disagrees with `NATIVE_ENDPOINT_PINNED_TERMS` at 1-2 ULP
on three terms* (numerics N14): the authoritative object is the bitwise-bound
one, the fixture is judged at `rel=1e-9`, and deriving the fixture from the
constant edits the admissibility proof's own input between the reviews and the
root. — The round-3 deferrals (`XLA_FLAGS`, the timed child's device UUID, the
physics gate's terminal-side tie, numerics M2 and A1, the `--preflight-only`
reason) are unchanged and their reasons are unchanged.

### 12.11 Rulings on the fifth review round (two roles NO-GO)

Round 5 reviewed `2f4244dca` (reviews at
`~/simsopt-campaigns/projected-route-root-reviews-round5-20260813T140000Z/`).
numerics-physics and protocol-receipt returned GO; reproducibility-engineering
and adversarial-redteam returned NO-GO on **two CRITICALs and one MAJOR**. It is
also the round in which the previous remediation genuinely worked: all four
round-4 blocking findings are closed BY EXECUTION — the coverage meta-test was
proven by mutation to detect an ABSENT shape, `execution_sources` refuses eleven
escape shapes by name on both lanes, and six honest cold-lane outcomes that
round 4 refused now publish. The findings below are what the review found
underneath that.

The pattern the round-4 ledger named reproduced for the fifth time: *each
round's completeness fix stops exactly one level above the fact it protects.*
Round 2 froze the top-level keys; round 3 froze one level down; round 4 froze a
tree and left `execution_sources` with no shape; round 5 froze the SHAPE of
`options` and re-derived its DELTA, and left the VALUE unconstrained. **Freezing
a shape is not binding a value**, and this remediation is stated in those terms.

**19. THE RECEIPT IS BOUND TO THE CERTIFIED ROUTE'S VALUES.** §1's claim is a
claim about ONE route, and the campaign's whole substitution argument is that a
bounded CPU rehearsal and the certified GPU run are the same configuration with
`maximum_iterations` replaced (plan:339-346, `rehearse…:410-417`,
`run_…:2576-2578`). The validator checked the options KEY SET against
`CERTIFIED_ROUTE_OPTIONS.__dataclass_fields__` and re-derived the published
DELTA from the published options — and then constrained that delta to nothing at
all. Twenty-one of the twenty-four fields were free. Three roles reached the
same defect from three entry points (forgery, substitution-soundness audit,
plan-claim audit) and published, through the real `publish_root`, roots that
seal `CLAIM_DISCHARGED` and re-validate clean while declaring
`lagrangian_newton: false` — the reduced-Lagrangian Newton–CG arm that IS the
route under certification — `gauss_newton: true`, `frozen_projector_line_search:
false`, `backtracking_factor: 1.0` (a line search that never contracts) and
`feasibility_tolerance: 1e-3` beside `claim.feasibility_tolerance: 1e-10`, each
with a self-consistent delta. `UNSHAPED_LEAVES:624` declared "every value
re-derived"; no value was re-derived.

The ruling: `_validate_certified_route_options` compares **every published
option value, field by field, to `CERTIFIED_ROUTE_OPTIONS`'s own literals**, and
the only field a budget may replace is the budget. At the certified budget the
permitted delta is therefore EMPTY; at a bounded one it is exactly
`{"maximum_iterations": n}`. This is the enforcement the CPU rehearsal's suite
has had since round 1 (`test_rehearse…:272`), ported to the publication gate.

**20. A SUMMARY MAY NOT CONTRADICT THE ROWS IT SUMMARISES.** One of the five
claim-bearing comparisons — the worst iterate against the feasibility bound —
read the scalar `solve.maximum_feasibility_inf`, which in the producer is `max`
over the very iterates the same receipt publishes as `solve.rows`. A receipt
carrying recorded iterates at 0.005 and 0.027 — nine decades outside the
tolerance §1 states the claim at — sealed `CLAIM_DISCHARGED` beside a summary of
1e-14, so a reader doing the arithmetic the receipt invites gets a different
answer from the validator that accepted it. The same receipt published
`iterations_run: 700` with zero rows, `latched: true` beside `status_name:
LINE_SEARCH_COLLAPSE`, and `stored_pairs: -5`; `status`, `status_name`,
`iterations_run`, `monotone_descent`, `stored_pairs` and
`terminal_feasibility_inf` had zero readers.

The ruling: `_validate_solve_telemetry` **re-derives the summary from the rows**
— `iterations_run == len(rows)`, the worst feasibility from the rows'
`feasibility_inf`, `monotone_descent` from the rows' objectives, `status_name`
from the engine's own enumeration and `latched` from that name, plus counts that
are counts. Every identity was verified EXACTLY against both lanes of a real
5090 receipt and a live CPU solve before it was made a gate, so it cannot burn
an honest root. Where a nonfinite value makes an identity un-re-derivable from
the published bytes (`json_scalar` writes null and the raw value is gone), the
check admits every reading the producer could have written rather than guessing
one.

**21. A CLAIMED COLD DRAW IS A DRAW.** Ruling 18 bound the cold-lane
DIRECTORY's existence, not a cold draw, and `cold_lane_authorized` remains the
lane's only channel to `PREREGISTERED` and therefore to the headline verdict. A
forger paid one `mkdir`: an EMPTY `cold-lane/` beside a `TIMEOUT` record, and a
`cold-lane/` holding a byte-copy of `attempts/attempt-1` beside a copy of
attempt 1's own record, both minted `CLAIM_DISCHARGED`.

What separates a draw from a retelling is **not its endpoint**: two honest draws
of the same problem at the same budget produced BITWISE IDENTICAL worst iterates
on the 5090 (measured, both lanes of the round-4 smoke), so demanding a
different terminal state would burn an honest root. It is the INVOCATION, the
CACHE and the WALL. `_validate_cold_lane_draw` requires the lane's invocation
digest to be no timed attempt's — the supervisor launches every child at its own
index into its own directory — and requires the lane not to have run warm, now
that `warm` is itself re-derived from the entry count the child sampled before
it traced anything. `_validate_supervised_launch` requires every draw's device
telemetry to be an observation OF THAT CHILD (`child_argv_sha256 ==
argv_sha256`), on the device §1.2 names, and requires a record claiming a
timeout to have waited the timeout it publishes — `communicate(timeout=…)`
cannot raise before its timeout elapses. **Ruling 17 is preserved in both
directions**, and the suite pins it: a lane that really timed out publishes with
its anomaly recorded beside a discharged root.

**22. EVERY NAMED VALIDATOR OWNS A FORGERY THAT KILLS IT.** protocol-receipt
deleted `_validate_lowering_pre_gate` and `_validate_problem_identity` outright
and the launcher suite stayed 83/83 green; line-trace coverage showed **37 of
the 82 refusal sites in the re-validation path are never reached by any test**,
and the six named round-4 forgery tests all refuse at the outer shape frozenset
before reaching one of ruling 16's own gates. Ruling 16's re-derivations were
therefore protected by nothing: a gate no test can kill is a gate the next
revision can delete.

The ruling: `_VALIDATOR_KILLS` pairs **every** `_validate_*` function in the
launcher with one forgery that ONLY it refuses, and each case is published twice
through the real `publish_root` — once whole, where the refusal must name that
validator's own words, and once with the validator replaced by a no-op, where
the same receipt must PUBLISH. The structural half is
`test_every_named_validator_is_covered_by_the_mutation_kill_set`, which asserts
the table is exactly the module's `_validate_*` surface, so a validator cannot
be added without a forgery that proves it necessary. The forgery tests are
narrowed to the gate they claim: scenario I hollows each custody block
INDIVIDUALLY and asserts each block's own refusal, and scenario I2 gains the two
gates of `_validate_execution_sources` no test reached — the manifest comparison
and the empty-bound-set refusal.

**23. RULING 16'S KERNEL LIST, RE-DERIVED.** All four roles refuted the same
clause: `lowering_pre_gate` "has its budgets, its kernel list and its IR total
re-derived" was false for the kernel list, which was checked for non-emptiness
and for its own internal sum. numerics-physics executed the real CPU lowering (6
kernels, 65 204 569 IR bytes) and showed the shipped fixture's two invented
kernels and 12 288 bytes are ACCEPTED. WHICH kernels a configuration lowers is a
function of that configuration — `evaluate_carried` exists only above a
projector refresh period of one, `frozen_retract` only under the frozen-projector
line search, `lagrangian_newton_direction` only under the reduced-Lagrangian arm
— so the list IS a statement about the route and is now re-derived against
`CERTIFIED_LOWERED_KERNEL_NAMES`, the campaign's own, bound to the real producer
BY EXECUTION in the rehearsal suite against a record a real rehearsal published.
Their SIZES are deliberately not frozen: the same six kernels lowered 65 204 569
bytes on CPU here, 65 207 733 in another CPU process and 65 200 869 on the 5090,
so a byte count would be a false reject waiting to happen — which is why §6.1's
substantive gate (identical IR at both budgets) runs in the child, where one
process lowers both sides.

#### Minors closed in the same remediation

`execution_sources.unmanifested_repository_modules` — the half of the custody
block that catches the scikit-build-core editable-finder class it exists for —
was shape-checked and read by nothing; a certified launch imports nothing from
the tree but the three manifested roots (measured `[]` on both lanes of the
bounded 5090 smoke and on CPU), so a non-empty list is refused by name
(reproducibility E5-4, adversarial A5-4). — The timing chain was not a chain:
`engine_wall <= supervised_seconds` was enforced while `attempt_wall` sat
outside both and the three phase durations were unconstrained, so a receipt
could publish `attempt_wall: 1e-9` around a 187 s engine beside three `-50.0`
phases; the three measurements nest and each phase is a duration (adversarial
A5-8, first half). — Every attempt's `gpu_memory.device_uuid` was unread, so an
attempt could name another device beside a supervisor pinned to §1.2's; it is
compared to `GPU_UUID`, which is also the cheap half of the timed child's device
deferral (adversarial A5-9, protocol-receipt P5-6). — `warm` was a published
boolean beside the cache state it is a function of; it is re-derived from
`at_entry.entry_count` on both lanes, which is what gives the cold lane's own
"may not have run warm" refusal something to stand on (adversarial A5-7, first
half). — The one nullable numeric leaf `_validate_attempt_record` reads
unguarded, `terminal_objective`, reached `>` as an unnamed `TypeError` on a
`LATCHED` record; it is refused by name (reproducibility E5-3). — The preflight's
temporary-directory refusal reported the declared/probed pair of the OTHER field
and read as self-contradictory to a third party; it names the pair it checked
(protocol-receipt P5-10). — `LOWERED_KERNEL_SHAPE` is bound to
`lowering_payload` by execution in the launcher suite and the whole
`LOWERING_PRE_GATE_SHAPE` record to `measure_lowering_pre_gate` in the rehearsal
suite, where a real rehearsal already pays for the bootstrap and the two
lowerings (reproducibility E5-2, protocol-receipt P5-3, numerics N5-5). — Three
`UNSHAPED_LEAVES` reasons were false or stale and are corrected in place:
`options` ("every value re-derived", refuted by ruling 19 and now true of ruling
19's own comparison), `endpoint ledger.native` and its gated twin (true for a
pre-registered attempt, false for the cold lane, which shares the node and by
ruling 17's design never reaches `certify_native_reference`), and `solve.rows`
("§6 gates none of it", while §6's feasibility gate is a projection of exactly
those rows) (protocol-receipt P5-4, reproducibility E5-7, numerics N5-7). —
Three numerics documentation defects: ruling 12's refuted "a fraction of its
band" sentence still stood verbatim in the module comment attached to the
constant it describes and is corrected there (N5-1); the N11 closure attached
the right number (1.0206e-11) to a formula that is false by seven decades
(`rel * NATIVE_TARGET_OBJECTIVE = 4.482e-19`) and contradicted itself two lines
later — the honest ceiling is the derived ceiling PLUS the cross-executable
RELATIVE tolerance, `2.0610196e-13 + 1e-11` (N5-3); and `_validate_attempt`'s
docstring said "four comparisons" over a list of five.

**Deferred, with reasons.** *§12.10's "four decades under the tightest banked
margin on that term (26.6x)" is 3.206 decades, measured* (numerics N5-2): the
figure is wrong and the conclusion it supports — that recomputing the gate
against the frozen literals cannot flip a verdict — holds by 3.2 decades on the
binding arm. Correcting a decimal in a closed ruling's prose is the one edit
class §9 makes expensive between the reviews and the root, and the measurement
is recorded here instead: **3.206 decades on Q2, 2.371 on Q1**. — *`source_snapshot`
is re-derived by nothing and not bound to the artifact tree* (adversarial A5-6,
protocol-receipt N4-2's X11, unchanged from §12.10): tying it means re-deriving
`publish_immutable_snapshot`'s manifest over the sealed tree inside the
validator, which is new code on the publication path between the reviews and the
root; the artifact manifest already covers every byte after publication and this
is a forgery-only exposure. — *No floor under the certified wall* (adversarial
A5-8, second half): a latch whose whole certified wall is `1e-300 s` seals under
a 287.30 s bar. A floor is a new frozen constant about GPU compile time, which
is exactly the class numerics-physics recommends against introducing between the
reviews and the root; the wall's halves, their nesting and the supervised wall
are all bound, so the residue is a receipt whose own three measurements agree on
an impossible number. — *The sealed terminal state is not bound to the problem's
dimension* (adversarial A5-10): the array's length is the joint DOF count, which
the receipt does not otherwise carry, so binding it means a new frozen constant
about the case; the digest, the array and the ledger's terminal side are each
bound to their own owners. — *`endpoint_agreement.terminal_feasibility_inf` and
`feasibility_absolute_tolerance` are unread* (A5-10, second half): the claim's
feasibility gate reads the solve summary, which ruling 20 now binds to the
iterates; the agreement block's copy is a second telling with no reader, and
giving it one means deciding which of the two is authoritative — a contract
question, not a gate. — *`interpreter_installation_modules.roots` is unread and
would close the `.venv-qn-gpu` carry-over* (reproducibility E5-4, second half):
`roots` names hidden top-level directories INSIDE the checkout, so a venv
outside the tree publishes `{count: 0, roots: []}` and is equally honest; the
field cannot discharge an interpreter pin, and pinning `.venv-qn-gpu` would
refuse an equally honest launch. — *The launcher cannot be imported at all when
the engine resolves outside the tree, and the failure is a bare `ValueError` at
module scope* (reproducibility E5-5): fail-closed, before argv is parsed, at no
root cost; converting it means module-scope error handling on the import path
the root itself runs. — *`bound_modules` is a function of the import graph, so
two honest attempts may publish different custody lists* (E5-6): both are
honest, the three chain modules are pinned by name, and a superset rule is what
keeps the block from being a false-reject surface. — *`UNMANIFESTED_MODULE_SHAPE`
is bound to no producer by execution* (protocol-receipt P5-3, third shape): the
producer emits an element only when a repository module resolves outside the
manifest's roots, which no honest launch does — and ruling 19's minor now
refuses any receipt that carries one, so the shape is exercised in the refusal
path rather than in an accepted receipt. — *Ruling 17's "never validated for the
CLAIM" is inexact: a sixth comparison against `NATIVE_TARGET_OBJECTIVE` runs on
the lane* (protocol-receipt P5-5): proven unreachable by an honest draw — the
engine breaks at the first iterate at or below the target and returns that
point's objective — and it is now guarded against the null reading (E5-3), so
the sentence's residue is one comparison an honest lane cannot fail. — *The
attempt's compilation-cache directory is unbound* (P5-7), *`chain_wall` and the
ledger's superset rule* (P5-8). — *Five integer quantities are read through
`int(...)`, which truncates a non-integral value the `_NUMBER` leaf admits*
(numerics N5-6): no producer writes a non-integral iteration count or IR size
(verified by execution), a receipt claiming 700.9 certified iterations describes
nothing physical, and an integrality leaf kind is new shape-tree vocabulary
between the reviews and the root. — The round-3 and round-4 deferrals
(`XLA_FLAGS`, the preflight's fstype self-reference, the physics gate's
terminal-side tie, AL5b/H06, the fixture's storage record, numerics N7, N13,
N14, M2 and A1, the `--preflight-only` reason) are unchanged and their reasons
are unchanged.

### 12.12 Rulings on the sixth review round (three roles NO-GO)

Round 6 reviewed `a0400d6eb` (reviews at
`~/simsopt-campaigns/projected-route-root-reviews-round6-20260813T154303Z/`).
numerics-physics returned GO; protocol-receipt, reproducibility-engineering and
adversarial-redteam returned NO-GO on **one CRITICAL and three MAJORs**. It is
also the round the round-5 remediation is vindicated in: ruling 19 is closed by
all three roles that raised it — twelve route substitutions refused by name on
both lanes, the options proven to be the same object from the same import in
both lanes, `json_scalar` proven lossless in value and type on all 24 fields
with +1 ULP refused twelve for twelve — ruling 20's feasibility half holds
13/13 exactly against a real solve, ruling 22 kills 16 validators of 16, and
ruling 23's kernel list is this repository's real lowering.

The pattern the round-4 ledger named reproduced for the **sixth** time, and this
round it landed INSIDE the commit that retired it. The generalisation the
orchestrator states, and which this remediation is written in: **a re-derivation
is only as strong as the ANCHOR it derives against, and an anchor inside the
document under judgement is not an anchor.** Ruling 19 anchors on
`CERTIFIED_ROUTE_OPTIONS` — a frozen literal outside the receipt — and held
against every attack three roles could mount. Ruling 21 anchored on a receipt
field and fell to a one-float edit.

**24. THE OBJECTIVE COLUMN IS BOUND, AND IT IS BOUND OUTSIDE THE RECEIPT.**
Ruling 20 re-derived the feasibility column from the rows and left the objective
ENDPOINT free — the scalar the latch gate reads and the one section 1's claim is
made of. A receipt whose 700 recorded iterates never fell below 1.0 sealed
`CLAIM_DISCHARGED` beside `terminal_objective: 4.48e-8`, `latched: true` and
`OBJECTIVE_TARGET_REACHED`: the latch denied by the receipt's own arithmetic by
seven decades. Three further tellings of the same number sat beside it and none
was compared to another.

The closure is a CHAIN, and it ends outside the document.
`_validate_terminal_endpoint_column` requires `solve.terminal_objective` to be
`endpoint_agreement.loop_terminal_objective` (one float through two writers),
`solve.terminal_feasibility_inf` to be the agreement's copy, and the agreement's
`standalone_terminal_objective` to be the endpoint ledger's terminal
`weighted_total` — both of which are
`float(case.standalone_evaluation(run.coordinates).weighted_total)` evaluated
twice in one process on one input, measured bitwise equal at these bytes. On the
attempt that discharges the claim `weighted_total` is a pinned quality term, so
`gate_endpoint_ledger_against_frozen_native` judges it against the campaign's
frozen native literal, and nothing in the chain is free. The agreement's
feasibility tolerance is bound to the route's own frozen tolerance and the
terminal feasibility required to be within it, which is the child's own gate
re-derived.

**The proposed row-side closure was FALSIFIED before it was gated, and would
have burned the root.** The adversarial closure — "require `latched` to imply
`min(objectives) <= NATIVE_TARGET_OBJECTIVE`" — is false of the producer:
measured on BOTH banked 5090 latches and on a live CPU solve, the engine breaks
at the TOP of its loop when the current point reaches the target, so the target
is reached at a point no row records and every recorded objective is strictly
ABOVE it (Q1 `4.529e-8`, Q2 `4.517e-8`, target `4.482e-8`). Gating it would have
refused the campaign's own banked evidence. `terminal_objective ==
rows[-1].candidate_objective` is also false — bitwise, they differ at the last
digits (measured **RELATIVE**: `1.3e-16` on CPU, `2.8e-14` and `1.7e-14` on the
two 5090 latches; round 7 re-measured the two latches at `2.7547e-14` and
`1.6993e-14` relative against the gate's `rel_tol=1e-11` — margins of 363× and
588× — with the absolute deviations `1.22e-21` and `7.61e-22` also below the
gate's `abs_tol=1e-19` floor) because the terminal point is re-evaluated through
a different kernel — and `terminal_feasibility_inf ==
rows[-1].candidate_feasibility_inf` holds on Q1 and FAILS on Q2 by `3.5e-4`.
**The unit is now stated because it decides the gate's meaning**: read as
ABSOLUTE against a `4.48e-8` objective, `2.8e-14` is `6.2e-7` relative and the
plain reading implies this gate refuses both banked latches by five decades,
which is the opposite of the measurement (numerics-physics N7-3). What is true,
and is now gated, is the REVERSE implication (no recorded iterate at or below
the target, since such an iterate would have ended the loop before it was
recorded), that a latch records at least one iterate, and ADJACENCY: the
terminal objective is one of the two endpoints the last recorded iteration
names, compared through the campaign's own cross-executable endpoint band rather
than a new constant — three hundred times the worst deviation this campaign has
measured.

**The row-side half of this ruling is CUSTODY-SCOPED, and that scope is the
claim it makes.** The reverse implication scans the `objective` column only,
while adjacency accepts either endpoint of the last recorded iteration —
`objective[-1]` **or** `candidate_objective[-1]`. That asymmetry is deliberate,
physically required, and **must not be widened**: the engine breaks at the TOP
of its loop, so the last ACCEPTED CANDIDATE is the latch point while every
recorded OPENING point is necessarily above the target. Measured on BOTH banked
5090 latches, `count(candidate_objective <= target) = 1`; extending the
reverse-implication scan to the candidate column would therefore refuse the
campaign's own banked evidence, exactly as the falsified forward form would
have. Ruling 24 accordingly closes the TERMINAL scalar — against a frozen
campaign literal that lives outside the receipt and is reached on every
discharging path — and it does **not** close the CUSTODY of the rows, which
remain unbound to an engine trace (the deferral below, unchanged). Section 1's
claim is carried by the terminal scalar and its frozen-native anchor; the
published iterate columns are corroborating provenance at exactly the strength
custody gives them, which under §12.13 is that of an honest-tooling record
(adversarial-redteam A7-1).

**25. A PRE-REGISTERED TIMEOUT IS THE FROZEN ONE.** Ruling 21 requires a record
claiming a timeout to have waited "the timeout it publishes", and took BOTH
sides of that comparison out of the document being judged:
`supervisor.attempt_timeout_seconds` was compared to nothing at all, and the
frozen `ATTEMPT_TIMEOUT_SECONDS = 3600.0` reached the validator through no path.
Roots carrying `1e-9`, `0.0`, `-1.0`, `3600` and `1e12` all sealed; three roles
found it independently and rated it MAJOR unanimously, the first unanimous
severity in six rounds. A fabricated lane timeout erases the pre-registered cold
measurement — the one thing that makes the cache an accounting device rather
than a hiding place — while keeping `PREREGISTERED` and `CLAIM_DISCHARGED`,
because ruling 13 deliberately derives conformance from whether the lane was
AUTHORIZED. The suite's own test of that branch pinned the honest timeout in its
fixture and could only ever prove the weaker property.

The ruling takes ruling 19's shape exactly: `attempt_protocol_conformance` gains
the timeout, so a run supervised under anything but the frozen literal is a
`BOUNDED_SMOKE` and cannot reach the headline verdict.
`--attempt-timeout-seconds` survives, because an operator who moves it is
running a real experiment and refusing it would be the false-reject the budget
exemption already avoids.

**26. A COUNT, AN INDEX, A BUDGET AND A SIZE ARE WHOLE NUMBERS.** The `int(...)`
deferral's reason — "a receipt claiming 700.9 certified iterations describes
nothing physical" — is true and was beside the point: the TRUNCATION is what let
it seal, and it defeated ruling 20's own new count gate in the same commit.
`int(-0.5) == 0` passed the check whose words are *"which is not a count"*;
`status: 2.9` passed *"which is not one the engine reports"* and minted
`latched: true`; `maximum_iterations: 700.9` sealed as `CERTIFIED_BUDGET` /
`PREREGISTERED`. The shape tree gains `_INTEGER` and `_INTEGER_OR_NULL` and every
count, index, budget, process id and size in bytes is declared with it, `bool`
excluded as it already was for `_NUMBER`; the budget inside the unshaped
`options` mapping is checked where that mapping is judged. Verified against the
real producers at these bytes: every one of these leaves is a Python `int` by
construction, so the float form refuses nothing an honest chain writes.

**27. AN UNBOUND CLAIM-BEARING LEAF IS UNREPRESENTABLE.** Ruling 15 made an
ABSENT SHAPE unrepresentable by deriving the required key sets from the shape
tree. Six rounds later the same defect keeps arriving one leaf over, so the same
move is made for an unbound VALUE. `LEAF_BINDINGS` declares every one of the 222
typed leaves as a frozen-literal comparison, a re-derivation, a digest
recomputation, or unbound WITH ITS REASON, and names the ANCHOR — the
module-level constant compared against, or the function that re-derives it. The
suite requires the map to be exactly the leaves the walker finds, requires every
anchor to resolve in the module, and requires `CLAIM_BEARING_LEAVES` to carry no
unbound entry. What it does not prove is that the named anchor is reached on
every path; that is ruling 28's job, and the two are meant to be read together.

**28. RULING 22 AT CHECK GRANULARITY.** Ruling 22 is true of a validator's
EXISTENCE and was false of its CONTENT: 52 of 127 refusal sites were reached by
no test, 30 of them inside kill-table validators, and seven individual checks
were deleted one at a time with the suite green each time — including the
re-hash of the published terminal state, the receipt's only re-evaluatable
artifact. `_CHECK_KILLS` pairs eighteen checks with a forgery published through
the real path and requires the refusal to arrive from THAT EXACT `raise` line,
so deleting the check makes the case red twice over. The structural half is
`_REFUSAL_SITES`, which is to refusal sites what `UNSHAPED_LEAVES` is to
unshaped blocks: the suite walks the launcher's own `raise` statements and
requires the census to be exactly what it finds, with a disposition per site.
The census records what is NOT covered — 30 sites — because a census listing
only the covered ones would be the overclaim it exists to retire. Three of its
six dispositions are CHECKED rather than asserted: `_CHECK_KILLED` against the
kill table, `_OWNER_KILLED` against `_VALIDATOR_KILLS`, and `_PRODUCER_ONLY`
against an AST call graph rooted at `validate_root_artifact`.

One site is declared UNREACHABLE with its derivation rather than given a kill
test that would have to forge the impossible: `_validate_attempt_record`'s
*"attempt carries no evidence document"* needs an outcome that is neither
`TIMEOUT` nor `PROTOCOL_FAILURE` beside evidence that is not a document, and
`_validate_attempt_outcome` runs first and derives `PROTOCOL_FAILURE` for
exactly that. It is defensive, and dead.

#### Minors closed in the same remediation

The preflight's temporary directory was checked for absoluteness only in the
PRODUCER, so a receipt declaring `relative/tmp` on all four fields sealed — a
directory the children spill through that no reader can resolve; the receipt is
now checked too, closing round-3 N18 at the level its closure missed
(adversarial A6-7d). — `visible_gpu_uuids` was declared an inventory of device
UUIDs and only membership was enforced, so integers, nulls and nested documents
published beside the pinned one; every element is required to be a string
(A6-7a). — The root's own `chain_wall` was read by nothing while every attempt
phase had just been bound to be a duration, so `-1e9` and `1e-300` sealed; the
lane and the timed attempts run sequentially inside one supervised session, so
the chain wall is required to span their sum (A6-6, P5-8's residue). — The
`objective_target` comparison ran eight lines BEFORE the field-set gate, so a
truncated options block refused with a bare `KeyError` instead of the sentence
that names the defect; the two are reordered (reproducibility E6-2). — The
stated reason for leaving the lowered-kernel SIZES unfrozen is refuted by
execution: three independent CPU processes at one commit lower the six kernels
to 65 204 569 bytes to the byte, and what moves is the COMMIT (65 207 733 one
commit earlier with the engine byte-identical, 65 200 869 on the 5090). The
decision is right for the measured reason and both docstrings now state it
(E6-3). — Two `UNSHAPED_LEAVES` reasons were false or loose at these bytes and
are corrected: `endpoint ledger.terminal` claimed "the gate is recomputed from
it" on the node that carries no gate (protocol-receipt P6-6), and `solve.rows`
called the whole solve summary a projection of the rows when two of its scalars
are measured at a point the rows do not contain (numerics N6-2). — Two refusal
sites in `_validate_leaf` read identically, which no coverage census can tell
apart; the boolean form names its own defect.

**Deferred, with reasons.** *The cold lane's empty-directory and byte-copy forms*
(adversarial A6-2, protocol-receipt P6-1's second half): ruling 25 raises the
price — a fabricated `TIMEOUT` under the frozen 3600 s must publish a supervised
wall above it — but the honest-timeout form survives, and closing it by
comparing the lane directory's CONTENTS against its own outcome collides with
ruling 17, which requires an honest timed-out lane to publish whatever it left
behind. Recorded rather than closed, because a gate that burns an honest lane is
worse than the forgery. — *No floor under the lowered IR sizes* (A6-4): a floor
is a new frozen constant about GPU compile size, the class numerics-physics
recommends against introducing between the reviews and the root, and the total
is commit-dependent by measurement. — *The supervisor's own runtime identity,
the draw's process facts, the sampler's observation, and `bound_modules` as a
superset* (A6-8, P6-4, E6-6): each is declared `BINDING_NONE` with its reason in
`LEAF_BINDINGS`, which is where the exposure is now visible to a reader instead
of implicit. — *`source_snapshot`, the sealed terminal state's dimension, the
cache configuration and directory, `interpreter_installation_modules`, the
launcher's module-scope import failure, `XLA_FLAGS` and the inherited
environment* (A5-6, A5-10a, P5-7, P6-5, E6-4, E6-5, E6-7): unchanged from
§12.11 and §12.10, and every one of them is now a declared `BINDING_NONE` entry
rather than a silence. — *Binding a published row to the terminal STATE* (the
custody half of A6-1): the only exact link runs through re-evaluating the sealed
array through the objective, which means a bootstrapped case and new heavy
compute on the re-validation path — the change class §9 makes expensive between
the reviews and the root. Ruling 24 binds the objective column to a frozen
literal instead; the rows remain unbound to an engine trace, and that is stated
rather than implied. — *The round-3, round-4 and round-5 deferrals* (numerics
N7, N13, N14, N6-1, N6-4, N6-5, the preflight's fstype self-reference, AL5b/H06,
the fixture's storage record, `--preflight-only`) are unchanged and their
reasons are unchanged. numerics-physics's recommendation AGAINST retuning
`NATIVE_ENDPOINT_REFERENCE_RELATIVE_TOLERANCE` between the reviews and the root
is carried forward for the fourth time and accepted.

**The commit's own suite claim, corrected.** The round-5 commit message stated
"qualifier 60" under the default environment and protocol-receipt measured 1
failed / 59 passed (P6-8). Root-caused rather than widened: a `multiprocessing`
**spawn** child re-imports the test module but not `tests/conftest.py`, where the
parent's FP64 is an in-process `jax.config.update`, so both racing children died
in `observe_cpu_runtime` with *"qualification requires JAX FP64"* before either
could reach the exclusion the test measures — and the suite's 60th test passed
or failed on whether the operator had exported `JAX_ENABLE_X64`. The racer now
configures the runtime the environment it is judged against declares. The
qualifier's gate is untouched: a child that is not FP64 CPU is still refused.

### 12.13 Residual adjudication — the forgery axis is out of scope, and the root is authorized

Rounds 1–7 converged. What still returns NO-GO is one axis and only one axis:
an author who hand-types receipt fields. That axis is now ruled on rather than
remediated for an eighth round.

**Adversarial-author forgery resistance is adjudicated OUT OF SCOPE for this
certification: receipts are produced by this repository's own tooling at a
pinned commit on a single operator-controlled box, and the review trail (rounds
1-7, sealed) hardened all honest-tooling failure classes to strict closure
(numerics GO x5, reproducibility GO, protocol minors-only at round 7).
Accepted-residual findings: A7-1..A7-14 class = hand-forged receipt fields;
disposition = accepted, root authorized by user 2026-08-13.**

What the adjudication does and does not license:

* It does **not** weaken any landed gate. Every validator, kill test, refusal
  site, shape declaration and leaf binding rulings 15–28 installed stays
  exactly as it is. Nothing is deleted to make this ruling true.
* It does **not** restate section 1's claim. The claim is carried by the
  TERMINAL scalar and its frozen-native anchor, which ruling 24 binds outside
  the receipt and which round 7 confirmed is reached on every discharging path.
  The row columns are corroborating provenance at custody strength, which
  §12.12's row-side paragraph now states in the ruling itself rather than
  leaving to be inferred.
* It **does** bound the threat model to an honest producer. A hand-forged
  field — A7-1's `candidate_objective` escape, A7-2's one-sided quality band,
  A7-3's null objective column, A7-4's two cold-lane forms, A7-6's unbound
  timed child, A7-12's understated residue count, A7-14's tautological
  anchors — requires an author who edits receipt bytes. That author is the
  operator, on the operator's own box, against the operator's own claim. The
  campaign's remaining defence against that person is custody and the sealed
  review trail, not another validator.
* It **does** carry forward every DEFERRED-SOUND disposition of rounds 3–7
  unchanged, including numerics-physics's standing recommendation against
  minting constants between the reviews and the root, which this commit
  honours: no numeric threshold moves here.

Two round-7 fixes land beside the adjudication because they cost nothing and
one of them removes the only reading under which ruling 24 is catastrophic:
the three adjacency deviations are now published with their unit (RELATIVE) at
both sites that state them — §12.12 above and the launcher's
`_validate_iterate_columns` docstring (N7-3) — and ruling 24's row-side closure
is restated as CUSTODY-SCOPED with the scan asymmetry marked as deliberate and
not-to-be-widened (A7-1). **The scan is not widened.** Round 7 measured
`count(candidate_objective <= target) = 1` on both banked 5090 latches, so
widening it would refuse the campaign's own evidence and burn the root; the
ledger recorded that prediction and this commit honours it.

**Root authorization.** Under §12.1 the root remains one invocation and one
invocation only. This adjudication authorizes that single spend at the full
certified configuration; it does not authorize a successor. Whatever verdict
the invocation publishes — `CLAIM_DISCHARGED`, `NO_LATCH_IN_PROTOCOL`,
`QUALITY_ONLY` or `GATE_REFUSED:<gate>` — is final, and a no-latch outcome is
reported rather than retried.

---

### 12.14 Accurate claim statement

The root ran, latched on its first attempt, and published
`verdict = CLAIM_DISCHARGED` at
`~/simsopt-campaigns/projected-route-root-20260813T184930Z/final/`
(`root-evidence.json` sha256
`6937fc68a417d6968655cbdc460fa5655bd8cb5980a6e4c735506b3008231412`). The
empirical result is sound. This section states what it does and does not
establish, because the announcement wording ran ahead of the artifact on three
axes and this is the source of truth for the corrected form.

**The claim, stated accurately.**

> **2.304x RTX 5090 engineering time-to-quality win over the frozen native bar**
> (engine compile+solve boundary, 124.707842 s vs 287.304218 s; attempt wall
> 156.856 s) **on the full VMEC-free single-stage workload, with ten endpoint
> quality gates and the whole-run feasibility bound satisfied, under the
> narrowed honest-producer pinned-checkout threat model of §12.13.**

**NOT claimed** — each item below was checked against the artifact, not assumed:

* **Not unanimous GO reviews.** All seven sealed review ledgers carry
  `overall_verdict = NO-GO`. Round 7 was two GO (numerics-physics,
  reproducibility-engineering) and two NO-GO (protocol-receipt,
  adversarial-redteam). The root is authorized because §12.13 *adjudicated*
  the surviving residual out of scope, not because the reviews converted to GO.
  The honest phrase is "seven rounds closed NO-GO with the residual
  adjudicated", never "unanimous GO" and never "reviews passed".
* **Not a provenance-equivalent native certification.** The bar is a
  hash-bound historical run whose own preserved receipt
  (`8118529751f1…`) self-describes as `authoritative: false`,
  `repository_dirty: true`, `normalized_status: budget_exhausted`. The claim is
  therefore "reaches the endpoint native reached", never "converged better than
  native" and never "native-equivalent certification".
* **Not coordinate, trajectory, or complete-termwise parity.** The contract is
  the ten pinned endpoint-quality terms of §1.1 plus the feasibility bound. The
  GPU route deliberately takes a different optimizer, trajectory and iteration
  count, and the ledger's own `relative_difference` block shows the
  informational terms (`observable.G`, `raw.residual`, `raw.major_radius`)
  diverging by orders of magnitude while every *pinned* term passes. Quoting
  agreement on the pinned ten is honest; quoting it as parity is not.
* **Not a hardware-general or version-general result.** RTX 5090, one box, one
  interpreter (CPython 3.11.15). §7's A100 lane replicates quality, not speed.

**The timing boundary, stated symmetrically.** `124.707842 s` is
`engine_compile + engine_solve`, not process wall; the attempt's full child
wall was `156.856340 s` and the supervised wall `158.741789 s`. This is not a
thumb on the scale: the 287.304218 s bar is *itself* an interior
time-to-quality figure — the timestamp of the native run's final accepted
trajectory row — so both sides exclude their own process bootstrap, which is
why §3 fixes the boundary there and every receipt carries `timing_boundary`
explicitly. The win survives the strictest possible reading, and the whole
ratio family is published rather than only the flattering one:

| Boundary | GPU (s) | ratio vs 287.304218 s bar |
|---|---|---|
| warm engine compile+solve (**certified**) | 124.707842 | **2.304x** |
| warm attempt wall (full child process) | 156.856340 | 1.832x |
| warm supervised wall | 158.741789 | 1.810x |
| cold-lane engine compile+solve | 170.693673 | 1.683x |
| cold-lane attempt wall | 226.003532 | 1.271x |

Even the cold lane measured end-to-end at process wall — compile from an empty
cache, nothing cached, nothing excluded — beats the bar. Anyone quoting 2.304x
must name the engine boundary in the same sentence; the defensible
boundary-free statement is "beats the native bar on every published boundary,
by 2.304x at the certified one and 1.271x at the most conservative one".

**Artifact size, stated correctly.** `final/` contains **608 files** (607
manifest members plus `artifact-manifest.json`). The figure 1137 is the whole
campaign directory including the JAX compilation cache and launcher logs, and
1142 after the supplement below. Quoting 1137 as the certificate's size
overstates it by ~1.9x.

#### 12.14.1 Two follow-ups, executed 2026-08-13

Both were opened as gaps by the review of the announcement, and both are now
closed rather than merely recorded. Neither touched a sealed byte: `final/`
still holds 608 files, `root-evidence.json` still digests to `6937fc68…`, and
no file under `final/` has an mtime after the seal.

**(a) Provenance supplement — the certificate is now self-contained.** Three
files the receipt names by digest but does not carry lived outside `final/`,
so a reader holding only the certificate could not check them:

| File | Pinned digest | Why it was outside |
|---|---|---|
| `src/simsopt/_version.py` | `e7da6f35…` | executed as `simsopt._version`, but git-ignored and build-generated, so the tracked-source snapshot walker never saw it — the only 1 of 297 bound modules with no snapshot member |
| `benchmarks/single_stage_native_equivalent_quality_gntr3_execution_sources.json` | `5a40391f…` | the execution-source authority (614 entries) the receipt cites by digest only |
| native endpoint state `2639a955….npy` | file `2ec9a9e3…`, content `2639a955…` | the reference side of the quality gate, pinned by absolute path into a *different* campaign directory |

They are now copied, byte-verified against those pinned digests at both read
and write, into a new sibling directory
`projected-route-root-20260813T184930Z/provenance-supplement/`, sealed 0444
files / 0555 dir, with a `supplement-manifest.json` that lists each member's
sha256, names the `root-evidence.json` digest it supplements, and states in its
own bytes that it was added post-seal at user direction. Recorded honestly
there: the executed `_version.py` carries
`1.10.7.dev1074+g320e5cba8.d20260806`, i.e. it was generated at commit
`320e5cba8` and is **stale** relative to the pinned execution commit
`b7857e6e8`; it contributes a version string and no numerics, so it does not
touch the claim, and it is published rather than smoothed over. The supplement
is additive: `final/` does not reference it and is not re-sealed.

**(b) Cross-version revalidation — now real, and now durable.** The earlier
"different jaxlib build" wording was misleading. The producer
(`.venv-qn-gpu`) and the revalidator (`.venv-qn-cpu`) both run jax/jaxlib
**0.10.0**; they differ by wheel and backend (`jax_cuda12_plugin` /
`jax_cuda12_pjrt` vs CPU-only jaxlib) but not by version — and no record of any
revalidation existed inside the artifact at all. Both defects are fixed.
`validate_root_artifact(sealed=True)` was re-run against `final/` from
throwaway `uv run --no-project` environments carrying genuinely different
versions on either side of the producer's:

| Lane | jax / jaxlib | Environment | Outcome |
|---|---|---|---|
| control | 0.10.0 | `.venv-qn-cpu` (CPU wheel) | PASS |
| cross-version, older | 0.9.2 | ephemeral uv env | PASS |
| cross-version, newer | 0.10.2 | ephemeral uv env | PASS |

All three re-derive `CLAIM_DISCHARGED` and return the **byte-identical** result
document, sha256
`93f5e072574911f98ea8b3396e108848ae1114c25efc46ceca7199c2f4372f7a`. No existing
virtual environment was installed into or modified; the prebuilt cp311
`simsoptpp` extension was reached through a symlink on `PYTHONPATH`. jax
`0.11.0` was attempted and is recorded as **not run**, with its real blocker:
it requires Python ≥ 3.12 while this artifact's execution identity pins CPython
3.11.15 and a cp311 extension. The outcome — versions, interpreter, verdict,
result digest, timestamps, and that refusal — is written durably into the
supplement as `revalidation-record.json` before sealing.

**What this converts.** "Self-contained standalone certificate" and
"cross-version revalidation" move from NOT-claimed to claimed, in the bounded
form above: the certificate is self-contained *as `final/` plus its sealed
supplement*, and the revalidation is cross-**version** across three adjacent
releases on one box and one interpreter — not a claim across all jaxlib builds
or all hardware. Everything else in the NOT-claimed list stands unchanged: the
reviews are still seven NO-GO ledgers with an adjudicated residual, the native
bar is still a non-authoritative dirty-tree budget-exhausted historical run,
and the contract is still endpoint quality, not parity.
