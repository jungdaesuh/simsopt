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
   (§12.8, adjudication 5).
10. **Sealed publication** — 0444/0555, artifact manifest written last,
    `renameat2(RENAME_NOREPLACE)`, parent fsync, sealed modes re-checked from
    the published tree.

Every external resource the protocol depends on — the NVIDIA tooling, the GPU
UUID the receipt names, and the sealed native endpoint of §12.8 adjudication 6
— is preflighted before the first child is spawned, so none of them can spend
the root at step 3.

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
* **GPU launches set `TMPDIR` outside tmpfs.** XLA spills PTX through the
  system temporary directory, and on a full `/tmp` the spill fails with
  `RESOURCE_EXHAUSTED: … Disk quota exceeded` *inside the bootstrap gate* — a
  third instance of the quota class, and the one that would spend the root.
  Measured 2026-08-13: the bounded GPU smoke published
  `GATE_REFUSED:bootstrap` for exactly this reason, and passed the whole chain
  when relaunched with `TMPDIR` pointed off tmpfs. `TMPDIR` is not one of the
  pinned environment variables (§6 step 1) because it names no property of the
  run; it is an operator precondition, checked the same way free space is.
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
and for `weighted_total`, on the latch only. `weighted_total` cannot refuse a
latch by construction: a latching attempt's objective is at or below
`NATIVE_TARGET_OBJECTIVE`, and the native endpoint re-evaluated through this
repository's objective lands 2.1e-08 relative below that literal
(cross-executable ULP, the class §5 exists for), nearly two decades inside the
1e-6 band. The absolute legs (`constraint.*`, `raw.residual`) are unchanged at
1e-10 and pass with ≥26× margin (worst measured 3.8e-12).

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
a reader of the sealed bytes can re-identify the reference the gate used.

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

**Deferred, with reasons.** The order-dependence of `max()` over a sequence
containing NaN (numerics advisory A1) stays: recorded rows are provably finite
at this tree and the terminal point is separately gated by a nonfinite-refusing
`certify_agreement`, so a guard there would be an unreachable branch. The four
tautological read-backs (`timed_against_bar`, `sha_is_binding`, `bound`,
`budget_independent`; protocol-receipt finding 5) stay: each is a tamper check
whose substantive gate exists upstream, and the artifact-tree digest already
covers the hand-edit case. Sanitizing the endpoint ledger's own term rows would
require the pinned-term gate to model null terms, changing the verdict
contract; the rows are finite at any finite iterate, and a nonfinite one stays
contained as a published `PROTOCOL_FAILURE`. Constraining `--output-root` to
`~/simsopt-campaigns/` (reproducibility finding 7) stays operator-enforced per
§11. `src/simsopt/configs/NCSX.dat` (finding 11) stays covered by the identity
gate and the sealed source snapshot rather than by the module-hash gate. The
output namespace is still claimed only at the `renameat2`; §11 constrains the
root to one supervised session, so the misleading docstring is corrected rather
than the mechanism changed (protocol-receipt advisory 7).
