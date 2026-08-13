# Projected-route certification protocol — single-stage VMEC-free, GPU beats native

Status: PHASE 1 LANDED (CPU machinery), phases 2–4 NOT STARTED.
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

Measured against native at the banked 5090 endpoints:

* `iota`, `volume`, `major_radius`, `total_length` agree to 1e-6 relative or
  better; `volume` is machine-identical, being an equality constraint.
* non-QS is slightly *better* than native (0.9% on Q1, 0.09% on Q2).
* Boozer residuals are machine-zero on both sides.
* **`G` is the one differing observable**, ~0.8–0.9% below native.

`G` is **reported, never gated**. The non-QS term is a field-scale-invariant
ratio and nothing in the shared objective pins the net poloidal current, so `G`
is a flat valley direction along which distinct equal-quality minima exist —
consistent with the terminal-to-native scaled distance of ~2.3. A `G` gate
would manufacture a false reject on a direction the objective deliberately
leaves free, which is the V260/ρ-floor failure class this campaign has now hit
three times.

Owner of the two sets: `PINNED_ENDPOINT_QUALITY_TERMS` and
`INFORMATIONAL_ENDPOINT_OBSERVABLES` in the rehearsal module. An artifact may
not restate them — validation compares the recorded sets against the module's
before reading the ledger. The ledger is *reported* at the rehearsal budget
(three attempts sit four orders of magnitude from the endpoint, so a gate there
would fail on every term and prove nothing) and *gated* at the certified
budget.

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
| An attempt latches under the bar | `CLAIM_DISCHARGED` | root spent, successfully |
| All N attempts complete, none latches | `NO_LATCH_IN_PROTOCOL` | root spent; the claim is *not* refuted, the draw failed. A successor root requires **new user authorization** and is never automatic (§12.1) |
| An attempt latches over the bar | `QUALITY_ONLY` | root spent; quality replicated, speed not |
| Any attempt fails a gate (identity, feasibility, receipt) | `GATE_REFUSED:<gate>` | root spent; this is a defect report, not a science result |

There is no undefined outcome. Roots 1–4 of the predecessor route all died in
stages whose semantics had never been written down.

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
   §1.1, gated on the pinned set at the certified budget.
9. **Sealed publication** — 0444/0555, artifact manifest written last,
   `renameat2(RENAME_NOREPLACE)`, parent fsync.
10. **Independent re-validation** of the published bytes, in-process, before
    the launcher exits.

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

## 9. Phases 2–4 (NOT STARTED)

The order below is the execution order, and it is binding: the examples script
lands **before** the root opens (§12.6).

**Phase 2 — GPU lane.** Persistent-cache preflight and the warm/cold pair of
§3; the attempt protocol of §4 wired into a supervised launcher (reuse
`run_single_stage_native_equivalent_quality_campaign.py`'s supervisor, GPU
monitor and atomic publication rather than a new launcher); artifact schema
extended from the rehearsal's with `attempts[]`, the cache evidence, the GPU
runtime identity, and the sealed source snapshot of §12.4.

**Phase 3 — examples landing.** The route lands as an `examples/jax/3_Advanced`
script reusing `build_projected_lbfgs_kernels`; the shipped mirror still uses
the old nested formulation, and the claim's wording ("the examples workload")
is not literally true until it does. Landing it here — not after the root —
puts the certified bytes and the shipped bytes in one freeze.

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

At this freeze: **612 entries** (benchmarks 115, examples 156, src 327, plus 14
non-broad qualified paths).

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
* **Sealed trees are 0555/0444.** Removal restores modes first and uses a
  Python walk; recursive-force shell globs are blocked by policy.

---

## 12. Adjudicated decisions

These were open questions when phase 1 landed. All six are now ruled on, and
the rulings are binding on the phases that follow.

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
made it a member of the manifest it writes, which is why this freeze is 612 and
not 611.

### 12.6 Examples landing order — before the root

The examples script lands in **phase 3, before the root opens** (§9). The
claim's wording ("the audited full single-stage VMEC-free examples workload")
is only literally true once the script ships, and landing it first puts the
certified bytes and the shipped bytes in one freeze instead of forcing a
post-certification refreeze that no review covered.
